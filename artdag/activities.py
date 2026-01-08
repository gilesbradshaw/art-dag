# artdag/activities.py
"""
Persistent activity (job) tracking for cache management.

Activities represent executions of DAGs. They track:
- Input node IDs (sources)
- Output node ID (terminal node)
- Intermediate node IDs (everything in between)

This enables deletion rules:
- Shared items (ActivityPub published) cannot be deleted
- Inputs/outputs of activities cannot be deleted
- Intermediates can be deleted (reconstructible)
- Activities can only be discarded if no items are shared
"""

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from .cache import Cache, CacheEntry
from .dag import DAG

logger = logging.getLogger(__name__)


def make_is_shared_fn(activitypub_store: "ActivityStore") -> Callable[[str], bool]:
    """
    Create an is_shared function from an ActivityPub ActivityStore.

    Args:
        activitypub_store: The ActivityPub activity store
            (from artdag.activitypub.activity)

    Returns:
        Function that checks if a content_hash has been published
    """
    def is_shared(content_hash: str) -> bool:
        activities = activitypub_store.find_by_object_hash(content_hash)
        return any(a.activity_type == "Create" for a in activities)
    return is_shared


@dataclass
class Activity:
    """
    A recorded execution of a DAG.

    Tracks which cache entries are inputs, outputs, and intermediates
    to enforce deletion rules.
    """
    activity_id: str
    input_ids: List[str]        # Source node cache IDs
    output_id: str              # Terminal node cache ID
    intermediate_ids: List[str] # Everything in between
    created_at: float
    status: str = "completed"   # pending|running|completed|failed
    dag_snapshot: Optional[Dict[str, Any]] = None  # Serialized DAG for reconstruction

    def to_dict(self) -> Dict[str, Any]:
        return {
            "activity_id": self.activity_id,
            "input_ids": self.input_ids,
            "output_id": self.output_id,
            "intermediate_ids": self.intermediate_ids,
            "created_at": self.created_at,
            "status": self.status,
            "dag_snapshot": self.dag_snapshot,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Activity":
        return cls(
            activity_id=data["activity_id"],
            input_ids=data["input_ids"],
            output_id=data["output_id"],
            intermediate_ids=data["intermediate_ids"],
            created_at=data["created_at"],
            status=data.get("status", "completed"),
            dag_snapshot=data.get("dag_snapshot"),
        )

    @classmethod
    def from_dag(cls, dag: DAG, activity_id: str = None) -> "Activity":
        """
        Create an Activity from a DAG.

        Classifies nodes as inputs, output, or intermediates.
        """
        if activity_id is None:
            activity_id = str(uuid.uuid4())

        # Find input nodes (nodes with no inputs - sources)
        input_ids = []
        for node_id, node in dag.nodes.items():
            if not node.inputs:
                input_ids.append(node_id)

        # Output is the terminal node
        output_id = dag.output_id

        # Intermediates are everything else
        intermediate_ids = []
        for node_id in dag.nodes:
            if node_id not in input_ids and node_id != output_id:
                intermediate_ids.append(node_id)

        return cls(
            activity_id=activity_id,
            input_ids=sorted(input_ids),
            output_id=output_id,
            intermediate_ids=sorted(intermediate_ids),
            created_at=time.time(),
            status="completed",
            dag_snapshot=dag.to_dict(),
        )

    @property
    def all_node_ids(self) -> List[str]:
        """All node IDs involved in this activity."""
        return self.input_ids + [self.output_id] + self.intermediate_ids


class ActivityStore:
    """
    Persistent storage for activities.

    Provides methods to check deletion eligibility and perform deletions.
    """

    def __init__(self, store_dir: Path | str):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._activities: Dict[str, Activity] = {}
        self._load()

    def _index_path(self) -> Path:
        return self.store_dir / "activities.json"

    def _load(self):
        """Load activities from disk."""
        index_path = self._index_path()
        if index_path.exists():
            try:
                with open(index_path) as f:
                    data = json.load(f)
                self._activities = {
                    a["activity_id"]: Activity.from_dict(a)
                    for a in data.get("activities", [])
                }
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to load activities: {e}")
                self._activities = {}

    def _save(self):
        """Save activities to disk."""
        data = {
            "version": "1.0",
            "activities": [a.to_dict() for a in self._activities.values()],
        }
        with open(self._index_path(), "w") as f:
            json.dump(data, f, indent=2)

    def add(self, activity: Activity) -> None:
        """Add an activity."""
        self._activities[activity.activity_id] = activity
        self._save()

    def get(self, activity_id: str) -> Optional[Activity]:
        """Get an activity by ID."""
        return self._activities.get(activity_id)

    def remove(self, activity_id: str) -> bool:
        """Remove an activity record (does not delete cache entries)."""
        if activity_id not in self._activities:
            return False
        del self._activities[activity_id]
        self._save()
        return True

    def list(self) -> List[Activity]:
        """List all activities."""
        return list(self._activities.values())

    def find_by_input_ids(self, input_ids: List[str]) -> List[Activity]:
        """Find activities with the same inputs (for UI grouping)."""
        sorted_inputs = sorted(input_ids)
        return [
            a for a in self._activities.values()
            if sorted(a.input_ids) == sorted_inputs
        ]

    def find_using_node(self, node_id: str) -> List[Activity]:
        """Find all activities that reference a node ID."""
        return [
            a for a in self._activities.values()
            if node_id in a.all_node_ids
        ]

    def __len__(self) -> int:
        return len(self._activities)


class ActivityManager:
    """
    Manages activities and cache deletion with sharing rules.

    Deletion rules:
    1. Shared items (ActivityPub published) cannot be deleted
    2. Inputs/outputs of activities cannot be deleted
    3. Intermediates can be deleted (reconstructible)
    4. Activities can only be discarded if no items are shared
    """

    def __init__(
        self,
        cache: Cache,
        activity_store: ActivityStore,
        is_shared_fn: Callable[[str], bool],
    ):
        """
        Args:
            cache: The L1 cache
            activity_store: Activity persistence
            is_shared_fn: Function that checks if a content_hash is shared
                          (published via ActivityPub)
        """
        self.cache = cache
        self.activities = activity_store
        self._is_shared = is_shared_fn

    def record_activity(self, dag: DAG) -> Activity:
        """Record a completed DAG execution as an activity."""
        activity = Activity.from_dag(dag)
        self.activities.add(activity)
        return activity

    def is_shared(self, node_id: str) -> bool:
        """Check if a cache entry is shared (published via ActivityPub)."""
        entry = self.cache.get_entry(node_id)
        if not entry or not entry.content_hash:
            return False
        return self._is_shared(entry.content_hash)

    def can_delete_cache_entry(self, node_id: str) -> bool:
        """
        Check if a cache entry can be deleted.

        Returns False if:
        - Entry is shared (ActivityPub published)
        - Entry is an input or output of any activity
        """
        # Check if shared
        if self.is_shared(node_id):
            return False

        # Check if it's an input or output of any activity
        for activity in self.activities.list():
            if node_id in activity.input_ids:
                return False
            if node_id == activity.output_id:
                return False

        # It's either an intermediate or orphaned - can delete
        return True

    def can_discard_activity(self, activity_id: str) -> bool:
        """
        Check if an activity can be discarded.

        Returns False if any cache entry (input, output, or intermediate)
        is shared via ActivityPub.
        """
        activity = self.activities.get(activity_id)
        if not activity:
            return False

        # Check if any item is shared
        for node_id in activity.all_node_ids:
            if self.is_shared(node_id):
                return False

        return True

    def discard_activity(self, activity_id: str) -> bool:
        """
        Discard an activity and delete its intermediate cache entries.

        Returns False if the activity cannot be discarded (has shared items).

        When discarded:
        - Intermediate cache entries are deleted
        - The activity record is removed
        - Inputs remain (may be used by other activities)
        - Output is deleted if orphaned (not shared, not used elsewhere)
        """
        if not self.can_discard_activity(activity_id):
            return False

        activity = self.activities.get(activity_id)
        if not activity:
            return False

        output_id = activity.output_id
        intermediate_ids = list(activity.intermediate_ids)

        # Remove the activity record first
        self.activities.remove(activity_id)

        # Delete intermediates
        for node_id in intermediate_ids:
            self.cache.remove(node_id)
            logger.debug(f"Deleted intermediate: {node_id}")

        # Check if output is now orphaned
        if self._is_orphaned(output_id) and not self.is_shared(output_id):
            self.cache.remove(output_id)
            logger.debug(f"Deleted orphaned output: {output_id}")

        # Inputs remain - they may be used by other activities
        # But check if any are orphaned now
        for input_id in activity.input_ids:
            if self._is_orphaned(input_id) and not self.is_shared(input_id):
                self.cache.remove(input_id)
                logger.debug(f"Deleted orphaned input: {input_id}")

        return True

    def _is_orphaned(self, node_id: str) -> bool:
        """Check if a node is not referenced by any activity."""
        for activity in self.activities.list():
            if node_id in activity.all_node_ids:
                return False
        return True

    def get_deletable_entries(self) -> List[CacheEntry]:
        """Get all cache entries that can be deleted."""
        deletable = []
        for entry in self.cache.list_entries():
            if self.can_delete_cache_entry(entry.node_id):
                deletable.append(entry)
        return deletable

    def get_discardable_activities(self) -> List[Activity]:
        """Get all activities that can be discarded."""
        return [
            a for a in self.activities.list()
            if self.can_discard_activity(a.activity_id)
        ]

    def cleanup_intermediates(self) -> int:
        """
        Delete all intermediate cache entries.

        Intermediates are safe to delete as they can be reconstructed
        from inputs using the DAG.

        Returns:
            Number of entries deleted
        """
        deleted = 0
        for activity in self.activities.list():
            for node_id in activity.intermediate_ids:
                if self.cache.has(node_id):
                    self.cache.remove(node_id)
                    deleted += 1
        return deleted
