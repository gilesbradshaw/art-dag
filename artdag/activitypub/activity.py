# primitive/activitypub/activity.py
"""
ActivityPub Activity types.

Activities represent actions taken by actors on objects.
Key activity types for Art DAG:
- Create: Actor creates/claims ownership of an object
- Announce: Actor shares/boosts an object
- Like: Actor endorses an object
"""

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .actor import Actor, DOMAIN


def _generate_id() -> str:
    """Generate unique activity ID."""
    return str(uuid.uuid4())


@dataclass
class Activity:
    """
    Base ActivityPub Activity.

    Attributes:
        activity_id: Unique identifier
        activity_type: Type (Create, Announce, Like, etc.)
        actor_id: ID of the actor performing the activity
        object_data: The object of the activity
        published: ISO timestamp
        signature: Cryptographic signature (added after signing)
    """
    activity_id: str
    activity_type: str
    actor_id: str
    object_data: Dict[str, Any]
    published: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    signature: Optional[Dict[str, Any]] = None

    def to_activitypub(self) -> Dict[str, Any]:
        """Return ActivityPub JSON-LD representation."""
        activity = {
            "@context": "https://www.w3.org/ns/activitystreams",
            "type": self.activity_type,
            "id": f"https://{DOMAIN}/activities/{self.activity_id}",
            "actor": self.actor_id,
            "object": self.object_data,
            "published": self.published,
        }
        if self.signature:
            activity["signature"] = self.signature
        return activity

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage."""
        return {
            "activity_id": self.activity_id,
            "activity_type": self.activity_type,
            "actor_id": self.actor_id,
            "object_data": self.object_data,
            "published": self.published,
            "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Activity":
        """Deserialize from storage."""
        return cls(
            activity_id=data["activity_id"],
            activity_type=data["activity_type"],
            actor_id=data["actor_id"],
            object_data=data["object_data"],
            published=data.get("published", ""),
            signature=data.get("signature"),
        )


@dataclass
class CreateActivity(Activity):
    """
    Create activity - establishes ownership of an object.

    Used when an actor creates or claims an asset.
    """
    activity_type: str = field(default="Create", init=False)

    @classmethod
    def for_asset(
        cls,
        actor: Actor,
        asset_name: str,
        content_hash: str,
        asset_type: str = "Image",
        metadata: Dict[str, Any] = None,
    ) -> "CreateActivity":
        """
        Create a Create activity for an asset.

        Args:
            actor: The actor claiming ownership
            asset_name: Name of the asset
            content_hash: SHA-3 hash of the asset content
            asset_type: ActivityPub object type (Image, Video, Audio, etc.)
            metadata: Additional metadata

        Returns:
            CreateActivity establishing ownership
        """
        object_data = {
            "type": asset_type,
            "name": asset_name,
            "id": f"https://{DOMAIN}/objects/{content_hash}",
            "contentHash": {
                "algorithm": "sha3-256",
                "value": content_hash,
            },
            "attributedTo": actor.id,
        }
        if metadata:
            object_data["metadata"] = metadata

        return cls(
            activity_id=_generate_id(),
            actor_id=actor.id,
            object_data=object_data,
        )


class ActivityStore:
    """
    Persistent storage for activities.

    Activities are stored as an append-only log for auditability.
    """

    def __init__(self, store_dir: Path | str):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._activities: List[Activity] = []
        self._load()

    def _log_path(self) -> Path:
        return self.store_dir / "activities.json"

    def _load(self):
        """Load activities from disk."""
        log_path = self._log_path()
        if log_path.exists():
            with open(log_path) as f:
                data = json.load(f)
            self._activities = [
                Activity.from_dict(a) for a in data.get("activities", [])
            ]

    def _save(self):
        """Save activities to disk."""
        data = {
            "version": "1.0",
            "activities": [a.to_dict() for a in self._activities],
        }
        with open(self._log_path(), "w") as f:
            json.dump(data, f, indent=2)

    def add(self, activity: Activity) -> None:
        """Add an activity to the log."""
        self._activities.append(activity)
        self._save()

    def get(self, activity_id: str) -> Optional[Activity]:
        """Get an activity by ID."""
        for a in self._activities:
            if a.activity_id == activity_id:
                return a
        return None

    def list(self) -> List[Activity]:
        """List all activities."""
        return list(self._activities)

    def find_by_actor(self, actor_id: str) -> List[Activity]:
        """Find activities by actor."""
        return [a for a in self._activities if a.actor_id == actor_id]

    def find_by_object_hash(self, content_hash: str) -> List[Activity]:
        """Find activities referencing an object by hash."""
        results = []
        for a in self._activities:
            obj_hash = a.object_data.get("contentHash", {})
            if isinstance(obj_hash, dict) and obj_hash.get("value") == content_hash:
                results.append(a)
            elif a.object_data.get("contentHash") == content_hash:
                results.append(a)
        return results

    def __len__(self) -> int:
        return len(self._activities)
