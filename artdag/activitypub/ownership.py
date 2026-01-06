# primitive/activitypub/ownership.py
"""
Ownership integration between ActivityPub and Registry.

Connects actors, activities, and assets to establish provable ownership.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .actor import Actor, ActorStore
from .activity import Activity, CreateActivity, ActivityStore
from .signatures import sign_activity, verify_activity_ownership
from ..registry import Registry, Asset


@dataclass
class OwnershipRecord:
    """
    A verified ownership record linking actor to asset.

    Attributes:
        actor_handle: The actor's fediverse handle
        asset_name: Name of the owned asset
        content_hash: SHA-3 hash of the asset
        activity_id: ID of the Create activity establishing ownership
        verified: Whether the signature has been verified
    """
    actor_handle: str
    asset_name: str
    content_hash: str
    activity_id: str
    verified: bool = False


class OwnershipManager:
    """
    Manages ownership relationships between actors and assets.

    Integrates:
    - ActorStore: Identity management
    - Registry: Asset storage
    - ActivityStore: Ownership activities
    """

    def __init__(self, base_dir: Path | str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Initialize stores
        self.actors = ActorStore(self.base_dir / "actors")
        self.activities = ActivityStore(self.base_dir / "activities")
        self.registry = Registry(self.base_dir / "registry")

    def create_actor(self, username: str, display_name: str = None) -> Actor:
        """Create a new actor identity."""
        return self.actors.create(username, display_name)

    def get_actor(self, username: str) -> Optional[Actor]:
        """Get an actor by username."""
        return self.actors.get(username)

    def register_asset(
        self,
        actor: Actor,
        name: str,
        content_hash: str,
        url: str = None,
        local_path: Path | str = None,
        tags: List[str] = None,
        metadata: Dict[str, Any] = None,
    ) -> tuple[Asset, Activity]:
        """
        Register an asset and establish ownership.

        Creates the asset in the registry and a signed Create activity
        proving the actor's ownership.

        Args:
            actor: The actor claiming ownership
            name: Name for the asset
            content_hash: SHA-3-256 hash of the content
            url: Public URL (canonical location)
            local_path: Optional local path
            tags: Optional tags
            metadata: Optional metadata

        Returns:
            Tuple of (Asset, signed CreateActivity)
        """
        # Add to registry
        asset = self.registry.add(
            name=name,
            content_hash=content_hash,
            url=url,
            local_path=local_path,
            tags=tags,
            metadata=metadata,
        )

        # Create ownership activity
        activity = CreateActivity.for_asset(
            actor=actor,
            asset_name=name,
            content_hash=asset.content_hash,
            asset_type=self._asset_type_to_ap(asset.asset_type),
            metadata=metadata,
        )

        # Sign the activity
        signed_activity = sign_activity(activity, actor)

        # Store the activity
        self.activities.add(signed_activity)

        return asset, signed_activity

    def _asset_type_to_ap(self, asset_type: str) -> str:
        """Convert registry asset type to ActivityPub type."""
        type_map = {
            "image": "Image",
            "video": "Video",
            "audio": "Audio",
            "unknown": "Document",
        }
        return type_map.get(asset_type, "Document")

    def get_owner(self, asset_name: str) -> Optional[Actor]:
        """
        Get the owner of an asset.

        Finds the earliest Create activity for the asset and returns
        the actor if the signature is valid.
        """
        asset = self.registry.get(asset_name)
        if not asset:
            return None

        # Find Create activities for this asset
        activities = self.activities.find_by_object_hash(asset.content_hash)
        create_activities = [a for a in activities if a.activity_type == "Create"]

        if not create_activities:
            return None

        # Get the earliest (first owner)
        earliest = min(create_activities, key=lambda a: a.published)

        # Extract username from actor_id
        # Format: https://artdag.rose-ash.com/users/{username}
        actor_id = earliest.actor_id
        if "/users/" in actor_id:
            username = actor_id.split("/users/")[-1]
            actor = self.actors.get(username)
            if actor and verify_activity_ownership(earliest, actor):
                return actor

        return None

    def verify_ownership(self, asset_name: str, actor: Actor) -> bool:
        """
        Verify that an actor owns an asset.

        Checks for a valid signed Create activity linking the actor
        to the asset.
        """
        asset = self.registry.get(asset_name)
        if not asset:
            return False

        activities = self.activities.find_by_object_hash(asset.content_hash)
        for activity in activities:
            if activity.activity_type == "Create" and activity.actor_id == actor.id:
                if verify_activity_ownership(activity, actor):
                    return True

        return False

    def list_owned_assets(self, actor: Actor) -> List[Asset]:
        """List all assets owned by an actor."""
        activities = self.activities.find_by_actor(actor.id)
        owned = []

        for activity in activities:
            if activity.activity_type == "Create":
                # Find asset by hash
                obj_hash = activity.object_data.get("contentHash", {})
                if isinstance(obj_hash, dict):
                    hash_value = obj_hash.get("value")
                else:
                    hash_value = obj_hash

                if hash_value:
                    asset = self.registry.find_by_hash(hash_value)
                    if asset:
                        owned.append(asset)

        return owned

    def get_ownership_records(self) -> List[OwnershipRecord]:
        """Get all ownership records."""
        records = []

        for activity in self.activities.list():
            if activity.activity_type != "Create":
                continue

            # Extract info
            actor_id = activity.actor_id
            username = actor_id.split("/users/")[-1] if "/users/" in actor_id else "unknown"
            actor = self.actors.get(username)

            obj_hash = activity.object_data.get("contentHash", {})
            hash_value = obj_hash.get("value") if isinstance(obj_hash, dict) else obj_hash

            records.append(OwnershipRecord(
                actor_handle=actor.handle if actor else f"@{username}@unknown",
                asset_name=activity.object_data.get("name", "unknown"),
                content_hash=hash_value or "unknown",
                activity_id=activity.activity_id,
                verified=verify_activity_ownership(activity, actor) if actor else False,
            ))

        return records
