# primitive/activitypub/__init__.py
"""
ActivityPub implementation for Art DAG.

Provides decentralized identity and ownership for assets.
Domain: artdag.rose-ash.com

Core concepts:
- Actor: A user identity with cryptographic keys
- Object: An asset (image, video, etc.)
- Activity: An action (Create, Announce, Like, etc.)
- Signature: Cryptographic proof of authorship
"""

from .actor import Actor, ActorStore
from .activity import Activity, CreateActivity, ActivityStore
from .signatures import sign_activity, verify_signature, verify_activity_ownership
from .ownership import OwnershipManager, OwnershipRecord

__all__ = [
    "Actor",
    "ActorStore",
    "Activity",
    "CreateActivity",
    "ActivityStore",
    "sign_activity",
    "verify_signature",
    "verify_activity_ownership",
    "OwnershipManager",
    "OwnershipRecord",
]

DOMAIN = "artdag.rose-ash.com"
