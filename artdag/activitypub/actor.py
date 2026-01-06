# primitive/activitypub/actor.py
"""
ActivityPub Actor management.

An Actor is an identity with:
- Username and display name
- RSA key pair for signing
- ActivityPub-compliant JSON-LD representation
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding

DOMAIN = "artdag.rose-ash.com"


def _generate_keypair() -> tuple[bytes, bytes]:
    """Generate RSA key pair for signing."""
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return private_pem, public_pem


@dataclass
class Actor:
    """
    An ActivityPub Actor (identity).

    Attributes:
        username: Unique username (e.g., "giles")
        display_name: Human-readable name
        public_key: PEM-encoded public key
        private_key: PEM-encoded private key (kept secret)
        created_at: Timestamp of creation
    """
    username: str
    display_name: str
    public_key: bytes
    private_key: bytes
    created_at: float = field(default_factory=time.time)
    domain: str = DOMAIN

    @property
    def id(self) -> str:
        """ActivityPub actor ID (URL)."""
        return f"https://{self.domain}/users/{self.username}"

    @property
    def handle(self) -> str:
        """Fediverse handle."""
        return f"@{self.username}@{self.domain}"

    @property
    def inbox(self) -> str:
        """ActivityPub inbox URL."""
        return f"{self.id}/inbox"

    @property
    def outbox(self) -> str:
        """ActivityPub outbox URL."""
        return f"{self.id}/outbox"

    @property
    def key_id(self) -> str:
        """Key ID for HTTP Signatures."""
        return f"{self.id}#main-key"

    def to_activitypub(self) -> Dict[str, Any]:
        """Return ActivityPub JSON-LD representation."""
        return {
            "@context": [
                "https://www.w3.org/ns/activitystreams",
                "https://w3id.org/security/v1",
            ],
            "type": "Person",
            "id": self.id,
            "preferredUsername": self.username,
            "name": self.display_name,
            "inbox": self.inbox,
            "outbox": self.outbox,
            "publicKey": {
                "id": self.key_id,
                "owner": self.id,
                "publicKeyPem": self.public_key.decode("utf-8"),
            },
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage."""
        return {
            "username": self.username,
            "display_name": self.display_name,
            "public_key": self.public_key.decode("utf-8"),
            "private_key": self.private_key.decode("utf-8"),
            "created_at": self.created_at,
            "domain": self.domain,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Actor":
        """Deserialize from storage."""
        return cls(
            username=data["username"],
            display_name=data["display_name"],
            public_key=data["public_key"].encode("utf-8"),
            private_key=data["private_key"].encode("utf-8"),
            created_at=data.get("created_at", time.time()),
            domain=data.get("domain", DOMAIN),
        )

    @classmethod
    def create(cls, username: str, display_name: str = None) -> "Actor":
        """Create a new actor with generated keys."""
        private_pem, public_pem = _generate_keypair()
        return cls(
            username=username,
            display_name=display_name or username,
            public_key=public_pem,
            private_key=private_pem,
        )


class ActorStore:
    """
    Persistent storage for actors.

    Structure:
        store_dir/
            actors.json       # Index of all actors
            keys/
                <username>.private.pem
                <username>.public.pem
    """

    def __init__(self, store_dir: Path | str):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._actors: Dict[str, Actor] = {}
        self._load()

    def _index_path(self) -> Path:
        return self.store_dir / "actors.json"

    def _load(self):
        """Load actors from disk."""
        index_path = self._index_path()
        if index_path.exists():
            with open(index_path) as f:
                data = json.load(f)
            self._actors = {
                username: Actor.from_dict(actor_data)
                for username, actor_data in data.get("actors", {}).items()
            }

    def _save(self):
        """Save actors to disk."""
        data = {
            "version": "1.0",
            "domain": DOMAIN,
            "actors": {
                username: actor.to_dict()
                for username, actor in self._actors.items()
            },
        }
        with open(self._index_path(), "w") as f:
            json.dump(data, f, indent=2)

    def create(self, username: str, display_name: str = None) -> Actor:
        """Create and store a new actor."""
        if username in self._actors:
            raise ValueError(f"Actor {username} already exists")

        actor = Actor.create(username, display_name)
        self._actors[username] = actor
        self._save()
        return actor

    def get(self, username: str) -> Optional[Actor]:
        """Get an actor by username."""
        return self._actors.get(username)

    def list(self) -> list[Actor]:
        """List all actors."""
        return list(self._actors.values())

    def __contains__(self, username: str) -> bool:
        return username in self._actors

    def __len__(self) -> int:
        return len(self._actors)
