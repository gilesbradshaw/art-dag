#!/usr/bin/env python3
"""
Sign assets in the registry with giles's private key.

Creates ActivityPub Create activities with RSA signatures.
"""

import base64
import hashlib
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend


def load_private_key(path: Path):
    """Load private key from PEM file."""
    pem_data = path.read_bytes()
    return serialization.load_pem_private_key(pem_data, password=None, backend=default_backend())


def sign_data(private_key, data: str) -> str:
    """Sign data with RSA private key, return base64 signature."""
    signature = private_key.sign(
        data.encode(),
        padding.PKCS1v15(),
        hashes.SHA256(),
    )
    return base64.b64encode(signature).decode()


def create_activity(actor_id: str, asset_name: str, content_hash: str, asset_type: str, domain: str = "artdag.rose-ash.com"):
    """Create a Create activity for an asset."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    return {
        "activity_id": str(uuid.uuid4()),
        "activity_type": "Create",
        "actor_id": actor_id,
        "object_data": {
            "type": asset_type_to_ap(asset_type),
            "name": asset_name,
            "id": f"https://{domain}/objects/{content_hash}",
            "contentHash": {
                "algorithm": "sha3-256",
                "value": content_hash
            },
            "attributedTo": actor_id
        },
        "published": now,
    }


def asset_type_to_ap(asset_type: str) -> str:
    """Convert asset type to ActivityPub type."""
    type_map = {
        "image": "Image",
        "video": "Video",
        "audio": "Audio",
        "effect": "Application",
        "infrastructure": "Application",
    }
    return type_map.get(asset_type, "Document")


def sign_activity(activity: dict, private_key, actor_id: str, domain: str = "artdag.rose-ash.com") -> dict:
    """Add signature to activity."""
    # Create canonical string to sign
    to_sign = json.dumps(activity["object_data"], sort_keys=True, separators=(",", ":"))

    signature_value = sign_data(private_key, to_sign)

    activity["signature"] = {
        "type": "RsaSignature2017",
        "creator": f"{actor_id}#main-key",
        "created": activity["published"],
        "signatureValue": signature_value
    }

    return activity


def main():
    username = "giles"
    domain = "artdag.rose-ash.com"
    actor_id = f"https://{domain}/users/{username}"

    # Load private key
    private_key_path = Path.home() / ".artdag" / "keys" / f"{username}.pem"
    if not private_key_path.exists():
        print(f"Private key not found: {private_key_path}")
        print("Run setup_actor.py first.")
        sys.exit(1)

    private_key = load_private_key(private_key_path)
    print(f"Loaded private key: {private_key_path}")

    # Load registry
    registry_path = Path.home() / "artdag-registry" / "registry.json"
    with open(registry_path) as f:
        registry = json.load(f)

    # Create signed activities for each asset
    activities = []

    for asset_name, asset_data in registry["assets"].items():
        print(f"\nSigning: {asset_name}")
        print(f"  Hash: {asset_data['content_hash'][:16]}...")

        activity = create_activity(
            actor_id=actor_id,
            asset_name=asset_name,
            content_hash=asset_data["content_hash"],
            asset_type=asset_data["asset_type"],
            domain=domain,
        )

        signed_activity = sign_activity(activity, private_key, actor_id, domain)
        activities.append(signed_activity)

        print(f"  Activity ID: {signed_activity['activity_id']}")
        print(f"  Signature: {signed_activity['signature']['signatureValue'][:32]}...")

    # Save activities
    activities_path = Path.home() / "artdag-registry" / "activities.json"
    activities_data = {
        "version": "1.0",
        "activities": activities
    }

    with open(activities_path, "w") as f:
        json.dump(activities_data, f, indent=2)

    print(f"\nSaved {len(activities)} signed activities to: {activities_path}")


if __name__ == "__main__":
    main()
