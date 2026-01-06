#!/usr/bin/env python3
"""
Register the identity effect owned by giles.
"""

import hashlib
from pathlib import Path
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from artdag.activitypub.ownership import OwnershipManager


def folder_hash(folder: Path) -> str:
    """
    Compute SHA3-256 hash of an entire folder.

    Hashes all files in sorted order for deterministic results.
    Each file contributes: relative_path + file_contents
    """
    hasher = hashlib.sha3_256()

    # Get all files sorted by relative path
    files = sorted(folder.rglob("*"))

    for file_path in files:
        if file_path.is_file():
            # Include relative path in hash for structure
            rel_path = file_path.relative_to(folder)
            hasher.update(str(rel_path).encode())

            # Include file contents
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    hasher.update(chunk)

    return hasher.hexdigest()


def main():
    # Use .cache as the ownership data directory
    base_dir = Path(__file__).parent.parent / ".cache" / "ownership"
    manager = OwnershipManager(base_dir)

    # Create or get giles actor
    actor = manager.get_actor("giles")
    if not actor:
        actor = manager.create_actor("giles", "Giles Bradshaw")
        print(f"Created actor: {actor.handle}")
    else:
        print(f"Using existing actor: {actor.handle}")

    # Register the identity effect folder
    effect_path = Path(__file__).parent.parent / "effects" / "identity"
    content_hash = folder_hash(effect_path)

    asset, activity = manager.register_asset(
        actor=actor,
        name="effect:identity",
        content_hash=content_hash,
        local_path=effect_path,
        tags=["effect", "primitive", "identity"],
        metadata={
            "type": "effect",
            "description": "The identity effect - returns input unchanged",
            "signature": "identity(input) → input",
        },
    )

    print(f"\nRegistered: {asset.name}")
    print(f"  Hash: {asset.content_hash}")
    print(f"  Path: {asset.local_path}")
    print(f"  Activity: {activity.activity_id}")
    print(f"  Owner: {actor.handle}")

    # Verify ownership
    verified = manager.verify_ownership(asset.name, actor)
    print(f"  Ownership verified: {verified}")

if __name__ == "__main__":
    main()
