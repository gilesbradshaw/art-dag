#!/usr/bin/env python3
"""
Set up actor with keypair stored securely.

Private key: ~/.artdag/keys/{username}.pem
Public key: exported for registry
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add artdag to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend


def create_keypair():
    """Generate RSA-2048 keypair."""
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend(),
    )
    return private_key


def save_private_key(private_key, path: Path):
    """Save private key to PEM file."""
    pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pem)
    os.chmod(path, 0o600)  # Owner read/write only
    return pem.decode()


def get_public_key_pem(private_key) -> str:
    """Extract public key as PEM string."""
    public_key = private_key.public_key()
    pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return pem.decode()


def create_actor_json(username: str, display_name: str, public_key_pem: str, domain: str = "artdag.rose-ash.com"):
    """Create ActivityPub actor JSON."""
    return {
        "@context": [
            "https://www.w3.org/ns/activitystreams",
            "https://w3id.org/security/v1"
        ],
        "type": "Person",
        "id": f"https://{domain}/users/{username}",
        "preferredUsername": username,
        "name": display_name,
        "inbox": f"https://{domain}/users/{username}/inbox",
        "outbox": f"https://{domain}/users/{username}/outbox",
        "publicKey": {
            "id": f"https://{domain}/users/{username}#main-key",
            "owner": f"https://{domain}/users/{username}",
            "publicKeyPem": public_key_pem
        }
    }


def main():
    username = "giles"
    display_name = "Giles Bradshaw"
    domain = "artdag.rose-ash.com"

    keys_dir = Path.home() / ".artdag" / "keys"
    private_key_path = keys_dir / f"{username}.pem"

    # Check if key already exists
    if private_key_path.exists():
        print(f"Private key already exists: {private_key_path}")
        print("Delete it first if you want to regenerate.")
        sys.exit(1)

    # Create new keypair
    print(f"Creating new keypair for @{username}@{domain}...")
    private_key = create_keypair()

    # Save private key
    save_private_key(private_key, private_key_path)
    print(f"Private key saved: {private_key_path}")
    print(f"  Mode: 600 (owner read/write only)")
    print(f"  BACK THIS UP!")

    # Get public key
    public_key_pem = get_public_key_pem(private_key)

    # Create actor JSON
    actor = create_actor_json(username, display_name, public_key_pem, domain)

    # Output actor JSON
    actor_json = json.dumps(actor, indent=2)
    print(f"\nActor JSON (for registry/actors/{username}.json):")
    print(actor_json)

    # Save to registry
    registry_path = Path.home() / "artdag-registry" / "actors" / f"{username}.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(actor_json)
    print(f"\nSaved to: {registry_path}")


if __name__ == "__main__":
    main()
