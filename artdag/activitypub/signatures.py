# primitive/activitypub/signatures.py
"""
Cryptographic signatures for ActivityPub.

Uses RSA-SHA256 signatures compatible with HTTP Signatures spec
and Linked Data Signatures for ActivityPub.
"""

import base64
import hashlib
import json
import time
from typing import Any, Dict

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.exceptions import InvalidSignature

from .actor import Actor
from .activity import Activity


def _canonicalize(data: Dict[str, Any]) -> str:
    """
    Canonicalize JSON for signing.

    Uses JCS (JSON Canonicalization Scheme) - sorted keys, no whitespace.
    """
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def _hash_sha256(data: str) -> bytes:
    """Hash string with SHA-256."""
    return hashlib.sha256(data.encode()).digest()


def sign_activity(activity: Activity, actor: Actor) -> Activity:
    """
    Sign an activity with the actor's private key.

    Uses Linked Data Signatures with RsaSignature2017.

    Args:
        activity: The activity to sign
        actor: The actor whose key signs the activity

    Returns:
        Activity with signature attached
    """
    # Load private key
    private_key = serialization.load_pem_private_key(
        actor.private_key,
        password=None,
    )

    # Create signature options
    created = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # Canonicalize the activity (without signature)
    activity_data = activity.to_activitypub()
    activity_data.pop("signature", None)
    canonical = _canonicalize(activity_data)

    # Create the data to sign: hash of options + hash of document
    options = {
        "@context": "https://w3id.org/security/v1",
        "type": "RsaSignature2017",
        "creator": actor.key_id,
        "created": created,
    }
    options_hash = _hash_sha256(_canonicalize(options))
    document_hash = _hash_sha256(canonical)
    to_sign = options_hash + document_hash

    # Sign with RSA-SHA256
    signature_bytes = private_key.sign(
        to_sign,
        padding.PKCS1v15(),
        hashes.SHA256(),
    )
    signature_value = base64.b64encode(signature_bytes).decode("utf-8")

    # Attach signature to activity
    activity.signature = {
        "type": "RsaSignature2017",
        "creator": actor.key_id,
        "created": created,
        "signatureValue": signature_value,
    }

    return activity


def verify_signature(activity: Activity, public_key_pem: bytes) -> bool:
    """
    Verify an activity's signature.

    Args:
        activity: The activity with signature
        public_key_pem: PEM-encoded public key

    Returns:
        True if signature is valid
    """
    if not activity.signature:
        return False

    try:
        # Load public key
        public_key = serialization.load_pem_public_key(public_key_pem)

        # Reconstruct signature options
        options = {
            "@context": "https://w3id.org/security/v1",
            "type": activity.signature["type"],
            "creator": activity.signature["creator"],
            "created": activity.signature["created"],
        }

        # Canonicalize activity without signature
        activity_data = activity.to_activitypub()
        activity_data.pop("signature", None)
        canonical = _canonicalize(activity_data)

        # Recreate signed data
        options_hash = _hash_sha256(_canonicalize(options))
        document_hash = _hash_sha256(canonical)
        signed_data = options_hash + document_hash

        # Decode and verify signature
        signature_bytes = base64.b64decode(activity.signature["signatureValue"])
        public_key.verify(
            signature_bytes,
            signed_data,
            padding.PKCS1v15(),
            hashes.SHA256(),
        )
        return True

    except (InvalidSignature, KeyError, ValueError):
        return False


def verify_activity_ownership(activity: Activity, actor: Actor) -> bool:
    """
    Verify that an activity was signed by the claimed actor.

    Args:
        activity: The activity to verify
        actor: The claimed actor

    Returns:
        True if the activity was signed by this actor
    """
    if not activity.signature:
        return False

    # Check creator matches actor
    if activity.signature.get("creator") != actor.key_id:
        return False

    # Verify signature
    return verify_signature(activity, actor.public_key)
