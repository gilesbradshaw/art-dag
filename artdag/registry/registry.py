# primitive/registry/registry.py
"""
Asset registry for the Art DAG.

The registry stores named assets with metadata, enabling:
- Named references to source files
- Tagging and categorization
- Content-addressed deduplication
- Asset discovery and search
"""

import hashlib
import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


def _file_hash(path: Path, algorithm: str = "sha3_256") -> str:
    """
    Compute content hash of a file.

    Uses SHA-3 (Keccak) by default for quantum resistance.
    SHA-3-256 provides 128-bit security against quantum attacks (Grover's algorithm).

    Args:
        path: File to hash
        algorithm: Hash algorithm (sha3_256, sha3_512, sha256, blake2b)

    Returns:
        Full hex digest (no truncation)
    """
    hasher = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


@dataclass
class Asset:
    """
    A registered asset in the Art DAG.

    Attributes:
        name: Unique name for the asset
        path: Path to the source file
        content_hash: SHA256 hash of file content (first 16 chars)
        asset_type: Type of asset (image, video, audio, etc.)
        tags: List of tags for categorization
        metadata: Additional metadata (dimensions, duration, etc.)
        created_at: Timestamp when added to registry
    """
    name: str
    path: Path
    content_hash: str
    asset_type: str = "unknown"
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "path": str(self.path),
            "content_hash": self.content_hash,
            "asset_type": self.asset_type,
            "tags": self.tags,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Asset":
        return cls(
            name=data["name"],
            path=Path(data["path"]),
            content_hash=data["content_hash"],
            asset_type=data.get("asset_type", "unknown"),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", time.time()),
        )


class Registry:
    """
    The Art DAG registry.

    Stores named assets that can be referenced by DAGs.

    Structure:
        registry_dir/
            registry.json     # Index of all assets
            assets/           # Optional: copied asset files
                <hash>/
                    <filename>
    """

    def __init__(self, registry_dir: Path | str, copy_assets: bool = False):
        """
        Initialize the registry.

        Args:
            registry_dir: Directory to store registry data
            copy_assets: If True, copy assets into registry (content-addressed)
        """
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.copy_assets = copy_assets
        self._assets: Dict[str, Asset] = {}
        self._load()

    def _index_path(self) -> Path:
        return self.registry_dir / "registry.json"

    def _assets_dir(self) -> Path:
        return self.registry_dir / "assets"

    def _load(self):
        """Load registry from disk."""
        index_path = self._index_path()
        if index_path.exists():
            with open(index_path) as f:
                data = json.load(f)
            self._assets = {
                name: Asset.from_dict(asset_data)
                for name, asset_data in data.get("assets", {}).items()
            }

    def _save(self):
        """Save registry to disk."""
        data = {
            "version": "1.0",
            "assets": {name: asset.to_dict() for name, asset in self._assets.items()},
        }
        with open(self._index_path(), "w") as f:
            json.dump(data, f, indent=2)

    def add(
        self,
        name: str,
        path: Path | str,
        asset_type: str = None,
        tags: List[str] = None,
        metadata: Dict[str, Any] = None,
    ) -> Asset:
        """
        Add an asset to the registry.

        Args:
            name: Unique name for the asset
            path: Path to the source file
            asset_type: Type of asset (auto-detected if not provided)
            tags: List of tags for categorization
            metadata: Additional metadata

        Returns:
            The created Asset
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Asset file not found: {path}")

        # Compute content hash
        content_hash = _file_hash(path)

        # Auto-detect asset type from extension
        if asset_type is None:
            ext = path.suffix.lower()
            type_map = {
                ".jpg": "image", ".jpeg": "image", ".png": "image",
                ".gif": "image", ".webp": "image", ".bmp": "image",
                ".mp4": "video", ".mkv": "video", ".avi": "video",
                ".mov": "video", ".webm": "video",
                ".mp3": "audio", ".wav": "audio", ".flac": "audio",
                ".ogg": "audio", ".aac": "audio",
            }
            asset_type = type_map.get(ext, "unknown")

        # Copy asset if configured
        if self.copy_assets:
            assets_dir = self._assets_dir()
            asset_dir = assets_dir / content_hash
            asset_dir.mkdir(parents=True, exist_ok=True)
            dest_path = asset_dir / path.name
            if not dest_path.exists():
                shutil.copy2(path, dest_path)
            path = dest_path

        asset = Asset(
            name=name,
            path=path.resolve(),
            content_hash=content_hash,
            asset_type=asset_type,
            tags=tags or [],
            metadata=metadata or {},
        )

        self._assets[name] = asset
        self._save()
        return asset

    def get(self, name: str) -> Optional[Asset]:
        """Get an asset by name."""
        return self._assets.get(name)

    def remove(self, name: str) -> bool:
        """Remove an asset from the registry."""
        if name not in self._assets:
            return False
        del self._assets[name]
        self._save()
        return True

    def list(self) -> List[Asset]:
        """List all assets."""
        return list(self._assets.values())

    def find_by_tag(self, tag: str) -> List[Asset]:
        """Find assets with a specific tag."""
        return [a for a in self._assets.values() if tag in a.tags]

    def find_by_type(self, asset_type: str) -> List[Asset]:
        """Find assets of a specific type."""
        return [a for a in self._assets.values() if a.asset_type == asset_type]

    def find_by_hash(self, content_hash: str) -> Optional[Asset]:
        """Find an asset by content hash."""
        for asset in self._assets.values():
            if asset.content_hash == content_hash:
                return asset
        return None

    def __contains__(self, name: str) -> bool:
        return name in self._assets

    def __len__(self) -> int:
        return len(self._assets)

    def __iter__(self):
        return iter(self._assets.values())
