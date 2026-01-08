# primitive/cache.py
"""
Content-addressed file cache for node outputs.

Each node's output is stored at: cache_dir / node_id / output_file
This enables automatic reuse when the same operation is requested.
"""

import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def _file_hash(path: Path, algorithm: str = "sha3_256") -> str:
    """
    Compute content hash of a file.

    Uses SHA-3 (Keccak) by default for quantum resistance.
    """
    import hashlib
    hasher = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


@dataclass
class CacheEntry:
    """Metadata about a cached output."""
    node_id: str
    output_path: Path
    created_at: float
    size_bytes: int
    node_type: str
    content_hash: str = ""  # SHA-3 hash of file content
    execution_time: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "node_id": self.node_id,
            "output_path": str(self.output_path),
            "created_at": self.created_at,
            "size_bytes": self.size_bytes,
            "node_type": self.node_type,
            "content_hash": self.content_hash,
            "execution_time": self.execution_time,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "CacheEntry":
        return cls(
            node_id=data["node_id"],
            output_path=Path(data["output_path"]),
            created_at=data["created_at"],
            size_bytes=data["size_bytes"],
            node_type=data["node_type"],
            content_hash=data.get("content_hash", ""),
            execution_time=data.get("execution_time", 0.0),
        )


@dataclass
class CacheStats:
    """Statistics about cache usage."""
    total_entries: int = 0
    total_size_bytes: int = 0
    hits: int = 0
    misses: int = 0
    hit_rate: float = 0.0

    def record_hit(self):
        self.hits += 1
        self._update_rate()

    def record_miss(self):
        self.misses += 1
        self._update_rate()

    def _update_rate(self):
        total = self.hits + self.misses
        self.hit_rate = self.hits / total if total > 0 else 0.0


class Cache:
    """
    Content-addressed file cache.

    Structure:
        cache_dir/
            index.json           # Cache metadata
            <node_id>/
                output.ext       # Actual output file
                metadata.json    # Entry metadata
    """

    def __init__(self, cache_dir: Path | str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.stats = CacheStats()
        self._entries: Dict[str, CacheEntry] = {}
        self._load_index()

    def _index_path(self) -> Path:
        return self.cache_dir / "index.json"

    def _load_index(self):
        """Load cache index from disk."""
        index_path = self._index_path()
        if index_path.exists():
            try:
                with open(index_path) as f:
                    data = json.load(f)
                self._entries = {
                    k: CacheEntry.from_dict(v)
                    for k, v in data.get("entries", {}).items()
                }
                self.stats.total_entries = len(self._entries)
                self.stats.total_size_bytes = sum(e.size_bytes for e in self._entries.values())
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to load cache index: {e}")
                self._entries = {}

    def _save_index(self):
        """Save cache index to disk."""
        data = {
            "entries": {k: v.to_dict() for k, v in self._entries.items()},
            "stats": {
                "total_entries": self.stats.total_entries,
                "total_size_bytes": self.stats.total_size_bytes,
            },
        }
        with open(self._index_path(), "w") as f:
            json.dump(data, f, indent=2)

    def _node_dir(self, node_id: str) -> Path:
        """Get the cache directory for a node."""
        return self.cache_dir / node_id

    def get(self, node_id: str) -> Optional[Path]:
        """
        Get cached output path for a node.

        Returns the output path if cached, None otherwise.
        """
        entry = self._entries.get(node_id)
        if entry is None:
            self.stats.record_miss()
            return None

        # Verify file still exists
        if not entry.output_path.exists():
            logger.warning(f"Cache entry {node_id} missing file, removing")
            self._remove_entry(node_id)
            self.stats.record_miss()
            return None

        self.stats.record_hit()
        logger.debug(f"Cache hit: {node_id}")
        return entry.output_path

    def put(self, node_id: str, source_path: Path, node_type: str,
            execution_time: float = 0.0, move: bool = False) -> Path:
        """
        Store a file in the cache.

        Args:
            node_id: The content-addressed node ID
            source_path: Path to the file to cache
            node_type: Type of the node (for metadata)
            execution_time: How long the node took to execute
            move: If True, move the file instead of copying

        Returns:
            Path to the cached file
        """
        node_dir = self._node_dir(node_id)
        node_dir.mkdir(parents=True, exist_ok=True)

        # Preserve extension
        ext = source_path.suffix or ".out"
        output_path = node_dir / f"output{ext}"

        # Copy or move file (skip if already in place)
        source_resolved = Path(source_path).resolve()
        output_resolved = output_path.resolve()
        if source_resolved != output_resolved:
            if move:
                shutil.move(source_path, output_path)
            else:
                shutil.copy2(source_path, output_path)

        # Compute content hash
        content_hash = _file_hash(output_path)

        # Create entry
        entry = CacheEntry(
            node_id=node_id,
            output_path=output_path,
            created_at=time.time(),
            size_bytes=output_path.stat().st_size,
            node_type=node_type,
            content_hash=content_hash,
            execution_time=execution_time,
        )

        # Update index
        self._entries[node_id] = entry
        self.stats.total_entries = len(self._entries)
        self.stats.total_size_bytes = sum(e.size_bytes for e in self._entries.values())
        self._save_index()

        logger.debug(f"Cached: {node_id} ({entry.size_bytes} bytes)")
        return output_path

    def has(self, node_id: str) -> bool:
        """Check if a node is cached (without affecting stats)."""
        entry = self._entries.get(node_id)
        if entry is None:
            return False
        return entry.output_path.exists()

    def remove(self, node_id: str) -> bool:
        """Remove a node from the cache."""
        return self._remove_entry(node_id)

    def _remove_entry(self, node_id: str) -> bool:
        """Remove entry and its files."""
        if node_id not in self._entries:
            return False

        entry = self._entries.pop(node_id)
        node_dir = self._node_dir(node_id)
        if node_dir.exists():
            shutil.rmtree(node_dir)

        self.stats.total_entries = len(self._entries)
        self.stats.total_size_bytes = sum(e.size_bytes for e in self._entries.values())
        self._save_index()
        return True

    def clear(self):
        """Clear all cached entries."""
        for node_id in list(self._entries.keys()):
            self._remove_entry(node_id)
        self.stats = CacheStats()

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self.stats

    def list_entries(self) -> List[CacheEntry]:
        """List all cache entries."""
        return list(self._entries.values())

    def get_entry(self, node_id: str) -> Optional[CacheEntry]:
        """Get cache entry metadata (without affecting stats)."""
        return self._entries.get(node_id)

    def find_by_content_hash(self, content_hash: str) -> Optional[CacheEntry]:
        """Find a cache entry by its content hash."""
        for entry in self._entries.values():
            if entry.content_hash == content_hash:
                return entry
        return None

    def prune(self, max_size_bytes: int = None, max_age_seconds: float = None) -> int:
        """
        Prune cache based on size or age.

        Args:
            max_size_bytes: Remove oldest entries until under this size
            max_age_seconds: Remove entries older than this

        Returns:
            Number of entries removed
        """
        removed = 0
        now = time.time()

        # Remove by age first
        if max_age_seconds is not None:
            for node_id, entry in list(self._entries.items()):
                if now - entry.created_at > max_age_seconds:
                    self._remove_entry(node_id)
                    removed += 1

        # Then by size (remove oldest first)
        if max_size_bytes is not None and self.stats.total_size_bytes > max_size_bytes:
            sorted_entries = sorted(
                self._entries.items(),
                key=lambda x: x[1].created_at
            )
            for node_id, entry in sorted_entries:
                if self.stats.total_size_bytes <= max_size_bytes:
                    break
                self._remove_entry(node_id)
                removed += 1

        return removed

    def get_output_path(self, node_id: str, extension: str = ".mkv") -> Path:
        """Get the output path for a node (creates directory if needed)."""
        node_dir = self._node_dir(node_id)
        node_dir.mkdir(parents=True, exist_ok=True)
        return node_dir / f"output{extension}"
