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
    cid: str = ""  # Content identifier (IPFS CID or local hash)
    execution_time: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "node_id": self.node_id,
            "output_path": str(self.output_path),
            "created_at": self.created_at,
            "size_bytes": self.size_bytes,
            "node_type": self.node_type,
            "cid": self.cid,
            "execution_time": self.execution_time,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "CacheEntry":
        # Support both "cid" and legacy "content_hash"
        cid = data.get("cid") or data.get("content_hash", "")
        return cls(
            node_id=data["node_id"],
            output_path=Path(data["output_path"]),
            created_at=data["created_at"],
            size_bytes=data["size_bytes"],
            node_type=data["node_type"],
            cid=cid,
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
    Code-addressed file cache.

    The filesystem IS the index - no JSON index files needed.
    Each node's hash is its directory name.

    Structure:
        cache_dir/
            <hash>/
                output.ext       # Actual output file
                metadata.json    # Per-node metadata (optional)
    """

    def __init__(self, cache_dir: Path | str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.stats = CacheStats()

    def _node_dir(self, node_id: str) -> Path:
        """Get the cache directory for a node."""
        return self.cache_dir / node_id

    def _find_output_file(self, node_dir: Path) -> Optional[Path]:
        """Find the output file in a node directory."""
        if not node_dir.exists() or not node_dir.is_dir():
            return None
        for f in node_dir.iterdir():
            if f.is_file() and f.name.startswith("output."):
                return f
        return None

    def get(self, node_id: str) -> Optional[Path]:
        """
        Get cached output path for a node.

        Checks filesystem directly - no in-memory index.
        Returns the output path if cached, None otherwise.
        """
        node_dir = self._node_dir(node_id)
        output_file = self._find_output_file(node_dir)

        if output_file:
            self.stats.record_hit()
            logger.debug(f"Cache hit: {node_id[:16]}...")
            return output_file

        self.stats.record_miss()
        return None

    def put(self, node_id: str, source_path: Path, node_type: str,
            execution_time: float = 0.0, move: bool = False) -> Path:
        """
        Store a file in the cache.

        Args:
            node_id: The code-addressed node ID (hash)
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

        # Compute content hash (IPFS CID of the result)
        cid = _file_hash(output_path)

        # Store per-node metadata (optional, for stats/debugging)
        metadata = {
            "node_id": node_id,
            "output_path": str(output_path),
            "created_at": time.time(),
            "size_bytes": output_path.stat().st_size,
            "node_type": node_type,
            "cid": cid,
            "execution_time": execution_time,
        }
        metadata_path = node_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.debug(f"Cached: {node_id[:16]}... ({metadata['size_bytes']} bytes)")
        return output_path

    def has(self, node_id: str) -> bool:
        """Check if a node is cached (without affecting stats)."""
        return self._find_output_file(self._node_dir(node_id)) is not None

    def remove(self, node_id: str) -> bool:
        """Remove a node from the cache."""
        node_dir = self._node_dir(node_id)
        if node_dir.exists():
            shutil.rmtree(node_dir)
            return True
        return False

    def clear(self):
        """Clear all cached entries."""
        for node_dir in self.cache_dir.iterdir():
            if node_dir.is_dir() and not node_dir.name.startswith("_"):
                shutil.rmtree(node_dir)
        self.stats = CacheStats()

    def get_stats(self) -> CacheStats:
        """Get cache statistics (scans filesystem)."""
        stats = CacheStats()
        for node_dir in self.cache_dir.iterdir():
            if node_dir.is_dir() and not node_dir.name.startswith("_"):
                output_file = self._find_output_file(node_dir)
                if output_file:
                    stats.total_entries += 1
                    stats.total_size_bytes += output_file.stat().st_size
        stats.hits = self.stats.hits
        stats.misses = self.stats.misses
        stats.hit_rate = self.stats.hit_rate
        return stats

    def list_entries(self) -> List[CacheEntry]:
        """List all cache entries (scans filesystem)."""
        entries = []
        for node_dir in self.cache_dir.iterdir():
            if node_dir.is_dir() and not node_dir.name.startswith("_"):
                entry = self._load_entry_from_disk(node_dir.name)
                if entry:
                    entries.append(entry)
        return entries

    def _load_entry_from_disk(self, node_id: str) -> Optional[CacheEntry]:
        """Load entry metadata from disk."""
        node_dir = self._node_dir(node_id)
        metadata_path = node_dir / "metadata.json"
        output_file = self._find_output_file(node_dir)

        if not output_file:
            return None

        if metadata_path.exists():
            try:
                with open(metadata_path) as f:
                    data = json.load(f)
                return CacheEntry.from_dict(data)
            except (json.JSONDecodeError, KeyError):
                pass

        # Fallback: create entry from filesystem
        return CacheEntry(
            node_id=node_id,
            output_path=output_file,
            created_at=output_file.stat().st_mtime,
            size_bytes=output_file.stat().st_size,
            node_type="unknown",
            cid=_file_hash(output_file),
        )

    def get_entry(self, node_id: str) -> Optional[CacheEntry]:
        """Get cache entry metadata (without affecting stats)."""
        return self._load_entry_from_disk(node_id)

    def find_by_cid(self, cid: str) -> Optional[CacheEntry]:
        """Find a cache entry by its content hash (scans filesystem)."""
        for entry in self.list_entries():
            if entry.cid == cid:
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
        entries = self.list_entries()

        # Remove by age first
        if max_age_seconds is not None:
            for entry in entries:
                if now - entry.created_at > max_age_seconds:
                    self.remove(entry.node_id)
                    removed += 1

        # Then by size (remove oldest first)
        if max_size_bytes is not None:
            stats = self.get_stats()
            if stats.total_size_bytes > max_size_bytes:
                sorted_entries = sorted(entries, key=lambda e: e.created_at)
                total_size = stats.total_size_bytes
                for entry in sorted_entries:
                    if total_size <= max_size_bytes:
                        break
                    self.remove(entry.node_id)
                    total_size -= entry.size_bytes
                    removed += 1

        return removed

    def get_output_path(self, node_id: str, extension: str = ".mkv") -> Path:
        """Get the output path for a node (creates directory if needed)."""
        node_dir = self._node_dir(node_id)
        node_dir.mkdir(parents=True, exist_ok=True)
        return node_dir / f"output{extension}"

    # Effect storage methods

    def _effects_dir(self) -> Path:
        """Get the effects subdirectory."""
        effects_dir = self.cache_dir / "_effects"
        effects_dir.mkdir(parents=True, exist_ok=True)
        return effects_dir

    def store_effect(self, source: str) -> str:
        """
        Store an effect in the cache.

        Args:
            source: Effect source code

        Returns:
            Content hash (cache ID) of the effect
        """
        import hashlib as _hashlib

        # Compute content hash
        cid = _hashlib.sha3_256(source.encode("utf-8")).hexdigest()

        # Try to load full metadata if effects module available
        try:
            from .effects.loader import load_effect
            loaded = load_effect(source)
            meta_dict = loaded.meta.to_dict()
            dependencies = loaded.dependencies
            requires_python = loaded.requires_python
        except ImportError:
            # Fallback: store without parsed metadata
            meta_dict = {}
            dependencies = []
            requires_python = ">=3.10"

        effect_dir = self._effects_dir() / cid
        effect_dir.mkdir(parents=True, exist_ok=True)

        # Store source
        source_path = effect_dir / "effect.py"
        source_path.write_text(source, encoding="utf-8")

        # Store metadata
        metadata = {
            "cid": cid,
            "meta": meta_dict,
            "dependencies": dependencies,
            "requires_python": requires_python,
            "stored_at": time.time(),
        }
        metadata_path = effect_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Stored effect '{loaded.meta.name}' with hash {cid[:16]}...")
        return cid

    def get_effect(self, cid: str) -> Optional[str]:
        """
        Get effect source by content hash.

        Args:
            cid: SHA3-256 hash of effect source

        Returns:
            Effect source code if found, None otherwise
        """
        effect_dir = self._effects_dir() / cid
        source_path = effect_dir / "effect.py"

        if not source_path.exists():
            return None

        return source_path.read_text(encoding="utf-8")

    def get_effect_path(self, cid: str) -> Optional[Path]:
        """
        Get path to effect source file.

        Args:
            cid: SHA3-256 hash of effect source

        Returns:
            Path to effect.py if found, None otherwise
        """
        effect_dir = self._effects_dir() / cid
        source_path = effect_dir / "effect.py"

        if not source_path.exists():
            return None

        return source_path

    def get_effect_metadata(self, cid: str) -> Optional[dict]:
        """
        Get effect metadata by content hash.

        Args:
            cid: SHA3-256 hash of effect source

        Returns:
            Metadata dict if found, None otherwise
        """
        effect_dir = self._effects_dir() / cid
        metadata_path = effect_dir / "metadata.json"

        if not metadata_path.exists():
            return None

        try:
            with open(metadata_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, KeyError):
            return None

    def has_effect(self, cid: str) -> bool:
        """Check if an effect is cached."""
        effect_dir = self._effects_dir() / cid
        return (effect_dir / "effect.py").exists()

    def list_effects(self) -> List[dict]:
        """List all cached effects with their metadata."""
        effects = []
        effects_dir = self._effects_dir()

        if not effects_dir.exists():
            return effects

        for effect_dir in effects_dir.iterdir():
            if effect_dir.is_dir():
                metadata = self.get_effect_metadata(effect_dir.name)
                if metadata:
                    effects.append(metadata)

        return effects

    def remove_effect(self, cid: str) -> bool:
        """Remove an effect from the cache."""
        effect_dir = self._effects_dir() / cid

        if not effect_dir.exists():
            return False

        shutil.rmtree(effect_dir)
        logger.info(f"Removed effect {cid[:16]}...")
        return True
