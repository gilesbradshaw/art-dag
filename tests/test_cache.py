# tests/test_primitive_new/test_cache.py
"""Tests for primitive cache module."""

import pytest
import tempfile
from pathlib import Path

from artdag.cache import Cache, CacheStats


@pytest.fixture
def cache_dir():
    """Create temporary cache directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def cache(cache_dir):
    """Create cache instance."""
    return Cache(cache_dir)


@pytest.fixture
def sample_file(cache_dir):
    """Create a sample file to cache."""
    file_path = cache_dir / "sample.txt"
    file_path.write_text("test content")
    return file_path


class TestCache:
    """Test Cache class."""

    def test_cache_creation(self, cache_dir):
        """Test cache directory is created."""
        cache = Cache(cache_dir / "new_cache")
        assert cache.cache_dir.exists()

    def test_cache_put_and_get(self, cache, sample_file):
        """Test putting and getting from cache."""
        node_id = "abc123"
        cached_path = cache.put(node_id, sample_file, "TEST")

        assert cached_path.exists()
        assert cache.has(node_id)

        retrieved = cache.get(node_id)
        assert retrieved == cached_path

    def test_cache_miss(self, cache):
        """Test cache miss returns None."""
        result = cache.get("nonexistent")
        assert result is None

    def test_cache_stats_hit_miss(self, cache, sample_file):
        """Test cache hit/miss stats."""
        cache.put("abc123", sample_file, "TEST")

        # Miss
        cache.get("nonexistent")
        assert cache.stats.misses == 1

        # Hit
        cache.get("abc123")
        assert cache.stats.hits == 1

        assert cache.stats.hit_rate == 0.5

    def test_cache_remove(self, cache, sample_file):
        """Test removing from cache."""
        node_id = "abc123"
        cache.put(node_id, sample_file, "TEST")
        assert cache.has(node_id)

        cache.remove(node_id)
        assert not cache.has(node_id)

    def test_cache_clear(self, cache, sample_file):
        """Test clearing cache."""
        cache.put("node1", sample_file, "TEST")
        cache.put("node2", sample_file, "TEST")

        assert cache.stats.total_entries == 2

        cache.clear()

        assert cache.stats.total_entries == 0
        assert not cache.has("node1")
        assert not cache.has("node2")

    def test_cache_preserves_extension(self, cache, cache_dir):
        """Test that cache preserves file extension."""
        mp4_file = cache_dir / "video.mp4"
        mp4_file.write_text("fake video")

        cached = cache.put("video_node", mp4_file, "SOURCE")
        assert cached.suffix == ".mp4"

    def test_cache_list_entries(self, cache, sample_file):
        """Test listing cache entries."""
        cache.put("node1", sample_file, "TYPE1")
        cache.put("node2", sample_file, "TYPE2")

        entries = cache.list_entries()
        assert len(entries) == 2

        node_ids = {e.node_id for e in entries}
        assert "node1" in node_ids
        assert "node2" in node_ids

    def test_cache_persistence(self, cache_dir, sample_file):
        """Test cache persists across instances."""
        # First instance
        cache1 = Cache(cache_dir)
        cache1.put("abc123", sample_file, "TEST")

        # Second instance loads from disk
        cache2 = Cache(cache_dir)
        assert cache2.has("abc123")

    def test_cache_prune_by_age(self, cache, sample_file):
        """Test pruning by age."""
        import time

        cache.put("old_node", sample_file, "TEST")

        # Manually set old creation time
        entry = cache._entries["old_node"]
        entry.created_at = time.time() - 3600  # 1 hour ago

        removed = cache.prune(max_age_seconds=1800)  # 30 minutes

        assert removed == 1
        assert not cache.has("old_node")

    def test_cache_output_path(self, cache):
        """Test getting output path for node."""
        path = cache.get_output_path("abc123", ".mp4")
        assert path.suffix == ".mp4"
        assert "abc123" in str(path)
        assert path.parent.exists()


class TestCacheStats:
    """Test CacheStats class."""

    def test_hit_rate_calculation(self):
        """Test hit rate calculation."""
        stats = CacheStats()

        stats.record_hit()
        stats.record_hit()
        stats.record_miss()

        assert stats.hits == 2
        assert stats.misses == 1
        assert abs(stats.hit_rate - 0.666) < 0.01

    def test_initial_hit_rate(self):
        """Test hit rate with no requests."""
        stats = CacheStats()
        assert stats.hit_rate == 0.0
