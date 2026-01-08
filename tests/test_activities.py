# tests/test_activities.py
"""Tests for the activity tracking and cache deletion system."""

import tempfile
import time
from pathlib import Path

import pytest

from artdag import Cache, DAG, Node, NodeType
from artdag.activities import Activity, ActivityStore, ActivityManager, make_is_shared_fn


class MockActivityPubStore:
    """Mock ActivityPub store for testing is_shared functionality."""

    def __init__(self):
        self._shared_hashes = set()

    def mark_shared(self, content_hash: str):
        """Mark a content hash as shared (published)."""
        self._shared_hashes.add(content_hash)

    def find_by_object_hash(self, content_hash: str):
        """Return mock activities for shared hashes."""
        if content_hash in self._shared_hashes:
            return [MockActivity("Create")]
        return []


class MockActivity:
    """Mock ActivityPub activity."""
    def __init__(self, activity_type: str):
        self.activity_type = activity_type


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def cache(temp_dir):
    """Create a cache instance."""
    return Cache(temp_dir / "cache")


@pytest.fixture
def activity_store(temp_dir):
    """Create an activity store instance."""
    return ActivityStore(temp_dir / "activities")


@pytest.fixture
def ap_store():
    """Create a mock ActivityPub store."""
    return MockActivityPubStore()


@pytest.fixture
def manager(cache, activity_store, ap_store):
    """Create an ActivityManager instance."""
    return ActivityManager(
        cache=cache,
        activity_store=activity_store,
        is_shared_fn=make_is_shared_fn(ap_store),
    )


def create_test_file(path: Path, content: str = "test content") -> Path:
    """Create a test file with content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


class TestCacheEntryContentHash:
    """Tests for content_hash in CacheEntry."""

    def test_put_computes_content_hash(self, cache, temp_dir):
        """put() should compute and store content_hash."""
        test_file = create_test_file(temp_dir / "input.txt", "hello world")

        cache.put("node1", test_file, "test")
        entry = cache.get_entry("node1")

        assert entry is not None
        assert entry.content_hash != ""
        assert len(entry.content_hash) == 64  # SHA-3-256 hex

    def test_same_content_same_hash(self, cache, temp_dir):
        """Same file content should produce same hash."""
        file1 = create_test_file(temp_dir / "file1.txt", "identical content")
        file2 = create_test_file(temp_dir / "file2.txt", "identical content")

        cache.put("node1", file1, "test")
        cache.put("node2", file2, "test")

        entry1 = cache.get_entry("node1")
        entry2 = cache.get_entry("node2")

        assert entry1.content_hash == entry2.content_hash

    def test_different_content_different_hash(self, cache, temp_dir):
        """Different file content should produce different hash."""
        file1 = create_test_file(temp_dir / "file1.txt", "content A")
        file2 = create_test_file(temp_dir / "file2.txt", "content B")

        cache.put("node1", file1, "test")
        cache.put("node2", file2, "test")

        entry1 = cache.get_entry("node1")
        entry2 = cache.get_entry("node2")

        assert entry1.content_hash != entry2.content_hash

    def test_find_by_content_hash(self, cache, temp_dir):
        """Should find entry by content hash."""
        test_file = create_test_file(temp_dir / "input.txt", "unique content")
        cache.put("node1", test_file, "test")

        entry = cache.get_entry("node1")
        found = cache.find_by_content_hash(entry.content_hash)

        assert found is not None
        assert found.node_id == "node1"

    def test_content_hash_persists(self, temp_dir):
        """content_hash should persist across cache reloads."""
        cache1 = Cache(temp_dir / "cache")
        test_file = create_test_file(temp_dir / "input.txt", "persistent")
        cache1.put("node1", test_file, "test")
        original_hash = cache1.get_entry("node1").content_hash

        # Create new cache instance (reload from disk)
        cache2 = Cache(temp_dir / "cache")
        entry = cache2.get_entry("node1")

        assert entry.content_hash == original_hash


class TestActivity:
    """Tests for Activity dataclass."""

    def test_activity_from_dag(self):
        """Activity.from_dag() should classify nodes correctly."""
        # Build a simple DAG: source -> transform -> output
        dag = DAG()
        source = Node(NodeType.SOURCE, {"path": "/test.mp4"})
        transform = Node(NodeType.TRANSFORM, {"effect": "blur"}, inputs=[source.node_id])
        output = Node(NodeType.RESIZE, {"width": 100}, inputs=[transform.node_id])

        dag.add_node(source)
        dag.add_node(transform)
        dag.add_node(output)
        dag.set_output(output.node_id)

        activity = Activity.from_dag(dag)

        assert source.node_id in activity.input_ids
        assert activity.output_id == output.node_id
        assert transform.node_id in activity.intermediate_ids

    def test_activity_with_multiple_inputs(self):
        """Activity should handle DAGs with multiple source nodes."""
        dag = DAG()
        source1 = Node(NodeType.SOURCE, {"path": "/a.mp4"})
        source2 = Node(NodeType.SOURCE, {"path": "/b.mp4"})
        sequence = Node(NodeType.SEQUENCE, {}, inputs=[source1.node_id, source2.node_id])

        dag.add_node(source1)
        dag.add_node(source2)
        dag.add_node(sequence)
        dag.set_output(sequence.node_id)

        activity = Activity.from_dag(dag)

        assert len(activity.input_ids) == 2
        assert source1.node_id in activity.input_ids
        assert source2.node_id in activity.input_ids
        assert activity.output_id == sequence.node_id
        assert len(activity.intermediate_ids) == 0

    def test_activity_serialization(self):
        """Activity should serialize and deserialize correctly."""
        dag = DAG()
        source = Node(NodeType.SOURCE, {"path": "/test.mp4"})
        dag.add_node(source)
        dag.set_output(source.node_id)

        activity = Activity.from_dag(dag)
        data = activity.to_dict()
        restored = Activity.from_dict(data)

        assert restored.activity_id == activity.activity_id
        assert restored.input_ids == activity.input_ids
        assert restored.output_id == activity.output_id
        assert restored.intermediate_ids == activity.intermediate_ids

    def test_all_node_ids(self):
        """all_node_ids should return all nodes."""
        activity = Activity(
            activity_id="test",
            input_ids=["a", "b"],
            output_id="c",
            intermediate_ids=["d", "e"],
            created_at=time.time(),
        )

        all_ids = activity.all_node_ids
        assert set(all_ids) == {"a", "b", "c", "d", "e"}


class TestActivityStore:
    """Tests for ActivityStore persistence."""

    def test_add_and_get(self, activity_store):
        """Should add and retrieve activities."""
        activity = Activity(
            activity_id="test1",
            input_ids=["input1"],
            output_id="output1",
            intermediate_ids=["inter1"],
            created_at=time.time(),
        )

        activity_store.add(activity)
        retrieved = activity_store.get("test1")

        assert retrieved is not None
        assert retrieved.activity_id == "test1"

    def test_persistence(self, temp_dir):
        """Activities should persist across store reloads."""
        store1 = ActivityStore(temp_dir / "activities")
        activity = Activity(
            activity_id="persist",
            input_ids=["i1"],
            output_id="o1",
            intermediate_ids=[],
            created_at=time.time(),
        )
        store1.add(activity)

        # Reload
        store2 = ActivityStore(temp_dir / "activities")
        retrieved = store2.get("persist")

        assert retrieved is not None
        assert retrieved.activity_id == "persist"

    def test_find_by_input_ids(self, activity_store):
        """Should find activities with matching inputs."""
        activity1 = Activity(
            activity_id="a1",
            input_ids=["x", "y"],
            output_id="o1",
            intermediate_ids=[],
            created_at=time.time(),
        )
        activity2 = Activity(
            activity_id="a2",
            input_ids=["y", "x"],  # Same inputs, different order
            output_id="o2",
            intermediate_ids=[],
            created_at=time.time(),
        )
        activity3 = Activity(
            activity_id="a3",
            input_ids=["z"],  # Different inputs
            output_id="o3",
            intermediate_ids=[],
            created_at=time.time(),
        )

        activity_store.add(activity1)
        activity_store.add(activity2)
        activity_store.add(activity3)

        found = activity_store.find_by_input_ids(["x", "y"])
        assert len(found) == 2
        assert {a.activity_id for a in found} == {"a1", "a2"}

    def test_find_using_node(self, activity_store):
        """Should find activities referencing a node."""
        activity = Activity(
            activity_id="a1",
            input_ids=["input1"],
            output_id="output1",
            intermediate_ids=["inter1"],
            created_at=time.time(),
        )
        activity_store.add(activity)

        # Should find by input
        found = activity_store.find_using_node("input1")
        assert len(found) == 1

        # Should find by intermediate
        found = activity_store.find_using_node("inter1")
        assert len(found) == 1

        # Should find by output
        found = activity_store.find_using_node("output1")
        assert len(found) == 1

        # Should not find unknown
        found = activity_store.find_using_node("unknown")
        assert len(found) == 0

    def test_remove(self, activity_store):
        """Should remove activities."""
        activity = Activity(
            activity_id="to_remove",
            input_ids=["i"],
            output_id="o",
            intermediate_ids=[],
            created_at=time.time(),
        )
        activity_store.add(activity)
        assert activity_store.get("to_remove") is not None

        result = activity_store.remove("to_remove")
        assert result is True
        assert activity_store.get("to_remove") is None


class TestActivityManager:
    """Tests for ActivityManager deletion rules."""

    def test_can_delete_orphaned_entry(self, manager, cache, temp_dir):
        """Orphaned entries (not in any activity) can be deleted."""
        test_file = create_test_file(temp_dir / "orphan.txt", "orphan")
        cache.put("orphan_node", test_file, "test")

        assert manager.can_delete_cache_entry("orphan_node") is True

    def test_cannot_delete_shared_entry(self, manager, cache, temp_dir, ap_store):
        """Shared entries (ActivityPub published) cannot be deleted."""
        test_file = create_test_file(temp_dir / "shared.txt", "shared content")
        cache.put("shared_node", test_file, "test")

        # Mark as shared
        entry = cache.get_entry("shared_node")
        ap_store.mark_shared(entry.content_hash)

        assert manager.can_delete_cache_entry("shared_node") is False

    def test_cannot_delete_activity_input(self, manager, cache, activity_store, temp_dir):
        """Activity inputs cannot be deleted."""
        test_file = create_test_file(temp_dir / "input.txt", "input")
        cache.put("input_node", test_file, "test")

        activity = Activity(
            activity_id="a1",
            input_ids=["input_node"],
            output_id="output_node",
            intermediate_ids=[],
            created_at=time.time(),
        )
        activity_store.add(activity)

        assert manager.can_delete_cache_entry("input_node") is False

    def test_cannot_delete_activity_output(self, manager, cache, activity_store, temp_dir):
        """Activity outputs cannot be deleted."""
        test_file = create_test_file(temp_dir / "output.txt", "output")
        cache.put("output_node", test_file, "test")

        activity = Activity(
            activity_id="a1",
            input_ids=["input_node"],
            output_id="output_node",
            intermediate_ids=[],
            created_at=time.time(),
        )
        activity_store.add(activity)

        assert manager.can_delete_cache_entry("output_node") is False

    def test_can_delete_intermediate(self, manager, cache, activity_store, temp_dir):
        """Intermediate entries can be deleted (they're reconstructible)."""
        test_file = create_test_file(temp_dir / "inter.txt", "intermediate")
        cache.put("inter_node", test_file, "test")

        activity = Activity(
            activity_id="a1",
            input_ids=["input_node"],
            output_id="output_node",
            intermediate_ids=["inter_node"],
            created_at=time.time(),
        )
        activity_store.add(activity)

        assert manager.can_delete_cache_entry("inter_node") is True

    def test_can_discard_activity_no_shared(self, manager, activity_store):
        """Activity can be discarded if nothing is shared."""
        activity = Activity(
            activity_id="a1",
            input_ids=["i1"],
            output_id="o1",
            intermediate_ids=["m1"],
            created_at=time.time(),
        )
        activity_store.add(activity)

        assert manager.can_discard_activity("a1") is True

    def test_cannot_discard_activity_with_shared_output(self, manager, cache, activity_store, temp_dir, ap_store):
        """Activity cannot be discarded if output is shared."""
        test_file = create_test_file(temp_dir / "output.txt", "output content")
        cache.put("o1", test_file, "test")

        activity = Activity(
            activity_id="a1",
            input_ids=["i1"],
            output_id="o1",
            intermediate_ids=[],
            created_at=time.time(),
        )
        activity_store.add(activity)

        # Mark output as shared
        entry = cache.get_entry("o1")
        ap_store.mark_shared(entry.content_hash)

        assert manager.can_discard_activity("a1") is False

    def test_cannot_discard_activity_with_shared_input(self, manager, cache, activity_store, temp_dir, ap_store):
        """Activity cannot be discarded if input is shared."""
        test_file = create_test_file(temp_dir / "input.txt", "input content")
        cache.put("i1", test_file, "test")

        activity = Activity(
            activity_id="a1",
            input_ids=["i1"],
            output_id="o1",
            intermediate_ids=[],
            created_at=time.time(),
        )
        activity_store.add(activity)

        entry = cache.get_entry("i1")
        ap_store.mark_shared(entry.content_hash)

        assert manager.can_discard_activity("a1") is False

    def test_discard_activity_deletes_intermediates(self, manager, cache, activity_store, temp_dir):
        """Discarding activity should delete intermediate cache entries."""
        # Create cache entries
        input_file = create_test_file(temp_dir / "input.txt", "input")
        inter_file = create_test_file(temp_dir / "inter.txt", "intermediate")
        output_file = create_test_file(temp_dir / "output.txt", "output")

        cache.put("i1", input_file, "test")
        cache.put("m1", inter_file, "test")
        cache.put("o1", output_file, "test")

        activity = Activity(
            activity_id="a1",
            input_ids=["i1"],
            output_id="o1",
            intermediate_ids=["m1"],
            created_at=time.time(),
        )
        activity_store.add(activity)

        # Discard
        result = manager.discard_activity("a1")

        assert result is True
        assert cache.has("m1") is False  # Intermediate deleted
        assert activity_store.get("a1") is None  # Activity removed

    def test_discard_activity_deletes_orphaned_output(self, manager, cache, activity_store, temp_dir):
        """Discarding activity should delete output if orphaned."""
        output_file = create_test_file(temp_dir / "output.txt", "output")
        cache.put("o1", output_file, "test")

        activity = Activity(
            activity_id="a1",
            input_ids=[],
            output_id="o1",
            intermediate_ids=[],
            created_at=time.time(),
        )
        activity_store.add(activity)

        manager.discard_activity("a1")

        assert cache.has("o1") is False  # Orphaned output deleted

    def test_discard_activity_keeps_shared_output(self, manager, cache, activity_store, temp_dir, ap_store):
        """Discarding should fail if output is shared."""
        output_file = create_test_file(temp_dir / "output.txt", "shared output")
        cache.put("o1", output_file, "test")

        activity = Activity(
            activity_id="a1",
            input_ids=[],
            output_id="o1",
            intermediate_ids=[],
            created_at=time.time(),
        )
        activity_store.add(activity)

        entry = cache.get_entry("o1")
        ap_store.mark_shared(entry.content_hash)

        result = manager.discard_activity("a1")

        assert result is False  # Cannot discard
        assert cache.has("o1") is True  # Output preserved
        assert activity_store.get("a1") is not None  # Activity preserved

    def test_discard_keeps_input_used_elsewhere(self, manager, cache, activity_store, temp_dir):
        """Input used by another activity should not be deleted."""
        input_file = create_test_file(temp_dir / "input.txt", "shared input")
        cache.put("shared_input", input_file, "test")

        activity1 = Activity(
            activity_id="a1",
            input_ids=["shared_input"],
            output_id="o1",
            intermediate_ids=[],
            created_at=time.time(),
        )
        activity2 = Activity(
            activity_id="a2",
            input_ids=["shared_input"],
            output_id="o2",
            intermediate_ids=[],
            created_at=time.time(),
        )
        activity_store.add(activity1)
        activity_store.add(activity2)

        manager.discard_activity("a1")

        # Input still used by a2
        assert cache.has("shared_input") is True

    def test_get_deletable_entries(self, manager, cache, activity_store, temp_dir):
        """Should list all deletable entries."""
        # Orphan (deletable)
        orphan = create_test_file(temp_dir / "orphan.txt", "orphan")
        cache.put("orphan", orphan, "test")

        # Intermediate (deletable)
        inter = create_test_file(temp_dir / "inter.txt", "inter")
        cache.put("inter", inter, "test")

        # Input (not deletable)
        inp = create_test_file(temp_dir / "input.txt", "input")
        cache.put("input", inp, "test")

        activity = Activity(
            activity_id="a1",
            input_ids=["input"],
            output_id="output",
            intermediate_ids=["inter"],
            created_at=time.time(),
        )
        activity_store.add(activity)

        deletable = manager.get_deletable_entries()
        deletable_ids = {e.node_id for e in deletable}

        assert "orphan" in deletable_ids
        assert "inter" in deletable_ids
        assert "input" not in deletable_ids

    def test_cleanup_intermediates(self, manager, cache, activity_store, temp_dir):
        """cleanup_intermediates() should delete all intermediate entries."""
        inter1 = create_test_file(temp_dir / "i1.txt", "inter1")
        inter2 = create_test_file(temp_dir / "i2.txt", "inter2")
        cache.put("inter1", inter1, "test")
        cache.put("inter2", inter2, "test")

        activity = Activity(
            activity_id="a1",
            input_ids=["input"],
            output_id="output",
            intermediate_ids=["inter1", "inter2"],
            created_at=time.time(),
        )
        activity_store.add(activity)

        deleted = manager.cleanup_intermediates()

        assert deleted == 2
        assert cache.has("inter1") is False
        assert cache.has("inter2") is False


class TestMakeIsSharedFn:
    """Tests for make_is_shared_fn factory."""

    def test_returns_true_for_shared(self, ap_store):
        """Should return True for shared content."""
        is_shared = make_is_shared_fn(ap_store)
        ap_store.mark_shared("hash123")

        assert is_shared("hash123") is True

    def test_returns_false_for_not_shared(self, ap_store):
        """Should return False for non-shared content."""
        is_shared = make_is_shared_fn(ap_store)

        assert is_shared("unknown_hash") is False
