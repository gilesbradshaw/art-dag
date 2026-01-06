# tests/test_primitive_new/test_dag.py
"""Tests for primitive DAG data structures."""

import pytest
from artdag.dag import Node, NodeType, DAG, DAGBuilder


class TestNode:
    """Test Node class."""

    def test_node_creation(self):
        """Test basic node creation."""
        node = Node(node_type=NodeType.SOURCE, config={"path": "/test.mp4"})
        assert node.node_type == NodeType.SOURCE
        assert node.config == {"path": "/test.mp4"}
        assert node.node_id is not None

    def test_node_id_is_content_addressed(self):
        """Same content produces same node_id."""
        node1 = Node(node_type=NodeType.SOURCE, config={"path": "/test.mp4"})
        node2 = Node(node_type=NodeType.SOURCE, config={"path": "/test.mp4"})
        assert node1.node_id == node2.node_id

    def test_different_config_different_id(self):
        """Different config produces different node_id."""
        node1 = Node(node_type=NodeType.SOURCE, config={"path": "/test1.mp4"})
        node2 = Node(node_type=NodeType.SOURCE, config={"path": "/test2.mp4"})
        assert node1.node_id != node2.node_id

    def test_node_with_inputs(self):
        """Node with inputs includes them in ID."""
        node1 = Node(node_type=NodeType.SEGMENT, config={"duration": 5}, inputs=["abc123"])
        node2 = Node(node_type=NodeType.SEGMENT, config={"duration": 5}, inputs=["abc123"])
        node3 = Node(node_type=NodeType.SEGMENT, config={"duration": 5}, inputs=["def456"])

        assert node1.node_id == node2.node_id
        assert node1.node_id != node3.node_id

    def test_node_serialization(self):
        """Test node to_dict and from_dict."""
        original = Node(
            node_type=NodeType.SEGMENT,
            config={"duration": 5.0, "offset": 10.0},
            inputs=["abc123"],
            name="my_segment",
        )
        data = original.to_dict()
        restored = Node.from_dict(data)

        assert restored.node_type == original.node_type
        assert restored.config == original.config
        assert restored.inputs == original.inputs
        assert restored.name == original.name
        assert restored.node_id == original.node_id

    def test_custom_node_type(self):
        """Test node with custom string type."""
        node = Node(node_type="CUSTOM_TYPE", config={"custom": True})
        assert node.node_type == "CUSTOM_TYPE"
        assert node.node_id is not None


class TestDAG:
    """Test DAG class."""

    def test_dag_creation(self):
        """Test basic DAG creation."""
        dag = DAG()
        assert len(dag.nodes) == 0
        assert dag.output_id is None

    def test_add_node(self):
        """Test adding nodes to DAG."""
        dag = DAG()
        node = Node(node_type=NodeType.SOURCE, config={"path": "/test.mp4"})
        node_id = dag.add_node(node)

        assert node_id in dag.nodes
        assert dag.nodes[node_id] == node

    def test_node_deduplication(self):
        """Same node added twice returns same ID."""
        dag = DAG()
        node1 = Node(node_type=NodeType.SOURCE, config={"path": "/test.mp4"})
        node2 = Node(node_type=NodeType.SOURCE, config={"path": "/test.mp4"})

        id1 = dag.add_node(node1)
        id2 = dag.add_node(node2)

        assert id1 == id2
        assert len(dag.nodes) == 1

    def test_set_output(self):
        """Test setting output node."""
        dag = DAG()
        node = Node(node_type=NodeType.SOURCE, config={"path": "/test.mp4"})
        node_id = dag.add_node(node)
        dag.set_output(node_id)

        assert dag.output_id == node_id

    def test_set_output_invalid(self):
        """Setting invalid output raises error."""
        dag = DAG()
        with pytest.raises(ValueError):
            dag.set_output("nonexistent")

    def test_topological_order(self):
        """Test topological ordering."""
        dag = DAG()

        # Create simple chain: source -> segment -> output
        source = Node(node_type=NodeType.SOURCE, config={"path": "/test.mp4"})
        source_id = dag.add_node(source)

        segment = Node(node_type=NodeType.SEGMENT, config={"duration": 5}, inputs=[source_id])
        segment_id = dag.add_node(segment)

        dag.set_output(segment_id)
        order = dag.topological_order()

        # Source must come before segment
        assert order.index(source_id) < order.index(segment_id)

    def test_validate_valid_dag(self):
        """Test validation of valid DAG."""
        dag = DAG()
        node = Node(node_type=NodeType.SOURCE, config={"path": "/test.mp4"})
        node_id = dag.add_node(node)
        dag.set_output(node_id)

        errors = dag.validate()
        assert len(errors) == 0

    def test_validate_no_output(self):
        """DAG without output is invalid."""
        dag = DAG()
        node = Node(node_type=NodeType.SOURCE, config={"path": "/test.mp4"})
        dag.add_node(node)

        errors = dag.validate()
        assert len(errors) > 0
        assert any("output" in e.lower() for e in errors)

    def test_validate_missing_input(self):
        """DAG with missing input reference is invalid."""
        dag = DAG()
        node = Node(node_type=NodeType.SEGMENT, config={"duration": 5}, inputs=["nonexistent"])
        node_id = dag.add_node(node)
        dag.set_output(node_id)

        errors = dag.validate()
        assert len(errors) > 0
        assert any("missing" in e.lower() for e in errors)

    def test_dag_serialization(self):
        """Test DAG to_dict and from_dict."""
        dag = DAG(metadata={"name": "test_dag"})
        source = Node(node_type=NodeType.SOURCE, config={"path": "/test.mp4"})
        source_id = dag.add_node(source)
        dag.set_output(source_id)

        data = dag.to_dict()
        restored = DAG.from_dict(data)

        assert len(restored.nodes) == len(dag.nodes)
        assert restored.output_id == dag.output_id
        assert restored.metadata == dag.metadata

    def test_dag_json(self):
        """Test DAG JSON serialization."""
        dag = DAG()
        node = Node(node_type=NodeType.SOURCE, config={"path": "/test.mp4"})
        node_id = dag.add_node(node)
        dag.set_output(node_id)

        json_str = dag.to_json()
        restored = DAG.from_json(json_str)

        assert len(restored.nodes) == 1
        assert restored.output_id == node_id


class TestDAGBuilder:
    """Test DAGBuilder class."""

    def test_builder_source(self):
        """Test building source node."""
        builder = DAGBuilder()
        source_id = builder.source("/test.mp4")

        assert source_id in builder.dag.nodes
        node = builder.dag.nodes[source_id]
        assert node.node_type == NodeType.SOURCE
        assert node.config["path"] == "/test.mp4"

    def test_builder_segment(self):
        """Test building segment node."""
        builder = DAGBuilder()
        source_id = builder.source("/test.mp4")
        segment_id = builder.segment(source_id, duration=5.0, offset=10.0)

        node = builder.dag.nodes[segment_id]
        assert node.node_type == NodeType.SEGMENT
        assert node.config["duration"] == 5.0
        assert node.config["offset"] == 10.0
        assert source_id in node.inputs

    def test_builder_chain(self):
        """Test building a chain of nodes."""
        builder = DAGBuilder()
        source = builder.source("/test.mp4")
        segment = builder.segment(source, duration=5.0)
        resized = builder.resize(segment, width=1920, height=1080)
        builder.set_output(resized)

        dag = builder.build()

        assert len(dag.nodes) == 3
        assert dag.output_id == resized
        errors = dag.validate()
        assert len(errors) == 0

    def test_builder_sequence(self):
        """Test building sequence node."""
        builder = DAGBuilder()
        s1 = builder.source("/clip1.mp4")
        s2 = builder.source("/clip2.mp4")
        seq = builder.sequence([s1, s2], transition={"type": "crossfade", "duration": 0.5})
        builder.set_output(seq)

        dag = builder.build()
        node = dag.nodes[seq]
        assert node.node_type == NodeType.SEQUENCE
        assert s1 in node.inputs
        assert s2 in node.inputs

    def test_builder_mux(self):
        """Test building mux node."""
        builder = DAGBuilder()
        video = builder.source("/video.mp4")
        audio = builder.source("/audio.mp3")
        muxed = builder.mux(video, audio)
        builder.set_output(muxed)

        dag = builder.build()
        node = dag.nodes[muxed]
        assert node.node_type == NodeType.MUX
        assert video in node.inputs
        assert audio in node.inputs

    def test_builder_transform(self):
        """Test building transform node."""
        builder = DAGBuilder()
        source = builder.source("/test.mp4")
        transformed = builder.transform(source, effects={"saturation": 1.5, "contrast": 1.2})
        builder.set_output(transformed)

        dag = builder.build()
        node = dag.nodes[transformed]
        assert node.node_type == NodeType.TRANSFORM
        assert node.config["effects"]["saturation"] == 1.5

    def test_builder_validation_fails(self):
        """Builder raises error for invalid DAG."""
        builder = DAGBuilder()
        builder.source("/test.mp4")
        # No output set

        with pytest.raises(ValueError):
            builder.build()
