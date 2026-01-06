# primitive/dag.py
"""
Core DAG data structures.

Nodes are content-addressed: node_id = hash(type + config + input_ids)
This enables automatic caching and deduplication.
"""

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional


class NodeType(Enum):
    """Built-in node types."""
    # Source operations
    SOURCE = auto()      # Load file from path

    # Transform operations
    SEGMENT = auto()     # Extract time range
    RESIZE = auto()      # Scale/crop/pad
    TRANSFORM = auto()   # Visual effects (color, blur, etc.)

    # Compose operations
    SEQUENCE = auto()    # Concatenate in time
    LAYER = auto()       # Stack spatially (overlay)
    MUX = auto()         # Combine video + audio streams
    BLEND = auto()       # Blend two inputs
    SWITCH = auto()      # Time-based input switching

    # Analysis operations
    ANALYZE = auto()     # Extract features (audio, motion, etc.)

    # Generation operations
    GENERATE = auto()    # Create content (text, graphics, etc.)


def _stable_hash(data: Any, algorithm: str = "sha3_256") -> str:
    """
    Create stable hash from arbitrary data.

    Uses SHA-3 (Keccak) for quantum resistance.
    Returns full hash - no truncation.

    Args:
        data: Data to hash (will be JSON serialized)
        algorithm: Hash algorithm (default: sha3_256)

    Returns:
        Full hex digest
    """
    # Convert to JSON with sorted keys for stability
    json_str = json.dumps(data, sort_keys=True, separators=(",", ":"))
    hasher = hashlib.new(algorithm)
    hasher.update(json_str.encode())
    return hasher.hexdigest()


@dataclass
class Node:
    """
    A node in the execution DAG.

    Attributes:
        node_type: The operation type (NodeType enum or string for custom types)
        config: Operation-specific configuration
        inputs: List of input node IDs (resolved during execution)
        node_id: Content-addressed ID (computed from type + config + inputs)
        name: Optional human-readable name for debugging
    """
    node_type: NodeType | str
    config: Dict[str, Any] = field(default_factory=dict)
    inputs: List[str] = field(default_factory=list)
    node_id: Optional[str] = None
    name: Optional[str] = None

    def __post_init__(self):
        """Compute node_id if not provided."""
        if self.node_id is None:
            self.node_id = self._compute_id()

    def _compute_id(self) -> str:
        """Compute content-addressed ID from node contents."""
        type_str = self.node_type.name if isinstance(self.node_type, NodeType) else str(self.node_type)
        content = {
            "type": type_str,
            "config": self.config,
            "inputs": sorted(self.inputs),  # Sort for stability
        }
        return _stable_hash(content)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize node to dictionary."""
        type_str = self.node_type.name if isinstance(self.node_type, NodeType) else str(self.node_type)
        return {
            "node_id": self.node_id,
            "node_type": type_str,
            "config": self.config,
            "inputs": self.inputs,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Node":
        """Deserialize node from dictionary."""
        type_str = data["node_type"]
        try:
            node_type = NodeType[type_str]
        except KeyError:
            node_type = type_str  # Custom type as string

        return cls(
            node_type=node_type,
            config=data.get("config", {}),
            inputs=data.get("inputs", []),
            node_id=data.get("node_id"),
            name=data.get("name"),
        )


@dataclass
class DAG:
    """
    A directed acyclic graph of nodes.

    Attributes:
        nodes: Dictionary mapping node_id -> Node
        output_id: The ID of the final output node
        metadata: Optional metadata about the DAG (source, version, etc.)
    """
    nodes: Dict[str, Node] = field(default_factory=dict)
    output_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_node(self, node: Node) -> str:
        """Add a node to the DAG, returning its ID."""
        if node.node_id in self.nodes:
            # Node already exists (deduplication via content addressing)
            return node.node_id
        self.nodes[node.node_id] = node
        return node.node_id

    def set_output(self, node_id: str) -> None:
        """Set the output node."""
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not in DAG")
        self.output_id = node_id

    def get_node(self, node_id: str) -> Node:
        """Get a node by ID."""
        if node_id not in self.nodes:
            raise KeyError(f"Node {node_id} not found")
        return self.nodes[node_id]

    def topological_order(self) -> List[str]:
        """Return nodes in topological order (dependencies first)."""
        visited = set()
        order = []

        def visit(node_id: str):
            if node_id in visited:
                return
            visited.add(node_id)
            node = self.nodes[node_id]
            for input_id in node.inputs:
                visit(input_id)
            order.append(node_id)

        # Visit all nodes (not just output, in case of disconnected components)
        for node_id in self.nodes:
            visit(node_id)

        return order

    def validate(self) -> List[str]:
        """Validate DAG structure. Returns list of errors (empty if valid)."""
        errors = []

        if self.output_id is None:
            errors.append("No output node set")
        elif self.output_id not in self.nodes:
            errors.append(f"Output node {self.output_id} not in DAG")

        # Check all input references are valid
        for node_id, node in self.nodes.items():
            for input_id in node.inputs:
                if input_id not in self.nodes:
                    errors.append(f"Node {node_id} references missing input {input_id}")

        # Check for cycles (skip if we already found missing inputs)
        if not any("missing" in e for e in errors):
            try:
                self.topological_order()
            except (RecursionError, KeyError):
                errors.append("DAG contains cycles or invalid references")

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Serialize DAG to dictionary."""
        return {
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()},
            "output_id": self.output_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DAG":
        """Deserialize DAG from dictionary."""
        dag = cls(metadata=data.get("metadata", {}))
        for node_data in data.get("nodes", {}).values():
            dag.add_node(Node.from_dict(node_data))
        dag.output_id = data.get("output_id")
        return dag

    def to_json(self) -> str:
        """Serialize DAG to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "DAG":
        """Deserialize DAG from JSON string."""
        return cls.from_dict(json.loads(json_str))


class DAGBuilder:
    """
    Fluent builder for constructing DAGs.

    Example:
        builder = DAGBuilder()
        source = builder.source("/path/to/video.mp4")
        segment = builder.segment(source, duration=5.0)
        builder.set_output(segment)
        dag = builder.build()
    """

    def __init__(self):
        self.dag = DAG()

    def _add(self, node_type: NodeType | str, config: Dict[str, Any],
             inputs: List[str] = None, name: str = None) -> str:
        """Add a node and return its ID."""
        node = Node(
            node_type=node_type,
            config=config,
            inputs=inputs or [],
            name=name,
        )
        return self.dag.add_node(node)

    # Source operations

    def source(self, path: str, name: str = None) -> str:
        """Add a SOURCE node."""
        return self._add(NodeType.SOURCE, {"path": path}, name=name)

    # Transform operations

    def segment(self, input_id: str, duration: float = None,
                offset: float = 0, precise: bool = True, name: str = None) -> str:
        """Add a SEGMENT node."""
        config = {"offset": offset, "precise": precise}
        if duration is not None:
            config["duration"] = duration
        return self._add(NodeType.SEGMENT, config, [input_id], name=name)

    def resize(self, input_id: str, width: int, height: int,
               mode: str = "fit", name: str = None) -> str:
        """Add a RESIZE node."""
        return self._add(
            NodeType.RESIZE,
            {"width": width, "height": height, "mode": mode},
            [input_id],
            name=name
        )

    def transform(self, input_id: str, effects: Dict[str, Any],
                  name: str = None) -> str:
        """Add a TRANSFORM node."""
        return self._add(NodeType.TRANSFORM, {"effects": effects}, [input_id], name=name)

    # Compose operations

    def sequence(self, input_ids: List[str], transition: Dict[str, Any] = None,
                 name: str = None) -> str:
        """Add a SEQUENCE node."""
        config = {"transition": transition or {"type": "cut"}}
        return self._add(NodeType.SEQUENCE, config, input_ids, name=name)

    def layer(self, input_ids: List[str], configs: List[Dict] = None,
              name: str = None) -> str:
        """Add a LAYER node."""
        return self._add(
            NodeType.LAYER,
            {"inputs": configs or [{}] * len(input_ids)},
            input_ids,
            name=name
        )

    def mux(self, video_id: str, audio_id: str, shortest: bool = True,
            name: str = None) -> str:
        """Add a MUX node."""
        return self._add(
            NodeType.MUX,
            {"video_stream": 0, "audio_stream": 1, "shortest": shortest},
            [video_id, audio_id],
            name=name
        )

    def blend(self, input1_id: str, input2_id: str, mode: str = "overlay",
              opacity: float = 0.5, name: str = None) -> str:
        """Add a BLEND node."""
        return self._add(
            NodeType.BLEND,
            {"mode": mode, "opacity": opacity},
            [input1_id, input2_id],
            name=name
        )

    # Output

    def set_output(self, node_id: str) -> "DAGBuilder":
        """Set the output node."""
        self.dag.set_output(node_id)
        return self

    def build(self) -> DAG:
        """Build and validate the DAG."""
        errors = self.dag.validate()
        if errors:
            raise ValueError(f"Invalid DAG: {errors}")
        return self.dag
