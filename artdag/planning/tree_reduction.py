# artdag/planning/tree_reduction.py
"""
Tree reduction for parallel composition.

Instead of sequential pairwise composition:
    A → AB → ABC → ABCD  (3 sequential steps)

Use parallel tree reduction:
    A ─┬─ AB ─┬─ ABCD
    B ─┘      │
    C ─┬─ CD ─┘
    D ─┘

This reduces O(N) to O(log N) levels of sequential dependency.
"""

import math
from dataclasses import dataclass
from typing import List, Tuple, Any, Dict


@dataclass
class ReductionNode:
    """A node in the reduction tree."""
    node_id: str
    input_ids: List[str]
    level: int
    position: int  # Position within level


class TreeReducer:
    """
    Generates tree reduction plans for parallel composition.

    Used to convert N inputs into optimal parallel SEQUENCE operations.
    """

    def __init__(self, node_type: str = "SEQUENCE"):
        """
        Initialize the reducer.

        Args:
            node_type: The composition node type (SEQUENCE, AUDIO_MIX, etc.)
        """
        self.node_type = node_type

    def reduce(
        self,
        input_ids: List[str],
        id_prefix: str = "reduce",
    ) -> Tuple[List[ReductionNode], str]:
        """
        Generate a tree reduction plan for the given inputs.

        Args:
            input_ids: List of input step IDs to reduce
            id_prefix: Prefix for generated node IDs

        Returns:
            Tuple of (list of reduction nodes, final output node ID)
        """
        if len(input_ids) == 0:
            raise ValueError("Cannot reduce empty input list")

        if len(input_ids) == 1:
            # Single input, no reduction needed
            return [], input_ids[0]

        if len(input_ids) == 2:
            # Two inputs, single reduction
            node_id = f"{id_prefix}_final"
            node = ReductionNode(
                node_id=node_id,
                input_ids=input_ids,
                level=0,
                position=0,
            )
            return [node], node_id

        # Build tree levels
        nodes = []
        current_level = list(input_ids)
        level_num = 0

        while len(current_level) > 1:
            next_level = []
            position = 0

            # Pair up nodes at current level
            i = 0
            while i < len(current_level):
                if i + 1 < len(current_level):
                    # Pair available
                    left = current_level[i]
                    right = current_level[i + 1]
                    node_id = f"{id_prefix}_L{level_num}_P{position}"
                    node = ReductionNode(
                        node_id=node_id,
                        input_ids=[left, right],
                        level=level_num,
                        position=position,
                    )
                    nodes.append(node)
                    next_level.append(node_id)
                    i += 2
                else:
                    # Odd one out, promote to next level
                    next_level.append(current_level[i])
                    i += 1

                position += 1

            current_level = next_level
            level_num += 1

        # The last remaining node is the output
        output_id = current_level[0]

        # Rename final node for clarity
        if nodes and nodes[-1].node_id == output_id:
            nodes[-1].node_id = f"{id_prefix}_final"
            output_id = f"{id_prefix}_final"

        return nodes, output_id

    def get_reduction_depth(self, n: int) -> int:
        """
        Calculate the number of reduction levels needed.

        Args:
            n: Number of inputs

        Returns:
            Number of sequential reduction levels (log2(n) ceiling)
        """
        if n <= 1:
            return 0
        return math.ceil(math.log2(n))

    def get_total_operations(self, n: int) -> int:
        """
        Calculate total number of reduction operations.

        Args:
            n: Number of inputs

        Returns:
            Total composition operations (always n-1)
        """
        return max(0, n - 1)

    def reduce_with_config(
        self,
        input_ids: List[str],
        base_config: Dict[str, Any],
        id_prefix: str = "reduce",
    ) -> Tuple[List[Tuple[ReductionNode, Dict[str, Any]]], str]:
        """
        Generate reduction plan with configuration for each node.

        Args:
            input_ids: List of input step IDs
            base_config: Base configuration to use for each reduction
            id_prefix: Prefix for generated node IDs

        Returns:
            Tuple of (list of (node, config) pairs, final output ID)
        """
        nodes, output_id = self.reduce(input_ids, id_prefix)
        result = [(node, dict(base_config)) for node in nodes]
        return result, output_id


def reduce_sequence(
    input_ids: List[str],
    transition_config: Dict[str, Any] = None,
    id_prefix: str = "seq",
) -> Tuple[List[Tuple[str, List[str], Dict[str, Any]]], str]:
    """
    Convenience function for SEQUENCE reduction.

    Args:
        input_ids: Input step IDs to sequence
        transition_config: Transition configuration (default: cut)
        id_prefix: Prefix for generated step IDs

    Returns:
        Tuple of (list of (step_id, inputs, config), final step ID)
    """
    if transition_config is None:
        transition_config = {"transition": {"type": "cut"}}

    reducer = TreeReducer("SEQUENCE")
    nodes, output_id = reducer.reduce(input_ids, id_prefix)

    result = [
        (node.node_id, node.input_ids, dict(transition_config))
        for node in nodes
    ]

    return result, output_id


def reduce_audio_mix(
    input_ids: List[str],
    mix_config: Dict[str, Any] = None,
    id_prefix: str = "mix",
) -> Tuple[List[Tuple[str, List[str], Dict[str, Any]]], str]:
    """
    Convenience function for AUDIO_MIX reduction.

    Args:
        input_ids: Input step IDs to mix
        mix_config: Mix configuration
        id_prefix: Prefix for generated step IDs

    Returns:
        Tuple of (list of (step_id, inputs, config), final step ID)
    """
    if mix_config is None:
        mix_config = {"normalize": True}

    reducer = TreeReducer("AUDIO_MIX")
    nodes, output_id = reducer.reduce(input_ids, id_prefix)

    result = [
        (node.node_id, node.input_ids, dict(mix_config))
        for node in nodes
    ]

    return result, output_id
