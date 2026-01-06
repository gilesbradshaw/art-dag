# artdag - Content-addressed DAG execution engine with ActivityPub ownership
#
# A standalone execution engine that processes directed acyclic graphs (DAGs)
# where each node represents an operation. Nodes are content-addressed for
# automatic caching and deduplication.
#
# Core concepts:
# - Node: An operation with type, config, and inputs
# - DAG: A graph of nodes with a designated output node
# - Executor: Implements the actual operation for a node type
# - Engine: Executes DAGs by resolving dependencies and running executors

from .dag import Node, DAG, DAGBuilder, NodeType
from .cache import Cache
from .executor import Executor, register_executor, get_executor
from .engine import Engine
from .registry import Registry, Asset

__all__ = [
    "Node",
    "DAG",
    "DAGBuilder",
    "NodeType",
    "Cache",
    "Executor",
    "register_executor",
    "get_executor",
    "Engine",
    "Registry",
    "Asset",
]

__version__ = "0.1.0"
