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
from .cache import Cache, CacheEntry
from .executor import Executor, register_executor, get_executor
from .engine import Engine
from .registry import Registry, Asset
from .activities import Activity, ActivityStore, ActivityManager, make_is_shared_fn

# Analysis and planning modules (optional, require extra dependencies)
try:
    from .analysis import Analyzer, AnalysisResult
except ImportError:
    Analyzer = None
    AnalysisResult = None

try:
    from .planning import RecipePlanner, ExecutionPlan, ExecutionStep
except ImportError:
    RecipePlanner = None
    ExecutionPlan = None
    ExecutionStep = None

__all__ = [
    # Core
    "Node",
    "DAG",
    "DAGBuilder",
    "NodeType",
    "Cache",
    "CacheEntry",
    "Executor",
    "register_executor",
    "get_executor",
    "Engine",
    "Registry",
    "Asset",
    "Activity",
    "ActivityStore",
    "ActivityManager",
    "make_is_shared_fn",
    # Analysis (optional)
    "Analyzer",
    "AnalysisResult",
    # Planning (optional)
    "RecipePlanner",
    "ExecutionPlan",
    "ExecutionStep",
]

__version__ = "0.1.0"
