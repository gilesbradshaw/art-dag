# primitive/registry/__init__.py
"""
Art DAG Registry.

The registry is the foundational data structure that maps named assets
to their source paths or content-addressed IDs. Assets in the registry
can be referenced by DAGs.

Example:
    registry = Registry("/path/to/registry")
    registry.add("cat", "/path/to/cat.jpg", tags=["animal", "photo"])

    # Later, in a DAG:
    builder = DAGBuilder()
    cat = builder.source(registry.get("cat").path)
"""

from .registry import Registry, Asset

__all__ = ["Registry", "Asset"]
