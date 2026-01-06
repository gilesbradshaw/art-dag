# artdag

Content-addressed DAG execution engine with ActivityPub ownership.

## Features

- **Content-addressed nodes**: `node_id = SHA3-256(type + config + inputs)` for automatic deduplication
- **Quantum-resistant hashing**: SHA-3 throughout for future-proof integrity
- **ActivityPub ownership**: Cryptographically signed ownership claims
- **Federated identity**: `@user@artdag.rose-ash.com` style identities
- **Pluggable executors**: Register custom node types
- **Built-in video primitives**: SOURCE, SEGMENT, RESIZE, TRANSFORM, SEQUENCE, MUX, BLEND

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from artdag import Engine, DAGBuilder, Registry
from artdag.activitypub import OwnershipManager

# Create ownership manager
manager = OwnershipManager("./my_registry")

# Create your identity
actor = manager.create_actor("alice", "Alice")
print(f"Created: {actor.handle}")  # @alice@artdag.rose-ash.com

# Register an asset with ownership
asset, activity = manager.register_asset(
    actor=actor,
    name="my_image",
    path="/path/to/image.jpg",
    tags=["photo", "art"],
)
print(f"Owned: {asset.name} (hash: {asset.content_hash})")

# Build and execute a DAG
engine = Engine("./cache")
builder = DAGBuilder()

source = builder.source(str(asset.path))
resized = builder.resize(source, width=1920, height=1080)
builder.set_output(resized)

result = engine.execute(builder.build())
print(f"Output: {result.output_path}")
```

## Architecture

```
artdag/
├── dag.py           # Node, DAG, DAGBuilder
├── cache.py         # Content-addressed file cache
├── executor.py      # Base executor + registry
├── engine.py        # DAG execution engine
├── registry/        # Asset registry
├── activitypub/     # Identity + ownership
│   ├── actor.py     # Actor identity with RSA keys
│   ├── activity.py  # Create, Announce activities
│   ├── signatures.py # RSA signing/verification
│   └── ownership.py # Links actors to assets
└── nodes/           # Built-in executors
    ├── source.py    # SOURCE
    ├── transform.py # SEGMENT, RESIZE, TRANSFORM
    └── compose.py   # SEQUENCE, LAYER, MUX, BLEND
```

## License

MIT
