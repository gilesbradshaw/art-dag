# IPFS-Primary Architecture (Sketch)

A simplified L1 architecture for large-scale distributed rendering where IPFS is the primary data store.

## Current vs Simplified

| Component | Current | Simplified |
|-----------|---------|------------|
| Local cache | Custom, per-worker | IPFS node handles it |
| Redis content_index | content_hash → node_id | Eliminated |
| Redis ipfs_index | content_hash → ipfs_cid | Eliminated |
| Step inputs | File paths | IPFS CIDs |
| Step outputs | File path + CID | Just CID |
| Cache lookup | Local → Redis → IPFS | Just IPFS |

## Core Principle

**Steps receive CIDs, produce CIDs. No file paths cross machine boundaries.**

```
Step input:  [cid1, cid2, ...]
Step output: cid_out
```

## Worker Architecture

Each worker runs:

```
┌─────────────────────────────────────┐
│           Worker Node               │
│                                     │
│  ┌───────────┐    ┌──────────────┐  │
│  │  Celery   │────│  IPFS Node   │  │
│  │  Worker   │    │  (local)     │  │
│  └───────────┘    └──────────────┘  │
│       │                  │          │
│       │            ┌─────┴─────┐    │
│       │            │ Local     │    │
│       │            │ Blockstore│    │
│       │            └───────────┘    │
│       │                             │
│  ┌────┴────┐                        │
│  │ /tmp    │  (ephemeral workspace) │
│  └─────────┘                        │
└─────────────────────────────────────┘
         │
         │ IPFS libp2p
         ▼
   ┌─────────────┐
   │ Other IPFS  │
   │   Nodes     │
   └─────────────┘
```

## Execution Flow

### 1. Plan Generation (unchanged)

```python
plan = planner.plan(recipe, input_hashes)
# plan.steps[].cache_id = deterministic hash
```

### 2. Input Registration

Before execution, register inputs with IPFS:

```python
input_cids = {}
for name, path in inputs.items():
    cid = ipfs.add(path)
    input_cids[name] = cid

# Plan now carries CIDs
plan.input_cids = input_cids
```

### 3. Step Execution

```python
@celery.task
def execute_step(step_json: str, input_cids: dict[str, str]) -> str:
    """Execute step, return output CID."""
    step = ExecutionStep.from_json(step_json)

    # Check if already computed (by cache_id as IPNS key or DHT lookup)
    existing_cid = ipfs.resolve(f"/ipns/{step.cache_id}")
    if existing_cid:
        return existing_cid

    # Fetch inputs from IPFS → local temp files
    input_paths = []
    for input_step_id in step.input_steps:
        cid = input_cids[input_step_id]
        path = ipfs.get(cid, f"/tmp/{cid}")  # IPFS node caches automatically
        input_paths.append(path)

    # Execute
    output_path = f"/tmp/{step.cache_id}.mkv"
    executor = get_executor(step.node_type)
    executor.execute(step.config, input_paths, output_path)

    # Add output to IPFS
    output_cid = ipfs.add(output_path)

    # Publish cache_id → CID mapping (optional, for cache hits)
    ipfs.name_publish(step.cache_id, output_cid)

    # Cleanup temp files
    cleanup_temp(input_paths + [output_path])

    return output_cid
```

### 4. Orchestration

```python
@celery.task
def run_plan(plan_json: str) -> str:
    """Execute plan, return final output CID."""
    plan = ExecutionPlan.from_json(plan_json)

    # CID results accumulate as steps complete
    cid_results = dict(plan.input_cids)

    for level in plan.get_steps_by_level():
        # Parallel execution within level
        tasks = []
        for step in level:
            step_input_cids = {
                sid: cid_results[sid]
                for sid in step.input_steps
            }
            tasks.append(execute_step.s(step.to_json(), step_input_cids))

        # Wait for level to complete
        results = group(tasks).apply_async().get()

        # Record output CIDs
        for step, cid in zip(level, results):
            cid_results[step.step_id] = cid

    return cid_results[plan.output_step]
```

## What's Eliminated

### No more Redis indexes

```python
# BEFORE: Complex index management
self._set_content_index(content_hash, node_id)  # Redis + local
self._set_ipfs_index(content_hash, ipfs_cid)    # Redis + local
node_id = self._get_content_index(content_hash)  # Check Redis, fallback local

# AFTER: Just CIDs
output_cid = ipfs.add(output_path)
return output_cid
```

### No more local cache management

```python
# BEFORE: Custom cache with entries, metadata, cleanup
cache.put(node_id, source_path, node_type, execution_time)
cache.get(node_id)
cache.has(node_id)
cache.cleanup_lru()

# AFTER: IPFS handles it
ipfs.add(path)  # Store
ipfs.get(cid)   # Retrieve (cached by IPFS node)
ipfs.pin(cid)   # Keep permanently
ipfs.gc()       # Cleanup unpinned
```

### No more content_hash vs node_id confusion

```python
# BEFORE: Two identifiers
content_hash = sha3_256(file_bytes)  # What the file IS
node_id = cache_id                    # What computation produced it
# Need indexes to map between them

# AFTER: One identifier
cid = ipfs.add(file)  # Content-addressed, includes hash
# CID IS the identifier
```

## Cache Hit Detection

Two options:

### Option A: IPNS (mutable names)

```python
# Publish: cache_id → CID
ipfs.name_publish(key=cache_id, value=output_cid)

# Lookup before executing
existing = ipfs.name_resolve(cache_id)
if existing:
    return existing  # Cache hit
```

### Option B: DHT record

```python
# Store in DHT: cache_id → CID
ipfs.dht_put(cache_id, output_cid)

# Lookup
existing = ipfs.dht_get(cache_id)
```

### Option C: Redis (minimal)

Keep Redis just for cache_id → CID mapping:

```python
# Store
redis.hset("artdag:cache", cache_id, output_cid)

# Lookup
existing = redis.hget("artdag:cache", cache_id)
```

This is simpler than current approach - one hash, one mapping, no content_hash/node_id confusion.

## Claiming (Preventing Duplicate Work)

Still need Redis for atomic claiming:

```python
# Claim before executing
claimed = redis.set(f"artdag:claim:{cache_id}", worker_id, nx=True, ex=300)
if not claimed:
    # Another worker is doing it - wait for result
    return wait_for_result(cache_id)
```

Or use IPFS pubsub for coordination.

## Data Flow Diagram

```
                    ┌─────────────┐
                    │   Recipe    │
                    │   + Inputs  │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │   Planner   │
                    │ (compute    │
                    │  cache_ids) │
                    └──────┬──────┘
                           │
                           ▼
         ┌─────────────────────────────────┐
         │     ExecutionPlan               │
         │  - steps with cache_ids         │
         │  - input_cids (from ipfs.add)   │
         └─────────────────┬───────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
         ┌────────┐   ┌────────┐   ┌────────┐
         │Worker 1│   │Worker 2│   │Worker 3│
         │        │   │        │   │        │
         │ IPFS   │◄──│ IPFS   │◄──│ IPFS   │
         │ Node   │──►│ Node   │──►│ Node   │
         └───┬────┘   └───┬────┘   └───┬────┘
             │            │            │
             └────────────┼────────────┘
                          │
                          ▼
                   ┌─────────────┐
                   │ Final CID   │
                   │ (output)    │
                   └─────────────┘
```

## Benefits

1. **Simpler code** - No custom cache, no dual indexes
2. **Automatic distribution** - IPFS handles replication
3. **Content verification** - CIDs are self-verifying
4. **Scalable** - Add workers = add IPFS nodes = more cache capacity
5. **Resilient** - Any node can serve any content

## Tradeoffs

1. **IPFS dependency** - Every worker needs IPFS node
2. **Initial fetch latency** - First fetch may be slower than local disk
3. **IPNS latency** - Name resolution can be slow (Option C avoids this)

## Trust Domains (Cluster Key)

Systems can share work through IPFS, but how do you trust them?

**Problem:** A malicious system could return wrong CIDs for computed steps.

**Solution:** Cluster key creates isolated trust domains:

```bash
export ARTDAG_CLUSTER_KEY="my-secret-shared-key"
```

**How it works:**
- The cluster key is mixed into all cache_id computations
- Systems with the same key produce the same cache_ids
- Systems with different keys have separate cache namespaces
- Only share the key with trusted partners

```
cache_id = SHA3-256(cluster_key + node_type + config + inputs)
```

**Trust model:**
| Scenario | Same Key? | Can Share Work? |
|----------|-----------|-----------------|
| Same organization | Yes | Yes |
| Trusted partner | Yes (shared) | Yes |
| Unknown system | No | No (different cache_ids) |

**Configuration:**
```yaml
# docker-compose.yml
environment:
  - ARTDAG_CLUSTER_KEY=your-secret-key-here
```

**Programmatic:**
```python
from artdag.planning.schema import set_cluster_key
set_cluster_key("my-secret-key")
```

## Implementation

The simplified architecture is implemented in `art-celery/`:

| File | Purpose |
|------|---------|
| `hybrid_state.py` | Hybrid state manager (Redis + IPNS) |
| `tasks/execute_cid.py` | Step execution with CIDs |
| `tasks/analyze_cid.py` | Analysis with CIDs |
| `tasks/orchestrate_cid.py` | Full pipeline orchestration |

### Key Functions

**Registration (local → IPFS):**
- `register_input_cid(path)` → `{cid, content_hash}`
- `register_recipe_cid(path)` → `{cid, name, version}`

**Analysis:**
- `analyze_input_cid(input_cid, input_hash, features)` → `{analysis_cid}`

**Planning:**
- `generate_plan_cid(recipe_cid, input_cids, input_hashes, analysis_cids)` → `{plan_cid}`

**Execution:**
- `execute_step_cid(step_json, input_cids)` → `{cid}`
- `execute_plan_from_cid(plan_cid, input_cids)` → `{output_cid}`

**Full Pipeline:**
- `run_recipe_cid(recipe_cid, input_cids, input_hashes)` → `{output_cid, all_cids}`
- `run_from_local(recipe_path, input_paths)` → registers + runs

### Hybrid State Manager

For distributed L1 coordination, use the `HybridStateManager` which provides:

**Fast path (local Redis):**
- `get_cached_cid(cache_id)` / `set_cached_cid(cache_id, cid)` - microsecond lookups
- `try_claim(cache_id, worker_id)` / `release_claim(cache_id)` - atomic claiming
- `get_analysis_cid()` / `set_analysis_cid()` - analysis cache
- `get_plan_cid()` / `set_plan_cid()` - plan cache
- `get_run_cid()` / `set_run_cid()` - run cache

**Slow path (background IPNS sync):**
- Periodically syncs local state with global IPNS state (default: every 30s)
- Pulls new entries from remote nodes
- Pushes local updates to IPNS

**Configuration:**
```bash
# Enable IPNS sync
export ARTDAG_IPNS_SYNC=true
export ARTDAG_IPNS_SYNC_INTERVAL=30  # seconds
```

**Usage:**
```python
from hybrid_state import get_state_manager

state = get_state_manager()

# Fast local lookup
cid = state.get_cached_cid(cache_id)

# Fast local write (synced in background)
state.set_cached_cid(cache_id, output_cid)

# Atomic claim
if state.try_claim(cache_id, worker_id):
    # We have the lock
    ...
```

**Trade-offs:**
- Local Redis: Fast (microseconds), single node
- IPNS sync: Slow (seconds), eventually consistent across nodes
- Duplicate work: Accepted (idempotent - same inputs → same CID)

### Redis Usage (minimal)

| Key | Type | Purpose |
|-----|------|---------|
| `artdag:cid_cache` | Hash | cache_id → output CID |
| `artdag:analysis_cache` | Hash | input_hash:features → analysis CID |
| `artdag:plan_cache` | Hash | plan_id → plan CID |
| `artdag:run_cache` | Hash | run_id → output CID |
| `artdag:claim:{cache_id}` | String | worker_id (TTL 5 min) |

## Migration Path

1. Keep current system working ✓
2. Add CID-based tasks ✓
   - `execute_cid.py` ✓
   - `analyze_cid.py` ✓
   - `orchestrate_cid.py` ✓
3. Add `--ipfs-primary` flag to CLI ✓
4. Add hybrid state manager for L1 coordination ✓
5. Gradually deprecate local cache code
6. Remove old tasks when CID versions are stable

## See Also

- [L1_STORAGE.md](L1_STORAGE.md) - Current L1 architecture
- [EXECUTION_MODEL.md](EXECUTION_MODEL.md) - 3-phase model
