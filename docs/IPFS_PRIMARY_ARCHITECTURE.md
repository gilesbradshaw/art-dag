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

## Migration Path

1. Keep current system working
2. Add `--ipfs-primary` flag to CLI
3. New execute_step that works with CIDs
4. Gradually deprecate local cache code
5. Simplify Redis to just claims + cache_id→CID

## See Also

- [L1_STORAGE.md](L1_STORAGE.md) - Current L1 architecture
- [EXECUTION_MODEL.md](EXECUTION_MODEL.md) - 3-phase model
