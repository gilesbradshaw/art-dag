# L1 Distributed Storage Architecture

This document describes how data is stored when running artdag on L1 (the distributed rendering layer).

## Overview

L1 uses four storage systems working together:

| System | Purpose | Data Stored |
|--------|---------|-------------|
| **Local Cache** | Hot storage (fast access) | Media files, plans, analysis |
| **IPFS** | Durable content-addressed storage | All media outputs |
| **Redis** | Coordination & indexes | Claims, mappings, run status |
| **PostgreSQL** | Metadata & ownership | User data, provenance |

## Storage Flow

When a step executes on L1:

```
1. Executor produces output file
2. Store in local cache (fast)
3. Compute content_hash = SHA3-256(file)
4. Upload to IPFS → get ipfs_cid
5. Update indexes:
   - content_hash → node_id (Redis + local)
   - content_hash → ipfs_cid (Redis + local)
```

Every intermediate step output (SEGMENT, SEQUENCE, etc.) gets its own IPFS CID.

## Local Cache

Hot storage on each worker node:

```
cache_dir/
  index.json                    # Cache metadata
  content_index.json            # content_hash → node_id
  ipfs_index.json               # content_hash → ipfs_cid
  plans/
    {plan_id}.json              # Cached execution plans
  analysis/
    {hash}.json                 # Analysis results
  {node_id}/
    output.mkv                  # Media output
    metadata.json               # CacheEntry metadata
```

## IPFS - Durable Media Storage

All media files are stored in IPFS for durability and content-addressing.

**Supported pinning providers:**
- Pinata
- web3.storage
- NFT.Storage
- Infura IPFS
- Filebase (S3-compatible)
- Storj (decentralized)
- Local IPFS node

**Configuration:**
```bash
IPFS_API=/ip4/127.0.0.1/tcp/5001  # Local IPFS daemon
```

## Redis - Coordination

Redis handles distributed coordination across workers.

### Key Patterns

| Key | Type | Purpose |
|-----|------|---------|
| `artdag:run:{run_id}` | String | Run status, timestamps, celery task ID |
| `artdag:content_index` | Hash | content_hash → node_id mapping |
| `artdag:ipfs_index` | Hash | content_hash → ipfs_cid mapping |
| `artdag:claim:{cache_id}` | String | Task claiming (prevents duplicate work) |

### Task Claiming

Lua scripts ensure atomic claiming across workers:

```
Status flow: PENDING → CLAIMED → RUNNING → COMPLETED/CACHED/FAILED
TTL: 5 minutes for claims, 1 hour for results
```

This prevents two workers from executing the same step.

## PostgreSQL - Metadata

Stores ownership, provenance, and sharing metadata.

### Tables

```sql
-- Core cache (shared)
cache_items (content_hash, ipfs_cid, created_at)

-- Per-user ownership
item_types (content_hash, actor_id, type, metadata)

-- Run cache (deterministic identity)
run_cache (
  run_id,           -- SHA3-256(sorted_inputs + recipe)
  output_hash,
  ipfs_cid,
  provenance_cid,
  recipe, inputs, actor_id
)

-- Storage backends
storage_backends (actor_id, provider_type, config, capacity_gb)

-- What's stored where
storage_pins (content_hash, storage_id, ipfs_cid, pin_type)
```

## Cache Lookup Flow

When a worker needs a file:

```
1. Check local cache by cache_id (fastest)
2. Check Redis content_index: content_hash → node_id
3. Check PostgreSQL cache_items
4. Retrieve from IPFS by CID
5. Store in local cache for next hit
```

## Local vs L1 Comparison

| Feature | Local Testing | L1 Distributed |
|---------|---------------|----------------|
| Local cache | Yes | Yes |
| IPFS | No | Yes |
| Redis | No | Yes |
| PostgreSQL | No | Yes |
| Multi-worker | No | Yes |
| Task claiming | No | Yes (Lua scripts) |
| Durability | Filesystem only | IPFS + PostgreSQL |

## Content Addressing

All storage uses SHA3-256 (quantum-resistant):

- **Files:** `content_hash = SHA3-256(file_bytes)`
- **Computation:** `cache_id = SHA3-256(type + config + input_hashes)`
- **Run identity:** `run_id = SHA3-256(sorted_inputs + recipe)`
- **Plans:** `plan_id = SHA3-256(recipe + inputs + analysis)`

This ensures:
- Same inputs → same outputs (reproducibility)
- Automatic deduplication across workers
- Content verification (tamper detection)

## Configuration

Default locations:

```bash
# Local cache
~/.artdag/cache           # Default
/data/cache               # Docker

# Redis
redis://localhost:6379/5

# PostgreSQL
postgresql://user:pass@host/artdag

# IPFS
/ip4/127.0.0.1/tcp/5001
```

## See Also

- [OFFLINE_TESTING.md](OFFLINE_TESTING.md) - Local testing without L1
- [EXECUTION_MODEL.md](EXECUTION_MODEL.md) - 3-phase execution model
