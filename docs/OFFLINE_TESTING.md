# Offline Testing Strategy

This document describes how to test artdag locally without requiring Redis, IPFS, Celery, or any external distributed infrastructure.

## Overview

The artdag system uses a **3-Phase Execution Model** that enables complete offline testing:

1. **Analysis** - Extract features from input media
2. **Planning** - Generate deterministic execution plan with pre-computed cache IDs
3. **Execution** - Run plan steps, skipping cached results

This separation allows testing each phase independently and running full pipelines locally.

## Quick Start

Run a full offline test with a video file:

```bash
./examples/test_local.sh ../artdag-art-source/dog.mkv
```

This will:
1. Compute the SHA3-256 hash of the input video
2. Run the `simple_sequence` recipe
3. Store all outputs in `test_cache/`

## Test Scripts

### `test_local.sh` - Full Pipeline Test

Location: `./examples/test_local.sh`

Runs the complete artdag pipeline offline with a real video file.

**Usage:**
```bash
./examples/test_local.sh <video_file>
```

**Example:**
```bash
./examples/test_local.sh ../artdag-art-source/dog.mkv
```

**What it does:**
- Computes content hash of input video
- Runs `artdag run-recipe` with `simple_sequence.yaml`
- Stores outputs in `test_cache/` directory
- No external services required

### `test_plan.py` - Planning Phase Test

Location: `./examples/test_plan.py`

Tests the planning phase without requiring any media files.

**Usage:**
```bash
python3 examples/test_plan.py
```

**What it tests:**
- Recipe loading and YAML parsing
- Execution plan generation
- Cache ID computation (deterministic)
- Multi-level parallel step organization
- Human-readable step names
- Multi-output support

**Output:**
- Prints plan structure to console
- Saves full plan to `test_plan_output.json`

### `simple_sequence.yaml` - Sample Recipe

Location: `./examples/simple_sequence.yaml`

A simple recipe for testing that:
- Takes a video input
- Extracts two segments (0-2s and 5-7s)
- Concatenates them with SEQUENCE

## Test Outputs

All test outputs are stored locally and git-ignored:

| Output | Description |
|--------|-------------|
| `test_cache/` | Cached execution results (media files, analysis, plans) |
| `test_cache/plans/` | Cached execution plans by plan_id |
| `test_cache/analysis/` | Cached analysis results by input hash |
| `test_plan_output.json` | Generated execution plan from `test_plan.py` |

## Unit Tests

The project includes a comprehensive pytest test suite in `tests/`:

```bash
# Run all unit tests
pytest

# Run specific test file
pytest tests/test_dag.py
pytest tests/test_engine.py
pytest tests/test_cache.py
```

## Testing Each Phase

### Phase 1: Analysis Only

Extract features without full execution:

```bash
python3 -m artdag.cli analyze <recipe> -i <name>:<hash>@<path> --features beats,energy
```

### Phase 2: Planning Only

Generate an execution plan (no media needed):

```bash
python3 -m artdag.cli plan <recipe> -i <name>:<hash>
```

Or use the test script:

```bash
python3 examples/test_plan.py
```

### Phase 3: Execution Only

Execute a pre-generated plan:

```bash
python3 -m artdag.cli execute plan.json
```

With dry-run to see what would execute:

```bash
python3 -m artdag.cli execute plan.json --dry-run
```

## Key Testing Features

### Content Addressing

All nodes have deterministic IDs computed as:
```
SHA3-256(type + config + sorted(input_IDs))
```

Same inputs always produce same cache IDs, enabling:
- Reproducibility across runs
- Automatic deduplication
- Incremental execution (only changed steps run)

### Local Caching

The `test_cache/` directory stores:
- `plans/{plan_id}.json` - Execution plans (deterministic hash of recipe + inputs + analysis)
- `analysis/{hash}.json` - Analysis results (audio beats, tempo, energy)
- `{cache_id}/output.mkv` - Media outputs from each step

Subsequent test runs automatically skip cached steps. Plans are cached by their `plan_id`, which is a SHA3-256 hash of the recipe, input hashes, and analysis results - so the same recipe with the same inputs always produces the same plan.

### No External Dependencies

Offline testing requires:
- Python 3.9+
- ffmpeg (for media processing)
- No Redis, IPFS, Celery, or network access

## Debugging Tips

1. **Check cache contents:**
   ```bash
   ls -la test_cache/
   ls -la test_cache/plans/
   ```

2. **View cached plan:**
   ```bash
   cat test_cache/plans/*.json | python3 -m json.tool | head -50
   ```

3. **View execution plan structure:**
   ```bash
   cat test_plan_output.json | python3 -m json.tool
   ```

4. **Run with verbose output:**
   ```bash
   python3 -m artdag.cli run-recipe examples/simple_sequence.yaml \
       -i "video:HASH@path" \
       --cache-dir test_cache \
       -v
   ```

5. **Dry-run to see what would execute:**
   ```bash
   python3 -m artdag.cli execute plan.json --dry-run
   ```

## See Also

- [L1_STORAGE.md](L1_STORAGE.md) - Distributed storage on L1 (IPFS, Redis, PostgreSQL)
- [EXECUTION_MODEL.md](EXECUTION_MODEL.md) - 3-phase execution model
