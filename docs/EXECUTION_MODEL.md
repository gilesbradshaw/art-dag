# Art DAG 3-Phase Execution Model

## Overview

The execution model separates DAG processing into three distinct phases:

```
Recipe + Inputs → ANALYZE → Analysis Results
                      ↓
Analysis + Recipe → PLAN → Execution Plan (with cache IDs)
                      ↓
Execution Plan → EXECUTE → Cached Results
```

This separation enables:
1. **Incremental development** - Re-run recipes without reprocessing unchanged steps
2. **Parallel execution** - Independent steps run concurrently via Celery
3. **Deterministic caching** - Same inputs always produce same cache IDs
4. **Cost estimation** - Plan phase can estimate work before executing

## Phase 1: Analysis

### Purpose
Extract features from input media that inform downstream processing decisions.

### Inputs
- Recipe YAML with input references
- Input media files (by content hash)

### Outputs
Analysis results stored as JSON, keyed by input hash:

```python
@dataclass
class AnalysisResult:
    input_hash: str
    features: Dict[str, Any]
    # Audio features
    beats: Optional[List[float]]        # Beat times in seconds
    downbeats: Optional[List[float]]    # Bar-start times
    tempo: Optional[float]              # BPM
    energy: Optional[List[Tuple[float, float]]]  # (time, value) envelope
    spectrum: Optional[Dict[str, List[Tuple[float, float]]]]  # band envelopes
    # Video features
    duration: float
    frame_rate: float
    dimensions: Tuple[int, int]
    motion_tempo: Optional[float]       # Estimated BPM from motion
```

### Implementation
```python
class Analyzer:
    def analyze(self, input_hash: str, features: List[str]) -> AnalysisResult:
        """Extract requested features from input."""

    def analyze_audio(self, path: Path) -> AudioFeatures:
        """Extract all audio features using librosa/essentia."""

    def analyze_video(self, path: Path) -> VideoFeatures:
        """Extract video metadata and motion analysis."""
```

### Caching
Analysis results are cached by:
```
analysis_cache_id = SHA3-256(input_hash + sorted(feature_names))
```

## Phase 2: Planning

### Purpose
Convert recipe + analysis into a complete execution plan with pre-computed cache IDs.

### Inputs
- Recipe YAML (parsed)
- Analysis results for all inputs
- Recipe parameters (user-supplied values)

### Outputs
An ExecutionPlan containing ordered steps, each with a pre-computed cache ID:

```python
@dataclass
class ExecutionStep:
    step_id: str                    # Unique identifier
    node_type: str                  # Primitive type (SOURCE, SEQUENCE, etc.)
    config: Dict[str, Any]          # Node configuration
    input_steps: List[str]          # IDs of steps this depends on
    cache_id: str                   # Pre-computed: hash(inputs + config)
    estimated_duration: float       # Optional: for progress reporting

@dataclass
class ExecutionPlan:
    plan_id: str                    # Hash of entire plan
    recipe_id: str                  # Source recipe
    steps: List[ExecutionStep]      # Topologically sorted
    analysis: Dict[str, AnalysisResult]
    output_step: str                # Final step ID

    def compute_cache_ids(self):
        """Compute all cache IDs in dependency order."""
```

### Cache ID Computation

Cache IDs are computed in topological order so each step's cache ID
incorporates its inputs' cache IDs:

```python
def compute_cache_id(step: ExecutionStep, resolved_inputs: Dict[str, str]) -> str:
    """
    Cache ID = SHA3-256(
        node_type +
        canonical_json(config) +
        sorted([input_cache_ids])
    )
    """
    components = [
        step.node_type,
        json.dumps(step.config, sort_keys=True),
        *sorted(resolved_inputs[s] for s in step.input_steps)
    ]
    return sha3_256('|'.join(components))
```

### Plan Generation

The planner expands recipe nodes into concrete steps:

1. **SOURCE nodes** → Direct step with input hash as cache ID
2. **ANALYZE nodes** → Step that references analysis results
3. **TRANSFORM nodes** → Step with static config
4. **TRANSFORM_DYNAMIC nodes** → Expanded to per-frame steps (or use BIND output)
5. **SEQUENCE nodes** → Tree reduction for parallel composition
6. **MAP nodes** → Expanded to N parallel steps + reduction

### Tree Reduction for Composition

Instead of sequential pairwise composition:
```
A → B → C → D  (3 sequential steps)
```

Use parallel tree reduction:
```
A ─┬─ AB ─┬─ ABCD
B ─┘      │
C ─┬─ CD ─┘
D ─┘

Level 0: [A, B, C, D]     (4 parallel)
Level 1: [AB, CD]         (2 parallel)
Level 2: [ABCD]           (1 final)
```

This reduces O(N) to O(log N) levels.

## Phase 3: Execution

### Purpose
Execute the plan, skipping steps with cached results.

### Inputs
- ExecutionPlan with pre-computed cache IDs
- Cache state (which IDs already exist)

### Process

1. **Claim Check**: For each step, atomically check if result is cached
2. **Task Dispatch**: Uncached steps dispatched to Celery workers
3. **Parallel Execution**: Independent steps run concurrently
4. **Result Storage**: Each step stores result with its cache ID
5. **Progress Tracking**: Real-time status updates

### Hash-Based Task Claiming

Prevents duplicate work when multiple workers process the same plan:

```lua
-- Redis Lua script for atomic claim
local key = KEYS[1]
local data = redis.call('GET', key)
if data then
    local status = cjson.decode(data)
    if status.status == 'running' or
       status.status == 'completed' or
       status.status == 'cached' then
        return 0  -- Already claimed/done
    end
end
local claim_data = ARGV[1]
local ttl = tonumber(ARGV[2])
redis.call('SETEX', key, ttl, claim_data)
return 1  -- Successfully claimed
```

### Celery Task Structure

```python
@app.task(bind=True)
def execute_step(self, step_json: str, plan_id: str) -> dict:
    """Execute a single step with caching."""
    step = ExecutionStep.from_json(step_json)

    # Check cache first
    if cache.has(step.cache_id):
        return {'status': 'cached', 'cache_id': step.cache_id}

    # Try to claim this work
    if not claim_task(step.cache_id, self.request.id):
        # Another worker is handling it, wait for result
        return wait_for_result(step.cache_id)

    # Do the work
    executor = get_executor(step.node_type)
    input_paths = [cache.get(s) for s in step.input_steps]
    output_path = cache.get_output_path(step.cache_id)

    result_path = executor.execute(step.config, input_paths, output_path)
    cache.put(step.cache_id, result_path)

    return {'status': 'completed', 'cache_id': step.cache_id}
```

### Execution Orchestration

```python
class PlanExecutor:
    def execute(self, plan: ExecutionPlan) -> ExecutionResult:
        """Execute plan with parallel Celery tasks."""

        # Group steps by level (steps at same level can run in parallel)
        levels = self.compute_dependency_levels(plan.steps)

        for level_steps in levels:
            # Dispatch all steps at this level
            tasks = [
                execute_step.delay(step.to_json(), plan.plan_id)
                for step in level_steps
                if not self.cache.has(step.cache_id)
            ]

            # Wait for level completion
            results = [task.get() for task in tasks]

        return self.collect_results(plan)
```

## Data Flow Example

### Recipe: beat-cuts
```yaml
nodes:
  - id: music
    type: SOURCE
    config: { input: true }

  - id: beats
    type: ANALYZE
    config: { feature: beats }
    inputs: [music]

  - id: videos
    type: SOURCE_LIST
    config: { input: true }

  - id: slices
    type: MAP
    config: { operation: RANDOM_SLICE }
    inputs:
      items: videos
      timing: beats

  - id: final
    type: SEQUENCE
    inputs: [slices]
```

### Phase 1: Analysis
```python
# Input: music file with hash abc123
analysis = {
    'abc123': AnalysisResult(
        beats=[0.0, 0.48, 0.96, 1.44, ...],
        tempo=125.0,
        duration=180.0
    )
}
```

### Phase 2: Planning
```python
# Expands MAP into concrete steps
plan = ExecutionPlan(
    steps=[
        # Source steps
        ExecutionStep(id='music', cache_id='abc123', ...),
        ExecutionStep(id='video_0', cache_id='def456', ...),
        ExecutionStep(id='video_1', cache_id='ghi789', ...),

        # Slice steps (one per beat group)
        ExecutionStep(id='slice_0', cache_id='hash(video_0+timing)', ...),
        ExecutionStep(id='slice_1', cache_id='hash(video_1+timing)', ...),
        ...

        # Tree reduction for sequence
        ExecutionStep(id='seq_0_1', inputs=['slice_0', 'slice_1'], ...),
        ExecutionStep(id='seq_2_3', inputs=['slice_2', 'slice_3'], ...),
        ExecutionStep(id='seq_final', inputs=['seq_0_1', 'seq_2_3'], ...),
    ]
)
```

### Phase 3: Execution
```
Level 0: [music, video_0, video_1] → all cached (SOURCE)
Level 1: [slice_0, slice_1, slice_2, slice_3] → 4 parallel tasks
Level 2: [seq_0_1, seq_2_3] → 2 parallel SEQUENCE tasks
Level 3: [seq_final] → 1 final SEQUENCE task
```

## File Structure

```
artdag/
├── artdag/
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── analyzer.py      # Main Analyzer class
│   │   ├── audio.py         # Audio feature extraction
│   │   └── video.py         # Video feature extraction
│   ├── planning/
│   │   ├── __init__.py
│   │   ├── planner.py       # RecipePlanner class
│   │   ├── schema.py        # ExecutionPlan, ExecutionStep
│   │   └── tree_reduction.py # Parallel composition optimizer
│   └── execution/
│       ├── __init__.py
│       ├── executor.py      # PlanExecutor class
│       └── claiming.py      # Hash-based task claiming

art-celery/
├── tasks/
│   ├── __init__.py
│   ├── analyze.py           # analyze_inputs task
│   ├── plan.py              # generate_plan task
│   ├── execute.py           # execute_step task
│   └── orchestrate.py       # run_plan (coordinates all)
├── claiming.py              # Redis Lua scripts
└── ...
```

## CLI Interface

```bash
# Full pipeline
artdag run-recipe recipes/beat-cuts/recipe.yaml \
    -i music:abc123 \
    -i videos:def456,ghi789

# Phase by phase
artdag analyze recipes/beat-cuts/recipe.yaml -i music:abc123
# → outputs analysis.json

artdag plan recipes/beat-cuts/recipe.yaml --analysis analysis.json
# → outputs plan.json

artdag execute plan.json
# → runs with caching, skips completed steps

# Dry run (show what would execute)
artdag execute plan.json --dry-run
# → shows which steps are cached vs need execution
```

## Benefits

1. **Development Speed**: Change recipe, re-run → only affected steps execute
2. **Parallelism**: Independent steps run on multiple Celery workers
3. **Reproducibility**: Same inputs + recipe = same cache IDs = same output
4. **Visibility**: Plan shows exactly what will happen before execution
5. **Cost Control**: Estimate compute before committing resources
6. **Fault Tolerance**: Failed runs resume from last successful step
