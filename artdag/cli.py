#!/usr/bin/env python3
"""
Art DAG CLI

Command-line interface for the 3-phase execution model:
  artdag analyze - Extract features from inputs
  artdag plan - Generate execution plan
  artdag execute - Run the plan
  artdag run-recipe - Full pipeline

Usage:
  artdag analyze <recipe> -i <name>:<hash>[@<path>] [--features <list>]
  artdag plan <recipe> -i <name>:<hash> [--analysis <file>]
  artdag execute <plan.json> [--dry-run]
  artdag run-recipe <recipe> -i <name>:<hash>[@<path>]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def parse_input(input_str: str) -> Tuple[str, str, Optional[str]]:
    """
    Parse input specification: name:hash[@path]

    Returns (name, hash, path or None)
    """
    if "@" in input_str:
        name_hash, path = input_str.rsplit("@", 1)
    else:
        name_hash = input_str
        path = None

    if ":" not in name_hash:
        raise ValueError(f"Invalid input format: {input_str}. Expected name:hash[@path]")

    name, hash_value = name_hash.split(":", 1)
    return name, hash_value, path


def parse_inputs(input_list: List[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Parse list of input specifications.

    Returns (input_hashes, input_paths)
    """
    input_hashes = {}
    input_paths = {}

    for input_str in input_list:
        name, hash_value, path = parse_input(input_str)
        input_hashes[name] = hash_value
        if path:
            input_paths[name] = path

    return input_hashes, input_paths


def cmd_analyze(args):
    """Run analysis phase."""
    from .analysis import Analyzer

    # Parse inputs
    input_hashes, input_paths = parse_inputs(args.input)

    # Parse features
    features = args.features.split(",") if args.features else ["all"]

    # Create analyzer
    cache_dir = Path(args.cache_dir) if args.cache_dir else Path("./analysis_cache")
    analyzer = Analyzer(cache_dir=cache_dir)

    # Analyze each input
    results = {}
    for name, hash_value in input_hashes.items():
        path = input_paths.get(name)
        if path:
            path = Path(path)

        print(f"Analyzing {name} ({hash_value[:16]}...)...")

        result = analyzer.analyze(
            input_hash=hash_value,
            features=features,
            input_path=path,
        )

        results[hash_value] = result.to_dict()

        # Print summary
        if result.audio and result.audio.beats:
            print(f"  Tempo: {result.audio.beats.tempo:.1f} BPM")
            print(f"  Beats: {len(result.audio.beats.beat_times)}")
        if result.video:
            print(f"  Duration: {result.video.duration:.1f}s")
            print(f"  Dimensions: {result.video.width}x{result.video.height}")

    # Write output
    output_path = Path(args.output) if args.output else Path("analysis.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nAnalysis saved to: {output_path}")


def cmd_plan(args):
    """Run planning phase."""
    from .analysis import AnalysisResult
    from .planning import RecipePlanner, Recipe

    # Load recipe
    recipe = Recipe.from_file(Path(args.recipe))
    print(f"Recipe: {recipe.name} v{recipe.version}")

    # Parse inputs
    input_hashes, _ = parse_inputs(args.input)

    # Load analysis if provided
    analysis = {}
    if args.analysis:
        with open(args.analysis, "r") as f:
            analysis_data = json.load(f)
        for hash_value, data in analysis_data.items():
            analysis[hash_value] = AnalysisResult.from_dict(data)

    # Create planner
    planner = RecipePlanner(use_tree_reduction=not args.no_tree_reduction)

    # Generate plan
    print("Generating execution plan...")
    plan = planner.plan(
        recipe=recipe,
        input_hashes=input_hashes,
        analysis=analysis,
    )

    # Print summary
    print(f"\nPlan ID: {plan.plan_id[:16]}...")
    print(f"Steps: {len(plan.steps)}")

    steps_by_level = plan.get_steps_by_level()
    max_level = max(steps_by_level.keys()) if steps_by_level else 0
    print(f"Levels: {max_level + 1}")

    for level in sorted(steps_by_level.keys()):
        steps = steps_by_level[level]
        print(f"  Level {level}: {len(steps)} steps (parallel)")

    # Write output
    output_path = Path(args.output) if args.output else Path("plan.json")
    with open(output_path, "w") as f:
        f.write(plan.to_json())

    print(f"\nPlan saved to: {output_path}")


def cmd_execute(args):
    """Run execution phase."""
    from .planning import ExecutionPlan
    from .cache import Cache
    from .executor import get_executor
    from .dag import NodeType
    from . import nodes  # Register built-in executors

    # Load plan
    with open(args.plan, "r") as f:
        plan = ExecutionPlan.from_json(f.read())

    print(f"Executing plan: {plan.plan_id[:16]}...")
    print(f"Steps: {len(plan.steps)}")

    if args.dry_run:
        print("\n=== DRY RUN ===")

        # Check cache status
        cache = Cache(Path(args.cache_dir) if args.cache_dir else Path("./cache"))
        steps_by_level = plan.get_steps_by_level()

        cached_count = 0
        pending_count = 0

        for level in sorted(steps_by_level.keys()):
            steps = steps_by_level[level]
            print(f"\nLevel {level}:")
            for step in steps:
                if cache.has(step.cache_id):
                    print(f"  [CACHED] {step.step_id}: {step.node_type}")
                    cached_count += 1
                else:
                    print(f"  [PENDING] {step.step_id}: {step.node_type}")
                    pending_count += 1

        print(f"\nSummary: {cached_count} cached, {pending_count} pending")
        return

    # Execute locally (for testing - production uses Celery)
    cache = Cache(Path(args.cache_dir) if args.cache_dir else Path("./cache"))

    cache_paths = {}
    for name, hash_value in plan.input_hashes.items():
        if cache.has(hash_value):
            entry = cache.get(hash_value)
            cache_paths[hash_value] = str(entry.output_path)

    steps_by_level = plan.get_steps_by_level()
    executed = 0
    cached = 0

    for level in sorted(steps_by_level.keys()):
        steps = steps_by_level[level]
        print(f"\nLevel {level}: {len(steps)} steps")

        for step in steps:
            if cache.has(step.cache_id):
                cached_path = cache.get(step.cache_id)
                cache_paths[step.cache_id] = str(cached_path)
                cache_paths[step.step_id] = str(cached_path)
                print(f"  [CACHED] {step.step_id}")
                cached += 1
                continue

            print(f"  [RUNNING] {step.step_id}: {step.node_type}...")

            # Get executor
            try:
                node_type = NodeType[step.node_type]
            except KeyError:
                node_type = step.node_type

            executor = get_executor(node_type)
            if executor is None:
                print(f"    ERROR: No executor for {step.node_type}")
                continue

            # Resolve inputs
            input_paths = []
            for input_id in step.input_steps:
                if input_id in cache_paths:
                    input_paths.append(Path(cache_paths[input_id]))
                else:
                    input_step = plan.get_step(input_id)
                    if input_step and input_step.cache_id in cache_paths:
                        input_paths.append(Path(cache_paths[input_step.cache_id]))

            if len(input_paths) != len(step.input_steps):
                print(f"    ERROR: Missing inputs")
                continue

            # Execute
            output_path = cache.get_output_path(step.cache_id)
            try:
                result_path = executor.execute(step.config, input_paths, output_path)
                cache.put(step.cache_id, result_path, node_type=step.node_type)
                cache_paths[step.cache_id] = str(result_path)
                cache_paths[step.step_id] = str(result_path)
                print(f"    [DONE] -> {result_path}")
                executed += 1
            except Exception as e:
                print(f"    [FAILED] {e}")

    # Final output
    output_step = plan.get_step(plan.output_step)
    output_path = cache_paths.get(output_step.cache_id) if output_step else None

    print(f"\n=== Complete ===")
    print(f"Cached: {cached}")
    print(f"Executed: {executed}")
    if output_path:
        print(f"Output: {output_path}")


def cmd_run_recipe(args):
    """Run complete pipeline: analyze → plan → execute."""
    from .analysis import Analyzer, AnalysisResult
    from .planning import RecipePlanner, Recipe
    from .cache import Cache
    from .executor import get_executor
    from .dag import NodeType
    from . import nodes  # Register built-in executors

    # Load recipe
    recipe = Recipe.from_file(Path(args.recipe))
    print(f"Recipe: {recipe.name} v{recipe.version}")

    # Parse inputs
    input_hashes, input_paths = parse_inputs(args.input)

    # Parse features
    features = args.features.split(",") if args.features else ["beats", "energy"]

    cache_dir = Path(args.cache_dir) if args.cache_dir else Path("./cache")

    # Phase 1: Analyze
    print("\n=== Phase 1: Analysis ===")
    analyzer = Analyzer(cache_dir=cache_dir / "analysis")

    analysis = {}
    for name, hash_value in input_hashes.items():
        path = input_paths.get(name)
        if path:
            path = Path(path)
            print(f"Analyzing {name}...")

            result = analyzer.analyze(
                input_hash=hash_value,
                features=features,
                input_path=path,
            )
            analysis[hash_value] = result

            if result.audio and result.audio.beats:
                print(f"  Tempo: {result.audio.beats.tempo:.1f} BPM, {len(result.audio.beats.beat_times)} beats")

    # Phase 2: Plan
    print("\n=== Phase 2: Planning ===")

    # Check for cached plan
    plans_dir = cache_dir / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)

    # Generate plan to get plan_id (deterministic hash)
    planner = RecipePlanner(use_tree_reduction=True)
    plan = planner.plan(
        recipe=recipe,
        input_hashes=input_hashes,
        analysis=analysis,
    )

    plan_cache_path = plans_dir / f"{plan.plan_id}.json"

    if plan_cache_path.exists():
        print(f"Plan cached: {plan.plan_id[:16]}...")
        from .planning import ExecutionPlan
        with open(plan_cache_path, "r") as f:
            plan = ExecutionPlan.from_json(f.read())
    else:
        # Save plan to cache
        with open(plan_cache_path, "w") as f:
            f.write(plan.to_json())
        print(f"Plan saved: {plan.plan_id[:16]}...")

    print(f"Plan: {len(plan.steps)} steps")
    steps_by_level = plan.get_steps_by_level()
    print(f"Levels: {len(steps_by_level)}")

    # Phase 3: Execute
    print("\n=== Phase 3: Execution ===")

    cache = Cache(cache_dir)

    # Build initial cache paths
    cache_paths = {}
    for name, hash_value in input_hashes.items():
        path = input_paths.get(name)
        if path:
            cache_paths[hash_value] = path
            cache_paths[name] = path

    executed = 0
    cached = 0

    for level in sorted(steps_by_level.keys()):
        steps = steps_by_level[level]
        print(f"\nLevel {level}: {len(steps)} steps")

        for step in steps:
            if cache.has(step.cache_id):
                cached_path = cache.get(step.cache_id)
                cache_paths[step.cache_id] = str(cached_path)
                cache_paths[step.step_id] = str(cached_path)
                print(f"  [CACHED] {step.step_id}")
                cached += 1
                continue

            # Handle SOURCE specially
            if step.node_type == "SOURCE":
                content_hash = step.config.get("content_hash")
                if content_hash in cache_paths:
                    cache_paths[step.cache_id] = cache_paths[content_hash]
                    cache_paths[step.step_id] = cache_paths[content_hash]
                    print(f"  [SOURCE] {step.step_id}")
                    continue

            print(f"  [RUNNING] {step.step_id}: {step.node_type}...")

            try:
                node_type = NodeType[step.node_type]
            except KeyError:
                node_type = step.node_type

            executor = get_executor(node_type)
            if executor is None:
                print(f"    SKIP: No executor for {step.node_type}")
                continue

            # Resolve inputs
            input_paths_list = []
            for input_id in step.input_steps:
                if input_id in cache_paths:
                    input_paths_list.append(Path(cache_paths[input_id]))
                else:
                    input_step = plan.get_step(input_id)
                    if input_step and input_step.cache_id in cache_paths:
                        input_paths_list.append(Path(cache_paths[input_step.cache_id]))

            if len(input_paths_list) != len(step.input_steps):
                print(f"    ERROR: Missing inputs for {step.step_id}")
                continue

            output_path = cache.get_output_path(step.cache_id)
            try:
                result_path = executor.execute(step.config, input_paths_list, output_path)
                cache.put(step.cache_id, result_path, node_type=step.node_type)
                cache_paths[step.cache_id] = str(result_path)
                cache_paths[step.step_id] = str(result_path)
                print(f"    [DONE]")
                executed += 1
            except Exception as e:
                print(f"    [FAILED] {e}")

    # Final output
    output_step = plan.get_step(plan.output_step)
    output_path = cache_paths.get(output_step.cache_id) if output_step else None

    print(f"\n=== Complete ===")
    print(f"Cached: {cached}")
    print(f"Executed: {executed}")
    if output_path:
        print(f"Output: {output_path}")


def cmd_run_recipe_ipfs(args):
    """Run complete pipeline with IPFS-primary mode.

    Everything stored on IPFS:
    - Inputs (media files)
    - Analysis results (JSON)
    - Execution plans (JSON)
    - Step outputs (media files)
    """
    import hashlib
    import shutil
    import tempfile

    from .analysis import Analyzer, AnalysisResult
    from .planning import RecipePlanner, Recipe, ExecutionPlan
    from .executor import get_executor
    from .dag import NodeType
    from . import nodes  # Register built-in executors

    # Check for ipfs_client
    try:
        from art_celery import ipfs_client
    except ImportError:
        # Try relative import for when running from art-celery
        try:
            import ipfs_client
        except ImportError:
            print("Error: ipfs_client not available. Install art-celery or run from art-celery directory.")
            sys.exit(1)

    # Check IPFS availability
    if not ipfs_client.is_available():
        print("Error: IPFS daemon not available. Start IPFS with 'ipfs daemon'")
        sys.exit(1)

    print("=== IPFS-Primary Mode ===")
    print(f"IPFS Node: {ipfs_client.get_node_id()[:16]}...")

    # Load recipe
    recipe_path = Path(args.recipe)
    recipe = Recipe.from_file(recipe_path)
    print(f"\nRecipe: {recipe.name} v{recipe.version}")

    # Parse inputs
    input_hashes, input_paths = parse_inputs(args.input)

    # Parse features
    features = args.features.split(",") if args.features else ["beats", "energy"]

    # Phase 0: Register on IPFS
    print("\n=== Phase 0: Register on IPFS ===")

    # Register recipe
    recipe_bytes = recipe_path.read_bytes()
    recipe_cid = ipfs_client.add_bytes(recipe_bytes)
    print(f"Recipe CID: {recipe_cid}")

    # Register inputs
    input_cids = {}
    for name, hash_value in input_hashes.items():
        path = input_paths.get(name)
        if path:
            cid = ipfs_client.add_file(Path(path))
            if cid:
                input_cids[name] = cid
                print(f"Input '{name}': {cid}")
            else:
                print(f"Error: Failed to add input '{name}' to IPFS")
                sys.exit(1)

    # Phase 1: Analyze
    print("\n=== Phase 1: Analysis ===")

    # Create temp dir for analysis
    work_dir = Path(tempfile.mkdtemp(prefix="artdag_ipfs_"))
    analysis_cids = {}
    analysis = {}

    try:
        for name, hash_value in input_hashes.items():
            input_cid = input_cids.get(name)
            if not input_cid:
                continue

            print(f"Analyzing {name}...")

            # Fetch from IPFS to temp
            temp_input = work_dir / f"input_{name}.mkv"
            if not ipfs_client.get_file(input_cid, temp_input):
                print(f"  Error: Failed to fetch from IPFS")
                continue

            # Run analysis
            analyzer = Analyzer(cache_dir=None)
            result = analyzer.analyze(
                input_hash=hash_value,
                features=features,
                input_path=temp_input,
            )

            if result.audio and result.audio.beats:
                print(f"  Tempo: {result.audio.beats.tempo:.1f} BPM, {len(result.audio.beats.beat_times)} beats")

            # Store analysis on IPFS
            analysis_cid = ipfs_client.add_json(result.to_dict())
            if analysis_cid:
                analysis_cids[hash_value] = analysis_cid
                analysis[hash_value] = result
                print(f"  Analysis CID: {analysis_cid}")

        # Phase 2: Plan
        print("\n=== Phase 2: Planning ===")

        planner = RecipePlanner(use_tree_reduction=True)
        plan = planner.plan(
            recipe=recipe,
            input_hashes=input_hashes,
            analysis=analysis if analysis else None,
        )

        # Store plan on IPFS
        import json
        plan_dict = json.loads(plan.to_json())
        plan_cid = ipfs_client.add_json(plan_dict)
        print(f"Plan ID: {plan.plan_id[:16]}...")
        print(f"Plan CID: {plan_cid}")
        print(f"Steps: {len(plan.steps)}")

        steps_by_level = plan.get_steps_by_level()
        print(f"Levels: {len(steps_by_level)}")

        # Phase 3: Execute
        print("\n=== Phase 3: Execution ===")

        # CID results
        cid_results = dict(input_cids)
        step_cids = {}

        executed = 0
        cached = 0

        for level in sorted(steps_by_level.keys()):
            steps = steps_by_level[level]
            print(f"\nLevel {level}: {len(steps)} steps")

            for step in steps:
                # Handle SOURCE
                if step.node_type == "SOURCE":
                    source_name = step.config.get("name") or step.step_id
                    cid = cid_results.get(source_name)
                    if cid:
                        step_cids[step.step_id] = cid
                        print(f"  [SOURCE] {step.step_id}")
                    continue

                print(f"  [RUNNING] {step.step_id}: {step.node_type}...")

                try:
                    node_type = NodeType[step.node_type]
                except KeyError:
                    node_type = step.node_type

                executor = get_executor(node_type)
                if executor is None:
                    print(f"    SKIP: No executor for {step.node_type}")
                    continue

                # Fetch inputs from IPFS
                input_paths_list = []
                for i, input_step_id in enumerate(step.input_steps):
                    input_cid = step_cids.get(input_step_id) or cid_results.get(input_step_id)
                    if not input_cid:
                        print(f"    ERROR: Missing input CID for {input_step_id}")
                        continue

                    temp_path = work_dir / f"step_{step.step_id}_input_{i}.mkv"
                    if not ipfs_client.get_file(input_cid, temp_path):
                        print(f"    ERROR: Failed to fetch {input_cid}")
                        continue
                    input_paths_list.append(temp_path)

                if len(input_paths_list) != len(step.input_steps):
                    print(f"    ERROR: Missing inputs")
                    continue

                # Execute
                output_path = work_dir / f"step_{step.step_id}_output.mkv"
                try:
                    result_path = executor.execute(step.config, input_paths_list, output_path)

                    # Add to IPFS
                    output_cid = ipfs_client.add_file(result_path)
                    if output_cid:
                        step_cids[step.step_id] = output_cid
                        print(f"    [DONE] CID: {output_cid}")
                        executed += 1
                    else:
                        print(f"    [FAILED] Could not add to IPFS")
                except Exception as e:
                    print(f"    [FAILED] {e}")

        # Final output
        output_step = plan.get_step(plan.output_step)
        output_cid = step_cids.get(output_step.step_id) if output_step else None

        print(f"\n=== Complete ===")
        print(f"Executed: {executed}")
        if output_cid:
            print(f"Output CID: {output_cid}")
            print(f"Fetch with: ipfs get {output_cid}")

        # Summary of all CIDs
        print(f"\n=== All CIDs ===")
        print(f"Recipe: {recipe_cid}")
        print(f"Plan: {plan_cid}")
        for name, cid in input_cids.items():
            print(f"Input '{name}': {cid}")
        for hash_val, cid in analysis_cids.items():
            print(f"Analysis '{hash_val[:16]}...': {cid}")
        if output_cid:
            print(f"Output: {output_cid}")

    finally:
        # Cleanup temp
        shutil.rmtree(work_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        prog="artdag",
        description="Art DAG - Declarative media composition",
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Extract features from inputs")
    analyze_parser.add_argument("recipe", help="Recipe YAML file")
    analyze_parser.add_argument("-i", "--input", action="append", required=True,
                                help="Input: name:hash[@path]")
    analyze_parser.add_argument("--features", help="Features to extract (comma-separated)")
    analyze_parser.add_argument("-o", "--output", help="Output file (default: analysis.json)")
    analyze_parser.add_argument("--cache-dir", help="Analysis cache directory")

    # plan command
    plan_parser = subparsers.add_parser("plan", help="Generate execution plan")
    plan_parser.add_argument("recipe", help="Recipe YAML file")
    plan_parser.add_argument("-i", "--input", action="append", required=True,
                             help="Input: name:hash")
    plan_parser.add_argument("--analysis", help="Analysis JSON file")
    plan_parser.add_argument("-o", "--output", help="Output file (default: plan.json)")
    plan_parser.add_argument("--no-tree-reduction", action="store_true",
                             help="Disable tree reduction optimization")

    # execute command
    execute_parser = subparsers.add_parser("execute", help="Execute a plan")
    execute_parser.add_argument("plan", help="Plan JSON file")
    execute_parser.add_argument("--dry-run", action="store_true",
                                help="Show what would execute")
    execute_parser.add_argument("--cache-dir", help="Cache directory")

    # run-recipe command
    run_parser = subparsers.add_parser("run-recipe", help="Full pipeline: analyze → plan → execute")
    run_parser.add_argument("recipe", help="Recipe YAML file")
    run_parser.add_argument("-i", "--input", action="append", required=True,
                            help="Input: name:hash[@path]")
    run_parser.add_argument("--features", help="Features to extract (comma-separated)")
    run_parser.add_argument("--cache-dir", help="Cache directory")
    run_parser.add_argument("--ipfs-primary", action="store_true",
                            help="Use IPFS-primary mode (everything on IPFS, no local cache)")

    args = parser.parse_args()

    if args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "plan":
        cmd_plan(args)
    elif args.command == "execute":
        cmd_execute(args)
    elif args.command == "run-recipe":
        if getattr(args, 'ipfs_primary', False):
            cmd_run_recipe_ipfs(args)
        else:
            cmd_run_recipe(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
