#!/usr/bin/env python3
"""
Test the planning phase locally.

This tests the new human-readable names and multi-output support
without requiring actual video files or execution.
"""

import hashlib
import json
import sys
from pathlib import Path

# Add artdag to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from artdag.planning import RecipePlanner, Recipe, ExecutionPlan


def main():
    # Load recipe
    recipe_path = Path(__file__).parent / "simple_sequence.yaml"
    if not recipe_path.exists():
        print(f"Recipe not found: {recipe_path}")
        return 1

    recipe = Recipe.from_file(recipe_path)
    print(f"Recipe: {recipe.name} v{recipe.version}")
    print(f"Nodes: {len(recipe.nodes)}")
    print()

    # Fake input hash (would be real content hash in production)
    fake_input_hash = hashlib.sha3_256(b"fake video content").hexdigest()
    input_hashes = {"video": fake_input_hash}

    print(f"Input: video -> {fake_input_hash[:16]}...")
    print()

    # Generate plan
    planner = RecipePlanner(use_tree_reduction=True)
    plan = planner.plan(
        recipe=recipe,
        input_hashes=input_hashes,
        seed=42,  # Optional seed for reproducibility
    )

    print("=== Generated Plan ===")
    print(f"Plan ID: {plan.plan_id[:24]}...")
    print(f"Plan Name: {plan.name}")
    print(f"Recipe Name: {plan.recipe_name}")
    print(f"Output: {plan.output_name}")
    print(f"Steps: {len(plan.steps)}")
    print()

    # Show steps by level
    steps_by_level = plan.get_steps_by_level()
    for level in sorted(steps_by_level.keys()):
        steps = steps_by_level[level]
        print(f"Level {level}: {len(steps)} step(s)")
        for step in steps:
            # Show human-readable name
            name = step.name or step.step_id[:20]
            print(f"  - {name}")
            print(f"    Type: {step.node_type}")
            print(f"    Cache ID: {step.cache_id[:16]}...")
            if step.outputs:
                print(f"    Outputs: {len(step.outputs)}")
                for out in step.outputs:
                    print(f"      - {out.name} ({out.media_type})")
            if step.inputs:
                print(f"    Inputs: {[inp.name for inp in step.inputs]}")
        print()

    # Save plan for inspection
    plan_path = Path(__file__).parent.parent / "test_plan_output.json"
    with open(plan_path, "w") as f:
        f.write(plan.to_json())
    print(f"Plan saved to: {plan_path}")

    # Show plan JSON structure
    print()
    print("=== Plan JSON Preview ===")
    plan_dict = json.loads(plan.to_json())
    # Show first step as example
    if plan_dict.get("steps"):
        first_step = plan_dict["steps"][0]
        print(json.dumps(first_step, indent=2)[:500] + "...")

    return 0


if __name__ == "__main__":
    sys.exit(main())
