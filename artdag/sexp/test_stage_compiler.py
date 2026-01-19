"""
Tests for stage compilation and scoping.

Tests the CompiledStage dataclass, stage form parsing,
variable scoping, and dependency validation.
"""

import pytest

from .parser import parse, Symbol, Keyword
from .compiler import (
    compile_recipe,
    CompileError,
    CompiledStage,
    CompilerContext,
    _topological_sort_stages,
)


class TestStageCompilation:
    """Test stage form compilation."""

    def test_parse_stage_form_basic(self):
        """Stage parses correctly with name and outputs."""
        recipe = '''
        (recipe "test-stage"
          (def audio (source :path "test.mp3"))

          (stage :analyze
            :outputs [beats]
            (def beats (-> audio (analyze beats)))
            (-> audio (segment :times beats) (sequence))))
        '''
        compiled = compile_recipe(parse(recipe))

        assert len(compiled.stages) == 1
        assert compiled.stages[0].name == "analyze"
        assert "beats" in compiled.stages[0].outputs
        assert len(compiled.stages[0].node_ids) > 0

    def test_parse_stage_with_requires(self):
        """Stage parses correctly with requires and inputs."""
        recipe = '''
        (recipe "test-requires"
          (def audio (source :path "test.mp3"))

          (stage :analyze
            :outputs [beats]
            (def beats (-> audio (analyze beats))))

          (stage :process
            :requires [:analyze]
            :inputs [beats]
            :outputs [segments]
            (def segments (-> audio (segment :times beats)))
            (-> segments (sequence))))
        '''
        compiled = compile_recipe(parse(recipe))

        assert len(compiled.stages) == 2
        process_stage = next(s for s in compiled.stages if s.name == "process")
        assert process_stage.requires == ["analyze"]
        assert "beats" in process_stage.inputs
        assert "segments" in process_stage.outputs

    def test_stage_outputs_recorded(self):
        """Stage outputs are tracked in CompiledStage."""
        recipe = '''
        (recipe "test-outputs"
          (def audio (source :path "test.mp3"))

          (stage :analyze
            :outputs [beats tempo]
            (def beats (-> audio (analyze beats)))
            (def tempo (-> audio (analyze tempo)))
            (-> audio (segment :times beats) (sequence))))
        '''
        compiled = compile_recipe(parse(recipe))

        stage = compiled.stages[0]
        assert "beats" in stage.outputs
        assert "tempo" in stage.outputs
        assert "beats" in stage.output_bindings
        assert "tempo" in stage.output_bindings

    def test_stage_order_topological(self):
        """Stages are topologically sorted."""
        recipe = '''
        (recipe "test-order"
          (def audio (source :path "test.mp3"))

          (stage :analyze
            :outputs [beats]
            (def beats (-> audio (analyze beats))))

          (stage :output
            :requires [:analyze]
            :inputs [beats]
            (-> audio (segment :times beats) (sequence))))
        '''
        compiled = compile_recipe(parse(recipe))

        # analyze should come before output
        assert compiled.stage_order.index("analyze") < compiled.stage_order.index("output")


class TestStageValidation:
    """Test stage dependency and input validation."""

    def test_stage_requires_validation(self):
        """Error if requiring non-existent stage."""
        recipe = '''
        (recipe "test-bad-require"
          (def audio (source :path "test.mp3"))

          (stage :process
            :requires [:nonexistent]
            :inputs [beats]
            (def result audio)))
        '''
        with pytest.raises(CompileError, match="requires undefined stage"):
            compile_recipe(parse(recipe))

    def test_stage_inputs_validation(self):
        """Error if input not produced by required stage."""
        recipe = '''
        (recipe "test-bad-input"
          (def audio (source :path "test.mp3"))

          (stage :analyze
            :outputs [beats]
            (def beats (-> audio (analyze beats))))

          (stage :process
            :requires [:analyze]
            :inputs [nonexistent]
            (def result audio)))
        '''
        with pytest.raises(CompileError, match="not an output of any required stage"):
            compile_recipe(parse(recipe))

    def test_undeclared_output_error(self):
        """Error if stage declares output not defined in body."""
        recipe = '''
        (recipe "test-missing-output"
          (def audio (source :path "test.mp3"))

          (stage :analyze
            :outputs [beats nonexistent]
            (def beats (-> audio (analyze beats)))))
        '''
        with pytest.raises(CompileError, match="not defined in the stage body"):
            compile_recipe(parse(recipe))

    def test_forward_reference_detection(self):
        """Error when requiring a stage not yet defined."""
        # Forward references are not allowed - stages must be defined
        # before they can be required
        recipe = '''
        (recipe "test-forward"
          (def audio (source :path "test.mp3"))

          (stage :a
            :requires [:b]
            :outputs [out-a]
            (def out-a audio))

          (stage :b
            :outputs [out-b]
            (def out-b audio)
            audio))
        '''
        with pytest.raises(CompileError, match="requires undefined stage"):
            compile_recipe(parse(recipe))


class TestStageScoping:
    """Test variable scoping between stages."""

    def test_pre_stage_bindings_accessible(self):
        """Sources defined before stages accessible to all stages."""
        recipe = '''
        (recipe "test-pre-stage"
          (def audio (source :path "test.mp3"))
          (def video (source :path "test.mp4"))

          (stage :analyze-audio
            :outputs [beats]
            (def beats (-> audio (analyze beats))))

          (stage :analyze-video
            :outputs [scenes]
            (def scenes (-> video (analyze scenes)))
            (-> video (segment :times scenes) (sequence))))
        '''
        # Should compile without error - audio and video accessible to both stages
        compiled = compile_recipe(parse(recipe))
        assert len(compiled.stages) == 2

    def test_stage_bindings_flow_through_requires(self):
        """Stage bindings accessible to dependent stages via :inputs."""
        recipe = '''
        (recipe "test-binding-flow"
          (def audio (source :path "test.mp3"))

          (stage :analyze
            :outputs [beats]
            (def beats (-> audio (analyze beats))))

          (stage :process
            :requires [:analyze]
            :inputs [beats]
            :outputs [result]
            (def result (-> audio (segment :times beats)))
            (-> result (sequence))))
        '''
        # Should compile without error - beats flows from analyze to process
        compiled = compile_recipe(parse(recipe))
        assert len(compiled.stages) == 2


class TestTopologicalSort:
    """Test stage topological sorting."""

    def test_empty_stages(self):
        """Empty stages returns empty list."""
        assert _topological_sort_stages({}) == []

    def test_single_stage(self):
        """Single stage returns single element."""
        stages = {
            "a": CompiledStage(
                name="a",
                requires=[],
                inputs=[],
                outputs=["out"],
                node_ids=["n1"],
                output_bindings={"out": "n1"},
            )
        }
        assert _topological_sort_stages(stages) == ["a"]

    def test_linear_chain(self):
        """Linear chain sorted correctly."""
        stages = {
            "a": CompiledStage(name="a", requires=[], inputs=[], outputs=["x"],
                              node_ids=["n1"], output_bindings={"x": "n1"}),
            "b": CompiledStage(name="b", requires=["a"], inputs=["x"], outputs=["y"],
                              node_ids=["n2"], output_bindings={"y": "n2"}),
            "c": CompiledStage(name="c", requires=["b"], inputs=["y"], outputs=["z"],
                              node_ids=["n3"], output_bindings={"z": "n3"}),
        }
        result = _topological_sort_stages(stages)
        assert result.index("a") < result.index("b") < result.index("c")

    def test_parallel_stages_same_level(self):
        """Parallel stages are both valid orderings."""
        stages = {
            "a": CompiledStage(name="a", requires=[], inputs=[], outputs=["x"],
                              node_ids=["n1"], output_bindings={"x": "n1"}),
            "b": CompiledStage(name="b", requires=[], inputs=[], outputs=["y"],
                              node_ids=["n2"], output_bindings={"y": "n2"}),
        }
        result = _topological_sort_stages(stages)
        # Both a and b should be in result (order doesn't matter)
        assert set(result) == {"a", "b"}

    def test_diamond_dependency(self):
        """Diamond pattern: A -> B, A -> C, B+C -> D."""
        stages = {
            "a": CompiledStage(name="a", requires=[], inputs=[], outputs=["x"],
                              node_ids=["n1"], output_bindings={"x": "n1"}),
            "b": CompiledStage(name="b", requires=["a"], inputs=["x"], outputs=["y"],
                              node_ids=["n2"], output_bindings={"y": "n2"}),
            "c": CompiledStage(name="c", requires=["a"], inputs=["x"], outputs=["z"],
                              node_ids=["n3"], output_bindings={"z": "n3"}),
            "d": CompiledStage(name="d", requires=["b", "c"], inputs=["y", "z"], outputs=["out"],
                              node_ids=["n4"], output_bindings={"out": "n4"}),
        }
        result = _topological_sort_stages(stages)
        # a must be first, d must be last
        assert result[0] == "a"
        assert result[-1] == "d"
        # b and c must be before d
        assert result.index("b") < result.index("d")
        assert result.index("c") < result.index("d")
