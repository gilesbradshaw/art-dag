"""
End-to-end integration tests for staged recipes.

Tests the complete flow: compile -> plan -> execute
for recipes with stages.
"""

import pytest
import tempfile
from pathlib import Path

from .parser import parse, serialize
from .compiler import compile_recipe, CompileError
from .planner import ExecutionPlanSexp, StagePlan
from .stage_cache import StageCache, StageCacheEntry, StageOutput
from .scheduler import StagePlanScheduler, StagePlanResult


class TestSimpleTwoStageRecipe:
    """Test basic two-stage recipe flow."""

    def test_compile_two_stage_recipe(self):
        """Compile a simple two-stage recipe."""
        recipe = '''
        (recipe "test-two-stages"
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

        assert len(compiled.stages) == 2
        assert compiled.stage_order == ["analyze", "output"]

        analyze_stage = compiled.stages[0]
        assert analyze_stage.name == "analyze"
        assert "beats" in analyze_stage.outputs

        output_stage = compiled.stages[1]
        assert output_stage.name == "output"
        assert output_stage.requires == ["analyze"]
        assert "beats" in output_stage.inputs


class TestParallelAnalysisStages:
    """Test parallel analysis stages."""

    def test_compile_parallel_stages(self):
        """Two analysis stages can run in parallel."""
        recipe = '''
        (recipe "test-parallel"
          (def audio-a (source :path "a.mp3"))
          (def audio-b (source :path "b.mp3"))

          (stage :analyze-a
            :outputs [beats-a]
            (def beats-a (-> audio-a (analyze beats))))

          (stage :analyze-b
            :outputs [beats-b]
            (def beats-b (-> audio-b (analyze beats))))

          (stage :combine
            :requires [:analyze-a :analyze-b]
            :inputs [beats-a beats-b]
            (-> audio-a (segment :times beats-a) (sequence))))
        '''
        compiled = compile_recipe(parse(recipe))

        assert len(compiled.stages) == 3

        # analyze-a and analyze-b should both be at level 0 (parallel)
        analyze_a = next(s for s in compiled.stages if s.name == "analyze-a")
        analyze_b = next(s for s in compiled.stages if s.name == "analyze-b")
        combine = next(s for s in compiled.stages if s.name == "combine")

        assert analyze_a.requires == []
        assert analyze_b.requires == []
        assert set(combine.requires) == {"analyze-a", "analyze-b"}


class TestDiamondDependency:
    """Test diamond dependency pattern: A -> B, A -> C, B+C -> D."""

    def test_compile_diamond_pattern(self):
        """Diamond pattern compiles correctly."""
        recipe = '''
        (recipe "test-diamond"
          (def audio (source :path "test.mp3"))

          (stage :source-stage
            :outputs [audio-ref]
            (def audio-ref audio))

          (stage :branch-b
            :requires [:source-stage]
            :inputs [audio-ref]
            :outputs [result-b]
            (def result-b (-> audio-ref (effect gain :amount 0.5))))

          (stage :branch-c
            :requires [:source-stage]
            :inputs [audio-ref]
            :outputs [result-c]
            (def result-c (-> audio-ref (effect gain :amount 0.8))))

          (stage :merge
            :requires [:branch-b :branch-c]
            :inputs [result-b result-c]
            (-> result-b (blend result-c :mode "mix"))))
        '''
        compiled = compile_recipe(parse(recipe))

        assert len(compiled.stages) == 4

        # Check dependencies
        source = next(s for s in compiled.stages if s.name == "source-stage")
        branch_b = next(s for s in compiled.stages if s.name == "branch-b")
        branch_c = next(s for s in compiled.stages if s.name == "branch-c")
        merge = next(s for s in compiled.stages if s.name == "merge")

        assert source.requires == []
        assert branch_b.requires == ["source-stage"]
        assert branch_c.requires == ["source-stage"]
        assert set(merge.requires) == {"branch-b", "branch-c"}

        # source-stage should come first in order
        assert compiled.stage_order.index("source-stage") < compiled.stage_order.index("branch-b")
        assert compiled.stage_order.index("source-stage") < compiled.stage_order.index("branch-c")
        # merge should come last
        assert compiled.stage_order.index("branch-b") < compiled.stage_order.index("merge")
        assert compiled.stage_order.index("branch-c") < compiled.stage_order.index("merge")


class TestStageReuseOnRerun:
    """Test that re-running recipe uses cached stages."""

    def test_stage_reuse(self):
        """Re-running recipe uses cached stages."""
        with tempfile.TemporaryDirectory() as tmpdir:
            stage_cache = StageCache(tmpdir)

            # Simulate first run by caching a stage
            entry = StageCacheEntry(
                stage_name="analyze",
                cache_id="fixed_cache_id",
                outputs={"beats": StageOutput(cache_id="beats_out", output_type="analysis")},
            )
            stage_cache.save_stage(entry)

            # Verify cache exists
            assert stage_cache.has_stage("fixed_cache_id")

            # Second run should find cache
            loaded = stage_cache.load_stage("fixed_cache_id")
            assert loaded is not None
            assert loaded.stage_name == "analyze"


class TestExplicitDataFlowEndToEnd:
    """Test that analysis results flow through :inputs/:outputs."""

    def test_data_flow_declaration(self):
        """Explicit data flow is declared correctly."""
        recipe = '''
        (recipe "test-data-flow"
          (def audio (source :path "test.mp3"))

          (stage :analyze
            :outputs [beats tempo]
            (def beats (-> audio (analyze beats)))
            (def tempo (-> audio (analyze tempo))))

          (stage :process
            :requires [:analyze]
            :inputs [beats tempo]
            :outputs [result]
            (def result (-> audio (segment :times beats) (effect speed :factor tempo)))
            (-> result (sequence))))
        '''
        compiled = compile_recipe(parse(recipe))

        analyze = next(s for s in compiled.stages if s.name == "analyze")
        process = next(s for s in compiled.stages if s.name == "process")

        # Analyze outputs
        assert set(analyze.outputs) == {"beats", "tempo"}
        assert "beats" in analyze.output_bindings
        assert "tempo" in analyze.output_bindings

        # Process inputs
        assert set(process.inputs) == {"beats", "tempo"}
        assert process.requires == ["analyze"]


class TestRecipeFixtures:
    """Test using recipe fixtures."""

    @pytest.fixture
    def test_recipe_two_stages(self):
        return '''
        (recipe "test-two-stages"
          (def audio (source :path "test.mp3"))

          (stage :analyze
            :outputs [beats]
            (def beats (-> audio (analyze beats))))

          (stage :output
            :requires [:analyze]
            :inputs [beats]
            (-> audio (segment :times beats) (sequence))))
        '''

    @pytest.fixture
    def test_recipe_parallel_stages(self):
        return '''
        (recipe "test-parallel"
          (def audio-a (source :path "a.mp3"))
          (def audio-b (source :path "b.mp3"))

          (stage :analyze-a
            :outputs [beats-a]
            (def beats-a (-> audio-a (analyze beats))))

          (stage :analyze-b
            :outputs [beats-b]
            (def beats-b (-> audio-b (analyze beats))))

          (stage :combine
            :requires [:analyze-a :analyze-b]
            :inputs [beats-a beats-b]
            (-> audio-a (blend audio-b :mode "mix"))))
        '''

    def test_two_stages_fixture(self, test_recipe_two_stages):
        """Two-stage recipe fixture compiles."""
        compiled = compile_recipe(parse(test_recipe_two_stages))
        assert len(compiled.stages) == 2

    def test_parallel_stages_fixture(self, test_recipe_parallel_stages):
        """Parallel stages recipe fixture compiles."""
        compiled = compile_recipe(parse(test_recipe_parallel_stages))
        assert len(compiled.stages) == 3


class TestStageValidationErrors:
    """Test error handling for invalid stage recipes."""

    def test_missing_output_declaration(self):
        """Error when stage output not declared."""
        recipe = '''
        (recipe "test-missing-output"
          (def audio (source :path "test.mp3"))

          (stage :analyze
            :outputs [beats nonexistent]
            (def beats (-> audio (analyze beats)))))
        '''
        with pytest.raises(CompileError, match="not defined in the stage body"):
            compile_recipe(parse(recipe))

    def test_input_without_requires(self):
        """Error when using input not from required stage."""
        recipe = '''
        (recipe "test-bad-input"
          (def audio (source :path "test.mp3"))

          (stage :analyze
            :outputs [beats]
            (def beats (-> audio (analyze beats))))

          (stage :process
            :requires []
            :inputs [beats]
            (def result audio)))
        '''
        with pytest.raises(CompileError, match="not an output of any required stage"):
            compile_recipe(parse(recipe))

    def test_forward_reference(self):
        """Error when requiring stage not yet defined (forward reference)."""
        recipe = '''
        (recipe "test-forward-ref"
          (def audio (source :path "test.mp3"))

          (stage :a
            :requires [:b]
            :outputs [out-a]
            (def out-a audio)
            audio)

          (stage :b
            :outputs [out-b]
            (def out-b audio)
            audio))
        '''
        with pytest.raises(CompileError, match="requires undefined stage"):
            compile_recipe(parse(recipe))


class TestBeatSyncDemoRecipe:
    """Test the beat-sync demo recipe from examples."""

    BEAT_SYNC_RECIPE = '''
    ;; Simple staged recipe demo
    (recipe "beat-sync-demo"
      :version "1.0"
      :description "Demo of staged beat-sync workflow"

      ;; Pre-stage definitions (available to all stages)
      (def audio (source :path "input.mp3"))

      ;; Stage 1: Analysis (expensive, cached)
      (stage :analyze
        :outputs [beats tempo]
        (def beats (-> audio (analyze beats)))
        (def tempo (-> audio (analyze tempo))))

      ;; Stage 2: Processing (uses analysis results)
      (stage :process
        :requires [:analyze]
        :inputs [beats]
        :outputs [segments]
        (def segments (-> audio (segment :times beats)))
        (-> segments (sequence))))
    '''

    def test_compile_beat_sync_recipe(self):
        """Beat-sync demo recipe compiles correctly."""
        compiled = compile_recipe(parse(self.BEAT_SYNC_RECIPE))

        assert compiled.name == "beat-sync-demo"
        assert compiled.version == "1.0"
        assert compiled.description == "Demo of staged beat-sync workflow"

    def test_beat_sync_stage_count(self):
        """Beat-sync has 2 stages in correct order."""
        compiled = compile_recipe(parse(self.BEAT_SYNC_RECIPE))

        assert len(compiled.stages) == 2
        assert compiled.stage_order == ["analyze", "process"]

    def test_beat_sync_analyze_stage(self):
        """Analyze stage has correct outputs."""
        compiled = compile_recipe(parse(self.BEAT_SYNC_RECIPE))

        analyze = next(s for s in compiled.stages if s.name == "analyze")
        assert analyze.requires == []
        assert analyze.inputs == []
        assert set(analyze.outputs) == {"beats", "tempo"}
        assert "beats" in analyze.output_bindings
        assert "tempo" in analyze.output_bindings

    def test_beat_sync_process_stage(self):
        """Process stage has correct dependencies and inputs."""
        compiled = compile_recipe(parse(self.BEAT_SYNC_RECIPE))

        process = next(s for s in compiled.stages if s.name == "process")
        assert process.requires == ["analyze"]
        assert "beats" in process.inputs
        assert "segments" in process.outputs

    def test_beat_sync_node_count(self):
        """Beat-sync generates expected number of nodes."""
        compiled = compile_recipe(parse(self.BEAT_SYNC_RECIPE))

        # 1 SOURCE + 2 ANALYZE + 1 SEGMENT + 1 SEQUENCE = 5 nodes
        assert len(compiled.nodes) == 5

    def test_beat_sync_node_types(self):
        """Beat-sync generates correct node types."""
        compiled = compile_recipe(parse(self.BEAT_SYNC_RECIPE))

        node_types = [n["type"] for n in compiled.nodes]
        assert node_types.count("SOURCE") == 1
        assert node_types.count("ANALYZE") == 2
        assert node_types.count("SEGMENT") == 1
        assert node_types.count("SEQUENCE") == 1

    def test_beat_sync_output_is_sequence(self):
        """Beat-sync output node is the sequence node."""
        compiled = compile_recipe(parse(self.BEAT_SYNC_RECIPE))

        output_node = next(n for n in compiled.nodes if n["id"] == compiled.output_node_id)
        assert output_node["type"] == "SEQUENCE"


class TestAsciiArtStagedRecipe:
    """Test the ASCII art staged recipe."""

    ASCII_ART_STAGED_RECIPE = '''
    ;; ASCII art effect with staged execution
    (recipe "ascii_art_staged"
      :version "1.0"
      :description "ASCII art effect with staged execution"
      :encoding (:codec "libx264" :crf 20 :preset "medium" :audio-codec "aac" :fps 30)

      ;; Registry
      (effect ascii_art :path "sexp_effects/effects/ascii_art.sexp")
      (analyzer energy :path "../artdag-analyzers/energy/analyzer.py")

      ;; Pre-stage definitions
      (def color_mode "color")
      (def video (source :path "monday.webm"))
      (def audio (source :path "dizzy.mp3"))

      ;; Stage 1: Analysis
      (stage :analyze
        :outputs [energy-data]
        (def audio-clip (-> audio (segment :start 60 :duration 10)))
        (def energy-data (-> audio-clip (analyze energy))))

      ;; Stage 2: Process
      (stage :process
        :requires [:analyze]
        :inputs [energy-data]
        :outputs [result audio-clip]
        (def clip (-> video (segment :start 0 :duration 10)))
        (def audio-clip (-> audio (segment :start 60 :duration 10)))
        (def result (-> clip
          (effect ascii_art
            :char_size (bind energy-data values :range [2 32])
            :color_mode color_mode))))

      ;; Stage 3: Output
      (stage :output
        :requires [:process]
        :inputs [result audio-clip]
        (mux result audio-clip)))
    '''

    def test_compile_ascii_art_staged(self):
        """ASCII art staged recipe compiles correctly."""
        compiled = compile_recipe(parse(self.ASCII_ART_STAGED_RECIPE))

        assert compiled.name == "ascii_art_staged"
        assert compiled.version == "1.0"

    def test_ascii_art_stage_count(self):
        """ASCII art has 3 stages in correct order."""
        compiled = compile_recipe(parse(self.ASCII_ART_STAGED_RECIPE))

        assert len(compiled.stages) == 3
        assert compiled.stage_order == ["analyze", "process", "output"]

    def test_ascii_art_analyze_stage(self):
        """Analyze stage outputs energy-data."""
        compiled = compile_recipe(parse(self.ASCII_ART_STAGED_RECIPE))

        analyze = next(s for s in compiled.stages if s.name == "analyze")
        assert analyze.requires == []
        assert analyze.inputs == []
        assert "energy-data" in analyze.outputs

    def test_ascii_art_process_stage(self):
        """Process stage requires analyze and outputs result."""
        compiled = compile_recipe(parse(self.ASCII_ART_STAGED_RECIPE))

        process = next(s for s in compiled.stages if s.name == "process")
        assert process.requires == ["analyze"]
        assert "energy-data" in process.inputs
        assert "result" in process.outputs
        assert "audio-clip" in process.outputs

    def test_ascii_art_output_stage(self):
        """Output stage requires process and has mux."""
        compiled = compile_recipe(parse(self.ASCII_ART_STAGED_RECIPE))

        output = next(s for s in compiled.stages if s.name == "output")
        assert output.requires == ["process"]
        assert "result" in output.inputs
        assert "audio-clip" in output.inputs

    def test_ascii_art_node_count(self):
        """ASCII art generates expected nodes."""
        compiled = compile_recipe(parse(self.ASCII_ART_STAGED_RECIPE))

        # 2 SOURCE + 2 SEGMENT + 1 ANALYZE + 1 EFFECT + 1 MUX = 7+ nodes
        assert len(compiled.nodes) >= 7

    def test_ascii_art_has_mux_output(self):
        """ASCII art output is MUX node."""
        compiled = compile_recipe(parse(self.ASCII_ART_STAGED_RECIPE))

        output_node = next(n for n in compiled.nodes if n["id"] == compiled.output_node_id)
        assert output_node["type"] == "MUX"


class TestMixedStagedAndNonStagedRecipes:
    """Test that non-staged recipes still work."""

    def test_recipe_without_stages(self):
        """Non-staged recipe compiles normally."""
        recipe = '''
        (recipe "no-stages"
          (-> (source :path "test.mp3")
              (effect gain :amount 0.5)))
        '''
        compiled = compile_recipe(parse(recipe))

        assert compiled.stages == []
        assert compiled.stage_order == []
        # Should still have nodes
        assert len(compiled.nodes) > 0

    def test_mixed_pre_stage_and_stages(self):
        """Pre-stage definitions work with stages."""
        recipe = '''
        (recipe "mixed"
          ;; Pre-stage definitions
          (def audio (source :path "test.mp3"))
          (def volume 0.8)

          ;; Stage using pre-stage definitions, ending with output expression
          (stage :process
            :outputs [result]
            (def result (-> audio (effect gain :amount volume)))
            result))
        '''
        compiled = compile_recipe(parse(recipe))

        assert len(compiled.stages) == 1
        # audio and volume should be accessible in stage
        process = compiled.stages[0]
        assert process.name == "process"
        assert "result" in process.outputs


class TestEffectParamsBlock:
    """Test :params block parsing in effect definitions."""

    def test_parse_effect_with_params_block(self):
        """Parse effect with new :params syntax."""
        from .effect_loader import load_sexp_effect

        effect_code = '''
        (define-effect test_effect
          :params (
            (size :type int :default 10 :range [1 100] :desc "Size parameter")
            (color :type string :default "red" :desc "Color parameter")
            (enabled :type int :default 1 :range [0 1] :desc "Enable flag")
          )
          frame)
        '''
        name, process_fn, defaults, param_defs = load_sexp_effect(effect_code)

        assert name == "test_effect"
        assert len(param_defs) == 3
        assert defaults["size"] == 10
        assert defaults["color"] == "red"
        assert defaults["enabled"] == 1

        # Check ParamDef objects
        size_param = param_defs[0]
        assert size_param.name == "size"
        assert size_param.param_type == "int"
        assert size_param.default == 10
        assert size_param.range_min == 1.0
        assert size_param.range_max == 100.0
        assert size_param.description == "Size parameter"

        color_param = param_defs[1]
        assert color_param.name == "color"
        assert color_param.param_type == "string"
        assert color_param.default == "red"

    def test_parse_effect_with_choices(self):
        """Parse effect with choices in :params."""
        from .effect_loader import load_sexp_effect

        effect_code = '''
        (define-effect mode_effect
          :params (
            (mode :type string :default "fast"
              :choices [fast slow medium]
              :desc "Processing mode")
          )
          frame)
        '''
        name, _, defaults, param_defs = load_sexp_effect(effect_code)

        assert name == "mode_effect"
        assert defaults["mode"] == "fast"

        mode_param = param_defs[0]
        assert mode_param.choices == ["fast", "slow", "medium"]

    def test_legacy_effect_syntax_rejected(self):
        """Legacy effect syntax should be rejected."""
        from .effect_loader import load_sexp_effect
        import pytest

        effect_code = '''
        (define-effect legacy_effect
          ((width 100)
           (height 200)
           (name "default"))
          frame)
        '''
        with pytest.raises(ValueError) as exc_info:
            load_sexp_effect(effect_code)

        assert "Legacy parameter syntax" in str(exc_info.value)
        assert ":params" in str(exc_info.value)

    def test_effect_params_introspection(self):
        """Test that effect params are available for introspection."""
        from .effect_loader import load_sexp_effect_file
        from pathlib import Path

        # Create a temp effect file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sexp', delete=False) as f:
            f.write('''
            (define-effect introspect_test
              :params (
                (alpha :type float :default 0.5 :range [0 1] :desc "Alpha value")
              )
              frame)
            ''')
            temp_path = Path(f.name)

        try:
            name, _, defaults, param_defs = load_sexp_effect_file(temp_path)
            assert name == "introspect_test"
            assert len(param_defs) == 1
            assert param_defs[0].name == "alpha"
            assert param_defs[0].param_type == "float"
        finally:
            temp_path.unlink()


class TestConstructParamsBlock:
    """Test :params block parsing in construct definitions."""

    def test_parse_construct_params_helper(self):
        """Test the _parse_construct_params helper function."""
        from .planner import _parse_construct_params
        from .parser import Symbol, Keyword

        params_list = [
            [Symbol("duration"), Keyword("type"), Symbol("float"),
             Keyword("default"), 5.0, Keyword("desc"), "Duration in seconds"],
            [Symbol("count"), Keyword("type"), Symbol("int"),
             Keyword("default"), 10],
        ]

        param_names, param_defaults = _parse_construct_params(params_list)

        assert param_names == ["duration", "count"]
        assert param_defaults["duration"] == 5.0
        assert param_defaults["count"] == 10

    def test_construct_params_with_no_defaults(self):
        """Test construct params where some have no default."""
        from .planner import _parse_construct_params
        from .parser import Symbol, Keyword

        params_list = [
            [Symbol("required_param"), Keyword("type"), Symbol("string")],
            [Symbol("optional_param"), Keyword("type"), Symbol("int"),
             Keyword("default"), 42],
        ]

        param_names, param_defaults = _parse_construct_params(params_list)

        assert param_names == ["required_param", "optional_param"]
        assert param_defaults["required_param"] is None
        assert param_defaults["optional_param"] == 42


class TestParameterValidation:
    """Test that unknown parameters are rejected."""

    def test_effect_rejects_unknown_params(self):
        """Effects should reject unknown parameters."""
        from .effect_loader import load_sexp_effect
        import numpy as np
        import pytest

        effect_code = '''
        (define-effect test_effect
          :params (
            (brightness :type int :default 0 :desc "Brightness")
          )
          frame)
        '''
        name, process_frame, defaults, _ = load_sexp_effect(effect_code)

        # Create a test frame
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        state = {}

        # Valid param should work
        result, _ = process_frame(frame, {"brightness": 10}, state)
        assert isinstance(result, np.ndarray)

        # Unknown param should raise
        with pytest.raises(ValueError) as exc_info:
            process_frame(frame, {"unknown_param": 42}, state)

        assert "Unknown parameter 'unknown_param'" in str(exc_info.value)
        assert "brightness" in str(exc_info.value)

    def test_effect_no_params_rejects_all(self):
        """Effects with no params should reject any parameter."""
        from .effect_loader import load_sexp_effect
        import numpy as np
        import pytest

        effect_code = '''
        (define-effect no_params_effect
          :params ()
          frame)
        '''
        name, process_frame, defaults, _ = load_sexp_effect(effect_code)

        # Create a test frame
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        state = {}

        # Empty params should work
        result, _ = process_frame(frame, {}, state)
        assert isinstance(result, np.ndarray)

        # Any param should raise
        with pytest.raises(ValueError) as exc_info:
            process_frame(frame, {"any_param": 42}, state)

        assert "Unknown parameter 'any_param'" in str(exc_info.value)
        assert "(none)" in str(exc_info.value)
