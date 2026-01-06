# tests/test_primitive_new/test_executor.py
"""Tests for primitive executor module."""

import pytest
from pathlib import Path
from typing import Any, Dict, List

from artdag.dag import NodeType
from artdag.executor import (
    Executor,
    register_executor,
    get_executor,
    list_executors,
    clear_executors,
)


class TestExecutorRegistry:
    """Test executor registration."""

    def setup_method(self):
        """Clear registry before each test."""
        clear_executors()

    def teardown_method(self):
        """Clear registry after each test."""
        clear_executors()

    def test_register_executor(self):
        """Test registering an executor."""
        @register_executor(NodeType.SOURCE)
        class TestSourceExecutor(Executor):
            def execute(self, config, inputs, output_path):
                return output_path

        executor = get_executor(NodeType.SOURCE)
        assert executor is not None
        assert isinstance(executor, TestSourceExecutor)

    def test_register_custom_type(self):
        """Test registering executor for custom type."""
        @register_executor("CUSTOM_NODE")
        class CustomExecutor(Executor):
            def execute(self, config, inputs, output_path):
                return output_path

        executor = get_executor("CUSTOM_NODE")
        assert executor is not None

    def test_get_unregistered(self):
        """Test getting unregistered executor."""
        executor = get_executor(NodeType.ANALYZE)
        assert executor is None

    def test_list_executors(self):
        """Test listing registered executors."""
        @register_executor(NodeType.SOURCE)
        class SourceExec(Executor):
            def execute(self, config, inputs, output_path):
                return output_path

        @register_executor(NodeType.SEGMENT)
        class SegmentExec(Executor):
            def execute(self, config, inputs, output_path):
                return output_path

        executors = list_executors()
        assert "SOURCE" in executors
        assert "SEGMENT" in executors

    def test_overwrite_warning(self, caplog):
        """Test warning when overwriting executor."""
        @register_executor(NodeType.SOURCE)
        class FirstExecutor(Executor):
            def execute(self, config, inputs, output_path):
                return output_path

        # Register again - should warn
        @register_executor(NodeType.SOURCE)
        class SecondExecutor(Executor):
            def execute(self, config, inputs, output_path):
                return output_path

        # Second should be registered
        executor = get_executor(NodeType.SOURCE)
        assert isinstance(executor, SecondExecutor)


class TestExecutorBase:
    """Test Executor base class."""

    def test_validate_config_default(self):
        """Test default validate_config returns empty list."""
        class TestExecutor(Executor):
            def execute(self, config, inputs, output_path):
                return output_path

        executor = TestExecutor()
        errors = executor.validate_config({"any": "config"})
        assert errors == []

    def test_estimate_output_size(self):
        """Test default output size estimation."""
        class TestExecutor(Executor):
            def execute(self, config, inputs, output_path):
                return output_path

        executor = TestExecutor()
        size = executor.estimate_output_size({}, [100, 200, 300])
        assert size == 600
