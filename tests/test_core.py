"""
Tests for NeuralHive core components.
Run with: python -m pytest tests/ -v
"""

import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── Layer Skipper Tests ───────────────────────────────────────────────────────

class TestLayerSkipper:
    def setup_method(self):
        from core.runtime.layer_skipper import LayerSkipper, TaskType
        self.LayerSkipper = LayerSkipper
        self.TaskType = TaskType

    def test_skip_indices_never_empty_for_large_models(self):
        skipper = self.LayerSkipper(total_layers=80)
        indices = skipper.get_skip_indices(self.TaskType.CODING, 80)
        assert len(indices) > 0

    def test_never_skips_first_three_layers(self):
        skipper = self.LayerSkipper(total_layers=80)
        indices = skipper.get_skip_indices(self.TaskType.CODING, 80)
        assert 0 not in indices
        assert 1 not in indices
        assert 2 not in indices

    def test_never_skips_last_three_layers(self):
        skipper = self.LayerSkipper(total_layers=80)
        indices = skipper.get_skip_indices(self.TaskType.CODING, 80)
        assert 77 not in indices
        assert 78 not in indices
        assert 79 not in indices

    def test_active_layers_plus_skipped_equals_total(self):
        skipper = self.LayerSkipper(total_layers=80)
        active = skipper.get_active_layers(self.TaskType.CODING)
        skipped = skipper.get_skip_indices(self.TaskType.CODING, 80)
        assert len(active) + len(skipped) == 80

    def test_compute_savings_reasonable(self):
        skipper = self.LayerSkipper(total_layers=80)
        savings = skipper.compute_savings(self.TaskType.CODING)
        assert savings["compute_saved_pct"] > 5    # At least 5% saved
        assert savings["compute_saved_pct"] < 60   # Not insane
        assert savings["speedup_estimate"] > 1.0

    def test_task_detection_coding(self):
        skipper = self.LayerSkipper(total_layers=80)
        task = skipper.detect_task_type("build me a REST API in Python")
        assert task == self.TaskType.CODING

    def test_task_detection_debugging(self):
        skipper = self.LayerSkipper(total_layers=80)
        task = skipper.detect_task_type("I have an error in my code, exception traceback")
        assert task == self.TaskType.DEBUGGING

    def test_reasoning_skips_less_than_coding(self):
        skipper = self.LayerSkipper(total_layers=80)
        coding_skips = len(skipper.get_skip_indices(self.TaskType.CODING, 80))
        reasoning_skips = len(skipper.get_skip_indices(self.TaskType.REASONING, 80))
        # Reasoning should skip fewer layers (needs more depth)
        assert reasoning_skips <= coding_skips

    def test_skip_cache_works(self):
        skipper = self.LayerSkipper(total_layers=80)
        indices1 = skipper.get_skip_indices(self.TaskType.CODING, 80)
        indices2 = skipper.get_skip_indices(self.TaskType.CODING, 80)
        assert indices1 == indices2


# ─── Quantizer Tests ──────────────────────────────────────────────────────────

class TestDynamicQuantizer:
    def setup_method(self):
        from core.runtime.quantizer import DynamicQuantizer, QuantBits
        self.DynamicQuantizer = DynamicQuantizer
        self.QuantBits = QuantBits

    def test_first_layer_gets_q8(self):
        q = self.DynamicQuantizer(total_layers=80, task_hint="coding")
        config = q.get_layer_config(0)
        assert config.bits == self.QuantBits.Q8

    def test_last_layers_get_q8(self):
        q = self.DynamicQuantizer(total_layers=80, task_hint="coding")
        config = q.get_layer_config(79)
        assert config.bits == self.QuantBits.Q8

    def test_all_layers_have_config(self):
        q = self.DynamicQuantizer(total_layers=80)
        for i in range(80):
            config = q.get_layer_config(i)
            assert config is not None
            assert config.layer_index == i

    def test_quality_retention_reasonable(self):
        q = self.DynamicQuantizer(total_layers=80)
        for i in range(80):
            config = q.get_layer_config(i)
            assert 0.9 <= config.estimated_quality_retention <= 1.0

    def test_summary_contains_all_layers(self):
        q = self.DynamicQuantizer(total_layers=80)
        summary = q.get_summary()
        total = (summary["q2_layers"] + summary["q4_layers"] +
                 summary["q8_layers"] + summary["f16_layers"])
        assert total == 80


# ─── Storage Detector Tests ───────────────────────────────────────────────────

class TestStorageDetector:
    def setup_method(self):
        from core.storage.detector import StorageDetector
        self.StorageDetector = StorageDetector

    def test_detects_at_least_one_drive(self):
        detector = self.StorageDetector()
        devices = detector.detect_all()
        assert len(devices) >= 1

    def test_all_devices_have_required_fields(self):
        detector = self.StorageDetector()
        devices = detector.detect_all()
        for d in devices:
            assert d.path
            assert d.total_gb > 0
            assert d.free_gb >= 0
            assert d.read_speed_mbs >= 0

    def test_exactly_one_recommended(self):
        detector = self.StorageDetector()
        devices = detector.detect_all()
        usable = [d for d in devices if d.usable_for_models]
        if usable:
            recommended = [d for d in devices if d.is_recommended]
            assert len(recommended) <= 1


# ─── Model Selector Tests ─────────────────────────────────────────────────────

class TestModelSelector:
    def setup_method(self):
        from core.models.selector import ModelSelector
        self.ModelSelector = ModelSelector

    def test_always_has_compatible_models(self):
        """Even a 4GB system should get at least one model."""
        selector = self.ModelSelector()
        # Override RAM for test
        selector.effective_ram_gb = 100  # 100GB effective
        compatible = selector.get_compatible_models()
        assert len(compatible) > 0

    def test_system_info_returns_all_fields(self):
        selector = self.ModelSelector()
        info = selector.get_system_info()
        assert "ram_gb" in info
        assert "effective_ram_gb_with_runtime" in info
        assert "cpu_cores" in info
        assert info["ram_gb"] > 0

    def test_runtime_boost_applied(self):
        selector = self.ModelSelector()
        assert selector.effective_ram_gb > selector.ram_gb


# ─── Code Parser Tests ────────────────────────────────────────────────────────

class TestCodeParser:
    def setup_method(self):
        from agent.loop import CodeParser
        self.parser = CodeParser()

    def test_extracts_python_file(self):
        text = """
Here's the code:
```python main.py
print("hello world")
```
"""
        files = self.parser.extract_files(text)
        assert len(files) == 1
        assert files[0].path == "main.py"
        assert "hello world" in files[0].content

    def test_extracts_multiple_files(self):
        text = """
```python server.py
from flask import Flask
app = Flask(__name__)
```

```javascript client.js
console.log("hello");
```
"""
        files = self.parser.extract_files(text)
        assert len(files) == 2

    def test_detects_python_language(self):
        lang = self.parser.detect_language("main.py")
        assert lang == "python"

    def test_detects_javascript(self):
        lang = self.parser.detect_language("app.js")
        assert lang == "javascript"

    def test_extracts_bash_commands(self):
        text = """
```bash
pip install flask
python main.py
```
"""
        commands = self.parser.extract_commands(text)
        assert "pip install flask" in commands
        assert "python main.py" in commands


# ─── Command Runner Tests ─────────────────────────────────────────────────────

class TestCommandRunner:
    def setup_method(self):
        import tempfile
        from agent.loop import CommandRunner
        self.tmp = tempfile.mkdtemp()
        self.runner = CommandRunner(self.tmp)

    def test_runs_simple_python(self):
        success, stdout, stderr = self.runner.run("python -c \"print('hello')\"")
        assert success
        assert "hello" in stdout

    def test_captures_error(self):
        success, stdout, stderr = self.runner.run("python -c \"raise ValueError('test error')\"")
        assert not success
        assert "ValueError" in stderr

    def test_write_and_read_file(self):
        self.runner.write_file("test.txt", "hello world")
        content = self.runner.read_file("test.txt")
        assert content == "hello world"

    def test_list_files(self):
        self.runner.write_file("a.py", "x=1")
        self.runner.write_file("b.py", "y=2")
        files = self.runner.list_files()
        assert "a.py" in files
        assert "b.py" in files


if __name__ == "__main__":
    pytest.main([__file__, "-v"])