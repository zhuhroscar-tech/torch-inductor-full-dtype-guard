"""Tests for the CLI entry point: argument parsing, --version, --json,
--no-color, and exit codes -- independent of whether torch is installed.
Mirrors the test_cli.py pattern already used across the fleet.

The branch-coverage tests below mock ``core.diagnose`` so every CLI
message path (torch-unavailable, no-divergence "info" lines, and a
guard-mismatch "fail" line) is exercised deterministically, regardless
of whether this host's installed torch build happens to reproduce the
underlying bug. Real fleet-wide gap found by pygount/pytest-cov
inspection: cli.py sat at 80% coverage with the negative branches of
each status line (lines 45-51, 63, 68, 73) and the ``__main__`` guard
(line 107) never hit by any existing test."""
from __future__ import annotations

import json
import runpy
import sys

import pytest

from torch_inductor_full_dtype_guard import core
from torch_inductor_full_dtype_guard.cli import main


def _fake_report(**overrides):
    report = {
        "torch_version": "9.9.9-fake",
        "issue_urls": ["https://github.com/pytorch/pytorch/issues/194062"],
        "bool_fill_cases": [],
        "int8_overflow_cases": [],
        "any_bool_fill_divergence": False,
        "any_int8_silent_overflow": False,
        "guard_fully_correct": True,
    }
    report.update(overrides)
    return report


def test_version_flag(capsys):
    code = main(["--version"])
    out = capsys.readouterr().out
    assert code == 0
    assert "torch-inductor-full-dtype-guard" in out


def test_json_output_is_valid_json_and_reports_guard_status(capsys):
    torch = pytest.importorskip("torch")
    code = main(["--json"])
    out = capsys.readouterr().out
    report = json.loads(out)
    assert "torch_version" in report
    assert report["torch_version"] == torch.__version__
    assert "guard_fully_correct" in report
    assert code in (0, 1)


def test_json_exit_code_matches_guard_fully_correct(capsys):
    pytest.importorskip("torch")
    code = main(["--json"])
    out = capsys.readouterr().out
    report = json.loads(out)
    assert code == (0 if report["guard_fully_correct"] else 1)


def test_text_output_no_color_has_no_ansi_escapes(capsys):
    pytest.importorskip("torch")
    main(["--no-color"])
    out = capsys.readouterr().out
    assert "\x1b[" not in out


def test_text_output_reports_bool_fill_and_int8_sections(capsys):
    pytest.importorskip("torch")
    main(["--no-color"])
    out = capsys.readouterr().out
    assert "bool-fill cases" in out
    assert "int8-overflow cases" in out


def test_torch_unavailable_json_mode_reports_error_and_exit_2(monkeypatch, capsys):
    """cli.py lines 45-47: TorchUnavailableError + --json emits a JSON
    error object and exits 2, regardless of torch install state."""

    def _raise(*args, **kwargs):
        raise core.TorchUnavailableError("torch is required for diagnosis")

    monkeypatch.setattr(core, "diagnose", _raise)
    code = main(["--json"])
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload == {"error": "torch is required for diagnosis"}
    assert code == 2


def test_torch_unavailable_text_mode_reports_fail_headline_and_exit_2(monkeypatch, capsys):
    """cli.py lines 48-51: TorchUnavailableError in text mode prints a
    'fail' status headline (not the JSON branch) and exits 2."""

    def _raise(*args, **kwargs):
        raise core.TorchUnavailableError("torch is required for diagnosis")

    monkeypatch.setattr(core, "diagnose", _raise)
    code = main(["--no-color"])
    out = capsys.readouterr().out
    assert "torch unavailable: torch is required for diagnosis" in out
    assert "[X]" in out  # fail glyph, ASCII form since --no-color
    assert code == 2


def test_no_bool_fill_divergence_prints_info_line(monkeypatch, capsys):
    """cli.py line 63: the 'info' (not 'warn') branch when the host's
    torch build does NOT reproduce the bool-fill divergence."""
    monkeypatch.setattr(core, "diagnose", lambda: _fake_report())
    main(["--no-color"])
    out = capsys.readouterr().out
    assert "no bool-fill divergence reproduced on this host's installed torch build" in out
    assert "torch.compile(inductor) bool-fill dtype-cast divergence reproduced" not in out


def test_no_int8_silent_overflow_prints_info_line(monkeypatch, capsys):
    """cli.py line 68: the 'info' (not 'fail') branch when the host's
    torch build does NOT silently skip the int8 overflow check."""
    monkeypatch.setattr(core, "diagnose", lambda: _fake_report())
    main(["--no-color"])
    out = capsys.readouterr().out
    assert "no silent int8-overflow case reproduced on this host" in out
    assert "int8 overflow check silently skipped" not in out


def test_guard_mismatch_prints_fail_line_and_exit_1(monkeypatch, capsys):
    """cli.py line 73 + 103: guard_fully_correct=False prints the 'fail'
    headline (not the 'ok' one) and the process exits 1, both in text
    and JSON mode."""
    monkeypatch.setattr(core, "diagnose", lambda: _fake_report(guard_fully_correct=False))
    code = main(["--no-color"])
    out = capsys.readouterr().out
    assert "guard did NOT match eager on at least one case" in out
    assert "safe_full() matches eager on every case" not in out
    assert code == 1

    monkeypatch.setattr(core, "diagnose", lambda: _fake_report(guard_fully_correct=False))
    code_json = main(["--json"])
    capsys.readouterr()
    assert code_json == 1


def test_divergence_and_overflow_reproduced_prints_warn_and_fail_lines(monkeypatch, capsys):
    """Positive-branch companion to the two 'info'-line tests above:
    when the bug IS reproduced, the 'warn'/'fail' headlines fire
    instead (lines 61 and 66), independent of real torch/hardware."""
    monkeypatch.setattr(
        core,
        "diagnose",
        lambda: _fake_report(any_bool_fill_divergence=True, any_int8_silent_overflow=True),
    )
    main(["--no-color"])
    out = capsys.readouterr().out
    assert "torch.compile(inductor) bool-fill dtype-cast divergence reproduced" in out
    assert "int8 overflow check silently skipped under torch.compile" in out


def test_module_entry_point_runs_main_and_exits_with_its_code(monkeypatch):
    """cli.py line 107 (``if __name__ == "__main__": sys.exit(main())``):
    running the module as a script must invoke main() and propagate its
    return code via SystemExit, not just be dead code."""
    monkeypatch.setattr(core, "diagnose", lambda: _fake_report(guard_fully_correct=False))
    monkeypatch.setattr(sys, "argv", ["torch-inductor-full-dtype-guard", "--no-color"])
    monkeypatch.delitem(sys.modules, "torch_inductor_full_dtype_guard.cli", raising=False)
    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module("torch_inductor_full_dtype_guard.cli", run_name="__main__")
    assert exc_info.value.code == 1
