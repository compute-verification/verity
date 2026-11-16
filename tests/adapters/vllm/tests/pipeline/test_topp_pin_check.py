"""The top-p pin check's command: which mode it runs, with what, and its exit code.

`kernel` and `served` need a GPU and vLLM.  A stand-in takes the place of each here, so these cases show the command and not
the check: that the pin holds on a device is what the two modes say when a GPU runs them.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from verity_vllm.pipeline import cli
from verity_vllm.pipeline import topp_pin_check as tc


def _run(monkeypatch, argv: list[str], ok: bool) -> tuple[list[tuple], int | str | None]:
    """The command with a stand-in for each mode, which notes its arguments and whether its directory was there."""
    calls: list[tuple] = []

    def kernel(out: Path) -> dict:
        calls.append(("kernel", out, out.is_dir()))
        return {"ok": ok}

    def served(workload: str, out: Path) -> dict:
        calls.append(("served", workload, out, out.is_dir()))
        return {"ok": ok}

    monkeypatch.setattr(tc, "kernel", kernel)
    monkeypatch.setattr(tc, "served", served)
    with pytest.raises(SystemExit) as e:
        tc.main(cli.parse("topp-pin-check", argv, {}))
    return calls, e.value.code


@pytest.mark.parametrize(("ok", "code"), [(True, 0), (False, 1)])
def test_kernel_runs_the_kernel_check_in_a_directory_the_command_makes(tmp_path, monkeypatch, ok, code):
    out = tmp_path / "a" / "b"
    calls, rc = _run(monkeypatch, ["kernel", "--out", str(out)], ok)
    assert calls == [("kernel", out, True)] and rc == code


@pytest.mark.parametrize(("ok", "code"), [(True, 0), (False, 1)])
def test_served_runs_the_served_check_on_its_workload(tmp_path, monkeypatch, ok, code):
    out, workload = tmp_path / "out", str(tmp_path / "w.json")
    calls, rc = _run(monkeypatch, ["served", "--workload", workload, "--out", str(out)], ok)
    assert calls == [("served", workload, out, True)] and rc == code


def test_a_directory_that_is_there_is_kept_as_it_is(tmp_path, monkeypatch):
    (tmp_path / "kept.txt").write_text("x")
    calls, rc = _run(monkeypatch, ["kernel", "--out", str(tmp_path)], True)
    assert calls == [("kernel", tmp_path, True)] and rc == 0 and (tmp_path / "kept.txt").read_text() == "x"


@pytest.mark.parametrize("argv", [["served", "--out", "DIR"], ["kernel"], ["kernel", "--out", "DIR", "--workload", "w.json"], []])
def test_a_mode_without_its_options_or_with_another_s_is_a_usage_error(argv, capsys):
    with pytest.raises(SystemExit) as e:
        cli.parse("topp-pin-check", argv, {})
    assert e.value.code == 2 and "usage: verity-vllm topp-pin-check" in capsys.readouterr().err
