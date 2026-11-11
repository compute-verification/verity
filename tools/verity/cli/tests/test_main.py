"""`verity`'s dispatch: an unknown command is named on its own line, then the usage; `verity bench` finds a runner under
`benchmarks/`, or where `BENCH_ELSEWHERE` names it."""
from __future__ import annotations

import pytest

import verity_cli


def test_an_unknown_command_is_named_then_the_usage_follows():
    with pytest.raises(SystemExit) as e:
        verity_cli.main(["nosuch"])
    first, rest = str(e.value.code).split("\n", 1)
    assert first == "verity: no command 'nosuch'"
    assert rest == verity_cli.__doc__


def test_bench_runs_a_runner_under_benchmarks_or_where_its_name_is_listed(tmp_path, monkeypatch):
    for d in ("benchmarks/here", "protocols/priced"):
        (tmp_path / d).mkdir(parents=True)
        (tmp_path / d / "run.py").write_text("")
    ran = []
    monkeypatch.setattr(verity_cli, "ROOT", tmp_path)
    monkeypatch.setattr(verity_cli, "BENCH_ELSEWHERE", {"priced": "protocols/priced"})
    monkeypatch.setattr(verity_cli, "_run", lambda argv: ran.append(argv[1:]) or 0)
    assert verity_cli.main(["bench", "here", "--out", "D"]) == 0 and verity_cli.main(["bench", "priced"]) == 0
    assert ran == [[str(tmp_path / "benchmarks/here/run.py"), "--out", "D"], [str(tmp_path / "protocols/priced/run.py")]]
    with pytest.raises(SystemExit) as e:
        verity_cli.main(["bench", "gone"])
    assert str(e.value.code).endswith("NAME is one of here, priced")


def test_each_runner_listed_elsewhere_is_there():
    for name, d in verity_cli.BENCH_ELSEWHERE.items():
        assert any((verity_cli.ROOT / d / r).is_file() for r in ("run.py", "run.sh")), f"verity bench {name}: {d} has no runner"
