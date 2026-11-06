"""`verity_flock.challengers`, the sweeps' challenger per `serve` process, and `circuit_bench.serve_plan`'s use of it."""
from pathlib import Path
from types import SimpleNamespace

from verity_flock import challengers as CH
from verity_flock import circuit_bench as CB


class _Proc:
    def __init__(self, rc: int = 0):
        self.returncode = rc

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        pass


def test_a_challenger_passes_only_if_it_exits_0_and_audits_at_least_one_record(monkeypatch, tmp_path: Path):
    for name in ("l0000-7", "l0001-7", "l0000-8"):
        (tmp_path / "sessions" / name).mkdir(parents=True)
        (tmp_path / "sessions" / name / "session.json").write_text("{}")
    (tmp_path / "sessions" / "l0002-7").mkdir()                    # a session that wrote no record
    assert CH.records(tmp_path / "sessions", "*-7") == [str(tmp_path / "sessions" / n) for n in ("l0000-7", "l0001-7")]
    audit = {"rc": 0}
    monkeypatch.setattr(CH.subprocess, "run", lambda argv, **_kw: SimpleNamespace(returncode=audit["rc"], stdout="", stderr="no"))
    recs = CH.records(tmp_path / "sessions")
    assert CH.end(tmp_path / "c", _Proc(0), recs)["ok"] is True
    assert CH.end(tmp_path / "c", _Proc(1), recs)["ok"] is False
    assert CH.end(tmp_path / "c", _Proc(0), [])["ok"] is False
    audit["rc"] = 1
    assert CH.end(tmp_path / "c", _Proc(0), recs) == {"exit": 0, "sessions": 3, "audit": False, "ok": False, "audit_tail": "no"}


def test_the_verifier_pods_servers_each_take_their_own_challengers_trees_and_fail_the_plan_if_one_fails(monkeypatch, tmp_path: Path):
    def stage(a, low, input_set, total, d, public):
        d.mkdir(parents=True, exist_ok=True)
        return [(0, total // 2), (total // 2, total)], 0.0

    argvs, started = [], []

    def start(d: Path):
        started.append(d)
        return _Proc(), ["--tree-from", f"127.0.0.1:{7000 + len(started)}", "--tree-secret-file", str(d / "secret")]

    monkeypatch.setattr(CB, "stage_point", stage)
    monkeypatch.setattr(CB.subprocess, "call", lambda argv, **_kw: argvs.append(argv) or 0)
    monkeypatch.setattr(CB.CH, "start", start)
    a = SimpleNamespace(bin="flock-circuit", listen="0.0.0.0:7501", runs=3, operator="pod", tables=[])
    monkeypatch.setenv("FLOCK_CHALLENGER", "1")
    for fails, rc in ((set(), 0), ({"p0-4/challenger-s1"}, 1)):
        argvs.clear()
        started.clear()
        monkeypatch.setattr(CB.CH, "end", lambda d, p, s: {"ok": f"{d.parent.name}/{d.name}" not in fails})
        assert CB.serve_plan(a, None, None, [4], tmp_path, tmp_path / "circuit.txt", "pin") == rc
        assert [d.name for d in started] == ["challenger-s0", "challenger-s1"]
        assert [x[x.index("--tree-from") + 1] for x in argvs] == ["127.0.0.1:7001", "127.0.0.1:7002"]
        assert all(x[x.index("--sessions") + 1] == "5" for x in argvs)
    monkeypatch.setenv("FLOCK_CHALLENGER", "0")
    argvs.clear()
    assert CB.serve_plan(a, None, None, [4], tmp_path, tmp_path / "circuit.txt", "pin") == 0
    assert not any("--tree-from" in x for x in argvs)
