"""advice_c.py on the served endpoint fixture (benchmarks/network_e2e/tests/data/endpoint), rebuilt as a boundary-rows arm:
its Commit windows' requests become boundary windows' index.json spans, its trace loses its window records, and the client
records carry the ids vLLM made its engine ids from. Run from a tree's root:

    PYTHONPATH=integrations/vllm:. python -m pytest -q PATH/test_advice_c.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import advice_c as AC  # noqa: E402
from verity_vllm.pipeline import advice as ADV  # noqa: E402

FIX = Path("benchmarks/network_e2e/tests/data/endpoint")


def lines(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def client_id(engine_id: str) -> str:
    return re.fullmatch(r"cmpl-([0-9a-f]+)-0", engine_id).group(1)


@pytest.fixture
def arm(tmp_path: Path) -> Path:
    a = tmp_path / "boundary-rows"
    (a / "server").mkdir(parents=True)
    recs = lines(FIX / "advice_trace.jsonl")
    (a / "server" / "step_trace.1.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs if r["kind"] != "window"))
    reqs = {r["engine_id"]: r for r in lines(FIX / "requests.jsonl")}
    for w in lines(FIX / "windows.jsonl"):
        d = a / "server" / "boundary" / f"w{w['window']}" / "p16-o16"
        d.mkdir(parents=True)
        spans = [{"engine_id": e, "prompt": reqs[e]["prompt_ids"], "sampling": {}, "served": reqs[e]["streamed_ids"],
                  "finish": reqs[e]["finish"]["reason"]} for e in w["requests"]]
        (d / "index.json").write_text(json.dumps({"spans": spans}))
    (a / "warm").mkdir()
    (a / "warm" / "client.jsonl").write_text("".join(json.dumps({"id": client_id(e)}) + "\n" for e in reqs))
    return a


def test_the_cut_gives_the_served_windows_slices(arm: Path):
    recs = AC.records(next((arm / "server").glob("step_trace.*")))
    windows = {w: set(m) for w, m in AC.boundary(arm).items()}
    served, _ = ADV.windows_of(lines(FIX / "advice_trace.jsonl"))
    assert AC.cut(recs, windows) == served


def test_each_window_gets_an_advice_and_only_its_binding_is_published(arm: Path, tmp_path: Path, monkeypatch, capsys):
    priv, out = tmp_path / "priv", tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["advice_c.py", str(arm), str(priv), str(out), "--controls"])
    assert AC.main() == 0
    said = capsys.readouterr().out
    assert "lines up with the boundary windows w0, w1, w2" in said and said.count(": refused") == 2
    for w in range(3):
        pub = json.loads((out / "advice" / f"w{w}" / "registration.json").read_text())
        full = json.loads((priv / f"w{w}" / "registration.json").read_text())
        assert list(pub) == ["binding"] and pub["binding"] == full["binding"]
        assert {r["user"] for r in lines(priv / f"w{w}" / "requests.jsonl")} <= {client_id(e) for e in AC.boundary(arm)[w]}
    assert [p.relative_to(out).as_posix() for p in sorted(out.rglob("*")) if p.is_file()] == \
        [f"advice/w{w}/registration.json" for w in range(3)]


def test_a_request_carried_across_a_close_is_refused(arm: Path):
    recs = AC.records(next((arm / "server").glob("step_trace.*")))
    windows = {w: set(m) for w, m in AC.boundary(arm).items()}
    late = sorted(windows[1])[0]
    windows[1].discard(late)
    windows[2].add(late)
    with pytest.raises(AC.CutError, match="wrong window|overlap"):
        AC.cut(recs, windows)


def test_a_request_in_no_window_is_refused(arm: Path):
    recs = AC.records(next((arm / "server").glob("step_trace.*")))
    windows = {w: set(m) for w, m in AC.boundary(arm).items()}
    windows[2].discard(sorted(windows[2])[0])
    with pytest.raises(AC.CutError, match="no boundary window"):
        AC.cut(recs, windows)


def test_an_engine_id_without_a_client_id_is_refused(arm: Path):
    (arm / "warm" / "client.jsonl").write_text(json.dumps({"id": "someone-else"}) + "\n")
    with pytest.raises(AC.CutError, match="carries no client id"):
        AC.users(arm, set().union(*map(set, AC.boundary(arm).values())))
