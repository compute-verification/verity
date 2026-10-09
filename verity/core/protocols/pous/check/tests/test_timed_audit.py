"""`timed_audit.py`: `pous-timed-audit` gives `memory_challenger.verdict`'s verdict on every generated record and its
negative controls, a difference is reported, on the deployment's record (eight segments, twenty audits) and on the strict
memory challenger's own documents; and a replay names what a run's record lacks."""

from __future__ import annotations

import importlib.util
import json
import random
import shutil
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pous_timed_audit", HERE / "timed_audit.py")
ta = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ta)


@pytest.fixture(scope="module")
def compared():
    return ta.compare()


@pytest.mark.slow
def test_every_generated_record_agrees(compared, capsys):
    assert ta.report(compared) == 0
    negatives = sum(r["negative"] for r in compared)
    assert capsys.readouterr().out.endswith(f"{len(compared)} instances, 0 differences, {negatives} negative controls\n")
    outs = [json.loads(r["python"]) for r in compared]
    assert {o["fault"] for o in outs if "fault" in o} == {"ready", "leaves", "record", "root", "limit", "short", "index",
                                                          "long"}
    assert sum("refused" in o for o in outs) >= 10 and negatives >= 200
    judged = [(r, o) for r, o in zip(compared, outs) if "accepted" in o]
    assert any(o["accepted"] for _, o in judged) and any(r["negative"] and not o["accepted"] for r, o in judged)
    assert any(o["accepted"] and o["profile"] is not None and o["gaps"] for _, o in judged)


@pytest.mark.slow
def test_a_difference_is_reported(compared, capsys):
    lean = json.loads(compared[0]["lean"])
    lean["accepted"] = not lean["accepted"]
    tampered = [compared[0] | {"lean": ta.canonical(lean)}] + compared[1:]
    assert ta.report(tampered) == 1
    out = capsys.readouterr().out
    assert f"DIFFERENCE {compared[0]['id']}" in out and ", 1 differences, " in out


@pytest.mark.slow
def test_the_deployments_record_agrees_from_its_files():
    """The record the GPU run writes at the certificate's point: eight segments (2^19 leaves) and twenty audits of 105,
    judged from files as the harness keeps them; and one wrong block in the last audit rejects it both ways."""
    from verity.core.protocols.pous.schemes import p2 as P2

    s = ta.Store(P2.P2Params(*P2.POINT, segments=P2.CERT_BLOCKS // P2.POINT[2]), b"deployment")
    doc = ta.record(s, random.Random(0), k=P2.CERT_K, deadline=P2.DELTA_NS, rtt=P2.RTT_ALLOWANCE_CAP_NS, strict=True,
                    audits=20)
    mc = ta.challenger()
    py = mc.verdict(s.ready, s.leaves, doc)
    assert py["accepted"] and py["audits_accepted"] == 20 and py["gaps"] == [] and py["profile"]["blocks"] == 1 << 19
    assert ta.lean_verdict(s.ready, s.leaves, doc) == py
    a = doc["records"][-1]["answers"][-1]
    a["block"] = ta._flip(a["block"])
    py = mc.verdict(s.ready, s.leaves, doc)
    assert not py["accepted"] and py["audits_accepted"] == 19 and py["audits"][-1]["answers"][-1] == [False, True]
    assert ta.lean_verdict(s.ready, s.leaves, doc) == py


def _live(tmp_path, limit_ns: int, wrong=lambda i: False):
    """`memory_challenger.py --strict`'s document of `p2-16448/v4` at its registered point (one segment) against a responder
    whose blocks match the leaves (but for the indices `wrong` picks, answered with zeros), with its ready file and leaves."""
    import hashlib

    sys.path.insert(0, str(ta.ROOT / "benchmarks" / "pous"))
    from band_gpu import harness as H
    from band_gpu import memory_challenger as live

    from verity.core.protocols.pous import schemes
    from verity.core.protocols.pous.verifier import VerifierKey, leaf_hash

    scheme = schemes.get("p2-16448/v4")
    params, salt = scheme.params, b"s" * 24
    block = lambda i: hashlib.shake_256(i.to_bytes(4, "big")).digest(params.block_bytes)  # noqa: E731
    leaves = [leaf_hash(params, salt, i, block(i)) for i in range(params.blocks)]
    vk = VerifierKey.from_leaves(params, salt, leaves)
    answer = lambda i: (bytes(params.block_bytes) if wrong(i) else block(i), 0, 1_000, 0)  # noqa: E731
    responder = H.ControlResponder(params.block_bytes, answer)
    try:
        doc = live.audit("127.0.0.1", responder.port(), scheme, vk, audits=2, pings=20, limit_ns=limit_ns,
                         clock=live.LogicalClock(), research=False)
    finally:
        responder.stop()
    ready = {"scheme": scheme.name, "params": params.manifest(), "salt": salt.hex(), "port": 0}
    return live, ready, b"".join(leaves), {"root": vk.root.hex(), "clock": "logical", **doc}


@pytest.mark.slow
def test_the_strict_memory_challengers_documents_are_judged_alike(tmp_path):
    """`judge` by either verdict gives the same document, the one the memory challenger wrote: accepted at the RTT cap,
    refused before any challenge one microsecond past it, and rejected where the responder serves zeros for every index
    divisible by 7 (each audit of 105 challenges meets one but with probability (6/7)^105 < 2^-23)."""
    from verity.core.protocols.pous.schemes import p2

    cap = 500_000 + p2.RTT_ALLOWANCE_CAP_NS
    honest = lambda i: False  # noqa: E731
    for limit, wrong, accepted in [(cap, honest, 2), (cap + 1_000, honest, 0), (cap, lambda i: i % 7 == 0, 0)]:
        live, ready, leaves, doc = _live(tmp_path, limit, wrong)
        by = {name: live.judge(ready, leaves, doc, by=name) for name in ("python", "lean")}
        assert live.judge(ready, leaves, doc)["verdict"]["by"] == live.VERDICT == "lean"
        assert by["python"]["verdict"].pop("by") == "python" and by["lean"]["verdict"].pop("by") == "lean"
        assert by["python"] == by["lean"]
        assert by["python"]["verdicts"] == {"lean": by["python"]["verdict"], "python": by["python"]["verdict"]}
        judged = {k: v for k, v in by["python"].items() if k not in ("verdict", "verdicts")}
        assert judged == doc and by["python"]["audits_accepted"] == accepted
        assert (doc.get("wrong", 0) > 0) == (wrong is not honest)
        assert ("refused" in by["python"]["verdict"]) == (limit != cap)


@pytest.fixture(scope="module")
def strict_doc(tmp_path_factory):
    """The strict memory challenger's accepted document at the RTT cap, and the same with one answer's block flipped,
    which both verdicts reject."""
    from verity.core.protocols.pous.schemes import p2

    live, ready, leaves, doc = _live(tmp_path_factory.mktemp("strict"), 500_000 + p2.RTT_ALLOWANCE_CAP_NS)
    bad = json.loads(json.dumps(doc))
    a = bad["records"][1]["answers"][3]
    a["block"] = a["block"][:-1] + ("0" if a["block"][-1] != "0" else "1")
    return live, ready, leaves, doc, bad


@pytest.mark.parametrize("record", ["lean", "python"])
@pytest.mark.parametrize("forced", ["lean", "python"])
def test_a_disagreement_between_the_verdicts_rejects_the_document(strict_doc, forced, record, monkeypatch):
    """`forced`'s verdict flips its accept bit (rejecting the honest document the other accepts, and accepting the flipped
    one the other rejects): the document accepts nothing, its verdict names the disagreement and both outcomes, and both
    verdicts are kept, whichever one accepted and whichever is the record's."""
    live, ready, leaves, honest, bad = strict_doc
    real = live.VERDICTS[forced]

    def flipped(*args):
        v = real(*args)
        return {**v, "accepted": not v["accepted"], "audits_accepted": len(v["audits"]) - v["audits_accepted"]}

    monkeypatch.setitem(live.VERDICTS, forced, flipped)
    other = "python" if forced == "lean" else "lean"
    for doc, accepts in ((honest, other), (bad, forced)):
        out = live.judge(ready, leaves, doc, by=record)
        v, both = out["verdict"], out["verdicts"]
        assert set(v) == {"by", "rejected"} and v["by"] == record and v["rejected"].startswith(live.DISAGREE + ": lean {")
        assert both[accepts]["accepted"] and not both[forced if accepts == other else other]["accepted"]
        assert out["audits_accepted"] == 0 and out["profile"] is None and not out["audit_of_record"]["accepted"]


def test_without_the_lean_the_python_is_the_records_and_a_lean_failure_rejects(strict_doc, monkeypatch):
    """Where the Lean can't run (the GPU pods' venv), the Python's verdict is the record's and the Lean's is marked
    unavailable, for `--replay`; asked for the Lean's there, nothing is accepted.  Where the Lean should run and raises,
    the document is rejected for the disagreement."""
    live, ready, leaves, honest, _ = strict_doc
    monkeypatch.setattr(live, "lean_unavailable", lambda: "lake is not on the PATH")
    out = live.judge(ready, leaves, honest)
    assert out["verdict"]["by"] == "python" and out["verdict"]["accepted"] and out["audits_accepted"] == 2
    assert out["verdicts"]["lean"] == {"unavailable": "lake is not on the PATH"}
    assert live.judge(ready, leaves, honest, by="lean")["audits_accepted"] == 0
    monkeypatch.setattr(live, "lean_unavailable", lambda: None)

    def broken(*args):
        raise RuntimeError("pous-timed-audit failed (2)")

    monkeypatch.setitem(live.VERDICTS, "lean", broken)
    out = live.judge(ready, leaves, honest)
    assert out["verdicts"]["lean"] == {"failed": "RuntimeError: pous-timed-audit failed (2)"}
    assert out["verdict"]["by"] == "lean" and out["verdict"]["rejected"].startswith(live.DISAGREE)
    assert out["audits_accepted"] == 0


def test_the_lean_is_unavailable_without_lake_or_its_server(monkeypatch):
    """The Lean runs where `lake` is on the PATH and `verity_difftest` imports; otherwise `lean_unavailable` says which."""
    live = ta.challenger()
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert live.lean_unavailable() == "lake is not on the PATH"
    monkeypatch.setattr(shutil, "which", lambda name: f"/opt/elan/bin/{name}")
    monkeypatch.setitem(sys.modules, "verity_difftest", None)
    assert live.lean_unavailable().startswith("verity_difftest is not importable here (")
    monkeypatch.delitem(sys.modules, "verity_difftest")
    assert live.lean_unavailable() is None


def test_a_replay_writes_out_and_holds_the_recorded_verdicts_to_its_own(strict_doc, tmp_path, capsys):
    """`--replay --out`: one line per document with both outputs; the verdicts `judge` recorded in the document are held to
    the replay's, and one recorded otherwise is a difference."""
    live, ready, leaves, honest, _ = strict_doc
    bench = tmp_path / "run" / "bench"
    bench.mkdir(parents=True)
    judged = live.judge(ready, leaves, honest, by="python")
    (bench / "ready-p2-honest.json").write_text(json.dumps(ready))
    (bench / "audit-p2-honest.json").write_text(json.dumps(judged))
    (bench / "leaves-p2.bin").write_bytes(leaves)
    assert ta.main(["--replay", str(tmp_path / "run"), "--out", str(tmp_path / "out.jsonl")]) == 0
    assert capsys.readouterr().out.endswith("1 runs, 1 same verdict, 0 negative controls\n"
                                            "1 with recorded verdicts, 1 recorded as replayed\n")
    [row] = [json.loads(s) for s in (tmp_path / "out.jsonl").read_text().splitlines()]
    assert row["same"] and row["recorded_same"] and not row["negative"] and set(row["recorded"]) == {"lean", "python"}
    assert row["recorded"]["python"] == row["python"] == row["lean"]
    judged["verdicts"]["python"]["accepted"] = False
    (bench / "audit-p2-honest.json").write_text(json.dumps(judged))
    assert ta.main(["--replay", str(tmp_path / "run")]) == 1
    assert "recorded lean and python DIFFERENT" in capsys.readouterr().out


@pytest.mark.slow
def test_a_replay_judges_a_record_with_blocks_and_names_what_one_without_lacks(tmp_path, capsys):
    from verity.core.protocols.pous.schemes import p2

    _, ready, leaves, doc = _live(tmp_path, 500_000 + p2.RTT_ALLOWANCE_CAP_NS)
    bench = tmp_path / "r-blocks" / "bench"
    bench.mkdir(parents=True)
    (bench / "ready-p2-honest.json").write_text(json.dumps(ready))
    (bench / "audit-p2-honest.json").write_text(json.dumps(doc))
    (bench / "leaves-p2.bin").write_bytes(leaves)
    for a in doc["records"][1]["answers"]:
        a["block"] = "00" * 2056
    (bench / "ready-p2-value.json").write_text(json.dumps(ready))
    (bench / "audit-p2-value.json").write_text(json.dumps(doc))
    old = tmp_path / "r-old" / "bench"
    old.mkdir(parents=True)
    for a in doc["records"][0]["answers"]:
        del a["block"]
    (old / "ready-p2-honest.json").write_text(json.dumps(ready))
    (old / "audit-p2-honest.json").write_text(json.dumps(doc))
    (old / "leaves-p2.bin").write_bytes(leaves)
    (old / "ready-p2-v3.json").write_text(json.dumps({**ready, "scheme": "p2-16448/v3",
                                                      "params": {**ready["params"], "expander": "shake256-v1"}}))
    (old / "audit-p2-v3.json").write_text(json.dumps(doc))
    assert ta.replay([tmp_path / "r-blocks", tmp_path / "r-old"]) == 0
    out = capsys.readouterr().out
    assert "r-old honest: cannot replay, 105 of 210 answers do not record their block" in out
    assert ("r-old v3: cannot replay, p2-16448/v3 with expander shake256-v1 is not P2 v4; 105 of 210 answers do not record "
            "their block") in out
    assert out.endswith("2 runs, 2 same verdict, 1 negative controls\n")
