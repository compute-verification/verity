"""The prove path's advice, and its prover's and verifier's windows on the CPU TP stand-in (integrations/vllm's
`tests.pipeline.served_tp`, deferred), without proving: each rank's window over the compiler's Programs for the declared run's
advice is held to the verifier's recompiled window, the advice's negatives are refused by name, and a forged unit is committed
consistently (one bit of its rows changed, every check before the draw passing), so only the proof can refuse it."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from verity.core.protocols.sampled_proofs.one_stage import registration as R
from verity_vllm.commit import serving_rows as CS
from verity_vllm.pipeline import advice_windows as AW
from verity_vllm.pipeline.tp import serving_rows as SRT

HERE = Path(__file__).resolve().parent
VLLM = HERE.parents[2] / "integrations" / "vllm"
sys.path.insert(0, str(HERE.parent))
import prove_path as PP  # noqa: E402

PROMPTS = (5, 3)


def _workload(tmp_path: Path, arrive: tuple[int, ...] = (0, 2), caps: tuple[int, ...] = (3, 2)) -> Path:
    f = tmp_path / "workload.json"
    f.write_text(json.dumps({"requests": [{"prompt_token_ids": list(range(lp)), "max_tokens": t, "arrive_step": s}
                                          for lp, t, s in zip(PROMPTS, caps, arrive, strict=True)]}))
    return f


def test_a_declared_runs_advice_compiles_to_the_requests_tp_commit_served(tmp_path):
    advice = PP.declared_advice(_workload(tmp_path))
    assert [len(s["rows"]) for s in advice["steps"]] == [1, 1, 2, 1]
    assert advice["steps"][2]["rows"] == [{"request": 0, "kind": "decode", "start": 6, "n": 1, "cache_hit": 0, "preempted": False},
                                          {"request": 1, "kind": "prefill", "start": 0, "n": 3, "cache_hit": 0, "preempted": False}]
    assert PP.requests_of(advice) == [{"request_id": "r0", "arrive_step": 0, "prompt_len": 5, "max_tokens": 3},
                                      {"request_id": "r1", "arrive_step": 2, "prompt_len": 3, "max_tokens": 2}]


def _row(advice: dict, step: int, k: int = 0) -> dict:
    return advice["steps"][step]["rows"][k]


@pytest.mark.parametrize("alter, why", [
    (lambda a: _row(a, 0).update(cache_hit=2, start=2, n=3), "r0: a prefix-cache hit at step 0"),
    (lambda a: _row(a, 0).update(n=4), "r0: a chunked prefill, 4 of its 5 prompt tokens"),
    (lambda a: _row(a, 1).update(preempted=True), "step 1: request r0 preempted"),
    (lambda a: _row(a, 1).update(start=4), r"r0: its row 1 is \(step, start, n\) = \(1, 4, 1\)"),
    (lambda a: _row(a, 3).update(request=7), "step 3: a row of request 7, outside the advice's 2 requests"),
    (lambda a: a["steps"][1].update(rows=[]), r"r0: its row 1 is \(step, start, n\) = \(2, 6, 1\)"),
], ids=["prefix-hit", "chunked-prefill", "preempted", "recompute", "unknown-request", "gap"])
def test_advice_the_compiler_cannot_take_is_refused_by_name(tmp_path, alter, why):
    advice = PP.declared_advice(_workload(tmp_path))
    alter(advice)
    with pytest.raises(ValueError, match=why):
        PP.requests_of(advice)


def test_trailing_rows_past_the_finish_set_the_shape_and_a_request_without_rows_is_not_compiled(tmp_path):
    advice = PP.declared_advice(_workload(tmp_path, caps=(3, 1)))
    advice["requests"][0]["output_len"] = 2
    advice["requests"].append({"id": "r2", "prompt_len": 4, "output_len": 0, "finish": "abort"})
    assert [(q["request_id"], q["max_tokens"]) for q in PP.requests_of(advice)] == [("r0", 3), ("r1", 1)]


def test_only_the_advices_canonical_encoding_is_read(tmp_path):
    doc = PP.declared_advice(_workload(tmp_path))
    (tmp_path / PP.ADVICE).write_bytes(PP.encode(doc))
    assert PP.read_advice(tmp_path) == (doc, PP.encode(doc))
    (tmp_path / PP.ADVICE).write_text(json.dumps(doc, indent=1))
    with pytest.raises(ValueError, match="not the advice's canonical encoding"):
        PP.read_advice(tmp_path)


def _summary(drawn: list[int], accepted: bool | None, lean: bool, before: bool = True) -> dict:
    return {"members": [{"k": 0, "template": "T", "n": 8, "base": 0}], "draw": drawn,
            "checks_before_draw": {"registration": before, "source": True},
            "audit": {"accepted": accepted, "verdicts": {"m0-verifier/m0": lean, "lean-verify/m0": lean}, "why": {}}}


@pytest.mark.parametrize("summary, ok", [
    (_summary([0, 1, 2], False, False), True),
    (_summary([0, 2], False, False), False),
    (_summary([0, 1, 2], True, True), False),
    (_summary([0, 1, 2], False, True), False),
    (_summary([0, 1, 2], False, False, before=False), False),
], ids=["refused-by-lean", "forged-unit-not-drawn", "accepted", "lean-accepted", "refused-before-the-draw"])
def test_a_forged_rank_passes_only_when_lean_refuses_its_drawn_forged_unit(summary, ok):
    assert PP.forged_row(summary, {"member": 0, "unit": 1})["ok"] is ok


@pytest.fixture(scope="module")
def tp(tmp_path_factory):
    """The deferred stand-in TP Commit and its declared advice; both ranks' windows over every unit (subset:8), and rank 0's again
    with member 0's unit 1 forged."""
    pytest.importorskip("torch")
    root = tmp_path_factory.mktemp("prove-path")
    p = subprocess.run([sys.executable, "-m", "tests.pipeline.served_tp", str(root / "served"), "--world", "2", "--I", "256",
                        "--law", "subset:4", "--deferred"], cwd=VLLM, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-3000:]
    c = json.loads(p.stdout.strip().splitlines()[-1])
    assert PP.main(["advice", "--workload", c["workload"], "--out", str(root / "advice")]) == 0
    c["all"] = root / "all.json"
    c["all"].write_text(json.dumps({"templates": ["SiluMul_v2{I=256}"], "law": "subset:8"}))
    base = ["--tp-commit", c["commit"], "--build-root", c["build"], "--advice", str(root / "advice"), "--partition", str(c["all"]),
            "--workers", "1"]
    assert PP.main(["commit", *base, "--out", str(root / "honest")]) == 0
    assert PP.main(["commit", *base, "--out", str(root / "forged"), "--ranks", "0", "--forge", "0:1"]) == 0
    c.update(root=root, advice=root / "advice", honest=root / "honest", forged=root / "forged")
    c["compiled"] = PP.compiled(c["commit"], c["build"], str(c["advice"]))
    return c


def _mine(c, r: int):
    compiler, requests, bound = c["compiled"]
    return AW.window_of_advice(c["commit"], r, c["all"], compiler, requests, bound)


def _publics(d: Path, mine) -> dict[str, bytes]:
    return {m.key: (d / f"m{k}-public.bin").read_bytes() for k, m in enumerate(mine.members)}


def _registered(d: Path) -> dict:
    return json.loads((d / "registration.json").read_text())


def _check(rec: dict, mine, publics: dict[str, bytes]):
    return R.check(rec, mine.program, mine.query, mine.law, CS.SCHEME, CS.expected_ports(mine), CS.leaf_layer(publics))


def _rows(d: Path) -> bytes:
    m = json.loads((d / "window.json").read_text())["members"][0]
    body = (d / "m0-private.bin").read_bytes()
    return body[:len(body) - m["n"] * len(m["ports"]) * CS.SALT_BYTES]


@pytest.mark.slow
def test_each_ranks_window_is_held_to_the_verifiers_window_recompiled_from_the_advice(tp):
    _compiler, requests, bound = tp["compiled"]
    data = (tp["advice"] / PP.ADVICE).read_bytes()
    assert bound["advice"] == {"digest": hashlib.sha512(data).hexdigest(), "bits": 8 * len(data)}
    assert [q["request_id"] for q in requests] == [f"r{i}" for i in range(len(requests))] and requests
    for r in range(2):
        mine, held = _mine(tp, r)
        d = tp["honest"] / f"rank{r}"
        rec = _registered(d)
        assert (rec["window"]["compiler"], rec["window"]["advice"]) == (bound["compiler"], bound["advice"])
        assert SRT.source(rec["window"], mine, held)["ok"]
        assert _check(rec, mine, _publics(d, mine)).ok
        assert [(m.template, m.n) for m in mine.members] == [("SiluMul_v2{I=256}", 8)]


@pytest.mark.slow
def test_the_advices_negatives_are_each_refused_by_name(tp):
    mine, held = _mine(tp, 0)
    got = PP.advice_negatives(0, 2, mine, held, tp["honest"] / "rank0")
    assert {k: v["codes"] for k, v in got.items() if k != "registration-of-another-rank (source)"} == {
        "registration-names-another-compiler (source)": ["compiler"],
        "registration-names-other-advice (source)": ["advice"],
        "registration-binds-no-compiler-or-advice (source)": ["compiler", "advice"]}
    other = got["registration-of-another-rank (source)"]
    assert other["refused"] and {"run", "tp"} <= set(other["codes"])


@pytest.mark.slow
def test_a_forged_unit_passes_every_check_before_the_draw_and_differs_in_one_bit(tp):
    """The forged window's registration holds to the verifier's window and its own public bodies, as the honest one does: its
    rows, commit strings and roots agree with the flipped word. Its rows differ from the honest window's in exactly one bit."""
    mine, held = _mine(tp, 0)
    d = tp["forged"] / "rank0"
    rec = _registered(d)
    assert SRT.source(rec["window"], mine, held)["ok"]
    assert _check(rec, mine, _publics(d, mine)).ok
    honest, forged = _rows(tp["honest"] / "rank0"), _rows(d)
    assert len(honest) == len(forged) and sum(bin(x ^ y).count("1") for x, y in zip(honest, forged, strict=True)) == 1
    assert json.loads((tp["forged"] / "commit.json").read_text())["0"]["forged"] == {"member": 0, "unit": 1}
