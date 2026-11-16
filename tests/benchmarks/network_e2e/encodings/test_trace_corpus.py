"""The trace corpus's traces as advice (`trace_corpus.py`): the mapping on a hand-made trace in the corpus's format, and
every code's round trip on it under the trace's own serving caps."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


def _load(name: str):
    if f"enc_{name}" in sys.modules:
        return sys.modules[f"enc_{name}"]
    spec = importlib.util.spec_from_file_location(f"enc_{name}", HERE.parent / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


C = _load("code")
S = _load("synth")
M = _load("measure")
TC = _load("trace_corpus")

T0 = 1_000.0


def _row(i, start, n, resumed=False):
    return {"id": i, "num_computed_tokens": start, "num_scheduled_tokens": n, "resumed": resumed}


def _write_trace(root: Path) -> Path:
    """Four requests: b hits a's first two blocks; c is preempted after two outputs, sits out a step while d runs, and
    resumes from a re-hit of its own first block. A step that ran nothing (a finished-only report) is dropped."""
    (root / "meta.json").write_text(json.dumps({"traffic": {"caps": [2, 3]}}))
    d = root / "tiny"
    d.mkdir()
    (d / "meta.json").write_text(json.dumps({"server_argv": ["--max-num-seqs", "8", "--max-num-batched-tokens", "64",
                                                             "--max-model-len", "128"]}))
    reqs = [("a", list(range(40)), 2, "length", False, 0.00),
            ("b", list(range(32)) + list(range(100, 108)), 2, "stop", True, 0.05),
            ("c", list(range(200, 220)), 3, "length", False, 0.25), ("d", list(range(300, 316)), 1, "stop", True, 0.26)]
    (d / "requests.jsonl").write_text("".join(json.dumps(
        {"id": i, "arrival": T0 + t, "prompt_ids": p, "prompt_len": len(p), "output_len": o, "finish_reason": f, "eos": e})
        + "\n" for i, p, o, f, e, t in reqs))
    steps = [([], [_row("a", 0, 40)], 0.01, 0.02), ([_row("a", 40, 1)], [_row("b", 32, 8)], 0.06, 0.07),
             ([_row("b", 40, 1)], [], 0.0705, 0.09), ([], [], 0.10, None), ([], [_row("c", 0, 20)], 0.30, 0.31),
             ([_row("c", 20, 1)], [], 0.312, 0.33), ([], [_row("d", 0, 16)], 0.332, 0.34),
             ([], [_row("c", 16, 6, True)], 0.35, 0.36)]
    (d / "steps.jsonl").write_text("".join(json.dumps(
        {"step": k, "t": T0 + t, "cached": c, "new": n, **({"t_output": T0 + to} if to is not None else {})}) + "\n"
        for k, (c, n, t, to) in enumerate(steps)))
    return d


def test_a_trace_maps_to_i1_advice(tmp_path):
    t = TC.load_trace(C, _write_trace(tmp_path))
    adv, pub = t["advice"], t["public"]
    assert [(q.id, q.prompt_len, q.output_len, q.finish) for q in adv.requests] == \
        [("a", 40, 2, "length"), ("b", 40, 2, "eos"), ("c", 20, 3, "length"), ("d", 16, 1, "eos")]
    R = C.Row
    assert [s.rows for s in adv.steps] == [
        (R(0, C.PREFILL, 0, 40),), (R(0, C.DECODE, 40, 1), R(1, C.PREFILL, 32, 8, 32, False, ((0, 32),))),
        (R(1, C.DECODE, 40, 1),), (R(2, C.PREFILL, 0, 20),), (R(2, C.DECODE, 20, 1),), (R(3, C.PREFILL, 0, 16),),
        (R(2, C.PREFILL, 16, 6, 16, True, ((2, 16),)),)]
    assert adv.release == (0, 0, 0, 0, 3, 3, 3, 3)
    assert pub == C.Public((0, 0, 2, 2), 600, 1)
    assert t["admit"] == [0, 1, 3, 5]
    assert t["durations"] == [10_000, 10_000, 20_000, 10_000, 20_000, 10_000, 10_000]
    assert (t["caps"].seqs_max, t["caps"].tokens_max, t["caps"].prompt_max, t["caps"].output_max,
            t["caps"].block_size) == (8, 64, 128, 3, 16)


def test_a_trace_round_trips_under_its_caps(tmp_path):
    t = TC.load_trace(C, _write_trace(tmp_path))
    p = dict(rate=1.5, windows=1, T=600, prompt_max=1024, max_tokens=256, max_batched_tokens=1024)
    caps = C.Caps(prompt_max=1024, output_max=256, seqs_max=256, tokens_max=1024, steps_per_bucket=32, block_size=16)
    cb = M._cb_for(M.fit(M.synth_corpus_params([S.Params(seed=5, **p)]), caps, 600, 100_000), t)
    adv, pub, admit = t["advice"], t["public"], t["admit"]
    rep = C.TraceReplay(adv, pub)
    res = C.residual_of(adv, admit)
    for fam in C.FAMILIES:
        assert C.decode_v0(C.BitReader(C.encode_v0(adv, pub, cb, fam)[0].bits), pub, cb, fam).steps == adv.steps
        assert C.decode_admit(C.BitReader(C.encode_admit(admit, pub, cb, fam, rep)[0].bits), pub, cb, fam, rep) == admit
        assert C.decode_residual(C.BitReader(C.encode_residual(res, pub, cb, fam, rep)[0].bits), pub, cb, fam, rep) == res
    names = [name for name, _, _, _ in C.symbols_v0(adv, pub, cb, "cond")[0]]
    assert "rehit_new" in names and "start_delta" not in names  # every step, the resume included, is in the predicted form


def test_a_stop_string_is_refused(tmp_path):
    d = _write_trace(tmp_path)
    lines = (d / "requests.jsonl").read_text().splitlines()
    q = json.loads(lines[1])
    q["eos"] = False
    (d / "requests.jsonl").write_text("\n".join([lines[0], json.dumps(q), *lines[2:]]) + "\n")
    with pytest.raises(ValueError, match="I1's finish can't say"):
        TC.load_trace(C, d)
