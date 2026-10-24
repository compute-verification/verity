"""Advice v1: `decode(encode(a)) == a` on the replay's step traces and a TRACE-CORPUS trace, the strings are prefix-free
(so the charge meets LEAK-LEAN's `BitsCount`), and `decode` is total, refusing by name everything but a canonical
string."""

from __future__ import annotations

import dataclasses
import gzip
import importlib.util
import json
import random
import shutil
import sys
from itertools import product
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str):
    key = f"enc_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, HERE / f"{name}.py")
        sys.modules[key] = mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return sys.modules[key]


V, R, C = _load("v1"), _load("replay"), _load("code")
from verity.core.protocols.certifier import advice as I1  # noqa: E402

CASES = sorted(p.stem for p in FIXTURES.glob("*.jsonl"))
EOS = 2


def step_trace(name: str):
    """A replay fixture (`test_replay`'s), each request given tokens where the trace has none: its own prompt id, and
    outputs that end on the stop id exactly when it stopped there."""
    tok = FIXTURES / f"{name}.tokens.json"
    t = R.read_step_trace(FIXTURES / f"{name}.jsonl", tokens=json.loads(tok.read_text()) if tok.exists() else None)
    t.requests = [r if r.tokens is not None else dataclasses.replace(
        r, tokens=(1000 + i,) * r.prompt_len + (3,) * (r.output_len - r.eos) + (EOS,) * r.eos)
        for i, r in enumerate(t.requests)]
    return t


def case(name: str, **kw):
    t = step_trace(name)
    rec = V.record_of(t, (EOS,))
    reg = V.Registered(t.engine, (EOS,), row_order="scheduling", **kw)
    return t, rec, reg, V.advice_of(t, rec)


def exact_bits(data: bytes, rec, reg) -> bytearray:
    return C.BitReader.from_bytes(data, sum(V.bits_by_window(data, rec, reg).values())).bits


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("contexts", R.CONTEXTS)
def test_decode_inverts_encode(name, contexts):
    t, rec, reg, adv = case(name, T=20, contexts=contexts)
    data = V.encode(adv, rec, reg)
    assert V.decode(data, rec, reg) == adv
    charge = V.bits_by_window(data, rec, reg)
    assert len(charge) == V.windows(rec, reg.T) > 1
    assert 0 <= 8 * len(data) - sum(charge.values()) < 8
    assert sum(charge.values()) < I1.total_bits(adv) // 100
    assert V.decode_bits(exact_bits(data, rec, reg), rec, reg) == adv


def test_the_case_set_covers_hits_resumes_and_async():
    advs = {name: case(name)[3] for name in CASES}
    rows = [x for a in advs.values() for st in a.steps for x in st.rows]
    assert any(x.sources for x in rows) and any(x.preempted for x in rows)
    assert {case(n)[0].engine.async_scheduling for n in CASES} == {False, True}


def _corpus(tmp_path: Path) -> Path:
    """ENC-DERIVED's TRACE-CORPUS fixture (4 s at 12/s on 100 KV blocks, prefix caching, preemptions) with each batch in
    Model Runner V2's order (the scheduler's, sorted stably by n) and each stop-id finish marked `eos`."""
    src = FIXTURES / "corpus-smallkv"
    for f in ("meta.json", "server.log"):
        shutil.copy(src / f, tmp_path / f)
    out = []
    for line in gzip.decompress((src / "steps.jsonl.gz").read_bytes()).decode().splitlines():
        s = json.loads(line)
        if s.get("batch"):
            sched = ([r for r in s["cached"] if not r["resumed"]] + [r for r in s["cached"] + s["new"] if r["resumed"]]
                     + [r for r in s["new"] if not r["resumed"]])
            s["batch"] = [r["id"] for r in sorted((r for r in sched if r["id"] in set(s["batch"])),
                                                  key=lambda r: r["num_scheduled_tokens"])]
        out.append(json.dumps(s) + "\n")
    (tmp_path / "steps.jsonl").write_text("".join(out))
    reqs = [json.loads(line) for line in gzip.decompress((src / "requests.jsonl.gz").read_bytes()).decode().splitlines()]
    (tmp_path / "requests.jsonl").write_text("".join(json.dumps({**q, "eos": q["finish_reason"] == "stop"}) + "\n"
                                                     for q in reqs))
    return tmp_path


def test_a_trace_corpus_trace_round_trips(tmp_path):
    stops = (128009, 128001, 128008)
    t = R.read_corpus_trace(_corpus(tmp_path), ingress="warden")
    rec = V.record_of(t, stops)
    adv = V.advice_of(t, rec)
    rows = [x for st in adv.steps for x in st.rows]
    assert any(x.preempted for x in rows) and any(x.sources for x in rows)
    for contexts in ("plain", "egress"):
        reg = V.Registered(t.engine, stops, contexts=contexts)
        data = V.encode(adv, rec, reg)
        assert V.decode(data, rec, reg) == adv
        assert sum(V.bits_by_window(data, rec, reg).values()) < 200
    with pytest.raises(V.V1Error, match="outside v1"):
        V.encode(adv, rec, V.Registered(t.engine, stops, row_order="scheduling"))


def _tiny(windows: int):
    """Three requests into an engine of two sequences and an 8-token budget, so the replay asks about each arrival."""
    e = R.Engine(max_num_seqs=2, max_num_batched_tokens=8, max_model_len=64, num_gpu_blocks=32, block_size=4)
    reqs = tuple(V.Request(f"u{k}", (10 + k,) * 6, 4, (5, 5, EOS), end, (end, end, end + 1))
                 for k, end in enumerate((0, 0, 3 * (windows - 1))))
    return V.Record(reqs), V.Registered(e, (EOS,), T=3, contexts="plain", pinned=True)


@pytest.mark.parametrize("w", (1, 2))
def test_the_strings_are_prefix_free(w):
    """Every bit string of up to 18 bits through `decode_bits`: no accepted one is a proper prefix of another, so
    #{accepted strings of at most c bits} ≤ 2^c (Kraft), LEAK-LEAN's `BitsCount` for the charge `bits_by_window`; and
    each accepted one is the encoding of the advice it decodes to."""
    rec, reg = _tiny(w)
    accepted = {}
    for n in range(19):
        for bits in product((0, 1), repeat=n):
            try:
                accepted[bits] = V.decode_bits(bits, rec, reg)
            except V.V1Error:
                pass
    assert len(set(accepted.values())) == len(accepted) >= 3
    strings = sorted(accepted)
    assert not any(b[:len(a)] == a for a, b in zip(strings, strings[1:]))
    assert sum(2.0 ** -len(s) for s in strings) <= 1
    for c in range(19):
        assert sum(len(s) <= c for s in strings) <= 2 ** c
    for s, adv in accepted.items():
        data = V.encode(adv, rec, reg)
        assert exact_bits(data, rec, reg) == bytearray(s) and sum(V.bits_by_window(data, rec, reg).values()) == len(s)


def test_truncated_and_extended_strings_are_refused():
    t, rec, reg, adv = case("async-prefix-cache", T=20)
    bits = exact_bits(V.encode(adv, rec, reg), rec, reg)
    for n in range(len(bits)):
        with pytest.raises(V.V1Error):
            V.decode_bits(bits[:n], rec, reg)
    for b in (0, 1):
        with pytest.raises(V.V1Error, match="past the last window"):
            V.decode_bits(bits + bytearray([b]), rec, reg)
    data = V.encode(adv, rec, reg)
    bad = [data[:-1], data + b"\0"]
    if 8 * len(data) > len(bits):
        bad.append(data[:-1] + bytes([data[-1] | 1]))
    for x in bad:
        with pytest.raises(V.V1Error):
            V.decode(x, rec, reg)


def test_decode_is_total():
    """Random strings and every one-bit change of an encoding: `V1Error`, or an advice whose encoding they are under
    the pool their header sends."""
    rng = random.Random(5)
    t, rec, reg, adv = case("sync-kv-pressure", T=20)
    data = V.encode(adv, rec, reg)
    tries = [bytes(rng.randrange(256) for _ in range(rng.randrange(40))) for _ in range(200)]
    tries += [bytes(b ^ (1 << j if k == i else 0) for k, b in enumerate(data)) for i in range(len(data)) for j in range(8)]
    decoded = 0
    for x in tries:
        try:
            got = V.decode(x, rec, reg)
        except V.V1Error:
            continue
        decoded += 1
        assert V.encode(got, rec, V.registered_of(x, rec, reg)) == x
    assert decoded


def test_refuses_over_the_cap_and_off_the_record():
    t, rec, reg, adv = case("async-light", T=20)
    data = V.encode(adv, rec, reg)
    with pytest.raises(V.V1Error, match="cap"):
        V.decode(data, rec, dataclasses.replace(reg, cap_bits=8 * len(data) - 1))
    with pytest.raises(V.V1Error, match="pool"):
        V.encode(adv, rec, dataclasses.replace(reg, engine=dataclasses.replace(t.engine, num_gpu_blocks=2)))
    with pytest.raises(V.V1Error, match="bytes"):
        V.decode(list(data), rec, reg)
    moved = I1.Advice(adv.requests, adv.steps, adv.release[:-1] + (adv.release[-1] + 1,))
    with pytest.raises(V.V1Error, match="another advice"):
        V.encode(moved, rec, reg)
    s = next(k for k, st in enumerate(adv.steps) if len(st.rows) > 1 and k > 0
             and all(any(x.request == y.request for z in adv.steps[:k] for y in z.rows) for x in st.rows))
    swapped = I1.Advice(adv.requests, adv.steps[:s] + (I1.StepAdvice(adv.steps[s].rows[::-1]),) + adv.steps[s + 1:],
                        adv.release)
    with pytest.raises(V.V1Error, match="outside v1"):
        V.encode(swapped, rec, reg)
