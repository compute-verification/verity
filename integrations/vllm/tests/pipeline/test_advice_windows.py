"""Advice-driven serving-rows windows (`pipeline.advice_windows`) on the CPU TP Commit of `tests.pipeline.served_tp`, deferred: each
rank's stand-in from its sealed bundle over the stand-in compiler's Programs for the advice's requests, its registration binding the
compiler's and the advice's digests, held to the verifier's own window from the record, the advice and the compiler alone."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from verity.core.protocols.sampled_proofs.one_stage import registration as R
from verity_vllm.commit import serving_rows as CS
from verity_vllm.pipeline import advice_windows as AW
from verity_vllm.pipeline.tp import serving_rows as SRT


def _requests(c, drop=(), extra=()) -> list[dict]:
    doc = json.loads(c["workload"].read_text())
    reqs = [{"request_id": f"r{i}", "arrive_step": int(q.get("arrive_step") or 0), "prompt_len": len(q["prompt_token_ids"]),
             "max_tokens": int(q["max_tokens"])} for i, q in enumerate(doc["requests"])]
    return [q for q in reqs if q["request_id"] not in drop] + [dict(reqs[0], **x) for x in extra]


def _bound(compiler, advice: bytes) -> dict:
    return AW.binding(compiler, hashlib.sha512(advice).hexdigest(), 8 * len(advice))


@pytest.fixture(scope="module")
def advice_tp(tmp_path_factory):
    """The deferred TP Commit, its stand-in compiler, and each rank's window under advice A (the workload's requests) and under
    advice A' (A less request r1), each from the rank's bundle."""
    pytest.importorskip("torch")
    from tests.pipeline import served_tp as ST

    root = tmp_path_factory.mktemp("advice-tp")
    c = ST.commit(root / "tp", world=2, I=256, law="subset:2", deferred=True)
    c["compiler"] = AW.Traced.for_commit(c["commit"], c["build"])
    c["advice"] = {"A": (_requests(c), _bound(c["compiler"], b"advice A")),
                   "A'": (_requests(c, drop=("r1",)), _bound(c["compiler"], b"advice A'"))}
    c["windows"] = {name: root / name for name in c["advice"]}
    for name, (reqs, bound) in c["advice"].items():
        for r in range(2):
            AW.commit_window(c["commit"], r, c["partition"], c["windows"][name] / f"rank{r}", c["compiler"], reqs, bound, workers=1)
    return c


def _registered(c, name: str, rank: int) -> dict:
    return json.loads((c["windows"][name] / f"rank{rank}" / "registration.json").read_text())


def _mine(c, name: str, rank: int):
    reqs, bound = c["advice"][name]
    return AW.window_of_advice(c["commit"], rank, c["partition"], c["compiler"], reqs, bound)


def _check(rec: dict, mine, publics: dict[str, bytes]):
    return R.check(rec, mine.program, mine.query, mine.law, CS.SCHEME, CS.expected_ports(mine), CS.leaf_layer(publics))


def _publics(d: Path, mine) -> dict[str, bytes]:
    return {m.key: (d / f"m{k}-public.bin").read_bytes() for k, m in enumerate(mine.members)}


def test_each_ranks_advice_window_is_held_to_the_verifiers_window_from_the_advice(advice_tp):
    c = advice_tp
    reqs, bound = c["advice"]["A"]
    for r in range(2):
        mine, held = _mine(c, "A", r)
        rec = _registered(c, "A", r)
        w = rec["window"]
        assert (w["kind"], w["run"], w["compiler"], w["advice"]) == ("stand-in", mine.run, bound["compiler"], bound["advice"])
        assert w["compiler"] == {"name": AW.TRACED, "digest": c["compiler"].digest} and w["advice"]["bits"] == 8 * len(b"advice A")
        assert SRT.source(w, mine, held)["ok"]
        assert _check(rec, mine, _publics(c["windows"]["A"] / f"rank{r}", mine)).ok
        assert [(m.template, m.n) for m in mine.members] == [("SiluMul_v2{I=256}", 8)]


def test_the_advice_window_commits_the_rows_of_the_ranks_window_of_record(advice_tp, tmp_path):
    """Under advice stating the requests the Commit served, the compiled window is the record's window: the same population, the
    same partition and the same committed rows, only its provenance naming the compiler and the advice."""
    c = advice_tp
    for r in range(2):
        mine, _held = _mine(c, "A", r)
        of_record, _ = SRT.window_of_record(c["commit"], r, c["partition"], c["build"], c["workload"])
        assert (mine.run, mine.members, mine.partition_digest) == (of_record.run, of_record.members, of_record.partition_digest)
    SRT.from_bundle(c["commit"], 0, c["partition"], tmp_path / "rank0", workers=1)
    rows = []
    for d in (c["windows"]["A"] / "rank0", tmp_path / "rank0"):
        m = json.loads((d / "window.json").read_text())["members"][0]
        body = (d / "m0-private.bin").read_bytes()
        rows.append(body[:len(body) - m["n"] * len(m["ports"]) * CS.SALT_BYTES])
    assert rows[0] == rows[1] and rows[0]


@pytest.mark.parametrize("alter, codes", [
    (lambda w: {**w, "compiler": {**w["compiler"], "digest": "00" * 64}}, ["compiler"]),
    (lambda w: {**w, "compiler": {**w["compiler"], "name": "another compiler"}}, ["compiler"]),
    (lambda w: {**w, "advice": {**w["advice"], "digest": hashlib.sha512(b"advice A'").hexdigest()}}, ["advice"]),
    (lambda w: {**w, "advice": {**w["advice"], "bits": w["advice"]["bits"] - 1}}, ["advice"]),
    (lambda w: {k: v for k, v in w.items() if k not in SRT.BINDING}, ["compiler", "advice"]),
], ids=["other-compiler", "other-compiler-name", "other-advice", "other-advice-bits", "no-binding"])
def test_a_registration_that_names_another_compiler_or_advice_is_refused(advice_tp, alter, codes):
    mine, held = _mine(advice_tp, "A", 0)
    got = SRT.source(alter(_registered(advice_tp, "A", 0)["window"]), mine, held)
    assert (got["ok"], got["codes"]) == (False, codes)


def test_a_window_of_record_is_no_advice_window_and_an_advice_window_no_window_of_record(advice_tp, tmp_path):
    c = advice_tp
    SRT.from_bundle(c["commit"], 0, c["partition"], tmp_path / "rank0", workers=1)
    plain = json.loads((tmp_path / "rank0" / "registration.json").read_text())["window"]
    mine, held = _mine(c, "A", 0)
    assert SRT.source(plain, mine, held)["codes"] == ["compiler", "advice"]
    of_record, held0 = SRT.window_of_record(c["commit"], 0, c["partition"], c["build"], c["workload"])
    assert SRT.source(plain, of_record, held0)["ok"]
    assert SRT.source(_registered(c, "A", 0)["window"], of_record, held0)["codes"] == ["compiler", "advice"]


def test_advice_that_disagrees_with_the_registered_window_is_refused(advice_tp):
    """The verifier handed advice A' (A less request r1) for the window registered under A derives a window of 4 units, not 8: the
    registration fails R2 against it, and its window names A, not A', and A's served Programs."""
    c = advice_tp
    for r in range(2):
        mine, held = _mine(c, "A'", r)
        rec = _registered(c, "A", r)
        assert mine.population == 4 and rec["population"] == 8
        got = SRT.source(rec["window"], mine, held)
        assert (got["ok"], got["codes"]) == (False, ["served_programs", "advice"])
        v = _check(rec, mine, _publics(c["windows"]["A"] / f"rank{r}", _mine(c, "A", r)[0]))
        assert not v.ok and "R2-partition" in v.codes


@pytest.mark.parametrize("extra, why", [
    ({"request_id": "r9"}, "r9"),
    ({"request_id": "r0", "arrive_step": 5}, "r0"),
], ids=["unserved-request", "other-arrival"])
def test_advice_naming_rows_the_commit_did_not_commit_commits_no_window(advice_tp, tmp_path, extra, why):
    """The prover can't commit a window over advice that names a row the Commit didn't commit: a request it never served, or a served
    request at another arrival step. The committed store holds no such value, and the window isn't written."""
    c = advice_tp
    reqs = [q for q in _requests(c) if q["request_id"] != extra["request_id"]] + _requests(c, drop=("r0", "r1"), extra=(extra,))
    with pytest.raises(LookupError, match=why):
        AW.commit_window(c["commit"], 0, c["partition"], tmp_path / "w", c["compiler"], reqs, _bound(c["compiler"], b"bad"), workers=1)
    assert not (tmp_path / "w" / "registration.json").exists()


def test_a_unit_committed_under_other_advice_is_refused(advice_tp):
    """Member 0's public file from the window committed under A', spliced into A's: its rows are another commitment, so the
    registration's leaf layer over A's public bodies (R4) refuses it."""
    c = advice_tp
    mine, _held = _mine(c, "A", 0)
    rec = _registered(c, "A", 0)
    theirs = _publics(c["windows"]["A'"] / "rank0", _mine(c, "A'", 0)[0])
    spliced = dict(_publics(c["windows"]["A"] / "rank0", mine), **theirs)
    assert spliced != _publics(c["windows"]["A"] / "rank0", mine)
    v = _check(rec, mine, spliced)
    assert (v.ok, list(v.codes)) == (False, ["R4-leaf-layer"])


def test_the_stand_in_compilers_digest_names_its_programs_of_record(advice_tp):
    c = advice_tp["compiler"]
    other = AW.Traced(c.programs, {**c.of_record, "0": ["00" * 32]})
    assert len(c.digest) == 128 and other.digest != c.digest and AW.Traced(dict(c.programs), dict(c.of_record)).digest == c.digest


@pytest.mark.parametrize("digest, bits, why", [
    ("00" * 32, 8, "advice's digest is a SHA-512"),
    ("00" * 64, -1, "whole number of bits"),
    ("00" * 64, True, "whole number of bits"),
])
def test_a_binding_is_two_sha512_digests_and_a_length(advice_tp, digest, bits, why):
    with pytest.raises(ValueError, match=why):
        AW.binding(advice_tp["compiler"], digest, bits)
