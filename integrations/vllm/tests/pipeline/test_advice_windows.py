"""Advice-driven serving-rows windows (`pipeline.advice_windows`) on the CPU TP Commit of `tests.pipeline.served_tp`, deferred: each
rank's stand-in from its sealed bundle over the stand-in compiler's Programs for the advice's requests, its registration binding the
compiler's digest and the root of the advice's hiding commitment, held to the verifier's own window from the record, the advice's
opening (I1) and the compiler alone."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from verity.core.protocols.certifier import advice as A
from verity.core.protocols.sampled_proofs.one_stage import registration as R
from verity.ml.windows import rows as CS
from verity_vllm.pipeline import advice_windows as AW
from verity_vllm.pipeline.tp import serving_rows as SRT


def _by_id(reqs: list[dict]) -> list[dict]:
    return sorted(reqs, key=lambda q: q["request_id"])


def _requests(c, drop=(), extra=()) -> list[dict]:
    doc = json.loads(c["workload"].read_text())
    reqs = [{"request_id": f"r{i}", "arrive_step": int(q.get("arrive_step") or 0), "prompt_len": len(q["prompt_token_ids"]),
             "max_tokens": int(q["max_tokens"])} for i, q in enumerate(doc["requests"])]
    return [q for q in reqs if q["request_id"] not in drop] + [dict(reqs[0], **x) for x in extra]


def _advice(c, drop=()) -> bytes:
    return A.encode(AW.advice_of_workload(json.loads(c["workload"].read_text()), drop=drop))


@pytest.fixture(scope="module")
def advice_tp(tmp_path_factory):
    """The deferred TP Commit, its stand-in compiler, and each rank's window under advice A (the stand-in's advice for the workload)
    and under advice A' (A less request r1), each from the rank's bundle."""
    pytest.importorskip("torch")
    from tests.pipeline import served_tp as ST

    root = tmp_path_factory.mktemp("advice-tp")
    c = ST.commit(root / "tp", world=2, I=256, law="subset:2", deferred=True)
    c["compiler"] = AW.Traced.for_commit(c["commit"], c["build"])
    c["bytes"] = {"A": _advice(c), "A'": _advice(c, drop=("r1",))}
    c["advice"], c["reg"], c["opening"] = {}, {}, {}
    for name, data in c["bytes"].items():
        reqs, bound, reg = AW.of_advice(c["compiler"], data)
        c["advice"][name], c["reg"][name], c["opening"][name] = (reqs, bound), reg, AW.opening(data, reg)
    c["windows"] = {name: root / name for name in c["advice"]}
    for name, (reqs, bound) in c["advice"].items():
        for r in range(2):
            AW.commit_window(c["commit"], r, c["partition"], c["windows"][name] / f"rank{r}", c["compiler"], reqs, bound, workers=1)
    return c


def _registered(c, name: str, rank: int) -> dict:
    return json.loads((c["windows"][name] / f"rank{rank}" / "registration.json").read_text())


def _mine(c, name: str, rank: int, opening: dict | None = None):
    """The verifier's window of rank ``rank`` from the opening of advice ``name`` (or ``opening``)."""
    reqs, bound = AW.of_opening(c["compiler"], opening if opening is not None else c["opening"][name])
    return AW.window_of_advice(c["commit"], rank, c["partition"], c["compiler"], reqs, bound)


def _resalted(opening: dict) -> dict:
    s = opening["salts"]
    return {**opening, "salts": [f"{int(s[0][:2], 16) ^ 1:02x}{s[0][2:]}", *s[1:]]}


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
        assert w["compiler"] == {"name": AW.TRACED, "digest": c["compiler"].digest}
        assert w["advice"] == {"root": c["reg"]["A"].root.hex(), "bits": A.total_bits(A.decode(c["bytes"]["A"]))}
        assert 8 * len(c["bytes"]["A"]) - 8 < w["advice"]["bits"] <= 8 * len(c["bytes"]["A"])
        assert SRT.source(w, mine, held)["ok"]
        assert _check(rec, mine, _publics(c["windows"]["A"] / f"rank{r}", mine)).ok
        assert [(m.template, m.n) for m in mine.members] == [("SiluMul_v2{I=256}", 8)]


def test_the_advice_is_committed_hiding_and_none_of_its_bytes_is_public(advice_tp):
    """The registration binds a root of hiding rows, not a digest of the advice: the same advice under fresh salts has another root,
    each opening to the same requests, neither the SHA-512 of the advice, and no window file holds the advice."""
    c = advice_tp
    data = c["bytes"]["A"]
    again_reqs, again, again_reg = AW.of_advice(c["compiler"], data)
    assert again["advice"]["root"] != c["advice"]["A"][1]["advice"]["root"] and again["advice"]["bits"] == c["advice"]["A"][1]["advice"]["bits"]
    assert AW.of_opening(c["compiler"], AW.opening(data, again_reg)) == (again_reqs, again) and again_reqs == c["advice"]["A"][0]
    assert hashlib.sha512(data).hexdigest() not in {again["advice"]["root"], c["advice"]["A"][1]["advice"]["root"]}
    assert len(AW.advice_rows(data)) == -(-again["advice"]["bits"] // AW.ROW_BITS) == len(again_reg.salts)
    for f in (c["windows"]["A"] / "rank0").iterdir():
        if f.suffix == ".json" or f.name.endswith("-public.bin"):
            assert data.hex() not in f.read_bytes().hex() and data.hex() not in f.read_text(errors="replace"), f.name


def test_an_opening_under_another_salt_opens_no_registered_root(advice_tp):
    """The advice A opened under one salt changed recomputes another root: the verifier's window is A's, and the registration is
    refused on its advice alone."""
    c = advice_tp
    mine, held = _mine(c, "A", 0, _resalted(c["opening"]["A"]))
    assert mine.population == 8 and held["advice"]["root"] != c["reg"]["A"].root.hex()
    got = SRT.source(_registered(c, "A", 0)["window"], mine, held)
    assert (got["ok"], got["codes"]) == (False, ["advice"])


@pytest.mark.parametrize("alter, why", [
    (lambda o: {**o, "scheme": "verity-vllm/advice-windows/advice/v0"}, "advice opening is a"),
    (lambda o: {k: v for k, v in o.items() if k != "salts"}, "in hex"),
    (lambda o: {**o, "salts": o["salts"] + o["salts"]}, "one salt per row"),
    (lambda o: {**o, "salts": [s[:-2] for s in o["salts"]]}, "salt is"),
    (lambda o: {**o, "advice": o["advice"] + "00"}, "past the advice"),
], ids=["other-scheme", "no-salts", "a-salt-too-many", "a-short-salt", "non-canonical-advice"])
def test_a_malformed_opening_is_refused(advice_tp, alter, why):
    with pytest.raises(ValueError, match=why):
        AW.of_opening(advice_tp["compiler"], alter(advice_tp["opening"]["A"]))


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
    (lambda w: {**w, "advice": {**w["advice"], "root": "00" * 64}}, ["advice"]),
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
        AW.commit_window(c["commit"], 0, c["partition"], tmp_path / "w", c["compiler"], reqs, AW.binding(c["compiler"], "00" * 64, 8), workers=1)
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


def test_the_stand_in_advice_states_the_workloads_requests_at_their_arrivals(advice_tp):
    """I1 advice of the workload: each request's whole-prompt prefill at its arrival, then a decode row a step to its cap, every
    output kept and released in its step's bucket; the compiler's requests read back from it are the workload's."""
    c = advice_tp
    advice = A.decode(c["bytes"]["A"])
    assert _by_id(AW.requests_of(advice)) == _requests(c) and _by_id(c["advice"]["A'"][0]) == _requests(c, drop=("r1",))
    rows = {(s, advice.requests[r.request].id, r.kind, r.start, r.n) for s, st in enumerate(advice.steps) for r in st.rows}
    for q in _requests(c):
        rid, lp, t, a = q["request_id"], q["prompt_len"], q["max_tokens"], q["arrive_step"]
        assert {x for x in rows if x[1] == rid} == {(a, rid, "prefill", 0, lp)} | {(a + j, rid, "decode", lp + j - 1, 1) for j in range(1, t)}
    assert [o.step for o in A.outputs(advice)] == list(advice.release) and {q.finish for q in advice.requests} == {"length"}


def test_advice_is_read_only_from_its_canonical_encoding(advice_tp):
    c = advice_tp
    for bad in (b"\x01" + c["bytes"]["A"][1:], c["bytes"]["A"] + b"\0"):
        with pytest.raises(ValueError):
            AW.of_advice(c["compiler"], bad)


def _served(*steps, release) -> A.Advice:
    """I1 advice of q0 (3 prompt tokens, 2 kept outputs) and q1 (1 and 1), each step's rows given as (request, start, n)."""
    reqs = (A.RequestAdvice("q0", 3, 2, "length"), A.RequestAdvice("q1", 1, 1, "length"))
    rows = [[A.Row(q, "decode" if n == 1 and start >= reqs[q].prompt_len else "prefill", start, n) for q, start, n in step] for step in steps]
    return A.Advice(reqs, tuple(A.StepAdvice(tuple(r)) for r in rows), release)


def test_the_row_async_scheduling_runs_past_the_last_kept_output_is_no_program_row():
    advice = _served([(0, 0, 3)], [(0, 3, 1), (1, 0, 1)], [(0, 4, 1)], release=(0, 1, 1))
    assert AW.refusals(advice) == {}
    assert AW.requests_of(advice) == [{"request_id": "q0", "arrive_step": 0, "prompt_len": 3, "max_tokens": 2},
                                      {"request_id": "q1", "arrive_step": 1, "prompt_len": 1, "max_tokens": 1}]


def test_the_report_on_served_advice_binds_it_and_names_what_the_stand_in_refuses(advice_tp):
    """`report`: I1's round trip, the binding a registration would carry, the charge per window and the refused requests by id, with
    none of the advice's bytes."""
    c = advice_tp
    advice = _served([(0, 0, 2)], [(0, 2, 1), (1, 0, 1)], [(0, 3, 1)], release=(1, 1, 2))
    data = A.encode(advice)
    got = AW.report(c["compiler"], data, 2)
    assert got["binding"]["compiler"] == c["advice"]["A"][1]["compiler"] and got["binding"]["advice"]["bits"] == A.total_bits(advice)
    assert got["binding"]["advice"]["root"] not in (AW.report(c["compiler"], data, 2)["binding"]["advice"]["root"], hashlib.sha512(data).hexdigest())
    assert (got["bits_by_window"], got["requests"], got["compiled"], got["refused"]) == (
        {str(w): b for w, b in A.bits_by_window(advice, 2).items()}, 2, 1, {"q0": "a chunked prefill, 2 of its 3 prompt tokens at step 0"})
    assert sum(got["bits_by_window"].values()) == A.total_bits(advice) and data.hex() not in json.dumps(got)


@pytest.mark.parametrize("steps, release, why", [
    pytest.param(([(0, 0, 2)], [(0, 2, 1), (1, 0, 1)], [(0, 3, 1)]), (1, 1, 2), "a chunked prefill, 2 of its 3 prompt tokens at step 0",
                 id="chunked-prefill"),
    pytest.param(([(0, 0, 3)], [(1, 0, 1)], [(0, 3, 1)]), (0, 1, 2), r"its row at step 2 is \(start, n\) = \(3, 1\), not the next decode",
                 id="a-gap"),
])
def test_a_request_the_stand_in_cannot_compile_refuses_the_advice_by_name(steps, release, why):
    """Served advice the stand-in's Programs of record don't run (COMPILER's v0 refusals): refused naming the request and why, never
    compiled as if its prompt ran whole."""
    advice = _served(*steps, release=release)
    assert list(AW.refusals(advice)) == [0]
    with pytest.raises(ValueError, match=rf"compiles 1 of the advice's 2 requests \(q0: {why}\)"):
        AW.requests_of(advice)


def test_the_stand_in_advice_refuses_a_workload_that_leaves_a_step_idle():
    doc = {"requests": [{"prompt_token_ids": [1, 2], "max_tokens": 2, "arrive_step": 0}, {"prompt_token_ids": [3], "max_tokens": 1, "arrive_step": 3}]}
    with pytest.raises(ValueError, match=r"engine steps \[2\] idle"):
        AW.advice_of_workload(doc)
    doc["requests"][1]["arrive_step"] = 1
    assert [len(s.rows) for s in AW.advice_of_workload(doc).steps] == [1, 2]


def test_the_command_writes_the_stand_in_advice_and_commits_its_windows(advice_tp, tmp_path):
    """``verity-vllm advice-windows`` with ``--write-from-workload --drop r1``: A' written, its opening written, and each rank's
    window bound to the root that opening opens, under its own fresh salts."""
    from verity_vllm.pipeline import cli

    c = advice_tp
    o = cli.parse("advice-windows", [str(c["commit"]), "--advice", str(tmp_path / "a.bin"), "--opening", str(tmp_path / "a.opening.json"),
                                     "--partition", str(c["partition"]), "--out", str(tmp_path / "w"), "--build-root", str(c["build"]),
                                     "--write-from-workload", str(c["workload"]), "--drop", "r1", "--workers", "1"], environ={})
    assert AW.main(o) == 0 and (tmp_path / "a.bin").read_bytes() == c["bytes"]["A'"]
    reqs, opened = AW.of_opening(c["compiler"], json.loads((tmp_path / "a.opening.json").read_text()))
    assert reqs == c["advice"]["A'"][0]
    for r in range(2):
        got, want = (json.loads((d / f"rank{r}" / "registration.json").read_text()) for d in (tmp_path / "w", c["windows"]["A'"]))
        assert (got["window"]["compiler"], got["window"]["advice"], got["population"]) == (want["window"]["compiler"], opened["advice"], 4)
        assert got["window"]["advice"]["root"] != want["window"]["advice"]["root"]


def test_the_stand_in_compilers_digest_names_its_programs_of_record_and_its_transfer_gates(advice_tp):
    c = advice_tp["compiler"]
    other = AW.Traced(c.programs, {**c.of_record, "0": ["00" * 32]})
    assert len(c.digest) == 128 and other.digest != c.digest and AW.Traced(dict(c.programs), dict(c.of_record)).digest == c.digest
    gated = AW.Traced(c.programs, c.of_record, transfer_gates=(ROPE,))
    assert gated.digest not in (c.digest, other.digest) and gated.digest == AW.Traced(c.programs, c.of_record, transfer_gates=(ROPE,)).digest


ROPE = "RoPEHead_v1{D=64}"
SILU = "SiluMul_v2{I=256}"


@pytest.fixture(scope="module")
def gated_tp(tmp_path_factory):
    """The deferred TP Commit whose partition holds the toy's RoPE heads before its SiluMul rows under ``work:2``, the stand-in
    compiler listing the RoPE heads as its transfer gates (and the same compiler listing none), and rank 0's window under the
    workload's advice from each."""
    pytest.importorskip("torch")
    from tests.pipeline import served_tp as ST

    root = tmp_path_factory.mktemp("gated-tp")
    c = ST.commit(root / "tp", world=2, I=256, law="work:2", deferred=True, templates=(ROPE,))
    data = _advice(c)
    for name, gates in (("gated", (ROPE,)), ("bare", ())):
        comp = AW.Traced.for_commit(c["commit"], c["build"], gates)
        reqs, bound, reg = AW.of_advice(comp, data)
        AW.commit_window(c["commit"], 0, c["partition"], root / name / "rank0", comp, reqs, bound, workers=1)
        c[name] = {"compiler": comp, "opening": AW.opening(data, reg), "dir": root / name / "rank0",
                   "rec": json.loads((root / name / "rank0" / "registration.json").read_text())}
    return c


def _gated_mine(c, name: str = "gated"):
    comp = c[name]["compiler"]
    return AW.window_of_advice(c["commit"], 0, c["partition"], comp, *AW.of_opening(comp, c[name]["opening"]))


class _Coins:
    """Deterministic coin bytes for a test's sample: SHA-512 in counter mode."""

    def __init__(self) -> None:
        self.at = 0

    def __call__(self, n: int) -> bytes:
        out = b""
        while len(out) < n:
            out += hashlib.sha512(self.at.to_bytes(8, "big")).digest()
            self.at += 1
        return out[:n]


def _draw(rec: dict, law: dict) -> dict:
    from verity.core.protocols.sampled_proofs import service as S

    return S.select_draw(S.Law(f"work:{law['k']}"), law, rec["population"], _Coins())


def test_the_compilers_transfer_gate_units_are_drawn_whole_beside_the_drawn_rest(gated_tp):
    """The window's work law proves every RoPE head (work 0, its floor all 32: an All stratum) and draws 2 of the 8 SiluMul rows, as
    the registration's law states and the verifier's own window derives; a sample by it holds every transfer-gate unit."""
    from verity.core.protocols.sampled_proofs.one_stage import draw as D

    c = gated_tp
    mine, held = _gated_mine(c)
    rec = c["gated"]["rec"]
    assert [(m.template, m.n) for m in mine.members] == [(ROPE, 32), (SILU, 8)]
    assert mine.work_table == {ROPE: {"work": 0, "floor": 32}, SILU: {"work": 1, "floor": 1}}
    assert rec["law"] == {"law": "work", "k": 2, "strata": [{"template": ROPE, "units": [[0, 32]], "work": 0, "floor": 32, "k": 32},
                                                            {"template": SILU, "units": [[32, 40]], "work": 1, "floor": 1, "k": 2}]}
    assert json.loads((c["gated"]["dir"] / "partition.json").read_text())["work_table"] == mine.work_table
    assert SRT.source(rec["window"], mine, held)["ok"]
    v = R.check(rec, mine.program, mine.query, mine.law, CS.SCHEME, CS.expected_ports(mine), CS.leaf_layer(_publics(c["gated"]["dir"], mine)),
                work_table=mine.work_table)
    assert v.ok, v.codes
    draw = _draw(rec, rec["law"])
    assert set(range(32)) <= set(draw["units"]) and len(draw["units"]) == 34 and D.check_draw(draw, rec, rec["law"]).ok


def test_a_transfer_gate_unit_left_out_of_the_proved_set_is_refused(gated_tp):
    """Each way of leaving a RoPE head unproved: the proved set less one (`check_proved`), a sample trading one for an undrawn SiluMul
    row (`check_draw`, the stratum's count), and a registration whose law draws the RoPE heads as the rest is (R1)."""
    from verity.core.protocols.sampled_proofs.one_stage import draw as D

    c = gated_tp
    mine, _held = _gated_mine(c)
    rec = c["gated"]["rec"]
    draw = _draw(rec, rec["law"])
    left = [u for u in draw["units"] if u != 0]
    assert not D.check_proved(draw, left, len(left)).ok and D.check_proved(draw, draw["units"], len(draw["units"])).ok
    spare = next(u for u in range(32, 40) if u not in draw["units"])
    v = D.check_draw({**draw, "units": sorted(set(draw["units"]) - {0} | {spare})}, rec, rec["law"])
    assert (v.ok, list(v.codes)) == (False, ["strata-count"])
    sampled = {t: {"work": 1, "floor": 1} for t in mine.work_table}
    v = R.check({**rec, "law": D.derive(mine.program, mine.query, mine.law, sampled)}, mine.program, mine.query, mine.law, CS.SCHEME,
                CS.expected_ports(mine), CS.leaf_layer(_publics(c["gated"]["dir"], mine)), work_table=mine.work_table)
    assert (v.ok, list(v.codes)) == (False, ["R1-law"])


def test_a_window_under_the_compiler_listing_no_transfer_gates_is_refused(gated_tp):
    """The prover's window committed under the same Programs by a compiler that lists no transfer gates draws 2 of the RoPE heads and
    1 SiluMul row: the verifier, holding the registered compiler that lists them, refuses it on the compiler's digest, on the advice's
    root (its tree's domain binds the compiler, `commitment_binding`) and on R1."""
    c = gated_tp
    mine, held = _gated_mine(c)
    bare = c["bare"]["rec"]
    assert [s["k"] for s in bare["law"]["strata"]] == [2, 1]
    got = SRT.source(bare["window"], mine, held)
    assert (got["ok"], got["codes"]) == (False, ["compiler", "advice"])
    v = R.check(bare, mine.program, mine.query, mine.law, CS.SCHEME, CS.expected_ports(mine), CS.leaf_layer(_publics(c["bare"]["dir"], mine)),
                work_table=mine.work_table)
    assert not v.ok and "R1-law" in v.codes


def test_a_partition_that_leaves_out_or_samples_a_transfer_gate_template_is_refused(gated_tp, tmp_path):
    from verity_vllm.pipeline import serving_rows as SRW

    c = gated_tp
    comp, opening = c["gated"]["compiler"], c["gated"]["opening"]
    for doc, why in (({"templates": [SILU], "law": "work:2"}, "would be left out of the proved set"),
                     ({"templates": [ROPE, SILU], "law": "stratified:2"}, r"work law's draw \(law work:K\)"),
                     ({"templates": [ROPE], "law": "work:2"}, "its selection is all")):
        (pf := tmp_path / "partition.json").write_text(json.dumps(doc))
        with pytest.raises(ValueError, match=why):
            AW.window_of_advice(c["commit"], 0, pf, comp, *AW.of_opening(comp, opening))
    assert SRW.work_table_of({"law": "work", "k": 2}, {ROPE: 32, SILU: 8}) == {ROPE: {"work": 1, "floor": 1}, SILU: {"work": 1, "floor": 1}}
    assert SRW.work_table_of({"law": "subset", "k": 2}, {ROPE: 32, SILU: 8}) is None


@pytest.mark.parametrize("root, bits, why", [
    ("00" * 32, 8, "advice's root is 64 bytes"),
    ("00" * 64, -1, "whole number of bits"),
    ("00" * 64, True, "whole number of bits"),
])
def test_a_binding_is_a_compiler_digest_an_advice_root_and_a_length(advice_tp, root, bits, why):
    with pytest.raises(ValueError, match=why):
        AW.binding(advice_tp["compiler"], root, bits)


def test_a_compiler_without_transfer_gates_lists_none():
    class Bare:
        name, digest = "bare", "ab" * 64

    class Listing(Bare):
        transfer_gates = ["RoPEHead_v1{D=64}"]

    assert AW.transfer_gates_of(Bare()) == ()
    assert AW.transfer_gates_of(Listing()) == ("RoPEHead_v1{D=64}",)
