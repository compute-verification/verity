"""The prove path's compiler, its advice commitment and its sample-order check, and its prover's and verifier's windows on the CPU
TP stand-in (integrations/vllm's `tests.pipeline.served_tp`, deferred), without proving.

* The compiler is total: every byte string is the requests or a `ValueError` naming why, the same on every run; random and
  truncated advice is refused, advice over the registered cap ℓ is refused before it is read, and a malformed opening is
  refused the same way.
* The advice is fixed before the sample: `advice_before_sample` holds the registration the verifier opens the advice under to the
  one the sample's receipt names, and refuses the advice re-committed after the coins.
* On the stand-in: each rank's window over the compiler's Programs for the advice is held to the verifier's own from the
  opening, the advice's negatives are refused by name, and a forged unit is committed consistently (one bit of its rows
  changed, every check before the sample passing), so only the proof can refuse it. A plain audit (`--plain`) takes its window
  only inside plain's interim scope, one table under a stratified sample (`a5.plain_refusal`).
* An accepted audit's charge terms: λ_δ, β_δ and their sum, off the audit's profile (`charge_terms`).
* A single-GPU run's pieces without its Commit: I2's adapter (its requests, its refusal by name, its Program bodies by digest),
  the requests' bodies from the record, the stand-in window's link and `source1`'s codes, and the CLI's one committed run.
* a5's R2 negative registers another population, one instance over where a template has a single one, so a one-unit window
  (the forged TP1 case at step 24) is audited through its negatives.
* A served window's pieces on a small synthetic serve-openai run dir (no Commit): its requests' bodies by I1 id, φ of the served
  engine, the registered compiler, the prover committing under the published binding (no re-commit), `serve_check`'s codes
  (a compiler without the cap, a root that isn't rows-sha512) and its negatives, and the CLI's served run.
* What no Program digest covers (a Build's row names, its parameters' names and types, its correspondence): the verifier's I2
  reads it only from its own Build of each Program, which its compile writes (`own_builds`, `naming.write_build`), whatever the
  row dir holds under the same digests, and its compile command runs with no vLLM importable."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import socket
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from verity.core.protocols.sampled_proofs.consumers import exfiltration_bound, location_bits
from verity.core.protocols.sampled_proofs.one_stage import audit as AU, partition as P, registration as R
from verity.ml.circuits.codec import canonical_json
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.types import Array, Value as Bits
from verity.ml.windows import rows as CS
from verity_vllm.engine.served_seed import DEFAULT_SEED
from verity_vllm.pipeline import advice_windows as AW
from verity_vllm.pipeline.tp import serving_rows as SRT

HERE = Path(__file__).resolve().parent
VLLM = HERE.parents[2] / "integrations" / "vllm"
sys.path.insert(0, str(HERE.parent))
import prove_path as PP  # noqa: E402

sys.path.insert(0, str(HERE.parents[1] / "one_stage"))
import a5 as A5  # noqa: E402

PROMPTS = (5, 3)


class Inner:
    """A compiler's name and digest, for the advice side alone (no Programs to load)."""

    name = "test-compiler"
    digest = "ab" * 64

    def load(self, rank, world, requests):
        raise AssertionError("the advice side loads no Program")


def _workload(tmp_path: Path, arrive: tuple[int, ...] = (0, 2), caps: tuple[int, ...] = (3, 2)) -> Path:
    f = tmp_path / "workload.json"
    f.write_text(json.dumps({"requests": [{"prompt_token_ids": list(range(lp)), "max_tokens": t, "arrive_step": s}
                                          for lp, t, s in zip(PROMPTS, caps, arrive, strict=True)]}))
    return f


def _advice(tmp_path: Path, **kw) -> bytes:
    assert PP.main(["advice", "--workload", str(_workload(tmp_path, **kw)), "--out", str(tmp_path / "advice.bin")]) == 0
    return (tmp_path / "advice.bin").read_bytes()


def _outcome(f, *args):
    """What the compiler does on an input: ("ok", its result) or ("refused", its message); any other exception fails the test."""
    try:
        return "ok", f(*args)
    except ValueError as e:
        return "refused", str(e)


def test_a_declared_runs_advice_compiles_to_the_requests_tp_commit_served(tmp_path):
    assert PP.compile_advice(PP.Capped(Inner()), _advice(tmp_path)) == [
        {"request_id": "r0", "arrive_step": 0, "prompt_len": 5, "max_tokens": 3},
        {"request_id": "r1", "arrive_step": 2, "prompt_len": 3, "max_tokens": 2}]


def test_the_cap_is_in_the_compilers_registered_digest():
    a, b = PP.Capped(Inner(), 1024), PP.Capped(Inner(), 2048)
    assert a.digest != b.digest and a.digest == PP.Capped(Inner(), 1024).digest and len(a.digest) == 128
    for bad in (True, 7, 1.5, "1024"):
        with pytest.raises(ValueError, match="the cap ℓ is a whole number of bits"):
            PP.Capped(Inner(), bad)


def test_advice_over_the_cap_is_refused_before_it_is_read(tmp_path):
    data = _advice(tmp_path)
    assert _outcome(PP.compile_advice, PP.Capped(Inner(), 8 * len(data)), data)[0] == "ok"
    for junk in (data + b"\0", b"\xff" * (len(data) + 1)):
        assert _outcome(PP.compile_advice, PP.Capped(Inner(), 8 * len(data)), junk) == (
            "refused", f"the advice is {8 * len(junk)} bits, over the registered cap ℓ = {8 * len(data)} bits")


def test_the_compiler_is_total_and_deterministic_on_random_truncated_and_flipped_advice(tmp_path):
    """Every input is the requests or a refusal by name, never another exception, and two runs give the same answer. Every random
    string and every strict prefix of the advice is refused; a flipped bit is either refused or other advice."""
    data, compiler = _advice(tmp_path), PP.Capped(Inner(), 4096)
    rng = random.Random(20261024)
    randoms = [bytes(rng.randrange(256) for _ in range(rng.randrange(0, 160))) for _ in range(1500)]
    prefixes = [data[:k] for k in range(len(data))]
    flips = [bytes(b ^ (1 << j) if i == k else b for i, b in enumerate(data)) for k in range(len(data)) for j in range(8)]
    tails = [data[:k] + bytes(rng.randrange(256) for _ in range(rng.randrange(0, 40))) for k in range(1, len(data)) for _ in range(40)]
    for inputs, all_refused in ((randoms, True), (prefixes, True), (flips, False), (tails, False),
                                ([None, "advice", bytearray(data)], True)):
        first = [_outcome(PP.compile_advice, compiler, x) for x in inputs]
        assert first == [_outcome(PP.compile_advice, compiler, x) for x in inputs]
        if all_refused:
            assert all(kind == "refused" and why for kind, why in first)
    assert len({why for kind, why in first if kind == "refused"}) >= 1


def test_a_malformed_opening_is_refused_by_name_and_the_same_on_every_run(tmp_path):
    data, compiler = _advice(tmp_path), PP.Capped(Inner())
    good = AW.opening(data, AW.commit_advice(compiler, data))
    reqs, bound = PP.open_advice(compiler, good)
    assert reqs == PP.compile_advice(compiler, data) and bound["advice"]["root"] == AW.commit_advice(
        compiler, data, [bytes.fromhex(s) for s in good["salts"]]).root.hex()
    bad = [None, [], {}, {**good, "scheme": "other"}, {**good, "advice": 7}, {**good, "advice": "zz"},
           {**good, "advice": data[:-1].hex()}, {**good, "salts": []}, {**good, "salts": good["salts"] + good["salts"]},
           {**good, "salts": ["zz"] * len(good["salts"])}, {**good, "salts": None},
           {**good, "salts": [s[:-2] for s in good["salts"]]}]
    for doc in bad:
        first = _outcome(PP.open_advice, compiler, doc)
        assert first[0] == "refused" and first == _outcome(PP.open_advice, compiler, doc), doc


def _rec(root: str, compiler: str = "cd" * 64) -> dict:
    return {"format": "verity/registration/v1", "population": 8,
            "window": {"kind": "stand-in", "compiler": {"name": PP.CAPPED, "digest": compiler}, "advice": {"root": root, "bits": 120}}}


def _bound(rec: dict) -> dict:
    return {k: rec["window"][k] for k in SRT.BINDING}


def test_the_advice_must_be_in_the_registration_the_sample_followed():
    rec = _rec("11" * 64)
    drawn = R.receipt(rec, A5.VERIFIER, received="0")
    assert PP.advice_before_sample(drawn, rec, _bound(rec)) == {"ok": True, "codes": [], "detail": {
        "drawn_after": R.digest(rec), "registration": R.digest(rec), "received": "0"}}
    late = _rec("22" * 64)
    assert PP.advice_before_sample(drawn, late, _bound(late))["codes"] == ["after-the-draw"]
    assert PP.advice_before_sample(R.receipt(rec, "one-stage-e2e verifier (a5.py)"), rec, _bound(rec))["codes"] == ["after-the-draw"]
    assert PP.advice_before_sample(drawn, rec, _bound(late))["codes"] == ["advice"]
    other = _rec("11" * 64, compiler="ef" * 64)
    assert PP.advice_before_sample(drawn, rec, _bound(other))["codes"] == ["compiler"]


def test_on_the_challengers_coins_the_advice_must_be_in_the_registration_the_challengers_stream_sampled_after():
    rec = _rec("11" * 64)
    theirs, ours = R.receipt(rec, A5.CHALLENGER, received="0"), R.receipt(rec, A5.VERIFIER, received="0")
    assert PP.advice_before_sample(theirs, rec, _bound(rec), "challenger")["ok"]
    assert PP.advice_before_sample(ours, rec, _bound(rec), "challenger")["codes"] == ["after-the-draw"]
    assert PP.advice_before_sample(theirs, rec, _bound(rec))["codes"] == ["after-the-draw"]
    late = _rec("22" * 64)
    assert PP.advice_before_sample(theirs, late, _bound(late), "challenger")["codes"] == ["after-the-draw"]


@dataclass(frozen=True)
class Listing:
    """A compiler listing ``transfer_gates``, which its digest covers, for the advice side alone."""

    name: str = "test-compiler"
    transfer_gates: tuple[str, ...] = ()

    @property
    def digest(self) -> str:
        return hashlib.sha512(canonical_json({"name": self.name, "transfer_gates": sorted(self.transfer_gates)})).hexdigest()


def test_the_transfer_gates_are_in_the_compilers_registered_digest(monkeypatch):
    """I2's digest covers the templates it lists as holding transfer gates, and `Capped` passes the list and its digest on, so a
    registration under a compiler proving other units whole names another compiler."""
    monkeypatch.setattr("verity_vllm.program.predict.score.code_version", lambda: "c0de")
    r = SimpleNamespace(phi=SimpleNamespace(to_json=lambda: {"model": "m"}), default_seed=0, cap_bits=PP.CAP_BITS)
    bare, listing = (PP.I2Compiler(r, {}, "/nonexistent", transfer_gates=tg) for tg in ((), ("RoPEHead_v1{D=64}",)))
    assert bare.digest != listing.digest and len(listing.digest) == 128
    assert PP.Capped(listing).transfer_gates == ("RoPEHead_v1{D=64}",) and PP.Capped(bare).transfer_gates == ()
    assert PP.Capped(listing).digest != PP.Capped(bare).digest
    assert PP.Capped(Inner()).transfer_gates == ()


def test_a_registration_under_the_compiler_listing_no_transfer_gates_or_one_fewer_is_refused(tmp_path):
    """Where the registered compiler lists transfer gates, the same window under its twin listing none, or under it less any one
    gate (either would sample them), is refused by `source1` on ``compiler``; a compiler listing none has no such case."""
    data = _advice(tmp_path)
    for i, gates in enumerate((("RoPEHead_v1{D=64}", "SiluMul_v2{I=8192}"), ("RoPEHead_v1{D=64}",), ())):
        compiler = PP.Capped(Listing(transfer_gates=gates))
        opening = AW.opening(data, AW.commit_advice(compiler, data))
        _reqs, bound = PP.open_advice(compiler, opening)
        w, mine, held = _single()
        w, held = {**w, **bound}, {**held, **bound}
        served = tmp_path / str(i)
        served.mkdir()
        (served / "registration.json").write_text(json.dumps({"window": w}))
        assert PP.source1(w, mine, held)["ok"]
        got = PP.more_negatives(argparse.Namespace(bundle="b"), 0, 1, compiler, mine, held, served, opening)
        assert all(v["refused"] for v in got.values()), got
        keys = ["registration-under-the-compiler-listing-no-transfer-gates (source)"][:len(gates)] + [
            f"registration-under-the-compiler-less-the-gate-{g} (source)" for g in gates]
        assert {k: got[k]["codes"] for k in keys} == {k: ["compiler"] for k in keys}
        assert not [k for k in got if ("transfer-gates" in k or "the-gate" in k) and k not in keys]


def test_a_single_gpu_window_proves_the_templates_its_compiler_lists_whole(monkeypatch):
    """`window1` hands the registered compiler's transfer gates to the window (`serving_rows.population`), which proves them
    whole under the work law."""
    from verity_vllm.pipeline import serving_rows as SRW

    got = {}
    rec = SimpleNamespace(run=SimpleNamespace(run_root=bytes(64)), store=SimpleNamespace(commit_scheme=SRT.ROWS_SCHEME),
                          bundle={"manifest_sha256": "bb" * 32})
    monkeypatch.setattr(PP, "record1", lambda bundle, row_dir: rec)
    monkeypatch.setattr(SRW, "population", lambda progs, partition, run, dirs, arrivals=None, transfer_gates=():
                        got.update(transfer_gates=transfer_gates) or ("window", []))

    class Loads(Listing):
        def load(self, rank, world, requests):
            return SimpleNamespace(progs={}, dirs={}, digs={"r0": "cc" * 64}, arrivals={"r0": 0})

    compiler = PP.Capped(Loads(transfer_gates=("RoPEHead_v1{D=64}",)))
    part = Path(__file__).parents[1] / "llama32_1b_tg_l0_step1.json"
    window, held = PP.window1(argparse.Namespace(bundle="b", row_dir="d", partition=str(part)), compiler, [], {})
    assert window == "window" and got == {"transfer_gates": ("RoPEHead_v1{D=64}",)}
    assert held["served_programs"] == {"r0": "cc" * 64}


def test_the_challengers_coins_need_main_and_the_gates_peer_and_key(monkeypatch):
    run = ["--bundle", "b", "--row-dir", "d", "--workload", "f", "--opening", "o", "--windows", "w", "--partition", "p",
           "--template-sets", "s", "--m0-pythonpath", "m", "--bin", "b", "--lean", "l", "--firewall", "f", "--out", "o"]
    gate = ["--challenger", "gate:research@vy-challenger-1", "--main", "m.txt"]
    for refused in (["--challenger", "local"], gate, [*gate, "--peer", "vy-nebius-2"], ["--challenger", "remote", "--main", "m.txt"]):
        with pytest.raises(SystemExit):
            PP.main(["audit", *run, *refused])
    got = []
    monkeypatch.setattr(PP, "audit", lambda a: got.append(a) or 0)
    assert PP.main(["audit", *run, *gate, "--peer", "vy-nebius-2", "--challenger-key", "key.json", "--gate-key", "id"]) == 0
    assert (got[0].challenger, got[0].peer, got[0].challenger_key, got[0].gate_key) == \
        ("gate:research@vy-challenger-1", "vy-nebius-2", "key.json", "id")


def test_an_audit_listens_on_ports_no_other_audit_holds():
    with socket.socket() as held, socket.socket() as spare:
        held.bind(("127.0.0.1", 0))
        spare.bind(("127.0.0.1", 0))
        taken, free = held.getsockname()[1], spare.getsockname()[1]
        spare.close()
        assert PP.free_ports(1, [taken, free]) == free
        with pytest.raises(SystemExit):
            PP.free_ports(1, [taken])
    base = PP.free_ports(2 * A5.RANK_PORTS)
    assert PP.PORTS.start <= base and base + 2 * A5.RANK_PORTS <= PP.PORTS.stop


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
def test_a_forged_rank_passes_only_when_lean_refuses_its_sampled_forged_block(summary, ok):
    assert PP.forged_row(summary, {"member": 0, "unit": 1})["ok"] is ok


@pytest.fixture(scope="module")
def tp(tmp_path_factory):
    """The deferred stand-in TP Commit and its advice; both ranks' windows over every unit (subset:8), and rank 0's again
    with member 0's unit 1 forged, each commit with its own opening."""
    pytest.importorskip("torch")
    root = tmp_path_factory.mktemp("prove-path")
    p = subprocess.run([sys.executable, "-m", "tests.pipeline.served_tp", str(root / "served"), "--world", "2", "--I", "256",
                        "--law", "subset:4", "--deferred"], cwd=VLLM, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-3000:]
    c = json.loads(p.stdout.strip().splitlines()[-1])
    assert PP.main(["advice", "--workload", c["workload"], "--out", str(root / "advice.bin")]) == 0
    c["all"] = root / "all.json"
    c["all"].write_text(json.dumps({"templates": ["SiluMul_v2{I=256}"], "law": "subset:8"}))
    base = ["--tp-commit", c["commit"], "--build-root", c["build"], "--advice", str(root / "advice.bin"), "--partition", str(c["all"]),
            "--workers", "1"]
    assert PP.main(["commit", *base, "--opening", str(root / "honest-opening.json"), "--out", str(root / "honest")]) == 0
    assert PP.main(["commit", *base, "--opening", str(root / "forged-opening.json"), "--out", str(root / "forged"), "--ranks", "0",
                    "--forge", "0:1"]) == 0
    c.update(root=root, honest=root / "honest", forged=root / "forged")
    c["args"] = {w: SimpleArgs(c, root / f"{w}-opening.json") for w in ("honest", "forged")}
    return c


def SimpleArgs(c: dict, opening: Path):
    return argparse.Namespace(tp_commit=c["commit"], build_root=c["build"], opening=str(opening), cap_bits=PP.CAP_BITS,
                              partition=str(c["all"]))


def _mine(c, which: str, r: int):
    compiler, requests, bound, opening = PP.compiled(c["args"][which])
    return (compiler, bound, opening, *AW.window_of_advice(c["commit"], r, c["all"], compiler, requests, bound))


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
def test_each_ranks_window_is_held_to_the_verifiers_window_from_the_opening(tp):
    """Both ranks bind one advice root, the one the verifier recomputes from the opening, and the compiler's digest with ℓ in it."""
    compiler, bound, opening, _mine0, _held0 = _mine(tp, "honest", 0)
    data = bytes.fromhex(opening["advice"])
    assert bound["advice"]["root"] == AW.commit_advice(compiler, data, [bytes.fromhex(s) for s in opening["salts"]]).root.hex()
    assert bound["compiler"] == {"name": PP.CAPPED, "digest": PP.Capped(compiler.inner, PP.CAP_BITS).digest}
    assert json.loads((tp["honest"] / "commit.json").read_text())["binding"] == bound
    for r in range(2):
        _c, _b, _o, mine, held = _mine(tp, "honest", r)
        d = tp["honest"] / f"rank{r}"
        rec = _registered(d)
        assert {k: rec["window"][k] for k in SRT.BINDING} == bound
        assert SRT.source(rec["window"], mine, held)["ok"]
        assert _check(rec, mine, _publics(d, mine)).ok
        assert [(m.template, m.n) for m in mine.members] == [("SiluMul_v2{I=256}", 8)]


@pytest.mark.slow
def test_the_advices_negatives_are_each_refused_by_name(tp):
    compiler, _bound, opening, mine, held = _mine(tp, "honest", 0)
    a = argparse.Namespace(**vars(tp["args"]["honest"]))
    got = PP.more_negatives(a, 0, 2, compiler, mine, held, tp["honest"] / "rank0", opening)
    assert {k: v["codes"] for k, v in got.items() if k != "registration-of-another-rank (source)"} == {
        "registration-naming-another-compiler (source)": ["compiler"],
        "registration-naming-another-advice (source)": ["advice"],
        "opening-under-another-salt (source)": ["advice"],
        "registration-binds-no-compiler-or-advice (source)": ["compiler", "advice"],
        "registration-names-another-cap (source)": ["compiler"]}
    other = got["registration-of-another-rank (source)"]
    assert other["refused"] and {"run", "tp"} <= set(other["codes"])


@pytest.mark.slow
def test_advice_committed_after_the_sample_passes_every_check_before_it_and_is_refused(tp):
    """With the receipt the stream writes on receiving the honest registration, the honest registration holds; the same advice
    committed again after the coins (a fresh root, a true commitment of it, which `source` accepts with its opening) is refused."""
    compiler, bound, opening, mine, held = _mine(tp, "honest", 0)
    rec = _registered(tp["honest"] / "rank0")
    receipt = R.receipt(rec, A5.VERIFIER, received="0")
    assert PP.advice_before_sample(receipt, rec, bound)["ok"]
    got = PP.advice_after_sample(compiler, receipt, rec, opening, mine, held)
    assert got == {"refused": True, "codes": ["after-the-draw"], "before_draw_checks_pass": True}


@pytest.mark.slow
def test_a_forged_block_passes_every_check_before_the_sample_and_differs_in_one_bit(tp):
    """The forged window's registration holds to the verifier's window and its own public bodies, as the honest one does: its
    rows, commit strings and roots agree with the flipped word. Its rows differ from the honest window's in exactly one bit."""
    _c, _b, _o, mine, held = _mine(tp, "forged", 0)
    d = tp["forged"] / "rank0"
    rec = _registered(d)
    assert SRT.source(rec["window"], mine, held)["ok"]
    assert _check(rec, mine, _publics(d, mine)).ok
    honest, forged = _rows(tp["honest"] / "rank0"), _rows(d)
    assert len(honest) == len(forged) and sum(bin(x ^ y).count("1") for x, y in zip(honest, forged, strict=True)) == 1
    assert json.loads((tp["forged"] / "commit.json").read_text())["ranks"]["0"]["forged"] == {"member": 0, "unit": 1}


@pytest.mark.slow
def test_a_plain_audit_takes_the_stand_ins_window_only_inside_one_table_under_a_stratified_sample(tp):
    """`--plain` covers one table under a stratified sample (`a5.PLAIN_SCOPE`): the stand-in's window, one member over every
    unit, is inside it under stratified:8, and refused before the sample under its own subset:8, drawn outside a stream, with a
    second member, or with registered reads or shared rows. A `--zk` audit is refused for none of these."""
    _c, _b, _o, mine, _held = _mine(tp, "honest", 0)
    plain, zk = (argparse.Namespace(plain=p, coins=None, verifier_coins=True) for p in (True, False))
    stratified = replace(mine, law={"law": "stratified", "k": 8})
    assert A5.plain_refusal(plain, stratified, None) is None
    m = mine.members[0]

    def window(*members):
        return SimpleNamespace(members=members, law=stratified.law, population=sum(x.n for x in members), row_tables=())

    cases = [("a plain session's draw is stratified", plain, mine, None),
             ("through a rank's service.Stream", argparse.Namespace(plain=True, coins=None, verifier_coins=False), stratified, None),
             ("the window has 2 members", plain, window(m, replace(m, base=m.n)), None),
             ("registered reads, shared rows", plain, stratified, {m.key: {"ports": {}}}),
             ("registered reads, shared rows", plain, window(replace(m, tables=(("x", 1),))), None)]
    for why, a, w, registered in cases:
        got = A5.plain_refusal(a, w, registered)
        assert got is not None and why in got and A5.PLAIN_SCOPE in got, (why, got)
        assert A5.plain_refusal(argparse.Namespace(**{**vars(a), "plain": False}), w, registered) is None
    assert A5.plain_refusal(zk, mine, None) is None


def _accepted(law: dict, delta: float = 0.5) -> dict:
    reg = {"population": 8, "partition": "ab" * 64, "program": "cd" * 64}
    draw = {**law, "population": 8, "units": [0, 2, 5, 7]}
    return {"accepted": True, "registration": reg, "draw": draw,
            "wrong_units": AU.wrong_units(reg, draw, delta)}


def test_the_charge_terms_are_the_accepted_profiles_location_and_committed_bits_bounds():
    """An accepted audit's charge terms (README, "What an accepted audit means"): λ_δ is `consumers.location_bits` of the
    audit's profile, β_δ its `harm_bound` with each template's units weighing their committed bits (one weight, the most, where
    the law's strata aren't the templates), and `exfiltration_bound` their sum; a refused audit's record bounds nothing."""
    stratified = {"law": "stratified", "k": 4, "strata": [{"template": "T", "units": [[0, 8]], "k": 4}]}
    for law, bits in ((stratified, {"T": 40}), ({"law": "subset", "k": 4}, {"T": 40, "U": 24})):
        rec = _accepted(law)
        p = AU.profile(rec["registration"], rec["draw"], 0.5)
        got = PP.charge_terms(rec, bits, 169, 0.5)
        weight = {"T": 40.0} if law is stratified else 40
        assert got == {"advice_bits": 169, "delta": 0.5, "wrong_units": rec["wrong_units"], "template_bits": bits,
                       "location_bits": location_bits(p), "harm_bound": p.harm_bound(weight),
                       "exfiltration_bound": exfiltration_bound(p, weight)}, law
        assert got["exfiltration_bound"] == got["harm_bound"] + got["location_bits"] and got["location_bits"] > 0
    assert PP.charge_terms({**_accepted(stratified), "accepted": False}, {"T": 40}, 169, 0.5) is None
    assert PP.charge_terms({}, {"T": 40}, 169, 0.5) is None


def test_i2s_requests_are_its_compiled_units_once_per_advice_and_its_refusal_is_named(monkeypatch):
    from verity_vllm.program.predict import steps as ST
    from verity_vllm.program.predict.common import Unexpressible

    calls = []

    def compile_steps(registered, data, ingress, *, named=False):
        calls.append(data)
        if data == b"no":
            raise Unexpressible("a chunked prefill")
        return SimpleNamespace(requests=[ST.RequestUnit(0, "d0" * 64, 5, 2, {}, None, 0, "r0"),
                                         ST.RequestUnit(1, "d1" * 64, 3, 1, {}, None, 2, "r1")], named={})

    monkeypatch.setattr(ST, "compile_steps", compile_steps)
    i2 = PP.I2Compiler(registered=None, ingress={}, programs_root="/nonexistent")
    want = [{"request_id": "r0", "arrive_step": 0, "prompt_len": 5, "max_tokens": 3, "digest": "d0" * 64},
            {"request_id": "r1", "arrive_step": 2, "prompt_len": 3, "max_tokens": 2, "digest": "d1" * 64}]
    assert i2.requests(b"ok") == want and i2.requests(b"ok") == want and calls == [b"ok"]
    with pytest.raises(ValueError, match="I2 refuses the advice: a chunked prefill"):
        i2.requests(b"no")


def test_i2_loads_only_a_single_gpu_session_and_only_programs_a_build_dir_holds(tmp_path):
    for name, ok, digest in (("build_request_LP5_T2", True, "d0" * 64), ("build_request_LP3_T1", False, "d1" * 64),
                             ("build_step", True, "d2" * 64)):
        (tmp_path / name).mkdir()
        (tmp_path / name / "result.json").write_text(json.dumps({"ok": ok, "program": {"digest": digest}}))
    assert PP.bodies_by_digest(str(tmp_path)) == {"d0" * 64: str(tmp_path / "build_request_LP5_T2")}
    i2 = PP.I2Compiler(registered=None, ingress={}, programs_root=str(tmp_path))
    for rank, world in ((1, 2), (0, 2), (1, 1)):
        with pytest.raises(ValueError, match="I2 compiles a single-GPU session"):
            i2.load(rank, world, [])
    with pytest.raises(ValueError, match="request r1: no Build dir under .* holds I2's Program d1d1"):
        i2.load(0, 1, [{"request_id": "r1", "arrive_step": 0, "digest": "d1" * 64}])


#: integrations/vllm's `tests.program.predict.test_predict.TINY`: a two-layer Llama whose request Programs are I2's in a second
TINY = {"model_type": "llama", "architectures": ["LlamaForCausalLM"], "hidden_size": 64, "intermediate_size": 128,
        "num_attention_heads": 4, "num_key_value_heads": 2, "num_hidden_layers": 2, "vocab_size": 256, "rms_norm_eps": 1e-5,
        "tie_word_embeddings": True, "max_position_embeddings": 64, "hidden_act": "silu"}
STOCHASTIC = {"kind": "stochastic", "temperature": 0.8, "top_p": 0.95, "top_k": 0, "seeded": True, "splits": "derived"}


def _request_build(sampling: dict | None = None, named: bool = False, **shape):
    from verity_vllm.program.predict.build import build_program
    from verity_vllm.program.predict.naming import named_program
    from verity_vllm.program.predict.phi import Phi

    phi = Phi(model={"repo": "org/m", "revision": "r", "hf_config": TINY}, workload={"requests": []},
              target={"compute_capability": [12, 0], "num_sms": 188})
    phi = Phi.from_json({**phi.to_json(), "sampling": sampling or {"kind": "greedy"}})
    return named_program(phi, **shape) if named else build_program(phi, "request", **shape)


def _request_program(sampling: dict | None = None, **shape):
    return _request_build(sampling, **shape).program


@pytest.fixture(scope="module")
def requests_of_tiny():
    """TINY's greedy request Programs at (LP, T) = (4, 1) and (4, 2), its stochastic one (seed and splits beside the prompt)
    and its cached one (a prefix of 16 rows beside the prompt)."""
    return {"greedy": _request_program(LP=4, T=1), "longer": _request_program(LP=4, T=2),
            "stochastic": _request_program(STOCHASTIC, LP=4, T=2), "cached": _request_program(LP=24, T=2, C=16)}


def _build(d: Path, program, *, desc: dict | None = None, digest: str | None = None) -> str:
    """Build dir ``d`` of ``program`` as the Build writes one (`build_instances.write_instances`): its descriptor (``desc``, the
    program's by default), and its instances and ``result.json`` under ``digest`` (the descriptor's by default); the digest."""
    import gzip

    from verity.ml.circuits.codec import encode_program, program_digest
    from verity_vllm.pipeline.build_instances import write_instances

    d.mkdir(parents=True, exist_ok=True)
    desc = encode_program(program) if desc is None else desc
    digest = digest or program_digest(desc)
    with gzip.open(d / "descriptor.json.gz", "wt") as f:
        json.dump(desc, f)
    write_instances(str(d), program, digest, None, {})
    (d / "result.json").write_text(json.dumps({"ok": True, "program": {"digest": digest}}))
    return digest


def _edit_instances(d: Path, edit) -> None:
    import gzip

    with gzip.open(d / "instances.json.gz", "rt") as f:
        inst = json.load(f)
    edit(inst)
    with gzip.open(d / "instances.json.gz", "wt") as f:
        json.dump(inst, f)


def _load(root: Path, digest: str):
    return PP.I2Compiler(registered=None, ingress={}, programs_root=str(root)).load(
        0, 1, [{"request_id": "r0", "arrive_step": 0, "digest": digest}])


def test_i2_loads_a_build_whose_body_is_the_program_it_compiled(tmp_path, requests_of_tiny):
    """I2's adapter holds each Build dir to the digest I2 compiled (`build_of`) before the window reads its rows: an honest
    Build of each request Program loads, its Input batches read as the parameters they bind (prompt, seed, splits, prefix,
    weights)."""
    for name, program in requests_of_tiny.items():
        dg = _build(tmp_path / name / "build_request", program)
        p = _load(tmp_path / name, dg)
        assert p.digs == {"r0": dg} and len(p.progs["r0"]["rows"]) == len(program.fn.body.nodes), name


def test_i2_refuses_a_build_whose_body_was_swapped_under_its_old_digest(tmp_path, requests_of_tiny):
    """The Build's ``result.json`` (and its instances head) keep the digest I2 compiled, and its body is another Program's:
    each is refused by name, the descriptor by its recomputed digest, the instances by the descriptor's rows."""
    from verity.ml.circuits.codec import encode_program, program_digest

    greedy, longer = requests_of_tiny["greedy"], requests_of_tiny["longer"]
    dg, other = (program_digest(encode_program(p)) for p in (greedy, longer))
    d = tmp_path / "desc" / "build_request"
    _build(d, greedy, desc=encode_program(longer), digest=dg)
    with pytest.raises(ValueError, match=rf"^request r0: {re.escape(str(d))}/descriptor.json.gz is Program {other[:16]}, "
                                         rf"not I2's {dg[:16]}$"):
        _load(tmp_path / "desc", dg)

    d = tmp_path / "inst" / "build_request"
    _build(d, longer, desc=encode_program(greedy), digest=dg)
    with pytest.raises(ValueError, match=rf"^request r0: {re.escape(str(d))}'s instances.json.gz has {len(longer.fn.body.nodes)} "
                                         rf"rows, and its descriptor.json.gz {len(greedy.fn.body.nodes)} nodes$"):
        _load(tmp_path / "inst", dg)

    k = next(k for k, n in enumerate(greedy.fn.body.nodes) if len(n.args) == 2 and n.args[0].refs != n.args[1].refs)
    for what, edit in (("args", lambda r: r["args"].reverse()), ("spec", lambda r: r.update(spec=r["spec"] + "x")),
                       ("i", lambda r: r.update(i=r["i"] + 1))):
        d = tmp_path / what / "build_request"
        _build(d, greedy)
        _edit_instances(d, lambda inst: edit(inst["rows"][k]))
        with pytest.raises(ValueError, match=rf"^request r0: {re.escape(str(d))}'s instances.json.gz row {k} is not its "
                                             rf"descriptor.json.gz's node {k}: they differ in {what}$"):
            _load(tmp_path / what, dg)


def test_i2_refuses_a_build_that_records_another_digest_or_has_no_descriptor(tmp_path, requests_of_tiny):
    greedy = requests_of_tiny["greedy"]
    d = tmp_path / "head" / "build_request"
    dg = _build(d, greedy)
    _edit_instances(d, lambda inst: inst.update(digest="ab" * 64))
    with pytest.raises(ValueError, match=rf"'s instances.json.gz records Program {'ab' * 8}, and its descriptor.json.gz is "
                                         rf"Program {dg[:16]}$"):
        _load(tmp_path / "head", dg)

    d = tmp_path / "result" / "build_request"
    _build(d, greedy)
    assert PP.build_of(str(d), dg)["instances"]["digest"] == dg
    (d / "result.json").write_text(json.dumps({"ok": True, "program": {"digest": "ab" * 64}}))
    with pytest.raises(ValueError, match=rf"'s result.json records Program {'ab' * 8}, and its descriptor.json.gz is Program"):
        PP.build_of(str(d), dg)

    d = tmp_path / "boolean" / "build_request"
    _build(d, greedy)
    _edit_instances(d, lambda inst: inst.update(ir="boolean"))
    with pytest.raises(ValueError, match="is a Build in IR 'boolean', and I2 compiles word Programs"):
        _load(tmp_path / "boolean", dg)
    (d / "descriptor.json.gz").unlink()
    with pytest.raises(ValueError, match=rf"^request r0: {re.escape(str(d))} has no descriptor.json.gz: its Program's digest "
                                         rf"can't be recomputed$"):
        _load(tmp_path / "boolean", dg)


def test_a_builds_input_batches_are_its_parameters_as_src_shapes_them():
    src = 'Derived[X]_v1{SRC={"dict":{"inputs":[[2],[],[0],[2,2]]}}}'
    assert PP.input_layout({"program_fn": src}, [2, 1, 3, 1, 5, 7]) == [(0, 0), (1, 0), (3, 0), (3, 3), (4, 0), (4, 5)]
    with pytest.raises(ValueError, match="Input batch 1 runs past parameter 1's 1 leaves"):
        PP.input_layout({"program_fn": src}, [2, 2, 4])
    with pytest.raises(ValueError, match="the root's 2 Input batches don't fill the 4 parameters SRC shapes"):
        PP.input_layout({"program_fn": src}, [2, 1])
    for pf in ("Derived[X]_v1", 'Derived[X]_v1{SRC={"dict":{}}}', 'Derived[X]_v1{SRC={"dict":{"inputs":[1]}}}'):
        with pytest.raises(ValueError, match="records no SRC input shapes: not one of I2's Programs"):
            PP.input_layout({"program_fn": pf}, [1])
    assert PP.merged([["p", 1, 0, 3], ["p", 1, 3, 5], ["p", 0, 5, 6], ["n", 2, 0, 1], ["n", 2, 2, 3]]) == \
        [["p", 1, 0, 5], ["p", 0, 5, 6], ["n", 2, 0, 1], ["n", 2, 2, 3]]


def test_each_requests_body_is_its_prompt_cap_and_sampling_by_its_i1_id():
    ctx = {"requests": [{"request_id": "x", "prompt_token_ids": [1, 2], "max_tokens": 3},
                        {"request_id": "y", "prompt_token_ids": [4], "max_tokens": 2}],
           "sampling": {"temperature": 0.0, "top_p": 1.0, "logprobs": 5}}
    assert PP.ingress_of(ctx) == {"r0": {"prompt": [1, 2], "max_tokens": 3, "temperature": 0.0, "top_p": 1.0},
                                  "r1": {"prompt": [4], "max_tokens": 2, "temperature": 0.0, "top_p": 1.0}}


def _single(root: str = "aa" * 64) -> tuple[dict, SimpleNamespace, dict]:
    r = SimpleNamespace(store=SimpleNamespace(commit_scheme=SRT.ROWS_SCHEME), run=SimpleNamespace(run_root=bytes.fromhex(root)),
                        bundle={"manifest_sha256": "bb" * 32})
    si = PP.stand_in1(r)
    held = {"commitment": SRT.ROWS_SCHEME, "served_programs": {"r0": "cc" * 64}, "stand_in": si,
            "compiler": {"name": PP.CAPPED, "digest": "dd" * 64}, "advice": {"root": "ee" * 64, "bits": 120}}
    w = {"kind": "stand-in", "run": root, "commitment": SRT.ROWS_SCHEME, "of": si["of"], "link": si["link"],
         "served_programs": held["served_programs"], "compiler": held["compiler"], "advice": held["advice"]}
    return w, SimpleNamespace(run=root), held


def test_a_single_gpu_window_is_a_stand_in_of_the_records_sha512_run_root():
    w, mine, held = _single()
    assert held["stand_in"]["of"] == {"commitment": "rows-sha512", "run_root": "aa" * 64, "bundle_manifest_sha256": "bb" * 32,
                                      "rows": PP.ROWS1}
    assert "provenance, not proof" in held["stand_in"]["link"]
    assert PP.source1(w, mine, held)["ok"]
    for key, other in (("kind", "tp-stand-in"), ("run", "ff" * 64), ("commitment", "vllm-v1"), ("of", {**w["of"], "run_root": "ff" * 64}),
                       ("link", "proof"), ("served_programs", {"r0": "ff" * 64}), ("compiler", {**w["compiler"], "digest": "ff" * 64}),
                       ("advice", {**w["advice"], "root": "ff" * 64})):
        assert PP.source1({**w, key: other}, mine, held)["codes"] == [key], key
    assert set(PP.source1(None, mine, held)["codes"]) == {"kind", "run", "commitment", "of", "link", "served_programs", "compiler", "advice"}
    w32, mine32, held32 = _single("aa" * 32)
    assert PP.source1(w32, mine32, held32)["codes"] == ["sha512"]


def test_a_run_is_one_committed_run_and_a_single_gpu_runs_one_window_is_rank_0():
    tail = ["--advice", "a.bin", "--opening", "o.json", "--partition", "p.json", "--out", "w"]
    for run in ([], ["--tp-commit", "c"], ["--bundle", "b", "--row-dir", "d"], ["--bundle", "b", "--workload", "f"],
                ["--tp-commit", "c", "--build-root", "b", "--bundle", "b", "--row-dir", "d", "--workload", "f"]):
        with pytest.raises(SystemExit):
            PP.main(["commit", *run, *tail])
    assert PP.ranks_of(argparse.Namespace(bundle="b", ranks=None)) == (1, [0])
    with pytest.raises(SystemExit, match=r"ranks \[1\] are not ranks of the world of 1"):
        PP.ranks_of(argparse.Namespace(bundle="b", ranks="0,1"))


def test_per_block_timing_is_over_the_members_sampled_blocks_not_flocks_virtual_units():
    s = {"draw_counts": {"T": 2}, "timing": {"T": {"session_s": 501.2, "verify_s": 6.2, "lean_verify_s": 47.0,
                                                   "timing": {"prove_total_s": 428.9, "units": 16384, "k_log": 26,
                                                              "proof_bytes": [830330, 830330]}}}}
    assert PP.per_unit(s) == {"T": {"units": 2, "k_log": 26, "proof_bytes": 1660660, "prove_s_per_unit": 214.45,
                                    "session_s_per_unit": 250.6, "verify_s_per_unit": 3.1, "lean_verify_s_per_unit": 23.5}}


def test_a5s_r2_partition_is_another_population_even_of_a_one_unit_window():
    xor8 = PrimitiveDefinition("ProvePathTestXor8", 1, (("a", Bits(8)), ("b", Bits(8))), Bits(8), lambda a, b: a ^ b,
                               register=False)
    pair = bind(CompositeDefinition("ProvePathTestPair", 0, (), lambda S_: ((("x", Array(2, Bits(8))),), Bits(8)),
                                    lambda B, S_, x: B.call(xor8, x[0], x[1]), register=False))
    triple = bind(CompositeDefinition("ProvePathTestTriple", 0, (), lambda S_: ((("x", Array(3, Bits(8))),), Bits(8)),
                                      lambda B, S_, x: B.call(xor8, B.call(xor8, x[0], x[1]), x[2]), register=False))
    for defs, want in (([(pair, 1)], P.population_program(pair, 2)), ([(pair, 4)], P.population_program(pair, 3)),
                       ([(pair, 1), (triple, 1)], P.mixed_population_program([(pair, 2), (triple, 1)])),
                       ([(pair, 3), (triple, 1)], P.mixed_population_program([(pair, 2), (triple, 1)]))):
        assert P.program_digest(A5.resized_program(defs)) == P.program_digest(want)


TARGET = {"device_name": "test-gpu", "execution_label": "bi-eager", "batch_invariant": True}


def _serve_request(i: int, prompt: list[int], cap: int, windows: list[int], user: str | None = None) -> dict:
    return {"http_id": f"cmpl-{i}", "engine_id": f"cmpl-{i}-0", "user": user, "prompt_ids": prompt, "windows": windows,
            "ingress": {"brought": {"max_tokens": cap, "temperature": 0.0},
                        "params": {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "seed": None, "max_tokens": cap, "min_tokens": 0,
                                   "stop": []}}}


#: the server's registered compiler in these tests (`served_identity`'s, from `verity-vllm advice` on the server's tree)
SERVER = ("verity_vllm.program.predict.steps.compile_steps", "5e" * 64)


def _served(tmp_path: Path, monkeypatch, *, root: str = "aa" * 64, scheme: str = SRT.ROWS_SCHEME, published: str | None = None,
            uncapped: bool = False, gates: tuple[str, ...] | None = None) -> SimpleNamespace:
    """A serve-openai run dir with one closed window (w0, the stand-in workload's two requests by their `user`) and a request of
    another window, its advice committed as `verity-vllm advice` commits it under the served compiler (or, ``uncapped``, under
    the inner compiler alone; ``gates``, under the server's listing those instead of the model's of record), and its row dir's
    request Programs as vllm's registered compiler writes them (``manifest.json``);
    the server's compiler identity, I2's compile and the row dir's record stubbed."""
    from verity_vllm.program.predict import steps as ST

    monkeypatch.setattr(PP, "served_identity", lambda: SERVER)
    run, rows = tmp_path / "serve", tmp_path / "rows"
    (run / "advice" / "w0" / "private").mkdir(parents=True)
    rows.mkdir()
    (rows / "manifest.json").write_text(json.dumps({"schema": PP.SERVED_PROGRAMS, "window": 0,
                                                    "requests": {"r0": {"digest": "d0" * 64}, "r1": {"digest": "d1" * 64}}}))
    pub = root if published is None else published
    (run / "serve.json").write_text(json.dumps({"model": "unsloth/Llama-3.2-1B", "revision": "9535bd9b1d1dea6acafbdc4813b728796aeb28da",
                                                "default_seed": DEFAULT_SEED, "engine": {"execution": "bi-eager", "pin": {"world": 1}},
                                                "target": TARGET, "run_id": "RUN"}))
    reqs = [_serve_request(0, list(range(5)), 3, [0], "r0"), _serve_request(1, list(range(3)), 2, [0], "r1"),
            _serve_request(2, [9], 1, [1])]
    (run / "requests.jsonl").write_text("".join(json.dumps(q) + "\n" for q in reqs))
    (run / "windows.jsonl").write_text("".join(json.dumps({"window": w, "run_id": f"RUN/w{w}", "run_root": r}) + "\n"
                                               for w, r in ((0, pub), (1, "bb" * 64))))
    data = _advice(tmp_path)
    compiler = PP.Capped(PP.i2_of_serve(str(run), 0, str(rows), None, None))
    if gates is not None:
        compiler = PP.Capped(replace(compiler.inner, transfer_gates=tuple(gates)))
    inner = SimpleNamespace(name=compiler.inner.name, digest=compiler.inner.digest)
    _r, bound, reg = AW.of_advice(inner if uncapped else compiler, data)
    adv = run / "advice" / "w0"
    (adv / "registration.json").write_text(json.dumps({
        "schema": "verity-vllm/advice-registration/v1", "scheme": AW.SCHEME, "binding": bound,
        "compiler": {**bound["compiler"], "cap_bits": PP.CAP_BITS, "inner": {"name": SERVER[0], "digest": SERVER[1]}},
        "window": 0, "commit": {"run_id": "RUN/w0", "run_root": pub}}))
    (adv / "private" / "advice.bin").write_bytes(data)
    (adv / "private" / "opening.json").write_text(json.dumps(AW.opening(data, reg)))

    def compile_steps(registered, data, ingress, *, named=False):
        assert set(ingress) == {"r0", "r1"} and registered.default_seed == DEFAULT_SEED
        return SimpleNamespace(requests=[ST.RequestUnit(0, "d0" * 64, 5, 2, {}, None, 0, "r0"),
                                         ST.RequestUnit(1, "d1" * 64, 3, 1, {}, None, 2, "r1")], named={})

    monkeypatch.setattr(ST, "compile_steps", compile_steps)
    record = SimpleNamespace(run=SimpleNamespace(run_root=bytes.fromhex(root)), store=SimpleNamespace(commit_scheme=scheme),
                             digests={"r0": "d0" * 64, "r1": "d1" * 64})
    monkeypatch.setattr(PP, "record1", lambda bundle, row_dir: record)
    a = argparse.Namespace(serve_run=str(run), window=0, row_dir=str(rows), bundle=str(rows / "b"),
                           hf_config=None, target=None, cap_bits=PP.CAP_BITS, opening=str(adv / "private" / "opening.json"),
                           tp_commit=None, workload=None)
    return SimpleNamespace(run=run, a=a, data=data, bound=bound, compiler=compiler, root=root)


#: the scope a naming test's partition takes: layer 0's MLP, as the Llama partitions do (the ``model.`` wrapper's prefix stripped)
LAYER0 = {"templates": ["SiluMul_v2{I=128}"], "law": "subset:2", "scope": {"op_path_prefix": "model.layers.0.mlp."}}


def _permuted(named: Any, swap: tuple[str, str] | None = None, params: bool = False) -> Any:
    """``named`` (a request Program's `naming.Named`) as a prover could write it beside the same Program, under the same digest:
    ``swap``, two name prefixes, the first Calls named under each exchanging their names and their records; ``params``: layers 0
    and 1's ``down_proj`` weights exchanging names, under the one shape both have."""
    names, doc = dict(named.names), dict(named.correspondence)
    if swap:
        a, b = (min(k for k, v in names.items() if v.startswith(x)) for x in swap)
        names[a], names[b] = names[b], names[a]
        doc["calls"] = calls = list(doc["calls"])
        calls[a], calls[b] = calls[b], calls[a]
    if not params:
        return replace(named, names=names, correspondence=doc)
    w0, w1 = (f"model.model.layers.{i}.mlp.down_proj.weight" for i in (0, 1))

    def exchange(t: str) -> str:
        return t.replace(w0, "\0").replace(w1, w0).replace("\0", w1)

    return replace(named, names=names, correspondence={**doc, "weights": [exchange(f) for f in doc["weights"]]},
                   params=[[n, exchange(t)] for n, t in named.params])


def _tiny_served(tmp_path: Path, monkeypatch) -> SimpleNamespace:
    """`_served`'s window with its two requests running TINY's request Programs as I2 compiles them (r0 at LP 4, T 1; r1 at LP 4,
    T 2), each with the names I2's compile gives it where it is asked for them (``named``, `naming`); the served model lists no
    transfer gates (TINY has no ``RoPEHead_v1{D=64}``). -> `_served`'s, with the Programs and their names by (LP, T), the units,
    and each compile's ``named``."""
    from verity_vllm.program.predict import steps as ST

    monkeypatch.setattr(PP, "GATES_OF_RECORD", {})
    s = _served(tmp_path, monkeypatch)
    builds = {(4, 1): _request_build(LP=4, T=1, named=True), (4, 2): _request_build(LP=4, T=2, named=True)}
    units = [ST.RequestUnit(0, builds[(4, 1)].digest, 4, 1, {"kind": "greedy"}, None, 0, "r0"),
             ST.RequestUnit(1, builds[(4, 2)].digest, 4, 2, {"kind": "greedy"}, None, 2, "r1")]
    compiles: list[bool] = []

    def compile_steps(registered, data, ingress, *, named=False):
        compiles.append(named)
        return SimpleNamespace(requests=units, programs={b.digest: b.program for b in builds.values()},
                               named={b.digest: b.named for b in builds.values()} if named else {})

    monkeypatch.setattr(ST, "compile_steps", compile_steps)
    return SimpleNamespace(**vars(s), programs={k: b.program for k, b in builds.items()}, named={k: b.named for k, b in builds.items()},
                           units=units, compiles=compiles)


def _row_dir(rows: Path, s: SimpleNamespace, swap: tuple[str, str] | None = None, params: bool = False) -> Path:
    """The prover's row dir of `_tiny_served`'s Programs, each Build written beside its names as `_permuted` permutes them under
    ``swap`` and ``params`` (`naming.write_build`), each still I2's Program (`build_of`)."""
    from verity_vllm.program.predict import steps as ST
    from verity_vllm.program.predict.naming import write_build

    for u in s.units:
        d = rows / ST.trace_dir(u)
        write_build(str(d), s.programs[(u.LP, u.T)], _permuted(s.named[(u.LP, u.T)], swap, params))
        assert PP.build_of(str(d), u.digest)["instances"]["digest"] == u.digest
    return rows


def _circuit(compiler: Any, data: bytes, partition: Mapping[str, Any]) -> tuple[dict, list]:
    """What a window over ``compiler``'s Programs for the advice ``data`` is: its circuit record (`serving_rows.circuit_record`, as
    the prover's commit writes ``circuit.json``) and its rows."""
    from verity_vllm.pipeline import serving_rows as SRW

    p = compiler.load(0, 1, PP.compile_advice(compiler, data))
    window, rows = SRW.population(p.progs, dict(partition), "", p.dirs, arrivals=p.arrivals, transfer_gates=compiler.transfer_gates)
    return SRW.circuit_record(window), rows


def test_a_served_windows_bodies_are_its_requests_ingress_by_i1_id(tmp_path, monkeypatch):
    s = _served(tmp_path, monkeypatch)
    w = PP.serve_window(str(s.run), 0)
    assert [PP.serve_id(q) for q in w.requests] == ["r0", "r1"] and w.commit["run_root"] == s.root
    assert PP.serve_ingress(w.requests) == {"r0": {"max_tokens": 3, "temperature": 0.0, "prompt": [0, 1, 2, 3, 4]},
                                            "r1": {"max_tokens": 2, "temperature": 0.0, "prompt": [0, 1, 2]}}
    assert [PP.serve_id(q) for q in PP.serve_requests(s.run, 1)] == ["cmpl-2"]
    with pytest.raises(ValueError, match="request r0 is recorded twice"):
        PP.serve_ingress([w.requests[0], w.requests[0]])
    assert PP.serve_ingress([{**w.requests[0], "ingress": {}}]) == {"r0": {"prompt": [0, 1, 2, 3, 4]}}
    with pytest.raises(ValueError, match="0 lines for window 2, not one"):
        PP.serve_window(str(s.run), 2)


def test_phi_of_a_served_engine_is_advice_compiles_and_reads_its_target_and_execution(tmp_path):
    serve = {"model": "m/x", "revision": "r", "default_seed": DEFAULT_SEED, "engine": {"execution": "bi-eager"}, "target": TARGET}
    phi = PP.phi_of_serve(serve)
    assert (phi.model["repo"], phi.model["revision"], phi.target, phi.execution) == ("m/x", "r", TARGET, "bi-eager")
    assert phi.model["hf_config"] == json.loads(PP.LLAMA32_1B_CONFIG.read_text()) and not phi.requests
    with pytest.raises(ValueError, match="serve.json records no target profile .* --target F gives it"):
        PP.phi_of_serve({**serve, "target": None})
    (tmp_path / "target.json").write_text(json.dumps({**TARGET, "device_name": "other"}))
    assert PP.phi_of_serve({**serve, "target": None}, target=str(tmp_path / "target.json")).target["device_name"] == "other"
    with pytest.raises(ValueError, match="no execution label"):
        PP.phi_of_serve({**serve, "engine": {}})
    assert PP.phi_of_serve({**serve, "execution": "eager"}).execution == "eager"


def test_a_served_pins_phi_reads_its_config_of_record_as_advice_compile_does():
    """A pinned served model's φ reads the pin's config.json of record (`pinned_hf_config`, advice_compile's too), so an int8
    serve compiles on both sides with its quantization config beside the target's int8 scope, never the bf16 fixture's."""
    from verity_vllm.engine.profiles.dense_generic import pinned_hf_config

    rev = "15b0976d400d796ee589e7737d03a919526bf800"
    serve = {"model": "verity-int8/Llama-3.2-1B-INT8", "revision": rev, "execution": "bi-eager",
             "target": {**TARGET, "int8_exact": "linears"}}
    phi = PP.phi_of_serve(serve)
    assert phi.model["hf_config"] == json.loads(pinned_hf_config(serve["model"], rev).read_text())
    assert phi.model["hf_config"]["quantization_config"]["quant_method"] == "verity_int8" and phi.target["int8_exact"] == "linears"
    assert phi.digest() != PP.phi_of_serve(serve, hf_config=str(PP.LLAMA32_1B_CONFIG)).digest()
    bf16 = {**serve, "model": "unsloth/Llama-3.2-1B", "revision": "9535bd9b1d1dea6acafbdc4813b728796aeb28da", "target": TARGET}
    assert PP.phi_of_serve(bf16).model["hf_config"] == json.loads(PP.LLAMA32_1B_CONFIG.read_text())


def test_a_served_windows_compiler_is_a_single_gpu_i2s_over_the_servers_code(monkeypatch):
    """The served I2 is named as the server names its compiler, and its digest is a single-GPU window's I2's (one
    `I2Compiler.digest`) with the server's compiler (`served_identity`) as its code: φ, the default seed, ℓ and the served
    model name move it as they move a single-GPU one's."""
    serve = {"model": "m/x", "revision": "r", "default_seed": DEFAULT_SEED, "execution": "bi-eager", "target": TARGET}
    reg = PP.registered_of_serve(serve, None, None, PP.CAP_BITS)
    one = PP.I2Compiler(reg, {}, "", name=SERVER[0], code=SERVER[1])
    assert one.name == SERVER[0] and one.digest != SERVER[1] and len(one.digest) == 128
    monkeypatch.setattr(PP, "code_identity", lambda: SERVER[1])
    assert PP.I2Compiler(reg, {}, "", name=SERVER[0]).digest == one.digest
    assert PP.I2Compiler(reg, {}, "", name=SERVER[0], code="c0" * 64).digest != one.digest
    others = [PP.registered_of_serve(other, None, None, cap) for other, cap in (
        ({**serve, "model": "m/y"}, PP.CAP_BITS), ({**serve, "revision": "s"}, PP.CAP_BITS),
        ({**serve, "execution": "eager"}, PP.CAP_BITS), ({**serve, "target": {**TARGET, "device_name": "other"}}, PP.CAP_BITS),
        (serve, 2 * PP.CAP_BITS))] + [replace(reg, default_seed=DEFAULT_SEED + 1), replace(reg, served_model="llama")]
    digests = {PP.I2Compiler(r, {}, "", name=SERVER[0], code=SERVER[1]).digest for r in others}
    assert len(digests) == len(others) and one.digest not in digests


#: a served run's ``serve.json`` edited after its advice was registered, in each field the compile reads of it (`phi_of_serve`,
#: `registered_of_serve`, `served_gates`), and the verifier's own ``--hf-config`` other than the pin's config of record
SERVE_EDITS = {"model": {"model": "verity-int8/Llama-3.2-1B-INT8"}, "revision": {"revision": "0" * 40},
               "default_seed": {"default_seed": DEFAULT_SEED + 1}, "execution": {"execution": "eager"},
               "engine-execution": {"engine": {"execution": "eager", "pin": {"world": 1}}},
               "target": {"target": {**TARGET, "device_name": "other-gpu"}}, "hf_config": None}


@pytest.mark.parametrize("field", list(SERVE_EDITS))
def test_a_served_window_under_another_serve_json_than_its_advice_was_registered_under_is_refused(tmp_path, monkeypatch, field):
    """The served compiler's digest covers every field the compile reads of ``serve.json`` (`I2Compiler.digest`), so a window
    whose ``serve.json`` says another model, revision, execution or target than the one `verity-vllm advice` registered its
    advice under (or whose verifier compiles under another config) is refused by name before any commit (`serve_check`'s
    ``compiler``, and ``advice``: its advice tree's domain is the registered compiler), and the records audit's compiler
    table (`compiler_doc`, ``prove_path.py compiler``) holds another digest; one recording another default seed is refused
    outright (`registered_seed`); the honest one passes."""
    from verity_vllm.program.predict import steps as ST

    s = _served(tmp_path, monkeypatch)
    monkeypatch.setitem(PP.GATES_OF_RECORD, SERVE_EDITS["model"]["model"], PP.GATES_OF_RECORD["unsloth/Llama-3.2-1B"])
    units = [ST.RequestUnit(0, "d0" * 64, 5, 2, {}, None, 0, "r0"), ST.RequestUnit(1, "d1" * 64, 3, 1, {}, None, 2, "r1")]
    monkeypatch.setattr(ST, "compile_steps", lambda registered, data, ingress, named=False: SimpleNamespace(requests=units, named={}))
    _q, held, _o = PP.served_advice(s.a)
    assert held == s.bound and PP.serve_check_of(s.a, held)["ok"]
    assert PP.compiler_doc(str(s.run), None, None, PP.CAP_BITS)["digest"] == s.bound["compiler"]["digest"]
    serve = json.loads((s.run / "serve.json").read_text())
    if SERVE_EDITS[field] is None:
        config = tmp_path / "config.json"
        config.write_text(json.dumps({**json.loads(PP.LLAMA32_1B_CONFIG.read_text()), "rms_norm_eps": 1e-6}))
        s.a.hf_config = str(config)
    else:
        (s.run / "serve.json").write_text(json.dumps({**serve, **SERVE_EDITS[field]}))
    if field == "default_seed":
        for refused in (lambda: PP.served_advice(s.a), lambda: PP.compiler_doc(str(s.run), None, None, PP.CAP_BITS)):
            with pytest.raises(ValueError, match=f"filled in default seed {DEFAULT_SEED + 1}"):
                refused()
        return
    _c, _r, bound, _o = PP.compiled(s.a)
    assert bound["compiler"] != s.bound["compiler"] and PP.serve_check_of(s.a, bound)["codes"] == ["compiler", "advice"]
    with pytest.raises(ValueError, match=r"published records are refused \(compiler, advice\)"):
        PP.served_advice(s.a)
    assert PP.compiler_doc(str(s.run), s.a.hf_config, None, PP.CAP_BITS)["digest"] != s.bound["compiler"]["digest"]


def test_a_served_run_registers_the_public_default_seed_and_one_that_records_another_or_none_is_refused(tmp_path, monkeypatch,
                                                                                                         capsys):
    """A served window's registration fixes `engine.served_seed.DEFAULT_SEED`, as vllm's registered compiler does: a run
    whose serve.json says its server filled in another seed (7, 0, the public one as a string or a float) or none is refused
    by name, by the served compiler and its commands, never compiled under the seed the run recorded."""
    serve = {"model": "m/x", "revision": "r", "default_seed": DEFAULT_SEED, "execution": "bi-eager", "target": TARGET}
    assert PP.registered_seed(serve) == PP.registered_of_serve(serve, None, None, PP.CAP_BITS).default_seed == DEFAULT_SEED
    for seed in (7, 0, None, str(DEFAULT_SEED), float(DEFAULT_SEED)):
        why = f"serve.json: the run's server filled in default seed {seed!r}, and the registered one is engine.served_seed."
        with pytest.raises(ValueError, match=re.escape(why)):
            PP.registered_of_serve({**serve, "default_seed": seed}, None, None, PP.CAP_BITS)
    with pytest.raises(ValueError, match="filled in default seed None"):
        PP.registered_seed({k: v for k, v in serve.items() if k != "default_seed"})
    s = _served(tmp_path, monkeypatch)
    assert s.compiler.inner.registered.default_seed == DEFAULT_SEED
    recorded = json.loads((s.run / "serve.json").read_text())
    (tmp_path / "advice.bin").write_bytes(s.data)
    for doc in ({**recorded, "default_seed": 7}, {k: v for k, v in recorded.items() if k != "default_seed"}):
        (s.run / "serve.json").write_text(json.dumps(doc))
        why = f"filled in default seed {doc.get('default_seed')!r}"
        with pytest.raises(ValueError, match=re.escape(why)):
            PP.i2_of_serve(str(s.run), 0, s.a.row_dir, None, None)
        capsys.readouterr()
        assert PP.main(["circuit", "--serve-run", str(s.run), "--window", "0", "--partition", str(tmp_path / "part.json"),
                        "--advice", str(tmp_path / "advice.bin"), "--out", str(tmp_path / "circuit.json")]) == PP.REFUSED
        assert why in capsys.readouterr().err and not (tmp_path / "circuit.json").exists()


def test_a_stand_in_records_the_public_default_seed_and_its_seedless_sampled_requests_bring_the_records(tmp_path, monkeypatch):
    """`serve-standin` lays a declared Commit out as serve-openai does, so its serve.json records the public default seed and
    passes `registered_seed`; a sampled request of the record that named no seed brings the declared run's default seed, the
    one its sample used, and a greedy or seeded one brings what the record served it under."""
    phi = SimpleNamespace(model={"repo": "unsloth/Llama-3.2-1B", "revision": "9535bd9b1d1dea6acafbdc4813b728796aeb28da"},
                          execution="bi-eager", target=TARGET)
    ingress = {"r0": {"prompt": [0, 1, 2, 3, 4], "max_tokens": 3, "temperature": 0.0},
               "r1": {"prompt": [0, 1, 2], "max_tokens": 2, "temperature": 0.8},
               "r2": {"prompt": [5], "max_tokens": 2, "temperature": 0.8, "seed": 11}}
    record = SimpleNamespace(row={"label": "ROW"}, run=SimpleNamespace(run_root=bytes.fromhex("aa" * 64)))
    monkeypatch.setattr(PP, "record1", lambda bundle, row_dir: record)
    monkeypatch.setattr(PP, "i2_of_record", lambda *a, **k: SimpleNamespace(
        registered=SimpleNamespace(phi=phi, default_seed=5), ingress=ingress))
    monkeypatch.setattr(PP, "served_identity", lambda: SERVER)
    out = tmp_path / "serve"
    a = argparse.Namespace(bundle="b", row_dir=str(tmp_path / "rows"), workload=str(_workload(tmp_path)), out=str(out),
                           hf_config=None, cap_bits=PP.CAP_BITS)
    assert PP.serve_standin(a)["requests"] == 3
    serve = json.loads((out / "serve.json").read_text())
    assert serve["default_seed"] == PP.registered_seed(serve) == DEFAULT_SEED
    brought = {q["user"]: q["ingress"]["brought"] for q in map(json.loads, (out / "requests.jsonl").read_text().splitlines())}
    assert brought == {"r0": {"max_tokens": 3, "temperature": 0.0}, "r1": {"max_tokens": 2, "temperature": 0.8, "seed": 5},
                       "r2": {"max_tokens": 2, "temperature": 0.8, "seed": 11}}
    assert PP.compiler_doc(str(out), None, None, PP.CAP_BITS)["default_seed"] == DEFAULT_SEED


def test_the_served_identity_is_verity_vllm_advices_and_refused_where_the_tree_has_none(monkeypatch):
    import types

    PP.served_identity.cache_clear()
    fake = types.ModuleType("verity_vllm.pipeline.advice")
    fake.registered_compiler = lambda: SimpleNamespace(name=PP.CAPPED, digest="ff" * 64, inner=SimpleNamespace(name=SERVER[0], digest=SERVER[1]))
    monkeypatch.setitem(sys.modules, "verity_vllm.pipeline.advice", fake)
    assert PP.served_identity() == SERVER
    PP.served_identity.cache_clear()
    fake.registered_compiler = lambda: SimpleNamespace(name=SERVER[0], digest=SERVER[1])
    with pytest.raises(ValueError, match="registers no capped compiler"):
        PP.served_identity()
    monkeypatch.setitem(sys.modules, "verity_vllm.pipeline.advice", None)
    with pytest.raises(ValueError, match="which this tree lacks"):
        PP.served_identity()
    PP.served_identity.cache_clear()


def test_the_served_compiler_is_what_the_registration_must_name(tmp_path, monkeypatch, capsys):
    s = _served(tmp_path, monkeypatch)
    doc = PP.compiler_doc(str(s.run), None, None, PP.CAP_BITS)
    assert {"name": doc["name"], "digest": doc["digest"]} == s.bound["compiler"] == {"name": PP.CAPPED, "digest": s.compiler.digest}
    assert doc["served"] == {"name": SERVER[0], "digest": SERVER[1]} and doc["default_seed"] == DEFAULT_SEED and doc["phi"]["target"] == TARGET
    assert doc["transfer_gates"] == list(PP.GATES_OF_RECORD["unsloth/Llama-3.2-1B"])
    assert PP.compiler_doc(str(s.run), None, None, 2 * PP.CAP_BITS)["digest"] != doc["digest"]
    capsys.readouterr()
    assert PP.main(["compiler", "--serve-run", str(s.run)]) == 0
    assert json.loads(capsys.readouterr().out)["digest"] == doc["digest"]


#: `engine.int8_checkpoint`'s v1 recipe's quantization config, the one Llama-3.2-1B-INT8's config.json carries
INT8_V1 = {"modules_to_not_convert": ["lm_head"], "quant_method": "verity_int8", "weight_max": 127}


def _tiny_checkpoint(d: Path, model: str, revision: str, int8: bool = False) -> Path:
    """TINY's checkpoint under ``d`` (random bf16 weights by their HF names, one safetensors shard, and its config.json) and a
    manifest pinning the shard at ``model`` and ``revision``: the manifest. ``int8``: as `engine.int8_checkpoint`'s v1 recipe
    makes one, each decoder linear an I8 weight under its row rule and its F32 ``weight_scale`` [N, 1], under ``INT8_V1``."""
    import struct

    import numpy as np

    rng, H, I, D = np.random.default_rng(3), TINY["hidden_size"], TINY["intermediate_size"], 16

    def bf16(*shape):
        return (rng.random(shape, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)

    def linear(name, n, k):
        if not int8:
            return {name + ".weight": bf16(n, k)}
        w = rng.random((n, k), dtype=np.float32) - np.float32(0.5)
        s = np.maximum(np.abs(w).max(axis=1, keepdims=True), np.float32(1e-30)) / np.float32(127)
        return {name + ".weight": np.clip(np.rint(w / s), -127, 127).astype(np.int8), name + ".weight_scale": s}
    t = {"model.embed_tokens.weight": bf16(TINY["vocab_size"], H), "model.norm.weight": bf16(H)}
    for i in range(TINY["num_hidden_layers"]):
        p = f"model.layers.{i}."
        for name, n, k in (("self_attn.q_proj", 4 * D, H), ("self_attn.k_proj", 2 * D, H), ("self_attn.v_proj", 2 * D, H),
                           ("self_attn.o_proj", H, 4 * D), ("mlp.gate_proj", I, H), ("mlp.up_proj", I, H), ("mlp.down_proj", H, I)):
            t |= linear(p + name, n, k)
        t |= {p + "input_layernorm.weight": bf16(H), p + "post_attention_layernorm.weight": bf16(H)}
    hdr, off = {"__metadata__": {"format": "pt"}}, 0
    for name, a in t.items():
        hdr[name] = {"dtype": {"uint16": "BF16", "int8": "I8", "float32": "F32"}[a.dtype.name], "shape": list(a.shape),
                     "data_offsets": [off, off + a.nbytes]}
        off += a.nbytes
    head = json.dumps(hdr).encode()
    d.mkdir(parents=True)
    (d / "model.safetensors").write_bytes(struct.pack("<Q", len(head)) + head + b"".join(a.tobytes() for a in t.values()))
    (d / "config.json").write_text(json.dumps({**TINY, "rope_theta": 10000.0, **({"quantization_config": INT8_V1} if int8 else {})}))
    sha = hashlib.sha256((d / "model.safetensors").read_bytes()).hexdigest()
    man = d.parent / "checkpoints.json"
    man.write_text(json.dumps({"checkpoints": [{"repo": model, "revision": revision, "local_path": str(d),
                                                "files": {"model.safetensors": {"sha256": sha}}}]}))
    return man


def test_a_served_runs_weights_registration_is_the_committers_under_its_own_key(tmp_path, capsys, monkeypatch):
    """``weights-registration``, the committer's: the served run's weights (`window_weights.weights_registration` under the
    φ its window circuit lowers under, here at a grain TINY lowers at) committed under salts sampled from ``--key``, a 32-byte
    file drawn there (0600) the first time and read thereafter, once the committer has checked the checkpoint's shards
    against its manifest. Written to ``--out``: a root for each weight parameter but the rotary table's, what `verity-vllm
    advice --weights-of-record` records and ``window-circuit --pins`` takes (the embedding's table its own, its ``row-seg``
    rows), and what the verifier holds to its own weights
    record's shapes (`registration_pins`) without computing a root. With the engine's served rotary table in the run's
    ``private/cos_sin_cache.npy`` (serve-openai's) it roots that too, the other roots unchanged, and refuses by name a
    table outside the analytic table's bound. Another key gives other roots for every weight; a key
    file of another length, a checkpoint the manifest pins as another model than the run served, or a shard other than the
    one the manifest pins, is refused by name, nothing registered."""
    import functools

    import numpy as np

    from verity_vllm.pipeline import advice as ADV
    from verity_vllm.program import model as M
    from verity_vllm.program.predict import window_weights as WW
    from verity_vllm.program.predict.window_weights import Grain

    grain = Grain(gemm_words=512, rows=9, select=64)
    monkeypatch.setattr(WW, "weights_registration", functools.partial(WW.weights_registration, grain=grain))
    run, hf, ck = tmp_path / "serve", tmp_path / "tiny.json", tmp_path / "ck" / "snap"
    run.mkdir()
    serve = {"model": "unsloth/Llama-3.2-1B", "revision": "9535bd9b1d1dea6acafbdc4813b728796aeb28da", "default_seed": DEFAULT_SEED,
             "execution": "bi-eager", "target": {"compute_capability": [12, 0], "num_sms": 188}}
    (run / "serve.json").write_text(json.dumps(serve))
    hf.write_text(json.dumps(TINY))
    man = _tiny_checkpoint(ck, serve["model"], serve["revision"])
    out, key = tmp_path / "weights_registration.json", tmp_path / "private" / "weights.key"

    def register(key_file: Path) -> int:
        return PP.main(["weights-registration", "--serve-run", str(run), "--checkpoint-dir", str(ck), "--checkpoints",
                        str(man), "--hf-config", str(hf), "--key", str(key_file), "--out", str(out)])

    capsys.readouterr()
    assert register(key) == 0
    reg, said = json.loads(out.read_text()), json.loads(capsys.readouterr().out)
    assert len(key.read_bytes()) == 32 and key.stat().st_mode & 0o777 == 0o600
    assert key.read_bytes().hex() not in out.read_text()
    pins = ADV.registered_weights(reg, serve)
    assert said["roots"] == len(pins) == 16 and PP.pins_of((), [str(out)]) == pins
    assert [p for p in pins if p.endswith("/table")] == ["weights/model.embed_tokens.weight@64/table"]
    assert said["unrooted"] == sorted(reg["unrooted"]) and all("rotary_emb" in p for p in said["unrooted"])
    registered = PP.registered_of_serve(serve, str(hf), None, PP.CAP_BITS)
    record = WW.weights_record(registered, str(ck), str(man), grain=grain)
    assert WW.registration_pins(reg, record) == pins
    assert register(key) == 0 and json.loads(out.read_text()) == reg
    C = next(f for f in record["fields"] if f.get("pinned") == "analytic")["analytic"]["C"]
    table = run / "private" / "cos_sin_cache.npy"
    table.parent.mkdir()
    np.save(table, M.cos_sin_table(C))
    capsys.readouterr()
    assert register(key) == 0
    rooted, said = json.loads(out.read_text()), json.loads(capsys.readouterr().out)
    assert said["roots"] == 17 and said["unrooted"] == [] == list(rooted["unrooted"])
    assert {p: e for p, e in rooted["values"].items() if p not in reg["unrooted"]} == reg["values"]
    assert WW.registration_pins(rooted, record) == ADV.registered_weights(rooted, serve)
    np.save(table, M.cos_sin_table({**C, "ROPE_THETA": 2 * C["ROPE_THETA"]}))
    assert register(key) == PP.REFUSED
    assert "served words outside the analytic table's bound" in capsys.readouterr().err
    table.unlink()
    assert register(tmp_path / "other.key") == 0
    other = ADV.registered_weights(json.loads(out.read_text()), serve)
    assert sorted(other) == sorted(pins) and all(other[p] != pins[p] for p in pins)
    (tmp_path / "short.key").write_bytes(bytes(16))
    capsys.readouterr()
    assert register(tmp_path / "short.key") == PP.REFUSED
    assert "a committer's salt key is 32 bytes, not 16" in capsys.readouterr().err
    out.unlink()
    (run / "serve.json").write_text(json.dumps({**serve, "model": "org/other"}))
    assert register(key) == PP.REFUSED
    assert f"manifest entry is {serve['model']}@{serve['revision']}, not the registered model org/other@" in capsys.readouterr().err
    assert not out.exists()
    (run / "serve.json").write_text(json.dumps(serve))
    shard = ck / "model.safetensors"
    body = bytearray(shard.read_bytes())
    body[-1] ^= 1
    shard.write_bytes(bytes(body))
    assert register(key) == PP.REFUSED
    assert "the checkpoint is not of record" in capsys.readouterr().err and not out.exists()


def test_an_int8_served_runs_weights_registration_roots_its_scales_fused_linears_and_tied_embedding(tmp_path, capsys, monkeypatch):
    """The int8 Llama-3.2-1B is an exact-int8 checkpoint (`engine.int8_checkpoint`'s v1: each decoder linear an s8 weight and an f32
    row scale, the embedding bf16 and tied to the LM head), served under a target whose ``int8_exact`` is ``linears``, and
    ``weights-registration`` roots its weight parameters as the window circuit reads them: the s8 weights and their scales of
    q|k|v and gate|up fused as vLLM serves them, o and down, the norms, and the tied embedding at the embedding's and the LM
    head's row widths and as the embedding's table (its own parameter, `window_builder.table_param`), each the checkpoint's
    words; only the rotary table is unrooted. The verifier holds it to its own weights record's shapes (`registration_pins`)."""
    import functools

    from verity_vllm.pipeline import advice as ADV
    from verity_vllm.program.predict import window_weights as WW
    from verity_vllm.program.predict.window_weights import Grain

    grain = Grain(gemm_words=512, rows=9, select=64)
    monkeypatch.setattr(WW, "weights_registration", functools.partial(WW.weights_registration, grain=grain))
    run, hf, ck = tmp_path / "serve", tmp_path / "tiny-int8.json", tmp_path / "ck" / "snap"
    run.mkdir()
    serve = {"model": "verity-int8/Llama-3.2-1B-INT8", "revision": "15b0976d400d796ee589e7737d03a919526bf800", "default_seed": DEFAULT_SEED,
             "execution": "bi-eager", "target": {"compute_capability": [12, 0], "num_sms": 188, "int8_exact": "linears"}}
    (run / "serve.json").write_text(json.dumps(serve))
    hf.write_text(json.dumps({**TINY, "quantization_config": INT8_V1}))
    man = _tiny_checkpoint(ck, serve["model"], serve["revision"], int8=True)
    out = tmp_path / "weights_registration.json"
    capsys.readouterr()
    assert PP.main(["weights-registration", "--serve-run", str(run), "--checkpoint-dir", str(ck), "--checkpoints", str(man),
                    "--hf-config", str(hf), "--key", str(tmp_path / "private" / "weights.key"), "--out", str(out)]) == 0
    reg, said = json.loads(out.read_text()), json.loads(capsys.readouterr().out)
    pins = ADV.registered_weights(reg, serve)
    assert said["roots"] == len(pins) == 24 and said["unrooted"] == ["weights/model.layers.0.self_attn.rotary_emb.cos_sin_cache@16"]
    assert [p for p in pins if p.endswith("/table")] == ["weights/model.embed_tokens.weight@64/table"]
    words = {p.removeprefix("weights/"): e["words"] * e["leaves"] for p, e in reg["values"].items() if not p.endswith("/table")}
    H, I, QKV = TINY["hidden_size"], TINY["intermediate_size"], 8 * 16
    for i in range(TINY["num_hidden_layers"]):
        p = f"model.layers.{i}."
        assert {k: v for k, v in words.items() if k.startswith(p) and "_proj" in k} == {
            p + "self_attn.qkv_proj.weight@512": QKV * H // 2, p + "self_attn.qkv_proj.weight_scale@32": 2 * QKV,
            p + "self_attn.o_proj.weight@512": H * H // 2, p + "self_attn.o_proj.weight_scale@32": 2 * H,
            p + "mlp.gate_up_proj.weight@512": 2 * I * H // 2, p + "mlp.gate_up_proj.weight_scale@32": 2 * 2 * I,
            p + "mlp.down_proj.weight@512": H * I // 2, p + "mlp.down_proj.weight_scale@16": 2 * H}
    assert words["model.embed_tokens.weight@64"] == words["model.embed_tokens.weight@512"] == TINY["vocab_size"] * H
    assert not any("lm_head" in p for p in words)
    registered = PP.registered_of_serve(serve, str(hf), None, PP.CAP_BITS)
    assert WW.registration_pins(reg, WW.weights_record(registered, str(ck), str(man), grain=grain)) == pins


def test_the_transfer_gates_of_a_served_model_are_its_compiler_of_records():
    """`served_gates`: Llama-3.2-1B's compiler of record lists its RoPE heads (the TP1 stand-in), ``--transfer-gate`` may restate
    the list and nothing else (another template, or the record's with one more), and a model with no line lists none."""
    llama, tg = {"model": "unsloth/Llama-3.2-1B"}, "RoPEHead_v1{D=64}"
    assert PP.served_gates(llama) == PP.served_gates(llama, [tg, tg]) == (tg,)
    for other in (["SiluMul_v2{I=8192}"], [tg, "SiluMul_v2{I=8192}"]):
        with pytest.raises(ValueError, match=r"unsloth/Llama-3.2-1B's compiler of record lists \['RoPEHead_v1\{D=64\}'\]"):
            PP.served_gates(llama, other)
    assert PP.served_gates({"model": "m/x"}) == ()
    with pytest.raises(ValueError, match=r"m/x's compiler of record lists \[\]"):
        PP.served_gates({"model": "m/x"}, [tg])


def test_a_served_compiler_lists_its_models_transfer_gates_of_record_beside_the_servers(tmp_path, monkeypatch, capsys):
    """Its inner digest covers the server's registered one and the model's gates of record, so a registration under the server's
    compiler alone is refused before any commit (its advice tree's domain is another compiler's too), and one `verity-vllm
    advice` makes under the name and digest `compiler` prints passes; ``--transfer-gate`` only restates the list."""
    from verity_vllm.pipeline import advice as ADV

    s, tg = _served(tmp_path, monkeypatch, gates=()), "RoPEHead_v1{D=64}"
    doc = PP.compiler_doc(str(s.run), None, None, PP.CAP_BITS)
    assert doc["served"] == {"name": SERVER[0], "digest": SERVER[1]} and doc["transfer_gates"] == [tg]
    assert doc == PP.compiler_doc(str(s.run), None, None, PP.CAP_BITS, [tg])
    assert doc["inner"]["name"] == SERVER[0] and doc["inner"]["digest"] != SERVER[1] and doc["digest"] != s.bound["compiler"]["digest"]
    gated = PP.Capped(PP.i2_of_serve(str(s.run), 0, s.a.row_dir, None, None, transfer_gates=(tg, tg)))
    assert gated.transfer_gates == (tg,) and {"name": PP.CAPPED, "digest": gated.digest} == {"name": doc["name"], "digest": doc["digest"]}
    capsys.readouterr()
    assert PP.main(["compiler", "--serve-run", str(s.run), "--transfer-gate", tg]) == 0
    assert json.loads(capsys.readouterr().out)["inner"] == doc["inner"]
    _c, _r, bound, _o = PP.compiled(s.a)
    assert bound["compiler"] == {"name": doc["name"], "digest": doc["digest"]}
    assert PP.serve_check_of(s.a, bound)["codes"] == ["compiler", "advice"]
    with pytest.raises(ValueError, match=r"published records are refused \(compiler, advice\)"):
        PP.served_advice(s.a)
    _q, bound, reg = AW.of_advice(ADV.registered_compiler(doc["inner"]["name"], doc["inner"]["digest"], PP.CAP_BITS), s.data)
    adv = s.run / "advice" / "w0"
    (adv / "registration.json").write_text(json.dumps({**json.loads((adv / "registration.json").read_text()), "binding": bound}))
    (adv / "private" / "opening.json").write_text(json.dumps(AW.opening(s.data, reg)))
    requests, held, _o = PP.served_advice(s.a)
    assert held == bound and PP.serve_check_of(s.a, held)["ok"] and [q["digest"] for q in requests] == ["d0" * 64, "d1" * 64]


@pytest.mark.parametrize("gates", [(), ("SiluMul_v2{I=8192}",), ("RoPEHead_v1{D=64}", "SiluMul_v2{I=8192}")],
                         ids=["none", "another", "the-records-and-another"])
def test_a_served_window_registered_under_another_gate_list_than_its_models_of_record_is_refused(tmp_path, monkeypatch, gates):
    """The verifier's served compiler lists the model's gates of record whatever it is told (`served_gates`), so a registration
    under any other list names another compiler and is refused before any commit, and the verifier refuses to run under that
    list instead."""
    s = _served(tmp_path, monkeypatch, gates=gates)
    _c, _r, bound, _o = PP.compiled(s.a)
    assert bound["compiler"] != s.bound["compiler"] and PP.serve_check_of(s.a, bound)["codes"] == ["compiler", "advice"]
    with pytest.raises(ValueError, match=r"published records are refused \(compiler, advice\)"):
        PP.served_advice(s.a)
    if gates:
        s.a.transfer_gate = list(gates)
        with pytest.raises(ValueError, match=r"compiler of record lists \['RoPEHead_v1\{D=64\}'\]"):
            PP.compiled(s.a)


def test_the_prover_commits_a_served_window_under_the_published_binding(tmp_path, monkeypatch):
    """The server committed the advice once: the prover's requests are I2's from the opened advice and its binding is the published
    one (no re-commit, so no fresh root), and the window's records pass `serve_check`."""
    s = _served(tmp_path, monkeypatch)
    requests, bound, opening = PP.served_advice(s.a)
    assert bound == s.bound and opening["advice"] == s.data.hex()
    assert [q["digest"] for q in requests] == ["d0" * 64, "d1" * 64]
    assert PP.serve_check_of(s.a, bound) == {"ok": True, "codes": [], "detail": {
        "window": 0, "published_root": s.root, "commitment": SRT.ROWS_SCHEME, "compiler": s.bound["compiler"],
        "advice": s.bound["advice"], "run_id": "RUN/w0"}}
    by_id = PP.i2_against_record(s.a, requests)
    assert by_id["equal"] and by_id["units"]["r1"] == {"i2": "d1" * 64, "of_record": "d1" * 64, "equal": True}
    (Path(s.a.row_dir) / "manifest.json").unlink()
    assert PP.i2_against_record(s.a, requests) == {"units": {"r0": {"i2": "d0" * 64}, "r1": {"i2": "d1" * 64}},
                                                   "of_record": ["d0" * 64, "d1" * 64], "equal": True}


@pytest.mark.parametrize("kw, codes", [
    ({"uncapped": True}, ["cap", "compiler", "advice"]),
    ({"published": "aa" * 32}, ["sha512", "run_root"]),
    ({"scheme": "vllm-v1"}, ["sha512"]),
    ({"published": "cc" * 64}, ["run_root"]),
], ids=["compiler-without-the-cap", "vllm-v1-root", "store-not-rows-sha512", "another-commits-root"])
def test_a_served_window_is_refused_before_any_commit_by_name(tmp_path, monkeypatch, kw, codes):
    """A registration whose compiler carries no cap (the inner compiler, so its advice tree's domain is another too), a published
    root of 32 bytes, a row dir whose store isn't rows-sha512, or a published root that isn't the row dir's."""
    s = _served(tmp_path, monkeypatch, **kw)
    _c, _r, bound, _o = PP.compiled(s.a)
    assert PP.serve_check_of(s.a, bound)["codes"] == codes
    with pytest.raises(ValueError, match=rf"serve-openai window 0's published records are refused \({', '.join(codes)}\)"):
        PP.served_advice(s.a)


def test_a_served_windows_negatives_are_each_refused_by_name(tmp_path, monkeypatch):
    s = _served(tmp_path, monkeypatch)
    w = PP.serve_window(str(s.run), 0)
    assert {k: v["codes"] for k, v in PP.serve_negatives(w, s.compiler, s.bound, s.root, SRT.ROWS_SCHEME).items()} == {
        "registration-without-the-cap (serve)": ["cap", "compiler"],
        "registration-of-another-window (serve)": ["window"],
        "published-root-not-sha512 (serve)": ["sha512", "run_root", "commit"],
        "published-root-of-another-commit (serve)": ["run_root", "commit"]}
    ok = PP.source1({}, SimpleNamespace(run=s.root), {"commitment": "x", "served_programs": {}, "stand_in": {"of": 1, "link": 2}})
    both = PP.with_serve(ok, PP.serve_check(w, s.bound, "cc" * 64, SRT.ROWS_SCHEME))
    assert not both["ok"] and both["codes"][-1] == "run_root" and both["detail"]["serve"]["published_root"] == s.root


def test_a_served_run_is_its_window_and_row_dir_with_the_servers_advice_and_opening(tmp_path, monkeypatch):
    got = {}
    monkeypatch.setattr(PP, "commit", lambda a: got.update(vars(a)) or 0)
    base = ["--serve-run", "RUN", "--partition", "p.json", "--out", "w"]
    assert PP.main(["commit", *base, "--window", "3", "--row-dir", "R"]) == 0
    assert (got["bundle"], got["opening"]) == ("R/commit/replay_bundle_p0", "RUN/advice/w3/private/opening.json")
    for extra in ([], ["--window", "3"], ["--row-dir", "R"], ["--window", "3", "--row-dir", "R", "--workload", "f"],
                  ["--window", "3", "--row-dir", "R", "--advice", "a.bin"], ["--window", "3", "--row-dir", "R", "--tp-commit", "c"]):
        with pytest.raises(SystemExit):
            PP.main(["commit", *base, *extra])
    assert PP.single(argparse.Namespace(serve_run="RUN")) and PP.ranks_of(argparse.Namespace(serve_run="RUN", ranks=None)) == (1, [0])


def test_the_verifiers_i2_reads_only_builds_it_wrote_itself_and_the_provers_its_row_dir(tmp_path, monkeypatch):
    """No Program digest covers a Build's row names or its parameters' names (nor its ``supplied``, correspondence or
    ``prefix_cache.json``), so the audit's I2 (``--builds``) reads them only from its own Build of each Program it compiled
    (`own_builds`: each Program once, written into ``--builds`` beside the names its compile gives it, `naming.write_build`),
    whatever the row dir holds under the same digests; the prover's I2 compiles no names and reads its row dir. ``--builds``
    defaults to OUT/builds and is refused under the row dir, ``--vllm-python`` is accepted and ignored, and the compile command
    takes no row dir."""
    from verity_vllm.program.predict import naming as N, steps as ST

    s = _tiny_served(tmp_path, monkeypatch)
    rows = _row_dir(Path(s.a.row_dir), s, ("model.model.layers.0.mlp.act_fn/", "model.model.layers.1.mlp.act_fn/"), params=True)
    written, write_build = [], N.write_build
    monkeypatch.setattr(N, "write_build", lambda out, program, named: written.append(out) or write_build(out, program, named))
    builds, args = tmp_path / "builds", {**vars(s.a), "transfer_gate": []}
    mine = PP.compiler_of_args(argparse.Namespace(**args, builds=str(builds)))
    theirs = PP.compiler_of_args(argparse.Namespace(**args))
    p = mine.load(0, 1, PP.compile_advice(mine, s.data))
    assert written == [str(builds / ST.trace_dir(u)) for u in s.units]
    q = theirs.load(0, 1, PP.compile_advice(theirs, s.data))
    assert s.compiles == [True, False]

    def named(inst: dict) -> dict[int, str]:
        return {int(x["i"]): x["name"] for x in inst["rows"]}

    for u in s.units:
        n, r = s.named[(u.LP, u.T)], u.id
        assert (p.digs[r], q.digs[r]) == (u.digest, u.digest) and (Path(p.dirs[r]).parent, Path(q.dirs[r]).parent) == (builds, rows)
        assert named(p.progs[r]) == n.names != named(q.progs[r])
        assert p.progs[r]["params"] == n.params != q.progs[r]["params"]
    mine.load(0, 1, PP.compile_advice(mine, s.data))
    assert len(written) == len(s.units)
    with pytest.raises(ValueError, match=f"request rx: I2 compiled no Program {'ab' * 8} here"):
        PP.own_builds(mine.inner, [{"request_id": "rx", "digest": "ab" * 64}])

    got = []
    monkeypatch.setattr(PP, "audit", lambda a: got.append(a) or 0)
    run = ["--serve-run", str(s.run), "--window", "0", "--row-dir", str(rows), "--windows", "w", "--partition", "p", "--template-sets",
           "t", "--m0-pythonpath", "m", "--bin", "b", "--lean", "l", "--firewall", "f", "--out", str(tmp_path / "out")]
    assert PP.main(["audit", *run]) == 0 and PP.main(["audit", *run, "--vllm-python", "pinned-python"]) == 0
    assert [a.builds for a in got] == [str(tmp_path / "out" / "builds")] * 2
    with pytest.raises(SystemExit):
        PP.main(["audit", *run, "--builds", str(rows / "builds")])
    with pytest.raises(SystemExit):
        PP.main(["circuit", "--serve-run", str(s.run), "--window", "0", "--row-dir", str(rows), "--partition", "p", "--advice", "a",
                 "--out", "o"])


def test_the_compile_command_writes_its_own_builds_with_no_vllm(tmp_path, monkeypatch):
    """``circuit`` (the served compile check's command) compiles the advice with its names, writes its own Build of each
    Program and records the circuit over them with no vLLM importable: the auditor needs no serving stack."""
    monkeypatch.setitem(sys.modules, "vllm", None)
    s = _tiny_served(tmp_path, monkeypatch)
    (tmp_path / "part.json").write_text(json.dumps(LAYER0))
    (tmp_path / "advice.bin").write_bytes(s.data)
    out = tmp_path / "mine" / "circuit.json"
    out.parent.mkdir()
    assert PP.main(["circuit", "--serve-run", str(s.run), "--window", "0", "--partition", str(tmp_path / "part.json"),
                    "--advice", str(tmp_path / "advice.bin"), "--out", str(out)]) == 0
    assert s.compiles == [True] and list(out.parent.iterdir()) == [out] and sys.modules["vllm"] is None
    assert json.loads(out.read_text())["templates"]


def test_the_committers_key_is_made_once_private_and_read_back(tmp_path):
    """`committer_key`: 32 bytes made mode 0600 where the file doesn't exist, the same bytes read back every window after;
    a file of another size is refused."""
    p = tmp_path / "private" / "committer.key"
    k = PP.committer_key(p)
    assert len(k) == 32 and (p.stat().st_mode & 0o777) == 0o600 and PP.committer_key(p) == k
    p.write_bytes(bytes(31))
    with pytest.raises(ValueError, match="32 bytes, not 31"):
        PP.committer_key(p)
