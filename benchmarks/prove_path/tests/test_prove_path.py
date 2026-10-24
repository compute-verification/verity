"""The prove path's compiler, its advice commitment and its draw-order check, and its prover's and verifier's windows on the CPU
TP stand-in (integrations/vllm's `tests.pipeline.served_tp`, deferred), without proving.

* The compiler is total: every byte string is the requests or a `ValueError` naming why, the same on every run; random and
  truncated advice is refused, advice over the registered cap ℓ is refused before it is read, and a malformed opening is
  refused the same way.
* The advice is fixed before the draw: `advice_before_draw` holds the registration the verifier opens the advice under to the
  one the draw's receipt names, and refuses the advice re-committed after the coins.
* On the stand-in: each rank's window over the compiler's Programs for the advice is held to the verifier's own from the
  opening, the advice's negatives are refused by name, and a forged unit is committed consistently (one bit of its rows
  changed, every check before the draw passing), so only the proof can refuse it.
* A single-GPU run's pieces without its Commit: I2's adapter (its requests, its refusal by name, its Program bodies by digest),
  the requests' bodies from the record, the stand-in window's link and `source1`'s codes, and the CLI's one committed run."""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from verity.core.protocols.sampled_proofs.one_stage import registration as R
from verity_vllm.commit import serving_rows as CS
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


def test_the_advice_must_be_in_the_registration_the_draw_followed():
    rec = _rec("11" * 64)
    drawn = R.receipt(rec, A5.VERIFIER, received="0")
    assert PP.advice_before_draw(drawn, rec, _bound(rec)) == {"ok": True, "codes": [], "detail": {
        "drawn_after": R.digest(rec), "registration": R.digest(rec), "received": "0"}}
    late = _rec("22" * 64)
    assert PP.advice_before_draw(drawn, late, _bound(late))["codes"] == ["after-the-draw"]
    assert PP.advice_before_draw(R.receipt(rec, "one-stage-e2e verifier (a5.py)"), rec, _bound(rec))["codes"] == ["after-the-draw"]
    assert PP.advice_before_draw(drawn, rec, _bound(late))["codes"] == ["advice"]
    other = _rec("11" * 64, compiler="ef" * 64)
    assert PP.advice_before_draw(drawn, rec, _bound(other))["codes"] == ["compiler"]


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
def test_advice_committed_after_the_draw_passes_every_check_before_it_and_is_refused(tp):
    """With the receipt the stream writes on receiving the honest registration, the honest registration holds; the same advice
    committed again after the coins (a fresh root, a true commitment of it, which `source` accepts with its opening) is refused."""
    compiler, bound, opening, mine, held = _mine(tp, "honest", 0)
    rec = _registered(tp["honest"] / "rank0")
    receipt = R.receipt(rec, A5.VERIFIER, received="0")
    assert PP.advice_before_draw(receipt, rec, bound)["ok"]
    got = PP.advice_after_draw(compiler, receipt, rec, opening, mine, held)
    assert got == {"refused": True, "codes": ["after-the-draw"], "before_draw_checks_pass": True}


@pytest.mark.slow
def test_a_forged_unit_passes_every_check_before_the_draw_and_differs_in_one_bit(tp):
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


def test_i2s_requests_are_its_compiled_units_once_per_advice_and_its_refusal_is_named(monkeypatch):
    from verity_vllm.predict import steps as ST
    from verity_vllm.predict.common import Unexpressible

    calls = []

    def compile_steps(registered, data, ingress):
        calls.append(data)
        if data == b"no":
            raise Unexpressible("a chunked prefill")
        return SimpleNamespace(requests=[ST.RequestUnit(0, "d0" * 64, 5, 2, {}, None, 0, "r0"),
                                         ST.RequestUnit(1, "d1" * 64, 3, 1, {}, None, 2, "r1")])

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
