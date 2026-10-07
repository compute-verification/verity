"""`verity.ml.circuits.silicon.relation`: the step relation is complete on silicon (every capture satisfies it and yields the measured
word), locally sound (each witness column is constrained), and its census is what the module claims."""
from __future__ import annotations

import gzip
import json
import random
from array import array
from pathlib import Path

import pytest

from verity.ml.circuits.silicon import relation as R
from verity.ml.circuits.silicon.silicon import AMPERE_BF16_M16N8K16 as AMP
from verity.ml.circuits.silicon.total import HOPPER_BF16_WGMMA_K16 as HOP, cvt_rn_bf16_f32, fp32_value, pack_value, tc_dot_total

FIXTURES = Path(__file__).parent / "fixtures"
PIPES = {"ampere": AMP, "hopper": HOP}
STRIDE = 7   # the captures are replayed at this stride to keep the pure-Python relation under a few seconds


def _run(pipe, acc: int, a, b, align=R.ALIGN_DIV) -> tuple[int, R.Census]:
    w = R.step_witness(pipe, R.decode_state(acc, pipe), list(a), list(b), align)
    c = R.Census()
    out = R.check_step(pipe, w, c)
    return R.pack_state(out), c


# ---- completeness: every capture satisfies the relation and the relation's output is the measured word -----------------

def _words(packed: str) -> list[int]:
    """Sixteen BF16 words from the captures' 64-hex-digit operand string."""
    return [int(packed[i:i + 4], 16) for i in range(0, 64, 4)]


def test_ampere_golden_records() -> None:
    doc = json.loads((FIXTURES / "golden" / "ada_bf16_m16n8k16.json").read_text())
    n = 0
    for rec in doc["records"]:
        acc, a, b, d = int(rec["acc"], 16), _words(rec["a"]), _words(rec["b"]), int(rec["d"], 16)
        for align in (R.ALIGN_DIV, R.ALIGN_TABLE):
            got, _ = _run(AMP, acc, a, b, align)
            assert got == d, rec
        n += 1
    assert n == 360


@pytest.mark.parametrize("device", ["a100", "4090"])
def test_special_value_campaign(device: str) -> None:
    with gzip.open(FIXTURES / "tc-total-2026-09-07" / device / "mma_special.reduced.json.gz", "rt") as f:
        doc = json.load(f)
    assert doc["n_nondeterministic"] == 0
    n = 0
    for case in doc["cases"][::STRIDE]:
        acc, a, b, d = int(case["acc"], 16), _words(case["a"]), _words(case["b"]), int(case["d"], 16)
        got, _ = _run(AMP, acc, a, b)
        assert got == d == tc_dot_total(AMP, acc, a, b), case["name"]
        n += 1
    for case in doc["pos"][::STRIDE]:
        acc, a, b, d = int(case["acc"], 16), _words(case["a"]), _words(case["b"]), int(case["d"][0], 16)
        got, _ = _run(AMP, acc, a, b, R.ALIGN_TABLE)
        assert got == d, case["name"]
        n += 1
    assert n >= (len(doc["cases"]) + len(doc["pos"])) // STRIDE


def test_hopper_campaign() -> None:
    with gzip.open(FIXTURES / "tc-hopper-2026-09-07" / "special.reduced.json.gz", "rt") as f:
        doc = json.load(f)
    assert doc["tiles_differing_between_reps"] == 0
    n = 0
    for fam, cases in doc["families"].items():
        for case in cases[::STRIDE]:
            acc, a, b, d = int(case["acc"], 16), _words(case["a"]), _words(case["b"]), int(case["d"], 16)
            got, _ = _run(HOP, acc, a, b, R.ALIGN_DIV if n % 2 else R.ALIGN_TABLE)
            assert got == d == tc_dot_total(HOP, acc, a, b), (fam, case)
            n += 1
    assert n >= doc["records"] // STRIDE - len(doc["families"])


def _u16(path: Path) -> list[int]:
    arr = array("H")
    arr.frombytes(path.read_bytes())
    return list(arr)


def test_verification_unit_on_captured_gemm_coordinates() -> None:
    """A proof unit: zero state, K/16 chained steps, then `pack_relation`; the bf16 word is the committed one (4 coordinates)."""
    root = FIXTURES / "gemm-B1-b3eba618b1b5-20260907T1731Z"
    dirs = sorted(p for p in root.iterdir() if p.is_dir())
    picked = [dirs[0], dirs[7], dirs[15], dirs[-1]]
    for d in picked:
        f = json.loads((d / "fixture.json").read_text())
        x = _u16(d / "x.u16")
        if f["x_arg_cols"] is not None:
            lo, hi = f["x_arg_cols"]
            x = x[lo:hi]
        w = _u16(d / "w_row.u16")
        K = f["K"]
        st = R.zero_state(AMP)
        c = R.Census()
        for s in range(K // 16):
            st = R.check_step(AMP, R.step_witness(AMP, st, x[16 * s:16 * s + 16], w[16 * s:16 * s + 16]), c)
        word, bf16 = R.pack_relation(st, AMP, c)
        assert bf16 == f["committed_word"], d.name
        # the same proof unit's total against the published census
        vc = R.vu_census(AMP, K)
        assert c.summary()["mul"] == vc["total"]["mul"] and c.lookup_total == vc["total"]["lookup"]


# ---- boundary --------------------------------------------------------------------------------------------------------

def test_pack_relation_is_pack_value_then_cvt() -> None:
    rng = random.Random(2)
    words = [0, 0x80000000, 0x3F800000, 0x3F808000, 0x3F818000, 0x7F7FFFFF, 0x7F800000, 0xFF800000, 0x7FC00000,
             0xFFC00001, 0x00000001, 0x007FFFFF, 0x00800000, 0xFFFFFFFF] + [rng.getrandbits(32) for _ in range(3000)]
    for word in words:
        st = R.decode_state(word, AMP)
        canonical = pack_value(fp32_value(word))          # the model's own canonicalisation (-0 -> +0, NaN -> 0x7FFFFFFF)
        assert R.pack_state(st) == canonical
        c = R.Census()
        w, bf = R.pack_relation(st, AMP, c)
        assert (w, bf) == (canonical, cvt_rn_bf16_f32(canonical)), hex(word)


# ---- the witness's shape -------------------------------------------------------------------------------------------------

def test_a_witness_of_the_wrong_shape_is_rejected() -> None:
    """A witness whose lists are shorter than the relation's is refused, not checked on what it holds.  Before the check,
    sixteen products of 1.0 were accepted as 1.0: the honest witness of one product and fifteen zeros, truncated to one
    product row and two term rows, with the real operands."""
    ones, one_then_zeros = [0x3F80] * 16, [0x3F80] + [0] * 15
    w = R.step_witness(HOP, R.decode_state(0, HOP), one_then_zeros, one_then_zeros)
    w["prod"] = w["prod"][:1]
    w["groups"][0]["terms"] = w["groups"][0]["terms"][:2]
    w["a"], w["b"] = ones, ones
    with pytest.raises(R.Violation, match="witness shape"):
        R.check_step(HOP, w, R.Census())
    c = R.Census(strict=False)
    R.check_step(HOP, w, c)
    assert c.violations and tc_dot_total(HOP, 0, ones, ones) == 0x41800000

    def mutated(pipe, change):
        w = R.step_witness(pipe, R.zero_state(pipe), ones, ones)
        change(w)
        return w
    for pipe, change in [
        (HOP, lambda w: w["prod"].append(dict(w["prod"][0]))),                        # an extra product row
        (HOP, lambda w: w["groups"][0]["terms"].append(dict(w["groups"][0]["terms"][0]))),  # an extra term row
        (HOP, lambda w: w.update(a=w["a"][:8], b=w["b"][:8])),                        # short operands
        (AMP, lambda w: w["groups"].pop()),                                           # a missing group
    ]:
        with pytest.raises(R.Violation, match="witness shape"):
            R.check_step(pipe, mutated(pipe, change), R.Census())
    assert R.witness_shape_error(HOP, R.step_witness(HOP, R.zero_state(HOP), ones, ones)) is None


# ---- local soundness: every witness column is constrained ------------------------------------------------------------

def _flatten(w: dict) -> list[tuple[list, str]]:
    """(container, key) for every integer witness column of a step witness, output state included."""
    cols: list[tuple[list, str]] = []
    for q in w["prod"]:
        for k in q:
            cols.append((q, k))
    for g in w["groups"]:
        for k, v in g.items():
            if k == "terms":
                for t in v:
                    for kk in t:
                        cols.append((t, kk))
            elif k == "out":
                pass
            elif isinstance(v, int):
                cols.append((g, k))
    return cols


@pytest.mark.parametrize("name", ["ampere", "hopper"])
@pytest.mark.slow
def test_every_column_is_constrained(name: str) -> None:
    """Perturbing any single witness column (or any field of any group's output state) breaks some predicate.  This is
    the local-soundness check the census rests on; it is not a proof of the argument system."""
    pipe = PIPES[name]
    rng = random.Random(3)
    S16 = [0x7FC0, 0x7F80, 0xFF80, 0, 0x8000, 0x3F80, 0x7F7F, 0x0001]
    for trial in range(12):
        acc = rng.choice([0, 0x3F800000, 0x7F800000, 0x7FC00000, rng.getrandbits(32), rng.getrandbits(32)])
        a = [rng.choice(S16) if rng.random() < 0.2 else rng.getrandbits(16) for _ in range(16)]
        b = [rng.choice(S16) if rng.random() < 0.2 else rng.getrandbits(16) for _ in range(16)]
        base = R.step_witness(pipe, R.decode_state(acc, pipe), a, b)
        R.check_step(pipe, base, R.Census())
        for cont, key in _flatten(base):
            for delta in (1, -1, 1 << 20):
                w = R.step_witness(pipe, R.decode_state(acc, pipe), a, b)
                # relocate (cont, key) inside the fresh copy
                tgt = _locate(w, base, cont)
                tgt[key] = tgt[key] + delta
                c = R.Census(strict=False)
                try:
                    R.check_step(pipe, w, c)
                except Exception as e:  # a perturbed column may make a lookup key non-integer etc.
                    c.violations.append(repr(e))
                assert c.violations, (name, trial, key, delta)
        # output state fields
        for gi, g in enumerate(base["groups"]):
            for fld in ("nan", "inf", "neg", "e", "M"):
                w = R.step_witness(pipe, R.decode_state(acc, pipe), a, b)
                out = w["groups"][gi]["out"]
                w["groups"][gi]["out"] = R.State(**{**out.__dict__, fld: getattr(out, fld) + 1})
                if gi + 1 < len(w["groups"]):
                    pass  # the next group's witness still refers to the unperturbed state; predicates must notice
                c = R.Census(strict=False)
                try:
                    R.check_step(pipe, w, c)
                except Exception as e:
                    c.violations.append(repr(e))
                assert c.violations, (name, trial, gi, fld)


def _locate(fresh: dict, base: dict, cont: dict) -> dict:
    """The dict in `fresh` at the same position as `cont` in `base`."""
    for q_b, q_f in zip(base["prod"], fresh["prod"]):
        if q_b is cont:
            return q_f
    for g_b, g_f in zip(base["groups"], fresh["groups"]):
        if g_b is cont:
            return g_f
        for t_b, t_f in zip(g_b["terms"], g_f["terms"]):
            if t_b is cont:
                return t_f
    raise KeyError("container not found")


# ---- the census --------------------------------------------------------------------------------------------------------

def test_census_is_instance_independent_and_pinned() -> None:
    """The headline numbers.  If a change to the relation moves them, update the table in the module docstring / report."""
    amp, amp_t = R.census(AMP), R.census(AMP, R.ALIGN_TABLE)
    hop, hop_t = R.census(HOP), R.census(HOP, R.ALIGN_TABLE)
    assert (amp["mul"], amp["lin"], amp["lookup"]) == (146, 104, 172)
    assert (amp_t["mul"], amp_t["lin"], amp_t["lookup"]) == (114, 72, 124)
    assert (hop["mul"], hop["lin"], hop["lookup"]) == (121, 84, 150)
    assert (hop_t["mul"], hop_t["lin"], hop_t["lookup"]) == (89, 52, 102)
    assert amp["lookups_by_table"]["T_bf16"] == 32 and amp["lookups_by_table"]["T_pcls"] == 16
    assert amp["table_rows"]["T_bf16"] == 1 << 16 and amp["table_rows"]["T_pcls"] == 64
    assert amp_t["table_rows"]["T_shift26"] == (1 << 16) * 27
    vu = R.vu_census(AMP, 8960)
    assert vu["steps"] == 560 and vu["total"]["mul"] == 560 * 146 + vu["boundary"]["mul"]
