"""Generate the CK-SELECT vectors from the REGISTRY semantics (never from the guest): ``conformance.json`` (rows with
ties, NaN, -inf, +inf, signed zeros, all-equal, subnormals; expected index = ``verity.ir.evaluate`` over the
``TokenSelect_v1{V}`` program, cross-checked with the pure-integer mirror) and ``negatives.json`` (>= 12 statement/witness
pairs a sound checker must reject, with the code the guest's mirror gives), each case a complete statement + witness over
a synthetic RU-shaped tree so ``veritor-select-host vectors --file`` runs them through the ELF.

    PYTHONPATH=. .venv/bin/python e2e/coverage_v1/select_vectors/make_vectors.py [--record <REC dir>]

``--record`` adds ``record_tiny.json``: statements built from a real record's committed rows (``ck_select.build``) plus
one forged row.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
for p in (REPO, REPO / "vllm-poc"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from e2e.coverage_v1 import ck_select as CS  # noqa: E402

SCHEMA = "veritor/e5/select-vectors/v1"
PROFILE = "p=e5vectors00/r=00000000-0000-5000-8000-000000000000"

# bf16 words
INF, NINF, NAN, NAN2 = 0x7F80, 0xFF80, 0x7FC0, 0x7F81
ZERO, NZERO, ONE, TWO, THREE, HALF, MONE, MTWO, SUBN = 0x0000, 0x8000, 0x3F80, 0x4000, 0x4040, 0x3F00, 0xBF80, 0xC000, 0x0001


def _ids(t: int) -> tuple[str, str]:
    return (f"{PROFILE}/t={t}/ph=decode/l=lm_head/op=lm_head.gemm/port=0", f"{PROFILE}/t={t}/ph=decode/l=sampler/op=sampler.select/port=0")


class Tree:
    """A synthetic tree in the serving side's shape: ``leaf_count`` leaves, our two at ``logits_rank`` / ``select_rank``."""

    def __init__(self, leaves: list[bytes]) -> None:
        self.levels = [leaves]
        while len(self.levels[-1]) > 1:
            lvl = self.levels[-1]
            nxt = [CS._H(CS.NODE_TAG, lvl[k], lvl[k + 1]) if k + 1 < len(lvl) else CS._H(CS.LIFT_TAG, lvl[k]) for k in range(0, len(lvl), 2)]
            self.levels.append(nxt)
        self.root = self.levels[-1][0]

    def path(self, index: int) -> tuple[bytes | None, ...]:
        out: list[bytes | None] = []
        cursor = index
        for lvl in self.levels[:-1]:
            sib = cursor ^ 1
            out.append(lvl[sib] if sib < len(lvl) else None)
            cursor //= 2
        return tuple(out)


def make_pair(words: list[int], token: int, *, t: int = 3, leaf_count: int = 7, logits_rank: int = 4, select_rank: int = 6,
              lits: CS.Literals = CS.REGISTRY_LITERALS, index_bits: int = CS.INDEX_BITS) -> tuple[CS.SelectStatement, CS.SelectWitness]:
    V = len(words)
    logits = struct.pack(f"<{V}H", *words)
    select = struct.pack("<I", token & 0xFFFFFFFF)
    lid, sid = _ids(t)
    leaves = [CS._H(CS.LEAF_TAG, f"{PROFILE}/t={t}/ph=decode/l={k}/op=layers.{k}.mlp.down_proj.gemm/port=0".encode(), b"bfloat16", b"[8]", bytes(16)) for k in range(leaf_count)]
    leaves[logits_rank] = CS.leaf_hash(lid, CS.LOGITS_DTYPE, f"[{V}]", logits)
    leaves[select_rank] = CS.leaf_hash(sid, CS.SELECT_DTYPE, "[]", select)
    tree = Tree(leaves)
    st = CS.SelectStatement(V, index_bits, lits, f"ru/synthetic/{t:06d}", tree.root, leaf_count, logits_rank, lid, select_rank, sid)
    w = CS.SelectWitness(logits, tree.path(logits_rank), select, tree.path(select_rank))
    return st, w


def case(name: str, words: list[int], *, note: str, token: int | None = None, expect: str = "accept", lits: CS.Literals = CS.REGISTRY_LITERALS,
         oracle: bool = True, **kw) -> dict:
    mirror = CS.argmax_pinned(words, lits)
    orc = CS.argmax_oracle(words) if oracle and lits == CS.REGISTRY_LITERALS else None
    if orc is not None and orc != mirror:
        raise SystemExit(f"{name}: registry oracle {orc} != mirror {mirror}")
    tok = mirror if token is None else token
    st, w = make_pair(words, tok, lits=lits, **kw)
    sb, wb = st.encode(), w.encode()
    got = CS.verify_transparent(sb, wb)
    if got != expect:
        raise SystemExit(f"{name}: mirror says {got}, the case expects {expect}")
    return {"name": name, "V": len(words), "words_hex": [f"{x:04x}" for x in words] if len(words) <= 64 else None, "literals": lits.as_tuple(),
            "expected_index": mirror, "registry_oracle_index": orc, "committed_token": tok, "expect": expect, "note": note,
            "statement_hex": sb.hex(), "witness_hex": wb.hex(), "statement": st.to_json()}


def conformance() -> list[dict]:
    import random

    rng = random.Random(20260911)
    cases = [
        case("unique_max", [ONE, TWO, THREE, HALF, MONE, ZERO, TWO, ONE], note="one strict maximum (index 2)"),
        case("tie_lowest_index", [ONE, TWO, TWO, HALF, TWO], note="three equal maxima: the LOWEST index (1) -- Bf16GtStrict is strict"),
        case("tie_all_equal", [HALF] * 9, note="all equal: 0 (no step is strictly greater)"),
        case("tie_all_equal_neg_inf", [NINF] * 6, note="a row of -inf: 0 (-inf > -inf is false)"),
        case("neg_inf_then_finite", [NINF, NINF, MTWO, NINF], note="-inf everywhere but one finite: that index (2)"),
        case("nan_at_index0_wins_by_init", [NAN, ONE, TWO, THREE], note="NaN at index 0 is the initial carry; x > NaN is false for every x: 0"),
        case("nan_inside_never_wins", [ONE, NAN, THREE, TWO, NAN2], note="NaN candidates at i >= 1 never replace the carry: the max among non-NaN (2)"),
        case("nan_after_max", [THREE, ONE, NAN], note="NaN after the maximum: unchanged (0)"),
        case("all_nan", [NAN, NAN2, 0xFFC0, NAN], note="all NaN: 0 (nothing exceeds a NaN carry)"),
        case("pos_inf_wins", [ONE, THREE, INF, TWO], note="+inf is the maximum (2)"),
        case("pos_inf_tie_lowest", [INF, ONE, INF], note="two +inf: lowest index (0)"),
        case("neg_zero_then_pos_zero", [NZERO, ZERO], note="-0.0 then +0.0: equal under f32 >, keep 0"),
        case("pos_zero_then_neg_zero", [ZERO, NZERO], note="+0.0 then -0.0: equal, keep 0"),
        case("neg_one_then_signed_zeros", [MONE, NZERO, ZERO], note="-1, -0, +0: -0 > -1 (index 1); +0 > -0 is false"),
        case("max_at_last_index", [ONE] * 7 + [TWO], note="maximum at V-1 (7)"),
        case("all_negative", [MTWO, MONE, MTWO, 0xC040], note="all negative: the least negative (1)"),
        case("subnormal_beats_zero", [ZERO, SUBN], note="smallest positive subnormal > +0 (1)"),
        case("negative_subnormal_loses_to_zero", [0x8001, ZERO], note="-subnormal then +0: +0 > -subnormal (1)"),
        case("two_elements_min_V", [TWO, THREE], note="V = 2, the smallest TokenSelect (1)"),
        case("index_bits_18_boundary", [ONE] * 64, note="index 0 in an all-equal row of 64 (unit width 18 bits declared)"),
    ]
    # random small rows, oracle-checked
    for k in range(6):
        V = rng.choice([3, 5, 8, 13, 21, 34])
        pool = [INF, NINF, NAN, ZERO, NZERO, ONE, TWO, MONE] + [rng.randrange(65536) for _ in range(8)]
        words = [rng.choice(pool) for _ in range(V)]
        cases.append(case(f"random_small_{k}", words, note="random small row with specials; expected index from the registry oracle"))
    # a real-size row (V = 49152): oracle-checked once (slow), ties planted at the max
    V = 49152
    words = [rng.randrange(0, 0x4000) for _ in range(V)]  # positive finite, below 2.0
    words[1234] = words[40000] = 0x4100  # tie at the max: lowest wins
    words[777] = NAN
    words[9] = NINF
    cases.append(case("vocab_49152_tie_nan_neginf", words, note="V = 49152 (the SMALL vocabulary): a tie at the max (1234, 40000), a NaN, a -inf; lowest tied index"))
    # literal application: with init_i = 2 the guest reports position + 1; the statement's literal is applied, not assumed
    cases.append(case("literal_init_i_2_applied", [ONE, THREE, TWO], token=2, lits=CS.Literals(0, 2, 1), oracle=False,
                      note="guest semantics with a different literal (init_i = 2): the maximum at position 1 is reported as 2. The VERIFIER "
                           "refuses this statement by name (`literal init_i: statement 2 != program 1`, negatives/literal_changed_statement)"))
    return cases


def negatives() -> list[dict]:
    words = [ONE, TWO, THREE, HALF, MONE, ZERO, TWO, ONE]  # argmax 2
    base_st, base_w = make_pair(words, 2)
    out: list[dict] = []

    def neg(name: str, st: CS.SelectStatement | bytes, w: CS.SelectWitness | bytes, expect: str, note: str, **extra) -> None:
        sb = st if isinstance(st, bytes) else st.encode()
        wb = w if isinstance(w, bytes) else w.encode()
        got = CS.verify_transparent(sb, wb)
        if got != expect:
            raise SystemExit(f"{name}: mirror says {got}, expected {expect}")
        out.append({"name": name, "expect": expect, "note": note, "statement_hex": sb.hex(), "witness_hex": wb.hex(), **extra})

    # relation negatives: the RECORD committed the wrong token (the tree is consistent, the function disagrees)
    st, w = make_pair(words, 3)
    neg("served_token_off_by_one", st, w, "reject:SELECT_MISMATCH", "a consistent record whose sampler.select is argmax + 1")
    st, w = make_pair([ONE, TWO, TWO, HALF], 2)
    neg("served_tie_higher_index", st, w, "reject:SELECT_MISMATCH", "tie at the max (1, 2); the record served 2 -- the registered rule is the lowest index")
    st, w = make_pair([ONE, NAN, THREE], 1)
    neg("served_nan_index_torch_rule", st, w, "reject:SELECT_MISMATCH", "the record served the NaN's index (torch.argmax propagates NaN); the registered function never selects a NaN at i >= 1")
    st, w = make_pair([NAN, ONE, TWO], 2)
    neg("served_max_ignoring_nan_at_0", st, w, "reject:SELECT_MISMATCH", "NaN at index 0: the registered function returns 0 (initial carry); the record served the finite maximum")
    st, w = make_pair([NINF] * 4, 3)
    neg("served_last_of_all_equal", st, w, "reject:SELECT_MISMATCH", "all -inf: the registered result is 0; the record served the last index")
    # authentication negatives
    lw = bytearray(base_w.logits)
    lw[4] ^= 0x01
    neg("logits_bitflip_honest_path", base_st, replace(base_w, logits=bytes(lw)), "reject:LOGITS_AUTH", "one bit of the logits row flipped, honest path")
    sw = bytearray(base_w.select)
    sw[0] ^= 0x01
    neg("select_bitflip_honest_path", base_st, replace(base_w, select=bytes(sw)), "reject:SELECT_AUTH", "one bit of the committed token flipped, honest path")
    neg("wrong_root", replace(base_st, root=bytes(32)), base_w, "reject:LOGITS_AUTH", "statement root is not the tree's")
    neg("swapped_ranks", replace(base_st, logits_rank=base_st.select_rank, select_rank=base_st.logits_rank), base_w, "reject:LOGITS_AUTH", "logits/select ranks swapped in the statement")
    neg("wrong_logits_canonical_id", replace(base_st, logits_id=_ids(4)[0]), base_w, "reject:LOGITS_AUTH", "statement names another step's logits leaf (t=4) over t=3's bytes and path")
    neg("wrong_select_canonical_id", replace(base_st, select_id=_ids(4)[1]), base_w, "reject:SELECT_AUTH", "statement names another step's sampler.select leaf")
    p = list(base_w.logits_path)
    k = next(i for i, s in enumerate(p) if s is not None)
    p[k] = None
    neg("path_sibling_turned_lift", base_st, replace(base_w, logits_path=tuple(p)), "reject:LOGITS_AUTH", "a sibling step presented as a lift: wrong shape for (leaf_count, rank), refused before comparing hashes")
    p = list(base_w.logits_path)
    p[0] = bytes(32)
    neg("path_wrong_sibling", base_st, replace(base_w, logits_path=tuple(p)), "reject:LOGITS_AUTH", "a zero sibling digest")
    # codec negatives
    neg("path_short", base_st, base_w.encode()[:-33], "reject:WITNESS_MALFORMED", "the select path is one level short")
    neg("witness_trailing_bytes", base_st, base_w.encode() + b"\x00", "reject:WITNESS_MALFORMED", "one trailing witness byte")
    neg("witness_bad_path_flag", base_st, base_w.encode()[:4 + len(base_w.logits)] + b"\x02" + base_w.encode()[4 + len(base_w.logits) + 1:], "reject:WITNESS_MALFORMED", "a path flag that is neither 0 nor 1")
    neg("logits_truncated", replace(base_st), replace(base_w, logits=base_w.logits[:-2]), "reject:LOGITS_LENGTH", "the logits leaf has 2V-2 bytes")
    neg("statement_V_plus_one", replace(base_st, V=base_st.V + 1), base_w, "reject:LOGITS_LENGTH", "statement V = 9 over an 8-word leaf")
    neg("select_8_bytes", base_st, replace(base_w, select=base_w.select + bytes(4)), "reject:SELECT_LENGTH", "the select leaf has 8 bytes")
    neg("bad_magic", b"VTSELCT0" + base_st.encode()[8:], base_w, "reject:STATEMENT_MALFORMED", "wrong magic")
    neg("statement_truncated", base_st.encode()[:-1], base_w, "reject:STATEMENT_MALFORMED", "statement one byte short")
    neg("statement_trailing_byte", base_st.encode() + b"\x00", base_w, "reject:STATEMENT_MALFORMED", "statement one byte long")
    neg("V_is_1", replace(base_st, V=1), base_w, "reject:STATEMENT_MALFORMED", "V = 1 is not a TokenSelect")
    neg("index_bits_0", replace(base_st, index_bits=0), base_w, "reject:STATEMENT_MALFORMED", "index_bits = 0")
    neg("rank_ge_leaf_count", replace(base_st, select_rank=base_st.leaf_count), base_w, "reject:STATEMENT_MALFORMED", "select rank == leaf_count")
    neg("same_rank_twice", replace(base_st, select_rank=base_st.logits_rank), base_w, "reject:STATEMENT_MALFORMED", "logits and select at one rank")
    # index unit
    st, w = make_pair([ONE] * 5 + [TWO] + [ONE] * 2, 5, index_bits=2)
    neg("index_range_18bit_unit_violated", st, w, "reject:INDEX_RANGE", "argmax 5 with a 2-bit declared index unit: the result is not below 2^index_bits")
    # the changed-Const32 regression, statement side: the literal differs from the program's; the guest applies the literal it is
    # given (the committed token no longer matches), the verifier refuses the statement BY NAME before any proof is read
    st_lit = replace(base_st, literals=CS.Literals(0, 2, 1))
    neg("literal_changed_statement", st_lit, base_w, "reject:SELECT_MISMATCH", "statement literal init_i = 2 over an honest row/token (guest: the shifted index != committed); "
        "verifier: ck_select.check_expected names `literal init_i: statement 2 != program 1`",
        expected_check=CS.check_expected(st_lit, base_st))
    st_lit = replace(base_st, literals=CS.Literals(1, 1, 1))
    neg("literal_init_index_changed", st_lit, base_w, "accept" if CS.verify_transparent(st_lit.encode(), base_w.encode()) == "accept" else "reject:SELECT_MISMATCH",
        "statement literal init_index = 1 (only matters when nothing exceeds logits[0]; here the max is at 2 so the guest accepts) -- the VERIFIER still refuses by name",
        expected_check=CS.check_expected(st_lit, base_st))
    st_lit = replace(base_st, literals=CS.Literals(0, 1, 2))
    neg("literal_step_changed", st_lit, base_w, "reject:SELECT_MISMATCH", "statement literal step = 2: index 2 is reported as 3", expected_check=CS.check_expected(st_lit, base_st))
    return out


def record_vectors(rec: Path) -> list[dict]:
    from e2e.coverage_v1 import ck_elem as CE
    from e2e.verifier_v3.locator import ProgramLocator, build_pinned_program

    inst = json.loads((rec / "protocol_instance.json").read_text())
    program, C, problems = build_pinned_program(inst["program"]["identity"], expected_digest=inst["program"]["digest"], expected_gates=inst["program"]["gates"]["value"])
    if program is None or problems:
        raise SystemExit(f"program: {problems}")
    loc = ProgramLocator(program, C)
    handle = CE.RecordHandle.open(rec / "adapter" / "selective", program, C, loc)
    uns = json.loads((rec / "obligations" / "unsupported_selected.json").read_text())
    steps = [u for u in uns["units"] if u["kind"] == CS.STEP_KIND]
    out: list[dict] = []
    seen: set[tuple[str, int]] = set()
    for u in steps:
        r = CS.resolve(u, handle)
        if (r.request_id, r.t) in seen or len(seen) >= 2:
            continue
        seen.add((r.request_id, r.t))
        b = CS.build(u, handle)
        out.append({"name": f"record_{r.request_id[:8]}_t{r.t}", "expect": "accept", "note": f"real committed row: {b.record['description']}; located as {u['kind']} vu {u['vu_index']} (regrouped step {r.regrouped_step})",
                    "statement_hex": b.statement_bytes.hex(), "witness_hex": b.witness_bytes.hex(), "statement": b.statement.to_json(), "record": str(rec), "vu_index": u["vu_index"]})
        if len(out) == 1:
            sw = bytearray(b.witness.select)
            sw[0] ^= 0x01
            wb = replace(b.witness, select=bytes(sw)).encode()
            got = CS.verify_transparent(b.statement_bytes, wb)
            assert got == "reject:SELECT_AUTH", got
            out.append({"name": f"record_{r.request_id[:8]}_t{r.t}_forged_token", "expect": got, "note": "the same row with the committed token's low bit flipped (honest path): the leaf no longer authenticates",
                        "statement_hex": b.statement_bytes.hex(), "witness_hex": wb.hex()})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=HERE)
    a = ap.parse_args()
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    sem = {
        "function": "verity_vllm/registry/b1.py:471-476 TokenSelect; :458-468 ArgmaxStep; prims.py:538-542 Bf16GtStrict, :528-535 SelectBf16/SelectI32, :511-514 I32Add",
        "equality": "strict > : equal keeps the earlier index (lowest index wins ties); +0 == -0",
        "nan": "x > best is false when either is NaN: a NaN at i >= 1 never wins; a NaN at index 0 is the initial carry and wins (result 0)",
        "neg_inf": "-inf > -inf is false: an all -inf row selects 0",
        "literals": "Const32[0] (init best_index), Const32[1] (init i), Const32[1] (step) are read from the program text and carried in the statement",
    }
    conf = conformance()
    (a.out / "conformance.json").write_text(json.dumps({"schema": SCHEMA, "family": CS.FAMILY, "generated": stamp, "semantics": sem, "cases": conf}, indent=1))
    negs = negatives()
    (a.out / "negatives.json").write_text(json.dumps({"schema": SCHEMA, "family": CS.FAMILY, "generated": stamp, "semantics": sem, "cases": negs}, indent=1))
    print(f"conformance: {len(conf)} cases; negatives: {len(negs)} cases")
    if a.record is not None:
        recs = record_vectors(a.record)
        (a.out / "record_tiny.json").write_text(json.dumps({"schema": SCHEMA, "family": CS.FAMILY, "generated": stamp, "record": str(a.record), "cases": recs}, indent=1))
        print(f"record: {len(recs)} cases from {a.record}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
