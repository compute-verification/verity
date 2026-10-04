"""stage.py ROLE K N OUT: one Pearl-C4 unit class over N instances, staged for C-Flock by `class_statement.stage_unit` with
PoUW's rows (`leaves.statement_rows`) and, for a row unit, its anchors as public input ports (`anchors.public_ports`).

    row_a  Pc4RowA_v1{K}: x (bf16, row-seg) -> A'_i's codes, scales, P_A,i (nvfp4-p, v2) and ok; public e, fa, fb
    row_b  Pc4RowB_v1{K}: x (bf16, v2) -> B~_j's codes, scales, P_B,j and ok; public e, fb, fa, g
    the row of A cut along k (`rowk`, row_k HZ_ROW_K, 2,048 by default; `rowk.PORTS4`; registered lowerings, no scratch):
    cut_a_head      Pc4RowAHead_v1{K}: x (bf16, row-seg) -> its pc4-head row (alpha, a_E) and ok; no public input
    cut_a_first     Pc4RowAFirst_v1{K,KS}: x and the head row -> segment 0's nvfp4-codes row and its pc4-carry row; public
                    e and segment 0 of fa and fb (the slice rule)
    cut_a_form<J>   Pc4RowAForm_v1{K,KS,J}: the same and segment J-1's carry -> segment J's codes and carry; public e and
                    segment J of fa and fb
    cut_a_tail      Pc4RowATail_v1{K,KS}: the head row, every segment's codes row and the last carry -> A'_i's codes,
                    scales and P_A,i (nvfp4-p, row-seg); public e
                    each row's inputs its honest chain (the head on x, then the forms in order) by the IR reference
    tile   Pc4Tile_v1{K,TM=1,TN=1}: an A row and a B row (nvfp4-p) -> C~ and U (one v2 run); no public input
    tileTMxTN   the same over TM A rows and TN B rows
    hidden[TMxTN]   Pc4TileHidden_v1{K,TM,TN}: the same rows -> the tile digest (its tile row)
    pc8tile[TMxTN]  Pearl-C's Pc8Tile_v1{K,TM,TN,DEV='sm120'}: TM A rows and TN B rows (fp8-p) -> C~ and U (one v2 run)
    cap[TMxTN]      Pc8TileCap_v1{K,TM,TN,N=TN,RHO=1000,DEV='sm120'} (`verity_pouw.circuit.cap`): the same rows, each A row's
                    ok row and an 'adv' row (sd, cert) -> C~, U and the cap's verdict (one v2 run)
    capword         Pc8CapWord_v1{K,DEV='sm120'}: a word's A' and B~ codes (one v2 row each) and an 'adv' row (cert) -> C~, its
                    flags, its later +0 promotions and good (one v2 run)
    the cap's split, cut per tile row (`cap.unit_templates`, `cap.PORTS`; at tile TMxTN, 1x1 by default, and the unit's n
    HZ_CAP_N, 16,384; the split's own files from cursor/pouw-cap-scope-e3fa 23fb50c9c):
    splitword       Pc8CapWordUnit_v1{K,DEV='sm120'}: A'_i || P_A,i (fp8-p, row-seg), B~_j || P_B,j (fp8-p, v2) and an 'adv'
                    row (cert) -> its cap-word row (C~, U) and cap-flags row (flags, count, later +0 promotions, good)
    splitstrip[TMxTN]  Pc8CapStrip_v1: TM ok rows -> its cap-rows row (admission bits, bound, zero words; v2, one block)
    splitrowdigest[TMxTN]  Pc8CapRowDigest_v1{TN}: a tile row's TN cap-word rows -> its cap-chunks row (the TurboSHAKE128
                    digests of its 32-column chunks, then zero words; v2, one block)
    splitrowtally[TMxTN]  Pc8CapRowTally_v1{K,TN}: a tile row's TN cap-flags rows and its A row's sd row (row-seg) -> its
                    cap-sums row (atoms, zeros, elements, good)
    splittile[TMxTN]   Pc8CapTile_v1{TM,TN}: TM cap-chunks rows -> the tile digest (its tile row)
    splittally[TMxTN]  Pc8CapTally_v1: TM cap-sums rows and a cap-rows row (its zero words unread) -> the cap's verdict,
                    staged as one v2 row (a stand-in: in the design it is the tally's public output, in no row)
  Each row is an hm96 slot of 2^18 bits and a unit's slots the next power of two of its rows, so a unit holds at most 256
  rows (K_MAX 27); and C-Flock's composer refuses a unit none of whose rows has a compression before its last block.  The
  per-row cut keeps every unit of a 64x64 tile at 66 rows or fewer, each with such a row (`cap`'s and `leaves`' docstrings).

A Pearl-C row's codes are finite E4M3 (the atom has no NaN operand) and its P words BF16 below 2^-15 in magnitude, seeded;
the certificates are the prover's (`cap.certificates`), and sd a seeded salt-dead count per block.  The split tally's even
instances read zero sums (so its verdict is ok) and its odd ones seeded sums.  THE 'adv' ROW IS A
TIMING STAND-IN: in the design sd is RowA's committed port and cert the tile unit's private advice, neither a row of the
tile's; here they are one committed v2 row (public input ports must fill whole 128-bit words, which they don't), so the
unit's ANDs include that row's hashing.  The cap's roles lower BitAnd_v1 through `core3_scratch` and are labelled
'core3 scratch'.

x is a seeded N(0, 1) row in BF16; a tile's rows are `pouw_rows_vectors`' seeded nvfp4-p rows. The anchors are the
verifier's own (`TileAnchors.of` the scheme's seeds). Writes circuit.txt, inst.bin, pub.bin, pin.txt and stage.json under
OUT, and for a row unit the verifier's own file public-inputs.bin (`TileAnchors.public_inputs`) and the controls' files:
    public-inputs-wrong.bin (one anchor bit flipped, the verifier's file wrong) and pub-chosen.bin (the same bit flipped in the
public file's tail, an anchor the prover chose).  stage.json holds each input and output row's SHA-256 over its instances
(``in_row_sha256``, ``out_row_sha256``), and for a piece of a cut A row its pc4-head row's as every reader reads it
(``head_row_sha256``, `leaves.words_pc4_head`); the head's staging stops unless the pc4-head row it commits is that row.

`HZ_STAGE` (``fast``, the default, or ``library``): whether `fast_stage` patches the library first, for the same bytes in
less time and memory; stage.json records which.

DEVELOPMENT BINDING: core's BitAnd_v1, E4m3ToF32_v1 and F32ToE2m1Sat_v1 have no C-Flock piece on this branch, so the row
units lower them through `core3_scratch`'s pieces; every result of a row unit staged here is labelled 'core3 scratch'. The
PoUW word primitives Keccak-f is written in have none either, so the hiding tile lowers them through `hash_scratch`'s
pieces, checked against their primitives first; its results are labelled 'hash scratch'."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

from verity.commitments.rowleaf import ROLE_W, ROLE_X, SCHEMA_HM96_SHA512_ROW_V2
from verity_flock import circuit as FC
from verity_flock import class_statement as CS
from verity_flock import ir_lower as IL
from verity_pouw.circuit import leaves as L
from verity_pouw.circuit import pc4
from verity_pouw.circuit.anchors import TileAnchors, public_ports
from verity_pouw.protocol import Context, Matmul, Shape
from verity_pouw.schemes import pearl_c4 as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ROLE, K, N, OUT = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
#: the anchor whose word 0, instance 0, bit 0 the controls flip
FLIP = {"row_a": "fa", "row_b": "g", "row_a_first": "fa", "row_a_form": "fa", "row_a_tail": "e"}
STAGE = os.environ.get("HZ_STAGE", "fast")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def part(name: str) -> bytes:
    return hashlib.sha256(f"verity/pouw/hidden-zk/v1\0{name}".encode()).digest()


def bf16(v: np.ndarray) -> np.ndarray:
    u = v.astype(np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def gb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)


def _sha(words: np.ndarray) -> str:
    """The SHA-256 of a row's words over its instances, little-endian u16."""
    return hashlib.sha256(np.ascontiguousarray(words, dtype="<u2").tobytes()).hexdigest()


def flip_at(data: bytes, ports: list[dict], port: str, tail: bool) -> int:
    """The byte offset of word 0 of ``port`` in instance 0's regions: in a --public-inputs file (after its header line), or
    in a public file's tail (the last ``instances`` x regions words)."""
    words = sum(p["words"] for p in ports)
    before = sum(p["words"] for p in ports[:[p["port"] for p in ports].index(port)])
    if tail:
        return len(data) - 2 * N * words + 2 * before
    return len(data.split(b"\n", 1)[0]) + 1 + 2 * before


def _cut_a(piece: str):
    """One class of Pearl-C4's row of A cut along k (`rowk`; row_k ``HZ_ROW_K``, `rowk.ROW_K` by default) over N rows:
    ``piece`` head, first, form<J> (1 <= J < k/row_k) or tail.  Each row's inputs are its honest chain by the IR reference
    (the head on x, then each form from the carry the one before it returned); a form's regions are its segment's."""
    from verity.evaluation import evaluate
    from verity_pouw.circuit import rowk

    ks = int(os.environ.get("HZ_ROW_K", rowk.ROW_K))
    segs = K // ks
    head, first, forms, tail = rowk.pieces4(K, ks, "a")
    if piece in ("head", "first", "tail"):
        unit_role, d, seg = f"row_a_{piece}", {"head": head, "first": first, "tail": tail}[piece], 0 if piece == "first" else None
    elif piece.startswith("form") and piece[4:].isdigit() and 1 <= int(piece[4:]) < segs:
        unit_role, seg = "row_a_form", int(piece[4:])
        d = forms[seg - 1]
    else:
        raise SystemExit(f"stage.py: a cut_a_ piece is head, first, form<J> with 1 <= J < {segs}, or tail, not {piece!r}")
    shape = pc4.Shape(N, K, 1, tile=(1, 1), row_k=ks)
    shape.check()
    ctx = Context(part("salt"), Matmul(3, Shape(shape.m, shape.k, shape.n), 9), part("root_a"), part("root_b"))
    ta = TileAnchors.of(F.PearlC4("nvfp4"), ctx, shape)
    der, seeds = shape.derivations, {"seed_a": ta.seed_a, "seed_b": ta.seed_b}
    unit = shape.ports[unit_role]
    reads = {"cs": segs, "ss": segs} if piece == "tail" else {}
    sr = L.statement_rows(unit, K, reads, {"nvfp4-codes": ks})
    rng = np.random.default_rng(20261003 + K)
    x = np.stack([bf16(rng.standard_normal(K)) for _ in range(N)])
    fa, fb = (list(der[a].fn(shape, seeds[der[a].seed], None)) for a in ("fa", "fb"))
    upto = segs if piece == "tail" else (seg or 0)
    heads, carries, codes = [], [], []
    for i in range(N):
        xi = [int(v) for v in x[i]]
        h = list(evaluate(head, xi))
        e = list(der["ea"].fn(shape, seeds[der["ea"].seed], ta.rows[i]))
        carry, cs = [], []
        for j, fn in enumerate((first, *forms)[:upto]):
            lo, hi = rowk.R * ks * j, rowk.R * ks * (j + 1)
            out = list(evaluate(fn, xi, *h[:3], *([carry] if j else []), e, fa[lo:hi], fb[lo:hi]))
            cs.append((out[:ks], out[ks:ks + ks // 16]))
            carry = out[ks + ks // 16:]
        heads.append(h)
        carries.append(carry)
        codes.append(cs)
    names = {"bf16": "x", "pc4-head": "head", "pc4-carry": "carry", "nvfp4-codes": "codes"}
    u16 = lambda ws: np.array(ws, dtype=np.uint16)  # noqa: E731
    schemas, at, in_rows, rows = sr["row_schemas"], 0, [], {}
    for row, _role, ps in unit["rows"]:
        if not all(isinstance(p, str) for p in ps):
            continue
        n = reads.get(ps[0], 1)
        for r in range(n):
            name = f"{names[row]}{r}" if n > 1 else names[row]
            in_rows.append((name, *schemas[at], [f"{p}[{r}]" for p in ps] if n > 1 else list(ps)))
            at += 1
            rows[name] = (x if row == "bf16" else u16([L.words_pc4_head(h[0], h[1]) for h in heads]) if row == "pc4-head"
                          else u16([L.words_pc4_carry(c) for c in carries]) if row == "pc4-carry"
                          else u16([L.words_nvfp4_codes(*codes[i][r]) for i in range(N)]))
    ports = public_ports(shape, unit_role)
    regs = [ta.regions(unit_role, i, seg) for i in range(N)]
    for p in ports:
        rows[p] = np.array([g[p] for g in regs], dtype=np.uint16)
    rec = {"binding": "registered", "row_k": ks, "piece": piece, "unit_role": unit_role, "segment": seg,
           "ok": [int(h[-1]) for h in heads], "chain": "honest, by the IR reference",
           "head_row_sha256": _sha(u16([L.words_pc4_head(h[0], h[1]) for h in heads])),
           "out_row_names": [row for row, _role, ps in unit["rows"] if not all(isinstance(p, str) for p in ps)]}
    return d, unit_role, seg, ta, ports, in_rows, schemas[at:], sr, rows, rec


def main() -> None:
    t0 = time.time()
    if STAGE not in ("fast", "library"):
        raise SystemExit(f"stage.py: HZ_STAGE is fast or library, not {STAGE!r}")
    if STAGE == "fast":
        sys.path.insert(0, str(HERE))
        import fast_stage
        fast_stage.install(OUT)
    salt_key = hashlib.sha256(b"verity/pouw/hidden-zk/salts\0" + str(K).encode()).digest()
    rec: dict = {"role": ROLE, "k": K, "n": N, "staging": STAGE}
    unit_role, seg = ROLE, None
    if ROLE in ("row_a", "row_b"):
        sys.path.insert(0, str(HERE))
        import core3_scratch as K3
        IL.PIECES.update(K3.PIECES)
        rec["binding"] = "core3 scratch"
        shape = pc4.Shape(N, K, 1, tile=(1, 1)) if ROLE == "row_a" else pc4.Shape(1, K, N, tile=(1, 1))
        ctx = Context(part("salt"), Matmul(3, Shape(shape.m, shape.k, shape.n), 9), part("root_a"), part("root_b"))
        ta = TileAnchors.of(F.PearlC4("nvfp4"), ctx, shape)
        d = pc4.templates(K)[ROLE]
        ports = public_ports(shape, ROLE)
        sr = L.statement_rows(pc4.PORTS[ROLE], K)
        (xs, xb), outs_ = sr["row_schemas"][0], sr["row_schemas"][1:]
        in_rows = [("x", xs, xb, ["x"])]
        rng = np.random.default_rng(20261003 + K)
        rows = {"x": np.stack([bf16(rng.standard_normal(K)) for _ in range(N)])}
        regs = [ta.regions(ROLE, i) for i in range(N)]
        for p in ports:
            rows[p] = np.array([r[p] for r in regs], dtype=np.uint16)
    elif ROLE.startswith("cut_a_"):
        d, unit_role, seg, ta, ports, in_rows, outs_, sr, rows, rec_cut = _cut_a(ROLE.removeprefix("cut_a_"))
        rec.update(rec_cut)
    elif ROLE.startswith(("tile", "hidden")):
        V = _load("pouw_rows_vectors", ROOT / "protocols" / "pouw" / "tests" / "pouw_rows_vectors.py")
        hidden = ROLE.startswith("hidden")
        if hidden:
            sys.path.insert(0, str(HERE))
            import hash_scratch as H
            bad = H.check()
            assert not any(bad.values()), f"hash_scratch's pieces disagree with their primitives: {bad}"
            IL.PIECES.update(H.PIECES)
            rec["binding"] = "hash scratch"
        tm, tn = map(int, ROLE.removeprefix("hidden" if hidden else "tile").split("x")) if "x" in ROLE else (1, 1)
        rec["tile"] = [tm, tn]
        d = pc4.templates(K, (tm, tn), hidden=hidden)["tile"]
        ports = {}
        bits = 16 * L.ROW_WORDS["nvfp4-p"](K)
        in_rows = [(f"{s}{r}", L.ROW_SCHEMA["nvfp4-p", role], bits, [f"{p}[{r}]" for p in ps])
                   for s, role, ps, t in (("a", ROLE_X, ("ac", "as", "pa"), tm), ("b", ROLE_W, ("bc", "bs", "pb"), tn))
                   for r in range(t)]
        if hidden:
            sr = L.statement_rows(pc4.PORTS["tile"], K, {p: t for ps, t in ((("ac", "as", "pa"), tm), (("bc", "bs", "pb"), tn))
                                                          for p in ps})
            outs_ = sr["row_schemas"][tm + tn:]
        else:
            outs_ = [[SCHEMA_HM96_SHA512_ROW_V2, None]]
            sr = {"out_row_ports": [2]}
        rows = {name: np.array([V.words("nvfp4-p", V.inputs(f"hidden-zk-{K}-{name}-{i}", "nvfp4-p", K)) for i in range(N)],
                               dtype=np.uint16) for name, *_ in in_rows}
    elif ROLE.startswith("split"):
        from verity_pouw.circuit import cap, pc8
        sys.path.insert(0, str(HERE))
        import core3_scratch as K3
        IL.PIECES.update(K3.PIECES)
        rec["binding"] = "core3 scratch"
        kinds = {"word": "word", "strip": "strip", "rowdigest": "row_digest", "rowtally": "row_tally", "tally": "tally",
                 "tile": "tile"}
        kind, spec = next((kinds[kd], ROLE.removeprefix(f"split{kd}")) for kd in kinds if ROLE.startswith(f"split{kd}"))
        tm, tn = map(int, spec.split("x")) if spec else (1, 1)
        n_unit = int(os.environ.get("HZ_CAP_N", "16384"))
        rec.update(tile=[tm, tn], unit_n=n_unit)
        tp = cap.unit_templates(K, "sm120", (tm, tn), n_unit)
        d = tp[kind]
        sr = L.statement_rows(cap.PORTS[kind], K, cap.reads((tm, tn))[kind])
        schemas = sr["row_schemas"]
        rng = np.random.default_rng(20261004 + K)
        finite = np.array([c for c in range(256) if c not in (0x7F, 0xFF)])
        u16 = lambda ws: np.array(ws, dtype=np.uint16)  # noqa: E731
        f32_words = lambda vs: [h for v in vs for h in (v & 0xFFFF, v >> 16)]  # noqa: E731
        v2 = SCHEMA_HM96_SHA512_ROW_V2
        ports = {}
        if kind in ("row_digest", "tile"):
            sys.path.insert(0, str(HERE))
            import hash_scratch as H
            bad = H.check()
            assert not any(bad.values()), f"hash_scratch's pieces disagree with their primitives: {bad}"
            IL.PIECES.update(H.PIECES)
            rec["binding"] = "core3 scratch + hash scratch"
        if kind == "word":
            ac = [list(map(int, rng.choice(finite, K))) for _ in range(N)]
            bc = [list(map(int, rng.choice(finite, K))) for _ in range(N)]
            pa, pb = ([list(map(int, rng.integers(0, 0x3FFF, pc8.PEEL))) for _ in range(N)] for _ in range(2))
            cert = [cap.certificates(ac[i], bc[i], "sm120") for i in range(N)]
            in_rows = [("a", *schemas[0], ["a", "pa"]), ("b", *schemas[1], ["b", "pb"]),
                       ("adv", v2, 32 * len(cert[0]), ["cert"])]
            rows = {"a": u16([L.words_fp8_p(ac[i], pa[i]) for i in range(N)]),
                    "b": u16([L.words_fp8_p(bc[i], pb[i]) for i in range(N)]), "adv": u16([f32_words(c) for c in cert])}
            outs_ = schemas[2:]
            rec["advice_stand_in"] = "cert as one committed v2 row 'adv' (C-Flock's composer has no private input; timing only)"
        elif kind == "strip":
            in_rows = [(f"ok{r}", *schemas[r], [f"oka[{r}]"]) for r in range(tm)]
            rows = {f"ok{r}": u16([L.words_ok(int(rng.random() < 0.9)) for _ in range(N)]) for r in range(tm)}
            outs_ = schemas[tm:]
        elif kind == "row_digest":
            in_rows = [(f"w{j}", *schemas[j], [f"c[{j}]", f"u[{j}]"]) for j in range(tn)]
            word = lambda: int(np.float32(rng.standard_normal()).view(np.uint32))  # noqa: E731
            rows = {f"w{j}": u16([L.words_cap_word(word(), word()) for _ in range(N)]) for j in range(tn)}
            outs_ = schemas[tn:]
        elif kind == "row_tally":
            in_rows = [(f"f{j}", *schemas[j], [f"flags[{j}]", f"atoms[{j}]", f"zeros[{j}]", f"good[{j}]"]) for j in range(tn)]
            in_rows.append(("sd", *schemas[tn], ["sd"]))
            rows = {}
            for j in range(tn):
                fl = [[int(rng.random() < 1 / 64) for _ in range(K // 32)] for _ in range(N)]
                rows[f"f{j}"] = u16([L.words_cap_flags(f, sum(f), int(rng.integers(0, 3)), 1) for f in fl])
            rows["sd"] = u16([L.words_sd([int(rng.choice([0] * 30 + [1, 32])) for _ in range(K // 32)]) for _ in range(N)])
            outs_ = schemas[tn + 1:]
        elif kind == "tile":
            chunks = -(-tn // 32)
            in_rows = [(f"d{r}", *schemas[r], [f"d[{r}]"]) for r in range(tm)]
            rows = {f"d{r}": u16([L.words_cap_chunks([rng.bytes(32) for _ in range(chunks)]) for _ in range(N)])
                    for r in range(tm)}
            outs_ = schemas[tm:]
        else:
            bound = cap._cap_constants(K, tm, tn, n_unit, 1000, "sm120")[-1][tm]
            in_rows = [(f"s{r}", *schemas[r], [f"atoms[{r}]", f"zeros[{r}]", f"elems[{r}]", f"good[{r}]"]) for r in range(tm)]
            in_rows.append(("cr", *schemas[tm], ["mask", "bound", "tail"]))
            sums = lambda i: ((0, 0, 0) if i % 2 == 0 else  # noqa: E731
                              (int(rng.integers(0, tn * K // 2048 + 1)), int(rng.integers(0, 3)), int(rng.integers(0, K // 32 + 1))))
            rows = {f"s{r}": u16([L.words_cap_sums(*sums(i), 1) for i in range(N)]) for r in range(tm)}
            rows["cr"] = u16([L.words_cap_rows([1] * tm, bound) for _ in range(N)])
            outs_ = [[v2, None]]
            rec["public_stand_in"] = ("the tally's verdict, a public output in the design (`cap.PORTS`), staged as one v2 "
                                      "row (C-Flock's composer commits every output)")
        sr = {"out_row_ports": sr.get("out_row_ports") or [1] * len(outs_)}
    elif ROLE.startswith(("pc8tile", "cap")):
        from verity_pouw.circuit import cap, pc8
        sys.path.insert(0, str(HERE))
        import core3_scratch as K3
        IL.PIECES.update(K3.PIECES)
        rec["binding"] = "core3 scratch"
        rng = np.random.default_rng(20261003 + K)
        finite = np.array([c for c in range(256) if c not in (0x7F, 0xFF)])
        code_rows = lambda: [list(map(int, rng.choice(finite, K))) for _ in range(N)]  # noqa: E731
        p_rows = lambda: [list(map(int, rng.integers(0, 0x3FFF, pc8.PEEL))) for _ in range(N)]  # noqa: E731
        u16 = lambda ws: np.array(ws, dtype=np.uint16)  # noqa: E731
        f32_words = lambda vs: [h for v in vs for h in (v & 0xFFFF, v >> 16)]  # noqa: E731
        v2 = SCHEMA_HM96_SHA512_ROW_V2
        rec["advice_stand_in"] = "sd and cert as one committed v2 row 'adv' (timing only; module docstring)"
        if ROLE == "capword":
            d = cap.templates(K, "sm120")["word"]
            a, b = code_rows(), code_rows()
            in_rows = [("a", v2, 8 * K, ["a"]), ("b", v2, 8 * K, ["b"])]
            rows = {"a": u16([[x | y << 8 for x, y in zip(r[::2], r[1::2])] for r in a]),
                    "b": u16([[x | y << 8 for x, y in zip(r[::2], r[1::2])] for r in b])}
            cert = [cap.certificates(a[i], b[i], "sm120") for i in range(N)]
            in_rows.append(("adv", v2, 32 * len(cert[0]), ["cert"]))
            rows["adv"] = u16([f32_words(c) for c in cert])
            ports = {}
            outs_, sr = [[v2, None]], {"out_row_ports": [4]}
        else:
            capped = ROLE.startswith("cap")
            spec = ROLE.removeprefix("cap" if capped else "pc8tile")
            tm, tn = map(int, spec.split("x")) if spec else (1, 1)
            rec["tile"] = [tm, tn]
            d = (cap.templates(K, "sm120", (tm, tn)) if capped else pc8.templates(K, (tm, tn), "sm120"))["tile"]
            bits = 16 * L.ROW_WORDS["fp8-p"](K)
            ac, bc = [code_rows() for _ in range(tm)], [code_rows() for _ in range(tn)]
            pa, pb = [p_rows() for _ in range(tm)], [p_rows() for _ in range(tn)]
            in_rows = [(f"a{r}", L.ROW_SCHEMA["fp8-p", ROLE_X], bits, [f"ac[{r}]", f"pa[{r}]"]) for r in range(tm)]
            in_rows += [(f"b{r}", L.ROW_SCHEMA["fp8-p", ROLE_W], bits, [f"bc[{r}]", f"pb[{r}]"]) for r in range(tn)]
            rows = {f"a{r}": u16([L.words_fp8_p(ac[r][i], pa[r][i]) for i in range(N)]) for r in range(tm)}
            rows |= {f"b{r}": u16([L.words_fp8_p(bc[r][i], pb[r][i]) for i in range(N)]) for r in range(tn)}
            ports = {}
            if capped:
                in_rows += [(f"ok{r}", L.ROW_SCHEMA["ok", ROLE_X], 16 * L.ROW_WORDS["ok"](K), [f"oka[{r}]"]) for r in range(tm)]
                rows |= {f"ok{r}": u16([L.words_ok(1) for _ in range(N)]) for r in range(tm)}
                sd = [[int(x) for x in rng.choice([0] * 30 + [1, 32], tm * (K // 32))] for _ in range(N)]
                cert = [[w for r in range(tm) for j in range(tn) for w in cap.certificates(ac[r][i], bc[j][i], "sm120")]
                        for i in range(N)]
                in_rows.append(("adv", v2, 32 * (len(sd[0]) + len(cert[0])), ["sd", "cert"]))
                rows["adv"] = u16([f32_words(sd[i] + cert[i]) for i in range(N)])
            outs_, sr = [[v2, None]], {"out_row_ports": [3 if capped else 2]}
    else:
        raise SystemExit(f"stage.py: ROLE is row_a, row_b, cut_a_(head|first|form<J>|tail), tile[TMxTN], hidden[TMxTN], "
                         f"pc8tile[TMxTN], cap[TMxTN], capword, splitword or split(strip|rowdigest|rowtally|tile|tally)[TMxTN], "
                         f"not {ROLE!r}")
    runs = sr.get("out_row_ports") or [1] * len(outs_)
    out_rows, at = [], 0
    for (s, _b), r in zip(outs_, runs):
        out_rows.append((s, tuple(range(at, at + r))))
        at += r
    rec.update(definition=d.id, in_rows=[[r, s, b] for r, s, b, _ in in_rows], out_rows=[[s, list(ps)] for s, ps in out_rows],
               public_inputs=ports)
    got = CS.stage_unit(d, f"unit/pouw-pc4-{ROLE}", in_rows, out_rows, rows, OUT, salt_key, public_inputs=ports or None)
    rec["in_row_sha256"] = {r: _sha(rows[r]) for r, *_ in in_rows}
    rec["out_row_sha256"] = [_sha(np.concatenate([got["outs"][f"o{p}"] for p in ps], axis=1)) for _s, ps in out_rows]
    named = dict(zip(rec.get("out_row_names", ()), rec["out_row_sha256"]))
    if "pc4-head" in named and named["pc4-head"] != rec["head_row_sha256"]:
        raise SystemExit("stage.py: the head's committed pc4-head row is not the row the forms and tail read "
                         "(leaves.words_pc4_head)")
    t_stage = time.time() - t0
    text = fast_stage.composed() if STAGE == "fast" else (OUT / "circuit.txt").read_text()
    meta = got["meta"]
    pin = FC.digest(text)
    (OUT / "pin.txt").write_text(pin)
    if ports and unit_role in FLIP:
        own = ta.public_inputs(unit_role, range(N), seg)
        ta.check_public_inputs(unit_role, range(N), own, seg)
        (OUT / "public-inputs.bin").write_bytes(own)
        head = json.loads(own.split(b"\n", 1)[0])
        pub = (OUT / "pub.bin").read_bytes()
        assert pub.endswith(own.split(b"\n", 1)[1]), "the public file's tail is not the verifier's own words"
        wrong = bytearray(own)
        wrong[flip_at(own, head["ports"], FLIP[unit_role], tail=False)] ^= 1
        (OUT / "public-inputs-wrong.bin").write_bytes(bytes(wrong))
        chosen = bytearray(pub)
        chosen[flip_at(pub, head["ports"], FLIP[unit_role], tail=True)] ^= 1
        (OUT / "pub-chosen.bin").write_bytes(bytes(chosen))
        rec.update(own_bytes=len(own), own_sha256=hashlib.sha256(own).hexdigest(),
                   flipped={"port": FLIP[unit_role], "word": 0, "instance": 0, "bit": 0})
    if ROLE.startswith("splittally"):
        rec["verdicts"] = [int(w[0]) for w in got["outs"]["o0"]]
    rec.update(ands=got["ands"], k_log=meta["k_log"], vus_per_block=meta.get("vus_per_block"),
               row_schemas=meta["row_schemas"], out_row_ports=meta.get("out_row_ports"),
               circuit_bytes=fast_stage.size(text) if STAGE == "fast" else len(text),
               circuit_sha512=pin, class_sha512=FC.class_digest(text),
               inst_bytes=(OUT / "inst.bin").stat().st_size, pub_bytes=(OUT / "pub.bin").stat().st_size,
               seconds={"stage": round(t_stage, 1), "total": round(time.time() - t0, 1)}, maxrss_gb=gb())
    if STAGE == "fast":
        del text
        fast_stage.release()
    (OUT / "stage.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
