"""The 2:4 metadata rule of the sparse FP4 row on sm_120 (``probe.py --instruction sp_nvf4_meta | sp_mxf4_meta``): for each of
the 16 values a chunk's metadata nibble can hold, which logical k each of the chunk's four stored codes is multiplied at, under
which scale, and with what weight, measured on the device; then whether that map, fed to the pinned dense step, reproduces
every word on random operands.  ``mma.sp::ordered_metadata`` promises the hardware ordered pairs p0 < p1 (``sparse.META_VALID``,
six nibbles); the other ten are what a kernel can still pass, and the capture records what the device does with them.

Part A, the map (one-hot, exact): row config (stored slot s, nibble m) puts one nonzero code at stored column s and m in chunk
s // 4's nibble; B's column n is nonzero only at logical k = 8 c' + n for one chunk c' per tile; C = +0.  Each word is then
weight x a x b x the scale of one block, and the scales are distinct powers of two per block (A's in variant ``a``, B's in
variant ``b``), spaced so that (weight, block) decode uniquely for weights 1..15.  All 64 x 16 configs meet all 16 chunks c',
so the word at (config, k) says whether slot s lands at logical k under m.  Variant ``b`` also permutes the rows, draws the
other chunks' nibbles from all 16, and changes the codes' values and signs, so its agreement with ``a`` shows that the map
depends on the nibble alone.

Part B, the arithmetic: random codes, B nonzero everywhere, random scales and accumulators, nibbles drawn per row and chunk
from every nibble Part A found to be a gather (each slot at most one logical k, weight 1), against the pinned dense model
(``sparse.PINNED``) on the stored A and, per row, the B codes its nibbles gather (``gather_dense`` extended to all 16 nibbles).
Part B2 is the same with one metadata row per tile, against the dense ``mma.sync`` on the gathered B (hardware against
hardware).

Part C (Derived from A): the 256 zero patterns of an 8-chunk, and which of them the hardware can write (a pattern is
expressible under m when every nonzero offset is one some slot lands at), under the six ordered nibbles and under all 16,
beside D-24's element-wise rule (at most two nonzero codes per aligned 4-group).
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

import probe as P
import sparse as S
from verity_config.settings import SHARED

INSTRUCTIONS = {"sp_nvf4_meta": "sp_nvf4", "sp_mxf4_meta": "sp_mxf4"}
K_STORED, K_LOGICAL, CHUNKS = S.K_STORED, S.K_LOGICAL, S.CHUNKS
UNIT = {"ue4m3": 0x38, "ue8m0": 127}
#: Part A's distinct block scales: exponents spaced >= 4 apart, so weight x 2^e decodes uniquely for weights below 16
BLOCK_EXPS = {"ue4m3": (-6, -2, 2, 6), "ue8m0": (-8, 8)}
#: Part A's codes (value): variant a 1.0 x 1.0, variant b -3.0 x 6.0
CODES = {"a": (0x2, 0x2), "b": (0xD, 0x7)}
E2M1_VALUE = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])
BACKGROUND = 0b0100


def scale_code(kind: str, e: int) -> int:
    return (e + 7) << 3 if kind == "ue4m3" else e + 127


def pairs_from_bits(m: int) -> list[int]:
    """The hypothesis that the hardware reads the nibble as two pair indices whatever their order: slots 0, 1 at pair m & 3,
    slots 2, 3 at pair m >> 2 (for the six ordered nibbles, PTX's layout)."""
    p0, p1 = m & 3, m >> 2
    return [2 * p0, 2 * p0 + 1, 2 * p1, 2 * p1 + 1]


def gather_offsets(rule: dict) -> np.ndarray:
    """rule {nibble: [offset or None per slot]} -> (16, 4) offsets, -1 where the slot lands nowhere."""
    off = np.full((16, 4), -1, np.int64)
    for m, offs in rule.items():
        off[int(m)] = [-1 if o is None else o for o in offs]
    return off


def gathered_b(B, META, off) -> np.ndarray:
    """B (n,8,128) codes, META (n,16,16) nibbles, off (16,4) -> (n,16,8,64): each row's B codes at its stored slots (code 0
    where the slot lands nowhere)."""
    n = B.shape[0]
    o = off[np.asarray(META, np.int64)].reshape(n, 16, K_STORED)                     # the slots' offsets in their chunks
    cols = np.where(o < 0, 0, 8 * (np.arange(K_STORED) // 4) + o)
    idx = np.broadcast_to(cols[:, :, None, :], (n, 16, 8, K_STORED))
    g = np.take_along_axis(np.broadcast_to(np.asarray(B)[:, None, :, :], (n, 16, 8, K_LOGICAL)), idx, axis=-1)
    return np.where(np.broadcast_to((o < 0)[:, :, None, :], g.shape), 0, g).astype(np.uint8)


def gather_words(model, As, Bg, SFA, SFB, C) -> tuple[np.ndarray, np.ndarray]:
    """The dense pinned step per element on the stored A row and its gathered B: (D (n,16,8), domain-error mask)."""
    n = As.shape[0]
    out = np.zeros((n, 16, 8), np.uint32)
    err = np.zeros((n, 16, 8), np.bool_)
    Al, Bl, SAl, SBl, Cl = As.tolist(), Bg.tolist(), SFA.tolist(), SFB.tolist(), C.tolist()
    for t in range(n):
        for i in range(16):
            for j in range(8):
                try:
                    out[t, i, j] = model.step_scaled(Cl[t][i][j], Al[t][i], Bl[t][i][j], SAl[t][i], SBl[t][j])
                except Exception:  # noqa: BLE001 - outside the model's domain
                    err[t, i, j] = True
    return out, err


_CTX: dict = {}


def _init(instr: str) -> None:
    _CTX["model"] = S.PINNED[instr]


def _eval_chunk(args):
    return gather_words(_CTX["model"], *args)


def evaluate(pool, instr, As, Bg, SFA, SFB, C, procs: int):
    if pool is None:
        return gather_words(S.PINNED[instr], As, Bg, SFA, SFB, C)
    idx = np.array_split(np.arange(As.shape[0]), max(1, min(As.shape[0] // 4 + 1, procs * 8)))
    parts = pool.map(_eval_chunk, [(As[i], Bg[i], SFA[i], SFB[i], C[i]) for i in idx])
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


class FakeMeta:
    """CPU stand-in (``--fake-silicon pairs_from_bits``): the pinned dense step on the B each nibble gathers under
    ``pairs_from_bits``; refused with ``RESEARCH_RUN_ID`` set."""

    tcgen05 = False

    def __init__(self, instr: str, model_name: str):
        if model_name != "pairs_from_bits":
            raise SystemExit(f"--fake-silicon: {instr} takes only 'pairs_from_bits'")
        self.model_name = model_name
        self.off = gather_offsets({m: pairs_from_bits(m) for m in range(16)})

    def identity(self) -> dict:
        return {"device": f"FAKE SILICON ({self.model_name})", "compute_capability": "fake", "driver": None, "gpu_uuid": "GPU-fake",
                "backend": "fake", "cuda_visible_devices": SHARED.CUDA_VISIBLE_DEVICES}

    def clocks(self, uuid):
        return None

    def tiles(self, instr: str, As, B, META, SFA, SFB, C) -> np.ndarray:
        D, err = gather_words(S.PINNED[instr], np.asarray(As), gathered_b(B, META, self.off), np.asarray(SFA), np.asarray(SFB),
                              np.asarray(C, np.uint32))
        D[err] = 0x7FFFFFFF
        return D


# --- Part A: the map -------------------------------------------------------------------------------------------------------------


def part_a_operands(instr: str, variant: str, rng):
    """1024 tiles: B chunk c' = t // 64, row i of tile t holds config order[64 c' ... ] -> (s, m).  Returns the operands and
    (n,16) arrays of each row's slot and nibble."""
    kind, nb = S.SCALE_KIND[instr], S.NBLK[instr]
    acode, bcode = CODES[variant]
    n = CHUNKS * 64
    slot = np.zeros((n, 16), np.int64)
    nib = np.zeros((n, 16), np.int64)
    cfg = np.arange(64 * 16)
    for cb in range(CHUNKS):
        order = cfg if variant == "a" else rng.permutation(cfg)
        slot[cb * 64:(cb + 1) * 64] = (order // 16).reshape(64, 16)
        nib[cb * 64:(cb + 1) * 64] = (order % 16).reshape(64, 16)
    As = np.zeros((n, 16, K_STORED), np.uint8)
    np.put_along_axis(As, slot[:, :, None], acode, axis=-1)
    META = (np.full((n, 16, CHUNKS), BACKGROUND, np.uint8) if variant == "a"
            else rng.integers(0, 16, (n, 16, CHUNKS)).astype(np.uint8))
    np.put_along_axis(META, (slot // 4)[:, :, None], nib[:, :, None].astype(np.uint8), axis=-1)
    B = np.zeros((n, 8, K_LOGICAL), np.uint8)
    cb = np.arange(n) // 64
    for j in range(8):
        B[np.arange(n), j, 8 * cb + j] = bcode
    distinct = np.array([scale_code(kind, e) for e in BLOCK_EXPS[kind]], np.uint8)
    SFA = np.full((n, 16, nb), UNIT[kind], np.uint8)
    SFB = np.full((n, 8, nb), UNIT[kind], np.uint8)
    if variant == "a":
        SFA[:] = distinct
    else:
        SFB[:] = distinct
    return As, B, META, SFA, SFB, np.zeros((n, 16, 8), np.uint32), slot, nib


def decode(D, instr: str, variant: str):
    """Words -> (weight, block): weight 0 and block -1 for +0; weight -1 where the word is not weight x a x b x 2^e_block."""
    kind = S.SCALE_KIND[instr]
    acode, bcode = CODES[variant]
    q = np.asarray(D, np.uint32).view(np.float32).astype(np.float64) / (E2M1_VALUE[acode] * E2M1_VALUE[bcode])
    weight = np.full(q.shape, -1, np.int64)
    block = np.full(q.shape, -1, np.int64)
    zero = (np.asarray(D, np.uint32) & 0x7FFFFFFF) == 0
    weight[zero] = 0
    for b, e in enumerate(BLOCK_EXPS[kind]):
        r = q / 2.0 ** e
        hit = ~zero & (weight < 0) & np.isfinite(r) & (r == np.round(r)) & (r >= 1) & (r <= 15)
        weight[hit] = np.round(r[hit]).astype(np.int64)
        block[hit] = b
    return weight, block


def landing_table(D, slot, nib, instr: str, variant: str):
    """Part A's words -> W (64 slots, 16 nibbles, 128 k) weights and BLK the decoded blocks, plus the undecodable words."""
    weight, block = decode(D, instr, variant)
    n = D.shape[0]
    cb = np.arange(n) // 64
    W = np.full((K_STORED, 16, K_LOGICAL), -2, np.int64)
    BLK = np.full((K_STORED, 16, K_LOGICAL), -1, np.int64)
    k = 8 * cb[:, None, None] + np.arange(8)[None, None, :]                              # (n,1,8)
    s = np.broadcast_to(slot[:, :, None], (n, 16, 8))
    m = np.broadcast_to(nib[:, :, None], (n, 16, 8))
    k = np.broadcast_to(k, (n, 16, 8))
    W[s, m, k] = weight
    BLK[s, m, k] = block
    bad = [{"slot": int(s[t, i, j]), "nibble": int(m[t, i, j]), "k": int(k[t, i, j]), "word": f"{int(D[t, i, j]):08x}"}
           for t, i, j in zip(*np.nonzero(weight < 0))]
    return W, BLK, bad


def nibble_rule(W, BLK_a, BLK_b, nb: int):
    """Per nibble: where each slot of chunk c lands, relative to 8c, if that is the same in every chunk; the blocks against
    both ways of assigning one (the stored slot's, the logical k's)."""
    rule, per_nibble = {}, {}
    for m in range(16):
        maps, same = [], True
        for c in range(CHUNKS):
            chunk = []
            for j in range(4):
                s = 4 * c + j
                hits = [(int(k) - 8 * c, int(W[s, m, k])) for k in np.nonzero(W[s, m] > 0)[0]]
                chunk.append(hits)
            maps.append(chunk)
            same &= chunk == maps[0]
        is_gather = same and all(len(h) <= 1 and all(w == 1 and 0 <= o < 8 for o, w in h) for h in maps[0])
        per_nibble[m] = {"ordered": bool(m in S.META_VALID.tolist()), "same_in_every_chunk": bool(same),
                         "slots": [[{"offset": o, "weight": w} for o, w in h] for h in maps[0]], "gather": bool(is_gather)}
        if is_gather:
            rule[m] = [h[0][0] if h else None for h in maps[0]]
    landed = W > 0
    s_idx, _, k_idx = np.nonzero(landed)
    blocks = {"a_side_is_stored_slot_block": bool(np.array_equal(BLK_a[landed], s_idx // (K_STORED // nb))),
              "a_side_is_logical_k_block": bool(np.array_equal(BLK_a[landed], k_idx // (K_LOGICAL // nb))),
              "b_side_is_logical_k_block": bool(np.array_equal(BLK_b[landed], k_idx // (K_LOGICAL // nb)))}
    return rule, per_nibble, blocks


def predicate_tables(rule: dict) -> dict:
    """Part C: the 256 zero patterns of an 8-chunk against the hardware's images and D-24."""
    def image(m):
        return sum(1 << o for o in rule[m] if o is not None)

    ordered = [m for m in S.META_VALID.tolist() if m in rule]
    all16 = sorted(rule)
    img_o = [image(m) for m in ordered]
    img_a = [image(m) for m in all16]

    def d24(p):
        return bin(p & 0xF).count("1") <= 2 and bin(p >> 4).count("1") <= 2

    def pairs(p):
        return sum(1 for q in range(4) if p >> (2 * q) & 3)

    rows = []
    for p in range(256):
        rows.append({"pattern": p, "hw_ordered": any(p & ~i == 0 for i in img_o), "hw_all16": any(p & ~i == 0 for i in img_a),
                     "d24": d24(p), "pairs_touched": pairs(p)})

    def offs(p):
        return [o for o in range(8) if p >> o & 1]

    count = {k: sum(r[k] for r in rows) for k in ("hw_ordered", "hw_all16", "d24")}
    count["hw_ordered_and_d24"] = sum(r["hw_ordered"] and r["d24"] for r in rows)
    hw_only = [offs(r["pattern"]) for r in rows if r["hw_ordered"] and not r["d24"]]
    d24_only = [offs(r["pattern"]) for r in rows if r["d24"] and not r["hw_ordered"]]
    return {"patterns": 256, "counts": count,
            "hw_ordered_is_at_most_two_pairs": all(r["hw_ordered"] == (r["pairs_touched"] <= 2) for r in rows),
            "hw_all16_equals_hw_ordered": all(r["hw_all16"] == r["hw_ordered"] for r in rows),
            "hw_not_d24": {"count": len(hw_only), "examples": hw_only[:6]},
            "d24_not_hw": {"count": len(d24_only), "examples": d24_only[:6]},
            "note": "pattern = the set of nonzero logical offsets 0..7 of one chunk; hw_*: some nibble (ordered six / every nibble "
                    "Part A found to be a gather) lands a slot at every one of them; d24: at most two nonzero codes per aligned "
                    "4-group"}


# --- Part B ----------------------------------------------------------------------------------------------------------------------


def part_b_operands(instr: str, rng, n: int, nibbles: np.ndarray, shared: bool):
    kind, nb = S.SCALE_KIND[instr], S.NBLK[instr]
    As = rng.integers(0, 16, (n, 16, K_STORED)).astype(np.uint8)
    B = rng.integers(1, 16, (n, 8, K_LOGICAL)).astype(np.uint8)
    shape = (n, 1, CHUNKS) if shared else (n, 16, CHUNKS)
    META = np.broadcast_to(nibbles[rng.integers(0, len(nibbles), shape)], (n, 16, CHUNKS)).astype(np.uint8)
    if kind == "ue4m3":
        SFA, SFB = (P.UE4M3_VALID[rng.integers(0, len(P.UE4M3_VALID), s)] for s in ((n, 16, nb), (n, 8, nb)))
    else:
        SFA, SFB = (rng.integers(107, 148, s).astype(np.uint8) for s in ((n, 16, nb), (n, 8, nb)))
    C = P.f32_bits((rng.standard_normal((n, 16, 8)) * np.exp2(rng.integers(-30, 40, (n, 16, 8)))).astype(np.float32))
    C[: n // 2] = 0
    return As, B, np.ascontiguousarray(META), SFA, SFB, C


# --- driver ----------------------------------------------------------------------------------------------------------------------


def main(args) -> int:
    out = Path(os.path.expandvars(args.out))
    out.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    row = INSTRUCTIONS[args.instruction]
    arch = args.arch or "sm_120a"
    if args.chain != 1:
        raise SystemExit("--chain is not defined for the sparse rows")
    if args.fake_silicon and SHARED.RESEARCH_RUN_ID:
        raise SystemExit("--fake-silicon is a dry run and never publishes: unset RESEARCH_RUN_ID")
    if args.fake_silicon and not P.shutil.which(P.nvcc_bin()):
        so, nvcc = None, "not built (fake silicon)"
    else:
        so, nvcc = P.build(arch, out, "mma_fp4.cu")
    ptx = S.PTX[row]
    results: dict = {"schema": "pouw-rtxpro/tc-probe-fp4-meta-rule/v1", "instruction": ptx, "instruction_id": S.INSTRUCTION_ID[row],
                     "fake_silicon": args.fake_silicon}
    failures: list[str] = []

    def finish() -> int:
        results["seconds"] = round(time.time() - t_all, 1)
        results["failures"] = failures
        (out / "probe_results.json").write_text(json.dumps(results, indent=1))
        return 1 if failures else 0

    if args.sass_only:
        dev = dense = None
        identity = {"device": None, "gpu_uuid": None, "driver": None, "compute_capability": None, "backend": "none (--sass-only)"}
    else:
        dev = FakeMeta(row, args.fake_silicon) if args.fake_silicon else S.SparseDevice(so)
        dense = P.FakeProbe(S.DENSE[row], "verity") if args.fake_silicon else P.Probe(so)
        identity = dev.identity()
    identity.update(nvcc=nvcc, arch=arch)
    results["identity"] = identity
    print("device", json.dumps(identity), flush=True)
    if (args.expect_uuid and dev is not None and not args.fake_silicon
            and not P.uuid_allowed(identity.get("gpu_uuid"), args.expect_uuid)):
        failures.append(f"device UUID {identity.get('gpu_uuid')} != expected {args.expect_uuid}: stopped before any measurement")
        print("FAILURES", failures, flush=True)
        return finish()
    if so is not None and (args.sass_check or args.sass_only):
        sass = P.sass_check(so)
        results["sass"] = sass
        for kern, rec in sass.items():
            if rec["mma_count"] != 1:
                failures.append(f"sass: {kern} has {rec['mma_count']} MMA instructions")
        mine = [rec for kern, rec in sass.items() if S.KERNEL[row] in kern]
        if len(mine) != 1 or mine[0]["opcodes"] != [S.SASS_OPCODE[row]]:
            failures.append(f"sass: the {row} kernel {S.KERNEL[row]} must hold exactly one {S.SASS_OPCODE[row]}: {mine}")
        print("sass", json.dumps({k: v for k, v in sass.items() if "sparse" in k}), flush=True)
    if failures or args.sass_only:
        print("FAILURES" if failures else "SASS GATE PASSED", failures, flush=True)
        return finish()
    results["clocks_start"] = dev.clocks(identity.get("gpu_uuid"))
    try:
        results["layout_check"] = S.layout_check(dev, row, out)
    except RuntimeError as e:
        failures.append(str(e))
        print("FAILURES", failures, flush=True)
        return finish()
    print("layout_check", row, results["layout_check"], flush=True)

    rng = np.random.default_rng(args.seed)
    W, BLK, bad = {}, {}, {}
    for variant in ("a", "b"):
        As, B, META, SFA, SFB, C, slot, nib = part_a_operands(row, variant, rng)
        t0 = time.time()
        try:
            D = dev.tiles(row, As, B, META, SFA, SFB, C)
        except RuntimeError as e:
            failures.append(f"part A variant {variant}: {e}")
            print("FAILURES", failures, flush=True)
            return finish()
        results[f"part_a_{variant}_gpu_seconds"] = round(time.time() - t0, 2)
        np.savez_compressed(out / f"meta_part_a_{variant}.npz", As=As, B=B, META=META, SFA=SFA, SFB=SFB, C=C, D=D, slot=slot, nib=nib)
        W[variant], BLK[variant], bad[variant] = landing_table(D, slot, nib, row, variant)
    rule, per_nibble, blocks = nibble_rule(W["a"], BLK["a"], BLK["b"], S.NBLK[row])
    variants_agree = bool(np.array_equal(W["a"], W["b"]))
    ordered = S.META_VALID.tolist()
    ptx_offsets = {m: S.OFFSETS[m].tolist() for m in ordered}
    hyp = {m: pairs_from_bits(m) for m in range(16)}
    results["part_a"] = {
        "words": int(2 * CHUNKS * 64 * 128), "undecodable": {v: len(b) for v, b in bad.items()},
        "first_undecodable": {v: b[:20] for v, b in bad.items()}, "variants_agree": variants_agree, "blocks": blocks,
        "nibbles": per_nibble,
        "ordered_nibbles_match_ptx": all(rule.get(m) == ptx_offsets[m] for m in ordered),
        "pairs_from_bits": {"matches": sorted(m for m in range(16) if rule.get(m) == hyp[m]),
                            "differs": sorted(m for m in range(16) if rule.get(m) != hyp[m])},
        "note": "per nibble, slots[j] = where stored slot j of a chunk lands (offset from 8c, weight: how many times its product "
                "counts); gather = every slot at most one offset inside its chunk with weight 1; blocks: which scale each product "
                "took, against the stored slot's block and the logical k's"}
    print("part_a", json.dumps({k: v for k, v in results["part_a"].items() if k not in ("nibbles", "first_undecodable")}), flush=True)
    for m in range(16):
        print(f"  nibble {m:04b}", json.dumps(per_nibble[m]["slots"]), "gather" if per_nibble[m]["gather"] else "NOT A GATHER",
              flush=True)
    if bad["a"] or bad["b"]:
        ordered_bad = [x for v in bad.values() for x in v if x["nibble"] in ordered]
        if ordered_bad:
            failures.append(f"part A: {len(ordered_bad)} undecodable words under ordered nibbles (first {ordered_bad[:3]})")
    if not results["part_a"]["ordered_nibbles_match_ptx"]:
        failures.append("part A: an ordered nibble's map differs from PTX's (p0's pair first, then p1's)")
    if not variants_agree:
        failures.append("part A: the two variants' maps differ (the map depends on more than the nibble)")
    if not all(per_nibble[m]["same_in_every_chunk"] for m in ordered):
        failures.append("part A: an ordered nibble maps differently in different chunks")
    results["part_c"] = predicate_tables(rule)
    print("part_c", json.dumps({k: v for k, v in results["part_c"].items() if k != "note"}), flush=True)

    off = gather_offsets(rule)
    nibbles = np.array(sorted(rule), np.uint8)
    n = max(64, args.n_random)
    pool = mp.Pool(args.procs, initializer=_init, initargs=(row,)) if args.procs > 1 else None
    try:
        As, B, META, SFA, SFB, C = part_b_operands(row, rng, n, nibbles, shared=False)
        D = dev.tiles(row, As, B, META, SFA, SFB, C)
        t0 = time.time()
        words, errs = evaluate(pool, row, As, gathered_b(B, META, off), SFA, SFB, C, args.procs)
        mism = (words != D) | errs
        results["part_b"] = {"tiles": n, "elements": n * 128, "nibbles": nibbles.tolist(), "model": f"{S.PINNED[row].name} on the "
                             "stored A and each row's gathered B (Part A's map)", "mismatches": int(mism.sum()),
                             "domain_errors": int(errs.sum()), "model_seconds": round(time.time() - t0, 1),
                             "per_nibble_present": {int(m): int((META == m).sum()) for m in nibbles.tolist()},
                             "first_mismatches": [{"t": int(t), "i": int(i), "j": int(j), "d": f"{int(D[t, i, j]):08x}",
                                                   "model": "error" if errs[t, i, j] else f"{int(words[t, i, j]):08x}"}
                                                  for t, i, j in list(zip(*np.nonzero(mism)))[:20]]}
        np.savez_compressed(out / "meta_part_b.npz", As=As, B=B, META=META, SFA=SFA, SFB=SFB, C=C, D=D, model=words)
        if mism.any():
            failures.append(f"part B: {int(mism.sum())} of {n * 128} words differ from the pinned step on Part A's gather")
        print("part_b", json.dumps({k: v for k, v in results["part_b"].items() if k != "first_mismatches"}), flush=True)
        As, B, META, SFA, SFB, C = part_b_operands(row, rng, n, nibbles, shared=True)
        D = dev.tiles(row, As, B, META, SFA, SFB, C)
        Bg = gathered_b(B, META, off)[:, 0]
        Dd = dense.tiles(S.DENSE[row], As, Bg, C, SFA, SFB)
        diff = Dd != D
        results["part_b2"] = {"tiles": n, "elements": n * 128, "dense_row": S.DENSE[row], "differing_words": int(diff.sum()),
                              "note": "one metadata row per tile; the sparse word against the dense mma.sync on the stored A and "
                                      "the gathered B (hardware against hardware)"}
        np.savez_compressed(out / "meta_part_b2.npz", As=As, B=B, META=META, SFA=SFA, SFB=SFB, C=C, D=D, dense=Dd)
        if diff.any():
            failures.append(f"part B2: {int(diff.sum())} sparse words differ from the dense mma.sync on the gathered B")
        print("part_b2", json.dumps(results["part_b2"]), flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    results["clocks_end"] = dev.clocks(identity.get("gpu_uuid"))
    results["rule"] = {str(m): offs for m, offs in sorted(rule.items())}
    rc = finish()
    run_id = SHARED.RESEARCH_RUN_ID
    if run_id:
        pa = results["part_a"]
        result = {
            "schema": "research/result/v0.1", "kind": "tc-evidence/v1", "run_id": run_id, "instruction": S.INSTRUCTION_ID[row],
            "ptx": ptx, "arch": arch, "sku": identity["device"], "compute_capability": identity.get("compute_capability"),
            "driver": identity.get("driver"), "gpu_uuid": identity.get("gpu_uuid"), "nvcc": nvcc,
            "cuda_visible_devices": identity.get("cuda_visible_devices"),
            "sass_mma_counts": {k: v["mma_count"] for k, v in results.get("sass", {}).items()},
            "sass_opcodes": {k: sorted(set(v["opcodes"])) for k, v in results.get("sass", {}).items()},
            "layout_check": results.get("layout_check"),
            "metadata_rule": {"rule": results["rule"], "gather_nibbles": sorted(rule), "variants_agree": pa["variants_agree"],
                              "ordered_nibbles_match_ptx": pa["ordered_nibbles_match_ptx"], "pairs_from_bits": pa["pairs_from_bits"],
                              "blocks": pa["blocks"], "undecodable": pa["undecodable"],
                              "not_a_gather": {str(m): v["slots"] for m, v in pa["nibbles"].items() if not v["gather"]}},
            "predicate": results["part_c"],
            "counts": {"part_a_words": pa["words"], "part_b_elements": results["part_b"]["elements"],
                       "part_b_mismatches": results["part_b"]["mismatches"],
                       "part_b2_differing_words": results["part_b2"]["differing_words"]},
            "model": f"{S.PINNED[row].name} on the stored A and the gathered B", "model_agrees": not failures,
            "seed": args.seed, "n": n,
            "clocks": {"start": results.get("clocks_start"), "end": results.get("clocks_end"),
                       "note": "recorded, not locked: a capture's words do not depend on the clock"},
            "workload_fingerprint": {"probe": "benchmarks/tc_probe_fp4/meta_rule.py", "instruction": S.INSTRUCTION_ID[row], "arch": arch,
                                     "device": identity["device"], "seed": args.seed, "n_random": n},
            "validation": {"status": "passed" if not failures else "failed",
                           "checker": "tc_probe_fp4 meta_rule.py: UUID, SASS and layout gates; the ordered nibbles' map is PTX's, "
                                      "the same in both variants and every chunk; Part B and B2 reproduce every word",
                           "detail": "; ".join(failures) or f"rule measured for {len(rule)} gather nibbles; every word reproduced",
                           "evidence": ["probe_results.json"]},
            "measurements": [{"name": "wall", "value": results["seconds"], "unit": "s"}],
            "measurement_files": ["probe_results.json"],
            "artifacts": sorted(p.name for p in out.glob("meta_part_*.npz")),
        }
        (out / "result.json.tmp").write_text(json.dumps(result, indent=1))
        os.replace(out / "result.json.tmp", out / "result.json")
    print("FAILURES" if failures else "METADATA RULE MEASURED; ALL WORDS REPRODUCED", failures, f"{results['seconds']}s", flush=True)
    return rc
