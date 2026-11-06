"""The committer and prover of one H1 window (`run.py` drives it; the challenger and the verifier never import this file).

    python window.py commit LAYER.npz WINDOW PRIV OUT.json [--k K] [--cols N] [--calls q_proj,...] [--program-sha512 HEX]
        each call's tables, as a served round commits them before its coin: its A table a/<K>@<c>, the window's A rows
        (`layer.py`'s codes, each with its s_a, padded with zero rows of scale 0 to their slot, `h1unit.slot`) under the
        call's own salts, and its z table zy@<c>, its tiles' z/y records in tile order (`h1unit.records`: z = A Bᵀ exact
        and y its served dequant, what ``scaled_mm_zy`` writes); and every call's B rows (each with its s_w) as b/<K>, the
        weights' (`h1unit.tables`; `h1unit.register`: row/v2 bit rows under fresh salts, one frame-v3-sha512 tree each
        under the window Program's digest, which ``--program-sha512`` gives or this computes from the shape, each
        table's domain binding the window's open root, `h1unit.open_root` of b/<K>'s root): OUT.json the public
        commitment (the slot, each call's B rows, first B row, first tile and tiles, the Program's digest, the window's
        open root, the registered entries with their roots), PRIV the prover's (its token rows too)
    python window.py serve PRIV SALT_HEX OUT.json [--workers W]
        once the salt coin is open: every tile's CP-W receipt (`h1tile.reference`, the word reference the served kernel's
        receipts equal), each an output row under a fresh salt in the window's receipt tree, and how many of the tiles'
        z cells are their committed records' z
    python window.py tamper PRIV TAMPERED_PRIV COMMITMENT.json SERVED.json FLAG
        a negative's window: PRIV's, every salt kept, with unit U's
          --receipt U       receipt from another salt          --receipt-word U  receipt word 0 plus one (mod 2^32)
          --d-word U        receipt (and, under v3, record: z its recombination, y its dequant) from a D_1 word one too large
          --xa-flip U       (v3) record from a z with one entry's X wrap bit xa flipped, y its dequant; the receipt honest
          --zy-word U       record word 0 with its low bit flipped
          --y-cell U        record y of cell 0 plus one (mod 2^16)
          --z-cell U        record z of cell 0 plus one (mod 2^32), y of cell 0 its dequant
          --swap U V        records of units U and V exchanged
          --call U          record the one of the same tile (row block, column block) in the next call
          --z-only U        record of z alone: its y zeroed
        a record changed re-registers its call's z table (COMMITMENT.json the window's public commitment, that table's root
        and entry), a receipt changed re-commits the receipt tree (SERVED.json); the input rows a prover of it holds are
        PRIV's
    python window.py open PRIV SALT_HEX UNIT,... OUT_DIR [--tamper-a | --unregistered-ok]
        the sampled proof blocks' rows (their public row's and ok row's too) and salts for staging, and each A row's, B row's,
        record's and ok row's registered value, position and path (OUT_DIR/rows.npz, the prover's: ``v_``, ``r_`` and
        ``p_`` of its port), and their opening (OUT_DIR/opening.json: each committed row's b || c and its path to its tree's
        root). The ok row is the one word 1 under its public salt (`h1unit.ok_salt`), row u of the verifier's registered
        ``ok`` (`h1unit.register_ok`, all public). ``--tamper-a``: the rows a prover that serves an A row the registration
        does not hold stages, the first unit's a0 row (at every unit that reads it) with code 0's low bit flipped under a
        fresh salt, its registered path kept; the opening is the honest one. ``--unregistered-ok``: the rows of a prover
        that commits its ok rows afresh, with no registered position or path

A tile is (call, ti, tj): A rows [R ti, R (ti + 1)) of the slot and B rows [C tj, C (tj + 1)) of the call's; tiles are
numbered call by call, row block major, and a tile's number is its position, the unit's ``position`` input
(`h1window.PouwH1Window`). Tile t of call c reads its A rows at a/<K>@<c>, its record at row t of zy@<c> and its B rows
at b/<K> (`h1window.table_reads`). The window is one seed group, `h1unit.window_roots` every unit's ``roots`` input.
``--k`` truncates every row to its first K codes (a smoke run's), each row's scale kept.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h1unit as U  # noqa: E402

R, C = U.TILE
RC = R * C


def _salts(n: int) -> np.ndarray:
    return np.frombuffer(os.urandom(U.SALT_BYTES * n), np.uint8).reshape(n, U.SALT_BYTES)


def _bl(a: np.ndarray) -> list[bytes]:
    return [bytes(r) for r in a]


def _bcs(bcs) -> np.ndarray:
    return np.frombuffer(b"".join(bcs), np.uint8).reshape(len(bcs), 128)


def tiles(com: dict) -> list[tuple[int, int, int]]:
    out = []
    for c, call in enumerate(com["calls"]):
        for ti in range(com["slot"] // R):
            for tj in range(call["n"] // C):
                out.append((c, ti, tj))
    return out


def tile_of(com: dict, u: int) -> tuple[int, int, int]:
    for c, call in enumerate(com["calls"]):
        if call["tile0"] <= u < call["tile0"] + call["tiles"]:
            v = u - call["tile0"]
            return c, v // (call["n"] // C), v % (call["n"] // C)
    raise ValueError(f"unit {u} is no tile of the window")


def unit_of(com: dict, c: int, ti: int, tj: int) -> int:
    call = com["calls"][c]
    return call["tile0"] + ti * (call["n"] // C) + tj


def window_records(com: dict, A: np.ndarray, SA: np.ndarray, B: np.ndarray, SB: np.ndarray) -> np.ndarray:
    """Every tile's record (`h1unit.records`, `h1unit.record_words`), row u unit u's: each call's z = A Bᵀ and y at once."""
    out = np.zeros((com["tiles"], 3 * RC), np.uint16)
    for c, call in enumerate(com["calls"]):
        z, y = U.records(A, B[call["b0"]:call["b0"] + call["n"]], SA, SB[call["b0"]:call["b0"] + call["n"]])
        for ti in range(com["slot"] // R):
            for tj in range(call["n"] // C):
                cut = (slice(R * ti, R * ti + R), slice(C * tj, C * tj + C))
                out[unit_of(com, c, ti, tj)] = U.record_words(z[cut], y[cut])
    return out


def cmd_commit(a) -> int:
    lay = np.load(a.layer)
    priv = Path(a.priv)
    priv.mkdir(parents=True, exist_ok=True)
    calls = a.calls.split(",") if a.calls else sorted(k[2:] for k in lay.files if k.startswith("b_"))
    A = lay["a"][:, :a.k] if a.k else lay["a"]
    rows, k = A.shape
    shape = U.Shape(k, R, C)
    ports = {p.name: p for p in shape.ports()}
    m = U.slot(rows)
    A = np.concatenate([A, np.zeros((m - rows, k), A.dtype)])
    SA = np.concatenate([np.asarray(lay["sa"], np.float32), np.zeros(m - rows, np.float32)])
    out_calls, Bs, SBs, tile0, b0 = [], [], [], 0, 0
    for name in calls:
        B, SB = lay[f"b_{name}"][:, :k], np.asarray(lay[f"s_{name}"], np.float32)
        if a.cols:
            B, SB = B[:a.cols], SB[:a.cols]
        n = B.shape[0] - B.shape[0] % C
        t = (m // R) * (n // C)
        out_calls.append({"name": name, "n": n, "b0": b0, "tile0": tile0, "tiles": t})
        Bs.append(B[:n])
        SBs.append(SB[:n])
        tile0, b0 = tile0 + t, b0 + n
    B, SB = np.concatenate(Bs), np.concatenate(SBs)
    com = {"schema": "verity/pouw/h1-prove/commitment/v5", "window": a.window,
           "unit": {"definition": "PouwH1TileZy_v1", "tile": f"PouwH1Tile_{U.tile()}", "k": k, "r": R, "c": C,
                    "exact": "GemmAccumulatorS8_v2"}, "slot": m, "calls": out_calls, "b_rows": len(B), "tiles": tile0}
    if a.program_sha512:
        prog = a.program_sha512
    else:
        from verity.core.protocols.sampled_proofs.one_stage import partition as P
        prog = P.program_digest(shape.window([(c["name"], c["n"]) for c in out_calls], m)[0])
    ZY = window_records(com, A, SA, B, SB)
    wa, nc = U.row_words(A, SA), len(out_calls)
    sa, sb, szy = _salts(nc * m).reshape(nc, m, U.SALT_BYTES), _salts(len(B)), _salts(tile0)
    reg = {f"b/{k}": U.register(prog, f"b/{k}", U.row_words(B, SB), _bl(sb), None)}
    root = U.open_root(a.window, reg[f"b/{k}"].root)
    for c, call in enumerate(out_calls):
        cut = slice(call["tile0"], call["tile0"] + call["tiles"])
        reg[f"a/{k}@{c}"] = U.register(prog, f"a/{k}@{c}", wa, _bl(sa[c]), root)
        reg[f"zy@{c}"] = U.register(prog, f"zy@{c}", ZY[cut], _bl(szy[cut]), root)
    bczy = sum((reg[f"zy@{c}"].bcs for c in range(nc)), ())
    if list(reg) != list(U.tables(com)) or any(r.leaves != n for r, n in zip(reg.values(), U.tables(com).values())):
        raise SystemExit(f"the registered values {[(v, r.leaves) for v, r in reg.items()]} are not h1unit.tables'")
    if any(reg[f"a/{k}@{c}"].bcs != tuple(U.commit_strings(ports["a0"], wa, _bl(sa[c]))) for c in range(nc)) or \
            bczy != tuple(U.commit_strings(ports["zy"], ZY, _bl(szy))):
        raise SystemExit("the registered A and z table rows' b || c are not the statement ports a0's and zy's")
    np.savez(priv / "rows.npz", a=A, b=B, scale_a=SA, scale_b=SB, zyw=ZY, sa=sa, sb=sb, szy=szy,
             bca=np.stack([_bcs(reg[f"a/{k}@{c}"].bcs) for c in range(nc)]), bcb=_bcs(reg[f"b/{k}"].bcs), bczy=_bcs(bczy))
    com.update(program_sha512=prog, window_root=root, registered=[r.entry() for r in reg.values()], rows=rows,
               tokens=[int(x) for x in lay["tokens"]])
    (priv / "commitment.json").write_text(json.dumps(com, indent=1))
    Path(a.out).write_text(json.dumps({x: com[x] for x in PUBLIC}, indent=1))
    print(json.dumps({"window": a.window, "k": k, "slot": m, "calls": {c["name"]: c["n"] for c in out_calls}, "tiles": tile0}))
    return 0


#: the public commitment's fields; the prover's copy adds its token rows and tokens
PUBLIC = ("schema", "window", "unit", "slot", "calls", "b_rows", "program_sha512", "window_root", "registered", "tiles")


roots = U.window_roots


def _rows(priv: dict, com: dict, u: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Unit u's A rows and B rows (codes) and their scales."""
    c, ti, tj = tile_of(com, u)
    b0 = com["calls"][c]["b0"] + C * tj
    a, b = slice(R * ti, R * (ti + 1)), slice(b0, b0 + C)
    return priv["a"][a], priv["b"][b], priv["scale_a"][a], priv["scale_b"][b]


_W: dict = {}


def _init(priv_dir: str, salt: bytes) -> None:
    _W["priv"] = dict(np.load(Path(priv_dir) / "rows.npz"))
    _W["com"] = json.loads((Path(priv_dir) / "commitment.json").read_text())
    _W["salt"] = salt


def _xa_target(a: list, b: list, xa: list) -> tuple[int, int]:
    """The entry (i, t) whose X wrap bit an ``xa_flip`` prover flips: the first that is set where some B(j, t) is nonzero,
    else the first where some B(j, t) is nonzero (so the flip moves z either way)."""
    live = [(i, t) for i in range(len(a)) for t in range(len(a[0])) if any(bj[t] for bj in b)]
    return next(((i, t) for i, t in live if xa[i][t]), live[0])


def _serve_one(u: int, d_word: bool = False, xa_flip: bool = False) -> tuple[int, list[int], list[int], int, list[int]]:
    """Unit u's tile z and receipt (`h1tile.reference`, v3's under ``H1_TILE=v3``), how many of its cells' z are its
    committed record's (the served ``scaled_mm_zy``'s exact A Bᵀ), and how many entries of each of its rows set v3's X wrap
    bit xa (A = 127 with dither bit 1). ``d_word``: as a kernel whose lane 0 made its D_1 entry 0 one too large, folded and
    recombined like the honest words, so z and the receipt agree with each other. ``xa_flip``: v3's z with one entry's xa
    flipped (`_xa_target`), the receipt the honest one."""
    from verity.core.protocols.pouw.circuit import h1tile as H
    from verity.core.protocols.pouw.circuit import seeds as SD
    a, b, _sa, _sw = _rows(_W["priv"], _W["com"], u)
    roots_ = roots(_W["com"])
    a, b = [[int(x) & 0xFF for x in r] for r in a], [[int(x) & 0xFF for x in r] for r in b]
    v3 = U.tile() == "v3"
    lay = H.Layout(len(a), len(b), len(a[0]))
    rows, cols = H.noise(SD.seed(_W["salt"], u, *roots_), lay) if v3 or d_word or xa_flip else (None, None)
    xa = [H.row_wrap(a[i], rows[i]) for i in range(lay.r)] if rows else [[0] * lay.k for _ in range(lay.r)]
    if xa_flip and not v3:
        raise SystemExit("--xa-flip needs H1_TILE=v3: v1's z has no X wrap bit")
    if not d_word and not xa_flip:
        z, rc = H.reference(a, b, _W["salt"], u, roots_, stream=v3)
    else:
        keys = H.fold_keys(SD.receipt_seed(_W["salt"], u, *roots_), lay)
        cp, cells = [[0, 0] for _ in range(lay.sets)], {}
        xs = [[(a[i][t] + rows[i][0][t]) & 0xFF for t in range(lay.k)] for i in range(lay.r)]
        wbs = [H.col_wrap(b[j], cols[j]) for j in range(lay.c)]
        pxa = [list(r) for r in xa]
        if xa_flip:
            fi, ft = _xa_target(a, b, xa)
            pxa[fi][ft] ^= 1
        pis = [(rows[i][0][:lay.k], pxa[i]) for i in range(lay.r)]
        pjs = [(b[j], cols[j][0][:lay.k]) for j in range(lay.c)]
        for ln in range(lay.lanes):
            sets = H.lane_sets(a, b, rows, cols, lay, ln)
            if ln == 0 and d_word:
                sets[lay.ks + 1][0] = (sets[lay.ks + 1][0] + 1) & H.M32
            st = list(keys[ln])
            for s, p in enumerate(sets):
                st = H.SG.absorb(H.FOLD, st, p)
                for h in range(2):
                    for v in st[8 * h:8 * h + 8]:
                        cp[s][h] ^= v
            for e in range(32):
                i, j = lay.rows(ln)[H.SG.row_slot(e)], lay.cols(ln)[H.SG.col_slot(e)]
                cells[(i, j)] = H.stream_cell(sets[lay.ks][e], [d[e] for d in sets[lay.ks + 1:]], xs[i], wbs[j], pis[i],
                                              pjs[j])
        pairs = [(i, j) for i in range(lay.r) for j in range(lay.c)]
        z = [cells[p] for p in pairs] if v3 else [H.SG.seg_dot(0, a[i], b[j]) for i, j in pairs]
        rc = [w & H.M32 for s in cp for w in s]
    committed, _y = U.split_record(_W["priv"]["zyw"][u], RC)
    return u, z, rc, sum(int((x & H.M32) == y) for x, y in zip(z, committed)), [sum(r) for r in xa]


def _receipts(com: dict, rw: np.ndarray, sr: np.ndarray) -> tuple[list[bytes], bytes]:
    port = U.Shape(com["unit"]["k"], R, C).ports()[-2]
    bcr = U.commit_strings(port, rw, _bl(sr))
    return bcr, U.levels(U.domain(com["window"], "receipt", com["tiles"]), bcr)[-1][0]


def cmd_serve(a) -> int:
    priv_dir = Path(a.priv)
    com = json.loads((priv_dir / "commitment.json").read_text())
    salt = bytes.fromhex(a.salt)
    T = com["tiles"]
    rcs, eq, xas = [None] * T, 0, []
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(str(priv_dir), salt)) as ex:
        for u, _z, rc, n, xa in ex.map(_serve_one, range(T), chunksize=64):
            rcs[u], eq = rc, eq + n
            xas.append(xa)
    priv = dict(np.load(priv_dir / "rows.npz"))
    rw, sr = np.stack([U.receipt_words(rc) for rc in rcs]), _salts(T)
    bcr, rr = _receipts(com, rw, sr)
    priv.update(rw=rw, sr=sr, bcr=_bcs(bcr))
    np.savez(priv_dir / "rows.npz", **priv)
    served = {"receipt_root": rr.hex(), "tiles": T, "salt": salt.hex(), "tile": U.tile(),
              "z_is_served_product": {"cells": T * RC, "equal": eq},
              "xa": {"entries": sum(map(sum, xas)), "tile_rows": T * R, "tile_rows_with_xa": sum(int(n > 0) for r in xas for n in r),
                     "tiles_with_xa": sum(int(any(r)) for r in xas)}}
    (priv_dir / "served.json").write_text(json.dumps(served))
    Path(a.out).write_text(json.dumps(served, indent=1))
    print(json.dumps(served))
    return 0


def plus_one(words: np.ndarray, bits: int = 32) -> np.ndarray:
    """A ``bits``-bit value's u16 words (little-endian) plus one, mod 2^bits."""
    n = bits // 16
    v = (sum(int(w) << 16 * i for i, w in enumerate(words[:n])) + 1) % (1 << bits)
    return np.array([v >> 16 * i & 0xFFFF for i in range(n)], np.uint16)


def _with_z(priv: dict, com: dict, u: int, z: list[int]) -> np.ndarray:
    """Unit u's record with z ``z`` (s32 words) and y its dequant at the unit's scales."""
    _a, _b, sa, sw = _rows(priv, com, u)
    z = np.array([v - (1 << 32) if v >> 31 else v for v in (int(x) & U.M32 for x in z)], np.int64).reshape(R, C)
    return U.record_words(z, U.dequant(z, np.asarray(sa)[:, None], np.asarray(sw)[None, :]))


def _record(a, com: dict, priv: dict, salt: bytes) -> tuple[dict[int, np.ndarray], dict]:
    """The records a record flag gives (unit -> record words), and what it did."""
    zyw = priv["zyw"]
    if a.zy_word is not None:
        u, w = a.zy_word, zyw[a.zy_word].copy()
        w[0] ^= 1
        return {u: w}, {}
    if a.y_cell is not None:
        u, w = a.y_cell, zyw[a.y_cell].copy()
        w[2 * RC] = plus_one(w[2 * RC:2 * RC + 1], 16)[0]
        return {u: w}, {}
    if a.z_cell is not None:
        u = a.z_cell
        z, _y = U.split_record(zyw[u], RC)
        z[0] = (z[0] + 1) & U.M32
        return {u: _with_z(priv, com, u, z)}, {}
    if a.swap is not None:
        u, v = a.swap
        return {u: zyw[v].copy(), v: zyw[u].copy()}, {"with": v, "records_differ": bool(not np.array_equal(zyw[u], zyw[v]))}
    if a.call is not None:
        c, ti, tj = tile_of(com, a.call)
        c2 = (c + 1) % len(com["calls"])
        if c2 == c or tj >= com["calls"][c2]["n"] // C:
            raise SystemExit(f"unit {a.call}: no other call has its tile ({ti}, {tj})")
        v = unit_of(com, c2, ti, tj)
        return {a.call: zyw[v].copy()}, {"from": v, "records_differ": bool(not np.array_equal(zyw[a.call], zyw[v]))}
    if a.z_only is not None:
        w = zyw[a.z_only].copy()
        w[2 * RC:] = 0
        return {a.z_only: w}, {}
    u = a.d_word if a.d_word is not None else a.xa_flip
    _init(a.priv, salt)
    _u, z, rc, _n, xa = _serve_one(u, d_word=a.d_word is not None, xa_flip=a.xa_flip is not None)
    out = {u: _with_z(priv, com, u, z)} if U.tile() == "v3" else {}
    info = {"receipt": U.receipt_words(rc)}
    if a.xa_flip is not None:
        info["xa_flip"] = {"xa_set_in_unit": sum(xa), "record_words_changed": int((out[u] != zyw[u]).sum()),
                           "receipt_is_honest": bool((info["receipt"] == priv["rw"][u]).all())}
    return out, info


def cmd_tamper(a) -> int:
    src, dst = Path(a.priv), Path(a.tampered)
    dst.mkdir(parents=True, exist_ok=True)
    com = json.loads((src / "commitment.json").read_text())
    salt = bytes.fromhex(json.loads((src / "served.json").read_text())["salt"])
    priv = dict(np.load(src / "rows.npz"))
    what = next(x for x in FLAGS if getattr(a, x) is not None)
    records, receipts, info = {}, {}, {}
    if what == "receipt":
        _init(str(src), hashlib.sha512(b"verity/pouw/h1-prove/another-salt\0" + salt).digest())
        receipts[a.receipt] = U.receipt_words(_serve_one(a.receipt)[2])
    elif what == "receipt_word":
        receipts[a.receipt_word] = np.concatenate([plus_one(priv["rw"][a.receipt_word][:2]), priv["rw"][a.receipt_word][2:]])
    else:
        records, info = _record(a, com, priv, salt)
        if "receipt" in info:
            receipts[a.d_word if a.d_word is not None else a.xa_flip] = info.pop("receipt")
    receipts = {u: w for u, w in receipts.items() if not np.array_equal(w, priv["rw"][u])}
    records = {u: w for u, w in records.items() if not np.array_equal(w, priv["zyw"][u])}
    port = {p.name: p for p in U.Shape(com["unit"]["k"], R, C).ports()}
    tables = []
    if records:
        priv["zyw"], priv["bczy"] = priv["zyw"].copy(), priv["bczy"].copy()
        for u, w in records.items():
            priv["zyw"][u] = w
            priv["bczy"][u] = np.frombuffer(U.commit_strings(port["zy"], w[None, :], [bytes(priv["szy"][u])])[0], np.uint8)
        entries = {e["value"]: e for e in com["registered"]}
        for c in sorted({tile_of(com, u)[0] for u in records}):
            call, tables = com["calls"][c], [*tables, f"zy@{c}"]
            cut = slice(call["tile0"], call["tile0"] + call["tiles"])
            reg = U.register(com["program_sha512"], f"zy@{c}", priv["zyw"][cut], _bl(priv["szy"][cut]), com["window_root"])
            if reg.bcs != tuple(_bl(priv["bczy"][cut])):
                raise SystemExit(f"the re-registered zy@{c} rows' b || c are not the tampered rows'")
            entries[f"zy@{c}"] = reg.entry()
        com["registered"] = list(entries.values())
    if receipts:
        priv["rw"], priv["bcr"] = priv["rw"].copy(), priv["bcr"].copy()
        for u, w in receipts.items():
            priv["rw"][u] = w
            priv["bcr"][u] = np.frombuffer(U.commit_strings(port["o0"], w[None, :], [bytes(priv["sr"][u])])[0], np.uint8)
    served = dict(json.loads((src / "served.json").read_text()),
                  receipt_root=U.levels(U.domain(com["window"], "receipt", com["tiles"]), _bl(priv["bcr"]))[-1][0].hex(),
                  tampered={"what": what, "records": sorted(records), "tables": tables, "receipts": sorted(receipts), **info})
    np.savez(dst / "rows.npz", **priv)
    (dst / "commitment.json").write_text(json.dumps(com, indent=1))
    (dst / "served.json").write_text(json.dumps(served))
    Path(a.commitment).write_text(json.dumps({x: com[x] for x in PUBLIC}, indent=1))
    Path(a.out).write_text(json.dumps(served, indent=1))
    print(json.dumps(served))
    return 0


def registered_rows(priv: dict, com: dict, u: int) -> dict[str, tuple[str, int, np.ndarray, bytes, bytes]]:
    """Unit u's registered rows by statement port, each its value, position, words, salt and ``b || c``: A row i at row
    R ti + i of its call's A table, B row j at its call's row C tj + j of b/<K>, its record at row t of its call's z
    table (`h1window.table_reads`)."""
    k, (c, ti, tj) = com["unit"]["k"], tile_of(com, u)
    call, out = com["calls"][c], {}
    for i in range(R):
        r = R * ti + i
        out[f"a{i}"] = (f"a/{k}@{c}", r, U.row_words(priv["a"][r:r + 1], priv["scale_a"][r:r + 1])[0], bytes(priv["sa"][c][r]),
                        bytes(priv["bca"][c][r]))
    for j in range(C):
        r = call["b0"] + C * tj + j
        out[f"b{j}"] = (f"b/{k}", r, U.row_words(priv["b"][r:r + 1], priv["scale_b"][r:r + 1])[0], bytes(priv["sb"][r]),
                        bytes(priv["bcb"][r]))
    out["zy"] = (f"zy@{c}", u - call["tile0"], priv["zyw"][u], bytes(priv["szy"][u]), bytes(priv["bczy"][u]))
    return out


def table_bcs(priv: dict, com: dict, value: str) -> list[bytes]:
    """Every row's ``b || c`` of the registered value ``value`` (`h1unit.tables`), in order."""
    k = com["unit"]["k"]
    if value == f"b/{k}":
        return _bl(priv["bcb"])
    name, c = value.rsplit("@", 1)
    call = com["calls"][int(c)]
    return _bl(priv["bca"][int(c)]) if name == f"a/{k}" else _bl(priv["bczy"][call["tile0"]:call["tile0"] + call["tiles"]])


def cmd_open(a) -> int:
    priv_dir, out = Path(a.priv), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    com = json.loads((priv_dir / "commitment.json").read_text())
    priv = dict(np.load(priv_dir / "rows.npz"))
    units = [int(x) for x in a.units.split(",")]
    T, w, prog = com["tiles"], com["window"], com["program_sha512"]
    lr = U.levels(U.domain(w, "receipt", T), _bl(priv["bcr"]))
    lv: dict = {}
    salt = bytes.fromhex(a.salt)
    reg_ports = [f"a{i}" for i in range(R)] + [f"b{j}" for j in range(C)] + ["zy"]
    rows = {p: [] for p in reg_ports} | {"pub": [], "o0": [], "o1": []}
    salts = {p: [] for p in rows}
    stated = [*reg_ports, *(() if a.unregistered_ok else ("o1",))]
    values, positions, paths = {p: [] for p in stated}, {p: [] for p in stated}, {p: [] for p in stated}
    okr = U.register_ok(prog, w, T, com["window_root"])
    openings = []
    for u in units:
        c, ti, tj = tile_of(com, u)
        op = {"unit": u, "call": com["calls"][c]["name"], "tile": [ti, tj], "rows": {}}
        for port, (value, r, words, s, bc) in registered_rows(priv, com, u).items():
            if value not in lv:
                lv[value] = U.registered_levels(prog, value, table_bcs(priv, com, value), com["window_root"])
            rows[port].append(words)
            salts[port].append(s)
            values[port].append(value)
            positions[port].append(r)
            paths[port].append("".join(U.path(lv[value], r)))
            op["rows"][port] = {"value": value, "index": r, "bc": bc.hex(), "path": U.path(lv[value], r)}
        rows["pub"].append(U.pub_words(salt, u, roots(com)))
        salts["pub"].append(U.pub_salt(w, u))
        rows["o0"].append(priv["rw"][u])
        salts["o0"].append(bytes(priv["sr"][u]))
        rows["o1"].append(U.OK)
        salts["o1"].append(U.ok_salt(w, u))
        if "o1" in values:
            values["o1"].append(U.OK_VALUE)
            positions["o1"].append(u)
            paths["o1"].append(okr.path(u).hex())
        op["rows"]["o0"] = {"value": "receipt", "index": u, "bc": bytes(priv["bcr"][u]).hex(), "path": U.path(lr, u)}
        openings.append(op)
    tampered = None
    if a.tamper_a:
        # instance 0's a0 row, at every instance that reads its position, with code 0's low bit flipped under one fresh salt;
        # the registered path kept
        r0, fresh = positions["a0"][0], bytes(_salts(1)[0])
        at = [i for i, rv in enumerate(zip(positions["a0"], values["a0"])) if rv == (r0, values["a0"][0])]
        for i in at:
            rows["a0"][i] = rows["a0"][i].copy()
            rows["a0"][i][0] ^= 1
            salts["a0"][i] = fresh
        tampered = {"port": "a0", "value": values["a0"][0], "position": r0, "instances": at}
    np.savez(out / "rows.npz", units=np.array(units), salt=np.frombuffer(salt, np.uint8),
             roots=np.array([[np.frombuffer(r, np.uint8) for r in roots(com)]] * len(units)),
             **{f"w_{p}": np.stack(v) for p, v in rows.items()},
             **{f"s_{p}": np.frombuffer(b"".join(v), np.uint8).reshape(len(v), U.SALT_BYTES) for p, v in salts.items()},
             **{f"v_{p}": np.array(v) for p, v in values.items()},
             **{f"r_{p}": np.array(v, np.int64) for p, v in positions.items()}, **{f"p_{p}": np.array(v) for p, v in paths.items()})
    (out / "opening.json").write_text(json.dumps({"window": w, "units": openings}, indent=1))
    print(json.dumps({"opened": units, **({"tampered": tampered} if tampered else {})}))
    return 0


#: `tamper`'s flags, by their attribute
FLAGS = ("receipt", "receipt_word", "d_word", "xa_flip", "zy_word", "y_cell", "z_cell", "swap", "call", "z_only")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("commit")
    p.add_argument("layer")
    p.add_argument("window")
    p.add_argument("priv")
    p.add_argument("out")
    p.add_argument("--k", type=int, default=0)
    p.add_argument("--cols", type=int, default=0)
    p.add_argument("--calls", default="")
    p.add_argument("--program-sha512", default="", help="the window Program's digest, when the caller holds it")
    p = sub.add_parser("serve")
    p.add_argument("priv")
    p.add_argument("salt")
    p.add_argument("out")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p = sub.add_parser("tamper")
    p.add_argument("priv")
    p.add_argument("tampered")
    p.add_argument("commitment")
    p.add_argument("out")
    g = p.add_mutually_exclusive_group(required=True)
    for f in FLAGS:
        g.add_argument("--" + f.replace("_", "-"), type=int, nargs=2 if f == "swap" else None)
    p = sub.add_parser("open")
    p.add_argument("priv")
    p.add_argument("salt")
    p.add_argument("units")
    p.add_argument("out")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--tamper-a", action="store_true")
    g.add_argument("--unregistered-ok", action="store_true")
    a = ap.parse_args(argv)
    return {"commit": cmd_commit, "serve": cmd_serve, "tamper": cmd_tamper, "open": cmd_open}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
