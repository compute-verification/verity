"""The committer and prover of one H1 window (`run.py` drives it; the challenger and the verifier never import this file).

    python window.py commit LAYER.npz WINDOW PRIV OUT.json [--k K] [--cols N] [--calls q_proj,...]
        the window's A rows (`layer.py`'s codes, padded with zero rows to their slot, `h1unit.slot`) and every call's B
        rows as row/v2 bit rows under fresh salts, one frame-v3-sha512 tree per side: OUT.json the public commitment (the
        slot, each call's B rows, first B row and first tile, the two roots), PRIV the prover's (its token rows too)
    python window.py serve PRIV SALT_HEX OUT.json [--workers W]
        once the salt coin is open: every tile's z (its cells' exact products) and CP-W receipt (`h1tile.reference`, the
        word reference the served kernel's receipts equal), each an output row under a fresh salt in the window's z tree
        and receipt tree
    python window.py tamper PRIV TAMPERED_PRIV OUT.json (--z U | --receipt U | --z-cell U | --receipt-word U | --d-word U
                                                         | --xa-flip U)
        a negative's window: PRIV's, every salt kept, with unit U's z committed with one word flipped, its receipt from
        another salt, its z cell 0 plus one, its receipt word 0 plus one (both 32-bit, mod 2^32), both from a D_1 word one
        too large, or (v3) its z with one entry's X wrap bit xa flipped; the input rows a prover of it holds are PRIV's
    python window.py open PRIV SALT_HEX UNIT,... OUT_DIR
        the drawn units' rows (their public row's too) and salts for staging (OUT_DIR/rows.npz, the prover's), and their
        opening (OUT_DIR/opening.json: each committed row's b || c and its path to the window's registered root)

A tile is (call, ti, tj): A rows [R ti, R (ti + 1)) of the slot and B rows [C tj, C (tj + 1)) of the call's; tiles are
numbered call by call, row block major, and a tile's number is its position, the unit's ``position`` input
(`h1window.PouwH1Window`). The window is one seed group, its roots (root(a), root(b)) every unit's ``roots`` input. ``--k``
truncates every row to its first K codes (a smoke run's).
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

R, C = 4, 8


def _salts(n: int) -> np.ndarray:
    return np.frombuffer(os.urandom(U.SALT_BYTES * n), np.uint8).reshape(n, U.SALT_BYTES)


def _bl(a: np.ndarray) -> list[bytes]:
    return [bytes(r) for r in a]


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


def cmd_commit(a) -> int:
    lay = np.load(a.layer)
    priv = Path(a.priv)
    priv.mkdir(parents=True, exist_ok=True)
    calls = a.calls.split(",") if a.calls else sorted(k[2:] for k in lay.files if k.startswith("b_"))
    A = lay["a"][:, :a.k] if a.k else lay["a"]
    rows, k = A.shape
    shape = U.Shape(k, R, C)
    port = shape.ports()[0]
    m = U.slot(rows, R)
    A = np.concatenate([A, np.zeros((m - rows, k), A.dtype)])
    out_calls, Bs, tile0, b0 = [], [], 0, 0
    for name in calls:
        B = lay[f"b_{name}"][:, :k]
        if a.cols:
            B = B[:a.cols]
        n = B.shape[0] - B.shape[0] % C
        t = (m // R) * (n // C)
        out_calls.append({"name": name, "n": n, "b0": b0, "tile0": tile0, "tiles": t})
        Bs.append(B[:n])
        tile0, b0 = tile0 + t, b0 + n
    B = np.concatenate(Bs)
    sa, sb = _salts(m), _salts(len(B))
    bca = U.commit_strings(port, U.row_words(A), _bl(sa))
    bcb = U.commit_strings(port, U.row_words(B), _bl(sb))
    ra = U.levels(U.domain(a.window, "a", m), bca)[-1][0]
    rb = U.levels(U.domain(a.window, "b", len(B)), bcb)[-1][0]
    np.savez(priv / "rows.npz", a=A, b=B, sa=sa, sb=sb, bca=np.frombuffer(b"".join(bca), np.uint8).reshape(m, 128),
             bcb=np.frombuffer(b"".join(bcb), np.uint8).reshape(len(B), 128))
    com = {"schema": "verity/pouw/h1-prove/commitment/v1", "window": a.window, "unit": {"definition": "PouwH1Tile_v1", "k": k, "r": R, "c": C,
           "exact": "GemmAccumulatorS8_v2"}, "slot": m, "calls": out_calls, "b_rows": len(B), "root_a": ra.hex(),
           "root_b": rb.hex(), "tiles": tile0, "rows": rows, "tokens": [int(x) for x in lay["tokens"]]}
    (priv / "commitment.json").write_text(json.dumps(com, indent=1))
    Path(a.out).write_text(json.dumps({x: com[x] for x in PUBLIC}, indent=1))
    print(json.dumps({"window": a.window, "k": k, "slot": m, "calls": {c["name"]: c["n"] for c in out_calls}, "tiles": tile0}))
    return 0


#: the public commitment's fields; the prover's copy adds its token rows and tokens
PUBLIC = ("schema", "window", "unit", "slot", "calls", "b_rows", "root_a", "root_b", "tiles")


roots = U.window_roots


def _rows(priv: dict, com: dict, u: int) -> tuple[np.ndarray, np.ndarray, tuple[bytes, bytes], dict]:
    c, ti, tj = tile_of(com, u)
    call = com["calls"][c]
    b0 = call["b0"] + C * tj
    return priv["a"][R * ti:R * (ti + 1)], priv["b"][b0:b0 + C], roots(com), call


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
    """Unit u's z and receipt (`h1tile.reference`, v3's under ``H1_TILE=v3``), how many of its cells' z are the served
    product A Bᵀ, what vLLM's `_int_mm` returns on the call's int8 rows, computed here in Python (every cell's, v1's and v3's: v3's wrap correction and dither
    peel included), and how many entries of each of its rows set v3's X wrap bit xa (A = 127 with dither bit 1). ``d_word``: as a kernel
    whose lane 0 made its D_1 entry 0 one too large, folded and recombined like the honest words, so z and the receipt
    agree with each other. ``xa_flip``: v3's z with one entry's xa flipped (`_xa_target`), the receipt the honest one."""
    from verity.core.protocols.pouw.circuit import h1tile as H
    from verity.core.protocols.pouw.circuit import seeds as SD
    a, b, roots, _call = _rows(_W["priv"], _W["com"], u)
    a, b = [[int(x) & 0xFF for x in r] for r in a], [[int(x) & 0xFF for x in r] for r in b]
    v3 = U.tile() == "v3"
    lay = H.Layout(len(a), len(b), len(a[0]))
    rows, cols = H.noise(SD.seed(_W["salt"], u, *roots), lay) if v3 or d_word or xa_flip else (None, None)
    xa = [H.row_wrap(a[i], rows[i]) for i in range(lay.r)] if rows else [[0] * lay.k for _ in range(lay.r)]
    if xa_flip and not v3:
        raise SystemExit("--xa-flip needs H1_TILE=v3: v1's z has no X wrap bit")
    if not d_word and not xa_flip:
        z, rc = H.reference(a, b, _W["salt"], u, roots, stream=v3)
    else:
        keys = H.fold_keys(SD.receipt_seed(_W["salt"], u, *roots), lay)
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
    sx = lambda v: v - 256 if v & 0x80 else v  # noqa: E731
    served = [sum(sx(x) * sx(w) for x, w in zip(ai, bj)) & H.M32 for ai in a for bj in b]
    return u, z, rc, sum(int(x == y) for x, y in zip(z, served)), [sum(r) for r in xa]


def cmd_serve(a) -> int:
    priv_dir = Path(a.priv)
    com = json.loads((priv_dir / "commitment.json").read_text())
    salt = bytes.fromhex(a.salt)
    T = com["tiles"]
    shape = U.Shape(com["unit"]["k"], R, C)
    zs, rcs, eq, xas = [None] * T, [None] * T, 0, []
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(str(priv_dir), salt)) as ex:
        for u, z, rc, n, xa in ex.map(_serve_one, range(T), chunksize=64):
            zs[u], rcs[u], eq = z, rc, eq + n
            xas.append(xa)
    priv = dict(np.load(priv_dir / "rows.npz"))
    zw = np.stack([U.z_words(z) for z in zs])
    rw = np.stack([U.receipt_words(rc) for rc in rcs])
    ports = shape.ports()
    sz, sr = _salts(T), _salts(T)
    bcz = U.commit_strings(ports[-2], zw, _bl(sz))
    bcr = U.commit_strings(ports[-1], rw, _bl(sr))
    rz = U.levels(U.domain(com["window"], "z", T), bcz)[-1][0]
    rr = U.levels(U.domain(com["window"], "receipt", T), bcr)[-1][0]
    priv.update(zw=zw, rw=rw, sz=sz, sr=sr, bcz=np.frombuffer(b"".join(bcz), np.uint8).reshape(T, 128),
                bcr=np.frombuffer(b"".join(bcr), np.uint8).reshape(T, 128))
    np.savez(priv_dir / "rows.npz", **priv)
    served = {"z_root": rz.hex(), "receipt_root": rr.hex(), "tiles": T, "salt": salt.hex(), "tile": U.tile(),
              "z_is_served_product": {"cells": T * R * C, "equal": eq},
              "xa": {"entries": sum(map(sum, xas)), "tile_rows": T * R, "tile_rows_with_xa": sum(int(n > 0) for r in xas for n in r),
                     "tiles_with_xa": sum(int(any(r)) for r in xas)}}
    (priv_dir / "served.json").write_text(json.dumps(served))
    Path(a.out).write_text(json.dumps(served, indent=1))
    print(json.dumps(served))
    return 0


def plus_one(words: np.ndarray) -> np.ndarray:
    """A 32-bit value's two u16 words (little-endian) plus one, mod 2^32."""
    v = (int(words[0]) | int(words[1]) << 16) + 1 & 0xFFFFFFFF
    return np.array([v & 0xFFFF, v >> 16], np.uint16)


def cmd_tamper(a) -> int:
    import shutil
    src, dst = Path(a.priv), Path(a.tampered)
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(src / "commitment.json", dst / "commitment.json")
    com = json.loads((src / "commitment.json").read_text())
    salt = bytes.fromhex(json.loads((src / "served.json").read_text())["salt"])
    priv = dict(np.load(src / "rows.npz"))
    shape = U.Shape(com["unit"]["k"], R, C)
    ports = shape.ports()
    flip = None
    if a.d_word is not None or a.xa_flip is not None:
        u = a.d_word if a.d_word is not None else a.xa_flip
        _init(str(src), salt)
        _u, z, rc, _n, xa = _serve_one(u, d_word=a.d_word is not None, xa_flip=a.xa_flip is not None)
        if a.xa_flip is not None:
            hz = U.z_words(_serve_one(u)[1])
            flip = {"xa_set_in_unit": sum(xa), "z_words_changed": int((np.asarray(U.z_words(z)) != np.asarray(hz)).sum()),
                    "receipt_is_honest": bool((np.asarray(U.receipt_words(rc)) == priv["rw"][u]).all())}
        for out, words, port in (("z", U.z_words(z), ports[-2]), ("r", U.receipt_words(rc), ports[-1])):
            priv[f"{out}w"] = priv[f"{out}w"].copy()
            priv[f"{out}w"][u] = words
            priv[f"bc{out}"] = priv[f"bc{out}"].copy()
            priv[f"bc{out}"][u] = np.frombuffer(U.commit_strings(port, priv[f"{out}w"][u:u + 1], [bytes(priv[f"s{out}"][u])])[0],
                                                 np.uint8)
    elif a.z is not None or a.z_cell is not None or a.receipt_word is not None:
        u, out, w = (a.z, "z", None) if a.z is not None else (a.z_cell, "z", 0) if a.z_cell is not None else (a.receipt_word, "r", 0)
        rows, bcs, salts, port = f"{out}w", f"bc{out}", f"s{out}", ports[-2] if out == "z" else ports[-1]
        priv[rows] = priv[rows].copy()
        if w is None:
            priv[rows][u, 0] ^= 1
        else:
            priv[rows][u, 2 * w:2 * w + 2] = plus_one(priv[rows][u, 2 * w:2 * w + 2])
        priv[bcs] = priv[bcs].copy()
        priv[bcs][u] = np.frombuffer(U.commit_strings(port, priv[rows][u:u + 1], [bytes(priv[salts][u])])[0], np.uint8)
    else:
        u = a.receipt
        _init(str(src), hashlib.sha512(b"verity/pouw/h1-prove/another-salt\0" + salt).digest())
        priv["rw"] = priv["rw"].copy()
        priv["rw"][u] = U.receipt_words(_serve_one(u)[2])
        priv["bcr"] = priv["bcr"].copy()
        priv["bcr"][u] = np.frombuffer(U.commit_strings(ports[-1], priv["rw"][u:u + 1], [bytes(priv["sr"][u])])[0], np.uint8)
    T, w = com["tiles"], com["window"]
    served = {"z_root": U.levels(U.domain(w, "z", T), _bl(priv["bcz"]))[-1][0].hex(),
              "receipt_root": U.levels(U.domain(w, "receipt", T), _bl(priv["bcr"]))[-1][0].hex(), "tiles": T, "salt": salt.hex(),
              "tampered": {"unit": u, "what": next(k for k in ("z", "z_cell", "receipt_word", "receipt", "d_word", "xa_flip")
                                                  if getattr(a, k) is not None), **({"xa_flip": flip} if flip else {})}}
    np.savez(dst / "rows.npz", **priv)
    (dst / "served.json").write_text(json.dumps(served))
    Path(a.out).write_text(json.dumps(served, indent=1))
    print(json.dumps(served))
    return 0


def cmd_open(a) -> int:
    priv_dir, out = Path(a.priv), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    com = json.loads((priv_dir / "commitment.json").read_text())
    priv = dict(np.load(priv_dir / "rows.npz"))
    units = [int(x) for x in a.units.split(",")]
    T, w = com["tiles"], com["window"]
    lz = U.levels(U.domain(w, "z", T), _bl(priv["bcz"]))
    lr = U.levels(U.domain(w, "receipt", T), _bl(priv["bcr"]))
    la = U.levels(U.domain(w, "a", com["slot"]), _bl(priv["bca"]))
    lb = U.levels(U.domain(w, "b", com["b_rows"]), _bl(priv["bcb"]))
    salt = bytes.fromhex(a.salt)
    rows = {f"a{i}": [] for i in range(R)} | {f"b{j}": [] for j in range(C)} | {"pub": [], "o0": [], "o1": []}
    salts = {p: [] for p in rows}
    codes, openings = {"a": [], "b": []}, []
    for u in units:
        c, ti, tj = tile_of(com, u)
        call = com["calls"][c]
        A, B, _roots, _call = _rows(priv, com, u)
        codes["a"].append(A)
        codes["b"].append(B)
        op = {"unit": u, "call": call["name"], "tile": [ti, tj], "rows": {}}
        for i in range(R):
            r = R * ti + i
            rows[f"a{i}"].append(U.row_words(A[i:i + 1])[0])
            salts[f"a{i}"].append(bytes(priv["sa"][r]))
            op["rows"][f"a{i}"] = {"tree": "a", "index": r, "bc": bytes(priv["bca"][r]).hex(), "path": U.path(la, r)}
        for j in range(C):
            r = call["b0"] + C * tj + j
            rows[f"b{j}"].append(U.row_words(B[j:j + 1])[0])
            salts[f"b{j}"].append(bytes(priv["sb"][r]))
            op["rows"][f"b{j}"] = {"tree": "b", "index": r, "bc": bytes(priv["bcb"][r]).hex(), "path": U.path(lb, r)}
        rows["pub"].append(U.pub_words(salt, u, roots(com)))
        salts["pub"].append(U.pub_salt(w, u))
        rows["o0"].append(priv["zw"][u])
        rows["o1"].append(priv["rw"][u])
        salts["o0"].append(bytes(priv["sz"][u]))
        salts["o1"].append(bytes(priv["sr"][u]))
        op["rows"]["o0"] = {"tree": "z", "index": u, "bc": bytes(priv["bcz"][u]).hex(), "path": U.path(lz, u)}
        op["rows"]["o1"] = {"tree": "receipt", "index": u, "bc": bytes(priv["bcr"][u]).hex(), "path": U.path(lr, u)}
        openings.append(op)
    np.savez(out / "rows.npz", units=np.array(units), a=np.stack(codes["a"]), b=np.stack(codes["b"]),
             salt=np.frombuffer(salt, np.uint8), roots=np.array([[np.frombuffer(r, np.uint8) for r in roots(com)]] * len(units)),
             **{f"w_{p}": np.stack(v) for p, v in rows.items()},
             **{f"s_{p}": np.frombuffer(b"".join(v), np.uint8).reshape(len(v), U.SALT_BYTES) for p, v in salts.items()})
    (out / "opening.json").write_text(json.dumps({"window": w, "units": openings}, indent=1))
    print(json.dumps({"opened": units}))
    return 0


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
    p = sub.add_parser("serve")
    p.add_argument("priv")
    p.add_argument("salt")
    p.add_argument("out")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p = sub.add_parser("tamper")
    p.add_argument("priv")
    p.add_argument("tampered")
    p.add_argument("out")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--z", type=int)
    g.add_argument("--receipt", type=int)
    g.add_argument("--z-cell", type=int)
    g.add_argument("--receipt-word", type=int)
    g.add_argument("--d-word", type=int)
    g.add_argument("--xa-flip", type=int)
    p = sub.add_parser("open")
    p.add_argument("priv")
    p.add_argument("salt")
    p.add_argument("units")
    p.add_argument("out")
    a = ap.parse_args(argv)
    return {"commit": cmd_commit, "serve": cmd_serve, "tamper": cmd_tamper, "open": cmd_open}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
