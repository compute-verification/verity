"""The committer and prover of one H1 window (`run.py` drives it; the challenger and the verifier never import this file).

    python window.py commit LAYER.npz WINDOW PRIV OUT.json [--k K] [--cols N] [--calls q_proj,...]
        each call's A rows and B rows (`layer.py`'s codes) as row/v2 bit rows under fresh salts, one frame-v3-sha512 tree
        per side and call: OUT.json the public commitment (each call's shape, first tile and two roots), PRIV the prover's
    python window.py serve PRIV SALT_HEX OUT.json [--workers W]
        once the salt coin is open: every tile's z (its cells' exact products) and CP-W receipt (`h1tile.reference`, the
        word reference the served kernel's receipts equal), each an output row under a fresh salt in the window's z tree
        and receipt tree
    python window.py tamper PRIV TAMPERED_PRIV OUT.json (--z U | --receipt U)
        a negative's window: PRIV's, every salt kept, with unit U's z committed with one word flipped, or its receipt
        from another salt; the rows a prover of it holds are PRIV's, so its proof is PRIV's
    python window.py open PRIV SALT_HEX UNIT,... OUT_DIR
        the drawn units' rows (their public row's too) and salts for staging (OUT_DIR/rows.npz, the prover's), and their
        opening (OUT_DIR/opening.json: each committed row's b || c and its path to the window's registered root)

A tile is (call, ti, tj): A rows [R ti, R (ti + 1)) and B rows [C tj, C (tj + 1)) of the call; tiles are numbered call by
call, row block major, and a tile's number is its position, the unit's ``position`` input. A call is one seed group, its
roots (root(A), root(B)) the unit's ``roots`` input. ``--k`` truncates every row to its first K codes (a smoke run's).
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
        for ti in range(call["m"] // R):
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
    k = A.shape[1]
    shape = U.Shape(k, R, C)
    port = shape.ports()[0]
    m = A.shape[0] - A.shape[0] % R
    A = A[:m]
    out_calls, arrays, tile0 = [], {}, 0
    for name in calls:
        B = lay[f"b_{name}"][:, :k]
        if a.cols:
            B = B[:a.cols]
        n = B.shape[0] - B.shape[0] % C
        B = B[:n]
        sa, sb = _salts(m), _salts(n)
        bca = U.commit_strings(port, U.row_words(A), _bl(sa))
        bcb = U.commit_strings(port, U.row_words(B), _bl(sb))
        da, db = U.domain(a.window, f"a/{name}", m), U.domain(a.window, f"b/{name}", n)
        ra, rb = U.levels(da, bca)[-1][0], U.levels(db, bcb)[-1][0]
        t = (m // R) * (n // C)
        out_calls.append({"name": name, "m": m, "k": k, "n": n, "tile0": tile0, "tiles": t, "root_a": ra.hex(), "root_b": rb.hex()})
        tile0 += t
        arrays.update({f"{name}_b": B, f"{name}_sa": sa, f"{name}_sb": sb,
                       f"{name}_bca": np.frombuffer(b"".join(bca), np.uint8).reshape(m, 128),
                       f"{name}_bcb": np.frombuffer(b"".join(bcb), np.uint8).reshape(n, 128)})
    np.savez(priv / "rows.npz", a=A, **arrays)
    com = {"schema": "verity/pouw/h1-prove/commitment/v0", "window": a.window, "unit": {"definition": "PouwH1Tile_v1", "k": k, "r": R, "c": C,
           "exact": "GemmAccumulatorS8_v2"}, "calls": out_calls, "tiles": tile0, "tokens": [int(x) for x in lay["tokens"]]}
    (priv / "commitment.json").write_text(json.dumps(com, indent=1))
    Path(a.out).write_text(json.dumps({x: com[x] for x in ("schema", "window", "unit", "calls", "tiles")}, indent=1))
    print(json.dumps({"window": a.window, "k": k, "m": m, "calls": {c["name"]: c["n"] for c in out_calls}, "tiles": tile0}))
    return 0


def _rows(priv: dict, com: dict, u: int) -> tuple[np.ndarray, np.ndarray, tuple[bytes, bytes], dict]:
    c, ti, tj = tile_of(com, u)
    call = com["calls"][c]
    return (priv["a"][R * ti:R * (ti + 1)], priv[f"{call['name']}_b"][C * tj:C * (tj + 1)],
            (bytes.fromhex(call["root_a"]), bytes.fromhex(call["root_b"])), call)


_W: dict = {}


def _init(priv_dir: str, salt: bytes) -> None:
    _W["priv"] = dict(np.load(Path(priv_dir) / "rows.npz"))
    _W["com"] = json.loads((Path(priv_dir) / "commitment.json").read_text())
    _W["salt"] = salt


def _serve_one(u: int) -> tuple[int, list[int], list[int]]:
    from verity.core.protocols.pouw.circuit import h1tile as H
    a, b, roots, _call = _rows(_W["priv"], _W["com"], u)
    z, rc = H.reference([[int(x) & 0xFF for x in r] for r in a], [[int(x) & 0xFF for x in r] for r in b], _W["salt"], u, roots)
    return u, z, rc


def cmd_serve(a) -> int:
    priv_dir = Path(a.priv)
    com = json.loads((priv_dir / "commitment.json").read_text())
    salt = bytes.fromhex(a.salt)
    T = com["tiles"]
    shape = U.Shape(com["unit"]["k"], R, C)
    zs, rcs = [None] * T, [None] * T
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(str(priv_dir), salt)) as ex:
        for u, z, rc in ex.map(_serve_one, range(T), chunksize=64):
            zs[u], rcs[u] = z, rc
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
    served = {"z_root": rz.hex(), "receipt_root": rr.hex(), "tiles": T, "salt": salt.hex()}
    (priv_dir / "served.json").write_text(json.dumps(served))
    Path(a.out).write_text(json.dumps(served, indent=1))
    print(json.dumps(served))
    return 0


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
    if a.z is not None:
        u = a.z
        priv["zw"] = priv["zw"].copy()
        priv["zw"][u, 0] ^= 1
        priv["bcz"] = priv["bcz"].copy()
        priv["bcz"][u] = np.frombuffer(U.commit_strings(ports[-2], priv["zw"][u:u + 1], [bytes(priv["sz"][u])])[0], np.uint8)
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
              "tampered": {"unit": u, "what": "z" if a.z is not None else "receipt"}}
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
    side_levels = {}
    salt = bytes.fromhex(a.salt)
    rows = {f"a{i}": [] for i in range(R)} | {f"b{j}": [] for j in range(C)} | {"pub": [], "o0": [], "o1": []}
    salts = {p: [] for p in rows}
    codes, openings = {"a": [], "b": []}, []
    for u in units:
        c, ti, tj = tile_of(com, u)
        call = com["calls"][c]
        nm = call["name"]
        for s, n in (("a", call["m"]), ("b", call["n"])):
            if (nm, s) not in side_levels:
                side_levels[nm, s] = U.levels(U.domain(w, f"{s}/{nm}", n), _bl(priv[f"{nm}_bc{s}"]))
        A, B = priv["a"][R * ti:R * (ti + 1)], priv[f"{nm}_b"][C * tj:C * (tj + 1)]
        codes["a"].append(A)
        codes["b"].append(B)
        op = {"unit": u, "call": nm, "tile": [ti, tj], "rows": {}}
        for i in range(R):
            r = R * ti + i
            rows[f"a{i}"].append(U.row_words(A[i:i + 1])[0])
            salts[f"a{i}"].append(bytes(priv[f"{nm}_sa"][r]))
            op["rows"][f"a{i}"] = {"tree": f"a/{nm}", "index": r, "bc": bytes(priv[f"{nm}_bca"][r]).hex(),
                                   "path": U.path(side_levels[nm, "a"], r)}
        for j in range(C):
            r = C * tj + j
            rows[f"b{j}"].append(U.row_words(B[j:j + 1])[0])
            salts[f"b{j}"].append(bytes(priv[f"{nm}_sb"][r]))
            op["rows"][f"b{j}"] = {"tree": f"b/{nm}", "index": r, "bc": bytes(priv[f"{nm}_bcb"][r]).hex(),
                                   "path": U.path(side_levels[nm, "b"], r)}
        rows["pub"].append(U.pub_words(salt, u, (bytes.fromhex(call["root_a"]), bytes.fromhex(call["root_b"]))))
        salts["pub"].append(U.pub_salt(w, u))
        rows["o0"].append(priv["zw"][u])
        rows["o1"].append(priv["rw"][u])
        salts["o0"].append(bytes(priv["sz"][u]))
        salts["o1"].append(bytes(priv["sr"][u]))
        op["rows"]["o0"] = {"tree": "z", "index": u, "bc": bytes(priv["bcz"][u]).hex(), "path": U.path(lz, u)}
        op["rows"]["o1"] = {"tree": "receipt", "index": u, "bc": bytes(priv["bcr"][u]).hex(), "path": U.path(lr, u)}
        openings.append(op)
    roots = [[bytes.fromhex(com["calls"][tile_of(com, u)[0]][f"root_{s}"]) for s in "ab"] for u in units]
    np.savez(out / "rows.npz", units=np.array(units), a=np.stack(codes["a"]), b=np.stack(codes["b"]),
             salt=np.frombuffer(salt, np.uint8), roots=np.array([[np.frombuffer(r, np.uint8) for r in rr] for rr in roots]),
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
    p = sub.add_parser("open")
    p.add_argument("priv")
    p.add_argument("salt")
    p.add_argument("units")
    p.add_argument("out")
    a = ap.parse_args(argv)
    return {"commit": cmd_commit, "serve": cmd_serve, "tamper": cmd_tamper, "open": cmd_open}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
