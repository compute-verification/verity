"""Regenerate the ref.py (``ligerito-ref/v2``) fixtures that ``refpcs.rs`` pins, from ``backends/direct/ligerito/ref.py``.

    PYTHONPATH=<worktree> python backends/ligerito-verify/fixtures/gen_ref_fixtures.py [--redteam DIR]

Needs numpy + blake3 (no torch).  Writes, under ``backends/ligerito-verify/fixtures/``:

* ``ref/fixtures.json``: ``python -m backends.direct.ligerito.ref``'s stdout (4 honest toy proofs, byte-identical);
* ``ref/large_n16L3x3q64.json``: one set-B-shaped proof (tall 12288 x 16, three levels, 64 queries; ``f`` dropped);
* ``ref/negatives.json``: 20 declarative tampers per ``fixtures.json`` entry, each with ref.py's verdict and reason;
* ``forged/*.proof.neg``: must-reject forgeries (``batch --dir forged`` rejects every entry), ``forged/honest.proof``:
  the same statements proved honestly.  The red team's v1 forgeries (``--redteam``: ``weakfs.ref.json``,
  ``overflow.ref.json`` from ``lanes/red-team-ligerito/fixtures``) are copied verbatim and also relabelled v2; the
  weak-FS forgery (point chosen after the sumcheck challenges, F10) and the int64-wrap forgery (lifted word, F11) are
  re-run against the v2 transcript; plus statement substitutions and non-canonical encodings of honest proofs.

Every entry's ``expect`` is ref.py v2's own verdict; the script asserts that every forgery is rejected by ref.py.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
from pathlib import Path

import numpy as np

from backends.direct.ligerito import ref as R

P, EXT = R.P, R.EXT
K64 = (1 << 64) % P
HERE = Path(__file__).resolve().parent


def run_verify(fx: dict, proof_doc: dict) -> tuple[bool, str]:
    """ref.py's verdict on a fixture statement with the given proof JSON (JSON round trip, as the Rust side reads it)."""
    try:
        doc = json.loads(json.dumps(proof_doc))
        pf = R.proof_from_dict(doc)
        z = np.array(fx["z"], dtype=R.I64)
        if not R._words_canonical(fx["z"]) or ("value" in fx and not R._words_canonical(fx["value"])):
            return False, R.NONCANON
        w = R.point_tensor(z, pf.dims.radices)
        value = np.array(fx["value"], dtype=R.I64) if "value" in fx else pf.value
        return R.verify(bytes.fromhex(fx["commitment"]), w, value, pf, R.FSCoins(fx["seed"].encode()))
    except ValueError as e:
        if str(e).startswith(("non-canonical", "format")):
            return False, str(e)
        return False, f"exception: ValueError: {e}"
    except Exception as e:                                                   # noqa: BLE001 - any raise is a reject
        return False, f"exception: {type(e).__name__}: {e}"


# ------------------------------------------------------------------------------------------------ tampers ---------

def apply_ops(doc, ops):
    for op in ops:
        *path, last = op["path"]
        node = doc
        for p in path:
            node = node[p]
        kind, v = op["op"], op.get("v")
        if kind in ("add", "addraw", "xor"):
            x = node[last]
            node[last] = (x + v) % P if kind == "add" else (x + v if kind == "addraw" else x ^ v)
        elif kind == "set":
            node[last] = v
        elif kind == "pop":
            node[last].pop()
        elif kind == "push":
            node[last].append(v)
        elif kind == "dup0":
            node[last].append(node[last][0])
        else:
            raise ValueError(kind)


def tampers_for(i: int, name: str, fx: dict) -> list[dict]:
    pr = fx["proof"]
    L = len(pr["roots"])
    last = L - 1
    row_l = [1, 3] if L > 1 else [1]                                   # an entry of a level-L row (ext for L >= 2)
    other_root = R.H(b"other root " + name.encode()).hex()
    kinds = {
        "value": [{"path": ["value", 0], "op": "add", "v": 1}],
        "root0": [{"path": ["roots", 0], "op": "set", "v": "00" * 32}],
        "rootL": [{"path": ["roots", last], "op": "set", "v": other_root}],
        "row": [{"path": ["levels", 0, "opened_rows", 2, 0], "op": "add", "v": 1}],
        "rowL": [{"path": ["levels", last, "opened_rows", 0] + row_l, "op": "add", "v": 1}],
        "sib": [{"path": ["levels", 0, "siblings", 0], "op": "set", "v": "ff" * 32}],
        "sib_drop": [{"path": ["levels", 0, "siblings"], "op": "pop"}],
        "sib_extra": [{"path": ["levels", 0, "siblings"], "op": "dup0"}],
        "sc_c0": [{"path": ["levels", 0, "sumcheck", 0, 0, 0], "op": "add", "v": 1}],
        "sc_c2": [{"path": ["levels", last, "sumcheck", -1, 2, 1], "op": "add", "v": 1}],
        "sc_keepsum": [{"path": ["levels", 0, "sumcheck", 0, 1, 0], "op": "add", "v": 1},
                       {"path": ["levels", 0, "sumcheck", 0, 2, 0], "op": "add", "v": P - 1}],
        "final": [{"path": ["final", 3, 1], "op": "add", "v": 1}],
        "idx": [{"path": ["levels", last, "opened_idx", 0], "op": "xor", "v": 1}],
        "idx_dup": [{"path": ["levels", last, "opened_idx", 1], "op": "set", "v": pr["levels"][last]["opened_idx"][0]}],
        "queries": [{"path": ["dims", "queries", last], "op": "add", "v": P - 1},
                    {"path": ["levels", last, "opened_idx"], "op": "pop"}, {"path": ["levels", last, "opened_rows"], "op": "pop"}],
        "rows_short": [{"path": ["levels", 0, "opened_rows"], "op": "pop"}],
        "ext_in_base": [{"path": ["levels", 0, "opened_rows", 0, 0], "op": "set",
                         "v": [pr["levels"][0]["opened_rows"][0][0], 0, 0, 0, 0, 0]}],
        "final_shape": [{"path": ["final"], "op": "push", "v": [0] * EXT}],
        "nlog": [{"path": ["dims", "n_log2", 0], "op": "add", "v": 1}],
        "noncanon": [{"path": ["levels", 0, "opened_rows", 2, 0], "op": "addraw", "v": P}],
    }
    out = []
    for k, ops in kinds.items():
        doc = copy.deepcopy(pr)
        apply_ops(doc, ops)
        ok, why = run_verify(fx, doc)
        out.append({"name": f"{name}/{k}", "base": i, "ops": ops, "expect": {"ok": ok, "reason": why}})
    return out


# ------------------------------------------------------------------------------------------------ forgeries -------

def rand_ext(rng, k=None):
    return rng.integers(0, P, size=(EXT,) if k is None else (k, EXT), dtype=np.int64)


def einv(a):
    return R.epow(a, P ** EXT - 2)


def eqv(z, r):
    one = R.ext_one()
    return R.eadd(R.emul(R.esub(one, z), R.esub(one, r)), R.emul(z, r))


def forge_weakfs_v2(st: R.ProverState, value_star, rng, seed: bytes):
    """red-team-ligerito's weak-FS forgery (redteam_ref_weakfs.py) against the v2 transcript: the prover must absorb a
    statement before the first coin, so it commits to a guessed point z0, runs level-1 messages that merely sum to the
    FALSE value, then (as in v1) solves the last column coordinate of z after the challenges.  The proof claims z != z0."""
    dims = st.dims
    coins = R.FSCoins(seed)
    z0 = rand_ext(rng, len(dims.radices))
    R.absorb_statement(coins, dims, R.point_tensor(z0, dims.radices), value_star)
    M, U, tree = st.M1, st.U1, st.tree1
    roots, levels = [tree[-1][0]], []
    coins.absorb(b"root", roots[0])
    claim, msgs, r = value_star, [], []
    for _ in range(dims.wide_log2[0]):
        c1, c2 = rand_ext(rng), rand_ext(rng)
        c0 = R.escale(R.esub(R.esub(claim, c1), c2), R.INV2)
        msg = np.stack([c0, c1, c2])
        coins.absorb(b"sumcheck", R.row_bytes(msg))
        rj = coins.challenge(b"r", 1)[0]
        msgs.append(msg)
        r.append(rj)
        claim = R.eadd(c0, R.emul(rj, R.eadd(c1, R.emul(c2, rj))))
    r = np.stack(r)
    rbar = R.expand(R.eq_tensor(r))
    v = R.mat_vec(M, rbar)
    row_rad, k = dims.row_radices(0), dims.wide_log2[0]
    z_row = z0[:len(row_rad)].copy()
    A = R.esum(R.emul(R.expand(R.point_tensor(z_row, row_rad)), v))
    z_col = z0[len(row_rad):].copy()
    prod = R.ext_one()
    for j in range(k - 1):
        prod = R.emul(prod, eqv(z_col[j], r[j]))
    t = R.emul(claim, einv(R.emul(A, prod)))
    rl, one = r[k - 1], R.ext_one()
    z_col[k - 1] = R.emul(R.esub(t, R.esub(one, rl)), einv(R.esub(R.escale(rl, 2), one)))
    z = np.concatenate([z_row, z_col])
    w = R.point_tensor(z, dims.radices)
    ra, ca = R.split(w, dims.tall[0])
    terms, total = [(R.ext_one(), ra, ca)], claim
    levels.append(R.LevelProof(np.stack(msgs), [], [], []))
    carried = [(R.emul(g, R.tensor_eval_binary(b, r)), a) for (g, a, b) in terms]
    prev_U, prev_tree = U, tree
    for i in range(1, dims.L):
        M = R._reshape(v, dims.tall[i], dims.wide(i))
        U = R.encode(M, dims.n(i))
        tree = R.merkle_tree(U)
        roots.append(tree[-1][0])
        coins.absorb(b"root", roots[-1])
        prev_idx = coins.indices(b"queries", dims.n(i - 1), dims.queries[i - 1])
        prev_rows = [prev_U[s] for s in prev_idx]
        sibs = R.merkle_multi_open(prev_tree, prev_idx)
        R._absorb_opening(coins, prev_rows, sibs)
        levels[-1].opened_idx, levels[-1].opened_rows, levels[-1].siblings = prev_idx, prev_rows, sibs
        terms, total = R._batch_terms(coins, dims, i, carried, total, prev_rows, prev_idx, rbar)
        msgs_i, r, total = R._sumcheck_prove(coins, M, terms, dims.wide_log2[i])
        levels.append(R.LevelProof(msgs_i, [], [], []))
        rbar = R.expand(R.eq_tensor(r))
        v = R.mat_vec(M, rbar)
        carried = [(R.emul(g, R.tensor_eval_binary(b, r)), a) for (g, a, b) in terms]
        prev_U, prev_tree = U, tree
    coins.absorb(b"final", R.row_bytes(v))
    idx = coins.indices(b"queries", dims.n(dims.L - 1), dims.queries[dims.L - 1])
    levels[-1].opened_idx = idx
    levels[-1].opened_rows = [prev_U[s] for s in idx]
    levels[-1].siblings = R.merkle_multi_open(prev_tree, idx)
    return z, R.Proof(dims, roots, levels, v, value_star)


def lift(x: int, lo: int, hi: int) -> int:
    t = -(-(lo - x) // P)
    y = x + t * P
    assert lo <= y < hi
    return y


def entry(name, seed, z, root, value, proof_doc, f=None) -> dict:
    fx = {"name": name, "seed": seed.decode(), "z": np.asarray(z).tolist(), "commitment": root.hex(),
          "value": np.asarray(value).tolist(), "proof": proof_doc}
    if f is not None:
        fx["f"] = np.asarray(f).tolist()
    return fx


def with_verdict(fx: dict) -> dict:
    ok, why = run_verify(fx, fx["proof"])
    fx["expect"] = {"ok": ok, "reason": why}
    return fx


def forged_sets(rng, redteam: Path | None) -> dict[str, list[dict]]:
    seed = b"fixture"
    sets: dict[str, list[dict]] = {"honest.proof": []}
    weak, over, stmt, enc = [], [], [], []
    for L, odd3, log_n in ((1, False, 10), (2, True, 10), (3, False, 12)):
        tag = f"L{L}{'-radix3' if odd3 else ''}"
        dims = R.Dims.toy(log_n, L, queries=8, odd3=odd3)
        f = rng.integers(0, P, size=dims.N, dtype=np.int64)
        st = R.commit(f, dims)
        # honest proof of a random point
        z = rand_ext(rng, len(dims.radices))
        w = R.point_tensor(z, dims.radices)
        pr = R.prove_eval(st, w, R.FSCoins(seed))
        doc = R.proof_to_dict(pr)
        honest = entry(f"honest/{tag}", seed, z, st.root, pr.value, doc)
        sets["honest.proof"].append(with_verdict(honest))
        # F10: z chosen after the sumcheck challenges (v2 transcript)
        v_star = rand_ext(rng)
        zf, pf = forge_weakfs_v2(st, v_star, rng, seed)
        weak.append(with_verdict(entry(f"weakfs-v2/{tag}", seed, zf, st.root, v_star, R.proof_to_dict(pf))))
        # F11: honest messages for v, one lifted c0 word so the int64 round check would see v - K e0
        v_k = pr.value.copy()
        v_k[0] = (v_k[0] - K64) % P
        d = copy.deepcopy(doc)
        d["levels"][0]["sumcheck"][0][0][0] = lift(d["levels"][0]["sumcheck"][0][0][0], 1 << 62, (1 << 63) - P)
        d["value"] = v_k.tolist()
        over.append(with_verdict(entry(f"overflow-v2/{tag}", seed, z, st.root, v_k, d)))
        # statement substitutions on the honest proof: another point, another value (statement and proof), fewer queries
        z2 = z.copy()
        z2[-1, 0] = (z2[-1, 0] + 1) % P
        stmt.append(with_verdict(entry(f"stmt-point/{tag}", seed, z2, st.root, pr.value, doc)))
        v2 = pr.value.copy()
        v2[0] = (v2[0] + 1) % P
        d = copy.deepcopy(doc)
        d["value"] = v2.tolist()
        stmt.append(with_verdict(entry(f"stmt-value/{tag}", seed, z, st.root, v2, d)))
        d = copy.deepcopy(doc)
        d["dims"]["queries"][0] -= 1
        stmt.append(with_verdict(entry(f"stmt-queries/{tag}", seed, z, st.root, pr.value, d)))
        # non-canonical encodings of the honest proof (same values mod p)
        for what, path in (("final", ["final", 0, 0]), ("sumcheck", ["levels", 0, "sumcheck", 0, 1, 2]),
                           ("row", ["levels", 0, "opened_rows", 0, 0]), ("value", ["value", 1])):
            d = copy.deepcopy(doc)
            node = d
            for p in path[:-1]:
                node = node[p]
            node[path[-1]] += P
            fx = entry(f"noncanon-lift-{what}/{tag}", seed, z, st.root, pr.value, d)
            if what == "value":
                fx["value"][1] += P
            enc.append(with_verdict(fx))
            d = copy.deepcopy(doc)
            node = d
            for p in path[:-1]:
                node = node[p]
            node[path[-1]] = float(node[path[-1]])                       # JSON "123.0": not an integer literal
            enc.append(with_verdict(entry(f"noncanon-float-{what}/{tag}", seed, z, st.root, pr.value, d)))
        d = copy.deepcopy(doc)
        d["final"][0][0] += 1 << 64                                            # beyond u64 / int64
        enc.append(with_verdict(entry(f"noncanon-2^64/{tag}", seed, z, st.root, pr.value, d)))
    sets["weakfs_v2.proof.neg"] = weak
    sets["overflow_v2.proof.neg"] = over
    sets["statement_v2.proof.neg"] = stmt
    sets["noncanonical_v2.proof.neg"] = enc
    if redteam is not None:
        for name in ("weakfs", "overflow"):
            src = json.loads((redteam / f"{name}.ref.json").read_text())
            v1 = [with_verdict(copy.deepcopy(fx)) for fx in src]
            sets[f"redteam_{name}_v1.proof.neg"] = v1
            relabel = []
            for fx in src:
                fx = copy.deepcopy(fx)
                fx["name"] = fx["name"] + "/relabelled-v2"
                fx["proof"]["format"] = R.FORMAT
                fx["value"] = fx["proof"]["value"]
                relabel.append(with_verdict(fx))
            sets[f"redteam_{name}_relabelled_v2.proof.neg"] = relabel
    for name, fxs in sets.items():
        want = name.endswith(".proof")
        bad = [fx["name"] for fx in fxs if fx["expect"]["ok"] != want]
        assert not bad, f"{name}: ref.py v2 gives the wrong verdict on {bad}"
    return sets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--redteam", type=Path, default=Path.home() / ".research/notes/lanes/red-team-ligerito/fixtures")
    a = ap.parse_args()
    # 1. the module's own fixtures, byte-identical to `python -m backends.direct.ligerito.ref`
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        R.main()
    (HERE / "ref/fixtures.json").write_text(buf.getvalue())
    fxs = json.loads(buf.getvalue())
    assert all(fx["expect"]["ok"] for fx in fxs)
    # 2. set-B-shaped large fixture
    rng = np.random.default_rng(16)
    dims = R.Dims((12288, 768, 48), (4, 4, 4), (16, 12, 8), (64, 64, 64))
    f = rng.integers(0, P, dims.N, dtype=np.int64)
    z = rng.integers(0, P, (len(dims.radices), EXT), dtype=np.int64)
    big = R.fixture(f, dims, z)
    del big["f"]
    big["name"] = "large n16L3x3q64"
    assert big["expect"]["ok"]
    (HERE / "ref/large_n16L3x3q64.json").write_text(json.dumps([big]))
    # 3. tampers
    names = ["n8L1", "n10L2", "n12L3", "n10L2x3"]
    tampers = [t for i, fx in enumerate(fxs) for t in tampers_for(i, names[i], fx)]
    (HERE / "ref/negatives.json").write_text(json.dumps(tampers))
    # 4. forgeries
    out = HERE / "forged"
    out.mkdir(exist_ok=True)
    redteam = a.redteam if (a.redteam / "weakfs.ref.json").exists() else None
    sets = forged_sets(np.random.default_rng(2100), redteam)
    for name, fxs_ in sets.items():
        (out / name).write_text(json.dumps(fxs_))
    print(f"fixtures.json: {len(fxs)}; negatives.json: {len(tampers)} ({sum(not t['expect']['ok'] for t in tampers)} rejected by ref.py); "
          f"forged/: " + ", ".join(f"{k} {len(v)}" for k, v in sets.items()) + f" (red-team v1 files {'copied' if redteam else 'MISSING'})")


if __name__ == "__main__":
    main()
