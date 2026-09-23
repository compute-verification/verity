"""red-team-ligerito: non-canonical words + int64 overflow forge a FIXED, transcript-bound evaluation claim.

``ref.verify`` never range-checks the sumcheck coefficients.  The transcript hashes them reduced mod p (``row_bytes``), so a
lift ``c0 + t p`` leaves every challenge unchanged, but the round check ``escale(c0, 2)`` computes ``2 c0`` in int64: for
``c0 + t p`` in [2^62, 2^63 - p) it wraps and the verifier tests ``g(0) + g(1) - K == claim`` with ``K = 2^64 mod p``.  The
claim update ``eadd(c0, ...)`` does not wrap, so every later round is the honest one.  An honest proof of ``v`` with one
lifted coordinate verifies as a proof of ``v - K e_0``, with ``(w, value)`` absorbed before the first coin on both sides
(so it is independent of the missing statement binding).  JSON (``proof_to_dict``/``proof_from_dict``) carries the lift.

lane ligerito-pcs-fast's ``claims.verify_column_rounds`` has the same arithmetic (``ext.add(g0, g1)`` on raw int64): an
in-memory ``Proof`` with ``g0 + g1 >= 2^63`` passes the round check for ``claim - K``.  Its byte format carries u32 words,
which cannot reach the wrap, so only in-process callers are exposed.

    python -m backends.direct.ligerito.redteam_ref_overflow [--pcs-fast ~/projects/verity-main-wt/ligerito-pcs-fast]
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from backends.direct.ligerito import ref as R

P, EXT = R.P, R.EXT
K = (1 << 64) % P


def lift(x: int, lo: int, hi: int) -> int:
    t = -(-(lo - x) // P)
    y = x + t * P
    assert lo <= y < hi, (x, y)
    return y


def ref_forgery(rng, seed: bytes, bind: bool = True):
    out = []
    for L, odd3 in ((1, False), (2, True), (3, False)):
        dims = R.Dims.toy(12 if L == 3 else 10, L, queries=8, odd3=odd3)
        f = rng.integers(0, P, size=dims.N, dtype=np.int64)
        st = R.commit(f, dims)
        z = rng.integers(0, P, size=(len(dims.radices), EXT), dtype=np.int64)
        w = R.point_tensor(z, dims.radices)
        v = R.tensor_eval(f, w)
        v_star = v.copy()
        v_star[0] = (v_star[0] - K) % P
        stmt = b"".join(R.row_bytes(t) for t in w) + R.row_bytes(v_star)
        cp = R.FSCoins(seed)
        if bind:
            cp.absorb(b"stmt", stmt)
        pf = R.prove_eval(st, w, cp)                       # honest messages for the TRUE value
        msgs = np.array(pf.levels[0].sumcheck, dtype=np.int64)
        msgs[0, 0, 0] = lift(int(msgs[0, 0, 0]), 1 << 62, (1 << 63) - P)
        pf.levels[0].sumcheck = msgs
        pf.value = v_star
        d = R.proof_to_dict(pf)
        pf = R.proof_from_dict(json.loads(json.dumps(d)))  # through the wire format
        cv = R.FSCoins(seed)
        if bind:
            cv.absorb(b"stmt", stmt)
        ok, why = R.verify(st.root, w, v_star, pf, cv)
        fx = {"name": f"redteam-overflow/L{L}{'-radix3' if odd3 else ''}", "seed": seed.decode(), "f": f.tolist(),
              "z": z.tolist(), "commitment": st.root.hex(), "proof": d,
              "expect": {"ok": False, "reason": "forgery: non-canonical sumcheck word (int64 wrap shifts the round check by K)"}}
        out.append((dims, ok, why, not np.array_equal(v_star, v), fx))
    return out


def load_pcs_fast_claims(root: Path):
    proto = root / "backends/direct/ligerito/proto"
    spec = importlib.util.spec_from_file_location("pcsfast_proto", proto / "__init__.py", submodule_search_locations=[str(proto)])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules["pcsfast_proto"] = pkg
    spec.loader.exec_module(pkg)
    return importlib.import_module("pcsfast_proto.claims")


def pcs_fast_rounds(rng, cl, k: int = 6):
    """Honest column-round messages (values at 0, 1, 2) for random degree-2 round polynomials; lift round 0's g0, g1."""
    ext = cl.ext
    claim0 = rng.integers(0, P, size=EXT, dtype=np.int64)
    claim, msgs, rs = claim0, [], []
    for _ in range(k):
        g0 = rng.integers(0, P, size=EXT, dtype=np.int64)
        g1 = ext.sub(claim, g0)
        g2 = rng.integers(0, P, size=EXT, dtype=np.int64)
        r = rng.integers(0, P, size=EXT, dtype=np.int64)
        msgs.append(np.stack([g0, g1, g2]))
        rs.append(r)
        claim = cl.interpolate_deg2(g0, g1, g2, r)
    ok_h, run_h = cl.verify_column_rounds(claim0, msgs, rs)
    lifted = [m.copy() for m in msgs]
    lifted[0][0, 0] = lift(int(msgs[0][0, 0]), (1 << 63) - (1 << 61), (1 << 63) - P)
    lifted[0][1, 0] = lift(int(msgs[0][1, 0]), (1 << 61) + (1 << 60), 1 << 62)
    assert int(lifted[0][0, 0]) + int(lifted[0][1, 0]) >= 1 << 63
    same_transcript = all(np.array_equal(a % P, b) for a, b in zip(lifted, msgs))
    claim_star = claim0.copy()
    claim_star[0] = (claim_star[0] - K) % P
    ok_f, run_f = cl.verify_column_rounds(claim_star, lifted, rs)
    return ok_h, ok_f, np.array_equal(run_f, run_h), same_transcript


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcs-fast", default=str(Path.home() / "projects/verity-main-wt/ligerito-pcs-fast"))
    ap.add_argument("--dump", help="write unbound-transcript variants (ref.py's own transcript) as ref-fixtures JSON")
    a = ap.parse_args()
    rng = np.random.default_rng(1905)
    print(f"K = 2^64 mod p = {K}")
    for dims, ok, why, false, _ in ref_forgery(rng, b"ligerito-ref/v1"):
        print(f"ref.verify  L={dims.L} tall={dims.tall}: statement (w, v - K e0) bound before coin 1, one lifted c0 "
              f"word -> {ok} ({why}); claim false: {false}")
    if a.dump:
        res = ref_forgery(rng, b"ligerito-ref/v1", bind=False)
        assert all(ok for _, ok, _, _, _ in res)
        with open(a.dump, "w") as fh:
            json.dump([fx for *_, fx in res], fh)
        print(f"wrote {len(res)} fixtures (ref.py accepts all) to {a.dump} (hash: {R.HASH_NAME})")
    root = Path(a.pcs_fast)
    if (root / "backends/direct/ligerito/proto/claims.py").exists():
        cl = load_pcs_fast_claims(root)
        ok_h, ok_f, same_run, same_tr = pcs_fast_rounds(rng, cl)
        print(f"pcs-fast claims.verify_column_rounds: honest -> {ok_h}; claim - K e0 with lifted (g0, g1) -> {ok_f}; "
              f"running claim identical: {same_run}; canonical transcript bytes identical: {same_tr}")
    else:
        print(f"(pcs-fast worktree not found at {root}; skipped)")


if __name__ == "__main__":
    main()
