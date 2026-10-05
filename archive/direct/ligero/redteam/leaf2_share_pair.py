"""red-team-leaf-2: crafted negatives for the Rust verifier of row sharing (``ligero-verify verify --system-h``), which had
none committed (the lane's negatives run through the Python verifier on the pod only).  Builds honest fp8-ada tile pairs
on the CPU (2x2 tile, 4 VUs, l = 256; ``--soundness-bits 100`` because l = 256 lands under 2^-128), writes each case as a
``system.bin`` / ``system_h.bin`` / ``X.stmt`` / ``X.proof`` / ``X.hproof`` [``X.coins`` / ``X.hcoins``] set, and runs the
Rust binary on it.  Every case but the honest ones must be REFUSED.

    PYTHONPATH=<tree>:<tree>/packages/verity/src:<tree>/backends/numerical/python ~/projects/verity/.venv/bin/python \
        leaf2_share_pair.py --out /tmp/rt2/out/share-pair --bin <ligero-verify of the same tree>

Exit status 0 iff every expectation holds; ``--out/verdicts.json`` has every verdict."""
from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--mode", default="fiat-shamir", choices=("fiat-shamir", "interactive"))
    args = ap.parse_args()
    from archive.direct.ligero import protocol
    from archive.direct.ligero.compile import P
    from archive.direct.ligero.relations import relation
    from archive.direct.ligero.relchain import SharedHashedRunner, tile_instances
    from archive.direct.ligero.serialize import _statement_of, proof_bytes, statement_bytes, system_bytes

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rel = relation("fp8-ada")
    R = SharedHashedRunner(rel, "cpu", -128.0, zk=True, mode=args.mode)
    data, x_rows, w_cols = tile_instances(rel, 2, 2)
    R.commit_vus(data, tile=(2, 2), x_rows=x_rows, w_cols=w_cols)
    (out / "system.bin").write_bytes(system_bytes(R.sys))
    (out / "system_h.bin").write_bytes(system_bytes(R.sys_h))
    inter = R.interactive

    def prove(**kw):
        c, ch = (protocol.Coins.sample(), protocol.Coins.sample()) if inter else (None, None)
        pf, pubs, lay = R.prove_vus(data[:4], check=False, vu_ids=range(4), coins=c, coins_h=ch, **kw)
        return pf, pubs, lay, c, ch

    def write(name, pf, pubs, lay, c=None, ch=None, st=None, g=None, h=None):
        d = out / name
        d.mkdir(exist_ok=True)
        st = st if st is not None else _statement_of(R, pubs, lay, R.cfg(lay.l), 1)
        (d / "x.stmt").write_bytes(statement_bytes(st) if not isinstance(st, bytes) else st)
        (d / "x.proof").write_bytes(g if g is not None else proof_bytes(pf.g))
        (d / "x.hproof").write_bytes(h if h is not None else proof_bytes(pf.h))
        if c is not None:
            (d / "x.coins").write_bytes(c.r1 + c.s1 + c.r2 + c.s2)
        if ch is not None:
            (d / "x.hcoins").write_bytes(ch.r1 + ch.s1 + ch.r2 + ch.s2)
        return d

    def rust(d, sys_g="system.bin", sys_h="system_h.bin", extra=()):
        cmd = [args.bin, "verify", "--system", str(out / sys_g), "--system-h", str(out / sys_h), "--statement", str(d / "x.stmt"),
               "--proof", str(d / "x.proof"), "--proof-h", str(d / "x.hproof"), "--soundness-bits", "100", *extra]
        if (d / "x.coins").is_file() and "--no-coins" not in extra:
            cmd += ["--coins", str(d / "x.coins")]
        if (d / "x.hcoins").is_file() and "--no-coins" not in extra:
            cmd += ["--coins-h", str(d / "x.hcoins")]
        cmd = [x for x in cmd if x != "--no-coins"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        try:
            return json.loads(r.stdout.strip().splitlines()[-1])
        except Exception:
            return {"accepted": False, "reason": f"(crash) rc={r.returncode} {r.stderr.strip()[-300:]}"}

    cases: list[tuple[str, bool, dict]] = []

    def case(name, want_accept, js):
        cases.append((name, want_accept, js))
        ok = js["accepted"] == want_accept
        print(f"{'OK ' if ok else 'BAD'} want={'accept' if want_accept else 'reject'} got={'accept' if js['accepted'] else 'reject'} "
              f"pinned={js.get('system_pinned')} | {name} | {js['reason'][:160]}")

    # honest pair A (and a second honest pair B over the same VUs: zk masks make its root, hence rho, differ)
    pfA, pubsA, layA, cA, chA = prove()
    print("python verdict, honest pair A:", R.verify_vus(pfA, pubsA, layA, coins=(cA, chA) if inter else None))
    dA = write("honest_A", pfA, pubsA, layA, cA, chA)
    case("honest pair A (pinned G/H systems)", True, rust(dA))
    pfB, pubsB, layB, cB, chB = prove()
    dB = write("honest_B", pfB, pubsB, layB, cB, chB)
    case("honest pair B", True, rust(dB))
    stA = _statement_of(R, pubsA, layA, R.cfg(layA.l), 1)

    # system confusion
    case("systems swapped (--system H --system-h G)", False, rust(dA, "system_h.bin", "system.bin"))
    case("G system given as both sides", False, rust(dA, "system.bin", "system.bin"))
    # cross-pair splicing: H of pair B with G of pair A (H's statement binds rho_B, the verifier derives rho_A)
    d = write("splice_hB_into_A", pfA, pubsA, layA, cA, chB if inter else None, h=proof_bytes(pfB.h))
    case("H proof of another pair (same VUs, other G root => other rho)", False, rust(d))
    d = write("splice_gB_into_A", pfA, pubsA, layA, cB if inter else None, chA, g=proof_bytes(pfB.g))
    case("G proof of another pair against A's H", False, rust(d))
    # the fingerprint message edited (one coordinate of one VU)
    g2 = dataclasses.replace(pfA.g, fp=pfA.g.fp.copy())
    g2.fp[0, 0, 0] = (int(g2.fp[0, 0, 0]) + 1) % P
    d = write("fp_edited", pfA, pubsA, layA, cA, chA, g=proof_bytes(g2))
    case("F_G[0][x][0] + 1 in G's proof", False, rust(d))
    # statement edits: VU 0 claims another x rank; roles; digests; the mode flag; the relation string (leaf)
    h = stA.hash_auth
    x_other = next((v for v in h.x_index if v != h.x_index[0]), None)
    if x_other is not None:
        other = dataclasses.replace(h, x_index=(x_other,) + tuple(h.x_index[1:]))
        idx2 = np.asarray(stA.b).copy(); idx2[0, 1] = x_other
        d = write("x_index_edited", pfA, pubsA, layA, cA, chA, st=dataclasses.replace(stA, hash_auth=other, b=idx2))
        case("VU 0 claims VU 1's x rank in the statement", False, rust(d))
    sh = stA.shared
    dg = np.asarray(sh.digests).copy(); dg[0, 0] = (dg[0, 0] + 1) % P
    d = write("digest_edited", pfA, pubsA, layA, cA, chA, st=dataclasses.replace(stA, shared=dataclasses.replace(sh, digests=dg)))
    case("unit 0 digest lane 0 + 1 in the v6 file", False, rust(d))
    u = np.asarray(sh.units).copy()
    nx = int((u[:, 0] == 0).sum())
    dg = np.asarray(sh.digests).copy(); dg[[0, nx]] = dg[[nx, 0]]
    d = write("xw_digests_swapped", pfA, pubsA, layA, cA, chA, st=dataclasses.replace(stA, shared=dataclasses.replace(sh, digests=dg)))
    case("x unit 0 and W unit 0 digests swapped (role confusion)", False, rust(d))
    u2 = u.copy(); u2[0, 0] = 1
    d = write("role_flipped", pfA, pubsA, layA, cA, chA, st=dataclasses.replace(stA, shared=dataclasses.replace(sh, units=u2)))
    case("unit 0 role 0 -> 1 in the v6 file", False, rust(d))
    raw = statement_bytes(stA)
    rel_b = stA.relation.encode()
    i = raw.index(rel_b)
    forged = raw[:i - 4] + len(b"fp8-ada+blake3").to_bytes(4, "little") + b"fp8-ada+blake3" + raw[i + len(rel_b):]
    d = write("relation_blake3", pfA, pubsA, layA, cA, chA, st=forged)
    case("relation string fp8-ada -> fp8-ada+blake3 (hash side is Poseidon2)", False, rust(d))
    # prover-level negatives the Python gate has, now through Rust
    def flip_row(rows):
        rows = rows.copy(); rows[0, 7] ^= 1
        return rows
    pf, pubs, lay, c, ch = prove(rows_override=flip_row)
    case("H hashes a flipped copy of unit 0's row", False, rust(write("h_row_flipped", pf, pubs, lay, c, ch)))
    pf, pubs, lay, c, ch = prove(rho_override=lambda rho: (rho + 1) % P)
    case("H proved on rho + 1 (not G's coins)", False, rust(write("h_rho_shift", pf, pubs, lay, c, ch)))
    pf, pubs, lay, c, ch = prove(rho_override=lambda rho: rho * 0 + 1)
    case("H proved on rho = 1", False, rust(write("h_rho_one", pf, pubs, lay, c, ch)))
    lane0 = R.sys.chain["fp"]["lanes"][0][0]

    def flip_lane(W):
        W[lane0, 3] = (W[lane0, 3] + 1) % P
    try:
        pf, pubs, lay, c, ch = prove(mutate=flip_lane)
        case("G lane row a.lane[0] + 1 at column 3", False, rust(write("g_lane_flipped", pf, pubs, lay, c, ch)))
    except ValueError as e:
        print(f"(skipped: the honest prover refuses to emit it: {e})")
    if inter:
        # coin custody: H's coins are the verifier's only if X.hcoins came from the verifier
        case("interactive: G's coins given, H's replayed from its transcript (reason must say so)", True,
             rust(write("no_hcoins", pfA, pubsA, layA, cA, None)))
        d = write("hcoins_are_gcoins", pfA, pubsA, layA, cA, cA)
        case("interactive: --coins-h = G's coins (not H's)", False, rust(d))
    (out / "verdicts.json").write_text(json.dumps([{"case": n, "want_accept": w, "verdict": js} for n, w, js in cases], indent=1))
    bad = [n for n, w, js in cases if js["accepted"] != w]
    print(f"{len(cases) - len(bad)}/{len(cases)} as expected" + (f"; UNEXPECTED: {bad}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
