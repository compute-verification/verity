"""Synthesize OCC-SILICON's credited-mode lock and counters with SI-ALG's flow (synth.py: yosys + ABC on
generic.lib, registers at a DFF a bit), check every netlist bit for bit against an exact Python reference, and
price the blocks per SM against the SM's s8 MAC array and FT.

    python3 occ_synth.py --alg DIR_OF_SILICON_ALG --out OUT [--vectors N]

DIR_OF_SILICON_ALG is SI-ALG's benchmarks/pouw/silicon_alg (branch cursor/compute-research-silicon-alg-8e03), whose
synth.py runs unchanged.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

PATTERNS_EXACT = {"m0": 0x5A10C3, "m1": 0x5A10C7, "nop": 0x000000, "seal": 0x7F0001, "enter": 0x7F0000}
PATTERNS_16 = {"tma_ld": 0x3B01, "tma_st": 0x3B02}
PATTERNS_12 = {"stsm": 0x2E4, "ldsm": 0x2E5, "mbar": 0x3C0, "wgf": 0x5A2, "bar": 0x3D0}
SUPPORT_12 = {"u0": 0x1A0, "u1": 0x1A5, "u2": 0x1B2, "u3": 0x1C7, "br": 0x0B0}
M48 = (1 << 48) - 1
M64 = (1 << 64) - 1


def klass(w: int) -> dict:
    op16, op12 = w >> 8, w >> 12
    m0, m1 = w == PATTERNS_EXACT["m0"], w == PATTERNS_EXACT["m1"]
    tma = op16 in PATTERNS_16.values()
    seal, enter = w == PATTERNS_EXACT["seal"], w == PATTERNS_EXACT["enter"]
    c = m0 or m1 or tma or op12 in PATTERNS_12.values() or w == 0 or seal
    s = op12 in SUPPORT_12.values()
    return dict(c=c, s=s, m0=m0, m1=m1, seal=seal, enter=enter, tma=tma)


def lock_ref(v: dict, i: int) -> dict:
    g = lambda k: v[k][i]
    cm, p, tag = g("cm"), g("p"), g("tag")
    iv = g("iv")
    ks = [klass(g(f"w{j}")) for j in range(4)]
    on = [(iv >> j) & 1 for j in range(4)]
    x = any(on[j] and not ks[j]["c"] and not ks[j]["s"] for j in range(4))
    rbad = any((g("rv") >> j) & 1 and (not (g("rcm") >> j) & 1 or g(f"rtag{j}") != tag) for j in range(2))
    enter_req = any(on[j] and ks[j]["enter"] for j in range(4))
    enter_ok = enter_req and not cm
    bad = x or rbad or g("trap") or g("preempt") or g("dbg") or g("scan") or enter_req
    seal_fire = cm and any(on[j] and ks[j]["seal"] for j in range(4))
    cnt = lambda k: sum(1 for j in range(4) if cm and on[j] and ks[j][k])
    return {"cm_n": int(enter_ok or (cm and not seal_fire)), "p_n": 0 if enter_ok else int(p or (cm and bad)),
            "tag_n": g("new_tag") if enter_ok else tag, "n_m0": cnt("m0"), "n_m1": cnt("m1"), "n_s": cnt("s"),
            "n_tma": cnt("tma"), "any_issue": int(bool(cm and iv)), "seal_fire": int(bool(seal_fire)),
            "launch_ok": int(not cm)}


def word(rng: random.Random) -> int:
    r = rng.random()
    if r < 0.25:
        return rng.choice(list(PATTERNS_EXACT.values()))
    if r < 0.40:
        return (rng.choice(list(PATTERNS_16.values())) << 8) | rng.getrandbits(8)
    if r < 0.60:
        return (rng.choice(list(PATTERNS_12.values())) << 12) | rng.getrandbits(12)
    if r < 0.80:
        return (rng.choice(list(SUPPORT_12.values())) << 12) | rng.getrandbits(12)
    return rng.getrandbits(24)


def lock_gen(rng: random.Random, n: int) -> dict:
    v = {k: [] for k in ("cm", "p", "tag", "iv", "w0", "w1", "w2", "w3", "rv", "rcm", "rtag0", "rtag1",
                         "launch", "trap", "preempt", "dbg", "scan", "new_tag")}
    for _ in range(n):
        tag = rng.getrandbits(16)
        v["cm"].append(rng.getrandbits(1)); v["p"].append(int(rng.random() < 0.2)); v["tag"].append(tag)
        v["iv"].append(rng.getrandbits(4))
        for j in range(4):
            v[f"w{j}"].append(word(rng))
        v["rv"].append(rng.getrandbits(2)); v["rcm"].append(rng.getrandbits(2))
        for j in range(2):
            v[f"rtag{j}"].append(tag if rng.random() < 0.7 else rng.getrandbits(16))
        for k in ("launch", "trap", "preempt", "dbg", "scan"):
            v[k].append(int(rng.random() < 0.05))
        v["new_tag"].append(rng.getrandbits(16))
    return v


def ns_ref(v, i):
    return {"ns_n": 0 if v["enter_ok"][i] else (v["ns"][i] + v["n_s"][i]) & M48}


def diag_ref(v, i):
    g = lambda k: v[k][i]
    e = g("enter_ok")
    z = lambda val: 0 if e else val & M48
    return {"c_cm_n": z(g("c_cm") + g("cm")), "c_st_n": z(g("c_st") + (g("cm") & (1 - g("any_issue")))),
            "q0_n": z(g("q0") + g("n_m0")), "q1_n": z(g("q1") + g("n_m1")),
            "tb_n": z(g("tb") + (g("tma_bytes") if g("tma_v") else 0)),
            "t_en_n": g("gtimer") if e else g("t_en"), "t_se_n": g("gtimer") if g("seal_fire") else g("t_se")}


def near_wrap(rng, bits):
    return (1 << bits) - 1 - rng.getrandbits(3) if rng.random() < 0.2 else rng.getrandbits(bits)


def designs() -> dict:
    d = {}
    d["lock_step"] = dict(gen=lock_gen, ref=lock_ref, regs=1 + 1 + 16, role="lock: class ROM x4, gate, poison, tag")
    d["ns_step"] = dict(
        gen=lambda rng, n: {"enter_ok": [int(rng.random() < 0.1) for _ in range(n)],
                            "n_s": [rng.randrange(5) for _ in range(n)], "ns": [near_wrap(rng, 48) for _ in range(n)]},
        ref=ns_ref, regs=48, role="lock: support-issue counter n_S")
    d["diag_step"] = dict(
        gen=lambda rng, n: {**{k: [rng.getrandbits(1) for _ in range(n)]
                               for k in ("cm", "seal_fire", "any_issue", "tma_v")},
                            "enter_ok": [int(rng.random() < 0.1) for _ in range(n)],
                            "n_m0": [rng.randrange(5) for _ in range(n)], "n_m1": [rng.randrange(5) for _ in range(n)],
                            "tma_bytes": [rng.getrandbits(17) for _ in range(n)],
                            "gtimer": [rng.getrandbits(64) for _ in range(n)],
                            **{k: [near_wrap(rng, 48) for _ in range(n)] for k in ("c_cm", "c_st", "q0", "q1", "tb")},
                            **{k: [rng.getrandbits(64) for _ in range(n)] for k in ("t_en", "t_se")}},
        ref=diag_ref, regs=5 * 48 + 2 * 64, role="diagnostic counters (believed)")
    for name, w in (("seal_mux_min", 65), ("seal_mux_diag", 368)):
        d[name] = dict(
            gen=lambda rng, n, w=w: {"sel": [rng.getrandbits(1) for _ in range(n)],
                                     "wb": [rng.getrandbits(w) for _ in range(n)],
                                     "sb": [rng.getrandbits(w) for _ in range(n)]},
            ref=lambda v, i: {"y": v["sb"][i] if v["sel"][i] else v["wb"][i]}, regs=0,
            role=f"seal absorb mux, {w} bits")
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alg", required=True, help="SI-ALG's benchmarks/pouw/silicon_alg directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--vectors", type=int, default=4096)
    a = ap.parse_args()
    sys.path.insert(0, a.alg)
    import synth  # SI-ALG's flow, unchanged
    synth.RTL = [HERE / "lock.v"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = {}
    for top, spec in designs().items():
        with tempfile.TemporaryDirectory() as td:
            r = synth.run_yosys(top, Path(td))
        nets = r.pop("nets")
        rng = random.Random(f"{top}:0")
        vec = spec["gen"](rng, a.vectors)
        want = [spec["ref"](vec, i) for i in range(a.vectors)]
        chk = {}
        for kind, net in nets.items():
            got = synth.simulate(net, top, vec, a.vectors)
            bad = [i for i in range(a.vectors) for k in want[i] if got[k][i] != want[i][k]]
            chk[kind] = {"vectors": a.vectors, "mismatches": len(bad), "first": bad[:1]}
        r["check"] = chk
        r["ge_reg"] = spec["regs"] * synth.DFF_GE
        r["ge"] = r["ge_comb"] + r["ge_reg"]
        r["role"] = spec["role"]
        rows[top] = r
        ok = all(c["mismatches"] == 0 for c in chk.values())
        print(f"{top:14s} comb {r['ge_comb']:8.1f} reg {r['ge_reg']:7.1f} total {r['ge']:8.1f} GE  "
              f"ANDs {r['ands']:5d} depth {r['depth']:3d}  check {'ok' if ok else 'FAIL'}", flush=True)
    # per-SM pricing; anchors from SI-ALG (s8 MAC 817.9 GE) and SI-FOLD (FT 22.2 / 46.8 GE a MAC)
    base_mac = 817.9
    sm = {"H200": {"macs": 4096, "ft_per_mac": 22.2}, "sm_120": {"macs": 1024, "ft_per_mac": 46.8}}
    lock_min = rows["lock_step"]["ge"] + rows["ns_step"]["ge"] + rows["seal_mux_min"]["ge"]
    diag = rows["diag_step"]["ge"] + rows["seal_mux_diag"]["ge"]
    price = {}
    for k, s in sm.items():
        arr = s["macs"] * base_mac
        ft = s["macs"] * s["ft_per_mac"]
        price[k] = {"mac_array_ge": arr, "ft_ge": ft, "lock_min_ge": lock_min, "diag_ge": diag,
                    "lock_min_pct_array": 100 * lock_min / arr, "diag_pct_array": 100 * diag / arr,
                    "lock_min_pct_ft": 100 * lock_min / ft, "lock_plus_diag_pct_array": 100 * (lock_min + diag) / arr}
        print(f"{k:7s} SM MAC array {arr/1e6:.3f} MGE, FT {ft/1e3:.1f} kGE; lock {lock_min:.0f} GE = "
              f"{price[k]['lock_min_pct_array']:.3f}% of array ({price[k]['lock_min_pct_ft']:.2f}% of FT); "
              f"+diag {diag:.0f} GE -> {price[k]['lock_plus_diag_pct_array']:.3f}%")
    (out / "occ_synth.json").write_text(json.dumps({"rows": rows, "per_sm": price}, indent=1, default=str))
    bad = [t for t, r in rows.items() if any(c["mismatches"] for c in r["check"].values())]
    if bad:
        print("CHECK FAILED:", bad)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
