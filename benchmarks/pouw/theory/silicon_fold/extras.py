"""SI-FOLD's second pass: the four open items of the fused transcript (FT) and the reconciliation with SI-NOISE's sketch.

    python3 extras.py --f1-build BUILD.json --f1-result RESULT.json --synth SYNTH.json --json OUT.json --md OUT.md
        --verdict V.json [--mutations] [--scratch DIR]

- **C1, closure.** Every tensor form F1 (`benchmarks/tc_probe_fp4/f1_rates.py`) enumerates for sm_120a, with ptxas's verdict and
  SASS opcode from F1's ``build`` (BUILD.json) and its measured rate from F1's run of record (RESULT.json, r20260930-111747-26f3).
  For each form: the accumulator words it writes a cycle per SM, the trigger rule's verdict (advance τ only on a credited SASS
  tag), and the T1 a C-injection through it would reach if the rule let it advance τ, at the measured issue rate and under W1's
  MAC-proportional prices.
- **E6, separation on SI-NOISE's adder sketch** σ ← rotl(σ, 1) ⊕ (acc_t + n_t): a device run on hidden words H yields the sketch of
  zero-data words D without the noise, σ_D = σ_H ⊕ fold(acc^H), exactly when every hidden word is ≡ 0 mod 2^b. The fold's
  instructions are counted (32 rotation classes, 3-input XORs), so its T1 is a count, not an argument.
- **E7, energy.** The mapped s8 and s9 MAC netlists (yosys + ABC to the generic gates) are simulated bit-exactly over operand
  streams, with and without on-chip rank-1 noise on both operands (a' = a + s_i v_k, b' = b + u_j w_k), counting gate-output
  toggles weighted by cell area, plus the operand and accumulator flops. The noise is removed exactly from the outputs.
- **W, wiring.** The tap's width and length on each part, as arithmetic on stated parameters [estimate].
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import synth  # noqa: E402

M32 = 0xFFFFFFFF
F1_RUN = "r20260930-111747-26f3"
#: the credited SASS tags: Pearl-C's FP8 E4M3 step and Pearl-C4's NVFP4 step, each at its full k
CREDITED = ("QMMA.16832.F32.E4M3.E4M3", "OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X")
CREDITED_K = dict(zip(CREDITED, (32, 64)))
#: W1 prices per MAC on sm_120 (r20260930-060338-26b9), at each format's full-k shape
W1_PER_MAC = {"fp4": 0.5, "f8f6f4": 1.0, "mxf8f6f4": 1.0, "fp8": 1.0, "f16": 2.0, "tf32": 4.0, "int": 1.0}
W1_INT4 = 10.2
SM120_OP = 16.2  # LOP3 / SHF / PRMT, in FP8 MACs (r20260930-060338-26b9)
MACS_PER_WORD = 32


# ------------------------------------------------------------------------------------------------------------ C1: closure

def _f1_module():
    """F1's form list, imported without its device harness (its `probe` and settings imports are stubbed)."""
    class _Any:
        def __getattr__(self, n):
            return self

        def __call__(self, *a, **k):
            return self

    stubs = {"probe": types.ModuleType("probe"), "verity_config": types.ModuleType("verity_config"),
             "verity_config.settings": types.ModuleType("verity_config.settings")}
    stubs["probe"].__getattr__ = lambda n: _Any()
    stubs["verity_config.settings"].SHARED = _Any()
    stubs["verity_config"].settings = stubs["verity_config.settings"]
    saved = {k: sys.modules.get(k) for k in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location("f1_rates_forms", HERE.parents[2] / "tc_probe_fp4" / "f1_rates.py")
        m = importlib.util.module_from_spec(spec)
        sys.modules["f1_rates_forms"] = m
        spec.loader.exec_module(m)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return m


def closure(build: dict, result: dict) -> dict:
    f1 = _f1_module()
    rates = {x["name"][: -len("_products_per_sm_per_clock")]: x["value"] for x in result["measurements"]
             if x["name"].endswith("_products_per_sm_per_clock")}
    rows = []
    for f in f1.FORMS_TENSOR:
        acc_kind, acc_regs = f.acc
        words = 32 * acc_regs * (2 if acc_kind == "d" else 1)
        row = {"form": f.name, "ptx": f.ptx, "family": f.family, "mnk": list(f.mnk), "sparse": f.sparse, "words": words,
               "logical_macs": f.logical_macs, "products": f.products}
        if f.name in build.get("refused", {}):
            row.update(accepted=False, why=build["refused"][f.name][0].split("error   : ")[-1])
            rows.append(row)
            continue
        g = build["gate"][f.name]
        tags = sorted(g.get("mma_opcodes", {}))
        pp = rates[f.name]
        instr = pp / f.products
        w1 = None
        if f.family in W1_PER_MAC and f.product == "e2m1-capable":
            w1 = W1_PER_MAC[f.family] * f.products / words
        elif f.product in ("s4", "u4"):
            w1 = W1_INT4 * f.products / words
        row.update(accepted=True, tags=tags, emulated=g.get("emulated") or False, products_per_sm_clk=pp,
                   instr_per_sm_clk=instr, words_per_sm_clk=instr * words, macs_per_word=f.logical_macs / words,
                   w1_per_word=w1, credited=bool(tags) and set(tags) <= set(CREDITED) and not g.get("emulated"))
        rows.append(row)
    acc = [r for r in rows if r["accepted"]]
    cred = [r for r in acc if r["credited"]]
    w_cred = min(r["words_per_sm_clk"] for r in cred)
    for r in acc:
        if r["emulated"] and r["w1_per_word"] is not None and r["family"] != "int":
            r["w1_per_word"] = MACS_PER_WORD * w_cred / r["words_per_sm_clk"]  # W1 prices the native datapath; ptxas emulates this one
        r["t1_time_if_untagged"] = max(0.0, 1 - w_cred / r["words_per_sm_clk"])
        r["t1_w1_if_untagged"] = None if r["w1_per_word"] is None else max(0.0, 1 - r["w1_per_word"] / MACS_PER_WORD)
    fastest = max(acc, key=lambda r: r["words_per_sm_clk"])
    w1_rows = [r for r in acc if r["t1_w1_if_untagged"] is not None]
    w1_worst = max(w1_rows, key=lambda r: r["t1_w1_if_untagged"])
    cred_tags = {t for r in cred for t in r["tags"]}
    shared = sorted(r["form"] for r in acc if not r["credited"] and set(r["tags"]) & cred_tags)
    short = sorted(r["form"] for r in cred if r["sparse"] or r["mnk"][2] != CREDITED_K[r["tags"][0]])
    return {
        "source": {"forms": "benchmarks/tc_probe_fp4/f1_rates.py FORMS_TENSOR", "build_nvcc": build.get("nvcc"),
                   "rates_run": result.get("run_id", F1_RUN)},
        "forms": len(rows), "accepted": len(acc), "refused": len(rows) - len(acc),
        "native": sum(1 for r in acc if not r["emulated"]), "emulated": sum(1 for r in acc if r["emulated"]),
        "credited_forms": sorted(r["form"] for r in cred), "credited_words_per_sm_clk": w_cred,
        "fastest_writer": {"form": fastest["form"], "words_per_sm_clk": fastest["words_per_sm_clk"]},
        "max_words_ratio": fastest["words_per_sm_clk"] / w_cred,
        "t1_time_no_rule": max(r["t1_time_if_untagged"] for r in acc),
        "t1_w1_no_rule": w1_worst["t1_w1_if_untagged"], "t1_w1_no_rule_form": w1_worst["form"],
        "t1_w1_no_rule_forms": sorted(r["form"] for r in w1_rows if r["t1_w1_if_untagged"] >= w1_worst["t1_w1_if_untagged"] - 1e-9),
        "noncredited_sharing_a_credited_tag": shared,
        "credited_tag_below_full_k": short,
        "rows": rows,
    }


# ------------------------------------------------------------------------------------------------- E6: SI-NOISE separation

def _rotl(x: np.ndarray, r: int) -> np.ndarray:
    r %= 32
    return x if r == 0 else ((x << np.uint64(r)) | (x >> np.uint64(32 - r))) & np.uint64(M32)


def sketch(words: np.ndarray, noise: np.ndarray) -> np.ndarray:
    """SI-NOISE's P1 over axis 0: σ ← rotl(σ, 1) ⊕ (acc_t + n_t) mod 2^32."""
    s = np.zeros(words.shape[1:], dtype=np.uint64)
    for t in range(words.shape[0]):
        s = _rotl(s, 1) ^ ((words[t] + noise[t]) & np.uint64(M32))
    return s


def fold_counted(words: np.ndarray) -> tuple[np.ndarray, int]:
    """⊕_t rotl(w_t, T-1-t) over axis 0 the cheap way: XOR each rotation class (3-input LOP3s), rotate each class once
    (SHF), then XOR the classes. Returns the fold of every column and the instruction count for one column."""
    T = words.shape[0]
    cls: dict[int, list[int]] = {}
    for t in range(T):
        cls.setdefault((T - 1 - t) % 32, []).append(t)
    out = np.zeros(words.shape[1:], dtype=np.uint64)
    ops = 0
    for r, ts in cls.items():
        x = words[ts[0]].copy()
        for t in ts[1:]:
            x ^= words[t]
        ops += math.ceil((len(ts) - 1) / 2) + (1 if r else 0)
        out ^= _rotl(x, r)
    ops += math.ceil((len(cls) - 1) / 2)
    return out, ops


def _acc_words(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Running int32 accumulators after each k32 step for every cell: a (m, K), b (n, K) -> (K/32, m, n) as uint64 words."""
    m, K = a.shape
    steps = [np.einsum("ik,jk->ij", a[:, s:s + 32].astype(np.int64), b[:, s:s + 32].astype(np.int64)) for s in range(0, K, 32)]
    return (np.cumsum(np.stack(steps), axis=0) & M32).astype(np.uint64)


def e6_separation(rng: np.random.Generator, m: int = 8, n: int = 64, Ks: tuple[int, ...] = (2048, 8192)) -> dict:
    cases = {
        "b1_even": (1, lambda sh: 2 * rng.integers(-64, 64, sh), lambda sh: rng.integers(-128, 128, sh)),
        "b1_general": (1, lambda sh: rng.integers(-128, 128, sh), lambda sh: rng.integers(-128, 128, sh)),
        "b8_int4x16": (8, lambda sh: 16 * rng.integers(-8, 8, sh), lambda sh: 16 * rng.integers(-8, 8, sh)),
        "b8_even": (8, lambda sh: 2 * rng.integers(-64, 64, sh), lambda sh: rng.integers(-128, 128, sh)),
        "b32_int4x16": (32, lambda sh: 16 * rng.integers(-8, 8, sh), lambda sh: 16 * rng.integers(-8, 8, sh)),
    }
    out: dict = {"cases": {}, "fold": {}}
    K = Ks[0]
    for name, (b, ga, gb) in cases.items():
        A, B = ga((m, K)), gb((n, K))
        hw = _acc_words(A, B)
        noise = rng.integers(0, 1 << b, hw.shape, dtype=np.uint64) if b < 32 else rng.integers(0, 1 << 32, hw.shape, dtype=np.uint64)
        sig_h = sketch(hw, noise)
        sig_d = sketch(np.zeros_like(hw), noise)
        f, _ = fold_counted(hw)
        direct = sketch(hw, np.zeros_like(hw))
        cells = sig_d.size
        out["cases"][name] = {"b": b, "cells": int(cells), "separated": int(np.sum((sig_h ^ f) == sig_d)),
                              "fold_matches_direct": bool(np.array_equal(f, direct)),
                              "hidden_words_0_mod_2b": bool(np.all(hw % np.uint64(1 << min(b, 63)) == 0)) if b < 32 else bool(np.all(hw == 0))}
    for K in Ks:
        T = K // 32
        _, per_cell = fold_counted(np.zeros((T, 1), dtype=np.uint64))
        _, per_row = fold_counted(np.zeros((T * n, 1), dtype=np.uint64))
        per_row += 1  # the row's 16-bit fold of its 32-bit sum
        for fin, ops in (("cell", per_cell / T), ("row_linear", per_row / (T * n))):
            out["fold"][f"K{K}_{fin}"] = {
                "ops_per_word": ops,
                "w1_writes_per_word": ops, "t1_w1": max(0.0, 1 - ops),
                "sm120_share_of_stock": ops * SM120_OP / MACS_PER_WORD, "t1_sm120": max(0.0, 1 - ops * SM120_OP / MACS_PER_WORD),
            }
    return out


# ------------------------------------------------------------------------------------------------------- E7: noise energy

GATE = {
    "$_NOT_": lambda p: ~p["A"], "$_BUF_": lambda p: p["A"],
    "$_AND_": lambda p: p["A"] & p["B"], "$_NAND_": lambda p: ~(p["A"] & p["B"]),
    "$_OR_": lambda p: p["A"] | p["B"], "$_NOR_": lambda p: ~(p["A"] | p["B"]),
    "$_XOR_": lambda p: p["A"] ^ p["B"], "$_XNOR_": lambda p: ~(p["A"] ^ p["B"]),
    "$_ANDNOT_": lambda p: p["A"] & ~p["B"], "$_ORNOT_": lambda p: p["A"] | ~p["B"],
    "$_MUX_": lambda p: np.where(p["S"], p["B"], p["A"]), "$_NMUX_": lambda p: ~np.where(p["S"], p["B"], p["A"]),
    "$_AOI3_": lambda p: ~((p["A"] & p["B"]) | p["C"]), "$_OAI3_": lambda p: ~((p["A"] | p["B"]) & p["C"]),
    "$_AOI4_": lambda p: ~((p["A"] & p["B"]) | (p["C"] & p["D"])), "$_OAI4_": lambda p: ~((p["A"] | p["B"]) & (p["C"] | p["D"])),
}


class Netlist:
    """A yosys JSON netlist of generic gates, evaluated over a batch of cycles (one bool row per net)."""

    def __init__(self, path: Path, top: str):
        mod = json.loads(path.read_text())["modules"][top]
        self.ports = {p: v["bits"] for p, v in mod["ports"].items()}
        cells = list(mod["cells"].values())
        driver = {c["connections"]["Y"][0]: i for i, c in enumerate(cells)}
        order, state = [], {}

        def visit(i):
            stack = [(i, False)]
            while stack:
                j, done = stack.pop()
                if done:
                    if state.get(j) != 2:
                        state[j] = 2
                        order.append(j)
                    continue
                if state.get(j):
                    continue
                state[j] = 1
                stack.append((j, True))
                for k, bits in cells[j]["connections"].items():
                    if k != "Y" and isinstance(bits[0], int) and bits[0] in driver and not state.get(driver[bits[0]]):
                        stack.append((driver[bits[0]], False))

        for i in range(len(cells)):
            visit(i)
        self.cells = [cells[i] for i in order]
        unknown = {c["type"] for c in self.cells} - set(GATE)
        if unknown:
            sys.exit(f"extras.py: cells without a model: {sorted(unknown)}")
        self.ge = sum(synth.GE[c["type"]] for c in self.cells)

    def run(self, inputs: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], float]:
        """inputs: port -> uint64 array (cycles,). Returns output words and the area-weighted toggles a cycle."""
        cyc = next(iter(inputs.values())).shape[0]
        net: dict = {"0": np.zeros(cyc, bool), "1": np.ones(cyc, bool), "x": np.zeros(cyc, bool)}
        for p, v in inputs.items():
            for k, bit in enumerate(self.ports[p]):
                net[bit] = ((v >> np.uint64(k)) & np.uint64(1)).astype(bool)
        tog = 0.0
        for c in self.cells:
            y = GATE[c["type"]]({k: net[b[0]] for k, b in c["connections"].items() if k != "Y"})
            net[c["connections"]["Y"][0]] = y
            tog += synth.GE[c["type"]] * float(np.count_nonzero(y[1:] != y[:-1]))
        outs = {}
        for p, bits in self.ports.items():
            if p not in inputs:
                outs[p] = sum(net[b].astype(np.uint64) << np.uint64(k) for k, b in enumerate(bits))
        return outs, tog / (cyc - 1)


def _netlist(name: str, scratch: Path) -> Netlist:
    d = scratch / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "m.v").write_text(synth.rtl.verilog(name))
    synth._yosys(f"read_verilog -sv m.v; synth -flatten -top {name} -booth; abc -g {synth.GATES}; opt_clean; write_json n.json", d)
    return Netlist(d / "n.json", name)


def _pack_lanes(x: np.ndarray, w: int) -> np.ndarray:
    """(cycles, 4) signed ints -> one w*4-bit word a cycle (two's complement lanes)."""
    m = (1 << w) - 1
    return sum((x[:, i].astype(np.int64) & m).astype(np.uint64) << np.uint64(w * i) for i in range(4))


def _bit_toggles(v: np.ndarray, bits: int) -> float:
    d = (v[1:] ^ v[:-1]) & np.uint64((1 << bits) - 1)
    return float(sum(int(bin(int(x)).count("1")) for x in d)) / (len(v) - 1)


def e7_energy(rng: np.random.Generator, scratch: Path, m: int = 8, n: int = 8, K: int = 256) -> dict:
    """One 4-wide MAC unit walks an m x n tile's cells, K/4 cycles each, accumulator running (reset per cell)."""
    nl = {8: _netlist("mac4_s8", scratch), 9: _netlist("mac4_s9", scratch)}
    s = rng.choice([-1, 1], m)
    u = rng.choice([-1, 1], n)
    v = rng.integers(-128, 128, K)
    w = rng.integers(-128, 128, K)

    def llm(sh, sd, outlier):
        x = np.clip(np.rint(rng.normal(0, sd, sh)), -127, 127).astype(np.int64)
        mask = rng.random(sh) < outlier
        x[mask] = rng.choice([-1, 1], int(mask.sum())) * rng.integers(64, 128, int(mask.sum()))
        return x

    streams = {
        "zero": (np.zeros((m, K), np.int64), np.zeros((n, K), np.int64)),
        "const": (np.full((m, K), 85, np.int64), np.full((n, K), 85, np.int64)),
        "u2": (rng.integers(0, 4, (m, K)), rng.integers(0, 4, (n, K))),
        "llm_like": (llm((m, K), 10, 0.005), llm((n, K), 24, 0.0)),
        "rand": (rng.integers(-128, 128, (m, K)), rng.integers(-128, 128, (n, K))),
    }
    res: dict = {"ge": {f"mac4_s{k}": x.ge for k, x in nl.items()}, "streams": {}}
    ok_all, denoise_ok = True, True
    for name, (A, B) in streams.items():
        for noised in (False, True):
            An = A + s[:, None] * v[None, :] if noised else A
            Bn = B + u[:, None] * w[None, :] if noised else B
            widths = (9,) if noised else (8, 9)
            for wd in widths:
                a_cyc, b_cyc, cell_of = [], [], []
                for i in range(m):
                    for j in range(n):
                        a_cyc.append(An[i].reshape(-1, 4))
                        b_cyc.append(Bn[j].reshape(-1, 4))
                a_cyc, b_cyc = np.concatenate(a_cyc), np.concatenate(b_cyc)
                prods = np.sum(a_cyc * b_cyc, axis=1)
                steps = K // 4
                acc_in = np.zeros(len(prods), np.int64)
                acc_out = np.zeros(len(prods), np.int64)
                for c in range(m * n):
                    seg = prods[c * steps:(c + 1) * steps]
                    cs = np.cumsum(seg)
                    acc_out[c * steps:(c + 1) * steps] = cs
                    acc_in[c * steps:(c + 1) * steps] = cs - seg
                ins = {"a": _pack_lanes(a_cyc, wd), "b": _pack_lanes(b_cyc, wd), "acc": (acc_in & M32).astype(np.uint64)}
                outs, tog = nl[wd].run(ins)
                ok = bool(np.array_equal(outs["y"], (acc_out & M32).astype(np.uint64)))
                ok_all &= ok
                flops = synth.GE_DFF * (_bit_toggles(ins["a"], 4 * wd) + _bit_toggles(ins["b"], 4 * wd) + _bit_toggles(ins["acc"], 32))
                res["streams"][f"{name}{'+noise' if noised else ''}@s{wd}"] = {
                    "logic": tog, "flops": flops, "total": tog + flops, "netlist_exact": ok, "cycles": len(prods)}
            if noised:
                D = A @ B.T
                Dn = An @ Bn.T
                rec = Dn - np.outer(A @ w, u) - np.outer(s, B @ v) - np.outer(s, u) * int(v @ w)
                denoise_ok &= bool(np.array_equal(rec, D))
    S = res["streams"]
    base = S["rand@s8"]["total"]
    for k in S:
        S[k]["vs_rand_s8"] = S[k]["total"] / base
    z, u2, r = S["zero@s8"]["total"], S["u2@s8"]["total"], S["rand@s8"]["total"]
    noised = {k: x["total"] for k, x in S.items() if "+noise" in k}
    honest = S["llm_like+noise@s9"]["total"]
    res.update(
        netlists_exact=ok_all, denoise_exact=denoise_ok,
        u2_share_of_rand_above_zero=(u2 - z) / (r - z),
        gap_no_noise={"zero_vs_rand": 1 - z / r, "zero_vs_llm_like": 1 - z / S["llm_like@s8"]["total"],
                      "u2_vs_llm_like": 1 - u2 / S["llm_like@s8"]["total"]},
        gap_noise={"min_noised_vs_llm_like_noised": 1 - min(noised.values()) / honest,
                   "spread_noised": (max(noised.values()) - min(noised.values())) / honest,
                   "cheapest_noised": min(noised, key=noised.get)},
        honest_energy_rise=honest / S["llm_like@s8"]["total"],
        area={"s9_over_s8": nl[9].ge / nl[8].ge,
              "mac_ge_per_mac": {"s8": nl[8].ge / 4, "s9": nl[9].ge / 4}},
        tile={"m": m, "n": n, "K": K},
    )
    return res


def noise_costs(s8_ge4: float, s9_ge4: float, add9_ge: float, tile_m: int = 128, tile_n: int = 128, bcast: int = 16) -> dict:
    """Area and work of the rank-1 operand noise: each operand element gets one s8 + s8 add-or-subtract, shared by the
    `bcast` MACs it is broadcast to [area synthesized; broadcast and tile estimate]."""
    d_mac = (s9_ge4 - s8_ge4) / 4
    adders = add9_ge * 2 / bcast
    return {
        "s8_ge_per_mac": s8_ge4 / 4, "s9_ge_per_mac": s9_ge4 / 4,
        "s9_minus_s8_ge_per_mac": d_mac, "noise_adders_ge_per_mac": adders, "broadcast": bcast,
        "share_of_s8_mac": (d_mac + adders) / (s8_ge4 / 4),
        "denoise_work_share": 1 / tile_m + 1 / tile_n, "tile": [tile_m, tile_n],
    }


# --------------------------------------------------------------------------------------------------------------- W: wiring

def wiring(unit_ge: dict[str, float], um2_per_ge: float = 0.798, pitch_um: float = 0.28, layers: int = 2) -> dict:
    """The tap: a fan-out of the tensor core's accumulator write-back bus into the τ unit beside it [estimate]."""
    parts = {
        "sm_120 (RTX PRO 6000)": {"macs_per_tc_clk": 256, "rho": "keccak400_rnd"},
        "H100 class": {"macs_per_tc_clk": 1024, "rho": "keccak1600_rnd"},
    }
    out = {}
    for name, p in parts.items():
        wires = p["macs_per_tc_clk"] // MACS_PER_WORD * 32
        side = math.sqrt(unit_ge[p["rho"]] * um2_per_ge)
        tracks_um = wires * pitch_um / layers
        out[name] = {
            "words_per_tc_clk": p["macs_per_tc_clk"] // MACS_PER_WORD, "tap_wires": wires, "rho": p["rho"],
            "rho_ge": unit_ge[p["rho"]], "rho_side_um_45nm": side,
            "tap_track_width_um": tracks_um, "track_share_of_side": tracks_um / side,
            "dedicated_channel_ge_equiv": tracks_um * side / um2_per_ge,
            "dedicated_channel_ge_per_mac": tracks_um * side / um2_per_ge / p["macs_per_tc_clk"],
            "rho_ge_per_mac": unit_ge[p["rho"]] / p["macs_per_tc_clk"],
        }
    return out


# --------------------------------------------------------------------------------------------------------------- report

def closure_checks(c: dict) -> list[tuple[str, bool]]:
    return [
        ("C1 every F1 tensor form is accepted with a measured rate or refused by ptxas", c["accepted"] + c["refused"] == c["forms"]),
        ("C1 both credited steps are present (FP8 E4M3 k32, NVFP4 k64)", any("nvf4" == f for f in c["credited_forms"])
         and any("e4m3" in f for f in c["credited_forms"])),
        ("C1 no form writes accumulator words faster than the credited steps (2% tolerance)", c["max_words_ratio"] <= 1.02),
        ("C1 no non-credited form lowers to a credited SASS tag", not c["noncredited_sharing_a_credited_tag"]),
        ("C1 every form that reaches a credited tag is dense at that tag's full k", not c["credited_tag_below_full_k"]),
        ("C1 W1's MAC-proportional prices reproduce the no-rule T1 of 50%", abs(c["t1_w1_no_rule"] - 0.5) < 1e-9),
    ]


def checks(res: dict) -> list[tuple[str, bool]]:
    c, e6, e7 = res["closure"], res["e6"]["cases"], res["e7"]
    return closure_checks(c) + [
        ("E6 separation is exact on hidden words = 0 mod 2^b (b = 1, even operand)", e6["b1_even"]["separated"] == e6["b1_even"]["cells"]),
        ("E6 separation is exact for b = 8 on int4 work carried x16", e6["b8_int4x16"]["separated"] == e6["b8_int4x16"]["cells"]),
        ("E6 separation fails on general words (b = 1) and on even words at b = 8",
         e6["b1_general"]["separated"] < e6["b1_general"]["cells"] // 2 and e6["b8_even"]["separated"] < e6["b8_even"]["cells"] // 2),
        ("E6 separation fails at b = 32", e6["b32_int4x16"]["separated"] == 0),
        ("E6 the counted fold equals the sketch's own fold", all(x["fold_matches_direct"] for x in e6.values())),
        ("E7 every simulated netlist cycle matches the reference", e7["netlists_exact"]),
        ("E7 the rank-1 noise is removed exactly", e7["denoise_exact"]),
        ("E7 without noise zero data is cheaper than random by > 50%", e7["gap_no_noise"]["zero_vs_rand"] > 0.5),
        ("E7 with noise no stream is more than 10% cheaper than the noised honest stream",
         e7["gap_noise"]["min_noised_vs_llm_like_noised"] < 0.10),
    ]


def mutations(build: dict, result: dict, scratch: Path) -> list[tuple[str, bool]]:
    """Each mutation must turn its check red."""
    out = []
    b = json.loads(json.dumps(build))
    b["gate"]["fp8_e4m3_k16"]["mma_opcodes"] = {CREDITED[0]: 32}
    red = lambda c: not all(ok for _, ok in closure_checks(c))  # noqa: E731
    out.append(("a k16 form lowered to the credited tag is caught", red(closure(b, result))))
    r = json.loads(json.dumps(result))
    for x in r["measurements"]:
        if x["name"] == "s8_k16_products_per_sm_per_clock":
            x["value"] *= 2
    out.append(("a k16 form at twice its measured rate is caught", red(closure(build, r))))
    nl = _netlist("mac4_s8", scratch)
    i = next(k for k, c in enumerate(nl.cells) if c["type"] == "$_AND_")
    nl.cells[i] = dict(nl.cells[i], type="$_OR_")
    rng = np.random.default_rng(1)
    a, bb = rng.integers(-128, 128, (64, 4)), rng.integers(-128, 128, (64, 4))
    acc = rng.integers(0, 1 << 32, 64, dtype=np.uint64)
    want = (acc.astype(np.int64) + np.sum(a * bb, axis=1)) & M32
    got, _ = nl.run({"a": _pack_lanes(a, 8), "b": _pack_lanes(bb, 8), "acc": acc})
    out.append(("one AND gate turned OR breaks the netlist check", not np.array_equal(got["y"], want.astype(np.uint64))))
    return out


def table(res: dict) -> str:
    c = res["closure"]
    L = ["## C1: closure over every tensor form sm_120a accepts (rates: F1 run " + c["source"]["rates_run"] + ")", "",
         "| form | SASS | words /SM/clk | MACs a word | credited | T1 if untagged: time / W1 |", "|---|---|---:|---:|---|---|"]
    for r in c["rows"]:
        if not r["accepted"]:
            L.append(f"| {r['form']} | refused: {r['why'][:60]} | | | | |")
            continue
        tag = ", ".join(r["tags"]) or f"emulated ({r['emulated']})"
        w1 = "n/a" if r["t1_w1_if_untagged"] is None else f"{100 * r['t1_w1_if_untagged']:.0f}%"
        L.append(f"| {r['form']} | {tag} | {r['words_per_sm_clk']:.2f} | {r['macs_per_word']:.0f} | "
                 f"{'yes' if r['credited'] else ''} | {100 * r['t1_time_if_untagged']:.1f}% / {w1} |")
    e6 = res["e6"]
    L += ["", "## E6: separation on SI-NOISE's adder sketch", "", "| hidden words | b | separated cells |", "|---|---:|---:|"]
    for k, x in e6["cases"].items():
        L.append(f"| {k} | {x['b']} | {x['separated']}/{x['cells']} |")
    L += ["", "| K, finalization | fold ops a word | T1 in W1 | T1 at sm_120 prices |", "|---|---:|---:|---:|"]
    for k, x in e6["fold"].items():
        L.append(f"| {k} | {x['ops_per_word']:.3f} | {100 * x['t1_w1']:.1f}% | {100 * x['t1_sm120']:.1f}% |")
    e7 = res["e7"]
    L += ["", "## E7: toggles a cycle of one 4-wide MAC (area-weighted, logic + operand and accumulator flops)", "",
          "| stream | logic | flops | total | vs random s8 |", "|---|---:|---:|---:|---:|"]
    for k, x in e7["streams"].items():
        L.append(f"| {k} | {x['logic']:.0f} | {x['flops']:.0f} | {x['total']:.0f} | {x['vs_rand_s8']:.3f} |")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--f1-build", required=True)
    ap.add_argument("--f1-result", required=True)
    ap.add_argument("--synth", help="synth.py's JSON for the rho areas the wiring reads")
    ap.add_argument("--json", required=True)
    ap.add_argument("--md")
    ap.add_argument("--verdict")
    ap.add_argument("--scratch")
    ap.add_argument("--seed", type=int, default=20261030)
    ap.add_argument("--mutations", action="store_true")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    scratch = Path(a.scratch) if a.scratch else Path(tempfile.mkdtemp(prefix="si-fold-extras-"))
    res = {"closure": closure(json.loads(Path(a.f1_build).read_text()), json.loads(Path(a.f1_result).read_text()))}
    res["e6"] = e6_separation(rng)
    res["e7"] = e7_energy(rng, scratch)
    sy = json.loads(Path(a.synth).read_text())["modules"] if a.synth else {}
    if "noise_add9" in sy:
        res["noise_costs"] = noise_costs(sy["mac4_s8"]["ge"], sy["mac4_s9"]["ge"], sy["noise_add9"]["ge"])
    if "keccak400_rnd" in sy:
        dff = json.loads(Path(a.synth).read_text())["meta"]["ge_dff"]
        res["wiring"] = wiring({k: sy[k]["ge"] + sy["keccak_rcrom"]["ge"] + bits * dff
                                for k, bits in (("keccak400_rnd", 400), ("keccak1600_rnd", 1600))})
    ck = checks(res)
    if a.mutations:
        ck += [("mutation: " + n, ok) for n, ok in mutations(json.loads(Path(a.f1_build).read_text()),
                                                            json.loads(Path(a.f1_result).read_text()), scratch / "mut")]
    res["checks"] = [{"name": n, "ok": ok} for n, ok in ck]
    Path(a.json).write_text(json.dumps(res, indent=1, default=float))
    if a.md:
        Path(a.md).write_text(table(res))
    for n, ok in ck:
        print(("ok   " if ok else "FAIL ") + n)
    if a.verdict:
        Path(a.verdict).write_text(json.dumps({"ok": all(ok for _, ok in ck), "checks": len(ck),
                                               "failed": [n for n, ok in ck if not ok]}, indent=1))
    sys.exit(0 if all(ok for _, ok in ck) else 1)


if __name__ == "__main__":
    main()
