"""Energy proxy of always-on logging: area-weighted gate toggles a cycle of FT's Keccak-f[1600] round (absorbing 32
accumulator words a cycle, chained) against SI-FOLD's 4-wide s8 MAC (`mac4_s8`, the tensor MAC and an IDP4A lane),
both mapped by yosys + ABC to the same generic gates and simulated bit for bit over a stream of cycles.

    python3 gov_energy.py --fold DIR_OF_SILICON_FOLD --out OUT

DIR_OF_SILICON_FOLD is SI-FOLD's benchmarks/pouw/theory/silicon_fold (branch cursor/compute-research-silicon-fold-3c7a):
its rtl.py emits both modules, its rho.py is the Keccak reference, and its synth.py holds the gate areas.

The plane-limited unit (gov_ft.v's ft_plane4: four rounds a cycle, 8 words into plane 0 each) runs on the same words.

The proxy is zero-delay: glitches aren't counted, which undercounts multipliers more than Keccak's two-level round, so
FT's ratios here are upper bounds in that respect. ft_plane4 chains four rounds (depth 40), whose glitches are
uncounted too, so its ratios are not bounds either way. Clock pins are counted as half a data toggle for every flop
every cycle (an estimate, reported apart).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

M32 = (1 << 32) - 1
CLOCK_PER_FLOP = 0.5


def gate_fns():
    return {
        "$_NOT_": lambda p: ~p["A"], "$_BUF_": lambda p: p["A"],
        "$_AND_": lambda p: p["A"] & p["B"], "$_NAND_": lambda p: ~(p["A"] & p["B"]),
        "$_OR_": lambda p: p["A"] | p["B"], "$_NOR_": lambda p: ~(p["A"] | p["B"]),
        "$_XOR_": lambda p: p["A"] ^ p["B"], "$_XNOR_": lambda p: ~(p["A"] ^ p["B"]),
        "$_ANDNOT_": lambda p: p["A"] & ~p["B"], "$_ORNOT_": lambda p: p["A"] | ~p["B"],
        "$_MUX_": lambda p: np.where(p["S"], p["B"], p["A"]), "$_NMUX_": lambda p: ~np.where(p["S"], p["B"], p["A"]),
        "$_AOI3_": lambda p: ~((p["A"] & p["B"]) | p["C"]), "$_OAI3_": lambda p: ~((p["A"] | p["B"]) & p["C"]),
        "$_AOI4_": lambda p: ~((p["A"] & p["B"]) | (p["C"] & p["D"])),
        "$_OAI4_": lambda p: ~((p["A"] | p["B"]) & (p["C"] | p["D"])),
    }


class Netlist:
    """A generic-gate yosys netlist evaluated over a batch of cycles, ports of any width (a bool row a net)."""

    def __init__(self, path: Path, top: str, ge: dict):
        mod = json.loads(path.read_text())["modules"][top]
        self.ports = {p: v["bits"] for p, v in mod["ports"].items()}
        self.dirs = {p: v["direction"] for p, v in mod["ports"].items()}
        cells = [c for c in mod["cells"].values() if c["type"] != "$scopeinfo"]   # flatten's markers, no logic
        driver = {c["connections"]["Y"][0]: i for i, c in enumerate(cells)}
        order, state = [], {}
        for i in range(len(cells)):
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
        self.cells = [cells[i] for i in order]
        self.ge_of = ge
        self.ge = sum(ge[c["type"]] for c in self.cells)

    def run(self, inputs: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], float]:
        """inputs: port -> bool array (cycles, width). Returns output bit matrices and area-weighted toggles a cycle."""
        fn = gate_fns()
        cyc = next(iter(inputs.values())).shape[0]
        net: dict = {"0": np.zeros(cyc, bool), "1": np.ones(cyc, bool), "x": np.zeros(cyc, bool)}
        for p, m in inputs.items():
            for k, bit in enumerate(self.ports[p]):
                net[bit] = m[:, k]
        tog = 0.0
        for c in self.cells:
            y = fn[c["type"]]({k: net[b[0]] for k, b in c["connections"].items() if k != "Y"})
            net[c["connections"]["Y"][0]] = y
            tog += self.ge_of[c["type"]] * float(np.count_nonzero(y[1:] != y[:-1]))
        outs = {p: np.stack([net[b] if isinstance(b, int) else np.full(cyc, b == "1") for b in bits], axis=1)
                for p, bits in self.ports.items() if self.dirs[p] == "output"}
        return outs, tog / (cyc - 1)


def bits_of(vals: list[int], width: int) -> np.ndarray:
    nbytes = (width + 7) // 8
    raw = np.frombuffer(b"".join(v.to_bytes(nbytes, "little") for v in vals), np.uint8).reshape(len(vals), nbytes)
    return np.unpackbits(raw, axis=1, bitorder="little")[:, :width].astype(bool)


def flop_toggles(m: np.ndarray) -> float:
    return float(np.count_nonzero(m[1:] != m[:-1])) / (m.shape[0] - 1)


def build(synth, name: str, scratch: Path, extra: str = "") -> Netlist:
    d = scratch / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "m.v").write_text(extra + "\n" + synth.rtl.verilog("keccak1600_rnd") if extra else synth.rtl.verilog(name))
    synth._yosys(f"read_verilog -sv m.v; synth -flatten -top {name}; abc -g {synth.GATES}; opt_clean; "
                 f"write_json n.json", d)
    return Netlist(d / "n.json", name, synth.GE)


def llm(rng: np.random.Generator, shape, sd: float, outlier: float) -> np.ndarray:
    x = np.clip(np.rint(rng.normal(0, sd, shape)), -127, 127).astype(np.int64)
    mask = rng.random(shape) < outlier
    x[mask] = rng.choice([-1, 1], int(mask.sum())) * rng.integers(64, 128, int(mask.sum()))
    return x


def operand_streams(rng: np.random.Generator, m: int, n: int, K: int) -> dict:
    return {"rand": (rng.integers(-128, 128, (m, K)), rng.integers(-128, 128, (n, K))),
            "llm_like": (llm(rng, (m, K), 10, 0.005), llm(rng, (n, K), 24, 0.0)),
            "zero": (np.zeros((m, K), np.int64), np.zeros((n, K), np.int64))}


def mac_energy(synth, nl: Netlist, A: np.ndarray, B: np.ndarray) -> dict:
    """One 4-wide unit walks the m x n cells, K/4 cycles a cell, accumulator running (reset per cell)."""
    m, K = A.shape
    n = B.shape[0]
    a_cyc = np.concatenate([A[i].reshape(-1, 4) for i in range(m) for _ in range(n)])
    b_cyc = np.concatenate([B[j].reshape(-1, 4) for _ in range(m) for j in range(n)])
    prods = np.sum(a_cyc * b_cyc, axis=1)
    steps = K // 4
    acc_in = np.zeros(len(prods), np.int64)
    acc_out = np.zeros(len(prods), np.int64)
    for c in range(m * n):
        cs = np.cumsum(prods[c * steps:(c + 1) * steps])
        acc_out[c * steps:(c + 1) * steps] = cs
        acc_in[c * steps:(c + 1) * steps] = cs - prods[c * steps:(c + 1) * steps]
    pack = lambda x: [sum((int(v) & 0xFF) << (8 * i) for i, v in enumerate(row)) for row in x]  # noqa: E731
    ins = {"a": bits_of(pack(a_cyc), 32), "b": bits_of(pack(b_cyc), 32),
           "acc": bits_of([int(v) & M32 for v in acc_in], 32)}
    outs, tog = nl.run(ins)
    exact = bool(np.array_equal(outs["y"], bits_of([int(v) & M32 for v in acc_out], 32)))
    flops = synth.GE_DFF * (flop_toggles(ins["a"]) + flop_toggles(ins["b"]) + flop_toggles(ins["acc"]))
    clock = synth.GE_DFF * CLOCK_PER_FLOP * (32 + 32 + 32)
    return {"logic": tog, "flops": flops, "clock": clock, "data_total": tog + flops, "total": tog + flops + clock,
            "exact": exact, "cycles": len(prods), "acc_words": acc_out}


def keccak_energy(synth, nl: Netlist, words: np.ndarray, rng: np.random.Generator, plane: bool = False) -> dict:
    """FT's chain: tau <- round(tau ^ pad(32 words)), one round a cycle, the round constant cycling over 24. With
    `plane`, ft_plane4's: four rounds a cycle, each absorbing 8 of the 32 words into plane 0."""
    rho = synth.rho
    T = len(words) // 32
    st = [int.from_bytes(rng.bytes(200), "little")]
    ms, rcs = [], []
    for t in range(T):
        m = sum((int(w) & M32) << (32 * i) for i, w in enumerate(words[32 * t:32 * t + 32]))
        lanes = [(st[-1] >> (64 * i)) & ((1 << 64) - 1) for i in range(25)]
        rc_all = 0
        for k, (blk, nl_) in enumerate([(m >> (256 * k), 4) for k in range(4)] if plane else [(m, 16)]):
            for j in range(nl_):
                lanes[j] ^= (blk >> (64 * j)) & ((1 << 64) - 1)
            rc = rho.KECCAK_RC64[(4 * t + k if plane else t) % 24]
            lanes = rho.keccak_round(lanes, rc, 64)
            rc_all |= rc << (64 * k)
        st.append(sum(v << (64 * i) for i, v in enumerate(lanes)))
        ms.append(m)
        rcs.append(rc_all)
    ins = {"st": bits_of(st[:-1], 1600), "m": bits_of(ms, 1024), "rc": bits_of(rcs, 256 if plane else 64)}
    outs, tog = nl.run(ins)
    exact = bool(np.array_equal(outs["y"], bits_of(st[1:], 1600)))
    flops = synth.GE_DFF * (flop_toggles(ins["st"]) + flop_toggles(ins["m"]))
    clock = synth.GE_DFF * CLOCK_PER_FLOP * (1600 + 1024)
    return {"logic": tog, "flops": flops, "clock": clock, "data_total": tog + flops, "total": tog + flops + clock,
            "exact": exact, "cycles": T}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True, help="SI-FOLD's benchmarks/pouw/theory/silicon_fold directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--K", type=int, default=256)
    a = ap.parse_args()
    sys.path.insert(0, a.fold)
    import synth  # SI-FOLD's: rtl, rho, gate areas, yosys

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    scratch = out / "scratch"
    rng = np.random.default_rng(20261031)
    mac, kec = build(synth, "mac4_s8", scratch), build(synth, "keccak1600_rnd", scratch)
    kp4 = build(synth, "ft_plane4", scratch, extra=(Path(__file__).resolve().parent / "gov_ft.v").read_text())
    res: dict = {"ge": {"mac4_s8": mac.ge, "keccak1600_rnd": kec.ge, "ft_plane4": kp4.ge}, "streams": {},
                 "clock_per_flop": CLOCK_PER_FLOP}
    ok = True
    for name, (A, B) in operand_streams(rng, a.m, a.n, a.K).items():
        me = mac_energy(synth, mac, A, B)
        words = me.pop("acc_words")
        words = words[: (len(words) // 32) * 32]
        ke = keccak_energy(synth, kec, words, rng)
        k4 = keccak_energy(synth, kp4, words, rng, plane=True)
        ok &= me["exact"] and ke["exact"] and k4["exact"]
        row = {"mac4": me, "keccak_round": ke, "ft_plane4": k4}
        for basis in ("data_total", "total"):
            row[f"ft_over_mac_{basis}"] = ke[basis] / (256 * me[basis]) if me[basis] else None
            row[f"rt_over_idp4a_warp_{basis}"] = ke[basis] / (32 * me[basis]) if me[basis] else None
            row[f"ft4_over_mac_{basis}"] = k4[basis] / (256 * me[basis]) if me[basis] else None
            row[f"rt4_over_idp4a_warp_{basis}"] = k4[basis] / (32 * me[basis]) if me[basis] else None
        res["streams"][name] = row
        print(f"{name:9s} mac4 logic {me['logic']:7.1f} flops {me['flops']:6.1f} clock {me['clock']:6.1f} | "
              f"keccak logic {ke['logic']:8.1f} flops {ke['flops']:7.1f} clock {ke['clock']:7.1f} | "
              f"FT/MAC data {row['ft_over_mac_data_total'] if row['ft_over_mac_data_total'] is not None else float('nan'):.4f} "
              f"with clock {row['ft_over_mac_total']:.4f} | RT/IDP4A-warp data "
              f"{row['rt_over_idp4a_warp_data_total'] if row['rt_over_idp4a_warp_data_total'] is not None else float('nan'):.3f} "
              f"with clock {row['rt_over_idp4a_warp_total']:.3f} | plane4 logic {k4['logic']:8.1f} FT4/MAC with clock "
              f"{row['ft4_over_mac_total']:.4f} RT4/IDP4A-warp {row['rt4_over_idp4a_warp_total']:.3f}  "
              f"exact {me['exact'] and ke['exact'] and k4['exact']}", flush=True)
    res["netlists_exact"] = ok
    (out / "gov_energy.json").write_text(json.dumps(res, indent=1, default=float))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
