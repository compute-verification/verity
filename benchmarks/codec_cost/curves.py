"""The honest cost of each codec measured on the RTX PRO 6000, and the two curves the best-overhead model needs.

Nothing is measured here; every input is a recorded output (`research data fetch` the arts named):
- --table   codec_mix's codec-table-v5.json (art:b839fbd0979f4b2a): d at 8B B80 ctx 8,256 and 70B B8 ctx 9,280 by its
  step model (`benchmarks/codec_mix/table.py`), κ, x₁ and the margins, at 512-1,024 bits.
- --kv      word_codec's kv-codec.json (art:c1625f3cebececb0, r20261012-183859-2055): kv.cu's decode rates and the
  read-and-decode of a 2 GiB store over the plain read, plain and masked, k = 2 and 3.
- --tensor  codec_tensor's rates.json (art:bf586ded531b3f1a): squaring rates and x at 255-2,048 bits.
- --attack  codec_mix's mask_attack.jsonl (art:ffca24ef70aaff39): σ_K's floor margin at 512-2,048 bits.
- --cpu     cpu_lat.sh's $OUT (repeated): one word's dependent squaring chain on one core, and that core's clock.
- --gpu     gpu_lat.sh's $OUT: the same chain on one GPU thread, and the SM clock.
- --wide    wide_decode.sh's codec-tensor.json (optional): wt.cu's store decode and rates at 1,536 to 4,096 bits.

The model, step by step:
- Decode, a word at k rounds: kv.cu's rates at 512, 768 and 1,024 bits, linear in k past k = 3; with --wide, wt.cu's
  at k = 1 and 2 and k = 2's half a round past it; elsewhere t0·b + k·α/s(b), s(b) codec_tensor's squaring rate and
  α, t0 fitted to kv.cu. A decode's cost in reads per read is the plain read's rate over the store decode's in the
  same run, at least 1: the decode overlaps the read.
- d: table v5's where it has the codec, else codec_mix's `row` at the codec's κ (decode time a byte on p512's k axis),
  attention scaled by measured over projected as the nearest built width's.
- x per round, the adversary's: x₁ at the max clock (175 at 512, scaled by each width's squaring rate) over the stated
  factor a width (squaring analogue × the host's IFMA share; `timing.py`'s at 512 and 1,024). μ_rebuild is frontier.py's
  (d − τ)/x at κ = 1 and κ/x below it.
- t_e: k roots one after another (a round needs the last), each the root's squarings plus 40 squaring-equivalents.
  A squaring is the fewest cycles measured over cores and forms, at 5.7 GHz, or the IFMA issue floor of P2's re-gate.
  At these widths cores don't cooperate: an exchange costs a core-to-core round trip, more than the squaring.

`python curves.py --table T --kv KV --tensor R --attack A --cpu DIR [--cpu DIR] --gpu DIR [--wide J] [--json OUT]`
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import sys
from pathlib import Path


def _load(rel: str):
    spec = importlib.util.spec_from_file_location(rel.replace("/", "_")[:-3], Path(__file__).resolve().parents[1] / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MIX = _load("codec_mix/table.py")
P2 = _load("pous/fast_decode/model.py")

GHZ = 5.7               # Zen 5 desktop's max clock, P2's floor core (p2/PROTOCOL.md §12)
SM_MAX_MHZ = 2430.0     # the RTX PRO 6000's max SM clock
BOOST = 1.16            # frontier.py's: the max clock over the locked 2,092 MHz the GPU rates were timed at
BUDGET = 0.005          # μ_rebuild's share of capacity
TAU_TAG = 0.041         # a sketch receipt's tag share (frontier point, p512 70B B1, art:359729b1605199c2)
MULT_EQ = 40            # a root's multiplications, in squarings (P2's re-gate)
KAPPAS = (1.0, 0.5, 0.1, 0.01)
TAUS_US = (50, 100, 150, 300, 840)
#: The stated adversary factor a width: the squaring analogue times the host's IFMA share (ra-squaring-ceiling:
#: timing.py's GPU_SQUARINGS_PER_S at 512 and 1,024, wide.adversaries at 768). Other widths take the nearest.
FACTOR = {512: 1.567, 768: 1.562, 1024: 1.597}
#: x a round, measured encodes of word_codec's narrow builds against its p512 in the same arithmetic (p256 97 and p512
#: 326 in one run, art:77480321bd7ebe95; p127 32, art:96b9bcd36ce70e8e; p255 110, art:5fcba114c338a34d).
NARROW_X = {127: 32 / 326, 255: 110 / 326, 256: 97 / 326}
#: (name, n, c): the primes the codec lanes ran on the RTX PRO 6000.
FIELDS = [("p127", 127, 1), ("p255", 255, 19), ("p256", 256, 189), ("p511", 511, 481), ("p512", 512, 569),
          ("m521", 521, 1), ("p767", 767, 1081), ("p768", 768, 825), ("p1023", 1023, 361), ("p1024", 1024, 105),
          ("m1279", 1279, 1), ("p1536", 1536, 3453), ("p2048", 2048, 1557)]
#: Wider Sloth fields wide_decode.sh adds to a scratch copy of codec_tensor (c the least with p prime and p = 3 mod 4).
WIDE_C = {3072: 2165, 4096: 2549}
#: P2 v4 at 16,448 bits and its maps at narrower widths (ra-fast-decode, art:4ba37c436c467ec5): the best full pass
#: with the keys, GB/s; one root a 2,054-byte block.
P2_DECODE = {16448: P2.MEASURED["cuda_core_full_GBps"], 8256: 45.3, 4160: 53.4, 2112: 52.5}
P2_C = {16448: 21065, 8256: 7305, 4160: 1017, 2112: 2397}
#: P2's SeqRoot run: 8 cores of one EPYC 4564P chiplet, 778.5 ns a squaring at 5.475 GHz (model.py's COOP).
P2_SEQROOT_NS = 778.5 * 5.475 / GHZ
#: Security, from the red-team lanes (codec_mix table v5 and its notes). None: no priced margin.
PLAIN_K2_LOW = {b: v[1] for b, v in MIX.PLAIN_K2.items()}


def squarings_per_root(n: int, c: int) -> int:
    return (((1 << n) - c + 1) // 4).bit_length() - 1


def factor(b: int) -> float:
    return FACTOR[min(FACTOR, key=lambda w: abs(w - b))]


# --- the IFMA issue floor (P2's re-gate: Zen 5, 512-bit IFMA, two a cycle, four vector ops a cycle) ---------------
SPLIT = {"K": (2, 3, 1.06), "T3": (3, 5, 3.25), "T4": (4, 7, 6.1)}


def _plan_cost(n: int, plan: tuple) -> tuple[float, float]:
    if not plan:
        return n * (n + 1) / 8, n / 2
    k, m, alpha = SPLIT[plan[0]]
    i, o = _plan_cost(-(-n // k) + (plan[0] != "K"), plan[1:])
    return m * i, m * o + alpha * n


def floor_ns(w: int) -> float:
    """min(IFMA issue floor, mulx's one product a cycle) for one w-bit squaring at 5.7 GHz, each with the re-gate's
    25 cycles of fold-and-carry latency (the re-gate leaves it off mulx, which only matters below about 512 bits)."""
    n = -(-w // 52)
    best = math.inf
    for d in range(6):
        for plan in itertools.product(SPLIT, repeat=d):
            size, ok = n, True
            for s in plan:
                size = -(-size // SPLIT[s][0])
                ok &= size >= 3
            if ok:
                i, o = _plan_cost(n, plan)
                best = min(best, max((i + n / 4) / 2, (i + n / 4 + o + n) / 4) + 25)
    m = -(-w // 64)
    mulx = m * (m + 1) // 2
    for d in range(1, 6):
        for plan in itertools.product(SPLIT, repeat=d):
            size, mult = m, 1
            for s in plan:
                size = -(-size // SPLIT[s][0]) + (s != "K")
                mult *= SPLIT[s][1]
            if size >= 2:
                mulx = min(mulx, mult * size * (size + 1) // 2)
    return min(best, mulx + 25) / GHZ


# --- inputs ----------------------------------------------------------------------------------------------------------
def jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def cpu_cycles(dirs: list[Path]) -> dict[int, dict]:
    """The fewest cycles a squaring at each width over every verified chain of every run."""
    best: dict[int, dict] = {}
    for d in dirs:
        ghz = max(c["ghz_max"] for c in jsonl(d / "clock.jsonl"))
        for f in ("mpn", "u128", "ifma", "owner"):
            for r in jsonl(d / f"{f}.jsonl"):
                if not r.get("verified"):
                    continue
                cyc = r["sqr_min_ns"] * ghz
                if r["W"] not in best or cyc < best[r["W"]]["cycles"]:
                    best[r["W"]] = {"cycles": cyc, "form": f + (f" T{r['T']}" if "T" in r else ""), "ghz": ghz,
                                    "ns": r["sqr_min_ns"], "run": d.parent.name if d.name == "out" else d.name}
    return best


def gpu_latency(d: Path) -> dict[int, dict]:
    mhz = min(c["clock_mhz"] for c in jsonl(d / "clock.jsonl"))
    best: dict[int, dict] = {}
    for r in jsonl(d / "lat.jsonl"):
        if r["bits"] not in best or r["sqr_min_ns"] < best[r["bits"]]["ns"]:
            best[r["bits"]] = {"ns": r["sqr_min_ns"], "fold": r["fold"], "mhz": mhz,
                               "ns_max_clock": r["sqr_min_ns"] * mhz / SM_MAX_MHZ}
    return best


class Decode:
    """A word's decode time (s) at k rounds, and its cost in reads per read."""

    def __init__(self, kv: dict, tensor: list[dict], wide: dict | None = None):
        self.read = kv["read"]["gbps"]
        self.kv = {lib: v for lib, v in kv["libs"].items()}
        self.sq = {r["n"]: r["squarings_s_cuda"] for r in tensor}
        self.bytes = {r["n"]: 1 / r["words_per_plain_byte"] for r in tensor}
        # wt.cu's standalone store decode (wide_decode.sh): {(n, masked): {k: (store GB/s, chain words/s)}} at k = 1, 2
        self.wide, self.wide_read = {}, wide["read"]["gbps"] if wide else None
        for name, e in (wide or {}).get("libs", {}).items():
            rates = e.get("rates", {}).get("decode", {})
            ks = {int(k): (v["cuda"]["gbps_stored"], rates[k]["per_s"])
                  for k, v in e.get("decode", {}).items() if v.get("cuda", {}).get("ok") and k in rates}
            if ks:
                self.wide[(e["constants"]["n"], name.endswith("-masked"))] = ks
        fits = [self._fit(lib) for lib in ("p512", "p768", "p1024")]
        self.alpha = sum(f[1] * self.kv[lib]["rates"]["square_s"] for f, lib in zip(fits, ("p512", "p768", "p1024"))) / 3
        self.t0_bit = sum(f[0] / b for f, b in zip(fits, (512, 768, 1024))) / 3
        self.mask = sum(self.time_kv(f"{lib}-m", k) / self.time_kv(lib, k)
                        for lib in ("p512", "p768", "p1024") for k in (2, 3)) / 6

    def _fit(self, lib: str) -> tuple[float, float]:
        r = self.kv[lib]["rates"]["decode"]
        t2, t3 = 1 / r["2"]["per_s"], 1 / r["3"]["per_s"]
        return t2 - 2 * (t3 - t2), t3 - t2

    def time_kv(self, lib: str, k: int) -> float:
        t0, tr = self._fit(lib)
        return t0 + k * tr

    def lib(self, b: int, masked: bool) -> str | None:
        name = f"p{b}" + ("-m" if masked else "")
        return name if name in self.kv else None

    def word_bytes(self, b: int) -> float:
        return b / 8 if self.lib(b, False) else self.bytes.get(b, b // 8)

    def squarings_s(self, b: int) -> float:
        if b in self.sq:
            return self.sq[b]
        near = min(self.sq, key=lambda w: abs(w - b))
        return self.sq[near] * (near / b) ** 1.75

    def time(self, b: int, k: int, masked: bool) -> tuple[float, bool]:
        """(seconds a word, measured?)"""
        lib = self.lib(b, masked)
        if lib:
            return self.time_kv(lib, k), k in (2, 3)
        if (b, masked) in self.wide:   # wt.cu runs every round past the first in one loop: k = 2's half a round
            w = self.wide[(b, masked)]
            return (1 / w[k][1], True) if k in w else (k / (2 * w[2][1]), False)
        t = self.t0_bit * b + k * self.alpha / self.squarings_s(b)
        return t * (self.mask if masked else 1.0), False

    def gbps(self, b: int, k: int, masked: bool) -> float:
        lib = self.lib(b, masked)
        if lib and str(k) in self.kv[lib].get("store", {}):
            return self.kv[lib]["store"][str(k)]["decode_gbps"]
        if (b, masked) in self.wide:
            w = self.wide[(b, masked)]
            return w[k][0] if k in w else 2 * w[2][0] / k
        return self.word_bytes(b) / self.time(b, k, masked)[0] / 1e9

    def reads(self, b: int, k: int, masked: bool) -> float:
        lib = self.lib(b, masked)
        if lib and str(k) in self.kv[lib].get("store", {}):
            return self.kv[lib]["store"][str(k)]["decode_over_read"]
        if (b, masked) in self.wide:
            return max(1.0, self.wide_read / self.gbps(b, k, masked))
        return max(1.0, self.read / self.gbps(b, k, masked))

    def model_reads(self, b: int, k: int, masked: bool) -> float:
        """kv.cu's decode carried to width b (t0·b + k·α/s(b)), whatever was measured there."""
        t = (self.t0_bit * b + k * self.alpha / self.squarings_s(b)) * (self.mask if masked else 1.0)
        return max(1.0, self.read / (self.word_bytes(b) / t / 1e9))

    def kappa(self, b: int, k: int, masked: bool) -> float:
        per_byte = self.time(b, k, masked)[0] / self.word_bytes(b)
        t2, t3 = (self.time_kv("p512", j) / 64 for j in (2, 3))
        return 2 + (per_byte - t2) / (t3 - t2)


# --- the table and the curves ----------------------------------------------------------------------------------------
def x_ref(b: int, v5: dict, tensor: dict) -> float:
    """x₁ at the max clock: 175 at 512, scaled by the fastest measured build at each width."""
    if b in (512, 768, 1024):
        return next(r["x1_adv"] for r in v5["rows"] if r["b"] == b)
    def root_s_a_byte(r: dict) -> float:     # x_cuda without the run's read rate, which differs between runs
        return r["words_per_plain_byte"] * (r["n"] - 2) / r["squarings_s_cuda"]

    cands = []
    if b in tensor:
        cands.append(MIX.X1_P512 * root_s_a_byte(tensor[b]) / root_s_a_byte(tensor[512]))
    if b in NARROW_X:
        cands.append(MIX.X1_P512 * NARROW_X[b])
    return min(cands)


V5_NAMES = {(512, 1, False): "p512 k=1", (512, 2, False): "p512 k=2 plain", (512, 2, True): "p512 k=2 masked",
            (512, 3, False): "p512 k=3 plain", (512, 3, True): "p512 k=3 masked", (768, 2, False): "p768 k=2 plain",
            (768, 2, True): "p768 k=2 masked (σ_K)", (768, 3, True): "p768 k=3 masked",
            (1024, 2, False): "p1024 k=2 plain", (1024, 2, True): "p1024 k=2 masked (σ_K)",
            (1024, 3, True): "p1024 k=3 masked"}


def step_d(b: int, k: int, masked: bool, v5: dict, dec: Decode, xr: float) -> tuple[dict, str]:
    """d at the two step points, and whether it is v5's (built) or projected here."""
    name = V5_NAMES.get((b, k, masked))
    if name:
        r = next(r for r in v5["rows"] if r["codec"].startswith(name))
        return {p: v["d"] for p, v in r["points"].items()}, "v5"
    lib = f"p{512 if b <= 640 else 1024 if b >= 900 else 768}" + ("-m" if masked else "")
    ratio = {(r["codec"].split()[0] + ("-m" if "masked" in r["codec"] else ""), r["k"]):
             r["attn_b160_4k"]["measured"] / r["attn_b160_4k"]["projected"] for r in v5["rows"] if r.get("attn_b160_4k")}
    ratio |= {(c["lib"], c["k"]): c["measured"] / c["projected"] for c in v5["check"]}
    scale = ratio.get((lib, min(max(k, 2), 3)))
    attn = {"measured": scale, "projected": 1.0} if scale else None
    r = MIX.row("", b, k, fr=None, kappa=dec.kappa(b, k, masked), x1=xr, margin=None, note="", attn=attn)
    return {p: v["d"] for p, v in r["points"].items()}, "projected"


def security(b: int, k: int, masked: bool, margins: dict) -> tuple[float | None, str]:
    """(low margin or None, caveat)"""
    if k == 1:
        return 0.0, "broken: a word is partly kept and completed for less than an encode (F27, the 2D-lattice peel)"
    if masked and f"p{b}" in margins:
        low = 1.33 if k >= 3 else 1.0
        return low, (f"σ_K peel floor {margins[f'p{b}']:.2f}, free Coppersmith to b/2 → 1.0" if k == 2 else
                     "peel two roots: 2.0 (1.33)")
    if not masked and k == 2 and b in PLAIN_K2_LOW:
        return PLAIN_K2_LOW[b], f"red team's 4D floor {MIX.PLAIN_K2[b][0]} (low {PLAIN_K2_LOW[b]})"
    if not masked and k >= 3:
        return None, "unpriced: no degree-8 completion measured"
    return None, "unpriced: no partial-completion attack run at this width"


def build(a: argparse.Namespace) -> dict:
    v5 = json.loads(Path(a.table).read_text())
    kv = json.loads(Path(a.kv).read_text())
    tensor_list = json.loads(Path(a.tensor).read_text())
    wide = json.loads(Path(a.wide).read_text()) if a.wide else None
    known = {r["n"] for r in tensor_list}
    for name, e in (wide or {}).get("libs", {}).items():   # codec_tensor's rates rows for widths it hadn't run
        if not name.endswith("-masked") and e["constants"]["n"] not in known and "rates" in e:
            tensor_list.append({"n": e["constants"]["n"], "words_per_plain_byte": e["analysis"]["words_per_plain_byte"],
                                "squarings_s_cuda": e["analysis"]["squarings_s"]["cuda"], "run": a.wide})
    tensor = {r["n"]: r for r in tensor_list}
    margins = {d["field"]: d["margin"] for d in jsonl(Path(a.attack)) if d["experiment"] == "floor"}
    dec = Decode(kv, tensor_list, wide)
    cpu = cpu_cycles([Path(d) for d in a.cpu])
    gpu = gpu_latency(Path(a.gpu))
    fields = FIELDS + [(f"p{n}", n, WIDE_C[n]) for n in sorted(WIDE_C) if (n, False) in dec.wide]

    def t_sq(w: int) -> dict:
        """A width no CPU run measured takes the nearest measured one's cycles, scaled by the floor's ratio."""
        m = cpu.get(w)
        if m:
            return {"measured_ns": m["cycles"] / GHZ, "floor_ns": floor_ns(w), "cpu": m}
        near = min(cpu, key=lambda v: abs(math.log(v / w)))
        scaled = dict(cpu[near], cycles=cpu[near]["cycles"] * floor_ns(w) / floor_ns(near), scaled_from=near)
        return {"measured_ns": scaled["cycles"] / GHZ, "floor_ns": floor_ns(w), "cpu": scaled}

    rows = []
    for name, b, c in fields:
        xr = x_ref(b, v5, tensor)
        sq = squarings_per_root(b, c)
        ts = t_sq(b)
        words = 4096 / dec.word_bytes(b)
        for k, masked in [(1, False), (2, False), (2, True), (3, False), (3, True)]:
            d, d_src = step_d(b, k, masked, v5, dec, xr)
            low, caveat = security(b, k, masked, margins)
            rows.append({
                "codec": f"{name} k={k}" + (" masked" if masked else ""), "b": b, "c": c, "k": k, "masked": masked,
                "decode_gbps": dec.gbps(b, k, masked), "reads_per_read": dec.reads(b, k, masked),
                "reads_model": dec.model_reads(b, k, masked),
                "decode_measured": dec.time(b, k, masked)[1], "kappa": dec.kappa(b, k, masked),
                "d": d, "d_source": d_src, "x_ref_round": xr, "x_adv_round": xr / factor(b), "x_adv": k * xr / factor(b),
                "squarings_root": sq, "squarings_block": round(words * k * sq),
                "root_us_measured": (sq + MULT_EQ) * ts["measured_ns"] / 1e3 if ts["measured_ns"] else None,
                "root_us_floor": (sq + MULT_EQ) * ts["floor_ns"] / 1e3,
                "root_us_gpu": (sq + MULT_EQ) * gpu[b]["ns_max_clock"] / 1e3 if b in gpu else None,
                "cpu": ts["cpu"], "margin_low": low, "caveat": caveat})
    p2rows = []
    for w, gb in sorted(P2_DECODE.items()):
        sq = squarings_per_root(w, P2_C[w])
        ts = t_sq(w)
        floor_root = P2.seqroot(w)["coop_root_us"] if w == P2.W0 else (sq + MULT_EQ) * floor_ns(w) / 1e3
        meas = [(sq + MULT_EQ) * ts["measured_ns"] / 1e3] if ts["measured_ns"] else []
        if w == P2.W0:
            meas.append((w - 2 + MULT_EQ) * P2_SEQROOT_NS / 1e3)
        enc_x = None
        if w == P2.W0:
            squarings_s = 2 * P2.MEASURED["cuda_core_kernel_GBps"] * 1e9 / P2.PAYLOAD
            enc_x = P2.MEASURED["read_GBps"] * (sq + MULT_EQ) / (squarings_s * P2.PAYLOAD / 1e9) / BOOST
        p2rows.append({"codec": f"P2 v4 {w:,}" if w == P2.W0 else f"P2 map {w:,}", "b": w, "k": 1,
                       "decode_gbps": gb, "reads_per_read": P2.MEASURED["read_GBps"] / gb,
                       "root_us_floor": floor_root, "root_us_measured": min(meas) if meas else None,
                       "squarings_root": sq, "cpu": ts["cpu"], "x_ref_round": enc_x})

    curve_i = []
    cands = []
    for b in sorted({f[1] for f in fields}):
        for k in range(2, 7):
            for masked in (True, False):
                low, caveat = security(b, k, masked, margins)
                if low is not None and low < 1.0 and not (b == 512 and k == 2):
                    continue
                xr = x_ref(b, v5, tensor)
                d, src = step_d(b, k, masked, v5, dec, xr)
                cands.append({"codec": f"p{b} k={k}" + (" masked" if masked else ""), "b": b, "k": k,
                              "masked": masked, "d": d, "d_source": src, "x_adv": k * xr / factor(b),
                              "reads_per_read": dec.reads(b, k, masked), "margin_low": low})
    for r in v5["rows"]:
        if not r["built"] and r["margin"] and r["margin"][1] >= 1.0 and "permutation" not in r["codec"]:
            cands.append({"codec": r["codec"], "b": r["b"], "k": r["k"], "masked": True,
                          "d": {p: v["d"] for p, v in r["points"].items()}, "d_source": "v5 projected",
                          "x_adv": r["k"] * r["x1_adv"] / factor(r["b"]), "margin_low": r["margin"][1],
                          "reads_per_read": dec.reads(r["b"], r["k"], True) if r["b"] != 640 else None})
    priced = lambda c: c["margin_low"] is not None and c["margin_low"] >= 1.0
    tiers = {"priced": lambda c: priced(c) and c["d_source"] != "v5 projected",
             "with plain p512 k=2": lambda c: c["margin_low"] is not None and c["d_source"] != "v5 projected",
             "with v5's projected layouts": priced,
             "with unpriced widths": lambda c: c["d_source"] != "v5 projected"}
    for kappa in KAPPAS:
        out = {"kappa": kappa}
        for tier, admit in tiers.items():
            pick = {}
            for p in MIX.STEP:
                ok = [c for c in cands if admit(c) and
                      ((c["d"][p] - TAU_TAG) if kappa == 1.0 else kappa) / c["x_adv"] <= BUDGET]
                if ok:
                    best = min(ok, key=lambda c: c["d"][p])
                    pick[p] = {"codec": best["codec"], "d": best["d"], "d_source": best["d_source"],
                               "reads_per_read": best["reads_per_read"], "x_adv": best["x_adv"],
                               "mu": ((best["d"][p] - TAU_TAG) if kappa == 1.0 else kappa) / best["x_adv"]}
            out[tier] = pick
        curve_i.append(out)

    curve_ii = []
    sloth = []
    for b in [512, 768, 1024, 1536, 2048] + [n for n in sorted(WIDE_C) if (n, True) in dec.wide]:
        r = next(x for x in rows if x["b"] == b)
        for k in range(2, 400):
            sloth.append({"codec": f"p{b} k={k} masked", "k": k, "reads_per_read": dec.reads(b, k, True),
                          "reads_model": dec.model_reads(b, k, True),
                          "decode_measured": dec.time(b, k, True)[1], "t_e_floor": k * r["root_us_floor"],
                          "t_e_measured": k * r["root_us_measured"]})
    allc = sloth + [{"codec": p["codec"], "k": 1, "reads_per_read": p["reads_per_read"], "decode_measured": True,
                     "t_e_floor": p["root_us_floor"], "t_e_measured": p["root_us_measured"]} for p in p2rows]
    for tau in TAUS_US:
        out = {"tau_us": tau}
        for reading in ("floor", "measured"):
            ok = [c for c in allc if c[f"t_e_{reading}"] and c[f"t_e_{reading}"] > 2 * tau]
            best = min(ok, key=lambda c: c["reads_per_read"])
            out[reading] = best
            out[f"{reading}_runner_up"] = min((c for c in ok if c["codec"].split(" k=")[0] != best["codec"].split(" k=")[0]),
                                              key=lambda c: c["reads_per_read"], default=None)
        curve_ii.append(out)
    return {"rows": rows, "p2": p2rows, "gpu": gpu, "curve_i": curve_i, "curve_ii": curve_ii,
            "decode_fit": {"alpha": dec.alpha, "t0_ns_per_bit": dec.t0_bit * 1e9, "mask": dec.mask, "read": dec.read},
            "check_fit": {f"p{b} k{k}": {"kv": dec.time_kv(f"p{b}", k) * 1e9,
                                        "model": (dec.t0_bit * b + k * dec.alpha / dec.squarings_s(b)) * 1e9}
                          for b in (512, 768, 1024) for k in (2, 3)}}


def fmt(x, nd=2):
    return "—" if x is None else f"{x:,.{nd}f}"


def render(t: dict) -> str:
    p8, p70 = list(MIX.STEP)
    out = [("| codec | decode GB/s | reads/read | d 8B / 70B | x a round (adv.) | squarings a block | t_root CPU "
            "measured / floor, µs | t_e = k roots, measured / floor, µs | t_root GPU thread, µs | margin (low) |"),
           "|" + "---|" * 10]
    for r in t["rows"]:
        scaled = "†" if r["cpu"] and r["cpu"].get("scaled_from") else ""
        t_e = (f"{fmt(r['k'] * r['root_us_measured'], 1)}{scaled} / {fmt(r['k'] * r['root_us_floor'], 1)}"
               if r["root_us_measured"] else "—")
        out.append(f"| {r['codec']} | {r['decode_gbps']:.0f}{'' if r['decode_measured'] else '*'} | "
                   f"{r['reads_per_read']:.2f} | {r['d'][p8]:.2f} / {r['d'][p70]:.2f}{'' if r['d_source'] == 'v5' else '*'}"
                   f" | {r['x_adv_round']:.1f} | {r['squarings_block']:,} | {fmt(r['root_us_measured'])}{scaled} / "
                   f"{fmt(r['root_us_floor'])} | {t_e} | {fmt(r['root_us_gpu'])} | {fmt(r['margin_low'])} |")
    for p in t["p2"]:
        x = p["x_ref_round"] / factor(p["b"]) if p["x_ref_round"] else None
        out.append(f"| {p['codec']} | {p['decode_gbps']:.1f} | {p['reads_per_read']:.1f} | — | {fmt(x, 0)} | "
                   f"{p['squarings_root']:,} a root | {fmt(p['root_us_measured'], 0)} / {fmt(p['root_us_floor'], 0)} | "
                   f"(k = 1) | — | P2's |")
    out += ["", "| κ | candidates | point | codec | d 8B / 70B | reads/read | x adv | μ_rebuild |",
            "|---|---|---|---|---|---|---|---|"]
    for row in t["curve_i"]:
        for tier, pick in row.items():
            if tier == "kappa":
                continue
            for p, c in pick.items():
                out.append(f"| {row['kappa']} | {tier} | {p} | {c['codec']} | {c['d'][p8]:.3f} / {c['d'][p70]:.3f} | "
                           f"{fmt(c['reads_per_read'])} | {c['x_adv']:.0f} | {100 * c['mu']:.3f}% |")
    out += ["", "| τ µs | floor: codec, reads/read, t_e µs | measured: codec, reads/read, t_e µs |", "|---|---|---|"]
    for row in t["curve_ii"]:
        f, m = row["floor"], row["measured"]
        out.append(f"| {row['tau_us']} | {f['codec']}, {f['reads_per_read']:.2f}, {f['t_e_floor']:,.0f} | "
                   f"{m['codec']}, {m['reads_per_read']:.2f}, {m['t_e_measured']:,.0f} |")
    return "\n".join(out)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--table", required=True)
    ap.add_argument("--kv", required=True)
    ap.add_argument("--tensor", required=True)
    ap.add_argument("--attack", required=True)
    ap.add_argument("--cpu", action="append", required=True)
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--wide", help="wide_decode.sh's codec-tensor.json: wt.cu's store decode at 1,536 to 4,096 bits")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    t = build(a)
    if a.json:
        Path(a.json).write_text(json.dumps(t, indent=1, default=str))
    print(render(t))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
