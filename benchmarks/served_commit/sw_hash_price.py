"""DistinctWrites' interior commitment options in Llama-3.2-1B's served step on the H200, priced: the bytes a useful MAC, t_H/t_S
at T 8,192 and T 512, the T1 terms each premise leaves, the hiding work only the honest prover does, and the precision ShaSpeed
needs for T1 ≤ 1%. Stdlib only, no GPU.

    python3 benchmarks/served_commit/sw_hash_price.py SASS_JSON ROWS_JSON OUT_JSON

SASS_JSON is sha512_sass.py's output for the committer's build (a compression's instructions and W1 work at the floor and
alone); ROWS_JSON is sw_hash_rows.py's (each row size's leaves alone, GPU time by kernel). The step, NCP's GEMMs and the
16 KiB hash rate measured in the served volume are ncp_k32_price.py's.

An option commits, of each dot product of NCP's 3K-wide stacked product, every L-th accumulator word at w bits (64/w packed
to a 64-bit word; L 32 at w 32 is k32), and optionally the formed operands X (3K bytes an activation row) and Y (3K·n a
call) and the noise E₁, F₁ (K and K·n), into hm96 rows of R bytes. Every option also commits each call's activation
operand A (K bytes an activation row), whose root seeds the call's noise, so it is hashed before the noise. A forward
hashes B bytes in B / rate(R): rate(16 KiB) is ncp_k32_price's, and rate(R) scales it by ROWS_JSON's time a byte at R over
16 KiB's, the tree taken as 3 compressions a leaf at the data kernel's rate. "At the bar" scales every hash time by the
bar's MAC-times a compression over the data kernel's. Schedules: serial (the step, NCP's GEMM work and the hash one after
another, the words reaching the hash for free); export (serial plus each committed register's move out of the accumulator
to shared memory, and the pack's instructions, at the k32 export kernel's measured extra a register over the wgmma's own);
unfused (serial plus the same move to HBM instead); overlap (the hash wholly hidden beside the step and the export: the
larger of the two, a lower bound for async, beside-the-GEMM and fused-epilogue hashing alike).

T1 terms, MAC-weighted over the linears and additive over stock's m·n·K (W1 units, one dense int8 MAC = 1):
  interior under H32 alone: NCP's tensor work (3) plus the honest pack instructions less 32 a committed register a MAC;
  interior under SemK32 (the semantics premise that every k32 word is computed once any L-th is committed, and for a
  packed width that one accumulator register holds one dot product, so the packed register is a target): the pack
  instructions less 32 a packed register (0 unpacked);
  final word: 32/K, or L/K under SemK32 at L > 32 (0 with a final-word design, L ≤ 7K/8);
  forming: f/n + f/m at f = 24, closed when the formed operands are committed (8 an entry, forced at 32 a 4-code register);
  noise derivation: 8/n + 8/m, closed when E₁ and F₁ are committed;
  hiding: per leaf the salt kernel's time less the frame leaf's first block, which a prover with one fixed salt skips, as a
  share of the hash's time, times the hash's W1 work a MAC (priced as compression-time, an estimate). ShaSpeed credits it
  only where the drawn unit's proof checks y = PRF_ks(pos) and c; then M_key y alone remains, under no premise (the salt
    kernel's time from given salts less c's two compressions and the frame leaf's first block).
  pre-noise hashing: A's commitment, computed before the call's noise, which a model whose free set is every function of
  the data fixed before the noise (Theorem D's) doesn't charge the adversary: A's bytes a useful MAC times the hash's work
  a byte, in W1 (floor, alone, and the hash's MAC-times a byte) and in AND gates.
ShaSpeed's precision δ: 0.01 / (the hash's W1 work a useful MAC), at the floor and alone, and as instructions a compression.

In AND gates (hm96's AND model: 58,120 a SHA-512 compression, 53,568 one from round 8, XORs free; a MAC is main's Boolean
lowering of an s8 x s8 linear into int32, 2,079/16 ANDs) the hash is (row blocks + 12) compressions a leaf, the hiding work
the salt PRF and c (M_key y is XORs), and stock one MAC's ANDs a useful MAC.

Per monitoring period (week, month, year) the shares are pooled over the stock work they cover, on two bases: the work
held fixed (the reference datacenter's H200s at full capacity at the data sheet's int8 rate, and the world's inference
FLOP at 2 a MAC), where a design needs t_H/t_S times the hardware; and the hardware held fixed, where it covers stock's
work over t_H/t_S.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ncp_k32_price as N  # noqa: E402

BAR_RATE = 0.01803                                  # the bar: compressions of 128 B an SM-clock
BAR_MAC_TIMES = 4096 / BAR_RATE
FORM, NOISE = 24, 8                                 # W1 units an operand entry: ncp-v1's forming, and deriving a noise entry
PACK_INSTR = {32: 0, 16: 1, 64 / 3: 2}              # honest instructions a packed 32-bit register (PRMT; shifts and an OR)
PRICE = {"floor": 32.0, "alone": 64.13}             # W1 a pack instruction
ROWS_16K = 16384
ANDS_COMPRESSION, ANDS_FROM_8 = 58_120, 53_568
ANDS_MAC = 2079 / 16
ANDS_HIDING = 3 * ANDS_FROM_8 + 2 * ANDS_COMPRESSION
# extra MAC-times a 32-bit register moved out of the accumulator after each k32 wgmma step, over the wgmma's own 32, at the
# tensor core's peak (the k32 export kernel, measured): to shared memory, to HBM; and one LOP3 a register, the pack's price
EXPORT = {"sts": 125.9, "stg": 855.9}
TOUCH = 68.4
PERIODS = {"week": 7 * 86400, "month": 30 * 86400, "year": 365 * 86400}
# census: world-inference-2027 (adversary.json), dc-12500-machines-600s (deployments.json); the H200's int8 dense peak is
# its data sheet's
CENSUS = {"world FLOP a year": 9.36e30, "datacenter GPUs": 100_000, "H200 int8 dense ops/s": 1.979e15, "window s": 600}

OPTIONS = {
    "k32": {"L": 32, "w": 32},
    "W16 (lo16 of each k32 word, 2 a register)": {"L": 32, "w": 16},
    "P21 (21-bit deltas, 3 a 64-bit word)": {"L": 32, "w": 64 / 3},
    "L 64": {"L": 64, "w": 32},
    "L 128": {"L": 128, "w": 32},
    "L 256": {"L": 256, "w": 32},
    "L 1024": {"L": 1024, "w": 32},
    "k32 + C1": {"L": 32, "w": 32, "ops": True},
    "k32 + C1 + noise": {"L": 32, "w": 32, "ops": True, "noise": True},
    "W16 + C1 + noise": {"L": 32, "w": 16, "ops": True, "noise": True},
    "L 64 + C1 + noise": {"L": 64, "w": 32, "ops": True, "noise": True},
    "L 128 + C1 + noise": {"L": 128, "w": 32, "ops": True, "noise": True},
}


def blocks(R: int) -> int:
    return (R + 17 + 127) // 128


class Rows:
    """Each measured row size's cost a byte (the leaves' GPU time by kernel, plus the tree), its data kernel's time a
    compression, and the salt kernel's time a leaf, from ROWS_JSON. The kernels' sum, not the event-timed leg: a size's
    first-timed leg can carry time outside its kernels whichever its salts."""

    def __init__(self, path: Path) -> None:
        d = json.loads(path.read_text())
        self.src = {"device": d["device"], "sms": d["sms"], "phases": d["phases"]}
        self.by = {}
        for rb, e in d["sizes"].items():
            rb, m, x, gv = int(rb), e["rows"], e["derived"], e["given"]["kernels_ms"]
            k = x["kernels_ms"]
            c_ns = k["data"] * 1e6 / m / (blocks(rb) + 3)
            salt_ns = k["salt"] * 1e6 / m
            leaf_ns = (k["salt"] + k["data"]) * 1e6 / m
            self.by[rb] = {"ns_byte": (leaf_ns + 3 * c_ns) / rb, "c_ns": c_ns, "salt_ns_leaf": salt_ns, "leaf_ns": leaf_ns,
                           "timed_leaf_ns": x["ms"] * 1e6 / m, "mhz": x["mhz"],
                           "hiding_share": (salt_ns - 65 / 80 * c_ns) / (leaf_ns + 3 * c_ns),
                           "mask_share": (gv["salt"] * 1e6 / m - (2 + 65 / 80) * c_ns) / (leaf_ns + 3 * c_ns),
                           "prf_ns_leaf": (k["salt"] - gv["salt"]) * 1e6 / m,
                           "salt_given_ns_leaf": gv["salt"] * 1e6 / m}

    def ratio(self, R: int) -> float:
        return self.by[R]["ns_byte"] / self.by[ROWS_16K]["ns_byte"]

    def bar(self, R: int) -> float:
        """The bar's time over the data kernel's, a compression."""
        r = self.by[R]
        return BAR_MAC_TIMES / (r["c_ns"] * 1e-9 * N.RULER * r["mhz"] * 1e6)


def per_token(o: dict) -> dict:
    """Committed bytes a token (the words, A, and X, E₁ if committed), a forward (Y, F₁), the linears' MACs a token, and of
    those bytes the words, A's (hashed before the noise) and the formed operands' and noise's (moved out unpacked)."""
    words = pre = ops_tok = ops_fwd = 0
    for n, k in N.LINEARS.values():
        words += n * 8 * math.ceil(-(-3 * k // o["L"]) * o["w"] / 64)
        pre += k
        if o.get("ops"):
            ops_tok, ops_fwd = ops_tok + 3 * k, ops_fwd + 3 * k * n
        if o.get("noise"):
            ops_tok, ops_fwd = ops_tok + k, ops_fwd + k * n
    macs = sum(n * k for n, k in N.LINEARS.values())
    L = N.LAYERS
    return {"token": (words + pre + ops_tok) * L, "forward": ops_fwd * L, "macs": macs * L, "words_token": words * L,
            "pre_noise_token": pre * L, "ops_token": ops_tok * L, "ops_forward": ops_fwd * L}


def t1_terms(o: dict, m: int) -> dict:
    """The T1 terms that don't depend on the hash, MAC-weighted over the linears (fractions of stock)."""
    wt = {x: n * k for x, (n, k) in N.LINEARS.items()}
    tot = sum(wt.values())

    def agg(f) -> float:
        return sum(f(n, k) * wt[x] for x, (n, k) in N.LINEARS.items()) / tot

    def regs(k: int) -> float:
        return math.ceil(-(-3 * k // o["L"]) * o["w"] / 32)
    out = {}
    packed = o["w"] < 32
    for p, price in PRICE.items():
        out[f"interior_h32_{p}"] = agg(lambda n, k: max(0.0, 3 + regs(k) * PACK_INSTR[o["w"]] * price / k - 32 * regs(k) / k))
        out[f"interior_semk32_{p}"] = agg(lambda n, k: max(0.0, regs(k) * (PACK_INSTR[o["w"]] * price - 32 * packed) / k))
    out["final_word"] = agg(lambda n, k: 32 / k)
    out["final_word_semk32"] = agg(lambda n, k: max(32, o["L"]) / k)
    out["final_word_fixable"] = all(o["L"] <= 7 * k / 8 for _, k in N.LINEARS.values())
    out["forming"] = 0.0 if o.get("ops") else agg(lambda n, k: FORM / n + FORM / m)
    out["noise"] = 0.0 if o.get("noise") else agg(lambda n, k: NOISE / n + NOISE / m)
    return out


def periods(out: dict) -> dict:
    """Each option's T 8,192 terms pooled per window and monitoring period over the stock work they cover, for the
    datacenter and the world, in AND gates and W1 units (the export schedule, measured hash)."""
    dc_mac_s = CENSUS["datacenter GPUs"] * CENSUS["H200 int8 dense ops/s"] / 2
    scopes = {"datacenter": {"window": dc_mac_s * CENSUS["window s"], **{p: dc_mac_s * s for p, s in PERIODS.items()}},
              "world": {p: CENSUS["world FLOP a year"] / 2 * s / PERIODS["year"] for p, s in PERIODS.items()}}
    res: dict = {"stock_macs": scopes}
    for name, r in out["options"].items():
        res[name] = {}
        for R, x in r["points"]["T 8,192"]["rows"].items():
            e = x["measured"]
            a, en = e["ands"], e["engines"]
            per = {"t2_export": {g: en[g]["export"] for g in en}}
            for scope, ps in scopes.items():
                per[scope] = {p: {"stock_ands": mac * ANDS_MAC, "hash_ands": mac * ANDS_MAC * a["hash_of_stock"],
                                  "hiding_ands": mac * ANDS_MAC * a["hiding_of_stock"],
                                  "pre_noise_ands": mac * ANDS_MAC * a["pre_noise_of_stock"],
                                  "hiding_w1_floor": mac * e["t1_hiding_unforced"]["floor"],
                                  "pre_noise_w1_floor": mac * e["pre_noise"]["t1_w1"]["floor"],
                                  "stock_macs_covered_hardware_fixed": {g: mac / en[g]["export"] for g in en}}
                              for p, mac in ps.items()}
            res[name][R] = per
    return res


def main() -> None:
    sass = json.loads(Path(sys.argv[1]).read_text())["compression"]
    rows = Rows(Path(sys.argv[2]))
    rate16 = N.HASHED["64 GiB"][0] / N.HASHED["64 GiB"][1]          # bytes a ms
    w1c = {"floor": sass["w1_floor"], "alone": sass["w1_alone"]}
    out: dict = {"inputs": {"sass": {k: sass[k] for k in ("instructions", "registers_written", "w1_floor", "w1_alone", "rolled")},
                            "rows": {"src": rows.src, "by_size": rows.by}, "rate16_GBps": rate16 * 1e-6, "census": CENSUS},
                 "options": {}, "thresholds": {}}
    sizes = sorted(rows.by)
    mac_s = N.RULER * N.F
    for name, o in OPTIONS.items():
        t = per_token(o)
        res: dict = {"bytes_mac_token": t["token"] / t["macs"], "words_bytes_mac": t["words_token"] / t["macs"],
                     "pre_noise_bytes_mac": t["pre_noise_token"] / t["macs"], "points": {}}
        for pname, (real, m) in N.POINTS.items():
            B = t["token"] * real + t["forward"]
            regs = t["words_token"] * real / 4
            moved = t["ops_token"] * real + t["ops_forward"]
            pre = t["pre_noise_token"] * real
            macs = t["macs"] * real
            g = N.GEMM[min(N.GEMM, key=lambda x: abs(x - m))]
            gemm_ms = sum(g["plain_us"] * g[p] * 1e-3 for p in ("wide", "a_side", "noised", "w_side")) + t["macs"] / 64 / N.COMPRESSIONS_S * 1e3
            move_ms = {x: (regs * (u + PACK_INSTR[o["w"]] * TOUCH) + moved * u / 4) / mac_s * 1e3 for x, u in EXPORT.items()}
            terms = t1_terms(o, m)
            pt: dict = {"GB_forward": B / 1e9, "bytes_useful_mac": B / macs, "gemm_extra_ms": gemm_ms, "export_ms": move_ms,
                        "export_mac_times_useful_mac": {x: v * 1e-3 * mac_s / macs for x, v in move_ms.items()}, "t1": terms,
                        "rows": {}}
            whole = N.HASHED.get(pname, N.HASHED["64 GiB"])
            for R in sizes:
                base = B / (whole[0] / whole[1]) if R == ROWS_16K else B / rate16
                for hname, hm in (("measured", base * rows.ratio(R) if R != ROWS_16K else base),
                                  ("at the bar", base * rows.ratio(R) * rows.bar(R) if R != ROWS_16K else base * rows.bar(R))):
                    comps_byte = (blocks(R) + 9 + 3) / R
                    w1_hash = {p: B / macs * comps_byte * w for p, w in w1c.items()}
                    hid = rows.by[R]["hiding_share"]
                    ands = B / macs * comps_byte * ANDS_COMPRESSION
                    pre_ms = hm * pre / B
                    e = {"hash_ms": hm, "comps_byte": comps_byte, "w1_hash_mac": w1_hash,
                         "ands": {"hash_mac": ands, "hash_of_stock": ands / ANDS_MAC, "shaspeed_delta": 0.01 * ANDS_MAC / ands,
                                  "hiding_of_stock": B / macs / R * ANDS_HIDING / ANDS_MAC,
                                  "pre_noise_of_stock": pre / macs * comps_byte * ANDS_COMPRESSION / ANDS_MAC},
                         "shaspeed_delta": {p: 0.01 / v for p, v in w1_hash.items()},
                         "shaspeed_instr_compression": {p: 0.01 / v * sass["instructions"] for p, v in w1_hash.items()},
                         "hiding_share_of_hash": hid, "t1_hiding_unforced": {p: hid * v for p, v in w1_hash.items()},
                         "t1_hiding_mask_only": {p: rows.by[R]["mask_share"] * v for p, v in w1_hash.items()},
                         "pre_noise": {"share_of_hashed": pre / B, "ms": pre_ms,
                                       "t1_w1": {**{p: pre / macs * comps_byte * w for p, w in w1c.items()},
                                                 "mac_times": pre_ms * 1e-3 * mac_s / macs}},
                         "engines": {}}
                    for eng, (stock, _gap, src) in N.ENGINES[pname].items():
                        serial = stock + gemm_ms + hm
                        tH = serial + move_ms["sts"]
                        e["engines"][eng] = {"stock_ms": stock, "src": src, "serial": serial / stock, "export": tH / stock,
                                             "unfused": (serial + move_ms["stg"]) / stock,
                                             "overlap": max(stock + gemm_ms + move_ms["sts"], hm) / stock,
                                             "shaspeed_delta_time_stock": 0.01 * stock / hm,
                                             "shaspeed_delta_time_tH": 0.01 * tH / hm,
                                             "hiding_time_of_stock": hid * hm / stock, "hiding_time_of_tH": hid * hm / tH,
                                             "pre_noise_time_of_stock": pre_ms / stock, "pre_noise_time_of_tH": pre_ms / tH}
                    pt["rows"].setdefault(str(R), {})[hname] = e
            res["points"][pname] = pt
        out["options"][name] = res
    out["periods"] = periods(out)
    # the bytes a useful MAC (words, operands and noise all counted) under which t_H/t_S ≤ 100 at T 8,192
    pname = "T 8,192"
    real, m = N.POINTS[pname]
    g = N.GEMM[8192]
    t = per_token(OPTIONS["k32"])
    gemm_ms = sum(g["plain_us"] * g[p] * 1e-3 for p in ("wide", "a_side", "noised", "w_side")) + t["macs"] / 64 / N.COMPRESSIONS_S * 1e3
    sts_ms_byte = EXPORT["sts"] / 4 / mac_s * 1e3
    for R in sizes:
        for hname, f in (("measured", rows.ratio(R)), ("at the bar", rows.ratio(R) * rows.bar(R))):
            ms_byte = f / rate16
            for eng, (stock, _g, _s) in N.ENGINES[pname].items():
                out["thresholds"][f"{R} {hname} {eng}"] = {
                    "serial": (99 * stock - gemm_ms) / ms_byte / (t["macs"] * real),
                    "export": (99 * stock - gemm_ms) / (ms_byte + sts_ms_byte) / (t["macs"] * real),
                    "overlap": 100 * stock / ms_byte / (t["macs"] * real)}
    Path(sys.argv[3]).write_text(json.dumps(out, indent=1))
    print("rows:", "  ".join(f"{R}: {rows.ratio(R):.3f} of 16 KiB's time a byte, hiding {rows.by[R]['hiding_share']:.2%} of the hash "
                             f"(M_key y {rows.by[R]['mask_share']:.2%}), bar/measured {rows.bar(R):.3f}" for R in sizes))
    for name, res in out["options"].items():
        line = [f"{name:<44} {res['bytes_mac_token']:.4f} B/MAC a token"]
        for pname in ("T 8,192", "T 512, 64 real rows", "T 512, 512 real rows"):
            pt = res["points"][pname]
            for R in (ROWS_16K, 262144):
                if str(R) not in pt["rows"]:
                    continue
                e = pt["rows"][str(R)]["measured"]
                b = pt["rows"][str(R)]["at the bar"]
                es, ea, bs, ba = e["engines"]["sync"], e["engines"]["async"], b["engines"]["sync"], b["engines"]["async"]
                line.append(f"  {pname} {R // 1024} KiB: serial {es['serial']:.1f}/{ea['serial']:.1f}x, export "
                            f"{es['export']:.1f}/{ea['export']:.1f} (bar {bs['export']:.1f}/{ba['export']:.1f}, overlap "
                            f"{es['overlap']:.1f}/{ea['overlap']:.1f}, unfused {es['unfused']:.1f}/{ea['unfused']:.1f})"
                            f" δ {e['shaspeed_delta']['floor']:.2e}/{e['shaspeed_delta']['alone']:.2e}"
                            f" hiding {e['t1_hiding_unforced']['floor']:.3f}/{e['t1_hiding_unforced']['alone']:.3f}"
                            f" pre-noise {e['pre_noise']['t1_w1']['floor']:.3f}/{e['pre_noise']['t1_w1']['alone']:.3f}"
                            f"/{e['pre_noise']['t1_w1']['mac_times']:.3f}")
        for R in (ROWS_16K, 262144):
            x = res["points"]["T 8,192"]["rows"][str(R)]["measured"]
            a = x["ands"]
            line.append(f"  ANDs at T 8,192, {R // 1024} KiB: hash {a['hash_of_stock']:.3f}x stock, ShaSpeed δ {a['shaspeed_delta']:.2%}, "
                        f"hiding {a['hiding_of_stock']:.3%}, pre-noise {a['pre_noise_of_stock']:.4%} of stock; pre-noise "
                        f"{x['pre_noise']['share_of_hashed']:.4%} of the bytes, time {x['engines']['sync']['pre_noise_time_of_stock']:.2%}"
                        f" of stock, {x['engines']['sync']['pre_noise_time_of_tH']:.4%} of t_H")
        t1 = res["points"]["T 8,192"]["t1"]
        line.append(f"  T1 at T 8,192: interior H32 {t1['interior_h32_floor']:.2%}/{t1['interior_h32_alone']:.2%}, SemK32 "
                    f"{t1['interior_semk32_floor']:.2%}/{t1['interior_semk32_alone']:.2%}, final word "
                    f"{t1['final_word']:.2%} (SemK32 {t1['final_word_semk32']:.2%}, fixable {t1['final_word_fixable']}), forming "
                    f"{t1['forming']:.2%}, noise {t1['noise']:.2%}")
        print("\n".join(line))
    for name in OPTIONS:
        pt = out["options"][name]["points"]["T 8,192"]
        print(f"export at T 8,192 {name}: STS {pt['export_ms']['sts']:.1f} ms ({pt['export_mac_times_useful_mac']['sts']:.2f} a useful MAC),"
              f" STG {pt['export_ms']['stg']:.1f} ms")
    for k, v in out["thresholds"].items():
        print(f"T2 ≤ 100 at T 8,192 needs bytes a useful MAC ≤ {v['serial']:.4f} serial, {v['export']:.4f} with the export, "
              f"{v['overlap']:.4f} overlap ({k})")


if __name__ == "__main__":
    main()
