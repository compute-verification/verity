"""The NoLookupGain table from lookup_coissue.cu, fused_coissue.cu and register_lookup.cu output (their JSON lines, on
stdin or as files). Times are compared as ratios to each program's own interleaved cuBLASLt int8.

For each (n, alphabet): T_tensor = the fastest tensor-only exact program (cuBLASLt int8, cuBLASLt FP8, our own int8
kernel); T_best = the fastest exact program over every route and α; η̂ = 1 − T_best/T_tensor. Also the note's
η̂_note = 1 − T_lookup/T_fixed, with T_fixed the fastest fixed-address program (tensor-only, DP4A and DP4A+HFMA2
hybrids) and T_lookup the fastest program that uses table lookups. Every row also gives the in-kernel gain of the
best fused lookup hybrid over the same kernel's tensor-only form (own_mma).

--detail adds, from the full output (start, gate, control, prepass and summary lines, in program order): the gates and
controls, the standalone lookup rates in MACs/SM/clk at the SM clock read during their timing, the fused α sweeps
(pre-pass times over own_mma's) and the two-stream hybrids' overlap."""

import json
import re
import sys
from collections import defaultdict

TENSOR = {"int8", "fp8_e4m3", "own_mma"}
ORDER = {"k2": 0, "k4": 1, "k6": 2, "k8": 3, "u6": 4}


def kind(route):
    if route in TENSOR or route == "fused_fr_a0":
        return "tensor"
    if route.endswith(("_tensor_part", "_lookup_part")):
        return "part"
    if route.startswith(("fused_dp4a", "fused_mix")):
        return "fixed"
    return "lookup"  # fr_b*, fr_l1_b*, prmt*, shfl*, hyb_a*, fused_fr_b*


def read(srcs):
    """Every event, with (prog, n, alphabet) carried from the last start line onto the lines that lack them."""
    events, ctx = [], {}
    for src in srcs or ["-"]:
        text = sys.stdin.read() if src == "-" else open(src).read()  # noqa: SIM115
        lines = text.splitlines()
        for i, line in enumerate(lines):
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                if (
                    i == len(lines) - 1
                ):  # a run still writing: its last line may be cut off
                    continue
                raise
            if e.get("event") == "start":
                ctx = {
                    "prog": e.get("prog", "lookup_coissue"),
                    "n": e["n"],
                    "alphabet": e["alphabet"],
                }
            for k, v in ctx.items():
                e.setdefault(k, v)
            events.append(e)
    return events


def keyorder(k):
    return (k[0], ORDER.get(k[1], 9))


def rate(r):
    """MACs/SM/clk at the SM clock read around the item's timing (2092 MHz when NVML gave no reading)."""
    clk = 0.5 * (r["clk_min"] + r["clk_max"]) or 2092.0
    return r["macs_per_sm_clk_at_2092"] * 2092.0 / clk


def table(rows):
    # Each program times cuBLASLt int8 interleaved with its own routes, possibly on a different GPU from the others, so
    # routes are compared by ratio_vs_int8 (their program's own int8), never by milliseconds across programs.
    for r in rows:
        r["median_ms_raw"], r["median_ms"] = r["median_ms"], r["ratio_vs_int8"]
    by = defaultdict(dict)
    t8ms = defaultdict(list)
    for r in rows:
        key = (r["n"], r["alphabet"])
        if r["route"] == "int8":
            t8ms[key].append(r["median_ms_raw"])
        prev = by[key].get(r["route"])
        if prev is None or r["median_ms"] < prev["median_ms"]:
            by[key][r["route"]] = r
    print(
        "| n | S | T_int8 ms (each program) | T_tensor/T_int8 (route) | best route | α | b | T_best/T_tensor | η̂ | "
        "fused FR / fused α=0 | fused DP4A(+HFMA2) / fused α=0 | η̂_note | SM clock MHz |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for n, a in sorted(by, key=keyorder):
        d = by[(n, a)]
        full = {k: v for k, v in d.items() if kind(k) != "part"}
        tens = min(
            (v for k, v in full.items() if k in TENSOR), key=lambda v: v["median_ms"]
        )
        best = min(full.values(), key=lambda v: v["median_ms"])
        fixed = [v for k, v in full.items() if kind(k) in ("tensor", "fixed")]
        look = [v for k, v in full.items() if kind(k) == "lookup"]
        tfix = min(v["median_ms"] for v in fixed)
        tlook = min(v["median_ms"] for v in look) if look else float("nan")
        # the co-issue hybrids against the same 512-thread kernel with its CUDA-core warps idle (α = 0): same occupancy
        # and epilogue, so the ratio is what co-issue itself gains or loses
        a0 = d.get("fused_fr_a0")

        def inkernel(prefixes, full=full, a0=a0):
            h = [v["median_ms"] for k, v in full.items() if k.startswith(prefixes)]
            return min(h) / a0["median_ms"] if h and a0 else float("nan")

        clk = (
            min(v["clk_min"] for v in full.values()),
            max(v["clk_max"] for v in full.values()),
        )
        t8 = "/".join(f"{t:.3f}" for t in sorted(t8ms[(n, a)]))
        print(
            f"| {n} | {a} | {t8} | {tens['median_ms']:.4f} ({tens['route']}) | {best['route']} | "
            f"{best['alpha']:.4f} | {best['b']} | {best['median_ms'] / tens['median_ms']:.4f} | "
            f"{1 - best['median_ms'] / tens['median_ms']:+.4f} | {inkernel('fused_fr_b'):.4f} | "
            f"{inkernel(('fused_dp4a', 'fused_mix')):.4f} | {1 - tlook / tfix:+.4f} | "
            f"{clk[0]}–{clk[1]} |"
        )


def detail(events):
    gates = defaultdict(lambda: [0, 0, []])
    controls = defaultdict(list)
    for e in events:
        key = (e["n"], e["alphabet"])
        if e["event"] == "gate":
            g = gates[key]
            g[0] += 1
            if e["mismatches"]:
                g[1] += 1
                g[2].append(f"{e['route']}({e['mismatches']})")
        elif e["event"] == "control":
            controls[key].append("rejected" if e["rejected"] else "ACCEPTED")
    print("\n| n | S | gates passed | failed (mismatches) | controls |")
    print("|---|---|---|---|---|")
    for key in sorted(set(gates) | set(controls), key=keyorder):
        g = gates[key]
        print(
            f"| {key[0]} | {key[1]} | {g[0] - g[1]}/{g[0]} | {' '.join(g[2]) or '-'} | {' '.join(controls[key])} |"
        )

    summ = [e for e in events if e["event"] == "summary"]
    fams = [  # (column, route pattern whose group 1 labels the width)
        ("smem FR (k_fr)", r"fr_(b\d+)$"),
        ("smem FR (fused kernel, α = 63/64)", r"fused_fr_(b\d+)_a63$"),
        ("L1 FR", r"fr_l1_(b\d+)$"),
        ("PRMT", r"prmt_(b.*)$"),
        ("SHFL", r"shfl_(b.*)$"),
        ("DP4A (fused, α = 63/64)", r"fused_(dp4a)_a63$"),
        ("DP4A+HFMA2 (fused, α = 63/64)", r"fused_(mix)_a63$"),
    ]
    by = defaultdict(dict)
    for r in summ:
        by[(r["n"], r["alphabet"])].setdefault(r["prog"], {})[r["route"]] = r
    print(
        "\nStandalone lookups, MACs/SM/clk at the measured SM clock (int8 = cuBLASLt in the same program); "
        "every width timed, best marked *:\n"
    )
    print("| n | S | int8 | " + " | ".join(f for f, _ in fams) + " |")
    print("|---|---|---|" + "---|" * len(fams))
    for key in sorted(by, key=keyorder):
        cells, i8 = [], []
        for d in by[key].values():
            if "int8" in d:
                i8.append(f"{rate(d['int8']):.0f}")
        for _, pat in fams:
            got = [
                (rate(r), re.match(pat, name).group(1))
                for d in by[key].values()
                for name, r in d.items()
                if re.match(pat, name)
            ]
            if not got:
                cells.append("-")
                continue
            top = max(got)[1]
            cells.append(
                " ".join(
                    f"{'*' if label == top else ''}{label}:{v:.0f}"
                    for v, label in sorted(got, key=lambda x: x[1])
                )
            )
        print(f"| {key[0]} | {key[1]} | {'/'.join(i8)} | " + " | ".join(cells) + " |")

    pre = defaultdict(dict)
    for e in events:
        if e["event"] == "prepass" and e["prog"] == "fused_coissue":
            pre[(e["n"], e["alphabet"])][e["route"]] = e
    if pre:
        print(
            "\nFused co-issue α sweep: pre-pass time over the same kernel's at α = 0 (fused_fr_a0: CUDA-core warps idle, "
            "same occupancy and epilogue), best b per α (interleaved best in the main table):\n"
        )
        print("| n | S | role | α·64 : T/T_fused α=0 |")
        print("|---|---|---|---|")
        for key in sorted(pre, key=keyorder):
            d = pre[key]
            own = d.get("fused_fr_a0")
            if not own:
                continue
            for role, pat in (
                ("FR", r"fused_fr_b(\d+)_a(\d+)$"),
                ("DP4A", r"fused_dp4a_a(\d+)$"),
                ("DP4A+HFMA2", r"fused_mix_a(\d+)$"),
            ):
                curve = {}
                for name, e in d.items():
                    m = re.match(pat, name)
                    if not m:
                        continue
                    j = int(m.groups()[-1])
                    t = e["ms"] / own["ms"]
                    if j not in curve or t < curve[j][0]:
                        curve[j] = (t, e["b"])
                if curve:
                    print(
                        f"| {key[0]} | {key[1]} | {role} | "
                        + " ".join(
                            f"{j}:{t:.3f}" + (f"(b{b})" if role == "FR" else "")
                            for j, (t, b) in sorted(curve.items())
                        )
                        + " |"
                    )

    hyb = defaultdict(dict)
    for r in summ:
        m = re.match(r"(hyb_a\d+)(_tensor_part|_lookup_part)?$", r["route"])
        if m:
            hyb[(r["n"], r["alphabet"], m.group(1))][m.group(2) or ""] = r["median_ms"]
    if hyb:
        print(
            "\nTwo-stream hybrid (cuBLASLt on 1 − α beside the smem FR kernel on α, separate streams): overlap = "
            "(T_tensor_part + T_lookup_part − T_hybrid)/min(parts); 1 hides the smaller part, 0 serializes:\n"
        )
        print(
            "| n | S | hybrid | T_hybrid ms | T_tensor_part | T_lookup_part | overlap |"
        )
        print("|---|---|---|---|---|---|---|")
        for n, a, h in sorted(
            hyb, key=lambda k: (k[0], ORDER.get(k[1], 9), int(k[2][5:]))
        ):
            d = hyb[(n, a, h)]
            if {"", "_tensor_part", "_lookup_part"} <= set(d):
                tp, lp, th = d["_tensor_part"], d["_lookup_part"], d[""]
                print(
                    f"| {n} | {a} | {h} | {th:.4f} | {tp:.4f} | {lp:.4f} | {(tp + lp - th) / min(tp, lp):+.2f} |"
                )


def main():
    args = sys.argv[1:]
    want = "--detail" in args
    events = read([a for a in args if a != "--detail"])
    table([dict(e) for e in events if e.get("event") == "summary"])
    if want:
        detail(events)


if __name__ == "__main__":
    main()
