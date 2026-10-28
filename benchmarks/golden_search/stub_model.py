"""GD-SEARCH's stand-in for GD-MODEL's cost model of verity/remote/v0 (sampled proofs + PoUW on one served window of
Llama-3.2-1B int8, B4, one H200), until GD-MODEL's module lands.  Stdlib only.

`evaluate(design, inputs, point)` returns J = (t_remote/t_stock - 1) + proving GPU-s / stock GPU-s for one window, the
two terms apart, the verifier's CPU-s, each credited call's credit floor (a share of its plain s8 GEMM G), the largest
proof block in gates and what lowering it takes.  Every figure is an estimate built from the brief's known numbers
(`INPUTS`: value, low, high, source, status M measured / D derived / A assumed); none is a measurement of this design.

The parts:
- serve: the served no-Commit forward (b57f), the Commit's and PoUW's hashing against the co-run share beside the GEMM,
  the seal and the host remainder (COMMIT-BYTES, P1A-LEVERS), and PoUW's extra tensor work (Dither's peel);
- proving: Achievability's draws, q_u = min(1, Lambda w_u / C), so about Lambda/eps proof blocks a guarantee, each priced
  by C-Flock's fit (`costfit` on r20261018-060433-b55e) at its ANDs with its rows' openings (`optimize.row_ands`);
- credit: Dither's noised floor min(1, r_b + 3/t) (note:compute-research/20261028T0523Z-finding-dither-noised-floor),
  0 when L >= K (O alone is un-noised), times the word's unguessed share.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log

#: Llama-3.2-1B: 16 layers, hidden 2,048, intermediate 8,192, 32 query and 8 KV heads of 64, vocab 128,256
LAYERS, HIDDEN, INTER, VOCAB = 16, 2048, 8192, 128256
#: the int8 linears a layer: (K, N)
CALLS = {"qkv": (2048, 3072), "o": (2048, 2048), "gate_up": (2048, 16384), "down": (8192, 2048)}
#: SHA-512 compression and hm96's hiding part in ANDs (hm96 PROTOCOL section 5; optimize.SHA512_ANDS, HM96_ANDS)
SHA512_ANDS, HM96_ANDS = 58_120, 116_240
#: one s8 step of 32 MACs and an s8 coordinate's epilogue in ANDs (benchmarks/sampling_plan/inputs.py S8_STEP, EPILOGUE)
S8_STEP, EPILOGUE = 4_190, 5_000

POINTS = {"t8192": {"T": 8192, "real": 8192}, "t512": {"T": 512, "real": 64}}


@dataclass(frozen=True)
class Input:
    value: float
    low: float
    high: float
    source: str
    status: str
    uncertain: bool = True


INPUTS: dict[str, Input] = {
    # stock and the served path
    "stock_util_8192": Input(0.2397, 0.2397, 0.3387, "stock's tensor utilization at T 8,192 (brief); high = the served "
                             "no-Commit path b57f's 33.87%, i.e. J against our own int8 path", "M"),
    "ideal_ms_8192": Input(8.854, 8.854, 8.854, "b57f's tensor work at peak, 33.87% x 26.14 ms (COMMIT-BYTES s3)", "D",
                           False),
    "fwd_ms_8192": Input(26.14, 26.14, 26.14, "served path, no Commit, T 8,192, b57f", "M", False),
    "gemm_share": Input(10.4 / 26.14, 0.35, 0.6, "the int8 GEMMs' share of the no-Commit step (b921's Commit-off stack "
                        "10.4 ms, P1A-LEVERS)", "M"),
    "fwd_ms_512": Input(3.42, 2.5, 5.0, "T 512 / 64 real tokens: forward kernels 0.96 ms + host gap 2.46 ms (P1A-LEVERS' "
                        "V3 scenario, scaled from T 8,192's parts, not a profile)", "A"),
    "gemm_ms_512": Input(0.55, 0.3, 0.8, "the int8 GEMMs at 64 real tokens: 0.96 ms of kernels at the T 8,192 GEMM share "
                         "of kernel time (10.4 of 18.3 ms)", "A"),
    "stock_ms_512": Input(3.42, 2.5, 5.0, "stock at 64 real tokens, taken equal to the served forward (no profile)", "A"),
    "commit_rows_512": Input(512, 64, 512, "tokens the Commit hashes at T 512: 512 padded rows (upper bound) or 64 real",
                             "A"),
    "launch_us": Input(4.0, 2.0, 8.0, "floor a hash launch at small shapes (P1A-LEVERS)", "A"),
    "launches_512": Input(400, 100, 800, "the Commit's launches a T 512 step (P1A-LEVERS)", "A"),
    # hashing
    "bar_gcps": Input(4.354, 4.354, 4.354, "SHA-512 compressions/s on the whole H200, 16 KiB rows (HASH-38, "
                      "r20261027-042422-85b2)", "M", False),
    "eff_served": Input(0.51, 0.4, 0.66, "served hash kernels' share of the bar (b921)", "M"),
    "corun_share": Input(0.167, 0.10, 0.25, "share of the bar beside a streamed GEMM at no cost to it "
                         "(r20261028-012220-309e)", "M"),
    "corun4_share": Input(0.43, 0.43, 0.43, "4 hash warps beside the GEMM (309e)", "M", False),
    "corun4_gemm_cost": Input(0.19, 0.10, 0.25, "the GEMM's slowdown with 4 hash warps (309e)", "M"),
    "gather_ns_per_b": Input(12.2e-3 / 13.63e9 * 1e9, 0.5e-3, 1.5e-3, "the Commit's gather, 12.2 ms over 13.63 GB (b921)",
                             "M"),
    "seal_s_per_mrow": Input(1.3348 / 3.327254, 0.3, 0.5, "the sealer's 1,335 ms a step over 3.33 M rows (b921)", "M"),
    "host_ms_8192": Input(133.0, 100.0, 170.0, "the Commit's host remainder at T 8,192: event syncs, copies, gaps "
                          "(P1A-LEVERS row 7)", "M"),
    "bytes_tok_all": Input(1_663_627, 1_663_627, 1_667_472, "the Commit's bytes a token, every op's output, the int8 path "
                           "twice (b921, COMMIT-BYTES)", "M", False),
    "bytes_tok_dedupe": Input(1_270_000, 1_270_000, 1_270_000, "lever 11: drop SiluAndMul's and the quantizer-only norms' "
                              "bf16 outputs (COMMIT-BYTES)", "D", False),
    "bytes_tok_fuse": Input(880_000, 880_000, 880_000, "lever 12: gate_up+SiluAndMul+quantize one unit (COMMIT-BYTES)",
                            "D", False),
    # PoUW credit and serve
    "r_b": Input(1 / 8, 1 / 16, 1 / 4, "the binary product's cost a MAC against s8 (8x s8 a clock; the coordinator's "
                 "figure, not probed on Hopper)", "A"),
    "peel": Input(4.0, 2.0, 6.0, "Dither's peel, extra tensor work 4 L'/K (PROTOCOL.md)", "A"),
    "guess_p16": Input(2.79e-5, 1e-5, 1e-4, "most predictable word's free-guess rate at 16 bits on the H200 (PROTOCOL.md)",
                       "M"),
    "o_bytes": Input(4.0, 4.0, 4.0, "O at 32 bits a cell in the value root (PROTOCOL.md)", "D", False),
    # proving (C-Flock fit, RTX PRO 6000, r20261018-060433-b55e)
    "prove_and": Input(1.087e-8, 0.5e-8, 2.2e-8, "GPU-s an AND of a proved proof block (costfit, sigma 0.15; H200 or "
                       "sm_120 unpriced)", "M"),
    "staged_and": Input(1.016e-10, 0.5e-10, 2e-10, "GPU-s an AND of a staged proof block (costfit)", "M"),
    "stmt_fixed": Input(3.03, 1.5, 6.0, "GPU-s a touched statement (costfit)", "M"),
    "stmt_and": Input(3.51e-6, 0.0, 7e-6, "GPU-s an AND of the statement's proof block, a touched statement (costfit); "
                      "0 if a template's preprocessing is kept across windows", "M"),
    "window_steps": Input(1, 1, 1, "served steps a window: one (b921; V3 at S 1); proving a window doesn't grow with it",
                          "A", False),
    "verify_fixed": Input(12.44, 12.44, 12.44, "CPU-s Lean verify --zk a statement (costfit)", "M", False),
    "verify_and": Input(6.71e-6, 6.71e-6, 6.71e-6, "CPU-s an AND a statement (costfit)", "M", False),
    "nonlin_share": Input(0.2, 0.05, 0.5, "share of the window's AND gates outside the int8 linears (norms, SiLU, rope, "
                          "attention, quantize): no figure", "A"),
    "nonlin_ands_elem": Input(3000, 1000, 10000, "ANDs an element of a bf16 elementwise op in Boolean form: no figure",
                              "A"),
    "nonlin_templates": Input(10, 5, 30, "templates outside the linears (statements per window when per template)", "A"),
    # lowering (TOP-138)
    "gates_per_and": Input(2.1, 1.5, 4.0, "gates an AND: TOP-138's ~1e9 gates for the K 2,048 32x32 H1 tile over this "
                           "model's 4.75e8 ANDs", "D"),
    "lower_b_per_gate": Input(600.0, 16.0, 600.0, "lowering's bytes a gate: 600 today (TOP-138); 16 if the swarm's "
                              "compact/stream form lands", "M"),
    "lower_us_per_gate": Input(10.0, 0.5, 10.0, "lowering's microseconds a gate: 170 min at ~1e9 gates, OOM (TOP-138)",
                               "M"),
    "lower_mem_gib": Input(16.0, 16.0, 16.0, "TOP-138's target memory", "A", False),
    "lower_min": Input(15.0, 15.0, 15.0, "TOP-138's target wall", "A", False),
    # the guarantees
    "delta_log2": Input(-40, -40, -40, "delta = 2^-40 a window (brief)", "A", False),
    "eps": Input(0.01, 0.01, 0.01, "eps = 1% (brief)", "A", False),
}


def nominal() -> dict[str, float]:
    return {k: v.value for k, v in INPUTS.items()}


def row_ands(nbytes: int) -> int:
    """In-circuit ANDs to open one hm96-sha512 row of `nbytes` (optimize.row_ands)."""
    return HM96_ANDS + SHA512_ANDS * ceil((nbytes + 17) / 128)


def open_ands(nbytes: float, row: int) -> float:
    """Opening `nbytes` contiguous committed bytes in rows of `row` bytes: whole rows, at least one."""
    return max(1, ceil(nbytes / row)) * row_ands(row)


def comps_a_byte(row: int) -> float:
    """SHA-512 compressions a committed byte at `row`-byte rows: the row's data and padding, the hiding part (salt 3, c 2,
    h 2), the frame leaf 2 and the tree's node 2 (COMMIT-BYTES: 42 a 4 KiB row plus 2 a node, 138 a 16 KiB row)."""
    return (ceil((row + 17) / 128) + 11) / row


def credited(design: dict) -> list[str]:
    return ["qkv", "o", "gate_up"] if design["credit"] == "k2048" else list(CALLS)


def floor_of(design: dict, k: int, v: dict) -> float:
    """A credited call's forced work as a share of its plain s8 GEMM: Dither's noised floor at its noise structure, 0 when
    one segment covers K (O alone, un-noised), times the word's unguessed share."""
    lp, t = design["lp"], design["t"]
    seg = lp * t
    if seg >= k or k % lp:
        return 0.0
    if design["noise"] == "perm":
        f = min(1.0, v["r_b"] + 3.0 / t)
    elif design["noise"] == "periodic":
        f = min(1.0, v["r_b"] + 2.0 / t + lp / k)
    else:
        f = 1.0
    guess = min(1.0, v["guess_p16"] * 2.0 ** (16 - design["bits"]))
    return f * (1.0 - guess)


def evaluate(design: dict, v: dict | None = None, point: str = "t8192") -> dict:
    """One design at one point: J and its parts, the floor a credited call, the largest proof block and lowering."""
    v = nominal() if v is None else v
    T, real = POINTS[point]["T"], POINTS[point]["real"]
    lam, eps = -v["delta_log2"] * log(2.0), v["eps"]
    big = point == "t8192"

    # stock and the served forward
    if big:
        t_stock = v["ideal_ms_8192"] / v["stock_util_8192"] * 1e-3
        t_fwd = v["fwd_ms_8192"] * 1e-3
    else:
        t_stock, t_fwd = v["stock_ms_512"] * 1e-3, v["fwd_ms_512"] * 1e-3
    t_gemm = t_fwd * v["gemm_share"] if big else v["gemm_ms_512"] * 1e-3
    macs = {c: real * n * k * LAYERS * v["window_steps"] for c, (k, n) in CALLS.items()}
    mac_all = sum(macs.values())
    cred = credited(design)
    mac_cred = sum(macs[c] for c in cred)

    # the Commit's bytes and compressions
    tok = T if big else v["commit_rows_512"]
    commit_b = v[{"all": "bytes_tok_all", "dedupe": "bytes_tok_dedupe", "fuse": "bytes_tok_fuse"}[design["fuse"]]] * tok
    kc = design["kchunk"]
    for c in CALLS:
        k, n = CALLS[c]
        if kc > 1:
            commit_b += real * n * 4 * (kc - 1) * LAYERS
    comps = commit_b * comps_a_byte(design["row"])

    # PoUW's words: S - 1 prefix sums at `bits` a cell, and O at 32 bits, a credited call
    floors, words_b = {}, 0.0
    tensor_extra = 0.0
    for c in cred:
        k, n = CALLS[c]
        floors[c] = floor_of(design, k, v)
        S = ceil(k / (design["lp"] * design["t"]))
        cells = real * n * LAYERS
        words_b += cells * ((S - 1) * design["bits"] / 8 / design["agg"] + v["o_bytes"])
        share = macs[c] / mac_all
        if design["noise"] == "fresh":
            tensor_extra += share * 1.0
        elif design["noise"] == "perm":
            tensor_extra += share * v["peel"] * design["lp"] / k
        else:
            tensor_extra += share * v["peel"] / 2 * design["lp"] / k
    comps += words_b * comps_a_byte(design["drow"])

    # serve time
    bar = v["bar_gcps"] * 1e9
    eff = 1.0 if design["kernels"] == "bar" else v["eff_served"]
    t_tensor = tensor_extra * t_gemm
    gemm_t = t_gemm * (1 + tensor_extra)
    if design["overlap"] == "none":
        t_hash = comps / (bar * eff)
    elif design["overlap"] == "corun":
        t_hash = max(0.0, comps - v["corun_share"] * bar * gemm_t) / (bar * eff)
    else:
        t_hash = max(0.0, comps - v["corun4_share"] * bar * gemm_t) / (bar * eff) + v["corun4_gemm_cost"] * gemm_t
    if not big:
        t_hash = max(t_hash, v["launches_512"] * v["launch_us"] * 1e-6)
    t_gather = 0.0 if design["kernels"] == "bar" else commit_b * v["gather_ns_per_b"] * 1e-9
    rows = commit_b / design["row"] + words_b / design["drow"]
    t_seal = rows / 1e6 * v["seal_s_per_mrow"] if design["seal"] == "tree" else 0.0
    t_host = 0.0 if design["host"] == "fixed" else v["host_ms_8192"] * 1e-3 * (commit_b + words_b) / (
        v["bytes_tok_all"] * 8192)
    t_remote = t_fwd + t_tensor + t_hash + t_gather + t_seal + t_host
    serve = t_remote / t_stock - 1.0

    # proving.  Values (guarantee 1) over every proof block, weighted by its AND gates; compute (guarantee 2) over the
    # credited calls' MACs.  A block u is drawn at q_u = min(1, Lambda w_u / C) (Achievability), or at one flat rate.
    row, drow, seg = design["row"], design["drow"], design["lp"] * design["t"]
    r, cc = design["tile"]
    apm = S8_STEP / 32
    pa = v["prove_and"] + v["staged_and"]
    eta = 1.0 - min(1.0, v["guess_p16"] * 2.0 ** (16 - design["bits"]))
    joint = design["joint"] == "joint"
    # population rows: [name, blocks, AND gates w a block, ANDs a proved block, credited MACs a block, K]
    pop = []
    for c, (k, n) in CALLS.items():
        kk = k // kc
        ands = r * cc * (kk * apm + EPILOGUE) + (r + cc) * open_ands(kk, row) + r * open_ands(2 * cc, row)
        if kc > 1:
            ands += 2 * r * open_ands(4 * cc, row)
        pop.append([c, macs[c] / (r * cc * kk), r * cc * kk * apm, ands, r * cc * kk if c in cred else 0, k])
    F_lin = sum(p[1] * p[2] for p in pop)
    F = F_lin / (1 - v["nonlin_share"])
    nl_w = HIDDEN * v["nonlin_ands_elem"]
    pop.append(["nonlinear", (F - F_lin) / nl_w, nl_w, nl_w + 2 * open_ands(2 * HIDDEN, row), 0, 0])
    W_cred = sum(p[1] * p[4] for p in pop)
    C_v, C_c = eps * F, eps * W_cred
    w_max = max(p[2] for p in pop)
    p_values, p_c, draws_v, draws_c, max_ands = 0.0, 0.0, 0.0, 0.0, 0.0
    templates: list[float] = []
    seen: set[tuple[int, bool]] = set()
    for name, units, w, ands, mac_u, k in pop:
        q = min(1.0, lam * (w if design["sampling"] == "work" else w_max) / C_v)
        if joint and mac_u:
            S = ceil(k / seg)
            words = r * cc * (S - 1) * design["bits"] / 8 / design["agg"]
            ands = ands + w + open_ands(words, drow) + (2 + 2 * ceil(k / kc * 9 / 512)) * SHA512_ANDS * (r + cc)
            qc = min(1.0, lam * mac_u / (eta * C_c))
            draws_c += units * qc
            q = max(q, qc)
        p_values += units * q * ands * pa
        draws_v += units * q
        max_ands = max(max_ands, ands)
        if name == "nonlinear":
            templates += [ands] * int(v["nonlin_templates"])
        elif (k, bool(mac_u)) not in seen:
            seen.add((k, bool(mac_u)))
            templates.append(ands)
    if not joint:
        # Dither's proof blocks: one cell, one segment; drawn at q W_p / W_reg; two templates a K (s < S, s = S)
        dither_t: dict[int, tuple[float, float]] = {}
        for name, units, w, ands, mac_u, k in pop:
            if not mac_u:
                continue
            S = ceil(k / seg)
            seeds = lambda width: (2 + 2 * ceil(width * 9 / 512)) * SHA512_ANDS  # noqa: E731
            mid = seg * apm + 2 * open_ands(seg, row) + 2 * row_ands(drow) + seeds(seg)
            last = 2 * k * apm + 2 * open_ands(k, row) + row_ands(drow) + open_ands(4, drow) + seeds(k)
            blk = (mid * (S - 1) + last) / S
            n_draw = lam / (eta * eps) * units * mac_u / W_cred
            draws_c += n_draw
            p_c += n_draw * blk * pa
            if S > 1:
                max_ands = max(max_ands, mid, last)
                dither_t[k] = (mid, last)
        templates += [a for pair in dither_t.values() for a in pair]
    if design["stmt"] == "universal":
        templates = [max_ands, max_ands]
    p_stmt = sum(v["stmt_fixed"] + v["stmt_and"] * a for a in templates)
    prove = p_values + p_c + p_stmt
    verify = sum(v["verify_fixed"] + v["verify_and"] * a for a in templates)
    prove_ratio = prove / (t_stock * v["window_steps"])
    p_lin = p_values

    # lowering: the largest proof block's circuit
    gates = max_ands * v["gates_per_and"]
    mem_gib = gates * v["lower_b_per_gate"] / 2 ** 30
    wall_min = gates * v["lower_us_per_gate"] * 1e-6 / 60
    lowerable = mem_gib <= v["lower_mem_gib"] and wall_min <= v["lower_min"]
    fmin = min(floors.values()) if floors else 0.0
    reasons = []
    if fmin < 1 - eps:
        reasons.append("floor")
    if not lowerable:
        reasons.append("lowering")
    return {"J": serve + prove_ratio, "serve": serve, "prove_ratio": prove_ratio, "prove_gpu_s": prove,
            "verify_cpu_s": verify, "t_remote_ms": t_remote * 1e3, "t_stock_ms": t_stock * 1e3,
            "parts_ms": {"fwd": t_fwd * 1e3, "tensor": t_tensor * 1e3, "hash": t_hash * 1e3, "gather": t_gather * 1e3,
                         "seal": t_seal * 1e3, "host": t_host * 1e3},
            "prove_parts": {"values": p_lin, "compute": p_c, "statements": p_stmt},
            "commit_gb": commit_b / 1e9, "words_gb": words_b / 1e9, "floor_min": fmin, "floors": floors,
            "max_ands": max_ands, "gates": gates, "lower_gib": mem_gib, "lower_min": wall_min,
            "feasible": not reasons, "reasons": reasons, "draws_values": draws_v, "draws_compute": draws_c}
