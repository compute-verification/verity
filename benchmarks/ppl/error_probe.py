"""The V4 proxy's error probe (CMP-155): an arm's per-matmul relative error on synthetic operands of a model's shapes, so
that a model the bench runs can be given another's error (`steps.scale_error`), and the factors that do it.

    error_probe.py measure --out DIR [--arm A] [--tau 1,2,3] [--beta 0,4,8] [--seeds 0,1] [--workers W] [--threads T]
    error_probe.py factors PROBE.json --fit RESULTS.json [--spec OUT.toml [--upper]]

Linears: x (ROWS x k) with lognormal channel scales and a few outlier channels, W (n x k) Gaussian, through the arm's
linear chain (lm_head's own steps for the head; smoothing reads a calibration batch of the same x), r = |y - x W^T| /
|x W^T| in float64. n is capped at N_CAP rows (an output row's error does not depend on n), so a class's r is a
function of k. Attention: one KV group per (model, context), `rep` query heads of width d on K and V Gaussian, key 0 a
sink (its logit beta above the rest's, its value a tenth of theirs), the other logits of std tau; `Layer.edited_attention`
with the arm's attention chain, whole and as its Q K^T steps or its P V steps alone, r its check-window relative error
(the engine's `attn_rel_err_w0`). V4's sparse attention is causal attention over as many keys per query on average:
CSA at 2K about 128 + t / 4 (as a causal length 768), HCA 128 + t / 128 (272), pooled over its 30 and 31 layers.

`factors` fits (tau, beta) to DeepSeek-V2-Lite's measured attention errors (its results.json) of the arms that differ
from the adopted arm only in P V's block (`fit_targets`), and gives V4's r over V2-Lite's per class at that fit, and
the attention factors' range over the grid points near it; it checks the fit's context scaling on Llama-3.1-8B (fitted
at 2K, predicted at 8K, against H1's measured 8K errors)."""
from __future__ import annotations

import argparse
import itertools
import json
import math
import multiprocessing
import sys
import types
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import gates as G  # noqa: E402
import matmul  # noqa: E402
import methodology  # noqa: E402
import model as M  # noqa: E402
import spec  # noqa: E402
import steps  # noqa: E402,F401

ROWS, CAL_ROWS, N_CAP = 512, 256, 2048
#: each linear class's (k, n) per model; a class of two linears in sequence (V4's q, kv and o) adds their errors
LINEARS = {
    "llama-3.1-8b": {"q": [(4096, 4096)], "kv": [(4096, 1024)], "o": [(4096, 4096)], "gate_up": [(4096, 28672)],
                     "down": [(14336, 4096)], "head": [(4096, 128256)]},
    "llama-3.1-70b": {"q": [(8192, 8192)], "kv": [(8192, 1024)], "o": [(8192, 8192)], "gate_up": [(8192, 57344)],
                      "down": [(28672, 8192)], "head": [(8192, 128256)]},
    "deepseek-v2-lite": {"q": [(2048, 3072)], "kv": [(2048, 576), (512, 4096)], "o": [(2048, 2048)],
                         "gate_up": [(2048, 2816)], "down": [(1408, 2048)], "shared_gate_up": [(2048, 2816)],
                         "shared_down": [(2816, 2048)], "dense_gate_up": [(2048, 10944)],
                         "dense_down": [(10944, 2048)], "head": [(2048, 102400)]},
    "deepseek-v4-pro": {"q": [(7168, 1536), (1536, 65536)], "kv": [(7168, 512)], "o": [(4096, 1024), (16384, 7168)],
                        "gate_up": [(7168, 6144)], "down": [(3072, 7168)], "shared_gate_up": [(7168, 6144)],
                        "shared_down": [(3072, 7168)], "dense_gate_up": [(7168, 6144)], "dense_down": [(3072, 7168)],
                        "head": [(7168, 129280)]},
}
#: one KV group's attention: query heads per KV head, Q/K width, V width, keys (causal), KV groups sampled
ATTENTION = {
    "llama-3.1-8b@2048": {"rep": 4, "d": 128, "dv": 128, "s": 2048, "groups": 1},
    "llama-3.1-8b@8192": {"rep": 4, "d": 128, "dv": 128, "s": 8192, "groups": 1},
    "llama-3.1-70b@2048": {"rep": 8, "d": 128, "dv": 128, "s": 2048, "groups": 1},
    "deepseek-v2-lite@2048": {"rep": 1, "d": 192, "dv": 128, "s": 2048, "groups": 4},
    "deepseek-v4-pro-csa@2048": {"rep": 128, "d": 512, "dv": 512, "s": 768, "groups": 1},
    "deepseek-v4-pro-hca@2048": {"rep": 128, "d": 512, "dv": 512, "s": 272, "groups": 1},
}
V4_LAYERS = {"deepseek-v4-pro-csa@2048": 30, "deepseek-v4-pro-hca@2048": 31}
#: the arms each attention shape runs whole (the measured ones), besides `--arm`'s parts
ATTN_ARMS = {"llama-3.1-8b@2048": ["int8-all", "adopted-bd-lm-pv256", "adopted-bd-lm-pv512", "adopted-bd-lm-pv1024",
                                   "adopted-bd-lm-pv"],
             "llama-3.1-8b@8192": ["int8-all", "adopted-bd-lm-pv1024", "adopted-bd-lm-pv"],
             "llama-3.1-70b@2048": ["int8-all", "adopted-bd-lm-pv256", "adopted-bd-lm-pv512", "adopted-bd-lm-pv1024",
                                    "adopted-bd-lm-pv"],
             "deepseek-v2-lite@2048": ["int8-all", "adopted-bd-lm-pv1024", "adopted-bd-lm-pv"],
             "deepseek-v4-pro-csa@2048": ["int8-all", "adopted-bd-lm-pv1024", "adopted-bd-lm-pv"],
             "deepseek-v4-pro-hca@2048": ["int8-all", "adopted-bd-lm-pv1024", "adopted-bd-lm-pv"]}
#: H1's measured attn_rel_err_w0 for Llama-3.1-8B (seed 0): r20261017-065449-b05b (2K), r20261017-070733-6851 (8K)
LLAMA_8B = {"llama-3.1-8b@2048": {"adopted-bd-lm-pv256": 0.062, "adopted-bd-lm-pv512": 0.070,
                                  "adopted-bd-lm-pv1024": 0.084, "int8-all": 0.1415},
            "llama-3.1-8b@8192": {"adopted-bd-lm-pv1024": 0.0831, "adopted-bd-lm-pv": 0.1767, "int8-all": 0.21}}
#: the classes of DeepSeek-V2-Lite's linears, by module name (`spec.EXPERT` form)
V2_CLASSES = {"q": ["self_attn.q_proj"], "kv": ["self_attn.kv_a_proj_with_mqa", "self_attn.kv_b_proj"],
              "o": ["self_attn.o_proj"], "gate_up": ["mlp.experts.E.gate_up_proj"], "down": ["mlp.experts.E.down_proj"],
              "shared_gate_up": ["mlp.shared_experts.gate_proj", "mlp.shared_experts.up_proj"],
              "shared_down": ["mlp.shared_experts.down_proj"], "dense_gate_up": ["mlp.gate_proj", "mlp.up_proj"],
              "dense_down": ["mlp.down_proj"], "head": [spec.HEAD]}


def part(arm, kind: str):
    """The arm with only its `kind` (qk or pv) attention steps, each acting on that matmul alone; no linear steps."""
    import dataclasses
    att = tuple(dataclasses.replace(s, defn=dataclasses.replace(s.defn, on=(kind,))) for s in arm.attention
                if kind in s.defn.on)
    return dataclasses.replace(arm, name=f"{arm.name}:{kind}", linear=(), attention=att)


def linear_operands(k: int, n: int, g: torch.Generator):
    scale = torch.exp(torch.randn(k, generator=g))
    scale[torch.randperm(k, generator=g)[:max(1, k // 200)]] *= 10
    x = torch.randn(ROWS + CAL_ROWS, k, generator=g) * scale
    return x[:ROWS], x[ROWS:], torch.randn(min(n, N_CAP), k, generator=g) / math.sqrt(k)


def linear_error(arm, name: str, x, cal, w, seeds) -> float:
    lay = types.SimpleNamespace(li=0, groups={name: "g"}, weight=lambda _n: w, cfg=None)
    y0 = x.double() @ w.double().T
    num = den = 0.0
    for seed in seeds:
        run = matmul.Run(arm, seed, 1, lay, G.Gates(), {}, {}, {"amax": {"g": cal.abs().amax(0)}})
        y = matmul.run_chain(run.steps_for("linear", name), matmul.MM("linear", name, run, x, w))
        if not run.gates.passed:
            raise RuntimeError(f"{arm.name} {name}: gates {run.gates.log}")
        num += float(((y.double() - y0) ** 2).sum())
        den += float((y0 ** 2).sum())
    return math.sqrt(num / den)


def attention_operands(c: dict, tau: float, beta: float, g: torch.Generator):
    """Q (1, groups rep, s, d) zero-padded to a power of 2 wide, K (1, groups, s, ...), V (1, groups, s, dv)."""
    rep, d, dv, s, G_ = c["rep"], c["d"], c["dv"], c["s"], c["groups"]
    q = tau * torch.randn(1, G_ * rep, s, d, generator=g)
    k = torch.randn(1, G_, s, d, generator=g)
    v = torch.randn(1, G_, s, dv, generator=g)
    sink = 2 * tau
    q[..., 0] = sink
    k[..., 0] = 0
    k[:, :, 0] = 0
    k[:, :, 0, 0] = beta * math.sqrt(d) / sink
    v[:, :, 0] *= 0.1
    pad = (0, (1 << (d - 1).bit_length()) - d)
    return F.pad(q, pad), F.pad(k, pad), v


def attention_error(arm, q, k, v, d: int, seeds) -> float:
    lay = types.SimpleNamespace(li=0, scaling=d ** -0.5)
    num = den = 0.0
    for seed in seeds:
        run = matmul.Run(arm, seed, methodology.CHECK_WINDOW, lay, G.Gates(), {}, {}, None)
        M.Layer.edited_attention(lay, q, k, v, run)
        if not run.gates.passed:
            raise RuntimeError(f"{arm.name}: gates {run.gates.log}")
        num, den = num + run.attn_err[0], den + run.attn_err[1]
    return math.sqrt(num / den)


def task(args) -> list[dict]:
    what, key, params, arm_name, seeds, threads = args
    torch.set_num_threads(threads)
    g = torch.Generator().manual_seed(zlib.crc32(repr((what, key, params)).encode()))
    arm = spec.load(arm_name)
    out = []
    if what == "linear":
        model, cls = key
        for i, (k, n) in enumerate(LINEARS[model][cls]):
            x, cal, w = linear_operands(k, n, g)
            for a in sorted({arm_name, "int8-all"}):
                name = spec.HEAD if cls == "head" else "mlp.up_proj"
                out.append({"what": "linear", "model": model, "class": cls, "i": i, "k": k, "n": n, "arm": a,
                            "r": linear_error(spec.load(a), name, x, cal, w, seeds)})
        return out
    tau, beta = params
    c = ATTENTION[key]
    q, k, v = attention_operands(c, tau, beta, g)
    arms = [(a, spec.load(a)) for a in ATTN_ARMS[key]] + [(f"{arm_name}:{p}", part(arm, p)) for p in ("qk", "pv")]
    for a, x in arms:
        out.append({"what": "attention", "shape": key, "tau": tau, "beta": beta, "arm": a,
                    "r": attention_error(x, q, k, v, c["d"], seeds)})
    return out


def cmd_measure(a) -> int:
    seeds = tuple(int(s) for s in a.seeds.split(","))
    taus, betas = [float(x) for x in a.tau.split(",")], [float(x) for x in a.beta.split(",")]
    jobs = [("linear", (m, c), (), a.arm, seeds, a.threads) for m in LINEARS for c in LINEARS[m]]
    jobs += [("attention", s, (t, b), a.arm, seeds, a.threads) for s in ATTENTION for t, b in itertools.product(taus, betas)]
    jobs.sort(key=lambda j: -ATTENTION[j[1]]["s"] if j[0] == "attention" else 0)
    rows = []
    with multiprocessing.get_context("spawn").Pool(a.workers) as pool:
        for r in pool.imap_unordered(task, jobs):
            rows += r
            for x in r:
                print(json.dumps(x), flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "probe.json").write_text(json.dumps({"arm": a.arm, "seeds": list(seeds), "rows": rows}, indent=1))
    return 0


def misfits(rows, shape: str, targets: dict) -> dict:
    """(tau, beta) -> the RMS of the log misfit of its errors on `shape` against `targets` (arm -> r)."""
    grid: dict = {}
    for r in rows:
        if r["what"] == "attention" and r["shape"] == shape and r["arm"] in targets:
            grid.setdefault((r["tau"], r["beta"]), {})[r["arm"]] = r["r"]
    return {th: math.sqrt(sum(math.log(v[t] / targets[t]) ** 2 for t in targets) / len(targets))
            for th, v in grid.items() if set(v) == set(targets)}


def fit(rows, shape: str, targets: dict) -> tuple:
    """(tau, beta) whose errors on `shape` are nearest `targets` in log, and the RMS of the log misfit."""
    m = misfits(rows, shape, targets)
    th = min(m, key=lambda t: (m[t], t))
    return th, m[th]


def fit_targets(measured: dict) -> dict:
    """The arms a fit reads: those that differ from the adopted arm only in P V's block. The synthetic Q and K have no
    outlier channels, so int8-all's Q K^T error, which they drive, is not a target."""
    return {a: v for a, v in measured.items() if a != "int8-all"}


def attn(rows, shape: str, arm: str, th: tuple) -> float:
    return next(r["r"] for r in rows if r["what"] == "attention" and r["shape"] == shape and r["arm"] == arm
                and (r["tau"], r["beta"]) == th)


def lin(rows, model: str, cls: str, arm: str) -> float:
    """A class's r: its linears' errors added in quadrature."""
    return math.sqrt(sum(r["r"] ** 2 for r in rows if r["what"] == "linear" and r["model"] == model
                         and r["class"] == cls and r["arm"] == arm))


#: a grid point within this much log-RMS misfit of the best fit is near it: the attention factors' range is over those
NEAR = 0.05


def attention_factors(rows, arm: str, th: tuple) -> dict:
    """V4's Q K^T, P V and whole-attention error over V2-Lite's at (tau, beta), V4's pooled RMS over its CSA and HCA
    layers."""
    def v4(a):
        return math.sqrt(sum(n * attn(rows, s, a, th) ** 2 for s, n in V4_LAYERS.items()) / sum(V4_LAYERS.values()))
    names = {"qk": f"{arm}:qk", "pv": f"{arm}:pv", "attention_whole": arm}
    return {k: v4(a) / attn(rows, "deepseek-v2-lite@2048", a, th) for k, a in names.items()}


def factors(probe: dict, measured: dict) -> dict:
    """V4's per-class error over V2-Lite's, for `probe`'s arm: the linear classes, and Q K^T and P V at the (tau, beta)
    fitted to V2-Lite's measured attention errors (`measured`, arm -> r; `fit_targets` of it), with their range over
    the grid points near that fit."""
    rows, arm = probe["rows"], probe["arm"]
    out = {"linear": {c: lin(rows, "deepseek-v4-pro", c, arm) / lin(rows, "deepseek-v2-lite", c, arm)
                      for c in LINEARS["deepseek-v2-lite"]}}
    targets = fit_targets(measured)
    m = misfits(rows, "deepseek-v2-lite@2048", targets)
    th, misfit = fit(rows, "deepseek-v2-lite@2048", targets)
    out["fit"] = {"tau": th[0], "beta": th[1], "log_rms_misfit": misfit, "measured": measured,
                  "probe": {a: attn(rows, "deepseek-v2-lite@2048", a, th) for a in measured}}
    out |= attention_factors(rows, arm, th)
    near = sorted(t for t, v in m.items() if v <= misfit + NEAR)
    per = [attention_factors(rows, arm, t) for t in near]
    out["near"] = {"points": [list(t) for t in near],
                   **{k: [min(p[k] for p in per), max(p[k] for p in per)] for k in ("qk", "pv", "attention_whole")}}
    th8, mis8 = fit(rows, "llama-3.1-8b@2048", fit_targets(LLAMA_8B["llama-3.1-8b@2048"]))
    out["llama_8b_check"] = {"tau": th8[0], "beta": th8[1], "log_rms_misfit_2k": mis8,
                             "predicted_8k": {a: attn(rows, "llama-3.1-8b@8192", a, th8) for a in LLAMA_8B["llama-3.1-8b@8192"]},
                             "measured_8k": LLAMA_8B["llama-3.1-8b@8192"]}
    return out


def spec_toml(f: dict, arm: str, upper: bool = False) -> str:
    """The V4-scaled arm: scale_error on every V2-Lite linear class and on attention, then `arm`'s steps. `upper` takes
    the attention factors' upper end over the points near the fit instead of the fit's."""
    base = spec.load(arm).to_toml()
    qk, pv = (f["near"]["qk"][1], f["near"]["pv"][1]) if upper else (f["qk"], f["pv"])
    at = "the upper end over the points near the fit" if upper else "the fit"
    lines = [f'description = "{arm} with each matmul\'s error scaled to DeepSeek-V4-Pro\'s (error_probe.py factors: '
             f'V4 over V2-Lite per class, attention at {at} to V2-Lite\'s measured errors; CMP-155 proxy)"', ""]
    for c, names in V2_CLASSES.items():
        lines += ["[[linear]]", 'step = "scale_error"', f"factor = {f['linear'][c]:.4f}",
                  "linears = [" + ", ".join(f'"{n}"' for n in names) + "]", ""]
    for st in base["linear"]:
        lines += ["[[linear]]"] + [f"{k} = {json.dumps(v)}" for k, v in st.items()] + [""]
    lines += ["[[attention]]", 'step = "scale_error"', f"qk = {qk:.4f}", f"pv = {pv:.4f}", ""]
    for st in base["attention"]:
        lines += ["[[attention]]"] + [f"{k} = {json.dumps(v)}" for k, v in st.items()] + [""]
    return "\n".join(lines)


def measured_attention(results: dict) -> dict:
    """arm -> attn_rel_err_w0, pooled over its seeds, from a run's results.json."""
    out = {}
    for name, e in results["arms"].items():
        v = list(e.get("attn_rel_err_w0", {}).values())
        if v:
            out[name] = math.sqrt(sum(x * x for x in v) / len(v))
    return out


def cmd_factors(a) -> int:
    probe = json.loads(Path(a.probe).read_text())
    measured = {k: v for k, v in measured_attention(json.loads(Path(a.fit).read_text())).items()
                if k in ATTN_ARMS["deepseek-v2-lite@2048"]}
    f = factors(probe, measured)
    print(json.dumps(f, indent=1))
    if a.spec:
        Path(a.spec).write_text(spec_toml(f, probe["arm"], a.upper))
    return 0


def main(argv) -> int:
    p = argparse.ArgumentParser(prog="error_probe.py", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("measure")
    q.add_argument("--out", required=True)
    q.add_argument("--arm", default="adopted-bd-lm-pv")
    q.add_argument("--tau", default="1,1.5,2,2.5,3,4")
    q.add_argument("--beta", default="0,3,6,9,12,15,18")
    q.add_argument("--seeds", default="0,1")
    q.add_argument("--workers", type=int, default=8)
    q.add_argument("--threads", type=int, default=8)
    q = sub.add_parser("factors")
    q.add_argument("probe")
    q.add_argument("--fit", required=True, help="DeepSeek-V2-Lite's results.json")
    q.add_argument("--spec", help="write the V4-scaled arm's TOML here")
    q.add_argument("--upper", action="store_true", help="the spec's attention factors: the near points' upper end")
    a = p.parse_args(argv)
    return {"measure": cmd_measure, "factors": cmd_factors}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
