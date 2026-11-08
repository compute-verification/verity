"""What `predict.steps.compile_steps` costs on a class-G window, with the one part of `Compiled.summary()` that walks every node
(`transfer_gates`), as `advice_compile` runs them: GLM-5.3 at TP8 on H200, one rank's request Programs, GP-01 over them and the
batch rules, by phase, in wall seconds and resident memory.

A window is its requests' (arrival step, prompt length, max_tokens), greedy, scheduled as vLLM V1 schedules them (no preemption,
no prefix cache) at class G's 2,048 tokens a step and 256-row prefill chunks. A request Program reads only φ and its schedule
(LP, T, its chunks and launches), never its prompt's values, so a window's Program digests are those of the served window it
reproduces. `long-w0` is two prompts of 2,220 and 2,304 tokens, 15 decodes each.

    research run --on auto --cwd source --mem-gb 64 --timeout 60m -- bash -lc 'uv sync -q --all-packages --extra torch-cpu &&
      .venv/bin/python benchmarks/compile_cost/run.py --window long-w0 --out "$OUT" [--guard]'

writes OUT/compile_cost.json: the window, its digests, wall seconds and peak RSS (`ru_maxrss`), and each phase's wall seconds and
peak RSS while it ran (an RSS sampler thread), and prints `COMPILE-COST window=.. wall_min=.. peak_gib=.. digests=..`.
`--requests A:LP:MAX,...` gives the window directly, `--layers N` truncates the model to its first N layers (a scaling
measurement: its digests are another model's), `--profile` writes cProfile's stats (`profile.pstats`, and the top functions in
`profile.txt`). `--guard` exits 1 unless each digest is the one `PINS` holds for the window.

The two numbers are `long-w0`'s wall minutes and peak GiB on one core, against a target of 10 minutes and 16 GiB: a
production window (256 sequences, 8,192-token steps, 40k context) is compiled ahead of time from the same code.
"""
from __future__ import annotations

import argparse
import contextlib
import functools
import json
import resource
import sys
import threading
import time
from pathlib import Path

GLM53 = {"repo": "zai-org/GLM-5.3", "revision": "aca966e4e02791568aa6a4ced368624b3d897f42"}
TARGET = {"compute_capability": [9, 0], "num_sms": 132, "batch_invariant": False, "attention_backend": None,
          "vllm_revision": "d9105ea80", "numerics_profile": "b1-eager-v3", "flash_attn_version": None, "device_name": "NVIDIA H200",
          "moe_backend": "triton"}
ENGINE = {"max_num_batched_tokens": 2048, "max_model_len": 4096, "long_prefill_token_threshold": 256, "max_num_seqs": 256,
          "enable_prefix_caching": False}
WINDOWS = {
    "long-w0": [(0, 2220, 16), (2, 2304, 16)],
    "one-300": [(0, 300, 3)],
    "one-600": [(0, 600, 3)],
}
#: each window's request Program digests at rank 0, in request order
PINS: dict[str, list[str]] = {
    "long-w0": ["3304f85aea2585a1b41417f16f6a55a8ca0d106f32dbdc42a46d7ce22a1dd2dc432f2ce079937fc349a49058b6fe4ee43e68c28b76293becdb3884b086ed07f0",
                "821addd1e0ed89a1cdf1ae228421b6d7692fdbb988dbdf0beb5d9eb07c0bf8dcf0577722c935d4ab3e9a85766862d76d99e491c9fa1fa8c057860a1020e2e035"],
}


def scheduled(reqs: list[tuple[int, int, int]], budget: int, threshold: int, seqs: int = 256) -> list[list[tuple[int, int, int]]]:
    """vLLM V1's scheduler over requests (arrive_step, prompt_len, max_tokens), each step's rows as (request, start, n): the running
    requests in admission order, each its prompt rows left (or one decode row) up to `threshold` and the step's tokens left, then
    the arrived waiting requests while a sequence and a token are left. Steps with nothing to run are dropped."""
    done, outs = [0] * len(reqs), [0] * len(reqs)
    waiting, running, steps, e = sorted(range(len(reqs)), key=lambda i: reqs[i][0]), [], [], 0
    while waiting or running:
        left, rows = budget, []
        for i in running:
            if (n := min(max(reqs[i][1] - done[i], 1), threshold, left)):
                rows.append((i, done[i], n))
                left -= n
        while waiting and reqs[waiting[0]][0] <= e and left and len(running) < seqs:
            i = waiting.pop(0)
            running.append(i)
            rows.append((i, 0, min(reqs[i][1], threshold, left)))
            left -= rows[-1][2]
        for i, start, n in rows:
            done[i] = start + n
            outs[i] += done[i] >= reqs[i][1]
        running = [i for i in running if outs[i] < reqs[i][2]]
        steps.append(rows)
        e += 1
    return [s for s in steps if s]


def session(reqs: list[tuple[int, int, int]]):
    """The window's I1 advice and certified bodies: greedy requests `q<i>` to their max_tokens, prompts of token ids below 256."""
    from verity.core.protocols.certifier import advice as I1
    steps = scheduled(reqs, ENGINE["max_num_batched_tokens"], ENGINE["long_prefill_token_threshold"])
    requests = tuple(I1.RequestAdvice(f"q{i}", lp, m, "length") for i, (_, lp, m) in enumerate(reqs))
    advice = I1.Advice(requests, tuple(I1.StepAdvice(tuple(
        I1.Row(r, "decode" if n == 1 and start >= reqs[r][1] else "prefill", start, n) for r, start, n in st)) for st in steps),
        (0,) * sum(m for _, _, m in reqs))
    ingress = {f"q{i}": {"prompt": [(31 * i + 7 * k + 1) % 256 for k in range(lp)], "model": "m", "user": f"q{i}",
                         "temperature": 0.0, "max_tokens": m, "ignore_eos": True} for i, (_, lp, m) in enumerate(reqs)}
    return advice, ingress


def registered(rank: int, layers: int | None):
    """φ of GLM-5.3's checkpoint of record (its pinned config.json, FP8) on H200 at TP8 `rank`, class G's engine, the public seed."""
    from verity_vllm.engine.profiles.dense_generic import pinned_hf_config
    from verity_vllm.engine.served_seed import DEFAULT_SEED
    from verity_vllm.program.predict.phi import Phi
    from verity_vllm.program.predict.steps import Registered
    hf = json.loads(pinned_hf_config(GLM53["repo"], GLM53["revision"]).read_text())
    if layers is not None:
        hf = {**hf, "num_hidden_layers": layers, **{k: hf[k][:layers] for k in ("indexer_types", "mlp_layer_types")}}
    phi = Phi(model={**GLM53, "hf_config": hf, "dtype": "float8_e4m3fn"}, target=dict(TARGET), execution="eager",
              tp={"world": 8, "rank": rank}, engine=dict(ENGINE))
    return Registered.of(phi, default_seed=DEFAULT_SEED)


def _rss() -> int:
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * resource.getpagesize()


class Meter:
    """Phases by wall time and the peak RSS a sampler thread sees while each is open (nested phases nest)."""

    def __init__(self, every_s: float = 0.02) -> None:
        self.open: list[dict] = []
        self.done: list[dict] = []
        self.every_s, self.stop = every_s, threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()

    def _sample(self) -> None:
        while not self.stop.wait(self.every_s):
            r = _rss()
            for p in list(self.open):
                if r > p["peak"]:
                    p["peak"] = r

    @contextlib.contextmanager
    def phase(self, name: str):
        r = _rss()
        path = "/".join([q["phase"] for q in self.open] + [name])
        p = {"phase": name, "path": path, "depth": len(self.open), "t0": time.perf_counter(), "rss0": r, "peak": r}
        self.open.append(p)
        try:
            yield p
        finally:
            self.open.remove(p)
            r = _rss()
            self.done.append({"phase": name, "path": path, "depth": p["depth"], "wall_s": round(time.perf_counter() - p["t0"], 3),
                              "rss0_gib": round(p["rss0"] / 2**30, 3), "rss1_gib": round(r / 2**30, 3),
                              "peak_gib": round(max(p["peak"], r) / 2**30, 3)})

    def wrap(self, owner, attr: str, name: str | None = None) -> None:
        fn = getattr(owner, attr)

        @functools.wraps(fn)
        def wrapped(*a, **k):
            with self.phase(name or attr):
                return fn(*a, **k)
        setattr(owner, attr, wrapped)

    def by_phase(self) -> list[dict]:
        """The phases by call path, each under its caller, in order of first entry."""
        out: dict[str, dict] = {}
        for d in sorted(self.done, key=lambda d: d["path"].count("/")):
            o = out.setdefault(d["path"], {"phase": d["phase"], "path": d["path"], "depth": d["depth"], "calls": 0, "wall_s": 0.0,
                                           "peak_gib": 0.0})
            o["calls"] += 1
            o["wall_s"] = round(o["wall_s"] + d["wall_s"], 3)
            o["peak_gib"] = max(o["peak_gib"], d["peak_gib"])
        kids: dict[str, list[dict]] = {}
        for o in out.values():
            kids.setdefault(o["path"].rpartition("/")[0], []).append(o)
        order: list[dict] = []

        def walk(parent: str) -> None:
            for o in sorted(kids.get(parent, []), key=lambda o: -o["wall_s"]):
                order.append(o)
                walk(o["path"])
        walk("")
        return order


def instrument(m: Meter) -> None:
    from verity.ml.circuits import codec, defs

    # before the integration's modules bind it by name: each streamed digest is a phase, GP-01's (under compile_steps) and
    # each request Program's (under request_program)
    m.wrap(codec, "program_digest", "codec.program_digest")
    from verity_vllm.program.predict import batch, build, score, steps

    for owner, attrs in ((steps, ("plan_steps", "request_program", "dead_units")),
                         (build, ("decoder", "drive", "with_checks", "dead_free_form", "Program")),
                         (score, ("request_component",)),
                         (batch, ("check_compiled", "separable", "workload_labels", "_solo"))):
        for a in attrs:
            m.wrap(owner, a, f"{owner.__name__.rsplit('.', 1)[-1]}.{a}")
    finish = defs.Builder.finish

    def top_finish(self, *a, **k):
        if any(p["phase"] == "Builder.finish" for p in m.open):
            return finish(self, *a, **k)
        with m.phase("Builder.finish"):
            return finish(self, *a, **k)
    defs.Builder.finish = top_finish


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--window", choices=sorted(WINDOWS), default=None)
    ap.add_argument("--requests", default=None, help="ARRIVE:LP:MAX_TOKENS,... in place of --window")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--layers", type=int, default=None, help="the model's first N layers only (a scaling measurement)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--profile", action="store_true", help="cProfile the compile (profile.pstats, profile.txt)")
    ap.add_argument("--no-phases", action="store_true", help="no phase instrumentation")
    ap.add_argument("--guard", action="store_true", help="exit 1 unless the digests are the window's PINS")
    ap.add_argument("--engine", default=None, help="KEY=INT,... over class G's engine (max_model_len, max_num_batched_tokens, "
                    "long_prefill_token_threshold)")
    a = ap.parse_args()
    if a.engine:
        ENGINE.update({k: int(v) for k, v in (kv.split("=") for kv in a.engine.split(","))})
    if (a.window is None) == (a.requests is None):
        ap.error("one of --window and --requests")
    reqs = WINDOWS[a.window] if a.window else [tuple(int(x) for x in r.split(":")) for r in a.requests.split(",")]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    m = Meter()
    if not a.no_phases:
        instrument(m)
    from verity_vllm.program.predict.steps import compile_steps
    advice, ingress = session(reqs)
    reg = registered(a.rank, a.layers)
    prof = None
    if a.profile:
        import cProfile
        prof = cProfile.Profile()
    t0 = time.perf_counter()
    with m.phase("compile_steps"):
        if prof:
            prof.enable()
        compiled = compile_steps(reg, advice, ingress)
        if prof:
            prof.disable()
    with m.phase("transfer_gates"):
        transfer = compiled.transfer_gates
    wall = time.perf_counter() - t0
    m.stop.set()
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    digests = [u.digest for u in compiled.requests]
    nodes = {d: len(p.fn.body.nodes) for d, p in compiled.programs.items()}
    rec = {"window": a.window or a.requests, "requests": [list(r) for r in reqs], "rank": a.rank, "layers": a.layers, "engine": dict(ENGINE),
           "digests": digests, "workload_digest": compiled.workload.digest, "root_nodes": nodes, "transfer_gates": list(transfer), "wall_s": round(wall, 3),
           "wall_min": round(wall / 60, 3), "peak_gib": round(peak / 2**30, 3), "profiled": a.profile, "phases": m.by_phase(),
           "timeline": [d for d in m.done if d["phase"] != "Builder.finish"]}
    (out / "compile_cost.json").write_text(json.dumps(rec, indent=1) + "\n")
    if prof:
        import io
        import pstats
        prof.dump_stats(str(out / "profile.pstats"))
        s = io.StringIO()
        st = pstats.Stats(prof, stream=s)
        st.sort_stats("tottime").print_stats(60)
        st.sort_stats("cumulative").print_stats(80)
        (out / "profile.txt").write_text(s.getvalue())
    for p in rec["phases"]:
        print(f"  {'  ' * p['depth']}{p['phase']:<40} calls={p['calls']:<6} wall_s={p['wall_s']:<10} peak_gib={p['peak_gib']}")
    print(f"COMPILE-COST window={rec['window']} layers={a.layers} wall_min={rec['wall_min']} peak_gib={rec['peak_gib']} "
          f"nodes={sorted(nodes.values())} digests={','.join(d[:16] for d in digests)} workload={compiled.workload.digest[:16]}", flush=True)
    if a.guard:
        want = PINS.get(a.window or "")
        if a.layers is not None or want is None or digests != want:
            print(f"COMPILE-COST guard FAILED: digests {[d[:16] for d in digests]} against PINS {want}", flush=True)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
