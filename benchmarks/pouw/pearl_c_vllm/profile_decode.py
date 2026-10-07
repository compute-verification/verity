"""An untimed profile of Pearl-C mode on the smoke's chat prompts (`e2e.serve`'s ``chat``: vllm_bench's 16 prompts, 48 greedy
tokens), split into first use against steady state, prefill against decode, and GEMM, forming, hashing and other.

  python benchmarks/pouw/pearl_c_vllm/profile_decode.py --model DIR --kernel K --cubin C --out RUN_DIR/profile.json
  python benchmarks/pouw/pearl_c_vllm/profile_decode.py --nsys-kernels KERN_SUM.csv --out RUN_DIR/profile.json   # after nsys

``window.sh`` with ``MODE=profile`` runs the first under ``nsys profile --capture-range=cudaProfilerApi`` in a normal lease,
then ``nsys stats`` and the second.  One engine, the arm (BF16 with the Pearl-C wrapper, `e2e.install`), in this process, gated first (`e2e.gate`, the
screen's check included, untimed).
The passes, in order:
1. ``bf16-warm``: stock BF16, so vLLM's own first use is paid before Pearl-C's;
2. ``pearlc-first``: the device run's first use (its shapes, pipelines, the GEMMs' shared-memory probes and tensor maps);
3. ``pearlc-steady``: the same prompts again;
4. ``pearlc-nsys``: steady again, inside cudaProfilerStart/Stop, which nsys captures (NVTX ranges name each category);
5. ``bf16-steady``: the reference;
6. ``pearlc-cprofile``: steady under cProfile, for the host's hot spots (its times are inflated and not results).

Passes 2–5 record each forward's wall time (a forward pre-hook on the model; vLLM syncs each step when it samples) and its
token count, prefill being the pass's first forward.  For Pearl-C calls they also record the host time of each section
(`call_keys`, the key uploads, `screen`, each step's launch, the shapes and pipelines) and the GPU time of each step category
(CUDA events on the call's stream).  `CATEGORY` maps the steps; the pass end (its tile tree and roots) is a host section of
its own.  "Other" is a forward's wall time less the Pearl-C GPU categories: vLLM's own kernels, and all the host work, which
the sections itemize.
"""
from __future__ import annotations

import argparse
import contextlib
import cProfile
import csv
import io
import json
import os
import pstats
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

CATEGORY = {"gemm_pearlc": "gemm",
            **{x: "forming" for x in ("lines_ea", "stats_a", "eprime_a", "form_a", "af", "pa")},
            **{x: "hashing" for x in ("pad_a", "rowdig_a", "hash_rows_a", "tree_a", "seed_a", "seeds_a", "hash_leaf")}}
#: GPU 1's cubin's kernels by category, for nsys's per-kernel sums: short names exactly (``pa`` is not ``paged_attention``),
#: the rest by prefix; every other kernel is vLLM's or torch's
EXACT = {"lines": "forming", "eprime": "forming", "words": "forming", "words_u": "forming", "pa": "forming", "pb": "forming"}
PREFIXES = (("gemm_pearlc", "gemm"), ("gemm_ct", "gemm"), ("stats_s5", "forming"), ("form_s5", "forming"), ("gemm_words", "forming"),
            ("s512_", "hashing"), ("hm_", "hashing"))


def kernel_category(name: str) -> str:
    return EXACT.get(name) or next((c for prefix, c in PREFIXES if name.startswith(prefix)), "other")


class Profile:
    """Host sections and GPU step categories per (pass, forward kind), and each forward's wall time."""

    def __init__(self):
        import torch
        self.torch, self.phase, self.active, self.nvtx = torch, None, False, False
        self.host: dict = defaultdict(float)
        self.pending: list = []
        self.gpu: dict = defaultdict(float)
        self.forwards: dict = defaultdict(list)
        self.calls: dict = defaultdict(int)
        self.mark: tuple | None = None
        self.kind = "decode"
        self.cuda, self.hooked = torch.cuda.is_available(), False

    def begin(self, phase: str, active: bool, nvtx: bool = False) -> None:
        self.phase, self.active, self.nvtx, self.mark, self.kind = phase, active, nvtx, None, "prefill"

    def forward(self, tokens: int) -> None:
        """A forward starts: close the previous one's wall time.  The pass's first forward is its prefill."""
        now = time.perf_counter()
        if self.mark is not None:
            self.forwards[(self.phase, self.mark[1])].append((now - self.mark[0], self.mark[2]))
            self.kind = "decode"
        self.mark = (now, self.kind, tokens)

    def end(self) -> None:
        if self.cuda:
            self.torch.cuda.synchronize()
        if self.mark is not None:
            self.forwards[(self.phase, self.mark[1])].append((time.perf_counter() - self.mark[0], self.mark[2]))
            self.mark = None
        for key, a, b in self.pending:
            self.gpu[key] += a.elapsed_time(b)
        self.pending.clear()

    def section(self, name: str, f):
        """`f`, timed on the host under `name` while a pass is active."""
        def timed(*a, **kw):
            if not self.active:
                return f(*a, **kw)
            t0 = time.perf_counter()
            try:
                return f(*a, **kw)
            finally:
                self.host[(self.phase, self.kind, name)] += time.perf_counter() - t0
        return timed

    def step(self, name: str, f):
        """A pipeline step: its launch's host time, and its GPU time under its category, with an NVTX range."""
        cat = CATEGORY.get(name, "weight side")
        torch = self.torch

        def timed():
            if not self.active:
                return f()
            t0 = time.perf_counter()
            if self.nvtx:
                torch.cuda.nvtx.range_push(f"pearlc:{cat}:{name}")
            if self.cuda:
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
            f()
            if self.cuda:
                b.record()
                self.pending.append(((self.phase, self.kind, cat), a, b))
            if self.nvtx:
                torch.cuda.nvtx.range_pop()
            self.host[(self.phase, self.kind, f"launch {cat}")] += time.perf_counter() - t0
        return timed

    def report(self) -> dict:
        out: dict = {}
        for (phase, kind), fw in sorted(self.forwards.items()):
            walls = [w for w, _ in fw]
            d = out.setdefault(phase, {}).setdefault(kind, {})
            d.update(forwards=len(fw), tokens=[t for _, t in fw][:3], wall_s=sum(walls), median_forward_ms=1e3 * statistics.median(walls))
        for (phase, kind, name), s in self.host.items():
            out.setdefault(phase, {}).setdefault(kind, {}).setdefault("host_ms", {})[name] = round(1e3 * s, 3)
        for (phase, kind, cat), ms in self.gpu.items():
            out.setdefault(phase, {}).setdefault(kind, {}).setdefault("gpu_ms", {})[cat] = round(ms, 3)
        for (phase, kind), n in self.calls.items():
            out.setdefault(phase, {}).setdefault(kind, {})["pearlc_calls"] = n
        for phase in out.values():
            for d in phase.values():
                if "wall_s" in d:
                    gpu = sum(d.get("gpu_ms", {}).values())
                    d["split_ms"] = {**{c: d.get("gpu_ms", {}).get(c, 0.0) for c in ("gemm", "forming", "hashing")},
                                     "other": round(1e3 * d["wall_s"] - gpu, 3)}
                    d["host_pearlc_ms"] = round(sum(d.get("host_ms", {}).values()), 3)
        return out


def instrument(prof: Profile):
    """Wrap the device run's module functions before the install, so every call's steps and sections are timed.  Returns
    the function that puts them back."""
    from verity_vllm.protocol_options import pouw_pearl_c_device as PPD
    run_cls = PPD.PearlCDeviceRun
    saved = [(PPD, n, getattr(PPD, n)) for n in ("_steps", "screen_into", "excluded")]
    saved += [(run_cls, n, vars(run_cls)[n]) for n in ("_shape", "_pipe", "end", "gemm")]
    saved += [(PPD._Keys, "__call__", vars(PPD._Keys)["__call__"]), (PPD._Uploads, "put", vars(PPD._Uploads)["put"])]
    steps = PPD._steps

    def named_steps(pipe, names):
        present = {n for n, _, _ in pipe.steps()}
        return [prof.step(name, f) for name, f in zip([x for x in names if x in present], steps(pipe, names), strict=True)]

    PPD._steps = named_steps
    PPD.screen_into = prof.section("screen", PPD.screen_into)
    PPD.excluded = prof.section("excluded", PPD.excluded)
    PPD._Keys.__call__ = prof.section("call_keys", PPD._Keys.__call__)
    PPD._Uploads.put = prof.section("_put", PPD._Uploads.put)
    run_cls._shape = prof.section("shapes (first use: buffers)", run_cls._shape)
    run_cls._pipe = prof.section("pipelines (first use: build, tensor maps)", run_cls._pipe)
    end = run_cls.end
    run_cls.end = lambda self: prof.section("pass end (tile tree, roots)", end)(self)
    gemm = run_cls.gemm

    def counted(self, w, x, weight, bias):
        if prof.active:
            if w == 0 and not prof.hooked:
                prof.forward(int(x.reshape(-1, x.shape[-1]).shape[0]))
            prof.calls[(prof.phase, prof.kind)] += 1
        return prof.section("gemm wrapper (all of it)", gemm)(self, w, x, weight, bias)
    run_cls.gemm = counted

    def undo() -> None:
        for owner, name, f in saved:
            setattr(owner, name, f)
    return undo


def run(args) -> dict:
    import e2e
    import torch
    from vllm import SamplingParams
    from vllm_bench import PROMPTS, find_model

    prof = Profile()
    instrument(prof)
    llm = e2e.engine(args, "arm")
    model = find_model(llm)
    t0 = time.perf_counter()
    run_, weights, stack = e2e.install(model, args, os.urandom(32))
    install_s = time.perf_counter() - t0
    gate = e2e.gate(run_, weights)

    def hook(_module, a, kw):
        ids = kw.get("input_ids") if kw.get("input_ids") is not None else (a[0] if a else None)
        prof.hooked = True
        prof.forward(int(ids.shape[0]) if ids is not None else -1)
    model.register_forward_pre_hook(hook, with_kwargs=True)
    tok = llm.get_tokenizer()
    chat = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in PROMPTS]
    sp = SamplingParams(temperature=0.0, max_tokens=48)
    walls = {}

    def chat_pass(phase: str, pearl: bool, nvtx: bool = False, cprof: cProfile.Profile | None = None) -> None:
        stack.activate(e2e.arm_of("pearlc" if pearl else "bf16", run_, args))
        prof.begin(phase, pearl, nvtx)
        with stack.pass_() if pearl else contextlib.nullcontext():
            torch.cuda.synchronize()
            t = time.perf_counter()
            if cprof:
                cprof.enable()
            llm.generate(chat, sp, use_tqdm=False)
        if cprof:
            cprof.disable()
        prof.end()
        run_.gemm_ms()
        walls[phase] = time.perf_counter() - t

    chat_pass("bf16-warm", False)
    chat_pass("pearlc-first", True)
    chat_pass("pearlc-steady", True)
    torch.cuda.cudart().cudaProfilerStart()
    chat_pass("pearlc-nsys", True, nvtx=True)
    torch.cuda.cudart().cudaProfilerStop()
    chat_pass("bf16-steady", False)
    cprof = cProfile.Profile()
    chat_pass("pearlc-cprofile", True, cprof=cprof)
    buf = io.StringIO()
    pstats.Stats(cprof, stream=buf).sort_stats("tottime").print_stats(30)
    return {"schema": "verity-pouw/pearl-c-vllm-profile/v1", "run": os.environ.get("RESEARCH_RUN_ID"), "install_s": install_s,
            "host": run_.details["host"], "gate_ok": all(g["ok"] for g in gate), "gate": gate,
            "prompts": len(chat), "max_tokens": 48, "pass_wall_s": walls, "passes": prof.report(),
            "cprofile_top_tottime": buf.getvalue().splitlines()[:60]}


def nsys_kernels(path: Path) -> dict:
    """nsys stats' cuda_gpu_kern_sum CSV (pass 4's steady-state capture): the GPU time per category and the top kernels."""
    rows = list(csv.DictReader(path.open()))
    total_col = next(c for c in rows[0] if c.startswith("Total Time"))
    by_cat: dict = defaultdict(float)
    top = []
    for r in rows:
        ns, name = float(r[total_col]), r["Name"]
        by_cat[kernel_category(name)] += ns / 1e6
        top.append({"kernel": name[:120], "category": kernel_category(name), "ms": round(ns / 1e6, 3), "instances": int(r["Instances"])})
    top.sort(key=lambda k: -k["ms"])
    return {"gpu_ms_by_category": {k: round(v, 3) for k, v in by_cat.items()}, "top_kernels": top[:25]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model")
    ap.add_argument("--revision")
    ap.add_argument("--kernel", default=str(HERE.parent / "pearl_c_sm120"))
    ap.add_argument("--cubin")
    ap.add_argument("--mem-arm", type=float, default=0.3)
    ap.add_argument("--mem-base", type=float, default=0.2)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--nsys-kernels", type=Path, help="merge nsys's per-kernel sums into --out, and stop")
    ap.add_argument("--scheme", default="pearl-c-sm120-v1-h3", help="the Pearl-C scheme, and so its hashing format")
    ap.add_argument("--schedule", choices=["serial", "deferred", "deferred-caller"], default="serial")
    args = ap.parse_args()
    if args.nsys_kernels:
        doc = json.loads(args.out.read_text())
        doc["nsys_steady"] = nsys_kernels(args.nsys_kernels)
        args.out.write_text(json.dumps(doc, indent=1))
        print(json.dumps(doc["nsys_steady"]["gpu_ms_by_category"]))
        return
    doc = run(args)
    args.out.write_text(json.dumps(doc, indent=1))
    print(json.dumps({p: {k: v.get("split_ms") for k, v in d.items()} for p, d in doc["passes"].items()}), flush=True)


if __name__ == "__main__":
    main()
