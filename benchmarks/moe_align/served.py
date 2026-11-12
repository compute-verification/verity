"""DeepSeek-V4-Pro served TP8 eager with the canonical `moe_align_block_size` on its Marlin MoE path
(`verity_vllm.engine.moe_align`, installed in every rank by the integration's vLLM plugin and the rank worker's import), on one
8 x H200 node. Measurement only.

    bash benchmarks/moe_align/served.sh      (its header has the parameters)

One engine; every batch is the workload's first request with its prompt rotated by ``--rotate`` tokens per request
(`served_commit.tp_decode.requests_at`), all arriving at once and decoding to the cap. Its prefill budget
(``max_num_batched_tokens``) holds the largest batch's prompts, so every batch prefills in one step and then decodes with every
row; at the row's own budget the first requests of 256 would finish before the last began. Three phases, written to ``--out``:

- ``check.json``: at ``--check-batches`` (default 1 and 8: a prefill step of one prompt, under `moe_align.SMALL_MAX` pairs, and
  of eight at once, over it), a wrap over each rank's Marlin `moe_align_block_size` holds every call's three outputs to
  `moe_align.reference(topk_ids, num_experts, block_size)` bit for bit, by MoE layer, and counts the calls by step kind (a
  prefill step is one with more tokens than the batch has requests) and by the `Align` path the call took; with each rank's
  `Align.record` before and after, and the calls of vLLM's own aligners (`VLLM_ALIGNERS`), which must be none.
- ``decode.jsonl``: the decode step (TPOT, `served_commit.tp_decode.phases`) at ``--batches`` rows, the canonical op against
  vLLM's (`moe_align.uninstall` on every rank, then `moe_align.install` of the same `Align`), interleaved over ``--pairs``
  pairs after a warm-up of each, with a digest of each run's tokens.
- ``host.json``: per batch, op and rank, the host's time to issue a call of `moe_align_block_size` (no sync, eager), from one
  more run of each op under a timing wrap, split into decode and prefill calls.

``summary.json`` gathers them: per rank the MoE layers whose every call equalled the reference, the paths each rank took in
  prefill and in decode, per batch the TPOT of each op and the paired ratios, and the host cost per call and per decode step.

With ``--vary-repeats N``, ``vary.json``: at ``--vary-batches`` (default ``--check-batches``), for vLLM's op and then the
canonical one, each call Marlin makes repeated N more times on its own ``topk_ids`` (`Repeater`): per rank and step kind, the
calls whose outputs were not all one, among those where some expert holds more than one pair, the outputs that were the
reference's, and whether every output held the reference's pairs per expert. It runs between the check and the decode phases.
Exits 3 when a rank does not serve the canonical op or its Marlin experts do not call the module's aligner (`MA.loads_global`,
the guard `install` holds), an MoE layer's calls miss the op (they reach no `Align`, reach one of vLLM's aligners, or the layer
made none), a call differs from the reference, or a rank took either path in no prefill step.

``--dummy-layers N`` serves the model without its weights: the checkpoint's small files (`DUMMY_FILES`) fetched at its pinned
revision and held to the manifest's sha256, its config cut to the first N layers (`cut_config`), and ``load_format="dummy"``.
DeepSeek-V4-Pro has no dense layer: its first ``num_hash_layers`` (3) route by a token -> experts table, which vLLM fills with
random experts under dummy weights, and the rest by score, so N = 5 keeps both kinds and both attention compress ratios. The
expert count, top-k and widths are the checkpoint's; the routing distribution is not, so the decode figures are labelled
``weights: dummy``.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

from served_commit.tp_decode import phases, requests_at
from verity_vllm.engine import moe_align as MA

UNATTRIBUTED = "<outside every MoE layer>"
#: vLLM's MoE layer classes, by name since they vary across its versions; a decoder layer's outermost one is its MoE layer
MOE_CLASSES = {"MoERunner", "MoERunnerInterface", "FusedMoE", "SharedFusedMoE"}
#: vLLM's aligner ops (``vllm._custom_ops``): with the canonical op on, the Marlin experts call none of them
VLLM_ALIGNERS = ("moe_align_block_size", "batched_moe_align_block_size", "moe_lora_align_block_size")
#: rank-local state of the phases (`check_on`, `use`, `timer_on`)
_STATE: dict[str, Any] = {}
#: the checkpoint's files a ``--dummy-layers`` engine reads: everything but the weights
DUMMY_FILES = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json")


def _tp_rank() -> int:
    from vllm.distributed import get_tensor_model_parallel_rank

    return int(get_tensor_model_parallel_rank())


def _site() -> Any:
    return importlib.import_module(MA.SITE)


def _ops() -> Any:
    return importlib.import_module("vllm._custom_ops")


def moe_layers(model: Any) -> list[str]:
    """The name of each MoE layer of ``model``: per decoder layer, its outermost module whose class is one of `MOE_CLASSES`."""
    out: list[str] = []
    for name, m in model.named_modules():
        if ".layers." in name and {c.__name__ for c in type(m).__mro__} & MOE_CLASSES and not any(name.startswith(n + ".") for n in out):
            out.append(name)
    return out


def cut_config(cfg: dict[str, Any], layers: int) -> dict[str, Any]:
    """``cfg`` with its first ``layers`` decoder layers: ``num_hidden_layers`` and the per-layer ``compress_ratios``, whose entries
    past the decoder layers (the MTP layer's) stay."""
    n = int(cfg["num_hidden_layers"])
    if not 0 < layers <= n:
        raise SystemExit(f"--dummy-layers {layers}: the checkpoint has {n} layers")
    out = {**cfg, "num_hidden_layers": layers}
    if isinstance(cfg.get("compress_ratios"), list):
        out["compress_ratios"] = cfg["compress_ratios"][:layers] + cfg["compress_ratios"][n:]
    return out


def dummy_model(entry: dict[str, Any], layers: int, where: Path) -> dict[str, Any]:
    """Write into ``where`` the checkpoint's `DUMMY_FILES`, fetched from the hub at ``entry``'s revision and each held to the
    manifest's sha256, with config.json cut to ``layers`` layers; what was fetched and cut."""
    import urllib.request

    got = {}
    for name in DUMMY_FILES:
        url = f"https://huggingface.co/{entry['repo']}/resolve/{entry['revision']}/{name}"
        with urllib.request.urlopen(url, timeout=300) as r:
            data = r.read()
        sha = hashlib.sha256(data).hexdigest()
        if sha != entry["files"][name]["sha256"]:
            raise SystemExit(f"{url}: sha256 {sha}, not the manifest's {entry['files'][name]['sha256']}")
        got[name] = sha
        (where / name).write_bytes(data)
    cfg = json.loads((where / "config.json").read_text())
    cut = cut_config(cfg, layers)
    (where / "config.json").write_text(json.dumps(cut, indent=1) + "\n")
    return {"repo": entry["repo"], "revision": entry["revision"], "files_sha256": got, "layers": layers,
            "of_layers": cfg["num_hidden_layers"], "hash_layers": min(int(cfg.get("num_hash_layers") or 0), layers),
            "compress_ratios": cut.get("compress_ratios"), "experts": cfg.get("n_routed_experts"), "top_k": cfg.get("num_experts_per_tok")}


def status(worker: Any) -> dict[str, Any]:
    """On a rank: the function Marlin's experts call, whether they call it as the module's global (``guard``, what `MA.install`
    requires), the installed `Align`'s record, and the vLLM plugins this process sees."""
    from importlib.metadata import entry_points

    site = _site()
    fn = getattr(site, MA.NAME)
    a = MA.installed()
    return {"rank": _tp_rank(), "pid": os.getpid(), "calls": f"{fn.__module__}:{fn.__qualname__}", "canonical": a is not None,
            "guard": MA.loads_global(getattr(site, MA.CALLER, None), MA.NAME), "align": a.record() if a is not None else None,
            "plugins": sorted(f"{ep.name} = {ep.value}" for ep in entry_points(group="vllm.general_plugins"))}


class Checker:
    """Holds each `moe_align_block_size` call of one rank to `moe_align.reference`, attributed to the MoE layer whose forward
    made it, and counts the calls of vLLM's own aligners (`VLLM_ALIGNERS`)."""

    def __init__(self, model: Any) -> None:
        from verity_vllm.engine import hooks

        self.layers = moe_layers(model)
        self.modules = [(n, m) for n, m in model.named_modules() if n in self.layers]
        self.hooks, self.handles, self.cur, self.rows = hooks.Hooks(), [], None, 0
        self.by_layer: dict[str, dict[str, Any]] = {}
        self.vllm_calls: dict[str, int] = {}

    def install(self) -> None:
        for name, m in self.modules:
            self.handles.append(m.register_forward_pre_hook(self._enter(name)))
            self.handles.append(m.register_forward_hook(self._leave(name), always_call=True))
        self.hooks.wrap(_site(), MA.NAME, self.wrap)
        ops = _ops()
        for name in VLLM_ALIGNERS:
            if callable(getattr(ops, name, None)):
                self.vllm_calls[name] = 0
                self.hooks.wrap(ops, name, self._count(name))

    def _count(self, name: str) -> Any:
        def make(orig: Any) -> Any:
            def aligner(*a: Any, **k: Any) -> Any:
                self.vllm_calls[name] += 1
                return orig(*a, **k)
            return aligner
        return make

    def uninstall(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []
        self.hooks.uninstall()

    def _enter(self, name: str) -> Any:
        def pre(_m: Any, _a: tuple) -> None:
            self.cur = name
        return pre

    def _leave(self, name: str) -> Any:
        def post(_m: Any, _a: tuple, _o: Any) -> None:
            if self.cur == name:
                self.cur = None
        return post

    def wrap(self, orig: Any) -> Any:
        def moe_align_block_size(topk_ids: Any, block_size: int, num_experts: int, expert_map: Any = None, *a: Any, **k: Any) -> Any:
            out = orig(topk_ids, block_size, num_experts, expert_map, *a, **k)
            self.check(topk_ids, int(block_size), int(num_experts), expert_map, out)
            return out
        return moe_align_block_size

    def check(self, topk_ids: Any, block: int, num_experts: int, expert_map: Any, out: tuple) -> None:
        ids = topk_ids.reshape(-1).tolist()
        s, b, n = (t.reshape(-1).tolist() for t in out)
        ref = MA.reference(ids, num_experts, block, None if expert_map is None else expert_map.tolist(), length=len(s))
        equal = s == ref[0] and b == ref[1] and n[0] == ref[2]
        a = MA.installed()
        path = "none" if a is None else "one" if len(ids) <= a.small_max else "chunked"
        kind = "prefill" if int(topk_ids.shape[0]) > self.rows else "decode"
        row = self.by_layer.setdefault(self.cur or UNATTRIBUTED, {"calls": 0, "equal": 0, "paths": {}, "blocks": [], "experts": [],
                                                                    "expert_map": 0, "first_unequal": None})
        row["calls"] += 1
        row["equal"] += equal
        row["paths"][f"{kind}/{path}"] = row["paths"].get(f"{kind}/{path}", 0) + 1
        row["expert_map"] += expert_map is not None
        for key, v in (("blocks", block), ("experts", num_experts)):
            if v not in row[key]:
                row[key].append(v)
        if not equal and row["first_unequal"] is None:
            row["first_unequal"] = {"tokens": int(topk_ids.shape[0]), "pairs": len(ids), "block": block, "path": path,
                                    "sorted_ids": s != ref[0], "expert_ids": b != ref[1], "total": [n[0], ref[2]]}


def check_on(worker: Any, *, rows: int) -> dict[str, Any]:
    """On a rank: the `Checker` installed over Marlin's `moe_align_block_size` for a run of ``rows`` requests."""
    c = _STATE.get("checker")
    if c is None:
        c = _STATE["checker"] = Checker(worker.model_runner.get_model())
        c.install()
    c.rows = rows
    return {"rank": _tp_rank(), "layers": len(c.layers)}


def check_off(worker: Any) -> dict[str, Any]:
    """On a rank: the `Checker` uninstalled, and what it saw."""
    c = _STATE.pop("checker")
    c.uninstall()
    return {"rank": _tp_rank(), "layers": len(c.layers), "by_layer": c.by_layer, "vllm_calls": c.vllm_calls}


def use(worker: Any, *, canonical: bool) -> dict[str, Any]:
    """On a rank: Marlin's experts call the canonical op (the same `Align` again) or vLLM's."""
    a = MA.installed()
    if a is not None:
        _STATE["align"] = a
    if canonical and a is None:
        MA.install(_site(), align=_STATE["align"])
    elif not canonical and a is not None:
        MA.uninstall()
    return status(worker)


class Timer:
    """The host's time to issue each `moe_align_block_size` call, with its token count."""

    def __init__(self) -> None:
        from verity_vllm.engine import hooks

        self.hooks, self.calls = hooks.Hooks(), []

    def wrap(self, orig: Any) -> Any:
        def moe_align_block_size(topk_ids: Any, *a: Any, **k: Any) -> Any:
            t0 = time.perf_counter_ns()
            out = orig(topk_ids, *a, **k)
            self.calls.append((int(topk_ids.shape[0]), time.perf_counter_ns() - t0))
            return out
        return moe_align_block_size


def timer_on(worker: Any) -> None:
    t = _STATE["timer"] = Timer()
    t.hooks.wrap(_site(), MA.NAME, t.wrap)


class Repeater:
    """Calls the op Marlin's experts call ``repeats`` more times on each call's own inputs and records, per call, how many
    distinct outputs the ``repeats + 1`` gave over what Marlin reads (the first ``total`` slots, their blocks, the total), how
    many were the reference's, whether each held the reference's pairs per expert (`moe_align.canonical`), and whether some
    expert holds more than one pair (else the order within an expert is no choice)."""

    def __init__(self, repeats: int, rows: int) -> None:
        from verity_vllm.engine import hooks

        self.hooks, self.repeats, self.rows, self.calls = hooks.Hooks(), repeats, rows, []

    def wrap(self, orig: Any) -> Any:
        def moe_align_block_size(topk_ids: Any, block_size: int, num_experts: int, expert_map: Any = None, *a: Any, **k: Any) -> Any:
            outs = [orig(topk_ids, block_size, num_experts, expert_map, *a, **k) for _ in range(self.repeats + 1)]
            self.record(topk_ids, int(block_size), int(num_experts), expert_map, outs)
            return outs[0]
        return moe_align_block_size

    def record(self, topk_ids: Any, block: int, num_experts: int, expert_map: Any, outs: list[tuple]) -> None:
        ids = topk_ids.reshape(-1).tolist()
        ref = MA.reference(ids, num_experts, block, None if expert_map is None else expert_map.tolist())
        key = (tuple(ref[0][:ref[2]]), tuple(ref[1][:ref[2] // block]), ref[2])
        got = []
        for s, b, n in outs:
            t = int(n.reshape(-1)[0])
            got.append((tuple(s.reshape(-1)[:t].tolist()), tuple(b.reshape(-1)[:t // block].tolist()), t))
        held = [sum(1 for p in key[0][j * block:(j + 1) * block] if p < len(ids)) for j in range(len(key[1]))]
        count: dict[int, int] = {}
        for e, h in zip(key[1], held):
            count[e] = count.get(e, 0) + h
        self.calls.append({"kind": "prefill" if int(topk_ids.shape[0]) > self.rows else "decode", "pairs": len(ids),
                           "distinct": len(set(got)), "reference": sum(g == key for g in got),
                           "same_pairs": all(g[1] == key[1] and g[2] == key[2]
                                             and MA.canonical(g[0], g[1], g[2], block, len(ids)) == list(key[0]) for g in got),
                           "choice": any(v > 1 for v in count.values())})


def summarize_vary(calls: list[dict[str, Any]], repeats: int) -> dict[str, Any]:
    """By step kind: the calls, those where some expert holds more than one pair (``choice``), those whose ``repeats + 1``
    outputs were not all one (``varying``), the most distinct outputs of a call, the outputs that were the reference's, whether
    every output held the reference's pairs per expert, and the fewest pairs of a varying call."""
    out: dict[str, Any] = {}
    for kind in ("prefill", "decode"):
        cs = [c for c in calls if c["kind"] == kind]
        vary = [c for c in cs if c["distinct"] > 1]
        out[kind] = {"calls": len(cs), "choice": sum(c["choice"] for c in cs), "varying": len(vary),
                     "max_distinct": max((c["distinct"] for c in cs), default=0), "outputs": len(cs) * (repeats + 1),
                     "outputs_reference": sum(c["reference"] for c in cs), "same_pairs": all(c["same_pairs"] for c in cs),
                     "fewest_pairs_varying": min((c["pairs"] for c in vary), default=None)}
    return out


def vary_on(worker: Any, *, repeats: int, rows: int) -> dict[str, Any]:
    """On a rank: a `Repeater` over Marlin's `moe_align_block_size`, whichever op it calls now, for a run of ``rows`` requests."""
    r = _STATE["vary"] = Repeater(repeats, rows)
    r.hooks.wrap(_site(), MA.NAME, r.wrap)
    return {"rank": _tp_rank(), "canonical": MA.installed() is not None}


def vary_off(worker: Any) -> dict[str, Any]:
    """On a rank: the `Repeater` uninstalled, and its calls summarized (`summarize_vary`)."""
    r = _STATE.pop("vary")
    r.hooks.uninstall()
    return {"rank": _tp_rank(), **summarize_vary(r.calls, r.repeats)}


def timer_off(worker: Any, *, rows: int) -> dict[str, Any]:
    """On a rank: the timing wrap uninstalled; the median, 10th and 90th percentile host µs a call, decode and prefill apart."""
    t = _STATE.pop("timer")
    t.hooks.uninstall()
    out: dict[str, Any] = {"rank": _tp_rank()}
    for kind, ns in (("decode", [v for m, v in t.calls if m <= rows]), ("prefill", [v for m, v in t.calls if m > rows])):
        q = statistics.quantiles(ns, n=10) if len(ns) > 1 else [float(ns[0])] * 9 if ns else None
        out[kind] = {"calls": len(ns), "median_us": statistics.median(ns) / 1e3, "p10_us": q[0] / 1e3, "p90_us": q[-1] / 1e3} if ns else None
    return out


def digest(tokens: Any) -> str:
    return hashlib.sha256(json.dumps(tokens, sort_keys=True).encode()).hexdigest()[:16]


def summarize_check(seen: list[dict[str, Any]], before: list[dict[str, Any]], after: list[dict[str, Any]]) -> dict[str, Any]:
    """Per rank: the MoE layers all of whose calls equalled the reference, the unequal ones, and the calls by step kind and path,
    beside the `Align` counts' change and the calls of vLLM's aligners; ``ok`` when on every rank every MoE layer called it and
    every call equalled the reference, reached the rank's `Align` and none of vLLM's aligners, and both paths ran in prefill."""
    align0 = {int(s["rank"]): (s["align"] or {}).get("calls", {}) for s in before}
    ranks, ok = {}, True
    for r in sorted(seen, key=lambda r: int(r["rank"])):
        rank, layers = int(r["rank"]), r["by_layer"]
        paths: dict[str, int] = {}
        for row in layers.values():
            for k, v in row["paths"].items():
                paths[k] = paths.get(k, 0) + v
        unequal = sorted(n for n, row in layers.items() if row["equal"] != row["calls"])
        a1 = next(((s["align"] or {}).get("calls", {}) for s in after if int(s["rank"]) == rank), {})
        both = all(paths.get(f"prefill/{p}", 0) > 0 for p in ("one", "chunked"))
        calls = sum(row["calls"] for row in layers.values())
        reached = sum(v - align0.get(rank, {}).get(k, 0) for k, v in a1.items())
        bypassed = sum(r.get("vllm_calls", {}).values())
        ok &= (not unequal and both and UNATTRIBUTED not in layers and len(layers) == r["layers"] and reached == calls
               and not bypassed)
        ranks[rank] = {"layers": r["layers"], "layers_called": len(layers), "layers_equal": len(layers) - len(unequal),
                       "calls": calls, "calls_reaching_align": reached, "vllm_aligner_calls": r.get("vllm_calls", {}),
                       "unequal": unequal[:8],
                       "first_unequal": next((layers[n]["first_unequal"] for n in unequal), None), "paths": dict(sorted(paths.items())),
                       "both_paths_in_prefill": both,
                       "align_calls": {k: v - align0.get(rank, {}).get(k, 0) for k, v in a1.items()},
                       "blocks": sorted({b for row in layers.values() for b in row["blocks"]}),
                       "experts": sorted({e for row in layers.values() for e in row["experts"]}),
                       "expert_map_calls": sum(row["expert_map"] for row in layers.values())}
    return {"ok": ok and bool(ranks), "ranks": ranks}


def summarize_decode(rows: list[dict[str, Any]], host: dict[int, dict[str, list[dict[str, Any]]]], layers: int) -> dict[str, Any]:
    """Per batch: each op's median TPOT, the canonical/vLLM ratio over pairs, whether each op's runs gave one set of tokens and
    whether the two ops gave the same; the host µs a decode call (the median over ranks of each rank's median) and its change
    times ``layers`` calls a decode step."""
    out: dict[str, Any] = {}
    for b in sorted({r["batch"] for r in rows}):
        by = {op: {r["pair"]: r["phases"]["tpot_s"] for r in rows if r["batch"] == b and r["op"] == op and r["phases"]["tpot_s"]}
              for op in ("canonical", "vllm")}
        tpot = {op: statistics.median(v.values()) for op, v in by.items() if v}
        q = [by["canonical"][p] / by["vllm"][p] for p in sorted(set(by["canonical"]) & set(by["vllm"]))]
        tok = {op: sorted({r["tokens"] for r in rows if r["batch"] == b and r["op"] == op}) for op in by}
        hu = {op: [h["decode"]["median_us"] for h in host.get(b, {}).get(op, []) if h.get("decode")] for op in ("canonical", "vllm")}
        call = {op: statistics.median(v) if v else None for op, v in hu.items()}
        d_call = call["canonical"] - call["vllm"] if None not in call.values() else None
        out[str(b)] = {"tpot_ms": {op: 1e3 * v for op, v in tpot.items()},
                       "tpot_delta_ms": 1e3 * (tpot["canonical"] - tpot["vllm"]) if len(tpot) == 2 else None,
                       "tpot_ratio": {"median": statistics.median(q), "min": min(q), "max": max(q), "pairs": len(q)} if q else None,
                       "tokens_one_set": {op: len(v) == 1 for op, v in tok.items()},
                       "tokens_equal_across_ops": tok.get("canonical") == tok.get("vllm"),
                       "host_us_per_decode_call": call, "host_delta_us_per_call": d_call,
                       "host_delta_us_per_decode_step": None if d_call is None else d_call * layers}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="DeepSeek-V4-Pro eager: the canonical moe_align_block_size checked, timed and costed")
    ap.add_argument("--workload", required=True, help="the row (its first request is every batch's prompt)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--case", default=None)
    ap.add_argument("--checkpoints", default="manifests/checkpoints.json")
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--check-batches", default="1,8")
    ap.add_argument("--batches", default="1,16,64,256")
    ap.add_argument("--pairs", type=int, default=3)
    ap.add_argument("--rotate", type=int, default=8)
    ap.add_argument("--engine-arg", action="append", default=[])
    ap.add_argument("--rpc-timeout", type=float, default=3600.0)
    ap.add_argument("--dummy-layers", type=int, default=0, help="serve the first N layers with dummy weights (0: the checkpoint)")
    ap.add_argument("--vary-repeats", type=int, default=0, help="each op's every call repeated N more times on its inputs (0: none)")
    ap.add_argument("--vary-batches", default=None, help="the batches of the repeat phase (default --check-batches)")
    a = ap.parse_args()

    from verity_vllm.engine import engine_profile as prof
    prof.apply_env()
    from verity_vllm import protocols as PO
    from verity_vllm.engine import served as SV
    from verity_vllm.engine import vllm_adapter as va
    from verity_vllm.pipeline.commit import git_head, run_workload
    from verity_vllm.pipeline.tp import phases as PH
    from verity_vllm.pipeline.tp import target as TT
    from verity_vllm.pipeline.tp.capture import parse_engine_args
    from verity_vllm.pipeline.tp.rank_worker import EXTENSION
    from vllm import LLM

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    check_batches = [int(b) for b in a.check_batches.split(",")]
    batches = [int(b) for b in a.batches.split(",")]
    vary_batches = [int(b) for b in (a.vary_batches or a.check_batches).split(",")] if a.vary_repeats else []
    wl = va.load_workload(a.workload, None)
    target = PO.commit_guard(va.target_of_workload(wl), world=a.tp, installs=False)
    target_env, target_kw = TT.apply(target, wl["execution"])
    env_flags = {**prof.apply_env(), **target_env}
    case = va.case_of(wl, a.case)
    weights, dummy = "dummy" if a.dummy_layers else "checkpoint", None
    if a.dummy_layers:
        import tempfile

        entry = next(c for c in json.loads(Path(a.checkpoints).read_text())["checkpoints"] if c.get("repo") == wl["model"])
        cp = {**entry, "local_path": tempfile.mkdtemp(prefix="moe-align-dummy-")}
        dummy = dummy_model(entry, a.dummy_layers, Path(cp["local_path"]))
        print(f"[moe-align] dummy weights: {dummy}", flush=True)
    else:
        cp = prof.checkpoint_for(case, a.checkpoints)
    seqs = max(check_batches + batches + vary_batches)
    kw = prof.engine_kwargs(cp, replay=False, seed=prof.ENGINE_SEED, enforce_eager=True, tensor_parallel_size=a.tp)
    if dummy is not None:
        kw["load_format"] = "dummy"
    kw.update({"enable_prefix_caching": False, "max_num_seqs": seqs, **target_kw})
    kw.update({k: v for k, v in (wl.get("engine_args_required") or {}).items() if k != "tensor_parallel_size"})
    kw["max_num_batched_tokens"] = max(int(kw.get("max_num_batched_tokens") or 0), seqs * len(wl["requests"][0]["prompt_token_ids"]))
    kw.update(parse_engine_args(a.engine_arg), worker_extension_cls=EXTENSION)
    llm, build_s, topo = PH.engine(lambda: LLM(**kw), a.tp, a.rpc_timeout, model=cp["repo"], enforce_eager=True, case=case, max_num_seqs=seqs)
    TT.require_ranks(llm, target, a.rpc_timeout)
    served = SV.require(llm, wl, target, out, a.rpc_timeout)
    eng = va.Engine(llm=llm, runner=None, model=None, cp=cp, kwargs=kw, env_flags=env_flags, build_s=build_s, case=case, target=target)
    rpc = a.rpc_timeout
    to_cap = va.sampling_params

    def capped(req: dict[str, Any], sampling: dict[str, Any]) -> Any:
        p = to_cap(req, sampling)
        p.ignore_eos = True
        return p

    va.sampling_params = capped
    sampling = wl["sampling"]
    reqs = {b: requests_at(wl, b, a.rotate) for b in sorted(set(check_batches + batches + vary_batches))}
    start = llm.collective_rpc(status, timeout=rpc)
    engine_kwargs = {k: v for k, v in kw.items() if k not in ("model", "tokenizer", "worker_extension_cls")}
    versions = {"git_head": git_head(), "args": vars(a), "weights": weights, "dummy": dummy, "build_s": round(build_s, 1),
                "engine_kwargs": engine_kwargs, "target": TT.as_json(target), "served": served, "ranks": topo, "status": start}
    (out / "versions.json").write_text(json.dumps(versions, indent=1, default=str) + "\n")
    if len(start) != a.tp or not all(s["canonical"] and s["guard"] and s["calls"].startswith(f"{MA.__name__}:") for s in start):
        print(f"[moe-align] not every rank's Marlin experts call {MA.__name__}'s op as their module's global: {start}", flush=True)
        raise SystemExit(3)
    print("[moe-align] ranks: " + "; ".join(f"rank {s['rank']} pid {s['pid']}: calls {s['calls']}, guard {s['guard']}" for s in start),
          flush=True)

    def warm(b: int, seed: int) -> None:
        PH.warmups(1, lambda i: run_workload(eng, va.throwaway_requests(eng, reqs[b], seed=seed + i), sampling), tag="moe-align")

    for b in check_batches:
        warm(b, 100 + b)
    before = llm.collective_rpc(status, timeout=rpc)
    for b in check_batches:
        llm.collective_rpc(check_on, timeout=rpc, kwargs={"rows": b})
        r = run_workload(eng, reqs[b], sampling)
        print(f"[moe-align] check B{b}: {r['wall_s']:.2f}s, {len(r['step_wall_s'])} steps", flush=True)
    seen = llm.collective_rpc(check_off, timeout=rpc)
    after = llm.collective_rpc(status, timeout=rpc)
    check = summarize_check(seen, before, after)
    (out / "check.json").write_text(json.dumps({**check, "seen": seen, "before": before, "after": after}, indent=1) + "\n")
    print(f"[moe-align] check ok {check['ok']} ({weights} weights): " + "; ".join(
        f"rank {r}: {v['layers_equal']}/{v['layers']} layers equal, {v['calls']} calls, {v['calls_reaching_align']} reaching Align "
        f"{v['align_calls']}, vLLM aligners {v['vllm_aligner_calls']}, {v['paths']}" for r, v in check["ranks"].items()), flush=True)

    vary: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for b in vary_batches:
        for op in ("vllm", "canonical"):
            llm.collective_rpc(use, timeout=rpc, kwargs={"canonical": op == "canonical"})
            llm.collective_rpc(vary_on, timeout=rpc, kwargs={"repeats": a.vary_repeats, "rows": b})
            run_workload(eng, reqs[b], sampling)
            got = vary.setdefault(str(b), {})[op] = sorted(llm.collective_rpc(vary_off, timeout=rpc), key=lambda r: int(r["rank"]))
            print(f"[moe-align] repeats B{b} {op} x{a.vary_repeats + 1}: " + "; ".join(
                f"{kind} {sum(r[kind]['varying'] for r in got)} of {sum(r[kind]['calls'] for r in got)} calls vary "
                f"({sum(r[kind]['choice'] for r in got)} with a choice, at most {max(r[kind]['max_distinct'] for r in got)} distinct, "
                f"{sum(r[kind]['outputs_reference'] for r in got)} of {sum(r[kind]['outputs'] for r in got)} outputs the reference's, "
                f"same pairs {all(r[kind]['same_pairs'] for r in got)})" for kind in ("prefill", "decode")), flush=True)
    if vary:
        (out / "vary.json").write_text(json.dumps({"repeats": a.vary_repeats, "batches": vary}, indent=1) + "\n")

    rows: list[dict[str, Any]] = []
    host: dict[int, dict[str, list[dict[str, Any]]]] = {}
    with (out / "decode.jsonl").open("w") as f:
        for b in batches:
            for op in ("canonical", "vllm"):
                llm.collective_rpc(use, timeout=rpc, kwargs={"canonical": op == "canonical"})
                warm(b, 200 + 2 * b + (op == "vllm"))
            for pair in range(a.pairs):
                for k, op in enumerate(("canonical", "vllm") if pair % 2 == 0 else ("vllm", "canonical")):
                    llm.collective_rpc(use, timeout=rpc, kwargs={"canonical": op == "canonical"})
                    r = run_workload(eng, reqs[b], sampling)
                    row = {"batch": b, "pair": pair, "order_in_pair": k, "op": op, "wall_s": r["wall_s"],
                           "phases": phases(r["step_wall_s"], r["step_tokens_out"], b), "tokens": digest(r["tokens"]),
                           "step_wall_s": r["step_wall_s"]}
                    rows.append(row)
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    print(f"[moe-align] B{b} pair {pair} {op}: TPOT {1e3 * (row['phases']['tpot_s'] or 0):.3f} ms over "
                          f"{row['phases']['decode_steps']} steps, tokens {row['tokens']}", flush=True)
            for op in ("canonical", "vllm"):
                llm.collective_rpc(use, timeout=rpc, kwargs={"canonical": op == "canonical"})
                llm.collective_rpc(timer_on, timeout=rpc)
                run_workload(eng, reqs[b], sampling)
                host.setdefault(b, {})[op] = llm.collective_rpc(timer_off, timeout=rpc, kwargs={"rows": b})
    final = llm.collective_rpc(use, timeout=rpc, kwargs={"canonical": True})
    (out / "host.json").write_text(json.dumps({str(b): v for b, v in host.items()}, indent=1) + "\n")
    layers = max((v["layers"] for v in check["ranks"].values()), default=0)
    decode = summarize_decode(rows, host, layers)
    summary = {"weights": weights, "dummy": dummy, "check": check, "decode": decode, "vary": vary, "layers": layers, "world": len(topo),
               "guard": all(s["guard"] for s in final), "restored": all(s["canonical"] for s in final)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    for b, d in decode.items():
        print(f"[moe-align] B{b} ({weights} weights): TPOT {d['tpot_ms']} ms, ratio {d['tpot_ratio']}, host {d['host_us_per_decode_call']} us a call, "
              f"{d['host_delta_us_per_decode_step']} us a step; tokens one set {d['tokens_one_set']}, "
              f"equal across ops {d['tokens_equal_across_ops']}", flush=True)
    if not check["ok"]:
        raise SystemExit(3)


if __name__ == "__main__":
    from moe_align.served import main as _main

    _main()
