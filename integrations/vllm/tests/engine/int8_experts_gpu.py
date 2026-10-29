"""On one GPU: a MoE (OLMoE by default) pinned int8 (`int8-checkpoint`, recipe v1) served under `TargetProfile.int8_exact`
"linears", its routed experts on `Int8ExpertsMethod`, checked bit for bit on the CPU.

One process writes `<out>/int8_experts_gpu.json` and `<out>/pin_entry.json` (the int8 pin's entry):

  pin       the registered int8 pin of `--role` (re-made when its snapshot is not on this host), else a new one
            (`int8_exact_gpu.pin`).
  engine    `build_engine(case=ROLE, target=from_device() + int8_exact "linears")`, which refuses a target mismatch.
  applied   every linear but the routers runs `Int8PlainLinearMethod` and every MoE layer `Int8ExpertsMethod`, each serving
            calls during generation, none with a slot routed to no expert (vLLM's profile run routes every slot so); the
            routers are what `kernel_readback.routers` names, and stay bf16.
  generate  greedy on PROMPTS, batched and one by one.
  experts   the first and last MoE layers' first `--keep-calls` calls per phase keep x, the router's top-k ids and weights, the
            served output and every expert GEMM's call (its bf16 rows, s8 rows, scales, y and the tokens they are).  On the CPU:
              gemm      each expert GEMM's served y against `int8_exact.plain_linear` on the CPU on the served int8 weight, and
                        against steps.py's `int8_linear` (`ppl_oracle`) from the BF16 base pin's own expert weight (gate and
                        up concatenated by rows); its s8 rows and scales against `quantize_rows` of its rows
              calls     every (expert, projection) called once per layer call in the static order, its rows the tokens
                        `int8_experts.routes` gives, an idle expert's call reaching ``observe`` with zero rows
              act       each down's rows against `silu_and_mul` of its gate_up's served y, on the CPU
              combine   the layer's output against the f32 top-k combine of the served expert outputs, on the CPU
              layer     the whole layer (`int8_experts.forward`) on the CPU, and again on the GPU, from the served x, ids, weights
            and the served weight against the oracle's `quant_rows(w.float(), 127)` of the base weight.

    cd integrations/vllm; PY=/workspace/jobs/venv312/bin/python3     # PYTHONPATH=.:../..:../../tools/research/src
    $PY tests/engine/int8_experts_gpu.py --out DIR --at ISO_TIME --by LANE
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch

from tests.engine import int8_exact_gpu as G
from tests.engine import ppl_oracle as O
from verity_vllm.config import checkpoint_entries
from verity_vllm.engine import fp8_checkpoint as F
from verity_vllm.engine import int8_checkpoint as I
from verity_vllm.engine import int8_exact as E
from verity_vllm.engine import int8_experts as X

SCHEMA = "verity-vllm/int8-experts-gpu/v1"
RECORD = "int8_experts_gpu.json"
PROJ = {X.GATE_UP: ("gate_proj", "up_proj"), X.DOWN: ("down_proj",)}


def moe_methods(model: torch.nn.Module) -> list[tuple[str, Any]]:
    return [(n, m.quant_method) for n, m in model.named_modules()
            if type(getattr(m, "quant_method", None)).__name__ == "Int8ExpertsMethod"]


class ExpertTaps:
    """Keeps a MoE method's first `keep` calls per phase: its x, top-k ids and weights, output and every site's call."""

    def __init__(self, methods: list[tuple[str, Any]], keep: int):
        self.phase: str | None = None
        self.keep = keep
        self.got: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self.sites: list[dict[str, Any]] | None = None
        for name, qm in methods:
            qm.apply = self._apply(name, qm.apply)
            for s in qm.sites:
                s.quant_method.apply = self._site(s, s.quant_method.apply)
                s.quant_method.observe = self._observe(s.quant_method.observe)

    def _apply(self, name: str, inner):
        def apply(layer, x, topk_weights, topk_ids, **kw):
            calls = self.got.setdefault((self.phase, name), []) if self.phase else None
            if calls is None or len(calls) >= self.keep:
                return inner(layer=layer, x=x, topk_weights=topk_weights, topk_ids=topk_ids, **kw)
            self.sites = []
            out = inner(layer=layer, x=x, topk_weights=topk_weights, topk_ids=topk_ids, **kw)
            calls.append({"x": x.detach().cpu(), "w": topk_weights.detach().cpu(), "ids": topk_ids.detach().cpu(),
                          "out": out.detach().cpu(), "sites": self.sites})
            self.sites = None
            return out
        return apply

    def _site(self, site, inner):
        def apply(layer, x, bias=None):
            if self.sites is None:
                return inner(layer, x, bias)
            self.sites.append({"site": site, "x": x.detach().cpu(), "tokens": site.tokens.detach().cpu()})
            y = inner(layer, x, bias)
            self.sites[-1]["y"] = y.detach().cpu()
            return y
        return apply

    def _observe(self, inner):
        def observe(layer, xq, s_a, y, z=None):
            if self.sites is not None:
                self.sites[-1].update(xq=xq.detach().cpu(), s_a=s_a.detach().cpu(), observed_rows=int(xq.shape[0]))
            return inner(layer, xq, s_a, y, z) if z is not None else inner(layer, xq, s_a, y)
        return observe


def _words(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def _ne(a: torch.Tensor, b: torch.Tensor) -> int:
    if a.shape != b.shape:
        return max(a.numel(), b.numel(), 1)
    return int((_words(a) != _words(b)).sum()) if a.dtype == torch.bfloat16 else int((a != b).sum())


class ExpertBase:
    """The BF16 base pin's expert weights, by (MoE layer index, expert, projection)."""

    def __init__(self, base: G.BaseWeights):
        self.base = base

    def weight(self, prefix: str, expert: int, projection: str) -> torch.Tensor:
        layer = re.search(r"layers\.(\d+)\.", prefix).group(1)
        names = []
        for p in PROJ[projection]:
            hit = [k for k in self.base.where if re.search(rf"layers\.{layer}\..*experts\.{expert}\.{p}\.weight$", k)]
            if len(hit) != 1:
                raise KeyError(f"layer {layer} expert {expert} {p}: {len(hit)} base tensors")
            names += hit
        return torch.cat([self.base.tensor(n) for n in names])


def layer_checks(name: str, qm: Any, cpu_sites: list[X.ExpertSite], call: dict[str, Any], base: ExpertBase,
                 weights_seen: dict[str, bool]) -> dict[str, Any]:
    x, w, ids, out, recs = call["x"], call["w"], call["ids"], call["out"], call["sites"]
    m, k = ids.shape
    rows = X.routes(ids, len(qm.sites) // 2)
    order_ok = [r["site"] for r in recs] == list(qm.sites)
    tokens_bad = sum(not torch.equal(r["tokens"], rows[r["site"].expert] // k) for r in recs)
    empty = sum(r["x"].shape[0] == 0 for r in recs)
    observed_bad = sum(r.get("observed_rows") != r["x"].shape[0] for r in recs)
    gemm = {"cpu": 0, "oracle": 0, "xq": 0, "s_a": 0, "words": 0, "rows": 0}
    act = 0
    per = torch.zeros((m * k, out.shape[1]), dtype=torch.bfloat16)
    for r in recs:
        s = r["site"]
        if r["x"].shape[0] == 0:
            continue
        c = cpu_sites[2 * s.expert + (s.projection == X.DOWN)]
        wq, sw = c.weight, c.weight_scale
        xq, s_a = E.quantize_rows(r["x"])
        gemm["xq"] += _ne(xq, r["xq"])
        gemm["s_a"] += _ne(s_a.reshape(-1), r["s_a"].reshape(-1))
        gemm["cpu"] += _ne(E.plain_linear(r["x"], wq, sw), r["y"])
        wb = base.weight(s.prefix, s.expert, s.projection)
        gemm["oracle"] += _ne(O.int8_linear(r["x"], wb), r["y"])
        if s.name not in weights_seen:
            q, sb = O.int8_weights(wb)
            weights_seen[s.name] = bool(torch.equal(wq, q) and torch.equal(sw.reshape(-1), sb))
        gemm["words"] += r["y"].numel()
        gemm["rows"] += r["x"].shape[0]
        if s.projection == X.GATE_UP:
            h = r["y"]
        else:
            act += _ne(X.silu_and_mul(h), r["x"])
            per[rows[s.expert]] = r["y"]
    acc = torch.zeros((m, out.shape[1]), dtype=torch.float32)
    for j in range(k):
        acc = acc + w.float()[:, j:j + 1] * per.view(m, k, -1)[:, j].float()
    combine = _ne(acc.to(torch.bfloat16), out)
    layer_cpu = _ne(X.forward(_fresh(cpu_sites), x, w, ids), out)
    dev = qm.sites[0].weight.device
    layer_gpu = _ne(X.forward(_fresh(qm.sites), x.to(dev), w.to(dev), ids.to(dev)).cpu(), out)
    return {"layer": name, "M": m, "k": k, "calls": len(recs), "order_ok": order_ok, "tokens_bad": tokens_bad, "empty": empty,
            "observed_bad": observed_bad, "gemm": gemm, "act_mismatch": act, "combine_mismatch": combine,
            "layer_cpu_mismatch": layer_cpu, "layer_gpu_mismatch": layer_gpu}


def _fresh(sites: list[X.ExpertSite], device: str | None = None) -> list[X.ExpertSite]:
    """The same weights (on `device`, if given) under unwrapped calls (a rerun the taps don't see)."""
    return [X.ExpertSite(s.prefix, s.expert, s.projection, s.weight.to(device) if device else s.weight,
                         s.weight_scale.to(device) if device else s.weight_scale) for s in sites]


def run(a: argparse.Namespace) -> dict[str, Any]:
    from vllm.model_executor.layers.linear import LinearBase

    from verity_vllm.engine.build import build_engine, target_mismatches
    from verity_vllm.program.frontend.kernel_readback import int8_exact_of, routers
    from verity_vllm.program.frontend.target_profile import TargetProfile

    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    target = TargetProfile.from_device().with_overrides(int8_exact="linears")
    doc: dict[str, Any] = {"schema": SCHEMA, "argv": sys.argv, "torch": torch.__version__, "device": torch.cuda.get_device_name(0),
                           "target": target.to_json()}
    with tempfile.TemporaryDirectory() as scratch:
        entry, how = G.pin(a)
        doc["pin"] = {"role": entry["role"], "repo": entry["repo"], "revision": entry["revision"], "recipe": entry["recipe"],
                      "local_path": entry["local_path"], "source": how}
        (outdir / "pin_entry.json").write_text(json.dumps(entry, indent=1) + "\n")
        print(f"[moe] pin {json.dumps(doc['pin'])}", flush=True)
        base = ExpertBase(G.BaseWeights(F.base_entry(entry, checkpoint_entries(a.checkpoints), I.INT8)["local_path"]))
        eng = build_engine(G.scratch_checkpoints(a.checkpoints, entry, Path(scratch)), case=a.role, target=target,
                           max_num_seqs=len(G.PROMPTS))
    doc["engine"] = {"build_s": round(eng.build_s, 1), "target_mismatches": target_mismatches(eng), "int8_exact": int8_exact_of(eng.model),
                     "quantization": eng.kwargs.get("quantization"), "enforce_eager": eng.kwargs.get("enforce_eager")}
    print(f"[moe] engine {json.dumps(doc['engine'], default=str)}", flush=True)

    model = eng.model
    skip = routers(model)
    router_names = [n for n, m in model.named_modules() if id(m) in skip]
    linears = [n for n, m in model.named_modules() if isinstance(m, LinearBase) and id(m) not in skip]
    methods = moe_methods(model)
    before = {n: qm.calls for n, qm in methods}
    unrouted = {n: qm.unrouted for n, qm in methods}
    lin_before = E.applied_record(model)
    taps = ExpertTaps(methods[:1] + methods[-1:], a.keep_calls)
    taps.phase = "batched"
    batched = G.generate(eng.llm, G.PROMPTS, a.max_tokens)
    solo = []
    for i, p in enumerate(G.PROMPTS):
        taps.phase = f"solo{i}"
        solo += G.generate(eng.llm, [p], a.max_tokens)
    taps.phase = None
    after = {n: r for n, r in E.applied_record(model).items() if n not in router_names}
    lin = {n: r for n, r in after.items() if n in set(linears)}
    served = {n: (r["calls"] or 0) - (lin_before.get(n, {}).get("calls") or 0) for n, r in after.items()}
    doc["applied"] = {
        "linears": len(linears), "routers": router_names, "router_methods": sorted({type(m.quant_method).__name__ for n, m in
                                                                                     model.named_modules() if n in router_names}),
        "moe_layers": len(methods),
        "methods": sorted({r["method"] for r in after.values()}), "arms": sorted({str(r["arm"]) for r in after.values()}),
        "dense_methods": sorted({r["method"] for r in lin.values()}),
        "moe_min_calls": min((qm.calls - before[n] for n, qm in methods), default=0),
        "unrouted_slots_before_generation": sum(unrouted.values()),
        "unrouted_slots_during_generation": sum(qm.unrouted - unrouted[n] for n, qm in methods),
        "min_calls_during_generation": min(served.values(), default=0),
        "sites_per_layer": sorted({len(qm.sites) for _, qm in methods})}
    doc["applied"]["ok"] = bool(methods and router_names and len(router_names) == len(methods)
                                and set(lin) == set(linears) and doc["applied"]["router_methods"] == ["UnquantizedLinearMethod"]
                                and doc["applied"]["dense_methods"] == ["Int8PlainLinearMethod"]
                                and doc["applied"]["arms"] == [E.ARM] and doc["applied"]["moe_min_calls"] > 0
                                and doc["applied"]["unrouted_slots_during_generation"] == 0
                                and doc["applied"]["min_calls_during_generation"] > 0)
    print(f"[moe] applied {json.dumps(doc['applied'])}", flush=True)
    doc["generations"] = {"batched": batched, "solo": solo,
                          "batch_dependent": [b["prompt"] for b, s in zip(batched, solo) if b["token_ids"] != s["token_ids"]]}
    for b, s in zip(batched, solo):
        print(f"[moe] gen {b['prompt']!r}\n  batched: {b['text']!r}\n  solo:    {s['text']!r}", flush=True)

    by_name = dict(methods)
    cpu = {n: _fresh(by_name[n].sites, "cpu") for n in {n for _, n in taps.got}}
    weights_seen: dict[str, bool] = {}
    doc["experts"] = []
    for (phase, name), calls in sorted(taps.got.items()):
        for i, call in enumerate(calls):
            rec = {"phase": phase, "call": i, **layer_checks(name, by_name[name], cpu[name], call, base, weights_seen)}
            doc["experts"].append(rec)
            print(f"[moe] {phase} {name} call {i}: M={rec['M']} calls={rec['calls']} empty={rec['empty']} order={rec['order_ok']} "
                  f"tokens!={rec['tokens_bad']} observed!={rec['observed_bad']} gemm={json.dumps(rec['gemm'])} act!={rec['act_mismatch']} "
                  f"combine!={rec['combine_mismatch']} layer_cpu!={rec['layer_cpu_mismatch']} layer_gpu!={rec['layer_gpu_mismatch']}",
                  flush=True)
    ex = doc["experts"]
    g = {k: sum(r["gemm"][k] for r in ex) for k in ("cpu", "oracle", "xq", "s_a", "words", "rows")}
    doc["experts_totals"] = t = {
        "layer_calls": len(ex), "gemm_calls": sum(r["calls"] for r in ex), "empty_calls": sum(r["empty"] for r in ex),
        "decode_calls": sum(r["M"] <= len(G.PROMPTS) for r in ex), "gemm": g,
        "weights_checked": len(weights_seen), "weights_match_oracle": all(weights_seen.values()),
        **{k: sum(r[k] for r in ex) for k in ("tokens_bad", "observed_bad", "act_mismatch", "combine_mismatch",
                                               "layer_cpu_mismatch", "layer_gpu_mismatch")},
        "order_ok": all(r["order_ok"] for r in ex)}
    exact = (g["words"] and not (g["cpu"] or g["oracle"] or g["xq"] or g["s_a"]) and t["weights_match_oracle"] and t["order_ok"]
             and not (t["tokens_bad"] or t["observed_bad"] or t["combine_mismatch"] or t["layer_gpu_mismatch"]) and t["empty_calls"])
    doc["ok"] = bool(doc["applied"]["ok"] and not doc["engine"]["target_mismatches"] and doc["engine"]["int8_exact"] == "linears"
                     and exact and all(x["token_ids"] for x in batched + solo))
    (outdir / RECORD).write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print(f"INT8-EXPERTS-GPU {'OK' if doc['ok'] else 'FAIL'} totals={json.dumps(t)} "
          f"batch_dependent_tokens={len(doc['generations']['batch_dependent'])}", flush=True)
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="int8 routed experts (Int8ExpertsMethod) served by vLLM on this GPU, checked bit for bit")
    ap.add_argument("--out", required=True, help=f"directory for {RECORD} and pin_entry.json")
    ap.add_argument("--checkpoints", default="manifests/checkpoints.json")
    ap.add_argument("--base-role", default="OLMOE")
    ap.add_argument("--role", default="OLMOE_INT8")
    ap.add_argument("--repo", default="verity-int8/OLMoE-1B-7B-0924-INT8")
    ap.add_argument("--hf-home", default=os.environ.get("HF_HOME"))
    ap.add_argument("--at", required=True, help="resolved_at of the pin entry (UTC, ISO 8601)")
    ap.add_argument("--by", required=True, help="resolved_by of the pin entry")
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--keep-calls", type=int, default=3, help="calls kept per tapped MoE layer and phase")
    return 0 if run(ap.parse_args(argv))["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
