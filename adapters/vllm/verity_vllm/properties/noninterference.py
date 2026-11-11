"""Non-interference (DESIGN §3 "by construction", §8): the observer changes nothing the model computes.

Three FRESH processes on the same engine build (κ, `vllm_adapter.build_engine`) and the same workload, launched
sequentially by this module's orchestrator (`subprocess`), then a torch-free comparison:

  (a) `--variant observer`   the full observer (`observe.capture.Capture`: dispatch mode + re-entry + Triton hook + module
                             naming hooks) with snapshots of EVERY step, **hash-only** (`HashOnlyCapture`: the snapshot
                             machinery runs unchanged — pinned D2H copies stream-ordered after every recorded call /
                             launch, one `torch.cuda.synchronize()` per snapshot step — but `_flush_snapshots` records the
                             sha256 of every value and writes no `.bin`).  Output: `log.jsonl.gz`, `snapshots.json`
                             (index with sha256), `tokens.json`.
  (b) `--variant hooks`      NO observer, NO dispatch mode, NO Triton hook.  The only touch is one `register_forward_hook`
                             on each of the 30 decoder layers, on `model.norm`, on `logits_processor` (and on `lm_head`,
                             which vLLM never calls through `forward`: the LogitsProcessor applies the lm_head weight
                             directly, so that hook records nothing — reported).  Each hook copies the module's output
                             tensors D2H asynchronously into fresh pinned buffers (stream-ordered right after the module,
                             i.e. before the next layer's in-place `fused_add_rms_norm` overwrites them); after every
                             `engine.step()` the driver synchronises once and hashes.  Output: `hooks.json`, `tokens.json`.
  (c) `bench/m1_capture.py --no-observer`  fully bare (nothing installed, `vllm_adapter.run_requests`): `tokens.json`.

Tensors compared, per step s and layer L (values are bf16 row-major `[M, H]`, M = token rows of the step):
  hidden[s, L]   = `LlamaDecoderLayer.forward` return[0] = the MLP `down_proj` output.
                   (a): snapshot of the `matmul_kernel_persistent` launch with module hint `model.layers.L.mlp.down_proj`
                   (phase forward), argument `c_ptr`, taken after the launch;  (b): hook output[0] of `model.layers.L`.
                   MoE layers (OLMoE): (a) is the `vllm.moe_forward*` custom op's `outs` snapshot (module hint
                   `model.layers.L.mlp.experts`) — the layer has no `down_proj` launch.
  residual[s, L] = `LlamaDecoderLayer.forward` return[1] = `post_attention_layernorm`'s in-place residual (x + residual).
                   (a): snapshot of `_C.fused_add_rms_norm` with module hint `model.layers.L.post_attention_layernorm`,
                   argument `args[1]` after the call;  (b): hook output[1] of `model.layers.L`.
  final[s]       = `model.norm` output[0] (the normalised last hidden state).
                   (a): `_C.fused_add_rms_norm` with module hint `model.norm`, `args[0]` after the call;  (b): hook output[0].
  logits[s]      = `logits_processor` output = the lm_head GEMM's `c_ptr` (bf16 `[R, V]`, phase sample).
The observer's snapshot bytes are the *spanned storage bytes* of the view; for these contiguous tensors that is exactly the
tensor's bytes, which is what (b) hashes (`tensor.contiguous()` bytes; contiguity is recorded and asserted).

Comparison (`--compare DIR`, torch-free; also run automatically by the orchestrator): tokens (a) == (b) == (c) == the accepted
record (`bench.m1_capture.ACCEPTED_TOKENS`, and `data/logs/m1_ctl_tokens.json` when present); per step the sampled
ids of (a)'s `SampleEnd` == (b)'s engine outputs; hashes (a) == (b) for every (step, layer, hidden|residual) + final + logits.
Writes DIR/noninterference.json and DIR/noninterference.md; exit status 1 on any mismatch.

The record a run cites (`record`, `verity-vllm/noninterference/v2`, `<run>/properties/noninterference.json`) needs no engine of its
own: the run's observed Match arm against its bare arm, per request, at tensor-parallel world 1 or 2 (a larger world is refused), with
this comparison's boundary hashes folded in when given (world 1).

    HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 PYTHONPATH=. VLLM_BATCH_INVARIANT=1 VLLM_USE_V2_MODEL_RUNNER=1 \
    VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_FLASHINFER_SAMPLER=0 /workspace/venv-cu129/bin/python -m \
    verity_vllm.properties.noninterference --workload workloads/workload_32x16_1req.json --out /vol/cp/l5/nonint
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import subprocess
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verity_vllm import settings
from verity_vllm.config import option, workload_arg
from verity_vllm.properties.noninterference_families import FINAL_NORM_MODULES, parallel_residual_boundaries, unfused_norm_boundaries
from verity_vllm.properties.record import seal

# the decoder-layer module and the final norm under their family's vLLM names: Llama-style
# `model.layers.N` / `model.norm` (SmolLM2, Qwen, Gemma-2, Phi-3, the MoE families) and GPT-NeoX `gpt_neox.layers.N` /
# `gpt_neox.final_layer_norm` (Pythia).  The (b) hooks and the hooks-side parser read these; the Llama-shaped (a)-side rules
# (`_DOWN_RE`, `_POST_RE`) stay Llama-only and the GPT-NeoX (a)-side boundaries are `parallel_residual_boundaries`.
_LAYER_RE = re.compile(r"^(?:model|gpt_neox)\.layers\.(\d+)$")
_DOWN_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.down_proj$")
_POST_RE = re.compile(r"^model\.layers\.(\d+)\.post_attention_layernorm$")
#: MoE layers (OLMoE): the layer's return[0] is the `vllm.moe_forward*` custom op's output (module hint
#: `model.layers.L.mlp.experts`, snapshot path `outs`), not a `down_proj` launch — the experts' down GEMMs are one fused
#: `fused_moe_kernel` launch and `moe_sum` writes the combined rows.
_MOE_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts$")


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _read_json(path: Path) -> dict:
    """`path` or its gzipped twin (`path.gz`, how the committed artifacts are kept)."""
    if path.exists():
        return json.loads(path.read_text())
    with gzip.open(str(path) + ".gz", "rt") as f:
        return json.load(f)


# ---------------------------------------------------------------------------------------------------------
# (a) observer + hash-only snapshots of every step
# ---------------------------------------------------------------------------------------------------------


def run_observer_variant(workload: str, out_dir: str, checkpoints: str, *, max_tokens: int | None, max_num_seqs: int,
                         warmup: bool, case: str | None = None, engine_args: dict | None = None) -> None:
    from verity_vllm.engine import engine_profile as prof

    prof.apply_env()
    import torch  # noqa: F401

    from verity_vllm.engine import vllm_adapter as va
    from verity_vllm.observe.capture import Capture, ObserverConfig, make_header

    class HashOnlyCapture(Capture):
        """`Capture` whose snapshot flush hashes the pinned copies and writes no value files."""

        def _flush_snapshots(self, st) -> None:  # type: ignore[override]
            t0 = time.perf_counter()
            for item in self._snap_pending:
                data = item["host"].numpy().tobytes()
                entry = item["entry"]
                entry["sha256"] = _sha(data)
                entry["file"] = None
                entry["step"] = st.step
                self._snap_files.setdefault(entry["sha256"], "")   # sha -> file (D80 index); hash-only: no file
                self._snap_bytes += len(data)
                self._snap_index.append(entry)
            self._snap_pending = []
            self._arena.reset()
            self.timing["snapshot_s"] += time.perf_counter() - t0

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    wl = va.load_workload(workload, max_tokens)
    eng = va.build_engine(checkpoints, max_num_seqs=max_num_seqs, case=va.case_of(wl, case), engine_args=engine_args, target=va.target_of_workload(wl))
    print(f"[nonint/a] engine built in {eng.build_s:.1f}s (case {eng.case}, {eng.cp['repo']})", flush=True)
    (out / "versions.json").write_text(json.dumps(va.versions_doc(eng), indent=1) + "\n")
    if warmup:
        _, tw = va.run_requests(eng, va.throwaway_requests(eng, wl["requests"], seed=va.WARMUP_SEED), wl["sampling"])
        print(f"[nonint/a] warm-up: {tw['wall_s']:.2f}s, {tw['engine_steps']} steps", flush=True)
    try:
        revision = json.loads(Path("EXPORT.json").read_text()).get("commit", "unknown")
    except Exception:  # noqa: BLE001
        revision = "unknown"
    snap = frozenset(range(256))          # every step of the run (membership test only)
    header = make_header(eng, wl, adapter_revision=revision, extra_notes={"argv": sys.argv, "snapshot_steps": "all (hash-only)",
                                                                           "variant": "noninterference (a) observer"})
    cfg = ObserverConfig(mode="record", reenter=True, triton=True, module_hooks=True, write_log=True, snapshot_steps=snap,
                         snapshot_dir=out)
    cap = HashOnlyCapture(eng, header=header, log_path=out / "log.jsonl.gz", config=cfg)
    cap.install()
    t0 = time.perf_counter()
    tokens, timing = va.run_requests(eng, wl["requests"], wl["sampling"])
    run_s = time.perf_counter() - t0
    stats = cap.close()
    print(f"[nonint/a] observed run {run_s:.2f}s: {stats['events_written']} events, {stats['steps']} steps, snapshot entries "
          f"{len(cap._snap_index)}, hashed bytes {stats['snapshot_bytes']}, errors {len(stats['errors'])}", flush=True)
    (out / "stats.json").write_text(json.dumps(stats, indent=1, default=str) + "\n")
    _write_tokens(out, wl, tokens, timing, wl["accepted"], {str(s): ids for s, ids in cap.sample_ids.items()}, "observer")


# ---------------------------------------------------------------------------------------------------------
# (b) bare engine + module forward hooks only
# ---------------------------------------------------------------------------------------------------------


def run_hooks_variant(workload: str, out_dir: str, checkpoints: str, *, max_tokens: int | None, max_num_seqs: int,
                      warmup: bool, case: str | None = None, engine_args: dict | None = None) -> None:
    from verity_vllm.engine import engine_profile as prof

    prof.apply_env()
    import torch

    from verity_vllm.engine import arrivals as AR, vllm_adapter as va

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    wl = va.load_workload(workload, max_tokens)
    eng = va.build_engine(checkpoints, max_num_seqs=max_num_seqs, case=va.case_of(wl, case), engine_args=engine_args, target=va.target_of_workload(wl))
    print(f"[nonint/b] engine built in {eng.build_s:.1f}s (case {eng.case}, {eng.cp['repo']})", flush=True)
    (out / "versions.json").write_text(json.dumps(va.versions_doc(eng), indent=1) + "\n")
    if warmup:
        _, tw = va.run_requests(eng, va.throwaway_requests(eng, wl["requests"], seed=va.WARMUP_SEED), wl["sampling"])
        print(f"[nonint/b] warm-up: {tw['wall_s']:.2f}s, {tw['engine_steps']} steps", flush=True)
        torch.cuda.synchronize()

    pending: list[dict] = []            # this engine step's pinned copies
    fired: dict[str, int] = {}

    def make_hook(name: str):
        def hook(mod, args, output):
            outs = output if isinstance(output, (tuple, list)) else (output,)
            fired[name] = fired.get(name, 0) + 1
            for i, t in enumerate(outs):
                if not isinstance(t, torch.Tensor) or t.numel() == 0:
                    continue
                src = t if t.is_contiguous() else t.contiguous()
                host = torch.empty(src.shape, dtype=src.dtype, pin_memory=True)
                host.copy_(src, non_blocking=True)
                pending.append({"module": name, "index": i, "shape": list(t.shape), "strides": list(t.stride()), "dtype": str(t.dtype).replace("torch.", ""),
                                "contiguous": bool(t.is_contiguous()), "storage_offset": int(t.storage_offset()), "host": host})
        return hook

    handles = []
    hooked: list[str] = []
    for name, mod in eng.model.named_modules():
        if _LAYER_RE.match(name) or name in FINAL_NORM_MODULES or name in ("logits_processor", "lm_head"):
            handles.append(mod.register_forward_hook(make_hook(name)))
            hooked.append(name)
    print(f"[nonint/b] {len(hooked)} forward hooks: {hooked[:3]} ... {hooked[-3:]}", flush=True)

    engine = eng.llm.llm_engine
    # [coverage D87] the same step-indexed arrival rule as vllm_adapter.run_requests_arrivals (B8-mixed): request r is admitted
    # right before engine step r["arrive_step"]; an idle engine admits the next arrival at once.  All at step 0 = the old loop.
    arrivals = sorted(wl["requests"], key=lambda r: int(r.get("arrive_step") or 0))
    steps: list[dict] = []
    tokens: dict[str, list[int]] = {}
    k, forwards = 0, AR.ForwardCounter(engine, AR.scheduler_of(eng.llm))
    t_run0 = time.perf_counter()

    def _admit(r: dict) -> None:
        engine.add_request(request_id=r["request_id"], prompt={"prompt_token_ids": list(r["prompt_token_ids"])},
                           params=va.sampling_params(r, wl["sampling"]))

    while arrivals or engine.has_unfinished_requests():
        while arrivals and int(arrivals[0].get("arrive_step") or 0) <= forwards.n:
            _admit(arrivals.pop(0))
        if not engine.has_unfinished_requests():
            _admit(arrivals.pop(0))
            continue
        t0 = time.perf_counter()
        forwards.stepped(outs := engine.step())
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        recs = []
        for p in pending:
            data = p.pop("host").flatten().view(torch.uint8).numpy().tobytes()   # raw bytes (bf16 has no numpy dtype)
            p["nbytes"] = len(data)
            p["sha256"] = _sha(data)
            recs.append(p)
        pending.clear()
        before = {rid: list(t) for rid, t in tokens.items()}
        for o in outs:
            tokens[o.request_id] = [int(t) for t in o.outputs[0].token_ids]
        new = {rid: tokens[rid][len(before.get(rid, [])):] for rid in tokens}
        steps.append({"engine_step": k, "wall_s": wall, "records": recs, "new_tokens": new,
                      "tokens_after": {rid: list(t) for rid, t in tokens.items()}})
        k += 1
    run_s = time.perf_counter() - t_run0
    for h in handles:
        h.remove()
    n = 0
    for s in steps:
        if s["records"]:
            s["step"] = n
            n += 1
        else:
            s["step"] = None
    doc = {"schema": "verity-capture/noninterference-hooks/v1", "utc": _utc(), "variant": "hooks",
           "method": "bare engine (no observer, no dispatch mode, no Triton hook); register_forward_hook on model.layers.N, model.norm, "
                     "logits_processor, lm_head; async pinned D2H copy of each output tensor at hook time; one torch.cuda.synchronize() "
                     "+ sha256 after every engine.step()",
           "hooked_modules": hooked, "hook_fire_counts": fired, "engine_steps": len(steps), "steps_with_records": n, "run_s": run_s,
           "steps": steps}
    (out / "hooks.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(f"[nonint/b] run {run_s:.2f}s: {len(steps)} engine steps, {n} with hook records, "
          f"fire counts {dict(sorted(fired.items(), key=lambda kv: kv[0])[:2])} ... lm_head={fired.get('lm_head', 0)}", flush=True)
    # engine_steps = forwards (iterations whose hooks fired), the unit the observer / census / gates count; the raw iteration count is
    # kept as step_iterations (vLLM V1 returns the previous forward's sample: one empty first return + one drain iteration; D87)
    timing = {"wall_s": run_s, "engine_steps": n, "step_iterations": len(steps)}
    nxt = {s["engine_step"]: s["new_tokens"] for s in steps}        # engine.step() reports the previous step's sample (see hooks_boundary_hashes)
    _write_tokens(out, wl, tokens, timing, wl["accepted"], {str(s["step"]): [t for v in nxt.get(s["engine_step"] + 1, {}).values() for t in v]
                                                             for s in steps if s["step"] is not None}, "hooks")


def _write_tokens(out: Path, wl: dict, tokens: dict, timing: dict, accepted: dict, per_step: dict | None, variant: str) -> None:
    doc = {"requests": [], "per_step_sampled": per_step, "timing": timing, "variant": variant,
           "serving_profile": wl.get("serving_profile"), "context_class": wl.get("context_class"),
           "expected_engine_steps": wl.get("expected_engine_steps"), "engine_steps": timing.get("engine_steps")}
    all_eq = True
    per_req = timing.get("requests") or {}
    for r in wl["requests"]:
        rid = r["request_id"]
        acc = accepted.get(rid)
        gen = tokens.get(rid, [])
        eq = (gen == acc) if acc is not None else None
        all_eq = all_eq and bool(eq)
        sr = per_req.get(rid) or {}
        doc["requests"].append({"request_id": rid, "prompt_token_ids": r["prompt_token_ids"], "max_tokens": r["max_tokens"],
                                "generated_token_ids": gen, "accepted_record_token_ids": acc, "equals_accepted": eq,
                                "prompt_len": len(r["prompt_token_ids"]), "n_generated": len(gen),          # realized lengths (D87)
                                "arrive_step": int(r.get("arrive_step") or 0), "first_step": sr.get("first_step"),
                                "finish_step": sr.get("finish_step"), "finish_reason": sr.get("finish_reason")})
        print(f"[nonint/{variant}] {rid}: generated {gen} equals_accepted={eq}", flush=True)
    doc["all_equal_accepted"] = all_eq
    (out / "tokens.json").write_text(json.dumps(doc, indent=1) + "\n")


# ---------------------------------------------------------------------------------------------------------
# comparison (torch-free)
# ---------------------------------------------------------------------------------------------------------


def observer_boundary_hashes(a_dir: Path) -> tuple[dict, list[str]]:
    """From (a)'s log + snapshots.json: {step: {"hidden": {L: sha}, "residual": {L: sha}, "final": sha, "logits": sha}}."""
    from verity_vllm.observe.events import DispatchEnter, TritonLaunch
    from verity_vllm.observe.log import read_log, split_steps
    from verity_vllm.observe.fold.tree import step_events

    idx = _read_json(a_dir / "snapshots.json")
    by_key: dict[tuple[int, int, str], str] = {}
    for e in idx["entries"]:
        if "sha256" in e and "step" in e:
            by_key[(int(e["step"]), int(e["event"]), e["path"])] = e["sha256"]
    out: dict[int, dict] = {}
    problems: list[str] = []
    sample_ids: dict[int, list[int]] = {}
    sample_rows: dict[int, list[str]] = {}       # request id per sampled row (StepStart.logits_rows)
    for _pre, step in split_steps(read_log(a_dir / "log.jsonl.gz")):
        if not step:
            continue
        se = step_events(step)
        s = se.start.step
        rec = {"hidden": {}, "residual": {}, "final": None, "logits": None, "rows": len(se.start.rows)}
        if se.sample is not None:
            sample_ids[s] = list(se.sample.sampled_token_ids)
            sample_rows[s] = [str(r) for r, _ in se.start.logits_rows]
        for node in se.all_nodes():
            ev = node.enter
            mod = ev.module or ""
            if isinstance(ev, TritonLaunch) and ev.fn.endswith("matmul_kernel_persistent"):
                m = _DOWN_RE.match(mod)
                if m and ev.phase == "forward":
                    L = int(m.group(1))
                    sha = by_key.get((s, ev.id, "args.c_ptr"))
                    if sha is None:
                        problems.append(f"step {s}: no snapshot for down_proj launch id {ev.id} (layer {L})")
                    elif L in rec["hidden"]:
                        problems.append(f"step {s}: two down_proj launches for layer {L}")
                    else:
                        rec["hidden"][L] = sha
                elif mod == "logits_processor" or ev.phase == "sample":
                    sha = by_key.get((s, ev.id, "args.c_ptr"))
                    if rec["logits"] is not None:
                        problems.append(f"step {s}: two sample-phase GEMM launches (ids ..., {ev.id})")
                    rec["logits"] = sha
                    if sha is None:
                        problems.append(f"step {s}: no snapshot for the lm_head launch id {ev.id}")
            elif isinstance(ev, DispatchEnter) and ev.op.startswith("vllm.moe_forward") and ev.phase == "forward":
                m = _MOE_RE.match(mod)
                if m:
                    L = int(m.group(1))
                    sha = by_key.get((s, ev.id, "outs"))
                    if sha is None:
                        problems.append(f"step {s}: no snapshot for moe_forward id {ev.id} outs (layer {L})")
                    elif L in rec["hidden"]:
                        problems.append(f"step {s}: two MoE block outputs for layer {L}")
                    else:
                        rec["hidden"][L] = sha
            elif isinstance(ev, DispatchEnter) and ev.op.endswith(".fused_add_rms_norm.default"):
                m = _POST_RE.match(mod)
                if m:
                    L = int(m.group(1))
                    sha = by_key.get((s, ev.id, "args[1]"))
                    if sha is None:
                        problems.append(f"step {s}: no snapshot for post_attention_layernorm id {ev.id} args[1] (layer {L})")
                    else:
                        rec["residual"][L] = sha
                elif mod == "model.norm":
                    sha = by_key.get((s, ev.id, "args[0]"))
                    if sha is None:
                        problems.append(f"step {s}: no snapshot for model.norm id {ev.id} args[0]")
                    rec["final"] = sha
        if not rec["residual"] and rec["final"] is None:
            unfused_norm_boundaries(se, by_key, s, rec, problems)
        if not rec["hidden"] and rec["final"] is None:
            parallel_residual_boundaries(se, by_key, s, rec, problems)   # GPT-NeoX
        out[s] = rec
    return {"steps": out, "sample_ids": sample_ids, "sample_rows": sample_rows}, problems


def hooks_boundary_hashes(b_dir: Path) -> tuple[dict, list[str]]:
    doc = _read_json(b_dir / "hooks.json")
    out: dict[int, dict] = {}
    problems: list[str] = []
    for s in doc["steps"]:
        if s.get("step") is None:
            continue
        rec = {"hidden": {}, "residual": {}, "final": None, "logits": None, "rows": None, "noncontiguous": []}
        for r in s["records"]:
            if not r["contiguous"]:
                rec["noncontiguous"].append(f"{r['module']}[{r['index']}]")
            m = _LAYER_RE.match(r["module"])
            if m:
                L = int(m.group(1))
                if r["index"] == 0:
                    rec["hidden"][L] = r["sha256"]
                    rec["rows"] = r["shape"][0]
                elif r["index"] == 1:
                    rec["residual"][L] = r["sha256"]
            elif r["module"] in FINAL_NORM_MODULES and r["index"] == 0:
                rec["final"] = r["sha256"]
            elif r["module"] == "logits_processor" and r["index"] == 0:
                rec["logits"] = r["sha256"]
        out[s["step"]] = rec
    # vLLM's LLMEngine.step() returns the ids sampled by the PREVIOUS engine step (one-step output lag; the trailing 17th
    # engine step, which launches no kernel, carries the last token): the ids sampled in device step s are the new tokens
    # reported by engine step (engine_step(s) + 1).
    by_engine_step = {s["engine_step"]: s["new_tokens"] for s in doc["steps"]}
    new_tokens = {s["step"]: by_engine_step.get(s["engine_step"] + 1, {}) for s in doc["steps"] if s.get("step") is not None}
    return {"steps": out, "new_tokens": new_tokens,
            "hooked_modules": doc.get("hooked_modules"), "hook_fire_counts": doc.get("hook_fire_counts"), "method": doc.get("method")}, problems


def _tokens_of(path: Path) -> dict[str, list[int]] | None:
    if not path.exists():
        return None
    doc = json.loads(path.read_text())
    return {r["request_id"]: list(r["generated_token_ids"]) for r in doc["requests"]}


def _accepted_tokens(accepted_ctl: str | None, out: Path | None = None) -> dict[str, list[int]]:
    """The reference tokens of gate G1: what the CLEAN, UNINSTRUMENTED run generated — the control run's tokens.json
    (`--accepted-ctl`, `generated_token_ids`), else the c_bare variant's own generated tokens under --out.  The workload
    file's authored `accepted_record_token_ids` is never the reference (generalization plan: expected tokens are not a
    fixture its author can edit; see `_workload_accepted` for the diagnostic that still reports it)."""
    for cand in (accepted_ctl, str(out / "c_bare" / "tokens.json") if out else None):
        if cand and Path(cand).exists():
            doc = json.loads(Path(cand).read_text())
            gen = {r["request_id"]: list(r["generated_token_ids"]) for r in doc["requests"] if r.get("generated_token_ids") is not None}
            if gen:
                return gen
    return {}


def _workload_accepted(accepted_ctl: str | None, out: Path | None = None) -> dict[str, list[int]]:
    """Diagnostic only: the workload file's authored accepted ids as echoed by a run's tokens.json (may be empty)."""
    for cand in (accepted_ctl, str(out / "c_bare" / "tokens.json") if out else None):
        if cand and Path(cand).exists():
            doc = json.loads(Path(cand).read_text())
            acc = {r["request_id"]: list(r["accepted_record_token_ids"]) for r in doc["requests"] if r.get("accepted_record_token_ids")}
            if acc:
                return acc
    return {}


def case_of_args(args) -> str | None:
    """`--case`, else the workload file's `case`, else "B0" (mirrors `vllm_adapter.case_of` without importing torch)."""
    if getattr(args, "case", None):
        return args.case
    try:
        doc = json.loads(Path(args.workload).read_text())
        return doc.get("case") or "B0"
    except Exception:  # noqa: BLE001
        return "B0"


def expected_layers_for(case: str | None) -> int | None:
    """`C["NL"]` of the checkpoint role's profile — the architectural layer count the run must show (gate I3)."""
    if not case:
        return None
    try:
        from verity_vllm.observe.fold.profiles import profile_for_case
        return int(profile_for_case(case).config["NL"])
    except Exception:  # noqa: BLE001
        return None


def compare(dir_: str, accepted_ctl: str | None, expected_layers: int | None = None) -> dict:
    d = Path(dir_)
    a, b, c = d / "a_observer", d / "b_hooks", d / "c_bare"
    problems: list[str] = []
    toks = {"a_observer": _tokens_of(a / "tokens.json"), "b_hooks": _tokens_of(b / "tokens.json"), "c_bare": _tokens_of(c / "tokens.json"),
            "accepted": _accepted_tokens(accepted_ctl, Path(dir_))}
    if accepted_ctl and Path(accepted_ctl).exists():
        toks["accepted_ctl_file"] = _tokens_of(Path(accepted_ctl))
    for k, v in toks.items():
        if v is None:
            problems.append(f"missing tokens for {k}")
    ref = toks["accepted"]
    if not ref:
        problems.append("no clean reference tokens: neither --accepted-ctl (control run) nor c_bare/tokens.json has generated_token_ids")
    token_eq = {k: (v == {rid: ref[rid] for rid in v} if v is not None and ref and all(rid in ref for rid in v) else False) for k, v in toks.items() if k != "accepted"}
    wl = _workload_accepted(accepted_ctl, Path(dir_))
    workload_accepted_eq = ({k: (v == {rid: wl[rid] for rid in v} if v is not None and all(rid in wl for rid in v) else None)
                             for k, v in toks.items() if k != "accepted"} if wl else None)   # diagnostic, never gating
    hashes_a, pa = observer_boundary_hashes(a)
    hashes_b, pb = hooks_boundary_hashes(b)
    problems += pa + pb
    steps_a, steps_b = hashes_a["steps"], hashes_b["steps"]
    steps = sorted(set(steps_a) | set(steps_b))
    per_step = []
    n_cmp = n_eq = 0
    mismatches: list[dict] = []
    for s in steps:
        ra, rb = steps_a.get(s), steps_b.get(s)
        if ra is None or rb is None:
            problems.append(f"step {s} present only in {'(a)' if ra else '(b)'}")
            per_step.append({"step": s, "error": "missing on one side"})
            continue
        row = {"step": s, "rows_a": ra["rows"], "rows_b": rb["rows"], "layers_a": len(ra["hidden"]), "layers_b": len(rb["hidden"]),
               "hidden_eq": 0, "residual_eq": 0, "final_eq": None, "logits_eq": None, "compared": 0, "equal": 0,
               "noncontiguous_b": rb.get("noncontiguous", [])}
        for kind in ("hidden", "residual"):
            for L in sorted(set(ra[kind]) | set(rb[kind])):
                ha, hb = ra[kind].get(L), rb[kind].get(L)
                row["compared"] += 1
                if ha is not None and ha == hb:
                    row[f"{kind}_eq"] += 1
                    row["equal"] += 1
                else:
                    mismatches.append({"step": s, "layer": L, "tensor": kind, "a": ha, "b": hb})
        for kind in ("final", "logits"):
            ha, hb = ra[kind], rb[kind]
            row["compared"] += 1
            eq = ha is not None and ha == hb
            row[f"{kind}_eq"] = eq
            if eq:
                row["equal"] += 1
            else:
                mismatches.append({"step": s, "layer": None, "tensor": kind, "a": ha, "b": hb})
        sa = hashes_a["sample_ids"].get(s)
        sb = [t for v in (hashes_b["new_tokens"].get(s) or {}).values() for t in v]
        row["sampled_a"], row["sampled_b"] = sa, sb
        # the runner orders its batch by scheduled-token count, not by request: compare per request when the
        # log names the request of each sampled row (StepStart.logits_rows); the positional result stays as a diagnostic
        rows_a = (hashes_a.get("sample_rows") or {}).get(s)
        if sa is not None and rows_a is not None and len(rows_a) == len(sa):
            by_req_a = {r: int(t) for r, t in zip(rows_a, sa)}
            by_req_b = {r: int(v[0]) for r, v in (hashes_b["new_tokens"].get(s) or {}).items() if v}
            row["sampled_eq_by_request"] = by_req_a == by_req_b
            row["batch_order_is_request_order"] = sa == sb
            row["sampled_eq"] = row["sampled_eq_by_request"]
            if not row["sampled_eq"]:
                problems.append(f"step {s}: SampleEnd ids by request {by_req_a} != hooks-run new tokens {by_req_b}")
        else:
            row["sampled_eq"] = sa == sb
            if not row["sampled_eq"]:
                problems.append(f"step {s}: SampleEnd ids {sa} != hooks-run new tokens {sb}")
        n_cmp += row["compared"]
        n_eq += row["equal"]
        per_step.append(row)
    # expected shape of the run: one engine step per generated token of a one-request workload (16 for the M1 / M6
    # workloads), and the same decoder-layer count in both variants at every step (30 for SmolLM2, 28 for Qwen2.5-1.5B) [M6]
    expected_steps = 16
    expected_steps_source = "default"
    try:
        c_doc = json.loads((d / "c_bare" / "tokens.json").read_text())
        # [coverage D87] the bare control run's realized engine-step count is the reference (arrivals, EOS before max_tokens and
        # chunked prefill all make `max(max_tokens)` wrong); the historical rule is the fallback for older tokens.json files
        if (c_doc.get("timing") or {}).get("engine_steps") is not None:
            expected_steps = int(c_doc["timing"]["engine_steps"])
            expected_steps_source = "c_bare timing.engine_steps"
        else:
            expected_steps = max(int(r["max_tokens"]) for r in c_doc["requests"])
            expected_steps_source = "max(max_tokens)"
    except Exception:  # noqa: BLE001
        pass
    layer_counts = {(r.get("layers_a"), r.get("layers_b")) for r in per_step}
    layers_ok = len(layer_counts) == 1 and all(a == b and a and a > 0 for a, b in layer_counts)
    # gate I3 (generalization plan): the layer count is checked against the profile's C["NL"], not only for self-consistency
    # between the variants (the M6 form).  Unknown expectation => not PASS (say why in `problems`).
    seen_layers = sorted(layer_counts)[0][0] if layers_ok else None
    if expected_layers is None:
        problems.append("expected_layers unknown (no --case / --expected-layers): gate I3 needs the profile's C[\"NL\"]")
    elif layers_ok and seen_layers != expected_layers:
        problems.append(f"decoder layers per step {seen_layers} != profile C[\"NL\"] {expected_layers}")
    layers_ok = layers_ok and expected_layers is not None and seen_layers == expected_layers
    ok = (not problems and n_cmp > 0 and n_eq == n_cmp and all(token_eq.values()) and len(steps) == expected_steps and layers_ok)
    rep = {
        "expected_steps": expected_steps, "expected_steps_source": expected_steps_source, "expected_layers": expected_layers,
        "layers": seen_layers if seen_layers is not None else sorted(map(str, layer_counts)),
        "schema": "verity-capture/noninterference/v1", "utc": _utc(), "dir": str(d),
        "variants": {"a_observer": "full observer + hash-only snapshots of every step (HashOnlyCapture)",
                     "b_hooks": hashes_b.get("method"), "c_bare": "bench.m1_capture --no-observer (vllm_adapter.run_requests, nothing installed)"},
        "hooked_modules_b": hashes_b.get("hooked_modules"), "hook_fire_counts_b": hashes_b.get("hook_fire_counts"),
        "tokens": {k: v for k, v in toks.items()}, "tokens_equal_accepted": token_eq, "reference": "clean control run (generated_token_ids)",
        "tokens_equal_workload_accepted": workload_accepted_eq,
        "steps": per_step, "hash_comparisons": n_cmp, "hash_equal": n_eq, "mismatches": mismatches[:64], "problems": problems,
        "PASS": ok,
    }
    return rep


# ---------------------------------------------------------------------------------------------------------
# the property record, world-parametric: a run's observed arm against its bare arm
# ---------------------------------------------------------------------------------------------------------

SCHEMA = "verity-vllm/noninterference/v2"
#: the tensor-parallel worlds a record is taken at; there are no collective records above two ranks, so a larger world is refused
WORLDS = (1, 2)


def token_parity(observed: dict, bare: dict) -> dict:
    """The two arms' requests in order, as (request id, prompt ids, generated ids): equal, and at least one."""
    a, b = ([(q["request_id"], q["prompt_token_ids"], q["generated_token_ids"]) for q in d["requests"]] for d in (observed, bare))
    differing = [x[0] for x, y in zip(a, b) if x != y] + [x[0] for x in a[len(b):] + b[len(a):]]
    return {"equal": a == b and len(a) > 0, "requests": len(a), "requests_bare": len(b), "differing": differing[:16]}


def record(match_dir: str | Path, world: int, *, run: str | None = None, hashes: str | Path | None = None) -> dict:
    """The sealed record of one run at tensor-parallel `world`: `match_dir/capture/tokens.json` (the observed arm: the observer, at
    world 2 with the collective recorder, on every rank) against `match_dir/control/tokens.json` (the bare arm), plus, when
    `hashes` names a `compare` report (`noninterference.json`, world 1), its boundary-hash comparison."""
    if world not in WORLDS:
        raise ValueError(f"tensor-parallel world {world} refused: non-interference is recorded at worlds {WORLDS}")
    if hashes is not None and world != 1:
        raise ValueError(f"boundary hashes are compared at world 1, not {world}")
    d = Path(match_dir)
    arms: dict[str, dict] = {}
    docs: dict[str, dict] = {}
    problems: list[str] = []
    for arm, sub in (("observed", "capture"), ("bare", "control")):
        p = d / sub / "tokens.json"
        if not p.exists():
            problems.append(f"missing {sub}/tokens.json")
            continue
        raw = p.read_bytes()
        docs[arm] = json.loads(raw)
        arms[arm] = {"file": f"{sub}/tokens.json", "sha256": _sha(raw), "world": int(docs[arm].get("tp") or 1)}
        if arms[arm]["world"] != world:
            problems.append(f"{sub}/tokens.json was taken at world {arms[arm]['world']}, not {world}")
    parity = token_parity(docs["observed"], docs["bare"]) if len(docs) == 2 else None
    boundary = None
    if hashes is not None:
        raw = Path(hashes).read_bytes()
        rep = json.loads(raw)
        boundary = {"sha256": _sha(raw), "PASS": rep.get("PASS") is True, "problems": (rep.get("problems") or [])[:8],
                    **{k: rep.get(k) for k in ("hash_comparisons", "hash_equal", "expected_steps", "layers", "expected_layers")}}
    observed = docs.get("observed") or {}
    doc = {"schema": SCHEMA, "world": world, "run": run, "subject": {k: observed[k] for k in ("case", "model") if observed.get(k) is not None},
           "arms": arms, "tokens": parity, "boundary_hashes": boundary, "problems": problems}
    return seal(doc, not problems and bool(parity and parity["equal"]) and (boundary is None or boundary["PASS"]))


def compare_markdown(rep: dict) -> str:
    L: list[str] = []
    L.append("# Non-interference: observer vs bare (M1 workload)")
    L.append("")
    L.append(f"- compared {rep['utc']}; results in `{rep['dir']}`; **{'PASS' if rep['PASS'] else 'FAIL'}**")
    L.append(f"- (a) {rep['variants']['a_observer']}")
    L.append(f"- (b) {rep['variants']['b_hooks']}")
    L.append(f"- (c) {rep['variants']['c_bare']}")
    L.append(f"- hooks in (b): {rep['hooked_modules_b'] and len(rep['hooked_modules_b'])} modules; fire counts: "
             f"{ {k: v for k, v in (rep['hook_fire_counts_b'] or {}).items() if not k.startswith('model.layers.') or k.endswith('.0')} }")
    L.append("")
    L.append("## Tokens")
    L.append("")
    L.append("| variant | generated ids | == accepted record |")
    L.append("| --- | --- | --- |")
    for k, v in rep["tokens"].items():
        ids = next(iter(v.values())) if v else None
        eq = rep["tokens_equal_accepted"].get(k, "(reference)")
        L.append(f"| {k} | `{ids}` | {eq} |")
    L.append("")
    L.append("## Boundary hashes (a) vs (b)")
    L.append("")
    L.append(f"{rep['hash_equal']} / {rep['hash_comparisons']} sha256 pairs equal (per step: 30 hidden + 30 residual + final norm + logits).")
    L.append("")
    L.append("| step | rows a/b | layers a/b | hidden eq | residual eq | final eq | logits eq | sampled a == b |")
    L.append("| ---: | --- | --- | ---: | ---: | --- | --- | --- |")
    for r in rep["steps"]:
        if "error" in r:
            L.append(f"| {r['step']} | {r['error']} | | | | | | |")
            continue
        L.append(f"| {r['step']} | {r['rows_a']}/{r['rows_b']} | {r['layers_a']}/{r['layers_b']} | {r['hidden_eq']}/{r['layers_a']} | "
                 f"{r['residual_eq']}/{r['layers_a']} | {r['final_eq']} | {r['logits_eq']} | {r['sampled_eq']} ({r['sampled_a']}) |")
    L.append("")
    if rep["mismatches"]:
        L.append("## Mismatches")
        L.append("")
        for m in rep["mismatches"]:
            L.append(f"- step {m['step']} layer {m['layer']} {m['tensor']}: a={m['a']} b={m['b']}")
        L.append("")
    if rep["problems"]:
        L.append("## Problems")
        L.append("")
        for p in rep["problems"]:
            L.append(f"- {p}")
        L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------------------------------------


def _engine_args(items: list[str]) -> dict:
    from verity_vllm.engine import vllm_adapter as va  # torch import only on the GPU half
    return va.parse_engine_args(items)


def orchestrate(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    py = args.python or sys.executable
    env = settings.environ()
    common = ["--workload", args.workload, "--checkpoints", args.checkpoints, "--max-num-seqs", str(args.max_num_seqs)]
    if args.case:
        common += ["--case", args.case]
    if args.max_tokens is not None:
        common += ["--max-tokens", str(args.max_tokens)]
    common += [f"--engine-arg={x}" for x in args.engine_arg]
    runs = [
        ("a_observer", [py, "-m", "verity_vllm.pipeline.cli", "noninterference", "--variant", "observer", "--out", str(out / "a_observer"), *common]),
        ("b_hooks", [py, "-m", "verity_vllm.pipeline.cli", "noninterference", "--variant", "hooks", "--out", str(out / "b_hooks"), *common]),
        ("c_bare", [py, "-m", "verity_vllm.pipeline.cli", "m1-capture", "--no-observer", "--out", str(out / "c_bare"), *common]),
    ]
    timings = {}
    for name, cmd in runs:
        if args.only and name not in args.only.split(","):
            continue
        print(f"[nonint] === {name}: {' '.join(cmd)}", flush=True)
        t0 = time.perf_counter()
        with open(out / f"{name}.log", "w") as logf:
            rc = subprocess.call(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)
        timings[name] = {"rc": rc, "s": round(time.perf_counter() - t0, 1)}
        print(f"[nonint] === {name}: rc={rc} in {timings[name]['s']}s", flush=True)
        if rc != 0:
            print(Path(out / f"{name}.log").read_text()[-3000:], flush=True)
    rep = compare(str(out), args.accepted_ctl, args.expected_layers if args.expected_layers is not None else expected_layers_for(case_of_args(args)))
    rep["subprocesses"] = timings
    (out / "noninterference.json").write_text(json.dumps(rep, indent=1) + "\n")
    (out / "noninterference.md").write_text(compare_markdown(rep))
    print(f"[nonint] {'PASS' if rep['PASS'] else 'FAIL'}: hashes {rep['hash_equal']}/{rep['hash_comparisons']}, tokens {rep['tokens_equal_accepted']}, "
          f"problems {len(rep['problems'])} -> {out / 'noninterference.json'}", flush=True)
    return 0 if rep["PASS"] else 1


@dataclass(frozen=True, kw_only=True)
class Options:
    workload: str = option("--workload", type=workload_arg, default="workloads/workload_32x16_1req.json")
    out: str = option("--out")
    checkpoints: str = option("--checkpoints", default="manifests/checkpoints.json")
    max_tokens: int | None = option("--max-tokens", type=int, default=None)
    max_num_seqs: int = option("--max-num-seqs", type=int, default=8)
    no_warmup: bool = option("--no-warmup", default=False, action="store_true")
    variant: str | None = option("--variant", choices=["observer", "hooks"], default=None, help="(internal) run one variant in this process")
    compare: bool = option("--compare", default=False, action="store_true", help="torch-free: compare the three variants' outputs under --out")
    only: str | None = option("--only", default=None, help="orchestrator: comma list of a_observer,b_hooks,c_bare to (re)run")
    python: str | None = option("--python", default=None)
    accepted_ctl: str = option("--accepted-ctl", default="data/logs/m1_ctl_tokens.json")
    case: str | None = option("--case", default=None, help="checkpoint role (B0 / B1); default: the workload file's, else B0")
    expected_layers: int | None = option("--expected-layers", type=int, default=None, help="gate I3: decoder layers per step (default: the --case profile's C[\"NL\"])")
    engine_arg: list[str] = option("--engine-arg", action="append", default=[], help="LLM(**kw) override key=value (JSON value), repeatable; forwarded to all three variants")


def main(args: Options) -> int:
    if args.variant == "observer":
        run_observer_variant(args.workload, args.out, args.checkpoints, max_tokens=args.max_tokens, max_num_seqs=args.max_num_seqs,
                             warmup=not args.no_warmup, case=args.case, engine_args=_engine_args(args.engine_arg))
        return 0
    if args.variant == "hooks":
        run_hooks_variant(args.workload, args.out, args.checkpoints, max_tokens=args.max_tokens, max_num_seqs=args.max_num_seqs,
                          warmup=not args.no_warmup, case=args.case, engine_args=_engine_args(args.engine_arg))
        return 0
    if args.compare:
        rep = compare(args.out, args.accepted_ctl, args.expected_layers if args.expected_layers is not None else expected_layers_for(case_of_args(args)))
        out = Path(args.out)
        (out / "noninterference.json").write_text(json.dumps(rep, indent=1) + "\n")
        (out / "noninterference.md").write_text(compare_markdown(rep))
        print(f"[nonint] {'PASS' if rep['PASS'] else 'FAIL'}: hashes {rep['hash_equal']}/{rep['hash_comparisons']}, tokens {rep['tokens_equal_accepted']}, "
              f"problems {rep['problems'][:5]}")
        return 0 if rep["PASS"] else 1
    return orchestrate(args)
