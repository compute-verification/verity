"""[R9 CB-C] Pod driver: build the real pinned vLLM engine for SmolLM2-135M under the admitted BI / custom-op profile
with vLLM's own compilation pipeline (mode VLLM_COMPILE), capture the Dynamo FX graph the pipeline hands to
``VllmBackend`` (pre-split, pre-AOTAutograd, pre-Inductor), and record everything about how it was obtained.

    PYTHONPATH=. VERITY_CBC_OUT=/workspace/cp/cbc/run1 /workspace/venv-cu129/bin/python -m verity_capture.experimental.cb_c.dump_fx \
        --out /workspace/cp/cbc/run1 --backend eager --custom-ops all [--postgrad] [--prompt-len 8 --max-tokens 4]

Construction-input trace: ``sys.addaudithook`` logs every ``open`` (path, mode) into ``opens.jsonl`` from process start; the
summary classifies paths (vllm source / torch / model dir / repo / other).  The forbidden inputs (target program.json,
execution record, accepted digest, experimental/aot/build.py) are checked against the trace after the run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

FORBIDDEN_SUBSTRINGS = ("program.json", "instances.jsonl", "experimental/aot/build.py", "census/B0-cov", "archive-20260917", "log.jsonl")

_OPENS: list[dict[str, Any]] = []
_SEEN: set[str] = set()


def _audit(event: str, args: tuple) -> None:
    if event != "open":
        return
    try:
        path = args[0]
        if not isinstance(path, str):
            path = os.fsdecode(path) if isinstance(path, (bytes, os.PathLike)) else str(path)
        mode = args[1] if len(args) > 1 else None
        key = f"{path}|{mode}"
        if key in _SEEN:
            return
        _SEEN.add(key)
        _OPENS.append({"path": path, "mode": mode, "t": time.time()})
    except Exception:  # noqa: BLE001
        pass


def classify(path: str) -> str:
    if "/vllm/" in path and "site-packages" in path:
        return "vllm-source"
    if "/torch/" in path and "site-packages" in path:
        return "torch"
    if "site-packages" in path:
        return "site-packages-other"
    if "/hf/" in path or "snapshots" in path:
        return "model-dir"
    if "/veritor/" in path or "/verity" in path:
        return "repo"
    if "torch_compile_cache" in path or "inductor" in path.lower() or "/torchinductor" in path.lower():
        return "compile-cache"
    if path.startswith("/proc") or path.startswith("/sys") or path.startswith("/dev") or path.startswith("/etc") or path.startswith("/usr/lib") or path.startswith("/tmp"):
        return "system"
    return "other"


def main() -> None:
    sys.addaudithook(_audit)
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoints", default="manifests/checkpoints.json")
    ap.add_argument("--case", default="B0")
    ap.add_argument("--backend", default="eager", choices=["eager", "inductor"], help="CompilationConfig.backend; eager = no AOTAutograd/Inductor at all")
    ap.add_argument("--custom-ops", default="all", help="CompilationConfig.custom_ops base mode (all | none | e.g. 'all,-rms_norm')")
    ap.add_argument("--splitting-ops", default=None, help="JSON list; default None = vLLM default (attention + kv-cache update)")
    ap.add_argument("--postgrad", action="store_true", help="also install the W1d RecordInsertionPass in dry-run/dump mode (inductor backend only)")
    ap.add_argument("--prompt-len", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=4)
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--no-run", action="store_true", help="only build the engine (compile happens at build); skip the request")
    ap.add_argument("--no-aux", action="store_true", help="skip the Dynamo capture of compute_logits / gumbel_sample")
    ap.add_argument("--extra", default="{}", help="JSON dict of extra LLM kwargs (recorded)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    os.environ["VERITY_CBC_OUT"] = str(out)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(out / "inductor_cache"))
    os.environ.setdefault("VLLM_CACHE_ROOT", str(out / "vllm_cache"))
    t_start = time.time()

    from verity_vllm_adapter import profile as prof

    env_flags = prof.apply_env()
    import torch
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    from verity_vllm_adapter.compiled_fx import dynamo_capture

    patched = dynamo_capture.install(out)
    postgrad_table = None
    if args.postgrad:
        from verity_vllm_adapter.compiled_fx import fx_pass

        postgrad_table = fx_pass.FxIdTable()
        rpass = fx_pass.RecordInsertionPass(postgrad_table, dry_run=True, dump_graphs=True)
        patched += fx_pass.install_hooks(rpass)

    cp = prof.checkpoint_for(args.case, args.checkpoints)
    kw = prof.engine_kwargs(cp, replay=False, enforce_eager=False)
    cc: dict[str, Any] = {"mode": 3, "backend": args.backend, "custom_ops": [s for s in args.custom_ops.split(",") if s], "cudagraph_mode": "NONE"}
    if args.splitting_ops is not None:
        cc["splitting_ops"] = json.loads(args.splitting_ops)
    kw["compilation_config"] = cc
    kw["enable_prefix_caching"] = False
    kw["max_num_seqs"] = 1
    if args.max_model_len:
        kw["max_model_len"] = int(args.max_model_len)
    kw.update(json.loads(args.extra))
    doc: dict[str, Any] = {
        "schema": "verity-cbc/dump-fx/v1", "t_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "argv": sys.argv, "engine_kwargs": kw, "env_flags": env_flags,
        "env": {k: os.environ.get(k) for k in ("VLLM_BATCH_INVARIANT", "VLLM_USE_V2_MODEL_RUNNER", "VLLM_ENABLE_V1_MULTIPROCESSING", "VLLM_USE_FLASHINFER_SAMPLER",
                                                "TORCHINDUCTOR_CACHE_DIR", "VLLM_CACHE_ROOT", "HF_HOME", "HF_HUB_OFFLINE", "VLLM_ATTENTION_BACKEND")},
        "versions": {"vllm": getattr(vllm, "__version__", None), "vllm_commit": getattr(vllm, "__commit__", None) or getattr(getattr(vllm, "version", None), "__commit__", None),
                     "torch": torch.__version__, "python": sys.version.split()[0],
                     "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        "patched": patched, "checkpoint": {k: cp.get(k) for k in ("repo", "revision", "local_path", "config_sha256")},
    }
    export = Path("EXPORT.json")
    if export.exists():
        doc["shipped_tree"] = json.loads(export.read_text())
    (out / "run.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")

    t0 = time.perf_counter()
    err = None
    llm = None
    try:
        llm = LLM(**kw)
    except Exception as e:  # noqa: BLE001
        import traceback

        err = {"type": type(e).__name__, "msg": str(e)[:4000], "tb": traceback.format_exc()[-8000:]}
    doc["build_s"] = time.perf_counter() - t0
    doc["build_error"] = err
    if llm is not None:
        vc = llm.llm_engine.vllm_config
        ccx = vc.compilation_config
        doc["effective_compilation_config"] = {k: (str(v) if not isinstance(v, (int, float, str, bool, list, dict, type(None))) else v)
                                               for k, v in vars(ccx).items() if not k.startswith("_") and k not in ("inductor_compile_config", "inductor_passes", "static_forward_context", "traced_files")}
        doc["effective_compilation_config"]["enabled_custom_ops"] = dict(getattr(ccx, "enabled_custom_ops", {}) or {})
        doc["effective_compilation_config"]["disabled_custom_ops"] = dict(getattr(ccx, "disabled_custom_ops", {}) or {})
        doc["effective_compilation_config"]["traced_files"] = sorted(getattr(ccx, "traced_files", []) or [])
        doc["model_config"] = {"hf_config": vc.model_config.hf_config.to_dict(), "dtype": str(vc.model_config.dtype), "max_model_len": vc.model_config.max_model_len,
                               "architectures": list(vc.model_config.architectures or [])}
        doc["attention_backend"] = str(getattr(vc, "attention_config", None) and getattr(vc.attention_config, "backend", None))
        try:
            from vllm.compilation.counter import compilation_counter

            doc["compilation_counter"] = {k: v for k, v in vars(compilation_counter).items() if not k.startswith("_")}
        except Exception as e:  # noqa: BLE001
            doc["compilation_counter"] = {"error": repr(e)}
        try:
            runner = prof.get_model_runner(llm)
            model = runner.get_model()
            doc["model_class"] = type(model).__module__ + "." + type(model).__qualname__
            doc["param_names"] = [n for n, _ in model.named_parameters()]
            doc["buffer_names"] = [n for n, _ in model.named_buffers()]
        except Exception as e:  # noqa: BLE001
            doc["model_class"] = {"error": repr(e)}
        if not args.no_run:
            toks = [(37 * i + 11) % 4000 + 100 for i in range(args.prompt_len)]
            t1 = time.perf_counter()
            outs = llm.generate([TokensPrompt(prompt_token_ids=toks)], SamplingParams(temperature=0.0, max_tokens=args.max_tokens, ignore_eos=True))
            doc["request"] = {"prompt_token_ids": toks, "max_tokens": args.max_tokens, "greedy": True, "ignore_eos": True,
                              "output_token_ids": list(outs[0].outputs[0].token_ids), "gen_s": time.perf_counter() - t1}
        # the parts of a step outside the @support_torch_compile graph: lm_head logits (model.compute_logits) and the greedy
        # sampler (vllm.v1.worker.gpu.sample.gumbel.gumbel_sample, reached from Sampler.sample when no logits processing is
        # needed and top_k/top_p are None -- source-inspected control flow, recorded in run.json). Dynamo-traced on tiny real
        # tensors on the GPU; the model's weights are real (compile-time only, no values are read by the capture).
        aux: dict[str, Any] = {}
        if not args.no_aux:
            try:
                runner = prof.get_model_runner(llm)
                model = runner.get_model()
                H = int(vc.model_config.hf_config.hidden_size)
                V = int(vc.model_config.hf_config.vocab_size)
                dev = next(model.parameters()).device
                hs = torch.zeros(1, H, dtype=torch.bfloat16, device=dev)
                aux["compute_logits"] = {k: v for k, v in dynamo_capture.capture_callable("compute_logits", model.compute_logits, hs).items() if k != "graphs"}
            except Exception as e:  # noqa: BLE001
                aux["compute_logits"] = {"error": repr(e)}
            try:
                from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

                dev = torch.device("cuda:0")
                logits = torch.zeros(1, V, dtype=torch.bfloat16, device=dev)  # logits dtype as compute_logits returns (checked in the graph)
                idx_map = torch.zeros(1, dtype=torch.int32, device=dev)
                temperature = torch.zeros(1, dtype=torch.float32, device=dev)
                seeds = torch.zeros(1, dtype=torch.int64, device=dev)
                pos = torch.zeros(1, dtype=torch.int64, device=dev)
                aux["gumbel_sample"] = {k: v for k, v in dynamo_capture.capture_callable("gumbel_sample", gumbel_sample, logits, idx_map, temperature, seeds, pos,
                                                                                          apply_temperature=False, is_drafting=False).items() if k != "graphs"}
                try:
                    sampler = getattr(runner, "sampler", None)
                    aux["sampler_class"] = type(sampler).__module__ + "." + type(sampler).__qualname__ if sampler is not None else None
                    aux["sampler_use_flashinfer"] = getattr(sampler, "use_flashinfer", None)
                except Exception as e:  # noqa: BLE001
                    aux["sampler_class"] = {"error": repr(e)}
            except Exception as e:  # noqa: BLE001
                aux["gumbel_sample"] = {"error": repr(e)}
        doc["aux"] = aux
        # cache artefacts vLLM wrote on its own
        arts = []
        for root_key in ("VLLM_CACHE_ROOT", "TORCHINDUCTOR_CACHE_DIR"):
            root = Path(os.environ[root_key])
            if root.exists():
                for p in sorted(root.rglob("*")):
                    if p.is_file():
                        arts.append({"root": root_key, "path": str(p.relative_to(root)), "bytes": p.stat().st_size})
        doc["cache_artefacts"] = arts[:400]
        doc["cache_artefact_count"] = len(arts)
    if postgrad_table is not None:
        postgrad_table.write(out / "fx_ids.json")
        doc["postgrad_pieces"] = len(postgrad_table.pieces)
    # inputs trace
    opens = list(_OPENS)
    classes: dict[str, int] = {}
    for o in opens:
        o["class"] = classify(o["path"])
        classes[o["class"]] = classes.get(o["class"], 0) + 1
    forb = [o for o in opens if any(s in o["path"] for s in FORBIDDEN_SUBSTRINGS)]
    with (out / "opens.jsonl").open("w") as f:
        for o in opens:
            f.write(json.dumps(o) + "\n")
    doc["inputs_trace"] = {"opens": len(opens), "by_class": classes, "forbidden_hits": forb,
                           "vllm_source_files": sorted({o["path"].split("site-packages/")[-1] for o in opens if o["class"] == "vllm-source" and o["path"].endswith(".py")}),
                           "model_dir_files": sorted({o["path"].rsplit("/", 1)[-1] for o in opens if o["class"] == "model-dir"}),
                           "repo_files": sorted({o["path"] for o in opens if o["class"] == "repo"})}
    doc["dynamo_calls"] = len(dynamo_capture.STATE["calls"])
    doc["wall_s"] = time.time() - t_start
    (out / "run.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print(json.dumps({"build_s": doc["build_s"], "error": (err or {}).get("msg", "")[:300], "dynamo_calls": doc["dynamo_calls"],
                      "opens": len(opens), "forbidden_hits": len(forb), "out": str(out)}, indent=1))


if __name__ == "__main__":
    main()
