"""CB-B probe: does the real pinned vLLM `LlamaForCausalLM` forward run to completion on `meta` under the structural
mode, with every op intercepted?  Prints the op histogram / first refusal; dumps the trace and an open()-audit.

    PYTHONPATH=. /workspace/venv-cu129/bin/python -m verity_capture.experimental.cb_b.probe \
        --model /workspace/hf/hub/models--HuggingFaceTB--SmolLM2-135M/snapshots/93efa2f0... --out /workspace/cp/cbb \
        --decode 8 --prefill 8
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback

FORBIDDEN_SUBSTR = ("program.json", "instances.jsonl", "log.jsonl", "experimental/aot/", "/census/", "archive-20260917")


class OpenAudit:
    """`sys.addaudithook` trace of every file opened during construction (record-freeness evidence)."""

    def __init__(self, own_out: str | None = None) -> None:
        self.opens: list[str] = []
        self.forbidden: list[str] = []
        self.active = False
        self.own_out = os.path.abspath(own_out) if own_out else None    # the builder's own output dir (its writes are not inputs)

    def __call__(self, event: str, args: tuple) -> None:
        if not self.active or event != "open":
            return
        p = args[0]
        if not isinstance(p, str):
            p = str(p)
        mode = args[1] if len(args) > 1 and isinstance(args[1], str) else "r"
        self.opens.append(f"{p}\t{mode}")
        if any(s in p for s in FORBIDDEN_SUBSTR):
            if self.own_out and os.path.abspath(p).startswith(self.own_out + os.sep) and ("w" in mode or "a" in mode or "x" in mode):
                return                                                   # our own program.json being written
            self.forbidden.append(f"{p}\t{mode}")


def git_rev() -> str:
    """Shipped revision: EXPORT.json on a pod (no .git there), else git."""
    for p in ("EXPORT.json", os.path.join(os.path.dirname(__file__), "..", "..", "..", "EXPORT.json")):
        try:
            with open(p) as f:
                d = json.load(f)
            return f"{d['commit']} dirty={d.get('dirty_patch_sha256', '')[:12]}"
        except Exception:  # noqa: BLE001
            pass
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--decode", type=int, default=8, help="decode step at KV length L (query row = position L)")
    ap.add_argument("--prefill", type=int, default=0, help="prefill of N tokens (0 = skip)")
    ap.add_argument("--no-sampler", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    audit = OpenAudit()
    sys.addaudithook(audit)
    audit.active = True

    from . import fake_exec as fx

    fx.apply_env()
    import torch

    t0 = time.perf_counter()
    report: dict = {"rev": git_rev(), "argv": sys.argv, "torch": torch.__version__, "steps": {}}
    try:
        import vllm

        report["vllm"] = vllm.__version__
        cfg = fx.build_vllm_config(a.model)
        report["config_s"] = time.perf_counter() - t0
        fx.init_distributed(cfg)
        t1 = time.perf_counter()
        model = fx.instantiate_model(cfg)
        report["instantiate_s"] = time.perf_counter() - t1
        report["model_class"] = type(model).__name__
        report["n_params"] = sum(1 for _ in model.named_parameters())
        report["param_devices"] = sorted({str(p.device) for _, p in model.named_parameters()})
    except Exception as e:  # noqa: BLE001
        report["refusal"] = {"phase": "instantiate", "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-4000:]}
        print(json.dumps(report, indent=1, default=str))
        with open(os.path.join(a.out, "probe_report.json"), "w") as f:
            json.dump(report, f, indent=1, default=str)
        return 2

    plan = []
    if a.decode:
        plan.append(("decode", fx.StepSpec(pos0=a.decode, M=1)))
    if a.prefill:
        plan.append(("prefill", fx.StepSpec(pos0=0, M=a.prefill)))
    rc = 0
    for label, spec in plan:
        tr = fx.Trace()
        st: dict = {"spec": {"pos0": spec.pos0, "M": spec.M}}
        t2 = time.perf_counter()
        try:
            with fx.CudaGuard() as g:
                hidden, logits, sampled = fx.run_step(tr, model, cfg, spec, sampler=not a.no_sampler)
            st["ok"] = True
            st["cuda_kernel_events"] = getattr(g, "kernels", -1)
            st["hidden"] = list(hidden.shape) if hidden is not None else None
            st["logits"] = list(logits.shape) if logits is not None else None
            st["sampled"] = list(sampled.shape) if sampled is not None else None
        except Exception as e:  # noqa: BLE001
            rc = 1
            st["ok"] = False
            last = None
            for ev in reversed(tr.events):
                if isinstance(ev, fx.OpCall) and ev.executed == "error":
                    last = ev
                    break
            st["refusal"] = {"error": f"{type(e).__name__}: {str(e)[:600]}", "op": last.op if last else None,
                             "module": last.module if last else tr.module(), "traceback": traceback.format_exc()[-2500:]}
        st["wall_s"] = time.perf_counter() - t2
        st["n_events"] = len(tr.events)
        st["op_hist"] = tr.op_hist()
        st["rules_used"] = tr.rules_used
        st["executed_hist"] = {}
        for ev in tr.events:
            if isinstance(ev, fx.OpCall):
                st["executed_hist"][ev.executed] = st["executed_hist"].get(ev.executed, 0) + 1
        tr.dump(os.path.join(a.out, f"trace_{label}_{spec.pos0}_{spec.M}.json"))
        report["steps"][label] = st
        print(f"[cb-b] {label} pos0={spec.pos0} M={spec.M}: ok={st['ok']} events={len(tr.events)} wall={st['wall_s']:.2f}s", file=sys.stderr)
        if not st["ok"]:
            print(json.dumps(st["refusal"], indent=1)[:3000], file=sys.stderr)

    audit.active = False
    report["opens_n"] = len(audit.opens)
    report["opens_forbidden"] = audit.forbidden
    with open(os.path.join(a.out, "opens.txt"), "w") as f:
        f.write("\n".join(audit.opens))
    with open(os.path.join(a.out, "probe_report.json"), "w") as f:
        json.dump(report, f, indent=1, default=str)
    print(json.dumps({k: v for k, v in report.items() if k != "steps"}, indent=1, default=str))
    for label, st in report["steps"].items():
        print(label, json.dumps({k: v for k, v in st.items() if k not in ("op_hist",)}, indent=1, default=str)[:2500])
        print(label, "op_hist", json.dumps(st["op_hist"], indent=0)[:2500])
    return rc


if __name__ == "__main__":
    sys.exit(main())
