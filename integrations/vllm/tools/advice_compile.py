"""The compiler's verdict on a served run's advice: `predict.steps.compile_steps` (I2) over the advice `verity-vllm advice` wrote,
its request Programs traced by the Build, and the digests compared.

Reads a run of `tools/advice_run.py`: `advice/private/advice.bin` (I1, the prover's), `step_trace.jsonl` (each request's
sampling parameters, which advice v0 doesn't carry: they come with the request's certified I/O; a stand-in, see the TODO in
`main`), `requests.jsonl` (engine id to `user`) and `run.json` (the
model pin, the device's target profile, the execution label and the server's default seed).  The registration is φ of that
model on that target at that execution, without workload.  Writes, under `--out`:

    compiled.json     `Compiled.summary()`: the compiler digest, each request's Program digest and shape, each step's row refs
    traces/           one `verity-vllm build --wrapper request` per distinct request Program (`steps.trace_commands`)
    comparison.json   `steps.compare`: each compiled digest against its trace's

and exits 0 iff every compiled Program equals its trace.  `compile_steps` refusing the advice (`Unexpressible`) is written to
compiled.json and exits 2; `--partial` refuses request by request instead, and compiles the rest.

`--window K` reads a `verity-vllm serve-openai --seal` run instead: window K's advice (`advice/w<K>/private/advice.bin`, its
registration's ℓ), each request's certified body from `requests.jsonl` (by `user`: the sampling fields its ingress record says
it brought, and its prompt ids), and the model, target,
execution and default seed from `serve.json`.  The Build dirs go into `--out` itself (default the window's row dir,
`windows/w<K>/`, beside its sealed `commit/`), as `build_request_*/`, with `manifest.json`: each request's Program digest and
Build dir, and the registration the advice is bound to.

    cd integrations/vllm; PYTHONPATH=.:../.. /workspace/jobs/venv312/bin/python tools/advice_compile.py RUN --out DIR [--partial]
    ... tools/advice_compile.py RUN --window K [--out DIR] [--partial]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

MANIFEST = "verity-vllm/served-window-programs/v1"


def served_inputs(run: Path, window: int) -> tuple[Path, dict, dict, dict]:
    """A serve-openai window's (advice file, run meta, each request's certified body by user, registration): the body is the
    sampling fields the request brought (its ingress record) with its prompt ids."""
    adv = run / "advice" / f"w{window}"
    reg = json.loads((adv / "registration.json").read_text())
    serve = json.loads((run / "serve.json").read_text())
    bodies = {}
    for line in (run / "requests.jsonl").read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            bodies[d.get("user") or d["http_id"]] = {**((d.get("ingress") or {}).get("brought") or {}), "prompt": list(d["prompt_ids"])}
    meta = {"model": serve["model"], "revision": serve["revision"], "target": serve["target"], "execution": serve.get("execution"),
            "default_seed": serve.get("default_seed")}
    return adv / "private" / "advice.bin", meta, bodies, reg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run")
    ap.add_argument("--out", default=None, help="where compiled.json and the traces go (default: --window's row dir)")
    ap.add_argument("--window", type=int, default=None, help="a serve-openai --seal run's Commit window")
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--advice", default=None, help="the advice (default RUN/advice/private/advice.bin)")
    ap.add_argument("--partial", action="store_true",
                    help="refuse request by request (a measurement over served traffic) rather than the whole advice")
    a = ap.parse_args()
    run = Path(a.run)
    if a.out is None and a.window is None:
        ap.error("--out, or --window K (whose row dir is the default)")
    out = Path(a.out) if a.out else run / "windows" / f"w{a.window}"
    out.mkdir(parents=True, exist_ok=True)

    from verity_vllm.config import checkpoint_entries
    from verity_vllm.engine import step_trace as ST
    from verity_vllm.program.predict.build import Unexpressible
    from verity_vllm.program.predict.phi import Phi
    from verity_vllm.program.predict.steps import (
        CAP_BITS,
        Registered,
        compare,
        compile_steps,
        trace_commands,
    )

    from verity.core.protocols.certifier import advice as I1

    reg_doc = None
    if a.window is not None:
        path, meta, bodies, reg_doc = served_inputs(run, a.window)
        data = Path(a.advice or path).read_bytes()
    else:
        data = Path(a.advice or run / "advice" / "private" / "advice.bin").read_bytes()
        meta = json.loads((run / "run.json").read_text())
        recs = [d for d in map(json.loads, (run / "requests.jsonl").read_text().splitlines()) if d]
        user = {d["request"]: d["user"] for d in recs}
        prompt = {d["user"]: d.get("prompt_ids") or d.get("prompt") for d in recs}
        # TODO: read each request's body from the Commit's ingress record once advice_run records it; the step trace's arrive
        # records stand in for its sampling
        bodies = {user[r["request"]]: {**r["sampling"], "prompt": prompt[user[r["request"]]]}
                  for r in ST.load(run / "step_trace.jsonl") if r["kind"] == "arrive"}
    ids = {q.id for q in I1.decode(data).requests}
    ingress = {k: v for k, v in bodies.items() if k in ids}
    entry = next(c for c in checkpoint_entries() if c["repo"] == meta["model"] and c["revision"] == meta["revision"])
    hf_config = json.loads((Path(entry["local_path"]) / "config.json").read_text())
    phi = Phi(model={"repo": meta["model"], "revision": meta["revision"], "hf_config": hf_config}, target=dict(meta["target"]),
              execution=meta.get("execution") or "eager")
    cap = int(((reg_doc or {}).get("compiler") or {}).get("cap_bits") or CAP_BITS)
    reg = Registered.of(phi, default_seed=int(meta.get("default_seed") or 0), cap_bits=cap)
    try:
        compiled = compile_steps(reg, data, ingress, partial=a.partial)
    except Unexpressible as e:
        (out / "compiled.json").write_text(json.dumps({"refused": str(e)}, indent=1) + "\n")
        print(f"[advice-compile] compile_steps refuses the advice: {e}", flush=True)
        return 2
    summary = compiled.summary()
    (out / "compiled.json").write_text(json.dumps(summary, indent=1) + "\n")
    traces = out if a.window is not None else out / "traces"
    cmds = trace_commands(summary, reg, str(traces), max_model_len=a.max_model_len)
    print(f"[advice-compile] {len(summary['programs'])} request Programs over {sum(u is not None for u in summary['requests'])} "
          f"requests ({len(summary.get('refused') or {})} refused); tracing {len(cmds)}", flush=True)
    for i, why in (summary.get("refused") or {}).items():
        print(f"[advice-compile] refused {i}: {why}", flush=True)
    for cmd in cmds:
        rc = subprocess.run([sys.executable, "-m", "verity_vllm.pipeline.cli", *cmd[1:]]).returncode
        print(f"[advice-compile] {cmd[cmd.index('--out') + 1]}: rc={rc}", flush=True)
    cmp = compare(summary, str(traces))
    (out / "comparison.json").write_text(json.dumps(cmp, indent=1) + "\n")
    print(f"[advice-compile] equal {cmp['equal']}, different {cmp['different']}, no trace {cmp['no_trace']}", flush=True)
    if a.window is not None:
        from verity_vllm.program.predict.steps import trace_dir
        man = {"schema": MANIFEST, "run": str(run), "window": a.window, "registration": reg_doc, "cap_bits": cap,
               "compiler": summary["compiler"], "workload_digest": summary["workload_digest"],
               "requests": {u["id"]: {"digest": u["digest"], "build_dir": trace_dir(u), "LP": u["LP"], "T": u["T"], "first_step": u["first_step"]}
                            for u in summary["requests"] if u is not None},
               "component_digests": sorted(summary["programs"]), "refused": summary.get("refused") or {},
               "comparison": {k: cmp[k] for k in ("equal", "different", "no_trace", "mismatches") if k in cmp}}
        (out / "manifest.json").write_text(json.dumps(man, indent=1) + "\n")
        print(f"[advice-compile] {out / 'manifest.json'}: {len(man['requests'])} request Programs", flush=True)
    return 0 if cmp["mismatches"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
