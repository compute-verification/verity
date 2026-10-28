"""The compiler's verdict on a served run's advice: `predict.steps.compile_steps` (I2) over the advice `verity-vllm advice` wrote,
its request Programs traced by the Build, and the digests compared.

Reads a run of `tools/advice_run.py`: `advice/private/advice.bin` (I1, the prover's), `step_trace.jsonl` (each request's
sampling parameters, which advice v0 doesn't carry: they come with the request's certified I/O; a stand-in, see the TODO in
`main`), `requests.jsonl` (engine id to `user`) and `run.json` (the
model pin, the device's target profile, the execution label and the server's default seed).  The registration is φ of that
model, its config.json the pin's config of record (`dense_generic.pinned_hf_config`, as `prove_path.phi_of_serve` reads it), on
that target at that execution, without workload, and the public default seed (`engine.served_seed.DEFAULT_SEED`): a run whose
server says it filled in another is refused (exit 2, `registered_seed`).  Writes, under `--out`:

    compiled.json     `Compiled.summary()`: the compiler digest, each request's Program digest and shape, each step's row refs
    traces/           one `verity-vllm build --wrapper request` per distinct request Program (`steps.trace_commands`)
    comparison.json   `steps.compare`: each compiled digest against its trace's

and exits 0 iff every compiled Program equals its trace.  `compile_steps` refusing the advice (`Unexpressible`) is written to
compiled.json and exits 2; `--partial` refuses request by request instead, and compiles the rest.

`--window K` reads a `verity-vllm serve-openai --seal` run instead: window K's advice (`advice/w<K>/private/advice.bin`, its
registration's ℓ), the certified body of each of window K's requests from `requests.jsonl` (the records whose ``windows`` hold
K, by `user`: the sampling fields its ingress record says it brought, and its prompt ids), and the model, target,
execution and default seed from `serve.json`.  A window's advice is `window_advice`'s: a request window K carries in from the
one before it runs on at its carried state, and `carry_of` holds that state to the endpoint's records, the advice of the last
window before K that ran steps (`windows.jsonl`), whose close K must open on (`window_advice.follows`); K is refused (exit 2) when
it doesn't, or when it carries a request in and that advice isn't there.  The Build dirs go into `--out` itself (default the
window's workload dir, `windows/w<K>/`, beside its sealed `commit/`), as `build_request_*/`, with `manifest.json`: each request's
Program digest and Build dir, the registration the advice is bound to, and the carry.

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
    """A serve-openai window's (advice file, run meta, the certified body of each of its requests by user, registration): the
    body is the sampling fields the request brought (its ingress record) with its prompt ids. Only the window's own records
    count: a run can serve a user again in another window."""
    adv = run / "advice" / f"w{window}"
    reg = json.loads((adv / "registration.json").read_text())
    serve = json.loads((run / "serve.json").read_text())
    bodies = {}
    for line in (run / "requests.jsonl").read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            if int(window) not in (d.get("windows") or ()):
                continue
            user = d.get("user") or d["http_id"]
            if user in bodies:
                raise ValueError(f"requests.jsonl: user {user!r} has two requests in window {window}")
            bodies[user] = {**((d.get("ingress") or {}).get("brought") or {}), "prompt": list(d["prompt_ids"])}
    meta = {"model": serve["model"], "revision": serve["revision"], "target": serve["target"], "execution": serve.get("execution"),
            "default_seed": serve.get("default_seed")}
    return adv / "private" / "advice.bin", meta, bodies, reg


def registered_seed(meta: dict) -> int:
    """The default seed the registration fixes, `engine.served_seed.DEFAULT_SEED`; a ValueError when the run's server says it
    filled in another (or says none), since a request that sent no seed would be compiled under a seed its draw didn't use."""
    from verity_vllm.engine.served_seed import DEFAULT_SEED

    seed = meta.get("default_seed")
    if type(seed) is not int or seed != DEFAULT_SEED:
        raise ValueError(f"the run's server filled in default seed {seed!r}, and the registered one is "
                         f"engine.served_seed.DEFAULT_SEED ({DEFAULT_SEED}): serve it under that rule")
    return DEFAULT_SEED


def carry_of(run: Path, window: int, data: bytes) -> dict:
    """Window K's carry (``data`` its advice), held to the advice of the last window before K that ran steps (``windows.jsonl``'s
    last non-empty one): K must open where that window closed (`window_advice.follows`). The requests K carries in and the ones it
    leaves with outputs to keep, by id, and the window it follows; a ValueError when K doesn't follow it, or carries a request in
    with no advice before it to carry it from."""
    from verity_vllm.pipeline import window_advice as WA

    a = WA.decode(data)
    lines = (run / "windows.jsonl").read_text().splitlines() if (run / "windows.jsonl").exists() else []
    before = [int(d["window"]) for d in map(json.loads, filter(str.strip, lines)) if not d.get("empty") and int(d["window"]) < window]
    last = max(before, default=None)
    path = None if last is None else run / "advice" / f"w{last}" / "private" / "advice.bin"
    carry = {"in": [q.id for q in a.requests[:len(a.carried)]], "out": [a.requests[c.request].id for c in WA.close(a) if c.must],
             "follows": None}
    if path is not None and path.exists():
        try:
            WA.follows(WA.decode(path.read_bytes()), a)
        except WA.MalformedAdvice as e:
            raise ValueError(f"window {window} doesn't open where window {last} closed: {e}") from None
        carry["follows"] = last
    elif a.carried:
        raise ValueError(f"window {window} carries {len(a.carried)} requests in, {carry['in'][0]!r} first, and "
                         + (f"window {last}, the one before it, has no advice at {path}" if path else "no window ran before it"))
    return carry


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

    from verity_vllm.engine import step_trace as ST
    from verity_vllm.engine.profiles.dense_generic import pinned_hf_config
    from verity_vllm.pipeline import window_advice as WA
    from verity_vllm.program.predict.build import Unexpressible
    from verity_vllm.program.predict.phi import Phi
    from verity_vllm.program.predict.steps import (
        CAP_BITS,
        Registered,
        compare,
        compile_steps,
        trace_commands,
    )

    reg_doc, carry = None, None
    if a.window is not None:
        path, meta, bodies, reg_doc = served_inputs(run, a.window)
        data = Path(a.advice or path).read_bytes()
        try:
            carry = carry_of(run, a.window, data)
        except ValueError as e:
            (out / "compiled.json").write_text(json.dumps({"refused": str(e)}, indent=1) + "\n")
            print(f"[advice-compile] the advice's carry is refused: {e}", flush=True)
            return 2
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
    advice = WA.decode(data)
    ids = {q.id for q in advice.requests}
    ingress = {k: v for k, v in bodies.items() if k in ids}
    config = pinned_hf_config(meta["model"], meta["revision"])
    if config is None:
        raise SystemExit(f"[advice-compile] {meta['model']}@{meta['revision']} is pinned in no checkpoint manifest")
    hf_config = json.loads(config.read_text())
    phi = Phi(model={"repo": meta["model"], "revision": meta["revision"], "hf_config": hf_config}, target=dict(meta["target"]),
              execution=meta.get("execution") or "eager")
    cap = int(((reg_doc or {}).get("compiler") or {}).get("cap_bits") or CAP_BITS)
    try:
        seed = registered_seed(meta)
    except ValueError as e:
        (out / "compiled.json").write_text(json.dumps({"refused": str(e)}, indent=1) + "\n")
        print(f"[advice-compile] the run's default seed is refused: {e}", flush=True)
        return 2
    reg = Registered.of(phi, default_seed=seed, cap_bits=cap)
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
               "carry": {**carry, "idle": [advice.requests[i].id for i in summary["carry"]["idle"]]},
               "comparison": {k: cmp[k] for k in ("equal", "different", "no_trace", "mismatches") if k in cmp}}
        (out / "manifest.json").write_text(json.dumps(man, indent=1) + "\n")
        print(f"[advice-compile] {out / 'manifest.json'}: {len(man['requests'])} request Programs", flush=True)
    return 0 if cmp["mismatches"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
