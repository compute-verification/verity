"""`verity-vllm pouw-serve`: a workload served under ``pouw:ncp-v2``, and its calls replayed as committed and with the
first call's tiles tampered.

    verity-vllm pouw-serve --workload workloads/workload_qwen05_32x16_1req.json [--draws 16] [--out DIR]
        [--challenger HOST:PORT --session '{"session": ..., "seq": ..., "root": ..., "commitment": ..., "secret": ...}']

The engine is `verity_vllm.LLM` on the workload's pinned model, with ``pouw:ncp-v2`` installed. Every linear call is
committed natively as it is served (`protocol_options.pouw_native`), under an epoch whose salt is the salt coin of the
auditor host's challenger at ``--challenger``, in the session the gate opened for the window (``--session``, the
gate's answer, `protocol_options.pouw_h1_challenger`), opened after the weights' root. Without ``--challenger`` the salt
is the command's own challenger's (`verity.core.protocols.pouw.coins.Challenger`, epoch 0, its coins the Lean verifier of
record's, `pouw.verifier.coins`): the command is then its run's auditor, and the run is a measurement, not a window any
verifier credits. Once the engine closes, the circuit replay (`check.replay.pouw_circuit.replay`) runs twice, the run's
calls as one window under the work law at a budget of K = ``--draws``:

- **honest**: the calls as committed;
- **tampered**: the same calls, except that credited word 7 of every tile of the first call is flipped by its prover,
  committed and opened that way (`check.replay.pouw_circuit.tamper_tiles`), with the calls root recomputed over it.

Each replay's window draws under a fresh source (`verity.core.protocols.pouw.circuit.anchors.AdmittedCalls.open`), which its report records.

The command writes ``pouw_serve.json`` under ``--out`` (default ``$RESEARCH_RUN_DIR``), and ``result.json`` beside it
under a research run (``$RESEARCH_RUN_ID``). They hold
the executor's wall time for each forward, the commitment, both replay reports and the verdict. It prints
``POUW-SERVE honest=<accepted|rejected> tampered=<accepted|rejected>``, and exits 0 when the honest calls are accepted
and the tampered ones rejected.

This is not a Commit of record: the model's own rows are not replayed (there is no Build Program here), which
`verity-vllm row` does.
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from verity_vllm.config import RowEngine, option, workload_arg

ENTRY = "pouw:ncp-v2"


@dataclass(frozen=True, kw_only=True)
class Options:
    workload: str = option("--workload", type=workload_arg, help="a row file (config.RowConfig): the pinned model, its engine.revision and the requests")
    draws: int = option("--draws", type=int, default=16, help="the window's work-law budget K (default 16)")
    out: str = option("--out", env="RESEARCH_RUN_DIR", default=".", help="the output directory (default $RESEARCH_RUN_DIR, else .)")
    run_id: str = option("--run-id", env="RESEARCH_RUN_ID", default="", help="the research run's id: result.json is written when set")
    challenger: str | None = option("--challenger", default=None, metavar="HOST:PORT", help="the auditor host's challenger the epoch's salt coin comes from (the pouw setting 'challenger'); without it the command's own in-process challenger draws it, and the run is a measurement only")
    session: str | None = option("--session", default=None, metavar="JSON", help="with --challenger: the challenger gate's open answer for the window, {\"session\", \"seq\", \"root\", \"commitment\"} and its \"secret\" (the pouw setting 'session')")


def protocols_config(a: Options) -> dict[str, Any]:
    """PoUW's settings: the salt from the auditor host's challenger in the session the gate opened (``--challenger``,
    ``--session``), else from the command's own in-process challenger."""
    if a.challenger is None:
        if a.session is not None:
            raise ValueError("--session is the session of the challenger at --challenger")
        from verity.core.protocols.pouw import verifier
        from verity.core.protocols.pouw.coins import Challenger
        return {"pouw": {"coins": Challenger(0, verifier.coins), "draws": a.draws}}
    if a.session is None:
        raise ValueError("--challenger needs --session, the challenger gate's open answer for the window's Commit")
    return {"pouw": {"challenger": a.challenger, "session": json.loads(a.session), "draws": a.draws}}


def verdicts(records: Sequence[Any], commitment: dict[str, Any], draws: int) -> dict[str, Any]:
    """Both replays and the verdict."""
    from verity_vllm.check.replay import pouw_circuit as RC
    from verity_vllm.protocols.pouw import circuit as PC

    honest = RC.replay(records, commitment, draws)
    bad, where = RC.tamper_tiles(records)
    salt, wroot = bytes.fromhex(commitment["salt"]), bytes.fromhex(commitment["weights_root"])
    t = RC.replay(bad, dict(commitment, calls_root=PC.calls_root(salt, wroot, bad).hex()), draws)
    return {"honest": honest, "tampered": dict(t, tampered=where),
            "verdict": {"honest": "accepted" if honest["result"] else "rejected", "tampered": "accepted" if t["result"] else "rejected"}}


def per_forward(records: Sequence[Any], seconds: Sequence[float]) -> list[dict[str, Any]]:
    """The executor's wall time per forward (every linear call of it), with its calls and its padded rows."""
    by: dict[int, dict[str, Any]] = defaultdict(lambda: {"calls": 0, "rows": 0, "seconds": 0.0})
    for r, s in zip(records, seconds):
        f = by[r.call.forward]
        f["calls"] += 1
        f["rows"] = max(f["rows"], r.shape.m)
        f["seconds"] += s
    return [{"forward": k, **v, "seconds": round(v["seconds"], 3)} for k, v in sorted(by.items())]


def progress(run: Any, stop: threading.Event, every: float = 30.0) -> None:
    """One line every ``every`` seconds while the engine serves: the calls and forwards committed and the executor's time."""
    t0 = time.monotonic()
    while not stop.wait(every):
        recs = list(run.records)
        print(f"[pouw-serve] {time.monotonic() - t0:.0f}s: {len(recs)} calls, {recs[-1].call.forward + 1 if recs else 0} forwards, "
              f"executor {sum(run.seconds):.1f}s", flush=True)


def main(a: Options) -> int:
    from vllm import SamplingParams

    from verity_vllm.llm import LLM
    from verity_vllm.protocols import pouw

    wl = json.loads(Path(a.workload).read_text())
    rev = RowEngine.of(wl).revision
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    llm = LLM(wl["model"], revision=rev, protocols=ENTRY, protocols_config=protocols_config(a))
    built = time.monotonic() - t0
    prompts = [{"prompt_token_ids": r["prompt_token_ids"]} for r in wl["requests"]]
    params = [SamplingParams(temperature=0.0, max_tokens=int(r["max_tokens"])) for r in wl["requests"]]
    run, _sites = pouw.circuit_run(ENTRY)
    stop = threading.Event()
    threading.Thread(target=progress, args=(run, stop), daemon=True).start()
    t1 = time.monotonic()
    outs = llm.generate(prompts, params, use_tqdm=False)
    served = time.monotonic() - t1
    stop.set()
    llm.close()
    records, commitment = list(run.records), dict(run.commits)
    pf = per_forward(records, run.seconds)
    print(f"served {len(pf)} forwards, {len(records)} calls, {sum(r.units for r in records)} units in {served:.1f}s "
          f"(engine {built:.1f}s); calls root {commitment['calls_root']}", flush=True)
    t2 = time.monotonic()
    v = verdicts(records, commitment, a.draws)
    replayed = time.monotonic() - t2
    print(f"replayed twice in {replayed:.1f}s: honest {v['honest']['seconds']}s, tampered {v['tampered']['seconds']}s", flush=True)
    decode = [f["seconds"] for f in pf[1:]]
    doc = {"schema": "verity-vllm/pouw-serve/v1", "workload": a.workload, "model": wl["model"], "revision": rev, "draws": a.draws,
           "engine_build_s": round(built, 3), "serve_s": round(served, 3), "replay_s": round(replayed, 3),
           "forwards": len(pf), "per_forward": pf,
           "forward_s": {"prefill": pf[0]["seconds"] if pf else None, "decode_mean": round(sum(decode) / len(decode), 3) if decode else None,
                         "mean": round(sum(f["seconds"] for f in pf) / len(pf), 3) if pf else None, "serve_mean": round(served / max(1, len(pf)), 3)},
           "calls": len(records), "units": sum(r.units for r in records), "commitment": commitment, "details": run.details,
           "tokens": [[list(c.token_ids) for c in o.outputs] for o in outs], **v}
    (out / "pouw_serve.json").write_text(json.dumps(doc, indent=1) + "\n")
    ok = v["verdict"] == {"honest": "accepted", "tampered": "rejected"}
    line = f"POUW-SERVE honest={v['verdict']['honest']} tampered={v['verdict']['tampered']}"
    result = {"schema": "research/result/v0.1", "run_id": a.run_id,
              "workload_fingerprint": {"workload": a.workload, "model": wl["model"], "revision": rev, "draws": a.draws, "entry": ENTRY},
              "validation": {"status": "passed" if ok else "failed", "checker": "pouw-serve", "detail": line, "evidence": ["pouw_serve.json"]},
              "measurements": [{"name": "forward_executor_mean", "value": doc["forward_s"]["mean"], "unit": "s"},
                               {"name": "serve", "value": doc["serve_s"], "unit": "s"},
                               {"name": "replay_both", "value": doc["replay_s"], "unit": "s"}],
              "measurement_files": ["pouw_serve.json"], "artifacts": ["pouw_serve.json"]}
    if a.run_id:
        tmp = out / "result.json.tmp"
        tmp.write_text(json.dumps(result, indent=1) + "\n")
        os.replace(tmp, out / "result.json")
    print(line, flush=True)
    return 0 if ok else 1


__all__ = ["Options", "main", "per_forward", "protocols_config", "verdicts"]
