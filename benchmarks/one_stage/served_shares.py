"""A served run's two shares of the card's capacity, from its serve dir: the declared utilization, the work the served
model's steps declare at θ over the wall the engine took to serve them, term by term, and the credited share, the MACs
Pearl+'s PoUW verdict credited at θ over that same wall.

    python benchmarks/one_stage/served_shares.py SERVE_DIR [--credit ANSWER.json] [--model NAME] [--replay S --replay-cpus N]
        prints one JSON object

The wall is the trace's loads' busy seconds (`served_overhead.load_metrics`): each load's span, its first arrival to its
last step's output, plus the on-loop part of the closes of the Commit windows it opened. Each executed step of the trace
(``advice_trace.jsonl``) declares, at the served model's shapes (`MODELS`):

- the linears: every scheduled token through every layer's linears, at θ;
- attention: each row's queries against its causal keys (a row of n tokens from position p is n·p + n(n+1)/2
  query-key pairs a head and layer, QKᵀ and PV head_dim MACs each), at θ/2, the card's bf16 rate;
- the LM head: hidden × vocab MACs a token the step emitted, at θ/2.

θ is the H200's dense int8 (and FP8) rate in MACs at each reading of its clock (`RHO`, the tensor clock's and the boost
clock's), side by side. The declared utilization is the steps' ideal seconds over the wall, and ``terms`` splits it into
the three terms' own shares, which sum to it.

The credited share reads Pearl+'s verdict (``--credit``: ``served_passes.py verify``'s ``answer.json``, core's answer
form): its ``public.work.credited``, the linears' MACs credited at Pearl's Assumption 1 less what the draw leaves at its
δ (in the grid form, `verify_run.grid_work`, where the run is held to its class's fixed load: at most the exact credit),
at θ, over the same wall, and beside it that credit's share of the declared linears. Only the linears are Pearl+'s
work, so attention and the LM head are never credited. A rejected answer credits 0, and so does a run with no
``--credit``. The credited share is not part of the declared share, and the two are never added.

The replay (``--replay S --replay-cpus N``) is the prover's CPU cost beside those GPU shares: before the sampling coin it
replays every tile of its passes, as much work as checking every tile, to give up the rows over a cap
(`volunteer_pass.py`), and took S seconds on N cores. It is stated as the cores it takes to keep pace with the serving
(core-seconds a second of the wall), never as a share of θ. Pearl+'s served piece takes S and N from its
``served_passes.py volunteer`` document (`served_verdicts.pearlplus`), against its timed call's wall.

Pearl+'s own served run (`pearlplus_shares`, from `served_passes.py`'s serve and verify documents, the POUW=pearlplus
piece's) has no step trace: its wall is its timed ``generate`` call's, its declared work that call's passes' linear MACs
alone, and its credited work the verdict's credit over those passes (the verify document's ``timed``), both at θ. Every
pass of that run is written whole for the verdict inside its step; the same load served with no pass retained
(`served_passes.py serve --no-retain`) gives the declared utilization without that write (`pearlplus_unretained`): the
retained run's declared work over the unretained run's wall.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import served_overhead as O  # noqa: E402

#: the H200's clock readings, the tensor clock's (the data sheet's dense int8 and FP8 rate) and the boost clock's, and θ at
#: each: 132 SMs × 4,096 MACs a clock
CLOCKS_MHZ = {"tensor": 1830, "boost": 1980}
RHO = {name: 4096 * 132 * mhz * 1e6 for name, mhz in CLOCKS_MHZ.items()}
#: Llama-3.2-1B's layer: its four linears (name, n, k), merged as vLLM serves them at TP 1
LLAMA32_1B = {"layers": 16, "linears": (("self_attn.qkv_proj", 3072, 2048), ("self_attn.o_proj", 2048, 2048),
                                        ("mlp.gate_up_proj", 16384, 2048), ("mlp.down_proj", 2048, 8192)),
              "heads": 32, "head_dim": 64, "hidden": 2048, "vocab": 128256}
#: the served models' shapes: H1's int8 checkpoint, Pearl+'s FP8 one (`pearl_c_sm90/llama32_1b.py`) and the BF16 one
#: a window tied to Pearl+ serves (served_verdicts.sh's default REPO under the tie), one model
MODELS = {"verity-int8/Llama-3.2-1B-INT8": LLAMA32_1B, "verity-fp8/Llama-3.2-1B-FP8": LLAMA32_1B,
          "unsloth/Llama-3.2-1B": LLAMA32_1B}
TERMS = ("linears", "attention", "lm_head")
SCHEMA = "verity-bench/served-shares/v1"


def step_macs(step: dict[str, Any], model: str) -> dict[str, int]:
    """One step record's declared MACs, ``{linears, attention, lm_head}`` (module docstring)."""
    g = MODELS[model]
    if "rows" not in step:
        raise ValueError(f"step {step.get('step')} has no rows: a trace older than verity-vllm/step-trace/v1's rows")
    tokens = pairs = 0
    for rid, p, n in step["rows"]:
        if p is None:
            raise ValueError(f"step {step.get('step')}: row {rid} has no start position, so its attention is unknown")
        tokens += n
        pairs += n * p + n * (n + 1) // 2
    emitted = sum(int(v) for v in (step.get("emitted") or {}).values())
    return {"linears": tokens * g["layers"] * sum(n * k for _, n, k in g["linears"]),
            "attention": pairs * g["layers"] * g["heads"] * 2 * g["head_dim"], "lm_head": emitted * g["hidden"] * g["vocab"]}


def term_s(macs: dict[str, int], rho: float) -> dict[str, float]:
    """Each term's ideal seconds: the linears at θ, attention and the LM head at θ/2."""
    return {"linears": macs["linears"] / rho, "attention": macs["attention"] / (rho / 2), "lm_head": macs["lm_head"] / (rho / 2)}


def ideal_s(macs: dict[str, int], rho: float) -> float:
    return sum(term_s(macs, rho).values())


def credited_macs(doc: dict[str, Any] | None) -> tuple[int, str | None]:
    """The answer's credited MACs and its verdict (0 and None for no answer; 0 for one that doesn't accept)."""
    if doc is None:
        return 0, None
    verdict = doc.get("verdict")
    if verdict != "accept":
        return 0, verdict
    work = (doc.get("public") or {}).get("work")
    credited = work.get("credited") if isinstance(work, dict) else None
    if not (type(credited) is int and credited >= 0):
        raise ValueError(f"the answer's public.work.credited isn't a count of MACs: {credited!r}")
    return credited, verdict


def model_of(serve_dir: Path) -> str:
    s = json.loads((serve_dir / O.SERVE).read_text())
    return ((s.get("engine") or {}).get("pin") or {}).get("model")


def replay_term(seconds: float, cpus: int, wall: float) -> dict[str, float]:
    """The prover's replay before the coin, `seconds` on `cpus` cores, against `wall` seconds of serving (module docstring)."""
    if not (seconds >= 0 and cpus > 0 and wall > 0):
        raise ValueError(f"a replay is seconds ≥ 0 on cpus > 0 against a wall > 0, not {seconds}, {cpus}, {wall}")
    return {"s": seconds, "cpus": cpus, "core_s": seconds * cpus, "of_wall": seconds / wall,
            "cores_to_keep_pace": seconds * cpus / wall}


def shares(serve_dir: Path, credit: dict[str, Any] | None = None, model: str | None = None,
           replay: tuple[float, int] | None = None) -> dict[str, Any]:
    """The serve dir's two shares at each reading of the clock, and the prover's replay beside them (module docstring)."""
    model = model or model_of(serve_dir)
    if model not in MODELS:
        raise ValueError(f"no declared shapes for {model!r}: add them to MODELS ({sorted(MODELS)})")
    trace = O._jsonl(serve_dir / O.TRACE)
    if not trace or trace[0].get("kind") != "header":
        raise ValueError(f"{serve_dir}: no {O.TRACE} with a header")
    windows = {int(w["window"]): w for w in O._jsonl(serve_dir / O.WINDOWS)}
    loads, total, wall = [], dict.fromkeys(TERMS, 0), 0.0
    for ld in O.loads(trace):
        m = O.load_metrics(ld, windows)
        macs = dict.fromkeys(TERMS, 0)
        for s in ld["steps"]:
            for k, v in step_macs(s, model).items():
                macs[k] += v
        busy = m["span_s"] + m["windows"]["close_on_loop_s"]
        loads.append({"steps": m["steps"], "tokens": m["tokens"], "wall_s": busy, "macs": macs,
                      "declared": {r: ideal_s(macs, rho) / busy for r, rho in RHO.items()}})
        total = {k: total[k] + macs[k] for k in total}
        wall += busy
    if not loads:
        raise ValueError(f"{serve_dir}: no load in its trace")
    work, verdict = credited_macs(credit)
    readings = {}
    for r, rho in RHO.items():
        terms = term_s(total, rho)
        readings[r] = {"clock_mhz": CLOCKS_MHZ[r], "theta_macs_s": rho, "ideal_s": sum(terms.values()),
                       "declared": sum(terms.values()) / wall, "terms": {k: v / wall for k, v in terms.items()},
                       "credited": work / rho / wall}
    return {"schema": SCHEMA, "dir": str(serve_dir), "model": model, "wall_s": wall, "macs": total, "loads": loads,
            "credit": {"verdict": verdict, "macs": work,
                       "of_linears": work / total["linears"] if total["linears"] else None}, "readings": readings,
            "replay": replay_term(*replay, wall) if replay else None}


def pearlplus_shares(served: dict[str, Any], verified: dict[str, Any]) -> dict[str, Any]:
    """Pearl+'s served run's shares at each reading of the clock (module docstring): ``served`` is `served_passes.py
    serve`'s document, ``verified`` its ``verify``'s, whose ``timed`` credit an accepted verdict carries; a rejected one
    credits 0."""
    wall = served["generate_s"]
    if not (isinstance(wall, (int, float)) and wall > 0):
        raise ValueError(f"the served run's timed call has no wall: {wall!r}")
    timed = [p for p in served["passes"] if p.get("timed")]
    declared = sum(p["macs"] or 0 for p in timed)
    answer = verified.get("answer") or {}
    credited = verified["timed"]["credited"] if answer.get("verdict") == "accept" else 0
    work = (answer.get("public") or {}).get("work") or {}
    return {"schema": SCHEMA, "wall_s": wall, "passes": len(timed), "verdict": answer.get("verdict"),
            "credit": {k: work.get(k) for k in ("rho", "epsilon")} | {"delta": verified.get("delta"), "draws": verified.get("draws")},
            "macs": {"declared_linears": declared, "credited": credited},
            "readings": {r: {"clock_mhz": CLOCKS_MHZ[r], "theta_macs_s": rho, "declared_linears": declared / rho / wall,
                             "credited": credited / rho / wall}
                         for r, rho in RHO.items()}}


#: what two of `served_passes.py serve`'s documents agree on when they served one load
LOAD = ("arm", "requests", "prompt_tokens", "output_tokens")


def pearlplus_unretained(shares: dict[str, Any], served: dict[str, Any], unretained: dict[str, Any]) -> dict[str, Any]:
    """The declared utilization of `pearlplus_shares`' run (``shares``, of ``served``) without retention: its declared work
    over the wall of ``unretained``, `served_passes.py serve --no-retain`'s document of the same load, at each reading;
    ValueError when ``unretained`` retained its passes or served another load (`LOAD` and the engine's batched tokens)."""
    if unretained.get("retained") is not False:
        raise ValueError("the unretained run retained its passes")

    def load(d):
        return [d.get(k) for k in LOAD] + [(d.get("engine") or {}).get("max_num_batched_tokens")]
    if load(unretained) != load(served):
        raise ValueError(f"the unretained run served another load: {load(unretained)} against {load(served)}")
    wall = unretained["generate_s"]
    if not (isinstance(wall, (int, float)) and wall > 0):
        raise ValueError(f"the unretained run's timed call has no wall: {wall!r}")
    declared = shares["macs"]["declared_linears"]
    return {"wall_s": wall, "retained_over_unretained": shares["wall_s"] / wall,
            "readings": {r: {"clock_mhz": CLOCKS_MHZ[r], "theta_macs_s": rho, "declared_linears": declared / rho / wall}
                         for r, rho in RHO.items()}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("serve_dir", type=Path)
    ap.add_argument("--credit", type=Path, help="Pearl+'s answer.json (served_passes.py verify's)")
    ap.add_argument("--model", help="the served model (default: serve.json's engine pin)")
    ap.add_argument("--replay", type=float, metavar="S", help="the prover's replay before the coin, in seconds")
    ap.add_argument("--replay-cpus", type=int, metavar="N", help="the cores the replay ran on")
    a = ap.parse_args(argv)
    if (a.replay is None) != (a.replay_cpus is None):
        ap.error("--replay and --replay-cpus go together")
    try:
        got = shares(a.serve_dir, json.loads(a.credit.read_text()) if a.credit else None, a.model,
                     (a.replay, a.replay_cpus) if a.replay is not None else None)
    except (OSError, ValueError) as e:
        print(f"served_shares.py: {e}", file=sys.stderr)
        return 2
    print(json.dumps(got))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
