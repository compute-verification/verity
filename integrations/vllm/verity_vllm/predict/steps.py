"""I2, the compiler: a served run's advice -> the Programs its committed rows must pass (v0, request-major).

`compile_steps(registered, advice)` reads the registration's static part (`Registered`: φ without workload or sampling, and the
server's public default seed) and the advice (I1, `verity.core.protocols.certifier.advice`: what serving decided that nothing
registered predicts), and nothing else of the run.

- A request's Program is the Build's request wrapper at the shape the advice ran it at, `build.build_program(φ_r, "request", LP,
  T)`, under its own sampling policy (`policy_of`): LP its prompt length, T its live decode rows, the ones whose sample it kept
  (I1's `output_len` − 1).  The one row async scheduling runs past a request's last kept output samples nothing kept: it is
  `discarded`, in no Program.  Advice without `output_len` keeps every decode row.  Requests of one (LP, T, policy) share one
  Program.
- On a batch-invariant target a request's Program does not read its batch, so the advice's steps only place rows: each live row
  is a `RowRef` into its request's Program (positions start..start+n-1; `samples` the tokens the row samples), in I1's row order.
- GP-01 (`program.workload.WorkloadProgram`) composes the request Programs that ran, each at its first scheduled step and with
  cap T + 1 (the executed prefix), under the target's lag.

v0 compiles what the Build traces, and refuses the rest by name (`Unexpressible`), never approximated:
- a target without batch invariance, whose launches would come from the advice (`launch=`);
- a prompt not prefilled whole in one row from position 0: a prefix-cache hit (the cached request wrapper is not ported) or a
  chunked prefill (no chunked request wrapper on either side);
- a preemption, or decode rows that are not one position a step in consecutive steps;
- sampling other than greedy or seeded top-p with top_k 0: min_p, the penalties, logit_bias, min_tokens, n > 1, or a parameter
  this compiler does not know.

A refusal refuses the session.  `partial=True` (a measurement over served traffic, never a registration) refuses request by
request instead: a refused request's rows and Program leave the compilation, and `refused` names why.

The compiler digest is `score.code_version()`.  `trace_commands` lists the Build traces (`verity-vllm build --wrapper request`) of a
compilation's Programs, and `compare` holds the compiled digests to them.
"""
from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from .build import Unexpressible, build_program
from .phi import Phi

SCHEMA = "verity-vllm/compiled-steps/v1"
#: vLLM's `_SAMPLING_EPS`: a temperature below it samples greedily; the Build's policy is greedy only at 0
SAMPLING_EPS = 1e-5
#: the sampling parameters (`engine.step_trace.SAMPLING`) that leave a request's Program unchanged at these values
_INERT = {"min_p": (None, 0, 0.0), "presence_penalty": (None, 0, 0.0), "frequency_penalty": (None, 0, 0.0),
          "repetition_penalty": (None, 1, 1.0), "min_tokens": (None, 0), "logit_bias": (None, {}), "n": (None, 1),
          "top_k": (None, 0, -1)}
#: the parameters that only finish a request (its T, which the advice's rows carry), and the ones the policy reads
_FINISH = ("max_tokens", "stop", "stop_token_ids", "ignore_eos")
_READ = ("temperature", "top_p", "seed")


@dataclass(frozen=True)
class Registered:
    """The registration's static part: φ without workload or sampling (the model, target, TP, engine, execution and profile), and
    the default seed the server samples a request with when its client sends none."""

    phi: Phi
    default_seed: int

    def __post_init__(self) -> None:
        if self.phi.requests:
            raise ValueError("a registered φ carries no workload: the advice brings the requests")

    @classmethod
    def of(cls, phi: Phi, default_seed: int) -> Registered:
        return cls(Phi.from_json({**phi.to_json(), "workload": {}, "sampling": {"kind": "greedy"}}), int(default_seed))


@dataclass(frozen=True)
class RowRef:
    """One advice row: request `request`'s positions start..start+n-1 in its Program `digest`, sampling `samples` tokens."""

    request: int
    digest: str
    start: int
    n: int
    samples: int


@dataclass(frozen=True)
class RequestUnit:
    """A request that ran: its Program's shape and digest, its policy, the seed its `seed` Input takes, and its first step."""

    request: int
    digest: str
    LP: int
    T: int
    policy: dict
    seed: int | None
    first_step: int

    def key(self) -> str:
        return json.dumps([self.LP, self.T, self.policy], sort_keys=True)


@dataclass
class Compiled:
    programs: dict[str, Any]              # request Program digest -> Program
    requests: list[RequestUnit | None]    # by advice request index; None for a request with no row, or refused
    workload: Any                         # GP-01's WorkloadProgram over the requests that ran
    steps: list[list[RowRef]]             # per advice step, its live rows in I1's order
    discarded: list[tuple[int, int, int, int]] = field(default_factory=list)  # (step, request, start, n) of rows no Program holds
    refused: dict[int, str] = field(default_factory=dict)                     # request -> why (partial=True only)

    def summary(self) -> dict:
        from .score import code_version
        return {"schema": SCHEMA, "compiler": code_version(), "workload_digest": self.workload.digest, "lag": self.workload.lag,
                "programs": sorted(self.programs),
                "requests": [None if u is None else {"request": u.request, "digest": u.digest, "LP": u.LP, "T": u.T, "policy": u.policy,
                                                     "seed": u.seed, "first_step": u.first_step} for u in self.requests],
                "steps": [[[r.request, r.start, r.n, r.samples] for r in rows] for rows in self.steps],
                "discarded": [list(d) for d in self.discarded], "refused": {str(i): why for i, why in sorted(self.refused.items())}}


def _get(obj: Any, name: str, default: Any = None) -> Any:
    return obj.get(name, default) if isinstance(obj, Mapping) else getattr(obj, name, default)


def policy_of(i: int, sampling: Mapping | None, default_seed: int) -> tuple[dict, int | None]:
    """Request `i`'s sampling policy as φ.sampling takes it, and its seed Input's value (the client's, else `default_seed`; None when
    greedy), from the parameters it carried (`SamplingParams` names), refused where its Program would not be the Build's."""
    if sampling is None or _get(sampling, "temperature") is None:
        raise Unexpressible(f"request {i}: no temperature among its sampling parameters (the policy cannot be read)")
    s = dict(sampling) if isinstance(sampling, Mapping) else {f.name: getattr(sampling, f.name) for f in fields(sampling)}
    facts = [f"{k}={v!r}" for k, v in sorted(s.items()) if k in _INERT and v not in _INERT[k]]
    facts += [f"{k} (a parameter this compiler does not know)" for k in sorted(s) if s[k] is not None and k not in (*_INERT, *_FINISH, *_READ)]
    t, p = float(s["temperature"]), float(s.get("top_p") if s.get("top_p") is not None else 1.0)
    if 0 < t < SAMPLING_EPS:
        facts.append(f"temperature {t!r} (vLLM samples it greedily, the Build's policy is greedy at 0 only)")
    if facts:
        raise Unexpressible(f"request {i}: sampling the vocabulary cannot express: {'; '.join(facts)}")
    if t == 0:
        return {"kind": "greedy"}, None
    seed = s.get("seed")
    return ({"kind": "stochastic", "temperature": t, "top_p": p, "top_k": 0, "seeded": True, "splits": "derived"},
            int(default_seed if seed is None else seed))


def _rows_of(advice: Any) -> tuple[dict[int, list[tuple[int, Any]]], dict[int, str]]:
    """Each request's (step, row) pairs in step order, and why a preemption refuses a request; a step that names a request twice or
    one the advice lacks is refused."""
    n = len(advice.requests)
    rows: dict[int, list[tuple[int, Any]]] = {i: [] for i in range(n)}
    preempted: dict[int, str] = {}
    why = "v0 runs each request from its prefill without preemption"
    for s, step in enumerate(advice.steps):
        if pre := sorted(_get(step, "preempted", None) or ()):
            for r in pre:
                preempted.setdefault(int(r), f"step {s}: preempted requests {pre} ({why})")
        seen = set()
        for row in _get(step, "rows"):
            r = int(_get(row, "request"))
            if not 0 <= r < n or r in seen:
                raise Unexpressible(f"step {s}: a row of request {r} {'twice' if r in seen else f'outside the advice’s {n} requests'}")
            if _get(row, "preempted", False):
                preempted.setdefault(r, f"step {s}: request {r} preempted ({why})")
            seen.add(r)
            rows[r].append((s, row))
    return rows, preempted


def shape_of(i: int, prompt_len: int, rows: list[tuple[int, Any]], output_len: int | None = None) -> tuple[int, int, int]:
    """Request `i`'s (LP, T, first step) from its rows: the whole prompt at its first step from position 0, then one decode row a
    step; anything else refused by name.  T counts the live decode rows: `output_len` − 1 when the advice gives it (one row past
    the last kept output may follow, discarded), else every decode row."""
    LP = int(prompt_len)
    s0, first = rows[0]
    hit, start, n = int(_get(first, "cache_hit", 0) or 0), int(_get(first, "start")), int(_get(first, "n"))
    if hit or start:
        raise Unexpressible(f"request {i}: a prefix-cache hit of {hit or start} tokens at step {s0} (the cached request wrapper is not "
                            f"ported: v0 needs prefix caching off)")
    if n != LP:
        raise Unexpressible(f"request {i}: a chunked prefill, {n} of its {LP} prompt tokens at step {s0} (no chunked request wrapper: "
                            f"v0 needs chunked prefill off)")
    for k, (s, row) in enumerate(rows[1:], start=1):
        got, want = (s, int(_get(row, "start")), int(_get(row, "n"))), (s0 + k, LP + k - 1, 1)
        if got != want:
            raise Unexpressible(f"request {i}: its row {k} is (step, start, n) = {got}, not the decode {want} (a gap, a recompute or a "
                                f"chunk: v0 runs each request one position a step from its prefill)")
    for k, (s, row) in enumerate(rows):
        if (kind := _get(row, "kind")) is not None and kind != ("prefill" if k == 0 else "decode"):
            raise Unexpressible(f"request {i}: its row at step {s} is marked {kind!r} at position {_get(row, 'start')}")
    if output_len is None:
        return LP, len(rows) - 1, s0
    if not int(output_len) <= len(rows) <= int(output_len) + 1:
        raise Unexpressible(f"request {i}: {len(rows)} rows for {output_len} kept outputs (a row each, and at most one past the last)")
    return LP, int(output_len) - 1, s0


def compile_steps(registered: Registered, advice: Any, *, partial: bool = False) -> Compiled:
    """The Programs of `advice`'s run (module docstring): each request's at its (LP, T, policy), GP-01 over them, and each step's
    live rows as `RowRef`s.  Pure and deterministic: it reads `registered` and `advice` only.  `partial` refuses request by request
    (`Compiled.refused`) where a refusal would otherwise refuse the session."""
    from verity_vllm.pipeline.global_program import request_id_of
    from verity_vllm.program import workload as W

    from .score import request_component
    phi0 = registered.phi
    tp = phi0.target_profile()
    if not tp.batch_invariant:
        raise Unexpressible(f"a target without batch invariance ({phi0.execution!r}): its requests' launches would come from the advice (v1)")
    rows, refused = _rows_of(advice)
    if refused and not partial:
        raise Unexpressible(next(iter(refused.values())))
    units: list[RequestUnit | None] = []
    for i, req in enumerate(advice.requests):
        if i in refused or not rows[i]:
            units.append(None)
            continue
        try:
            LP, T, s0 = shape_of(i, _get(req, "prompt_len"), rows[i], _get(req, "output_len"))
            policy, seed = policy_of(i, _get(req, "sampling"), registered.default_seed)
        except Unexpressible as e:
            if not partial:
                raise
            refused[i] = str(e)
            units.append(None)
            continue
        units.append(RequestUnit(i, "", LP, T, policy, seed, s0))
    ran = [u for u in units if u is not None]
    if not ran:
        raise Unexpressible(f"every request that ran is refused, request {min(refused)}: {refused[min(refused)]}" if refused
                            else "no request of the advice ran a row")
    programs: dict[str, Any] = {}
    comps: dict[str, Any] = {}
    digests: dict[str, str] = {}
    canon: dict = {}
    for u in ran:
        if (key := u.key()) not in comps:
            phi = Phi.from_json({**phi0.to_json(), "sampling": u.policy})
            r = build_program(phi, "request", LP=u.LP, T=u.T)
            programs[r.digest], digests[key] = r.program, r.digest
            comps[key] = request_component(r, canon)
    units = [None if u is None else RequestUnit(u.request, digests[u.key()], u.LP, u.T, u.policy, u.seed, u.first_step) for u in units]
    wl = W.WorkloadProgram([W.WorkloadRequest(request_id_of(u.request), u.request, u.LP, u.T + 1, u.first_step, comps[u.key()], T=u.T)
                            for u in units if u is not None], manifest_digest=None, lag=int(tp.lag()))
    steps: list[list[RowRef]] = []
    discarded: list[tuple[int, int, int, int]] = []
    for s, step in enumerate(advice.steps):
        live = []
        for row in _get(step, "rows"):
            r, start, n = int(_get(row, "request")), int(_get(row, "start")), int(_get(row, "n"))
            if (u := units[r]) is None:
                continue
            if start >= u.LP + u.T:
                discarded.append((s, r, start, n))
            else:
                live.append(RowRef(r, u.digest, start, n, int(start + n >= u.LP)))
        steps.append(live)
    return Compiled(programs=programs, requests=units, workload=wl, steps=steps, discarded=discarded,
                    refused=dict(sorted(refused.items())))


def dead_gates_of(compiled: Compiled) -> dict:
    """Each compiled Program's dead gates as circuit-check counts them (`liveness.dead_gates`, transitive: gates no output depends
    on), with the unread inputs apart, and the dead gates under each of the root's nodes by definition (its id up to `{`)."""
    from verity.ml.circuits.defs import PrimitiveDefinition
    from verity.ml.circuits.intervals import full, total
    from verity.ml.circuits.liveness import Liveness, dead_gates
    out = {}
    for digest, program in sorted(compiled.programs.items()):
        rep = dead_gates(program, limit=0, transitive=True)
        root, lv = program.root, Liveness(transitive=True)
        res = lv.analyse(root, full(root.body.ret_type.leaves))
        by: dict[str, int] = {}
        for node, live in zip(root.body.nodes, res.node_live):
            if node.form == "call":
                n = lv.analyse(node.fn, live).dead
            elif node.form == "primitive" or isinstance(node.fn, PrimitiveDefinition):
                n = (node.n if node.form == "batch" else 1) - total(live)
            else:
                n = 0
            if n:
                name = node.fn.id.split("{", 1)[0]
                by[name] = by.get(name, 0) + n
        if rest := res.dead - sum(by.values()):
            by["(batch or scan at the root)"] = rest
        out[digest] = {"dead_computed": rep.computed, "unread_inputs": rep.input, "approximate": rep.approximate,
                       "by_definition": dict(sorted(by.items(), key=lambda kv: -kv[1]))}
    return {"programs": out, "dead_computed": sum(p["dead_computed"] for p in out.values())}


# ------------------------------------------------------------------------------------------------------------- the traced side


def _label(policy: dict) -> str:
    return "greedy" if policy["kind"] == "greedy" else f"t{policy['temperature']:g}-p{policy['top_p']:g}"


def trace_dir(u: RequestUnit | dict) -> str:
    """The derive dir name of a request Program's trace: `build_request_LP<LP>_T<T>_<policy>`."""
    LP, T, policy = (u.LP, u.T, u.policy) if isinstance(u, RequestUnit) else (u["LP"], u["T"], u["policy"])
    return f"build_request_LP{LP}_T{T}_{_label(policy)}"


def trace_commands(summary: dict, registered: Registered, root: str, *, max_model_len: int = 2048) -> list[list[str]]:
    """One `verity-vllm build --wrapper request` per distinct request Program of a compilation (`Compiled.summary()`), on the pinned
    vLLM (CPU, on meta), each writing `<root>/<trace_dir>`."""
    from verity_vllm.program.frontend.target_profile import TargetProfile
    phi, seen, cmds = registered.phi, set(), []
    target = {k: v for k, v in phi.target.items() if k in TargetProfile.__dataclass_fields__}
    for u in summary["requests"]:
        if u is None or (d := trace_dir(u)) in seen:
            continue
        seen.add(d)
        pol = u["policy"]
        sampling = "greedy" if pol["kind"] == "greedy" else json.dumps({k: pol[k] for k in ("temperature", "top_p", "top_k")}, sort_keys=True)
        cmds.append(["verity-vllm", "build", "--out", os.path.join(root, d), "--wrapper", "request", "--LP", str(u["LP"]), "--T", str(u["T"]),
                     "--model", phi.model["repo"], "--revision", phi.model["revision"], "--max-model-len", str(max_model_len),
                     "--target", json.dumps(target, sort_keys=True), "--profile", phi.profile, "--sampling", sampling])
    return cmds


def compare(summary: dict, root: str) -> dict:
    """Each compiled request Program's digest against its trace's (`<root>/<trace_dir>/result.json`): equal, different, or no trace."""
    out, seen = [], set()
    for u in summary["requests"]:
        if u is None or (d := trace_dir(u)) in seen:
            continue
        seen.add(d)
        res = Path(root, d, "result.json")
        doc = json.loads(res.read_text()) if res.exists() else {}
        traced = (doc.get("program") or {}).get("digest")
        out.append({"unit": d, "compiled": u["digest"], "traced": traced,
                    "outcome": "no_trace" if not traced else ("equal" if traced == u["digest"] else "different"),
                    **({} if traced or not doc else {"trace_error": str(doc.get("message") or doc.get("unsupported"))[:400]})})
    n = {k: sum(1 for x in out if x["outcome"] == k) for k in ("equal", "different", "no_trace")}
    return {"schema": SCHEMA + "/comparison", "compiler": summary["compiler"], "units": out, **n, "mismatches": n["different"] + n["no_trace"]}
