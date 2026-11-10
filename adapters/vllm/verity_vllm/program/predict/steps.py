"""I2, the compiler: a served session's advice -> the Programs its committed rows must pass (v0, request-major).

`compile_steps(registered, advice, ingress, salt=)` reads these and nothing else of the run, and of the advice and the bodies
only the fields `reads.READS` declares, each with its account in the output accounting (the advice through `reads.gate`):
- `registered` (`Registered`): φ without workload or sampling (the model, target, TP, engine, execution and profile), and ℓ, the
  public cap on the advice's length in bits.
- `salt`: the window's seed salt, public, which a stochastic request's seed derives from where its body names none
  (`reads.seed_of`): the bytes of the window record's ``seed_salt``.
- `advice`: I1 v0 (`verity.core.protocols.certifier.advice`), or a served Commit window's (`pipeline.window_advice`: I1 v0 with
  the state of the requests the window carries across its open and close, and its client aborts), private: what serving decided
  that nothing registered predicts. Its canonical bytes, or an `Advice` of either, which is held to its encoding. Nothing else is advice.
- `ingress`: each request's certified body by its id (I1's `RequestAdvice.id`, the body's `user`), the fixed ingress schema: an
  OpenAI completions request with `prompt` as token ids. Sampling parameters are values the request brings, never advice. A body
  given as the network certifier's attested bytes is ingress v0 (`program.registry.io_shell`, under `Registered.served_model`), and its
  request gets the parse unit `IngressPrompt_v1{MODEL, LP, V}` (`RequestUnit.parse`): the unit that reads the ingress transfer
  gates, whose ids output is the request Program's `prompt` input and whose ``ok`` proves the bytes are in the schema.

- A request's Program is the Build's request wrapper at the shape the advice ran it at, `build.build_program(φ_r, "request", LP,
  T, C)`, under the policy its body names (`policy_of`): LP its prompt length, T its live decode rows, the ones whose sample it kept
  (`output_len` − 1), C its prefix-cache hit (the cached request wrapper: every layer's first C key and value rows are its `prefix`
  input, the producers' rows, which GP-01 records per block). The one row async scheduling runs past a request's last kept output
  samples nothing kept: it is `discarded`, in no Program; a request its client aborted (finish `abort`) compiles as one that
  stopped at its last kept output, since the endpoint frees it after the step in flight. A request's (LP, T, C, policy) reads
  its own advice record and body alone, never its batch; requests of one (LP, T, C, policy) share one Program. On a
  batch-invariant target a chunked or preempted request compiles to the same Program: chunking and preemption are advice no
  Program there reads.
- On a batch-invariant target a request's Program does not read its batch, so the advice's steps only place rows: each live row
  is a `RowRef` into its request's Program (positions start..start+n-1; `samples` the tokens the row samples), in I1's row order.
  The Program holds each of its positions C..LP+T-1 once, in the row that first reaches it: a chunked prefill's chunks are its
  prefill rows in turn, and a resume's recompute of positions its request had reached is `discarded`, like the row past the last
  kept output. This rests on the target computing a position's row bit for bit alike however its prefill is chunked or
  recomputed; where it doesn't, the committed row isn't the Program's and the proof fails visibly.
- Off batch invariance the compiler serves class G alone (`launch_class`, `engine.profiles.engine_classes.CLASS_G`: GLM-5.3 at
  TP8, eager or compiled at breakable CUDA graphs, on H200, under a registration of at most `CLASS_G_LIMITS`: steps of at most
  8,192 tokens and 256 sequences, positions at most 131,072, prefill chunks of at most 256 rows, the prefix cache stated on or
  off), whose token-count-dependent kernel choices read the whole launch, padded to its capture size where it runs in a captured
  graph. There a request's Program is its rows in the window alone: the request wrapper after a
  history of C positions, at its launch context (`RequestUnit.launch`, `common._Launch`): per step of its Program, each of its
  prefill chunks and then its decodes, every row that step's forward ran as (n, start), in the advice's batch order, a refused
  request's rows and discarded rows included, since they ran. A prefill the engine ran over several steps is the chunked request
  wrapper (`RequestUnit.chunks`, `build_program(..., chunks=)`): one forward a chunk at its own launch, each chunk's attention
  reading the keys before it, the sample after the last. Its history (`segment_of`) is what it reads and never computes, its
  Program's boundary inputs (`build.build_program`'s C): a cache hit's C rows, the producers', which GP-01 records per block; or,
  for a request the window carries in, its own rows up to where it continues (its carried `computed`), from its prompt's end on
  with its last kept output (`last_token`, from which it resumes at decode C), which GP-01 reads from the last window's edge
  values (`segment.edge_of`, `workload.Edge`): registered parameters `edge.kv` (the latent rows), `edge.ik` and `edge.iks` (the
  indexer's keys and scales, of a request whose rows read past INDEX_TOPK keys in some window) and `edge.last`, each the carried
  requests' fields of one group in the last window's `close` order, that window's output ports; a history group no edge carries
  stays an Input of the request's own, which no row reads. A request the window leaves with outputs to keep reaches as far as
  its rows go, inside its prompt too (sampling nothing), and returns its rows so far and its last kept output
  (`build_program(edge=)`), which GP-01 composes into this window's edge ports in its `close` order; the next window carries its
  state. One whose rows here are all past its last kept output has no Program here. The launch and the chunks are the advice's
  schedule, never a batch-mate's value; requests of one (LP, T, C, policy, chunks, launch, edge) share one Program. Its Program
  reads every launch its rows ran in, so a recompute after preemption has no Program there. A launch the kernels' captured
  statics don't cover is the build's refusal, by name: today a step of more than 2,048 tokens (the all-reduce and cuBLAS
  captures). A row of more than INDEX_TOPK keys computes its top-k from its logits in the Program (`dsa._Model.attention`).
- At tensor-parallel world W, `registered.phi.tp` names one rank, and the Programs are that rank's over its weight shards, each
  collective's other partials its `peers_*` Inputs in call order (`vllm_meta.WithPeers`), in the registered compiler's form as at
  one rank (`build_program(..., dead_free=True)`, the Build's `--dead-free`). The advice is the node's, the same at every rank.
- Each request Program returns its length rule over its outputs as its last check, which the verifier requires to be 1
  (`frontend.stop_rule.of`, `RequestUnit.stop`): requests share a Program only under one rule, which with no stop ids reads its caps.
- GP-01 (`program.workload.WorkloadProgram`) composes the request Programs side by side, sharing only the weights, each at its
  first scheduled step and with cap T + 1 (the executed prefix), under the target's lag.
- On a batch-invariant target, a request a Commit window carries in (admitted in an earlier window) compiles to the Program it has
  whole, at the C its admission had, and each window places its own rows: from where it had reached (its carried `made`), its kept
  outputs counted from its carried `kept`. A request the window leaves with outputs to keep reaches only as far as its rows go. A
  carried request with no row in the window has no unit there (`Plan.idle`) and needs no body.

The compiler is total: it compiles its input or refuses it by name (`Unexpressible`), deterministically. Refused, in order:
- advice that isn't I1's bytes or an `Advice`, is over ℓ (before any of it is read), or isn't I1 v0 (`MalformedAdvice`: I1's
  validity, its types, ranges, order and the schedule's position rules);
- a target without batch invariance outside class G, and in class G a step over the registered `max_num_batched_tokens` or
  `max_num_seqs`, or a request's row over its `long_prefill_token_threshold`;
- a request whose body is missing or off the ingress schema, whose prompt isn't `prompt_len` token ids of the vocabulary,
  whose `output_len` and `finish` its `max_tokens` contradicts, whose stop fields its length rule doesn't read
  (`stop_rule.body_refusal`), that finished on a stop id with none, or stochastic with a seed `reads.seed_of` refuses;
- a cache hit where the registration states the prefix cache off (`reads.hits_off`), or whose `sources` don't hold the request's
  own KV: through each run, the source's certified prompt must equal the request's (the KV at a position reads every token up to
  it), in whole blocks of the engine's block size where the Program reads the hit (a resume's run of its own KV is I1's replay's);
- what v0 doesn't compile: on a batch-invariant target, a prefix-cache hit on a stochastic request (the cached stochastic wrapper
  isn't ported), or on a request a window carries in (its prefix's producers are the admitting window's requests); positions
  past the model's (in class G, past the registered `max_model_len` too); in class G, a request whose Program wouldn't hold every
  launch it ran in (`launch_of`), one the window leaves somewhere its Program doesn't reach (`edge_of`), and a hit on a request
  with no Program in the window; sampling other than greedy or seeded top-p with top_k 0 (min_p, the penalties, logit_bias,
  min_tokens, n > 1, or a parameter this compiler doesn't know);
- a request Program with a dead unit (`program.prune`: a node no output reads), whose fix is in the lowering, never the advice;
- a compilation `predict.batch.check_compiled` refuses (`BatchDependent`): one where a request's outputs could read its
  batch-mates (R3 a derived split count; R2 a request planned differently alone, which keeps a cache hit on another request's KV
  only when its tokens are checked equal; R1 GP-01's wiring and cones).

A refusal refuses the session. `partial=True` (a measurement over served traffic, never a registration) refuses request by
request instead: a refused request's rows and Program leave the compilation, and so does every request whose cache hit reads its
KV; `refused` names why. It records `check_compiled`'s refusal (`batch_rule`) rather than raising it.

The compiler digest is `score.code_version()`, with ℓ beside it. `Compiled.transfer_gates` lists the templates whose units hold
transfer gates (the ingress parse and the token selects), each of which a window proves whole. `trace_commands` lists the Build traces (`verity-vllm build
--wrapper request`, in class G at the unit's `--launch-context` and `--batched`) of a compilation's Programs, and `compare` holds the compiled digests to them. With `named=True` each Program
carries what its trace's Build says beside it (`Compiled.named`, `naming`), so an auditor writes that Build with no vLLM.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import verity.core.protocols.certifier.advice as I1
from verity.ml.circuits.codec import program_digest
from verity.ml.circuits.defs import bind
from verity_vllm.engine.profiles.engine_classes import CLASS_G, CLASS_G_EXECUTIONS, CLASS_G_LIMITS
from verity_vllm.pipeline import window_advice as WA
from verity_vllm.program import prefix_plan as PP
from verity_vllm.program.frontend import stop_rule as SR
from verity_vllm.program.prune import dead_units
from verity.ml.definitions.registry import io_shell as IO
from verity.ml.definitions.registry.topp_geometry import registered_splits

from . import reads
from .build import Unexpressible, build_program, decoder, model_shape
from .common import cyclic_gc_off, edge_config
from .phi import Phi
from .segment import edge_groups, edge_of, launch_of, segment_of

SCHEMA = "verity-vllm/compiled-steps/v2"
#: the token-select families, whose units' sampled ids are a request Program's egress (`Compiled.transfer_gates`)
EGRESS = ("TokenSelect", "GumbelTopPTokenSelect")
#: ℓ when a registration names none: 1 MiB of advice (PROVE-PATH's `CAP_BITS`), far above a session's few thousand rows
CAP_BITS = 1 << 23
#: vLLM's `_SAMPLING_EPS`: a temperature below it samples greedily; the Build's policy is greedy only at 0
SAMPLING_EPS = 1e-5
#: a body's fields by their roles in `reads.READS`: the sampling parameters (`engine.step_trace.SAMPLING`) that leave a request's
#: Program unchanged at these values, the ones a request's Program may read, and the ones that compute nothing
_INERT = {f.name: f.inert for f in reads.body("inert")}
_KNOWN = tuple(f.name for f in reads.body("inert", "finish", "policy"))
_TRANSPORT = tuple(f.name for f in reads.body("prompt", "transport"))


@dataclass(frozen=True)
class Registered:
    """The registration's static part: φ without workload or sampling (the model, target, TP, engine, execution and profile), the
    seed serving fills in (`engine.served_seed.DEFAULT_SEED`, under which no request compiles: `reads.seed_of`), ℓ, the cap on
    the advice's length in bits, and the served model name ingress v0's ``"model"`` carries (`io_shell`; None: no ingress is bytes)."""

    phi: Phi
    default_seed: int
    cap_bits: int = CAP_BITS
    served_model: str | None = None

    def __post_init__(self) -> None:
        if self.phi.requests:
            raise ValueError("a registered φ carries no workload: the advice brings the requests")
        if type(self.cap_bits) is not int or self.cap_bits < 8:
            raise ValueError(f"the cap ℓ is a whole number of bits, at least 8, not {self.cap_bits!r}")
        if self.served_model is not None and (why := IO.refuse({"MODEL": self.served_model})):
            raise ValueError(f"the served model name has no ingress v0 layout: {why}")

    @classmethod
    def of(cls, phi: Phi, default_seed: int, cap_bits: int = CAP_BITS, served_model: str | None = None) -> Registered:
        return cls(Phi.from_json({**phi.to_json(), "workload": {}, "sampling": {"kind": "greedy"}}), int(default_seed), cap_bits,
                   served_model)


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
    """A request that ran: its Program's shape and digest, its policy, the seed its `seed` Input takes, its first step and its id;
    with a prefix-cache hit, its C cached rows and the request that computed each cached block; in class G, its launch context and,
    for a prefill the engine ran in several launches, its chunks (`launch_of`), and C its history in the window (`segment_of`), for a
    request the window carries in its rows at the open."""

    request: int
    digest: str
    LP: int
    T: int
    policy: dict
    seed: int | None
    first_step: int
    id: str = ""
    C: int = 0
    producers: tuple[int, ...] = ()
    parse: Any = None   # with byte ingress, its `IngressPrompt_v1` binding: the unit that reads the ingress transfer gates
    carried_in: bool = False    # admitted in an earlier Commit window
    carried_out: bool = False   # left to the next window with outputs to keep
    launch: tuple[tuple[tuple[int, int], ...], ...] | None = None  # class G: per Program step, every row its forward ran, (n, start)
    chunks: tuple[int, ...] = ()  # class G: the prefill's rows per launch, where it ran in more than one
    edge: tuple[str, ...] = ()    # class G, carried out: the history groups its Program returns (`segment.edge_groups`)
    stop: dict | None = None    # its length rule over its outputs (`frontend.stop_rule.spec`, `stop_of`); None: it has none

    def key(self) -> str:
        return json.dumps([self.LP, self.T, self.policy, *([self.C] if self.C else []), *([{"chunks": self.chunks}] if self.chunks else []),
                           *([{"launch": self.launch}] if self.launch is not None else []), *([edge_config(self.edge)] if self.edge else []),
                           *([{"stop": self.stop}] if self.stop else [])],
                          sort_keys=True)

    def events(self) -> int:
        """The samples its Program makes: its prefill's when C < LP <= LP + T, then one a decode position from max(C, LP) to LP + T - 1."""
        return max(self.LP + self.T - max(self.C, self.LP), 0) + (self.C < self.LP <= self.LP + self.T)

    def prefix(self) -> dict | None:
        """GP-01's record of its history (`WorkloadRequest.prefix`): the C rows and the request that computed each cached block, or,
        for a request the window carries in, its own rows at the window's open (``carried``); None without one."""
        if not self.C:
            return None
        return {"rows": self.C, "carried": True} if self.carried_in else {"rows": self.C, "producers": list(self.producers)}

    def batch(self) -> list | None:
        """The launch context as `build_program`'s ``batch`` and the Build's `--launch-context '{"batch": ...}'` take it."""
        return None if self.launch is None else [[list(seq) for seq in step] for step in self.launch]


@dataclass
class Compiled:
    programs: dict[str, Any]              # request Program digest -> Program
    requests: list[RequestUnit | None]    # by advice request index; None for a refused request, or in class G a carried one with no live row
    workload: Any                         # GP-01's WorkloadProgram over the requests that ran
    steps: list[list[RowRef]]             # per advice step, its live rows in I1's order
    discarded: list[tuple[int, int, int, int]] = field(default_factory=list)  # (step, request, start, n) of rows no Program holds
    refused: dict[int, str] = field(default_factory=dict)                     # request -> why (partial=True only)
    cap_bits: int = CAP_BITS
    batch_rule: str | None = None         # partial=True only: `predict.batch`'s refusal of this compilation, which it doesn't raise
    idle: tuple[int, ...] = ()            # carried requests with no row in the window: no unit here
    named: dict[str, Any] = field(default_factory=dict)  # named=True only: request Program digest -> its names (`naming.Named`)

    @property
    def transfer_gates(self) -> tuple[str, ...]:
        """The templates whose units hold transfer gates, which a window samples whole: each request's ingress parse
        (`RequestUnit.parse`) and its Program's token selects, whose sampled ids are the egress until an in-circuit renderer takes
        their place. No model-step template is among them."""
        ids = {u.parse.id for u in self.requests if u is not None and u.parse is not None}
        callees = {n.fn.id for p in self.programs.values() for n in p.fn.body.nodes}
        ids |= {i for i in callees if i.split("{")[0].rsplit("_v", 1)[0] in EGRESS}
        return tuple(sorted(ids))

    def summary(self) -> dict:
        from .score import code_version
        return {"schema": SCHEMA, "compiler": code_version(), "cap_bits": self.cap_bits, "workload_digest": self.workload.digest,
                "lag": self.workload.lag, "programs": sorted(self.programs),
                "requests": [None if u is None else {"request": u.request, "id": u.id, "digest": u.digest, "LP": u.LP, "T": u.T, "C": u.C,
                                                     "policy": u.policy, "seed": u.seed, "first_step": u.first_step,
                                                     "parse": u.parse and u.parse.id, **({"launch": u.batch()} if u.launch else {}),
                                                     **({"chunks": list(u.chunks)} if u.chunks else {}), **edge_config(u.edge),
                                                     **({"stop": u.stop} if u.stop else {})}
                             for u in self.requests],
                "steps": [[[r.request, r.start, r.n, r.samples] for r in rows] for rows in self.steps],
                "discarded": [list(d) for d in self.discarded], "refused": {str(i): why for i, why in sorted(self.refused.items())},
                "batch_rule": self.batch_rule, "transfer_gates": list(self.transfer_gates),
                "carry": {"in": [u.request for u in self.requests if u is not None and u.carried_in],
                          "out": [u.request for u in self.requests if u is not None and u.carried_out], "idle": list(self.idle)}}


def _number(x: Any) -> bool:
    return type(x) in (int, float) and math.isfinite(x)


def policy_of(i: int, sampling: Any, num_sms: int | None, *, request_id: str = "", salt: bytes | None = None,
              carried: bool = False) -> tuple[dict, int | None]:
    """Request `i`'s sampling policy as φ.sampling takes it, and its seed Input's value (None when greedy), from the parameters its
    body carries (`SamplingParams` names), refused where its Program wouldn't be the Build's. Top-p takes the registered split
    count S = registered_splits(num_SMs), the S serving pins every top-p launch to (`engine.topp_pin`), at any top_p. The seed is
    `reads.SEED`'s, the body's or derived from ``request_id`` and the window's seed ``salt`` (`reads.seed_of`), else refused."""
    if not isinstance(sampling, Mapping) or any(type(k) is not str for k in sampling):
        raise Unexpressible(f"request {i}: its sampling parameters aren't an object of named values")
    s = dict(sampling)
    if s.get("temperature") is None:
        raise Unexpressible(f"request {i}: no temperature among its sampling parameters (the policy can't be read)")
    t, p, seed = s["temperature"], 1.0 if s.get("top_p") is None else s["top_p"], s.get("seed")
    if not (_number(t) and t >= 0 and _number(p) and 0 < p <= 1 and (seed is None or (type(seed) is int and 0 <= seed < 1 << 64))):
        raise Unexpressible(f"request {i}: temperature {t!r}, top_p {p!r} and seed {seed!r} aren't a number ≥ 0, a number in (0, 1] "
                            f"and an integer in [0, 2^64)")
    facts = [f"{k}={v!r}" for k, v in sorted(s.items()) if k in _INERT and v not in _INERT[k]]
    facts += [f"{k} (a parameter this compiler doesn't know)" for k in sorted(s) if s[k] is not None and k not in _KNOWN]
    if 0 < t < SAMPLING_EPS:
        facts.append(f"temperature {t!r} (vLLM samples it greedily, the Build's policy is greedy at 0 only)")
    if t > 0 and num_sms is None:
        facts.append("top-p on a target that declares no num_sms (its registered split count registered_splits(num_SMs))")
    if facts:
        raise Unexpressible(f"request {i}: sampling the vocabulary can't express: {'; '.join(facts)}")
    if t == 0:
        return {"kind": "greedy"}, None
    return ({"kind": "stochastic", "temperature": float(t), "top_p": float(p), "top_k": 0, "seeded": True,
             "splits": registered_splits(int(num_sms))}, reads.seed_of(i, request_id, seed, salt, carried))


def advice_of(registered: Registered, advice: Any) -> Any:
    """The advice as a Commit window's (`window_advice`; I1 v0's when it carries nothing), through the gate that refuses a field
    `reads.READS` doesn't declare (`reads.gate`): its canonical bytes within ℓ, or an I1 or window `Advice` whose encoding is;
    anything else refused by name."""
    if isinstance(advice, (I1.Advice, WA.Advice)):
        try:
            data = (WA if isinstance(advice, WA.Advice) else I1).encode(advice)
        except Exception as e:  # noqa: BLE001 -- an Advice built around its checks has no encoding: a refusal like any other
            raise Unexpressible(f"ill-formed advice: an Advice with no encoding ({type(e).__name__}: {e})") from None
    elif type(advice) is bytes:
        data = advice
    else:
        raise Unexpressible(f"the advice is I1's bytes or an I1 Advice, not a {type(advice).__name__}: nothing else is advice")
    if 8 * len(data) > registered.cap_bits:
        raise Unexpressible(f"the advice is {8 * len(data)} bits, over the registered cap ℓ = {registered.cap_bits} bits")
    try:
        return reads.gate(WA.decode(data))
    except Exception as e:  # noqa: BLE001 -- I1's refusal is MalformedAdvice; any other failure on these bytes refuses them too
        raise Unexpressible(f"ill-formed advice: {e}" if isinstance(e, I1.MalformedAdvice) else
                            f"ill-formed advice: its decode failed ({type(e).__name__}: {e})") from None


def ingress_body(i: int, q: I1.RequestAdvice, body: bytes, served: str | None, vocab: int) -> dict:
    """Request `i`'s ingress bytes as the body ingress v0 reads them (`io_shell.reference`: its ids, max_tokens and temperature 0),
    at the advice's `prompt_len`; refused unless they are in the schema, which `IngressPrompt_v1`'s ``ok`` then proves."""
    if served is None:
        raise Unexpressible(f"request {i}: its ingress is bytes, and the registration names no served model for ingress v0")
    n = IO.length(served, q.prompt_len, vocab)
    if len(body) != n:
        raise Unexpressible(f"request {i}: its ingress is {len(body)} bytes, not ingress v0's {n} at {q.prompt_len} prompt tokens")
    ok, ids, m = IO.reference(body, served, q.prompt_len, vocab)
    if not ok:
        raise Unexpressible(f"request {i}: its ingress bytes aren't in ingress schema v0 (`io_shell`)")
    return {"prompt": ids, "max_tokens": m, "temperature": 0}


def request_of(i: int, q: I1.RequestAdvice, body: Any, vocab: int) -> tuple[tuple[int, ...], dict]:
    """Request `i`'s certified prompt and sampling parameters, held to its advice record: the prompt is `prompt_len` token ids of
    the vocabulary, and its `max_tokens` bounds `output_len`, reached exactly when it finished on length (an abort is before it:
    at its cap vLLM has finished it on length, and the abort finds nothing to free)."""
    if not isinstance(body, Mapping) or any(type(k) is not str for k in body):
        raise Unexpressible(f"request {i} ({q.id!r}): no certified request body carries its id")
    prompt = body.get("prompt")
    if not isinstance(prompt, (list, tuple)) or not all(type(t) is int and 0 <= t < vocab for t in prompt):
        raise Unexpressible(f"request {i}: its certified prompt isn't a list of token ids below {vocab} (the ingress schema carries "
                            f"the prompt as token ids)")
    if len(prompt) != q.prompt_len:
        raise Unexpressible(f"request {i}: the advice says {q.prompt_len} prompt tokens, its certified prompt has {len(prompt)}")
    params = {k: v for k, v in body.items() if k not in _TRANSPORT}
    m = params.get("max_tokens")
    if type(m) is not int or m < 1:
        raise Unexpressible(f"request {i}: max_tokens {m!r} (v0 needs the cap in the body, a positive integer)")
    if q.output_len > m or (q.finish == "length" and q.output_len != m) or (q.finish == "abort" and q.output_len == m):
        raise Unexpressible(f"request {i}: {q.output_len} kept outputs, finished on {q.finish}, against its max_tokens {m}")
    if why := SR.body_refusal(params):
        raise Unexpressible(f"request {i}: {why}")
    if q.finish == "eos" and params.get("ignore_eos") is True:
        raise Unexpressible(f"request {i}: finished on a stop token with ignore_eos set (and no stop_token_ids, its only stop ids)")
    return tuple(prompt), params


def stop_of(i: int, q: I1.RequestAdvice, T: int, ids: tuple[int, ...], max_tokens: int, max_model_len: int) -> dict | None:
    """Request `i`'s length rule over its Program's outputs, the last at index T (`frontend.stop_rule.of`), under stop ids `ids`;
    None where it has none (T < 0: a class-G window leaves it inside its prompt)."""
    if T < 0:
        return None
    try:
        return SR.of(q.finish, q.output_len, T, ids, max_tokens=max_tokens, prompt_len=q.prompt_len, max_model_len=max_model_len)
    except ValueError as e:
        raise Unexpressible(f"request {i}: {e}") from None


def check_sources(s: int, row: I1.Row, prompts: Mapping[int, tuple[int, ...]]) -> None:
    """A cache hit's sources hold the KV of the request's own tokens: through each run, the source's certified prompt equals the
    hitting request's (the KV at a position reads every token up to it). Refused otherwise, or where a run leaves a prompt. A
    resume's run of its own KV is its own by I1's replay (the request computed those positions), past its prompt too."""
    r, at = row.request, 0
    for src in row.sources:
        end, a, b = at + src.n, prompts.get(row.request), prompts.get(src.request)
        if src.request == r and row.preempted:
            at = end
            continue
        if a is None or b is None:
            raise Unexpressible(f"request {r}: step {s}'s cache hit reads request {src.request}'s KV, and one of the two has no certified "
                                f"prompt")
        if end > min(len(a), len(b)):
            raise Unexpressible(f"request {r}: step {s}'s cache hit reads request {src.request}'s KV at positions {at} … {end - 1}, past "
                                f"a certified prompt (v0 checks sources against prompts only)")
        if a[:end] != b[:end]:
            raise Unexpressible(f"request {r}: a forged cache hit at step {s}: request {src.request}'s certified prompt differs from its "
                                f"own before position {end}, so that KV isn't its own")
        at = end


Rows = list[tuple[int, int, I1.Row]]  # a request's rows as (step, index in the step, row), in step order
Placed = tuple[tuple[int, int, int] | None, list[tuple[int, int]]]  # a row's part its Program holds, and its parts none holds


def shape_of(i: int, prompt_len: int, rows: Rows, output_len: int, carry: WA.Carried | None = None) -> tuple[int, int, int, int]:
    """Request `i`'s (LP, T, first step, C): its prompt length, its live decode rows (`output_len` − 1: one row past the last kept
    output may follow, discarded), and its first row's step and prefix-cache hit (0 without one), which leaves at least the prompt's
    last position to compute. A request the window carries in (``carry``) has the C of its admission."""
    LP = int(prompt_len)
    s0, _, first = rows[0]
    if carry is not None:
        return LP, int(output_len) - 1, s0, int(carry.C)
    C = int(first.cache_hit)
    if first.start != C or not C < LP:
        raise Unexpressible(f"request {i}: its first row starts at position {first.start} with a cache hit of {C} tokens, for a prompt of "
                            f"{LP} (a prefill computes the positions after its hit, and at least the last)")
    return LP, int(output_len) - 1, s0, C


def place(i: int, LP: int, T: int, C: int, rows: Rows, output_len: int, carry: WA.Carried | None = None,
          left: bool = False) -> dict[tuple[int, int], Placed]:
    """Each of request `i`'s rows by (step, index in the step): the part of it its Program holds, (start, n, samples) or None, and
    its parts no Program holds, [(start, n)]. The Program holds positions C..LP+T-1 once each, in the row that first reaches them:
    a resume's positions up to where its request had reached are a recompute, and the row past the last kept output is past the
    Program. `samples` is 1 where the row's sample is a kept output (I1's rule: the row reaches its request's known tokens). A
    request the window carries in (``carry``) has reached as far as it had computed and kept its carried outputs; one it leaves
    with outputs to keep (``left``) reaches as far as its rows go."""
    out: dict[tuple[int, int], Placed] = {}
    reached, known, kept = C, LP, 0
    if carry is not None:
        reached, known, kept = min(LP + T, max([C, *(hi for _, hi in carry.made)])), LP + carry.kept, carry.kept
    for s, j, row in rows:
        a, b = row.start, row.start + row.n
        sample = b == known and kept < output_len
        if sample:
            kept, known = kept + 1, known + 1
        lo, hi = max(a, reached), min(b, LP + T)
        if sample and not lo < hi == b:
            raise Unexpressible(f"request {i}: step {s} keeps the sample at position {b - 1}, which its Program doesn't compute in that row")
        out[s, j] = ((lo, hi - lo, int(sample)) if lo < hi else None,
                     [(x, y - x) for x, y in ((a, min(b, reached)), (max(a, LP + T), b)) if x < y])
        reached = max(reached, hi)
    if reached != LP + T and not left:
        raise Unexpressible(f"request {i}: its rows reach position {reached}, not the {LP + T} of its prompt and live decode rows")
    return out


def producers_of(i: int, s: int, row: I1.Row, bs: int) -> tuple[int, ...]:
    """The request that computed each cached block of `row`'s hit (`program.prefix_plan`'s producers), from its sources: whole blocks
    of the engine's `bs` tokens, as vLLM's prefix cache serves them."""
    out: list[int] = []
    for src in row.sources:
        if src.n % bs:
            raise Unexpressible(f"request {i}: step {s}'s cache hit reads {src.n} tokens of request {src.request}'s KV, not whole blocks of "
                                f"{bs} (vLLM's prefix cache serves whole blocks)")
        out += [src.request] * (src.n // bs)
    return tuple(out)


def launch_class(phi: Phi) -> None:
    """Refuse a registration without batch invariance outside class G (`CLASS_G`), naming each fact of it outside: the model,
    world and device, an execution the class doesn't serve (`CLASS_G_EXECUTIONS`) or a target compilation other than the one it
    serves that execution at, and the engine limits, each stated and within `CLASS_G_LIMITS`, and the prefix cache stated on or
    off (vLLM's default is on).  Async scheduling isn't one: the rows it runs past a request's stop are the advice's, `discarded`
    and in the launches they ran in."""
    eng = dict(phi.engine or {})
    facts = [f"{k} {v!r}, not {CLASS_G[k]!r}" for k, v in (("model_type", str(phi.hf_config.get("model_type", "")).lower()),
                                                        ("world", int((phi.tp or {}).get("world", 1))),
                                                        ("device_name", (phi.target or {}).get("device_name"))) if v != CLASS_G[k]]
    if phi.execution not in CLASS_G_EXECUTIONS:
        facts.append(f"execution {phi.execution!r}, not one of {', '.join(map(repr, CLASS_G_EXECUTIONS))}")
    elif (phi.target or {}).get("compilation") != CLASS_G_EXECUTIONS[phi.execution]:
        facts.append(f"target compilation {(phi.target or {}).get('compilation')!r} at execution {phi.execution!r}, not "
                     f"{CLASS_G_EXECUTIONS[phi.execution]!r}")
    facts += [f"engine arg {k} {eng.get(k)!r}, not a stated value of at most {cap}" for k, cap in CLASS_G_LIMITS.items()
              if type(eng.get(k)) is not int or not 0 < eng[k] <= cap]
    if type(eng.get("enable_prefix_caching")) is not bool:
        facts.append(f"engine arg enable_prefix_caching {eng.get('enable_prefix_caching')!r}, not stated True or False")
    if facts:
        raise Unexpressible(f"a target without batch invariance outside class G ({phi.execution!r}), whose launches the advice doesn't "
                            f"fix: {'; '.join(facts)}")


@dataclass
class Plan:
    """What the advice fixes before any Program is built: each request's unit (its digest unset), each step's live rows as
    (request, start, n, samples), the discarded rows, and the refused requests."""

    requests: list[RequestUnit | None]
    steps: list[list[tuple[int, int, int, int]]]
    discarded: list[tuple[int, int, int, int]]
    refused: dict[int, str]
    idle: tuple[int, ...] = ()

    def distinct(self) -> list[RequestUnit]:
        """The first request of each (LP, T, policy), in request order: one Program each."""
        seen: set[str] = set()
        out = []
        for u in self.requests:
            if u is not None and u.key() not in seen:
                seen.add(u.key())
                out.append(u)
        return out

    def without(self, why: dict[int, str]) -> Plan:
        """This plan with the requests ``why`` names refused, each for its reason, and so every request whose cache hit reads one of
        them, their rows and discarded rows dropped."""
        units, refused = list(self.requests), dict(self.refused)
        while why:
            for i, reason in why.items():
                refused.setdefault(i, reason)
                units[i] = None
            why = {}
            for u in units:
                if u is not None and (p := next((p for p in u.producers if units[p] is None), None)) is not None:
                    why[u.request] = f"request {u.request}: its cache hit reads request {p}'s KV, and request {p} is refused"
        if all(u is None for u in units):
            raise Unexpressible(f"every request is refused, request {min(refused)} first: {refused[min(refused)]}")
        gone = refused.keys() - self.refused.keys()
        return Plan(units, [[row for row in rows if row[0] not in gone] for rows in self.steps],
                    [d for d in self.discarded if d[1] not in gone], dict(sorted(refused.items())), self.idle)


def request_program(registered: Registered, unit: RequestUnit, named: bool = False) -> Any:
    """`unit`'s Program (a `build.BuildResult`): the Build's request wrapper at its (LP, T), under its policy, after its C rows of
    history, in class G at its launch context and in its prefill chunks, at φ's rank, in the registered compiler's form
    (`dead_free`) at any world, batched in class G, its length rule last; `named`: its trace's names (`naming.named_program`, no class-G decoder)."""
    phi = Phi.from_json({**registered.phi.to_json(), "sampling": unit.policy})
    if named:
        from .naming import named_program
        return named_program(phi, unit.LP, unit.T, unit.C, stop=unit.stop)
    r = build_program(phi, "request", LP=unit.LP, T=unit.T, C=unit.C, dead_free=True, batch=unit.batch(), chunks=unit.chunks,
                      edge=unit.edge, stop=unit.stop, with_desc=False, batched=not phi.target_profile().batch_invariant)
    return replace(r, digest=program_digest(r.program))


def compile_steps(registered: Registered, advice: Any, ingress: Any, *, salt: bytes | None = None, partial: bool = False,
                  named: bool = False) -> Compiled:
    """The Programs of `advice`'s session (module docstring): each request's at its (LP, T, policy), GP-01 over them, and each step's
    live rows as `RowRef`s. Total and deterministic: it reads `registered`, `advice`, `ingress` and the window's seed `salt` only, and
    compiles them or raises `Unexpressible`. `partial` refuses request by request (`Compiled.refused`) where a refusal would
    otherwise refuse the session, a request whose Program the vocabulary refuses (a launch no capture covers) included.
    `named`: each Program with what its Build says beside it, as its trace would (`Compiled.named`, `naming.write_build`), or
    `Unexpressible` where the naming doesn't reach it. The cyclic collector is off throughout (`cyclic_gc_off`, at the end)."""
    from verity_vllm.pipeline.global_program import request_id_of
    from verity_vllm.program import workload as W

    from .score import request_component
    plan = plan_steps(registered, advice, ingress, partial=partial, salt=salt)
    built: dict[str, Any] = {}
    lost: dict[int, str] = {}
    for u in plan.distinct():
        try:
            built[u.key()] = request_program(registered, u, named)
        except Unexpressible as e:
            if not partial:
                raise
            lost.update({w.request: f"request {w.request}: its Program is refused: {e}" for w in plan.requests
                         if w is not None and w.key() == u.key()})
    if lost:
        plan = plan.without(lost)
    programs: dict[str, Any] = {}
    comps: dict[str, Any] = {}
    digests: dict[str, str] = {}
    names: dict[str, Any] = {}
    for u in plan.distinct():
        r = built[u.key()]
        if named:
            names[r.digest] = r.named
        if dead := dead_units(body := r.program.fn.body):
            raise Unexpressible(f"request {u.request}'s Program {r.digest[:16]} holds {len(dead)} dead units, node {dead[0]} "
                                f"({body.nodes[dead[0]].fn.id}) first, which no output reads: drop them in the lowering, on the "
                                f"Build's side and predict's alike (`program.prune.without_dead_units` drops a request wrapper's)")
        programs[r.digest], digests[u.key()] = r.program, r.digest
        comps[u.key()] = request_component(r)
    units = [None if u is None else replace(u, digest=digests[u.key()]) for u in plan.requests]
    adv = advice_of(registered, advice)
    edge = None if registered.phi.target_profile().batch_invariant else edge_of(registered.phi, adv, units, plan.idle)
    wl = W.WorkloadProgram([W.WorkloadRequest(request_id_of(u.request), u.request, u.LP, u.events(), u.first_step, comps[u.key()],
                                              T=u.events() - 1, prefix=u.prefix())
                            for u in units if u is not None], manifest_digest=None, lag=int(registered.phi.target_profile().lag()),
                           edge=edge)
    steps = [[RowRef(r, units[r].digest, start, n, k) for r, start, n, k in rows] for rows in plan.steps]
    compiled = Compiled(programs=programs, requests=units, workload=wl, steps=steps, discarded=plan.discarded, refused=plan.refused,
                        cap_bits=registered.cap_bits, idle=plan.idle, named=names)
    from .batch import BatchDependent, check_compiled
    try:
        check_compiled(registered, adv, ingress, compiled, lambda *a: plan_steps(*a, salt=salt))
    except BatchDependent as e:
        if not partial:
            raise
        compiled.batch_rule = str(e)
    return compiled


def plan_steps(registered: Registered, advice: Any, ingress: Any, *, partial: bool = False, salt: bytes | None = None) -> Plan:
    """`compile_steps` up to building Programs: every refusal, each request's shape and policy, and each step's live rows."""
    adv = advice_of(registered, advice)
    phi0 = registered.phi
    if not (bi := phi0.target_profile().batch_invariant):
        launch_class(phi0)
    if not isinstance(ingress, Mapping):
        raise Unexpressible(f"the ingress is each request's certified body by its id, not a {type(ingress).__name__}")
    if not adv.requests:
        raise Unexpressible("the advice has no request")
    ids = {q.id for q in adv.requests}
    if not partial and (extra := [k for k in ingress if k not in ids]):
        raise Unexpressible(f"{len(extra)} certified requests have no row in the advice, {extra[0]!r} first")
    if bi:
        shape = model_shape(phi0.model["hf_config"])
        V, max_pos = shape.V, shape.max_pos
    else:
        hf, eng = phi0.hf_config, phi0.engine
        V, max_pos = int(hf["vocab_size"]), min(int(hf["max_position_embeddings"]), eng["max_model_len"])
    refused: dict[int, str] = {}

    def refuse(i: int, why: str) -> None:
        if not partial:
            raise Unexpressible(why)
        refused.setdefault(i, why)

    if not bi:
        for s, step in enumerate(adv.steps):
            over = [f"{n} rows, over the registered max_num_batched_tokens {eng['max_num_batched_tokens']}"
                    for n in [sum(row.n for row in step.rows)] if n > eng["max_num_batched_tokens"]]
            over += [f"{n} sequences, over the registered max_num_seqs {eng['max_num_seqs']}"
                     for n in [len(step.rows)] if n > eng["max_num_seqs"]]
            for row in step.rows:
                for why in over:
                    refuse(row.request, f"step {s} runs {why}")
                if row.n > eng["long_prefill_token_threshold"]:
                    refuse(row.request, f"request {row.request}: step {s} runs {row.n} of its rows, over the registered "
                                        f"long_prefill_token_threshold {eng['long_prefill_token_threshold']} (the engine schedules at "
                                        f"most that many of a request's rows a step)")

    rows: dict[int, Rows] = {i: [] for i in range(len(adv.requests))}
    for s, step in enumerate(adv.steps):
        for j, row in enumerate(step.rows):
            rows[row.request].append((s, j, row))
    carried = dict(enumerate(adv.carried))
    idle = tuple(i for i in carried if not rows[i])
    left = {c.request for c in WA.close(adv) if c.must}
    prompts, params, wired = {}, {}, set()
    for i, q in enumerate(adv.requests):
        if i in idle:
            continue
        try:
            body = ingress.get(q.id)
            if type(body) is bytes:
                body = ingress_body(i, q, body, registered.served_model, V)
                wired.add(i)
            prompts[i], params[i] = request_of(i, q, body, V)
        except Unexpressible as e:
            refuse(i, str(e))
    for i, why in reads.hits_off(adv, phi0.engine or {}):
        refuse(i, why)
    for s, step in enumerate(adv.steps):
        for row in step.rows:
            if row.sources:
                try:
                    check_sources(s, row, prompts)
                except Unexpressible as e:
                    refuse(row.request, str(e))
    bs = int((phi0.engine or {}).get("block_size") or PP.DEFAULT_BLOCK_SIZE)
    stops, mml = SR.stop_ids(phi0.hf_config), min(max_pos, int((phi0.engine or {}).get("max_model_len") or max_pos))
    units: list[RequestUnit | None] = []
    placed: dict[tuple[int, int], Placed] = {}
    spent: set[int] = set()
    for i, q in enumerate(adv.requests):
        if i not in refused and i not in idle:
            try:
                carry = carried.get(i)
                LP, T, s0, C = shape_of(i, q.prompt_len, rows[i], q.output_len, carry)
                if LP + T > max_pos:
                    raise Unexpressible(f"request {i}: positions to {LP + T - 1}, past the model's {max_pos} positions" if bi else
                                        f"request {i}: positions to {LP + T - 1}, past the {max_pos} the model and the registered "
                                        f"max_model_len allow")
                if not bi:
                    C, T = segment_of(i, LP, T, rows[i], carry, i in left)
                own = place(i, LP, T, C, rows[i], q.output_len, carry, i in left)
                if C >= LP + T:
                    spent.add(i)
                    placed.update(own)
                    units.append(None)
                    continue
                chunks, launch = ((), None) if bi else launch_of(i, LP, T, C, rows[i], adv.steps)
                edge = edge_groups(decoder(phi0, 1, 0), q) if i in left and launch is not None else ()
                policy, seed = policy_of(i, params[i], phi0.target_profile().num_sms, request_id=q.id, salt=salt,
                                         carried=carry is not None)
                if bi and C and policy["kind"] != "greedy":
                    raise Unexpressible(f"request {i}: a prefix-cache hit of {C} tokens on a stochastic request (the cached stochastic "
                                        f"request wrapper isn't ported)")
                if bi and C and carry is not None:
                    raise Unexpressible(f"request {i} is carried into the window with the prefix-cache hit of {C} tokens it was admitted "
                                        f"on: its Program's prefix is bound to the requests that computed it, in the window that "
                                        f"admitted it, which v0 doesn't carry")
                producers = producers_of(i, s0, rows[i][0][2], bs) if C and carry is None else ()
                stop = stop_of(i, q, T, () if params[i].get("ignore_eos") is True else stops, params[i]["max_tokens"], mml)
            except Unexpressible as e:
                refuse(i, str(e))
            else:
                parse = bind(IO.IngressPrompt, MODEL=registered.served_model, LP=LP, V=V) if i in wired else None
                units.append(RequestUnit(i, "", LP, T, policy, seed, s0, q.id, C, producers, parse, carry is not None, i in left,
                                         launch, chunks if len(chunks) > 1 else (), edge, stop))
                placed.update(own)
                continue
        units.append(None)
    for u in units:
        if u is not None and (p := next((p for p in u.producers if p in spent), None)) is not None:
            refuse(u.request, f"request {u.request}: its cache hit reads request {p}'s KV, which has no Program in the window (its rows "
                              f"here are past its last kept output)")
            units[u.request] = None
    while partial and (lost := [(u.request, p) for u in units if u is not None for p in u.producers if units[p] is None][:1]):
        (i, p), = lost
        refuse(i, f"request {i}: its cache hit reads request {p}'s KV, and request {p} is refused")
        units[i] = None
    if all(u is None for u in units):
        if not refused:
            raise Unexpressible("no request has a live row in the advice's steps" + (
                f" (request {min(spent)}'s rows here are past its last kept output)" if spent else ""))
        raise Unexpressible(f"every request is refused, request {min(refused)} first: {refused[min(refused)]}")
    steps: list[list[tuple[int, int, int, int]]] = []
    discarded: list[tuple[int, int, int, int]] = []
    for s, step in enumerate(adv.steps):
        live = []
        for j, row in enumerate(step.rows):
            if units[row.request] is None and row.request not in spent:
                continue
            held, past = placed[s, j]
            if held is not None:
                live.append((row.request, *held))
            discarded += [(s, row.request, start, n) for start, n in past]
        steps.append(live)
    return Plan(units, steps, discarded, dict(sorted(refused.items())), idle)


# ------------------------------------------------------------------------------------------------------------- the traced side


def _label(policy: dict) -> str:
    return "greedy" if policy["kind"] == "greedy" else f"t{policy['temperature']:g}-p{policy['top_p']:g}"


NO_EDGE_TRACE = "no trace: the Build's request wrapper returns its samples alone, never a window's edge rows"


def trace_dir(u: RequestUnit | dict) -> str:
    """The derive dir name of a request Program's trace: `build_request_<prefix_plan.shape_name>_<policy>`, in class G then `_l` and
    the first 12 hex digits of the SHA-256 of its launch context's compact JSON (with its chunks, `{"batch": …, "chunks": …}`), and
    with a length rule `_s` and the first 12 of its config's (`RequestUnit.stop`)."""
    LP, T, C, policy, launch, chunks, stop = ((u.LP, u.T, u.C, u.policy, u.batch(), list(u.chunks), u.stop) if isinstance(u, RequestUnit) else
                                              (u["LP"], u["T"], u.get("C") or 0, u["policy"], u.get("launch"), u.get("chunks"), u.get("stop")))
    ctx = {"batch": launch, "chunks": chunks} if chunks else launch
    tail = f"_l{hashlib.sha256(json.dumps(ctx, separators=(',', ':')).encode()).hexdigest()[:12]}" if launch else ""
    tail += f"_s{hashlib.sha256(json.dumps(stop, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:12]}" if stop else ""
    return f"build_request_{PP.shape_name(PP.shape(LP, T, C))}_{_label(policy)}{tail}"


def trace_commands(summary: dict, registered: Registered, root: str, *, max_model_len: int = 2048) -> list[list[str]]:
    """One `verity-vllm build --wrapper request` per distinct request Program of a compilation (`Compiled.summary()`), on the pinned
    vLLM (CPU, on meta), each writing `<root>/<trace_dir>` at a `max_model_len` of at least its LP + T, under φ's GPU's name where
    φ states one (`--device-name`: the tuned kernel configs a GPU-free host reads); at tensor-parallel world W, φ's rank's
    (`--tp W --rank r --dead-free`); in class G batched (`--batched`) at the unit's launch context (`--launch-context '{"batch": ...}'`), with its
    prefill chunks there when it has them (`'{"batch": ..., "chunks": ...}'`: the Build's chunked request wrapper,
    `vllm_meta.ChunkedGreedyRequest`); its length rule `--stop`. A Program with a window's edge rows has none (`NO_EDGE_TRACE`),
    so `compare` counts it `no_trace`."""
    from verity_vllm.program.frontend.target_profile import TargetProfile
    phi, seen, cmds = registered.phi, set(), []
    target = {k: v for k, v in phi.target.items() if k in TargetProfile.__dataclass_fields__}
    # the target's GPU name, which vLLM's get_device_name answers on any host (its tuned fused-MoE configs' key)
    device = ["--device-name", str(phi.target["device_name"])] if phi.target.get("device_name") else []
    world, batched = int(phi.tp.get("world", 1)), ([] if phi.target_profile().batch_invariant else ["--batched"])
    tp = ["--tp", str(world), "--rank", str(int(phi.tp.get("rank", 0))), "--dead-free"] if world > 1 else []
    for u in summary["requests"]:
        if u is None or u.get("edge") or (d := trace_dir(u)) in seen:
            continue
        seen.add(d)
        pol, ctx = u["policy"], {"batch": u.get("launch"), **({"chunks": list(u["chunks"])} if u.get("chunks") else {})}
        sampling = "greedy" if pol["kind"] == "greedy" else json.dumps({k: pol[k] for k in ("temperature", "top_p", "top_k")}, sort_keys=True)
        cmds.append(["verity-vllm", "build", "--out", os.path.join(root, d), "--wrapper", "request", "--LP", str(u["LP"]), "--T", str(u["T"]),
                     *(["--cached", str(u["C"])] if u.get("C") else []),
                     *(["--launch-context", json.dumps(ctx, separators=(",", ":"))] if u.get("launch") else []),
                     *(["--stop", json.dumps(u["stop"], sort_keys=True, separators=(",", ":"))] if u.get("stop") else []),
                     "--model", phi.model["repo"], "--revision", phi.model["revision"],
                     "--max-model-len", str(max(max_model_len, int(u["LP"]) + int(u["T"]))),
                     "--target", json.dumps(target, sort_keys=True), *device, "--profile", phi.profile, "--sampling", sampling, *tp, *batched])
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
        why = (NO_EDGE_TRACE if u.get("edge") else None) if not doc else str(doc.get("message") or doc.get("unsupported"))[:400]
        out.append({"unit": d, "compiled": u["digest"], "traced": traced,
                    "outcome": "no_trace" if not traced else ("equal" if traced == u["digest"] else "different"),
                    **({} if traced or why is None else {"trace_error": why})})
    n = {k: sum(1 for x in out if x["outcome"] == k) for k in ("equal", "different", "no_trace")}
    return {"schema": SCHEMA + "/comparison", "compiler": summary["compiler"], "units": out, **n, "mismatches": n["different"] + n["no_trace"]}


compile_steps = cyclic_gc_off(compile_steps)
