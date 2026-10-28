"""A tensor-parallel node's Program: every rank's forward in one body, each collective reading the ranks' own partials.

    build_node_program(phi, wrapper, *, M, pos0 | LP, T) -> NodeResult
    node_parameters(phi, wrapper, *, M, pos0 | LP, T) -> NodeParameters      its parameters alone, no forward built

A rank's Program (`build.build_program`) takes the other ranks' partials as Inputs, world - 1 `peers_<k>` per collective, so an
isolation block over it reads them as novel: 3.1·10¹⁰ bits a GLM-5.3 B1 rank. The node's Program has no
peer Input. It runs the world's rank forwards, each the one its rank's Program runs (`build.decoder`, `build.drive`), into one
Builder, one rank at a time in rank order, each up to its next collective. Collective k of every rank then reads every rank's own
rows at k, in rank order, as `tp_links.json`'s `binding` states the rank's peer Inputs. Joined call by call, the ranks' Programs
would be a cycle: a rank's collective k reads the other ranks' partials from the middle of their own forwards.

Its parameters are the prompt (and a stochastic request's seed and splits) once, then each rank's schedule words where its
decoder reads them (`rank<r>.schedule`), then each rank's weights, one parameter per field of its weights record
(`rank<r>.weights[i]`, as a workload Program's `shared.weights[i]` are), bound at registration and marked θ. It returns every
rank's samples, rank-major: each rank samples from the same gathered logits and feeds its own sample to its next forward.

The body is one Definition (`NodeResult.node_fn`), and the Program's function (`Node`, bound to it) calls it once, so the root is
the parameters' input nodes and one Call. `NciCheck.Desc` summarizes a Definition once and keeps a root node's reads as pieces of
gates, so it reads the node through that Call; a root holding the body's ~10⁷ nodes itself would be ~10¹⁰ pieces on GLM-5.3 B1.
`NodeResult.spans` says which rank emitted each run of the body's nodes.
"""
from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from verity.ml.circuits.codec import canonical_json, encode_program, program_digest
from verity.ml.circuits.defs import Builder, CompositeDefinition, bind
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import Affine, Coll, concat
from verity.ml.circuits.types import Array, Record, Type, Value

from .build import I32, I64, WRAPPERS, BUILDER_ID, BuildResult, _Model, decoder, drive
from .common import Unexpressible, _arr_shape, _Launch, _Peers, _Stochastic, launch_batch, launch_rows
from .phi import Phi


@dataclass
class NodeResult(BuildResult):
    world: int = 1
    theta: tuple[str, ...] = ()
    #: the body's Definition, which the root calls once
    node_fn: Any = None
    #: (rank, first body node, end) of each run of the body's nodes, in body order
    spans: list[tuple[int, int, int]] = field(default_factory=list)

    def summary(self) -> dict:
        return {**super().summary(), "world": self.world, "theta_parameters": len(self.theta), "node_fn": self.node_fn.id,
                "body_nodes": len(self.node_fn.body.nodes), "spans": len(self.spans)}

    def rank_of(self) -> list[int]:
        """The rank that emitted each body node."""
        out = [-1] * len(self.node_fn.body.nodes)
        for r, lo, hi in self.spans:
            out[lo:hi] = [r] * (hi - lo)
        return out


def check_ranks(phi: Phi, node: NodeResult, **shape) -> list[dict]:
    """Each rank's run of `node`'s body against its own rank Program's body (`build.build_program`, built one rank at a time):
    node for node, the same form and Definition. Raises on the first rank that differs."""
    from .build import build_program
    rank_of, nodes, out = node.rank_of(), node.node_fn.body.nodes, []
    for r in range(node.world):
        t0 = time.time()
        own = build_program(Phi.from_json({**phi.to_json(), "tp": {**phi.tp, "rank": r}}), node.wrapper, with_desc=False, **shape)
        mine = [(n.form, n.fn.id) for n, q in zip(nodes, rank_of) if q == r]
        theirs = [(n.form, n.fn.id) for n in own.program.fn.body.nodes]
        if mine != theirs:
            at = next((i for i, (a, b) in enumerate(zip(mine, theirs)) if a != b), min(len(mine), len(theirs)))
            raise AssertionError(f"rank {r}: the node's run of {len(mine)} nodes differs from its Program's {len(theirs)} at node {at}")
        out.append({"rank": r, "nodes": len(theirs), "gates": sum(n.gates for n in own.program.fn.body.nodes),
                    "secs": round(time.time() - t0, 1)})
        del own
    return out


class _Abort(Exception):
    """Another rank's forward failed; this one stops where it waits."""


class _Lockstep:
    """The ranks' forwards, one running at a time, in rank order, each up to its next collective or its end."""

    def __init__(self, B: Builder, world: int) -> None:
        self.B, self.world, self.cv = B, world, threading.Condition()
        self.turn: int | None = 0
        self.posted: list[list] = []
        self.taken = [0] * world
        self.done = [False] * world
        self.error: BaseException | None = None
        self.spans: list[tuple[int, int, int]] = []
        self._from = 0

    def _yield(self, r: int) -> None:
        if len(self.B.nodes) > self._from:
            self.spans.append((r, self._from, len(self.B.nodes)))
        live = [q for q in list(range(r + 1, self.world)) + list(range(r + 1)) if not self.done[q]]
        self.turn = live[0] if live else None
        self.cv.notify_all()

    def _wait(self, r: int) -> None:
        self.cv.wait_for(lambda: self.error is not None or self.turn == r)
        if self.error is not None:
            raise _Abort
        self._from = len(self.B.nodes)

    def start(self, r: int) -> None:
        with self.cv:
            self._wait(r)

    def take(self, r: int, own: list) -> list[list]:
        """Rank r's collective k: post its own rows, run the other ranks to theirs, then every rank's rows at k, rank order."""
        with self.cv:
            k = self.taken[r]
            self.taken[r] += 1
            if k == len(self.posted):
                self.posted.append([None] * self.world)
            self.posted[k][r] = list(own)
            self._yield(r)
            self._wait(r)
            missing = [q for q, p in enumerate(self.posted[k]) if p is None]
            if missing:
                self.error = RuntimeError(f"rank {missing[0]}'s forward ended after {self.taken[missing[0]]} collectives, while rank {r} "
                                          f"took collective {k + 1}: the ranks' forwards are not one node's")
                self.cv.notify_all()
                raise _Abort
            return self.posted[k]

    def end(self, r: int) -> None:
        with self.cv:
            self.done[r] = True
            self._yield(r)

    def fail(self, e: BaseException) -> None:
        with self.cv:
            if self.error is None:
                self.error = e
            self.cv.notify_all()


class _Joined(_Peers):
    """A rank's view of the others in a node's body: each collective's rows are every rank's own, posted in lockstep."""

    def __init__(self, lockstep: _Lockstep, rank: int, schedule: Coll | None) -> None:
        super().__init__([], lockstep.world, rank, schedule)
        self.lockstep = lockstep

    def take(self, own: list) -> list[list]:
        self.n += self.world - 1
        return self.lockstep.take(self.rank, own)


@dataclass(frozen=True)
class NodeParameters:
    """A node Program's parameters (module docstring) and what they are read from: each rank's φ and decoder, its schedule words
    (0 where its decoder reads none) and their parameter, and its weights record, whose fields are its θ parameters from `weights_at`."""
    phis: list[Phi]
    nets: list
    params: tuple[tuple[str, Type], ...]
    words: list[int]
    sched_at: dict[int, int]
    leaves: list[list[tuple[str, Type]]]
    weights_at: list[int]

    @property
    def theta(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.params[self.weights_at[0]:])

    def fields(self) -> dict[str, tuple[int, str]]:
        """θ parameter -> (its rank, the name of the rank's weights record field it is)."""
        return {self.params[at + i][0]: (r, name) for r, (at, ls) in enumerate(zip(self.weights_at, self.leaves))
                for i, (name, _) in enumerate(ls)}


def node_parameters(phi: Phi, wrapper: str, *, LP: int | None = None, T: int | None = None, M: int = 1,
                    pos0: int = 8) -> NodeParameters:
    """The parameters of φ's world's node Program of `wrapper` (φ's own rank is ignored), as `build_node_program` takes them."""
    world = int(phi.tp.get("world", 1))
    if world < 2:
        raise Unexpressible(f"a node Program joins a tensor-parallel world's ranks, and φ's world is {world}")
    kind = (phi.sampling or {}).get("kind", "greedy")
    if (wrapper, kind) not in WRAPPERS:
        raise Unexpressible(f"wrapper {wrapper!r} (have {sorted({w for w, _ in WRAPPERS})})")
    phis = [Phi.from_json({**phi.to_json(), "tp": {**phi.tp, "rank": r}}) for r in range(world)]
    nets = [decoder(p, M, pos0 + M if wrapper == "step" else int(LP) + int(T)) for p in phis]
    n = M if wrapper == "step" else int(LP)
    events = 1 if wrapper == "step" else int(T) + 1
    passes = [n] + [1] * (events - 1)
    params: list[tuple[str, Type]] = [("prompt", Array(n, I32))]
    if kind == "stochastic":
        params.append(("seed", Array(1, I64)))
        if phi.sampling["splits"] == "derived":
            params.append(("splits", Array(events, I32)))
    words = [sum(net.schedule_words(r) for r in passes) if hasattr(net, "schedule_words") else 0 for net in nets]
    sched_at = {}
    for r, w in enumerate(words):
        if w:
            sched_at[r] = len(params)
            params.append((f"rank{r}.schedule", Array(w, I32)))
    leaves = [net.leaves() for net in nets]
    weights_at = []
    for r, ls in enumerate(leaves):
        weights_at.append(len(params))
        params += [(f"rank{r}.weights[{i}]", t) for i, (_, t) in enumerate(ls)]
    return NodeParameters(phis, nets, tuple(params), words, sched_at, leaves, weights_at)


def build_node_program(phi: Phi, wrapper: str, *, LP: int | None = None, T: int | None = None, M: int = 1, pos0: int = 8,
                       with_desc: bool = True) -> NodeResult:
    """The node Program of the Build's `wrapper` for φ's world (φ's own rank is ignored): `build.build_program`'s, joined."""
    t0 = time.time()
    P = node_parameters(phi, wrapper, LP=LP, T=T, M=M, pos0=pos0)
    phis, nets, params, words, sched_at, theta = P.phis, P.nets, P.params, P.words, P.sched_at, P.theta
    world, kind = len(phis), (phi.sampling or {}).get("kind", "greedy")
    n = M if wrapper == "step" else int(LP)
    events = 1 if wrapper == "step" else int(T) + 1
    inputs = [list(_arr_shape(t)) for _, t in params[:P.weights_at[0]]]
    W = [Coll(Record(*ls), concat([Affine("p", at + i, 0, t.leaves) for i, (_, t) in enumerate(ls)]))
         for at, ls in zip(P.weights_at, P.leaves)]
    B = Builder(params, Value(1))
    ids = Coll(params[0][1], Affine("p", 0, 0, n))
    policy = None
    if kind == "stochastic":
        from verity_vllm.program.frontend.rules.vllm_sampling import f32_bits
        s = phi.sampling
        policy = _Stochastic(seed=Coll(I64, Affine("p", 1, 0, 1)), splits=Coll(params[2][1], Affine("p", 2, 0, events)) if s["splits"] == "derived" else None,
                             S=s["splits"], temperature_bits=f32_bits(s["temperature"]), top_p_bits=f32_bits(s.get("top_p", 1.0)))
    batch = launch_batch(phis[0], LP, T) if wrapper == "request" else None
    lockstep = _Lockstep(B, world)
    peers = [_Joined(lockstep, r, Coll(params[sched_at[r]][1], Affine("p", sched_at[r], 0, words[r])) if r in sched_at else None)
             for r in range(world)]
    models = []
    for r in range(world):
        model = nets[r].model(B, W[r], peers[r])
        if batch is not None:
            if not getattr(model, "reads_launch", False):
                raise Unexpressible(f"vocabulary cannot express: a request run in a launch batch (no batch invariance, {len(phi.requests)} "
                                    f"requests), whose kernel choices the {nets[r].model_class} decoder reads at its own rows")
            model.launch = _Launch(batch)
        if wrapper == "request" and isinstance(model, _Model):
            model.ctx = launch_rows(phis[r], LP, T)
        models.append(model)
    rows = [Coll(I32, ids.refs.slice(i, i + 1)) for i in range(n)]
    toks: list[list[Coll] | None] = [None] * world

    def run(r: int) -> None:
        try:
            lockstep.start(r)
            toks[r] = drive(models[r], wrapper, rows, M=M, pos0=pos0, LP=LP, T=T, policy=policy, batch=batch)
            lockstep.end(r)
        except _Abort:
            pass
        except BaseException as e:  # noqa: BLE001 - re-raised by the caller, once every rank has stopped
            lockstep.fail(e)

    threads = [threading.Thread(target=run, args=(r,), name=f"node-rank{r}") for r in range(world)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if lockstep.error is not None:
        raise lockstep.error
    if len(set(lockstep.taken)) != 1 or any(p.k != w for p, w in zip(peers, words)):
        raise AssertionError(f"the ranks took {lockstep.taken} collectives and read {[p.k for p in peers]} of {words} schedule words")
    out = Coll(Array(world * events, I32), concat([t.refs for ts in toks for t in ts]))
    B.ret_type = out.type
    body = B.finish(out)
    t_body = time.time()
    src_class, derive_name = WRAPPERS[(wrapper, kind)]
    name = f"{derive_name}Node[{nets[0].model_class}]"
    src_static = {"class": f"verity-vllm/node/v1[{src_class}]", "mode": "profile", "profile": phi.profile, "world": world,
                  "inputs": inputs, "config_sha256": hashlib.sha256(canonical_json({})).hexdigest()[:16]}
    from verity.ml.definitions.replay import replayer
    defn = CompositeDefinition(f"Derived[{name}]", 1, ("SRC",), lambda S, _p=params, _r=out.type: (tuple(_p), _r), replayer(body),
                               doc=f"a tensor-parallel node's ranks joined at their collectives, predicted ahead of time by "
                                   f"verity_vllm.program.predict.node ({BUILDER_ID})",
                               conformance="predicted: each rank's forward is its rank Program's (predict.build)", register=False)
    sd = bind(defn, SRC=src_static)
    sd.set_body(body)
    node = CompositeDefinition("Node", 1, ("body",), lambda S: S.body.signature(),
                               lambda B, S, *args: B.call(S.body, *args, name="node"),
                               doc="a tensor-parallel node's Program function: its body, the ranks' forwards joined, called once",
                               conformance="predicted: its body's", register=False)
    program = Program(bind(node, body=sd), registered=theta, theta=theta)
    timings = {"body_s": round(t_body - t0, 3), "program_s": round(time.time() - t_body, 3)}
    desc = digest = None
    if with_desc:
        t2 = time.time()
        desc = encode_program(program)
        digest = program_digest(desc)
        timings["encode_digest_s"] = round(time.time() - t2, 3)
    timings["total_s"] = round(time.time() - t0, 3)
    return NodeResult(wrapper=wrapper, program=program, digest=digest or "", desc=desc, function=program.fn.id,
                      n_root_nodes=len(program.root.body.nodes), timings=timings, world=world, theta=theta, node_fn=sd,
                      spans=list(lockstep.spans))
