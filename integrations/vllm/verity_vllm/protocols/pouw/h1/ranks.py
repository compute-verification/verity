"""H1 on a multiprocess engine, tensor- and pipeline-parallel (TP N × PP M on Model Runner V2's multiprocess executor): the
window held whole in the engine-core process (`Holder`, an unchanged `pouw_h1.H1Run` over every rank's linears) and each
rank's calls recorded in its own worker (`RankRun`, through the engine's ``worker_extension_cls``, `RankExtension`).

The window's linears are every rank's, rank-major: the ranks in (pp, tp) order, each rank's linears in its module order, so
rank r holds the contiguous range [lo_r, lo_r + L_r) of the window's L. A rank's linear is its shard: a column-parallel
one (qkv, gate_up) its rows of W, n/TP × K; a row-parallel one (o, down) its K slice, n × K/TP, whose z is the rank's
partial sums before vLLM's all-reduce (`engine.int8_vllm`). Call (s, w) is index s·L + w at position s·U + off(w), so the
window has one static order over every rank, with each rank's A and z tables committed where its rows are, under the
window's salt and hm96 key. At TP 1 × PP M the window's linears and layout are PP = 1's.

Admission stays with the scheduler: the holder wraps the executor's ``execute_model``, and before each step with
scheduled tokens it pulls every rank's calls so far (`RankExtension.h1rank_rows`, an RPC queued behind the earlier steps,
so it returns once they have run on every rank), takes them in index order, closing each round whose last linear is in,
then admits the step (`H1Run.admit`): one round a step, its root over every rank's calls before its coin, which only the
holder draws. The salt and key reach the ranks before the first forward (`Holder._push`). The pull blocks the engine
core until the previous step has run everywhere, so steps don't overlap in a pipeline under H1.

With ``served`` (`pouw_h1_served.Served`) the window's registered tables are the holder's: step s's A table of K holds
every call of K in step s, rank-major, so at TP > 1 each K's table spans the ranks, and ``b/<K>`` every rank's linears of
K. Each rank ships each call's A rows and scales, its z/y records and, once, its weight with the call (`RankRun`'s
``ship``), and the holder's one `Served` registers them as one process would (`ingest`). It registers the window's own
``b/<K>``: the period's registration of a rank's shards (`Weights`, a registry) is refused here, since no public rule
slices the pinned checkpoint into shards yet.

With ``commit`` (ROOTS1) the Commit runs on every rank (`acquire.h1_commit.RankCommit`, the engine's
``worker_extension_cls`` then: the rows-sha512 committer, the int8 source and the step-input port on the rank's model),
and the rank's tap (`RankTap`) ships what it acquired to the holder: each H1 linear's x_q and s_a at its site, renamed to
the window's name of the linear (``…qkv_proj@tp1/row_quant``), and the port's input ids, cut to the step's scheduled
tokens. The holder's one `pouw_h1_commit.CommitTap` takes them before the calls (`feed`), so its
AROW of each (step, K) is every rank's calls' rows, call after call, as the window's A table is. The input ids are the
first rank's (the one holding linear 0), one ``ids@<s>`` a round for the window; another rank's ids at the same engine
step that differ are refused."""
from __future__ import annotations

import importlib
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from verity.core.protocols.pouw import window as Wd
from verity.core.protocols.pouw.audit import Epoch, Workload, weights_root
from verity.core.protocols.pouw.protocol import Matmul, Shape
from verity.core.protocols.sampled_proofs.challenger import LiveRecord

from verity_vllm.engine import int8_ranks  # noqa: F401  (a worker registers the int8 parts before it builds its model)
from verity_vllm.engine.hooks import Hooks
from verity_vllm.protocols.pouw import h1 as PH
from verity_vllm.protocols.pouw.h1 import commit as CT
from verity_vllm.protocols.pouw.h1 import hooks as PHK
from verity_vllm.protocols.pouw.h1 import requests as RQ

ENTRY = f"pouw:{PH.SCHEME}"
EXTENSION = "verity_vllm.protocols.pouw.h1.ranks.RankExtension"
#: with ``install``'s ``commit``: `RankExtension` with the Commit on each rank (in ``acquire``, which builds the int8 source)
COMMIT_EXTENSION = "verity_vllm.acquire.h1_commit.RankCommit"
#: the module of Model Runner V2, whose `sort_batch_req_ids` orders a step's rows (`pouw_h1_requests.batch_order`)
MRV2 = "vllm.v1.worker.gpu.model_runner"


class RankRun(PH.H1Run):
    """One rank's side of the window: the calls of its linears [lo, hi) of the window's L, each at its global index
    s·L + w with its tables, under the salt the holder pushes (`RankExtension.h1rank_open`). A linear the rank's step
    already holds starts its next step (`H1Run.record`); rounds, admission and coins are the holder's. With ``ship`` each
    call's rows go to the holder's `Served` with it (`shipped`): its A rows and scales and its z/y records on the host, and
    its weight and scales when the call's weight is not the one last shipped for its linear."""

    def __init__(self, entry: str, modules: Mapping[int, str], shapes: Sequence[tuple[int, int]], rows: int, steps: int,
                 lo: int, hi: int, key: bytes, ship: bool = False) -> None:
        super().__init__(entry, None, modules, shapes, rows, steps, key=key, opener=self._unpushed)
        self.lo, self.hi, self.ship = int(lo), int(hi), bool(ship)
        self.shipped: dict[int, dict[str, Any]] = {}
        self._held: dict[int, Any] = {}

    def _unpushed(self, _root: bytes) -> Any:
        raise RuntimeError(f"{self.entry}: rank [{self.lo}, {self.hi}) used the window's salt before its holder pushed it")

    def _close_rounds(self, through: int) -> None:
        return None

    def _call(self, s: int, w: int, a: Any, s_a: Any, weight: Any, weight_scale: Any, records: Any = None,
              zy: Any = None) -> PH.Call:
        call = super()._call(s, w, a, s_a, weight, weight_scale, records, zy)
        if self.ship and self.commit:
            new = weight is not None and self._held.get(w) is not weight
            if new:
                self._held[w] = weight
            self.shipped[call.index] = {
                "a": None if a is None else np.array(PH.numpy_of(a)),
                "s_a": None if s_a is None else np.array(PH.numpy_of(s_a)), "records": records,
                "weight": np.array(PH.numpy_of(weight)) if new else None,
                "weight_scale": np.array(PH.numpy_of(weight_scale)) if new and weight_scale is not None else None}
        return call

    def rows_since(self, sent: int) -> list[dict[str, Any]]:
        """The calls from the `sent`-th on, each with its tables, its credit verdict, what its listeners bound and, with
        ``ship``, its rows (``served``)."""
        return [{"call": asdict(c), "tables": self.tables.get(c.index), "verdict": self.verdicts.get(c.index),
                 "bound": self.bound.get(c.index), **({"served": self.shipped.pop(c.index, None)} if self.ship else {})}
                for c in self.calls[sent:]]


@dataclass
class RankTap(CT.CommitTap):
    """A rank's tap on its Commit (`acquire.h1_commit.RankCommit.h1rank_commit`): what the holder's `pouw_h1_commit.CommitTap` reads of what
    the Commit acquired, each H1 linear's x_q (``/0``) and s_a (``/1``) at its site and the port's input ids, its first
    ``n`` (None with no count of the step's scheduled tokens), copied to the host with the engine step, each site under the
    window's name of its linear (``names``: the rank's module name, or a routed expert's H1 name
    ``prefix.expert.projection``, to the window's), kept until the holder's next pull (`drain`)."""

    names: dict[str, str] = field(default_factory=dict)
    out: list[tuple[str, int, np.ndarray | None]] = field(default_factory=list)

    def take(self, name: str, t: Any, step: int, n: int | None = None) -> None:
        if self.port is not None and name == self.port:
            self.out.append((name, int(step), None if n is None else np.array(CT._host(t)).reshape(-1)[:n]))
            return
        op, _, port = name.rpartition("/")
        module = op.removesuffix("/row_quant")
        if module != op and port in ("0", "1"):
            self.out.append((f"{CT.site(self.names.get(module, module))}/{port}", int(step), np.array(CT._host(t))))

    def drain(self) -> list[tuple[str, int, np.ndarray]]:
        out, self.out = self.out, []
        return out


def feed(tap: CT.CommitTap, acquired: Sequence[tuple[str, int, Any]], first: bool) -> None:
    """One rank's acquisitions (`RankTap.drain`) into the window's tap, in the order the rank acquired them: its sites' x_q
    and s_a, and with ``first`` (the rank holding linear 0) the port's input ids, which every other rank's at the same
    engine step must equal (refused otherwise)."""
    for name, step, t in acquired:
        if name != tap.port:
            tap.take(name, t, int(step))
        elif first:
            tap.take(name, t, int(step), None if t is None else len(t))
        elif t is None or (got := tap._ids.get(int(step))) is None or not np.array_equal(got, np.asarray(t).astype(np.int64)):
            tap.refused.append(f"engine step {step}: a rank's input ids are not the first rank's")


def wire_rank(run: RankRun, hooks: Any, layers: list[Any], rows: int) -> None:
    """`pouw_h1_hooks.wire` on a rank: its layer i is the window's linear lo + i; no runner hook (admission is the
    holder's)."""
    for i, layer in enumerate(layers):
        w = run.lo + i
        hooks.wrap(layer.quant_method, "observe", lambda orig, w=w: PHK.observer(run, w, orig))
        hooks.wrap(layer.quant_method, "apply", lambda orig, w=w: PHK.traced(run, w, orig))
        if run.commit and hasattr(type(layer.quant_method), "zy_rows") and layer.weight.shape[0] % PH.TILE[1] == 0:
            hooks.patch(layer.quant_method, "zy_rows", rows)


class RankExtension:
    """The ``worker_extension_cls`` of a multiprocess engine serving H1: each worker's methods, called from the holder with
    ``collective_rpc``; importing it registers the int8 parts in the worker (`engine.int8_ranks`)."""

    def _h1rank_layers(self) -> tuple[Any, list[Any]]:
        from verity_vllm.protocols import pouw
        model = self.model_runner.get_model()  # type: ignore[attr-defined]
        return model, PHK.layers(model, pouw.linear_types())

    def _h1rank_rank(self) -> tuple[int, int]:
        from vllm.distributed.parallel_state import get_pp_group, get_tp_group
        return get_pp_group().rank_in_group, get_tp_group().rank_in_group

    def h1rank_layout(self) -> dict[str, Any]:
        """This worker's (pp, tp) rank, its linears' shard shapes and module names (local ordinals) in module order, and its
        runner's decode_query_len."""
        from verity_vllm.protocols import pouw
        model, layers = self._h1rank_layers()
        pp, tp = self._h1rank_rank()
        return {"pp": pp, "tp": tp, "shapes": [[int(d) for d in m.weight.shape] for m in layers],
                "names": pouw._named(model, layers),
                "decode_query_len": int(self.model_runner.decode_query_len)}  # type: ignore[attr-defined]

    def h1rank_install(self, entry: str, los: Mapping[tuple[int, int], int], shapes: Sequence[Sequence[int]],
                       modules: Mapping[int, str], rows: int, steps: int, key: bytes, ship: bool = False) -> dict[str, Any]:
        """This rank's `RankRun` from linear ``los[(pp, tp)]`` on, its hooks wired (``ship``: its calls' rows go to the
        holder's `Served`); returns its weights' roots by global index (`pouw_h1.weight_root`, as `pouw.commit_weights`)."""
        _, layers = self._h1rank_layers()
        lo = int(los[self._h1rank_rank()])
        run = RankRun(entry, modules, [tuple(s) for s in shapes], rows, steps, lo, lo + len(layers), key, ship=ship)
        hooks = Hooks(tag=f"{entry}/rank{lo}")
        wire_rank(run, hooks, layers, rows)
        self._h1rank = {"run": run, "hooks": hooks, "sent": 0}
        return {"lo": lo, "roots": {lo + i: PH.weight_root(lo + i, PH.numpy_of(m.weight)).hex()
                                    for i, m in enumerate(layers)}}

    def h1rank_open(self, salt: bytes) -> None:
        self._h1rank["run"]._salt = bytes(salt)

    def h1rank_rows(self) -> dict[str, Any]:
        """The rank's calls since the last pull, its refusals so far, whether each step's calls came in its first step's
        order and, with the Commit, what it acquired since (`RankTap.drain`)."""
        st = self._h1rank
        run: RankRun = st["run"]
        out = run.rows_since(st["sent"])
        st["sent"] = len(run.calls)
        commit = st.get("commit")
        return {"lo": run.lo, "rows": out, "refused": list(run.refused), "order_ok": run.order_ok,
                "unopened": run._unopened, **({"acquired": commit[1].drain()} if commit is not None else {})}

    def h1rank_release(self) -> dict[str, Any] | None:
        """With the Commit, the rank's committer finalized and closed, then its hooks removed: newest first, as the Commit's
        source wraps a routed expert's ``apply`` over H1's (`hooks.Patch.uninstall`). Returns the rank's Commit (its run
        root and scheme, `pouw_h1_commit.committed`, which refuses any but rows-sha512's, its tensors and errors), else
        None."""
        st = getattr(self, "_h1rank", None)
        if st is None:
            return None
        out = None
        try:
            if (commit := st.get("commit")) is not None:
                com, tap = commit
                try:
                    rc = com.finalize()
                    out = {"lo": st["run"].lo, **CT.committed(com, rc, ("expected_tensors", "committed_tensors", "errors_n")),
                           "errors": [str(e)[:300] for e in (rc.extra.get("errors") or [])][:8]}
                finally:
                    com.close()
                    tap.untap()
        finally:
            self._h1rank = None
            st["hooks"].uninstall()
        return out


def executor_of(llm: Any) -> Any:
    """The in-process engine core's executor (``VLLM_ENABLE_V1_MULTIPROCESSING=0``), whose ``collective_rpc`` reaches every
    worker."""
    core = getattr(llm.llm_engine, "engine_core", None)
    inner = getattr(core, "engine_core", None) or core
    executor = getattr(inner, "model_executor", None)
    if executor is None or not callable(getattr(executor, "collective_rpc", None)):
        raise RuntimeError("pouw_h1_ranks: no in-process executor with collective_rpc (is VLLM_ENABLE_V1_MULTIPROCESSING=0?)")
    return executor


def _runner(decode_query_len: int) -> Any:
    """What `pouw_h1_requests.batch_order` reads of the ranks' Model Runner V2: its module's batch order (once `install`
    has imported it; `pouw_h1_requests.sort_batch_req_ids` until then) and its decode_query_len."""
    return type("GPUModelRunner", (), {"__module__": MRV2, "decode_query_len": int(decode_query_len)})()


class Holder:
    """The window of a multiprocess engine (`install`): its `H1Run` (``run``) fed by the ranks and admitted at the executor;
    `finish` pulls the last calls, closes and releases the window and the ranks' hooks."""

    def __init__(self, run: PH.H1Run, executor: Any, layout: list[dict[str, Any]], hooks: Hooks) -> None:
        self.run, self.executor, self.layout, self.hooks = run, executor, layout, hooks
        self.runner = _runner(layout[0]["decode_query_len"])
        self.ranks: dict[int, dict[str, Any]] = {}
        self.pushed = False
        self.pulls = 0
        self.committed: dict[int, dict[str, Any]] = {}

    def _push(self) -> None:
        if not self.pushed and self.run._salt is not None:
            self.executor.collective_rpc("h1rank_open", kwargs={"salt": self.run._salt})
            self.pushed = True

    def pull(self) -> int:
        """Every rank's calls since the last pull, taken in index order (`ingest`), after what each rank's Commit acquired
        is in the window's tap (`feed`, first rank first); returns how many."""
        outs = sorted(self.executor.collective_rpc("h1rank_rows"), key=lambda o: o["lo"])
        self.pulls += 1
        rows = []
        for o in outs:
            self.ranks[o["lo"]] = {k: o[k] for k in ("refused", "order_ok", "unopened")}
            rows += o["rows"]
        tap = None if self.run.served is None else self.run.served.tap
        if tap is not None:
            got = [o.get("acquired") or [] for o in outs]
            if len(steps := {int(x[1]) for a in got for x in a}) > 1:
                tap.refused.append(f"one pull held engine steps {sorted(steps)}: the window's tap keeps one step a site")
            for i, a in enumerate(got):
                feed(tap, a, first=i == 0)
        ingest(self.run, rows)
        return len(rows)

    def admitting(self, orig: Any) -> Any:
        def execute_model(scheduler_output: Any, non_block: bool = False) -> Any:
            if int(getattr(scheduler_output, "total_num_scheduled_tokens", 0) or 0) and not self.run.closed:
                self.pull()
                self.run.admit(RQ.new_requests(scheduler_output), RQ.batch_order(self.runner, scheduler_output))
                self._push()
            return orig(scheduler_output, non_block)
        return execute_model

    def close(self) -> None:
        if not self.run.closed:
            self.pull()
            self.run.refused[:] = [r for lo in sorted(self.ranks) for r in self.ranks[lo]["refused"]]
            self.run.order_ok = self.run.order_ok and all(s["order_ok"] for s in self.ranks.values())
            self.run.details["ranks"] = {str(lo): {"refused": len(s["refused"]), "order_ok": s["order_ok"],
                                                   **({"unopened": s["unopened"]} if s["unopened"] else {})}
                                         for lo, s in sorted(self.ranks.items())}
            self.run.details["pulls"] = self.pulls
            self.run.close()

    def finish(self) -> None:
        """`close`, the window finished, and every rank's hooks released; with the Commit, each rank's committer
        finalized (``committed``, by first linear)."""
        try:
            self.close()
            self.run.finish()
        finally:
            self.hooks.uninstall()
            outs = self.executor.collective_rpc("h1rank_release")
            self.committed = {o["lo"]: o for o in outs or () if o is not None}


def ingest(run: PH.H1Run, rows: Sequence[Mapping[str, Any]]) -> None:
    """Rank calls into the holder's `run`, in index order, as `H1Run._call` takes them: each call, its tables, verdict and
    bindings, and with ``served`` its rows registered (`pouw_h1_served.Served.commit`); a round closes with its step's last
    linear."""
    last = len(run.shapes_) - 1
    for r in sorted(rows, key=lambda r: r["call"]["index"]):
        c = PH.Call(**r["call"])
        if run.calls and c.index <= run.calls[-1].index:
            raise RuntimeError(f"{run.entry}: rank call {c.index} after call {run.calls[-1].index}")
        run.calls.append(c)
        for into, key in ((run.tables, "tables"), (run.verdicts, "verdict"), (run.bound, "bound")):
            if r.get(key):
                into[c.index] = dict(r[key])
        if run.served is not None and run.commit:
            got = r.get("served")
            if got is None:
                run.served.refused.append(f"call {c.index}: its rank shipped no rows for the window's registered tables")
                got = {}
            run.tables.setdefault(c.index, {})["registered"] = run.served.commit(
                run, c, got.get("a"), got.get("s_a"), got.get("weight"), got.get("weight_scale"), got.get("records"))
        run.step = c.step
        run.details["calls"] = run.commits["calls"] = len(run.calls)
        if c.layer == last:
            run._close_rounds(c.step)


def install(llm: Any, *, rows: int, steps: int, coins: Any = None, rounds: Any = None, opener: Any = None,
            admission: str = RQ.ROUND0, transcript_dir: str | None = None, key: bytes | None = None,
            entry: str = ENTRY, executor: Any = None, served: Any = None,
            commit: Mapping[str, Any] | None = None) -> Holder:
    """H1 on an engine built with ``worker_extension_cls`` = `EXTENSION`: the ranks' layout gathered, each rank's `RankRun`
    installed, the window's salt from ``coins`` (an in-process `pouw.coins.Challenger`, as `pouw._install_h1` without a
    challenger) over every rank's weight roots, or opened by ``opener`` at round 0 (the challenger's), and the executor's
    ``execute_model`` wrapped (`Holder.admitting`). ``rounds`` is the round coins' `LiveRecord`, its draw family the
    window's draw coin when ``coins`` are given. A module is named by its rank at TP > 1 (``…qkv_proj@tp1``). ``served``
    (`pouw_h1_served.Served`, without the period's `Weights` or a registry) registers the window's tables from the rows
    the ranks ship; ``commit`` (with ``served``, at PP 1, the engine's ``worker_extension_cls`` `COMMIT_EXTENSION`)
    puts the Commit on every rank, its keywords `acquire.h1_commit.RankCommit.h1rank_commit`'s, and ``served``'s tap is the
    window's `pouw_h1_commit.CommitTap`."""
    if (coins is None) == (opener is None):
        raise ValueError(f"{entry}: the window's salt is from coins or opened by its opener, one of the two")
    if served is not None and (served.weights is not None or served.registry is not None):
        raise ValueError(f"{entry}: a period's registered weights or a registry across ranks: no public rule slices the "
                         "checkpoint into the ranks' shards, so the window registers its own b/<K>")
    if commit is not None and served is None:
        raise ValueError(f"{entry}: the Commit's values tie to the window's registered tables, so it needs served")
    if llm is not None:
        importlib.import_module(MRV2)
    executor = executor if executor is not None else executor_of(llm)
    layout = sorted(executor.collective_rpc("h1rank_layout"), key=lambda o: (o["pp"], o["tp"]))
    if len({o["decode_query_len"] for o in layout}) != 1:
        raise ValueError(f"{entry}: the ranks' runners disagree on decode_query_len")
    tp = len({o["tp"] for o in layout})
    if commit is not None and len({o["pp"] for o in layout}) > 1:
        raise ValueError(f"{entry}: the Commit across pipeline stages: a later stage acquires no input ids")
    los, shapes, modules = {}, [], {}
    for o in layout:
        lo = los[(o["pp"], o["tp"])] = len(shapes)
        modules |= {lo + int(w): name + (f"@tp{o['tp']}" if tp > 1 else "") for w, name in o["names"].items()}
        shapes += [tuple(s) for s in o["shapes"]]
    modules = {w: f"linear {w}" for w in range(len(shapes))} | modules
    key = key if key is not None else os.urandom(32)
    roots = {}
    for o in executor.collective_rpc("h1rank_install", kwargs={"entry": entry, "los": los, "shapes": shapes,
                                                               "modules": modules, "rows": rows, "steps": steps, "key": key,
                                                               "ship": served is not None}):
        roots |= {int(w): bytes.fromhex(r) for w, r in o["roots"].items()}
    if sorted(roots) != list(range(len(shapes))):
        raise RuntimeError(f"{entry}: the ranks' weight roots cover {len(roots)} of {len(shapes)} linears")
    declared = Workload(tuple(Matmul(w, Shape(16, k, n), w) for w, (n, k) in enumerate(shapes)))
    salt, draw, commits = None, None, {}
    if coins is not None:
        commitment, opening = coins.commitment, coins.open("salt", weights_root(roots))
        salt = PH.program_salt(Epoch.start(declared, roots, commitment, opening).salt)
        draw = LiveRecord(rounds.challenger, Wd.DRAW) if rounds is not None else None
        commits = {"coins": commitment.to_json(), "salt_opening": opening.to_json(),
                   **({"round_coins": rounds.challenger.commitment.to_json()} if rounds is not None else {})}
    source = PH.CHALLENGER if opener is not None else "in-process"
    run = PH.H1Run(entry, salt, modules, shapes, rows, steps, transcript_dir=transcript_dir, rounds=rounds, key=key,
                   draw=draw, admission=admission, coin_source=source, opener=opener, served=served)
    if commit is not None:
        ports = {o["port"] for o in executor.collective_rpc("h1rank_commit", kwargs=dict(commit))}
        if len(ports) != 1:
            raise RuntimeError(f"{entry}: the ranks' Commits acquire input ids under {sorted(map(str, ports))}, not one port")
        served.tap = CT.CommitTap(port=ports.pop())
    run.commits.update(weights_root=weights_root(roots).hex(), coin_source=source, **commits,
                       ranks=[{"pp": o["pp"], "tp": o["tp"], "first_linear": los[(o["pp"], o["tp"])],
                               "linears": len(o["shapes"])} for o in layout])
    hooks = Hooks(tag=f"{entry}/holder")
    holder = Holder(run, executor, layout, hooks)
    holder._push()
    hooks.wrap(executor, "execute_model", holder.admitting)
    return holder


__all__ = ["COMMIT_EXTENSION", "ENTRY", "EXTENSION", "Holder", "RankExtension", "RankRun", "RankTap", "executor_of", "feed", "ingest", "install",
           "wire_rank"]
