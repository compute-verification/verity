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

Refused here: ``served`` (`pouw_h1_served.Served`, whose b/<K> needs every linear's weight in one process)."""
from __future__ import annotations

import importlib
import os
from dataclasses import asdict
from typing import Any, Mapping, Sequence

from verity.core.protocols.pouw import window as Wd
from verity.core.protocols.pouw.audit import Epoch, Workload, weights_root
from verity.core.protocols.pouw.protocol import Matmul, Shape
from verity.core.protocols.sampled_proofs.challenger import LiveRecord

from verity_vllm.engine import int8_ranks  # noqa: F401  (a worker registers the int8 parts before it builds its model)
from verity_vllm.engine.hooks import Hooks
from verity_vllm.protocol_options import pouw_h1 as PH
from verity_vllm.protocol_options import pouw_h1_hooks as PHK
from verity_vllm.protocol_options import pouw_h1_requests as RQ

ENTRY = f"pouw:{PH.SCHEME}"
EXTENSION = "verity_vllm.protocol_options.pouw_h1_ranks.RankExtension"
#: the module of Model Runner V2, whose `sort_batch_req_ids` orders a step's rows (`pouw_h1_requests.batch_order`)
MRV2 = "vllm.v1.worker.gpu.model_runner"


class RankRun(PH.H1Run):
    """One rank's side of the window: the calls of its linears [lo, hi) of the window's L, each at its global index
    s·L + w with its tables, under the salt the holder pushes (`RankExtension.h1rank_open`). A linear the rank's step
    already holds starts its next step (`H1Run.record`); rounds, admission and coins are the holder's."""

    def __init__(self, entry: str, modules: Mapping[int, str], shapes: Sequence[tuple[int, int]], rows: int, steps: int,
                 lo: int, hi: int, key: bytes) -> None:
        super().__init__(entry, None, modules, shapes, rows, steps, key=key, opener=self._unpushed)
        self.lo, self.hi = int(lo), int(hi)

    def _unpushed(self, _root: bytes) -> Any:
        raise RuntimeError(f"{self.entry}: rank [{self.lo}, {self.hi}) used the window's salt before its holder pushed it")

    def _close_rounds(self, through: int) -> None:
        return None

    def rows_since(self, sent: int) -> list[dict[str, Any]]:
        """The calls from the `sent`-th on, each with its tables, its credit verdict and what its listeners bound."""
        return [{"call": asdict(c), "tables": self.tables.get(c.index), "verdict": self.verdicts.get(c.index),
                 "bound": self.bound.get(c.index)} for c in self.calls[sent:]]


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
        from verity_vllm.protocol_options import pouw
        model = self.model_runner.get_model()  # type: ignore[attr-defined]
        return model, PHK.layers(model, pouw.linear_types())

    def _h1rank_rank(self) -> tuple[int, int]:
        from vllm.distributed.parallel_state import get_pp_group, get_tp_group
        return get_pp_group().rank_in_group, get_tp_group().rank_in_group

    def h1rank_layout(self) -> dict[str, Any]:
        """This worker's (pp, tp) rank, its linears' shard shapes and module names (local ordinals) in module order, and its
        runner's decode_query_len."""
        from verity_vllm.protocol_options import pouw
        model, layers = self._h1rank_layers()
        pp, tp = self._h1rank_rank()
        return {"pp": pp, "tp": tp, "shapes": [[int(d) for d in m.weight.shape] for m in layers],
                "names": pouw._named(model, layers),
                "decode_query_len": int(self.model_runner.decode_query_len)}  # type: ignore[attr-defined]

    def h1rank_install(self, entry: str, los: Mapping[tuple[int, int], int], shapes: Sequence[Sequence[int]],
                       modules: Mapping[int, str], rows: int, steps: int, key: bytes) -> dict[str, Any]:
        """This rank's `RankRun` from linear ``los[(pp, tp)]`` on, its hooks wired; returns its weights' roots by global
        index (`pouw_h1.weight_root`, as `pouw.commit_weights`)."""
        _, layers = self._h1rank_layers()
        lo = int(los[self._h1rank_rank()])
        run = RankRun(entry, modules, [tuple(s) for s in shapes], rows, steps, lo, lo + len(layers), key)
        hooks = Hooks(tag=f"{entry}/rank{lo}")
        wire_rank(run, hooks, layers, rows)
        self._h1rank = {"run": run, "hooks": hooks, "sent": 0}
        return {"lo": lo, "roots": {lo + i: PH.weight_root(lo + i, PH.numpy_of(m.weight)).hex()
                                    for i, m in enumerate(layers)}}

    def h1rank_open(self, salt: bytes) -> None:
        self._h1rank["run"]._salt = bytes(salt)

    def h1rank_rows(self) -> dict[str, Any]:
        """The rank's calls since the last pull, its refusals so far and whether each step's calls came in its first step's
        order."""
        st = self._h1rank
        run: RankRun = st["run"]
        out = run.rows_since(st["sent"])
        st["sent"] = len(run.calls)
        return {"lo": run.lo, "rows": out, "refused": list(run.refused), "order_ok": run.order_ok,
                "unopened": run._unopened}

    def h1rank_release(self) -> None:
        st = getattr(self, "_h1rank", None)
        if st is not None:
            st["hooks"].uninstall()
            self._h1rank = None


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

    def _push(self) -> None:
        if not self.pushed and self.run._salt is not None:
            self.executor.collective_rpc("h1rank_open", kwargs={"salt": self.run._salt})
            self.pushed = True

    def pull(self) -> int:
        """Every rank's calls since the last pull, taken in index order (`ingest`); returns how many."""
        outs = self.executor.collective_rpc("h1rank_rows")
        self.pulls += 1
        rows = []
        for o in outs:
            self.ranks[o["lo"]] = {k: o[k] for k in ("refused", "order_ok", "unopened")}
            rows += o["rows"]
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
        try:
            self.close()
            self.run.finish()
        finally:
            self.hooks.uninstall()
            self.executor.collective_rpc("h1rank_release")


def ingest(run: PH.H1Run, rows: Sequence[Mapping[str, Any]]) -> None:
    """Rank calls into the holder's `run`, in index order, as `H1Run._call` takes them: each call, its tables, verdict and
    bindings; a round closes with its step's last linear."""
    last = len(run.shapes_) - 1
    for r in sorted(rows, key=lambda r: r["call"]["index"]):
        c = PH.Call(**r["call"])
        if run.calls and c.index <= run.calls[-1].index:
            raise RuntimeError(f"{run.entry}: rank call {c.index} after call {run.calls[-1].index}")
        run.calls.append(c)
        for into, key in ((run.tables, "tables"), (run.verdicts, "verdict"), (run.bound, "bound")):
            if r.get(key):
                into[c.index] = dict(r[key])
        run.step = c.step
        run.details["calls"] = run.commits["calls"] = len(run.calls)
        if c.layer == last:
            run._close_rounds(c.step)


def install(llm: Any, *, rows: int, steps: int, coins: Any = None, rounds: Any = None, opener: Any = None,
            admission: str = RQ.ROUND0, transcript_dir: str | None = None, key: bytes | None = None,
            entry: str = ENTRY, executor: Any = None) -> Holder:
    """H1 on an engine built with ``worker_extension_cls`` = `EXTENSION`: the ranks' layout gathered, each rank's `RankRun`
    installed, the window's salt from ``coins`` (an in-process `pouw.coins.Challenger`, as `pouw._install_h1` without a
    challenger) over every rank's weight roots, or opened by ``opener`` at round 0 (the challenger's), and the executor's
    ``execute_model`` wrapped (`Holder.admitting`). ``rounds`` is the round coins' `LiveRecord`, its draw family the
    window's draw coin when ``coins`` are given. A module is named by its rank at TP > 1 (``…qkv_proj@tp1``)."""
    if (coins is None) == (opener is None):
        raise ValueError(f"{entry}: the window's salt is from coins or opened by its opener, one of the two")
    if llm is not None:
        importlib.import_module(MRV2)
    executor = executor if executor is not None else executor_of(llm)
    layout = sorted(executor.collective_rpc("h1rank_layout"), key=lambda o: (o["pp"], o["tp"]))
    if len({o["decode_query_len"] for o in layout}) != 1:
        raise ValueError(f"{entry}: the ranks' runners disagree on decode_query_len")
    tp = len({o["tp"] for o in layout})
    los, shapes, modules = {}, [], {}
    for o in layout:
        lo = los[(o["pp"], o["tp"])] = len(shapes)
        modules |= {lo + int(w): name + (f"@tp{o['tp']}" if tp > 1 else "") for w, name in o["names"].items()}
        shapes += [tuple(s) for s in o["shapes"]]
    modules = {w: f"linear {w}" for w in range(len(shapes))} | modules
    key = key if key is not None else os.urandom(32)
    roots = {}
    for o in executor.collective_rpc("h1rank_install", kwargs={"entry": entry, "los": los, "shapes": shapes,
                                                               "modules": modules, "rows": rows, "steps": steps, "key": key}):
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
                   draw=draw, admission=admission, coin_source=source, opener=opener)
    run.commits.update(weights_root=weights_root(roots).hex(), coin_source=source, **commits,
                       ranks=[{"pp": o["pp"], "tp": o["tp"], "first_linear": los[(o["pp"], o["tp"])],
                               "linears": len(o["shapes"])} for o in layout])
    hooks = Hooks(tag=f"{entry}/holder")
    holder = Holder(run, executor, layout, hooks)
    holder._push()
    hooks.wrap(executor, "execute_model", holder.admitting)
    return holder


__all__ = ["ENTRY", "EXTENSION", "Holder", "RankExtension", "RankRun", "executor_of", "ingest", "install", "wire_rank"]
