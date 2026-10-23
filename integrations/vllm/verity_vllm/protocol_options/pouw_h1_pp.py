"""H1 under pipeline parallelism (TP 1 × PP N on Model Runner V2's multiprocess executor): the window held whole in the
engine-core process (`Holder`, an unchanged `pouw_h1.H1Run` over every stage's linears) and each stage's calls recorded in
its own worker (`StageRun`, through the engine's ``worker_extension_cls``, `StageExtension`).

Stage p numbers its linears on from the stages before it, so call (s, w) is index s·L + w at position s·U + off(w) in
every stage: PP = 1's static layout, with its tables (A, z) committed where the rows are, under the window's salt and
hm96 key. Admission stays with the scheduler: the holder wraps the executor's ``execute_model``, and before each step with
scheduled tokens it pulls every stage's calls so far (`StageExtension.h1pp_rows`, an RPC queued behind the earlier steps,
so it returns once they have run in every stage), takes them in index order, closing each round whose last linear is in,
then admits the step (`H1Run.admit`): one round a step, its root over every stage's calls before its coin, which only the
holder draws. The salt and key reach the stages before the first forward (`Holder._push`). The pull blocks the engine
core until the previous step has left the last stage, so steps don't overlap in the pipeline under H1.

Refused here: TP > 1 within a stage (H1 at TP > 1 is not wired), and ``served`` (`pouw_h1_served.Served`, whose b/<K>
needs every linear's weight in one process)."""
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
EXTENSION = "verity_vllm.protocol_options.pouw_h1_pp.StageExtension"
#: the module of Model Runner V2, whose `sort_batch_req_ids` orders a step's rows (`pouw_h1_requests.batch_order`)
MRV2 = "vllm.v1.worker.gpu.model_runner"


class StageRun(PH.H1Run):
    """One pipeline stage's side of the window: the calls of its linears [lo, hi) of the window's L, each at its global
    index s·L + w with its tables, under the salt the holder pushes (`StageExtension.h1pp_open`). Its first linear starts
    the stage's next step; rounds, admission and coins are the holder's."""

    def __init__(self, entry: str, modules: Mapping[int, str], shapes: Sequence[tuple[int, int]], rows: int, steps: int,
                 lo: int, hi: int, key: bytes) -> None:
        super().__init__(entry, None, modules, shapes, rows, steps, key=key, opener=self._unpushed)
        self.lo, self.hi = int(lo), int(hi)

    def _unpushed(self, _root: bytes) -> Any:
        raise RuntimeError(f"{self.entry}: stage [{self.lo}, {self.hi}) used the window's salt before its holder pushed it")

    def _index(self, s: int, w: int) -> int:
        return s * len(self.shapes_) + w

    def _close_rounds(self, through: int) -> None:
        return None

    def record(self, w: int, a: Any, s_a: Any, weight: Any, weight_scale: Any, z: Any = None) -> PH.Call | None:
        if w == self.lo and self.lo:
            self.step += 1
            self.last = w - 1
        return super().record(w, a, s_a, weight, weight_scale, z)

    def rows_since(self, sent: int) -> list[dict[str, Any]]:
        """The calls from the `sent`-th on, each with its tables, its credit verdict and what its listeners bound."""
        return [{"call": asdict(c), "tables": self.tables.get(c.index), "verdict": self.verdicts.get(c.index),
                 "bound": self.bound.get(c.index)} for c in self.calls[sent:]]


def wire_stage(run: StageRun, hooks: Any, layers: list[Any], rows: int) -> None:
    """`pouw_h1_hooks.wire` on a stage: its layer i is the window's linear lo + i; no runner hook (admission is the
    holder's)."""
    for i, layer in enumerate(layers):
        w = run.lo + i
        hooks.wrap(layer.quant_method, "observe", lambda orig, w=w: PHK.observer(run, w, orig))
        hooks.wrap(layer.quant_method, "apply", lambda orig, w=w: PHK.traced(run, w, orig))
        if run.commit and hasattr(type(layer.quant_method), "zy_rows") and layer.weight.shape[0] % PH.TILE[1] == 0:
            hooks.patch(layer.quant_method, "zy_rows", rows)


class StageExtension:
    """The ``worker_extension_cls`` of a pipeline-parallel engine serving H1: each worker's methods, called from the holder
    with ``collective_rpc``; importing it registers the int8 parts in the worker (`engine.int8_ranks`)."""

    def _h1pp_layers(self) -> tuple[Any, list[Any]]:
        from verity_vllm.protocol_options import pouw
        model = self.model_runner.get_model()  # type: ignore[attr-defined]
        return model, PHK.layers(model, pouw.linear_types())

    def h1pp_layout(self) -> dict[str, Any]:
        """This worker's stage and TP rank, its linears' shapes and module names (local ordinals) in module order, and its
        runner's decode_query_len."""
        from vllm.distributed.parallel_state import get_pp_group, get_tp_group
        from verity_vllm.protocol_options import pouw
        model, layers = self._h1pp_layers()
        return {"pp": get_pp_group().rank_in_group, "tp": get_tp_group().rank_in_group,
                "shapes": [[int(d) for d in m.weight.shape] for m in layers], "names": pouw._named(model, layers),
                "decode_query_len": int(self.model_runner.decode_query_len)}  # type: ignore[attr-defined]

    def h1pp_install(self, entry: str, los: Mapping[int, int], shapes: Sequence[Sequence[int]], modules: Mapping[int, str],
                     rows: int, steps: int, key: bytes) -> dict[str, Any]:
        """This stage's `StageRun` from linear ``los[pp]`` on, its hooks wired; returns its weights' roots by global
        index (`pouw_h1.weight_root`, as `pouw.commit_weights`)."""
        from vllm.distributed.parallel_state import get_pp_group
        _, layers = self._h1pp_layers()
        lo = int(los[get_pp_group().rank_in_group])
        run = StageRun(entry, modules, [tuple(s) for s in shapes], rows, steps, lo, lo + len(layers), key)
        hooks = Hooks(tag=f"{entry}/stage{lo}")
        wire_stage(run, hooks, layers, rows)
        self._h1pp = {"run": run, "hooks": hooks, "sent": 0}
        return {"lo": lo, "roots": {lo + i: PH.weight_root(lo + i, PH.numpy_of(m.weight)).hex()
                                    for i, m in enumerate(layers)}}

    def h1pp_open(self, salt: bytes) -> None:
        self._h1pp["run"]._salt = bytes(salt)

    def h1pp_rows(self) -> dict[str, Any]:
        """The stage's calls since the last pull, its refusals so far and whether its calls came in layer order."""
        st = self._h1pp
        run: StageRun = st["run"]
        out = run.rows_since(st["sent"])
        st["sent"] = len(run.calls)
        return {"lo": run.lo, "rows": out, "refused": list(run.refused), "order_ok": run.order_ok,
                "unopened": run._unopened}

    def h1pp_release(self) -> None:
        st = getattr(self, "_h1pp", None)
        if st is not None:
            st["hooks"].uninstall()
            self._h1pp = None


def executor_of(llm: Any) -> Any:
    """The in-process engine core's executor (``VLLM_ENABLE_V1_MULTIPROCESSING=0``), whose ``collective_rpc`` reaches every
    worker."""
    core = getattr(llm.llm_engine, "engine_core", None)
    inner = getattr(core, "engine_core", None) or core
    executor = getattr(inner, "model_executor", None)
    if executor is None or not callable(getattr(executor, "collective_rpc", None)):
        raise RuntimeError("pouw_h1_pp: no in-process executor with collective_rpc (is VLLM_ENABLE_V1_MULTIPROCESSING=0?)")
    return executor


def _runner(decode_query_len: int) -> Any:
    """What `pouw_h1_requests.batch_order` reads of the stages' Model Runner V2: its module's batch order and its
    decode_query_len."""
    try:
        importlib.import_module(MRV2)
    except ImportError:
        pass
    return type("GPUModelRunner", (), {"__module__": MRV2, "decode_query_len": int(decode_query_len)})()


class Holder:
    """The window of a pipeline-parallel engine (`install`): its `H1Run` (``run``) fed by the stages and admitted at the
    executor; `finish` pulls the last calls, closes and releases the window and the stages' hooks."""

    def __init__(self, run: PH.H1Run, executor: Any, layout: list[dict[str, Any]], hooks: Hooks) -> None:
        self.run, self.executor, self.layout, self.hooks = run, executor, layout, hooks
        self.runner = _runner(layout[0]["decode_query_len"])
        self.stages: dict[int, dict[str, Any]] = {}
        self.pushed = False
        self.pulls = 0

    def _push(self) -> None:
        if not self.pushed and self.run._salt is not None:
            self.executor.collective_rpc("h1pp_open", kwargs={"salt": self.run._salt})
            self.pushed = True

    def pull(self) -> int:
        """Every stage's calls since the last pull, taken in index order (`ingest`); returns how many."""
        outs = self.executor.collective_rpc("h1pp_rows")
        self.pulls += 1
        rows = []
        for o in outs:
            self.stages[o["lo"]] = {k: o[k] for k in ("refused", "order_ok", "unopened")}
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
            self.run.refused[:] = [r for s in sorted(self.stages) for r in self.stages[s]["refused"]]
            self.run.order_ok = self.run.order_ok and all(s["order_ok"] for s in self.stages.values())
            self.run.details["stages"] = {str(lo): {"refused": len(s["refused"]), "order_ok": s["order_ok"],
                                                    **({"unopened": s["unopened"]} if s["unopened"] else {})}
                                          for lo, s in sorted(self.stages.items())}
            self.run.details["pulls"] = self.pulls
            self.run.close()

    def finish(self) -> None:
        try:
            self.close()
            self.run.finish()
        finally:
            self.hooks.uninstall()
            self.executor.collective_rpc("h1pp_release")


def ingest(run: PH.H1Run, rows: Sequence[Mapping[str, Any]]) -> None:
    """Stage calls into the holder's `run`, in index order, as `H1Run._call` takes them: each call, its tables, verdict and
    bindings; a round closes with its step's last linear."""
    last = len(run.shapes_) - 1
    for r in sorted(rows, key=lambda r: r["call"]["index"]):
        c = PH.Call(**r["call"])
        if run.calls and c.index <= run.calls[-1].index:
            raise RuntimeError(f"{run.entry}: stage call {c.index} after call {run.calls[-1].index}")
        run.calls.append(c)
        for into, key in ((run.tables, "tables"), (run.verdicts, "verdict"), (run.bound, "bound")):
            if r.get(key):
                into[c.index] = dict(r[key])
        run.step, run.last = c.step, c.layer
        run.details["calls"] = run.commits["calls"] = len(run.calls)
        if c.layer == last:
            run._close_rounds(c.step)


def install(llm: Any, *, rows: int, steps: int, coins: Any = None, rounds: Any = None, opener: Any = None,
            admission: str = RQ.ROUND0, transcript_dir: str | None = None, key: bytes | None = None,
            entry: str = ENTRY) -> Holder:
    """H1 on an engine built with ``worker_extension_cls`` = `EXTENSION` at PP > 1: the stages' layout gathered, each
    stage's `StageRun` installed, the window's salt from ``coins`` (an in-process `pouw.coins.Challenger`, as
    `pouw._install_h1` without a challenger) over every stage's weight roots, or opened by ``opener`` at round 0 (the
    challenger's), and the executor's ``execute_model`` wrapped (`Holder.admitting`). ``rounds`` is the round coins'
    `LiveRecord`, its draw family the window's draw coin when ``coins`` are given."""
    if (coins is None) == (opener is None):
        raise ValueError(f"{entry}: the window's salt is from coins or opened by its opener, one of the two")
    executor = executor_of(llm)
    layout = sorted(executor.collective_rpc("h1pp_layout"), key=lambda o: (o["pp"], o["tp"]))
    if any(o["tp"] for o in layout):
        raise ValueError(f"{entry}: H1 at TP > 1 is not wired; a pipeline stage is one GPU here")
    if len({o["decode_query_len"] for o in layout}) != 1:
        raise ValueError(f"{entry}: the stages' runners disagree on decode_query_len")
    los, shapes, modules = {}, [], {}
    for o in layout:
        los[o["pp"]] = len(shapes)
        modules |= {los[o["pp"]] + int(w): name for w, name in o["names"].items()}
        shapes += [tuple(s) for s in o["shapes"]]
    modules = {w: f"linear {w}" for w in range(len(shapes))} | modules
    key = key if key is not None else os.urandom(32)
    roots = {}
    for o in executor.collective_rpc("h1pp_install", kwargs={"entry": entry, "los": los, "shapes": shapes,
                                                             "modules": modules, "rows": rows, "steps": steps, "key": key}):
        roots |= {int(w): bytes.fromhex(r) for w, r in o["roots"].items()}
    if sorted(roots) != list(range(len(shapes))):
        raise RuntimeError(f"{entry}: the stages' weight roots cover {len(roots)} of {len(shapes)} linears")
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
                       pipeline={"stages": len(layout), "first_linear": [los[p] for p in sorted(los)]})
    hooks = Hooks(tag=f"{entry}/holder")
    holder = Holder(run, executor, layout, hooks)
    holder._push()
    hooks.wrap(executor, "execute_model", holder.admitting)
    return holder


__all__ = ["ENTRY", "EXTENSION", "Holder", "StageExtension", "StageRun", "executor_of", "ingest", "install", "wire_stage"]
