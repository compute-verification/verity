"""`protocol_options.pouw_h1_ranks`: a window split over two ranks (pipeline stages, or tensor-parallel ranks through
`install`), each rank's calls recorded where its linears are and the window held whole beside the scheduler, commits what
one process serving the same linears commits: the same calls at the same positions, the same tables, round roots, coins,
calls root and draw root."""
from __future__ import annotations

import hashlib
import itertools
import random
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from verity.core.protocols.pouw import window as Wd  # noqa: E402
from verity.core.protocols.pouw.coins import Challenger  # noqa: E402
from verity.core.protocols.sampled_proofs.challenger import LiveRecord  # noqa: E402
from verity_vllm.engine import int8_exact as E  # noqa: E402
from verity_vllm.engine.hooks import Hooks  # noqa: E402
from verity_vllm.protocols.pouw import h1 as PH  # noqa: E402
from verity_vllm.protocols.pouw.h1 import hooks as PHK  # noqa: E402
from verity_vllm.protocols.pouw.h1 import ranks as PHR  # noqa: E402
from verity_vllm.protocols.pouw.h1 import requests as RQ  # noqa: E402

K = 64
SALT = hashlib.sha512(b"salt").digest()
KEY = hashlib.sha512(b"key").digest()[:32]
SHAPES = [(32, K), (40, K), (32, K), (40, K)]
MODULES = {0: "l0.q", 1: "l0.o", 2: "l1.q", 3: "l1.o"}
#: stage p's linears start at SPLIT[p]
SPLIT = (0, 2)


class Int8Arm:
    """`engine.int8_vllm.Int8PlainLinearMethod`'s plain linear, handing its values to ``observe``."""

    int8_arm = E.ARM

    def apply(self, layer, x, bias=None):
        xq, s_a = E.quantize_rows(x)
        y = E.dequantize(E.int_mm(xq, layer.weight), s_a, layer.weight_scale)
        self.observe(layer, xq, s_a, y)
        return y

    def observe(self, layer, xq, s_a, y):
        pass


class Linear:
    def __init__(self, n, seed):
        g = torch.Generator().manual_seed(seed)
        self.weight = torch.randint(-127, 128, (n, K), generator=g, dtype=torch.int8)
        self.weight_scale = (torch.rand(n, 1, generator=g) * 0.01 + 1e-4).float()
        self.quant_method = Int8Arm()


class Runner:
    decode_query_len = 1

    def execute_model(self, scheduler_output, non_block=False):
        return None


class Executor:
    """The multiprocess executor as the holder uses it: ``execute_model`` and ``collective_rpc`` over its ranks' workers.
    Its ranks share this process, so their patches stack; ``h1rank_release`` releases them newest first, which separate
    processes never need, and answers in the workers' order."""

    def __init__(self, workers):
        self.workers, self.steps = workers, 0

    def execute_model(self, scheduler_output, non_block=False):
        self.steps += 1

    def collective_rpc(self, method, kwargs=None):
        """vLLM's: a worker method by name, or a callable called with each worker first."""
        def call(w):
            return method(w, **(kwargs or {})) if callable(method) else getattr(w, method)(**(kwargs or {}))
        if method == "h1rank_release":
            return [call(w) for w in reversed(self.workers)][::-1]
        return [call(w) for w in self.workers]


def _layers():
    return [Linear(n, seed=w) for w, (n, _k) in enumerate(SHAPES)]


def _rounds():
    clock = itertools.count()
    return LiveRecord(Wd.live_challenger(0, 4, random.Random(0).randbytes), Wd.ROUND, clock=lambda: float(next(clock)))


def _so(*new, running=None):
    reqs = [SimpleNamespace(req_id=i, prompt_token_ids=p, sampling_params=SimpleNamespace(max_tokens=c)) for i, p, c in new]
    scheduled = dict(running or {}) | {i: len(p or ()) for i, p, _c in new}
    return SimpleNamespace(scheduled_new_reqs=reqs, num_scheduled_tokens=scheduled,
                           total_num_scheduled_tokens=sum(scheduled.values()))


#: three served steps of a four-step window: a admitted in round 0, b and c in round 1, all three decoding in round 2
STEPS = (_so(("a", [1, 2, 3], 4)), _so(("b", [4], 2), ("c", [5, 6], 3), running={"a": 1}),
         _so(running={"a": 1, "b": 1, "c": 1}))


def _forward(layers, m, seed, rows=None):
    g = torch.Generator().manual_seed(seed)
    for w, layer in enumerate(layers):
        layer.quant_method.apply(layer, (torch.randn(rows or m, K, generator=g) * 3).bfloat16())


def _whole(salt=SALT, modules=MODULES):
    """One process serving every linear (PP = 1): `pouw_h1_hooks.wire` on the runner and the layers."""
    layers, runner, h = _layers(), Runner(), Hooks(tag="pp1")
    run = PH.H1Run("t", salt, modules, SHAPES, 32, 4, rounds=_rounds(), key=KEY, admission=RQ.PER_ROUND)
    run.draw = LiveRecord(run.rounds.challenger, Wd.DRAW)
    PHK.wire(run, h, layers, runner, 32)
    for i, so in enumerate(STEPS):
        runner.execute_model(so)
        _forward(layers, so.total_num_scheduled_tokens, seed=i)
    run.finish()
    h.uninstall()
    return run


def _staged(rows_of=None):
    """The same window over two stages (`RankRun` in each worker, `Holder` at the executor), empty batches between the
    served ones as a pipeline's batch queue schedules them; `rows_of(stage, step)` overrides a stage's served rows."""
    layers, hooks, workers = _layers(), [], []
    for p, lo in enumerate(SPLIT):
        hi = SPLIT[p + 1] if p + 1 < len(SPLIT) else len(SHAPES)
        stage = PHR.RankRun("t", MODULES, SHAPES, 32, 4, lo, hi, KEY)
        h = Hooks(tag=f"stage{p}")
        PHR.wire_rank(stage, h, layers[lo:hi], 32)
        worker = PHR.RankExtension()
        worker._h1rank = {"run": stage, "hooks": h, "sent": 0}
        workers.append(worker)
        hooks.append((layers[lo:hi], h))
    executor = Executor(workers)
    run = PH.H1Run("t", SALT, MODULES, SHAPES, 32, 4, rounds=_rounds(), key=KEY, admission=RQ.PER_ROUND)
    run.draw = LiveRecord(run.rounds.challenger, Wd.DRAW)
    holder = PHR.Holder(run, executor, [{"decode_query_len": 1}], Hooks(tag="holder"))
    holder._push()
    holder.hooks.wrap(executor, "execute_model", holder.admitting)
    for i, so in enumerate(STEPS):
        executor.execute_model(so)
        executor.execute_model(_so())
        g = torch.Generator().manual_seed(i)
        xs = [(torch.randn(so.total_num_scheduled_tokens, K, generator=g) * 3).bfloat16() for _ in SHAPES]
        for p, (stage_layers, _h) in enumerate(hooks):
            for j, layer in enumerate(stage_layers):
                x = xs[SPLIT[p] + j]
                if rows_of is not None and (m := rows_of(p, i)) is not None:
                    x = (torch.randn(m, K, generator=g) * 3).bfloat16()
                layer.quant_method.apply(layer, x)
    holder.finish()
    return holder, executor


def test_two_stages_commit_what_one_process_does():
    whole, (holder, executor) = _whole(), _staged()
    staged = holder.run
    assert staged.static() and whole.static()
    assert [c.summary() for c in staged.calls] == [c.summary() for c in whole.calls]
    assert [c.index for c in staged.calls] == list(range(4 * len(SHAPES)))
    assert staged.manifest() == whole.manifest()
    assert staged.round_log == whole.round_log and len(staged.round_log) == 4
    assert staged.calls_root() == whole.calls_root()
    assert staged.commits["draw"] == whole.commits["draw"] and staged.commits["draw"]["coin"] is not None
    assert staged.step_rows() == whole.step_rows() == [3, 4, 3, 0]
    assert executor.steps == 2 * len(STEPS) and holder.pulls == len(STEPS) + 1
    assert staged.details["ranks"] == {"0": {"refused": 0, "order_ok": True}, "2": {"refused": 0, "order_ok": True}}


def test_a_stage_call_past_the_rows_is_refused_and_the_window_is_not_static():
    holder, _ = _staged(rows_of=lambda p, s: 33 if (p, s) == (1, 1) else None)
    run = holder.run
    assert not run.static() and run.refused == [{"step": 1, "linear": 2, "m": 33}, {"step": 1, "linear": 3, "m": 33}]
    assert run.details["ranks"]["2"]["refused"] == 2


def test_a_stage_numbers_its_calls_from_the_window_and_waits_for_its_salt():
    stage = PHR.RankRun("t", MODULES, SHAPES, 32, 4, 2, 4, KEY)
    layer = Linear(32, seed=2)
    with pytest.raises(RuntimeError, match="before its holder pushed it"):
        stage.record(2, torch.zeros(1, K, dtype=torch.int8), torch.ones(1), layer.weight, layer.weight_scale)
    stage = PHR.RankRun("t", MODULES, SHAPES, 32, 4, 2, 4, KEY)
    stage._salt = SALT
    rng = np.random.default_rng(0)
    for s in range(2):
        for w in (2, 3):
            a = torch.from_numpy(rng.integers(-127, 128, (1, K), dtype=np.int8))
            n = SHAPES[w][0]
            stage.record(w, a, torch.ones(1), torch.zeros(n, K, dtype=torch.int8), torch.ones(n, 1))
    assert [(c.index, c.step, c.layer, c.base) for c in stage.calls] == [
        (s * 4 + w, s, w, s * stage.u_step + stage.offs[w]) for s in range(2) for w in (2, 3)]
    assert stage.order_ok and stage.round_log == []


class FakeRank(PHR.RankExtension):
    """A worker of a TP engine as `install` reaches it: its (pp, tp) rank and its shards, named as vLLM names them."""

    def __init__(self, tp, layers):
        self.rank, self.layers = (0, tp), layers
        self.model_runner = SimpleNamespace(decode_query_len=1)
        self.model = SimpleNamespace(named_modules=lambda: [(n, m) for n, m in zip(("l0.q", "l0.o"), layers)])

    def _h1rank_layers(self):
        return self.model, self.layers

    def _h1rank_rank(self):
        return self.rank


def test_install_lays_tensor_parallel_ranks_out_rank_major_and_commits_one_window():
    layers = _layers()
    workers = [FakeRank(1, layers[2:]), FakeRank(0, layers[:2])]
    executor = Executor(workers)
    holder = PHR.install(None, executor=executor, rows=32, steps=4, coins=Challenger(0, random.Random(0).randbytes),
                         rounds=_rounds(), key=KEY, entry="t", admission=RQ.PER_ROUND)
    run = holder.run
    assert run.modules == {0: "l0.q@tp0", 1: "l0.o@tp0", 2: "l0.q@tp1", 3: "l0.o@tp1"}
    assert run.commits["ranks"] == [{"pp": 0, "tp": 0, "first_linear": 0, "linears": 2},
                                    {"pp": 0, "tp": 1, "first_linear": 2, "linears": 2}]
    assert run.commits["weights_root"] and run.commits["salt_opening"] and run.commits["round_coins"]
    for i, so in enumerate(STEPS):
        executor.execute_model(so)
        _forward(layers, so.total_num_scheduled_tokens, seed=i)
    holder.finish()
    whole = _whole(salt=run._salt, modules=run.modules)
    assert run.static() and [c.index for c in run.calls] == list(range(4 * len(SHAPES)))
    assert [c.summary() for c in run.calls] == [c.summary() for c in whole.calls]
    assert run.round_log == whole.round_log and run.calls_root() == whole.calls_root()
    assert run.commits["draw"] == whole.commits["draw"]
    assert run.details["ranks"] == {"0": {"refused": 0, "order_ok": True}, "2": {"refused": 0, "order_ok": True}}
    assert all(w._h1rank is None for w in workers)
