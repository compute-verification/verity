"""serve-openai's learn-only warm-up (`serve_learn`) on CPU, over the stand-in engine and a stand-in committer whose plans follow
bounded staging's rules: a step's plan is keyed by its geometry and its step in the run, a reset keeps the plans the run before it
used, and a step that learns its plan over the transient host budget is refused unless the run is learn-only. After the warm-up,
the served execution blocks of the declared static shape learn no plan. Without it, each block's first step is refused, as at V3's
prefill mix ("its plan was not pre-learned"). ``--learn-host-budget-mb`` is the budget the committer reads."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from tests.pipeline.test_serve_openai import Committer, Engine, Runner, Tokenizer, records
from verity_vllm import settings
from verity_vllm.pipeline import cli
from verity_vllm.pipeline import serve_commit as sc
from verity_vllm.pipeline import serve_learn as sl
from verity_vllm.pipeline import serve_openai as so
from verity_vllm.pipeline import serve_protocol as proto

#: the host bytes a stand-in step stages a row, so a step of T rows stages T MiB
ROW = 1 << 20
#: 4 requests of 8 ids, 2 steps an execution block: the prefill stages 32 MiB, over a 16 MiB budget, and a decode step 4 MiB
SHAPE = ("--protocol", "plain", "--max-num-seqs", "4", "--max-num-batched-tokens", "32", "--window-steps", "2",
         "--max-model-len", "64", "--learn-host-budget-mb", "16")
#: what ``serve_load.py --fixed 4 --window-steps 2 --prompt-range 8:8`` sends an execution block: (requests, ids, max_tokens)
FIXED = (4, 8, 2)


def parse(*more: str, environ: dict[str, str] | None = None) -> so.Options:
    return cli.parse("serve-openai", ["--model", "m", "--revision", "r", *more], environ=environ or {})


class Bounded(Committer):
    """`Committer` under bounded staging: each step's outcome, in ``steps`` as (run, step, outcome), is a ``hit`` on a plan
    already learned, else it ``learned`` its plan within the budget, or, over it, ``learned only`` (a learn-only run records the
    plan and stages nothing) or ``refused`` (no plan). A reset keeps the plans the run before it used (``native_plans.h``)."""

    bounded = True

    def __init__(self, runner: Runner, budget_mb: float) -> None:
        self.learn_only, self.learn_budget_bytes = False, int(budget_mb * (1 << 20))
        self.plans: dict[Any, int] = {}
        self.run, self.used, self.steps = 0, False, []
        super().__init__(runner)

    def reset(self) -> None:
        if self.used:
            self.plans = {k: r for k, r in self.plans.items() if r == self.run}
            self.run, self.used = self.run + 1, False
        super().reset()

    def install(self) -> None:
        super().install()
        committed = self.runner.execute_model

        def execute_model(sched: Any) -> Any:
            ib, step = sched.ib, self._step_index
            qsl = [int(q) for q in ib.query_start_loc_np]
            key = (step, tuple(sorted(zip((int(c) for c in ib.num_computed_tokens_np), (b - a for a, b in zip(qsl, qsl[1:]))))))
            if key in self.plans:
                outcome = "hit"
            elif len(sched.ids) * ROW <= self.learn_budget_bytes:
                outcome = "learned"
            else:
                outcome = "learned only" if self.learn_only else "refused"
            if outcome != "refused":
                self.plans[key] = self.run
            self.used = True
            self.steps.append((self.run, step, outcome))
            return committed(sched)

        self.runner.execute_model = execute_model

    def finalize(self) -> Any:
        rc = super().finalize()
        only = [s for r, s, o in self.steps if r == self.run and o == "learned only"]
        rc.extra.update(bounded_staging={"learn_budget_bytes": self.learn_budget_bytes, "learned_only_steps": only,
                                         "learned_only_bytes": 32 * ROW * len(only), "transient_peak_bytes": 4 * ROW,
                                         "plan_mismatches": 0},
                        native_collect={"plans_cached": len(self.plans)})
        return rc


def served(tmp_path: Any, engine: Engine, com: Bounded, opts: so.Options, blocks: int) -> sc.CommitWindows:
    """``blocks`` execution blocks of `FIXED`'s load: each block's requests admitted together, prefilled at its first step and
    run through its last."""
    n, ids, steps = FIXED
    windows = sc.CommitWindows(com, tmp_path, run_id="t", window_steps=opts.window_steps, eos_ids=[Tokenizer.eos_token_id],
                               clock=lambda: 0.0, wall=lambda: 0.0)
    server = so.Server(proto=proto.PlainProtocol(True), served_name="m", max_model_len=opts.max_model_len, tokenizer=Tokenizer(),
                       defaults={}, submit=lambda _a: None, abort=lambda _e: None, clock=lambda: 7)
    for b in range(blocks):
        loop = so.EngineLoop(engine, windows)
        for i in range(n):
            body = {"prompt": [100 + 10 * b + i] * ids, "max_tokens": steps, "min_tokens": steps, "ignore_eos": True,
                    "temperature": 0, "stream": True}
            loop.submit(server.admit(body, lambda _d: None)[1])
        loop.drain("the block is served")
        loop.run()
        assert loop.error is None
    return windows


def stand_ins(opts: so.Options) -> tuple[Engine, Bounded, Any]:
    runner = Runner()
    engine = Engine(runner)
    engine.vllm_config = SimpleNamespace(model_config=SimpleNamespace(get_vocab_size=lambda: 32000))
    return engine, Bounded(runner, opts.learning.learn_host_budget_mb), SimpleNamespace(vllm=SimpleNamespace(llm_engine=engine))


def test_after_the_warm_up_no_served_step_of_the_declared_static_shape_learns(tmp_path):
    opts = parse(*SHAPE)
    engine, com, llm = stand_ins(opts)
    assert sl.warmed(com, llm, opts) is com
    assert com.steps == [(0, 0, "learned only"), (0, 1, "learned")]
    got = com.learn_warmup
    assert (got["shape"], got["engine_steps"], got["committed_steps"]) == ({"requests": 4, "prompt_ids": 8, "steps": 2}, 2, 2)
    assert (got["learned_only_steps"], got["learn_budget_bytes"], got["errors"]) == ([0], 16 * ROW, [])
    assert not com.learn_only and com._step_index == 0 and not engine.has_unfinished_requests()
    assert not list(tmp_path.iterdir())                      # no execution block, no record

    windows = served(tmp_path, engine, com, opts, blocks=3)
    assert [o for r, _s, o in com.steps if r > 0] == ["hit"] * 6
    assert [w["window"] for w in records(tmp_path, "windows.jsonl")] == [0, 1, 2]
    assert len(records(tmp_path, "requests.jsonl")) == 12 and windows.summary()["passed"]


def test_without_the_warm_up_every_execution_blocks_first_step_is_refused(tmp_path):
    opts = parse(*SHAPE, "--no-learn-warmup")
    engine, com, llm = stand_ins(opts)
    assert sl.warmed(com, llm, opts).learn_warmup is None and com.steps == []
    served(tmp_path, engine, com, opts, blocks=3)
    assert [(s, o) for _r, s, o in com.steps] == [(0, "refused"), (1, "learned"), (0, "refused"), (1, "hit"), (0, "refused"), (1, "hit")]


def test_a_committer_without_bounded_staging_has_no_warm_up():
    opts = parse(*SHAPE)
    runner = Runner()
    engine = Engine(runner)
    assert sl.warm_up(engine, Committer(runner), opts, proto.PlainProtocol(True)) is None
    assert engine.n == 0


def test_the_declared_static_shape_is_t_rows_among_the_slots_for_the_execution_blocks_steps():
    v3 = parse("--max-num-seqs", "8", "--max-num-batched-tokens", "8192", "--window-steps", "1", "--max-model-len", "1088")
    assert sl.shape_of(v3) == {"requests": 8, "prompt_ids": 1024, "steps": 1}
    capped = parse("--max-num-seqs", "8", "--window-requests", "4", "--max-num-batched-tokens", "2048", "--window-steps", "64",
                   "--max-model-len", "1024")
    assert sl.shape_of(capped) == {"requests": 4, "prompt_ids": 512, "steps": 64}
    assert sl.shape_of(parse("--max-num-seqs", "2", "--max-num-batched-tokens", "2048", "--window-steps", "64")) == \
        {"requests": 2, "prompt_ids": 511, "steps": 1}                  # within --max-model-len 512


def test_learn_host_budget_mb_is_the_committers_budget_and_the_variable_only_its_default():
    assert parse().learning.learn_host_budget_mb == 4096.0
    assert parse(environ={"VERITY_LEARN_HOST_BUDGET_MB": "65536"}).learning.learn_host_budget_mb == 65536.0
    opts = parse("--learn-host-budget-mb", "6144", environ={"VERITY_LEARN_HOST_BUDGET_MB": "65536"})
    assert settings.get("VERITY_LEARN_HOST_BUDGET_MB", environ=sl.settings_of(opts)) == 6144.0
    with pytest.raises(cli.UsageError, match=r"--learn-host-budget-mb 0\.0"):
        so.preflight(parse("--no-commit", "--learn-host-budget-mb", "0"), which=lambda _: None)
