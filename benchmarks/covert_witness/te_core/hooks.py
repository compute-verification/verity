"""What a standalone train.py may call when the harness runs it (`python -m te_core.run --script`), each a no-op when
it runs on its own:

    from te_core import hooks
    hooks.plan(n_steps)                  # the step count: a checkpoint after every 5% of it (default: of the MAC budget)
    hooks.output(lambda: state_dict)     # the output model now, θ's tensors in θ's dtypes: the 5% checkpoints read it
    with hooks.step(): ...               # explicit step boundaries (default: each call of the start's data stream)
    hooks.tokens(batch)                  # token data not from the start's data stream, reaching the model other than by an embedding lookup
    fam = hooks.family(name, plan)       # a counterexample family (counterexample.py), `fam.put(step, enc)`

None of them is needed: without them the harness counts the run, cuts its steps at the data stream's calls and scores
the encodings the program writes (`cw.py`).
"""
from __future__ import annotations

from contextlib import nullcontext

ACTIVE = None


def session():
    return ACTIVE


def plan(n_steps: int) -> None:
    if ACTIVE is not None:
        ACTIVE.plan(n_steps)


def output(fn) -> None:
    if ACTIVE is not None:
        ACTIVE.output(fn)


def step():
    if ACTIVE is None:
        return nullcontext()
    return ACTIVE.explicit_step()


def tokens(batch):
    return batch if ACTIVE is None else ACTIVE.tokens(batch)


class _NoFamily:
    def put(self, step, enc) -> None:
        pass


def family(name: str, plan, uniform: bool = True):
    return _NoFamily() if ACTIVE is None else ACTIVE.family(name, plan, uniform)
