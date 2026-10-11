"""`run` draws from its seed and Hypothesis's own constants alone, without Lean. Hypothesis also draws constants it scrapes
from every local module the process has loaded, so a fixed seed's draws moved with what was imported: the vstar
difftest's coverage assertion failed when `rec_residuals.py`, which its reference loads, gained literals (#1760). These
tests pin the hook `run` patches, so a Hypothesis upgrade that moves it, or stops drawing scraped constants, fails here."""

from __future__ import annotations

import importlib
import sys

import pytest
from hypothesis import given, settings, strategies as st
from hypothesis import seed as use_seed
from hypothesis.internal.conjecture import providers

import verity_difftest as D

#: A stand-in for `Difftest.serve` that echoes each request's input.
ECHO = r"""
import json, sys
for line in sys.stdin:
    print(json.dumps({"output": json.loads(line)["input"]}, separators=(",", ":")))
    sys.stdout.flush()
"""

SEED, EXAMPLES = 20261011, 200


@pytest.fixture
def server():
    with D.Server([sys.executable, "-c", ECHO]) as s:
        yield s


@pytest.fixture
def plant(tmp_path, monkeypatch):
    """Import a new local module holding integers that Hypothesis scrapes (it skips those within 100 of zero)."""
    names = []

    def plant(name: str) -> None:
        (tmp_path / f"{name}.py").write_text(f"PLANTED = ({', '.join(str(10**6 + 7919 * i) for i in range(64))})\n")
        monkeypatch.syspath_prepend(str(tmp_path))
        importlib.import_module(name)
        names.append(name)

    yield plant
    for name in names:
        sys.modules.pop(name, None)


def bare() -> list[int]:
    """What plain Hypothesis draws, seeded as `run` seeds it."""
    drawn = []

    @use_seed(SEED)
    @settings(derandomize=True, database=None, deadline=None, max_examples=EXAMPLES)
    @given(st.integers())
    def draw(x: int) -> None:
        drawn.append(x)

    draw()
    return drawn


def through_run(server: D.Server) -> list[int]:
    drawn = []
    port = D.Port("echo", "f", lambda x: drawn.append(x) or x, st.integers(), need_rejects=False)
    D.run(port, examples=EXAMPLES, seed=SEED, server=server)
    return drawn


def test_a_seeded_run_draws_the_same_inputs_whatever_the_process_has_loaded(server, plant):
    """The control: plain Hypothesis's draws move when a module with new constants is imported, so the pool `run` empties
    is still drawn from. `run`'s draws don't move."""
    before, ran_before = bare(), through_run(server)
    plant("difftest_planted_a")
    after, ran_after = bare(), through_run(server)
    assert before != after, "Hypothesis no longer draws scraped constants: `_no_local_constants` is dead code"
    assert ran_before == ran_after


def test_run_empties_the_scraped_pool_and_restores_it(server):
    scrape, seen = providers._get_local_constants, []
    port = D.Port("echo", "f", lambda x: seen.append(len(providers._get_local_constants())) or x, st.integers(),
                  need_rejects=False)
    D.run(port, examples=5, seed=SEED, server=server)
    assert seen == [0] * len(seen) and seen
    assert providers._get_local_constants is scrape and len(scrape()) > 0
