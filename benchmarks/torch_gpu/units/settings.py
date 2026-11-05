"""The units' settings: every variable check.py, bench.py, lib.py and fixture.py read, declared once
(`verity_config.settings`).

It owns ``UNITS_``: a variable with it that nothing here declares is refused when the registry loads.  build.sh ships this
file with the modules, and `verity_config.settings.of` finds it beside them in session.sh's scratch directory, where no
pyproject.toml is above them; the caller's PYTHONPATH, which session.sh keeps, brings the engine.  build.sh's and
session.sh's own variables (CUDA, CUBIN_ONLY, REPLAY, ROWS, VERITY_BENCH_VENV, LEASE_MIN) are the shell's.
"""
from __future__ import annotations

from verity_config.settings import DEBUG, VALUE, WHERE, Registry, Setting as S, ints, not_zero, one, present

REGISTRY = Registry(
    __file__, ("UNITS_",),
    S("UNITS_CUBIN", str, None, VALUE, "a cubin in place of the ship's units.ARCH.cubin (the check records its sha256)"),
    S("UNITS_LIBCUDA", str, "libcuda.so.1", WHERE, "the CUDA driver library the cubin loads through"),
    S("UNITS_EXPECT", str, None, WHERE, "the card name the identity check wants (default its arch's, check.EXPECT): which "
                                        "card runs, not what it computes"),
    S("UNITS_ANY_GPU", present, False, DEBUG, "set: no card or lease identity check (a run on another card)"),
    S("UNITS_REPLAY", str, None, VALUE, "fixture.py's --replay: the CPU replay its pools are cross-checked against"),
    S("UNITS_ROWS", str, None, VALUE, "fixture.py's --rows: an ops/rows.py loaded in place of the installed one"),
    S("SWEEP", not_zero, True, VALUE, "0: check.py runs no sweep"),
    S("SWEEP_ONLY", one, False, VALUE, "1: check.py runs the sweeps alone"),
    S("BATCHES", ints, (1, 7, 128, 4096, 16384, 20480), VALUE, "the batches check.py checks each unit at"),
    S("TIME_B", ints, (4096, 16384), VALUE, "the batches bench.py times"),
    S("REPS", int, 20, VALUE, "the timed reps an arm's median is over"),
    S("STEP_REPS", int, 5, VALUE, "the timed reps a measured step's median is over"),
    S("PROCS", int, 8, WHERE, "fixture.py's sweep processes: how fast it runs"))
