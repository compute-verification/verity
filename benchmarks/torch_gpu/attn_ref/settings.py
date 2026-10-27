"""attn_ref's settings: every variable attn_ref.py, check.py and fixture.py read, declared once (`verity_config.settings`).

It owns ``ATTN_REF_``: a variable with it that nothing here declares is refused when the registry loads.  ATTN_EXPECT and
ATTN_ANY_GPU are declared without owning ``ATTN_``, which attn_bf16's run.py reads too.  build.sh ships this file with
attn_ref.py and check.py, and `verity_config.settings.of` finds it beside them in session.sh's scratch directory, where no
pyproject.toml is above them; a research run's PYTHONPATH, which session.sh keeps, brings the engine.  build.sh's and
session.sh's own variables (CUDA, CUBIN_ONLY, VERITY_BENCH_VENV, LEASE_MIN) are the shell's.
"""
from __future__ import annotations

from verity_config.settings import DEBUG, VALUE, WHERE, Registry, Setting as S, ints, present

REGISTRY = Registry(
    __file__, ("ATTN_REF_",),
    S("ATTN_REF_CUBIN", str, None, VALUE, "a cubin in place of the ship's attn_ref.sm_90a.cubin (the check records its "
                                          "sha256)"),
    S("ATTN_REF_LIBCUDA", str, "libcuda.so.1", WHERE, "the CUDA driver library the cubin loads through"),
    S("ATTN_EXPECT", str, "NVIDIA H200", WHERE, "the card name the identity check wants: which card runs, not what it "
                                                "computes"),
    S("ATTN_ANY_GPU", present, False, DEBUG, "set: no card or lease identity check (a run on another card)"),
    S("BATCHES", ints, (1, 7, 128, 20480), VALUE, "the batches check.py checks the kernels at"),
    S("TIME_B", int, 20480, VALUE, "the batch check.py's diagnostic timing runs at"),
    S("REPS", int, 20, VALUE, "the timed reps a timing's median is over"),
    S("PROCS", int, 8, WHERE, "fixture.py's processes: how fast it runs"))
