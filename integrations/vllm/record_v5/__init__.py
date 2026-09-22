"""[A9 contract 2026-09-10] ``record_v5`` -- the contract-conforming experiment runner (docs/PROTOCOL.md).

Organised by contract step (see ``data/contract/a9-runner/design.md`` §1):

    config.py         run configuration + the LEGACY GUARD (R14): refuses VU-count B_ex, legacy keys, inputs as VUs, family
                      exemptions, serve-time capture beyond inputs/outputs/seed, legacy drivers/instances
    freeze.py         source freeze: commit + sha256(dirty patch) + file digests; every phase refuses if sources changed
    deps.py           where A3 / A5-6 modules come from (worktree after merge, or a read-only branch export), recorded
    population_v5.py  the sampled population WITHOUT input gates (legacy population minus Input kinds, labelled) -- until A1's lands
    instance_v5.py    step 0: the epoch instance (A3's schema) over the v5 population, with the enumerated validation allowlist
    serve_v5.py       the original execution: 3-field production (inputs, outputs, seed) + the stock baseline (R19)     [GPU]
    workload_v5.py    step 0: the auditor's request rows (epoch.inputs, WR-5), written before serving                        [CPU]
    protocol_v5.py    steps 1-4: A3's contract-ordered driver composed by import (+ output_copy_checks at step 4)         [GPU]
    obligations_v5.py prover witness: selected VUs -> statements; unsupported selections recorded, never "exempt (charged)"
    verify_v5.py      step 5: verifier_v3.reverify + check_expected_positions (A2) + coverage_v1 (A5/6) + certificate_v2 (A7)
    costlog_v5.py     the contract's eight cost classes, measured separately, totals without double counting
    certificate_v5.py the certificate for exactly this configuration (certificate_v2) + completion risk (R18)
    report_v5.py      the run report per docs/PROTOCOL.md §Reporting requirements

Nothing in ``record_6k/``, ``e2e/*``, ``verity_vllm_adapter/*`` or ``src/veritor/**`` is edited; those modules are imported.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
POC = ROOT / "vllm-poc"

__all__ = ["HERE", "ROOT", "POC", "add_paths"]


def add_paths() -> None:
    """Put this repo's ``vllm-poc``, ``scripts``, ``src`` and ``record_6k`` (read-only, by import) on ``sys.path``
    AFTER any dependency export ``deps.py`` inserted (a branch export must shadow the worktree's older copy)."""

    for p in (str(POC), str(POC / "scripts"), str(ROOT), str(ROOT / "src"), str(ROOT / "record_6k")):
        if p not in sys.path:
            sys.path.append(p)
