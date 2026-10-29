"""The statement `run.sh` proves: ``N`` instances of ``S8Matmul_v1{N=ROWS, K}``, one int8 row against ``ROWS`` int8 rows with
exact 32-bit sums (the plain arm: no noise, no tail, ``COMMIT = "final"``), as the population program of that template
(`verity.ml.circuits.population`), cut by ``Q_template_instance``. Each instance is one proof block of ``ROWS`` adjacent cells
of one output row; `verity_flock.class_statement` hashes its rows (the row ``a``, the ``ROWS`` rows of ``b`` and the ``ROWS``
32-bit outputs) in the circuit under ``hm96-sha512``. ``K``, ``ROWS`` and ``N`` come from the environment (`settings`).

    python -m verity_flock.class_statement --program-module program.py:program --partition q-template-instance ...
"""
from __future__ import annotations

import importlib.util
from pathlib import Path


def _settings():
    spec = importlib.util.spec_from_file_location("flock_s8_block_settings", Path(__file__).with_name("settings.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def template():
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops import s8 as S8

    s = _settings()
    return bind(S8.S8Matmul, N=s.rows(), **S8.SMOKE_PLAIN | {"K": s.k(), "L": s.k()})


def program():
    from verity.ml.circuits.population import population_program

    return population_program(template(), _settings().n())
