"""Flock's settings: the variables of ``verity_flock`` and its tests that are declared here, read by path
(``verity_config.settings.of(__file__)``).

It owns no prefix yet: the rest of Flock's variables (``VERITY_FLOCK_CACHE``, ``FLOCK_IR_SLOW``, ...) are still read directly,
by the files ``tests/lint/allowlists/repo_env_reads.json`` lists, until they move here.
"""
from __future__ import annotations

from verity_config.settings import WHERE, Registry, Setting as S, flag

REGISTRY = Registry(
    __file__, (),
    S("VERITY_FLOCK_STRUCT_TYPES", flag, False, WHERE, "which tests run: 1 stages each served type that reads a table, "
                                                       "flat and structural (test_tmpl_statement; about ten minutes, 40 GB)"),
)
