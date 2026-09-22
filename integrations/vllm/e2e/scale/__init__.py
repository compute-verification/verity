"""E3's SCALE-wave package (D46, 2026-09-12): the `costlog/v1` cost schema for one protocol run and the paper-table generator.

* :mod:`e2e.scale.costlog` -- schema constants, ``new()``, ``validate()``, ``bound_statement()``, ``from_record_v5()`` (a record's
  ``costlog_v5.json`` / ``verify_v5.json`` / ``reverify_clean.json`` / coverage accounting -> one costlog), CLI ``python -m e2e.scale.costlog``.
* :mod:`e2e.scale.table` -- ``python -m e2e.scale.table <costlog dir> <night table dir> -o out/scale/TABLE.md``: the results table (MEASURED rows,
  the MODELED column beside, guarantee columns, verdicts), the per-stage cost split, a CSV and the log-log figure.
"""
