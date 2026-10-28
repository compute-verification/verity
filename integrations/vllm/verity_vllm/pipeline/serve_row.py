"""serve-openai's scheduler: ``--async-scheduling`` (off by default) sets the engine's, and a bench row's Programs must be
specialized for it.

The engine is built with vLLM's ``async_scheduling`` as ``--async-scheduling`` says (`serve_openai.engine_options`), never
left to vLLM's own resolution (on at the pinned revision). Under ``--required-manifest`` the row's Build of record (predict's,
`serve_seal.row_refusal`) records the engine its workload Program models in `WORKLOAD_PROGRAM` (`global_program`'s
``applicability.engine``). Its ``async_scheduling`` is the value the Program's ``lag`` static is built from (LAG 1 under
async scheduling, 0 synchronous), and the one a record of the row must attest (GM-01 G1). serve-openai reads it (`of`;
``serve.json``'s ``row_scheduling``) and refuses a flag that disagrees with it before the model loads, and refuses a server
whose engine resolved other than the flag (`checked`) before it listens. Under the flag ``serve.json``'s target declares
async scheduling too (`serve_seal.target_of`), so the φ a served window's Programs are predicted from is specialized for
LAG 1 by declaration. The Commit (and H1 on it) serves under the flag as it does synchronously: the native collector learns
the declared static shape's plans before the server listens (`serve_learn`), and an async step commits the rows the
synchronous step does, under the same rows-sha512 root (``tests/pipeline/serve_async_commit_gpu.py``).
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from verity_vllm.config import UsageError
from verity_vllm.pipeline import serve_seal as SS

#: where a row's Build records the engine its workload Program models (``applicability.engine``: `global_program`'s, and
#: `program.predict.aot.write_row`'s)
WORKLOAD_PROGRAM = Path("build_workload", "workload_program.json")


def _flag(on: bool) -> str:
    return "--async-scheduling" if on else "--no-async-scheduling"


def of(opts: Any) -> dict[str, Any] | None:
    """The scheduler the bench row's workload Program is specialized for, ``{"async_scheduling", "source"}``: what its Build of
    record's `WORKLOAD_PROGRAM` (`serve_seal.programs_root_of`) declares as ``applicability.engine.async_scheduling``.
    None without ``--required-manifest``; `UsageError` naming the file when the row has none, it declares no boolean, or it
    declares other than ``--async-scheduling``."""
    if not opts.required_manifest:
        return None
    path = Path(SS.programs_root_of(opts)) / WORKLOAD_PROGRAM
    try:
        got = json.loads(path.read_text())["applicability"]["engine"]["async_scheduling"]
    except OSError as e:
        raise UsageError(f"--required-manifest {opts.required_manifest}: the row's Build has no workload Program to take the "
                         f"engine's scheduler from ({type(e).__name__}: {e})") from None
    except (ValueError, KeyError, TypeError) as e:
        raise UsageError(f"{path} declares no applicability.engine.async_scheduling ({type(e).__name__}: {e}): the engine has no "
                         "scheduler of the row's to follow") from None
    if not isinstance(got, bool):
        raise UsageError(f"{path} declares applicability.engine.async_scheduling {got!r}, not a boolean: the engine has no "
                         "scheduler of the row's to follow")
    if got is not bool(opts.async_scheduling):
        raise UsageError(f"{_flag(bool(opts.async_scheduling))}: the bench row's workload Program is specialized for "
                         f"async_scheduling {got} (LAG {int(got)}, {path}), so its records would attest a scheduler the row's "
                         f"Programs don't model: serve the row with {_flag(got)}, or a row whose Build's target declares "
                         f"\"async_scheduling\": {json.dumps(bool(opts.async_scheduling))}")
    return {"async_scheduling": got, "source": str(path)}


def refusal(want: bool, row: Mapping[str, Any] | None, engine: Any) -> str | None:
    """Why the built ``engine`` (a ``vllm.LLM``) can't serve: it resolved ``async_scheduling``
    (`engine.build.resolved_async_scheduling`) other than ``want``, ``--async-scheduling``'s value, which the bench row
    ``row`` (`of`), if any, declares. None when it resolved ``want``."""
    from verity_vllm.engine.build import resolved_async_scheduling

    if (got := resolved_async_scheduling(engine)) is want:
        return None
    return (f"async_scheduling: the engine resolved {got!r}, and serve-openai built it for {want!r} ({_flag(want)})"
            + (f", the scheduler the bench row's workload Program is specialized for (LAG {int(want)}, {row['source']}): its "
               "records would attest a scheduler the row's Programs don't model" if row is not None else ""))


def checked(opts: Any, row: Mapping[str, Any] | None, llm: Any) -> Any:
    """``llm`` (a `verity_vllm.LLM`) once its engine schedules as ``--async-scheduling`` says (and the bench row ``row``
    declares), with the scheduler it resolved printed; `RuntimeError` (`refusal`) otherwise."""
    want = bool(opts.async_scheduling)
    if why := refusal(want, row, llm.vllm):
        raise RuntimeError(why)
    print(f"SERVE-OPENAI engine resolved async_scheduling={want} ({_flag(want)}); "
          + (f"the bench row declares {row['async_scheduling']} ({row['source']})" if row is not None else "no bench row"), flush=True)
    return llm
