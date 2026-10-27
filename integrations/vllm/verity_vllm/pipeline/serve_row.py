"""serve-openai's bench row: the engine schedules as the row's workload Program is specialized for, not as vLLM defaults.

Under ``--required-manifest`` the row's Build records the engine its workload Program models in `WORKLOAD_PROGRAM`
(`global_program`'s ``applicability.engine``). Its ``async_scheduling`` is the value the Program's ``lag`` static is built
from, and the one a record of the row must attest. serve-openai reads it (`of`; ``serve.json``'s ``row_scheduling``),
builds its engine with it (`engine_args`), and refuses a server whose engine resolved another (`checked`) before it
listens.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from verity_vllm.config import UsageError
from verity_vllm.pipeline import serve_seal as SS

#: where a bench row's Build records the engine its workload Program models (`global_program`'s ``applicability.engine``)
WORKLOAD_PROGRAM = Path("build_workload", "workload_program.json")


def of(opts: Any) -> dict[str, Any] | None:
    """The scheduler the bench row's workload Program is specialized for, ``{"async_scheduling", "source"}``: what its Build's
    `WORKLOAD_PROGRAM` in the row dir (`serve_seal.programs_root_of`) declares as ``applicability.engine.async_scheduling``.
    None without ``--required-manifest``; `UsageError` naming the file when the row has none or it declares no boolean."""
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
    return {"async_scheduling": got, "source": str(path)}


def engine_args(row: Mapping[str, Any] | None) -> dict[str, Any]:
    """`verity_vllm.LLM`'s keyword arguments the bench row (`of`) fixes: its ``async_scheduling``; none without a row."""
    return {} if row is None else {"async_scheduling": row["async_scheduling"]}


def refusal(row: Mapping[str, Any] | None, engine: Any) -> str | None:
    """Why the built ``engine`` (a ``vllm.LLM``) can't serve the bench row ``row`` (`of`): it resolved ``async_scheduling``
    (`engine.build.resolved_async_scheduling`) other than the row declares. None without a row."""
    if row is None:
        return None
    from verity_vllm.engine.build import resolved_async_scheduling

    if (got := resolved_async_scheduling(engine)) is row["async_scheduling"]:
        return None
    return (f"async_scheduling: the engine resolved {got!r}, and the bench row's workload Program is specialized for "
            f"{row['async_scheduling']!r} (LAG {int(row['async_scheduling'])}, {row['source']}): its records would attest a "
            "scheduler the row's Programs don't model")


def checked(row: Mapping[str, Any] | None, llm: Any) -> Any:
    """``llm`` (a `verity_vllm.LLM`) once its engine schedules as the bench row ``row`` declares, with the scheduler it
    resolved printed; `RuntimeError` (`refusal`) otherwise."""
    if why := refusal(row, llm.vllm):
        raise RuntimeError(why)
    from verity_vllm.engine.build import resolved_async_scheduling

    print(f"SERVE-OPENAI engine resolved async_scheduling={resolved_async_scheduling(llm.vllm)}; "
          + (f"the bench row declares {row['async_scheduling']} ({row['source']})" if row is not None else "no bench row"), flush=True)
    return llm
