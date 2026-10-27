"""The served arm: the Commit at rows-sha512's 4 KiB rows timed against stock as vLLM serves it (compiled) and against the
batch-invariant eager stock every eager row is timed against, in one interleaved run on one GPU (`served_arm.sh`, the specs
`*-rtx-served-arm`). Measurement only; `benchmarks/g2/served.py` renders its runs.

- ``served-commit-rows-4k`` (`ROWS_4K`, `ROWS_4K_FP8`): the compiled Commit arm (`prof_arm.Rows4kArm`, the served compiled
  source under CUDA graphs), built at vLLM's serving defaults compiled whatever the spec's ``eager`` (its ``execution``), with
  ``stock@compiled`` (``stock-fp8@compiled``) as its token control.
- ``commit-rows-4k-eager`` (`EAGER_ROWS_4K`, `EAGER_ROWS_4K_FP8`): the harness's eager ``commit`` (`verity_bench.vllm.commit_arm`)
  with every chunk constant at 4 KiB, as `prof_arm.Rows4kArm` sets them, on the spec's batch-invariant eager engine, with
  ``stock`` (``stock-fp8``) as its token control.
"""
from __future__ import annotations

from typing import Any

from served_commit.prof_arm import Rows4kArm
from verity_bench.vllm.commit_arm import CommitArm

ROW_BYTES = 4096


class CompiledRows4kArm(Rows4kArm):
    execution = "compiled"

    def __init__(self, id: str, *, precision: str, control: str) -> None:
        super().__init__(id, retain="none")
        self.precision, self.control = precision, control


class EagerRows4kArm(CommitArm):
    def __init__(self, id: str, *, precision: str, control: str) -> None:
        super().__init__(id)
        self.precision, self.control = precision, control

    def ready(self, target: Any) -> None:
        from verity_vllm.commit import step_rows
        from verity_vllm.commit.committer import (
            host_retention,
            native_collect,
            native_host,
        )

        step_rows.ROW_BYTES = native_host.GPU_CHUNK_BYTES = native_collect.GPU_CHUNK_BYTES = host_retention.GPU_CHUNK_BYTES = ROW_BYTES
        super().ready(target)

    def gate(self, target: Any) -> dict[str, Any]:
        return {**super().gate(target), "row_bytes": ROW_BYTES}


ROWS_4K = CompiledRows4kArm("served-commit-rows-4k", precision="bfloat16", control="stock@compiled")
ROWS_4K_FP8 = CompiledRows4kArm("served-commit-rows-4k", precision="fp8", control="stock-fp8@compiled")
EAGER_ROWS_4K = EagerRows4kArm("commit-rows-4k-eager", precision="bfloat16", control="stock")
EAGER_ROWS_4K_FP8 = EagerRows4kArm("commit-rows-4k-eager", precision="fp8", control="stock-fp8")
