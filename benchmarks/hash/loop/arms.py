"""The hash loop's extra arms over circuits' rows arm (``served_commit.prof_arm.RowsArm``).

``served-commit-rows-blake3``: the sibling arm, committing under ``rows-blake3`` (``verity_vllm.commit.step_rows_b3``: the same
step with keyed BLAKE3 in every hash position).  Research only: BLAKE3 never serves.

``served-commit-rows-whatif-q``: what-if (a), q norm and RoPE moved into the attention unit's prologue, so the post-RoPE q leaves
the Commit (8 KiB of Qwen3-8B's 71,680 B a token a layer).  The source drops each compiled piece's ``cat`` tap shaped
[tokens, q heads, head dim]; k, v and the qkv projection stay committed, in the same layout.  What-if only: the unit that would
recompute q is not written, so its roots are no scheme's and it does not serve.

``job.sh`` names both to its copy of prof.sh."""
from __future__ import annotations

from typing import Any

from served_commit.prof_arm import RowsArm


class RowsBlake3Arm(RowsArm):
    commit_scheme = "rows-blake3"


def _q_prologue_source(**kw: Any) -> Any:
    import torch
    from verity_vllm.acquire.sources.served_compiled_source import ServedCompiledSource

    class QPrologueSource(ServedCompiledSource):
        # the engine captures its graphs before the arm sees the model, so the shape is the benchmark model's (Qwen3-8B, TP 1)
        # and note_engine refuses any other
        q_shape = (32, 128)
        kv_heads = 8

        def note_engine(self, model: Any) -> None:
            super().note_engine(model)
            att = next(m for m in model.modules() if hasattr(m, "num_kv_heads") and hasattr(m, "head_size"))
            got = (int(att.num_heads), int(att.head_size), int(att.num_kv_heads))
            if got != (*self.q_shape, self.kv_heads):
                raise ValueError(f"what-if q: attention (q heads, head size, kv heads) {got}, not Qwen3-8B's; "
                                 "set q_shape and kv_heads")

        def _tap(self, sub: int, name: str, val: torch.Tensor) -> None:
            if (name.split(".", 1)[1].split("/", 1)[0] == "cat" and val.dim() >= 2
                    and tuple(val.shape[-2:]) == self.q_shape):
                self.stats["whatif_q_dropped"] = self.stats.get("whatif_q_dropped", 0) + 1
                return
            super()._tap(sub, name, val)

    return QPrologueSource(**kw)


class QPrologueArm(RowsArm):
    def install(self, target: Any) -> QPrologueArm:
        self.source = _q_prologue_source(into=target.hooks, batch_copies=self.batch_copies)
        self.source.install_hooks()
        return self


SERVED_COMMIT_ROWS_BLAKE3 = RowsBlake3Arm("served-commit-rows-blake3", retain="none")
SERVED_COMMIT_ROWS_WHATIF_Q = QPrologueArm("served-commit-rows-whatif-q", retain="none")
