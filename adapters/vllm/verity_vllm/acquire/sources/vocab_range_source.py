"""A tensor-parallel rank's vocabulary range (`query.vocab_range`) as an acquisition source: per embedding site and step, the committed
range values of every token row, `<embedding module>/vocab_range` (int32 [rows, 3] = tok - START, START <= tok, tok <= START + VS - 1),
in the same step stream and run root as the rank's other values.  Opt-in (the row's VOCAB_TAP=1, commit `--vocab-range`), never the record.

The sites are the rank's required manifest's `vocab_range` identities.  A forward pre-hook marks the site module that runs; inside it,
during a committed step, the wrapper of `torch.ops._C.vocab_parallel_embedding` (the fused TP embedding kernel `EmbeddingShard_v1`
states) computes the three values from the launch's own token ids with the circuit's integer semantics (`query.vocab_range.values`: wrapping
32-bit subtract, signed 32-bit compares) on the device, then runs the installed kernel unchanged.  No kernel changes: the values are
integer functions of the committed token id and the rank's static range, and the vocabulary-range exactness property checks them against
the IR and against the kernel's own output (zero rows exactly where the range bits say the rank does not own the token).  A launch whose
range is not the manifest's, or on a padded or added-vocabulary shard (which `EmbeddingShard_v1` does not state), is a collector error.
"""
from __future__ import annotations

import time
from typing import Any, Mapping

from verity_vllm.engine import hooks
from verity_vllm.query import vocab_range as VR


def range_values(torch: Any, ids: Any, start: int, vs: int) -> Any:
    """int32 [rows, 3] = (tok - START, START <= tok, tok <= START + VS - 1) of a token-id tensor, on its device (`query.vocab_range.values`)."""
    t = ids.reshape(-1).to(torch.int64) & 0xFFFFFFFF
    t = t - ((t >> 31) << 32)                                   # the id's 32-bit word, signed
    local = (t - int(start)) & 0xFFFFFFFF
    local = local - ((local >> 31) << 32)
    ge = (t >= VR._s32(start)).to(torch.int64)
    le = (t <= VR._s32(start + vs - 1)).to(torch.int64)
    return torch.stack([local, ge, le], dim=-1).to(torch.int32)


def module_of(op: str, mods: Mapping[str, Any]) -> str | None:
    """The model module a manifest op path names: itself, else the one dotted suffix of it that is a module.  A global TP manifest
    spells paths with the Build's export-wrapper prefix and records none ([GLM-5.3] `inner.model.model.embed_tokens` names the
    implementation's `model.embed_tokens`); two candidate suffixes are ambiguous and name nothing."""
    if op in mods:
        return op
    parts = op.split(".")
    hits = [s for s in (".".join(parts[k:]) for k in range(1, len(parts))) if s in mods]
    return hits[0] if len(hits) == 1 else None


class VocabRangeSource:
    short_name = "vocab_range"

    def __init__(self, committer: Any, model: Any, sites: Mapping[str, Mapping[str, int]]) -> None:
        """`sites`: embedding module op path -> {"start", "shard_rows"} (`query.vocab_range.manifest_sites`)."""
        import torch

        if not sites:
            raise ValueError("vocab range: no embedding site to tap")
        self.torch, self.c, self.model = torch, committer, model
        self.sites = {str(k): dict(v) for k, v in sites.items()}
        self._hooks = hooks.Hooks()
        self._handles: list[Any] = []
        self._scope: list[str] = []
        self._held: list[Any] = []
        self.stats: dict[str, Any] = {"tensors": 0, "bytes": 0, "rows": 0, "sites": len(self.sites), "errors": 0}
        self.active = False

    def install(self) -> None:
        mods = dict(self.model.named_modules())
        where = {op: module_of(op, mods) for op in self.sites}
        missing = sorted(op for op, m in where.items() if m is None)
        if missing:
            raise RuntimeError(f"vocab range: the manifest's embedding sites are not modules of the model: {missing[:4]}")
        for op, m in where.items():
            self._handles.append(mods[m].register_forward_pre_hook(self._enter(op)))
            self._handles.append(mods[m].register_forward_hook(self._leave(op), always_call=True))
        self._hooks.patch(self.torch.ops._C, "vocab_parallel_embedding", self._wrap(self.torch.ops._C.vocab_parallel_embedding))
        self.active = True

    def uninstall(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
        self._hooks.uninstall()
        self._scope.clear()
        self._held.clear()
        self.active = False

    def on_step_begin(self) -> None:
        self._held.clear()

    def _enter(self, op: str) -> Any:
        def pre(_module: Any, _args: tuple) -> None:
            self._scope.append(op)

        return pre

    def _leave(self, op: str) -> Any:
        def post(_module: Any, _args: tuple, _out: Any) -> None:
            if self._scope and self._scope[-1] == op:
                self._scope.pop()

        return post

    def error(self, msg: str) -> None:
        self.stats["errors"] += 1
        self.c.errors.append(msg[:300])

    def _wrap(self, orig: Any) -> Any:
        me, torch = self, self.torch

        def vocab_parallel_embedding(out: Any, input_ids: Any, weight: Any, start: int, end: int, padding: int, added_start: int,
                                     added_end: int) -> None:
            job = me.c._cur
            if job is not None and me._scope:
                op = me._scope[-1]
                g = me.sites[op]
                if (int(start), int(end) - int(start)) != (g["start"], g["shard_rows"]) or int(padding) != 0 or int(added_start) != int(added_end):
                    me.error(f"vocab range: {op} launched vocab_parallel_embedding over [{start}, {end}) padding {padding} added "
                             f"[{added_start}, {added_end}); the manifest's shard is [{g['start']}, {g['start'] + g['shard_rows']}), plain")
                else:
                    t0 = time.perf_counter()
                    v = range_values(torch, input_ids, g["start"], g["shard_rows"])
                    me._held.append(v)
                    me.c._acquire(f"{op}/{VR.MEMBER}", v, job)
                    n = int(v.numel()) * 4
                    me.stats["tensors"] += 1
                    me.stats["bytes"] += n
                    me.stats["rows"] += int(v.shape[0])
                    spans = getattr(me.c, "spans", None)
                    if spans is not None:
                        spans.add("acquisition", time.perf_counter() - t0, step=job.step, nbytes=n)
            return orig(out, input_ids, weight, start, end, padding, added_start, added_end)

        return vocab_parallel_embedding

    def extra_step_bytes(self, envelope: Mapping[str, Any], model: Any) -> dict[str, int]:
        return {"per_token_bytes": 4 * VR.WORDS * len(self.sites), "per_seq_bytes": 0, "tensors_per_step": len(self.sites)}

    def declaration_classes(self) -> list[dict]:
        return [{"class": VR.FAMILY, "how": "bound-as-leaf", "rule": None, "count": self.stats["tensors"], "bytes": self.stats["bytes"],
                 "tensors": self.stats["tensors"], "rows": self.stats["rows"], "errors": self.stats["errors"],
                 "evidence": (f"vocabulary-range source: `<embedding module>/{VR.MEMBER}` of {len(self.sites)} sites, computed from the launch's "
                              "token ids (no kernel change); every value == the IR's EmbeddingShard_v1 value and the kernel's zero rows agree: "
                              "verity_vllm.properties.vocab_range_exactness")}]


def attach(com: Any, enabled: bool, manifest: Mapping[str, Any] | None, required_classes: tuple[str, ...], *,
           explicit_classes: bool = False, model: Any = None) -> tuple[tuple[str, ...], VocabRangeSource | None]:
    """Attach the source to `com` when `enabled`; the required manifest must name `vocab_range` identities, else the Commit is refused
    (exit 3).  Returns the required classes with `vocab_range` added unless they were given explicitly."""
    if not enabled:
        return required_classes, None
    sites = VR.manifest_sites(manifest)
    if not sites:
        print("[delta] REFUSE: --vocab-range but the required-value manifest names no vocab_range identity (manifest build --vocab-range)",
              flush=True)
        raise SystemExit(3)
    src = VocabRangeSource(com, model if model is not None else getattr(com, "model", None), sites)
    com.attach_source(src)
    print(f"[delta] vocabulary-range source attached: {len(sites)} sites -> committer {com.name}", flush=True)
    if not explicit_classes and VR.FAMILY not in required_classes:
        return (*required_classes, VR.FAMILY), src
    return required_classes, src


__all__ = ["VocabRangeSource", "range_values", "attach"]
