"""The sampler's inputs of chosen requests, taken in an engine that replays them at their served step shapes.

`install`, run in every worker (``engine.collective_rpc``), wraps the model runner's sampler: before each call it copies
the logits rows of the target requests whose scheduled tokens reach the end of their prompt (the rows the runner keeps a
token of), as bf16 bits, and after it their sampled ids. `take`, run in every worker, writes each target's rows under
``<directory>/rank<r>/`` and returns their ids and sha256s. `spans` lays them out: rank 0's rows, which every other rank
that sampled the request must have taken bit for bit with the same ids, at the engine steps its tokens were emitted at,
with the launch that fixes its Program. `write` writes them as `records.logits_capture`, which
`pipeline.interior_rows.captured` reads as a token select's span. CPU replay writes the same record in tests.

The runner is vLLM's V2 model runner (`v1.worker.gpu.model_runner`), whose sampler takes ``(logits, input_batch)``. A
runner that shards sampling across ranks (``batch_sharder``), samples draft tokens, or hands the sampler logits that
aren't bf16 or one row a request is refused: a token select's unit reads one gathered bf16 row a token."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from verity_vllm.records.logits_capture import LogitsCapture

__all__ = ["CAPTURE", "Capture", "install", "launch", "spans", "take", "write"]

#: a capture's record, beside its rows' files
CAPTURE = "capture.json"


class Capture:
    """``sampler`` wrapped: each target's kept rows (u16 bits) and ids, in call order. Every other attribute is the
    sampler's."""

    def __init__(self, sampler: Any, targets: Collection[str]) -> None:
        self.inner, self.targets = sampler, frozenset(targets)
        self.rows: dict[str, list[np.ndarray]] = {r: [] for r in self.targets}
        self.ids: dict[str, list[int]] = {r: [] for r in self.targets}

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def __call__(self, logits: Any, input_batch: Any, *a: Any, **kw: Any) -> Any:
        import torch
        b = input_batch
        keep = [i for i, r in enumerate(b.req_ids[:b.num_reqs]) if r in self.targets
                and b.num_computed_tokens_np[i] + b.num_scheduled_tokens[i] >= b.prefill_len_np[i]]
        rows = None
        if keep:
            if b.num_draft_tokens or logits.shape[0] != b.num_reqs or logits.dtype != torch.bfloat16:
                raise TypeError(f"the sampler takes {tuple(logits.shape)} {logits.dtype} logits for {b.num_reqs} requests "
                                f"and {b.num_draft_tokens} draft tokens: a token select reads one bf16 row a token")
            rows = logits[keep].contiguous().view(torch.int16).cpu().numpy().view("<u2")
        out = self.inner(logits, input_batch, *a, **kw)
        if keep:
            ids = out.sampled_token_ids[keep, 0].tolist()
            for k, i in enumerate(keep):
                self.rows[b.req_ids[i]].append(rows[k].copy())
                self.ids[b.req_ids[i]].append(int(ids[k]))
        return out


def install(worker: Any, targets: Sequence[str]) -> int:
    """Wrap ``worker``'s sampler for ``targets`` (the requests' engine ids); its rank."""
    runner = worker.model_runner
    if getattr(runner, "batch_sharder", None) is not None:
        raise TypeError("the runner shards sampling across ranks: each rank's logits are an all-to-all's, not the gathered row")
    runner.sampler = Capture(runner.sampler, targets)
    return int(worker.rank)


def take(worker: Any, directory: str) -> dict[str, Any]:
    """Write the worker's kept rows under ``<directory>/rank<r>/<k>.u16`` (k a target's place in sorted order); each
    target's file, sha256, ids and row count, and the rank."""
    c, rank = worker.model_runner.sampler, int(worker.rank)
    d = Path(directory) / f"rank{rank}"
    d.mkdir(parents=True, exist_ok=True)
    out = {}
    for k, rid in enumerate(sorted(c.targets)):
        if not c.rows[rid]:
            continue
        raw = np.stack(c.rows[rid]).tobytes()
        (d / f"{k}.u16").write_bytes(raw)
        out[rid] = {"file": str(d / f"{k}.u16"), "sha256": hashlib.sha256(raw).hexdigest(), "ids": c.ids[rid],
                    "rows": len(c.rows[rid])}
    return {"rank": rank, "spans": out}


def launch(rid: str, steps: Sequence[Mapping[str, Any]]) -> str:
    """What fixes request ``rid``'s Program at its served shapes: over the steps holding a row of it, its row's (start, n)
    and every row's, in the step's order (`StepTrace` rows ``(request, start, n)``)."""
    life = [[next([r[1], r[2]] for r in s["rows"] if r[0] == rid), [[r[1], r[2]] for r in s["rows"]]]
            for s in steps if any(r[0] == rid for r in s["rows"])]
    return "launch:" + hashlib.sha256(json.dumps(life, separators=(",", ":")).encode()).hexdigest()


def spans(takes: Sequence[Mapping[str, Any]], steps: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """`write`'s spans from every rank's `take` and the replay's own step trace: each target's rows from rank 0,
    held to every other rank's that sampled it, cut to the tokens its steps emitted (``emitted``), each with its step."""
    at: dict[str, list[int]] = {}
    for s in steps:
        for rid, n in s["emitted"].items():
            at.setdefault(rid, []).extend([s["step"]] * int(n))
    by = {t["rank"]: t["spans"] for t in takes}
    out = {}
    for rid, mine in sorted(by.get(0, {}).items()):
        for r, other in sorted(by.items()):
            if rid in other and (other[rid]["sha256"], other[rid]["ids"]) != (mine["sha256"], mine["ids"]):
                raise ValueError(f"request {rid}: rank {r} took other logits or ids than rank 0")
        n = len(at.get(rid, ()))
        if mine["rows"] < n:
            raise ValueError(f"request {rid}: {mine['rows']} rows taken for {n} tokens emitted")
        rows = np.fromfile(mine["file"], dtype="<u2").reshape(mine["rows"], -1)[:n]
        out[rid] = {"logits": rows, "tokens": mine["ids"][:n], "steps": at[rid], "program": launch(rid, steps)}
    return out


def write(directory: Path, template: str, spans: Mapping[str, Mapping[str, Any]]) -> Path:
    """Each request's span of ``spans`` written to ``directory`` (`records.logits_capture`, read by
    `pipeline.interior_rows.captured`): its ``logits`` ((n, V) u16, row j the bf16 bits its j-th token was selected from),
    ``tokens`` (the n ids selected), ``steps`` (the n engine steps) and ``program`` (what fixes its Program: the Program's
    digest, or its launch's). ``template`` is the token-select row template the rows are units of."""
    directory.mkdir(parents=True, exist_ok=True)
    out = {}
    for k, (rid, s) in enumerate(spans.items()):
        a = np.ascontiguousarray(s["logits"], dtype="<u2")
        (directory / f"{k}.u16").write_bytes(a.tobytes())
        out[rid] = {"file": f"{k}.u16", "sha256": hashlib.sha256(a.tobytes()).hexdigest(), "rows": list(a.shape),
                    "tokens": [int(t) for t in s["tokens"]], "steps": [int(t) for t in s["steps"]],
                    "program": str(s["program"])}
    path = directory / CAPTURE
    doc = LogitsCapture(template, out).dump()
    LogitsCapture.of(doc, str(path))
    path.write_text(json.dumps(doc, indent=1))
    return path
