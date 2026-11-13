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
from collections.abc import Callable, Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from verity_vllm.records.logits_capture import LogitsCapture

__all__ = ["CAPTURE", "Capture", "emitting", "install", "kept_rows", "launch", "piece_rows", "spans", "take", "write"]

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
        b = input_batch
        keep = emitting(b, self.targets.__contains__)
        rows = kept_rows(logits, b, keep) if keep else None
        out = self.inner(logits, input_batch, *a, **kw)
        if keep:
            ids = out.sampled_token_ids[keep, 0].tolist()
            for k, i in enumerate(keep):
                self.rows[b.req_ids[i]].append(rows[k].copy())
                self.ids[b.req_ids[i]].append(int(ids[k]))
        return out


def emitting(b: Any, wanted: Callable[[str], bool]) -> list[int]:
    """The rows of the sampler's batch ``b`` whose request ``wanted`` takes and emits a token at this call: its scheduled
    tokens reach the end of its prompt, so the runner keeps the token sampled from that row."""
    return [i for i, r in enumerate(b.req_ids[:b.num_reqs]) if wanted(r)
            and b.num_computed_tokens_np[i] + b.num_scheduled_tokens[i] >= b.prefill_len_np[i]]


def kept_rows(logits: Any, b: Any, keep: Sequence[int]) -> np.ndarray:
    """Rows ``keep`` of the sampler's ``logits`` for batch ``b``, as bf16 bits (u16) on the host. Refused for a batch with
    draft tokens, or logits that aren't bf16 or one row a request: a token select reads one bf16 row a token."""
    import torch

    if b.num_draft_tokens or logits.shape[0] != b.num_reqs or logits.dtype != torch.bfloat16:
        raise TypeError(f"the sampler takes {tuple(logits.shape)} {logits.dtype} logits for {b.num_reqs} requests "
                        f"and {b.num_draft_tokens} draft tokens: a token select reads one bf16 row a token")
    return logits[list(keep)].contiguous().view(torch.int16).cpu().numpy().view("<u2")


def piece_rows(logits: Any, ids: Any, c: int | None = None, *, pad: int | None = None,
               template: str | None = None) -> dict[str, dict[str, Any]]:
    """A token select's pieces' rows on the logits' device, as `pipeline.interior_rows.pieces` lays them out on the host:
    ``logits`` the n kept rows (bf16, or their int16 bits), ``ids`` the n sampled ids, ``template`` the token select
    the rows are units of (`TokenSelect_v4{V}` when None), ``c`` `TokenSelect_v4`'s chunk (``min(CHUNK, V)``). By
    template, its ports' u16 words as int16, each input row named after its parameter. `TokenSelect_v4`: each token's
    V / c `TokenSelectChunk_v2` units (a chunk of its row in, its first maximum above -inf and that maximum's index out,
    a NaN never taken), then its `TokenSelectCombine_v2` unit (logits[0] and the chunks' outputs in, the token out).
    `GreedySelect_v1{V, N}`: each token's one unit of each of `greedy_stages.plan`'s templates (`_greedy_rows`).
    Refused unless the combine selects every id. ``pad``: the size class's tokens, the last row and id repeated to it
    first, as `Span.padded` pads `pieces`' cells by whole tokens."""
    import torch

    from verity.definitions.registry.b1 import NEG_INF_BF16
    from verity.definitions.registry.dead_free import CHUNK
    x = logits.view(torch.int16) if logits.dtype == torch.bfloat16 else logits
    n, V = x.shape
    ids = torch.as_tensor(ids, device=x.device).to(torch.int64)
    if pad is not None:
        if not 1 <= n <= pad:
            raise ValueError(f"{n} tokens don't pad to a size class of {pad}")
        x, ids, n = torch.cat([x, x[-1:].expand(pad - n, V)]), torch.cat([ids, ids[-1:].expand(pad - n)]), pad
    if template is not None and template != f"TokenSelect_v4{{V={V}}}":
        if c is not None:
            raise ValueError(f"{template}: its chunks are its own, not {c}")
        return _greedy_rows(x, ids, template)
    c = min(CHUNK, V) if c is None else int(c)
    if not 1 <= c <= V or V % c:
        raise ValueError(f"{V} logits in chunks of {c} are not whole chunks")
    k = V // c
    f = x.view(torch.bfloat16).float()
    f = torch.where(torch.isnan(f), float("-inf"), f).view(n * k, c)
    idx = f.argmax(dim=1)
    none = ~(f > float("-inf")).any(dim=1)
    best = torch.where(none, NEG_INF_BF16 - (1 << 16), x.reshape(n * k, c).gather(1, idx[:, None])[:, 0])
    idx = torch.where(none, 0, idx)
    fb = best.view(torch.bfloat16).float().view(n, k)
    j = fb.argmax(dim=1)
    take = fb.gather(1, j[:, None])[:, 0] > x[:, 0].view(torch.bfloat16).float()
    tok = torch.where(take, j * c + idx.view(n, k).gather(1, j[:, None])[:, 0], 0)
    if not torch.equal(tok, ids):
        raise ValueError(f"the chunks and combine select {tok.tolist()[:8]}..., the sampler {ids.tolist()[:8]}...")
    i32 = idx.to(torch.int32).view(torch.int16)
    return {f"TokenSelectChunk_v2{{C={c}}}": {"x": x.reshape(n * k, c), "o0": best[:, None], "o1": i32.view(n * k, 2)},
            f"TokenSelectCombine_v2{{V={V},C={c}}}": {"x0": x[:, :1], "best": best.view(n, k), "idx": i32.view(n, 2 * k),
                                                      "o0": ids.to(torch.int32).view(torch.int16).view(n, 2)}}


def _greedy_rows(x: Any, ids: Any, template: str) -> dict[str, dict[str, Any]]:
    """`piece_rows` of `GreedySelect_v1{V, N}` (``template``) on the n rows ``x`` (int16 bits) and ids ``ids``, as
    `greedy_stages.twin` computes each unit: chunk 0 scans from (logits[0], 0), keeping a NaN logits[0] and taking a lane
    only strictly above it; a later chunk is its first maximum above -inf and that lane, (-inf, 0) where it has none; the
    combine takes chunk c's (best, OFF + i) from chunk 0's iff its best is not a NaN and strictly below chunk c's. A
    chunk's rows are its lanes and its output its best's word and its token index's low and high words; the combine's
    rows are the last chunk's lanes and each other chunk's output, and its output the token's two words."""
    import torch

    from verity.circuits.codec import descriptor_id
    from verity.definitions.registry import greedy_stages as GS
    from verity.definitions.registry.b1 import NEG_INF_BF16
    n, V = x.shape
    head, _, rest = template.partition("{")
    statics = dict(kv.split("=", 1) for kv in rest.rstrip("}").split(",")) if rest else {}
    if head != "GreedySelect_v1" or set(statics) != {"V", "N"} or not all(v.isdigit() for v in statics.values()):
        raise ValueError(f"{template} is no token select whose pieces are laid out here: TokenSelect_v4{{V}} or "
                         "GreedySelect_v1{V,N}")
    if int(statics["V"]) != V or V % int(statics["N"]) or descriptor_id(GS.select(V, V // int(statics["N"]))) != template:
        raise ValueError(f"{template}: rows of {V} logits are not its whole chunks")
    us = GS.plan(V, V // int(statics["N"]))
    K, N = len(us), V // len(us)
    xs = x.reshape(n, K, N)
    f = x.view(torch.bfloat16).float()
    f = torch.where(torch.isnan(f), float("-inf"), f).view(n, K, N)
    at = f.argmax(dim=2)
    best = xs.gather(2, at[:, :, None])[:, :, 0]
    later = ~(f > float("-inf")).any(dim=2)
    later[:, 0] = False
    nan0 = torch.isnan(x[:, 0].view(torch.bfloat16))
    at = torch.where(later, 0, at)
    at[:, 0] = torch.where(nan0, 0, at[:, 0])
    best = torch.where(later, NEG_INF_BF16 - (1 << 16), best)
    best[:, 0] = torch.where(nan0, x[:, 0], best[:, 0])
    tok = at + torch.arange(K, device=x.device)[None, :] * N
    keys = best.view(torch.bfloat16).float()
    b, t, held = keys[:, 0], tok[:, 0], torch.isnan(keys[:, 0])
    for c in range(1, K):
        take = ~held & (keys[:, c] > b)
        b, t = torch.where(take, keys[:, c], b), torch.where(take, tok[:, c], t)
    if not torch.equal(t, ids):
        raise ValueError(f"the units select {t.tolist()[:8]}..., the sampler {ids.tolist()[:8]}...")

    def words(v: Any) -> Any:
        return v.to(torch.int32).view(torch.int16).view(n, 2)

    outs = {u.key: torch.cat([best[:, c:c + 1], words(tok[:, c].contiguous())], dim=1) for c, u in enumerate(us[:-1])}
    got = {u.defn.id: {"logits": x[:, u.lanes[0]:u.lanes[1]], "o0": outs[u.key]} for u in us[:-1]}
    last = us[-1]
    got[last.defn.id] = {"logits": x[:, last.lanes[0]:last.lanes[1]], **{p: outs[k] for p, k, _f in last.wires},
                         "o0": words(t.contiguous())}
    return got


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
