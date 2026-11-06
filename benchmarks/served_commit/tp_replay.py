"""Replay of a drawn request under a TP engine in eager mode (GLM-5.3 TP8): whether a request served in a batch replays bit for bit
(its tokens, and the values the model's modules return at its rows, with the DSA indexer's top-k indices and each attention layer's
core output) when it runs again in the same batch, beside fillers of the same shapes, alone, or in another batch; and what a
replay costs in GPU-seconds with its prompt's KV computed (cold) or read from the prefix cache (warm). Measurement only.

    bash benchmarks/served_commit/tp_replay.sh      (its header has the knobs)

The served batch (`SERVED`, `arms`) is eight requests, each prompt a tiling of the row's prompt at its own rotation so that no two
share a cache block: three drawn (``d-short``, 256 tokens greedy; ``d-long``, 3,072 tokens greedy, whose later rows read more keys
than the 2,048 the DSA top-k keeps; ``d-seed``, 512 tokens sampled with its own seed) and five companions arriving over the
first six steps. Every request decodes to its cap (``ignore_eos``), so an arm's schedule doesn't depend on its tokens. The arms:

- ``served``, the reference, and ``served-2``, the same batch again (a window replay: the same requests and arrivals);
- ``shape``: the drawn requests in place, each companion a filler of its length, cap, arrival and sampling (random tokens);
- ``reordered`` and ``staggered``: the served requests in other compositions, admitted in reverse at step 0, or request i at
  step 2i;
- ``alone-<drawn>``: each drawn request by itself;
- ``other``: the drawn requests in another order, arriving at step 1, among `OTHER`'s twelve companions (four over 2,048 tokens);
- ``warm``: a producer with ``d-long``'s prompt and one output token, then ``d-long`` once the engine is idle, its prompt's blocks
  in the prefix cache.

Pass 1 runs every arm with `Interior` on every rank: a forward hook on every module, a wrapper on each DSA attention layer's
``_sparse_indexer_and_attn`` (the shared top-k buffer as the layer's indexer leaves it, and the attention's core output), and the
runner's ``prepare_inputs`` and ``sample`` for each step's rows. Each value is fingerprinted on the GPU per token row
(`fingerprint`: two keyed sums of the row's 32-bit words mod 2^64), a module group's calls (`group_of`) folded into one 128-bit
value per (request, position, group), and each arm compared on its rank with ``served`` for the drawn requests (`compare`).
Rank 0 also keeps each step's composition (`composition`) and the top-1 - top-2 logit gap of every sampled row, so each arm
says whether its schedule was served's and, per request both served, how many output tokens flipped, where the first did and
at what gap (`flips`); `distinct` counts each request's output sequences over every run. Values
inside fused kernels that no module returns (router ids, expert intermediates, quantized activations, all-reduce partials) are
compared only through the module outputs that read them. Pass 2 runs every arm again ``--reps`` times without hooks, its tokens
checked against pass 1's, and times it: GPU-seconds are wall time times the world size; ``warm``'s replay is ``d-long``'s span
after the producer finished, with its cached-token count, against ``alone-d-long``'s.

`summary.json` has per arm and drawn request: tokens equal to ``served``, the compared, equal and differing entries summed over
ranks, those only one run has, the first differing (position, group), the group kinds that differ, and whether the request
replayed bit for bit (its tokens, and every entry both runs have equal, with none only the replay has: ``warm`` skips the
positions its prefix cache held); and the timings. Exit 3 when the instrument failed: a rank's reference compared nothing for a drawn request, or a hook stayed installed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

DRAWN = ("d-short", "d-long", "d-seed")
REFERENCE = "served"
PRODUCER = "warm-producer"
#: (request id, prompt length, cap, arrive step, sampling): greedy unless the request names its own
SERVED = (("d-short", 256, 32, 0, {}), ("d-long", 3072, 32, 0, {}),
          ("d-seed", 512, 32, 0, {"temperature": 0.8, "top_p": 0.95, "seed": 1234}),
          ("c0", 128, 16, 0, {}), ("c1", 384, 24, 0, {"temperature": 1.0, "top_p": 0.9, "seed": 7}), ("c2", 640, 40, 2, {}),
          ("c3", 1024, 48, 4, {"temperature": 0.7, "top_k": 20, "seed": 11}), ("c4", 1536, 20, 6, {}))
OTHER = (("o0", 2304, 12, 0, {}), ("o1", 160, 40, 0, {"temperature": 0.9, "seed": 21}), ("o2", 2560, 16, 0, {}),
         ("o3", 704, 24, 1, {}), ("o4", 96, 48, 1, {"temperature": 0.6, "top_p": 0.8, "seed": 22}), ("o5", 2176, 20, 2, {}),
         ("o6", 1152, 28, 2, {}), ("o7", 448, 36, 3, {"temperature": 1.0, "top_k": 50, "seed": 23}), ("o8", 3328, 8, 3, {}),
         ("o9", 320, 44, 5, {}), ("o10", 832, 32, 7, {}), ("o11", 1792, 18, 9, {}))
#: the drawn requests' order and arrival in ``other``
OTHER_DRAWN = (("d-seed", 1), ("d-long", 1), ("d-short", 1))
MAX_SEQS = 16
FILLER_SEED = 0x7E9_1A7
FP_SEED = 0x5EED_F1A6
#: 32-bit words of the widest row a key table covers (GLM-5.3's f32 logits row is 154,880)
KEY_WORDS = 1 << 18
#: words of one chunk's product in `fingerprint`
CHUNK_WORDS = 1 << 22
#: bits of each of the three fields of a record key (request, position, group)
FIELD = 21
MASK = (1 << FIELD) - 1


def tiled(base: list[int], length: int, rotate: int) -> list[int]:
    """``length`` tokens of ``base`` rotated by ``rotate``, repeated."""
    r = rotate % len(base)
    tile = base[r:] + base[:r]
    return (tile * (length // len(tile) + 1))[:length]


def request(row: tuple[str, int, int, int, dict[str, Any]], base: list[int], rotate: int) -> dict[str, Any]:
    """A `SERVED` or `OTHER` row as a request, its prompt ``base`` tiled at ``rotate``."""
    rid, length, cap, arrive, sampling = row
    return {"request_id": rid, "prompt_token_ids": tiled(base, length, rotate), "max_tokens": cap, "arrive_step": arrive, **sampling}


def filler(r: dict[str, Any], i: int, vocab: int, rng: np.random.Generator) -> dict[str, Any]:
    """A request of ``r``'s length, cap, arrival and sampling fields, of random tokens, as `vllm_adapter.throwaway_requests` makes one."""
    keep = {k: v for k, v in r.items() if k not in ("request_id", "prompt_token_ids")}
    return {**keep, "request_id": f"filler-{i}", "prompt_token_ids": rng.integers(3, vocab - 1, size=len(r["prompt_token_ids"])).tolist()}


def arms(base: list[int], vocab: int) -> dict[str, list[dict[str, Any]]]:
    """Every arm's requests, in admission order (the module docstring has them). Each request's rotation of ``base`` is its own."""
    rot = iter(range(7, 7 * 64, 7))
    served = [request(row, base, next(rot)) for row in SERVED]
    drawn = {r["request_id"]: r for r in served if r["request_id"] in DRAWN}
    rng = np.random.default_rng(FILLER_SEED)
    out = {REFERENCE: served, "served-2": served,
           "shape": [r if r["request_id"] in DRAWN else filler(r, i, vocab, rng) for i, r in enumerate(served)]}
    out["reordered"] = [{**r, "arrive_step": 0} for r in reversed(served)]
    out["staggered"] = [{**r, "arrive_step": 2 * i} for i, r in enumerate(served)]
    out.update({f"alone-{d}": [{**drawn[d], "arrive_step": 0}] for d in DRAWN})
    others = [request(row, base, next(rot)) for row in OTHER]
    out["other"] = others[:3] + [{**drawn[d], "arrive_step": s} for d, s in OTHER_DRAWN] + others[3:]
    long = drawn["d-long"]
    out["warm"] = [{**long, "request_id": PRODUCER, "max_tokens": 1, "arrive_step": 0}, {**long, "arrive_step": 1 << 30}]
    return out


def group_of(name: str) -> str:
    """The group a module's values fold into: ``L<i>`` and the next name part inside decoder layer ``i`` (its attention, MLP and
    norms; ``topk`` and ``attn`` for what `Interior` takes from the attention), ``L<i>`` for the layer's own output, the name
    itself outside the layers, ``root`` for the model's output."""
    p = name.split(".")
    if len(p) >= 3 and p[0] == "model" and p[1] == "layers":
        return f"L{p[2]}" + (f".{p[3]}" if len(p) > 3 else "")
    return name or "root"


def kind_of(group: str) -> str:
    """A group without its layer: ``L12.mlp`` -> ``mlp``, ``L12`` -> ``layer``."""
    if group.startswith("L") and group[1:].split(".", 1)[0].isdigit():
        return group.split(".", 1)[1] if "." in group else "layer"
    return group


def key_table(device: Any = "cpu") -> Any:
    """[2, KEY_WORDS] int64, odd, from `FP_SEED`: the same on every rank and run."""
    import torch

    g = torch.Generator().manual_seed(FP_SEED)
    return (torch.randint(-(1 << 63), (1 << 63) - 1, (2, KEY_WORDS), dtype=torch.int64, generator=g) | 1).to(device)


def name_key(name: str) -> tuple[int, int]:
    """Two odd signed 64-bit keys of a call's name, from SHA-512."""
    d = hashlib.sha512(name.encode()).digest()
    return tuple(int.from_bytes(d[8 * i:8 * i + 8], "little", signed=True) | 1 for i in range(2))  # type: ignore[return-value]


def words(t: Any, rows: int) -> Any:
    """``t``'s first ``rows`` rows as little-endian 32-bit words, each row's bytes zero-padded to a whole word: [rows, W] int64 in
    [0, 2^32)."""
    import torch

    t = t[:rows].contiguous()
    if (t.storage_offset() * t.element_size()) % 4:
        t = t.clone()
    b = t.view(torch.uint8).reshape(rows, -1)
    if b.shape[1] % 4:
        b = torch.nn.functional.pad(b, (0, 4 - b.shape[1] % 4))
    return b.view(torch.int32).to(torch.int64) & 0xFFFFFFFF


def fingerprint(t: Any, rows: int, keys: Any) -> Any:
    """[rows, 2] int64: per row of ``t``, the sums over its words w_j of w_j * k_j mod 2^64 under each of ``keys``' two rows. Two
    rows with different bytes agree with probability about 2^-60 over the keys."""
    import torch

    w = words(t, rows)
    width = w.shape[1]
    if width > keys.shape[1]:
        raise ValueError(f"a row of {width} words is wider than the key table's {keys.shape[1]} (KEY_WORDS)")
    out = torch.empty((rows, 2), dtype=torch.int64, device=w.device)
    step = max(1, CHUNK_WORDS // max(width, 1))
    k = keys[None, :, :width]
    for lo in range(0, rows, step):
        out[lo:lo + step] = (w[lo:lo + step, None, :] * k).sum(-1)
    return out


def rows_of(ib: Any, code: Any) -> tuple[np.ndarray, np.ndarray]:
    """An input batch's token rows: each row's request (``code`` of its external id) and position."""
    from verity_vllm.engine.engine_driver import external_id

    n = int(ib.num_reqs)
    qsl = np.asarray(ib.query_start_loc_np[:n + 1], dtype=np.int64)
    nct = np.asarray(ib.num_computed_tokens_np[:n], dtype=np.int64)
    span = qsl[1:] - qsl[:-1]
    req = np.repeat(np.array([code(external_id(r)) for r in ib.req_ids[:n]], dtype=np.int64), span)
    pos = np.concatenate([nct[i] + np.arange(span[i], dtype=np.int64) for i in range(n)]) if n else np.zeros(0, np.int64)
    return req, pos


def composition(ib: Any) -> list[tuple[str, int, int]]:
    """An input batch's step composition: each request's external id, its first scheduled position and its scheduled tokens."""
    from verity_vllm.engine.engine_driver import external_id

    n = int(ib.num_reqs)
    qsl = [int(x) for x in ib.query_start_loc_np[:n + 1]]
    return [(external_id(r), int(ib.num_computed_tokens_np[i]), qsl[i + 1] - qsl[i]) for i, r in enumerate(ib.req_ids[:n])]


def logit_rows(ib: Any, code: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """An input batch's logits rows, one a request at its last scheduled position, and which of them are sampled from: not a
    partial prefill's, whose logits the engine computes and drops, at a position that depends on where the chunk ended. None when
    the batch has other logits rows (speculative decoding)."""
    from verity_vllm.engine.engine_driver import external_id

    n = int(ib.num_reqs)
    if int(ib.logits_indices.numel()) != n:
        return None
    qsl = np.asarray(ib.query_start_loc_np[:n + 1], dtype=np.int64)
    nct = np.asarray(ib.num_computed_tokens_np[:n], dtype=np.int64)
    pos = nct + (qsl[1:] - qsl[:-1]) - 1
    prompt = np.asarray(ib.prefill_len_np[:n], dtype=np.int64)
    return np.array([code(external_id(r)) for r in ib.req_ids[:n]], dtype=np.int64), pos, pos >= prompt - 1


def flatten(steps: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, int]:
    """A run's harvests (request [r], position [r], group [G], fingerprint [G, r, 2] each) as sorted keys (request, position, group
    in `FIELD` bits each) with their fingerprints, and how many keys came more than once (a recomputed row: its last value is kept)."""
    ks, fs = [], []
    for req, pos, groups, fp in steps:
        k = (req[None, :] << (2 * FIELD)) | (pos[None, :] << FIELD) | groups[:, None]
        ks.append(k.reshape(-1))
        fs.append(fp.reshape(-1, 2))
    keys = np.concatenate(ks) if ks else np.zeros(0, np.int64)
    fp = np.concatenate(fs) if fs else np.zeros((0, 2), np.int64)
    order = np.argsort(keys, kind="stable")
    keys, fp = keys[order], fp[order]
    last = np.ones(len(keys), dtype=bool)
    last[:-1] = keys[1:] != keys[:-1]
    return keys[last], fp[last], int((~last).sum())


def compare(ref: tuple[np.ndarray, np.ndarray], run: tuple[np.ndarray, np.ndarray], requests: dict[str, int],
            groups: list[str]) -> dict[str, dict[str, Any]]:
    """Per request: the (position, group) entries both runs have and how many are equal, those only one has, and the first that
    differs (by position, then group in first-call order), with the group kinds that differ."""
    rk, rf = ref
    uk, uf = run
    out: dict[str, dict[str, Any]] = {}
    for name, c in requests.items():
        a, b = (rk >> (2 * FIELD)) == c, (uk >> (2 * FIELD)) == c
        ak, af, bk, bf = rk[a], rf[a], uk[b], uf[b]
        both, ia, ib = np.intersect1d(ak, bk, assume_unique=True, return_indices=True)
        eq = (af[ia] == bf[ib]).all(axis=1)
        d = both[~eq]
        pos, grp = (d >> FIELD) & MASK, d & MASK
        first = None
        if len(d):
            j = np.lexsort((grp, pos))[0]
            first = {"pos": int(pos[j]), "group": groups[int(grp[j])]}
        out[name] = {"compared": int(len(both)), "equal": int(eq.sum()), "differ": int(len(d)), "only_ref": int(len(ak) - len(both)),
                     "only_run": int(len(bk) - len(both)), "positions": int(len(np.unique((both >> FIELD) & MASK))),
                     "positions_differ": int(len(np.unique(pos))), "first_differ": first,
                     "kinds_differ": dict(Counter(kind_of(groups[int(g)]) for g in grp).most_common(12))}
    return out


class Interior:
    """On one rank: every module's outputs, the DSA top-k and attention core, fingerprinted per token row and folded per (request,
    position, group); `end` compares a run with the reference run on this rank."""

    TAG = "tp-replay"

    def __init__(self, worker: Any) -> None:
        import torch
        from vllm.distributed import get_tensor_model_parallel_rank

        self.rank = int(get_tensor_model_parallel_rank())
        self.runner = worker.model_runner
        self.model = self.runner.get_model()
        self.device = next(self.model.parameters()).device
        self.keys = key_table(self.device)
        self.torch = torch
        self.handles: list[Any] = []
        self.hooks: Any = None
        self.req_code: dict[str, int] = {}
        self.group_code: dict[str, int] = {}
        self.groups: list[str] = []
        self._nk: dict[str, Any] = {}
        self.ref: tuple[np.ndarray, np.ndarray] | None = None
        self.begin()

    def begin(self) -> None:
        #: per row map: each row's request and position, the padded row count, and the rows to keep (None: all)
        self.maps: list[tuple[np.ndarray, np.ndarray, int, np.ndarray | None]] = []
        self.calls: list[tuple[int, int, Any, Any]] = []
        self.gap_calls: list[tuple[int, Any]] = []
        self.counts: Counter = Counter()
        self.steps: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        self.gaps: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self.schedule: list[list[tuple[str, int, int]]] = []
        self.skipped: Counter = Counter()
        self.active: int | None = None

    def code(self, rid: str) -> int:
        return self.req_code.setdefault(rid, len(self.req_code))

    def group(self, g: str) -> int:
        if g not in self.group_code:
            self.group_code[g] = len(self.groups)
            self.groups.append(g)
        return self.group_code[g]

    def name_key(self, name: str) -> Any:
        k = self._nk.get(name)
        if k is None:
            k = self._nk[name] = self.torch.tensor(name_key(name), dtype=self.torch.int64, device=self.device)
        return k

    def take(self, name: str, out: Any) -> None:
        m = self.active
        if m is None:
            return
        req, _pos, padded, _keep = self.maps[m]
        r = len(req)
        c = self.counts[name]
        self.counts[name] += 1
        g = self.group(group_of(name))
        for k, t in enumerate(out if isinstance(out, (tuple, list)) else (out,)):
            if not isinstance(t, self.torch.Tensor) or t.dim() == 0 or t.numel() == 0:
                continue
            if t.shape[0] not in (r, padded):
                self.skipped[f"{name}/{k}"] += 1
                continue
            self.calls.append((m, g, self.name_key(f"{name}#{c}/{k}"), fingerprint(t, r, self.keys)))
            if padded == -1 and name.endswith("logits_processor") and t.dim() == 2 and t.is_floating_point() and t.shape[1] > 1:
                top = t[:r].detach().float().topk(2, dim=-1).values
                self.gap_calls.append((m, top[:, 0] - top[:, 1]))

    def _map(self, req: np.ndarray, pos: np.ndarray, padded: int, keep: np.ndarray | None = None) -> int:
        self.maps.append((req, pos, padded, keep))
        return len(self.maps) - 1

    def flush(self) -> None:
        """Fold the calls since the last flush (a step's) into one fingerprint per (row, group) and move them to the host. A
        module's calls are numbered within a step, so a row's fingerprints don't depend on the steps before it."""
        self.counts.clear()
        if not self.calls:
            self.maps.clear()
            return
        if self.device.type == "cuda":
            self.torch.cuda.synchronize()
        acc: dict[tuple[int, int], Any] = {}
        for m, g, nk, fp in self.calls:
            v = fp * nk
            acc[(m, g)] = v if (m, g) not in acc else acc[(m, g)] + v
        by: dict[int, list[tuple[int, Any]]] = {}
        for (m, g), v in acc.items():
            by.setdefault(m, []).append((g, v))
        for m, items in by.items():
            req, pos, _padded, keep = self.maps[m]
            fp = self.torch.stack([v for _g, v in items]).cpu().numpy()
            if keep is not None:
                req, pos, fp = req[keep], pos[keep], fp[:, keep]
            self.steps.append((req, pos, np.array([g for g, _v in items], dtype=np.int64), fp))
        for m, gap in self.gap_calls:
            req, pos, _padded, keep = self.maps[m]
            v = gap.cpu().numpy()
            self.gaps.append((req[keep], pos[keep], v[keep]) if keep is not None else (req, pos, v))
        self.calls.clear()
        self.gap_calls.clear()
        self.maps.clear()

    def install(self) -> dict[str, Any]:
        from verity_vllm.engine.hooks import Hooks

        me = self
        self.hooks = Hooks(tag=self.TAG)
        self.base_hooks = sum(len(m._forward_hooks) for _n, m in self.model.named_modules())

        def hook_for(name: str) -> Any:
            return lambda _mod, _args, out: me.take(name, out)

        attn = 0
        for name, mod in self.model.named_modules():
            self.handles.append(mod.register_forward_hook(hook_for(name)))
            if hasattr(mod, "_sparse_indexer_and_attn") and hasattr(mod, "topk_indices_buffer"):
                self.hooks.wrap(mod, "_sparse_indexer_and_attn", self._attention(name.rsplit(".", 1)[0], mod))
                attn += 1

        def execute_model(orig: Any) -> Any:
            def run(*a: Any, **kw: Any) -> Any:
                me.flush()
                try:
                    return orig(*a, **kw)
                finally:
                    me.active = None
            return run

        def prepare_inputs(orig: Any) -> Any:
            def run(*a: Any, **kw: Any) -> Any:
                ib = orig(*a, **kw)
                me.active = me._map(*rows_of(ib, me.code), int(ib.num_tokens_after_padding))
                me.schedule.append(composition(ib))
                return ib
            return run

        def sample(orig: Any) -> Any:
            def run(*a: Any, **kw: Any) -> Any:
                ib = kw["input_batch"] if "input_batch" in kw else a[1]
                rows = logit_rows(ib, me.code)
                me.active = None if rows is None else me._map(rows[0], rows[1], -1, rows[2])
                if rows is None:
                    me.skipped["logits (not one row a request)"] += 1
                try:
                    return orig(*a, **kw)
                finally:
                    me.active = None
            return run

        for name, make in (("execute_model", execute_model), ("prepare_inputs", prepare_inputs), ("sample", sample)):
            self.hooks.wrap(self.runner, name, make)
        return {"rank": self.rank, "modules": len(self.handles), "dsa_attention": attn}

    def _attention(self, layer: str, mod: Any) -> Any:
        me = self
        has_topk = getattr(mod, "indexer", None) is not None and not getattr(mod, "skip_topk", False)

        def make(orig: Any) -> Any:
            def run(*a: Any, **kw: Any) -> Any:
                r = orig(*a, **kw)
                if me.active is not None:
                    n = len(me.maps[me.active][0])
                    me.take(f"{layer}.attn", kw["output"] if "output" in kw else a[-1])
                    if has_topk:
                        me.take(f"{layer}.topk", mod.topk_indices_buffer[:n])
                return r
            return run
        return make

    def end(self, arm: str, drawn: list[str], reference: str) -> dict[str, Any]:
        self.flush()
        keys, fp, dups = flatten(self.steps)
        codes = {d: self.req_code[d] for d in drawn if d in self.req_code}
        if arm == reference:
            self.ref = (keys, fp)
        res = compare(self.ref, (keys, fp), codes, self.groups) if self.ref is not None else {}
        doc = {"rank": self.rank, "arm": arm, "entries": int(len(keys)), "duplicates": dups, "groups": len(self.groups),
               "skipped": dict(self.skipped.most_common(8)), "per_request": res}
        if self.rank == 0:
            names = {c: r for r, c in self.req_code.items()}
            gaps: dict[str, dict[int, float]] = {}
            for req, pos, v in self.gaps:
                for c, p, x in zip(req.tolist(), pos.tolist(), v.tolist()):
                    gaps.setdefault(names[c], {})[int(p)] = float(x)
            doc.update(gaps=gaps, schedule=self.schedule)
        self.begin()
        return doc

    def uninstall(self) -> dict[str, Any]:
        from verity_vllm.engine import hooks as H

        for h in self.handles:
            h.remove()
        self.handles.clear()
        self.hooks.uninstall()
        left = sum(len(m._forward_hooks) for _n, m in self.model.named_modules()) - self.base_hooks
        return {"rank": self.rank, "hooks_left": left, "patches_left": len(H.live(self.TAG))}


def install(worker: Any) -> dict[str, Any]:
    """On a rank (`LLM.collective_rpc` with a callable)."""
    worker._tp_replay = Interior(worker)
    return worker._tp_replay.install()


def begin(worker: Any) -> None:
    worker._tp_replay.begin()


def end(worker: Any, *, arm: str, drawn: list[str], reference: str) -> dict[str, Any]:
    return worker._tp_replay.end(arm, drawn, reference)


def uninstall(worker: Any) -> dict[str, Any]:
    return worker._tp_replay.uninstall()


class Steps:
    """`run_workload`'s ``step``: the step at which each request finished, and each request's cached prompt tokens."""

    def __init__(self) -> None:
        self.n = 0
        self.done: dict[str, int] = {}
        self.cached: dict[str, int] = {}

    def __call__(self, step: Any) -> Any:
        outs = step()
        for o in outs:
            if o.finished:
                self.done.setdefault(o.request_id, self.n)
            if getattr(o, "num_cached_tokens", None) is not None:
                self.cached[o.request_id] = int(o.num_cached_tokens)
        self.n += 1
        return outs


def span(walls: list[float], done: dict[str, int], rid: str, after: str | None = None) -> float | None:
    """Seconds from the start of the run, or from the end of the step at which ``after`` finished, to the end of the step at which
    ``rid`` finished."""
    if rid not in done or (after is not None and after not in done):
        return None
    t = np.cumsum(walls)
    return float(t[done[rid]] - (t[done[after]] if after is not None else 0.0))


def flips(ref: dict[str, list[int]], run: dict[str, list[int]], ref_gaps: dict[str, dict[int, float]],
          run_gaps: dict[str, dict[int, float]], prompts: dict[str, int]) -> dict[str, dict[str, Any]]:
    """Per request both runs served: whether its tokens are equal, how many output positions differ, the first that does, and the
    top-1 - top-2 logit gap at the position that sampled it in each run (and the reference's smallest gap over the request)."""
    out: dict[str, dict[str, Any]] = {}
    for rid in sorted(set(ref) & set(run)):
        a, b = ref[rid], run[rid]
        diff = [i for i, (x, y) in enumerate(zip(a, b)) if x != y] + list(range(min(len(a), len(b)), max(len(a), len(b))))
        at = prompts[rid] - 1 + diff[0] if diff else None
        rg, ug = ref_gaps.get(rid, {}), run_gaps.get(rid, {})
        out[rid] = {"tokens_equal": not diff, "flipped": len(diff), "first_flip": diff[0] if diff else None,
                    "gap_ref": rg.get(at), "gap_run": ug.get(at), "min_gap_ref": min(rg.values()) if rg else None}
    return out


def distinct(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Per request id served in more than one run (fillers and the producer aside), its distinct output sequences over every run."""
    seqs: dict[str, set[tuple[int, ...]]] = {}
    runs: Counter = Counter()
    for r in rows:
        for rid, toks in (r.get("tokens") or {}).items():
            if not rid.startswith("filler-") and rid != PRODUCER:
                seqs.setdefault(rid, set()).add(tuple(toks))
                runs[rid] += 1
    return {rid: len(s) for rid, s in sorted(seqs.items()) if runs[rid] > 1}


def summarize(rows: list[dict[str, Any]], world: int) -> dict[str, Any]:
    """Per arm and drawn request, pass 1's comparison summed over ranks; per arm, every request's flips against ``served`` and
    whether its schedule was served's; pass 2's median wall and GPU-seconds; the drawn requests' replay GPU-seconds alone, cold,
    and ``d-long``'s warm; and each request's distinct output sequences over both passes."""
    doc: dict[str, Any] = {"world": world, "arms": {}, "distinct_sequences": distinct(rows)}
    for r in (x for x in rows if x["pass"] == 1):
        per: dict[str, Any] = {}
        for d in DRAWN:
            rk = [x["per_request"][d] for x in r["ranks"] if d in x["per_request"]]
            if not rk:
                continue
            s = {k: sum(x[k] for x in rk) for k in ("compared", "equal", "differ", "only_ref", "only_run", "positions_differ")}
            kinds: Counter = Counter()
            for x in rk:
                kinds.update(x["kinds_differ"])
            firsts = [x["first_differ"] for x in rk if x["first_differ"]]
            per[d] = {"tokens_equal": r["tokens_equal_served"].get(d), "ranks": len(rk), **s,
                      "ranks_differ": sum(1 for x in rk if x["differ"] or x["only_ref"] or x["only_run"]),
                      "first_differ": min(firsts, key=lambda f: f["pos"]) if firsts else None, "kinds_differ": dict(kinds.most_common(12)),
                      "bit_exact": bool(r["tokens_equal_served"].get(d)) and s["compared"] > 0 and not (s["differ"] or s["only_run"])}
        timed = [x for x in rows if x["pass"] == 2 and x["arm"] == r["arm"]]
        walls = [x["wall_s"] for x in timed]
        doc["arms"][r["arm"]] = {"drawn": per, "flips": r.get("flips"), "schedule_equal_served": r.get("schedule_equal_served"),
                                 "wall_s": statistics.median(walls) if walls else None,
                                 "gpu_s": world * statistics.median(walls) if walls else None,
                                 "tokens_equal_pass1": all(x["tokens_equal_pass1"] for x in timed) if timed else None}
    replay: dict[str, Any] = {}
    for d in DRAWN:
        s = [x["spans"][d] for x in rows if x["pass"] == 2 and x["arm"] == f"alone-{d}" and x["spans"].get(d) is not None]
        replay[f"{d} cold"] = {"gpu_s": world * statistics.median(s), "reps": len(s)} if s else None
    w = [x for x in rows if x["pass"] == 2 and x["arm"] == "warm" and x["spans"].get("d-long") is not None]
    replay["d-long warm"] = {"gpu_s": world * statistics.median(x["spans"]["d-long"] for x in w), "reps": len(w),
                             "cached_tokens": sorted({x["cached_tokens"].get("d-long") for x in w}, key=str)} if w else None
    doc["replay"] = replay
    return doc


def main() -> None:
    ap = argparse.ArgumentParser(description="replay of drawn requests: same batch, fillers, alone, another batch; cold and warm")
    ap.add_argument("--workload", required=True, help="the row (its first request's prompt is every request's tile)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--case", default=None)
    ap.add_argument("--checkpoints", default="manifests/checkpoints.json")
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--reps", type=int, default=2, help="pass 2's runs of every arm, interleaved")
    ap.add_argument("--engine-arg", action="append", default=[])
    ap.add_argument("--rpc-timeout", type=float, default=3600.0)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from served_commit.tp_decode import open_engine

    E = open_engine(a, out, max_num_seqs=MAX_SEQS, prefix_caching=True)
    from verity_vllm.pipeline.commit import run_workload

    llm, eng, wl, world = E.llm, E.eng, E.wl, E.world
    sampling = wl["sampling"]
    vocab = int(llm.llm_engine.vllm_config.model_config.hf_config.vocab_size)
    plan = arms([int(t) for t in wl["requests"][0]["prompt_token_ids"]], vocab)
    longest = max(len(r["prompt_token_ids"]) + int(r["max_tokens"]) for v in plan.values() for r in v)
    if longest > int(E.kw.get("max_model_len") or 0):
        raise SystemExit(f"max_model_len {E.kw.get('max_model_len')} is below the longest request's {longest} tokens")
    layout = {k: [[r["request_id"], len(r["prompt_token_ids"]), r["max_tokens"], r["arrive_step"]] for r in v] for k, v in plan.items()}
    (out / "versions.json").write_text(json.dumps({**E.versions, "arms": layout}, indent=1, default=str) + "\n")
    rng = np.random.default_rng(FILLER_SEED + 1)
    t0 = time.perf_counter()
    first = run_workload(eng, [filler(r, i, vocab, rng) for i, r in enumerate(plan[REFERENCE])], sampling)
    print(f"[tp-replay] engine built in {E.build_s:.0f}s; first run (fillers of the served batch) {time.perf_counter() - t0:.1f}s", flush=True)
    rows: list[dict[str, Any]] = []
    made = llm.collective_rpc(install, timeout=a.rpc_timeout)
    print(f"[tp-replay] interior on {len(made)} ranks: {made[0]}", flush=True)
    tok1: dict[str, dict[str, list[int]]] = {}
    gaps1: dict[str, dict[str, dict[int, float]]] = {}
    sched1: dict[str, list[Any]] = {}
    broken: list[str] = []
    with (out / "runs.jsonl").open("w") as f:
        def put(row: dict[str, Any]) -> None:
            rows.append(row)
            f.write(json.dumps(row) + "\n")
            f.flush()

        put({"pass": 0, "arm": "first-fillers", "wall_s": first["wall_s"], "steps": first["iterations"]})
        for arm, reqs in plan.items():
            llm.collective_rpc(begin, timeout=a.rpc_timeout)
            r = run_workload(eng, reqs, sampling)
            ranks = llm.collective_rpc(end, timeout=a.rpc_timeout, kwargs={"arm": arm, "drawn": list(DRAWN), "reference": REFERENCE})
            r0 = min(ranks, key=lambda x: x["rank"])
            tok1[arm], gaps1[arm], sched1[arm] = r["tokens"], r0.pop("gaps"), r0.pop("schedule")
            eq = {d: r["tokens"][d] == tok1[REFERENCE][d] for d in DRAWN if d in r["tokens"]}
            fl = flips(tok1[REFERENCE], r["tokens"], gaps1[REFERENCE], gaps1[arm], {q["request_id"]: len(q["prompt_token_ids"]) for q in reqs})
            put({"pass": 1, "arm": arm, "wall_s": r["wall_s"], "steps": r["iterations"], "tokens_equal_served": eq, "flips": fl,
                 "schedule_equal_served": sched1[arm] == sched1[REFERENCE], "schedule": sched1[arm], "gaps": gaps1[arm],
                 "tokens": r["tokens"], "ranks": ranks})
            if arm == REFERENCE:
                broken += [f"rank {x['rank']}: {d} compared nothing" for x in ranks for d in DRAWN
                           if not x["per_request"].get(d, {}).get("compared")]
            print(f"[tp-replay] pass 1 {arm}: tokens {eq}; schedule equal served {sched1[arm] == sched1[REFERENCE]}; flipped "
                  + str({k: v["flipped"] for k, v in fl.items() if v["flipped"]}) + "; rank 0 "
                  + "; ".join(f"{d} {v['equal']}/{v['compared']} equal, first {v['first_differ']}" for d, v in r0["per_request"].items()),
                  flush=True)
        left = llm.collective_rpc(uninstall, timeout=a.rpc_timeout)
        broken += [f"rank {x['rank']}: {x['hooks_left']} hooks, {x['patches_left']} patches left" for x in left
                   if x["hooks_left"] or x["patches_left"]]
        for rep in range(a.reps):
            for arm, reqs in plan.items():
                st = Steps()
                r = run_workload(eng, reqs, sampling, step=st)
                spans = {d: span(r["step_wall_s"], st.done, d, PRODUCER if arm == "warm" else None) for d in DRAWN}
                put({"pass": 2, "rep": rep, "arm": arm, "wall_s": r["wall_s"], "gpu_s": world * r["wall_s"], "steps": r["iterations"],
                     "tokens_equal_pass1": r["tokens"] == tok1[arm], "spans": spans, "cached_tokens": st.cached,
                     "step_wall_s": r["step_wall_s"], "tokens": r["tokens"]})
                print(f"[tp-replay] pass 2 rep {rep} {arm}: {r['wall_s']:.2f}s, tokens equal pass 1 {r['tokens'] == tok1[arm]}, "
                      f"spans {spans}, cached {st.cached}", flush=True)
    summary = {**summarize(rows, world), "first_fillers_s": first["wall_s"], "build_s": E.build_s, "instrument_failures": broken}
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    for arm, d in summary["arms"].items():
        print(f"[tp-replay] {arm}: " + "; ".join(f"{k} bit-exact {v['bit_exact']} (tokens {v['tokens_equal']}, {v['differ']} of "
                                                 f"{v['compared']} differ, first {v['first_differ']})" for k, v in d["drawn"].items())
              + f"; {d['gpu_s']} GPU-s", flush=True)
    print(f"[tp-replay] replay GPU-s: {summary['replay']}; distinct output sequences {summary['distinct_sequences']}", flush=True)
    if broken:
        print(f"[tp-replay] instrument failed: {broken[:4]}", flush=True)
        raise SystemExit(3)


if __name__ == "__main__":
    from served_commit.tp_replay import main as _main

    _main()
