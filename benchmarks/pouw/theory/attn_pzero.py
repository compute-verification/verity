"""The share of an int8 P·V kernel's u8 P that rounds to 0 (step_cover.py's ZERO_P), measured on a model's attention.

For every query row i >= 1 of every head and layer, over the live keys 1..i (the sink, key 0, is apart), in blocks of
`--blocks` keys aligned to key 0: p̂ = min(1, exp(s − m))·s_v[j]/max_block s_v, s the scaled logit and s_v V's per-key
int8 scale, rounded to u8 as rint(255·p̂) or under noise-keyed dither as ⌊255·p̂ + u⌋, u uniform on [0, 1) (zero with
probability max(0, 1 − 255·p̂)). The reference m is:

- `run`: the running row max when the key's block is processed (FlashAttention's, as benchmarks/ppl's `u8pv` reads it);
- `local`: the block's own max, the block entering O with the factor exp(m_b − m_run) (step_cover's block-local max);
- `local_s8_d1`: the max over every 8th key of the block plus a margin of 1 in log2 units, p saturated at 1
  (`strided_max`).

Counts are of live keys, of 4-key packs (the u8 fragment register: 4 consecutive keys) and of ssn_sh-tag's step
chunks (`CHUNK`: a P·V call's 104 keys as one 24-key step beside the tag lanes, then 28-key steps) holding a live key,
a pack or chunk zero when every live key in it is (a zero chunk's step word is its previous word plus the shared coin
byte, which a prover writes without the tensor instruction). They are binned by the row's position so that a
prefill's rows (positions under its prompt length) and a decode's (positions near its context) read their own bins.
One JSON out: per block, reference and bin, the keys, packs, chunks and both roundings' zero shares; and per layer,
the whole share at the first block.

Run on a CPU node from the repository root, with benchmarks/ppl's dataset sent as `inputs/` and the model's snapshot
under HF_HOME (`attn_pzero.sh`):

    python benchmarks/pouw/theory/attn_pzero.py --ctx 8192 --windows 4 --data inputs --out out
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

PPL = Path(__file__).resolve().parents[2] / "ppl"
sys.path.insert(0, str(PPL))
import methodology  # noqa: E402
import run as ppl  # noqa: E402
from model import Config, Weights, make_layer, rotary  # noqa: E402

EDGES = (1, 64, 256, 1024, 1792, 2304, 4096, 7168, 8193)
REFS = ("run", "local", "local_s8_d1")
FIELDS = ("keys", "zero_rint", "zero_nkd", "packs", "pack_zero_rint", "pack_zero_nkd", "chunks", "chunk_zero_rint",
          "chunk_zero_nkd")
#: (a P·V call's keys, its first step's keys, a later step's keys): step_cover's I8_BLOCK and ssn_sh-tag's steps
CHUNK = (104, 24, 28)


def new_tally(blocks) -> dict:
    return {b: {r: torch.zeros(len(EDGES) - 1, len(FIELDS), dtype=torch.float64) for r in REFS} for b in blocks}


def chunk_spans(s: int, chunk=CHUNK) -> list[tuple[int, int]]:
    call, first, step = chunk
    out = []
    for c0 in range(0, s, call):
        edges = [c0, *range(c0 + first, c0 + call, step), c0 + call]
        out += [(lo, min(hi, s)) for lo, hi in zip(edges, edges[1:]) if lo < s]
    return out


def tally_rows(t: dict, q, k, v, scaling: float, blocks, rows: int, stride: int = 8, margin: float = 1.0,
               row_stride: int = 1, chunk=CHUNK) -> None:
    """Adds one sequence's counts (q: H × s × d, k and v: G × s × d) to `t`, over query rows 1, 1 + row_stride, …"""
    H, s, _ = q.shape
    G = k.shape[0]
    rep = H // G
    assert s % 4 == 0 and all(b % 4 == 0 and b % stride == 0 and b % chunk[0] == 0 for b in blocks)
    sv = v.abs().amax(-1) / 127
    pos = torch.arange(s)
    rbin = torch.bucketize(pos, torch.tensor(EDGES), right=True) - 1
    spans = chunk_spans(s, chunk)
    cid = torch.cat([torch.full((hi - lo,), n) for n, (lo, hi) in enumerate(spans)])
    sel = torch.arange(1, s, row_stride)
    for g in range(G):
        for c0 in range(0, len(sel), rows):
            idx = sel[c0:c0 + rows]
            R = len(idx)
            live = pos[None, :] <= idx[:, None]
            live[:, 0] = False
            sc = ((q[g * rep:(g + 1) * rep, idx] @ k[g].T) * scaling).masked_fill(~live, -math.inf)
            any4 = live.view(R, s // 4, 4).any(-1)
            anyc = torch.zeros(R, len(spans), dtype=torch.float64).index_add_(1, cid, live.double()) > 0
            bins = rbin[idx]
            for b in blocks:
                nb = -(-s // b)
                pad = nb * b - s
                blk = pos // b
                top = F.pad(sv[g], (0, pad)).view(nb, b).clone()
                top[0, 0] = 0
                ratio = sv[g] / top.amax(-1)[blk]
                tiles = F.pad(sc, (0, pad), value=-math.inf).view(rep, R, nb, b)
                bmax = tiles.amax(-1)
                smax = tiles[..., ::stride].amax(-1)
                smax = torch.where(torch.isinf(smax) & ~torch.isinf(bmax), bmax, smax)
                refs = {"run": bmax.cummax(-1).values, "local": bmax, "local_s8_d1": smax + margin * math.log(2)}
                for r, m in refs.items():
                    y = torch.where(live, sc - m[..., blk], -math.inf)
                    x = 255 * torch.exp(y).clamp(max=1) * ratio
                    zr = (torch.round(x) == 0) & live
                    zn = torch.where(live, (1 - x).clamp(min=0), torch.zeros(()))
                    pr = ((torch.round(x) == 0) | ~live).view(rep, R, s // 4, 4).all(-1) & any4
                    pn = torch.where(live, (1 - x).clamp(min=0), torch.ones(())).view(rep, R, s // 4, 4).prod(-1) * any4
                    nz = (live & (torch.round(x) != 0)).double()
                    cr = (torch.zeros(rep, R, len(spans), dtype=torch.float64).index_add_(2, cid, nz) == 0) & anyc
                    lz = torch.where(live, torch.log((1 - x).clamp(min=0).double()), torch.zeros((), dtype=torch.float64))
                    cn = torch.zeros(rep, R, len(spans), dtype=torch.float64).index_add_(2, cid, lz).exp() * anyc
                    per_row = torch.stack([live.sum(-1).double().expand(rep, R), zr.sum(-1).double(), zn.sum(-1).double(),
                                           any4.sum(-1).double().expand(rep, R), pr.sum(-1).double(),
                                           pn.sum(-1).double(), anyc.sum(-1).double().expand(rep, R),
                                           cr.sum(-1).double(), cn.sum(-1)], -1).sum(0)
                    t[b][r].index_add_(0, bins, per_row)


def walk_rows(q, k, v, scaling: float, block: int, stride: int = 8, margin: float = 1.0, row_stride: int = 1,
              chunk=CHUNK) -> dict:
    """`tally_rows` for one block size as a kernel walks it: each row's blocks in order, a key at a time."""
    H, s, _ = q.shape
    rep = H // k.shape[0]
    sv = v.abs().amax(-1) / 127
    spans = chunk_spans(s, chunk)
    t = {r: torch.zeros(len(EDGES) - 1, len(FIELDS), dtype=torch.float64) for r in REFS}
    for h in range(H):
        g = h // rep
        for i in range(1, s, row_stride):
            sc = (q[h, i] @ k[g].T) * scaling
            bi = next(n for n in range(len(EDGES) - 1) if EDGES[n] <= i < EDGES[n + 1])
            mrun = -math.inf
            for a0 in range(0, s, block):
                lo, hi = max(a0, 1), min(a0 + block, i + 1)
                if lo >= hi:
                    continue
                bm = float(sc[lo:hi].max())
                mrun = max(mrun, bm)
                st = [float(sc[j]) for j in range(a0, hi, stride) if j >= 1]
                sm = (max(st) if st else bm) + margin * math.log(2)
                top = float(sv[g, max(a0, 1):a0 + block].max())
                for r, m in (("run", mrun), ("local", bm), ("local_s8_d1", sm)):
                    x = {j: 255 * min(1.0, math.exp(float(sc[j]) - m)) * float(sv[g, j]) / top for j in range(lo, hi)}
                    c = t[r][bi]
                    c[0] += len(x)
                    c[1] += sum(round(v_) == 0 for v_ in x.values())
                    c[2] += sum(max(0.0, 1 - v_) for v_ in x.values())
                    for p0 in range(a0, a0 + block, 4):
                        mem = [x[j] for j in range(p0, p0 + 4) if j in x]
                        if mem:
                            c[3] += 1
                            c[4] += all(round(v_) == 0 for v_ in mem)
                            c[5] += math.prod(max(0.0, 1 - v_) for v_ in mem)
                    for c0, c1 in spans:
                        mem = [x[j] for j in range(c0, c1) if j in x and a0 <= c0 < a0 + block]
                        if mem:
                            c[6] += 1
                            c[7] += all(round(v_) == 0 for v_ in mem)
                            c[8] += math.prod(max(0.0, 1 - v_) for v_ in mem)
    return t


def selfcheck() -> dict:
    """The vectorized counts against `walk_rows` on random heads, in float64, with rows crossing block and bin edges."""
    g = torch.Generator().manual_seed(0)
    H, G, s, d = 4, 2, 272, 8
    q = torch.randn(H, s, d, generator=g, dtype=torch.float64) * 2
    k = torch.randn(G, s, d, generator=g, dtype=torch.float64) * 2
    v = torch.randn(G, s, d, generator=g, dtype=torch.float64)
    worst, chunks = 0.0, 0.0
    for rs in (1, 3):
        t = new_tally((16, 32))
        tally_rows(t, q, k, v, 0.5, (16, 32), 50, row_stride=rs, chunk=(16, 4, 6))
        for b in (16, 32):
            w = walk_rows(q, k, v, 0.5, b, row_stride=rs, chunk=(16, 4, 6))
            for r in REFS:
                worst = max(worst, float((t[b][r] - w[r]).abs().max()))
                chunks += float(w[r][:, 7].sum())
    return {"max_abs_diff": worst, "zero_chunks": chunks, "ok": worst < 1e-6 and chunks > 0}


def shares(c: torch.Tensor) -> dict:
    k, p, n = float(c[0]), float(c[3]), float(c[6])
    return {"keys": k, "packs": p, "zero_rint": float(c[1]) / max(1, k), "zero_nkd": float(c[2]) / max(1, k),
            "pack_zero_rint": float(c[4]) / max(1, p), "pack_zero_nkd": float(c[5]) / max(1, p),
            "chunks": n, "chunk_zero_rint": float(c[7]) / max(1, n), "chunk_zero_nkd": float(c[8]) / max(1, n)}


def summary(t: dict) -> dict:
    out = {}
    for b, by_ref in t.items():
        out[str(b)] = {}
        for r, c in by_ref.items():
            bins = {f"{EDGES[i]}-{EDGES[i + 1] - 1}": shares(c[i]) for i in range(len(EDGES) - 1) if c[i, 0] > 0}
            out[str(b)][r] = {"all": shares(c.sum(0)), "bins": bins}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="llama-3.2-1b")
    ap.add_argument("--set", default="wikitext2")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--windows", type=int, default=4, help="the first N windows")
    ap.add_argument("--blocks", default="104,208")
    ap.add_argument("--rows", type=int, default=256, help="query rows a chunk")
    ap.add_argument("--row-stride", type=int, default=1, help="tally query rows 1, 1 + N, … (the forward runs every row)")
    ap.add_argument("--data")
    ap.add_argument("--out")
    ap.add_argument("--selfcheck", action="store_true", help="only check the counts against a per-key walk")
    a = ap.parse_args(argv)
    torch.set_grad_enabled(False)
    sc = selfcheck()
    print(json.dumps({"selfcheck": sc}), flush=True)
    if not sc["ok"] or a.selfcheck:
        return 0 if sc["ok"] else 1
    blocks = tuple(int(x) for x in a.blocks.split(","))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ids, _, wrec = ppl.build_windows(a.model, a.set, a.ctx, Path(a.data), False)
    ids = ids[:a.windows]
    snap = ppl.snapshot(a.model)
    cfg = Config.read(snap)
    W = Weights(snap, cfg.layers)
    emb = W.read(["model.embed_tokens.weight"])["model.embed_tokens.weight"].to(torch.bfloat16)
    N, T = ids.shape
    cos, sin = rotary(cfg, T, "cpu")
    h = emb[torch.from_numpy(ids)].clone()
    del emb
    whole, per_layer = new_tally(blocks), []
    B = methodology.REF_BATCH
    t0 = time.time()
    W.prefetch(0)
    for li in range(cfg.layers):
        layer = make_layer(cfg, li, W.layer(li), "cpu")
        if li + 1 < cfg.layers:
            W.prefetch(li + 1)
        lt = new_tally(blocks[:1])
        sdpa = layer.attention

        def attention(q, k, v, run, sdpa=sdpa, lt=lt, layer=layer):
            for i in range(q.shape[0]):
                one = new_tally(blocks)
                tally_rows(one, q[i].float(), k[i].float(), v[i].float(), layer.scaling, blocks, a.rows,
                           row_stride=a.row_stride)
                for b in blocks:
                    for r in REFS:
                        whole[b][r] += one[b][r]
                for r in REFS:
                    lt[blocks[0]][r] += one[blocks[0]][r]
            return sdpa(q, k, v, run)

        layer.attention = attention
        for j in range(0, N, B):
            h[j:j + B] = layer.forward(h[j:j + B], cos, sin)
        per_layer.append({r: shares(lt[blocks[0]][r].sum(0)) for r in REFS})
        print(json.dumps({"layer": li, "seconds": round(time.time() - t0, 1),
                          **{r: round(per_layer[-1][r]["zero_nkd"], 4) for r in REFS}}), flush=True)
    W.close()
    res = {"model": a.model, "set": a.set, "ctx": a.ctx, "windows": N, "row_stride": a.row_stride,
           "blocks": list(blocks), "edges": list(EDGES),
           "refs": list(REFS), "wrec": wrec, "all": summary(whole), "layers": per_layer,
           "seconds": round(time.time() - t0, 1)}
    (out / "attn_pzero.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({b: {r: v["all"] for r, v in by.items()} for b, by in res["all"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
