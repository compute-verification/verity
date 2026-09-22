"""Lane tp2 (pod side, torch + safetensors on CPU): map every parameter shard a rank holds to the checkpoint slice it equals.

    /workspace/venv-cu129/bin/python -m verity_capture.bench.tp2_shards RUN_DIR --safetensors 'SNAPSHOT/model*.safetensors' [--out]

Input: `param_hashes.rank<r>.json` written by `tp2_worker.tp2_param_hashes` (sha256 + shape of every shard, and the sharding
attributes of the parallel layers).  For each shard the candidate slices of the checkpoint tensor(s) are hashed with the same
recipe (contiguous bf16 bytes) and the first match is reported:

    replicated      shard == checkpoint tensor                                          (norm weights)
    col(r)          shard == W[r*n/tp:(r+1)*n/tp, :]                                     (ColumnParallelLinear weight/bias, embeddings)
    row(r)          shard == W[:, r*n/tp:(r+1)*n/tp]                                     (RowParallelLinear weight)
    qkv(r)          shard == cat(Wq[q rows of rank r], Wk[kv rows of rank r], Wv[...])   (QKVParallelLinear: heads split, then fused)
    gate_up(r)      shard == cat(Wgate[r half], Wup[r half])                             (MergedColumnParallelLinear)

This is exactly the map the root policy's checkpoint-identity check needs under TP: a Root's bytes are a *slice* (or a
concatenation of slices) of the checkpoint tensor(s) the profile names, and the slice is a function of (rank, tp_size, layer
kind, head geometry) only.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


def sha(t: torch.Tensor) -> str:
    t = t.contiguous().reshape(-1)
    return hashlib.sha256(t.view(torch.uint8).numpy().tobytes() if t.numel() else b"").hexdigest()


def load_ckpt(pattern: str) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for f in sorted(glob.glob(pattern)):
        with safe_open(f, framework="pt", device="cpu") as sf:
            for k in sf.keys():  # noqa: SIM118 -- safe_open handles are not dicts
                out[k] = sf.get_tensor(k)
    return out


def hf_name(vllm_name: str) -> str:
    return vllm_name


def candidates(name: str, shape: list[int], rank: int, tp: int, ckpt: dict[str, torch.Tensor], mods: dict, cfg: dict) -> list[tuple[str, torch.Tensor]]:
    """Candidate (rule, tensor) pairs for this shard."""
    out: list[tuple[str, torch.Tensor]] = []
    base, leaf = name.rsplit(".", 1) if "." in name else ("", name)
    if name in ckpt:
        w = ckpt[name]
        out.append(("replicated", w))
        if w.dim() >= 1 and w.shape[0] % tp == 0:
            n = w.shape[0] // tp
            out.append((f"col(r={rank}): rows [{rank * n}:{(rank + 1) * n}] of {list(w.shape)}", w[rank * n:(rank + 1) * n]))
        if w.dim() == 2 and w.shape[1] % tp == 0:
            n = w.shape[1] // tp
            out.append((f"row(r={rank}): cols [{rank * n}:{(rank + 1) * n}] of {list(w.shape)}", w[:, rank * n:(rank + 1) * n]))
    if leaf in ("weight", "bias") and base.endswith("qkv_proj"):
        pre = base[: -len("qkv_proj")]
        q, k, v = (ckpt.get(f"{pre}q_proj.{leaf}"), ckpt.get(f"{pre}k_proj.{leaf}"), ckpt.get(f"{pre}v_proj.{leaf}"))
        m = mods.get(base, {})
        hs, nh, nkv = m.get("head_size"), m.get("total_num_heads"), m.get("total_num_kv_heads")
        if q is not None and k is not None and v is not None and hs and nh and nkv:
            qn, kn = nh // tp * hs, nkv // tp * hs
            kv_rank = rank if nkv >= tp else rank // (tp // nkv)
            parts = [q[rank * qn:(rank + 1) * qn], k[kv_rank * kn:(kv_rank + 1) * kn], v[kv_rank * kn:(kv_rank + 1) * kn]]
            desc = (f"qkv(r={rank}): cat(q rows [{rank * qn}:{(rank + 1) * qn}], k rows [{kv_rank * kn}:{(kv_rank + 1) * kn}], "
                    f"v rows [same]) heads {nh}/{nkv} x {hs}")
            out.append((desc, torch.cat(parts, 0)))
    if leaf in ("weight", "bias") and base.endswith("gate_up_proj"):
        pre = base[: -len("gate_up_proj")]
        g, u = ckpt.get(f"{pre}gate_proj.{leaf}"), ckpt.get(f"{pre}up_proj.{leaf}")
        if g is not None and u is not None and g.shape[0] % tp == 0:
            n = g.shape[0] // tp
            out.append((f"gate_up(r={rank}): cat(gate rows [{rank * n}:{(rank + 1) * n}], up rows [same]) of {list(g.shape)}",
                        torch.cat([g[rank * n:(rank + 1) * n], u[rank * n:(rank + 1) * n]], 0)))
    if name == "lm_head.weight" and "lm_head.weight" not in ckpt and cfg.get("tie_word_embeddings", True):
        w = ckpt.get("model.embed_tokens.weight")
        if w is not None:
            n = w.shape[0] // tp
            out.append((f"tied lm_head = embed_tokens col(r={rank}) rows [{rank * n}:{(rank + 1) * n}]", w[rank * n:(rank + 1) * n]))
    # vocab padding: VocabParallelEmbedding pads the vocab to a multiple of 64 and shards the padded range
    if leaf == "weight" and base.endswith(("embed_tokens", "lm_head")):
        src = ckpt.get("model.embed_tokens.weight") if (name not in ckpt) else ckpt[name]
        m = mods.get(base, {})
        si = m.get("shard_indices") or {}
        if src is not None and si:
            s, e = si.get("org_vocab_start_index", 0), si.get("org_vocab_end_index", 0)
            per = m.get("num_embeddings_per_partition") or (e - s)
            sl = src[s:e]
            if sl.shape[0] < per:
                sl = torch.cat([sl, torch.zeros(per - sl.shape[0], *src.shape[1:], dtype=src.dtype)], 0)
            out.append((f"vocab(r={rank}): rows [{s}:{e}] (+{per - (e - s)} zero pad rows) of {list(src.shape)}", sl))
    return [(rule, t) for rule, t in out if list(t.shape) == list(shape)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--safetensors", required=True)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    run = Path(a.run)
    ckpt = load_ckpt(a.safetensors)
    cfg: dict = {}
    cfg_path = Path(glob.glob(a.safetensors)[0]).parent / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
    doc: dict[str, Any] = {"schema": "verity-capture/tp2-shards/v0", "run": str(run), "checkpoint_tensors": len(ckpt),
                           "checkpoint_numel": int(sum(t.numel() for t in ckpt.values())), "tie_word_embeddings": cfg.get("tie_word_embeddings"),
                           "ranks": {}}
    for p in sorted(run.glob("param_hashes.rank*.json")):
        ph = json.loads(p.read_text())
        rank, tp = int(ph["rank"]), int(ph["world_size"])
        mods = ph.get("modules", {})
        rows = []
        kinds: Counter = Counter()
        numel = 0
        cache: dict[str, str] = {}
        for name, t in ph["tensors"].items():
            numel += int(torch.Size(t["shape"]).numel())
            match = None
            tried = []
            for rule, cand in candidates(name, t["shape"], rank, tp, ckpt, mods, cfg):
                key = f"{name}|{rule}"
                h = cache.get(key) or sha(cand)
                cache[key] = h
                tried.append(rule.split(":")[0])
                if h == t["sha256"]:
                    match = rule
                    break
            kind = match.split(":")[0].split("(")[0] if match else ("buffer" if t["kind"] == "buffer" else "UNMATCHED")
            kinds[kind] += 1
            rows.append({"name": name, "kind": t["kind"], "shape": t["shape"], "dtype": t["dtype"], "match": match, "tried": tried})
        unmatched = [r for r in rows if r["match"] is None and r["kind"] == "param"]
        doc["ranks"][str(rank)] = {"tp_size": tp, "n_tensors": len(rows), "numel": numel, "numel_ratio_vs_checkpoint": round(numel / max(1, doc["checkpoint_numel"]), 4),
                                   "kinds": dict(kinds), "unmatched_params": [r["name"] for r in unmatched],
                                   "layer0_and_globals": [r for r in rows if ".layers." not in r["name"] or ".layers.0." in r["name"]],
                                   "rows": rows, "modules": {k: v for k, v in mods.items() if ".layers." not in k or ".layers.0." in k}}
        print(f"rank {rank}: {len(rows)} tensors, numel ratio {doc['ranks'][str(rank)]['numel_ratio_vs_checkpoint']}, kinds {dict(kinds)}, "
              f"unmatched params {[r['name'] for r in unmatched][:8]}")
    doc["embedding_body"] = embedding_body(run, ckpt, doc["ranks"])
    print(f"embedding body: {doc['embedding_body']['summary']}")
    out = Path(a.out) if a.out else run / "tp2_shards.json"
    out.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print(f"-> {out}")
    return 0


def embedding_body(run: Path, ckpt: dict[str, torch.Tensor], ranks: dict[str, Any]) -> dict[str, Any]:
    """The body of `_C.vocab_parallel_embedding` (the one Unsupported op whose operand is a Root, hence never snapshotted):
    out[i] = W_full[ids[i]] if start <= ids[i] < end else 0, checked against every rank's snapshotted result."""
    from ..events import StepStart
    from ..log import read_log

    W = ckpt["model.embed_tokens.weight"]
    inv_p = run / "tp2_inventory.json"
    res: dict[str, Any] = {"summary": {}, "checked": []}
    if not inv_p.exists():
        res["summary"] = {"skipped": "no tp2_inventory.json (run tp2_analyze inventory first)"}
        return res
    inv = json.loads(inv_p.read_text())
    ok = bad = unchecked = 0
    for r, rk in ranks.items():
        si = (rk["modules"].get("model.embed_tokens") or {}).get("shard_indices") or {}
        s, e = int(si.get("org_vocab_start_index", 0)), int(si.get("org_vocab_end_index", 0))
        ids_by_step = {}
        for ev in read_log(run / f"log.rank{r}.jsonl.gz"):
            if isinstance(ev, StepStart):
                ids_by_step[ev.step] = list(ev.input_token_ids)
        idx = {}
        snap = run / f"rank{r}" / "snapshots.json"
        if snap.exists():
            for en in json.loads(snap.read_text())["entries"]:
                idx[(en["event"], en["path"])] = en
        for row in inv["ranks"][r]["collectives"]:
            if not row["op"].startswith("_C.vocab_parallel_embedding"):
                continue
            # `_C.vocab_parallel_embedding(out, ids, weight, start, end, pad, added_start, added_end) -> ()` is out-variant: the
            # result is args[0] (snapshotted after the call); args[2] is the weight Root (never snapshotted)
            en = idx.get((row["id"], "outs")) or idx.get((row["id"], "args[0]"))
            ids = ids_by_step.get(row["step"])
            if en is None or not en.get("file") or ids is None:
                unchecked += 1
                continue
            got = torch.frombuffer(bytearray((run / f"rank{r}" / en["file"]).read_bytes()), dtype=torch.bfloat16).reshape(en["shape"])
            n = got.shape[0]
            ids_t = torch.tensor(ids[:n], dtype=torch.long)
            mask = (ids_t >= s) & (ids_t < e)
            pred = torch.zeros(n, W.shape[1], dtype=W.dtype)
            pred[mask] = W[ids_t[mask]]
            eq = torch.equal(pred, got)
            ok += int(eq)
            bad += int(not eq)
            res["checked"].append({"rank": int(r), "step": row["step"], "rows": n, "rows_owned_by_rank": int(mask.sum()), "vocab_range": [s, e], "equal": eq})
    res["summary"] = {"ok": ok, "mismatch": bad, "unchecked": unchecked,
                      "body": "out[i] = embed_tokens.weight[ids[i]] if start<=ids[i]<end else 0 (masked lookup of this rank's vocab shard); "
                              "the following vllm.all_reduce sums the two masked lookups (exactly one rank owns every id)"}
    return res


if __name__ == "__main__":
    sys.exit(main())
