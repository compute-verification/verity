"""A deferred Commit served on CPU, for the serving rows' tests and the one-stage audit's end-to-end run (`benchmarks/one_stage/a5.py`):
a Program with RoPE_v1 rows (D=64) executed honestly by the sampled replay's world (`tests.check.test_sampled_replay`), its members
committed by the production CPU committer and sealed into a replay bundle with its registered weights, beside the Commit's record
(``runs.jsonl``, ``verdict.json``, the binding map) and the row dir of its Program of record.

    python -m tests.pipeline.served_rope OUT        (from integrations/vllm)

writes ``OUT/commit``, ``OUT/row`` and ``OUT/partition.json`` (every RoPE head, ``subset:1``) and prints them with the run root as
one JSON line."""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np

from verity_vllm.check.replay.index import ProgramIndex
from verity_vllm.commit import scheme
from verity_vllm.pipeline import cpu_replay as CPU
from verity_vllm.pipeline import replay_bundle as RB

D = 64
ROPE = f"RoPEHead_v1{{D={D}}}"
LAW = "subset:1"
CHUNK = 64
HEADS = 2 * 4 * (3 + 1)       # 2 requests x 4 tokens (2 prefill + 2 decode) x (3 query + 1 key heads)


def commit(root: Path) -> dict:
    """The deferred Commit's output dir, its row dir and the verifier's partition file under ``root``, and the world that served it."""
    import tests.check.test_sampled_replay as T
    from tests.check.opened import commit_members
    from verity_vllm.commit import store_dump as SD

    dh = T.DH
    T.DH = D
    try:
        rng = np.random.default_rng(17)
        inst, woff = T._program(rope=True)
        w = T._bf16(rng, max(hi for _lo, hi in woff.values()), 0.5)
        for f in ("model.model.layers.0.input_layernorm.weight", "model.model.norm.weight"):
            w[woff[f][0]:woff[f][1]] = 0x3F80
        lo, hi = woff["model.model.layers.0.self_attn.rotary_emb.cos_sin_cache"]
        w[lo:hi] = (np.asarray(rng.uniform(-1, 1, hi - lo), dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)
        prompts = {"r0": [1, 2], "r1": [5, 3]}
        world = T.Honest(inst, woff, w, prompts)
        bmap, _ = world.bmap_and_reader()
        com, rc = commit_members(dict(sorted(world.store.items())))
    finally:
        T.DH = dh

    out, row_dir = root / "commit", root / "row"
    (row_dir / "build_request").mkdir(parents=True)
    with gzip.open(row_dir / "build_request" / "instances.json.gz", "wt") as f:
        json.dump(inst, f)
    (row_dir / "build_request" / "result.json").write_text(json.dumps({"program_digest": inst["digest"]}))
    p = RB.partial(out, 1)
    SD.dump(com, rc, p / "store")
    prefix = ProgramIndex(inst).prefix
    (p / "weights" / "extra").mkdir(parents=True)
    names, roots, extra = [], [], {}
    for i, (field, (flo, fhi)) in enumerate(woff.items()):
        raw = np.ascontiguousarray(w[flo:fhi]).view(np.uint8)
        name = field[len(prefix):]
        names.append([name, "bfloat16", [fhi - flo], int(raw.size)])
        roots.append(CPU.tensor_root(raw, CHUNK))
        (p / "weights" / "extra" / f"{i}.bin").write_bytes(raw.tobytes())
        extra[name] = i
    geo = bytes(range(32))
    wroot = scheme.weights_root(geo, scheme.names_digest(names), roots).hex()
    (p / "weights" / "tree.json").write_text(json.dumps({"names": names, "tensor_roots": [r.hex() for r in roots], "root": wroot,
                                                         "chunk": CHUNK, "geo_digest": geo.hex(), "extra": extra}))
    (out / "binding_map_p1.json").write_text(json.dumps(bmap))
    sr, lk = RB.deferred_records(RB.path(out, 1))
    row = {"pair": 1, "arm": "instrumented", "commit": {"run_root": rc.run_root.hex(), "commit_pass": None, "sampled_replay": sr,
                                                        "boundary_linkage": lk, "weight_registration": {"root": wroot}}}
    reqs = [{"request_id": rid, "index": k, "prompt_token_ids": pr, "max_tokens": 3, "arrive_step": 0} for k, (rid, pr) in enumerate(prompts.items())]
    ctx = {"pair": 1, "run_root": rc.run_root.hex(), "requests": reqs, "sampling": {"max_tokens": 3}, "code": "test",
           "args": {"program_dir": "/served/build_request", "programs_root": None, "program_digest": inst["digest"],
                    "required_manifest": None, "label": "rope-d64-cpu"}}
    m = RB.seal(out, 1, ctx, out / "binding_map_p1.json")
    (out / "runs.jsonl").write_text(json.dumps(row) + "\n")
    (out / "verdict.json").write_text(json.dumps({"replay_deferred": {"bundles": [{"pair": 1, "manifest_sha256": m["sha256"]}]}}))
    pf = root / "partition.json"
    pf.write_text(json.dumps({"templates": [ROPE], "law": LAW}))
    return {"out": out, "row_dir": row_dir, "partition": pf, "world": world, "run_root": rc.run_root.hex(), "inst": inst}


def main(argv: list[str]) -> int:
    root = Path(argv[0])
    root.mkdir(parents=True, exist_ok=True)
    c = commit(root)
    print(json.dumps({"commit": str(c["out"]), "bundle": str(RB.path(c["out"], 1)), "row_dir": str(c["row_dir"]),
                      "partition": str(c["partition"]), "run_root": c["run_root"], "program_digest": c["inst"]["digest"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
