"""A TP Commit served on CPU, for the TP serving rows' tests and the one-stage audit's TP run (`benchmarks/one_stage/a5.py --tp-commit`):
``world`` ranks, each the toy deferred Commit of `tests.pipeline.served_rope` with its own weights and its act rows ``SiluMul_v2{I}``
(GLM-5.3's Definition), committed under rows-sha512 as `tp-commit` commits a rank, each rank's window written by
`pipeline.tp.serving_rows.write` as a vLLM rank writes it, under the SHA-512 TP run root of the ranks' roots (`tp.run_root`), beside
what the verifier reads: the TP Commit's record (``commit/runs.jsonl``, ``commit/versions.json``), the Build's rank Programs
(``build/rank<r>``), the workload and the partition file.  ``--deferred``: the Commit wrote no window and deferred every rank's replay, as
``tp-commit --replay-deferred`` leaves it (each rank's store, weights tree and binding map sealed as its pair-0 bundle
``commit/replay_bundle_p0_rank<r>``, ``commit/replay_deferred.json`` naming them, every rank PENDING), so its windows come from the
bundles (`pipeline.tp.serving_rows.from_bundle`).

    python -m tests.pipeline.served_tp OUT [--world 2] [--I 256] [--law subset:2] [--template T ...] [--deferred]
    (from integrations/vllm)

``--template``: a template of the toy's Programs the partition holds before ``SiluMul_v2{I}`` (``RoPEHead_v1{D=64}``, the advice
windows' stand-in transfer gate), repeatable.

writes ``OUT/commit``, ``OUT/build``, ``OUT/workload.json``, ``OUT/partition.json`` and, unless deferred, ``OUT/serving-rows``
(``tp.json``, ``rank<r>/``), and prints them with the TP run root as one JSON line."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

from tests.pipeline import served_rope

LABEL = "served-tp"
ROWS = "rows-sha512"
MAX_TOKENS = 3


def commit(root: Path, *, world: int = 2, I: int = 256, law: str = "subset:2", deferred: bool = False,
           templates: tuple[str, ...] = ()) -> dict:
    """The TP Commit's files under ``root`` (module docstring), and its TP run root, record and windows (deferred: none)."""
    from verity_vllm.check.replay import driver as SR
    from verity_vllm.engine import vllm_adapter as va
    from verity_vllm.pipeline import deferred_rows as DRW
    from verity_vllm.pipeline import replay_bundle as RB
    from verity_vllm.pipeline.tp import serving_rows as SRT
    from verity_vllm.pipeline.tp import run_root as RT

    root.mkdir(parents=True, exist_ok=True)
    pf = root / "partition.json"
    pf.write_text(json.dumps({"templates": [*templates, f"SiluMul_v2{{I={int(I)}}}"], "law": law}))
    ranks = []
    for r in range(world):
        c = served_rope.commit(root / "ranks" / f"rank{r}", seed=17 + r, silu="SiluMul_v2", I=I)
        b = root / "build" / f"rank{r}"
        b.mkdir(parents=True)
        shutil.copyfile(c["row_dir"] / "build_request" / "instances.json.gz", b / "instances.json.gz")
        (b / "result.json").write_text(json.dumps({"program_digest": c["inst"]["digest"], "tp": {"world": world, "rank": r}}))
        ranks.append((c, DRW.of_record(RB.path(c["out"], 1), str(c["row_dir"]))))
    prompts = ranks[0][0]["world"].prompts
    wl = root / "workload.json"
    wl.write_text(json.dumps({"requests": [{"request_id": rid, "prompt_token_ids": pr, "max_tokens": MAX_TOKENS, "arrive_step": 0}
                                           for rid, pr in prompts.items()], "sampling": {"variant": "greedy"}}))
    wl_sha = va.load_workload(wl)["sha256"]
    reqs = [{"request_id": f"r{i}", "arrive_step": 0, "prompt_len": len(pr), "max_tokens": MAX_TOKENS} for i, pr in enumerate(prompts.values())]
    roots = {r: bytes(o.run.run_root) for r, (_c, o) in enumerate(ranks)}
    tag = RT.TP_ROOT_TAG.decode()
    tp_root = RT.combine(tag, world, roots)
    of_record = {str(r): [c["inst"]["digest"]] for r, (c, _o) in enumerate(ranks)}
    tp = {"world": world, "tp_run_root": tp_root.hex(), "roots": {str(r): x.hex() for r, x in roots.items()}, "root_tag": tag}
    replay = {"programs": {str(r): str(root / "build" / f"rank{r}") for r in range(world)}, "of_record": of_record, "requests": reqs}
    out = root / "commit"
    out.mkdir()
    per_rank = {str(r): {"run_root": x.hex()} for r, x in roots.items()}
    req, windows = {"dir": str(root / "serving-rows"), "partition": str(pf), "run": LABEL, "committer": {"source": __name__}, "row": LABEL}, None
    if deferred:
        seal(out, tp, ranks, replay, per_rank)
    else:
        got, provider = [], SR.model_weights_provider
        try:
            for r, (_c, o) in enumerate(ranks):
                weights = DRW.opened_weights(o)
                SR.model_weights_provider = lambda model, prefix, w=weights: w
                st = {"rank": r, "rc": o.run, "com": o.store, "bmap": json.loads((o.out / f"binding_map_p{o.pair}.json").read_text()),
                      "eng": SimpleNamespace(model=None), "weights": None, "manifest_doc": None,
                      "replay": dict(replay, programs={str(r): replay["programs"][str(r)]})}
                got.append({"rank": r, "window": SRT.write(st, req, tp)})
        finally:
            SR.model_weights_provider = provider
        windows = SRT.gather(req["dir"], tp, got)
    commit_doc = {"tp_run_root": tp_root.hex(), "world": world, "per_rank": per_rank}
    if deferred:
        commit_doc.update(commitment_vector=[{"rank": r, "run_root": x.hex()} for r, x in roots.items()], commit_pass=None)
    rows = [{"pair": 0, "arm": "control", "label": LABEL},
            {"pair": 0, "arm": "instrumented", "label": LABEL, "workload_sha256": wl_sha, "commit": commit_doc}]
    (out / "runs.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    (out / "versions.json").write_text(json.dumps({"root_tag": tag, "commit_scheme": ROWS, "sampled_replay": {"of_record": of_record},
                                                   "argv": ["tp-commit", "--workload", wl.name, "--tp", str(world), "--label", LABEL]}))
    return {"commit": out, "build": root / "build", "workload": wl, "partition": pf, "windows": None if deferred else Path(req["dir"]),
            "tp_run_root": tp_root.hex(), "world": world, "tp": windows}


def seal(out: Path, tp: dict, ranks: list, replay: dict, per_rank: dict) -> None:
    """Every rank's store, weights tree and binding map sealed as its pair-0 bundle under ``out``, as a deferring rank seals it
    (`pipeline.tp.c2.seal`'s context), ``replay_deferred.json`` naming them, and each rank's block of the record PENDING on it."""
    from verity_vllm.pipeline import replay_bundle as RB
    from verity_vllm.pipeline.tp import c2 as C2

    bundles = []
    for r, (_c, o) in enumerate(ranks):
        part, root = RB.partial(out, 0, rank=r), per_rank[str(r)]["run_root"]
        part.mkdir(parents=True)
        for d in ("store", "weights"):
            shutil.copytree(o.bdir / d, part / d)
        bmap = json.loads((o.out / f"binding_map_p{o.pair}.json").read_text())
        canon = json.dumps(bmap.get("entries"), sort_keys=True, separators=(",", ":"), default=str).encode()
        bmap.update(rank=r, run_root=root, digest=hashlib.sha256(canon).hexdigest())
        (part / "binding_map.json").write_text(json.dumps(bmap))
        wr = o.row["commit"]["weight_registration"]
        ctx = {"tp": {"world": tp["world"], "rank": r}, "pair": 0, "label": LABEL, "code": "test", "run_root": root, "tp_run_root": tp["tp_run_root"],
               "weight_registration": wr, "replay": replay, "required_manifest": None, "required_manifest_sha256": None,
               "engine_of_record": {}, "programs": {}}
        m = RB.seal(out, 0, ctx, None, rank=r)
        bundles.append({"pair": 0, "rank": r, "path": m["path"], "manifest_sha256": m["sha256"], "bytes": m["bytes"], "files": len(m["files"])})
        per_rank[str(r)].update(commit_pass=None, replay_deferred={"bundle": Path(m["path"]).name}, binding={"map_digest": bmap["digest"]},
                                weight_registration=wr)
    (out / C2.DEFERRED).write_text(json.dumps({"schema": C2.SCHEMA, "world": tp["world"], "pending_pairs": [0], "bundles": bundles}))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--world", type=int, default=2)
    ap.add_argument("--I", type=int, default=256)
    ap.add_argument("--law", default="subset:2")
    ap.add_argument("--template", action="append", default=[], help="a template the partition holds before SiluMul_v2{I} (repeatable)")
    ap.add_argument("--deferred", action="store_true", help="no window: every rank's replay deferred to its sealed bundle")
    a = ap.parse_args(argv)
    c = commit(Path(a.out), world=a.world, I=a.I, law=a.law, deferred=a.deferred, templates=tuple(a.template))
    print(json.dumps({k: str(c[k]) for k in ("commit", "build", "workload", "partition", "windows", "tp_run_root", "world")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
