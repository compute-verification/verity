"""A TP Commit served on CPU, for the TP serving rows' tests and the one-stage audit's TP run (`benchmarks/one_stage/a5.py --tp-commit`):
``world`` ranks, each the toy deferred Commit of `tests.pipeline.served_rope` with its own weights and its act rows ``SiluMul_v2{I}``
(GLM-5.3's Definition), each rank's window written by `pipeline.tp.serving_rows.write` as a vLLM rank writes it, under the TP run root
of the ranks' roots, beside what the verifier reads: the TP Commit's record (``commit/runs.jsonl``, ``commit/versions.json``), the
Build's rank Programs (``build/rank<r>``), the workload and the partition file.

    python -m tests.pipeline.served_tp OUT [--world 2] [--I 256] [--law subset:2]        (from integrations/vllm)

writes ``OUT/commit``, ``OUT/build``, ``OUT/workload.json``, ``OUT/partition.json`` and ``OUT/serving-rows`` (``tp.json``, ``rank<r>/``),
and prints them with the TP run root as one JSON line."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

from tests.pipeline import served_rope

LABEL = "served-tp"
MAX_TOKENS = 3


def commit(root: Path, *, world: int = 2, I: int = 256, law: str = "subset:2") -> dict:
    """The TP Commit's files under ``root`` (module docstring), and its TP run root, record and windows."""
    from verity_vllm.check.replay import driver as SR
    from verity_vllm.engine import vllm_adapter as va
    from verity_vllm.pipeline import deferred_rows as DRW
    from verity_vllm.pipeline import replay_bundle as RB
    from verity_vllm.pipeline.tp import serving_rows as SRT
    from verity_vllm.pipeline.tp.run_root import TP_ROOT_TAG, tp_run_root

    root.mkdir(parents=True, exist_ok=True)
    pf = root / "partition.json"
    pf.write_text(json.dumps({"templates": [f"SiluMul_v2{{I={int(I)}}}"], "law": law}))
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
    tp_root = tp_run_root(world, roots)
    of_record = {str(r): [c["inst"]["digest"]] for r, (c, _o) in enumerate(ranks)}
    tp = {"world": world, "tp_run_root": tp_root.hex(), "roots": {str(r): x.hex() for r, x in roots.items()}, "root_tag": TP_ROOT_TAG.decode()}
    req = {"dir": str(root / "serving-rows"), "partition": str(pf), "run": LABEL, "committer": {"source": __name__}, "row": LABEL}
    got, provider = [], SR.model_weights_provider
    try:
        for r, (_c, o) in enumerate(ranks):
            weights = DRW.opened_weights(o)
            SR.model_weights_provider = lambda model, prefix, w=weights: w
            st = {"rank": r, "rc": o.run, "com": o.store, "bmap": json.loads((o.out / f"binding_map_p{o.pair}.json").read_text()),
                  "eng": SimpleNamespace(model=None), "manifest_doc": None,
                  "replay": {"programs": {str(r): str(root / "build" / f"rank{r}")}, "of_record": of_record, "requests": reqs}}
            got.append({"rank": r, "window": SRT.write(st, req, tp)})
    finally:
        SR.model_weights_provider = provider
    windows = SRT.gather(req["dir"], tp, got)
    out = root / "commit"
    out.mkdir()
    rows = [{"pair": 0, "arm": "control", "label": LABEL},
            {"pair": 0, "arm": "instrumented", "label": LABEL, "workload_sha256": wl_sha,
             "commit": {"tp_run_root": tp_root.hex(), "world": world, "per_rank": {str(r): {"run_root": x.hex()} for r, x in roots.items()}}}]
    (out / "runs.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    (out / "versions.json").write_text(json.dumps({"root_tag": TP_ROOT_TAG.decode(), "sampled_replay": {"of_record": of_record},
                                                   "argv": ["tp-commit", "--workload", wl.name, "--tp", str(world), "--label", LABEL]}))
    return {"commit": out, "build": root / "build", "workload": wl, "partition": pf, "windows": Path(req["dir"]), "tp_run_root": tp_root.hex(),
            "world": world, "tp": windows}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--world", type=int, default=2)
    ap.add_argument("--I", type=int, default=256)
    ap.add_argument("--law", default="subset:2")
    a = ap.parse_args(argv)
    c = commit(Path(a.out), world=a.world, I=a.I, law=a.law)
    print(json.dumps({k: str(c[k]) for k in ("commit", "build", "workload", "partition", "windows", "tp_run_root", "world")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
