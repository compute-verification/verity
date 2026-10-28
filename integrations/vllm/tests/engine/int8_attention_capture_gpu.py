"""The recorded int8 attention fold test's capture (`test_int8_attention_recorded`): Llama-3.1-8B served on this GPU with
`TargetProfile.int8_exact` "all" under the Match capture (`pipeline.m1_capture`, observer on, snapshots of steps 0 and 1), then one
layer's attention scope cut out of it (`int8_attention_cut`) as the test's fixture.

    cd integrations/vllm; PY=/workspace/jobs/venv312/bin/python3     # PYTHONPATH=.:../..:../../tools/research/src
    $PY -m tests.engine.int8_attention_capture_gpu --out DIR --at ISO_TIME --by LANE [--layer 0]

The workload is the int8 Llama-3.2-1B sweep workload's engine and target (`WORKLOAD_FROM`, the same int8 arm and CUSTOM attention backend)
on the registered `LLAMA31_8B_INT8` pin (`int8_exact_gpu.pin`), with two requests of PROMPTS' lengths (that row's first prompt ids;
Llama 3.1 and 3.2 share the tokenizer) and two tokens each: step 0 is one batched prefill of both, step 1 one batched decode, so
each step's kv_quant_batch pads the shorter window.  m1_capture writes to a scratch directory; DIR gets workload.json, capture/
(its log and records, and the snapshots of the layer's scope at both steps: what the cutter reads) and the fixture, cut from
capture/, with its values beside it.
"""
from __future__ import annotations

from verity_vllm.pipeline import m1_capture  # first: m1_capture applies the engine environment before vLLM is imported

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from tests.engine import int8_attention_cut as C
from tests.engine.int8_exact_gpu import pin, scratch_checkpoints

WORKLOAD_FROM = Path(__file__).parents[2] / "workloads" / "llama32-1b__int8__rtxpro6000__tp1__b1__i256__o32__mixed__greedy__bi-eager.json"
PROMPTS = (5, 3)
MAX_TOKENS = 2
STEPS = (0, 1)


def workload(revision: str) -> dict:
    doc = json.loads(WORKLOAD_FROM.read_text())
    ids = doc["requests"][0]["prompt_token_ids"]
    at = [0]
    for n in PROMPTS:
        at.append(at[-1] + n)
    doc.update(model="verity-int8/Llama-3.1-8B-INT8", case=C.ROLE, prompt_lengths=list(PROMPTS), max_tokens=[MAX_TOKENS] * len(PROMPTS),
               expected_engine_steps=MAX_TOKENS, execution=doc["sweep"]["execution"], note="int8 attention fold capture (int8_attention_capture_gpu)")
    doc["engine"] = {**doc["engine"], "revision": revision, "args": {**doc["engine"]["args"], "max_model_len": 64}}
    doc["requests"] = [{"request_id": f"int8-attn-{i}", "index": i, "prompt_len": n, "max_tokens": MAX_TOKENS, "arrive_step": 0,
                        "arrival_offset_s": 0.0, "prompt_token_ids": ids[at[i]:at[i + 1]]} for i, n in enumerate(PROMPTS)]
    del doc["sweep"]
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--checkpoints", default="manifests/checkpoints.json")
    ap.add_argument("--base-role", default="LLAMA31_8B")
    ap.add_argument("--role", default=C.ROLE)
    ap.add_argument("--repo", default="verity-int8/Llama-3.1-8B-INT8")
    ap.add_argument("--hf-home", default=os.environ.get("HF_HOME"))
    ap.add_argument("--at", required=True, help="resolved_at of the pin entry, should it have to be made (UTC, ISO 8601)")
    ap.add_argument("--by", required=True, help="resolved_by of the pin entry, should it have to be made")
    ap.add_argument("--layer", type=int, default=0)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    entry, how = pin(a)
    print(f"[int8-attn-capture] pin {entry['repo']}@{entry['revision']} ({how})", flush=True)
    wl = a.out / "workload.json"
    wl.write_text(json.dumps(workload(entry["revision"]), indent=1) + "\n")
    kept = a.out / "capture"
    with tempfile.TemporaryDirectory() as scratch:
        checkpoints = scratch_checkpoints(a.checkpoints, entry, Path(scratch))
        capture = Path(scratch) / "capture"
        m1_capture.main(m1_capture.Options(workload=str(wl), out=str(capture), checkpoints=str(checkpoints), case=a.role,
                                           snapshot_steps=",".join(map(str, STEPS)), max_num_seqs=len(PROMPTS)))
        keep_scopes(capture, kept, a.layer)
    fixture = a.out / f"llama31_8b_int8_attn_l{a.layer}.jsonl.gz"
    rc = C.main([str(kept), "--layer", str(a.layer), "--steps", ",".join(map(str, STEPS)), "--out", str(fixture),
                 "--origin", f"{os.uname().nodename}:{a.out.parent.name}/out/capture/log.jsonl.gz"])
    print(f"INT8-ATTN-CAPTURE rc={rc} fixture={fixture}", flush=True)
    return rc


def keep_scopes(capture: Path, kept: Path, layer: int) -> None:
    """The capture's log, records and the snapshots of the layer's attention scope at STEPS, copied to `kept` (a SnapshotStore the
    cutter cuts the fixture from)."""
    span: dict[int, tuple[int, int]] = {}
    for _, step in C.split_steps(C.read_log(capture / "log.jsonl.gz")):
        if step and step[0].step in STEPS:
            ids = C.scope_of(C.step_events(step).roots, layer).ids()
            span[step[0].step] = (min(ids), max(ids))
    idx = json.loads((capture / "snapshots.json").read_text())
    idx["entries"] = [e for e in idx["entries"] if e.get("step") in span and span[e["step"]][0] <= e["event"] <= span[e["step"]][1]]
    for f in {e["file"] for e in idx["entries"] if e.get("file")} | {"log.jsonl.gz", "tokens.json", "stats.json", "versions.json"}:
        if (capture / f).is_file():
            (kept / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(capture / f, kept / f)
    (kept / "snapshots.json").write_text(json.dumps(idx, separators=(",", ":")))
    print(f"[int8-attn-capture] kept {len(idx['entries'])} snapshots of layer {layer}'s scope at steps {sorted(span)} "
          f"(events {span}) of {capture}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
