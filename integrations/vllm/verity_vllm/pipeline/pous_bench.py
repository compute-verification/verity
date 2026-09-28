"""`verity-vllm pous-bench`: serving throughput through vLLM, plaintext against POUS (`quantization="pous"`), on one GPU.

    verity-vllm pous-bench --mode plain|pous --case QWEN3B --out DIR --build-dir DIR [--prefill 512x1,2048x8] [--decode 1,8,32,256]

Each configuration steps the in-process engine (`LLMEngine.step`) and times every step:

* prefill `L x b`: `b` prompts of `L` tokens, one output token each; tokens/s = `b * L` over the time to every first token;
* decode `B`: `B` prompts of `--decode-prompt` tokens and `1 + N` output tokens each; tokens/s = `B * N` over the time after the
  last first token.

POUS decodes every weight from `C` at every use, so each forward costs the whole model's decode; `N` (`--decode-tokens`) is kept
small there and each configuration is a few forward passes.  POUS runs also record the store's decode time and decoded bytes per
configuration (where the time goes) and a codec-only decode of `--codec-segments` segments (latency against throughput).  The
salt is drawn locally: this is a throughput run, not an audit.  Prompts are fixed token ids, so both modes see the same inputs.
"""
from __future__ import annotations

import dataclasses
import json
import os
import time
from pathlib import Path
from typing import Any

from verity_vllm.config import option


@dataclasses.dataclass(frozen=True, kw_only=True)
class Options:
    """time prefill and decode through vLLM, plaintext or POUS"""

    mode: str = option("--mode", required=True, choices=["plain", "pous"], help="plaintext weights or quantization='pous'")
    case: str = option("--case", required=True, help="the checkpoint role in manifests/checkpoints.json")
    out: str = option("--out", required=True, help="where bench-<mode>.json goes")
    build_dir: str = option("--build-dir", required=True, help="parent directory of the pous_native JIT build")
    prefill: str = option("--prefill", default="512x1,512x8,512x32,2048x1,2048x8", help="prompt length x batch, comma separated")
    decode: str = option("--decode", default="1,8,32,256", help="decode batch sizes, comma separated")
    decode_prompt: int = option("--decode-prompt", type=int, default=128, help="prompt tokens of the decode configurations")
    decode_tokens: int = option("--decode-tokens", type=int, default=64, help="decode steps timed per decode configuration")
    max_batched_tokens: int = option("--max-batched-tokens", type=int, default=16384, help="scheduler token budget per step")
    layers_per_decode: int = option("--layers-per-decode", type=int, default=1, help="transformer layers decoded together")
    thread_rows: int = option("--thread-rows", type=int, default=0, help="rows from which a Feistel launch runs a thread per row")
    load_check: bool = option("--no-load-check", action="store_false", default=True, help="skip the load-time bit-exact check")
    codec: str = option("--codec", default="p3/v1", help="the reference codec served (p3/v1, dense-chain/v1)")
    kernels: str = option("--kernels", default="", help="Feistel dispatch as JSON: word_io, fused, pair, pair_max_rows, coalesced, chunked, chunked_threads")
    audit: int = option("--audit", type=int, default=0, help="after timing, this many timed audits against the live server")
    audit_k: int = option("--audit-k", type=int, default=111, help="challenges per audit")
    unencoded: bool = option("--unencoded", action="store_true", default=False,
                             help="timing only: serve W itself as C (decode costs the same; outputs are garbage)")
    codec_segments: str = option("--codec-segments", default="", help="codec-only decodes, segments:thread_rows, e.g. 11:0,11:1 (pous)")
    result: str | None = option("--result", default=None, help="also write the research result.json here")
    run_id: str | None = option("--run-id", default=None, help="the research run id result.json names")


def prompt(i: int, length: int) -> list[int]:
    """Fixed token ids (no sampling): request `i`'s prompt of `length` tokens."""
    return [1000 + (i * 7919 + j * 104729) % 30000 for j in range(length)]


def run_config(engine, requests: list[list[int]], max_tokens: int, store_info) -> dict[str, Any]:
    """Step the engine over `requests` until done; each step's wall time and the requests with a first token after it."""
    from vllm import SamplingParams

    llm_engine = engine.llm.llm_engine
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
    tag = f"r{time.monotonic_ns()}"
    for i, ids in enumerate(requests):
        llm_engine.add_request(f"{tag}-{i}", {"prompt_token_ids": ids}, params)
    before = store_info()
    steps, first, done = [], set(), {}
    t0 = time.perf_counter()
    while llm_engine.has_unfinished_requests():
        s = time.perf_counter()
        outs = llm_engine.step()
        e = time.perf_counter()
        for o in outs:
            if o.outputs and o.outputs[0].token_ids:
                first.add(o.request_id)
            if o.finished:
                done[o.request_id] = len(o.outputs[0].token_ids)
        steps.append({"s": e - s, "at": e - t0, "first": len(first)})
    after = store_info()
    row = {"requests": len(requests), "prompt": len(requests[0]), "max_tokens": max_tokens, "steps": steps,
           "generated": sum(done.values()), "wall_s": time.perf_counter() - t0}
    if before is not None:
        row["decode_s"] = after["decode_s"] - before["decode_s"]
        row["decodes"] = after["decodes"] - before["decodes"]
        row["decoded_bytes"] = after["decoded_bytes"] - before["decoded_bytes"]
    return row


def first_token_at(row: dict[str, Any]) -> float:
    return next(s["at"] for s in row["steps"] if s["first"] == row["requests"])


def prefill(engine, length: int, batch: int, store_info) -> dict[str, Any]:
    row = run_config(engine, [prompt(i, length) for i in range(batch)], 1, store_info)
    t = first_token_at(row)
    return {"kind": "prefill", "length": length, "batch": batch, "tokens": length * batch, "seconds": t,
            "tokens_per_s": length * batch / t, **row}


def decode(engine, batch: int, prompt_len: int, n: int, store_info) -> dict[str, Any]:
    row = run_config(engine, [prompt(i, prompt_len) for i in range(batch)], 1 + n, store_info)
    t = row["wall_s"] - first_token_at(row)
    return {"kind": "decode", "batch": batch, "tokens": batch * n, "seconds": t, "tokens_per_s": batch * n / t, **row}


def codec_only(segments: list[tuple[int, int]], build_dir: str, name: str) -> list[dict[str, Any]]:
    """The GPU decode of `G` random segments at once, for each `(G, thread_rows)`: seconds, and decoded MB/s."""
    import torch

    from verity_vllm.engine.pous import codec, native, twins

    native.load(build_dir)
    ref, geometry = twins.reference(name)
    decode_fn = twins.twin(ref)[1]
    rows = []
    for g, thread_rows in segments:
        codec.THREAD_ROWS[0] = thread_rows
        c = torch.randint(0, 256, (g, geometry.blocks_per_segment, geometry.block_bytes), dtype=torch.uint8, device="cuda")
        tags = codec.segment_tags(bytes(codec.SALT_BYTES), list(range(g)), c.device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        decode_fn(c, tags)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        rows.append({"segments": g, "thread_rows": thread_rows, "seconds": dt,
                     "decoded_mb_per_s": g * geometry.segment_bytes / dt / 1e6})
        del c
        torch.cuda.empty_cache()
    return rows


def live_audit(engine, out: Path, a: Options) -> dict[str, Any]:
    """Timed audits by a separate verifier process (`pous-dense-audit`) against the native responder, the server decoding
    and serving forward passes all the while."""
    import subprocess
    import sys

    from vllm import SamplingParams

    llm = engine.llm
    info = llm.collective_rpc("pous_info")[0]
    (out / "leaves-audit.bin").write_bytes(llm.collective_rpc("pous_leaves")[0])
    port = llm.collective_rpc("pous_start_responder", args=("127.0.0.1", 0, 2))[0]
    spec = json.loads((out / "pous-bench.json").read_text())
    from verity_vllm.engine.pous import twins

    geometry = twins.reference(a.codec)[1].with_segments(info["segments"])
    (out / "audit-ready.json").write_text(json.dumps({"port": port, "salt": spec["salt"], "codec": a.codec,
                                                     "segments": geometry.segments}))
    proc = subprocess.Popen([sys.executable, "-m", "verity_vllm.pipeline.cli", "pous-dense-audit", "--ready", str(out / "audit-ready.json"),
                             "--leaves", str(out / "leaves-audit.bin"), "--out", str(out / "audit.json"), "--k", str(a.audit_k),
                             "--audits", str(a.audit), "--verifier-cpu", "3"], stdout=(out / "audit.log").open("w"),
                            stderr=subprocess.STDOUT)
    forwards = 0
    while proc.poll() is None:
        llm.generate([{"prompt_token_ids": prompt(0, a.decode_prompt)}], SamplingParams(temperature=0.0, max_tokens=1),
                     use_tqdm=False)
        forwards += 1
    counts = llm.collective_rpc("pous_stop_responder")[0]
    doc = json.loads((out / "audit.json").read_text()) if (out / "audit.json").exists() else {}
    return {"rc": proc.returncode, "forwards_during_audit": forwards, "responder": counts, **doc}


def main(a: Options) -> int:
    from verity_vllm.pipeline import pous_run

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.audit:
        os.sched_setaffinity(0, set(os.sched_getaffinity(0)) - {2, 3})
    doc: dict[str, Any] = {"mode": a.mode, "case": a.case, "codec": a.codec, "unencoded": a.unencoded, "options": dataclasses.asdict(a)}
    if a.mode == "pous" and a.codec_segments:
        doc["codec_only"] = codec_only([tuple(int(y) for y in x.split(":")) for x in a.codec_segments.split(",")], a.build_dir,
                                       a.codec)
    spec = None
    if a.mode == "pous":
        spec = out / "pous-bench.json"
        spec.write_text(json.dumps({"safetensors": pous_run.checkpoint_files(a.case), "build_dir": a.build_dir,
                                    "salt": os.urandom(24).hex(), "drop_fraction": 0.0, "layers_per_decode": a.layers_per_decode,
                                    "thread_rows": a.thread_rows, "load_check": a.load_check, "codec": a.codec,
                                    "unencoded": a.unencoded, **(json.loads(a.kernels) if a.kernels else {})}))
    batches = [int(x) for x in a.decode.split(",")]
    t0 = time.perf_counter()
    engine = pous_run.build(a.case, spec, max_num_seqs=max(batches + [32]), engine_args={"max_num_batched_tokens": a.max_batched_tokens})
    doc["build_s"] = time.perf_counter() - t0
    info = (lambda: engine.llm.collective_rpc("pous_info")[0]) if a.mode == "pous" else (lambda: None)
    if a.mode == "pous":
        doc["store"] = {k: v for k, v in info().items() if k not in ("checks", "digests", "memory")}
        doc["load_checks"] = info()["checks"]
    rows = []
    for item in filter(None, a.prefill.split(",")):
        length, batch = (int(x) for x in item.split("x"))
        rows.append(prefill(engine, length, batch, info))
        print(json.dumps({k: rows[-1][k] for k in ("kind", "length", "batch", "tokens_per_s", "seconds")}), flush=True)
    for batch in batches:
        rows.append(decode(engine, batch, a.decode_prompt, a.decode_tokens, info))
        print(json.dumps({k: rows[-1][k] for k in ("kind", "batch", "tokens_per_s", "seconds")}), flush=True)
    doc["rows"] = rows
    if a.mode == "pous":
        from vllm import SamplingParams

        before = info()["decodes"]
        t1 = time.perf_counter()
        engine.llm.generate([{"prompt_token_ids": prompt(0, a.decode_prompt)}], SamplingParams(temperature=0.0, max_tokens=1),
                            use_tqdm=False)
        doc["probe_forward_s"] = time.perf_counter() - t1
        doc["store"] = {k: v for k, v in info().items() if k not in ("checks", "digests", "memory")}
        doc["store"]["groups_per_forward"] = doc["store"]["decodes"] - before
    if a.mode == "pous" and a.audit:
        doc["audit"] = live_audit(engine, out, a)
    (out / f"bench-{a.mode}.json").write_text(json.dumps(doc, indent=1, default=str))
    if a.result:
        Path(a.result).write_text(json.dumps({
            "schema": "research/result/v0.1", "run_id": a.run_id,
            "workload_fingerprint": {"mode": a.mode, "case": a.case, "codec": a.codec, "unencoded": a.unencoded,
                                     "prefill": a.prefill, "decode": a.decode},
            "validation": {"status": "passed", "checker": "verity-vllm pous-bench", "detail": "every configuration ran",
                           "evidence": [f"bench-{a.mode}.json"]},
            "measurements": [{"name": f"{r['kind']}_{r.get('length', '')}x{r['batch']}_tokens_per_s".replace("_x", "_"),
                              "value": r["tokens_per_s"], "unit": "token/s"} for r in rows],
            "measurement_files": [f"bench-{a.mode}.json"], "artifacts": [f"bench-{a.mode}.json"]}, indent=1))
    return 0
