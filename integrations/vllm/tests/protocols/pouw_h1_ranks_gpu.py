"""H1's fill window served on one GPU and at TP N × PP M (`pouw_h1_ranks`) on one node, and the two held to each other.

``--pp 1 --tp 1`` builds the engine as `pouw_h1_gpu.fill` does and installs H1 through the option (`pouw._install_h1`);
otherwise it builds N tensor-parallel ranks × M pipeline stages on the multiprocess executor, synchronous scheduling and
`pouw_h1_ranks.EXTENSION`, and installs the holder (`pouw_h1_ranks.install`). Each serves `pouw_h1.FILL`'s 64 one-token
greedy requests plain, then under the window, and writes its tokens, the window's record (``pouw-h1-calls.json``) and a
summary to ``--out``.

Both draw their in-process coins from one fixed stream (`entropy`) and use one fixed hm96 key (`KEY`), a measurement that
credits nothing (the coins aren't a challenger's) but makes the two windows' commitments comparable bit for bit.
``--compare ONE_DIR OTHER_DIR`` (no GPU) holds the served tokens equal and both layouts static and complete (every one of
the window's steps × linears called). At one TP (pipeline stages only) the layout is the same, so it also holds every
call's (index, step, linear, module, m, k, n, base, rows) equal in count and order, and each call's A and z table roots,
every round's root and coin, the calls root and the draw root equal. Across TPs the linears are shards, so it holds the
window's MACs a step equal instead (each linear's shard n·K summed over the ranks)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

SCHEMA = "verity/pouw-h1-ranks-gpu/v1"
KEY = hashlib.sha512(b"verity/pouw-h1-ranks-gpu/key/v1").digest()[:32]
SUMMARY = "ranks-summary.json"


def entropy(label: str) -> Callable[[int], bytes]:
    """A fixed SHA-512 counter stream for `label`: the same coins in every run, so no verifier trusts them."""
    state = {"i": 0, "buf": b""}

    def draw(n: int) -> bytes:
        while len(state["buf"]) < n:
            state["buf"] += hashlib.sha512(f"verity/pouw-h1-ranks-gpu/{label}/{state['i']}".encode()).digest()
            state["i"] += 1
        out, state["buf"] = state["buf"][:n], state["buf"][n:]
        return out
    return draw


def coins(steps: int) -> tuple[Any, Any]:
    from verity.core.protocols.pouw import window as Wd
    from verity.core.protocols.pouw.coins import Challenger
    from verity.core.protocols.sampled_proofs.challenger import LiveRecord
    return Challenger(0, entropy("salt")), LiveRecord(Wd.live_challenger(0, steps, entropy("rounds")), Wd.ROUND)


def summary(h1: Any) -> dict[str, Any]:
    return {"static": h1.static(), "step_rows": h1.step_rows(), "calls": len(h1.calls), "u_step": h1.u_step,
            "shapes": [list(s) for s in h1.shapes_], "refused": h1.refused[:20], "order_ok": h1.order_ok,
            "u": str(h1.utilization()), "details": {k: v for k, v in h1.details.items() if k not in ("refused",)}}


def serve(a: argparse.Namespace) -> dict[str, Any]:
    import torch
    from tests.engine.int8_exact_gpu import pin, scratch_checkpoints
    from tests.protocol_options.pouw_h1_gpu import fill_prompts, generate_static, static_check, tokens_digest
    from verity_vllm.program.frontend.target_profile import TargetProfile
    from verity_vllm.protocols.pouw import h1 as PH
    from verity_vllm.protocols.pouw.h1 import requests as RQ

    decl, out = PH.FILL, Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    target = TargetProfile.from_device().with_overrides(int8_exact="all")
    tag = f"[pp{a.pp}tp{a.tp}]"
    doc: dict[str, Any] = {"schema": SCHEMA, "argv": sys.argv, "pp": a.pp, "tp": a.tp, "torch": torch.__version__,
                           "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                           "public": {"slots": decl.slots, "rows": decl.rows, "steps": decl.steps,
                                      "max_tokens": decl.max_tokens}}
    args = decl.engine_args()
    max_num_seqs = args.pop("max_num_seqs")
    t0 = time.monotonic()
    with tempfile.TemporaryDirectory() as scratch:
        entry, how = pin(a)
        doc["pin"] = {"repo": entry["repo"], "revision": entry["revision"], "source": how}
        cps = scratch_checkpoints(a.checkpoints, entry, Path(scratch))
        if a.pp == a.tp == 1:
            from verity_vllm.engine.build import build_engine
            eng = build_engine(cps, case=a.role, target=target, max_num_seqs=max_num_seqs, engine_args=args)
            llm = eng.llm
        else:
            from vllm import LLM
            from verity_vllm.engine.build import engine_kwargs_for
            from verity_vllm.protocols.pouw.h1 import ranks as PHR
            kw, _env, _cp = engine_kwargs_for(cps, max_num_seqs=max_num_seqs, case=a.role, target=target, engine_args={
                **args, "pipeline_parallel_size": a.pp, "tensor_parallel_size": a.tp, "distributed_executor_backend": "mp",
                "worker_extension_cls": PHR.EXTENSION, "async_scheduling": a.async_scheduling})
            llm = LLM(**kw)
    cfg = llm.llm_engine.vllm_config
    doc["engine"] = {"build_s": round(time.monotonic() - t0, 1), "pp": cfg.parallel_config.pipeline_parallel_size,
                     "tp": cfg.parallel_config.tensor_parallel_size,
                     "async_scheduling": cfg.scheduler_config.async_scheduling,
                     "max_concurrent_batches": cfg.max_concurrent_batches,
                     "max_num_batched_tokens": cfg.scheduler_config.max_num_batched_tokens,
                     "max_num_seqs": cfg.scheduler_config.max_num_seqs}
    print(f"{tag} engine {json.dumps(doc['engine'])}", flush=True)
    prompts = fill_prompts(llm.get_tokenizer(), decl.slots)
    cap = decl.max_tokens
    t0 = time.monotonic()
    plain = generate_static(llm, prompts, cap)
    doc["plain"] = {"s": round(time.monotonic() - t0, 2), "tokens_digest": tokens_digest(plain),
                    "static": static_check(plain, cap)}
    print(f"{tag} plain {json.dumps(doc['plain'])}", flush=True)
    salt_coins, rounds = coins(decl.steps)
    t0 = time.monotonic()
    if a.pp == a.tp == 1:
        from verity_vllm import protocol_options as PO
        from verity_vllm.protocols import pouw
        ctx = PO.Context(stage="serve", enabled=(f"pouw:{PH.SCHEME}",), settings={"pouw": {
            "coins": salt_coins, "runner": eng.runner, "rounds": rounds, "rows": decl.rows, "steps": decl.steps,
            "transcript_dir": str(out), "admission": RQ.ROUND0}})
        comp = PO.install(eng.model, ctx)
        h1 = pouw.h1_run(f"pouw:{PH.SCHEME}")
        h1._key = KEY
        release = comp.close
    else:
        holder = PHR.install(llm, rows=decl.rows, steps=decl.steps, coins=salt_coins, rounds=rounds, admission=RQ.ROUND0,
                             transcript_dir=str(out), key=KEY)
        h1, release = holder.run, holder.finish
    doc["install_s"] = round(time.monotonic() - t0, 1)
    try:
        t0 = time.monotonic()
        got = generate_static(llm, prompts, cap)
        doc["h1_serve"] = {"s": round(time.monotonic() - t0, 2), "tokens_digest": tokens_digest(got),
                           "static": static_check(got, cap)}
    finally:
        t0 = time.monotonic()
        release()
        doc["release_s"] = round(time.monotonic() - t0, 1)
    doc["same_tokens_in_run"] = [g["token_ids"] for g in got] == [g["token_ids"] for g in plain]
    doc["h1"] = summary(h1)
    doc["tokens"] = {"plain": [g["token_ids"] for g in plain], "h1": [g["token_ids"] for g in got]}
    (out / SUMMARY).write_text(json.dumps(doc, default=str) + "\n")
    print(f"{tag} h1 static={doc['h1']['static']} calls={doc['h1']['calls']} same_tokens={doc['same_tokens_in_run']} "
          f"step_rows={doc['h1']['step_rows']}", flush=True)
    return doc


#: what two windows of one layout must agree on in their commitments (the ranks' own record, ``ranks``, is the multiprocess
#: run's only)
COMMITTED = ("scheme", "tile", "rows", "steps", "u_step", "units", "calls", "admission", "weights_root", "coins",
             "salt_opening", "round_coins", "calls_root", "static", "draw")


def macs(shapes: list[list[int]]) -> int:
    return sum(n * k for n, k in shapes)


def compare(one: Path, other: Path) -> dict[str, Any]:
    """`one` (one GPU) and `other` (TP N × PP M) held to each other: tokens, calls, layouts and, at one TP, commitments."""
    from verity_vllm.protocols.pouw.h1 import CALLS_FILE
    a, b = (json.loads((d / SUMMARY).read_text()) for d in (one, other))
    ca, cb = (json.loads((d / CALLS_FILE).read_text()) for d in (one, other))
    tps = [a.get("tp", 1), b.get("tp", 1)]
    same_layout = tps[0] == tps[1]
    complete = [x["h1"]["calls"] == x["public"]["steps"] * len(x["h1"]["shapes"]) for x in (a, b)]
    out: dict[str, Any] = {
        "pp": [a["pp"], b["pp"]], "tp": tps, "same_layout": same_layout,
        "tokens": {"plain_equal": a["tokens"]["plain"] == b["tokens"]["plain"],
                   "h1_equal": a["tokens"]["h1"] == b["tokens"]["h1"],
                   "h1_is_plain": [a["same_tokens_in_run"], b["same_tokens_in_run"]],
                   "digests": [a["h1_serve"]["tokens_digest"], b["h1_serve"]["tokens_digest"]],
                   "plain_digests": [a["plain"]["tokens_digest"], b["plain"]["tokens_digest"]],
                   "requests_unequal": sum(x != y for x, y in zip(a["tokens"]["plain"], b["tokens"]["plain"]))},
        "calls": {"count": [a["h1"]["calls"], b["h1"]["calls"]], "linears": [len(a["h1"]["shapes"]), len(b["h1"]["shapes"])],
                  "static": [a["h1"]["static"], b["h1"]["static"]], "complete": complete,
                  "u_step": [a["h1"]["u_step"], b["h1"]["u_step"]],
                  "step_rows_equal": a["h1"]["step_rows"] == b["h1"]["step_rows"],
                  "macs_a_step": [macs(a["h1"]["shapes"]), macs(b["h1"]["shapes"])],
                  "refused": [len(a["h1"]["refused"]), len(b["h1"]["refused"])],
                  "order_ok": [a["h1"]["order_ok"], b["h1"]["order_ok"]]},
        "ranks": cb["commitment"].get("ranks"), "rank_details": b["h1"]["details"].get("ranks"),
        "pulls": b["h1"]["details"].get("pulls"),
        "seconds": {"plain": [a["plain"]["s"], b["plain"]["s"]], "h1": [a["h1_serve"]["s"], b["h1_serve"]["s"]]},
    }
    out["tokens_equal"] = bool(out["tokens"]["plain_equal"] and out["tokens"]["h1_equal"] and all(out["tokens"]["h1_is_plain"]))
    c = out["calls"]
    out["calls_equal"] = bool(all(c["static"]) and all(c["complete"]) and c["step_rows_equal"] and all(c["order_ok"])
                              and not any(c["refused"]) and c["macs_a_step"][0] == c["macs_a_step"][1])
    if same_layout:
        fields = ("index", "step", "layer", "module", "m", "k", "n", "base", "rows")
        calls_a, calls_b = ([{k: x.get(k) for k in fields} for x in y["calls"]] for y in (ca, cb))
        first = next((i for i, (x, y) in enumerate(zip(ca["calls"], cb["calls"])) if x != y), None)
        rounds_a, rounds_b = ca["commitment"]["rounds"], cb["commitment"]["rounds"]
        committed = {k: ca["commitment"].get(k) == cb["commitment"].get(k) for k in COMMITTED}
        c.update(equal=calls_a == calls_b, shapes_equal=ca["public"] == cb["public"])
        out["manifest"] = {"rows_equal": ca["calls"] == cb["calls"], "first_unequal": first,
                           **({"at_first": [ca["calls"][first], cb["calls"][first]]} if first is not None else {})}
        out["rounds"] = {"count": [len(rounds_a), len(rounds_b)], "equal": rounds_a == rounds_b}
        out["committed"] = committed
        out["calls_equal"] = bool(out["calls_equal"] and c["equal"] and c["shapes_equal"] and len(set(c["u_step"])) == 1)
        out["commitments_equal"] = bool(out["manifest"]["rows_equal"] and out["rounds"]["equal"] and all(committed.values()))
    else:
        out["rounds"] = {"count": [len(ca["commitment"]["rounds"]), len(cb["commitment"]["rounds"])]}
        out["commitments_equal"] = None
    out["ok"] = bool(out["tokens_equal"] and out["calls_equal"] and out["commitments_equal"] is not False)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="H1's fill window on one GPU and at TP N × PP M on one node, held to each other")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pp", type=int, default=1)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--async-scheduling", action="store_true", help="the multiprocess engine: vLLM's async scheduling (default off)")
    ap.add_argument("--compare", nargs=2, metavar=("ONE_DIR", "OTHER_DIR"), help="no GPU: hold the two runs to each other")
    ap.add_argument("--checkpoints", default="manifests/checkpoints.json")
    ap.add_argument("--base-role", default="LLAMA31_8B")
    ap.add_argument("--role", default="LLAMA31_8B_INT8")
    ap.add_argument("--repo", default="verity-int8/Llama-3.1-8B-INT8")
    ap.add_argument("--hf-home", default=os.environ.get("HF_HOME"))
    ap.add_argument("--at", help="resolved_at of the pin entry, if one is made (UTC, ISO 8601)")
    ap.add_argument("--by", help="resolved_by of the pin entry, if one is made")
    a = ap.parse_args(argv)
    if a.compare:
        doc = compare(Path(a.compare[0]), Path(a.compare[1]))
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(doc, indent=1, default=str) + "\n")
        print(json.dumps({k: doc[k] for k in ("ok", "tokens_equal", "calls_equal", "commitments_equal")}), flush=True)
        print(json.dumps(doc, default=str)[:6000], flush=True)
        return 0 if doc["ok"] else 1
    doc = serve(a)
    return 0 if doc["same_tokens_in_run"] and doc["h1"]["static"] else 1


if __name__ == "__main__":
    sys.exit(main())
