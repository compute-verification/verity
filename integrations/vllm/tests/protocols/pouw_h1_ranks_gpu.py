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
every round's root and coin, the calls root and the sample root equal. Across TPs the linears are shards, so it holds the
window's MACs a step equal instead (each linear's shard n·K summed over the ranks).

``--commit`` (TP N > 1, PP 1; ROOTS1 across ranks) serves the window with ``served`` and the Commit on every
rank (`pouw_h1_ranks.install`'s ``commit``) in place of the bare one, each window into ``<out>/<name>/``, and holds each to
core's ROOTS1 steps (`pouw_h1_gpu.core_tie`: one manifest under calls_root, each table committed by its round, each the
Commit's value of its rows). With ``--negatives`` each `NEGATIVES` window hands H1 other codes than the arm's on rank
``--edit-tp`` only (one call's all zero, or one row's first code off by one), its Commit honest, and must be refused at
``arow`` (`pouw_h1_gpu.negative_verdict`). ``--windows`` serves a subset, so a run splits its windows across GPU leases;
on a MoE (``--role OLMOE_INT8 --int8-exact linears``) it may name ``commit-a-zero-expert`` (`EXPERT_NEGATIVE`), the rank's
first routed expert down call with rows handed all-zero codes, and each window counts its expert calls (`expert_calls`)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
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


#: ``--negatives``' windows: H1 handed one call's A codes all zero, or one row's first code off by one, on one rank
NEGATIVES = ("a-zero", "a-row-off")
#: the negative a MoE adds (``--windows``): the rank's first routed expert down call with rows handed all-zero codes
#: (`pouw_h1_gpu.editor`'s)
EXPERT_NEGATIVE = "a-zero-expert"
#: every negative ``--windows`` may name
KINDS = (*NEGATIVES, EXPERT_NEGATIVE)
#: a routed expert's call in the window's names (`int8_experts.ExpertSite.name` at its rank, ``…experts.3.down@tp1``)
EXPERT_CALL = re.compile(r"\.\d+\.(gate_up|down)(@tp\d+)?$")


def _edit_of(kind: str) -> Callable[[Any], Any]:
    """The codes H1 is handed of the ones the arm quantized, on the edited rank: its first served call's all zero
    (``a-zero``), or row 1's first code off by one in its first call of two rows or more (``a-row-off``)."""
    import torch
    done = [False]

    def edit(xq: Any) -> Any:
        if done[0] or xq.shape[0] < (1 if kind == "a-zero" else 2):
            return xq
        done[0] = True
        if kind == "a-zero":
            return torch.zeros_like(xq)
        xq = xq.clone()
        xq[1, 0] = 1 if int(xq[1, 0]) != 1 else 2
        return xq
    return edit


def _w_edit(worker: Any, kind: str, tp: int) -> dict[str, Any] | None:
    """On tensor-parallel rank ``tp`` only (`pouw_h1_ranks.RankExtension._h1rank_rank`): each of its H1 linears'
    ``observe`` handed `_edit_of(kind)` of the arm's codes (`EXPERT_NEGATIVE`: `pouw_h1_gpu.editor`'s), outside H1's own
    (`pouw_h1_ranks.wire_rank`), so the Commit acquires the arm's codes and the served y is the arm's."""
    from tests.protocols import pouw_h1_gpu as GPU
    from verity_vllm.engine.hooks import Hooks
    if worker._h1rank_rank()[1] != tp:
        return None
    _model, layers = worker._h1rank_layers()
    if kind == EXPERT_NEGATIVE:
        edit = GPU.editor(GPU.EXPERT_NEGATIVE)
    else:
        codes = _edit_of(kind)
        edit = lambda _layer, xq: codes(xq)  # noqa: E731
        edit.at = None
    hooks = Hooks(tag=f"roots1-{kind}-tp{tp}")
    for m in layers:
        hooks.wrap(m.quant_method, "observe", lambda orig: GPU.edited(orig, edit))
    worker._h1rank_edit = (hooks, edit)
    return {"tp": tp, "linears": len(layers), "first_linear": worker._h1rank["run"].lo}


def _w_unedit(worker: Any) -> dict[str, Any] | None:
    """The rank's edit removed; returns the expert site it zeroed (``at``, its rank-local name), on the edited rank."""
    got = getattr(worker, "_h1rank_edit", None)
    if got is None:
        return None
    hooks, edit = got
    hooks.uninstall()
    worker._h1rank_edit = None
    return {"tp": worker._h1rank_rank()[1], "at": edit.at}


def expert_calls(h1: Any) -> dict[str, Any]:
    """The window's routed expert calls (`EXPERT_CALL`): how many, how many had rows, and on which ranks."""
    got = [c for c in h1.calls if EXPERT_CALL.search(c.module or "")]
    return {"calls": len(got), "with_rows": sum(bool(c.m) for c in got),
            "ranks": sorted({c.module.rpartition("@")[2] for c in got if "@" in c.module})}


def commit_window(a: argparse.Namespace, llm: Any, prompts: list[int], plain: list[dict[str, Any]], decl: Any, out: Path,
                  cp: dict[str, Any], program: dict[str, Any], name: str, edit: str | None = None) -> dict[str, Any]:
    """One window at TP N with ``served`` and the Commit on every rank (`pouw_h1_ranks.install`), ``edit`` (a `NEGATIVES`
    kind) on rank ``--edit-tp`` when given, held to core's ROOTS1 steps (`pouw_h1_gpu.core_tie`) once released. Its
    Program is declared at its first call and kept in ``program`` for the next window."""
    from tests.protocols import pouw_h1_gpu as GPU
    from verity_vllm.protocols.pouw.h1 import ranks as PHR
    from verity_vllm.protocols.pouw.h1 import requests as RQ
    from verity_vllm.protocols.pouw.h1 import served as SV
    tag, tdir = f"[tp{a.tp} {name}]", out / name
    tdir.mkdir(parents=True, exist_ok=True)
    served = SV.Served(window=os.urandom(64), **program)
    salt_coins, rounds = coins(decl.steps)
    t0 = time.monotonic()
    holder = PHR.install(llm, rows=decl.rows, steps=decl.steps, coins=salt_coins, rounds=rounds, admission=RQ.ROUND0,
                         transcript_dir=str(tdir), key=KEY, served=served,
                         commit={"cp": cp, "run_id": f"pouw-h1-ranks-{name}-{os.getpid()}"})
    h1, w = holder.run, {"install_s": round(time.monotonic() - t0, 1)}
    if edit is not None:
        w["edit"] = {"kind": edit, "ranks": [x for x in holder.executor.collective_rpc(
            _w_edit, kwargs={"kind": edit, "tp": a.edit_tp}) if x is not None]}
    print(f"{tag} installed {json.dumps(w)} host={GPU.host_memory()}", flush=True)
    try:
        t0 = time.monotonic()
        got = GPU.generate_static(llm, prompts, decl.max_tokens)
        w["serve_s"] = round(time.monotonic() - t0, 1)
    finally:
        if edit is not None:
            at = [x for x in holder.executor.collective_rpc(_w_unedit) if x is not None]
            w["edit"]["at"], w["edited"] = at, next((x["at"] for x in at if x["at"]), None)
        t0 = time.monotonic()
        holder.finish()
        w["release_s"] = round(time.monotonic() - t0, 1)
    program.update(program=served.program, program_digest=served.program_digest)
    tap = served.tap
    w.update(same_tokens=[g["token_ids"] for g in got] == [g["token_ids"] for g in plain],
             tokens_digest=GPU.tokens_digest(got), static_schedule=GPU.static_check(got, decl.max_tokens),
             h1=summary(h1), experts=expert_calls(h1), roots1=h1.details.get("roots1"), committed=holder.committed,
             served={"refused": served.refused[:8], "registered": len(served.registered)},
             arow_sites={v: sorted({s.rpartition("@")[2].partition("/")[0] for s in at.get("sites") or ()})
                         for v, at in sorted(tap.where.items()) if "sites" in at},
             tap_refused=tap.refused[:8])
    w["core"] = GPU.core_tie({"commitment": dict(h1.commits), "calls": h1.manifest()}, served.program, tdir)
    w["host"] = GPU.host_memory()
    tied = bool(w["core"]["tied"] and (w["roots1"] or {}).get("tied"))
    w["ok"] = bool(w["same_tokens"] and w["h1"]["static"] and w["static_schedule"]["ok"] and not w["served"]["refused"]
                   and all(not c.get("errors_n") for c in holder.committed.values())
                   and (tied if edit is None else GPU.negative_verdict(w)["refused"] and bool(w.get("edit", {}).get("ranks"))
                        and (edit != EXPERT_NEGATIVE or bool(w.get("edited")))))
    print(f"{tag} ok={w['ok']} serve_s={w['serve_s']} release_s={w['release_s']} roots1={json.dumps(w['roots1'])[:600]} "
          f"core={json.dumps(w['core'], default=str)[:800]} committed={json.dumps(w['committed'])[:600]} host={w['host']}",
          flush=True)
    return w


def serve(a: argparse.Namespace) -> dict[str, Any]:
    import torch
    from tests.engine.int8_exact_gpu import pin, scratch_checkpoints
    from tests.protocols.pouw_h1_gpu import declared_of, fill_prompts, generate_static, static_check, tokens_digest
    from verity_vllm.program.frontend.target_profile import TargetProfile
    from verity_vllm.protocols.pouw import h1 as PH
    from verity_vllm.protocols.pouw.h1 import requests as RQ

    decl, out = declared_of(a), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    target = TargetProfile.from_device().with_overrides(int8_exact=a.int8_exact)
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
            kw, _env, cp = engine_kwargs_for(cps, max_num_seqs=max_num_seqs, case=a.role, target=target, engine_args={
                **args, "pipeline_parallel_size": a.pp, "tensor_parallel_size": a.tp, "distributed_executor_backend": "mp",
                "worker_extension_cls": PHR.COMMIT_EXTENSION if a.commit else PHR.EXTENSION, "async_scheduling": a.async_scheduling})
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
    if a.commit:
        import gc
        program: dict[str, Any] = {}
        doc["windows"] = {}
        for name in a.windows:
            edit = name.removeprefix("commit-") if name != "commit" else None
            doc["windows"][name] = commit_window(a, llm, prompts, plain, decl, out, {"repo": cp.get("repo"),
                                                 "revision": cp.get("revision")}, program, name, edit)
            gc.collect()
            (out / SUMMARY).write_text(json.dumps(doc, default=str) + "\n")
        doc["ok"] = all(w["ok"] for w in doc["windows"].values())
        (out / SUMMARY).write_text(json.dumps(doc, default=str) + "\n")
        print(f"POUW-H1-RANKS-COMMIT {'OK' if doc['ok'] else 'FAIL'} tp={a.tp} " + " ".join(
            f"{k}={'ok' if w['ok'] else 'FAIL'}" for k, w in doc["windows"].items()), flush=True)
        return doc
    salt_coins, rounds = coins(decl.steps)
    t0 = time.monotonic()
    if a.pp == a.tp == 1:
        from verity_vllm import protocols as PO
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
    ap.add_argument("--declare", metavar="SLOTS,CAP,T,S", type=lambda v: tuple(int(x) for x in v.split(",")),
                    help="serve SLOTS one-token prompts at cap CAP, T rows a call, S steps (pouw_h1.Static) in place of "
                         "pouw_h1.FILL's 64,16,64,18")
    ap.add_argument("--commit", action="store_true",
                    help="at TP N > 1: in place of the bare window, serve it with served (pouw_h1_served.Served) and the "
                         "Commit on every rank (pouw_h1_ranks.install's commit), each window into <out>/<name>/, and hold "
                         "each to core's ROOTS1 steps (pouw_h1_gpu.core_tie)")
    ap.add_argument("--negatives", action="store_true",
                    help="with --commit: after the honest window, one per NEGATIVES kind on rank --edit-tp, each to be "
                         "refused at arow")
    ap.add_argument("--windows", type=lambda v: v.split(","), default=None,
                    help="with --commit: the windows to serve, of commit and commit-<NEGATIVES kind> (default: commit, and "
                         "with --negatives each negative)")
    ap.add_argument("--edit-tp", type=int, default=1, help="the tensor-parallel rank a negative's edit is on")
    ap.add_argument("--int8-exact", default="all",
                    help="the int8_exact row's parts (TargetProfile; a MoE such as OLMOE_INT8 serves linears)")
    a = ap.parse_args(argv)
    if a.windows is None:
        a.windows = ["commit", *(f"commit-{k}" for k in NEGATIVES if a.negatives)]
    if a.commit and (a.tp < 2 or a.pp != 1 or any(w != "commit" and w.removeprefix("commit-") not in KINDS
                                                   for w in a.windows)):
        ap.error("--commit is at TP N > 1 and PP 1 (one GPU's is pouw_h1_gpu --commit), its windows commit and "
                 f"commit-<{'|'.join(KINDS)}>")
    if a.compare:
        doc = compare(Path(a.compare[0]), Path(a.compare[1]))
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(doc, indent=1, default=str) + "\n")
        print(json.dumps({k: doc[k] for k in ("ok", "tokens_equal", "calls_equal", "commitments_equal")}), flush=True)
        print(json.dumps(doc, default=str)[:6000], flush=True)
        return 0 if doc["ok"] else 1
    doc = serve(a)
    if a.commit:
        return 0 if doc["ok"] else 1
    return 0 if doc["same_tokens_in_run"] and doc["h1"]["static"] else 1


if __name__ == "__main__":
    sys.exit(main())
