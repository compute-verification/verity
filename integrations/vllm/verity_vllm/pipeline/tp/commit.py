"""Rank-aware Commit + overhead at `--tp 2`: `commit_delta`'s pairs protocol with the collector installed in
EVERY rank through the `worker_extension_cls` seam (`tp2_worker.tp2_commit_*`).

    ... tp2_commit --workload <manifest> --case LLAMA32_1B --tp 2 --committer native_collect_v2b --hidden-so <so> --openings 64 --pairs 3 \
        --max-num-seqs B --engine-arg gpu_memory_utilization=0.3 --label <row> --out <row>/commit

Run commitment over the ranks: every rank commits under rows-sha512 (`commit.step_rows`: hm96-sha512 hiding leaves over the
rank's `chunk-leaf-v1` step stream, frame-v3-sha512 trees, 64-byte roots), and (`tp.run_root`)
    tp_run_root = SHA-512("verity/tp-run-root/v2" || u32 world || (u32 rank_r || root_r) for r in 0..world-1)
i.e. per-rank run roots combined deterministically, rank-tagged, in rank order — the option "per-rank roots combined into the run
root" of the charter.  ``--commit-scheme`` takes rows-sha512 alone (`rows_only`); vllm-v1's 32-byte ``verity/tp-run-root/v1``
roots are only reread from recorded runs.  Every rank's root is fixed (finalize on every rank returns) BEFORE the pair's coins
are drawn (`pipeline.commit_coins`: with ``--challenger-gate``, the auditor host's challenger opens them after the TP run root;
without it they are this process's own, a measurement); the challenge of rank r is `challenge_positions(tp_run_root,
leaves_per_step_r, k, salt=pair/rank r)` from the pair's positions coin, and its population openings come from the identities coin,
so no rank's values can be recomputed after the challenge.  FAIL-CLOSED: a rank that does not return a root (fewer
results than `world` from the collective RPC, or an exception) fails the run; required classes are checked PER RANK; the
per-rank X-03 verdict (`commit_verdict`) must PASS on every rank; control vs instrumented tokens (the engine's = rank 0's sampler)
must be equal.  Output: `runs.jsonl` (one row per arm per pair: wall, per-rank commit blocks, the combined root, verdict) and
`summary.json` (delta wall, roots, per-rank coverage counts).  FAIL-FAST: the Commit stops after its first failed run (a rank's verdict
or unequal tokens), FAIL, with the GPUs back at once (`lease.stop`) and `summary.stopped` naming the runs not run, the value check's
among them; `--all-pairs` runs every one (timing studies).
"""

from __future__ import annotations

import gc
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from verity_vllm import settings
from verity_vllm.acquire import as_loaded as AL
from verity_vllm.acquire.sources import taps as TAPS
from verity_vllm.check import weights_of_record as WR
from verity_vllm.check.component_programs import post_load_tensors
from verity_vllm.config import UsageError, group, option, workload_arg
from verity_vllm.engine import served as SV
from verity_vllm.pipeline import commit_coins as CC, replay_bundle as RB
from verity_vllm.pipeline.tp import c2 as C2
from verity_vllm.pipeline.tp import config_run as TPC
from verity_vllm.pipeline.tp import phases as PH
from verity_vllm.pipeline.tp import serving_rows as SRT
from verity_vllm.pipeline.tp import warmup as WU
from verity_vllm.pipeline.tp.import_path import write_import_path_runtime
from verity_vllm.pipeline.tp.negatives import negatives_applied, parse_serve_shard_from
from verity_vllm.commit import step_rows
from verity_vllm.pipeline.tp import run_root as RT
from verity_vllm.records.commit_run import TP_SCHEMA as SCHEMA


def tree_of_record(root: str | os.PathLike | None = None) -> dict:
    """The executing tree BY CONTENT beside git_head: `source_identity.code_identity` of the repository this module runs from (`root`
    overrides it) -- its import closure, the integration plus Verity core, the same identity the hot Commit worker keys on -- so a dirty
    tree cannot pass as the tagged commit by git_head alone."""
    from verity_vllm.pipeline import research_tools, source_identity
    root = Path(root) if root is not None else source_identity.TREE
    return {"source": "source_identity.code_identity", "root": str(root), "closure": list(research_tools.CLOSURE), **source_identity.code_identity(root)}


def weights_as_loaded(pin: dict | None) -> partial | None:
    """What `tp2_commit_make` calls on every rank for the weights it registers (`acquire.as_loaded.attach_rank`), composing from the
    checkpoint of record; None without one."""
    if not pin:
        return None
    return partial(AL.attach_rank, composer=partial(WR.rank_composer, checkpoint_dir=pin["checkpoint_dir"],
                                                    checkpoints_manifest=pin["checkpoints_manifest"], model=pin["model"]))


def schedule(pairs: int, only_arm: str | None) -> list[tuple[int, str]]:
    """The Commit's measured runs in order: per pair (control, instrumented), reversed on odd pairs; `--only-arm` keeps that arm."""
    return [(p, a) for p in range(pairs) for a in (("control", "instrumented") if p % 2 == 0 else ("instrumented", "control")) if only_arm in (None, a)]


def ring_slots(committer: str, window_mb: str | None, window_slots: str | None, retain: str | None) -> int | None:
    """The ranks' window ring slots as their committers read VERITY_WINDOW_MB / VERITY_WINDOW_SLOTS / VERITY_RETAIN (an unset window
    is `native_collect.window_mb_default`), None when they stage in per-step blocks."""
    from verity_vllm.commit.committer.native_collect import WINDOW_SLOTS, window_mb_default

    window = (window_mb or "").strip() or window_mb_default(committer == "native_collect_v2b", retain or "device")
    return int(window_slots or WINDOW_SLOTS) if float(window) > 0 else None


def build_program_digest(d: str | None) -> tuple[str | None, str | None]:
    """A rank's Program digest FROM ITS BUILD dir ``d`` (never from the manifest), and which Program it is: a one-request row = the
    request Program's digest (result.json); a B>=2 row = the rank's GP-01 workload Program digest (build_workload/workload_program.json)
    when the compose ran, else the labelled components fold of the rank's request Programs (`build_request_LP*_T*/result.json`,
    required_manifest's rule).  (None, kind) when the Build's record is unreadable."""
    if not d:
        return None, None
    P, kind = Path(d), None
    try:
        wp = P / "workload_program.json" if (P / "workload_program.json").exists() else P / "build_workload" / "workload_program.json"
        if wp.exists():
            w = json.loads(wp.read_text())
            if w.get("workload_digest"):
                return str(w["workload_digest"]), "workload (GP-01)"
        comps = sorted(P.glob("build_request_LP*_T*/result.json"))
        if comps:
            from verity_vllm.query.manifest.format import components_fold_digest
            kind = f"components-fold of {len(comps)} request Programs"
            return components_fold_digest(json.loads(c.read_text())["program"]["digest"] for c in comps), kind
        kind = "request Program"
        return json.loads((P / "result.json").read_text())["program"]["digest"], kind
    except Exception:  # noqa: BLE001
        return None, kind


def _build_dirs(args) -> dict[int, str]:
    """rank -> Build dir from `--build-rank <dir>` repeated in rank order (world N) or the two-rank `--build-rank0/1` spelling."""
    if args.build_rank:
        if args.build_rank0 or args.build_rank1:
            raise SystemExit("--build-rank (rank order) and --build-rank0/1 are alternative spellings; give one")
        if len(args.build_rank) != int(args.tp):
            raise SystemExit(f"--build-rank given {len(args.build_rank)} time(s) for a world of {args.tp}: one Build dir per rank, in rank order")
        return {r: str(d) for r, d in enumerate(args.build_rank)}
    return {r: d for r, d in ((0, args.build_rank0), (1, args.build_rank1)) if d}


def rows_only(scheme: str) -> str:
    """``--commit-scheme``'s ``type``: rows-sha512, the one scheme a TP Commit commits under."""
    if scheme != step_rows.NAME:
        raise UsageError(f"{scheme!r}: a TP Commit commits every rank under {step_rows.NAME} (64-byte rank roots, {RT.TP_ROOT_TAG.decode()}); "
                         "vllm-v1's SHA-256 roots are only reread from recorded runs")
    return scheme


@dataclass(kw_only=True)
class Options:
    workload: str = option("--workload", type=workload_arg)
    out: str = option("--out")
    checkpoints: str = option("--checkpoints", default="manifests/checkpoints.json")
    case: str | None = option("--case", default=None)
    tp: int = option("--tp", type=int, default=2, help="communicator size")
    committer: str = option("--committer", default="native_collect_v2b")
    commit_scheme: str = option("--commit-scheme", default=step_rows.NAME, type=rows_only,
                                help="every rank's commitment scheme: rows-sha512, the only one (the windowed worker, native_collect_v2b "
                                     "with --window-mb > 0), under the SHA-512 TP run root")
    hidden_so: str | None = option("--hidden-so", default=None)
    hidden_ns: str | None = option("--hidden-ns", default=None)
    openings: int = option("--openings", type=int, default=64)
    pairs: int = option("--pairs", type=int, default=3)
    warmup: int = option("--warmup", type=int, default=2)
    max_num_seqs: int = option("--max-num-seqs", type=int, default=8)
    engine_arg: list[str] = option("--engine-arg", action="append", default=[])
    arena_mb: int | None = option("--arena-mb", type=int, default=None)
    label: str | None = option("--label", default=None)
    required_classes: str | None = option("--required-classes", default=None)
    rpc_timeout: float = option("--rpc-timeout", type=float, default=3600.0)
    build_rank0: str | None = option("--build-rank0", default=None, help="derive_step out dir of rank 0's Program (result.json) -> program_digest for the binding map / population")
    build_rank1: str | None = option("--build-rank1", default=None)
    build_rank: list[str] = option("--build-rank", action="append", default=[], help="rank r's Build dir, repeated in RANK order (world = --tp); the world-N spelling of --build-rank0/1")
    world: int | None = option("--world", type=int, default=None, help="alias of --tp (the communicator size)")
    no_tp_partials: bool = option("--no-tp-partials", default=False, action="store_true", help="NEGATIVE: collect collective outputs only; the required class tp_rank_partials must fail the verdict")
    omit_rank: int | None = option("--omit-rank", type=int, default=None, help="NEGATIVE (brief §4 'omit one TP rank'): drop this rank's entry from the commitment vector before the verdict; must fail closed")
    tap_sources: TAPS.TapSources = group(TAPS.TapSources)
    required_manifest: str | None = option("--required-manifest", default=None, help="the merged TP manifest (verity_vllm.pipeline.cli build-global --programs-root <rank r dir> --rank r ... --ranks W): the population of record per rank (rows with rank=r)")
    relabel_rank: str | None = option("--relabel-rank", default=None, help="NEGATIVE ('relabel rank 0 as rank 1'): `<from>:<to>` -- rank <from> tags its partials part_rank<to>; coverage must FAIL naming rank <from>")
    wrong_root_rank: int | None = option("--wrong-root-rank", type=int, default=None, help="NEGATIVE ('wrong root on one rank'): the commitment vector carries a wrong run root for this rank; its openings/context must FAIL")
    xrank_per_stratum: int = option("--xrank-per-stratum", type=int, default=2, help="cross-rank collective-output pass: picks per (site, kind, phase) stratum, seeded by the pair's combined root; " "each pick evaluates the committed received output against the definition over the two ranks' COMMITTED partials (all_reduce: bf16(f32+f32); " "all_gather: concat) on a row subset; a FAIL names the site; 0 disables (component result None)")
    flip_partial: str | None = option("--flip-partial", default=None, help="NEGATIVE: `<rank>:<module suffix>[:ulp]` -- that rank commits its genuine partial at that ONE site but feeds the collective a different " "tensor (element 0: 2x+1024, or with `:ulp` ONE bf16 ulp -- the discriminating case where tokens stay equal and only the cross-rank pass can name it); " "every per-rank record stays self-consistent; the cross-rank collective-output pass must FAIL naming the site")
    omit_partial_site: str | None = option("--omit-partial-site", default=None, help="NEGATIVE: `<rank>:<module suffix>` -- that rank drops the partial of ONE collective site only (e.g. 1:layers.3.mlp.down_proj); coverage must FAIL naming the rank and the site")
    omit_rank_partials: int | None = option("--omit-rank-partials", type=int, default=None, help="NEGATIVE ('omit rank 1's partials'): this rank collects collective outputs only; coverage must FAIL naming this rank")
    value_check: bool = option("--value-check", default=False, action="store_true", help="one extra instrumented run per rank: committed bytes vs hook-time reference clones + re-execution (module outputs and TP partials)")
    match_dir: str | None = option("--match-dir", default=None, help="tp2_match run dir (<dir>/capture/tokens.json = the Match record's per-request generated ids): per rank, " "every committed sampled-ids row is attributed to its request by the rank's request->rows map and compared to the record; a mismatch FAILS the row")
    swap_requests: str | None = option("--swap-requests", default=None, help="NEGATIVE: `<rank>:<ra>:<rb>[:<lo>-<hi>]` -- that rank labels request ra's rows as rb and rb's as ra (equal-count requests keep " "the identity counts; with `:<lo>-<hi>` the swap applies ONLY on engine steps lo..hi = the decode-only swap: prefill rows untouched, each swapped row one token, " "coverage/composition blind -- the per-(step, request) attribution alone must FAIL naming the rank)")
    no_match_oracle: bool = option("--no-match-oracle", default=False, action="store_true", help="with --match-dir: skip the per-rank compare of every committed TP partial / MoE received output against " "the capture recorder's stored collectives (TP family); default = run it and gate on it")
    require_fold_match: bool = option("--require-fold-match", default=False, action="store_true", help="the --match-dir MUST carry the per-rank fold Match (fold_match/tp2_fold_match.json + " "tp2_rank_match.json): its peer_binding digests are bound to the recorder rows the match oracle compared the committed partials against, " "both ranks, every (step, k); absence or any mismatch FAILs the row by name (default: absent = component None, Match collective-level ◐)")
    no_sampled_replay: bool = option("--no-sampled-replay", default=False, action="store_true", help="skip the per-rank sampled exact replay + boundary linkage (default with " "--required-manifest: run it on every rank and gate on it -- O1: absence / partial / mismatch = FAIL named)")
    replay_per_stratum: int = option("--replay-per-stratum", type=int, default=2)
    replay_max: int = option("--replay-max", type=int, default=2000, help="cap on the stratified sample per rank (never below one VU per stratum)")
    replay_seed: int | None = option("--replay-seed", type=int, default=None, help="targeted seed (default: each rank's run root after commitment, O6)")
    replay_workers: int | None = option("--replay-workers", type=int, default=None)
    replay_ranks_at_once: int | None = option("--replay-ranks-at-once", type=int, default=None, metavar="K", help="at most K ranks run their sampled replay at once, each giving its heap back before the next (`pipeline.tp.rank_replay.host_slot`; a rank holds its request Programs while it replays); default: every rank at once")
    replay_deferred: str | None = option("--replay-deferred", nargs="?", const="full", default=None, choices=("full",),
                                         help="each rank's C2 sampled replay + boundary linkage do not run in its process: every rank of every instrumented run "
                                              "seals a replay bundle (<out>/replay_bundle_p<pair>_rank<r>/, verity_vllm.pipeline.replay_bundle) and the Commit's "
                                              "verdict stays PENDING (exit 0 when every check this process runs passes) until `verity-vllm tp-commit-replay <out>` "
                                              "replays the bundles on CPU against the checkpoint and decides it as this process would have (TP2 and TP8 only: `c2.deferral_skipped`)")
    checkpoint_dir: str | None = option("--checkpoint-dir", default=None, help="the PINNED checkpoint snapshot (safetensors shards + config.json): each rank " "builds its weights of record from its rank Program's shard map (engine-free) and compares every served shard's sha256 " "to it right after register_weights; a difference FAILs named (rank, param). Required with --required-manifest unless " "--no-weights-pin (then the record says 'weights unpinned (MAN-09)' and the Commit FAILs closed)")
    checkpoints_manifest: str | None = option("--checkpoints-manifest", default=None, help="manifests/checkpoints.json (default: the repo's) -- the shard sha256s of record")
    checkpoint_model: str | None = option("--checkpoint-model", default=None, help="HF repo id selecting the checkpoints.json entry when --checkpoint-dir is not its local_path")
    no_weights_pin: bool = option("--no-weights-pin", default=False, action="store_true", help="do not require the weights pin (the record stays 'weights unpinned (MAN-09)'; verdict FAIL closed)")
    serve_shard_from: list[str] | None = option("--serve-shard-from", action="append", default=None, help="NEGATIVE ('TP shard / reference mismatch'): <rank>:<served param name>:<from_rank> -- rank <rank> serves " "rank <from_rank>'s shard of that parameter, composed from the SAME pinned checkpoint (every byte is checkpoint bytes; the shard set over the " "world is unchanged; only the rank<->slice assignment the rank Program's shard map prescribes is wrong -- a whole-tensor hash cannot see it). " "The per-rank weights pin must FAIL naming the rank and the parameter. Repeatable: 0:X:1 1:X:0 = the two ranks swap their shards of X.")
    flip_weight: str | None = option("--flip-weight", default=None, help="NEGATIVE (flipweight, TP form): <rank>:<served param name>:<flat index>[:<value>] -- one element of that " "rank's shard changed before register_weights; the weights pin must FAIL naming (rank, param)")
    replay_challenge: str | None = option("--replay-challenge", default=None, help="verifier-issued POST-COMMITMENT challenge (hex/string): each rank's replay " "seed = sha256(challenge | that rank's run root)[:8]; without it the seed is the rank's run root (honest-prover measurement)")
    challenger_gate: str | None = option("--challenger-gate", default=None, metavar="CMD", help="the auditor host's challenger gate, its command as one string: the Commit reserves one session before its engine and opens it on its first instrumented pair's TP run root, and each pair's opening positions and population identities are its coins, opened after that pair's TP run root (verity_vllm.pipeline.commit_coins); without it each pair's coins are this process's own, a measurement that credits nothing")
    challenger_peer: str | None = option("--challenger-peer", default=None, metavar="NAME", help="with --challenger-gate: this machine's name in the auditor host's peers")
    replay_target: list[str] | None = option("--replay-target", action="append", default=None, help="<op_path>[@<step>]: add every VU of a module to every rank's sample")
    replay_k: int | None = option("--replay-k", type=int, default=None, help="the config run's sampled replay: k units over the world, split across the ranks (rank r: k // W, one more for r < k %% W), each rank's share stratified by family, in place of the per-stratum sample")
    only_arm: str | None = option("--only-arm", choices=["control", "instrumented"], default=None, help="run only this arm (`--pairs` runs of it; the config run: one instrumented run)")
    all_pairs: bool = option("--all-pairs", default=False, action="store_true", help="run every pair and --value-check's run after a failed run (timing studies); "
                             "default: the Commit stops after the first failed run, FAIL, and summary.json's `stopped` names the runs not run")
    trace_dir: str | None = option("--trace-dir", default=None, help="after the timed pairs, one more pair under torch.profiler on every rank, outside "
                                   "runs.jsonl and the summary's timing (verity_vllm.pipeline.tp.trace)")
    trace_ranks: str = option("--trace-ranks", default="0", help="with --trace-dir: the ranks whose Kineto trace is kept")
    serving_rows: str | None = option("--serving-rows", default=None, metavar="REQUEST_JSON", help="opt-in, not on the record: in the first instrumented run, once every rank's root is final and the TP run root fixed and before the challenge, each rank commits its units of the verifier's partition in M0's serving row format (verity_vllm.pipeline.tp.serving_rows; the row stage writes the request); needs --required-manifest and --replay-deferred")
    memory_map: str | None = option("--memory-map", default=None, metavar="REQUEST_JSON", help="opt-in, not on the record: each rank commits its KV layers' `<layer>/slot_mapping` (acquire.sources.kv_slots_source) and, in its replay (in process or before its deferred bundle), writes memory-map/v1 of its KV to the request's path at `.rank<r>` (or the path `ranks[r]` names), its blocks from its request Programs of record at the pages those slots name, its digests from its committed KV words (verity_vllm.pipeline.memory_map, pipeline.tp.rank_replay.memory_map); the prover's witness, never published; needs --required-manifest with a --build-rank per rank")


def main(args: Options) -> int:
    from verity_vllm.engine import engine_profile as prof, env as engine_env
    prof.apply_env()
    from verity_vllm.pipeline.tp import lease as TPL, match as TPM, target as TT, trace as TRACE
    from vllm import LLM
    from verity_vllm.engine import vllm_adapter as va
    from verity_vllm.pipeline.commit import git_head, run_workload
    from verity_vllm.check.verdict import REQUIRED_CLASSES_DEFAULT, commit_verdict
    from verity_vllm.pipeline.tp.capture import parse_engine_args
    from verity_vllm.pipeline.tp.rank_worker import EXTENSION
    from verity_vllm.commit.challenge import challenge_positions
    from verity_vllm.commit.partition import require_partition
    from verity_vllm import protocols as PO
    from verity_vllm.acquire.sources.oracle_witness import SUMMARY_KEYS as TPS_SUMMARY_KEYS
    if args.world is not None:
        args.tp = int(args.world)
    if args.replay_deferred and (not args.required_manifest or args.no_sampled_replay):
        raise SystemExit("--replay-deferred defers the ranks' sampled replay: it needs --required-manifest and no --no-sampled-replay")
    deferral_skipped = C2.deferral_skipped(args)
    relabel = None
    if args.relabel_rank:
        a, b = args.relabel_rank.split(":")
        relabel = {str(int(a)): int(b)}
    swap_requests = None
    if args.swap_requests:
        parts = args.swap_requests.split(":")
        sr, sa, sb = parts[:3]
        swap_requests = {str(int(sr)): [sa, sb]}
        if len(parts) > 3:     # `<lo>-<hi>`: swap ONLY on engine steps lo..hi (decode-only swap; prefill rows untouched)
            lo, hi = (int(x) for x in parts[3].split("-"))
            swap_requests[str(int(sr))] += [lo, hi]
    required_classes = tuple(c for c in (args.required_classes.split(",") if args.required_classes is not None else REQUIRED_CLASSES_DEFAULT) if c)
    if args.tp > 1 and args.required_classes is None and "tp_rank_partials" not in required_classes:
        required_classes = required_classes + ("tp_rank_partials",)       # the rank-local collective inputs are required per rank

    (out := Path(args.out)).mkdir(parents=True, exist_ok=True)
    wl = va.load_workload(args.workload, None)
    coins = CC.of_options(args, sum(a != "control" for _p, a in schedule(args.pairs, args.only_arm)))    # before any root (None: a measurement)
    TPL.acquire()             # a leased Commit's GPUs, all at once, to its last run's finalize (pipeline.tp.lease); the target reads the device
    target = PO.commit_guard(va.target_of_workload(wl), world=args.tp, installs=False)      # exit 3 when the target declares protocols
    target_env, target_kw = TT.apply(target, wl["execution"])    # as `tp-capture` applies it, and every rank's engine checked against it (`tp.target`)
    case = va.case_of(wl, args.case)
    env_flags = {**prof.apply_env(), **target_env}
    engine_env.step_timeout(args.rpc_timeout)
    cp = prof.checkpoint_for(case, args.checkpoints)
    kw = prof.engine_kwargs(cp, replay=False, seed=prof.ENGINE_SEED, enforce_eager=True, tensor_parallel_size=args.tp)
    kw.update({"enable_prefix_caching": False, "max_num_seqs": int(args.max_num_seqs), **target_kw})
    kw.update({k: v for k, v in (wl.get("engine_args_required") or {}).items() if k not in ("tensor_parallel_size",)})
    kw.update(parse_engine_args(args.engine_arg), worker_extension_cls=EXTENSION)
    print(f"[tp2commit] tp={args.tp} case={case} committer={args.committer} kwargs={ {k: v for k, v in kw.items() if k not in ('model', 'tokenizer')} }", flush=True)
    llm, build_s, topo = PH.engine(lambda: LLM(**kw), args.tp, args.rpc_timeout, model=cp["repo"], enforce_eager=kw.get("enforce_eager"), case=case, max_num_seqs=args.max_num_seqs)
    TT.require_ranks(llm, target, args.rpc_timeout)
    served = SV.require(llm, wl, target, out, args.rpc_timeout)
    eng = va.Engine(llm=llm, runner=None, model=None, cp=cp, kwargs=kw, env_flags=env_flags, build_s=build_s, case=case, target=target)
    world = len(topo)
    versions = {"schema": SCHEMA, "git_head": git_head(), "tree": tree_of_record(), "argv": sys.argv, "tp": args.tp, "build_s": round(build_s, 1),
                "engine_kwargs": {k: v for k, v in kw.items() if k not in ("model", "tokenizer")},
                "effective_execution": va.effective_execution_doc(llm), "target": TT.as_json(target), "served": served,
                "env": settings.redact({k: v for k, v in os.environ.items() if k.startswith(("NCCL_", "VLLM_", "CUBLAS", "NATIVE_COLLECT", "HIDDEN_GPU"))}),
                "env_required": {k: v for k, v in engine_env.snapshot().items() if k == "NCCL_P2P_DISABLE"}, "ranks": topo,
                "root_combination": "tp_run_root = SHA-512(tag || u32 world || (u32 rank || root_rank)...)",
                "root_tag": RT.TP_ROOT_TAG.decode(), "commit_scheme": args.commit_scheme,
                "layout_version_per_rank": f"chunk-leaf-v1 step streams under {args.commit_scheme}", "required_classes": list(required_classes),
                "required_classes_source": "cli" if args.required_classes is not None else "default (REL-02) + tp_rank_partials (R14 TP-03), checked per rank",
                "negatives": {"omit_rank": args.omit_rank, "no_tp_partials": bool(args.no_tp_partials), "relabel_rank": args.relabel_rank, "swap_requests": args.swap_requests,
                              "wrong_root_rank": args.wrong_root_rank, "omit_rank_partials": args.omit_rank_partials, "omit_partial_site": args.omit_partial_site,
                              "flip_partial": args.flip_partial},
                "required_manifest": args.required_manifest, "value_check": bool(args.value_check), "rules_version": "r15-tp04-v1",
                "build_program_digest_kind": None}
    (out / "versions.json").write_text(json.dumps(versions, indent=1, default=str) + "\n")
    reqs, sampling = wl["requests"], wl["sampling"]
    PH.warmups(args.warmup, lambda i: run_workload(eng, va.throwaway_requests(eng, reqs, seed=100 + i), sampling))
    envelope = {"batch": args.max_num_seqs, "max_prompt": max(len(r["prompt_token_ids"]) for r in reqs),
                "max_new": max(int(r.get("max_tokens") or sampling.get("max_tokens") or 1) for r in reqs),
                "max_num_batched_tokens": kw.get("max_num_batched_tokens"), "max_model_len": kw.get("max_model_len")}
    builds = _build_dirs(args)
    built = {r: build_program_digest(builds.get(r)) for r in range(world)}
    pds = {r: d for r, (d, _k) in built.items()}
    versions["build_program_digest_kind"] = {str(r): k for r, (_d, k) in built.items() if k}
    versions["build_program_digest"] = {str(k): v for k, v in pds.items()}
    (out / "versions.json").write_text(json.dumps(versions, indent=1, default=str) + "\n")
    man_doc = require_partition(json.loads(Path(args.required_manifest).read_text()), None, None) if args.required_manifest else None
    if man_doc is not None:
        pdr = man_doc.get("program_digest_per_rank") or {}
        for r in range(world):
            if pds.get(r) is None:
                # the manifest names each rank's Program, but the manifest is an INPUT: without this rank's Build the
                # wrong-Program check (manifest of another Program that is internally consistent) would be vacuous -> refuse, never default
                raise SystemExit(f"fail-closed: --required-manifest needs --build-rank{r} (this rank's Build result.json) so the manifest's Program "
                                 f"for rank {r} ({str(pdr.get(str(r)))[:16]}) is checked against the Build, not against itself")
            elif pdr.get(str(r)) not in (None, pds[r]):
                raise SystemExit(f"fail-closed: --build-rank{r} Program {pds[r][:16]} != manifest Program for rank {r} {str(pdr.get(str(r)))[:16]}")
        if int(man_doc.get("ranks") or 1) != world:
            raise SystemExit(f"fail-closed: manifest ranks={man_doc.get('ranks')} != tp={world}")
    # workload -> request order / arrivals / caps (B>1 population; commit_delta's convention: r<i> in workload order)
    request_order = [str(r.get("request_id")) for r in reqs]
    # request ids are r<index>: the workload's declared `index` when present, else declaration position -- the same
    # rule commit_delta applies, so the manifest's r<k> rows and the rank's request->rows map name the same request
    request_ids = {str(r.get("request_id")): f"r{int(r.get('index', i))}" for i, r in enumerate(reqs)}
    arrivals = {f"r{int(r.get('index', i))}": int(r.get("arrive_step") or 0) for i, r in enumerate(reqs)}
    caps = {f"r{int(r.get('index', i))}": int(r.get("max_tokens") or sampling.get("max_tokens") or 1) for i, r in enumerate(reqs)}
    match_tokens, match_rec = TPM.match_record(args.match_dir, request_ids, world) if args.match_dir else (None, None)
    versions["match_record"] = match_rec
    match_capture_dir = (str(Path(args.match_dir) / "capture") if (args.match_dir and not args.no_match_oracle) else None)
    # the sampled exact replay's Programs OF RECORD per rank: the manifest's per-rank Program
    # (B1: the request Program itself) and, at B>=2, the component digests the GP-01 workload_program of that rank composed (its workload
    # digest is the manifest's per-rank digest, checked above) -- the worker refuses any loaded Program outside this set BY NAME
    replay_cfg = None
    retain_env = os.environ.get("VERITY_RETAIN")
    if args.required_manifest and not args.no_sampled_replay:
        import re as _re
        of_record: dict[str, list[str]] = {}
        for r in range(world):
            digs = set()
            pdr_ = (man_doc.get("program_digest_per_rank") or {}).get(str(r)) if man_doc else None
            if pdr_:
                digs.add(str(pdr_))
            wp_ = Path(builds[r]) / "build_workload" / "workload_program.json" if builds.get(r) else None
            if wp_ is not None and wp_.exists():
                try:
                    wdoc = json.loads(wp_.read_text())
                    if str(wdoc.get("workload_digest")) == str(pdr_):                 # only a composition the manifest names is of record
                        for q in wdoc.get("requests") or []:
                            spec = str(((q.get("component") or {}).get("specialization")) or "")
                            digs.update(_re.findall(r'program_digest=\\?"([0-9a-f]{128})', spec))
                except Exception as e:  # noqa: BLE001
                    print(f"[tp2commit] rank {r}: GP-01 workload_program unreadable for the replay's of-record set: {e!r}", flush=True)
            of_record[str(r)] = sorted(digs)
        replay_cfg = {"programs": {str(r): str(builds[r]) for r in range(world) if builds.get(r)}, "of_record": of_record,
                      "requests": [{"request_id": request_ids[str(r.get("request_id"))], "arrive_step": int(r.get("arrive_step") or 0), "prompt_len": len(r["prompt_token_ids"]),
                                    "max_tokens": int(r.get("max_tokens") or sampling.get("max_tokens") or 1), "prompt_token_ids": [int(t) for t in r["prompt_token_ids"]], "seed": int(r["seed"]) if r.get("seed") is not None else (int(sampling["seed"]) if sampling.get("seed") is not None else None)} for r in reqs],
                      "per_stratum": int(args.replay_per_stratum), "max_evaluate": int(args.replay_max), "seed": args.replay_seed, "workers": args.replay_workers,
                      "challenge": args.replay_challenge, "targets": list(args.replay_target or []), "ring_slots": ring_slots(args.committer, os.environ.get("VERITY_WINDOW_MB"), os.environ.get("VERITY_WINDOW_SLOTS"), retain_env)}
        versions["sampled_replay"] = {"of_record": of_record, "per_stratum": replay_cfg["per_stratum"], "max_evaluate": replay_cfg["max_evaluate"], "seed": args.replay_seed,
                                      "challenge": args.replay_challenge, "rule": "MAN-06 O1-O6 per rank: the rank's Programs of record x the rank's map x the rank's committed members + weight shards"}
        if args.replay_k:
            replay_cfg["k_per_rank"] = versions["sampled_replay"]["k_per_rank"] = TPC.replay_k_per_rank(args.replay_k, world)
    SRT.require(args, replay_cfg)
    TPC.replay_host(replay_cfg, versions, out, ranks_at_once=args.replay_ranks_at_once, memory_map=args.memory_map)
    # the weights of record per rank: each rank composes its record from ITS rank Program + the pinned checkpoint shards
    weights_pin_cfg = None
    if args.checkpoint_dir:
        weights_pin_cfg = {"checkpoint_dir": str(args.checkpoint_dir), "checkpoints_manifest": args.checkpoints_manifest, "model": args.checkpoint_model,
                           "program_dirs": {str(r): str(builds[r]) for r in range(world) if builds.get(r)}}
        if len(weights_pin_cfg["program_dirs"]) != world:
            raise SystemExit(f"--checkpoint-dir needs a rank Program dir for every rank (--build-rank<r>): have {sorted(weights_pin_cfg['program_dirs'])} of {world}")
        with PH.TL.span("prep.shard_pin"):      # every shard hashed once, here; each rank's record adopts the pin (WR.build_record shard_pin)
            weights_pin_cfg["shard_pin"] = WR.record_shard_pin(str(args.checkpoint_dir), checkpoints_manifest=args.checkpoints_manifest, model=args.checkpoint_model)
    versions["weights_pin"] = {"checkpoint_dir": args.checkpoint_dir, "checkpoints_manifest": args.checkpoints_manifest, "required": (args.required_manifest is not None and not args.no_weights_pin),
                               "rule": "MAN-09 per rank: sha256(served shard) == digest composed from the PINNED checkpoint shards by the rank Program's shard map, every field; absent with --required-manifest = FAIL 'weights unpinned (MAN-09)' unless --no-weights-pin"}
    flip_weight = None
    if args.flip_weight:
        fw = args.flip_weight.split(":")
        if len(fw) < 3:
            raise SystemExit("--flip-weight <rank>:<param>:<index>[:<value>]")
        flip_weight = {"rank": int(fw[0]), "param": fw[1], "index": int(fw[2]), "value": (float(fw[3]) if len(fw) > 3 else 1.0)}
        versions["negatives"]["flip_weight"] = flip_weight
    serve_shard_from = parse_serve_shard_from(args.serve_shard_from, world)
    if serve_shard_from:
        if not args.checkpoint_dir:
            raise SystemExit("--serve-shard-from needs --checkpoint-dir (the other rank's shard is composed from the pinned checkpoint)")
        versions["negatives"]["serve_shard_from"] = serve_shard_from
    (out / "versions.json").write_text(json.dumps(versions, indent=1, default=str) + "\n")
    with PH.TL.span("prep.committer_setup", kind=args.committer):
        made = llm.collective_rpc("tp2_commit_make", timeout=args.rpc_timeout,
                                  kwargs={"kind": args.committer, "commit_scheme": args.commit_scheme, "run_id": Path(args.workload).stem, "cp": cp, "case": case, "hidden_so": args.hidden_so,
                                          "hidden_ns": args.hidden_ns, "envelope": envelope, "arena_mb": args.arena_mb, "tp_partials": not args.no_tp_partials,
                                          "program_digest": {str(k): v for k, v in pds.items()}, "n_requests": len(reqs),
                                          "required_manifest": args.required_manifest, "request_order": request_order, "request_ids": request_ids, "arrivals": arrivals, "caps": caps,
                                          "tp_relabel": relabel, "tp_partials_omit_rank": args.omit_rank_partials,
                                          "match_tokens": match_tokens, "tp_swap_requests": swap_requests, "match_capture_dir": match_capture_dir, "replay": replay_cfg,
                                          "weights_pin": weights_pin_cfg, "flip_weight": flip_weight, "serve_shard_from": serve_shard_from, "tap_sources": TAPS.for_ranks(args.tap_sources, kv_slots=bool(args.memory_map)),
                                          "weights_as_loaded": weights_as_loaded(weights_pin_cfg), "post_load_tensors": post_load_tensors(builds),
                                          "tp_partials_flip_site": ({args.flip_partial.split(":", 1)[0]: args.flip_partial.split(":", 1)[1]} if args.flip_partial else None),
                                          "tp_partials_omit_site": ({args.omit_partial_site.split(":", 1)[0]: args.omit_partial_site.split(":", 1)[1]} if args.omit_partial_site else None),
                                          "engine_kwargs": {k: v for k, v in kw.items() if k not in ("model", "tokenizer", "worker_extension_cls")}})
    if len(made) != world:
        raise SystemExit(f"fail-closed: committer constructed on {len(made)}/{world} ranks")
    (out / "committer_make.json").write_text(json.dumps(made, indent=1, default=str) + "\n")
    # the per-rank attention plane classes (fa2_plane_classes on THIS rank's Program record) -- said per rank
    for m in made:
        apc = m.get("attention_plane_classes") or {}
        print(f"[tp2commit] rank {m.get('rank')}: X-03 attention plane classes: {apc.get('declarations', 0)} layer declaration(s) from {apc.get('source')} "
              f"-> classes {apc.get('classes')} heads(NH,KVH,D) {apc.get('per_rank_heads')}; {apc.get('rule')}" + (f"; ERROR {apc['error']}" if apc.get("error") else ""), flush=True)
    versions["attention_plane_classes"] = {str(m.get("rank")): {k: v for k, v in (m.get("attention_plane_classes") or {}).items() if k != "rank"} for m in made}
    # an MoE model's bound block is a required class of every rank (as in commit_delta): the ranks report the
    # attached MoeBlockSource; all ranks must agree (a rank without MoE layers on an MoE model is a named refusal, not a silent hole)
    moe_ranks = [int(m["rank"]) for m in made if isinstance(m.get("moe_source"), dict) and int((m["moe_source"] or {}).get("layers") or 0) > 0]
    if moe_ranks and len(moe_ranks) != world:
        raise SystemExit(f"fail-closed: the MoE block source attached on ranks {moe_ranks} of {world} -- every rank of an MoE model binds its planes")
    if moe_ranks and args.required_classes is None and "moe_block_stream" not in required_classes:
        required_classes = required_classes + ("moe_block_stream",)
        versions["required_classes"] = list(required_classes)
        versions["required_classes_source"] += " + moe_block_stream (R15 TP-MOE-SRC: MoE-runner modules on every rank)"
        versions["moe_source"] = {int(m["rank"]): m.get("moe_source") for m in made}
        (out / "versions.json").write_text(json.dumps(versions, indent=1, default=str) + "\n")
        print(f"[tp2commit] MoE block source attached on every rank ({versions['moe_source'][0]}); required classes {list(required_classes)}", flush=True)
    required_classes = TAPS.require_on_every_rank(made, required_classes, versions, args.tap_sources, explicit_classes=args.required_classes is not None)
    for m in made:
        asz = m.get("arena_sizing") or {}
        rm = m.get("required_manifest") or {}
        print(f"[tp2commit] rank {m['rank']}: {m['committer']} weights {str(m.get('weight_registration'))[:200]}; arena "
              f"{ {k: asz.get(k) for k in ('decode_block_bytes', 'prefill_block_bytes', 'step_bound_bytes', 'device_free_bytes')} }"
              + (f"; manifest {str(rm.get('manifest_digest'))[:16]} identities {rm.get('identities')} program_matches {rm.get('program_digest_matches')} complete {rm.get('complete')}" if rm else "")
              + (f"; tp partials {m.get('tp_partial_source')}" if (m.get("tp_partial_source") or {}).get("relabel_rank") is not None else ""), flush=True)
        if rm and rm.get("program_digest_matches") is False:
            raise SystemExit(f"fail-closed: rank {m['rank']}: manifest Program digest != this rank's Program digest")
    with PH.TL.span("prep.warmup_instrumented", kind=args.committer) as warm:
        warm["wall_s"] = round(WU.run(llm, lambda: run_workload(eng, va.throwaway_requests(eng, reqs, seed=200), sampling), out, args.committer, args.rpc_timeout)["wall_s"], 3)
    rows: list[dict] = []
    bundles_of: dict[int, dict[int, dict | None]] = {}     # --replay-deferred: pair -> rank -> what its finalize wrote for its bundle
    tokens_ref, started = None, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())   # started: this Commit's stamp on its rows, pair files, summary (tp.rejudge)
    all_pass = True
    xrank_pairs: list[dict] = []     # one verdict per instrumented pair
    from verity_vllm.program.frontend import collective_kinds
    xrank_families = collective_kinds.cross_rank_families(world)      # the cross-rank families of THIS world, from the registry's composites
    xrank_on = args.xrank_per_stratum > 0 and not args.no_tp_partials
    runs = schedule(args.pairs, args.only_arm)
    stopped = None
    for pair in range(args.pairs):
        order = [a for p, a in runs if p == pair]
        for k, arm in enumerate(order):
            gc.collect()
            with PH.TL.span(f"work.pair{pair}.{arm}", pair=pair, arm=arm) as ran:
                if arm != "control":
                    inst = llm.collective_rpc("tp2_commit_install", timeout=args.rpc_timeout)
                    if len(inst) != world:
                        raise SystemExit("fail-closed: install did not reach every rank")
                cpu0 = llm.collective_rpc(TRACE.thread_cpu, timeout=args.rpc_timeout)
                r = run_workload(eng, reqs, sampling)
                cpu = TRACE.cpu_delta(cpu0, llm.collective_rpc(TRACE.thread_cpu, timeout=args.rpc_timeout))
                ran.update(wall_s=round(r["wall_s"], 4), steps=len(r["step_wall_s"]))
                row = {"schema": SCHEMA, "label": args.label, "workload": args.workload, "workload_sha256": wl.get("sha256"), "case": case, "model": cp["repo"],
                       "tp": args.tp, "pair": pair, "order_in_pair": k, "arm": arm, "committer": None if arm == "control" else args.committer,
                       "wall_s": r["wall_s"], "iterations": r["iterations"], "engine_steps": len(r["step_wall_s"]), "generated_tokens": r["generated_tokens"],
                       "prompt_tokens": r["prompt_tokens"], "generated_tokens_per_s": r["generated_tokens"] / r["wall_s"],
                       "step_wall_prefill_s": r["step_wall_s"][0], "step_wall_decode_mean_s": statistics.fmean(r["step_wall_s"][1:]) if len(r["step_wall_s"]) > 1 else None,
                       "step_wall_s": r["step_wall_s"], "step_tokens_out": r["step_tokens_out"], "thread_cpu_s": cpu,
                       "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "commit_started": started}
                if tokens_ref is None:
                    tokens_ref = r["tokens"]
                row["tokens_equal_ref"] = r["tokens"] == tokens_ref
                if arm != "control":
                    t_f = time.perf_counter()
                    fin = TPL.finalize(llm, args.rpc_timeout, f"work.pair{pair}.{arm}", last=pair == args.pairs - 1 and k == len(order) - 1 and not args.value_check and not args.trace_dir,
                                       defer={"out": str(out), "pair": pair} if args.replay_deferred else None)
                    t_ready = time.perf_counter() - t_f
                    if args.replay_deferred:
                        bundles_of[pair] = {int(f["rank"]): (f.get("sampled_replay") or {}).pop("bundle", None) for f in fin}
            if arm != "control":
                if match_capture_dir is not None:
                    differs = sorted((w for f in fin for w in (((f.get("match_oracle") or {}).get("witnesses") or {}).get("differs") or [])),
                                     key=lambda w: (int(w["step"]), int(w["k"]), int(w["rank"])))
                    print(f"[tp2commit] match oracle (tp partials vs capture), {len(differs)} differing MoE align witness(es)"
                          + (f", first rank {differs[0]['rank']} step {differs[0]['step']} k {differs[0]['k']} {differs[0]['module']} implies {differs[0]['implies']}" if differs else "") + ": " + "; ".join(
                        f"rank {f['rank']} {mo.get('verdict')} steps {mo.get('steps_compared')} tensors {mo.get('tensors')} "
                        f"equal {mo.get('equal')} mismatch {mo.get('mismatch_n')} missing {mo.get('missing_n')} "
                        f"not_in_oracle {mo.get('not_in_oracle_n')} witnesses equal {(mo.get('witnesses') or {}).get('equal')} unpaired {(mo.get('witnesses') or {}).get('unpaired_n')} "
                        f"{mo.get('by_member')} {mo.get('error') or ''}"
                        for f in sorted(fin, key=lambda f: f["rank"]) for mo in [f.get("match_oracle") or {}]), flush=True)
                if replay_cfg is not None:
                    for f in sorted(fin, key=lambda f: f["rank"]):
                        srr = ((f.get("sampled_replay") or {}).get("sampled_replay") or {})
                        lkr = ((f.get("sampled_replay") or {}).get("boundary_linkage") or {})
                        smp, popn = srr.get("sample") or {}, srr.get("population") or {}
                        print(f"[tp2commit] sampled replay rank {f['rank']}: result {srr.get('result')} grade {srr.get('grade')} population {popn.get('vus')} strata {popn.get('strata_n')} "
                              f"picked {smp.get('picked')} evaluated {smp.get('evaluated')} equal {smp.get('equal')} mismatch {smp.get('mismatch_n')} not_evaluated {smp.get('not_evaluated_n')} "
                              f"weights self-consistent {((srr.get('weights') or {}).get('equal'))} (not a pin) wall {((srr.get('timing') or {}).get('wall_s'))}s; boundary linkage {lkr.get('result')} "
                              f"{lkr.get('linked')}/{lkr.get('pairs')}; {srr.get('why') or ''}", flush=True)
                        for uf in (srr.get("unreplayable") or [])[:8]:
                            print(f"[tp2commit]   rank {f['rank']} unreplayable: {uf}", flush=True)
                for f in sorted(fin, key=lambda f: f["rank"]):
                    wpp = f.get("weights_pin") or {}
                    print(f"[tp2commit] weights pin rank {f['rank']}: result {wpp.get('result')} root_of_record {str(wpp.get('root_of_record'))[:16]} compared {wpp.get('compared')} "
                          f"mismatched {len(wpp.get('mismatched') or [])} missing {len(wpp.get('missing') or [])} unpinned {len(wpp.get('unpinned') or [])} "
                          f"shards_pinned {wpp.get('all_shards_pinned')}; {wpp.get('why') or ''}", flush=True)
                if args.omit_rank is not None:
                    fin = [f for f in fin if int(f["rank"]) != args.omit_rank]      # NEGATIVE: one rank's commitment missing from the vector
                if len(fin) != world or sorted(f["rank"] for f in fin) != list(range(world)):
                    row["commit"] = {"commit_pass": False, "commit_fail_reason": f"fail-closed: {len(fin)}/{world} ranks finalized"}
                    all_pass = False
                    (out / "coverage.json").write_text(json.dumps({"checker": "verity/coverage/binding-r14 (per TP rank)", "verdict": "FAIL", "rules_version": "r14-v1", "world": world,
                                                                    "ranks_present": sorted(int(f["rank"]) for f in fin), "identity_coverage": False,
                                                                    "reason": f"omit-one-rank: the expected population spans ranks {list(range(world))}; ranks {sorted(set(range(world)) - {int(f['rank']) for f in fin})} have no commitment, no binding map, no openings -- every identity of those ranks is missing"}, indent=1) + "\n")
                else:
                    (out / f"finalize_pair{pair}.json").write_text(json.dumps(fin, indent=1, default=str) + "\n")
                    if args.wrong_root_rank is not None:
                        # NEGATIVE: the commitment vector the verifier holds carries a wrong root for one rank (first byte flipped); every
                        # opening of that rank is verified against the CLAIMED root and must fail; the map context (run_root) fails too
                        for f in fin:
                            if int(f["rank"]) == args.wrong_root_rank:
                                rb = bytearray(bytes.fromhex(f["run_root"]))
                                rb[0] ^= 0xFF
                                f["run_root_actual"] = f["run_root"]
                                f["run_root"] = bytes(rb).hex()
                    roots = {int(f["rank"]): bytes.fromhex(f["run_root"]) for f in fin}
                    root = RT.tp_run_root(world, roots)
                    claimed = {str(int(f["rank"])): f["run_root"] for f in fin}
                    served = SRT.commit(llm, args, root, claimed, world) if args.serving_rows and pair == 0 else None
                    per_rank = {}
                    reasons = []
                    # every rank's root is fixed and the TP run root with them: the pair's coins come now, and every rank's draws derive from them (one RPC: each rank opens its own)
                    coin = CC.tp_draw(coins, root)
                    src_pos, src_id = CC.tp_keys(coin)
                    positions = {str(int(f["rank"])): [list(p) for p in challenge_positions(root, f["leaves_per_step"], args.openings, salt=f"{pair}/rank{int(f['rank'])}".encode(),
                                                                                            legacy=False, source=src_pos)] for f in fin}
                    op = llm.collective_rpc("tp2_commit_open", timeout=args.rpc_timeout, kwargs={"positions_by_rank": positions, "claimed_roots": claimed})
                    for f in sorted(fin, key=lambda f: f["rank"]):
                        rk = int(f["rank"])
                        op_r = next(o for o in op if int(o["rank"]) == rk)
                        cblock = {"run_root": f["run_root"], "steps": f["steps"], "leaves": f["leaves"], "bytes_bound": f["bytes_bound"], "tensors": f["tensors"],
                                  "root_ready_delay_s": f["root_ready_delay_s"], "finalize_s": f["finalize_s"], "spans_total_s": f["spans_total_s"],
                                  "extra": {"expected_tensors": f["expected_tensors"], "committed_tensors": f["committed_tensors"], "errors_n": f["errors_n"], "errors": f["errors"],
                                            "collector_failure": f.get("collector_failure"), "identity_check": f.get("identity_check"), "expected_manifest": f.get("expected_manifest"), "bounded_staging": f.get("bounded_staging")},
                                  "openings": {k2: op_r.get(k2) for k2 in ("n", "verified", "generate_s", "verify_s", "bytes_mean", "error")},
                                  "classes": f["classes"], "weight_registration": f["weight_registration"],
                                  "persistent_bytes": f["persistent_bytes"], "temporary_bytes_peak": f["temporary_bytes_peak"]}
                        ok, why = commit_verdict(cblock, row["tokens_equal_ref"], min_openings=min(64, args.openings), declaration=f["declaration"],
                                                 required_classes=required_classes)
                        # the rank's binding map + coverage against its expected population gate the verdict too
                        b = f.get("binding") or {}
                        cblock["binding"] = b
                        if b.get("error"):
                            cblock["binding_note"] = b["error"]
                            ok, why = False, f"binding map / coverage error: {b['error']}"          # fail closed: no map = no coverage verdict = no Commit
                        elif b.get("skipped"):
                            cblock["binding_note"] = b["skipped"]
                            if man_doc is not None:
                                ok, why = False, f"identity coverage not run: {b['skipped']}"
                        elif not (b.get("coverage") or {}).get("ok"):
                            ok, why = False, f"identity coverage: {(b.get('coverage') or {}).get('missing_n')} of {(b.get('coverage') or {}).get('expected_n')} required identities have no binding (first: {((b.get('coverage') or {}).get('missing') or [None])[0]})"
                        elif b.get("context") and b["context"].get("ok") is False:
                            ok, why = False, f"binding map context: {b['context'].get('why')}"
                        elif (b.get("coverage") or {}).get("error"):
                            ok, why = False, f"identity coverage: {b['coverage']['error']}"
                        if ok and f.get("run_root_actual"):
                            # the claimed root differs from the rank's actual root: openings verified against the claim must have failed
                            ok, why = False, f"claimed run root for rank {rk} {f['run_root'][:16]} != the rank's root (openings verified {op_r.get('verified')}/{op_r.get('n')} against the claim)"
                        # per-(request, engine step) attribution against the Match record (when --match-dir is given)
                        att = f.get("attribution")
                        cblock["attribution"] = att
                        if match_tokens is not None and (att is None or not att.get("ok")):
                            why_ = f"request attribution vs Match record: {(att or {}).get('why') or 'not run'}"
                            ok, why = False, (f"{why}; also: {why_}" if not ok and why else why_)      # every failing gate is named, first fault first
                        # committed partials / MoE received outputs == the Match capture's stored
                        # collectives, per (step, k), this rank; exhaustive over the family or FAIL (a gate whenever the oracle was given)
                        mo = f.get("match_oracle")
                        cblock["match_oracle"] = mo
                        if match_capture_dir is not None and (mo is None or mo.get("verdict") != "PASS"):
                            why_ = f"match oracle (tp partials vs tp2_match capture): {(mo or {}).get('why') or (mo or {}).get('error') or 'not run'}"
                            ok, why = False, (f"{why}; also: {why_}" if not ok and why else why_)
                        ok, why = C2.c2_and_pin_gates(cblock, f, ok, why, rank=rk, replay_on=replay_cfg is not None, required=args.required_manifest is not None,
                                                   bundle=RB.path(out, pair, rank=rk).name, xrank_families=xrank_families if xrank_on else None)
                        cblock["commit_pass"], cblock["commit_fail_reason"] = ok, why
                        if ok is False:
                            reasons.append(f"rank {rk}: {why}")
                        per_rank[rk] = cblock
                        if pair == 0:
                            (out / f"declaration_rank{rk}.json").write_text(json.dumps(f["declaration"], indent=1, default=str) + "\n")
                    if pair == 0:
                        # the per-rank binding maps themselves (not only their digests) into the evidence: binding_map_rank<r>.json
                        dumped = llm.collective_rpc("tp2_commit_binding_map_dump", timeout=args.rpc_timeout, kwargs={"out_dir": str(out)})
                        print(f"[tp2commit] binding maps: {[(d_.get('rank'), str(d_.get('digest') or d_.get('skipped'))[:16], d_.get('entries_n')) for d_ in dumped]}", flush=True)
                    # population-drawn openings per rank (bind's challenge_identities from the COMBINED root + the rank's map digest)
                    pop_open = {}
                    if all((f.get("binding") or {}).get("map_digest") for f in fin):
                        po = []
                        for rk_c, croot in sorted(claimed.items()):
                            # one RPC per rank so each rank verifies against ITS claimed root (the commitment-vector entry)
                            po_r = llm.collective_rpc("tp2_commit_open_population", timeout=args.rpc_timeout,
                                                      kwargs={"k": args.openings, "salt": f"{pair}/population", "combined_root_hex": root.hex(), "source_hex": src_id.hex(),
                                                              "claimed_root_hex": croot})
                            po.extend(o for o in po_r if str(int(o["rank"])) == rk_c)
                        for o in po:
                            rk = int(o["rank"])
                            pop_open[rk] = o
                            per_rank[rk]["openings_population"] = o
                            if o.get("unbound_n") or o.get("verified") != o.get("opened") or not o.get("opened"):
                                per_rank[rk]["commit_pass"] = False
                                per_rank[rk]["commit_fail_reason"] = f"population openings: {o.get('verified')}/{o.get('opened')} verified, {o.get('unbound_n')} unbound ({o.get('error')})"
                                reasons.append(f"rank {rk}: {per_rank[rk]['commit_fail_reason']}")
                            elif (o.get("context") or {}).get("ok") is False:
                                per_rank[rk]["commit_pass"] = False
                                per_rank[rk]["commit_fail_reason"] = f"binding map context vs claimed root: {o['context'].get('why')}"
                                reasons.append(f"rank {rk}: {per_rank[rk]['commit_fail_reason']}")
                    # the cross-rank collective-output pass: the received AllReduce2 / AllGather2 / EmbeddingShard output
                    # each rank COMMITTED, evaluated bit-exactly against the definition over the two ranks' COMMITTED partials (picks seeded by the
                    # combined root, stratified per site x kind x phase).  The only evaluator of the collective-output VUs from committed bytes (the
                    # per-rank replay has none); a FAIL names the site and fails the ranks whose received output is not the definition.
                    xr = {"result": None, "why": "not run (--xrank-per-stratum 0)"}
                    if xrank_on:
                        try:
                            import torch as _torch

                            from verity_vllm.check.match import xrank_collectives as XC
                            xd = llm.collective_rpc("tp2_commit_xrank_dump", timeout=args.rpc_timeout,
                                                    kwargs={"out_dir": str(out), "seed_hex": root.hex(), "pair": pair, "per_stratum": args.xrank_per_stratum})
                            dumps = {int(d_["rank"]): _torch.load(d_["file"], weights_only=False) for d_ in xd}
                            xr = XC.check_picks(dumps, world=world)
                            XC.coverage_of_unreplayable(xr, xd)      # strata sites vs the Programs' unreplayable collective sites (fails by name)
                            xr["dumps"] = [{k: v for k, v in d_.items()} for d_ in xd]
                            xr["pair"] = pair
                            for fn in {d_["file"] for d_ in xd}:
                                Path(fn).unlink(missing_ok=True)      # the picks are re-derivable from the retained records; the JSON verdict is the evidence
                            (out / f"xrank_pair{pair}.json").write_text(json.dumps({"schema": "verity/tp2/xrank-collectives/v1", "tp": args.tp, "commit_started": started} | xr, indent=1, default=str) + "\n")
                            print(f"[tp2commit] cross-rank collectives pair {pair}: result {xr['result']} picks {xr['picks']} equal {xr['equal']} mismatch {xr.get('mismatch_n')} "
                                  f"missing {xr.get('missing_n')} families {sorted(xr.get('by_family') or {})} flipped {[d_.get('flipped_collective_inputs') for d_ in xd]}; {xr.get('why') or ''}", flush=True)
                        except Exception as e:  # noqa: BLE001
                            xr = {"result": False, "why": f"cross-rank collective pass did not run: {e!r}"[:300], "pair": pair}
                            print(f"[tp2commit] cross-rank collectives pair {pair}: ERROR {e!r}", flush=True)
                    xrank_pairs.append(xr)
                    if xr.get("result") is False:
                        # the ranks named by the first mismatches (received != definition) fail; when none is named (not evaluable) every rank fails
                        bad = {int(r_) for m_ in (xr.get("mismatches") or []) for r_, ok_ in (m_.get("received_equals_definition_per_rank") or {}).items() if not ok_}
                        for rk in (sorted(bad) if bad else sorted(per_rank)):
                            per_rank[rk]["commit_pass"] = False
                            per_rank[rk]["commit_fail_reason"] = f"cross-rank collective outputs (TP-12): {xr.get('why')}"
                            reasons.append(f"rank {rk}: {per_rank[rk]['commit_fail_reason']}")
                    for rk in per_rank:
                        per_rank[rk]["cross_rank_collectives"] = {k: xr.get(k) for k in ("result", "picks", "equal", "mismatch_n", "missing_n", "why")}
                    try:
                        C2.annotate_unreplayable([((f_.get("sampled_replay") or {}).get("sampled_replay") or {}) for f_ in fin], xr, pair, xrank_families)
                        (out / f"finalize_pair{pair}.json").write_text(json.dumps(fin, indent=1, default=str) + "\n")
                    except Exception as e:  # noqa: BLE001
                        print(f"[tp2commit] unreplayable.evaluated_by annotate failed: {e!r}", flush=True)
                    skipped = [str((c.get("binding") or {}).get("skipped")) for c in per_rank.values() if (c.get("binding") or {}).get("skipped")]
                    pending = sorted(rk for rk, c in per_rank.items() if c.get("commit_pass") is None)
                    if pair == 0 or reasons or pending:
                        cov_ok = all((c.get("binding") or {}).get("coverage", {}).get("ok") for c in per_rank.values()) and not reasons
                        man_hdrs = {r: ((c.get("binding") or {}).get("population") or {}).get("manifest") for r, c in per_rank.items()}
                        cov = {"checker": ("verity/coverage/binding-r15-tp04 (verity_vllm.commit.binding per TP rank; population of record = the required manifest "
                                           "(Program(rank) x canonical query, rows with rank=r: instance outputs incl. AllReduce2/AllGather2 outputs + tp_rank_partials + fa2 hidden + weights + tokens); "
                                           "coverage counted per rank; no rank folded)" if man_doc else
                                           "verity/coverage/binding-r14 (bind BIND-01 verity_vllm.commit.binding, per TP rank; population = structure x steps x rank + rank-local collective members)"),
                               "verdict": ("NOT_RUN" if len(skipped) == len(per_rank) else ("PASS" if cov_ok else "FAIL")), "rules_version": versions["rules_version"], "world": world, "ranks_present": sorted(per_rank),
                               "skipped": skipped or None,
                               "population_source": "required-manifest" if man_doc else "structural",
                               "required_manifest": ({"path": args.required_manifest, "manifest_digest": man_doc.get("manifest_digest"), "identities": (man_doc.get("populations") or {}).get("identities"),
                                                      "by_family": (man_doc.get("populations") or {}).get("by_family"), "complete": man_doc.get("complete"), "ranks": man_doc.get("ranks"),
                                                      "program_digest_per_rank": man_doc.get("program_digest_per_rank"), "query_digest": (man_doc.get("query") or {}).get("query_digest"),
                                                      "tp_rule": (man_doc.get("query") or {}).get("tp")} if man_doc else None),
                               "manifest_digest": (man_doc.get("manifest_digest") if man_doc else "+".join(str((c.get("binding") or {}).get("population", {}).get("digest"))[:16] for _r, c in sorted(per_rank.items()))),
                               "manifest_digests_per_rank": {r: (c.get("binding") or {}).get("population", {}).get("digest") for r, c in per_rank.items()},
                               "manifest_rows_per_rank": {r: (h or {}).get("rows_for_rank") for r, h in man_hdrs.items()} if man_doc else None,
                               # informational only: the manifest coverage above is the verdict; this is the old structural population beside it
                               "structural_coverage_ok_per_rank": {r: (h or {}).get("structural_coverage_ok") for r, h in man_hdrs.items()} if man_doc else None,
                               "structural_coverage_note": "informational (not a verdict): coverage of the collector's structural population, recorded beside the manifest coverage",
                               "population_n_per_rank": {r: (c.get("binding") or {}).get("population", {}).get("n") for r, c in per_rank.items()},
                               "binding_map_digest": "+".join(str((c.get("binding") or {}).get("map_digest"))[:16] for _r, c in sorted(per_rank.items())),
                               "binding_map_digests_per_rank": {r: (c.get("binding") or {}).get("map_digest") for r, c in per_rank.items()},
                               "binding_map_files": {r: f"binding_map_rank{r}.json" for r in per_rank},
                               "program_digests_per_rank": {r: (c.get("binding") or {}).get("program_digest") for r, c in per_rank.items()},
                               "coverage_per_rank": {r: {k2: (c.get("binding") or {}).get("coverage", {}).get(k2) for k2 in ("ok", "expected_n", "bound_n", "missing_n", "extra_n", "physical_copies_n")} for r, c in per_rank.items()},
                               "identity_coverage": (None if len(skipped) == len(per_rank) else cov_ok), "openings_population": ("expected population" if pop_open else None),
                               "openings_checked": sum(int(o.get("opened") or 0) for o in pop_open.values()), "openings_ok": sum(int(o.get("verified") or 0) for o in pop_open.values()),
                               "openings_unbound": sum(int(o.get("unbound_n") or 0) for o in pop_open.values()),
                               "openings_per_rank": pop_open, "late_step_in_sample": all(o.get("late_step_in_sample") for o in pop_open.values()) if pop_open else None,
                               "value_correctness": None,
                               "value_correctness_note": ("value correspondence (C2) is a separate labelled check: --value-check -> value_check.json (per rank: module outputs by "
                                                          "value_check.ValueChecker, TP partials by TPPartialSource.compare); never folded into this coverage verdict"),
                               # per-(request, engine step) attribution of the committed sampled ids against the Match record, per rank
                               "attribution_per_rank": {r: ({k2: (c.get("attribution") or {}).get(k2) for k2 in ("ok", "steps_checked", "rows_checked", "mismatches_n", "why", "swap_requests_negative")}
                                                            if c.get("attribution") is not None else None) for r, c in per_rank.items()},
                               "attribution_reference": versions.get("match_record"),
                               "openings_after_release": True, "collector_variant": "windowed", "retention": retain_env or "host",
                               "negatives_suite": "tp2_commit --omit-rank / --omit-rank-partials <r> / --omit-partial-site r:site / --flip-partial r:site (TP-12) / --serve-shard-from r:param:from (TP-10) / --relabel-rank a:b / --wrong-root-rank <r> / --no-tp-partials / --swap-requests r:ra:rb + bind test_binding_negatives.py (BIND-01, TP1)",
                               "negatives": versions["negatives"],
                               "reason": "; ".join(reasons) if reasons else None, "replay_pending_ranks": [int(r_) for r_ in pending] or None}
                        (out / ("coverage.json" if pair == 0 else f"coverage_pair{pair}.json")).write_text(json.dumps(cov, indent=1, default=str) + "\n")
                    row["commit"] = {"tp_run_root": root.hex(), "coins": coin, "world": world, "per_rank": per_rank, "root_combination": versions["root_combination"],
                                     "commitment_vector": [{"rank": int(f["rank"]), "run_root": f["run_root"], "leaves": f["leaves"], "steps": f["steps"]} for f in sorted(fin, key=lambda f: f["rank"])],
                                     "root_form": f"{RT.TP_ROOT_TAG.decode().removeprefix('verity/')} (PROVISIONAL tagged SHA-512 over the per-rank roots; canonical form = the per-rank entries of `commitment_vector` in the challenge-v1 commitment vector)",
                                     "leaves": sum(c["leaves"] for c in per_rank.values()), "bytes_bound": sum(c["bytes_bound"] for c in per_rank.values()),
                                     "tensors": sum(c["tensors"] for c in per_rank.values()),
                                     "openings": {"n": sum(c["openings"]["n"] for c in per_rank.values()), "verified": sum(c["openings"]["verified"] for c in per_rank.values())},
                                     "t_ready_after_run_s": t_ready, "serving_rows": served, "commit_pass": False if reasons else (None if pending else True),
                                     "commit_fail_reason": "; ".join(reasons) if reasons else (RB.PENDING if pending else "PASS")}
                    if reasons:
                        all_pass = False
                        print(f"[tp2commit] COMMIT FAIL (pair {pair}): {row['commit']['commit_fail_reason']}", flush=True)
                TPL.reset(llm, args.rpc_timeout)
            rows.append(row)
            with (out / "runs.jsonl").open("a") as f:
                f.write(json.dumps(row, default=str) + "\n")
            c = row.get("commit") or {}
            print(f"[tp2commit] pair {pair} {arm}: {r['wall_s']:.3f}s steps {len(r['step_wall_s'])} tokens_equal {row['tokens_equal_ref']}"
                  + (f" root {c.get('tp_run_root', '')[:16]} leaves {c.get('leaves')} openings {c.get('openings')} verdict {c.get('commit_fail_reason')}" if arm != "control" else ""),
                  flush=True)
            if not args.all_pairs and not (all_pass and row["tokens_equal_ref"]):
                # fail-fast: the verdict is FAIL whatever follows; this run was the Commit's last device work
                stopped = {"after": f"work.pair{pair}.{arm}", "pair": pair, "arm": arm,
                           "why": c.get("commit_fail_reason") if c.get("commit_pass") is False else "tokens differ from the first run's",
                           "not_run": [f"pair{p}.{a}" for p, a in runs[len(rows):]] + (["value_check"] if args.value_check else []),
                           "rule": "the Commit stops after its first failed run (--all-pairs runs every one)"}
                print(f"[tp2commit] FAIL-FAST: stopped after {stopped['after']}; not run: {stopped['not_run']}", flush=True)
                TPL.stop(llm, args.rpc_timeout, stopped["after"])
                break
        if stopped:
            break
    if coins is not None and (why_c := coins.close()):
        print(f"[tp2commit] the challenger session did not close (its ledger has no end, and its coins credit nothing): {why_c}", flush=True)
    if args.trace_dir and not stopped:
        gc.collect()
        with PH.TL.span("work.trace"):
            TRACE.run(llm, lambda: run_workload(eng, reqs, sampling), Path(args.trace_dir), [int(x) for x in args.trace_ranks.split(",")],
                      args.rpc_timeout, release=not args.value_check)
    vcheck = None
    if args.value_check and not stopped:
        # one extra instrumented run per rank (untimed): committed bytes vs hook-time reference clones + re-execution
        gc.collect()
        with PH.TL.span("validate.value_check"):
            inst = llm.collective_rpc("tp2_commit_value_check_install", timeout=args.rpc_timeout)
            if len(inst) != world:
                raise SystemExit("fail-closed: value-check install did not reach every rank")
            r = run_workload(eng, reqs, sampling)
            vres = llm.collective_rpc("tp2_commit_value_check", timeout=args.rpc_timeout)
        vcheck = {"schema": "verity/commit-integ/value-check-tp/v1", "label": "measured", "tp": args.tp, "tokens_equal_ref": r["tokens"] == tokens_ref,
                  "per_rank": {int(v["rank"]): v for v in vres}, "ranks_answered": sorted(int(v["rank"]) for v in vres),
                  "verdict": ("PASS" if len(vres) == world and all(v.get("verdict") == "PASS" for v in vres) and r["tokens"] == tokens_ref else "FAIL"),
                  "method": (vres[0].get("method") if vres else None)}
        (out / "value_check.json").write_text(json.dumps(vcheck, indent=1, default=str) + "\n")
        # the late-read detector record, per rank, from the same value-check run
        t64 = {int(v["rank"]): (v.get("t6_4") or {"result": False, "error": "no t6_4 in the rank's value-check result"}) for v in vres}
        # aggregate: True only when every rank is True; None (not applicable) when every rank recorded applicable False (dense TP row); else False
        t64_all_na = bool(t64) and all(v.get("applicable") is False and v.get("result") is None for v in t64.values())
        (out / "t6_4_check.json").write_text(json.dumps({"schema": "verity/commit-integ/t6_4-check/v1", "tp": args.tp, "per_rank": t64,
                                                           "applicable": (not t64_all_na),
                                                           "result": (None if t64_all_na else bool(len(t64) == world and all(v.get("result") is True for v in t64.values()))),
                                                           "layers_compared": sum(int(v.get("layers_compared") or 0) for v in t64.values()),
                                                           "equal_pairs": sum(int(v.get("equal_pairs") or 0) for v in t64.values()),
                                                           "missing_pairs": sum(int(v.get("missing_pairs") or 0) for v in t64.values()),
                                                           "clone_mismatches": (None if any(v.get("clone_mismatches") is None for v in t64.values())
                                                                                else sum(int(v.get("clone_mismatches") or 0) for v in t64.values())),
                                                           "out_tensors_vs_clone": sum(int(v.get("out_tensors_vs_clone") or 0) for v in t64.values()),
                                                           "clone_rule": next((v.get("clone_rule") for v in t64.values() if v.get("clone_rule")), None),
                                                           "pairs": [dict(p, rank=r) for r, v in sorted(t64.items()) for p in (v.get("pairs") or [])][:400],
                                                           "rule": next((v.get("rule") for v in t64.values() if v.get("rule")), None)},
                                                          indent=1, default=str) + "\n")
        for v in sorted(vres, key=lambda v: v["rank"]):
            mo, tp_ = v.get("modules") or {}, v.get("tp_partials") or {}
            print(f"[tp2commit] VALUE CHECK rank {v['rank']}: {v['verdict']} | modules {mo.get('verdict')} tensors {mo.get('tensors')} tap-eq {mo.get('tap_equal')} "
                  f"tap-mismatch {mo.get('tap_mismatch_n')} reexec {mo.get('reexec_equal')}/{mo.get('reexec_tensors')} missing {len(mo.get('missing_in_commit') or [])} "
                  f"| tp partials {tp_.get('verdict')} tensors {tp_.get('tensors')} tap-eq {tp_.get('tap_equal')} reexec {tp_.get('reexec_equal')}/{tp_.get('reexec_tensors')} "
                  f"missing {tp_.get('missing_n')} by_member {tp_.get('by_member')}", flush=True)
        if vcheck["verdict"] != "PASS":
            all_pass = False
        # the C2 record next to the C1 record (coverage.json) -- separate labelled check, never folded into the coverage verdict
        try:
            covp = out / "coverage.json"
            if covp.exists():
                cj = json.loads(covp.read_text())
                cj["value_correctness"] = vcheck["verdict"] == "PASS"
                roots_seen = sorted({str((r.get("commit") or {}).get("tp_run_root")) for r in rows if r["arm"] != "control" and (r.get("commit") or {}).get("tp_run_root")})
                cj["value_correspondence"] = {"method": vcheck["method"], "result": vcheck["verdict"] == "PASS", "file": "value_check.json", "artifact": str(out / "value_check.json"),
                                              # per-component grades: what ran, what it was compared against
                                              # `tap_reexec` is NOT a live oracle: the reference clone is taken in the committing process at hook time
                                              # (an independent copy of the same producer's bytes -- it catches late reads / wrong offsets / wrong tensors, not a wrong producer);
                                              # `oracle_compare` is reserved for a compare against the MATCH RECORD (an independent run): the per-(request, engine step)
                                              # sampled-ids attribution below when --match-dir was given, else not run
                                              "components": {"tap_reexec": {"result": vcheck["verdict"] == "PASS", "reference": "hook-time reference clone (T) of every module output and TP partial + re-execution (R) of the registered relation / rank-local shard GEMM on the tapped inputs, per rank, in the committing process (value_check.json); not an independent oracle",
                                                                            "per_rank": {r_: v.get("verdict") for r_, v in vcheck["per_rank"].items()}},
                                                             # [TP family + sampled ids; `complete` False: collective-level Match, see c2.oracle_compare_component]
                                                             "oracle_compare": C2.oracle_compare_component(rows, versions, man_doc),
                                                             "linkage": C2.linkage_component(rows, versions),
                                                             "root_compare": {"result": None, "reference": ("no known-good TP2 run root pinned (known_roots has no tp2 entry); the instrumented pairs' combined roots were " + ("identical: " + roots_seen[0][:16] if len(roots_seen) == 1 else f"NOT identical: {[x[:16] for x in roots_seen]}")) if roots_seen else "no run root"},
                                                             "replay": {"result": None, "reference": "not run (no independent replay of a TP2 run in this attempt)"}},
                                              "oracle": (vres[0].get("oracle") if vres else None), "identities_checked": {r_: {"module_outputs": (v.get("modules") or {}).get("tensors"), "tp_partials": (v.get("tp_partials") or {}).get("tensors")} for r_, v in vcheck["per_rank"].items()},
                                              "per_rank": {r_: {"verdict": v.get("verdict"), "modules": (v.get("modules") or {}).get("verdict"), "tp_partials": (v.get("tp_partials") or {}).get("verdict")}
                                                           for r_, v in vcheck["per_rank"].items()}}
                covp.write_text(json.dumps(cj, indent=1, default=str) + "\n")
        except Exception as e:  # noqa: BLE001
            print(f"[tp2commit] coverage.json value_correspondence annotate failed: {e!r}", flush=True)

    if vcheck is None and match_tokens is not None:
        # no value check this attempt, but the Match-record linkage ran: record it as a labelled component beside C1
        try:
            covp = out / "coverage.json"
            if covp.exists():
                cj = json.loads(covp.read_text())
                lk = C2.linkage_component(rows, versions)
                oc = C2.oracle_compare_component(rows, versions, man_doc)
                cj["value_correspondence"] = {"method": "Match-record compare only (no --value-check this attempt): sampled-ids attribution + TP partials vs the capture's stored collectives",
                                              "result": (lk["result"] and oc.get("result")) if lk["result"] is not None else None, "file": None,
                                              "components": {"tap_reexec": {"result": None, "reference": "not run (no --value-check)"},
                                                             "oracle_compare": C2.oracle_compare_component(rows, versions, man_doc),
                                                             "linkage": lk,
                                                             "root_compare": {"result": None, "reference": "not run"}, "replay": {"result": None, "reference": "not run"}}}
                covp.write_text(json.dumps(cj, indent=1, default=str) + "\n")
        except Exception as e:  # noqa: BLE001
            print(f"[tp2commit] coverage.json linkage annotate failed: {e!r}", flush=True)

    # the C2 OF RECORD components, whatever else ran this attempt: components.sampled_replay +
    # components.boundary_linkage (per rank, AND-ed), the value_correspondence result AND-ed with them, the method naming the grade
    sr_comp, lk_comp = C2.sampled_replay_components(rows, versions)
    wp_comp = C2.weights_pin_component(rows, versions)
    from verity_vllm.check.match import xrank_collectives as _XC
    xr_comp = _XC.fold_pairs(xrank_pairs) if xrank_pairs else {"result": None, "pairs": 0, "why": "no instrumented pair"}
    xr_comp["per_pair_artifacts"] = [str(out / f"xrank_pair{p}.json") for p in range(len(xrank_pairs)) if (out / f"xrank_pair{p}.json").exists()]
    (out / "xrank_collectives.json").write_text(json.dumps({"schema": "verity/tp2/xrank-collectives-fold/v1", "tp": args.tp} | xr_comp, indent=1, default=str) + "\n")
    (out / "weights_pin.json").write_text(json.dumps({"schema": "verity-sweep/tp-weights-pin/v1", "tp": args.tp} | wp_comp, indent=1, default=str) + "\n")
    # the peer-input BINDER: the fold Match's peer_binding digests (fold peers_r[s,k] leaf == recorder
    # partial of rank 1-r at (s,k)) against the recorder rows the match oracle compared the COMMITTED partials to (bit-exact) -- by value,
    # both directions, every (step, k) of every rank.  False -> FAIL named; absent -> None (◐) unless --require-fold-match (then FAIL named)
    from verity_vllm.check.match import rank_match as _RM
    fb_comp = (_RM.commit_binding(args.match_dir, world=world, require=args.require_fold_match,
                                  match_oracle=(None if match_capture_dir is None else {str(r_): (c.get("match_oracle") or {}) for r_, c in C2.last_per_rank(rows).items()}))
               if args.match_dir else {"result": (False if args.require_fold_match else None), "why": "no --match-dir: no fold Match record to bind the peer inputs to" + (" -- REQUIRED" if args.require_fold_match else "")})
    (out / "fold_match_binding.json").write_text(json.dumps({"tp": args.tp} | fb_comp, indent=1, default=str) + "\n")
    print(f"[tp2commit] fold-Match peer binding (D90, R1 at Commit): result {fb_comp.get('result')} bindings {fb_comp.get('n_bindings')} recorder {fb_comp.get('n_recorder')} "
          f"bad {fb_comp.get('n_bad_bindings')} {fb_comp.get('why') or ''}", flush=True)
    if fb_comp.get("result") is False:
        all_pass = False
    qp_comp = C2.query_population_component(rows, replay_cfg is not None)
    print(f"[tp2commit] C2 population vs Q(P): result {qp_comp['result']} per rank "
          f"{ {r_: ((b or {}).get('ok'), (b or {}).get('counts')) for r_, b in sorted(qp_comp['per_rank'].items())} } {qp_comp.get('why') or ''}", flush=True)
    all_pass_but_c2 = all_pass
    if qp_comp["result"] is False:
        all_pass = False
    vc_base = None
    try:
        covp = out / "coverage.json"
        if covp.exists():
            cj = json.loads(covp.read_text())
            vc_base = json.loads(json.dumps(cj.get("value_correspondence"), default=str))
            cj["value_correspondence"] = C2.annotate_value_correspondence(cj.get("value_correspondence"), sr_comp, lk_comp, qp_comp, wp_comp, xr_comp, fb_comp,
                                                                       replay_on=replay_cfg is not None)
            covp.write_text(json.dumps(cj, indent=1, default=str) + "\n")
    except Exception as e:  # noqa: BLE001
        print(f"[tp2commit] coverage.json sampled_replay annotate failed: {e!r}", flush=True)
    if replay_cfg is not None:
        (out / "sampled_replay.json").write_text(json.dumps({"schema": "verity-sweep/tp-sampled-replay/v1", "tp": args.tp, "sampled_replay": sr_comp, "boundary_linkage": lk_comp,
                                                             "per_pair_artifacts": [str(out / f"finalize_pair{r['pair']}.json") for r in rows if r["arm"] != "control"],
                                                             "note": "per-rank full records (strata, mismatches, programs) live in finalize_pair<k>.json[rank].sampled_replay",
                                                             "unreplayable_collectives_evaluated_by": {"pass": "cross_rank_collectives (TP-12)", "families": sorted(set(xrank_families) | {u["family"] for u in sr_comp.get("unreplayable") or [] if u.get("evaluated_by")}),
                                                                                                        "result": xr_comp.get("result"), "by_family": xr_comp.get("by_family"), "artifact": "xrank_collectives.json"}},
                                                            indent=1, default=str) + "\n")

    ctl = [r for r in rows if r["arm"] == "control"]
    trt = [r for r in rows if r["arm"] != "control"]
    deltas = [t["wall_s"] - c["wall_s"] for t, c in zip(trt, ctl)]
    summ = {"schema": SCHEMA, "commit_started": started, "label": args.label, "tp": args.tp, "committer": args.committer, "pairs": args.pairs, "openings": args.openings,
            "pairs_run": len({r["pair"] for r in rows}), "stopped": stopped,
            "control_wall": [r["wall_s"] for r in ctl], "treatment_wall": [r["wall_s"] for r in trt], "delta_wall": deltas,
            "delta_wall_mean": statistics.fmean(deltas) if deltas else None,
            "delta_wall_pct": (statistics.fmean(deltas) / statistics.fmean(r["wall_s"] for r in ctl) * 100) if deltas else None,
            "tp_run_roots": [(r.get("commit") or {}).get("tp_run_root") for r in trt],
            "per_rank_roots": [{k: v["run_root"] for k, v in ((r.get("commit") or {}).get("per_rank") or {}).items()} for r in trt],
            "per_rank_coverage": [{k: {"leaves": v["leaves"], "tensors": v["tensors"], "bytes_bound": v["bytes_bound"], "classes": v["classes"], "openings": v["openings"]}
                                   for k, v in ((r.get("commit") or {}).get("per_rank") or {}).items()} for r in trt],
            "tokens_equal": [r["tokens_equal_ref"] for r in rows], "commit_pass": [(r.get("commit") or {}).get("commit_pass") for r in trt],
            "value_check": (None if vcheck is None else {"verdict": vcheck["verdict"], "per_rank": {r_: v.get("verdict") for r_, v in vcheck["per_rank"].items()}}),
            "attribution": (None if match_tokens is None else {"match_record": versions.get("match_record"),
                                                               "per_rank": {r_: {k2: (c.get("attribution") or {}).get(k2) for k2 in ("ok", "steps_checked", "rows_checked", "mismatches_n", "why")}
                                                                            for r_, c in C2.last_per_rank(rows).items()}}),
            # committed TP partials / MoE received outputs vs the Match capture's stored collectives, per rank
            "match_oracle": (None if match_capture_dir is None else {r_: {k2: (c.get("match_oracle") or {}).get(k2) for k2 in TPS_SUMMARY_KEYS}
                                                                     for r_, c in C2.last_per_rank(rows).items()}),
            "oracle_compare": C2.oracle_compare_component(rows, versions, man_doc),
            # the C2 of record, form (A) per rank, folded over ranks and pairs (full records in finalize_pair<k>.json)
            "sampled_replay": (None if replay_cfg is None else {k: sr_comp.get(k) for k in C2.SR_SUMMARY_KEYS}),
            "boundary_linkage": (None if replay_cfg is None else lk_comp),
            "cross_rank_collectives": xr_comp,
            # fold peer leaves bound by digest to the recorder rows the committed partials were compared against
            "fold_match_binding": {k: fb_comp.get(k) for k in ("result", "n_bindings", "n_recorder", "n_bad_bindings", "why")},
            # the replay population reconciled against Q(P) per rank (population.query_checks), in the verdict
            "query_population": qp_comp,
            "weights_pin": {k: wp_comp.get(k) for k in ("result", "per_rank", "why", "checkpoint_dir")},
            "required_manifest": (None if man_doc is None else {"path": args.required_manifest, "manifest_digest": man_doc.get("manifest_digest"),
                                                                "identities": (man_doc.get("populations") or {}).get("identities"), "program_digest_per_rank": man_doc.get("program_digest_per_rank")}),
            "negatives": versions["negatives"], "served": versions["served"],
            "pass": all_pass and all(r["tokens_equal_ref"] for r in rows), "required_classes": list(required_classes)}
    if args.only_arm:
        summ.update(TPC.only_arm_summary(args.only_arm, rows, trt, summ["commit_pass"]))
    write_import_path_runtime(llm, out, args.rpc_timeout)
    # a NEGATIVE that did not inject is neither a detection nor a miss -- it is an ERROR and must never be
    # recorded as a PASS (the ':ulp' flip spec on a tree without parse_flip left the site unmatched -> flipped 0 -> an honest PASS that a
    # finalizer would read as 'negative not detected').  The ranks report what they applied; when the requested negative did not fire the
    # treatment rows' commit_pass becomes None (not False: not a detection) with the reason, summary.pass None, exit 3.
    applied = negatives_applied(args.flip_partial, args.swap_requests, args.omit_partial_site, xrank_pairs, made)
    summ["negative_applied"] = applied or None
    not_applied = sorted(k_ for k_, v_ in applied.items() if v_["applied"] is not True)
    if not_applied:
        summ["pass"], summ["negative_not_applied"] = None, not_applied
    if deferral_skipped:
        summ["replay_deferred"] = {"skipped": deferral_skipped}
    if args.replay_deferred:
        C2.seal(out, summ, rows, bundles_of, replay_cfg=replay_cfg, required_manifest=args.required_manifest, world=world, versions=versions,
                 all_pass=all_pass_but_c2, vc_base=vc_base, comps={"weights_pin": wp_comp, "cross_rank_collectives": xr_comp, "fold_match_binding": fb_comp},
                 xrank_pairs=xrank_pairs, xrank_families=sorted(xrank_families), label=args.label)
    if not_applied:
        why_na = f"NEGATIVE NOT APPLIED: {not_applied} -- {json.dumps(applied, default=str)[:400]}; the run is an error, not a verdict"
        try:
            fixed = []
            for l_ in (out / "runs.jsonl").read_text().splitlines():
                if not l_.strip():
                    continue
                r_ = json.loads(l_)
                if r_.get("arm") != "control" and isinstance(r_.get("commit"), dict):
                    r_["commit"]["commit_pass"], r_["commit"]["commit_fail_reason"] = None, why_na
                fixed.append(json.dumps(r_, default=str))
            (out / "runs.jsonl").write_text("\n".join(fixed) + "\n")
        except Exception as e:  # noqa: BLE001
            print(f"[tp2commit] runs.jsonl rewrite failed: {e!r}", flush=True)
        (out / "summary.json").write_text(json.dumps(summ, indent=1, default=str) + "\n")
        print(f"[tp2commit] {why_na}", flush=True)
        return 3
    (out / "summary.json").write_text(json.dumps(summ, indent=1, default=str) + "\n")
    verdict = "PENDING" if summ["pass"] is None and summ.get("replay_deferred") else "PASS" if summ["pass"] else "FAIL"
    print(f"[tp2commit] {verdict} delta {summ['delta_wall_mean']} s ({summ['delta_wall_pct']}%) roots {[x[:16] if x else None for x in summ['tp_run_roots']]}"
          + (f"; the C2 replay of every rank runs on CPU from {[b['path'] for b in summ['replay_deferred']['bundles']]}" if verdict == "PENDING" else "")
          + (f"; stopped after {stopped['after']}, not run {stopped['not_run']}" if stopped else ""), flush=True)
    return 0 if verdict != "FAIL" else 1
