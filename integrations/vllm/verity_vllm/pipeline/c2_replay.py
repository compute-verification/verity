"""The C2 of record of one instrumented run: the sampled exact VU replay + the boundary linkage over its committed store.

One implementation for both places it runs: the Commit, in process, right after the committed run (`pipeline.commit`), and the CPU
replay of a deferred Commit's bundle (`pipeline.replay_bundle`, `row stage replay`).  What differs between them is passed in on the
`ReplaySources`: the committed store to read (the committer, or the bundle's `commit.store_dump`), the weights provider (the live model, or the
checkpoint opened against the committed weights root), the weights check and the VU export hook.

For a seeded stratified sample of Program instance rows (spec x request x phase), read
the COMMITTED bytes of the row's inputs (the Program's dataflow -> the manifest's member rule -> the binding map), evaluate the registered
relation (verity.ml.definitions.numerics.derived_rows), compare with the COMMITTED output bytes bit-for-bit.  Plus the exhaustive ordered
boundary linkage (declared prompt -> input ids; sampled id[s-1] -> input id[s]) over every (engine step, request).  Needs no Match record.
"""
from __future__ import annotations

import copy
import gc
import hashlib
import inspect
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from verity_vllm.check.replay.linkage import splits_of_record
from verity_vllm.commit import challenge as CH
from verity_vllm.commit import opened as OV
from verity_vllm.commit import store_dump as SD


class ReplayReused(Exception):
    """Control flow: this pair's C2 record is the cached one (see --replay-cache)."""


def rss_kb() -> tuple[int | None, int | None]:
    try:
        cur = hwm = None
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                cur = int(line.split()[1])
            elif line.startswith("VmHWM:"):
                hwm = int(line.split()[1])
        return cur, hwm
    except Exception:  # noqa: BLE001
        return None, None


@dataclass
class ReplaySources:
    """Everything the C2 replay of one instrumented run reads.  `weights_of(prefix)`: the weights provider for the Programs' field
    prefix; `weights_check()`: the proof that the slices it serves are the committed weights ({"equal": bool, ...});
    `after_replay`: the VU export hook (in-process only), called as VUX.after_replay is.  `reads`: when a dict, the replay's draw is made
    again after it, read only, in this process (`plan`), and its record lands here: a forked evaluator's reads of `com` never reach this
    process, so this is what makes `commit.store_dump.touched` see every pick's."""
    args: Any
    out: Path
    pair: int
    com: Any
    rc: Any
    row: dict[str, Any]
    reqs: list[dict[str, Any]]
    sampling: dict[str, Any]
    tokens: dict[str, Any]
    manifest: Any
    stop_of_record: dict[str, Any]
    ids_full: Any
    versions: dict[str, Any]
    components: dict[str, Any]
    vc_result: Any
    weights_of: Callable[[str], Callable]
    weights_check: Callable[[], dict[str, Any]]
    tl: Any
    after_replay: Callable | None = None
    protocols: Any = None
    reads: dict[str, Any] | None = None


def live_weights(model: Any, prefix: str) -> Callable:
    from verity_vllm.check.replay import driver as SR

    return SR.model_weights_provider(model, prefix)


def vu_export(args: Any, pair: int, rc: Any, manifest: Any, versions: dict[str, Any], eng: Any) -> Callable:
    """The in-process replay's VU export hook (`pipeline.vu_store.after_replay`)."""
    from verity_vllm.pipeline import vu_store as VUX

    return lambda sr, progs, bm, rd, wp, arr, sr_kw, loaded, digs: VUX.after_replay(
        args.vu_export_dir, sr, pair, rc.run_root, progs, bm, rd, wp, arr, manifest, sr_kw, loaded, digs, versions, eng)


def live_weights_check(register: Callable[[], dict[str, Any]], com: Any, rc: Any) -> dict[str, Any]:
    """[O5] the weight slices the relations consume must be the COMMITTED weights: re-register the live parameters now (`register`) and
    compare the root with the run's weights root (registered before the workload; a param swap after it is a named FAIL)."""
    try:
        _w_saved = getattr(com, "weights", None)
        _w_now = register()     # through what a protocol option released
        if _w_saved is not None:
            com.weights = _w_saved
        _w_ref = (rc.weight_registration or {}).get("root") if isinstance(rc.weight_registration, dict) else (_w_saved or {}).get("root")
        return {"equal": (str(_w_now.get("root")) == str(_w_ref)) if _w_ref else None, "committed_root": _w_ref, "replay_root": _w_now.get("root"),
                "tensors": _w_now.get("tensors"), "wall_s": _w_now.get("wall_s"), "hash": _w_now.get("hash"),   # time + path
                "method": "live parameters re-registered at replay (register_weights) == the run's committed weights root",
                "why": None if (_w_ref and str(_w_now.get("root")) == str(_w_ref)) else ("no committed weights root on the run" if not _w_ref else "root differs")}
    except Exception as _we:  # noqa: BLE001
        return {"equal": None, "note": f"by reference, unverified at replay: {_we!r}"[:200]}


def cache_key(s: ReplaySources) -> str | None:
    """The replay cache (--replay-cache, default OFF): sampled_replay is a pure function of (committed bytes = the
    run root, the manifest digest, the seed, the replay code); a later pair whose FULL key equals an earlier pair's reuses that pair's
    record (copy carries `reused_from`; `evaluated_here` = 0 so nothing is counted twice).  Any differing component -> its own replay,
    refused by name in the log.  Within THIS Commit stage (this main() call) only."""
    args, rc, required_manifest = s.args, s.rc, s.manifest
    if not getattr(args, "replay_cache", False):
        return None
    try:
        if getattr(args, "replay_seed", None) is not None:
            _ks: Any = int(args.replay_seed)
        elif getattr(args, "replay_challenge", None):
            _ks = CH.challenge_seed(str(args.replay_challenge), rc.run_root)
        else:
            _ks = CH.root_seed(rc.run_root)
        _w_ref_k = (rc.weight_registration or {}).get("root") if isinstance(rc.weight_registration, dict) else None
        return json.dumps({"run_root": rc.run_root.hex(), "manifest_digest": (required_manifest or {}).get("manifest_digest") if isinstance(required_manifest, dict) else None,
                           "program_digest": getattr(args, "program_digest", None), "seed": _ks,
                           "binding_map_sha256": hashlib.sha256((s.out / f"binding_map_p{s.pair}.json").read_bytes()).hexdigest(),
                           "per_stratum": int(args.replay_per_stratum), "max_evaluate": int(args.replay_max), "targets": list(getattr(args, "replay_target", None) or []), "k": args.replay_k,
                           "code": s.versions.get("git_head"), "weights_root": _w_ref_k, "committer": s.com.name, "retain": getattr(args, "retain", None)}, sort_keys=True)
    except Exception as _ke:  # noqa: BLE001  -- no key, no reuse
        print(f"[delta] replay cache: no key for pair {s.pair} ({_ke!r}); replay runs", flush=True)
        return None


def run(s: ReplaySources, replay_cache: dict[str, dict[str, Any]]) -> None:
    """The replay + linkage of the run on `s`, their artifact (`sampled_replay_p<pair>.json`) and their components on `s.row`.  A
    replay this host cannot hold (`driver.PoolDoesNotFit`) is raised, not recorded: it says nothing about the run."""
    from verity_vllm.check.replay.driver import PoolDoesNotFit

    pair, rc = s.pair, s.rc
    _rc_key = cache_key(s)
    _cached = replay_cache.get(_rc_key) if _rc_key is not None else None
    if _rc_key is not None and _cached is None and replay_cache:
        _k0, _v0 = next(iter(replay_cache.items()))
        _d0 = sorted(k for k in json.loads(_k0) if json.loads(_k0)[k] != json.loads(_rc_key).get(k))
        print(f"[delta] replay cache: pair {pair} key differs from pair {_v0['pair']}'s in {_d0}: own replay (root {rc.run_root.hex()[:16]} vs {_v0['run_root'][:16]})", flush=True)
    s.tl.start("validate.sampled_replay", pair=pair, cached=_cached is not None)
    held: dict[str, Any] = {}
    try:
        if _cached is not None:
            raise ReplayReused()
        sr, lk = evaluate(s, held)
        if _rc_key is not None and sr.get("result") is not None:
            replay_cache[_rc_key] = {"pair": pair, "run_root": rc.run_root.hex(), "sr": sr, "lk": lk, "artifact": str(s.out / f"sampled_replay_p{pair}.json"),
                                     "seed": (sr.get("sample") or {}).get("seed"), "code": s.versions.get("git_head")}
    except ReplayReused:
        sr, lk = reused(s, _cached, _rc_key)
    except PoolDoesNotFit:
        raise
    except Exception as e:  # noqa: BLE001  -- fail closed: an unrunnable replay is 'not evaluated', never PASS
        sr = {"schema": "verity-sweep/sampled-replay/v1", "method": "sampled-exact-replay", "result": None, "grade": "not-evaluated",
              "why": f"sampled replay not run: {e!r}"[:300], "error": repr(e)[:300]}
        lk = {"schema": "verity-sweep/boundary-linkage/v1", "result": None, "why": f"boundary linkage not run: {e!r}"[:300], "pairs": 0, "linked": 0}
    _art = s.out / f"sampled_replay_p{pair}.json"
    _art.write_text(json.dumps({"sampled_replay": sr, "boundary_linkage": lk}, indent=1, default=str) + "\n")
    free_programs(sr, held, pair)
    s.tl.end("validate.sampled_replay", pair=pair, ok=sr.get("result"), n=(sr.get("sample") or {}).get("evaluated"), linkage=lk.get("result"),
             reused=("reused_from" in sr))
    record(s, sr, lk, _art)


def reused(s: ReplaySources, _cached: dict[str, Any], _rc_key: str) -> tuple[dict[str, Any], dict[str, Any]]:
    pair, rc = s.pair, s.rc
    _from = {"pair": _cached["pair"], "run_root": _cached["run_root"], "root_equal": rc.run_root.hex() == _cached["run_root"], "seed": _cached["seed"],
             "code": _cached["code"], "artifact": _cached["artifact"], "key_sha256": hashlib.sha256(_rc_key.encode()).hexdigest(),
             "rule": "identical (run_root, manifest_digest, seed, code, binding map, weights root, sample parameters) "
                     "within one Commit stage -> the same pure function; the ORIGINAL pair's record is the record of evaluation"}
    if not _from["root_equal"]:
        raise RuntimeError("replay cache: key matched but the run roots differ -- refused")     # cannot happen (root is in the key); fail closed anyway
    sr = copy.deepcopy(_cached["sr"])
    lk = copy.deepcopy(_cached["lk"])
    sr["reused_from"] = _from
    lk["reused_from"] = _from
    if isinstance(sr.get("sample"), dict):
        sr["sample"]["evaluated_here"] = 0
    _cached.setdefault("shared_by", []).append({"pair": pair, "run_root": rc.run_root.hex(), "root_equal": True,
                                                "artifact": str(s.out / f"sampled_replay_p{pair}.json")})
    print(f"[delta] C2 SAMPLED REPLAY (pair {pair}): REUSED pair {_cached['pair']}'s record (identical key: run root {rc.run_root.hex()[:16]}, "
          f"seed {_cached['seed']}, code {str(_cached['code'])[:9]}); no VU evaluated here", flush=True)
    return sr, lk


def programs_of_record(s: ReplaySources, _rl: dict[str, str], *, definitions: bool = True) -> tuple[dict[str, dict], dict[str, str]]:
    """The population is derived from the Program OF RECORD: every loaded Program's digest must be the Build digest of
    record (--program-digest, B=1) or one of the manifest's component digests (B>=2), and the Definitions this tree evaluates their
    rows by must be the ones their Builds bound (`check.replay.definitions`); else REFUSED by name.  ``definitions=False`` for a
    caller that evaluates no row (the replay's plan, the serving rows): the replay that grades the picks checks them."""
    from verity_vllm.check.replay import definitions as DF
    from verity_vllm.query.manifest import format as MF

    args, required_manifest, reqs, sampling = s.args, s.manifest, s.reqs, s.sampling
    _of_record = set()
    if getattr(args, "program_digest", None):
        _of_record.add(str(args.program_digest))
    if isinstance(required_manifest, dict):
        _of_record.update(str(x) for x in (required_manifest.get("component_digests") or []))
        for _c in (required_manifest.get("components") or {}).values():
            if isinstance(_c, dict) and _c.get("program_digest"):
                _of_record.add(str(_c["program_digest"]))
    if getattr(args, "programs_root", None):
        _fx = {"requests": [{"request_id": _rl[str(r.get("request_id"))], "prompt_len": len(r["prompt_token_ids"]),
                             "max_tokens": int(r.get("max_tokens") or sampling.get("max_tokens") or 1), "cached": int(r.get("cached") or 0)}
                            for r in reqs]}
        _loaded = MF.programs_for(_fx, args.programs_root)
    else:
        _l1 = MF.load_program(args.program_dir)
        _loaded = {_rl[str(r.get("request_id"))]: _l1 for r in reqs}
    _digs = {k: str((v.get("result") or {}).get("program_digest") or (v.get("instances") or {}).get("digest") or "") for k, v in _loaded.items()}
    if not _of_record:
        raise RuntimeError("no Program digest of record to check the replay Programs against (pass --program-digest / --required-manifest)")
    _bad = {k: d for k, d in _digs.items() if d not in _of_record}
    if _bad:
        _named = ", ".join(f"{k}: {d[:16]} ({_loaded[k].get('dir')})" for k, d in _bad.items())
        raise RuntimeError(f"replay Program(s) are not the Program of record: {_named} vs of-record {sorted(x[:16] for x in _of_record)} -- refused")
    if definitions:
        DF.of_record({k: v["dir"] for k, v in _loaded.items()}, _digs, {k: v["instances"]["rows"] for k, v in _loaded.items()}, what="replay")
    return _loaded, _digs


def request_labels(reqs: list[dict[str, Any]]) -> dict[str, str]:
    """Request id -> the Programs' request label (``r<index>``)."""
    return {str(r.get("request_id")): f"r{int(r.get('index', i))}" for i, r in enumerate(reqs)}


def arrivals(s: Any, rl: dict[str, str]) -> tuple[dict[str, int], bool]:
    """Each request's arrival step, and whether it moved off the declared one.  One schedule of record for every consumer: when the
    population was placed on the engine's committed-step axis (binding.coverage.schedule.use == observed), the replay's, the
    linkage's and the serving rows' arrivals are the same observed ones."""
    from verity_vllm.check.replay import driver as SR

    declared = {rl[str(r.get("request_id"))]: int(r.get("arrive_step") or 0) for r in s.reqs}
    sched = (((s.row.get("commit") or {}).get("binding") or {}).get("coverage") or {}).get("schedule") or {}
    return SR.arrivals_of_record(declared, sched)


def seed_of(args: Any, rc: Any) -> tuple[int, str, str, str | None]:
    """[O6] the seed is not prover-choosable in advance: derived from the run root AFTER commitment unless a targeted seed is given."""
    from verity_vllm.check.replay import driver as SR, replay_codes as RC

    _seed_chal = None
    if getattr(args, "replay_seed", None) is not None:
        _seed, _seed_src, _seed_form = int(args.replay_seed), RC.PROVER_SEED_SOURCE, "prover-chosen"
    elif getattr(args, "replay_challenge", None):
        (_seed, _seed_src), _seed_form = SR.challenge_seed(str(args.replay_challenge), rc.run_root), "challenge"
        _seed_chal = str(args.replay_challenge)            # the full challenge goes on the record (sample.challenge)
    else:
        _seed, _seed_src, _seed_form = CH.root_seed(rc.run_root), RC.ROOT_SEED_SOURCE, "run-root"
    return _seed, _seed_src, _seed_form, _seed_chal


def draw_of(args: Any) -> str:
    """The config run's draw (`driver.family_strata`): the record's `replay_draw`, `family` when it names none."""
    return getattr(args, "replay_draw", None) or "family"


def linkage_facts(loaded: dict, progs: dict, index: Any) -> dict[str, Any]:
    """What the linkage reads of the Programs: their steps of record, splits of record and the embedding's op path."""
    return {"program_steps": program_steps(loaded, progs), "splits": splits_of_record(loaded), "embed_op_path": index.embed_op_path()}


def program_steps(loaded: dict, progs: dict) -> dict[str, int | None]:
    """Each request Program's steps of record: its Build result's T + 1, else its rows' segment steps."""
    from verity_vllm.query.manifest import format as MF

    return {k: (int(v["result"]["T"]) + 1) if (v.get("result") or {}).get("T") is not None else (len(MF.segment_steps(progs[k]["rows"])) if k in progs else None)
            for k, v in loaded.items()}


def linkage_requests(s: ReplaySources, _rl: dict[str, str], _arr: dict[str, int], _pi_steps: dict[str, int | None], _horizons: dict | None) -> list[dict[str, Any]]:
    """The DECLARED seed of each request = the SamplingParams.seed the run was asked for (the same per-request /
    sampling-level resolution vllm_adapter.sampling_params uses; None on a greedy row), and the Program's S of record from each request
    Program's Build result -- the prescribed-input linkage reads BOTH from the record, never from the served run.  The linkage horizon is the request's PROGRAM OF RECORD's, not the workload cap: horizon = min(cap,
    Program steps) -- the Program's T from its Build result (result.json T) else its ProgramIndex.n_steps; a request whose committed steps
    are FEWER than its Program of record still fails by name (the chain rule is unchanged inside the horizon).  Recorded per request."""
    from verity_vllm.check.replay import driver as SR

    sampling = s.sampling
    return [{"request_id": _rl[str(r.get("request_id"))], "arrive_step": _arr[_rl[str(r.get("request_id"))]],
             "prompt_token_ids": list(r["prompt_token_ids"]), "cached": int(r.get("cached") or 0),
             "seed": (int(r["seed"]) if r.get("seed") is not None else (int(sampling["seed"]) if sampling.get("seed") is not None else None)),
             "max_tokens": int(r.get("max_tokens") or sampling.get("max_tokens") or 1),
             "program_steps": _pi_steps.get(_rl[str(r.get("request_id"))]),
             # the horizon is the executed prefix of record (cap on a length stop; served + LAG on an EOS stop):
             # boundary_linkage verifies the EOS at steps[-1 - lag] and the placeholder row(s) fed the EOS inside it
             "steps": SR.linkage_horizon(_arr[_rl[str(r.get("request_id"))]], int(r.get("max_tokens") or sampling.get("max_tokens") or 1),
                                         _pi_steps.get(_rl[str(r.get("request_id"))]),
                                         executed=(None if _horizons is None else _horizons.get(_rl[str(r.get("request_id"))])))} for r in s.reqs]


def cached_prefixes(reqs: list[dict[str, Any]], rl: dict[str, str]) -> dict[str, dict[str, Any]]:
    """Each warm request's cached rows and the label of the request that computed each cached block (`load_workload`: `cached`,
    `cached_from`): what the replay resolves its `prefix` reads to (`driver.populate`)."""
    return {rl[str(r.get("request_id"))]: {"cached": int(r["cached"]), "producers": [rl[str(q)] for q in r.get("cached_from") or ()]}
            for r in reqs if int(r.get("cached") or 0)}


def executed_prefix_kw(s: ReplaySources, _epr_r: dict, lk: dict, _arr: dict, _rl: dict) -> tuple[dict[str, Any], Any]:
    """ONE mechanism for the not-executed class (`commit.executed_prefix`): sampled_replay's
    `population.not_executed_by_rule`, fed with named record facts (`EPX.executed_prefix_facts`) -- never literals.  On a tree without
    the `executed_prefix=` kwarg nothing bounds the replay population: an EOS-stopped request's rows past the stop are sampled and grade 'not
    evaluated' -> partial, never PASS.  An older signature receives only the arithmetic's facts."""
    from verity_vllm.commit import executed_prefix as EPX
    from verity_vllm.check.replay import driver as SR

    _sr_kw: dict[str, Any] = {}
    _ep_facts = EPX.executed_prefix_facts(_epr_r, s.row["commit"].get("binding") or {}, lk, _arr, s.stop_of_record,
                                          served_tokens={_rl[str(k)]: [int(t) for t in v] for k, v in s.tokens.items() if str(k) in _rl})
    if hasattr(SR, "executed_prefix_of_record") and "executed_prefix" in inspect.signature(SR.sampled_replay).parameters:
        _ep_facts = EPX.facts_accepted_by(SR.executed_prefix_of_record, _ep_facts)
        _sr_kw["executed_prefix"] = {} if s.args.replay_k and not _epr_r["any_eos_stop"] else {k: SR.executed_prefix_of_record(k, **f) for k, f in _ep_facts.items()}
        _flt = {k: v["faults"] for k, v in _sr_kw["executed_prefix"].items() if v.get("faults")}
        if _epr_r["any_eos_stop"] or _flt:
            print("[delta] not-executed class -> sampled_replay(executed_prefix=): " + ", ".join(
                f"{k} a_r {v['a_r']} n_exec {v['n_exec']} cap {v['cap']}" for k, v in _sr_kw["executed_prefix"].items())
                  + (f"; FAULTS {_flt}" if _flt else "; no faults"), flush=True)
    elif _epr_r["any_eos_stop"]:
        print("[delta] WARNING: this tree's sampled_replay has no executed_prefix=: the EOS-stopped "
              "request's rows past the stop are sampled as ordinary strata and grade 'not evaluated' -> C2 partial, never PASS", flush=True)
    return _sr_kw, _ep_facts


def live_only(s: ReplaySources) -> str | None:
    """The protocol entry whose replay reads the live run (`pouw_circuit.around`): such a run's C2 cannot be deferred to a bundle."""
    entries = getattr(s.protocols, "entries", None) or ()
    if not entries:
        return None
    from verity_vllm.check.replay import pouw_circuit as PWC

    return next((e for e in entries if e.split(":", 1)[0] == PWC.pouw.NAME and e.split(":", 1)[1] in PWC.PC.C.SCHEMES), None)


def plan(s: ReplaySources, planner: Any = None) -> dict[str, Any]:
    """The replay's and the linkage's committed reads made against `s.com` without evaluating anything (what a slim replay bundle must
    hold, `commit.store_dump.touched`): the same Programs, readers and draw as `run`.  With a ``planner`` (`replay_plan.Planner`) the
    Programs, their population and the draw are its, prepared while the engine ran; only the reads are made here."""
    held: dict[str, Any] = {}
    sr, lk = evaluate(s, held, plan=True, planner=planner)
    held.clear()
    gc.collect()
    return {"replay": sr, "linkage": lk.get("result")}


def serving_rows(s: ReplaySources) -> None:
    """The in-Commit hook (`serving_rows.readers`) at the C2 site of a run whose replay is deferred to its bundle: the window
    committed from `s.com` over the Programs of record, as `evaluate` commits it in process, once the run's roots are final and
    before any draw (the deferred replay draws on CPU, after the bundle is sealed).  ``served``, as the in-process replay's is."""
    from verity_vllm.check.replay import driver as SR
    from verity_vllm.pipeline import serving_rows as SRW

    bm = json.loads((s.out / f"binding_map_p{s.pair}.json").read_text())
    rl = request_labels(s.reqs)
    loaded, digs = programs_of_record(s, rl, definitions=False)
    progs = {k: v["instances"] for k, v in loaded.items()}
    arr, _moved = arrivals(s, rl)
    k0 = next(iter(progs))
    prefix = SR.ProgramIndex(progs[k0], program_dir=loaded[k0].get("dir")).prefix
    SRW.readers(s.args.serving_rows, s.com, s.rc, s.weights_of(prefix), progs, bm, arr, loaded, s.manifest, s.args.label, digs)


def evaluate(s: ReplaySources, held: dict[str, Any], *, plan: bool = False, planner: Any = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """The replay + linkage records; the Programs and readers they read stay in `held` for `free_programs` to release and measure.
    ``plan``: the draw's reads only (`plan`), with the Programs and the draw the ``planner``'s when there is one."""
    from verity_vllm.commit import executed_prefix as EPX
    from verity_vllm.check.replay import driver as SR

    args, rc, reqs, sampling, row, pair = s.args, s.rc, s.reqs, s.sampling, s.row, s.pair
    held["bm"] = _bm = json.loads((s.out / f"binding_map_p{pair}.json").read_text())
    _rl = request_labels(reqs)
    facts = planner.ready() if planner is not None else None
    held["loaded"], held["digs"] = _loaded, _digs = programs_of_record(s, _rl, definitions=not plan) if facts is None else ({}, {})
    held["progs"] = _progs = {k: v["instances"] for k, v in _loaded.items()}
    _arr, _moved = arrivals(s, _rl)
    if _moved:
        _sched_r = (((row.get("commit") or {}).get("binding") or {}).get("coverage") or {}).get("schedule") or {}
        print(f"[delta] replay/linkage arrivals follow the engine schedule of record: {_arr} (declared {_sched_r.get('declared_arrive_steps')})", flush=True)
    if facts is None:
        _k0 = next(iter(_progs))
        held["pi0"] = _pi0 = SR.ProgramIndex(_progs[_k0], program_dir=_loaded[_k0].get("dir"))
        facts = linkage_facts(_loaded, _progs, _pi0)
    from verity_vllm.pipeline import serving_rows as SRW
    _wp, held["rd"] = SRW.readers(args.serving_rows, s.com, rc, s.weights_of(held["pi0"].prefix) if "pi0" in held else None, _progs, _bm, _arr,
                                  _loaded, s.manifest, args.label, _digs)
    if not plan and getattr(args, "memory_map", None):
        from verity_vllm.acquire.sources.kv_slots_source import caches_of
        from verity_vllm.check.oracle_compare import OpenedReader
        from verity_vllm.pipeline import memory_map as MMP
        MMP.after_commit(args.memory_map, loaded=_loaded, bmap=_bm, committed=OpenedReader(s.com, rc), manifest=s.manifest, arrivals=_arr,
                         caches=caches_of(s.com), session=s.out / MMP.SESSION, cached_prefixes=cached_prefixes(reqs, _rl))
    _rd = held["rd"]
    _wc = None if plan else s.weights_check()
    _seed, _seed_src, _seed_form, _seed_chal = seed_of(args, rc)
    # the committed store's retention decides the evaluator: DEVICE-retained blocks cannot be read by forked children
    # (CUDA after fork) -> sampled_replay runs serial in this process; named in the replay record
    _retention = retention_of(s.com, args)
    # the executed prefix of record of every request bounds the replay population and the linkage horizon; a
    # problem there leaves every horizon None (the cap's rows are then sampled and 'not evaluated' -> partial)
    _epr_r = EPX.executed_prefix_of_record(sorted(_arr, key=lambda k: int(k[1:])), {_rl[str(k)]: [int(t) for t in v] for k, v in s.tokens.items() if str(k) in _rl},
                                           caps={_rl[str(q.get("request_id"))]: int(q.get("max_tokens") or sampling.get("max_tokens") or 1) for q in reqs},
                                           eos_of_record=s.stop_of_record.get("eos"), lag=s.stop_of_record.get("lag"))
    _horizons = ({k: int(v["executed_prefix_len"]) for k, v in _epr_r["requests"].items()} if _epr_r["ok"] else None)
    if _epr_r["any_eos_stop"]:
        print(f"[delta] replay/linkage horizons = the executed prefix of record: {_horizons} (LAG {_epr_r['lag']}; "
              f"{'ok' if _epr_r['ok'] else 'PROBLEMS ' + str(_epr_r['problems'][:2])})", flush=True)
    _wreq = linkage_requests(s, _rl, _arr, facts["program_steps"], _horizons)
    _spr = facts["splits"]
    # the Program's declared stop rule + LAG = stop_rule_of_record; None = undeclared / unattested ->
    # a request linked over fewer steps than its cap fails closed by name -- never the engine value as the rule
    _eos, _lag = s.stop_of_record.get("eos"), s.stop_of_record.get("lag")
    lk = SR.boundary_linkage(_bm, _rd, _wreq, embed_op_path=facts["embed_op_path"], splits_of_record=_spr, eos_of_record=_eos, eos_lag_rows=_lag)
    _sr_kw, _ep_facts = executed_prefix_kw(s, _epr_r, lk, _arr, _rl)
    _kw = dict(arrivals=_arr, per_stratum=int(args.replay_per_stratum), seed=_seed, seed_source=_seed_src,
               seed_challenge=_seed_chal,
               max_evaluate=int(args.replay_max), weights_check=_wc, targets=list(getattr(args, "replay_target", None) or []), replay_k=args.replay_k,
               replay_draw=draw_of(args), workers=getattr(args, "replay_workers", None), retention=_retention,
               ring_slots=(int(args.window_slots) if getattr(args, "window_slots", None) and getattr(args, "window_mb", None) else None),
               # Q(P) of record: the replay's reach is reconciled against it identity by identity
               manifest=(s.manifest if isinstance(s.manifest, dict) else None),
               worker_peak=getattr(args, "replay_worker_peak", None),
               program_dirs={_k: _v["dir"] for _k, _v in _loaded.items() if _v.get("dir")}, cached_prefixes=cached_prefixes(reqs, _rl), **_sr_kw)
    if planner is not None:
        t0 = time.perf_counter()
        got = planner.plan({"binding_map": str(s.out / f"binding_map_p{pair}.json"), "arrivals": _arr, "seed": _seed, "seed_source": _seed_src,
                            "seed_challenge": _seed_chal, "sr_kw": _sr_kw})
        t1 = time.perf_counter()
        SD.touch(s.com, got["reads"])
        return got["plan"] | {"prepare_s": facts.get("prepare_s"), "reads": len(got["reads"]), "draw_s": round(t1 - t0, 3),
                              "reads_s": round(time.perf_counter() - t1, 3)}, lk
    sr = (lambda f: f() if plan or s.protocols is None else replayed(s.protocols, f))(lambda: SR.sampled_replay(_progs, _bm, _rd, _wp, plan=plan, **_kw))
    if plan:
        return sr, lk
    if s.reads is not None:
        s.reads.update(SR.sampled_replay(_progs, _bm, _rd, _wp, plan=True, **_kw))
    sr["sample"]["seed_form"] = _seed_form            # replay_codes.SEED_FORMS: the verdict reads the form, never the source text
    sr["programs"] = {k: {"dir": _loaded[k].get("dir"), "program_digest": _digs[k]} for k in _loaded}
    sr["executed_prefix_of_record"] = {k: {kk: vv for kk, vv in v.items() if kk in ("served_output_len", "executed_prefix_len", "cap", "cap_exhausted", "eos_stop")}
                                       for k, v in _epr_r["requests"].items()} | {"lag": _epr_r["lag"], "ok": _epr_r["ok"], "problems": _epr_r["problems"][:3],
                                                                                  "facts_of_record": _ep_facts, "mechanism": ("population.not_executed_by_rule" if "executed_prefix" in _sr_kw else "none in this tree (rows past the stop sampled as ordinary strata)")}
    reconcile(s, sr, _bm, _rd, _epr_r, _sr_kw)
    sr["value_source"] = dict(_rd.record(), forked_children=sr.pop("value_reads_in_children", None))
    print(f"[delta] C2 SAMPLED REPLAY + LINKAGE opened values (pair {pair}): {OV.filed(row['commit'], 'replay_linkage', sr['value_source'])}", flush=True)
    if s.after_replay is not None:
        s.after_replay(sr, _progs, _bm, _rd, _wp, _arr, _sr_kw, _loaded, _digs)
    return sr, lk


def replayed(protocols: Any, run: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    from verity_vllm.check.replay import pouw_circuit as PWC

    return PWC.replayed(protocols, run)


def retention_of(com: Any, args: Any) -> str | None:
    _blocks = getattr(com, "_gpu_blocks", None) or {}
    return ("device" if any(b is not None and bool(getattr(b, "is_cuda", False)) for b, _u in _blocks.values())
            else (str(args.retain) if getattr(args, "retain", None) else ("host" if _blocks else None)))


def reconcile(s: ReplaySources, sr: dict, _bm: dict, _rd: Any, _epr_r: dict, _sr_kw: dict) -> None:
    """ONE mechanism, reconciled on the record: the Commit's classification of the manifest identities
    (binding coverage) and the population's third bucket must name the same identities -- n_exec per request, count per row family; a
    batch-level / named-input family is outside the population split's reach and is named, not equal."""
    from verity_vllm.commit import executed_prefix as EPX

    row, pair = s.row, s.pair
    if getattr(s.com, "_padding", None):
        # a PADDED Commit: the population of record is the FULL cap; the
        # identities the replay excluded at n_exec are the padding's -- the named stratum `inactive_by_enablement`, every entry read back
        # == the bot word; the one-mechanism reconciliation compares the Commit's padding classification against that bucket family by family
        from verity_vllm.commit import padding_steps as _PSm
        _pp = _PSm.padded_population(sr, _bm, _rd.or_none, _epr_r)
        # the two documents agree on one number: oracle_compare.padding (by family, pairs) == this stratum
        _pp["oracle_c3"] = _PSm.padding_vs_oracle(_pp, row["commit"].get("oracle_compare"), _bm)
        row["commit"]["inactive_by_enablement"] = _pp
        print(f"[delta] inactive-by-enablement stratum (pair {pair}): {'OK' if _pp['ok'] else 'FAIL'} -- {_pp['padding_entries']} padding entries, "
              f"{_pp['checked_entries']} read back / {_pp['checked_words']:,} words == bot, {_pp['mismatches_n']} mismatches, {_pp['unread_n']} unread; "
              f"population inactive {_pp['population_inactive_identities']}; by family "
              + ", ".join(f"{f} {d['commit_padding_entries']}/{d['population_inactive']}" for f, d in _pp["by_family"].items())
              + (f"; PROBLEMS {_pp['problems'][:3]}" if _pp["problems"] else ""), flush=True)
        if not _pp["ok"]:
            _cv = row["commit"].get("coverage")
            if isinstance(_cv, dict) and _cv.get("ok"):
                _cv["ok"], _cv["error"] = False, "inactive stratum: " + "; ".join(_pp["problems"][:3])
    elif "executed_prefix" in _sr_kw and _epr_r["any_eos_stop"]:
        _cc = (((row["commit"].get("binding") or {}).get("population") or {}).get("manifest") or {}).get("not_executed_by_rule")
        _recon = EPX.reconcile_not_executed(_cc, (sr.get("population") or {}).get("not_executed_by_rule"), _epr_r, manifest_identities=s.ids_full)
        sr["not_executed_reconciliation"] = _recon
        row["commit"]["not_executed_reconciliation"] = _recon
        print(f"[delta] not-executed reconciliation (pair {pair}): {'AGREE' if _recon['ok'] else 'DISAGREE'} n_exec "
              + ", ".join(f"{k} {v['commit']}/{v['population']}" for k, v in _recon["n_exec"].items())
              + f"; row families {[(f, d['commit'], d['population']) for f, d in _recon['by_family'].items()]}"
              + (f"; outside the population split's reach {[(f, d['commit'], d['population_N_excluded']) for f, d in _recon['outside_population_reach'].items()]}"
                 if _recon["outside_population_reach"] else "")
              + (f"; PROBLEMS {_recon['problems'][:3]}" if _recon["problems"] else ""), flush=True)
        if not _recon["ok"]:
            _cv = row["commit"].get("coverage")
            if isinstance(_cv, dict) and _cv.get("ok"):            # in place: row["commit"]["coverage"] and rc.extra["coverage"] alias one dict
                _cv["ok"], _cv["error"] = False, "one mechanism: " + "; ".join(_recon["problems"][:3])


def free_programs(sr: dict, held: dict[str, Any], pair: int) -> None:
    """The Programs of record loaded for the replay (B8: 8 Programs, 7.85 M instances = 70,300 MiB
    of Python objects) and the binding map are dead once the replay + linkage records are written -- free them HERE, before the
    openings-after-release replay pool and the next pair's run.  The RSS drop is the evidence that the premise
    `c2_freed_before_replay` holds: 70 GB of small objects die into pymalloc arenas and glibc's heap, which keep pages the OS never sees
    back until trimmed -- ask glibc to return them (madvise/brk) so the drop measures the objects, not the allocators' retention."""
    _rss_c2a = rss_kb()[0]
    held.clear()
    gc.collect()
    _rss_c2g = rss_kb()[0]
    _trim = None
    try:
        import ctypes

        _trim = int(ctypes.CDLL("libc.so.6").malloc_trim(0))
    except Exception as e:  # noqa: BLE001  -- not glibc: the gc-only delta stands, named
        _trim = repr(e)[:80]
    _rss_c2b = rss_kb()[0]
    sr["programs_freed"] = {"rss_before_kb": _rss_c2a, "rss_after_gc_kb": _rss_c2g, "rss_after_kb": _rss_c2b, "freed_mib": (_rss_c2a - _rss_c2b) >> 10,
                            "freed_by_gc_mib": (_rss_c2a - _rss_c2g) >> 10, "freed_by_malloc_trim_mib": (_rss_c2g - _rss_c2b) >> 10, "malloc_trim": _trim,
                            "when": "after the sampled-replay + boundary-linkage records, before openings-after-release"}
    print(f"[delta] C2 PROGRAMS FREED (pair {pair}): RSS {_rss_c2a >> 10} -> {_rss_c2g >> 10} (gc) -> {_rss_c2b >> 10} MiB (malloc_trim {_trim}): "
          f"{(_rss_c2a - _rss_c2b) >> 10} MiB back before the openings-after-release replay", flush=True)


def record(s: ReplaySources, sr: dict, lk: dict, _art: Path) -> None:
    """The replay's and the linkage's records and components on the run's row (graded by finalize.grade_sampled_replay), and their lines."""
    row, pair, components = s.row, s.pair, s.components
    row["commit"]["sampled_replay"] = {k: v for k, v in sr.items() if k not in ("strata",)} | {"artifact": str(_art), "strata_n": len(sr.get("strata") or {})}
    row["commit"]["boundary_linkage"] = lk
    vc = row["commit"]["value_correspondence"]
    vc["method"] = f"sampled-exact-replay({sr.get('grade')}) + boundary-linkage + " + vc["method"]
    _pop, _smp = sr.get("population") or {}, sr.get("sample") or {}
    components["sampled_replay"] = {"result": sr.get("result"), "grade": sr.get("grade"), "why": sr.get("why"),
                                    "population": {"n": _pop.get("vus"), "source": _pop.get("source"), "by_family": _pop.get("by_family"),
                                                   "not_evaluable_by_family": _pop.get("not_evaluable_by_family")},
                                    "n": _smp.get("evaluated"), "equal": _smp.get("equal"), "mismatch_n": _smp.get("mismatch_n"), "picked": _smp.get("picked"),
                                    "per_stratum": _smp.get("per_stratum"), "strata": sr.get("strata_keys"), "strata_n": len(sr.get("strata") or {}),
                                    "strata_unevaluated": sr.get("strata_unevaluated"), "seed": _smp.get("seed"), "seed_source": _smp.get("seed_source"),
                                    "unreplayable": sr.get("unreplayable"), "weights": sr.get("weights"), "programs": sr.get("programs"),
                                    "evaluators": sr.get("evaluators"), "artifact": str(_art),
                                    "reference": "registered relation evaluators (verity.ml.definitions.numerics.derived_rows) on the COMMITTED input bytes (C2 of record)"}
    components["boundary_linkage"] = {"result": lk.get("result"), "method": lk.get("method"), "checked": lk.get("pairs"), "linked": lk.get("linked"),
                                      "unlinked_n": lk.get("unlinked_n"), "unlinked": lk.get("unlinked"), "why": lk.get("why"),
                                      "prescribed": lk.get("prescribed"),      # seed / splits linkage record
                                      "reference": "declared workload prompt ids (arrival step) + the committed sampled ids of the previous step (ordered interfaces)"
                                                   " + declared per-request seed / SplitsFor_v1(committed live count, num_SMs of record) (prescribed inputs)"}
    components["negatives"] = {"suite": "tests/check/test_sampled_replay.py + tests/commit/test_binding_negatives.py",
                               "results": "unit: flipped committed byte, equal-shape request swap, wrong member, missing committed input (partial), unlinked decode "
                                          "chain, foreign prompt, verdict gates"}
    for _c in (sr, lk):
        if _c.get("result") is False:
            vc["result"] = False
    if vc["result"] is None and sr.get("result") is True and lk.get("result") is True and s.vc_result in (None, True):
        vc["result"] = True
    print(f"[delta] C2 SAMPLED REPLAY (pair {pair}): {str(sr.get('grade')).upper()} evaluated {_smp.get('evaluated')} equal {_smp.get('equal')} "
          f"mismatch {_smp.get('mismatch_n')} not-evaluated {_smp.get('not_evaluated_n')} of population {(sr.get('population') or {}).get('vus')} VUs in "
          f"{(sr.get('population') or {}).get('strata_n')} strata (per-stratum {_smp.get('per_stratum')}, seed {_smp.get('seed')}); "
          f"not-evaluable {(sr.get('population') or {}).get('not_evaluable')}{(' -- ' + str(sr.get('why'))) if sr.get('why') else ''}", flush=True)
    if _smp.get("mismatch_n") and isinstance(sr.get("mismatch_by"), dict):        # every mismatch by name
        print(f"[delta] C2 SAMPLED REPLAY (pair {pair}): mismatches by family|request|phase {sr['mismatch_by'].get('by_family_request_phase')}", flush=True)
    print(f"[delta] boundary_linkage (C2 BOUNDARY LINKAGE, pair {pair}): {'OK' if lk.get('result') else ('FAIL' if lk.get('result') is False else 'NOT RUN')} "
          f"{lk.get('linked')}/{lk.get('pairs')} (step, request) boundaries linked{(' -- ' + str(lk.get('why'))) if lk.get('why') else ''}", flush=True)
    # one line: the committed up plane != the committed down plane, per (request, step, layer)
    _ac = (sr.get("moe_planes") or {}).get("alias_check") or {}
    if _ac.get("compared"):
        print(f"[delta] C2 MOE PLANE ALIAS CHECK (pair {pair}): {'OK' if _ac.get('result') else 'ALIAS'} up != down bytes on "
              f"{int(_ac.get('compared', 0)) - int(_ac.get('identical_n', 0))}/{_ac.get('compared')} (request, step, layer) triples, "
              f"{len(_ac.get('per_layer') or {})} layers, plane words {_ac.get('plane_words')}"
              f"{(' -- ' + str(_ac.get('why'))) if _ac.get('why') else ''}"
              f"{(' (read errors ' + str(_ac.get('read_errors_n')) + ')') if _ac.get('read_errors_n') else ''}", flush=True)
