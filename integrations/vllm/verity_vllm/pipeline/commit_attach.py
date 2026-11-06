"""The Commit of record's acquisition on a built committer: its required classes (`classes_of_record`) and every source, tap and
hook set its required-value manifest asks for (`attach_of_record`). `pipeline.commit.main` builds every committer of a Commit
through these, and serve-openai's rows-sha512 Commit (`pipeline.serve_seal.committer`) builds its own through them too, over
the manifest of the row's Build of record (predict's, `serve_row.row_refusal`)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from verity_vllm import settings
from verity_vllm.acquire.sources import hidden_source as HS, taps as TAPS
from verity_vllm.check.commit_rules import HIDDEN_CLASS_BY_FA, hidden_classes_required, required_classes_default


def classes_of_record(required_manifest: dict[str, Any] | None, model: Any, fa_version: int | None) -> tuple[tuple[str, ...], str | None]:
    """The required classes of an eager Commit whose engine runs FA``fa_version``, with the attention hidden class the manifest names
    for ``model`` (`hidden_source.attention_hidden`), and that class (None: none)."""
    hidden_family = HS.attention_hidden(required_manifest, model)["family"]
    return HS.with_hidden_class(required_classes_default(fa_version), hidden_family), hidden_family


def attach_of_record(com: Any, eng: Any, args: Any, required_manifest: dict[str, Any] | None, out: Path, *, kind: str,
                     required_classes: tuple[str, ...], required_classes_cli: bool, hidden_family: str | None, fa_version: int | None,
                     cb_plan: Any = None, served: Any = None) -> SimpleNamespace:
    """The Commit's acquisition on ``com`` (``eng``'s model and runner) for its required-value manifest: the attention hidden source
    (``args.hidden_so``) when the manifest names the hidden class, the taps (`taps.attach`: the norm scales and every other family
    the manifest names), the KV-slots source under ``args.memory_map``, the MoE block source, the promoted members' consumer
    inputs, the X-03 attention plane classes from the Build's record, and the committer's hook sets from the acquisition plan
    (written into ``out``; refused, exit 3, as the Commit refuses). `main` builds every committer of a Commit here, and
    serve-openai's rows-sha512 Commit (`serve_seal.committer`) builds its own here too, over its row's manifest of record.
    Returns the required classes, whether the MoE source attached and whether the manifest requires the hidden class."""
    moe_attached = False
    _hidden_required = hidden_classes_required(required_manifest)
    if args.hidden_so and _hidden_required is False:
        # a tap that never fires would fail the identity check and the required classes as an omitted producer
        if not required_classes_cli:
            required_classes = tuple(c for c in required_classes if c not in HIDDEN_CLASS_BY_FA.values())
        print(f"[delta] hidden source NOT attached: the required-value manifest names no {' / '.join(HIDDEN_CLASS_BY_FA.values())} identity "
              f"(families {sorted((required_manifest.get('populations') or {}).get('by_family') or {})}); required classes "
              f"{list(required_classes)}", flush=True)
    # MLA / DSA sparse MLA: no hidden class, no FA2 source
    if args.hidden_so and _hidden_required is not False and (required_classes_cli or hidden_family):
        from verity_vllm.acquire.sources.hidden_source import HiddenM1Source, make_hidden_source
        # the op namespace defaults to the .so stem (verity_fa2_matReq | verity_fa2_matAll), as e2e_commit_run.py does.
        # the source class follows the tap build (FA2 v7/v8 -> HiddenM1Source, FA3 v9 -> HiddenM1SourceFA3) and must agree with
        # the attention implementation the built engine executes (`observed_flash_attn_version`, the TargetProfile fact): a mismatch is
        # refused here, before any run (the FA2 tap under FA3 never fires).
        src_cls = getattr(com, "hidden_source_cls", None)                # native_collect_v2 supplies its C++ FA2 glue
        if args.tap_cap_mb is not None:
            settings.put("VERITY_FA2_TAP_CAP_MB", str(int(args.tap_cap_mb)))   # read by NativeHiddenM1Source at construction
        com.attach_source(make_hidden_source(com, args.hidden_so, ns=(args.hidden_ns or Path(args.hidden_so).stem), flash_attn_version=fa_version,
                                             fa2_cls=src_cls, fa3_cls=getattr(com, "hidden_source_cls_fa3", None), manifest=required_manifest))
        print(f"[delta] hidden source attached: {args.hidden_so} -> committer {com.name} ({type(com.sources[-1]).__name__ if getattr(com, 'sources', None) else '?'}, engine FA{fa_version})", flush=True)
        _hsrc = com.sources[-1] if getattr(com, "sources", None) else None
        if _hsrc is not None and hasattr(_hsrc, "tap_cap_bytes"):
            # the bounded tap's per-window telemetry + summary live in the attempt dir (fa2_tap_windows.jsonl / fa2_tap_bound.json)
            if args.out:
                _hsrc.telemetry_path = str(Path(args.out) / "fa2_tap_windows.jsonl")
            print(f"[delta] fa2 tap bound: cap {_hsrc.tap_cap_bytes >> 20} MiB "
                  f"({'ONE bounded device buffer; launches above it are split into sequence groups' if _hsrc.tap_cap_bytes else 'OFF: unbounded per-geometry buffers'}); "
                  f"telemetry {_hsrc.telemetry_path}", flush=True)
    required_classes = TAPS.attach(com, required_manifest, required_classes, args.tap_sources, explicit_classes=required_classes_cli, programs=(args.program_dir, args.programs_root), call_boundary_plan=cb_plan)
    if args.memory_map:
        from verity_vllm.acquire.sources import kv_slots_source
        kv_slots_source.attach(com)
    # the MoE block source stages six planes per MoE layer
    # per step by NAME (moe/L<k>/<plane>); the gather collector lays the arena out from the required-value MANIFEST, so a plane the manifest
    # does not name has no slot and the stage is a wild write (SIGSEGV in Collector::push_desc).  Under the PADDED construction the
    # Program has no MOE-01 rows -- the MoeFfnFixedL_v1 row's identity is the module output -- and the manifest names no
    # `moe_block_stream` identity: the source is then NOT attached (mode auto, the manifest decides; the reason is on the log), and
    # `--moe-source on` is refused by name.  A manifest that names the planes (MOE-01 rows) attaches exactly as before.
    _moe_planes_required = None
    if required_manifest is not None:
        _moe_planes_required = int(((required_manifest.get("populations") or {}).get("by_family") or {}).get("moe_block_stream", 0) or 0) > 0
    if args.moe_source != "off" and _moe_planes_required is False:
        if args.moe_source == "on":
            print(f"[delta] REFUSE: --moe-source on but the required-value manifest {args.required_manifest} names no moe_block_stream identity "
                  f"(the Program has no MOE-01 rows -- a padded construction's MoE identity is the module output): the source would stage "
                  f"planes the arena has no slot for", flush=True)
            sys.exit(3)
        print(f"[delta] MoE block source NOT attached: the required-value manifest names no moe_block_stream identity (families "
              f"{list((required_manifest.get('populations') or {}).get('by_family') or {})}; the Program's MoE construction has no MOE-01 rows) -- "
              f"nothing is staged that the manifest does not name", flush=True)
    if args.moe_source != "off" and _moe_planes_required is not False and hasattr(com, "attach_source") and getattr(com, "model", None) is not None:
        from verity_vllm.acquire.sources.moe_source import MoeBlockSource
        msrc = MoeBlockSource(com, com.model)
        if msrc.layers:
            com.attach_source(msrc)
            moe_attached = True
            if args.required_classes is None and "moe_block_stream" not in required_classes:
                required_classes = required_classes + ("moe_block_stream",)     # a bound MoE block is a required class of an MoE model
            print(f"[delta] MoE block source attached: {len(msrc.layers)} MoE layers -> committer {com.name}; required classes {list(required_classes)}", flush=True)
        elif args.moe_source == "on":
            print("[delta] REFUSE: --moe-source on but the model has no MoE-runner modules", flush=True)
            sys.exit(3)
    # MANIFEST-DRIVEN INPUT ACQUISITION: a promoted manifest row (an interior Program row whose output another
    # module reads -- e.g. a value produced in a parent module's forward body and consumed by a child leaf, which no module OUTPUT
    # carries) is observed at its first consumer's input: the committer acquires `<consumer>/in` (the embedding-ids mechanism, by the
    # manifest instead of by class) and the binding map binds it to the producer's identity.  Armed BEFORE size_arena / install; a
    # table the manifest cannot state unambiguously is refused by name (binding.PROMOTED_INPUT_RULE), never guessed.
    if required_manifest is not None and hasattr(com, "acquire_inputs") and getattr(com, "model", None) is not None:
        from verity_vllm.commit.binding import promoted_input_acquisition

        _pia = promoted_input_acquisition(required_manifest)
        if _pia["problems"]:
            print(f"[delta] REFUSE: the required-value manifest's promoted members have no unambiguous observation point: {_pia['problems'][:4]}", flush=True)
            sys.exit(3)
        if _pia["names"]:
            _mods = dict(com.model.named_modules())
            _absent = sorted(c for c in _pia["consumers"] if c not in _mods)
            com.acquire_inputs = dict(_pia["consumers"])
            print(f"[delta] acquisition extended by the manifest (promoted members): {len(_pia['names'])} consumer input(s) "
                  f"{sorted(_pia['names'])} bind {sorted(v[0] + '/' + v[1] for v in _pia['names'].values())} "
                  f"({sum(p['rows_n'] for p in _pia['producers'].values())} promoted manifest rows; "
                  f"{', '.join(f'{op}: {p['spec']} width {p['width']} {p['dtype']} at {p['consumer']}' for op, p in sorted(_pia['producers'].items()))})", flush=True)
            if _absent:
                print(f"[delta] NOTE: {len(_absent)} promoted-member consumer(s) are not modules of the built model ({_absent[:3]}): their identities stay unbound (coverage names them)", flush=True)
    # [X-03] the per-layer attention plane class is DERIVED from the Program's attention Call statics (NH, KVH, D, window per
    # layer -- the Build record's `attention_impls_observed`, the facts each Call's Definition and `inert_window` were selected from) x
    # the step's active sequences (fa2_plane_classes): a windowed layer's decode launch stages 6 per-sequence planes while a global GQA
    # layer's stages one swapped row, so "every layer the same count" is a consequence for one-class Programs, not a rule.  Without the
    # record the committer keeps the uniformity rule (fails closed on a two-class Program) and says so here.
    if required_manifest is not None and hasattr(com, "identity_check"):
        from collections import Counter as _Cnt

        from verity_vllm.acquire import fa2_plane_classes as FP

        _impls, _impls_path = FP.attention_impls_of_record(program_dir=getattr(args, "program_dir", None), programs_root=getattr(args, "programs_root", None))
        _decls = FP.declarations_from_attention_impls(_impls) if _impls else []
        if _decls:
            com.attention_layer_decls = _decls
            _cls = sorted(_Cnt(("windowed" if d.window is not None else "global") + ("-gqa" if d.NH > d.KVH else "-mha") for d in _decls).items())
            print(f"[delta] X-03 attention plane classes: {len(_decls)} layer declaration(s) from {_impls_path} -> classes {_cls}; "
                  f"per-layer expectation derived per step (fa2_plane_classes), uniformity not assumed", flush=True)
        else:
            print(f"[delta] NOTE: X-03 attention plane classes: no Build record with attention_impls_observed under "
                  f"--program-dir/--programs-root ({getattr(args, 'program_dir', None)!r}, {getattr(args, 'programs_root', None)!r}); "
                  f"the uniformity rule runs (a Program with two attention classes fails X-03 closed)", flush=True)
    # the committer's hook sets come from the ACQUISITION PLAN (required-value manifest x the built engine's
    # runtime tree): the plan is written into the attempt dir first; an UNSUPPORTED Value refuses by name.  An external source (the
    # served values) acquires every value itself: the committer wraps nothing.
    if served is not None:
        print(f"[delta] committer {com.name}: no acquisition plan (the served source acquires every value of the Build)", flush=True)
    else:
        if required_manifest is None or not hasattr(com, "_modules") or getattr(com, "model", None) is None:
            print(f"[delta] REFUSE: the acquisition plan needs --required-manifest and a hooking committer (got {kind})", flush=True)
            sys.exit(3)
        from verity_vllm.acquire.install import IncompletePlan as _IncompletePlan, apply_plan_to_committer as _apply_plan
        from verity_vllm.acquire.stage import plan_at_stage as _plan_at_stage

        try:
            _docs = _plan_at_stage(com.model, getattr(eng, "runner", None), required_manifest, out, source=f"commit_delta {kind}")
            for _ln in _docs.log_lines("delta"):
                print(_ln, flush=True)
            _rec = _apply_plan(com, _docs.plan)
        except _IncompletePlan as e:
            print(f"[delta] REFUSE: acquisition plan: {e}", flush=True)
            sys.exit(3)
        (out / "acquisition_engine.json").write_text(json.dumps({"engine": "v2", "plan_digest": _docs.plan.digest, **_rec}, indent=1) + "\n")
        print(f"[delta] committer {com.name}: {len(_rec['modules_after'])} module output hooks from the plan, "
              f"consumer inputs {_rec['plan_inputs']}, deferred sites {len(_rec['deferred_sites'])}", flush=True)
    return SimpleNamespace(required_classes=required_classes, moe_attached=moe_attached, hidden_required=_hidden_required)
