"""The Commit of record's acquisition on a built committer: its required classes (`classes_of_record`) and every source, tap and
hook set its required-value manifest asks for (`attach_of_record`). `pipeline.commit.main` builds every committer of a Commit
through these, and serve-openai's rows-sha512 Commit (`pipeline.serve_seal.committer`) builds its own through them too, over
the manifest of the row's Build of record (predict's, `serve_row.row_refusal`)."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from verity_vllm import settings
from verity_vllm.acquire import fa2_plane_classes as FP
from verity_vllm.acquire import install as INSTALL
from verity_vllm.acquire import stage as STAGE
from verity_vllm.acquire.sources import hidden_source as HS
from verity_vllm.acquire.sources import moe_source as MOE
from verity_vllm.acquire.sources import taps as TAPS
from verity_vllm.check.commit_rules import (
    HIDDEN_CLASS_BY_FA,
    hidden_classes_required,
    required_classes_default,
)
from verity_vllm.commit import binding as BD
from verity_vllm.log import init_logger

__all__ = ["attach_of_record", "classes_of_record"]

logger = init_logger(__name__)


def classes_of_record(required_manifest: dict[str, Any] | None, model: Any, fa_version: int | None) -> tuple[tuple[str, ...], str | None]:
    """The required classes of an eager Commit whose engine runs FA``fa_version``, with the attention hidden class the manifest names
    for ``model`` (`hidden_source.attention_hidden`), and that class (None: none)."""
    hidden_family = HS.attention_hidden(required_manifest, model)["family"]
    return HS.with_hidden_class(required_classes_default(fa_version), hidden_family), hidden_family


def families_named(required_manifest: dict[str, Any] | None) -> dict[str, Any]:
    """How many identities a required-value manifest names of each family (its `populations.by_family`): no family where there is
    no manifest, or it counts none by family."""
    if required_manifest is None:
        return {}
    populations = required_manifest.get("populations")
    by_family = populations.get("by_family") if populations else None
    return by_family if by_family else {}


def attach_of_record(com: Any, eng: Any, args: Any, required_manifest: dict[str, Any] | None, out: Path, *, kind: str,
                     required_classes: tuple[str, ...], required_classes_cli: bool, hidden_family: str | None, fa_version: int | None,
                     cb_plan: Any = None, served: Any = None) -> SimpleNamespace:
    """The Commit's acquisition on ``com`` (``eng``'s model and runner) for its required-value manifest: the attention hidden source
    (``args.hidden_so``) when the manifest names the hidden class, the taps (`taps.attach`: the norm scales and every other family
    the manifest names), the MoE block source, the promoted members' consumer
    inputs, the X-03 attention plane classes from the Build's record, and the committer's hook sets from the acquisition plan
    (written into ``out``; refused, exit 3, as the Commit refuses). `main` builds every committer of a Commit here, and
    serve-openai's rows-sha512 Commit (`serve_seal.committer`) builds its own here too, over its row's manifest of record.
    Returns the required classes, whether the MoE source attached and whether the manifest requires the hidden class."""
    moe_attached = False
    families = families_named(required_manifest)
    _hidden_required = hidden_classes_required(required_manifest)
    if args.hidden_so and _hidden_required is False:
        # a tap that never fires would fail the identity check and the required classes as an omitted producer
        if not required_classes_cli:
            required_classes = tuple(c for c in required_classes if c not in HIDDEN_CLASS_BY_FA.values())
        logger.info("[delta] hidden source NOT attached: the required-value manifest names no %s identity (families %s); required classes %s",
                    " / ".join(HIDDEN_CLASS_BY_FA.values()), sorted(families), list(required_classes))
    # MLA / DSA sparse MLA: no hidden class, no FA2 source
    if args.hidden_so and _hidden_required is not False and (required_classes_cli or hidden_family):
        # the op namespace defaults to the .so stem (verity_fa2_matReq | verity_fa2_matAll), as e2e_commit_run.py does.
        # the source class follows the tap build (FA2 v7/v8 -> HiddenM1Source, FA3 v9 -> HiddenM1SourceFA3) and must agree with
        # the attention implementation the built engine executes (`observed_flash_attn_version`, the TargetProfile fact): a mismatch is
        # refused here, before any run (the FA2 tap under FA3 never fires).
        src_cls = getattr(com, "hidden_source_cls", None)                # native_collect_v2 supplies its C++ FA2 glue
        if args.tap_cap_mb is not None:
            settings.put("VERITY_FA2_TAP_CAP_MB", str(int(args.tap_cap_mb)))   # read by NativeHiddenM1Source at construction
        com.attach_source(HS.make_hidden_source(com, args.hidden_so, ns=(args.hidden_ns or Path(args.hidden_so).stem), flash_attn_version=fa_version,
                                                fa2_cls=src_cls, fa3_cls=getattr(com, "hidden_source_cls_fa3", None), manifest=required_manifest))
        logger.info("[delta] hidden source attached: %s -> committer %s (%s, engine FA%s)", args.hidden_so, com.name,
                    type(com.sources[-1]).__name__ if getattr(com, "sources", None) else "?", fa_version)
        _hsrc = com.sources[-1] if getattr(com, "sources", None) else None
        if _hsrc is not None and hasattr(_hsrc, "tap_cap_bytes"):
            # the bounded tap's per-window telemetry + summary live in the attempt dir (fa2_tap_windows.jsonl / fa2_tap_bound.json)
            if args.out:
                _hsrc.telemetry_path = str(Path(args.out) / "fa2_tap_windows.jsonl")
            logger.info("[delta] fa2 tap bound: cap %s MiB (%s); telemetry %s", _hsrc.tap_cap_bytes >> 20,
                        "ONE bounded device buffer; launches above it are split into sequence groups" if _hsrc.tap_cap_bytes
                        else "OFF: unbounded per-geometry buffers", _hsrc.telemetry_path)
    required_classes = TAPS.attach(com, required_manifest, required_classes, args.tap_sources, explicit_classes=required_classes_cli, programs=(args.program_dir, args.programs_root), call_boundary_plan=cb_plan)
    # the MoE block source stages six planes per MoE layer
    # per step by NAME (moe/L<k>/<plane>); the gather collector lays the arena out from the required-value MANIFEST, so a plane the manifest
    # does not name has no slot and the stage is a wild write (SIGSEGV in Collector::push_desc).  Under the PADDED construction the
    # Program has no MOE-01 rows -- the MoeFfnFixedL_v1 row's identity is the module output -- and the manifest names no
    # `moe_block_stream` identity: the source is then NOT attached (mode auto, the manifest decides; the reason is on the log), and
    # `--moe-source on` is refused by name.  A manifest that names the planes (MOE-01 rows) attaches exactly as before.
    _moe_planes_required = None
    if required_manifest is not None:
        _moe_planes_required = int(families.get("moe_block_stream", 0) or 0) > 0
    if args.moe_source != "off" and _moe_planes_required is False:
        if args.moe_source == "on":
            logger.error("[delta] REFUSE: --moe-source on but the required-value manifest %s names no moe_block_stream identity "
                         "(the Program has no MOE-01 rows -- a padded construction's MoE identity is the module output): the source would stage "
                         "planes the arena has no slot for", args.required_manifest)
            sys.exit(3)
        logger.info("[delta] MoE block source NOT attached: the required-value manifest names no moe_block_stream identity (families %s; the "
                    "Program's MoE construction has no MOE-01 rows) -- nothing is staged that the manifest does not name", list(families))
    if args.moe_source != "off" and _moe_planes_required is not False and hasattr(com, "attach_source") and getattr(com, "model", None) is not None:
        msrc = MOE.MoeBlockSource(com, com.model)
        if msrc.layers:
            com.attach_source(msrc)
            moe_attached = True
            if args.required_classes is None and "moe_block_stream" not in required_classes:
                required_classes = required_classes + ("moe_block_stream",)     # a bound MoE block is a required class of an MoE model
            logger.info("[delta] MoE block source attached: %s MoE layers -> committer %s; required classes %s", len(msrc.layers), com.name,
                        list(required_classes))
        elif args.moe_source == "on":
            logger.error("[delta] REFUSE: --moe-source on but the model has no MoE-runner modules")
            sys.exit(3)
    # MANIFEST-DRIVEN INPUT ACQUISITION: a promoted manifest row (an interior Program row whose output another
    # module reads -- e.g. a value produced in a parent module's forward body and consumed by a child leaf, which no module OUTPUT
    # carries) is observed at its first consumer's input: the committer acquires `<consumer>/in` (the embedding-ids mechanism, by the
    # manifest instead of by class) and the binding map binds it to the producer's identity.  Armed BEFORE size_arena / install; a
    # table the manifest cannot state unambiguously is refused by name (binding.PROMOTED_INPUT_RULE), never guessed.
    if required_manifest is not None and hasattr(com, "acquire_inputs") and getattr(com, "model", None) is not None:
        _pia = BD.promoted_input_acquisition(required_manifest)
        if _pia["problems"]:
            logger.error("[delta] REFUSE: the required-value manifest's promoted members have no unambiguous observation point: %s", _pia["problems"][:4])
            sys.exit(3)
        if _pia["names"]:
            _mods = dict(com.model.named_modules())
            _absent = sorted(c for c in _pia["consumers"] if c not in _mods)
            com.acquire_inputs = dict(_pia["consumers"])
            logger.info("[delta] acquisition extended by the manifest (promoted members): %s consumer input(s) %s bind %s (%s promoted manifest "
                        "rows; %s)", len(_pia["names"]), sorted(_pia["names"]), sorted(v[0] + "/" + v[1] for v in _pia["names"].values()),
                        sum(p["rows_n"] for p in _pia["producers"].values()),
                        ", ".join(f"{op}: {p['spec']} width {p['width']} {p['dtype']} at {p['consumer']}" for op, p in sorted(_pia["producers"].items())))
            if _absent:
                logger.warning("[delta] NOTE: %s promoted-member consumer(s) are not modules of the built model (%s): their identities stay unbound "
                               "(coverage names them)", len(_absent), _absent[:3])
    # [X-03] the per-layer attention plane class is DERIVED from the Program's attention Call statics (NH, KVH, D, window per
    # layer -- the Build record's `attention_impls_observed`, the facts each Call's Definition and `inert_window` were selected from) x
    # the step's active sequences (fa2_plane_classes): a windowed layer's decode launch stages 6 per-sequence planes while a global GQA
    # layer's stages one swapped row, so "every layer the same count" is a consequence for one-class Programs, not a rule.  Without the
    # record the committer keeps the uniformity rule (fails closed on a two-class Program) and says so here.
    if required_manifest is not None and hasattr(com, "identity_check"):
        _impls, _impls_path = FP.attention_impls_of_record(program_dir=getattr(args, "program_dir", None), programs_root=getattr(args, "programs_root", None))
        _decls = FP.declarations_from_attention_impls(_impls) if _impls else []
        if _decls:
            com.attention_layer_decls = _decls
            _cls = sorted(Counter(("windowed" if d.window is not None else "global") + ("-gqa" if d.NH > d.KVH else "-mha") for d in _decls).items())
            logger.info("[delta] X-03 attention plane classes: %s layer declaration(s) from %s -> classes %s; per-layer expectation derived per "
                        "step (fa2_plane_classes), uniformity not assumed", len(_decls), _impls_path, _cls)
        else:
            logger.warning("[delta] NOTE: X-03 attention plane classes: no Build record with attention_impls_observed under "
                           "--program-dir/--programs-root (%r, %r); the uniformity rule runs (a Program with two attention classes fails X-03 "
                           "closed)", getattr(args, "program_dir", None), getattr(args, "programs_root", None))
    # the committer's hook sets come from the ACQUISITION PLAN (required-value manifest x the built engine's
    # runtime tree): the plan is written into the attempt dir first; an UNSUPPORTED Value refuses by name.  An external source (the
    # served values) acquires every value itself: the committer wraps nothing.
    if served is not None:
        logger.info("[delta] committer %s: no acquisition plan (the served source acquires every value of the Build)", com.name)
    else:
        if required_manifest is None or not hasattr(com, "_modules") or getattr(com, "model", None) is None:
            logger.error("[delta] REFUSE: the acquisition plan needs --required-manifest and a hooking committer (got %s)", kind)
            sys.exit(3)
        try:
            _docs = STAGE.plan_at_stage(com.model, getattr(eng, "runner", None), required_manifest, out, source=f"commit_delta {kind}")
            for _ln in _docs.log_lines("delta"):
                logger.info("%s", _ln)
            _plan = _docs.plan
            if _plan is None:
                raise INSTALL.IncompletePlan("the stage drew no acquisition plan")
            _rec = INSTALL.apply_plan_to_committer(com, _plan)
        except INSTALL.IncompletePlan as e:
            logger.error("[delta] REFUSE: acquisition plan: %s", e)
            sys.exit(3)
        (out / "acquisition_engine.json").write_text(json.dumps({"engine": "v2", "plan_digest": _plan.digest, **_rec}, indent=1) + "\n")
        logger.info("[delta] committer %s: %s module output hooks from the plan, consumer inputs %s, deferred sites %s", com.name,
                    len(_rec["modules_after"]), _rec["plan_inputs"], len(_rec["deferred_sites"]))
    return SimpleNamespace(required_classes=required_classes, moe_attached=moe_attached, hidden_required=_hidden_required)
