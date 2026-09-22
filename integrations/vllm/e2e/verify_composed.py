"""Second-process verification of a ``run_composed`` output directory (lane INT).

    python -m e2e.verify_composed --dir DIR [--host zk/sp1/host/target/release/veritor-sp1-host] [--out DIR/reverify]

No GPU, no vLLM, no checkpoint: everything is re-derived from the preserved artefacts and the code.
The checks, each a row of ``reverify.json`` with its own verdict:

* **program** -- the typed ``Program`` for the case named in ``manifest.json`` is *reconstructed* from
  TA1's static configuration (``verity.ir``) and its descriptor digest recomputed; it must equal the
  recorded ``program_digest``; the preserved descriptor's sha256 and the layer sub-program's digest are
  checked too.
* **plan** -- the layer sub-program is bridged to the plan IR again, the two partitions rebuilt and
  re-validated (``veritor.plans.validate``: executable, quotient acyclic, one producer); the plan digest
  must equal the recorded ``plan_digest`` and both partitions must be ``valid``.
* **records** -- the exported ``typed-obligation/v0`` object re-validates (``veritor.interface``); its
  canonical bytes give the recorded ``h_O``; its lowering's statement bytes equal the preserved
  ``statement.bin`` and hash to the recorded ``h_L``; **coverage** (``check_coverage``, IT-F1): the
  coordinate units the challenge selects are re-drawn from the recorded challenge digest and the
  verifier's rows and every one must have an obligation in the object (a missing unit is named);
  **placement** (``check_placement``, IT-F2): every leaf's id is the one the authenticated witness leaf
  at its ``(tree, rank)`` carries and its position is its rank; ``check_relations`` accepts the recorded
  boundary words; the transparent backend accepts the preserved format-3 witness.
* **proofs** -- each preserved proof (core, compressed) is re-verified by the SP1 host binary with
  ``SP1_PROVER=cpu`` (the host needs the GPU only to *prove*; ``verify`` runs ``ProverClient::from_env``
  setup + verify on CPU) against the statement; the vk hash must equal the recorded one and the public
  values must bind the object (``check_binding``: 65 bytes ``h_O || h_L || verdict``).  Without a host
  binary this row is ``not re-verified`` and the final verdict is ``INCOMPLETE`` unless
  ``--allow-recorded-proofs`` (then the recorded host JSON is re-read and the binding re-checked on the
  recorded public values, and the row says so).
* **bundle** -- ``verity_vllm.bundle.rederive_transcript`` unpacks the tarball (every member's sha256
  checked) and re-derives the staged security-mode challenge from the revealed seed and the bundled
  commitments (``T_0..T_3``, the selected replay units, the ranked population, both samples, the challenge
  digest); then ``verify_tarball`` drops in the preserved ``weights/values.bin`` (referenced by sha in the
  bundle) and runs the fresh CPU bundle verifier over a fresh sample (``--bundle-sample`` VUs; a security-mode
  run has no reproducible seed, so the sample is seeded from the recorded challenge digest and declared not
  comparable).  Without torch the same-runtime relations have no executable checker, so that fresh row is
  ``PARTIAL``; it is accepted only when everything the process could check held (structure, every
  authentication, every independent relation, every negative rejected) and is reported as PARTIAL, never
  relabelled PASS.
* **summary** -- the per-stage statuses of ``summary.json`` are re-read; the fresh verdict here is the
  conjunction of the rows above and must equal the recorded final verdict.

Exit status 0 iff the fresh verdict equals the recorded one and is PASS (a recorded FAIL that this process
also reaches is exit 0 with ``agrees: true``: the negatives are supposed to fail).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
POC = HERE.parent
ROOT = POC.parent
for p in (str(POC), str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from e2e.run_composed import PLAN_CHECKS, host_json, plans_digest, sha256_file, sha256_json, static_configuration, unit_plans, write_json  # noqa: E402

DEFAULT_HOST = ROOT / "zk" / "sp1" / "host" / "target" / "release" / "veritor-sp1-host"


class Row:
    def __init__(self, name: str) -> None:
        self.name = name
        self.status = "not run"  # ok | FAIL | skipped | not re-verified | error
        self.detail = ""
        self.data: dict[str, Any] = {}
        self.seconds = 0.0

    def to_json(self) -> dict[str, Any]:
        return {"status": self.status, "seconds": round(self.seconds, 2), "detail": self.detail, **self.data}


def _load(path: Path) -> Any:
    return json.loads(path.read_text())


def check_program_recorded(d: Path, manifest: dict[str, Any], summary: dict[str, Any], row: Row) -> None:
    """``--program recorded``: the ``Serve`` program is NOT reconstructed here (a B1 descriptor is 410 MB and its
    construction is forbidden on the workstation, int3-brief §4); the preserved descriptor's sha256 and digest are
    checked against ``program.json`` / the manifest and the (small) layer program is rebuilt and compared. The row is
    ``ok`` only as a *qualified* row: reconstruction is the pod's second process's job and its ``reverify.json`` says so."""
    from verity.ir import Program, bind, encode_program, program_digest
    from verity_vllm.registry import b1

    case = summary["case"]
    pdoc = _load(d / "program" / "program.json")
    rec = pdoc["program"]
    config = static_configuration(case)
    problems: list[str] = []
    h = hashlib.sha256()  # program.json records the sha256 of the *decompressed* descriptor bytes; streamed, never parsed
    with gzip.open(d / "program" / rec["descriptor"], "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    if h.hexdigest() != rec["descriptor_sha256"]:
        problems.append("preserved descriptor sha256 differs from program.json")
    if rec["digest"] != manifest["inputs"].get("program_digest"):
        problems.append(f"program.json digest {rec['digest'][:16]} != manifest {str(manifest['inputs'].get('program_digest'))[:16]}")
    layer = Program(bind(b1.LayerPost, C=config))
    ldigest = program_digest(encode_program(layer))
    if ldigest != pdoc["layer_program"]["digest"]:
        problems.append(f"layer program digest: recomputed {ldigest[:16]} != recorded {pdoc['layer_program']['digest'][:16]}")
    row.data = {"program": rec.get("function"), "digest": rec["digest"], "layer_program_digest": ldigest, "gates": rec.get("gates"), "problems": problems,
                "program_reconstructed": False,
                "qualification": ("--program recorded: the Serve program was not reconstructed by this process (descriptor sha256 + recorded digest "
                                  "checked, layer program rebuilt); the reconstruction and DC equivalence check are the pod second process's row")}
    row.status = "ok" if not problems else "FAIL"
    row.detail = (f"{rec.get('function')} recorded digest {rec['digest'][:16]} (descriptor sha256 checked, NOT reconstructed here); layer program rebuilt"
                  + (f"; {len(problems)} problem(s)" if problems else ""))


def check_program(d: Path, manifest: dict[str, Any], summary: dict[str, Any], row: Row) -> None:
    from verity.ir import Program, bind, encode_program, program_digest
    from verity_vllm.registry import b1

    case = summary["case"]
    pdoc = _load(d / "program" / "program.json")
    rec = pdoc["program"]
    config = static_configuration(case)
    lp, steps = int(rec["LP"]), int(rec["STEPS"])  # the Serve binding's workload parameters, recorded by the runner
    if rec.get("padded"):
        # [PAD-INT, additive; GUIDANCE §51.4] a padded record (run_composed --padded-program): the recorded program is the
        # DERIVED Serve_v3{C3, LP, D_MAX}; the independent reconstruction is the registered oracle serve3.serve_v3 at the
        # recorded statics (EOS_ID, D_MAX) — digest-equal to the derived descriptor by the spec's identity claim.
        from verity_vllm.registry import serve3

        program = Program(serve3.serve_v3({**config, "EOS_ID": int(rec.get("EOS_ID", 0))}, lp, int(rec.get("D_MAX", steps))))
    else:
        program = Program(bind(b1.Serve, C=config, LP=lp, STEPS=steps))
    desc = encode_program(program)
    digest = program_digest(desc)
    layer = Program(bind(b1.LayerPost, C=config))
    ldigest = program_digest(encode_program(layer))
    with gzip.open(d / "program" / rec["descriptor"], "rb") as f:
        raw = f.read()
    preserved = json.loads(raw)
    problems = []
    transitions: dict[str, Any] = {}
    if digest != rec["digest"] or digest != manifest["inputs"].get("program_digest"):
        # The digest hashes the serialised descriptor, so a descriptor-codec or builder change (DC, `descriptor-codec.md`)
        # moves it while the denoted circuit is unchanged. Across such a transition the test is not digest equality but
        # DC's denotational correspondence between the *preserved* descriptor (whose own digest must still be the recorded
        # one, checked below) and the program reconstructed here. Recorded as a qualification, never relabelled "equal".
        transitions["program"] = _codec_transition(preserved, program, desc)
        if transitions["program"]["verdict"] != "equivalent":
            problems.append(f"program digest: recomputed {digest[:16]} != recorded {rec['digest'][:16]} / manifest {str(manifest['inputs'].get('program_digest'))[:16]}; "
                            f"and the preserved descriptor is not equivalent to the reconstruction: {transitions['program'].get('detail')}")
    if hashlib.sha256(raw).hexdigest() != rec["descriptor_sha256"]:
        problems.append("preserved descriptor sha256 differs from program.json")
    if program_digest(preserved) != rec["digest"]:
        problems.append("preserved descriptor's digest differs from the recorded program digest")
    if ldigest != pdoc["layer_program"]["digest"]:
        lpath = d / "program" / pdoc["layer_program"].get("descriptor", "layer_descriptor.json")
        if lpath.exists():
            lpreserved = _load(lpath)
            transitions["layer_program"] = _codec_transition(lpreserved, layer, encode_program(layer))
            if program_digest(lpreserved) != pdoc["layer_program"]["digest"]:
                problems.append("preserved layer descriptor's digest differs from the recorded layer program digest")
        else:
            transitions["layer_program"] = {"verdict": "MISMATCH", "detail": "no preserved layer descriptor to compare"}
        if transitions["layer_program"]["verdict"] != "equivalent":
            problems.append(f"layer program digest: recomputed {ldigest[:16]} != recorded {pdoc['layer_program']['digest'][:16]}; "
                            f"not equivalent: {transitions['layer_program'].get('detail')}")
    if program.gates != rec["gates"]:
        problems.append(f"gate count {program.gates} != recorded {rec['gates']}")
    row.data = {"program": program.fn.id, "digest": digest, "layer_program_digest": ldigest, "gates": program.gates, "problems": problems}
    if transitions:
        row.data["codec_transition"] = transitions
        row.data["qualification"] = ("recorded digests were written under a different descriptor codec/builder; the preserved "
                                     "descriptors were certified denotationally equivalent to the reconstructed programs "
                                     "(verity_vllm.ir.descriptor_equivalence), not digest-equal")
    row.status = "ok" if not problems else "FAIL"
    row.detail = f"{program.fn.id} digest {digest[:16]} recomputed from {case}'s static configuration" + (f"; {len(problems)} problem(s)" if problems else "")
    if transitions and not problems:
        row.detail += "; recorded digest differs across a descriptor-codec transition, preserved descriptor certified equivalent (qualified)"


def _codec_transition(preserved_desc: dict[str, Any], reconstructed, reconstructed_desc: dict[str, Any]) -> dict[str, Any]:
    """DC's correspondence between a preserved descriptor and the program reconstructed from the registry."""
    from verity.ir.codec import decode_program
    from verity.ir.defs import REGISTRY
    from verity_vllm.ir.descriptor_equivalence import Mismatch, compare

    t0 = time.time()
    out: dict[str, Any] = {"recorded_schema": preserved_desc.get("schema"), "current_schema": reconstructed_desc.get("schema")}
    try:
        decoded = decode_program(preserved_desc, REGISTRY)
        rep = compare(decoded, reconstructed)
        out.update({"verdict": "equivalent", "report": rep})
    except Mismatch as e:
        out.update({"verdict": "MISMATCH", "detail": str(e)})
    except Exception as e:  # noqa: BLE001 -- a decode failure is a mismatch of the record, reported not raised
        out.update({"verdict": "MISMATCH", "detail": f"{type(e).__name__}: {e}"})
    out["seconds"] = round(time.time() - t0, 2)
    return out


def check_plan(d: Path, manifest: dict[str, Any], summary: dict[str, Any], row: Row) -> None:
    from veritor.plans.validate import validate_plan

    report = _load(d / "plan" / "validator_report.json")
    config = static_configuration(summary["case"])
    # v1 reports (INT): one unit, the layer, under `partitions`; v2 (INT3): `units` = {unit: {partitions, plan_digest}} + `unit_order`
    unit_reports: dict[str, dict[str, Any]] = report.get("units") or {"layer": {"program": report.get("program"), "partitions": report["partitions"], "plan_digest": report.get("plan_digest")}}
    unit_order: list[str] = list(report.get("unit_order") or unit_reports.keys())
    fresh_units: dict[str, Any] = {}
    digests: dict[str, str] = {}
    problems = []
    names = []
    for unit in unit_order:
        Q, c, partitions = unit_plans(config, unit)
        digests[unit] = plans_digest(Q, partitions)
        names.append(Q.name)
        rec_u = unit_reports.get(unit) or {}
        fresh: dict[str, Any] = {}
        for tag, plan in partitions:
            v = validate_plan(c, plan)
            fresh[tag] = {"status": v.status, "codes": [str(x) for x in v.codes], "vus": len(plan.verification_units), "cus": len(plan.commitment_units),
                          "vu_edges": v.vu_edges, "cu_edges": v.cu_edges}
            rec = (rec_u.get("partitions") or {}).get(tag, {})
            if v.status != "valid":
                problems.append(f"{unit} {tag}: {v.status} {fresh[tag]['codes']}")
            if rec.get("status") != v.status or rec.get("vus") != fresh[tag]["vus"]:
                problems.append(f"{unit} {tag}: recorded {rec.get('status')}/{rec.get('vus')} VUs, fresh {v.status}/{fresh[tag]['vus']}")
        if rec_u.get("plan_digest") and rec_u["plan_digest"] != digests[unit]:
            problems.append(f"{unit}: plan digest recomputed {digests[unit][:16]} != recorded {rec_u['plan_digest'][:16]}")
        fresh_units[unit] = {"program": Q.name, "plan_digest": digests[unit], "partitions": fresh}
        del Q, c, partitions
    digest = digests["layer"] if unit_order == ["layer"] else sha256_json({"units": digests})
    if digest != report.get("plan_digest") or digest != manifest["inputs"].get("plan_digest"):
        problems.append(f"plan digest: recomputed {digest[:16]} != recorded {str(report.get('plan_digest'))[:16]}")
    first = fresh_units[unit_order[0]]
    row.data = {"program": first["program"], "plan_digest": digest, "partitions": first["partitions"], "units": fresh_units, "checks": PLAN_CHECKS,
                "recorded_bound": {u: {k: v.get("bound") for k, v in (ur.get("partitions") or {}).items()} for u, ur in unit_reports.items()},
                "not_validated_units": report.get("not_validated_units"), "unit_probe_recorded": "unit_probe" in report,
                "full_program": report.get("full_program"), "problems": problems}
    row.status = "ok" if not problems else "FAIL"
    row.detail = "; ".join(f"{u} ({fu['program']}): " + ", ".join(f"{t} {v['status']} ({v['vus']} VUs)" for t, v in fu["partitions"].items()) for u, fu in fresh_units.items()) + f"; plan digest {digest[:16]}"
    if report.get("not_validated_units"):
        row.detail += f"; NOT validated by the recorded run (nor here): {', '.join(report['not_validated_units'])}"


def check_plan_recorded(d: Path, manifest: dict[str, Any], summary: dict[str, Any], row: Row) -> None:
    """``--plan recorded``: no re-validation (the units' circuits may exceed this machine's memory); the report's internal
    consistency is checked -- every partition of every unit recorded `valid`, the digest matches the manifest -- and the
    row is a *qualified* ok: re-validation is the pod second process's row."""
    report = _load(d / "plan" / "validator_report.json")
    unit_reports: dict[str, dict[str, Any]] = report.get("units") or {"layer": {"program": report.get("program"), "partitions": report["partitions"], "plan_digest": report.get("plan_digest")}}
    problems = []
    for unit, ur in unit_reports.items():
        for tag, e in (ur.get("partitions") or {}).items():
            if e.get("status") != "valid":
                problems.append(f"{unit} {tag}: recorded {e.get('status')}")
    if report.get("plan_digest") != manifest["inputs"].get("plan_digest"):
        problems.append("plan digest: report != manifest")
    if not report.get("valid"):
        problems.append("report.valid is not true")
    row.data = {"program": (unit_reports.get(list(unit_reports)[0]) or {}).get("program"), "plan_digest": report.get("plan_digest"),
                "partitions": {t: {"status": e.get("status"), "vus": e.get("vus")} for t, e in (unit_reports[list(unit_reports)[0]].get("partitions") or {}).items()},
                "units": {u: {"program": ur.get("program"), "partitions": {t: {"status": e.get("status"), "vus": e.get("vus")} for t, e in (ur.get("partitions") or {}).items()}} for u, ur in unit_reports.items()},
                "not_validated_units": report.get("not_validated_units"), "full_program": report.get("full_program"), "problems": problems, "plan_revalidated": False,
                "qualification": "--plan recorded: the plans were NOT re-validated by this process (report consistency and digest checked); re-validation is the pod second process's row"}
    row.status = "ok" if not problems else "FAIL"
    row.detail = f"{len(unit_reports)} unit(s) recorded valid, NOT re-validated here; plan digest {str(report.get('plan_digest'))[:16]}" + (f"; {len(problems)} problem(s)" if problems else "")


def _verifier_rows(d: Path) -> list[dict[str, Any]]:
    """The verifier's per-VU rows: RC's ``results.jsonl`` (selective) or WB's ``verify.json`` report (full replay)."""
    vdir = d / "verify"
    if (vdir / "results.jsonl").exists():
        return [json.loads(l) for l in (vdir / "results.jsonl").read_text().splitlines() if l.strip()]
    if (vdir / "verify.json").exists():
        return list(((_load(vdir / "verify.json").get("report") or {}).get("results")) or [])
    return []


def _coordinate_obligations(obj: Any) -> dict[tuple[str, int], int]:
    """``(output row canonical id, j) -> obligation index`` of every GEMM-coordinate obligation of the object: the
    obligation's outgoing boundary word, located through the reference table (leaf id + word offset)."""
    by_ref = {r.ref: r for r in obj.references}
    out: dict[tuple[str, int], int] = {}
    for k, ob in enumerate(obj.obligations):
        if not ob.boundary.incoming:  # an Input equation has no incoming boundary; a coordinate has x row + W row
            continue
        ref = by_ref.get(ob.boundary.outgoing[0]) if ob.boundary.outgoing else None
        if ref is None or not ref.offset or not isinstance(ref.offset[0], int):
            continue
        out[(ref.leaf, int(ref.offset[0]))] = k
    return out


def check_coverage(d: Path, manifest: dict[str, Any], record: dict[str, Any], obj: Any) -> dict[str, Any]:
    """IT-F1 (independent-tests.md §5 case c): the coordinate units the *challenge* demands, re-drawn here from the
    recorded challenge digest and the verifier's own rows, must every one be covered by an obligation of the object.

    ``select_coordinate_units`` is prover-side code in ``run_composed``; nothing in the record-level checks
    re-derived its input before this, so an obligation removed from an otherwise consistent object (plan, seed,
    ``selection.json``, proofs intact) passed every row but the recorded proof's digest -- and a re-proof of the
    reduced object would pass.  Here the sampled GEMM rows are rebuilt from the verifier's data alone (the
    challenge's certified / coverage / adjacent membership, the verifier's ``op_type`` / ``status`` / output
    bytes per VU), the same seeded draw is re-run with the run's parameters, and the result is compared with the
    object (missing units name themselves) and with the prover's ``selection.json``.  The chain preference's
    candidate set (which sampled rows' ``x`` is itself a sampled VU) needs the description, which the record does
    not carry: the recorded candidates are used after being checked against the verifier's sample, and that
    limit is stated in the result.  The object's boundary and weights roots are also compared with the roots the
    verifier bound (``verify/statement.json``); the RU roots are behind the challenge's commitments digest and are
    not re-checkable from the record."""
    from verity_vllm import identity
    from verity_vllm_adapter.coordinate_check import FAMILIES, GEMM_TYPES
    from verity_vllm_adapter.obligations import SampledRow, select_coordinate_units

    res: dict[str, Any] = {"check": "the obligation set covers the coordinate units the challenge selects (IT-F1)"}
    problems: list[str] = []
    vdir, odir = d / "verify", d / "obligations"
    if not (vdir / "challenge.json").exists():
        res.update({"status": "skipped", "reason": "no verify/challenge.json in the record"})
        return res
    ch = _load(vdir / "challenge.json")
    rows_by_id = {r["canonical_id"]: r for r in _verifier_rows(d)}
    membership: dict[str, list[str]] = {}
    for key, name in (("certified_sample", "certified"), ("coverage_sample", "coverage"), ("adjacent_vus", "adjacent")):
        for cid in ch.get(key) or []:
            membership.setdefault(cid, []).append(name)
    if not membership:
        for cid in ch.get("sample") or []:
            membership.setdefault(cid, []).append("sample")
    # a row's N = its output bytes over the word size of the committed rows (the leaves' layout: Value<16> = bf16)
    widths = {w for w in (_layout_value_bits(lf.layout) for lf in obj.leaves) if w}
    word_bytes = max(1, (min(widths) if widths else 16) // 8)
    rows: list[SampledRow] = []
    for cid, samples in membership.items():
        r = rows_by_id.get(cid)
        if r is None or r.get("op_type") not in GEMM_TYPES:
            continue
        p = identity.parse(cid)
        if p.op is None or p.request_id is None:
            continue
        family = "lm_head" if r["op_type"] == "lm_head" else next((f for f in FAMILIES if f in str(r.get("op") or p.op)), "other")
        rows.append(SampledRow(cid, str(r.get("op") or p.op), str(r["op_type"]), family, str(p.phase), str(p.layer), p.request_id,
                               int(r["output_bytes"]) // word_bytes, tuple(samples), r.get("status")))
    # the run's draw parameters: the command line when given, else run_composed's defaults
    cmd = str(manifest.get("command") or "")
    def _arg(flag: str, default: int) -> int:
        m = re.search(rf"{re.escape(flag)}(?:=|\s+)(\d+)", cmd)
        return int(m.group(1)) if m else default
    params = {"rows_sharing_x": _arg("--rows-sharing-x", 2), "coords_per_row": _arg("--coords-per-row", 2), "ops_sharing_w": _arg("--ops-sharing-w", 1),
              "chain": bool((manifest.get("inputs") or {}).get("chain", "--no-chain" not in cmd))}
    recorded_sel = _load(odir / "selection.json") if (odir / "selection.json").exists() else {}
    candidates: dict[str, dict[str, Any]] | None = None
    if params["chain"]:
        candidates = {}
        in_rows = {r.canonical_id for r in rows}
        for cid, p in (recorded_sel.get("producers") or {}).items():
            producer = p.get("producer")
            if cid in in_rows and producer in membership:
                candidates[cid] = {"producer": producer, "producer_verifier_status": (rows_by_id.get(producer) or {}).get("status")}
    sel = select_coordinate_units(rows, ch, rows_sharing_x=params["rows_sharing_x"], coords_per_row=params["coords_per_row"], ops_sharing_w=params["ops_sharing_w"],
                                  chain_candidates=candidates or None)
    demanded = [(c, int(j)) for c, j in sel.units]
    covered = _coordinate_obligations(obj)
    missing = [f"{c}[j={j}]" for c, j in demanded if (c, j) not in covered]
    extra = [f"{c}[j={j}]" for c, j in covered if (c, j) not in set(demanded)]
    for m in missing:
        problems.append(f"coverage: the challenge selects coordinate unit {m} and the object has no obligation for it")
    recorded_units = [(u["canonical_id"], int(u["j"])) for u in ((recorded_sel.get("selection") or {}).get("units") or [])]
    if recorded_sel and sorted(recorded_units) != sorted(demanded):
        problems.append(f"coverage: the recorded selection.json names {len(recorded_units)} unit(s), the re-derived selection {len(demanded)} -- they differ")
    # the roots the verifier bound (boundary trees and weights) vs the object's roots
    roots_checked = 0
    if (vdir / "statement.json").exists():
        vst = _load(vdir / "statement.json")
        bound = dict(vst.get("boundary_roots") or {})
        if vst.get("weights_root"):
            bound["weights"] = vst["weights_root"]
        for root in obj.roots:
            b = bound.get(root.id)
            if b is None:
                continue
            roots_checked += 1
            if b.get("root") != root.root or int(b.get("leaf_count", -1)) != int(root.count) or (b.get("ids_digest") and b["ids_digest"] != root.domain):
                problems.append(f"roots: the object's {root.id} (root {root.root[:16]}, {root.count} leaves) is not the one the verifier bound")
    res.update({"status": "ok" if not problems else "FAIL", "sampled_gemm_rows": len(rows), "draw": params, "demanded_units": len(demanded), "covered_units": len(covered),
                "missing": missing, "extra_obligations_not_demanded": extra, "recorded_selection_agrees": sorted(recorded_units) == sorted(demanded) if recorded_sel else None,
                "chain_candidates": (len(candidates) if candidates is not None else None),
                "chain_candidates_source": "selection.json's recorded producers, each checked to be a sampled GEMM row whose producer is in the verifier's sample "
                                           "(the description that resolves x is not in the record: completeness of the candidate set is not re-derived here)",
                "roots_checked_against_verifier": roots_checked, "ru_roots_note": "RU tree roots are behind the challenge's commitments digest, not re-checkable from the record",
                "problems": problems})
    return res


def check_placement(obj: Any, statement_bytes_: bytes, witness_bytes: bytes) -> dict[str, Any]:
    """IT-F2 (independent-tests.md §5 cases b1 / b3): which leaf a logical value lives in is the store's word, not
    the prover's.  Under the serving side's framing a packed leaf hashes as ``H(framing, canonical_id, dtype,
    shape, bytes)`` and folds to the committed root, so the canonical id the witness carries at ``(tree, rank)``
    is the commit store's id -> rank binding, authenticated.  Every object leaf must (1) name that same id at its
    rank -- a same-layout leaf swap (b1) is refused here, by layout authentication, before arithmetic gets to it --
    and (2) have ``position == rank``, which is what the exporter writes for this framing (the domain position of
    a vLLM tree leaf is its rank; b3 swaps the position alone and was refused by nothing).  The leaf's layout word
    count is compared with the witness leaf's shape as well.  The witness leaves themselves are authenticated by
    the transparent backend afterwards (``check_packed_openings``); this check runs first so a placement refusal
    is named as such."""
    from veritor.protocol.proofs.packed import decode_packed_witness, packed_table
    from veritor.protocol.proofs.wire import decode_any_statement

    problems: list[str] = []
    # any recorded statement format (2 or 3, by magic), then its format-2 meaning: a range is exactly its expansion
    st = decode_any_statement(statement_bytes_).expand()
    table = packed_table(st)
    witness = decode_packed_witness(witness_bytes)
    by_root: dict[tuple[str, str], int] = {}
    for t, tree in enumerate(table.trees):
        by_root[(tree.commitment.root.hex(), tree.commitment.domain_id.hex())] = t
    checked = 0
    for leaf in obj.leaves:
        root = next((r for r in obj.roots if r.id == leaf.root), None)
        if root is None:
            problems.append(f"placement: leaf {leaf.id} names root {leaf.root}, which the object does not carry")
            continue
        t = by_root.get((root.root, root.domain))
        if t is None:
            continue  # a leaf no obligation opens has no witness leaf to compare with
        tree = table.trees[t]
        if leaf.leaf not in tree.ranks:
            continue
        k = tree.ranks.index(leaf.leaf)
        if k >= len(witness.trees[t]):
            problems.append(f"placement: no witness leaf for rank {leaf.leaf} of {root.id}")
            continue
        packed = witness.trees[t][k]
        checked += 1
        if packed.canonical_id != leaf.id:
            problems.append(f"placement: the object says {leaf.id} is rank {leaf.leaf} of {root.id}, the authenticated leaf at that rank is {packed.canonical_id}")
        if leaf.position != leaf.leaf:
            problems.append(f"placement: leaf {leaf.id} has position {leaf.position} but rank {leaf.leaf} (under {root.framing} the position of a leaf is its rank)")
        try:
            words = 1
            for n in json.loads(packed.shape_text):
                words *= int(n)
            layout_words = _layout_words(leaf.layout)
            if layout_words is not None and layout_words != words:
                problems.append(f"placement: leaf {leaf.id} layout packs {layout_words} words, the authenticated leaf holds {words}")
        except (ValueError, TypeError):
            pass
    return {"check": "each object leaf's id and position agree with the authenticated (tree, rank) it is opened at (IT-F2)",
            "status": "ok" if not problems else "FAIL", "leaves_checked": checked, "leaves": len(obj.leaves), "problems": problems}


def _layout_doc(layout: Any) -> Any:
    try:
        return layout.to_json() if hasattr(layout, "to_json") else layout
    except Exception:  # noqa: BLE001
        return None


def _layout_words(layout: Any) -> int | None:
    """Words packed by a layout ``Type`` (``array[n] of Value<w>`` -> n; a bare value -> 1); None when unknown."""
    doc = _layout_doc(layout)
    if isinstance(doc, dict) and "array" in doc:
        n, inner = doc["array"]
        sub = _layout_words(inner)
        return int(n) * sub if sub is not None else None
    if isinstance(doc, dict) and "value" in doc:
        return 1
    return None


def _layout_value_bits(layout: Any) -> int | None:
    """The word width of a layout's innermost ``Value<w>``; None when unknown."""
    doc = _layout_doc(layout)
    if isinstance(doc, dict) and "array" in doc:
        return _layout_value_bits(doc["array"][1])
    if isinstance(doc, dict) and "value" in doc:
        return int(doc["value"])
    return None


def check_records(d: Path, row: Row, manifest: dict[str, Any] | None = None) -> None:
    from veritor.interface import TypedObligationSet, canonical_bytes, check_relations, family_gate_set, object_digest, statement_bytes, validate
    from veritor.interface.lowering import statement_bytes_v3
    from veritor.protocol.messages import Reject
    from veritor.protocol.proofs import TransparentBackend
    from veritor.protocol.proofs.wire import decode_any_statement, statement_digest, statement_format_of

    odir = d / "obligations"
    if not (odir / "composed.json").exists():
        row.status, row.detail = "skipped", "no exported record (the run stopped before the obligations stage)"
        return
    record = _load(odir / "composed.json")
    obj = TypedObligationSet.from_json(record["object"])
    validate(obj)
    problems = []
    h_o = object_digest(record["object"])
    if h_o != record["digest"]:
        problems.append("h_O differs from the record")
    if canonical_bytes(record["object"]).decode() != record["canonical"]:
        problems.append("canonical bytes differ from the record")
    # GUEST (S12 opt-in): re-lower in the RECORDED statement format.  A record without the field is a format-2
    # (historical) record; the field, the manifest pin and statement.bin's magic must all agree, fail-closed.
    statement_format = int(record.get("statement_format", 2))
    preserved = (odir / "statement.bin").read_bytes()
    manifest_format = int((manifest or {}).get("inputs", {}).get("statement_format", 2)) if manifest else statement_format
    if statement_format not in (2, 3):
        problems.append(f"statement_format {statement_format!r} is not 2 or 3")
        statement_format = 2
    if manifest_format != statement_format:
        problems.append(f"manifest inputs.statement_format {manifest_format} != record statement_format {statement_format}")
    try:
        if statement_format_of(preserved) != statement_format:
            problems.append(f"statement.bin is format {statement_format_of(preserved)}, the record says {statement_format}")
    except Exception as e:  # noqa: BLE001 -- unknown magic
        problems.append(f"statement.bin has no known statement magic ({e})")
    st = statement_bytes(obj) if statement_format == 2 else statement_bytes_v3(obj)
    if st != preserved:
        problems.append("re-lowered statement bytes differ from statement.bin")
    if st.hex() != record["statement" if statement_format == 2 else "statement_v3"]:
        problems.append("re-lowered statement differs from the record's statement hex")
    h_l = statement_digest(st).hex()
    if h_l != record["statement_digest"]:
        problems.append("h_L differs from the record")
    if statement_format == 3:
        # the format-3 record also names the format-2 digest of the same lowering; both must be this object's
        v2 = statement_digest(statement_bytes(obj)).hex()
        if v2 != record.get("statement_v2_digest"):
            problems.append("statement_v2_digest differs from the re-lowered format-2 digest")
        if decode_any_statement(st).expand() != decode_any_statement(statement_bytes(obj)):
            problems.append("the format-3 statement does not expand to the format-2 lowering")
    # IT-F1 / IT-F2: what the challenge demanded is covered; where each value lives is the store's word
    coverage = check_coverage(d, manifest or {}, record, obj)
    problems += coverage.get("problems") or []
    placement = check_placement(obj, st, bytes.fromhex(record["witness"]))
    problems += placement.get("problems") or []
    relations = "accept" if check_relations(obj, record["values"]) else "reject"
    if relations != "accept" or record.get("relations") != "accept":
        problems.append(f"relations: fresh {relations}, recorded {record.get('relations')}")
    witness = bytes.fromhex(record["witness"])
    if hashlib.sha256(witness).hexdigest() != hashlib.sha256((odir / "witness.bin").read_bytes()).hexdigest():
        problems.append("witness.bin differs from the record's witness")
    gate_set = family_gate_set(obj.family.id, obj.family.width)
    if gate_set is None:
        raise RuntimeError(f"no pinned gate set for {obj.family.id} at width {obj.family.width}")
    try:
        TransparentBackend(gate_set).verify(decode_any_statement(st), witness)
        witness_verdict = "accept"
    except Reject as e:
        witness_verdict = f"reject:{e.code.name}"
    if witness_verdict != "accept":
        problems.append(f"transparent backend on the witness: {witness_verdict}")
    comp = (record.get("provenance") or {}).get("composition") or {}
    row.data = {"h_O": h_o, "h_L": h_l, "statement_format": statement_format, "statement_v2_digest": record.get("statement_v2_digest"),
                "obligations": len(obj.obligations), "leaves": len(obj.leaves), "roots": len(obj.roots), "relations": relations,
                "witness_verdict": witness_verdict, "shared_x_rows": len(comp.get("shared_x_rows", [])), "shared_w_rows": len(comp.get("shared_w_rows", [])),
                "input_equations": comp.get("input_equations"), "coverage": coverage, "placement": placement, "problems": problems}
    row.status = "ok" if not problems else "FAIL"
    cov = (f"coverage {coverage['demanded_units']}/{coverage['demanded_units']} demanded units covered" if coverage.get("status") == "ok" else
           f"coverage {coverage.get('status')}" + (f" ({len(coverage['missing'])} demanded unit(s) missing)" if coverage.get("missing") else ""))
    row.detail = (f"{len(obj.obligations)} obligations over {len(obj.leaves)} leaves / {len(obj.roots)} roots; h_O {h_o[:16]} h_L {h_l[:16]} (statement format {statement_format}); {cov}; "
                  f"placement {placement['status']} ({placement['leaves_checked']} leaves at their authenticated rank); relations {relations}; witness {witness_verdict}")
    if problems:
        row.detail += f"; first problem: {problems[0]}"


def _host(host: str, out_log: Path, *hargs: str) -> dict[str, Any]:
    env = dict(os.environ)
    env["SP1_PROVER"] = "cpu"
    env.setdefault("RUST_LOG", "info")
    res = subprocess.run([host, *hargs], capture_output=True, text=True, env=env)
    out_log.write_text(res.stderr + "\n" + res.stdout)
    if res.returncode != 0:
        raise RuntimeError(f"host exited {res.returncode}: {res.stderr[-600:]}")
    return host_json(res.stdout)


def check_proofs(d: Path, manifest: dict[str, Any], row: Row, *, host: str | None, out: Path, allow_recorded: bool,
                 historical_v2: bool = False, approved_guests: Path | None = None) -> None:
    from veritor.interface import TypedObligationSet, check_binding

    pdir, odir = d / "proofs", d / "obligations"
    if not (pdir / "info.json").exists():
        row.status, row.detail = "skipped", "no proofs (the run stopped before the prove stage)"
        return
    info = _load(pdir / "info.json")
    record = _load(odir / "composed.json")
    obj = TypedObligationSet.from_json(record["object"])
    statement_format = int(record.get("statement_format", 2))  # h_L is recomputed from the object in the RECORDED format
    # the REQUIRED set is what the record's prove stage claims (prove-<mode>.json), not whatever
    # files happen to survive: a deleted required proof must FAIL here, never silently narrow
    required_modes = [m for m in ("core", "compressed") if (pdir / f"prove-{m}.json").exists()]
    modes = [m for m in required_modes if (pdir / f"proof-{m}.bin").exists()]
    rows: dict[str, Any] = {}
    problems = [f"required proof missing: proof-{m}.bin (prove-{m}.json records it was produced)" for m in required_modes if m not in modes]
    # GUEST (S12 caveat, fail-closed): the recorded (ELF, vk) must be one the VERIFIER approves, whatever the record
    # claims about its features / build.  STRICT: a problem (GUEST_NOT_APPROVED); --historical-v2: a loud warning.
    from e2e.approved_guests import check_guest

    guest, g_problems, g_warnings = check_guest(info.get("elf_sha256"), info.get("vk_hash"), (manifest.get("inputs") or {}).get("guest_build"),
                                               strict=not historical_v2, policy_path=approved_guests, what="guest")
    problems.extend(g_problems)
    for w in g_warnings:
        print(f"[verify_composed] WARNING: {w}", file=sys.stderr)
    if host and Path(host).exists():
        fresh_info = _host(host, out / "host-info.log", "info")
        if fresh_info.get("vk_hash") != info.get("vk_hash") or fresh_info.get("elf_sha256") != info.get("elf_sha256"):
            problems.append(f"host binary differs from the recorded one: vk {str(fresh_info.get('vk_hash'))[:16]} vs {str(info.get('vk_hash'))[:16]}")
        for m in modes:
            t = time.perf_counter()
            vr = _host(host, out / f"host-verify-{m}.log", "verify", "--proof", str(pdir / f"proof-{m}.bin"), "--statement", str(odir / "statement.bin"))
            pv = bytes.fromhex(vr["public_values"])
            b = check_binding(obj, pv, statement_format=statement_format)
            ok = bool(vr.get("ok")) and bool(vr.get("statement_match")) and bool(vr.get("verdict")) and not vr.get("unsound") and vr.get("vk_hash") == info.get("vk_hash") and b.ok
            rows[m] = {"re_verified": True, "ok": ok, "host_ok": vr.get("ok"), "statement_match": vr.get("statement_match"), "verdict": vr.get("verdict"), "unsound": vr.get("unsound"),
                       "vk_hash": vr.get("vk_hash"), "public_values_len": len(pv), "binding": {"ok": b.ok, "code": str(b.code or ""), "verdict": b.verdict},
                       "proof_bytes": (pdir / f"proof-{m}.bin").stat().st_size, "proof_sha256": sha256_file(pdir / f"proof-{m}.bin"), "seconds": round(time.perf_counter() - t, 2)}
            if not ok:
                problems.append(f"{m}: host ok={vr.get('ok')} statement_match={vr.get('statement_match')} verdict={vr.get('verdict')} binding={b.ok}/{b.code}")
        row.status = "ok" if not problems else "FAIL"
        row.detail = f"{len(modes)} proof(s) re-verified by the host (SP1_PROVER=cpu), vk {str(info.get('vk_hash'))[:16]}" + (f"; {len(problems)} problem(s)" if problems else "")
    else:
        for m in modes:
            vr = _load(pdir / f"verify-{m}.json")
            pv = bytes.fromhex(vr["public_values"])
            b = check_binding(obj, pv, statement_format=statement_format)
            ok = bool(vr.get("ok")) and bool(vr.get("statement_match")) and b.ok and vr.get("vk_hash") == info.get("vk_hash")
            rows[m] = {"re_verified": False, "recorded_ok": vr.get("ok"), "statement_match": vr.get("statement_match"), "binding_rechecked": {"ok": b.ok, "code": str(b.code or ""), "verdict": b.verdict},
                       "proof_bytes": (pdir / f"proof-{m}.bin").stat().st_size, "proof_sha256": sha256_file(pdir / f"proof-{m}.bin")}
            if not ok:
                problems.append(f"{m}: recorded ok={vr.get('ok')} binding={b.ok}")
        if allow_recorded:
            row.status = "ok" if not problems else "FAIL"
            row.detail = "no host binary: proofs NOT re-verified; recorded host verification re-read and binding re-checked on the recorded public values (--allow-recorded-proofs)"
        else:
            row.status = "not re-verified"
            row.detail = "no host binary (--host): proofs not re-verified; pass --allow-recorded-proofs to accept the recorded host verification"
    if not guest["approval"]["ok"]:
        row.detail += f"; guest {'REFUSED' if not historical_v2 else 'NOT APPROVED (warning only under --historical-v2)'}: {guest['approval']['detail']}"
    else:
        row.detail += f"; guest approved ({guest['approval']['approval']})"
    row.data = {"modes": rows, "vk_hash": info.get("vk_hash"), "elf_sha256": info.get("elf_sha256"), "sp1_version": info.get("sp1_version"),
                "guest_features_claimed": info.get("guest_features"), "statement_format": statement_format, "guest": {**guest, "warnings": g_warnings}, "problems": problems}


_BY_DESIGN = re.compile(r"^(?:first-stage: (?:ru/)?|output linkage )([^/:\s]+)")


def _selective_fresh_check(tar: Path, *, seed: int, sample_size: int, out_dir: Path, weights_values: Path | None, selected: list[str]) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """The fresh CPU bundle verifier (``verity_vllm.verify.verify``, WB) over a *selective* bundle (RC): the sample
    is drawn over J only (``request_ids=selected``; the requests outside J have a committed boundary and, by
    design, no interior).  WB's whole-bundle first-stage and output-linkage invariants are stated for a bundle
    where every request is replayed, so their findings about requests *outside J* are the selective design, not
    a defect; they are separated out and returned (``by_design``) and the row's verdict is recomputed without
    them -- every other structural finding, every authentication, relation and negative counts as in the full
    case.  First-stage completeness *over J* is ``selective.first_stage_problems_for`` (bundle rederive).
    Returns ``(reverify doc, table-1 row, by-design findings)``; the doc records both verdicts."""
    from verity_vllm import bundle as vbundle
    from verity_vllm import verify as vverify
    from verity_vllm.negatives import NEGATIVE_IDS

    out_dir.mkdir(parents=True, exist_ok=True)
    unpacked = vbundle.unpack(tar, out_dir / "unpacked")
    recorded = unpacked.manifest.get("recorded_verification") or {}
    if weights_values is not None:
        wdir = unpacked.bundle_dir / "weights"
        wdir.mkdir(exist_ok=True)
        shutil.copyfile(weights_values, wdir / "values.bin")
    if not selected:
        raise RuntimeError("selective bundle without recorded selected requests (BUNDLE.json recorded_verification.selected_requests)")
    result = vverify.verify(unpacked.bundle_dir, seed=seed, sample_size=sample_size, out_dir=out_dir / "verify", production_record=unpacked.production_record,
                            torch_mode="never", plugins=list(vverify.DEFAULT_PLUGINS), negatives=list(NEGATIVE_IDS), request_ids=selected)
    chosen = set(selected)
    by_design, real = [], []
    for p in result.structural_problems:
        m = _BY_DESIGN.match(p)
        (by_design if (m and m.group(1) not in chosen) else real).append(p)
    t1 = dict(result.table1)
    t1["structural"] = {"result": "ok" if not real else "fail", "failing_invariant": real[0] if real else None,
                        "whole_bundle_result": (result.table1.get("structural") or {}).get("result"),
                        "by_design_outside_J": len(by_design)}
    # the verdict without the by-design findings: what WB's verifier would say of a bundle in which J were every request
    generic_verdict, generic_blocker = result.verdict, result.table1.get("first_blocker")
    statuses = result.report.statuses()
    auth = result.report.authentication()
    if real:
        verdict, blocker = "FAIL", f"structural: {real[0]}"
    elif auth.get("attempted", 0) != auth.get("valid", -1):
        verdict, blocker = "FAIL", "authentication"
    elif statuses.get("mismatch") or statuses.get("auth_failed"):
        verdict, blocker = "FAIL", f"relation: {statuses}"
    elif (t1.get("negative") or {}).get("attempted", 0) != (t1.get("negative") or {}).get("rejected", -1):
        verdict, blocker = "FAIL", "negatives: not every mutation rejected"
    elif any(s not in ("exact",) for s in statuses):
        verdict, blocker = "PARTIAL", generic_blocker if "no executable checker" in str(generic_blocker) else "relation: no executable checker (torch never)"
    else:
        verdict, blocker = "PASS", None
    doc = {
        "tarball": str(tar), "name": unpacked.manifest["name"], "profile_id": result.table1.get("profile_id"), "seed": seed, "recorded": recorded or None,
        "fresh": {"verdict": verdict, "first_blocker": blocker, "challenge_digest": result.challenge.digest().hex(), "sample_size": len(result.sample),
                  "structural_problems": real, "generic_whole_bundle": {"verdict": generic_verdict, "first_blocker": generic_blocker, "structural_problems_by_design": by_design,
                                                                        "note": "WB's whole-bundle invariants applied to a selective bundle: findings about requests outside J are the design (no interior committed outside J)"}},
        "challenge_reproduced": None,
        "challenge_comparison": "not comparable: the recorded challenge is the selective protocol's (r_q, r_s from the verifier's secret seed); this is a fresh sample over J",
        "verdict_agrees": (verdict == recorded.get("verdict")) if recorded.get("verdict") else None,
    }
    write_json(out_dir / "reverify.json", doc)
    return doc, t1, by_design


def check_record_format(d: Path, manifest: dict[str, Any], row: Row) -> None:
    """INT-PAD-2's record-format additions (NOTE 15:20 + milestone reqs. 4/5), re-checked by a second
    process: the pre-challenge sampling policy (sha-bound in the manifest, realized selections must
    match the declaration), the served-token correspondence record, and the archived prover binaries.
    A record from before these fields is 'skipped' -- old records verify exactly as before."""
    inp = manifest.get("inputs") or {}
    pf, st = d / "sampling_policy.json", d / "served_tokens.json"
    if not pf.exists() and not st.exists() and not inp.get("archived_binaries"):
        row.status, row.detail = "skipped", "pre-INT-PAD-2 record (no sampling_policy.json / served_tokens.json / archived binaries)"
        return
    problems: list[str] = []
    detail: list[str] = []
    if pf.exists():
        if inp.get("sampling_policy_sha256") and sha256_file(pf) != inp["sampling_policy_sha256"]:
            problems.append("sampling_policy.json does not hash to manifest inputs.sampling_policy_sha256")
        pol = _load(pf)
        sv = d / "verify" / "selective_verify.json"
        if sv.exists():
            doc = _load(sv)
            if pol.get("selector") == "fixed_size" and doc.get("k") != pol.get("k"):
                problems.append(f"realized |J| = {doc.get('k')} != declared k = {pol.get('k')} (policy violated)")
            if doc.get("requests_total") not in (None, pol.get("requests_total")):
                problems.append(f"population {doc.get('requests_total')} != declared {pol.get('requests_total')}")
        detail.append(f"policy: {pol.get('selector')} q={pol.get('q_declared')} s={pol.get('s_declared')} bound {pol.get('bound_utc')}")
    else:
        problems.append("manifest carries sampling_policy_sha256 but sampling_policy.json is missing" if inp.get("sampling_policy_sha256") else "")
        problems = [p for p in problems if p]
    if st.exists():
        if inp.get("served_tokens_sha256") and sha256_file(st) != inp["served_tokens_sha256"]:
            problems.append("served_tokens.json does not hash to manifest inputs.served_tokens_sha256")
        doc = _load(st)
        if not doc.get("ok"):
            problems.append(f"served-token correspondence recorded NOT ok: {(doc.get('mismatches') or ['?'])[:2]}")
        bad = [e["request_id"] for e in (doc.get("requests") or []) if e.get("output_match") is False or e.get("prompt_match") is False]
        if bad:
            problems.append(f"served/committed token mismatch in {bad[:3]}")
        detail.append(f"served tokens: {len(doc.get('requests') or [])} requests, ok={doc.get('ok')}")
    for name, sha in (inp.get("archived_binaries") or {}).items():
        b = d / "bin" / name
        if not b.exists():
            problems.append(f"archived binary bin/{name} missing")
        elif sha256_file(b) != sha:
            problems.append(f"archived binary bin/{name} does not hash to the manifest pin")
        else:
            detail.append(f"bin/{name} sha ok")
    row.status = "ok" if not problems else "FAIL"
    row.detail = "; ".join(problems or detail)


def check_instance(d: Path, manifest: dict[str, Any], row: Row, *, historical_v2: bool) -> None:
    """[PROTO-A v3, brief A.4] the pre-challenge protocol instance + digest-pinned compact whole-program validation artifact +
    population (+ chronology for a Bernoulli law) are REQUIRED; missing or inconsistent evidence FAILs the row.  A record from
    before the instance verifies only under --historical-v2, which marks the row as attesting no instance (not 'ok')."""
    from e2e.instance_check import check_instance_evidence

    if historical_v2:
        row.status, row.detail = "UNCERTIFIED", "LEGACY --historical-v2 PATH: NO protocol instance attested, v3 mandatory checks NOT run, NO Bound (integration evidence only)"
        row.data = {"attests_protocol_instance": False}
        return
    problems, summary = check_instance_evidence(d, manifest)
    row.data = {"attests_protocol_instance": not problems, **summary, "problems": problems}
    inst = summary.get("instance") or {}
    row.status = "ok" if not problems else "FAIL"
    row.detail = "; ".join(problems) if problems else (f"instance {inst.get('digest', '')[:16]} law {inst.get('law')} N_V {inst.get('N_V')} N_R {inst.get('N_R')}; "
                                                        f"compact validation {((summary.get('validation') or {}).get('status'))}; chronology "
                                                        f"{'re-derived' if (summary.get('chronology') or {}).get('events') else 'n/a'}")


def check_bundle(d: Path, summary: dict[str, Any], row: Row, *, out: Path, sample_size: int) -> None:
    from verity_vllm import bundle as vbundle

    bst = (summary.get("stages") or {}).get("bundle") or {}
    tar = d / bst.get("artefacts", {}).get("tarball", "bundle/__missing__")
    if not tar.exists():
        row.status, row.detail = "skipped", "no tarball (the run stopped before the bundle stage)"
        return
    values = bst.get("artefacts", {}).get("weights_values")
    values_path = Path(values) if values else None
    if values_path is not None and not values_path.exists():
        # the recorded path is the run machine's (absolute); re-verification elsewhere finds values.bin at the
        # run-relative layout: INT's full replay (adapter/commit-v2) or RC's selective protocol (adapter/selective)
        values_path = next((alt for alt in (d / "adapter" / "commit-v2" / "weights" / "values.bin",
                                            d / "adapter" / "selective" / "weights" / "values.bin") if alt.exists()), None)
    bdir = out / "bundle"
    if bdir.exists():
        shutil.rmtree(bdir)
    bdir.mkdir(parents=True)
    problems = []
    recorded = vbundle.read_manifest(tar).get("recorded_verification") or {}
    # A staged (security-mode) challenge has no reproducible seed: the verifier drew its secret from the OS and
    # revealed it in the transcript.  Its re-verification is `rederive_transcript` (below).  The generic bundle
    # verifier still re-checks the row with a *fresh* sample; in security mode that sample is seeded from the
    # recorded challenge digest and declared not comparable (verify_tarball says so in `challenge_comparison`).
    seed = recorded.get("seed")
    seed_note = "the recorded (diagnostic) seed: the challenge must reproduce"
    if seed is None:
        seed = int.from_bytes(bytes.fromhex(recorded["challenge_digest"])[:8], "big") if recorded.get("challenge_digest") else 0
        seed_note = "security mode (no reproducible seed): a fresh sample seeded from the recorded challenge digest; the staged challenge itself is re-derived below"
    rd: dict[str, Any] | None = None
    try:
        t = time.perf_counter()
        rd = vbundle.rederive_transcript(tar, out_dir=bdir / "rederive")
        rd_seconds = round(time.perf_counter() - t, 1)
        if not rd.get("ok"):
            problems.append("staged challenge transcript did not re-derive")
    except vbundle.BundleError as e:
        rd_seconds = 0.0
        if recorded.get("challenge_mode") == "security":
            problems.append(f"rederive: {e}")
        else:
            rd = {"skipped": str(e)}
    selective = recorded.get("protocol") == "selective"
    selected = list(recorded.get("selected_requests") or [])
    t = time.perf_counter()
    if selective:
        rv, t1, by_design = _selective_fresh_check(tar, seed=int(seed), sample_size=sample_size, out_dir=bdir / "verify", weights_values=values_path, selected=selected)
    else:
        rv = vbundle.verify_tarball(tar, seed=int(seed), sample_size=sample_size, out_dir=bdir / "verify", weights_values=values_path)
        by_design = []
        t1_path = bdir / "verify" / "verify" / "table1.jsonl"
        t1 = [json.loads(l) for l in t1_path.read_text().splitlines() if l.strip()][-1] if t1_path.exists() else {}
    rv_seconds = round(time.perf_counter() - t, 1)
    # what the fresh CPU verifier could and could not check: without the runtime (no torch / vLLM here) the
    # same-runtime relations have "no executable checker" and the row is PARTIAL, not FAIL.  That PARTIAL is
    # accepted only when everything the process *could* check held: structure, every authentication, every
    # independent relation, every negative rejected, no mismatch -- and it is reported as such, not as PASS.
    fresh_verdict = rv["fresh"]["verdict"]
    auth, rel, neg, struct = t1.get("authentication") or {}, t1.get("relation_checks") or {}, t1.get("negative") or {}, t1.get("structural") or {}
    all_checked_held = (struct.get("result") == "ok" and auth.get("attempted", 0) == auth.get("valid", -1)
                        and all((v or {}).get("attempted", 0) == (v or {}).get("exact", -1) for v in rel.values())
                        and neg.get("attempted", 0) == neg.get("rejected", -1) and not rv["fresh"].get("structural_problems"))
    partial_ok = fresh_verdict == "PARTIAL" and "no executable checker" in str(rv["fresh"].get("first_blocker")) and all_checked_held
    if selective and rd is not None and not (rd.get("first_stage") or {}).get("ok", False):
        problems.append(f"first-stage over J did not hold: {((rd.get('first_stage') or {}).get('problems') or ['?'])[0]}")
    recorded_verdict = (rv.get("recorded") or {}).get("verdict")
    if recorded_verdict == "PASS":
        if not (fresh_verdict == "PASS" or partial_ok):
            problems.append(f"fresh bundle verdict {fresh_verdict} (first blocker {rv['fresh'].get('first_blocker')}) vs recorded PASS")
    elif rv.get("verdict_agrees") is not True:
        problems.append(f"fresh bundle verdict {fresh_verdict} vs recorded {recorded_verdict}")
    if rv.get("challenge_reproduced") is False:
        problems.append("challenge digest not reproduced")
    row.data = {"tarball": str(tar), "tarball_sha256": sha256_file(tar), "weights_values": str(values_path) if values_path else None,
                "fresh_verdict": fresh_verdict, "recorded_verdict": recorded_verdict, "first_blocker": rv["fresh"].get("first_blocker"),
                "fresh_checks": {"authentication": auth, "relation_checks": rel, "negatives": neg, "structural": struct},
                "partial_accepted_as": ("the runtime relations (same_runtime) have no executable checker without torch/vLLM; everything this process could check held"
                                        if partial_ok else None),
                "fresh_sample": {"seed": int(seed), "note": seed_note, "size": rv["fresh"].get("sample_size"),
                                 "restricted_to": (f"J = the {len(selected)} r_q-selected request(s) (the only ones with committed interiors)" if selective else "every request")},
                "protocol": "selective (RC)" if selective else "staged full replay",
                "selective_by_design": by_design or None,
                "challenge_reproduced": rv.get("challenge_reproduced"), "challenge_comparison": rv.get("challenge_comparison"),
                "verify_seconds": rv_seconds, "rederive": None if rd is None else {"ok": rd.get("ok"), "challenge_digest": rd.get("challenge_digest"), "population": rd.get("population"),
                                                                                 "statement": rd.get("statement"), "first_stage": rd.get("first_stage"), "protocol": rd.get("protocol"),
                                                                                 "seconds": rd_seconds, **({"skipped": rd["skipped"]} if "skipped" in rd else {})},
                "problems": problems}
    # the bundle row must agree with what the run recorded: a negative's recorded verdict is FAIL and the fresh one must be too
    row.status = "ok" if not problems else "FAIL"
    row.detail = (f"fresh CPU verifier {fresh_verdict} on {rv['fresh'].get('sample_size')} VUs (recorded {recorded_verdict}"
                  f"{'; PARTIAL = runtime relations not checkable here, all else held' if partial_ok else ''}): auth {auth.get('valid')}/{auth.get('attempted')}, "
                  f"independent {(rel.get('independent') or {}).get('exact')}/{(rel.get('independent') or {}).get('attempted')}, negatives {neg.get('rejected')}/{neg.get('attempted')}; "
                  f"staged challenge re-derived {None if rd is None else rd.get('ok')}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="a run_composed output directory")
    ap.add_argument("--host", default=str(DEFAULT_HOST) if DEFAULT_HOST.exists() else None, help="the SP1 host binary (verify runs on CPU)")
    ap.add_argument("--out", default=None, help="where to write reverify.json and the unpacked bundle (default DIR/reverify)")
    ap.add_argument("--allow-recorded-proofs", action="store_true", help="without a host binary, accept the recorded host verification (binding is still re-checked)")
    ap.add_argument("--skip-bundle", action="store_true", help="skip the tarball re-verification (it needs weights/values.bin beside the run)")
    ap.add_argument("--bundle-sample", type=int, default=64, help="VUs the fresh bundle verifier samples (CPU; ~5 s/VU for the v2 weights openings)")
    ap.add_argument("--program", choices=("rebuild", "recorded"), default="rebuild",
                    help="rebuild: reconstruct Serve{C,LP,STEPS} and compare (default); recorded: check the preserved descriptor's sha256/digest "
                         "and rebuild only the layer program (for B1 records on a machine that must not build the 410 MB descriptor)")
    ap.add_argument("--plan", choices=("rebuild", "recorded"), default="rebuild",
                    help="rebuild: re-validate every recorded plan unit (default); recorded: check the report's consistency only (for B1 records on a memory-bound machine)")
    ap.add_argument("--historical-v2", action="store_true",
                    help="verify a record that predates the protocol instance (the 'instance' row attests nothing instead of FAILing; "
                         "the reverify document says so). Without it, protocol_instance.json + compact_validation.json + population.json "
                         "(+ chronology.json for a Bernoulli law) are REQUIRED in the record.")
    ap.add_argument("--approved-guests", default=None,
                    help="the verifier's approved-guests policy (default e2e/approved_guests.json). Strict path: a recorded ELF/vk not in it FAILS the "
                         "proofs row (GUEST_NOT_APPROVED) regardless of claimed guest features; --historical-v2 warns loudly instead.")
    a = ap.parse_args(argv)
    d = Path(a.dir).resolve()
    out = Path(a.out).resolve() if a.out else d / "reverify"
    out.mkdir(parents=True, exist_ok=True)
    manifest = _load(d / "manifest.json")
    summary = _load(d / "summary.json")
    rows: dict[str, Row] = {}

    def run(name: str, fn, *args, **kw) -> None:
        row = Row(name)
        rows[name] = row
        t = time.perf_counter()
        try:
            fn(*args, row, **kw)
        except Exception as e:  # noqa: BLE001
            row.status, row.detail = "error", f"{type(e).__name__}: {e}"
        row.seconds = time.perf_counter() - t
        print(f"[verify_composed] {name:8s} {row.status:14s} {row.detail}", flush=True)

    run("program", check_program if a.program == "rebuild" else check_program_recorded, d, manifest, summary)
    run("plan", check_plan if a.plan == "rebuild" else check_plan_recorded, d, manifest, summary)
    run("records", check_records, d, manifest=manifest)
    run("format", check_record_format, d, manifest)
    run("instance", check_instance, d, manifest, historical_v2=a.historical_v2)
    run("proofs", check_proofs, d, manifest, host=a.host, out=out, allow_recorded=a.allow_recorded_proofs, historical_v2=a.historical_v2,
        approved_guests=Path(a.approved_guests) if a.approved_guests else None)
    if a.skip_bundle:
        rows["bundle"] = Row("bundle")
        rows["bundle"].status, rows["bundle"].detail = "skipped", "--skip-bundle"
    else:
        run("bundle", check_bundle, d, summary, out=out, sample_size=a.bundle_sample)

    recorded_verdict = summary.get("verdict")
    recorded_stages = {k: v.get("status") for k, v in (summary.get("stages") or {}).items()}
    # the fresh verdict: every row that ran is ok, AND the recorded run's own stages were ok.  A negative run
    # (recorded FAIL at some stage) is reached again when this process also fails or when the recorded stage
    # statuses show the failure (its proofs/records are absent, the bundle row FAILs).
    ran = [r for r in rows.values() if r.status not in ("skipped", "UNCERTIFIED")]
    incomplete = any(r.status == "not re-verified" for r in ran)
    fresh_ok = all(r.status == "ok" for r in ran) and all(v == "ok" for k, v in recorded_stages.items() if k in ("program", "plan", "production", "preparation", "verify", "obligations", "prove"))
    fresh_verdict = "INCOMPLETE" if incomplete else ("PASS" if fresh_ok else "FAIL")
    agrees = fresh_verdict == recorded_verdict
    if a.historical_v2:
        print("[verify_composed] ******** UNCERTIFIED: LEGACY --historical-v2 PATH -- no protocol instance attested; v3 mandatory checks NOT run; no Bound ********", flush=True)
    doc = {"schema": "verity-vllm/e2e-composed/reverify/v1", "dir": str(d), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           **({"UNCERTIFIED": "LEGACY --historical-v2 PATH: no protocol instance attested; v3 mandatory checks NOT run; no Bound; integration evidence only",
               "certification_status": "UNCERTIFIED"} if a.historical_v2 else {"certification_status": "v3-strict"}),
           "recorded_verdict": recorded_verdict, "recorded_stage_statuses": recorded_stages, "recorded_negative": (summary.get("scope") or {}).get("negative"),
           "fresh_verdict": fresh_verdict, "agrees": agrees, "rows": {k: v.to_json() for k, v in rows.items()},
           "process": {"python": sys.version.split()[0], "host": a.host, "gpu": False, "vllm": False, "checkpoint": False,
                       "program_reconstructed": a.program == "rebuild", "plan_revalidated": a.plan == "rebuild",
                       "attests_protocol_instance": (rows["instance"].data or {}).get("attests_protocol_instance", False),
                       "options": {"program": a.program, "plan": a.plan, "skip_bundle": a.skip_bundle,
                                   "allow_recorded_proofs": a.allow_recorded_proofs, "historical_v2": a.historical_v2}}}
    write_json(out / "reverify.json", doc)
    print(f"[verify_composed] {'UNCERTIFIED (legacy v2) ' if a.historical_v2 else ''}fresh verdict {fresh_verdict}, recorded {recorded_verdict}, agrees={agrees} -> {out / 'reverify.json'}")
    return 0 if agrees and fresh_verdict != "INCOMPLETE" else 1


if __name__ == "__main__":
    sys.exit(main())
