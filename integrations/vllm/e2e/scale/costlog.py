"""``costlog/v1`` -- the cost record of ONE protocol run for the paper table (E3, SCALE wave D46; `out/scale/table/SCHEMA.md` is the prose).

One JSON per (model, prompt x output, run id).  Every leaf is a FIELD ``{"value": v, "label": L, "source": s, "note": n?}`` with ``L`` in
:data:`LABELS` -- a number never travels without its label and its provenance (the user's rule: no borrowed numbers; unknown = ``null`` under
``"UNKNOWN"``).  Blocks: ``id``, ``status``, ``stages`` (one object per :data:`STAGES`), ``storage``, ``transfer`` (E9's split: the ARCHIVE check
that recomputes roots from archived leaves vs the DEPLOYED verifier's authenticated openings of the sampled units), ``guarantee`` (the sampling
parameters and the integrity bound in R1's words), ``sources``.

Programmatic use::

    doc = new("HuggingFaceTB/SmolLM2-135M", 4, 9, "n4")
    doc["stages"]["proving"]["wall_s"] = F(158.6, "MEASURED", "costlog_v5.json class proving")
    errors = validate(doc)            # [] when the document is well-formed
    dump(doc, "out/scale/table/SmolLM2-135M__4x9__n4.costlog.json")

CLI: ``python -m e2e.scale.costlog validate <file>...`` | ``python -m e2e.scale.costlog from-record --help``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping

SCHEMA = "veritor/scale/costlog/v1"

LABELS = ("MEASURED", "MODELED", "PROVISIONAL", "NOT RUN", "UNKNOWN")

#: the stages of one run, in protocol order (``status.stage_reached`` names the last one that completed)
STAGES = ("serve", "reconstruction", "commitment", "obligations", "proving", "verify_live", "reverify")
STAGE_TITLES = {
    "serve": "serving (the recorded pass of the request: vLLM + capture)",
    "reconstruction": "reconstruction (E1 producer: the interior rows of the retained RUs from the recorded boundary)",
    "commitment": "commitment (step 1 boundary trees + I/O, step 3 interior trees)",
    "obligations": "obligations (statements: positions + bodies + witness for the sampled units)",
    "proving": "proving (SP1 proofs of every statement)",
    "verify_live": "verification, live (the driver's step 5 on the run pod)",
    "reverify": "re-verification (the shipped verifier in a fresh environment over the archived record)",
}
STAGE_FIELDS = ("wall_s", "cpu_s", "peak_rss_bytes", "gpu_peak_bytes", "device", "bytes_read", "bytes_written", "bytes_hashed")
STORAGE_FIELDS = ("rows_bytes", "trees_bytes", "proofs_bytes", "record_bytes", "archive_tgz_bytes", "slim_bytes")
#: E9's buckets (agreed 21:10Z, out/scale/reports/e9.md): ARCHIVE work = everything that walks all leaves (`V3.trees_rehashed` + the whole-tree rebuilds a
#: deployed check triggers through TreeCache); DEPLOYED = authenticated openings of the sampled units, pins, proofs (a PART of deployed_sparse, printed on
#: its own line), statements, direct checks; `unclassified` = reads E9's bucket map did not attribute; `checkpoint` = the verifier's own safetensors.
_BUCKET = ("bytes_read", "bytes_hashed", "files_opened", "leaves_hashed", "nodes_hashed", "trees_rebuilt", "openings", "wall_s", "peak_rss_bytes",
           "ratio_bytes_read_to_record", "checks")
TRANSFER_MODES = {
    "archive_check": _BUCKET,
    "deployed_sparse": _BUCKET + ("sampled_units",),
    "unclassified": _BUCKET,
    "checkpoint": ("bytes_read", "bytes_hashed"),
}
#: raw pass-through keys of E9's `verify_accounting.json`, copied verbatim (not labelled fields): per_check, bucket_map, cross_check, deployed_ideal, rule
TRANSFER_RAW = ("per_check", "bucket_map", "cross_check", "deployed_ideal", "rule", "record_bytes", "source", "wall_s_total")
GUARANTEE_FIELDS = ("N_V", "N_RU", "s", "q", "delta", "B_ex", "W_bits", "U_bits", "U_precap_bits", "capped", "calculator",
                    "j_star", "beta",  # E6 (e6.md 21:15Z): j* = least j with (1-s)^j <= delta = the faulty-VU count the law detects w.p. >= 1-delta; beta = j*/N_V
                    "Q", "proof_checked_units", "directly_checked_units", "input_units", "uncovered", "n_statements", "n_proofs",
                    "epsilon_crypto", "bound_statement")
STATUS_FIELDS = ("stage_reached", "verdict_live", "verdict_reverify", "verdict_reverify_repaired", "verdict_independent", "refusal", "tree",
                 "record_path", "record_sha256", "utc", "history")
#: `status.stage_reached` may also name a pre-serve step: "none" (nothing ran), "instance" (E6's pin exists, no serve yet)
PSEUDO_STAGES = ("none", "instance")
ID_FIELDS = ("model", "model_short", "prompt_len", "output_len", "tokens", "run_id", "record_id", "architecture", "program_revision", "attention")
#: fields a later revision added as OPTIONAL: absent is allowed (the generator prints "--"); present = a labelled field like every other leaf.
#: `id.program_revision` (D54 §5) = the bound PROGRAM's digest (`protocol_instance.json#program.digest`, pinned by the record's freeze) -- records accepted
#: on the current program and re-runs on E7's fixed program (the digest changes for every size) are told apart by it.
#: `id.attention` (D57) = the attention kernel geometry the pinned program IDENTITY declares, as text: `protocol_instance.json#program.identity.attention`
#: = {kernel, kBlockM, kBlockN, ...} -> "<kernel> <kBlockM>×<kBlockN>" (e.g. "fa2-splitkv-align 64×128"); an identity that carries no geometry
#: (every pre-D54 record: `Serve_v1{C, LP, STEPS}`) -> "legacy BN=64" (`attention_geometry_text`).  Printed beside `program_revision`.
OPTIONAL_FIELDS = {"id": ("program_revision", "attention")}
LEGACY_ATTENTION_TEXT = "legacy BN=64"


def attention_geometry_text(identity: Mapping[str, Any] | None) -> str:
    """[D57] The one-line geometry of a `protocol_instance.json#program.identity`: its declared ``attention`` record printed as
    "<kernel> <kBlockM>×<kBlockN>", or ``LEGACY_ATTENTION_TEXT`` when the identity carries none (the D57 default: the `Serve_v1` bodies'
    module constant kBlockN = 64).  A malformed record (not a dict / no kBlockN) prints what it has, marked "?" -- never silently the legacy text."""
    att = (identity or {}).get("attention") if isinstance(identity, Mapping) else None
    if att is None:
        return LEGACY_ATTENTION_TEXT
    if not isinstance(att, Mapping) or att.get("kBlockN") is None:
        return f"? {att!r}"
    return f"{att.get('kernel') or '?'} {att.get('kBlockM') if att.get('kBlockM') is not None else '?'}×{att['kBlockN']}"


def attach_program_identity(doc: dict[str, Any], program: Mapping[str, Any], source: str, note: str | None = None) -> dict[str, Any]:
    """[D54 §5 + D57] Fill ``id.program_revision`` (the digest) and ``id.attention`` (the identity's geometry text) from a
    `protocol_instance.json#program` record ``{"identity": {...}, "digest": <hex>}``; ``source`` names the file.  Fields already carrying a
    value are left alone (the record's own instance beats E6's bound)."""
    ident = program.get("identity") if isinstance(program, Mapping) else None
    dig = program.get("digest") if isinstance(program, Mapping) else None
    if dig and (doc["id"].get("program_revision") or {}).get("value") is None:
        doc["id"]["program_revision"] = F(str(dig), "MEASURED", source + "#program.digest", note)
    if isinstance(ident, Mapping) and (doc["id"].get("attention") or {}).get("value") is None:
        fam = ident.get("family") or "Serve_v1"
        doc["id"]["attention"] = F(attention_geometry_text(ident), "MEASURED", source + "#program.identity",
                                   f"{fam}: " + ("identity.attention as declared (D57)" if ident.get("attention") is not None
                                                  else "the identity carries no geometry -> the D57 default, Serve_v1's kBlockN = 64"))
    return doc
VERDICTS = ("VERIFIED", "REFUSED", "ABORTED", None)


def F(value: Any, label: str, source: str | None = None, note: str | None = None) -> dict[str, Any]:
    """A labelled field.  ``label`` must be one of :data:`LABELS`; a ``None`` value is ``UNKNOWN`` or ``NOT RUN`` only."""
    if label not in LABELS:
        raise ValueError(f"label {label!r} not in {LABELS}")
    if value is None and label not in ("UNKNOWN", "NOT RUN"):
        raise ValueError(f"a null value must be labelled UNKNOWN or NOT RUN, not {label}")
    d: dict[str, Any] = {"value": value, "label": label, "source": source}
    if note:
        d["note"] = note
    return d


def unknown(note: str | None = None) -> dict[str, Any]:
    return F(None, "UNKNOWN", None, note)


def not_run(note: str | None = None) -> dict[str, Any]:
    return F(None, "NOT RUN", None, note)


def new(model: str, prompt_len: int, output_len: int, run_id: str, *, record_id: str | None = None, architecture: str | None = None,
        model_short: str | None = None) -> dict[str, Any]:
    """A costlog with every field ``UNKNOWN`` except the identity."""
    short = model_short or model.split("/")[-1]
    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "id": {
            "model": F(model, "MEASURED", "declared"),
            "model_short": F(short, "MEASURED", "declared"),
            "prompt_len": F(int(prompt_len), "MEASURED", "declared"),
            "output_len": F(int(output_len), "MEASURED", "declared"),
            "tokens": F(int(prompt_len) + int(output_len), "MEASURED", "prompt_len + output_len (tokens the record covers: prefill + every decode step)"),
            "run_id": F(str(run_id), "MEASURED", "declared"),
            "record_id": F(record_id, "MEASURED", "declared") if record_id else unknown(),
            "architecture": F(architecture, "MEASURED", "declared") if architecture else unknown(),
        },
        "status": {k: unknown() for k in STATUS_FIELDS},
        "stages": {st: {k: unknown() for k in STAGE_FIELDS} for st in STAGES},
        "storage": {k: unknown() for k in STORAGE_FIELDS},
        "transfer": {mode: {k: unknown() for k in keys} for mode, keys in TRANSFER_MODES.items()},
        "guarantee": {k: unknown() for k in GUARANTEE_FIELDS},
        "sources": {},
    }
    return doc


# --- validation -----------------------------------------------------------------------------------------------------------------------------------

def _is_field(x: Any) -> bool:
    return isinstance(x, dict) and "value" in x and "label" in x and x.get("label") in LABELS


def validate(doc: dict[str, Any]) -> list[str]:
    """Every structural rule of the schema; ``[]`` means well-formed.  Values are not judged (a MODELED number is allowed anywhere), only
    their form: every leaf a labelled field, every block present with exactly the schema's keys (extra keys under ``detail`` / ``sources`` are free)."""
    errs: list[str] = []
    if doc.get("schema") != SCHEMA:
        errs.append(f"schema: {doc.get('schema')!r} != {SCHEMA!r}")

    def block(name: str, keys: Iterable[str], obj: Any) -> None:
        if not isinstance(obj, dict):
            errs.append(f"{name}: not an object")
            return
        optional = OPTIONAL_FIELDS.get(name, ())
        for k in keys:
            if k not in obj:
                if k not in optional:
                    errs.append(f"{name}.{k}: missing")
            elif not _is_field(obj[k]):
                errs.append(f"{name}.{k}: not a labelled field {{value,label,source}} with label in {LABELS}")
        for k in obj:
            if k not in keys and k not in ("detail", "parts"):
                errs.append(f"{name}.{k}: unknown key (put extras under {name}.detail)")
        det = obj.get("detail")
        if det is not None:
            if not isinstance(det, dict):
                errs.append(f"{name}.detail: not an object")
            else:
                for k, v in det.items():
                    if not _is_field(v):
                        errs.append(f"{name}.detail.{k}: not a labelled field")
        parts = obj.get("parts")  # transfer buckets: {part name: {field: labelled field}} (E9's parts: proofs, statements, openings, pins, ...)
        if parts is not None:
            if not isinstance(parts, dict):
                errs.append(f"{name}.parts: not an object")
            else:
                for pn, part in parts.items():
                    if not isinstance(part, dict) or not all(_is_field(v) for v in part.values()):
                        errs.append(f"{name}.parts.{pn}: must be an object of labelled fields")

    block("id", ID_FIELDS, doc.get("id"))
    block("status", STATUS_FIELDS, doc.get("status"))
    stages = doc.get("stages")
    if not isinstance(stages, dict):
        errs.append("stages: not an object")
    else:
        for st in STAGES:
            if st not in stages:
                errs.append(f"stages.{st}: missing")
            else:
                block(f"stages.{st}", STAGE_FIELDS, stages[st])
        for st in stages:
            if st not in STAGES:
                errs.append(f"stages.{st}: not a schema stage {STAGES}")
    block("storage", STORAGE_FIELDS, doc.get("storage"))
    def transfer_block(name: str, tr: Any, allow_deployed: bool) -> None:
        if not isinstance(tr, dict):
            errs.append(f"{name}: not an object")
            return
        for mode, keys in TRANSFER_MODES.items():
            if mode not in tr:
                errs.append(f"{name}.{mode}: missing")
            else:
                block(f"{name}.{mode}", keys, tr[mode])
        for mode in tr:
            if mode not in TRANSFER_MODES and mode not in TRANSFER_RAW and not (allow_deployed and mode == "deployed_mode"):
                errs.append(f"{name}.{mode}: unknown mode (buckets {tuple(TRANSFER_MODES)}, raw pass-through {TRANSFER_RAW}, deployed_mode)")
        pc = tr.get("per_check")
        if pc is not None and not (isinstance(pc, dict) and all(isinstance(v, dict) for v in pc.values())):
            errs.append(f"{name}.per_check: must be {{check_id: {{bytes_read, bytes_hashed, files_opened, wall_s, bucket, part, ...}}}} (E9's table verbatim)")
        if allow_deployed and tr.get("deployed_mode") is not None:
            transfer_block(f"{name}.deployed_mode", tr["deployed_mode"], False)

    # `transfer` = the default entry point (archive check ON); `transfer.deployed_mode` = the same buckets for `reverify.py --deployed-openings`
    transfer_block("transfer", doc.get("transfer"), True)
    block("guarantee", GUARANTEE_FIELDS, doc.get("guarantee"))
    if not isinstance(doc.get("sources"), dict):
        errs.append("sources: not an object")
    try:  # value-form checks on whatever blocks are present (a structural defect elsewhere does not hide them)
        st = doc["status"]
        sr = st["stage_reached"]["value"]
        if sr is not None and sr not in STAGES and sr not in PSEUDO_STAGES:
            errs.append(f"status.stage_reached: {sr!r} not in {STAGES} or {PSEUDO_STAGES}")
        for k in ("verdict_live", "verdict_reverify", "verdict_reverify_repaired", "verdict_independent"):
            if st[k]["value"] not in VERDICTS:
                errs.append(f"status.{k}: {st[k]['value']!r} not in {VERDICTS}")
        for st_name, stage in doc["stages"].items():
            for k in ("wall_s", "cpu_s", "peak_rss_bytes", "gpu_peak_bytes", "bytes_read", "bytes_written", "bytes_hashed"):
                v = (stage.get(k) or {}).get("value") if isinstance(stage, dict) else None
                if v is not None and not (isinstance(v, (int, float)) and v >= 0):
                    errs.append(f"stages.{st_name}.{k}: {v!r} is not a non-negative number")
        for k in STORAGE_FIELDS:
            v = doc["storage"][k]["value"]
            if v is not None and not (isinstance(v, int) and v >= 0):
                errs.append(f"storage.{k}: {v!r} is not a non-negative integer (bytes)")
        g = doc["guarantee"]
        for k in ("s", "q", "delta"):
            v = g[k]["value"]
            if v is not None:
                try:
                    Fraction(str(v))
                except (ValueError, ZeroDivisionError):
                    errs.append(f"guarantee.{k}: {v!r} is not a fraction string")
        if g["bound_statement"]["value"] is not None and g["U_bits"]["value"] is None:
            errs.append("guarantee.bound_statement present without guarantee.U_bits")
    except (KeyError, TypeError, AttributeError):
        pass  # already reported structurally above
    return errs


def load(path: str | Path) -> dict[str, Any]:
    return json.load(open(path))


def dump(doc: dict[str, Any], path: str | Path) -> None:
    errs = validate(doc)
    if errs:
        raise ValueError("costlog does not validate:\n  " + "\n  ".join(errs))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(doc, indent=1, sort_keys=False) + "\n")


def value(doc: dict[str, Any], dotted: str, default: Any = None) -> Any:
    """``value(doc, "stages.proving.wall_s")`` -> the field's value (``default`` when absent or null)."""
    cur: Any = doc
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    if _is_field(cur):
        return default if cur["value"] is None else cur["value"]
    return cur


def label(doc: dict[str, Any], dotted: str) -> str:
    cur: Any = doc
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return "UNKNOWN"
        cur = cur[part]
    return cur.get("label", "UNKNOWN") if isinstance(cur, dict) else "UNKNOWN"


# --- the integrity bound in R1's words --------------------------------------------------------------------------------------------------------------

def _frac(v: Any) -> str:
    if v is None:
        return "?"
    try:
        f = Fraction(str(v))
    except (ValueError, ZeroDivisionError):
        return str(v)
    if f.denominator == 1:
        return str(f.numerator)
    if f.numerator == 1 and (f.denominator & (f.denominator - 1)) == 0:
        return f"2^-{f.denominator.bit_length() - 1}"
    if f.numerator == 1 and f.denominator % 10 == 0 and str(f.denominator).strip("0") == "1":
        return f"1e-{len(str(f.denominator)) - 1}"
    return f"{f.numerator}/{f.denominator}"


def bound_statement(g: dict[str, Any]) -> str | None:
    """R1's per-configuration statement (`docs/theory/statements-2026-09-11.md` §1.1, §1.5 root-only, §5) with THIS run's numbers plugged in.
    ``None`` when ``U_bits`` is unknown (E6 supplies U per point; nothing is invented)."""
    def v(k: str) -> Any:
        x = g.get(k)
        return x.get("value") if isinstance(x, dict) else x
    U, W = v("U_bits"), v("W_bits")
    if U is None:
        return None
    s, q, delta, bex, NV, NR = _frac(v("s")), _frac(v("q")), _frac(v("delta")), v("B_ex"), v("N_V"), v("N_RU")
    Pn = v("n_proofs")
    cap = ""
    if W is not None and U == W:
        cap = f" -- U = W: the cap case (PROVED, R1 §5.3: a completion with a single faulty VU escapes with probability 1 - s = 1 - {s} > delta, so S is the whole output domain; only s = 1 gives U < W for this population)"
    elif W is not None:
        cap = f" (W = {W} declared output bits)"
    NVs = f"{NV:,}" if isinstance(NV, int) else str(NV)
    return (
        f"(U, delta + eps_crypto)-integrity, ROOT-ONLY mode (R1 §1.1, §1.5): for every prover strategy and every output y outside a set S with "
        f"log2|S| <= U = {U} bits{cap}, Pr[accept y] <= delta + eps_crypto = {delta} + eps_crypto, where eps_crypto is the NAMED sum "
        f"eps_CR(B1) + sqrt(n_RO * eps_CR(B2)) + P_n * eps_KS + eps_PRF + 2^-189 (P_n = {Pn if Pn is not None else '?'} proof objects; no number is attached "
        f"beyond the assumed generic-attack estimates). S = {{out(v): v a full completion with a(v) > delta}}, escape law a(v) = E_J[max_{{F ⊆ J, |F| <= B_ex}} "
        f"(1 - s)^(sum_{{R in J \\ F}} j_R(v))] with q = {q}, s = {s}, B_ex = {bex}, N_R = {NR}, N_V = {NVs}; "
        f"U = certified_bound(summary, VerificationPolicy(q, s), eta = delta, allowance = B_ex).bits. CONDITIONAL on (CR) SHA-256, (KS) SP1 for every "
        f"approved vk the record's proofs use, (PRF) + seed custody, (LAW), (LOW) the lowering theorems of every VU kind in Q, (POS) the position oracle."
    )


# --- from a record_v5 record ------------------------------------------------------------------------------------------------------------------------

def _classes(costlog_v5: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for r in costlog_v5.get("rows", []):
        out[r["class"]] = out.get(r["class"], 0.0) + float(r.get("seconds") or 0.0)
    return out


def _rows(costlog_v5: dict[str, Any], cls: str) -> list[dict[str, Any]]:
    return [r for r in costlog_v5.get("rows", []) if r.get("class") == cls]


def _phase_wall(costlog_v5: dict[str, Any], phase: str) -> float | None:
    for r in costlog_v5.get("rows", []):
        if r.get("class") == "wall_only" and r.get("stage") == f"run_v5 phase wall: {phase}":
            return float(r["seconds"])
    return None


def _gb(x: Any) -> int | None:
    return None if x is None else int(round(float(x) * 1e9))


def parse_time_v(text: str) -> dict[str, float]:
    """``/usr/bin/time -v`` output -> ``{wall_s, user_s, sys_s, max_rss_bytes, fs_inputs, fs_outputs}`` (GNU time; RSS in KB there)."""
    out: dict[str, float] = {}
    m = re.search(r"Elapsed \(wall clock\) time.*?:\s*([\d:.]+)", text)
    if m:
        parts = [float(p) for p in m.group(1).split(":")]
        out["wall_s"] = sum(p * 60 ** i for i, p in enumerate(reversed(parts)))
    for key, pat in (("user_s", r"User time \(seconds\):\s*([\d.]+)"), ("sys_s", r"System time \(seconds\):\s*([\d.]+)"),
                     ("fs_inputs", r"File system inputs:\s*(\d+)"), ("fs_outputs", r"File system outputs:\s*(\d+)")):
        m = re.search(pat, text)
        if m:
            out[key] = float(m.group(1))
    m = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", text)
    if m:
        out["max_rss_bytes"] = float(m.group(1)) * 1024
    return out


def from_record_v5(*, model: str, prompt_len: int, output_len: int, run_id: str, record_id: str | None = None, architecture: str | None = None,
                   costlog_v5: dict[str, Any] | None = None, verify_v5: dict[str, Any] | None = None, reverify_clean: dict[str, Any] | None = None,
                   coverage: dict[str, Any] | None = None, certificate: dict[str, Any] | None = None, reverify_time_v: str | None = None,
                   sources: dict[str, str] | None = None) -> dict[str, Any]:
    """One costlog from the artefacts a `record_v5` run writes.  Everything it can read is MEASURED with its file named as the source; everything
    else stays UNKNOWN for the caller to fill (storage sizes, the guarantee's U, E9's transfer accounting)."""
    src = dict(sources or {})
    doc = new(model, prompt_len, output_len, run_id, record_id=record_id, architecture=architecture)
    S = doc["stages"]
    cl5 = src.get("costlog_v5", "costlog_v5.json")
    if costlog_v5:
        hw = str(costlog_v5.get("hardware") or "")
        C = _classes(costlog_v5)
        gpu = hw.split(",")[0].strip() if hw else None
        # serve: the recorded pass of the request (class 2 `execution`); the native baseline and the whole phase wall go to detail
        ex = _rows(costlog_v5, "execution")
        if ex:
            S["serve"]["wall_s"] = F(round(C["execution"], 6), "MEASURED", cl5, "class 2 `execution`: the recorded pass (matched C, batch-invariant eager), engine build and warm-up excluded")
            S["serve"]["device"] = F(gpu, "MEASURED", cl5)
            nat = [r for r in _rows(costlog_v5, "native") if str(r.get("stage", "")).startswith("native baseline")]
            S["serve"]["detail"] = {"native_s": F(round(sum(float(r["seconds"]) for r in nat), 6), "MEASURED", cl5, "class 1 native baseline: stock vLLM, same request, recording off (serving side only)") if nat else unknown(),
                                    "phase_wall_s": F(_phase_wall(costlog_v5, "serve"), "MEASURED" if _phase_wall(costlog_v5, "serve") is not None else "UNKNOWN", cl5, "run_v5 serve phase incl. engine build, warm-up, reps")}
        # reconstruction: class V value production (E1 producer, other hardware)
        vp = _rows(costlog_v5, "value_production")
        if vp:
            S["reconstruction"]["wall_s"] = F(round(C["value_production"], 3), "MEASURED", cl5, vp[0].get("note") or vp[0].get("stage"))
            dev = re.search(r"\(([^()]*vCPU[^()]*)\)", str(vp[0].get("note") or "") + " " + str(vp[0].get("hardware") or ""))
            S["reconstruction"]["device"] = F(dev.group(1) if dev else (vp[0].get("hardware") or None), "MEASURED" if (dev or vp[0].get("hardware")) else "UNKNOWN", cl5, "CPU pod of the rows job (not the serve GPU)")
        # commitment: class 3 commit_io (phase 0 boundary + I/O) + class 5 interior commitments; the selective_prover process RSS
        if "serving" in C or "commit_io" in C or "interior_commit" in C:
            b = C.get("commit_io", C.get("serving", 0.0))
            i = C.get("interior_commit", 0.0)
            S["commitment"]["wall_s"] = F(round(b + i, 3), "MEASURED", cl5, "class 3 (step-1 boundary commitments + I/O checks) + class 5 (step-3 interior commitments)")
            S["commitment"]["detail"] = {"boundary_io_s": F(round(b, 3), "MEASURED", cl5), "interior_s": F(round(i, 3), "MEASURED", cl5),
                                         "replay_s": F(round(float((costlog_v5.get("classes") or {}).get("replay", {}).get("seconds", sum(float(r["seconds"]) for r in _rows(costlog_v5, "replay") if "(phase 2)" not in str(r.get("stage", ""))))), 3), "MEASURED", cl5, "class 4 replay (forced replay of J + boundary re-hash + phase-2 residual; the whole-phase row excluded)")}
            pt = costlog_v5.get("prover_time") or {}
            if pt.get("peak_rss_gb") is not None:
                S["commitment"]["peak_rss_bytes"] = F(_gb(pt["peak_rss_gb"]), "MEASURED", cl5, "prover_time: the whole selective_prover process (phase 0 + phase 2 + interior trees), /usr/bin/time -v")
                S["commitment"]["cpu_s"] = F(round(float(pt.get("user_s", 0)) + float(pt.get("sys_s", 0)), 2), "MEASURED", cl5, "prover_time user + sys")
            S["commitment"]["device"] = F(hw or None, "MEASURED" if hw else "UNKNOWN", cl5)
        # obligations: class prove_witness (registry statements + openings + witness)
        if "prove_witness" in C:
            S["obligations"]["wall_s"] = F(round(C["prove_witness"], 3), "MEASURED", cl5, "class prove_witness: registry statements (E7 builders over E3 positions) + committed-row openings + witness bytes")
            S["obligations"]["device"] = F(hw or None, "MEASURED" if hw else "UNKNOWN", cl5)
            S["obligations"]["detail"] = {"phase_wall_s": F(_phase_wall(costlog_v5, "obligations"), "MEASURED" if _phase_wall(costlog_v5, "obligations") is not None else "UNKNOWN", cl5)}
        # proving: class proving (one SP1 process per chunk)
        pr = _rows(costlog_v5, "proving")
        if pr:
            S["proving"]["wall_s"] = F(round(C["proving"], 3), "MEASURED", cl5, f"class proving: sum of {len(pr)} chunk proves (compressed, SP1 cuda)")
            S["proving"]["device"] = F(gpu, "MEASURED", cl5)
            S["proving"]["detail"] = {"n_chunks": F(len(pr), "MEASURED", cl5), "phase_wall_s": F(_phase_wall(costlog_v5, "prove"), "MEASURED" if _phase_wall(costlog_v5, "prove") is not None else "UNKNOWN", cl5, "run_v5 prove phase wall (incl. process start-up per chunk)")}
        # verify_live: classes verify_proofs + verifier_expected (reverify rows) + verifier record checks
        vp_ = _rows(costlog_v5, "verify_proofs")
        if vp_ or verify_v5:
            det = {}
            if vp_:
                det["proof_verification_s"] = F(round(C["verify_proofs"], 3), "MEASURED", cl5, "V4.*: vk derivation once + host verify + binding of every proof")
            ve = [r for r in _rows(costlog_v5, "verifier_expected") if "reverify" in str(r.get("stage", ""))]
            if ve:
                det["expected_statements_s"] = F(round(sum(float(r["seconds"]) for r in ve), 3), "MEASURED", cl5, "V5.expected_statements: the verifier's own derivation of every sampled unit's statement")
            vr = [r for r in _rows(costlog_v5, "verifier") if "record checks" in str(r.get("stage", ""))]
            if vr:
                det["record_checks_s"] = F(round(sum(float(r["seconds"]) for r in vr), 3), "MEASURED", cl5, "V1-V3, V5 location/coverage, V6-V8, D6")
            S["verify_live"]["detail"] = det
            S["verify_live"]["device"] = F(hw or None, "MEASURED" if hw else "UNKNOWN", cl5, "the run pod's CPUs (the verifier does not use the GPU)")
    if verify_v5:
        vsrc = src.get("verify_v5", "verify_v5.json")
        S["verify_live"]["wall_s"] = F(float(verify_v5.get("seconds")), "MEASURED", vsrc, "verify_v5 driver wall (step 5 live: record checks + proofs + expected statements)") if verify_v5.get("seconds") is not None else unknown()
        st = verify_v5.get("status")
        doc["status"]["verdict_live"] = F(st if st in VERDICTS else ("REFUSED" if st else None), "MEASURED", vsrc, f"verify_v5 status {st!r}")
        doc["status"]["utc"] = F(verify_v5.get("utc"), "MEASURED", vsrc)
        r3 = verify_v5.get("record_v3") or {}
        smp = r3.get("sampled") or {}
        g = doc["guarantee"]
        if smp.get("Q") is not None:
            g["Q"] = F(int(smp["Q"]), "MEASURED", vsrc, "record_v3.sampled.Q: |Q| drawn by r_s (ONE sample)")
        if smp.get("candidates") is not None:
            g["N_V"] = F(int(smp["candidates"]), "MEASURED", vsrc, "record_v3.sampled.candidates = N_V of the retained RUs")
        if smp.get("J") is not None:
            g["N_RU"] = F(len(smp["J"]) + len(smp.get("F") or []), "MEASURED", vsrc, f"|J| + |F| = {len(smp['J'])} + {len(smp.get('F') or [])}")
        if smp.get("input_units") is not None:
            g["input_units"] = F(int(smp["input_units"]), "MEASURED", vsrc)
        if smp.get("uncovered_selected") is not None:
            g["uncovered"] = F(int(smp["uncovered_selected"]), "MEASURED", vsrc, "record_v3.sampled.uncovered_selected")
        ep = r3.get("epoch") or {}
        if ep.get("delta"):
            g["delta"] = F(str(ep["delta"]), "MEASURED", vsrc, "record_v3.epoch.delta")
        bx = r3.get("B_ex") or {}
        if bx.get("value") is not None:
            g["B_ex"] = F(int(bx["value"]), "MEASURED", vsrc, bx.get("provenance"))
        cert = verify_v5.get("certificate") or {}
        cr = cert.get("completion_risk") or {}
        if cr.get("s"):
            g["s"] = F(str(cr["s"]), "MEASURED", vsrc, "certificate.completion_risk.s (the instance's sampling rate)")
        if cert.get("U_bits") is not None:
            g["U_bits"] = F(int(cert["U_bits"]), "MEASURED", vsrc, "certificate_v2 U_bits (shipped certified_bound on the pinned kind table)")
        if r3.get("checks_failed"):
            doc["status"]["refusal"] = F([str(c).split(":")[0] for c in r3["checks_failed"]], "MEASURED", vsrc, "record_v3.checks_failed (check ids)")
        elif st == "VERIFIED":
            doc["status"]["refusal"] = F(None, "NOT RUN", vsrc, "no refusal: VERIFIED")
    if reverify_clean:
        rsrc = src.get("reverify_clean", "reverify_clean.json")
        sec = reverify_clean.get("seconds")
        if isinstance(sec, dict):
            S["reverify"]["wall_s"] = F(float(sec.get("reverify", sec.get("total"))), "MEASURED", rsrc, f"reverify_clean seconds {sec} (venv build excluded)")
        elif sec is not None:
            S["reverify"]["wall_s"] = F(float(sec), "MEASURED", rsrc)
        st = reverify_clean.get("status")
        doc["status"]["verdict_reverify"] = F(st if st in VERDICTS else ("REFUSED" if st else None), "MEASURED", rsrc, f"reverify_clean status {st!r}, checks_failed {reverify_clean.get('checks_failed')}")
        host = reverify_clean.get("host") or {}
        env = reverify_clean.get("environment") or {}
        S["reverify"]["device"] = F(env.get("host") or env.get("hostname") or reverify_clean.get("snapshot", {}).get("host") or "run pod CPUs (fresh venv)", "MEASURED", rsrc)
        if reverify_clean.get("checks_failed"):
            doc["status"]["refusal"] = F([str(c).split(":")[0] for c in reverify_clean["checks_failed"]], "MEASURED", rsrc, "reverify_clean.checks_failed (check ids)")
    if reverify_time_v:
        t = parse_time_v(reverify_time_v)
        tsrc = src.get("reverify_time_v", "reverify time -v")
        if "max_rss_bytes" in t:
            S["reverify"]["peak_rss_bytes"] = F(int(t["max_rss_bytes"]), "MEASURED", tsrc, "/usr/bin/time -v max RSS of reverify_clean.sh (venv + reverify + SP1 host)")
        if "user_s" in t:
            S["reverify"]["cpu_s"] = F(round(t["user_s"] + t.get("sys_s", 0.0), 2), "MEASURED", tsrc, "user + sys of the whole reverify_clean.sh")
        if "fs_inputs" in t:
            S["reverify"]["detail"] = {"fs_inputs_blocks": F(int(t["fs_inputs"]), "MEASURED", tsrc, "GNU time 'File system inputs' (blocks; page cache hides most reads -- E9's instrument is the number that counts)")}
    if coverage:
        csrc = src.get("coverage", "reverify_v3/coverage.json")
        cnt = coverage.get("counts") or {}
        g = doc["guarantee"]
        if cnt.get("proof_checked_vus") is not None:
            g["proof_checked_units"] = F(int(cnt["proof_checked_vus"]), "MEASURED", csrc, "counts.proof_checked_vus (units covered by an accepted proof)")
        dc = cnt.get("directly_checked_vus")
        if isinstance(dc, dict):
            g["directly_checked_units"] = F(int(sum(dc.values())), "MEASURED", csrc, "counts.directly_checked_vus by rule: " + ", ".join(f"{k} {v}" for k, v in dc.items()))
        elif dc is not None:
            g["directly_checked_units"] = F(int(dc), "MEASURED", csrc)
        if cnt.get("uncovered_vus") is not None:
            g["uncovered"] = F(int(cnt["uncovered_vus"]), "MEASURED", csrc, "counts.uncovered_vus")
        if cnt.get("proof_objects_verified") is not None:
            g["n_proofs"] = F(int(cnt["proof_objects_verified"]), "MEASURED", csrc, "counts.proof_objects_verified")
        if cnt.get("input_vus_V6") is not None:
            g["input_units"] = F(int(cnt["input_vus_V6"]), "MEASURED", csrc)
        if cnt.get("sampled_vus_Q") is not None and g["Q"]["value"] is None:
            g["Q"] = F(int(cnt["sampled_vus_Q"]), "MEASURED", csrc)
        if cnt.get("coverage_line"):
            g["detail"] = {"coverage_line": F(cnt["coverage_line"], "MEASURED", csrc)}
    if certificate:
        csrc = src.get("certificate", "verify/certificate_v2.json")
        g = doc["guarantee"]
        law = certificate.get("law") or {}
        for k in ("q", "s", "delta"):
            if law.get(k) is not None:
                g[k] = F(str(law[k]), "MEASURED", csrc, f"certificate law.{k} ({law.get('provenance', '')})")
        b = certificate.get("bound") or {}
        if b.get("U_bits") is not None:
            g["U_bits"] = F(int(b["U_bits"]), "MEASURED", csrc, b.get("statement"))
        if b.get("W_bits") is not None:
            g["W_bits"] = F(int(b["W_bits"]), "MEASURED", csrc, "declared output width (tokens x 16 declared bits)")
        if b.get("precap_bits") is not None:
            g["U_precap_bits"] = F(int(str(b["precap_bits"])), "MEASURED", csrc)
        if b.get("U_equals_W") is not None:
            g["capped"] = F(bool(b["U_equals_W"]), "MEASURED", csrc, "U = W: the cap case")
        if b.get("calculator"):
            g["calculator"] = F(str(b["calculator"]), "MEASURED", csrc)
    doc["sources"].update({k: v for k, v in src.items()})
    return doc


def attach_e6_bound(doc: dict[str, Any], bound: dict[str, Any], source: str) -> dict[str, Any]:
    """Copy E6's per-point integrity bound (`out/scale/e6/bounds/<model>__<PxO>.json`, schema `veritor/e6/integrity-bound/v1`) into ``guarantee``:
    the law (q, s, delta, B_ex, W), U (E6's label -- MODELED when from the rule, not a certificate), j*, beta, and E6's acceptance sentence in detail.
    Fields the run itself measured (a certificate's U, the drawn |Q|) are NOT overwritten."""
    if bound.get("schema") != "veritor/e6/integrity-bound/v1":
        raise ValueError(f"not an e6 integrity-bound/v1 document: {bound.get('schema')!r}")
    g = doc["guarantee"]
    smp = bound.get("sampling") or {}
    for k in ("q", "s", "delta"):
        if smp.get(k) is not None and g[k]["value"] is None:
            g[k] = F(str(smp[k]), "MEASURED", source, f"sampling.{k} ({smp.get('s_source', '')})")
    if smp.get("B_ex") is not None and g["B_ex"]["value"] is None:
        g["B_ex"] = F(int(smp["B_ex"]), "MEASURED", source, "sampling.B_ex")
    if smp.get("W_out_bits") is not None and g["W_bits"]["value"] is None:
        g["W_bits"] = F(int(smp["W_out_bits"]), "MEASURED", source, "sampling.W_out_bits (the pinned instance's declared output width)")
    pop = bound.get("population") or {}
    if pop.get("N_V") is not None and g["N_V"]["value"] is None:
        g["N_V"] = F(int(pop["N_V"]), str(pop.get("label", "MEASURED")).split(" ")[0] if str(pop.get("label", "MEASURED")).split(" ")[0] in LABELS else "MEASURED", source, pop.get("source"))
    if pop.get("N_R") is not None and g["N_RU"]["value"] is None:
        g["N_RU"] = F(int(pop["N_R"]), "MEASURED", source, "population.N_R")
    b = bound.get("bound") or {}
    U = b.get("U") or {}
    ulab = str(U.get("label", "MODELED")).split(" ")[0]
    ulab = ulab if ulab in LABELS else "MODELED"
    if U.get("bits") is not None and g["U_bits"]["value"] is None:
        g["U_bits"] = F(int(U["bits"]), ulab, source, U.get("reading"))
        if U.get("precap_bits") is not None:
            g["U_precap_bits"] = F(int(U["precap_bits"]), ulab, source)
        if U.get("capped") is not None:
            g["capped"] = F(bool(U["capped"]), ulab, source, "U = W: the cap case")
        if U.get("source"):
            g["calculator"] = F(str(U["source"]), "MEASURED", source)
    if b.get("j_star") is not None:
        g["j_star"] = F(int(b["j_star"]), "MEASURED", source, b.get("j_star_rule"))
    if b.get("beta") is not None:
        g["beta"] = F(float(b["beta"]), "MEASURED", source, "j* / N_V: the fraction of faulty VU relations the law detects with probability >= 1 - delta (E6)")
    det = g.get("detail") or {}
    for k in ("acceptance", "escape_law", "single_fault_escape", "exempt_all_term"):
        if b.get(k) is not None:
            det[f"e6_{k}"] = F(b[k], "MEASURED", source, "E6's bounds JSON, verbatim")
    if b.get("same_beta_at_this_size"):
        det["e6_same_beta_at_this_size"] = F(b["same_beta_at_this_size"], "MODELED", source, "the s that would hold a given beta at this N_V, and its E|Q| (proving cost ∝ E|Q|)")
    if smp.get("expected_Q") is not None:
        det["expected_Q"] = F(float(smp["expected_Q"]), "MODELED", source, "E|Q| = s N_V (q = 1)")
    g["detail"] = det
    inst = bound.get("instance") or {}
    if inst.get("program_digest") and (doc["id"].get("program_revision") or {}).get("value") is None:
        doc["id"]["program_revision"] = F(str(inst["program_digest"]), "MEASURED", source, f"protocol_instance.json#program.digest of the pinned instance {inst.get('id')} (E6's bound JSON `instance.program_digest`)")
    # [D57] the geometry when E6's bound carries the pinned identity (`instance.program_identity`, the protocol_instance.json#program.identity
    # record); a bound without it leaves `id.attention` absent -- the caller resolves the pinned instance file (build_rows: attach_program_identity)
    ident = inst.get("program_identity")
    if isinstance(ident, dict) and (doc["id"].get("attention") or {}).get("value") is None:
        attach_program_identity(doc, {"identity": ident}, source)
    doc["sources"]["e6_bound"] = source
    return doc


def _empty_transfer() -> dict[str, Any]:
    return {mode: {k: unknown() for k in keys} for mode, keys in TRANSFER_MODES.items()}


def attach_verify_accounting(doc: dict[str, Any], acct: dict[str, Any], source: str, mode: str = "default") -> dict[str, Any]:
    """Copy E9's `verify_accounting.json` (schema `veritor/verify-accounting/v1`) into ``transfer``: every bucket leaf wrapped as a MEASURED field
    with ``source``, E9's ``parts`` wrapped the same way (a null leaf is omitted -- nested ARCHIVE parts carry no RSS sample), and ``per_check`` /
    ``bucket_map`` / ``cross_check`` / ``deployed_ideal`` / ``rule`` verbatim.  ``mode="default"`` is the shipped entry point as archives are checked
    (archive check ON); ``mode="deployed"`` is `reverify.py --deployed-openings` and lands under ``transfer.deployed_mode``."""
    if acct.get("schema") not in ("veritor/verify-accounting/v1",):
        raise ValueError(f"not a verify-accounting/v1 document: {acct.get('schema')!r}")
    if mode not in ("default", "deployed"):
        raise ValueError(f"mode {mode!r} not in ('default', 'deployed')")
    if mode == "deployed":
        doc["transfer"].setdefault("deployed_mode", _empty_transfer())
        tr = doc["transfer"]["deployed_mode"]
    else:
        tr = doc["transfer"]
    for bucket, keys in TRANSFER_MODES.items():
        b = (acct.get("buckets") or {}).get(bucket) or {}
        for k in keys:
            src_key = "peak_rss_mb" if k == "peak_rss_bytes" else k
            if src_key in b and b[src_key] is not None:
                v = b[src_key]
                if k == "peak_rss_bytes":
                    v = int(round(float(v) * 1e6))
                tr[bucket][k] = F(v, "MEASURED", source, f"buckets.{bucket}.{src_key}")
        if isinstance(b.get("parts"), dict):
            tr[bucket]["parts"] = {pn: {fk: F(fv, "MEASURED", source, f"buckets.{bucket}.parts.{pn}.{fk}") for fk, fv in part.items()
                                        if fv is not None and not isinstance(fv, (dict, list))}  # a null leaf (no RSS sample in nested work) is omitted
                                   for pn, part in b["parts"].items() if isinstance(part, dict)}
    for k in TRANSFER_RAW:
        if k in acct:
            tr[k] = acct[k]
    tr["source"] = source
    if acct.get("wall_s_total") is not None:
        tr["wall_s_total"] = acct["wall_s_total"]
    if acct.get("record_bytes") is not None and doc["storage"]["record_bytes"]["value"] is None:
        doc["storage"]["record_bytes"] = F(int(acct["record_bytes"]), "MEASURED", source, "verify_accounting record_bytes: sum of file sizes under the record directory the verifier was pointed at (slim copy = slim size)")
    doc["sources"][f"verify_accounting_{mode}"] = source
    return doc


def upgrade(doc: dict[str, Any]) -> dict[str, Any]:
    """Fill fields a newer schema revision added (as UNKNOWN) so rows written by earlier emitters keep validating.  Idempotent."""
    for k in ID_FIELDS:
        if k in OPTIONAL_FIELDS.get("id", ()):  # D54 §5: an optional field (program_revision) stays ABSENT -- "—" in the table, never a fabricated UNKNOWN
            continue
        doc.setdefault("id", {}).setdefault(k, unknown())
    for k in STATUS_FIELDS:
        doc.setdefault("status", {}).setdefault(k, unknown())
    for st in STAGES:
        stage = doc.setdefault("stages", {}).setdefault(st, {})
        for k in STAGE_FIELDS:
            stage.setdefault(k, unknown())
    for k in STORAGE_FIELDS:
        doc.setdefault("storage", {}).setdefault(k, unknown())
    tr = doc.setdefault("transfer", {})
    for mode, keys in TRANSFER_MODES.items():
        for k in keys:
            tr.setdefault(mode, {}).setdefault(k, unknown())
    if isinstance(tr.get("deployed_mode"), dict):
        for mode, keys in TRANSFER_MODES.items():
            for k in keys:
                tr["deployed_mode"].setdefault(mode, {}).setdefault(k, unknown())
    for k in GUARANTEE_FIELDS:
        doc.setdefault("guarantee", {}).setdefault(k, unknown())
    doc.setdefault("sources", {})
    return doc


def _time_v_fields(stage: dict[str, Any], tv: dict[str, Any], source: str, what: str) -> None:
    if tv.get("wall_s") is not None:
        stage["wall_s"] = F(float(tv["wall_s"]), "MEASURED", source, f"{what}: /usr/bin/time -v wall")
    if tv.get("user_s") is not None:
        stage["cpu_s"] = F(round(float(tv["user_s"]) + float(tv.get("sys_s") or 0.0), 2), "MEASURED", source, f"{what}: user + sys")
    if tv.get("peak_rss_kb") is not None:
        stage["peak_rss_bytes"] = F(int(tv["peak_rss_kb"]) * 1024, "MEASURED", source, f"{what}: max RSS (GNU time, KB x 1024)")


def from_e8_raw(raw: dict[str, Any], source: str, run_id: str | None = None) -> dict[str, Any]:
    """E8's `veritor/e8/costlog-raw/v1` (per-stage RAW numbers of a run_v5 run, e.g. `out/scale/e8/costlog.32x16.json`) -> a costlog/v1.
    Every value is copied with E8's own label text in the note and ``source`` as the file; nothing is invented (E8 is not asked to redo it)."""
    if not str(raw.get("schema", "")).startswith("veritor/e8/costlog-raw/v1"):
        raise ValueError(f"not an e8 costlog-raw/v1 document: {raw.get('schema')!r}")
    doc = new(raw["model"], int(raw["prompt_tokens"]), int(raw["decode_tokens"]), run_id or raw.get("tag") or "?", record_id=raw.get("record"))
    S, st, g = doc["stages"], doc["status"], doc["guarantee"]
    hw = str(raw.get("hardware") or "")
    gpu = hw.split(",")[0].strip() if hw else None
    stg = raw.get("stages") or {}
    # serve
    sv = stg.get("serve") or {}
    if sv.get("recorded_pass_wall_s") is not None:
        S["serve"]["wall_s"] = F(float(sv["recorded_pass_wall_s"]), "MEASURED", source, "stages.serve.recorded_pass_wall_s (the recorded pass; engine build excluded)")
    S["serve"]["device"] = F(gpu, "MEASURED", source) if gpu else unknown()
    if sv.get("gpu_mem_peak_mib_sampled") is not None:
        S["serve"]["gpu_peak_bytes"] = F(int(sv["gpu_mem_peak_mib_sampled"]) * 1024 * 1024, "MEASURED", source, "stages.serve.gpu_mem_peak_mib_sampled (nvidia-smi sampled)")
    S["serve"]["detail"] = {k2: F(sv[k1], "MEASURED", source, f"stages.serve.{k1}") for k1, k2 in (("run_v5_phase_wall_s", "phase_wall_s"), ("wall_s_matched_production_total", "matched_production_wall_s"), ("engine_build_s", "engine_build_s")) if sv.get(k1) is not None}
    # reconstruction (rows)
    rw = stg.get("rows") or {}
    tv = rw.get("time_v") or {}
    _time_v_fields(S["reconstruction"], tv, source, "stages.rows.time_v (E1 producer)")
    pd = rw.get("progress_done") or {}
    det = {}
    if pd.get("gates") is not None and pd.get("wall_s"):
        det["gates"] = F(int(pd["gates"]), "MEASURED", source, "progress.jsonl gates evaluated")
        det["gates_per_s"] = F(round(float(pd["gates"]) / float(pd["wall_s"]), 1), "MEASURED", source, "gates / progress wall")
    if pd.get("verdict"):
        det["boundary_compare"] = F(pd["verdict"], "MEASURED", source)
    if rw.get("label"):
        det["note"] = F(str(rw["label"]), "MEASURED", source)
    m = re.search(r"cpu5g (\d+) vCPU|(\d+) vCPU", str(rw.get("label", "")) + " " + str(tv.get("command", "")))
    S["reconstruction"]["device"] = F("RunPod CPU pod (E1 producer, --parallel 8; reused rows produced once)", "MEASURED", source, str(rw.get("label")))
    if rw.get("bytes") is not None:
        doc["storage"]["rows_bytes"] = F(int(rw["bytes"]), "MEASURED", source, "stages.rows.bytes (du -sb on the pod)")
    if det:
        S["reconstruction"]["detail"] = det
    # commitment
    cm = stg.get("commit") or {}
    cs = cm.get("class_seconds") or {}
    if cs.get("commit_io") is not None or cs.get("interior_commit") is not None:
        b, i = float(cs.get("commit_io") or 0.0), float(cs.get("interior_commit") or 0.0)
        S["commitment"]["wall_s"] = F(round(b + i, 3), "MEASURED", source, "class_seconds commit_io + interior_commit (steps 1 + 3)")
        S["commitment"]["detail"] = {"boundary_io_s": F(b, "MEASURED", source), "interior_s": F(i, "MEASURED", source), "replay_s": F(float(cs.get("replay") or 0.0), "MEASURED", source, "class 4 replay")}
        if cm.get("run_v5_phase_wall_s_protocol") is not None:
            S["commitment"]["detail"]["protocol_phase_wall_s"] = F(float(cm["run_v5_phase_wall_s_protocol"]), "MEASURED", source, "run_v5 protocol phase wall (steps 1-4 + r_q + r_s)")
    ptv = cm.get("selective_prover_time_v") or {}
    if ptv.get("peak_rss_kb") is not None:
        S["commitment"]["peak_rss_bytes"] = F(int(ptv["peak_rss_kb"]) * 1024, "MEASURED", source, "selective_prover process max RSS (phase 0 + phase 2 + interior trees)")
    if ptv.get("user_s") is not None:
        S["commitment"]["cpu_s"] = F(round(float(ptv["user_s"]) + float(ptv.get("sys_s") or 0.0), 2), "MEASURED", source, "selective_prover user + sys")
    S["commitment"]["device"] = F(hw or None, "MEASURED" if hw else "UNKNOWN", source)
    by = cm.get("bytes") or {}
    for k1, k2 in (("trees_bytes", "trees_bytes"), ("record_bytes", "record_bytes"), ("record_proofs_bytes", "proofs_bytes")):
        if by.get(k1) is not None:
            doc["storage"][k2] = F(int(by[k1]), "MEASURED", source, f"stages.commit.bytes.{k1} (du -sb on the pod)")
    if by.get("rows_bytes") is not None and doc["storage"]["rows_bytes"]["value"] is None:
        doc["storage"]["rows_bytes"] = F(int(by["rows_bytes"]), "MEASURED", source)
    # obligations
    ob = stg.get("obligations") or {}
    if ob.get("run_v5_phase_wall_s") is not None:
        S["obligations"]["wall_s"] = F(float(ob["run_v5_phase_wall_s"]), "MEASURED", source, "run_v5 obligations phase wall")
    prs = ob.get("peak_rss_sampled") or {}
    if prs.get("peak_rss_kb") is not None:
        S["obligations"]["peak_rss_bytes"] = F(int(prs["peak_rss_kb"]) * 1024, "MEASURED", source, str(prs.get("label")))
    S["obligations"]["device"] = F(hw or None, "MEASURED" if hw else "UNKNOWN", source)
    # proving
    pv = stg.get("prove") or {}
    if pv.get("run_v5_phase_wall_s") is not None:
        S["proving"]["wall_s"] = F(float(pv["run_v5_phase_wall_s"]), "MEASURED", source, "run_v5 prove phase wall (every chunk, incl. process start-up)")
    S["proving"]["device"] = F(gpu, "MEASURED", source) if gpu else unknown()
    prs = pv.get("peak_rss_sampled") or {}
    if prs.get("peak_rss_kb") is not None:
        S["proving"]["peak_rss_bytes"] = F(int(prs["peak_rss_kb"]) * 1024, "MEASURED", source, f"{prs.get('pattern')} {prs.get('label')}")
    if pv.get("gpu_mem_peak_mib_sampled") is not None:
        S["proving"]["gpu_peak_bytes"] = F(int(pv["gpu_mem_peak_mib_sampled"]) * 1024 * 1024, "MEASURED", source, "nvidia-smi sampled peak")
    S["proving"]["detail"] = {}
    if pv.get("n_proof_objects") is not None:
        S["proving"]["detail"]["n_chunks"] = F(int(pv["n_proof_objects"]), "MEASURED", source)
        g["n_proofs"] = F(int(pv["n_proof_objects"]), "MEASURED", source, "stages.prove.n_proof_objects")
        g["n_statements"] = F(int(pv["n_proof_objects"]), "MEASURED", source, "one statement per proof object")
    if pv.get("class_seconds_prove") is not None:
        S["proving"]["detail"]["class_prove_s"] = F(float(pv["class_seconds_prove"]), "MEASURED", source, "costlog_v5 class prove (proving + witness rows summed)")
    # verify_live
    vl = stg.get("verify_live") or {}
    if vl.get("verify_v5_seconds") is not None:
        S["verify_live"]["wall_s"] = F(float(vl["verify_v5_seconds"]), "MEASURED", source, "verify_v5 driver wall")
    prs = vl.get("peak_rss_sampled") or {}
    if prs.get("peak_rss_kb") is not None:
        S["verify_live"]["peak_rss_bytes"] = F(int(prs["peak_rss_kb"]) * 1024, "MEASURED", source, str(prs.get("label")))
    vcs = vl.get("class_seconds") or {}
    S["verify_live"]["detail"] = {k2: F(float(vcs[k1]), "MEASURED", source, f"class {k1}") for k1, k2 in (("verify_proofs", "proof_verification_s"), ("verifier_expected", "expected_statements_s")) if vcs.get(k1) is not None}
    if vl.get("run_v5_phase_wall_s") is not None:
        S["verify_live"]["detail"]["phase_wall_s"] = F(float(vl["run_v5_phase_wall_s"]), "MEASURED", source)
    S["verify_live"]["device"] = F(hw or None, "MEASURED" if hw else "UNKNOWN", source, "run pod CPUs")
    if vl.get("status"):
        st["verdict_live"] = F(vl["status"] if vl["status"] in VERDICTS else "REFUSED", "MEASURED", source, f"stages.verify_live.status {vl['status']!r}")
    # reverify
    rv = stg.get("reverify") or {}
    _time_v_fields(S["reverify"], rv.get("time_v") or {}, source, "reverify_clean.sh (fresh venv incl. SP1 host)")
    S["reverify"]["device"] = F("run pod CPUs, fresh venv (reverify_clean.sh)", "MEASURED", source)
    if rv.get("status"):
        st["verdict_reverify"] = F(rv["status"] if rv["status"] in VERDICTS else "REFUSED", "MEASURED", source, f"checks_failed {rv.get('checks_failed')}")
    # archive
    ar = stg.get("archive") or {}
    if ar.get("archive_bytes") is not None:
        doc["storage"]["archive_tgz_bytes"] = F(int(ar["archive_bytes"]), "MEASURED", source, "stat on the pod")
    if ar.get("slim_bytes") is not None:
        doc["storage"]["slim_bytes"] = F(int(ar["slim_bytes"]), "MEASURED", source, "stat on the pod")
    # sampling / guarantee
    smp = raw.get("sampling") or {}
    for k1, k2 in (("N_V", "N_V"), ("Q_selected_vus", "Q"), ("proof_checked_units", "proof_checked_units"), ("directly_checked_units", "directly_checked_units"),
                   ("input_units_V6", "input_units"), ("uncovered_units", "uncovered"), ("B_ex", "B_ex")):
        if smp.get(k1) is not None:
            g[k2] = F(int(smp[k1]), "MEASURED", source, f"sampling.{k1} ({smp.get('label', '')})")
    for k in ("s", "q", "delta"):
        if smp.get(k) is not None:
            g[k] = F(str(smp[k]), "MEASURED", source, f"sampling.{k} ({smp.get('s_source', '')})")
    g["N_RU"] = F(int(raw.get("requests") or 1), "MEASURED", source, "requests (one RU per request)")
    if smp.get("coverage_line"):
        g["detail"] = {"coverage_line": F(smp["coverage_line"], "MEASURED", source)}
    st["utc"] = F(raw.get("utc"), "MEASURED", source) if raw.get("utc") else unknown()
    st["record_path"] = F(raw.get("record_path_pod"), "MEASURED", source, "record_path_pod (the run pod; archive on the volume)") if raw.get("record_path_pod") else unknown()
    if smp.get("instance_id"):
        doc["id"]["detail"] = {"instance_id": F(smp["instance_id"], "MEASURED", source)}
    doc["sources"]["e8_raw"] = source
    return doc


def attach_t1_verify(doc: dict[str, Any], t1: dict[str, Any], source: str) -> dict[str, Any]:
    """T1's independent re-verification (`out/scale/t1/verify.<record>.json`): the verdict under ``status.verdict_independent`` and T1's measured
    re-verify wall / RSS / bytes under ``stages.reverify.detail.t1_*`` (the row's ``reverify`` stage stays the run pod's own shipped re-verification)."""
    v = t1.get("verdict") or {}
    stt = v.get("status") or v.get("record_v3_status")
    tree = (t1.get("tree") or {}).get("commit") if isinstance(t1.get("tree"), dict) else t1.get("tree")
    mach = (t1.get("machine") or {}).get("pod") if isinstance(t1.get("machine"), dict) else t1.get("machine")
    doc["status"]["verdict_independent"] = F(stt if stt in VERDICTS else ("REFUSED" if stt else None), "MEASURED" if stt else "UNKNOWN", source,
                                             f"T1 shipped reverify.py @ {str(tree)[:8]} on {mach}: {v.get('pass')}/{v.get('fail')}/{v.get('not_applicable')} (pass/fail/n-a); failed {v.get('failed')}")
    det = doc["stages"]["reverify"].get("detail") or {}
    cf = (t1.get("costlog_v1_fields") or {}).get("stages.reverify") or {}
    for k in ("wall_s", "cpu_s", "peak_rss_bytes", "bytes_read"):
        f = cf.get(k)
        if isinstance(f, dict) and f.get("value") is not None:
            det[f"t1_{k}"] = F(f["value"], "MEASURED", source, f.get("note"))
    doc["stages"]["reverify"]["detail"] = det
    doc["sources"]["t1_verify"] = source
    return doc


def finish(doc: dict[str, Any]) -> dict[str, Any]:
    """Derive ``status.stage_reached`` from the stages that carry a wall and ``guarantee.bound_statement`` from the guarantee (idempotent)."""
    reached = "none"
    for st in STAGES:
        if doc["stages"][st]["wall_s"]["value"] is not None:
            reached = st
    if doc["status"]["stage_reached"]["value"] is None:
        doc["status"]["stage_reached"] = F(reached, "MEASURED", "the last stage with a measured wall")
    for k in ("verdict_live", "verdict_reverify", "verdict_reverify_repaired", "verdict_independent"):
        doc["status"].setdefault(k, unknown())
    g = doc["guarantee"]
    if g["epsilon_crypto"]["value"] is None:
        g["epsilon_crypto"] = F("named (R1 §1.5, root-only): eps_CR(B1) + sqrt(n_RO eps_CR(B2)) + P_n eps_KS + eps_PRF + 2^-189; no number attached", "MEASURED", "docs/theory/statements-2026-09-11.md §1.5, §5")
    bs = bound_statement(g)
    if bs is not None:
        g["bound_statement"] = F(bs, "MODELED" if g["U_bits"]["label"] == "MODELED" else "MEASURED", "costlog.bound_statement over this guarantee block (R1 §1.1/§1.5/§5 wording)")
    return doc


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --- CLI ---------------------------------------------------------------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m e2e.scale.costlog", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate", help="validate costlog files; exit 1 on the first malformed one")
    v.add_argument("files", nargs="+")
    fr = sub.add_parser("from-record", help="build a costlog from a record_v5 run's artefacts (fills what it can; the rest stays UNKNOWN)")
    fr.add_argument("--model", required=True)
    fr.add_argument("--prompt-len", type=int, required=True)
    fr.add_argument("--output-len", type=int, required=True)
    fr.add_argument("--run-id", required=True)
    fr.add_argument("--record-id")
    fr.add_argument("--architecture")
    fr.add_argument("--costlog-v5")
    fr.add_argument("--verify-v5")
    fr.add_argument("--reverify-clean")
    fr.add_argument("--coverage")
    fr.add_argument("--certificate", help="verify/certificate_v2.json (law q/s/delta, bound U/W)")
    fr.add_argument("--reverify-time-v", help="a /usr/bin/time -v capture of reverify_clean.sh")
    fr.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "validate":
        bad = 0
        for f in a.files:
            errs = validate(load(f))
            print(f"{f}: {'ok' if not errs else str(len(errs)) + ' error(s)'}")
            for e in errs:
                print("  -", e)
            bad += bool(errs)
        return 1 if bad else 0
    def rd(p):
        return json.load(open(p)) if p else None
    doc = from_record_v5(model=a.model, prompt_len=a.prompt_len, output_len=a.output_len, run_id=a.run_id, record_id=a.record_id, architecture=a.architecture,
                         costlog_v5=rd(a.costlog_v5), verify_v5=rd(a.verify_v5), reverify_clean=rd(a.reverify_clean), coverage=rd(a.coverage),
                         certificate=rd(a.certificate), reverify_time_v=(open(a.reverify_time_v).read() if a.reverify_time_v else None),
                         sources={k: getattr(a, k) for k in ("costlog_v5", "verify_v5", "reverify_clean", "coverage", "certificate", "reverify_time_v") if getattr(a, k)})
    finish(doc)
    dump(doc, a.out)
    print(f"wrote {a.out} ({len(validate(doc))} errors)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
