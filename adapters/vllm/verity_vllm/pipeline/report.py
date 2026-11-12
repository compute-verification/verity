"""verity_vllm/pipeline/report.py — the configuration card (`verity-gen/card/v1`): one JSON per configuration, torch-free.

A card summarises one run directory (`bench/run_config.py`) for one configuration `config_id =
<model-slug>__<engine-args-slug>__<kappa-short>`: the model and its configuration record `C` with per-field provenance,
the engine arguments, the workload, κ (versions.json incl. host), the seven gate results and the verdict, what the run
used (census kernels, patterns, roots, instances per step) and what it cost.  `out/gen/cards/SCHEMA.md` documents every
field.  `summarize_md(card)` renders the fixed-width top section (CASE / provenance / counts / replay / HOLDOUTS /
SUPPORT CLAIM / VERDICT) followed by a short markdown block.

The five provenance rows come from `provenance.json`, which no stage in this tree writes; without it the card is
`unsealed: true` and those rows print `n/a`.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import random
import re
import statistics
import time
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from verity_vllm.check import gates as G
from verity_vllm.check.gates import GateResult, RunArtifacts
from verity_vllm.config import RowEngine
from verity_vllm.pipeline.row_json import field_of, listed_of
from verity_vllm.records.schema_ids import CONFIGURATION_CARD_SCHEMA

CARD_SCHEMA = CONFIGURATION_CARD_SCHEMA

#: `C` fields the profile derives from others (never read from the HF config).
DERIVED_FIELDS = ("QKV", "GU", "BN")
#: `C` fields of a profile configuration, in card order.
C_FIELDS = ("model", "revision", "H", "NL", "NH", "KVH", "D", "I", "V", "QKV", "GU", "EPS", "ROPE_THETA", "MAX_POS", "BIAS",
            "tie_word_embeddings", "BN")
#: Engine arguments the card records (missing ones are `null`).
ENGINE_ARG_KEYS = ("max_num_seqs", "enable_prefix_caching", "enable_chunked_prefill", "dtype", "quantization", "tensor_parallel_size",
                   "enforce_eager", "seed", "max_model_len", "gpu_memory_utilization", "attention_backend", "kv_cache_dtype", "block_size",
                   "vllm_commit")


# ---------------------------------------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------------------------------------


def slug(s: Any) -> str:
    """Lower-case, `/` -> `--`, anything outside `[a-z0-9._-]` -> `-`."""
    s = str(s).strip().lower().replace("/", "--")
    s = re.sub(r"[^a-z0-9._-]+", "-", s)
    return s.strip("-") or "unknown"


_DTYPE_SHORT = {"bfloat16": "bf16", "torch.bfloat16": "bf16", "float16": "fp16", "half": "fp16", "float32": "fp32", "auto": "auto"}


def dtype_short(d: Any) -> str:
    return _DTYPE_SHORT.get(str(d), slug(d)) if d is not None else "dtype-unknown"


def engine_args_slug(engine_args: dict) -> str:
    """`bf16-eager-tp1-seqs8-pc0-cp1[-<quant>]`: dtype, eager/graph, tensor parallel, max_num_seqs, prefix caching, chunked
    prefill (`x` when unknown), quantization when set."""
    ea = engine_args or {}
    parts = [dtype_short(ea.get("dtype")), "eager" if ea.get("enforce_eager", True) else "graph",
             f"tp{int(ea.get('tensor_parallel_size') or 1)}", f"seqs{ea.get('max_num_seqs') if ea.get('max_num_seqs') is not None else 'x'}",
             f"pc{_flag(ea.get('enable_prefix_caching'))}", f"cp{_flag(ea.get('enable_chunked_prefill'))}"]
    if ea.get("quantization"):
        parts.append(slug(ea["quantization"]))
    return "-".join(parts)


def _flag(v: Any) -> str:
    return "x" if v is None else ("1" if v else "0")


def kappa_short(versions: dict) -> str:
    """First 10 hex digits of sha256 over the canonical κ pins {torch, cuda, triton, vllm, gpu.name, gpu.cc}."""
    v = versions or {}
    gpu = v.get("gpu")   # a GPU's {name, cc}, or its bare name; versions that name no GPU pin neither
    name, cc = (gpu.get("name"), gpu.get("cc")) if isinstance(gpu, dict) else (gpu if gpu else None, None)
    pins = {"torch": v.get("torch"), "cuda": v.get("cuda"), "triton": v.get("triton"), "vllm": v.get("vllm"), "gpu": name, "cc": cc}
    return hashlib.sha256(json.dumps(pins, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:10]


def config_id_of(model_repo: str, engine_args: dict, versions: dict) -> str:
    return f"{slug(model_repo)}__{engine_args_slug(engine_args)}__{kappa_short(versions)}"


# ---------------------------------------------------------------------------------------------------------
# the card
# ---------------------------------------------------------------------------------------------------------


@dataclass
class ReportCard:
    schema: str
    utc: str
    config_id: str
    model: dict                      # {repo, revision, family, sha256?, hf_model_type?, architectures?}
    C: dict                          # {field: {"value", "provenance": "hf-config"|"hand"|"derived"}}
    engine_args: dict                # ENGINE_ARG_KEYS + sampling + env_flags + source
    workload: dict                   # {path, sha256, n_requests, prompt_lens, max_tokens}
    kappa: dict                      # versions.json (torch, cuda, triton, vllm, python, gpu, host, export)
    gates: dict                      # {G1..G7: GateResult json}
    verdict: str
    census_kernels: list = field(default_factory=list)
    patterns_used: dict = field(default_factory=dict)
    patterns_missing: list = field(default_factory=list)
    roots_by_class: dict = field(default_factory=dict)
    per_step_instances: list = field(default_factory=list)
    mechanism_notes: list = field(default_factory=list)
    spend: dict = field(default_factory=dict)
    artifacts: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    provenance: dict = field(default_factory=dict)      # provenance.json of the sealed verifier (or {} when unsealed)
    unsealed: bool = True
    diagnostics: dict = field(default_factory=dict)     # gates.diagnostics(): structure, family_match, overhead
    counts: dict = field(default_factory=dict)          # the numbers of the top section
    replay: dict = field(default_factory=dict)          # tier a / tier b / kv linkage summary
    stages: dict = field(default_factory=dict)          # {stage: {status, seconds, returncode}}
    case: str | None = None
    profile: str | None = None
    run_dir: str | None = None

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> ReportCard:
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


def load_card(path: str | Path) -> ReportCard:
    return ReportCard.from_json(json.loads(Path(path).read_text()))


def write_card(card: ReportCard, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(card.to_json(), indent=1, default=str) + "\n")
    return p


# ---------------------------------------------------------------------------------------------------------
# building a card from a run directory
# ---------------------------------------------------------------------------------------------------------


def c_with_provenance(C: dict, hf_C: dict | None) -> dict:
    """`{field: {"value", "provenance"}}`: `derived` for QKV/GU/BN, `hf-config` when `hf_C` (from `analytic.config_of`)
    carries the same value, else `hand`."""
    out = {}
    keys = [k for k in C_FIELDS if k in C] + [k for k in C if k not in C_FIELDS and k != "profile"]
    for k in keys:
        v = C[k]
        if k in DERIVED_FIELDS:
            prov = "derived"
        elif hf_C is not None and k in hf_C and hf_C[k] == v:
            prov = "hf-config"
        else:
            prov = "hand"
        out[k] = {"value": v, "provenance": prov}
    return out


def log_header(path: str | Path) -> dict | None:
    """The first record of a raw observation log (the RunHeader) as JSON, or None."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        with gzip.open(p, "rt", encoding="utf-8") if p.suffix == ".gz" else open(p, encoding="utf-8") as f:
            d = json.loads(f.readline())
    except (OSError, EOFError, ValueError, zlib.error):    # unreadable, cut short, or a first line that is not JSON
        return None
    return d if isinstance(d, dict) and d.get("$e") == "RunHeader" else None


def _detail(gates: dict, gid: str) -> dict:
    """What gate ``gid`` recorded beside its status; the empty block for a gate that did not run or recorded nothing."""
    d = field_of(gates.get(gid), "detail")
    return d if d else {}


def workload_summary(path: str | Path | None) -> dict:
    """{path, sha256, n_requests, prompt_lens, max_tokens, requests: [{request_id, prompt_len, max_tokens}]} of a workload
    file (`sha256` as `vllm_adapter.load_workload` computes it: canonical JSON of the file)."""
    if not path:
        return {"path": None, "sha256": None, "n_requests": 0, "prompt_lens": [], "max_tokens": [], "requests": []}
    p = Path(path)
    if not p.exists():
        return {"path": str(path), "sha256": None, "n_requests": 0, "prompt_lens": [], "max_tokens": [], "requests": [], "error": "missing"}
    doc = json.loads(p.read_text())
    sha = hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    reqs = [{"request_id": r.get("request_id"), "prompt_len": len(listed_of(r, "prompt_token_ids")), "max_tokens": r.get("max_tokens"),
             "arrive_step": int(r.get("arrive_step") or 0)}
            for r in doc.get("requests", [])]
    return {"path": str(path), "sha256": sha, "n_requests": len(reqs), "prompt_lens": sorted({r["prompt_len"] for r in reqs}),
            "max_tokens": sorted({r["max_tokens"] for r in reqs if r["max_tokens"] is not None}), "requests": reqs,
            "model": doc.get("model"), "case": doc.get("case"), "sampling": field_of(doc.get("variants"), "greedy"),
            # [coverage D87] serving-regime dimensions of the coverage campaign (None for the older hand-authored workloads)
            "serving_profile": doc.get("serving_profile"), "context_class": doc.get("context_class"),
            "length_seed": doc.get("length_seed"), "arrive_steps": [r["arrive_step"] for r in reqs],
            "expected_engine_steps": doc.get("expected_engine_steps"), "engine_args_required": dict(RowEngine.of(doc).args)}


def engine_args_of(run: RunArtifacts) -> dict:
    """Engine arguments from the census (`engine_kwargs`), the capture's log header (`profile.cache`, `.backends`,
    `.parallelism`, `.dtype`, `.quantization`) or the manifest's requested values, in that order of authority."""
    out: dict[str, Any] = {k: None for k in ENGINE_ARG_KEYS}
    src = []
    hdr = log_header(run.path("log")) if run.exists("log") else None
    if hdr:
        prof = hdr.get("profile")
        if not prof:
            prof = {}   # a header without a profile states no argument: each stays None
        cache, par, comp = prof.get("cache"), prof.get("parallelism"), prof.get("compilation")   # a block the profile leaves out states none of its arguments
        out.update({**{k: field_of(cache, k) for k in ("max_num_seqs", "enable_prefix_caching", "enable_chunked_prefill", "kv_cache_dtype", "block_size")},
                    "dtype": prof.get("dtype"), "quantization": prof.get("quantization"),
                    "tensor_parallel_size": field_of(par, "tensor_parallel_size"), "enforce_eager": field_of(comp, "enforce_eager"),
                    "attention_backend": field_of(prof.get("backends"), "attention")})
        kw = prof.get("engine_kwargs")
        for k in ("seed", "max_model_len", "gpu_memory_utilization"):
            out[k] = field_of(kw, k)
        out["sampling"] = prof.get("sampling") or field_of(hdr.get("workload"), "sampling")
        out["env_flags"] = prof.get("engine_flags")
        out["vllm_commit"] = field_of(prof.get("vllm"), "commit")
        src.append("capture log header")
    census = run.json("census")
    if isinstance(census, dict):
        kw = census.get("engine_kwargs")   # what the census's engine was made with; a census that records none overrides nothing
        if kw:
            for k in ("dtype", "seed", "max_model_len", "gpu_memory_utilization", "enforce_eager", "enable_prefix_caching", "max_num_seqs",
                      "tensor_parallel_size", "quantization", "enable_chunked_prefill"):
                if k in kw:
                    out[k] = kw[k]
        out.setdefault("sampling", None)
        if out.get("sampling") is None:
            out["sampling"] = field_of(census.get("workload"), "sampling")
        if not out.get("env_flags"):
            out["env_flags"] = census.get("env_flags")
        src.append("census.json engine_kwargs")
    req = run.manifest.get("engine_args_requested")   # what the driver asked for: it fills what neither the header nor the census states
    if req:
        for k, v in req.items():
            if out.get(k) is None:
                out[k] = v
        src.append("run.json engine_args_requested")
    if out.get("tensor_parallel_size") is None:
        out["tensor_parallel_size"] = 1
    if out.get("enforce_eager") is None and hdr is None and not census:
        out["enforce_eager"] = None
    out["source"] = src
    return out


def kappa_of(run: RunArtifacts, hdr: dict | None) -> dict:
    """versions.json (incl. host) when the capture wrote one, else the census's, else the pins of the log header."""
    v = run.json("versions")
    if isinstance(v, dict) and v:
        return v
    census = run.json("census")
    if isinstance(census, dict) and isinstance(census.get("versions"), dict):
        return dict(census["versions"], source="census.json")
    if not hdr:
        return {}
    prof = hdr.get("profile")   # the capture's own pins; a header that leaves out the profile, or a block of it, pins None for each
    py, gpu = field_of(prof, "python"), field_of(prof, "gpu")
    return {"torch": field_of(py, "torch"), "cuda": field_of(py, "cuda"), "triton": field_of(py, "triton"),
            "vllm": field_of(field_of(prof, "vllm"), "version"), "python": field_of(py, "version"),
            "gpu": {"name": field_of(gpu, "name"), "cc": field_of(gpu, "compute_capability"), "driver": field_of(gpu, "driver")},
            "host": None, "source": "log header (no versions.json)"}


def _model_of(run: RunArtifacts, C: dict | None, hdr: dict | None) -> dict:
    m: dict[str, Any] = {"repo": (C or {}).get("model"), "revision": (C or {}).get("revision"),
                         "family": run.manifest.get("family") or "Serve_v4", "sha256": None}
    if hdr:
        hm = hdr.get("model")   # the model the capture loaded; a header that names none adds nothing to what the record states
        m["repo"] = m["repo"] or field_of(hm, "repo")
        m["revision"] = m["revision"] or field_of(hm, "revision")
        cfg = field_of(hm, "config")
        m["hf_model_type"] = field_of(cfg, "model_type")
        m["architectures"] = field_of(cfg, "architectures")
        pm = field_of(hdr.get("profile"), "model")   # the digests of the files it loaded; a profile without them leaves both None
        ws = field_of(pm, "weights_sha256")
        if isinstance(ws, dict) and ws:
            m["sha256"] = ws if len(ws) > 1 else next(iter(ws.values()))
        m["config_sha256"] = field_of(pm, "config_sha256")
    if run.manifest.get("model_sha256"):
        m["sha256"] = run.manifest["model_sha256"]
    return m


def _stages_of(run: RunArtifacts) -> dict:
    st = run.json("stages")
    recs = st.get("stages", st) if isinstance(st, dict) else (st or [])
    if isinstance(recs, dict):
        recs = list(recs.values())
    return {r["stage"]: {k: r.get(k) for k in ("status", "seconds", "returncode", "log", "reason")} for r in recs if isinstance(r, dict) and "stage" in r}


def _artifacts_of(run: RunArtifacts) -> dict:
    out = {}
    for key in list(G.ARTIFACTS) + [G.replay_b_key(s) for s in run.snapshot_steps()]:
        if key in ("run", "card") or not run.exists(key):
            continue
        out[key] = run.rel(key)
    return dict(sorted(out.items()))


def _gate_status(gates: dict, gid: str) -> str | None:
    g = gates.get(gid)
    return g.get("status") if isinstance(g, dict) else None


def vu_partition_text(gates: dict) -> str:
    """The "VU partition" row: G8's status with the set count and the largest w_out seen (`n/a` for a record that predates G8)."""
    g8 = gates.get("G8")
    if not isinstance(g8, dict):
        return "n/a (record predates G8)"
    st = g8.get("status")
    d = g8.get("detail")   # the partition's status, its set count and its widths; a gate that recorded none prints n/a for each
    if st == "skipped":
        return f"skipped ({g8.get('reason') or 'no query-use document'})"
    if st == "error":
        return f"error ({str(g8.get('reason') or '')[:80]})"
    part, width = field_of(d, "partition"), field_of(d, "width")
    count = field_of(d, "count")
    max_seen, w_max = field_of(width, "max_seen"), field_of(width, "w_max")
    txt = (f"{field_of(part, 'status')}, {count if count is not None else 'n/a'} sets, max w_out {max_seen if max_seen is not None else 'n/a'}"
           f"/{w_max if w_max is not None else 'n/a'}")
    if st == "unsupported":                        # undecided: neither PASS nor FAIL; the reason names what was undecidable
        return f"unsupported ({str(g8.get('reason') or '').removeprefix('unsupported (').removesuffix(')')[:120]}); {txt}"
    if st == "fail":
        codes = field_of(part, "codes")   # the partition's failure codes; a detail that lists none names none
        extra = [f"codes {list(codes)[:4]}"] if codes else []
        violators, undecided = field_of(width, "n_violators"), field_of(width, "n_unsupported")
        if violators:
            extra.append(f"{violators} violators")
        if undecided:
            extra.append(f"{undecided} unsupported")
        vs = field_of(d, "status") or field_of(d, "verdict_status")
        if vs not in (None, "ok"):
            extra.append(f"status {vs}")
        if any(str(p).startswith("binding:") for p in str(g8.get("reason") or "").split("; ")):
            extra.append("binding mismatch")
        return txt + (f"; {', '.join(extra)}" if extra else "") + " FAIL"
    return txt + " PASS"


def counts_of(run: RunArtifacts, gates: dict, provenance: dict | None) -> dict:
    """The numbers the top section prints (each with its PASS/FAIL where the shape asks for one)."""
    res = run.json("resolution") or {}
    fs = run.json("fold_summary") or {}
    cr = run.json("census_reconcile") or {}
    rp = run.json("root_policy") or {}
    q = run.json("quarantine")
    ra = run.json("replay_a") or {}
    g2, g5 = _detail(gates, "G2"), _detail(gates, "G5")
    unsupported = field_of(res.get("totals"), "unsupported")
    ambiguous = g2.get("ambiguous", field_of(cr.get("totals"), "ambiguous"))
    unknown_roots = fs.get("unknown_roots") or field_of(res.get("roots"), "unknown")   # the fold's, else the resolution's; a run that names none has none
    unknown = len(unknown_roots) if unknown_roots else 0
    classes = rp.get("classes")   # the root policy's classes, then its roots; a policy that lists neither leaves none unvalidated
    unvalidated = sum(int(c.get("failed") or 0) for c in classes.values() if isinstance(c, dict)) if classes else 0
    unvalidated += sum(1 for r in listed_of(rp, "roots") if isinstance(r, dict) and r.get("status") not in ("pass", "validated", "ok"))
    residue = field_of(cr.get("totals"), "residue_kernels")
    prov = provenance or {}
    cov = ra.get("coverage")   # tier a's specs, each with what covered it; a run without tier a counts the resolution's specs and compares none
    if cov:
        n_specs = len(cov)
        n_covered = len(cov) - len(listed_of(ra, "uncovered_specs"))
    else:
        n_specs, n_covered = len(listed_of(res, "by_spec")), None
    diff = field_of(q, "differential")   # the quarantine's differential tests; a run without them prints n/a
    # G5 (D82): exhaustive operand provenance, sampled VUs (verifier seed), served tokens, chained slices: a part G5 did not reach is skipped
    pv = g5.get("provenance")
    if pv and pv.get("reads") is not None:
        wiring_ok = (pv.get("n_mismatch") == 0 and pv.get("uncovered") == 0 and pv.get("single_stream") is not False
                     and not (pv.get("writes_outside_schema") or 0))
        wiring_txt = f"{pv['reads']} reads, {pv.get('n_mismatch', '?')} mismatches" + (" PASS" if wiring_ok else " FAIL")
    else:
        wiring_txt = "skipped"
    smp = g5.get("sample")
    sample_txt = (f"k={smp.get('k')} x {smp.get('strata')} strata, seed {smp.get('seed')} " + ("PASS" if smp.get("status") == "pass" else "FAIL")
                  if smp and smp.get("status") else "skipped")
    tk = g5.get("tokens")
    if tk and tk.get("evaluated"):
        tokens_txt = f"{tk.get('equal')}/{tk.get('evaluated')} rows at steps {tk.get('steps')}" + (" PASS" if tk.get("all_rows_equal") else " FAIL")
    else:
        tokens_txt = "skipped"
    chn = g5.get("chain")
    chain_txt = (f"{chn.get('n_slices')} slices, {chn.get('n_edges_chained')} chained edges " + ("PASS" if chn.get("status") == "pass" else "FAIL")
                 if chn and chn.get("status") else "skipped")
    ni_status = _gate_status(gates, "G1")
    # D85: what the sampled population covers (vu_query; informational, never gates)
    vp = ra.get("vu_population")   # what the sampled population covers; a run whose tier a did not compute it says so
    if vp and vp.get("status") == "ok":
        n_zero, n_partial = len(listed_of(vp, "zero_mass_specs")), len(listed_of(vp, "partial_tiling_specs"))
        vu_txt = (f"{vp.get('count')} VUs / {vp.get('instances')} instances, {vp.get('oversize_instances')} oversize; "
                  f"{vp.get('instances_without_units')} zero-mass instances ({n_zero} specs), {n_partial} partially tiled specs"
                  + ("" if not (n_zero or n_partial) else " -- SEE replay_a.vu_population"))
    elif vp and vp.get("status") == "error":
        vu_txt = f"unavailable ({vp.get('error')})"
    else:
        vu_txt = "not computed"
    return {
        "vu_coverage": vu_txt, "vu_zero_mass_specs": field_of(vp, "zero_mass_specs"), "vu_partial_tiling_specs": field_of(vp, "partial_tiling_specs"),
        # G8: the VU-partition structural gate over query_use.json -- skipped / pass / fail + set count + max w_out
        "vu_partition": vu_partition_text(gates),
        # each provenance row counts what the sealed verifier's provenance lists; an unsealed card has none, and prints n/a
        "protected_files_modified": len(listed_of(prov, "protected_files_modified")) if prov else None,
        "existing_primitives_modified": len(listed_of(prov, "existing_primitives_modified")) if prov else None,
        "new_root_classes": len(listed_of(prov, "new_root_classes")) if prov else (len(listed_of(rp, "new_root_classes")) if rp else None),
        "new_allowlist_entries": len(listed_of(prov, "new_allowlist_entries")) if prov else None,
        "golden_digest_changes": len(listed_of(prov, "golden_digest_changes")) if prov else None,
        "unsupported": unsupported, "ambiguous": ambiguous, "unknown_or_unvalidated_roots": unknown + unvalidated if (fs or res or rp) else None,
        "census_residue": residue,
        "new_patterns": len(listed_of(prov, "new_patterns")) if prov else None,
        "new_definitions": len(listed_of(prov, "new_definitions")) if prov else None,
        "differential_passed": field_of(diff, "passed"), "differential_total": field_of(diff, "total"),
        "specs_captured": n_specs, "specs_compared": n_covered,
        "operand_provenance": wiring_txt, "sampled_vus": sample_txt, "served_tokens": tokens_txt, "chained_slices": chain_txt,
        "control_noninterference": "PASS" if ni_status == "pass" else ("FAIL" if ni_status == "fail" else "n/a"),
    }


def build_card(run_dir: str | Path, results: list[GateResult] | None = None, *, config_id: str | None = None,
               mechanism_notes: list[str] | None = None, notes: list[str] | None = None, hf_config: dict | None = None) -> ReportCard:
    """Assemble the card of a run directory.  `results` default to `gates.json` when present, else the gates are evaluated
    here.  `hf_config` is `analytic.config_of(<hf config.json>)` when the driver could obtain it (per-field provenance)."""
    run = RunArtifacts(run_dir)
    man = run.manifest
    if results is None:
        gj = run.json("gates")
        results = [GateResult.from_json(v) for v in (gj or {}).get("gates", {}).values()] if gj else G.evaluate_gates(run_dir)
    gates = {r.id: r.to_json() for r in results}
    C = run.C or {}
    hdr = log_header(run.path("log")) if run.exists("log") else None
    versions = kappa_of(run, hdr)
    engine_args = engine_args_of(run)
    model = _model_of(run, C, hdr)
    wl = man.get("workload") or workload_summary(man.get("workload_path"))
    cid = config_id or man.get("config_id") or config_id_of(model.get("repo") or "model", engine_args, versions)
    res = run.json("resolution") or {}
    cr = run.json("census_reconcile") or {}
    fs = run.json("fold_summary") or {}
    prov = run.json("provenance")
    unsealed = not isinstance(prov, dict)
    kernels = [k.get("kernel") for k in listed_of(cr, "kernels") if isinstance(k, dict)]
    if not kernels and isinstance(run.json("census"), dict):
        histogram = run.json("census").get("run_histogram")   # a run without the reconcile's kernels names the census's own; a census without them names none
        kernels = list(histogram.keys()) if histogram else []
    missing = []
    seen = set()
    for u in listed_of(res, "unsupported_examples"):
        key = (u.get("op"), u.get("reason"))
        if key not in seen:
            seen.add(key)
            missing.append({"op": u.get("op"), "reason": u.get("reason"), "step": u.get("step")})
    primary = res.get("primary_report")   # the resolution's report by pattern; one that resolved nothing uses no pattern and misses no op
    unsupported_ops = field_of(primary, "unsupported_ops")
    if unsupported_ops:
        for op, n in unsupported_ops.items():
            if not any(m["op"] == op for m in missing):
                missing.append({"op": op, "count": n})
    by_pattern = field_of(primary, "by_pattern")
    by_class, unknown = field_of(res.get("roots"), "classes"), field_of(res.get("roots"), "unknown")   # a resolution that names no root has neither
    roots = dict(by_class) if by_class else {}
    if unknown:
        roots["unknown_names"] = unknown
    spend = run.json("spend") or {}
    hf_C = hf_config if hf_config is not None else man.get("hf_config")
    g4, g5 = _detail(gates, "G4"), _detail(gates, "G5")
    card = ReportCard(
        schema=CARD_SCHEMA, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), config_id=cid, model=model,
        C=c_with_provenance(C, hf_C), engine_args=engine_args,
        workload={**{k: wl.get(k) for k in ("path", "sha256", "n_requests", "prompt_lens", "max_tokens")},
                  # [coverage D87] serving regime + the realized shape (run.json["realized"], written by the driver after the GPU phase)
                  **{k: wl.get(k) for k in ("serving_profile", "context_class", "arrive_steps", "expected_engine_steps") if wl.get(k) is not None},
                  **({"realized": man["realized"]} if isinstance(man.get("realized"), dict) else {})},
        kappa=versions, gates=gates, verdict=G.verdict(results), census_kernels=kernels,
        patterns_used=dict(by_pattern) if by_pattern else {}, patterns_missing=missing, roots_by_class=roots,
        per_step_instances=[{"step": s.get("step"), "phase": s.get("phase"), "n_instances": s.get("n_instances")} for s in listed_of(fs, "per_step")],
        mechanism_notes=list(mechanism_notes if mechanism_notes is not None else listed_of(man, "mechanism_notes")),
        spend={"pod_hours": spend.get("pod_hours"), "usd": spend.get("usd"), **{k: v for k, v in spend.items() if k not in ("pod_hours", "usd")}},
        artifacts=_artifacts_of(run), notes=list(notes if notes is not None else listed_of(man, "notes")),
        provenance=dict(prov) if isinstance(prov, dict) else {}, unsealed=unsealed, diagnostics=G.diagnostics(run),
        counts=counts_of(run, gates, prov if isinstance(prov, dict) else None),
        replay={"tier_a": {k: g4.get(k) for k in ("status", "k", "specs", "covered", "uncovered_specs", "pass", "fail", "skipped")},
                "provenance": g5.get("provenance"), "sample": g5.get("sample"), "tokens": g5.get("tokens"), "chain": g5.get("chain"),
                "full_replay": g5.get("full_replay"),
                "quarantine": g4.get("quarantine")},
        stages=_stages_of(run), case=man.get("case"), profile=man.get("profile"), run_dir=str(Path(run_dir)),
    )
    return card


# ---------------------------------------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------------------------------------

_W = 34   # label column width of the top section


def _row(label: str, n: Any, status: str | None = None) -> str:
    s = f"{label:<{_W}}{str(n if n is not None else 'n/a'):<6}"
    return (s + status) if status else s.rstrip()


def _pf(n: Any, zero_passes: bool = True) -> str:
    if n is None:
        return "n/a"
    return "PASS" if (n == 0) == zero_passes else "FAIL"


def _attention_short(backend: Any) -> str:
    if not backend:
        return "attention-unknown"
    b = str(backend).rsplit(".", 1)[-1]
    return {"FlashAttentionImpl": "FA2"}.get(b, b)


def _family_short(card: ReportCard) -> str:
    m = card.model or {}
    return str(m.get("hf_model_type") or m.get("family") or "unknown")


def top_section(card: ReportCard) -> str:
    """The fixed-width top section (one line per row, labels padded to 34 columns)."""
    ea, k, c = card.engine_args or {}, card.kappa or {}, card.counts or {}
    prov = card.provenance or {}
    gpu = k.get("gpu")   # a GPU's {name, cc}, or its bare name; a card whose κ states no cc claims sm_unknown
    cc = str(gpu.get("cc") or "").replace(".", "") if isinstance(gpu, dict) else ""
    sm = f"sm_{cc}" if cc else "sm_unknown"
    vllm_commit = ea.get("vllm_commit") or prov.get("vllm_commit")
    if not vllm_commit:
        v = str(k.get("vllm") or "")
        m = re.search(r"\+g([0-9a-f]{7,})", v)
        vllm_commit = m.group(1) if m else (v or "unknown")
    dtype = dtype_short(ea.get("dtype"))
    eager = "eager" if ea.get("enforce_eager") in (True, None) else "graph"
    att = _attention_short(ea.get("attention_backend"))
    batch = ea.get("max_num_seqs") if ea.get("max_num_seqs") is not None else "n/a"
    L = [
        f"CASE: {card.model.get('repo') or 'model-unknown'} / {dtype} / {eager} / {att} / batch={batch}",
        f"CORE COMMIT: {str(prov.get('core_commit') or 'n/a'):<10}      LANE COMMIT: {prov.get('lane_commit') or 'n/a'}",
        _row("PROTECTED FILES MODIFIED", c.get("protected_files_modified"), _pf(c.get("protected_files_modified"))),
        _row("EXISTING PRIMITIVES MODIFIED", c.get("existing_primitives_modified"), _pf(c.get("existing_primitives_modified"))),
        _row("NEW ROOT CLASSES", c.get("new_root_classes"), _pf(c.get("new_root_classes"))),
        _row("NEW ALLOWLIST ENTRIES", c.get("new_allowlist_entries"), _pf(c.get("new_allowlist_entries"))),
        _row("OLD GOLDEN DIGEST CHANGES", c.get("golden_digest_changes"), _pf(c.get("golden_digest_changes"))),
        _row("Unsupported", c.get("unsupported"), _pf(c.get("unsupported"))),
        _row("Ambiguous matches", c.get("ambiguous"), _pf(c.get("ambiguous"))),
        _row("Unknown/unvalidated roots", c.get("unknown_or_unvalidated_roots"), _pf(c.get("unknown_or_unvalidated_roots"))),
        _row("CUDA census residue", c.get("census_residue"), _pf(c.get("census_residue"))),
        _row("VU coverage (informational)", c.get("vu_coverage")),
        _row("New patterns", c.get("new_patterns")),
        _row("New registered definitions", c.get("new_definitions")),
        _row("Differential semantic tests", f"{c.get('differential_passed')}/{c.get('differential_total')}" if c.get("differential_total") is not None else "n/a"),
        _row("Captured specializations", f"{c.get('specs_compared') if c.get('specs_compared') is not None else 'n/a'}/{c.get('specs_captured') if c.get('specs_captured') is not None else 'n/a'}"),
        _row("Operand provenance (exhaustive)", c.get("operand_provenance") or "skipped"),
        _row("Sampled VUs (bit exact)", c.get("sampled_vus") or "skipped"),
        _row("Served tokens from logits", c.get("served_tokens") or "skipped"),
        _row("Chained slices", c.get("chained_slices") or "skipped"),
        _row("Control non-interference", c.get("control_noninterference") or "n/a"),
        _row("VU partition (G8, structural)", c.get("vu_partition") or vu_partition_text(card.gates or {})),
        "HOLDOUTS",
    ]
    hold = _detail(card.gates, "G7").get("holdouts")   # the holdout cases G7 ran; a card without them says (none)
    if hold:
        for name, h in hold.items():
            L.append(f"  {name:<{_W - 2}}{'PASS' if h.get('pass') else 'FAIL'}")
    else:
        L.append(f"  {'(none)':<{_W - 2}}n/a")
    L.append("SUPPORT CLAIM")
    L.append(f"  {_family_short(card)} dense / {dtype} / {eager} / {att} / batch={batch} / {sm} / vLLM {vllm_commit}")
    # D82: what the token rows mean — never "full root-to-token replay"
    L.append("  served tokens reproduced from the derived Program under exhaustive operand")
    L.append("  provenance + sampled local semantic checks (full replay: diagnostic only)")
    L.append(f"VERDICT: {card.verdict}")
    return "\n".join(L)


def summarize_md(card: ReportCard) -> str:
    """The top section (in a fenced block) followed by a short markdown summary."""
    ea, k = card.engine_args or {}, card.kappa or {}
    C = {f: v.get("value") for f, v in (card.C or {}).items()}
    gpu, host = k.get("gpu"), k.get("host")   # κ that names no GPU or no host prints None for each of their fields
    gpu_name, gpu_cc = (gpu.get("name"), gpu.get("cc")) if isinstance(gpu, dict) else (gpu, "") if gpu else (None, None)
    L = [f"# Card `{card.config_id}` — **{card.verdict}**" + ("  (unsealed: no provenance.json)" if card.unsealed else ""), "", "~~~text",
         top_section(card), "~~~", ""]
    L.append(f"- model: `{card.model.get('repo')}` @ `{str(card.model.get('revision') or '')[:12]}` (family reference `{card.model.get('family')}`, "
             f"hf `{card.model.get('hf_model_type')}`); C: " + " ".join(f"{f}={C.get(f)}" for f in ("H", "NL", "NH", "KVH", "D", "I", "V", "EPS", "BN") if f in C))
    L.append(f"- engine: dtype `{ea.get('dtype')}`, enforce_eager {ea.get('enforce_eager')}, max_num_seqs {ea.get('max_num_seqs')}, prefix caching "
             f"{ea.get('enable_prefix_caching')}, chunked prefill {ea.get('enable_chunked_prefill')}, tp {ea.get('tensor_parallel_size')}, "
             f"attention `{ea.get('attention_backend')}`; sampling {ea.get('sampling')}")
    wl = card.workload or {}
    L.append(f"- workload: `{wl.get('path')}` ({wl.get('n_requests')} requests, prompt lens {wl.get('prompt_lens')}, max_tokens {wl.get('max_tokens')}), "
             f"sha256 `{str(wl.get('sha256') or '')[:12]}`")
    if wl.get("serving_profile") or wl.get("realized"):
        rz = wl.get("realized")   # what the driver saw served; a card written before the GPU phase prints None for each
        L.append(f"- serving regime: profile {wl.get('serving_profile')}, context class {wl.get('context_class')}, arrivals {wl.get('arrive_steps')}; "
                 f"realized prompt lens {field_of(rz, 'prompt_lens')}, generated {field_of(rz, 'n_generated')}, engine steps capture/control "
                 f"{field_of(rz, 'engine_steps_capture')}/{field_of(rz, 'engine_steps_control')} (declared upper bound {wl.get('expected_engine_steps')}), "
                 f"peak concurrent requests {field_of(rz, 'max_concurrent_requests')}")
    L.append(f"- κ: vLLM `{k.get('vllm')}`, torch `{k.get('torch')}`, triton `{k.get('triton')}`, CUDA {k.get('cuda')}, GPU `{gpu_name}`"
             f" cc {gpu_cc}; host clocksource `{field_of(host, 'clocksource')}` ({field_of(host, 'clock_read_us')} µs/clock read)")
    L.append("")
    L.append("| gate | name | status | reason |")
    L.append("| --- | --- | --- | --- |")
    for gid in G.GATES:
        g = card.gates.get(gid)
        if not g:
            g = {}   # a gate the card has no result for prints as missing
        L.append(f"| {gid} | {g.get('name', G.GATE_NAMES.get(gid))} | **{g.get('status', 'missing')}** | {str(g.get('reason') or '').replace('|', '/')[:200]} |")
    for gid in G.STRUCTURAL_GATES:     # outside the frozen set: `n/a` (not `missing`) for a record that predates it
        g = card.gates.get(gid)
        if isinstance(g, dict):
            L.append(f"| {gid} | {g.get('name', G.GATE_NAMES.get(gid))} (structural) | **{g.get('status', 'missing')}** | {str(g.get('reason') or '').replace('|', '/')[:200]} |")
        else:
            L.append(f"| {gid} | {G.GATE_NAMES.get(gid)} (structural) | n/a | record predates G8 |")
    L.append("")
    d = card.diagnostics or {}
    ds, dfm, do = d.get("structure"), d.get("family_match"), d.get("overhead")   # a diagnostic the run did not compute prints None for each of its fields
    L.append(f"- diagnostics: per-decode-step histograms equal mod T: {field_of(ds, 'histograms_equal_mod_T')}, instance count constant: {field_of(ds, 'count_constant')} "
             f"(decode {field_of(ds, 'decode_instances')}, prefill {field_of(ds, 'prefill_instances')}), linear: {field_of(ds, 'linear')} (residual {field_of(ds, 'linear_residual')}); "
             f"family match `{field_of(dfm, 'family')}`: {field_of(dfm, 'status')} (extra {field_of(dfm, 'n_extra')}, unclassified {field_of(dfm, 'unclassified')})")
    if do and do.get("status") == "ok":
        rungs = do.get("rungs")   # the ladder's rungs, each with its step time; an overhead without them lists none
        L.append("- overhead ladder (step wall ms, × baseline, per-event µs minus clock): " + "; ".join(
            f"{r}: {v.get('step_wall_ms')} ({v.get('x_baseline')}×" + (f", {v.get('overhead_per_event_us_minus_clock')} µs" if v.get("overhead_per_event_us_minus_clock") is not None else "") + ")"
            for r, v in (rungs.items() if rungs else ())) + f"; host clock read {do.get('host', {}).get('clock_read_us')} µs ({do.get('host', {}).get('clocksource')})")
    L.append(f"- census: {len(card.census_kernels)} distinct kernel names; patterns used: {len(card.patterns_used)}; unsupported ops: {len(card.patterns_missing)}; "
             f"roots: {card.roots_by_class}")
    if card.per_step_instances:
        pre = [s['n_instances'] for s in card.per_step_instances if s.get('phase') == 'prefill']
        dec = sorted({s['n_instances'] for s in card.per_step_instances if s.get('phase') == 'decode'})
        L.append(f"- instances per step: prefill {pre}, decode {dec} over {len(card.per_step_instances)} steps")
    if card.spend.get("pod_hours") is not None or card.spend.get("usd") is not None:
        L.append(f"- spend: {card.spend.get('pod_hours')} pod-hours, ${card.spend.get('usd')}")
    failed = [s for s, r in (card.stages or {}).items() if r.get("status") not in ("ok", None)]
    if failed:
        L.append("- stages not ok: " + ", ".join(f"{s} ({card.stages[s].get('status')}, {card.stages[s].get('log')})" for s in failed))
    if card.mechanism_notes:
        L.append("- mechanism notes: " + ", ".join(f"`{p}`" for p in card.mechanism_notes))
    if card.notes:
        L.append("- notes: " + "; ".join(str(n) for n in card.notes))
    return "\n".join(L) + "\n"


def bootstrap_ci(xs: list[float], n: int = 4000, seed: int = 0) -> tuple[float, float]:
    """The 95% bootstrap interval of the mean of `xs` (`n` resamples), NaN for fewer than two."""
    if len(xs) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(xs, k=len(xs))) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


__all__ = ["CARD_SCHEMA", "C_FIELDS", "DERIVED_FIELDS", "ENGINE_ARG_KEYS", "ReportCard", "slug", "dtype_short",
           "engine_args_slug", "kappa_short", "config_id_of", "c_with_provenance", "log_header", "workload_summary", "engine_args_of",
           "vu_partition_text", "counts_of", "build_card", "load_card", "write_card", "top_section", "summarize_md", "bootstrap_ci"]
