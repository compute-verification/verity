"""The attempt ledger: every (arm, shape) a verified run reports becomes one evidence-store record of kind
``kernel-attempt/v1``, so what was tried, on which kernel, with what result and why it was kept or dropped is a query, not
prose.

A record's meta is everything the run knows, numbers included: the kernel variant (`verity.core.protocols.pouw.serving.KernelVariant`'s
id, ``{family}-{device}/{source_sha256[:12]}``), the parent attempt, the tier, the harness commit and gate version, the
metrics and the per-step time split, γ at both FP32 prices, the verifier's lines, and the one-line hypothesis. Judgments
that change after the run are labels on it (`KERNEL_OUTCOME`, `KERNEL_REASON`), since a record is immutable.

γ is never typed. It is looked up (`price_twins.lookup`) from the Lean-pinned price twins of the attempt's statement (its
panel line and version) on the priced path it ran (the gate's ``prices``, from the variant's), at both FP32 prices, and the
larger is published. An arm whose lineage declares ``gamma`` gets no record (`typed_gamma`); the arm's own `Work.gamma`
is kept as ``arm_claim`` and never published.

  python ledger.py publish RUN_DIR|RESULT_JSON [--store ROOT] [--push] [--by LANE]   records verify.py wrote -> the store
  python ledger.py outcome ART OUTCOME --reason TEXT --by LANE [--store ROOT] [--push]   label an attempt after the fact
  python ledger.py rows [--store ROOT] [--refresh] [--json]   the store's attempts as `panel.py` rows

`verify.py` builds the records (`records`), writes them under the run's ``ledger/`` (so they are preserved with the run's
files wherever the store isn't reachable) and publishes them with ``--publish``. `rows` is what `panel.py` reads beside
``attempts.jsonl``. Stdlib only; `research.store` and `verity.core.protocols.pouw.serving` are imported when present (from this checkout's
``tools/research/src`` and ``verity/core/protocols/pouw`` otherwise), so a node-2 job without them still writes its records.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

import price_twins as P

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

KIND = "kernel-attempt/v1"
#: the gate code's version, so a record's validity is derived from it rather than voided by hand: 1 = the reference verifier
#: accepts the timed run's own transcript (08:35Z); 2 = outputs poisoned before the dumped call and a no-write
#: negative control the verifier rejects (11:50Z)
GATES = "pouw-gates/2"
GATES_REQUIRED = 2
TIERS = ("1", "2a", "2b", "3")
OUTCOMES = ("screened", "frontier", "kept", "dropped", "voided", "superseded")
#: label keys (`research.store.vocab`): the producing lane about its own attempt
KERNEL_OUTCOME, KERNEL_REASON = "kernel_outcome", "kernel_reason"
#: a call's step groups, as `verity.core.protocols.pouw.serving.GROUPS` (#567) names them; checked against it when it is importable
GROUPS = ("a_tree", "forming", "gemm", "cleanup", "tile_hash", "screen")
PHASES = ("before", "gemm", "after")
ONE_LINE = 240


class LedgerError(ValueError):
    pass


def _import_path(rel: str) -> None:
    p = str(REPO / rel)
    if (REPO / rel).is_dir() and p not in sys.path:
        sys.path.append(p)


def serving():
    """`verity.core.protocols.pouw.serving` when importable (#567), else None."""
    try:
        from verity.core.protocols.pouw import serving as S  # noqa: PLC0415
    except ImportError:
        return None
    return S if hasattr(S, "KernelVariant") else None


def store_module():
    """`research.store.local` and `research.store.vocab`, from the environment or this checkout."""
    try:
        from research.store import local, vocab  # noqa: PLC0415
    except ImportError:
        _import_path("tools/research/src")
        from research.store import local, vocab  # noqa: PLC0415
    return local, vocab


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def one_line(s: str | None, what: str) -> str | None:
    if s is None:
        return None
    s = " ".join(str(s).split())
    if len(s) > ONE_LINE:
        raise LedgerError(f"{what} is one line of at most {ONE_LINE} characters, not {len(s)}")
    return s or None


# ---- the kernel variant ------------------------------------------------------------------------------------------------------


#: the digest entry a variant's run-time configuration takes beside its sources; no source file may have this path
CONFIG_PATH = "(run-time config)"


def source_sha256(files: Iterable[Path], root: Path, config: dict[str, str] | None = None) -> str:
    """The digest of the sources a binary was built from: sha256 over the canonical JSON of their sorted
    ``{path, sha256, bytes}`` entries, paths relative to `root`, the store's tree-payload rule (research.store README §2).
    `config` is what picks code from that binary at run time (``{"PEARLC_A_FORMS": "p,fold"}``): it enters as one more
    entry, the canonical JSON of its keys and values, so two forms of one binary are two variants. Without it the digest is
    the sources' alone."""
    entries = []
    for f in files:
        f = Path(f)
        data = (f if f.is_absolute() else root / f).read_bytes()
        rel = (f if f.is_absolute() else root / f).resolve().relative_to(root.resolve()).as_posix()
        if rel == CONFIG_PATH:
            raise LedgerError(f"a source may not be named {CONFIG_PATH!r}: the run-time config takes that entry")
        entries.append({"path": rel, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
    if not entries:
        raise LedgerError("a variant names at least one source")
    if config:
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in config.items()):
            raise LedgerError("a variant's run-time config maps names to strings")
        data = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        entries.append({"path": CONFIG_PATH, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
    entries.sort(key=lambda e: e["path"])
    if len({e["path"] for e in entries}) != len(entries):
        raise LedgerError("a variant's sources name a file twice")
    canon = json.dumps(entries, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(canon).hexdigest()


def variant_id(family: str, device: str, source_digest: str) -> str:
    return f"{family}-{device}/{source_digest[:12]}"


def variant_record(*, family: str, device: str, schemes: Iterable[str], sources: Iterable[Path], root: Path,
                   binaries: Iterable[Path] = (), schedules: Iterable[str] = ("serial",), loader: str = "cubin-ctypes",
                   prices: str | None = None, config: dict[str, str] | None = None) -> dict:
    """What an arm's `build` returns as ``variant``: `verity.core.protocols.pouw.serving.KernelVariant`'s fields (its shape rule and step
    IO aside) and its id. ``binary_sha256`` is the one binary's digest, or the digest of the sorted digests of several.
    `config` names every run-time choice that selects code from the binary (environment knobs such as
    ``PEARLC_A_FORMS``): it is part of the source digest, so the id tells the forms apart, and serving must pin it."""
    src = source_sha256(sources, root, config)
    shas = sorted(hashlib.sha256(Path(b).read_bytes()).hexdigest() for b in binaries)
    binary = shas[0] if len(shas) == 1 else hashlib.sha256("".join(shas).encode()).hexdigest() if shas else None
    rec = {"id": variant_id(family, device, src), "family": family, "device": device, "schemes": sorted(set(schemes)),
           "schedules": sorted(set(schedules)), "source_sha256": src, "binary_sha256": binary, "loader": loader,
           "prices": prices, "config": dict(sorted((config or {}).items())) or None}
    S = serving()
    if S is not None and binary is not None:
        v = S.KernelVariant(family=family, device=device, schemes=frozenset(rec["schemes"]),
                            schedules=frozenset(rec["schedules"]), shapes=S.ShapeRule(), source_sha256=src,
                            binary_sha256=binary, loader=loader, prices=prices)
        if v.id != rec["id"]:
            raise LedgerError(f"variant id {rec['id']} disagrees with verity.core.protocols.pouw.serving's {v.id}")
    return rec


# ---- γ at both FP32 prices, from the Lean-pinned price twins ------------------------------------------------------------------


def typed_gamma(result: dict) -> dict[str, str]:
    """The arms whose build record types γ in (a lineage ``gamma``): they get no record, since γ comes from the pins."""
    out = {}
    for name, arm in (result.get("arms") or {}).items():
        lineage = ((arm or {}).get("build") or {}).get("lineage") or {}
        if "gamma" in lineage:
            out[name] = ("γ is typed into its lineage: the ledger looks γ up from the Lean-pinned price twins on the "
                         "priced path the attempt ran (the variant's prices), so drop lineage.gamma and declare the path")
    return out


#: `verity.core.protocols.pouw.serving.SchemeId`'s FP8 grammar (#567), where the device names the record: sm120 (G = 4) is the panel's v1,
#: sm120-unpromoted its v2; an older name (``pearl-c-sm120-v2-hot-h3``) isn't checked
_SCHEME = re.compile(r"pearl-c-(?P<device>sm120|sm120-unpromoted)-(?:v0|v1)(?:-h[0-9]+)?(?:@[a-z0-9_]+)?")


def _scheme_conflict(scheme: str | None, statement: str | None) -> str | None:
    """The scheme's device record against the panel version's statement."""
    m = _SCHEME.fullmatch(scheme or "")
    if m is None or not statement or not statement.startswith("pearl-c-sm120/"):
        return None
    if (m["device"] == "sm120-unpromoted") != statement.split("/", 1)[1].startswith("v2"):
        return f"scheme {scheme} isn't {statement}'s device record"
    return None


def gamma(twins: list[dict] | None, panel: dict | None, scheme: str | None, gate: dict, shape: dict,
          arm_claim: Any) -> dict:
    """γ as the record keeps it: the forming-credited twins at both FP32 prices and the larger published, the chain-only
    reading beside it, every pin with its status (pinned in the Lean store, or staged), and the arm's own derivation kept
    as ``arm_claim``, never published."""
    stmt = P.statement((panel or {}).get("line"), (panel or {}).get("version"))
    if twins is None:
        out = {"statement": stmt, "path": gate.get("prices"), "published": None, "by_price": {},
               "missing": ["the price-twin table (price_twins.json)"]}
    else:
        out = P.lookup(twins, stmt, gate.get("prices"), shape.get("n"), shape.get("k"))
        co = P.lookup(twins, stmt, gate.get("prices"), shape.get("n"), shape.get("k"), reading="chain-only")
        out["chain_only"] = {"published": co["published"], "published_at": co["published_at"],
                             "pins": sorted(v["pin"] for v in co["by_price"].values())}
    conflict = _scheme_conflict(scheme, stmt)
    if conflict:
        out["missing"].append(conflict)
        out["published"] = None
    out["arm_claim"] = arm_claim
    return out


def _twins() -> list[dict] | None:
    try:
        return P.load()
    except (OSError, ValueError, KeyError):
        return None


# ---- one record per (arm, shape) ------------------------------------------------------------------------------------------------


def _num(x: Any) -> Any:
    """A JSON number the store's canonical form takes (no NaN, infinities or -0.0)."""
    if isinstance(x, float):
        if x != x or x in (float("inf"), float("-inf")):
            return None
        return x + 0.0
    return x


def _clean(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return _num(obj)


def _clocks(p: dict, side: str) -> list[float]:
    raw = p.get(f"sm_clock_{side}") or ""
    try:
        return [float(x) for x in str(raw).split(",") if x.strip()]
    except ValueError:
        return []


def _gate(result: dict, arm: str, shape: dict, variant: dict | None, verdict: dict, run_id: str) -> dict:
    """`verity.core.protocols.pouw.serving.GateRecord`'s fields for this (arm, shape), from what the run checked: every device gate of the
    arm at the shape passed (``exact``), the transcript the harness dumped from poisoned buffers verified
    (``poisoned_accepted``: an arm that dumps its own gets False), every control of the arm rejected (the harness's
    known-bad variants and the verifier's negative control), the SASS gate held. ``schedules_equal`` holds trivially for a
    variant with one schedule; the harness doesn't compare schedules, so it is None otherwise."""
    keys = [g for g in result.get("gates") or [] if len(g.get("key", [])) > 2 and g["key"][0] in ("arm", "chain")
            and g["key"][1] == shape["label"] and g["key"][2] == arm]
    negs = [x for x in result.get("negative_controls") or [] if arm in (x.get("key") or [])]
    sass = result.get("sass_gate") or {}
    toolkits = sorted({c.get("toolkit") or "unpinned" for b in sass.get("binaries") or [] if b.get("source") == f"arm:{arm}"
                       for c in b.get("cubins") or [] if c.get("functions")})
    poisoned = any(r.get("arm") == arm and r.get("shape") == shape["label"] and r.get("by") == "harness"
                   and not r.get("error") and not r.get("wrote_nothing") for r in result.get("poisoning") or [])
    schedules = (variant or {}).get("schedules") or []
    return {"variant": (variant or {}).get("id"), "run": run_id, "gates": GATES, "shapes": [[shape.get("n"), shape.get("k")]],
            "exact": bool(keys) and all(g.get("ok") for g in keys),
            "poisoned_accepted": bool(verdict.get("accepted")) and poisoned,
            "controls_rejected": bool(negs) and all(x.get("rejected") for x in negs) and bool(verdict.get("negative_control")),
            "schedules_equal": True if len(schedules) <= 1 else None,
            "sass_ok": bool(sass.get("ok")), "toolkit": ",".join(toolkits) or None, "prices": (variant or {}).get("prices")}


def _metrics(r: dict, p: dict) -> dict:
    g = r.get("graph") or {}
    m = {"method": "graph", "slowdown": p.get("slowdown", g.get("total_x")), "hash_free": p.get("hash_free", g.get("hash_free_x")),
         "baseline": p.get("baseline", g.get("baseline")), "baseline_kernel": p.get("baseline_kernel"),
         "baseline_ms": p.get("baseline_ms", g.get("baseline_ms")), "total_ms": g.get("total_ms"),
         "hash_free_ms": g.get("hash_free_ms"), "weight_side_ms": g.get("weight_side_ms"), "reuse_1pct": g.get("reuse_1pct"),
         "decode_method": p.get("decode_method"), "chain_length": p.get("chain_length"),
         "slowdown_side_stream": p.get("slowdown_side_stream"),
         "sm_clock": {"arm": _clocks(p, "arm"), "base": _clocks(p, "base")}, "clock_gap": p.get("clock_gap"),
         "consistency": r.get("consistency"), "rejected": r.get("rejected")}
    return {k: v for k, v in m.items() if v is not None}


def _steps(r: dict, chain: dict | None, lineage: dict) -> dict:
    """The call's time split: hashing against the hash-free call, extra flag roles, and at decode the chain's phases
    (before, during, after the GEMM, and on a side stream), each mapped to `GROUPS` when the lineage says how."""
    g = r.get("graph") or {}
    out: dict = {"hashing_ms": g.get("hashing_ms"), "extra_roles_x": g.get("extra_roles_x") or None, "phases": None,
                 "phase_groups": None}
    split = (chain or {}).get("split")
    if isinstance(split, dict) and isinstance(split.get("graph"), dict):
        s = split["graph"]
        out["phases"] = {"before": s.get("before_ms"), "gemm": s.get("during_ms"), "after": s.get("after_ms"),
                         "derive": s.get("derive_ms"), "side": s.get("side_ms"), "hidden_by_side": s.get("hidden_by_side_ms")}
    pg = lineage.get("phase_groups")
    if pg:
        groups = serving().GROUPS if serving() else GROUPS
        bad = sorted({x for v in pg.values() for x in v} - set(groups)) + sorted(set(pg) - set(PHASES))
        if bad:
            raise LedgerError(f"phase_groups names {bad}: phases are {PHASES}, groups {groups}")
        out["phase_groups"] = {k: list(v) for k, v in pg.items()}
    return {k: v for k, v in out.items() if v is not None}


def records(result: dict, *, run_id: str, commit: str | None, tier: str = "2a", hypothesis: str | None = None,
            parent: str | None = None, outcome: str | None = None, reason: str | None = None, by: str | None = None,
            time: str | None = None, twins: list[dict] | None | str = "table") -> list[dict]:
    """One record per arm row of every shape in a verified result (`verify.py`'s rewritten RESULT_JSON): ``kind``,
    ``meta``, ``refs`` and the ``labels`` to assert. `hypothesis`, `parent`, `outcome` and `reason` (the run's, from
    verify.py's flags) win over the arm's lineage. An arm that types γ in (`typed_gamma`) gets none. `twins` is the
    price-twin table (default: `price_twins.json`)."""
    if tier not in TIERS:
        raise LedgerError(f"tier {tier!r} is not one of {TIERS}")
    if outcome is not None and outcome not in OUTCOMES:
        raise LedgerError(f"outcome {outcome!r} is not one of {OUTCOMES}")
    if outcome not in (None, "screened") and not reason:
        raise LedgerError(f"outcome {outcome} needs a one-line reason")
    hypothesis, reason = one_line(hypothesis, "the hypothesis"), one_line(reason, "the reason")
    verification = {(v["arm"], v["shape"]): v for v in (result.get("verification") or {}).get("rows", [])}
    versions = result.get("versions") or {}
    refused = typed_gamma(result)
    table = _twins() if twins == "table" else twins
    out = []
    for shape in result.get("shapes", []):
        for name, r in (shape.get("arms") or {}).items():
            if name in refused:
                continue
            arm = (result.get("arms") or {}).get(name) or {}
            build = arm.get("build") or {}
            lineage = dict(build.get("lineage") or {})
            variant = build.get("variant")
            p = r.get("panel") or {}
            v = verification.get((name, shape["label"]), {})
            chain = ((shape.get("chain") or {}).get("arms") or {}).get(name)
            hyp = hypothesis or one_line(lineage.get("hypothesis"), "the lineage's hypothesis")
            par = parent or lineage.get("parent")
            panel = lineage.get("panel") or build.get("panel")
            meta = {
                "schema": KIND, "time": time or now(), "run": run_id, "run_id": run_id, "harness": commit, "tier": tier, "by": by,
                "arm": name, "stream": arm.get("stream"), "precision": arm.get("precision") or r.get("precision"),
                "shape": shape["label"], "m": shape.get("m"), "k": shape.get("k"), "n": shape.get("n"),
                "phase": shape.get("phase"), "panel_shape": p.get("shape"),
                "variant": (variant or {}).get("id"), "variant_record": variant, "parent": par,
                "scheme": lineage.get("scheme") or build.get("scheme"), "schedule": lineage.get("schedule"),
                "panel": panel, "approach": lineage.get("approach"), "mechanism": sorted(set(lineage.get("mechanism") or [])),
                "hypothesis": hyp,
                "gates": {"version": GATES, "arm_interface": (result.get("harness") or {}).get("interface"),
                          "verify": (result.get("verification") or {}).get("interface"),
                          "sass_gate": (result.get("sass_gate") or {}).get("version")},
                "gate": (gate := _gate(result, name, shape, variant, v, run_id)),
                "metrics": _metrics(r, p), "steps": _steps(r, chain, lineage),
                "gamma": gamma(table, panel, lineage.get("scheme") or build.get("scheme"), gate, shape, r.get("gamma")),
                "work": {"w_ref": r.get("w_ref"), "credited": r.get("credited"), "unit": r.get("unit")},
                "verify": {"accepted": bool(v.get("accepted")), "why": v.get("why"), "commit": p.get("verifier_commit"),
                           "accept": p.get("verifier_accept"), "transcript": p.get("transcript"),
                           "negative_control": p.get("negative_control")},
                "versions": {k: versions.get(k) for k in ("driver", "cuda_runtime", "nvcc", "cublaslt") if k in versions},
                "clock_label": (result.get("clocks") or {}).get("label"),
            }
            meta["missing"] = [f for f, ok in (("variant", meta["variant"]), ("hypothesis", hyp), ("panel line", panel),
                                               ("verifier accept", meta["verify"]["accepted"]),
                                               ("negative control", meta["verify"]["negative_control"]))
                               if not ok] + meta["gamma"]["missing"]
            refs = {"parent": par} if isinstance(par, str) and par.startswith("art:") else {}
            labels = []
            if outcome or tier in ("2a", "2b"):
                labels.append((KERNEL_OUTCOME, outcome or "screened"))
            if reason:
                labels.append((KERNEL_REASON, reason))
            out.append({"kind": KIND, "meta": _clean(meta), "refs": refs, "labels": labels})
    return out


def record_path(run_dir: Path, rec: dict) -> Path:
    m = rec["meta"]
    return run_dir / "ledger" / m["arm"] / f"{m['shape']}.json"


def write(run_dir: Path, recs: list[dict]) -> list[Path]:
    paths = []
    for rec in recs:
        p = record_path(run_dir, rec)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rec, indent=1, sort_keys=True))
        paths.append(p)
    return paths


def read(run_dir: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted((run_dir / "ledger").glob("*/*.json"))]


OUTPUTS = "outputs.json"
OUTPUTS_SCHEMA = "research/outputs/v0.1"


def write_outputs(run_dir: Path, recs: list[dict]) -> Path:
    """Declares the records in the run's ``outputs.json`` (`research.store.attempt.typed_outputs`), so the runner of a
    ``research run --on`` publishes them with the attempt, with the store key it holds and the workload doesn't. Each is
    a payload-less typed output named ``kernel-attempt/<arm>/<shape>``, whose manifest is the one `publish` puts, so either
    path gives the same art: id. Entries the workload declared before are kept."""
    path = run_dir / OUTPUTS
    doc = json.loads(path.read_text()) if path.is_file() else {"schema": OUTPUTS_SCHEMA, "outputs": []}
    if doc.get("schema") != OUTPUTS_SCHEMA:
        raise LedgerError(f"{path} is {doc.get('schema')!r}, not {OUTPUTS_SCHEMA}")
    mine = {f"{KIND.split('/')[0]}/{r['meta']['arm']}/{r['meta']['shape']}": r for r in recs}
    doc["outputs"] = [e for e in doc.get("outputs") or [] if e.get("name") not in mine]
    doc["outputs"] += [{"name": n, "kind": r["kind"], "meta": r["meta"], "refs": r.get("refs") or {}} for n, r in mine.items()]
    tmp = path.with_name(path.name + ".ledger-tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True))
    os.replace(tmp, path)
    return path


# ---- the store -----------------------------------------------------------------------------------------------------------------


def open_store(root: Path | str | None = None):
    local, _ = store_module()
    return local.LocalStore(root=Path(root) if root else None)


def _label(st, vocab, target: str, key: str, value: Any, by: str, ref: str | None) -> bool:
    """Asserts (key, value) on target unless `by` already said exactly that; refuses what the vocabulary can't place."""
    why = vocab.check(key, value)
    if why:
        raise LedgerError(why)
    latest = [lab for lab in st.labels(target, key=key) if lab.by == by]
    if latest and latest[-1].value == value:
        return False
    st.add_label(target, key, value, by, ref=ref)
    return True


def publish(recs: list[dict], *, store=None, by: str, push: bool = False) -> list[dict]:
    """Puts each record (content-addressed, so publishing the same run twice is a no-op) and its labels; with `push`, onto
    the store's remote. Returns ``{art, arm, shape, labels, pushed}`` per record."""
    if not by:
        raise LedgerError("publishing needs --by (the lane that asserts the outcome labels)")
    _, vocab = store_module()
    st = store if store is not None else open_store()
    out = []
    for rec in recs:
        art = st.put_artifact(rec["kind"], rec["meta"], rec.get("refs") or {})
        added = [k for k, v in rec.get("labels", []) if _label(st, vocab, art, k, v, by, rec["meta"].get("run"))]
        pushed = None
        if push:
            pushed = st.push(art)
        out.append({"art": art, "arm": rec["meta"]["arm"], "shape": rec["meta"]["shape"], "labels": added,
                    "pushed": bool(pushed)})
    return out


def outcome(art: str, value: str, *, reason: str | None, by: str, ref: str | None = None, store=None) -> list[str]:
    if value not in OUTCOMES:
        raise LedgerError(f"outcome {value!r} is not one of {OUTCOMES}")
    if value != "screened" and not reason:
        raise LedgerError(f"outcome {value} needs a one-line reason")
    _, vocab = store_module()
    st = store if store is not None else open_store()
    st.get_manifest(art)
    done = [KERNEL_OUTCOME] if _label(st, vocab, art, KERNEL_OUTCOME, value, by, ref) else []
    if reason and _label(st, vocab, art, KERNEL_REASON, one_line(reason, "the reason"), by, ref):
        done.append(KERNEL_REASON)
    return done


def attempts(store=None) -> list[tuple[str, dict, dict]]:
    """Every ``kernel-attempt/v1`` record in the store's index: (art id, meta, latest label per key)."""
    st = store if store is not None else open_store()
    out = []
    for row in st.index.select_artifacts(kind=KIND):
        m = row["manifest"]
        latest: dict = {}
        for lab in st.labels(row["id"]):
            if lab.key in (KERNEL_OUTCOME, KERNEL_REASON):
                latest[lab.key] = lab.value
        out.append((row["id"], m.get("meta") or {}, latest))
    return out


# ---- panel rows ----------------------------------------------------------------------------------------------------------------


def measured_refusal(meta: dict) -> str | None:
    """Why a tier-2b record can't stand as a measured panel row, as `panel.py append` would refuse it, or None."""
    v, run = meta.get("verify") or {}, meta.get("run") or ""
    try:
        gates = int(str((meta.get("gates") or {}).get("version", "")).rsplit("/", 1)[-1])
    except ValueError:
        gates = 0
    if gates < GATES_REQUIRED:
        return f"gate version {gates} predates {GATES_REQUIRED}"
    if not v.get("accepted") or "accept" not in (v.get("accept") or "").lower():
        return "no reference-verifier accept on the run's own transcript"
    if run not in (v.get("transcript") or ""):
        return "the transcript doesn't name the run"
    nc = v.get("negative_control") or ""
    if "reject" not in nc.lower() or run not in nc:
        return "no verifier reject line, naming the run, on the negative control"
    clocks = (meta.get("metrics") or {}).get("sm_clock") or {}
    if not clocks.get("arm") or not clocks.get("base"):
        return "no per-rep SM clocks for both sides"
    return None


def panel_row(art: str, meta: dict, labels: dict) -> dict | None:
    """The record as a `panel.py` row, or None when it isn't one (no panel line, not a headline phase, tier 1 or 3).
    A tier-2a record is a screen: its shape gets the ``:screen`` suffix, which the panel logs and never plots. Either tier
    carries ``verifier`` only when it meets every rule `panel.py append` enforces; otherwise it stays in the table,
    unverified, so only a verified 2b record can plot. ``attempt`` is the lineage's, or None for `panel.py` to number."""
    panel, phase, tier = meta.get("panel") or {}, meta.get("phase"), meta.get("tier")
    if not panel.get("line") or not panel.get("version") or phase not in ("prefill", "decode") or tier not in ("2a", "2b"):
        return None
    mt = meta.get("metrics") or {}
    if mt.get("slowdown") is None:
        return None
    shape = meta.get("panel_shape") or meta.get("shape")
    g = (meta.get("gamma") or {}).get("published")
    outcome, reason = labels.get(KERNEL_OUTCOME), labels.get(KERNEL_REASON)
    what = [meta.get("hypothesis") or "no hypothesis recorded"]
    if meta.get("variant"):
        what.append(f"variant {meta['variant']}" + (f", parent {meta['parent']}" if meta.get("parent") else ""))
    if meta.get("mechanism"):
        what.append("mechanism " + ", ".join(meta["mechanism"]))
    gm = meta.get("gamma") or {}
    if g is not None:
        what.append(f"γ {gm.get('path')} at FADD {gm.get('published_at')}, the larger, from "
                    + ("staged price twins (not yet in the Lean store)" if "staged" in (gm.get("pins") or []) else
                       "Lean-store price twins"))
    else:
        what.append("γ not published: " + "; ".join(gm.get("missing") or ["no price twin"]))
    if outcome:
        what.append(f"outcome {outcome}" + (f": {reason}" if reason else ""))
    row = {"attempt": panel.get("attempt"), "line": panel["line"], "version": panel["version"],
           "precision": meta.get("stream"), "phase": phase, "shape": shape if tier == "2b" else f"{shape}:screen",
           "kind": "measured", "slowdown": mt["slowdown"], "lo": None, "hi": None,
           "gamma": float(Fraction(g)) if g is not None else None, "change": panel.get("change") or "kernel",
           "description": "; ".join(what), "source": f"{art} (kernel-attempt/v1)", "run": meta.get("run"),
           "basis": None, "assumptions": [], "hash_free_slowdown": mt.get("hash_free"),
           "decode_method": mt.get("decode_method"), "by": meta.get("by"), "time": (meta.get("time") or "")[:16] + "Z",
           "record": art, "tier": tier, "variant": meta.get("variant")}
    why = measured_refusal(meta)
    if why:
        row["store_refused"] = why
    else:
        v = meta["verify"]
        row["verifier"] = {"commit": v["commit"], "accept": v["accept"], "transcript": v["transcript"],
                           "negative_control": v["negative_control"]}
        row["sm_clock_mhz"] = {"arm": mt["sm_clock"]["arm"], "baseline": mt["sm_clock"]["base"]}
    if outcome == "voided":
        row["voided"] = reason or "voided by its lane"
    return row


def panel_rows(entries: list[tuple[str, dict, dict]]) -> list[dict]:
    rows = [r for art, meta, labels in entries if (r := panel_row(art, meta, labels))]
    return sorted(rows, key=lambda r: (r["time"], r["record"]))


# ---- the CLI -------------------------------------------------------------------------------------------------------------------


def _run_dir(path: Path) -> Path:
    return path if path.is_dir() else path.resolve().parent


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("publish", help="put a run's records (its ledger/ directory) into the store")
    p.add_argument("run", type=Path, help="the run's directory, or its result JSON")
    p.add_argument("--by", default=os.environ.get("GPU_LEASE_WHO") or os.environ.get("RESEARCH_LANE"))
    o = sub.add_parser("outcome", help="label an attempt: screened, frontier, kept, dropped, voided or superseded")
    o.add_argument("art")
    o.add_argument("outcome", choices=OUTCOMES)
    o.add_argument("--reason")
    o.add_argument("--by", required=True)
    o.add_argument("--ref", help="the evidence: a run id, art: id or note:")
    r = sub.add_parser("rows", help="the store's attempts as panel.py rows")
    r.add_argument("--refresh", action="store_true", help="refresh the local index from the store's remote first")
    r.add_argument("--json", action="store_true")
    for s in (p, o, r):
        s.add_argument("--store", help="the store's root (default: $RESEARCH_STORE or ~/.research/store)")
    for s in (p, o):
        s.add_argument("--push", action="store_true", help="push onto the store's remote")
    a = ap.parse_args(argv)
    try:
        st = open_store(a.store)
        if a.cmd == "publish":
            recs = read(_run_dir(a.run))
            if not recs:
                raise LedgerError(f"no records under {_run_dir(a.run) / 'ledger'} (verify.py writes them)")
            for x in publish(recs, store=st, by=a.by, push=a.push):
                print(f"{x['art']}  {x['arm']} {x['shape']}" + (f"  +{','.join(x['labels'])}" if x["labels"] else "")
                      + ("  pushed" if x["pushed"] else ""))
        elif a.cmd == "outcome":
            done = outcome(a.art, a.outcome, reason=a.reason, by=a.by, ref=a.ref, store=st)
            if a.push:
                st.push(a.art)
            print(f"{a.art}: {', '.join(done) or 'unchanged'}")
        else:
            if a.refresh:
                from research.store import cli as SC  # noqa: PLC0415
                SC.main(["refresh"] + (["--store", a.store] if a.store else []))
            rows = panel_rows(attempts(st))
            if a.json:
                print(json.dumps(rows))
            else:
                for x in rows:
                    print(f"{x['record']}  {x['line']} {x['version']} {x['phase']} {x['shape']} {x['slowdown']}x "
                          f"{'measured' if x.get('verifier') else x.get('store_refused') or 'screen'}")
    except (LedgerError, OSError, KeyError) as ex:
        print(f"ledger: {type(ex).__name__}: {ex}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
