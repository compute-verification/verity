"""W1cF: closes audit findings F2/F3/F4/F5/F7/F9/F10 of ``docs/vllm-poc/audit-compiled-pass.md`` for
``verity_vllm_adapter.verify``. Kept out of ``verify.py`` on purpose (GUIDANCE-2026-09-07 §6: additive
edits, WB's F1/F11 land in the same file): every helper here is a pure function over things the
verifier already holds, and ``verify.main`` adds one call site per finding.

====  ====================================  ====================================================
F2    :func:`kernel_set_binding`            recompute production's kernel-set hash over the bundled
                                            ``kernels/*.py`` and compare with the manifest's
                                            ``compilation.generated_kernels.kernel_set_sha256``;
                                            mismatch = structural FAIL (WA's M8)
F3    :func:`description_binding`           ``required.json.description_digest`` vs the digest of the
                                            loaded ``description.json``; mismatch = structural FAIL
F4    :func:`relation_errors_of`            checker calls that raised -> Table 1 ``relation_errors``;
                                            ``rows.verdict`` counts them as rejections (WA's M3b)
F5    :func:`negatives_not_run`             applicable negative ids with 0 attempted mutations are
                                            NOT_RUN (Table 1 ``negatives_not_run``), not passed
F7    :func:`verifier_environment`          installed vllm / torch / triton identities of *this*
                                            process vs the manifest's -> qualifications, not FAIL
F9    :func:`refusal_row`                   a refusal (bad bundle) is a Table 1 row ``FAIL structural``
                                            plus a RESULT line, not a traceback (WA's M2/M3/M6)
F10   :func:`ptx_identity`                  the PTX this process compiles from the bundled sources vs
                                            the production PTX shipped in the bundle, per variant;
                                            the batch-invariant GEMM's ``NUM_SMS`` specialisation is
                                            reported as ``specialised: NUM_SMS``, not a mismatch
====  ====================================  ====================================================

Vocabulary (GUIDANCE §10.2): the fused-kernel relation and everything launch-level here are
*capture-level*: evidence that the replay reproduces production's function and a means of obtaining
logical values, not Verity primitives and not verification units.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from verity_vllm import rows

_DEF = re.compile(r"^def (\w+)\(", re.M)
_TARGET = re.compile(r"^\.target\s+(\S+)", re.M)
GEMM_SPECIALISED = "NUM_SMS"  # the batch-invariant GEMM bakes the SM count into its compiled code


def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# --------------------------------------------------------------------------------------- F2 ---


def kernel_set_of_dir(kernels_dir: Path) -> dict[str, Any]:
    """Recompute :func:`verity_vllm_adapter.kernel_identity.collect`'s ``kernel_set_sha256`` rule over
    a bundle's ``kernels/<sha256>.py``: the sha256 of the JSON of the sorted set of
    ``(kernel function name, source sha256)`` pairs. Files that do not hash to their name are listed
    and excluded (the verifier already refuses such a bundle)."""
    d = Path(kernels_dir)
    pairs: set[tuple[str, str]] = set()
    bad: list[str] = []
    files = 0
    # W1d's compiled-fx bundles (`compiled_fx.capture.write_kernels`) key `kernels/<sha256[:16]>.py` by the
    # *name-normalised* source and declare it in kernels/index.json ("hash"); under W1c's rule every such file is
    # a bad hash (CF2, 2026-09-08: the merged verifier refused the H100 bundle and W1d's PASSed A100 bundle alike).
    idx = d / "index.json"
    contract = (json.loads(idx.read_text()).get("hash") or "") if idx.exists() else ""
    normalised = "name-normalised" in contract
    if normalised:
        from verity_vllm_adapter.compiled_fx.kernels import normalise_kernel
    for p in sorted(d.glob("*.py")):
        src = p.read_text()
        sha = sha256_hex(normalise_kernel(src).encode())[:16] if normalised else sha256_hex(src.encode())
        files += 1
        if sha != p.stem:
            bad.append(p.name)
            continue
        m = _DEF.search(src)
        pairs.add((m.group(1) if m else p.stem, sha))
    ordered = sorted(pairs)
    return {
        "files": files,
        "bad_hash": bad,
        "pairs": [list(x) for x in ordered],
        "contract": contract or "sha256 of the file content (W1c)",
        "kernel_set_sha256": sha256_hex(json.dumps(ordered).encode()) if ordered else None,
    }


def kernel_set_binding(run_dir: Path, manifest: dict[str, Any], production_kernels: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """F2. ``None`` when not applicable (no ``kernels/``: eager profile, nothing generated). Otherwise
    ``ok`` iff the recomputed hash equals production's recorded one; ``problem`` is the structural
    failing-invariant text when it does not (or when production recorded none)."""
    kd = Path(run_dir) / "kernels"
    if not kd.is_dir():
        return None
    recomputed = kernel_set_of_dir(kd)
    recorded = ((manifest.get("compilation") or {}).get("generated_kernels") or {}).get("kernel_set_sha256")
    out: dict[str, Any] = {
        "rule": "sha256(json(sorted {(triton def name, sha256(source))})) over kernels/*.py == manifest.compilation.generated_kernels.kernel_set_sha256",
        "recomputed": recomputed["kernel_set_sha256"],
        "recorded": recorded,
        "kernels": len(recomputed["pairs"]),
        "bad_hash": recomputed["bad_hash"],
        "ok": recorded is not None and recorded == recomputed["kernel_set_sha256"],
        "problem": None,
    }
    if recorded is None:
        out["problem"] = "kernel set unbound: the production manifest records no compilation.generated_kernels.kernel_set_sha256 for a bundle that ships kernels/"
    elif "name-normalised" in recomputed.get("contract", ""):
        # W1d's bundle: its kernels/ are the name-normalised sources (one file per kernel body across pieces); the
        # manifest's kernel_set_sha256 is W1c's rule over production's Inductor cache (full sources, one per piece).
        # The two are different commitments to the same kernel set and cannot be compared word for word; W1d binds
        # them through kernel_identity.json (production cache vs replay cache, both normalised). Reported, not ok.
        out["ok"] = False
        out["problem"] = (f"kernel set recorded under another contract: the bundle's {len(recomputed['pairs'])} kernels/ are name-normalised "
                          f"sources (sha256[:16]); production's manifest kernel_set_sha256 {recorded[:12]}… is W1c's rule over "
                          f"{((manifest.get('compilation') or {}).get('generated_kernels') or {}).get('distinct_kernel_sources')} full sources — "
                          "no production commitment under the bundle's contract (W1c/W1d integration; CF2 2026-09-08)")
    elif not out["ok"]:
        out["problem"] = (f"bundled kernels/ hash to kernel set {(recomputed['kernel_set_sha256'] or 'none')[:12]}…, production's manifest recorded "
                          f"{recorded[:12]}… ({len(recomputed['pairs'])} bundled kernel sources)")
        if production_kernels:
            prod = {(k["name"], k["source_sha256"]) for w in production_kernels.get("wrappers", []) for k in w.get("kernels", [])}
            here = {tuple(p) for p in recomputed["pairs"]}
            out["only_bundle"] = sorted(f"{n}:{s[:12]}" for n, s in here - prod)
            out["only_production"] = sorted(f"{n}:{s[:12]}" for n, s in prod - here)
    return out


# --------------------------------------------------------------------------------------- F3 ---


def description_binding(required_doc: dict[str, Any] | None, description: Any) -> dict[str, Any] | None:
    """F3. ``None`` without a ``required.json``. Otherwise compares its ``description_digest`` with the
    digest of the description actually loaded; ``problem`` set on mismatch or absence."""
    if required_doc is None:
        return None
    recorded = required_doc.get("description_digest")
    loaded = description.digest().hex()
    out = {"recorded": recorded, "loaded": loaded, "ok": recorded is not None and recorded == loaded, "problem": None}
    if recorded is None:
        out["problem"] = "required.json carries no description_digest"
    elif not out["ok"]:
        out["problem"] = f"required.json.description_digest {recorded[:12]}… != digest {loaded[:12]}… of the loaded description.json"
    return out


# --------------------------------------------------------------------------------------- F4 ---


def relation_errors_of(report: Any) -> dict[str, Any]:
    """F4. Checker calls that raised (``VUResult.status == 'error'``), for Table 1 ``relation_errors``.
    ``rows.verdict`` turns ``count > 0`` into ``FAIL relation_checks.error``."""
    errs = [r for r in report.results if getattr(r, "status", None) == "error"]
    by_rel = Counter(str(r.relation_id) for r in errs)
    first = None
    if errs:
        e = errs[0]
        first = f"{e.relation_id} on {e.canonical_id}: {e.detail}"[:400]
    return {"count": len(errs), "first": first, "by_relation": dict(sorted(by_rel.items()))}


# --------------------------------------------------------------------------------------- F5 ---


def negatives_not_run(negatives: list[dict[str, Any]], requested_ids: list[str]) -> list[str]:
    """F5. Requested negative ids whose harness attempted 0 mutations (every variant inconclusive,
    the harness raised, or the id never ran). They are NOT_RUN: excluded from the ``negative``
    count (they contribute 0/0 anyway) and listed so ``rows.verdict`` downgrades to PARTIAL."""
    attempted = {str(d.get("negative_id")): int(d.get("attempted", 0) or 0) for d in negatives}
    return [nid for nid in requested_ids if attempted.get(nid, 0) == 0]


def mark_not_run(negatives: list[dict[str, Any]], not_run: list[str]) -> None:
    for d in negatives:
        d["status"] = "NOT_RUN" if d.get("negative_id") in not_run else ("rejected" if d.get("unexpectedly_accepted", 0) == 0 else "ACCEPTED")


# --------------------------------------------------------------------------------------- F7 ---


def dist_identity(name: str) -> dict[str, Any]:
    """Identity of an installed distribution: version, sha256 of its RECORD (every installed file with
    its hash: an installation fingerprint), pip's ``direct_url.json`` and, when that names a local wheel
    file that still exists, the wheel's sha256."""
    import importlib.metadata as md

    try:
        dist = md.distribution(name)
    except md.PackageNotFoundError:
        return {"installed": False}
    out: dict[str, Any] = {"installed": True, "version": dist.version, "record_sha256": None, "direct_url": None, "wheel_sha256": None}
    rec = dist.read_text("RECORD")
    if rec:
        out["record_sha256"] = sha256_hex(rec.encode())
    du = dist.read_text("direct_url.json")
    if du:
        try:
            d = json.loads(du)
        except ValueError:
            d = {}
        url = d.get("url")
        out["direct_url"] = url
        h = (d.get("archive_info") or {}).get("hash")
        if isinstance(h, str) and h.startswith("sha256="):
            out["wheel_sha256"] = h.split("=", 1)[1]
        if out["wheel_sha256"] is None and isinstance(url, str) and url.startswith("file://"):
            p = Path(url[len("file://"):])
            if p.is_file():
                hh = hashlib.sha256()
                with p.open("rb") as f:
                    for chunk in iter(lambda: f.read(1 << 22), b""):
                        hh.update(chunk)
                out["wheel_sha256"] = hh.hexdigest()
    return out


def installed_environment() -> dict[str, Any]:
    """What this verifier process runs on (F7): the three distributions whose kernels the same-runtime
    checkers launch, CUDA and the device (name + SM count: the GEMM's ``NUM_SMS`` specialisation)."""
    env: dict[str, Any] = {k: dist_identity(k) for k in ("vllm", "torch", "triton")}
    try:  # the manifest records vllm.__version__ (no wheel-local suffix); the dist version carries e.g. ".cu129"
        import vllm

        env["vllm"]["module_version"] = vllm.__version__
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch

        env["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            env["gpu"] = {"name": p.name, "multi_processor_count": int(p.multi_processor_count), "compute_capability": f"{p.major}.{p.minor}"}
        else:
            env["gpu"] = None
    except Exception as e:  # noqa: BLE001
        env["cuda"] = None
        env["gpu"] = {"error": f"{type(e).__name__}: {e}"}
    return env


def environment_qualifications(env: dict[str, Any], manifest: dict[str, Any]) -> list[str]:
    """F7. Differences between the verifier's environment and the manifest's, as qualification
    strings (printed in the verdict line, never a FAIL: a same-runtime check under a different wheel
    is still a check, but the row must say so)."""
    q: list[str] = []
    m_vllm = manifest.get("vllm") or {}
    m_py = manifest.get("python") or {}
    m_gpu = manifest.get("gpu") or {}
    m_dists = manifest.get("dists") or {}  # W1cF manifests: record_sha256 / wheel_sha256 per dist
    v = env.get("vllm") or {}
    if not v.get("installed"):
        q.append("vllm not importable in the verifier")
    else:
        if v.get("wheel_sha256") and m_vllm.get("wheel_sha256"):
            if v["wheel_sha256"] != m_vllm["wheel_sha256"]:
                q.append(f"vllm wheel sha256 differs: verifier {v['wheel_sha256'][:12]}… vs production {m_vllm['wheel_sha256'][:12]}…")
        elif m_vllm.get("wheel_sha256"):
            q.append("vllm wheel sha256 not comparable: the verifier's installed wheel file is not measurable here (pip recorded no hash)")
        ver = v.get("module_version") or v.get("version")
        if m_vllm.get("version") and ver and ver != m_vllm["version"]:
            q.append(f"vllm version differs: verifier {ver} vs production {m_vllm['version']}")
    for dist, key in (("torch", "torch"), ("triton", "triton")):
        e = env.get(dist) or {}
        if not e.get("installed"):
            q.append(f"{dist} not importable in the verifier")
            continue
        if m_py.get(key) and e.get("version") and e["version"] != m_py[key]:
            q.append(f"{dist} version differs: verifier {e['version']} vs production {m_py[key]}")
    for dist in ("vllm", "torch", "triton"):
        e = env.get(dist) or {}
        md_ = m_dists.get(dist) or {}
        if e.get("record_sha256") and md_.get("record_sha256") and e["record_sha256"] != md_["record_sha256"]:
            q.append(f"{dist} installation differs (RECORD sha256): verifier {e['record_sha256'][:12]}… vs production {md_['record_sha256'][:12]}…")
    if m_py.get("cuda") and env.get("cuda") and env["cuda"] != m_py["cuda"]:
        q.append(f"CUDA differs: verifier {env['cuda']} vs production {m_py['cuda']}")
    g = env.get("gpu") or {}
    if m_gpu.get("name") and g.get("name") and g["name"] != m_gpu["name"]:
        q.append(f"GPU differs: verifier {g['name']} vs production {m_gpu['name']}")
    if m_gpu.get("multi_processor_count") and g.get("multi_processor_count") and int(g["multi_processor_count"]) != int(m_gpu["multi_processor_count"]):
        q.append(f"SM count differs: verifier {g['multi_processor_count']} vs production {m_gpu['multi_processor_count']} (GEMM NUM_SMS specialisation)")
    return q


def verifier_environment(manifest: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    env = installed_environment()
    return env, environment_qualifications(env, manifest)


# -------------------------------------------------------------------------------------- F10 ---


def ptx_target(ptx: str) -> str | None:
    m = _TARGET.search(ptx)
    return m.group(1) if m else None


def load_shipped_ptx(kernels_dir: Path) -> dict[str, Any]:
    """The production PTX shipped in a bundle (``kernels/ptx-index.json`` + ``kernels/<ptx sha>.ptx``,
    capture-format revision W1cF). Files must hash to their names (normalised text)."""
    d = Path(kernels_dir)
    idx_path = d / "ptx-index.json"
    if not idx_path.is_file():
        return {"available": False}
    idx = json.loads(idx_path.read_text())
    by_name: dict[str, dict[str, dict[str, Any]]] = {}
    bad: list[str] = []
    missing: list[str] = []
    for e in idx.get("entries", []):
        sha = e["ptx_code_sha256"]
        p = d / f"{sha}.ptx"
        if not p.is_file():
            missing.append(sha)
            continue
        if sha256_hex(p.read_bytes()) != sha:
            bad.append(p.name)
            continue
        by_name.setdefault(e["name"], {})[sha] = {**e, "target": e.get("target") or ptx_target(p.read_text())}
    return {"available": True, "by_name": by_name, "bad_hash": bad, "missing": missing,
            "ptx_code_set_sha256": idx.get("ptx_code_set_sha256"), "production_gpu": idx.get("gpu")}


def ptx_paths(triton_dir: str | Path | None) -> set[str]:
    """Every ``*.ptx`` already in a Triton cache dir (snapshot taken when the verifier starts: anything
    present then was compiled by another process, e.g. a replay sharing the cache root)."""
    root = Path(triton_dir) if triton_dir else None
    if root is None or not root.is_dir():
        return set()
    return {str(p.relative_to(root)) for p in root.rglob("*.ptx")}


def scan_triton_cache(triton_dir: Path, names: set[str]) -> dict[str, dict[str, dict[str, Any]]]:
    """PTX in a Triton cache dir (``<key>/<fn name>.ptx``) for the given kernel names, keyed by name
    then by normalised-code sha256."""
    from verity_vllm_adapter.kernel_identity import ptx_code

    out: dict[str, dict[str, dict[str, Any]]] = {}
    root = Path(triton_dir)
    if not root.is_dir():
        return out
    for p in sorted(root.rglob("*.ptx")):
        if p.stem not in names:
            continue
        text = p.read_text()
        code = ptx_code(text)
        sha = sha256_hex(code.encode())
        out.setdefault(p.stem, {})[sha] = {"path": str(p.relative_to(root)), "target": ptx_target(text), "cache_key": p.parent.name}
    return out


def compare_ptx_variants(shipped: dict[str, dict[str, Any]], compiled: dict[str, dict[str, Any]], *,
                         specialised_names: set[str], gpu_differs: bool) -> dict[str, Any]:
    """Per kernel name: ``identical`` (same variant set), ``subset`` (every variant compiled here is one
    production shipped; production compiled more, e.g. autotune candidates), ``specialised: NUM_SMS``
    (the GEMM differs and the verifier's GPU differs from production's: legitimate), ``target``
    (different ``.target`` architecture: PTX not comparable), ``mismatch`` (a variant compiled here
    is not among production's), ``not_compiled`` (nothing compiled here for the name),
    ``not_shipped`` (compiled here, production shipped none)."""
    per: dict[str, dict[str, Any]] = {}
    for name in sorted(set(shipped) | set(compiled)):
        s, c = shipped.get(name, {}), compiled.get(name, {})
        entry: dict[str, Any] = {"shipped": sorted(s), "compiled": sorted(c)}
        if not c:
            entry["status"] = "not_compiled"
        elif not s:
            entry["status"] = "not_shipped"
        elif set(c) == set(s):
            entry["status"] = "identical"
        elif set(c) <= set(s):
            entry["status"] = "subset"
        else:
            st = {e.get("target") for e in s.values()} - {None}
            ct = {e.get("target") for e in c.values()} - {None}
            if st and ct and st != ct:
                entry["status"] = "target"
                entry["detail"] = f"production {sorted(st)} vs verifier {sorted(ct)}: PTX not comparable across architectures"
            elif name in specialised_names and gpu_differs:
                entry["status"] = f"specialised: {GEMM_SPECIALISED}"
                entry["detail"] = "batch-invariant GEMM: the SM count is a compile-time constant of the kernel and the verifier's GPU differs from production's"
            else:
                entry["status"] = "mismatch"
                entry["detail"] = f"{len(set(c) - set(s))} variant(s) compiled here are not among production's {len(s)}"
        per[name] = entry
    summary = Counter(e["status"].split(":")[0] for e in per.values())
    return {"kernels": per, "summary": dict(sorted(summary.items())),
            "ok": all(e["status"] in ("identical", "subset") for e in per.values()) if per else None}


def specialised_kernel_names(sources: dict[str, str]) -> set[str]:
    """Kernel function names whose bundled source is specialised on the SM count (``NUM_SMS``)."""
    out = set()
    for src in sources.values():
        if GEMM_SPECIALISED in src:
            m = _DEF.search(src)
            if m:
                out.add(m.group(1))
    return out


def ptx_identity(run_dir: Path, *, sources: dict[str, str], triton_cache_dir: str | None, manifest: dict[str, Any],
                 env: dict[str, Any], compile_fn: Any, preexisting: set[str] | None = None) -> dict[str, Any]:
    """F10. Compile every bundled kernel source in this process (``compile_fn(sha, fn_name)`` ->
    Inductor's ``CachingAutotuner``, which precompiles every launch config), collect the PTX Triton
    wrote to ``triton_cache_dir`` and compare, per kernel and variant, with the production PTX shipped
    in the bundle. Identity is evidence about the environment (a qualification), never a verdict
    input by itself (§16.1; a source hash does not prove an execution)."""
    kd = Path(run_dir) / "kernels"
    shipped = load_shipped_ptx(kd)
    out: dict[str, Any] = {"available": shipped.get("available", False), "triton_cache_dir": triton_cache_dir, "qualifications": []}
    if not shipped.get("available"):
        out["qualifications"].append("PTX identity not checkable: the bundle ships no production PTX (kernels/ptx-index.json; archive predates F10)")
        return out
    out["shipped_bad_hash"] = shipped["bad_hash"]
    out["shipped_missing"] = shipped["missing"]
    if shipped["bad_hash"] or shipped["missing"]:
        out["problem"] = f"shipped PTX files do not hash to their names / are missing: {shipped['bad_hash'][:3]} {shipped['missing'][:3]}"
    names: dict[str, str] = {}
    for sha, src in sources.items():
        m = _DEF.search(src)
        if m:
            names[sha] = m.group(1)
    t0 = time.perf_counter()
    compile_errors: dict[str, str] = {}
    for sha, fn in sorted(names.items()):
        try:
            compile_fn(sha, fn)
        except Exception as e:  # noqa: BLE001
            compile_errors[f"{fn}:{sha[:12]}"] = f"{type(e).__name__}: {e}"[:300]
    out["compile_seconds"] = time.perf_counter() - t0
    out["compile_errors"] = compile_errors
    compiled = scan_triton_cache(Path(triton_cache_dir), set(names.values())) if triton_cache_dir else {}
    # PTX that was in the cache dir before this process started was compiled elsewhere (a shared cache root):
    # still compared, but not evidence about this process's toolchain — said so.
    pre = preexisting or set()
    stale = 0
    for name, variants in compiled.items():
        for sha_, e in variants.items():
            e["compiled_here"] = e["path"] not in pre
            stale += 0 if e["compiled_here"] else 1
    out["variants_compiled_here"] = sum(1 for v in compiled.values() for e in v.values() if e["compiled_here"])
    out["variants_preexisting_in_cache"] = stale
    m_gpu = manifest.get("gpu") or {}
    g = env.get("gpu") or {}
    gpu_differs = bool(
        (m_gpu.get("name") and g.get("name") and m_gpu["name"] != g["name"])
        or (m_gpu.get("multi_processor_count") and g.get("multi_processor_count") and int(m_gpu["multi_processor_count"]) != int(g["multi_processor_count"]))
    )
    bundled_names = set(names.values())
    shipped_by_name = {n: v for n, v in shipped["by_name"].items() if n in bundled_names}
    out["shipped_not_in_bundle"] = sorted(n for n in shipped["by_name"] if n not in bundled_names)  # production's other Triton kernels (informational)
    cmp = compare_ptx_variants(shipped_by_name, compiled, specialised_names=specialised_kernel_names(sources), gpu_differs=gpu_differs)
    out.update(cmp)
    out["gpu_differs"] = gpu_differs
    out["ptx_code_set_sha256_shipped"] = shipped.get("ptx_code_set_sha256")
    recorded = ((manifest.get("compilation") or {}).get("generated_kernels") or {}).get("ptx_code_set_sha256")
    out["ptx_code_set_sha256_manifest"] = recorded
    if recorded and shipped.get("ptx_code_set_sha256") and recorded != shipped["ptx_code_set_sha256"]:
        out["problem"] = f"shipped PTX set {shipped['ptx_code_set_sha256'][:12]}… != manifest's ptx_code_set_sha256 {recorded[:12]}…"
    for name, e in cmp["kernels"].items():
        st = e["status"]
        if st in ("identical", "subset"):
            continue
        out["qualifications"].append(f"PTX {name}: {st}" + (f" ({e['detail']})" if e.get("detail") else ""))
    if compile_errors:
        out["qualifications"].append(f"PTX: {len(compile_errors)} bundled kernel source(s) failed to compile here: {sorted(compile_errors)[:3]}")
    if stale:
        out["qualifications"].append(f"PTX: {stale} of {stale + out['variants_compiled_here']} compared variants were already in the Triton cache dir "
                                     f"before this process (shared cache root): not this process's compile")
    return out


# --------------------------------------------------------------------------------------- F9 ---


class StructuralFailure(Exception):
    """WB2 (audit-security-baseline finding 4): a structural check of the pipeline that did not pass and
    stops verification -- the description does not validate, a committed document is malformed, a tree
    cannot be built.  ``verify.run`` turns it into the Table 1 row ``FAIL structural: <check>: <detail>``
    (no relation counts claimed) plus a RESULT line; never a traceback alone."""

    def __init__(self, check: str, detail: str) -> None:
        super().__init__(f"{check}: {detail}")
        self.check = check
        self.detail = detail


# exception type -> the structural check it belongs to (for exceptions the pipeline did not wrap itself)
_STRUCTURAL_CHECKS: tuple[tuple[str, str], ...] = (
    ("DescriptionError", "description validation"),
    ("CommitError", "commitment"),
    ("CaptureError", "capture format"),
    ("FirstStageError", "first-stage completeness"),
    ("IdentityError", "canonical id"),
    ("ChallengeError", "challenge"),
    ("RowError", "row schema"),
    ("DtypeError", "dtype"),
    ("JSONDecodeError", "malformed JSON document"),
    ("FileNotFoundError", "missing file"),
    ("KeyError", "missing document field"),
    ("ImportError", "verifier environment"),
    ("ModuleNotFoundError", "verifier environment"),
)


def structural_reason(exc: BaseException) -> str:
    """``<check>: <detail>`` for any exception that escaped the pipeline: the check is named from the
    exception's type (:data:`_STRUCTURAL_CHECKS`), else ``verifier exception``."""
    if isinstance(exc, StructuralFailure):
        return f"{exc.check}: {exc.detail}"
    name = type(exc).__name__
    check = next((c for n, c in _STRUCTURAL_CHECKS if n == name), "verifier exception")
    return f"{check}: {name}: {exc}"


def refusal_row(out_dir: Path, *, case: str, profile_id: str, worker: str, adapter_revision: str, reason: str,
                exception: str | None = None, result: str = "refused") -> dict[str, Any]:
    """F9. Write the Table 1 row for a bundle the verifier refuses before checking anything: every
    count 0, ``structural.failing_invariant = reason`` -> ``FAIL structural: …`` by ``rows.verdict``.
    Also writes ``verify.json`` with the exception text and prints the RESULT line.  ``result`` is the
    ``structural.result`` word: ``refused`` (the bundle was not checked) or ``failed`` (a structural
    check inside the pipeline did not pass, WB2)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    utc = time.strftime("%Y%m%dT%H%MZ", time.gmtime())
    exp_id = rows.experiment_id(case, profile_id, utc)
    common = rows.Common(exp_id, case, profile_id, worker, adapter_revision)
    zero = {"exact": 0, "attempted": 0}
    t1 = rows.table1_row(
        common,
        workload={"requests": 0, "generated_positions": 0, "input_len_range": [0, 0], "peak_concurrency_achieved": 0},
        production_changes=("not assessed: bundle refused before checking" if result == "refused"
                            else "not assessed: a structural check failed before the checks ran"),
        replay_restrictions=[],
        token_selection=zero,
        boundary_noninterference={"identical": False, "compared": 0, "mismatched_scalars": 0, "reference_method": "not run"},
        required_coverage={"bound": 0, "required": 0},
        authentication={"valid": 0, "attempted": 0},
        relation_checks={"same_runtime": dict(zero), "independent": dict(zero)},
        structural={"result": result, "failing_invariant": reason},
        negative={"rejected": 0, "attempted": 0},
    )
    rows.write_jsonl(out / "table1.jsonl", [t1])
    (out / "verify.json").write_text(json.dumps({"experiment_id": exp_id, "case": case, "profile_id": profile_id, "refused": True,
                                                  "structural": t1["structural"], "reason": reason, "exception": exception,
                                                  "table1": t1, "utc": utc}, indent=1, default=str) + "\n")
    print(f"RESULT {t1['result']} first_blocker={t1['first_blocker']} refused=True structural={t1['structural']}", flush=True)
    return t1


def fused_limitation(node: Any, launch_used: dict[str, Any] | None) -> str | None:
    """Table 2 ``limitation`` for a node checked by the capture-level fused-kernel relation: its
    row-independence assumption and which autotune config the standalone launch used (F8/F9)."""
    if node.relation == "unresolved":
        return "no executable relation"
    if not str(node.relation).startswith("inductor.fused."):
        return None
    base = ("capture-level relation (not a Verity primitive / VU): assumes the generated kernel is row-independent "
            "(1-row standalone launch vs the row captured inside production's batch)")
    used = (launch_used or {}).get(node.op)
    if used is None:
        return base
    src = used.get("source")
    if src == "production":
        return base + "; launched with production's recorded autotune best_config"
    if src == "single-config":
        return base + "; single launch config (no autotune choice)"
    if src == "replay":
        return base + "; launched with the REPLAY's autotune choice: production's best_config is not recorded for this kernel"
    return base + f"; launch config source: {src}"
