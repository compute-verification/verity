"""Independent black-box tests of the artifact applicability contract.

Written from the contract BEFORE reading the implementation. The contract:

  1. one authoritative target/profile input rather than duplicated literals;
  2. the reported manifest describes the facts actually used during construction;
  3. construction versions, target/profile facts and applicability conditions are bound into the artifact/cache identity;
  4. use of an artifact outside its declared applicability is rejected;
  5. the Program digest is NOT required to change when the circuit is genuinely unchanged (an identical digest across targets is
     not the bug; the bug is that construction assumptions, target declaration and cache identity can DISAGREE).

Every "dynamic" test here drives the real builder (`verity_vllm.pipeline.build`, B0 `request 8/4`, GPU-free) in a
subprocess over a byte-copy of this tree with at most a one-fact mutation, and asserts on the ARTEFACTS (`result.json`,
`construction_manifest.json`, any artifact/applicability file the build writes). Nothing here imports the implementation; the only
implementation-specific knowledge allowed is the ADAPTER block, and every adapter entry has a generic fallback so the tests run
unchanged on a base without the contract (where the contract tests are EXPECTED to fail).

Run (pod, venv with the pinned vLLM, SmolLM2 snapshot under $HF_HOME):
    CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 HF_HOME=/workspace/hf PYTHONPATH=. CBTEST_OUT=/workspace/cp/cbtest/tests \
      python -m pytest tests/program/test_artifact_applicability_independent.py -q -rA
Env: CBTEST_OUT (build outputs; default a tmp dir), CBTEST_REUSE=1 (reuse finished builds under CBTEST_OUT), CBTEST_JOBS (parallel
builds, default 6), CBTEST_GPU_HOST=1 (also run the GPU-visible host-mismatch build), CBTEST_SKIP_BUILDS=1 (static tests only).
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
CB_A = tuple(REPO / f for f in ("verity_vllm/check/match/global_match.py", "verity_vllm/check/match/global_match_fast.py", "verity_vllm/check/match/program_compare.py", "verity_vllm/check/match/batch_decomp.py", "verity_vllm/pipeline/build.py", "verity_vllm/program/compact.py", "verity_vllm/pipeline/global_program.py", "verity_vllm/program/instances_form.py", "verity_vllm/program/frontend/export_ops.py"))   # cb_a dissolved: its modules outside frontend/
FRONTEND = REPO / "verity_vllm" / "program" / "frontend"
#: frontend modules that are not its rules: the registered target records and the GEMM targets name every GPU's arch (tables,
#: not the declared input)
NOT_RULES = (FRONTEND / "targets.py", FRONTEND / "gemm_targets.py")
#: the baseline builds (`request_8_4` on a 4090 with its GPU visible, `sens/base` without a GPU, `repro_base` on an L40S with it hidden)
B0_REQUEST_8_4_DIGEST = "59b35d80faf29438d21a0ea1c96c877ec69e2f21174f11b835ac1ef04c1ed496"
R9_REFERENCE_MANIFESTS = [  # other construction hosts, same profile (pre-fix manifests: Program digest only)
    REPO / "out/gen/r9/cb-c/validation/sens/base/construction_manifest.json",                 # no GPU, cuda_available false
    REPO / "out/gen/r9/cb-a/request_8_4/construction_manifest.json",                          # 4090, cuda_available true
    REPO / "out/gen/r10/cb-test/repro_base/sens/base/construction_manifest.json",             # L40S, GPU hidden
]
SOURCE_ROOTS = ("verity.ml.circuits", "verity_vllm", "EXPORT.json", "pyproject.toml")  # what a build needs (the driver imports only these)
#: core `verity` is not copied into the build trees (only the integration is mutated); the builds import it from the repo
CORE_SRC = REPO.parents[1]


def _pythonpath(tree: pathlib.Path) -> str:
    return os.pathsep.join([str(tree), str(CORE_SRC)])

# ---------------------------------------------------------------------------------------------------------------------------------
# ADAPTER — the ONLY place allowed to know the implementation's names. Every entry has a generic fallback.
# ---------------------------------------------------------------------------------------------------------------------------------
ADAPTER: dict = {
    # regex on manifest KEY names whose subtree is the artifact/cache identity (fallback: generic name search, see identity_of)
    "identity_key_re": r"^(artifact|cache)([_-]?(id|identity|key|digest|hash|sha256))?$|^identity$|applicability[_-]?(id|key|digest)",
    # regex on manifest KEY names under which the declared/applied target facts live (fallback: any key matching /capab|num_sms/)
    "target_key_re": r"capab|num_sms|sm_count",
    # files the fix may write next to result.json that carry the identity (globbed; hashed as a fallback identity source)
    # `artifact.json` next to `descriptor.json.gz` (identity + construction_version, target_profile, applicability, program.digest)
    "identity_file_globs": ("artifact*.json", "applicab*.json", "cache_key*", "identity*.json", "*.artifact.json"),
    # reuse check (contract bullet 4): callable(artifact_dir: Path, mismatched_profile: dict, tree: Path, python: str) -> (rejected: bool|None, reason: str)
    # the implementation's names: `verity_vllm.program.frontend.target_profile.TargetProfile(compute_capability=(8, 9), num_sms=128, …)`,
    # `load_artifact(dir, profile)` / `check_reuse(artifact, profile, applicability_now)`; module of the latter two unstated → probed.
    "reuse_check": None,  # None → reuse_check_cbfix (the implementation's names) then reuse_check_generic
    "target_profile_module": "verity_vllm.program.frontend.target_profile",
    "reuse_api_modules": (
        "verity_vllm.program.frontend.target_profile",
        "verity_vllm.check.artifact",
        "verity_vllm.check.artifact_identity",
        "verity_vllm.check.applicability",
        "verity_vllm.pipeline.build",
        "verity_vllm.program.frontend.export_compat",
        "verity_vllm.program.frontend.vllm_meta",
    ),
    # module[:callable] names to probe generically for a reuse/applicability check (each tried in a subprocess on the pod venv)
    "reuse_probe_candidates": (
        "verity_vllm.check.artifact:check_applicability",
        "verity_vllm.check.artifact:load_artifact",
        "verity_vllm.check.applicability:check",
        "verity_vllm.pipeline.build:check_applicability",
    ),
}

# ---------------------------------------------------------------------------------------------------------------------------------
# Static contract tests (run anywhere, no torch): one authoritative target input, no duplicated literals
# ---------------------------------------------------------------------------------------------------------------------------------
CAP_LITERAL_PATTERNS = [
    re.compile(r"major\s*=\s*8\s*,\s*minor\s*=\s*9"),        # DeviceCapability(major=8, minor=9)
    re.compile(r"DeviceCapability\(\s*8\s*,\s*9\s*\)"),
    re.compile(r"8\.9\s*\(sm_89"),                            # the manifest literal "8.9 (sm_89, Ada)"
    re.compile(r"\bsm_89\b"),
    re.compile(r"(?<![\d.])8\.9(?![\d.])"),                   # a bare 8.9 (not 0.28.9, not 8.9.1)
    re.compile(r"\(\s*8\s*,\s*9\s*\)"),                       # a (major, minor) tuple literal, e.g. compute_capability=(8, 9)
]
NUM_SMS_LITERAL_PATTERNS = [re.compile(r"[\"']NUM_SMS[\"']\s*:\s*128\b"), re.compile(r"\bNUM_SMS\s*=\s*128\b"),
                            re.compile(r"\bnum_sms\s*(?::\s*int)?\s*=\s*128\b")]


def _source_files() -> list[pathlib.Path]:
    files = []
    for root in (*CB_A, FRONTEND):
        for p in ([root] if root.is_file() else sorted(root.rglob("*.py"))):
            if "tests" in p.parts or "__pycache__" in p.parts or p in NOT_RULES:
                continue
            files.append(p)
    return files


def _strip_comments_and_docstrings(src: str) -> str:
    """Code-only view: literals in comments/docstrings are documentation, not duplicated construction inputs."""
    import io
    import tokenize
    out = []
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenError, SyntaxError):
        return src
    prev_type = None
    for t in toks:
        if t.type == tokenize.COMMENT:
            continue
        if t.type == tokenize.STRING and prev_type in (None, tokenize.NEWLINE, tokenize.INDENT, tokenize.NL, tokenize.DEDENT):
            prev_type = t.type
            continue  # expression-statement string = docstring
        if t.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENCODING, tokenize.ENDMARKER):
            out.append(t.string)
        prev_type = t.type
    return " ".join(out)


def literal_sites(patterns) -> dict[str, list[str]]:
    """{relative file: [matched literals]} over the CODE (comments/docstrings stripped) of the builder + the frontend rules."""
    hits: dict[str, list[str]] = {}
    for p in _source_files():
        code = _strip_comments_and_docstrings(p.read_text())
        found = [m.group(0) for pat in patterns for m in pat.finditer(code)]
        if found:
            hits[str(p.relative_to(REPO))] = found
    return hits


def test_static_capability_literal_has_one_authoritative_site():
    """Contract bullet 1. The declared compute capability (8.9 / sm_89) must be spelled in ONE module. On a base without the contract it is spelled
    twice: `cb_a/export_compat.py:71` (applied by the shim) and `frontend/rules/vllm_bindings.py:206` (written into the manifest) —
    the two copies are what let the manifest go stale (a1)."""
    sites = literal_sites(CAP_LITERAL_PATTERNS)
    report = json.dumps(sites, indent=1)
    print("\n[cb-test] capability literal sites (code only):\n" + report)
    assert len(sites) <= 1, f"compute-capability literal duplicated in {len(sites)} modules (contract: one authoritative input):\n{report}"


def test_static_num_sms_pin_has_one_authoritative_site():
    """Contract bullet 1 for the SM count: `NUM_SMS = 128` spelled once (a base without the contract: only `GEMM_PINS` — the shim reads it through
    `declared_num_sms()`; kept as a regression guard so the fix does not introduce a second copy)."""
    sites = literal_sites(NUM_SMS_LITERAL_PATTERNS)
    print("\n[cb-test] NUM_SMS literal sites (code only):\n" + json.dumps(sites, indent=1))
    assert len(sites) <= 1, f"NUM_SMS pin literal duplicated in {len(sites)} modules:\n{json.dumps(sites, indent=1)}"


def test_static_manifest_declared_capability_is_not_a_string_literal():
    """Contract bullet 2, statically: the manifest's `compute capability` entry must be computed from the input the shim applies, not
    a quoted string sitting next to `"field": "compute capability"`."""
    bad = []
    for p in _source_files():
        code = _strip_comments_and_docstrings(p.read_text())
        for m in re.finditer(r"compute capability[\"']\s*,\s*[\"']value[\"']\s*:\s*([\"'][^\"']*[\"'])", code):
            bad.append((str(p.relative_to(REPO)), m.group(1)))
    assert not bad, f"manifest 'compute capability' value is a string literal (stale-manifest bug, a1): {bad}"


def test_static_reference_hosts_share_the_program_digest():
    """(iii)/(v) on the recorded artefacts (laptop-runnable): every construction host that built B0 request 8/4 for the sm_89
    profile — a 4090 with its GPU visible, a host without a GPU, an L40S with it hidden — recorded the same Program digest."""
    seen = []
    for p in R9_REFERENCE_MANIFESTS:
        if p.exists():
            m = json.load(open(p))
            seen.append((str(p.relative_to(REPO)), (m.get("program") or {}).get("digest"), (m.get("construction") or {}).get("cuda_available")))
    if not seen:
        pytest.skip("reference manifests not in this checkout (sparse tree without out/gen)")
    assert all(d == B0_REQUEST_8_4_DIGEST for _p, d, _c in seen), seen
    print(f"\n[cb-test] reference hosts: {seen}")


# ---------------------------------------------------------------------------------------------------------------------------------
# Build harness: the driver in a subprocess over a mutated byte-copy of the tree, GPU-free
# ---------------------------------------------------------------------------------------------------------------------------------
def _hf_snapshot_present() -> bool:
    home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    snap = pathlib.Path(home) / "hub/models--HuggingFaceTB--SmolLM2-135M/snapshots/93efa2f097d58c2a74874c7e644dbc9b0cee75a2/config.json"
    return snap.exists()


def _can_build() -> tuple[bool, str]:
    if os.environ.get("CBTEST_SKIP_BUILDS") == "1":
        return False, "CBTEST_SKIP_BUILDS=1"
    try:
        import vllm  # noqa: F401
        import torch  # noqa: F401
    except Exception as e:  # noqa: BLE001
        return False, f"vllm/torch not importable here ({type(e).__name__}); run on the pod venv"
    if not _hf_snapshot_present():
        return False, "SmolLM2-135M @ 93efa2f0 snapshot not under HF_HOME"
    return True, ""


CAN_BUILD, WHY_NOT = _can_build()
needs_builder = pytest.mark.skipif(not CAN_BUILD, reason=f"builder not runnable here: {WHY_NOT}")


@dataclass
class Build:
    name: str
    tree: pathlib.Path
    out: pathlib.Path
    rc: int = -1
    log: str = ""
    result: dict = field(default_factory=dict)
    manifest: dict = field(default_factory=dict)
    mutations: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.result.get("ok"))

    @property
    def digest(self) -> str | None:
        return (self.result.get("program") or {}).get("digest")

    @property
    def refusal(self) -> str:
        u = self.result.get("unsupported") or {}
        return " ".join(str(u.get(k, "")) for k in ("reason", "node", "hint")) + " " + str(self.result.get("message", ""))

    def files(self) -> list[str]:
        return sorted(p.name for p in self.out.iterdir()) if self.out.exists() else []


class Harness:
    def __init__(self, work: pathlib.Path, reuse: bool, jobs: int):
        self.work, self.reuse, self.jobs = work, reuse, jobs
        self.python = sys.executable
        self.env_base = {k: v for k, v in os.environ.items() if not k.startswith("CBTEST_")}
        self.env_base.update({"CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "VLLM_BATCH_INVARIANT": "1"})
        self.env_base.setdefault("HF_HOME", "/workspace/hf")
        self._lock = threading.Lock()
        self._builds: dict[str, Build] = {}

    # -- tree copies -------------------------------------------------------------------------------------------------------------
    def copy_tree(self, name: str) -> pathlib.Path:
        dst = self.work / name / "tree"
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True)
        for r in SOURCE_ROOTS:
            src = REPO / r
            if not src.exists():
                continue
            if src.is_dir():
                shutil.copytree(src, dst / r, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "tests"))
            else:
                shutil.copy2(src, dst / r)
        return dst

    # -- one build ---------------------------------------------------------------------------------------------------------------
    def build(self, name: str, mutate=None, env: dict | None = None, preamble: str = "", gpu_visible: bool = False) -> Build:
        with self._lock:
            if name in self._builds:
                return self._builds[name]
        out = self.work / name / "out"
        b = Build(name=name, tree=self.work / name / "tree", out=out)
        if self.reuse and (out / "result.json").exists() and (out / "construction_manifest.json").exists():
            self._load(b)
            with self._lock:
                self._builds[name] = b
            return b
        tree = self.copy_tree(name)
        if mutate is not None:
            b.mutations = mutate(tree) or []
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        e = dict(self.env_base)
        e["PYTHONPATH"] = _pythonpath(tree)
        if env:
            e.update(env)
        if gpu_visible:
            e.pop("CUDA_VISIBLE_DEVICES", None)
        argv = ["verity_vllm.pipeline.cli", "build", "--out", str(out), "--wrapper", "request", "--LP", "8", "--T", "4"]
        code = (preamble + "\nimport runpy, sys\nsys.argv = " + repr(argv) + "\nrunpy.run_module(sys.argv[0], run_name='__main__')\n")
        (self.work / name / "driver.py").write_text(code)
        p = subprocess.run([self.python, str(self.work / name / "driver.py")], cwd=str(tree), env=e, capture_output=True, text=True)
        b.rc = p.returncode
        b.log = (p.stdout + "\n" + p.stderr)
        (self.work / name / "build.log").write_text(b.log)
        (self.work / name / "mutations.json").write_text(json.dumps(b.mutations, indent=1, default=str))
        self._load(b)
        with self._lock:
            self._builds[name] = b
        return b

    def _load(self, b: Build) -> None:
        for attr, fn in (("result", "result.json"), ("manifest", "construction_manifest.json")):
            try:
                setattr(b, attr, json.load(open(b.out / fn)))
            except Exception:  # noqa: BLE001
                setattr(b, attr, {})
        if b.rc == -1:  # reused build: recover rc / mutations / log from the files the first run left behind
            b.rc = 0 if b.result.get("ok") else 1
            try:
                b.mutations = json.load(open(self.work / b.name / "mutations.json"))
                b.log = (self.work / b.name / "build.log").read_text()
            except Exception:  # noqa: BLE001
                pass

    def build_many(self, specs: list[tuple]) -> dict[str, Build]:
        with ThreadPoolExecutor(max_workers=self.jobs) as ex:
            futs = {spec[0]: ex.submit(self.build, *spec[:1], **spec[1]) for spec in specs}
            return {k: f.result() for k, f in futs.items()}


# -- mutators (one-fact edits, expressed as regex rewrites so they apply to any layout of the same literals) ----
def _rewrite(tree: pathlib.Path, subs: list[tuple[re.Pattern, str]], only: str | None = None) -> list[dict]:
    done = []
    for p in sorted(tree.rglob("*.py")):
        rel = str(p.relative_to(tree))
        if "tests" in p.parts or not (rel.startswith(("verity_vllm/check/match/global_match.py", "verity_vllm/check/match/global_match_fast.py", "verity_vllm/check/match/program_compare.py", "verity_vllm/check/match/batch_decomp.py", "verity_vllm/pipeline/build.py", "verity_vllm/program/compact.py", "verity_vllm/program/frontend/export_compat.py", "verity_vllm/program/frontend/inputs_trace.py", "verity_vllm/program/frontend/triton_capture.py", "verity_vllm/program/frontend/vllm_meta.py", "verity_vllm/pipeline/global_program.py", "verity_vllm/program/instances_form.py", "verity_vllm/program/frontend/export_ops.py")) or rel.startswith("verity.ir/frontend")):
            continue
        if only is not None and rel != only:
            continue
        src = p.read_text()
        new, n_total = src, 0
        for pat, rep in subs:
            new, n = pat.subn(rep, new)
            n_total += n
        if n_total:
            p.write_text(new)
            done.append({"file": rel, "n_subs": n_total})
    return done


def capability_subs(major: int, minor: int) -> list[tuple[re.Pattern, str]]:
    name = {(9, 0): "Hopper", (8, 0): "Ampere", (8, 9): "Ada"}.get((major, minor), "GPU")
    return [
        (re.compile(r"major\s*=\s*8\s*,\s*minor\s*=\s*9"), f"major={major}, minor={minor}"),
        (re.compile(r"DeviceCapability\(\s*8\s*,\s*9\s*\)"), f"DeviceCapability({major}, {minor})"),
        (re.compile(r"8\.9\s*\(sm_89,\s*Ada\)"), f"{major}.{minor} (sm_{major}{minor}, {name})"),
        (re.compile(r"\bsm_89\b"), f"sm_{major}{minor}"),
        (re.compile(r"(?<![\d.])8\.9(?![\d.])"), f"{major}.{minor}"),
        (re.compile(r"\(\s*8\s*,\s*9\s*\)"), f"({major}, {minor})"),
    ]


def num_sms_subs(n: int) -> list[tuple[re.Pattern, str]]:
    return [(re.compile(r"([\"']NUM_SMS[\"']\s*:\s*)128\b"), rf"\g<1>{n}"), (re.compile(r"(\bNUM_SMS\s*=\s*)128\b"), rf"\g<1>{n}"),
            (re.compile(r"(\bnum_sms\s*(?::\s*int)?\s*=\s*)128\b"), rf"\g<1>{n}")]


def mutate_capability_all(major: int, minor: int):
    return lambda tree: _rewrite(tree, capability_subs(major, minor))


def mutate_capability_site(site: str, major: int, minor: int):
    return lambda tree: _rewrite(tree, capability_subs(major, minor), only=site)


def mutate_target(major: int, minor: int, num_sms: int | None):
    def f(tree):
        done = _rewrite(tree, capability_subs(major, minor)) if (major, minor) != (8, 9) else []
        if num_sms is not None:
            done += _rewrite(tree, num_sms_subs(num_sms))
        return done
    return f


def host_capability_preamble(major: int, minor: int) -> str:
    """The HOST answers capability (major, minor) everywhere (instantiation + export): the a2/a3 'consistent' declaration, applied
    as an environmental fact of the build process rather than a tree edit (so it composes with any shim the fix installs)."""
    return (
        "import verity_vllm.program.frontend.vllm_meta  # the driver's env defaults (VLLM_USE_LAYERNAME=0, …) before vLLM reads them, as in a2\n"
        "from vllm.platforms import current_platform as _cbt_cp\n"
        "from vllm.platforms.interface import DeviceCapability as _cbt_DC\n"
        f"type(_cbt_cp).get_device_capability = classmethod(lambda cls, device_id=0: _cbt_DC(major={major}, minor={minor}))\n"
    )


# -- manifest readers (schema-agnostic: the fix may rename/move fields; we search by meaning) ----------------------------------------
def walk(obj, path=()):
    """(path, leaf) over a JSON tree; a `{"field": F, "value": V}` row (the `declared[]`/`derived[]` form) is yielded as path+(F,) -> V."""
    if isinstance(obj, dict):
        if "field" in obj and "value" in obj and isinstance(obj["field"], str):
            yield from walk(obj["value"], path + (obj["field"],))
            return
        for k, v in obj.items():
            yield from walk(v, path + (str(k),))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from walk(v, path + (str(i),))
    else:
        yield path, obj


EXCLUDED_PATH_TOKENS = ("unsupported", "hint", "bindings_rules", "why", "inputs_trace", "files", "construction", "rules_applied", "spec_histogram", "hf_config", "derived",
                        "host", "observed", "env_ignored", "ignored")  # host/observed facts are not declarations


def _norm_capability(v):
    if v is None:
        return None
    if isinstance(v, dict) and "major" in v and "minor" in v:
        return f"{int(v['major'])}.{int(v['minor'])}"
    if isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(x, int) for x in v):
        return f"{v[0]}.{v[1]}"
    if isinstance(v, int) and 10 <= v <= 129:
        return f"{v // 10}.{v % 10}"
    if isinstance(v, str):
        m = re.search(r"(?<![\d.])(\d{1,2})\.(\d)(?![\d.])", v) or re.search(r"sm_?(\d)(\d)\b", v)
        if m:
            return f"{int(m.group(1))}.{int(m.group(2))}"
    return None


def declared_capabilities(m: dict) -> set[str]:
    """Normalised 'major.minor' of every manifest entry whose key names the compute capability (excluding refusal texts, docs, host
    observations)."""
    found = set()
    for path, v in walk(m):
        if any(tok in path for tok in EXCLUDED_PATH_TOKENS):
            continue
        if not any(re.search(r"capab", k, re.I) for k in path):
            continue
        n = _norm_capability(v)
        if n:
            found.add(n)
    return found


def declared_num_sms(m: dict) -> set[int]:
    found = set()
    for path, v in walk(m):
        if any(tok in path for tok in EXCLUDED_PATH_TOKENS):
            continue
        if not any(re.search(r"num_?sms|sm_count", k, re.I) for k in path):
            continue
        if isinstance(v, int) and 1 <= v <= 1024:
            found.add(v)
        elif isinstance(v, str) and v.isdigit():
            found.add(int(v))
    return found


def declared_paths_with_key(m: dict, key: str) -> list[tuple]:
    """Paths of `key` that sit under a 'declared…'/'applied…'/'target'/'profile facts' section (what the manifest ASSERTS the build used)."""
    out = []
    for path, _v in walk(m):
        if key in path and any(re.search(r"declared|applied|target|profile_facts|applicab|consulted", k, re.I) for k in path) \
                and not any(re.search(r"ignored|inert|not_consulted|unused|host", k, re.I) for k in path):
            out.append(path)
    return out


def identity_of(b: Build):
    """The artifact/cache identity the fix binds (contract bullet 3): the manifest subtree(s) under an identity-named key, else the
    content of any identity/applicability file written beside `result.json`. None when the artefact carries no identity (a base without the contract)."""
    key_re = re.compile(ADAPTER["identity_key_re"], re.I)
    found = {}
    for path, v in walk(b.manifest):
        if any(tok in path for tok in ("files", "inputs_trace", "unsupported", "bindings_rules")):
            continue
        if any(key_re.search(k) for k in path):
            found["/".join(path)] = v
    if found:
        return found
    for g in ADAPTER["identity_file_globs"]:
        for p in sorted(b.out.glob(g)):
            try:
                found[p.name] = json.loads(p.read_text())
            except Exception:  # noqa: BLE001
                found[p.name] = p.read_text()
    return found or None


def identity_targets(ident) -> dict:
    """The target/profile-fact part of an identity (so a code-version component cannot mask an unchanged target)."""
    t = {}
    if ident is None:
        return t
    for path, v in walk(ident):
        if any(re.search(ADAPTER["target_key_re"], k, re.I) for k in path):
            t["/".join(path)] = v
    return t


# ---------------------------------------------------------------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def harness(tmp_path_factory) -> Harness:
    work = pathlib.Path(os.environ.get("CBTEST_OUT") or tmp_path_factory.mktemp("cbtest"))
    work.mkdir(parents=True, exist_ok=True)
    return Harness(work, reuse=os.environ.get("CBTEST_REUSE") == "1", jobs=int(os.environ.get("CBTEST_JOBS", "6")))


CAP_SITES = sorted(literal_sites(CAP_LITERAL_PATTERNS)) or ["<no-capability-literal-site>"]


@pytest.fixture(scope="session")
def builds(harness: Harness) -> dict[str, Build]:
    """All dynamic builds, in parallel (each ≈ 25–60 s GPU-free): the mutation matrix plus the host/env cases of this contract."""
    if not CAN_BUILD:
        pytest.skip(WHY_NOT)
    specs = [
        ("base", {}),
        ("hostB", {"env": {"HOSTNAME": "cbtest-host-B"}}),  # (iii) another construction host, same profile (other hostname/out/tree paths/utc)
        ("cap90_consistent", {"mutate": mutate_capability_all(9, 0), "preamble": host_capability_preamble(9, 0)}),  # (ii-a) a2
        ("cap80_sms108_consistent", {"mutate": mutate_target(8, 0, 108), "preamble": host_capability_preamble(8, 0)}),  # (ii-b) a3
        ("sms132_pin", {"mutate": mutate_target(8, 9, 132)}),  # (ii-c) b2: NUM_SMS re-pinned consistently
        ("attn_env_dead", {"env": {"VLLM_ATTENTION_BACKEND": "TRITON_ATTN"}}),  # (vi) d1: inert env var
        ("host_says_90_profile_89", {"preamble": host_capability_preamble(9, 0)}),  # (i-d) host platform answers 9.0, declared profile 8.9
    ]
    for s in CAP_SITES:  # (i-b) mutate ONE capability literal site at a time (export-time only, as the shim applies it)
        if s.startswith("<"):
            continue
        specs.append((f"cap90_site__{s.replace('/', '__')}", {"mutate": mutate_capability_site(s, 9, 0)}))
    if os.environ.get("CBTEST_GPU_HOST") == "1":
        specs.append(("gpu_host_mismatch", {"gpu_visible": True}))  # (vii) construct on the L40S (142 SMs) for the 4090 profile (128)
    res = harness.build_many(specs)
    summary = {k: {"rc": b.rc, "ok": b.ok, "digest": b.digest, "mutations": b.mutations, "files": b.files(),
                   "declared_capabilities": sorted(declared_capabilities(b.manifest)), "declared_num_sms": sorted(declared_num_sms(b.manifest)),
                   "identity": identity_of(b), "refusal": (b.refusal.strip()[:600] if not b.ok else None)} for k, b in res.items()}
    (harness.work / "SUMMARY.json").write_text(json.dumps(summary, indent=1, default=str))
    return res


# ---------------------------------------------------------------------------------------------------------------------------------
# (v) the Program digest of B0 request 8/4 is unchanged by the patch (identical digests across targets are not the bug)
# ---------------------------------------------------------------------------------------------------------------------------------
@needs_builder
def test_v_program_digest_unchanged_by_the_patch(builds):
    b = builds["base"]
    assert b.ok, f"base build refused/crashed (rc={b.rc}): {b.refusal[:800]}\n{b.log[-2000:]}"
    assert b.digest == B0_REQUEST_8_4_DIGEST, f"B0 request 8/4 digest changed: {b.digest} != {B0_REQUEST_8_4_DIGEST}"
    refs = []
    for p in R9_REFERENCE_MANIFESTS:
        if p.exists():
            d = (json.load(open(p)).get("program") or {}).get("digest")
            refs.append((str(p.relative_to(REPO)), d))
            assert d == b.digest, f"reference host manifest {p} has digest {d} != {b.digest}"
    print(f"\n[cb-test] digest {b.digest} == {len(refs)} reference host manifests: {refs}")


# ---------------------------------------------------------------------------------------------------------------------------------
# (i) the inconsistent-capability case
# ---------------------------------------------------------------------------------------------------------------------------------
@needs_builder
def test_i_a_base_manifest_declares_the_capability_the_shim_applied(builds):
    """Contract bullet 2 on the unmutated build: the manifest names exactly one declared capability and it is the profile's 8.9."""
    b = builds["base"]
    caps = declared_capabilities(b.manifest)
    assert caps == {"8.9"}, f"base manifest declared capabilities {caps} (expected exactly {{'8.9'}})"
    assert declared_num_sms(b.manifest) == {128}, f"base manifest declared NUM_SMS {declared_num_sms(b.manifest)}"


@needs_builder
@pytest.mark.parametrize("site", CAP_SITES)
def test_i_b_mutating_the_applied_capability_changes_the_manifest(builds, site):
    """Contract bullets 1–2 (mutation test, a1): change the capability literal at ONE site (export-time only, as the shim applies
    it) → the manifest's declared capability must follow (9.0), or the build must refuse naming the capability. On a base without the contract the
    `export_compat.py` site is applied but the manifest keeps saying "8.9 (sm_89, Ada)" — the stale-manifest FAIL."""
    if site.startswith("<"):
        pytest.skip("no capability literal site left in the code — the single input is not a literal; covered by the consistent-target tests")
    b = builds[f"cap90_site__{site.replace('/', '__')}"]
    assert b.mutations, f"mutation of {site} touched nothing: {b.mutations}"
    caps = declared_capabilities(b.manifest)
    if not b.ok:
        assert re.search(r"capab|sm_9|9\.0|flash_attn_version", b.refusal, re.I), f"refused for an unrelated reason: {b.refusal[:600]}"
        return
    assert caps == {"9.0"}, (f"site {site} mutated to 9.0 (subs={b.mutations}); build ok, digest={b.digest}, but the manifest declares {caps} "
                             f"— the declaration is not the applied value (stale manifest, a1)")


@needs_builder
def test_i_c_artifact_identity_distinguishes_targets_even_when_digests_are_equal(builds):
    """Contract bullet 3: an 8.9 construction and a construction for another target must not share an artifact/cache identity, even
    when the Program digest is identical (a3/b2: digest 59b35d80… for 8.9/128, 8.0/108 and 8.9/132 alike)."""
    base = builds["base"]
    ib = identity_of(base)
    assert ib is not None, f"base artefact carries NO artifact/cache identity (files: {base.files()}; manifest keys: {sorted(base.manifest)})"
    others = [builds[k] for k in ("cap80_sms108_consistent", "sms132_pin", "cap90_consistent") if builds[k].ok]
    assert others, "every changed-target build was refused; identity distinctness cannot be exercised (see test_ii_*)"
    for o in others:
        io_ = identity_of(o)
        assert io_ is not None, f"{o.name}: no artifact identity (files: {o.files()})"
        assert io_ != ib, f"{o.name}: same artifact identity as base although the declared target differs: {io_}"
        if identity_targets(ib) or identity_targets(io_):  # structured identity: the TARGET component itself must differ (not only a code hash)
            assert identity_targets(io_) != identity_targets(ib), (f"{o.name}: identity differs but its TARGET component is identical to base's "
                                                                  f"({identity_targets(io_)}) — the difference is not the target")
        else:
            print(f"\n[cb-test] {o.name}: identity is opaque (no target-named component); distinctness shown, attribution to the target not separable")
        if o.digest == base.digest:
            print(f"\n[cb-test] {o.name}: Program digest identical to base ({o.digest}) — allowed by the ruling — identity differs: OK")


@needs_builder
def test_i_d_host_answering_another_capability_than_declared(builds):
    """The inconsistent case from the HOST side: the platform answers 9.0 (instantiation + export) while the declared profile is 8.9.
    Contract: never a silently built 8.9-labelled artefact of a 9.0 construction. Either the declared target is applied over the host's
    answer everywhere (then manifest declared == applied == 8.9, FA2 observed, identity == base's) or the build refuses naming the
    capability / FA version with both values. On a base without the contract this is a2 (refused on FA3)."""
    b, base = builds["host_says_90_profile_89"], builds["base"]
    if not b.ok:
        _assert_refusal_names_field_and_both_values(b, r"capab|flash_attn_version|target profile mismatch", ())
        assert re.search(r"9\D*0|flash_attn_version=3", b.refusal), f"refusal does not state the host's value: {b.refusal[:600]}"
        return
    caps = declared_capabilities(b.manifest)
    assert caps == {"8.9"}, f"host answered 9.0; manifest declares {caps}"
    fa = {v for p, v in walk(b.manifest) if p and p[-1] == "vllm_flash_attn_version"}
    assert fa == {2}, f"observed FA versions {fa} under a build labelled 8.9"
    assert b.digest == base.digest
    if identity_of(base) is not None:
        assert identity_of(b) == identity_of(base), "same declared profile, host fact differs → identity must not differ"


# ---------------------------------------------------------------------------------------------------------------------------------
# (ii) a consistently changed target: refused with a verbatim reason, or built with a distinct identity and a truthful manifest
# ---------------------------------------------------------------------------------------------------------------------------------
def _assert_refusal_names_field_and_both_values(b: Build, field_re: str, values: tuple[str, ...]):
    assert re.search(field_re, b.refusal, re.I), f"{b.name}: refusal does not name the field /{field_re}/: {b.refusal[:800]}"
    for v in values:
        assert v in b.refusal, f"{b.name}: refusal does not state value {v!r}: {b.refusal[:800]}"
    assert not (b.out / "descriptor.json.gz").exists(), f"{b.name}: refused but a descriptor was written (not fail-closed)"
    leaked = [p.name for g in ADAPTER["identity_file_globs"] for p in b.out.glob(g)]
    assert not leaked, f"{b.name}: refused but an artifact/identity file was written (not fail-closed): {leaked}"


@needs_builder
def test_ii_a_capability_9_0_declared_consistently(builds):
    """a2. Host answers 9.0 at instantiation + export and the profile input says 9.0: on this vLLM pin FA3 is selected, so the
    attention rule must refuse naming `vllm_flash_attn_version` 3 vs 2 (fail-closed, verbatim). If the fix instead builds, the manifest
    must declare 9.0 and the identity must differ from base."""
    b = builds["cap90_consistent"]
    base = builds["base"]
    if not b.ok:
        _assert_refusal_names_field_and_both_values(b, r"flash_attn_version|capab", ("3", "2"))
        return
    assert declared_capabilities(b.manifest) == {"9.0"}, declared_capabilities(b.manifest)
    assert identity_of(b) is not None and identity_of(b) != identity_of(base)


@needs_builder
def test_ii_b_target_8_0_with_108_sms_declared_consistently(builds):
    """a3. An A100 target declared consistently: no derived consequence differs for B0 (FA2, default tiling), so the build may
    legitimately produce the SAME Program digest — but the manifest must say 8.0 / 108 (not "8.9") and the artifact identity must differ
    from base's. Or the build refuses naming the field and both values."""
    b = builds["cap80_sms108_consistent"]
    base = builds["base"]
    assert b.mutations, b.mutations
    if not b.ok:
        _assert_refusal_names_field_and_both_values(b, r"capab|NUM_SMS|flash_attn_version", ())
        return
    caps, sms = declared_capabilities(b.manifest), declared_num_sms(b.manifest)
    assert caps == {"8.0"}, f"manifest declares capability {caps} for an 8.0 construction (stale manifest, a3)"
    assert sms == {108}, f"manifest declares NUM_SMS {sms} for a 108-SM construction"
    if b.digest == base.digest:
        print(f"\n[cb-test] ii-b: identical Program digest {b.digest} for 8.0/108 — allowed (circuit unchanged)")
    ib, io_ = identity_of(base), identity_of(b)
    assert io_ is not None, f"no artifact identity in the 8.0/108 artefact (files: {b.files()})"
    assert io_ != ib, f"8.0/108 construction shares base's artifact identity: {io_}"


@needs_builder
def test_ii_c_num_sms_repinned_132_consistently(builds):
    """b2. NUM_SMS re-pinned 128 → 132 (shim and rule agree): identical Program (NUM_SMS sizes the persistent grid only), manifest
    NUM_SMS = 132, and the artifact identity must differ from base's."""
    b = builds["sms132_pin"]
    base = builds["base"]
    assert b.mutations, b.mutations
    if not b.ok:
        _assert_refusal_names_field_and_both_values(b, r"NUM_SMS", ("132",))
        return
    assert declared_num_sms(b.manifest) == {132}, declared_num_sms(b.manifest)
    assert identity_of(b) is not None and identity_of(b) != identity_of(base), f"132-SM construction shares base's identity: {identity_of(b)}"


# ---------------------------------------------------------------------------------------------------------------------------------
# (iii) two construction hosts, same profile → same artifact identity (and same Program digest)
# ---------------------------------------------------------------------------------------------------------------------------------
@needs_builder
def test_iii_two_hosts_same_profile_same_identity(builds):
    a, b = builds["base"], builds["hostB"]
    assert a.ok and b.ok, (a.refusal[:400], b.refusal[:400])
    assert a.digest == b.digest == B0_REQUEST_8_4_DIGEST
    ia, ib = identity_of(a), identity_of(b)
    assert ia is not None and ib is not None, f"no artifact identity (base: {ia is not None}, hostB: {ib is not None})"
    assert ia == ib, f"same profile, two hosts (hostname/out path/tree path/utc differ) → different identity:\n{ia}\n{ib}"
    # the baseline hosts (one with a GPU, one without) built the same Program; their pre-fix manifests carry only the digest
    for p in R9_REFERENCE_MANIFESTS:
        if p.exists():
            assert (json.load(open(p)).get("program") or {}).get("digest") == a.digest


@needs_builder
def test_iii_identity_excludes_host_facts(builds):
    """The identity must be a function of profile/target/construction-version facts, not of where it was built."""
    ident = identity_of(builds["base"])
    assert ident is not None
    text = json.dumps(ident, default=str)
    for host_fact in (os.uname().nodename, str(builds["base"].out), str(builds["base"].tree), builds["base"].manifest.get("utc") or "\0"):
        assert host_fact not in text, f"artifact identity embeds a host fact {host_fact!r}: {text[:400]}"


# ---------------------------------------------------------------------------------------------------------------------------------
# (iv) attempted reuse under a mismatched profile → rejected fail-closed
# ---------------------------------------------------------------------------------------------------------------------------------
def reuse_check_generic(artifact_dir: pathlib.Path, mismatched: dict, tree: pathlib.Path, python: str) -> tuple[bool | None, str]:
    """Probe the candidate applicability APIs in a subprocess (pod venv). Returns (rejected, reason); rejected=None when no API exists."""
    for cand in ADAPTER["reuse_probe_candidates"]:
        mod, _, fn = cand.partition(":")
        code = f"""
import importlib, json, sys
try:
    m = importlib.import_module({mod!r})
except Exception as e:
    print(json.dumps({{"api": False, "err": repr(e)}})); sys.exit(0)
f = getattr(m, {fn!r}, None)
if f is None:
    print(json.dumps({{"api": False, "err": "no attribute"}})); sys.exit(0)
try:
    r = f({str(artifact_dir)!r}, {json.dumps(mismatched)})
    print(json.dumps({{"api": True, "rejected": False, "ret": repr(r)[:800]}}))
except Exception as e:
    print(json.dumps({{"api": True, "rejected": True, "reason": repr(e)[:1200]}}))
"""
        p = subprocess.run([python, "-c", code], cwd=str(tree), env={**os.environ, "PYTHONPATH": _pythonpath(tree), "CUDA_VISIBLE_DEVICES": ""}, capture_output=True, text=True)
        try:
            j = json.loads(p.stdout.strip().splitlines()[-1])
        except Exception:  # noqa: BLE001
            continue
        if j.get("api"):
            return (bool(j.get("rejected")), j.get("reason") or j.get("ret") or "")
    return None, "no applicability/reuse API found among ADAPTER['reuse_probe_candidates'] (fill ADAPTER from the implementation's names)"


REUSE_PROBE_CBFIX = r'''
import importlib, inspect, json, sys, pathlib
art_dir = pathlib.Path(sys.argv[1]); req = json.loads(sys.argv[2]); tweak_app = json.loads(sys.argv[3])
out = {{"api": False}}
try:
    TP = importlib.import_module({tp_mod!r}).TargetProfile
except Exception as e:
    out["err"] = f"no TargetProfile: {{e!r}}"; print(json.dumps(out)); sys.exit(0)
la = cr = None; where = None
for mn in {mods!r}:
    try:
        m = importlib.import_module(mn)
    except Exception:
        continue
    la = la or getattr(m, "load_artifact", None); cr = cr or getattr(m, "check_reuse", None)
    if la or cr:
        where = where or mn
if la is None and cr is None:
    out["err"] = "no load_artifact/check_reuse in " + repr({mods!r}); print(json.dumps(out)); sys.exit(0)
out["api"] = True; out["where"] = where
out["sig"] = {{"load_artifact": str(inspect.signature(la)) if la else None, "check_reuse": str(inspect.signature(cr)) if cr else None}}
# the requester's profile: the DEFAULT profile with the requested fields overridden (only fields TargetProfile has)
fields = set(getattr(TP, "__dataclass_fields__", {{}}).keys()) or set(inspect.signature(TP).parameters)
kw = {{}}
for k, v in req.items():
    if k in fields:
        kw[k] = tuple(v) if isinstance(v, list) else v
try:
    prof = TP(**kw)
except Exception as e:
    out["err"] = f"TargetProfile({{kw}}) failed: {{e!r}}"; print(json.dumps(out)); sys.exit(0)
out["profile_kw"] = kw
# applicability_now / construction_version_now: the artifact's OWN records (same venv, same tree → the requester's read-back facts are
# the builder's), applicability optionally tweaked (a requester whose facts differ)
app_now = cv_now = None
art_path = art_dir / "artifact.json" if (art_dir / "artifact.json").exists() else art_dir
try:
    aj = json.load(open(art_dir / "artifact.json"))
    app_now = dict(aj.get("applicability") or {{}})
    cv_now = aj.get("construction_version")
    for k, v in tweak_app.items():
        app_now[k] = v
except Exception as e:
    out["artifact_json_err"] = repr(e)
def attempt(fn, *args):
    try:
        r = fn(*args)
        return {{"rejected": False, "ret": repr(r)[:600]}}
    except TypeError as e:
        return {{"type_error": repr(e)[:400]}}
    except Exception as e:
        return {{"rejected": True, "reason": (type(e).__name__ + ": " + str(e))[:2000]}}
def load_any():
    """the artifact object, by whichever load_artifact signature works (None when load itself refuses: that IS the check)"""
    last = {{"type_error": "no signature tried"}}
    for args in ((str(art_path), prof, app_now, cv_now), (str(art_path), prof, app_now), (str(art_path), prof), (str(art_path),)):
        try:
            return la(*args), None
        except TypeError as e:
            last = {{"type_error": repr(e)[:400]}}
            continue
        except Exception as e:
            return None, {{"rejected": True, "reason": (type(e).__name__ + ": " + str(e))[:2000]}}
    return None, last
res = None
if la is not None:
    art, r = load_any()
    if art is None:
        res = r                                  # load refused (or no signature fits)
    elif cr is not None:
        res = attempt(cr, art, prof, app_now, cv_now)    # loaded: the explicit reuse check decides
        if "type_error" in res:
            res = attempt(cr, art, prof, app_now)
        if "type_error" in res:
            res = attempt(cr, art, prof)
    else:
        res = {{"rejected": False, "ret": "load_artifact accepted; no check_reuse"}}
elif cr is not None:
    res = attempt(cr, str(art_path), prof, app_now)
out.update(res or {{"err": "no attempt succeeded"}})
print(json.dumps(out, default=str))
'''


def reuse_check_cbfix(artifact_dir: pathlib.Path, requested: dict, tree: pathlib.Path, python: str, tweak_applicability: dict | None = None) -> tuple[bool | None, str]:
    """Reuse check through the implementation's names (`TargetProfile`, `load_artifact` / `check_reuse`); the module of the
    latter two is probed. Returns (rejected, reason); rejected=None when the API is absent (e.g. on the unpatched base)."""
    code = REUSE_PROBE_CBFIX.format(tp_mod=ADAPTER["target_profile_module"], mods=tuple(ADAPTER["reuse_api_modules"]))
    p = subprocess.run([python, "-c", code, str(artifact_dir), json.dumps(requested), json.dumps(tweak_applicability or {})], cwd=str(tree),
                       env={**os.environ, "PYTHONPATH": _pythonpath(tree), "CUDA_VISIBLE_DEVICES": ""}, capture_output=True, text=True)
    try:
        j = json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        return None, f"probe produced no JSON (rc={p.returncode}): {p.stderr[-800:]}"
    if not j.get("api"):
        return None, j.get("err", "no API")
    if "rejected" not in j:
        return None, json.dumps(j)[:1200]
    info = json.dumps({k: j.get(k) for k in ("where", "sig", "profile_kw", "ret", "type_error")}, default=str)
    print(f"\n[cb-test] reuse probe: {info}")
    # the REASON TEXT alone (the refusal the API produced) — never the probe's own echo of the requested values
    return bool(j["rejected"]), (j.get("reason") or j.get("ret") or "")


def _reuse_check(artifact_dir, requested, tree, python, tweak_applicability=None):
    if ADAPTER["reuse_check"] is not None:
        return ADAPTER["reuse_check"](artifact_dir, requested, tree, python)
    rejected, reason = reuse_check_cbfix(artifact_dir, requested, tree, python, tweak_applicability)
    if rejected is None and not tweak_applicability:
        rejected, reason2 = reuse_check_generic(artifact_dir, requested, tree, python)
        reason = reason + " | " + reason2
    return rejected, reason


@needs_builder
def test_iv_reuse_under_mismatched_profile_is_rejected(builds, harness):
    """Contract bullet 4: the 8.9/128 artefact presented under a 9.0/132 profile → rejected, naming the differing fields with both values."""
    base = builds["base"]
    assert base.ok
    mismatched = {"compute_capability": [9, 0], "capability": [9, 0], "NUM_SMS": 132, "num_sms": 132}
    rejected, reason = _reuse_check(base.out, mismatched, base.tree, harness.python)
    print(f"\n[cb-test] reuse check under {mismatched}: rejected={rejected} reason={reason[:600]}")
    assert rejected is not None, f"contract bullet 4 untestable: {reason}"
    assert rejected, f"artifact built for 8.9/128 was ACCEPTED for reuse under {mismatched}: {reason}"
    assert re.search(r"capab|NUM_SMS|num_sms|applicab|profile", reason, re.I), f"rejection does not name the mismatched field: {reason}"
    assert re.search(r"9\D*0|\b132\b", reason), f"rejection does not state the requester's value (9.0 / 132): {reason}"
    assert re.search(r"8\D*9|\b128\b", reason), f"rejection does not state the artefact's value (8.9 / 128): {reason}"


@needs_builder
def test_iv_reuse_under_mismatched_applicability_is_rejected(builds, harness):
    """Contract bullet 4, second half: same declared profile but a requester whose read-back facts differ (FA3 host) → rejected."""
    base = builds["base"]
    matching = {"compute_capability": [8, 9], "num_sms": 128}
    rejected, reason = _reuse_check(base.out, matching, base.tree, harness.python, tweak_applicability={"vllm_flash_attn_version": 3})
    print(f"\n[cb-test] reuse check, same profile, applicability tweaked (FA3): rejected={rejected} reason={reason[:600]}")
    assert rejected is not None, f"untestable: {reason}"
    assert rejected, f"artifact accepted under a mismatched applicability: {reason}"
    assert re.search(r"flash_attn|applicab", reason, re.I) and "3" in reason, f"rejection does not name the applicability field/values: {reason}"


def _tampered_copy(base: Build, work: pathlib.Path, name: str, edit) -> pathlib.Path:
    """A copy of the artefact directory whose identity/applicability file has been edited by `edit(dict) -> dict` (a forged wrapper)."""
    src = None
    for g in ADAPTER["identity_file_globs"]:
        hits = sorted(base.out.glob(g))
        if hits:
            src = hits[0]
            break
    assert src is not None, f"no identity file to tamper with in {base.files()}"
    dst = work / name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(base.out, dst, ignore=shutil.ignore_patterns("inputs.json"))
    doc = json.loads((dst / src.name).read_text())
    (dst / src.name).write_text(json.dumps(edit(doc), indent=1, default=str))
    return dst


def _set_path(doc: dict, dotted: str, value):
    cur = doc
    keys = dotted.split(".")
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value
    return doc


def _get_path(doc, dotted: str):
    cur = doc
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _digest_field(doc: dict) -> str:
    """The wrapper's Program-digest field, whichever spelling it uses (must exist: a wrapper that does not name its Program binds nothing)."""
    for k in ("program_digest", "program.digest", "digest"):
        cur, ok = doc, True
        for part in k.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                ok = False
                break
        if ok and isinstance(cur, str) and len(cur) == 64:
            return k
    raise AssertionError(f"wrapper carries no Program digest field: {sorted(doc)}")


@needs_builder
@pytest.mark.parametrize("field,value,requester", [
    ("target_profile.compute_capability", [9, 0], {"compute_capability": [9, 0], "num_sms": 128}),  # relabel the wrapper as a 9.0 artefact
    ("<program digest>", "ff" * 32, {"compute_capability": [8, 9], "num_sms": 128}),                # claim another Program
    ("applicability.vllm_flash_attn_version", ["3"], {"compute_capability": [8, 9], "num_sms": 128}),  # widen the applicability
])
def test_iv_tampered_wrapper_is_rejected(builds, harness, field, value, requester):
    """Contract bullet 3 ('bind … into the identity'): the wrapper's facts must be BOUND, i.e. editing a stored fact to make the artefact
    look applicable must be caught (the identity no longer matches its components), not trusted."""
    base = builds["base"]
    assert identity_of(base) is not None

    def edit(d):
        f = _digest_field(d) if field == "<program digest>" else field
        assert _get_path(d, f) is not None, f"tamper target {f} absent from the wrapper (keys: {sorted(d)})"
        return _set_path(d, f, value)
    tampered = _tampered_copy(base, harness.work, f"tampered__{field.strip('<>').replace('.', '_').replace(' ', '_')}", edit)
    rejected, reason = reuse_check_cbfix(tampered, requester, base.tree, harness.python)
    print(f"\n[cb-test] tampered {field}={value!r}, requester {requester}: rejected={rejected} reason={reason[:500]}")
    assert rejected is not None, f"untestable: {reason}"
    assert rejected, f"forged wrapper ({field} := {value!r}) ACCEPTED for reuse: {reason}"


@needs_builder
def test_iii_identity_is_recomputable_from_its_published_definition(builds):
    """Independent recomputation: the wrapper states how its identity is formed; an outsider must be able to reproduce it from the
    stored components alone (else 'bound into the identity' is unverifiable). Tries the obvious canonicalisations; reports the one
    that reproduces. Also shows the identity DEPENDS on each component (changing any one changes the recomputed value)."""
    import hashlib
    import itertools
    base = builds["base"]
    docs = [json.loads(p.read_text()) for g in ADAPTER["identity_file_globs"] for p in sorted(base.out.glob(g))]
    docs = [d for d in docs if isinstance(d, dict) and isinstance(d.get("identity"), str)]
    assert docs, f"no wrapper with an 'identity' field in {base.files()}"
    doc = docs[0]
    comps = {k: v for k, v in doc.items() if k not in ("identity", "identity_definition", "schema")}
    digest_key = _digest_field(doc)
    assert doc[digest_key] == base.digest, (doc[digest_key], base.digest)
    ordered = [k for k in ("construction_version", "target_profile", "applicability") if k in comps] + [digest_key]
    assert len(ordered) == 4, f"components found: {ordered} (expected construction version, target profile, applicability, program digest)"

    def recompute(parts_by_key, sep, endian, prefixed):
        h = hashlib.sha256()
        for k in ordered:
            v = parts_by_key[k]
            b = v.encode() if isinstance(v, str) else json.dumps(v, sort_keys=True, separators=sep).encode()
            if prefixed:
                h.update(len(b).to_bytes(8, endian))
            h.update(b)
        return h.hexdigest()

    hit = None
    for sep, endian, prefixed in itertools.product([(",", ":"), (", ", ": ")], ["big", "little"], [True, False]):
        if recompute(doc, sep, endian, prefixed) == doc["identity"]:
            hit = (sep, endian, prefixed)
            break
    assert hit is not None, f"identity {doc['identity']} not reproducible from its components with any tried canonicalisation; definition: {doc.get('identity_definition')}"
    print(f"\n[cb-test] identity reproduced: canonical json separators={hit[0]!r}, len8 {hit[1]}-endian={hit[2]}; definition: {doc.get('identity_definition')}")
    for k in ordered:  # each component is load-bearing
        forged = dict(doc)
        forged[k] = ("00" * 32) if isinstance(doc[k], str) else {**doc[k], "__cbtest__": 1}
        assert recompute(forged, *hit) != doc["identity"], f"identity does not depend on component {k}"


@needs_builder
def test_iii_manifest_and_wrapper_agree_on_the_identity(builds):
    """The identity the manifest reports and the identity in the wrapper file must be the same value (no second copy that can drift —
    the very failure mode of the two capability literals)."""
    base = builds["base"]
    ids = set()
    for path, v in walk(base.manifest):
        if path and path[-1] == "identity" and isinstance(v, str) and len(v) >= 32:
            ids.add(v)
    for g in ADAPTER["identity_file_globs"]:
        for p in base.out.glob(g):
            try:
                doc = json.loads(p.read_text())
            except Exception:  # noqa: BLE001
                continue
            for path, v in walk(doc):
                if path and path[-1] == "identity" and isinstance(v, str) and len(v) >= 32:
                    ids.add(v)
    assert ids, "no identity value found in manifest or wrapper"
    assert len(ids) == 1, f"manifest and wrapper disagree on the identity: {ids}"
    print(f"\n[cb-test] identity {next(iter(ids))}")


@needs_builder
def test_iv_reuse_under_the_matching_profile_is_accepted(builds, harness):
    """Control for (iv): the artefact must remain usable under its own declared profile (fail-closed must not mean always-closed)."""
    base = builds["base"]
    ib = identity_of(base)
    assert ib is not None
    matching = {"compute_capability": [8, 9], "capability": [8, 9], "NUM_SMS": 128, "num_sms": 128}
    rejected, reason = _reuse_check(base.out, matching, base.tree, harness.python)
    assert rejected is not None, reason
    assert not rejected, f"artifact rejected under its OWN profile: {reason}"


# ---------------------------------------------------------------------------------------------------------------------------------
# (vi) an inert env var is not a declared fact
# ---------------------------------------------------------------------------------------------------------------------------------
@needs_builder
def test_vi_dead_env_var_is_not_recorded_as_a_declared_fact(builds):
    """d1: `VLLM_ATTENTION_BACKEND` is not read by this vLLM pin; setting it changes nothing (same digest, FA2 observed) so it must not
    appear among the facts the manifest DECLARES the build used (a cache keyed on it would key on noise), and it must not alter the
    artifact identity."""
    b, base = builds["attn_env_dead"], builds["base"]
    assert b.ok and b.digest == base.digest, (b.refusal[:400], b.digest)
    hits = declared_paths_with_key(b.manifest, "VLLM_ATTENTION_BACKEND")
    assert not hits, f"inert env var recorded as a declared fact at {hits} = {[dict(walk(b.manifest)).get(h) for h in hits]}"
    ib, ie = identity_of(base), identity_of(b)
    if ib is not None:
        assert ie == ib, f"an inert env var changed the artifact identity:\n{ib}\n{ie}"


# ---------------------------------------------------------------------------------------------------------------------------------
# (vii) constructing on a host that is not the declared target must refuse (the pod's L40S has 142 SMs; the profile pins 128)
# ---------------------------------------------------------------------------------------------------------------------------------
@needs_builder
@pytest.mark.skipif(os.environ.get("CBTEST_GPU_HOST") != "1", reason="CBTEST_GPU_HOST=1 to run the GPU-visible host-mismatch build")
def test_vii_construction_on_a_mismatched_host_never_mislabels(builds):
    """The construction host is an L40S (142 SMs, sm_89); the declared profile is the 4090 record (128 SMs). Two acceptable outcomes:
    (a) refuse, naming NUM_SMS with both values (a base without the contract: the real query answers 142 and the GEMM rule refuses against the pin), or
    (b) build FOR THE DECLARED TARGET — then the manifest's declared == applied == 128 / 8.9, the Program digest and the artifact
    identity equal the GPU-free build's (a host fact must not enter the identity), and the host GPU is recorded as a host fact.
    Unacceptable: a built artefact whose declared/applied facts show the host's 142, or whose identity differs from base's."""
    b, base = builds["gpu_host_mismatch"], builds["base"]
    assert b.result.get("cuda_available") is True, b.result.get("cuda_available")
    if not b.ok:
        _assert_refusal_names_field_and_both_values(b, r"NUM_SMS|capab", ("128",))
        print(f"\n[cb-test] host-mismatch outcome (a) refusal: {b.refusal[:500]}")
        return
    assert declared_num_sms(b.manifest) == {128}, f"declared/applied NUM_SMS {declared_num_sms(b.manifest)} on a 142-SM host"
    assert declared_capabilities(b.manifest) == {"8.9"}, declared_capabilities(b.manifest)
    assert b.digest == base.digest, (b.digest, base.digest)
    ib, ih = identity_of(base), identity_of(b)
    assert ih is not None and ih == ib, f"identity differs between the GPU-free and the GPU-visible construction of the same profile:\n{ib}\n{ih}"
    host_leaves = [(p, v) for p, v in walk(b.manifest) if any(re.search(r"host", k, re.I) for k in p) and v not in (None, "", False)]
    assert any(("NVIDIA" in str(v)) or ("L40" in str(v)) or v == 142 for _p, v in host_leaves), f"host GPU not recorded as a host fact: {host_leaves[:12]}"
    print(f"\n[cb-test] host-mismatch outcome (b): built for the declared target on a foreign host; host facts recorded: {host_leaves[:8]}")


# ---------------------------------------------------------------------------------------------------------------------------------
# host != target is acceptable iff (c) construction on an L40S (142 SMs, GPU visible) with the 128-SM/8.9 profile yields the SAME
# Program digest AND the same artifact identity as the baseline constructions (no GPU; a 4090)
# ---------------------------------------------------------------------------------------------------------------------------------
def _wrapper_doc(b: "Build") -> dict | None:
    for g in ADAPTER["identity_file_globs"]:
        for p in sorted(b.out.glob(g)):
            try:
                d = json.loads(p.read_text())
            except Exception:  # noqa: BLE001
                continue
            if isinstance(d, dict) and isinstance(d.get("identity"), str):
                return d
    return None


def _identity_recomputer(doc: dict):
    """Return f(components_doc) -> hex reproducing doc['identity'] from its stored components (the canonicalisation the wrapper uses)."""
    import hashlib
    import itertools
    digest_key = _digest_field(doc)
    ordered = [k for k in ("construction_version", "target_profile", "applicability") if k in doc] + [digest_key]

    def make(sep, endian, prefixed):
        def f(d):
            h = hashlib.sha256()
            for k in ordered:
                v = d[k]
                b = v.encode() if isinstance(v, str) else json.dumps(v, sort_keys=True, separators=sep).encode()
                if prefixed:
                    h.update(len(b).to_bytes(8, endian))
                h.update(b)
            return h.hexdigest()
        return f

    for sep, endian, prefixed in itertools.product([(",", ":"), (", ", ": ")], ["big", "little"], [True, False]):
        f = make(sep, endian, prefixed)
        if f(doc) == doc["identity"]:
            return f
    raise AssertionError(f"identity {doc['identity']} not reproducible from its components; definition: {doc.get('identity_definition')}")


def _r9_recorded_applicability(m: dict) -> dict:
    """The applicability facts a PRE-FIX manifest recorded, mapped onto the fix's applicability field names: attention impl /
    backend / FA version (observed on every layer), the numerics profile, torch/triton/vLLM versions (site-packages dist-info names)."""
    out: dict = {}
    impls = (m.get("observed") or {}).get("attention_impls") or {}
    if impls:
        out["attention_impl"] = sorted({str(v.get("impl")) for v in impls.values()})
        out["attention_backend"] = sorted({str(v.get("backend")) for v in impls.values()})
        out["vllm_flash_attn_version"] = sorted({str(v.get("vllm_flash_attn_version")) for v in impls.values()})
    if m.get("profile"):
        out["numerics_profile"] = m["profile"]
    pkgs = ((m.get("inputs_trace") or {}).get("site_packages_by_package") or {})
    for name in pkgs:
        for pkg, field in (("torch-", "torch_version"), ("triton-", "triton_version"), ("vllm-", "vllm_version")):
            if name.startswith(pkg) and name.endswith(".dist-info"):
                ver = name[len(pkg):-len(".dist-info")]
                out[field] = ver[:-len(".cu129")] if field == "vllm_version" and ver.endswith(".cu129") else ver
    return out


@needs_builder
@pytest.mark.skipif(os.environ.get("CBTEST_GPU_HOST") != "1", reason="CBTEST_GPU_HOST=1 to run the GPU-visible host-mismatch build")
def test_ruling_c_cp_cb8_gpu_construction_equals_the_r9_hosts(builds):
    """Rule (c): the GPU-visible construction on an L40S (142 SMs) for the 128-SM/8.9 profile must carry the same
    Program digest AND the same artifact identity as the baseline constructions without a GPU and on a 4090. The baseline hosts built
    on the pre-fix code (no identity), so their identity is RECOMPUTED under the fix's published definition from what they recorded:
    same construction code (the fix's `construction_version`, a code fact), the default profile, and their recorded applicability
    facts (attention impl/backend/FA version, numerics profile, torch/triton/vLLM versions) substituted into the applicability block;
    applicability fields the baseline manifests did not record (kernel-source / tuned-table hashes, dispatch) are functions of the identical
    vLLM wheel pin and are taken from the GPU-visible record — reported as such."""
    gpu, base = builds["gpu_host_mismatch"], builds["base"]
    assert gpu.result.get("cuda_available") is True and gpu.ok, f"GPU-visible construction did not build: {gpu.refusal[:400]}"
    assert gpu.digest == base.digest == B0_REQUEST_8_4_DIGEST, (gpu.digest, base.digest)
    doc_gpu, doc_free = _wrapper_doc(gpu), _wrapper_doc(base)
    assert doc_gpu and doc_free, "no identity wrapper"
    assert doc_gpu["identity"] == doc_free["identity"], f"GPU-visible vs GPU-free identity differ on the same host:\n{doc_gpu['identity']}\n{doc_free['identity']}"
    recompute = _identity_recomputer(doc_gpu)
    refs = [p for p in R9_REFERENCE_MANIFESTS if p.exists() and "r9/" in str(p)]
    if not refs:
        pytest.skip("baseline reference manifests not in this checkout")
    unrecorded = None
    for p in refs:
        m = json.load(open(p))
        assert (m.get("program") or {}).get("digest") == gpu.digest, (str(p), (m.get("program") or {}).get("digest"))
        rec = _r9_recorded_applicability(m)
        assert rec.get("vllm_flash_attn_version") and rec.get("torch_version"), f"{p}: recorded facts too thin: {rec}"
        app = dict(doc_gpu["applicability"])
        mism = {k: (v, app.get(k)) for k, v in rec.items() if app.get(k) != v}
        assert not mism, f"{p.name} ({p.parent.parent.name}): recorded applicability facts differ from the GPU-visible build's: {mism}"
        unrecorded = sorted(set(app) - set(rec))
        substituted = {**doc_gpu, "applicability": {**app, **rec}}
        ident_ref = recompute(substituted)
        assert ident_ref == doc_gpu["identity"], f"identity recomputed for {p} differs: {ident_ref} vs the GPU-visible build's {doc_gpu['identity']}"
        host = (m.get("construction") or {}).get("cuda_available")
        print(f"\n[cb-test] ruling (c): {p.relative_to(REPO)} (cuda_available={host}) -> digest {m['program']['digest'][:16]}…, identity {ident_ref[:16]}… == the GPU-visible build's")
    print(f"[cb-test] applicability fields not recorded by the baseline manifests (taken from the identical vLLM wheel pin on the GPU-visible host): {unrecorded}")
