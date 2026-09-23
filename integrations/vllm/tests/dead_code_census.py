"""Dead-code census of `integrations/vllm` (static import reachability from the live entrypoints, optionally merged with a
dynamic `sys.modules` record from the pod).  Machine-readable output beside this file: `dead_code_census.json`; the dynamic record
(`{module: [where it was seen imported]}`, sitecustomize/atexit dumps of the harness + the CPU entrypoints) is `dead_code_imported.json`.
The census of main 22e10e0 that produced the 2026-09-23 deletions (plan, dedicated tests, artifacts) is this file's first commit.

    cd integrations/vllm
    python tests/dead_code_census.py [--imported tests/dead_code_imported.json] [--out tests/dead_code_census.json] [--git-dates]

LIVE ENTRYPOINTS (roots; `ENTRYPOINT_SHELLS` are parsed for `-m module`, `python <path>.py`, `<name>.py` tokens and heredoc
`from X import` / `import X` lines; `ENTRYPOINT_MODULES` are the Python entrypoints named directly):

  * the sweep runners `verity_capture/sweep/row_pod.sh`, `row_pod_tp2.sh` (TP2, rows #70/#75), `run_row_v2.sh` (the research
    Tools' wrapper), the acquisition gates `verity_vllm/acquire/pod_gate.sh`, `pod_match_v2.sh`, the pod runtime `pod_bootstrap.sh`
    and the tap builders it runs (`pod_hidden_gpu.sh`, `pod_fa2_tap.sh`, `pod_fa3_tap.sh`);
  * `verity_vllm.research_tools` (the `vllm.build/match/commit` Tools registered in `tools/research/.../tools_registry.py`),
    `verity_capture.sweep.rebuild_digest_gate`, `verity_vllm.query.cli`, the R19 reference regression harness `tests/regression/**`.

STATIC REACHABILITY: an ast import graph over every `.py` under the integration root.  Edges: `import a.b`, `from a.b import c`
(`c` as a submodule when it is one), relative imports, the parents of every imported dotted name (their `__init__`), string
literals that name an integration module (`importlib.import_module("verity_capture.analytic")`, dotted registry paths, the
`profiles` meta_path consumers), f-string prefixes (`f"verity_vllm.registry.{m}"` + every short literal of the same file that
completes to a module: `REGISTRY_MODULES`), and path literals (`verity_capture/sweep/hidden_gpu_src/hidden_gpu.py`).  Anything
reachable from a root is LIVE.  `verity_capture/profiles/**` is resolved by name at runtime (workload -> case -> profile module):
the whole package is treated as reachable when any consumer is (and it is another lane's anyway, see PROTECTED).

CLASSES (per module under the census packages):
    LIVE       static_reachable or dynamically_imported
    TEST-ONLY  not LIVE, imported by at least one test file
    DEAD       not LIVE, imported by no test

DELETION PLAN (also in the JSON): a test file is KEPT when it is part of the harness, imports at least one LIVE integration
module, or imports no integration module at all; it is DELETED when every integration module it imports is non-LIVE, or when
it is the dedicated test of a non-LIVE module (`test_<module>*.py`) -- then the module goes with it.  Non-LIVE modules that a
KEPT test still imports (transitively through non-LIVE modules) are fixture generators used by live tests: kept, listed
under `fixtures_kept`.  Modules under PROTECTED paths are never in the plan (other lanes own them); they are listed apart.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                     # integrations/vllm
REPO = ROOT.parent.parent

PACKAGES = ("verity_vllm", "verity_capture", "verity_vllm_numerics", "verity_vllm_adapter")
CENSUS_PACKAGES = ("verity_vllm", "verity_capture", "verity_vllm_numerics")
_TOPS = "|".join(PACKAGES + ("tests",))
TOP_MODULE_RE = re.compile(rf"^(?:{_TOPS})(?:\.[A-Za-z_][A-Za-z0-9_]*)+$")
PATH_LITERAL_RE = re.compile(rf"^(?:{_TOPS})(?:/[A-Za-z0-9_.-]+)+\.py$")

ENTRYPOINT_SHELLS = (
    "verity_capture/sweep/row_pod.sh",
    "verity_capture/sweep/row_pod_tp2.sh",
    "verity_capture/sweep/run_row_v2.sh",
    "verity_vllm/acquire/pod_gate.sh",
    "verity_vllm/acquire/pod_match_v2.sh",
    "pod_bootstrap.sh",
    "verity_capture/sweep/pod_hidden_gpu.sh",
    "verity_capture/sweep/pod_fa2_tap.sh",
    "verity_capture/sweep/pod_fa3_tap.sh",
)
ENTRYPOINT_MODULES = (
    "verity_vllm.research_tools",                # tools/research/src/research/store/tools_registry.py: vllm.build / match / commit
    "verity_capture.sweep.rebuild_digest_gate",
    "verity_vllm.query.cli",
)
ENTRYPOINT_DIRS = ("tests/regression",)        # the R19 reference regression harness: every module under it is a root

#: Other lanes own these; the census reports them but the deletion plan never touches them.
PROTECTED = (
    "verity_capture/profiles/",                  # profiles lane
    "verity_vllm/query/",                        # memory / gm lanes
    "verity_vllm/acquire/",
    "verity_capture/experimental/cb_a/global_match.py",
    "verity_vllm/runtime_correspondence.py",
    "verity_capture/commit/required_manifest.py",  # lane/vllm-v2-default (v1 alternatives: required_manifest tables, v1 acquisition, finalize)
    "verity_capture/sweep/finalize.py",
)

#: Non-LIVE modules kept on purpose (module -> why).  Fixture generators / reference models that tests of LIVE modules import, and
#: the TP2-path modules the brief keeps by name (`tp2_*`, `tp/`).  Everything else non-LIVE is in the deletion plan.
KEEP_FIXTURES: dict[str, str] = {
    "verity_capture.bench.gen_batch_candidate": "fixture generator: cb_a/tests/test_compact.py (LIVE compact / global_match_fast / rank_match), tests/test_gen_ovbatch.py (LIVE fold)",
    "verity_capture.experimental.cb_a.instances_form": "reference of the progressions form; cb_a/tests/test_compact.py checks the LIVE fold's output against it",
    "verity_capture.bench.gen_coverage_workloads": "workload generator: test_chunked_attribution.py / test_coverage_workloads.py (LIVE chunk_attribution, arrivals, gates)",
    "verity_capture.bench.gen_ov_sampling_libdevice": "libdevice transcription: test_gen_ov_sampling.py checks it against the LIVE resolver / gumbel definitions",
    "verity_capture.bench.topp_split_probe": "CPU checker + GPU-truth fixture: tests/ir/test_topp_split_geometry.py (LIVE registry.topp_split)",
    "verity_capture.experimental.fa2_commit.derived": "fa2_commit fixtures: test_negatives.py / test_roundtrip.py exercise the LIVE fa2_commit.engine (native_collect's committer)",
    "verity_capture.experimental.fa2_commit.encoding": "fa2_commit fixtures (see derived)",
    "verity_capture.experimental.fa2_commit.fixture": "fa2_commit fixtures (see derived)",
    "verity_capture.experimental.fa2_commit.layouts": "fa2_commit fixtures (see derived)",
    "verity_capture.experimental.fa2_commit.reference": "fa2_commit fixtures (see derived)",
    "verity_vllm.frontend.derive_examples": "source modules of the derivation tests (LIVE frontend.derive, 9 tests)",
    "verity_vllm.frontend.examples": "authored examples of the torch-frontend tests (LIVE frontend.torch_frontend / correspond)",
    "verity_vllm.frontend.b1_authored": "B1 authored through the frontend: digest-equality acceptance of LIVE registry.b1 (test_frontend_rulings, test_composition)",
    "verity_vllm.frontend.serve3_authored": "serve3 authored through the frontend: digest-equality acceptance of LIVE registry.serve3",
    "verity_vllm.lifting.dump": "complete-assignment dump used by tests/ir/test_lifted_tiny.py (LIVE registry.lifted / pad_prims)",
    "verity_vllm.ir.descriptor_equivalence": "denotation compare used by test_codec.py / test_heldout_codec_compose.py (LIVE registry b1 / b1_tp2 / hopper)",
    "verity_vllm.ir.profile_descriptor": "descriptor byte profile used by test_codec.py",
    "verity_vllm.registry.conformance": "conformance record read by test_nan_conversion.py (LIVE registry.b1 / prims)",
    "verity_vllm.synthetic": "numpy toy model of the commit_bench tests (LIVE commit_bench.fasttree / stream / semantic)",
    "verity_vllm.required": "imported by verity_vllm.synthetic",
    "verity_vllm_numerics.fa3_model": "independent CPU FA3 model cross-checked by test_composition.py / test_derived_rows.py (LIVE crosscheck / derived_rows)",
    "verity_vllm_numerics.inductor_models": "CPU bit-models cross-checked by test_composition.py",
    "verity_vllm.correspondence.program": "used by tests/ir/test_derive_negative.py (LIVE frontend.derive negatives)",
    "verity_capture.sweep.fa2_attn_oracle": "the P0 FA2 tap oracle (R8): verity_capture/tests/test_fa2_tap_geometry.py loads it by file name beside the LIVE "
                                            "commit_integ.hidden_source checks (softcap params vs flash_api, MUFU.TANH rules)",
    "verity_capture.tp.collective_link": "TP2 path kept by the brief (`tp/`); statically unreachable from row_pod_tp2.sh -- verity_capture/tests/test_tp_collective.py",
    "verity_capture.tp.collective_record": "TP2 path kept by the brief (`tp/`); statically unreachable from row_pod_tp2.sh -- test_tp_collective.py",
    "verity_capture.bench.tp2_analyze": "TP2 path kept by the brief (`tp2_*`); statically unreachable from row_pod_tp2.sh -- test_tp2_analyze.py / test_tp_collective.py",
    "verity_capture.quarantine_lint": "imported by test_tp_collective.py (TP2 path test); lints verity_vllm/registry/quarantine",
    "verity_vllm.registry.quarantine.collective": "provisional TP2 AllReduce definitions; test_tp_collective.py (TP2 path test)",
    "verity_vllm.registry.quarantine.collective.allreduce": "see quarantine.collective",
    "verity_vllm.registry.quarantine.collective.allreduce_difftest": "see quarantine.collective",
}

#: Tests kept although one test FUNCTION imported a deleted module: that function is removed by the deletion commit (test -> note).
TRIMMED_TESTS: dict[str, str] = {
    "tests/ir/test_lifted_r17.py": "test_a2_lifted_instances_emit_reads_the_declared_lag_and_reports_true_widths removed (it imported "
                                   "verity_vllm.lifting.record + tests/ir/lifted_instances_emit.py); the other 14 tests exercise LIVE registry.lifted",
}

#: row_pod.sh's Match runs run_config with `--skip census,census_reconcile,nonint,nonint_compare,compare,root_policy,golden,wiring,replay_a,
#: replay_chain,fold_2`: the modules those stages would `-m` are statically reachable (run_config's stage table names them) but never run by
#: the live launcher.  Reported, not deleted (the rule is static reachability): `live_only_via_run_config_skipped_stages`.
RUN_CONFIG_SKIPPED_STAGE_MODULES = ("verity_capture.census", "verity_capture.noninterference", "verity_capture.compare",
                                    "verity_capture.root_policy", "verity_capture.golden", "verity_capture.provenance")

#: Non-.py artifacts.  An unreferenced file (no tracked file of the whole repo names it, its ancestor directories included) is deleted
#: only inside these scopes (the brief's list: shell scripts, `bench/fixtures`, `profiles/workload_*.json`); elsewhere it is reported
#: (`artifacts.unreferenced_out_of_scope`) -- `workloads/<row>.json` are the qualification table's 96 row manifests, read by row id.
ARTIFACT_DELETE_SCOPES = ("verity_capture/bench/fixtures/", "verity_capture/profiles/workload_")
ARTIFACT_DELETE_SUFFIXES = (".sh",)
#: Unreferenced by name but read through a name PATTERN by a kept test (file -> the reader).
ARTIFACT_KEEP: dict[str, str] = {
    **{f"verity_capture/profiles/workload_cov_b0_{p}_{c}.json": "verity_capture/tests/test_coverage_workloads.py builds the name from gw.PROFILES x gw.CTX"
       for p in ("b1", "b8-mixed", "b8-static") for c in ("c256", "c1024")},
    "verity_capture/profiles/workload_b1_sampling_topk50_topp09.json": "verity_capture/tests/test_gen_ov_sampling.py parametrizes workload_b1_sampling_{name} over B1",
    "verity_capture/profiles/workload_cov_b0_b1_c256_t1.json": "named by verity_capture/profiles/gen_ov_sampling_b0.py (profiles lane) as `cov_pod.sh run B0 cov_b0_b1_c256_t1`",
    "verity_capture/profiles/workload_cov_b0_b8-mixed_c256_t1.json": "the D87 `workload_cov_b0_*_t1.json` glob named by profiles/gen_ov_sampling_b0.py (profiles lane)",
    "verity_capture/profiles/hf_configs/QWEN3_4B_FP8.config.json": "profiles lane; loaded by role",
    **{f"tests/regression/expected/{r}.json": "harness contract read as expected/<row>.json" for r in (
        "olmoe-1b-7b__bf16__l40s__tp2__b8__i1024__o128__mixed__greedy__bi-eager", "qwen3-30b-a3b__bf16__l40s__tp2__b2__i1024__o128__mixed__greedy__bi-eager",
        "smollm2-135m__bf16__l40s__tp1__b16__i1024__o128__mixed__greedy__bi-eager")},
    **{f"verity_capture/commit/tests/fixtures/admission/{n}.json": "commit/tests/test_admission*.py read FIX/<name>.json (the table grows by name)"
       for n in ("b4", "b8stoch", "phi3b1")},
    **{f"verity_capture/tests/fixtures/tp2/fold_gap.shard.rank{r}.json": "test_tp_collective.py reads fold_gap.shard.rank{r}.json (TP2 path)" for r in (0, 1)},
    "verity_capture/experimental/fa2_commit/fixtures/p0/p0_summary.json": "record of the P0 kernel dumps beside them (kernel_dump.py reads the .npz)",
}

SHELL_MODULE_RE = re.compile(r"-m\s+([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+)")
SHELL_PY_RE = re.compile(r"(?<![\w/.-])((?:[A-Za-z0-9_./-]+/)?[A-Za-z0-9_]+\.py)(?![\w])")
SHELL_IMPORT_RE = re.compile(r"^\s*(?:from\s+([A-Za-z_][\w.]*)\s+import|import\s+([A-Za-z_][\w.]*))", re.MULTILINE)


# ---------------------------------------------------------------------------------------------------------------------------- files
def py_files() -> list[Path]:
    out: list[Path] = []
    for top in PACKAGES + ("tests",):
        for p in sorted((ROOT / top).rglob("*.py")):
            if "__pycache__" in p.parts or p.resolve() == Path(__file__).resolve():
                continue                                             # this tool names modules in string literals; it is not a consumer
            out.append(p)
    return out


def modname(p: Path) -> str:
    rel = p.relative_to(ROOT).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def is_test_file(rel: str) -> bool:
    parts = rel.split("/")
    return parts[0] == "tests" or "tests" in parts[:-1] or parts[-1].startswith("test_")


def loc(p: Path) -> int:
    with open(p, "rb") as f:
        return sum(1 for _ in f)


def protected(rel: str) -> str | None:
    for pre in PROTECTED:
        if rel == pre or rel.startswith(pre):
            return pre
    return None


# ---------------------------------------------------------------------------------------------------------------------------- graph
class Graph:
    def __init__(self) -> None:
        self.files = py_files()
        self.mod_of: dict[str, str] = {}            # rel path -> module
        self.file_of: dict[str, str] = {}           # module -> rel path
        for p in self.files:
            rel = p.relative_to(ROOT).as_posix()
            m = modname(p)
            self.mod_of[rel] = m
            self.file_of[m] = rel
        self.by_basename: dict[str, list[str]] = defaultdict(list)   # "x.py" -> modules with that file name (non-__init__)
        for rel, m in self.mod_of.items():
            if Path(rel).name != "__init__.py":
                self.by_basename[Path(rel).name].append(m)
        self.edges: dict[str, set[str]] = defaultdict(set)      # module -> modules it imports (resolved, existing)
        self.edge_kinds: dict[tuple[str, str], set[str]] = defaultdict(set)
        for p in self.files:
            self._scan(p)

    # -- resolution
    def exists(self, m: str) -> bool:
        return m in self.file_of

    def parents(self, m: str) -> list[str]:
        parts = m.split(".")
        return [".".join(parts[:i]) for i in range(1, len(parts))]

    def add(self, src: str, dst: str, kind: str) -> None:
        for m in self.parents(dst) + [dst]:
            if self.exists(m) and m != src:
                self.edges[src].add(m)
                self.edge_kinds[(src, m)].add(kind)

    def resolve_from(self, src_mod: str, src_is_pkg: bool, node: ast.ImportFrom) -> list[str]:
        if node.level == 0:
            base = node.module or ""
        else:
            anchor = src_mod.split(".")
            if not src_is_pkg:
                anchor = anchor[:-1]
            anchor = anchor[: len(anchor) - (node.level - 1)] if node.level > 1 else anchor
            base = ".".join(anchor + ([node.module] if node.module else []))
        out = [base] if base else []
        for a in node.names:
            cand = f"{base}.{a.name}" if base else a.name
            if self.exists(cand):
                out.append(cand)
        return out

    def _scan(self, p: Path) -> None:
        rel = p.relative_to(ROOT).as_posix()
        src_mod = self.mod_of[rel]
        src_is_pkg = p.name == "__init__.py"
        try:
            tree = ast.parse(p.read_bytes(), filename=str(p))
        except SyntaxError as e:
            print(f"WARN syntax error {rel}: {e}", file=sys.stderr)
            return
        strings: list[str] = []
        fprefixes: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    self.add(src_mod, a.name, "import")
            elif isinstance(node, ast.ImportFrom):
                for m in self.resolve_from(src_mod, src_is_pkg, node):
                    self.add(src_mod, m, "from-import")
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                strings.append(node.value)
            elif isinstance(node, ast.JoinedStr) and node.values and isinstance(node.values[0], ast.Constant) \
                    and isinstance(node.values[0].value, str):
                fprefixes.append(node.values[0].value)
        for s in strings:
            if TOP_MODULE_RE.match(s) and self.exists(s):
                self.add(src_mod, s, "string-literal")
            elif PATH_LITERAL_RE.match(s) and s in self.mod_of:
                self.add(src_mod, self.mod_of[s], "path-literal")
            elif s.endswith(".py") and "/" not in s and s in self.by_basename and len(self.by_basename[s]) == 1:
                # a file name joined at runtime (os.path.join(REPO, "verity_capture", "sweep", "fa2_attn_oracle.py")): unique basename
                self.add(src_mod, self.by_basename[s][0], "basename-literal")
        for pre in fprefixes:
            if not pre.endswith("."):
                continue
            pkg = pre[:-1]
            if not TOP_MODULE_RE.match(pkg) or not self.exists(pkg):
                continue
            self.add(src_mod, pkg, "fstring-package")
            if pkg == "verity_capture.profiles":
                # resolved by name from the workload's case at runtime (and a meta_path finder synthesises dense_<ROLE>): whole package
                for m in self.file_of:
                    if m.startswith(pkg + "."):
                        self.add(src_mod, m, "fstring-profiles")
                continue
            for s in strings:
                if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", s) and self.exists(f"{pkg}.{s}"):
                    self.add(src_mod, f"{pkg}.{s}", "fstring-literal")

    def closure(self, roots: set[str], skip: set[str] | None = None) -> set[str]:
        seen = set()
        stack = [r for r in roots if self.exists(r)]
        while stack:
            m = stack.pop()
            if m in seen or (skip and m in skip):
                continue
            seen.add(m)
            stack.extend(self.edges.get(m, ()))
        return seen

    def why(self, roots: set[str], target: str) -> list[str] | None:
        """Shortest import path root -> ... -> target (BFS), or None."""
        from collections import deque
        prev: dict[str, str | None] = {r: None for r in roots if self.exists(r)}
        q = deque(prev)
        while q:
            m = q.popleft()
            if m == target:
                path = []
                while m is not None:
                    path.append(m)
                    m = prev[m]
                return path[::-1]
            for n in self.edges.get(m, ()):
                if n not in prev:
                    prev[n] = m
                    q.append(n)
        return None


# ---------------------------------------------------------------------------------------------------------------------------- roots
def shell_roots(g: Graph) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """(root module -> shells naming it, shell -> modules) from the live shells."""
    by_mod: dict[str, set[str]] = defaultdict(set)
    by_shell: dict[str, set[str]] = defaultdict(set)
    non_test_by_base: dict[str, list[str]] = defaultdict(list)
    for rel in g.mod_of:
        non_test_by_base[os.path.basename(rel)].append(rel)
    for sh in ENTRYPOINT_SHELLS:
        text = (ROOT / sh).read_text()
        mods: set[str] = set()
        for m in SHELL_MODULE_RE.findall(text):
            if g.exists(m):
                mods.add(m)
        for a, b in SHELL_IMPORT_RE.findall(text):
            m = a or b
            if g.exists(m):
                mods.add(m)
        for tok in SHELL_PY_RE.findall(text):
            tok = tok.lstrip("./")
            if tok in g.mod_of:
                if not is_test_file(tok):         # row_pod.sh names the tests that extract its shell blocks, in comments
                    mods.add(g.mod_of[tok])
                continue
            cands = [c for c in non_test_by_base.get(os.path.basename(tok), []) if not is_test_file(c)]
            if len(cands) > 1:   # `$SRC/build_ext.py`: the dir the shell names (fa2_tap_src / fa3_tap_src / hidden_gpu_src) disambiguates
                cands = [c for c in cands if any(d in text for d in Path(c).parts[:-1] if d.endswith("_src"))]
            if len(cands) == 1:
                mods.add(g.mod_of[cands[0]])
        for m in mods:
            by_mod[m].add(sh)
            by_shell[sh].add(m)
    return by_mod, by_shell


def all_shell_refs(g: Graph) -> dict[str, set[str]]:
    """module -> every .sh under the integration that names it (live or not): `referenced_by_shell`."""
    out: dict[str, set[str]] = defaultdict(set)
    for sh in sorted(ROOT.rglob("*.sh")):
        rel = sh.relative_to(ROOT).as_posix()
        text = sh.read_text(errors="replace")
        for m in SHELL_MODULE_RE.findall(text):
            if g.exists(m):
                out[m].add(rel)
        for a, b in SHELL_IMPORT_RE.findall(text):
            if g.exists(a or b):
                out[a or b].add(rel)
        for tok in SHELL_PY_RE.findall(text):
            tok = tok.lstrip("./")
            if tok in g.mod_of:
                out[g.mod_of[tok]].add(rel)
    return out


def git_dates(rels: list[str]) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for rel in rels:
        try:
            r = subprocess.run(["git", "log", "-1", "--format=%cs", "--", rel], cwd=ROOT, capture_output=True, text=True, timeout=30, check=False)
            out[rel] = r.stdout.strip() or None
        except Exception:  # noqa: BLE001
            out[rel] = None
    return out


# ---------------------------------------------------------------------------------------------------------------------------- artifacts
def artifact_refs() -> dict[str, dict]:
    """Every tracked non-.py file under the integration: which tracked files of the WHOLE repo (tools/research and docs included) name
    it -- by basename, or by one of its ancestor directories (data read by directory / glob).  `referenced_by == []` = a deletion
    candidate (reviewed by hand: the census lists, the commit decides)."""
    files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split("\n")
    self_files = {"integrations/vllm/tests/dead_code_census.json", "integrations/vllm/tests/dead_code_census.py"}
    files = [f for f in files if f and not f.endswith(".py") and f"integrations/vllm/{f}" not in self_files]
    out: dict[str, dict] = {}

    def grep(needle: str) -> list[str]:
        r = subprocess.run(["git", "grep", "-l", "-I", "-F", "--", needle], cwd=REPO, capture_output=True, text=True, check=False)
        return [x for x in r.stdout.split("\n") if x and x not in self_files]   # this tool and its census name what they judge

    dir_cache: dict[str, list[str]] = {}
    for f in files:
        self_rel = f"integrations/vllm/{f}"
        base = os.path.basename(f)
        refs = {x for x in grep(base) if x != self_rel}
        dir_refs: dict[str, list[str]] = {}
        parts = f.split("/")[:-1]
        for i in range(len(parts), 0, -1):
            d = "/".join(parts[:i])
            if d in ("", "verity_vllm", "verity_capture", "verity_vllm_numerics", "verity_vllm_adapter", "tests"):
                continue
            if any(p.suffix == ".py" for p in (ROOT / d).iterdir()):
                continue                                             # a code directory being named is not a reference to its data files
            if d not in dir_cache:
                dir_cache[d] = [x for x in grep(d + "/") if not x.startswith(f"integrations/vllm/{d}/")]
            if dir_cache[d]:
                dir_refs[d] = dir_cache[d]
        out[f] = {"bytes": os.path.getsize(ROOT / f), "referenced_by": sorted(refs), "directory_referenced_by": dir_refs}
    return out


def slim_index(arts: dict[str, dict]) -> dict[str, dict]:
    """The committed form: counts + the first few referrers (the full lists are recomputed by running the tool)."""
    return {f: {"bytes": e["bytes"], "n_referenced_by": len(e["referenced_by"]), "referenced_by": e["referenced_by"][:6],
                "directory_referenced": sorted(e["directory_referenced_by"])} for f, e in arts.items()}


def artifact_plan(arts: dict[str, dict], deleted_files: set[str]) -> dict:
    """Fixpoint: an artifact is unreferenced when every file naming it (or an ancestor directory) is itself deleted.  In scope -> delete;
    out of scope -> report.  `deleted_files` are integration-relative paths (modules + tests + artifacts)."""
    gone = {f"integrations/vllm/{p}" for p in deleted_files}
    deleted: dict[str, str] = {}
    changed = True
    while changed:
        changed = False
        for f, e in arts.items():
            if f in deleted or f in ARTIFACT_KEEP or f in ENTRYPOINT_SHELLS:
                continue
            refs = [x for x in e["referenced_by"] if x not in gone]
            drefs = {d: [x for x in xs if x not in gone] for d, xs in e["directory_referenced_by"].items()}
            drefs = {d: xs for d, xs in drefs.items() if xs}
            if refs or drefs:
                continue
            if f.startswith(ARTIFACT_DELETE_SCOPES) or f.endswith(ARTIFACT_DELETE_SUFFIXES):
                deleted[f] = "referenced by nothing" if not e["referenced_by"] else "referenced only by deleted files: " + ", ".join(e["referenced_by"])
                gone.add(f"integrations/vllm/{f}")
                changed = True
    out_of_scope = sorted(f for f, e in arts.items() if f not in deleted and f not in ARTIFACT_KEEP and f not in ENTRYPOINT_SHELLS
                          and not [x for x in e["referenced_by"] if x not in gone]
                          and not any([x for x in xs if x not in gone] for xs in e["directory_referenced_by"].values()))
    # unreferenced by NAME but their directory is named somewhere (data read by directory / glob): listed for the record
    dir_only = sorted(f for f, e in arts.items() if f not in deleted and not e["referenced_by"] and e["directory_referenced_by"]
                      and f not in ARTIFACT_KEEP and f not in ENTRYPOINT_SHELLS)
    return {"deleted": deleted, "deleted_bytes": sum(arts[f]["bytes"] for f in deleted), "kept_by_pattern": ARTIFACT_KEEP,
            "unreferenced_out_of_scope": {f: arts[f]["bytes"] for f in out_of_scope},
            "unreferenced_by_name_directory_referenced": {f: sorted(arts[f]["directory_referenced_by"]) for f in dir_only}}


# ---------------------------------------------------------------------------------------------------------------------------- census
def dedicated(test_rel: str, mod: str) -> bool:
    """`test_<stem>*.py` is the dedicated test of module `<stem>` (or of `<pkg>.<stem>` when the test sits beside the package)."""
    stem = Path(test_rel).stem
    if not stem.startswith("test_"):
        return False
    stem = stem[len("test_"):]
    mstem = mod.rsplit(".", 1)[-1]
    if mstem == "__init__":
        return False
    return stem == mstem or stem.startswith(mstem + "_")


def build(imported: set[str] | None, with_dates: bool) -> dict:
    g = Graph()
    roots_by_mod, roots_by_shell = shell_roots(g)
    roots = set(roots_by_mod)
    for m in ENTRYPOINT_MODULES:
        roots.add(m)
        roots_by_mod[m].add("<python entrypoint>")
    for d in ENTRYPOINT_DIRS:
        for rel, m in g.mod_of.items():
            if rel.startswith(d + "/"):
                roots.add(m)
                roots_by_mod[m].add(f"<{d}>")
    missing = [m for m in roots if not g.exists(m)]
    live = g.closure(roots)
    # the same closure with `verity_capture.profiles.*` cut: a module LIVE only through a profile module is the profiles lane's to decide
    profiles_mods = {m for m in g.file_of if m.startswith("verity_capture.profiles.")}
    live_sans_profiles = g.closure(roots, skip=profiles_mods)
    # the closure without run_config's edges to the stages row_pod.sh skips: what falls out is live only through a skipped stage
    saved = {m: set(g.edges["verity_capture.bench.run_config"]) for m in ("verity_capture.bench.run_config",)}
    g.edges["verity_capture.bench.run_config"] -= set(RUN_CONFIG_SKIPPED_STAGE_MODULES)
    skipped_only = {m for m in live - g.closure(roots) if not is_test_file(g.file_of[m])}
    g.edges["verity_capture.bench.run_config"] = saved["verity_capture.bench.run_config"]
    if imported:
        live |= {m for m in imported if g.exists(m)}
    shell_refs = all_shell_refs(g)

    tests = {rel: m for rel, m in g.mod_of.items() if is_test_file(rel)}
    prod = {rel: m for rel, m in g.mod_of.items() if not is_test_file(rel)}
    # direct imports of integration (production) modules per test file
    test_imports: dict[str, set[str]] = {}
    for rel, m in tests.items():
        test_imports[rel] = {x for x in g.edges.get(m, ()) if not is_test_file(g.file_of[x])}
    imported_by_tests: dict[str, set[str]] = defaultdict(set)
    for rel, mods in test_imports.items():
        for x in mods:
            imported_by_tests[x].add(rel)

    def cls(m: str) -> str:
        if m in live:
            return "LIVE"
        return "TEST-ONLY" if imported_by_tests.get(m) else "DEAD"

    # deletion plan ----------------------------------------------------------------------------------------------------------
    # Modules, tests and artifacts decide each other (a shell kept as an artifact keeps the modules it runs; a deleted test frees the
    # JSON only it named), so the three are iterated to a fixpoint.  `arts` (one git grep per artifact) is computed once.
    arts = artifact_refs()
    fixtures = {m for m in KEEP_FIXTURES if g.exists(m) and m not in live}
    unused_keeps = sorted(m for m in KEEP_FIXTURES if not g.exists(m) or m in live)
    deleted_shells: set[str] = set()
    shell_only: dict[str, list[str]] = {}
    while True:
        # 1. production modules: non-LIVE, not protected, not kept on purpose, not run by a shell script that stays -> delete
        deleted_mods: set[str] = set()
        shell_only = {}
        for rel, m in prod.items():
            if m in live or protected(rel) or m in KEEP_FIXTURES:
                continue
            kept_shells = sorted(s for s in shell_refs.get(m, ()) if s not in deleted_shells)
            if kept_shells:
                shell_only[rel] = kept_shells
                continue
            deleted_mods.add(m)
        # 2. tests: a test file (or a helper under a tests dir) goes when it imports a deleted module, directly or through another
        #    deleted test file (fixpoint); the harness dir never goes
        deleted_tests: dict[str, str] = {}
        changed = True
        while changed:
            changed = False
            for rel, m in tests.items():
                if rel in deleted_tests or any(rel.startswith(d + "/") for d in ENTRYPOINT_DIRS) or Path(rel).name == "__init__.py" \
                        or rel in TRIMMED_TESTS:
                    continue
                gone_prod = sorted(x for x in g.edges.get(m, ()) if x in deleted_mods)
                gone_tests = sorted(x for x in g.edges.get(m, ()) if g.file_of[x] in deleted_tests)
                if gone_prod or gone_tests:
                    deleted_tests[rel] = "imports deleted " + ", ".join(gone_prod + [g.file_of[x] for x in gone_tests])
                    changed = True
        # 3. artifacts, given the files above; the shells among them feed step 1
        deleted_files = {g.file_of[m] for m in deleted_mods} | set(deleted_tests)
        aplan = artifact_plan(arts, deleted_files)
        new_shells = {f for f in aplan["deleted"] if f.endswith(".sh")}
        if new_shells == deleted_shells:
            break
        deleted_shells = new_shells
    kept_tests = {rel for rel in tests if rel not in deleted_tests}
    dedicated_pairs = [{"test": rel, "modules": sorted(x for x in test_imports[rel] if x in deleted_mods and dedicated(rel, x))}
                       for rel in sorted(deleted_tests) if any(x in deleted_mods and dedicated(rel, x) for x in test_imports[rel])]
    # 4. consistency: nothing kept imports anything deleted (the graph says the tree still imports after the plan is applied)
    broken = []
    for rel, m in g.mod_of.items():
        if m in deleted_mods or rel in deleted_tests:
            continue
        for x in g.edges.get(m, ()):
            if x in deleted_mods or g.file_of[x] in deleted_tests:
                broken.append((rel, x))
    kept_helpers = sorted(rel for rel in kept_tests if not Path(rel).name.startswith("test_") and Path(rel).name != "__init__.py")

    dates = git_dates(sorted(g.mod_of)) if with_dates else {}
    modules: dict[str, dict] = {}
    for rel, m in sorted(prod.items()):
        top = rel.split("/")[0]
        if top not in CENSUS_PACKAGES + ("verity_vllm_adapter",):
            continue
        c = cls(m)
        entry = {
            "module": m,
            "package": top,
            "class": c,
            "static_reachable": m in live and (imported is None or m not in imported or m in g.closure(roots)),
            "dynamically_imported": bool(imported and m in imported),
            "imported_by_tests_only": c == "TEST-ONLY",
            "n_imported_by_tests": len(imported_by_tests.get(m, ())),
            "imported_by_tests": sorted(imported_by_tests.get(m, ()))[:8],
            "loc": loc(ROOT / rel),
            "last_commit_date": dates.get(rel),
            "referenced_by_shell": sorted(shell_refs.get(m, ())),
            "root_via": sorted(roots_by_mod.get(m, ())),
            "protected": protected(rel),
            "live_only_via_profiles": (m in live and m not in live_sans_profiles and m not in profiles_mods),
            "why": g.why(roots, m) if m in live else None,
        }
        if c != "LIVE":
            entry["plan"] = ("protected" if entry["protected"] else "keep" if m in fixtures else "keep-shell-only" if rel in shell_only
                             else "delete")
            if m in fixtures:
                entry["kept_because"] = KEEP_FIXTURES[m]
        modules[rel] = entry

    static_only = g.closure(roots)
    dyn_not_static = sorted(m for m in (imported or ()) if g.exists(m) and m not in static_only and not is_test_file(g.file_of[m]))

    # totals
    totals: dict[str, dict] = defaultdict(lambda: {"files": 0, "loc": 0})
    for rel, e in modules.items():
        for key in (("all", e["class"]), (e["package"], e["class"]), ("all", "TOTAL"), (e["package"], "TOTAL")):
            k = f"{key[0]}::{key[1]}"
            totals[k]["files"] += 1
            totals[k]["loc"] += e["loc"]
    plan_totals = defaultdict(lambda: {"files": 0, "loc": 0})
    for rel, e in modules.items():
        if e.get("plan") == "delete":
            plan_totals[e["package"]]["files"] += 1
            plan_totals[e["package"]]["loc"] += e["loc"]
            plan_totals["all"]["files"] += 1
            plan_totals["all"]["loc"] += e["loc"]
    for rel in deleted_tests:
        plan_totals["tests"]["files"] += 1
        plan_totals["tests"]["loc"] += loc(ROOT / rel)

    return {
        "schema": "vllm-dead-code-census/v1",
        "root": "integrations/vllm",
        "entrypoints": {"shells": list(ENTRYPOINT_SHELLS), "modules": list(ENTRYPOINT_MODULES), "dirs": list(ENTRYPOINT_DIRS),
                        "roots_from_shells": {sh: sorted(ms) for sh, ms in sorted(roots_by_shell.items())},
                        "roots_missing": sorted(missing), "n_roots": len(roots)},
        "dynamic_record": {"present": imported is not None, "n_modules": len(imported or ()), "dynamic_not_static": dyn_not_static},
        "totals": dict(sorted(totals.items())),
        "plan_totals": dict(sorted(plan_totals.items())),
        "modules": modules,
        "tests": {"kept": sorted(kept_tests), "deleted": dict(sorted(deleted_tests.items())), "kept_helpers": kept_helpers,
                  "dedicated_pairs": dedicated_pairs},
        "fixtures_kept": {m: KEEP_FIXTURES[m] for m in sorted(fixtures)},
        "kept_shell_only": dict(sorted(shell_only.items())),
        "keep_entries_unused": unused_keeps,
        "plan_consistency_broken_imports": broken,
        "trimmed_tests": TRIMMED_TESTS,
        "protected_non_live": sorted(rel for rel, e in modules.items() if e["protected"] and e["class"] != "LIVE"),
        "live_only_via_run_config_skipped_stages": {g.file_of[m]: loc(ROOT / g.file_of[m]) for m in sorted(skipped_only)},
        "live_only_via_profiles": sorted(rel for rel, e in modules.items() if e["live_only_via_profiles"]),
        "artifacts": {**aplan, "index": slim_index(arts)},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--imported", default=None, help="imported.json from the pod (a list or {module: ...} of sys.modules names)")
    ap.add_argument("--out", default=str(HERE / "dead_code_census.json"))
    ap.add_argument("--git-dates", action="store_true", help="record last_commit_date per file (one git log per file)")
    a = ap.parse_args(argv)
    imported: set[str] | None = None
    if a.imported:
        with open(a.imported) as f:
            doc = json.load(f)
        mods = doc.get("modules", doc) if isinstance(doc, dict) else doc
        imported = set(mods if isinstance(mods, list) else mods.keys())
    census = build(imported, a.git_dates)
    with open(a.out, "w") as f:
        json.dump(census, f, indent=1, sort_keys=False)
        f.write("\n")
    t = census["totals"]
    for pkg in ("all",) + CENSUS_PACKAGES + ("verity_vllm_adapter",):
        row = "  ".join(f"{c}: {t.get(f'{pkg}::{c}', {}).get('files', 0)} files / {t.get(f'{pkg}::{c}', {}).get('loc', 0)} loc"
                        for c in ("LIVE", "TEST-ONLY", "DEAD", "TOTAL"))
        print(f"{pkg:24s} {row}")
    print(f"plan: delete {census['plan_totals']}")
    print(f"tests: keep {len(census['tests']['kept'])}, delete {len(census['tests']['deleted'])}; fixtures kept {len(census['fixtures_kept'])}; "
          f"roots {census['entrypoints']['n_roots']} (missing {census['entrypoints']['roots_missing']}); dynamic-not-static {census['dynamic_record']['dynamic_not_static']}")
    ar = census["artifacts"]
    print(f"artifacts: delete {len(ar['deleted'])} files / {ar['deleted_bytes']} bytes; unreferenced out of scope {len(ar['unreferenced_out_of_scope'])}; "
          f"broken imports after plan: {census['plan_consistency_broken_imports']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
