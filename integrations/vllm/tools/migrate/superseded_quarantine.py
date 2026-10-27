"""Retire superseded catalog Definition versions, and move the vLLM registry's quarantine to `verity/experimental/`.

    uv run python integrations/vllm/tools/migrate/superseded_quarantine.py apply [--only ID ...] [--dest DIR | --promote-only]
    uv run python integrations/vllm/tools/migrate/superseded_quarantine.py commands
    uv run python integrations/vllm/tools/migrate/superseded_quarantine.py snapshot OUT.json
    uv run python integrations/vllm/tools/migrate/superseded_quarantine.py verify BEFORE.json AFTER.json

The layout it moves toward: `verity/experimental/` is a top-level directory, `verity/` holds only what cited claims depend on, and a superseded catalog Definition version stays only so
that old records replay. A superseded version that no kept Program or record names may be deleted, but only after a
`circuits/<slug>` superseded approach entry citing the last commit that has it is recorded (`commands` prints those entries).

`apply` deletes exactly the versions in `DELETE`, with their circuit-check bindings (`_model_roots`) and lowering pins
(`pins.json`), and refuses any other id, any id in `KEPT`, and any deletion that would leave a reference behind. It then moves
the quarantine lanes in `MOVE` to `<DEST>/<PACKAGE>/<lane>` (`--dest`, by default `experimental/vllm/quarantine`; the package's
import name is the one constant `PACKAGE`, and its distribution's name is derived from it), rewrites their imports, test paths
and the lint and dead-code lists that name them, and registers the new workspace member. Before that it promotes the lanes core
binds into the registry as ordinary Definitions (`PROMOTE`): each
module's Definitions keep their bodies verbatim, so every id and descriptor digest stays what it was; the code that imports them
imports the registry module instead, the lists keyed by them follow them or drop them, the target facts they carry out of P8's
home are allowlisted where they now live (`P08_ADDED`), and an emptied lane package goes; core's quarantine package stays, the
admission home of any later lane, kept by name in the dead-code list. Every
edit is planned, and refused if it would leave a name of a promoted lane behind, before anything is written. Running it again
changes nothing. `--promote-only` deletes and promotes but moves nothing: the `MOVE` lanes stay in core's quarantine, for a tree
that has no `verity/experimental/` yet.

`snapshot` records circuit-check's task list and every registered Definition, each with its descriptor digest (sha512 of the
canonical JSON of its definition entry). `verify` compares two snapshots: the task list may lose only the deleted versions'
targets and gain only what the promoted Definitions' bindings reach, and every other id must keep its digest.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.util
import json
import pkgutil
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parents[4]
VLLM = "integrations/vllm"
REGISTRY_DIR = f"{VLLM}/verity_vllm/program/registry"
QUARANTINE = f"{REGISTRY_DIR}/quarantine"
QUARANTINE_MODULE = "verity_vllm.program.registry.quarantine"
REGISTRY_MODULE = "verity_vllm.program.registry"
#: Where `apply` puts the experimental package unless `--dest` names another directory.
DEST = "experimental/vllm/quarantine"
#: The experimental package's import name. Everything else that names it (its distribution, its docs, the rewritten imports of
#: the moved lanes, the lists keyed by them) is derived from this one constant.
PACKAGE = "verity_quarantine"
DISTRIBUTION = PACKAGE.replace("_", "-")
TARGETS = "tools/verity/circuit_check/src/circuit_check/targets.py"
PINS = "tools/verity/circuit_check/src/circuit_check/pins.json"
#: Where Definitions are registered: a version is deleted from one of these.
DEFINITION_DIRS = (REGISTRY_DIR, "verity")
#: JSON lists keyed by a moved file or module: their entries for the moved lanes are dropped (each list only shrinks).
KEYED_LISTS = (f"{VLLM}/tests/lint/allowlists/p10_size.json", f"{VLLM}/tests/lint/allowlists/p11_names.json",
               f"{VLLM}/tests/dead_code_keep.json")
#: Never rewritten: this script, its test, and the quarantine's recorded admission records (their adapter names are history).
UNTOUCHED = (f"{VLLM}/tools/migrate/", f"{VLLM}/tests/migrate/", f"{VLLM}/verity_vllm/engine/profiles/quarantine/")


@dataclass(frozen=True)
class Retirement:
    slug: str            # the circuits/<slug> approach entry recorded as superseded
    superseded_by: str   # the circuits/<slug> of the approach that replaces it
    reason: str          # one sentence
    cite: str            # verity@<sha>: the last commit that has the version


#: The only versions `apply` deletes, by id. Empty: every superseded version reviewed is kept (`KEPT`).
DELETE: dict[str, Retirement] = {}

#: The superseded versions reviewed and kept, each with what keeps it (a Build vocabulary, a record, a capture map, a pinned
#: identity, or a kept Program that calls it). `apply` refuses to delete any of them.
KEPT: dict[str, str] = {
    "RMSNormFusedCuda_v1": "capture map integrations/vllm/manifests/capture-maps/glm-moe-dsa-eager-hopper-tp8-v0.json:243 cites it; "
                           "its divergence record (verity_vllm/program/registry/conformance.py:40, fixtures/verity-ir/rmsnorm-fused-b0); "
                           "the width-8 target spec (tests/program/test_rmsnorm_fused_target.py:32)",
    "SiluMul_v1": "the b1-eager and b1-eager-v2 vocabularies' silu_mul kind (verity_vllm/program/frontend/rules/vocab.py:106, :207); "
                  "the fold's silu_and_mul pattern (verity_vllm/observe/fold/patterns/activations.py:37); census/subcircuits.json:832",
    "SiluMulBf16_v1": "SiluMul_v1's element (verity_vllm/program/registry/prims.py:66); the word view of SiluMulBf16_v3 "
                      "(verity_vllm/program/registry/boolean_silu.py:139); D-SP1's frozen gate set (archive/sp1/common/src/gateset.rs:140)",
    "SiluMul_v3": "SiluMul_v1's Boolean form, which the Boolean switch binds for it (verity_vllm/program/registry/boolean_silu.py:148)",
    "SiluMulBf16_v3": "SiluMulBf16_v1's Boolean form, SiluMul_v3's element (verity_vllm/program/registry/boolean_silu.py:139)",
    "Attention_v1": "the b1-eager vocabulary's attention kind (verity_vllm/program/frontend/rules/vocab.py:105); "
                    "FLASH_ATTENTION_LAUNCHES (verity_vllm/query/required.py:92)",
    "AttentionHead_v1": "Attention_v1's callee (verity_vllm/program/registry/b1.py:472)",
    "AttnBlock_v1": "AttentionHead_v1's callee (verity_vllm/program/registry/b1.py:414)",
    "Serve_v1": "Family[Serve_v1], the certificate of record's program identity (verity_vllm/program/frontend/rules/family.py:487); "
                "ServeHopper_v1 and ServeFp8_v1 are ports of it (verity_vllm/program/registry/hopper.py:153, fp8.py:615)",
    "Serve_v2": "the b1-eager-v2 skeleton (verity_vllm/program/frontend/rules/vocab.py:179); the Hopper eager program "
                "(verity_vllm/program/registry/hopper.py:177); LServe_v1's skeleton (verity_vllm/program/registry/lifted.py:1362)",
    "Serve_v3": "Family[Serve_v3], the padded circuit derivation verifies against (verity_vllm/program/frontend/rules/family.py:517)",
    "LayerPre_v1": "Serve_v1's callee (verity_vllm/program/frontend/rules/family.py:545); LayerPreHopper_v1 and LayerPreFp8_v1 port it "
                   "(verity_vllm/program/registry/hopper.py:148, fp8.py:621)",
    "LayerPost_v1": "Serve_v1's callee (verity_vllm/program/frontend/rules/family.py:551); LayerPostHopper_v1 and LayerPostFp8_v1 port it "
                    "(verity_vllm/program/registry/hopper.py:149, fp8.py:622)",
    "Final_v1": "Serve_v1's callee (verity_vllm/program/registry/b1.py:646); FinalHopper_v1 ports it (verity_vllm/program/registry/hopper.py:150)",
}

#: The quarantine lanes that move: nothing in core's library code imports them.
MOVE = ("act", "collective", "dense", "ov_moe")


@dataclass(frozen=True)
class Promotion:
    """A quarantine module whose Definitions become registry ones.  Where `dst` does not exist the module moves there, with
    `edits` (exact replacements, each of which must match) applied; otherwise its statements after its imports, but for its
    `SEMANTIC_ID` and `__all__`, go into `dst` before `dst`'s `__all__` (or at its end), and the `imports` `dst` lacks join its
    imports.  A merged module's `X.name`, for an alias `X` in `unqualify`, becomes `name`, which `dst` must bind (the sampling
    family's split left `sampling` importing each primitive by name from `verity.ml.circuits.ops`, not through `registry.prims`)."""
    src: str                                   # under QUARANTINE
    dst: str                                   # under REGISTRY_DIR
    edits: tuple[tuple[str, str], ...] = ()
    imports: tuple[str, ...] = ()
    unqualify: tuple[str, ...] = ()


#: The lanes core binds, promoted into the registry beside their families (each passes circuit-check at its catalog binding):
#: LayerNorm beside the norms, GeluErf in `act`, the static-seed Gumbel argmax in `sampling` with the operand-seed select.
PROMOTE: tuple[Promotion, ...] = (
    Promotion("ln/layer_norm_aten.py", "layer_norm.py", edits=(
        ("from verity_vllm.program.registry.quarantine.ln._common import BF16, F32, ZERO32, Array, P, f32c\n",
         "from verity.ml.circuits.types import Array\nfrom verity_vllm.program.registry import prims as P\n"
         "from verity_vllm.program.registry.b1 import ZERO32, f32c\nfrom verity_vllm.program.registry.prims import BF16, F32\n"),
        ("The lane's difftest (`layer_norm_aten_difftest.py`) is the bit-exact check.",
         "Its difftest (`layer_norm_difftest.py`) is the bit-exact check."))),
    Promotion("ln/layer_norm_aten_difftest.py", "layer_norm_difftest.py"),
    Promotion("ln/gelu_erf_table.py", "gelu_erf_table.py", edits=(("(see `gelu_erf_bf16.py`)", "(see `act.GeluErfBf16`)"),)),
    Promotion("ln/gelu_erf_bf16.py", "act.py", imports=("from verity_vllm.program.registry.gelu_erf_table import TABLE_WORDS",)),
    Promotion("ln/gelu_erf.py", "act.py"),
    Promotion("ov_sampling/gumbel.py", "sampling.py", unqualify=("P",)),
)
#: What `PROMOTE`'s modules register: `apply` refuses when they declare anything else, and `verify` lets circuit-check's task list
#: gain what their bindings reach.
PROMOTED_IDS = ("LayerNormAten_v1", "LayerNormAten_v2", "GeluErfBf16_v1", "GeluErf_v1", "GumbelArgmaxStep_v1", "GumbelArgmaxBf16_v1",
                "GumbelArgmaxTempStep_v1", "GumbelArgmaxTempBf16_v1")
PROMOTED_LANES = tuple(dict.fromkeys(p.src.split("/")[0] for p in PROMOTE))
#: A lane package's own files, deleted with it once every other module of the lane is promoted (its lists and shared imports).
LANE_FILES = ("__init__.py", "_common.py")
P08 = f"{VLLM}/tests/lint/allowlists/p08_facts.json"
#: The target facts the promoted conformance strings and the GELU table's provenance carry out of P8's home
#: (`program/registry/quarantine/`), allowlisted where they now live, as the promoted `dense` lane's are.
P08_ADDED = tuple({"file": f"verity_vllm/program/registry/{f}", "kind": k, "symbol": s, "detail": d} for f, k, s, d in (
    ("act.py", "sm-arch", "GeluErf", "sm_89"), ("act.py", "sm-arch", "GeluErfBf16", "sm_89"),
    ("gelu_erf_table.py", "gpu", "<module>", "rtx4090"), ("layer_norm.py", "sm-arch", "LayerNormAten", "sm_89")))

def redirects(dest: str) -> tuple[tuple[str, str, str], ...]:
    """Lines of core tests that name a moved lane by where it was: a use of a re-export of a promoted Definition imports the
    registry module that owns it, and a lint of the lane's directory lints the experimental package."""
    return tuple((f"{VLLM}/tests/{f}", old, new) for f, old, new in (
        ("program/fixtures_query.py", "from verity_vllm.program.registry.quarantine.ov_moe import moe as Q",
         "from verity_vllm.program.registry import moe as Q"),
        ("program/_wout_prototype_20260917.py", "from verity_vllm.program.registry.quarantine.ov_moe import moe as Q",
         "from verity_vllm.program.registry import moe as Q"),
        ("query/test_boundary_oracle.py", "from verity_vllm.program.registry.quarantine.ov_moe import moe as Q",
         "from verity_vllm.program.registry import moe as Q"),
        ("program/test_gen_ov_moe_vu_shapes.py", "from verity_vllm.program.registry.quarantine.ov_moe.moe import MoeExpertGemm",
         "from verity_vllm.program.registry.moe import MoeExpertGemm"),
        ("collectives/test_tp_collective.py", 'rep = lint(REPO / "verity_vllm" / "program" / "registry" / "quarantine")',
         "rep = lint(Path(QC.__file__).resolve().parents[1])"),
    )) + (
        # the shipped PYTHONPATH's roots follow the root's dependencies, which now include the experimental member
        ("tools/research/tests/test_pythonpath.py", '"verity/ml/flock/python", "tools/verity/circuit_check/src"]',
         f'"verity/ml/flock/python", "tools/verity/circuit_check/src",\n              "{dest}"]'),
    )


PYPROJECT = f'''[project]
name = "{DISTRIBUTION}"
version = "0.1.0"
description = "Experimental: the vLLM registry's quarantine lanes that core does not bind (provisional Definitions, the re-exports of promoted ones, their difftest adapters)."
requires-python = ">=3.12"
dependencies = ["verity", "verity-vllm"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["{PACKAGE}"]

[tool.uv.sources]
verity = {{ workspace = true }}
verity-vllm = {{ workspace = true }}

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q -p no:cacheprovider"
'''

def _lanes(lanes) -> str:
    return ", ".join(f"`{m}`" for m in lanes)


PACKAGE_DOC = f'''"""{PACKAGE} — the vLLM registry's quarantine lanes that core does not bind, moved out of
`{QUARANTINE_MODULE}`: {_lanes(MOVE)}.
The lanes core bound ({_lanes(PROMOTED_LANES)}) were promoted into `{REGISTRY_MODULE}`.

Each lane is a sub-package of provisional numerical Definitions pending admission, or of re-exports of Definitions already promoted
to `verity_vllm.program.registry`, with a `*_difftest.py` adapter for `verity-vllm properties-admission` beside each. Admission
is as in core's quarantine (`verity_vllm.properties.quarantine_lint`, `verity_vllm.properties.admission`): a body is a function of
its operands and statics only, each module names its `SEMANTIC_ID = "<lane>/<Definition>@v<k>"`, a re-derived body takes a new
version, and promotion into the registry is an integrator move. Nothing in core imports this package.
"""
'''

def core_doc(dest: str) -> str:
    promoted = _lanes(dict.fromkeys(p.dst.removesuffix(".py") for p in PROMOTE))
    return f'''"""{QUARANTINE_MODULE} — provisional numerical Definitions that core binds, one sub-package per lane, pending
admission. No lane is pending. The lanes core bound ({_lanes(PROMOTED_LANES)}) were promoted into `registry`:
{promoted}.
The lanes core does not bind ({_lanes(MOVE)}) are the experimental package `{PACKAGE}` (`{dest}`).
Existing registered bodies (`b1.py`, `prims.py`, ...) are immutable; a lane that needs different exact semantics adds a NEW
versioned Definition and never edits an old one.

Admission (gate G4 + `verity_vllm.properties.quarantine_lint` + `verity_vllm.properties.admission`): the body is a function of its
operands and statics only; `SEMANTIC_ID = "<lane>/<Definition>@v<k>"` at module level; the lane's pattern has positive and negative
recognizer tests; `difftest produce` + `difftest check` are bit-exact on randomized and edge inputs; every specialization the run
uses is replayed. Promotion into `registry` is an integrator move.
"""
'''


class Refused(Exception):
    pass


# --- shared ------------------------------------------------------------------------------------------------------------

def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout


def tracked(root: Path) -> list[str]:
    return [p for p in _git(root, "ls-files", "-z").split("\0") if p and (root / p).is_file()]


def family(target: str) -> str:
    """`Name_vN{...}` and `Name_vN/label` belong to `Name_vN`."""
    return re.split(r"[{/]", target, maxsplit=1)[0]


def _drop_lines(text: str, drop) -> str:
    """``text`` without the lines ``drop`` selects, and without a trailing comma left before a closing bracket."""
    out: list[str] = []
    for line in text.splitlines(keepends=True):
        if drop(line):
            continue
        if line.lstrip().startswith(("]", "}")) and out and out[-1].rstrip().endswith(","):
            out[-1] = out[-1].rstrip()[:-1] + "\n"
        out.append(line)
    return "".join(out)


def _imports(path: Path) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            out.add(node.module)
            out.update(f"{node.module}.{a.name}" for a in node.names)
        elif (isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)
              and (node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)) in ("import_module", "__import__")):
            out.add(node.args[0].value)
    return out


def library_importers(root: Path, prefixes: tuple[str, ...]) -> list[str]:
    """Core's vLLM library files (outside the quarantine and the tests) that import a module under one of ``prefixes``."""
    out = []
    for p in sorted((root / VLLM / "verity_vllm").rglob("*.py")):
        rel = p.relative_to(root).as_posix()
        if rel.startswith(QUARANTINE + "/") or "__pycache__" in rel:
            continue
        hit = sorted(m for m in _imports(p) if any(m == x or m.startswith(x + ".") for x in prefixes))
        if hit:
            out.append(f"{rel}: {', '.join(hit)}")
    return out


# --- deleting versions -------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Site:
    path: str
    start: int           # 1-based, first decorator included
    end: int             # inclusive
    names: tuple[str, ...]


def _declares(node: ast.AST, name: str, version: int) -> bool:
    calls = [d for d in getattr(node, "decorator_list", []) if isinstance(d, ast.Call)]
    if isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(node.value, ast.Call):
        calls.append(node.value)
    return any(len(c.args) >= 2 and isinstance(c.args[0], ast.Constant) and c.args[0].value == name
               and isinstance(c.args[1], ast.Constant) and c.args[1].value == version for c in calls)


def _bound(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        return (node.name,)
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    return tuple(n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name))


def find_sites(root: Path, version_id: str) -> list[Site]:
    m = re.fullmatch(r"(\w+)_v(\d+)", version_id)
    if not m:
        raise Refused(f"{version_id}: not a Definition version id (Name_vN)")
    name, version = m.group(1), int(m.group(2))
    sites = []
    for d in DEFINITION_DIRS:
        for p in sorted((root / d).rglob("*.py")) if (root / d).is_dir() else []:
            rel = p.relative_to(root).as_posix()
            if "__pycache__" in rel or rel.startswith(QUARANTINE + "/"):
                continue
            for node in ast.parse(p.read_text(encoding="utf-8")).body:
                if _declares(node, name, version):
                    first = min([node.lineno, *(x.lineno for x in getattr(node, "decorator_list", []))])
                    sites.append(Site(rel, first, node.end_lineno, _bound(node)))
    return sites


def _target_entries(text: str, version_id: str) -> list[tuple[int, int]]:
    """Line spans (1-based, inclusive) of `_model_roots`' entries labelled ``version_id``, each with the comment lines above it."""
    tree = ast.parse(text)
    lines = text.splitlines()
    spans = []
    for fn in ast.walk(tree):
        if not (isinstance(fn, ast.FunctionDef) and fn.name == "_model_roots"):
            continue
        for ret in ast.walk(fn):
            if not (isinstance(ret, ast.Return) and isinstance(ret.value, ast.List)):
                continue
            elts = ret.value.elts
            for i, e in enumerate(elts):
                if not (isinstance(e, ast.Tuple) and e.elts and isinstance(e.elts[0], ast.Constant)
                        and isinstance(e.elts[0].value, str) and family(e.elts[0].value) == version_id):
                    continue
                shared = {x.lineno for x in elts[:i] + elts[i + 1:]} | {x.end_lineno for x in elts[:i] + elts[i + 1:]}
                if set(range(e.lineno, e.end_lineno + 1)) & shared:
                    raise Refused(f"{TARGETS}:{e.lineno}: the binding of {version_id} shares a line with another entry")
                start = e.lineno
                while start > 1 and lines[start - 2].strip().startswith("#"):
                    start -= 1
                spans.append((start, e.end_lineno))
    return spans


def plan_deletions(root: Path, ids: list[str]) -> dict[str, list[tuple[int, int]]]:
    """{file: [line spans to remove]}, after checking each id is listed, located once, and referenced nowhere else: neither its
    id in any tracked file nor, in a Python file, the name it is bound to (another module's object of the same name refuses
    too; resolve such a refusal by hand)."""
    unlisted = [i for i in ids if i not in DELETE]
    if unlisted:
        raise Refused(f"not in DELETE: {', '.join(unlisted)}; only the versions listed in this script are ever deleted")
    kept = [i for i in ids if i in KEPT]
    if kept:
        raise Refused(f"listed in both DELETE and KEPT: {', '.join(kept)}")
    spans: dict[str, list[tuple[int, int]]] = {}
    names: dict[str, tuple[str, ...]] = {}
    for vid in ids:
        sites = find_sites(root, vid)
        if len(sites) != 1:
            raise Refused(f"{vid}: expected one registration, found {len(sites)}: {[f'{s.path}:{s.start}' for s in sites]}")
        s = sites[0]
        spans.setdefault(s.path, []).append((s.start, s.end))
        names[vid] = s.names
        targets = root / TARGETS
        if targets.is_file():
            spans.setdefault(TARGETS, []).extend(_target_entries(targets.read_text(encoding="utf-8"), vid))
    residue = []
    for rel in tracked(root):
        if rel.startswith(UNTOUCHED) or rel == PINS:
            continue
        try:
            text = (root / rel).read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        removed = {n for a, b in spans.get(rel, []) for n in range(a, b + 1)}
        for n, line in enumerate(text.splitlines(), 1):
            if n in removed:
                continue
            for vid in ids:
                words = [vid, *names[vid]] if rel.endswith(".py") else [vid]
                hit = [w for w in words if re.search(rf"\b{re.escape(w)}\b", line)]
                if hit:
                    residue.append(f"{rel}:{n}: {', '.join(hit)}")
    if residue:
        raise Refused("still referenced, so not deletable (a reference that keeps it belongs in KEPT):\n  " + "\n  ".join(residue))
    return spans


def delete_versions(root: Path, ids: list[str]) -> list[str]:
    if not ids:
        return []
    spans = plan_deletions(root, ids)
    changed = []
    for rel, ss in spans.items():
        p = root / rel
        lines = p.read_text(encoding="utf-8").splitlines(keepends=True)
        removed = {n for a, b in ss for n in range(a, b + 1)}
        kept = [line for n, line in enumerate(lines, 1) if n not in removed]
        text = re.sub(r"\n{4,}", "\n\n\n", "".join(kept))
        if rel.endswith(".py"):
            ast.parse(text)
        p.write_text(text, encoding="utf-8")
        changed.append(rel)
    pins = root / PINS
    if pins.is_file():
        text = pins.read_text(encoding="utf-8")
        data = json.loads(text)
        if json.dumps(data, indent=1, sort_keys=True) + "\n" != text:
            raise Refused(f"{PINS} does not round-trip through json.dumps(indent=1, sort_keys=True); not rewriting it")
        before = len(data["definitions"])
        data["definitions"] = {k: v for k, v in data["definitions"].items() if family(k) not in ids}
        if len(data["definitions"]) != before:
            pins.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="utf-8")
            changed.append(PINS)
    return changed


# --- promoting lanes into the registry ----------------------------------------------------------------------------------

GELU_ERF_DOC = """

`GeluErfBf16_v1` is one output word of `aten.gelu(x, approximate='none')` on bf16 (torch 2.13.0+cu129
`aten/src/ATen/native/cuda/ActivationGeluKernel.cu`, `GeluCUDAKernelImpl`, `opmath_t = float`):

    return static_cast<opmath_t>(x) * opmath_t(0.5) * (opmath_t(1) + ::erf(static_cast<opmath_t>(x) * kAlpha));   // kAlpha = M_SQRT1_2

i.e. `bf16( (f * 0.5f) * (1.0f + erff(f * 0x3F3504F3)) )`, `f = (float)x`, stored through `c10::BFloat16(float)` (`cvt.rn.bf16.f32`,
NaN -> 0x7FFF).  `erff` is CUDA libdevice's `__nv_erff` (a minimax polynomial pair with FMAs, not correctly rounded), so the
word is NOT `RN(gelu(x))` for every x; rather than transcribe libdevice, the function is pinned by its **exhaustive table**: the
domain is the 65,536 bf16 words, and `gelu_erf_table.py` holds the kernel's output word for every one of them, measured on the
pod (RTX 4090, sm_89, torch 2.13.0+cu129) by `verity_vllm/ops/gelu_erf_table_probe.py` (`aten.gelu` on
`arange(65536).view(bfloat16)`, then zlib + base64).  The primitive is the table lookup.  Precedent: `prims.MufuEx2Ftz` /
`MufuRcpFtz` (measured hardware tables).  `GeluErf_v1{N}` is `out[j] = GeluErfBf16(x[j])` over a row, the GPT-NeoX MLP
activation (`nn.GELU()`, vLLM `get_act_fn("gelu")`).  Both came from the quarantine lane `cov-ln` with their bodies verbatim,
so their ids are the lane's; the Match fold binds them (`engine/profiles/gen_ln_patterns.py`), and `tests/engine/test_gen_ln.py`
checks the table's sanity facts: gelu(+0) = +0, gelu(-0) = -0, gelu(x) = x for x >= 2^4, gelu(x) = -0 for x <= -2^4, NaN words
-> 0x7FFF, +inf -> +inf, -inf -> NaN."""


def promote_edits(dest: str | None) -> tuple[tuple[str, str, str], ...]:
    """(file, old, new): the hand edits of a promotion, applied after the modules move and merge and before the import rewrite.
    Each ``old`` occurs exactly once while the promotion is pending. ``dest`` None: the `MOVE` lanes stay in core's quarantine."""
    reg = REGISTRY_DIR.removeprefix(VLLM + "/")
    kept_by = "the experimental `act` modules" if dest else "the quarantine `act` modules"
    residue = (f"it holds no lane: those core\nbinds were promoted into `registry/`, the rest are `{PACKAGE}` under `{dest}`" if dest
               else "the lanes core\nbinds were promoted into `registry/`")
    edits = tuple((f"{VLLM}/{f}" if not f.startswith(("tools/", "integrations/")) else f, old, new) for f, old, new in (
        (f"{reg}/act.py", "  _C.silu_and_mul   act_and_mul_kernel<bf16, silu_kernel>   ->  SiluMul_v2{I} over SiluMulBf16_v2\n",
         "  _C.silu_and_mul   act_and_mul_kernel<bf16, silu_kernel>   ->  SiluMul_v2{I} over SiluMulBf16_v2\n"
         "  aten.gelu (erf)   torch's GeluCUDAKernelImpl              ->  GeluErf_v1{N} over GeluErfBf16_v1\n"),
        (f"{reg}/act.py", "stand-in for expf + div.rn.\n\"\"\"", "stand-in for expf + div.rn." + GELU_ERF_DOC + "\n\"\"\""),
        (f"{reg}/act.py",
         "#: `SEMANTIC_ID`s of the promoted Definitions (quarantine identity; the quarantine modules keep the same strings)\n"
         'SEMANTIC_IDS = {"SiluMulBf16": "act/SiluMulBf16@v2", "SiluMul": "act/SiluMul@v2"}',
         f"#: `SEMANTIC_ID`s of the promoted Definitions (their lanes' identities; {kept_by} keep the same strings)\n"
         'SEMANTIC_IDS = {"SiluMulBf16": "act/SiluMulBf16@v2", "SiluMul": "act/SiluMul@v2", "GeluErfBf16": "ln/GeluErfBf16@v1",\n'
         '                "GeluErf": "ln/GeluErf@v1"}'),
        (f"{reg}/sampling.py", "RTX 4090 in `registry/quarantine/ov_sampling/gumbel.py`, 64/64 G4 cases, which now imports them from here):",
         "RTX 4090 by the static-seed composites below, 64/64 G4 cases):"),
        (f"{reg}/sampling.py", "registered greedy `TokenSelect_v1{V}` (first maximum).\n",
         "registered greedy `TokenSelect_v1{V}` (first maximum).\n"
         "  * `GumbelArgmaxBf16_v1{V, SEED, POS}(logits: Array<V, bf16>) -> i32` and `GumbelArgmaxTempBf16_v1{V, SEED, POS, T}` (steps\n"
         "    `GumbelArgmaxStep_v1`, `GumbelArgmaxTempStep_v1{T}`) — the same selection at temperature 1 and at a T not in {0, 1}, with\n"
         "    seed, position and T as STATICS: the forms checked bit-exact against vLLM's kernels on the RTX 4090, which the sampled-row\n"
         "    profiles bind (`engine/profiles/gen_ov_sampling_patterns.py`).  They came from the quarantine lane `ov-sampling`\n"
         "    (`SEMANTIC_ID` \"ov-sampling/GumbelArgmaxBf16@v1\") with their bodies verbatim, so their ids are the lane's.\n"),
        (f"{reg}/sampling.py", '"nv_log1pf_gpu", "noise_lane_gpu"]',
         '"nv_log1pf_gpu", "noise_lane_gpu",\n'
         '           # the static-seed forms\n'
         '           "GumbelArgmaxStep", "GumbelArgmaxBf16", "GumbelArgmaxTempStep", "GumbelArgmaxTempBf16"]'),
        (f"{reg}/value_inputs.py",
         "reference norms;\nthe quarantine's `LayerNormAten_v2` stays in `quarantine.ln.layer_norm_aten`, out of the registry's load)",
         "reference norms, and\n`layer_norm`'s `LayerNormAten_v2`)"),
        ("verity_vllm/program/frontend/rules/vocab.py", "(quarantine lane cov-ln, `engine/profiles/gen_ln_patterns.py`)",
         "(`registry/layer_norm.py`, `engine/profiles/gen_ln_patterns.py`)"),
        ("verity_vllm/engine/profiles/gen_ln_patterns.py",
         "from verity_vllm.program.registry.quarantine.ln import gelu_erf, layer_norm_aten\n",
         "from verity_vllm.program.registry import act, layer_norm\n"),
        ("verity_vllm/engine/profiles/gen_ln_patterns.py", "layer_norm_aten.VEC", "layer_norm.VEC"),
        ("verity_vllm/engine/profiles/gen_ln_patterns.py", "bind(layer_norm_aten.LayerNormAten,", "bind(layer_norm.LayerNormAten,"),
        ("verity_vllm/engine/profiles/gen_ln_patterns.py", "bind(gelu_erf.GeluErf,", "bind(act.GeluErf,"),
        ("tests/engine/test_gen_ln.py",
         "from verity_vllm.program.registry.quarantine.ln import ALL_DEFINITIONS, ALL_PRIMITIVES, gelu_erf_bf16, gelu_erf_table, "
         "layer_norm_aten\n",
         "from verity_vllm.program.registry import act, gelu_erf_table, layer_norm\n"),
        ("tests/engine/test_gen_ln.py", "g = gelu_erf_bf16.gelu_erf_bf16_bits", "g = act.gelu_erf_bf16_bits"),
        ("tests/dead_code_keep.json",
         '"verity_vllm.program.registry.quarantine.ln.layer_norm_aten_difftest": "properties-admission adapter of LayerNormAten_v1, '
         'loaded by --adapter name"',
         '"verity_vllm.program.registry.quarantine": "the admission home of the quarantine lanes core binds, empty while none is '
         'pending; quarantine_lint\'s ROOT, protected.py, verify_lane.sh and P8\'s HOMES name it"'),
        ("tests/engine/test_gen_ln.py",
         "def test_lane_registry_lists_the_three_definitions():\n"
         '    assert [d.name for d in ALL_DEFINITIONS] == ["LayerNormAten", "GeluErf"] and [p.name for p in ALL_PRIMITIVES] == '
         '["GeluErfBf16"]\n'
         '    assert layer_norm_aten.VEC == 4 and layer_norm_aten.NTHREADS == 128 and layer_norm_aten.MEAN_FMA_ORDER == "A"\n',
         "def test_layer_norm_and_gelu_erf_are_registry_definitions():\n"
         "    from verity.ml.circuits.defs import REGISTRY\n"
         "    fns = (layer_norm.LayerNormAten, layer_norm.LayerNormAtenV2, act.GeluErf, act.GeluErfBf16)\n"
         '    assert [f.id for f in fns] == ["LayerNormAten_v1", "LayerNormAten_v2", "GeluErf_v1", "GeluErfBf16_v1"]\n'
         "    assert all(REGISTRY[f.id] is f for f in fns)\n"
         '    assert layer_norm.VEC == 4 and layer_norm.NTHREADS == 128 and layer_norm.MEAN_FMA_ORDER == "A"\n'),
        ("tests/engine/test_gen_ln.py", "bind(layer_norm_aten.LayerNormAten,", "bind(layer_norm.LayerNormAten,"),
        ("tests/engine/test_gen_ov_sampling.py", "# quarantine Definition `GumbelArgmaxBf16_v1`", "# registry Definition `GumbelArgmaxBf16_v1`"),
        # the promoted norm and GELU table join the dense family's Definition modules (that test's boundary)
        ("tests/program/test_dense_family_boundary.py", '          "boolean_rope", "gemm_bias")\n',
         '          "boolean_rope", "gemm_bias", "layer_norm", "gelu_erf_table")\n'),
        ("tests/program/test_layer_norm_aten.py", "(`quarantine/ln/layer_norm_aten_difftest.py`, on the GPU)",
         "(`registry/layer_norm_difftest.py`, on the GPU)"),
        ("tests/program/test_registered_value_chains.py", "the reference and quarantine norms' epsilon", "the reference and aten norms' epsilon"),
        ("tests/program/test_sampling_operands.py",
         "RTX-4090-validated static-seed composites of `registry/quarantine/ov_sampling/gumbel.py`, the kernel's",
         "RTX-4090-validated static-seed composites (`GumbelArgmaxBf16_v1`, `GumbelArgmaxTempBf16_v1`), the kernel's"),
        ("tests/program/test_sampling_operands.py",
         '    """The quarantine module re-exports this module\'s helpers (one transcription, one set of evidence)."""\n'
         "    from verity_vllm.program.registry.quarantine.ov_sampling import gumbel as Q\n"
         "    assert Q.noise_lane is S.noise_lane and Q.stream_key is S.stream_key and Q.nv_logf is S.nv_logf\n"
         "    assert Q.GumbelNoiseLane is S.GumbelNoiseLane and Q.GumbelStreamKey is S.GumbelStreamKey\n",
         '    """The libdevice logs and the Philox stream key at their fixed points."""\n'),
        ("README.md",
         "model-name tables (`registry/quarantine/` is the by-name residue and only shrinks).  `registry/fa3_check_inf.py` is FA3's",
         f"model-name tables (`registry/quarantine/` is the by-name residue and only shrinks; {residue}).\n"
         "`registry/fa3_check_inf.py` is FA3's"),
        (TARGETS, "    from verity_vllm.program.registry import serve3, spec\n",
         "    from verity_vllm.program.registry import act, serve3, spec\n"),
        (TARGETS, '        ("GumbelTopPTokenSelectSharedGreedy_v1", lambda: bind(sampling.GumbelTopPTokenSelectSharedGreedy, V=16)),\n',
         '        ("GumbelTopPTokenSelectSharedGreedy_v1", lambda: bind(sampling.GumbelTopPTokenSelectSharedGreedy, V=16)),\n'
         "        # the static-seed Gumbel argmax (T = 0.7f) and the erf GELU; LayerNormAten is the b1-eager-v3 vocabulary's\n"
         '        ("GumbelArgmaxBf16_v1", lambda: bind(sampling.GumbelArgmaxBf16, V=16, SEED=12345, POS=3)),\n'
         '        ("GumbelArgmaxTempBf16_v1", lambda: bind(sampling.GumbelArgmaxTempBf16, V=16, SEED=12345, POS=3, T=0x3F333333)),\n'
         '        ("GeluErf_v1", lambda: bind(act.GeluErf, N=8)),\n'),
    ))
    # while the MOVE lanes stay, core's quarantine package is not empty, and the promoted adapter's entry just drops
    return edits if dest else tuple(e for e in edits if not e[0].endswith("/tests/dead_code_keep.json"))


def _promoted_modules() -> dict[str, str]:
    """{quarantine module: the registry module that now holds its Definitions}."""
    return {f"{QUARANTINE_MODULE}.{p.src.removesuffix('.py').replace('/', '.')}": f"{REGISTRY_MODULE}.{p.dst.removesuffix('.py')}"
            for p in PROMOTE}


def _promote_rewrite(text: str) -> str:
    """``text`` importing the registry modules that hold what it imported from a promoted lane's modules; refused for an import of a
    promoted lane's package or of a name of it that is not a promoted module."""
    lanes = "|".join(map(re.escape, PROMOTED_LANES))
    qm = re.escape(QUARANTINE_MODULE)
    if m := re.search(rf"from {qm} import [^\n]*\b({lanes})\b", text):
        raise Refused(f"`{m.group(0)}` imports a promoted lane's package: import the registry module that holds what it uses")
    modules = _promoted_modules()

    def from_lane(m: re.Match) -> str:
        lane, names = m.group(1), m.group(2)
        body, hash_, comment = names.partition("#")
        if "(" in body or body.rstrip().endswith("\\"):
            raise Refused(f"`{m.group(0)}`: a multi-line import of a promoted lane; rewrite it by hand")
        out = []
        for item in body.split(","):
            name, _, alias = (s.strip() for s in item.strip().partition(" as "))
            src = f"{QUARANTINE_MODULE}.{lane}.{name}"
            if src not in modules:
                raise Refused(f"`{m.group(0).strip()}`: {name} is not a promoted module of lane {lane}; rewrite it by hand")
            leaf = modules[src].rsplit(".", 1)[1]
            out.append(leaf if (alias or name) == leaf else f"{leaf} as {alias or name}")
        return f"from {REGISTRY_MODULE} import {', '.join(out)}" + (f"  {hash_}{comment}" if hash_ else "")

    text = re.sub(rf"from {qm}\.({lanes}) import ([^\n]*)", from_lane, text)
    for old in sorted(modules, key=len, reverse=True):
        text = re.sub(re.escape(old) + r"(?!\w)", modules[old], text)
    return text


def _declared_ids(text: str) -> set[str]:
    out = set()
    for node in ast.parse(text).body:
        calls = [d for d in getattr(node, "decorator_list", []) if isinstance(d, ast.Call)]
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(node.value, ast.Call):
            calls.append(node.value)
        for c in calls:
            f = c.func.attr if isinstance(c.func, ast.Attribute) else getattr(c.func, "id", None)
            if (f in ("composite", "primitive") and len(c.args) >= 2 and isinstance(c.args[0], ast.Constant)
                    and isinstance(c.args[1], ast.Constant)):
                out.add(f"{c.args[0].value}_v{c.args[1].value}")
    return out


def _top_names(tree: ast.Module) -> set[str]:
    """The names a module binds at its top level."""
    out: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            out.update((a.asname or a.name).split(".")[0] for a in node.names)
        else:
            out.update(_binds(node))
    return out


def _binds(node: ast.stmt) -> tuple[str, ...]:
    return _bound(node) if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign)) else ()


def _merge(src_text: str, dst_text: str, p: Promotion) -> str:
    """``dst_text`` with ``src_text``'s statements after its imports (each with the comment lines just above it), but for its
    docstring, `SEMANTIC_ID` and `__all__`, inserted before `__all__` (or at the end), and ``p.imports`` after its last import;
    refused when a statement rebinds a name of ``dst`` or uses an imported name ``dst`` does not bind."""
    dst = ast.parse(dst_text)
    for alias in p.unqualify:
        names = sorted(set(re.findall(rf"(?<![\w.]){re.escape(alias)}\.(\w+)", src_text)))
        unbound = [n for n in names if n not in _top_names(dst)]
        if unbound:
            raise Refused(f"{p.src} -> {p.dst}: {p.dst} does not bind {', '.join(unbound)}, which {p.src} reads as {alias}.<name>")
        src_text = re.sub(rf"(?<![\w.]){re.escape(alias)}\.(\w+)", r"\1", src_text)
    src = ast.parse(src_text)
    lines = src_text.splitlines(keepends=True)
    kept, prev_end = [], 0
    for i, node in enumerate(src.body):
        start = min([node.lineno, *(d.lineno for d in getattr(node, "decorator_list", []))])
        while start - 1 > prev_end and lines[start - 2].lstrip().startswith("#"):
            start -= 1
        prev_end = node.end_lineno
        docstring = i == 0 and isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
        if docstring or isinstance(node, (ast.Import, ast.ImportFrom)) or {"SEMANTIC_ID", "__all__"} & set(_binds(node)):
            continue
        kept.append((node, "".join(lines[start - 1:node.end_lineno])))
    have = _top_names(dst) | _top_names(ast.parse("\n".join(p.imports)))
    clash = sorted({n for node, _ in kept for n in _binds(node)} & _top_names(dst))
    if clash:
        raise Refused(f"{p.src} -> {p.dst}: {', '.join(clash)} already bound in {p.dst}")
    imported = _top_names(ast.Module(body=[n for n in src.body if isinstance(n, (ast.Import, ast.ImportFrom))], type_ignores=[]))
    used = {n.id for node, _ in kept for n in ast.walk(node) if isinstance(n, ast.Name)}
    missing = sorted((imported & used) - have)
    if missing:
        raise Refused(f"{p.src} -> {p.dst}: {p.dst} does not bind {', '.join(missing)}; add it to the promotion's imports")
    block = "\n\n".join(t.rstrip("\n") + "\n" for _, t in kept)
    out = dst_text
    new_imports = [i for i in p.imports if i not in out]
    if new_imports:
        last = max(n.end_lineno for n in dst.body if isinstance(n, (ast.Import, ast.ImportFrom)))
        dl = out.splitlines(keepends=True)
        out = "".join(dl[:last] + [i + "\n" for i in new_imports] + dl[last:])
    m = re.search(r"^__all__ = ", out, re.M)
    if m:
        out = out[:m.start()] + block + "\n\n" + out[m.start():]
    else:
        out = out.rstrip("\n") + "\n\n\n" + block
    ast.parse(out)
    return out


@dataclass
class PromotionPlan:
    moves: list[tuple[str, str]]          # git mv src dst
    writes: dict[str, str]                # final text of each file written (a moved file under its new path)
    removes: list[str]                    # git rm
    lanes: list[str]                      # lane directories to remove once empty


def _replace_once(text: str, old: str, new: str, where: str) -> str:
    n = text.count(old)
    if n != 1:
        raise Refused(f"{where}: expected one {old[:80]!r}, found {n}")
    return text.replace(old, new)


def _residue(rel: str, text: str) -> list[str]:
    lanes = "|".join(map(re.escape, PROMOTED_LANES))
    pat = re.compile(rf"quarantine[./]({lanes})\b|{re.escape(QUARANTINE_MODULE)} import [^\n]*\b({lanes})\b")
    return [f"{rel}:{n}: {line.strip()[:120]}" for n, line in enumerate(text.splitlines(), 1) if pat.search(line)]


def plan_promotions(root: Path, dest: str | None) -> PromotionPlan:
    """Everything `promote` writes, computed without writing, and refused when a promoted module declares an id outside
    `PROMOTED_IDS`, when a lane would keep a module that is not promoted, or when any file would still name a promoted lane."""
    plan = PromotionPlan([], {}, [], [])
    pending = [p for p in PROMOTE if (root / QUARANTINE / p.src).is_file()]
    if not pending:
        return plan
    if len(pending) != len(PROMOTE):
        raise Refused(f"partly promoted: {', '.join(p.src for p in PROMOTE if p not in pending)} gone, the rest not; finish by hand")
    srcs = {f"{QUARANTINE}/{p.src}" for p in PROMOTE}
    declared = set().union(*(_declared_ids((root / s).read_text(encoding="utf-8")) for s in srcs))
    if declared != set(PROMOTED_IDS):
        raise Refused(f"the promoted modules declare {sorted(declared)}, not PROMOTED_IDS {sorted(PROMOTED_IDS)}")
    for lane in PROMOTED_LANES:
        left = sorted(p.relative_to(root).as_posix() for p in (root / QUARANTINE / lane).rglob("*") if p.is_file()
                      and "__pycache__" not in p.parts and p.relative_to(root).as_posix() not in srcs and p.name not in LANE_FILES)
        if left:
            raise Refused(f"lane {lane} keeps {', '.join(left)}, which is not promoted")
        plan.removes += [f"{QUARANTINE}/{lane}/{f}" for f in LANE_FILES if (root / QUARANTINE / lane / f).is_file()]
        plan.lanes.append(f"{QUARANTINE}/{lane}")
    texts: dict[str, str] = {}

    def read(rel: str) -> str:
        return texts[rel] if rel in texts else (root / rel).read_text(encoding="utf-8")

    rekey: dict[str, str] = {}
    for p in PROMOTE:
        src, dst = f"{QUARANTINE}/{p.src}", f"{REGISTRY_DIR}/{p.dst}"
        text = (root / src).read_text(encoding="utf-8")
        if dst not in texts and not (root / dst).exists():
            for old, new in p.edits:
                text = _replace_once(text, old, new, src)
            plan.moves.append((src, dst))
            texts[dst] = text
            rekey[src.removeprefix(VLLM + "/")] = dst.removeprefix(VLLM + "/")
        else:
            if p.edits:
                raise Refused(f"{p.src}: edits apply to a module that moves, and {p.dst} exists")
            texts[dst] = _merge(text, read(dst), p)
            plan.removes.append(src)
    for rel, old, new in promote_edits(dest):
        texts[rel] = _replace_once(read(rel), old, new, rel)
    gone = {s for s, _ in plan.moves} | set(plan.removes)
    residue = []
    for rel in sorted((set(tracked(root)) - gone) | set(texts)):
        if rel.startswith(UNTOUCHED):
            continue
        try:
            text = read(rel)
        except UnicodeDecodeError:
            continue
        if rel in KEYED_LISTS:
            new = _drop_lines(_rekey(text, rekey), _keyed_by_promoted)
            json.loads(new)
        else:
            new = _promote_rewrite(text)
            if rel.endswith(".py"):
                ast.parse(new, filename=rel)
        if new != text or rel in texts:
            texts[rel] = new
        residue += _residue(rel, new)
    if residue:
        raise Refused("would still name a promoted lane:\n  " + "\n  ".join(residue))
    if (root / P08).is_file():
        texts[P08] = _add_entries(read(P08), P08_ADDED, P08)
    plan.writes = texts
    return plan


def _add_entries(text: str, add, rel: str) -> str:
    """An allowlist's text (one entry per line under `entries`) with the entries of ``add`` it lacks, each before the first entry
    that sorts after it: in its sorted place in a sorted list, and with the other entries' order kept in one that is not (a list
    is read as a multiset, and the dense and prims splits re-key entries in place)."""
    data = json.loads(text)
    lines = text.splitlines(keepends=True)
    at = [i for i, line in enumerate(lines) if line.lstrip().startswith('{"file"')]
    key = lambda e: (e["file"], e["kind"], e["symbol"], e["detail"])  # noqa: E731
    if (not at or at != list(range(at[0], at[-1] + 1)) or [lines[i].strip().rstrip(",") for i in at] != [json.dumps(e) for e in data["entries"]]):
        raise Refused(f"{rel}: not one entry per line; add {[key(e) for e in add]} by hand")
    entries = list(data["entries"])
    for e in sorted(add, key=key):
        if e not in entries:
            entries.insert(next((i for i, x in enumerate(entries) if key(x) > key(e)), len(entries)), e)
    indent = lines[at[0]][:len(lines[at[0]]) - len(lines[at[0]].lstrip())]
    body = [f"{indent}{json.dumps(e)}{',' if j < len(entries) - 1 else ''}\n" for j, e in enumerate(entries)]
    out = "".join(lines[:at[0]] + body + lines[at[-1] + 1:])
    json.loads(out)
    return out


def _rekey(text: str, rekey: dict[str, str]) -> str:
    """A path-keyed list's entries for a moved module, keyed by its new path."""
    for old, new in rekey.items():
        text = text.replace(f'"{old}"', f'"{new}"')
    return text


def _keyed_by_promoted(line: str) -> bool:
    """An entry keyed by a promoted lane's file or module that did not follow a move: a merged module's, a lane file's, or a module
    key (every registry module is imported by `catalog.load`, so none is dead code)."""
    paths = tuple(f'"{QUARANTINE.removeprefix(VLLM + "/")}/{m}/' for m in PROMOTED_LANES)
    modules = tuple(f'"{QUARANTINE_MODULE}.{m}' for m in PROMOTED_LANES)
    return any(re.search(re.escape(x) + r'[^"]*"', line) for x in paths + modules)


def promote(root: Path, dest: str | None = DEST) -> list[str]:
    plan = plan_promotions(root, dest)
    changed = []
    for src, dst in plan.moves:
        _move_path(root, src, dst)
        changed.append(f"{src} -> {dst}")
    for rel, text in plan.writes.items():
        p = root / rel
        if not p.is_file() or p.read_text(encoding="utf-8") != text:
            p.write_text(text, encoding="utf-8")
            if rel not in {d for _, d in plan.moves}:
                changed.append(rel)
    if plan.removes:
        _git(root, "rm", "-q", *plan.removes)
        changed += [f"{rel} (removed)" for rel in plan.removes]
    for lane in plan.lanes:
        d = root / lane
        for cache in sorted(d.rglob("__pycache__"), reverse=True):
            for f in cache.iterdir():
                f.unlink()
            cache.rmdir()
        if d.is_dir():
            d.rmdir()
    return changed


# --- moving the quarantine ---------------------------------------------------------------------------------------------

def _move_path(root: Path, src: str, dst: str) -> None:
    (root / dst).parent.mkdir(parents=True, exist_ok=True)
    _git(root, "mv", src, dst)


def _rewrite(text: str) -> str:
    text = re.sub(rf"from {re.escape(QUARANTINE_MODULE)} import ([^\n]*)", _from_lanes, text)
    return re.sub(rf"\b{re.escape(QUARANTINE_MODULE)}\.({'|'.join(MOVE)})\b", rf"{PACKAGE}.\1", text)


def _from_lanes(m: re.Match) -> str:
    names = m.group(1)
    imported = {n.strip().split(" as ")[0].strip("() ") for n in names.split("#")[0].split(",")} - {""}
    if not imported & set(MOVE):
        return m.group(0)
    if not imported <= set(MOVE):
        raise Refused(f"`{m.group(0)}` imports moved and other lanes (or a multi-line list): split it by hand")
    return f"from {PACKAGE} import {names}"


def _keyed_by_moved(line: str) -> bool:
    paths = tuple(f'"{QUARANTINE.removeprefix(VLLM + "/")}/{m}/' for m in MOVE)
    modules = tuple(f'"{QUARANTINE_MODULE}.{m}' for m in MOVE)
    return any(re.search(re.escape(x) + r'[^"]*"', line) for x in paths + modules)


def _add_to_array(text: str, key: str, value: str, section: str | None = None) -> str:
    """``text`` with ``"value"`` added to the TOML array ``key`` (in ``section``, else at top level), when absent."""
    start = 0
    if section is not None:
        m = re.search(rf"^\[{re.escape(section)}\]\s*$", text, re.M)
        if not m:
            raise Refused(f"no [{section}] table")
        start = m.end()
    m = re.compile(rf"^{re.escape(key)} = \[", re.M).search(text, start)
    if not m:
        raise Refused(f"no array {key!r}")
    end = text.index("]", m.end())
    body = text[m.end():end]
    if f'"{value}"' in body:
        return text
    if "\n" in body:                                   # a multi-line array: first, so the existing lines keep their layout
        return text[:m.end()] + f'"{value}", ' + text[m.end():]
    return text[:end] + (", " if body.strip() else "") + f'"{value}"' + text[end:]


def check_dest(root: Path, dest: str) -> str:
    """``dest`` as a repository-relative directory, refused when it is not one or when `PACKAGE` already lives elsewhere (a
    relocation is a move of that directory, not a second copy)."""
    p = PurePosixPath(dest)
    if p.is_absolute() or not p.parts or ".." in p.parts or p.as_posix() != dest.rstrip("/"):
        raise Refused(f"--dest {dest!r}: give a normalized directory relative to the repository root")
    dest = p.as_posix()
    if dest == VLLM or dest.startswith((f"{VLLM}/", "packages/")):
        raise Refused(f"--dest {dest}: the experimental package lives outside core's distributions")
    elsewhere = sorted({rel.removesuffix(f"/{PACKAGE}/__init__.py") for rel in tracked(root)
                        if rel.endswith(f"/{PACKAGE}/__init__.py")} - {dest})
    if elsewhere:
        raise Refused(f"{PACKAGE} already lives in {', '.join(elsewhere)}, not {dest}: move that directory instead")
    return dest


def check_move(root: Path, dest: str) -> tuple[str, list[str]]:
    """(the checked ``dest``, the lanes still to move), after every refusal the move can raise before it writes."""
    dest = check_dest(root, dest)
    pending = [m for m in MOVE if (root / QUARANTINE / m).is_dir()]
    if pending:
        importers = library_importers(root, tuple(f"{QUARANTINE_MODULE}.{m}" for m in MOVE))
        importers += [f"{rel}: from {QUARANTINE_MODULE} import ..." for rel in _from_quarantine_importers(root)]
        if importers:
            raise Refused("core library code imports a lane that would move:\n  " + "\n  ".join(importers))
    for _rel, text in _scoped(root, dest):
        _rewrite(text)
    _member_texts(root, dest)
    return dest, pending


def move_quarantine(root: Path, dest: str = DEST) -> list[str]:
    changed: list[str] = []
    dest, pending = check_move(root, dest)
    dst_pkg = f"{dest}/{PACKAGE}"
    for lane in pending:
        if (root / dst_pkg / lane).exists():
            raise Refused(f"{lane}: both {QUARANTINE}/{lane} and {dst_pkg}/{lane} exist")
        for cache in (root / QUARANTINE / lane).rglob("__pycache__"):
            for f in cache.iterdir():
                f.unlink()
            cache.rmdir()
        _move_path(root, f"{QUARANTINE}/{lane}", f"{dst_pkg}/{lane}")
        changed.append(f"{QUARANTINE}/{lane} -> {dst_pkg}/{lane}")
    for rel, text in ((f"{dst_pkg}/__init__.py", PACKAGE_DOC), (f"{QUARANTINE}/__init__.py", core_doc(dest)),
                      (f"{dest}/pyproject.toml", PYPROJECT)):
        p = root / rel
        if not p.is_file() or p.read_text(encoding="utf-8") != text:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8")
            changed.append(rel)
    changed += _register_member(root, dest)
    changed += _rewrite_tree(root, dest)
    importers = library_importers(root, (PACKAGE,))
    if importers:
        raise Refused("core library code imports the experimental package:\n  " + "\n  ".join(importers))
    return changed


def _from_quarantine_importers(root: Path) -> list[str]:
    lanes = set(MOVE)
    out = []
    for p in sorted((root / VLLM / "verity_vllm").rglob("*.py")):
        rel = p.relative_to(root).as_posix()
        if rel.startswith(QUARANTINE + "/") or "__pycache__" in rel:
            continue
        for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module == QUARANTINE_MODULE and {a.name for a in node.names} & lanes:
                out.append(rel)
    return out


def _member_texts(root: Path, dest: str) -> dict[str, str]:
    """{pyproject: its text with the experimental member registered}: the workspace's members, the root's dependencies and
    sources, and the vLLM suite's inputs (its tests import the moved lanes)."""
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    new = _add_to_array(text, "members", dest, "tool.uv.workspace")
    new = _add_to_array(new, "dependencies", DISTRIBUTION, "project")
    if not re.search(rf"^{re.escape(DISTRIBUTION)} = ", new, re.M):
        lines = new.splitlines(keepends=True)
        try:
            at = lines.index("[tool.uv.sources]\n") + 1
        except ValueError:
            raise Refused("no [tool.uv.sources] table in the root pyproject.toml") from None
        end = next((i for i in range(at, len(lines)) if lines[i].startswith("[")), len(lines))
        last = [i for i in range(at, end) if lines[i].rstrip().endswith("{ workspace = true }")]
        lines.insert(last[-1] + 1 if last else at, f"{DISTRIBUTION} = {{ workspace = true }}\n")
        new = "".join(lines)
    out = {"pyproject.toml": new}
    text = (root / VLLM / "pyproject.toml").read_text(encoding="utf-8")
    new = _add_to_array(text, "inputs", dest, "tool.verity.tests")
    note = f"# Test-only: tests that check a quarantine lane's re-exports, adapters or collective import `{PACKAGE}` ({dest}).\n"
    if new != text and note not in new:
        m = re.search(r"^\[tool\.verity\.tests\]\s*\n", new, re.M)
        new = new[:m.end()] + note + new[m.end():]
    out[f"{VLLM}/pyproject.toml"] = new
    for t in out.values():
        tomllib.loads(t)
    return out


def _register_member(root: Path, dest: str) -> list[str]:
    changed = []
    for rel, new in _member_texts(root, dest).items():
        p = root / rel
        if p.read_text(encoding="utf-8") != new:
            p.write_text(new, encoding="utf-8")
            changed.append(rel)
    return changed


def _scoped(root: Path, dest: str):
    """(path, text) of each tracked text file the module rewrite applies to: the vLLM integration and the experimental package."""
    for rel in tracked(root):
        if not rel.startswith((f"{VLLM}/", f"{dest}/")) or rel.startswith(UNTOUCHED) or rel in KEYED_LISTS:
            continue
        try:
            yield rel, (root / rel).read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue


def _rewrite_tree(root: Path, dest: str) -> list[str]:
    changed = []
    for rel in KEYED_LISTS:
        p = root / rel
        if not p.is_file():
            continue
        text = p.read_text(encoding="utf-8")
        new = _drop_lines(text, _keyed_by_moved)
        if new != text:
            json.loads(new)
            p.write_text(new, encoding="utf-8")
            changed.append(rel)
    for rel, old, new_line in redirects(dest):
        p = root / rel
        if p.is_file() and old in (text := p.read_text(encoding="utf-8")):
            p.write_text(text.replace(old, new_line), encoding="utf-8")
            changed.append(rel)
    for rel, text in list(_scoped(root, dest)):
        new = _rewrite(text)
        if new != text:
            (root / rel).write_text(new, encoding="utf-8")
            changed.append(rel)
    return changed


# --- snapshot and verify -----------------------------------------------------------------------------------------------

def _digest(fn) -> str:
    from verity.ml.circuits.codec import canonical_json, definition_entry
    from verity.ml.circuits.defs import FunctionDefinition
    if not isinstance(fn, FunctionDefinition):
        return f"not a definition: {type(fn).__name__}"
    try:
        return hashlib.sha512(canonical_json(definition_entry(fn))).hexdigest()
    except Exception as e:  # an unbound generic Definition has no self-contained entry; its id is what is compared
        return f"unencodable: {type(e).__name__}"


def _import_tree(name: str) -> None:
    pkg = importlib.import_module(name)
    for m in pkgutil.walk_packages(pkg.__path__, f"{name}."):
        importlib.import_module(m.name)


def snapshot() -> dict:
    from circuit_check import targets as T
    from verity.ml.circuits.defs import REGISTRY
    suite = T.suite()
    _import_tree(QUARANTINE_MODULE)
    if importlib.util.find_spec(PACKAGE) is not None:
        _import_tree(PACKAGE)
    return {"suite": {i: _digest(fn) for i, fn in suite.items()},
            "registry": {i: _digest(d) for i, d in sorted(REGISTRY.defs.items())},
            "root_errors": sorted(map(list, T.ROOT_ERRORS)),
            "promoted_closure": sorted(T.reachable([fn for i, fn in suite.items() if family(i) in PROMOTED_IDS]))}


def verify(before: dict, after: dict, deleted: list[str]) -> list[str]:
    """What is wrong with ``after`` as the result of deleting ``deleted`` from ``before``: [] when nothing is.  An id ``after`` has and
    ``before`` lacks is new only outside what the promoted Definitions' suite targets reach (``after["promoted_closure"]``)."""
    problems = []
    promoted = set(after.get("promoted_closure", ()))
    for part in ("suite", "registry"):
        b, a = before[part], after[part]
        gone = {i for i in b if family(i) in deleted}
        lost = sorted(set(b) - set(a) - gone)
        kept_deleted = sorted(gone & set(a))
        added = sorted(set(a) - set(b) - promoted)
        changed = sorted(i for i in set(a) & set(b) if a[i] != b[i])
        problems += [f"{part}: lost {i}" for i in lost] + [f"{part}: deleted but present {i}" for i in kept_deleted]
        problems += [f"{part}: new {i}" for i in added] + [f"{part}: digest changed {i}: {b[i]} -> {a[i]}" for i in changed]
    if before.get("root_errors") != after.get("root_errors"):
        problems.append(f"root errors changed: {before.get('root_errors')} -> {after.get('root_errors')}")
    return problems


def commands() -> list[str]:
    out = []
    for vid, r in DELETE.items():
        out.append(f"research notes claim circuits/{r.slug} --by circuits   # if circuits/{r.slug} is not yet an approach entry")
        out.append(f"research notes approach circuits/{r.slug} superseded --superseded-by circuits/{r.superseded_by} --by circuits "
                   f"--reason {json.dumps(r.reason)} --cite {r.cite}   # {vid}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("apply", help="delete the DELETE versions, promote the PROMOTE modules and move the MOVE lanes")
    a.add_argument("--only", nargs="+", metavar="ID", help="delete only these versions (each must be in DELETE)")
    a.add_argument("--dest", default=DEST, metavar="DIR",
                   help=f"the directory, relative to the repository root, that gets the experimental package {PACKAGE} (default {DEST})")
    a.add_argument("--promote-only", action="store_true",
                   help="delete and promote, but move no lane: the MOVE lanes stay in core's quarantine")
    a.add_argument("--root", type=Path, default=REPO, help=argparse.SUPPRESS)
    sub.add_parser("commands", help="print the approach entries each deletion needs recorded first")
    s = sub.add_parser("snapshot", help="record circuit-check's task list and the registry, with digests")
    s.add_argument("out", type=Path)
    v = sub.add_parser("verify", help="compare two snapshots against DELETE")
    v.add_argument("before", type=Path)
    v.add_argument("after", type=Path)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "apply":
            ids = list(DELETE) if args.only is None else args.only
            changed = delete_versions(args.root, ids)
            if args.promote_only:
                changed += promote(args.root, None)
            else:
                dest, _ = check_move(args.root, args.dest)
                changed += promote(args.root, dest) + move_quarantine(args.root, dest)
            print("\n".join(changed) if changed else "nothing to do")
        elif args.cmd == "commands":
            print("\n".join(commands()) if DELETE else "no deletions: no approach entries needed")
        elif args.cmd == "snapshot":
            args.out.write_text(json.dumps(snapshot(), indent=1, sort_keys=True) + "\n", encoding="utf-8")
        else:
            before, after = json.loads(args.before.read_text()), json.loads(args.after.read_text())
            problems = verify(before, after, list(DELETE))
            gained = {part: sorted(set(after[part]) - set(before[part])) for part in ("suite", "registry")}
            print("\n".join(problems) if problems else
                  "identical but for the deleted versions and what the promoted Definitions' targets reach:\n"
                  + "\n".join(f"  {part} gains {i}" for part, ids in gained.items() for i in ids))
            return 1 if problems else 0
    except Refused as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
