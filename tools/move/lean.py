#!/usr/bin/env python3
"""The Lean layout move: split Lean into code and `verity/Security/` (Definitions, Specs, Proofs) as `lean_map.toml` says.

    python3 tools/move/lean.py              move, rename and regenerate; prints what it changed
    python3 tools/move/lean.py --dry-run    print the module renames and the file moves, and change nothing
    python3 tools/move/lean.py --resolve M  the name Lean module M has after the move

What it does, on the tree as it is (so it reruns on a later main, and a branch is restacked with it):
1. names every module of the packages `[packages]` lists from its file, and gives it its new name ([modules], longest
   prefix; "" drops a package umbrella, whose importers import what it imported);
2. moves the files with `layout.py`'s machinery (`layout.Run` on a map it generates: one row per file, one per module
   directory, the old package directories as tokens), which also rewrites every path that names them in the tree;
3. rewrites every `import` of every `.lean` file, and the module names after `lake build`, `lake env lean` and `import` in
   scripts and docs, and has each `Proofs.<Area>` import what its dropped package root imported and nothing else does;
4. writes the umbrellas (`Definitions`, `Specs`, `Proofs.<Area>`), the lakefiles, manifests and toolchains of `security` and
   `security_proofs`, and the code packages' lakefiles;
5. merges the old locks into `verity/Security/lean-audit.json` (the guarantees and their `reads`, under the modules' new
   names; `proved_in` the proofs package) and `verity/Security/Proofs/lean-audit.json` (roots, exempt, runs, compile_time,
   upstream, the layers of proof modules), and regenerates the code packages' locks, which keep the guarantees their code
   files declare. Declarations keep their names, so no guarantee changes:
   `audit.py --update --moved` on the result prints each as the same statement.
Old locks and package files are removed once merged. A step that finds nothing left to do does nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "verity" / "lean"))
import layout as L  # noqa: E402

POLICY = "lean-audit.json"
PKG_FILES = ("lakefile.toml", "lake-manifest.json", "lean-toolchain", POLICY, ".gitignore")
#: the PoUS grader was exempt from its package's audit; as a code package of its own it is audited, and its one escape is this
GRADER_ESCAPE = ("@[extern] opaque: Lean's resident-memory limit (`lean -M`, the runtime's lean_internal_set_max_memory), "
                 "which the grader sets from POUS_GRADE_MEMORY_MB before elaborating a submission; no proof is about it")
AREAS = ("Core", "Nci", "Pous", "Pouw", "Warden", "Flock")
IMPORT = re.compile(r"^(\s*(?:public\s+|private\s+|meta\s+)*import\s+)(.*)$")


def git(root: Path, *args: str) -> str:
    return L.git(root, *args)


class Map:
    def __init__(self, data: dict, root: Path):
        self.data, self.root = data, root
        self.sec = data["security"]["dir"]
        self.proofs = data["security"]["proofs"]
        self.pkgs = data["packages"]
        self.rules = data["modules"]

    def rename(self, module: str, pkg: str) -> str | None:
        """The new name of `module` of package `pkg`: None when it stays, "" when it is dropped."""
        parts = module.split(".")
        for i in range(len(parts), 0, -1):
            head = ".".join(parts[:i])
            for key in (f"{pkg}:{head}", head):
                if key in self.rules:
                    new = self.rules[key]
                    if new == "":
                        return "" if i == len(parts) else None
                    return ".".join([new, *parts[i:]])
        return None

    def new_path(self, new: str, pkg: str) -> str:
        if new.split(".")[0] in ("Definitions", "Specs", "Proofs"):
            return f"{self.sec}/{new.replace('.', '/')}.lean"
        return f"{self.pkgs[pkg]['dir']}/{new.replace('.', '/')}.lean"


def tracked(root: Path) -> list[str]:
    return [p for p in git(root, "ls-files", "-z").split("\0") if p]


def owner(m: Map, path: str) -> str | None:
    """The package `path` belongs to (the deepest package directory holding it)."""
    best = None
    for k, v in m.pkgs.items():
        if path.startswith(v["dir"] + "/") and (best is None or len(v["dir"]) > len(m.pkgs[best]["dir"])):
            best = k
    return best


def modules(m: Map, files: list[str]) -> dict[str, tuple[str, str]]:
    """{path: (package, old module)} for every tracked `.lean` file of the old packages."""
    out = {}
    for p in files:
        if not p.endswith(".lean") or (k := owner(m, p)) is None:
            continue
        rel = p[len(m.pkgs[k]["dir"]) + 1:]
        if rel == "lakefile.lean":
            continue
        out[p] = (k, rel[:-5].replace("/", "."))
    return out


def header_imports(text: str) -> tuple[list[tuple[int, str, list[str]]], int]:
    """The import lines of a Lean file's header: (line index, prefix, modules), and the index of the first other line."""
    lines, out, depth = text.split("\n"), [], 0
    for i, line in enumerate(lines):
        s = line.strip()
        if depth:
            depth += s.count("/-") - s.count("-/")
            continue
        if not s or s.startswith("--"):
            continue
        if s.startswith("/-"):
            depth = s.count("/-") - s.count("-/")
            continue
        if (mo := IMPORT.match(line)):
            out.append((i, mo.group(1), mo.group(2).split("--")[0].split()))
            continue
        return out, i
    return out, len(lines)


def rename_all(m: Map, mods: dict) -> tuple[dict[str, str], dict[str, list[str]], dict[str, str]]:
    """({old: new} for every renamed module whose old name one package holds, {dropped umbrella: what it imported},
    {path: new module} for every renamed file)."""
    ren, dropped, by_path, names = {}, {}, {}, {}
    for p, (k, old) in mods.items():
        names.setdefault(old, []).append(p)
        new = m.rename(old, k)
        if new == "":
            text = (m.root / p).read_text()
            dropped[old] = [x for _, _, ms in header_imports(text)[0] for x in ms]
        elif new is not None and new != old:
            by_path[p] = new
            ren[old] = new
    for old, ps in names.items():
        if len(ps) > 1:
            ren.pop(old, None)
    clash = {}
    for p, n in by_path.items():
        clash.setdefault(n, []).append(p)
    if bad := {n: ps for n, ps in clash.items() if len(ps) > 1}:
        raise SystemExit(f"two modules would get one name: {bad}")
    return ren, dropped, by_path


DECL = re.compile(r"^\s*(?:@\[[^\]]*\]\s*)?(?:(?:private|protected|noncomputable)\s+)*(?:theorem|lemma)\s+([^\s:({\[]+)")
NAMESPACE = re.compile(r"^\s*namespace\s+(\S+)")
SECTION = re.compile(r"^\s*(?:noncomputable\s+)?section(?:\s+\S+)?\s*$")
END = re.compile(r"^\s*end(?:\s+\S+)?\s*$")


def theorems(text: str) -> set[str]:
    """The full names of the theorems a Lean file declares, from its `namespace`/`section`/`end` lines."""
    stack: list[list[str]] = []
    out = set()
    for line in text.split("\n"):
        if (mo := NAMESPACE.match(line)):
            stack.append(mo.group(1).split("."))
        elif SECTION.match(line):
            stack.append([])
        elif END.match(line):
            if stack:
                stack.pop()
        elif (mo := DECL.match(line)):
            n = mo.group(1)
            out.add(n[len("_root_."):] if n.startswith("_root_.") else ".".join([p for ps in stack for p in ps] + [n]))
    return out


def code_theorems(m: Map, mods: dict) -> dict[str, set[str]]:
    """{code package: the theorems its code files declare}: a guarantee proved beside the code stays in its lock."""
    out: dict[str, set[str]] = {}
    for p, (k, old) in mods.items():
        if old.split(".")[0] in m.pkgs[k].get("code", []):
            out.setdefault(k, set()).update(theorems((m.root / p).read_text()))
    return out


def expand(name: str, ren: dict, dropped: dict, seen=()) -> list[str]:
    if name in dropped:
        return [y for x in dropped[name] if x not in seen for y in expand(x, ren, dropped, (*seen, name))]
    return [ren.get(name, name)]


# ------------------------------------------------------------------------------------------------------------- step 2

def layout_map(m: Map, files: list[str], mods: dict, by_path: dict) -> dict:
    base = tomllib.loads((HERE / "layout_map.toml").read_text())
    paths: dict[str, str] = {}
    moved_dirs: dict[str, str] = {}
    for p, (k, old) in mods.items():
        if p in by_path:
            paths[p] = m.new_path(by_path[p], k)
            d_old = p[:-5]
            if any(f.startswith(d_old + "/") for f in files):
                moved_dirs[d_old] = paths[p][:-5]
    for o, n in m.data.get("files", {}).items():
        paths[o] = n
    for d, n in moved_dirs.items():
        paths[d + "/"] = n + "/"
    rw = dict(base["rewrite"])
    rw["exclude"] = [*rw["exclude"], f"**/{POLICY}", "**/lakefile.toml", "**/lake-manifest.json"]
    rw["include"] = [x for x in rw["include"] if x != "**/lake-manifest.json"]
    data = {"roots": {"old": base["roots"]["new"], "new": base["roots"]["new"]}, "paths": paths,
            "tokens": {k + "/": v + "/" for k, v in m.data.get("dirs", {}).items()}, "rewrite": rw}
    data["modules"] = L.derived_modules(L.Layout(data))
    return data


def run_layout(root: Path, data: dict, dry: bool) -> dict:
    layout = L.Layout(data)
    run = L.Run(root, layout, dry)
    moves = run.plan()
    run.index(moves)
    run.move(moves)
    run.rewrite_all()
    if moves:
        run.record()
    run.write()
    if not dry:
        git(root, "add", "-A")
        if moves:
            subprocess.run(["git", "--literal-pathspecs", "add", "-f", "--pathspec-from-file=-", "--pathspec-file-nul"],
                           cwd=root, input="\0".join(n for _, n in moves).encode(), check=True, capture_output=True)
    if report := os.environ.get("LEAN_MOVE_REPORT"):
        Path(report).write_text(json.dumps({"notes": run.notes, "files": {p: dict(c) for p, c in run.files_changed.items()}},
                                           indent=1, sort_keys=True, default=str))
    return {"files moved": len(moves), "files rewritten": len(run.files_changed), "notes": {k: len(v) for k, v in run.notes.items()}}


# ------------------------------------------------------------------------------------------------------------- step 3

def rewrite_imports(root: Path, ren: dict, dropped: dict, dry: bool) -> int:
    n = 0
    for p in tracked(root):
        if not p.endswith(".lean") or not (root / p).exists():
            continue
        text = (root / p).read_text()
        imps, _ = header_imports(text)
        if not imps:
            continue
        lines = text.split("\n")
        changed = False
        seen: set[str] = set()
        for i, pre, ms in reversed(imps):
            new = [y for x in ms for y in expand(x, ren, dropped)]
            if new != ms:
                changed = True
                lines[i] = "\n".join(f"{pre}{x}" for x in new)
        if changed:
            out, emitted = [], set()
            for line in "\n".join(lines).split("\n"):
                mo = IMPORT.match(line)
                if mo and len(mo.group(2).split()) == 1 and line in emitted:
                    continue
                if mo:
                    emitted.add(line)
                out.append(line)
            n += 1
            if not dry:
                (root / p).write_text("\n".join(out))
    return n


def keep_reachable(m: Map, root: Path, ren: dict, dropped: dict, dry: bool) -> list[str]:
    """What a dropped package root imported, and nothing under `Proofs.<Area>` imports any more, is imported by that root:
    a module only the old root imported (`Nci.SecurityProofs.Throttling`) would otherwise be built and audited by nobody."""
    sec = root / m.sec

    def path(mod: str) -> Path:
        return sec / (mod.replace(".", "/") + ".lean")

    def reach(start: str) -> set[str]:
        seen, todo = set(), [start]
        while todo:
            x = todo.pop()
            if x in seen or not path(x).exists():
                continue
            seen.add(x)
            todo += [y for _, _, ms in header_imports(path(x).read_text())[0] for y in ms]
        return seen

    added = []
    for old in sorted(dropped):
        if "." in old:
            continue
        for x in expand(old, ren, dropped):
            parts = x.split(".")
            if parts[0] != "Proofs" or len(parts) < 3 or not path(top := ".".join(parts[:2])).exists():
                continue
            if x in reach(top) or not path(x).exists():
                continue
            text = path(top).read_text()
            imps, _ = header_imports(text)
            lines = text.split("\n")
            lines.insert(imps[-1][0] + 1 if imps else 0, f"import {x}")
            if not dry:
                path(top).write_text("\n".join(lines))
            added.append(f"{top} imports {x}")
    return added


CMD = re.compile(r"(lake\s+(?:build|env\s+lean|exe)\s+|import\s+)([A-Za-z_][\w.' ]*)")


def rewrite_commands(root: Path, ren: dict, dry: bool, skip=("tools/move/",)) -> int:
    """Module names after `lake build`, `lake exe`, `import` in scripts and docs (never a declaration: only whole tokens
    that are old module names)."""
    n = 0
    names = sorted(ren, key=len, reverse=True)
    tok = re.compile(r"(?<![\w.])(" + "|".join(re.escape(x) for x in names) + r")(?![\w.'])") if names else None
    for p in tracked(root):
        if p.endswith((".lean", ".json", ".lock")) or p.startswith(skip) or not (root / p).exists():
            continue
        try:
            text = (root / p).read_text()
        except (UnicodeDecodeError, OSError):
            continue
        if "lake" not in text and "import" not in text:
            continue
        new = []
        for line in text.split("\n"):
            if tok and re.search(r"lake\s+(build|env|exe)|^\s*import\s|\bimport [A-Z]", line):
                line = tok.sub(lambda mo: ren[mo.group(1)], line)
            new.append(line)
        out = "\n".join(new)
        if out != text:
            n += 1
            if not dry:
                (root / p).write_text(out)
    return n


# ------------------------------------------------------------------------------------------------------------- step 4

def write(root: Path, rel: str, text: str, dry: bool):
    if dry:
        return
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    (root / rel).write_text(text)


def umbrellas(m: Map, root: Path, dry: bool) -> list[str]:
    """`Definitions.lean`, `Specs.lean` and `Proofs/Flock.lean` over what is there; the area umbrellas move as modules."""
    sec = root / m.sec
    out = []

    def have(mod):
        return (sec / (mod.replace(".", "/") + ".lean")).exists()
    defs = [f"Definitions.{a}" for a in AREAS if have(f"Definitions.{a}")]
    specs = [f"Specs.{a}.{s}" for a in AREAS for s in ("Assumptions", "Guarantees") if have(f"Specs.{a}.{s}")]
    flock = [f"Proofs.Flock.{x}" for x in ("Verifier", "Level3", "Soundness") if have(f"Proofs.Flock.{x}")]
    for rel, doc, ims in ((f"{m.sec}/Definitions.lean", "The trusted vocabulary: each component's definitions.", defs),
                          (f"{m.sec}/Specs.lean", "Each component's named assumptions and guarantee statements.", specs),
                          (f"{m.proofs}/Flock.lean", "C-Flock's proofs: the verifier's, level3's and soundness's.", flock)):
        if ims:
            write(root, rel, "".join(f"import {x}\n" for x in ims) + f"\n/-! {doc} -/\n", dry)
            out.append(rel)
    return out


def manifest(old: dict, name: str, path_pkgs: list[tuple[str, str]], drop=()) -> dict:
    """A manifest named `name`: `old`'s git packages, and `path_pkgs` ((name, dir relative to the new package))."""
    pk = [p for p in old["packages"] if p.get("type") == "git" and p["name"] not in drop]
    paths = [{"type": "path", "scope": "", "name": n, "manifestFile": "lake-manifest.json", "inherited": i > 0,
              "dir": d, "configFile": "lakefile.toml"} for i, (n, d) in enumerate(path_pkgs)]
    return {"version": old["version"], "packagesDir": old["packagesDir"], "packages": [*paths, *pk], "name": name,
            "lakeDir": old["lakeDir"], "fixedToolchain": old.get("fixedToolchain", False)}


def lake_files(m: Map, root: Path, old_manifests: dict, toolchain: str, dry: bool) -> list[str]:
    sec, prf = m.sec, m.proofs
    mathlib_rev = "5ed2965256430c3649e86755f9576b54eca72435"
    arklib_rev = "b2e456fc48db5bff1b69c7469fde73f9b83e94c2"
    verifier = os.path.relpath(m.pkgs["verifier"]["dir"], prf)
    write(root, f"{sec}/lakefile.toml", f'''# The security analysis's trusted text: Definitions (each component's vocabulary) and Specs (its named assumptions
# and guarantee statements), on Mathlib. The proofs are the nested package `Proofs/` (security_proofs).
name = "security"
defaultTargets = ["Definitions", "Specs"]

[leanOptions]
autoImplicit = false
relaxedAutoImplicit = false

[[require]]
name = "mathlib"
git = "https://github.com/leanprover-community/mathlib4"
rev = "{mathlib_rev}"

[[lean_lib]]
name = "Definitions"

[[lean_lib]]
name = "Specs"
''', dry)
    roots = [f"Proofs.{a}" for a in AREAS if (root / prf / f"{a}.lean").exists() or dry]
    write(root, f"{prf}/lakefile.toml", f'''# The security analysis's proofs (untrusted): every proof of a guarantee `security` states, C-Flock's level3 and
# soundness among them. Its modules sit beside the trusted text's (`srcDir = ".."`) under a root of their own, `Proofs`.
name = "security_proofs"
srcDir = ".."
defaultTargets = ["Proofs"]

[leanOptions]
autoImplicit = false
relaxedAutoImplicit = false

[[require]]
name = "Arklib"
git = "https://github.com/Verified-zkEVM/ArkLib"
rev = "{arklib_rev}"

[[require]]
name = "security"
path = ".."

[[require]]
name = "flock_verifier"
path = "{verifier}"

[[lean_lib]]
name = "Proofs"
roots = {json.dumps(roots)}
''', dry)
    vdir = m.pkgs["verifier"]["dir"]
    write(root, f"{vdir}/lakefile.toml", '''name = "flock_verifier"
version = "0.1.0"
defaultTargets = ["flock-verify", "flock-rows"]

[leanOptions]
autoImplicit = false

[[lean_lib]]
name = "Flock"

[[lean_exe]]
name = "flock-verify"
root = "FlockVerify"

[[lean_exe]]
name = "flock-rows"
root = "FlockRows"
''', dry)
    pdir = m.pkgs["pous"]["dir"]
    write(root, f"{pdir}/lakefile.toml", '''# The PoUS grader: a program that imports only Lean. What it grades against is `verity/Security`'s trusted text.
name = "pous_grader"
defaultTargets = ["Grader", "pous-grader"]

[leanOptions]
autoImplicit = false
relaxedAutoImplicit = false

[[lean_lib]]
name = "Grader"
roots = ["Grader.Registry"]

[[lean_exe]]
name = "pous-grader"
root = "Grader.Main"
''', dry)
    sm = manifest(old_manifests["pouw"], "security", [])
    pm = manifest(old_manifests["soundness"], "security_proofs", [("security", ".."), ("flock_verifier", verifier)])
    vm = old_manifests["verifier"]
    gm = dict(manifest(old_manifests["verifier"], "pous_grader", []), packages=[])
    for rel, man in ((f"{sec}/lake-manifest.json", sm), (f"{prf}/lake-manifest.json", pm),
                     (f"{vdir}/lake-manifest.json", vm), (f"{pdir}/lake-manifest.json", gm)):
        write(root, rel, json.dumps(man, indent=1) + "\n", dry)
    for d in (sec, prf):
        write(root, f"{d}/lean-toolchain", toolchain, dry)
    write(root, f"{sec}/.gitignore", "/.lake\n/Proofs/.lake\n", dry)
    return [sec, prf, vdir, pdir]


def rekey_bundles(m: Map, root: Path, old_keys: dict, LA) -> dict:
    """`tools/lean/lean-deps.json` under the new manifests' keys: `security` fetches what PoUW's package fetched (Mathlib),
    `security_proofs` what soundness's did (ArkLib and Mathlib), byte for byte, so their pinned bundles serve them."""
    path = root / LA.PIN
    pin = json.loads(path.read_text())
    deps = pin["deps"]
    out = {}
    for new_dir, old in ((m.sec, "pouw"), (m.proofs, "soundness")):
        key = LA.deps_key(root / new_dir)
        if (ok := old_keys.get(old)) in deps and key:
            deps[key] = dict(deps[ok], package=new_dir)
            out[new_dir] = key[:12]
    for k, ok in old_keys.items():
        if ok and ok in deps and ok not in {LA.deps_key(root / d) for d in (m.sec, m.proofs)}:
            del deps[ok]
    path.write_text(json.dumps(pin, indent=1, sort_keys=True) + "\n")
    return out


# ------------------------------------------------------------------------------------------------------------- step 5

def merge_locks(m: Map, root: Path, locks: dict, code_thms: dict, dry: bool):
    import check as A
    sec = {"roots": ["Definitions", "Specs"], "proved_in": ["Proofs"], "assumptions": [], "meaning": [],
           "layers": {}, "reads_exempt": {}, "guarantees": {}, "reads": {}, "dependencies": {}}
    prf = {"roots": [f"Proofs.{a}" for a in AREAS], "exempt": {}, "runs": {}, "compile_time": {},
           "upstream": {"imports": [], "watch": {}}, "dependencies": {}}
    code = {"verifier": {"roots": ["Flock", "FlockVerify"], "exempt": {}, "meaning": ["Flock"], "escapes": {},
                         "dependencies": {}},
            "pous": {"roots": ["Grader.Registry", "Grader.Main"], "exempt": {}, "escapes": {"setMaxMemory": GRADER_ESCAPE},
                     "dependencies": {}}}
    conflicts: list[str] = []
    gen = {"pouw": "verity/protocols/accounting/work/pouw/lean", "warden": "verity/protocols/accounting/communication/warden/lean"}
    for k, pol in locks.items():
        pol = dict(pol)

        def mp(name: str, ren=None, k=k) -> str:
            return m.rename(name, k) or name
        if "pins" in pol:
            pol["guarantees"] = pol.pop("pins")
            for g in pol.get("reads", {}).values():
                if "pins" in g:
                    g["guarantees"] = g.pop("pins")
        for x in pol.get("assumptions", []):
            if mp(x) not in sec["assumptions"]:
                sec["assumptions"].append(mp(x))
        for x in pol.get("meaning", []):
            if mp(x) not in sec["meaning"]:
                sec["meaning"].append(mp(x))
        for lay, rule in pol.get("layers", {}).items():
            (prf if mp(lay).startswith("Proofs.") else sec).setdefault("layers", {})[mp(lay)] = {"imports": [mp(x) for x in rule["imports"]],
                                           "except": [mp(x) for x in rule.get("except", [])], "why": rule["why"]}
        here = code_thms.get(k, set()) if k in code else set()
        for x, why in pol.get("reads_exempt", {}).items():
            sec["reads_exempt"].setdefault(mp(x), why)
            if here and x.split(".")[0] in m.pkgs[k]["code"]:
                code[k].setdefault("reads_exempt", {})[x] = why
        for g, rec in pol.get("guarantees", {}).items():
            (code[k] if g in here else sec).setdefault("guarantees", {})[g] = rec
        for mod, g in pol.get("reads", {}).items():
            defs = g.get("definitions", {})
            if isinstance(defs, list):
                raise SystemExit(f"{k}: `reads` {mod} is a record written before SHA-256 hashes; `audit.py --update` it first")
            mine = [x for x in g.get("guarantees", []) if x in here]
            for pol_to, gs in ((code.get(k), mine), (sec, [x for x in g.get("guarantees", []) if x not in here])):
                if pol_to is None or not gs:
                    continue
                cur = pol_to.setdefault("reads", {}).setdefault(mp(mod), {"digest": "", "definitions": {}, "guarantees": []})
                for d, h in defs.items():
                    if cur["definitions"].setdefault(d, h) != h:
                        conflicts.append(f"{d} ({k})")
                cur["guarantees"] = sorted({*cur["guarantees"], *gs})
                cur["digest"] = A.module_digest(cur["definitions"])
        for x, why in pol.get("exempt", {}).items():
            n = mp(x)
            if k == "verifier" and x == "FlockRows":
                code["verifier"]["exempt"][x] = why
            elif k == "pous" and x in ("Grader", "submissions"):
                if x == "submissions":
                    code["pous"]["exempt"][x] = why
            else:
                prf["exempt"][n] = why
        for x, run in pol.get("runs", {}).items():
            run = json.loads(json.dumps(run))
            if run.get("generate"):
                rel = os.path.relpath(gen[k], m.proofs)
                run["generate"] = [a if not a.startswith("scripts/") else f"{rel}/{a}" for a in run["generate"]]
            prf["runs"][mp(x)] = run
        for x, ct in pol.get("compile_time", {}).items():
            prf["compile_time"][mp(x)] = ct
        if up := pol.get("upstream"):
            prf["upstream"]["imports"] = sorted({*prf["upstream"]["imports"], *up.get("imports", [])})
            for w, e in up.get("watch", {}).items():
                if w in prf["upstream"]["watch"]:
                    raise SystemExit(f"upstream watch {w} in two locks")
                prf["upstream"]["watch"][w] = e
            for s, e in up.get("scan", {}).items():
                if s in prf["upstream"].setdefault("scan", {}):
                    raise SystemExit(f"upstream scan {s} in two locks")
                prf["upstream"]["scan"][s] = e
        if k == "verifier":
            code["verifier"]["escapes"] = pol.get("escapes", {})
        for d, h in pol.get("dependencies", {}).items():
            if k in code:
                if d == "toolchain":
                    code[k]["dependencies"][d] = h
            else:
                (prf if k == "soundness" else sec)["dependencies"].setdefault(d, h)
    sec["reads_exempt"] = {k: v for k, v in sec["reads_exempt"].items() if k not in ("Grader",)}
    sec["reads"] = dict(sorted(sec["reads"].items()))
    sec["guarantees"] = dict(sorted(sec["guarantees"].items()))
    for pol in code.values():
        for key in ("guarantees", "reads"):
            if key in pol:
                pol[key] = dict(sorted(pol[key].items()))
    prf["dependencies"] = {**sec["dependencies"], **prf["dependencies"]}
    for pol in (sec, prf, *code.values()):
        for key in [k for k, v in pol.items() if v in ({}, [], {"imports": [], "watch": {}})]:
            if key not in ("dependencies", "guarantees", "reads"):
                del pol[key]
    out = {"definitions two locks hashed differently (a definition's hash covers what its readers read; `--update` "
           "rewrites them)": conflicts}
    for rel, pol in ((f"{m.sec}/{POLICY}", sec), (f"{m.proofs}/{POLICY}", prf),
                     (f"{m.pkgs['verifier']['dir']}/{POLICY}", code["verifier"]),
                     (f"{m.pkgs['pous']['dir']}/{POLICY}", code["pous"])):
        write(root, rel, A.text(pol), dry)
        out[rel] = {k: len(v) if hasattr(v, "__len__") else v for k, v in pol.items()}
    return out


# ---------------------------------------------------------------------------------------------------------------- main

def case_clashes(paths) -> list[list[str]]:
    """Paths (and their directories) that differ only in case: a checkout on a case-insensitive filesystem holds one."""
    seen: dict[str, set[str]] = {}
    for f in paths:
        parts = f.split("/")
        for i in range(1, len(parts) + 1):
            seen.setdefault("/".join(parts[:i]).lower(), set()).add("/".join(parts[:i]))
    return sorted(sorted(v) for v in seen.values() if len(v) > 1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--map", type=Path, default=HERE / "lean_map.toml")
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resolve", metavar="MODULE")
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args(argv)
    root = (a.root or Path(git(HERE, "rev-parse", "--show-toplevel").strip())).resolve()
    m = Map(tomllib.loads(a.map.read_text()), root)
    if a.resolve:
        for k in m.pkgs:
            if (n := m.rename(a.resolve, k)) is not None:
                print(n or "(dropped)")
                return 0
        print(a.resolve)
        return 0
    if not a.allow_dirty and git(root, "status", "--porcelain").strip():
        raise SystemExit(f"{root}: the worktree isn't clean")
    files = tracked(root)
    mods = modules(m, files)
    ren, dropped, by_path = rename_all(m, mods)
    print(f"lean move at {git(root, 'rev-parse', 'HEAD').strip()}{' (dry run)' if a.dry_run else ''}")
    print(f"  modules: {len(mods)}, renamed {len(by_path)}, umbrellas dropped {len(dropped)}")
    old_pkgs = {k: v["dir"] for k, v in m.pkgs.items() if (root / v["dir"] / POLICY).exists()}
    locks = {k: json.loads((root / d / POLICY).read_text()) for k, d in old_pkgs.items()}
    manifests = {k: json.loads((root / d / "lake-manifest.json").read_text()) for k, d in old_pkgs.items()
                 if (root / d / "lake-manifest.json").exists()}
    if a.dry_run:
        for p, n in sorted(by_path.items(), key=lambda x: x[1]):
            print(f"    {mods[p][1]} -> {n}")
        return 0
    if not by_path and not dropped:
        print("  nothing left to move")
        return 0
    toolchain = (root / m.pkgs["verifier"]["dir"] / "lean-toolchain").read_text()
    sys.path.insert(0, str(root / "tools" / "lean"))
    import lean_audit as LA
    old_keys = {k: LA.deps_key(root / d) for k, d in old_pkgs.items()}
    code_thms = code_theorems(m, mods)
    data = layout_map(m, files, mods, by_path)
    print(f"  layout: {run_layout(root, data, False)}")
    for p, (k, o) in mods.items():
        if o in dropped and (root / p).exists():
            git(root, "rm", "-q", p)
    print(f"  imports rewritten in {rewrite_imports(root, ren, dropped, False)} files")
    print(f"  kept reachable: {keep_reachable(m, root, ren, dropped, False)}")
    print(f"  commands rewritten in {rewrite_commands(root, {**ren, 'Main': 'FlockVerify'}, False)} files")
    print(f"  umbrellas: {umbrellas(m, root, False)}")
    keep = {m.pkgs["verifier"]["dir"], m.pkgs["pous"]["dir"]}
    for k, d in old_pkgs.items():
        for f in PKG_FILES:
            if (root / d / f).exists() and d not in keep:
                git(root, "rm", "-q", f"{d}/{f}")
    print(f"  lake: {lake_files(m, root, manifests, toolchain, False)}")
    print(f"  lean-deps.json: {rekey_bundles(m, root, old_keys, LA)}")
    print(f"  locks: {json.dumps(merge_locks(m, root, locks, code_thms, False))}")
    git(root, "add", "-A")
    if clash := case_clashes(tracked(root)):
        raise SystemExit(f"paths that differ only in case (a [files] entry in {a.map.name} must keep them apart): {clash}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
