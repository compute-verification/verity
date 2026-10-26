"""`map`: the repository's Lean, mapped from its sources, with no build and no Lake.

    python tools/verity/lean/map.py [--only abcde] [--by component|area] [--moves MAP.toml] [--top N] [--json FILE]
                                    [--root DIR]

(a) every Lake package: its requires (a path one as the package there, a git one at its revision), the roots its
    libraries list, its executables and its size in modules, lines and declarations, and the path requires none of its
    modules imports;
(b) the areas against the trees: each area of `Security/` (a directory under `Definitions/`, `Properties/` or `Proofs/`)
    and each code package's area, against Code, Definitions, Properties, Proofs and the properties' proofs, with the
    trees an area is missing from and its entry modules; then the same regrouped into the Glossary's components
    (`COMPONENT`);
(c) the import graph condensed to components (`--by area`: to areas): who imports whom, the cycles and the fewest
    thinnest edges that break them (each with an import that makes it), the same in the Definitions tree alone, code
    that imports `Security/`, and the modules most of the tree imports; an umbrella that only umbrellas import gathers
    modules for the build, and its imports are left out of the graph;
(d) module granularity: module sizes by tree, the largest modules, generated and umbrella modules, and the directories
    holding one module or fifty and more;
(e) code and `Security/`: which code packages each Security package requires, by which chain of requires, and how many
    modules of each tree import each one's modules.

`--moves MAP.toml` maps the tree as a move map would leave it: the dotted entries of its `[words]` (the
`tools/move/*_map.toml` form, `"Definitions.Accounting.Device" = "Definitions.Core.Device"`) rename modules where a
word starts at a part of the name, as `rename.py` does, with every import of them, before anything is counted; a module
whose new root is another package's
(`Proofs.` to `Definitions.`) moves into it. So a proposed move shows the cycles it leaves before anyone runs it.

A module belongs to the deepest package directory that holds its file, and takes its name from that package's `srcDir`.
A file under no package's `srcDir` (the auditor's own `.lean`, benchmarks' and tests' scratch programs) is loose. A
declaration is a line that starts, after attributes and modifiers, with a declaration keyword, outside comments: a count
for comparing modules, not Lean's. A module is generated when its first lines say so. A Security module's area is its
second component (`Definitions.Pouw.X` is `Pouw`'s, `Proofs.Properties.Nci.X` is `Nci`'s), a code module's is its
package's (`CODE_AREA`), and `COMPONENT` (longest module prefix first; one ending in `*` is a raw prefix) reads each into
one of the Glossary's components. Standard library only, like the auditor beside it.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tomllib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

#: the trees of `Security/`, longest prefix first; a module of a package outside `Security/` is code
TREES = [("Proofs.Properties", "Proofs/Properties"), ("Definitions", "Definitions"), ("Models", "Definitions"),
         ("Properties", "Properties"), ("Proofs", "Proofs")]
COLUMNS = ["Code", "Definitions", "Properties", "Proofs", "Proofs/Properties"]

#: a code package's area: the component its Lean implements
CODE_AREA = {
    "flock_verifier": "Flock", "core_vstar": "Flock", "core_sigma": "Flock", "core_firewall": "Service",
    "core_sampled_proofs": "SampledProofs", "pouw_window": "Pouw", "pouw_check": "Pouw", "pouw_commit": "Pouw",
    "pous_check": "Pous", "pous_grader": "Pous", "core_memory_challenger": "Pous", "remote_audit": "Bandwidth",
    "network_check": "NetworkCertifier", "certifier_program": "NetworkCertifier", "certifier_device": "NetworkCertifier",
    "nci_check": "Nci", "core_identity": "Hash", "lms": "Hash", "core_randomness": "Core", "verity_evaluate": "Core",
    "core_world": "Core", "core_protocol": "Core", "core_requirement": "Accounting", "core_composition": "Verity",
    "core_onsite": "Verity", "core_remote": "Verity", "verity_difftest": "Tooling", "control": "Tooling",
}

#: the Glossary's components (ARCHITECTURE.md), in a reading order: Core (with the device model it refines) and Hash,
#: what the audits share above Core (Device: schedules, the actual circuit, confidence sets and the joint run), the six
#: subprotocols, NCI (a registration check), Accounting's joint audit and physical layer, the proof system (Flock) and
#: the proof service's other parts, the one theorem (Verity), the claims' shared vocabulary, the tooling
COMPONENTS = ["Core", "Hash", "Device", "SampledProofs", "Pouw", "Pous", "Bandwidth", "NetworkCertifier", "Wipe",
              "Nci", "Accounting", "Flock", "Service", "Verity", "Claims", "Tooling", "(root)"]
OUTSIDE = {"Tooling", "(root)"}

#: the Accounting area's stems that belong to another component, under each of its trees and sub-areas (a key ending
#: in `*` is a raw prefix: `RateAudit*` is `RateAuditSketch` too)
ACCOUNTING_ROOTS = ["Definitions.Accounting", "Definitions.Accounting.Guarantees", "Definitions.Accounting.Assumptions",
                    "Proofs.Accounting", "Properties.Accounting", "Proofs.Properties.Accounting"]
ACCOUNTING = {
    "Device": "Device", "Schedule": "Device", "Actual": "Device", "Confidence": "Device", "Physical": "Device",
    "Localization": "Device", "Bridge": "Device",
    "Actual.Wipe*": "Wipe", "Wipe*": "Wipe",
    "Bandwidth*": "Bandwidth", "RateAudit*": "Bandwidth", "Remote*": "Bandwidth", "BoundReceipt*": "Bandwidth",
    "BoundSketch*": "Bandwidth", "Sketch": "Bandwidth", "Tag": "Bandwidth",
    "TS": "Pouw", "Headline*": "Pouw", "Responder": "Pous",
}

#: module prefix -> component, where a module's area is not its component; the longest prefix wins
COMPONENT = {
    **{f"{r}.{s}": c for r in ACCOUNTING_ROOTS for s, c in ACCOUNTING.items()},
    # the device model imports only Mathlib, and Core's accelerator model and blocks refine it (`layers`); the joint
    # run (`Accounting.Run`) is the vocabulary each audit's certificate is stated in; the actual circuit's guarantees
    # over NCI's reads, and their proofs, are the physical layer's
    "Definitions.Accounting.Device": "Core", "Definitions.Accounting.Joint": "Device",
    "Definitions.Accounting.Guarantees.Actual": "Accounting",
    "Proofs.Accounting.Actual": "Accounting",
    **{f"Proofs.Accounting.Actual.{s}": "Device" for s in ("Confidence", "Device", "Localization", "Physical", "Source",
                                                           "Split", "Ticks", "Transducer")},
    "Definitions.Population": "SampledProofs", "Proofs.Population": "SampledProofs",
    "Definitions.Claims.Sampling": "SampledProofs", "Proofs.Flock.Claims.Sampling": "SampledProofs",
    "Proofs.Flock.Sampled": "SampledProofs", "Proofs.Flock.SamplingScope": "SampledProofs",
    "Proofs.Flock.SamplingClaims": "SampledProofs", "Proofs.Flock.SamplingBoundary": "SampledProofs",
    "Definitions.Claims.Recursion": "Flock", "Definitions.Claims.RecursionLin": "Flock", "Definitions.Claims.Json": "Flock",
    "Definitions.Claims.Hm96": "Hash",
    "Definitions.Pous.Model.StreamedWipe": "Wipe",
    "Definitions.Pous.Guarantees.StreamedWipe": "Wipe", "Proofs.Pous.StreamedWipe": "Wipe",
    "MemoryChallenger.StreamedWipe": "Wipe", "MemoryChallenger.Wipe*": "Wipe",
    # C-Flock's verifier package also holds what every code package reads: bytes, canonical JSON, the program file,
    # draws, and the hashes and hiding leaf
    "Flock.Bytes": "Core", "Flock.Canon": "Core", "Flock.Program": "Core", "Flock.Sizes": "Core", "Flock.Steps": "Core",
    "Flock.Draw": "Core", "Flock.Hash": "Hash", "Flock.Keccak": "Hash", "Flock.Hm96": "Hash",
    "Properties.CertifierDevice": "NetworkCertifier", "Properties.CertifierNetwork": "NetworkCertifier",
    "Properties.NetworkRecords": "NetworkCertifier", "Properties.NciDesc": "Nci", "Properties.Identity": "Hash",
    "Properties.Evaluate": "Core", "Properties.RemoteAudit": "Bandwidth", "Properties.SampledProofs": "SampledProofs",
    "Proofs.Properties.CertifierDevice": "NetworkCertifier", "Proofs.Properties.CertifierNetwork": "NetworkCertifier",
    "Proofs.Properties.NetworkRecords": "NetworkCertifier", "Proofs.Properties.NciDesc": "Nci",
    "Proofs.Properties.Identity": "Hash", "Proofs.Properties.Evaluate": "Core",
    "Proofs.Properties.RemoteAudit": "Bandwidth", "Proofs.Properties.SampledProofs": "SampledProofs",
    "Properties.Composition": "Verity", "Proofs.Properties.Composition": "Verity",
}

#: a declaration's keyword, after attributes and modifiers, at the start of a line
DECL = re.compile(r"^\s*(?:@\[[^\]]*\]\s*)*(?:(?:private|protected|public|noncomputable|partial|unsafe|nonrec|scoped|"
                  r"local)\s+)*(theorem|lemma|def|abbrev|structure|inductive|class|instance|opaque|axiom|example)\b")
KIND = {"theorem": "thm", "lemma": "thm", "def": "def", "abbrev": "def", "opaque": "def", "axiom": "def",
        "structure": "type", "inductive": "type", "class": "type", "instance": "inst", "example": "ex"}
IMPORT = re.compile(r"^\s*(?:public\s+|meta\s+|private\s+)*import\s+(?:all\s+)?([A-Za-z_][\w.'«»]*)")
GENERATED = re.compile(r"generated by|do not edit|@generated", re.IGNORECASE)
NUMBER = re.compile(r"^[-\d,./%]+$")
DOTTED = re.compile(r"^[A-Z][\w']*(?:\.[\w']+)*$")
OUTER, INNER, STRING = re.compile(r'/-|--|"'), re.compile(r"/-|-/"), re.compile(r'"(?:[^"\\]|\\.)*"', re.DOTALL)


@dataclass
class Package:
    name: str
    dir: str                                        # repository path of its lakefile's directory
    src: str                                        # repository path of its srcDir
    requires: list[tuple[str, str]]                 # (name, repository path) or (name, "git:URL@REV")
    libs: dict[str, list[str]]
    exes: int
    modules: list[str] = field(default_factory=list)

    @property
    def security(self) -> bool:
        return self.dir == "Security" or self.dir.startswith("Security/")


@dataclass
class Module:
    name: str
    file: str
    package: str                                    # its package's directory
    lines: int
    decls: Counter
    imports: list[str]
    umbrella: bool                                  # imports and comments only
    generated: bool

    @property
    def ndecls(self) -> int:
        return sum(v for k, v in self.decls.items() if k != "ex")


def tracked(root: Path) -> list[str]:
    try:
        out = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], capture_output=True, check=True).stdout
        return sorted(f for f in out.decode().split("\0") if f and "/.lake/" not in f and not f.startswith(".lake/"))
    except (OSError, subprocess.CalledProcessError):
        return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and ".lake" not in p.parts
                      and ".git" not in p.parts)


def norm(base: str, rel: str) -> str:
    return os.path.normpath(os.path.join(base, rel)) if base else os.path.normpath(rel)


def read_packages(root: Path, files: list[str]) -> dict[str, Package]:
    pkgs = {}
    for f in files:
        if f.rsplit("/", 1)[-1] != "lakefile.toml":
            continue
        d = f.rsplit("/", 1)[0] if "/" in f else ""
        t = tomllib.loads((root / f).read_text())
        reqs = []
        for r in t.get("require", []):
            if "path" in r:
                reqs.append((r["name"], norm(d, r["path"])))
            else:
                reqs.append((r["name"], f"git:{r.get('git', r.get('scope', '?'))}@{str(r.get('rev', '?'))[:10]}"))
        libs = {lib["name"]: lib.get("roots", [lib["name"]]) for lib in t.get("lean_lib", [])}
        pkgs[d] = Package(t["name"], d, norm(d, t.get("srcDir", ".")), reqs, libs, len(t.get("lean_exe", [])))
    return pkgs


def strip_comments(text: str) -> str:
    """The text with `--` line comments and nested `/- -/` block comments (docstrings among them) blanked, lines kept."""
    out, i, depth, n = [], 0, 0, len(text)
    while i < n:
        m = (INNER if depth else OUTER).search(text, i)
        if depth:
            end = m.start() if m else n
            out.append("\n" * text.count("\n", i, end))
            if not m:
                break
            depth += 1 if m.group() == "/-" else -1
            i = m.end()
        elif not m:
            out.append(text[i:])
            break
        else:
            out.append(text[i:m.start()])
            if m.group() == "/-":
                depth, i = 1, m.end()
            elif m.group() == "--":
                j = text.find("\n", m.start())
                i = n if j < 0 else j
            else:
                s = STRING.match(text, m.start())
                i = s.end() if s else n
                out.append(text[m.start():i])
    return "".join(out)


def read_module(root: Path, f: str, pkg: Package) -> Module:
    text = (root / f).read_text(errors="replace")
    decls, imports, other = Counter(), [], False
    for line in strip_comments(text).splitlines():
        if m := IMPORT.match(line):
            imports.append(m.group(1))
        elif m := DECL.match(line):
            decls[KIND[m.group(1)]] += 1
            other = True
        elif line.strip():
            other = True
    name = os.path.relpath(root / f, root / pkg.src)[:-len(".lean")].replace(os.sep, ".")
    lines = text.count("\n") + (bool(text) and not text.endswith("\n"))
    head = "\n".join(text.splitlines()[:12])
    return Module(name, f, pkg.dir, lines, decls, imports, bool(imports) and not other, bool(GENERATED.search(head)))


def owner(f: str, dirs: list[str]) -> str | None:
    best = None
    for d in dirs:
        if (d == "" or f.startswith(d + "/")) and (best is None or len(d) > len(best)):
            best = d
    return best


def longest(name: str, prefixes) -> str | None:
    """The longest of ``prefixes`` that is ``name`` or a dotted prefix of it, or, for one ending in `*`, a raw prefix."""
    hit = lambda p: name.startswith(p[:-1]) if p.endswith("*") else name == p or name.startswith(p + ".")
    return max((p for p in prefixes if hit(p)), key=lambda p: len(p.rstrip("*")), default=None)


def read_moves(path: Path) -> dict[str, str]:
    """The dotted entries of a move map's `[words]`: old module prefix -> new."""
    words = tomllib.loads(path.read_text()).get("words", {})
    return {k: v for k, v in words.items() if DOTTED.match(k) and DOTTED.match(v)}


def apply_moves(pkgs: dict[str, Package], mods: dict[str, Module], moves: dict[str, str]) -> dict[str, Module]:
    """The modules renamed by ``moves``, their imports with them; a module whose new root is another package's moves into
    that package, and its file to its new name's path there. A word renames where it starts at a part of the name, as
    `rename.py` renames it in an import (`Properties.X` in `Proofs.Properties.X` too), the leftmost and then the longest."""
    def rename(n: str) -> str:
        parts = n.split(".")
        for i in range(len(parts)):
            if p := longest(".".join(parts[i:]), moves):
                return ".".join([*parts[:i], moves[p] + ".".join(parts[i:])[len(p):]])
        return n

    home = {}
    for m in mods.values():
        home.setdefault(m.name.split(".")[0], m.package)
    out = {}
    for m in mods.values():
        new = rename(m.name)
        if new == m.name:
            out[new] = Module(new, m.file, m.package, m.lines, m.decls, [rename(i) for i in m.imports], m.umbrella,
                              m.generated)
            continue
        pkg = home.get(new.split(".")[0], m.package)
        f = norm(pkgs[pkg].src, new.replace(".", "/") + ".lean")
        if new in out:
            print(f"--moves: {m.name} and another module both become {new}", file=sys.stderr)
        out[new] = Module(new, f, pkg, m.lines, m.decls, [rename(i) for i in m.imports], m.umbrella, m.generated)
    for p in pkgs.values():
        p.modules = []
    for m in out.values():
        pkgs[m.package].modules.append(m.name)
    return out


def tree_of(mod: str, pkg: Package) -> str:
    if not pkg.security:
        return "Code"
    for prefix, tree in TREES:
        if mod == prefix or mod.startswith(prefix + "."):
            return tree
    return "Code"


def area_of(mod: str, pkg: Package, tree: str) -> str:
    if tree == "Code":
        return CODE_AREA.get(pkg.name, pkg.name)
    parts = mod.split(".")
    k = 2 if tree == "Proofs/Properties" else 1
    return parts[k] if len(parts) > k else "(root)"


def component_of(mod: str, area: str) -> str:
    best = longest(mod, COMPONENT)
    if best:
        return COMPONENT[best]
    return {"Population": "SampledProofs", "RemoteAudit": "Bandwidth", "Identity": "Hash", "Evaluate": "Core"}.get(area, area)


def load(root: Path, moves: dict[str, str] | None = None):
    files = tracked(root)
    pkgs = read_packages(root, files)
    dirs = sorted(pkgs)
    mods, loose, dup = {}, [], defaultdict(list)
    for f in files:
        if not f.endswith(".lean"):
            continue
        d = owner(f, dirs)
        pkg = pkgs.get(d) if d is not None else None
        if pkg is None or not (pkg.src == "." or f.startswith(pkg.src + "/")):
            loose.append(f)
            continue
        m = read_module(root, f, pkg)
        dup[m.name].append(m.file)
        mods[m.name] = m
        pkg.modules.append(m.name)
    if moves:
        mods = apply_moves(pkgs, mods, moves)
    info = {}
    for m in mods.values():
        pkg = pkgs[m.package]
        t = tree_of(m.name, pkg)
        a = area_of(m.name, pkg, t)
        info[m.name] = (t, a, component_of(m.name, a))
    return pkgs, mods, info, loose, {k: v for k, v in dup.items() if len(v) > 1}


def size(ms) -> tuple[int, int, int]:
    ms = list(ms)
    return len(ms), sum(m.lines for m in ms), sum(m.ndecls for m in ms)


def fmt(n: int) -> str:
    return f"{n:,}"


def order(names) -> list[str]:
    return sorted(set(names), key=lambda x: (COMPONENTS.index(x) if x in COMPONENTS else 99, x))


def table(rows: list[list], head: list[str]) -> str:
    rows = [[str(c) for c in r] for r in rows]
    w = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(head)]
    num = [bool(rows) and all(NUMBER.match(r[i]) for r in rows) for i in range(len(head))]
    line = lambda r: "  ".join((c.rjust(w[i]) if num[i] else c.ljust(w[i])) for i, c in enumerate(r)).rstrip()
    return "\n".join([line(head), line(["-" * x for x in w]), *map(line, rows)])


def sccs(nodes: list[str], edges: dict[str, set[str]]) -> list[list[str]]:
    """The strongly connected components of more than one node (Tarjan's, iterative)."""
    index, low, on, stack, out, n = {}, {}, set(), [], [], 0
    for s in nodes:
        if s in index:
            continue
        work = [(s, iter(sorted(edges.get(s, ()))))]
        index[s] = low[s] = n
        n += 1
        stack.append(s)
        on.add(s)
        while work:
            v, it = work[-1]
            w = next(it, None)
            if w is not None:
                if w not in index:
                    index[w] = low[w] = n
                    n += 1
                    stack.append(w)
                    on.add(w)
                    work.append((w, iter(sorted(edges.get(w, ())))))
                elif w in on:
                    low[v] = min(low[v], index[w])
                continue
            work.pop()
            if work:
                low[work[-1][0]] = min(low[work[-1][0]], low[v])
            if low[v] == index[v]:
                c = []
                while True:
                    w = stack.pop()
                    on.discard(w)
                    c.append(w)
                    if w == v:
                        break
                if len(c) > 1:
                    out.append(order(c))
    return out


def breakers(nodes: list[str], weight: Counter) -> list[tuple[str, str]]:
    """A minimal set of edges whose removal leaves no cycle, the thinnest first: remove the lightest edge inside a cycle
    until none is left, then put back, heaviest first, each removed edge that closes no cycle."""
    live = {e for e in weight if weight[e] > 0}
    graph = lambda es: {a: {b for (x, b) in es if x == a} for a in nodes}
    removed = []
    while True:
        comps = sccs(nodes, graph(live))
        if not comps:
            break
        inside = [e for e in live if any(e[0] in c and e[1] in c for c in comps)]
        e = min(inside, key=lambda e: (weight[e], e))
        live.discard(e)
        removed.append(e)
    for e in sorted(removed, key=lambda e: -weight[e]):
        if not sccs(nodes, graph(live | {e})):
            live.add(e)
    return sorted((e for e in removed if e not in live), key=lambda e: (weight[e], e))


def section_a(pkgs, mods, loose, dup):
    print("(a) Lake packages\n")
    rows = []
    for d, p in sorted(pkgs.items(), key=lambda kv: -size(mods[n] for n in kv[1].modules)[1]):
        f, l, k = size(mods[n] for n in p.modules)
        path = [n for n, r in p.requires if not r.startswith("git:")]
        git = [f"{n}@{r.rsplit('@', 1)[1]}" for n, r in p.requires if r.startswith("git:")]
        roots = sum(len(v) for v in p.libs.values())
        rows.append([p.name, d, f, fmt(l), fmt(k), roots, p.exes, " ".join(git + path) or "-"])
    print(table(rows, ["package", "dir", "modules", "lines", "decls", "roots", "exes", "requires"]))
    f, l, k = size(mods.values())
    code = [m for m in mods.values() if not pkgs[m.package].security]
    print(f"\n{len(pkgs)} packages ({sum(p.security for p in pkgs.values())} under Security/), {fmt(f)} modules, "
          f"{fmt(l)} lines, {fmt(k)} declarations; code: {len(code)} modules, {fmt(sum(m.lines for m in code))} lines")
    print(f"{len(loose)} loose .lean files, under no package's srcDir: "
          + ", ".join(sorted({x.rsplit('/', 1)[0] for x in loose})))
    if dup:
        print(f"module names two files declare: {', '.join(sorted(dup))}")
    listed = [(p.name, sum(len(v) for v in p.libs.values())) for p in pkgs.values()]
    listed = [(n, k) for n, k in listed if k > 5]
    if listed:
        print("libraries listing roots one by one (each new root edits the lakefile): "
              + ", ".join(f"{n} {k}" for n, k in sorted(listed, key=lambda x: -x[1])))
    used = defaultdict(set)
    for m in mods.values():
        for i in m.imports:
            if i in mods and mods[i].package != m.package:
                used[m.package].add(mods[i].package)
    idle = [(p.name, [n for n, r in p.requires if not r.startswith("git:") and r not in used[d]])
            for d, p in sorted(pkgs.items())]
    idle = [(n, r) for n, r in idle if r]
    if idle:
        print("path requires none of the package's modules imports (needed only for a require behind it, or idle): "
              + "; ".join(f"{n}: {', '.join(r)}" for n, r in idle))


def entries(mods, area: str) -> str:
    """Which entry modules an area of `Security/` has: D(efinitions), A(ssumptions), G(uarantees), P(roperties), R (its
    proofs' root) and Q (its properties' proofs' root)."""
    have = [("D", f"Definitions.{area}"), ("A", f"Definitions.{area}.Assumptions"),
            ("G", f"Definitions.{area}.Guarantees"), ("P", f"Properties.{area}"), ("R", f"Proofs.{area}"),
            ("Q", f"Proofs.Properties.{area}")]
    return "".join(k if n in mods else "." for k, n in have)


def section_b(pkgs, mods, info):
    print("\n(b) areas x trees: modules/lines\n")
    for key, label in ((1, "area"), (2, "component")):
        cell = defaultdict(list)
        for n, (t, a, c) in info.items():
            cell[((t, a, c)[key], t)].append(mods[n])
        rows, holes, single = [], [], []
        for a in order(k[0] for k in cell):
            r, present = [a], []
            for t in COLUMNS:
                f, l, _ = size(cell.get((a, t), []))
                r.append(f"{f}/{fmt(l)}" if f else "-")
                if f:
                    present.append(t)
            r.append(fmt(sum(size(cell.get((a, t), []))[1] for t in COLUMNS)))
            if key == 1:
                r.append(entries(mods, a))
            rows.append(r)
            if a in OUTSIDE:
                continue
            sec = [t for t in COLUMNS[1:4] if t in present]
            if len(sec) == 1 and "Code" not in present:
                single.append(f"{a} ({sec[0]})")
            missing = [t for t in COLUMNS[1:4] if t not in present]
            if missing:
                holes.append(f"{a}: no {'/'.join(missing)}")
        print(table(rows, [label, *COLUMNS, "lines", *(["entries"] if key == 1 else [])]))
        if key == 1:
            print("entries: D Definitions.<A>, A .Assumptions, G .Guarantees, P Properties.<A>, R Proofs.<A>, "
                  "Q Proofs.Properties.<A>")
        print("missing trees: " + "; ".join(holes))
        if single:
            print("in one tree of Security/ only, with no code: " + "; ".join(single))
        print()
    moved = Counter((a, c) for n, (t, a, c) in info.items() if a != c and a != "(root)")
    print("areas read into another component (COMPONENT): "
          + "; ".join(f"{a} -> {c} {k}" for (a, c), k in sorted(moved.items(), key=lambda kv: -kv[1])))


def closure_counts(mods: dict[str, Module]) -> dict[str, int]:
    """For each module, how many modules import it, directly or not (bitsets over a topological order)."""
    names = sorted(mods)
    idx = {n: i for i, n in enumerate(names)}
    topo, seen = [], set()
    for s in names:
        if s in seen:
            continue
        seen.add(s)
        work = [(s, iter(mods[s].imports))]
        while work:
            v, it = work[-1]
            w = next(it, None)
            if w is None:
                work.pop()
                topo.append(v)
            elif w in mods and w not in seen:
                seen.add(w)
                work.append((w, iter(mods[w].imports)))
    below = {}
    for n in topo:
        b = 1 << idx[n]
        for i in mods[n].imports:
            if i in below:
                b |= below[i]
        below[n] = b
    count = [0] * len(names)
    for n, b in below.items():
        b &= ~(1 << idx[n])
        while b:
            low = b & -b
            count[low.bit_length() - 1] += 1
            b ^= low
    return {n: count[idx[n]] for n in names}


def aggregators(mods: dict[str, Module]) -> set[str]:
    """The umbrella modules that only umbrellas import, or none: they gather a tree for the build, and nothing reads
    through them."""
    read = {i for m in mods.values() if not m.umbrella for i in m.imports}
    return {n for n, m in mods.items() if m.umbrella and n not in read}


def graph(mods, info, k: int, keep=lambda n: True):
    """Edges between groups (`info[n][k]`) with their import lines and one import that makes each, less an aggregating
    umbrella's (`aggregators`)."""
    weight, example = Counter(), {}
    skip = aggregators(mods)
    for n, m in mods.items():
        if not keep(n) or n in skip:
            continue
        a = info[n][k]
        for i in m.imports:
            if i in mods and keep(i) and (b := info[i][k]) != a:
                weight[(a, b)] += 1
                example.setdefault((a, b), f"{n} -> {i}")
    return weight, example


def cycles(title: str, weight: Counter, example: dict):
    nodes = order({x for e in weight for x in e} - OUTSIDE)
    w = Counter({e: v for e, v in weight.items() if e[0] not in OUTSIDE and e[1] not in OUTSIDE})
    edges = defaultdict(set)
    for a, b in w:
        edges[a].add(b)
    comps = sccs(nodes, edges)
    print(f"\n{title}: " + ("; ".join(" ".join(c) for c in comps) if comps else "none"))
    if comps:
        cut = breakers(nodes, w)
        print(f"  the {len(cut)} edges whose removal leaves none, thinnest first (import lines, one import):")
        for a, b in cut:
            print(f"    {a} -> {b} ({w[(a, b)]}): {example[(a, b)]}")


def section_c(pkgs, mods, info, by: str, top: int):
    k = 2 if by == "component" else 1
    print(f"\n(c) the import graph, condensed to {by}s\n")
    weight, example = graph(mods, info, k)
    ext = defaultdict(Counter)
    for n, m in mods.items():
        for i in m.imports:
            if i not in mods:
                ext[info[n][k]][i.split(".")[0]] += 1
    groups = order(v[k] for v in info.values())
    rows = []
    for g in groups:
        out = sorted((b for a, b in weight if a == g), key=lambda b: -weight[(g, b)])
        fan_in = {a for a, b in weight if b == g and a not in OUTSIDE}
        lib = ", ".join(f"{r} {c}" for r, c in ext[g].most_common(3))
        rows.append([g, len(fan_in), len(out), " ".join(f"{b}({weight[(g, b)]})" for b in out) or "-", lib or "-"])
    print(table(rows, [by, "fan-in", "fan-out", "imports (import lines)", "outside the repository"]))
    cycles(f"cycles among {by}s (Tooling and the roots aside)", weight, example)
    dw, de = graph(mods, info, k, keep=lambda n: info[n][0] == "Definitions")
    cycles(f"cycles among {by}s in the Definitions tree alone", dw, de)
    bad = [(n, i) for n, m in mods.items() if not pkgs[m.package].security
           for i in m.imports if i in mods and pkgs[mods[i].package].security]
    print("\ncode importing Security/: " + (", ".join(f"{n} -> {i}" for n, i in bad) if bad else "none"))
    counts = closure_counts(mods)
    total = len(mods)
    print(f"\nmodules most of the tree imports, directly or not ({fmt(total)} modules):")
    rows = [[n, pkgs[mods[n].package].name, fmt(counts[n]), f"{100 * counts[n] / total:.0f}%", fmt(mods[n].lines)]
            for n in sorted(counts, key=lambda x: (-counts[x], x))[:top]]
    print(table(rows, ["module", "package", "importers", "share", "lines"]))
    sec = [n for n in mods if pkgs[mods[n].package].security]
    rows = [[n, info[n][0], fmt(counts[n]), fmt(mods[n].lines)]
            for n in sorted(sec, key=lambda x: (-counts[x], x))[:top]]
    print("\nthe same within Security/:")
    print(table(rows, ["module", "tree", "importers", "lines"]))
    direct = Counter(i for m in mods.values() for i in m.imports if i in mods)
    print("\nmost imported directly:")
    print(table([[n, c, info[n][k]] for n, c in direct.most_common(top)], ["module", "importers", by]))


def section_d(pkgs, mods, info, top: int):
    print("\n(d) module granularity\n")
    bounds = [50, 100, 200, 500, 1000, 2000, 5000]
    bucket = lambda n: next((i for i, b in enumerate(bounds) if n <= b), len(bounds))
    labels = [f"<={b}" for b in bounds] + [f">{bounds[-1]}"]
    by_tree = defaultdict(Counter)
    for n, m in mods.items():
        by_tree[info[n][0]][bucket(m.lines)] += 1
    rows = [[t, *[by_tree[t][i] for i in range(len(labels))], sum(by_tree[t].values())] for t in COLUMNS]
    print(table(rows, ["lines \\ tree", *labels, "modules"]))
    ls = sorted(m.lines for m in mods.values())
    q = lambda p: ls[min(len(ls) - 1, int(p * len(ls)))]
    print(f"\nmodule lines: median {q(.5)}, p90 {q(.9)}, p99 {q(.99)}, max {fmt(ls[-1])}")
    gen = [m for m in mods.values() if m.generated]
    print(f"generated modules: {len(gen)}, {fmt(sum(m.lines for m in gen))} lines")
    rows = [[m.name, pkgs[m.package].name, fmt(m.lines), m.ndecls, m.decls["thm"], m.decls["def"],
             "yes" if m.generated else ""] for m in sorted(mods.values(), key=lambda m: -m.lines)[:top]]
    print("\nlargest modules:")
    print(table(rows, ["module", "package", "lines", "decls", "thm", "def", "generated"]))
    umb = [m for m in mods.values() if m.umbrella]
    tiny = [m for m in mods.values() if not m.umbrella and m.ndecls <= 1]
    print(f"\numbrella modules (imports only): {len(umb)}; other modules of at most one declaration: {len(tiny)}")
    per_dir = defaultdict(list)
    for m in mods.values():
        per_dir[m.file.rsplit("/", 1)[0]].append(m.name)
    one = sorted(d for d, ms in per_dir.items() if len(ms) == 1)
    trees = Counter(info[per_dir[d][0]][0] for d in one)
    print(f"directories holding one module: {len(one)} of {len(per_dir)} ("
          + ", ".join(f"{t} {k}" for t, k in trees.most_common()) + ")")
    print("  e.g. " + ", ".join(d for d in one if d.startswith("Security"))[:600])
    big = sorted(((d, len(ms)) for d, ms in per_dir.items() if len(ms) >= 50), key=lambda x: -x[1])
    print(f"directories holding 50 modules or more: {len(big)}")
    for d, k in big:
        print(f"  {k:5}  {d}")
    deep = Counter(m.name.count(".") + 1 for m in mods.values())
    print("module name depth (components: modules): " + ", ".join(f"{d}: {k}" for d, k in sorted(deep.items())))


def section_e(pkgs, mods, info):
    print("\n(e) code and Security/\n")
    by_name = {p.name: p for p in pkgs.values()}

    def routes(start: Package) -> dict[str, list[str]]:
        """Each package reachable through path requires, with the shortest chain of requires to it."""
        seen, frontier = {start.name: [start.name]}, [start]
        while frontier:
            nxt = []
            for p in frontier:
                for n, r in p.requires:
                    if r.startswith("git:") or n in seen or r not in pkgs:
                        continue
                    seen[n] = seen[p.name] + [n]
                    nxt.append(pkgs[r])
            frontier = nxt
        return seen

    for p in sorted((p for p in pkgs.values() if p.security), key=lambda p: p.dir):
        r = routes(p)
        code = {n: c for n, c in r.items() if n != p.name and not by_name[n].security}
        imp = defaultdict(Counter)
        for n in p.modules:
            for i in mods[n].imports:
                if i in mods and not pkgs[mods[i].package].security:
                    imp[pkgs[mods[i].package].name][info[n][0]] += 1
        direct = {n for n, rr in p.requires if not rr.startswith("git:")}
        print(f"{p.name} ({p.dir}): requires {len(code)} code packages, {len(direct & set(code))} of them directly")
        rows = [[n, CODE_AREA.get(n, "?"), " > ".join(code[n][1:]),
                 " ".join(f"{t}:{k}" for t, k in sorted(imp[n].items())) or "-"]
                for n in sorted(code, key=lambda n: (len(code[n]), n))]
        if rows:
            print(table(rows, ["code package", "component", "route", "its importers here, by tree"]))
        print()
    per_tree = Counter(info[n][0] for n, m in mods.items() if pkgs[m.package].security
                       and any(i in mods and not pkgs[mods[i].package].security for i in m.imports))
    print("Security/ modules that import code, by tree: " + ", ".join(f"{t} {k}" for t, k in per_tree.most_common()))


def summary(pkgs, mods, info, loose) -> dict:
    return {
        "packages": [{"name": p.name, "dir": d, "modules": len(p.modules), "lines": sum(mods[n].lines for n in p.modules),
                      "decls": sum(mods[n].ndecls for n in p.modules), "requires": [n for n, _ in p.requires],
                      "roots": p.libs, "exes": p.exes} for d, p in sorted(pkgs.items())],
        "modules": {n: {"package": pkgs[m.package].name, "file": m.file, "lines": m.lines, "decls": dict(m.decls),
                        "generated": m.generated, "umbrella": m.umbrella, "tree": info[n][0], "area": info[n][1],
                        "component": info[n][2], "imports": m.imports} for n, m in sorted(mods.items())},
        "loose": loose,
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=ROOT, help="the checkout to map (default: this one)")
    ap.add_argument("--only", default="abcde", help="the sections to print, by letter")
    ap.add_argument("--by", choices=["component", "area"], default="component", help="what (c) condenses the graph to")
    ap.add_argument("--moves", type=Path, help="map the tree as this move map's dotted [words] would leave it")
    ap.add_argument("--top", type=int, default=15, help="rows in each top list")
    ap.add_argument("--json", type=Path, help="also write every module's facts here")
    a = ap.parse_args(argv)
    pkgs, mods, info, loose, dup = load(a.root.resolve(), read_moves(a.moves) if a.moves else None)
    if a.moves:
        print(f"mapped as {a.moves} would leave it\n")
    for letter, show in (("a", lambda: section_a(pkgs, mods, loose, dup)), ("b", lambda: section_b(pkgs, mods, info)),
                         ("c", lambda: section_c(pkgs, mods, info, a.by, a.top)),
                         ("d", lambda: section_d(pkgs, mods, info, a.top)), ("e", lambda: section_e(pkgs, mods, info))):
        if letter in a.only:
            show()
    if a.json:
        a.json.write_text(json.dumps(summary(pkgs, mods, info, loose), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
