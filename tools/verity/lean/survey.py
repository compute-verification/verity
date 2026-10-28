"""`survey.py`: the repository's Lean read as text, four ways over one reader; `size` and `unused` also read a build.

    python tools/verity/lean/survey.py map [--only abcde] [--by component|area] [--moves MAP.toml] [--top N]
                                           [--json FILE] [--root DIR]
    python tools/verity/lean/survey.py conventions [--json FILE] [--top N] [--root DIR] [PACKAGE_DIR...]
    python tools/verity/lean/survey.py size [PKG] [--prefix P] [--facts FACTS.json | --run RUN] [--lock F] [--unread]
                                            [--json]
    python tools/verity/lean/survey.py unused [PKG] [--out DIR] [--facts FACTS.json] [--queued QUEUED.json]
                                              [--frozen PREFIX ...]
    python tools/verity/lean/survey.py unused [PKG] --delete AREA --list DIR/unused.json --facts DIR/facts.json
                                              [--queued QUEUED.json] [--except MODULE ...]
    python tools/verity/lean/survey.py unused --fetch-queued QUEUED.json

The reader (`read`): a package is a directory with a lakefile or a `lean-audit.json` (`check.SKIP_DIRS` aside), with its
lakefile's requires (a path one as the package there, a git one at its revision), libraries, executables and `srcDir`,
and its lock. A module is a `.lean` file of the deepest package directory holding it, named from that package's `srcDir`;
a file under no package's `srcDir` (the auditor's own `.lean`, benchmarks' and tests' scratch programs) is loose. Its text
is read with comments stripped as `check.py` strips them. A declaration is a line that starts, after attributes and
modifiers, with a declaration keyword, named in the namespaces open where it stands. A Security module's tree is its first
component (`TREES`) and its area its next (`Definitions.Pouw.X` is `Pouw`'s, `Proofs.Properties.Nci.X` is `Nci`'s); a code
module's area is its package's (`CODE_AREA`). Text, not the elaborated environment: a declaration a macro writes is
invisible, and a definition's result is read from its header (`: Prop`), so a casing count is a lower bound.

`map` maps the repository's Lean:
(a) every Lake package: its requires, the roots its libraries list, its executables and its size in modules, lines and
    declarations, and the path requires none of its modules imports;
(b) the areas against the trees (Code, Definitions, Properties, Proofs and the properties' proofs), with the trees an
    area is missing from and its entry modules; then the same regrouped into the Glossary's components (`COMPONENT`,
    longest module prefix first; one ending in `*` is a raw prefix);
(c) the import graph condensed to components (`--by area`: to areas): who imports whom, the cycles and the fewest
    thinnest edges that break them (each with an import that makes it), the same in the Definitions tree alone, code
    that imports `Security/`, and the modules most of the tree imports; an umbrella that only umbrellas import gathers
    modules for the build, and its imports are left out of the graph;
(d) module granularity: module sizes by tree, the largest modules, generated and umbrella modules, and the directories
    holding one module or fifty and more;
(e) code and `Security/`: which code packages each Security package requires, by which chain of requires, and how many
    modules of each tree import each one's modules.
`--moves MAP.toml` maps the tree as a move map would leave it: the dotted entries of its `[words]` (the
`tools/move/*_map.toml` form) rename modules where a word starts at a part of the name, as `rename.py` does, with every
import of them; a module whose new root is another package's (`Proofs.` to `Definitions.`) moves into it. So a proposed
move shows the cycles it leaves before anyone runs it.

`conventions` surveys the Lean's form, behind the Lean standards, over every package with a
`lean-audit.json` (default: all, the auditor's fixtures aside) and their locks: module path against namespace, where
guarantees and assumptions live and how they are named, casing, theorem name forms, retired words, file sizes, heartbeat
overrides with and without a reason, imports, docstrings on spec declarations, duplicated statements, and how many
guarantees have honest and nontrivial lemmas beside them. It refuses nothing: it is what a check of a standard would count.

`size` is the size of a package's spec per area. PKG (default `Security`) is a Lake package
with a `lean-audit.json`; its spec is the modules under `--prefix` (default `Definitions`), grouped by their next
component. For each area it prints the files, their lines (and the lines that are neither blank nor comment nor
docstring), the theorems, the definitions, how many of them a listed statement reads, and how many guarantees read the
area. The definitions and what reads them come from a run of the Lean check on PKG: its `facts.json` (`--facts`), or that
of `--run RUN`, a records run (`lean_changed.py --records PKG`) the evidence store holds. A definition is one of
`spec_definitions` (projections and constructors go with their structure), read when a listed guarantee's statement reads
it, followed through `meaning`: when the lock's `reads` lists it (`--lock`, default PKG's; a run's own with `--run`), or
the lock of a package requiring PKG by path does (`check.read_by_dependents`), as the audit's `unread_spec` counts them.
Without facts the definitions are counted from the sources and the read ones are the lock's (projections and
constructors included), both marked `~`. `--unread` lists the unread ones per module.

`unused` lists the modules of a built Lake package PKG (default `Security/Proofs`) that no listed guarantee uses. Its names
are every guarantee of each lock whose statements PKG's build can hold (PKG's, each package PKG requires by path, and
each that names PKG under `proved_in`), and every other qualified name those locks hold, walked when it is a constant.
`Facts.lean` in its usage mode reads the build (`--facts`: a document it wrote before): each guarantee's module, and each
own module's constants, the ones their closure holds and the modules its constants use. A module is kept by the rules
`decide` states (used, listed in the lock, a syntax or attribute others use, named by a file outside PKG, touched by a
queued PR, a conformance check, cited by a Lean comment outside PKG, its path's end written outside PKG, needed by a
kept one); every other is unused, and those only a queued PR or the `checks`, `cited` or `suffix` rule keeps are listed
apart (`held`). `--frozen PREFIX` lists the unused modules under PREFIX apart, not for deletion. Each carries its bytes,
area, owner and the documents naming it. A guarantee no module of the build holds is reported only when no module of a
package whose lock lists it declares it.
`--out DIR` writes the usage document (`facts.json`) and the list (`unused.json`). `--delete AREA` removes the area's
unused modules that `--list` holds (less `--except`, what a PR of `--queued` touches or imports, and the frozen ones),
rewrites the imports of the modules left so the rest builds as before, and prints the documents and locks still naming
what it deleted. Nothing here judges a proof: the package's build and its records (`lean_changed.py --records PKG
--update`) decide whether the deletion holds.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import tomllib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if (C := sys.modules.get("lean_check")) is None:
    _spec = importlib.util.spec_from_file_location("lean_check", HERE / "check.py")
    C = sys.modules["lean_check"] = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(C)

# ---- the reader

#: the trees of `Security/`, longest prefix first; a module of a package outside `Security/` is code
TREES = [("Proofs.Properties", "Proofs/Properties"), ("Definitions", "Definitions"), ("Models", "Definitions"),
         ("Properties", "Properties"), ("Proofs", "Proofs")]
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
#: what a line of comment-stripped text starts with: a declaration (after attributes and modifiers), a scope, an import
DECL = re.compile(
    r"[ \t]*(?:@\[[^\]]*\][ \t]*)*(?P<mods>(?:(?:private|protected|public|noncomputable|nonrec|partial|unsafe|scoped|"
    r"local)[ \t]+)*)(?P<kw>theorem|lemma|def|abbrev|structure|class[ \t]+inductive|class|inductive|instance|opaque|axiom|"
    r"example)\b(?P<rest>.*)")
SCOPE = re.compile(r"[ \t]*(?:(?P<nc>noncomputable)[ \t]+)?(?P<kw>namespace|section|end|mutual)\b[ \t]*(?P<name>\S*)")
IMPORT = re.compile(r"[ \t]*(?:(?:public|meta|private)[ \t]+)*import[ \t]+(?:all[ \t]+)?([A-Za-z_][\w.'«»]*)")
NAME = re.compile(r"[ \t]*(?:\(priority[^)]*\)[ \t]*)?(?P<name>[^\s:({\[⦃]+)")
HEADER_END = re.compile(r":=|\bwhere\b|^[ \t]*\||\n(?=\S)", re.M)
GENERATED = re.compile(r"generated by|do not edit|@generated", re.IGNORECASE)
TYPE_KINDS = {"structure", "inductive", "class", "class inductive"}
DEF_KINDS = {"def", "abbrev", "opaque"}
THEOREM_KINDS = {"theorem", "lemma"}
KIND = {"theorem": "thm", "lemma": "thm", "def": "def", "abbrev": "def", "opaque": "def", "axiom": "def",
        "structure": "type", "inductive": "type", "class": "type", "class inductive": "type", "instance": "inst",
        "example": "ex"}


@dataclass
class Package:
    name: str
    dir: str                                        # repository path of its directory ("" for the root)
    src: str                                        # repository path of its srcDir ("." for the root)
    requires: list[tuple[str, str]]                 # (name, repository path) or (name, "git:URL@REV")
    libs: dict[str, list[str]]
    exes: int
    lock: dict | None                               # its lean-audit.json
    modules: list[str] = field(default_factory=list)

    @property
    def security(self) -> bool:
        return self.dir == "Security" or self.dir.startswith("Security/")


@dataclass
class Decl:
    kind: str
    name: str
    full: str
    line: int
    doc: str | None
    result: str                                     # prop, type, data or unknown
    signature: str
    noncomputable: bool
    private: bool = False


@dataclass
class Module:
    name: str
    file: str                                       # repository path
    pkg: Package
    path: Path                                      # the source read (a move renames `file`, not this)

    @cached_property
    def raw(self) -> str:
        return self.path.read_text(errors="replace")

    @cached_property
    def code(self) -> str:
        """The text with its comments blanked (`check.strip_comments`): the same length, so offsets and lines agree."""
        return C.strip_comments(self.raw)

    @cached_property
    def scan(self) -> tuple[list[str], list[tuple[int, int, re.Match]], bool]:
        """One pass over the comment-stripped lines: the imports, each declaration and scope line (its offset, its number
        and its match), and whether the module is an umbrella (imports and comments only)."""
        imports, events, other, pos = [], [], False, 0
        for n, s in enumerate(self.code.split("\n"), 1):
            if m := IMPORT.match(s):
                imports.append(m.group(1))
            elif s.strip():
                other = True
                if m := DECL.match(s) or SCOPE.match(s):
                    events.append((pos, n, m))
            pos += len(s) + 1
        return imports, events, bool(imports) and not other

    @cached_property
    def imports(self) -> list[str]:
        return self.scan[0]

    @cached_property
    def umbrella(self) -> bool:
        return self.scan[2]

    @cached_property
    def keywords(self) -> Counter:
        """How many declarations each keyword opens."""
        return Counter(" ".join(m.group("kw").split()) for _, _, m in self.scan[1] if m.re is DECL)

    @cached_property
    def kinds(self) -> Counter:
        out = Counter()
        for kw, n in self.keywords.items():
            out[KIND[kw]] += n
        return out

    @property
    def ndecls(self) -> int:
        return sum(v for k, v in self.kinds.items() if k != "ex")

    @cached_property
    def lines(self) -> int:
        return self.raw.count("\n") + (bool(self.raw) and not self.raw.endswith("\n"))

    @cached_property
    def generated(self) -> bool:
        return bool(GENERATED.search("\n".join(self.raw.splitlines()[:12])))

    @property
    def tree(self) -> str:
        for prefix, tree in TREES if self.pkg.security else ():
            if self.name == prefix or self.name.startswith(prefix + "."):
                return tree
        return "Code"

    @property
    def area(self) -> str:
        if (t := self.tree) == "Code":
            return CODE_AREA.get(self.pkg.name, self.pkg.name)
        parts, k = self.name.split("."), 2 if t == "Proofs/Properties" else 1
        return parts[k] if len(parts) > k else "(root)"

    @cached_property
    def decls(self) -> list[Decl]:
        """Each declaration with its name in the namespaces open where it stands: `end A.B` closes `namespace A.B`, or the
        namespaces `A` and `B` opened one by one."""
        out, stack = [], []
        for pos, line, m in self.scan[1]:
            if m.re is SCOPE:
                kw, name = m.group("kw"), m.group("name")
                if kw == "namespace":
                    stack.append(("namespace", name, False))
                elif kw in ("section", "mutual"):
                    stack.append((kw, name, bool(m.group("nc")) and kw == "section"))
                elif stack and name:
                    want = name.split(".")
                    while want and stack:
                        got = n.split(".") if (n := stack.pop()[1]) else []
                        if not got or want[-len(got):] != got:
                            break
                        want = want[:-len(got)]
                elif stack:
                    stack.pop()
                continue
            kw = " ".join(m.group("kw").split())
            nm = NAME.match(m.group("rest"))
            name = nm.group("name") if nm and kw != "example" else ""
            ns = ".".join(n for k, n, _ in stack if k == "namespace" and n)
            full = name.removeprefix("_root_.") if name.startswith("_root_.") else ".".join(x for x in (ns, name) if x)
            sig = header_of(self.code, pos + m.start("rest") + (nm.end() if name else 0))
            out.append(Decl(kw, name, full, line, doc_before(self.raw, line_start(self.raw, pos)),
                            "prop" if kw in THEOREM_KINDS else result_class(sig), sig,
                            "noncomputable" in m.group("mods") or any(x for _, _, x in stack),
                            "private" in m.group("mods").split()))
        return out


def norm(base: str, rel: str) -> str:
    return os.path.normpath(os.path.join(base, rel)) if base else os.path.normpath(rel)


def package(root: Path, d: str) -> Package:
    lake, lock = root / d / "lakefile.toml", root / d / C.POLICY
    t = tomllib.loads(lake.read_text()) if lake.is_file() else {}
    reqs = [(r["name"], norm(d, r["path"]) if "path" in r else
             f"git:{r.get('git', r.get('scope', '?'))}@{str(r.get('rev', '?'))[:10]}") for r in t.get("require", [])]
    return Package(t.get("name", Path(d).name or root.name), d, norm(d, t.get("srcDir", ".")), reqs,
                   {lib["name"]: lib.get("roots", [lib["name"]]) for lib in t.get("lean_lib", [])},
                   len(t.get("lean_exe", [])), json.loads(lock.read_text()) if lock.is_file() else None)


def read(root: Path) -> tuple[dict[str, Package], list[Module], list[str]]:
    """Every package under ``root`` by its directory, every module of them (read when first asked), and the loose `.lean`
    files."""
    files = []
    for d, dirs, fs in os.walk(root):
        dirs[:] = [x for x in dirs if x not in C.SKIP_DIRS]
        rel = Path(d).relative_to(root).as_posix()
        files += [f if rel == "." else f"{rel}/{f}" for f in fs]
    files.sort()
    pkgs = {}
    for f in files:
        d, _, base = f.rpartition("/")
        if base in ("lakefile.toml", "lakefile.lean", C.POLICY) and d not in pkgs:
            pkgs[d] = package(root, d)
    mods, loose = [], []
    for f in (f for f in files if f.endswith(".lean")):
        d = max((d for d in pkgs if d == "" or f.startswith(d + "/")), key=len, default=None)
        p = pkgs[d] if d is not None else None
        if p is None or not (p.src == "." or f.startswith(p.src + "/")):
            loose.append(f)
            continue
        mods.append(Module(os.path.relpath(f, p.src)[:-len(".lean")].replace(os.sep, "."), f, p, root / f))
        p.modules.append(mods[-1].name)
    return pkgs, mods, loose


def header_of(code: str, start: int) -> str:
    """The declaration's signature: from after its name to its `:=`, `where`, `|` or the next line at column 0."""
    window = code[start:start + 4000]
    m = HEADER_END.search(window)
    return " ".join(window[:m.start() if m else len(window)].split())


def line_start(raw: str, pos: int) -> int:
    """The start of the first attribute line above the declaration at ``pos`` (`@[simp]` on its own line)."""
    s = raw.rfind("\n", 0, pos) + 1
    while s > 0 and raw[(prev := raw.rfind("\n", 0, s - 1) + 1):s].strip().startswith("@["):
        s = prev
    return s


def doc_before(raw: str, pos: int) -> str | None:
    """The docstring (`/-- … -/`) right before a declaration at ``pos``, attributes and modifiers between them aside."""
    def back(end: int) -> int:
        while end and raw[end - 1].isspace():
            end -= 1
        return end

    end = back(pos)
    while raw.endswith("]", 0, end) and (i := raw.rfind("@[", 0, end)) >= 0 and raw.find("]", i, end - 1) < 0:
        end = back(i)
    if not raw.endswith("-/", 0, end):
        return None
    start = raw.rfind("/--", 0, end)
    if start < 0 or raw.find("-/", start + 3, end - 2) >= 0:
        return None
    return " ".join(raw[start + 3:end - 2].split())


def result_class(header: str) -> str:
    """What a declaration's header says its result is: `prop`, `type`, `data`, or `unknown` with no `:` at depth 0."""
    depth, colon = 0, -1
    for i, c in enumerate(header):
        if c in "([{⦃⟨":
            depth += 1
        elif c in ")]}⦄⟩":
            depth -= 1
        elif c == ":" and depth == 0 and header[i + 1:i + 2] != "=":
            colon = i
    if colon < 0:
        return "unknown"
    last = re.split(r"→|->", header[colon + 1:].strip())[-1].strip()
    return "prop" if last == "Prop" else "type" if re.match(r"(Type|Sort)\b", last) else "data"


def table(rows: list[list], head: list[str]) -> str:
    rows = [[str(c) for c in r] for r in rows]
    w = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(head)]
    num = [bool(rows) and all(re.match(r"^[-\d,./%]+$", r[i]) for r in rows) for i in range(len(head))]
    def line(r):
        return "  ".join((c.rjust(w[i]) if num[i] else c.ljust(w[i])) for i, c in enumerate(r)).rstrip()
    return "\n".join([line(head), line(["-" * x for x in w]), *map(line, rows)])


def fmt(n: int) -> str:
    return f"{n:,}"


def pct(n: int, d: int) -> str:
    return f"{100 * n / d:.1f}%" if d else "-"


# ---- map

COLUMNS = ["Code", "Definitions", "Properties", "Proofs", "Proofs/Properties"]
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
DOTTED = re.compile(r"^[A-Z][\w']*(?:\.[\w']+)*$")


def longest(name: str, prefixes) -> str | None:
    """The longest of ``prefixes`` that is ``name`` or a dotted prefix of it, or, for one ending in `*`, a raw prefix."""
    def hit(p):
        return name.startswith(p[:-1]) if p.endswith("*") else name == p or name.startswith(p + ".")
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
        home.setdefault(m.name.split(".")[0], m.pkg)
    out = {}
    for m in mods.values():
        n, new = copy.copy(m), rename(m.name)
        n.imports = [rename(i) for i in m.imports]
        if new != m.name:
            n.name, n.pkg = new, home.get(new.split(".")[0], m.pkg)
            n.file = norm(n.pkg.src, new.replace(".", "/") + ".lean")
            if new in out:
                print(f"--moves: {m.name} and another module both become {new}", file=sys.stderr)
        out[new] = n
    for p in pkgs.values():
        p.modules = []
    for m in out.values():
        m.pkg.modules.append(m.name)
    return out


def component_of(mod: str, area: str) -> str:
    if best := longest(mod, COMPONENT):
        return COMPONENT[best]
    return {"Population": "SampledProofs", "RemoteAudit": "Bandwidth", "Identity": "Hash", "Evaluate": "Core"}.get(area, area)


def load(root: Path, moves: dict[str, str] | None = None):
    """The packages, the modules by name (a later file of a name another declares wins), each module's (tree, area,
    component), the loose files, and the names two files declare."""
    pkgs, ms, loose = read(root)
    mods, dup = {}, defaultdict(list)
    for m in ms:
        dup[m.name].append(m.file)
        mods[m.name] = m
    if moves:
        mods = apply_moves(pkgs, mods, moves)
    info = {n: (m.tree, m.area, component_of(n, m.area)) for n, m in mods.items()}
    return pkgs, mods, info, loose, {k: v for k, v in dup.items() if len(v) > 1}


def totals(ms) -> tuple[int, int, int]:
    ms = list(ms)
    return len(ms), sum(m.lines for m in ms), sum(m.ndecls for m in ms)


def order(names) -> list[str]:
    return sorted(set(names), key=lambda x: (COMPONENTS.index(x) if x in COMPONENTS else 99, x))


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
    def graph(es):
        return {a: {b for (x, b) in es if x == a} for a in nodes}
    removed = []
    while comps := sccs(nodes, graph(live)):
        e = min((e for e in live if any(e[0] in c and e[1] in c for c in comps)), key=lambda e: (weight[e], e))
        live.discard(e)
        removed.append(e)
    for e in sorted(removed, key=lambda e: -weight[e]):
        if not sccs(nodes, graph(live | {e})):
            live.add(e)
    return sorted((e for e in removed if e not in live), key=lambda e: (weight[e], e))


def section_a(pkgs, mods, loose, dup):
    print("(a) Lake packages\n")
    rows = []
    for d, p in sorted(pkgs.items(), key=lambda kv: -totals(mods[n] for n in kv[1].modules)[1]):
        f, l, k = totals(mods[n] for n in p.modules)
        path = [n for n, r in p.requires if not r.startswith("git:")]
        git = [f"{n}@{r.rsplit('@', 1)[1]}" for n, r in p.requires if r.startswith("git:")]
        rows.append([p.name, d, f, fmt(l), fmt(k), sum(len(v) for v in p.libs.values()), p.exes, " ".join(git + path) or "-"])
    print(table(rows, ["package", "dir", "modules", "lines", "decls", "roots", "exes", "requires"]))
    f, l, k = totals(mods.values())
    code = [m for m in mods.values() if not m.pkg.security]
    print(f"\n{len(pkgs)} packages ({sum(p.security for p in pkgs.values())} under Security/), {fmt(f)} modules, "
          f"{fmt(l)} lines, {fmt(k)} declarations; code: {len(code)} modules, {fmt(sum(m.lines for m in code))} lines")
    print(f"{len(loose)} loose .lean files, under no package's srcDir: "
          + ", ".join(sorted({x.rsplit('/', 1)[0] for x in loose})))
    if dup:
        print(f"module names two files declare: {', '.join(sorted(dup))}")
    listed = [(p.name, k) for p in pkgs.values() if (k := sum(len(v) for v in p.libs.values())) > 5]
    if listed:
        print("libraries listing roots one by one (each new root edits the lakefile): "
              + ", ".join(f"{n} {k}" for n, k in sorted(listed, key=lambda x: -x[1])))
    used = defaultdict(set)
    for m in mods.values():
        for i in m.imports:
            if i in mods and mods[i].pkg.dir != m.pkg.dir:
                used[m.pkg.dir].add(mods[i].pkg.dir)
    idle = [(p.name, r) for d, p in sorted(pkgs.items())
            if (r := [n for n, r in p.requires if not r.startswith("git:") and r not in used[d]])]
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


def section_b(mods, info):
    print("\n(b) areas x trees: modules/lines\n")
    for key, label in ((1, "area"), (2, "component")):
        cell = defaultdict(list)
        for n, i in info.items():
            cell[(i[key], i[0])].append(mods[n])
        rows, holes, single = [], [], []
        for a in order(k[0] for k in cell):
            r, present = [a], []
            for t in COLUMNS:
                f, l, _ = totals(cell.get((a, t), []))
                r.append(f"{f}/{fmt(l)}" if f else "-")
                if f:
                    present.append(t)
            r.append(fmt(sum(totals(cell.get((a, t), []))[1] for t in COLUMNS)))
            rows.append(r + ([entries(mods, a)] if key == 1 else []))
            if a in OUTSIDE:
                continue
            sec = [t for t in COLUMNS[1:4] if t in present]
            if len(sec) == 1 and "Code" not in present:
                single.append(f"{a} ({sec[0]})")
            if missing := [t for t in COLUMNS[1:4] if t not in present]:
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


def graph(mods, info, k: int, keep=lambda n: True):
    """Edges between groups (`info[n][k]`) with their import lines and one import that makes each, less the imports of
    an umbrella only umbrellas import: it gathers a tree for the build, and nothing reads through it."""
    weight, example = Counter(), {}
    read_through = {i for m in mods.values() if not m.umbrella for i in m.imports}
    for n, m in mods.items():
        if not keep(n) or (m.umbrella and n not in read_through):
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


def section_c(mods, info, by: str, top: int):
    k = 2 if by == "component" else 1
    print(f"\n(c) the import graph, condensed to {by}s\n")
    weight, example = graph(mods, info, k)
    ext = defaultdict(Counter)
    for n, m in mods.items():
        for i in m.imports:
            if i not in mods:
                ext[info[n][k]][i.split(".")[0]] += 1
    rows = []
    for g in order(v[k] for v in info.values()):
        out = sorted((b for a, b in weight if a == g), key=lambda b: -weight[(g, b)])
        fan_in = {a for a, b in weight if b == g and a not in OUTSIDE}
        lib = ", ".join(f"{r} {c}" for r, c in ext[g].most_common(3))
        rows.append([g, len(fan_in), len(out), " ".join(f"{b}({weight[(g, b)]})" for b in out) or "-", lib or "-"])
    print(table(rows, [by, "fan-in", "fan-out", "imports (import lines)", "outside the repository"]))
    cycles(f"cycles among {by}s (Tooling and the roots aside)", weight, example)
    cycles(f"cycles among {by}s in the Definitions tree alone",
           *graph(mods, info, k, keep=lambda n: info[n][0] == "Definitions"))
    bad = [(n, i) for n, m in mods.items() if not m.pkg.security for i in m.imports if i in mods and mods[i].pkg.security]
    print("\ncode importing Security/: " + (", ".join(f"{n} -> {i}" for n, i in bad) if bad else "none"))
    counts, total = closure_counts(mods), len(mods)
    print(f"\nmodules most of the tree imports, directly or not ({fmt(total)} modules):")
    rows = [[n, mods[n].pkg.name, fmt(counts[n]), f"{100 * counts[n] / total:.0f}%", fmt(mods[n].lines)]
            for n in sorted(counts, key=lambda x: (-counts[x], x))[:top]]
    print(table(rows, ["module", "package", "importers", "share", "lines"]))
    rows = [[n, info[n][0], fmt(counts[n]), fmt(mods[n].lines)]
            for n in sorted((n for n in mods if mods[n].pkg.security), key=lambda x: (-counts[x], x))[:top]]
    print("\nthe same within Security/:")
    print(table(rows, ["module", "tree", "importers", "lines"]))
    direct = Counter(i for m in mods.values() for i in m.imports if i in mods)
    print("\nmost imported directly:")
    print(table([[n, c, info[n][k]] for n, c in direct.most_common(top)], ["module", "importers", by]))


def section_d(mods, info, top: int):
    print("\n(d) module granularity\n")
    bounds = [50, 100, 200, 500, 1000, 2000, 5000]
    def bucket(n):
        return next((i for i, b in enumerate(bounds) if n <= b), len(bounds))
    labels = [f"<={b}" for b in bounds] + [f">{bounds[-1]}"]
    by_tree = defaultdict(Counter)
    for n, m in mods.items():
        by_tree[info[n][0]][bucket(m.lines)] += 1
    rows = [[t, *[by_tree[t][i] for i in range(len(labels))], sum(by_tree[t].values())] for t in COLUMNS]
    print(table(rows, ["lines \\ tree", *labels, "modules"]))
    ls = sorted(m.lines for m in mods.values())
    def q(p):
        return ls[min(len(ls) - 1, int(p * len(ls)))]
    print(f"\nmodule lines: median {q(.5)}, p90 {q(.9)}, p99 {q(.99)}, max {fmt(ls[-1])}")
    gen = [m for m in mods.values() if m.generated]
    print(f"generated modules: {len(gen)}, {fmt(sum(m.lines for m in gen))} lines")
    rows = [[m.name, m.pkg.name, fmt(m.lines), m.ndecls, m.kinds["thm"], m.kinds["def"], "yes" if m.generated else ""]
            for m in sorted(mods.values(), key=lambda m: -m.lines)[:top]]
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
                if i in mods and not mods[i].pkg.security:
                    imp[mods[i].pkg.name][info[n][0]] += 1
        direct = {n for n, rr in p.requires if not rr.startswith("git:")}
        print(f"{p.name} ({p.dir}): requires {len(code)} code packages, {len(direct & set(code))} of them directly")
        rows = [[n, CODE_AREA.get(n, "?"), " > ".join(code[n][1:]),
                 " ".join(f"{t}:{k}" for t, k in sorted(imp[n].items())) or "-"]
                for n in sorted(code, key=lambda n: (len(code[n]), n))]
        if rows:
            print(table(rows, ["code package", "component", "route", "its importers here, by tree"]))
        print()
    per_tree = Counter(info[n][0] for n, m in mods.items() if m.pkg.security
                       and any(i in mods and not mods[i].pkg.security for i in m.imports))
    print("Security/ modules that import code, by tree: " + ", ".join(f"{t} {k}" for t, k in per_tree.most_common()))


def summary(pkgs, mods, info, loose) -> dict:
    return {
        "packages": [{"name": p.name, "dir": d, "modules": len(p.modules), "lines": sum(mods[n].lines for n in p.modules),
                      "decls": sum(mods[n].ndecls for n in p.modules), "requires": [n for n, _ in p.requires],
                      "roots": p.libs, "exes": p.exes} for d, p in sorted(pkgs.items())],
        "modules": {n: {"package": m.pkg.name, "file": m.file, "lines": m.lines, "decls": dict(m.kinds),
                        "generated": m.generated, "umbrella": m.umbrella, "tree": info[n][0], "area": info[n][1],
                        "component": info[n][2], "imports": m.imports} for n, m in sorted(mods.items())},
        "loose": loose,
    }


def run_map(a) -> int:
    pkgs, mods, info, loose, dup = load(a.root.resolve(), read_moves(a.moves) if a.moves else None)
    if a.moves:
        print(f"mapped as {a.moves} would leave it\n")
    for letter, show in (("a", lambda: section_a(pkgs, mods, loose, dup)), ("b", lambda: section_b(mods, info)),
                         ("c", lambda: section_c(mods, info, a.by, a.top)), ("d", lambda: section_d(mods, info, a.top)),
                         ("e", lambda: section_e(pkgs, mods, info))):
        if letter in a.only:
            show()
    if a.json:
        a.json.write_text(json.dumps(summary(pkgs, mods, info, loose), indent=1))
    return 0


# ---- conventions

#: namespaces of dependencies a package extends for dot notation (`Finset.sum_foo`), not its own
DEPENDENCY_ROOTS = {
    "Array", "BitVec", "Bool", "ByteArray", "Char", "Complex", "Decidable", "ENNReal", "EReal", "Equiv", "Fin", "Finite",
    "Finset", "Finsupp", "Fintype", "Function", "HashMap", "Int", "Lean", "List", "MeasureTheory", "Multiset", "NNReal",
    "Nat", "Option", "Ordinal", "PMF", "Polynomial", "Prod", "ProbabilityTheory", "Quotient", "Rat", "Real", "Set",
    "Std", "String", "Subtype", "Sum", "UInt16", "UInt32", "UInt64", "UInt8", "USize", "Vector", "ZMod", "Matrix",
    "OracleComp", "OracleSpec", "Sym2", "SimpleGraph", "Filter", "Cardinal", "Mathlib", "Encodable", "Prop", "Pi", "Sigma",
    "Except", "IO", "Thunk", "Ordering", "Nat.Prime", "Rel", "Relation", "Setoid", "Quot", "Order", "Monotone", "Antitone",
    "StateT", "ExceptT", "ReaderT", "Id", "Bitwise", "Float", "Subarray", "ByteSlice", "SProd", "Module", "Submodule",
    "LinearMap", "Mathlib.Meta", "Tactic", "Mul", "Add", "Fintype.card", "WithTop", "WithBot", "ENat",
    "Probability", "Dist", "Metric", "Nonempty", "HEq", "Eq", "Ne", "Iff", "And", "Or", "Not", "Exists",
}
BANNED_CODE = re.compile(r"\b(sorry|native_decide)\b|#exit\b")
UNINFORMATIVE = re.compile(r"^(?:aux|helper|key|main|claim|step|lemma|thm|foo|bar|test|tmp|temp|h|this|goal|fact)\d*'*$|"
                           r"_aux\d*'*$|_helper\d*'*$")
#: terms the Glossary retires (its "Older names")
RETIRED_PROSE = {
    "leg": re.compile(r"\blegs?\b", re.I),
    "warden": re.compile(r"\bwardens?\b", re.I),
    "tap": re.compile(r"\btaps?\b", re.I),
    "physical block": re.compile(r"\bphysical[ -]blocks?\b", re.I),
    "provider name": re.compile(r"\b(?:nebius|coreweave|aws|gcp|azure|lambda ?labs|runpod)\b", re.I),
    "proof unit": re.compile(r"\bproof[ -]units?\b", re.I),
    "isolation/replay/verification unit": re.compile(r"\b(?:isolation|replay|verification)[ -]units?\b", re.I),
}
#: the same words as an identifier's components (`legBound`, `warden_rows`, `NebiusHost`)
RETIRED_WORDS = {"leg": {"leg", "legs"}, "warden": {"warden", "wardens"}, "tap": {"tap", "taps"},
                 "provider name": {"nebius", "coreweave", "aws", "gcp", "azure"}}
#: the prefixes a block takes (the Glossary's "block")
BLOCK_PREFIXES = {"proof", "sampling", "server-time", "time", "serving", "server", "isolation", "replay", "inference",
                  "multiplication", "check", "flock", "c-flock", "flock's", "c-flock's"}
PROSE_BLOCK = re.compile(r"(?:\b([\w'-]+)[ \t]+)?\bblocks?\b", re.I)
WORDS = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
COMMENT = re.compile(r"/-[\s\S]*?-/|--[^\n]*")


def locked(pkgs: dict[str, Package]) -> list[str]:
    """The directories of the packages with a `lean-audit.json`, the auditor's fixtures aside."""
    return sorted(d for d, p in pkgs.items() if p.lock is not None and not d.startswith("tools/verity/lean/tests/"))


def tree(f: Module) -> str:
    return f.tree.lower()


def words(name: str) -> list[str]:
    return [w.lower() for part in re.split(r"[._']", name) for w in WORDS.findall(part)]


def dist(xs: list[int]) -> dict:
    if not xs:
        return {}
    s = sorted(xs)
    def q(p):
        return s[min(len(s) - 1, int(p * len(s)))]
    return {"n": len(s), "sum": sum(s), "p50": int(statistics.median(s)), "p90": q(0.9), "p99": q(0.99), "max": s[-1]}


def namespace_class(f: Module, d: Decl) -> str:
    """How a declaration's namespace sits against its module's path: `path` (one is a prefix of the other, the tree's root
    module aside), `area` (the same area root, then apart), `extension` (a dependency's namespace, for dot notation),
    `none` (the root namespace) or `foreign` (another of the repository's roots)."""
    ns = d.full.split(".")[:-1]
    if not ns:
        return "none"
    mod, t = f.name.split("."), tree(f)
    cands = [mod, *([mod[1:]] if t in ("definitions", "proofs", "properties") else []),
             *([mod[1:], mod[2:]] if t == "proofs/properties" else [])]
    if any((k := min(len(c), len(ns))) and c[:k] == ns[:k] for c in cands):
        return "path"
    if ns[0] == (f.area if t != "code" else mod[0]) or (t == "proofs/properties" and ns[0] == "Properties"):
        return "area"
    return "extension" if ns[0] in DEPENDENCY_ROOTS or ".".join(ns[:2]) in DEPENDENCY_ROOTS else "foreign"


def overrides(root: Path, files: list[Module]) -> list[dict]:
    """Each heartbeat override, as the Lean check lists it (`check.heartbeats`), with its package and tree."""
    by_pkg: dict[str, dict[str, Module]] = defaultdict(dict)
    for f in files:
        if "maxHeartbeats" in f.raw:
            by_pkg[f.pkg.dir][os.path.relpath(f.file, f.pkg.dir or ".")] = f
    return [h | {"package": d, "tree": tree(fs[h["file"]])} for d, fs in sorted(by_pkg.items())
            for h in C.heartbeats(root / d, {f.name: rel for rel, f in fs.items()}, {})]


def survey(root: Path, dirs: list[str], top: int) -> dict:
    """The counts over the packages in ``dirs`` (repository paths; none: every one `locked` lists)."""
    pkgs, ms, _ = read(root)
    dirs = dirs or locked(pkgs)
    files = [m for m in ms if m.pkg.dir in dirs]
    locks = {d: pkgs[d].lock or {} for d in dirs}
    by_full: dict[str, list[tuple[Module, Decl]]] = defaultdict(list)
    by_tree: dict[str, list[Module]] = defaultdict(list)
    for f in files:
        by_tree[tree(f)].append(f)
        for d in f.decls:
            if d.full:
                by_full[d.full].append((f, d))
    trees, of = sorted(by_tree), lambda t: by_tree[t]
    out: dict = {"files": len(files), "packages": len(dirs), "decls": sum(len(f.decls) for f in files)}

    # ---- naming: module path against namespace
    ns = {t: Counter() for t in trees}
    worst_ns: Counter = Counter()
    roots_by_area: dict[str, Counter] = defaultdict(Counter)
    foreign_roots: Counter = Counter()
    for f in files:
        for d in f.decls:
            if not d.name or d.kind in ("example", "instance"):
                continue
            c = namespace_class(f, d)
            ns[tree(f)][c] += 1
            parts = d.full.split(".")
            if len(parts) > 1:
                roots_by_area[f"{tree(f)}:{f.area}"][parts[0]] += 1
            if c in ("foreign", "none"):
                worst_ns[f.file] += 1
                foreign_roots[parts[0] if c == "foreign" else "(root namespace)"] += 1
    out["namespace"] = {"by_tree": {t: dict(c) for t, c in ns.items()}, "worst_files": worst_ns.most_common(top),
                        "foreign_roots": foreign_roots.most_common(top),
                        "area_roots": {a: c.most_common(6) for a, c in sorted(roots_by_area.items())
                                       if len(c) > 1 or next(iter(c)) != a.split(":")[1]}}

    # ---- naming: casing and theorem forms
    case, forms, n_thm = Counter(), Counter(), 0
    case_ex: dict[str, list[str]] = defaultdict(list)
    form_ex: dict[str, list[str]] = defaultdict(list)

    def flag(key: str, f: Module, d: Decl, into=case, ex=case_ex):
        into[key] += 1
        if len(ex[key]) < top:
            ex[key].append(f"{d.full} ({f.file}:{d.line})")

    for f in files:
        for d in f.decls:
            last = d.full.split(".")[-1] if d.full else ""
            if not last or d.kind in ("example", "instance", "axiom"):
                continue
            upper = last[:1].isupper()
            if d.kind in TYPE_KINDS:
                case["type: total"] += 1
                if not upper:
                    flag("type: lowerCamel", f, d)
            elif d.kind in DEF_KINDS:
                case[f"def ({d.result}): total"] += 1
                if d.result in ("prop", "type") and not upper:
                    flag(f"def ({d.result}): lowerCamel", f, d)
                elif d.result == "data" and upper:
                    flag("def (data): UpperCamel", f, d)
                if not upper and "_" in last.strip("_"):
                    flag("def: snake_case", f, d)
            elif d.kind in THEOREM_KINDS:
                n_thm += 1
                parent = d.full.split(".")[-2] if "." in d.full else ""
                if upper:
                    if parent == "SecurityProofs":
                        flag("theorem: UpperCamel, SecurityProofs.G", f, d)
                    elif re.search(r"_(fails|meets)\b", last):
                        flag("theorem: UpperCamel, claim witness", f, d)
                    else:
                        flag("theorem: UpperCamel, other", f, d)
                elif "_" not in last and re.search(r"[a-z][A-Z]", last) and len(WORDS.findall(last)) >= 3:
                    flag("theorem: camelCase, no underscore", f, d)
                for key, pat in (("_of_", r"_of_"), ("_iff", r"_iff(_|$)"), ("_le/_lt/_ge/_eq/_ne", r"_(le|lt|ge|eq|ne)(_|'|$)"),
                                 ("_honest", r"honest"), ("_nontrivial", r"nontrivial"), ("_fails/_meets", r"_(fails|meets)"),
                                 ("_sound/_complete", r"_(sound|complete)"), ("primed (')", r"'$"),
                                 ("numbered (foo2)", r"[A-Za-z]\d+'*$"), ("uninformative (aux, key, step3)", UNINFORMATIVE.pattern)):
                    if re.search(pat, last):
                        flag(key, f, d, forms, form_ex)
    out["casing"] = {"counts": dict(case), "examples": dict(case_ex)}
    out["theorem_forms"] = {"theorems": n_thm, "counts": dict(forms), "examples": dict(form_ex)}

    # ---- naming: retired words, in identifiers and in prose
    retired_id, retired_prose, block_prev = Counter(), Counter(), Counter()
    retired_id_ex: dict[str, list[str]] = defaultdict(list)
    retired_prose_files: dict[str, Counter] = defaultdict(Counter)
    for f in files:
        for full, line in [(d.full, d.line) for d in f.decls if d.full] + [(f.name, 0)]:
            w = words(full)
            hits = {key for key, ws in RETIRED_WORDS.items() if ws & set(w)}
            for i, x in enumerate(w):
                if x in ("block", "blocks"):
                    prev = w[i - 1] if i else ""
                    if prev == "physical":
                        hits.add("physical block")
                    elif prev not in BLOCK_PREFIXES:
                        hits.add("unprefixed block")
            for h in hits:
                retired_id[h] += 1
                if len(retired_id_ex[h]) < top:
                    retired_id_ex[h].append(f"{full} ({f.file}{':' + str(line) if line else ''})")
        prose = "\n".join(COMMENT.findall(f.raw))
        for key, pat in RETIRED_PROSE.items():
            n = len(pat.findall(prose))
            retired_prose[key] += n
            if n:
                retired_prose_files[key][f.file] += n
        for m in PROSE_BLOCK.finditer(prose):
            prev = (m.group(1) or "").lower()
            if prev not in BLOCK_PREFIXES:
                retired_prose["unprefixed block"] += 1
                retired_prose_files["unprefixed block"][f.file] += 1
                block_prev[prev or "(none)"] += 1
    out["retired"] = {"identifiers": dict(retired_id), "identifier_examples": dict(retired_id_ex),
                      "prose": dict(retired_prose),
                      "prose_files": {k: {"files": len(c), "worst": c.most_common(5)} for k, c in retired_prose_files.items()},
                      "prose_block_preceded_by": block_prev.most_common(15)}

    # ---- structure: sizes, imports, heartbeats, noncomputable, banned tokens
    out["file_lines"] = {"by_tree": {t: dist([f.lines for f in of(t)]) for t in trees},
                         "over": {n: sum(f.lines > n for f in files) for n in (500, 1000, 2000, 5000)},
                         "over_by_tree": {t: {n: sum(f.lines > n for f in of(t)) for n in (500, 1000, 2000)} for t in trees},
                         "worst": [(f.file, f.lines, len(f.decls)) for f in sorted(files, key=lambda f: -f.lines)[:top]]}
    out["decls_per_file"] = {"by_tree": {t: dist([len(f.decls) for f in of(t)]) for t in trees},
                             "over_300": sum(len(f.decls) > 300 for f in files)}
    out["imports"] = {"by_tree": {t: dist([len(f.imports) for f in of(t)]) for t in trees},
                      "wholesale_mathlib": {t: sum("Mathlib" in f.imports for f in of(t)) for t in trees},
                      "wholesale_mathlib_files": [f.file for f in files if "Mathlib" in f.imports and tree(f) in
                                                  ("definitions", "code")][:top],
                      "worst": [(f.file, len(f.imports)) for f in sorted(files, key=lambda f: -len(f.imports))[:top]]}
    hb = overrides(root, files)
    out["heartbeats"] = {
        "overrides": len(hb), "with_reason": sum(bool(h["reason"]) for h in hb), "unlimited (0)": sum(h["value"] == 0 for h in hb),
        "by_tree": {t: {"overrides": sum(h["tree"] == t for h in hb),
                        "no_reason": sum(h["tree"] == t and not h["reason"] for h in hb)} for t in trees},
        "files_with_override": len({(h["package"], h["file"]) for h in hb}),
        "max_value": max((h["value"] for h in hb), default=0),
        "worst_files_no_reason": Counter(norm(h["package"], h["file"]) for h in hb if not h["reason"]).most_common(top)}
    def nc_sections(f):
        return sum(m.re is SCOPE and m.group("kw") == "section" and bool(m.group("nc")) for *_, m in f.scan[1])
    out["noncomputable"] = {t: {"sections": sum(nc_sections(f) for f in of(t)),
                                "files_with_section": sum(nc_sections(f) > 0 for f in of(t)),
                                "decls": sum(d.noncomputable for f in of(t) for d in f.decls)} for t in trees}
    banned = [(f.file, [f"{f.code.count(chr(10), 0, m.start()) + 1}:{m.group(0)}" for m in BANNED_CODE.finditer(f.code)])
              for f in files]
    out["banned_tokens"] = [b for b in banned if b[1]][:top]

    # ---- structure: duplicated statements across areas
    sigs: dict[str, list[tuple[Module, Decl]]] = defaultdict(list)
    by_name: dict[str, list[tuple[Module, Decl]]] = defaultdict(list)
    for f in files:
        for d in f.decls:
            if d.kind in THEOREM_KINDS and len(d.signature) >= 40:
                sigs[d.signature].append((f, d))
            if d.kind in THEOREM_KINDS and d.full:
                by_name[d.full.split(".")[-1]].append((f, d))
    def spread(xs):
        return len({(x.pkg.dir, x.area) for x, _ in xs}) > 1
    dup = sorted(((s, xs) for s, xs in sigs.items() if spread(xs)), key=lambda e: -len(e[1]))
    out["duplicates"] = {
        "statements_in_2+_areas": len(dup), "theorems_in_them": sum(len(xs) for _, xs in dup),
        "worst": [{"statement": s[:160], "at": [f"{d.full} ({f.file}:{d.line})" for f, d in xs[:4]], "copies": len(xs)}
                  for s, xs in dup[:top]],
        "names_in_2+_areas": sum(spread(xs) and len(n) > 6 for n, xs in by_name.items())}

    # ---- organization: areas per tree, theorems and docstrings in the definitions tree
    areas = {t: sorted({f.area for f in of(t) if f.area != "(root)"}) for t in trees if t != "code"}
    every = sorted(set().union(*areas.values())) if areas else []
    out["areas"] = {"by_tree": areas, "missing": {a: [t for t in areas if a not in areas[t]] for a in every
                                                  if any(a not in areas[t] for t in areas)}}
    defs_tree = of("definitions")
    thm_in_defs = Counter(f.file for f in defs_tree for d in f.decls if d.kind in THEOREM_KINDS)
    reads = {n for lock in locks.values() for mod in (lock.get("reads") or {}).values() for n in mod.get("definitions", {})}
    spec_defs = [(f, d) for f in defs_tree for d in f.decls if d.kind in DEF_KINDS | TYPE_KINDS and d.full]
    unread_by_file = Counter(f.file for f, d in spec_defs if d.full not in reads)
    out["definitions_tree"] = {
        "files": len(defs_tree), "decls": sum(len(f.decls) for f in defs_tree),
        "theorems": sum(thm_in_defs.values()), "files_with_theorems": len(thm_in_defs),
        "theorems_worst": thm_in_defs.most_common(top),
        "securityproofs_decls": sum(".SecurityProofs." in d.full for f in defs_tree for d in f.decls),
        "spec_definitions": len(spec_defs), "read_by_no_guarantee": sum(unread_by_file.values()),
        "unread_worst_files": unread_by_file.most_common(top)}

    def docs(fs: list[Module]) -> dict:
        ds = [d for f in fs for d in f.decls if d.kind in DEF_KINDS | TYPE_KINDS and d.name]
        return {"decls": len(ds), "no_docstring": sum(d.doc is None for d in ds)}

    def is_g(f):
        return "Guarantees" in f.name.split(".")

    def is_a(f):
        return "Assumptions" in f.name.split(".")
    nodoc = Counter(f.file for f in defs_tree for d in f.decls if d.kind in DEF_KINDS | TYPE_KINDS and d.name and d.doc is None)
    out["docstrings"] = {
        "definitions_tree": docs(defs_tree), "guarantees_modules": docs([f for f in defs_tree if is_g(f)]),
        "assumptions_modules": docs([f for f in defs_tree if is_a(f)]),
        "vocabulary_modules": docs([f for f in defs_tree if not is_g(f) and not is_a(f)]),
        "properties_tree": docs(of("properties")), "code": docs(of("code")),
        "no_module_doc": {t: sum("/-!" not in f.raw for f in of(t)) for t in trees},
        "worst_definitions_files": nodoc.most_common(top)}

    # ---- the locks: where guarantees and assumptions live, their names, honest/nontrivial lemmas
    all_thm_lower = {d.full.lower() for f in files for d in f.decls if d.kind in THEOREM_KINDS and d.full}
    honest_names = [n for n in all_thm_lower if "honest" in n]
    nontrivial_names = [n for n in all_thm_lower if "nontrivial" in n]
    lk = {}
    for pkg_rel, lock in locks.items():
        gs = lock.get("guarantees") or {}
        if not gs and not lock.get("assumptions"):
            continue
        frozen = (lock.get("shape") or {}).get("frozen", {})
        g_form, g_tree, g_root = Counter(), Counter(), Counter()
        both = one = 0
        for g in gs:
            parts = g.split(".")
            g_root[parts[0]] += 1
            if "SecurityProofs" in parts[1:-1]:
                # `check.four_named`'s form, read from the text: `X.SecurityProofs.G` beside a definition `X.Guarantees.G`
                at = parts.index("SecurityProofs", 1)
                spec = ".".join([*parts[:at], "Guarantees", *parts[at + 1:]])
                g_form["four names" if spec in by_full else "X.SecurityProofs.G, no X.Guarantees.G"] += 1
            else:
                g_form["other"] += 1
            where = by_full.get(g)
            g_tree[tree(where[0][0]) if where else "not found in text"] += 1
            key = parts[-1].lower()
            h, nt = any(key in n for n in honest_names), any(key in n for n in nontrivial_names)
            both += h and nt
            one += h != nt
        asm = lock.get("assumptions") or {}
        a_form, a_mod, a_doc = Counter(), Counter(), Counter()
        for a in asm:
            a_form["<Area>.Assumptions.X" if ".Assumptions." in a else "other"] += 1
            if not (where := by_full.get(a)):
                a_mod["not found in text"] += 1
                continue
            f, d = where[0]
            a_mod["Assumptions module" if is_a(f) else f"{tree(f)}, not an Assumptions module"] += 1
            a_doc["docstring" if d.doc else "no docstring"] += 1
        lk[pkg_rel] = {"guarantees": len(gs), "frozen": len(frozen),
                       "frozen_by_category": dict(Counter(v.split()[0].strip(":,;") for v in frozen.values())),
                       "forms": dict(g_form), "declared_in": dict(g_tree), "roots": dict(g_root.most_common()),
                       "honest_and_nontrivial": both, "only_one": one, "neither": len(gs) - both - one,
                       "four_names_roots": lock.get("four_names"), "assumptions": len(asm), "assumption_names": dict(a_form),
                       "assumption_modules": dict(a_mod), "assumption_docstrings": dict(a_doc)}
    out["locks"] = lk
    out["honest_nontrivial_theorems"] = {"honest": len(honest_names), "nontrivial": len(nontrivial_names)}
    return out


def render(r: dict, top: int) -> str:
    L = [f"{r['packages']} packages, {r['files']} files, {r['decls']} declarations (text)"]
    def sec(t):
        L.append(f"\n== {t}")
    sec("namespace against module path (declarations; path: one a prefix of the other, the tree root aside)")
    for t, c in r["namespace"]["by_tree"].items():
        n = sum(c.values())
        L.append(f"  {t:18} {n:6}  " + "  ".join(f"{k} {v} ({pct(v, n)})" for k, v in sorted(c.items(), key=lambda e: -e[1])))
    L.append("  foreign roots: " + ", ".join(f"{k} {v}" for k, v in r["namespace"]["foreign_roots"]))
    L.append("  areas whose declarations use another root: " + "; ".join(
        f"{a}: " + ", ".join(f"{k} {v}" for k, v in c) for a, c in list(r["namespace"]["area_roots"].items())[:30]))
    L.append("  worst files (foreign or root): " + ", ".join(f"{k} {v}" for k, v in r["namespace"]["worst_files"]))
    sec("casing")
    for k, v in sorted(r["casing"]["counts"].items()):
        L.append(f"  {k:42} {v}")
    for k, ex in r["casing"]["examples"].items():
        L.append(f"  e.g. {k}: " + "; ".join(ex[:5]))
    sec(f"theorem name forms ({r['theorem_forms']['theorems']} theorems)")
    for k, v in sorted(r["theorem_forms"]["counts"].items(), key=lambda e: -e[1]):
        L.append(f"  {k:34} {v:6} ({pct(v, r['theorem_forms']['theorems'])})")
    for k in ("uninformative (aux, key, step3)", "numbered (foo2)"):
        L.append(f"  e.g. {k}: " + "; ".join(r["theorem_forms"]["examples"].get(k, [])[:6]))
    sec("retired words")
    L.append("  identifiers (declaration names, module paths): " + json.dumps(r["retired"]["identifiers"]))
    for k, ex in r["retired"]["identifier_examples"].items():
        L.append(f"    {k}: " + "; ".join(ex[:6]))
    L.append("  prose (docstrings, comments): " + json.dumps(r["retired"]["prose"]))
    for k, v in r["retired"]["prose_files"].items():
        L.append(f"    {k}: {v['files']} files, worst " + ", ".join(f"{a} {b}" for a, b in v["worst"]))
    L.append("    'block' preceded by: " + ", ".join(f"{a} {b}" for a, b in r["retired"]["prose_block_preceded_by"]))
    sec("file lines")
    for t, d in r["file_lines"]["by_tree"].items():
        L.append(f"  {t:18} {json.dumps(d)}  over 500/1000/2000: {json.dumps(r['file_lines']['over_by_tree'][t])}")
    L.append(f"  over 500/1000/2000/5000 lines: {json.dumps(r['file_lines']['over'])}")
    L.append("  worst: " + ", ".join(f"{a} {b} lines/{c} decls" for a, b, c in r["file_lines"]["worst"]))
    L.append(f"  declarations per file: {json.dumps(r['decls_per_file'])}")
    sec("imports")
    for t, d in r["imports"]["by_tree"].items():
        L.append(f"  {t:18} {json.dumps(d)}  `import Mathlib`: {r['imports']['wholesale_mathlib'][t]}")
    L.append("  `import Mathlib` in definitions/code: " + ", ".join(r["imports"]["wholesale_mathlib_files"]))
    L.append("  most imports: " + ", ".join(f"{a} {b}" for a, b in r["imports"]["worst"]))
    sec("heartbeat overrides")
    h = r["heartbeats"]
    L.append(f"  {h['overrides']} overrides in {h['files_with_override']} files, {h['with_reason']} with a reason "
             f"({pct(h['with_reason'], h['overrides'])}), {h['unlimited (0)']} unlimited, max {h['max_value']}")
    L.append("  by tree: " + json.dumps(h["by_tree"]))
    L.append("  most without a reason: " + ", ".join(f"{a} {b}" for a, b in h["worst_files_no_reason"]))
    sec("noncomputable, banned tokens")
    L.append("  " + json.dumps(r["noncomputable"]))
    L.append("  sorry/native_decide/#exit in code: " + json.dumps(r["banned_tokens"]))
    sec("duplicated theorem statements across areas")
    d = r["duplicates"]
    L.append(f"  {d['statements_in_2+_areas']} statements in two or more areas ({d['theorems_in_them']} theorems); "
             f"{d['names_in_2+_areas']} theorem names in two or more areas")
    for x in d["worst"][:top]:
        L.append(f"    {x['copies']}x {x['statement'][:100]} | {'; '.join(x['at'][:3])}")
    sec("areas per tree")
    for t, a in r["areas"]["by_tree"].items():
        L.append(f"  {t:18} {', '.join(a)}")
    L.append("  missing: " + json.dumps(r["areas"]["missing"]))
    sec("the definitions tree (the spec)")
    dt = r["definitions_tree"]
    L.append(f"  {dt['files']} files, {dt['decls']} declarations; {dt['theorems']} theorems in {dt['files_with_theorems']} "
             f"files; {dt['securityproofs_decls']} SecurityProofs declarations; {dt['spec_definitions']} definitions, "
             f"{dt['read_by_no_guarantee']} read by no guarantee ({pct(dt['read_by_no_guarantee'], dt['spec_definitions'])})")
    L.append("  most theorems: " + ", ".join(f"{a} {b}" for a, b in dt["theorems_worst"]))
    L.append("  most unread definitions: " + ", ".join(f"{a} {b}" for a, b in dt["unread_worst_files"]))
    sec("docstrings on definitions (def, abbrev, structure, inductive, class)")
    for k in ("definitions_tree", "guarantees_modules", "assumptions_modules", "vocabulary_modules", "properties_tree", "code"):
        x = r["docstrings"][k]
        L.append(f"  {k:20} {x['decls']:6} decls, {x['no_docstring']:5} without ({pct(x['no_docstring'], x['decls'])})")
    L.append("  files without a module docstring: " + json.dumps(r["docstrings"]["no_module_doc"]))
    L.append("  worst definitions files: " + ", ".join(f"{a} {b}" for a, b in r["docstrings"]["worst_definitions_files"]))
    sec("locks: guarantees and assumptions")
    for p, x in r["locks"].items():
        L.append(f"  {p}: {x['guarantees']} guarantees ({x['frozen']} frozen: {json.dumps(x['frozen_by_category'])})")
        L.append(f"    forms {json.dumps(x['forms'])}; declared in {json.dumps(x['declared_in'])}")
        L.append(f"    roots {json.dumps(x['roots'])}")
        L.append(f"    honest and nontrivial lemmas: both {x['honest_and_nontrivial']}, one {x['only_one']}, "
                 f"neither {x['neither']}")
        L.append(f"    {x['assumptions']} assumptions: names {json.dumps(x['assumption_names'])}; in "
                 f"{json.dumps(x['assumption_modules'])}; {json.dumps(x['assumption_docstrings'])}")
    L.append(f"  theorems named honest/nontrivial anywhere: {json.dumps(r['honest_nontrivial_theorems'])}")
    return "\n".join(L)


def run_conventions(a) -> int:
    root = a.root.resolve()
    dirs = [d.resolve().relative_to(root).as_posix() for d in a.packages]
    r = survey(root, ["" if d == "." else d for d in dirs], a.top)
    print(render(r, a.top))
    if a.json:
        a.json.write_text(json.dumps(r, indent=1, ensure_ascii=False, default=str) + "\n")
    return 0


# ---- size

SIZE_DEFINITIONS = DEF_KINDS | TYPE_KINDS | {"instance"}


def code_lines(code: str) -> int:
    """The lines of comment-stripped ``code`` (`Module.code`) holding something besides blanks."""
    return sum(bool(s.strip()) for s in code.splitlines())


def under(prefix: str, module: str) -> str | None:
    """The area of ``module`` under ``prefix`` (`Definitions.Pouw.Basic` → `Pouw`), `(root)` for ``prefix`` itself."""
    if module == prefix:
        return "(root)"
    return module[len(prefix) + 1:].split(".")[0] if module.startswith(prefix + ".") else None


def fetch_run(run: str, pkg: Path) -> tuple[Path, Path]:
    """``run``'s `facts.json` and lock for ``pkg`` from the evidence store (its `run_files` tree), in a temporary
    directory."""
    shown = json.loads(subprocess.run(["research", "data", "show", run, "--json"], capture_output=True, text=True,
                                      check=True, cwd=ROOT).stdout)
    if not (tree_id := shown.get("outputs", {}).get("run_files")):
        raise SystemExit(f"survey size: {run} has no run_files in the store")
    want = [f"lean-audit/audit-{pkg.name}/{f}" for f in ("facts.json", C.POLICY)]
    out = Path(tempfile.mkdtemp(prefix="trusted-size-"))
    subprocess.run(["research", "data", "fetch", tree_id, "--to", str(out), *(x for w in want for x in ("--path", w))],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    if not all((out / w).is_file() for w in want):
        raise SystemExit(f"survey size: {run}'s outputs hold no {' or '.join(want)}: was it a records run of {pkg}?")
    return out / want[0], out / want[1]


def measure(pkg: Path, prefix: str, facts: dict | None = None, lock: dict | None = None) -> dict:
    """{area: its row} and {module: its unread definitions} for ``pkg``'s modules under ``prefix``: the definitions from
    ``facts`` (a Lean check's `facts.json` for ``pkg``), else from the sources; the read ones from ``lock`` (default
    ``pkg``'s `lean-audit.json`) and the locks of the packages requiring ``pkg`` by path."""
    pkg = pkg.resolve()
    base = ROOT if pkg.is_relative_to(ROOT) else pkg
    d = pkg.relative_to(base).as_posix()
    modules = {m.name: m for m in read(base)[1] if m.pkg.dir == ("" if d == "." else d) and under(prefix, m.name)}
    read_by: dict[str, set[str]] = defaultdict(set)
    reads: dict[str, set[str]] = defaultdict(set)
    for m, g in C.as_guarantees(lock if lock is not None else C.load(pkg)).get("reads", {}).items():
        reads[m] |= {C.parent(n) for n in C.definitions(g)}
        read_by[m] |= set(g.get("guarantees", []))
    elsewhere = C.read_by_dependents(pkg, ROOT)
    defs: dict[str, set[str]] | None = None
    if facts is not None:
        defs = defaultdict(set)
        for x in facts.get("spec_definitions", []):
            if x["module"] in modules and C.parent(x["name"]) == x["name"]:
                defs[x["module"]].add(x["name"])
    rows: dict[str, dict] = {}
    unread: dict[str, list[str]] = {}
    for name, m in sorted(modules.items()):
        r = rows.setdefault(under(prefix, name), {"files": 0, "lines": 0, "code": 0, "theorems": 0, "definitions": 0,
                                                  "read": 0, "unread": None, "guarantees": set()})
        r["files"] += 1
        r["lines"] += m.lines
        r["code"] += code_lines(m.code)
        r["theorems"] += sum(m.keywords[k] for k in THEOREM_KINDS)
        r["guarantees"] |= read_by.get(name, set())
        if defs is None:
            r["definitions"] += sum(m.keywords[k] for k in SIZE_DEFINITIONS)
            r["read"] += len(reads.get(name, ()))
            continue
        mine = defs.get(name, set())
        got = {x for x in mine if x in reads.get(name, ()) or x in elsewhere}
        r["definitions"] += len(mine)
        r["read"] += len(got)
        if left := sorted(mine - got):
            unread[name] = left
    for r in rows.values():
        r["unread"] = r["definitions"] - r["read"] if defs is not None else None
        r["guarantees"] = len(r["guarantees"])
    return {"exact": defs is not None, "areas": rows, "unread": unread}


def size_table(m: dict) -> str:
    cols = ["files", "lines", "code", "theorems", "definitions", "read", "unread", "guarantees"]
    rows = sorted(m["areas"].items(), key=lambda kv: -kv[1]["lines"])
    total = {c: sum((r[c] or 0) for _, r in rows) for c in cols}
    mark = "" if m["exact"] else "~"
    out = ["| area | " + " | ".join(cols) + " | read % |", "|" + "---|" * (len(cols) + 2)]
    for a, r in [*rows, ("total", total)]:
        cells = [str(r[c]) if c not in ("definitions", "unread") else f"{mark}{r[c] if r[c] is not None else '-'}"
                 for c in cols]
        share = f"{100 * r['read'] / r['definitions']:.0f}" if r["definitions"] and m["exact"] else "-"
        out.append(f"| {a} | " + " | ".join(cells) + f" | {share} |")
    return "\n".join(out)


def run_size(a) -> int:
    pkg = (ROOT / a.pkg) if not Path(a.pkg).is_absolute() else Path(a.pkg)
    facts, lock = fetch_run(a.run, pkg) if a.run else (a.facts, a.lock)
    m = measure(pkg, a.prefix, json.loads(facts.read_text()) if facts else None,
                json.loads(lock.read_text()) if lock else None)
    if a.json:
        print(json.dumps(m, indent=1, sort_keys=True))
        return 0
    print(size_table(m))
    if not m["exact"]:
        print("\n~ no facts given: definitions counted from the sources, read ones from the lock, unread not known")
    if a.unread:
        for mod, ds in m["unread"].items():
            print(f"\n{mod} ({len(ds)}): " + ", ".join(ds))
    return 0


# ---- unused

#: a declaration that extends the syntax or the attributes other modules elaborate with, `local` ones aside
SYNTAX = re.compile(r"(?m)^[ \t]*(?:@\[[^\]\n]*\][ \t]*)*(?:(?:private|protected|noncomputable|scoped|partial|unsafe)[ \t]+)*"
                    r"(?:\([^)\n]*\)[ \t]*)?(notation\d*|infixl|infixr|infix|prefix|postfix|syntax|macro_rules|macro|elab_rules"
                    r"|elab|declare_syntax_cat|register_simp_attr|register_label_attr|simproc|dsimproc|simproc_decl"
                    r"|dsimproc_decl|initialize|builtin_initialize)\b([^\n]*)")
#: a global attribute command (`attribute [simp] f`), `local` and `scoped` ones aside
ATTRIBUTE = re.compile(r"(?m)^[ \t]*attribute[ \t]*\[(?![ \t]*(?:local|scoped)\b)")
QUALIFIED = re.compile(r"[^\W\d][\w'!?₀-₉]*(?:\.[^\W\d][\w'!?₀-₉]*)+")
#: a qualified name written anywhere, not the tail of a longer one; and a name, qualified or not
WRITTEN = re.compile(r"(?<![\w.'])" + QUALIFIED.pattern)
NAME_OR_WORD = re.compile(r"(?<![\w.'])[^\W\d][\w'!?₀-₉]*(?:\.[^\W\d][\w'!?₀-₉]*)*")
CODE_SPAN = re.compile(r"`([^`\n]+)`")
#: a name's last component that reads as a lemma's or a definition's, not a word: an underscore, or a capital after a
#: letter or digit (`ksJoint_le`, `TagCutSplit`; not `pads`, `rep0`, `Capacity`)
DISTINCT = re.compile(r"_|[^\W_][A-Z]")
#: an `open` command's namespaces (and the words of its `hiding`, `renaming` or `in`, which name none)
OPEN = re.compile(r"(?m)^[ \t]*open[ \t]+(?:scoped[ \t]+)?([^\n(]*)")
STRING = re.compile(r'"((?:[^"\\\n]|\\.)*)"')
#: tracked files whose mentions of a module don't read it: the layout moves' maps, what `archive/` froze, and this tool's
#: tests, whose stand-ins write real modules' paths
NOT_READERS = ("tools/move/", "archive/", "tools/research/src/research/store/moves.json",
               "tools/verity/lean/tests/test_survey.py")
#: a Lean `import` written in any file (a test's Lean text, a script's), and one a diff adds
LEAN_IMPORT = re.compile(r"(?<![\w.])import[ \t]+(?:all[ \t]+)?([^\W\d][\w'.]*[\w'])")
ADDED_IMPORT = re.compile(r"^\+" + IMPORT.pattern)
#: a path joined from quoted parts (`ROOT / "Security" / "Proofs" / "A.lean"`), read as the one path it is
JOINED = re.compile(r"[\"'][ \t]*/[ \t]*[\"']")
#: a Lean file's path of two or more components, as written (`B/C.lean`, `../B/C.lean`)
LEAN_PATH = re.compile(r"(?<![\w./-])[\w.-]+(?:/[\w.-]+)+\.lean(?![\w/])")
#: a conformance check: a module named for one, or holding an `example` or a `#guard` (which nothing can use)
CHECK_NAME = re.compile(r"(?:Checks|Vectors)(?:\.|$)")
EXAMPLE = re.compile(r"(?m)^[ \t]*(?:example\b|#guard(?:_msgs)?\b)")
#: a module doc saying the module shows some hypotheses can be met (a non-vacuity or satisfiability witness), or, in its
#: title, that one is needed (a necessity result, a counterexample)
WITNESS_DOC = re.compile(r"(?i)not vacuous|non-?vacu|satisfiab|hypothes[ei]s? holds?\b|holds of one|have teeth|can be met"
                         r"|meets every hypothesis")
WITNESS_TITLE = re.compile(r"(?i)vacuous|vacuity|\bneeds\b|\bbeats\b|counterexample|necessary")
CODE = (".py", ".sh", ".toml", ".lean", ".json", ".yaml", ".yml", ".cfg", ".txt")
BRANCH_LANE = re.compile(r"cursor/([a-z][a-z0-9]*(?:-[a-z0-9]+)*?)-[a-z0-9]+-[0-9a-f]{4}\b|cursor/([a-z][a-z0-9]*)-")
#: how a queued PR reaches a module, weakest first
REACH = ("mentions", "edits", "reads")


def witness(text: str) -> bool:
    """Whether ``text``'s first module doc (`/-! … -/`) says what `WITNESS_DOC` or, in its title, `WITNESS_TITLE` does."""
    if (i := text.find("/-!")) < 0:
        return False
    doc = text[i + 3:text.find("-/", i)]
    title = next((line for line in doc.splitlines() if line.strip()), "")
    return bool(WITNESS_DOC.search(doc) or title.lstrip().startswith("#") and WITNESS_TITLE.search(title))


def tracked(root: Path) -> list[str]:
    return subprocess.run(["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True).stdout.decode().split("\0")[:-1]


def lock_packages(pkgs: dict[str, Package], pkg: str) -> list[str]:
    """The packages whose locks' names PKG's build can hold: PKG, those it requires by path (transitively), and those whose
    lock names it under `proved_in`."""
    out, stack = [], [pkg]
    while stack:
        if (d := stack.pop()) in pkgs and d not in out:
            out.append(d)
            stack += [r for _, r in pkgs[d].requires if not r.startswith("git:")]
    out += [d for d, q in pkgs.items() if pkg in {norm(d, x) for x in (q.lock or {}).get("proved_in", []) if isinstance(x, str)}]
    return [d for d in dict.fromkeys(out) if pkgs[d].lock is not None]


def names_of(policy: dict) -> tuple[list[str], list[str]]:
    """(the lock's guarantees, every other qualified name its keys and strings hold)."""
    guarantees = sorted(policy.get("guarantees", {}))
    return guarantees, sorted(set(QUALIFIED.findall(json.dumps(policy, ensure_ascii=False))) - set(guarantees))


def usage_of(pkg: Path, roots: list[str], pins: list[str], extra: list[str], out: Path) -> dict:
    """`Facts.lean`'s usage document for ``pkg``'s build, written to ``out``."""
    config = out.with_name("usage-config.json")
    config.write_text(json.dumps({"roots": roots, "pins": pins, "usage": extra, "out": str(out)}))
    rc, text = C.run(["lake", "env", C.facts_program(pkg), str(config)], pkg, out.with_name("usage.log"))
    if rc or not out.is_file():
        raise SystemExit(f"survey unused: `Facts.lean` failed (exit {rc}; a root that was not built?):\n{text[-3000:]}")
    return json.loads(out.read_text())


def declared(m: Module) -> set[str]:
    """The full names of a module's declarations another file can name: a `private` one and an anonymous instance aside."""
    return {d.full for d in m.decls if d.name and not d.private}


def opened(m: Module) -> set[str]:
    """The namespaces a Lean file's names may be written relative to: each one its declarations stand in (and those
    enclosing it) and each it `open`s, as written and under those."""
    spaces = set()
    for d in m.decls:
        if not d.name.startswith("_root_.") and (ns := d.full[:len(d.full) - len(d.name)].strip(".")):
            parts = ns.split(".")
            spaces |= {".".join(parts[:k]) for k in range(1, len(parts) + 1)}
    opens = {w for o in OPEN.finditer(m.code) for w in o.group(1).split() if NAME_OR_WORD.fullmatch(w)}
    return spaces | opens | {f"{s}.{o}" for s in spaces for o in opens}


def closure(start: set[str], imports: dict[str, list[str]], within: set[str]) -> set[str]:
    """``start`` and every module of ``within`` it imports, transitively."""
    seen, stack = set(), list(start)
    while stack:
        if (m := stack.pop()) not in seen:
            seen.add(m)
            stack += [i for i in imports.get(m, []) if i in within]
    return seen


def atoms(decl: str) -> set[str]:
    """What a module using a syntax declaration's rest (``decl``, after its keyword) writes: its string atoms, and for an
    attribute or simproc its name."""
    got = {a.strip() for a in STRING.findall(decl) if a.strip()}
    if not got and (m := re.match(r"[ \t]*([^\W\d][\w'.]*)", decl)):
        got.add(m.group(1).rsplit(".", 1)[-1])
    return got


def references(root: Path, files: list[str], pkg: Package, mods: dict[str, str],
               consts: dict[str, set[str]] | None = None) -> dict:
    """{module of PKG (``mods``: each one's file): [(file, whether it reads what the module imports)]}, for each file of
    ``files`` outside PKG's sources (and `NOT_READERS`) naming the module's file (from ``root``, or from PKG's source
    directory, `Proofs/A/B.lean`; a path joined from quoted parts, `"Proofs" / "A"`, is read as one; a path without `.lean`
    names it only where no directory has that path), its module name, a constant one component under it, or one of its
    constants by full name (``consts``: each name's modules), or a name under one; a Lean file names only what it
    imports. A Lean file's name in backticks may be relative to its namespaces and opens (`opened`), and one that names no
    constant but ends in a `DISTINCT` last component of ``consts``' that no Lean file outside PKG declares cites each
    module declaring one so. The mentions that read nothing are under the key `(docs, module)` for a document's, which a
    deletion edits, and `(cited, module)` for a Lean comment's, whose flag says whether it names the module itself. A file
    outside PKG, a document aside, that writes the end of a module's path, two components or more (`B/C.lean` for
    `Proofs/A/B/C.lean`, its longest such end), is under `(suffix, module)` with that end, unless the path is one of those
    above or a tracked file of its own from the file's directory or one above it."""
    consts = consts or {}
    ends: dict[str, set[str]] = defaultdict(set)
    for m, f in mods.items():
        parts = f.split("/")
        for k in range(2, len(parts) + 1):
            ends["/".join(parts[-k:])].add(m)
    tracked_ = set(files)
    lean = {f: Module("", f, None, root / f) for f in files
            if f.endswith(".lean") and not f.startswith((pkg.dir + "/", *NOT_READERS)) and (root / f).is_file()}
    short: dict[str, set[str]] = defaultdict(set)
    for n, ms in consts.items():
        if DISTINCT.search(s := n.rsplit(".", 1)[-1]):
            short[s] |= ms
    for m in lean.values() if short else ():
        for n in declared(m):
            short.pop(n.rsplit(".", 1)[-1], None)
    by_path = {f: m for m, f in mods.items()}
    by_src = {m.replace(".", "/") + ".lean": m for m in mods}
    out: dict = defaultdict(list)
    tops = sorted({m.split(".", 1)[0] for m in mods})
    either = "|".join(map(re.escape, tops))
    named = re.compile(r"(?<![\w.])(?:" + either + r")(?:\.[^\W\d][\w']*)+")
    paths = re.compile(re.escape(pkg.dir) + r"/[\w./-]*\w")
    local = re.compile(r"(?<![\w./-])(?:" + either + r")/[\w./-]*\w")

    def mentioned(text: str, within: set[str] | None = None) -> dict[str, bool]:
        """{module: whether the text names the module itself (its path or name), not only a constant under it}; a Lean
        file's (``within``: the namespaces it is in and opens) name written as code may be relative to one of them"""
        hit: dict[str, bool] = {}
        for found, table_, base in ((paths, by_path, root), (local, by_src, root / pkg.src)):
            for p in found.finditer(text):
                x = p.group(0).rstrip(".")
                if x in table_:
                    hit[table_[x]] = True
                elif x + ".lean" in table_ and text[p.end():p.end() + 1] != "/" and not (base / x).is_dir():
                    hit[table_[x + ".lean"]] = True
        for x in named.findall(text):
            if x in mods:
                hit[x] = True
            elif x.rsplit(".", 1)[1][:1].islower() and (x := x.rsplit(".", 1)[0]) in mods:
                hit.setdefault(x, False)
        written = [(x, ()) for x in WRITTEN.findall(text)] if consts else []
        written += [(x, within) for code in CODE_SPAN.findall(text) for x in NAME_OR_WORD.findall(code)] if within is not None else []
        for x, under_ in written:
            for full in (x, *(f"{w}.{x}" for w in under_)):
                parts = full.split(".")
                if n := next((n for k in range(len(parts), 1, -1) if (n := ".".join(parts[:k])) in consts), None):
                    for m in consts[n]:
                        hit.setdefault(m, False)
        for code in CODE_SPAN.findall(text) if within is not None else []:
            for w in NAME_OR_WORD.finditer(code):
                x = w.group(0)
                if (s := x.rsplit(".", 1)[-1]) not in short or "/" in code[max(w.start() - 1, 0):w.end() + 1]:
                    continue
                if not any(".".join(f.split(".")[:k]) in consts for f in (x, *(f"{u}.{x}" for u in within))
                           for k in range(2, f.count(".") + 2)):
                    for m in short[s]:
                        hit.setdefault(m, False)
        return hit

    for f in files:
        if (f.startswith(pkg.dir + "/") and f.endswith(".lean")) or f.startswith(NOT_READERS) or not f.endswith(CODE + (".md",)):
            continue
        if f.endswith((C.POLICY, "lake-manifest.json")):
            continue
        try:
            text = JOINED.sub("/", (root / f).read_text(errors="replace"))
        except OSError:
            continue
        if not f.startswith(pkg.dir + "/") and not f.endswith(".md"):
            bases = [str(p) for p in Path(f).parents]
            for x in dict.fromkeys(LEAN_PATH.findall(text)):
                if x in by_path or x in by_src or any(
                        (q := os.path.normpath(os.path.join(b, x))) in tracked_ and q not in by_path for b in bases):
                    continue
                parts = x.split("/")
                if end := next((e for k in range(len(parts), 1, -1) if (e := "/".join(parts[-k:])) in ends), None):
                    for m in ends[end]:
                        out[("suffix", m)].append((f, end))
        if not consts and pkg.dir not in text and not any(t + "." in text or t + "/" in text for t in tops):
            continue
        if f in lean:
            imported = {i for i in lean[f].imports if i in mods}
            for m in imported:
                out[m].append((f, True))
            for m, whole in mentioned(text, opened(lean[f]) if consts else None).items():
                if m not in imported:
                    out[("cited", m)].append((f, whole))
            continue
        writes = {i for i in LEAN_IMPORT.findall(text) if i in mods}
        reads = "difftest" in f or f.endswith(".sh")
        for m in set(mentioned(text)) | writes:
            out[("docs", m) if f.endswith(".md") else m].append((f, reads or m in writes))
    return out


def pr_entry(diff: list[tuple[str, str, str]]) -> dict:
    """A PR's entry of `fetch_queued`'s document from its diff, (file, previous path or "", patch) per file."""
    files, imports, names, reads = set(), set(), set(), set()
    for fn, prev, patch in diff:
        files |= {fn, prev} - {""}
        if fn.endswith(".lean"):
            imports |= {m.group(1) for line in patch.splitlines() if (m := ADDED_IMPORT.match(line))}
        else:
            lines = [line for line in patch.splitlines() if line.startswith("+")]
            added = {q for line in lines for q in QUALIFIED.findall(line)}
            names |= added
            reads |= added if "difftest" in fn or fn.endswith(".sh") else set()
            reads |= {i for line in lines for i in LEAN_IMPORT.findall(line)}
    return {"files": sorted(files), "imports": sorted(imports), "names": sorted(names), "reads": sorted(reads)}


def fetch_queued(path: Path) -> dict:
    """Every PR `research queue status` lists, in any state, with the files its diff touches (old and new paths), the modules
    its added `import` lines name, the qualified names its other files' added lines hold (a lock's entry, a test's
    module), and those of them a difftest or a shell script holds or an added `import` line names, written to ``path``:
    {PR: {"files": [...], "imports": [...], "names": [...], "reads": [...]}}."""
    from concurrent.futures import ThreadPoolExecutor

    status = subprocess.run(["research", "queue", "status"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    prs = sorted({int(n) for n in re.findall(r"(?m)^\s*\S+\s+#(\d+)\s", status)})

    def one(n: int) -> tuple[str, dict]:
        r = subprocess.run(["gh", "api", "--paginate", f"repos/{{owner}}/{{repo}}/pulls/{n}/files?per_page=100", "--jq",
                            '.[] | [.filename, (.previous_filename // ""), (.patch // "")] | @json'],
                           cwd=ROOT, capture_output=True, text=True)
        if r.returncode:
            raise SystemExit(f"survey unused: the files of PR #{n} can't be read: {r.stderr.strip()[-300:]}")
        return str(n), pr_entry([tuple(json.loads(line)) for line in r.stdout.splitlines() if line.strip()])

    with ThreadPoolExecutor(8) as ex:
        doc = dict(ex.map(one, prs))
    path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return doc


def queued_modules(queued: dict, mods: dict[str, str]) -> dict[str, tuple[list[str], str]]:
    """{module (``mods``: each one's file): (the PRs of ``queued`` (`fetch_queued`'s document) whose diff touches its file,
    or adds an import or another mention of it; the strongest of `REACH` among them: `reads` (an added import, or a name
    ``reads`` holds, every name of a document without ``reads``), `edits` (its file) or `mentions` (any other name))}."""
    by_path = {f: m for m, f in mods.items()}
    out: dict[str, set[str]] = defaultdict(set)
    how: dict[str, str] = {}

    def mark(m: str, n: str, h: str) -> None:
        out[m].add(f"#{n}")
        how[m] = max(how.get(m, h), h, key=REACH.index)

    for n, pr in queued.items():
        reads = {*pr.get("imports", []), *pr.get("reads", pr.get("names", []))}
        for m in [by_path[f] for f in pr.get("files", []) if f in by_path]:
            mark(m, n, "edits")
        for i in [*pr.get("imports", []), *pr.get("names", [])]:
            if i in mods:
                mark(i, n, "reads" if i in reads else "mentions")
    return {m: (sorted(v, key=lambda s: int(s[1:])), how[m]) for m, v in out.items()}


def lane_from_history(root: Path, path: str) -> str | None:
    """The lane most of the branches merged into ``path``'s history name (`cursor/<lane>-…`)."""
    log = subprocess.run(["git", "log", "--format=%s", "-n", "400", "--", path], cwd=root, capture_output=True, text=True).stdout
    c = Counter(next(g for g in m.groups() if g) for m in BRANCH_LANE.finditer(log))
    return f"@{c.most_common(1)[0][0]}" if c else None


def import_graph(mods: dict[str, Module], usage: dict) -> dict[str, list[str]]:
    """What each module imports: PKG's from their sources, every other one from the usage document's `graph`."""
    return {**{m: g["imports"] for m, g in usage.get("graph", {}).items()}, **{m: x.imports for m, x in mods.items()}}


def decide(root: Path, pkg: Package, mods: dict[str, Module], policy: dict, usage: dict, files: list[str],
           frozen: list[str], queued: dict | None = None, doubts: bool = True) -> dict:
    """The kept modules of PKG (``mods``) with their reasons, and the unused ones. A module is kept, with the first reason
    that holds (``doubts`` False drops the `checks`, `cited` and `suffix` rules, as `held` is found):
    - `used`: it holds a constant of the closure (`by` names the first guarantee whose closure reaches it);
    - `root`, `runs`, `exempt`, `compile_time`: the lock lists it so (an exempt or `runs` module's imports are kept too);
    - `syntax`: it declares a notation, macro, syntax, elaborator, simproc or attribute (`local` ones aside) whose atom or
      name a module importing it uses, or one whose atoms can't be read off its source;
    - `named`: a tracked file other than PKG's Lean sources names it (`references`; the move maps and `archive/` aside, a
      document's mention is listed as `docs`, a Lean comment's is `cited`). What it needs to build is what its
      declarations use (`refs`); a reader that writes Lean against it (a difftest, a shell script, any file holding
      `import M`) keeps what it imports too, transitively;
    - `queued`: a PR in the merge queue (``queued``, `fetch_queued`'s document) edits, renames or deletes its file, or adds
      an `import` of it or (in another file) its name; what it imports is kept too where `named` would keep it or the
      PRs edit it, unless they only edit it and it declares nothing;
    - `checks`: a conformance check, which no guarantee reaches by design: it holds an `example` or a `#guard`, or its
      name (or an enclosing module's) ends in `Checks` or `Vectors`; or a non-vacuity witness or a necessity result
      (`witness`). Its owner decides whether it goes;
    - `cited`: a comment of a Lean file outside PKG (a Definition's docstring) cites it, by its module's name or path, a
      constant's name, or a lemma's name written bare (`references`): what that text argues rests on it. A module that
      declares nothing, cited by its own name or path, is cited for what it gathers, so what it imports is kept too (a
      lock's root aside);
    - `suffix`: a file outside PKG, a document aside, writes the end of its path, two components or more
      (`B/C.lean`; `references`): a test that opens the file names it so. A bare file name is too common to count;
    - `refs`: a kept module's declarations use its constants (a module is deleted whole, so every declaration of a kept
      module must still build);
    - `attribute`: it sets an attribute globally (`attribute [simp] f`), and a kept module imports it;
    - `aggregator`: it declares nothing and imports a kept module;
    - `unbuilt`: the usage document holds no `.olean` of it.
    Every other module is unused."""
    scope = set(mods)
    imports = import_graph(mods, usage)
    facts = usage.get("usage", {}).get("modules", {})
    importers: dict[str, set[str]] = defaultdict(set)
    for m in scope:
        for i in imports.get(m, []):
            importers[i].add(m)
    above: dict[str, set[str]] = {}

    def importing(m: str) -> set[str]:
        if m not in above:
            seen, stack = set(), list(importers.get(m, ()))
            while stack:
                if (x := stack.pop()) not in seen:
                    seen.add(x)
                    stack += list(importers.get(x, ()))
            above[m] = seen
        return above[m]

    kept: dict[str, str] = {}

    def keep(m: str, why: str) -> bool:
        if m in scope and m not in kept:
            kept[m] = why
            return True
        return False

    def keep_imports(m: str, why: str) -> None:
        for i in sorted(closure({m}, imports, scope) - {m}):
            keep(i, why)

    for m in sorted(scope):
        if (f := facts.get(m)) and f.get("used"):
            keep(m, f"used: {f['used']} of its {f['constants']} constants are in the closure (first reached by {f['by']})")
    for m in policy.get("roots", []):
        keep(m, "root: a root of the lock")
    for m in policy.get("compile_time", {}):
        keep(m, "compile_time: listed under compile_time")
    for key, what in (("runs", "a `runs` test"), ("exempt", "exempt")):
        for m in policy.get(key, {}):
            reason = policy[key][m] if key == "exempt" else policy[key][m].get("why", "")
            keep(m, f"{key}: {what} ({str(reason)[:120]})")
            keep_imports(m, f"{key}: imported by {m}, {what}")
    for m in sorted(scope):
        if not (decls := SYNTAX.findall(mods[m].code)):
            continue
        kinds = ", ".join(sorted({k for k, _ in decls}))
        if not (words := set().union(*(atoms(rest) for _, rest in decls))):
            keep(m, f"syntax: declares {kinds} whose atoms its source doesn't show")
        elif users := sorted(u for u in importing(m) if any(w in mods[u].code for w in words)):
            keep(m, f"syntax: declares {kinds} ({', '.join(sorted(words)[:4])}), used by {users[0]}"
                    + (f" and {len(users) - 1} more" if len(users) > 1 else ""))
    consts: dict[str, set[str]] = defaultdict(set)
    for m in scope:
        for n in declared(mods[m]):
            consts[n].add(m)
    files_of = {m: x.file for m, x in mods.items()}
    refs = references(root, files, pkg, files_of, consts)
    for m in sorted(k for k in refs if isinstance(k, str)):
        by = sorted(refs[m])
        keep(m, f"named: {by[0][0]}" + (f" and {len(by) - 1} more" if len(by) > 1 else ""))
        for f, reads in by:
            if reads:
                keep_imports(m, f"named: imported by {m}, which {f} reads")
    for m, (by, how) in sorted(queued_modules(queued or {}, files_of).items()):
        keep(m, f"queued: {', '.join(by[:4])}" + (f" and {len(by) - 4} more" if len(by) > 4 else "") + " touch or import it")
        if not (how == "mentions" or how == "edits" and (facts.get(m) or {}).get("constants", 1) == 0):
            keep_imports(m, f"queued: imported by {m}, which {by[0]} touches or imports")
    if doubts:
        for m in sorted(scope):
            if CHECK_NAME.search(m) or EXAMPLE.search(mods[m].code):
                keep(m, "checks: a conformance check (it holds an `example` or a `#guard`, or its name says Checks or Vectors)")
            elif witness(mods[m].raw):
                keep(m, "checks: a non-vacuity witness or a necessity result (its module doc says it shows some hypotheses "
                        "can be met, or one is needed)")
        for m in sorted(k[1] for k in refs if isinstance(k, tuple) and k[0] == "cited"):
            by = sorted({f for f, _ in refs[("cited", m)]})
            keep(m, f"cited: a Lean comment of {by[0]}" + (f" and {len(by) - 1} more" if len(by) > 1 else "") + " cites it")
            if (any(w for _, w in refs[("cited", m)]) and m not in policy.get("roots", [])
                    and (facts.get(m) or {}).get("constants", 1) == 0):
                keep_imports(m, f"cited: imported by {m}, which {by[0]} cites as what it gathers")
        for m in sorted(k[1] for k in refs if isinstance(k, tuple) and k[0] == "suffix"):
            (f, end), n = min(refs[("suffix", m)]), len({f for f, _ in refs[("suffix", m)]})
            keep(m, f"suffix: {f}" + (f" and {n - 1} more write" if n > 1 else " writes") + f" its path's end, `{end}`")
    changed = True
    while changed:
        changed = False
        for m in sorted(kept):
            for r in (facts.get(m) or {}).get("refs", []):
                changed |= keep(r, f"refs: the declarations of {m} (kept) use its constants")
        for m in sorted(scope - set(kept)):
            if ATTRIBUTE.search(mods[m].code) and (users := sorted(importing(m) & set(kept))):
                changed |= keep(m, f"attribute: sets an attribute globally, and {users[0]} (kept) imports it")
        for m in sorted(scope - set(kept)):
            f = facts.get(m)
            if ((f is None and (mods[m].umbrella or not mods[m].code.strip())) or (f is not None and f["constants"] == 0)) \
                    and any(i in kept for i in imports.get(m, [])):
                changed |= keep(m, "aggregator: declares nothing, and imports a kept module")
    for m in sorted(scope - set(kept) - set(facts)):
        keep(m, "unbuilt: no .olean was read for it, so what it holds is not known")
    unused = sorted(scope - set(kept))
    docs = {m: sorted({f for f, _ in refs[("docs", m)]}) for m in scope if ("docs", m) in refs}
    return {"kept": kept, "unused": unused, "docs": docs, "frozen_under": list(frozen),
            "frozen": [m for m in unused if any(m == f or m.startswith(f + ".") for f in frozen)]}


def owners(policies: dict[str, dict], pins: dict[str, str | None]) -> dict[str, str]:
    """{module prefix: the owner most of the guarantees declared under it have}, for every prefix of a declaring module."""
    votes: dict[str, Counter] = defaultdict(Counter)
    for policy in policies.values():
        for g, e in policy.get("guarantees", {}).items():
            if (m := pins.get(g)) and (who := (e or {}).get("owner")):
                parts = m.split(".")
                for i in range(1, len(parts) + 1):
                    votes[".".join(parts[:i])][who] += 1
    return {p: c.most_common(1)[0][0] for p, c in votes.items()}


def listing(root: Path, mods: dict[str, Module], decided: dict, policies: dict[str, dict], usage: dict,
            held: dict[str, str] | None = None) -> dict:
    """The list: ``decided``'s unused modules, each with its bytes, area (`Module.area`), owner (the owner most of the
    guarantees declared under its nearest enclosing namespace of modules have, in the locks; else the lane most of the
    branches merged into its directory's history name) and the documents naming it (`docs`), totals per area and in all;
    and ``held`` ({module: why}), the modules only a queued PR or the `checks`, `cited` or `suffix` rule keeps, with the
    same columns."""
    lead = owners(policies, usage.get("usage", {}).get("pins", {}))
    history: dict[str, str | None] = {}

    def row(m: str) -> dict:
        parts = m.split(".")
        who = next((lead[".".join(parts[:i])] for i in range(len(parts) - 1, 1, -1) if ".".join(parts[:i]) in lead), None)
        how = "lock"
        if who is None:
            if (d := os.path.dirname(mods[m].file)) not in history:
                history[d] = lane_from_history(root, d)
            who, how = history[d], "git log"
        return {"module": m, "path": mods[m].file, "bytes": mods[m].path.stat().st_size, "area": mods[m].area,
                "owner": who, "owner_from": how, "docs": decided["docs"].get(m, []),
                "frozen": any(m == f or m.startswith(f + ".") for f in decided.get("frozen_under", []))}

    rows = [row(m) for m in decided["unused"]]
    kept_rows = [{**row(m), "why": why} for m, why in sorted((held or {}).items())]
    areas: dict[str, dict] = {}
    for r in rows:
        a = areas.setdefault(r["area"], {"modules": 0, "bytes": 0, "frozen": 0, "frozen_bytes": 0, "owners": Counter()})
        a["frozen" if r["frozen"] else "modules"] += 1
        a["frozen_bytes" if r["frozen"] else "bytes"] += r["bytes"]
        a["owners"][r["owner"] or "-"] += 1
    for a in areas.values():
        a["owners"] = dict(a["owners"].most_common())
    total = {"modules": len(mods), "kept": len(decided["kept"]), "unused": len(rows),
             "unused_bytes": sum(r["bytes"] for r in rows), "frozen": sum(r["frozen"] for r in rows),
             "held": len(kept_rows), "held_bytes": sum(r["bytes"] for r in kept_rows),
             "package_bytes": sum(x.path.stat().st_size for x in mods.values()),
             "kept_by": dict(Counter(why.split(":", 1)[0] for why in decided["kept"].values()))}
    return {"total": total, "areas": areas, "unused": rows, "held": kept_rows, "kept": decided["kept"]}


def undeclared(modules: list[Module], policies: dict[str, dict], pins: dict[str, str | None]) -> list[str]:
    """The guarantees no module of PKG's build holds (``pins``' null ones) that no module of a package whose lock lists
    them declares either, read as text (`Module.decls`): a lock entry nothing declares."""
    where = {g: {d for d, p in policies.items() if g in p.get("guarantees", {})} for g, m in pins.items() if m is None}
    found: dict[str, set[str]] = defaultdict(set)
    for m in modules:
        if any(m.pkg.dir in ds for ds in where.values()):
            for d in m.decls:
                if d.full in where:
                    found[d.full].add(m.pkg.dir)
    return sorted(g for g, ds in where.items() if not found[g] & ds)


def report(doc: dict) -> str:
    t = doc["total"]
    out = [f"{t['unused']} of {t['modules']} modules of {doc.get('package', 'PKG')} unused, {fmt(t['unused_bytes'])} of "
           f"{fmt(t['package_bytes'])} bytes ({t['frozen']} of them frozen); {t['held']} more, {fmt(t['held_bytes'])} bytes, "
           f"held by queued PRs, as checks, cited or by a path's end; kept: " + ", ".join(f"{k} {v}" for k, v in sorted(t["kept_by"].items())),
           f"locks read: {', '.join(doc.get('locks', []))}", "",
           table([[a, r["modules"], fmt(r["bytes"]), r["frozen"], fmt(r["frozen_bytes"]),
                   ", ".join(f"{o} {n}" for o, n in r["owners"].items())]
                  for a, r in sorted(doc["areas"].items(), key=lambda kv: -kv[1]["bytes"])],
                 ["area", "modules", "bytes", "frozen", "frozen bytes", "owners"])]
    for title, rows, last in (("unused", doc["unused"], "docs"), ("held", doc["held"], "why")):
        if rows:
            out += ["", f"{title} ({len(rows)} modules, {fmt(sum(r['bytes'] for r in rows))} bytes)",
                    table([[r["module"], fmt(r["bytes"]), r["area"], f"{r['owner'] or '-'} ({r['owner_from']})",
                            "frozen" if r["frozen"] else "", ", ".join(r[last]) if last == "docs" else r[last]]
                           for r in sorted(rows, key=lambda r: (r["area"], r["module"]))],
                          ["module", "bytes", "area", "owner", "frozen", last])]
    out += [f"a guarantee no package whose lock lists it declares: {g}" for g in doc.get("undeclared", [])]
    return "\n".join(out) + "\n"


def rewrite_imports(text: str, deleted: set[str], replace: dict[str, list[str]], reach) -> str:
    """``text`` with each import of a deleted module replaced by ``replace[module]`` less what its other imports reach
    (``reach``: a module's import closure) and what it imports already, each written as the line it replaces was."""
    lines, code = text.split("\n"), C.strip_comments(text).split("\n")
    kept = [m.group(1) for s in code if (m := IMPORT.match(s)) and m.group(1) not in deleted]
    have = set(kept).union(*(reach(i) for i in kept))
    out = []
    for line, s in zip(lines, code):
        if not (m := IMPORT.match(s)) or m.group(1) not in deleted:
            out.append(line)
            continue
        for r in replace[m.group(1)]:
            if r not in have:
                have |= {r} | reach(r)
                out.append(s[:m.start(1)] + r)
    return "\n".join(out)


def delete(root: Path, mods: dict[str, Module], area: str, doc: dict, usage: dict, spare: set[str]) -> list[str]:
    """Delete ``area``'s unused modules of ``doc`` (less ``spare`` and the frozen ones) and, in every module left that
    imported one, replace that import with the kept modules it reached through the deleted ones that the module's other
    imports don't already reach, so the rest builds as before; the modules deleted."""
    gone = {r["module"] for r in doc["unused"] if r["area"] == area and not r["frozen"] and r["module"] not in spare}
    gone &= set(mods)
    imports = import_graph(mods, usage)
    memo: dict[str, list[str]] = {}

    def expand(m: str) -> list[str]:
        if m not in memo:
            memo[m] = []
            memo[m] = list(dict.fromkeys(x for i in imports.get(m, []) for x in (expand(i) if i in gone else [i])))
        return memo[m]

    reached: dict[str, set[str]] = {}

    def reach(m: str) -> set[str]:
        if m not in reached:
            reached[m] = closure({m}, imports, set(imports)) - {m}
        return reached[m]

    replace = {m: expand(m) for m in gone}
    for m, x in mods.items():
        if m not in gone and any(i in gone for i in x.imports):
            x.path.write_text(rewrite_imports(x.raw, gone, replace, reach))
    for m in sorted(gone):
        mods[m].path.unlink()
    for m in sorted(gone):
        d = mods[m].path.parent
        while d != root / mods[m].pkg.dir and d.is_dir() and not any(d.iterdir()):
            d.rmdir()
            d = d.parent
    return sorted(gone)


def stale_mentions(root: Path, gone: list[str], files: list[str], mods: dict[str, str]) -> list[str]:
    """The tracked files (documents and locks) still naming a deleted module's path (``mods``: where each was) or name."""
    paths, names, out = {mods[m] for m in gone}, set(gone), []
    for f in files:
        if not f.endswith((".md", C.POLICY, "lakefile.toml")) or not (root / f).is_file():
            continue
        text = (root / f).read_text(errors="replace")
        if hits := sorted({x for x in paths if x in text} | {n for n in names if f'"{n}"' in text}):
            out.append(f"{f}: {', '.join(hits[:6])}" + (f" and {len(hits) - 6} more" if len(hits) > 6 else ""))
    return out


def run_unused(a) -> int:
    if a.fetch_queued:
        doc = fetch_queued(a.fetch_queued)
        print(f"survey unused: {len(doc)} PRs of the queue, {sum(len(v['files']) for v in doc.values())} files -> {a.fetch_queued}")
        return 0
    pkgs, ms, _ = read(ROOT)
    p = os.path.relpath((ROOT / a.pkg).resolve(), ROOT)
    if p not in pkgs or pkgs[p].lock is None:
        raise SystemExit(f"survey unused: {p} is no Lake package with a {C.POLICY}")
    mods = {m.name: m for m in ms if m.pkg.dir == p}
    queued = json.loads(a.queued.read_text()) if a.queued else {}
    if a.delete:
        if not a.list or not a.facts:
            raise SystemExit("survey unused: --delete needs --list (an unused.json) and --facts (the facts.json beside it)")
        doc, usage = json.loads(a.list.read_text()), json.loads(a.facts.read_text())
        files_of = {m: x.file for m, x in mods.items()}
        gone = delete(ROOT, mods, a.delete, doc, usage, set(a.spare) | set(queued_modules(queued, files_of)))
        print(f"survey unused: deleted {len(gone)} modules of {a.delete}, "
              f"{fmt(sum(r['bytes'] for r in doc['unused'] if r['module'] in set(gone)))} bytes")
        for line in stale_mentions(ROOT, gone, tracked(ROOT), files_of):
            print(f"  still named by {line}")
        return 0
    policies = {q: C.as_guarantees(pkgs[q].lock) for q in lock_packages(pkgs, p)}
    pins, extra = set(), set()
    for policy in policies.values():
        g, x = names_of(policy)
        pins, extra = pins | set(g), extra | set(x)
    if a.facts:
        usage = json.loads(a.facts.read_text())
    else:
        out = a.out or Path(tempfile.mkdtemp(prefix="survey-unused-"))
        out.mkdir(parents=True, exist_ok=True)
        usage = usage_of(ROOT / p, list(policies[p]["roots"]), sorted(pins), sorted(extra - pins), out / "facts.json")
    files = tracked(ROOT)
    decided = decide(ROOT, pkgs[p], mods, policies[p], usage, files, a.frozen, queued)
    bare = decide(ROOT, pkgs[p], mods, policies[p], usage, files, a.frozen, doubts=False)["unused"]
    doc = listing(ROOT, mods, decided, policies, usage, {m: decided["kept"][m] for m in bare if m in decided["kept"]})
    doc |= {"package": p, "locks": sorted(policies), "queued_prs": len(queued),
            "undeclared": undeclared(ms, policies, usage.get("usage", {}).get("pins", {}))}
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        if a.facts and a.facts.resolve() != (a.out / "facts.json").resolve():
            (a.out / "facts.json").write_text(a.facts.read_text())
        (a.out / "unused.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(report(doc), end="")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    m = sub.add_parser("map", help="packages, areas by tree, the area graph, module sizes")
    m.add_argument("--root", type=Path, default=ROOT, help="the checkout to map (default: this one)")
    m.add_argument("--only", default="abcde", help="the sections to print, by letter")
    m.add_argument("--by", choices=["component", "area"], default="component", help="what (c) condenses the graph to")
    m.add_argument("--moves", type=Path, help="map the tree as this move map's dotted [words] would leave it")
    m.add_argument("--top", type=int, default=15, help="rows in each top list")
    m.add_argument("--json", type=Path, help="also write every module's facts here")
    c = sub.add_parser("conventions", help="the Lean's form counted: naming, structure, organization, statement form")
    c.add_argument("packages", nargs="*", type=Path, help="package directories (default: every one with a lean-audit.json)")
    c.add_argument("--root", type=Path, default=ROOT, help="the checkout to survey (default: this one), e.g. a worktree "
                   "of an older commit, to compare")
    c.add_argument("--json", type=Path, help="also write the full survey here")
    c.add_argument("--top", type=int, default=10, help="worst offenders listed per measure")
    s = sub.add_parser("size", help="the spec's size per area, and what no listed statement reads")
    s.add_argument("pkg", nargs="?", default="Security", help="the Lake package (default Security)")
    s.add_argument("--prefix", default="Definitions", help="the spec modules' prefix (default Definitions)")
    src = s.add_mutually_exclusive_group()
    src.add_argument("--facts", type=Path, help="a Lean check's facts.json for PKG")
    src.add_argument("--run", help="a records run of PKG whose facts.json and lock the evidence store holds")
    s.add_argument("--lock", type=Path, help="the lock whose reads count (default PKG's lean-audit.json)")
    s.add_argument("--unread", action="store_true", help="list the unread definitions per module")
    s.add_argument("--json", action="store_true", help="print the measurement as JSON")
    u = sub.add_parser("unused", help="the modules no listed guarantee uses, and their deletion per area")
    u.add_argument("pkg", nargs="?", default="Security/Proofs", help="the built Lake package (default Security/Proofs)")
    u.add_argument("--facts", type=Path, help="the usage document Facts.lean wrote for PKG (--out's facts.json)")
    u.add_argument("--out", type=Path, help="write facts.json and unused.json here")
    u.add_argument("--frozen", nargs="*", default=[], metavar="PREFIX", help="list the unused modules under PREFIX apart")
    u.add_argument("--delete", metavar="AREA", help="delete AREA's unused modules of --list")
    u.add_argument("--list", type=Path, help="with --delete: an unused.json this wrote")
    u.add_argument("--except", dest="spare", nargs="*", default=[], metavar="MODULE", help="with --delete: keep these")
    u.add_argument("--queued", type=Path, help="keep what the queued PRs of this document (--fetch-queued's) touch or import")
    u.add_argument("--fetch-queued", type=Path, metavar="QUEUED.json", help="write the merge queue's PRs' files and imports")
    a = ap.parse_args(argv)
    return {"map": run_map, "conventions": run_conventions, "size": run_size, "unused": run_unused}[a.command](a)


if __name__ == "__main__":
    sys.exit(main())
