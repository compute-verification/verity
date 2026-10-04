#!/usr/bin/env python3
"""Split PoUS's Lean package into its spec and its proofs (the repository plan's step 3.1, "Split and move").

    python protocols/pous/lean_split.py            the plan: what moves, what is edited, how the lock splits (changes nothing)
    python protocols/pous/lean_split.py --apply    perform it in this checkout (`git mv`, then the edits), and verify it

The spec stays in `protocols/pous/lean/` with its lock; the proofs become the Lake package `security_proofs/pous/`, which
`require`s the spec by path. No module is renamed, so every pin and `reads` entry of the lock stays byte-identical: only the
lock's package-level sections are split (`roots`, `exempt`, `layers`, `dependencies`), and the umbrella `Pous` moves to the
proofs. The run refuses, changing nothing, when the spec is not separable: a spec module importing a proof module, a pin
whose statement names a definition of a proof module, a module neither rule places, or a trusted file (`TRUSTED.sha256`,
the grader's allowed imports) among the moved ones. It is idempotent: on a tree already split it says so and exits 0, and a
half-split tree is refused. A verified `--apply` also removes this script and its test (`OBSOLETE`): from then on the
package boundary holds what they check, and a rerun on a later `main` starts from the commit that adds them.

The spec is `Pous.Protocol`, `Pous.Assumptions` and `Pous.Guarantees` with what is under them, plus `SPEC_EXTRA`: two
modules under the proofs' name that the grader trusts (`grade.sh` lets a submission import them, `TRUSTED.sha256` hashes
them) and whose definitions pinned statements read. They stay with the spec under their names, as PoUW's
`Pouw.SecurityProofs.Dimension.Lifting` does. The grader, its controls and the open targets stay beside the spec.

Lake resolves a module to the first package whose library claims it, the root package's first, and a library rooted at
`Pous` claims every `Pous.*` module. So the proofs' library claims its modules by glob, generated here from the tree.

Three things wait on the validator (`tools/lean/audit.py`): validating `security_proofs/pous` against the spec's lock; loading
two packages that share the root `Pous` (Lean's search path resolves a module's first component to one directory, so
`Facts.lean` and `Replay.lean`, which import through it, find only one package's `Pous/`); and recomputing the spec's
`dependencies` record, which covers fewer Mathlib modules once the proofs are gone (copied here unchanged until then).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SPEC = "protocols/pous/lean"
PROOFS = "security_proofs/pous"
PROOFS_PACKAGE = "pous_security_proofs"
PROOFS_LIB = "PousSecurityProofs"

SPEC_PREFIXES = ("Pous.Protocol", "Pous.Assumptions", "Pous.Guarantees")
SPEC_EXTRA = {
    "Pous.SecurityProofs.Sanity": "trusted by the grader; the sanity pins read its schemes (`honestSim`, `rawSeq`, `junk`)",
    "Pous.SecurityProofs.Accounting.Numbers": "trusted by the grader; `report_k214` reads its `reportP₀`",
}
#: modules that stay beside the spec but are not spec: the open targets, the grader and its positive control
STAY = ("PousTargets", "Grader", "submissions")
#: the umbrella and the proofs, and the headline `#print axioms` list, which imports the umbrella
PROOF_PREFIXES = ("Pous",)
PROOF_EXTRA = ("CheckAxioms",)

#: the lock's package-level sections, split here; every other section stays with the spec byte for byte
PACKAGE_SECTIONS = ("roots", "exempt", "layers", "dependencies", "runs", "compile_time")
#: sections whose split needs a person: the run refuses while one is non-empty
UNSPLIT = ("escapes", "dependency_unchecked")
PIN_SECTIONS = ("guarantees", "pins")
#: removed by a verified --apply: after the split the package boundary holds what these check
OBSOLETE = ("protocols/pous/lean_split.py", "protocols/pous/tests/test_pous_lean_split.py")

PROOFS_LAKEFILE_NOTE = """\
# PoUS's proofs, which require its spec (`protocols/pous/lean`). Lake resolves a module to the first package whose library
# claims it, and a library rooted at `Pous` would claim the spec's modules too, so this one claims its own by glob: a new
# module directly under `Pous/SecurityProofs/`, or a new directory there, needs a glob here."""


def under(m: str, prefixes) -> bool:
    return any(m == p or m.startswith(p + ".") for p in prefixes)


def modules(pkg: Path) -> dict[str, str]:
    """{module: path relative to ``pkg``} for the package's `.lean` files (not `.lake`, not a nested package)."""
    out = {}
    for p in sorted(pkg.rglob("*.lean")):
        rel = p.relative_to(pkg)
        if ".lake" in rel.parts or any((pkg / Path(*rel.parts[:i]) / "lakefile.toml").exists() for i in range(1, len(rel.parts))):
            continue
        out[".".join(rel.with_suffix("").parts)] = rel.as_posix()
    return out


def imports(path: Path) -> list[str]:
    return [line.split()[1] for line in path.read_text().splitlines() if re.match(r"\s*import\s+\S", line)]


DECL = re.compile(r"^\s*(?:@\[[^\]]*\]\s*)*(?:(?:private|protected|noncomputable|partial|unsafe)\s+)*"
                  r"(?:def|abbrev|structure|inductive|class|theorem|lemma|instance|opaque|axiom)\s+([^\s:({\[]+)")


def declarations(pkg: Path, mods: dict[str, str]) -> dict[str, str]:
    """{declaration: module} from the sources, namespaces followed (an approximation: the exact list is the validator's)."""
    where: dict[str, str] = {}
    for m, rel in mods.items():
        ns: list[str] = []
        for line in (pkg / rel).read_text().splitlines():
            s = line.strip()
            if s.startswith("namespace "):
                ns.append(s.split()[1])
            elif s.startswith("end ") and ns and s.split()[1] in (ns[-1], ns[-1].split(".")[-1]):
                ns.pop()
            elif d := DECL.match(line):
                where.setdefault(".".join([*ns, d.group(1)]), m)
    return where


def place(m: str) -> str:
    if under(m, SPEC_PREFIXES) or m in SPEC_EXTRA:
        return "spec"
    if under(m, STAY):
        return "stay"
    if under(m, PROOF_PREFIXES) or m in PROOF_EXTRA:
        return "proofs"
    return "?"


def pin_section(lock: dict) -> str:
    return next((k for k in PIN_SECTIONS if k in lock), "pins")


def proof_globs(proof_mods: set[str], spec_mods: set[str]) -> list[str]:
    """Globs claiming exactly ``proof_mods``: `X.+` for a directory with no spec module under it, else its modules one by one."""
    out: list[str] = []

    def walk(prefix: str) -> None:
        here = {m for m in proof_mods | spec_mods if m.startswith(prefix + ".")}
        kids = sorted({m[len(prefix) + 1:].split(".")[0] for m in here})
        for k in kids:
            name = f"{prefix}.{k}"
            sub = {m for m in here if m.startswith(name + ".")}
            if name in proof_mods:
                out.append(name)
            if not sub:
                continue
            if sub <= proof_mods:
                out.append(f"{name}.+")
            else:
                walk(name)

    for top in sorted({m.split(".")[0] for m in proof_mods}):
        if top in proof_mods:
            out.append(top)
        if {m for m in spec_mods if m.startswith(top + ".")}:
            walk(top)
        elif {m for m in proof_mods if m.startswith(top + ".")}:
            out.append(f"{top}.+")
    return out


def split_by_module(section: dict, placed: dict[str, str]) -> tuple[dict, dict, list[str]]:
    """``section`` ({module or module prefix: v}) split into (spec's, proofs', problems) by where its modules go."""
    spec, proofs, problems = {}, {}, []
    for k, v in section.items():
        sides = {p for m, p in placed.items() if under(m, (k,))} or {place(k)}
        if sides <= {"spec", "stay"}:
            spec[k] = v
        elif sides == {"proofs"}:
            proofs[k] = v
        else:
            problems.append(f"{k}: its modules go to both packages ({', '.join(sorted(sides))})")
    return spec, proofs, problems


class Plan:
    def __init__(self, repo: Path):
        self.repo = repo
        self.spec_dir, self.proofs_dir = repo / SPEC, repo / PROOFS
        self.mods = modules(self.spec_dir)
        self.lock = json.loads((self.spec_dir / "lean-audit.json").read_text())
        self.placed = {m: place(m) for m in self.mods}
        self.spec = {m for m, p in self.placed.items() if p == "spec"}
        self.proofs = {m for m, p in self.placed.items() if p == "proofs"}
        self.problems: list[str] = []
        self.notes: list[str] = []
        if self.proofs_dir.exists() or not self.proofs:
            return
        self._separable()
        self._lock()

    @property
    def done(self) -> bool:
        return (self.proofs_dir / "lakefile.toml").exists() and not self.proofs

    @property
    def partial(self) -> bool:
        return self.proofs_dir.exists() and not self.done

    def _separable(self) -> None:
        P = self.problems
        P += [f"{m} ({self.mods[m]}): placed by no rule; add it to SPEC_EXTRA, STAY or PROOF_EXTRA"
              for m, p in sorted(self.placed.items()) if p == "?"]
        for m in sorted(self.spec):
            bad = [i for i in imports(self.spec_dir / self.mods[m]) if i in self.mods and self.placed[i] != "spec"]
            P += [f"spec module {m} imports {i}, which is not spec" for i in bad]
        where = declarations(self.spec_dir, self.mods)
        pins = self.lock.get(pin_section(self.lock), {})
        for pin, rec in sorted(pins.items()):
            text = " ".join(rec.get("signature", []))
            read = sorted({(n, where[n]) for n in re.findall(r"[A-Za-z_][\w₀-₉'.]*[\w₀-₉']", text)
                           if n != pin and n in where and self.placed.get(where[n]) != "spec"})
            P += [f"pin {pin} reads {n}, declared in {m}, which is not spec" for n, m in read]
        reads = self.lock.get("reads", {})
        P += [f"reads module {m} is not spec" for m in sorted(reads) if m in self.mods and self.placed[m] != "spec"]
        trusted = (self.spec_dir / "TRUSTED.sha256").read_text().split()[1::2]
        moved = {self.mods[m] for m in self.proofs}
        P += [f"{f} is trusted (TRUSTED.sha256) but would move" for f in trusted if f in moved]
        grade = (self.spec_dir / "grade.sh").read_text()
        allowed = {x for case in re.findall(r"^\s*([\w.*|]+)\)\s*;;", grade, re.M) for x in case.split("|")}
        P += [f"grade.sh lets a submission import {m}, which would move" for m in sorted(allowed) if m in self.proofs]
        self.notes.append(f"spec import closure: {len(self.spec)} modules, none imports a module outside the spec")
        self.notes.append(f"pins: {len(pins)} statements checked against the declarations of the spec's modules "
                          "(names read from the records' signatures; the validator's reads check is exact)")

    def _lock(self) -> None:
        lock = self.lock
        owner = lambda m: self.placed.get(m, place(m))  # noqa: E731
        spec, proofs = {}, {}
        for k in UNSPLIT:
            if lock.get(k):
                self.problems.append(f"lock section {k!r} is not empty: split it by hand")
        for k in ("exempt", "layers", "runs", "compile_time"):
            if k in lock:
                a, b, bad = split_by_module(lock[k], self.placed)
                self.problems += [f"lock {k}: {x}" for x in bad]
                if a or k in ("exempt", "layers"):
                    spec[k] = a
                if b:
                    proofs[k] = b
        spec["roots"] = self.spec_roots()
        proofs["roots"] = sorted(r for r in lock.get("roots", []) if owner(r) == "proofs") or ["Pous"]
        if "exempt" in proofs:
            # `lake env lean` resolves `Pous.*` through one search directory, which now holds only one package's modules
            proofs["exempt"] = {k: v.replace("lake env lean ", "lake lean ") for k, v in proofs["exempt"].items()}
        if "dependencies" in lock:
            spec["dependencies"] = proofs["dependencies"] = lock["dependencies"]
        self.spec_lock = {k: (spec[k] if k in spec else v) for k, v in lock.items() if k not in PACKAGE_SECTIONS or k in spec}
        self.spec_lock |= {k: v for k, v in spec.items() if k not in self.spec_lock}
        self.proofs_lock = proofs

    def spec_roots(self) -> list[str]:
        """The spec's umbrellas: the three layers, then each `SPEC_EXTRA` module none of them imports."""
        roots = [p for p in SPEC_PREFIXES if p in self.mods]
        seen, todo = set(), list(roots)
        while todo:
            m = todo.pop()
            if m in seen or m not in self.mods:
                continue
            seen.add(m)
            todo += imports(self.spec_dir / self.mods[m])
        return roots + sorted(m for m in self.spec if m not in seen)

    def moves(self) -> list[tuple[str, str]]:
        return [(f"{SPEC}/{self.mods[m]}", f"{PROOFS}/{self.mods[m]}") for m in sorted(self.proofs)]

    def print(self) -> None:
        if self.done:
            print(f"lean_split: {SPEC} is split already ({PROOFS} exists, no proof module left in the spec): nothing to do")
            return
        if self.partial:
            print(f"lean_split: REFUSED: {PROOFS} exists but {len(self.proofs)} proof modules are still in {SPEC}")
            return
        counts = {p: sum(v == p for v in self.placed.values()) for p in ("spec", "proofs", "stay")}
        print(f"lean_split: {len(self.mods)} modules in {SPEC}: {counts['spec']} spec, {counts['proofs']} to {PROOFS}, "
              f"{counts['stay']} stay beside the spec (the open targets, the grader, its control)")
        for m, why in SPEC_EXTRA.items():
            print(f"  spec under the proofs' name: {m}: {why}")
        print("separability:")
        for n in self.notes:
            print(f"  {n}")
        for p in self.problems:
            print(f"  VIOLATION: {p}")
        if self.problems:
            return
        print(f"git mv ({len(self.moves())} files):")
        for a, b in self.moves():
            print(f"  {a} -> {b}")
        print("written:")
        print(f"  {PROOFS}/lakefile.toml        requires mathlib (the spec's rev) and pous by path; library {PROOFS_LIB}, "
              f"{len(self.globs())} globs")
        print(f"  {PROOFS}/lake-manifest.json   the spec's, with the path entry first and the name {PROOFS_PACKAGE}")
        print(f"  {PROOFS}/lean-toolchain       the spec's")
        policy = json.dumps({k: v for k, v in self.proofs_lock.items() if k != "dependencies"}, ensure_ascii=False)
        print(f"  {PROOFS}/lean-audit.json      {policy} + the spec's dependencies record")
        print("edited:")
        print(f"  {SPEC}/lakefile.toml          library Pous rooted at {', '.join(self.spec_roots())}")
        changed = [k for k in self.spec_lock if self.spec_lock[k] != self.lock.get(k)]
        changed += [k for k in self.lock if k not in self.spec_lock]
        print(f"  {SPEC}/lean-audit.json        {', '.join(changed)} changed; byte-identical: "
              f"{', '.join(k for k in self.lock if k not in changed and k in self.spec_lock)}")
        print(f"  {SPEC}/TRUSTED.sha256         regenerated with README.md's command (lakefile.toml's hash)")
        print(f"removed: {', '.join(OBSOLETE)}")
        print(f"unchanged: Grader/Registry.lean (its targets are Pous.Guarantees.*), grade.sh (it builds only spec modules), "
              f"check.sh, PousTargets.lean")

    def globs(self) -> list[str]:
        """The proofs' library: every proof module but `PROOF_EXTRA`, which is run file by file (`lake lean CheckAxioms.lean`)."""
        return proof_globs(self.proofs - set(PROOF_EXTRA), self.spec)


def load_audit(repo: Path):
    spec = importlib.util.spec_from_file_location("lean_audit_for_split", repo / "tools" / "lean" / "audit.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def manifest_text(doc: dict) -> str:
    """A `lake-manifest.json` as Lake writes it."""
    def entry(e: dict) -> str:
        return "{" + ",\n   ".join(f"{json.dumps(k)}: {json.dumps(v)}" for k, v in e.items()) + "}"
    head = [f"{json.dumps(k)}: {json.dumps(v)}" for k, v in doc.items() if k != "packages"]
    i = list(doc).index("packages")
    pk = '"packages":\n [' + ",\n  ".join(entry(e) for e in doc["packages"]) + "]"
    return "{" + ",\n ".join([*head[:i], pk, *head[i:]]) + "}\n"


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout


def apply(plan: Plan) -> list[str]:
    """Perform ``plan``; the problems the checks after it find (empty: done and verified)."""
    repo, sd, pd = plan.repo, plan.spec_dir, plan.proofs_dir
    if dirty := git(repo, "status", "--porcelain", "--", SPEC, PROOFS, *OBSOLETE).strip():
        return [f"uncommitted changes under {SPEC}, {PROOFS} or to this script:\n{dirty}"]
    audit = load_audit(repo)
    old_text = (sd / "lean-audit.json").read_text()
    if audit.text(plan.lock) != old_text:
        return ["tools/lean/audit.py does not write lean-audit.json as it is: the lock would be reformatted, not split"]
    man_text = (sd / "lake-manifest.json").read_text()
    man = json.loads(man_text)
    if manifest_text(man) != man_text:
        return ["lake-manifest.json is not in the layout this script writes"]

    for a, b in plan.moves():
        (repo / b).parent.mkdir(parents=True, exist_ok=True)
        git(repo, "mv", a, b)

    req = re.search(r'\[\[require\]\]\s*\nname = "mathlib"\n(?:[^\[\n].*\n)*', (sd / "lakefile.toml").read_text())
    opts = re.search(r"\[leanOptions\]\n(?:[^\[\n].*\n)*", (sd / "lakefile.toml").read_text())
    rel = Path("../" * len(Path(PROOFS).parts)) / SPEC
    globs = ",\n  ".join(json.dumps(g) for g in plan.globs())
    (pd / "lakefile.toml").write_text(
        f'name = "{PROOFS_PACKAGE}"\ndefaultTargets = ["{PROOFS_LIB}"]\n\n{opts.group(0).rstrip()}\n\n{req.group(0).rstrip()}\n\n'
        f'[[require]]\nname = "{man["name"]}"\npath = "{rel.as_posix()}"\n\n{PROOFS_LAKEFILE_NOTE}\n'
        f'[[lean_lib]]\nname = "{PROOFS_LIB}"\nroots = []\nglobs = [\n  {globs}]\n')
    path_entry = {"type": "path", "scope": "", "name": man["name"], "manifestFile": "lake-manifest.json",
                  "inherited": False, "dir": rel.as_posix(), "configFile": "lakefile.toml"}
    (pd / "lake-manifest.json").write_text(manifest_text(man | {"packages": [path_entry, *man["packages"]],
                                                                "name": PROOFS_PACKAGE}))
    (pd / "lean-toolchain").write_bytes((sd / "lean-toolchain").read_bytes())
    (pd / "lean-audit.json").write_text(audit.text(plan.proofs_lock))

    lake = (sd / "lakefile.toml").read_text()
    roots = "roots = [" + ", ".join(json.dumps(r) for r in plan.spec_roots()) + "]"
    block = re.search(r'(\[\[lean_lib\]\]\nname = "Pous"\n)((?:roots = .*\n)?)', lake)
    lake = lake[:block.start()] + "# The spec; its proofs are the package security_proofs/pous.\n" * (not block.group(2)) + \
        block.group(1) + roots + "\n" + lake[block.end():]
    (sd / "lakefile.toml").write_text(lake)
    new_text = audit.text(plan.spec_lock)
    (sd / "lean-audit.json").write_text(new_text)

    cmd = re.search(r"^(sha256sum [^\n]*(?:\\\n[^\n]*)*> TRUSTED\.sha256)$", (sd / "README.md").read_text(), re.M)
    if not cmd:
        return ["README.md has no TRUSTED.sha256 regeneration command"]
    subprocess.run(["bash", "-c", cmd.group(1)], cwd=sd, check=True, env={"LC_ALL": "C", "PATH": "/usr/bin:/bin"})
    git(repo, "add", "--", SPEC, PROOFS)
    problems = verify(plan, old_text, new_text, audit)
    if not problems:
        git(repo, "rm", "-q", "--", *OBSOLETE)
    return problems


def verify(plan: Plan, old_text: str, new_text: str, audit) -> list[str]:
    problems = []
    for k, v in plan.lock.items():
        if k in PACKAGE_SECTIONS:
            continue
        section = f" {json.dumps(k, ensure_ascii=False)}: {audit.layout(v, ' ')}"
        if section not in old_text:
            problems.append(f"lock section {k}: not found as laid out in the original (the check is unreliable)")
        elif section not in new_text:
            problems.append(f"lock section {k}: not byte-identical in the spec's lock")
    left = Plan(plan.repo)
    if not left.done:
        problems.append(f"after the moves the spec still holds {len(left.proofs)} proof modules")
    gone = sorted(set(left.mods) ^ (plan.spec | {m for m, p in plan.placed.items() if p == "stay"}))
    problems += [f"{m}: in the spec after the split, or missing from it" for m in gone]
    trusted = subprocess.run(["sha256sum", "--quiet", "-c", "TRUSTED.sha256"], cwd=plan.spec_dir, capture_output=True, text=True)
    if trusted.returncode:
        problems.append(f"TRUSTED.sha256 does not check: {trusted.stdout}{trusted.stderr}")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--apply", action="store_true", help="perform the split in this checkout and verify it")
    ap.add_argument("--repo", type=Path, default=REPO, help="the checkout (default: the one holding this script)")
    ns = ap.parse_args(argv)
    plan = Plan(ns.repo.resolve())
    plan.print()
    if plan.done:
        return 0
    if plan.partial or plan.problems:
        return 1
    if not ns.apply:
        print("dry run: nothing changed (--apply performs it)")
        return 0
    problems = apply(plan)
    for p in problems:
        print(f"lean_split: FAILED CHECK: {p}")
    if not problems:
        print("lean_split: applied and staged; every non-package section of the lock is byte-identical, TRUSTED.sha256 checks")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
