"""`audit.py`: the Lean audit, one run over every Lake package that carries a `lean-audit.json`.

    python tools/lean/audit.py [--build] [--update] [--out DIR] PACKAGE_DIR...
    python tools/lean/audit.py --all [--root DIR] [--build] [--out DIR]   the controls, then every package under DIR (default .)

For each package it imports the policy's `roots` into a file that `Audit.lean` completes, runs it with `lake env lean`, and
replays the audited modules through the kernel (`Replay.lean`). It fails a package when:

    axioms       a declaration of an audited module uses an axiom outside `axioms` (default propext, Classical.choice,
                 Quot.sound): `sorryAx` is a sorry, `Lean.ofReduceBool` or `Lean.trustCompiler` a native_decide
    declared     an audited module declares an `axiom` (assumptions are named `Prop`s, taken as hypotheses)
    escapes      a declaration a proof cannot unfold (`opaque`, which a `partial def` is, `unsafe`, `@[implemented_by]`,
                 `@[extern]`) is not listed in `escapes` with a reason, or a listed one no longer is one
    orphans      a .lean file of the package is neither an audited module nor under an `exempt` prefix
    unchecked    a source file or the lakefile names `debug.skipKernelTC`, an option that turns proofs into `sorry`, a
                 `bootstrap.*` option, `addDeclWithoutChecking` or `addDeclCore`
    layers       a module under a `layers` prefix (other than its `except` modules) imports a module outside the layer's
                 `imports` prefixes
    pins         a theorem in `pins` is missing or not a theorem, uses a disallowed axiom, has a closed `Prop` hypothesis
                 that is not a constant of an `assumptions` module, or no longer says what its record says: signature,
                 named assumptions, type hash, and (under `reads`) the definitions the pinned statements read, per module
    replay       the kernel rejects a declaration of an audited module when `Replay.lean` sends them all to it again, on
                 one import of what they import from outside: a declaration that skipped the kernel shows no axiom, and
                 only this catches it (the `kernel` control); or the axioms found by walking the replayed environment
                 differ from those the `.olean` files recorded; a replay killed by a signal is reported as a kill
    upstream     a dependency that the policy's `upstream` section scans proves, at the revision `lake-manifest.json` pins,
                 a theorem about a watched assumption or hand proof that the entry's `ack` does not hold (`upstream.py`):
                 a bump that brings such a proof fails until someone has looked at it
    fresh        (`--fresh` only) `leanchecker --fresh` on a root rejects a declaration of its whole import closure,
                 Mathlib and core included
    tamper       this tool's files or the toolchain's `lean`, `lake` or `leanchecker` changed during the run (a build runs
                 arbitrary code); the Lean files are run from a copy taken before the first build

A record holds the statement, not the proof, so a pin may be recorded on a `sorry` stub and pass once a proof with allowed
axioms replaces it.  `--update` writes the records and `reads` from the build instead of comparing them, and prints what
changed: each changed pin's signature before and after, and each changed definition a pin reads with its source now.  A
weakened definition passes every axiom check and the replay, so those changes need a named statement reviewer in the merge
handoff; `--update` records them, it does not approve them.  `--all` first runs the controls (`controls`): a
fixture package built with each audited toolchain, which must pass clean and fail each check once a violation is added.
Exit 0 when every package passes, 1 when one fails, 2 on a usage error.

This directory is self-contained (the standard library, `lake` and the toolchain's `lean` and `leanchecker`): another
repository can copy it and use it with policies of its own (`README.md`).
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
#: where the Lean files are run from: `main` points it at a copy taken before the first build
TOOL = HERE
ROOT = HERE.parents[1]
POLICY = "lean-audit.json"
STANDARD = ["Classical.choice", "Quot.sound", "propext"]
#: compiler trust: `native_decide` declares an axiom `<decl>._native.native_decide.ax_<i>` (older toolchains used these two)
NATIVE = {"Lean.ofReduceBool", "Lean.trustCompiler"}


def native(axiom: str) -> bool:
    return axiom in NATIVE or "._native." in axiom
SKIP_DIRS = {".lake", ".git", ".venv", "node_modules", "__pycache__"}
#: options that skip the kernel or turn proofs into `sorry`, and the entry points that add a declaration unchecked; refused
#: anywhere in a package's sources and lakefile.  A scan is evaded by building the name at run time, which the replay catches.
UNCHECKED = re.compile(r"\b(debug\.skipKernelTC|skipKernelTC|debug\.byAsSorry|debug\.proofAsSorry|debug\.terminalTacticsAsSorry"
                       r"|bootstrap\.\w+|addDeclWithoutChecking|addDeclCore)\b")
#: the controls' own packages; `--all` does not audit them as the repository's
FIXTURES = HERE / "tests" / "fixtures"
#: name components of the declarations Lean generates for a definition (listed under `reads` with it, not by name)
GENERATED = re.compile(r"^(_.*|match_\d+|proof_\d+|eq_\d+|eq_def|mk|rec|recOn|casesOn|noConfusion|noConfusionType|below|brecOn"
                       r"|binductionOn|ibelow|ctorIdx|inj|injEq|sizeOf_spec)$")
ORDER = ("roots", "exempt", "axioms", "assumptions", "meaning", "escapes", "layers", "upstream", "pins", "reads")


def sibling(name: str):
    """This directory's `name.py`, loaded by path (the directory is copied into other repositories, not installed)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(f"lean_{name}", HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def lake_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PATH"] = f"{Path.home() / '.elan' / 'bin'}{os.pathsep}{env.get('PATH', '')}"
    return env


def run(argv: list[str], cwd: Path, log: Path | None = None, timeout: float = 4 * 3600) -> tuple[int, str]:
    r = subprocess.run(argv, cwd=cwd, env=lake_env(), capture_output=True, text=True, timeout=timeout)
    text = r.stdout + r.stderr
    if log is not None:
        log.write_text(text)
    return r.returncode, text


def load(pkg: Path) -> dict:
    return json.loads((pkg / POLICY).read_text())


def layout(v, pad: str = "") -> str:
    """JSON with a list of short scalars on one line and everything else one entry per line: a record's diff is readable."""
    if isinstance(v, dict) and v:
        return "{\n" + ",\n".join(f"{pad} {json.dumps(k, ensure_ascii=False)}: {layout(x, pad + ' ')}" for k, x in v.items()) + \
            f"\n{pad}}}"
    one = json.dumps(v, ensure_ascii=False)
    if not isinstance(v, list) or not v or (len(one) + len(pad) < 110 and not any(isinstance(x, (dict, list)) for x in v)):
        return one
    return "[\n" + ",\n".join(f"{pad} {layout(x, pad + ' ')}" for x in v) + f"\n{pad}]"


def dump(pkg: Path, policy: dict) -> None:
    doc = {k: policy[k] for k in ORDER if k in policy} | {k: v for k, v in policy.items() if k not in ORDER}
    (pkg / POLICY).write_text(layout(doc) + "\n")


def under(prefixes, name: str) -> bool:
    return any(name == p or name.startswith(p + ".") for p in prefixes)


def lean_modules(pkg: Path) -> dict[str, str]:
    """{module: path relative to the package} for the package's own .lean files (not `.lake`, not a nested package)."""
    out = {}
    for d, dirs, files in os.walk(pkg):
        here = Path(d)
        dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS and not any((here / x / f).exists()
                                                                          for f in ("lakefile.toml", "lakefile.lean")))
        for f in files:
            if f.endswith(".lean") and not (here == pkg and f == "lakefile.lean"):
                rel = (here / f).relative_to(pkg)
                out[".".join(rel.with_suffix("").parts)] = str(rel)
    return out


def unchecked(pkg: Path, files: dict[str, str]) -> list[str]:
    """Where the package's sources or lakefile name an option or entry point that bypasses the kernel (`UNCHECKED`)."""
    hits = []
    for rel in [*sorted(files.values()), *(f for f in ("lakefile.toml", "lakefile.lean") if (pkg / f).exists())]:
        for i, line in enumerate((pkg / rel).read_text(errors="replace").splitlines(), 1):
            if m := UNCHECKED.search(line):
                hits.append(f"unchecked: {rel}:{i} names {m.group(1)}, which lets a declaration skip the kernel or a proof "
                            f"become sorry")
    return hits


def facts_of(pkg: Path, policy: dict, work: Path) -> dict:
    """Run `Audit.lean` on the package: the facts document it writes."""
    roots = policy["roots"]
    lit = lambda names: "#[" + ", ".join(f"`{n}" for n in names) + "]"   # noqa: E731
    out = work / "facts.json"
    head = ["import Lean", *(f"import {r}" for r in roots), "", "set_option maxHeartbeats 0", "set_option format.width 100",
            "set_option pp.proofs false", "", "namespace LeanAudit.Config",
            f"def prefixes : Array Lean.Name := {lit(roots)}",
            f"def meaning : Array Lean.Name := {lit(policy.get('meaning', roots))}",
            f"def allowed : Array Lean.Name := {lit(policy.get('axioms', STANDARD))}",
            f"def pins : Array Lean.Name := {lit(sorted(policy.get('pins', {})))}",
            f"def out : System.FilePath := {json.dumps(str(out))}", "end LeanAudit.Config", ""]
    src = work / "LeanAuditRun.lean"
    src.write_text("\n".join(head) + "\n" + (TOOL / "Audit.lean").read_text())
    rc, text = run(["lake", "env", "lean", str(src)], pkg, work / "lean.log")
    if rc or not out.exists():
        raise RuntimeError(f"`lake env lean` on the audit file failed (exit {rc}):\n{text[-3000:]}")
    return json.loads(out.read_text())


def digest(entries: list[dict]) -> str:
    return hashlib.sha256("".join(f"{e['name']} {e['kind']} {e['hash']}\n" for e in sorted(entries, key=lambda e: e["name"]))
                          .encode()).hexdigest()[:32]


def parent(name: str) -> str:
    """The declaration a generated constant belongs to: `Foo.match_1`, `Foo._f`, `Enc.mk` belong to `Foo`, `Foo`, `Enc`."""
    parts = name.split(".")
    while len(parts) > 1 and GENERATED.match(parts[-1]):
        parts.pop()
    return ".".join(parts)


def definitions(record: dict) -> dict:
    """A `reads` record's {definition: hash} (a record written before hashes were kept: {name: None})."""
    d = record.get("definitions", {})
    return d if isinstance(d, dict) else dict.fromkeys(d)


def judge(policy: dict, facts: dict, files: dict[str, str], update: bool = False) -> tuple[list[str], dict]:
    """(failures, the policy with pins and reads as the build has them).  A pure function of the facts: the tests run it."""
    fails = []
    allowed = set(policy.get("axioms", STANDARD))
    by_axiom: dict[str, list[str]] = {}
    for o in facts["offenders"]:
        for a in o["axioms"]:
            by_axiom.setdefault(a, []).append(o["name"])
    for a, names in sorted(by_axiom.items()):
        what = "sorry" if a == "sorryAx" else "native_decide" if native(a) else "axiom"
        names = sorted(n for n in names if n != a)
        more = f" and {len(names) - 8} more" if len(names) > 8 else ""
        fails.append(f"{what}: {a} is used by {', '.join(names[:8])}{more}")
    for d in facts["declared_axioms"]:
        if not native(d["name"]):
            fails.append(f"declared: `axiom {d['name']}` in {d['module']}; state an assumption as a named Prop, taken as a "
                         f"hypothesis")
    listed_escapes = policy.get("escapes", {})
    found = {e["name"]: e for e in facts["escapes"]}
    for n, e in sorted(found.items()):
        if n not in listed_escapes:
            fails.append(f"escape: {n} ({', '.join(e['why'])}, in {e['module']}) is compiled code no proof sees; list it under "
                         f"`escapes` with the reason, or remove it")
    for n in sorted(set(listed_escapes) - set(found)):
        fails.append(f"escape: {n} is listed under `escapes` but is not one; remove the entry")
    modules = set(facts["modules"])
    exempt = policy.get("exempt", {})
    for m, rel in sorted(files.items()):
        if m not in modules and not under(exempt, m):
            fails.append(f"orphan: {rel} is not imported by the roots ({', '.join(policy['roots'])}) nor under `exempt`: "
                         f"nothing builds or audits it")
    for p in sorted(exempt):
        if not any(under([p], m) for m in files):
            fails.append(f"orphan: `exempt` lists {p}, which matches no file; remove the entry")
    for layer, rule in sorted(policy.get("layers", {}).items()):
        inside = [m for m in sorted(modules) if under([layer], m)]
        if not inside:
            fails.append(f"layers: {layer} matches no audited module; remove the entry")
        for m in rule.get("except", []):
            if m not in inside:
                fails.append(f"layers: {layer}'s `except` lists {m}, which is not one of its modules")
        for m in inside:
            if m in rule.get("except", []):
                continue
            for i in facts["imports"].get(m, []):
                if not under(rule["imports"], i) and i != "Init":
                    fails.append(f"layers: {m} imports {i}, outside {layer}'s layer ({', '.join(rule['imports'])})")
    pins, reads = {}, {}
    for name, rec in sorted(policy.get("pins", {}).items()):
        f = facts["pins"].get(name, {"kind": "missing"})
        if f["kind"] != "theorem":
            fails.append(f"pins: {name} is {'missing' if f['kind'] == 'missing' else 'a ' + f['kind'] + ', not a theorem'}")
            pins[name] = rec
            continue
        if not set(f["axioms"]) <= allowed:
            fails.append(f"pins: {name} uses {', '.join(sorted(set(f['axioms']) - allowed))}")
        named = []
        for h in f["hypotheses"]:
            if not h["closed"]:
                continue
            if h["named"] and under(policy.get("assumptions", []), h["module"] or ""):
                named.append(h["named"])
            else:
                fails.append(f"pins: {name} assumes ({h['binder']} : {h['type']}), a closed Prop that is not a named assumption "
                             f"(a Prop constant of {', '.join(policy.get('assumptions', [])) or 'an `assumptions` module'})")
        now = {"signature": f["signature"].splitlines(), "assumptions": sorted(set(named)), "type_hash": f["type_hash"]}
        pins[name] = now
        for r in f["reads"]:
            reads.setdefault(r["module"], {}).setdefault(r["name"], r | {"pins": set()})["pins"].add(name)
        if update:
            continue
        if not rec:
            fails.append(f"pins: {name} has no record; run `audit.py --update` and commit the record")
            continue
        for k in ("signature", "assumptions", "type_hash"):
            if rec.get(k) != now[k]:
                diff = "\n".join(difflib.unified_diff(rec.get("signature", []), now["signature"], "recorded", "built", lineterm="",
                                                      n=1)) if k == "signature" else f"{rec.get(k)} -> {now[k]}"
                fails.append(f"pins: {name}'s {k.replace('_', ' ')} changed:\n{diff}")
                break
    now_reads = {}
    for m, es in sorted(reads.items()):
        groups: dict[str, list] = {}
        for n, e in es.items():
            groups.setdefault(parent(n), []).append(e)
        now_reads[m] = {"digest": digest(list(es.values())), "definitions": {p: digest(g)[:16] for p, g in sorted(groups.items())},
                        "pins": sorted(set().union(*(e["pins"] for e in es.values())))}
    if not update and ("reads" in policy or any(policy.get("pins", {}).values())):
        old = policy.get("reads", {})
        for m in sorted(set(old) | set(now_reads)):
            a, b = old.get(m), now_reads.get(m)
            if a is None:
                fails.append(f"pins: the pinned statements now read definitions in {m} ({', '.join(b['pins'])})")
            elif b is None:
                fails.append(f"pins: the pinned statements no longer read {m}")
            elif a["digest"] != b["digest"]:
                x, y = definitions(a), definitions(b)
                change = "; ".join(s for s in (
                    f"changed: {', '.join(d for d in sorted(set(x) & set(y)) if x[d] is not None and x[d] != y[d])}",
                    f"new: {', '.join(sorted(set(y) - set(x)))}", f"gone: {', '.join(sorted(set(x) - set(y)))}")
                    if not s.endswith(": ")) or "a definition's content"
                fails.append(f"pins: definitions in {m} that {', '.join(b['pins'])} read changed ({change})")
    out = dict(policy)
    if pins:
        out["pins"], out["reads"] = pins, now_reads
    return fails, out


def module_files(pkg: Path) -> dict[str, Path]:
    """{module: source file} for the package and the packages it requires by path, whose definitions a pin may read."""
    manifest = pkg / "lake-manifest.json"
    deps = [(pkg / d["dir"]).resolve() for d in json.loads(manifest.read_text()).get("packages", [])
            if d.get("type") == "path"] if manifest.exists() else []
    return {m: d / rel for d in [*deps, pkg] for m, rel in lean_modules(d).items()}


def review(old: dict, new: dict, facts: dict, sources: dict[str, Path]) -> list[str]:
    """What a statement reviewer reads for an `--update`: each changed pin's signature before and after, and each changed
    definition a pin reads, with the pins that read it and its source as it is now."""
    out = []
    op, np = old.get("pins", {}), new.get("pins", {})
    for name in sorted(set(op) | set(np)):
        a, b = op.get(name) or {}, np.get(name) or {}
        if a == b:
            continue
        out.append(f"pin {name}: {'new' if not a else 'removed' if not b else 'changed'}")
        out += ["  before:", *(f"    {s}" for s in a.get("signature", []))] if a else []
        out += ["  after:", *(f"    {s}" for s in b.get("signature", []))] if b else []
        if a and b and a.get("assumptions") != b.get("assumptions"):
            out.append(f"  named assumptions: {', '.join(a.get('assumptions', [])) or 'none'} -> "
                       f"{', '.join(b.get('assumptions', [])) or 'none'}")
    spans = {r["name"]: r.get("lines") for p in facts["pins"].values() for r in p.get("reads", [])}
    oreads, nreads = old.get("reads", {}), new.get("reads", {})
    for m in sorted(set(oreads) | set(nreads)):
        a, b = oreads.get(m), nreads.get(m)
        if a and b and a.get("digest") == b.get("digest"):
            continue
        x, y = definitions(a or {}), definitions(b or {})
        for d in sorted(set(x) | set(y)):
            if d in x and d in y and x[d] is not None and x[d] == y[d]:
                continue
            state = "new" if d not in x else "gone" if d not in y else "changed" if x[d] is not None else "in a changed module"
            out.append(f"definition {d} ({m}), read by {', '.join((b or a)['pins'])}: {state}")
            span, src = spans.get(d), sources.get(m)
            if d in y and span and src is not None and src.exists():
                text = src.read_text().splitlines()[span[0] - 1:span[1]]
                out += [f"  now ({os.path.relpath(src)}:{span[0]}-{span[1]}):", *(f"    {s}" for s in text)]
    return out


def killed(rc: int) -> bool:
    """A process ended by a signal (the machine's OOM killer, a timeout), which says nothing about the kernel's verdict."""
    return rc < 0 or rc >= 128


def audit(pkg: Path, out: Path, build: bool = False, update: bool = False, replay: bool = True, fresh: bool = False,
          log=print) -> dict:
    """Audit one package; returns its report (`failures` empty when it passes)."""
    policy = load(pkg)
    tag = re.sub(r"[^A-Za-z0-9]+", "_", str(pkg.resolve().relative_to(ROOT)) if pkg.resolve().is_relative_to(ROOT) else pkg.name)
    work = out / f"audit-{tag}"
    if work.resolve().is_relative_to(pkg.resolve()):
        raise ValueError(f"the audit's output {work} is inside the package {pkg}")
    work.mkdir(parents=True, exist_ok=True)
    rep = {"package": str(pkg), "toolchain": (pkg / "lean-toolchain").read_text().strip(), "seconds": {}, "failures": []}
    t = time.monotonic()
    if build:
        for step, argv in (("setup", ["bash", str(TOOL / "setup.sh"), str(pkg)]), ("build", ["lake", "build"]),
                           ("build-roots", ["lake", "build", *(f"+{r}" for r in policy["roots"])])):
            s = time.monotonic()
            rc, text = run(argv, pkg, work / f"{step}.log")
            rep["seconds"][step] = round(time.monotonic() - s, 1)
            if rc:
                rep["failures"].append(f"{step}: exit {rc}\n{text[-3000:]}")
                return rep
    s = time.monotonic()
    try:
        facts = facts_of(pkg, policy, work)
    except RuntimeError as e:
        rep["failures"].append(f"lean: {e}")
        return rep
    rep["seconds"]["audit"] = round(time.monotonic() - s, 1)
    files = lean_modules(pkg)
    fails, new = judge(policy, facts, files, update)
    fails = unchecked(pkg, files) + fails
    if "upstream" in policy:
        s = time.monotonic()
        rep["upstream"], missed = sibling("upstream").pinned(pkg, policy["upstream"])
        fails += missed
        rep["seconds"]["upstream"] = round(time.monotonic() - s, 1)
    if replay:
        s = time.monotonic()
        found = work / "replay.json"
        found.unlink(missing_ok=True)
        rc, text = run(["lake", "env", "lean", "--run", str(TOOL / "Replay.lean"), str(found), *facts["modules"]], pkg,
                       work / "replay.log")
        if killed(rc):
            fails.append(f"replay: killed (exit {rc}; out of memory?), which is not a kernel verdict: rerun it alone\n{text[-1000:]}")
        elif rc or not found.exists():
            fails.append(f"replay: the kernel rejected a declaration under {', '.join(policy['roots'])} (exit {rc}):\n"
                         f"{text[-2000:]}")
        elif (walked := sorted(json.loads(found.read_text())["axioms"])) != sorted(facts["axioms"]):
            fails.append(f"replay: the replayed environment uses the axioms {', '.join(walked) or 'none'}, the .olean files "
                         f"record {', '.join(sorted(facts['axioms'])) or 'none'}")
        rep["seconds"]["replay"] = round(time.monotonic() - s, 1)
    if fresh:
        s = time.monotonic()
        for r in policy["roots"]:
            rc, text = run(["lake", "env", "leanchecker", "--fresh", r], pkg, work / f"fresh-{r}.log")
            if killed(rc):
                fails.append(f"fresh: `leanchecker --fresh {r}` was killed (exit {rc}; out of memory?), not a kernel verdict")
            elif rc:
                fails.append(f"fresh: the kernel rejected a declaration in the closure of {r} (exit {rc}):\n{text[-2000:]}")
        rep["seconds"]["fresh"] = round(time.monotonic() - s, 1)
    if update and new != policy:
        dump(pkg, new)
        notes = review(policy, new, facts, module_files(pkg))
        (work / "review.txt").write_text("\n".join(notes) + "\n")
        rep["review"] = notes
        log(f"audit: {pkg}/{POLICY}: pins and reads rewritten from the build" + (
            "; a named statement reviewer must read these changes before the merge:\n" + "\n".join(f"  {s}" for s in notes)
            if notes else ""))
    rep |= {"lean": facts["lean"], "modules": len(facts["modules"]), "declarations": facts["declarations"],
            "axioms": facts["axioms"], "escapes": facts["escapes"], "pins": new.get("pins", {}), "reads": new.get("reads", {}),
            "failures": fails}
    rep["seconds"]["total"] = round(time.monotonic() - t, 1)
    (work / "report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n")
    return rep


def packages(root: Path) -> list[Path]:
    """Every directory under ``root`` with a `lean-audit.json`, the controls' fixtures excepted."""
    found = []
    for d, dirs, files in os.walk(root):
        dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS)
        here = Path(d)
        if POLICY in files and not here.is_relative_to(FIXTURES):
            found.append(here)
    return sorted(found)


def controls(toolchain: str, out: Path, log=print) -> list[str]:
    """Build the fixture package with ``toolchain``: it must pass clean, and fail each check once a violation is added."""
    fails = []
    work = out / f"controls-{re.sub(r'[^A-Za-z0-9]+', '_', toolchain)}"
    shutil.rmtree(work, ignore_errors=True)
    shutil.copytree(FIXTURES / "control", work / "pkg")
    pkg = work / "pkg"
    (pkg / "lean-toolchain").write_text(toolchain + "\n")
    rc, text = run(["lake", "build"], pkg, work / "build.log")
    if rc:
        return [f"controls: the clean fixture does not build with {toolchain}:\n{text[-2000:]}"]
    reports = work / "reports"
    audit(pkg, reports, update=True, log=lambda *_: None)
    clean = audit(pkg, reports)["failures"]
    if clean:
        fails.append("controls: the clean fixture fails the audit:\n" + "\n".join(clean))
    expect = json.loads((FIXTURES / "violations.json").read_text())
    for name, v in expect.items():
        bad = work / f"bad-{name}"
        shutil.rmtree(bad, ignore_errors=True)
        shutil.copytree(pkg, bad)
        for rel, text in v.get("write", {}).items():
            (bad / rel).parent.mkdir(parents=True, exist_ok=True)
            (bad / rel).write_text(text)
        for rel, (old, new) in v.get("replace", {}).items():
            src = (bad / rel).read_text()
            assert old in src, (name, rel, old)
            (bad / rel).write_text(src.replace(old, new))
        rc, text = run(["lake", "build"], bad, work / f"bad-{name}.log")
        if rc:
            fails.append(f"controls: `{name}` does not build:\n{text[-1000:]}")
            continue
        if v.get("replay_only"):
            if missed := audit(bad, reports, replay=False)["failures"]:
                fails.append(f"controls: `{name}` should pass every check but the replay; it failed:\n" + "\n".join(missed))
        got = audit(bad, reports, replay=v.get("replay", False) or v.get("replay_only", False))["failures"]
        if not any(f.startswith(v["fails"]) for f in got):
            fails.append(f"controls: `{name}` should fail `{v['fails']}`; the audit said:\n" + ("\n".join(got) or "PASS"))
    log(f"audit: controls with {toolchain}: {'PASS' if not fails else 'FAIL'} (clean, {', '.join(expect)})")
    return fails


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def toolchain_bins(toolchain: str) -> list[Path]:
    """The toolchain's `lean`, `lake` and `leanchecker`, where elan installs them."""
    d = Path.home() / ".elan" / "toolchains" / toolchain.replace("/", "--").replace(":", "---")
    return [d / "bin" / b for b in ("lean", "lake", "leanchecker")]


def tampered(seals: dict[str, str], watch: list[Path]) -> list[str]:
    """Paths among ``seals`` whose content changed; ``watch`` paths seen for the first time are sealed now."""
    for p in watch:
        if str(p) not in seals and p.exists():
            seals[str(p)] = sha256(p)
    return [p for p, h in seals.items() if not Path(p).exists() or sha256(Path(p)) != h]


def main(argv: list[str]) -> int:
    global TOOL
    ap = argparse.ArgumentParser(prog="audit.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("package", nargs="*", type=Path, help="a Lake package directory with a lean-audit.json")
    ap.add_argument("--all", action="store_true", help="the controls, then every package under --root")
    ap.add_argument("--root", type=Path, default=Path.cwd(), help="--all: where to look for lean-audit.json (default .)")
    ap.add_argument("--build", action="store_true", help="`setup.sh` (elan, the toolchain, Mathlib's cache) and `lake build` first")
    ap.add_argument("--update", action="store_true", help="write the pins' records from the build instead of comparing them")
    ap.add_argument("--no-replay", action="store_true", help="skip the kernel replay (`Replay.lean`)")
    ap.add_argument("--fresh", action="store_true", help="also `leanchecker --fresh` each root: its whole closure, Mathlib and core "
                    "included, through the kernel (slow: see README)")
    ap.add_argument("--out", type=Path, default=None, help="where the reports and logs go (default $RESEARCH_RUN_DIR/lean-audit in a "
                    "recorded run, else a temporary directory)")
    a = ap.parse_args(argv)
    pkgs = packages(a.root.resolve()) if a.all else a.package
    if not pkgs or (a.all and a.package) or (a.all and a.update):
        ap.error("name package directories, or --all (which does not --update)")
    run_dir = os.environ.get("RESEARCH_RUN_DIR")
    out = a.out or (Path(run_dir) / "lean-audit" if run_dir else Path(tempfile.mkdtemp(prefix="lean-audit-")))
    out.mkdir(parents=True, exist_ok=True)
    TOOL = out / "tool"
    shutil.rmtree(TOOL, ignore_errors=True)
    shutil.copytree(HERE, TOOL, ignore=shutil.ignore_patterns("tests", "__pycache__"))
    seals = {str(p): sha256(p) for d in (HERE, TOOL) for p in sorted(d.iterdir()) if p.is_file()}
    toolchains = sorted({(p / "lean-toolchain").read_text().strip() for p in pkgs})
    bins = [b for tc in toolchains for b in toolchain_bins(tc)]
    failed = []
    reports = []
    if a.all:
        for tc in toolchains:
            if fs := controls(tc, out):
                failed.append(f"controls ({tc})")
                print("\n".join("  " + f.replace("\n", "\n    ") for f in fs), flush=True)
    for pkg in pkgs:
        rep = audit(pkg, out, a.build, a.update, not a.no_replay, a.fresh)
        rep["failures"] += [f"tamper: {p} changed while {pkg} was built and audited" for p in tampered(seals, bins)]
        reports.append(rep)
        n = sum(rep.get("declarations", {}).values())
        head = f"AUDIT {pkg}: {'PASS' if not rep['failures'] else 'FAIL'}"
        if "modules" in rep:
            head += (f"  {n} declarations in {rep['modules']} modules; axioms {', '.join(rep['axioms']) or 'none'}; "
                     f"{len(rep['pins'])} pinned theorems; {rep['seconds']}")
        print(head, flush=True)
        for f in rep["failures"]:
            print("  " + f.replace("\n", "\n    "), flush=True)
        if rep["failures"]:
            failed.append(str(pkg))
    if changed := [r["package"] for r in reports if r.get("review")]:
        print(f"audit: pinned statements or the definitions they read changed in {', '.join(changed)}: the merge handoff must "
              f"name the statement reviewer who read the changes above (review.txt in each package's reports)", flush=True)
    (out / "audit.json").write_text(json.dumps({"passed": not failed, "failed": failed, "packages": reports}, indent=1,
                                               ensure_ascii=False) + "\n")
    print(f"AUDIT: {'PASS' if not failed else 'FAIL (' + ', '.join(failed) + ')'}; reports in {out}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
