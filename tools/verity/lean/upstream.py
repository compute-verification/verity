"""`upstream.py`: notice when a dependency proves something a package assumes, or proves by hand.

    python tools/verity/lean/upstream.py PACKAGE_DIR [--rev REV] [--fetch]   no build: the statements at the pinned revision (or REV);
                                                                      --fetch clones a dependency that has no checkout yet
    python tools/verity/lean/upstream.py PACKAGE_DIR --lean [--out DIR]      exact: build `imports`, then `Upstream.lean`
    add --ack to either to list what is new under the entries' `ack`, after someone has looked at it

The watch list is the `upstream` section of the package's `lean-audit.json`, one entry per named assumption or hand-proved
lemma that an upstream theorem could discharge or replace:

    "upstream": {
     "scan": {"Arklib": {"baseline": "scripts/axiom_baseline.json", "sources": ["ArkLib"]}},
     "imports": ["ArkLib"],
     "watch": {"A1-johnson-mca": {"about": "Assumptions.BCHKS25Thm46 ...",
                                  "mentions": ["CoreDefinitions.mcaError", "ReedSolomon.code"], "ack": ["..."]}}
    }

A hit is an upstream theorem whose statement mentions every constant under `mentions`. Matching is on what statements are
about, not on names or citations: names move, and a proof can arrive as a different theorem with a different constant.

- The no-build pass reads each `scan` package's sources with `git show`, at the revision `lake-manifest.json` pins or at
  `--rev`. It matches the constants' last components as words in each theorem's statement, comments aside, and takes proved
  or not from the package's own transitive `sorry` baseline (ArkLib's `scripts/axiom_baseline.json`: `sorry` and
  `nonstandard` name what is not proved).
- The exact pass (`Upstream.lean`) matches the constants in the elaborated types of everything `imports` brings in, and
  collects each hit's axioms. It also covers packages that keep no baseline (VCVio, CompPoly).

A proved hit not under its entry's `ack` is new. Someone should check whether it discharges the assumption or replaces the
hand proof, then acknowledge it. Exit 1 when a hit is new, 0 otherwise. `audit.py` runs the no-build pass at the pinned
revisions, so a bump that brings a new proof about a watched statement fails until it is acknowledged.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
POLICY = "lean-audit.json"
STANDARD = {"propext", "Classical.choice", "Quot.sound"}
THEOREM = re.compile(r"^(?:@\[[^\]]*\]\s*)?(?:(?:private|protected|nonrec|noncomputable)\s+)*(?:theorem|lemma)\s+(\S+)", re.M)
DECLARATION = re.compile(r"\n(?:@\[|(?:private|protected|noncomputable|nonrec|partial|unsafe)\s|theorem\s|lemma\s|def\s|"
                         r"abbrev\s|instance\s|structure\s|class\s|inductive\s|namespace\s|section\b|end\b|open\s|variable\s)")


COMMENT = re.compile(r"/-|-/|--[^\n]*")


def strip_comments(text: str) -> str:
    """``text`` without its Lean comments (`--`, and `/- -/` nested, docstrings included), newlines kept."""
    out, last, depth, opened = [], 0, 0, 0
    for m in COMMENT.finditer(text):
        tok = m.group()
        if tok == "/-":
            if not depth:
                out.append(text[last:m.start()])
                opened = m.start()
            depth += 1
        elif tok == "-/" and depth:
            depth -= 1
            if not depth:
                out.append("\n" * text.count("\n", opened, m.end()))
                last = m.end()
        elif tok.startswith("--") and not depth:
            out.append(text[last:m.start()])
            last = m.end()
    out.append(text[last:] if not depth else "")
    return "".join(out)


def statement(text: str, start: int) -> str:
    """A theorem's statement: from its name to the first `:=`, `where` or `|` outside brackets, or the next declaration."""
    end = (m.start() if (m := DECLARATION.search(text, start)) else len(text))
    depth = 0
    for i in range(start, end):
        c = text[i]
        if c in "([{⟨":
            depth += 1
        elif c in ")]}⟩":
            depth -= 1
        elif depth > 0:
            continue
        elif text.startswith(":=", i):
            return text[start:i]
        elif c == "\n":
            j = i + 1
            while j < end and text[j] == " ":
                j += 1
            if j < end and text[j] == "|":
                return text[start:i]
        elif text.startswith("where", i) and not text[i - 1].isalnum() and not text[i + 5:i + 6].isalnum():
            return text[start:i]
    return text[start:end]


def namespaces(text: str, pos: int) -> list[str]:
    ns: list[str] = []
    for m in re.finditer(r"^\s*(namespace|end)\s+(\S+)", text[:pos], re.M):
        if m.group(1) == "namespace":
            ns.append(m.group(2))
        elif ns and ns[-1] == m.group(2):
            ns.pop()
    return ns


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True,
                          errors="surrogateescape").stdout


def manifested(pkg: Path, name: str) -> dict:
    """``name``'s entry in the package's `lake-manifest.json` ({} when it has none)."""
    return next((p for p in json.loads((pkg / "lake-manifest.json").read_text()).get("packages", []) if p.get("name") == name), {})


def pinned_rev(pkg: Path, name: str) -> str:
    """The revision of ``name`` that the package's `lake-manifest.json` pins."""
    if rev := manifested(pkg, name).get("rev"):
        return rev
    raise ValueError(f"lake-manifest.json pins no revision of {name}")


def fetch(pkg: Path, name: str) -> None:
    """`git fetch` the checkout of ``name`` under `.lake/packages/`.  Where there is none (a machine that never ran `setup.sh`),
    clone it from the manifest's url at the pinned revision, as `lake` would: the no-build pass needs git, not Lean."""
    repo = pkg / ".lake" / "packages" / name
    if (repo / ".git").exists():
        git(repo, "fetch", "--quiet", "origin")
        return
    if not (url := manifested(pkg, name).get("url")):
        raise ValueError(f"lake-manifest.json has no url for {name}")
    subprocess.run(["git", "clone", "--quiet", "--no-checkout", url, str(repo)], check=True, capture_output=True, text=True)
    git(repo, "checkout", "--quiet", "--detach", pinned_rev(pkg, name))


def scan(repo: Path, rev: str, scanned: dict, watch: dict) -> dict[str, list[dict]]:
    """{entry: [{name, file, proved}]}: the theorems of ``repo`` at ``rev`` whose statements mention each entry's constants."""
    base = json.loads(git(repo, "show", f"{rev}:{scanned['baseline']}"))
    unproved = set(base.get("sorry", [])) | set(base.get("nonstandard", []))
    words = {w: [re.compile(r"(?<![\w.])(?:[\w.]*\.)?%s(?![\w'])" % re.escape(c.split(".")[-1])) for c in e["mentions"]]
             for w, e in watch.items()}
    hits: dict[str, list[dict]] = {w: [] for w in watch}
    files = [f for f in git(repo, "ls-tree", "-r", "-z", "--name-only", rev, "--", *scanned["sources"]).split("\0")
             if f.endswith(".lean")]
    for f in files:
        text = strip_comments(git(repo, "show", f"{rev}:{f}"))
        for m in THEOREM.finditer(text):
            short, stmt = m.group(1), statement(text, m.end())
            ns = namespaces(text, m.start())
            for w, pats in words.items():
                if all(p.search(stmt) for p in pats):
                    names = [".".join(ns[i:] + [short]) for i in range(len(ns) + 1)]
                    hits[w].append({"name": names[0], "file": f, "proved": not any(n in unproved for n in names)})
    return hits


def new(hits: dict[str, list[dict]], watch: dict) -> dict[str, list[str]]:
    """{entry: proved hits its `ack` does not hold}."""
    return {w: sorted({h["name"] for h in hs if h["proved"]} - set(watch[w].get("ack", []))) for w, hs in hits.items()}


def pinned(pkg: Path, cfg: dict, rev: str | None = None) -> tuple[dict, list[str]]:
    """The no-build pass over every `scan` package: ({package: {rev, entries: {entry: {hits, proved, new}}}}, failures)."""
    status, fails = {}, []
    for name, scanned in cfg.get("scan", {}).items():
        repo = pkg / ".lake" / "packages" / name
        if not (repo / ".git").exists():
            fails.append(f"upstream: {repo} is not a git checkout; `setup.sh` fetches the package's dependencies")
            continue
        try:
            r = rev or pinned_rev(pkg, name)
            hits = scan(repo, r, scanned, cfg["watch"])
        except (ValueError, subprocess.CalledProcessError) as e:
            fails.append(f"upstream: cannot scan {name}: {getattr(e, 'stderr', '') or e}".strip())
            continue
        fresh = new(hits, cfg["watch"])
        status[name] = {"rev": r, "entries": {w: {"hits": len(hs), "proved": sum(h["proved"] for h in hs), "new": fresh[w]}
                                              for w, hs in hits.items()}}
        for w, names in fresh.items():
            for n in names:
                fails.append(f"upstream: {name} at {r[:12]} proves {n}, about {w} ({cfg['watch'][w]['about']}), and "
                             f"upstream.watch.{w}.ack does not hold it: check whether it discharges that or replaces a hand "
                             f"proof, then acknowledge it (`upstream.py --ack`)")
    return status, fails


def lean_pass(pkg: Path, cfg: dict, out: Path, build: bool = True) -> dict[str, list[dict]]:
    """The exact pass: {entry: [{name, module, axioms, proved}]} over everything `imports` brings in."""
    env = {**os.environ, "PATH": f"{Path.home() / '.elan' / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}"}

    def run(argv: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(argv, cwd=pkg, capture_output=True, text=True, env=env)

    if build and (r := run(["lake", "build", *cfg["imports"]])).returncode:
        raise RuntimeError(f"lake build {' '.join(cfg['imports'])} failed:\n{(r.stdout + r.stderr)[-3000:]}")
    out.mkdir(parents=True, exist_ok=True)
    config, found = out / "upstream-config.json", out / "upstream.json"
    found.unlink(missing_ok=True)
    config.write_text(json.dumps({"imports": cfg["imports"], "out": str(found),
                                  "watch": [{"entry": w, "mentions": e["mentions"]} for w, e in cfg["watch"].items()]}))
    r = run(["lake", "env", "lean", "--run", str(HERE / "Upstream.lean"), str(config)])
    if r.returncode or not found.exists():
        raise RuntimeError(f"Upstream.lean failed (exit {r.returncode}):\n{(r.stdout + r.stderr)[-3000:]}")
    result = json.loads(found.read_text())
    for w, missing in result["missing"].items():
        if missing:
            print(f"upstream: {w}: {', '.join(missing)} not in the imported environment", file=sys.stderr)
    return {w: [h | {"proved": not (set(h["axioms"]) - STANDARD)} for h in hs] for w, hs in result["watch"].items()}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="upstream.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("package", type=Path, help="a Lake package directory whose lean-audit.json has an `upstream` section")
    ap.add_argument("--rev", help="no-build pass: scan this revision of each `scan` package instead of the pinned one")
    ap.add_argument("--fetch", action="store_true", help="no-build pass: `git fetch` each `scan` package first (for --rev), "
                    "cloning it from lake-manifest.json when it has no checkout")
    ap.add_argument("--lean", action="store_true", help="the exact pass (builds `imports` first)")
    ap.add_argument("--no-build", action="store_true", help="--lean: skip `lake build`")
    ap.add_argument("--out", type=Path, default=None, help="--lean: where the config and result go (default a temporary directory)")
    ap.add_argument("--ack", action="store_true", help="add what is new to the entries' `ack` lists in lean-audit.json")
    a = ap.parse_args(argv)
    pkg = a.package.resolve()
    policy = json.loads((pkg / POLICY).read_text())
    cfg = policy.get("upstream")
    if not cfg:
        ap.error(f"{pkg / POLICY} has no `upstream` section")
    if a.lean:
        hits = lean_pass(pkg, cfg, a.out or Path(tempfile.mkdtemp(prefix="upstream-")), build=not a.no_build)
        where = "the imported environment"
    else:
        for name in cfg.get("scan", {}) if a.fetch else ():
            fetch(pkg, name)
        hits: dict[str, list[dict]] = {w: [] for w in cfg["watch"]}
        where = []
        for name, scanned in cfg.get("scan", {}).items():
            rev = a.rev or pinned_rev(pkg, name)
            where.append(f"{name} at {git(pkg / '.lake' / 'packages' / name, 'rev-parse', '--short=12', rev).strip()}")
            for w, hs in scan(pkg / ".lake" / "packages" / name, rev, scanned, cfg["watch"]).items():
                hits[w] += hs
        where = ", ".join(where)
    fresh = new(hits, cfg["watch"])
    print(f"upstream: {where}")
    for w, e in cfg["watch"].items():
        hs = hits[w]
        print(f"  {w}: {len(hs)} theorems, {sum(h['proved'] for h in hs)} proved, {len(fresh[w])} new  [{e['about']}]")
        for h in sorted(hs, key=lambda h: h["name"]):
            mark = "NEW   " if h["name"] in fresh[w] else "proved" if h["proved"] else "open  "
            print(f"    {mark} {h['name']}  ({h.get('module') or h.get('file')})")
    if a.ack and any(fresh.values()):
        for w, names in fresh.items():
            cfg["watch"][w]["ack"] = sorted(set(cfg["watch"][w].get("ack", [])) | set(names))
        import importlib.util
        spec = importlib.util.spec_from_file_location("lean_audit", HERE / "check.py")
        audit = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(audit)
        audit.dump(pkg, policy)
        print(f"upstream: acknowledged in {pkg / POLICY}: " + "; ".join(f"{w}: {', '.join(n)}" for w, n in fresh.items() if n))
        return 0
    return 1 if any(fresh.values()) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
