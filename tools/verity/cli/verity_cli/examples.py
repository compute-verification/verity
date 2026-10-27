"""The ladder: each example's level of doneness, from its `examples/<name>/example.toml` (`examples/README.md`).

An example names its protocol (or a composite, or a list), its workload and its placement (role -> machine class), the level
it has reached, and the evidence for that level and every one below it. `problems` says what a claimed level lacks, in the
form that can be checked offline: run-id format, a guarantee listed in a `lean-audit.json`, declaration names found in the
tracked `.lean` files. `tests/test_examples.py` fails an example with any, and `verity example --status` prints every
example with what its next level needs.
"""
from __future__ import annotations

import json
import re
import subprocess
import tomllib
from dataclasses import dataclass
from functools import cache
from pathlib import Path

LEVELS = ("moved", "runs", "placed", "stated", "proved", "measured")
#: the roles a placement names -> whose they are; the auditor's run only on an `auditor-host` (AGENTS.md "Auditor-trusted
#: devices"). A new example's role goes here with its side.
ROLES = {"prover": "prover", "endpoint": "prover", "store": "prover",
         "verifier": "auditor", "challenger": "auditor", "certifier": "auditor"}
#: machine classes: a node the prover runs (`gpu-node`, `cpu-node`), a machine it doesn't control (`auditor-host`), or one
#: CPU machine holding every role (`local`: a laptop or a suite)
MACHINES = ("gpu-node", "cpu-node", "auditor-host", "local")
RUN = re.compile(r"r\d{8}-\d{6}-[0-9a-f]{4}|art:[0-9a-f]{64}")
NAME = re.compile(r"[a-z0-9_]+")
PROOF = re.compile(r"\w+(\.\w+)*\.SecurityProofs\.\w+(\.\w+)*")
KEYS = {"protocol", "workload", "level", "placement", "evidence"}
EVIDENCE = {"runs", "tests", "guarantee", "honest", "nontrivial", "proof", "premises"}
SCHEMA = "examples/README.md has the schema"
WHERE = "tools/verity/cli/verity_cli/examples.py"
DECL = re.compile(r"^[ \t]*(?:@\[[^\]\n]*\][ \t]*\n?[ \t]*)?(?:(?:private|protected|noncomputable|nonrec|partial|unsafe)[ \t]+)*"
                  r"(?:theorem|lemma|def|abbrev|instance|structure|inductive|class|opaque)[ \t]+([^\s:({\[]+)", re.M)
NAMESPACE = re.compile(r"^[ \t]*namespace[ \t]+(\S+)", re.M)


@dataclass(frozen=True)
class Example:
    name: str
    data: dict | None       # example.toml, or None when it is missing or doesn't parse (`error`)
    error: str = ""
    runner: str = ""        # run.py or run.sh, what `verity example NAME` runs


def examples(root: Path) -> list[Example]:
    out = []
    for d in sorted(p for p in (root / "examples").iterdir() if p.is_dir() and not p.name.startswith((".", "_"))):
        runner = next((r for r in ("run.py", "run.sh") if (d / r).is_file()), "")
        f = d / "example.toml"
        if not f.is_file():
            out.append(Example(d.name, None, f"no example.toml: add one ({SCHEMA})", runner))
            continue
        try:
            out.append(Example(d.name, tomllib.loads(f.read_text()), "", runner))
        except tomllib.TOMLDecodeError as e:
            out.append(Example(d.name, None, f"example.toml doesn't parse: {e}", runner))
    return out


def _git(root: Path, *argv: str) -> list[str]:
    p = subprocess.run(["git", *argv], cwd=root, capture_output=True, text=True)
    if p.returncode not in (0, 1):      # 1: git grep found nothing
        raise RuntimeError(f"git {' '.join(argv[:2])} in {root} failed: {p.stderr.strip()} (the checks read a git checkout)")
    return [x for x in p.stdout.split("\0") if x]


@cache
def _locks(root: Path) -> tuple[frozenset[str], frozenset[str]]:
    """Every name a `lean-audit.json` lists as a guarantee (a proof, its statement, or a claim), and every assumption."""
    listed, assumptions = set(), set()
    for f in _git(root, "ls-files", "-z", "--", "lean-audit.json", "*/lean-audit.json"):
        if "/tests/" in f:
            continue
        d = json.loads((root / f).read_text())
        for name, rec in d.get("guarantees", {}).items():
            listed.add(name)
            for s in (rec or {}).get("signature", []):
                if " : " in s:
                    listed.add(re.sub(r"\.\{[^}]*\}$", "", s.split(" : ", 1)[1].split()[0]))
        listed |= set(d.get("claims", {}).get("witnesses", {}))
        assumptions |= set(d.get("assumptions", {}))
    return frozenset(listed), frozenset(assumptions)


def _within(prefix: list[str], namespaces: list[str]) -> bool:
    it = iter(namespaces)
    return all(c in it for c in prefix)


@cache
def declared(root: Path, name: str) -> bool:
    """Whether a tracked `.lean` file declares ``name``: a declaration of a dotted suffix of it, in a file whose `namespace`
    lines hold the rest in order."""
    for f in _git(root, "grep", "-z", "-l", "-F", name.rsplit(".", 1)[-1], "--", "*.lean"):
        text = (root / f).read_text(errors="replace")
        spaces = [c for m in NAMESPACE.finditer(text) for c in m.group(1).split(".")]
        for m in DECL.finditer(text):
            d = m.group(1)
            if d.startswith("_root_."):
                if d[len("_root_."):] == name:
                    return True
            elif name == d or name.endswith("." + d):
                rest = name[: -len(d) - 1] if name != d else ""
                if _within(rest.split(".") if rest else [], spaces):
                    return True
    return False


def _names(v) -> list[str]:
    return [v] if isinstance(v, str) else list(v) if isinstance(v, list) else []


def vocabulary(root: Path) -> tuple[list[str], list[str]]:
    """The known protocols (packages under verity/core/protocols/, the composites among them) and workloads."""
    def dirs(p: Path) -> list[str]:
        return sorted(d.name for d in p.iterdir() if d.is_dir() and not d.name.startswith((".", "_")) and d.name != "tests")
    return dirs(root / "verity" / "core" / "protocols"), [*dirs(root / "integrations"), "none"]


def schema(ex: Example, root: Path) -> list[str]:
    """What's wrong with ``ex``'s example.toml as a document, whatever its level."""
    if ex.data is None:
        return [ex.error]
    d, out = ex.data, []
    if not NAME.fullmatch(ex.name):
        out.append(f"name {ex.name!r}: rename the directory to lowercase letters, digits and _ (`verity example NAME` reads a - "
                   "as a _)")
    if extra := sorted(set(d) - KEYS):
        out.append(f"unknown keys {extra}: example.toml has only {sorted(KEYS)} ({SCHEMA})")
    protocols, workloads = vocabulary(root)
    p = d.get("protocol")
    ps = _names(p)
    if not ps or not all(isinstance(x, str) for x in ps) or (isinstance(p, list) and len(ps) < 2):
        out.append(f"protocol: give a protocol or composite ({', '.join(protocols)}) or a list of two or more")
    elif unknown := [x for x in ps if x not in protocols]:
        out.append(f"protocol {unknown} is no package under verity/core/protocols/: one of {', '.join(protocols)}")
    if d.get("workload") not in workloads:
        out.append(f"workload {d.get('workload')!r}: one of {', '.join(workloads)} (a directory under integrations/, or none)")
    lv = d.get("level")
    if type(lv) is not int or not 0 <= lv < len(LEVELS):
        out.append(f"level {lv!r}: an integer from 0 to {len(LEVELS) - 1} ({', '.join(f'{i} {n}' for i, n in enumerate(LEVELS))})")
    pl = d.get("placement")
    if not isinstance(pl, dict) or not pl:
        out.append("placement: add a [placement] table from each role to its machine class, e.g. prover = \"gpu-node\"")
    else:
        if bad := sorted(r for r in pl if r not in ROLES):
            out.append(f"placement roles {bad} aren't known ({', '.join(ROLES)}): add each to ROLES in {WHERE} with its side, "
                       "prover or auditor")
        if bad := sorted(r for r, m in pl.items() if m not in MACHINES):
            out.append(f"placement {', '.join(f'{r} = {pl[r]!r}' for r in bad)}: a machine class, one of {', '.join(MACHINES)}")
    ev = d.get("evidence", {})
    if not isinstance(ev, dict):
        out.append("evidence: a table ([evidence])")
    else:
        if extra := sorted(set(ev) - EVIDENCE):
            out.append(f"unknown evidence {extra}: [evidence] has only {sorted(EVIDENCE)} ({SCHEMA})")
        out += [f"evidence runs: {r!r} is neither a run id (rYYYYMMDD-HHMMSS-xxxx) nor an art: id"
                for r in _names(ev.get("runs")) if not RUN.fullmatch(str(r))]
    return out


def missing(ex: Example, level: int, root: Path) -> list[str]:
    """What ``ex`` lacks for ``level`` itself (not the levels below it), each saying what to add."""
    ev = ex.data.get("evidence", {}) if ex.data else {}
    ev = ev if isinstance(ev, dict) else {}
    head = f"level {level} ({LEVELS[level]})"
    runs, out = _names(ev.get("runs")), []
    if level == 1:
        tests = _names(ev.get("tests"))
        for t in tests:
            path, _, fn = str(t).partition("::")
            f = root / path
            if not fn or not f.is_file() or not re.search(rf"^\s*(?:async\s+)?def {re.escape(fn)}\(", f.read_text(), re.M):
                out.append(f"tests: {t!r} names no test; give PATH::NAME of a test in a suite that runs the example on CPU")
        if not runs and not tests:
            out.append(f"{head}: add a run of record to [evidence] runs, or a CPU test that runs the example to tests "
                       "(PATH::NAME)")
        return out
    if level == 2:
        if not runs:
            out.append(f"{head}: add a run of record on the declared placement to [evidence] runs")
        pl = ex.data.get("placement") or {}
        if off := sorted(r for r, m in pl.items() if ROLES.get(r) == "auditor" and m != "auditor-host"):
            out.append(f"{head}: the auditor's roles run off the prover's node: put "
                       f"{', '.join(f'{r} (now {pl[r]})' for r in off)} on auditor-host, and cite a run of that placement")
        if on := sorted(r for r, m in pl.items() if ROLES.get(r) == "prover" and m == "auditor-host"):
            out.append(f"{head}: the prover's roles {on} are on auditor-host, a machine the prover doesn't control")
        if not any(ROLES.get(r) == "auditor" for r in pl):
            out.append(f"{head}: place the auditor's roles (a verifier at least) in [placement]")
        return out
    if level == 3:
        listed, _ = _locks(root)
        g = ev.get("guarantee")
        if not isinstance(g, str) or not g:
            out.append(f"{head}: add [evidence] guarantee, the guarantee's statement as a lean-audit.json lists it")
        elif g not in listed:
            out.append(f"{head}: guarantee {g!r} is listed in no lean-audit.json (its proof's signature, a guarantees key or a "
                       "claim): list it with `tools/verity/lean/check.py --update`")
        for key, what in (("honest", "its hypotheses hold at the honest deployed instance"),
                          ("nontrivial", "its conclusion fails for a verifier that accepts everything")):
            names = _names(ev.get(key))
            if not names:
                out.append(f"{head}: add [evidence] {key}, the lemma that {what}")
            out += [f"{head}: {key} lemma {n!r} is declared in no tracked .lean file" for n in names if not declared(root, str(n))]
        return out
    if level == 4:
        p = ev.get("proof")
        if not isinstance(p, str) or not PROOF.fullmatch(p):
            out.append(f"{head}: add [evidence] proof, the guarantee's <Area>.SecurityProofs.G")
        elif not declared(root, p):
            out.append(f"{head}: proof {p!r} is declared in no tracked .lean file")
        return out
    if level == 5:
        _, assumptions = _locks(root)
        prem = ev.get("premises")
        if not isinstance(prem, dict) or not prem:
            out.append(f"{head}: add [evidence.premises], each premise of the guarantee -> the run that measured it")
            return out
        for k, v in prem.items():
            if k not in assumptions:
                out.append(f"{head}: premise {k!r} is an assumption no lean-audit.json lists")
            if not isinstance(v, str) or not RUN.fullmatch(v):
                out.append(f"{head}: premise {k!r}: {v!r} is neither a run id (rYYYYMMDD-HHMMSS-xxxx) nor an art: id")
        return out
    return out


def problems(ex: Example, root: Path) -> list[str]:
    """Everything ``ex`` lacks for the level it claims: the document's faults, then each level's evidence up to it."""
    out = schema(ex, root)
    lv = ex.data.get("level") if ex.data else None
    if out or type(lv) is not int:
        return out
    return [m for level in range(1, lv + 1) for m in missing(ex, level, root)]


def next_missing(ex: Example, root: Path) -> str:
    if p := problems(ex, root):
        return "BROKEN: " + p[0]
    lv = ex.data["level"]
    if lv + 1 >= len(LEVELS):
        return "-"
    return "; ".join(missing(ex, lv + 1, root)) or f"level {lv + 1} ({LEVELS[lv + 1]}): its evidence is there; raise level"


def status(root: Path) -> int:
    """Print every example's protocol, workload, level and next missing evidence; 1 when one lacks its claimed level's."""
    rows, broken = [], 0
    for ex in examples(root):
        d = ex.data or {}
        lv = d.get("level")
        nxt = next_missing(ex, root)
        broken += nxt.startswith("BROKEN")
        rows.append((ex.name, "+".join(map(str, _names(d.get("protocol")))) or "?", str(d.get("workload", "?")),
                     f"{lv} {LEVELS[lv]}" if type(lv) is int and 0 <= lv < len(LEVELS) else "?", ex.runner or "missing", nxt))
    head = ("example", "protocol", "workload", "level", "runner", "next missing evidence")
    w = [max(len(r[i]) for r in [head, *rows]) for i in range(5)]
    for r in [head, *rows]:
        print("  ".join(c.ljust(w[i]) for i, c in enumerate(r[:5])) + "  " + r[5])
    return 1 if broken else 0
