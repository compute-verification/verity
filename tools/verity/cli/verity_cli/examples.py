"""The ladder: each example's doneness on two independent axes, from its `examples/<name>/example.toml` (`examples/README.md`).

An example names its protocol (or a composite, or a list), its workload and its placement (role -> machine class), how far it
runs (`run`: moved, runs, placed) and how far its guarantee is proved (`proof`: none, stated, proved, measured), and the
evidence for each step up to those on each axis. `problems` says what a claimed step lacks, in the form that can be checked
offline: run-id format, a guarantee listed in a `lean-audit.json`, declaration names found in the tracked `.lean` files.
`tests/test_examples.py` fails an example with any, and `verity example --status` prints every example with what the next
step on each axis needs.
"""
from __future__ import annotations

import json
import re
import subprocess
import tomllib
from dataclasses import dataclass
from functools import cache
from pathlib import Path

#: each axis's steps in order; each axis needs the evidence of every step up to the one it claims, and none of the other's
AXES = {"run": ("moved", "runs", "placed"), "proof": ("none", "stated", "proved", "measured")}
#: the roles a placement names -> whose they are; the auditor's run only on an `auditor-host` (AGENTS.md "Auditor-trusted
#: devices"). A new example's role goes here with its side.
ROLES = {"prover": "prover", "endpoint": "prover", "store": "prover",
         "verifier": "auditor", "challenger": "auditor", "certifier": "auditor", "memory_challenger": "auditor"}
#: machine classes: a node the prover runs (`gpu-node`, `cpu-node`), a machine it doesn't control (`auditor-host`), or one
#: CPU machine holding every role (`local`: a laptop or a suite)
MACHINES = ("gpu-node", "cpu-node", "auditor-host", "local")
RUN = re.compile(r"r\d{8}-\d{6}-[0-9a-f]{4}|art:[0-9a-f]{64}")
NAME = re.compile(r"[a-z0-9_]+")
PROOF = re.compile(r"\w+(\.\w+)*\.SecurityProofs\.\w+(\.\w+)*")
KEYS = {"protocol", "workload", *AXES, "placement", "evidence"}
#: the retired single ladder's `level` -> (run, proof), read from a file that gives neither axis. Remove once no
#: example.toml says `level`.
LEVEL = {0: ("moved", "none"), 1: ("runs", "none"), 2: ("placed", "none"), 3: ("placed", "stated"), 4: ("placed", "proved"),
         5: ("placed", "measured")}
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
    runner: str = ""        # config.toml (`verity run`), run.py or run.sh, what `verity example NAME` runs


def examples(root: Path) -> list[Example]:
    out = []
    for d in sorted(p for p in (root / "examples").iterdir() if p.is_dir() and not p.name.startswith((".", "_"))):
        runner = next((r for r in ("config.toml", "run.py", "run.sh") if (d / r).is_file()), "")
        f = d / "example.toml"
        if not f.is_file():
            out.append(Example(d.name, None, f"no example.toml: add one ({SCHEMA})", runner))
            continue
        try:
            data = tomllib.loads(f.read_text())
            if data.get("level") in LEVEL and not set(AXES) & set(data):
                data = {k: v for k, v in data.items() if k != "level"} | dict(zip(AXES, LEVEL[data["level"]]))
            out.append(Example(d.name, data, "", runner))
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
    """What's wrong with ``ex``'s example.toml as a document, whatever it claims."""
    if ex.data is None:
        return [ex.error]
    d, out = ex.data, []
    if not NAME.fullmatch(ex.name):
        out.append(f"name {ex.name!r}: rename the directory to lowercase letters, digits and _ (`verity example NAME` reads a - "
                   "as a _)")
    if "level" in d:
        out.append("level is retired: give run (moved, runs, placed; the old 0-2) and proof (none, stated, proved, measured; "
                   "the old 3-5), each with its own evidence")
    if extra := sorted(set(d) - KEYS - {"level"}):
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
    for axis, steps in AXES.items():
        if axis not in d:
            out.append(f"add {axis}, one of {', '.join(steps)}")
        elif d[axis] not in steps:
            out.append(f"{axis} {d[axis]!r}: one of {', '.join(steps)}")
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


def missing(ex: Example, axis: str, step: str, root: Path) -> list[str]:
    """What ``ex`` lacks for ``step`` of ``axis`` itself (not the steps below it), each saying what to add."""
    ev = ex.data.get("evidence", {}) if ex.data else {}
    ev = ev if isinstance(ev, dict) else {}
    head = f"{axis} {step}"
    runs, out = _names(ev.get("runs")), []
    if step == "runs":
        tests = _names(ev.get("tests"))
        for t in tests:
            path, _, fn = str(t).partition("::")
            f = root / path
            if not fn or not f.is_file() or not re.search(rf"^\s*(?:async\s+)?def {re.escape(fn)}\(", f.read_text(), re.M):
                out.append(f"tests: {t!r} names no test; give PATH::NAME of a test in a suite that runs the example on CPU")
        if not runs and not tests:
            out.append(f"{head}: add a run of record to [evidence] runs, or a CPU test that runs the example to tests "
                       "(PATH::NAME)")
    elif step == "placed":
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
    elif step == "stated":
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
    elif step == "proved":
        p = ev.get("proof")
        if not isinstance(p, str) or not PROOF.fullmatch(p):
            out.append(f"{head}: add [evidence] proof, the guarantee's <Area>.SecurityProofs.G")
        elif not declared(root, p):
            out.append(f"{head}: proof {p!r} is declared in no tracked .lean file")
    elif step == "measured":
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


def problems(ex: Example, root: Path) -> list[str]:
    """Everything ``ex`` lacks for the steps it claims: the document's faults, then each axis's evidence up to its step."""
    if out := schema(ex, root):
        return out
    return [m for axis, steps in AXES.items() for step in steps[1:steps.index(ex.data[axis]) + 1]
            for m in missing(ex, axis, step, root)]


def next_missing(ex: Example, axis: str, root: Path) -> str:
    """What the step after ``ex``'s on ``axis`` needs, or why it has none."""
    steps = AXES[axis]
    i = steps.index(ex.data[axis])
    if i + 1 == len(steps):
        return f"{axis}: at its last step, {steps[i]}"
    head = f"{axis} {steps[i + 1]}"
    if not (lack := missing(ex, axis, steps[i + 1], root)):
        return f"{head}: its evidence is there; raise {axis} to {steps[i + 1]!r}"
    return f"{head}: " + "; ".join(m.removeprefix(f"{head}: ") for m in lack)


def status(root: Path) -> int:
    """Print every example's protocol, workload, both axes and each axis's next missing evidence; 1 when one lacks what it
    claims."""
    head = ("example", "protocol", "workload", *AXES, "runner")
    rows, notes, broken = [head], {}, 0
    for ex in examples(root):
        d = ex.data or {}
        rows.append((ex.name, "+".join(map(str, _names(d.get("protocol")))) or "?", str(d.get("workload", "?")),
                     *(str(d.get(a, "?")) for a in AXES), ex.runner or "missing"))
        if p := problems(ex, root):
            broken += 1
            notes[ex.name] = ["BROKEN: " + m for m in p]
        else:
            notes[ex.name] = ["next " + next_missing(ex, a, root) for a in AXES]
    w = [max(len(r[i]) for r in rows) for i in range(len(head))]
    for r in rows:
        print("  ".join(c.ljust(w[i]) for i, c in enumerate(r)).rstrip())
        for n in notes.get(r[0], []) if r is not head else []:
            print("    " + n)
    return 1 if broken else 0
