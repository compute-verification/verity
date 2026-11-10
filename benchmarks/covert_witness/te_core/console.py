"""The panels cut to the console's panels route's limits, and one panel's PUT.

    python -m te_core.console DIR    # DIR: te_core.plots' panels; writes DIR/console/

A body is at most `BODY_MAX` bytes (compact JSON, no NaN), a note at most `NOTE_MAX` characters, a title at most
`TITLE_MAX` and the columns at most `COLUMNS_MAX` distinct names (`refused` has the rest of what the route refuses). Each panel keeps
its first cut that fits (`CUTS`, in order, the first the whole panel) with numbers to 4 significant digits (losses 6),
and its note is the panel's short note (`SHORT`), the cut's, and `COMMON`; the uncut panel is DIR/<name>.json beside
it. The console's training panels are `training/algorithm-policy` and `training/frontier-fit` (te_core.algorithm_policy
PUTs both with `put`), `training/standard-workloads` (islands/standard-workloads/standard_workloads.py's) and
`training/size-scaling` (islands/ladder-v3/size_scaling.py's): te_core.plots' five panels are off it, so nothing here
PUTs them. `put` PUTs one panel to `PANELS_URL`/<id> with a Cursor OIDC token (research.slack's `oidc_token`, imported
only there) and returns its HTTP status; `served` GETs the panels the route serves for a producer.

A number reaches the console only as a te_core VERSION, so `put` refuses before any request (`refusal`):
- an id not in `PANELS`, or one `RETIRED` names;
- a te_core object that a loaded te_core module binds at its top level and that isn't the one its source binds there
  (`replaced`): a function, a class or a method its body defines, a name it imports from te_core, or a constant bound
  to a string, number, bool or tuple of them, so no wrapper, swapped function or changed setting publishes;
- a trace the ledger shares among its records changed in place (te_core.ledger's `changed_traces`);
- a te_core tree with uncommitted changes or untracked files, or off `BRANCH`, where the integrator merges te_core: a
  file of te_core at the branch's head (fetched from origin) that this tree's HEAD lacks or has otherwise
  (`off_branch`). A file only this tree has is its publisher's own module (standard-workloads' panel module).
"""
from __future__ import annotations

import argparse
import ast
import functools
import inspect
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from .plots import NAMES, frame

PANELS_URL = "https://website-docs-sage.vercel.app/api/panels"
#: the panels `put` PUTs: te_core.algorithm_policy's two, islands/standard-workloads/standard_workloads.py's and
#: islands/ladder-v3/size_scaling.py's
PANELS = ("training/algorithm-policy", "training/frontier-fit", "training/standard-workloads", "training/size-scaling")
#: the panel ids `put` refuses, each with why
RETIRED = {"training/frontier-branches": "frontier-branches is stored only, never PUT (te_core.panels)"} | {
    f"training/{n}": "te_core.plots' panels are off the console" for n in NAMES}
#: the branch whose head's te_core every published panel is built on
BRANCH = "cursor/training-replot-55a1"
HERE = Path(__file__).resolve().parent
BODY_MAX = 65536
NOTE_MAX = 500
#: the route's cell limit, live since the dashboard's 05:05Z change
CELL_MAX = 4096
COLUMNS_MAX, NAME_MAX, SOURCE_MAX, TITLE_MAX = 24, 60, 300, 120
REF_TARGET = -1.0
BAND_FEW = ("c1e15/k1e5", "c1e15/k1e6", "c1e14/k1e4")
LOSSES = ("loss", "loss_lo", "loss_hi", "l0")
COMMON = ("Formalism v2: a frozen tail always charged. basis: fit = rho c/(c-d) by Fit; target = Slowdown's bound, "
          "unproved, from A1/A1' upper bounds. Delta, u(1), a: witness upper bounds; tau a stub. One seed. Uncut: the "
          "panel's source.")
SHORT = {
    "p1-loss-vs-compute": "Loss (val bpb) against compute after theta over every checkpoint of every valid v2.3 record: "
                          "honest |C| (gates), covert |C| rho at c=4096w k=10kB, covert-policy at other (c, k).",
    "p2-slowdown-vs-gain": "S = W*/H at c=4096w k=10kB per start and target, H and W* over every checkpoint (S >= 1); "
                           "x0, s_star: the claim's bounds; *_known: the attack side (tightness). g null until F.",
    "p3-policy-plane": "S = W*/H over te_core.policy's (c, k) grid and each start's marked shapes, k in bits, H and W* "
                       "over every checkpoint of every valid v2.3 record.",
    "p4-decomposition": "Per algorithm, its least-|C|rho checkpoint per target at c=4096w k=10kB: p = |C|/H, s = rho, "
                        "S = p s; x0, s_star bounds, *_known the attack side.",
    "a2-efficiency-floor": "Least compute to each target among checkpoints with Delta <= delta_bits; fit_*_k_bits: the "
                           "least k where Fit holds at c=4096w (running theta', training).",
}


def body(p: dict) -> bytes:
    return json.dumps(p, separators=(",", ":"), allow_nan=False, ensure_ascii=False).encode()


def refused(p: dict) -> list[str]:
    """What the route refuses in `p`: its title is 1 to `TITLE_MAX` characters and its source 1 to `SOURCE_MAX`, its
    columns are 1 to `COLUMNS_MAX` distinct names of at most `NAME_MAX` characters, and a row has one cell per column,
    each a number, a string of at most `CELL_MAX` characters, or null (a boolean is none of them)."""
    def ok(v):
        return v is None or isinstance(v, (int, float)) and not isinstance(v, bool) or \
            isinstance(v, str) and len(v) <= CELL_MAX
    cols, out, source, title = p["columns"], [], p.get("source") or "", p.get("title") or ""
    if not 1 <= len(title) <= TITLE_MAX:
        out.append(f"its title is {len(title)} characters, not 1 to {TITLE_MAX}")
    if not 1 <= len(source) <= SOURCE_MAX:
        out.append(f"its source is {len(source)} characters, not 1 to {SOURCE_MAX}")
    if not 1 <= len(cols) <= COLUMNS_MAX or len(set(cols)) < len(cols) or any(len(c) > NAME_MAX for c in cols):
        out.append(f"its {len(cols)} columns aren't 1 to {COLUMNS_MAX} distinct names of at most {NAME_MAX} "
                   "characters")
    for i, r in enumerate(p["rows"]):
        if len(r) != len(cols):
            out.append(f"row {i} has {len(r)} cells, not {len(cols)}")
        out += [f"row {i}'s {c} is {v!r:.40}" for c, v in zip(cols, r) if not ok(v)]
    return out


def compact(p: dict, keep=None, drop: tuple = (), pick=None) -> dict:
    """`p` with the rows `keep` takes, of those the ones `pick` picks, without the columns `drop`, numbers to 4
    significant digits (losses 6)."""
    cols = [c for c in p["columns"] if c not in drop]

    def val(c, v):
        if isinstance(v, float):
            return round(v, 6) if c in LOSSES else float(f"{v:.4g}")
        return v
    ds = [d for d in frame(p) if keep is None or keep(d)]
    ds = pick(ds) if pick else ds
    return p | {"columns": cols, "rows": [[val(c, d[c]) for c in cols] for d in ds]}


def thinned(step: float):
    """A pick: each P1 curve at the points that lower its loss by `step` or reach a deeper target, its ends kept."""
    def pick(ds: list[dict]) -> list[dict]:
        groups: dict[tuple, list] = defaultdict(list)
        for d in ds:
            groups[(d["start_id"], d["curve"], d["policy"])].append(d)
        out = []
        for g in groups.values():
            kept = [g[0]]
            for i, d in enumerate(g[1:], 1):
                if i == len(g) - 1 or d["target_pct"] != kept[-1]["target_pct"] or \
                        d["loss"] <= kept[-1]["loss"] * (1 - step):
                    kept.append(d)
            out += kept
        return out
    return pick


def front(x: str, y: str):
    """A pick: per start and target, the rows no other beats in both `x` and `y` (smaller is better), in order."""
    def pick(ds: list[dict]) -> list[dict]:
        groups: dict[tuple, list] = defaultdict(list)
        for i, d in enumerate(ds):
            groups[(d["start_id"], d["target_pct"])].append(i)
        kept = set()
        for ix in groups.values():
            best = float("inf")
            for i in sorted(ix, key=lambda i: (ds[i][x], ds[i][y])):
                if ds[i][y] is not None and ds[i][y] < best:
                    best = ds[i][y]
                    kept.add(i)
        return [d for i, d in enumerate(ds) if i in kept]
    return pick


def band(policies: tuple):
    return lambda d: d["curve"] != "covert-policy" or d["policy"] in policies


def ref(d: dict) -> bool:
    return d["target_pct"] == REF_TARGET


LEAN = ("loss_lo", "loss_hi")
CUTS = {
    "p1-loss-vs-compute": [
        ("", None, LEAN, None),
        (" Points 0.25% apart in loss.", None, LEAN, thinned(0.0025)),
        (f" Band: {', '.join(BAND_FEW)}; points 0.25% apart.", band(BAND_FEW), LEAN, thinned(0.0025)),
        (" No band; points 0.25% apart.", band(()), LEAN, thinned(0.0025)),
        (" No band; points 0.5% apart.", band(()), LEAN, thinned(0.005)),
        (" No band; points 1% apart.", band(()), (*LEAN, "start", "checkpoint"), thinned(0.01))],
    "p3-policy-plane": [
        ("", None, (), None),
        (" Here -1% only.", ref, (), None),
        (" Here -1% only, no record, g, k_bytes or p (= S/rho).", ref, ("record", "g", "k_bytes", "p"), None),
        (" Here -1% only, no record, g, k_bytes, p (= S/rho) or checkpoint.", ref,
         ("record", "g", "k_bytes", "p", "checkpoint"), None),
        (" Here -1% only, no record, g, k_bytes, p (= S/rho), checkpoint, method or c in tokens.", ref,
         ("record", "g", "k_bytes", "p", "checkpoint", "method", "c_tokens_per_block"), None)],
    "p4-decomposition": [
        ("", None, (), None),
        (" Per start and target, the (p, s) front only.", None, (), front("p", "s"))],
    "a2-efficiency-floor": [("", None, (), None)],
    "p2-slowdown-vs-gain": [("", None, (), None)],
}


class NoCut(ValueError):
    """No cut of a panel fits the route's limits."""


def cut(p: dict) -> dict:
    """`p` cut to the route's limits: its first cut (`CUTS`) whose body fits, with the console's note."""
    name = p["id"].split("/", 1)[1]
    for why, keep, drop, pick in CUTS[name]:
        q = compact(p, keep, drop, pick)
        note = f"{SHORT[name]}{why} {COMMON}"
        q["note"] = note if len(note) <= NOTE_MAX else note[:NOTE_MAX - 1] + "…"
        if len(body(q)) <= BODY_MAX:
            return q
    raise NoCut(f"{p['id']}: no cut fits {BODY_MAX} bytes (the last is {len(body(q))})")


def write(src: Path, out: Path) -> dict[str, dict]:
    """Each of `src`'s panels (te_core.plots' DIR/<name>.json) cut, written to `out`/<name>.json. None of them is on the
    console, so a panel no cut fits is left out, with why on stderr, rather than failing the re-score that writes them."""
    out.mkdir(parents=True, exist_ok=True)
    made = {}
    for name in NAMES:
        p = json.loads((src / f"{name}.json").read_text())
        try:
            made[name] = cut(p)
        except NoCut as e:
            print(f"te_core.console: {e}, so {out / name}.json is left out", file=sys.stderr)
            continue
        (out / f"{name}.json").write_bytes(body(made[name]))
    return made


def _constant(v) -> bool:
    return isinstance(v, (str, int, float, bytes)) or isinstance(v, tuple) and all(map(_constant, v))


_STDLIB = Path(inspect.__file__).resolve().parent


def _library(obj) -> bool:
    """Whether `obj`, a decorator's product, is installed code's: a function whose code is the standard library's or
    an installed package's (torch wraps a dispatch mode's `__torch_dispatch__`), or an object of a type the standard
    library defines (functools.lru_cache's wrapper), a partial aside."""
    if inspect.isfunction(obj):
        path = Path(obj.__code__.co_filename).resolve()
        return path.is_relative_to(_STDLIB) or bool({"site-packages", "dist-packages"} & set(path.parts))
    return type(obj).__module__.split(".")[0] in sys.stdlib_module_names and not isinstance(obj, functools.partial)


def _defines(obj, file: Path, name: str) -> bool:
    """Whether `obj` is the function `file` defines as `name`: that function, or an installed decorator's product
    wrapping it (`__wrapped__`), as a contextmanager or an lru_cache is. A method is read through its staticmethod,
    classmethod or property."""
    if isinstance(obj, (staticmethod, classmethod)):
        obj = obj.__func__
    elif isinstance(obj, property):
        obj = obj.fget
    try:
        inner = inspect.unwrap(obj)
    except ValueError:
        return False
    return (inspect.isfunction(inner) and Path(inner.__code__.co_filename).resolve() == file
            and inner.__code__.co_name == name and (inner is obj or _library(obj)))


def _bound(mod) -> dict[str, object]:
    """What `mod`'s source binds at its top level, by name, where `replaced` can check it: a def or class's ast node,
    ("import", module, name) for a name from te_core, ("constant", value) for a `_constant` literal of an upper-case
    name. A later top-level assignment to a name leaves it out."""
    spec = mod.__spec__
    pkg = spec.name if spec.submodule_search_locations else spec.name.rpartition(".")[0]
    out: dict[str, object] = {}
    for s in ast.parse(Path(mod.__file__).read_text()).body:
        if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out[s.name] = s
        elif isinstance(s, ast.ImportFrom):
            src = s.module or ""
            if s.level:
                base = ".".join(pkg.split(".")[:len(pkg.split(".")) - s.level + 1])
                src = f"{base}.{s.module}" if s.module else base
            for a in s.names:
                out.pop(a.asname or a.name, None)
                if src.split(".")[0] == "te_core":
                    out[a.asname or a.name] = ("import", src, a.name)
        elif isinstance(s, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = s.targets if isinstance(s, ast.Assign) else [s.target]
            names = [t.id for t in targets if isinstance(t, ast.Name)]
            for n in names:
                out.pop(n, None)
            if not isinstance(s, ast.AugAssign) and s.value is not None and len(targets) == len(names) == 1 \
                    and names[0].isupper():
                try:
                    v = ast.literal_eval(s.value)
                except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
                    continue
                if _constant(v):
                    out[names[0]] = ("constant", v)
    return out


def replaced() -> list[str]:
    """Each object a loaded te_core module (`python -m`'s ``__main__`` among them) binds at its top level that isn't
    the one its source binds (module docstring), and each such module loaded from outside this tree."""
    bad = []
    for name, mod in list(sys.modules.items()):
        spec = getattr(mod, "__spec__", None)
        if spec is None or spec.name.split(".")[0] != "te_core" or not getattr(mod, "__file__", None):
            continue
        file = Path(mod.__file__).resolve()
        if not file.is_relative_to(HERE):
            bad.append(f"{spec.name} is loaded from {file}, not {HERE}")
            continue
        for k, b in _bound(mod).items():
            got = getattr(mod, k, None)
            if isinstance(b, tuple) and b[0] == "constant":
                if type(got) is not type(b[1]) or got != b[1]:
                    bad.append(f"{spec.name}.{k} is {got!r:.60}, its source's {b[1]!r:.60}")
            elif isinstance(b, tuple):
                src = sys.modules.get(b[1])
                if src is not None and got is not getattr(src, b[2], sys.modules.get(f"{b[1]}.{b[2]}")):
                    bad.append(f"{spec.name}.{k} isn't {b[1]}.{b[2]}")
            elif isinstance(b, ast.ClassDef):
                if not isinstance(got, type) or got.__module__ != name or got.__qualname__ != k:
                    bad.append(f"{spec.name}.{k} isn't the class its source defines")
                    continue
                bad += [f"{spec.name}.{k}.{f.name} isn't the method its source defines" for f in b.body
                        if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and not _defines(vars(got).get(f.name), file, f.name)]
            elif not _defines(got, file, k):
                bad.append(f"{spec.name}.{k} isn't the function its source defines")
    return bad


def _git(here: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(here), *args], capture_output=True, text=True, check=False)
    if r.returncode:
        raise RuntimeError(f"git {' '.join(args)}: {(r.stderr or r.stdout).strip()[:300]}")
    return r.stdout


def off_branch(ref: str, here: Path = HERE) -> list[str]:
    """Why the te_core tree at `here` isn't `ref`'s (module docstring): its uncommitted changes and untracked files,
    and each file of `here` at `ref` that HEAD lacks or has otherwise."""
    dirty = [line[3:] for line in _git(here, "status", "--porcelain", "--untracked-files=all", "--", ".").splitlines()]
    prefix = _git(here, "rev-parse", "--show-prefix").strip().rstrip("/")

    def files(rev: str) -> dict[str, str]:
        listed = _git(here, "ls-tree", "-r", "--full-tree", f"{rev}:{prefix}").splitlines()
        return {path: meta.split()[2] for meta, path in (line.split("\t", 1) for line in listed)}

    mine, theirs = files("HEAD"), files(ref)
    differ = sorted(f for f, blob in theirs.items() if mine.get(f) != blob)
    head = _git(here, "rev-parse", "--short=10", ref).strip()
    out = [f"uncommitted changes or untracked files in {prefix}: {', '.join(dirty[:4])}"
           f"{f' and {len(dirty) - 4} more' if len(dirty) > 4 else ''}"] if dirty else []
    if differ:
        out.append(f"{len(differ)} files of {prefix} at {ref} ({head}) differ here or are missing: "
                   f"{', '.join(differ[:4])}{' ...' if len(differ) > 4 else ''}")
    return out


def refusal(p: dict) -> str | None:
    """Why `put` refuses `p` (module docstring), or None. The tree is checked last, against `BRANCH`'s head fetched
    now."""
    from . import ledger

    pid = p.get("id")
    if pid in RETIRED:
        return f"retired: {RETIRED[pid]}"
    if pid not in PANELS:
        return (f"not a panel te_core PUTs ({', '.join(PANELS)}); a new one is added to console.PANELS in a te_core "
                f"VERSION")
    if changed := ledger.changed_traces():
        return (f"{changed} of the ledger's shared traces changed in place after it read them, so every record sharing "
                f"them changed and this panel doesn't hold; find the reader that writes into a record's "
                f"formalism.trace and make it copy first")
    if bad := replaced():
        return (f"{len(bad)} te_core objects aren't their source's ({'; '.join(bad[:4])}): a ruling reaches a published "
                f"number only as a te_core VERSION, so run its builder unwrapped, as `python -m te_core.MODULE`")
    try:
        _git(HERE, "fetch", "--quiet", "origin", f"+refs/heads/{BRANCH}:refs/remotes/origin/{BRANCH}")
        off = off_branch(f"refs/remotes/origin/{BRANCH}")
    except (OSError, RuntimeError) as e:
        return f"te_core's tree can't be held to {BRANCH}'s head ({e}); publish from a git checkout with origin"
    if off:
        return f"{'; '.join(off)}: publish from a clean checkout of {BRANCH}'s head (git fetch origin {BRANCH})"
    return None


def put(p: dict) -> str:
    """PUT `p` (module docstring), refused before any request where `refusal` gives a reason."""
    import urllib.error
    import urllib.request

    why = refusal(p)
    if why:
        raise SystemExit(f"console.put {p.get('id')}: {why}")
    from research.slack import oidc_token

    req = urllib.request.Request(f"{PANELS_URL}/{p['id']}", body(p), method="PUT",  # noqa: S310 (a fixed https URL)
                                 headers={"Authorization": f"Bearer {oidc_token(PANELS_URL)}",
                                          "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310
            return str(r.status)
    except urllib.error.HTTPError as e:
        return f"{e.code} {e.read()[:400].decode(errors='replace')}"


def served(producer: str = "training") -> dict[str, dict]:
    """The panels the route serves for `producer`, by id (GET `PANELS_URL`?producer=, the same token as `put`)."""
    import urllib.request

    from research.slack import oidc_token
    req = urllib.request.Request(f"{PANELS_URL}?producer={producer}",  # noqa: S310 (a fixed https URL)
                                 headers={"Authorization": f"Bearer {oidc_token(PANELS_URL)}"})
    with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310
        return {x["panel"]["id"]: x["panel"] for x in json.loads(r.read())["panels"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dir", type=Path, help="te_core.plots' panels")
    a = ap.parse_args(argv)
    for p in write(a.dir, a.dir / "console").values():
        print(f"{p['id']}: {len(body(p))} bytes, {len(p['rows'])} rows, note {len(p['note'])} characters")
    return 0


if __name__ == "__main__":
    sys.exit(main())
