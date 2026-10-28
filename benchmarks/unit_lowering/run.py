"""`verity bench unit_lowering`: what writing the flat circuit file of PoUW's H1 tile unit, or of an R x C leaf's lane and
combine statements, costs, and the file it writes.

    run.py --k K --tile R,C [--unit tile|lane|combine] [--form flat|structural] --scratch DIR --out DIR [--guard [all]]
           [--tree PATH] [--staging fast|library]
    run.py --guard [all] --scratch DIR --out DIR [--tree PATH]
    run.py --pin-reference --scratch DIR --out DIR

The tile's lowering is `pouw_h1_circuits --lower K R C BIND PATH` (integrations/vllm), the auditor's lowering whose file a PoUW
period-open commits as ``k<K>``, bound to `BIND`. A lane's or the combine's is h1_lanes' staging of that statement of the R x C
leaf (`lanes.py`), bound to the Program a served window's verifier derives for it. Each runs in a child process with
``HZ_STAGE`` (``fast`` by default) and ``H1_TMPL`` unset for the flat form or ``structural``. Its peak RSS is the child's
``ru_maxrss`` (`os.wait4`, KiB on Linux) and its wall time the child's. The file is then read once for its SHA-512, bytes,
META's ``k_log`` and AND rows (`scan`), and deleted.

``--k`` writes OUT/unit_lowering.json and prints one ``UNIT-LOWERING`` line. ``--guard`` lowers each shape of `PINS` (``--guard
all``: and of `WIDE`) flat, holds its SHA-512 to `REFERENCE`'s and its ``k_log`` to C-Flock's 2^`K_MAX` block, writing
OUT/guard.json and exiting 1 naming each shape that fails. ``--pin-reference`` writes OUT/reference.json from this tree's
lowering at `PINS` and `WIDE`, the guard's reference when this tree is the one pinned. ``--tree`` runs another checkout's
lowering: its integrations/vllm and h1_lanes, its import roots first on the child's PYTHONPATH
(`research.pythonpath.tree_pythonpath`), and its .venv's Python where it has one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REFERENCE = HERE / "reference.json"
#: the bind every tile file here is bound to; a period-open's (its Program's and partition's digests) moves META's bytes only
BIND = {"partition": "0" * 128, "program": "0" * 128}
#: the statements: the H1 tile unit, and a leaf's lane (all its lanes one circuit) and combine (h1_lanes)
UNITS = ("tile", "lane", "combine")
#: the guard's shapes (unit, K, R, C), about 10 min and 12 GiB together. A tile or leaf is R = 4 i rows by C = 8 j columns, i j
#: lanes (`h1tile.Layout.lanes`, `leaf_lanes.lanes`), at K a multiple of 32 from 64; the combine takes K >= 384
#: (`h1zy.combine`), and 448 is h1_lanes' default
PINS = (("tile", 64, 4, 8), ("tile", 256, 8, 8), ("lane", 448, 32, 32), ("combine", 448, 32, 32))
#: pinned as well, and guarded only under ``--guard all``: a long K, and the leaf's own tile (40 and 56 GiB)
WIDE = (("tile", 2048, 4, 8), ("tile", 64, 32, 32))
#: C-Flock's largest block, 2^K_MAX bits (`verity_flock.circuit.K_MAX`): no file past it can be proved
K_MAX = 27
#: ``H1_TMPL`` for each form (`h1unit.layout`)
FORMS = {"flat": None, "structural": "structural"}
#: ``flock-ir-unit/v2``'s columns per word (`ir_lower.WORD`)
WORD = 128
PAD = b"0 0\n"
CHUNK = 1 << 26
GIB = 1 << 30


def commit(tree: Path) -> str:
    """The commit the lowering ran from: a shipped tree's `RESEARCH_SOURCE_SHA` (it has no .git), else git's HEAD, ``+dirty``
    with uncommitted changes."""
    if tree == ROOT and os.environ.get("RESEARCH_SOURCE_SHA"):
        return os.environ["RESEARCH_SOURCE_SHA"]
    try:
        head = subprocess.run(["git", "-C", str(tree), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(tree), "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                               text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return head + ("+dirty" if dirty else "")


def lower(k: int, r: int, c: int, path: Path, form: str = "flat", tree: Path = ROOT, staging: str = "fast",
          unit: str = "tile") -> tuple[int, float]:
    """Lower ``unit`` at (K, R, C) in ``form`` to ``path`` in a child process: its peak RSS in KiB and its wall seconds."""
    env = dict(os.environ, HZ_STAGE=staging)
    env.pop("H1_TMPL", None)
    if FORMS[form]:
        env["H1_TMPL"] = FORMS[form]
    python = sys.executable
    if tree != ROOT:
        from research.pythonpath import tree_pythonpath
        env["PYTHONPATH"] = ":".join(p for p in (tree_pythonpath(tree), env.get("PYTHONPATH")) if p)
        if (tree / ".venv" / "bin" / "python").exists():
            python = str(tree / ".venv" / "bin" / "python")
    if unit == "tile":
        argv, cwd = [python, "-m", "tests.protocols.pouw_h1_circuits", "--lower", str(k), str(r), str(c), json.dumps(BIND),
                     str(path)], tree / "integrations" / "vllm"
    else:
        argv, cwd = [python, str(HERE / "lanes.py"), unit, str(k), str(r), str(c), str(path), str(tree)], tree
    start = time.monotonic()
    child = subprocess.Popen(argv, cwd=cwd, env=env)
    _, status, usage = os.wait4(child.pid, 0)
    wall = time.monotonic() - start
    child.returncode = os.waitstatus_to_exitcode(status)
    if child.returncode:
        raise SystemExit(f"unit_lowering: the {unit} lowering at K={k} tile={r}x{c} form={form} exited {child.returncode} after "
                         f"{wall / 60:.1f} min at {usage.ru_maxrss / (1 << 20):.2f} GiB peak")
    return usage.ru_maxrss, wall


def _sha512(path: Path) -> tuple[str, int]:
    h, size = hashlib.sha512(), 0
    with open(path, "rb") as f:
        while b := f.read(CHUNK):
            h.update(b)
            size += len(b)
    return h.hexdigest(), size


def _is_assert(line: bytes, row: int, const: int) -> bool:
    """Whether ``line`` is an assertion's row, ``(L + z_j) * 1 = z_j``: B the constant column alone, and A holding the row's own
    column, which no AND row's A holds."""
    t = line.split()
    la = int(t[0])
    return t[la + 1:] == [b"1", b"%d" % const] and b"%d" % row in t[1:la + 1]


def _netlist(f, lines: int) -> dict:
    """A ``flock-ir-unit/v2`` section's rows and AND rows: the rows between its input groups and its output groups (`ir_lower.layout`:
    the ANDs and table reads, the assertions, then forced-zero rows up to the first output group), less forced-zero rows and
    assertions."""
    head = f.readline().split()
    useful, const, at, spans = int(head[2]), int(head[3]), 4, []
    for _ in range(2):
        groups, at = int(head[at]), at + 1
        cols = []
        for _ in range(groups):
            cols.append((int(head[at]), int(head[at + 1])))
            at += 3 + int(head[at + 2])
        spans.append(cols)
    if lines != useful + 1:
        raise ValueError(f"a section of {lines} lines holds a netlist of {useful} rows")
    lo = (spans[0][-1][0] + spans[0][-1][1]) * WORD if spans[0] else 0
    hi = spans[1][0][0] * WORD if spans[1] else const
    tail, pads, asserts = b" 1 %d\n" % const, 0, 0
    for _ in islice(f, lo):
        pass
    for row, line in enumerate(islice(f, hi - lo), lo):
        if line == PAD:
            pads += 1
        elif line.endswith(tail) and _is_assert(line, row, const):
            asserts += 1
    for _ in islice(f, useful - hi):
        pass
    return {"rows": useful, "ands": hi - lo - pads - asserts, "asserts": asserts}


def scan(path: Path) -> dict:
    """A circuit file's SHA-512, bytes, META's ``k_log`` (an instance's block, 2^k_log bits) and each ``CIRCUIT`` section's rows
    and AND rows (`_netlist`); ``ands`` the unit section's, None for a templated file, whose unit is META's template and no
    section."""
    with ThreadPoolExecutor(1) as pool:
        digest = pool.submit(_sha512, path)
        sections = {}
        with open(path, "rb") as f:
            first = f.readline()
            tag, meta = f.readline().split(b" ", 1)
            if tag != b"META":
                raise ValueError(f"{path}: {tag[:80]!r} where META is")
            k_log = json.loads(meta).get("k_log")
            for line in f:
                tag, name, lines = line.split()
                if tag != b"CIRCUIT":
                    raise ValueError(f"{path}: {line[:80]!r} where a CIRCUIT section starts")
                sections[name.decode()] = _netlist(f, int(lines))
        sha512, size = digest.result()
    unit = sections.get("unit")
    return {"sha512": sha512, "bytes": size, "format": first.split()[0].decode(), "k_log": k_log,
            "ands": unit["ands"] if unit else None, "sections": sections}


def measure(k: int, r: int, c: int, scratch: Path, form: str = "flat", tree: Path = ROOT, staging: str = "fast",
            keep: bool = False, unit: str = "tile") -> dict:
    """Lower ``unit`` at (K, R, C) under ``scratch`` and read the file: the benchmark's record."""
    d = scratch / f"{unit}-k{k}-{r}x{c}-{form}"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"k{k}.circuit.txt"
    kib, wall = lower(k, r, c, path, form, tree, staging, unit)
    doc = {"unit": unit, "k": k, "tile": [r, c], "form": form, "staging": staging, "peak_gib": round(kib / (1 << 20), 3),
           "wall_min": round(wall / 60, 3), **scan(path), "peak_kib": kib, "wall_s": round(wall, 2), "commit": commit(tree),
           "bind": BIND if unit == "tile" else "derived"}
    if not keep:
        path.unlink()
    return doc


def line(doc: dict) -> str:
    return (f"UNIT-LOWERING unit={doc['unit']} k={doc['k']} tile={doc['tile'][0]}x{doc['tile'][1]} form={doc['form']} "
            f"peak_gib={doc['peak_gib']} wall_min={doc['wall_min']} ands={doc['ands']} bytes={doc['bytes']} "
            f"k_log={doc['k_log']} sha512={doc['sha512'][:16]}")


def _key(s: dict) -> tuple:
    return (s.get("unit", "tile"), s["k"], *s["tile"])


def guard(scratch: Path, tree: Path = ROOT, staging: str = "fast", shapes: list[tuple] | None = None) -> dict:
    """Each shape of ``shapes`` (each ``(unit, K, R, C)``, by default `PINS`) lowered flat, its SHA-512 held to `REFERENCE`'s and
    its ``k_log`` to `K_MAX`: ``{guard, shapes}``, each shape with its ``reference`` and whether it ``matches``."""
    ref = json.loads(REFERENCE.read_text())
    pinned = {_key(s): s for s in ref["shapes"]}
    out = []
    for unit, k, r, c in shapes or PINS:
        if (unit, k, r, c) not in pinned:
            raise SystemExit(f"unit_lowering: {unit} K={k} tile={r}x{c} is not one of {REFERENCE.name}'s shapes")
        doc = measure(k, r, c, scratch, tree=tree, staging=staging, unit=unit)
        want = pinned[(unit, k, r, c)]["sha512"]
        ok = doc["sha512"] == want and doc["k_log"] is not None and doc["k_log"] <= K_MAX
        out.append({**doc, "reference": want, "matches": ok})
        print(f"UNIT-LOWERING-GUARD unit={unit} k={k} tile={r}x{c} sha512={doc['sha512'][:16]} reference={want[:16]} "
              f"k_log={doc['k_log']} {'pass' if ok else 'FAIL'}", flush=True)
    return {"guard": "pass" if all(s["matches"] for s in out) else "fail", "reference_commit": ref["commit"], "shapes": out}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--k", type=int)
    ap.add_argument("--tile", type=lambda v: tuple(int(x) for x in v.split(",")), metavar="R,C")
    ap.add_argument("--unit", choices=UNITS, default="tile")
    ap.add_argument("--form", choices=sorted(FORMS), default="flat")
    ap.add_argument("--scratch", type=Path, required=True, help="where the circuit file is written (deleted once read)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--guard", nargs="?", const="pins", choices=("pins", "all"),
                    help=f"hold the flat file's SHA-512 to {REFERENCE.name}'s at PINS {PINS}, or with 'all' at WIDE's too")
    ap.add_argument("--pin-reference", action="store_true", help=f"write OUT/{REFERENCE.name} from this tree at PINS and WIDE")
    ap.add_argument("--tree", type=Path, default=ROOT, help="the checkout whose lowering runs (default: this one)")
    ap.add_argument("--staging", choices=("fast", "library"), default="fast", help="HZ_STAGE: fast_stage's patches, or none")
    ap.add_argument("--keep", action="store_true", help="keep the circuit file under --scratch")
    a = ap.parse_args(argv)
    if not (a.k or a.guard or a.pin_reference):
        ap.error("--k, --guard or --pin-reference")
    if a.k and not a.tile:
        ap.error("--k needs --tile R,C")
    tree = a.tree.resolve()
    a.out.mkdir(parents=True, exist_ok=True)
    rc = 0
    if a.pin_reference:
        shapes = []
        for unit, k, r, c in PINS + WIDE:
            doc = measure(k, r, c, a.scratch, tree=tree, staging=a.staging, unit=unit)
            print(line(doc), flush=True)
            shapes.append({x: doc[x] for x in ("unit", "k", "tile", "sha512", "bytes", "ands", "k_log", "peak_gib", "wall_min")})
        (a.out / REFERENCE.name).write_text(json.dumps({"commit": commit(tree), "bind": BIND, "shapes": shapes}, indent=1) + "\n")
    if a.guard:
        g = guard(a.scratch, tree, a.staging, list(PINS + WIDE) if a.guard == "all" else None)
        (a.out / "guard.json").write_text(json.dumps(g, indent=1) + "\n")
        if g["guard"] != "pass":
            bad = [f"{s['unit']} K={s['k']} tile={s['tile'][0]}x{s['tile'][1]}" for s in g["shapes"] if not s["matches"]]
            print(f"unit_lowering: guard FAILED: the flat file's SHA-512 is not the reference's, or its k_log is past {K_MAX}, "
                  f"at {', '.join(bad)}", file=sys.stderr)
            rc = 1
    if a.k:
        doc = measure(a.k, *a.tile, a.scratch, a.form, tree, a.staging, a.keep, a.unit)
        if a.guard:
            doc["guard"] = g["guard"]
        (a.out / "unit_lowering.json").write_text(json.dumps(doc, indent=1) + "\n")
        print(line(doc), flush=True)
        if doc["k_log"] is not None and doc["k_log"] > K_MAX:
            print(f"unit_lowering: k_log {doc['k_log']} is past C-Flock's 2^{K_MAX} block: no proof of this file", file=sys.stderr)
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
