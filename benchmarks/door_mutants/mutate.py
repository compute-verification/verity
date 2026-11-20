"""Mutation run over the door's Python: one small edit at a time in the named functions, each run against the test files
that reach that function; KILLED if a test fails, SURVIVED if all pass, INVALID if the module no longer imports.

  python mutate.py gen  SPEC.json OUT/mutants.jsonl
  python mutate.py reach SPEC.json OUT/reach.json --jobs N        (baseline: each test file's time and reached functions)
  python mutate.py run  SPEC.json OUT --jobs N [--step NAME]      (needs OUT/mutants.jsonl and OUT/reach.json)
  python mutate.py table OUT

SPEC: {"steps": [{"name", "targets": [{"file", "funcs": [qualname, ...]}], "tests": [test file, ...]}]}; paths from the
repository root. Each worker is a copy of the tree, first on PYTHONPATH, so mutations never touch the checkout.
"""
from __future__ import annotations

import ast
import copy
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG_ROOTS = ["", "verity/protocols/proofs/flock/prover/python", "tools/verity/circuit_check/src", "tools/difftest/python"]
CAP = int(os.environ.get("MUT_CAP", "400"))


# ---------------------------------------------------------------------------------------------------------- generation

def _funcs(tree: ast.Module, names: list[str]) -> dict[str, ast.AST]:
    out: dict[str, ast.AST] = {}

    def walk(node, prefix):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                q = f"{prefix}{ch.name}"
                if not isinstance(ch, ast.ClassDef):
                    out[q] = ch
                walk(ch, q + ".")
    walk(tree, "")
    if names == ["*"]:
        return {q: f for q, f in out.items() if "." not in q or q.count(".") == 1}
    missing = [n for n in names if n not in out]
    if missing:
        raise SystemExit(f"no such function(s): {missing}")
    return {n: out[n] for n in names}


CMP = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq,
       ast.In: ast.NotIn, ast.NotIn: ast.In, ast.Is: ast.IsNot, ast.IsNot: ast.Is}
BIN = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.FloorDiv: ast.Mult, ast.Mult: ast.FloorDiv, ast.LShift: ast.RShift,
       ast.RShift: ast.LShift, ast.BitXor: ast.BitOr, ast.BitAnd: ast.BitOr, ast.BitOr: ast.BitAnd}
NONCOMM = (ast.Sub, ast.FloorDiv, ast.Mod, ast.LShift, ast.RShift, ast.Div)


def _seg(src_lines: list[str], node: ast.AST) -> tuple[int, int]:
    """Absolute [start, end) character offsets of ``node`` in the source."""
    starts = [0]
    for ln in src_lines:
        starts.append(starts[-1] + len(ln))
    return starts[node.lineno - 1] + _bcol(src_lines[node.lineno - 1], node.col_offset), \
        starts[node.end_lineno - 1] + _bcol(src_lines[node.end_lineno - 1], node.end_col_offset)


def _bcol(line: str, col: int) -> int:
    return len(line.encode()[:col].decode(errors="ignore"))


def _expr_mutants(node: ast.AST):
    """(op, description, replacement node) for one expression node."""
    if isinstance(node, ast.Compare):
        for i, op in enumerate(node.ops):
            if type(op) in CMP:
                m = copy.deepcopy(node)
                m.ops[i] = CMP[type(op)]()
                yield "cmp", f"{type(op).__name__}->{CMP[type(op)].__name__}", m
        if len(node.ops) == 1 and type(node.ops[0]) in (ast.Lt, ast.LtE, ast.Gt, ast.GtE):
            m = copy.deepcopy(node)
            m.left, m.comparators[0] = m.comparators[0], m.left
            yield "swap", "compare operands swapped", m
    elif isinstance(node, ast.BinOp):
        if type(node.op) in BIN:
            m = copy.deepcopy(node)
            m.op = BIN[type(node.op)]()
            yield "binop", f"{type(node.op).__name__}->{BIN[type(node.op)].__name__}", m
        if isinstance(node.op, NONCOMM):
            m = copy.deepcopy(node)
            m.left, m.right = m.right, m.left
            yield "swap", f"{type(node.op).__name__} operands swapped", m
        if isinstance(node.op, ast.Add):
            for side in ("left", "right"):
                other = node.right if side == "left" else node.left
                if isinstance(getattr(node, side), (ast.List, ast.Tuple)) or isinstance(other, (ast.List, ast.Tuple)):
                    yield "concat", f"concatenation keeps only its {side}", copy.deepcopy(getattr(node, side))
    elif isinstance(node, ast.BoolOp):
        m = copy.deepcopy(node)
        m.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
        yield "boolop", f"{type(node.op).__name__}->{type(m.op).__name__}", m
        for i in range(len(node.values)):
            if len(node.values) > 1:
                m = copy.deepcopy(node)
                del m.values[i]
                yield "boolop", f"operand {i} of {type(node.op).__name__} dropped", m.values[0] if len(m.values) == 1 else m
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        yield "not", "not dropped", copy.deepcopy(node.operand)
    elif isinstance(node, ast.Constant) and type(node.value) is int:
        v = node.value
        for nv in sorted({v + 1, v - 1} | ({v // 2, v * 2} if v in (8, 16, 32, 64) else set())):
            if nv >= 0 or v < 0:
                yield ("width" if v in (8, 16, 32, 64) and nv in (v // 2, v * 2) else "obo"), f"{v}->{nv}", ast.Constant(nv)
    elif isinstance(node, ast.Constant) and type(node.value) is bool:
        yield "bool", f"{node.value}->{not node.value}", ast.Constant(not node.value)
    elif isinstance(node, ast.Slice):
        for part in ("lower", "upper"):
            b = getattr(node, part)
            if b is not None:
                for d in (1, -1):
                    m = copy.deepcopy(node)
                    setattr(m, part, ast.BinOp(copy.deepcopy(b), ast.Add() if d > 0 else ast.Sub(), ast.Constant(1)))
                    yield "slice", f"slice {part} {'+' if d > 0 else '-'}1", m
    elif isinstance(node, ast.Subscript) and not isinstance(node.slice, (ast.Slice, ast.Tuple, ast.Constant)) \
            and isinstance(node.ctx, ast.Load):
        for d in (1, -1):
            m = copy.deepcopy(node)
            m.slice = ast.BinOp(copy.deepcopy(node.slice), ast.Add() if d > 0 else ast.Sub(), ast.Constant(1))
            yield "index", f"index {'+' if d > 0 else '-'}1", m
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("min", "max"):
        m = copy.deepcopy(node)
        m.func.id = "max" if node.func.id == "min" else "min"
        yield "minmax", f"{node.func.id}->{m.func.id}", m
    elif isinstance(node, ast.Tuple) and isinstance(node.ctx, ast.Load) and len(node.elts) == 2:
        m = copy.deepcopy(node)
        m.elts.reverse()
        yield "swap", "pair swapped", m


def _stmt_mutants(node: ast.stmt):
    if isinstance(node, ast.Raise):
        yield "raise", "refusal removed (raise -> pass)", "pass"
    elif isinstance(node, ast.Assert):
        yield "raise", "assert removed", "pass"
    elif isinstance(node, (ast.For, ast.AsyncFor)):
        for sl, what in (("[1:]", "first"), ("[:-1]", "last")):
            yield "loop", f"loop drops its {what} iteration", ("iter", f"list({ast.unparse(node.iter)}){sl}")
    elif isinstance(node, ast.Continue):
        yield "loop", "continue removed", "pass"


def generate(spec: dict) -> list[dict]:
    out: list[dict] = []
    for step in spec["steps"]:
        per: list[dict] = []
        for tg in step["targets"]:
            path = ROOT / tg["file"]
            src = path.read_text()
            lines = src.splitlines(keepends=True)
            tree = ast.parse(src)
            for q, fn in _funcs(tree, tg["funcs"]).items():
                body_nodes = [n for s in fn.body for n in ast.walk(s)]
                skip = {id(n) for d in fn.body if isinstance(d, ast.Expr) and isinstance(d.value, ast.Constant)
                        for n in ast.walk(d)}
                for n in body_nodes:
                    msg = n.exc if isinstance(n, ast.Raise) else n.msg if isinstance(n, ast.Assert) else \
                        n if isinstance(n, ast.JoinedStr) else None
                    if msg is not None:
                        skip |= {id(x) for x in ast.walk(msg)}
                for n in body_nodes:
                    if id(n) in skip or not hasattr(n, "lineno"):
                        continue
                    if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
                        continue
                    muts = list(_stmt_mutants(n)) if isinstance(n, ast.stmt) else list(_expr_mutants(n))
                    for op, desc, rep in muts:
                        a, b = _seg(lines, n)
                        if isinstance(rep, str):
                            text = rep
                        elif isinstance(rep, tuple) and rep[0] == "iter":
                            a, b = _seg(lines, n.iter)
                            text = rep[1]
                        elif isinstance(rep, tuple) and rep[0] == "test":
                            a, b = _seg(lines, n.test)
                            text = rep[1]
                        else:
                            try:
                                text = "(" + ast.unparse(rep) + ")"
                            except Exception:
                                continue
                        if isinstance(n, ast.Slice) or isinstance(rep, ast.Slice):
                            text = text[1:-1]
                        per.append({"step": step["name"], "file": tg["file"], "func": q, "line": n.lineno, "op": op,
                                    "desc": desc, "a": a, "b": b, "text": text,
                                    "orig": src[a:b][:120]})
        # Every refusal mutant kept; the rest sampled to the step's cap, spread over its functions.
        cap = step.get("cap", CAP)
        must = [m for m in per if m["op"] in ("raise", "guard")]
        rest = [m for m in per if m["op"] not in ("raise", "guard")]
        rng = random.Random(step["name"])
        rng.shuffle(rest)
        keep = must + rest[:cap]
        keep.sort(key=lambda m: (m["file"], m["a"], m["op"], m["desc"]))
        for i, m in enumerate(keep):
            m["id"] = f"{step['name']}-{i:04d}"
        out.extend(keep)
        print(f"{step['name']}: {len(per)} candidate mutants, kept {len(keep)} ({len(must)} refusal guards)", file=sys.stderr)
    return out


# ------------------------------------------------------------------------------------------------------------- workers

PLUGIN = r'''
import json, os, sys
_TARGETS = json.loads(os.environ["MUT_TARGETS"])
_seen = set()
_hit = {}
def _prof(frame, event, arg):
    if event == "call":
        c = frame.f_code
        f = c.co_filename
        t = _hit.get(f, 0)
        if t == 0:
            t = _hit[f] = next((t for t in _TARGETS if f.endswith(t)), None)
        if t is not None:
            _seen.add((t, c.co_qualname))
def pytest_configure(config):
    sys.setprofile(_prof)
    import threading; threading.setprofile(_prof)
def pytest_unconfigure(config):
    sys.setprofile(None)
    with open(os.environ["MUT_REACH_OUT"], "w") as fh:
        json.dump(sorted(_seen), fh)
'''


def make_worker(base: Path, i: int) -> Path:
    w = base / f"w{i}"
    if not w.exists():
        w.mkdir(parents=True)
        tar = subprocess.Popen(["tar", "-C", str(ROOT), "--exclude=./.git", "--exclude=./.venv", "--exclude=*/.venv",
                                "-cf", "-", "."], stdout=subprocess.PIPE)
        subprocess.run(["tar", "-C", str(w), "-xf", "-"], stdin=tar.stdout, check=True)
        tar.wait()
        (w / ".mut").mkdir(exist_ok=True)
        (w / ".mut" / "mutreach.py").write_text(PLUGIN)
    return w


def env_for(w: Path, extra: dict | None = None) -> dict:
    e = dict(os.environ)
    e["PYTHONPATH"] = os.pathsep.join([str(w / ".mut")] + [str(w / p) if p else str(w) for p in PKG_ROOTS])
    e["PYTHONDONTWRITEBYTECODE"] = "1"
    e.pop("PYTEST_ADDOPTS", None)
    if os.environ.get("MUT_NOLAKE"):
        e["PATH"] = os.pathsep.join(p for p in e["PATH"].split(os.pathsep) if ".elan" not in p)
    e.update(extra or {})
    return e


def module_of(file: str) -> str:
    for p in sorted(PKG_ROOTS, key=len, reverse=True):
        if p and file.startswith(p + "/"):
            return file[len(p) + 1:-3].replace("/", ".")
    return file[:-3].replace("/", ".")


def run_tests(w: Path, tests: list[str], timeout: float, extra_env: dict | None = None, reach_out: Path | None = None,
              targets: list[str] | None = None, maxfail: bool = True,
              deselect: dict | None = None) -> tuple[str, float, str]:
    """('pass'|'fail'|'timeout', seconds, tail) running ``tests`` (grouped by suite, one pytest each) in worker ``w``."""
    t0 = time.monotonic()
    py = str(ROOT / ".venv" / "bin" / "python")
    args = [py, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-m", "not slow", "--no-header", "-rfE", "-W", "ignore"]
    if os.environ.get("MUT_SLOW"):
        args[args.index("-m"):args.index("-m") + 2] = []
    if maxfail:
        args.append("-x")
    extra = dict(extra_env or {})
    if reach_out is not None:
        args += ["-p", "mutreach"]
        extra["MUT_REACH_OUT"] = str(reach_out)
        extra["MUT_TARGETS"] = json.dumps(targets or [])
    groups: dict[Path, list[str]] = {}
    for t in tests:
        groups.setdefault(suite_of(t), []).append(t)
    tail = ""
    for _suite, ts in groups.items():
        left = timeout - (time.monotonic() - t0)
        try:
            des = [x for t in ts for d in (deselect or {}).get(t, []) for x in ("--deselect", d)]
            p = subprocess.run(args + des + ts, cwd=w, env=env_for(w, extra), capture_output=True, text=True,
                               timeout=max(left, 1))
        except subprocess.TimeoutExpired:
            return "timeout", time.monotonic() - t0, ""
        tail = (p.stdout + p.stderr)[-40000:]
        if p.returncode not in (0, 5):
            return "fail", time.monotonic() - t0, tail
    return "pass", time.monotonic() - t0, tail


def suite_of(test: str) -> Path:
    """The suite directory: the nearest ancestor whose pyproject.toml configures pytest."""
    p = (ROOT / test).parent
    while p != ROOT:
        pp = p / "pyproject.toml"
        if pp.exists() and "[tool.pytest.ini_options]" in pp.read_text():
            return p
        p = p.parent
    return ROOT


# --------------------------------------------------------------------------------------------------------------- reach

def reach(spec: dict, out: Path, jobs: int, base: Path, step: str | None = None) -> dict:
    if step:
        spec = {"steps": [s for s in spec["steps"] if s["name"] in step.split(",")]}
    targets = sorted({tg["file"] for s in spec["steps"] for tg in s["targets"]})
    files = sorted({t for s in spec["steps"] for t in s["tests"]})
    workers = [make_worker(base, i) for i in range(min(jobs, len(files)))]
    res: dict = {}

    def one(i_f):
        i, f = i_f
        w = workers[i % len(workers)]
        ro = w / ".mut" / f"reach-{i}.json"
        st, dt, tail = run_tests(w, [f], timeout=900, reach_out=ro, targets=targets, maxfail=False)
        seen = json.loads(ro.read_text()) if ro.exists() else []
        bad = []
        if st == "fail":
            bad = [ln.split(" ", 1)[1].split(" - ", 1)[0].strip() for ln in tail.splitlines()
                   if ln.startswith(("FAILED ", "ERROR "))]
            if bad:
                st2, dt2, tail2 = run_tests(w, [f], timeout=900, maxfail=False, deselect={f: bad})
                st, tail = ("pass" if st2 == "pass" else "fail"), tail2
        res[f] = {"status": st, "secs": round(dt, 1), "reached": seen, "deselect": bad,
                  "tail": tail[-1500:] if (st != "pass" or bad) else ""}
        print(f"reach {f}: {st} {dt:.0f}s, {len(seen)} target functions, {len(bad)} deselected", flush=True)

    with ThreadPoolExecutor(len(workers)) as ex:
        list(ex.map(one, enumerate(files)))
    out.write_text(json.dumps(res, indent=1))
    return res


# ----------------------------------------------------------------------------------------------------------------- run

def _qual_match(func: str, reached_q: str) -> bool:
    return reached_q == func or reached_q.startswith(func + ".")


def run(spec: dict, outdir: Path, jobs: int, base: Path, step: str | None) -> None:
    muts = [json.loads(x) for x in (outdir / "mutants.jsonl").read_text().splitlines()]
    reach_map = json.loads((outdir / "reach.json").read_text())
    steps = {s["name"]: s for s in spec["steps"]}
    if step:
        muts = [m for m in muts if m["step"] in step.split(",")]
    if os.environ.get("MUT_IDS"):
        ids = set(Path(os.environ["MUT_IDS"]).read_text().split())
        muts = [m for m in muts if m["id"] in ids]
    done_path = outdir / os.environ.get("MUT_RESULTS", "results.jsonl")
    done = {json.loads(x)["id"] for x in done_path.read_text().splitlines()} if done_path.exists() else set()
    muts = [m for m in muts if m["id"] not in done]
    workers = [make_worker(base, i) for i in range(jobs)]
    free = list(workers)
    import threading
    lock = threading.Lock()
    srcs = {f: (ROOT / f).read_text() for f in {m["file"] for m in muts}}

    def tests_for(m) -> list[str]:
        ts = []
        for t in steps[m["step"]]["tests"]:
            r = reach_map.get(t)
            if r and r["status"] == "pass" and any(f == m["file"] and _qual_match(m["func"], q) for f, q in r["reached"]):
                ts.append(t)
        if os.environ.get("MUT_SLOW"):
            ts += [t for t in steps[m["step"]].get("slow", []) if t not in ts]
        return sorted(ts, key=lambda t: reach_map.get(t, {"secs": 1e9})["secs"])

    def one(m):
        with lock:
            w = free.pop()
        try:
            ts = tests_for(m)
            rec = dict(m, tests=ts)
            src = srcs[m["file"]]
            new = src[:m["a"]] + m["text"] + src[m["b"]:]
            target = w / m["file"]
            try:
                ast.parse(new)
            except SyntaxError as e:
                rec.update(status="INVALID", why=f"syntax: {e}")
                return rec
            target.write_text(new)
            try:
                imp = subprocess.run([str(ROOT / ".venv" / "bin" / "python"), "-c",
                                      f"import {module_of(m['file'])} as M; print(M.__file__)"],
                                     cwd=w, env=env_for(w), capture_output=True, text=True, timeout=300)
                if imp.returncode != 0:
                    rec.update(status="INVALID", why=imp.stderr[-400:])
                    return rec
                if not imp.stdout.strip().startswith(str(w)):
                    rec.update(status="ERROR", why=f"imported from {imp.stdout.strip()}, not the worker")
                    return rec
                if not ts:
                    rec.update(status="SURVIVED", why="no test file reaches this function", secs=0)
                    return rec
                budget = float(os.environ["MUT_BUDGET"]) if os.environ.get("MUT_BUDGET") else \
                    3 * sum(reach_map[t]["secs"] for t in ts) + 120
                st, dt, tail = run_tests(w, ts, timeout=budget,
                                         deselect={t: reach_map.get(t, {}).get("deselect", []) for t in ts})
                rec.update(status={"pass": "SURVIVED", "fail": "KILLED", "timeout": "KILLED"}[st],
                           how=st, secs=round(dt, 1), tail=tail[-500:] if st == "fail" else "")
                return rec
            finally:
                target.write_text(src)
        finally:
            with lock:
                free.append(w)
                with done_path.open("a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                print(f"{rec['id']} {rec['status']:8s} {rec['func']}:{rec['line']} {rec['desc']}", flush=True)

    with ThreadPoolExecutor(jobs) as ex:
        list(ex.map(one, muts))


def table(outdir: Path) -> str:
    rs = [json.loads(x) for x in (outdir / "results.jsonl").read_text().splitlines()]
    by: dict = {}
    for r in rs:
        s = by.setdefault(r["step"], {"KILLED": 0, "SURVIVED": 0, "INVALID": 0, "ERROR": 0, "funcs": set(), "tests": set()})
        s[r["status"]] += 1
        s["funcs"].add(r["func"])
        s["tests"].update(r.get("tests", []))
    lines = ["| step | functions | mutants | killed | survived | invalid | kill rate |", "|---|---|---|---|---|---|---|"]
    for k, s in by.items():
        n = s["KILLED"] + s["SURVIVED"] + s["INVALID"] + s["ERROR"]
        valid = s["KILLED"] + s["SURVIVED"]
        lines.append(f"| {k} | {len(s['funcs'])} | {n} | {s['KILLED']} | {s['SURVIVED']} | {s['INVALID']} | "
                     f"{100 * s['KILLED'] / max(1, valid):.0f}% |")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    cmd = argv[0]
    if cmd == "table":
        print(table(Path(argv[1])))
        return 0
    spec = json.loads(Path(argv[1]).read_text())
    jobs = int(argv[argv.index("--jobs") + 1]) if "--jobs" in argv else os.cpu_count() or 4
    base = Path(os.environ.get("MUT_BASE") or tempfile.mkdtemp(prefix="mutants-"))
    if cmd == "gen":
        ms = generate(spec)
        Path(argv[2]).write_text("".join(json.dumps(m) + "\n" for m in ms))
    elif cmd == "reach":
        reach(spec, Path(argv[2]), jobs, base, argv[argv.index("--step") + 1] if "--step" in argv else None)
    elif cmd == "run":
        step = argv[argv.index("--step") + 1] if "--step" in argv else None
        run(spec, Path(argv[2]), jobs, base, step)
    else:
        raise SystemExit(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
