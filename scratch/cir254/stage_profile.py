"""CIR-254: where a unit's staging memory goes.  One unit staged as `chunk_rows.chunk_cap` stages it (`UnitStager` with its
circuit written to a file, no instance staged), in the order CIR-215's figures script ran (the description, the ANDs, the Flock
block, then the stager), with the process's memory read at every phase boundary.  Nothing of the tree is edited: the boundaries
are wrappers set on the modules' functions for this process.

    python stage_profile.py one CASE [--trace F] [--cap-gb G] [--fix NAME]   one case: $OUT/CASE.marks.jsonl, CASE.json
    python stage_profile.py all [-j N] [flags] CASE[@FIX]...                  each case in a process of its own, then figures.json

A case is KIND.N[.kK][.atA]: ``positions`` (`DsaMaskPositions_v2{N,AT,K}`), ``block`` (`DsaTopKMaskBlock_v2{N,AT,TOPK}`),
``whole`` (`DsaTopKMaskWhole_v2{N,TOPK}`) or ``merge`` (`DsaPositionsMerge_v2{M=N,K}`); K (TOPK) is 2,048 and AT 16,384 unless
the case says otherwise.

A mark is a line of CASE.marks.jsonl, written as it is read, so a killed run leaves its phases: the seconds since the last
mark, the resident memory now (``rss``), the highest since the last mark (``hwm``: VmHWM, reset at each mark through
/proc/self/clear_refs; ``samp`` is a sampling thread's own maximum, the cross-check) and the highest so far (``peak``).  ``py``
is pymalloc's own count (bytes in blocks in use, bytes its arenas hold) and ``glibc`` is mallinfo2's (in use, free in the heap,
mmapped): what the allocators hold against what the process's objects use.  With ``--trace F`` tracemalloc runs with F frames
and a snapshot is taken at the marks `SNAP` names (CASE.trace.json: the sites by bytes at each).

At the end of the lowering's gates the forms the stager holds are counted where they stand (``wires``: every returned gate's
linear form; the circuit's AND operands), and each builder-lowered activation's new forms are added up by Definition as it
ends (``by_definition``).
"""
from __future__ import annotations

import ctypes
import gc
import json
import os
import re
import resource
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

GIB = 2**30
#: the marks a traced run snapshots at
SNAP = ("< lower_gates", "< lower", "< layout", "< unit_lowering", "< _dummy", "< _write_netlist", "< compose_to", "del stager")


def status() -> dict:
    out = {}
    with open("/proc/self/status") as f:
        for line in f:
            k, _, v = line.partition(":")
            if k in ("VmRSS", "VmHWM", "RssAnon", "RssFile", "VmSwap"):
                out[k] = int(v.split()[0]) * 1024
    return out


def reset_hwm() -> bool:
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")
        return True
    except OSError:
        return False


class _Mallinfo2(ctypes.Structure):
    _fields_ = [(n, ctypes.c_size_t) for n in ("arena", "ordblks", "smblks", "hblks", "hblkhd", "usmblks", "fsmblks", "uordblks",
                                               "fordblks", "keepcost")]


def glibc() -> dict | None:
    """mallinfo2 of the main heap: bytes in use, bytes free in it, bytes in mmapped blocks."""
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.mallinfo2.restype = _Mallinfo2
        m = libc.mallinfo2()
        return {"in_use": m.uordblks, "free": m.fordblks, "mmapped": m.hblkhd, "heap": m.arena}
    except (OSError, AttributeError):
        return None


def pymalloc() -> dict | None:
    """`sys._debugmallocstats`, read off stderr: bytes in blocks in use, bytes in free blocks, the arenas' bytes, and the
    blocks in use by size class (the eight largest by bytes)."""
    try:
        sys.stderr.flush()
        with tempfile.TemporaryFile() as f:
            saved = os.dup(2)
            try:
                os.dup2(f.fileno(), 2)
                sys._debugmallocstats()
            finally:
                os.dup2(saved, 2)
                os.close(saved)
            f.seek(0)
            text = f.read().decode(errors="replace")
    except (OSError, AttributeError):
        return None
    num = lambda s: int(s.replace(",", ""))  # noqa: E731
    out: dict = {}
    classes = []
    for line in text.splitlines():
        if (m := re.match(r"\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$", line)) is not None:
            classes.append((int(m[2]) * int(m[4]), int(m[2]), int(m[4])))
        elif (m := re.match(r"# bytes in allocated blocks\s*=\s*([\d,]+)", line)) is not None:
            out["in_use"] = num(m[1])
        elif (m := re.match(r"# bytes in available blocks\s*=\s*([\d,]+)", line)) is not None:
            out["free"] = num(m[1])
        elif (m := re.match(r"(\d+) arenas \* (\d+) bytes/arena\s*=\s*([\d,]+)", line)) is not None:
            out["arenas"] = num(m[3])
        if line.startswith("Total") and "arenas" in out:
            break
    out["classes"] = [{"size": s, "blocks": n, "bytes": b} for b, s, n in sorted(classes, reverse=True)[:8]]
    return out


class Stop(Exception):
    """``--stop-after MARK``: the case ends there."""


class Marks:
    def __init__(self, path: Path, cap_gb: float | None, trace: int, detail: bool = True, count: bool = True,
                 stop_after: str | None = None):
        self.path, self.cap, self.trace, self.detail, self.count, self.stop_after = path, cap_gb, trace, detail, count, stop_after
        self.t0 = self.last = time.time()
        self.peak = 0
        self.resets = reset_hwm()
        self.samp = 0
        self.armed = False
        self.seen: dict[str, int] = {}
        self.snaps: dict[str, dict] = {}
        self.rows: list[dict] = []
        self.series = open(path.with_suffix("").with_suffix(".rss.tsv"), "w")
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def _sample(self) -> None:
        k = 0
        while not self._stop.wait(0.25):
            rss = status().get("VmRSS", 0)
            self.samp = max(self.samp, rss)
            k += 1
            if k % 8 == 0:
                self.series.write(f"{time.time() - self.t0:.1f}\t{rss / GIB:.3f}\n")
                self.series.flush()
            if self.cap is not None and rss > self.cap * GIB:
                with open(self.path, "a") as f:
                    f.write(json.dumps({"mark": "CAP", "t": round(time.time() - self.t0, 1), "rss": rss, "cap_gb": self.cap}) + "\n")
                print(f"CIR254 CAP: resident {rss / GIB:.2f} GiB is past the cap of {self.cap} GiB; exiting 3", flush=True)
                os._exit(3)

    def mark(self, name: str, **extra) -> dict:
        now = time.time()
        k = self.seen[name] = self.seen.get(name, 0) + 1
        name = name if k == 1 else f"{name}#{k}"
        s = status()
        hwm = s.get("VmHWM", 0) if self.resets else max(self.samp, s.get("VmRSS", 0))
        self.peak = max(self.peak, hwm, s.get("VmRSS", 0))
        row = {"mark": name, "t": round(now - self.t0, 2), "dt": round(now - self.last, 2), "rss": s.get("VmRSS", 0), "hwm": hwm,
               "samp": max(self.samp, s.get("VmRSS", 0)), "peak": self.peak, "anon": s.get("RssAnon"), **extra}
        if self.detail:
            row["py"], row["glibc"] = pymalloc(), glibc()
        if self.trace:
            import tracemalloc

            cur, pk = tracemalloc.get_traced_memory()
            row["traced"], row["traced_peak"], row["tracemalloc_own"] = cur, pk, tracemalloc.get_tracemalloc_memory()
            if name in SNAP:
                self.snaps[name] = self._snapshot(row)
            tracemalloc.reset_peak()
        with open(self.path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"CIR254 mark {name:28s} dt {row['dt']:9.2f} s  rss {row['rss'] / GIB:8.3f}  hwm {hwm / GIB:8.3f}  "
              f"peak {self.peak / GIB:8.3f} GiB" + (f"  traced {row['traced'] / GIB:8.3f}" if self.trace else ""), flush=True)
        self.rows.append(row)
        self.samp = 0
        reset_hwm()
        self.last = time.time()
        if name == self.stop_after:
            raise Stop(name)
        return row

    def _snapshot(self, row: dict) -> dict:
        import tracemalloc

        snap = tracemalloc.take_snapshot()
        snap = snap.filter_traces([tracemalloc.Filter(False, tracemalloc.__file__), tracemalloc.Filter(False, __file__)])
        short = lambda p: "/".join(Path(p).parts[-3:])  # noqa: E731
        by_line = [{"site": f"{short(st.traceback[0].filename)}:{st.traceback[0].lineno}", "bytes": st.size, "blocks": st.count}
                   for st in snap.statistics("lineno")[:40]]
        by_tb = [{"bytes": st.size, "blocks": st.count, "frames": [f"{short(fr.filename)}:{fr.lineno}" for fr in st.traceback]}
                 for st in snap.statistics("traceback")[:25]] if self.trace > 1 else []
        total = sum(st.size for st in snap.statistics("filename"))
        del snap
        return {"rss": row["rss"], "traced": row["traced"], "snapshot_total": total, "by_line": by_line, "by_traceback": by_tb}

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()
        self.series.close()


def unit(case: str):
    from verity.circuits.defs import bind
    from verity.definitions.registry import dsa_topk_mask_cut as X
    from verity.definitions.registry import dsa_topk_reads as R

    kind, n, *rest = case.split(".")
    n, k, at = int(n), 2048, X.BLOCK
    for r in rest:
        if r.startswith("k"):
            k = int(r[1:])
        elif r.startswith("at"):
            at = int(r[2:])
        else:
            raise ValueError(f"{case}: {r}?")
    if kind == "positions":
        return bind(R.DsaMaskPositionsV2, N=n, AT=at, K=k)
    if kind == "block":
        return bind(X.DsaTopKMaskBlockV2, N=n, AT=at, TOPK=k)
    if kind == "whole":
        return bind(X.DsaTopKMaskWholeV2, N=n, TOPK=k)
    if kind == "merge":
        return bind(R.DsaPositionsMergeV2, M=n, K=k)
    raise ValueError(f"{case}: positions, block, whole or merge")


def _forms(x, out: list) -> list:
    """The `forms.L`s of a builder's return (words, lists of words, a lone form), flat."""
    if hasattr(x, "s"):
        out.append(x)
    elif isinstance(x, (list, tuple)):
        for y in x:
            _forms(y, out)
    return out


class _Hist:
    """sets, entries and bytes by the set's length: 1, 2-4, 5-16, 17-64, ..."""

    def __init__(self):
        self.h: dict[int, list] = {}

    def add(self, n: int, b: int) -> None:
        k = 0 if n <= 1 else 1 if n <= 4 else (n - 1).bit_length() + 1 >> 1
        e = self.h.setdefault(k, [0, 0, 0])
        e[0] += 1
        e[1] += n
        e[2] += b

    def table(self) -> dict:
        lab = lambda k: "<=1" if k == 0 else "2-4" if k == 1 else f"{(1 << (2 * k - 2)) + 1}-{1 << (2 * k)}"  # noqa: E731
        return {lab(k): {"sets": v[0], "entries": v[1], "bytes": v[2]} for k, v in sorted(self.h.items())}


class _Owners:
    """Whose gate ``j`` is: the root's call that holds it, and the innermost Definition (no statics), where they differ.  A
    gate in the activation the last one was in, when that activation calls no composite, is that activation's."""

    def __init__(self, circ):
        self.circ, self.lo, self.hi, self.name = circ, 0, 0, ""
        self.leaf: dict[int, bool] = {}

    def __call__(self, j: int) -> str:
        if self.lo <= j < self.hi:
            return self.name
        g = self.circ.gate(j)
        path = [p.split("{")[0].split("#")[0] for p in g.scope.path()]
        if len(path) < 2:
            return f"(root) {str(g.prim.id).split('{')[0]}"
        name = path[1] if path[-1] == path[1] else f"{path[1]} / {path[-1]}"
        spec = g.scope.spec
        if id(spec) not in self.leaf:
            self.leaf[id(spec)] = all(getattr(nd.fn, "body", None) is None for nd in spec.body.nodes)
        if self.leaf[id(spec)]:
            self.lo, self.hi, self.name = g.scope.offset, g.scope.offset + spec.gates, name
        return name


def count_wires(C, wires: dict, circ=None) -> dict:
    """What the lowering holds at the end of its gates: ``wires``' forms (each distinct `L` and each distinct set once, by
    identity; ``by_owner`` the sets by the Definition whose gate first holds each, `_owner`) and the circuit's AND operands."""
    seen_s: set[int] = set()
    forms = entries = set_bytes = words = 0
    hist = _Hist()
    owners: dict[str, list] = {}
    owner = None if circ is None else _Owners(circ)
    for j, word in wires.items():
        words += 1
        own = None
        for f in word:
            forms += 1
            if id(f.s) in seen_s:
                continue
            seen_s.add(id(f.s))
            n, b = len(f.s), sys.getsizeof(f.s)
            entries += n
            set_bytes += b
            hist.add(n, b)
            if owner is not None:
                if own is None:
                    own = owners.setdefault(owner(j), [0, 0, 0, 0])
                    own[0] += 1
                own[1] += 1
                own[2] += n
                own[3] += b
    out = {"words": words, "word_list_bytes": sum(sys.getsizeof(w) for w in wires.values()), "dict_bytes": sys.getsizeof(wires),
           "forms": forms, "form_bytes": forms * 48, "sets": len(seen_s), "set_entries": entries, "set_bytes": set_bytes,
           "sets_by_length": hist.table(),
           "by_owner": {k: {"gates": v[0], "sets": v[1], "entries": v[2], "set_bytes": v[3]}
                        for k, v in sorted(owners.items(), key=lambda kv: -kv[1][3])}}
    del seen_s
    ands = a_n = b_n = tup = 0
    for kind, data in zip(C.kind, C.data):
        if kind == "and":
            a, b = data
            ands += 1
            a_n += len(a.s)
            b_n += len(b.s)
            tup += sys.getsizeof(a.s) + sys.getsizeof(b.s) + 96 + sys.getsizeof(data)
    out["circuit"] = {"bits": len(C.kind), "ands": ands, "a_entries": a_n, "b_entries": b_n, "operand_bytes": tup,
                      "list_bytes": sys.getsizeof(C.kind) + sys.getsizeof(C.data) + sys.getsizeof(C.phase_of)}
    return out


def count_layout(lay) -> dict:
    rows = lambda rs: {"rows": len(rs), "entries": sum(len(r) for r in rs if r is not None),  # noqa: E731
                       "bytes": sys.getsizeof(rs) + sum(sys.getsizeof(r) for r in rs if r is not None)}
    return {"a": rows(lay.rows_a), "b": rows(lay.rows_b)}


def install(M: Marks, rec: dict, fix: str | None) -> None:
    """The phase boundaries: each function below marks as it is entered and as it returns, while the stager runs."""
    from verity.circuits.boolean import forms
    from verity_flock import circuit as FC
    from verity_flock import class_statement as CS
    from verity_flock import ir_lower as IL

    def span(mod, attr: str, label: str | None = None, after=None, outer_only: bool = False):
        orig = getattr(mod, attr)
        name = label or attr
        depth = [0]

        def wrapped(*a, **kw):
            if not M.armed or (outer_only and depth[0]):
                return orig(*a, **kw)
            depth[0] += 1
            M.mark(f"> {name}")
            try:
                out = orig(*a, **kw)
            finally:
                depth[0] -= 1
            M.mark(f"< {name}")
            if after is not None:
                after(out, *a, **kw)
            return out

        wrapped.__wrapped__ = orig
        setattr(mod, attr, wrapped)

    by_def: dict[str, dict] = {}
    rec["by_definition"] = by_def
    lower_activation = IL._lower_activation

    def tallied_activation(C, s, build, wires, pieces):
        if not M.armed:
            return lower_activation(C, s, build, wires, pieces)
        k0 = len(C.kind)
        got: list = []

        def run(C_, *els):
            out = build(C_, *els)
            got.append((els, out))
            return out

        ok = lower_activation(C, s, run, wires, pieces)
        if ok and got:
            els, out = got[0]
            name = getattr(getattr(s.spec, "defn", s.spec), "name", None) or str(s.spec.id).split("{")[0]
            e = by_def.setdefault(name, {"activations": 0, "in_forms": 0, "in_entries": 0, "out_forms": 0, "out_entries": 0,
                                         "out_set_bytes": 0, "new_sets": 0, "new_entries": 0, "new_set_bytes": 0, "ands": 0,
                                         "and_entries": 0, "max_out": 0})
            ins: set[int] = set()
            for el in els:
                for f in el:
                    ins.add(id(f.s))
                    e["in_forms"] += 1
                    e["in_entries"] += len(f.s)
            seen: set[int] = set()
            for f in _forms(out, []):
                n, b = len(f.s), sys.getsizeof(f.s)
                e["out_forms"] += 1
                e["out_entries"] += n
                e["out_set_bytes"] += b
                e["max_out"] = max(e["max_out"], n)
                if id(f.s) not in ins and id(f.s) not in seen and n:
                    seen.add(id(f.s))
                    e["new_sets"] += 1
                    e["new_entries"] += n
                    e["new_set_bytes"] += b
            e["activations"] += 1
            for kind, data in zip(C.kind[k0:], C.data[k0:]):
                if kind == "and":
                    e["ands"] += 1
                    e["and_entries"] += len(data[0].s) + len(data[1].s)
        return ok

    IL._lower_activation = tallied_activation

    def after_gates(_out, C, gates, wires, *a, **kw):
        if "wires" not in rec and M.count:
            rec["wires"] = count_wires(C, wires, getattr(gates, "circuit", None))
            M.mark("count wires")

    def after_layout(lay, *a, **kw):
        if "layout" not in rec and M.count:
            rec["layout"] = count_layout(lay)
            M.mark("count layout")

    span(FC, "refuse_past_range")
    span(CS, "unit_ports")
    span(IL, "standalone")
    span(IL, "lower_gates", after=after_gates, outer_only=True)
    span(IL, "lower", outer_only=True)
    span(CS, "unit_circuit")
    span(forms.Circuit, "live")
    span(IL, "layout", after=after_layout)
    span(CS, "unit_lowering")
    span(CS, "cache_slot_circuits")
    span(FC, "_dummy")
    span(FC, "_write_netlist")
    span(FC, "compose")
    span(FC, "_file_digests")
    span(FC, "compose_to")
    span(FC, "meta_of")
    if fix:
        import fixes

        fixes.apply(fix, rec)


def one(case: str, M: Marks, fix: str | None, rec: dict) -> dict:
    from verity.circuits.boolean import trace as TR
    from verity.circuits.codec import canonical_json, descriptor_id, encode_program
    from verity.circuits.population import population_program
    from verity.commitments.windows import interface as IF
    from verity.protocols.spot.verifier.one_stage.proofs import ks_form_why
    from verity_flock import circuit as FC
    from verity_flock import ir_lower as IL
    from verity_flock.class_statement import UnitStager

    rec.update({"case": case, "fix": fix, "python": sys.version.split()[0], "trace_frames": M.trace, "hwm_resets": M.resets})
    save = lambda: M.path.with_name(M.path.name.replace(".marks.jsonl", ".json")).write_text(  # noqa: E731
        json.dumps(rec, indent=1, default=str) + "\n")
    M.mark("imports")
    install(M, rec, fix)
    fn = unit(case)
    rec["id"] = descriptor_id(fn)
    in_rows, out_rows = IF.unit_rows(fn)
    g = {nm: k for k, (nm, _ty) in enumerate(fn.signature()[0])}
    in_rows = [(f"in{g[p]}", *rest) for p, *rest in in_rows]
    rec["rows"] = {"in": [(p, sch, bits) for p, sch, bits, _ in in_rows], "out": len(out_rows)}
    M.mark("bind")
    blob = canonical_json(encode_program(population_program(fn, 1)))
    rec["program_json_bytes"] = len(blob)
    del blob
    M.mark("description")
    rec["definition_ands"] = int(IL.definition_ands(fn))
    M.mark("definition_ands")
    rec["unit_block"] = FC.unit_block(fn)
    M.mark("unit_block")
    try:
        rec["word_block"] = FC.unit_block(TR.word_of(fn))
    except Exception as e:  # noqa: BLE001
        rec["word_block"] = f"no word view: {type(e).__name__}: {e}"[:300]
    M.mark("word_block")
    save()
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        rec["tmp"] = str(d.parent)
        M.armed = True
        M.mark("> stager")
        st = UnitStager(fn, "unit/serving-rows-cap", in_rows, out_rows, circuit=d / "circuit.txt",
                        bind={"partition": "0" * 128, "program": "0" * 128}, reads="inline")
        M.mark("< stager")
        M.armed = False
        m = st.meta
        rec["meta"] = {k: m.get(k) for k in ("out_net", "lookups", "k_log", "unit_log", "vus_per_block", "statement")}
        rec["nets"] = {k: int(st.low.layout.counts[k]) for k in ("ands", "rows", "nnz_a", "nnz_b", "asserts")}
        rec["pin"], rec["class_sha512"] = st.pin, st.class_sha512
        rec["circuit_bytes"] = (d / "circuit.txt").stat().st_size
        M.mark("meta")
        rec["ks_form_why"] = ks_form_why(d / "circuit.txt", d / "pub.bin")
        M.mark("ks_form_why")
        del st, m
        gc.collect()
        M.mark("del stager")
    M.mark("end")
    rec["staged"] = True
    return rec


def run_one(argv: list[str]) -> int:
    case = argv[0]
    trace = int(argv[argv.index("--trace") + 1]) if "--trace" in argv else 0
    cap = float(argv[argv.index("--cap-gb") + 1]) if "--cap-gb" in argv else None
    fix = argv[argv.index("--fix") + 1] if "--fix" in argv else None
    tag = argv[argv.index("--tag") + 1] if "--tag" in argv else ""
    out = Path(os.environ.get("OUT", "out"))
    out.mkdir(parents=True, exist_ok=True)
    name = case + (f"+{tag}" if tag else "")
    sys.setrecursionlimit(100000)
    sys.path.insert(0, str(Path(__file__).parent))
    if trace:
        import tracemalloc

        tracemalloc.start(trace)
    marks = out / f"{name}.marks.jsonl"
    marks.unlink(missing_ok=True)
    stop = argv[argv.index("--stop-after") + 1] if "--stop-after" in argv else None
    M = Marks(marks, cap, trace, detail="--no-detail" not in argv, count="--no-count" not in argv, stop_after=stop)
    rec: dict = {}
    try:
        one(case, M, fix, rec)
    except Stop as e:
        rec["stopped_after"] = str(e)
    except BaseException as e:  # noqa: BLE001
        rec.update({"crashed": f"{type(e).__name__}: {e}"[:1000], "traceback": traceback.format_exc()[-4000:]})
    M.stop()
    rec["name"] = name
    rec["maxrss_gib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 3)
    rec["peak_gib"], rec["seconds"] = round(M.peak / GIB, 3), round(time.time() - M.t0, 1)
    rec["marks"] = M.rows
    (out / f"{name}.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
    if M.snaps:
        (out / f"{name}.trace.json").write_text(json.dumps(M.snaps, indent=1) + "\n")
    print("CIR254", name, json.dumps({k: rec.get(k) for k in ("id", "definition_ands", "nets", "pin", "circuit_bytes", "peak_gib",
                                                              "maxrss_gib", "seconds", "crashed", "traceback")})[:3000], flush=True)
    return 1 if "crashed" in rec else 0


def run_all(argv: list[str]) -> int:
    jobs = int(argv[argv.index("-j") + 1]) if "-j" in argv else 1
    valued = ("-j", "--trace", "--cap-gb", "--fix", "--tag", "--stop-after")
    flags, cases, i = [], [], 0
    while i < len(argv):
        if argv[i] in valued:
            if argv[i] != "-j":
                flags += argv[i:i + 2]
            i += 2
        elif argv[i].startswith("--"):
            flags.append(argv[i])
            i += 1
        else:
            cases.append(argv[i])
            i += 1
    out = Path(os.environ.get("OUT", "out"))
    (out / "logs").mkdir(parents=True, exist_ok=True)
    tag = flags[flags.index("--tag") + 1] if "--tag" in flags else ""
    # CASE@FIX: that case with --fix FIX, its record named CASE+FIX (commas as underscores)
    names = [c.split("@")[0] + (f"+{c.split('@')[1].replace(',', '_')}" if "@" in c else f"+{tag}" if tag else "") for c in cases]

    def launch(case: str, name: str) -> int:
        own = ["--fix", case.split("@")[1], "--tag", name.split("+", 1)[1]] if "@" in case else []
        with open(out / "logs" / f"{name}.log", "w") as log:
            rc = subprocess.run([sys.executable, __file__, "one", case.split("@")[0], *flags, *own], stdout=log,
                                stderr=subprocess.STDOUT).returncode
        print(f"CIR254 rc={rc} {name}", flush=True)
        return rc

    with ThreadPoolExecutor(jobs) as pool:
        rcs = list(pool.map(launch, cases, names))
    recs = [json.loads(f.read_text()) if (f := out / f"{n}.json").exists() else {"name": n, "crashed": "no record"} for n in names]
    (out / "figures.json").write_text(json.dumps([{k: v for k, v in r.items() if k != "marks"} for r in recs], indent=1) + "\n")
    for r in recs:
        print("CIR254 row", json.dumps({k: r.get(k) for k in ("name", "id", "definition_ands", "nets", "circuit_bytes", "peak_gib",
                                                              "maxrss_gib", "seconds", "crashed")})[:1500], flush=True)
    return 1 if any(rcs) else 0


if __name__ == "__main__":
    raise SystemExit(run_one(sys.argv[2:]) if sys.argv[1] == "one" else run_all(sys.argv[2:]))
