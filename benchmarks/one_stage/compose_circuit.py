"""Compose the circuit prover's circuit for a template, the verifier's own copy: the lowering comes from an input set of the template, and
the statement binds the given partition and program digests. Run in the circuit prover's tree (its `verity_flock`).

    python compose_circuit.py --input-set DIR --out FILE [--partition H --program H]
    python compose_circuit.py --template ID --out FILE --partition H --program H [--reads slots] [--cache DIR]

``--template``: a serving-rows template whose units are bit rows (`verity_vllm.pipeline.serving_rows.chunked`: a
``ScaledMmS8Coordinate``, a ``COORDS`` coordinate, a ``COLS`` column block, a ``SLICES`` slice, a ``KCHUNKS`` K-chunk, a
``SPLITS`` split part or combine, or a row template the partition's ``chunk_cap`` names), composed by Flock's `UnitStager` on
the rows the window commits (`verity_vllm.pipeline.chunk_rows.unit_rows`), bound to the partition and program; no input set is
read. ``--reads slots``: `UnitStager`'s ``reads``, a unit reading a pinned MUFU table laid out in lookup slots
(``MoeRouterSigmoidBias``'s sigmoid inline is about 5e10 nonzeros), with the MUFU tables the prover reads written beside the
circuit (``tables/``); a unit reading none keeps its flat circuit. The default is the template's own (`chunk_rows.reads`), the
circuit a5 composes: slots for a ``SPLITS`` part of whole blocks (`verity_vllm.pipeline.split_rows`) and for a whole attention
row (``chunk_rows.SLOT_ROWS``), else inline.
"""
from __future__ import annotations

import argparse
import fcntl
import functools
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from collections.abc import Callable
from pathlib import Path

from verity_flock import circuit as FC

#: the packages a template's circuit is composed by: a cache entry is one tree's (`cache_key`)
COMPOSERS = ("verity", "verity_flock", "verity_vllm")
#: a package's directories no compose reads: its tests, and what Python and tools write beside its code
UNREAD = ("tests", "__pycache__")
#: a package's files no compose reads: bytecode and docs (a suite's inputs leave markdown out too)
UNREAD_SUFFIXES = (".pyc", ".md")
#: the ``FLOCK_*`` variables that name where things are, not how a circuit is lowered: every other one is in the key
PLACES = ("FLOCK_WORK", "FLOCK_STAGE_CACHE", "FLOCK_CIRCUIT")
_CHUNK = 1 << 26
_HEX = frozenset("0123456789abcdef")


@functools.cache
def tree_digest(packages: tuple[str, ...] = COMPOSERS) -> str:
    """SHA-512 over every file of ``packages`` but those under `UNREAD` and dot directories and those ending in `UNREAD_SUFFIXES`
    (its path and bytes): their code and the data their lowerings read (the MUFU tables, the pins), once a process."""
    h = hashlib.sha512()
    for name in packages:
        for root in importlib.util.find_spec(name).submodule_search_locations:
            for dp, dns, fns in os.walk(root):
                dns[:] = sorted(d for d in dns if not d.startswith(".") and d not in UNREAD)
                for f in sorted(x for x in fns if not x.endswith(UNREAD_SUFFIXES)):
                    p = Path(dp, f)
                    body = p.read_bytes()
                    h.update(f"{name}/{p.relative_to(root)}\0{len(body)}\0".encode() + body)
    return h.hexdigest()


def cache_key(what: object, packages: tuple[str, ...] = COMPOSERS, files: tuple[Path, ...] = (Path(__file__),)) -> str:
    """The cache entry (`cached`) of a compose that reads ``what`` besides its binding and its code (JSON: the template id, its
    Definition's program digest where the caller holds it, the stager's label, its rows and their reads): SHA-512 over
    ``what``, the ``FLOCK_*`` environment but `PLACES` (`verity_flock.class_statement`'s flags, read at import), Python's and
    NumPy's versions, the bytes of ``files`` (the composing scripts: this one, and a caller's composer's) and `tree_digest`'s,
    so that a compose whose inputs, code or data differ in any byte has an entry of its own."""
    import numpy

    env = sorted((k, v) for k, v in os.environ.items() if k.startswith("FLOCK_") and k not in PLACES)
    h = hashlib.sha512(json.dumps(["compose-cache/v3", what, env, sys.version, numpy.__version__], sort_keys=True).encode())
    for f in files:
        body = Path(f).read_bytes()
        h.update(f"{Path(f).name}\0{len(body)}\0".encode() + body)
    h.update(tree_digest(tuple(packages)).encode())
    return h.hexdigest()


def _needles(bind: dict) -> dict[str, bytes]:
    """The binding as `circuit.compose` writes it into META (compact, sorted keys): the partition's digest and the program."""
    if set(bind) != {"partition", "program"} or any(len(v) != 128 or not set(v) <= _HEX for v in bind.values()):
        raise ValueError(f"bind: a partition and a program, each a SHA-512 digest in 128 lowercase hex digits: {bind}")
    part = json.dumps({"digest": bind["partition"], "rule": FC.PARTITION_V1}, sort_keys=True, separators=(",", ":"))
    return {"partition": f'"partition":{part}'.encode(), "program": f'"program_sha512":"{bind["program"]}"'.encode()}


def binding_at(path: Path, bind: dict, pin: str) -> dict[str, int]:
    """Where the circuit at ``path``, composed bound to ``bind``, holds that binding: the offset of each value, found once in
    the whole file (else refused), with the file's SHA-512 held to ``pin``."""
    needles, hits, h = _needles(bind), {"partition": [], "program": []}, hashlib.sha512()
    keep = max(map(len, needles.values())) - 1
    with path.open("rb") as f:
        tail, pos = b"", 0      # pos: the file offset of buf[0]
        while chunk := f.read(_CHUNK):
            h.update(chunk)
            buf = tail + chunk
            for k, n in needles.items():
                i = buf.find(n)
                while i >= 0:
                    if i + len(n) > len(tail):      # a match inside the tail was found with the chunk before
                        hits[k].append(pos + i)
                    i = buf.find(n, i + 1)
            tail = buf[-keep:]
            pos += len(buf) - len(tail)
    if h.hexdigest() != pin:
        raise RuntimeError(f"{path}: its SHA-512 is not the pin its compose gave")
    if any(len(v) != 1 for v in hits.values()):
        raise RuntimeError(f"{path}: the binding is not in the file once each: {({k: len(v) for k, v in hits.items()})}")
    return {k: v[0] for k, v in hits.items()}


def restamp(src: Path, made: dict, out: Path, bind: dict) -> dict:
    """The cache entry's circuit ``src`` (its line ``made``: composed bound to ``made["bind"]``, the values at ``made["at"]``)
    bound to ``bind`` instead, written to ``out``. `circuit.compose` writes the binding only as those two META values, each 128
    hex digits (the class, `circuit.class_digest`, leaves them out), so overwriting them in place is the circuit composed
    bound to ``bind``. ``src`` is held to its pin as it is copied. -> compose's line for ``out``."""
    old, new = _needles(made["bind"]), _needles(bind)
    patches = [(made["at"][k], old[k], new[k]) for k in old]
    h_src, h_out = hashlib.sha512(), hashlib.sha512()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f"{out.name}.tmp{os.getpid()}")
    try:
        with src.open("rb") as f, tmp.open("wb") as g:
            pos = 0
            while chunk := f.read(_CHUNK):
                h_src.update(chunk)
                buf = bytearray(chunk)
                for at, o, n in patches:
                    lo, hi = max(at, pos), min(at + len(o), pos + len(buf))
                    if lo < hi:
                        if buf[lo - pos:hi - pos] != o[lo - at:hi - at]:
                            raise RuntimeError(f"{src}: the binding is not at the offset its cache entry recorded")
                        buf[lo - pos:hi - pos] = n[lo - at:hi - at]
                h_out.update(buf)
                g.write(buf)
                pos += len(buf)
        if h_src.hexdigest() != made["pin"]:
            raise RuntimeError(f"{src}: not the circuit its cache entry kept (its SHA-512 is not the pin)")
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, out)
    return {**{k: v for k, v in made.items() if k not in ("bind", "at")}, "pin": h_out.hexdigest()}


def head(path: Path) -> dict:
    """The META of the circuit at ``path``, read off its first two lines and nothing after them."""
    with path.open("rb") as f:
        f.readline()
        line = f.readline()
    if not line.startswith(b"META "):
        raise RuntimeError(f"{path}: its second line is not its META")
    return json.loads(line[5:])


def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.unlink(missing_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def cached(cache: Path, key: str, out: Path | None, bind: dict, compose: Callable[[Path], dict]) -> dict:
    """Compose's line for ``out`` through ``cache``'s entry ``key`` (`--cache`: a machine's composed circuits, so that the ranks
    it audits compose each template once, and a window's composes can be made before its timed audit, `a5.warm`). Under an
    exclusive lock on the entry, the first caller composes into it (``compose(path)`` writes the circuit at ``path`` bound to
    ``bind``, with any MUFU tables in ``tables/`` beside it, and returns its line) and keeps it with where its binding is
    (`binding_at`, which holds it to its pin). Then ``out`` gets the kept circuit and tables as links (a copy across file
    systems), never read, when ``bind`` is the kept one's, and `restamp`'s copy otherwise: seconds where the compose of a
    GLM-5.3 GEMM's template takes half an hour and a 4,32,4 window's Int8RowQuant_v3 four hours. The line carries the circuit's
    ``meta`` (`head`), so that no caller reads a circuit, past 100 GB, to stage public files under it; a kept circuit is held
    to its pin when it is copied, and by the caller when a sampled proof block's prover reads it (a5's `held`). ``out`` None
    only makes the entry. A compose that fails keeps nothing."""
    _needles(bind)
    cache.mkdir(parents=True, exist_ok=True)
    entry = cache / key
    with (cache / f"{key}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (entry / "made.json").exists():
            tmp = cache / f"{key}.tmp{os.getpid()}"
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir()
            try:
                made = compose(tmp / "circuit.txt")
                made = {**made, "meta": head(tmp / "circuit.txt")}
                at = binding_at(tmp / "circuit.txt", bind, made["pin"])
                (tmp / "made.json").write_text(json.dumps({**made, "bind": bind, "at": at}, sort_keys=True))
            except BaseException:
                shutil.rmtree(tmp, ignore_errors=True)
                raise
            os.replace(tmp, entry)
            how = "composed and kept as"
        else:
            how = "taken from"
    kept = json.loads((entry / "made.json").read_text())
    if out is None:
        return {k: v for k, v in kept.items() if k not in ("bind", "at")}
    for f in sorted((entry / "tables").glob("*")) if (entry / "tables").is_dir() else ():
        _link(f, out.parent / "tables" / f.name)
    if kept["bind"] == bind:
        _link(entry / "circuit.txt", out)
        print(f"compose cache: {how} {entry}", file=sys.stderr)
        return {k: v for k, v in kept.items() if k not in ("bind", "at")}
    print(f"compose cache: {how} {entry}, restamped", file=sys.stderr)
    return {**restamp(entry / "circuit.txt", kept, out, bind), "meta": head(out)}


def template(a: argparse.Namespace) -> int:
    from verity_flock.class_statement import UnitStager
    from verity_vllm.pipeline import chunk_rows as CR, serving_rows as SRW

    if not (a.partition and a.program):
        raise SystemExit("--template binds the statement: give --partition and --program")
    bind = {"partition": a.partition, "program": a.program}
    tp = SRW.template(a.template)
    reads = a.reads or CR.reads(tp)

    def compose(out: Path) -> dict:
        in_rows, out_rows = CR.unit_rows(tp)
        st = UnitStager(tp.fn, "unit/serving-rows", in_rows, out_rows, circuit=out, bind=bind, reads=reads)
        if reads == "slots":
            FC.IL.write_tables(out.parent / "tables")
        return {"subcircuit": st.meta.get("subcircuit"), "pin": st.pin, "class_sha512": st.class_sha512, "k_log": st.meta["k_log"],
                "meta": st.meta}

    out = Path(a.out)
    if a.cache:
        made = cached(Path(a.cache), cache_key(["unit/serving-rows", a.template, reads]), out, bind, compose)
    else:
        out.unlink(missing_ok=True)
        made = compose(out)
    print(json.dumps(made))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-set", default=None)
    ap.add_argument("--template", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--partition", default=None)
    ap.add_argument("--program", default=None)
    ap.add_argument("--cache", default=None, help="with --template: this machine's composed circuits (`cached`), one entry per "
                    "template and tree (`cache_key`), so that its ranks compose each template once")
    ap.add_argument("--reads", choices=("inline", "slots"), default=None, help="with --template: UnitStager's reads, a "
                    "pinned MUFU table's reads inline or in lookup slots (default: the template's own, chunk_rows.reads)")
    a = ap.parse_args()
    if bool(a.input_set) == bool(a.template):
        ap.error("give either --input-set or --template")
    if a.cache and not a.template:
        ap.error("--cache keeps --template circuits only")
    if a.reads and not a.template:
        ap.error("--reads lays out a --template unit's reads; an input set's lowering is its own")
    if a.template:
        return template(a)
    from verity_numerical.bench.input_sets import InputSet

    s = InputSet.open(a.input_set)
    low = FC.lowering_for_set(s)
    bind = {k: v for k, v in (("partition", a.partition), ("program", a.program)) if v}
    text = FC.compose(low, s.subcircuit, **bind)
    Path(a.out).write_text(text)
    if low.units.cut:
        FC.IL.write_tables(Path(a.out).parent / "tables")
    out = {"subcircuit": s.subcircuit.id, "pin": FC.digest(text)}
    if hasattr(FC, "class_digest"):
        out["class_sha512"] = FC.class_digest(text)
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
