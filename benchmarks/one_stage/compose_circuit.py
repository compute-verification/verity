"""Compose the circuit prover's circuit for a template, the verifier's own copy: the lowering comes from an input set of the template, and
the statement binds the given partition and program digests. Run in the circuit prover's tree (its `verity_flock`).

    python compose_circuit.py --input-set DIR --out FILE [--partition H --program H]
    python compose_circuit.py --template ID --out FILE --partition H --program H [--reads slots] [--cache DIR]

``--template``: a serving-rows template whose units are bit rows (`verity_vllm.pipeline.serving_rows.chunked`: a
``ScaledMmS8Coordinate``, a ``COORDS`` coordinate, a ``COLS`` column block, a ``SLICES`` slice, a ``KCHUNKS`` K-chunk, a
``SPLITS`` split part or combine, or a row template the partition's ``chunk_cap`` names), composed by C-Flock's `UnitStager` on
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
_CHUNK = 1 << 26
_HEX = frozenset("0123456789abcdef")


def cache_key(template: str, reads: str = "inline", packages: tuple[str, ...] = COMPOSERS,
              composer: Path = Path(__file__)) -> str:
    """``template``'s cache entry (`cached`) composed with ``reads``: SHA-512 over the template id, ``reads``, this file's bytes
    (``composer``) and every ``.py`` file of ``packages`` (its path and bytes), so that a tree whose composing code differs in
    any file has entries of its own."""
    own = composer.read_bytes()
    h = hashlib.sha512(json.dumps(["compose-cache/v2", template, reads, len(own)]).encode() + own)
    for name in packages:
        for root in importlib.util.find_spec(name).submodule_search_locations:
            for dp, dns, fns in os.walk(root):
                dns[:] = sorted(d for d in dns if not d.startswith(".") and d != "__pycache__")
                for f in sorted(x for x in fns if x.endswith(".py")):
                    p = Path(dp, f)
                    body = p.read_bytes()
                    h.update(f"{name}/{p.relative_to(root)}\0{len(body)}\0".encode() + body)
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


def cached(cache: Path, key: str, out: Path, bind: dict, compose: Callable[[], dict]) -> dict:
    """Compose's line for ``out`` through ``cache``'s entry ``key`` (`--cache`: a machine's composed circuits, so that the ranks
    it audits compose each template once). Under an exclusive lock on the entry, the first caller composes (``compose()`` writes
    ``out`` bound to ``bind`` and returns its line) and keeps a copy with where its binding is (`binding_at`); every later caller
    gets `restamp`'s copy, seconds where the compose of a GLM-5.3 GEMM's template takes half an hour. A compose that fails
    keeps nothing."""
    cache.mkdir(parents=True, exist_ok=True)
    entry = cache / key
    with (cache / f"{key}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (entry / "made.json").exists():
            made = compose()
            at = binding_at(out, bind, made["pin"])
            tmp = cache / f"{key}.tmp{os.getpid()}"
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir()
            shutil.copyfile(out, tmp / "circuit.txt")
            (tmp / "made.json").write_text(json.dumps({**made, "bind": bind, "at": at}, sort_keys=True))
            os.replace(tmp, entry)
            print(f"compose cache: composed and kept as {entry}", file=sys.stderr)
            return made
    print(f"compose cache: {entry}'s circuit, restamped", file=sys.stderr)
    return restamp(entry / "circuit.txt", json.loads((entry / "made.json").read_text()), out, bind)


def template(a: argparse.Namespace) -> int:
    from verity_flock.class_statement import UnitStager
    from verity_vllm.pipeline import chunk_rows as CR, serving_rows as SRW

    if not (a.partition and a.program):
        raise SystemExit("--template binds the statement: give --partition and --program")
    bind = {"partition": a.partition, "program": a.program}
    tp = SRW.template(a.template)
    reads = a.reads or CR.reads(tp)

    def compose() -> dict:
        in_rows, out_rows = CR.unit_rows(tp)
        st = UnitStager(tp.fn, "unit/serving-rows", in_rows, out_rows, circuit=Path(a.out), bind=bind, reads=reads)
        return {"subcircuit": st.meta.get("subcircuit"), "pin": st.pin, "class_sha512": st.class_sha512, "k_log": st.meta["k_log"]}

    made = cached(Path(a.cache), cache_key(a.template, reads), Path(a.out), bind, compose) if a.cache else compose()
    if reads == "slots":  # on a restamp too, which copies circuit.txt alone
        FC.IL.write_tables(Path(a.out).parent / "tables")
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
                    "template and tree, so that its ranks compose each template once")
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
