#!/usr/bin/env python3
"""The peak memory of an interior's commitment, made whole and made into its window directory
(``verity.commitments.windows.rows.commit``, with and without ``into``), each in a process of its own.

    research run --on auto --cwd source --mem-gb G -- bash -c 'd=$(mktemp -d); trap "rm -rf $d" EXIT; \\
        python benchmarks/commitments/interior_commit_peak.py --spans 6 --units 262144 --words 1536 --scratch "$d" --out "$OUT"'

The interior is ``--spans`` shared-row members of ``--units`` units each, one table row a unit a port, every port's rows
``--words`` u16 words under ``hm96-sha512/row/v2``: an input ``x`` and an output ``y`` a member, and every member's
``x`` but the first's a wired port with no tree of its own, the output rows of the member before (``Member.wired_trees``
false), as an interior's window lays its templates out. The members are consecutive chunks of one template's units; a
row's words are a function of its member, port and index alone, so both arms commit the same rows.

``whole`` holds every member's rows as arrays and commits them in one call, as an interior is committed today.
``streamed`` hands the commit a producer a port, which makes a run's rows when the commit asks for them, and the commit
writes the members' bodies under ``--scratch`` as it goes. Each arm reports the most its process held (``ru_maxrss``)
once the window is built and again once the commit returns, the commit's seconds, and a digest of the commitment: every
root in the registration's order, then the leaf layer. With no ``--arm`` both run, one after the other, and the result
is valid only where the two digests are equal: both arms take their salts from one key (``commit``'s ``salt_key``),
where a served commit takes as many bytes from the OS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from verity.circuits.codec import descriptor_id
from verity.circuits.defs import bind
from verity.circuits.ops import gemm as G
from verity.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2
from verity.commitments.windows import rows as SR
from verity_config.settings import SHARED

RESULT_SCHEMA = "research/result/v0.1"
ARMS = ("whole", "streamed")
TEMPLATE = bind(G.Gemm, K=16, N=16, **G.GEMM_AMPERE)
RUN = hashlib.sha512(b"interior_commit_peak").hexdigest()
KEY = hashlib.sha256(b"interior_commit_peak").digest()


def parse(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--spans", type=int, default=6, help="members")
    ap.add_argument("--units", type=int, default=1 << 16, help="units a member")
    ap.add_argument("--words", type=int, default=1536, help="u16 words a row")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--arm", choices=ARMS, default=None, help="one arm, in this process; both, a process each, when not given")
    ap.add_argument("--scratch", default=None, help="where the streamed arm writes its window (default: a temporary directory)")
    ap.add_argument("--out", default=None, help="directory for result.json (default $RESEARCH_RUN_DIR)")
    ns = ap.parse_args(argv)
    if min(ns.spans, ns.units, ns.words, ns.workers) <= 0:
        ap.error("--spans, --units, --words and --workers must be positive")
    return ns


class Rows:
    """A port's ``n`` rows of ``words`` u16 words, made when they are sliced: row i's words are a function of ``seed``, i
    and the word's index, so a slice is the same rows whoever asks."""

    def __init__(self, n: int, words: int, seed: int) -> None:
        self.shape, self.seed = (n, words), seed
        self._word = np.arange(words, dtype=np.uint16) * np.uint16(40503)

    def __getitem__(self, s: slice) -> np.ndarray:
        i = np.arange(*s.indices(self.shape[0]), dtype=np.uint64)
        return self._word[None, :] + ((i * np.uint64(25173) + (i >> np.uint64(16)) + np.uint64(self.seed)) & np.uint64(0xFFFF)).astype(np.uint16)[:, None]


def window(spans: int, units: int, words: int) -> SR.Binding:
    """The interior's window: ``spans`` chunk members of one template, each ``units`` units reading one row of ``x`` and
    writing one of ``y``; a member's ``x`` after the first is the output rows of the member before, with no tree. It names
    no sampling design: no registration is made of it."""
    program, query = SR.population_program([(TEMPLATE, spans * units)])
    tid = descriptor_id(TEMPLATE)
    return SR.Binding(program, query, {}, RUN, tuple(
        SR.Member(tid, (("x", words),), (("y", words),), units, s * units, tables=(("x", units),), name=f"{tid}/c{s}",
                  schemas=(("x", V2), ("y", V2)), wired_trees=False,
                  wired=(("x", "y", tuple(range((s - 1) * units, s * units))),) if s else ())
        for s in range(spans)))


def rows(w: SR.Binding, held: bool) -> Iterator[dict[str, Any]]:
    """Each member's rows as `rows.commit` takes them, a member when it is asked for: arrays (``held``) or producers."""
    for s, m in enumerate(w.members):
        x, y = (Rows(m.n, words, 2 * s + j) for j, (_p, words) in enumerate(m.all_rows))
        yield {"rows": {"x": SR.WIRED if m.wired else x[:] if held else x}, "outs": {"y": y[:] if held else y},
               "refs": np.arange(m.n, dtype=np.int64)[:, None]}


def peak() -> int:
    """The most this process has held, in bytes (Linux counts ``ru_maxrss`` in KiB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def arm(name: str, ns: argparse.Namespace, scratch: Path) -> dict[str, Any]:
    """One arm in this process: its peaks, its seconds and its commitment's digest."""
    w = window(ns.spans, ns.units, ns.words)
    built, t = peak(), time.perf_counter()
    if name == "whole":
        c = SR.commit(w, list(rows(w, True)), salt_key=KEY, workers=ns.workers)
    else:
        c = SR.commit(w, rows(w, False), salt_key=KEY, workers=ns.workers, into=scratch / "streamed")
    seconds, held = time.perf_counter() - t, peak()
    said = [[p, r.hex()] for p, r in c.roots.items()] + [c.leaf_layer]
    return {"arm": name, "built_rss_bytes": built, "peak_rss_bytes": held, "seconds": seconds, "commit": c.seconds,
            "workers_peak_rss_bytes": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024,
            "commitment": hashlib.sha512(json.dumps(said).encode()).hexdigest()}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    ns = parse(argv)
    started = time.perf_counter()
    scratch = Path(ns.scratch or tempfile.mkdtemp())
    try:
        if ns.arm:
            arms = [arm(ns.arm, ns, scratch)]
        else:
            size = ["--spans", str(ns.spans), "--units", str(ns.units), "--words", str(ns.words), "--workers", str(ns.workers)]
            arms = [json.loads(subprocess.run([sys.executable, __file__, *size, "--arm", a, "--scratch", str(scratch)],
                                              check=True, capture_output=True, text=True).stdout.splitlines()[-1]) for a in ARMS]
    finally:
        shutil.rmtree(scratch if ns.scratch is None else scratch / "streamed", ignore_errors=True)
    for a in arms:
        print(json.dumps(a))
    same = len({a["commitment"] for a in arms}) == 1
    ports = 2 * ns.spans - (ns.spans - 1)
    summary = {"spans": ns.spans, "units": ns.units, "words": ns.words, "workers": ns.workers, "leaves": 2 * ns.spans * ns.units,
               "row_bytes": 2 * ns.words * ns.units * ports, "salt_bytes": SR.SALT_BYTES * ns.units * ports, "arms": arms}
    out = ns.out or SHARED.RESEARCH_RUN_DIR
    if out and not ns.arm:
        d = Path(out)
        d.mkdir(parents=True, exist_ok=True)
        (d / "interior_commit_peak.json").write_text(json.dumps(summary, indent=1) + "\n")
        result: dict[str, Any] = {
            "schema": RESULT_SCHEMA,
            "run_id": SHARED.RESEARCH_RUN_ID or "local",
            "kind": "bench-result/v1",
            "workload_fingerprint": {"workload": "interior_commit_peak", "scheme": SR.SCHEME, "row_schema": V2,
                                     "spans": ns.spans, "units": ns.units, "words": ns.words, "workers": ns.workers,
                                     "rows": "shared-row members of one table row a unit a port, an input and an output a member, "
                                             "each member's input after the first its writer's output rows with no tree of its own",
                                     "implementation": "verity.commitments.windows.rows.commit, whole and into a directory",
                                     "cpu": platform.processor() or platform.machine(), "nproc": os.cpu_count(),
                                     "python": sys.version.split()[0]},
            "validation": {"status": "passed" if same else "failed",
                           "checker": "the two arms' commitments are equal: every root in the registration's order and the leaf layer",
                           "detail": "ok" if same else "the streamed commitment is not the whole one", "evidence": []},
            "measurements": [m for a in arms for m in (
                {"name": f"{a['arm']}.peak_rss", "value": a["peak_rss_bytes"], "unit": "bytes"},
                {"name": f"{a['arm']}.built_rss", "value": a["built_rss_bytes"], "unit": "bytes"},
                {"name": f"{a['arm']}.seconds", "value": a["seconds"], "unit": "s"})] + [
                {"name": "rows", "value": summary["row_bytes"], "unit": "bytes"},
                {"name": "workload_wall", "value": time.perf_counter() - started, "unit": "s"}],
            "measurement_files": ["interior_commit_peak.json"],
            "artifacts": [],
        }
        tmp = d / "result.json.tmp"
        tmp.write_text(json.dumps(result, indent=1) + "\n")
        os.replace(tmp, d / "result.json")
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
