"""Each stratum's circuit file of a period's declared window Program, built with the auditor's own lowering, and the
``--circuit-files`` its period-open commits (core's `credit.committed_files`).

A drawn proof block at K is proved on K's template, `h1window.template(K, h1window.tile_unit(K, TILE))` (its unit in
`weights.chunks` (K) chunks past `weights.CHUNK_MAX`, `h1window._chunked`, as the Program declares it), one of the declared
Program's ``TILES``. It is lowered by C-Flock's `class_statement.UnitStager`, as ``benchmarks/pouw/h1_prove/stage.py``
lowers it (`h1unit.Shape.stager`, under ``HZ_STAGE``, ``fast`` by default), and bound to the window's Program and
partition: ``{"program": its program_digest, "partition": the digest of its Q_template_instances partition object}``,
its templates by ascending K, the object core's credit verifies a statement under (`credit.partition_of`). The file's
SHA-512 is what the period-open commits for ``k<K>``, and core's credit refuses a statement whose circuit file hashes
to anything else (`proofs.verify`'s ``circuit_sha512``). Until the verifier derives the circuit from the descriptor
itself, that tie rests on this lowering being the auditor's. A Program over J10-R's tile unit (serve-openai's pouw
setting ``unit`` j10r) binds the same way, but `h1unit.Shape`'s statement is v1's or v3's, so its strata are refused
and only its live file is built (``--strata none``).

With ``--live T,S`` the manifest's ``files`` also holds ``live``: the SHA-512 of the circuit file of the window's live
Program's template, `live.template_v2(T)` bound to `live.program_v2(T, S, PREFIX)` (PREFIX the declared Program's,
`credit._prefix`) and its `live.query_v2` partition, lowered as ``h1_prove/live.py`` lowers it (`live_stager`), which is
what core's credit verifies each round's ``live@<s>`` on (``circuit_files["live"]``). T is the period's rows and S the
window's rounds. Past T = `live.PIECE_ROWS` it is the piece's, `live.template_rows(T, PIECE_ROWS)` bound to
`live.program_rows` (a V3 prefill round at T 8,192 is 128 pieces of 64 rows). It takes seconds.

The lowering runs once per K, each in its own process (`lower`). At K = 4096 and 14336 on the 32 × 32 tile it is the
long step, so it runs once as its own CPU job and later runs reuse its directory (`build`, `MANIFEST`). ``--strata``
lowers only the K it lists (``none``: only the live file), and the manifest then holds no ``k<K>`` for the rest:

    python -m tests.protocols.pouw_h1_circuits --program PROGRAM_DIR --out DIR [--live T,S] [--strata K,...|none]
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

HERE = Path(__file__).resolve().parent
H1_PROVE = HERE.parents[3] / "benchmarks" / "pouw" / "h1_prove"
#: the statement's name, `h1_prove/stage.py`'s
NAME = "unit/pouw-h1-tile"
MANIFEST = "circuit-files.json"
LIVE_FILE = "live.circuit.txt"
#: ``lower(k, tile, bind, path)``: K's template's circuit file at `path`, bound to `bind`
Lower = Callable[[int, tuple[int, int], dict, Path], None]
#: ``lower_live(T, part, path)``: the live template's circuit file at `path`, bound to the live Program's partition `part`
LowerLive = Callable[[int, dict, Path], None]


def strata(layout: dict) -> list[int]:
    """The declared window's strata: each K its linears read, ascending (``layout["linears"]``, ``[name, K, n]``)."""
    return sorted({int(k) for _name, k, _n in layout["linears"]})


def bind_of(descriptor: bytes, layout: dict) -> tuple[dict[str, str], dict[int, str], set[int]]:
    """What every stratum's circuit file is bound to, ``{"partition", "program"}``, each K's template id, and the K's
    whose template is J10-R's, for the declared window Program whose descriptor is `descriptor` (the period's
    ``--circuit`` its SHA-512) and whose layout is `layout`. K's template is over v1's tile unit at the layout's tile
    (`h1window.tile_unit`) or J10-R's (`h1tile.PouwH1Tile5`, serve-openai's pouw setting ``unit`` j10r). ValueError when
    neither is one of the descriptor's definitions, or (with core's credit) the partition is not the one core's credit
    verifies statements under."""
    from verity.core.protocols.pouw import credit as CR
    from verity.core.protocols.pouw.circuit import h1tile as HT
    from verity.core.protocols.pouw.circuit import h1window as HW
    from verity.ml.circuits import partition_object as PO
    from verity.ml.circuits.codec import descriptor_id
    program, desc = hashlib.sha512(descriptor).hexdigest(), json.loads(descriptor)
    tile = tuple(layout["tile"])

    def declared(k: int) -> tuple[str, bool] | None:
        for definition in (None, HT.PouwH1Tile5):
            i = descriptor_id(HW.template(k, HW._chunked(k, HW.tile_unit(k, tile, definition))))
            if i in desc["definitions"]:
                return i, definition is not None
        return None
    got = {k: declared(k) for k in strata(layout)}
    missing = sorted(k for k, g in got.items() if g is None)
    if missing:
        raise ValueError(f"K = {missing}'s template (h1window.template over v1's or J10-R's tile_unit at {list(tile)}) "
                         "is not one of the declared Program's definitions, so no circuit file of it is the Program's")
    ids, j10r = {k: g[0] for k, g in got.items()}, {k for k, g in got.items() if g[1]}
    part = PO.build(desc, PO.template_instances_query([ids[k] for k in sorted(ids)]))
    if part["program"] != program:
        raise ValueError(f"the descriptor's program_digest {part['program'][:16]}... is not its SHA-512 {program[:16]}...")
    if hasattr(CR, "partition_of") and CR.partition_of(program, CR.shape_of(descriptor)) != part:
        raise ValueError("the partition the circuit files are bound to is not the one core's credit verifies under "
                         "(credit.partition_of)")
    return {"partition": PO.digest(part), "program": program}, ids, j10r


def file_name(k: int) -> str:
    return f"k{int(k)}.circuit.txt"


def lower(k: int, tile: tuple[int, int], bind: dict, path: Path) -> None:
    """The auditor's lowering of K's template into `path`, in its own process (`_lower`)."""
    subprocess.run([sys.executable, "-m", "tests.protocols.pouw_h1_circuits", "--lower", str(k), str(tile[0]),
                    str(tile[1]), json.dumps(bind), str(path)], check=True, cwd=HERE.parents[1])


def _lower(k: int, r: int, c: int, bind: dict, path: Path) -> None:
    """`h1_prove/stage.py`'s build of the statement's circuit, at R × C, bound to `bind`, written to `path`."""
    sys.path.insert(0, str(H1_PROVE))
    sys.path.insert(0, str(H1_PROVE.parent / "hidden_zk"))
    if os.environ.get("HZ_STAGE", "fast") == "fast":
        import fast_stage
        fast_stage.install(path.parent / f".scratch-k{k}")
    import h1unit as U
    U.Shape(k, r, c).stager(NAME, circuit=path, bind=bind)


def _h1_live():
    import importlib.util
    spec = importlib.util.spec_from_file_location("h1_prove_live", H1_PROVE / "live.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def live_part(T: int, S: int, prefix: str) -> dict:
    """The partition object of the live Program a window of S rounds at T rows under `prefix` proves on: v2's, or past
    `live.PIECE_ROWS` rows the pieced one (``h1_prove/live.py``'s `program_of`)."""
    from verity.core.protocols.sampled_proofs.one_stage import partition as P
    return P.partition(*_h1_live().program_of({"T": T, "S": S, "prefix": prefix}))


def lower_live(T: int, part: dict, path: Path) -> None:
    """``h1_prove/live.py``'s lowering of the live template at T rows (a piece's past `live.PIECE_ROWS`), bound to `part`,
    written to `path`."""
    module = _h1_live()
    r, p = module.form({"T": T})
    module.live_stager(T, path, part, None if p == 1 else r)


def build(program_dir: Path, out: Path, lower: Lower = lower, live: tuple[int, int] | None = None,
          lower_live: LowerLive = lower_live, only: Sequence[int] | None = None) -> dict[str, Any]:
    """Each stratum's circuit file of the declared window Program in `program_dir` (the rehearsal's ``PROGRAM`` and
    ``PROGRAM_META``), lowered by `lower` into `out` (`file_name`) unless `out`'s `MANIFEST` already records it under
    the same bind, and the manifest: ``{program, bind, templates, files}``, ``files`` the period-open's
    ``--circuit-files`` (``{k<K>: SHA-512 of K's file}``). With `live` (T, S) the live template's file too (`LIVE_FILE`,
    by `lower_live`, kept under the same ``live_bind``), ``files["live"]`` its SHA-512. A J10-R stratum isn't lowered:
    `_lower`'s statement (`h1unit.Shape`) is v1's or v3's, so ValueError unless `only` leaves it out."""
    from tests.protocols.pouw_h1_rehearsal import PROGRAM, PROGRAM_META
    meta = json.loads((program_dir / PROGRAM_META).read_text())
    descriptor = gzip.decompress((program_dir / PROGRAM).read_bytes())
    bind, ids, j10r = bind_of(descriptor, meta["layout"])
    if unbuilt := sorted(k for k in j10r if only is None or k in only):
        raise ValueError(f"K = {unbuilt}'s template is over J10-R's tile unit, whose statement h1unit.Shape doesn't "
                         "lower (v1's or v3's): give --strata none, the live file alone")
    out.mkdir(parents=True, exist_ok=True)
    try:
        kept = json.loads((out / MANIFEST).read_text())
    except (OSError, ValueError):
        kept = {}
    files, extra = {}, {}
    if live is not None:
        from verity.core.protocols.pouw import credit as CR
        from verity.core.protocols.sampled_proofs.one_stage import partition as P
        T, S = live
        part = live_part(T, S, CR._prefix(descriptor))
        extra["live_bind"] = {"T": T, "S": S, "partition": P.digest(part), "program": part["program"]}
        path = out / LIVE_FILE
        if not (kept.get("live_bind") == extra["live_bind"] and path.is_file()):
            lower_live(T, part, path)
        files[CR.LIVE] = hashlib.sha512(path.read_bytes()).hexdigest()
    for k in strata(meta["layout"]):
        if only is not None and k not in only:
            continue
        path = out / file_name(k)
        if not (kept.get("bind") == bind and path.is_file()):
            lower(k, tuple(meta["layout"]["tile"]), bind, path)
        files[f"k{k}"] = hashlib.sha512(path.read_bytes()).hexdigest()
    doc = {"program": bind["program"], "bind": bind, "templates": {f"k{k}": i for k, i in ids.items()}, "files": files,
           **extra}
    (out / MANIFEST).write_text(json.dumps(doc, indent=1) + "\n")
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--program", metavar="DIR", help="the declared window Program (a --program-only run's out/)")
    ap.add_argument("--out", metavar="DIR", help="where each stratum's circuit file and the manifest go")
    ap.add_argument("--live", metavar="T,S", type=lambda v: tuple(int(x) for x in v.split(",")),
                    help="also the live template's circuit file, for windows of S rounds at T rows")
    ap.add_argument("--strata", metavar="K,...|none", type=lambda v: () if v == "none" else tuple(int(x) for x in v.split(",")),
                    help="lower only these strata's files (none: only --live's)")
    ap.add_argument("--lower", nargs=5, metavar=("K", "R", "C", "BIND", "PATH"), help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.lower:
        k, r, c, bind, path = a.lower
        _lower(int(k), int(r), int(c), json.loads(bind), Path(path))
        return 0
    if not (a.program and a.out):
        ap.error("--program and --out")
    doc = build(Path(a.program), Path(a.out), live=a.live, only=a.strata)
    print(f"POUW-H1-CIRCUITS program={doc['program'][:16]} files={json.dumps(doc['files'])}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
