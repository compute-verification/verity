"""Each stratum's circuit file of a period's declared window Program, built with the auditor's own lowering, and the
``--circuit-files`` its period-open commits (core's `credit.committed_files`).

A sampled proof block at K is proved on K's template, `h1window.template(K, h1window.tile_unit(K, TILE))` (its unit in
`weights.chunks` (K) chunks past `weights.CHUNK_MAX`, `h1window._chunked`, as the Program declares it), one of the declared
Program's ``TILES``. It is lowered by Flock's `class_statement.UnitStager`, as ``benchmarks/pouw/h1_prove/stage.py``
lowers it (`h1unit.Shape.stager`, under ``HZ_STAGE``, ``fast`` by default), and bound to the window's Program and
partition: ``{"program": its program_digest, "partition": the digest of its Q_template_instances partition object}``,
its templates by ascending K, the object core's credit verifies a statement under (`credit.partition_of`). The file's
SHA-512 is what the period-open commits for ``k<K>``, and core's credit refuses a statement whose circuit file hashes
to anything else (`proofs.verify`'s ``circuit_sha512``). Until the verifier derives the circuit from the descriptor
itself, that tie rests on this lowering being the auditor's. A Program over J10-R's tile unit (serve-openai's pouw
setting ``unit`` j10r) binds the same way, and its strata are lowered as J10-R's statement (`h1unit.Shape`'s ``j10r``:
its rows named as credit's ``--registered`` file reads them, its public row with the unit's call and blocks).

With ``--live T,S`` the manifest's ``files`` also holds ``live``: the SHA-512 of the circuit file of the window's live
Program's template, `live.template_v2(T)` bound to `live.program_v2(T, S, PREFIX)` (PREFIX the declared Program's,
`credit._prefix`) and its `live.query_v2` partition, lowered as ``h1_prove/live.py`` lowers it (`live_stager`), which is
what core's credit verifies each round's ``live@<s>`` on (``circuit_files["live"]``). T is the period's rows and S the
window's rounds. Past T = `live.PIECE_ROWS` it is the piece's, `live.template_rows(T, PIECE_ROWS)` bound to
`live.program_rows` (a V3 prefill round at T 8,192 is 128 pieces of 64 rows). It takes seconds.

With ``--lanes K,...`` each listed K is committed in credit's lanes form (`credit.form_of`) instead of ``k<K>``: a sampled
leaf at K is proved as its lanes and combine (`leaf_lanes.statements`), each statement on its own file, ``lane<K>`` and
``combine<K>``, the flat statement of `h1window.leaf_unit` of the declared window Program at K (rebuilt from the
descriptor's layout, `rebuilt`, its templates held to the descriptor's) on `drawn.leaf_rows`, bound to that statement's
leaf Program and its partition (`h1window.leaf_program`, a population of one), which core's credit derives itself
(`credit.leaf_of`, `credit.leaf_partition_of`). The manifest's ``files`` holds both files' SHA-512 and
``leaf_programs`` each leaf Program's descriptor's (written beside the file, `drawn.LEAF_PROGRAM`), and its
``templates`` each statement's Definition id. An sm_90 window's Program is the sm_120 one's byte for byte
(`h1device`), so the files serve either card.

The lowering runs once per K, each in its own process (`lower`, `lower_leaf`). At K = 4096 and 14336 on the 32 × 32 tile
it is the long step, so it runs once as its own CPU job and later runs reuse its directory (`build`, `MANIFEST`).
``--strata`` lowers only the K it lists (``none``: only the live file), and the manifest then holds no ``k<K>`` for the
rest:

    python -m tests.protocols.pouw_h1_circuits --program PROGRAM_DIR --out DIR [--live T,S] [--strata K,...|none]
                                               [--lanes K,...]
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
H1_PROVE = HERE.parents[3] / "benchmarks" / "pouw" / "h1_prove"
#: the statement's name, `h1_prove/stage.py`'s
NAME = "unit/pouw-h1-tile"
MANIFEST = "circuit-files.json"
LIVE_FILE = "live.circuit.txt"
#: ``lower(k, tile, bind, path[, j10r=True])``: K's template's circuit file at `path`, bound to `bind` (J10-R's with
#: ``j10r``, given only for a J10-R stratum)
Lower = Callable[..., None]
#: ``lower_live(T, part, path)``: the live template's circuit file at `path`, bound to the live Program's partition `part`
LowerLive = Callable[[int, dict, Path], None]
#: ``lower_leaf(k, kind, program, path)``: lanes-form K's ``kind`` statement's circuit file at `path`, of the declared
#: window Program whose gzipped descriptor is at `program`
LowerLeaf = Callable[[int, str, Path, Path], None]


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
    from verity.circuits import partition_object as PO
    from verity.circuits.codec import descriptor_id
    from verity.protocols.pouw.prover.circuit import h1tile as HT
    from verity.protocols.pouw.prover.circuit import h1window as HW
    from verity.protocols.pouw.verifier import credit as CR
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
    found = {k: g for k, g in got.items() if g is not None}
    ids, j10r = {k: g[0] for k, g in found.items()}, {k for k, g in found.items() if g[1]}
    part = PO.build(desc, PO.template_instances_query([ids[k] for k in sorted(ids)]))
    if part["program"] != program:
        raise ValueError(f"the descriptor's program_digest {part['program'][:16]}... is not its SHA-512 {program[:16]}...")
    if hasattr(CR, "partition_of") and CR.partition_of(program, CR.shape_of(descriptor)) != part:
        raise ValueError("the partition the circuit files are bound to is not the one core's credit verifies under "
                         "(credit.partition_of)")
    return {"partition": PO.digest(part), "program": program}, ids, j10r


def file_name(k: int) -> str:
    return f"k{int(k)}.circuit.txt"


def lower(k: int, tile: tuple[int, int], bind: dict, path: Path, j10r: bool = False) -> None:
    """The auditor's lowering of K's template into `path`, in its own process (`_lower`)."""
    subprocess.run([sys.executable, "-m", "tests.protocols.pouw_h1_circuits", "--lower", str(k), str(tile[0]),
                    str(tile[1]), json.dumps(bind), str(path), *(["--j10r"] if j10r else [])], check=True,
                   cwd=HERE.parents[1])


def _lower(k: int, r: int, c: int, bind: dict, path: Path, j10r: bool = False) -> None:
    """`h1_prove/stage.py`'s build of the statement's circuit, at R × C (J10-R's with `j10r`), bound to `bind`, written
    to `path`."""
    sys.path.insert(0, str(H1_PROVE))
    sys.path.insert(0, str(H1_PROVE.parent / "hidden_zk"))
    if os.environ.get("HZ_STAGE", "fast") == "fast":
        import fast_stage
        fast_stage.install(path.parent / f".scratch-k{k}")
    import h1unit as U
    U.Shape(k, r, c, j10r).stager(NAME, circuit=path, bind=bind)


def _h1_live():
    spec = importlib.util.spec_from_file_location("h1_prove_live", H1_PROVE / "live.py")
    assert spec is not None and spec.loader is not None, f"{H1_PROVE / 'live.py'} is no module"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def live_part(T: int, S: int, prefix: str) -> dict:
    """The partition object of the live Program a window of S rounds at T rows under `prefix` proves on: v2's, or past
    `live.PIECE_ROWS` rows the pieced one (``h1_prove/live.py``'s `program_of`)."""
    from verity.protocols.spot.verifier.one_stage import partition as P
    return P.partition(*_h1_live().program_of({"T": T, "S": S, "prefix": prefix}))


def lower_live(T: int, part: dict, path: Path) -> None:
    """``h1_prove/live.py``'s lowering of the live template at T rows (a piece's past `live.PIECE_ROWS`), bound to `part`,
    written to `path`."""
    module = _h1_live()
    r, p = module.form({"T": T})
    module.live_stager(T, path, part, None if p == 1 else r)


def rebuilt(descriptor: bytes) -> Any:
    """The declared window Program rebuilt from its descriptor's layout (`credit.layout_of`, its prefix `credit.shape_of`'s),
    each K's template held to the descriptor's by its descriptor id, which names every binding. ValueError otherwise."""
    from verity.circuits.codec import descriptor_id
    from verity.protocols.pouw.prover.circuit import h1window as HW
    from verity.protocols.pouw.verifier import credit as CR
    layout, why = CR.layout_of(descriptor)
    if why is not None:
        raise ValueError(f"the declared Program's layout: {why[1]}")
    shape = CR.shape_of(descriptor)
    win = HW.program([tuple(x) for x in layout["linears"]], [tuple(x) for x in layout["calls"]], layout["m"],
                     tuple(layout["tile"]), shape["prefix"])
    got = {int(k): descriptor_id(t) for k, t in win.fn.bindings["TILES"]}
    want = {int(k): t["fn"] for k, t in shape["tiles"].items()}
    if got != want:
        raise ValueError(f"the window Program rebuilt from the descriptor's layout has other templates at K "
                         f"{sorted(k for k in got.keys() | want.keys() if got.get(k) != want.get(k))}: a lanes-form K's "
                         "leaf is v1's tile unit's")
    return win


def leaf_bound(win: Any, k: int, kind: str) -> tuple[Any, bytes, dict, dict]:
    """K's leaf statement ``kind`` of the window Program `win`: its Definition (`h1window.leaf_unit`), its leaf Program's
    descriptor (`served.descriptor`), its partition object and what its file is bound to, ``{"partition",
    "program"}``."""
    from verity.protocols.pouw.prover.circuit import h1window as HW
    from verity.protocols.spot.verifier.one_stage import partition as P

    from verity_vllm.protocols.pouw.h1 import served as SV
    program, query = HW.leaf_program(win, k, kind)
    part = P.partition(program, query)
    desc = SV.descriptor(program)
    if hashlib.sha512(desc).hexdigest() != part["program"]:
        raise ValueError(f"{kind}{k}: the leaf Program's descriptor is not the digest its partition names")
    return HW.leaf_unit(win, k, kind), desc, part, {"partition": P.digest(part), "program": part["program"]}


def leaf_file(kind: str, k: int) -> str:
    return f"{kind}{int(k)}.circuit.txt"


def lower_leaf(k: int, kind: str, program: Path, path: Path) -> None:
    """The auditor's lowering of K's leaf statement ``kind`` into `path`, in its own process (`_lower_leaf`)."""
    subprocess.run([sys.executable, "-m", "tests.protocols.pouw_h1_circuits", "--lower-leaf", str(k), kind,
                    str(Path(program).resolve()), str(Path(path).resolve())], check=True, cwd=HERE.parents[1])


def _lower_leaf(k: int, kind: str, program: Path, path: Path) -> None:
    """K's leaf statement ``kind`` (`leaf_bound`) as the flat statement on `drawn_rows.leaf_rows`, its output
    `drawn_rows.LEAF_OUT_ROWS`, lowered under ``HZ_STAGE`` (``fast`` by default) and written to `path`."""
    from verity_flock import class_statement as CS

    from verity_vllm.protocols.pouw.h1 import drawn_rows as D
    if os.environ.get("HZ_STAGE", "fast") == "fast":
        sys.path.insert(0, str(H1_PROVE.parent / "hidden_zk"))
        import fast_stage
        fast_stage.install(path.parent / f".scratch-{kind}{k}")
    from verity.protocols.pouw.verifier import credit as CR
    descriptor = gzip.decompress(program.read_bytes())
    defn, _desc, _part, bind = leaf_bound(rebuilt(descriptor), k, kind)
    tile = CR.layout_of(descriptor)[0]["tile"]
    CS.UnitStager(defn, D.LEAF_NAMES[kind], D.leaf_rows(kind, k, tile), D.LEAF_OUT_ROWS, circuit=path, bind=bind)


def leaf_files(program_dir: Path, out: Path, descriptor: bytes, ks: Sequence[int], kept: dict,
               lower_leaf: LowerLeaf = lower_leaf) -> tuple[dict, dict, dict]:
    """Each lanes-form K's ``lane<K>`` and ``combine<K>`` (`leaf_bound` of the rebuilt window Program), lowered by
    `lower_leaf` unless `kept` (the manifest so far) records the file under the same bind, each leaf Program's
    descriptor written beside it (`drawn_rows.LEAF_PROGRAM`): ``files`` (each file's SHA-512 and ``leaf_programs``), each
    statement's bind (``leaf_binds``) and Definition id (``templates``). ValueError when credit derives another leaf
    Program or partition at K, or refuses the K (`credit.lanes_shape`)."""
    from verity.protocols.pouw.verifier import credit as CR
    from verity.protocols.pouw.verifier import leaf_lanes as LL

    from tests.protocols.pouw_h1_rehearsal import PROGRAM
    from verity_vllm.protocols.pouw.h1 import drawn_rows as D
    layout, _why = CR.layout_of(descriptor)
    shape, win = CR.shape_of(descriptor), rebuilt(descriptor)
    files: dict[str, Any] = {CR.LEAF_PROGRAMS: {}}
    binds: dict[str, dict] = {}
    templates: dict[str, str] = {}
    for k in ks:
        if why := CR.lanes_shape(layout, shape, k):
            raise ValueError(f"K = {k}: {why}")
        for kind in (LL.LANE, LL.COMBINE):
            name = f"{kind}{k}"
            defn, desc, part, bind = leaf_bound(win, k, kind)
            unit = CR.leaf_unit_id(shape, k, kind)
            if not (unit and CR.leaf_of(desc, unit, k, kind) and CR.leaf_partition_of(part["program"], unit) == part):
                raise ValueError(f"{name}: core's credit derives another leaf Program or partition at K = {k}")
            (out / D.LEAF_PROGRAM.format(name=name)).write_bytes(desc)
            path = out / leaf_file(kind, k)
            if not (kept.get("leaf_binds", {}).get(name) == bind and path.is_file()):
                lower_leaf(k, kind, program_dir / PROGRAM, path)
            files[name] = hashlib.sha512(path.read_bytes()).hexdigest()
            files[CR.LEAF_PROGRAMS][name] = hashlib.sha512(desc).hexdigest()
            binds[name], templates[name] = bind, defn.id
    return files, binds, templates


def build(program_dir: Path, out: Path, lower: Lower = lower, live: tuple[int, int] | None = None,
          lower_live: LowerLive = lower_live, only: Sequence[int] | None = None, lanes: Sequence[int] = (),
          lower_leaf: LowerLeaf = lower_leaf) -> dict[str, Any]:
    """Each stratum's circuit file of the declared window Program in `program_dir` (the rehearsal's ``PROGRAM`` and
    ``PROGRAM_META``), lowered by `lower` into `out` (`file_name`) unless `out`'s `MANIFEST` already records it under
    the same bind, and the manifest: ``{program, bind, templates, files}``, ``files`` the period-open's
    ``--circuit-files`` (``{k<K>: SHA-512 of K's file}``). With `live` (T, S) the live template's file too (`LIVE_FILE`,
    by `lower_live`, kept under the same ``live_bind``), ``files["live"]`` its SHA-512. A J10-R stratum is lowered as
    J10-R's statement (`lower`'s ``j10r``). Each K of `lanes` is committed in the lanes form instead (`leaf_files`, by
    `lower_leaf`, kept under the same ``leaf_binds``)."""
    from tests.protocols.pouw_h1_rehearsal import PROGRAM, PROGRAM_META
    meta = json.loads((program_dir / PROGRAM_META).read_text())
    descriptor = gzip.decompress((program_dir / PROGRAM).read_bytes())
    bind, ids, j10r = bind_of(descriptor, meta["layout"])
    out.mkdir(parents=True, exist_ok=True)
    try:
        kept = json.loads((out / MANIFEST).read_text())
    except (OSError, ValueError):
        kept = {}
    files, extra = {}, {}
    if live is not None:
        from verity.protocols.pouw.verifier import credit as CR
        from verity.protocols.spot.verifier.one_stage import partition as P
        T, S = live
        part = live_part(T, S, CR._prefix(descriptor))
        extra["live_bind"] = {"T": T, "S": S, "partition": P.digest(part), "program": part["program"]}
        path = out / LIVE_FILE
        if not (kept.get("live_bind") == extra["live_bind"] and path.is_file()):
            lower_live(T, part, path)
        files[CR.LIVE] = hashlib.sha512(path.read_bytes()).hexdigest()
    for k in strata(meta["layout"]):
        if (only is not None and k not in only) or k in lanes:
            continue
        path = out / file_name(k)
        if not (kept.get("bind") == bind and path.is_file()):
            lower(k, tuple(meta["layout"]["tile"]), bind, path, **({"j10r": True} if k in j10r else {}))
        files[f"k{k}"] = hashlib.sha512(path.read_bytes()).hexdigest()
    templates = {f"k{k}": i for k, i in ids.items() if k not in lanes}
    if lanes:
        got, extra["leaf_binds"], leaf_ids = leaf_files(program_dir, out, descriptor, sorted(set(lanes)), kept, lower_leaf)
        files.update(got)
        templates.update(leaf_ids)
    doc = {"program": bind["program"], "bind": bind, "templates": templates, "files": files, **extra}
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
    ap.add_argument("--lanes", metavar="K,...", type=lambda v: tuple(int(x) for x in v.split(",")), default=(),
                    help="commit these K in the lanes form: lane<K> and combine<K> files, each leaf statement's")
    ap.add_argument("--lower", nargs=5, metavar=("K", "R", "C", "BIND", "PATH"), help=argparse.SUPPRESS)
    ap.add_argument("--lower-leaf", nargs=4, metavar=("K", "KIND", "PROGRAM", "PATH"), help=argparse.SUPPRESS)
    ap.add_argument("--j10r", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.lower:
        k, r, c, bind, path = a.lower
        _lower(int(k), int(r), int(c), json.loads(bind), Path(path), a.j10r)
        return 0
    if a.lower_leaf:
        k, kind, program, path = a.lower_leaf
        _lower_leaf(int(k), kind, Path(program), Path(path))
        return 0
    if not (a.program and a.out):
        ap.error("--program and --out")
    doc = build(Path(a.program), Path(a.out), live=a.live, only=a.strata, lanes=a.lanes)
    print(f"POUW-H1-CIRCUITS program={doc['program'][:16]} files={json.dumps(doc['files'])}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
