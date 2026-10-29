"""The window Program a served H1 window declares before it serves (`declared`), the layout it is declared over
(`layout`, `declared_layout`), its tile unit (`unit_of`), the bytes the period commits of it (`descriptor`, written by
`write_program`), and the card it is declared for (`card_arch`): `pouw_h1_served.Served` declares one per period and every window of the period
serves it."""
from __future__ import annotations

import functools
import gc
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from verity.core.protocols.pouw.circuit import h1device as HD
from verity.core.protocols.pouw.circuit import h1tile as HT
from verity.core.protocols.pouw.circuit import h1window as HW
from verity.ml.circuits.codec import canonical_json, descriptor_chunks, encode_program
from verity.ml.circuits.program import Program

from verity_vllm.protocols.pouw import h1 as PH

#: the window Program's parameter prefix, the same for every window of a period
PREFIX = "h1/"


def layout(shapes: Sequence[tuple[int, int]], calls: Sequence[tuple[int, int]],
           blocks: Sequence[int] | None = None) -> list[int]:
    """Each call's A block of its K, ``calls`` each call's (step, linear) and ``blocks`` each linear's input block within a
    step (`Served`'s): call (s, w) reads block s·B_K + i, i the place of ``blocks[w]`` among the blocks K's linears read."""
    per = list(blocks) if blocks is not None else list(range(len(shapes)))
    if len(per) != len(shapes):
        raise ValueError(f"blocks {per} are not one input block for each of the window's {len(shapes)} linears")
    of_k: dict[int, list[int]] = {}
    for w, (_n, k) in enumerate(shapes):
        of_k.setdefault(k, [])
        if per[w] not in of_k[k]:
            of_k[k].append(per[w])
    of_k = {k: sorted(b) for k, b in of_k.items()}
    return [s * len(of_k[shapes[w][1]]) + of_k[shapes[w][1]].index(per[w]) for s, w in calls]


def card_arch(device: Any = None) -> str:
    """The architecture of the CUDA card ``device`` (the current one when None) as H1's device records name it
    (`h1device.DEVICES`): an H100 or H200's, compute capability 9.0, or the RTX PRO 6000's, 12.0. ValueError on a card
    with no record."""
    import torch

    major, minor = torch.cuda.get_device_capability(device)
    return HD.device(f"sm_{major}{minor}").arch


def declared(shapes: Sequence[tuple[int, int]], modules: Mapping[int, str], rows: int, steps: int,
             tile: tuple[int, int] = PH.TILE, blocks: Sequence[int] | None = None, prefix: str = PREFIX,
             unit: Callable[[int], Any] | None = None, arch: str | None = None) -> Program:
    """The window Program a static window declares before it serves: ``steps`` steps, each calling every linear (``shapes``,
    each ``(n, K)``, named by ``modules``) in order, laid out by `layout`, its A slot for ``rows`` token rows, its
    parameters under ``prefix`` (`PREFIX`, which names no window): one Program for every window of the declaration, the
    period's ``--circuit`` (its `program_digest`) and each window's `Served.program`. Without ``unit``, each K's tile
    unit is `h1window.tile_unit`'s on ``arch`` (`card_arch`; sm_120 when None), which refuses a K outside that card's
    record's domain (ValueError); sm_90's Program is sm_120's byte for byte."""
    lay = declared_layout(shapes, modules, rows, steps, tile, blocks)
    if unit is None and arch is not None:
        unit = functools.partial(HW.tile_unit, tile=tuple(tile), arch=arch)
    return HW.program([tuple(x) for x in lay["linears"]], [tuple(x) for x in lay["calls"]], lay["m"], tile, prefix, unit)


def unit_of(name: str | None, tile: tuple[int, int] = PH.TILE) -> Callable[[int], Any] | None:
    """`declared`'s ``unit`` for the pouw setting ``unit`` ``name`` (`pouw_h1_settings.UNITS`): None for v1 (unset), the
    default; for ``j10r`` `h1window.tile_unit` over J10-R's `h1tile.PouwH1Tile5`, one noise stream a call, the tile a
    period's guarantee covers (core's `period_check.COVERED_TILE`)."""
    if name in (None, "v1"):
        return None
    if name == "j10r":
        return functools.partial(HW.tile_unit, tile=tuple(tile), definition=HT.PouwH1Tile5)
    raise ValueError(f"no tile unit {name!r}: the pouw setting 'unit' is v1 or j10r")


def descriptor(program: Program) -> bytes:
    """`program`'s descriptor as the period commits it: its canonical JSON without annotations, the bytes whose SHA-512
    is its `codec.program_digest` (the period-open's ``circuit``) and from which core's credit reads the window's
    layout (`credit.layout_of`)."""
    return canonical_json({k: v for k, v in encode_program(program).items() if k != "annotations"})


def write_program(program: Program, layout: Mapping[str, Any], out: Path | str, *, rows: int, steps: int) -> str:
    """`program`'s `codec.program_digest`, streamed once into ``out``'s `pouw_h1.PROGRAM` (its `descriptor`, gzipped) as it
    is hashed, and its `pouw_h1.PROGRAM_META`: the digest, the descriptor's bytes, its `layout` (`declared_layout`, which
    core's `credit.layout_of` reads from it), T ``rows`` and S ``steps``. A period-open commits the digest and the circuit
    files built from these (``served_verdicts.py period``)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    h, size, on = hashlib.sha512(), 0, gc.isenabled()
    gc.disable()
    try:
        with gzip.open(out / PH.PROGRAM, "wb", compresslevel=1) as f:
            for chunk in descriptor_chunks(program):
                h.update(chunk)
                f.write(chunk)
                size += len(chunk)
    finally:
        if on:
            gc.enable()
    digest = h.hexdigest()
    meta = {"program_digest": digest, "file_sha512": digest, "bytes": size, "rows": int(rows), "steps": int(steps),
            "layout": dict(layout)}
    (out / PH.PROGRAM_META).write_text(json.dumps(meta, indent=1) + "\n")
    return digest


def declared_layout(shapes: Sequence[tuple[int, int]], modules: Mapping[int, str], rows: int, steps: int,
                    tile: tuple[int, int] = PH.TILE, blocks: Sequence[int] | None = None) -> dict[str, Any]:
    """`declared`'s arguments to `h1window.program` as the window's layout, ``{calls: [[linear, A block], ...],
    linears: [[name, K, n], ...], m, tile}``: what core's `credit.layout_of` reads from `declared`'s `descriptor`, built
    without the Program (the proof system sizes the period's K from it, `pouw_h1_challenger.draw_k`)."""
    mods = [modules.get(w, str(w)) for w in range(len(shapes))]
    calls = [(s, w) for s in range(steps) for w in range(len(shapes))]
    return {"calls": [[mods[w], b] for (_s, w), b in zip(calls, layout(shapes, calls, blocks))],
            "linears": [[mods[w], k, n] for w, (n, k) in enumerate(shapes)], "m": HW.slot(rows, tile), "tile": list(tile)}


__all__ = ["PREFIX", "card_arch", "declared", "declared_layout", "descriptor", "layout", "unit_of", "write_program"]
