"""The one circuit at the served verifier's door: C_S, which the door derives from C (the registered Program), the
partition and the block's units (`restrict`), its Boolean block circuit, Lean's lowering of C_S (`lower`), and the holes
it does not yet derive, each refusing by name (`NotDerived`)."""

from __future__ import annotations

import json
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from verity.circuits import partition_object as PO
from verity.circuits import restrict as R
from verity.circuits.codec import definition_digest

from .certified_rows import _fields

PREFIX = "one-circuit: not derived yet: "

RESTRICT = "restrict : C -> Partition -> Block -> Circuit (C_S, the lowering of the block's Definition in C)"

LOWER = "lower : C_S of a partition that names no one template (Q_template_instance) -> its Boolean block circuit"

#: Lean's lowering of C_S (`Flock.Lower.lower`, `Flock.Compose.compose`), a Lake target of Flock's verifier.
LOWER_EXE = "flock-lower-check"

#: The templated statement's layout mode (`Flock.Tmpl`'s `iv/hybrid/T`) the door lowers C_S at.
LAYOUT = "iv/hybrid/4"

Refused = R.Refused


class NotDerived(ValueError):
    """A derivation from C and the partition that is not written yet (`Flock.OneCircuit.notDerived`)."""

    def __init__(self, what: str) -> None:
        super().__init__(PREFIX + what)


def block(c: bytes, partition: Mapping[str, Any], k: int) -> list[tuple[int, int]]:
    """Proof block ``k``'s units, the ``k``-th pair of the partition object's one level of ``blocks``, which cover every unit
    of C's population exactly once (`partition_object.blocks`); `Refused` where the object is not valid, names no blocks or
    more than one level (`blocks-levels`), its blocks end short of or past the population, or has no block ``k``."""
    v = PO.validate(partition)
    if not v.ok:
        raise Refused("E_PARTITION", f"the partition object is not valid: {list(v.codes)}")
    try:
        bs = PO.blocks(partition)
    except ValueError as e:
        raise Refused("E_BLOCKS", str(e)) from e
    if bs is None:
        raise Refused("E_BLOCKS", "the partition object names no blocks")
    try:
        population = PO.TemplatePartition(R.program(c), partition["query"]).population
    except PO.QueryInapplicable as e:
        raise Refused("E_QUERY", str(e)) from e
    if bs[-1][1] != population:
        raise Refused("E_BLOCKS", f"the partition's blocks end at {bs[-1][1]}, not at C's population {population}")
    if not (type(k) is int and 0 <= k < len(bs)):
        raise Refused("E_BLOCKS", f"no block {k!r} among the partition's {len(bs)}")
    return [bs[k]]


def restrict(c: bytes, partition: Mapping[str, Any], units: Sequence[tuple[int, int]]) -> bytes:
    """C_S (`OneStageZk.restrict`): the descriptor bytes of C compacted to the block of ``units`` (ascending ``[lo, hi)``
    ranges of the partition's units), Core's `Circuit.restrict` (`restrict.restrict`); `Refused` where C, the partition
    or the units give no block."""
    return R.restrict(c, partition, [tuple(u) for u in units]).descriptor


def lower(c_s: bytes, partition: Mapping[str, Any], exe: Callable[[], Path | str], layout: str = LAYOUT) -> bytes:
    """C_S's Boolean block circuit: Lean's lowering (`Flock.Lower.lower`, then `Flock.Compose.compose`),
    `flock-lower-check compose`, the executable ``exe()`` builds, on C_S's template (the partition's ``Q_template_instance`` template): its input ports
    the template's parameters as bit rows, in order, and one output port a return field, ``o{j}``, its bits in whole u16
    words, each a field's tree (`Flock.Registered.outReads`); META binds the partition, C's digest and the template's
    `definition_digest`. `NotDerived` where the partition names no one template; `Refused` where C_S defines no
    such template, or the lowering refuses."""
    query = partition.get("query") or {}
    key = (query.get("params") or {}).get("template") if query.get("name") == "Q_template_instance" else None
    if not isinstance(key, str):
        raise NotDerived(LOWER)
    D = R.program(c_s)
    fn = D.defs.get(key)
    if fn is None:
        raise Refused("E_LOWER", f"the partition's template {key!r} is no Definition of C_S")
    params, ret = fn.signature()
    ins = ",".join(f"{name}:{sum(t.leaf_widths())}" for name, t in params)
    outs = ",".join(f"o{j}:{-(-sum(f.leaf_widths()) // 16)}" for j, f in enumerate(_fields(ret)))
    digests = json.dumps({key: definition_digest(D.desc["definitions"][key], D.desc)})
    with tempfile.TemporaryDirectory(prefix="one-circuit-lower-") as d:
        src, dst = Path(d, "c_s.json"), Path(d, "circuit.txt")
        src.write_bytes(c_s)
        p = subprocess.run([str(exe()), "compose", str(src), key, f"unit/{key}", ins, outs, layout, partition["program"],
                            PO.digest(partition), digests, str(dst)], capture_output=True, text=True)
        if p.returncode != 0:
            raise Refused("E_LOWER", f"{LOWER_EXE} compose ({p.returncode}): {(p.stderr or p.stdout)[-2000:]}")
        return dst.read_bytes()
