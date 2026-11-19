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
from verity.circuits.codec import DecodedProgram, definition_digest, descriptor_id
from verity.circuits.program import input_id

from .certified_rows import _fields

PREFIX = "one-circuit: not derived yet: "

RESTRICT = "restrict : C -> Partition -> Block -> Circuit (C_S, the lowering of the block's Definition in C)"

LOWER = "lower : C_S of a partition that is no template query (Q_template_instance, Q_template_instances) -> its Boolean block circuit"

#: Lean's lowering of C_S (`Flock.Lower.lower`, `Flock.Compose.compose`), a Lake target of Flock's verifier.
LOWER_EXE = "flock-lower-check"

#: The templated statement's layout mode (`Flock.Tmpl`'s `iv/hybrid/T`) the door lowers C_S at.
LAYOUT = "iv/hybrid/4"

Refused = R.Refused


class NotDerived(ValueError):
    """A derivation from C and the partition that is not written yet (`Flock.OneCircuit.notDerived`)."""

    def __init__(self, what: str) -> None:
        super().__init__(PREFIX + what)


class Decoded:
    """C's bytes, decoded at most once (`restrict.program`) for the door's steps on that C."""

    def __init__(self, c: bytes) -> None:
        self.c, self._D = c, None

    def get(self) -> DecodedProgram:
        if self._D is None:
            self._D = R.program(self.c)
        return self._D


def block(c: bytes, partition: Mapping[str, Any], k: int, decoded: Decoded | None = None) -> list[tuple[int, int]]:
    """Proof block ``k``'s units, the ``k``-th pair of the partition object's ``blocks``, which cover every unit of C's
    population exactly once (`partition_object.blocks`); `Refused` where the object is not valid, names no blocks, its
    blocks end short of or past the population, or has no block ``k``. ``decoded``: C's, when the caller holds it."""
    v = PO.validate(partition)
    if not v.ok:
        raise Refused("E_PARTITION", f"the partition object is not valid: {list(v.codes)}")
    bs = PO.blocks(partition)
    if bs is None:
        raise Refused("E_BLOCKS", "the partition object names no blocks")
    try:
        population = PO.TemplatePartition((decoded or Decoded(c)).get(), partition["query"]).population
    except PO.QueryInapplicable as e:
        raise Refused("E_QUERY", str(e)) from e
    if bs[-1][1] != population:
        raise Refused("E_BLOCKS", f"the partition's blocks end at {bs[-1][1]}, not at C's population {population}")
    if not (type(k) is int and 0 <= k < len(bs)):
        raise Refused("E_BLOCKS", f"no block {k!r} among the partition's {len(bs)}")
    return [bs[k]]


def restrict(c: bytes, partition: Mapping[str, Any], units: Sequence[tuple[int, int]],
             decoded: Decoded | None = None) -> bytes:
    """C_S (`OneStageZk.restrict`): the descriptor bytes of C compacted to the block of ``units`` (ascending ``[lo, hi)``
    ranges of the partition's units), Core's `Circuit.restrict` (`restrict.restrict`); `Refused` where C, the partition
    or the units give no block."""
    return R.restrict(c, partition, [tuple(u) for u in units], D=(decoded or Decoded(c)).get()).descriptor


def _listed(query: Mapping[str, Any]) -> list[Any] | None:
    """The templates a template query lists; None for another query."""
    params = query.get("params") or {}
    if query.get("name") == "Q_template_instance":
        return [params.get("template")]
    if query.get("name") == "Q_template_instances" and isinstance(params.get("templates"), list):
        return list(params["templates"])
    return None


def lower(c_s: bytes, partition: Mapping[str, Any], exe: Callable[[], Path | str], layout: str = LAYOUT) -> bytes:
    """C_S's Boolean block circuit: Lean's lowering (`Flock.Lower.lower`, then `Flock.Compose.compose`),
    `flock-lower-check compose`, the executable ``exe()`` builds, on C_S's template (the one Definition its root calls, which
    the partition's template query lists): its input ports
    the template's parameters as bit rows, in order, and one output port a return field, ``o{j}``, its bits in whole u16
    words, each a field's tree (`Flock.Registered.outReads`); META binds the partition, C's digest and the template's
    `definition_digest`. `NotDerived` where the partition is no template query; `Refused` where C_S's root calls other
    than one Definition, one the query does not list, or the lowering refuses."""
    listed = _listed(partition.get("query") or {})
    if listed is None:
        raise NotDerived(LOWER)
    D = R.program(c_s)
    called = sorted({descriptor_id(nd.fn) for nd in D.root.body.nodes if input_id(nd.fn.id) is None})
    if len(called) != 1:
        raise Refused("E_LOWER", f"C_S's root calls {called}, and a block's units are instances of one template")
    key = called[0]
    if key not in listed:
        raise Refused("E_LOWER", f"C_S's template {key!r} is not one the partition's query lists")
    fn = D.defs.get(key)
    if fn is None:
        raise Refused("E_LOWER", f"C_S's template {key!r} is no Definition of C_S")
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
