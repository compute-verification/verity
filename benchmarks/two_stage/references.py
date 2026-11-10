"""The auditor's reference statements of a two-stage design's interior window (`service.template_circuits`' ``proofs``):
each template's circuit file, and its reference statement's header and registered reads, which round 2 holds each of the
template's sessions to once it places the header in the run (`second_stage.at_run`).

A reference is `member_stager`'s statement of the template's member, bound to the window's partition and program as the
sessions' circuits are, on zero rows at the member's slots, each port's table and refs as the window lays them out, salted
under a key of the design's own; its prover file is deleted. Staged from a window of the design built at a run of its own,
with no replay sample's values, both files are a function of Q and the design alone, byte for byte: nothing the first
stage draws reaches them (`toy.references`).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from verity.core.protocols.spot.one_stage import registered as REG
from verity.ml.windows import links as LK, rows as CS


def stage(window: CS.Binding, definitions: Mapping[str, Any], links: Mapping[str, Mapping[str, str]], out: Path,
          key: bytes) -> dict[str, list[dict[str, Any]]]:
    """The auditor's params for ``window``, a template a member in the members' order: its circuit file's SHA-512, its
    reference statement's header and its registered reads, with ``circuit.txt`` and ``pub.bin`` under ``out``/m<k>.
    ``definitions``: each template's bound Definition, by descriptor id. ``links``: by member key, each link port's value,
    the name its link reads (the boundary port it opens, or the value the interior registers under the link's name); every
    other port reads the port that holds it (`Binding.held`: a wired port its writer's output port, any other its own).
    ``key``: the stand-in rows' salt key. Raises `ValueError` for a template split into chunks (round 2 places one
    reference a template in each of its parts, which this hasn't been checked against), a member whose template
    ``definitions`` doesn't name, or a link port its member doesn't have."""
    from verity_flock.class_statement import member_stager

    keys = [m.key for m in window.members]
    if len({m.template for m in window.members}) != len(keys):
        raise ValueError(f"a template is split into chunks ({keys}): a reference is staged for one member a template")
    bind = {"partition": window.partition_digest, "program": window.partition["program"]}
    named = []
    for k, m in enumerate(window.members):
        if m.template not in definitions:
            raise ValueError(f"member {m.key}: no Definition is given for its template")
        linked, ports = dict(links.get(m.key, {})), dict(m.all_rows)
        if not linked.keys() <= ports.keys():
            raise ValueError(f"member {m.key}: link ports {sorted(linked.keys() - ports.keys())} are none of its ports "
                             f"{sorted(ports)}")
        d, rows = Path(out) / f"m{k}", {r: np.zeros((m.n, w), dtype=np.uint16) for r, w in m.rows}
        member_stager(definitions[m.template], m, bind).stage(
            rows, d, key, indices=range(m.base, m.base + m.n),
            tables={r: np.zeros((dict(m.tables)[r], w), dtype=np.uint16) for r, w in m.rows},
            refs={r: np.asarray(window.held_leaves(m, r)) for r in rows})
        (d / "inst.bin").unlink()
        with open(d / "pub.bin", "rb") as f:
            header = json.loads(f.readline())
        reads = LK.reads_file({p: {"value": linked[p] if p in linked else window.held(m, p), "words": int(words),
                                   "word_bits": REG.WORD_BITS, REG.SCHEMA_KEY: m.schema(p)}
                               for p, words in m.all_rows}, registered=REG)
        named.append({"template": m.template, "circuit": header["circuit_sha512"], "header": header, "registered": reads})
    return {"templates": named}
