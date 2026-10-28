"""The registration log NCI's full check reads (`nci-check INSTANCE TRUSTED`, `NciCheck.Baseline`), for a Program whose θ is
registered (`RegisteredInput<w>_v1`, `WorkloadProgram(theta_registered=True)`).

    write(program_digest, params, theta, values, window) -> {"record", "digest", "trusted", "provenance", "salts"}
    record(program_digest, params, theta, entries, window) -> {"record", "digest", "trusted", "provenance"}

`record` takes each θ parameter's entry as `registered_rows.commit` made it, where the values were committed elsewhere (one
process per rank of a node Program: `predict.node_registration`), and `write` commits `values` here.

* **The record.** One `verity/registration/v2` record, `previous` null, whose `registered` list has one value per parameter
  that holds a θ node, named as the parameter, its rows committed under hiding leaves (`commit.registered_rows`, R7's
  format). One log is one deployment and θ is what its first record registers, so this record opens the log and is the
  run's own: the deployment's first run. Its keys are the ones nci-check reads (`program`, `registered`, `previous`) with
  `protocol`, `window` and `roots`, as NCI's difftest vectors write them. It carries none of the one-stage draw's fields
  (query, partition, population, law, scheme, leaf layer, anchors): it registers θ, and no sample follows it.
* **`provenance`**: the instance's, one `{"kind": "registered", "registration", "value", "root"}` per θ node, in θ's order.
* **`trusted`**: `{"log": [record], "run": digest, "params": [[name, type], ...]}`, the types as the descriptor writes them.
* **`salts`**: per value its rows' salts, the prover's, which nothing public carries.

`params` are the Program function's parameters in order (§5.1: the root starts with each parameter's input nodes, in
parameter order), `theta` the root input nodes `annotations["theta"]` names, and `values(name)` a θ parameter's leaves in
its leaf order. A θ parameter is one leaf width (`registered.width`) or it is refused: its rows' `word_bits` is the width of
every θ node in it (nci-check's `theta-width`).
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np

from verity.core.protocols.sampled_proofs.one_stage import registration as REG
from verity.ml.circuits.types import Array, Record, Tuple, Type
from verity_vllm.commit import registered_rows as RR


def lower(t: Type) -> list[tuple[int, int]]:
    """The input nodes a parameter of type `t` lowers to, `(width, leaves)` each (PROTOCOL.md §5.1, as nci-check lowers
    `params`)."""
    ws = t.width_set()
    if len(ws) == 1:
        return [(next(iter(ws)), t.leaves)]
    if isinstance(t, Array):
        return lower(t.elem) * t.n
    if isinstance(t, Tuple):
        return [x for f in t.fields for x in lower(f)]
    if isinstance(t, Record):
        return [x for _, f in t.fields for x in lower(f)]
    raise ValueError(f"a parameter of type {t!r} lowers to no input nodes")


def theta_parameters(params: Sequence[tuple[str, Type]], theta: Sequence[int]) -> dict[int, tuple[str, int]]:
    """Root input node -> (parameter, leaf width) for every θ node; refused unless each θ parameter is one width."""
    owner = [(i, w) for i, (_, t) in enumerate(params) for w, _n in lower(t)]
    out: dict[int, tuple[str, int]] = {}
    for k in theta:
        if not 0 <= k < len(owner):
            raise ValueError(f"θ node {k} is no input node of the parameters' {len(owner)}")
        i, w = owner[k]
        name, t = params[i]
        if len(t.width_set()) != 1:
            raise ValueError(f"θ parameter {name!r} is of more than one leaf width ({sorted(t.width_set())}): R7 registers it as "
                             "one value of one width, so split it into one parameter per width")
        out[k] = (name, w)
    return out


def word_bits(t: Type) -> int:
    """The `word_bits` a θ parameter of type `t` is registered at: its one leaf width, rounded up to whole u16 words."""
    return 16 * -(-next(iter(t.width_set())) // 16)


def record(program_digest: str, params: Sequence[tuple[str, Type]], theta: Sequence[int],
           entries: Mapping[str, Mapping[str, Any]], window: Mapping[str, Any]) -> dict[str, Any]:
    """The record of θ's `entries` (θ parameter -> its `registered_rows.commit` entry), its digest, nci-check's trusted input and
    the instance's `provenance`: refused unless there is one entry per θ parameter, each of its parameter's leaves and width."""
    nodes = theta_parameters(params, theta)
    types = dict(params)
    names = list(dict.fromkeys(n for n, _w in nodes.values()))
    if set(entries) != set(names):
        raise ValueError(f"entries for {sorted(set(entries) - set(names))[:4]} are no θ parameter's, and θ parameters "
                         f"{sorted(set(names) - set(entries))[:4]} have none")
    for name in names:
        e, t = entries[name], types[name]
        if e.get("value") != name or e["leaves"] * e["words"] != t.leaves or e.get("word_bits", 16) != word_bits(t):
            raise ValueError(f"θ parameter {name!r}: its entry registers {e.get('value')!r}, {e['leaves']} rows of {e['words']} "
                             f"{e.get('word_bits', 16)}-bit leaves, its type is {t.leaves} of {word_bits(t)}")
    rec = {"format": REG.FORMAT_V2, "protocol": REG.PROTOCOL, "program": program_digest, "window": dict(window),
           "registered": [dict(entries[n]) for n in names], "roots": [], "previous": None}
    d = REG.digest(rec)
    return {"record": rec, "digest": d,
            "trusted": {"log": [rec], "run": d, "params": [[n, t.to_json()] for n, t in params]},
            "provenance": [{"kind": "registered", "registration": d, "value": nodes[k][0], "root": entries[nodes[k][0]]["root"]}
                           for k in theta]}


def write(program_digest: str, params: Sequence[tuple[str, Type]], theta: Sequence[int],
          values: Callable[[str], np.ndarray], window: Mapping[str, Any]) -> dict[str, Any]:
    """The record, its digest, nci-check's trusted input and the instance's `provenance`, and the prover's salts."""
    types = dict(params)
    entries, salts = {}, {}
    for name in dict.fromkeys(n for n, _w in theta_parameters(params, theta).values()):
        leaves = np.asarray(values(name))
        if leaves.size != types[name].leaves:
            raise ValueError(f"θ parameter {name!r}: {leaves.size} leaves given, its type holds {types[name].leaves}")
        c = RR.commit(program_digest, name, leaves, word_bits(types[name]))
        entries[name], salts[name] = c.entry, c.salts
    return record(program_digest, params, theta, entries, window) | {"salts": salts}
