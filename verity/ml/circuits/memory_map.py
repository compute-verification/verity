"""memory-map/v1: per step, the values a Program's execution holds and the reads it moves, from its census (`census`).

A memory map is a pure function of a census/v1: the census's values are the Program's dataflow across its root's Calls (a KV
cache row is a value a Call of one step produces and Calls of later steps read), and its steps say when each Call runs. Written as
canonical JSON under a SHA-512 digest, like the census.

A value is *born* in the step of the Call that produces it. A registered input (`RegisteredInput<w>`: weights, tables) is born
before the first step (-1), a run input (`Input<w>`: a request's data, a coin) in the step before the first step that reads it
(-1 if none does). Its *last read* is the last step with a Call that reads it or has it among a selection's candidates (or its
birth, if none does, an output included: an output is emitted when it is born). A value is *held* in step `s` when
`birth < s <= last_read`: it has to be kept from an earlier step into `s`. A held value read directly in `s` is a *moved* read, once
per value and step. Selections are data-dependent, so they are reported apart: they keep their candidates held, and move what they
return (`select_bits`), never their candidates.

    {"schema": "verity/memory-map/v1",
     "census": <the census's digest>, "program": <the census's program>,
     "steps": {"count": S,
               "held": {"computed" | "registered" | "run": [<bits held in s>]},
               "moved": {"computed" | "registered" | "run": [<bits of held values read directly in s>]},
               "born": [<bits produced in s>], "read": [<bits of the values read directly in s, held or not>],
               "select_gates": [...], "select_bits": [...]},
     "values": {"birth": [...], "last_read": [...], "read_steps": [<steps after its birth that read it directly>],
                "moved_bits": [<bits x read_steps>]},
     "totals": {"peak_held_bits", "peak_step", "held_bit_steps", "moved_bits", "born_bits", "select_gates", "select_bits"},
     "digest": <SHA-512 of the canonical JSON of every other field>}

`values` is aligned with the census's `values` (its bits, producer, reads and output are there). `held_bit_steps` sums bits times
the steps a value is held, `peak_held_bits` is the largest step's held bits and `peak_step` the first step that holds them (-1 with
no step).
"""

from __future__ import annotations

from .census import SCHEMA as CENSUS, seal, step_of

SCHEMA = "verity/memory-map/v1"
KINDS = ("computed", "registered", "run")


def memory_map(c: dict) -> dict:
    """The memory-map/v1 of a census/v1 document."""
    if c.get("schema") != CENSUS:
        raise ValueError(f"a memory map reads a {CENSUS}, not {c.get('schema')!r}")
    S, I = len(c["steps"]["ends"]), c["inputs"]
    step = step_of(c)
    binding = [c["classes"][c["callees"][i]]["binding"] for i in c["nodes"]["callee"][:I]]
    vals = c["values"]
    V = len(vals["producer"])
    pool_steps: list[set[int]] = [set() for _ in range(V)]
    for k, vs in zip(c["selects"]["call"], c["selects"]["values"]):
        for v in vs:
            pool_steps[v].add(step(k))

    diff = {kind: [0] * (S + 1) for kind in KINDS}
    moved = {kind: [0] * S for kind in KINDS}
    born, read = [0] * S, [0] * S
    out: dict[str, list] = {"birth": [], "last_read": [], "read_steps": [], "moved_bits": []}
    held_bit_steps = 0
    for v in range(V):
        j, bits = vals["producer"][v], vals["bits"][v]
        rs = sorted({step(k) for k in vals["reads"][v]})
        touched = set(rs) | pool_steps[v]
        if j >= I:
            kind, birth = "computed", step(j)
            born[birth] += bits
        else:
            kind = "registered" if binding[j] == "registration" else "run"
            birth = -1 if kind == "registered" or not touched else min(touched) - 1
        last = max(touched | {birth})
        if last > birth:
            diff[kind][birth + 1] += bits
            diff[kind][last + 1] -= bits
            held_bit_steps += bits * (last - birth)
        after = [s for s in rs if s > birth]
        for s in rs:
            read[s] += bits
        for s in after:
            moved[kind][s] += bits
        out["birth"].append(birth)
        out["last_read"].append(last)
        out["read_steps"].append(len(after))
        out["moved_bits"].append(bits * len(after))

    held = {}
    for kind in KINDS:
        run, col = 0, []
        for s in range(S):
            run += diff[kind][s]
            col.append(run)
        held[kind] = col
    per_step = [sum(held[kind][s] for kind in KINDS) for s in range(S)]
    peak = max(per_step, default=0)
    cs = c["steps"]
    return seal({"schema": SCHEMA, "census": c["digest"], "program": c["program"],
                 "steps": {"count": S, "held": held, "moved": moved, "born": born, "read": read,
                           "select_gates": cs["select_gates"], "select_bits": cs["select_bits"]},
                 "values": out,
                 "totals": {"peak_held_bits": peak, "peak_step": per_step.index(peak) if S else -1, "held_bit_steps": held_bit_steps,
                            "moved_bits": sum(out["moved_bits"]), "born_bits": sum(born),
                            "select_gates": sum(cs["select_gates"]), "select_bits": sum(cs["select_bits"])}})


__all__ = ["SCHEMA", "KINDS", "memory_map"]
