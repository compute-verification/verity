"""The class this tree builds, for `run.py`: its circuit record and its inference-only policy by the integration's code
(`verity_vllm`), the record's three divisions, and the Lean count on it.

* **The class** is a pinned checkpoint's class on the H200 with its int8 linears inside the circuit
  (`registered`): the class circuit `window_circuit.class_circuit` lowers, as its record (`record`), θ its registered
  weights.
* **The policy** (`policy`): a block is one slot's session (`block_counts.session`), `c` its Boolean gates by the gate
  table (`block_counts.gates`, as the Lean count counts them) and `k` its fresh bits by the record's description
  (`block_counts.fresh`). One block's work is the class's fixed load priced as a served step is (`block_work`).
* **The divisions** (`divisions`) are the three the record carries: the session (a slot), the position (a slot at a
  step, the record's inference block) and the proof block (a unit). A block's gates are its units' templates'; its fresh
  bits are the rows it reads of registered parameters outside θ, once a row, and the output rows it reads of units in
  another block, once a row. A division's policy is its largest block's `c` and `k`.
* **The check** (`lean_count`) is the Lean count, the package's check executable with `--count --detail` (`EXECUTABLE`,
  built with `lake`), on the session
  Program the integration gives it: the session Definition activated once a slot, with no θ named and no wire between
  units, since the record's wires are declarations the Definition doesn't carry.
"""
from __future__ import annotations

import itertools
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import tomllib
from collections import defaultdict
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

import policy as P

# a served step's pricing, `block_work`'s
sys.path.append(str(P.ROOT / "benchmarks" / "one_stage"))

#: the pinned exact-int8 checkpoint of Llama 3.2 1B
MODEL = "verity-int8/Llama-3.2-1B-INT8"
#: the card the class is registered for, serving an exact-int8 checkpoint's linears int8
H200 = {"compute_capability": (9, 0), "num_sms": 132, "int8_exact": "linears"}
DIVISIONS = ("session", "position", "proof block")
#: the inference-only policy's check, a Lake package
PACKAGE = P.ROOT / "verity/protocols/inference_only/verifier/nci/lean"


def _declared() -> tuple[tuple[str, ...], str]:
    """The count as its package declares it (`lakefile.toml`): the command `lean_count` runs, the package's check
    executable with `--count --detail`, and the rule, its library's `Count` module deciding `CompliesAtMost c k n`."""
    lake = tomllib.loads((PACKAGE / "lakefile.toml").read_text())
    (lib,) = (x["name"] for x in lake["lean_lib"])
    (exe,) = (x["name"] for x in lake["lean_exe"] if x["root"] == f"{lib}.CheckMain")
    return (exe, "--count", "--detail"), f"{lib}.Count: CompliesAtMost c k n"


EXECUTABLE, RULE = _declared()
#: what the count's `failure` names, as the policy's parameter
FAILED = {"gates": "c", "fresh": "k", "count": "n"}


class Refused(Exception):
    """The integration lowers no circuit of the class asked for; the message is its own reason."""


def registered(model: str = MODEL) -> Any:
    """What a serve of the pinned checkpoint ``model`` on the H200 registers (`steps.Registered`): its φ at the
    integration's default seed and execution."""
    from verity_vllm import TargetProfile
    from verity_vllm.engine.served_seed import DEFAULT_SEED
    from verity_vllm.program.predict.phi import Phi
    from verity_vllm.program.predict.steps import Registered

    revision, config = P.pinned(model)
    target = TargetProfile.default().with_overrides(**H200).to_json()
    phi = Phi(model={"repo": model, "revision": revision, "hf_config": json.loads(config.read_text())},
              target=dict(target), execution="bi-eager")
    return Registered.of(phi, DEFAULT_SEED)


def record(circuit: Any) -> tuple[dict[str, Any], dict[str, str]]:
    """A circuit's record, as its description and its blocks, and θ as the pins that hold every registered weight."""
    from verity_vllm.check.weights_of_record import PARAM

    doc = {**circuit.description(), "name": circuit.name, "partition": "",
           "blocks": circuit.blocks.doc(sum(circuit.units.values()), "")}
    return doc, {p: "00" for p in circuit.registered.params if p.split("/", 1)[0] == PARAM}


def block_work(cls: tuple[int, int, int], model: str) -> dict[str, int]:
    """One block's work: one slot of the class's fixed load (`window_class.Tables.schedule`) priced as
    `served_shares.step_macs` prices a served step, and its total in the policy's unit (`policy.TOTAL`)."""
    import served_shares as SS
    import served_verdicts as SV
    from verity_vllm.program.predict.window_class import Tables

    if model not in SS.MODELS or model not in SV.DECLARED:
        raise ValueError(f"served_shares.MODELS and served_verdicts.DECLARED hold no shapes for {model!r}, so a block's "
                         f"work isn't priced: add them there (they hold {sorted(SS.MODELS)})")
    macs = dict.fromkeys(P.WORK, 0)
    for rows in Tables.schedule(P.class_of(cls)):
        mine = [r for r in rows if r[0] == 0]
        for kind, v in SS.step_macs({"rows": mine, "emitted": {"0": len(mine)}}, model).items():
            macs[kind] += v
    return {**macs, P.TOTAL: macs["linears"] + 2 * (macs["attention"] + macs["lm_head"])}


def policy(doc: Mapping[str, Any], pins: Mapping[str, str]) -> dict[str, Any]:
    """The session policy of the record ``doc`` under θ = ``pins``: its block Definition and slots, `c` and `k`, the
    gate table and each template's gates a unit."""
    from verity_vllm.program.predict import block_counts as BC

    from verity.definitions.boolean import gate_table

    fn, m, steps = BC.session(doc)
    table, unpriced = gate_table([fn])
    memo: dict = {}
    c, missing = BC.gates(fn, table, memo)
    if unpriced or missing:
        raise ValueError(f"{doc.get('name')}: a block reaches primitives the gate table doesn't price, so its gates "
                         f"aren't known: {sorted({*unpriced, *missing})}")
    by_template = {t: BC.gates(BC.resolve(t), table, memo)[0] for t, _ in doc["templates"]}
    return {"fn": fn, "slots": m, "steps": steps, "c": c, "k": BC.fresh(doc, m, pins, pins), "table": dict(table),
            "gates": by_template}


def _family(tid: str) -> str:
    return re.sub(r"\{.*", "", tid)


def divisions(doc: Mapping[str, Any], pins: Mapping[str, str], gates: Mapping[str, int]) -> dict[str, dict[str, Any]]:
    """The record's three divisions (module docstring): for each, its blocks a session, the largest block's gates and
    the largest block's fresh bits, and what that block's fresh bits are."""
    from verity_vllm.program.predict import block_counts as BC

    from verity.commitments.windows import interface as IF

    units = {t: int(n) for t, n in doc["templates"]}
    lays = {t: IF.layout(BC.resolve(t)) for t in units}
    wires, cuts, reg, ingress = IF.declarations(doc)
    reg = reg or IF.Registered({})
    held = IF.held_params(pins, pins)
    if ingress or any(t.param not in held for t in reg.tables):
        raise ValueError(f"{doc.get('name')}: ingress rows, or a table over a parameter outside θ, which this count of a "
                         "block's reads doesn't take")
    blocks, _ = IF.Blocks.of(doc["blocks"], units, doc["partition"])
    slots, steps = blocks.slots, blocks.steps
    slot, step = {}, {}
    for t, n in units.items():
        slot[t], step[t] = blocks.slots_steps(t, 0, n)
    first = dict(zip(units, itertools.accumulate(units.values(), initial=0)))
    total = sum(units.values())

    def row_bits(tid: str, port: str) -> int:
        return 16 * dict(lays[tid].rows)[port] // int(cuts.get(tid, {}).get(port, 1))

    def reads():
        """Every read a unit makes of a row that may lie outside its block: (the source, who reads it, the reader's
        template and units, the rows read, the template that writes them). A source is a registered parameter outside
        θ (its writer None) or a template's output port: its name, its rows' bits, how many rows it has, and its key."""
        for r in reg.reads:
            if r.param not in held and r.port in dict(lays[r.reader].rows):
                i = np.arange(r.units, dtype=np.int64)
                yield ((f"registered {r.param}", row_bits(r.reader, r.port), leaves[r.param], r.param),
                       f"{_family(r.reader)}.{r.port}", r.reader, r.at + i // r.cut[1], r.source(i), None)
        for w in wires:
            i = np.arange(w.units, dtype=np.int64)
            yield ((f"{_family(w.writer)}.{w.out}", row_bits(w.reader, w.port), units[w.writer], w.writer),
                   f"{_family(w.reader)}.{w.port}", w.reader, w.at + i // w.cut[1], w.source(i), w.writer)

    leaves: dict[str, int] = defaultdict(int)
    for r in reg.reads:
        if r.param not in held:
            leaves[r.param] = max(leaves[r.param], int(np.max(r.source(np.arange(r.units, dtype=np.int64)))) + 1)
    labels = {
        "session": (lambda t: slot[t], slots),
        "position": (lambda t: slot[t] * steps + step[t], slots * steps),
        "proof block": (lambda t: first[t] + np.arange(units[t], dtype=np.int64), total),
    }
    out = {}
    for name, (of, count) in labels.items():
        block = {t: of(t) for t in units}
        c = np.zeros(count, dtype=np.int64)
        for t in units:
            c += np.bincount(block[t], minlength=count) * gates[t]
        # per source, the (block, row) pairs a block reads from outside itself, as block × rows + row: a row read twice
        # in a block counts once
        pairs: dict[tuple, list[np.ndarray]] = defaultdict(list)
        readers: dict[tuple, set[str]] = defaultdict(set)
        for source, who, reader, at, rows, writer in reads():
            mine, rows = block[reader][at], np.asarray(rows, dtype=np.int64)
            if writer is not None:
                outside = mine != block[writer][rows]
                mine, rows = mine[outside], rows[outside]
            if len(mine):
                pairs[source].append(mine * source[2] + rows)
                readers[source].add(who)
        k = np.zeros(count, dtype=np.int64)
        read = {}
        for source, parts in pairs.items():
            read[source] = np.unique(np.concatenate(parts)) // source[2]
            k += np.bincount(read[source], minlength=count) * source[1]
        worst = int(k.argmax())
        kinds: dict[str, int] = defaultdict(int)
        for source, of_block in read.items():
            if rows := int((of_block == worst).sum()):
                kinds[f"{source[0]}, read by {', '.join(sorted(readers[source]))}"] += rows * source[1]
        out[name] = {"blocks_a_session": count // slots if count % slots == 0 else count / slots, "c": int(c.max()),
                     "k": int(k.max()), "k_by_kind": sorted(kinds.items(), key=lambda kv: -kv[1])}
    return out


def instance(pol: Mapping[str, Any], c: int, k: int, n: int) -> dict[str, Any]:
    """The count's instance for the session policy: the session Definition activated once a slot, the gate table
    and the policy; no θ, and each block Definition one block."""
    from verity.circuits import codec
    from verity.circuits.population import population_program

    text = codec.canonical_json(codec.encode_program(population_program(pol["fn"], pol["slots"]))).decode()
    return {"program": text, "theta": [], "blocks": [codec.descriptor_id(pol["fn"])], "table": pol["table"], "c": c,
            "k": k, "n": n}


def lean_count(inst: Mapping[str, Any], clock: Callable[[], float] = time.perf_counter) -> dict[str, Any]:
    """The count (`EXECUTABLE`) on ``inst``, built first with `lake`: its verdict as it printed it, and
    what the run cost (wall and CPU seconds, peak memory, the instance's bytes)."""
    try:
        build = subprocess.run(["lake", "build", EXECUTABLE[0]], cwd=PACKAGE, capture_output=True, text=True)
    except FileNotFoundError:
        raise RuntimeError("`lake` isn't on PATH, so the Lean check can't be built: install the toolchain "
                           f"{(PACKAGE / 'lean-toolchain').read_text().strip()} (elan) and run again") from None
    if build.returncode:
        raise RuntimeError(f"`lake build {EXECUTABLE[0]}` in {PACKAGE.relative_to(P.ROOT)} failed:\n"
                           + (build.stdout + build.stderr)[-2000:])
    with tempfile.TemporaryDirectory() as d:
        path, out = Path(d) / "instance.json", Path(d) / "verdict.json"
        path.write_text(json.dumps(inst))
        with out.open("wb") as fh:
            start = clock()
            child = subprocess.Popen([str(PACKAGE / ".lake/build/bin" / EXECUTABLE[0]), *EXECUTABLE[1:], str(path)],
                                     stdout=fh, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL)
            with child.stderr:
                err = child.stderr.read()
            # the child's own resources, which `wait` doesn't give
            _, status, used = os.wait4(child.pid, 0)
            wall = clock() - start
            child.returncode = os.waitstatus_to_exitcode(status)
        if child.returncode:
            raise RuntimeError(f"{' '.join(EXECUTABLE)} exited {child.returncode}: {err.decode(errors='replace')[-2000:]}")
        return {"verdict": json.loads(out.read_text()),
                "cost": {"wall_s": wall, "cpu_s": used.ru_utime + used.ru_stime, "peak_mib": used.ru_maxrss / 1024,
                         "instance_bytes": path.stat().st_size}}


def judged(got: Mapping[str, Any], c: int, k: int, n: int) -> dict[str, Any]:
    """`lean_count`'s verdict as the policy's check: refused or accepted, and where refused which of c, k and n failed,
    with what the Lean counted against it."""
    v, row = got["verdict"], got["verdict"]["blocks"][0]
    counted = {"c": row[1], "k": row[2], "n": v["total"]}
    check = {"verdict": "accepted" if v["complies"] else "refused", "executable": " ".join(EXECUTABLE), "rule": RULE,
             "complies": v["complies"], "counted": counted, "policy": {"c": c, "k": k, "n": n}, "cost": got["cost"]}
    if not v["complies"]:
        check["failed"] = FAILED[v["failure"][1]]
        check["why"] = {"k": "the registered Program has no θ and no dataflow, so every row a unit reads counts as "
                             "fresh, its weights and its wired rows too",
                        "c": "the session Definition holds more gates than the policy's c",
                        "n": "the Program activates more blocks than the policy's n"}[check["failed"]]
    return check


def build(model: str, cls: tuple[int, int, int], n: int, check: Callable[[Mapping[str, Any]], dict[str, Any]] = lean_count,
          say: Callable[[str], None] = lambda _line: None, circuit: Any = None) -> tuple[P.Policy, dict[str, dict[str, Any]]]:
    """The built class's session policy, checked by ``check``, and its divisions. ``n`` is the blocks the policy
    declares, which nothing here derives. ``say`` gets a line as each step starts; ``circuit`` is the class's where the
    caller lowered it (default: `class_circuit` of what a serve of ``model`` registers)."""
    from verity_vllm.program.predict.common import Unexpressible
    from verity_vllm.program.predict.window_circuit import class_circuit

    work = block_work(cls, model)
    say(f"lowering class {','.join(map(str, cls))} of {model}")
    try:
        circuit = class_circuit(registered(model), P.class_of(cls)) if circuit is None else circuit
    except Unexpressible as e:
        raise Refused(f"the integration lowers no circuit of class {','.join(map(str, cls))} of {model}, so there is "
                      f"nothing to count: {e}") from e
    doc, pins = record(circuit)
    say("the session's Definition, its gate table, c and k")
    pol = policy(doc, pins)
    say("the record's divisions")
    div = divisions(doc, pins, pol["gates"])
    if div["session"]["c"] != pol["c"]:
        raise ValueError(f"{doc.get('name')}: a session's units' templates hold {div['session']['c']:,} gates and its "
                         f"Definition {pol['c']:,}")
    # the policy's k is the integration's count; what a session's own reads leave of it shows as unexplained
    parts = tuple((kind, bits, "the record's reads") for kind, bits in div["session"]["k_by_kind"])
    div["session"]["k"] = pol["k"]
    say(f"{' '.join(EXECUTABLE)} on the session Program")
    verdict = judged(check(instance(pol, pol["c"], pol["k"], n)), pol["c"], pol["k"], n)
    source = ("built here by verity_vllm: the class circuit (window_circuit.class_circuit), a session's gates by the "
              "gate table and its fresh bits by the record's description (block_counts)")
    return P.Policy("built", model, cls, pol["c"], pol["k"], n, work, parts, 0, source, verdict), div
