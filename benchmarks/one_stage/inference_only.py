"""A served run's inference-only verdict: its declared circuit accepted as at most n (c, k)-inference blocks, or refused,
by `NciCheck.Count`'s rule (`verity/protocols/inference_only/verifier/nci/lean/NciCheck/Count.lean`), against a policy the run
registered before it served.

    inference_only.py policy --class S,P,O --model REPO --revision REV --lease-min M [--utilization U]
        [--serve PHI] [--pearl-tables SCHEME] --out POLICY
    inference_only.py verdict --policy POLICY --serve-run D --circuit RECORD --registration REGISTRATION
        [--pins WEIGHTS_REGISTRATION] --out VERDICT --private FULL [--lean NCI_CHECK [--lean-timeout S]] [--retroactive WHY]

**The policy** (`policy`, `POLICY_FORMAT`) is written before the serve from the run's settings alone, before its lease
unless ``--serve`` states φ, and fixes (c, k, n) with every input to them and its source:

- the session: the class's circuit under the registered φ (`window_circuit.class_circuit` of `prove_path.registered_of_serve`
  on `serve_facts`, the facts an H200 int8 serve records in its ``serve.json``, or on ``--serve``'s, `prove_path.py
  serve-phi`'s statement of the ``serve.json`` a serve will write, read on its card under the lease before it serves, as
  a checkpoint served in BF16 needs; with ``--pearl-tables`` cut at its linears at that PoUW scheme, Pearl+'s
  (`window_class.PearlTables`), as `prove_path.py window-circuit` records a window served under Pearl+), read from its
  description (`class_doc`), whose one block Definition is a slot's session over every position (`block_counts.session`);
- c, that Definition's Boolean gates by the gate table (`table_of`), and k, its fresh-input bits (`block_counts.fresh`) with
  θ every weight parameter the circuit registers held (`weights_of_record.PARAM`), B's k (`inference_blocks.budgets`): what
  the weights registration pins before the first request, so a run whose pins hold fewer weights reads more fresh bits
  than k and is refused. B's own c is the templates' gates, not the table's Boolean gates the count rule reads;
- n, the H200's capacity in sessions over the serve's registered span at the declared utilization (`capacity`):
  ⌈u·θ·T / W⌉ rounded up to the public grid (`partition_v2.grid`), θ the card's dense int8 MAC rate at its tensor clock
  (`served_shares.RHO`), T the GPU lease's limit (``--lease-min``, `served_verdicts.sh`'s LEASE_MIN, which no serve
  outlasts), u the declared utilization (``--utilization``, `UTILIZATION` by default) and W one session's declared work
  at θ: the class's fixed load (`window_class.Tables.schedule`, one slot) priced as `served_shares.step_macs` prices a
  served step, its linears at θ and its attention and LM head at θ/2.

Nothing in the policy reads the run, and n is no count of it, so the check can fail.

**The gate table** is `definitions.boolean.gate_table`'s, each primitive's two-input gates through its Boolean version,
plus `PRICES` for a primitive it can't price: `BitAtx{W}_v1` (bit i of a word, 0 outside it), which the table prices by
its Boolean version where it has one, and otherwise by an explicit circuit of it (`bit_at_gates`) that this suite
evaluates against the primitive. Any other primitive the table can't price leaves the count undecided, and the verdict
refuses (`gate-table`).

**The verdict** (`verdict`, `VERDICT_FORMAT`) reads the run's declared circuit: the picked window's circuit record at the
policy's class (``--circuit``, `served_verdicts.sh`'s step 3), whose session is the run's block Definition, its gates
by the table, and its fresh bits at the verifier's pins (``--pins``, the weights registration) and the window's roots
(``--registration``), as `block_counts` counts them; and the serve dir's sealed windows, each of which the class takes
(`class_windows`) holding the class's m blocks, since a class's circuit is the same for every window it takes. Its rows
are `NciCheck.Count`'s (``[definition, gates, fresh, blocks]``) at the exact counts, judged by its rule (`judge`): every
class with a block at most c gates and k fresh bits, and the blocks at most n. Anything the rule can't decide refuses
(`Undecided`): no policy, a serve dir that records no start, a policy written once the serve had begun (unless
``--retroactive`` says why it's read anyway), a record with no inference blocks, an unpriced primitive, a closed window
with no seal, or a sealed window the class doesn't take.
With ``--lean``, `nci-check --count` runs on one window's session Program (`lean_count`), with ``--detail`` where the
binary takes it, as its usage says (`takes_detail`): such a binary prints only ``{complies}`` without it and its whole
verdict with it, as one that doesn't take it prints by default. A block's gates that differ from its own refuse too.
That Program's inputs are positional (`Input<w>`), so its θ is empty: every input bit, the weights' too, is fresh to it,
and its fresh bits and its verdict there (``with_theta_empty``) are said beside the verdict's, not judged. Its whole
verdict counts that one window's blocks and names the row that failed, so the public verdict takes only its gates a
block, fresh bits and ``complies``, or its refusal's ``error`` and ``message`` (`lean_said`), and the private one the
whole call.

``--out`` is the public verdict: the judgment against (c, k, n) and no count, of blocks or of windows, in any form. It
holds the verdict, ``complies``, ``failure`` and ``reason``, each block class's session and its gates and fresh bits a
block (``classes``), the class's slots and positions, and ``counts`` saying that n was judged on the exact count, which
it doesn't publish. ``--private`` (the serve dir's, 0600) holds it with the exact counts (``blocks``, `NciCheck.Count`'s
rows, and ``total``), each window the class doesn't take and why, and the `nci-check --count` call (``lean_detail``).
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import served_shares as SS  # noqa: E402
import served_verdicts as SV  # noqa: E402

POLICY_FORMAT = "verity-bench/inference-only-policy/v1"
VERDICT_FORMAT = "verity-bench/inference-only-verdict/v4"
#: what the public verdict says of the counts: a count on the grid, floored or not, still gives the count of windows
#: (at 4 blocks a window, exactly for 5 to 16 of them), so it publishes the judgment against n alone
COUNTS = "none published: n judged on the blocks' exact count, which only the serve dir's private verdict holds"
NCI_PACKAGE = ROOT / "verity/protocols/inference_only/verifier/nci/lean"
NCI_REFERENCE = NCI_PACKAGE / "scripts" / "nci_reference.py"
#: `prove_path.py window-circuit`'s default ``--cap``: the units a stated member holds in the class's record
CAP = 64
#: the H200 an int8 serve's ``serve.json`` names (`serve_seal.target_of` on the card: ``TargetProfile.from_device`` with
#: the int8 scope ``--precision verity_int8`` runs)
H200 = {"compute_capability": (9, 0), "num_sms": 132, "int8_exact": "linears"}
CLOCK = "tensor"
UTILIZATION = Fraction(1, 4)
UTILIZATION_SOURCE = ("compute-accounting's declared share for this week's served run: at least 9%, 25% with the GPU "
                      "commit (note:20261102T2333Z-draft-cmp231-plan-97, posted Mon 2 Nov 2:55 PM PST)")
RATE_SOURCE = ("served_shares.RHO['tensor']: 132 SMs x 4,096 int8 MACs a clock at the tensor clock's 1,830 MHz, the H200's "
               "dense int8 rate, compute-accounting's θ (Daniel, 29 Oct)")
SPAN_SOURCE = "the serve's GPU lease limit, served_verdicts.sh's LEASE_MIN (gpu-lease --max-min), which no serve outlasts"
WORK_SOURCE = ("one slot of the class's fixed load (window_class.Tables.schedule) priced as served_shares.step_macs prices "
               "a served step: its linears at θ, its attention and LM head at θ/2, so counted twice in θ's MACs")
BIT_AT = re.compile(r"BitAtx([1-9][0-9]*)_v1")


class Undecided(Exception):
    """The count rule can't decide the run: the message says why, and the verdict refuses."""


# -- the gate table ---------------------------------------------------------------------------------------------------

def bit_at_makes(width: int) -> bool:
    """Whether `bit_at_gates` makes `BitAtx{width}`: ``width`` a power of two up to 2^31."""
    return 0 < width <= 1 << 31 and width & (width - 1) == 0


def bit_at_gates(width: int) -> list[tuple[str, int, int]]:
    """An explicit Boolean circuit of `BitAtx{width}` (`registry.sampling.bit_at`: bit i of the word, 0 for i outside
    [0, width)), for ``width`` a power of two up to 2^31, each gate ``(op, a, b)`` (``b`` -1 for a Not) over wires: the
    word's bits 0 .. width-1, the index's 32 bits (two's complement, bit 0 first), then one a gate, the last the output.
    It tests the index's bits above log2(width) all zero (a Not of each, then their And), selects the word's bit by the
    rest (a mux tree, a ⊕ (s ∧ (a ⊕ b)) a mux) and Ands the two: two-input Ands and Xors and Nots, one gate each as the
    census counts them (`census.BooleanWork`)."""
    if not bit_at_makes(width):
        raise ValueError(f"BitAtx{width}: a power of two up to 2^31 has an index test this circuit makes")
    b = width.bit_length() - 1
    gates: list[tuple[str, int, int]] = []

    def gate(op: str, x: int, y: int = -1) -> int:
        gates.append((op, x, y))
        return width + 32 + len(gates) - 1

    zero = [gate("not", width + j) for j in range(b, 32)]
    ok = zero[0]
    for z in zero[1:]:
        ok = gate("and", ok, z)
    level = list(range(width))
    for j in range(b):
        level = [gate("xor", lo, gate("and", width + j, gate("xor", lo, hi))) for lo, hi in zip(level[0::2], level[1::2])]
    gate("and", level[0], ok)
    return gates


def run_gates(gates: Sequence[tuple[str, int, int]], width: int, word: int, i: int) -> int:
    """`bit_at_gates`' circuit on ``word`` and the i32 ``i``."""
    w = [(word >> j) & 1 for j in range(width)] + [((i & 0xFFFFFFFF) >> j) & 1 for j in range(32)]
    for op, x, y in gates:
        w.append(1 - w[x] if op == "not" else (w[x] & w[y]) if op == "and" else (w[x] ^ w[y]))
    return w[-1]


def price(key: str) -> dict[str, Any] | None:
    """A primitive's gates by `PRICES`' rule, where it has one: ``{gates, by}``."""
    m = BIT_AT.fullmatch(key)
    if m is None or not bit_at_makes(int(m[1])):
        return None
    return {"gates": len(bit_at_gates(int(m[1]))),
            "by": f"bit_at_gates({m[1]}): an explicit circuit of it, no Boolean version circuit-check checks"}


PRICES = ("BitAtx{W}_v1", price)


def table_of(fns: Sequence[Any]) -> tuple[dict[str, int], dict[str, dict[str, Any]], list[str]]:
    """The gate table over the primitives ``fns`` reach: `definitions.boolean.gate_table`'s, each primitive it can't
    price priced by `price` where it can be, and the keys of the rest."""
    from verity.definitions import boolean as BL

    table, unpriced = BL.gate_table(fns)
    priced, left = {}, []
    for key in unpriced:
        p = price(key)
        if p is None:
            left.append(key)
        else:
            table[key], priced[key] = p["gates"], p
    return table, priced, left


def digest(doc: Any) -> str:
    return hashlib.sha512(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# -- the session and the policy ---------------------------------------------------------------------------------------

def serve_facts(model: str, revision: str) -> dict[str, Any]:
    """What φ reads of an H200 int8 serve (`prove_path.phi_of_serve`, `registered_seed`), as its ``serve.json`` records it."""
    from verity_vllm import TargetProfile
    from verity_vllm.engine.served_seed import DEFAULT_SEED

    target = TargetProfile.default().with_overrides(**H200).to_json()
    return {"model": model, "revision": revision, "target": target, "execution": "bi-eager", "default_seed": DEFAULT_SEED}


def stated_serve(doc: Mapping[str, Any], model: str, revision: str) -> dict[str, Any]:
    """What φ reads of a serve as ``doc`` states it (`prove_path.py serve-phi`'s statement of the ``serve.json`` it will
    write, or a run's ``serve.json``), the fields `serve_facts` gives; a ValueError unless it is ``model`` at
    ``revision``."""
    serve = {k: doc.get(k) for k in ("model", "revision", "target", "execution", "default_seed")}
    serve["execution"] = serve["execution"] or (doc.get("engine") or {}).get("execution")
    if (serve["model"], serve["revision"]) != (model, revision):
        raise ValueError(f"--serve states φ of {serve['model']} at {serve['revision']}, not of --model {model} at "
                         f"--revision {revision}")
    return serve


def class_doc(serve: Mapping[str, Any], cls: Any, scheme: str | None = None) -> tuple[dict[str, Any], list[str]]:
    """The class's circuit under ``serve``'s φ (`window_circuit.class_circuit`, the circuit `prove_path.py window-circuit
    --class` records for a window the class takes), with a PoUW ``scheme`` cut at its linears (`window_class.PearlTables`),
    as the description its session and fresh bits are read from, the way `inference_blocks.budgets` reads B's k (no
    member stated, which is work per template the counts don't read), and the parameters it registers; `Undecided` where
    the vocabulary can't express it."""
    sys.path.insert(0, str(ROOT / "benchmarks" / "prove_path"))
    import prove_path as PP
    from verity_vllm.program.predict.common import Unexpressible
    from verity_vllm.program.predict.window_circuit import class_circuit
    from verity_vllm.program.predict.window_class import PearlTables

    try:
        circuit = class_circuit(PP.registered_of_serve(serve, None, None, PP.CAP_BITS), cls,
                                tables=PearlTables(scheme) if scheme else None)
    except Unexpressible as e:
        raise Undecided(f"the class's circuit under the policy's φ: {e}") from e
    doc = {**circuit.description(), "name": circuit.name, "partition": "",
           "blocks": circuit.blocks.doc(sum(circuit.units.values()), "")}
    return doc, sorted(circuit.registered.params if circuit.registered is not None else ())


def weight_pins(params: Sequence[str]) -> dict[str, str]:
    """Every weight parameter of ``params`` held (`weights_of_record.PARAM`'s, as `inference_blocks.budgets` holds them):
    one root a parameter, the same as the pin and as the window's."""
    from verity_vllm.check.weights_of_record import PARAM

    return {p: "00" for p in sorted(params) if p.split("/", 1)[0] == PARAM}


def session_row(doc: Mapping[str, Any], pins: Mapping[str, str] | None, roots: Mapping[str, str] | None) -> dict[str, Any]:
    """A circuit record's block class: its session Definition's digest and key, m slots and positions, gates by
    `table_of`'s table and fresh bits at ``pins`` and ``roots`` (`block_counts`), and the table."""
    from verity.circuits import codec, partition_v2 as PV2
    from verity_vllm.program.predict import block_counts as BC

    try:
        fn, m, positions = BC.session(doc)
    except (KeyError, TypeError, ValueError) as e:
        raise Undecided(f"the record's inference blocks: {e}") from e
    table, priced, unpriced = table_of([fn])
    if unpriced:
        raise Undecided(f"gate-table: no gates for {', '.join(unpriced)}, which the block computes")
    g, miss = BC.gates(fn, table, {})
    if miss:
        raise Undecided(f"gate-table: no gates for {', '.join(sorted(miss))}, which the block computes")
    try:
        k = BC.fresh(doc, m, pins, roots)
    except (KeyError, TypeError, ValueError) as e:
        raise Undecided(f"the record's fresh bits: {e}") from e
    return {"definition": PV2.id_digest(fn), "key": codec.descriptor_id(fn), "slots": m, "positions": positions,
            "gates": g, "fresh": k, "table": table, "priced": priced}


def session_work(cls: Any, model: str) -> dict[str, int]:
    """One session's declared MACs at the class's fixed load, one slot's rows (`served_shares.step_macs`), and the
    θ-MACs they cost: the linears once, attention and the LM head twice (`served_shares.ideal_s`)."""
    from verity_vllm.program.predict.window_class import Tables

    macs = {"linears": 0, "attention": 0, "lm_head": 0}
    for rows in Tables.schedule(cls):
        mine = [r for r in rows if r[0] == 0]
        for k, v in SS.step_macs({"rows": mine, "emitted": {"0": len(mine)}}, model).items():
            macs[k] += v
    return {**macs, "theta_macs": macs["linears"] + 2 * (macs["attention"] + macs["lm_head"])}


def capacity(work: int, lease_min: int, utilization: Fraction, clock: str = CLOCK) -> dict[str, Any]:
    """n: the sessions of ``work`` θ-MACs the card serves in ``lease_min`` minutes at ``utilization``, rounded up to the
    public grid, with its inputs."""
    from verity.circuits import partition_v2 as PV2

    rate = 4096 * 132 * SS.CLOCKS_MHZ[clock] * 10**6
    if work <= 0 or lease_min <= 0 or not 0 < utilization <= 1:
        raise ValueError(f"a session's work, the span and the utilization are positive, u at most 1: {work}, {lease_min}, "
                         f"{utilization}")
    raw = math.ceil(utilization * rate * lease_min * 60 / work)
    return {"n": PV2.grid(raw), "raw": raw, "grid_bits": PV2.GRID_BITS,
            "rate": {"macs_per_s": rate, "clock_mhz": SS.CLOCKS_MHZ[clock], "source": RATE_SOURCE},
            "utilization": [utilization.numerator, utilization.denominator],
            "span_s": lease_min * 60}


def window_class(arg: str) -> Any:
    """``--class``'s SLOTS,PROMPT,OUTPUTS as a `window_circuit.WindowClass` (`prove_path.window_class_of`'s)."""
    from verity_vllm.program.predict.window_circuit import WindowClass

    return WindowClass(*(int(x) for x in arg.split(",")))


def policy(cls_arg: str, model: str, revision: str, lease_min: int, utilization: Fraction, source: str,
           written: str, tree: str | None, stated: Mapping[str, Any] | None = None,
           scheme: str | None = None) -> dict[str, Any]:
    """The registered policy (module docstring), from the run's settings alone: φ ``stated`` (`stated_serve`), else
    `serve_facts`', and its circuit cut at a PoUW ``scheme``'s linears."""
    cls = window_class(cls_arg)
    serve = serve_facts(model, revision) if stated is None else stated_serve(stated, model, revision)
    doc, params = class_doc(serve, cls, scheme)
    pins = weight_pins(params)
    row = session_row(doc, pins, pins)
    work = session_work(cls, model)
    cap = capacity(work["theta_macs"], lease_min, utilization)
    return {"format": POLICY_FORMAT, "written_utc": written, "tree": tree,
            "class": {"slots": cls.slots, "prompt": cls.prompt, "outputs": cls.outputs, "positions": cls.positions},
            "serve": serve, "pearl_tables": scheme,
            "session": {k: row[k] for k in ("definition", "key", "slots", "positions")},
            "c": row["gates"], "k": row["fresh"], "n": cap["n"],
            "theta": f"every weight parameter the class's circuit registers ({len(pins)}), held",
            "table": {"sha512": digest(row["table"]), "entries": len(row["table"]), "priced": row["priced"]},
            "n_inputs": {**cap, "utilization_source": source, "span_source": SPAN_SOURCE,
                         "work": {**work, "source": WORK_SOURCE}}}


# -- the run's blocks and the verdict ---------------------------------------------------------------------------------

def outputs_of(r: Mapping[str, Any]) -> Sequence[int]:
    """A request record's outputs: its ``output_ids``, else its ``streamed_ids`` (`advice.io_record`)."""
    return r.get("output_ids") if r.get("output_ids") is not None else (r.get("streamed_ids") or ())


def class_windows(serve_dir: Path, cls: Mapping[str, int]) -> tuple[int, dict[int, str]]:
    """The serve's sealed windows, and by window why the class can't take one: `served_verdicts.pick`'s refusals but
    for a window with no request (its slots padded, the class's circuit still), a closed window with no seal, and a
    request past the class's prompt or outputs."""
    seals = {int(s["window"]): s for s in SV._lines(serve_dir / "seals.jsonl")}
    lines = {int(w["window"]): w for w in SV._lines(serve_dir / "windows.jsonl")}
    reqs = SV._lines(serve_dir / "requests.jsonl")
    why = {k: "closed, with no seal" for k in lines if k not in seals}
    for k in sorted(seals):
        mine = [r for r in reqs if k in (r.get("windows") or ())]
        w = SV._refusal(k, seals[k], lines.get(k), mine, cls["slots"])
        if w is None:
            long = [r.get("user") or r.get("engine_id") for r in mine
                    if len(r.get("prompt_ids") or ()) > cls["prompt"] or len(outputs_of(r)) > cls["outputs"]]
            if long:
                w = f"requests past the class's {cls['prompt']} prompt tokens or {cls['outputs']} outputs: {sorted(map(str, long))}"
        if w is not None and w != "no request":
            why[k] = w
    return len(seals), dict(sorted(why.items()))


def judge(rows: Sequence[Sequence[Any]], c: int, k: int, n: int) -> dict[str, Any]:
    """`NciCheck.Count`'s verdict on its rows (``[key, gates, fresh, blocks]``): it complies when every row with a block
    has at most ``c`` gates and ``k`` fresh bits and the blocks are at most ``n``; ``failure`` the first that fails,
    ``[row, "gates"|"fresh"]``, else ``[None, "count"]`` (`nci_reference.evaluate_count`)."""
    total = sum(r[3] for r in rows)
    failure = next(([i, "gates" if g > c else "fresh"] for i, (_, g, x, m) in enumerate(rows) if m > 0 and (g > c or x > k)),
                   None)
    if failure is None and total > n:
        failure = [None, "count"]
    return {"blocks": [list(r) for r in rows], "total": total, "complies": failure is None, "failure": failure}


def _reference():
    spec = importlib.util.spec_from_file_location("nci_reference", NCI_REFERENCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def count_instance(doc: Mapping[str, Any], table: Mapping[str, int], c: int, k: int, n: int) -> dict[str, Any]:
    """`NciCheck.Count`'s instance on one window: its session Definition activated at its m slots (the session
    Program, `population.population_program`) with no θ, its block that Definition."""
    from verity.circuits import codec
    from verity.circuits.population import population_program
    from verity_vllm.program.predict import block_counts as BC

    fn, m, _ = BC.session(doc)
    text = codec.canonical_json(codec.encode_program(population_program(fn, m))).decode()
    return {"program": text, "theta": [], "blocks": [codec.descriptor_id(fn)], "table": dict(table), "c": c, "k": k, "n": n}


def takes_detail(exe: Path, timeout_s: int) -> bool:
    """Whether ``exe``'s ``--count`` takes ``--detail``, as the usage it prints given no argument says. One that does
    prints only ``{complies}`` without it; one that doesn't answers ``--detail`` with its usage."""
    p = subprocess.run([str(exe)], capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=timeout_s)
    return "--detail" in p.stdout + p.stderr


def lean_count(exe: Path, inst: Mapping[str, Any], scratch: Path, timeout_s: int = 1800) -> dict[str, Any]:
    """`nci-check --count` on ``inst``, in ``scratch``, with ``--detail`` where ``exe`` takes it (`takes_detail`): its
    ``flags`` and the JSON object it printed (``out``), its whole verdict or its refusal, else ``error``, why it printed
    none, and where it said something, its tail (``said``). All of it is private; `lean_said` is what the public verdict
    takes."""
    fd, path = tempfile.mkstemp(dir=scratch, suffix=".json")
    with os.fdopen(fd, "w") as fh:
        json.dump(inst, fh)
    flags = ["--count"]
    try:
        if takes_detail(exe, timeout_s):
            flags.append("--detail")
        p = subprocess.run([str(exe), *flags, path], capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return {"flags": flags, "error": f"nci-check {' '.join(flags)} gave no verdict within {timeout_s} s"}
    finally:
        os.unlink(path)
    if p.returncode != 0:
        return {"flags": flags, "error": f"nci-check {' '.join(flags)} exit {p.returncode}", "said": p.stderr.strip()[-300:]}
    try:
        out = json.loads(p.stdout)
        if not isinstance(out, dict):
            raise ValueError("not an object")
    except ValueError:
        return {"flags": flags, "error": f"nci-check {' '.join(flags)} printed no JSON object",
                "said": p.stdout.strip()[-300:]}
    return {"flags": flags, "out": out}


def lean_said(call: Mapping[str, Any], gates: int) -> dict[str, Any]:
    """What the public verdict takes of `lean_count`'s ``call``: why it gave no verdict, its refusal (``error`` and
    ``message``), or its gates a block, whether they are ``gates``, and with θ empty its fresh bits and ``complies``.
    Never its rows' counts, total or failure, nor any other text it printed."""
    out = call.get("out")
    if out is None:
        return {"error": call["error"]}
    if "error" in out:
        return {key: out[key] for key in ("error", "message") if key in out}
    rows = out.get("blocks")
    if not (isinstance(rows, list) and len(rows) == 1 and isinstance(rows[0], list) and len(rows[0]) == 4):
        return {"error": f"nci-check {' '.join(call['flags'])} printed no row of the block"}
    _, g, x, _ = rows[0]
    return {"gates": g, "agrees_on_gates": g == gates, "with_theta_empty": {"fresh": x, "complies": out.get("complies")}}


def served_start(serve_dir: Path) -> str:
    """The serve's start (its ``serve.json`` clocks' time, stamped as the endpoint listens), UTC, ISO 8601."""
    t = json.loads((serve_dir / "serve.json").read_text())["clocks"]["time"]
    return datetime.datetime.fromtimestamp(float(t), datetime.UTC).isoformat(timespec="seconds")


def before(written: Any, start: str | None) -> bool:
    """Whether ``written`` (the policy's ``written_utc``) is an ISO 8601 time with its offset strictly before ``start``."""
    try:
        w, s = datetime.datetime.fromisoformat(str(written)), datetime.datetime.fromisoformat(str(start))
        return w.tzinfo is not None and s.tzinfo is not None and w < s
    except ValueError:
        return False


def verdict(pol: Mapping[str, Any] | None, doc: Mapping[str, Any] | None, serve_dir: Path,
            pins: Mapping[str, str] | None, roots: Mapping[str, str] | None, retroactive: str | None = None,
            lean: Path | None = None, scratch: Path | None = None,
            lean_timeout_s: int = 1800) -> tuple[dict[str, Any], dict[str, Any]]:
    """The run's verdict (module docstring): its public form and its private one."""
    base: dict[str, Any] = {"format": VERDICT_FORMAT}
    private: dict[str, Any] = {}
    try:
        if pol is None or pol.get("format") != POLICY_FORMAT:
            raise Undecided("no policy registered before the serve (inference_only.py policy)")
        c, k, n = int(pol["c"]), int(pol["k"]), int(pol["n"])
        base |= {"policy": {"sha512": digest(pol), "written_utc": pol.get("written_utc")}, "c": c, "k": k, "n": n}
        try:
            start = served_start(serve_dir)
        except (OSError, ValueError, KeyError, TypeError) as e:
            raise Undecided(f"the serve dir records no start (its serve.json's clocks' time): {type(e).__name__}: {e}") from e
        base["serve_began_utc"] = start
        if not before(pol.get("written_utc"), start):
            if retroactive is None:
                raise Undecided(f"the policy was written at {pol.get('written_utc')}, not before the serve began at {start}")
            base["retroactive"] = retroactive
        if doc is None:
            raise Undecided("no circuit record of the run's window at the policy's class")
        row = session_row(doc, pins, roots)
        base["table"] = {"sha512": digest(row["table"]), "entries": len(row["table"]), "priced": row["priced"]}
        base["registered_session"] = row["definition"] == pol["session"]["definition"]
        sealed, out = class_windows(serve_dir, pol["class"])
        private |= {"sealed": sealed, "outside_class": {str(w): why for w, why in out.items()}}
        if out:
            raise Undecided("the serve sealed or closed a window the class doesn't take, so its blocks are undecided "
                            "(each such window and why in the private verdict)")
        if not sealed:
            raise Undecided("the serve sealed nothing")
        blocks = sealed * row["slots"]
        rows = [[row["definition"], row["gates"], row["fresh"], blocks]]
        got = judge(rows, c, k, n)
        private |= {"blocks": rows, "total": got["total"]}
        base |= {"classes": [[d, g, x] for d, g, x, _ in rows], "counts": COUNTS, "complies": got["complies"],
                 "failure": got["failure"], "positions": row["positions"], "slots": row["slots"]}
        if lean is not None:
            inst = count_instance(doc, row["table"], c, k, n)
            call = lean_count(lean, inst, scratch or Path(tempfile.gettempdir()), lean_timeout_s)
            private["lean_detail"] = call
            base["lean"] = said = lean_said(call, row["gates"])
            if "gates" in said and not said["agrees_on_gates"]:
                raise Undecided(f"nci-check --count counts {said['gates']} gates a block, and this {row['gates']}")
        if got["complies"]:
            base |= {"verdict": "accept", "reason": None}
        else:
            i, what = got["failure"]
            base |= {"verdict": "refuse", "reason": {
                "gates": f"a block of {row['gates']} gates, more than c = {c}",
                "fresh": f"a block of {row['fresh']} fresh bits, more than k = {k}",
                "count": f"the blocks' exact count is more than n = {n}"}[what]}
    except Undecided as e:
        base |= {"verdict": "refuse", "reason": f"undecided: {e}", "complies": False}
    except Exception as e:  # noqa: BLE001 - anything else the count can't get past is undecided too, and refuses
        base |= {"verdict": "refuse", "reason": f"undecided: {type(e).__name__}: {e}"[:600], "complies": False}
    return base, {**base, **private}


# -- the command line -------------------------------------------------------------------------------------------------

def _write(path: Path, doc: Mapping[str, Any], private: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(doc, indent=1, sort_keys=True) + "\n"
    if private:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
        os.chmod(path, 0o600)
    else:
        path.write_text(text)


def line(v: Mapping[str, Any]) -> str:
    """The verdict's one public line."""
    if "classes" not in v:
        return f"{v['verdict']}: {v['reason']}"
    (d, g, x), = v["classes"]
    lean = v.get("lean")
    said = ("" if lean is None else f"; nci-check --count: {lean['error']}" if "error" in lean else
            f"; nci-check --count: {lean['gates']} gates a block ({'agrees' if lean['agrees_on_gates'] else 'DIFFERS'})")
    return (f"{v['verdict']}{'' if v['reason'] is None else ': ' + v['reason']}: session {d[:16]}… "
            f"({'the registered one' if v.get('registered_session') else 'not the registered one'}), {g} gates and {x} "
            f"fresh bits a block, against c {v['c']}, k {v['k']}, n {v['n']} (no count published: n is judged on the "
            f"blocks' exact count, held on the node){said}")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("policy", help="the registered (c, k, n), before the serve")
    p.add_argument("--class", dest="cls", required=True, help="the window class, SLOTS,PROMPT,OUTPUTS")
    p.add_argument("--model", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--lease-min", type=int, required=True, help="the serve's GPU lease limit in minutes (LEASE_MIN)")
    p.add_argument("--utilization", type=Fraction, default=UTILIZATION, help=f"the declared utilization (default {UTILIZATION})")
    p.add_argument("--utilization-source", default=UTILIZATION_SOURCE)
    p.add_argument("--tree", default=None, help="the tree's commit, recorded")
    p.add_argument("--serve", type=Path, default=None,
                   help="φ as the serve will record it: prove_path.py serve-phi's statement of its serve.json, read on its "
                        "card before it serves (default: an H200 int8 serve's, serve_facts)")
    p.add_argument("--pearl-tables", metavar="SCHEME", default=None,
                   help="the class's circuit cut at its linears at this PoUW scheme, Pearl+'s (window_class.PearlTables)")
    p.add_argument("--out", type=Path, required=True)
    v = sub.add_parser("verdict", help="the run's verdict against its policy")
    v.add_argument("--policy", type=Path, required=True)
    v.add_argument("--serve-run", type=Path, required=True)
    v.add_argument("--circuit", type=Path, required=True, help="the picked window's circuit record at the policy's class")
    v.add_argument("--registration", type=Path, required=True, help="the window's registration (its roots)")
    v.add_argument("--pins", type=Path, default=None, help="the weights registration, the verifier's pins")
    v.add_argument("--lean", type=Path, default=None, help="nci-check, to run --count on one window's session Program")
    v.add_argument("--lean-timeout", type=int, default=1800, help="nci-check --count's limit in seconds (default 1800)")
    v.add_argument("--retroactive", default=None, help="why a policy written after the serve began is read anyway")
    v.add_argument("--out", type=Path, required=True)
    v.add_argument("--private", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "policy":
        written = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
        try:
            stated = None if a.serve is None else json.loads(a.serve.read_text())
            pol = policy(a.cls, a.model, a.revision, a.lease_min, a.utilization, a.utilization_source, written, a.tree,
                         stated, a.pearl_tables)
        except (Undecided, ValueError, KeyError, OSError) as e:
            print(f"INFERENCE-ONLY POLICY refused: {type(e).__name__}: {e}"[:1200], flush=True)
            return 1
        _write(a.out, pol)
        i = pol["n_inputs"]
        print(f"INFERENCE-ONLY POLICY at class {a.cls}: c {pol['c']} gates, k {pol['k']} fresh bits, n {pol['n']} sessions "
              f"(⌈{i['utilization'][0]}/{i['utilization'][1]} x {i['rate']['macs_per_s']} MAC/s x {i['span_s']} s / "
              f"{i['work']['theta_macs']} θ-MACs⌉ = {i['raw']}, on the grid); session {pol['session']['definition'][:16]}…, "
              f"policy {digest(pol)[:16]}…", flush=True)
        return 0
    pol = json.loads(a.policy.read_text()) if a.policy.is_file() else None
    doc = json.loads(a.circuit.read_text()) if a.circuit.is_file() else None
    from verity.commitments.windows import interface as IF
    from verity_vllm.program.predict.window_weights import WeightsRegistration

    pins = (dict(WeightsRegistration.of(json.loads(a.pins.read_text()), str(a.pins)).roots)
            if a.pins is not None and a.pins.is_file() else None)
    roots = IF.registered_roots(json.loads(a.registration.read_text())) if a.registration.is_file() else None
    with tempfile.TemporaryDirectory(dir=a.private.parent if a.private.parent.is_dir() else None) as scratch:
        pub, full = verdict(pol, doc, a.serve_run, pins, roots, a.retroactive, a.lean, Path(scratch), a.lean_timeout)
    _write(a.private, full, private=True)
    _write(a.out, SV.public_json(pub))
    print(f"INFERENCE-ONLY {line(SV.public_json(pub))}"[:2000], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
