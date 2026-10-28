"""run.py: one recorded H1 window's drawn tile units through sampled proofs, the challenger's and the verifier's side.

    python run.py prepare --work W --stage SW --lean FLOCK_VERIFY --layer L1.npz --layer2 L2.npz [--draw 8] [--k K] [--cols N]
    python run.py summarize --work W --out OUT

``prepare`` runs one epoch of the window and plans its proofs, the prover's steps by `window.py` (a subprocess: the
challenger and the verifier never hold a row, a salt or a served value):

1. the challenger (`pouw.coins.Challenger`) commits to the epoch's salt and flip coins; the prover commits each call's
   tables, as a served round commits them before its coin (`h1window.table`): its A table a/<K>@<c> (the window's A
   rows, each with its s_a) and its z table zy@<c> (its tiles' z/y records: z = A Bᵀ exact and y its served dequant,
   what ``scaled_mm_zy`` writes), and the calls' B rows (each with its s_w) as the weights' b/<K> (`window.py commit`),
   and the salt coin opens after those roots are recorded;
2. the prover serves every tile's receipt under the unit salt, PoUW's window salt (`h1unit.unit_salt`: `pouw_salt` of the
   coin's stand-in for the certifier's reveal, the epoch as the window, a stand-in stream and prover) and commits them
   (`window.py serve`); the registration (the window's commitment and its receipt root) is recorded, and the sampling coin
   opens after it. The verifier builds the window's public Program and its template-instance partition from the
   commitment's public shape (`window_program`: WORK/verifier/program.json and partition.json, Lean's ``--program`` and
   ``--partition``), holds the commitment's registration to them (`check_registration`), and the stager builds its
   statement bound to them;
3. the verifier registers the window's ok rows itself (`ok_registration`: value ``ok``, row u the one word 1 under unit
   u's public salt), draws ``--draw`` units of the window's tiles by Lean ``flock-verify draw`` on the coin's stream,
   groups them by call (a registered port reads one value, and a call's units read its own tables, so a statement holds
   one call's units), and writes each call's ``--registered`` file (`registered_file`: `h1unit.registered_reads` from
   its own Program under the window's roots and its own ``ok``, the ok port o1 beside the A, B and record ports,
   WORK/verifier/w1-c<c>.registered.json);
4. the prover opens the sampled proof blocks call by call (`window.py open`), and the verifier checks each opened row is the one
   the Program's unit reads (`h1unit.unit_reads`) with a path to its root (its call's A table's and z table's and
   b/<K>'s registered roots under the Program's digest, the receipt's under the window's), and builds its own public
   file of each call's units (`h1unit.verifier_pub`: a shared-row file, each registered port's table the registered rows
   its units read, at the positions its Program reads and the opened paths) from the opened ``b || c``, its own public
   and ok rows and the units' indices;
5. the stager (`stage.py`, building the circuit once the window's shape is known) stages each call's sampled proof blocks as one
   shared-row statement on the prover's rows at their indices (``honest``, the first sampled proof block's call, and
   ``honest-c<c>`` each other drawn call), and each prover's pub.bin must be the verifier's byte for byte: every unit's
   ok is 1, its record the tile's product and dequant. Lean (``verify --zk --registered``) derives each instance's unit
   from its own Program, and an accepted check must say so (``units: derived``), and opens each A row, B row, record and
   ok row against its registered root at the position its Program reads, so ok is bound through the verifier's
   registration (`Accounting.Guarantees.TS.CheckedSessionSound`'s port ``ok`` of ``own``), not through its public file.

The negatives are on ``honest``'s units, the sampled proof blocks of the first sampled proof block's call.

The planted negatives, each checked by Lean against the verifier's own file:

* ``flipped-zy``: a window registered with the first sampled proof block's record word 0 flipped (`window.py tamper --zy-word`,
  every salt kept). Its openings pass their paths (a consistent window), so the verifier's file holds the flipped leaf
  and its registered file the window's z table root; the only proof of those rows the unit admits is the honest one,
  which Lean must reject against them.
* ``wrong-receipt``: the same with that unit's receipt from another salt.
* ``other-window``: the first sampled proof block's tile, opened from a second window (other tokens, its own epoch and salt) and
  proved as a statement of its own: its opening fails window 1's roots, and Lean must reject its proof against the
  verifier's file of window 1's unit. ``other-window-own`` (its own window's file) must accept: the proof is sound for
  the window it came from.

Those three fail at binding (the proof's statement is not the verifier's). ``tampered-a`` passes binding and fails the
registered reads:

* ``tampered-a``: the sampled proof blocks' statement staged on rows whose first unit's a0 row (at every unit that reads it) is not
  the registered one (`window.py open --tamper-a`: one code flipped, a fresh salt, the registered path kept), checked
  against its own public file with the verifier's ``--registered`` file. The proof is of the rows it commits, so Lean
  must reject it for that table row not opening its call's A table's registered root, and say so; the verifier's own
  check of the row (`registered_opens`) refuses it too, and opens the honest statement's.

Two more pass binding against their own public files and fail the ok row's registered read, each a proof Lean would
accept were ok bound only through the public file:

* ``ok-zero``: the base statement of ``in-z`` (`OK_ZERO`: the sampled proof blocks on the tampered window's committed rows, the
  unit computing ok = 0 at the moved record, `stage.py`'s outputs), whose public file states that 0 under the public
  ok salt: its o1 table row opens no root of the verifier's ``ok``, and Lean must say so.
* ``ok-unregistered``: the honest units staged with their ok rows committed afresh (`window.py open --unregistered-ok`),
  so the statement names no o1 registered read: Lean must refuse the o1 port of its ``--registered`` file as committed
  with no path to its registered root.

The in-proof negatives (`IN_PROOF`) pass both and carry a wrong witness instead. Each is a window registered with one
value of the first sampled proof block changed (`window.py tamper`, every salt kept), opened and checked like the honest one under
its own registered file, and a statement of it whose committed rows are all the window's (`forge`: the sampled proof blocks'
statement staged on the window's rows, each output row the circuit computes but the window does not commit replaced with
the committed one, `h1unit.forge_statement`). The prover's pub.bin is then the verifier's, and some output row is not
what the circuit computes on its unit's inputs. A changed record makes the unit compute ok = 0 where the verifier's file
holds 1:

* ``in-receipt``: the receipt's word 0 plus one (``--receipt-word``).
* ``in-d``: a kernel's D_1 word (lane 0, entry 0) one too large (``--d-word``): the receipt computed from it, and under
  ``H1_TILE=v3`` the record's z that word's recombination, y its dequant.
* ``in-y``: the record's y of cell 0 plus one (``--y-cell``): y is not dq(z).
* ``in-z``: the record's z of cell 0 plus one, its y that z's dequant (``--z-cell``): z is not A Bᵀ.
* ``in-swap``: the records of the first sampled proof block and another unit of its call exchanged (``--swap``: the next sampled
  proof block of the call when there is one, so both swapped records are in the statement).
* ``in-call``: the record of the same tile in the next call (``--call``), a row of another call's z table.
* ``in-z-only``: a record of z alone, its y zeroed (``--z-only``).
* ``in-xa`` (``H1_TILE=v3`` only): the record's z computed with one entry's X wrap bit xa = [A = 127] ∧ b flipped
  (``--xa-flip``: the first set one, else the first clear one; the receipt the honest one), as a prover that skipped (or
  invented) the activation-wrap correction would serve it. Its tampered window's served.json says whether the flipped
  bit was set and how many record words moved.

Each is proved like the honest statement, whose prover stops on the failing witness before it sends a proof, and again as
``<tag>-skip`` with ``FC_ZK_SELF_CHECK=skip`` (Flock's negative-control prover, which sends its proofs anyway), so the
in-session verifier and Lean judge a proof. ``summarize`` records which layer refused each (the prover, the in-session
verifier, Lean) and with what message. A registration of z alone (``z-only-registration``: the window's z table entries
rows of 2 R C words, no y) is refused by the verifier's own check of the registration (`check_registration`) before any
draw.

Every per-window registered value (each call's A table and z table, and ``ok``) is registered under the window's open root
(PRF-127: `h1unit.open_root`, a stand-in, as this harness has no auditor's ledger), and only the weights' b/<K> under none;
the verifier computes the root itself and refuses a commitment that registers its tables without it
(``unrooted-registration``). Window credit's last two steps (`credit_registration`: the layout of the descriptor, each
session's ok held to the verifier's own, `credit._ok_held`, one accepted outcome per sampled proof block, and
`credit._registration`) take the window's evidence: in ``prepare`` with each sampled proof block's finding the ports of its
call's ``--registered`` file, and refused for entries without the root, sessions that opened ``ok`` at its rows
registered without the root (``ok-unrooted``: credit reads no entry of ``ok``, the verifier's own) and another window's
root; in ``summarize`` on Lean's own VERDICT findings.

``summarize`` reads the job's proving and Lean records into OUT/summary.json: the unit and statement counts, proving
time and Lean verify time, every check's verdict against its expectation, and credit's registration on Lean's findings.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import h1unit as U  # noqa: E402
from verity.ml.circuits.codec import canonical_json, encode_program  # noqa: E402
from verity.ml.randomness import derive  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import partition as P  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import registered as G  # noqa: E402
from verity.core.protocols.pouw import core_check as CC  # noqa: E402
from verity.core.protocols.pouw import credit as CR  # noqa: E402
from verity.core.protocols.pouw import verifier  # noqa: E402
from verity.core.protocols.pouw.coins import Challenger, coin_reason  # noqa: E402

DRAW_DOMAIN = "verity/pouw/h1-prove-draw/v0"
R, C = U.TILE
#: the in-proof negatives, by `window.py tamper`'s flag
IN_PROOF = {"in-receipt": "--receipt-word", "in-d": "--d-word", "in-y": "--y-cell", "in-z": "--z-cell", "in-swap": "--swap",
            "in-call": "--call", "in-z-only": "--z-only", **({"in-xa": "--xa-flip"} if U.tile() == "v3" else {})}
#: the negatives that fail at binding, by `window.py tamper`'s flag
BINDING = {"flipped-zy": "--zy-word", "wrong-receipt": "--receipt"}
#: the in-proof negative whose base statement (its unit computes ok = 0 where its record moved) ``ok-zero`` proves
OK_ZERO = "in-z"
#: a statement tag's suffix whose prover runs with FC_ZK_SELF_CHECK=skip (job.sh), sending its proof though the witness fails
SKIP = "-skip"


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


def prover(*args: str) -> dict:
    """One prover step (`window.py`), its last line's JSON."""
    p = subprocess.run([sys.executable, str(HERE / "window.py"), *args], capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(f"window.py {args[0]}: exit {p.returncode}: {p.stderr[-2000:]}")
    return json.loads(p.stdout.strip().splitlines()[-1])


def digest(*parts: bytes) -> bytes:
    h = hashlib.sha512(b"verity/pouw/h1-prove/record/v0\0")
    for x in parts:
        h.update(len(x).to_bytes(8, "big") + x)
    return h.digest()


def tamper_flag(flag: str, units: list[int], call: tuple[int, int]) -> list[str]:
    """`window.py tamper`'s ``FLAG U`` on the first of one call's drawn ``units`` (``call``: the call's first unit and how
    many), and ``--swap``'s U and V another unit of the call: the next drawn one when there is one, so both swapped
    records are in the statement."""
    if flag != "--swap":
        return [flag, str(units[0])]
    first, n = call
    v = units[1] if len(units) > 1 else next((v for v in range(first, first + n) if v != units[0]), None)
    if v is None:
        raise SystemExit("--swap exchanges two records of one call's z table, and the call has one tile")
    return [flag, str(units[0]), str(v)]


# -- the verifier's view of a window ----------------------------------------------------------------------------------------

def program_of(k: int, calls, slot: int) -> SimpleNamespace:
    """The public Program of a window shape (`h1unit.Shape.window`: its calls' names and B rows, its slot), its
    template-instance query and partition (``tp`` as core evaluates it, ``obj`` the `verity/partition/v1` object)."""
    program, query = U.Shape(k, R, C).window(calls, slot)
    return SimpleNamespace(program=program, query=query, tp=P.evaluate(program, query), obj=P.partition(program, query))


def window_program(com: dict) -> SimpleNamespace:
    """The verifier's own `program_of` a window commitment's shape."""
    win = program_of(com["unit"]["k"], *U.window_shape(com))
    if win.tp.population != com["tiles"]:
        raise SystemExit(f"the window's Program has {win.tp.population} units, its commitment {com['tiles']} tiles")
    return win


def check_registration(win: SimpleNamespace, com: dict) -> list[str]:
    """Why a window commitment's registration is not of the verifier's own Program (``win``): it registers under the
    Program's digest, and its values are the weights' b/<K> (the calls' B rows) and each call's A table (the slot's rows)
    and z table (its tiles' records), the tables the verifier's Program names (`h1unit.program_tables`), each its own
    width of 16-bit leaves (`h1unit.value_words`), under that digest's domains (`h1unit.registered_domain`), each table's
    binding the window's open root, which the verifier computes itself (`h1unit.window_open_root`) and the commitment
    records, and b/<K>'s none (PRF-127), with the commitment's roots."""
    k, prog, why = com["unit"]["k"], win.obj["program"], []
    if com.get("program_sha512") != prog:
        why.append(f"it registers under Program {com.get('program_sha512')}, not the verifier's {prog}")
    roots = {e.get("value"): e.get("root") for e in com.get("registered", [])}
    try:
        wr = U.window_open_root(com)
    except (KeyError, TypeError, ValueError) as e:
        return [*why, f"it registers no b/{k} root to compute the window's open root from: {e!r}"]
    if com.get("window_root") != wr:
        why.append(f"it records the window's open root as {com.get('window_root')}, not the verifier's {wr}")
    want = [{"value": v, "root": roots.get(v), "leaves": n, "words": U.value_words(v),
             "domain": U.registered_domain(prog, v, n, wr).domain_id.hex()} for v, n in U.program_tables(win.program, k).items()]
    if com.get("registered") != want:
        why.append(f"its registered entries {com.get('registered')} are not the verifier's {want}")
    return why


def z_only(com: dict) -> dict:
    """``com`` as a prover that registers its records' z alone states it: its z tables' entries rows of 2 R C words, no y."""
    words = 2 * R * C
    return dict(com, registered=[dict(e, words=words, domain=G.domain(com["program_sha512"], e["value"], words, e["leaves"],
                                                                       window=com["window_root"]).domain_id.hex())
                                 if e["value"].startswith("zy@") else e for e in com["registered"]])


def unrooted(com: dict) -> dict:
    """``com`` as a prover that registers its per-window values with no window root states it (main's form before
    PRF-127): every entry's domain the registered domain of its value with no window bound."""
    return dict(com, registered=[dict(e, domain=G.domain(com["program_sha512"], e["value"], e["words"],
                                                         e["leaves"]).domain_id.hex()) for e in com["registered"]])


def check_opening(win: SimpleNamespace, com: dict, served: dict, op: dict, u: int) -> list[str]:
    """Why unit ``u``'s opening ``op`` is not the registered window's (``win`` its Program, ``com`` its commitment, held to
    the Program by `check_registration`, ``served`` its receipt root): each committed row is the one the Program's unit
    ``u`` reads (`h1unit.unit_reads`), and its path reaches that value's registered root: its call's A table's and z
    table's and b/<K>'s under the verifier's own Program digest (`h1unit.verify_registered`), the receipt's under the
    window's domain. The ok row is the verifier's own."""
    w, T, why = com["window"], com["tiles"], []
    trees = U.registered_of(com) | {"receipt": (T, served["receipt_root"])}
    if op.get("unit") != u:
        why.append(f"the opening is unit {op.get('unit')}'s, not {u}'s")
    for port, (value, index) in U.unit_reads(win.program, win.tp, u).items():
        if port not in op["rows"] and value not in trees:
            continue
        r = op["rows"].get(port)
        if r is None or r["value"] != value or r["index"] != index:
            why.append(f"{port}: not row {index} of {value}")
            continue
        if value not in trees:
            why.append(f"{port}: the window registers no {value}")
            continue
        count, root = trees[value]
        bc = bytes.fromhex(r["bc"])
        if value == "receipt":
            ok = U.verify(U.domain(w, value, count), bytes.fromhex(root), index, bc, r["path"])
        else:
            ok = U.verify_registered(win.obj["program"], value, count, bytes.fromhex(root), index, bc, r["path"],
                                     U.window_open_root(com))
        if not ok:
            why.append(f"{port}: its path does not reach the registered {value} root")
    return why


def unit_bcs(shape: U.Shape, com: dict, salt: bytes, op: dict) -> dict[str, bytes]:
    """The unit's rows' ``b || c`` as the verifier holds them: the opened leaves, its own public row's, and its own ok row's
    (the one word 1 under the public `h1unit.ok_salt`)."""
    port = {p.name: p for p in shape.ports()}
    u, w = op["unit"], com["window"]
    pub = U.commit_strings(port["pub"], U.pub_words(salt, u, U.window_roots(com))[None, :], [U.pub_salt(w, u)])[0]
    ok = U.commit_strings(port["o1"], U.OK[None, :], [U.ok_salt(w, u)])[0]
    return {**{p: bytes.fromhex(op["rows"][p]["bc"]) for p in [*shape.in_rows, "o0"]}, "pub": pub, "o1": ok}


def ok_registration(win: SimpleNamespace, com: dict) -> "U.G.Registered":
    """The verifier's own registration of a window's ok rows (`h1unit.register_ok`: the one word 1 under each unit's public
    salt), under its own Program's digest and the window's open root (`h1unit.window_open_root`, the verifier's own), from
    the window's name, its b/<K> root and the Program's population alone."""
    return U.register_ok(win.obj["program"], com["window"], win.tp.population, U.window_open_root(com))


def roots_of(com: dict, okr) -> dict[str, tuple[int, str]]:
    """Every registered value a statement of the window reads, each its rows and root: the commitment's (the prover's), and
    the verifier's own ``ok`` (``okr``, `ok_registration`)."""
    return U.registered_of(com) | {U.OK_VALUE: (okr.leaves, okr.root.hex())}


def unit_registered(win: SimpleNamespace, k: int, op: dict, u: int, okr) -> dict[str, tuple[str, int, str]]:
    """Unit ``u``'s registered reads as the verifier's public file states them (`h1unit.verifier_pub`'s ``reads``): each
    registered port's value, the position the verifier's Program reads (`h1unit.unit_reads`) and the opened path, and the
    ok port's at the verifier's own ``ok`` (``okr``, `ok_registration`) and its path there."""
    reg = set(U.Shape(k).in_rows)
    out = {port: (value, index, "".join(op["rows"][port]["path"]))
           for port, (value, index) in U.unit_reads(win.program, win.tp, u).items() if port in reg}
    value, index = U.unit_reads(win.program, win.tp, u)["o1"]
    return {**out, "o1": (value, index, okr.path(index).hex())}


def registered_file(path: Path, win: SimpleNamespace, com: dict, call: int, units: list[int], okr) -> tuple[dict, dict]:
    """The verifier's ``--registered`` file of a statement over call ``call``'s units (`h1unit.registered_reads`, under
    its commitment's roots and its own ``ok``, ``okr``), written to ``path``, held to the h1 window's own reads
    (`h1unit.unit_reads`) at the drawn ``units``. Returns what it wrote and the file."""
    k, t = com["unit"]["k"], time.time()
    reg = {*U.Shape(k).in_rows, "o1"}
    doc = U.registered_reads(win.program, win.tp, call, {v: r for v, (_n, r) in roots_of(com, okr).items()},
                             com["window"], U.window_open_root(com), win.obj["program"])
    if why := U.check_known(win.obj["program"], U.window_open_root(com), U.OUT_PORTS["ok"], doc["ports"][U.OUT_PORTS["ok"]]):
        raise SystemExit(f"call {call}'s registered file: {why}")
    for u in units:
        want = {p: (v, i) for p, (v, i) in U.unit_reads(win.program, win.tp, u).items() if p in reg}
        got = {p: (e["value"], CC.port_reads(e)[u]) for p, e in doc["ports"].items()}
        if got != want:
            raise SystemExit(f"unit {u}: the registered reads {got} are not the window's {want}")
    path.write_text(json.dumps(doc))
    return {"call": call, "ports": sorted(doc["ports"]),
            "roots": {e["value"]: e["root"] for e in doc["ports"].values()}, "seconds": round(time.time() - t, 1)}, doc


def registered_opens(pub: bytes, ports: list, port: str, row: int, win: SimpleNamespace, roots: dict, window_root: str
                     ) -> bool:
    """Whether table row ``row`` of registered port ``port`` in a shared-row public file opens its value's registered root
    (``roots``, `roots_of`, of the window whose open root is ``window_root``) at the position and through the path its
    header states: the check flock-verify ``--registered`` makes of each table row (`h1unit.verify_registered`)."""
    nl = pub.index(b"\n") + 1
    head, body, names = json.loads(pub[:nl]), pub[nl:], [p.name for p in ports]
    at = sum(int(head["shared_rows"][q]) for q in names[:names.index(port)]) + row
    e = head["registered"][port]
    leaves, root = roots[e["value"]]
    return U.verify_registered(win.obj["program"], e["value"], leaves, bytes.fromhex(root), e["positions"][row],
                               body[128 * at:128 * (at + 1)], [e["paths"][row]], window_root)


def ok_known_root(path: Path, win: SimpleNamespace, com: dict, doc: dict, unit: int) -> dict:
    """A ``--registered`` file ``doc`` whose ``ok`` port keeps its known content (every unit's row the one word 1) and
    states the root of a registration of ``ok`` with ``unit``'s row 0 (a prover's, whose unit computed 0) under the same
    salts, written to ``path``: flock-verify must refuse it as its root is not the known rows' (`h1unit.check_known`).
    Returns the Python mirror's refusal of it and its acceptance of ``doc``."""
    o1, wr = U.OUT_PORTS["ok"], U.window_open_root(com)
    e = doc["ports"][o1]
    rows = np.array(e["known"]["rows"], np.uint16)
    rows[unit] = 0
    zero = U.register(win.obj["program"], U.OK_VALUE, rows, [bytes.fromhex(s) for s in e["known"]["salts"]], wr)
    bad = dict(doc, ports=dict(doc["ports"], **{o1: dict(e, root=zero.root.hex())}))
    path.write_text(json.dumps(bad))
    return {"unit": unit, "root": zero.root.hex(), "honest": U.check_known(win.obj["program"], wr, o1, e),
            "refused": U.check_known(win.obj["program"], wr, o1, bad["ports"][o1])}


def credit_descriptor(win: SimpleNamespace) -> bytes:
    """The window Program's descriptor as a period's ``circuit`` hashes it and `credit.layout_of` reads it: canonical JSON
    without annotations (`codec.program_digest`'s bytes, so its SHA-512 is the Program's digest)."""
    return canonical_json({k: v for k, v in encode_program(win.program).items() if k != "annotations"})


def findings_of(doc: dict) -> dict:
    """A ``--registered`` file's ports as a session verified against it states them (flock-verify's VERDICT
    ``registered``, `Registered.finding`: ``{port: {value, root}}``)."""
    return {p: {"value": e["value"], "root": e["root"]} for p, e in doc["ports"].items()}


def credit_registration(descriptor: bytes, drawn: list[int], entries: list, findings: dict[int, dict], window_root: str,
                        ok: dict) -> list | None:
    """Window credit's last two steps (`credit.window`'s 5 and 6) on h1_prove's evidence, ``[code, why]`` of the first
    that refuses, or None: the layout of ``descriptor`` (`credit.layout_of`, `credit.strata`), each session's ok rows
    held to the verifier's own (`credit._ok_held`, step 5's rule: each sampled proof block's finding opened ``o1`` at ``ok``,
    the value and root of the verifier's ``ok``, `ok_registration`), one accepted outcome per sampled proof block
    (`credit._proved`, ``drawn`` h1_prove's own draw, not credit's work-law draw), and the registration
    (`credit._registration`): the window's registered ``entries`` (the commitment's: no entry registers ``ok``), each
    proof's ``registered`` the finding of the session that proved the unit (``findings``, unit -> `findings_of`), under
    the open root ``window_root``, each call a step of its own (`h1unit.steps`). Each proof's ``program`` is the
    descriptor's SHA-512, the Program Lean was given. The
    steps before them (the challenger's log, the credit bits, the round roots) need a served window's ledger, which this
    harness has none of."""
    layout, why = CR.layout_of(descriptor)
    if why is None:
        ss, why = CR.strata(layout)
    if why is None:
        starts = list(itertools.accumulate((size for size, _work in CR.call_work(layout)), initial=0))[:-1]
        why = next((w for u, f in sorted(findings.items())
                    if (w := CR._ok_held(bisect.bisect_right(starts, u) - 1, f"of unit {u}", f, ok)) is not None), None)
    if why is None:
        program = hashlib.sha512(descriptor).hexdigest()
        proofs = [{"program": program, "registered": f, "unit": int(u), "verdict": "accept"} for u, f in sorted(findings.items())]
        why = CR._proved(sorted(drawn), proofs) or CR._registration(
            window_root, program, CR._prefix(descriptor), entries, ss, layout, len(layout["calls"]), proofs)
    return None if why is None else list(why)


def unrooted_ok(win: SimpleNamespace, com: dict) -> G.Registered:
    """The window's ok rows as `ok_registration` registers them, the same rows and salts, but under the registered domain
    of ``ok`` with no window root: the ok a session opens when its ``--registered`` file states ok's domain without the
    window's open root (flock-verify roots known rows at the domain the file gives)."""
    n = int(win.tp.population)
    return G.Registered.commit(win.obj["program"], U.OK_VALUE, np.tile(U.OK, (n, 1)).tolist(),
                               [U.ok_salt(com["window"], u) for u in range(n)], G.WORD_BITS, U.V2, None)


def forge(base: Path, out: Path, pub: bytes, outs: dict, rows: dict, circuit: Path) -> list[dict]:
    """`h1unit.forge_statement` at every output row (port, instance) where ``base``'s staged outputs (``outs``, its
    outs.npz) are not the window's committed ones (``rows``, its rows.npz), into ``out``: a statement whose committed
    rows are all the window's. Refused as vacuous where none differs."""
    diffs = [(p, i) for p in U.OUT_PORTS.values() for i in range(len(rows["units"]))
             if not np.array_equal(outs[p][i], rows[f"w_{p}"][i])]
    if not diffs:
        raise SystemExit(f"{out.name}: every output row the window commits is the circuit's, so no witness is wrong")
    got, src = [], base
    for p, i in diffs:
        got.append(U.forge_statement(src, out, pub, i, p, rows[f"w_{p}"][i], circuit))
        src = out
    return got


def wait_for(path: Path, what: str, pid: int | None) -> None:
    while not path.exists():
        if pid:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                if not path.exists():
                    raise SystemExit(f"the stager (pid {pid}) exited before {path}") from None
        time.sleep(5)
    log(f"{what}: {path}")


# -- prepare ----------------------------------------------------------------------------------------------------------------

def epoch(work: Path, name: str, layer: str, number: int, a, record: dict,
          program: str = "") -> tuple[dict, dict, bytes, Challenger]:
    """One window's epoch to its registration: commit (``program``: the window Program's digest, when known), the salt
    coin, serve. Returns its commitment, served root, the unit salt and the challenger (its sampling coin still closed)."""
    ch = Challenger(number, verifier.coins)
    rec = record.setdefault("windows", {}).setdefault(name, {})
    rec["coin_commitment"] = {"epoch": ch.epoch, "root": ch.commitment.root.hex()}
    t = time.time()
    args = ["commit", layer, name, str(work / name), str(work / f"{name}-commitment.json")]
    if a.k:
        args += ["--k", str(a.k)]
    if a.cols:
        args += ["--cols", str(a.cols)]
    if program:
        args += ["--program-sha512", program]
    prover(*args)
    com_bytes = (work / f"{name}-commitment.json").read_bytes()
    com = json.loads(com_bytes)
    rec["commit_seconds"] = round(time.time() - t, 1)
    after = digest(com_bytes)
    so = ch.open("salt", after)
    assert coin_reason(ch.commitment, so, "salt", after) is None
    salt = U.unit_salt(U.stand_in_reveal(so.coin), number)
    rec["salt_opening"] = so.to_json()
    rec["salt"] = {"by": "pouw_salt", "reveal": "stand-in: h1unit.stand_in_reveal of the salt coin", "window": number,
                   "stream": U.STREAM.hex(), "prover": U.PROVER.hex()}
    t = time.time()
    served = prover("serve", str(work / name), salt.hex(), str(work / f"{name}-served.json"), "--workers", str(a.workers))
    rec["serve_seconds"] = round(time.time() - t, 1)
    rec.update(tiles=com["tiles"], slot=com["slot"], calls={c["name"]: [com["slot"], com["unit"]["k"], c["n"]] for c in com["calls"]},
               served=served)
    log(f"{name}: {com['tiles']} tiles committed ({rec['commit_seconds']} s) and served ({rec['serve_seconds']} s)")
    return com, served, salt, ch


def registration(com: dict, served: dict) -> bytes:
    return digest(json.dumps(com, sort_keys=True).encode(), json.dumps({"receipt_root": served["receipt_root"]}).encode())


def lean_draw(lean: str, n: int, k: int, src: bytes) -> list[int]:
    p = subprocess.run([lean, "draw", "--population", str(n), "--k", str(k), "--stream", src.hex()], capture_output=True, text=True)
    lines = [x[5:] for x in p.stdout.splitlines() if x.startswith("DRAW ")]
    if p.returncode != 0 or len(lines) != 1:
        raise SystemExit(f"flock-verify draw: exit {p.returncode}: {(p.stderr or p.stdout)[-300:]}")
    return [int(u) for u in json.loads(lines[0])["units"]]


def draw_units(lean: str, coin: bytes, window: str, registration: str, population: int, k: int) -> list[int]:
    """The `k` units of `population` a window's sampling coin selects, after its registration (hex) was recorded: Lean
    ``flock-verify draw`` on the coin's stream under `DRAW_DOMAIN`, bound to the window's name and its registration."""
    key = derive(coin, DRAW_DOMAIN, {"window": window, "registration": registration})
    return sorted(lean_draw(lean, population, k, key.stream("draw", 2 * k * CC.width(population) + 64)))


def cmd_prepare(a) -> int:
    work, sw = Path(a.work), Path(a.stage)
    work.mkdir(parents=True, exist_ok=True)
    (sw / "requests").mkdir(parents=True, exist_ok=True)
    record: dict = {"schema": "verity/pouw/h1-prove/record/v1", "draw_k": a.draw}
    w1, w2 = f"{a.name}-w1", f"{a.name}-w2"
    com1, served1, salt1, ch1 = epoch(work, w1, a.layer, 1, a, record)
    # the window's public Program and partition: the verifier's own copies, and the shape the stager binds its statement to
    t = time.time()
    win = window_program(com1)
    calls, slot = U.window_shape(com1)
    (sw / "window.json").write_text(json.dumps({"k": com1["unit"]["k"], "slot": slot, "calls": [list(c) for c in calls]}))
    vp = work / "verifier"
    vp.mkdir(exist_ok=True)
    descriptor = canonical_json(encode_program(win.program))
    (vp / "program.json").write_bytes(descriptor)
    (vp / "partition.json").write_text(json.dumps(win.obj))
    record["program"] = {"descriptor_bytes": len(descriptor), "program_sha512": win.obj["program"],
                         "partition": P.digest(win.obj), "query": win.query, "population": win.tp.population,
                         "seconds": round(time.time() - t, 1)}
    log(f"the window's Program: {win.tp.population} units, {len(descriptor)} descriptor bytes "
        f"({record['program']['seconds']} s)")
    if why := check_registration(win, com1):
        raise SystemExit(f"{w1}'s registration is not of the verifier's Program: {why}")
    record["z_only_registration"] = {"refused": check_registration(win, z_only(com1))}
    if not record["z_only_registration"]["refused"]:
        raise SystemExit("a registration of the records' z alone is accepted")
    record["unrooted_registration"] = {"refused": check_registration(win, unrooted(com1))}
    if not record["unrooted_registration"]["refused"]:
        raise SystemExit("a registration of the window's tables with no window root is accepted")
    ok1 = ok_registration(win, com1)
    record["ok_registration"] = {w1: ok1.entry()}
    reg = registration(com1, served1)
    do = ch1.open("draw", reg)
    assert coin_reason(ch1.commitment, do, "draw", reg) is None
    T = com1["tiles"]
    units = draw_units(a.lean, do.coin, w1, reg.hex(), T, a.draw)
    u0 = units[0]
    by_call: dict[int, list[int]] = {}
    for u in units:
        by_call.setdefault(U.HW.call_of(win.program, u)[0], []).append(u)
    c0 = U.HW.call_of(win.program, u0)[0]
    units0 = by_call[c0]
    # one statement a drawn call: "honest" the first sampled proof block's (the negatives'), "honest-c<c>" each other's
    honest = {c: "honest" if c == c0 else f"honest-c{c}" for c in by_call}
    record["windows"][w1].update(registration=reg.hex(), draw_opening=do.to_json())
    record["draw"] = {"population": T, "k": a.draw, "units": units, "by": "flock-verify draw (Lean)",
                      "calls": {honest[c]: {"call": c, "units": us} for c, us in by_call.items()},
                      "reads": {str(u): U.unit_reads(win.program, win.tp, u) for u in units}}
    log(f"drew {units} of {T}: {record['draw']['calls']}")
    k = com1["unit"]["k"]
    shape = U.Shape(k, R, C)
    regs1, docs1, record["registered_reads"] = {c: vp / f"w1-c{c}.registered.json" for c in by_call}, {}, {}
    for c, us in by_call.items():
        record["registered_reads"][f"w1-c{c}"], docs1[c] = registered_file(regs1[c], win, com1, c, us, ok1)
    log(f"the verifier's registered reads of {w1}: {record['registered_reads']}")

    # the honest openings, a drawn call's units each, and each negative's tampered window's on the first drawn call's
    # units (every salt kept, one value of them changed), each checked against its own registration and the honest one,
    # with its own registered file where its z table moved
    opened, reads, regs = {}, {}, {honest[c]: regs1[c] for c in by_call}
    for tag, flag, us in (*((honest[c], None, by_call[c]) for c in by_call),
                          *((t, f, units0) for t, f in (*BINDING.items(), *IN_PROOF.items()))):
        priv = w1 if flag is None else f"{w1}-{tag}"
        com_t, served_t = com1, served1
        if flag:
            served_t = prover("tamper", str(work / w1), str(work / priv), str(work / f"{priv}-commitment.json"),
                              str(work / f"{priv}-served.json"), *tamper_flag(flag, units0, U.HW.call_units(win.program)[c0]))
            com_t = json.loads((work / f"{priv}-commitment.json").read_text())
            done = served_t["tampered"]
            record.setdefault("tampered", {})[tag] = done
            if not done["records"] and not done["receipts"]:
                raise SystemExit(f"{tag}: {flag} changed no record and no receipt of the window")
            if why := check_registration(win, com_t):
                raise SystemExit(f"{tag}'s registration is not of the verifier's Program: {why}")
            if U.registered_of(com_t) != U.registered_of(com1):
                regs[tag] = vp / f"{tag}.registered.json"
                record["registered_reads"][tag], _doc = registered_file(regs[tag], win, com_t, c0, units0, ok1)
            else:
                regs[tag] = regs1[c0]
        prover("open", str(work / priv), salt1.hex(), ",".join(map(str, us)), str(work / f"open-{tag}"))
        ops = json.loads((work / f"open-{tag}" / "opening.json").read_text())["units"]
        why = {u: check_opening(win, com_t, served_t, op, u) for u, op in zip(us, ops)}
        why_vs_honest = {u: check_opening(win, com1, served1, op, u) for u, op in zip(us, ops)}
        opened[tag] = [unit_bcs(shape, com_t, salt1, op) for op in ops]
        reads[tag] = [unit_registered(win, k, op, u, ok1) for u, op in zip(us, ops)]
        zy = f"zy@{U.HW.call_of(win.program, us[0])[0]}"
        record.setdefault("openings", {})[tag] = {
            "units": us, "registered": {zy: U.registered_of(com_t)[zy][1], "receipt_root": served_t["receipt_root"]},
            "refused": {str(u): w for u, w in why.items() if w},
            "against_honest_registration": {str(u): w for u, w in why_vs_honest.items() if w}}
        log(f"opening {tag}: {sum(1 for w in why.values() if w)} of {len(us)} refused against its registration, "
            f"{sum(1 for w in why_vs_honest.values() if w)} against the honest one")
        if why_t := {u: w for u, w in why.items() if w}:
            raise SystemExit(f"{tag}'s openings are refused against its own registration: {why_t}")
    # a prover that stages an A row the registration does not hold (`window.py open --tamper-a`): its opening is the honest one
    record["tampered_a"] = prover("open", str(work / w1), salt1.hex(), ",".join(map(str, units0)),
                                  str(work / "open-tampered-a"), "--tamper-a")["tampered"]
    # a prover that commits its ok rows afresh (`window.py open --unregistered-ok`): no registered position or path for o1
    prover("open", str(work / w1), salt1.hex(), ",".join(map(str, units0)), str(work / "open-ok-unregistered"),
           "--unregistered-ok")

    # the second window, its first sampled proof block's tile presented as window 1's
    com2, served2, salt2, _ch2 = epoch(work, w2, a.layer2, 2, a, record, win.obj["program"])
    if U.window_shape(com2) != U.window_shape(com1) or com2["tiles"] != T:
        raise SystemExit("the second window's shape (its slot and calls) is not the first's, so its Program is another")
    if why := check_registration(win, com2):
        raise SystemExit(f"{w2}'s registration is not of the verifier's Program: {why}")
    ok2 = ok_registration(win, com2)
    record["ok_registration"][w2] = ok2.entry()
    reg2 = vp / f"w2-c{c0}.registered.json"
    record["registered_reads"][f"w2-c{c0}"], _doc = registered_file(reg2, win, com2, c0, [u0], ok2)
    prover("open", str(work / w2), salt2.hex(), str(u0), str(work / "open-other"))
    op2 = json.loads((work / "open-other" / "opening.json").read_text())["units"][0]
    record["openings"]["other-window"] = {"unit": u0, "refused": check_opening(win, com1, served1, op2, u0),
                                          "against_own_window": check_opening(win, com2, served2, op2, u0)}
    log(f"opening other-window: refused against window 1 for {len(record['openings']['other-window']['refused'])} rows")

    # window credit's registration step on this window's evidence, each sampled proof block's finding the ports of its call's
    # --registered file (what a session verified against it states; `summarize` repeats it on Lean's own findings), and
    # refused for entries without the window root, sessions that opened ok at its rows registered without it, and
    # another window's root
    descriptor_c = credit_descriptor(win)
    (vp / "descriptor.json").write_bytes(descriptor_c)
    wr1, entries1, okv = U.window_open_root(com1), com1["registered"], {"value": U.OK_VALUE, "root": ok1.root.hex()}
    found = {u: findings_of(docs1[c]) for c, us in by_call.items() for u in us}
    oku = unrooted_ok(win, com1).root.hex()
    if oku == okv["root"]:
        raise SystemExit("the ok rows registered without the window root have the window's ok root: ok-unrooted is vacuous")
    found_oku = {u: dict(f, **{U.OUT_PORTS["ok"]: dict(f[U.OUT_PORTS["ok"]], root=oku)}) for u, f in found.items()}
    cr = {"window": w1, "window_root": wr1, "descriptor_sha512_is_program": hashlib.sha512(descriptor_c).hexdigest() == win.obj["program"],
          "reads": credit_registration(descriptor_c, units, entries1, found, wr1, okv),
          "unrooted": credit_registration(descriptor_c, units, unrooted(com1)["registered"], found, wr1, okv),
          "ok-unrooted": credit_registration(descriptor_c, units, entries1, found_oku, wr1, okv),
          "other-window-root": credit_registration(descriptor_c, units, entries1, found, U.window_open_root(com2), okv)}
    record["credit_registration"] = cr
    log(f"credit's registration on {w1}'s reads: {cr}")
    if (not cr["descriptor_sha512_is_program"] or cr["reads"] is not None
            or any((cr[x] or [None])[0] != "registration" for x in ("unrooted", "ok-unrooted", "other-window-root"))):
        raise SystemExit(f"window credit's registration step does not take {w1}'s evidence and refuse the unrooted: {cr}")
    # an ok registration whose root is not the known rows' (row u0 a 0): flock-verify refuses the file before any session
    kr = record["ok_known_root"] = ok_known_root(vp / "ok-known-root.registered.json", win, com1, docs1[c0], u0)
    log(f"ok-known-root: {kr}")
    if kr["honest"] is not None or not kr["refused"]:
        raise SystemExit(f"the known-content check does not hold ok's root to its known rows: {kr}")

    # the statements: each drawn call's units on the honest rows, the other window's unit, the tampered A row, and each
    # in-proof negative's base (the first drawn call's units on its window's rows) where its records are not the honest
    # window's
    sets, set_o = {c: f"{w1}-draw-c{c}" for c in by_call}, f"{w1}-u{u0}"
    set_h = sets[c0]
    req = lambda rows, set_, out, idx: {"rows": str(rows), "set": set_, "out": str(out), "indices": idx}  # noqa: E731
    stmts = {**{h: req(work / f"open-{h}" / "rows.npz", sets[c], sw / f"st-{h}", by_call[c]) for c, h in honest.items()},
             "other": req(work / "open-other" / "rows.npz", set_o, sw / "st-other", [u0]),
             "tampered-a": req(work / "open-tampered-a" / "rows.npz", set_h, sw / "st-tampered-a", units0),
             "ok-unregistered": req(work / "open-ok-unregistered" / "rows.npz", set_h, sw / "st-ok-unregistered", units0)}
    base = {t: sw / "st-honest" for t in IN_PROOF}
    for t in IN_PROOF:
        if record["tampered"][t]["records"]:
            stmts[f"base-{t}"] = req(work / f"open-{t}" / "rows.npz", set_h, sw / f"st-base-{t}", units0)
            base[t] = sw / f"st-base-{t}"
    if f"base-{OK_ZERO}" not in stmts:
        raise SystemExit(f"{OK_ZERO} moved no record, so no statement of it computes ok = 0 for ok-zero")
    for name, r in stmts.items():
        (sw / "requests" / f"{name}.json").write_text(json.dumps(r))

    wait_for(sw / "circuit.json", "circuit built", a.stager_pid)
    head = json.loads((sw / "circuit.json").read_text())
    record["circuit"] = {x: head[x] for x in head if x != "meta"}
    bound = {"partition": head["meta"].get("partition"), "program_sha512": head["meta"].get("program_sha512")}
    if bound != {"partition": {"rule": "verity/partition/v1", "digest": P.digest(win.obj)}, "program_sha512": win.obj["program"]}:
        raise SystemExit(f"the unit's statement binds {bound}, not the window's partition and Program")
    files = {**{h: (sets[c], opened[h], by_call[c], reads[h]) for c, h in honest.items()},
             **{t: (set_h, opened[t], units0, reads[t]) for t in (*BINDING, *IN_PROOF)},
             "other-window": (set_o, [opened["honest"][0]], [u0], [reads["honest"][0]]),
             "other-window-own": (set_o, [unit_bcs(shape, com2, salt2, op2)], [u0], [unit_registered(win, k, op2, u0, ok2)])}
    for tag, (set_name, bcs, indices, rd) in files.items():
        (vp / f"{tag}.pub.bin").write_bytes(U.verifier_pub(head, set_name, head["subcircuit"], bcs, indices, rd))
    for name, r in stmts.items():
        wait_for(Path(r["out"]) / "stage.json", f"staged {name}", a.stager_pid)
        st = json.loads((Path(r["out"]) / "stage.json").read_text())
        if "error" in st:
            raise SystemExit(f"staging {name} failed: {st['error']}")
        record.setdefault("staged", {})[name] = st
    fair = (*honest.values(), "other")
    if any(record["staged"][x]["outputs_differ"][p] for x in fair for p in U.OUT_PORTS.values()):
        raise SystemExit(f"an honest statement's outputs are not its window's: "
                         f"{ {x: record['staged'][x]['outputs_differ'] for x in fair} }")
    eq = {**{h: (sw / f"st-{h}" / "pub.bin").read_bytes() == (vp / f"{h}.pub.bin").read_bytes() for h in honest.values()},
          "other-window-own": (sw / "st-other" / "pub.bin").read_bytes() == (vp / "other-window-own.pub.bin").read_bytes()}
    record["prover_pub_is_verifiers"] = eq
    log(f"the prover's pub.bin is the verifier's own: {eq}")
    if not all(eq[h] for h in honest.values()):
        raise SystemExit(f"an honest statement's pub.bin is not the verifier's own file: {eq}")
    # the tampered A row's table row in its statement's own public file, as flock-verify --registered checks it
    ta = record["tampered_a"]
    roots1 = roots_of(com1, ok1)
    record["tampered_a"]["table_row_0_opens"] = {
        x: registered_opens((sw / f"st-{x}" / "pub.bin").read_bytes(), shape.ports(), ta["port"], 0, win, roots1, wr1)
        for x in ("honest", "tampered-a")}
    log(f"tampered-a: {ta}")
    if ta["table_row_0_opens"] != {"honest": True, "tampered-a": False}:
        raise SystemExit(f"the tampered A row's table row opens the registered root as {ta['table_row_0_opens']}, not only the honest one")
    # the ok row, bound through the verifier's own registration: each of honest's ok table rows opens ok's root; the base
    # statement of OK_ZERO (its unit computing ok = 0 where its record moved) states that 0 in its own public file, whose
    # o1 table row opens no root there; and a statement that commits its ok rows afresh names no o1 registered
    zero = record["tampered"][OK_ZERO]["records"][0]
    zi = units0.index(zero)
    if not np.array_equal(np.load(base[OK_ZERO] / "outs.npz")["o1"][zi], np.zeros_like(U.OK)):
        raise SystemExit(f"{OK_ZERO}'s base statement does not compute ok = 0 at unit {zero}, whose record moved")
    pubs = {x: (sw / f"st-{x}" / "pub.bin").read_bytes() for x in ("honest", "ok-unregistered", f"base-{OK_ZERO}")}
    heads = {x: json.loads(p[:p.index(b"\n")]) for x, p in pubs.items()}
    record["ok_negatives"] = {
        "honest_rows_open": [registered_opens(pubs["honest"], shape.ports(), "o1", i, win, roots1, wr1)
                             for i in range(len(units0))],
        "ok_zero": {"unit": zero, "instance": zi, "table_row_opens": registered_opens(
            pubs[f"base-{OK_ZERO}"], shape.ports(), "o1", zi, win, roots1, wr1)},
        "ok_unregistered": {"registered_ports": sorted(heads["ok-unregistered"]["registered"])}}
    log(f"ok: {record['ok_negatives']}")
    if (not all(record["ok_negatives"]["honest_rows_open"]) or record["ok_negatives"]["ok_zero"]["table_row_opens"]
            or "o1" in heads["ok-unregistered"]["registered"] or "o1" not in heads["honest"]["registered"]):
        raise SystemExit(f"the ok row is not bound through the verifier's registration: {record['ok_negatives']}")
    # the in-proof negatives: the first drawn call's statement on the tampered window's rows with every output row the
    # window's, under the verifier's own file of that window, so binding holds and the witness is wrong
    for t in IN_PROOF:
        rows_t, outs = dict(np.load(work / f"open-{t}" / "rows.npz")), dict(np.load(base[t] / "outs.npz"))
        moved = {units0.index(u) for u in record["tampered"][t]["records"] if u in units0}
        not_ok = {i for i in range(len(units0)) if not np.array_equal(outs["o1"][i], U.OK)}
        if not_ok != moved:
            raise SystemExit(f"{t}: the unit computes ok = 0 at instances {sorted(not_ok)}, its records moved at {sorted(moved)}")
        for d in (t, f"{t}{SKIP}"):
            record.setdefault("forged", {})[d] = forge(base[t], sw / f"st-{d}", (vp / f"{t}.pub.bin").read_bytes(), outs,
                                                       rows_t, sw / "circuit.txt")
        log(f"forged {t} on {base[t].name}: {record['forged'][t]}")
    st = record["staged"]
    r1, r2 = str(regs1[c0]), str(reg2)
    refusal = f"registered reads: port {ta['port']}: table row 0 does not open {ta['value']}'s registered root at position {ta['position']}"
    ok_zero = f"registered reads: port o1: table row {zi} does not open {U.OK_VALUE}'s registered root at position {zero}"
    ok_afresh = f"registered reads: the statement commits port o1's rows of {U.OK_VALUE} afresh, with no path to its registered root"
    plan = {"statements": [*({"tag": h, "dir": stmts[h]["out"], "n": len(by_call[c])} for c, h in honest.items()),
                           {"tag": "other", "dir": stmts["other"]["out"], "n": 1},
                           {"tag": "tampered-a", "dir": stmts["tampered-a"]["out"], "n": len(units0)},
                           {"tag": "ok-unregistered", "dir": stmts["ok-unregistered"]["out"], "n": len(units0)},
                           {"tag": "ok-zero", "dir": str(base[OK_ZERO]), "n": len(units0)},
                           *({"tag": d, "dir": str(sw / f"st-{d}"), "n": len(units0)} for t in IN_PROOF for d in (t, f"{t}{SKIP}"))],
            "checks": [*({"tag": h, "statement": h, "pub": str(vp / f"{h}.pub.bin"), "registered": str(regs1[c]),
                          "expect": "ACCEPT"} for c, h in honest.items()),
                       *({"tag": t, "statement": "honest", "pub": str(vp / f"{t}.pub.bin"), "registered": str(regs[t]),
                          "expect": "REJECT"} for t in BINDING),
                       {"tag": "other-window", "statement": "other", "pub": str(vp / "other-window.pub.bin"), "registered": r1,
                        "expect": "REJECT"},
                       {"tag": "other-window-own", "statement": "other", "pub": str(vp / "other-window-own.pub.bin"),
                        "registered": r2, "expect": "ACCEPT"},
                       # against its own public file, so only the registered reads can refuse it, and they must say why
                       {"tag": "tampered-a", "statement": "tampered-a", "pub": str(sw / "st-tampered-a" / "pub.bin"),
                        "registered": r1, "expect": "REJECT", "why": refusal},
                       # each against its own public file too: only the ok row's registered read can refuse it
                       {"tag": "ok-unregistered", "statement": "ok-unregistered",
                        "pub": str(sw / "st-ok-unregistered" / "pub.bin"), "registered": r1, "expect": "REJECT",
                        "why": ok_afresh},
                       {"tag": "ok-zero", "statement": "ok-zero", "pub": str(base[OK_ZERO] / "pub.bin"),
                        "registered": str(regs[OK_ZERO]), "expect": "REJECT", "why": ok_zero},
                       # refused reading its --registered file, before any session: ok's root is not its known rows'
                       {"tag": "ok-known-root", "statement": "honest", "pub": str(vp / "honest.pub.bin"),
                        "registered": str(vp / "ok-known-root.registered.json"), "expect": "INPUT", "why": kr["refused"]},
                       *({"tag": d, "statement": d, "pub": str(vp / f"{t}.pub.bin"), "registered": str(regs[t]),
                          "expect": "REFUSED"} for t in IN_PROOF for d in (t, f"{t}{SKIP}"))]}
    if a.only:
        keep = set(a.only.split())
        plan["checks"] = [c for c in plan["checks"] if c["tag"] in keep]
        used = keep | {c["statement"] for c in plan["checks"]}
        plan["statements"] = [s for s in plan["statements"] if s["tag"] in used]
    (work / "plan.json").write_text(json.dumps(plan, indent=1))
    (work / "record.json").write_text(json.dumps(record, indent=1))
    with open(work / "plan.tsv", "w") as f:
        for s in plan["statements"]:
            f.write(f"S\t{s['tag']}\t{s['dir']}\t{s['n']}\n")
        for c in plan["checks"]:
            f.write(f"C\t{c['tag']}\t{next(s['dir'] for s in plan['statements'] if s['tag'] == c['statement'])}\t{c['pub']}"
                    f"\t{c['registered']}\n")
    log(f"planned: {len(plan['statements'])} statements, {len(plan['checks'])} Lean checks; staged in "
        f"{st['honest']['seconds']} s (honest, {st['honest']['instances']} instances), {st['other']['seconds']} s (other)")
    return 0


# -- summarize --------------------------------------------------------------------------------------------------------------

def _time(p: Path) -> dict | None:
    return json.loads(p.read_text()) if p.exists() else None


def _verdict(p: Path) -> dict | None:
    """The check's verdict over every session it verified: accepted iff each was, with the first refusal's reason."""
    if not p.exists():
        return None
    v = [json.loads(x[8:]) for x in p.read_text().splitlines() if x.startswith("VERDICT ")]
    if not v:
        return None
    return {**v[0], "accepted": all(x.get("accepted") for x in v), "sessions": len(v),
            "why": next((x.get("why", "") for x in v if not x.get("accepted")), "")}


def lean_phases(err: Path) -> dict | None:
    """flock-verify's own timings from its stderr: ``inputs`` (reading and parsing the statement's files, once a process),
    each statement ``setup``, and each session's verification, in ms."""
    if not err.exists():
        return None
    out: dict = {"inputs_ms": None, "setup_ms": [], "session_ms": []}
    for x in err.read_text(errors="replace").splitlines():
        head, _, ms = x.rpartition(": ")
        if not ms.endswith(" ms") or not ms[:-3].isdigit():
            continue
        if head == "inputs":
            out["inputs_ms"] = int(ms[:-3])
        elif head.startswith("setup"):
            out["setup_ms"].append(int(ms[:-3]))
        else:
            out["session_ms"].append(int(ms[:-3]))
    return out


def _verdict_word(v: dict | None, err: Path) -> str:
    if v is None:
        tail = err.read_text().strip().splitlines()[-1:] if err.exists() else []
        return "REFUSED" if tail and "refused" in tail[0] else "NONE"
    return "ACCEPT" if v.get("accepted") else "REJECT"


def _tagged(p: Path, tag: str) -> dict | None:
    """The last ``TAG<tab>{json}`` line of a Flock output."""
    if not p.exists():
        return None
    got = [x.split("\t", 1)[1] for x in p.read_text(errors="replace").splitlines() if x.startswith(tag + "\t")]
    return json.loads(got[-1]) if got else None


def layers(out: Path, tag: str) -> dict:
    """Where a statement's proof stopped: the prover's exit and its own reason (its LIVE line), the in-session verifier's
    (the serve side's) last SESSION verdict, and the tail of the prover's stderr."""
    t, po, log_ = _time(out / tag / "prove.time"), _tagged(out / tag / "prove.out", "LIVE"), out / tag / "serve.log"
    sess = [json.loads(x.split("\t", 1)[1]) for x in log_.read_text(errors="replace").splitlines()
            if x.startswith("SESSION\t")] if log_.exists() else []
    err = out / tag / "prove.err"
    return {"prover_exit": t and t["exit"], "prover_stopped": (po or {}).get("prover_stopped"),
            "prover_accepted": (po or {}).get("accepted"),
            "session": {k: sess[-1].get(k) for k in ("accepted", "aborted", "why")} if sess else None,
            "prove_err": err.read_text(errors="replace").strip().splitlines()[-3:] if err.exists() else None}


def refused_by(lay: dict, lean: str, why: str) -> list[str]:
    """Each layer that refused an in-proof negative, with its message."""
    out = []
    if lay["prover_stopped"]:
        out.append(f"prover: {lay['prover_stopped']}")
    elif lay["prover_exit"] not in (0, None):
        out.append(f"prover: exit {lay['prover_exit']}: {' | '.join(lay['prove_err'] or [])}")
    s = lay["session"]
    if s and (s["accepted"] is False or s["aborted"]):
        out.append(f"in-session verifier: accepted={s['accepted']} aborted={s['aborted']} why={s['why']}")
    if lean == "REJECT":
        out.append(f"Lean: {why}")
    return out


def lean_credit(work: Path, out: Path, record: dict) -> dict:
    """`credit_registration` of the first window on Lean's own findings: each sampled proof block's the ``registered`` of the
    VERDICTs of the honest check of its call (every session of a check the same), under the verifier's own descriptor,
    the window's public commitment and ``ok``, and the open root the verifier computes. Not run (``ran`` false) when an
    honest check has no VERDICT (``--only``)."""
    cr = record.get("credit_registration")
    if not cr:
        return {"ran": False, "why": "the record has no credit registration (a window from before PRF-127)"}
    found: dict[int, dict] = {}
    for tag, call in record["draw"]["calls"].items():
        p = out / "lean" / f"{tag}.out"
        regs = [json.loads(x[8:]).get("registered") for x in p.read_text().splitlines() if x.startswith("VERDICT ")] \
            if p.exists() else []
        if not regs:
            return {"ran": False, "why": f"{tag} has no Lean VERDICT"}
        if any(r != regs[0] for r in regs):
            return {"ran": True, "result": ["registration", f"{tag}'s sessions state different registered findings"]}
        found.update({u: regs[0] for u in call["units"]})
    com = json.loads((work / f"{cr['window']}-commitment.json").read_text())
    ok = record["ok_registration"][cr["window"]]
    got = credit_registration((work / "verifier" / "descriptor.json").read_bytes(), record["draw"]["units"],
                              com["registered"], found, U.window_open_root(com), {"value": ok["value"], "root": ok["root"]})
    return {"ran": True, "units": sorted(found), "result": got}


def cmd_summarize(a) -> int:
    work, out = Path(a.work), Path(a.out)
    plan = json.loads((work / "plan.json").read_text())
    record = json.loads((work / "record.json").read_text())
    proofs = {}
    for s in plan["statements"]:
        t = _time(out / s["tag"] / "prove.time")
        po = out / s["tag"] / "prove.out"
        text = po.read_text(errors="replace") if po.exists() else ""
        pe = out / s["tag"] / "prove.err"
        fw = text + (pe.read_text(errors="replace") if pe.exists() else "")
        proofs[s["tag"]] = {"instances": s["n"], "prove": t, "prove_out": text[-1500:] or None,
                            "serve": _time(out / s["tag"] / "serve.time"), "firewall_timed": "t.firewall" in fw}
    firewall = a.firewall == "1"
    checks, ok = [], bool(record["z_only_registration"]["refused"])
    if firewall and not proofs.get("honest", {}).get("firewall_timed"):
        print("FAIL: H1_FIREWALL=1 but the honest prover's records carry no firewall times")
        ok = False
    for c in plan["checks"] + [{"tag": f"st-{x}", "statement": "honest", "expect": "ACCEPT" if x == "honest" else "REJECT"}
                               for x in a.controls.split()]:
        v = _verdict(out / "lean" / f"{c['tag']}.out")
        word = _verdict_word(v, out / "lean" / f"{c['tag']}.err")
        t = _time(out / "lean" / f"{c['tag']}.time")
        extra = {}
        if c["expect"] == "REFUSED":
            lay = layers(out, c["statement"])
            by = refused_by(lay, word, (v or {}).get("why", ""))
            good = word != "ACCEPT" and bool(by)
            extra = {"layers": lay, "refused_by": by,
                     "lean_err": (out / "lean" / f"{c['tag']}.err").read_text(errors="replace").strip().splitlines()[-2:]
                     if (out / "lean" / f"{c['tag']}.err").exists() else None}
        elif c["expect"] == "INPUT":
            # refused reading its inputs: no VERDICT, exit 2, and the reason on stderr
            err = out / "lean" / f"{c['tag']}.err"
            text = err.read_text(errors="replace") if err.exists() else ""
            word = "INPUT" if v is None and (t or {}).get("exit") == 2 and c["why"] in text else word
            good = word == "INPUT"
            extra = {"why_expected": c["why"], "lean_err": text.strip().splitlines()[-2:]}
        else:
            # an accepted unit's index binds its rows only as far as Lean derived it from the verifier's own Program
            good = (word == c["expect"] and (word != "ACCEPT" or (v or {}).get("partition", {}).get("units") == "derived")
                    and c.get("why", "") in (v or {}).get("why", ""))
            extra = {"why_expected": c["why"]} if c.get("why") else {}
        ok &= good
        checks.append({"tag": c["tag"], "expect": c["expect"], "verdict": word, "pass": good,
                       "partition_units": (v or {}).get("partition", {}).get("units"),
                       "why": (v or {}).get("why", "")[:300], "lean_seconds": t and t["wall_s"],
                       "lean_maxrss_gb": t and round(t["maxrss_kb"] / 2**20, 1), "sessions": (v or {}).get("sessions"),
                       "lean_phases": lean_phases(out / "lean" / f"{c['tag']}.err"), **extra})
    credit = lean_credit(work, out, record)
    ok &= not credit["ran"] or credit["result"] is None
    honest = next(c for c in checks if c["tag"] == "honest")
    summary = {"schema": "verity/pouw/h1-prove/summary/v1", "pass": ok, "firewall": firewall,
               "unit": record["circuit"]["definition"], "k": record["circuit"]["k"], "tile_shape": [R, C],
               "ands": record["circuit"]["ands"], "k_log": record["circuit"]["k_log"],
               "circuit_bytes": record["circuit"]["circuit_bytes"], "circuit_build_seconds": record["circuit"]["build_seconds"],
               "window_tiles": record["draw"]["population"], "drawn_units": len(record["draw"]["units"]),
               "program": record.get("program"),
               "units": record["draw"]["units"], "draw_calls": record["draw"].get("calls"),
               "statements": len(plan["statements"]),
               "statement_instances": {s["tag"]: s["n"] for s in plan["statements"]},
               "epsilon_at_delta": {f"{d}": round(1 - d ** (1 / record["draw_k"]), 4) for d in (0.5, 0.1, 0.01)},
               "proving_seconds": {k: (v["prove"] or {}).get("wall_s") for k, v in proofs.items()},
               "lean_verify_seconds": honest["lean_seconds"], "checks": checks, "proofs": proofs,
               "openings": record["openings"], "prover_pub_is_verifiers": record["prover_pub_is_verifiers"],
               "registered_reads": record.get("registered_reads"), "tampered_a": record.get("tampered_a"),
               "tampered": record.get("tampered"), "forged": record.get("forged"),
               "z_only_registration": record["z_only_registration"],
               "unrooted_registration": record.get("unrooted_registration"),
               "credit_registration": {"from_reads": record.get("credit_registration"), "from_lean": credit},
               "windows": {w: {k: v[k] for k in ("tiles", "slot", "calls", "commit_seconds", "serve_seconds") if k in v}
                           for w, v in record["windows"].items()},
               "tile": U.tile(), "z_is_served_product": {w: v["served"].get("z_is_served_product")
                                                          for w, v in record["windows"].items()},
               "xa": {w: v["served"].get("xa") for w, v in record["windows"].items()},
               "xa_flip": (record.get("tampered") or {}).get("in-xa", {}).get("xa_flip"),
               "staged": record["staged"]}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("pass", "unit", "k", "ands", "window_tiles", "drawn_units", "statements",
                                                "proving_seconds", "lean_verify_seconds")}))
    print(f"  z-only-registration  refused by check_registration: {bool(record['z_only_registration']['refused'])}")
    print(f"  credit-registration  on Lean's findings: {'accepted' if credit['ran'] and credit['result'] is None else credit}")
    for c in checks:
        print(f"  {c['tag']:<20} expect {c['expect']:<7} got {c['verdict']:<8} {'ok' if c['pass'] else 'FAIL'}  "
              f"{c['lean_seconds']} s  units {c['partition_units']}  {c['why'][:120]}")
        for r in c.get("refused_by", []):
            print(f"      refused by {r[:300]}")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--work", required=True)
    p.add_argument("--stage", required=True)
    p.add_argument("--lean", required=True)
    p.add_argument("--layer", required=True)
    p.add_argument("--layer2", required=True)
    p.add_argument("--name", default="h1")
    p.add_argument("--draw", type=int, default=8)
    p.add_argument("--k", type=int, default=0)
    p.add_argument("--cols", type=int, default=0)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--stager-pid", type=int, default=0)
    p.add_argument("--only", default="", help="the plan's statements and Lean checks to run, by tag (space-separated); "
                                              "each check's statement is kept")
    p = sub.add_parser("summarize")
    p.add_argument("--work", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--controls", default="")
    p.add_argument("--firewall", default="0", help="1: the provers ran under the ZK server (FC_FIREWALL=1), whose times "
                                                    "the honest prover's records must carry")
    a = ap.parse_args(argv)
    return {"prepare": cmd_prepare, "summarize": cmd_summarize}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
