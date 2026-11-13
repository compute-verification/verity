"""A window's audit from its records alone: what an integration hands Verity (`verity.commitments.windows.interface`), and nothing else.

The records, every one in the window's directory but the advice's opening:

* **the circuit** in proof blocks with its public layout (``circuit.json``), which the registration binds by SHA-512;
* **the compiler and the advice**, when the circuit depends on run-time choices: the registration's window names the compiler
  (``compiler``: its name and digest, which carries the public cap on the advice's length, and the commit its code is at)
  and the advice's hiding commitment (``advice``: its root and its length in bits, which is charged); the advice is a bit
  string the audit never interprets;
* **the values**, committed before any coin: the registration (``registration.json``) and the files `rows.write` writes;
* **on a sample, the sampled proof blocks' openings** (``m<k>-private.bin``), which reach only the prover behind the ZK server.

The verifier's window is `interface.of_records`'s, each template resolved in Verity's catalog (`catalog_resolve`), or, for a
template the verifier lowers from its own input set (``--template-sets``), that set's Definition; a window with row tables
is refused until its records carry their segment roots. Before the sample the audit holds the registration to that circuit,
to the run root the run published (``--run``) and, for a compiled circuit, to the compiler and advice it was compiled from
(`source`), and runs the compile check (`compile_check`, the named gap until the compile unit exists): the advice's opening
opens the registered root under the registered compiler, and that compiler, run as the command the verifier's own table
names for it (``--compilers``) from a checkout of the registered commit whose code hashes to the registered digest, makes
this circuit from the advice and charges the registered length. A circuit with no compiler is the one the verifier
registered (``--circuit``). Then `a5.audit`, its stream's rule the rate its caller states (``--law``: ``rate:NUM/DEN``,
``stratified-rate:NUM/DEN`` or ``work-rate:NUM/DEN``; a fixed count ``subset:K``, ``stratified:K`` or ``work:K`` is refused
by name, `service.Law`) and the window's sampling design the one that rate names over the window's own proof blocks
(`verifier_window`), which the stream holds the registration to:
the registration checked by Lean and Python, the sample on live coins read after
the receipt, Flock ``--zk`` behind the Lean ZK server on the sampled members, Lean's verdicts and the audit record. Then the
advice was fixed before the sample (`fixed_before_draw`), and advice re-committed after the coins is refused (`after_draw`).
Units whose outputs are served out (``--outputs`` and the circuit's registered reads of output ports, `output_class`) would be
proved whole beside the sample only on a closure map, which the stream refuses, so a5 refuses such a window before the
sample (a5's `output_refusal`).
The summary reports the leak bound's count beside the advice's charge, ``vacuous`` while the circuit's other inputs exceed
it (`leak_count`; a record that declares its inputs, `declared_count`): the run figure, and for a record that declares
its window's ingress the window figure beside it, with the premise it rests on named. An integration's labels are never read.

A circuit that reads registered values at public positions (`interface.Read`) is audited against the entries the verifier
pinned before the run (``--registered FILE``: ``{"program", "values": {param: entry}}`` or a list of them, each entry core's
``registered.Registered.entry()``, its domain held to the one its program, value and rows make, `pinned`): each sampled
member's ``--registered`` file comes from the record and those entries alone (`interface.registered_reads`), and Lean's
registered reads open every row of its read ports against the pinned root at the row the record reads for its unit
(a5's ``registered``, with its negatives: no registered reads, another root, every unit reading another row, a port
committed afresh); a5 adds the member's other ports at the registration's roots (`rows.own_reads`). The reads
of a pinned parameter leave the free inputs where the window's root for it is the pinned root: `declared_count` counts
them registered, its pins the ``--registered`` file's roots and the window's roots those its registration records
(`window_roots`: its ``registered_roots``, `interface.Window.write`, and its committed values' entries). A record that
lists its members (a served window's) declares its reads beside them, as a served window reads its model's weights
(each ``weights/<label>@<words>``, pinned at the root of the committer's weights registration): each read port its
members' own rows or a shared-row table of them (`interface.listed_reads`), audited and counted as a stated record's
reads are, and the count names every parameter it reads whose window root is no pin (``unpinned``), whose reads count
free. A history table (`interface.History`) is pinned at the root the window's registration records for it
(`history.pins`), which a5's registration check before the draw and its stream's registrations hold to the root the
verifier of record computes over its writers' public bodies (a5's ``history``, `history.params` of the same pins as the
reads); a held run's strings it doesn't hold, so a window with one is refused.

A registration whose window commits per-window values (``activations``: ``{program, values: {value: entry}}``, a
loop's activation rows) has them fixed before the sample with the rest of the registration;
the audit holds each entry's domain to the one core's ``registered.domain`` makes of the program, the value, its rows and
the window's run root (`committed_activations`), and reports their roots. Those the circuit doesn't read take nothing from
the free inputs. Those it reads as ties (a wired loop's activations: an output port writes every row an input port
takes, `tied_values`) are pinned from the registration beside ``--registered``'s (`window_pins`), so Lean's registered
reads tie each reader's row to its writer's. One it reads only through input ports is opened against its committed root
too, but the prover chose its rows: its reads count free, and the count names it (``leak.untied``). PoUW's H1 tiles
have theirs pinned through ``--registered``, from the window's entries.

A circuit whose program takes values ahead of its templates' parameters (`interface.Registered.values`: a served window's
H1 tables, or each Pearl+ call's A rows and y, which its proof blocks make where the model makes them) has them registered
first by an earlier record W (``--tie W``: PoUW H1's registration of the window, fixed before H1's sample coin, or Pearl+'s
window record of its retained timed rounds, written after the retention and before the verifier's coin), and the window's
registration, chained right after W, registers each as
W's entry of it (one-stage's ``outputs``, `tie_of`): H1's table `credit.window_table` of it under ``--tie-prefix``, or
Pearl+'s by the value's own name. R7 takes its domain and root from W's entry, so the proof blocks read the values at W's
roots (each pinned at W's entry, `interface.registered_reads`) and nothing of them is registered afresh. Their readers are
sampled like any other proof block (under a stratified rate, a stratum a table), never proved whole as an output class.
The summary's ``tie`` names W, and for Pearl+'s W what holds its roots to the ones its passes credited: Pearl+'s verdict,
the prover's development verifier (`PEARL_CREDIT`). Once the records are read and before any proof, a ``TIE`` line and
``OUT/tie.json`` name W too, with the registration's ``previous`` and whether it is W's digest (``chained``), so an
audit stopped before its proofs end still says what it was tied to.

``--day N --day-log LOG``: the window is day N's, and the deployment's registration log declares the day's one circuit
(`verity.hardware.registration.Declaration`). R2 holds the registration's program to it (`declared_day`, one-stage's
Declared day), a negative moves the declaration to another program, and the summary's ``day`` names the day's declared
program and where the day came from (`DAY_SOURCES`). ``--day`` and the prover's log are the development fallback: they
bind the day to no time the auditor's device signed. ``--day-ledger LEDGER`` reads the day from the challenger's signed
ledger instead (`device_day`, the challenger's ``declared_day``): the UTC day of the window's signed open and the
declarations its lane signed before it, so a window that opens before UTC midnight and closes after is its open's day's.
It reads the device's own ledger whole: a copy PoUS cut to a window's sessions
(`pous.period.cite`) drops the declarations and the chain, and is refused. A window opened before its day's declaration,
or of a lane that never declares, has no declaration of its day, so R2 refuses it (``R2-undeclared``).

The negatives (`negatives` before the sample, a5's after it) cost Lean verifies of their own; ``--skip-negatives`` leaves
them out of an audit (its summary says so), for a run of several audits of one circuit, which runs them in its first.

``warm``: a window's templates and binding are public, so every template its composers make is composed into a5's compose
cache before the timed audit (`warm`, `a5.warm`), under the tree and ``FLOCK_*`` environment the audit then runs in, since
the cache's key covers both; the audit with ``--compose-cache`` then composes none. An audit's time quoted with the cache
quotes the warm's beside it.

    records_audit.py audit --window DIR --law L (--opening FILE --compilers FILE [--repo DIR] | --circuit SHA512) --run HEX \\
        [--template-sets DIR] [--outputs FILE] --m0-pythonpath PATH --bin flock-circuit --lean flock-verify --firewall flock-firewall \\
        --out DIR [--delta D] [--port P] [--ports N] [--forged] [--challenger local|loopback:HOST:PORT|gate:USER@HOST --main FILE] \\
        [--peer NAME --challenger-key RECORD [--gate-key KEY]] [--registered FILE] [--tie W [--tie-prefix P]] [--skip-negatives] \\
        [--compose-cache DIR] [--day N --day-log LOG [--day-params FILE] | --day-ledger LEDGER]
    records_audit.py warm --window DIR --law L [--template-sets DIR] --compose-cache DIR [--jobs N]
    records_audit.py count --circuit FILE --bits N [--catalog MODULE ...] [--pin NAME=ROOT ...] [--pins FILE ...] [--registration FILE]
    records_audit.py period --window DIR --audit DIR --out FILE [--pin NAME=ROOT ...] [--pins FILE ...]

``period``: a window's audit as one compute audit of the network certifier's period (`NetworkCheck.Period`'s rows form),
the ``audits`` entry `scan_period.rows_case` reads (`scan_zk.answers` writes the block round's in that shape): the window's
registration (``--window``'s ``registration.json``), the population Program's descriptor the audit sampled (``--audit``'s
``window/program.json``, the one the registration names), the audit's one-stage answer (`served_integrity.answer`:
undecided for an accepted audit whose record names no challenger key and session a check verified its coins under, or
doesn't say whether it wrote the case for the compiled check), and its terms, ``free`` and ``steps`` 0. ``free`` is the count `audit` reports (`declared_count` at the pins, with the window's own
values pinned, its untied ones free) but for the scan's state (`TIED`): the period refuses an audit whose state isn't
the scan's state tree (``stateTied``), so its reads count registered. Refused where the window commits no scan state.

``count``: a record that declares its inputs, counted before the committer writes its members' rows (`declared_count`): a
stated one (cuts, registered reads, ingress: `interface.Circuit.declared`) from its description alone, and one that lists
its members from them and the registered reads and window ingress it declares beside them (`interface.listed_reads`,
`interface.listed_ingress`), the window figure printed beside the run figure with its premise. A registered read counts as
registered only at a parameter ``--pin`` or ``--pins`` names (`pins_of`), with the root the verifier fixed before the run,
and only where the window's registration (``--registration``) records that root for it (`registration_roots`,
`interface.held_params`), any other free. A pin names its parameter alone: for vLLM's window circuit, each weight's
``weights/<label>@<words>``, at the hiding root the committer registered for it once per model and key before the first
window's coins (``--pins REGISTRATION``, the ``{program, values}`` the audit's ``--registered`` takes:
`verity_vllm.program.predict.window_weights.weights_registration`), never a root the verifier computes from a checkpoint,
and which `verity-vllm advice --weights-of-record` records in each window's registration. Its rotary table has a root
only where the committer registered the words its engine serves (within the analytic table's bound); else its reads stay
free.

``--compilers FILE``: ``{name: {"digest": hex, "argv": [...], "code": [path, ...], "pythonpath": [path, ...]}}``, the
registered compilers the verifier holds, each by the name and digest a registration names it under. A registration names
the commit its compiler's code is at; the audit checks that commit out of ``--repo`` (default this tree's repository) into
a worktree of its own, recomputes the digest there over the repository paths ``code`` (`interface.compiler_digest`),
refuses the compiler unless it is the registered one, and runs ``argv`` in that checkout with ``pythonpath`` (repository
paths) as its ``PYTHONPATH``: never the auditing tree's code. A registration naming no commit runs ``argv`` from the
auditing tree. The compile command is ``argv --advice FILE --out FILE``:

* ``--advice FILE``: the opened advice's bytes (the opening's ``advice``), nothing else of the window: the command's other
  inputs are fixed in ``argv``;
* it writes to ``--out`` the circuit record it compiles the advice to (``verity/window-circuit/v1``, `interface.circuit_bytes`),
  its `interface.RUN_FIELDS` (``run``, ``context``) empty, since it knows no run; the check compares the rest;
* it prints ``{"bits": N}`` as its last stdout line: the advice's length in bits it charges;
* it exits 0 having compiled, `REFUSED` (2) on advice it refuses, the reason on stderr, and any other code on an error.

``--challenger``: the sample on the auditor's challenger's coins (`a5_coins`), the only coins a sample is on, as `prove_path.py audit`
takes them: ``local``, a development session on this machine (with ``--deployed-key``, under a key made at the deployed
LMS parameters, the only ones SPOT's v2 reads a challenger's signatures under), ``loopback:HOST:PORT``, that session on the challenger
party's host (`challenger_party.py`, `verity run CONFIG --party challenger`), or ``gate:USER@HOST``, the auditor host through its SSH gate with
the lane's gate key ``--gate-key``, its challenger on ``--peer``'s 127.0.0.1 (so the audit runs on that peer), under the
deployment's key record ``--challenger-key``. The gate opens a root once, ever.

``--forged``: the window holds a forged unit, and the audit passes when every check before the sample passed, the audit
record rejects and the ZK server released no proof of the false member (`forged_outcome`). The prover runs with its
self-check skipped, ``FC_ZK_SELF_CHECK=skip``, so it proves the false witness, and the ZK server holds the proof back.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import math
import os
import random
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "one_stage"))

from verity.hardware import site as ST  # noqa: E402
from verity.hardware import registration as G  # noqa: E402
from verity.protocols.certifier.verifier.blocks import SCAN_STATE  # noqa: E402
from verity.protocols.pouw.verifier import credit as CR  # noqa: E402
from verity.protocols.spot.verifier import service as S  # noqa: E402
from verity.protocols.spot.verifier.one_stage import registered as REG, registration as R  # noqa: E402
from verity.circuits.codec import program_digest  # noqa: E402
from verity.commitments.windows import history as HW, interface as IF  # noqa: E402

#: where an audit's ports start when ``--port`` isn't given (`free_ports`): below Linux's ephemeral range (32768-60999), from
#: which outbound connections take theirs
PORTS = range(10000, 32000)
#: a compile command's exit code for advice it refuses (not advice it compiles: malformed, over its cap, or a choice it can't
#: compile), the reason on stderr; any other nonzero exit is an error. Either fails the compile check (``compiles``).
REFUSED = 2


def catalog_resolve(tid: str) -> Any:
    """`interface.resolve` over the whole of Verity's catalog, the registry's Definitions loaded too (a window's io shell and
    its blocks, which `interface`'s own walk can't import); a refusal names the registry modules that failed to import."""
    from verity.definitions.registry.catalog import load

    _, errors = load()
    try:
        return IF.resolve(tid)
    except ValueError as e:
        if errors:
            raise ValueError(f"{e} (registry modules that failed to import: {'; '.join(errors)})") from e
        raise


def free_ports(span: int, bases: Iterable[int] | None = None) -> int:
    """A base port whose ``span`` loopback ports are all free now, the first of ``bases`` (default every base of `PORTS` in
    random order). An audit's sessions listen at fixed offsets from it (here one a member with a proved unit, within
    ``--ports``, `a5.port_refusal`; ``a5.RANK_PORTS`` a rank in prove_path's); a fixed default made two audits on one node
    collide, the second's verifier failing to bind and its prover reaching the first's, which refused it."""
    if bases is None:
        bases = list(range(PORTS.start, PORTS.stop - span + 1, span))
        random.SystemRandom().shuffle(bases)
    for base in bases:
        try:
            with contextlib.ExitStack() as held:
                for port in range(base, base + span):
                    held.enter_context(socket.socket()).bind(("127.0.0.1", port))
            return base
        except OSError:
            continue
    raise SystemExit(f"no {span} free loopback ports in a row to listen on; give --port")


def _verdict(ok: Mapping[str, bool], detail: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {"ok": all(ok.values()), "codes": [c for c, v in ok.items() if not v], "detail": dict(detail or {})}


def tree_root() -> Path:
    """The auditing tree: the repository this file is in, whose compilers run for a registration naming no commit."""
    return Path(subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=HERE, capture_output=True, text=True,
                               check=True).stdout.strip())


@contextlib.contextmanager
def checkout(repo: Path, commit: str, at: Path) -> Iterator[Path]:
    """``commit`` of ``repo`` checked out at ``at`` in its own worktree, removed after; a commit ``repo`` doesn't hold is a
    ValueError (fetch it into ``repo`` first)."""
    p = subprocess.run(["git", "-C", str(repo), "worktree", "add", "--detach", "--quiet", str(at), commit],
                       capture_output=True, text=True)
    if p.returncode != 0:
        raise ValueError(f"{repo} can't check out commit {commit}: {p.stderr.strip()[-300:]}")
    try:
        yield at
    finally:
        subprocess.run(["git", "-C", str(repo), "worktree", "remove", "--force", str(at)], capture_output=True)


def compiler_code(entry: Mapping[str, Any], compiler: Mapping[str, Any], at: Path | None) -> None:
    """That the compiler about to run is the registered one, or a ValueError saying why not. At a checkout ``at`` of the
    registration's ``commit``, the digest this verifier computes (`interface.compiler_digest`, over the code paths its own
    table names, ``code``) must be the registered digest: the commit only says where to fetch that code. With no commit, the
    auditing tree's compiler runs, the table's digest the verifier's word for it, recomputed there when the table names
    ``code``."""
    if at is not None and not entry.get("code"):
        raise ValueError("the verifier's table names no code for this compiler, so a checkout's digest can't be recomputed")
    if entry.get("code"):
        got = IF.compiler_digest(compiler["name"], at or tree_root(), entry["code"])
        if got != compiler.get("digest"):
            where = f"commit {compiler['commit']}" if at is not None else "the auditing tree"
            raise ValueError(f"the code at {where} hashes to {got[:16]}, not the registered {str(compiler.get('digest'))[:16]}")


def _bits(stdout: str) -> int | None:
    """The length a compile command charges: its last stdout line's ``{"bits": N}``, or None for anything else."""
    try:
        bits = json.loads(stdout.strip().splitlines()[-1]).get("bits")
    except (IndexError, ValueError, AttributeError):
        return None
    return bits if isinstance(bits, int) and not isinstance(bits, bool) else None


def compile_check(circuit: Mapping[str, Any], window: Mapping[str, Any], opening: Any, compilers: Mapping[str, Any],
                  scratch: Path, repo: Path | None = None) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """The compile check: the registration's compiler is one the verifier holds, under the digest it holds (``compiler``); the
    opening opens the registered advice root under that compiler (``advice``); the code that runs is the registered compiler's
    (``code``, `compiler_code`: a registration naming a ``commit`` is compiled from that commit of ``repo``, default the
    auditing tree's repository, checked out in a worktree of its own, with the table's ``pythonpath`` under it); the
    compiler's command on the opened advice exits 0 (``compiles``), the circuit record it writes is ``circuit`` less its
    run's fields (``circuit``) and the length it charges is the registered one (``length``). The command runs only on advice
    that opens, and writes inside a fresh directory. -> (the verdict, the compiler and advice it verified, or None)."""
    c, adv = window.get("compiler"), window.get("advice")
    c, adv = (c if isinstance(c, Mapping) else {}), (adv if isinstance(adv, Mapping) else {})
    entry = compilers.get(c.get("name")) if isinstance(c.get("name"), str) else None
    ok = {"compiler": entry is not None and entry.get("digest") == c.get("digest"), "advice": False, "code": False,
          "compiles": False, "circuit": False, "length": False}
    detail: dict[str, Any] = {"compiler": dict(c), "advice": dict(adv)}
    try:
        data, root = IF.opened_advice(c, opening)
        ok["advice"] = root == adv.get("root")
    except (ValueError, KeyError, TypeError) as e:
        data, detail["opening"] = None, str(e)
    scratch.mkdir(parents=True, exist_ok=True)
    (scratch / "compiled.json").unlink(missing_ok=True)
    if ok["compiler"] and ok["advice"]:
        with tempfile.TemporaryDirectory(dir=scratch) as d, contextlib.ExitStack() as held:
            d, at, env = Path(d), None, None
            try:
                if "commit" in c:
                    at = held.enter_context(checkout(repo or tree_root(), str(c["commit"]), d / "tree"))
                    env = {**os.environ, "PYTHONPATH": os.pathsep.join(str(at / p) for p in entry.get("pythonpath", ()))}
                compiler_code(entry, c, at)
                ok["code"] = True
            except (ValueError, OSError, subprocess.CalledProcessError) as e:
                detail["code"] = str(e)
            if ok["code"]:
                (d / "advice.bin").write_bytes(data)   # the opened advice, the verifier's under the named gap
                t0 = time.time()
                p = subprocess.run([*entry["argv"], "--advice", str(d / "advice.bin"), "--out", str(d / "compiled.json")],
                                   cwd=at, env=env, capture_output=True, text=True)
                detail.update(compile_s=round(time.time() - t0, 1), rc=p.returncode, refused=p.returncode == REFUSED,
                              stderr=p.stderr[-2000:], at=c.get("commit", "the auditing tree"))
                ok["compiles"] = p.returncode == 0
                if ok["compiles"]:
                    try:
                        theirs = json.loads((d / "compiled.json").read_text())
                        ok["circuit"] = IF.circuit_of(theirs) == IF.circuit_of(circuit)
                        (scratch / "compiled.json").write_text(json.dumps(theirs))
                    except (OSError, ValueError, AttributeError) as e:
                        detail["output"] = f"{type(e).__name__}: {e}"
                    detail["bits"] = _bits(p.stdout)
                    ok["length"] = detail["bits"] is not None and detail["bits"] == adv.get("bits")
    v = _verdict(ok, detail)
    return v, ({"compiler": dict(c), "advice": dict(adv)} if v["ok"] else None)


def source(w: Mapping[str, Any] | None, held: Mapping[str, Any]) -> dict[str, Any]:
    """The registration's window held to what the verifier holds: the circuit record's SHA-512 (``circuit``), the run root it
    names (``run``), and the compiler and advice the compile check verified, or neither for a circuit with no compiler
    (``compiler``, ``advice``); ``compiled``: the compile check's verdict (true for a circuit with no compiler)."""
    w = w or {}
    ok = {"circuit": w.get("circuit") == held["circuit"], "run": w.get("run") == held["run"],
          "compiler": w.get("compiler") == held.get("compiler"), "advice": w.get("advice") == held.get("advice"),
          "compiled": bool(held["compiled"])}
    return _verdict(ok, {"kind": w.get("kind"), "circuit": w.get("circuit"), "run": w.get("run")})


#: what the window figure rests on and the run figure doesn't (`declared_count`), named beside it wherever it is printed
WINDOW_PREMISE = ("the earlier rows' committed activations the window reads (its window ingress) are its writers' outputs: "
                  "the link checks bind them (SPOT's wiring; the binding error ε_bind)")


def leak_count(members: Iterable[Any], external: Iterable[str], bits: int) -> dict[str, Any]:
    """The leak bound's count against the advice's charge (``bits``, the registered length; 0 for a circuit with no compiler).
    ``other_inputs`` is #(otherInputs \\ registered): ``otherInputs`` (``SampledCharge.lean``), the circuit's input
    gates no step lists as ingress, less the inputs the registration fixes. The circuit's input gates are each of the
    verifier's ``members``' parameter rows' bits (16 a u16 word: `interface.layout` refuses a port of part words), once per
    unit, or once per table row for a shared-row port (``tables``), and none for a port wired from other units' outputs
    (``wired``, which a5's wiring check holds). A record that declares no input lists no steps and registers no input (a
    row table's registered reads are refused, `verifier_window`), so none is ingress or registered: the ``external``
    ports are reported, not subtracted. A record that declares its registered reads or its window's ingress (vLLM's
    served windows) is counted by `declared_count`. ``bound`` is ``vacuous`` while ``other_inputs`` exceeds the charge."""
    inputs = wired = ext_bits = 0
    ext = set(external)
    for m in members:
        tables, wires = dict(m.tables), {p for p, _o, _ws in m.wired}
        for p, words in m.rows:
            n = 16 * int(words) * int(tables.get(p, m.n))
            inputs += n
            wired += n if p in wires else 0
            ext_bits += n if f"{m.template}/{p}" in ext and p not in wires else 0
    other = inputs - wired
    return {"inputs": inputs, "wired": wired, "ingress": 0, "registered": 0, "external": ext_bits, "other_inputs": other,
            "advice_bits": bits, "bound": "vacuous" if other > bits else "within the charge"}


def declared_count(doc: Mapping[str, Any], bits: int, resolve: Callable[[str], Any] = IF.resolve,
                   pins: Mapping[str, str] | str | None = None, roots: Mapping[str, str] | None = None,
                   untied: Iterable[str] = ()) -> dict[str, Any]:
    """The leak bound's count of a record that declares its inputs (`interface.input_count`: a stated record's from its
    description alone, a listed one's from its members, its registered reads and its window's ingress): its input rows
    less those its wires,
    registered reads of a parameter whose window root (``roots``, those the registration records: `window_roots`) is the
    verifier's pin (``pins``, ``{param: root}``) and the run's ingress fill
    leave ``free``, and ``other_inputs`` is those plus the advice's ``bits``, the compile unit's one input, a statement of
    its own. So ``other_inputs`` equals the charge exactly when the circuit leaves no input free. That is the run figure,
    the one to quote: only Lean's run ingress (the ingress gates ``freeInputs`` reads) counts as ingress. A listed
    record's window ingress (`interface.listed_ingress`: what its members read from rows before the window) adds the
    window figure beside it (``window``, those rows subtracted too), which holds only under its ``premise``
    (`WINDOW_PREMISE`). ``unpinned`` names the parameters the record reads whose window root is no pin of the
    verifier's (none pinned, another root, or none the registration records): their reads of input rows are free. An
    ``untied`` value
    (`window_pins`: a per-window value no output port of the circuit writes, whose rows the prover chose) is no pin here,
    though the audit opens its reads against its committed root: its input rows are free, and the count names it."""
    untied = set(untied)
    held = {p: r for p, r in pins.items() if p not in untied} if isinstance(pins, Mapping) else pins
    n = IF.input_count(doc, resolve, held, roots)
    reg = IF.declarations(doc)[2]
    if unpinned := sorted({r.param for r in (reg.reads if reg else ())} - IF.held_params(held, roots) - untied):
        n = {**n, "unpinned": unpinned}
    if untied:
        n = {**n, "untied": sorted(untied)}
    other = n["free"] + bits
    out = {**n, "other_inputs": other, "advice_bits": bits, "bound": "vacuous" if other > bits else "within the charge"}
    if n.get(IF.WINDOW_INGRESS):
        free = n["free"] - n[IF.WINDOW_INGRESS]
        out["window"] = {"ingress": n[IF.WINDOW_INGRESS], "free": free, "other_inputs": free + bits,
                         "bound": "vacuous" if free > 0 else "within the charge", "premise": WINDOW_PREMISE}
    return out


def pin_arg(s: str) -> tuple[str, str]:
    """``NAME=ROOT``: a registered parameter and the root the verifier pinned for it (hex digits)."""
    name, eq, root = s.partition("=")
    if not (name and eq and re.fullmatch(r"[0-9a-fA-F]+", root)):
        raise argparse.ArgumentTypeError(f"a pin is NAME=ROOT, ROOT hex digits, not {s!r}")
    return name, root


def pins_of(pin: Iterable[tuple[str, str]], files: Iterable[str] = ()) -> dict[str, str]:
    """The verifier's pins, ``{param: root}``: each ``--pin NAME=ROOT`` and each ``--pins FILE``'s, a JSON ``{param: root}``
    or a registration of values ``{program, values: {param: entry}}`` (or a list of them) whose entries' roots they are,
    each entry held to its domain as the audit's ``--registered`` holds it (`pinned`): vLLM's committer's weights
    registration (`window_weights.weights_registration`, each weight's ``weights/<label>@<words>``). Refused (ValueError)
    where two name one parameter, a root isn't hex digits, or the file is a weights record, which states each weight's
    shape and no root."""
    out = list(pin)
    for f in files:
        doc = json.loads(Path(f).read_text())
        if isinstance(doc, Mapping) and ("registered_entries" in doc or "registered_roots" in doc):
            raise ValueError(f"{f}: a weights record states each weight's shape, not its root: pin the committer's "
                             "registration of its weights")
        if isinstance(doc, list) or (isinstance(doc, Mapping) and "values" in doc):
            doc = {p: e["root"] for p, e in pinned(doc).items()}
        if not isinstance(doc, Mapping):
            raise ValueError(f"{f}: pins are {{param: root}}, not {type(doc).__name__}")
        out += list(doc.items())
    pins = dict(out)
    if len(pins) != len(out):
        raise ValueError(f"two pins name one parameter: {sorted(n for n in pins if [p for p, _ in out].count(n) > 1)}")
    IF.held_params(pins, {})
    return pins


#: the groups of committed values a window's registration may name, each ``{program, values: {param: entry}}``: its
#: per-window values (``activations``: `TorchLoop.commit_activations`, PoUW's operands) and values pinned for it
#: (``registered``: PoUW's weights), PoUW's `core_check.WINDOW_VALUES`
ENTRY_GROUPS = ("activations", "registered")


def window_roots(w: Mapping[str, Any]) -> dict[str, str]:
    """The roots a window's registration records for its parameters, by parameter: its registered roots
    (``registered_roots``, `interface.registered_roots`) and each committed value's entry's root (`ENTRY_GROUPS`). Refused
    (ValueError) where an entry names no root of hex digits or two of them give one parameter two roots."""
    roots = IF.registered_roots(w)
    for g in ENTRY_GROUPS:
        if w.get(g) is None:
            continue
        try:
            got = {str(v): e["root"] for v, e in w[g]["values"].items()}
        except (KeyError, TypeError, AttributeError) as e:
            raise ValueError(f"the window's {g} is {{program, values: {{param: entry}}}}, each entry with its root: "
                             f"{type(e).__name__}: {e}") from e
        IF.held_params({}, got)
        if two := sorted(v for v, r in got.items() if roots.setdefault(v, r).lower() != r.lower()):
            raise ValueError(f"the window's registration records two roots for {two}")
    return roots


def registration_roots(rec: Mapping[str, Any]) -> dict[str, str]:
    """The roots a window's registration records for its parameters (`window_roots`): core's registration's window's, or
    an integration's own window registration's, whose ``window`` is no mapping (vLLM's advice registration names the
    window's index there)."""
    w = rec.get("window")
    return window_roots(w if isinstance(w, Mapping) else rec)


def pinned(doc: Any) -> dict[str, dict[str, Any]]:
    """The registered values' entries the verifier pinned before the run (``--registered``), by parameter, each refused
    (ValueError) unless it names its parameter and its domain is the one core's ``registered.domain`` makes of the pin's
    program, the value and its rows: a root is pinned under its domain, never the prover's word for one. A list of pins
    is their values together, each under its own program (the weights under the model's, the output a network certifier
    relayed under the reading program's), a value pinned once."""
    if isinstance(doc, list):
        out: dict[str, dict[str, Any]] = {}
        for d in doc:
            got = pinned(d)
            if twice := sorted(set(got) & set(out)):
                raise ValueError(f"values {twice} pinned twice")
            out |= got
        return out
    try:
        program, values = doc["program"], doc["values"]
        for v, e in values.items():
            want = REG.domain(program, v, e["words"], e["leaves"], e.get(REG.WIDTH_KEY, REG.WORD_BITS)).domain_id.hex()
            if e.get("value") != v or e.get("domain") != want or not isinstance(e.get("root"), str):
                raise ValueError(f"the pin of {v!r} names another value or domain than {program!r}'s, or no root")
    except (KeyError, TypeError, AttributeError) as e:
        raise ValueError(f"a pin is {{program, values: {{param: entry}}}}, or a list of them: {type(e).__name__}: {e}") from e
    return {str(v): dict(e) for v, e in values.items()}


def pin_roots(entries: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    """The pins `interface.input_count` holds to a window's roots (`window_roots`), ``{param: root}``, of pinned entries
    (`pinned`, `window_pins`'): each entry's root alone, its domain and shape staying with the entries that
    `interface.registered_reads` opens against. Refused (ValueError) where a root isn't hex digits."""
    pins = {v: e["root"] for v, e in entries.items()}
    IF.held_params(pins, {})
    return pins


def tied_values(circuit: Mapping[str, Any], values: Iterable[str], resolve: Callable[[str], Any] = IF.resolve) -> set[str]:
    """Those of ``values`` (registered parameters of the stated record ``circuit``) that are ties: every leaf an input
    port's registered read takes is a leaf some output port's read of the circuit writes, so the Lean verifier's opening
    of each writer's output row against the value's root fixes the rows its readers take."""
    values = set(values)
    reg = IF.declarations(circuit)[2]
    if reg is None or not values:
        return set()
    lays = {t: IF.layout(resolve(t)) for t, _ in circuit["templates"]}
    written: dict[str, set[int]] = {}
    taken: dict[str, set[int]] = {}
    for r in reg.reads:
        if r.param not in values:
            continue
        outs = dict(lays[r.reader].outs)
        words = outs.get(r.port) or dict(lays[r.reader].rows)[r.port]
        leaves = (r.source(np.arange(r.reader_units))[:, None] + np.arange(words)).ravel()
        (written if r.port in outs else taken).setdefault(r.param, set()).update(leaves.tolist())
    return {v for v in values if v in taken and taken[v] <= written.get(v, set())}


def window_pins(w: Mapping[str, Any], circuit: Mapping[str, Any],
                resolve: Callable[[str], Any] = IF.resolve) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """The per-window values the circuit reads (its registered parameters among the registration's ``activations``), by
    parameter, each refused (ValueError) unless it names its value and a root and its domain binds the window's run root,
    as ``(ties, untied)``. The ties (`tied_values`: a wired loop's activations, each written by an output port and read by
    an input port) are pinned: the registration fixes them before the sample, and the Lean verifier's registered reads open
    each writer's and each reader's rows against the same root, so a reader's rows are its writer's. An untied value (read
    only by input ports) is opened against its committed root too, since the Lean verifier opens every registered read, but
    nothing in the circuit fixes its rows: the prover chose them, so its reads count free (`declared_count`)."""
    params = set((circuit.get("registered") or {}).get("params") or {})
    acts = w.get("activations") or {}
    values = {v: dict(e) for v, e in (acts.get("values") or {}).items() if v in params}
    for v, e in values.items():
        want = REG.domain(acts["program"], v, e["words"], e["leaves"], e.get(REG.WIDTH_KEY, REG.WORD_BITS),
                          window=w.get("run")).domain_id.hex()
        if e.get("value") != v or e.get("domain") != want or not isinstance(e.get("root"), str):
            raise ValueError(f"the window's value {v!r} names another value or a domain not bound to its run root, or no root")
    ties = tied_values(circuit, values, resolve)
    return {v: e for v, e in values.items() if v in ties}, {v: e for v, e in values.items() if v not in ties}


def tie_of(circuit: Mapping[str, Any], w: Mapping[str, Any] | None,
           prefix: str | None) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """The window's tie to ``w`` (``--tie``: W, the earlier registration that registered first the values the circuit's
    program takes, `interface.Registered.values`): a5's ``tie``, one-stage's ``log`` (W), ``registered`` (each value's
    words), ``outputs`` (each value W's entry of it) and ``schemas`` (each value's rows, `interface.value_format`), and the
    pins of the values, each W's entry under the value's name, which `interface.registered_reads` holds to the rows the
    value's readers read. W names a value under ``prefix``, its descriptor's ``PREFIX``, as its table
    `credit.window_table` (PoUW H1's registration of the window, each value a Commit table in ``row/v2`` rows), or, with
    no prefix, by the value's own name (Pearl+'s window record, `pearl_c.window`: each call's A rows and y, FP32 words in
    ``row-seg/v1`` rows, beside each weight's B rows, which the circuit doesn't take). ``({}, {})`` for a circuit that
    takes no values with no W. Refused (ValueError) for values without W, W without values, and W registering a value
    other than once."""
    reg = IF.declarations(circuit)[2]
    values = tuple(reg.values) if reg is not None else ()
    if not values and w is None:
        return {}, {}
    if w is None:
        raise ValueError(f"the circuit takes values {list(values)}, an earlier registration's: tie the audit to it (--tie W, "
                         "and --tie-prefix P for a W that names its tables under a PREFIX)")
    if not values:
        raise ValueError("the audit is tied to an earlier registration (--tie), and the circuit takes no values")
    entries: dict[str, list[dict[str, Any]]] = {}
    for e in w.get("registered") or ():
        entries.setdefault(e.get("value") if isinstance(e, Mapping) else None, []).append(dict(e))
    pins, outputs, d = {}, {}, R.digest(w)
    for v in values:
        name = v if prefix is None else CR.window_table(v, prefix)
        if len(got := entries.get(name, [])) != 1:
            raise ValueError(f"the earlier registration registers the circuit's value {v!r} as {name!r} {len(got)} times, "
                             "not once" + ("" if prefix is not None else " (a W naming its tables under its descriptor's "
                                                                          "PREFIX, H1's: give --tie-prefix)"))
        pins[v], outputs[v] = {**got[0], "value": v}, {"record": d, "value": name}
    return ({"log": [dict(w)], "registered": {v: int(pins[v]["words"]) for v in values}, "outputs": outputs,
             "schemas": {v: IF.value_format(reg, v)[0] for v in values}}, pins)


#: what holds the roots of Pearl+'s window record to the ones its passes credited, which an audit tied to it rests on
PEARL_CREDIT = ("W's roots are the ones Pearl+'s verdict credits (`served_passes.py verify --tie W`), the prover's "
                "development verifier: it reads the prover's salt key and opens drawn rows in the clear, values this audit's "
                "proof hides")


def declared_day(day: int | None, log: str | None, params: str | None,
                 w: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """``--day N --day-log LOG``: the window's day and the declarations of the registration log the verifier holds (its
    view's, `registration.check_log` under ``--day-params``, by default the log's composite under an auditor of no sheets or
    keys), as one-stage's R2 takes them (a5's ``declared``); empty without ``--day``. ValueError where core's check refuses
    the log, and where the tied W (``--tie``) names a day's declaration (its window's ``day`` and ``declaration``,
    `serve_day.Day.naming`) other than one of day N the log holds by digest, or names one and the audit has no ``--day``."""
    named = {k: (w.get("window") or {}).get(k) for k in ("day", "declaration")} if isinstance(w, Mapping) else {}
    if day is None:
        if log or params:
            raise ValueError("--day-log and --day-params go with --day")
        if any(v is not None for v in named.values()):
            raise ValueError(f"the tied registration names day {named['day']}'s declaration, and the audit has no --day: "
                             "R2 holds its window to that day with --day and --day-log")
        return {}
    if not log:
        raise ValueError("--day needs --day-log, the registration log whose declarations R2 reads")
    composite, entries = G.log_of(json.loads(Path(log).read_text()))
    p = G.params_of(json.loads(Path(params).read_text())) if params else (composite, ST.Auditor())
    got = G.check_log(p, (composite, entries))
    if got[0] != "accept":
        raise ValueError(f"--day-log {log}: core's check refuses it, {got[1]}")
    if w is not None:
        _held_naming(named, day, {_rebuilt(e.day, e.program, e.blocks) for e in entries
                                  if isinstance(e, G.Declaration) and e.day == day}, f"--day-log {log}")
    return {"day": day, "declarations": got[1]["declarations"]}


def _rebuilt(day: int, program: str, blocks: Any) -> str:
    """The digest a window names its day's declaration by (serve_day's ``named``): `registration.Declaration`'s with no
    ``previous``, rebuilt alike from the prover's log and from the challenger device's entry."""
    return G.Declaration(day, program, tuple(blocks)).digest()


def _held_naming(named: Mapping[str, Any], day: int, digests: set[str], where: str) -> None:
    if named.get("day") != day or named.get("declaration") not in digests:
        raise ValueError(f"the tied registration names day {named.get('day')}'s declaration "
                         f"{str(named.get('declaration'))[:16]}…, not one of day {day}'s in {where}")


#: where R2's day came from, as the summary's ``day`` says: the challenger device's signed ledger, or the development
#: fallback, DAY and the prover's own registration log, which binds no day to a time the auditor's device signed
DAY_SOURCES = {"device": "the challenger's signed ledger (--day-ledger)",
               "development": "development: --day and the prover's --day-log, not device-bound"}


def device_day(lines: list[str], root: str, key: Any, verify: Callable[[Any, bytes, bytes], bool],
               w: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """``--day-ledger LEDGER``: the window's day and its prover's declarations as the auditor's challenger device signed
    them, read by the challenger's ``declared_day`` from the device's ledger ``lines`` (``{day, declarations}`` in
    one-stage R2's shape: the UTC day of the window open's signed ``started_utc``, and the declarations its lane signed
    before that open), never ``--day``. The tied W's naming is held to `registration.Declaration` rebuilt from the
    device's entry of its day. ValueError (`challenger.CoinError`) where the reader refuses the ledger."""
    from verity.protocols.spot.device import challenger as Ch
    got = Ch.declared_day(lines, {"root": root}, key, verify)
    day = got["day"]
    declarations = [{"blocks": list(d["blocks"]), "day": d["day"], "program": d["program"]} for d in got["declarations"]]
    if w is not None:
        named = {k: (w.get("window") or {}).get(k) for k in ("day", "declaration")}
        _held_naming(named, day, {_rebuilt(d["day"], d["program"], d["blocks"]) for d in declarations if d["day"] == day},
                     "the challenger's ledger")
    return {"day": day, "declarations": declarations}


def tie_row(tied: Mapping[str, Any]) -> dict[str, Any]:
    """The audit summary's ``tie`` for a5's ``tie`` (`tie_of`): W's digest and the values tied to it, and for Pearl+'s window
    record (a W whose window names a Pearl-C scheme) what credits its roots (`PEARL_CREDIT`)."""
    w = tied["log"][0]
    scheme = (w.get("window") or {}).get("scheme")
    pearl = isinstance(scheme, str) and scheme.startswith("pearl-c-")
    return {"record": R.digest(w), "values": sorted(tied["outputs"]), **({"credited_by": PEARL_CREDIT} if pearl else {})}


def committed_activations(w: Mapping[str, Any]) -> dict[str, Any] | None:
    """The per-window values the registration's window commits (``activations``), each refused unless it names its value
    and a root and its domain binds the window's run root (a root committed in another window opens under none of this
    window's domains): -> {ok, codes, detail}, ``detail["values"]`` each value's root and shape; None for a window that
    commits none."""
    if "activations" not in w:
        return None
    try:
        program, values = w["activations"]["program"], dict(w["activations"]["values"])
        other = [v for v, e in values.items() if e.get("value") != v or not isinstance(e.get("root"), str) or e.get("domain")
                 != REG.domain(program, v, e["words"], e["leaves"], e.get(REG.WIDTH_KEY, REG.WORD_BITS),
                               window=w.get("run")).domain_id.hex()]
    except (KeyError, TypeError, AttributeError, ValueError) as e:
        return _verdict({"activations": False}, {"why": f"{type(e).__name__}: {e}"})
    return _verdict({"activations": bool(values), "domains": not other},
                    {"values": {v: {k: e.get(k) for k in ("root", "leaves", "words")} for v, e in values.items()},
                     "other_domains": other})


def negatives(w: Mapping[str, Any], held: Mapping[str, Any], circuit: Mapping[str, Any], scratch: Path) -> dict[str, dict]:
    """What the audit refuses before any sample about the records: a registration naming another circuit or another run root,
    one binding a compiler for a circuit registered with none; for a compiled circuit, a registration naming another
    compiler, another commit of it, another advice root or another length, or binding neither (a circuit passed off as the
    compiler's), the advice's opening under another salt (which opens no registered root), and a circuit record other than
    the compiler's output (``held["output"]``, the compile check's, its first template one unit short)."""
    cases = {"registration-naming-another-circuit": {**w, "circuit": "00" * 64},
             "registration-naming-another-run": {**w, "run": "00" * 64}}
    if held.get("compiler") is None:
        cases["registration-binding-a-compiler-for-a-fixed-circuit"] = {**w, "compiler": {"name": "c", "digest": "00" * 64}}
    else:
        cases |= {"registration-naming-another-compiler": {**w, "compiler": {**w["compiler"], "digest": "00" * 64}},
                  "registration-naming-another-advice-root": {**w, "advice": {**w["advice"], "root": "00" * 64}},
                  "registration-charging-another-length": {**w, "advice": {**w["advice"], "bits": w["advice"]["bits"] + 1}},
                  "registration-binding-no-compiler-or-advice": {k: v for k, v in w.items() if k not in ("compiler", "advice")}}
        if "commit" in w["compiler"]:
            cases["registration-naming-another-commit"] = {**w, "compiler": {**w["compiler"], "commit": "00" * 20}}
    out = {name: source(x, held) for name, x in cases.items()}
    if held.get("compiler") is not None:
        op = dict(held["opening"])
        op["salts"] = [f"{int(op['salts'][0][:2], 16) ^ 1:02x}{op['salts'][0][2:]}", *op["salts"][1:]]
        v, _ = compile_check(circuit, w, op, held["compilers"], scratch, held.get("repo"))
        out["opening-under-another-salt"] = v
        if (theirs := held.get("output")) is not None:
            (t0, n0), *rest = theirs["templates"]
            short = {**theirs, "templates": [[t0, n0 - 1], *rest]}
            out["circuit-record-not-the-compilers-output"] = {"ok": IF.circuit_of(theirs) == IF.circuit_of(short),
                                                              "codes": ["circuit"]}
    return {name: {"refused": not v["ok"], "codes": v["codes"]} for name, v in out.items()}


def fixed_before_draw(receipt: Mapping[str, Any], rec: Mapping[str, Any], bound: Mapping[str, Any] | None) -> dict[str, Any]:
    """That the circuit, its compiler and its advice were fixed before the coins, checked rather than assumed: the digest of the
    registration (which covers the window's ``circuit``, ``compiler`` and ``advice``) is the one the sample's receipt names, the
    receipt the stream wrote when it received the registration and before its select read any coin (the challenger's
    coins also commit to that digest before any is read: a5_coins); and that registration
    binds the compiler and advice the compile check verified (``bound``, None for a circuit with none). -> {ok, codes, detail}:
    ``after-the-draw`` where the sample followed another registration or another stream's, ``compiler`` or ``advice`` where the
    registration binds another."""
    import a5 as A5

    w, b = rec.get("window") or {}, bound or {}
    ok = {"after-the-draw": receipt.get("registration") == R.digest(rec) and receipt.get("verifier") == A5.CHALLENGER,
          "compiler": w.get("compiler") == b.get("compiler"), "advice": w.get("advice") == b.get("advice")}
    return _verdict(ok, {"drawn_after": receipt.get("registration"), "registration": R.digest(rec),
                         "received": receipt.get("received")})


def after_draw(receipt: Mapping[str, Any], rec: Mapping[str, Any], held: Mapping[str, Any]) -> dict:
    """The negative: once the coins are read, the prover commits the same advice again (fresh salts, so a fresh root) and hands
    in a registration binding that root, with its opening. Every check before the sample passes on it (the new opening opens
    the new root, the same advice compiles to the same circuit, `source` holds), and `fixed_before_draw` refuses it
    (``after-the-draw``): the sample followed the registration received first."""
    c = held["compiler"]
    data = bytes.fromhex(held["opening"]["advice"])
    late = IF.commit_advice(c, data)
    opens = IF.opened_advice(c, IF.advice_opening(data, late))[1] == late.root.hex()
    advice = {**rec["window"]["advice"], "root": late.root.hex()}
    late_rec = {**rec, "window": {**rec["window"], "advice": advice}}
    src = source(late_rec["window"], {**held, "advice": advice})
    v = fixed_before_draw(receipt, late_rec, {"compiler": held["compiler"], "advice": advice})
    return {"refused": not v["ok"], "codes": v["codes"], "before_draw_checks_pass": opens and src["ok"]}


def composer(fn: Any, st: Mapping[str, Any]) -> Callable[[Path, str, str], dict[str, Any]]:
    """A template's circuit from its Definition and its statement in the circuit record (Flock's `UnitStager` on the record's
    ``in_rows``, ``out_rows`` and a flat statement's ``reads`` or a templated one's ``layout`` and ``structural``), bound to the
    window's partition and program: a5's composer for its members. a5's compose cache keys it by its ``inputs``
    (`a5.compose_key`; a composer without them is never cached): the stager's label, the statement and, for a table read, the
    executable whose template its unit is (`read_tmpl.digest`), as `compose_circuit.template_inputs` keys a ``--template``
    compose, so a circuit kept under another build of the verifier's read is not taken."""
    from verity_flock import read_tmpl as RT

    form = {"layout": st["layout"], "structural": bool(st["structural"])} if st.get("layout") else {"reads": st["reads"]}

    def compose(out: Path, partition: str, program: str) -> dict[str, Any]:
        from verity_flock import circuit as FC
        from verity_flock.class_statement import UnitStager

        s = UnitStager(fn, "unit/window", [(p, sc, int(b), list(ps)) for p, sc, b, ps in st["in_rows"]],
                       [(sc, tuple(ix)) for sc, ix in st["out_rows"]], circuit=Path(out),
                       bind={"partition": partition, "program": program}, **form)
        if form.get("reads") == "slots":
            FC.IL.write_tables(Path(out).parent / "tables")
        return {"subcircuit": s.meta.get("subcircuit"), "pin": s.pin, "class_sha512": s.class_sha512, "k_log": s.meta["k_log"]}
    compose.inputs = ["unit/window", st, *([RT.digest()] if RT.is_read(fn) else [])]
    return compose


#: the window value the network certifier's period ties to its scan's state tree (``stateTied``): the scan's state rows
TIED = SCAN_STATE


def period(window: Path, audit: Path, pins: Mapping[str, str]) -> dict[str, Any]:
    """The window's audit as one compute audit of the network certifier's period (module docstring's ``period``):
    ``{answer, free, program, registration, steps}``. ``pins`` are the verifier's (`pins_of`). ValueError where the
    circuit isn't the one the registration binds, the window commits no `TIED` value its circuit reads, the audit wrote no
    Program or another than the registration names, or its answer commits to another registration."""
    import served_integrity as SI

    rec = json.loads((window / "registration.json").read_text())
    body, own = (window / "circuit.json").read_bytes(), rec.get("window") or {}
    if hashlib.sha512(body).hexdigest() != own.get("circuit"):
        raise ValueError(f"{window / 'circuit.json'} is not the circuit record the registration binds")
    doc = json.loads(body)
    ties, untied = window_pins(own, doc, catalog_resolve)
    if TIED not in ties | untied:
        raise ValueError(f"the window commits no {TIED} its circuit reads: the period has no state to tie to its scan's")
    count = declared_count(doc, 0, catalog_resolve, pins={**pins, **pin_roots({**ties, **untied})},
                           roots=window_roots(own), untied=set(untied) - {TIED})
    try:
        program = json.loads((audit / "window" / "program.json").read_text())
    except OSError:
        raise ValueError(f"the audit under {audit} wrote no Program: it stopped before staging its members") from None
    if program_digest(program) != rec.get("program"):
        raise ValueError(f"the audit's Program ({program_digest(program)[:16]}…) isn't the one the registration names")
    answer = SI.answer(audit, rec)
    if answer["commitment"] != R.digest(rec):
        raise ValueError("the audit's answer commits to another registration than the window's")
    return {"answer": answer, "free": count["free"], "program": program, "registration": rec, "steps": 0}


def verifier_window(window_dir: Path, law: Mapping[str, Any] | str, template_sets: Path | None) -> tuple[Any, dict, dict]:
    """The verifier's window and circuit record from the records (`interface.of_records`), and a5's template sets: each
    statement's template composed from its Definition (`composer`), every other template lowered from the verifier's own
    input set, whose Definition it resolves to; any other template from Verity's catalog (`catalog_resolve`). ``law`` is
    the window's sampling design, or as text the rate a stream states as its rule (`service.Law`, which refuses a fixed
    count by name): the window's sampling design and work table are then the ones that rate names over the window's own proof
    blocks (`a5.rated`), as the stream derives them. A window with row tables is refused: a5 checks their segment roots
    against the ones recorded before each coin (``step_roots``), which the records don't carry yet."""
    import a5 as A5

    own = A5.template_sets(template_sets) if template_sets else {}
    resolve = lambda t: own[t][1] if t in own else catalog_resolve(t)  # noqa: E731
    if isinstance(law, str):
        S.Law(law)
        mine, doc = IF.of_records(window_dir, {}, resolve)
        mine = A5.rated(mine, law)
    else:
        mine, doc = IF.of_records(window_dir, law, resolve)
    if mine.row_tables:
        raise ValueError(f"the window has {len(mine.row_tables)} row tables, and its records don't carry the segment roots "
                         "recorded before each segment's coin, so the audit can't check the tables against them: refused "
                         "until they do")
    sets = {**own, **{t: (composer(resolve(t), st), resolve(t)) for t, st in doc["statements"].items()}}
    return mine, doc, sets


def output_class(circuit: Mapping[str, Any], given: Iterable[Iterable[int]] = (), own: Iterable[str] = ()) -> list[list[int]]:
    """The units whose outputs are served out (a5's output class, which it refuses), as ascending ``[lo, hi)`` ranges: the ``given`` ranges (the
    verifier's own, from the circuit's shape) and every unit an uncut registered read of an output port (``o<j>``)
    covers in the circuit record (``registered``'s ``reads``: units ``[at, at + prod(radix))`` of its reader template,
    the population's indices from the templates' order), unless it reads one of the window's ``own`` values
    (`window_pins`: a tie inside the window, a wired loop's activations; `tie_of`: the earlier registration's values the
    window's registration reuses, each table's readers a stratum of its own), whose units the sample samples like any other. A
    value pinned from outside the window (a served ``out`` that the next step also reads) keeps its units in the class."""
    base, at = {}, 0
    for t, n in circuit["templates"]:
        base[t], at = at, at + n
    units, own = {u for lo, hi in given for u in range(lo, hi)}, set(own)
    for r in (circuit.get("registered") or {}).get("reads", ()):
        j = r["port"][1:]
        if r["port"][:1] == "o" and j.isascii() and j.isdigit() and r.get("cut", [0, 1]) == [0, 1] and r.get("param") not in own:
            units.update(range(base[r["reader"]] + r["at"], base[r["reader"]] + r["at"] + math.prod(r["radix"])))
    out: list[list[int]] = []
    for u in sorted(units):
        if out and out[-1][1] == u:
            out[-1][1] = u + 1
        else:
            out.append([u, u + 1])
    return out


def _code(fn: Any) -> str | None:
    """The module a Definition's code is in: where the summary says each template's Definition came from."""
    d = getattr(fn, "defn", fn)
    return getattr(getattr(d, "body", None) or getattr(d, "evaluate", None), "__module__", None)


def proof_edits(lean: str, wdir: Path, members: tuple, out: Path, most: int) -> dict:
    """At most ``most`` of `zk_mutants.py`'s one-bit edits of the first sampled session's proof (a prover that sent the edited
    proof, the round messages that carried it rebuilt and the coins kept), each verified by Lean `verify --zk` on the honest
    member's circuit and public file: what the verifier refuses, not the prover. Then that public file with one
    bit flipped (``public-bit-*``) under the honest proof."""
    import a0
    # the directory of `zk_mutants.py` and the modules it imports beside it
    sys.path.insert(0, str((Path(__file__).resolve().parents[2] / "verity/protocols/flock/agreement/zk_mutants.py").parent))
    import zk_mutants as ZM

    k, honest = next(((k, s.parent) for k in range(len(members))
                      for s in sorted((wdir / f"session-m{k}" / "sessions").glob("*/session.json"))), (None, None))
    if honest is None:
        return {"edits": 0, "refused": 0, "why": "no drawn session"}
    st = {"dir": wdir / f"m{k}", "partition_file": wdir / "partition.json", "program_file": wdir / "program.json"}
    ZM.main(honest, out / "mutants")
    cases = {}
    for c in [c for c in json.loads((out / "mutants" / "zk_mutants.json").read_text()) if c.get("rep") is not None][:most]:
        sd = out / c["name"] / "sessions" / honest.name
        shutil.copytree(honest, sd, copy_function=os.link, dirs_exist_ok=True)
        for f in (out / "mutants" / c["name"]).iterdir():
            (sd / f.name).unlink(missing_ok=True)
            shutil.copyfile(f, sd / f.name)
        v = a0.lean_verify(lean, st, members[k].n, out / c["name"], zk=True)
        cases[c["name"]] = {"refused": a0.lean_refused(v), "why": v.get("why"), "expected": c["expect"]["why"]}
    n = members[k].n
    pub = (st["dir"] / f"pub-{n}.bin").read_bytes()
    for name, off in (("public-bit-mid", len(pub) // 2), ("public-bit-last", len(pub) - 1)):
        d = out / name / "member"
        shutil.copytree(st["dir"], d, copy_function=os.link, dirs_exist_ok=True)
        (d / f"pub-{n}.bin").unlink()
        (d / f"pub-{n}.bin").write_bytes(pub[:off] + bytes([pub[off] ^ 1]) + pub[off + 1:])
        shutil.copytree(honest, out / name / "sessions" / honest.name, copy_function=os.link, dirs_exist_ok=True)
        v = a0.lean_verify(lean, {**st, "dir": d}, n, out / name, zk=True)
        cases[name] = {"refused": v["accepted"] is False and v.get("digest") is not None, "why": v.get("why"),
                       "expected": f"the public file's byte {off} of {len(pub)}, bit 0 flipped, under the honest proof"}
    return {"member": k, "session": honest.name, "edits": len(cases), "refused": sum(v["refused"] for v in cases.values()),
            "cases": cases}


#: how the ZK server's reason for a message it holds back opens (its Lean's own text), which the prover's record of its
#: session gives as why it stopped (``prover_stopped``; live PROTOCOL.md, the ZK server's send check)
ZK_SERVER_REFUSED = "FIREWALL-REFUSED"
#: how a Lean verdict's reason opens where a file of the session's (its record, a rep's proof) was not there to read
#: (`FlockVerify`'s ``verify``): like a refusal at setup, which `a0.lean_refused` leaves out, the verifier read no proof
NO_SESSION_FILES = "session files:"


def session_proofs(sdir: Path) -> dict[str, Any]:
    """What member session ``sdir`` (a5's ``session-m<k>``) records of its proofs leaving the prover: ``proofs``, how many
    proof streams the verifier's record of the session lists (``session.json``'s ``proof_sha512``; None where the session
    has no record, or its record no such list), and ``prover_stopped``, why the prover's record says it stopped short (its
    ``LIVE`` line, in `a0.session`'s ``prove.log``; None where it didn't, or printed no such line)."""
    recs = sorted((sdir / "sessions").glob("*/session.json"))
    listed = json.loads(recs[-1].read_text()).get("proof_sha512") if recs else None
    log = sdir / "prove.log"
    live = next((json.loads(x[5:]) for x in (log.read_text().splitlines() if log.is_file() else ()) if x.startswith("LIVE\t")), {})
    return {"proofs": None if listed is None else len(listed), "prover_stopped": live.get("prover_stopped")}


def forged_outcome(s: dict, wdir: Path, member: int | None = None, more: Mapping[str, bool] | None = None) -> dict[str, Any]:
    """A forged window's verdict, from a5's summary ``s`` and its audit directory ``wdir`` (``audit.json``, each sampled
    member's ``session-m<k>``). Every check before the sample passed, ``more`` (the caller's own checks) holds, the audit
    record rejects, and the false member was stopped where its sessions stop one:

    * behind the ZK server (the summary names its binary), no proof of it may leave the prover: ``no_proof_released``, the
      session of the member holding the forged unit (``member``; any sampled member's where None) has no proof in the
      verifier's record of it, and the prover's record gives the ZK server's refusal as why it stopped (`session_proofs`,
      `ZK_SERVER_REFUSED`). The Lean verifier refuses that session too, for want of its files: a verdict on no proof.
    * behind none (a plain audit), the proof leaves, and Lean's own verdict must refuse it: ``lean_refused``, the Lean
      verifier refused a proof it read (`a0.lean_refused`, and not for want of the session's files, `NO_SESSION_FILES`).

    ``checks`` holds both, and the window passes on the one its sessions call for (``passes_on``). A check with nothing to
    read has no value, and ``no_value`` says why: ``no_proof_released`` behind no ZK server, and ``lean_refused`` where
    Lean refused only sessions it read no proof of (False where it refused none). ``sessions`` is what each session read
    records (`session_proofs`), and ``why`` the audit's reasons."""
    import a0

    audit, behind, no_value = s.get("audit") or {}, bool(s.get("firewall")), {}
    record = json.loads((wdir / "audit.json").read_text()) if (wdir / "audit.json").is_file() else {}
    lean = {k: v for k, v in (record.get("verdicts") or {}).items() if k.startswith("lean-verify/") and v.get("accepted") is not True}
    read = [k for k, v in lean.items() if a0.lean_refused(v) and not v["why"].startswith(NO_SESSION_FILES)]
    refused: bool | None = bool(read)
    if lean and not read:
        refused = None
        no_value["lean_refused"] = "the Lean verifier read no proof: " + "; ".join(
            f"{k}: {(v.get('why') or 'no reason given').split(chr(10), 1)[0]}" for k, v in sorted(lean.items()))
    dirs = sorted(wdir.glob("session-m*")) if member is None else [wdir / f"session-m{member}"]
    sessions = {d.name.removeprefix("session-"): session_proofs(d) for d in dirs}
    released: bool | None = None
    if behind:
        released = any(x["proofs"] == 0 and str(x["prover_stopped"] or "").startswith(ZK_SERVER_REFUSED) for x in sessions.values())
    else:
        no_value["no_proof_released"] = "the sessions ran behind no ZK server"
    on, other = ("no_proof_released", "lean_refused") if behind else ("lean_refused", "no_proof_released")
    checks = {"checks_before_draw": bool(s.get("checks_before_draw")) and all(s["checks_before_draw"].values()), **(more or {}),
              "rejected": audit.get("accepted") is False, "no_proof_released": released, "lean_refused": refused}
    return {"ok": all(v is True for c, v in checks.items() if c != other), "checks": checks, "passes_on": on,
            "no_value": no_value, "sessions": sessions, "why": audit.get("why")}


def audit(a: argparse.Namespace) -> int:
    """The audit (module docstring) into ``a.out``: ``summary.json`` and a ``RECORDS`` line, its ``definitions`` the module
    each member template's Definition is in and ``loaded`` the ``verity*`` packages the process loaded. -> 0 when it passed."""
    import a5 as A5
    import a5_coins as AC

    t0, wdir, out = time.time(), Path(a.window), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        mine, doc, sets = verifier_window(wdir, a.law, Path(a.template_sets) if a.template_sets else None)
        pins = pinned(json.loads(Path(a.registered).read_text())) if a.registered else {}
        def resolve(t: str) -> Any:
            return sets[t][1] if t in sets else catalog_resolve(t)
        registration = json.loads((wdir / "registration.json").read_text())
        own = registration.get("window") or {}
        ties, untied = window_pins(own, doc, resolve)
        if twice := sorted((set(ties) | set(untied)) & set(pins)):
            raise ValueError(f"values {twice} are both pinned and the window's own")
        pins |= ties | untied
        w_tie = json.loads(Path(a.tie_file).read_text()) if a.tie_file else None
        tied, tie_pins = tie_of(doc, w_tie, a.tie_prefix)
        if twice := sorted(set(tie_pins) & set(pins)):
            raise ValueError(f"values {twice} are both pinned and the earlier registration's")
        pins |= tie_pins
        roots = window_roots(own)
        for e in registration.get("registered") or ():
            if e.get("value") in tie_pins and roots.setdefault(e["value"], e["root"]) != e["root"]:
                raise ValueError(f"the window's registration records two roots for {e['value']!r}")
        histories = HW.pins(doc, roots)
        if twice := sorted(set(histories) & set(pins)):
            raise ValueError(f"values {twice} are both pinned and the window's history tables")
        pins |= histories
        reads = IF.registered_reads(doc, mine, pins, registered=REG)
        if a.day_ledger:
            from verity.protocols.spot.device import challenger as Ch, remote as SPR
            key = Ch.key_of_record(json.loads(Path(a.challenger_key).read_text()))
            day = device_day(SPR.ledger_lines(a.day_ledger), mine.run, key, AC.lms_verify, w_tie)
        else:
            day = declared_day(a.day, a.day_log, a.day_params, w_tie)
    except (ValueError, KeyError, TypeError) as e:
        row = {"ok": False, "refused": {"records": f"{type(e).__name__}: {e}"}, "seconds": round(time.time() - t0, 1)}
        (out / "summary.json").write_text(json.dumps(row, indent=1, sort_keys=True))
        print("RECORDS " + json.dumps(row, sort_keys=True))
        return 1
    rec = json.loads((wdir / "registration.json").read_text())
    w = rec.get("window") or {}
    body = (wdir / "circuit.json").read_bytes()
    held: dict[str, Any] = {"run": a.run, "circuit": a.circuit or hashlib.sha512(body).hexdigest(),
                            "compiled": True, "compiler": None, "advice": None}
    compiled = None
    if a.opening:
        compilers, opening = json.loads(Path(a.compilers).read_text()), json.loads(Path(a.opening).read_text())
        repo = Path(a.repo) if a.repo else None
        compiled, bound = compile_check(doc, w, opening, compilers, out / "compile", repo)
        output = out / "compile" / "compiled.json"
        held.update(compiled=compiled["ok"], compiler=(bound or {}).get("compiler") or w.get("compiler"),
                    advice=(bound or {}).get("advice") or w.get("advice"), compilers=compilers, opening=opening, repo=repo,
                    output=json.loads(output.read_text()) if compiled["ok"] else None)
    window_s = round(time.time() - t0, 2)
    more = {} if a.skip_negatives else negatives(w, held, doc, out / "negatives")
    a.ports = a.ports if a.ports is not None else max(1, len(mine.members))
    a.port = a.port if a.port is not None else free_ports(a.ports)
    a.coins, a.verifier_coins, a.window_root, a.rule = None, not a.challenger, mine.run, a.law
    a.outputs = output_class(doc, json.loads(Path(a.outputs).read_text()) if a.outputs else (),
                             own=set(ties) | set(untied) | set(tie_pins))
    a.tie, a.declared = tied, day
    if tied:
        named = {**tie_row(tied), "previous": rec.get("previous"), "chained": rec.get("previous") == R.digest(tied["log"][0])}
        (out / "tie.json").write_text(json.dumps(named, indent=1, sort_keys=True))
        print("TIE " + json.dumps(named, sort_keys=True), flush=True)
    if a.forged:
        os.environ["FC_ZK_SELF_CHECK"] = "skip"
    manager, a.coins, finisher = AC.start(a.challenger, mine.run, [0], out / "challenger", a.main, ssh_key=a.gate_key,
                                          peer=a.peer, key_record=a.challenger_key, deployed=a.deployed_key)
    try:
        s, ok = A5.audit(a, sets, out / "window", mine, wdir, lambda x: source(x, held), a.port,
                         {"verifier_window_s": window_s, "circuit": held["circuit"], **({"compile": compiled} if compiled else {})},
                         more, rank=0, registered=reads or None, ports=a.ports,
                         history=(lambda publics: HW.params(doc, mine, publics, pins)) if histories else None)
    finally:
        a.coins.skip(0)
        finisher.join()
        published = a.coins.evidence()
        manager.shutdown()
    row = {"seconds": round(time.time() - t0, 1), "verifier_window_s": window_s, "circuit": held["circuit"],
           "compiler": held["compiler"], "advice": held["advice"], "compile": compiled, "law": a.law,
           **{x: s.get(x) for x in ("run_root", "population", "members", "registration", "source", "refused", "checks_before_draw",
                                    "audit", "draw", "draw_counts", "negatives", "negatives_skipped", "coins",
                                    "timing", "outputs")}}
    bits = int((held["advice"] or {}).get("bits", 0))
    counted = pin_roots(pins)
    row["leak"] = (declared_count(doc, bits, resolve, counted, roots, untied)
                   if {*IF.DECLARED, IF.WINDOW_INGRESS} & set(doc) else leak_count(mine.members, doc.get("external") or (), bits))
    if reads:
        row["registered"] = {"pins": counted, "roots": roots, "members": len(reads)}
    if tied:
        row["tie"] = tie_row(tied)
    if day:
        row["day"] = {"day": day["day"], "declared": [d["program"] for d in day["declarations"] if d["day"] == day["day"]],
                      "source": DAY_SOURCES["device" if a.day_ledger else "development"]}
    acts = committed_activations(w)
    if acts is not None:
        row["activations"] = acts
    row["definitions"] = {t: _code(fn) for t, (_c, fn) in sets.items() if t in {m.template for m in mine.members}}
    row["loaded"] = sorted({m.split(".")[0] for m in sys.modules if m.startswith("verity")})
    row["coins_session"] = {"trusted": published.get("trusted"), "error": published.get("error")}
    if s.get("draw") is not None:
        rec, receipt = (json.loads((out / "window" / f).read_text()) for f in ("registration.json", "receipt.json"))
        bound = {"compiler": held["compiler"], "advice": held["advice"]} if held["compiler"] is not None else None
        row["fixed_before_draw"] = fixed_before_draw(receipt, rec, bound)
        ordered = row["fixed_before_draw"]["ok"]
        if bound is not None:
            row["after_draw"] = after_draw(receipt, rec, held)
            row["negatives"] = {**(row["negatives"] or {}), "advice-root-after-the-draw": row["after_draw"]["refused"]}
            ordered = ordered and row["after_draw"]["refused"]
    else:
        ordered = False
    if a.forged:
        row["forged"] = forged_outcome(s, out / "window")
        row["ok"] = row["forged"]["ok"] and ordered
    else:
        row["ok"] = ok and ordered and (compiled is None or compiled["ok"])
        if a.flip_proofs and (s.get("audit") or {}).get("accepted"):
            pe = row["proof_edits"] = proof_edits(a.lean, out / "window", mine.members, out / "proof-edits", a.flip_proofs)
            row["ok"] = row["ok"] and pe["edits"] > 0 and pe["refused"] == pe["edits"]
            print("PROOF-EDITS " + json.dumps({x: pe.get(x) for x in ("edits", "refused", "member", "session")}))
    row["ok"] = row["ok"] and not row["coins_session"]["error"]
    row["ok"] = row["ok"] and (acts is None or acts["ok"])
    (out / "summary.json").write_text(json.dumps(row, indent=1, sort_keys=True, default=str))
    print("RECORDS " + json.dumps({x: row.get(x) for x in ("ok", "circuit", "population", "draw", "draw_counts", "outputs", "seconds")}
                                  | {"compile": (compiled or {}).get("ok"), "fixed_before_draw": (row.get("fixed_before_draw") or {}).get("ok"),
                                     "accepted": (s.get("audit") or {}).get("accepted"),
                                     "negatives_refused": all((row.get("negatives") or {"none": False}).values()),
                                     "definitions": row["definitions"], "loaded": row["loaded"],
                                     "activations": None if acts is None else acts["ok"],
                                     "leak": {k: row["leak"][k] for k in ("other_inputs", "free", "registered", "advice_bits",
                                                                          "bound", "unpinned", "untied", "window")
                                              if k in row["leak"]},
                                     **({"forged": row["forged"]} if a.forged else {})}, sort_keys=True, default=str))
    return 0 if row["ok"] else 1


def warm(a: argparse.Namespace) -> int:
    """Every template of the verifier's window that its composers make, composed into ``a.compose_cache`` bound to the window's
    partition and program (`a5.warm`), ``a.jobs`` at a time, before the timed audit: an audit with that cache then composes
    none. One JSON line a template (its seconds, pin and whether the cache held it), then a ``WARM`` line."""
    import a5 as A5

    t0 = time.time()
    mine, _doc, sets = verifier_window(Path(a.window), a.law, Path(a.template_sets) if a.template_sets else None)
    window_s = round(time.time() - t0, 1)
    got = A5.warm(a, sets, mine, a.jobs)
    for g in got:
        print(json.dumps(g, sort_keys=True))
    print("WARM " + json.dumps({"templates": len(got), "kept": sum(bool(g["kept"]) for g in got), "verifier_window_s": window_s,
                                "seconds": round(time.time() - t0, 1)}, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    import a5 as A5

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("audit", help="a window's audit from its records alone")
    p.add_argument("--window", required=True, help="the window's directory: circuit.json, registration.json, the value files")
    p.add_argument("--law", required=True, help="the rate the audit's stream states as its rule (rate:NUM/DEN, "
                   "stratified-rate:NUM/DEN or work-rate:NUM/DEN), whose sampling design over the window's proof blocks the "
                   "registration is held to; a fixed count (subset:K, stratified:K, work:K) is refused")
    p.add_argument("--opening", help="the advice's opening (interface.advice_opening): the named gap, until the compile unit")
    p.add_argument("--compilers", help="with --opening: the registered compilers the verifier holds, {name: {digest, argv, "
                   "code, pythonpath}}")
    p.add_argument("--repo", help="with --opening: the repository a registered compiler's commit is checked out from (default "
                   "this tree's)")
    p.add_argument("--circuit", help="instead: the SHA-512 of the circuit record the verifier registered, a circuit with no compiler")
    p.add_argument("--run", required=True, help="the run root the verifier holds the window to, the one the run published")
    p.add_argument("--template-sets", help="the verifier's own input sets, for templates whose units are not bit rows")
    p.add_argument("--outputs", help="JSON: the verifier's own [lo, hi) ranges of the units whose outputs are served out "
                   "(`output_class`), which a5 refuses before the sample (`output_refusal`)")
    p.add_argument("--m0-pythonpath", required=True)
    p.add_argument("--bin", required=True)
    p.add_argument("--lean", required=True)
    p.add_argument("--firewall", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--delta", type=float, default=0.5)
    p.add_argument("--port", type=int, default=None, help="the sessions' base port (default a base whose --ports ports are "
                   "free now)")
    p.add_argument("--ports", type=int, default=None, help="how many ports from --port are the audit's: a session each for "
                   "every member with a proved unit, a sample needing more refused before any listens (default one a member)")
    p.add_argument("--session-timeout", type=float, default=14400)
    p.add_argument("--session-jobs", type=A5.one_session, default=1, help="1 only: a5 runs the sessions one at a time (a5.ONE_AT_A_TIME)")
    p.add_argument("--forged", action="store_true", help="the window holds a forged unit: it must be refused")
    p.add_argument("--flip-proofs", type=int, default=0, metavar="N",
                   help="after an accepted audit, at most N one-bit edits of its first drawn session's proof, each of which "
                   "Lean verify must refuse (`proof_edits`)")
    p.add_argument("--challenger", help="draw on the auditor's challenger's coins (a5_coins): local, loopback:HOST:PORT or "
                   "gate:USER@HOST (the auditor host through its SSH gate, with --peer and --challenger-key)")
    p.add_argument("--deployed-key", action="store_true",
                   help="with --challenger local: the session's key at the deployed LMS parameters (a5_coins.start's deployed), "
                   "its whole tree made over this machine's cores")
    p.add_argument("--peer", help="with --challenger gate:USER@HOST: this machine as the gate's /etc/vy/peers names it, whose "
                   "127.0.0.1 the session's challenger listens on")
    p.add_argument("--challenger-key", help="with --challenger gate:USER@HOST: the challenger's key record of its deployment "
                   "(challenger-key/v1), never the session's key.json")
    p.add_argument("--gate-key", help="with --challenger gate:USER@HOST: the lane's own gate key (default ssh's own)")
    p.add_argument("--main", help="with --challenger: the verifier's copy of main's first-parent history")
    p.add_argument("--registered", help="the registered values' entries the verifier pinned before the run, {program, "
                   "values} (`pinned`), for a circuit that reads them")
    p.add_argument("--tie", dest="tie_file", help="the earlier registration W (PoUW H1's of the window, or Pearl+'s window "
                   "record) that registered first the values the circuit's program takes, which the window's registration "
                   "registers as W's, chained right after it (`tie_of`); a circuit with values needs it")
    p.add_argument("--tie-prefix", help="with --tie: W's descriptor's PREFIX, which its tables' names take "
                   "(`credit.window_table`; h1/ for PoUW H1's window). Without it, W names each value as the circuit does "
                   "(Pearl+'s)")
    p.add_argument("--day", type=int, default=None, help="the window's day, by its index among the monitoring period's days: "
                   "R2 holds the registration's program to the day's one declaration in --day-log (`declared_day`)")
    p.add_argument("--day-log", help="with --day: the deployment's registration log (verity/registration/log/v1)")
    p.add_argument("--day-params", help="with --day: the log's parameters, {auditor, composite} (default the log's composite "
                   "under an auditor of no sheets or keys)")
    p.add_argument("--day-ledger", help="the challenger device's own ledger (windows.jsonl, whole: never a copy PoUS cut), "
                   "verified under --challenger-key: R2's day is the window open's signed UTC day and its declarations the "
                   "device's (`device_day`), in place of --day's development fallback")
    p.add_argument("--skip-negatives", action="store_true", help="no negatives in this audit (module docstring): a run of "
                   "several audits runs them in its first")
    p.add_argument("--compose-cache", help="a5's compose cache, which `warm` filled under this tree and FLOCK_* environment: "
                   "staging links each kept circuit and composes none (default none: every template composed here)")
    p = sub.add_parser("warm", help="every template of a window composed into a5's compose cache, before its timed audit")
    p.add_argument("--window", required=True, help="the window's directory, as its audit takes it")
    p.add_argument("--law", required=True, help="the window's rule, as its audit takes it")
    p.add_argument("--template-sets", help="the verifier's own input sets, as its audit takes them")
    p.add_argument("--compose-cache", required=True, help="the cache the audit's --compose-cache names")
    p.add_argument("--jobs", type=int, default=1, help="templates composed at once, each in a forked process: what the "
                   "machine's memory takes (a5.warm)")
    p = sub.add_parser("count", help="a circuit record's leak count (declared_count): a stated record's from its description "
                       "alone, a listed one's from its members and the registered reads and window ingress it declares; the "
                       "run figure, and beside it, where the record declares window ingress, the window figure with its premise")
    p.add_argument("--circuit", required=True, help="the circuit record (circuit.json), or its description")
    p.add_argument("--bits", type=int, required=True, help="the advice's registered length in bits, the charge")
    p.add_argument("--catalog", action="append", default=[], help="a module whose Definitions the record's templates name, "
                   "imported before they resolve (repeatable; `interface.resolve` loads the ops and the embedding read)")
    p.add_argument("--pin", action="append", default=[], type=pin_arg, metavar="NAME=ROOT", help="a registered parameter, by "
                   "the record's name for it, and the root the verifier fixed for it before the run (repeatable; none: every "
                   "registered read is free)")
    p.add_argument("--pins", action="append", default=[], metavar="FILE", help="pins as a JSON {param: root}, or a "
                   "registration of values {program, values} (vLLM's committer's weights registration, "
                   "window_weights.weights_registration), with --pin's (repeatable)")
    p.add_argument("--registration", help="the window's registration, whose roots for its parameters (`registration_roots`) are held "
                   "to the pins (none: no root is the window's, and every registered read is free)")
    p = sub.add_parser("period", help="a window's audit as one compute audit of the network certifier's period (`period`)")
    p.add_argument("--window", required=True, type=Path, help="the window's directory: circuit.json, registration.json")
    p.add_argument("--audit", required=True, type=Path, help="the audit's --out: summary.json, window/program.json, "
                   "window/audit.json")
    p.add_argument("--out", required=True, type=Path, help="the period's audit entry, JSON")
    p.add_argument("--pin", action="append", default=[], type=pin_arg, metavar="NAME=ROOT", help="as count's")
    p.add_argument("--pins", action="append", default=[], metavar="FILE", help="as count's")
    a = ap.parse_args(argv)
    if a.cmd == "period":
        try:
            got = period(a.window, a.audit, pins_of(a.pin, a.pins))
        except (ValueError, RuntimeError) as e:
            print(f"records_audit.py period: {e}", file=sys.stderr)
            return 1
        a.out.write_text(json.dumps(got, indent=1, sort_keys=True) + "\n")
        print("PERIOD " + json.dumps({"verdict": got["answer"]["verdict"], "reason": got["answer"]["reason"],
                                      "free": got["free"], "commitment": got["answer"]["commitment"],
                                      "program": got["registration"]["program"]}, sort_keys=True))
        return 0
    if a.cmd == "count":
        for m in a.catalog:
            importlib.import_module(m)
        try:
            pins = pins_of(a.pin, a.pins)
        except ValueError as e:
            ap.error(f"--pin/--pins: {e}")
        roots = registration_roots(json.loads(Path(a.registration).read_text())) if a.registration else {}
        print(json.dumps(declared_count(json.loads(Path(a.circuit).read_text()), a.bits, pins=pins, roots=roots),
                         sort_keys=True))
        return 0
    if a.cmd == "warm":
        return warm(a)
    if bool(a.opening) == bool(a.circuit) or bool(a.opening) != bool(a.compilers):
        ap.error("a compiled circuit: --opening FILE --compilers FILE; a circuit with no compiler: --circuit SHA512")
    if not a.challenger:
        ap.error("audit draws on the auditor's challenger's coins, never the verifier's own: give --challenger local --main FILE")
    if a.challenger:
        if a.challenger != "local" and not a.challenger.startswith(("gate:", "loopback:")):
            ap.error(f"--challenger is local, loopback:HOST:PORT or gate:USER@HOST, not {a.challenger!r}")
        if a.deployed_key and a.challenger != "local":
            ap.error("--deployed-key is a local session's key: it goes with --challenger local")
        if a.challenger.startswith("gate:") and not (a.peer and a.challenger_key):
            ap.error("--challenger gate:USER@HOST needs --peer and --challenger-key")
        if not a.main:
            ap.error("--challenger needs --main")
    if a.day_ledger and (a.day is not None or a.day_log or a.day_params or not a.challenger_key):
        ap.error("--day-ledger reads the day from the challenger's signed ledger under --challenger-key, without --day, "
                 "--day-log or --day-params")
    return audit(a)


if __name__ == "__main__":
    raise SystemExit(main())
