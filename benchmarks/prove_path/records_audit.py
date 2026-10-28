"""A window's audit from its records alone: what an integration hands Verity (`verity.ml.windows.interface`), and nothing else.

The records, every one in the window's directory but the advice's opening:

* **the circuit** in proof blocks with its public layout (``circuit.json``), which the registration binds by SHA-512;
* **the compiler and the advice**, when the circuit depends on run-time choices: the registration's window names the compiler
  (``compiler``: its name and digest, which carries the public cap on the advice's length, and the commit its code is at)
  and the advice's hiding commitment (``advice``: its root and its length in bits, which is charged); the advice is a bit
  string the audit never interprets;
* **the values**, committed before any coin: the registration (``registration.json``) and the files `rows.write` writes;
* **on a draw, the drawn units' openings** (``m<k>-private.bin``), which reach only the prover behind the firewall.

The verifier's window is `interface.of_records`'s, each template resolved in Verity's catalog (`interface.resolve`), or, for a
template the verifier lowers from its own input set (``--template-sets``), that set's Definition; a window with row tables
is refused until its records carry their segment roots. Before the draw the audit holds the registration to that circuit,
to the run root the run published (``--run``) and, for a compiled circuit, to the compiler and advice it was compiled from
(`source`), and runs the compile check (`compile_check`, the named gap until the compile unit exists): the advice's opening
opens the registered root under the registered compiler, and that compiler, run as the command the verifier's own table
names for it (``--compilers``) from a checkout of the registered commit whose code hashes to the registered digest, makes
this circuit from the advice and charges the registered length. A circuit with no compiler is the one the verifier
registered (``--circuit``). Then `a5.audit`: the registration checked by Lean and Python, the draw on live coins read after
the receipt, C-Flock ``--zk`` behind the Lean firewall on the drawn members, Lean's verdicts and the audit record. Then the
advice was fixed before the draw (`fixed_before_draw`), and advice re-committed after the coins is refused (`after_draw`).
Units whose outputs are served out (``--outputs`` and the circuit's registered reads of output ports, `output_class`) are
proved whole beside the draw, under a work law: every draw's closure (a5's `output_closure`).
The summary reports the leak bound's count beside the advice's charge, ``vacuous`` while the circuit's other inputs exceed
it (`leak_count`; a record that declares its inputs, `declared_count`). An integration's labels are never read.

A circuit that reads registered values at public positions (`interface.Read`) is audited against the entries the verifier
pinned before the run (``--registered FILE``: ``{"program", "values": {param: entry}}`` or a list of them, each entry core's
``registered.Registered.entry()``, its domain held to the one its program, value and rows make, `pinned`): each drawn
member's ``--registered`` file comes from the record and those entries alone (`interface.registered_reads`), and Lean's
registered reads open every row of its read ports against the pinned root at the row the record reads for its unit
(a5's ``registered``, with its negatives: no registered reads, another root, every unit reading another row). The reads
of a pinned parameter leave the free inputs where the window's root for it is the pinned root: `declared_count` counts
them registered, its pins the ``--registered`` file's roots and the window's roots those its registration records
(`window_roots`: its ``registered_roots``, `interface.ExecutionBlockBuilder.write`, and its committed values' entries).

A registration whose window commits per-window values (``activations``: ``{program, values: {value: entry}}``, a
loop's activation rows) has them fixed before the draw with the rest of the registration;
the audit holds each entry's domain to the one core's ``registered.domain`` makes of the program, the value, its rows and
the window's run root (`committed_activations`), and reports their roots. Those the circuit doesn't read take nothing from
the free inputs. Those it reads as ties (a wired loop's activations: an output port writes every row an input port
takes, `tied_values`) are pinned from the registration beside ``--registered``'s (`window_pins`), so Lean's registered
reads tie each reader's row to its writer's. One it reads only through input ports is opened against its committed root
too, but the prover chose its rows: its reads count free, and the count names it (``leak.untied``). PoUW's H1 tiles
have theirs pinned through ``--registered``, from the window's entries.

The negatives (`negatives` before the draw, a5's after it) cost Lean verifies of their own; ``--skip-negatives`` leaves
them out of an audit (its summary says so), for a run of several audits of one circuit, which runs them in its first.

    records_audit.py audit --window DIR --law L (--opening FILE --compilers FILE [--repo DIR] | --circuit SHA512) --run HEX \\
        [--template-sets DIR] [--outputs FILE] --m0-pythonpath PATH --bin flock-circuit --lean flock-verify --firewall flock-firewall \\
        --out DIR [--delta D] [--port P] [--ports N] [--forged] [--challenger local|loopback:HOST:PORT|gate:USER@HOST --main FILE] \\
        [--peer NAME --challenger-key RECORD [--gate-key KEY]] [--registered FILE] [--skip-negatives]
    records_audit.py count --circuit FILE --bits N [--catalog MODULE ...] [--pin NAME=ROOT ...] [--pins FILE ...] [--registration FILE]

``count``: a stated record that declares its inputs (cuts, registered reads, ingress: `interface.Circuit.declared`) counted
from its description alone (`declared_count`), before the committer writes its members' rows. A registered read counts as
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

``--challenger``: the draw on the auditor's challenger's coins (`a5_coins`), not the verifier's own, as `prove_path.py audit`
takes them: ``local``, a development session on this machine, ``loopback:HOST:PORT``, that session on the challenger
party's host (`challenger_party.py`, `verity run CONFIG --party challenger`), or ``gate:USER@HOST``, the auditor host through its SSH gate with
the lane's gate key ``--gate-key``, its challenger on ``--peer``'s 127.0.0.1 (so the audit runs on that peer), under the
deployment's key record ``--challenger-key``. The gate opens a root once, ever.

``--forged``: the window holds a forged unit, and the
audit passes when every check before the draw passed and Lean refused the outcome (the prover runs with its self-check
skipped, ``FC_ZK_SELF_CHECK=skip``, so it sends proofs of the false witness).
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

from verity.core.protocols.sampled_proofs.one_stage import draw as D, registered as REG, registration as R  # noqa: E402
from verity.ml.windows import interface as IF  # noqa: E402

#: where an audit's ports start when ``--port`` isn't given (`free_ports`): below Linux's ephemeral range (32768-60999), from
#: which outbound connections take theirs
PORTS = range(10000, 32000)
#: a compile command's exit code for advice it refuses (not advice it compiles: malformed, over its cap, or a choice it can't
#: compile), the reason on stderr; any other nonzero exit is an error. Either fails the compile check (``compiles``).
REFUSED = 2


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


def leak_count(members: Iterable[Any], external: Iterable[str], bits: int) -> dict[str, Any]:
    """The leak bound's count against the advice's charge (``bits``, the registered length; 0 for a circuit with no compiler).
    ``other_inputs`` is #(otherInputs \\ registered): ``otherInputs`` (``SampledCharge.lean``), the circuit's input
    gates no step lists as ingress, less the inputs the registration fixes. The circuit's input gates are each of the
    verifier's ``members``' parameter rows' bits (16 a u16 word: `interface.layout` refuses a port of part words), once per
    unit, or once per table row for a shared-row port (``tables``), and none for a port wired from other units' outputs
    (``wired``, which a5's wiring check holds). The records list no steps and register no input (a row table's registered
    reads are refused, `verifier_window`), so none is ingress or registered yet: the ``external`` ports are reported, not
    subtracted, until a layout lists the ingress at its steps. ``bound`` is ``vacuous`` while ``other_inputs`` exceeds the
    charge."""
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
    """The leak bound's count of a stated record that declares its inputs (`interface.input_count`, from its description
    alone): its input rows less those its wires, registered reads of a parameter whose window root (``roots``, those the
    registration records: `window_roots`) is the verifier's pin (``pins``, ``{param: root}``) and ingress fill
    leave ``free``, and ``other_inputs`` is those plus the advice's ``bits``, the compile unit's one input, a statement of
    its own. So ``other_inputs`` equals the charge exactly when the circuit leaves no input free. An ``untied`` value
    (`window_pins`: a per-window value no output port of the circuit writes, whose rows the prover chose) is no pin here,
    though the audit opens its reads against its committed root: its input rows are free, and the count names it."""
    untied = set(untied)
    held = {p: r for p, r in pins.items() if p not in untied} if isinstance(pins, Mapping) else pins
    n = IF.input_count(doc, resolve, held, roots)
    if untied:
        n = {**n, "untied": sorted(untied)}
    other = n["free"] + bits
    return {**n, "other_inputs": other, "advice_bits": bits, "bound": "vacuous" if other > bits else "within the charge"}


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
    an input port) are pinned: the registration fixes them before the draw, and the Lean verifier's registered reads open
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
    """What the audit refuses before any draw about the records: a registration naming another circuit or another run root,
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


def fixed_before_draw(receipt: Mapping[str, Any], rec: Mapping[str, Any], bound: Mapping[str, Any] | None,
                      coins: str = "verifier") -> dict[str, Any]:
    """That the circuit, its compiler and its advice were fixed before the coins, checked rather than assumed: the digest of the
    registration (which covers the window's ``circuit``, ``compiler`` and ``advice``) is the one the draw's receipt names, the
    receipt the stream wrote when it received the registration and before its select read any coin (on the challenger's
    coins, ``coins="challenger"``, the coins also commit to that digest before any is read: a5_coins); and that registration
    binds the compiler and advice the compile check verified (``bound``, None for a circuit with none). -> {ok, codes, detail}:
    ``after-the-draw`` where the draw followed another registration or another stream's, ``compiler`` or ``advice`` where the
    registration binds another."""
    import a5 as A5

    w, b = rec.get("window") or {}, bound or {}
    drew = {"verifier": A5.VERIFIER, "challenger": A5.CHALLENGER}[coins]
    ok = {"after-the-draw": receipt.get("registration") == R.digest(rec) and receipt.get("verifier") == drew,
          "compiler": w.get("compiler") == b.get("compiler"), "advice": w.get("advice") == b.get("advice")}
    return _verdict(ok, {"drawn_after": receipt.get("registration"), "registration": R.digest(rec),
                         "received": receipt.get("received")})


def after_draw(receipt: Mapping[str, Any], rec: Mapping[str, Any], held: Mapping[str, Any], coins: str = "verifier") -> dict:
    """The negative: once the coins are read, the prover commits the same advice again (fresh salts, so a fresh root) and hands
    in a registration binding that root, with its opening. Every check before the draw passes on it (the new opening opens
    the new root, the same advice compiles to the same circuit, `source` holds), and `fixed_before_draw` refuses it
    (``after-the-draw``): the draw followed the registration received first."""
    c = held["compiler"]
    data = bytes.fromhex(held["opening"]["advice"])
    late = IF.commit_advice(c, data)
    opens = IF.opened_advice(c, IF.advice_opening(data, late))[1] == late.root.hex()
    advice = {**rec["window"]["advice"], "root": late.root.hex()}
    late_rec = {**rec, "window": {**rec["window"], "advice": advice}}
    src = source(late_rec["window"], {**held, "advice": advice})
    v = fixed_before_draw(receipt, late_rec, {"compiler": held["compiler"], "advice": advice}, coins)
    return {"refused": not v["ok"], "codes": v["codes"], "before_draw_checks_pass": opens and src["ok"]}


def composer(fn: Any, st: Mapping[str, Any]) -> Callable[[Path, str, str], dict[str, Any]]:
    """A template's circuit from its Definition and its statement in the circuit record (C-Flock's `UnitStager` on the record's
    ``in_rows``, ``out_rows`` and ``reads``), bound to the window's partition and program: a5's composer for its members."""
    def compose(out: Path, partition: str, program: str) -> dict[str, Any]:
        from verity_flock import circuit as FC
        from verity_flock.class_statement import UnitStager

        s = UnitStager(fn, "unit/window", [(p, sc, int(b), list(ps)) for p, sc, b, ps in st["in_rows"]],
                       [(sc, tuple(ix)) for sc, ix in st["out_rows"]], circuit=Path(out),
                       bind={"partition": partition, "program": program}, reads=st["reads"])
        if st["reads"] == "slots":
            FC.IL.write_tables(Path(out).parent / "tables")
        return {"subcircuit": s.meta.get("subcircuit"), "pin": s.pin, "class_sha512": s.class_sha512, "k_log": s.meta["k_log"]}
    return compose


def verifier_window(window_dir: Path, law: Mapping[str, Any], template_sets: Path | None) -> tuple[Any, dict, dict]:
    """The verifier's window and circuit record from the records (`interface.of_records`), and a5's template sets: each
    statement's template composed from its Definition (`composer`), every other template lowered from the verifier's own
    input set, whose Definition it resolves to; any other template from Verity's catalog (`interface.resolve`). A window
    with row tables is refused: a5 checks their segment roots against the ones recorded before each coin (``step_roots``),
    which the records don't carry yet."""
    import a5 as A5

    own = A5.template_sets(template_sets) if template_sets else {}
    resolve = lambda t: own[t][1] if t in own else IF.resolve(t)  # noqa: E731
    mine, doc = IF.of_records(window_dir, law, resolve)
    if mine.row_tables:
        raise ValueError(f"the window has {len(mine.row_tables)} row tables, and its records don't carry the segment roots "
                         "recorded before each segment's coin, so the audit can't check the tables against them: refused "
                         "until they do")
    sets = {**own, **{t: (composer(resolve(t), st), resolve(t)) for t, st in doc["statements"].items()}}
    return mine, doc, sets


def output_class(circuit: Mapping[str, Any], given: Iterable[Iterable[int]] = (), own: Iterable[str] = ()) -> list[list[int]]:
    """The units the verifier proves whole (a5's output class), as ascending ``[lo, hi)`` ranges: the ``given`` ranges (the
    verifier's own, from the circuit's shape) and every unit an uncut registered read of an output port (``o<j>``)
    covers in the circuit record (``registered``'s ``reads``: units ``[at, at + prod(radix))`` of its reader template,
    the population's indices from the templates' order), unless it reads one of the window's ``own`` values
    (`window_pins`: a tie inside the window, a wired loop's activations), whose units the draw samples like any other. A
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
    """At most ``most`` of `zk_mutants.py`'s one-bit edits of the first drawn session's proof (a prover that sent the edited
    proof, the round messages that carried it rebuilt and the coins kept), each verified by Lean `verify --zk` on the honest
    member's circuit and public file: what the verifier of record refuses, not the prover. Then that public file with one
    bit flipped (``public-bit-*``) under the honest proof."""
    import a0
    sys.path.insert(0, str(HERE.parents[1] / "verity" / "ml" / "flock" / "verifier"))
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


def audit(a: argparse.Namespace) -> int:
    """The audit (module docstring) into ``a.out``: ``summary.json`` and a ``RECORDS`` line, its ``definitions`` the module
    each member template's Definition is in and ``loaded`` the ``verity*`` packages the process loaded. -> 0 when it passed."""
    import a5 as A5
    import a5_coins as AC

    t0, wdir, out = time.time(), Path(a.window), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    law = D.parse_law(a.law)
    try:
        mine, doc, sets = verifier_window(wdir, law, Path(a.template_sets) if a.template_sets else None)
        pins = pinned(json.loads(Path(a.registered).read_text())) if a.registered else {}
        def resolve(t: str) -> Any:
            return sets[t][1] if t in sets else IF.resolve(t)
        own = json.loads((wdir / "registration.json").read_text()).get("window") or {}
        ties, untied = window_pins(own, doc, resolve)
        if twice := sorted((set(ties) | set(untied)) & set(pins)):
            raise ValueError(f"values {twice} are both pinned and the window's own")
        pins |= ties | untied
        roots = window_roots(own)
        reads = IF.registered_reads(doc, mine, pins, registered=REG)
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
    a.compose_cache = None
    a.outputs = output_class(doc, json.loads(Path(a.outputs).read_text()) if a.outputs else (), own=set(ties) | set(untied))
    if a.forged:
        os.environ["FC_ZK_SELF_CHECK"] = "skip"
    if a.challenger:
        manager, a.coins, finisher = AC.start(a.challenger, mine.run, [0], out / "challenger", a.main, ssh_key=a.gate_key,
                                              peer=a.peer, key_record=a.challenger_key)
    try:
        s, ok = A5.audit(a, sets, out / "window", mine, wdir, lambda x: source(x, held), a.port,
                         {"verifier_window_s": window_s, "circuit": held["circuit"], **({"compile": compiled} if compiled else {})},
                         more, rank=0, registered=reads or None, ports=a.ports)
    finally:
        if a.coins is not None:
            a.coins.skip(0)
            finisher.join()
            published = a.coins.evidence()
            manager.shutdown()
    row = {"seconds": round(time.time() - t0, 1), "verifier_window_s": window_s, "circuit": held["circuit"],
           "compiler": held["compiler"], "advice": held["advice"], "compile": compiled, "law": law,
           **{x: s.get(x) for x in ("run_root", "population", "members", "registration", "source", "refused", "checks_before_draw",
                                    "audit", "draw", "draw_counts", "negatives", "negatives_skipped", "coins",
                                    "timing", "outputs")}}
    bits = int((held["advice"] or {}).get("bits", 0))
    counted = pin_roots(pins)
    row["leak"] = (declared_count(doc, bits, resolve, counted, roots, untied)
                   if set(IF.DECLARED) & set(doc) else leak_count(mine.members, doc.get("external") or (), bits))
    if reads:
        row["registered"] = {"pins": counted, "roots": roots, "members": len(reads)}
    acts = committed_activations(w)
    if acts is not None:
        row["activations"] = acts
    row["definitions"] = {t: _code(fn) for t, (_c, fn) in sets.items() if t in {m.template for m in mine.members}}
    row["loaded"] = sorted({m.split(".")[0] for m in sys.modules if m.startswith("verity")})
    if a.challenger:
        row["coins_session"] = {"trusted": published.get("trusted"), "error": published.get("error")}
    if s.get("draw") is not None:
        coins = "challenger" if a.challenger else "verifier"
        rec, receipt = (json.loads((out / "window" / f).read_text()) for f in ("registration.json", "receipt.json"))
        bound = {"compiler": held["compiler"], "advice": held["advice"]} if held["compiler"] is not None else None
        row["fixed_before_draw"] = fixed_before_draw(receipt, rec, bound, coins)
        ordered = row["fixed_before_draw"]["ok"]
        if bound is not None:
            row["after_draw"] = after_draw(receipt, rec, held, coins)
            row["negatives"] = {**(row["negatives"] or {}), "advice-root-after-the-draw": row["after_draw"]["refused"]}
            ordered = ordered and row["after_draw"]["refused"]
    else:
        ordered = False
    if a.forged:
        verdicts = (s.get("audit") or {}).get("verdicts") or {}
        checks = {"checks_before_draw": bool(s.get("checks_before_draw")) and all(s["checks_before_draw"].values()),
                  "rejected": (s.get("audit") or {}).get("accepted") is False,
                  "lean_refused": any(v is False for k, v in verdicts.items() if k.startswith("lean-verify/"))}
        row["forged"] = {"ok": all(checks.values()), "checks": checks, "why": (s.get("audit") or {}).get("why")}
        row["ok"] = row["forged"]["ok"] and ordered
    else:
        row["ok"] = ok and ordered and (compiled is None or compiled["ok"])
        if a.flip_proofs and (s.get("audit") or {}).get("accepted"):
            pe = row["proof_edits"] = proof_edits(a.lean, out / "window", mine.members, out / "proof-edits", a.flip_proofs)
            row["ok"] = row["ok"] and pe["edits"] > 0 and pe["refused"] == pe["edits"]
            print("PROOF-EDITS " + json.dumps({x: pe.get(x) for x in ("edits", "refused", "member", "session")}))
    if a.challenger:
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
                                                                          "bound", "untied") if k in row["leak"]},
                                     **({"forged": row["forged"]} if a.forged else {})}, sort_keys=True, default=str))
    return 0 if row["ok"] else 1


def main(argv: list[str] | None = None) -> int:
    import a5 as A5

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("audit", help="a window's audit from its records alone")
    p.add_argument("--window", required=True, help="the window's directory: circuit.json, registration.json, the value files")
    p.add_argument("--law", required=True, help="the proof system's law of the window (subset:K, stratified:K, work:K)")
    p.add_argument("--opening", help="the advice's opening (interface.advice_opening): the named gap, until the compile unit")
    p.add_argument("--compilers", help="with --opening: the registered compilers the verifier holds, {name: {digest, argv, "
                   "code, pythonpath}}")
    p.add_argument("--repo", help="with --opening: the repository a registered compiler's commit is checked out from (default "
                   "this tree's)")
    p.add_argument("--circuit", help="instead: the SHA-512 of the circuit record the verifier registered, a circuit with no compiler")
    p.add_argument("--run", required=True, help="the run root the verifier holds the window to, the one the run published")
    p.add_argument("--template-sets", help="the verifier's own input sets, for templates whose units are not bit rows")
    p.add_argument("--outputs", help="JSON: the verifier's own [lo, hi) ranges of the units whose outputs are served out, proved "
                   "whole beside the draw (`output_class`; a work law)")
    p.add_argument("--m0-pythonpath", required=True)
    p.add_argument("--bin", required=True)
    p.add_argument("--lean", required=True)
    p.add_argument("--firewall", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--delta", type=float, default=0.5)
    p.add_argument("--port", type=int, default=None, help="the sessions' base port (default a base whose --ports ports are "
                   "free now)")
    p.add_argument("--ports", type=int, default=None, help="how many ports from --port are the audit's: a session each for "
                   "every member with a proved unit, a draw needing more refused before any listens (default one a member)")
    p.add_argument("--session-timeout", type=float, default=14400)
    p.add_argument("--session-jobs", type=A5.one_session, default=1, help="1 only: a5 runs the sessions one at a time (a5.ONE_AT_A_TIME)")
    p.add_argument("--forged", action="store_true", help="the window holds a forged unit: it must be refused")
    p.add_argument("--flip-proofs", type=int, default=0, metavar="N",
                   help="after an accepted audit, at most N one-bit edits of its first drawn session's proof, each of which "
                   "Lean verify must refuse (`proof_edits`)")
    p.add_argument("--challenger", help="draw on the auditor's challenger's coins (a5_coins), not the verifier's: local, loopback:HOST:PORT or "
                   "gate:USER@HOST (the auditor host through its SSH gate, with --peer and --challenger-key)")
    p.add_argument("--peer", help="with --challenger gate:USER@HOST: this machine as the gate's /etc/vy/peers names it, whose "
                   "127.0.0.1 the session's challenger listens on")
    p.add_argument("--challenger-key", help="with --challenger gate:USER@HOST: the challenger's key record of its deployment "
                   "(challenger-key/v1), never the session's key.json")
    p.add_argument("--gate-key", help="with --challenger gate:USER@HOST: the lane's own gate key (default ssh's own)")
    p.add_argument("--main", help="with --challenger: the verifier's copy of main's first-parent history")
    p.add_argument("--registered", help="the registered values' entries the verifier pinned before the run, {program, "
                   "values} (`pinned`), for a circuit that reads them")
    p.add_argument("--skip-negatives", action="store_true", help="no negatives in this audit (module docstring): a run of "
                   "several audits runs them in its first")
    p = sub.add_parser("count", help="a stated circuit record's leak count from its description alone (declared_count)")
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
    a = ap.parse_args(argv)
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
    if bool(a.opening) == bool(a.circuit) or bool(a.opening) != bool(a.compilers):
        ap.error("a compiled circuit: --opening FILE --compilers FILE; a circuit with no compiler: --circuit SHA512")
    if a.challenger:
        if a.challenger != "local" and not a.challenger.startswith(("gate:", "loopback:")):
            ap.error(f"--challenger is local, loopback:HOST:PORT or gate:USER@HOST, not {a.challenger!r}")
        if a.challenger.startswith("gate:") and not (a.peer and a.challenger_key):
            ap.error("--challenger gate:USER@HOST needs --peer and --challenger-key")
        if not a.main:
            ap.error("--challenger needs --main")
    return audit(a)


if __name__ == "__main__":
    raise SystemExit(main())
