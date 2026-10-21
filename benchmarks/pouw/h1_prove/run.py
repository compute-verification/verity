"""run.py: one recorded H1 window's drawn tile units through sampled proofs, the challenger's and the verifier's side.

    python run.py prepare --work W --stage SW --lean FLOCK_VERIFY --layer L1.npz --layer2 L2.npz [--draw 8] [--k K] [--cols N]
    python run.py summarize --work W --out OUT

``prepare`` runs one epoch of the window and plans its proofs, the prover's steps by `window.py` (a subprocess: the
challenger and the verifier never hold a row, a salt or a served value):

1. the challenger (`pouw.coins.Challenger`) commits to the epoch's salt and draw coins; the prover commits the window's
   A and B rows (`window.py commit`) as the registered values a/<K> and b/<K> of the window's public Program, and the
   salt coin opens after those roots are recorded;
2. the prover serves every tile's z and receipt under the unit salt, PoUW's window salt (`h1unit.unit_salt`: `pouw_salt`
   of the coin's stand-in for the certifier's reveal, the epoch as the window, a stand-in stream and prover) and commits
   them (`window.py serve`); the registration (the window's four roots) is recorded, and the draw coin opens after it. The
   verifier builds the window's public Program and its template-instance partition from the commitment's public shape
   (`window_program`: WORK/verifier/program.json and partition.json, Lean's ``--program`` and ``--partition``), holds
   the commitment's registration to them (`check_registration`), and the stager builds its statement bound to them;
3. the verifier draws ``--draw`` units of the window's tiles by Lean ``flock-verify draw`` on the coin's stream, and
   writes its ``--registered`` file (`registered_file`: `h1unit.registered_reads` from its own Program under the
   window's roots, WORK/verifier/w1.registered.json);
4. the prover opens the drawn units (`window.py open`), and the verifier checks each opened row is the one the Program's
   unit reads (`h1unit.unit_reads`) with a path to its root (a/<K>'s and b/<K>'s registered roots under the Program's
   digest), and builds its own public file (`h1unit.verifier_pub`: a shared-row file, each A and B port's table the
   registered rows its units read, at the positions its Program reads and the opened paths) from the opened ``b || c``,
   its own public row and the drawn units' indices;
5. the stager (`stage.py`, building the circuit once the window's shape is known) stages the drawn units as one
   shared-row statement on the prover's rows at their indices, and the prover's pub.bin must be the verifier's byte for
   byte. Lean (``verify --zk --registered``) derives each instance's unit from its own Program, and an accepted check
   must say so (``units: derived``), and opens each A and B table row against its registered root at the position its
   Program reads.

The planted negatives, each checked by Lean against the verifier's own file:

* ``flipped-z``: a window registered with the first drawn unit's z committed with one word flipped (`window.py tamper`,
  every salt kept). Its openings pass their paths (a consistent window), so the verifier's file holds the flipped leaf;
  the only proof of those rows the unit admits is the honest one, which Lean must reject against that file.
* ``wrong-receipt``: the same with that unit's receipt from another salt.
* ``other-window``: the first drawn unit's tile, opened from a second window (other tokens, its own epoch and salt) and
  proved as a statement of its own: its opening fails window 1's roots, and Lean must reject its proof against the
  verifier's file of window 1's unit. ``other-window-own`` (its own window's file) must accept: the proof is sound for
  the   window it came from.

Those three fail at binding (the proof's statement is not the verifier's). ``tampered-a`` passes binding and fails the
registered reads:

* ``tampered-a``: the drawn units' statement staged on rows whose first unit's a0 row (at every unit that reads it) is not
  the registered one (`window.py open --tamper-a`: one code flipped, a fresh salt, the registered path kept), checked
  against its own public file with the verifier's ``--registered`` file. The proof is of the rows it commits, so Lean
  must reject it for that table row not opening a/<K>'s registered root, and say so; the verifier's own check of the
  row (`registered_opens`) refuses it too, and opens the honest statement's.

The in-proof negatives (`IN_PROOF`) pass both and carry a wrong witness instead:

* ``in-receipt``: a window registered with the first drawn unit's receipt word 0 plus one (`window.py tamper
  --receipt-word`), opened and checked like the honest one, and a statement of it (`h1unit.forge_statement`): the honest
  prover file with that unit's receipt row the window's, under the verifier's own file of the window. The prover's
  pub.bin is the verifier's, every committed row is the window's, and the receipt row is not what the circuit computes on
  the unit's inputs, salt, position and roots.
* ``in-z``: the same with its z cell 0 plus one (``--z-cell``).
* ``in-d``: the same with a kernel's D_1 word (lane 0, entry 0) one too large (``--d-word``): z and the receipt both
  computed from it, so both output rows are replaced; under ``H1_TILE=v3`` z is that word's recombination.
* ``in-xa`` (``H1_TILE=v3`` only): the same with its z computed with one entry's X wrap bit xa = [A = 127] ∧ b flipped
  (``--xa-flip``: the first set one, else the first clear one; the receipt the honest one), as a prover that skipped (or
  invented) the activation-wrap correction would serve it. Its tampered window's served.json says whether the flipped
  bit was set and how many z words moved.

Each is proved like the honest statement, whose prover stops on the failing witness before it sends a proof, and again as
``<tag>-skip`` with ``FC_ZK_SELF_CHECK=skip`` (C-Flock's negative-control prover, which sends its proofs anyway), so the
in-session verifier and Lean judge a proof. ``summarize`` records which layer refused each (the prover, the in-session
verifier, Lean) and with what message.

``summarize`` reads the job's proving and Lean records into OUT/summary.json: the unit and statement counts, proving
time and Lean verify time, and every check's verdict against its expectation.
"""
from __future__ import annotations

import argparse
import hashlib
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
from verity.core.protocols.pouw import core_check as CC  # noqa: E402
from verity.core.protocols.pouw import verifier  # noqa: E402
from verity.core.protocols.pouw.coins import Challenger, coin_reason  # noqa: E402

DRAW_DOMAIN = "verity/pouw/h1-prove-draw/v0"
R, C = U.TILE
#: the in-proof negatives: the tampered window's flag, and the output row whose committed words its statement carries
IN_PROOF = {"in-receipt": ("--receipt-word", ("o1",)), "in-z": ("--z-cell", ("o0",)), "in-d": ("--d-word", ("o0", "o1")),
            **({"in-xa": ("--xa-flip", ("o0",))} if U.tile() == "v3" else {})}
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
    Program's digest, and its values a/<K> and b/<K> are the slot's rows and the calls' B rows, K/2 16-bit leaves a row,
    under that digest's domains (`h1unit.registered_domain`), with the commitment's roots."""
    k, prog, why = com["unit"]["k"], win.obj["program"], []
    if com.get("program_sha512") != prog:
        why.append(f"it registers under Program {com.get('program_sha512')}, not the verifier's {prog}")
    want = [{"value": U.value(t, k), "root": com[f"root_{t}"], "leaves": n, "words": k // 2,
             "domain": U.registered_domain(prog, U.value(t, k), k, n).domain_id.hex()}
            for t, n in (("a", com["slot"]), ("b", com["b_rows"]))]
    if com.get("registered") != want:
        why.append(f"its registered entries {com.get('registered')} are not the verifier's {want}")
    return why


def check_opening(win: SimpleNamespace, com: dict, served: dict, op: dict, u: int) -> list[str]:
    """Why unit ``u``'s opening ``op`` is not the registered window's (``win`` its Program, ``com`` its commitment, ``served``
    its output roots): each committed row is the one the Program's unit ``u`` reads (`h1unit.unit_reads`), and its path
    reaches that tree's registered root: a/<K>'s and b/<K>'s under the verifier's own Program digest
    (`h1unit.verify_registered`), z's and the receipt's under the window's domains."""
    w, T, k, why = com["window"], com["tiles"], com["unit"]["k"], []
    trees = {"a": (com["slot"], com["root_a"]), "b": (com["b_rows"], com["root_b"]), "z": (T, served["z_root"]),
             "receipt": (T, served["receipt_root"])}
    if op.get("unit") != u:
        why.append(f"the opening is unit {op.get('unit')}'s, not {u}'s")
    for port, (tree, index) in U.unit_reads(win.program, win.tp, u).items():
        if tree not in trees:
            continue
        count, root = trees[tree]
        r = op["rows"].get(port)
        if r is None or r["tree"] != tree or r["index"] != index:
            why.append(f"{port}: not row {index} of tree {tree}")
            continue
        bc = bytes.fromhex(r["bc"])
        if tree in ("a", "b"):
            ok = U.verify_registered(win.obj["program"], U.value(tree, k), k, count, bytes.fromhex(root), index, bc, r["path"])
        else:
            ok = U.verify(U.domain(w, tree, count), bytes.fromhex(root), index, bc, r["path"])
        if not ok:
            why.append(f"{port}: its path does not reach the registered {tree} root")
    return why


def unit_bcs(shape: U.Shape, com: dict, salt: bytes, op: dict) -> dict[str, bytes]:
    """The unit's rows' ``b || c`` as the verifier holds them: the opened leaves, and its own public row's."""
    pub = shape.ports()[R + C]
    own = U.commit_strings(pub, U.pub_words(salt, op["unit"], U.window_roots(com))[None, :], [U.pub_salt(com["window"], op["unit"])])[0]
    return {**{p: bytes.fromhex(op["rows"][p]["bc"]) for p in [*shape.in_rows, "o0", "o1"]}, "pub": own}


def unit_registered(win: SimpleNamespace, k: int, op: dict, u: int) -> dict[str, tuple[str, int, str]]:
    """Unit ``u``'s registered reads as the verifier's public file states them (`h1unit.verifier_pub`'s ``reads``): each A
    and B port's value, the position the verifier's Program reads (`h1unit.unit_reads`) and the opened path."""
    return {port: (U.value(tree, k), index, "".join(op["rows"][port]["path"]))
            for port, (tree, index) in U.unit_reads(win.program, win.tp, u).items() if tree in ("a", "b")}


def registered_file(path: Path, win: SimpleNamespace, com: dict, units: list[int]) -> dict:
    """The verifier's ``--registered`` file of a window (`h1unit.registered_reads`, under its commitment's roots), written
    to ``path``, held to the h1 window's own reads (`h1unit.unit_reads`) at the drawn ``units``."""
    k, t = com["unit"]["k"], time.time()
    doc = U.registered_reads(win.program, win.query, {U.value(x, k): com[f"root_{x}"] for x in "ab"})
    for u in units:
        want = {p: i for p, (tree, i) in U.unit_reads(win.program, win.tp, u).items() if tree in ("a", "b")}
        got = {p: e["reads"][u] for p, e in doc["ports"].items()}
        if got != want:
            raise SystemExit(f"unit {u}: the registered reads {got} are not the window's {want}")
    path.write_text(json.dumps(doc))
    return {"ports": sorted(doc["ports"]), "seconds": round(time.time() - t, 1)}


def registered_opens(pub: bytes, ports: list, port: str, row: int, win: SimpleNamespace, com: dict) -> bool:
    """Whether table row ``row`` of registered port ``port`` in a shared-row public file opens its value's registered root
    at the position and through the path its header states: the check flock-verify ``--registered`` makes of each table
    row (`h1unit.verify_registered`)."""
    nl = pub.index(b"\n") + 1
    head, body, names = json.loads(pub[:nl]), pub[nl:], [p.name for p in ports]
    at = sum(int(head["shared_rows"][q]) for q in names[:names.index(port)]) + row
    e, k = head["registered"][port], com["unit"]["k"]
    tree = e["value"].split("/")[0]
    return U.verify_registered(win.obj["program"], e["value"], k, com["slot"] if tree == "a" else com["b_rows"],
                               bytes.fromhex(com[f"root_{tree}"]), e["positions"][row], body[128 * at:128 * (at + 1)],
                               [e["paths"][row]])


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
    coin, serve. Returns its commitment, served roots, the unit salt and the challenger (its draw coin still closed)."""
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
    return digest(json.dumps(com, sort_keys=True).encode(), json.dumps({k: served[k] for k in ("z_root", "receipt_root")},
                                                                       sort_keys=True).encode())


def lean_draw(lean: str, n: int, k: int, src: bytes) -> list[int]:
    p = subprocess.run([lean, "draw", "--population", str(n), "--k", str(k), "--stream", src.hex()], capture_output=True, text=True)
    lines = [x[5:] for x in p.stdout.splitlines() if x.startswith("DRAW ")]
    if p.returncode != 0 or len(lines) != 1:
        raise SystemExit(f"flock-verify draw: exit {p.returncode}: {(p.stderr or p.stdout)[-300:]}")
    return [int(u) for u in json.loads(lines[0])["units"]]


def draw_units(lean: str, coin: bytes, window: str, registration: str, population: int, k: int) -> list[int]:
    """The `k` units of `population` a window's draw coin selects, after its registration (hex) was recorded: Lean
    ``flock-verify draw`` on the coin's stream under `DRAW_DOMAIN`, bound to the window's name and its registration."""
    key = derive(coin, DRAW_DOMAIN, {"window": window, "registration": registration})
    return sorted(lean_draw(lean, population, k, key.stream("draw", 2 * k * CC.width(population) + 64)))


def cmd_prepare(a) -> int:
    work, sw = Path(a.work), Path(a.stage)
    work.mkdir(parents=True, exist_ok=True)
    (sw / "requests").mkdir(parents=True, exist_ok=True)
    record: dict = {"schema": "verity/pouw/h1-prove/record/v0", "draw_k": a.draw}
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
    reg = registration(com1, served1)
    do = ch1.open("draw", reg)
    assert coin_reason(ch1.commitment, do, "draw", reg) is None
    T = com1["tiles"]
    units = draw_units(a.lean, do.coin, w1, reg.hex(), T, a.draw)
    u0 = units[0]
    record["windows"][w1].update(registration=reg.hex(), draw_opening=do.to_json())
    record["draw"] = {"population": T, "k": a.draw, "units": units, "by": "flock-verify draw (Lean)",
                      "reads": {str(u): U.unit_reads(win.program, win.tp, u) for u in units}}
    log(f"drew {units} of {T}")
    k = com1["unit"]["k"]
    shape = U.Shape(k, R, C)
    reg1 = vp / "w1.registered.json"
    record["registered_reads"] = {"w1": registered_file(reg1, win, com1, units)}
    log(f"the verifier's registered reads of {w1}: {record['registered_reads']['w1']}")

    # the honest openings, and the two tampered windows' (every salt kept, the first drawn unit's z or receipt changed)
    opened, reads = {}, {}
    for tag, priv, flag in (("honest", w1, None), ("flipped-z", f"{w1}-z", "--z"), ("wrong-receipt", f"{w1}-r", "--receipt"),
                            *((t, f"{w1}-{t}", f) for t, (f, _p) in IN_PROOF.items())):
        if flag:
            served_t = prover("tamper", str(work / w1), str(work / priv), str(work / f"{priv}-served.json"), flag, str(u0))
            if flag == "--xa-flip":
                record["xa_flip"] = served_t["tampered"]
        else:
            served_t = served1
        prover("open", str(work / priv), salt1.hex(), ",".join(map(str, units)), str(work / f"open-{tag}"))
        ops = json.loads((work / f"open-{tag}" / "opening.json").read_text())["units"]
        why = {u: check_opening(win, com1, served_t, op, u) for u, op in zip(units, ops)}
        why_vs_honest = {u: check_opening(win, com1, served1, op, u) for u, op in zip(units, ops)}
        opened[tag] = [unit_bcs(shape, com1, salt1, op) for op in ops]
        reads[tag] = [unit_registered(win, k, op, u) for u, op in zip(units, ops)]
        record.setdefault("openings", {})[tag] = {
            "registered": {k: served_t[k] for k in ("z_root", "receipt_root")},
            "refused": {str(u): w for u, w in why.items() if w},
            "against_honest_registration": {str(u): w for u, w in why_vs_honest.items() if w}}
        log(f"opening {tag}: {sum(1 for w in why.values() if w)} of {len(units)} refused against its registration, "
            f"{sum(1 for w in why_vs_honest.values() if w)} against the honest one")
    if record["openings"]["honest"]["refused"]:
        raise SystemExit(f"the honest openings are refused: {record['openings']['honest']['refused']}")
    # a prover that stages an A row the registration does not hold (`window.py open --tamper-a`): its opening is the honest one
    record["tampered_a"] = prover("open", str(work / w1), salt1.hex(), ",".join(map(str, units)), str(work / "open-tampered-a"),
                                  "--tamper-a")["tampered"]

    # the second window, its first drawn unit's tile presented as window 1's
    com2, served2, salt2, _ch2 = epoch(work, w2, a.layer2, 2, a, record, win.obj["program"])
    if U.window_shape(com2) != U.window_shape(com1) or com2["tiles"] != T:
        raise SystemExit("the second window's shape (its slot and calls) is not the first's, so its Program is another")
    if why := check_registration(win, com2):
        raise SystemExit(f"{w2}'s registration is not of the verifier's Program: {why}")
    reg2 = vp / "w2.registered.json"
    record["registered_reads"]["w2"] = registered_file(reg2, win, com2, [u0])
    prover("open", str(work / w2), salt2.hex(), str(u0), str(work / "open-other"))
    op2 = json.loads((work / "open-other" / "opening.json").read_text())["units"][0]
    record["openings"]["other-window"] = {"unit": u0, "refused": check_opening(win, com1, served1, op2, u0),
                                          "against_own_window": check_opening(win, com2, served2, op2, u0)}
    log(f"opening other-window: refused against window 1 for {len(record['openings']['other-window']['refused'])} rows")

    set_h, set_o = f"{w1}-draw", f"{w1}-u{u0}"
    stmts = {"honest": {"rows": str(work / "open-honest" / "rows.npz"), "set": set_h, "out": str(sw / "st-honest"),
                        "indices": units},
             "other": {"rows": str(work / "open-other" / "rows.npz"), "set": set_o, "out": str(sw / "st-other"), "indices": [u0]},
             "tampered-a": {"rows": str(work / "open-tampered-a" / "rows.npz"), "set": set_h, "out": str(sw / "st-tampered-a"),
                            "indices": units}}
    for name, req in stmts.items():
        (sw / "requests" / f"{name}.json").write_text(json.dumps(req))

    wait_for(sw / "circuit.json", "circuit built", a.stager_pid)
    head = json.loads((sw / "circuit.json").read_text())
    record["circuit"] = {x: head[x] for x in head if x != "meta"}
    bound = {"partition": head["meta"].get("partition"), "program_sha512": head["meta"].get("program_sha512")}
    if bound != {"partition": {"rule": "verity/partition/v1", "digest": P.digest(win.obj)}, "program_sha512": win.obj["program"]}:
        raise SystemExit(f"the unit's statement binds {bound}, not the window's partition and Program")
    files = {"honest": (set_h, opened["honest"], units, reads["honest"]),
             "flipped-z": (set_h, opened["flipped-z"], units, reads["flipped-z"]),
             "wrong-receipt": (set_h, opened["wrong-receipt"], units, reads["wrong-receipt"]),
             "other-window": (set_o, [opened["honest"][0]], [u0], [reads["honest"][0]]),
             "other-window-own": (set_o, [unit_bcs(shape, com2, salt2, op2)], [u0], [unit_registered(win, k, op2, u0)]),
             **{t: (set_h, opened[t], units, reads[t]) for t in IN_PROOF}}
    for tag, (set_name, bcs, indices, rd) in files.items():
        (vp / f"{tag}.pub.bin").write_bytes(U.verifier_pub(head, set_name, head["subcircuit"], bcs, indices, rd))
    for name, req in stmts.items():
        wait_for(Path(req["out"]) / "stage.json", f"staged {name}", a.stager_pid)
        st = json.loads((Path(req["out"]) / "stage.json").read_text())
        if "error" in st:
            raise SystemExit(f"staging {name} failed: {st['error']}")
        record.setdefault("staged", {})[name] = st
    eq = {"honest": (sw / "st-honest" / "pub.bin").read_bytes() == (vp / "honest.pub.bin").read_bytes(),
          "other-window-own": (sw / "st-other" / "pub.bin").read_bytes() == (vp / "other-window-own.pub.bin").read_bytes()}
    record["prover_pub_is_verifiers"] = eq
    log(f"the prover's pub.bin is the verifier's own: {eq}")
    if not eq["honest"]:
        raise SystemExit("the honest statement's pub.bin is not the verifier's own file")
    # the tampered A row's table row in its statement's own public file, as flock-verify --registered checks it
    ta = record["tampered_a"]
    record["tampered_a"]["table_row_0_opens"] = {
        x: registered_opens((sw / f"st-{x}" / "pub.bin").read_bytes(), shape.ports(), ta["port"], 0, win, com1)
        for x in ("honest", "tampered-a")}
    log(f"tampered-a: {ta}")
    if ta["table_row_0_opens"] != {"honest": True, "tampered-a": False}:
        raise SystemExit(f"the tampered A row's table row opens the registered root as {ta['table_row_0_opens']}, not only the honest one")
    # the in-proof negatives: the honest statement with the first drawn unit's (instance 0's) output row the tampered
    # window's, under the verifier's own file of that window, so binding holds and the witness is wrong
    for t, (_f, ports) in IN_PROOF.items():
        for d in (t, f"{t}{SKIP}"):
            src = sw / "st-honest"
            for port in ports:
                words = np.load(work / f"open-{t}" / "rows.npz")[f"w_{port}"][0]
                record.setdefault("forged", {}).setdefault(d, {})[port] = U.forge_statement(
                    src, sw / f"st-{d}", (vp / f"{t}.pub.bin").read_bytes(), 0, port, words, sw / "circuit.txt")
                src = sw / f"st-{d}"
        log(f"forged {t}: {record['forged'][t]}")
    st = record["staged"]
    r1, r2 = str(reg1), str(reg2)
    refusal = f"registered reads: port {ta['port']}: table row 0 does not open {U.value('a', k)}'s registered root at position {ta['position']}"
    plan = {"statements": [{"tag": "honest", "dir": stmts["honest"]["out"], "n": len(units)},
                           {"tag": "other", "dir": stmts["other"]["out"], "n": 1},
                           {"tag": "tampered-a", "dir": stmts["tampered-a"]["out"], "n": len(units)},
                           *({"tag": d, "dir": str(sw / f"st-{d}"), "n": len(units)} for t in IN_PROOF for d in (t, f"{t}{SKIP}"))],
            "checks": [{"tag": "honest", "statement": "honest", "pub": str(vp / "honest.pub.bin"), "registered": r1, "expect": "ACCEPT"},
                       {"tag": "flipped-z", "statement": "honest", "pub": str(vp / "flipped-z.pub.bin"), "registered": r1,
                        "expect": "REJECT"},
                       {"tag": "wrong-receipt", "statement": "honest", "pub": str(vp / "wrong-receipt.pub.bin"), "registered": r1,
                        "expect": "REJECT"},
                       {"tag": "other-window", "statement": "other", "pub": str(vp / "other-window.pub.bin"), "registered": r1,
                        "expect": "REJECT"},
                       {"tag": "other-window-own", "statement": "other", "pub": str(vp / "other-window-own.pub.bin"),
                        "registered": r2, "expect": "ACCEPT"},
                       # against its own public file, so only the registered reads can refuse it, and they must say why
                       {"tag": "tampered-a", "statement": "tampered-a", "pub": str(sw / "st-tampered-a" / "pub.bin"),
                        "registered": r1, "expect": "REJECT", "why": refusal},
                       *({"tag": d, "statement": d, "pub": str(vp / f"{t}.pub.bin"), "registered": r1, "expect": "REFUSED"}
                         for t in IN_PROOF for d in (t, f"{t}{SKIP}"))]}
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
    """The last ``TAG<tab>{json}`` line of a C-Flock output."""
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
    checks, ok = [], True
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
    honest = next(c for c in checks if c["tag"] == "honest")
    summary = {"schema": "verity/pouw/h1-prove/summary/v0", "pass": ok, "firewall": firewall,
               "unit": record["circuit"]["definition"], "k": record["circuit"]["k"], "tile": [R, C],
               "ands": record["circuit"]["ands"], "k_log": record["circuit"]["k_log"],
               "circuit_bytes": record["circuit"]["circuit_bytes"], "circuit_build_seconds": record["circuit"]["build_seconds"],
               "window_tiles": record["draw"]["population"], "drawn_units": len(record["draw"]["units"]),
               "program": record.get("program"),
               "units": record["draw"]["units"], "statements": len(plan["statements"]),
               "statement_instances": {s["tag"]: s["n"] for s in plan["statements"]},
               "epsilon_at_delta": {f"{d}": round(1 - d ** (1 / record["draw_k"]), 4) for d in (0.5, 0.1, 0.01)},
               "proving_seconds": {k: (v["prove"] or {}).get("wall_s") for k, v in proofs.items()},
               "lean_verify_seconds": honest["lean_seconds"], "checks": checks, "proofs": proofs,
               "openings": record["openings"], "prover_pub_is_verifiers": record["prover_pub_is_verifiers"],
               "registered_reads": record.get("registered_reads"), "tampered_a": record.get("tampered_a"),
               "windows": {w: {k: v[k] for k in ("tiles", "slot", "calls", "commit_seconds", "serve_seconds") if k in v}
                           for w, v in record["windows"].items()},
               "tile": U.tile(), "z_is_served_product": {w: v["served"].get("z_is_served_product")
                                                          for w, v in record["windows"].items()},
               "xa": {w: v["served"].get("xa") for w, v in record["windows"].items()},
               "xa_flip": record.get("xa_flip"),
               "staged": record["staged"]}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("pass", "unit", "k", "ands", "window_tiles", "drawn_units", "statements",
                                                "proving_seconds", "lean_verify_seconds")}))
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
    p.add_argument("--firewall", default="0", help="1: the provers ran under the firewall (FC_FIREWALL=1), whose times "
                                                    "the honest prover's records must carry")
    a = ap.parse_args(argv)
    return {"prepare": cmd_prepare, "summarize": cmd_summarize}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
