"""run.py: one recorded H1 window's drawn tile units through sampled proofs, the challenger's and the verifier's side.

    python run.py prepare --work W --stage SW --lean FLOCK_VERIFY --layer L1.npz --layer2 L2.npz [--draw 8] [--k K] [--cols N]
    python run.py summarize --work W --out OUT

``prepare`` runs one epoch of the window and plans its proofs, the prover's steps by `window.py` (a subprocess: the
challenger and the verifier never hold a row, a salt or a served value):

1. the challenger (`pouw.coins.Challenger`) commits to the epoch's salt and draw coins; the prover commits the window's
   A and B rows (`window.py commit`), and the salt coin opens after those roots are recorded;
2. the prover serves every tile's z and receipt under the unit salt (`h1unit.unit_salt` of the coin) and commits them
   (`window.py serve`); the registration (the window's four kinds of root) is recorded, and the draw coin opens after it;
3. the verifier draws ``--draw`` units of the window's tiles by Lean ``flock-verify draw`` on the coin's stream;
4. the prover opens the drawn units (`window.py open`), and the verifier checks each opened row's path to its registered
   root and builds its own public file (`h1unit.verifier_pub`) from the opened ``b || c`` and its own public row;
5. the stager (`stage.py`, building the circuit since the job started) stages the drawn units as one statement on the
   prover's rows, and the prover's pub.bin must be the verifier's byte for byte.

The planted negatives, each checked by Lean against the verifier's own file:

* ``flipped-z``: a window registered with the first drawn unit's z committed with one word flipped (`window.py tamper`,
  every salt kept). Its openings pass their paths (a consistent window), so the verifier's file holds the flipped leaf;
  the only proof of those rows the unit admits is the honest one, which Lean must reject against that file.
* ``wrong-receipt``: the same with that unit's receipt from another salt.
* ``other-window``: the first drawn unit's tile, opened from a second window (other tokens, its own epoch and salt) and
  proved as a statement of its own: its opening fails window 1's roots, and Lean must reject its proof against the
  verifier's file of window 1's unit. ``other-window-own`` (its own window's file) must accept: the proof is sound for
  the   window it came from.

Those three fail at binding (the proof's statement is not the verifier's). The in-proof negatives (`IN_PROOF`) pass it and
carry a wrong witness instead:

* ``in-receipt``: a window registered with the first drawn unit's receipt word 0 plus one (`window.py tamper
  --receipt-word`), opened and checked like the honest one, and a statement of it (`h1unit.forge_statement`): the honest
  prover file with that unit's receipt row the window's, under the verifier's own file of the window. The prover's
  pub.bin is the verifier's, every committed row is the window's, and the receipt row is not what the circuit computes on
  the unit's inputs, salt, position and roots.
* ``in-z``: the same with its z cell 0 plus one (``--z-cell``).

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

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import h1unit as U  # noqa: E402
from verity.ml.randomness import derive  # noqa: E402
from verity.core.protocols.pouw import core_check as CC  # noqa: E402
from verity.core.protocols.pouw.coins import Challenger, coin_reason  # noqa: E402

DRAW_DOMAIN = "verity/pouw/h1-prove-draw/v0"
R, C = 4, 8
#: the in-proof negatives: the tampered window's flag, and the output row whose committed words its statement carries
IN_PROOF = {"in-receipt": ("--receipt-word", "o1"), "in-z": ("--z-cell", "o0")}
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

def tile_of(com: dict, u: int) -> tuple[dict, int, int]:
    for call in com["calls"]:
        if call["tile0"] <= u < call["tile0"] + call["tiles"]:
            v = u - call["tile0"]
            return call, v // (call["n"] // C), v % (call["n"] // C)
    raise ValueError(f"unit {u} is no tile of the window")


def check_opening(com: dict, served: dict, op: dict, u: int) -> list[str]:
    """Why unit ``u``'s opening ``op`` is not the registered window's (``com`` its commitment, ``served`` its output roots):
    each row's tree and index are the unit's, and its path reaches the registered root."""
    call, ti, tj = tile_of(com, u)
    w, T, why = com["window"], com["tiles"], []
    if op.get("unit") != u:
        why.append(f"the opening is unit {op.get('unit')}'s, not {u}'s")
    want = ({f"a{i}": (f"a/{call['name']}", R * ti + i, call["m"], call["root_a"]) for i in range(R)}
            | {f"b{j}": (f"b/{call['name']}", C * tj + j, call["n"], call["root_b"]) for j in range(C)}
            | {"o0": ("z", u, T, served["z_root"]), "o1": ("receipt", u, T, served["receipt_root"])})
    for port, (tree, index, count, root) in want.items():
        r = op["rows"].get(port)
        if r is None or r["tree"] != tree or r["index"] != index:
            why.append(f"{port}: not row {index} of tree {tree}")
        elif not U.verify(U.domain(w, tree, count), bytes.fromhex(root), index, bytes.fromhex(r["bc"]), r["path"]):
            why.append(f"{port}: its path does not reach the registered {tree} root")
    return why


def unit_bcs(shape: U.Shape, com: dict, salt: bytes, op: dict) -> dict[str, bytes]:
    """The unit's rows' ``b || c`` as the verifier holds them: the opened leaves, and its own public row's."""
    call, _ti, _tj = tile_of(com, op["unit"])
    pub = shape.ports()[R + C]
    roots = (bytes.fromhex(call["root_a"]), bytes.fromhex(call["root_b"]))
    own = U.commit_strings(pub, U.pub_words(salt, op["unit"], roots)[None, :], [U.pub_salt(com["window"], op["unit"])])[0]
    return {**{p: bytes.fromhex(op["rows"][p]["bc"]) for p in [*shape.in_rows, "o0", "o1"]}, "pub": own}


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

def epoch(work: Path, name: str, layer: str, number: int, a, record: dict) -> tuple[dict, dict, bytes, Challenger]:
    """One window's epoch to its registration: commit, the salt coin, serve. Returns its commitment, served roots, the unit
    salt and the challenger (its draw coin still closed)."""
    ch = Challenger(number)
    rec = record.setdefault("windows", {}).setdefault(name, {})
    rec["coin_commitment"] = {"epoch": ch.epoch, "root": ch.commitment.root.hex()}
    t = time.time()
    args = ["commit", layer, name, str(work / name), str(work / f"{name}-commitment.json")]
    if a.k:
        args += ["--k", str(a.k)]
    if a.cols:
        args += ["--cols", str(a.cols)]
    prover(*args)
    com_bytes = (work / f"{name}-commitment.json").read_bytes()
    com = json.loads(com_bytes)
    rec["commit_seconds"] = round(time.time() - t, 1)
    after = digest(com_bytes)
    so = ch.open("salt", after)
    assert coin_reason(ch.commitment, so, "salt", after) is None
    salt = U.unit_salt(so.coin)
    rec["salt_opening"] = so.to_json()
    t = time.time()
    served = prover("serve", str(work / name), salt.hex(), str(work / f"{name}-served.json"), "--workers", str(a.workers))
    rec["serve_seconds"] = round(time.time() - t, 1)
    rec.update(tiles=com["tiles"], calls={c["name"]: [c["m"], c["k"], c["n"]] for c in com["calls"]}, served=served)
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


def cmd_prepare(a) -> int:
    work, sw = Path(a.work), Path(a.stage)
    work.mkdir(parents=True, exist_ok=True)
    (sw / "requests").mkdir(parents=True, exist_ok=True)
    record: dict = {"schema": "verity/pouw/h1-prove/record/v0", "draw_k": a.draw}
    w1, w2 = f"{a.name}-w1", f"{a.name}-w2"
    com1, served1, salt1, ch1 = epoch(work, w1, a.layer, 1, a, record)
    reg = registration(com1, served1)
    do = ch1.open("draw", reg)
    assert coin_reason(ch1.commitment, do, "draw", reg) is None
    T = com1["tiles"]
    key = derive(do.coin, DRAW_DOMAIN, {"window": w1, "registration": reg.hex()})
    units = sorted(lean_draw(a.lean, T, a.draw, key.stream("draw", 2 * a.draw * CC.width(T) + 64)))
    u0 = units[0]
    record["windows"][w1].update(registration=reg.hex(), draw_opening=do.to_json())
    record["draw"] = {"population": T, "k": a.draw, "units": units, "by": "flock-verify draw (Lean)",
                      "tiles": {str(u): {"call": c["name"], "ti": i, "tj": j} for u in units for c, i, j in [tile_of(com1, u)]}}
    log(f"drew {units} of {T}")
    shape = U.Shape(com1["unit"]["k"], R, C)

    # the honest openings, and the two tampered windows' (every salt kept, the first drawn unit's z or receipt changed)
    opened = {}
    for tag, priv, flag in (("honest", w1, None), ("flipped-z", f"{w1}-z", "--z"), ("wrong-receipt", f"{w1}-r", "--receipt"),
                            *((t, f"{w1}-{t}", f) for t, (f, _p) in IN_PROOF.items())):
        if flag:
            served_t = prover("tamper", str(work / w1), str(work / priv), str(work / f"{priv}-served.json"), flag, str(u0))
        else:
            served_t = served1
        prover("open", str(work / priv), salt1.hex(), ",".join(map(str, units)), str(work / f"open-{tag}"))
        ops = json.loads((work / f"open-{tag}" / "opening.json").read_text())["units"]
        why = {u: check_opening(com1, served_t, op, u) for u, op in zip(units, ops)}
        why_vs_honest = {u: check_opening(com1, served1, op, u) for u, op in zip(units, ops)}
        opened[tag] = [unit_bcs(shape, com1, salt1, op) for op in ops]
        record.setdefault("openings", {})[tag] = {
            "registered": {k: served_t[k] for k in ("z_root", "receipt_root")},
            "refused": {str(u): w for u, w in why.items() if w},
            "against_honest_registration": {str(u): w for u, w in why_vs_honest.items() if w}}
        log(f"opening {tag}: {sum(1 for w in why.values() if w)} of {len(units)} refused against its registration, "
            f"{sum(1 for w in why_vs_honest.values() if w)} against the honest one")
    if record["openings"]["honest"]["refused"]:
        raise SystemExit(f"the honest openings are refused: {record['openings']['honest']['refused']}")

    # the second window, its first drawn unit's tile presented as window 1's
    com2, served2, salt2, _ch2 = epoch(work, w2, a.layer2, 2, a, record)
    if [c["name"] for c in com2["calls"]] != [c["name"] for c in com1["calls"]] or com2["tiles"] != T:
        raise SystemExit("the second window's calls are not the first's")
    prover("open", str(work / w2), salt2.hex(), str(u0), str(work / "open-other"))
    op2 = json.loads((work / "open-other" / "opening.json").read_text())["units"][0]
    record["openings"]["other-window"] = {"unit": u0, "refused": check_opening(com1, served1, op2, u0),
                                          "against_own_window": check_opening(com2, served2, op2, u0)}
    log(f"opening other-window: refused against window 1 for {len(record['openings']['other-window']['refused'])} rows")

    set_h, set_o = f"{w1}-draw", f"{w1}-u{u0}"
    stmts = {"honest": {"rows": str(work / "open-honest" / "rows.npz"), "set": set_h, "out": str(sw / "st-honest")},
             "other": {"rows": str(work / "open-other" / "rows.npz"), "set": set_o, "out": str(sw / "st-other")}}
    for name, req in stmts.items():
        (sw / "requests" / f"{name}.json").write_text(json.dumps(req))

    wait_for(sw / "circuit.json", "circuit built", a.stager_pid)
    head = json.loads((sw / "circuit.json").read_text())
    record["circuit"] = {x: head[x] for x in head if x != "meta"}
    vp = work / "verifier"
    vp.mkdir(exist_ok=True)
    files = {"honest": (set_h, opened["honest"]), "flipped-z": (set_h, opened["flipped-z"]),
             "wrong-receipt": (set_h, opened["wrong-receipt"]), "other-window": (set_o, [opened["honest"][0]]),
             "other-window-own": (set_o, [unit_bcs(shape, com2, salt2, op2)]),
             **{t: (set_h, opened[t]) for t in IN_PROOF}}
    for tag, (set_name, bcs) in files.items():
        (vp / f"{tag}.pub.bin").write_bytes(U.verifier_pub(head, set_name, head["subcircuit"], bcs))
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
    # the in-proof negatives: the honest statement with the first drawn unit's (instance 0's) output row the tampered
    # window's, under the verifier's own file of that window, so binding holds and the witness is wrong
    for t, (_f, port) in IN_PROOF.items():
        words = np.load(work / f"open-{t}" / "rows.npz")[f"w_{port}"][0]
        for d in (t, f"{t}{SKIP}"):
            record.setdefault("forged", {})[d] = U.forge_statement(sw / "st-honest", sw / f"st-{d}", (vp / f"{t}.pub.bin").read_bytes(),
                                                                   0, port, words, sw / "circuit.txt")
        log(f"forged {t}: {record['forged'][t]}")
    st = record["staged"]
    plan = {"statements": [{"tag": "honest", "dir": stmts["honest"]["out"], "n": len(units)},
                           {"tag": "other", "dir": stmts["other"]["out"], "n": 1},
                           *({"tag": d, "dir": str(sw / f"st-{d}"), "n": len(units)} for t in IN_PROOF for d in (t, f"{t}{SKIP}"))],
            "checks": [{"tag": "honest", "statement": "honest", "pub": str(vp / "honest.pub.bin"), "expect": "ACCEPT"},
                       {"tag": "flipped-z", "statement": "honest", "pub": str(vp / "flipped-z.pub.bin"), "expect": "REJECT"},
                       {"tag": "wrong-receipt", "statement": "honest", "pub": str(vp / "wrong-receipt.pub.bin"), "expect": "REJECT"},
                       {"tag": "other-window", "statement": "other", "pub": str(vp / "other-window.pub.bin"), "expect": "REJECT"},
                       {"tag": "other-window-own", "statement": "other", "pub": str(vp / "other-window-own.pub.bin"),
                        "expect": "ACCEPT"},
                       *({"tag": d, "statement": d, "pub": str(vp / f"{t}.pub.bin"), "expect": "REFUSED"}
                         for t in IN_PROOF for d in (t, f"{t}{SKIP}"))]}
    (work / "plan.json").write_text(json.dumps(plan, indent=1))
    (work / "record.json").write_text(json.dumps(record, indent=1))
    with open(work / "plan.tsv", "w") as f:
        for s in plan["statements"]:
            f.write(f"S\t{s['tag']}\t{s['dir']}\t{s['n']}\n")
        for c in plan["checks"]:
            f.write(f"C\t{c['tag']}\t{next(s['dir'] for s in plan['statements'] if s['tag'] == c['statement'])}\t{c['pub']}\n")
    log(f"planned: {len(plan['statements'])} statements, {len(plan['checks'])} Lean checks; staged in "
        f"{st['honest']['seconds']} s (honest, {st['honest']['instances']} instances), {st['other']['seconds']} s (other)")
    return 0


# -- summarize --------------------------------------------------------------------------------------------------------------

def _time(p: Path) -> dict | None:
    return json.loads(p.read_text()) if p.exists() else None


def _verdict(p: Path) -> dict | None:
    if not p.exists():
        return None
    v = [json.loads(x[8:]) for x in p.read_text().splitlines() if x.startswith("VERDICT ")]
    return v[0] if v else None


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
        proofs[s["tag"]] = {"instances": s["n"], "prove": t, "prove_out": po.read_text()[-1500:] if po.exists() else None,
                            "serve": _time(out / s["tag"] / "serve.time")}
    checks, ok = [], True
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
            good = word == c["expect"]
        ok &= good
        checks.append({"tag": c["tag"], "expect": c["expect"], "verdict": word, "pass": good,
                       "why": (v or {}).get("why", "")[:300], "lean_seconds": t and t["wall_s"],
                       "lean_maxrss_gb": t and round(t["maxrss_kb"] / 2**20, 1), **extra})
    honest = next(c for c in checks if c["tag"] == "honest")
    summary = {"schema": "verity/pouw/h1-prove/summary/v0", "pass": ok,
               "unit": record["circuit"]["definition"], "k": record["circuit"]["k"], "tile": [R, C],
               "ands": record["circuit"]["ands"], "k_log": record["circuit"]["k_log"],
               "circuit_bytes": record["circuit"]["circuit_bytes"], "circuit_build_seconds": record["circuit"]["build_seconds"],
               "window_tiles": record["draw"]["population"], "drawn_units": len(record["draw"]["units"]),
               "units": record["draw"]["units"], "statements": len(plan["statements"]),
               "statement_instances": {s["tag"]: s["n"] for s in plan["statements"]},
               "epsilon_at_delta": {f"{d}": round(1 - d ** (1 / record["draw_k"]), 4) for d in (0.5, 0.1, 0.01)},
               "proving_seconds": {k: (v["prove"] or {}).get("wall_s") for k, v in proofs.items()},
               "lean_verify_seconds": honest["lean_seconds"], "checks": checks, "proofs": proofs,
               "openings": record["openings"], "prover_pub_is_verifiers": record["prover_pub_is_verifiers"],
               "windows": {w: {k: v[k] for k in ("tiles", "calls", "commit_seconds", "serve_seconds") if k in v}
                           for w, v in record["windows"].items()},
               "staged": record["staged"]}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("pass", "unit", "k", "ands", "window_tiles", "drawn_units", "statements",
                                                "proving_seconds", "lean_verify_seconds")}))
    for c in checks:
        print(f"  {c['tag']:<20} expect {c['expect']:<7} got {c['verdict']:<8} {'ok' if c['pass'] else 'FAIL'}  "
              f"{c['lean_seconds']} s  {c['why'][:120]}")
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
    p = sub.add_parser("summarize")
    p.add_argument("--work", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--controls", default="")
    a = ap.parse_args(argv)
    return {"prepare": cmd_prepare, "summarize": cmd_summarize}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
