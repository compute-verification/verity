"""A window's audit from its records alone: what an integration hands Verity (`verity.ml.windows.interface`), and nothing else.

The records, every one in the window's directory but the advice's opening:

* **the circuit** in proof units with its public layout (``circuit.json``), which the registration binds by SHA-512;
* **the compiler and the advice**, when the circuit depends on run-time choices: the registration's window names the compiler
  (``compiler``: its name and digest, which carries the public cap on the advice's length) and the advice's hiding commitment
  (``advice``: its root and its length in bits, which is charged); the advice is a bit string the audit never interprets;
* **the values**, committed before any coin: the registration (``registration.json``) and the files `rows.write` writes;
* **on a draw, the drawn units' openings** (``m<k>-private.bin``), which reach only the prover behind the firewall.

The verifier's window is `interface.of_records`'s, each template resolved in Verity's catalog (`interface.resolve`), or, for a
template the verifier lowers from its own input set (``--template-sets``), that set's Definition. Before the draw the audit
holds the registration to that circuit, to the run root the run published (``--run``, default the circuit record's) and,
for a compiled circuit, to the compiler and advice it was compiled from (`source`), and runs the compile check
(`compile_check`, the named gap until the compile unit exists): the advice's opening opens the registered root under the
registered compiler, and that compiler, run as the command the verifier's own table names for it (``--compilers``), makes
this circuit from the advice and charges the registered length. A circuit with no compiler is the one the verifier
registered (``--circuit``). Then `a5.audit`: the registration checked by Lean and Python, the draw on live coins read after
the receipt, C-Flock ``--zk`` behind the Lean firewall on the drawn members, Lean's verdicts and the audit record. Then the
advice was fixed before the draw (`fixed_before_draw`), and advice re-committed after the coins is refused (`after_draw`).
An integration's labels are never read.

    records_audit.py audit --window DIR --law L (--opening FILE --compilers FILE | --circuit SHA512) [--run HEX] \\
        [--template-sets DIR] --m0-pythonpath PATH --bin flock-circuit --lean flock-verify --firewall flock-firewall \\
        --out DIR [--delta D] [--port P] [--forged] [--challenger local --main FILE]

``--compilers FILE``: ``{name: {"digest": hex, "argv": [...]}}``, the registered compilers the verifier holds. A compiler's
command takes ``--advice FILE --out FILE``, writes the circuit record it compiles to ``--out`` (its run's own fields,
`interface.RUN_FIELDS`, aside) and prints ``{"bits": N}``, the advice's length it charges, as its last line; it refuses
advice it can't compile, or advice over its cap, with a nonzero exit. ``--forged``: the window holds a forged unit, and the
audit passes when every check before the draw passed and Lean refused the outcome (the prover runs with its self-check
skipped, ``FC_ZK_SELF_CHECK=skip``, so it sends proofs of the false witness).
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import random
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "one_stage"))

from verity.core.protocols.sampled_proofs.one_stage import draw as D, registration as R  # noqa: E402
from verity.ml.windows import interface as IF  # noqa: E402

#: where an audit's ports start when ``--port`` isn't given (`free_ports`): below Linux's ephemeral range (32768-60999), from
#: which outbound connections take theirs
PORTS = range(10000, 32000)


def free_ports(span: int, bases: Iterable[int] | None = None) -> int:
    """A base port whose ``span`` loopback ports are all free now, the first of ``bases`` (default every base of `PORTS` in
    random order). An audit's sessions listen at fixed offsets from it (``a5.RANK_PORTS`` a rank); a fixed default made two
    audits on one node collide, the second's verifier failing to bind and its prover reaching the first's, which refused it."""
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


def compile_check(circuit: Mapping[str, Any], window: Mapping[str, Any], opening: Any, compilers: Mapping[str, Any],
                  scratch: Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """The compile check: the registration's compiler is one the verifier holds, under the digest it holds (``compiler``); the
    opening opens the registered advice root under that compiler (``advice``); the compiler's command on the opened advice
    exits 0 (``compiles``), the circuit record it writes is ``circuit`` less its run's fields (``circuit``) and the length it
    charges is the registered one (``length``). The command runs only on advice that opens. -> (the verdict, the compiler and
    advice it verified, or None)."""
    c, adv = window.get("compiler"), window.get("advice")
    c, adv = (c if isinstance(c, Mapping) else {}), (adv if isinstance(adv, Mapping) else {})
    entry = compilers.get(c.get("name")) if isinstance(c.get("name"), str) else None
    ok = {"compiler": entry is not None and entry.get("digest") == c.get("digest"), "advice": False, "compiles": False,
          "circuit": False, "length": False}
    detail: dict[str, Any] = {"compiler": dict(c), "advice": dict(adv)}
    try:
        data, root = IF.opened_advice(c, opening)
        ok["advice"] = root == adv.get("root")
    except (ValueError, KeyError, TypeError) as e:
        data, detail["opening"] = None, str(e)
    if ok["compiler"] and ok["advice"]:
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as d:   # the opened advice, the verifier's under the named gap
            (Path(d) / "advice.bin").write_bytes(data)
            t0 = time.time()
            p = subprocess.run([*entry["argv"], "--advice", str(Path(d) / "advice.bin"), "--out", str(scratch / "compiled.json")],
                               capture_output=True, text=True)
        detail.update(compile_s=round(time.time() - t0, 1), rc=p.returncode, stderr=p.stderr[-2000:])
        ok["compiles"] = p.returncode == 0
        if ok["compiles"]:
            theirs = json.loads((scratch / "compiled.json").read_text())
            ok["circuit"] = IF.circuit_of(theirs) == IF.circuit_of(circuit)
            bits = json.loads(p.stdout.strip().splitlines()[-1]).get("bits")
            ok["length"] = isinstance(bits, int) and not isinstance(bits, bool) and bits == adv.get("bits")
            detail["bits"] = bits
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


def negatives(w: Mapping[str, Any], held: Mapping[str, Any], circuit: Mapping[str, Any], scratch: Path) -> dict[str, dict]:
    """What the audit refuses before any draw about the records: a registration naming another circuit or another run root,
    one binding a compiler for a circuit registered with none; for a compiled circuit, a registration naming another
    compiler, another advice root or another length, or binding neither (a circuit passed off as the compiler's), the
    advice's opening under another salt (which opens no registered root), and a circuit record other than the compiler's
    output (``held["output"]``, the compile check's, its first member one unit short)."""
    cases = {"registration-naming-another-circuit": {**w, "circuit": "00" * 64},
             "registration-naming-another-run": {**w, "run": "00" * 64}}
    if held.get("compiler") is None:
        cases["registration-binding-a-compiler-for-a-fixed-circuit"] = {**w, "compiler": {"name": "c", "digest": "00" * 64}}
    else:
        cases |= {"registration-naming-another-compiler": {**w, "compiler": {**w["compiler"], "digest": "00" * 64}},
                  "registration-naming-another-advice-root": {**w, "advice": {**w["advice"], "root": "00" * 64}},
                  "registration-charging-another-length": {**w, "advice": {**w["advice"], "bits": w["advice"]["bits"] + 1}},
                  "registration-binding-no-compiler-or-advice": {k: v for k, v in w.items() if k not in ("compiler", "advice")}}
    out = {name: source(x, held) for name, x in cases.items()}
    if held.get("compiler") is not None:
        op = dict(held["opening"])
        op["salts"] = [f"{int(op['salts'][0][:2], 16) ^ 1:02x}{op['salts'][0][2:]}", *op["salts"][1:]]
        v, _ = compile_check(circuit, w, op, held["compilers"], scratch)
        out["opening-under-another-salt"] = v
        if (theirs := held.get("output")) is not None:
            m0 = theirs["members"][0]
            short = {**theirs, "members": [{**m0, "n": m0["n"] - 1}, *theirs["members"][1:]]}
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
    input set, whose Definition it resolves to; any other template from Verity's catalog (`interface.resolve`)."""
    import a5 as A5

    own = A5.template_sets(template_sets) if template_sets else {}
    resolve = lambda t: own[t][1] if t in own else IF.resolve(t)  # noqa: E731
    mine, doc = IF.of_records(window_dir, law, resolve)
    sets = {**own, **{t: (composer(resolve(t), st), resolve(t)) for t, st in doc["statements"].items()}}
    return mine, doc, sets


def _code(fn: Any) -> str | None:
    """The module a Definition's code is in: where the summary says each template's Definition came from."""
    d = getattr(fn, "defn", fn)
    return getattr(getattr(d, "body", None) or getattr(d, "evaluate", None), "__module__", None)


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
    except (ValueError, KeyError, TypeError) as e:
        row = {"ok": False, "refused": {"records": f"{type(e).__name__}: {e}"}, "seconds": round(time.time() - t0, 1)}
        (out / "summary.json").write_text(json.dumps(row, indent=1, sort_keys=True))
        print("RECORDS " + json.dumps(row, sort_keys=True))
        return 1
    rec = json.loads((wdir / "registration.json").read_text())
    w = rec.get("window") or {}
    body = (wdir / "circuit.json").read_bytes()
    held: dict[str, Any] = {"run": a.run or mine.run, "circuit": a.circuit or hashlib.sha512(body).hexdigest(),
                            "compiled": True, "compiler": None, "advice": None}
    compiled = None
    if a.opening:
        compilers, opening = json.loads(Path(a.compilers).read_text()), json.loads(Path(a.opening).read_text())
        compiled, bound = compile_check(doc, w, opening, compilers, out / "compile")
        output = out / "compile" / "compiled.json"
        held.update(compiled=compiled["ok"], compiler=(bound or {}).get("compiler") or w.get("compiler"),
                    advice=(bound or {}).get("advice") or w.get("advice"), compilers=compilers, opening=opening,
                    output=json.loads(output.read_text()) if compiled["ok"] else None)
    window_s = round(time.time() - t0, 2)
    more = negatives(w, held, doc, out / "negatives")
    a.port = a.port if a.port is not None else free_ports(A5.RANK_PORTS)
    a.coins, a.verifier_coins, a.window_root, a.rule = None, not a.challenger, mine.run, a.law
    a.compose_cache = None
    if a.forged:
        os.environ["FC_ZK_SELF_CHECK"] = "skip"
    if a.challenger:
        manager, a.coins, finisher = AC.start(a.challenger, mine.run, [0], out / "challenger", a.main)
    try:
        s, ok = A5.audit(a, sets, out / "window", mine, wdir, lambda x: source(x, held), a.port,
                         {"verifier_window_s": window_s, "circuit": held["circuit"], **({"compile": compiled} if compiled else {})},
                         more, rank=0)
    finally:
        if a.coins is not None:
            a.coins.skip(0)
            finisher.join()
            published = a.coins.evidence()
            manager.shutdown()
    row = {"seconds": round(time.time() - t0, 1), "verifier_window_s": window_s, "circuit": held["circuit"],
           "compiler": held["compiler"], "advice": held["advice"], "compile": compiled, "law": law,
           **{x: s.get(x) for x in ("run_root", "population", "members", "registration", "source", "refused", "checks_before_draw",
                                    "audit", "draw", "draw_counts", "negatives", "coins", "timing")}}
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
    if a.challenger:
        row["ok"] = row["ok"] and not row["coins_session"]["error"]
    (out / "summary.json").write_text(json.dumps(row, indent=1, sort_keys=True, default=str))
    print("RECORDS " + json.dumps({x: row.get(x) for x in ("ok", "circuit", "population", "draw", "draw_counts", "seconds")}
                                  | {"compile": (compiled or {}).get("ok"), "fixed_before_draw": (row.get("fixed_before_draw") or {}).get("ok"),
                                     "accepted": (s.get("audit") or {}).get("accepted"),
                                     "negatives_refused": all((row.get("negatives") or {"none": False}).values()),
                                     "definitions": row["definitions"], "loaded": row["loaded"],
                                     **({"forged": row["forged"]} if a.forged else {})}, sort_keys=True, default=str))
    return 0 if row["ok"] else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("audit", help="a window's audit from its records alone")
    p.add_argument("--window", required=True, help="the window's directory: circuit.json, registration.json, the value files")
    p.add_argument("--law", required=True, help="the proof service's law of the window (subset:K, stratified:K, work:K)")
    p.add_argument("--opening", help="the advice's opening (interface.advice_opening): the named gap, until the compile unit")
    p.add_argument("--compilers", help="with --opening: the registered compilers the verifier holds, {name: {digest, argv}}")
    p.add_argument("--circuit", help="instead: the SHA-512 of the circuit record the verifier registered, a circuit with no compiler")
    p.add_argument("--run", help="the run root the verifier holds the window to, the one the run published (default the "
                   "circuit record's)")
    p.add_argument("--template-sets", help="the verifier's own input sets, for templates whose units are not bit rows")
    p.add_argument("--m0-pythonpath", required=True)
    p.add_argument("--bin", required=True)
    p.add_argument("--lean", required=True)
    p.add_argument("--firewall", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--delta", type=float, default=0.5)
    p.add_argument("--port", type=int, default=None, help="the sessions' base port (default a base whose ports are free now)")
    p.add_argument("--session-timeout", type=float, default=14400)
    p.add_argument("--forged", action="store_true", help="the window holds a forged unit: it must be refused")
    p.add_argument("--challenger", choices=["local"], help="draw on the auditor's challenger's coins (a5_coins), not the verifier's")
    p.add_argument("--main", help="with --challenger: the verifier's copy of main's first-parent history")
    a = ap.parse_args(argv)
    if bool(a.opening) == bool(a.circuit) or bool(a.opening) != bool(a.compilers):
        ap.error("a compiled circuit: --opening FILE --compilers FILE; a circuit with no compiler: --circuit SHA512")
    if a.challenger and not a.main:
        ap.error("--challenger needs --main")
    return audit(a)


if __name__ == "__main__":
    raise SystemExit(main())
