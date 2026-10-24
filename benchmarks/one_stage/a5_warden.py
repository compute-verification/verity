"""One-stage audit, run A5, on a transfer-gate stand-in: a window whose transfer-gate units open the warden's signed cells,
audited by `a5.audit` as rank 0 of a one-rank step through its `service.Stream`, on a local challenger's coins
(``--challenger local``) or the verifier's own (``--coins verifier``). No served window has transfer gates yet (the I/O
shell's ingress parse and egress render units have no C-Flock lowering, and no Commit was served behind the warden), so
this window stands in for one, its cells a warden run's (`warden_serve.py rehearse`, ``--run``).

* **The window** (`window`), under ``<ROOT>/rank0`` and the law ``work:2``: 2n units of the transfer-gate template
  `WardenCellIdentity{B}` (`warden_zk.py`'s toy, one cell's 8·B bits in and the same bits out) in two chunks, and the rest,
  ``--rest`` units of `WardenCellIdentity{2B}` over rows of their own (another template, so another stratum; at 2B its
  statement keeps C-Flock's m >= 25), standing in for the compiled model steps. The verifier's work table
  (`serving_rows.work_table_of`) proves every transfer-gate unit (work 0, floor all of them: selection All) beside the rest's
  drawn share, and a5's transfer-gate negatives (`a5.transfer_gate_negatives`) refuse each way of leaving one out.
  - chunk ``c0``, the ingress gates: unit i's ``cell`` is row i of connection q's ingress cells (`links`, data cells first),
    committed under the warden's salt for that row, so its ``b || c`` is the warden's;
  - chunk ``c1``, the egress gates: unit i's ``o0`` is row i of connection q's egress cells, under the warden's salts.
  Each chunk is a shared-row member (its ``cell`` rows a table, each unit its own row) whose header names its registered
  port's rows, each row's position in the link-window and its path to the warden's root (`warden_rows.Cells.statement`).
* **The verifier's reads** (`reads`): each link-window's registration read under the deployment's key and session and the
  schedule's grid and parameters (`warden_rows.check_registration`), and from it the verifier's own ``--registered`` file of
  each chunk (`warden_rows.port`: the head's root, the row each of the chunk's units opens), which a5 hands Lean's verify of
  each drawn chunk (`a5.audit`'s ``registered``). Which rows a unit opens (the connection's cells) is opened to the verifier as
  the units' wiring is: `warden_rows`' named gap. A transfer gate is addressed as (row, byte, bit): bit 8k + i of a unit's
  registered port is bit i of byte k of its row, row (t − first_bucket)·S + j slot j of bucket t (the registration's
  ``layout``), which the summary lists for each unit (``links``).
* **Provenance** (`source`): the registration's window names, for each link, the warden's value, root, certificate digest,
  the registration's SHA-512 and the rows its units open, and the verifier holds them to its own read.
* **Negatives**, beside a5's (the registered reads' three and the verdict reporting another root, on chunk c0; the
  transfer-gate negatives; R1 to R4 and the draw's): the prover's own root over the link-window's rows (one byte off) under
  the warden's certificate, refused by `check_registration`; and the window's provenance naming another root, refused by
  `source`. ``--altered``: unit 0 of c0 reads its cell one bit off under the warden's salt, its output the same bits (the
  circuit holds), so the run is to pass every check before the draw, draw the unit and be refused by Lean's registered reads.

The run exits 0 when it is as expected: accepted with every negative refused, or (``--altered``) refused by Lean's
registered reads. ``--out`` holds the window as committed (``window/``, its private files too) and a5's records
(``rank0/``, ``challenger/``, ``summary.json``); the warden's rows and salts stay in ``--run``.

    PYTHONPATH=<tree>:<tree>/benchmarks/numerical/python:<tree>/integrations/vllm \\
      python benchmarks/one_stage/a5_warden.py --run WARDEN_RUN --m0-pythonpath PATH --bin flock-circuit --lean flock-verify \\
        --firewall flock-firewall (--challenger local --main FILE | --coins verifier) [--altered] --out DIR
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import random
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "network_traces"))

import warden_serve as ws  # noqa: E402
from warden_zk import WardenCellIdentity, flip  # noqa: E402
from verity.ml.circuits.defs import bind  # noqa: E402
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2  # noqa: E402
from verity.core.protocols.certifier import warden_rows as WR  # noqa: E402
from verity.core.protocols.certifier.active import DATA  # noqa: E402
from verity.core.protocols.certifier.certificate import Certificate  # noqa: E402
from verity.core.protocols.certifier.warden import END, WardenParams  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import draw as D, partition as P  # noqa: E402
from verity_vllm.commit import serving_rows as CS  # noqa: E402
from verity_vllm.pipeline import serving_rows as SRW  # noqa: E402

#: the stand-in's session root, as a TP Commit's run root is its ranks' session root
ROOT = hashlib.sha512(b"verity/a5/warden-stand-in/v1").hexdigest()
RULE = "work:2"
#: each chunk's registered port, by the warden record it opens: the ingress cells are the circuit's input, the egress its output
LINKS = {ws.INGRESS: ("c0", "cell"), ws.EGRESS: ("c1", "o0")}


def links(run: Path, n: int) -> dict[str, SimpleNamespace]:
    """Each link's link-window of connection q, the lowest that has n cells each way in link-windows the warden registered:
    the registration as the verifier reads it, the warden's opening (the record's rows under its salts) held to its root, and
    the rows the n units open, data cells first."""
    dep = json.loads((run / "deployment.json").read_text())
    params = WardenParams(**dep["params"])
    key, session = ws.cs.read_public(dep["key"]), bytes.fromhex(dep["session"])
    records = ws.read_records(run, params)
    cells = {(d["registration"]["direction"], d["registration"]["window"]): (d["registration"], salts)
             for d, salts in ws.read_cells(run) if "registration" in d}
    found: dict[str, dict[int, tuple]] = {}
    for name in LINKS:
        link, direction = ws.CERTIFIED[name]
        found[name] = {}
        for record in records[name]:
            if (direction.value, record.window) in cells:
                for q, rows in WR.connection_rows(record, params.grid).items():
                    if len(rows) >= n:
                        found[name].setdefault(q, (record, rows))
    both = sorted(set.intersection(*(set(f) for f in found.values())))
    if not both:
        raise SystemExit(f"no connection of {run} has {n} cells each way in link-windows the warden registered")
    q, out = both[0], {}
    for name, f in found.items():
        link, direction = ws.CERTIFIED[name]
        record, rows = f[q]
        doc, salts = cells[direction.value, record.window]
        head, dom = WR.check_registration(doc, key, session, params.grid, params.digest, link, direction, record.window)
        opening = WR.Cells.commit(link, direction, record.window, WR.window_rows(record, params.grid), params.digest, session,
                                  salts)
        if opening.root != head.root:
            raise SystemExit(f"{head.value}: the warden's rows under its salts aren't its registered root")
        data = [p for p in rows if opening.rows[p][0] in (DATA, END)]
        out[name] = SimpleNamespace(params=params, key=key, session=session, link=link, direction=direction,
                                    window=record.window, connection=q, registration=doc, head=head, dom=dom, opening=opening,
                                    positions=(data + [p for p in rows if p not in data])[:n], data_rows=len(data))
    return out


def templates(B: int) -> tuple:
    """The transfer-gate template over a B-byte cell, and the rest's at twice the width."""
    return bind(WardenCellIdentity, B=B), bind(WardenCellIdentity, B=2 * B)


def window(at: dict[str, SimpleNamespace], n: int, rest: int) -> CS.Window:
    """The verifier's window: the transfer-gate template's two chunks of n units (ingress, egress), then the rest."""
    B = at[ws.INGRESS].params.cell_bytes
    tg, other = templates(B)
    program, query = CS.population_program([(tg, 2 * n), (other, rest)])
    (tid, _, _), (rid, _, _) = P.ranges(program, query)
    law = D.parse_law(RULE)

    def ports(words: int) -> dict:
        return {"rows": (("cell", words),), "outs": (("o0", words),), "schemas": (("cell", V2), ("o0", V2))}

    members = tuple(CS.Member(tid, n=n, base=j * n, tables=(("cell", n),), name=f"{tid}/{c}", **ports(B // 2))
                    for j, (c, _p) in enumerate(LINKS.values())) + (CS.Member(rid, n=rest, base=2 * n, **ports(B)),)
    return CS.Window(program, query, law, f"{ROOT}/rank0", members, work_table=SRW.work_table_of(law, {tid: 2 * n, rid: rest}, [tid]))


def chunks(mine: CS.Window) -> dict[str, CS.Member]:
    """Each link's transfer-gate chunk."""
    return {name: next(m for m in mine.members if m.name.endswith(f"/{c}")) for name, (c, _p) in LINKS.items()}


def provenance(at: dict[str, SimpleNamespace]) -> dict:
    """What the window's registration names of the warden's cells: per link, its value, root, certificate digest, the
    registration's SHA-512 and the rows the units open."""
    return {"stand-in": "benchmarks/one_stage/a5_warden.py", "connection": at[ws.INGRESS].connection,
            "links": {name: {"value": x.head.value, "root": x.head.root.hex(), "certificate": x.registration["certificate"]["digest"],
                             "registration_sha512": hashlib.sha512(json.dumps(x.registration, sort_keys=True).encode()).hexdigest(),
                             "rows": list(x.positions)} for name, x in at.items()}}


def rest_rows(m: CS.Member) -> np.ndarray:
    """The rest's rows: bytes of their own, not the warden's."""
    w = dict(m.rows)["cell"]
    return np.asarray([WR.words(hashlib.shake_256(b"a5-warden/rest/%d" % i).digest(2 * w)) for i in range(m.n)], dtype=np.uint16)


def commit(out: Path, mine: CS.Window, at: dict[str, SimpleNamespace], altered: bool = False) -> CS.Committed:
    """The window committed into ``out`` as serving would commit it: each chunk's registered port's rows under the warden's
    salts and its header naming their positions and paths (``altered``: unit 0 of c0 reads its cell one bit off), the rest
    under fresh salts."""
    gates, data, salts_of, registered, entries = chunks(mine), [], {}, {}, []
    for m in mine.members:
        name = next((x for x, g in gates.items() if g is m), None)
        if name is None:
            rows = rest_rows(m)
            data.append({"rows": {"cell": rows}, "outs": {"o0": rows}})
            entries.append([[m.base, m.base + m.n, "rest", 0, 0, "rest", m.key, 0, 0]])
            continue
        x, port = at[name], LINKS[name][1]
        _words, salts, entry = x.opening.statement(x.positions)
        cells = [x.opening.rows[p] for p in x.positions]
        if altered and name == ws.INGRESS:
            cells[0] = flip(cells[0], bit=1)
        rows = np.asarray([WR.words(r) for r in cells], dtype=np.uint16)
        data.append({"rows": {"cell": rows}, "outs": {"o0": rows}, "refs": np.arange(m.n)[:, None]})
        salts_of[m.port(port)] = b"".join(salts)
        registered[m.key] = {port: entry}
        entries.append([[m.base, m.base + m.n, f"connection-{x.connection}", 0, 0, x.head.value, m.key, 0, 0]])
    c = CS.commit(mine, data, salts_of=salts_of)
    CS.write(c, out, {"of": provenance(at), "committer": {"module": "a5_warden.py"}}, CS.index_doc(mine.members, entries, mine.run),
             kind="stand-in", registered=registered)
    return c


def reads(mine: CS.Window, at: dict[str, SimpleNamespace]) -> dict[str, dict]:
    """The verifier's own ``--registered`` file of each chunk: its registered port at the root the warden signed, each unit
    (from the chunk's first) opening its row."""
    return {m.key: WR.reads_file({LINKS[name][1]: WR.port(at[name].head, at[name].positions, m.base)})
            for name, m in chunks(mine).items()}


def source_of(mine: CS.Window, at: dict[str, SimpleNamespace]):
    """The verifier's check of the registration's window: a stand-in of its run naming the warden's cells as it read them."""
    want = provenance(at)

    def source(w: dict) -> dict:
        ok = {"kind": w.get("kind") == "stand-in", "run": w.get("run") == mine.run, "of": w.get("of") == want}
        return {"ok": all(ok.values()), "codes": [c for c, v in ok.items() if not v], "detail": {"kind": w.get("kind"), "run": w.get("run")}}
    return source


def gate_rows(x: SimpleNamespace) -> list[dict]:
    """Each unit's row and its (bucket, slot), the address of its transfer gates: bit 8k + i of the unit's registered port is
    bit i of the row's byte k."""
    lay = x.registration["layout"]
    return [{"row": r, "bucket": lay["first_bucket"] + r // lay["slots"], "slot": r % lay["slots"]} for r in x.positions]


def negatives(at: dict[str, SimpleNamespace], source, rec: dict) -> dict[str, dict]:
    """The prover's own root over the ingress link-window's rows (one byte off) under the warden's certificate, and the window's
    provenance naming another root."""
    x = at[ws.INGRESS]
    rows = list(x.opening.rows)
    rows[x.positions[0]] = flip(rows[x.positions[0]])
    own = WR.Cells.commit(x.link, x.direction, x.window, rows, x.opening.parameters, x.session)
    c = x.registration["certificate"]
    forged = WR.registration(own.head, Certificate(bytes.fromhex(c["key"]), x.link, x.direction, x.window,
                                                   bytes.fromhex(c["digest"]), bytes.fromhex(c["signature"])), x.params.grid)
    try:
        WR.check_registration(forged, x.key, x.session, x.params.grid, x.params.digest, x.link, x.direction, x.window)
        why = None
    except ValueError as e:
        why = str(e)
    w = rec["window"]
    of = json.loads(json.dumps(w["of"]))
    of["links"][ws.INGRESS]["root"] = own.head.root.hex()
    s = source({**w, "of": of})
    return {"prover-root-under-the-wardens-certificate (check_registration)": {"refused": why is not None, "why": why},
            "window-naming-another-warden-root (source)": {"refused": not s["ok"], "codes": s["codes"]}}


def composer(defn, bits: int):
    """``a5.audit``'s composer of the template's circuit: C-Flock's `UnitStager` over its one ``cell`` row in and ``o0`` out,
    bound to the window's partition and program."""
    def compose(circuit: Path, partition: str, program: str) -> dict:
        from verity_flock.class_statement import UnitStager

        st = UnitStager(defn, "unit/a5-warden", [("cell", V2, bits, ["cell"])], [(V2, (0,))], circuit=circuit,
                        bind={"partition": partition, "program": program})
        return {"subcircuit": st.meta.get("subcircuit"), "pin": st.pin, "class_sha512": st.class_sha512, "k_log": st.meta["k_log"]}
    return compose


def free_ports(span: int) -> int:
    """A base port whose ``span`` loopback ports are all free now, so two audits on one node don't share a session's port."""
    bases = list(range(20000, 60000 - span, span))
    random.SystemRandom().shuffle(bases)
    for base in bases:
        try:
            with contextlib.ExitStack() as held:
                for port in range(base, base + span):
                    held.enter_context(socket.socket()).bind(("127.0.0.1", port))
            return base
        except OSError:
            continue
    raise SystemExit(f"no {span} free loopback ports")


def expected(a: argparse.Namespace, s: dict, ok: bool) -> bool:
    """Honest: accepted with every negative refused. ``--altered``: every check before the draw passed, and Lean's verify
    of chunk c0 refused at its registered reads, so the audit isn't accepted."""
    if not a.altered:
        return ok
    why = ((s.get("audit") or {}).get("why") or {}).get("lean-verify/m0") or ""
    return all((s.get("checks_before_draw") or {}).values()) and (s.get("audit") or {}).get("accepted") is False \
        and "registered reads" in why


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="a5_warden.py", description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, type=Path, help="a warden-serve run's directory (warden_serve.py rehearse)")
    ap.add_argument("--units", type=int, default=16, help="n: each transfer-gate chunk's units")
    ap.add_argument("--rest", type=int, default=8, help="the rest's units, drawn by the work law")
    ap.add_argument("--m0-pythonpath", required=True)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--lean", required=True)
    ap.add_argument("--firewall", required=True)
    ap.add_argument("--challenger", choices=["local"], help="draw on a local challenger session's coins")
    ap.add_argument("--coins", choices=["verifier"], help="draw on the verifier's own live coins")
    ap.add_argument("--main", help="the verifier's own `git rev-list --first-parent origin/main` (--challenger)")
    ap.add_argument("--altered", action="store_true", help="unit 0 of the ingress chunk reads its cell one bit off")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--delta", type=float, default=0.5)
    ap.add_argument("--port", type=int, default=0, help="the sessions' first port (0: a free block)")
    ap.add_argument("--session-timeout", type=float, default=3600)
    a = ap.parse_args(argv)
    if bool(a.challenger) == bool(a.coins):
        ap.error("draw on a local challenger's coins (--challenger local --main FILE) or on the verifier's own (--coins verifier)")
    if a.challenger and not a.main:
        ap.error("--challenger local needs --main")
    import a5
    import a5_coins as AC

    out = a.out
    at = links(a.run, a.units)
    mine = window(at, a.units, a.rest)
    commit(out / "window", mine, at, a.altered)
    source = source_of(mine, at)
    rec = json.loads((out / "window" / "registration.json").read_text())
    B = at[ws.INGRESS].params.cell_bytes
    tg, rid = mine.members[0].template, mine.members[-1].template
    sets = {t: (composer(d, bits), d) for t, d, bits in zip((tg, rid), templates(B), (8 * B, 16 * B))}
    a.rule, a.window_root, a.compose_cache, a.transfer_gate = RULE, ROOT, None, [tg]
    a.verifier_coins = a.coins == "verifier"
    port = a.port or free_ports(a5.RANK_PORTS)
    summary0 = {"rank": 0, "standin": provenance(at), "altered": a.altered}
    if a.challenger:
        manager, a.coins, finisher = AC.start("local", ROOT, [0], out / "challenger", a.main)
    else:
        a.coins = None
    published: dict = {}
    try:
        s, ok = a5.audit(a, sets, out / "rank0", mine, out / "window", source, port, summary0,
                         more_negatives=negatives(at, source, rec), rank=0, registered=reads(mine, at))
    finally:
        if a.coins is not None:
            a.coins.skip(0)
            finisher.join()
            published = a.coins.evidence()
            manager.shutdown()
    summary = {"root": ROOT, "coins": "challenger" if a.challenger else "verifier", "altered": a.altered,
               "session": (published.get("plan") or {}).get("session"),
               "window": {"population": mine.population, "transfer_gates": [tg], "work_table": mine.work_table,
                          "members": [{"key": m.key, "template": m.template, "n": m.n, "base": m.base} for m in mine.members]},
               "links": {name: {"value": x.head.value, "root": x.head.root.hex(), "connection": x.connection,
                                "data_rows": x.data_rows, "chunk": chunks(mine)[name].key, "port": LINKS[name][1],
                                "cell_bytes": B, "gates": gate_rows(x)} for name, x in at.items()},
               "challenger": {x: published.get(x) for x in ("trusted", "error")},
               "rank0": {x: s.get(x) for x in ("checks_before_draw", "refused", "audit", "draw", "draw_counts", "negatives", "coins")},
               "accepted": ok and not published.get("error")}
    summary["as_expected"] = expected(a, s, summary["accepted"])
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    print("A5-WARDEN " + json.dumps({x: summary[x] for x in ("accepted", "as_expected", "altered", "coins")}
                                    | {"draw": s.get("draw"), "audit": (s.get("audit") or {}).get("why")}, sort_keys=True, default=str))
    return 0 if summary["as_expected"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
