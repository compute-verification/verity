"""One-stage audit, run A5, on a transfer-gate stand-in: a window whose transfer-gate units open the warden's signed cells,
audited by `a5.audit` as rank 0 of a one-rank step through its `service.Stream`, on a local challenger's coins
(``--challenger local``) or the verifier's own (``--coins verifier``). No served window has transfer gates yet (the I/O
shell's ingress parse and egress render units have no C-Flock lowering, and no Commit was served behind the warden), so
this window stands in for one, its cells a warden run's (`warden_serve.py rehearse`, ``--run``).

* **The rows** are the warden's, one a bucket (`warden_rows`, registration v2): row t − first_bucket of a link-window is
  bucket t, S = lanes·r slots of B bytes, slot j at bytes j·B (words j·B/2), a connection's cells at the slots its
  wiring gives it, a slot no connection holds B zero bytes.
* **The window** (`window`), under ``<ROOT>/rank0`` and the law ``work:2``: 2n units of the transfer-gate template
  `WardenRowIdentity{S·B}` (`warden_zk.py`'s toy, one bucket row's 8·S·B bits in and the same bits out) in two chunks, and
  the rest, ``--rest`` units of `WardenRowIdentity{2B}` over rows of their own (another template, so another stratum),
  standing in for the compiled model steps. The verifier's work table (`serving_rows.work_table_of`) proves every
  transfer-gate unit (work 0, floor all of them: selection All) beside the rest's drawn share, and a5's transfer-gate
  negatives (`a5.transfer_gate_negatives`) refuse each way of leaving one out.
  - chunk ``c0``, the ingress gates: unit i reads connection q's ingress cell k_i (`links`, data cells first), its ``row``
    the bucket row that cell is in, committed under the warden's salt for that row, so its ``b || c`` is the warden's;
  - chunk ``c1``, the egress gates: unit i's ``o0`` is the bucket row of connection q's egress cell k_i, under the warden's
    salts.
  r of a connection's cells share a row, so r units open each row. Each chunk is a shared-row member (its ``row`` rows a
  table, each unit its own row) whose header names its registered port's rows, each row's position in the link-window
  and its path to the warden's root (`warden_rows.Cells.statement`).
* **The verifier's reads** (`reads`): each link-window's registration read under the deployment's key and session and the
  schedule (`warden_rows.check_registration`), and from it the verifier's own ``--registered`` file of each chunk
  (`warden_rows.port`: the head's root, and the row of each unit's read (q, k_i), placed by the wiring the warden signed),
  which a5 hands Lean's verify of each drawn chunk (`a5.audit`'s ``registered``). The wiring is opened to the verifier as
  the units' wiring is: `warden_rows`' named gap. A transfer gate is addressed as (row, byte, bit): bit 8k + i of a unit's
  registered port is bit i of byte k of its row, row t − first_bucket bucket t (the registration's ``layout``), and the
  unit's own cell is its slot's bytes, which the summary lists for each unit (``links``: the cell, row, bucket, slot and
  words).
* **Provenance** (`source`): the registration's window names, for each link, the warden's value, root, certificate digest,
  the registration's SHA-512, and the reads and rows its units open, and the verifier holds them to its own read.
* **Negatives**, beside a5's (the registered reads' three and the verdict reporting another root, on chunk c0; the
  transfer-gate negatives; R1 to R4 and the draw's): the prover's own root over the link-window's rows (one byte off) under
  the warden's certificate, refused by `check_registration`; the window's provenance naming another root, refused by
  `source`; and the reads no honest read is, on the ingress link-window, each refused by `warden_rows.port` by name: a
  foreign read (another connection's cell, under a wiring that drops the connection holding its slot), a duplicate read,
  a padding-slot read (of a connection holding no slot, and under a wiring that puts one in empty slots) and a read past
  the connection's cells. ``--altered``: unit 0 of c0 reads its row one bit off in its cell, under the warden's salt,
  its output the same bits (the circuit holds), so the run is to pass every check before the draw, draw the unit and be
  refused by Lean's registered reads.

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
from warden_zk import WardenRowIdentity, flip  # noqa: E402
from verity.ml.circuits.defs import bind  # noqa: E402
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2  # noqa: E402
from verity.core.protocols.certifier import warden_rows as WR  # noqa: E402
from verity.core.protocols.certifier.active import DATA, HEADER  # noqa: E402
from verity.core.protocols.certifier.certificate import Certificate  # noqa: E402
from verity.core.protocols.certifier.warden import END, WardenParams  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import draw as D, partition as P  # noqa: E402
from verity_vllm.commit import serving_rows as CS  # noqa: E402
from verity_vllm.pipeline import serving_rows as SRW  # noqa: E402

#: the stand-in's session root, as a TP Commit's run root is its ranks' session root
ROOT = hashlib.sha512(b"verity/a5/warden-stand-in/v1").hexdigest()
RULE = "work:2"
#: each chunk's registered port, by the warden record it opens: the ingress cells are the circuit's input, the egress its output
LINKS = {ws.INGRESS: ("c0", "row"), ws.EGRESS: ("c1", "o0")}


def links(run: Path, n: int) -> dict[str, SimpleNamespace]:
    """Each link's link-window of connection q, the lowest that has n cells each way in link-windows the warden registered:
    the registration as the verifier reads it, the warden's opening (the record's rows and wiring under its salts) held to
    its head, the n units' reads (q, k), data cells first, and the row and slot the warden's wiring places each at."""
    dep = json.loads((run / "deployment.json").read_text())
    params = WardenParams(**dep["params"])
    key, session = ws.cs.read_public(dep["key"]), bytes.fromhex(dep["session"])
    records = ws.read_records(run, params)
    cells = {(d["registration"]["direction"], d["registration"]["window"]): (d["registration"], salts, wiring_salt)
             for d, salts, wiring_salt in ws.read_cells(run) if "registration" in d}
    found: dict[str, dict[int, tuple]] = {}
    for name in LINKS:
        link, direction = ws.CERTIFIED[name]
        found[name] = {}
        for record in records[name]:
            if (direction.value, record.window) in cells:
                for q, _first, k in WR.wiring(record, params).connections:
                    if k >= n:
                        found[name].setdefault(q, (record, k))
    both = sorted(set.intersection(*(set(f) for f in found.values())))
    if not both:
        raise SystemExit(f"no connection of {run} has {n} cells each way in link-windows the warden registered")
    q, out, B = both[0], {}, params.cell_bytes
    for name, f in found.items():
        link, direction = ws.CERTIFIED[name]
        record, k = f[q]
        doc, salts, wiring_salt = cells[direction.value, record.window]
        head, dom = WR.check_registration(doc, key, session, params, link, direction, record.window)
        opening = WR.Cells.of(link, direction, record, params, session, salts, wiring_salt)
        if opening.head != head:
            raise SystemExit(f"{head.value}: the warden's rows and wiring under its salts aren't its registered head")
        kinds = [opening.rows[t][j * B] for t, j in opening.wiring.place(params, [(q, i) for i in range(k)])]
        data = [i for i in range(k) if kinds[i] in (DATA, END)]
        reads = [(q, i) for i in (data + [i for i in range(k) if i not in data])[:n]]
        placed = opening.wiring.place(params, reads)
        out[name] = SimpleNamespace(params=params, key=key, session=session, link=link, direction=direction,
                                    window=record.window, connection=q, registration=doc, head=head, dom=dom, opening=opening,
                                    reads=reads, positions=[t for t, _ in placed], slots=[j for _, j in placed],
                                    data_cells=len(data))
    return out


def templates(params: WardenParams) -> tuple:
    """The transfer-gate template over one bucket row's S·B bytes, and the rest's over 2B bytes."""
    tg, rest = WR.row_bytes(params), 2 * params.cell_bytes
    if tg == rest:
        raise SystemExit(f"a row of {tg} bytes is the rest's width: the rest needs a template of its own")
    return bind(WardenRowIdentity, B=tg), bind(WardenRowIdentity, B=rest)


def window(at: dict[str, SimpleNamespace], n: int, rest: int) -> CS.Window:
    """The verifier's window: the transfer-gate template's two chunks of n units (ingress, egress), then the rest."""
    params = at[ws.INGRESS].params
    tg, other = templates(params)
    program, query = CS.population_program([(tg, 2 * n), (other, rest)])
    (tid, _, _), (rid, _, _) = P.ranges(program, query)
    law = D.parse_law(RULE)

    def ports(words: int) -> dict:
        return {"rows": (("row", words),), "outs": (("o0", words),), "schemas": (("row", V2), ("o0", V2))}

    members = tuple(CS.Member(tid, n=n, base=j * n, tables=(("row", n),), name=f"{tid}/{c}", **ports(WR.row_bytes(params) // 2))
                    for j, (c, _p) in enumerate(LINKS.values())) + (CS.Member(rid, n=rest, base=2 * n, **ports(params.cell_bytes)),)
    return CS.Window(program, query, law, f"{ROOT}/rank0", members, work_table=SRW.work_table_of(law, {tid: 2 * n, rid: rest}, [tid]))


def chunks(mine: CS.Window) -> dict[str, CS.Member]:
    """Each link's transfer-gate chunk."""
    return {name: next(m for m in mine.members if m.name.endswith(f"/{c}")) for name, (c, _p) in LINKS.items()}


def provenance(at: dict[str, SimpleNamespace]) -> dict:
    """What the window's registration names of the warden's cells: per link, its value, root, certificate digest, the
    registration's SHA-512, and the units' reads (q, k) and the rows they open."""
    return {"stand-in": "benchmarks/one_stage/a5_warden.py", "connection": at[ws.INGRESS].connection,
            "links": {name: {"value": x.head.value, "root": x.head.root.hex(), "certificate": x.registration["certificate"]["digest"],
                             "registration_sha512": hashlib.sha512(json.dumps(x.registration, sort_keys=True).encode()).hexdigest(),
                             "reads": [list(r) for r in x.reads], "rows": list(x.positions)} for name, x in at.items()}}


def rest_rows(m: CS.Member) -> np.ndarray:
    """The rest's rows: bytes of their own, not the warden's."""
    w = dict(m.rows)["row"]
    return np.asarray([WR.words(hashlib.shake_256(b"a5-warden/rest/%d" % i).digest(2 * w)) for i in range(m.n)], dtype=np.uint16)


def commit(out: Path, mine: CS.Window, at: dict[str, SimpleNamespace], altered: bool = False) -> CS.Committed:
    """The window committed into ``out`` as serving would commit it: each chunk's registered port's rows under the warden's
    salts and its header naming their positions and paths (``altered``: unit 0 of c0 reads its row one bit off, in its
    cell's last byte), the rest under fresh salts."""
    gates, data, salts_of, registered, entries = chunks(mine), [], {}, {}, []
    for m in mine.members:
        name = next((x for x, g in gates.items() if g is m), None)
        if name is None:
            rows = rest_rows(m)
            data.append({"rows": {"row": rows}, "outs": {"o0": rows}})
            entries.append([[m.base, m.base + m.n, "rest", 0, 0, "rest", m.key, 0, 0]])
            continue
        x, port = at[name], LINKS[name][1]
        _words, salts, entry = x.opening.statement(x.positions)
        unit_rows = [x.opening.rows[p] for p in x.positions]
        if altered and name == ws.INGRESS:
            unit_rows[0] = flip(unit_rows[0], byte=(x.slots[0] + 1) * x.params.cell_bytes - 1, bit=1)
        rows = np.asarray([WR.words(r) for r in unit_rows], dtype=np.uint16)
        data.append({"rows": {"row": rows}, "outs": {"o0": rows}, "refs": np.arange(m.n)[:, None]})
        salts_of[m.port(port)] = b"".join(salts)
        registered[m.key] = {port: entry}
        entries.append([[m.base, m.base + m.n, f"connection-{x.connection}", 0, 0, x.head.value, m.key, 0, 0]])
    c = CS.commit(mine, data, salts_of=salts_of)
    CS.write(c, out, {"of": provenance(at), "committer": {"module": "a5_warden.py"}}, CS.index_doc(mine.members, entries, mine.run),
             kind="stand-in", registered=registered)
    return c


def reads(mine: CS.Window, at: dict[str, SimpleNamespace]) -> dict[str, dict]:
    """The verifier's own ``--registered`` file of each chunk: its registered port at the root the warden signed, each unit
    (from the chunk's first) opening the row of its read (q, k), placed by the wiring the warden signed."""
    return {m.key: WR.reads_file({LINKS[name][1]: WR.port(x.head, x.opening.wiring, x.params, x.reads, m.base)})
            for name, m in chunks(mine).items() for x in [at[name]]}


def source_of(mine: CS.Window, at: dict[str, SimpleNamespace]):
    """The verifier's check of the registration's window: a stand-in of its run naming the warden's cells as it read them."""
    want = provenance(at)

    def source(w: dict) -> dict:
        ok = {"kind": w.get("kind") == "stand-in", "run": w.get("run") == mine.run, "of": w.get("of") == want}
        return {"ok": all(ok.values()), "codes": [c for c, v in ok.items() if not v], "detail": {"kind": w.get("kind"), "run": w.get("run")}}
    return source


def gate_rows(x: SimpleNamespace) -> list[dict]:
    """Each unit's cell (q, k), its row and bucket, and its slot and words in the row, the address of its transfer gates: bit
    8k + i of the unit's registered port is bit i of the row's byte k, its own cell the slot's words."""
    first, half = x.registration["layout"]["first_bucket"], x.params.cell_bytes // 2
    return [{"cell": list(c), "row": t, "bucket": first + t, "slot": j, "words": [j * half, (j + 1) * half]}
            for c, t, j in zip(x.reads, x.positions, x.slots)]


def refusal(f, name: str) -> dict:
    """``f()`` refused by `name`: a ValueError whose message says it, not merely one."""
    try:
        f()
    except ValueError as e:
        return {"refused": name in str(e), "why": str(e)}
    return {"refused": False, "why": None}


def crossed(x: SimpleNamespace) -> tuple[WR.Wiring, tuple[int, int], tuple[int, int], tuple[int, int]]:
    """A foreign read on the link-window: a wiring without connection q1 (the warden's but for it), which places connection
    q2's cell k2, live just above q1 in one of q1's rows, on q1's cell k1; and that (q2, k2), (q1, k1) and their (row, slot)."""
    w, r = x.opening.wiring, x.params.slots
    live, held = w.live(x.params), {q: (first, cells) for q, first, cells in w.connections}
    for q2, first, cells in w.connections:
        for k2 in range(0, cells, r):
            t = first + k2 // r
            if live[t].index(q2):
                q1 = live[t][live[t].index(q2) - 1]
                forged = WR.Wiring(tuple(c for c in w.connections if c[0] != q1), w.salt)
                k1 = (t - held[q1][0]) * r
                (at,) = forged.place(x.params, [(q2, k2)])
                assert [at] == w.place(x.params, [(q1, k1)])
                return forged, (q2, k2), (q1, k1), at
    raise SystemExit(f"{x.head.value}: no two connections share a row, so no read of one can land on another's cell")


def padded(x: SimpleNamespace) -> tuple[WR.Wiring, tuple[int, int], tuple[int, int]]:
    """A padding-slot read on the link-window: the warden's wiring with one more connection g, live for one row past the last
    connection's first, which places (g, 0) on an empty slot; and that (g, 0) and its (row, slot)."""
    w, p = x.opening.wiring, x.params
    live, g = w.live(p), w.connections[-1][0] + 1
    t = next(t for t in range(w.connections[-1][1], p.window) if len(live[t]) < p.lanes)
    forged = WR.Wiring(w.connections + ((g, t, p.slots),), w.salt)
    (at,) = forged.place(p, [(g, 0)])
    return forged, (g, 0), at


def negatives(at: dict[str, SimpleNamespace], source, rec: dict) -> dict[str, dict]:
    """The prover's own root over the ingress link-window's rows (one byte off) under the warden's certificate, the window's
    provenance naming another root, and the reads no honest read is, each refused by `warden_rows.port` by name: a foreign
    read (`crossed`), a duplicate read, a padding-slot read (of a connection holding no slot, and under `padded`'s wiring)
    and a read past the connection's cells."""
    x = at[ws.INGRESS]
    p, B, w = x.params, x.params.cell_bytes, x.opening.wiring
    rows = list(x.opening.rows)
    rows[x.positions[0]] = flip(rows[x.positions[0]], byte=(x.slots[0] + 1) * B - 1)
    own = WR.Cells.commit(x.link, x.direction, x.window, rows, w, p, x.session)
    c = x.registration["certificate"]
    forged = WR.registration(own.head, Certificate(bytes.fromhex(c["key"]), x.link, x.direction, x.window,
                                                   bytes.fromhex(c["digest"]), bytes.fromhex(c["signature"])), p)
    check = refusal(lambda: WR.check_registration(forged, x.key, x.session, p, x.link, x.direction, x.window),
                    "doesn't sign its head")
    win = rec["window"]
    of = json.loads(json.dumps(win["of"]))
    of["links"][ws.INGRESS]["root"] = own.head.root.hex()
    s = source({**win, "of": of})

    def port(wiring: WR.Wiring, reads: list) -> None:
        WR.port(x.head, wiring, p, reads)

    def cell_at(t: int, j: int) -> bytes:
        return x.opening.rows[t][j * B:(j + 1) * B]

    cross, (q2, k2), (q1, k1), (t, j) = crossed(x)
    foreign = refusal(lambda: port(cross, [(q2, k2)]), "the wiring isn't the one the warden signed")
    pad, (g, _), (tp, jp) = padded(x)
    q, k0 = x.reads[0]
    cells = next(n for c, _, n in w.connections if c == q)
    return {"prover-root-under-the-wardens-certificate (check_registration)": check,
            "window-naming-another-warden-root (source)": {"refused": not s["ok"], "codes": s["codes"]},
            "foreign-read-under-a-crossed-wiring (port)": foreign | {
                "read": [q2, k2], "opens": [q1, k1], "row": t, "slot": j, "cell_of": HEADER.unpack_from(cell_at(t, j))[1]},
            "duplicate-read (port)": refusal(lambda: port(w, [x.reads[0], x.reads[0]]), f"cell {k0} is read twice")
            | {"read": [q, k0]},
            "padding-slot-read-of-a-connection-holding-no-slot (port)": refusal(lambda: port(w, [(g, 0)]), "never an empty slot")
            | {"read": [g, 0]},
            "padding-slot-read-under-a-padded-wiring (port)": refusal(lambda: port(pad, [(g, 0)]),
                                                                      "the wiring isn't the one the warden signed")
            | {"read": [g, 0], "row": tp, "slot": jp, "empty": cell_at(tp, jp) == bytes(B)},
            "read-past-the-connections-cells (port)": refusal(lambda: port(w, [(q, cells)]), f"not cell {cells}")
            | {"read": [q, cells]}}


def composer(defn, bits: int):
    """``a5.audit``'s composer of the template's circuit: C-Flock's `UnitStager` over its one ``row`` in and ``o0`` out,
    bound to the window's partition and program."""
    def compose(circuit: Path, partition: str, program: str) -> dict:
        from verity_flock.class_statement import UnitStager

        st = UnitStager(defn, "unit/a5-warden", [("row", V2, bits, ["row"])], [(V2, (0,))], circuit=circuit,
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
    ap.add_argument("--altered", action="store_true", help="unit 0 of the ingress chunk reads its row one bit off, in its cell")
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
    params = at[ws.INGRESS].params
    B, row = params.cell_bytes, WR.row_bytes(params)
    tg, rid = mine.members[0].template, mine.members[-1].template
    sets = {t: (composer(d, bits), d) for t, d, bits in zip((tg, rid), templates(params), (8 * row, 16 * B))}
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
                                "data_cells": x.data_cells, "chunk": chunks(mine)[name].key, "port": LINKS[name][1],
                                "cell_bytes": B, "row_bytes": row, "layout": x.registration["layout"], "gates": gate_rows(x)}
                         for name, x in at.items()},
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
