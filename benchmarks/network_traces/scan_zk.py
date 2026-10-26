"""scan-zk: NET-68's step-indexed scan composed, in zero knowledge at w63's class. Ingress bucket t
(`ingress_scan.IngressScanBucket`) reads the warden's ingress row t, the scan's state S_{t−1} and the Programs' row t,
and writes S_t and its egress view; egress bucket t (`egress.EgressBucket`) reads that view and the warden's egress row
t. A link-window's buckets are chained in one registration, every row of every unit a registered read, proved by C-Flock
`prove --zk` behind the Lean firewall and held by Lean `verify --zk --registered`. Standalone: in no Spec, except the
honest round on the challenger's coins (``--challenger``), a `service.Stream`'s.

    python benchmarks/network_traces/scan_zk.py prove --bin flock-circuit --lean flock-verify --out DIR [--dry]
        [--warden-run RUN] [--draw K [--challenger local|gate:USER@HOST --main FILE
        [--peer NAME --challenger-key RECORD] [--gate-key KEY]]]
    python benchmarks/network_traces/scan_zk.py schedule    # PARAMS as warden_serve.py's schedule flags

* **The class** (`CLASS`): w63's (B 512, r 4, 4 lanes, C 47, IDB 17, EVERY 10, WITHIN 12) with the warden's delay D 21
  and KO 64 outputs a request; egress at `CLASS.egress` (16 cells in lanes of 4, 4 request slots of 64 17-bit ids).
  The warden at `PARAMS`, 32-bucket link-windows, which the class serves.
* **The run** (`scan_traffic.serve`: the warden's own `_ingress` and `_egress`, ticked as `Warden._tick` ticks them)
  carries five warden-form id requests (`PROMPTS`, `ARRIVALS`) and their responses
  (`RESPONSES`: 3, 40, 64, 8 and 5 ids). The fifth waits in the queue until request 1's egress ends at bucket 21 and
  takes its slot at 22. Link-window 0 of each link is committed by the warden and its head signed by a test-size LMS
  key (`warden_rows.attest`); each registration is read back by the verifier (`warden_rows.check_registration`).
* **A warden's run** (``--warden-run RUN``, `warden_run`) in its place: a warden_serve run's directory (``serve``'s, or
  the gate's ``warden-rehearse --scan``'s with its witness under ``witness/``) at `PARAMS` (``scan_zk.py schedule``)
  carrying the scan's traffic (``rehearse --ids``). The verifier reads ``deployment.json`` (the warden's public key, the session,
  the schedule and its digest) and each link's window-0 registration in ``cells.jsonl``, its head's signature verified
  under that key; the prover's rows are the records under ``cells.salts``, held to that head. Each is refused by name
  where it doesn't hold, as is a schedule the class doesn't serve and a response that isn't id-form.
* **The scan's rows**, committed by the prover before the draw as hiding rows (`verity.ml.commitments.advice`, each
  tree's root in the registration): ``state``, S_{−1} (the empty state) … S_{T−1}, leaf t S_{t−1}; ``prog``, the
  Programs' rows, each request's response ids at its anchor (standing in for the anchored Programs' outputs); and
  ``view``, leaf t S_t's egress view.
* **The honest round, one registration**: one Program of two templates, T ingress units then T egress units, every
  unit drawn by the verifier's Lean. Ingress unit t reads ``row`` (the warden's ingress row t), ``sin`` (state leaf t),
  ``prog`` (leaf t), ``sout`` (state leaf t + 1) and ``view`` (leaf t), and outputs ``o0``, its ok bit. Egress unit t
  reads ``state`` (view leaf t) and ``row`` (the warden's egress row t), and outputs ``o0``, its verdict word. So S_t is
  the one leaf unit t writes and unit t + 1 reads, and the view the one leaf ingress t writes and egress t reads: the
  chain is the registered roots', not the prover's word. Accepted: every ok and every verdict opens to 1.
* **Negatives.** ``chain``: the same Program, its ingress units on a state tree with S_a and S_{a+1} swapped (the
  state rows of two buckets): the proofs hold and units a … a + 2's ok bits open to 0, so the window is refused.
  ``forged``: an egress-only Program, one egress row's `DATA` id changed, a row the warden never signed: refused by
  Lean's registered reads ("does not open").
* **A sampled draw** (``--draw K``): Lean ``draw --stratified K`` on the verifier's own coins (its OS randomness, the
  rehearsal's live coins), K/2 a template; only the drawn units are proved and opened, and an accepted audit carries
  ε at δ (``wrong_units``: at most ``bound`` of ``of`` units wrong, except with probability δ), its charge terms
  (`charge_terms`) and each member's cost per drawn unit (`per_unit`). The forged round draws K/2 of its T and is
  refused all the same: Lean holds every table row of a registered port, drawn or not. There is no chain round: its
  refusal is three units' ok bits, which a draw may not hold.
* **The challenger's coins** (``--challenger local|gate:USER@HOST --main FILE``, with ``--draw K``): the honest round's
  K drawn as prove_path's records audit draws (`a5_coins`): one session of the auditor's challenger on the window's root
  (`window_root`: the warden's session and link-window 0's two signed registrations), and the round's `service.Stream`
  (`spec_of`) on its coins (`a5_coins.RankCoins`): the stream registers the round, the coins commit to the
  registration's digest, and its select draws once (Lean's `Select.select` on the bytes read). Once the session has
  ended, the outcome of record is the stream's with the coins' evidence, the Lean's whole run (`Run.run`, coin source
  the challenger) held to the Python's, and each altered evidence (`a5_coins.negatives`, the evidence for another
  window) refused by `coins.coins_reason` and the Lean's run (`coins_outcome`). ``local`` is a development session on
  this machine (its own key; it shows the path, not an auditor's coins); ``gate:USER@HOST`` is vy-challenger-1's through
  infra's gate, its challenger on ``--peer``'s 127.0.0.1, under the deployment's key record ``--challenger-key``. The
  gate opens a root once, ever: a run's window is drawn once (the synthetic run's registrations carry fresh salts, so
  each synthetic run has a root of its own). The forged round keeps the verifier's Lean's own coins: one session serves
  one draw, and its refusal is Lean's registered reads, whatever the coins.
* **What the verifier receives** is scanned for every piece of every unit's rows and every salt but ``o0``'s, which the
  prover sends so the verifier opens each unit's output (`a8.received`), less the pieces public anyway (the circuits,
  zeros, the window's pads and `END` cells) and the sparse pieces (at most one nonzero byte); the prover's instance
  file, which holds them all, is the control.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import leak_scan as LS  # noqa: E402
import warden_zk as WZ  # noqa: E402
from verity.ml.circuits.codec import canonical_json  # noqa: E402
from verity.ml.circuits.defs import bind  # noqa: E402
from verity.ml.circuits.program import Program  # noqa: E402
from verity.ml.commitments import advice as ADV  # noqa: E402
from verity.ml.commitments.frame_v3 import BitRowLeaf  # noqa: E402
from verity.ml.commitments.rowleaf import bit_row, row_leaf  # noqa: E402
from verity.ml.crypto import lms  # noqa: E402
from verity.core.protocols.certifier import egress as EG, ingress_scan as IS, io_tie as IT, warden as W  # noqa: E402
from verity.core.protocols.certifier import warden_rows as WR  # noqa: E402
from verity.core.protocols.certifier.active import DATA, id_request  # noqa: E402
from verity.core.protocols.certifier.certificate import Direction, Issuer, LmsSigner  # noqa: E402
from verity.core.protocols.sampled_proofs import coins as CC, service as S  # noqa: E402
from verity.core.protocols.sampled_proofs.consumers import exfiltration_bound, location_bits  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import audit as AU, draw as DR, partition as PA, produced  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import registered as REG  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage.draw import stratum_k  # noqa: E402

_TRAFFIC = HERE.parents[1] / "verity/core/protocols/certifier/tests/scan_traffic.py"
_spec = importlib.util.spec_from_file_location("scan_traffic", _TRAFFIC)
traffic = sys.modules["scan_traffic"] = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(traffic)

B = 512
CLASS = IS.Class(B=B, r=4, lanes=4, C=47, IDB=17, EVERY=10, WITHIN=12, D=21, KO=64)
SHAPE = CLASS.egress
PARAMS = W.WardenParams(cell_bytes=B, slots=4, bucket_ms=100, delay=21, lanes=4, queue=4, window=32, ingress_cells=47,
                        ingress_buckets=12, egress_cells=65)
PROMPTS = ([(7 * k + 11) % 128_256 for k in range(1500)], [100_000 + k for k in range(600)],
           [131_071 - 3 * k for k in range(40)], [(k * k) % 128_256 for k in range(300)], [5_000 + k for k in range(100)])
ARRIVALS = (0, 2, 2, 12, 15)
RESPONSES = tuple(tuple((97 * k + 13 * q) % (1 << 17) for k in range(n)) for q, n in enumerate((3, 40, 64, 8, 5), 1))
#: link-windows the warden serves (every connection ends by the second's end); window 0 is proved
WINDOWS = 2
SESSION = bytes(range(32))
#: the chain round's swap: S_A and S_{A+1}, state leaves A + 1 and A + 2
A = 9
#: each round's outcome: accepted; refused on ok bits the verifier opens to 0; refused by Lean's registered reads
ROUNDS = {"honest": "accepted", "chain": "ok", "forged": "does not open"}
VALUES = {k: f"verity/certifier/scan-{k}/v1" for k in ("state", "prog", "view")}
ING_ROWS = (("row", 8 * CLASS.S * B), ("sin", IS.state_bits(CLASS)), ("prog", IS.prog_bits(CLASS)),
            ("sout", IS.state_bits(CLASS)), ("view", IS.view_bits(CLASS)))
EG_ROWS = (("state", SHAPE.state_bits), ("row", 8 * SHAPE.R * B))
TIMING = ("e2e_s", "wait_s", "witness_s", "prove_total_s", "proof_bytes", "wire_sent", "wire_recv", "k_log", "nbl",
          "dense_m", "units", "ands_per_vu", "prover", "gpu")
#: the honest round's stream on the auditor's challenger's coins (``--challenger``): its challenger, the format of the
#: window's root its session opens on (`window_root`), and each member's share of the proof deadline
CHALLENGER = "scan_zk.py on the auditor's challenger's coins"
WINDOW_ROOT = "verity/certifier/scan-window-root/v1"
MEMBER_DEADLINE_S = 5 * 3600


def log(msg: str) -> None:
    print(f"SCZK {time.strftime('%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


def _request(i: int, prompt: list[int]) -> bytes:
    return id_request(model="unsloth/Llama-3.2-1B", user=f"user-{i}", prompt=prompt, max_tokens=16 + i, seed=7 * i,
                      temperature=0.7, top_p=0.95, frame_bytes=B, offset=0)


def _words(bits) -> list[int]:
    return [sum(int(b) << i for i, b in enumerate(bits[w:w + 16])) for w in range(0, len(bits), 16)]


# --- the scan's rows: hiding rows the prover registers -----------------------------------------------------------------

def tree(kind: str, rows: list[list[int]], head, session: bytes) -> ADV.Registration:
    """`kind`'s rows (bit rows of whole u16 words) committed as hiding rows, bound to the session and the ingress
    link-window's head."""
    binding = hashlib.sha512(b"\0".join([VALUES[kind].encode(), session, head.value.encode(), head.root])).digest()
    return ADV.register([bit_row((_words(r), 16))[0] for r in rows], len(rows[0]), binding)


def tree_bc(reg: ADV.Registration, i: int) -> bytes:
    """Leaf i's row commitment b‖c, as a proof that holds the row under its salt commits it."""
    return ADV.INST.commit_string(reg.key, BitRowLeaf(reg.n_bits).layout().digest(reg.rows[i]), reg.salts[i])


def tree_port(kind: str, reg: ADV.Registration, reads: list[int], reads_from: int) -> dict:
    """The verifier's own ``--registered`` port of `kind`'s registration: unit ``reads_from + i`` reads leaf reads[i]."""
    n = len(reg.rows)
    e = {"leaves": n, "words": reg.n_bits // 16, "word_bits": 16, "domain": ADV.domain(reg.binding, n).domain_id.hex()}
    port = produced.port(VALUES[kind], e, reg.root.hex(), reads, BitRowLeaf(reg.n_bits).hiding_schema)
    return port | ({REG.READS_FROM: int(reads_from)} if reads_from else {})


def tree_entry(kind: str, reg: ADV.Registration, positions) -> dict:
    return {"value": VALUES[kind], "positions": [int(p) for p in positions],
            "paths": [b"".join(reg.open(p).path).hex() for p in positions]}


def tree_refusal(port: dict, reg: ADV.Registration, entry: dict, bcs: list[bytes]) -> str | None:
    """The host's read of a statement's entry on a scan registration, as Lean's registered reads make it (in
    `warden_rows.refusal`'s order): None when it holds."""
    if entry["value"] != port["value"]:
        return f"registered reads: the port reads {port['value']}, not another value ({entry['value']})"
    n, root = len(reg.rows), bytes.fromhex(port["root"])
    d, scheme = ADV.domain(reg.binding, n), ADV._scheme(reg.binding, n, reg.n_bits)
    for r, (p, bc, path) in enumerate(zip(entry["positions"], bcs, entry["paths"], strict=True)):
        leaf = row_leaf(d, p, ADV.INST.tree_leaf(reg.key, bc), BitRowLeaf(reg.n_bits).hiding_schema)
        nodes = tuple(bytes.fromhex(path[i:i + 128]) for i in range(0, len(path), 128))
        if not scheme.verify_path(ADV.PORT, root, n, p, leaf, nodes):
            return f"registered reads: table row {r} does not open the registered root at position {p}"
    base = int(port.get(REG.READS_FROM, 0))
    for i, (p, want) in enumerate(zip(entry["positions"], port["reads"], strict=True)):
        if p != want:
            return f"registered reads: unit {base + i} reads position {p}, and the verifier's program reads {want}"
    return None


# --- the run --------------------------------------------------------------------------------------------------------

def synthetic(out: Path) -> SimpleNamespace:
    """The synthetic run (`scan_traffic.serve`), link-window 0 of each link committed by the warden and its head signed
    by the driver's test-size LMS key (`warden_rows.attest`), each registration read back by the verifier."""
    CLASS.serves(PARAMS)
    reqs = [_request(i, p) for i, p in enumerate(PROMPTS)]
    served = traffic.serve(PARAMS, reqs, ARRIVALS, [1 << 20] * len(reqs), RESPONSES, WINDOWS)
    if served.fails:
        raise SystemExit(f"the warden failed {len(served.fails)} connections")
    out.mkdir(parents=True, exist_ok=True)
    signer = LmsSigner.create(lms.keygen(lms.Params(64, 1, 6), hashlib.sha512(b"scan-zk").digest()),
                              out / "signer-state.json")
    sides = {}
    for side, records, direction in (("ingress", served.ingress, Direction.INGRESS),
                                     ("egress", served.egress, Direction.EGRESS)):
        cells, cert = WR.attest(Issuer(signer, SESSION), records[0], PARAMS, direction)
        if cert is None:
            raise SystemExit("the test signer refused a link-window")
        registration = WR.registration(cells.head, cert, PARAMS)
        head, dom = WR.check_registration(registration, signer.public_key(), SESSION, PARAMS, records[0].link,
                                          direction, 0)
        if list(cells.rows) != WR.window_rows(records[0], PARAMS):
            raise SystemExit(f"{side}: the warden's committed rows aren't its record's")
        sides[side] = SimpleNamespace(cells=cells, registration=registration, head=head, dom=dom)
    return SimpleNamespace(source="the synthetic run (scan_traffic.serve), signed with the driver's own key",
                           params=PARAMS, session=SESSION, sides=sides, outputs=served.outputs, run_id=None)


def _found(run: Path, name: str) -> Path:
    """A warden run's file: in the run's directory (`warden_serve.py serve --out`), else its ``witness/`` (the gate's
    `warden-rehearse --scan`)."""
    for d in (run, run / "witness"):
        if (d / name).is_file():
            return d / name
    raise SystemExit(f"--warden-run {run}: no {name} there or in its witness/ (warden_serve.py serve writes it)")


def warden_run(run: Path) -> SimpleNamespace:
    """A warden_serve run's link-window 0 as the verifier reads it, refused by name where it doesn't hold: the
    deployment's schedule (its digest, and the class serving it) and public key; each link's registration in
    ``cells.jsonl`` verified under that key and session (`warden_rows.check_registration`: the warden's signature over
    the head of the rows' root); and the records' rows under ``cells.salts`` (`warden_rows.Cells.of`) that head. The
    units read those rows. The Programs' outputs (the ``prog`` row's stand-in) are each connection's ids as its egress
    record carries them (`io_tie.parse_cell`), every egress cell an id-form one."""
    import warden_serve as WS

    dep = json.loads(_found(run, "deployment.json").read_text())
    params = W.WardenParams(**dep["params"])
    if params.digest.hex() != dep["digest"]:
        raise SystemExit(f"--warden-run {run}: deployment.json's digest isn't its parameters'")
    try:
        CLASS.serves(params)
    except ValueError as e:
        raise SystemExit(f"--warden-run {run}: {e} (warden_serve.py rehearse at scan_zk's schedule, PARAMS, or the "
                         "gate's warden-rehearse --scan)")
    key, session = WS.cs.read_public(dep["key"]), bytes.fromhex(dep["session"])
    records = {name: [WS.ar.unpack_record(b, params.cell_bytes) for b in WS.ar.read_blobs(_found(run, f"{name}.records"))]
               for name in WS.CERTIFIED}
    lines = [json.loads(x) for x in _found(run, "cells.jsonl").read_text().splitlines()]
    salts = WS.ar.read_blobs(_found(run, "cells.salts"))
    if len(salts) != len(lines):
        raise SystemExit(f"--warden-run {run}: cells.jsonl has {len(lines)} lines and cells.salts {len(salts)} blobs")
    sides = {}
    for side, name, direction in (("ingress", WS.INGRESS, Direction.INGRESS), ("egress", WS.EGRESS, Direction.EGRESS)):
        got = [(doc["registration"], blob) for doc, blob in zip(lines, salts) if "registration" in doc and
               (doc["registration"]["link"], doc["registration"]["direction"], doc["registration"]["window"])
               == (WS.LINK, direction.value, 0)]
        record = next((r for r in records[name] if r.window == 0), None)
        if not got or record is None:
            raise SystemExit(f"--warden-run {run}: {side} link-window 0 has no registration or no record (the signer "
                             "refused it, or the run ended before it)")
        registration, blob = got[0]
        try:
            head, dom = WR.check_registration(registration, key, session, params, WS.LINK, direction, 0)
        except ValueError as e:
            raise SystemExit(f"--warden-run {run}: {side} link-window 0's registration doesn't verify under the "
                             f"deployment's key and session: {e}")
        row_salts, wiring_salt = WR.read_witness(blob)
        cells = WR.Cells.of(WS.LINK, direction, record, params, session, row_salts, wiring_salt)
        if cells.head != head:
            raise SystemExit(f"--warden-run {run}: {side} link-window 0's record under cells.salts isn't the rows "
                             "whose root the warden signed")
        sides[side] = SimpleNamespace(cells=cells, registration=registration, head=head, dom=dom)
    failed = sum(f is not None for r in records[WS.FAILS] if r.window == 0 for row in r.grid for f in row)
    if failed:
        raise SystemExit(f"--warden-run {run}: the warden failed {failed} connections in link-window 0")
    B, outputs = params.cell_bytes, {}
    for rec in sorted(records[WS.EGRESS], key=lambda r: r.window):
        for t, row in enumerate(WR.window_rows(rec, params)):
            for s in range(params.lanes * params.slots):
                cell = IT.parse_cell(row[s * B:(s + 1) * B])
                if cell is None:
                    raise SystemExit(f"--warden-run {run}: egress bucket {rec.window * params.window + t} slot {s} "
                                     "isn't an id-form cell (io_tie.parse_cell): the run's responses aren't the "
                                     "scan's (warden_serve.py rehearse --ids)")
                if cell.data:
                    outputs.setdefault(cell.stream, []).append(cell.id)
    return SimpleNamespace(source=f"warden run {dep.get('run_id')} (warden_serve.py, its deployment's key)",
                           params=params, session=session, sides=sides,
                           outputs={q: tuple(xs) for q, xs in outputs.items()}, run_id=dep.get("run_id"))


def link(out: Path, run: Path | None = None) -> SimpleNamespace:
    """The run (`synthetic`, or `warden_run`'s of a warden_serve run), window 0 of each link registered by the warden
    and read back by the verifier, the scan's chain over its rows (`scan_traffic.chain`) and the prover's three
    registrations."""
    t0 = time.time()
    got = synthetic(out) if run is None else warden_run(run)
    T = got.params.window
    ingress, egress = (list(got.sides[s].cells.rows) for s in ("ingress", "egress"))
    try:
        ch = traffic.chain(CLASS, ingress, egress, got.outputs)
    except (ValueError, KeyError) as e:
        raise SystemExit(f"the scan refuses link-window 0's rows: {e!r} (the scan reads warden-form id requests at its "
                         "class, answered id-form: warden_serve.py rehearse --ids)")
    if [EG.verdict(SHAPE, ch.view(t), ch.egress[t]) for t in range(T)] != [EG.ACCEPT] * T:
        raise SystemExit("an egress bucket refuses the view its ingress bucket wrote")
    head = got.sides["ingress"].head
    trees = {k: tree(k, rows, head, got.session) for k, rows in (("state", ch.states), ("prog", ch.progs),
                                                                ("view", ch.views))}
    swapped = list(ch.states)
    swapped[A + 1], swapped[A + 2] = swapped[A + 2], swapped[A + 1]
    admitted: dict[int, int] = {}
    for st in ch.scan:
        for s in st.slots:
            if s.live:
                admitted.setdefault(s.q, s.admitted)
    return SimpleNamespace(chain=ch, T=T, params=got.params, session=got.session, source=got.source,
                           run_id=got.run_id, outputs=got.outputs, sides=got.sides, trees=trees,
                           swapped=tree("state", swapped, head, got.session), ordinals=sorted(admitted),
                           admitted=dict(sorted(admitted.items())), run_s=round(time.time() - t0, 1))


def _forged(at: SimpleNamespace) -> tuple[int, bytes]:
    """The first egress row with a `DATA` cell, the last digit of that cell's id changed."""
    for t, row in enumerate(at.chain.egress):
        for s in range(SHAPE.R):
            if row[s * B] == DATA:
                x = bytearray(row)
                k = s * B + IT.ID_AT + IT.ID_DIGITS - 1
                x[k] = ord("0") + (x[k] - ord("0") + 1) % 10
                return t, bytes(x)
    raise SystemExit("the window has no DATA cell")


def member(at: SimpleNamespace, side: str, name: str, base: int) -> SimpleNamespace:
    """Round `name`'s `side` member: each unit's rows (in its signature's order, as bits or bytes), the salts its rows
    are committed under, its header's `registered` entries and the verifier's own ports, the host's read of them and
    each unit's output."""
    ch, T = at.chain, at.T
    cells = at.sides[side].cells
    _w, row_salts, row_entry = cells.statement(range(T))
    port = {"row": WR.bucket_port(at.sides[side].head, at.params, base)}
    if side == "ingress":
        st, prog, view = (at.swapped if name == "chain" else at.trees["state"]), at.trees["prog"], at.trees["view"]
        reads = {"sin": ("state", st, list(range(T))), "prog": ("prog", prog, list(range(T))),
                 "sout": ("state", st, list(range(1, T + 1))), "view": ("view", view, list(range(T)))}
        rows = list(ch.ingress)
        units = [(rows[t], *(_bits(r, p) for _k, r, ps in reads.values() for p in [ps[t]])) for t in range(T)]
        outputs = [IS.reference(CLASS, *u) for u in units]
    else:
        reads = {"state": ("view", at.trees["view"], list(range(T)))}
        rows = list(ch.egress)
        changed = None
        if name == "forged":
            t, rows[t] = _forged(at)
            changed = t
        units = [(_bits(at.trees["view"], t), rows[t]) for t in range(T)]
        if any(u[0] != EG.state_bits(SHAPE, ch.view(t)) for t, u in enumerate(units)):
            raise SystemExit("a view leaf isn't its bucket's egress view")
        outputs = [EG.verdict(SHAPE, ch.view(t), rows[t]) for t in range(T)]
    salts = {"row": row_salts, **{p: [reg.salts[i] for i in ps] for p, (_k, reg, ps) in reads.items()}}
    entries = {"row": row_entry, **{p: tree_entry(k, reg, ps) for p, (k, reg, ps) in reads.items()}}
    port.update({p: tree_port(k, reg, ps, base) for p, (k, reg, ps) in reads.items()})
    bcs = {"row": [WR.bc(r, s) for r, s in zip(rows, row_salts)],
           **{p: [tree_bc(reg, i) for i in ps] for p, (_k, reg, ps) in reads.items()}}
    refused = WR.refusal(port["row"], at.sides[side].dom, row_entry, bcs["row"])
    for p, (_k, reg, _ps) in reads.items():
        refused = refused or tree_refusal(port[p], reg, entries[p], bcs[p])
    return SimpleNamespace(side=side, name=name, units=units, rows=rows, salts=salts, entries=entries, ports=port,
                           bcs=bcs, refused=refused, outputs=outputs, changed=changed if side == "egress" else None)


def _bits(reg: ADV.Registration, i: int) -> list[int]:
    row = reg.rows[i]
    return [row[k // 8] >> k % 8 & 1 for k in range(reg.n_bits)]


def predicted(name: str, members: list[SimpleNamespace]) -> bool:
    """The host's read of a round before any proof, as the round's: honest, every read held and every output 1;
    chain, every read held and the swapped buckets' ok bits 0; forged, its egress row port refused."""
    if ROUNDS[name] == "accepted":
        return all(m.refused is None and set(m.outputs) == {1} for m in members)
    if ROUNDS[name] == "ok":
        ing = next(m for m in members if m.side == "ingress")
        return all(m.refused is None for m in members) and [t for t, o in enumerate(ing.outputs) if not o] == [
            A, A + 1, A + 2]
    return all(ROUNDS[name] in (m.refused or "") for m in members if m.side == "egress")


def host(out: Path, run: Path | None = None, draw: int | None = None) -> SimpleNamespace:
    """The run (`link`'s, of `run` when given) and every round's members as the host reads them (no lowering, no
    proof): what the test holds. Under a sampled draw (`draw`) there is no chain round: its refusal is the ok bits of
    three units the draw may not hold."""
    at = link(out, run)
    rounds = {"honest": ("ingress", "egress"), "chain": ("ingress", "egress"), "forged": ("egress",)}
    if draw is not None:
        del rounds["chain"]
    got = {}
    for name, sides in rounds.items():
        got[name] = [member(at, side, name, base) for side, base in zip(sides, _bases(sides, at.T))]
        if not predicted(name, got[name]):
            raise SystemExit(f"{name}: the host's read of the round isn't {ROUNDS[name]}: "
                             f"{[(m.side, m.refused, m.outputs) for m in got[name]]}")
    return SimpleNamespace(at=at, rounds=got)


def _bases(sides, T: int) -> list[int]:
    """Each member's first unit in its round's Program: the templates in order, T units each."""
    return [k * T for k in range(len(sides))]


# --- the statements ---------------------------------------------------------------------------------------------------

def _defs() -> dict[str, object]:
    return {"ingress": bind(IS.IngressScanBucket, B=CLASS.B, r=CLASS.r, lanes=CLASS.lanes, C=CLASS.C, IDB=CLASS.IDB,
                            EVERY=CLASS.EVERY, WITHIN=CLASS.WITHIN, D=CLASS.D, KO=CLASS.KO),
            "egress": bind(EG.EgressBucket, B=SHAPE.B, R=SHAPE.R, r=SHAPE.r, NR=SHAPE.NR, K=SHAPE.K, IB=SHAPE.IB,
                           D=SHAPE.D)}


def program(M, sides: tuple[str, ...], T: int, out: Path) -> SimpleNamespace:
    """A round's Program and partition (a template a side, T units each) and each side's stager."""
    defs = _defs()
    prog = M.P.mixed_population_program([(defs[s], T) for s in sides])
    query = M.P.template_instances_query([defs[s] for s in sides])
    part = M.P.partition(prog, query)
    out.mkdir(parents=True, exist_ok=True)
    (out / "partition.json").write_text(json.dumps(part, sort_keys=True))
    (out / "program.json").write_bytes(M.canonical_json(M.encode_program(prog)))
    bound = {"partition": M.P.digest(part), "program": part["program"]}
    stagers = {}
    for side, (tid, base, cnt) in zip(sides, M.P.ranges(prog, query), strict=True):
        assert tid == M.P.template_id(defs[side]) and cnt == T and base == _bases(sides, T)[sides.index(side)]
        t = time.time()
        rows = ING_ROWS if side == "ingress" else EG_ROWS
        sdir = out / side
        sdir.mkdir(exist_ok=True)
        st = M.CS.UnitStager(defs[side], f"unit/scan-{side}", [(r, M.V2, b, [r]) for r, b in rows], [(M.V2, (0,))],
                             circuit=sdir / "circuit.txt", bind=bound)
        st.pin
        stagers[side] = SimpleNamespace(defn=defs[side], st=st, base=base, n=T, out=sdir, rows=rows,
                                        lower_s=round(time.time() - t, 1))
    return SimpleNamespace(program=prog, query=query, sides=sides, stagers=stagers, partition_file=out / "partition.json",
                           program_file=out / "program.json")


def stage(M, pg: SimpleNamespace, mr: SimpleNamespace, out: Path) -> dict:
    """Member mr's statement: each unit's rows in a shared-row file (each instance its own table row), every input row
    under the salts and paths of its registered entry, ``o0`` under a fresh salt; and the verifier's own `--registered`
    file."""
    t0 = time.time()
    sg = pg.stagers[mr.side]
    st, n = sg.st, sg.n
    nbl = max(3, (-(-n // st.meta["vus_per_block"]) - 1).bit_length())
    if not 25 <= st.meta["k_log"] + nbl <= M.a7.M_MAX:
        raise SystemExit(f"{mr.side}: m = {st.meta['k_log']} + {nbl} is outside 25 .. {M.a7.M_MAX}")
    words = {}
    for k, (r, _b) in enumerate(sg.rows):
        col = [u[k] for u in mr.units]
        words[r] = M.np.asarray([WR.words(x) if isinstance(x, bytes) else _words(x) for x in col], dtype=M.np.uint16)
    set_name = f"{st.iset.name}-{mr.name}"
    salts = {**mr.salts, "o0": [os.urandom(WR.SALT_BYTES) for _ in range(n)]}
    got = st.stage(words, out, os.urandom(32), salts=salts, indices=range(sg.base, sg.base + n), set_name=set_name,
                   tables=words, refs={r: list(range(n)) for r in words}, registered=mr.entries)
    o0 = [int(x[0]) for x in M.np.asarray(got["outs"]["o0"])]
    if o0 != mr.outputs:
        raise SystemExit(f"{mr.name}/{mr.side}: the units' outputs {o0} are not the reference's {mr.outputs}")
    (out / "inst.bin").rename(out / f"inst-{n}.bin")
    (out / "pub.bin").rename(out / f"pub-{n}.bin")
    (out / "tables").mkdir(exist_ok=True)
    os.link(sg.out / "circuit.txt", out / "circuit.txt")
    (out / "registered.json").write_text(json.dumps(WR.reads_file(mr.ports), sort_keys=True))
    m = {"tag": f"{mr.name}-{mr.side}", "name": mr.side, "dir": out, "meta": st.meta, "pin": st.pin,
         "class": st.class_sha512, "set": set_name, "n": n, "base": sg.base, "defn": sg.defn, "ands": got["ands"],
         "unit_nnz": got["unit_nnz"], "salts": got["salts"], "ports": st.ports, "outputs": o0, "units": mr.units,
         "partition_file": pg.partition_file, "program_file": pg.program_file,
         "registered_file": out / "registered.json", "k_log": st.meta["k_log"], "nbl": nbl,
         "stage_s": round(time.time() - t0, 1)}
    m["header"] = M.a0.public_header(out, n)
    m["bcs"] = shared_bcs(out / f"pub-{n}.bin", st.ports)
    if any(m["bcs"][i][p] != bc for p, col in mr.bcs.items() for i, bc in enumerate(col)):
        raise SystemExit(f"{mr.name}/{mr.side}: a staged row's b‖c isn't the one its registered entry opens")
    if not all(M.a7.opens(next(p for p in st.ports if p.name == "o0"), [w], salts["o0"][i], m["bcs"][i]["o0"])
               for i, w in enumerate(o0)):
        raise SystemExit(f"{mr.name}/{mr.side}: a unit's output doesn't open against its public o0 commitment")
    return m


def shared_bcs(path: Path, ports: list) -> list[dict[str, bytes]]:
    """Each instance's b‖c by port from a shared-row public file (its header line, each port's table rows' b‖c,
    port-major in table order, then every instance's refs, u32 little-endian, instance-major, ports in order)."""
    import numpy as np

    head, body = path.read_bytes().split(b"\n", 1)
    h, w = json.loads(head), 128
    n, at = h["instances"], [0]
    for p in ports:
        at.append(at[-1] + int(h["shared_rows"][p.name]))
    refs = np.frombuffer(body[at[-1] * w:at[-1] * w + 4 * n * len(ports)], dtype="<u4").reshape(n, len(ports))
    return [{p.name: body[(at[q] + int(refs[i, q])) * w:(at[q] + int(refs[i, q]) + 1) * w] for q, p in enumerate(ports)}
            for i in range(n)]


# --- the challenger's coins ---------------------------------------------------------------------------------------------

def window_root(at: SimpleNamespace) -> str:
    """The root the challenger's session opens on (128 hex digits): the SHA-512 of the warden's session and link-window
    0's two registrations, whose heads the warden signed. The gate opens a root once, ever, so a warden run's window is
    drawn once."""
    return hashlib.sha512(canonical_json({"format": WINDOW_ROOT, "session": at.session.hex(),
                                          "registrations": {s: x.registration for s, x in sorted(at.sides.items())}}
                                         )).hexdigest()


@contextmanager
def challenger(a, at: SimpleNamespace, out: Path):
    """The challenger's session for the honest round's draw (``--challenger``, as prove_path's `records_audit.audit`
    starts it): `a5_coins.start` on `window_root` for one rank, its evidence copied to ``out / "challenger"``; once the
    rounds are done the rank's turn ends whether or not it drew (`Draws.skip`) and the session's evidence is published,
    which the yielded namespace's ``published`` then holds. None without ``--challenger``. Needs `WZ._one_stage`'s path
    (a5_coins is one-stage's)."""
    if not a.challenger:
        yield None
        return
    import a5_coins as AC

    root = window_root(at)
    manager, draws, finisher = AC.start(a.challenger, root, [0], out / "challenger", a.main, ssh_key=a.gate_key,
                                        peer=a.peer, key_record=a.challenger_key)
    coins = SimpleNamespace(AC=AC, draws=draws, root=root, challenger=a.challenger, published=None)
    try:
        yield coins
    finally:
        draws.skip(0)
        finisher.join()
        coins.published = draws.evidence()
        manager.shutdown()


def spec_of(pg: SimpleNamespace, members: list[dict], want: dict, schemas: dict[str, str], k: int,
            root: str) -> tuple[S.Spec, list, dict[str, int]]:
    """The honest round's `service.Spec` (prove_path's `a5.stream_of`): its unit the round's templates in order, its
    values the round's ports (each member's frame ports and the scan's trees, under their hiding ``schemas``), its
    public items what the registration shows (N, each template's stratum, the window, the leaf layer and each port's
    layout), its selection ``stratified:k``, its proof deadline `MEMBER_DEADLINE_S` a member and its context the
    window's root. -> (the spec, its units, `Stream.register`'s counts)."""
    units = [pg.stagers[s].defn for s in pg.sides]
    law = S.Law(f"stratified:{k}")
    n = sum(m["n"] for m in members)
    strata = [s["template"] for s in DR.derive(pg.program, pg.query, S.selection_law(law, n)).get("strata") or ()]
    public = [("population", "N, the round's proof units"), *((f"population:{t}", "a template's stratum") for t in strata),
              ("window", "the round's provenance: link-window 0 of the warden's run"),
              ("leaf_layer", "the leaf layer over the members' public files"),
              *((f"{x}:{p}", "a port's layout") for p in sorted(want) for x in ("leaves", "domain"))]
    check = S.Check(tuple(PA.program_digest(Program(u)) for u in units), pg.query,
                    proof_deadline=float(MEMBER_DEADLINE_S * len(members)))
    spec = S.Spec(tuple(S.Value(p, schemas[p], "serve") for p in sorted(want)), law, check,
                  tuple(S.Public(item, "structure", why) for item, why in public), bytes.fromhex(root))
    return spec, units, {PA.template_id(m["defn"]): m["n"] for m in members}


def coins_outcome(a, coins: SimpleNamespace, spec: S.Spec, units: list, counts: dict, stream: S.Stream, draw: S.Draw,
                  rec: dict, want: dict, leaf_layer: str, read: bytes, segment: dict | None, sessions: list[str],
                  verdicts: dict, statement: dict, given: dict) -> tuple[dict | None, dict, dict]:
    """The honest draw's outcome of record once the challenger's session has ended (prove_path's `a5.coins_outcome`):
    its proof delivered and its outcome with the coins' evidence (`a5_coins.evidence`), the Lean's whole run (`Run.run`,
    `coin_source` challenger, its window `window_root`) held to the Python's, the audit failing ``coins`` where
    `coins.coins_reason` refuses; then each altered evidence (`a5_coins.negatives`, and the evidence for another window)
    as its own stream's outcome on the honest draw's bytes (`a5_coins.Replay`), refused where `coins_reason` refuses and
    the Lean's run refuses at its select. -> (the audit record, None where the stream refused or holds the outcome; the
    negatives; the coins' record)."""
    AC, root, after = coins.AC, coins.root, bytes.fromhex(draw.registration.digest)
    published = coins.draws.evidence()
    ev = AC.evidence(published, segment) if segment is not None else None

    def outcome(s: S.Stream, d: S.Draw, evidence: dict | None) -> tuple[S.Outcome, dict | None]:
        late = s.deliver(d, sessions)
        return (late if late is not None else s.outcome(d, verdicts, statement, coins=evidence, **given)), \
            s.records[-1]["audit"]

    def reason(evidence: dict | None, window: str = root) -> str | None:
        return CC.coins_reason(evidence, window, after, read, AC.lms_verify)

    why = (published.get("error") or "the draw committed no segment") if ev is None else reason(ev)
    try:
        o, result = outcome(stream, draw, ev)
        told = {"stream": stream.digest, "record": o.record, "accepted": o.accepted, "codes": list(o.verdict.codes)}
    except (S.Refused, S.Pending) as e:
        result, told = None, {"stream": stream.digest, "accepted": False, "refused": AC._refusal(e)}
    if result is not None and why is not None:
        result = {**result, "accepted": False, "profile": None, "integrity_profile": None, "wrong_units": None,
                  "checks": {**result["checks"], "coins": {"ok": False, "codes": ["coins"], "detail": {"coins": why}}}}
    neg: dict[str, dict] = {}
    bad = {**AC.negatives(ev, coins.draws.pre()), "coins-for-another-window": ev} if ev is not None else {}
    for name, evidence in bad.items():
        window = "00" * 64 if name == "coins-for-another-window" else root
        s = S.Stream(spec, units, budget=1, delta=a.delta, challenger=CHALLENGER, coins=AC.Replay(read))
        try:
            d = s.select(s.register(rec, want, leaf_layer=leaf_layer, window_root=window, counts=counts))
            if d.draw != draw.draw:
                raise ValueError(f"the replayed draw {d.units} is not the honest draw {draw.units}")
            got, audit = outcome(s, d, evidence)
            accepted, side = got.accepted, AC.lean_side(audit)
        except (S.Refused, S.Pending, ValueError, AC.Ch.CoinError) as e:
            accepted, side = None, {"refused": AC._refusal(e)}
        py = reason(evidence, window)
        step = ((side or {}).get("step"), (side or {}).get("codes"))
        neg[name] = {"refused": py is not None and accepted is False and step == AC.SELECT_COINS, "accepted": accepted,
                     "python": py, "lean": side}
    checks = (result or {}).get("checks") or {}
    record = {"challenger": coins.challenger, "source": stream.coin_source, "window_root": root, "segment": segment,
              "read": read.hex(), "coins_reason": why, "outcome": told,
              "run": None if result is None or not result["verifiers"]["agree"] else checks.get("run", {"ok": True})}
    return result, neg, record


# --- the audit ----------------------------------------------------------------------------------------------------------

def audit(M, a, at: SimpleNamespace, pg: SimpleNamespace, name: str, members: list[dict], trees: dict, out: Path,
          port: int, k: int | None = None, coins: SimpleNamespace | None = None) -> dict:
    """One round registered (its members' ports and the scan's registrations' roots, before any coin), its units drawn
    by the verifier's Lean (`draw --stratified k` on its own coins; every unit when `k` is None) or, given the
    challenger's ``coins``, by the round's `service.Stream` on them (`spec_of`; `a5_coins.RankCoins`, rank 0), each
    member's drawn units proved --zk behind the firewall and held by Lean with the verifier's own registered reads; then
    each held unit's ``o0`` opened under the salt the prover sends. Writes what the verifier received under
    ``out / "verifier"``. An accepted audit carries its charge terms (`charge_terms`) and each member's cost per drawn
    unit; on the challenger's coins, the outcome of record is the stream's (`coins_outcome`)."""
    a0, a7, P, R, V = M.a0, M.a7, M.P, M.R, M.V
    out.mkdir(parents=True, exist_ok=True)
    want, roots, publics, schemas = {}, [], {}, {}
    for m in members:
        fv = m["header"]["frame_v3"]
        for p_, w in a7.frame_ports(m["set"], m["meta"], m["n"]).items():
            want[f"{m['name']}/{p_}"] = w
            schemas[f"{m['name']}/{p_}"] = fv["schemas"][p_]
            roots.append({"port": f"{m['name']}/{p_}", "root": fv["roots"][p_], "leaves": w["leaves"],
                          "domain": fv["domain_ids"][p_]})
        publics[m["name"]] = a0.sha512_file(m["dir"] / f"pub-{m['n']}.bin")
    for kind, reg in trees.items():
        d = ADV.domain(reg.binding, len(reg.rows)).domain_id.hex()
        want[f"scan/{kind}"] = {"leaves": len(reg.rows), "domain": d}
        schemas[f"scan/{kind}"] = BitRowLeaf.hiding_schema
        roots.append({"port": f"scan/{kind}", "root": reg.root.hex(), "leaves": len(reg.rows), "domain": d})
    leaf_layer = hashlib.sha512(M.canonical_json(publics)).hexdigest()
    k = sum(m["n"] for m in members) if k is None else k
    law = M.D.parse_law(f"stratified:{k}")
    win = {"kind": "stand-in", "of": f"round {name}: link-window 0 of {at.source}, its ingress and egress at w63's "
                                     "class (scan_zk.py)",
           "members": {m["name"]: {"template": P.template_id(m["defn"]), "units": [m["base"], m["base"] + m["n"]]}
                       for m in members},
           "link": "registered: each row port at the warden's signed root, unit base + t row t; the scan's state, "
                   "prog and view at the prover's registered roots (scan/*)"}
    rec = R.record(pg.program, pg.query, law, a0.SCHEME, roots, win, leaf_layer=leaf_layer)
    (out / "registration.json").write_text(json.dumps(rec, indent=1, sort_keys=True))
    checks = {"registration": V.check(rec, pg.program, pg.query, law, a0.SCHEME, want, leaf_layer=leaf_layer).to_json()}
    for m in members:
        checks[f"statement/{m['name']}"] = a7.statement_checks(m, rec, want)
    receipt = R.receipt(rec, "one-stage-e2e verifier (scan_zk.py)")
    res: dict = {"checks_before_draw": {c: v["ok"] for c, v in checks.items()}}
    vdir = out / "verifier"
    vdir.mkdir(exist_ok=True)
    if not all(v["ok"] for v in checks.values()):
        return {**res, "checks": checks, "accepted": False, "window_accepted": False, "verdicts": {}, "why": {},
                "opened": {}}
    stream = None
    if coins is None:
        p = a0.run([a.lean, "draw", "--program", str(pg.program_file), "--partition", str(pg.partition_file),
                    "--stratified", str(k)], out / "draw.log")
        lean_draw = json.loads(next(x for x in p.stdout.splitlines() if x.startswith("DRAW ")).split(" ", 1)[1])
    else:
        spec, units, counts = spec_of(pg, members, want, schemas, k, coins.root)
        rank = coins.AC.RankCoins(coins.draws, 0)
        draw = refused = None
        try:
            stream = S.Stream(spec, units, budget=1, delta=a.delta, challenger=CHALLENGER, coins=rank)
            rank.plan(coins.AC.need(f"stratified:{k}", pg.program, pg.query, rec["population"]))
            reg = stream.register(rec, want, leaf_layer=leaf_layer, window_root=coins.root, counts=counts)
            rank.commit(bytes.fromhex(reg.digest))
            draw = stream.select(reg)
        except S.Refused as e:
            refused = e.verdict.to_json()
        except (ValueError, coins.AC.Ch.CoinError, V.LeanError) as e:
            refused = {"ok": False, "codes": ["stream"], "detail": {"stream": str(e)}}
        finally:
            segment = rank.drawn()
        read = bytes(rank.read)
        (out / "segment.json").write_text(json.dumps({"challenger": coins.challenger, "window_root": coins.root,
                                                      "segment": segment, "read": read.hex(), "need": rank.need},
                                                     sort_keys=True))
        if draw is None:
            coins.draws.skip(0)
            log(f"{name}: the stream drew nothing on the challenger's coins: {refused}")
            for f in (out / "registration.json", out / "segment.json"):
                os.link(f, vdir / f.name)
            return {**res, "accepted": False, "window_accepted": False, "verdicts": {}, "why": {"draw": refused},
                    "opened": {}, "draw": {"law": f"stratified:{k}", "coins": coins.challenger, "refused": refused}}
        lean_draw, receipt = dict(draw.draw), dict(draw.registration.receipt)
        (out / "receipt.json").write_text(json.dumps(receipt, indent=1, sort_keys=True))
    (out / "draw.json").write_text(json.dumps(lean_draw, sort_keys=True))
    proved, served, n_inst, verdicts, statement, sessions, session_roots = [], [], 0, {}, {}, [], {}
    for j, m in enumerate(members):
        local = sorted(u - m["base"] for u in lean_draw["units"] if m["base"] <= u < m["base"] + m["n"])
        sdir = out / f"session-{m['name']}"
        record, live, proc = M.a8.session(a, m, sdir, local, port + j, env=WZ.firewall(M, m))
        draw_t = (record or {}).get("unit_draw") or {}
        v = V.check_draw(draw_t, {"population": m["n"]}, {"law": "subset", "k": len(local)})
        verdicts[f"m0-verifier/{m['name']}"] = {
            "accepted": ((record or {}).get("verdict") or {}).get("accepted") is True
            and (live or {}).get("roots") == m["header"]["frame_v3"]["roots"] and v.ok,
            "why": ((record or {}).get("verdict") or {}).get("why"), "draw_check": v.to_json()}
        lv = a0.lean_verify(a.lean, m, m["n"], sdir, zk=True)
        verdicts[f"lean-verify/{m['name']}"] = {**lv, "tool": "flock-verify verify --zk --archive --partition "
                                                              "--program --registered"}
        proved += a0.proved_units(m["header"], live or {})
        served += [m["base"] + u for u in draw_t.get("units", [])]
        n_inst += (live or {}).get("instances", 0)
        session_roots.update({f"{m['name']}/{p_}": r for p_, r in ((live or {}).get("roots") or {}).items()})
        timing = {x: (live or {}).get(x) for x in TIMING}
        statement[m["name"]] = {"session_s": proc["wall_s"], "timing": timing, "process": proc,
                                "lean_verify_s": lv.get("verify_s"), "drawn": len(local),
                                **{x: (live or {}).get(x) for x in ("circuit_sha512", "statement_digest", "sigma")}}
        sessions += [a0.sha512_file(f) for f in sorted((sdir / "sessions").glob("*/session.json"))]
        log(f"{name}/{m['name']}: {len(local)} units drawn, Lean {lv.get('accepted')} {lv.get('why') or ''}, "
            f"prove {timing.get('prove_total_s')} s, session {proc['wall_s']} s")
    neg, told = {}, None
    if stream is None:
        result = V.audit_record(rec, receipt, lean_draw, rec["law"], sorted(proved), n_inst, statement, verdicts,
                                a.delta, sessions=sessions, served=sorted(served), session_roots=session_roots)
    else:  # every verdict of record names the sessions delivered (`service.check_verdicts`)
        verdicts = {x: {**y, "sessions": sessions} if y.get("of_record") else y for x, y in verdicts.items()}
        given = {"proved_units": sorted(proved), "instances": n_inst, "served": sorted(served),
                 "session_roots": session_roots}
        result, neg, told = coins_outcome(a, coins, spec, units, counts, stream, draw, rec, want, leaf_layer, read,
                                          segment, sessions, verdicts, statement, given)
        (out / "coins.json").write_text(json.dumps({**told, "negatives": neg}, indent=1, sort_keys=True, default=str))
        log(f"{name}: the challenger's coins ({coins.challenger}): outcome {told['outcome']}, coins_reason "
            f"{told['coins_reason']}, run {told['run']}, negatives refused "
            f"{ {x: y['refused'] for x, y in neg.items()} }")
        result = result or {"accepted": False, "outcome": told["outcome"]}
    (out / "audit.json").write_text(json.dumps(result, indent=1, sort_keys=True, default=str))
    held = set(proved) if result.get("accepted") else set()
    opened, sent = {}, []
    for m in members:
        o0 = next(p for p in m["ports"] if p.name == "o0")
        opened[m["name"]] = [w if (m["base"] + i) in held and M.a7.opens(o0, [w], m["salts"]["o0"][i], m["bcs"][i]["o0"])
                             else None for i, w in enumerate(m["outputs"])]
        sent += [{"unit": m["base"] + i, "o0": w, "salt/o0": m["salts"]["o0"][i].hex()}
                 for i, w in enumerate(m["outputs"]) if (m["base"] + i) in held]
    (vdir / "openings.json").write_text(json.dumps(sent, indent=1))
    for side, s in at.sides.items():
        (vdir / f"warden-registration-{side}.json").write_text(json.dumps(s.registration, indent=1, sort_keys=True))
    streamed = (out / "receipt.json", out / "segment.json", out / "coins.json") if stream is not None else ()
    for f in (out / "registration.json", pg.partition_file, pg.program_file, out / "draw.json", out / "audit.json",
              *streamed):
        os.link(f, vdir / f.name)
    for m in members:
        for f in (m["dir"] / "circuit.txt", m["dir"] / f"pub-{m['n']}.bin", m["registered_file"]):
            os.link(f, vdir / f"{m['name']}-{f.name}")
        for f in sorted((out / f"session-{m['name']}" / "sessions").rglob("*")):
            if f.is_file():
                os.link(f, vdir / f"{m['name']}-session-{f.parent.name}-{f.name}")
    got = [w for col in opened.values() for w in col if w is not None]
    window_ok = bool(result.get("accepted")) and len(got) == len(held) and all(w == 1 for w in got)
    source = "the verifier's Lean's own (flock-verify draw, OS randomness)" if stream is None else \
        f"the auditor's challenger's ({coins.challenger}: a5_coins, the round's service.Stream)"
    return {**res, "accepted": bool(result.get("accepted")), "window_accepted": window_ok, "opened": opened,
            "draw": {"law": f"stratified:{k}", "coins": source, "units": lean_draw["units"], "of": rec["population"]},
            **({"coins": told, "coin_negatives": {x: y["refused"] for x, y in neg.items()}} if told else {}),
            "audit": {x: result.get(x) for x in ("accepted", "complete", "wrong_units")},
            "charge": charge_terms(M, result, members, a.delta),
            "per_unit": per_unit(members, statement),
            "verdicts": {x: y.get("accepted") for x, y in verdicts.items()},
            "why": {x: y.get("why") for x, y in verdicts.items()}, "timing": statement}


def charge_terms(M, result: dict, members: list[dict], delta: float) -> dict | None:
    """An accepted audit's terms (prove_path's `charge_terms`): ε, at most ``wrong_units.bound`` of its ``of`` units
    wrong except with probability δ; λ_δ (``location_bits``), β_δ (``harm_bound``: each template's units weighing the
    bits one of its units commits, `partition._template_bits`) and ``exfiltration_bound`` = β_δ + λ_δ. None where the
    audit isn't accepted, which bounds nothing."""
    if not result.get("accepted"):
        return None
    p = AU.profile(result["registration"], result["draw"], delta)
    bits = {M.P.template_id(m["defn"]): float(M.P._template_bits(m["defn"])) for m in members}
    wu = result.get("wrong_units") or {}
    return {"delta": delta, "wrong_units": wu, "epsilon": wu["bound"] / wu["of"] if wu.get("of") else None,
            "template_bits": bits, "location_bits": location_bits(p), "harm_bound": p.harm_bound(bits),
            "exfiltration_bound": exfiltration_bound(p, bits)}


def per_unit(members: list[dict], statement: dict) -> dict[str, dict]:
    """Each member's session per drawn unit (prove_path's `per_unit`): C-Flock's ``prove_total_s``, the session's wall
    and Lean's verify, over the units the session proved; and a unit's ANDs and m."""
    out = {}
    for m in members:
        t = statement.get(m["name"]) or {}
        n, tm = t.get("drawn") or 0, t.get("timing") or {}
        pb = tm.get("proof_bytes")
        out[m["name"]] = {"units": n, "ands": m["ands"], "m": m["k_log"] + m["nbl"],
                          "proof_bytes": sum(pb) if isinstance(pb, list) else pb,
                          **{f"{x}_per_unit": round(v / n, 3) if n and isinstance(v, int | float) else None
                             for x, v in (("prove_s", tm.get("prove_total_s")), ("session_s", t.get("session_s")),
                                          ("lean_verify_s", t.get("lean_verify_s")))}}
    return out


def expected(name: str, res: dict) -> bool:
    """honest: accepted, every output 1; chain: the proofs held and the swapped buckets' ok bits 0; forged: refused by
    Lean's registered reads."""
    lean = {k.split("/", 1)[1]: v for k, v in res["verdicts"].items() if k.startswith("lean-verify/")}
    if ROUNDS[name] == "accepted":
        return res["window_accepted"] and all(lean.values())
    if ROUNDS[name] == "ok":
        ing = res["opened"].get("ingress") or []
        return res["accepted"] and all(lean.values()) and [t for t, w in enumerate(ing) if w == 0] == [A, A + 1, A + 2]
    why = res["why"].get("lean-verify/egress") or ""
    return not res["accepted"] and lean.get("egress") is False and "registered reads" in why and ROUNDS[name] in why


def blobs_of(members: list[dict]) -> tuple[dict[str, bytes], set[str]]:
    """The witness pieces the verifier must not receive: every unit's rows and every salt but ``o0``'s."""
    blobs = {}
    for m in members:
        for i, u in enumerate(m["units"]):
            for (r, _b), x in zip(ING_ROWS if m["name"] == "ingress" else EG_ROWS, u):
                blobs[f"{m['name']} {r} {i}"] = x if isinstance(x, bytes) else bytes(
                    b for w in _words(x) for b in w.to_bytes(2, "little"))
        blobs.update({f"{m['name']} salt {p} {i}": s for p, ss in m["salts"].items() if p != "o0"
                      for i, s in enumerate(ss)})
    return blobs, set(blobs)


def known(members: list[dict], at: SimpleNamespace) -> bytes:
    """What the scan leaves out: the circuits, zeros, and the window's pads and ENDs (fixed by the ordinals)."""
    cells = [IT.pad_cell(B, q) + IT.end_cell(q, B) for q in [0, *at.ordinals]]
    return b"".join((m["dir"] / "circuit.txt").read_bytes() for m in members) + bytes(WR.SALT_BYTES) + b"".join(cells)


def sparse(pieces: dict[str, bytes], held: set[str]) -> tuple[dict[str, bytes], set[str], int]:
    out = {k for k, x in pieces.items() if sum(map(bool, x)) <= 1}
    return {k: x for k, x in pieces.items() if k not in out}, held - out, len(out)


def prove(a) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    h = host(out / "warden", a.warden_run, a.draw)
    at, T = h.at, h.at.T
    ch = at.chain
    if a.draw is not None and not 2 <= a.draw <= 2 * T:
        raise SystemExit(f"--draw {a.draw}: a round's draw is 2 .. {2 * T} of its {2 * T} units (one a template at least)")
    ks = {"honest": a.draw, "chain": None, "forged": None if a.draw is None else stratum_k(a.draw, T, 2 * T)}
    summary: dict = {"source": at.source, "warden_run": at.run_id, "draw": ks if a.draw is not None else "every unit",
                     "class": asdict(CLASS), "params": asdict(at.params), "shape": asdict(SHAPE), "T": T,
                     "bits": {f"{side}/{r}": b for side, rows in (("ingress", ING_ROWS), ("egress", EG_ROWS))
                              for r, b in rows}, "admitted": at.admitted,
                     "anchors": {q: CLASS.anchor(s, s % CLASS.EVERY) for q, s in at.admitted.items()},
                     "responses": {q: len(x) for q, x in sorted(at.outputs.items())}, "run_s": at.run_s,
                     **({"swap": [A, A + 1]} if "chain" in h.rounds else {}),
                     "cells": {side: {"data": sum(r[s * B] == DATA for r in rows for s in range(SHAPE.R)),
                                      "end": sum(r[s * B] == W.END for r in rows for s in range(SHAPE.R))}
                               for side, rows in (("ingress", ch.ingress), ("egress", ch.egress))},
                     "roots": {k: r.root.hex() for k, r in at.trees.items()},
                     "host": {name: [{"side": m.side, "refused": m.refused, "outputs": m.outputs} for m in ms]
                              for name, ms in h.rounds.items()}}
    log(f"run: {at.source}; {T} buckets of link-window 0, admitted {at.admitted}, responses {summary['responses']}, "
        f"anchors {summary['anchors']}, cells {summary['cells']}; every egress bucket accepts its ingress bucket's "
        f"view; host reads as expected ({at.run_s} s); draw {summary['draw']}")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    if a.dry:
        log("dry: done")
        return 0
    M = WZ._one_stage()
    joint = program(M, ("ingress", "egress"), T, out / "joint")
    solo = program(M, ("egress",), T, out / "solo")
    summary["stagers"] = {f"{k}/{s}": {"unit": g.defn.id, "lower_s": g.lower_s, "k_log": g.st.meta["k_log"],
                                       "vus_per_block": g.st.meta["vus_per_block"], "pin": g.st.pin, "base": g.base}
                          for k, pg in (("joint", joint), ("solo", solo)) for s, g in pg.stagers.items()}
    log(f"stagers: {summary['stagers']}")
    ok = True
    with challenger(a, at, out) as coins:
        if coins is not None:
            summary["coins"] = {"challenger": coins.challenger, "window_root": coins.root, "rounds": ["honest"]}
            log(f"the honest round draws on the challenger's coins ({coins.challenger}), its session on the window's "
                f"root {coins.root[:16]}…; the forged round on the verifier's Lean's own")
        for j, (name, mrs) in enumerate(h.rounds.items()):
            pg = solo if name == "forged" else joint
            members = [stage(M, pg, mr, out / name / f"stage-{mr.side}") for mr in mrs]
            if name == "honest":
                same = [x["view"] for x in members[0]["bcs"]] == [x["state"] for x in members[1]["bcs"]]
                summary["view_bcs_shared"] = same
                if not same:
                    raise SystemExit("the ingress units' view rows aren't the egress units' state rows")
            summary.setdefault("rounds", {})[name] = {
                m["name"]: {"ands": m["ands"], "k_log": m["k_log"], "nbl": m["nbl"], "m": m["k_log"] + m["nbl"],
                            "stage_s": m["stage_s"], "outputs": m["outputs"]} for m in members}
            log(f"staged {name}: " + "; ".join(f"{m['name']} {m['ands']} ANDs a unit, m = {m['k_log']} + {m['nbl']}, "
                                               f"outputs {m['outputs']}" for m in members))
            (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
            av = SimpleNamespace(bin=a.bin, lean=a.lean, lease=[], gpu=False, delta=a.delta)
            trees = {**at.trees, **({"state": at.swapped} if name == "chain" else {})}
            trees = {k: v for k, v in trees.items() if name != "forged" or k == "view"}
            res = audit(M, av, at, pg, name, members, trees, out / name, a.port + 4 * j, ks[name],
                        coins if name == "honest" else None)
            blobs, held = blobs_of(members)
            blobs, held, public = LS.public_pieces(M.a8, blobs, held, known(members, at))
            blobs, held, nsparse = sparse(blobs, held)
            inst = out / name / "instances.bin"
            inst.write_bytes(b"".join((m["dir"] / f"inst-{m['n']}.bin").read_bytes() for m in members))
            res["received"] = {**M.a8.received(out / name / "verifier", inst, blobs, held),
                               "public_pieces": len(public), "sparse_pieces": nsparse}
            inst.unlink()
            rec = res["received"]
            good = expected(name, res) and not rec["verifier_found"] and rec["control"]["holds_all"]
            if coins is not None and name == "honest":
                good = good and bool(res.get("coin_negatives")) and all(res["coin_negatives"].values())
            ok &= good
            summary["rounds"][name]["audit"] = res
            log(f"round {name}: drawn {len(res.get('draw', {}).get('units') or [])} of {2 * T if pg is joint else T}, "
                f"charge {res.get('charge')}, per unit {res.get('per_unit')}; proofs {res['verdicts']}, audit "
                f"{res.get('audit')}, opened "
                f"{ {k: v for k, v in res.get('opened', {}).items()} }, Lean {res['why']}, verifier received "
                f"{rec['verifier_bytes']} bytes, witness pieces found there: {rec['verifier_found'] or 'none'} (of "
                f"{len(blobs)}; {len(public)} public, {nsparse} sparse left out), control "
                f"{rec['control']['blobs_found']}; {'as expected' if good else 'UNEXPECTED'}")
            (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    if coins is not None:
        pub = coins.published or {}
        summary["coins"]["session"] = {x: pub.get(x) for x in ("trusted", "error", "plan")}
        ok &= pub.get("error") is None and (pub.get("trusted") or {}).get("pinned_session") is None
        log(f"the challenger's session: {summary['coins']['session']}")
    summary["ok"] = ok
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    log("done: " + ("ok" if ok else "FAILED"))
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="scan_zk.py", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prove", help="a link-window's ingress and egress units chained by registered reads, --zk")
    p.add_argument("--bin")
    p.add_argument("--lean")
    p.add_argument("--out", required=True)
    p.add_argument("--port", type=int, default=7811)
    p.add_argument("--delta", type=float, default=2.0 ** -20)
    p.add_argument("--dry", action="store_true", help="the run and the host's read of every round: nothing proved")
    p.add_argument("--warden-run", type=Path,
                   help="a warden_serve run's directory (rehearse --ids at PARAMS): its link-window 0, verified under "
                        "its deployment's key, in place of the synthetic run")
    p.add_argument("--draw", type=int,
                   help="K: Lean draw --stratified K on the verifier's own coins, only the drawn units proved (no "
                        "chain round); default every unit")
    p.add_argument("--challenger", help="with --draw: the honest round's K drawn on the auditor's challenger's coins "
                   "(a5_coins, the round's service.Stream), not the verifier's: local, or gate:USER@HOST "
                   "(vy-challenger-1 through infra's gate, with --peer and --challenger-key)")
    p.add_argument("--peer", help="with --challenger gate:USER@HOST: this machine as the gate's /etc/vy/peers names it, "
                   "whose 127.0.0.1 the session's challenger listens on")
    p.add_argument("--challenger-key", help="with --challenger gate:USER@HOST: the challenger's key record of its "
                   "deployment (infra's challenger-key/v1), never the session's key.json")
    p.add_argument("--gate-key", help="with --challenger gate:USER@HOST: the lane's own gate key (default ssh's own)")
    p.add_argument("--main", help="with --challenger: the verifier's own copy of main's first-parent history (git "
                   "rev-list --first-parent origin/main), which must hold the commit the session's host pins")
    sub.add_parser("schedule", help="PARAMS as warden_serve.py's schedule flags (serve, rehearse): the warden runs "
                                    "--warden-run reads")
    a = ap.parse_args(argv)
    if a.command == "schedule":
        print(" ".join(f"--{k.replace('_', '-')} {v}" for k, v in asdict(PARAMS).items()))
        return 0
    if a.challenger:
        if a.challenger != "local" and not a.challenger.startswith("gate:"):
            p.error(f"--challenger is local or gate:USER@HOST, not {a.challenger!r}")
        if a.draw is None:
            p.error("--challenger draws the honest round's --draw K on the challenger's coins: give --draw")
        if a.challenger.startswith("gate:") and not (a.peer and a.challenger_key):
            p.error("--challenger gate:USER@HOST needs --peer and --challenger-key")
        if not a.main:
            p.error("--challenger needs --main: the verifier's git rev-list --first-parent origin/main, which must hold "
                    "the commit the session's host pins")
    if not a.dry and not (a.bin and a.lean):
        raise SystemExit("prove needs --bin and --lean, or --dry")
    return prove(a)


if __name__ == "__main__":
    raise SystemExit(main())
