"""A stand-in for N2's proof of an execution phase, for the ingress tie's of an ingress execution phase (`prove_tie`)
and for the egress tie's of one member of an execution phase's scan (`prove_scan`), for tests: what Flock's prover
behind the ZK server would deliver, and a stand-in for the verifier's own run of Flock's Lean verifier
(`proofs.verify`), which reads each session's verdict from the session file the stand-in prover writes. The proof's
outputs are the proof block's on the prover's rows by each gate's reference (`LinkWindows.outputs`,
`IngressTie.outputs`, `ingress_scan.reference`, `egress.outputs`), which the tests hold to the circuits: a
Lean-accepted proof can't claim others. So the tests check everything the verifier does around the proof: its run's findings, the public row
opened to the verifier's own copy, the output rows its registered reads open, the certificates and the sync."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from verity.ml.commitments import hm96
from verity.ml.commitments.merkle import tree_levels
from verity.ml.commitments.rowleaf import sha512_row_v2_digest
from verity.core.protocols.certifier import scan_service as SC
from verity.core.protocols.certifier import service as SVC, tie_service as TS, warden_rows as WR
from verity.core.protocols.spot import service as S
from verity.core.protocols.spot.one_stage import partition as P, proofs as PF, registered as G
from verity.core.protocols.spot.one_stage.proofs import Finding
from verity.ml.protocols.certifier import egress as EG, ingress_scan as IS

_INST = hm96.SHA512
SALTS = {"pub": b"\x51" * 192, "wit": b"\x52" * 192, "o0": b"\x53" * 192}


def _bc(data: bytes, bits: int, salt: bytes) -> bytes:
    return _INST.commit_string(_INST.default_key, sha512_row_v2_digest(data, bits), salt)


def prove(lw: SVC.LinkWindows, opening: SVC.Hidden, draw: S.Draw, syncs: Sequence[Any], out: Path, *,
          program: str | None = None, accepted: bool = True, word: int | None = None, root: bytes | None = None,
          salts: dict[str, bytes] | None = None, units: Sequence[int] | None = None) -> SVC.Delivered:
    """The delivered proof of the sampled proof block on `opening` under the registered root (or `root`): the
    statement's public file, its one instance's rows committed under `SALTS` (``pub``, ``wit``, ``o0``), one session
    naming `program` (default: the link's registered Program) that the stand-in Lean `accepted`, and the circuit file.
    Where the unit takes no instance, the rows are zero and the output 0. `word` forges the output word,
    `salts` the salts delivered for ``pub`` and ``o0``, and `units` the public file's unit indices."""
    root = bytes.fromhex(draw.registration.record["roots"][0]["root"]) if root is None else root
    args = lw.arguments(opening, root, syncs)
    rows = {r: lw.row(args, r) if args is not None else bytes(lw.bits(r) // 8) for r in SVC.ROWS}
    rows["o0"] = SVC.output_row(lw.outputs(opening, root, syncs) if args is not None else 0)
    return _deliver(lw, rows, draw, out, program=program, accepted=accepted, word=word, salts=salts, units=units)


def _deliver(unit: Any, rows: dict[str, bytes], draw: S.Draw, out: Path, *, program: str | None, accepted: bool,
             word: int | None, salts: dict[str, bytes] | None, units: Sequence[int] | None) -> SVC.Delivered:
    """The statement's files for `rows` on `unit` (a `LinkWindows` or an `IngressTie`) and what the prover delivers."""
    out.mkdir(parents=True, exist_ok=True)
    public = out / "public.bin"
    header = {"units": {"indices": list(draw.units if units is None else units)}, "instances": 1}
    public.write_bytes(json.dumps(header).encode() + b"\n"
                       + b"".join(_bc(rows[p], unit.bits(p), SALTS[p]) for p in SVC.PORTS))
    session = out / "session-0"
    session.mkdir(exist_ok=True)
    (session / "session.json").write_text(json.dumps({"accepted": accepted, "units": [0],
                                                      "program": program or unit.partition["program"]}))
    circuit = out / "circuit.txt"
    circuit.write_text(f"stand-in circuit of {unit.unit.id}\n")
    delivered = {"pub": SALTS["pub"], "o0": SALTS["o0"], **(salts or {})}
    output = int.from_bytes(rows["o0"], "little") if word is None else word
    return SVC.Delivered(PF.Proof(circuit, public, (session,)), delivered, output)


#: the domains by id, as `prove_tie` and `prove_scan` commit under them (the network certifier's, the committed trace's
#: and the known values'): the stand-in Lean's registered reads climb each port's rows under the domain its id names,
#: which the real verifier derives from the port
DOMAINS: dict[str, Any] = {}


def prove_tie(tie: Any, cells: Any, draw: S.Draw, lanes: Any, out: Path, *, program: str | None = None,
              accepted: bool = True, words: dict[int, int] | None = None,
              salts: dict[int, dict[str, bytes]] | None = None, units: Sequence[int] | None = None,
              rows: Sequence[bytes] | None = None) -> TS.Delivered:
    """The delivered proof of the sampled ingress-tie proof blocks (`tie_service.IngressTie`) on the network certifier's
    opening `cells` (`warden_rows.Cells`) and the prover's `lanes` (each bucket's), as `prove` delivers N2's: proof
    block t's ``wit`` row its
    bucket row under the warden's salt and its header's registered read of the warden's root (`Cells.statement`), its
    ``pub`` row its lanes and its ``o0`` row its output bit by IoTieRequest's reference (`io_request.reference`),
    which the tests hold to the circuit, each read off the root of the prover's rows of the port as the verifier
    registers its own (`IngressTie.known_tree`, under their public salts). `rows` are the rows the prover proves on
    instead, under the warden's salts and paths; where a unit takes no instance of its lanes, its public row is zero
    and its output 0. `words` forges output words, `salts` the salts delivered for ``pub`` and ``o0``, and `units`
    the public file's unit indices. The public file is a shared-row file as `verity_flock.circuit.write` lays one out:
    each port's table, one row an instance, port-major, then the instances' refs."""
    rows = list(cells.rows if rows is None else rows)
    indices = list(draw.units if units is None else units)
    got, outputs = [], {}
    for u in range(tie.T):
        args = None if lanes is None else tie.arguments(rows[u], lanes[u])
        word = tie.outputs(rows[u], lanes[u]) if args is not None else 0
        outputs[u] = word if words is None or u not in words else words[u]
        got.append({"pub": tie.row(args, "pub") if args is not None else bytes(tie.bits("pub") // 8), "wit": rows[u],
                    "o0": SVC.output_row(word)})
    known = {p: tie.known_tree(cells.head, p, [g[p] for g in got]) for p in TS.KNOWN}
    DOMAINS.update({d.domain_id.hex(): d for d in (cells.domain, *(v.domain for v in known.values()))})
    row_salts = {"wit": cells.salts, **{p: v.salts for p, v in known.items()}}
    entries = {TS.REGISTERED: cells.statement(indices)[2], **{p: v.statement(indices)[2] for p, v in known.items()}}
    tables = {p: [_bc(got[u][p], 8 * len(got[u][p]), row_salts[p][u]) for u in indices] for p in SVC.PORTS}
    outputs = {u: outputs[u] for u in indices}
    out.mkdir(parents=True, exist_ok=True)
    public, n = out / "public.bin", len(indices)
    header = {"units": {"indices": indices}, "instances": n, "registered": entries,
              "shared_rows": {p: n for p in SVC.PORTS}}
    refs = b"".join(i.to_bytes(4, "little") for i in range(n) for _ in SVC.PORTS)
    public.write_bytes(json.dumps(header).encode() + b"\n" + b"".join(b"".join(tables[p]) for p in SVC.PORTS) + refs)
    session = out / "session-0"
    session.mkdir(exist_ok=True)
    (session / "session.json").write_text(json.dumps({"accepted": accepted, "units": list(range(len(indices))),
                                                      "program": program or tie.partition["program"]}))
    circuit = out / "circuit.txt"
    circuit.write_text(f"stand-in circuit of {tie.unit.id}\n")
    delivered = {u: {"pub": row_salts["pub"][u], "o0": row_salts["o0"][u], **((salts or {}).get(u) or {})}
                 for u in indices}
    return TS.Delivered(PF.Proof(circuit, public, (session,)), delivered, outputs)


def _bits(row: Sequence[int]) -> list[int]:
    """A registered row of u16 words as its bits, low bit first."""
    return [w >> i & 1 for w in row for i in range(16)]


def _view(sh: EG.Shape, bits: Sequence[int]) -> EG.State:
    """An egress view row read back, in `egress.state_bits`' layout, as `EgressBucket` reads its fields."""
    def take(at: int, w: int) -> int:
        return sum(int(bits[at + i]) << i for i in range(w))

    return EG.State(take(0, 32), tuple(
        EG.Slot(take(a + EG.LIVE, 16), take(a + EG.Q, 32), take(a + EG.ADMITTED, 32), take(a + EG.N, 32),
                tuple(take(a + EG.IDS + k * sh.IB, sh.IB) for k in range(sh.K)))
        for a in map(sh.slot_at, range(sh.NR))))


def _bc_words(words: Sequence[int], salt: bytes) -> bytes:
    return _INST.commit_string(_INST.default_key, G.row_digest(words, G.WORD_BITS, SC.SCHEMA), salt)


def prove_scan(tie: SC.EgressTie, cells: SC.Heads, state: G.Registered, progs: Sequence[G.Registered],
               view: G.Registered, last: G.Registered, member: SC.Member, out: Path, *, accepted: bool = True,
               program: str | None = None) -> PF.Proof:
    """The delivered proof of `member` of an execution phase's egress tie (`scan_service.EgressTie`): each block's input
    rows the registered rows it reads, `cells` the network certifier's openings of both links' rows (`warden_rows.Cells`, as
    `Heads`), `state` and `progs` the committed trace's trees and `view` and `last` the prover's; its outputs by the
    templates' references (`ingress_scan.reference`, `egress.outputs` on the view row read back), each committed under
    the salt of the registered row the verifier's read of it opens (an output the reference gives none of, where ok is 0,
    zero bytes, which open no leaf), and every ``o0`` under the known tree's public salts. The public file is a
    shared-row file as `verity_flock.circuit.write` lays one out: each port's table in the circuit's port order, port-major,
    a slot-entries port one row a boundary its units read (`registered.table`), then the instances' refs."""
    cls, sh, every, bs = tie.cls, tie.cls.egress, tie.cls.EVERY, list(member.buckets)
    heads = SC.Heads(cells.ingress.head, cells.egress.head)
    known = tie.known_tree(heads)
    DOMAINS.update({d.domain_id.hex(): d for d in (cells.ingress.domain, cells.egress.domain, state.domain, view.domain,
                                                    last.domain, known.domain, *(p.domain for p in progs))})
    identity = list(range(len(bs)))
    ports: dict[str, tuple[list[bytes], dict[str, Any], list[int]]] = {}

    def read(name: str, tree: G.Registered, positions: Sequence[int]) -> None:
        table, refs = G.table(positions)
        ports[name] = ([tree.bcs[p] for p in table], tree.statement(table)[2], refs)

    def made(name: str, tree: G.Registered, positions: Sequence[int], rows: Sequence[Sequence[int] | None]) -> None:
        ports[name] = ([bytes(SVC.BC_BYTES) if r is None else _bc_words(SC.words(r), tree.salts[p])
                        for p, r in zip(positions, rows)], tree.statement(positions)[2], identity)

    if member.template == "egress":
        got = [EG.outputs(sh, _view(sh, _bits(view.rows[b]))) for b in bs]
        read("state", view, bs)
        oks = [ok for ok, _ in got]
        ports["o1"] = ([WR.bc(row, cells.egress.salts[b]) for b, (_, row) in zip(bs, got)],
                       cells.egress.statement(bs)[2], identity)
        order = ["state", "o0", "o1"]
    else:
        ingress = member.template == "ingress"
        got = [IS.reference(cls, cells.ingress.rows[b], _bits(state.rows[b]) if ingress else None,
                            [x for p in progs for x in _bits(p.rows[b // every])]) for b in bs]
        ports["row"] = ([cells.ingress.bcs[b] for b in bs], cells.ingress.statement(bs)[2], identity)
        if ingress:
            read("sin", state, bs)
        for j, p in enumerate(progs):
            read(f"prog{j}", p, [b // every for b in bs])
        oks = [ok for ok, _, _ in got]
        if member.name == "last":
            made("o1", last, [0], [s for _, s, _ in got])
        else:
            made("o1", state, [b + 1 for b in bs], [s for _, s, _ in got])
        made("o2", view, bs, [v for _, _, v in got])
        order = ["row", *(["sin"] if ingress else []), *(f"prog{j}" for j in range(len(progs))), "o0", "o1", "o2"]
    ports["o0"] = ([_bc_words([ok], known.salts[u]) for u, ok in zip(member.units, oks)],
                   known.statement(member.units)[2], identity)
    out.mkdir(parents=True, exist_ok=True)
    public, n = out / "public.bin", len(bs)
    header = {"units": {"indices": list(member.units)}, "instances": n,
              "registered": {p: ports[p][1] for p in order}, "shared_rows": {p: len(ports[p][0]) for p in order}}
    refs = b"".join(ports[p][2][i].to_bytes(4, "little") for i in range(n) for p in order)
    public.write_bytes(json.dumps(header).encode() + b"\n" + b"".join(b"".join(ports[p][0]) for p in order) + refs)
    session = out / "session-0"
    session.mkdir(exist_ok=True)
    (session / "session.json").write_text(json.dumps({"accepted": accepted, "units": identity,
                                                      "program": program or tie.partition["program"]}))
    circuit = out / "circuit.txt"
    circuit.write_text(_circuit_text(tie, member.template))
    return PF.Proof(circuit, public, (session,))


def _circuit_text(tie: SC.EgressTie, template: str) -> str:
    return f"stand-in circuit of {P.template_id(tie.templates[template])}\n"


def scan_circuits(tie: SC.EgressTie) -> dict[str, str]:
    """The auditor's circuit file's SHA-512 of each template, as `EgressTie.verify` takes them: the stand-in's own."""
    return {tp: hashlib.sha512(_circuit_text(tie, tp).encode()).hexdigest() for tp in SC.TEMPLATES}


def circuit_sha512(delivered: SVC.Delivered) -> str:
    return PF.file_sha512(delivered.proof.circuit)


def _known(registered: dict) -> str | None:
    """The stand-in for Lean's check of the ports a `--registered` file names at known content
    (`Flock.Registered.checkKnownAll`, which refuses the file before any statement): each port's root the root of its
    known rows under their salts; None when they hold."""
    for name, port in registered["ports"].items():
        known, dom = port.get(SC.KNOWN_KEY), DOMAINS.get(port["domain"])
        if known is None:
            continue
        rows, salts = known["rows"], [bytes.fromhex(s) for s in known["salts"]]
        if dom is None or len(rows) != len(salts) or len(rows) != port["leaves"]:
            return f"registered reads: port {name}: {len(rows)} known rows of {port['value']}'s {port['leaves']}"
        bcs = [_INST.commit_string(_INST.default_key, G.row_digest(r, G.WORD_BITS, SC.SCHEMA), s)
               for r, s in zip(rows, salts)]
        if tree_levels(dom, [G.leaf(dom, i, b, SC.SCHEMA) for i, b in enumerate(bcs)])[-1][0].hex() != port["root"]:
            return f"registered reads: port {name}: {port['value']}'s registered root is not the root of its known rows"
    return None


def _port(port: dict, dom: Any, entry: dict, table: Sequence[bytes], refs: Sequence[int],
          indices: Sequence[int]) -> str | None:
    """Lean's read of one registered port (`Flock.Registered.checkPort`, in its order): the entry names the port's
    value, each table row opens the registered root at its stated position, and each instance's ref names a table row
    at the position the verifier's program reads for its unit (`Port.readAt`: none below ``reads_from``)."""
    if (port.get("domain"), port.get("leaves")) != (dom.domain_id.hex(), dom.count):
        raise ValueError("the port names another domain or leaf count than `dom`, and the Lean verifier climbs the port's")
    if entry.get("value") != port["value"]:
        return f"registered reads: the port reads {port['value']}, not another value ({entry.get('value')})"
    positions, paths = list(entry.get("positions", ())), list(entry.get("paths", ()))
    if len(paths) != len(table) or len(positions) != len(table):
        return "registered reads: one position and one path per table row"
    root, schema = bytes.fromhex(port["root"]), port.get(G.SCHEMA_KEY, G.SCHEMA)
    for r, (p, b, path) in enumerate(zip(positions, table, paths)):
        if not G.verify_read(dom, root, p, b, bytes.fromhex(path), schema):
            return f"registered reads: table row {r} does not open the registered root at position {p}"
    base, reads = int(port.get(G.READS_FROM, 0)), list(port["reads"])
    for i, (ref, u) in enumerate(zip(refs, indices)):
        if ref >= len(table):
            return f"registered reads: instance {i}'s ref names no row of its table"
        want = None if u < base or u - base >= len(reads) else reads[u - base]
        if want is None:
            return f"registered reads: the verifier's program reads no row of {port['value']} in unit {u}"
        if positions[ref] != want:
            return f"registered reads: unit {u} reads position {positions[ref]}, and the verifier's program reads {want}"
    return None


def _registered(public: Path, registered: dict) -> str | None:
    """The stand-in for Lean's registered reads (`Flock.Registered.check`): every port of the statement (its shared-row
    tables, in the circuit's port order) one of the verifier's (`checkComplete`), the header naming none the verifier's
    don't, and each of the verifier's ports read by the statement's header entry (`_port`), its table rows the public
    file's commitments (port-major) and each instance's ref its row of that table (u32s after the tables,
    instance-major); None when they hold."""
    data = public.read_bytes()
    head, body = data.split(b"\n", 1)
    header = json.loads(head)
    shared = header.get("shared_rows")
    circuit = list(SVC.PORTS if shared is None else shared)
    for p in circuit:
        if p not in registered["ports"]:
            return (f"registered reads (complete): the statement's port {p} is no port of the verifier's registered "
                    "reads, which name every port the statement's outputs read")
    if shared is None:
        return "registered reads: a registered port is a shared-row file's table"
    entries = header.get("registered") or {}
    extra = next((p for p in entries if p not in registered["ports"]), None)
    if extra is not None:
        return f"registered reads: the statement names port {extra} registered, and the verifier's registered reads do not"
    n, indices, np_ = int(header["instances"]), list(header["units"]["indices"]), len(circuit)
    starts = [sum(shared[p] for p in circuit[:q]) for q in range(np_)]
    at = sum(shared.values()) * SVC.BC_BYTES
    refs = [[int.from_bytes(body[at + 4 * (i * np_ + q):at + 4 * (i * np_ + q + 1)], "little") for i in range(n)]
            for q in range(np_)]
    for name, port in registered["ports"].items():
        entry, dom = entries.get(name), DOMAINS.get(port["domain"])
        if entry is None or dom is None:
            return f"registered reads: the statement reads no registered {name} under the verifier's domain"
        if name not in circuit:
            return f"registered reads: {name} is not a row of the circuit (an input port, or an output port's row)"
        q = circuit.index(name)
        table = [body[(starts[q] + r) * SVC.BC_BYTES:(starts[q] + r + 1) * SVC.BC_BYTES] for r in range(shared[name])]
        why = _port(port, dom, entry, table, refs[q], indices)
        if why is not None:
            return why.replace("registered reads: ", f"registered reads: port {name}: ", 1)
    return None


def lean(proof: PF.Proof, *, circuit_sha512: str, program: bytes, partition: dict,
         registered: dict | None = None) -> tuple[list[Finding] | None, tuple[str, str] | None]:
    """`proofs.verify`'s stand-in: its refusal of a circuit file other than the committed one, or of a `registered`
    file whose known content isn't its roots' (`_known`), then each session's finding as its session file says
    (`prove`), with the units the public file's indices name, each rejected when the verifier's `registered` reads
    don't hold (`_registered`)."""
    assert isinstance(program, bytes) and "program" in partition
    if PF.file_sha512(proof.circuit) != circuit_sha512:
        return None, ("circuit", "the statement's circuit file is not the one the auditor committed")
    if registered is not None and (bad := _known(registered)) is not None:
        return None, ("registered", bad)
    why = None if registered is None else _registered(proof.public, registered)
    out = []
    for s in proof.sessions:
        v = json.loads((s / "session.json").read_text())
        indices = json.loads(proof.public.read_bytes().split(b"\n", 1)[0])["units"]["indices"]
        ok = v["accepted"] and why is None
        out.append(Finding(str(s), ok, "" if ok else why or "rejected", v["program"], {},
                           tuple(indices[u] for u in v["units"]) if ok else ()))
    return out, None


def decide(lw: SVC.LinkWindows, stream: S.Stream, draw: S.Draw, opening: SVC.Hidden, syncs: Sequence[Any], out: Path,
           *, certificates: Sequence[Any] | None = None, **kw: Any) -> S.Outcome:
    """The sampled link-window proved on `opening` (`prove`, with `kw`), delivered and decided by the verifier
    on the proof and the certificates (by default the opening's), with the opening's in-the-clear check beside it as the
    debug comparison, which must agree with the proof's verdict when nothing is forged."""
    delivered = prove(lw, opening, draw, syncs, out, **kw)
    assert stream.deliver(draw, delivered.sessions()) is None
    certificates = [r.certificate for r in opening.records] if certificates is None else certificates
    outcome = lw.outcome(stream, draw, certificates, syncs, delivered, circuit_sha512=circuit_sha512(delivered),
                         opening=opening)
    verdict = stream.records[-1]["audit"]["verdicts"][SVC.VERIFIER]
    assert verdict["decided_by"] == "proof" and verdict["accepted"] == outcome.accepted
    if not kw:
        assert verdict["clear"]["accepted"] == outcome.accepted
    return outcome
