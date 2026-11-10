"""The two-stage audit's toy: a replay block of five templates, two of them link templates, its boundary registration (what
serving commits) and its interior registration from replay (what the proof server commits), laid out as the second stage
proves them (`audit.py`).

* **The replay block** `REPLAY` (`TwoStageToyReplay{H}`): a span's ingress u (2H bits) in, its egress ~(u[0:H] & u[H:2H])
  out, in six proof blocks, every root node an instance of one of `TEMPLATES`, each called once: `LINK_IN` (b_in, u through
  two NOTs a bit), `FOLD` (the halves' AND, H AND gates), `FLIP` (a NOT a bit), `FLIP_B` (Flip's body under a name of its
  own, a batch of two instances each reading FLIP's row, which nothing reads: the one cell of two proof blocks, so the
  second stage samples within a replay block) and `LINK_OUT` (b_out, FLIP's row through two NOTs a bit). Every edge is one
  instance's whole row (`WRITERS`), so the interior wires each one, each from one writer.
* **The boundary registration** (`boundary_window`, `boundary`): N replay blocks, one member of REPLAY's instances under the
  replay block's query, its ``ingress`` and ``o0`` ports a row a span (`BOUNDARY_PORTS`), under fresh salts.
* **The interior registration** (`interior_window`, `interior`): the replay sample S's Q·n_rho proof blocks, template-major,
  span S[i]'s j-th instance of template t at slot off_t + i·n_t + j (off_t = Q·Σ_{t'<t} n_t'), one member a template, under
  the interior run (`interior_run`). Each internal edge is a wire (`rows.Member.wired`) with no tree of its own
  (`Member.wired_trees`): a reader's row is its writer's output value, a leaf of the writer's tree under that row's salt,
  so the interior commits each value once (`Binding.held`). Each span is replayed from its ingress, the boundary record's
  row at S[i]. Every member is a shared-row member and every port of it a registered read
  (`verity.ml.windows.links.Port`), so Lean opens every row a session proves: a link port (b_in's input, b_out's output)
  at the boundary record, its rows committed under the boundary's salts at S[i] so an honest replay's ``b || c`` are the
  boundary's, its header naming the boundary port, the leaves S and each one's path to the boundary root; a wired port at
  its writer's port of the interior record, its table the writer's output rows (table row r at leaf r, as the header
  names them) and unit i reading its writer unit's leaf; every other port at the interior record, unit i reading its own
  leaf i.
* **The verifier's reads** (`reads`): each member's ``--registered`` file from the two registration records and S alone.
* **The auditor's params** (`references`), fixed before the run: each template's circuit file and its reference
  statement's header, which round 2 holds the template's statements to once it places it in the run
  (`second_stage.at_run`).
* **The challenger's design** (`spec`): `service.TwoStage` over REPLAY's instances, the boundary ports committed while
  serving and the interior's after the first stage, each sampled proof block checked as an instance of its template, for a
  `service.Stream` over `TEMPLATES` with ``replay=REPLAY``, whose cells are `N_T` and whose interior program is the
  interior window's.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from verity.core.protocols.spot import service as S
from verity.core.protocols.spot.one_stage import partition as P, registered as REG, registration as R
from verity.ml.circuits import boolean as BL
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.program import Program
from verity.ml.circuits.types import Array
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2
from verity.ml.windows import links as LK, rows as CS

#: a row's hm96 hash compresses a block before its last only from 112 bytes on, and Flock's unit statement needs one such row
H = 1024
WORD = 16
#: the toy's boundary run, and the tag the interior's run is taken under
RUN = hashlib.sha512(b"verity/two-stage/toy/v1").hexdigest()
RUN_TAG = b"verity/two-stage/toy/interior-run/v1\x00"
#: the salts' key of each reference statement's stand-in rows, so the params are one file for every run of the design
REFERENCE_KEY = hashlib.sha512(b"verity/two-stage/toy/reference/v1").digest()


def _identity_signature(S):
    return (("row", Array(S.W, BL.BIT)),), Array(S.W, BL.BIT)


def _identity_body(B, S, row):
    return B.batch(BL.NOT, B.batch(BL.NOT, row, axes=(0,)), axes=(0,))


LinkIn = CompositeDefinition(
    "TwoStageToyLinkIn", 1, ("W",), _identity_signature, _identity_body, conformance="exact-model-tested", register=False,
    doc="TwoStageToyLinkIn{W}: b_in, a span's W-bit ingress row through two NOTs a bit: the link template whose row opens "
        "the boundary record's ingress at the span's leaf")
LinkOut = CompositeDefinition(
    "TwoStageToyLinkOut", 1, ("W",), _identity_signature, _identity_body, conformance="exact-model-tested", register=False,
    doc="TwoStageToyLinkOut{W}: b_out, a W-bit row through two NOTs a bit: the link template whose output opens the boundary "
        "record's egress at the span's leaf")
Fold = CompositeDefinition(
    "TwoStageToyFold", 1, ("H",), lambda S: ((("x", Array(2 * S.H, BL.BIT)),), Array(S.H, BL.BIT)),
    lambda B, S, x: B.batch(BL.AND, x.reshape(Array(2, Array(S.H, BL.BIT))), axes=(1,)), conformance="exact-model-tested",
    register=False, doc="TwoStageToyFold{H}: a 2H-bit row's halves' AND")
def _flip_signature(S):
    return (("x", Array(S.H, BL.BIT)),), Array(S.H, BL.BIT)


def _flip_body(B, S, x):
    return B.batch(BL.NOT, x, axes=(0,))


Flip = CompositeDefinition(
    "TwoStageToyFlip", 1, ("H",), _flip_signature, _flip_body, conformance="exact-model-tested", register=False,
    doc="TwoStageToyFlip{H}: an H-bit row, each bit NOT")
FlipB = CompositeDefinition(
    "TwoStageToyFlipB", 1, ("H",), _flip_signature, _flip_body, conformance="exact-model-tested", register=False,
    doc="TwoStageToyFlipB{H}: Flip's body under a name of its own, the replay block's batch of two flips of FLIP's row, so "
        "that each template is called once and the second stage samples a cell of two proof blocks")


def _replay_body(B, S, ingress):
    h = S.H
    a = B.call(bind(LinkIn, W=2 * h), ingress)
    f = B.call(bind(Fold, H=h), a)
    g = B.call(bind(Flip, H=h), f)
    B.batch(bind(FlipB, H=h), g, axes=(None,), n=2)
    return B.call(bind(LinkOut, W=h), g)


Replay = CompositeDefinition(
    "TwoStageToyReplay", 1, ("H",), lambda S: ((("ingress", Array(2 * S.H, BL.BIT)),), Array(S.H, BL.BIT)), _replay_body,
    conformance="exact-model-tested", register=False,
    doc="TwoStageToyReplay{H}: a span's 2H-bit ingress to its H-bit egress, the halves' NAND, in six proof blocks: b_in, "
        "a fold, a flip, a batch of two flips of the flip's row and b_out")

LINK_IN, FOLD, FLIP, FLIP_B, LINK_OUT, REPLAY = (bind(LinkIn, W=2 * H), bind(Fold, H=H), bind(Flip, H=H),
                                                 bind(FlipB, H=H), bind(LinkOut, W=H), bind(Replay, H=H))
TEMPLATES = (LINK_IN, FOLD, FLIP, FLIP_B, LINK_OUT)
#: each template's instances in one replay block, and each one's rows (input, output) in 16-bit words
N_T = (1, 1, 1, 2, 1)
N_RHO = sum(N_T)
LAYOUT = (((("row", 2 * H // WORD),), (("o0", 2 * H // WORD),)), ((("x", 2 * H // WORD),), (("o0", H // WORD),)),
          ((("x", H // WORD),), (("o0", H // WORD),)), ((("x", H // WORD),), (("o0", H // WORD),)),
          ((("row", H // WORD),), (("o0", H // WORD),)))
#: each link template's port that opens the boundary record, and the boundary port it opens
LINKS = {0: ("row", "ingress"), 4: ("o0", "o0")}
#: each template's writer, the template whose one instance's output its input row is in each span (None: b_in's, a link)
WRITERS = (None, 0, 1, 2, 2)
BOUNDARY_ROWS = (("ingress", 2 * H // WORD),)
BOUNDARY_OUTS = (("o0", H // WORD),)


def ingress(n: int, seed: bytes = b"toy") -> np.ndarray:
    """``n`` spans' ingress rows, ``(n, 2H/16)`` u16, from ``seed``."""
    return np.stack([np.frombuffer(hashlib.shake_256(b"two-stage-toy/ingress/%s/%d" % (seed, i)).digest(2 * H // 8),
                                   dtype="<u2") for i in range(n)]).astype(np.uint16)


def replay(u: np.ndarray) -> dict[str, np.ndarray]:
    """REPLAY on rows ``u`` (``(n, 2H/16)`` u16), each proof block's rows: ``fold`` (u's halves' AND), ``flip`` (FLIP's),
    ``flip_b`` (each of FLIP_B's two) and ``egress`` (flip's, through b_out)."""
    u = np.asarray(u, dtype=np.uint16)
    fold = u[:, :H // WORD] & u[:, H // WORD:]
    flip = ~fold
    return {"ingress": u, "fold": fold, "flip": flip, "flip_b": ~flip, "egress": flip}


def one_bit_off(rows: np.ndarray) -> np.ndarray:
    """Ingress rows ``rows`` (``(q, 2H/16)`` u16) with the first one bit off: the last bit of its upper half whose
    lower-half bit is set, which the fold's AND carries to the egress, so both its link rows leave the boundary's."""
    out = np.asarray(rows, dtype=np.uint16).copy()
    j = H + int(np.flatnonzero(bits(out[0])[:H])[-1])
    out[0, j // WORD] ^= np.uint16(1 << (j % WORD))
    return out


def bits(words: np.ndarray) -> np.ndarray:
    """Rows of 16-bit words as their bits, word w's bit b at w·16 + b: the layout `replay` and the circuits agree on."""
    w = np.asarray(words, dtype=np.uint16)
    return ((w[..., :, None] >> np.arange(WORD, dtype=np.uint16)) & 1).reshape(*w.shape[:-1], w.shape[-1] * WORD).astype(np.uint8)


# --- the boundary registration ----------------------------------------------------------------------------------------

def boundary_window(n: int, q: int, run: str = RUN) -> CS.Binding:
    """The boundary window: ``n`` replay blocks, the first stage drawing ``q`` of them."""
    program, query = CS.population_program([(REPLAY, n)])
    (rid, _, _), = P.ranges(program, query)
    m = CS.Member(rid, n=n, rows=BOUNDARY_ROWS, outs=BOUNDARY_OUTS, schemas=tuple((p, V2) for p, _ in BOUNDARY_ROWS + BOUNDARY_OUTS))
    return CS.Binding(program, query, {"law": "subset", "k": q}, run, (m,))


def boundary_ports(window: CS.Binding) -> dict[str, str]:
    """The boundary record's port of each boundary row name (``ingress``, ``o0``)."""
    m, = window.members
    return {p: m.port(p) for p, _ in BOUNDARY_ROWS + BOUNDARY_OUTS}


def spec(n: int, q: int, k: int | Mapping[str, int], proof_deadline: float = 3600.0) -> S.Spec:
    """The challenger's two-stage design over ``n`` spans: the first stage samples ``q`` of them, the second ``k`` proof
    blocks in each (replay block, template) cell (an int, or one count a template id), each proof by ``proof_deadline``
    seconds after its sample."""
    names = {}
    for phase, w in (("serve", boundary_window(n, q)), ("interior", interior_window("00" * 64, range(q)))):
        names |= {m.port(p): (m.schema(p), phase) for m in w.members for p in CS.port_leaves(m)}
    public = (S.Public("population", "structure", "N, the sample's range"),
              S.Public("window", "structure", "provenance: the two-stage toy"),
              S.Public("leaf_layer", "structure", "the members' public bodies' digest"),
              *(S.Public(f"{x}:{p}", "structure", f"{p}'s layout") for p in sorted(names) for x in ("leaves", "domain")))
    return S.Spec(tuple(S.Value(p, s, phase) for p, (s, phase) in sorted(names.items())),
                  S.TwoStage(P.template_instance_query(REPLAY), f"rate:{q}/{n}", k),
                  S.Check(tuple(P.program_digest(Program(t)) for t in TEMPLATES), P.template_instances_query(TEMPLATES),
                          proof_deadline=proof_deadline),
                  public, RUN_TAG)


def links(window: CS.Binding) -> list[dict[str, str]]:
    """The design's links as the verifier takes them: each link template of the interior ``window``, its session port that
    reads the boundary record, and the boundary port it reads (``{"template", "port", "value"}``)."""
    names = boundary_ports(boundary_window(1, 1))
    return [{"template": window.members[t].template, "port": p, "value": names[b]} for t, (p, b) in LINKS.items()]


def boundary(u: np.ndarray, q: int, *, salt_key: bytes | None = None) -> CS.Committed:
    """Serving's commitment of the spans ``u``: each span's ingress and egress under fresh salts (``salt_key``: tests only)."""
    win = boundary_window(len(u), q)
    return CS.commit(win, [{"rows": {"ingress": np.asarray(u, dtype=np.uint16)}, "outs": {"o0": replay(u)["egress"]}}],
                     salt_key=salt_key)


# --- the interior registration ----------------------------------------------------------------------------------------

def interior_run(boundary_digest: str, sample: Sequence[int]) -> str:
    """The interior window's run: SHA-512 over the boundary registration's digest and the replay sample, in order."""
    body = json.dumps({"boundary": boundary_digest, "replay_sample": [int(u) for u in sample]}, separators=(",", ":"))
    return hashlib.sha512(RUN_TAG + body.encode()).hexdigest()


def offsets(q: int) -> list[int]:
    """off_t, each template's first slot for ``q`` spans."""
    return [q * sum(N_T[:t]) for t in range(len(TEMPLATES))]


def wires(q: int) -> list[tuple[tuple[str, str, tuple[int, ...]], ...]]:
    """Each member's wires for ``q`` spans: REPLAY's edges, each instance of a template in span i reading its writer's
    (`WRITERS`) output row there."""
    off = offsets(q)
    return [() if w is None else ((LAYOUT[t][0][0][0], "o0", tuple(off[w] + i * N_T[w] for i in range(q)
                                                                    for _ in range(N_T[t]))),)
            for t, w in enumerate(WRITERS)]


def interior_window(boundary_digest: str, sample: Sequence[int], law: Mapping | None = None) -> CS.Binding:
    """The interior window of the replay sample ``sample`` after the boundary registration ``boundary_digest``: one member a
    template, template-major, each wired port its writer's output value with no tree of its own (`rows.Binding.held`), its
    table the writer's Q·n_w output rows; ``law`` the interior registration's (default: every unit, ``subset:Q·n_rho``)."""
    q = len(sample)
    program, query = CS.population_program([(t, q * n) for t, n in zip(TEMPLATES, N_T)])
    ids = [tid for tid, _, _ in P.ranges(program, query)]
    members = []
    for tid, n_t, (rows, outs), wired, base, w in zip(ids, N_T, LAYOUT, wires(q), offsets(q), WRITERS):
        table = q * (n_t if w is None else N_T[w])
        members.append(CS.Member(tid, rows=rows, outs=outs, n=q * n_t, base=base, wired=wired, wired_trees=False,
                                 tables=tuple((p, table) for p, _ in rows), schemas=tuple((p, V2) for p, _ in rows + outs)))
    return CS.Binding(program, query, dict(law or {"law": "subset", "k": q * N_RHO}), interior_run(boundary_digest, sample),
                      tuple(members))


def link(t: int) -> str | None:
    """Template ``t``'s port that opens the boundary record, if it is a link template."""
    return LINKS[t][0] if t in LINKS else None


@dataclass(frozen=True)
class Interior:
    """The interior as the proof server committed it: the window, its commitment, the link members' boundary ports
    (`links.Port`, by member key) and every member's header entries (`rows.write`'s ``registered``)."""

    window: CS.Binding
    committed: CS.Committed
    ports: dict[str, LK.Port]
    registered: dict[str, dict[str, dict]]


def interior(bc: CS.Committed, boundary_digest: str, sample: Sequence[int], law: Mapping | None = None, *,
             salt_key: bytes | None = None, replayed: np.ndarray | None = None) -> Interior:
    """The proof server's interior of the replay sample ``sample``: each span replayed from its ingress, the boundary record's
    row (``bc``, the boundary's commitment), and committed, each link member's boundary port under the boundary's salts at the
    sample's leaves. ``replayed``: the ingress replayed instead, ``(Q, 2H/16)`` u16: a prover's whose link rows don't open the
    boundary record (tests and refused runs)."""
    at = [int(s) for s in sample]
    win = interior_window(boundary_digest, at, law)
    names = boundary_ports(bc.window)
    ports = {win.members[t].key: LK.Port.of(bc, names[b]) for t, (_p, b) in LINKS.items()}
    r = replay(ports[win.members[0].key].rows_at(at) if replayed is None else replayed)
    data = [{"rows": {"row": r["ingress"]}, "outs": {"o0": r["ingress"]}},
            {"rows": {"x": CS.WIRED}, "outs": {"o0": r["fold"]}},
            {"rows": {"x": CS.WIRED}, "outs": {"o0": r["flip"]}},
            {"rows": {"x": CS.WIRED}, "outs": {"o0": np.repeat(r["flip_b"], N_T[3], axis=0)}},
            {"rows": {"row": CS.WIRED}, "outs": {"o0": r["egress"]}}]
    for d, m in zip(data, win.members):
        d["refs"] = np.array([win.held_leaves(m, p) for p, _ in m.rows], dtype=np.int64).T
    salts_of = {win.members[t].port(p): ports[win.members[t].key].salts_at(at) for t, (p, _b) in LINKS.items()}
    c = CS.commit(win, data, salt_key=salt_key, salts_of=salts_of)

    def whole(value: str) -> dict:
        port = LK.Port.of(c, value)
        return port.header(range(port.leaves))

    registered = {m.key: {p: ports[m.key].header(at) if p == link(t) else whole(win.held(m, p))
                          for p, _ in m.all_rows} for t, m in enumerate(win.members)}
    return Interior(win, c, ports, registered)


def reads(boundary_record: Mapping, interior_record: Mapping, window: CS.Binding, sample: Sequence[int]) -> dict[str, dict]:
    """The verifier's own ``--registered`` file of each member (by member key), from the two registration records and the
    replay sample alone: a link port at the root, leaves and domain the boundary record names for its boundary port, unit
    off_t + i reading leaf S[i]; a wired port at its writer's port of the interior record, unit base + i reading its writer
    unit's leaf; every other port at its own, unit base + i reading its own leaf i (`rows.Binding.held`)."""
    names = boundary_ports(boundary_window(int(boundary_record["population"]), len(sample)))
    out = {}
    for t, m in enumerate(window.members):
        out[m.key] = LK.reads_file({
            p: LK.reads(boundary_record, names[LINKS[t][1]], words, m.schema(p), sample, m.base, registered=REG)
            if p == link(t) else
            LK.reads(interior_record, window.held(m, p), words, m.schema(p), window.held_leaves(m, p), m.base,
                     registered=REG)
            for p, words in m.all_rows}, registered=REG)
    return out


def references(q: int, out: Path) -> dict:
    """The auditor's params (`service.template_circuits`' ``proofs``), fixed before the run: by template, in the check's
    order, its circuit file's SHA-512 and its reference statement's header and registered reads. The reference is
    `member_stager`'s statement of the template's member in the interior window of ``q`` replay blocks
    (`interior_window`, under a run of its own, whose fields round 2 places or blanks), bound to that window's partition
    and program as the sessions' circuits are, on zero rows at the member's slots, each port's table and refs as
    `interior` lays them out, salted under `REFERENCE_KEY`: its public file is ``out``/m<k>/pub.bin, its ``circuit.txt``
    beside it, and its prover file is deleted. Both files are a function of ``q`` and the design alone, byte for byte.
    Its registered reads are `reads`' file, which names none of the run's fields: each port the value it reads (a link
    port the boundary port it opens, a wired port its writer's output port, any other the member's own), its words and
    schema."""
    from verity_flock.class_statement import member_stager

    window, names = interior_window("00" * 64, range(q)), boundary_ports(boundary_window(q, q))
    bind = {"partition": window.partition_digest, "program": window.partition["program"]}
    named = []
    for k, (m, defn) in enumerate(zip(window.members, TEMPLATES)):
        d, rows = Path(out) / f"m{k}", {r: np.zeros((m.n, w), dtype=np.uint16) for r, w in m.rows}
        member_stager(defn, m, bind).stage(
            rows, d, REFERENCE_KEY, indices=range(m.base, m.base + m.n),
            tables={r: np.zeros((dict(m.tables)[r], w), dtype=np.uint16) for r, w in m.rows},
            refs={r: np.asarray(window.held_leaves(m, r)) for r in rows})
        (d / "inst.bin").unlink()
        with open(d / "pub.bin", "rb") as f:
            header = json.loads(f.readline())
        reads = LK.reads_file({p: {"value": names[LINKS[k][1]] if p == link(k) else window.held(m, p), "words": int(words),
                                   "word_bits": REG.WORD_BITS, REG.SCHEMA_KEY: m.schema(p)}
                               for p, words in m.all_rows}, registered=REG)
        named.append({"template": m.template, "circuit": header["circuit_sha512"], "header": header,
                      "registered": reads})
    return {"templates": named}


def write(c: CS.Committed, out, provenance: Mapping, *, previous: str | None = None,
          registered: Mapping | None = None, index_salt: bytes | None = None) -> dict:
    """``c``'s window files in ``out``, its registration after ``previous``, what the committer hands over: -> the
    registration record."""
    CS.write(c, out, dict(provenance), {"toy": "two-stage", "members": [m.key for m in c.window.members]},
             kind="stand-in", registered=registered, registrar=R, previous=previous, index_salt=index_salt)
    return json.loads((Path(out) / "registration.json").read_text())
