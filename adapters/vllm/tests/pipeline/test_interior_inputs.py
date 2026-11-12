"""A replay block's parameter round 1's record commits as a port (`interior_links.Input`), on a token select's
``logits``: `links` leaves it out, so the interior registers no value under its name, and `interior` commits the port
that reads it under the serve's salts at the rows its units read, so each row is round 1's leaf and the member's header
opens it at round 1's root. Where the replay block is the check's instances the port holds one row a token; where it is
the greedy select's units, one row a unit of a token, the lanes that unit reads.

The serve's side of the port (the class's layout, the engine's commit) is not this module's: round 1's port is made
here from the rows a serve would commit, as `boundary_rows.Committed.port` opens the block's return."""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest

from verity.protocols.spot.verifier import second_stage as SS, service as S
from verity.protocols.spot.verifier.one_stage import partition as P
from verity.circuits.codec import descriptor_id
from verity.circuits.evaluation.reference import evaluate
from verity.circuits.program import Program
from verity.commitments.indexed import RangeIndexedDomain
from verity.commitments.merkle import CommitmentDomain
from verity.commitments.windows import links as LK, rows as CS
from verity_vllm.engine import sampler_capture as SC
from verity_vllm.pipeline import boundary_rows as BR, interior_rows as IR

V, O, N = 64, 4, 3                # the vocabulary, a class's outputs and its spans
K = 4                             # the greedy select's units a token
SERVED = {"r0": 4, "r1": 4, "r2": 3}
SAMPLE = [2, 0]                   # the replay sample: r2, served one token short of its class, then r0
BOUNDARY = "aa" * 64
ROW = "interior-inputs-test"
NAME = "logits"


def _layout(units: int = 0) -> BR.Layout:
    t = f"GreedySelect_v1{{V={V},N={V // units}}}" if units else f"TokenSelect_v4{{V={V}}}"
    return BR.Layout(BR.check_template(t), 0, 1, BR.cap(3), O, N, BR.run_root("interior-inputs", 0), bool(units), ids=True)


def _design(L: BR.Layout) -> tuple[str, tuple[CS.Member, ...], tuple[int, ...]]:
    """The class's key, members and cells, as the two-stage driver derives them from its layout."""
    members = tuple(IR.member(P.template_id(t)) for t in L.templates)
    return descriptor_id(L.replay), members, S.replay_cells(Program(L.replay), P.template_instances_query(L.templates))


def _record(L: BR.Layout, *more: str) -> dict:
    """Round 1's record's ports, by name: the layout's, then ``more``."""
    return {"roots": [{"port": p} for p in (*L.ports(), *more)]}


def _captured(tmp_path, L: BR.Layout) -> dict[str, IR.Span]:
    """Each request's captured span of the class's check, of its served tokens, each token its Definition's on its row."""
    t, fn = P.template_id(L.check), L.check
    rng = np.random.default_rng(20261112)
    rows = {r: rng.integers(0, 1 << 16, (n, V), dtype=np.uint16) for r, n in SERVED.items()}

    def token(row: np.ndarray) -> int:
        out = evaluate(fn, row.tolist())
        return int(out[0]) if len(out) == 1 else int(out[0]) | int(out[1]) << 16
    SC.write(tmp_path, t, {r: {"logits": a, "tokens": [token(x) for x in a], "steps": list(range(len(a))),
                               "program": "p" * 64} for r, a in rows.items()})
    return {r: IR.captured(tmp_path, r) for r in SERVED}


def _port(rows: np.ndarray, schema: str) -> LK.Port:
    """Round 1's port ``logits`` of ``rows``, committed as a serve commits a row value: each row under a salt of its own."""
    domain = CommitmentDomain(hashlib.sha512(b"interior-inputs-test").digest(), CS.OWNER, RangeIndexedDomain(0, len(rows)),
                              hash="sha512")
    salts = hashlib.shake_256(b"the serve's salts").digest(len(rows) * CS.SALT_BYTES)
    _roots, bc = CS.commit_ports({NAME: domain}, {NAME: rows}, {}, {NAME: salts}, schemas={NAME: schema})
    return LK.Port(NAME, schema, int(rows.shape[1]), domain, bytes(bc[NAME]), rows, salts)


def _served_rows(pieces: dict[str, tuple[IR.Span, ...]], ins: tuple[IR.Input, ...], members, cells) -> np.ndarray:
    """The rows a serve commits for the port: every span's, block by block, each unit's row at the place round 2 reads
    it (`Input.at`), a span served short of its class padded by its last token's rows as its replay block's instances
    past them are copies of its last."""
    per, words = ins[0].per, ins[0].words
    out = np.zeros((N * per, words), dtype=np.uint16)
    at = {m.template: k for k, m in enumerate(members)}
    for s, r in enumerate(SERVED):
        for x in ins:
            k = at[x.template]
            out[[s * per + a for a in x.at]] = pieces[r][k].padded(cells[k])["rows"][x.port]
    return out


def test_links_leave_out_a_parameter_round_1s_record_commits_and_inputs_give_it():
    """Where round 1's record has no port ``logits`` the block's parameter is an `Operand`, a value the interior
    registers; where it has one, `links` gives no link for it, so the stream takes no value a sampled span and the
    interior registers none, and `inputs` gives the port that reads it: its rows a block and the row each unit reads.
    The verifier supplies no value of a name round 1's record commits."""
    L = _layout()
    key, members, cells = _design(L)
    tid = members[0].template
    mine, served = _record(L), _record(L, NAME)
    link, = IR.links(L.program, key, mine, members, cells)
    assert (type(link), link.port, link.value, link.per, link.at, link.words) == (IR.Operand, NAME, NAME, O, (0, 1, 2, 3), V)
    assert IR.inputs(L.program, key, mine, members, cells) == ()
    assert IR.links(L.program, key, served, members, cells) == () and IR.span_values(()) == ()
    x, = IR.inputs(L.program, key, served, members, cells)
    assert x == IR.Input(tid, NAME, NAME, O, (0, 1, 2, 3), V)
    assert x.rows(SAMPLE) == [8, 9, 10, 11, 0, 1, 2, 3]
    assert [(r.port, r.value, r.held) for r in IR.returns(L.program, key, served, members, cells)] == [("o0", "o0", True)]
    with pytest.raises(ValueError, match="round 1's record commits it as a port, so the verifier supplies no value"):
        IR.links(L.program, key, served, members, cells, supplied={NAME: None})
    with pytest.raises(ValueError, match="units reading rows"):
        replace(x, at=(0, 1, 2, O))
    with pytest.raises(ValueError, match="rows of 0 words"):
        replace(x, words=0)


def test_the_interior_commits_an_input_under_the_serves_salts_and_its_header_opens_round_1s_rows(tmp_path):
    """The port that reads round 1's ``logits`` is committed under the serve's salts at the rows its units read: each of
    its Q·O strings is round 1's leaf's at ``sample[i]·O + j``, a span served short of its class at its last token's row
    again, and the member's header names round 1's port, those rows and each one's path in round 1's tree. The interior
    registers no value, its program takes none, and every other port keeps salts of its own."""
    L = _layout()
    key, members, cells = _design(L)
    x, = IR.inputs(L.program, key, _record(L, NAME), members, cells)
    spans = {r: (IR.whole(s),) for r, s in _captured(tmp_path / "capture", L).items()}
    port = _port(_served_rows(spans, (x,), members, cells), members[0].schema(NAME))
    per = [spans[r] for r in ("r2", "r0")]
    it = IR.interior(per, cells, BOUNDARY, SAMPLE, row=ROW, inputs=[(x, port)])

    at, w, n = x.rows(SAMPLE), it.committed.window, CS.BC_BYTES
    name = w.members[0].port(NAME)
    assert not it.record.get("registered") and it.inputs == ((x, port),)
    assert P.program_digest(w.program) == P.program_digest(IR.interior_window(members, cells, BOUNDARY, SAMPLE).program)
    assert it.committed.bc[name] == b"".join(port.bcs[a * n:(a + 1) * n] for a in at)
    assert it.committed.salts[name] == port.salts_at(at)
    assert it.committed.salts[w.members[0].port("o0")] != port.salts_at(at)
    assert np.array_equal(port.rows_at([2 * O + 3]), port.rows_at([2 * O + 2]))
    IR.write(it, tmp_path / "interior")
    entry = CS.registered_header(json.loads((tmp_path / "interior" / "window.json").read_text()), 0)[NAME]
    assert entry == port.header(at) and LK.Port.of(it.committed, name).root != port.root

    link, = IR.links(L.program, key, _record(L), members, cells)
    old = IR.interior(per, cells, BOUNDARY, SAMPLE, row=ROW, links=[link])
    assert [e["value"] for e in old.record["registered"]] == [NAME]
    assert P.program_digest(old.committed.window.program) != P.program_digest(w.program)


def test_an_input_that_is_not_round_1s_is_refused_by_name(tmp_path):
    """Before anything is committed: a replayed unit that takes another row than round 1's at its place (by slot and
    row), a port round 1 padded otherwise than by the span's last token, a port opened as another value, in another
    width, without the serve's rows, or in other than whole blocks, a replay sample past it, a port given twice or
    that also reads a value the interior registers, a template the window has none of, and a commitment on the rows'
    device."""
    L = _layout()
    key, members, cells = _design(L)
    x, = IR.inputs(L.program, key, _record(L, NAME), members, cells)
    spans = {r: (IR.whole(s),) for r, s in _captured(tmp_path / "capture", L).items()}
    rows = _served_rows(spans, (x,), members, cells)
    port, per = _port(rows, members[0].schema(NAME)), [spans[r] for r in ("r2", "r0")]

    def refused(match: str, inputs, sample=SAMPLE, **more) -> None:
        with pytest.raises(ValueError, match=match):
            IR.interior(per, cells, BOUNDARY, sample, row=ROW, inputs=inputs, **more)

    other = rows.copy()
    other[1, 5] ^= 1
    refused(r"unit 5 \(sampled span 1\) takes at its 'logits' a row that is not row 1 of logits, the port round 1 committed",
            [(x, _port(other, port.schema))])
    zero = rows.copy()
    zero[2 * O + 3] = 0
    refused(r"unit 3 \(sampled span 0\) takes at its 'logits' a row that is not row 11 of logits",
            [(x, _port(zero, port.schema))])
    refused("which is opened as 'logits'", [(replace(x, value="other"), port)])
    refused(f"in rows of {V - 1} words, which is none of its rows", [(replace(x, words=V - 1), port)])
    refused(f"in rows of {V} words, which is none of its rows", [(replace(x, port="o0"), port)])
    refused("the serve's rows and salts are not held", [(x, replace(port, rows=None))])
    refused("the serve's rows and salts are not held", [(x, replace(port, salts=None))])
    refused("not whole blocks of 5", [(replace(x, per=5), port)])
    refused(f"the replay sample reads past its {N * O} rows", [(x, port)], sample=[N, 0])
    refused("or is given twice", [(x, port), (x, port)])
    link, = IR.links(L.program, key, _record(L), members, cells)
    refused("and a value the interior registers", [(x, port)], links=[link])
    refused("a template the window has none of", [(replace(x, template="TokenSelect_v4{V=128}"), port)])
    whole = IR.interior(per, cells, BOUNDARY, SAMPLE, row=ROW, inputs=[(x, port)]).committed
    refused("a replay block whose return or parameter round 1 commits is committed on the host", [(x, port)],
            device=IR.DeviceCommit(whole.roots, whole.bc, whole.salts))


def test_a_greedy_selects_units_each_read_their_lanes_of_round_1s_logits(tmp_path):
    """The greedy select as its units, each a proof block: every unit's ``logits`` parameter is a link at round 1's
    port, one row a unit of a token (its lanes), the K rows of a token read once each and no row of a block left
    unread, so `links` gives nothing and the interior registers nothing. Each unit's port is committed under the serve's
    salts at its own rows, beside the wires the combine reads the chunks' outputs through, and each member's header
    opens round 1's rows."""
    L = _layout(K)
    key, members, cells = _design(L)
    wires, rws = SS.replay_wires(L.program, key), SS.replay_rows(L.program, key)
    ins = IR.inputs(L.program, key, _record(L, NAME), members, cells)
    assert IR.links(L.program, key, _record(L, NAME), members, cells) == ()
    assert [x.template for x in ins] == [m.template for m in members] and {x.value for x in ins} == {NAME}
    assert {(x.per, x.words, len(x.at)) for x in ins} == {(K * O, V // K, O)}
    assert sorted(a for x in ins for a in x.at) == list(range(K * O))
    assert all(sorted({a // K for a in x.at}) == list(range(O)) for x in ins)

    spans = {r: IR.pieces(s) for r, s in _captured(tmp_path / "capture", L).items()}
    assert [s.member.template for s in spans["r0"]] == [m.template for m in members]
    port = _port(_served_rows(spans, ins, members, cells), members[0].schema(ins[0].port))
    per = [spans[r] for r in ("r2", "r0")]
    it = IR.interior(per, cells, BOUNDARY, SAMPLE, row=ROW, wires=wires, rows=rws, inputs=[(x, port) for x in ins])
    assert not it.record.get("registered") and len(it.inputs) == K
    w, n = it.committed.window, CS.BC_BYTES
    IR.write(it, tmp_path / "interior")
    win = json.loads((tmp_path / "interior" / "window.json").read_text())
    for k, x in enumerate(ins):
        at, name = x.rows(SAMPLE), w.members[k].port(x.port)
        assert it.committed.bc[name] == b"".join(port.bcs[a * n:(a + 1) * n] for a in at)
        assert CS.registered_header(win, k)[x.port] == port.header(at)
    with pytest.raises(ValueError, match="is none of its rows .* or is wired"):
        IR.interior(per, cells, BOUNDARY, SAMPLE, row=ROW, wires=wires, rows=rws,
                    inputs=[(replace(ins[-1], port=wires[0].port, words=dict(members[-1].rows)[wires[0].port]), port)])

