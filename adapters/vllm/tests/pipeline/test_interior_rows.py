"""A two-stage sampled audit's interior registration from replay (`pipeline.interior_rows`), on the sampled replay's
CPU-served world (`tests.check.test_sampled_replay`'s RoPE Program at D = 64, three requests): each request's replay
reproduces the RoPE heads serving committed, bit for bit, evaluating only its readers' cone; and on a two-stage
`Stream` over RoPE heads, after a stand-in boundary registration over the three request spans and its first stage, the
replay sample's interior registration, one shared-row member whose roots are its word rows', is accepted and the second
stage samples k in each sampled span, while a forged
leaf, a wrong `previous`, a wrong population and the spans in another order than the replay sample's are refused by
name, and spans replayed from two Programs are not committed. A sampler capture written from the replays (the GPU
replay's dry run) reads back as each request's token-select span and registers as a two-stage interior over token
selects."""

from __future__ import annotations

import copy
import json
from dataclasses import replace

import numpy as np
import pytest

from verity.protocols.spot.verifier import second_stage as SS, service as S
from verity.protocols.spot.verifier.one_stage import partition as P, registration as R
from verity.circuits.codec import descriptor_id
from verity.circuits.defs import CompositeDefinition, bind
from verity.circuits.evaluation.reference import evaluate
from verity.circuits.program import Program
from verity.circuits.types import Array
from verity.definitions.numerics.rows import SAMPLED_MEMBER
from verity.definitions.registry import dead_free as DF
from verity.commitments.windows import rows as CS
from verity_vllm.check.replay.index import SAMPLER_OP, index_of
from verity_vllm.check.replay.opening import CommittedStore
from verity_vllm.check.replay.request_replay import Replay
from verity_vllm.engine import sampler_capture as SC
from verity_vllm.pipeline import boundary_rows as BR, chunk_rows as CR, interior_rows as IR, serving_rows as SRW

ROPE = "RoPEHead_v1{D=64}"
PROMPTS = {"r0": [1, 2], "r1": [5, 3], "r2": [4, 6]}
UNITS = 16                        # a request's RoPE heads: 4 tokens (2 prefill, 2 decode) x (3 query + 1 key heads)
N_RHO, Q, K = 16, 2, 1
SALT = bytes(range(32))
ROW = "interior-rows-test"
DELTA = 2.0 ** -40


@pytest.fixture(scope="module")
def world():
    """The CPU-served world: the honest execution of the RoPE Program for the three requests, every member it committed."""
    import tests.check.test_sampled_replay as T

    dh, v, T.DH, T.V = T.DH, T.V, 64, 64       # a token select's logits row then one whole SHA-512 block, as a row needs
    try:
        rng = np.random.default_rng(23)
        inst, woff = T._program(rope=True)
        w = T._bf16(rng, max(hi for _lo, hi in woff.values()), 0.5)
        for f in ("model.model.layers.0.input_layernorm.weight", "model.model.norm.weight"):
            w[woff[f][0]:woff[f][1]] = 0x3F80
        lo, hi = woff["model.model.layers.0.self_attn.rotary_emb.cos_sin_cache"]
        w[lo:hi] = (np.asarray(rng.uniform(-1, 1, hi - lo), dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)
        return T.Honest(inst, woff, w, PROMPTS)
    finally:
        T.DH, T.V = dh, v


def _weights(world):
    def read(field, lo, hi):
        flo, _fhi = world.woff[field]
        return world.weights[flo + lo:flo + hi]
    return read


def _replay(world, rid: str) -> Replay:
    return Replay(index_of(world.inst), rid, PROMPTS[rid], _weights(world))


def _served(world, rid: str):
    """Request ``rid``'s RoPE heads read from what serving committed, as the serving rows read them."""
    bmap, committed = world.bmap_and_reader(order=tuple(PROMPTS))
    window, rows = SRW.population({rid: world.inst}, {"templates": [ROPE], "law": "subset:1"}, "00" * 64)
    (data, entries), = SRW.read_all(window.members, rows, CommittedStore(bmap, committed), _weights(world), {rid: 0})
    return data, entries


def test_a_replayed_span_is_the_span_serving_committed_and_only_its_cone(world):
    """Each request's replay reads its RoPE heads bit for bit as the committed store holds them, index entries
    included, and evaluates only the rows its heads read: never its last step's MLP, logits or sampled token."""
    for rid in PROMPTS:
        replay = _replay(world, rid)
        got = IR.span(world.inst, replay, ROPE)
        data, entries = _served(world, rid)
        assert got.n == UNITS and got.entries == entries
        for kind in ("rows", "outs"):
            assert sorted(got.data[kind]) == sorted(data[kind])
            for p, a in data[kind].items():
                assert np.array_equal(got.data[kind][p], a), (rid, kind, p)
        last = max(s for (s, _rid, op, m) in world.blocks if op == SAMPLER_OP and m == SAMPLED_MEMBER)
        sel = replay.at[(last, SAMPLER_OP, SAMPLED_MEMBER)][2][0]
        assert 0 < replay.rows < len(world.inst["rows"]) and sel not in replay.out
        assert set(replay.seconds) >= {"Embedding_v1", "Gemm_v1", "RoPE_v1", "TokenSelect_v1"}


def test_a_replay_names_what_it_cannot_give(world):
    replay = _replay(world, "r0")
    assert replay.words("r1", 0, replay.embed, "in", 0, 1) == (None, "the replay holds request r0, not r1")
    w, why = replay.words("r0", 0, "model.nowhere", "out", 0, 1)
    assert w is None and "nothing gives it" in why
    w, why = replay.words("r0", 0, replay.embed, "in", 0, 3)
    assert w is None and "prompt ids [0, 3)" in why
    cached = copy.copy(index_of(world.inst))
    cached.kv_prefix = {"C": 2}
    with pytest.raises(ValueError, match=r"another request computed: .*recompute_hits=True"):
        Replay(cached, "r0", PROMPTS["r0"], _weights(world))


def test_a_replay_s_arrival_is_its_program_s_first_step_so_steps_ahead_of_the_request_s_shift_it(world):
    """A class-G Program that recomputes a hit runs its chunks in steps ahead of the request's first served step: replayed
    with ``arrive`` that many steps earlier, the request's served step reads the Program's step after them."""
    op = next(op for (s, _rid, op, m) in world.blocks if s == 1 and m == "out" and op != SAMPLER_OP)
    ahead = Replay(index_of(world.inst), "r0", PROMPTS["r0"], _weights(world), arrive=-1)
    got, _ = ahead.words("r0", 0, op, "out", 0, 4)
    want, _ = _replay(world, "r0").words("r0", 1, op, "out", 0, 4)
    assert want is not None and np.array_equal(got, want)


def test_a_port_map_says_which_committed_word_each_units_word_is(world):
    """`port_map` reads off `read_all`, before any value, which committed element or registered weight each RoPE head's
    port word is; gathered from what serving committed, it is each request's span word for word, from its committed
    members and the rotary cache. A template whose read computes its words (a GEMM's shared rows) is refused by name."""
    bmap, committed = world.bmap_and_reader(order=tuple(PROMPTS))
    store = CommittedStore(bmap, committed)
    for rid in PROMPTS:
        pm = IR.port_map(world.inst, ROPE, rid)
        data, _entries = _served(world, rid)
        got, = IR.gather(pm, store, _weights(world))
        assert {p: a.tolist() for p, a in got.items()} == {p: a.tolist() for k in ("rows", "outs") for p, a in data[k].items()}
        assert {s[0] for s in pm["sources"]} == {"store", "weights"} and all(s[1] == rid for s in pm["sources"] if s[0] == "store")
    with pytest.raises(ValueError, match="computed by its read, not moved"):
        IR.port_map(world.inst, "GemmCoordinate_v1{K=64}", "r0")


def test_a_span_pads_to_its_size_class_with_its_last_unit(world):
    s = IR.span(world.inst, _replay(world, "r1"), ROPE)
    got = s.padded(UNITS + 3)
    for kind in ("rows", "outs"):
        for p, a in s.data[kind].items():
            assert np.array_equal(got[kind][p][:UNITS], a) and all(np.array_equal(r, a[-1]) for r in got[kind][p][UNITS:])
    with pytest.raises(ValueError, match="not 1 to its size class's 15"):
        s.padded(UNITS - 1)


# --- the two-stage stream, after a stand-in boundary registration --------------------------------------------------------

UNIT = SRW.template(ROPE).fn


def _block(n: int, unit=UNIT, name: str = "InteriorRowsTestSpan"):
    """The stand-in replay block: ``n`` units of ``unit`` (RoPE heads), a request span at its size class."""
    params, ret = unit.signature()
    return bind(CompositeDefinition(name, 0, (), lambda S_: (tuple((p, Array(n, t)) for p, t in params), Array(n, ret)),
                                    lambda B, S_, *xs: B.batch(unit, *xs, axes=(0,) * len(xs)), register=False))


BLOCK = _block(N_RHO)


def _boundary_window(n: int, block=BLOCK) -> CS.Binding:
    """The stand-in boundary: ``n`` request spans, each its certified prompt ids in and its sampled tokens out."""
    return CS.Binding(P.population_program(block, n), P.template_instance_query(block), {"law": "subset", "k": Q}, "b0" * 64,
                      (CS.Member(P.template_id(block), (("ingress", 64),), (("egress", 64),), n),))


def _ids(ids) -> np.ndarray:
    a = np.zeros(64, dtype=np.uint16)
    w = np.asarray(ids, dtype="<u4").view("<u2")
    a[:w.size] = w
    return a


def _tokens(world) -> dict[str, list[int]]:
    return {rid: [int(world.blocks[k][0]) for k in sorted(world.blocks) if k[1] == rid and k[2] == SAMPLER_OP] for rid in PROMPTS}


def _boundary(world, block=BLOCK) -> tuple[dict, dict, str]:
    window, tokens = _boundary_window(len(PROMPTS), block), _tokens(world)
    c = CS.commit(window, [{"rows": {"ingress": np.stack([_ids(p) for p in PROMPTS.values()])},
                            "outs": {"egress": np.stack([_ids(tokens[r]) for r in PROMPTS])}}], salt_key=SALT)
    rec = CS.registration(c, {"stand_in": "boundary"}, None, registrar=R)
    return rec, CS.expected_ports(window), CS.leaf_layer({m.key: c.public_body(k) for k, m in enumerate(window.members)})


def _spec(member: CS.Member, unit=UNIT, block=BLOCK, n_rho: int = N_RHO) -> S.Spec:
    names = {}
    for phase, w in (("serve", _boundary_window(len(PROMPTS), block)), ("interior", IR.interior_window(member, n_rho, "00", [0]))):
        names |= {m.port(p): (m.schema(p), phase) for m in w.members for p in CS.port_leaves(m)}
    public = (S.Public("population", "structure", "N, the sample's range"),
              S.Public("window", "structure", "provenance: served, on test prompts"),
              S.Public("leaf_layer", "structure", "the members' public bodies' digest"),
              *(S.Public(f"{k}:{p}", "structure", f"{p}'s layout") for p in sorted(names) for k in ("leaves", "domain")))
    return S.Spec(tuple(S.Value(p, s, phase) for p, (s, phase) in sorted(names.items())),
                  S.TwoStage(P.template_instance_query(block), f"rate:{Q}/{len(PROMPTS)}", K),
                  S.Check(P.program_digest(Program(unit)), P.template_instance_query(unit), proof_deadline=60.0),
                  public, b"interior-rows-test")


class _Clock:
    def __call__(self) -> float:
        return 0.0


def _refused(f) -> tuple[str, ...]:
    with pytest.raises(S.Refused) as e:
        f()
    return e.value.verdict.codes


@pytest.fixture(scope="module")
def spans(world):
    return {rid: IR.span(world.inst, _replay(world, rid), ROPE) for rid in PROMPTS}


def _first(world, spans, unit=UNIT, block=BLOCK, n_rho: int = N_RHO):
    """A two-stage stream over ``unit`` (RoPE heads), the stand-in boundary registered and its first stage drawn (spans 0
    and 2)."""
    draws = iter([[0, 2]])
    stream = S.Stream(_spec(spans["r0"].member, unit, block, n_rho), unit, replay=block, budget=1, delta=DELTA,
                      challenger="challenger-test", clock=_Clock(), sampler=lambda law, n: next(draws) if law["law"] == "subset"
                      else [i * n_rho for i in range(len(law["strata"]))])
    rec, ports, layer = _boundary(world, block)
    reg = stream.register(rec, ports, leaf_layer=layer)
    return stream, reg, stream.select(reg)


def test_the_replay_samples_interior_registers_after_its_boundary_and_the_second_stage_samples_it(world, spans):
    """The interior of spans 0 and 2, committed from their replays, registers after the boundary on the challenger's
    own ports (from the size class, the boundary's digest and its replay sample) and the leaf layer of the bodies it
    received; the second stage then samples k heads in each sampled span. Every leaf's salt is fresh from the OS, so two
    commitments of one replay share their ports and no root. By both verifiers, so it builds the Lean."""
    stream, reg, first = _first(world, spans)
    assert first.units == [0, 2]
    it = IR.interior([spans["r0"], spans["r2"]], N_RHO, reg.digest, first.units, row=ROW, replay="replay-test")
    again = IR.interior([spans["r0"], spans["r2"]], N_RHO, reg.digest, first.units, row=ROW)
    assert it.ports == again.ports == IR.interior_ports(spans["r0"].member, N_RHO, reg.digest, first.units)
    assert not set(it.committed.roots.values()) & set(again.committed.roots.values())
    ireg = IR.register(stream, first, it)
    assert ireg.record["population"] == Q * N_RHO and ireg.record["previous"] == reg.digest and stream.log[-1] == ireg.record
    assert ireg.record["window"] == {"kind": "served", "run": IR.interior_run(reg.digest, [0, 2]), "row": ROW,
                                     "interior": True, "replay_sample": [0, 2], "replay": "replay-test"}
    assert IR.interior_run(reg.digest, first.units) == SS.interior_root(reg.digest, first.units)
    second = stream.select(ireg)
    assert second.replay_sample == first and [u // N_RHO for u in second.units] == [0, 1]


def test_the_interior_is_one_shared_row_member_of_its_word_rows_roots(spans):
    """The interior's member is a shared-row member, each input port a table of its own Q·n_rho rows and unit i reading row
    i, so every row a second-stage session proves is a registered read of the interior record. Under the same salts its
    roots and every row's b || c are the word-row layout's: only the public body, and so the leaf layer, differs."""
    it = IR.interior([spans["r0"], spans["r2"]], N_RHO, "aa" * 64, [0, 2], row=ROW, salt_key=SALT)
    w = it.committed.window
    m, = w.members
    assert dict(m.tables) == {p: Q * N_RHO for p, _ in m.rows} and m.n == Q * N_RHO
    assert np.array_equal(it.committed.refs[0], np.tile(np.arange(Q * N_RHO)[:, None], (1, len(m.all_rows))))
    padded = [spans[r].padded(N_RHO) for r in ("r0", "r2")]
    data = {k: {p: np.concatenate([d[k][p] for d in padded]) for p in padded[0][k]} for k in ("rows", "outs")}
    plain = CS.commit(replace(w, members=(replace(m, tables=()),)), [data], salt_key=SALT)
    assert plain.roots == it.committed.roots and plain.bc == it.committed.bc
    assert plain.public_body(0) != it.committed.public_body(0)


def test_a_commitment_made_on_the_rows_device_registers_as_the_hosts_would(world, spans):
    """A GPU commit hands `interior` each port's root, every row's b || c and the salts it drew: from spans that hold no
    rows, it registers the record, ports and leaf layer the host's commitment of the rows makes under the same salts, and
    the stream accepts it. A commitment missing a port, or with a short root, b || c or salts, is refused by name."""
    stream, reg, first = _first(world, spans)
    host = IR.interior([spans["r0"], spans["r2"]], N_RHO, reg.digest, first.units, row=ROW, salt_key=SALT)
    c = host.committed
    device = IR.DeviceCommit(c.roots, c.bc, c.salts)
    bare = [replace(spans[r], data={}) for r in ("r0", "r2")]
    gpu = IR.interior(bare, N_RHO, reg.digest, first.units, row=ROW, device=device)
    assert (gpu.record, gpu.ports, gpu.leaf_layer) == (host.record, host.ports, host.leaf_layer)
    assert not gpu.committed.rows and IR.register(stream, first, gpu).record == host.record
    q = next(iter(c.roots))
    for bad, why in (({k: v for k, v in c.roots.items() if k != q}, "not the window's"),
                     ({**c.roots, q: c.roots[q][:32]}, "a 32-byte root")):
        with pytest.raises(ValueError, match=why):
            IR.interior(bare, N_RHO, reg.digest, first.units, row=ROW, device=replace(device, roots=bad))
    with pytest.raises(ValueError, match="bytes of b \\|\\| c, for its"):
        IR.committed(c.window, replace(device, bc={**c.bc, q: c.bc[q][:-1]}))
    with pytest.raises(ValueError, match="bytes of salts, for its"):
        IR.committed(c.window, replace(device, salts={**c.salts, q: c.salts[q] + b"\0"}))


def test_an_interior_registration_is_refused_by_name(world, spans):
    """Against the challenger's copies: a leaf forged after the replay (`R4-leaf-layer`, the record's leaf layer not
    the one of the bodies it released), a record not chained after the boundary (`R6-previous`), spans committed at
    another size class (`interior-population`), and the replay sample's spans in another order, whose record names
    another run (`R3-domain`). Which span's values sit at which n_rho units is the second stage's proof's to hold: the
    registration hides them. Spans replayed from two Programs are refused before any commitment."""
    stream, reg, first = _first(world, spans)
    with pytest.raises(ValueError, match="not one: a size class's spans share its replay Definition"):
        IR.interior([spans["r0"], replace(spans["r2"], program="p" * 64)], N_RHO, reg.digest, first.units, row=ROW)
    honest = IR.interior([spans["r0"], spans["r2"]], N_RHO, reg.digest, first.units, row=ROW, salt_key=SALT)
    bad = spans["r2"].data["outs"]["out"].copy()
    bad[3, 5] ^= 1
    forged = replace(spans["r2"], data={**spans["r2"].data, "outs": {"out": bad}})
    lie = IR.interior([spans["r0"], forged], N_RHO, reg.digest, first.units, row=ROW, salt_key=SALT)
    assert lie.ports == honest.ports and lie.leaf_layer != honest.leaf_layer
    assert _refused(lambda: IR.register(stream, first, lie, leaf_layer=honest.leaf_layer)) == ("R4-leaf-layer",)
    late = IR.interior([spans["r0"], spans["r2"]], N_RHO, reg.digest, first.units, row=ROW, previous="00" * 64,
                       salt_key=SALT)
    assert _refused(lambda: IR.register(stream, first, late)) == ("R6-previous",)
    wide = IR.interior([spans["r0"], spans["r2"]], N_RHO + 1, reg.digest, first.units, row=ROW, salt_key=SALT)
    assert _refused(lambda: IR.register(stream, first, wide)) == ("interior-population",)
    swapped = IR.interior([spans["r2"], spans["r0"]], N_RHO, reg.digest, [2, 0], row=ROW, salt_key=SALT)
    assert _refused(lambda: IR.register(stream, first, swapped, ports=honest.ports)) == ("R3-domain",)
    assert IR.register(stream, first, honest).record["population"] == Q * N_RHO


# --- a token select's span from a sampler capture ------------------------------------------------------------------------

TSEL = "TokenSelect_v1{V=64}"
TSEL_RHO = 4                      # a request's sampled tokens: 3 (its prefill's and two decodes')


def test_a_sampler_capture_is_the_token_select_span_and_registers_as_its_interior(world, tmp_path):
    """The GPU replay's dry run: each request's capture, written from its CPU replay's token-select inputs (logits rows,
    ids, steps), reads back as the span `span` reads from its Program, member and words, and the replay sample's
    captured spans register as the interior of a two-stage stream over token selects. A capture whose ids are not the
    ones served, or whose rows changed after it was written, is refused before any commitment, in words that say how
    many ids differ and carry none of them."""
    replayed = {rid: IR.span(world.inst, _replay(world, rid), TSEL) for rid in PROMPTS}
    served = _tokens(world)
    SC.write(tmp_path, TSEL, {rid: {"logits": s.data["rows"]["in0"], "tokens": served[rid], "steps": [e[3] for e in s.entries],
                                            "program": s.program} for rid, s in replayed.items()})
    got = {rid: IR.captured(tmp_path, rid, served[rid]) for rid in PROMPTS}
    for rid, s in replayed.items():
        assert got[rid].member == s.member and got[rid].program == s.program
        assert all(np.array_equal(got[rid].data[k][p], a) for k in ("rows", "outs") for p, a in s.data[k].items())
    n = len(served["r1"])
    with pytest.raises(ValueError, match=f"request r1: the replay selected {n} ids and serving emitted {n}, 1 of them "
                                         "different: it didn't reproduce the span"):
        IR.captured(tmp_path, "r1", served["r1"][:-1] + [served["r1"][-1] + 1])
    with pytest.raises(ValueError, match=f"selected {n} ids and serving emitted {n - 1}, 2 of them different") as e:
        IR.captured(tmp_path, "r1", [served["r1"][0] + 100000, *served["r1"][1:-1]])
    assert str(served["r1"][0] + 100000) not in str(e.value)
    unit = SRW.template(TSEL).fn
    block = _block(TSEL_RHO, unit, "InteriorRowsTestSelects")
    stream, reg, first = _first(world, got, unit, block, TSEL_RHO)
    it = IR.interior([got["r0"], got["r2"]], TSEL_RHO, reg.digest, first.units, row=ROW)
    assert IR.register(stream, first, it).record["population"] == Q * TSEL_RHO
    raw = bytearray((tmp_path / "1.u16").read_bytes())
    raw[0] ^= 1
    (tmp_path / "1.u16").write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="is not the capture's"):
        IR.captured(tmp_path, "r1")
    doc = json.loads((tmp_path / SC.CAPTURE).read_text())
    doc["spans"]["r1"]["steps"] = doc["spans"]["r1"]["steps"][:-1]
    (tmp_path / SC.CAPTURE).write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="for 3 tokens at 2 steps"):
        IR.captured(tmp_path, "r1")


# --- a token select as the units Flock proves it by ----------------------------------------------------------------------

NAN, NEG_NAN, NEG_ZERO, NEG_INF, INF, ONE, NEG_ONE = 0x7FC0, 0xFFC0, 0x8000, 0xFF80, 0x7F80, 0x3F80, 0xBF80


def _select_rows(V: int) -> np.ndarray:
    """Logits rows of random words, then of edge cases: none above -inf, a NaN logits[0], a tie across chunks, -0 before
    +0, +inf last, and NaNs but one number in the last chunk."""
    edge = np.full((6, V), NEG_INF, dtype=np.uint16)
    edge[1, :], edge[1, 0] = ONE, NAN
    edge[2, [5, V // 2 + 5]] = ONE
    edge[3, 3], edge[3, 4] = NEG_ZERO, 0
    edge[4, -1] = INF
    edge[5, :], edge[5, -2] = NEG_NAN, NEG_ONE
    return np.concatenate([np.random.default_rng(20261108).integers(0, 1 << 16, (4, V), dtype=np.uint16), edge])


def _selects(tmp_path, V: int, t: str | None = None) -> dict[str, IR.Span]:
    """Two requests' captured spans of five tokens each of the token select ``t`` (`TokenSelect_v4{V}` by default), each
    token its Definition's on its row."""
    t, rows = t or f"TokenSelect_v4{{V={V}}}", _select_rows(V)
    fn = SRW.template(t).fn
    SC.write(tmp_path, t, {rid: {"logits": part, "tokens": [int(evaluate(fn, r.tolist())[0]) for r in part],
                                 "steps": list(range(len(part))), "program": "p" * 64}
                           for rid, part in (("a", rows[:5]), ("b", rows[5:]))})
    return {rid: IR.captured(tmp_path, rid) for rid in ("a", "b")}


@pytest.mark.parametrize("V,c", [(128, None), (64, 16)])
def test_a_token_selects_pieces_are_its_chunks_and_combine_by_their_definitions(tmp_path, V, c):
    """`pieces` lays a captured token select out as `TokenSelect_v4` calls it: each token's V / C chunk units, each its
    Definition's (best, idx) of its logits, then its combine unit over logits[0] and those partials, which its Definition
    takes to the span's token, on random words and on NaN, ±0, ±inf and ties across chunks; each piece's member is its
    template's bit rows as Flock stages a unit, each row of exactly its words (Flock refuses the whole select at a served
    vocabulary). Pieces that don't select the span's token, and chunks that don't tile the row, are refused."""
    C, out = c or min(DF.CHUNK, V), f"out.{SAMPLED_MEMBER}"
    k = V // C
    for s in _selects(tmp_path, V).values():
        chunks, comb = IR.pieces(s, c)
        assert (chunks.member, comb.member) == (IR.named(CR.unit_member(SRW.template(f"TokenSelectChunk_v2{{C={C}}}"), s.n * k)),
                                                IR.named(CR.unit_member(SRW.template(f"TokenSelectCombine_v2{{V={V},C={C}}}"), s.n)))
        assert [p for p, _w in chunks.member.rows + comb.member.rows] == ["x", "x0", "best", "idx"]
        assert [w for _p, w in comb.member.rows] == [1, k, 2 * k]
        best, idx = chunks.data["outs"]["o0"][:, 0].tolist(), chunks.data["outs"]["o1"].view("<u4")[:, 0].tolist()
        for j, row in enumerate(s.data["rows"]["in0"].tolist()):
            parts = [evaluate(bind(DF.TokenSelectChunkV2, C=C), row[a:a + C]) for a in range(0, V, C)]
            assert list(zip(best[j * k:(j + 1) * k], idx[j * k:(j + 1) * k])) == [tuple(p) for p in parts]
            token = evaluate(bind(DF.TokenSelectCombineV2, V=V, C=C), row[0], [p[0] for p in parts], [p[1] for p in parts])
            assert token == (int(s.data["outs"][out].view("<u4")[j, 0]),)
        assert comb.data["rows"]["best"].reshape(-1).tolist() == best and comb.data["rows"]["x0"][:, 0].tolist() == [
            r[0] for r in s.data["rows"]["in0"].tolist()]
    wrong = s.data["outs"][out].copy()
    wrong[0, 0] ^= 1
    with pytest.raises(ValueError, match=f"its chunks and combine select other ids than its span's at 1 of its {s.n} tokens"):
        IR.pieces(replace(s, data={**s.data, "outs": {out: wrong}}), c)
    with pytest.raises(ValueError, match="are not whole chunks"):
        IR.pieces(s, 3)


@pytest.mark.parametrize("V,c", [(128, None), (64, 16)])
def test_a_token_selects_pieces_on_the_logits_device_are_its_pieces_on_the_host(tmp_path, V, c):
    """`sampler_capture.piece_rows`, on the device the sampler's logits are on (CPU torch here), lays every piece's rows out
    word for word as `pieces` does from the captured span, NaN, ±0, ±inf and ties across chunks included, and padded to a
    size class as `Span.padded` pads each piece by whole tokens; ids the combine doesn't select are refused."""
    import torch
    for s in _selects(tmp_path, V).values():
        ids = s.data["outs"][f"out.{SAMPLED_MEMBER}"].view("<u4")[:, 0].astype(np.int64)
        logits = torch.from_numpy(s.data["rows"]["in0"].view(np.int16).copy()).view(torch.bfloat16)
        got, padded = SC.piece_rows(logits, torch.from_numpy(ids), c), SC.piece_rows(logits, ids, c, pad=s.n + 3)
        for piece in IR.pieces(s, c):
            want = {p: a.tolist() for kind in ("rows", "outs") for p, a in piece.data[kind].items()}
            assert {p: a.numpy().view(np.uint16).tolist() for p, a in got[piece.member.template].items()} == want
            whole = piece.padded(piece.n // s.n * (s.n + 3))
            want = {p: a.tolist() for kind in ("rows", "outs") for p, a in whole[kind].items()}
            assert {p: a.numpy().view(np.uint16).tolist() for p, a in padded[piece.member.template].items()} == want
        with pytest.raises(ValueError, match="the chunks and combine select"):
            SC.piece_rows(logits, torch.from_numpy(ids ^ 1), c)
        with pytest.raises(ValueError, match="don't pad to a size class of 4"):
            SC.piece_rows(logits, ids, c, pad=4)


@pytest.mark.parametrize("V,K", [(256, 2), (192, 3), (128, 1)])
def test_a_token_select_by_units_pieces_are_its_units_and_combine_by_their_definitions(tmp_path, V, K):
    """`pieces` lays a captured `TokenSelectUnits_v1{V, K}` out as it calls its templates: each token's K
    `TokenSelectGroups_v2` units, each its Definition's (bests, indices, first logit) of its logits, then its
    `TokenSelectCombineUnits_v1` unit, whose ``x0`` row is its first unit's third output row and whose ``best<j>`` and
    ``idx<j>`` rows are its unit j's first two, whole, and which its Definition takes to the span's token, on random
    words and on NaN, ±0, ±inf and ties across chunks; a span pads by whole tokens. Pieces that don't select the span's
    token, and another chunk than the Definition's, are refused."""
    U, out = V // K, f"out.{SAMPLED_MEMBER}"
    g = U // DF.CHUNK
    unit, combine = bind(DF.TokenSelectGroupsV2, C=U, G=DF.CHUNK), bind(DF.TokenSelectCombineUnitsV1, V=V, C=DF.CHUNK, K=K)
    for s in _selects(tmp_path, V, f"TokenSelectUnits_v1{{V={V},K={K}}}").values():
        units, comb = IR.pieces(s)
        assert (units.member, comb.member) == (IR.member(unit.id, s.n * K), IR.member(combine.id, s.n))
        assert (units.group, comb.group) == (K, 1)
        assert [p for p, _w in units.member.rows + comb.member.rows] == [
            "x", "x0", *(f"best{j}" for j in range(K)), *(f"idx{j}" for j in range(K))]
        assert [w for _p, w in comb.member.rows] == [1, *[g] * K, *[2 * g] * K]
        o0, o1, o2 = (units.data["outs"][o] for o in ("o0", "o1", "o2"))
        for j, row in enumerate(s.data["rows"]["in0"].tolist()):
            parts = [evaluate(unit, row[a:a + U]) for a in range(0, V, U)]
            got = [(*o0[j * K + u].tolist(), *o1[j * K + u].view("<u4").tolist(), int(o2[j * K + u, 0])) for u in range(K)]
            assert got == parts
            token = evaluate(combine, row[0], *(p[:g] for p in parts), *(p[g:2 * g] for p in parts))
            assert token == (int(s.data["outs"][out].view("<u4")[j, 0]),)
        rows = comb.data["rows"]
        assert np.array_equal(rows["x0"], o2[::K])
        assert all(np.array_equal(rows[f"best{u}"], o0[u::K]) and np.array_equal(rows[f"idx{u}"], o1[u::K]) for u in range(K))
    wrong = s.data["outs"][out].copy()
    wrong[0, 0] ^= 1
    with pytest.raises(ValueError, match="its chunks and combine select"):
        IR.pieces(replace(s, data={**s.data, "outs": {out: wrong}}))
    with pytest.raises(ValueError, match="its units' chunks are 64 logits, not 16"):
        IR.pieces(s, 16)


def test_a_token_select_by_units_interior_reads_each_wired_row_at_the_unit_the_design_places_it(tmp_path):
    """The interior of two spans of `TokenSelectUnits_v1{V=256, K=2}` inlined in a replay block of six tokens, with the
    wires and rows round 2 derives from that block: the combine's five input ports commit no tree, each a table of the
    units' Q·12 output rows, and token i of span s reads its ``x0`` at unit 12 s + 2 i's third output, its ``best<j>``
    and ``idx<j>`` at unit 12 s + 2 i + j's first and second, which are the rows `pieces` replayed. It derives the
    challenger's ports. With the wires alone every unit of a wired port reads its span's first writer, which these rows
    are not, and rows that place a unit past its span's writers are refused, both before any commitment."""
    V, K, N = 256, 2, 6
    check = bind(DF.TokenSelectUnitsV1, V=V, K=K)
    replay = BR.inlined(check, N)
    program, key = P.population_program(replay, 3), descriptor_id(replay)
    wires, rows = SS.replay_wires(program, key), SS.replay_rows(program, key)
    per = [IR.pieces(s) for s in _selects(tmp_path, V, P.template_id(check)).values()]
    cells, members = (N * K, N), [s.member for s in per[0]]
    it = IR.interior(per, cells, "aa" * 64, [0, 2], row=ROW, salt_key=SALT, wires=wires, rows=rows)
    w = it.committed.window
    mu, mc = w.members
    assert (mc.wired_trees, [p for p, _o, _ws in mc.wired]) == (False, ["x0", "best0", "best1", "idx0", "idx1"])
    at = {p: (o, ws) for p, o, ws in mc.wired}
    for p, o, j in (("x0", "o2", 0), ("best0", "o0", 0), ("best1", "o0", 1), ("idx0", "o1", 0), ("idx1", "o1", 1)):
        assert at[p] == (o, tuple(N * K * s + K * i + j for s in range(Q) for i in range(N)))
        assert w.held(mc, p) == mu.port(o) and mc.port(p) not in it.ports and dict(mc.tables)[p] == Q * N * K
    assert it.ports == IR.interior_ports(members, cells, "aa" * 64, [0, 2], wires, rows=rows)
    with pytest.raises(ValueError, match="rows are not the 'o[0-2]' rows of the"):
        IR.interior(per, cells, "aa" * 64, [0, 2], row=ROW, salt_key=SALT, wires=wires)
    past = [replace(r, runs=(SS.Run(0, N, 0, 3),)) if r.port == "best1" else r for r in rows]
    with pytest.raises(ValueError, match="the design's rows place its 'best1' units at"):
        IR.interior(per, cells, "aa" * 64, [0, 2], row=ROW, salt_key=SALT, wires=wires, rows=past)


def test_a_token_selects_pieces_stage_as_the_second_stage_stages_an_interior_member():
    """`named` gives each piece's member the rows the second stage stages it on (`class_statement.member_stager`: row r
    its Definition's input r), at a vocabulary whose combine rows hold a whole SHA-512 block, while `chunk_rows.
    unit_member`'s positional ``in<g>`` rows are refused there."""
    from verity_flock import class_statement as CST
    for t in ("TokenSelectChunk_v2{C=64}", "TokenSelectCombine_v2{V=4096,C=64}"):
        fn, m = SRW.template(t).fn, CR.unit_member(SRW.template(t), 2)
        CST.member_stager(fn, IR.named(m), None)
        with pytest.raises(ValueError, match="are not the parameters"):
            CST.member_stager(fn, m, None)


def test_a_token_selects_interior_is_template_major_one_shared_row_member_a_piece(tmp_path):
    """The interior of two spans' pieces, each padded to its cell (12 chunk units, 6 combines): the chunks' Q·12 units,
    then the combines' Q·6, each template one shared-row member whose ports are tables of its own rows and unit off_t + k
    reads row k, over the verifier's mixed population program; spans of other templates or cells of another count are
    refused before any commitment."""
    per = [IR.pieces(s) for s in _selects(tmp_path, 128).values()]
    it = IR.interior(per, (12, 6), "aa" * 64, [0, 2], row=ROW, salt_key=SALT)
    w = it.committed.window
    assert [(m.template, m.base, m.n) for m in w.members] == [(per[0][0].member.template, 0, Q * 12),
                                                               (per[0][1].member.template, Q * 12, Q * 6)]
    assert all(dict(m.tables) == {p: m.n for p, _ in m.rows} for m in w.members)
    assert all(np.array_equal(it.committed.refs[j], np.tile(np.arange(m.n)[:, None], (1, len(m.all_rows))))
               for j, m in enumerate(w.members))
    mixed = P.mixed_population_program([(SRW.template(m.template).fn, m.n) for m in w.members])
    assert w.program.root.id == mixed.root.id
    assert it.ports == IR.interior_ports([s.member for s in per[0]], (12, 6), "aa" * 64, [0, 2])
    with pytest.raises(ValueError, match="not the same templates'"):
        IR.interior([per[0], per[1][:1]], (12, 6), "aa" * 64, [0, 2], row=ROW)
    with pytest.raises(ValueError, match=r"2 templates and cells \[12\]"):
        IR.interior(per, 12, "aa" * 64, [0, 2], row=ROW)


def test_a_token_selects_pieces_pad_by_whole_tokens_as_its_inlined_replay_blocks_instances(tmp_path):
    """A replay block inlining `TokenSelect_v4` holds one instance a served token, each its V / C chunks then its combine,
    so the pieces are the templates its body calls in the order of their first call and a span of fewer tokens than its
    class pads by its last token: the chunk cell repeats that token's V / C chunks, not its last chunk, and the interior is
    the interior of the span padded to its class by its last row. Cells that aren't whole tokens are refused by name."""
    t, rows = "TokenSelect_v4{V=128}", _select_rows(128)[5:]
    s = _selects(tmp_path, 128)["b"]
    chunks, comb = IR.pieces(s)
    calls = SRW.template(t).fn.body.nodes
    assert tuple({P.template_id(n.fn): None for n in calls}) == (chunks.member.template, comb.member.template)
    assert (chunks.group, comb.group) == (2, 1)
    pad = chunks.padded(14)["rows"]["x"]
    assert np.array_equal(pad[10:], np.tile(pad[8:10], (2, 1))) and not np.array_equal(pad[10], pad[11])
    whole = np.concatenate([rows, rows[-1:], rows[-1:]])
    fn = SRW.template(t).fn
    SC.write(tmp_path / "whole", t, {"b": {"logits": whole, "tokens": [int(evaluate(fn, r.tolist())[0]) for r in whole],
                                          "steps": list(range(len(whole))), "program": "p" * 64}})
    got, want = (IR.interior([IR.pieces(x)], (14, 7), "aa" * 64, [0], row=ROW, salt_key=SALT)
                 for x in (s, IR.captured(tmp_path / "whole", "b")))
    assert (got.record, got.leaf_layer) == (want.record, want.leaf_layer)
    with pytest.raises(ValueError, match="in whole instances of 2 units"):
        IR.interior([(chunks, comb)], (13, 7), "aa" * 64, [0], row=ROW)
