"""A two-stage sampled audit's interior registration from replay (`pipeline.interior_rows`), on the sampled replay's
CPU-served world (`tests.check.test_sampled_replay`'s RoPE Program at D = 64, three requests): each request's replay
reproduces the RoPE heads serving committed, bit for bit, evaluating only its readers' cone; and on a two-stage
`Stream` over RoPE heads, after a stand-in boundary registration over the three request spans and its first stage, the
replay sample's interior registration is accepted and the second stage samples k in each sampled span, while a forged
leaf, a wrong `previous`, a wrong population and the spans in another order than the replay sample's are refused by
name, and spans replayed from two Programs are not committed."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from verity.core.protocols.sampled_proofs import service as S
from verity.core.protocols.sampled_proofs.one_stage import partition as P, registration as R
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.program import Program
from verity.ml.circuits.types import Array
from verity.ml.definitions.numerics.rows import SAMPLED_MEMBER
from verity.ml.windows import rows as CS
from verity_vllm.check.replay.index import SAMPLER_OP, index_of
from verity_vllm.check.replay.opening import CommittedStore
from verity_vllm.pipeline import interior_rows as IR, serving_rows as SRW

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

    dh, T.DH = T.DH, 64
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
        T.DH = dh


def _weights(world):
    def read(field, lo, hi):
        flo, _fhi = world.woff[field]
        return world.weights[flo + lo:flo + hi]
    return read


def _replay(world, rid: str) -> IR.Replay:
    return IR.Replay(index_of(world.inst), rid, PROMPTS[rid], _weights(world))


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


def _block(n: int):
    """The stand-in replay block: ``n`` RoPE heads, a request span at its size class."""
    params, ret = UNIT.signature()
    return bind(CompositeDefinition("InteriorRowsTestSpan", 0, (),
                                    lambda S_: (tuple((p, Array(n, t)) for p, t in params), Array(n, ret)),
                                    lambda B, S_, *xs: B.batch(UNIT, *xs, axes=(0,) * len(xs)), register=False))


BLOCK = _block(N_RHO)
GRAIN = P.template_instance_query(BLOCK)


def _boundary_window(n: int) -> CS.Binding:
    """The stand-in boundary: ``n`` request spans, each its certified prompt ids in and its sampled tokens out."""
    return CS.Binding(P.population_program(BLOCK, n), GRAIN, {"law": "subset", "k": Q}, "b0" * 64,
                      (CS.Member(P.template_id(BLOCK), (("ingress", 64),), (("egress", 64),), n),))


def _ids(ids) -> np.ndarray:
    a = np.zeros(64, dtype=np.uint16)
    w = np.asarray(ids, dtype="<u4").view("<u2")
    a[:w.size] = w
    return a


def _boundary(world) -> tuple[dict, dict, str]:
    window = _boundary_window(len(PROMPTS))
    tokens = {rid: [int(world.blocks[k][0]) for k in sorted(world.blocks) if k[1] == rid and k[2] == SAMPLER_OP]
              for rid in PROMPTS}
    c = CS.commit(window, [{"rows": {"ingress": np.stack([_ids(p) for p in PROMPTS.values()])},
                            "outs": {"egress": np.stack([_ids(tokens[r]) for r in PROMPTS])}}], salt_key=SALT)
    rec = CS.registration(c, {"stand_in": "boundary"}, None, registrar=R)
    return rec, CS.expected_ports(window), CS.leaf_layer({m.key: c.public_body(k) for k, m in enumerate(window.members)})


def _spec(member: CS.Member) -> S.Spec:
    names = {}
    for phase, w in (("serve", _boundary_window(len(PROMPTS))), ("interior", IR.interior_window(member, N_RHO, "00", [0]))):
        names |= {m.port(p): (m.schema(p), phase) for m in w.members for p in CS.port_leaves(m)}
    public = (S.Public("population", "structure", "N, the sample's range"),
              S.Public("window", "structure", "provenance: served, on test prompts"),
              S.Public("leaf_layer", "structure", "the members' public bodies' digest"),
              *(S.Public(f"{k}:{p}", "structure", f"{p}'s layout") for p in sorted(names) for k in ("leaves", "domain")))
    return S.Spec(tuple(S.Value(p, s, phase) for p, (s, phase) in sorted(names.items())), S.TwoStage(GRAIN, f"subset:{Q}", K),
                  S.Check(P.program_digest(Program(UNIT)), P.template_instance_query(UNIT), proof_deadline=60.0),
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


def _first(world, spans):
    """A two-stage stream over RoPE heads, the stand-in boundary registered and its first stage drawn (spans 0 and 2)."""
    draws = iter([[0, 2]])
    stream = S.Stream(_spec(spans["r0"].member), UNIT, replay=BLOCK, budget=1, delta=DELTA, challenger="challenger-test",
                      clock=_Clock(), sampler=lambda law, n: next(draws) if law["law"] == "subset" else
                      [i * N_RHO for i in range(len(law["strata"]))])
    rec, ports, layer = _boundary(world)
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
                                     "interior": True, "replay": "replay-test"}
    second = stream.select(ireg)
    assert second.replay_sample == first and [u // N_RHO for u in second.units] == [0, 1]


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
