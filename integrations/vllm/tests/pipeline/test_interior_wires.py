"""A two-stage interior whose replay block wires one template's output into another's input (`interior_rows`, ``wires``):
a replay block calling `TokenSelectCombine_v2{V=128}` then `TokenSelectCombine_v2{V=64}` on its token as ``idx``, beside
two `TokenSelectChunk_v2` units on its own logits, the wire round 2 derives from it (`second_stage.replay_wires`), and
its interior over a replay sample of two of three spans.
The wired port commits no tree of its own: its rows are its writer's output rows, each unit reading its span's writer
unit, as round 2's wired read does, it registers on the challenger's own ports, and a commitment made on the device
holds the same refs. A replay whose wired rows aren't its writer's outputs, a wire between templates the interior lacks,
and the challenger's ports without the wire are refused."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from verity.core.protocols.spot import second_stage as SS, service as S
from verity.core.protocols.spot.one_stage import partition as P, registration as R
from verity.ml.circuits.codec import descriptor_id
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.evaluation.reference import evaluate
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import coll_concat, tuple_of
from verity.ml.circuits.types import Array, Tuple
from verity.ml.definitions.registry import dead_free as DF
from verity.ml.windows import rows as CS
from verity_vllm.pipeline import interior_rows as IR, serving_rows as SRW

W, T = bind(DF.TokenSelectCombineV2, V=128, C=64), bind(DF.TokenSelectCombineV2, V=64, C=64)
X = bind(DF.TokenSelectChunkV2, C=64)
SPANS, Q, CELLS, SALT, ROW, DELTA = 3, 2, (1, 1, 2), bytes(range(32)), "interior-wires-test", 2.0 ** -40


def _block():
    """Each span one call of `W` on its parameters, then one of `T` reading `W`'s token as its ``idx``, and a batch of two
    `X` on its own logits, so no cell is proved whole by k = 1."""
    (wp, _wr), (tp, tr), ((_x, xt),), xr = W.signature(), T.signature(), *X.signature()

    def body(B, S_, x0, best, idx, t_x0, t_best, xs):
        tok = B.call(T, t_x0, t_best, coll_concat(Array(1, W.signature()[1]), [B.call(W, x0, best, idx)]))
        return tuple_of(tok, B.batch(X, xs, axes=(0,)))

    return bind(CompositeDefinition("InteriorWiresTest", 0, (), lambda S_: (tuple(wp) + tuple(
        (f"t_{n}", t) for n, t in tp[:2]) + (("xs", Array(2, xt)),), Tuple(tr, Array(2, xr))), body, register=False))


BLOCK = _block()
WIRES = SS.replay_wires(P.population_program(BLOCK, SPANS), descriptor_id(BLOCK))


def _u32(xs) -> np.ndarray:
    return np.asarray(xs, dtype="<u4").view("<u2").reshape(len(xs), -1)


def _spans(seed: int) -> tuple[IR.Span, ...]:
    """One span's `W` unit, `T` unit and two `X` units, each its Definition's on random bf16 words and indices."""
    rng = np.random.default_rng(seed)
    x0, best, idx = int(rng.integers(0, 1 << 16)), rng.integers(0, 1 << 16, 2).tolist(), [3, 64 + 9]
    tok, = evaluate(W, x0, best, idx)
    t_x0, t_best = int(rng.integers(0, 1 << 16)), [int(rng.integers(0, 1 << 16))]
    out, = evaluate(T, t_x0, t_best, [tok])
    got = []
    for fn, ins, o in ((W, [[x0], best, _u32(idx).reshape(-1).tolist()], tok), (T, [[t_x0], t_best, _u32([tok])[0].tolist()], out)):
        m = IR.named(SRW.unit_member(P.template_id(fn), 1))
        data = {"rows": {p: np.asarray([a], dtype=np.uint16) for (p, _w), a in zip(m.rows, ins, strict=True)},
                "outs": {"o0": _u32([o])}}
        got.append(IR.Span(f"s{seed}", m, data, [[0, 1]], "p" * 64))
    xs = rng.integers(0, 1 << 16, (2, 64), dtype=np.uint16)
    bi = [evaluate(X, r.tolist()) for r in xs]
    data = {"rows": {"x": xs}, "outs": {"o0": np.asarray([[b] for b, _i in bi], np.uint16), "o1": _u32([i for _b, i in bi])}}
    return (*got, IR.Span(f"s{seed}", IR.named(SRW.unit_member(P.template_id(X), 2)), data, [[0, 2]], "p" * 64))


def _boundary_window() -> CS.Binding:
    return CS.Binding(P.population_program(BLOCK, SPANS), P.template_instance_query(BLOCK), {"law": "subset", "k": Q},
                      "b0" * 64, (CS.Member(P.template_id(BLOCK), (("ingress", 64),), (("egress", 64),), SPANS),))


def _first(members):
    """A two-stage stream over `W` and `T`, a stand-in boundary registered and its first stage drawn (spans 0 and 2)."""
    names = {}
    for phase, w in (("serve", _boundary_window()), ("interior", IR.interior_window(members, CELLS, "00", [0], WIRES))):
        names |= {m.port(p): (m.schema(p), phase) for m in w.members for p in CS.port_leaves(m)}
    public = (S.Public("population", "structure", "N, the sample's range"),
              S.Public("window", "structure", "provenance: served, on test values"),
              S.Public("leaf_layer", "structure", "the members' public bodies' digest"),
              *(S.Public(f"{k}:{p}", "structure", f"{p}'s layout") for p in sorted(names) for k in ("leaves", "domain")))
    spec = S.Spec(tuple(S.Value(p, s, phase) for p, (s, phase) in sorted(names.items())),
                  S.TwoStage(P.template_instance_query(BLOCK), f"rate:{Q}/{SPANS}", 1),
                  S.Check(tuple(P.program_digest(Program(u)) for u in (W, T, X)), P.template_instances_query((W, T, X)),
                          proof_deadline=60.0), public, b"interior-wires-test")
    draws = iter([[0, 2]])
    stream = S.Stream(spec, (W, T, X), replay=BLOCK, budget=1, delta=DELTA, challenger="challenger-test", clock=lambda: 0.0,
                      sampler=lambda law, n: next(draws) if law["law"] == "subset" else [0] * len(law["strata"]))
    window = _boundary_window()
    c = CS.commit(window, [{"rows": {"ingress": np.zeros((SPANS, 64), np.uint16)},
                            "outs": {"egress": np.ones((SPANS, 64), np.uint16)}}], salt_key=SALT)
    reg = stream.register(CS.registration(c, {"stand_in": "boundary"}, None, registrar=R), CS.expected_ports(window),
                          leaf_layer=CS.leaf_layer({m.key: c.public_body(k) for k, m in enumerate(window.members)}))
    return stream, reg, stream.select(reg)


def _refused(f) -> tuple[str, ...]:
    with pytest.raises(S.Refused) as e:
        f()
    return e.value.verdict.codes


def test_a_wired_port_is_its_writers_output_rows_and_registers_on_the_challengers_ports():
    """Round 2 derives one wire, `T`'s ``idx`` from `W`'s output, the interior's row of that name (`named`). The
    interior commits `W`'s Q tokens once: `T`'s ``idx`` has no tree, its table `W`'s Q output rows, unit u of span s
    reading `W`'s unit s, round 2's wired read. It registers on the challenger's ports derived with the wire, after a
    copy without it is refused, and a device commitment of the same roots holds the same refs. A replay whose ``idx``
    rows aren't `W`'s tokens, and a wire to a template the interior lacks, are refused before any commitment."""
    assert WIRES == (SS.Wire(P.template_id(T), "idx", P.template_id(W), "o0"),)
    per = [_spans(s) for s in (0, 2)]
    members = [s.member for s in per[0]]
    stream, reg, first = _first(members)
    it = IR.interior(per, CELLS, reg.digest, first.units, row=ROW, wires=WIRES)
    w = it.committed.window
    mw, mt, _mx = w.members
    assert (mt.wired, mt.wired_trees, dict(mt.tables)["idx"]) == ((("idx", "o0", (0, 1)),), False, Q)
    assert w.held(mt, "idx") == mw.port("o0") and mt.port("idx") not in it.ports and mw.port("o0") in it.ports
    assert it.committed.refs[1][:, 2].tolist() == [0, 1]
    assert it.ports == IR.interior_ports(members, CELLS, reg.digest, first.units, WIRES)
    assert _refused(lambda: IR.register(stream, first, it, ports=IR.interior_ports(members, CELLS, reg.digest,
                                                                                   first.units))) == ("R3-ports",)
    assert IR.register(stream, first, it).record["population"] == Q * sum(CELLS)
    c = it.committed
    device = IR.DeviceCommit(c.roots, {q: c.bc[q] for q in c.roots}, {q: c.salts[q] for q in c.roots})
    gpu = IR.interior([[replace(s, data={}) for s in p] for p in per], CELLS, reg.digest, first.units, row=ROW,
                      device=device, wires=WIRES)
    assert (gpu.record, gpu.ports, gpu.leaf_layer) == (it.record, it.ports, it.leaf_layer)
    assert (gpu.committed.bc, gpu.committed.salts) == (c.bc, c.salts)
    assert all(np.array_equal(gpu.committed.refs[k][:, :len(m.rows)], c.refs[k][:, :len(m.rows)])
               for k, m in enumerate(w.members))
    bad = [(a, replace(b, data={**b.data, "rows": {**b.data["rows"], "idx": b.data["rows"]["idx"] ^ 1}}), x) for a, b, x in per]
    with pytest.raises(ValueError, match="rows are not the 'o0' rows of the"):
        IR.interior(bad, CELLS, reg.digest, first.units, row=ROW, wires=WIRES)
    with pytest.raises(ValueError, match="templates the window has not both of"):
        IR.interior(per, CELLS, reg.digest, first.units, row=ROW, wires=(replace(WIRES[0], writer="Elsewhere_v1"),))
