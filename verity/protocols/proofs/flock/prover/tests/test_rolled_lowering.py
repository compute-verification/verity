"""Flock lowers the FP8 group quantisers written as calls of their elements' circuits (`elementwise.rolled`: their `In`, `Core`
and `Out` parts, which have no builder) to the circuit it lowered them to before: an activation of a rolled group is lowered by
its ``fold_build``, the flat circuit's gates in `trace.emit`'s order, never through the calls. Each pin is the lowered circuit's
useful rows and its flock-ir-unit/v2 description's sha256 that `ir_lower.whole` gave with the groups' flat bodies: a group
alone, the row quantisers and the linears that call them."""
from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from verity.circuits import boolean as BL
from verity.circuits.boolean.forms import or_reduce
from verity.circuits.boolean.trace import emit
from verity.circuits.defs import CompositeDefinition, bind
from verity.definitions.ops.boolean import rolled as RO
from verity.circuits.types import Array, Tuple
from verity.definitions.registry import boolean_deepgemm as BDG
from verity_flock import struct_lower as SL
from verity_flock.ir_lower import _sha256, layout, netlist_pieces, whole

#: target -> (useful rows, the description's sha256) with the groups' flat bodies
PINS = {
    "FlashInferFp8Quant1x128Group_v2": (247809, "3b81211e2c18f472773fd3c531e025ced7c3243b9b7978b872fa0add6ebdbe40"),
    "Fp8GroupQuantUe8m0Group_v2{G=128}": (204801, "e5aec782357df065469c02434a827b0f7873418c6293761f84f57f6a1a315fb6"),
    "FlashInferFp8Quant1x128_v2{K=256}": (495617, "f66f4bd52b42d09ad626ed99cb01f334d9bd640cdc9f55ea266a45b955444d51"),
    "Fp8GroupQuantUe8m0_v2{K=256,G=128}": (409601, "5f637954540b4f5f210878c3fd815cbc6b8dd7d7f75a5fdb1262de2b62de7588"),
    "FlashInferFp8Linear_v2{K=256,N=1}": (561665, "196ca0d1903d7e4da86daf05a0ca2ec7201dfa448a68658d3b10611633abb71e"),
    "DeepGemmFp8Linear_v2{K=256,N=1}": (475137, "49ec053637afd505c50949efbae18a74b257636cd4046c3881b5ca1ffe9d3ea1"),
}


def _target(name: str):
    return {
        "FlashInferFp8Quant1x128Group_v2": lambda: bind(BDG.FlashInferFp8Quant1x128GroupV2),
        "Fp8GroupQuantUe8m0Group_v2{G=128}": lambda: bind(BDG.Fp8GroupQuantUe8m0GroupV2, G=128),
        "FlashInferFp8Quant1x128_v2{K=256}": lambda: bind(BDG.FlashInferFp8Quant1x128V2, K=256),
        "Fp8GroupQuantUe8m0_v2{K=256,G=128}": lambda: bind(BDG.Fp8GroupQuantUe8m0V2, K=256, G=128),
        "FlashInferFp8Linear_v2{K=256,N=1}": lambda: bind(BDG.FlashInferFp8LinearV2, K=256, N=1),
        "DeepGemmFp8Linear_v2{K=256,N=1}": lambda: bind(BDG.DeepGemmFp8LinearV2, K=256, N=1),
    }[name]()


def _lowered(target) -> tuple[int, str]:
    U = whole(target)
    lay = layout(U.circuit(), [f"in{j}" for j in range(len(U.in_widths))], [f"out{j}" for j in range(len(U.out_widths))])
    h = hashlib.sha256()
    for piece in netlist_pieces(lay, "unit"):
        h.update(_sha256(piece).digest())
    return lay.useful, h.hexdigest()


def _flat(d, statics=()) -> CompositeDefinition:
    """``d``'s whole circuit as one body, as `elementwise.specialized` writes it."""
    def body(B, S, *c):
        params, ret = d.signature(S)
        return emit(d.traced(S), [t for _, t in params], ret, B, *c)
    return CompositeDefinition(d.name + "Flat", 1, tuple(statics), d.signature, body, register=False)


def test_a_rolled_group_lowers_to_its_flat_circuit() -> None:
    """A group of 2-bit words with two classes and a word of its own lowers to its flat body's circuit, and its templated
    lowering (`struct_lower`, through the calls) is the flat lowering on random inputs."""
    def sig(S):
        return (("x", Array(6, Array(2, BL.BIT))),), Tuple(Array(6, Array(2, BL.BIT)), Array(1, BL.BIT))

    def build(S):
        def run(C, *x):
            d = [C.AND(a[0], a[1]) for a in x]
            s = or_reduce(C, d)
            q = [[C.AND(x[0][0], s), s]] + [[C.AND(d[j], s) ^ x[j][0], C.AND(x[j][1], s)] for j in (1, 2)]
            return q + [[C.AND(C.AND(x[j][0], s), x[j][1] ^ s), d[j]] for j in range(3, 6)] + [[s]]
        return run

    d = RO.rolled("RolledLoweringCheck", 1, [], build, sig, register=False)
    assert len(d.cut(SimpleNamespace())["classes"]) == 2
    assert _lowered(bind(d)) == _lowered(bind(_flat(d)))
    assert SL.check(bind(d))["mismatched_bits"] == 0


@pytest.mark.slow
@pytest.mark.parametrize("name", PINS)
def test_a_rolled_quantiser_lowers_to_the_flat_circuit(name: str) -> None:
    """25 to 55 s and up to 4 GB a target (the lowering): what holds the lowered circuit, the one a proof is of, to the one
    before the groups were rolled; the circuits suite's tests hold only a small group's."""
    assert _lowered(_target(name)) == PINS[name]


@pytest.mark.slow
@pytest.mark.parametrize("name", ["FlashInferFp8Quant1x128Group_v2", "Fp8GroupQuantUe8m0Group_v2{G=128}"])
def test_a_rolled_quantiser_templated_is_its_flat_lowering(name: str) -> None:
    """12 to 20 s a group: `struct_lower` lowers the group through its parts' calls, and the template unrolled is the flat
    lowering on random inputs."""
    target = _target(name)
    assert SL.check(target)["mismatched_bits"] == 0
    _t, lw, _sp = SL.lower(target)
    group = name.split("{")[0].removesuffix("_v2")
    assert {f"{group}{p}_v2" for p in ("In", "Core", "Out")} <= {k[0].split("{")[0] for k in lw.specs}
