"""The relations the B-Ligero chain prover can be instantiated for, by ``--relation`` name (lane b-hopper).

A relation is a tensor-core pipeline (``verity.ml.tc.models``: groups, adder width, floor, operand format) proved as a
chain of transition units: the gadget system compiled from ``Params.from_model(model)`` (the BF16 unit of ``unit.py``
or the FP8 unit of ``fp8/relation.py`` -- ONE code path per operand format, parametrised by the model), its hint
generator, the verifier's public decode, and the statement's word widths (``protocol.RelationHooks``; torch is imported only when a runner needs it).  ``relchain.py``
runs any of them: gate, bench, proof dumps, verification.

    bf16-hopper   GemmCoordinate<1536> over sm90.mma.m16n8k16.bf16   HOPPER_BF16_M16N8K16  (16,) / 26 bits / -133; f32_to_bf16 epilogue
    fp8-hopper    GemmAccumulator<1536> over sm90.wgmma.m64n8k32.e4m3 HOPPER_E4M3_K32      (32,) / 14 bits / -139; bare FP32 word
    fp8-ada       GemmAccumulator<1536> over sm89.mma.m16n8k32.e4m3   ADA_E4M3_M16N8K32    (16, 16) / 14 bits / -139; bare FP32 word

The Ampere BF16 relation (``--relation bf16``, the frozen target FIRST on the frozen instance set) stays in ``vu.py``;
``REAL == Params.from_model(AMPERE_BF16_M16N8K16)`` is the same unit at the Ampere widths.

Statement binding: the compiled system's id is in every statement digest (``protocol.statement_digest``), so two
relations never share a transcript.  Every registered relation's statement FILE is the relation-named
``ligero-statement/v4`` (``serialize.py``: the relation's name and word widths are in the header; ``RelationChainRunner.relation``
selects it) -- the Ampere BF16 relation alone keeps v2 / v3, the formats ``backends/ligero-verify`` reads today.  The BF16
relations additionally keep the empty relation tag so that the Rust verifier (which knows no tags and decodes BF16 operands)
can check their proofs with ``--allow-any-system`` once it reads the v4 header; the FP8 relations carry 8-bit operands and
22-bit public words it cannot decode at all.  Until then the file-level check for every registered relation is the Python
file verifier (``serialize.verify_files``: shares code with the prover, a self-check).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from verity.ml.tc.cast import f32_to_bf16_word
from verity.ml.tc.models import ADA_E4M3_M16N8K32, AMPERE_BF16_M16N8K16, HOPPER_BF16_M16N8K16, HOPPER_E4M3_K32, GroupSum
from verity.verification.target import BF16_HOPPER, FIRST, FP8_ADA_MMA, FP8_HOPPER_WGMMA, Target
from verity_numerical.bench import contract
from verity_numerical.checker.params import Params

__all__ = ["RELATIONS", "Relation", "relation"]

K_VU = 1536


@dataclass(frozen=True)
class Relation:
    name: str                              # the --relation value
    target: Target
    model: GroupSum
    params: Params
    compile: Callable                      # (params, chain: bool) -> System
    hints: Callable                        # (params, sys, c_in, a, b, y_out, device) -> (n_hints, l) int64
    public_vectors: Callable               # (params, c, a, b, y, device) -> dict (the verifier's public decode)
    y_public: Callable[[int], int]         # the final FP32 accumulator word -> the chain-end public word
    y_bits: int                            # range of the chain-end public word
    word_dtype: str                        # operand words in the statement digest
    y_dtype: str                           # chain-end words in the statement digest
    tag: bytes                             # relation tag in the statement digest (b"" for the BF16 relations)
    operands: Callable                     # (np.random.Generator, shape) -> finite operand words (int64)
    off_domain: tuple[int, ...]            # operand words the verifier must reject (off the finite domain)
    instances_dataset: str
    instances_tier: str
    instance_seed: int
    relation_name: str
    model_name: str
    backend_name: str
    witness_generation: str
    claim_note: str
    reference_hints: Callable | None = None  # (params, sys, units, device): the host reference generator, for gates
    rust_verifiable: bool = False            # backends/ligero-verify decodes the operands (BF16 words); its statement reader
    #                                          still lacks the v4 header every registered relation writes (see the module doc)
    instances_recipe: str | None = None      # how the synthetic set is drawn (the fingerprint's instances.recipe), if stated

    @property
    def k(self) -> int:
        return self.model.k

    @property
    def steps(self) -> int:
        return K_VU // self.k

    @property
    def hooks(self):
        from .protocol import RelationHooks
        # protocol.verify calls hooks.public_vectors(p, c, a, b, y, device=...) with the system's params: the same ``p``
        return RelationHooks(lambda p, c, a, b, y, device="cpu": self.public_vectors(p, c, a, b, y, device=device),
                             word_dtype=self.word_dtype, y16_dtype=self.y_dtype, y16_bits=self.y_bits, tag=self.tag)


def _bf16_hints(p, sys, c_in, a, b, y_out, device):
    from .hints_torch import hints_torch
    return hints_torch(p, sys, c_in, a, b, y_out, device)


def _bf16_reference_hints(p, sys, units, device):
    from .witness import hints_tensor
    return hints_tensor(p, sys, units, device)


def _fp8_hints(p, sys, c_in, a, b, y_out, device):
    from .fp8.witness import hints_fp8
    return hints_fp8(p, sys, c_in, a, b, device)


def _bf16_public_vectors(p, c, a, b, y, device="cpu"):
    from .unit import public_vectors
    return public_vectors(p, c, a, b, y, device=device)


def _fp8_public_vectors(p, c, a, b, y, device="cpu"):
    from .fp8.relation import public_vectors_fp8
    return public_vectors_fp8(p, c, a, b, y, device=device)


def _bf16_compile(p, chain):
    from .unit import compile_unit
    return compile_unit(p, chain=chain)


def _fp8_compile(p, chain):
    from .fp8.relation import compile_fp8_unit
    return compile_fp8_unit(p, chain=chain)


def _fp8_pack(y: int) -> int:
    from .fp8.relation import pack_public
    return pack_public(y)


def random_finite_bf16(rng: np.random.Generator, shape) -> np.ndarray:
    """Finite BF16 words with exponent fields in [96, 160) (magnitudes 2^-31 .. 2^32: a K=1536 chain stays finite),
    one in ten a signed zero, one in ten a subnormal -- the recipe of the target's reference vectors, vectorised."""
    sign = rng.integers(0, 2, size=shape, dtype=np.int64) << 15
    t = rng.integers(96, 160, size=shape, dtype=np.int64)
    f = rng.integers(0, 128, size=shape, dtype=np.int64)
    kind = rng.integers(0, 10, size=shape)
    w = sign | (t << 7) | f
    w = np.where(kind == 0, sign, w)
    w = np.where(kind == 1, sign | np.maximum(f, 1), w)
    return w


def _fp8_operands(rng: np.random.Generator, shape) -> np.ndarray:
    from .fp8.witness import random_finite_operands
    return random_finite_operands(rng, shape)


def _fp8_params(model):
    from .fp8.relation import FP8Params
    return FP8Params.from_model(model)


_FP8_WITNESS = ("torch hint generator (fp8/witness.py, gen_step semantics on the adder-side params) + torch program "
                "(derived rows)")
_FP8_CLAIM = "public claim per VU = the final FP32 word (22-bit packing: low 10 bits provably zero)"

RELATIONS: dict[str, Relation] = {}


def _register(r: Relation) -> Relation:
    RELATIONS[r.name] = r
    return r


BF16_HOPPER_RELATION = _register(Relation(
    name="bf16-hopper",
    target=BF16_HOPPER,
    model=HOPPER_BF16_M16N8K16,
    params=Params.from_model(HOPPER_BF16_M16N8K16),
    compile=_bf16_compile,
    hints=_bf16_hints,
    public_vectors=_bf16_public_vectors,
    y_public=f32_to_bf16_word,
    y_bits=16,
    word_dtype="<u2",
    y_dtype="<u2",
    tag=b"",
    operands=random_finite_bf16,
    off_domain=(0x7F80, 0xFF80, 0x7FC0),
    # the frozen BF16 Hopper set (contract.BF16_HOPPER_INSTANCES_*): the canonical tables accept exactly this dataset/tier at
    # range [0, 4096) with manifest_sha256 == contract.BF16_HOPPER_INSTANCES_DIGEST == relchain.instances_digest(bf16-hopper, 4096)
    instances_dataset=contract.BF16_HOPPER_INSTANCES_DATASET,
    instances_tier=contract.BF16_HOPPER_INSTANCES_TIER,
    instance_seed=contract.BF16_HOPPER_INSTANCES_SEED,
    instances_recipe="bf16-hopper synthetic (random_finite_bf16: finite BF16 words incl. subnormals and exact zeros, model-recorded accumulators)",
    relation_name="GemmCoordinate<1536> over sm90.mma.m16n8k16.bf16",
    model_name="HOPPER_BF16_M16N8K16",
    backend_name="b-ligero chain, bf16-hopper relation (backends/direct/ligero/unit.py at Params.from_model(HOPPER_BF16_M16N8K16))",
    witness_generation="torch hint generator (hints_torch.py, gen_step semantics at the Hopper widths) + torch program (derived rows)",
    claim_note="public claim per VU = f32_to_bf16(final accumulator), the BF16 word the kernel stores",
    reference_hints=_bf16_reference_hints,
    rust_verifiable=True,
))

FP8_HOPPER_RELATION = _register(Relation(
    name="fp8-hopper",
    target=FP8_HOPPER_WGMMA,
    model=HOPPER_E4M3_K32,
    params=_fp8_params(HOPPER_E4M3_K32),
    compile=_fp8_compile,
    hints=_fp8_hints,
    public_vectors=_fp8_public_vectors,
    y_public=_fp8_pack,
    y_bits=22,
    word_dtype="<u1",
    y_dtype="<u4",
    tag=b"|rel=fp8-hopper|v1",
    operands=_fp8_operands,
    off_domain=(0x7F, 0xFF),
    # the frozen FP8 Hopper set (contract.FP8_HOPPER_INSTANCES_*), pinned like the Ada one below
    instances_dataset=contract.FP8_HOPPER_INSTANCES_DATASET,
    instances_tier=contract.FP8_HOPPER_INSTANCES_TIER,
    instance_seed=contract.FP8_HOPPER_INSTANCES_SEED,
    instances_recipe="fp8-hopper synthetic (uniform finite E4M3 bytes, model-recorded accumulators)",
    relation_name="GemmAccumulator<1536> over sm90.wgmma.m64n8k32.e4m3",
    model_name="HOPPER_E4M3_K32",
    backend_name="b-ligero chain, fp8-hopper relation (backends/direct/ligero/fp8 at FP8Params.from_model(HOPPER_E4M3_K32))",
    witness_generation=_FP8_WITNESS,
    claim_note=_FP8_CLAIM,
))

FP8_ADA_RELATION = _register(Relation(
    name="fp8-ada",
    target=FP8_ADA_MMA,
    model=ADA_E4M3_M16N8K32,
    params=_fp8_params(ADA_E4M3_M16N8K32),
    compile=_fp8_compile,
    hints=_fp8_hints,
    public_vectors=_fp8_public_vectors,
    y_public=_fp8_pack,
    y_bits=22,
    word_dtype="<u1",
    y_dtype="<u4",
    tag=b"|rel=fp8-ada|v1",
    operands=_fp8_operands,
    off_domain=(0x7F, 0xFF),
    # the frozen FP8 Ada set (contract.FP8_ADA_INSTANCES_*): the canonical tables accept exactly this dataset/tier at range
    # [0, 4096) with manifest_sha256 == contract.FP8_ADA_INSTANCES_DIGEST == relchain.instances_digest(fp8-ada, 4096)
    instances_dataset=contract.FP8_ADA_INSTANCES_DATASET,
    instances_tier=contract.FP8_ADA_INSTANCES_TIER,
    instance_seed=contract.FP8_ADA_INSTANCES_SEED,
    instances_recipe="fp8-ada synthetic (uniform finite E4M3 bytes, model-recorded accumulators)",
    relation_name="GemmCoordinate<1536> over sm89.mma.m16n8k32.e4m3",
    model_name="ADA_E4M3_M16N8K32",
    backend_name="b-ligero chain, fp8-ada relation (backends/direct/ligero/fp8)",
    witness_generation=_FP8_WITNESS,
    claim_note=_FP8_CLAIM,
))


# ---------------------------------------------------------------------------------------------------------------------
# The v2 relations (lane relmin-lookup): the same four pipelines compiled with public selection (pubsel/relation.py) -- the
# verifier pins the group maximum of the products and the truncated product sums for every alignment shift, the prover
# commits only what depends on the private accumulator.  Additive: new names beside the v1 ones, which stay the default.
# ---------------------------------------------------------------------------------------------------------------------

def _v2_compile(p, chain):
    from .pubsel.relation import compile_v2_unit
    return compile_v2_unit(p, chain=chain)


def _v2_hints(p, sys, c_in, a, b, y_out, device):
    from .pubsel.hints import hints_v2
    return hints_v2(p, sys, c_in, a, b, y_out, device)


def _v2_reference_hints(p, sys, units, device):
    from .pubsel.hints import reference_hints_v2
    return reference_hints_v2(p, sys, units, device)


def _v2_public_vectors(p, c, a, b, y, device="cpu"):
    from .pubsel.relation import public_vectors_v2
    return public_vectors_v2(p, c, a, b, y, device=device)


_V2_WITNESS = ("torch hint generator (pubsel/hints.py: accumulator components, two shift hints, sign, |sum| in bits, the "
               "normalisation shift one-hot and the normalised significand) + torch program (windows, selectors, products)")
_V2_BACKEND = "b-ligero chain, {name} relation (backends/direct/ligero/pubsel/relation.py at {params})"


def _v2_of(v1: Relation, name: str, params_note: str, **over) -> Relation:
    fields = dict(
        name=name, target=v1.target, model=v1.model, params=v1.params, compile=_v2_compile, hints=_v2_hints,
        public_vectors=_v2_public_vectors, y_public=v1.y_public, y_bits=v1.y_bits, word_dtype=v1.word_dtype, y_dtype=v1.y_dtype,
        tag=v1.tag if v1.tag == b"" else f"|rel={name}|v1".encode(), operands=v1.operands, off_domain=v1.off_domain,
        instances_dataset=v1.instances_dataset, instances_tier=v1.instances_tier, instance_seed=v1.instance_seed,
        instances_recipe=v1.instances_recipe, relation_name=v1.relation_name, model_name=v1.model_name,
        backend_name=_V2_BACKEND.format(name=name, params=params_note), witness_generation=_V2_WITNESS,
        claim_note=v1.claim_note, reference_hints=_v2_reference_hints, rust_verifiable=False)
    fields.update(over)
    return Relation(**fields)


BF16_HOPPER_V2_RELATION = _register(_v2_of(BF16_HOPPER_RELATION, "bf16-hopper-v2", "Params.from_model(HOPPER_BF16_M16N8K16)"))
FP8_HOPPER_V2_RELATION = _register(_v2_of(FP8_HOPPER_RELATION, "fp8-hopper-v2", "FP8Params.from_model(HOPPER_E4M3_K32)"))
FP8_ADA_V2_RELATION = _register(_v2_of(FP8_ADA_RELATION, "fp8-ada-v2", "FP8Params.from_model(ADA_E4M3_M16N8K32)"))

# The Ampere BF16 pipeline on the generic chain runner (the v1 Ampere relation is ``vu.py``'s ``--relation bf16`` on the frozen
# instance set; this one runs on synthetic instances drawn like the Hopper BF16 set, so the two are not the same claim).
BF16_AMPERE_V2_RELATION = _register(_v2_of(
    BF16_HOPPER_RELATION, "bf16-ampere-v2", "Params.from_model(AMPERE_BF16_M16N8K16)",
    target=FIRST, model=AMPERE_BF16_M16N8K16, params=Params.from_model(AMPERE_BF16_M16N8K16),
    instances_dataset="bench-instances-bf16-ampere/v1", instances_tier="vu-k1536-bf16-ampere", instance_seed=20260922,
    instances_recipe="bf16-ampere synthetic (random_finite_bf16: finite BF16 words incl. subnormals and exact zeros, model-recorded accumulators)",
    relation_name="GemmCoordinate<1536> over sm80.mma.m16n8k16.bf16", model_name="AMPERE_BF16_M16N8K16"))


# --- v3: lane relmin-private -- private operand words (committed hint bits pinned to the public words), everything else
# proven in the relation with row-free selection (backends/direct/ligero/privsel/) -----------------------------------------

def _v3_compile(p, chain):
    from .privsel.relation import compile_v3_unit
    return compile_v3_unit(p, chain=chain)


def _v3_hints(p, sys, c_in, a, b, y_out, device):
    from .privsel.hints import hints_v3
    return hints_v3(p, sys, c_in, a, b, y_out, device)


def _v3_reference_hints(p, sys, units, device):
    from .privsel.hints import reference_hints_v3
    return reference_hints_v3(p, sys, units, device)


def _v3_public_vectors(p, c, a, b, y, device="cpu"):
    from .privsel.relation import public_vectors_v3
    return public_vectors_v3(p, c, a, b, y, device=device)


_V3_WITNESS = ("torch hint generator (privsel/hints.py: operand word bits, product bits, the two-stage shifter's one-hots, "
               "digits and aligned limbs, the group maximum's certificate, sign, |sum| in bits, the normalisation shift one-hot "
               "and the normalised significand) + torch program (windows, lookups, products, inverses)")
_V3_BACKEND = "b-ligero chain, {name} relation (backends/direct/ligero/privsel/relation.py at {params})"


def _v3_of(v2: Relation, name: str, params_note: str) -> Relation:
    fields = dict(
        name=name, target=v2.target, model=v2.model, params=v2.params, compile=_v3_compile, hints=_v3_hints,
        public_vectors=_v3_public_vectors, y_public=v2.y_public, y_bits=v2.y_bits, word_dtype=v2.word_dtype, y_dtype=v2.y_dtype,
        tag=f"|rel={name}|v1".encode(), operands=v2.operands, off_domain=v2.off_domain,
        instances_dataset=v2.instances_dataset, instances_tier=v2.instances_tier, instance_seed=v2.instance_seed,
        instances_recipe=v2.instances_recipe, relation_name=v2.relation_name, model_name=v2.model_name,
        backend_name=_V3_BACKEND.format(name=name, params=params_note), witness_generation=_V3_WITNESS,
        claim_note=v2.claim_note, reference_hints=_v3_reference_hints, rust_verifiable=False)
    return Relation(**fields)


BF16_HOPPER_V3_RELATION = _register(_v3_of(BF16_HOPPER_V2_RELATION, "bf16-hopper-v3", "Params.from_model(HOPPER_BF16_M16N8K16)"))
BF16_AMPERE_V3_RELATION = _register(_v3_of(BF16_AMPERE_V2_RELATION, "bf16-ampere-v3", "Params.from_model(AMPERE_BF16_M16N8K16)"))
FP8_HOPPER_V3_RELATION = _register(_v3_of(FP8_HOPPER_V2_RELATION, "fp8-hopper-v3", "FP8Params.from_model(HOPPER_E4M3_K32)"))
FP8_ADA_V3_RELATION = _register(_v3_of(FP8_ADA_V2_RELATION, "fp8-ada-v3", "FP8Params.from_model(ADA_E4M3_M16N8K32)"))


def relation(name: str) -> Relation:
    try:
        return RELATIONS[name]
    except KeyError:
        raise SystemExit(f"unknown --relation {name!r}; known: {sorted(RELATIONS)}") from None
