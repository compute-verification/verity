"""Registered function library for the B1 (Qwen2.5-1.5B) eager batch-invariant serving program.

    Gate -> exactly one Value<w>.
    Composite interfaces -> ordered collections of value references.
    Commitment positions -> physical authentication locations for those values,
                            not semantic components of gate outputs.

`prims`  — one-output primitives with total evaluators on encodings (bf16 = Value<16>, f32 = Value<32>,
           i32 = Value<32>); hardware-shaped ones (`TcDot16`, MUFU tables) wrap the existing exact models.
`b1`     — composites (`Gemm`, `RMSNormTriton`, `FusedAddRMSNormCuda`, `RoPE`, `Attention`, `SiluMul`,
           `EmbeddingGather`, `LMHead`, `TokenSelect`, `Layer`, `Prefill`, `DecodeStep`, `Serve`) and the
           static configuration record for the eager profile.
`conformance` — the separate conformance record: which served kernels implement which registered
           definitions, and each definition's conformance status.

Glossary (Draft 3 §12): a registered definition is a Definition; a Program is a sequence of Calls of Definitions; the Value<w>
above is the encoding-level value type, and a Call's inputs/outputs are its Values.

PROTOCOL SUMMARY (Build stage: the registry = the closed vocabulary every Program is written in)
  Purpose: the registered function library the frontend binds vLLM's traced ATen ops to (Build), the fold names captured kernels with
    (Match) and the replay evaluates bit-exactly with (Commit): primitives with total evaluators on encodings (`prims`), the B1 eager
    composites (`b1`), the sampler chain (`sampling`), and the kernel conformance record.
  Inputs: none at import beyond `verity.circuits` (register_lazy_family); a descriptor's spec ids (e.g. `Const32[0x..]_v1`,
    `GatherBf16x64_v1`) are resolved on demand through the regex families registered here.
  Outputs: a populated `verity.circuits.defs.REGISTRY`; every Program descriptor (descriptor.json.gz) decodes only against it.
  Invariants: importing ANY `verity_vllm.definitions.registry.*` module registers the lazy families first (Python imports the package), so a
    descriptor decoded after such an import resolves the same way; a spec id no family matches is a decode failure, never a guessed definition.
"""


# ----------------------------------------------------------------------------------------------------------------------
# On-demand primitive families.  veritor hardcoded these in verity_ir/codec.py:_LAZY_PRIMITIVE_FAMILIES; verity.circuits registers
# only its own `Input<w>` and exposes `register_lazy_family` for semantics packages.  Importing any
# `verity_vllm.definitions.registry.*` module runs this (Python imports the package first), so a descriptor decoded after such an import
# resolves as it did in veritor.  The makers import lazily to keep this package import light and cycle-free.
#: Definition ids of a model that is not served, refused by name when no module registered them.
RETIRED = r"^(Lifted\[(Dsv4|Mhc|Marlin|MoeAlign)\w*_v\d+(\{.*\})?\]_v[12]|(Dsv4|Mhc|Marlin|MoeAlign)\w*_v\d+(\{.*\})?)$"
RETIRED_WHY = "Definition {fid} is refused: its model is not served through the one circuit, and its serving path is deleted"


def _register_lazy_families() -> None:
    from verity.circuits.codec import register_lazy_family

    def _b1():
        from verity_vllm.definitions.registry import b1
        return b1

    def _prims():
        from verity_vllm.definitions.registry import prims
        return prims

    def _sampling():
        from verity_vllm.definitions.registry import sampling
        return sampling

    def _topp_word_gates():
        from verity_vllm.definitions.registry import topp_word_gates
        return topp_word_gates

    def _fp8_moe():
        from verity_vllm.definitions.registry import fp8_moe_experts
        return fp8_moe_experts

    def _dsa_topk():
        from verity_vllm.definitions.registry import dsa_topk
        return dsa_topk

    def _dsa_topk_mask():
        from verity_vllm.definitions.registry import dsa_topk_mask
        return dsa_topk_mask

    def _retired(m):
        raise KeyError(RETIRED_WHY.format(fid=m[0]))

    register_lazy_family(r"^Const([1-9][0-9]*)\[0x([0-9a-f]+)\]_v1$", lambda m: _b1().const(int(m[1]), int(m[2], 16)))  # prims.CONST_ID
    register_lazy_family(r"^GatherBf16x(\d+)_v1$", lambda m: _prims().gather_bf16(int(m[1])))
    # [GLM-5.3] the FP8 MoE expert-row gathers are built per vocabulary while deriving (`fp8_moe_experts._gather`), so a cold decode
    # of such a descriptor (the Match of record, replay) must resolve them by id like GatherBf16
    register_lazy_family(r"^GatherE4m3x(\d+)_v1$", lambda m: _fp8_moe().gather_e4m3(int(m[1])))
    register_lazy_family(r"^GatherF32x(\d+)_v1$", lambda m: _fp8_moe().gather_f32(int(m[1])))
    register_lazy_family(RETIRED, _retired)
    # [GLM-5.3] the canonical top-k row's accelerator, one per (row length, TOPK), built while deriving a row past TOPK
    register_lazy_family(r"^DsaTopKLongRowCheck(\d+)x(\d+)_v1$", lambda m: _dsa_topk().long_row_check(int(m[1]), int(m[2])))
    # the mask form's accelerators, one per (row length, TOPK): the threshold `DsaTopKMask` compares each key with (its key and its
    # index) and `DsaTopKMaskCounted`'s, and a slot of the ascending top set (`DsaTopKRow`)
    register_lazy_family(r"^DsaTopKThreshold(Key|Index)(\d+)x(\d+)_v1$",
                         lambda m: _dsa_topk_mask().threshold(int(m[2]), int(m[3]))[m[1] == "Index"])
    register_lazy_family(r"^DsaTopKCountedThreshold(Key|Index)(\d+)x(\d+)_v1$",
                         lambda m: _dsa_topk_mask().counted_threshold(int(m[2]), int(m[3]))[m[1] == "Index"])
    register_lazy_family(r"^DsaTopKSlot(\d+)x(\d+)_v1$", lambda m: _dsa_topk_mask().slot(int(m[1]), int(m[2])))
    # the top-p keep word (one per vocabulary; the split count S is its i32 `splits` OPERAND) and the bit extractor
    register_lazy_family(r"^TopPMaskWordx(\d+)_v1$", lambda m: _sampling().topp_mask_word(int(m[1])))
    register_lazy_family(r"^BitAtx(\d+)_v1$", lambda m: _sampling().bit_at(int(m[1])))
    # the keep word as word gates at a constant S (`topp_words`, the registered S): its own word gates, which no module a
    # decoder imports for the sampler (`sampling`, among the modules the workload Program imports) registers
    register_lazy_family(r"^(BitXor|U32Xor|SelectBit|I32ToF32|NvLogf|TopPNumkeep)_v1$", lambda m: getattr(_topp_word_gates(), m[1]))


_register_lazy_families()
