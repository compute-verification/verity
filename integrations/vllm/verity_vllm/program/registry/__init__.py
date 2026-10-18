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
`conformance` — the separate conformance record: which production kernels implement which registered
           definitions, and each definition's conformance status.

Glossary (Draft 3 §12): a registered definition is a Definition; a Program is a sequence of Calls of Definitions; the Value<w>
above is the encoding-level value type, and a Call's inputs/outputs are its Values.

PROTOCOL SUMMARY (Build stage: the registry = the closed vocabulary every Program is written in)
  Purpose: the registered function library the frontend binds vLLM's traced ATen ops to (Build), the fold names captured kernels with
    (Match) and the replay evaluates bit-exactly with (Commit): primitives with total evaluators on encodings (`prims`), the B1 eager
    composites (`b1`), the sampler chain (`sampling`), the ⊥-lifted families (`lifted`), and the kernel conformance record.
  Inputs: none at import beyond `verity.ml.circuits` (REGISTRY, register_lazy_family); a descriptor's spec ids (e.g. `Const32[0x..]_v1`,
    `GatherBf16x64_v1`, `Lifted[..]_v2`) are resolved on demand through the regex families registered here.
  Outputs: a populated `verity.ml.circuits.defs.REGISTRY`; every Program descriptor (descriptor.json.gz) decodes only against it.
  Invariants: importing ANY `verity_vllm.program.registry.*` module registers the lazy families first (Python imports the package), so a
    descriptor decoded after such an import resolves as it did in veritor; family ORDER matters -- the two catch-alls (`{ORD=n}`,
    `Lifted[..]`) are registered last; a spec id no family matches is a decode failure, never a guessed definition; this module is
    the one verity_vllm/ops/pod_bootstrap.sh's readiness check imports to prove `verity_vllm` resolves from the shipped tree (verity_vllm_from_tree).
  Protocol spec: Notion -> "vLLM integration — protocol spec (Build · Match · Commit)".
"""


# ----------------------------------------------------------------------------------------------------------------------
# On-demand primitive families.  veritor hardcoded these in verity_ir/codec.py:_LAZY_PRIMITIVE_FAMILIES; verity.ml.circuits registers
# only its own `Input<w>` and exposes `register_lazy_family` for semantics packages.  Order matters for the two catch-alls
# (`{ORD=n}` and `Lifted[..]`), which is why they are registered last, exactly as veritor listed them.  Importing any
# `verity_vllm.program.registry.*` module runs this (Python imports the package first), so a descriptor decoded after such an import
# resolves as it did in veritor.  The makers import lazily to keep this package import light and cycle-free.
def _register_lazy_families() -> None:
    from verity.ml.circuits.codec import lookup_primitive, register_lazy_family
    from verity.ml.circuits.defs import REGISTRY

    def _b1():
        from verity_vllm.program.registry import b1
        return b1

    def _prims():
        from verity_vllm.program.registry import prims
        return prims

    def _sampling():
        from verity_vllm.program.registry import sampling
        return sampling

    def _lifted():
        from verity_vllm.program.registry import lifted
        return lifted

    def _topp_word_gates():
        from verity_vllm.program.registry import topp_word_gates
        return topp_word_gates

    def _fp8_moe():
        from verity_vllm.program.registry import fp8_moe_experts
        return fp8_moe_experts

    register_lazy_family(r"^Const([1-9][0-9]*)\[0x([0-9a-f]+)\]_v1$", lambda m: _b1().const(int(m[1]), int(m[2], 16)))  # prims.CONST_ID
    register_lazy_family(r"^GatherBf16x(\d+)_v1$", lambda m: _prims().gather_bf16(int(m[1])))
    # [GLM-5.3] the FP8 MoE expert-row gathers are built per vocabulary while deriving (`fp8_moe_experts._gather`), so a cold decode
    # of such a descriptor (the Match of record, replay) must resolve them by id like GatherBf16
    register_lazy_family(r"^GatherE4m3x(\d+)_v1$", lambda m: _fp8_moe().gather_e4m3(int(m[1])))
    register_lazy_family(r"^GatherF32x(\d+)_v1$", lambda m: _fp8_moe().gather_f32(int(m[1])))
    # [R15 stoch] the top-p keep word (one per vocabulary; [R16] the split count S is its i32 `splits` OPERAND) and the bit extractor
    register_lazy_family(r"^TopPMaskWordx(\d+)_v1$", lambda m: _sampling().topp_mask_word(int(m[1])))
    register_lazy_family(r"^BitAtx(\d+)_v1$", lambda m: _sampling().bit_at(int(m[1])))
    # the keep word as word gates at a constant S (`topp_words`, a single-request Program): its own word gates, which no module a
    # decoder imports for the sampler (`sampling`, GP-01's list) registers
    register_lazy_family(r"^(BitXor|U32Xor|SelectBit|I32ToF32|NvLogf|TopPNumkeep)_v1$", lambda m: getattr(_topp_word_gates(), m[1]))
    # [pad, lifted-padded] the ⊥-lifted families (LIFTING-SPEC §8): the strict lift of a registered primitive is a total function
    # of the original's id, the specified ones of their width/size parameters
    register_lazy_family(r"^LSelect(\d+)_v1$", lambda m: _lifted().l_select(int(m[1])))
    register_lazy_family(r"^LIsActive(\d+)_v1$", lambda m: _lifted().l_is_active(int(m[1])))
    register_lazy_family(r"^LBot(\d+)_v1$", lambda m: _lifted().l_bot(int(m[1])))
    register_lazy_family(r"^LGatherRow(\d+)x(\d+)_v1$", lambda m: _lifted().l_gather_row(int(m[1]), int(m[2])))
    register_lazy_family(r"^LStepLive(\d+)_v1$", lambda m: _lifted().l_step_live(int(m[1])))
    # [pad, R17-32 Continuation_v1 members and R17-42 §3 FixedSelect] a cold decode of a lifted Program must resolve these by id too
    register_lazy_family(r"^LIsEOS(\d+)_v1$", lambda m: _lifted().l_is_eos(int(m[1])))
    register_lazy_family(r"^LEmitNext\[H=(\d+)\]_v1$", lambda m: _lifted().l_emit_next(int(m[1])))
    register_lazy_family(r"^LFixedSelect(\d+)x(\d+)_v1$", lambda m: _lifted().l_fixed_select(int(m[1]), int(m[2])))
    register_lazy_family(r"^LEnable_v1$", lambda m: _lifted().l_enable())  # [pad, LIFTING-SPEC v0.6 B7] the conditional enable
    register_lazy_family(r"^SplitsForSMS(\d+)_v1$", lambda m: _lifted().splits_for_prim(int(m[1])))  # PAD-04 and its lift
    register_lazy_family(r"^LSplitsForSMS(\d+)_v1$", lambda m: _lifted().l_splits_for(int(m[1])))
    # [pad, LIFTING-SPEC v0.7 A11 / G1] a lifted primitive reading the masked operands at their ordinary type
    register_lazy_family(r"^(.+)\{ORD=(\d+)\}$",
                         lambda m: _lifted().with_ordinary(lookup_primitive(REGISTRY, m[1]), int(m[2])))
    # lifted-v1 (propagation semantics, LIFTING-SPEC v0.5.2): legacy, decodes as it was written
    register_lazy_family(r"^Lifted\[(.+)\]_v1$", lambda m: _lifted().lifted_prim_v1(lookup_primitive(REGISTRY, m[1])))
    # padded-v1 (R17-43 explicit enablement, LIFTING-SPEC v0.6 B7): the enable is the last operand
    register_lazy_family(r"^Lifted\[(.+)\]_v2$", lambda m: _lifted().lifted_prim(lookup_primitive(REGISTRY, m[1])))


_register_lazy_families()
