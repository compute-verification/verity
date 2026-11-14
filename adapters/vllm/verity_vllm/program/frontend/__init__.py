"""verity_vllm.program.frontend: the Torch-to-circuit translator.

    Gate -> exactly one Value<w>.
    Composite interfaces -> ordered collections of value references.
    Commitment positions -> physical authentication locations for those values,
                            not semantic components of gate outputs.

`derive` (needs torch) turns a traced torch model into a `Program`. `correspond` (program correspondence across
reference encodings) and `provenance` (the registry's version, which a derivation's provenance holds) are pure
`verity.circuits` and import without torch. Dead gates are `verity.circuits.liveness`. The package re-exports nothing:
a name is imported from the module that defines it, and importing a module loads only what that module imports.

What the translator imports from the rest of the adapter. Its modules are those directly in this directory but
`train_step`, which derives a training step on top of it; `rules/` is not counted here. Every import they make of
`verity_vllm` outside this directory is one of these names, and each name here is imported
(`tests/program/test_frontend_adapter_imports.py`): a new one is added here with what it is for, or done without.

    verity_vllm.settings                                      the declared environment, for the record of what the host set
    verity_vllm.engine.env                                    vLLM's switches, pinned before the export imports vLLM
    verity_vllm.engine.hooks                                  the one owner of a patch on a vLLM, torch or Triton object
    verity_vllm.engine.served_seed                            the served seed's derivation, which `seed_rule` states in gates
    verity_vllm.config.VLLM                                   the pinned vLLM revision, a target's default
    verity_vllm.config.DEFAULT_ROW_PROTOCOLS                  the protocol set a target runs when it declares none
    verity_vllm.config.parse_protocols                        a protocol set's one spelling
    verity_vllm.config.vllm_revision                          the installed vLLM's version with its commit, for the read-back
    verity_vllm.program.inputs                                what may fill a served Program's input, for a rule's run inputs
    verity_vllm.program.inputs.run_field                      the same, for one field a registration reads
    verity_vllm.program.prune.kept_records                    a derivation's body in the registered compiler's form
    verity_vllm.correspondence.runtime.NAMESPACE              the runtime correspondence `emit` writes: its key,
    verity_vllm.correspondence.runtime.SCHEMA                 its schema id
    verity_vllm.correspondence.runtime.correspondence_digest  and its digest

Of the served kernels' circuits (`verity_vllm.definitions`), which the translator binds a graph's launches to:

    verity_vllm.definitions.boolean.boolean_function                    a derived Program's function on bits,
    verity_vllm.definitions.boolean.NoBooleanVersion                    why a word Definition has no Boolean version,
    verity_vllm.definitions.boolean.sigma_check                         and the check of bits against σ of the words
    verity_vllm.definitions.replay.replayer                             the body builder that re-emits a recorded body
    verity_vllm.definitions.numerics.fa3_model                          FlashAttention-3's CPU model, a target's FA3 tile
    verity_vllm.definitions.registry.prims                              the B1 profile's primitives, a target's steps,
    verity_vllm.definitions.registry.b1                                 and its composites; both name the registry's version
    verity_vllm.definitions.registry.dense                              the elementwise, reduction and activation Definitions
    verity_vllm.definitions.registry.gemm_bias                          the linear with a bias
    verity_vllm.definitions.registry.cuda_rmsnorm                       the CUDA RMSNorm kernels
    verity_vllm.definitions.registry.value_inputs                       the Definitions that take a program value as input
    verity_vllm.definitions.registry.attention_vocab                    FlashAttention's launch rules for `Check_inf`,
    verity_vllm.definitions.registry.fa2_check_inf                      FA2's attention chain,
    verity_vllm.definitions.registry.fa3_check_inf                      FA3's,
    verity_vllm.definitions.registry.fa2_softcap                        FA2's with logit soft-capping,
    verity_vllm.definitions.registry.hopper.Fa3InvSum                   and FA3's softmax finalize
    verity_vllm.definitions.registry.fp8                                the FP8 linear,
    verity_vllm.definitions.registry.fp8_vocab                          which blockwise GEMM a scale construction binds,
    verity_vllm.definitions.registry.nvfp4                              and the NVFP4 linear
    verity_vllm.definitions.registry.moe                                the fused-MoE block's expert GEMMs,
    verity_vllm.definitions.registry.fp8_moe                            in FP8 on Hopper,
    verity_vllm.definitions.registry.fp8_moe_experts                    and with the expert index as a value
    verity_vllm.definitions.registry.bf16_gemm_check.LL_MAX_M           the bf16 GEMM's largest M on its low-latency path
    verity_vllm.definitions.registry.b1_tp2.AllReduce                   the TP collectives' Definitions, for `collective_kinds`:
    verity_vllm.definitions.registry.b1_tp2.AllReduce2                  the all-reduces,
    verity_vllm.definitions.registry.b1_tp2.AllGather                   the all-gathers,
    verity_vllm.definitions.registry.b1_tp2.AllGather2
    verity_vllm.definitions.registry.b1_tp2.EmbeddingShard              the embedding shard,
    verity_vllm.definitions.registry.b1_tp2.all_reduce                  the all-reduce and all-gather bound for a world size,
    verity_vllm.definitions.registry.b1_tp2.all_gather
    verity_vllm.definitions.registry.tp_allreduce.AllReduceCustom       and the served all-reduces by kernel
    verity_vllm.definitions.registry.tp_allreduce.AllReduceMnnvl
    verity_vllm.definitions.registry.tp_allreduce.AllReduceMultimem
    verity_vllm.definitions.registry.tp_allreduce.AllReduceNcclTree
    verity_vllm.definitions.registry.tp_allreduce.AllReduceRmsNormMnnvl
    verity_vllm.definitions.registry.b1_tp2.allreduce_order             the all-reduce's hop order, for `export_ops`
    verity_vllm.definitions.registry.sampling.bit_at                    a word's bit, which `seed_rule` states the seed in
    verity_vllm.definitions.registry.stop_rule                          where a request's output stops, for `stop_rule`,
    verity_vllm.definitions.registry.stop_check                         and the check of a request's length
    verity_vllm.definitions.registry.tables.table_type                  a registered table's parameter, for `registration`
    verity_vllm.definitions.registry.topp_geometry.registered_splits    the registered top-p split count, for `vllm_meta`

Of vLLM itself, the modules that turn a graph into a Program make one read, and a guarded one: `targets` and
`target_profile` import `gemm_targets`, whose `vllm_tuned_gemm_arch_family` asks vLLM's own function where vLLM imports
and answers by the pinned revision's rule where it does not. The modules that stand a vLLM model up for the export
(`export_compat`, `export_ops`, `loader_parity`, `vllm_meta`) import vLLM inside the functions that need it.
"""
