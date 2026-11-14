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

Of vLLM itself, the modules that turn a graph into a Program make one read, and a guarded one: `targets` and
`target_profile` import `gemm_targets`, whose `vllm_tuned_gemm_arch_family` asks vLLM's own function where vLLM imports
and answers by the pinned revision's rule where it does not. The modules that stand a vLLM model up for the export
(`export_compat`, `export_ops`, `loader_parity`, `vllm_meta`) import vLLM inside the functions that need it.
"""
