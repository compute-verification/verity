"""The applicability contract of an ahead-of-time constructed Program (cb-fix).

Three objects, all pure Python (no torch; testable on any host):

* `TargetProfile` — the ONE declared input a construction is built for: compute capability, SM count, batch-invariant switch,
  attention-backend knob, pinned vLLM revision, numerics profile (derive vocabulary). Everything the builder's shims answer to vLLM
  and everything the manifest reports as "declared" comes from this object; there is no second copy of any of these values.
* `ArtifactIdentity` — the cache key / wrapper identity of a constructed artifact:
      sha256( canonical(construction_version) || canonical(TargetProfile) || canonical(applicability) || program_digest )
  The Program digest is a component, not a function of the profile: two targets that genuinely produce the same circuit share the
  Program digest and differ in identity; two hosts building the same profile share both (no host fact enters any component).
* `check_reuse(artifact, profile, applicability_now)` — fail-closed: an artifact may be reused only under the profile it was built
  for and only when every applicability condition it was built under holds for the requester; the refusal names every differing
  field with both values.

Canonical bytes = `json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)`; tuples are serialised as lists.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from typing import TYPE_CHECKING, Any

from verity_vllm.config import DEFAULT_ROW_PROTOCOLS, VLLM, parse_protocols
from verity_vllm.program.frontend.kernel_readback import applied_kernels, fp8_block_gemm_kernel_of, int8_exact_of, moe_experts_class_of  # noqa: F401

if TYPE_CHECKING:
    from verity_vllm.program.frontend.gemm_targets import GemmTarget

SCHEMA = "verity/aot-artifact/v1"


def canonical_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=_json_default).encode()


def _json_default(o):
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(f"not canonicalisable: {type(o).__name__}")


def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@dataclass(frozen=True)
class TargetProfile:
    """The declared target. Field semantics = what the builder APPLIES with it (see `verity_vllm.program.frontend.export_compat`):

    compute_capability  -> vLLM `current_platform.get_device_capability()` (instantiation AND export; selects the FA version, the
                           batch-invariant GEMM override family and the tuned-GEMM table)
    num_sms             -> vLLM `num_compute_units()` (the persistent GEMM grid; the GEMM rule checks every launch against it)
    batch_invariant     -> env `VLLM_BATCH_INVARIANT` (the switch `RMSNorm.forward_cuda` / `init_batch_invariance` read)
    attention_backend   -> `EngineArgs(attention_backend=...)`; None = vLLM's selector for the declared capability
    flash_attn_version  -> `EngineArgs(attention_config={"flash_attn_version": v})` — the ONLY FA-version input of the
                           pinned vLLM (`fa_utils.get_flash_attn_version` step 2; no env knob exists in d9105ea80). None = the
                           selector's default for the declared capability (`effective_flash_attn_version`: 3 on cc 9.x, 2 below);
                           2 on cc 9.x = the labelled "forced FA2 on Hopper" execution (`execution_label` -> `bi-eager-fa2`).
                           The builder reads the instantiated layers' version back and refuses on a mismatch; the eager Match
                           resolver and the collector receive the same object (run header `profile.target`).
    vllm_revision       -> must be a prefix-match of the running `vllm.__version__` (`+g<rev>`), else refused
    numerics_profile    -> the derive vocabulary (`--profile`); already part of the Program's `SRC` static
    compilation         -> the EXECUTION the profile declares (`execution_label` -> `bi-compiled`): None = eager
                           (`enforce_eager=True`, every module forward launches its own kernels); `"piecewise"` = vLLM torch.compile
                           (`CompilationMode.VLLM_COMPILE`, Inductor) + CUDA graphs `CUDAGraphMode.PIECEWISE` (attention = splitting op,
                           runs outside the graphs; the decode graphs are captured per batch size and REPLAYED).  Applied at run time as
                           `enforce_eager=False, compilation_config={"cudagraph_mode": "PIECEWISE", "inductor_compile_config":
                           {"emulate_precision_casts": false}}` (`piecewise-emu`: true) (`engine_args()`; the run header records it, GM-01 G1 compares it with the
                           declaration).  `emulate_precision_casts` (Inductor's own switch) makes every fused Triton kernel perform the bf16
                           roundings the Program specifies at each op boundary; default Inductor carries fp32 through a fused region, so the
                           Program's intermediate values do not exist in that execution and any observation of them changes it
                           (instrumented != production).  With the emulation the collection boundary is invisible by
                           construction (measured: production vs instrumented top-8 logprobs bit-identical over 32 steps).
                           The Program itself is DERIVED eagerly by construction (`engine_args(execution=False)` omits the compile knobs);
                           what the declaration claims for the compiled execution is stated in `modelled_engine()` (boundary-level
                           conformance on committed bytes; fused region == composed relations NOT established).
                           Omitted from `to_json()` when None: eager profiles serialise (and digest) exactly as before.
    fp8_block_gemm      -> env `VLLM_USE_DEEP_GEMM`: the ONLY knob of the pinned vLLM's block-scaled FP8 linear kernel choice
                           (`kernels/linear._POSSIBLE_FP8_BLOCK_KERNELS[CUDA]`, first that `can_implement`: FlashInfer-DeepGEMM,
                           DeepGEMM, CUTLASS, ...).  `"cutlass"` -> `VLLM_USE_DEEP_GEMM=0` (both DeepGEMM kernels excluded; on cc 9.x
                           the selector takes `CutlassFp8BlockScaledMMKernel`, the sm90 blockwise CUTLASS GEMM whose arithmetic is the
                           registered `ScaledMmFp8Block_v1`); None = vLLM's own selection (DeepGEMM JIT when its toolchain exists —
                           no registered Definition: the FP8 rules refuse its custom op by name).  Builder AND runtime read the selected
                           kernel class back from the instantiated `Fp8LinearMethod.fp8_linear` and refuse a mismatch; a model without
                           FP8 linears makes a declared kernel a mismatch too (nothing to select — the declaration would be idle).
                           Omitted from `to_json()` when None so every profile digest from before the field is unchanged.
    int8_exact          -> which layers run the plain exact-int8 arm (`INT8_EXACT_SCOPES`): every linear (`engine.int8_exact`, an offline int8
                           checkpoint, vLLM quantization `INT8_EXACT_QUANTIZATION`), every attention layer (`engine.int8_attention`, the LUT
                           softmax, attention backend `INT8_EXACT_ATTENTION_BACKEND`), or both; read back from every layer (`int8_exact_of`).  Omitted from `to_json()` when None.
    moe_backend         -> `EngineArgs.moe_backend`, the fused-MoE experts kernel (applied to `KernelConfig.moe_backend`; an explicit backend that
                           cannot run raises in vLLM's `select_*_moe_backend`, never falls back).  `"triton"` -> `TritonExperts` (`fused_moe_kernel`,
                           the launch `MoeExpertGemm_v1` and the block-FP8 `MoeExpertFp8Gemm_v1` state); None = vLLM's selection.  Builder and runtime
                           read the class back from every fused-MoE layer's `quant_method.experts_cls` and refuse a mismatch; the header attests it
                           (`backends.moe_experts_class`) only when declared, so undeclared MoE runs keep their ids.  Omitted from `to_json()` when None.
    async_scheduling    -> [DP-pad-02 (b)] the serving profile's RESOLVED async-scheduling configuration the workload Program is
                           SPECIALIZED for (`SchedulerConfig.async_scheduling` after the pinned vLLM's resolution, `config/vllm.py:1273-1322`
                           of d9105ea80).  Under async scheduling step t+1 is scheduled before step t's sample is known, so a request that
                           samples an EOS at step t is forwarded ONCE more at t+1 (LAG = 1: active internal computation of the Program whose
                           sampled token is not a served output); a synchronous scheduler computes no such row (LAG = 0).  None = not
                           declared explicitly: `effective_async_scheduling()` states the pinned resolver's value for this profile (True: a
                           generate-runner model without speculative decoding on the CUDA uniproc/mp executors) -- a DECLARATION the Program
                           carries (`lag()`), verified by GM-01 G1 against the record header's resolved read-back
                           (`RunHeader.profile.engine_kwargs.async_scheduling`, `vllm_adapter.resolved_async_scheduling`); nothing is ever
                           inferred from the trace.  Omitted from `to_json()` when None so every profile digest from before the field is unchanged.
    moe_construction    -> [route A (3)] HOW the fused-MoE block of a mixture-of-experts model is CONSTRUCTED in the
                           Program (a Program STATIC, declared-and-bound, NEVER attested: the record is construction-agnostic -- the same
                           trace serves both constructions).  `"indexed-read"` = MOE-01: router ->
                           TOPK indexed pair rows `MoeExpertGemm_v1` / `MoeExpertGemmW_v1` -> `MoeSum_v1`, the expert id an ordinary value
                           operand (`verity_vllm.program.frontend.rules.vllm_moe` MOE-01 rules).  `"indexed-read-ordered"` (the default when
                           None) = MOE-01 with the router stated in its selection order over one committed softmax
                           (`MoeRouterTopKOrdered_v1`).  `"padded"` = the
                           fixed-slot construction: ONE `MoeFfnFixedL_v1{E,TOPK,VPT,H,I}` call per token over ALL E experts on their own
                           weights, the TOPK slot outputs selected by the router ids (`verity.ml.definitions.registry.moe_pad`; its lift of record is the
                           hand-authored `LMoeFfnL_v1` with the unselected slots DISABLED by the enable vector `a_{r,t,e}`).  Two Programs
                           of the same trace differ in digest by construction; GP-01 refuses a workload whose components were derived under
                           another construction than the declared one, and labels the workload (`moe_construction` block) so the row's
                           population is reported as what it is (no forced #Q correspondence).  A dense model ignores the
                           field (no MoE block to construct).  Omitted from `to_json()` when None so every earlier profile digest is unchanged.
    fa3_construction    -> HOW FA3's attention rows are constructed (a Program static, declared-and-bound, never attested).
                           `"guard-every-block"` binds `Attention_v2`, whose blocks all guard a running max of -inf;
                           `"check-inf-per-iteration"` (the default when None) binds `Attention_v4` (`registry.fa3_check_inf`), which
                           guards only the kernel's `Check_inf` iterations (the first and the causal-masked ones) as FA3's mainloop does.
                           Only an effective FA3 target reads it.  Omitted from `to_json()` when None.
    weight_only_calls   -> how often the Program issues a Call that reads only weights (a Program static, declared-and-bound, never
                           attested).  `"per-forward"` = as vLLM executes it: every forward issues its own, so a request Program of T+1
                           forwards computes the same value T+1 times (Gemma's `GemmaRMSNorm` weight + 1).  `"once"` (the default when
                           None) = the Program issues it at its first forward and every later forward reads that value (Build: the
                           `vllm_bindings.norm_chain` rules; Match: the fold issues each weight-only Call once per run).  Omitted from
                           `to_json()` when None so every earlier profile digest is unchanged.
    sampler_construction -> how the stochastic token select is constructed (a Program static, declared-and-bound, never attested).
                           `"greedy-check-per-stage"` binds `GumbelTokenSelect_v1` / `GumbelTopPTokenSelect_v1`, whose temperature scale
                           and Gumbel select each compute `temp == 0`; `"shared-greedy"` (the default when None) binds their
                           `registry.sampling_vocab.SHARED_GREEDY` entry (the same function), which computes it once for both.  Omitted from
                           `to_json()` when None.
    fp8_scale_construction -> how the blockwise FP8 GEMM's scale products `sx[kb] * sw[nb][kb]` are constructed (a Program static,
                           declared-and-bound, never attested).  `"shared-per-block"` (the default when None) binds
                           `ScaledMmFp8BlockSharedScale_v1`: each weight block's K/G products once, read by its G coordinates.
                           `"per-coordinate"` binds `ScaledMmFp8Block_v1`, whose every coordinate forms its own.  Omitted from `to_json()` when None.
    protocols           -> the protocol set the run executes (`verity_vllm.protocols`; grammar `config.parse_protocols`).  None =
                           sampled proofs alone (`config.DEFAULT_ROW_PROTOCOLS`), the Commit as it has always run.  Declared, it is the
                           whole set in its one spelling, sampled proofs included (a profile is the target of sampled proofs' Program);
                           the default spelled out is refused (declare None).  The Commit installs the other protocols beside its
                           committer (`protocol_options.at_commit`) and refuses any it cannot trace (an executor needs the Definition
                           of what it computes).  Omitted from `to_json()` when None.
    ir                  -> the circuit profile the Program is written in (a Program static, declared-and-bound, never attested).  None = word
                           (every Call of the Definition its derive rule binds, on w-bit values).  `"boolean"` = the same Calls,
                           each of the Boolean version of that Definition, on bits (`verity_vllm.program.frontend.boolean.derived`); the
                           Build refuses by name every Definition with no Boolean version.  Declaring `"word"` is refused (declare None).
                           Omitted from `to_json()` when None.
    registered_values   -> where the Program's values live (a Program static, declared-and-bound, never attested).  None = in the
                           Definitions, as constants.  True = the norms' eps and the attention scale are fields of the registered
                           `values` parameter, read by the Definitions that take them as inputs (`RMSNormTriton_v3`,
                           `RMSNormFusedCuda_v4`, `Attention_v10` / `Attention_v11`); a Definition that has no such version keeps
                           its constants.  Under `ir = "boolean"` each is its Boolean version, which reads the same field's bits.
                           Declaring false is refused (declare None).  Omitted from `to_json()` when None.

    Derived (not fields; one rule each, `verity_vllm.program.frontend.targets` consumes them): `tensor_core_family` (the registered k16
    tensor-core step of the capability: `ampere` for sm_8x, `hopper` for sm_90, `blackwell_consumer` for sm_120) and `tuned_gemm_arch_family` (vLLM's tuned
    batch-invariant GEMM table for the capability: `ada` 8.9, `hopper` 9.x, None otherwise — mirrors
    `batch_invariant_configs._get_tuned_matmul_arch_family`, read back by the builder).
    """
    compute_capability: tuple[int, int] = (8, 9)
    num_sms: int = 128
    batch_invariant: bool = True
    attention_backend: str | None = None
    vllm_revision: str = VLLM
    numerics_profile: str = "b1-eager-v3"
    flash_attn_version: int | None = None
    compilation: str | None = None

    #: the one admitted compiled execution and the vLLM facts it applies / declares (`vllm/config/compilation.py`)
    #: keys from GP-01's COMPILATION_FACT_KEYS (mode, cudagraph_mode, capture_sizes, use_inductor_graph_partition, compile_cache) in the run
    #: header's read-back form (vllm_adapter.compilation_facts); `capture_sizes` (depends on the row's batch) and `compile_cache` (a process
    #: switch; cached and fresh Inductor graphs are the same code) are NOT declared = not constrained (GM-01 compares only keys both sides carry)
    # `piecewise` = vLLM's PRODUCTION DEFAULT compiled configuration (default Inductor config: fp32 carried through
    # the fused regions) -- the panel's bi-compiled workloads; its Program is P_compiled at Inductor-kernel granularity (S1/S2).  `piecewise-emu`
    # = the SEPARATE declared specialization with Inductor emulate_precision_casts (the eager roundings performed inside the fused kernels;
    # a different numerical computation) -- not a panel workload unless added by a 7-field note.
    COMPILATIONS = {"piecewise": {"mode": "VLLM_COMPILE", "backend": "inductor", "cudagraph_mode": "PIECEWISE", "use_inductor_graph_partition": False,
                                  "emulate_precision_casts": False,
                                  "attention": "splitting op (vllm::unified_attention_with_output) executed outside the CUDA graphs"},
                    "piecewise-emu": {"mode": "VLLM_COMPILE", "backend": "inductor", "cudagraph_mode": "PIECEWISE", "use_inductor_graph_partition": False,
                                      "emulate_precision_casts": True,
                                      "attention": "splitting op (vllm::unified_attention_with_output) executed outside the CUDA graphs"}}
    fp8_block_gemm: str | None = None
    int8_exact: str | None = None        # the layers on exact int8 (None = none; see the doc); applied and read back
    #: the fused-MoE experts kernel (None = vLLM's selection; see the doc); applied and read back
    moe_backend: str | None = None
    #: the resolved async-scheduling configuration the Program is specialized for (None = the pinned resolver's value, see the doc)
    async_scheduling: bool | None = None
    #: the fused-MoE block's construction in the Program (None = "indexed-read-ordered"; see the doc); never attested
    moe_construction: str | None = None
    #: FA3's attention construction (None = "check-inf-per-iteration", `Attention_v4`; see the doc)
    fa3_construction: str | None = None
    #: how often the Program issues a weight-only Call (None = "once"; see the doc); never attested
    weight_only_calls: str | None = None
    #: the stochastic token select's construction (None = "shared-greedy"; see the doc); never attested
    sampler_construction: str | None = None

    #: the blockwise FP8 GEMM's scale-product construction (None = "shared-per-block"; see the doc); never attested
    fp8_scale_construction: str | None = None
    #: the protocol set the run executes (None = sampled proofs alone; see the doc); declared, never attested
    protocols: tuple[str, ...] | None = None
    #: the circuit profile the Program is written in (None = word; see the doc); declared, never attested
    ir: str | None = None
    #: the Program's values as registered inputs (None = constants; see the doc); declared, never attested
    registered_values: bool | None = None

    IRS = ("boolean",)
    WEIGHT_ONLY_CALLS = ("per-forward", "once")
    DEFAULT_WEIGHT_ONLY_CALLS = "once"
    SAMPLER_CONSTRUCTIONS = ("greedy-check-per-stage", "shared-greedy")
    DEFAULT_SAMPLER_CONSTRUCTION = "shared-greedy"
    FP8_SCALE_CONSTRUCTIONS = ("per-coordinate", "shared-per-block")
    DEFAULT_FP8_SCALE_CONSTRUCTION = "shared-per-block"

    # the registered block-scaled FP8 GEMM selections: value -> the vLLM kernel class the selector must have taken
    FP8_BLOCK_GEMM_KERNELS = {"cutlass": "CutlassFp8BlockScaledMMKernel"}
    # the exact-int8 scopes (steps.py's plain specs `int8-linears`, `int8-attention`, `int8-all`), and what applies each part
    INT8_EXACT_SCOPES = {"linears": ("linears",), "attention": ("attention",), "all": ("linears", "attention")}
    INT8_EXACT_QUANTIZATION, INT8_EXACT_ATTENTION_BACKEND = "verity_int8", "CUSTOM"
    # the registered fused-MoE backends: value -> the experts class every fused-MoE layer must carry
    MOE_BACKEND_KERNELS = {"triton": "TritonExperts"}

    #: the registered MoE constructions: value -> the frontend rule family that owns the block's leaf kernels and the Definitions
    #: it binds (one Program per construction; the digest differs by construction on the same trace)
    MOE_CONSTRUCTIONS = {"indexed-read": {"rules": "vllm_moe MOE-01 (TopkSoftmax / MoeAlignBlockSize / FusedMoeKernel / MoeSum)",
                                          "definitions": ("MoeRouterTopK_v1{E,TOPK,VPT}", "MoeExpertGemm_v1{E,K,N}", "MoeExpertGemmW_v1{E,K,N}",
                                                          "SiluMul_v2{I}", "MoeSum_v1{TOPK,H}"),
                                          "population": "|Q(P)|: one VU per (token, slot) pair row + router + sum per token"},
                         "padded": {"rules": "vllm_moe padded (PaddedTopkSoftmax / PaddedMoeAlignBlockSize / PaddedFusedMoeKernel / PaddedSiluAndMul / "
                                             "PaddedMoeSum -> ONE MoeFfnFixedL_v1 per token)",
                                    "definitions": ("MoeFfnFixedL_v1{E,TOPK,VPT,H,I}",),
                                    "lift_of_record": "LMoeFfnL_v1{E,TOPK,VPT,H,I} (hand-authored; registry.lifted.register_hand_lift)",
                                    "population": "|Q(Lift(P_fixed-slot))|: every expert slot of every token, the unselected ones DISABLED (bot) -- "
                                                  "labelled with its actual population, never equated to the indexed-read row's (R17-45 (1))"},
                         "indexed-read-ordered": {"rules": "vllm_moe MOE-01, the router stated in its selection order over one committed softmax (the same function as MOE-01's)",
                                                 "definitions": ("MoeRouterTopKOrdered_v1{E,TOPK,VPT}", "MoeExpertGemm_v1{E,K,N}", "MoeExpertGemmW_v1{E,K,N}",
                                                                 "SiluMul_v2{I}", "MoeSum_v1{TOPK,H}"),
                                                 "population": "indexed-read's; the router commits its softmax once (2E+2 values per token), not the rounds' masked scores"}}
    DEFAULT_MOE_CONSTRUCTION = "indexed-read-ordered"
    #: the registered FA3 attention constructions: value -> the attention Definition every FA3 row binds
    FA3_CONSTRUCTIONS = {"guard-every-block": {"definition": "Attention_v2{T,NH,KVH,D,BN,DOT,INV}",
                                               "rule": "every block guards a running max of -inf (AttnBlock_v2)"},
                         "check-inf-per-iteration": {"definition": "Attention_v4{T,NH,KVH,D,BN,DOT,INV,MASKED_FROM}",
                                                     "rule": "block b guards iff b >= the tile's n_block_min_causal_local_mask (AttnBlock_v4{CHECK})"}}
    DEFAULT_FA3_CONSTRUCTION = "check-inf-per-iteration"

    #: the pinned vLLM's resolution of `SchedulerConfig.async_scheduling is None` (d9105ea80 `VllmConfig.__post_init__`,
    #: config/vllm.py:1273-1322): True unless the model is a pooling runner, a speculative method other than EAGLE/MTP/draft-model/
    #: NGram-GPU/DSpark is configured, the distributed executor does not support async scheduling, or ROCm DeepEP-HT DBO is on.  None of
    #: these holds for an admitted profile (generate runner, no speculative config, CUDA uniproc / mp executor), so the mirrored value is
    #: True; the record's resolved read-back is compared with the declaration by GM-01 G1 -- a profile for which the mirror is wrong FAILS
    #: by name there, it is never absorbed.
    PINNED_ASYNC_SCHEDULING_RESOLUTION = {"vllm_revision": "d9105ea80", "source": "vllm/config/vllm.py:1273-1322 (VllmConfig.__post_init__)",
                                          "value_when_unset": True,
                                          "false_when": ("model_config.runner_type == 'pooling'",
                                                         "speculative_config.method not in EAGLE/MTP/draft_model/NGram-GPU/DSpark",
                                                         "speculative_config.disable_padded_drafter_batch",
                                                         "not executor_class.supports_async_scheduling()",
                                                         "ROCm and enable_dbo and all2all_backend == 'deepep_high_throughput'")}

    def __post_init__(self) -> None:
        cc = tuple(int(x) for x in self.compute_capability)
        if len(cc) != 2 or cc[0] < 1 or cc[1] < 0:
            raise ValueError(f"compute_capability must be (major, minor), got {self.compute_capability!r}")
        object.__setattr__(self, "compute_capability", cc)
        if int(self.num_sms) < 1:
            raise ValueError(f"num_sms must be positive, got {self.num_sms!r}")
        object.__setattr__(self, "num_sms", int(self.num_sms))
        object.__setattr__(self, "batch_invariant", bool(self.batch_invariant))
        if self.flash_attn_version is not None:
            v = int(self.flash_attn_version)
            if v not in (2, 3):
                raise ValueError(f"flash_attn_version must be None (vLLM's selector), 2 or 3; got {self.flash_attn_version!r}")
            object.__setattr__(self, "flash_attn_version", v)
        if self.compilation is not None and self.compilation not in self.COMPILATIONS:
            raise ValueError(f"compilation must be None (eager) or one of {sorted(self.COMPILATIONS)}; got {self.compilation!r}")
        if self.fp8_block_gemm is not None and self.fp8_block_gemm not in self.FP8_BLOCK_GEMM_KERNELS:
            raise ValueError(f"fp8_block_gemm must be None (vLLM's selector) or one of {sorted(self.FP8_BLOCK_GEMM_KERNELS)}; "
                             f"got {self.fp8_block_gemm!r}")
        if self.int8_exact is not None and self.int8_exact not in self.INT8_EXACT_SCOPES:
            raise ValueError(f"int8_exact must be None (no int8 layers) or one of {list(self.INT8_EXACT_SCOPES)}; got {self.int8_exact!r}")
        if "attention" in self.int8_parts and self.attention_backend is not None:
            raise ValueError(f"int8_exact {self.int8_exact!r} selects the attention backend; attention_backend must be None")
        if self.moe_backend is not None and self.moe_backend not in self.MOE_BACKEND_KERNELS:
            raise ValueError(f"moe_backend must be None (vLLM's selector) or one of {sorted(self.MOE_BACKEND_KERNELS)}; got {self.moe_backend!r}")
        if self.async_scheduling is not None and not isinstance(self.async_scheduling, bool):
            # a declared value is a boolean (JSON true/false) -- "1"/"true" strings are not accepted (one spelling, no coercion)
            raise ValueError(f"async_scheduling must be None (the pinned resolver's value), true or false; got {self.async_scheduling!r}")
        if self.moe_construction is not None and self.moe_construction not in self.MOE_CONSTRUCTIONS:
            # one spelling per construction; "MOE-01" / "fixed-slot" / booleans are not accepted (no coercion, no alias)
            raise ValueError(f"moe_construction must be None ({self.DEFAULT_MOE_CONSTRUCTION}) or one of {sorted(self.MOE_CONSTRUCTIONS)}; "
                             f"got {self.moe_construction!r}")
        if self.fa3_construction is not None and self.fa3_construction not in self.FA3_CONSTRUCTIONS:
            raise ValueError(f"fa3_construction must be None ({self.DEFAULT_FA3_CONSTRUCTION}) or one of {sorted(self.FA3_CONSTRUCTIONS)}; "
                             f"got {self.fa3_construction!r}")
        if self.weight_only_calls is not None and self.weight_only_calls not in self.WEIGHT_ONLY_CALLS:
            raise ValueError(f"weight_only_calls must be None ({self.DEFAULT_WEIGHT_ONLY_CALLS}) or one of {list(self.WEIGHT_ONLY_CALLS)}; "
                             f"got {self.weight_only_calls!r}")
        if self.sampler_construction is not None and self.sampler_construction not in self.SAMPLER_CONSTRUCTIONS:
            raise ValueError(f"sampler_construction must be None ({self.DEFAULT_SAMPLER_CONSTRUCTION}) or one of "
                             f"{list(self.SAMPLER_CONSTRUCTIONS)}; got {self.sampler_construction!r}")
        if self.fp8_scale_construction is not None and self.fp8_scale_construction not in self.FP8_SCALE_CONSTRUCTIONS:
            raise ValueError(f"fp8_scale_construction must be None ({self.DEFAULT_FP8_SCALE_CONSTRUCTION}) or one of "
                             f"{list(self.FP8_SCALE_CONSTRUCTIONS)}; got {self.fp8_scale_construction!r}")
        if self.protocols is not None:
            if isinstance(self.protocols, str):
                raise ValueError(f"protocols is a list of name:scheme entries, got the string {self.protocols!r}")
            got = tuple(self.protocols)
            canon = parse_protocols(got)
            if got != canon:
                raise ValueError(f"protocols must be spelled {list(canon)} (one entry per protocol, in install order); got {list(got)}")
            if not any(e.startswith("sampled-proofs:") for e in canon):
                raise ValueError(f"protocols {list(canon)} lacks sampled-proofs: a TargetProfile is the target of sampled proofs' Program")
            if canon == DEFAULT_ROW_PROTOCOLS:
                raise ValueError(f"protocols {list(canon)} is the default (sampled proofs alone): declare None")
            object.__setattr__(self, "protocols", canon)
        if self.ir is not None and self.ir not in self.IRS:
            raise ValueError(f"ir must be None (word) or one of {list(self.IRS)}; got {self.ir!r}")
        if self.registered_values is not None and self.registered_values is not True:
            raise ValueError(f"registered_values must be None (constants) or true; got {self.registered_values!r}")

    # -- derived facts (one rule each; no second table anywhere) ---------------------------------------------------------
    def effective_moe_construction(self) -> str:
        """[route A (3)] The fused-MoE construction the Program is built with: the declared knob, else `indexed-read-ordered`.  A
        Program static, declared-and-bound (`moe_construction_declaration()` is what GP-01 writes; the frontend's MoE rules read it through
        `vllm_bindings.active_target_profile()` and each family applies only under its own construction) -- never attested by a record."""
        return self.moe_construction if self.moe_construction is not None else self.DEFAULT_MOE_CONSTRUCTION

    def effective_fa3_construction(self) -> str:
        """The construction FA3's attention rows are built with: the declared knob, else `check-inf-per-iteration` (`Attention_v4`)."""
        return self.fa3_construction if self.fa3_construction is not None else self.DEFAULT_FA3_CONSTRUCTION
    def effective_weight_only_calls(self) -> str:
        """How often the Program issues a weight-only Call: the declared knob, else `once`."""
        return self.weight_only_calls if self.weight_only_calls is not None else self.DEFAULT_WEIGHT_ONLY_CALLS
    def effective_sampler_construction(self) -> str:
        """The stochastic token select's construction: the declared knob, else `shared-greedy`."""
        return self.sampler_construction if self.sampler_construction is not None else self.DEFAULT_SAMPLER_CONSTRUCTION

    def effective_fp8_scale_construction(self) -> str:
        """The blockwise FP8 GEMM's scale-product construction: the declared knob, else `shared-per-block`."""
        return self.fp8_scale_construction if self.fp8_scale_construction is not None else self.DEFAULT_FP8_SCALE_CONSTRUCTION

    def moe_construction_declaration(self) -> dict:
        """The construction block GP-01 writes into the workload Program header: value, whether it was declared explicitly, the
        rule family and Definitions it binds, and how it is verified (component coupling at GP-01; the digest carries it)."""
        v = self.effective_moe_construction()
        return {"value": v, "declared_explicitly": self.moe_construction is not None, **self.MOE_CONSTRUCTIONS[v],
                "attested": False, "profile_digest": self.digest(),
                "verified_by": "GP-01: every component's derive-time target.declared resolves to the same construction and its Program binds "
                               "that construction's Definitions (refused by name otherwise); the workload digest differs by construction on the "
                               "same trace; GM-01 reports the label and compares nothing against the record (the record is construction-agnostic)"}

    def effective_async_scheduling(self) -> bool:
        """The async-scheduling configuration the workload Program is specialized for: the declared knob, else the pinned vLLM's
        resolution of an unset `SchedulerConfig.async_scheduling` for an admitted profile (`PINNED_ASYNC_SCHEDULING_RESOLUTION`: True).
        A declaration, not an observation: GM-01 G1 compares it with the record header's resolved read-back and FAILS by name on a
        disagreement; the lag is never read from the trace."""
        if self.async_scheduling is not None:
            return bool(self.async_scheduling)
        return bool(self.PINNED_ASYNC_SCHEDULING_RESOLUTION["value_when_unset"])

    def lag(self) -> int:
        """[DP-pad-02 (b)] LAG: the rows the declared scheduler computes for a request AFTER its EOS sample (1 under async scheduling,
        0 synchronous) -- a REQUIRED specialization parameter of the workload Program (`Workload_v1{requests, shared, lag}`)."""
        return 1 if self.effective_async_scheduling() else 0

    def async_scheduling_declaration(self) -> dict:
        """The lag declaration GP-01 writes into the workload Program (`lag` block): value, the resolved configuration it is coupled
        to, its source (declared knob / pinned-resolver mirror) and the implementation version pinned."""
        declared = self.async_scheduling is not None
        return {"value": self.lag(), "async_scheduling": self.effective_async_scheduling(),
                "source": ("target.async_scheduling declared" if declared else
                           f"TargetProfile.effective_async_scheduling(): not declared in the target; the pinned resolver's value for an admitted "
                           f"profile ({self.PINNED_ASYNC_SCHEDULING_RESOLUTION['source']}, vLLM {self.PINNED_ASYNC_SCHEDULING_RESOLUTION['vllm_revision']})"),
                "declared_explicitly": declared, "vllm_revision": self.vllm_revision, "profile_digest": self.digest(),
                "verified_by": "GM-01 G1: RunHeader.profile.engine_kwargs.async_scheduling (the engine's RESOLVED value, vllm_adapter.resolved_async_scheduling) "
                               "must equal the declaration; G6 consumes the Program's lag only"}

    def effective_flash_attn_version(self) -> int:
        """The FA version vLLM d9105ea80 selects for this profile: the declared knob, else the selector's default for the declared
        capability (`fa_utils.get_flash_attn_version` step 1: cc 9.x -> 3 when the wheel ships FA3; cc 10.x -> 4 only when the wheel ships
        FA4, which the pinned build does not (`is_fa_version_supported(4)` is False: read back 2 on a cc-10.0-declared build); else 2).
        A Blackwell target is unregistered here regardless (the GEMM target registry refuses first).  The builder verifies this mirror
        against the instantiated layers (`profile_mismatches`)."""
        if self.flash_attn_version is not None:
            return self.flash_attn_version
        return 3 if self.compute_capability[0] == 9 else 2

    @property
    def tensor_core_family(self) -> str:
        """The registered k16 tensor-core pipeline family of the capability: `ampere` (sm_80/8.6/8.9: `AmpereBF16TcDot16`, one
        registered function serves all — `silicon_total.AMPERE_BF16_DOT16_TOTAL`), `hopper` (sm_90: `HopperBF16WgmmaDot16`;
        Hawkeye's `HOPPER_BF16_M16N8K16` and the wgmma re-measurement share the 16:26:-133 pipeline, so FA2's `mma.sync` and
        the GEMM's `wgmma` on an H100 are one registered step), `blackwell_consumer` (sm_120, cc 12.0 exactly: `mma.sync` only, on
        the same 16:26:-133 pipeline, so `HopperBF16WgmmaDot16` again).  Anything else is `unregistered:<cc>` (rules refuse)."""
        major = self.compute_capability[0]
        if major == 8:
            return "ampere"
        if major == 9:
            return "hopper"
        if self.compute_capability == (12, 0):
            return "blackwell_consumer"
        return f"unregistered:{self.capability_str}"

    @property
    def int8_parts(self) -> tuple[str, ...]:
        return self.INT8_EXACT_SCOPES.get(self.int8_exact, ())

    @property
    def execution_label(self) -> str:
        """The row's `execution` label: `bi-eager` (batch-invariant eager, vLLM's own attention selection) with the suffix `-fa2`
        when FA2 is FORCED on a capability whose selector would pick another version (a separately labelled configuration,
        never a substitute for the native row)."""
        mode = "compiled" if self.compilation is not None else "eager"       # `bi-compiled`: the panel's compiled rows
        base = f"bi-{mode}" if self.batch_invariant else mode
        if self.flash_attn_version is not None and self.flash_attn_version != self.with_overrides(flash_attn_version=None).effective_flash_attn_version():
            return f"{base}-fa{self.flash_attn_version}"
        return base

    def engine_args(self, *, execution: bool = True) -> dict:
        """The vLLM `LLM(**kw)` / `EngineArgs` overrides this profile APPLIES at engine construction (runtime side: Match capture and
        the collector build their engines with exactly these, so the executing kernels are the declared ones).  A
        compiled profile adds `enforce_eager=False` + `compilation_config` (the executed mode; recorded in the run header);
        `execution=False` omits them: the Program DERIVATION (`derive_step`) traces module forwards eagerly by construction and declares
        the compiled execution it models in its applicability block (`modelled_engine()`) instead of executing it."""
        kw: dict = {}
        if self.attention_backend is not None:
            kw["attention_backend"] = self.attention_backend
        if self.flash_attn_version is not None:
            kw["attention_config"] = {"flash_attn_version": int(self.flash_attn_version)}
        if self.moe_backend is not None:
            kw["moe_backend"] = self.moe_backend
        if "linears" in self.int8_parts:
            kw["quantization"] = self.INT8_EXACT_QUANTIZATION
        if "attention" in self.int8_parts:
            kw["attention_backend"] = self.INT8_EXACT_ATTENTION_BACKEND
        if execution and self.compilation is not None:
            c = self.COMPILATIONS[self.compilation]
            kw["enforce_eager"] = False
            kw["compilation_config"] = {"cudagraph_mode": c["cudagraph_mode"],
                                        "inductor_compile_config": {"emulate_precision_casts": bool(c["emulate_precision_casts"])}}
        if execution and self.async_scheduling is not None:
            # an EXPLICIT declaration configures the engine for the declared scheduler (vLLM still resolves and the header attests the
            # read-back: `resolved_async_scheduling`, compared with the Program's declaration by GM-01 G1); None leaves vLLM's own resolution
            kw["async_scheduling"] = bool(self.async_scheduling)
        return kw

    def modelled_engine(self) -> dict:
        """The engine conditions a Program built for this profile DECLARES it models (GP-01 `applicability.engine`, compared by
        GM-01 G1 with the record's effective engine kwargs): eager profiles = `enforce_eager True`; a compiled profile = `enforce_eager False`
        + the CompilationMode / CUDAGraphMode facts, with the claim named: boundary-level correspondence (module outputs, attention, logits,
        sampled ids checked on the committed bytes), kernel interiors Inductor-fused and NOT certified launch-by-launch."""
        if self.compilation is None:
            return {"enforce_eager": True}
        return {"enforce_eager": False, "compilation": {**self.COMPILATIONS[self.compilation], "profile": self.compilation},
                "program_derivation": ("P_compiled: one registered relation per Inductor-generated kernel of the production artifact (compiled_relations.py; "
                                       "GEMM/attention/lm_head existing Definitions; fused norm/act/RoPE regions new Definitions), applicability pinned by the "
                                       "compilation facts + kernel_set digest (DECISIONS-2 §7 S1)" if not self.COMPILATIONS[self.compilation]["emulate_precision_casts"]
                                       else "eager module-forward trace (the emulation performs the eager roundings inside the fused kernels)"),
                "claim": ("P_compiled kernel-level conformance: every non-weight buffer of every kernel launch is committed under graph replay "
                          "(wrapper-level hooks, kernels untouched) and the kernel relations are replayed bit-for-bit from the committed bytes "
                          "(S2: 4016/4016 launches on the B1 row); the eager Program is NOT a model of this execution (P_compiled != P_eager: "
                          "roundings elided, residual recomputed unrounded, other reduction orders; tokens diverge from eager). Attention = eager "
                          "FA2 splitting op; lm_head eager (DECISIONS-2 §7, ruling M-0380)")}

    # -- serialisation ---------------------------------------------------------------------------------------------------
    def to_json(self) -> dict:
        d = asdict(self)
        d["compute_capability"] = list(self.compute_capability)
        if d.get("compilation") is None:      # eager profiles serialise / digest exactly as before the field existed
            d.pop("compilation", None)
        if d.get("fp8_block_gemm") is None:
            d.pop("fp8_block_gemm", None)     # absent = vLLM's selector; every earlier profile digest unchanged
        if d.get("int8_exact") is None:
            d.pop("int8_exact", None)         # absent = no int8 linears; every earlier profile digest unchanged
        if d.get("moe_backend") is None:
            d.pop("moe_backend", None)
        if d.get("async_scheduling") is None:
            d.pop("async_scheduling", None)   # absent = the pinned resolver's value (effective_async_scheduling); every earlier profile digest unchanged
        if d.get("moe_construction") is None:
            d.pop("moe_construction", None)   # absent = DEFAULT_MOE_CONSTRUCTION; every earlier profile digest unchanged
        if d.get("fa3_construction") is None:
            d.pop("fa3_construction", None)   # absent = DEFAULT_FA3_CONSTRUCTION; every earlier profile digest unchanged
        if d.get("weight_only_calls") is None:
            d.pop("weight_only_calls", None)  # absent = DEFAULT_WEIGHT_ONLY_CALLS; every earlier profile digest unchanged
        if d.get("sampler_construction") is None:
            d.pop("sampler_construction", None)
        if d.get("fp8_scale_construction") is None:
            d.pop("fp8_scale_construction", None)  # absent = shared-per-block; every earlier profile digest unchanged
        if d.get("protocols") is None:
            d.pop("protocols", None)               # absent = sampled proofs alone; every earlier profile digest unchanged
        else:
            d["protocols"] = list(d["protocols"])
        if d.get("ir") is None:
            d.pop("ir", None)                      # absent = word; every earlier profile digest unchanged
        if d.get("registered_values") is None:
            d.pop("registered_values", None)       # absent = constants; every earlier profile digest unchanged
        return d

    def fp8_block_gemm_kernel(self) -> str | None:
        """The vLLM kernel class the declared `fp8_block_gemm` must have selected (None = not declared: whatever vLLM takes)."""
        return None if self.fp8_block_gemm is None else self.FP8_BLOCK_GEMM_KERNELS[self.fp8_block_gemm]

    def moe_experts_class(self) -> str | None:
        """The experts class the declared `moe_backend` must have selected (None = not declared: whatever vLLM takes)."""
        return None if self.moe_backend is None else self.MOE_BACKEND_KERNELS[self.moe_backend]

    @classmethod
    def from_json(cls, d: dict) -> "TargetProfile":
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown TargetProfile fields: {sorted(unknown)}")
        return cls(**d)

    def canonical(self) -> bytes:
        return canonical_bytes(self.to_json())

    def digest(self) -> str:
        return sha256_hex(self.canonical())

    def with_overrides(self, **kw: Any) -> "TargetProfile":
        return replace(self, **kw)

    # -- derived GEMM target facts (one source: this profile's capability -> verity_vllm.program.frontend.gemm_targets) ----------------
    def tuned_gemm_arch_family(self) -> str | None:
        """vLLM's tuned-GEMM arch family for the declared capability (`gemm_targets.vllm_tuned_gemm_arch_family`: vLLM's own function
        when importable, else its pinned rule). Selects vLLM's tuned table — and therefore the launch geometry — on the target."""
        from verity_vllm.program.frontend.gemm_targets import vllm_tuned_gemm_arch_family  # lazy: keeps this module light
        return vllm_tuned_gemm_arch_family(self.compute_capability)

    def gemm_target(self, arch_family: Any = ...) -> GemmTarget:
        """The registered GEMM specialization for this target (`gemm_targets.GemmTarget`: the launch geometry source, the Definition's k16
        step, the correspondence evidence) — what Bet A pins, the eager resolver pins and the applicability manifest requires.  `arch_family`
        = the family the host actually reported (construction read-back / capture header), when known: it must be the record's.  Raises
        `gemm_targets.UnregisteredGemmTarget` (fail closed) for a target without a registered specialization."""
        from verity_vllm.program.frontend.gemm_targets import gemm_target_for
        return gemm_target_for(self.compute_capability, arch_family)

    @classmethod
    def default(cls) -> "TargetProfile":
        """The admitted record's target (profile `vllm_d9105ea80_sm89_eager`: sm_89 / 128 SMs / batch-invariant / FA2 by selector)."""
        return cls()

    @classmethod
    def from_device(cls, device: int = 0) -> "TargetProfile":
        """[X-01] The target = THIS host's GPU, queried ONCE through `torch.cuda.get_device_properties(device)`: `multi_processor_count`
        is `num_sms` (vLLM's `num_compute_units()`, the persistent-GEMM grid) and `(major, minor)` the compute capability.  The one
        source of the SM count for a construction on a GPU host: the builder applies it (shim), pins it (GEMM rule), and the artifact
        identity carries it (`ArtifactIdentity.target_profile.num_sms`).  Torch is imported lazily: this module stays torch-free."""
        import torch  # lazy: the module is torch-free on CPU hosts
        p = torch.cuda.get_device_properties(device)
        return cls.default().with_overrides(compute_capability=(int(p.major), int(p.minor)), num_sms=int(p.multi_processor_count))


    @classmethod
    def from_gpu_document(cls, doc: Any) -> "TargetProfile | None":
        """[X-01] The SAME device query as a capture recorded it (`RunHeader.profile["gpu"]` from `verity_vllm.engine.engine_profile.gpu_info`,
        or `versions.gpu` of a bench run): `multi_processor_count` (or `sms`) and `compute_capability` (`[8, 9]` or `"8.9"`, or `cc`).
        The eager Match resolver reads the target from here — the record's own host, not a literal.  None when the document carries no
        SM count (synthetic / hand-built logs): callers then keep the default profile."""
        if not isinstance(doc, dict):
            return None
        sms = doc.get("multi_processor_count", doc.get("sms"))
        if sms is None:
            return None
        cc = doc.get("compute_capability", doc.get("cc"))
        over: dict = {"num_sms": int(sms)}
        try:
            if isinstance(cc, str) and "." in cc:
                maj, mi = cc.split(".")[:2]
                over["compute_capability"] = (int(maj), int(mi))
            elif isinstance(cc, (list, tuple)) and len(cc) == 2:
                over["compute_capability"] = (int(cc[0]), int(cc[1]))
        except (TypeError, ValueError):
            pass
        return cls.default().with_overrides(**over)

    @classmethod
    def from_run_profile(cls, profile_doc: Any) -> "TargetProfile | None":
        """The target of a captured record, from its run header's profile document: the DECLARED profile the engine was
        built with when the capture recorded one (`profile["target"]`, written by `vllm_adapter.build_engine(target=...)` — exact,
        including `flash_attn_version`), else the X-01 device document (`from_gpu_document`) with the FA version the runner's attention
        layers reported (`profile["backends"]["attention_flash_attn_version"]`) when it is NOT the selector's default for that
        capability — i.e. a forced version is never silently read as the default one.  None when neither is present."""
        if not isinstance(profile_doc, dict):
            return None
        t = profile_doc.get("target")
        if isinstance(t, dict) and t:
            try:
                return cls.from_json(t)
            except (TypeError, ValueError):
                pass
        tp = cls.from_gpu_document(profile_doc.get("gpu"))
        if tp is None:
            return None
        fav = ((profile_doc.get("backends") or {}).get("attention_flash_attn_version"))
        if fav is not None and int(fav) != tp.effective_flash_attn_version():
            tp = tp.with_overrides(flash_attn_version=int(fav))
        return tp


    @property
    def capability_str(self) -> str:
        return f"{self.compute_capability[0]}.{self.compute_capability[1]}"

    @property
    def sm_arch(self) -> str:
        return f"sm_{self.compute_capability[0]}{self.compute_capability[1]}"

    def env(self) -> dict[str, str]:
        """The environment variables this profile APPLIES (only variables the pinned vLLM's `envs` module defines)."""
        env = {"VLLM_BATCH_INVARIANT": "1" if self.batch_invariant else "0"}
        if self.fp8_block_gemm == "cutlass":
            env["VLLM_USE_DEEP_GEMM"] = "0"   # excludes FlashInfer-DeepGEMM + DeepGEMM; the selector then takes CUTLASS (cc 9.x)
        return env


DEFAULT_TARGET = TargetProfile.default()


# -- declared vs applied ---------------------------------------------------------------------------------------------------


def profile_mismatches(profile: TargetProfile, applied: dict) -> list[str]:
    """Fields of `applied` (the read-back record) that disagree with the declared profile. Absent = mismatch (never assumed).
    `applied` keys: compute_capability [maj, min], num_sms, batch_invariant, vllm_version (contains `+g<rev>`), attention_backend."""
    bad: list[str] = []
    cc = applied.get("compute_capability")
    if cc is None or tuple(int(x) for x in cc) != profile.compute_capability:
        bad.append(f"compute_capability declared {list(profile.compute_capability)}, applied {cc!r}")
    if applied.get("num_sms") != profile.num_sms:
        bad.append(f"num_sms declared {profile.num_sms}, applied {applied.get('num_sms')!r}")
    if applied.get("batch_invariant") is None or bool(applied.get("batch_invariant")) != profile.batch_invariant:
        bad.append(f"batch_invariant declared {profile.batch_invariant}, applied {applied.get('batch_invariant')!r}")
    vv = applied.get("vllm_version")
    if not isinstance(vv, str) or f"+g{profile.vllm_revision}" not in vv:
        bad.append(f"vllm_revision declared {profile.vllm_revision!r}, running vllm {vv!r}")
    ab, want_ab = applied.get("attention_backend", "<absent>"), profile.engine_args(execution=False).get("attention_backend")
    if ab == "<absent>" or (ab or None) != (want_ab or None):
        bad.append(f"attention_backend declared {want_ab!r}, applied {ab!r}")
    # the tuned-GEMM arch family vLLM applied for the declared capability must be the one the registered GEMM target
    # (verity_vllm.program.frontend.gemm_targets) requires: another answer means another launch geometry than the Definition was admitted for.
    fam = applied.get("tuned_gemm_arch_family", "<absent>")
    if fam == "<absent>" or fam != profile.tuned_gemm_arch_family():
        bad.append(f"tuned_gemm_arch_family for cc {profile.capability_str} is {profile.tuned_gemm_arch_family()!r} (vLLM {profile.vllm_revision}), applied {fam!r}")
    # the FA version the instantiated attention layers carry (`vllm_flash_attn_version`, one value over all layers) must be
    # the profile's effective one; a record without the read-back (older drivers) is not judged here (the attention rule pins it per layer)
    if "flash_attn_version" in applied:
        fav = applied.get("flash_attn_version")
        want = profile.effective_flash_attn_version()
        ok = isinstance(fav, int) and not isinstance(fav, bool) and fav == want     # a non-integer read-back ('<absent>', None) is a mismatch, never a crash
        if not ok:
            bad.append(f"flash_attn_version declared {profile.flash_attn_version!r} (effective {want}), applied {fav!r}")
    # the block-scaled FP8 GEMM kernel vLLM selected for the model's FP8 linears (`applied["fp8_block_gemm_kernel"]`: the one
    # `Fp8LinearMethod.fp8_linear` class over all layers, None = no FP8 linear layer, '<mixed>' = several) must be the declared one; a
    # declared kernel on a model with no FP8 linears is a mismatch (an idle declaration is not a fact of the run).  Undeclared (None):
    # not judged here -- the FP8 rules refuse an unregistered kernel's op by name at derive time.
    if profile.fp8_block_gemm is not None:
        got = applied.get("fp8_block_gemm_kernel", "<absent>")
        if got != profile.fp8_block_gemm_kernel():
            bad.append(f"fp8_block_gemm declared {profile.fp8_block_gemm!r} ({profile.fp8_block_gemm_kernel()}), applied {got!r}")
    if profile.int8_exact is not None and applied.get("int8_exact", "<absent>") != profile.int8_exact:
        bad.append(f"int8_exact declared {profile.int8_exact!r}, applied {applied.get('int8_exact', '<absent>')!r}")
    # a declared MoE backend on a model with no fused-MoE layer is a mismatch too (`moe_experts_class_of` is None there)
    if profile.moe_backend is not None:
        got = applied.get("moe_experts_class", "<absent>")
        if got != profile.moe_experts_class():
            bad.append(f"moe_backend declared {profile.moe_backend!r} ({profile.moe_experts_class()}), applied {got!r}")
    return bad



# -- the artifact wrapper --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ArtifactIdentity:
    construction_version: dict
    target_profile: dict
    applicability: dict
    program_digest: str
    identity: str = field(default="")

    @staticmethod
    def compute(construction_version: dict, target_profile: TargetProfile | dict, applicability: dict, program_digest: str) -> str:
        tp = target_profile.to_json() if isinstance(target_profile, TargetProfile) else target_profile
        h = hashlib.sha256()
        for part in (canonical_bytes(construction_version), canonical_bytes(tp), canonical_bytes(applicability), program_digest.encode()):
            h.update(len(part).to_bytes(8, "big"))  # length-prefixed: no component can absorb its neighbour
            h.update(part)
        return h.hexdigest()

    @classmethod
    def build(cls, construction_version: dict, target_profile: TargetProfile, applicability: dict, program_digest: str) -> "ArtifactIdentity":
        tp = target_profile.to_json()
        return cls(construction_version=construction_version, target_profile=tp, applicability=applicability, program_digest=program_digest,
                   identity=cls.compute(construction_version, tp, applicability, program_digest))

    def to_json(self) -> dict:
        return {"schema": SCHEMA, "identity": self.identity, "construction_version": self.construction_version, "target_profile": self.target_profile,
                "applicability": self.applicability, "program_digest": self.program_digest,
                "identity_definition": "sha256(len8||canonical(construction_version) || len8||canonical(target_profile) || len8||canonical(applicability) || len8||program_digest)"}

    @classmethod
    def from_json(cls, d: dict) -> "ArtifactIdentity":
        if d.get("schema") != SCHEMA:
            raise ValueError(f"artifact schema {d.get('schema')!r} is not {SCHEMA!r}")
        return cls(construction_version=d["construction_version"], target_profile=d["target_profile"], applicability=d["applicability"],
                   program_digest=d["program_digest"], identity=d["identity"])

    def recomputed(self) -> str:
        return self.compute(self.construction_version, self.target_profile, self.applicability, self.program_digest)


class ApplicabilityError(RuntimeError):
    """An artifact was requested outside its declared applicability; `.reasons` lists every differing field verbatim."""

    def __init__(self, reasons: list[str]) -> None:
        self.reasons = list(reasons)
        super().__init__("artifact not applicable: " + "; ".join(self.reasons))


def reuse_mismatches(artifact: ArtifactIdentity, profile: TargetProfile, applicability_now: dict | None, construction_version_now: dict | None = None) -> list[str]:
    """Every reason `artifact` may NOT be reused by a requester declaring `profile` under `applicability_now` (the facts read back from
    the requester's environment; None = not provided = every condition unverifiable = refused)."""
    bad: list[str] = []
    if artifact.recomputed() != artifact.identity:
        bad.append(f"identity {artifact.identity} does not match its components (recomputed {artifact.recomputed()})")
    want, have = profile.to_json(), dict(artifact.target_profile)
    for k in sorted(set(want) | set(have)):
        if want.get(k, "<absent>") != have.get(k, "<absent>"):
            bad.append(f"target_profile.{k}: requested {want.get(k, '<absent>')!r}, artifact built for {have.get(k, '<absent>')!r}")
    if applicability_now is None:
        bad.append("applicability conditions of the requester were not provided (every condition is unverifiable)")
    else:
        for k in sorted(artifact.applicability):
            if k not in applicability_now:
                bad.append(f"applicability.{k}: artifact requires {artifact.applicability[k]!r}, requester did not report it")
            elif applicability_now[k] != artifact.applicability[k]:
                bad.append(f"applicability.{k}: artifact requires {artifact.applicability[k]!r}, requester has {applicability_now[k]!r}")
    if construction_version_now is not None and canonical_bytes(construction_version_now) != canonical_bytes(artifact.construction_version):
        bad.append("construction_version differs from the artifact's")
    return bad


def check_reuse(artifact: ArtifactIdentity, profile: TargetProfile, applicability_now: dict | None, construction_version_now: dict | None = None) -> ArtifactIdentity:
    """Return `artifact` if it is applicable for the requester, else raise `ApplicabilityError` (fail-closed)."""
    bad = reuse_mismatches(artifact, profile, applicability_now, construction_version_now)
    if bad:
        raise ApplicabilityError(bad)
    return artifact


def load_artifact(path: str, profile: TargetProfile, applicability_now: dict | None, construction_version_now: dict | None = None) -> ArtifactIdentity:
    """Read `artifact.json` and admit it only under `profile` / `applicability_now` (`check_reuse`)."""
    with open(path) as f:
        art = ArtifactIdentity.from_json(json.load(f))
    return check_reuse(art, profile, applicability_now, construction_version_now)


def parse_target_arg(s: str | None, *, base: "TargetProfile | None" = None) -> TargetProfile:
    """CLI helper: None -> default; a JSON object -> overrides laid over `base` (`{"compute_capability": [9, 0]}`); a path to a JSON
    file -> the whole profile.  `base` defaults to THIS host's device profile when CUDA is available, else `default()` -- the same
    layering `vllm_adapter.target_of_workload` gives the workload's `target` block, so a row that declares only a knob
    (`{"fp8_block_gemm": "cutlass"}`, `{"flash_attn_version": 2}`) is built for the pod it runs on.  rc3 laid the overrides over
    `default()` (capability 8.9 / 128 SMs): on cp-h100 the FP8 row's Build declared an L40S and vLLM selected the sm89 kernel
    ('HummingFP8ScaledMMLinearKernel'), refused as a target mismatch against the declared cutlass kernel."""
    if not s:
        return TargetProfile.default()
    s = s.strip()
    if s.startswith("{"):
        d = json.loads(s)
    else:
        with open(s) as f:
            d = json.load(f)
    if base is None:
        try:
            import torch
            base = TargetProfile.from_device() if torch.cuda.is_available() else TargetProfile.default()
        except Exception:                                   # no torch on this host: the documented default
            base = TargetProfile.default()
    base = base.to_json()
    unknown = set(d) - set(TargetProfile.__dataclass_fields__)   # the fields, not the serialised keys (`compilation` / `fp8_block_gemm` are omitted when None)
    if unknown:
        raise ValueError(f"unknown TargetProfile fields: {sorted(unknown)}")
    base.update(d)
    return TargetProfile.from_json(base)


__all__ = ["TargetProfile", "DEFAULT_TARGET", "ArtifactIdentity", "ApplicabilityError", "canonical_bytes", "sha256_hex", "profile_mismatches",
           "reuse_mismatches", "check_reuse", "load_artifact", "parse_target_arg", "SCHEMA"]
