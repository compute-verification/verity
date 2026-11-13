"""The closed list of vLLM's modules and names that the integration's library code imports, at the pin
(`verity_vllm.config.VLLM`).

`SURFACE` maps each to why the integration needs it: ``"vllm.mod"`` for a module imported whole (``import vllm.mod``) and
``"vllm.mod:name"`` for a name imported from one (``from vllm.mod import name``). `test_vllm_surface.py` refuses an
import under ``verity_vllm/`` that it doesn't list, and warns on an entry that nothing imports. Where vLLM is installed,
it resolves every entry against it, so moving the pin starts from this list.
"""
from __future__ import annotations

SURFACE = {
    # the engine and its requests
    "vllm": "the package, for its version, which records and facts carry beside config.VLLM",
    "vllm:LLM": "the engine engine.build builds, as the tensor-parallel capture, commit and match drivers do",
    "vllm:EngineArgs": "the arguments engine.stock builds a stock in-process LLMEngine from",
    "vllm:SamplingParams": "a request's sampling parameters, which the adapter, the servers and served_match pass",
    "vllm.sampling_params:RequestOutputKind": "DELTA outputs, which the arbitrary-request committer streams",
    "vllm.engine.arg_utils:EngineArgs": "vLLM's own engine config, which the meta-device frontend builds with no engine",
    "vllm.entrypoints.openai.completion:protocol": "the OpenAI completion request and stream types serve_protocol speaks",
    "vllm.v1.engine.llm_engine:LLMEngine": "the in-process engine, which engine.handle walks to its scheduler and runner",
    "vllm.v1.engine.core:EngineCore": "the engine core, whose two steps engine.core_gate wraps for its edges",
    "vllm.v1.engine.core:EngineCoreProc": "the engine core process, whose input loop and has_work a gate holds",
    "vllm.v1.core.sched.scheduler:Scheduler": "the V1 scheduler, which engine.handle returns and the tick's gate and PoUB's "
                                              "binder hook",
    "vllm.v1.core.block_pool:BlockPool": "the KV block pool, whose allocations PoUB's binder records",
    "vllm.v1.worker.gpu_model_runner:GPUModelRunner": "the model runner, which engine.handle returns",
    "vllm.v1.executor.multiproc_executor:WorkerProc": "the tensor-parallel worker process, which the TP lease hooks",
    # configuration and the platform
    "vllm.envs": "vLLM's environment switches, which engine.build, the inventory and the served facts read back",
    "vllm:envs": "vLLM's environment switches, which the MoE and RMSNorm sources read",
    "vllm.config:get_current_vllm_config": "the model config, which the int8 attention backend reads",
    "vllm.config:set_current_vllm_config": "the config the meta-device frontend builds its model under",
    "vllm.platforms:current_platform": "the platform, whose device capability the frontend's profile pins and facts record",
    "vllm.platforms.interface:DeviceCapability": "a compute capability, to ask vLLM's tuned GEMM arch family for a device",
    "vllm.utils:deep_gemm": "vLLM's DeepGEMM: its verdict, which the build inventory records",
    "vllm.utils:flashinfer": "vLLM's FlashInfer verdict, which the build inventory records",
    "vllm.utils.flashinfer:has_flashinfer": "vLLM's FlashInfer verdict for this process, which engine.build records",
    "vllm.utils.platform_utils:num_compute_units": "the SM count that fixes the top-p split's launch geometry",
    "vllm.utils.torch_utils:is_quantized_kv_cache": "whether the KV cache is quantized, which the int8 attention backend "
                                                    "refuses",
    "vllm.utils.torch_utils:set_default_torch_dtype": "the dtype a meta-device model is built under",
    "vllm.ir": "vLLM's IR ops, which the frontend makes torch-wrapped so that its export sees through them",
    # distributed
    "vllm.distributed:ensure_model_parallel_initialized": "a one-rank model-parallel group for the meta-device frontend",
    "vllm.distributed:init_distributed_environment": "a one-rank process group for the meta-device frontend",
    "vllm.distributed:get_tensor_model_parallel_rank": "the TP rank, which served facts, the inventory and the TP target "
                                                       "key on",
    "vllm.distributed:get_tensor_model_parallel_world_size": "the TP size, which the int8 attention backend shards by",
    "vllm.distributed:parallel_state": "the TP group, whose collectives the rank worker, the profile and the export hook",
    "vllm.distributed:tensor_model_parallel_all_reduce": "the all-reduce the export's TP ops stand in for",
    "vllm.distributed.parallel_state:get_pp_group": "a PoUW H1 rank's pipeline-parallel rank",
    "vllm.distributed.parallel_state:get_tp_group": "the TP group: its device communicator, and a PoUW H1 rank's TP rank",
    "vllm.forward_context:set_forward_context": "a forward context with no attention metadata, for the frontend's export",
    # compilation and CUDA graphs
    "vllm.compilation:cuda_graph": "CUDAGraphWrapper, whose replays the served compiled source hooks",
    "vllm.compilation:piecewise_backend": "PiecewiseBackend, whose pieces the served compiled source hooks",
    "vllm.v1.worker:workspace": "the engine's persistent MoE workspace, which export_compat resets",
    "vllm.v1.worker.gpu:cudagraph_utils": "CudaGraphManager, whose captures the served compiled source hooks",
    "vllm.v1.worker.gpu:model_runner": "the V2 model runner, whose CUDA graph dispatch the served compiled source hooks",
    # determinism
    "vllm.model_executor.determinism:batch_invariant": "batch-invariant mode, which replay facts, the RMSNorm source and "
                                                       "the frontend read",
    "vllm.model_executor.determinism.batch_invariant:bmm_batch_invariant": "the batch-invariant bmm the MLA capture calls",
    "vllm.model_executor.determinism.batch_invariant_configs:_get_tuned_matmul_arch_family":
        "vLLM's tuned GEMM arch family for a device, which the GEMM targets and the profile record (private to vLLM)",
    # layers
    "vllm.model_executor.layers.attention:Attention": "the attention layer, whose implementation the frontend observes",
    "vllm.model_executor.layers.attention.attention:Attention": "the attention layer, which served_match finds by type",
    "vllm.model_executor.layers.attention:MLAAttention": "the MLA layer, whose weight split loader parity and the "
                                                         "frontend read",
    "vllm.model_executor.layers.attention:attention": "the attention module, whose forward the served compiled source hooks",
    "vllm.model_executor.layers.fused_moe:FusedMoEMethodBase": "the base of the int8 experts method",
    "vllm.model_executor.layers.fused_moe:FusedMoeWeightScaleSupported": "the per-channel kind of the int8 experts' scales",
    "vllm.model_executor.layers.fused_moe:RoutedExperts": "the routed experts layer the int8 recipe quantizes",
    "vllm.model_executor.layers.fused_moe:fused_moe": "the fused MoE module, whose kernels the MoE sources and the export "
                                                      "hook",
    "vllm.model_executor.layers.fused_moe.fused_moe": "the fused MoE module, whose dispatch the MoE difftest calls",
    "vllm.model_executor.layers.fused_moe.activation:MoEActivation": "the activation served_match checks a MoE layer for",
    "vllm.model_executor.layers.fused_moe.experts:triton_moe": "the Triton experts module, whose kernels the served "
                                                               "compiled source hooks",
    "vllm.model_executor.layers.fused_moe.experts.triton_moe:TritonExperts": "the Triton experts served_match checks a MoE "
                                                                             "layer for",
    "vllm.model_executor.layers.fused_moe.router.fused_topk_router:FusedTopKRouter": "the softmax router served_match "
                                                                                     "checks a MoE layer for",
    "vllm.model_executor.layers.fused_moe.runner.moe_runner:MoERunner": "the MoE runner, which served_match finds and the "
                                                                        "served compiled source hooks",
    "vllm.model_executor.layers.linear": "the linear layers' module, whose collectives the rank worker hooks",
    "vllm.model_executor.layers.linear:LinearBase": "every linear layer, which PoUW, PoUS and the int8 recipe find by type",
    "vllm.model_executor.layers.linear:LinearMethodBase": "the base of the int8 linear method",
    "vllm.model_executor.layers.linear:UnquantizedLinearMethod": "the bf16 method of the layers the int8 recipe keeps",
    "vllm.model_executor.layers.linear:register_weight_loader_v2_supported_method": "registers the int8 linear method's "
                                                                                    "weight loader",
    "vllm.model_executor.layers.logits_processor": "the logits processor's module, whose gathers the rank worker hooks",
    "vllm.model_executor.layers.logits_processor:LogitsProcessor": "the logits processor, whose logits the served compiled "
                                                                   "source records",
    "vllm.model_executor.layers.quantization:register_quantization_config": "registers the int8 quantization config",
    "vllm.model_executor.layers.quantization.base_config:QuantizationConfig": "the base of the int8 quantization config",
    "vllm.model_executor.layers.quantization.utils.quant_utils:is_layer_skipped": "the layers the int8 recipe leaves bf16",
    "vllm.model_executor.layers.vocab_parallel_embedding": "the embedding's module, whose all-reduce the rank worker hooks",
    "vllm.model_executor.layers.vocab_parallel_embedding:ParallelLMHead": "the LM head, which the int8 recipe keeps bf16",
    "vllm.model_executor.layers.vocab_parallel_embedding:VocabParallelEmbedding": "the embedding the export's TP ops bind",
    "vllm.model_executor.kernels.linear.scaled_mm.flashinfer:FlashInferFp8DeepGEMMDynamicBlockScaledKernel":
        "the FP8 block-scaled linear kernel export_compat makes the default",
    "vllm.model_executor.model_loader.utils:get_model_architecture": "a config's model class, for the inventory and the "
                                                                     "frontend",
    "vllm.model_executor.model_loader.utils:initialize_model": "a meta-device model, for the build inventory",
    "vllm.model_executor.model_loader.weight_tying:_get_untied_lm_head": "an untied LM head, which loader parity checks "
                                                                         "(private to vLLM)",
    "vllm.model_executor.parameter:ChannelQuantScaleParameter": "the int8 linear method's per-channel scales",
    "vllm.model_executor.parameter:ModelWeightParameter": "the int8 linear method's weights",
    "vllm.model_executor.utils:set_weight_attrs": "the int8 experts' weight loaders",
    # attention backends and kernels
    "vllm.v1.attention.backend:AttentionType": "the decoder attention type the int8 backend requires",
    "vllm.v1.attention.backends.flash_attn:FlashAttentionBackend": "the base of the int8 attention backend",
    "vllm.v1.attention.backends.flash_attn:FlashAttentionImpl": "the base of the int8 attention implementation",
    "vllm.v1.attention.backends.registry:AttentionBackendEnum": "the backends' names: CUSTOM for the int8 backend, and the "
                                                                "frontend's profile knob",
    "vllm.v1.attention.backends.registry:register_backend": "registers the int8 attention backend as CUSTOM",
    "vllm.v1.attention.ops.triton_decode_attention:decode_attention_fwd": "the Triton MLA decode the MLA capture calls",
    "vllm.v1.kv_cache_layout:KVCacheLayout": "the KV layout the int8 attention backend declares",
    "vllm._custom_ops": "vLLM's custom ops: the MoE router and sum the MoE sources hook, and the silu-mul difftest",
    "vllm:_custom_ops": "vLLM's custom ops: the rotary embedding the RoPE difftests call",
    # sampling
    "vllm.v1.sample.ops:topk_topp_triton": "the Triton top-p kernels, whose split statistics the top-p probe reads",
    "vllm.v1.sample.ops.topk_topp_triton": "the Triton top-p module, which topp_pin pins and its check installs",
    "vllm.v1.sample.ops.topk_topp_sampler:apply_top_k_top_p": "vLLM's top-k and top-p, which the top-p adapter calls",
    "vllm.v1.worker.gpu.sample.gumbel:apply_temperature": "the temperature kernel the top-p adapter and probe call",
    "vllm.v1.worker.gpu.sample.gumbel:gumbel_sample": "the Gumbel sampling kernel the top-p adapter calls",
}
