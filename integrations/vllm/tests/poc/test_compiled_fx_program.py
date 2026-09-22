"""CPU tests (no torch) for the W1d compiled-graph tooling that works on Inductor's generated text:

* ``compiled_fx.kernels``: kernel-source normalisation / content hashing and the production-vs-replay
  comparison (identical kernels vs a kernel whose body changed);
* ``compiled_fx.program``: parsing a generated wrapper (``call``) into launches, recorded buffers,
  symbolic size expressions in the row count ``M`` (the ``get_args``/benchmark tail must not leak
  static sizes into the expressions).
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

pytest.importorskip("torch")  # the module under test imports torch._inductor

from verity_vllm_adapter.compiled_fx import kernels as K
from verity_vllm_adapter.compiled_fx import program as P
from verity_vllm_adapter.compiled_fx.fx_pass import ID_STRIDE

KERNEL_A = """
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import persistent_reduction

@persistent_reduction(size_hints={'x': 8192, 'r0_': 1024}, meta={'kernel_name': 'triton_per_fused_add_rms_norm_%d', 'backend_hash': '%s'})
@triton.jit
def triton_per_fused_add_rms_norm_%d(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    tmp0 = tl.load(in_ptr0 + (r0_1 + 576*x0), rmask & xmask, other=0.0)
    tl.store(out_ptr0 + (r0_1 + 576*x0), tmp7, rmask & xmask)
"""


def wrapper(piece: int, kernel_idx: int, backend: str, body_variant: str = "") -> str:
    k = KERNEL_A % (kernel_idx, backend, kernel_idx)
    if body_variant:
        k = k.replace("tmp7", body_variant)
    id0 = piece * ID_STRIDE
    txt = textwrap.dedent(
        f"""
        async_compile = AsyncCompile()

        triton_per_fused_add_rms_norm_{kernel_idx} = async_compile.triton('triton_per_fused_add_rms_norm_{kernel_idx}', '''__KERNEL_A__''', device_str='cuda')

        matmul_kernel_persistent_0 = async_compile.triton('matmul_kernel_persistent_0', '''
        @triton.jit
        def matmul_kernel_persistent_0(a_ptr, b_ptr, c_ptr, bias_ptr, M, N, K):
            pass
        ''', device_str='cuda')

        class Runner:
            def call(self, args):
                arg0_1, arg1_1, arg2_1, arg3_1, arg4_1 = args
                args.clear()
                s72 = arg1_1
                s80 = s72
                assert_size_stride(arg0_1, (s72, 576), (576, 1), 'input')
                with torch.cuda._DeviceGuard(0):
                    torch.cuda.set_device(0)
                    buf2 = empty_strided_cuda((s72, 576), (576, 1), torch.bfloat16)
                    matmul_kernel_persistent_0.run(arg0_1, reinterpret_tensor(arg2_1, (576, 576), (1, 576), 0), buf2, -1, s72, 576, 576, 576, 1, 1, 576, 576, 1, min(108, 8*((127 + s72) // 128)), 1, 1, stream=raw_stream0)
                    torch.ops.verity.record.default(buf2, {id0 + 2})
                    buf6 = empty_strided_cuda((s72, 576), (576, 1), torch.bfloat16)
                    triton_per_fused_add_rms_norm_{kernel_idx}_xnumel = 576*s80
                    triton_per_fused_add_rms_norm_{kernel_idx}.run(buf2, arg3_1, arg4_1, buf6, triton_per_fused_add_rms_norm_{kernel_idx}_xnumel, 576, stream=raw_stream0)
                    del arg3_1
                    torch.ops.verity.record.default(buf6, {id0 + 3})
                return (buf6, )

        runner = Runner(partitions=[])
        call = runner.call


        def get_args():
            arg0_1 = rand_strided((8192, 576), (576, 1), device='cuda:0', dtype=torch.bfloat16)
            arg1_1 = 8192
            return [arg0_1, arg1_1]


        def benchmark_compiled_module(args, times=10, repeat=10):
            return 0
        """
    )
    return txt.replace("__KERNEL_A__", k)


def write_cache(root: Path, name: str, txt: str) -> Path:
    d = root / name / "ab"
    d.mkdir(parents=True)
    p = d / "cwrapper.py"
    p.write_text(txt)
    return p


def test_kernel_normalisation_ignores_names_and_backend_hash():
    a = KERNEL_A % (0, "hashA", 0)
    b = KERNEL_A % (7, "hashB", 7)
    assert K.normalise_kernel(a) == K.normalise_kernel(b)
    assert K.normalise_kernel(a) != K.normalise_kernel(a.replace("tmp7", "tmp8"))


def test_compare_identical_and_changed(tmp_path: Path):
    write_cache(tmp_path, "prod", wrapper(1, 0, "h1").replace("torch.ops.verity.record.default(buf2, 10002)\n", ""))
    write_cache(tmp_path, "replay_same", wrapper(1, 3, "h2"))
    write_cache(tmp_path, "replay_changed", wrapper(1, 3, "h2", body_variant="tmp7 + 0.0"))
    same = K.compare(tmp_path / "prod", tmp_path / "replay_same")
    assert same["kernels_identical"] is True
    assert same["production"]["distinct_kernels"] == 2 == same["replay"]["distinct_kernels"]
    assert same["replay_record_ops_in_wrappers"] == 2 and same["production_record_ops_in_wrappers"] == 1
    changed = K.compare(tmp_path / "prod", tmp_path / "replay_changed")
    assert changed["kernels_identical"] is False
    assert len(changed["only_in_production"]) == 1 and len(changed["only_in_replay"]) == 1
    assert changed["only_in_replay"][0]["kernel_families"] == ["triton_per_fused_add_rms_norm"]


def test_parse_program_launches_records_and_symbolic_sizes(tmp_path: Path):
    p = write_cache(tmp_path, "replay", wrapper(2, 0, "h"))
    prog = P.parse_program(p)
    assert prog is not None
    assert prog.piece == 2
    assert prog.m_symbol == "s72"
    assert [L.kernel_var for L in prog.launches] == ["matmul_kernel_persistent_0", "triton_per_fused_add_rms_norm_0"]
    gemm, norm = prog.launches
    assert gemm.outputs == ["buf2"] and gemm.inputs == ["arg0_1", "arg2_1"]
    assert norm.outputs == ["buf6"] and norm.inputs == ["buf2", "arg3_1", "arg4_1"]
    assert prog.record_buffer == {2 * ID_STRIDE + 2: "buf2", 2 * ID_STRIDE + 3: "buf6"}
    assert prog.returned == ["buf6"]
    # the static sizes of get_args() (8192) must not leak into the size expressions
    x = P._resolve("triton_per_fused_add_rms_norm_0_xnumel", prog.syms).replace(prog.m_symbol, "M")
    assert "8192" not in x and x.replace(" ", "") in ("(576*(M))", "(576*M)")
    assert P._resolve("s72", prog.syms) == "s72"
    assert P._resolve("min(108, 8*((127 + s72) // 128))", prog.syms) == "min(108, 8*((127 + s72) // 128))"
    assert prog.kernel_hash["triton_per_fused_add_rms_norm_0"] == K._sha(K.normalise_kernel(prog.kernel_src["triton_per_fused_add_rms_norm_0"]))


def test_load_programs_dedupes_by_piece(tmp_path: Path):
    write_cache(tmp_path, "r1", wrapper(1, 0, "h"))
    (tmp_path / "r1" / "cd").mkdir()
    (tmp_path / "r1" / "cd" / "other.py").write_text(wrapper(4, 1, "h"))
    progs = P.load_programs(tmp_path / "r1")
    assert sorted(progs) == [1, 4]
    assert progs[4].record_buffer == {4 * ID_STRIDE + 2: "buf2", 4 * ID_STRIDE + 3: "buf6"}


def test_split_args_and_base_token():
    args = P.split_args("buf2, reinterpret_tensor(arg2_1, (576, 576), (1, 576), 0), min(108, 8*((127 + s72) // 128)), 1")
    assert args == ["buf2", "reinterpret_tensor(arg2_1, (576, 576), (1, 576), 0)", "min(108, 8*((127 + s72) // 128))", "1"]
    assert P._base_token(args[1]) == "arg2_1"
    assert P._base_token("buf2") == "buf2"
    assert P._base_token("1") is None and P._base_token("min(108, 8*((127 + s72) // 128))") is None


# ---------------------------------------------------------------------------------------------- [COMP, CX1] TP wrapper shapes
TP_KERNEL = """
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import reduction

@reduction(size_hints={'x': 8192, 'r0_': 2048}, reduction_hint=ReductionHint.INNER, filename=__file__, triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=%d, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}}, inductor_meta={'kernel_name': 'triton_red_fused_fused_add_rms_norm_2', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': 'B'})
@triton.jit
def triton_red_fused_fused_add_rms_norm_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    tmp0 = tl.load(in_out_ptr0 + (r0_1 + 1536*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 1536*x0), tmp32, r0_mask & xmask)
"""


def tp_wrapper(device_index: int) -> str:
    """The shapes cx1's rank wrappers had that W1d's TP=1 wrappers did not: an extern custom op whose functional result is
    aliased (`buf3 = torch.ops.vllm.all_reduce...; buf4 = buf3`) with the record on the alias, and an in-place `in_out_ptr`
    launch over a storage-reuse rename of that result (`buf23 = buf19; del buf19  # reuse`)."""
    id0 = 28 * ID_STRIDE
    return textwrap.dedent(
        f"""
        async_compile = AsyncCompile()

        triton_red_fused_fused_add_rms_norm_2 = async_compile.triton('triton_red_fused_fused_add_rms_norm_2', '''__K__''', device_str='cuda')

        matmul_kernel_persistent_2 = async_compile.triton('matmul_kernel_persistent_2', '''
        @triton.jit
        def matmul_kernel_persistent_2(a_ptr, b_ptr, c_ptr, bias_ptr, M, N, K):
            pass
        ''', device_str='cuda')

        class Runner:
            def call(self, args):
                arg0_1, arg1_1, arg5_1, arg8_1, arg9_1 = args
                args.clear()
                s72 = arg1_1
                with torch.cuda._DeviceGuard({device_index}):
                    torch.cuda.set_device({device_index})
                    buf14 = empty_strided_cuda((s72, 1536), (1536, 1), torch.bfloat16)
                    matmul_kernel_persistent_2.run(arg0_1, reinterpret_tensor(arg8_1, (4480, 1536), (1, 4480), 0), buf14, -1, s72, 1536, 4480, 4480, 1, 1, 4480, 1536, 1, min(108, 12*((127 + s72) // 128)), 1, 1, stream=raw_stream0)
                    buf18 = torch.ops.vllm.all_reduce.default(buf14, 'tp:0')
                    buf19 = buf18
                    assert_size_stride(buf19, (s72, 1536), (1536, 1), 'torch.ops.vllm.all_reduce.default')
                    assert_alignment(buf19, 16, 'torch.ops.vllm.all_reduce.default')
                    del buf18
                    torch.ops.verity.record.default(buf14, {id0 + 7})
                    torch.ops.verity.record.default(buf19, {id0 + 8})
                    buf23 = buf19; del buf19  # reuse
                    triton_red_fused_fused_add_rms_norm_2.run(buf23, buf14, arg5_1, arg9_1, s72, 1536, stream=raw_stream0)
                    torch.ops.verity.record.default(buf23, {id0 + 9})
                return (buf23, )

        runner = Runner(partitions=[])
        call = runner.call


        def get_args():
            arg0_1 = rand_strided((8192, 4480), (4480, 1), device='cuda:{device_index}', dtype=torch.bfloat16)
            arg1_1 = 8192
            return [arg0_1, arg1_1]


        def benchmark_compiled_module(args, times=10, repeat=10):
            return 0
        """
    ).replace("__K__", TP_KERNEL % device_index)


def test_parse_program_extern_alias_and_in_place_reuse(tmp_path: Path):
    p = write_cache(tmp_path, "rank0", tp_wrapper(0))
    prog = P.parse_program(p)
    assert prog is not None and prog.piece == 28
    gemm, allreduce, norm = prog.launches
    # the extern op is a launch: functional result first (port 0), the group name a scalar arg
    assert allreduce.kernel_name == "vllm.all_reduce" and allreduce.kernel_var == "extern_vllm_all_reduce"
    assert allreduce.outputs == ["buf18"] and allreduce.inputs == ["buf14"]
    # the record on the alias (buf19) resolves to the extern's result buffer
    assert prog.record_buffer[28 * ID_STRIDE + 8] == "buf18"
    assert prog.buffer_record["buf18"] == 28 * ID_STRIDE + 8
    # the in-place norm: buf23 (storage reuse of buf19 = buf18) is its output AND it reads the all-reduce value first
    assert norm.outputs == ["buf23"] and norm.inout == {"buf23": "buf18"}
    assert norm.inputs[0] == "buf18" and "buf14" in norm.inputs and "arg5_1" in norm.inputs
    assert prog.returned == ["buf23"]


def test_kernel_identity_modulo_device_index(tmp_path: Path):
    """Rank r compiles on cuda:r; Inductor bakes DeviceProperties(index=r) into every kernel: the ranks' kernels differ in that
    integer only. norm_hash keeps telling them apart (frozen for the TP=1 cells); the comparison also reports the device-agnostic view."""
    write_cache(tmp_path, "prod", tp_wrapper(0) + "\n" + tp_wrapper(1).replace("class Runner", "class Runner1"))
    write_cache(tmp_path, "rank1", tp_wrapper(1))
    a = K.normalise_kernel(TP_KERNEL % 0)
    b = K.normalise_kernel(TP_KERNEL % 1)
    assert a != b and K.device_agnostic(a) == K.device_agnostic(b)
    # Inductor's combo kernels carry a benchmark harness in the source string with the device index in four more spellings
    # (cx1 commit: rank 1's k-rope+qkv-bias combo counted as production-only until these were normalised too)
    combo = ("\ndef get_args():\n    arg_0 = rand_strided((8192, 1024), (1024, 1), device='cuda:%d', dtype=torch.bfloat16)\n"
             "def call(args):\n    with torch.cuda._DeviceGuard(%d):\n        torch.cuda.set_device(%d)\n"
             "        raw_stream%d = get_raw_stream(%d)\n        triton_poi_fused_3.run(*args, stream=raw_stream%d)\n")
    ca = K.normalise_kernel(TP_KERNEL % 0 + combo % ((0,) * 6))
    cb = K.normalise_kernel(TP_KERNEL % 1 + combo % ((1,) * 6))
    assert ca != cb and K.device_agnostic(ca) == K.device_agnostic(cb)
    assert "cuda:N" in K.device_agnostic(ca) and "raw_streamN = get_raw_stream(N)" in K.device_agnostic(ca)
    cmp = K.compare(tmp_path / "prod", tmp_path / "rank1")
    assert cmp["kernels_identical"] is False and cmp["only_in_replay"] == []
    assert cmp["replay_subset_of_production_modulo_device_index"] is True
    assert cmp["only_in_replay_modulo_device_index"] == 0
    assert cmp["production"]["distinct_kernels_modulo_device_index"] == cmp["replay"]["distinct_kernels_modulo_device_index"] == 2
