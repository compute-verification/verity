"""`verity_vllm/acquire/fa3_tap_src/apply_fa3_tap.py`, the script a pod runs to turn the `hopper` directory of the pinned
FlashAttention tree into the FA3-tapped one, without that tree.

The script is run whole, as `ops/pod_fa3_tap.sh` starts it: with the directory to edit and this directory as its two
arguments. The directory is a stand-in. Each of the four files the script reads holds lines of the pinned tree's file, in
that file's order: the lines the script looks for, and at three places what an edit must leave as it is. Those are the
pinned `scale_apply_exp2`, whose arithmetic the script writes a second time; `finalize_aux`, which ends as `finalize`
does and gets no tap; and the branch without `IntraWGOverlap`. So a script that asked for a line these lack is refused
here, and so is one that could not tell `finalize` from `finalize_aux`.
"""

from __future__ import annotations

import difflib
import hashlib
import runpy
import sys
from pathlib import Path

import pytest

import verity_vllm.acquire.fa3_tap_src.build_fa3_ext as BUILD

#: the tap's directory: the script's second argument on a pod, and the build's own `--src-dir`
MINE = Path(BUILD.__file__).parent
SCRIPT = MINE / "apply_fa3_tap.py"
HEADERS = ("verity_tap_fa3.h", "verity_tap_params.h")
SOFTMAX_H = "softmax.h"
MAINLOOP_HPP = "mainloop_fwd_sm90_tma_gmma_ws.hpp"
LAUNCH_H = "flash_fwd_launch_template.h"
LIBRARY_CPP = "flash_api_torch_lib.cpp"
TAPPED_CPP = "fa3_torch_lib_verity.cpp"
#: the two conditions the script puts its lines under: the tap's build, and the build that also keeps each row's `max_scaled`
TAP_ON = "#if defined(VERITY_TAP) && VERITY_TAP == 3"
MS_ON = "#if defined(VERITY_MAT_SRC) && (VERITY_MAT_SRC & 64)"

#: of the pinned `softmax.h`: `scale_apply_exp2` whole; the first lines of `Softmax` and its two members; the call in
#: `online_softmax`; `finalize` from its sum to its end; and `finalize_aux` from its inverse to its end
SOFTMAX = """\
// Apply the exp to all the elements.
template <bool Scale_max=true, bool Check_inf=true, int Max_offset=0,
        typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void scale_apply_exp2(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> const &max, const float scale) {
    // For FP8, we can subtract max by 8.0 so that the value after exp2 is in the range of [0, 256].
    // This lets us use more of the FP8 range (instead of just [0, 1]) to reduce underflow.
    static constexpr float max_offset = float(Max_offset);  // We can only template on int, not float
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        // If max is -inf, then all elements must have been -inf (possibly due to masking).
        // We don't want (-inf - (-inf)) since that would give NaN.
        const float max_scaled = Check_inf
            ? (max(mi) == -INFINITY ? 0.f : (!Scale_max ? max(mi) : max(mi) * scale) - max_offset)
            : (!Scale_max ? max(mi) : max(mi) * scale) - max_offset;
        #pragma unroll
        for (int ni = 0; ni < size<1>(tensor); ++ni)  {
            // Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
            // max * log_2(e)). This allows the compiler to use the ffma
            // instruction instead of fadd and fmul separately.
            tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
        }
    }
}
template <int kNRows, int Max_offset=0>
struct Softmax {
    TensorT row_max, row_sum;
        flash::template scale_apply_exp2</*Scale_max=*/true, Check_inf, Max_offset>(scores, row_max, softmax_scale_log2);
    __forceinline__ __device__ TensorT finalize(float const final_scale=1.f) {
            float sum = row_sum(mi);
            float inv_sum = (sum == 0.f || sum != sum) ? 0.f : 1.f / sum;
            scores_scale(mi) = inv_sum * final_scale;
            // For FP8, we might have scaled the output of exp by 2**8 so we need to divide sum by that amount.
            if constexpr (Max_offset != 0) {
                static constexpr float sum_scale = 1.f / float(1 << Max_offset);
                sum *= sum_scale;
            }
            row_sum(mi) = (sum == 0.f || sum != sum) ? -INFINITY : row_max(mi) * (softmax_scale_log2 * float(M_LN2)) + __logf(sum);
        }
        return scores_scale;
    };

    __forceinline__ __device__ TensorT finalize_aux(TensorT const& tSrSAux, float const final_scale=1.f) {
            float inv_sum = (sum == 0.f || sum != sum) ? 0.f : 1.f / sum;
            scores_scale(mi) = inv_sum * final_scale;
            // For FP8, we might have scaled the output of exp by 2**8 so we need to divide sum by that amount.
            if constexpr (Max_offset != 0) {
                static constexpr float sum_scale = 1.f / float(1 << Max_offset);
                sum *= sum_scale;
            }
            row_sum(mi) = (sum == 0.f || sum != sum) ? -INFINITY : row_max(mi) * (softmax_scale_log2 * float(M_LN2)) + __logf(sum);
        }
        return scores_scale;
    };
"""
#: of the pinned mainloop: the two includes the tap's follows; the ends of the arguments and of the kernel's own, and
#: where one is copied into the other; the block count of `mma`; the `IntraWGOverlap` branch's first block, its step and
#: its last product; and three lines of the other branch
MAINLOOP = """\
#include "utils.h"
#include "sm90_pipeline_no_cluster.hpp"
        int const* const cp_tot_seqused_k = nullptr;
    };

    // Device side kernel params
    struct Params {
        int const* const cp_tot_seqused_k = nullptr;
    };

    static Params
    to_underlying_arguments(Arguments const& args) {
                args.cp_world_size, args.cp_rank, args.cp_tot_seqused_k};
    }
        int const seqlen_k = seqlen_info.seqlen_k - n_offset;
        int n_block = n_block_max - 1;

        if constexpr (IntraWGOverlap) {
            Tensor tSrS = partition_fragment_C(tiled_mma_qk, select<0, 1>(TileShape_MNK{}));
            mask.template apply<true /*Seqlenk_mask*/, Is_causal, Is_local>(tSrS, m_block, n_block);

            Tensor scores_scale = softmax.template max_get_scale</*Is_first=*/true, /*Check_inf=*/true>(tSrS);
            softmax.template online_softmax</*Is_first=*/true, /*Check_inf=*/true>(tSrS);
            if constexpr (Is_FP8 && !V_colmajor) { flash::permute_Cregs_fp8(tSrS); }
            Tensor tOrP_acc = make_tensor(tSrS.data(), flash::convert_layout_acc_Aregs<TiledMmaPV>(tSrS.layout()));
            Tensor tOrP = make_tensor_like<Element>(tOrP_acc);
            convert_type_out(tOrP_acc, tOrP);
                scoremod_premask_fn(tSrS);
                mask_fn(tSrS, n_block);
                cute::copy(softmax.template max_get_scale</*Is_first=*/false, Check_inf>(tSrS), scores_scale);
                softmax.template online_softmax</*Is_first=*/false, Check_inf>(tSrS);
                if constexpr (!HasQv) {
                    warpgroup_wait<0>();
                    pipeline_v.consumer_release(smem_pipe_read_v);  // release V
                }
                if constexpr (Is_FP8 && !V_colmajor) { flash::permute_Cregs_fp8(tSrS); }
                convert_type_out(make_tensor(tSrS.data(), tOrP.layout()), tOrP);
            warpgroup_wait<0>();
            pipeline_v.consumer_release(smem_pipe_read);  // release V, otherwise producers will hang
            softmax.rescale_o(tOrO, scores_scale);
        } else {  // No intra-WG overlap
                scoremod_premask_fn(tSrS);
                mask_fn(tSrS, n_block);
            softmax.rescale_o(tOrO, scores_scale);
"""
#: of the pinned launch: where the mainloop's arguments begin and end
LAUNCH = """\
    typename CollectiveMainloop::Arguments mainloop_args {
        params.cp_world_size, params.cp_rank, params.cp_tot_seqused_k
    };
    typename CollectiveEpilogue::Arguments epilogue_args {
"""
#: of the pinned library file: the registration's comment, the end of its declarations, and the extension's last line
LIBRARY = """\
/**
 *  Torch Library Registration
 */
STABLE_TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
            "    int      sm_margin) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, ops) {
    ops.impl("fwd", TORCH_BOX(&mha_fwd_stable));
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME);
"""
STAND_IN = {SOFTMAX_H: SOFTMAX, MAINLOOP_HPP: MAINLOOP, LAUNCH_H: LAUNCH, LIBRARY_CPP: LIBRARY}
MEMBERS = "    TensorT row_max, row_sum;"
CALL = "        flash::template scale_apply_exp2</*Scale_max=*/true, Check_inf, Max_offset>(scores, row_max, softmax_scale_log2);"
INVERSE = "            float inv_sum = (sum == 0.f || sum != sum) ? 0.f : 1.f / sum;"
FINALIZE_AUX = "    __forceinline__ __device__ TensorT finalize_aux(TensorT const& tSrSAux, float const final_scale=1.f) {"
ARGUMENTS = "                args.cp_world_size, args.cp_rank, args.cp_tot_seqused_k};"
OTHER_BRANCH = "        } else {  // No intra-WG overlap"
MAINLOOP_ARGUMENTS = "    typename CollectiveMainloop::Arguments mainloop_args {"
LAST_ARGUMENT = "        params.cp_world_size, params.cp_rank, params.cp_tot_seqused_k"
EXPAND = "STABLE_TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {"
REGISTER = "REGISTER_EXTENSION(TORCH_EXTENSION_NAME);"
#: the last three lines of the `IntraWGOverlap` branch that the script looks for, after the last block's product
LAST_PRODUCT = "\n".join([
    "            warpgroup_wait<0>();",
    "            pipeline_v.consumer_release(smem_pipe_read);  // release V, otherwise producers will hang",
    "            softmax.rescale_o(tOrO, scores_scale);",
])
#: what the script puts into `softmax.h` after its function: a member for each row's `max_scaled`, two for `finalize`'s
#: sum and its inverse, the call that keeps `max_scaled` above the pinned call, and the statement that keeps the two sums
KEPT = [
    MS_ON,
    "    TensorT tap_max_scaled;   // [verity MS] each row's max * scale this block's exp2 FMAs read",
    "#endif",
    TAP_ON,
    "    TensorT tap_sum, tap_inv_sum;   // [verity fa3 tap] finalize's quad-reduced sum and inv_sum (row_sum becomes the lse)",
    "#endif",
    MS_ON,
    CALL.replace("scale_apply_exp2<", "scale_apply_exp2_ms<").replace("log2);", "log2, tap_max_scaled);"),
    "#else",
    "#endif",
    TAP_ON,
    "            tap_sum(mi) = sum; tap_inv_sum(mi) = inv_sum;",
    "#endif",
]
#: of `KEPT`, the lines of the script's first edit of `softmax.h`: the two sums
SUMS = KEPT[3:6] + KEPT[10:]
#: the pinned lines of the `IntraWGOverlap` branch that tap macros follow, each with its calls: the scores once masked;
#: the probabilities and each row's state after the softmax; the probabilities as the product reads them; the output
#: once a block's product is done; and after the last product, the sums `finalize` kept
PLACES = [
    ("            mask.template apply<true /*Seqlenk_mask*/, Is_causal, Is_local>(tSrS, m_block, n_block);",
     ["            VERITY_FA3_S(tSrS, n_block)"]),
    ("            softmax.template online_softmax</*Is_first=*/true, /*Check_inf=*/true>(tSrS);",
     ["            VERITY_FA3_P(tSrS, scores_scale, n_block)"]),
    ("            convert_type_out(tOrP_acc, tOrP);", ["            VERITY_FA3_RP(tOrP, tSrS, n_block)"]),
    ("                mask_fn(tSrS, n_block);", ["                VERITY_FA3_S(tSrS, n_block)"]),
    ("                softmax.template online_softmax</*Is_first=*/false, Check_inf>(tSrS);",
     ["                VERITY_FA3_P(tSrS, scores_scale, n_block)"]),
    ("                    pipeline_v.consumer_release(smem_pipe_read_v);  // release V\n                }",
     ["                VERITY_FA3_O(tOrO, n_block + 1)   // the P.V wgmma of block n_block + 1 has completed: acc_o after that block"]),
    ("                convert_type_out(make_tensor(tSrS.data(), tOrP.layout()), tOrP);",
     ["                VERITY_FA3_RP(tOrP, tSrS, n_block)"]),
    ("            pipeline_v.consumer_release(smem_pipe_read);  // release V, otherwise producers will hang",
     ["            VERITY_FA3_O(tOrO, n_block + 1)   // final P.V (block n_block + 1 == n_block_min) complete",
      "            VERITY_FA3_FIN()"]),
]
CALLS = [call for _, calls in PLACES for call in calls]
AFTER_MASK, AFTER_SOFTMAX, AFTER_CONVERT, _, STEP_SOFTMAX, _, STEP_CONVERT, LAST_RELEASE = (line for line, _ in PLACES)
#: the last line of the first three, and of the next three, that the script looks for in the branch
FIRST_SCALE = "            Tensor scores_scale = softmax.template max_get_scale</*Is_first=*/true, /*Check_inf=*/true>(tSrS);"
STEP_SCALE = "                cute::copy(softmax.template max_get_scale</*Is_first=*/false, Check_inf>(tSrS), scores_scale);"
#: the five macros as a build without the tap defines them
EMPTY = [
    "  #define VERITY_FA3_S(tSrS_, nb)",
    "  #define VERITY_FA3_P(tSrS_, sc_, nb)",
    "  #define VERITY_FA3_RP(tOrP_, tSrS_, nb)",
    "  #define VERITY_FA3_O(tOrO_, nb)",
    "  #define VERITY_FA3_FIN()",
]
#: the mainloop's lines outside `TAP_ON`: the tap's header, the parameter `vt` in the arguments and in the kernel's own, and its copy
PARAMETER = [
    '#include "verity_tap_fa3.h"',
    "        Verity_tap vt = {};   // [verity fa3 tap]",
    "        Verity_tap vt = {};   // [verity fa3 tap]",
    ARGUMENTS.replace("};", ", args.vt};"),
]
#: the ops the script adds to the pinned library file, each with its schema
OPS = {
    "verity_set_tap_v9": "(int mode, Tensor? diag, int HB, int M, int NB, int BN, int D, int tag) -> ()",
    "verity_set_tap_off": "() -> ()",
    "verity_tap_skipped": "() -> int",
    "verity_tap_mode": "() -> int",
    "verity_tap_src": "() -> int",
    "verity_tap_layout_version": "() -> int",
    "verity_tap_hdims": "() -> int",
    "verity_tap_row3": "() -> int",
}
#: SHA-512 of each file the script edits or writes, as it leaves the stand-in
TAPPED = {
    SOFTMAX_H: "922e6ebd9fac9bc70b40ee30d061d99fe6f76994ae8c14ee548ca6387b80cfd9207adb72c6bac8aa05276417253d058d2f62cde00abb63400d3fc371db8e296a",
    MAINLOOP_HPP: "e1e4d50f9d4eafbf4c4fb2147d74009acfa5bd60a404a2eedfddeb544c9ec44abe042b3c6df62baeb954392a6f7f67ca2fa8be1f4c127fba2d7c777095be2a06",
    LAUNCH_H: "a072d3bbf58117548c05cf128fce4c40c58dad073ce38779912a4e32af363023c847c88f378e288bf1af56224470decf43b84298f5915cb2a7a99a174ba2eaa5",
    TAPPED_CPP: "cc5e56f2238d615dd794202f5a564abfe4af25deacf17f4c35ae38790fc60ce49a34b5aa4ff6df3e2a9c13b486a04dc782cfc544c50d5c91973b8c8c6e7a0fc7",
}
SAID = [
    "[fa3_tap] softmax.h: patched",
    "[fa3_tap] softmax.h: patched",
    f"[fa3_tap] {MAINLOOP_HPP}: patched",
    f"[fa3_tap] {MAINLOOP_HPP}: patched",
    "[fa3_tap] flash_fwd_launch_template.h: patched",
    "[fa3_tap] fa3_torch_lib_verity.cpp: written",
    "[fa3_tap] OK",
]


def _tree(root: Path, changed: dict[str, str] | None = None) -> Path:
    """The stand-in directory at `root`, with `changed` in place of the files it names."""
    root.mkdir(parents=True)
    for name, text in {**STAND_IN, **(changed or {})}.items():
        (root / name).write_text(text)
    return root


def _read(root: Path) -> dict[str, str]:
    return {path.name: path.read_text() for path in root.iterdir()}


def _apply(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), str(root), str(MINE)])
    runpy.run_path(str(SCRIPT), run_name="__main__")


def _changed(before: str, after: str) -> tuple[list[str], list[str]]:
    """The lines of `before` that `after` lacks, and the lines it has more, each in its file's order."""
    lines = list(difflib.ndiff(before.splitlines(), after.splitlines()))
    return [line[2:] for line in lines if line.startswith("- ")], [line[2:] for line in lines if line.startswith("+ ")]


def _without_the_tap(text: str) -> str:
    """`text` as a compiler reads it that defines neither `VERITY_TAP` nor `VERITY_MAT_SRC`: of each block under `TAP_ON`
    or `MS_ON`, the lines after its `#else` and no other."""
    out: list[str] = []
    depth, kept = 0, False
    for line in text.splitlines():
        word = line.strip()
        if not depth:
            if line in (TAP_ON, MS_ON):
                depth, kept = 1, False
            else:
                out.append(line)
        elif word.startswith("#if"):
            depth += 1
        elif word == "#endif":
            depth -= 1
        elif word == "#else" and depth == 1:
            kept = True
        elif kept and depth == 1:
            out.append(line)
    return "\n".join(out) + "\n"


def _keeping_max_scaled(pinned: list[str]) -> list[str]:
    """The pinned `scale_apply_exp2` as the script writes it a second time, under `MS_ON`: its lines without their
    comments, named `scale_apply_exp2_ms`, with one tensor more, `ms`, and the one statement that fills it."""
    code = [line.partition("  //")[0] for line in pinned if not line.lstrip().startswith("//")]
    return [
        MS_ON,
        "// [verity MS] scale_apply_exp2 with the same arithmetic that also keeps each row's max_scaled, the register its exp2 FMAs read, in `ms`",
        code[0],
        code[1].replace("Layout1>", "Layout1, typename Engine2, typename Layout2>"),
        code[2].replace("scale_apply_exp2(", "scale_apply_exp2_ms(").replace("scale) {", "scale,"),
        " " * 52 + "Tensor<Engine2, Layout2> &ms) {",
        *code[3:12],
        "        ms(mi) = max_scaled;",
        *code[12:],
        "#endif",
        "",
    ]


def _out(line: str) -> str:
    """`line` as a comment: a tree that has it so lacks it."""
    return line.replace(line.lstrip(), "// " + line.lstrip())


def _from(name: str, first: str, length: int) -> str:
    """What a refusal shows of the lines it looked for: `length` characters of the stand-in, from `first` on."""
    at = STAND_IN[name].index(first)
    return STAND_IN[name][at:at + length]


@pytest.fixture
def tapped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """The stand-in's files after one run of the script."""
    root = _tree(tmp_path / "hopper")
    _apply(root, monkeypatch)
    return _read(root)


def test_the_script_makes_the_tapped_directory_of_a_pinned_one_and_says_each_edit(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """It copies this directory's two headers in, edits three files, and writes the tapped library file beside the pinned
    one, which it leaves. Each of the four is held whole, by digest; the tests below say what the edits are. The last
    assertion is the line `ops/pod_fa3_tap.sh` looks for before it builds."""
    root = _tree(tmp_path / "hopper")

    _apply(root, monkeypatch)

    after = _read(root)
    assert capsys.readouterr().out.splitlines() == SAID
    assert sorted(after) == sorted([*STAND_IN, *HEADERS, TAPPED_CPP])
    assert all(after[name] == (MINE / name).read_text() for name in HEADERS)
    assert after[LIBRARY_CPP] == LIBRARY
    assert {name: hashlib.sha512(after[name].encode()).hexdigest() for name in TAPPED} == TAPPED
    assert "Verity_mat_ctx_sm90<kBlockM, kBlockN, kHeadDim> vmat(" in after[MAINLOOP_HPP]


def test_a_build_without_the_tap_reads_the_pinned_lines_and_one_parameter_more(tapped: dict[str, str]) -> None:
    """Everything else the script writes is under `TAP_ON` or `MS_ON`. Without either, `softmax.h` is the pinned file and
    an empty line; the mainloop gains the tap's header, the parameter `vt` and five macros that are empty, with their
    calls; and the launch gains an empty `vt` to pass."""
    assert _changed(SOFTMAX, _without_the_tap(tapped[SOFTMAX_H])) == ([], [""])

    gone, new = _changed(MAINLOOP, _without_the_tap(tapped[MAINLOOP_HPP]))
    assert gone == [ARGUMENTS] and new == [*PARAMETER, *EMPTY, "", *CALLS]
    assert {call.split("(")[0].strip() for call in CALLS} == {macro.split("(")[0].split()[1] for macro in EMPTY}

    gone, new = _changed(LAUNCH, _without_the_tap(tapped[LAUNCH_H]))
    assert gone == [LAST_ARGUMENT]
    assert new == ["    Verity_tap vt = {};", LAST_ARGUMENT + ",", "        vt   // [verity fa3 tap]"]


def test_the_tap_reads_each_value_where_the_kernel_has_it_and_repeats_the_pinned_arithmetic(tapped: dict[str, str]) -> None:
    """In the mainloop each macro's call follows the pinned line that makes its value, and the branch without
    `IntraWGOverlap` gets none. In `softmax.h` the function that keeps `max_scaled` is the pinned function with `ms`, the
    sums are kept in `finalize` alone, and the pinned call stays for a build without `MS_ON`."""
    mainloop, softmax = tapped[MAINLOOP_HPP], tapped[SOFTMAX_H]
    for line, calls in PLACES:
        assert "\n".join([line, *calls]) + "\n" in mainloop
    assert "VERITY_FA3" not in mainloop.partition(OTHER_BRANCH)[2]

    gone, new = _changed(SOFTMAX, softmax)
    assert gone == [] and new == [*_keeping_max_scaled(SOFTMAX.splitlines()[:26]), *KEPT]
    assert "\n".join([MEMBERS, *KEPT[:6]]) + "\n" in softmax
    assert "\n".join([*KEPT[6:9], CALL, "#endif"]) + "\n" in softmax
    assert "\n".join([INVERSE, "            scores_scale(mi) = inv_sum * final_scale;", *KEPT[10:]]) + "\n" in softmax
    assert softmax.index(KEPT[11]) < softmax.index(FINALIZE_AUX)


def test_the_library_file_defines_declares_and_registers_the_same_eight_ops(tapped: dict[str, str]) -> None:
    """`fa3_torch_lib_verity.cpp` is the pinned file's lines and three blocks more: the ops' functions before the
    registration, their schemas at the end of the pinned declarations, and their registration before the extension's."""
    pinned = LIBRARY.splitlines()
    gone, new = _changed(LIBRARY, tapped[TAPPED_CPP])
    functions, declared, registered = new[:45], new[45:53], new[53:]

    assert gone == []
    assert tapped[TAPPED_CPP].splitlines() == [*functions, *pinned[:5], *declared, *pinned[5:11], *registered, REGISTER]
    for name in OPS:
        assert sum(line.startswith((f"void {name}(", f"int64_t {name}(")) for line in functions) == 1
    assert declared == [f'    ops.def("{name}{schema}");' for name, schema in OPS.items()]
    assert registered == [
        "STABLE_TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CompositeExplicitAutograd, ops) {",
        *[f'    ops.impl("{name}", TORCH_BOX(&{name}));' for name in OPS],
        "}",
        "",
    ]


def test_a_second_run_edits_nothing_again_and_writes_the_headers_and_the_library_file_again(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Each edit stands behind a mark of its own in the file it edits, so the script may be run on its own result. What
    it copies and what it writes whole, it copies and writes on every run."""
    root = _tree(tmp_path / "hopper")
    _apply(root, monkeypatch)
    once = _read(root)
    capsys.readouterr()
    for name in (HEADERS[0], TAPPED_CPP):
        (root / name).write_text("// a file of an earlier day\n")

    _apply(root, monkeypatch)

    assert _read(root) == once
    assert capsys.readouterr().out.splitlines() == [line.replace(": patched", ": already patched") for line in SAID]


def _refused(name: str, old: str, new: str, refusal: type[Exception], message: str, said: int, tmp_path: Path,
             monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """The script on a directory whose file `name` has `new` for `old` is refused with `message` after `said` lines. The
    refused edit writes nothing: the file is as it was, but for the two sums when the script's second edit of `softmax.h`
    is the one refused; the edits before stand and are said; no library file is written. With `old` back, a second run
    makes the directory that a run on a whole stand-in makes."""
    assert STAND_IN[name].count(old) == 1
    lacking = STAND_IN[name].replace(old, new)
    root = _tree(tmp_path / "hopper", {name: lacking})

    with pytest.raises(refusal) as refused:
        _apply(root, monkeypatch)

    assert str(refused.value) == message.replace("<path>", str(root / name))
    assert capsys.readouterr().out.splitlines() == SAID[:said]
    kept = (root / name).read_text()
    assert _changed(lacking, kept) == ([], SUMS if (name, said) == (SOFTMAX_H, 1) else [])
    assert not (root / TAPPED_CPP).exists()

    (root / name).write_text(kept.replace(new, old))
    _apply(root, monkeypatch)
    whole = _tree(tmp_path / "whole" / "hopper")
    _apply(whole, monkeypatch)
    assert _read(root) == _read(whole)


@pytest.mark.parametrize(("name", "old", "new", "found", "shown", "said"), [
    (SOFTMAX_H, MEMBERS, None, 0, MEMBERS + "\n", 0),
    (SOFTMAX_H, MEMBERS, MEMBERS + "\n" + MEMBERS, 2, MEMBERS + "\n", 0),
    (SOFTMAX_H, FINALIZE_AUX, None, 0, _from(SOFTMAX_H, INVERSE, 200), 0),
    (SOFTMAX_H, "struct Softmax {", None, 0, "template <int kNRows, int Max_offset=0>\nstruct Softmax {\n", 1),
    (SOFTMAX_H, CALL, None, 0, CALL + "\n", 1),
    (MAINLOOP_HPP, FIRST_SCALE, None, 0, _from(MAINLOOP_HPP, AFTER_MASK, 160), 2),
    (MAINLOOP_HPP, AFTER_CONVERT, None, 0, _from(MAINLOOP_HPP, AFTER_SOFTMAX, 160), 2),
    (MAINLOOP_HPP, STEP_SCALE, None, 0, _from(MAINLOOP_HPP, "                scoremod_premask_fn(tSrS);", 160), 2),
    (MAINLOOP_HPP, STEP_CONVERT, None, 0, _from(MAINLOOP_HPP, STEP_SOFTMAX, 160), 2),
    (MAINLOOP_HPP, LAST_RELEASE, None, 0, _from(MAINLOOP_HPP, LAST_PRODUCT, 160), 2),
    (MAINLOOP_HPP, LAST_PRODUCT, LAST_PRODUCT + "\n" + LAST_PRODUCT, 2, _from(MAINLOOP_HPP, LAST_PRODUCT, 160), 2),
    (LAUNCH_H, MAINLOOP_ARGUMENTS, None, 0, MAINLOOP_ARGUMENTS + "\n", 4),
    (LAUNCH_H, LAST_ARGUMENT, None, 0, LAUNCH[LAUNCH.index(LAST_ARGUMENT):-1], 4),
], ids=["no-members", "members-twice", "no-finalize-before-finalize-aux", "no-struct", "no-call", "no-first-scale",
        "no-first-convert", "no-step-scale", "no-step-convert", "no-last-release", "last-product-twice",
        "no-mainloop-arguments", "no-last-argument"])
def test_the_script_refuses_a_directory_that_lacks_lines_it_edits_and_shows_them(
        name: str, old: str, new: str | None, found: int, shown: str, said: int, tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """An `AssertionError` that says how often the lines were found and shows their first 200 characters; in the mainloop
    160, without the file's path. `new` is `old` twice, or nothing: `old` as a comment."""
    message = f"mainloop anchor found {found}x:\n{shown}" if name == MAINLOOP_HPP else (
        f"<path>: anchor found {found}x (expected 1):\n{shown}")
    _refused(name, old, new or _out(old), AssertionError, message, said, tmp_path, monkeypatch, capsys)


@pytest.mark.parametrize(("name", "old", "new", "refusal", "said"), [
    (MAINLOOP_HPP, "    struct Params {", None, AssertionError, 2),
    (MAINLOOP_HPP, "    to_underlying_arguments(Arguments const& args) {", None, AssertionError, 2),
    (MAINLOOP_HPP, ARGUMENTS, None, AssertionError, 2),
    (MAINLOOP_HPP, "            Tensor tSrS = partition_fragment_C(tiled_mma_qk, select<0, 1>(TileShape_MNK{}));", None, ValueError, 2),
    (MAINLOOP_HPP, OTHER_BRANCH, None, ValueError, 2),
    (MAINLOOP_HPP, "        int const seqlen_k = seqlen_info.seqlen_k - n_offset;", None, AssertionError, 2),
    (LIBRARY_CPP, EXPAND, EXPAND + "  // verity_set_tap_v9", AssertionError, 5),
    (LIBRARY_CPP, " *  Torch Library Registration", None, AssertionError, 5),
    (LIBRARY_CPP, '            "    int      sm_margin) -> Tensor");', None, AssertionError, 5),
    (LIBRARY_CPP, REGISTER, REGISTER.replace("NAME", "NAMES"), AssertionError, 5),
], ids=["no-kernel-parameters", "no-copy-of-the-arguments", "no-last-copied-argument", "no-branch", "no-other-branch",
        "no-block-count", "an-op-already", "no-registration", "no-last-declaration", "no-extension"])
def test_the_script_refuses_without_a_message_a_directory_that_lacks_the_other_lines_it_edits(
        name: str, old: str, new: str | None, refusal: type[Exception], said: int, tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Four places of the mainloop and three of the library file are asserted with no message, and so is a library file
    that names one of the ops already. The two ends of the `IntraWGOverlap` branch are looked for with `str.index`,
    whose `ValueError` names neither the file nor the line."""
    message = "substring not found" if refusal is ValueError else ""
    _refused(name, old, new or _out(old), refusal, message, said, tmp_path, monkeypatch, capsys)
