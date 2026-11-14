"""`verity_vllm/acquire/fa2_tap_src/apply_v8.py`, the script a pod runs to turn the v7-tapped FlashAttention tree into the
v8 one, without that tree.

The script is run whole, as `ops/pod_fa2_tap.sh` starts it: from the tree's root, with the source directory relative and
this directory as its second argument. The tree is a stand-in made from the directory's own `verity_tap_v7.patch`: each
file the patch touches holds the lines of the patch's hunks as they stand after it. So the script reads five of the seven
lines it looks for, and one of the two calls of a sixth, from the patch's text, and a script that asked for a line the
patch does not make is refused here. `softmax.h` gets three lines more, which the patch does not show and which stand in
for the pinned tree's: the comment and template line the script puts its function before, and the second call of
`scale_apply_exp2`.
"""

from __future__ import annotations

import difflib
import hashlib
import runpy
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import verity_vllm.acquire.fa2_tap_src.build_ext as BUILD

#: the tap's directory: the script's second argument on a pod, and the build's own `--src-dir`
MINE = Path(BUILD.__file__).parent
SCRIPT = MINE / "apply_v8.py"
SRC = Path("csrc", "flash_attn", "src")
FILES = ("flash.h", "flash_fwd_kernel.h", "flash_fwd_launch_template.h", "softmax.h", "verity_tap.h")
MS_ON = "#if defined(VERITY_MAT_SRC) && (VERITY_MAT_SRC & 64)"
#: stands in for the two lines the script puts its function before; the script reads them as far as `Engine0`
BEFORE_THE_FUNCTION = "// Apply the exp to all the elements.\ntemplate <bool zero_init=true, typename Engine0, typename Layout0>\n"
MEMBER = "    TensorT row_max, row_sum;"
CALL = "            FLASH_NAMESPACE::scale_apply_exp2(scores, row_max, softmax_scale_log2);"
MACRO_CALL = "            VERITY_SCALE_APPLY_EXP2(scores, row_max, softmax_scale_log2);"
#: what follows the first of the two calls in the patch's hunk
REDUCED = "\n            FLASH_NAMESPACE::reduce_sum</*zero_init=*/true>(scores, row_sum);"
TAG = "    int tag;              // v7: per-launch tag (layer / step id), bound into every leaf header"
CONTEXT = "    Verity_mat_ctx vmat(params.vt, bidb, params.h, bidh, m_block, tidx, kNWarps, binfo.actual_seqlen_q); \\"
TEMPLATED = CONTEXT.replace("Verity_mat_ctx vmat", "Verity_mat_ctx<kBlockM, kBlockN, kHeadDim> vmat")
ROWSTAT = ("    if constexpr (VERITY_MAT_SRC & 8) "
           "{ VERITY_MAT(vmat.rowstat(softmax.row_max, softmax.tap_scale, softmax.row_sum, n_block, vt_step)) }")
ARMED = ("        const bool ok = params.num_splits <= 1 && Kernel_traits::kBlockN == 128 && params.vt.diag_ptr != nullptr "
         "&& params.b * params.h <= params.vt.H")
#: the lines the script writes: one in `flash.h`, three in place of `ARMED`, one in the `Softmax` struct
TILE = ("    int BN, D;            // v8: host-declared kernel tile (kBlockN, kHeadDim) the M1 layout is arithmetic in; "
        "must equal the selected kernel's")
ARMED_BY_TILE = [
    ("        // v8: the host declared the tile (BN = kBlockN, D = kHeadDim) its layout is arithmetic in; "
     "arm only if this kernel IS that tile"),
    ("        const bool ok = params.num_splits <= 1 && Kernel_traits::kBlockN == params.vt.BN "
     "&& Kernel_traits::kHeadDim == params.vt.D"),
    ("                        && Kernel_traits::kBlockM == 64 && params.vt.diag_ptr != nullptr "
     "&& params.b * params.h <= params.vt.H"),
]
MAX_SCALED = "    TensorT tap_max_scaled;   // [verity MS] each row's max * scale this block's exp2 FMAs read"
#: SHA-512 of the 27 lines the script puts before `max_scale_exp2_sum`, joined by newlines: `scale_apply_exp2_ms` and the macro
MS_FUNCTION = ("89185d4efc8bfd4ef54f0c5f7f63d3cab0fce51c9c0200fbb3d6b3331d91b5b9"
               "62da0aabd83534b9e7f100869572de11175f9657f536c66eea3f085650107564")
SAID = [
    "[apply_v8] flash.h: Verity_tap.BN/D added",
    "[apply_v8] flash_fwd_kernel.h: Verity_mat_ctx<kBlockM, kBlockN, kHeadDim>",
    "[apply_v8] flash_fwd_launch_template.h: arm iff (kBlockN, kHeadDim) == (vt.BN, vt.D)",
    "[apply_v8] softmax.h: tap_max_scaled via scale_apply_exp2_ms (VERITY_MAT_SRC & 64)",
    "[apply_v8] flash_fwd_kernel.h: VERITY_TAP_P stores the MS class",
]


def _after_v7() -> dict[str, str]:
    """Each file the v7 patch touches, as the lines of its hunks after the patch, and `softmax.h` with its three lines more."""
    files: dict[str, list[str]] = {}
    name = ""
    for line in (MINE / "verity_tap_v7.patch").read_text().splitlines():
        if line.startswith("+++ b/"):
            name = Path(line[6:]).name
            files[name] = []
        elif name and line[:1] in (" ", "+"):
            files[name].append(line[1:])
    texts = {name: "\n".join(lines) + "\n" for name, lines in files.items()}
    texts["softmax.h"] = BEFORE_THE_FUNCTION + texts["softmax.h"] + CALL + "\n"
    return texts


def _tree(root: Path, **texts: str) -> Path:
    """The stand-in tree under `root`, with `texts` in place of the files it names (`flash` for `flash.h`)."""
    files = _after_v7()
    assert tuple(sorted(files)) == FILES
    files.update({name + ".h": text for name, text in texts.items()})
    (root / SRC).mkdir(parents=True)
    for name, text in files.items():
        (root / SRC / name).write_text(text)
    return root


def _read(root: Path) -> dict[str, str]:
    return {name: (root / SRC / name).read_text() for name in FILES}


def _apply(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(root)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), str(SRC), str(MINE)])
    runpy.run_path(str(SCRIPT), run_name="__main__")


def _changed(before: str, after: str) -> tuple[list[str], list[str]]:
    """The lines of `before` that `after` lacks, and the lines it has more, each in its file's order."""
    lines = list(difflib.ndiff(before.splitlines(), after.splitlines()))
    return [line[2:] for line in lines if line.startswith("- ")], [line[2:] for line in lines if line.startswith("+ ")]


@pytest.fixture
def programs(monkeypatch: pytest.MonkeyPatch) -> list[tuple[list[str], dict[str, Any]]]:
    """Every program the script starts, recorded and not started."""
    started: list[tuple[list[str], dict[str, Any]]] = []
    monkeypatch.setattr(subprocess, "run", lambda argv, **kwargs: started.append((argv, kwargs)))
    return started


def test_the_script_makes_the_v8_tree_of_a_v7_tree_line_by_line_and_starts_no_program(
        programs: list[Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """A tree that has the tap header is not patched again. The header becomes this directory's; in the other four files
    the script takes out and puts in the lines written here and no other; and it says each edit, then `APPLY-V8-OK`."""
    root = _tree(tmp_path / "tree")
    before = _read(root)

    _apply(root, monkeypatch)

    after = _read(root)
    assert programs == []
    assert capsys.readouterr().out.splitlines() == [*SAID, f"APPLY-V8-OK {root / SRC}"]
    assert after["verity_tap.h"] == (MINE / "verity_tap.h").read_text() != before["verity_tap.h"]

    gone, new = _changed(before["flash.h"], after["flash.h"])
    assert gone == [] and new == [TILE]
    assert f"{TAG}\n{TILE}\n}};\n" in after["flash.h"]

    gone, new = _changed(before["flash_fwd_kernel.h"], after["flash_fwd_kernel.h"])
    assert gone == [CONTEXT, ROWSTAT]
    assert new == [TEMPLATED, ROWSTAT + " \\", "    VERITY_MAT_MS(vmat, softmax, n_block)"]

    gone, new = _changed(before["flash_fwd_launch_template.h"], after["flash_fwd_launch_template.h"])
    assert gone == [ARMED] and new == ARMED_BY_TILE

    gone, new = _changed(before["softmax.h"], after["softmax.h"])
    assert gone == [CALL, CALL]
    function, rest = new[:27], new[27:]
    assert rest == [MS_ON, MAX_SCALED, "#endif", MACRO_CALL, MACRO_CALL]
    assert hashlib.sha512("\n".join(function).encode()).hexdigest() == MS_FUNCTION
    assert [line for line in function if line.startswith("#")] == [
        MS_ON,
        "#define VERITY_SCALE_APPLY_EXP2(t_, m_, s_) FLASH_NAMESPACE::scale_apply_exp2_ms(t_, m_, s_, tap_max_scaled)",
        "#else",
        "#define VERITY_SCALE_APPLY_EXP2(t_, m_, s_) FLASH_NAMESPACE::scale_apply_exp2(t_, m_, s_)",
        "#endif",
    ]
    assert after["softmax.h"].startswith("\n".join(function) + "\n" + BEFORE_THE_FUNCTION)
    assert f"{MEMBER}\n{MS_ON}\n{MAX_SCALED}\n#endif\n" in after["softmax.h"]


def test_a_second_run_changes_no_byte_and_says_only_that_it_is_done(
        programs: list[Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Each edit stands behind a mark of its own in the file it edits, so the script may be run on its own result."""
    root = _tree(tmp_path / "tree")
    _apply(root, monkeypatch)
    once = _read(root)
    capsys.readouterr()

    _apply(root, monkeypatch)

    assert _read(root) == once and programs == []
    assert capsys.readouterr().out == f"APPLY-V8-OK {root / SRC}\n"


def test_a_tree_without_the_tap_header_gets_the_v7_patch_by_git_at_its_root_first(
        programs: list[Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """The one program the script starts: `git apply` of this directory's patch, three directories above the source
    directory, and a failure of it is the script's (`check=True`). It is said before any edit. The recorder applies
    nothing: the tree is the patched one already, less the header."""
    root = _tree(tmp_path / "tree")
    (root / SRC / "verity_tap.h").unlink()

    _apply(root, monkeypatch)

    assert programs == [(["git", "apply", str(MINE / "verity_tap_v7.patch")], {"cwd": str(root), "check": True})]
    assert capsys.readouterr().out.splitlines() == [
        "[apply_v8] verity_tap_v7.patch applied", *SAID, f"APPLY-V8-OK {root / SRC}"]
    assert (root / SRC / "verity_tap.h").read_text() == (MINE / "verity_tap.h").read_text()


@pytest.mark.parametrize(("name", "old", "new", "refusal", "said"), [
    ("flash", TAG, TAG.replace("v7:", "v6:"), "flash.h: Verity_tap.tag line not found", 0),
    ("flash_fwd_kernel", CONTEXT, CONTEXT.replace("kNWarps", "kNWarps_"),
     "flash_fwd_kernel.h: mode-3 VERITY_TAP_INIT not found", 1),
    ("flash_fwd_launch_template", ARMED, ARMED.replace("== 128", "== 64"),
     "flash_fwd_launch_template.h: mode-3 arm condition not found", 2),
    ("softmax", MEMBER, MEMBER + "\n" + MEMBER, "softmax.h: Softmax row_max/row_sum member not found", 3),
    ("softmax", BEFORE_THE_FUNCTION, BEFORE_THE_FUNCTION.replace("zero_init=true", "zero_init=false"),
     "softmax.h: max_scale_exp2_sum anchor not found", 3),
    ("softmax", CALL + REDUCED, CALL.replace("softmax_scale_log2", "softmax_scale") + REDUCED,
     "softmax.h: 1 scale_apply_exp2 calls in softmax_rescale_o (expected 2)", 3),
    ("flash_fwd_kernel", ROWSTAT, ROWSTAT + "\n" + ROWSTAT, "flash_fwd_kernel.h: mode-3 VERITY_TAP_P rowstat line not found", 4),
], ids=["no-tag-line", "no-context-line", "no-arming-line", "two-members", "no-function-after", "one-call", "two-rowstat-lines"])
def test_the_script_refuses_a_tree_that_lacks_a_line_it_edits_and_finishes_once_the_line_is_back(
        name: str, old: str, new: str, refusal: str, said: int, programs: list[Any], tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Each refusal is an `AssertionError` that names the file and the line. The refused edit writes nothing, the edits
    before it stand and are said, and `APPLY-V8-OK` is not. With the line back, a second run makes the tree that a run
    on a whole tree makes."""
    text = _after_v7()[name + ".h"]
    assert text.count(old) == 1
    root = _tree(tmp_path / "tree", **{name: text.replace(old, new)})
    lacking = (root / SRC / (name + ".h")).read_text()

    with pytest.raises(AssertionError) as refused:
        _apply(root, monkeypatch)

    assert str(refused.value) == refusal
    assert capsys.readouterr().out.splitlines() == SAID[:said]
    kept = (root / SRC / (name + ".h")).read_text()
    assert kept.replace(TEMPLATED, CONTEXT) == lacking  # the kernel file's first edit stands when its second is refused

    (root / SRC / (name + ".h")).write_text(kept.replace(new, old))
    _apply(root, monkeypatch)
    whole = _tree(tmp_path / "whole")
    _apply(whole, monkeypatch)
    assert _read(root) == _read(whole) and programs == []
