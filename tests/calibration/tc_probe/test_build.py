from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

TC_PROBE = Path(__file__).resolve().parents[3] / "calibration" / "tc_probe"
if str(TC_PROBE) not in sys.path:
    sys.path.insert(0, str(TC_PROBE))
if "tc_probe" not in sys.modules:
    spec = importlib.util.spec_from_file_location("tc_probe", TC_PROBE / "tc_probe.py")
    sys.modules["tc_probe"] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sys.modules["tc_probe"])
tc_probe = sys.modules["tc_probe"]

H200, RTX_PRO, B200 = "NVIDIA H200", "NVIDIA RTX PRO 6000 Blackwell Server Edition", "NVIDIA B200"


def nvcc_line(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tu: str, card: str) -> str:
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        return SimpleNamespace(stdout="nvcc: NVIDIA (R) Cuda compiler driver\nCuda compilation tools, release 12.8\nBuild cuda_12.8\n")

    monkeypatch.setattr(tc_probe.subprocess, "run", run)
    monkeypatch.setattr(tc_probe, "cuda_bin", lambda tool: "nvcc")
    tc_probe.build(tu, card, tmp_path)
    return " ".join(calls[-1])


@pytest.mark.parametrize("tu, card, line", [
    ("mma_tiles.cu", H200, "nvcc -O2 -arch=sm_90a -Xcompiler -fPIC -shared -o {out}/libmma_tiles_sm_90a.so {here}/mma_tiles.cu"),
    ("wgmma_tiles.cu", H200, "nvcc -O2 -arch=sm_90a -Xcompiler -fPIC -shared -o {out}/libwgmma_tiles_sm_90a.so {here}/wgmma_tiles.cu"),
    ("mma_tiles.cu", RTX_PRO, "nvcc -O2 -arch=sm_120a -Xcompiler -fPIC -shared -o {out}/libmma_tiles_sm_120a.so {here}/mma_tiles.cu"),
    ("tcgen05_tiles.cu", B200, "nvcc -O2 -gencode arch=compute_100a,code=sm_100a -Xcompiler -fPIC -shared -o "
                               "{out}/libtcgen05_tiles_sm_100a.so {here}/tcgen05_tiles.cu")])
def test_the_card_the_run_opened_gives_the_compiler_the_line_the_table_gave_it(monkeypatch, tmp_path, tu, card, line) -> None:
    assert nvcc_line(monkeypatch, tmp_path, tu, card) == line.format(out=tmp_path, here=tc_probe.HERE)


def test_an_instruction_of_another_architecture_is_refused_by_name() -> None:
    assert tc_probe.card_target(H200, ["sm90.wgmma.m64n8k32.e4m3", "sm90.mma.m16n8k16.bf16"]) == "sm_90a"
    with pytest.raises(SystemExit, match=r"the card 'NVIDIA H200' is sm_90a, and sm120\.mma\.m16n8k32\.e4m3 is sm_120a"):
        tc_probe.card_target(H200, ["sm120.mma.m16n8k32.e4m3"])


def test_a_card_no_instruction_is_for_and_a_host_with_no_gpu_are_refused_by_name(monkeypatch) -> None:
    from verity_bench import kit
    with pytest.raises(SystemExit, match="no GPU here: --card names the card"):
        tc_probe.card_target("none", ["sm90.mma.m16n8k16.bf16"])
    monkeypatch.setattr(kit, "card_arch", lambda name, **k: "sm_103a")
    with pytest.raises(SystemExit, match="no instruction is for the card 'NVIDIA B300' \\(sm_103a\\)"):
        tc_probe.card_target("NVIDIA B300", ["sm100.mma.m16n8k16.bf16"])


@pytest.mark.parametrize("argv, why", [(["--out", "x"], "--instruction is required"),
                                       (["--out", "x", "--instruction", "sm90.mma.m16n8k16.bf16", "--card", H200],
                                        "--card names the card of a --compile-only build")])
def test_the_probe_names_no_architecture_of_its_own(monkeypatch, tmp_path, capsys, argv, why) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["tc_probe.py", *argv])
    with pytest.raises(SystemExit):
        tc_probe.main()
    assert why in capsys.readouterr().err
