"""pearl_c_sm90's arm for `verity bench overhead` (`overhead_arm.py`) off the device: the harness's interface and its verifier
declaration, the cubin it loads, its domain and work at the door's linears, its refused ε, and the door's names for it
resolving here as the door resolves them."""

from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

POUW = Path(__file__).resolve().parents[1]
ROOT = POUW.parents[1]
sys.path.insert(0, str(POUW / "harness"))

import arm as A  # noqa: E402
import baselines as B  # noqa: E402
from verity.core.protocols.pouw.protocol import Shape as Unit  # noqa: E402
from verity_vllm.protocols.pouw import pearl_c as PPD  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = sys.modules[spec.name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


OA = _load("pearl_c_sm90_overhead_arm", POUW / "pearl_c_sm90" / "overhead_arm.py")
DOOR = _load("verity_bench_overhead_door", ROOT / "benchmarks" / "harness" / "verity_bench" / "overhead.py")
LINEARS = DOOR.MODELS[DOOR.KERNEL_MODEL]
CTX = {"device": None, "arch": "sm_90a"}


def _built(tmp_path: Path, monkeypatch, body: bytes = b"\x7fELF a sent cubin") -> tuple[OA.PearlPlus, dict, Path]:
    cubin = tmp_path / "inputs" / "pearl_c_sm90.cubin"
    cubin.parent.mkdir()
    cubin.write_bytes(body)
    monkeypatch.setenv(OA.CUBIN_ENV, str(cubin))
    arm = OA.PearlPlus()
    return arm, arm.build({**CTX, "out_dir": str(tmp_path / "run"), "transcripts_dir": str(tmp_path / "run" / "transcripts")}), cubin


def test_the_arm_meets_the_harness_interface():
    arm = OA.PearlPlus()
    A.check_arm(arm, set(B.FAMILIES))
    assert isinstance(arm, A.BenchArm)
    assert (arm.name, arm.stream, arm.precision, arm.inputs) == ("pearl-plus", "fp8", "fp8-e4m3-rowwise", "bf16")
    assert arm.flag_sets == {"hash": {"hash"}, "nohash": set()}
    assert not hasattr(arm, "known_bad"), "an unpoisoned skipped write leaves correct stale bytes: no gate could catch it"


def test_the_build_record_lists_the_sent_cubin_and_declares_the_verifier(tmp_path, monkeypatch):
    body = b"\x7fELF a sent cubin"
    arm, rec, cubin = _built(tmp_path, monkeypatch, body)
    assert rec["binaries"] == [str(cubin)] and arm.cubin == cubin
    assert rec["cubin"] == {"path": str(cubin), "sha256": hashlib.sha256(body).hexdigest(), "source": "sent"}
    assert (rec["scheme"], rec["served"], rec["schedule"], rec["index"]) == ("pearl-plus", "pearl-c-h100-v1-h3", "serial", 0)
    v = A.check_verifier(arm.name, rec)
    assert v["per"] == "shape" and v["transcript"] == "transcripts/{arm}/{shape}/manifest.json"
    assert v["control"] == "negative/{arm}/{shape}/manifest.json"
    [cmd] = v["commands"]
    assert cmd[:2] == ["{python}", "benchmarks/pouw/pearl_c_vllm/verify_run.py"] and (ROOT / cmd[1]).is_file()
    assert cmd[2:] == ["{transcript}", "--coins", "{dir}/coins.json", "--commit", "{commit}"]


def test_the_record_keeps_the_path_the_door_sent(tmp_path, monkeypatch):
    (tmp_path / "store.cubin").write_bytes(b"cubin")
    (tmp_path / "pearl_c_sm90.cubin").symlink_to(tmp_path / "store.cubin")
    assert OA.cubin_of({OA.CUBIN_ENV: str(tmp_path / "pearl_c_sm90.cubin")}, tmp_path) == (tmp_path / "pearl_c_sm90.cubin", "sent")
    monkeypatch.chdir(tmp_path)
    assert OA.cubin_of({OA.CUBIN_ENV: "pearl_c_sm90.cubin"}, tmp_path) == (tmp_path / "pearl_c_sm90.cubin", "sent")


def test_a_sent_cubin_that_is_no_file_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError, match="isn't a file"):
        OA.cubin_of({OA.CUBIN_ENV: str(tmp_path / "absent.cubin")}, tmp_path)


@pytest.mark.parametrize("env", [{}, {OA.CUBIN_ENV: ""}])
def test_without_a_sent_cubin_the_arm_builds_one_for_development(tmp_path, env):
    made = []

    def build(out: Path) -> Path:
        made.append(out)
        return out / "pearl_c_sm90.cubin"

    assert OA.cubin_of(env, tmp_path, build) == (tmp_path / "pearl-plus-cubin" / "pearl_c_sm90.cubin", "development")
    assert made == [tmp_path / "pearl-plus-cubin"]


def test_the_development_build_is_build_sh_s_cubin_and_gates_alone(tmp_path, monkeypatch):
    runs = []

    def run(argv, env, stdout, stderr):
        runs.append((argv, env["CUBIN_ONLY"]))
        return subprocess.CompletedProcess(argv, len(runs) - 1)

    monkeypatch.setattr(OA.subprocess, "run", run)
    out = tmp_path / "pearl-plus-cubin"
    assert OA.build_dev(out) == out / "pearl_c_sm90.cubin"
    assert runs == [(["bash", str(POUW / "pearl_c_sm90" / "build.sh"), str(out)], "1")]
    with pytest.raises(RuntimeError, match=r"build.sh exited 1 \(see .*pearl-plus-cubin.log\): send the cubin"):
        OA.build_dev(out)


def test_the_arm_builds_only_for_its_cubins_target(tmp_path, monkeypatch):
    monkeypatch.setenv(OA.CUBIN_ENV, str(tmp_path / "absent.cubin"))
    with pytest.raises(RuntimeError, match="runs on sm_90a .* not sm_120a: time it with --card h200"):
        OA.PearlPlus().build({**CTX, "arch": "sm_120a", "out_dir": str(tmp_path)})


@pytest.mark.parametrize("m", [1, 16, 8192, DOOR.M_MAX])
@pytest.mark.parametrize("linear", sorted(LINEARS))
def test_the_door_s_linears_are_in_the_domain(linear, m):
    n, k = LINEARS[linear]
    assert OA.PearlPlus().domain(A.Shape(linear, m, k, n)) is None


@pytest.mark.parametrize("m, k, n, why", [
    (16, 8192 + 64, 8192, "the kernels take k a multiple of 256"),
    (16, 8192, 8192 + 32, "n of 64"),
    (16, 32768 + 256, 8192, "at most 128 1 KiB segments"),
    (PPD.FLAG_ROWS + 1, 8192, 8192, f"at most {PPD.FLAG_ROWS} rows"),
])
def test_outside_the_domain_the_arm_says_why(m, k, n, why):
    got = OA.PearlPlus().domain(A.Shape("x", m, k, n))
    assert got is not None and why in got and not got.startswith("linear 0")


@pytest.mark.parametrize("m, m16", [(1, 16), (16, 16), (17, 32), (8192, 8192)])
def test_work_is_the_served_scheme_s_at_the_padded_rows(m, m16):
    n, k = LINEARS["o"]
    w, s, unit = OA.PearlPlus().work(A.Shape("o", m, k, n)), PPD.scheme_of(OA.SERVED), Unit(m16, k, n)
    assert w.tensor_macs == {"fp8-e4m3-rowwise": m16 * k * n} and w.unit == "fp8-mac"
    assert (w.w_ref, w.credited) == (str(s.wref(unit)), str(s.credit_of(unit)))
    assert w.hashed_bytes == {"a_rows": 4 * m16 * k, "tile_leaves": 8 * m16 * n}
    assert w.weight_hashed_bytes == {"b_rows": 4 * n * k} and w.gamma is None


@pytest.mark.parametrize("m", [1, 16, 64, 65, 8192])
def test_the_tiles_are_the_served_pipeline_s(m):
    R, (n, k) = OA.kernels(), LINEARS["qkv"]
    assert OA.tiles_of(m, n) * R.NB == R.Pipeline.sizes(R.BM, n, k, OA.m16(m))["leaves"]


def test_no_epsilon_at_one_linear_is_invented():
    with pytest.raises(NotImplementedError, match="PoUW's ε is a window's sample's"):
        OA.epsilon(8192, *LINEARS["o"], "h200")


def test_the_door_names_the_arm_and_its_epsilon_here(monkeypatch):
    door = DOOR.SCHEMES[OA.SCHEME]
    assert (ROOT / door.arm_path).resolve() == OA.HERE
    monkeypatch.syspath_prepend(str(ROOT / door.arm_path))
    monkeypatch.delitem(sys.modules, "overhead_arm", raising=False)
    try:
        (mod, cls), (emod, fn) = door.arm.split(":"), door.epsilon.split(":")
        assert isinstance(getattr(importlib.import_module(mod), cls)(), A.BenchArm)
        eps = getattr(importlib.import_module(emod), fn)
        for n, k in LINEARS.values():
            with pytest.raises(NotImplementedError):
                eps(8192, n, k, "h200")
    finally:
        sys.modules.pop("overhead_arm", None)


def test_the_door_s_node_side_imports_the_arm_without_the_harness_or_torch():
    top = set()
    for node in ast.parse((OA.HERE / "overhead_arm.py").read_text()).body:
        if isinstance(node, ast.Import):
            top |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            top.add(node.module.split(".")[0])
    assert top <= set(sys.stdlib_module_names), top
