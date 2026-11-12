"""pearl_c_sm90's arm for `verity bench overhead` (`overhead_arm.py`) off the device: the harness's interface and its verifier
declaration, the cubin it loads, its domain and work at the door's linears, its ε, and the door's names for it
resolving here as the door resolves them; and its y commitment (`_YTree`) on the host twin, where its tree is
`audit.commit_rows`'s, and on a recording stand-in, where its launches are the served call's and it allocates nothing and
calls no torch."""

from __future__ import annotations

import ast
import ctypes
import hashlib
import importlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

POUW = Path(__file__).resolve().parents[1]
ROOT = POUW.parents[1]
sys.path.insert(0, str(POUW / "harness"))

import arm as A  # noqa: E402
import baselines as B  # noqa: E402
from verity.protocols.pouw.verifier import audit  # noqa: E402
from verity.protocols.pouw.verifier import window as Wd  # noqa: E402
from verity.protocols.pouw.verifier.protocol import Shape as Unit  # noqa: E402
from verity.protocols.spot.verifier.profile import WINDOW_DELTA  # noqa: E402
from verity_vllm.protocols.pouw import pearl_c as PPD  # noqa: E402
from verity_vllm.protocols.pouw.pearl_c import served_y as SY  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = sys.modules[spec.name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


OA = _load("pearl_c_sm90_overhead_arm", POUW / "pearl_c_sm90" / "overhead_arm.py")
DOOR = _load("verity_bench_overhead_door", ROOT / "benchmarks" / "harness" / "verity_bench" / "overhead.py")
#: the host twin's build of hash_host.cpp (`host`, a module fixture) and its `Dev`
HH = _load("pearl_c_hash_host_test", POUW / "tests" / "test_pearl_c_hash_host.py")
host = HH.host
LINEARS = DOOR.MODELS[DOOR.KERNEL_MODEL]
CTX = {"device": None, "arch": "sm_90a"}


def _built(tmp_path: Path, monkeypatch, body: bytes = b"\x7fELF a sent cubin") -> tuple[OA.PearlPlus, dict, Path]:
    cubin = tmp_path / "inputs" / "pearl_c_sm90.cubin"
    cubin.parent.mkdir(parents=True)
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
    assert rec["cubin"] == {"path": str(cubin), "sha256": hashlib.sha256(body).hexdigest(), "source": "sent", "art": None}
    assert (rec["scheme"], rec["served"], rec["schedule"], rec["index"]) == ("pearl-plus", "pearl-c-h100-v1-h3", "serial", 0)
    v = A.check_verifier(arm.name, rec)
    assert v["per"] == "shape" and v["transcript"] == "transcripts/{arm}/{shape}/manifest.json"
    assert v["control"] == "negative/{arm}/{shape}/manifest.json"
    [cmd] = v["commands"]
    assert cmd[:2] == ["{python}", "benchmarks/pouw/pearl_c_vllm/verify_run.py"] and (ROOT / cmd[1]).is_file()
    assert cmd[2:] == ["{transcript}", "--coins", "{dir}/coins.json", "--commit", "{commit}"]


def test_the_cubin_of_record_is_one_art_by_its_bytes():
    pin = OA.pinned()
    assert set(pin) == {"art", "sha256", "bytes"}
    assert re.fullmatch(r"art:[0-9a-f]{64}", pin["art"]) and re.fullmatch(r"[0-9a-f]{64}", pin["sha256"])
    assert type(pin["bytes"]) is int and pin["bytes"] > 0


def test_the_record_names_the_art_of_record_only_for_its_bytes(tmp_path, monkeypatch):
    body = b"\x7fELF the cubin of record"
    art = "art:" + "ab" * 32
    monkeypatch.setattr(OA, "pinned", lambda: {"art": art, "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body)})
    assert _built(tmp_path / "a", monkeypatch, body)[1]["cubin"]["art"] == art
    assert _built(tmp_path / "b", monkeypatch, body + b"\0")[1]["cubin"]["art"] is None


def test_the_record_keeps_the_path_the_door_sent(tmp_path, monkeypatch):
    (tmp_path / "store.cubin").write_bytes(b"cubin")
    (tmp_path / "pearl_c_sm90.cubin").symlink_to(tmp_path / "store.cubin")
    assert OA.cubin_of(str(tmp_path / "pearl_c_sm90.cubin"), tmp_path) == (tmp_path / "pearl_c_sm90.cubin", "sent")
    monkeypatch.chdir(tmp_path)
    assert OA.cubin_of("pearl_c_sm90.cubin", tmp_path) == (tmp_path / "pearl_c_sm90.cubin", "sent")


def test_a_sent_cubin_that_is_no_file_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError, match="isn't a file"):
        OA.cubin_of(str(tmp_path / "absent.cubin"), tmp_path)


@pytest.mark.parametrize("sent", [None, ""])
def test_without_a_sent_cubin_the_arm_builds_one_for_development(tmp_path, monkeypatch, sent):
    made = []

    def build(out: Path) -> Path:
        made.append(out)
        return out / "pearl_c_sm90.cubin"

    assert OA.cubin_of(sent, tmp_path, build) == (tmp_path / "pearl-plus-cubin" / "pearl_c_sm90.cubin", "development")
    assert made == [tmp_path / "pearl-plus-cubin"]
    monkeypatch.delenv(OA.CUBIN_ENV, raising=False)
    assert OA.settings().VERITY_BENCH_CUBIN is None


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
    (16, 8192, 8192 + 64, "y's row digests take rows of whole 1 KiB segments"),
    (16, 8192, 1024 * 256 + 256, "at most 1024 (n a multiple of 256"),
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
    assert w.hashed_bytes == {"a_rows": 4 * m16 * k, "tile_leaves": 8 * m16 * n, "y_rows": 4 * m * n}
    assert w.weight_hashed_bytes == {"b_rows": 4 * n * k} and w.gamma is None


def test_the_door_s_entry_says_the_call_commits_y_only_once_the_call_hashes_y_s_rows():
    """The door notes every figure of a scheme whose entry says its timed call commits no y (`Scheme.commits_y`). The
    arm's call hashes y's rows (`_YTree`), and the entry says so."""
    n, k = LINEARS["o"]
    hashed = OA.PearlPlus().work(A.Shape("o", 16, k, n)).hashed_bytes
    assert DOOR.SCHEMES[OA.SCHEME].commits_y is ("y_rows" in hashed), (
        f"the arm's call hashes {sorted(hashed)}: the door's entry (verity_bench.overhead.SCHEMES) says commits_y only "
        "in the change that makes the call hash y's rows (`y_rows` in work's hashed bytes), and then it must")


@pytest.mark.parametrize("m", [1, 16, 64, 65, 8192])
def test_the_tiles_are_the_served_pipeline_s(m):
    R, (n, k) = OA.kernels(), LINEARS["qkv"]
    assert OA.tiles_of(m, n) * R.NB == R.Pipeline.sizes(R.BM, n, k, OA.m16(m))["leaves"]


def test_the_gate_is_the_call_s_check_as_the_harness_s_verdict():
    """`gate` waits for the device, then hands the call's check to the harness as its `GateResult`, which `check_gate`
    takes."""
    class Dev:
        synced = 0

        def sync(self) -> None:
            self.synced += 1

    class Call:
        outputs = ("y", "a_root")

        def check(self) -> tuple[bool, dict[str, str], dict]:
            return True, {"y": "all", "a_root": "all"}, {"index": OA.INDEX}

    dev, call = Dev(), Call()
    g = OA.PearlPlus().gate(dev, call, A.Shape("o", 16, 64, 64), frozenset({"hash"}))
    assert isinstance(g, A.GateResult) and dev.synced == 1
    assert (g.ok, g.checked, g.detail) == (True, {"y": "all", "a_root": "all"}, {"index": OA.INDEX})
    A.check_gate(g, call)


def test_without_a_window_of_record_no_epsilon_is_invented(tmp_path, monkeypatch):
    pinned = tmp_path / "overhead_arm.json"
    pinned.write_text(json.dumps({"cubin": OA.pinned()}))
    monkeypatch.setattr(OA, "PINNED", pinned)
    with pytest.raises(LookupError, match="PoUW's ε is a window's sample's .* states no window of record"):
        OA.epsilon(8192, *LINEARS["o"], "h200")


def test_the_pinned_window_s_epsilon_says_whether_it_is_provisional_and_names_its_source():
    w = json.loads(OA.PINNED.read_text())["window"]
    got = OA.epsilon(8192, *LINEARS["o"], w["card"])
    assert got["basis"].startswith("provisional, ") == bool(w.get("provisional")) and w["source"] in got["basis"]
    assert 0 < got["epsilon"] < 1


def test_epsilon_is_the_window_of_record_s_at_its_sample_at_every_linear(tmp_path, monkeypatch):
    window = {"basis": "a stated window's sample", "card": "h200", "k": 27_713, "credited": 8 << 30, "work": 9 << 30}
    pinned = tmp_path / "overhead_arm.json"
    pinned.write_text(json.dumps(json.loads(OA.PINNED.read_text()) | {"window": window}))
    monkeypatch.setattr(OA, "PINNED", pinned)
    eps = Wd.eps_at(window["k"], OA.DELTA, window["credited"], window["work"], 1)
    for m in (16, 8192):
        for n, k in LINEARS.values():
            got = OA.epsilon(m, n, k, "h200")
            assert (got["epsilon"], got["delta"]) == (float(eps), 0.01) and 0 < eps < 1
            assert got["basis"].startswith("a stated window's sample: window.eps_at(K=27713, δ=1/100, C=8589934592, "
                                           "W=9663676416")
    tight = OA.epsilon(8192, *LINEARS["o"], "h200", delta=WINDOW_DELTA)
    assert (tight["epsilon"], tight["delta"]) == (float(Wd.eps_at(window["k"], WINDOW_DELTA, window["credited"],
                                                                  window["work"], 1)), float(WINDOW_DELTA))
    assert tight["epsilon"] > got["epsilon"] and "δ=1/1099511627776" in tight["basis"]
    with pytest.raises(ValueError, match="ran on the h200, not the rtx-pro-6000"):
        OA.epsilon(8192, *LINEARS["o"], "rtx-pro-6000")


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
            assert eps(8192, n, k, "h200") == OA.epsilon(8192, n, k, "h200")
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


SALT, SALT_KEY = bytes(range(32)), bytes(range(32, 64))
#: y's shapes held to `audit.commit_rows`: the door's linears at 80 rows (not a power of two, so the pads are written, and
#: one chunk) and at 384 (past s512_tree_small's widest level, so s512_nodes runs, and two chunks at gate_up's width), and
#: the LM head's width at 16 rows (501 segments, the wide row digests)
Y_SHAPES = [(m, n) for m in (80, 384) for n in sorted({n for n, _ in LINEARS.values()})] + [(16, 128256)]


class Twin:
    """The host twin as the arm's `Dev`: hash_host.cpp's kernels (`test_pearl_c_hash_host.Host`), with each row-digest
    launch refused as the device refuses it (`rs_refusal`), and widen_a, which the host build doesn't hold, in numpy:
    each BF16 word's bits shifted up 16, the rows from `rows` to `filled` - 1 zero, over the grid's words only.  Its
    `pd_smem` stands set: a y tree launches nothing of hash_decode.cuh, whose tables pd_setup makes."""

    pd_smem = {"no hash_decode.cuh launch": 0}

    def __init__(self, host, widen: bool = True):
        self.host, self.widen, self.stream, self.launched, self.allocs = host, widen, None, [], 0

    def has(self, name: str) -> bool:
        return True

    def alloc(self, n: int) -> int:
        self.allocs += 1
        return self.host.alloc(n)

    def free(self, p: int) -> None:
        self.host.free(p)

    def sync(self) -> None:
        pass

    def launch(self, name, grid, block, smem=0, p=(), i=(), stream=None, pdl=True) -> None:
        R = OA.kernels()
        self.launched.append(name)
        if name in R.RS_KERNELS and (why := R.rs_refusal(name, i)):
            raise ValueError(why)
        if name != "widen_a":
            return self.host.launch(name, grid, block, smem, p, i)
        rows, k, filled = i[:3]
        assert not pdl and grid * block * 8 >= filled * k
        if self.widen:
            src = np.ctypeslib.as_array((ctypes.c_uint16 * (rows * k)).from_address(p[0]))
            out = np.ctypeslib.as_array((ctypes.c_uint32 * (filled * k)).from_address(p[2]))
            out[:rows * k] = src.astype(np.uint32) << 16
            out[rows * k:] = 0


class Recording:
    """A stand-in `Dev` that records each launch (name, pointers, ints) and runs nothing; its allocations are counted."""

    pd_smem, rs_heads = {"no hash_decode.cuh launch": 0}, True

    def __init__(self, wide: bool = True):
        self.log, self.allocs, self.stream, self.wide = [], 0, None, wide

    def has(self, name: str) -> bool:
        return self.wide or not name.endswith("_w")

    def alloc(self, n: int) -> int:
        self.allocs += 1
        return (self.allocs << 40) + 0x1000

    def free(self, p: int) -> None:
        pass

    def sync(self) -> None:
        pass

    def launch(self, name, grid, block, smem=0, p=(), i=(), stream=None, pdl=True) -> None:
        self.log.append((name, tuple(p), tuple(i)))


def _trees(dev):
    import torch
    s, cpu = PPD.scheme_of(OA.SERVED), torch.device("cpu")
    const = PPD._constants(s, cpu, SALT_KEY)
    run = SimpleNamespace(kernel=SimpleNamespace(run=OA.kernels(), dev=dev), variant=s.device.name, const=const,
                          const_keys=PPD._constant_keys(const), device=cpu, salt=SALT, salt_key=SALT_KEY, cur=None)
    return SY.DeviceTrees(run)


@pytest.fixture(scope="module")
def twin_trees(host):
    """One run's y trees on the host twin, each width's pipeline held to `commit_y` at its first use, as served."""
    return _trees(Twin(host))


def _y(m: int, n: int):
    import torch
    rng = np.random.default_rng(m * 1_000_003 + n)
    return torch.from_numpy(rng.integers(0, 1 << 16, (m, n), dtype=np.uint16).view(np.int16)).view(torch.bfloat16)


def _torch_calls(f) -> list[str]:
    """What of torch `f` calls, Python or C: each function whose module is torch's, and each C method of a torch object."""
    seen: list[str] = []

    def prof(frame, event, arg):
        if event == "call" and str(frame.f_globals.get("__name__", "")).split(".")[0] == "torch":
            seen.append(frame.f_code.co_qualname)
        elif event == "c_call":
            owner = getattr(arg, "__self__", None)
            mod = getattr(arg, "__module__", None) or type(owner).__module__
            if str(mod).split(".")[0] == "torch":
                seen.append(arg.__qualname__)
    sys.setprofile(prof)
    try:
        f()
    finally:
        sys.setprofile(None)
    return seen


def test_the_torch_call_probe_sees_torch_s_python_and_c_calls():
    import torch
    t = torch.zeros(4)
    assert _torch_calls(lambda: t.data_ptr()) and _torch_calls(lambda: torch.zeros(1)) and not _torch_calls(lambda: len([t]))


@pytest.mark.slow
@pytest.mark.parametrize("m, n", Y_SHAPES)
def test_the_arm_s_y_tree_is_commit_rows_s_on_the_host_twin(twin_trees, m, n):
    """The arm's y commitment, `_YTree`'s launches from its buffers made beforehand over y's BF16 rows, writes
    `commit_y`'s whole tree, whose root is `audit.commit_rows`'s over y's FP32 rows under `y_binding` at nonce
    `HM96_Y` + index, the root the verifier opens y's rows under; and its launch allocates nothing and calls no torch."""
    import torch
    yt, y, dev = OA._YTree(twin_trees, m, n, OA.INDEX), _y(m, n), twin_trees.run.kernel.dev
    raw = y.contiguous().view(torch.uint8).numpy().tobytes()
    allocs, at, ptr = dev.allocs, len(dev.launched), y.data_ptr()
    assert _torch_calls(lambda: yt.launch(ptr)) == [] and dev.allocs == allocs
    want = SY.commit_y(raw, n, SALT, OA.INDEX, SALT_KEY)
    assert yt.tree.numpy().tobytes() == want[1]
    rows = audit.commit_rows(SY.y_binding(SALT, OA.INDEX), OA.INDEX, SY.f32_rows(raw, n), audit.HM96, SALT_KEY,
                             SY.HM96_Y + OA.INDEX)
    assert yt.tree.numpy()[-64:].tobytes() == want[0] == rows.commitment.root
    assert dev.launched[at:].count("widen_a") == -(-m // SY.chunk_rows(m, n))


def test_a_y_tree_whose_rows_were_never_widened_is_not_commit_y_s(host):
    """The control on the widening: the same launches with widen_a writing nothing leave the digests over the poison the
    words buffer held, and the tree is not `commit_y`'s."""
    import torch
    trees = _trees(Twin(host))
    m, n = 80, 8192
    trees.pipe(n, OA.INDEX)
    trees.run.kernel.dev.widen = False
    yt, y = OA._YTree(trees, m, n, OA.INDEX), _y(m, n)
    for t in yt.bufs:
        t.fill_(0xA5)
    yt.launch(y.data_ptr())
    raw = y.contiguous().view(torch.uint8).numpy().tobytes()
    assert yt.tree.numpy()[-64:].tobytes() != SY.commit_y(raw, n, SALT, OA.INDEX, SALT_KEY)[0]


@pytest.mark.parametrize("m, n", [(80, 8192), (384, 57344), (16, 128256), (4100, 4096)])
def test_the_arm_s_y_launches_are_the_served_call_s_from_its_own_buffers(m, n):
    """On a recording stand-in, the arm's y launches are the served `DeviceTrees`'s for the same y, the same kernels with
    the same ints, but for its widen_a launches: one a chunk, from y's rows into the words buffer its row digests read
    next; every pointer is into a buffer made before the launch, and the launch allocates nothing and calls no torch."""
    import torch
    dev = Recording()
    trees = _trees(dev)
    trees.pipes[n] = trees._pipe(n)
    yt = OA._YTree(trees, m, n, OA.INDEX)
    y = torch.zeros((m, n), dtype=torch.bfloat16)
    allocs, ptr, dev.log = dev.allocs, y.data_ptr(), []
    assert _torch_calls(lambda: yt.launch(ptr)) == [] and dev.allocs == allocs
    arm, dev.log = dev.log, []
    trees._tree(trees.pipes[n], y, OA.INDEX)
    served = dev.log
    assert [(name, i) for name, _, i in arm if name != "widen_a"] == [(name, i) for name, _, i in served]
    tree, dig, seg, words = yt.at
    rows = SY.chunk_rows(m, n)
    widens = [(j, p, i) for j, (name, p, i) in enumerate(arm) if name == "widen_a"]
    assert [(p, i) for _, p, i in widens] == [((ptr + 2 * n * r, 0, words), (min(rows, m - r), n, min(rows, m - r)))
                                              for r in range(0, m, rows)]
    assert all(arm[j + 1][1][0] == words for j, _, _ in widens)
    spans = [(ptr, 2 * m * n)] + [(t.data_ptr(), t.numel()) for t in [*yt.bufs, trees.dom, trees.run.const]]
    held = set(trees.pipes[n].b.values())
    for name, p, _ in arm:
        for q in p:
            assert q == 0 or q in held or any(a <= q < a + size for a, size in spans), (name, hex(q))


def test_a_cubin_without_the_wide_row_digests_is_refused_where_y_s_rows_need_them():
    """y's rows past 128 segments (gate_up's 57,344 words, the LM head's 128,256) take the wide row-digest kernels: on a
    cubin that lacks them the call is refused at prepare, naming them, and narrower widths still commit y."""
    trees = _trees(Recording(wide=False))
    trees.pipes[8192] = trees._pipe(8192)
    assert OA._YTree(trees, 16, 8192, OA.INDEX).tree.numel()
    for n in (57344, 128256):
        with pytest.raises(RuntimeError, match=f"{n // 256} 1 KiB segments, past 128: .* wide row-digest kernels"):
            OA._YTree(trees, 16, n, OA.INDEX)
