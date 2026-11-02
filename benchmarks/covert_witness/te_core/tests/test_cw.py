"""Script mode: a covert_witness-shaped train.py runs unmodified under the harness (cw.py), and the counter recognizes θ's
weights by value after a recast."""
import gc
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import safetensors.torch
import torch
from torch.overrides import TorchFunctionMode

from te_core import run as te_run
from te_core.flops import Counter, plain

FAKE = Path(__file__).parent / "cw_fake"


def fake_root(root: Path, monkeypatch) -> Path:
    """A held-out set under `root`. A run's audit hook outlives it, so two runs in one process need two roots."""
    monkeypatch.setenv("CW_FAKE_ROOT", str(root / "root"))
    for split in ("search", "final"):
        d = root / "root" / "eval" / split
        d.mkdir(parents=True)
        torch.save(torch.randint(0, 32, (4, 9), generator=torch.Generator().manual_seed(7)), d / "rows.pt")
    return root


@pytest.fixture
def cw_root(tmp_path, monkeypatch):
    return fake_root(tmp_path, monkeypatch)


def run_island(tmp_path, *island_args, harness=()) -> tuple[int, dict]:
    out = tmp_path / "out"
    rc = te_run.main(["--script", str(FAKE / "islands" / "sgd" / "train.py"), "--out", str(out), *harness, "--",
                      "--start", "fake", "--out", str(tmp_path / "sub"), *island_args])
    return rc, json.loads((out / "run.json").read_text())


SCORE = """import json, sys
sys.path[:0] = sys.argv[1:3]
import prepare
from te_core.cw import score_run
out = sys.argv[3]
rec = json.load(open(out + "/run.json"))
start = {"name": "fake", "digest": "fake", "config": {"vocab": prepare.VOCAB, "ctx": prepare.EVAL_CTX}}
rec = score_run(prepare, rec, out, prepare.theta_of(), start, "cpu")
print(json.dumps({"rec": rec, "sealed": prepare.SEALED_KEYS}))
"""


def score(out: Path) -> tuple[dict, list[str]]:
    """The evaluation job, a process of its own (the training job's audit hook isn't in it)."""
    r = subprocess.run([sys.executable, "-c", SCORE, str(FAKE), str(Path(te_run.__file__).parents[1]), str(out)],
                       capture_output=True, text=True, check=True)
    got = json.loads(r.stdout.strip().splitlines()[-1])
    return got["rec"], got["sealed"]


def test_an_island_runs_unmodified_and_its_encodings_are_scored_by_the_evaluation_job(cw_root):
    rc, rec = run_island(cw_root, "--steps", "8")
    assert rc == 0 and rec["invalid"] == []
    assert len(rec["steps"]) == 8 and all(s["tokens"] == 4 * 8 for s in rec["steps"])
    assert rec["setup"]["tokens"] == 0 and rec["tail"]["tokens"] == 0
    # the fp32 recast of θ is θ: step 0's forward reads θ's values, every later one the updated weights
    assert "fwd/updated" not in rec["steps"][0]["by"] and rec["steps"][0]["by"]["fwd/theta"] > 0
    assert rec["steps"][1]["by"]["fwd/updated"] > 0
    # the buffer multiplies like θ: it is the architecture's, not a weight
    assert not any(k.startswith("buffer") for k in rec["new_weights"])
    cps = rec["checkpoints"]
    assert [cp["step"] for cp in cps] == [1, 3, 5, 7]
    # the training job scores nothing: each checkpoint is its E, written for the evaluation job
    assert all(cp["source"] == "encode" and cp["val_bpb"] is None and "val_bpb" not in rec["start"] for cp in cps)
    assert all(cp["format"] == "full" for cp in cps)
    assert all((cw_root / "out" / cp["path"]).is_file() for cp in cps)
    assert cps[-1]["submitted"] and rec["submitted"]["e_sha256"] == cps[-1]["e_sha256"]
    assert cps[0]["macs"] < cps[-1]["macs"] == rec["totals"]["macs"]
    assert rec["c"] == 4096 * rec["forward_macs_per_token"]
    rec, sealed = score(cw_root / "out")
    cps = rec["checkpoints"]
    assert rec["scored"]["problems"] == [] and all(math.isfinite(cp["val_bpb"]) for cp in cps)
    assert all(cp["rebuild_macs"] == 0 for cp in cps)
    assert rec["start"]["val_bpb"] > cps[-1]["val_bpb"]
    # each scored model's value is sealed under its output digest, and its score names that digest
    assert {cp["output_sha256"] for cp in cps} <= set(sealed)
    assert all(cp["score"]["slice"]["weights_digest"] == cp["output_sha256"] for cp in cps)


def test_an_e_file_that_is_not_the_recorded_one_is_not_scored(cw_root):
    rc, rec = run_island(cw_root, "--steps", "4")
    assert rc == 0
    (cw_root / "out" / rec["checkpoints"][0]["path"]).write_bytes(b"CWFULL1\nnot the encoding")
    rec, _ = score(cw_root / "out")
    assert rec["checkpoints"][0]["val_bpb"] is None and len(rec["scored"]["problems"]) == 1
    assert all(cp["val_bpb"] is not None for cp in rec["checkpoints"][1:])


def test_a_sealed_split_is_never_returned(cw_root):
    rc, rec = run_island(cw_root, "--steps", "4", harness=("--tier", "2"))
    assert rc == 0 and rec["invalid"] == []
    assert rec["checkpoints"] and all("test_bpb" not in cp for cp in rec["checkpoints"])


@pytest.mark.parametrize("flags", [("--peek",), ("--peek-sealed",),
                                   *(("--peek-with", w) for w in ("from_file", "storage", "load_file", "safe_open"))],
                         ids=lambda f: f[-1])
def test_the_held_out_and_sealed_sets_are_unreadable_to_the_island(cw_root, flags):
    """Also through the loaders that open their file in C++ or Rust (`cw.c_readers`), which raise no audit event."""
    (cw_root / "root" / "sealed").mkdir()
    safetensors.torch.save_file({"w": torch.zeros(2)}, cw_root / "root" / "eval" / "search" / "w.safetensors")
    rc, rec = run_island(cw_root, "--steps", "2", *flags)
    assert rc == 1 and "PermissionError" in rec["error"] and "held out" in rec["error"]
    assert any("PermissionError" in p for p in rec["invalid"])


@pytest.mark.parametrize("how, path, refused", [
    ("read", "runs/r0/out/E.bin", True), ("move", "runs/r0/out/E.bin", True),
    ("read", "root/candidates/e100kb.b/sub/E.bin", True),
    ("read", "runs/r1/inputs/E.bin", False), ("read", "root/candidates/e100kb.a/sub/E.bin", False)])
def test_a_run_reads_its_own_outputs_and_no_other_runs(cw_root, monkeypatch, how, path, refused):
    """An earlier run's E (beside this run's `$RESEARCH_RUN_DIR`) or another candidate's scratch would hand this run
    compute it never did; its own run directory and its own scratch (its families' parent) stay readable."""
    for rel in ("runs/r0/out/E.bin", "root/candidates/e100kb.b/sub/E.bin", "runs/r1/inputs/E.bin",
                "root/candidates/e100kb.a/sub/E.bin"):
        (cw_root / rel).parent.mkdir(parents=True, exist_ok=True)
        (cw_root / rel).write_bytes(b"E")
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(cw_root / "runs" / "r1"))
    act = ("--peek-path", str(cw_root / path)) if how == "read" else ("--move", str(cw_root / path),
                                                                      str(cw_root / "runs" / "r1" / "E.bin"))
    families = cw_root / "root" / "candidates" / "e100kb.a" / "families"
    rc, rec = run_island(cw_root, "--steps", "2", *act, harness=("--families", str(families)))
    if refused:
        assert rc == 1 and "another run's" in rec["error"] and (cw_root / path).exists()
    else:
        assert rc == 0 and rec["invalid"] == []


def test_a_narrowed_heads_saving_is_read_op_by_op():
    """The 13:42Z floor rule on a sampled head (V, d = 1000, 16; 64 token rows a call): the head's slice and its input
    gradient are credited; the full head, a narrow adapter, a block matrix and a weight gradient over the token rows
    aren't; the credit stops at the analytic head."""
    from te_core.cw import accounted_over, narrowed_head

    V, d, R = 1000, 16, 64
    ops = {f"fwd theta mm [[{R}, {d}], [{d}, 100]]": [10, 10 * R * d * 100],  # the sampled head: 100 of V rows
           f"bwd grad mm [[{R}, 100], [100, {d}]]": [10, 10 * R * 100 * d],  # its input gradient
           f"fwd theta mm [[{R}, {d}], [{d}, {V}]]": [1, R * d * V],  # the whole head
           f"fwd updated mm [[{R}, {d}], [{d}, 4]]": [10, 10 * R * d * 4],  # a rank-4 adapter
           f"fwd theta mm [[{R}, {d}], [{d}, 64]]": [10, 10 * R * d * 64],  # a block matrix (64, 16)
           f"bwd grad mm [[64, {R}], [{R}, {d}]]": [10, 10 * 64 * R * d]}  # its weight gradient
    got = narrowed_head(ops, (V, d), {(64, d)}, {"train": 10 * R})
    assert sorted(c["head_rows"] for c in got["credited"]) == [100, 100]
    assert got["saved_by_phase"] == {"fwd": 10 * R * d * 900, "bwd": 10 * R * d * 900}
    assert got["saved_macs"] == 2 * 10 * R * d * 900
    capped = narrowed_head(ops, (V, d), {(64, d)}, {"train": R})  # one call's tokens: the head's own work is the cap
    assert capped["saved_macs"] == 2 * R * V * d
    assert accounted_over(48.0, 100.0, 13) == 48.0 / 74.0 and accounted_over(1.0, 10.0, 5) is None


def test_a_recast_copy_of_theta_is_theta_and_a_lossy_one_is_not():
    torch.manual_seed(0)
    theta = {"w": torch.randn(8, 8), "emb": torch.randn(10, 8)}
    c = Counter(theta, digests=False)
    w = torch.nn.Parameter(theta["w"].clone())
    emb = theta["emb"].clone()  # θ's table: its lookups compute nothing on the token
    c.register({"w": w, "emb": emb})
    idx = torch.tensor([[1, 2, 3]])
    with c:
        c.begin_step()
        w.data = w.data.double()
        torch.nn.functional.embedding(idx, emb).double() @ w.T
        lossy = theta["w"].bfloat16()  # a cast of θ inside a step is computed from θ; carried, it is a weight
        c.end_step()
        torch.nn.functional.embedding(idx, emb).bfloat16() @ lossy.T
        c.end_step()
        with torch.no_grad():
            w.add_(1e-12)
        torch.nn.functional.embedding(idx, emb).double() @ w.T
        c.end_step()
    by = [s.by for s in c.steps]
    assert "fwd/updated" not in by[0] and by[0]["fwd/theta"] == 3 * 8 * 8
    assert by[1]["fwd/updated"] == 3 * 8 * 8
    assert by[2]["fwd/updated"] == 3 * 8 * 8


def test_the_screen_reads_names_not_substrings():
    assert te_run.screen("t0 = time.perf_counter()\nfwd = torch.compile(net)\nfrom te_core import hooks\n") == []
    assert te_run.screen("model.eval()\npat = re.compile(r'x')\nempty = torch.empty(3)\n") == []
    prose = '"""Runs under te_core.run --script."""\n\n\ndef f():\n    """No threading here."""\n    return 1  # nor subprocess\n'
    assert te_run.screen(prose) == [] and te_run.screen(prose + "x = 'subprocess'\n") == ["subprocess"]
    for bad in ("te.counter.paused = 1", "from te_core import flops", "import te_core.run", "torch._dynamo.reset()",
                "with uncounted():", "sys.modules['x']", "import threading", "from concurrent import futures",
                "os.fork()", "os.popen('ls')", "torch.library.custom_op", "exec('x = 1')", "eval ('1')",
                "compile(src, 'f', 'exec')", "import builtins", "torch._C.TensorBase.numpy(t)",
                "torch._C._VariableFunctions.from_file(p)", "from torch import nn, _C", "torch.UntypedStorage.__mro__[1]",
                "torch.from_file.__closure__[0].cell_contents", "import pyarrow.parquet as pq", "from datasets import Dataset",
                "import runpy", "runpy.run_path(p, run_name='__main__')", "run = exec", "f = eval", "g = compile",
                "import marshal", "pickle.loads(b)", "import dill", "types.FunctionType(code, {})", "f.__code__",
                "sys.meta_path.append(Finder())", "torch.load(p, weights_only=False)"):
        assert te_run.screen(bad), bad
    assert te_run.screen("datasets = {}\nfrom torch import nn\ntorch.from_file(p)\n") == []
    assert te_run.screen("mode = 'eval'\nnet.eval()\nfwd = torch.compile(net)\nsd = torch.load(p, weights_only=True)\n") == []


def run_probe(tmp_path, monkeypatch, variant: str) -> dict:
    out = fake_root(tmp_path / variant, monkeypatch) / "out"
    te_run.main(["--script", str(FAKE / "islands" / "probe" / "train.py"), "--out", str(out), "--",
                 "--start", "fake", "--out", str(out / "sub"), "--variant", variant])
    return json.loads((out / "run.json").read_text())


def test_the_programs_code_in_what_it_hands_the_encoder_is_counted(tmp_path, monkeypatch):
    base, hidden = run_probe(tmp_path, monkeypatch, "baseline"), run_probe(tmp_path, monkeypatch, "dict_encoder")
    assert base["invalid"] == [] and hidden["invalid"] == []
    (b,), (h,) = base["steps"], hidden["steps"]
    assert h["tokens"] == b["tokens"] == 4 * 8
    assert h["by"]["fwd/updated"] - b["by"]["fwd/updated"] == 50 * 4 * 8 * 16 * 16


def test_a_program_that_rebinds_decode_changes_only_its_own_decoding(tmp_path, monkeypatch):
    """The checkpoint is the E the program encoded, and the evaluation job decodes it with its own prepare."""
    run_probe(tmp_path, monkeypatch, "rebind_decode")
    rec, _ = score(tmp_path / "rebind_decode" / "out")
    (cp,) = rec["checkpoints"]
    assert rec["scored"]["problems"] == [] and cp["val_bpb"] == rec["start"]["val_bpb"]


@pytest.mark.parametrize("variant,name", [("rebind_torch", "torch.nn.functional.cross_entropy"),
                                          ("rebind_coin", "os.urandom")])
def test_rebinding_what_the_harness_calls_makes_the_run_invalid(tmp_path, monkeypatch, variant, name):
    rec = run_probe(tmp_path, monkeypatch, variant)
    assert any(name in p for p in rec["invalid"]), rec["invalid"]


def test_the_watch_covers_the_fixed_modules_prepare_imports(tmp_path, monkeypatch):
    """A module beside prepare.py that it imports (covert_witness's `sealed`) is the harness's too."""
    from types import SimpleNamespace

    from te_core import cw

    (tmp_path / "fixed_sibling.py").write_text("VALUES = 'values'\n")
    (tmp_path / "prepare.py").write_text("from fixed_sibling import VALUES\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        own = cw.private_copy(SimpleNamespace(__file__=str(tmp_path / "prepare.py")))
        watched = {(name, k) for name, _, k, _ in cw.watched(own)}
    finally:
        sys.modules.pop("fixed_sibling", None)
    assert ("fixed_sibling", "VALUES") in watched and ("os", "urandom") in watched


def test_writing_the_theta_load_start_returned_does_not_move_the_harness_theta(tmp_path, monkeypatch):
    base, moved = run_probe(tmp_path, monkeypatch, "baseline"), run_probe(tmp_path, monkeypatch, "theta_write")
    assert moved["invalid"] == [] and moved["steps"][0]["by"] == base["steps"][0]["by"]
    assert moved["theta_sha256"] == base["theta_sha256"]


def test_the_stream_batch_is_token_data_without_an_embedding(tmp_path, monkeypatch):
    rec = run_probe(tmp_path, monkeypatch, "index_lookup")
    assert rec["invalid"] == [], rec["invalid"]
    (s,) = rec["steps"]
    assert s["tokens"] == 4 * 8 and s["by"]["fwd/updated"] == 4 * 8 * 16 * 16


@pytest.mark.parametrize(("variant", "why"), [
    ("custom_op", "outside torch's own"), ("bilinear", "no formula for"), ("numpy_mm", "read out of torch"),
    ("thread", "started threads or processes"),
    pytest.param("fork", "started threads or processes",
                 marks=pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")),
    ("sibling", "modules the program imported"), ("written", "written during the run"),
])
def test_a_route_past_the_counter_makes_the_run_invalid(tmp_path, monkeypatch, variant, why):
    rec = run_probe(tmp_path, monkeypatch, variant)
    assert any(why in p for p in rec["invalid"]), rec["invalid"]


@pytest.mark.parametrize("variant", ["div_mm", "add_exp"])
def test_a_matmul_written_without_a_multiplication_is_counted_as_one(tmp_path, monkeypatch, variant):
    rec = run_probe(tmp_path, monkeypatch, variant)
    assert rec["invalid"] == [], rec["invalid"]
    (s,) = rec["steps"]
    assert s["tokens"] == 4 * 8 and s["by"]["fwd/updated"] == 4 * 8 * 16 * 16


def test_a_statistic_per_sequence_may_leave_torch(tmp_path, monkeypatch):
    rec = run_probe(tmp_path, monkeypatch, "numpy_stat")
    assert rec["invalid"] == [], rec["invalid"]
    (s,) = rec["steps"]
    assert s["exited"] == 4 + 4 < s["tokens"]


def test_the_fixed_prepare_above_an_island_wins_and_its_modules_are_imported_first(tmp_path, monkeypatch):
    from te_core import cw

    (tmp_path / "prepare.py").write_text("def load():\n    import te_fixed_lazy\n    return te_fixed_lazy.WHO\n")
    (tmp_path / "te_fixed_lazy.py").write_text("WHO = 'fixed'\n")
    island = tmp_path / "islands" / "x"
    island.mkdir(parents=True)
    (island / "prepare.py").write_text("def load():\n    return 'island'\n")
    (island / "te_fixed_lazy.py").write_text("WHO = 'island'\n")
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.delitem(sys.modules, "te_fixed_lazy", raising=False)
    monkeypatch.setitem(sys.modules, "prepare", None)
    prepare = cw.import_prepare(island / "train.py")
    sys.path.insert(0, str(island))
    assert Path(prepare.__file__) == tmp_path / "prepare.py" and prepare.load() == "fixed"
    assert Path(sys.modules["te_fixed_lazy"].__file__) == tmp_path / "te_fixed_lazy.py"


def test_a_module_imported_while_the_program_runs_runs_from_its_source(tmp_path, monkeypatch):
    """A bytecode cache whose header matches its source's size and mtime but not its text (one an island could ship) is
    what Python would run; while the program runs it is never read, so the source the screen reads is what runs."""
    import py_compile
    from importlib.util import cache_from_source

    src = tmp_path / "te_stale.py"
    src.write_text("WHO = 'pyc'\n")
    st = src.stat()
    py_compile.compile(str(src), cfile=cache_from_source(str(src)))
    src.write_text("WHO = 'src'\n")
    os.utime(src, ns=(st.st_atime_ns, st.st_mtime_ns))
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "te_stale", raising=False)
    import te_stale

    assert te_stale.WHO == "pyc"
    del sys.modules["te_stale"]
    with te_run.no_bytecode():
        import te_stale
    del sys.modules["te_stale"]
    assert te_stale.WHO == "src"


def test_the_harness_work_imports_from_the_path_it_began_with(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    te = te_run.Session({}, tmp_path)
    sys.path.insert(0, str(tmp_path / "program"))
    with te.uncounted():
        inside = list(sys.path)
    assert str(tmp_path / "program") not in inside and sys.path[0] == str(tmp_path / "program")


def test_plain_copies_containers_and_refuses_what_could_run_code():
    class Sub(dict):
        pass

    class T(torch.Tensor):
        pass

    t = torch.randn(3)
    out = plain(Sub(a=t, b=[1, (2.0, "x")], c=np.float32(1.5), d=Path("/x"), p=torch.nn.Parameter(t), e=bytearray(b"1")))
    assert type(out) is dict and type(out["a"]) is torch.Tensor and type(out["p"]) is torch.Tensor
    assert out["b"] == [1, (2.0, "x")] and type(out["c"]) is float and out["d"] == Path("/x") and out["e"] == b"1"
    for bad in (t.as_subclass(T), object(), {"k": lambda: 0}):
        with pytest.raises(TypeError):
            plain(bad)


def test_the_harness_work_runs_no_mode_hook_or_profiler_of_the_programs(tmp_path):
    te = te_run.Session({}, tmp_path)
    seen = []

    class Mode(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            seen.append("mode")
            return func(*args, **(kwargs or {}))

    def profiler(*args):
        return None

    lin = torch.nn.Linear(2, 2)
    hook = torch.nn.modules.module.register_module_forward_hook(lambda m, i, o: seen.append("hook"))
    sys.setprofile(profiler)
    try:
        with Mode():
            with te.uncounted():
                n, collecting, profiling = len(seen), gc.isenabled(), sys.getprofile()
                lin(torch.ones(2))
                inside = seen[n:]
            lin(torch.ones(2))
        restored = sys.getprofile() is profiler
    finally:
        sys.setprofile(None)
        hook.remove()
    assert inside == [] and not collecting and profiling is None and restored and gc.isenabled()
    assert "mode" in seen and "hook" in seen


def test_the_harness_work_builds_on_the_device_it_names_and_not_the_programs(tmp_path):
    te = te_run.Session({}, tmp_path)
    seen = []

    class Mode(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            seen.append(func)
            return func(*args, **(kwargs or {}))

    with torch.device("meta"), Mode():
        with te.uncounted():
            plain_device = torch.empty(1).device.type
            with torch.device("meta"):
                named = torch.empty(1).device.type, torch.nn.Linear(2, 2).weight.device.type
            inside = list(seen)
        after = torch.empty(1).device.type
    assert plain_device == "cpu" and named == ("meta", "meta") and inside == [] and after == "meta" and seen
    assert torch._C._len_torch_function_stack() == 0 and torch.empty(1).device.type == "cpu"



def run_mezo(root, *island_args) -> tuple[int, Path, dict]:
    out = root / "out"
    rc = te_run.main(["--script", str(FAKE / "islands" / "mezo" / "train.py"), "--out", str(out), "--",
                      "--start", "fake", "--out", str(root / "sub"), *island_args])
    return rc, out, json.loads((out / "run.json").read_text())


def test_declared_steps_are_a_family_rebuilt_by_prepares_decoder(cw_root):
    """Each step's two perturbed reads are credited through `prepare.declare_step` (cw.CWPlan)."""
    rc, _, rec = run_mezo(cw_root)
    assert rc == 0 and rec["invalid"] == [], rec["invalid"]
    assert rec["digest_key"] and all(len(ds) == 2 for ds in rec["steps"][2]["weights"].values())
    (fam,) = rec["families"]
    assert fam["family"] == "E" and fam["failed"] == {} and fam["covers"] == list(range(6))
    assert all(b > 0 for b in fam["beta"].values())


def test_a_declared_steps_cost_is_at_least_prepares_rebuild_rule():
    from types import SimpleNamespace

    from te_core.cw import CWPlan, frame

    theta = {"w": torch.zeros(2)}
    priced = CWPlan(SimpleNamespace(rebuild_macs=lambda th, blob: 10 * len(blob)), theta, "cpu")
    assert priced.cost_macs(frame(b"abc", (b"de", b"f")), {}) == 60
    assert priced.cost_macs(frame(b"", (b"de",)), {}) == 20
    assert CWPlan(SimpleNamespace(), theta, "cpu").cost_macs(frame(b"abc", ()), {}) == 0


def test_a_step_that_reads_what_it_did_not_declare_fails_its_rebuild(cw_root):
    import sys

    from te_core.counterexample import check
    from te_core.cw import CWPlan

    rc, out, rec = run_mezo(cw_root, "--bad-step", "3")
    prepare = sys.modules["prepare"]
    theta = prepare.theta_of()
    v = check(out / "families" / "E", rec, theta, rec["c"], n_sample=6, coin=b"x", plan=CWPlan(prepare, theta, "cpu"))
    assert list(v.failed) == [3] and v.covers == [0, 1, 2, 4, 5]
