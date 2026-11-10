"""A row's setup and precheck past the tree's own checks (`Row.setup`, `prepare`, `run`): its stages' environment, the cuBLAS
names, the release line, the target-family preflight, the Commit's session checks. A row's stages run with the run store's tool
(`tools/research/src`) on their path, so these run only where it is beside the tree; `test_row` holds a tree without it to the
refusal at setup, by name."""
from __future__ import annotations

import dataclasses
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

from tests.pipeline.test_cublas_names import _node
from tests.pipeline.test_row import GLM53, ROW, TP_DENSE, TP_MOE, _row, _tp_row
from verity_vllm.pipeline import cli
from verity_vllm.pipeline.row_driver import Stop

TREE = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------------------------------------------- the stages' environment

@pytest.mark.parametrize("row, execution, bi", [(ROW, "bi-eager", "1"), ("glm53__fp8__h200__tp8__b1__i256__o32__mixed__greedy__eager", "eager", "0")])
def test_a_rows_stages_run_batch_invariant_exactly_when_its_execution_is(tmp_path, monkeypatch, row, execution, bi):
    monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)
    r = _row(tmp_path, row=row)
    r.setup({})
    assert (r.execution(), r.child_env["VLLM_BATCH_INVARIANT"], r.bi_eager()) == (execution, bi, execution == "bi-eager")


def test_remote_verifier_reaches_the_single_gpu_commit_which_checks_its_session_before_anything_runs(tmp_path):
    """REMOTE_VERIFIER reaches the single-GPU Commit as ``--remote-verifier``, and a TP row refuses it; the Commit
    refuses a bad address, a session of more than one mode or a Commit with no instrumented run to audit before anything
    runs (`store.remote.CommitSession`)."""
    from verity_vllm.pipeline import commit

    single = _row(tmp_path, environ={"REMOTE_VERIFIER": "10.0.0.7:7781"})
    single.hidden_so, single.B = "/x/verity_fa2_matReqG.so", 1
    cmd = single.commit_cmd([])
    assert cmd[cmd.index("--remote-verifier") + 1] == "10.0.0.7:7781"
    assert "--remote-verifier" not in _tp_row(tmp_path, TP_DENSE).commit_cmd([])
    with pytest.raises(SystemExit, match="REMOTE_VERIFIER"):
        _tp_row(tmp_path, TP_DENSE, environ={"REMOTE_VERIFIER": "10.0.0.7:7781"}).commit_cmd([])
    args = cli.parse("commit", cmd[cmd.index("commit") + 1:], {})
    with pytest.raises(ValueError, match="one mode"):
        commit.main(dataclasses.replace(args, remote_plan="serve,servedrop:1"))
    with pytest.raises(ValueError, match="HOST:PORT"):
        commit.main(dataclasses.replace(args, remote_verifier="7781"))
    with pytest.raises(SystemExit, match="first instrumented run"):
        commit.main(dataclasses.replace(args, only_arm="control"))


@pytest.mark.parametrize("row, p2p", [
    ("glm53__fp8__h200__tp8__b1__i256__o32__mixed__greedy__eager", None),   # an H box: its engine.env names none
    (TP_DENSE, "1"),                                                          # an L40S row's engine.env names it
    (TP_MOE, "1"),
])
def test_a_tp_rows_stages_have_nccl_p2p_disable_only_when_its_workload_requires_it(tmp_path, monkeypatch, row, p2p):
    """The caller's NCCL_P2P_DISABLE=1 (an L40S operator's shell) reaches the stages only when the row's workload requires it."""
    t = _tp_row(tmp_path, row, environ={"NCCL_P2P_DISABLE": "1"})

    def precheck_reached() -> str:
        raise Stop(99)
    monkeypatch.setattr(t, "hidden_build", precheck_reached)
    with pytest.raises(Stop) as e:
        t.prepare()
    assert e.value.rc == 99 and t.nccl == p2p
    os.makedirs(t.d, exist_ok=True)
    stage = [sys.executable, "-c", "import os; print('NCCL_P2P_DISABLE=' + os.environ.get('NCCL_P2P_DISABLE', 'unset'))"]
    assert t.run_to(0, f"{t.d}/stage.log", stage) == 0
    assert open(f"{t.d}/stage.log").read().splitlines()[-1] == f"NCCL_P2P_DISABLE={p2p or 'unset'}"


def test_a_single_and_a_tp_rows_folds_store_the_instances_form_of_record(tmp_path, monkeypatch):
    """Every child of either row (a TP row's per-rank folds under tp-fold-match among them) stores instances.jsonl in `--instances-form`,
    progressions by default, not the fold library's run-by-run default (a transposed weight's operand is ~1e6 runs per instance)."""
    def precheck_reached(*a, **kw):
        raise Stop(99)
    for argv, form in (((), "progressions"), (("--instances-form", "runs"), "runs")):
        single, tp = _row(tmp_path, *argv), _tp_row(tmp_path, GLM53, *argv)
        monkeypatch.setattr(single, "release_line", precheck_reached)
        monkeypatch.setattr(tp, "hidden_build", precheck_reached)
        for r in (single, tp):
            with pytest.raises(Stop):
                r.prepare()
            assert r.child_env["VERITY_INSTANCES_FORM"] == form


def test_a_tensor_parallel_row_declaring_prefix_caching_is_refused_before_any_stage(tmp_path, monkeypatch):
    """A TP row's stages serve with prefix caching off and its rank Programs take no cached prefix, so a TP row whose file declares it is
    refused at its precheck; a single-GPU prefix row passes it (its warm requests' Programs read their cached rows as an input)."""
    tp = _tp_row(tmp_path, "glm53__fp8__h200__tp8__b8__i256__o32__prefix-arrivals__greedy__eager")
    monkeypatch.setattr(tp, "hidden_build", lambda: "/x/verity_fa2_matReqG.so")
    monkeypatch.setattr(tp, "release_line", lambda *a: pytest.fail("a stage after the precheck"))
    with pytest.raises(Stop) as e:
        tp.prepare()
    last = open(f"{tp.d}/stages.txt").read().splitlines()[-1].split(" ", 3)
    assert e.value.rc == 3 and last[:2] == ["precheck", "FAIL"] and "enable_prefix_caching" in last[3]
    single = _row(tmp_path, row="llama32-1b__bf16__h200__tp1__b2__i168__o10__prefix__greedy__bi-eager")

    def past_precheck(*a):
        raise Stop(99)
    monkeypatch.setattr(single, "release_line", past_precheck)
    with pytest.raises(Stop) as e:
        single.prepare()
    assert e.value.rc == 99


@pytest.mark.parametrize("argv, reads", [
    (("--stages", "build,match", "--match-phase", "cpu"), False),   # a declared-target Build and the CPU half: a CPU node runs them
    (("--stages", "match", "--match-phase", "gpu"), True),
    (("--stages", "match"), True),                                  # the GPU stages and the CPU half in one run
    (("--stages", "commit"), True),
])
def test_a_tp_rows_target_precheck_reads_the_device_only_when_a_stage_runs_on_it(tmp_path, monkeypatch, argv, reads):
    t = _tp_row(tmp_path, GLM53, *argv)
    seen: list[tuple[str, ...]] = []

    def target_family(*extra, out):
        seen.append(extra)
        raise Stop(99)
    for name, fn in (("hidden_build", lambda: "/x/verity_fa2_matReqG.so"), ("release_line", lambda arg: ("PASS|x", None)),
                     ("export_commit", lambda: ""), ("source_identity", lambda pys, out: False), ("target_family", target_family)):
        monkeypatch.setattr(t, name, fn)
    with pytest.raises(Stop) as e:
        t.prepare()
    assert e.value.rc == 99 and seen == [() if reads else ("--no-device",)]


# ---------------------------------------------------------------------------------------------------------------- cuBLAS names

def test_a_rows_stages_get_the_names(tmp_path, monkeypatch, capsys):
    from tests.pipeline.test_row import _row
    py, _, lib = _node(tmp_path)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)
    r = _row(tmp_path, "--py", py)
    r.setup({})
    d = r.child_env["LD_LIBRARY_PATH"].split(":")[0]
    assert os.path.dirname(d) == str(tmp_path) and os.path.realpath(os.path.join(d, "libcublas.so")) == os.path.realpath(lib / "libcublas.so.12")
    shutil.copy(lib / "libcublasLt.so.12", lib / "libcublasLt.so.13")
    with pytest.raises(Stop) as e:
        _row(tmp_path, "--py", py).setup({})
    assert e.value.rc == 3 and "PRECHECK_FAIL_CUBLAS_NAMES" in capsys.readouterr().err


# ---------------------------------------------------------------------------------------------------------------- the release line

def test_the_row_reads_the_release_through_the_module(tmp_path, monkeypatch):
    """The row's release line: from `release-json line` (tree-local first), the record USED is the one copied into the evidence, and a
    released tree takes the release's collector build dir unless the caller set NATIVE_COLLECT_BUILD."""
    from tests.pipeline.test_row import _row
    rel = tmp_path / "RELEASE.json"
    rel.write_text(json.dumps({"tag": "r16-int-3", "build_dirs": {"native_collect": str(tmp_path / "nc_rel")}}))

    def prepared(*flags, environ=None):
        r = _row(tmp_path / "sweep", "--native-collect-default", str(tmp_path / "nc_default"), *flags, environ=environ)
        monkeypatch.setattr(r, "release_line", lambda fallback: (f"r16-int-3|commit=a2dabe493 tree={r.tree}", str(rel)))
        for step in ("precheck_identity", "precheck_target", "admission", "install_signal_forwarding"):
            monkeypatch.setattr(r, step, lambda: None)

        def shape():
            r.B = r.I = r.O = r.LP = r.T = 1
            r.ck = r.tgt = r.hidden_so = r.code9 = ""
        monkeypatch.setattr(r, "shape", shape)
        r.prepare()
        return r
    r = prepared()
    assert json.loads((Path(r.d) / "release.json").read_text()) == json.loads(rel.read_text())
    assert r.ncb == r.child_env["NATIVE_COLLECT_BUILD"] == str(tmp_path / "nc_rel")
    assert json.loads((Path(r.d) / "tree.json").read_text())["release_line"].startswith("r16-int-3|commit=a2dabe493 ")
    for pinned in (prepared("--native-collect-build", str(tmp_path / "nc_own")),
                   prepared(environ={"NATIVE_COLLECT_BUILD": str(tmp_path / "nc_own")})):
        assert pinned.ncb == pinned.child_env["NATIVE_COLLECT_BUILD"] == str(tmp_path / "nc_own")


# ---------------------------------------------------------------------------------------------------------------- the target-family preflight

def test_the_row_runs_the_preflight_before_build_and_canary_waives_by_name(tmp_path, monkeypatch):
    from tests.pipeline.test_row import _row
    r = _row(tmp_path, "--stages", "build,match")
    seen = []
    for step in ("precheck_identity", "shape", "admission", "build", "match"):
        monkeypatch.setattr(r, step, lambda step=step: seen.append(step))
    monkeypatch.setattr(r, "release_line", lambda fallback: (f"UNRELEASED|commit=unknown tree={r.tree}{fallback}", "-"))
    monkeypatch.setattr(r, "target_family", lambda *a, out: seen.append(("target_family", r.stages, a)) or
                        (3, "PRECHECK_FAIL_TARGET_FAMILY: Build target 9.0 != row family 8.9"))
    assert r.run() == 3
    assert seen == ["precheck_identity", "shape", ("target_family", "build,match", ())]
    assert "precheck-target FAIL" in (tmp_path / r.row / "stages.txt").read_text()
    can = (TREE / "verity_vllm" / "ops" / "canary.sh").read_text()
    assert 'TARGET_FAMILY_WAIVER=${TARGET_FAMILY_WAIVER:-"canary $TAG"}' in can
