"""Records a reader fills from the environment, pinned at their bytes on `main` before the reader moves to `verity_vllm.settings`, so
the move can't change them: the hot worker's key (`hot.env_identity`, `hot.so_identity`), the fast Match's declared configuration
(`global_match_fast.config`, on every result document's impl block), the Commit's reasons to run cold with the staging
environment it hands its committers (`commit.hot_unsafe_reasons`, `commit.staging_environ`), and the configuration a committer
takes from that environment (the collector's on `test_staging_defaults`' CPU fakes, its name among it)."""
from __future__ import annotations

import hashlib
import json
import os

import pytest

from verity_vllm.check.match import global_match as GM
from verity_vllm.check.match import global_match_fast as GMF
from verity_vllm.pipeline import cli, hot


def _digest(doc: object) -> str:
    return hashlib.sha512(json.dumps(doc, sort_keys=True).encode()).hexdigest()[:32]


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for k in list(os.environ):
        monkeypatch.delenv(k)
    return monkeypatch


@pytest.fixture(scope="module")
def commit():
    """`pipeline.commit`, imported before `env` clears the environment: the import pins the engine's variables (`engine.env`)."""
    pytest.importorskip("torch")
    from verity_vllm.pipeline import commit

    return commit


@pytest.fixture(scope="module")
def committers():
    pytest.importorskip("torch")
    from tests.commit import test_staging_defaults
    from verity_vllm.commit.committer import leafhash, native_host

    return test_staging_defaults, leafhash, native_host


def test_the_hot_key_takes_the_environment_as_before(env: pytest.MonkeyPatch) -> None:
    for k, v in {"PATH": "/bin", "PYTHONPATH": "/src", "VLLM_BATCH_INVARIANT": "1", "VERITY_WINDOW_MB": "256", "VERITY_TIMELINE": "/t",
                 "VERITY_HOT_SUBMIT": "1", "HASH_THREADS": "4", "CUDA_HOME": "/cuda", "TORCH_X": "", "OMP_NUM_THREADS": "8",
                 "NATIVE_COLLECT_BUILD": "/no-such-build", "HIDDEN_GPU_BUILD": "", "HOME": "/h", "NCCL_P2P_DISABLE": "1"}.items():
        env.setenv(k, v)
    assert _digest([hot.env_identity(), hot.so_identity([])]) == "0019aeeec8c143005b7a8b4b19531a0c"


@pytest.mark.parametrize(("values", "digest"), [
    ({"MATCH_WORKERS": "1"}, "37c433ce7a6f3d0ef54af75f49134685"),
    ({"MATCH_WORKERS": " 3 ", "MATCH_COMPACT": " 0 "}, "aee01761c9e2b4dd6f984669223ef09f"),
    ({"MATCH_WORKERS": "2", "MATCH_COMPACT": "false"}, "07dfbb9661bbc2f54bfcceb284cf36ab"),
    ({"MATCH_WORKERS": "1", "MATCH_COMPACT": ""}, "37c433ce7a6f3d0ef54af75f49134685"),
])
def test_the_fast_match_declares_its_configuration_as_before(env: pytest.MonkeyPatch, values: dict[str, str], digest: str) -> None:
    for k, v in values.items():
        env.setenv(k, v)
    GMF.ops()
    config = {k: v for k, v in GMF.config.items() if k != "cpu_count"}
    assert _digest([config, GM.impl_banner(GMF)]) == digest


@pytest.mark.parametrize(("values", "argv", "digest"), [
    ({}, [], "28285790244bd016269ddf08993ba559"),
    ({"VERITY_FAULT": " skip_copy:1 ", "VERITY_SWITCH": "0.001"}, ["--arms", "a,b", "--window-mb", "0"], "e7f7725d89caf134f7ea831181381df2"),
    ({"VERITY_FAULT": "  ", "VERITY_SWITCH": ""}, ["--bounded-staging", "--retain", "host", "--retain-exclude", "fa3_hidden_m1",
                                                   "--learn-budget-mb", "1024", "--window-mb", "256", "--window-slots", "4"],
     "ce0b116482d3de00d51cf1c1a00a940f"),
    ({"VERITY_FAULT": "", "VERITY_RETAIN": "device"}, ["--no-bounded-staging", "--retain", "none"], "ab1c19f50687282fe9ef2e4012560ec4"),
])
def test_the_commit_names_its_cold_reasons_and_stages_as_before(commit, env: pytest.MonkeyPatch, values: dict[str, str],
                                                                 argv: list[str], digest: str) -> None:
    for k, v in values.items():
        env.setenv(k, v)
    ns = cli.parse("commit", ["--out", "o", *argv], {})
    reasons = commit.hot_unsafe_reasons(ns)
    commit.staging_environ(commit.resolve_staging(ns))
    assert _digest([reasons, {k: v for k, v in os.environ.items() if k != "PYTEST_CURRENT_TEST"}]) == digest


@pytest.mark.parametrize(("retain", "scheme", "values", "digest"), [
    ("none", None, {}, "ff0b0f64489f1687c56406d80f95c9b3"),
    ("host", None, {"VERITY_STAGING_BOUNDED": " yes ", "VERITY_HI_PRIO": "", "VERITY_RETAIN_PINNED": "false", "VERITY_INLINE_D2H": "1",
                    "VERITY_PIN": "true", "VERITY_FAULT": " skip_copy:3", "VERITY_WINDOW_MB": " 16 ", "VERITY_WINDOW_SLOTS": "3",
                    "NATIVE_GATHER_BLOCKS": "64", "NATIVE_GATHER_AUTO_BYTES": "4096", "VERITY_CONSUMER_DELAY_MS": "2.5",
                    "VERITY_LEARN_HOST_BUDGET_MB": "1024", "VERITY_RETAIN_EXCLUDE": "fa3_hidden_m1", "VERITY_SKIP": "a,,b",
                    "VERITY_WEIGHTS_HASH": "device"}, "1eb12e9bd19c559e217418921c4074b6"),
    ("none", None, {"VERITY_STAGING_BOUNDED": "on", "VERITY_FAULT": "skip_copy:2", "VERITY_HI_PRIO": "1", "VERITY_RETAIN_PINNED": " no",
                    "VERITY_INLINE_D2H": "true"}, "cbc856bbf4739c6b70ec3bed50df6bbd"),
    ("none", "vllm-v1", {"VERITY_STAGING_BOUNDED": "  ", "VERITY_WINDOW_MB": "0", "VERITY_FAULT": "",
                         "VERITY_WEIGHTS_HASH": "device-checked"}, "b1a4cae3c13e13d6dce3be0e97c0c589"),
])
def test_a_committer_takes_its_configuration_as_before(committers, env: pytest.MonkeyPatch, retain: str, scheme: str | None,
                                                       values: dict[str, str], digest: str) -> None:
    staging, leafhash, native_host = committers
    com = staging.committer(env, retain, commit_scheme=scheme, **values)
    keys = ("win_bytes", "win_slots_req", "bounded", "learn_budget_bytes", "hi_prio", "consumer_delay_s", "fault", "retain", "retain_pinned",
            "inline_d2h", "pin", "name", "windowed")
    nc = com._nc
    doc = {k: getattr(com, k) for k in keys} | {
        "retain_exclude": sorted(com.retain_exclude), "footprint": com.footprint is not None,
        "blocks_per_desc": getattr(nc, "blocks_per_desc", None), "auto_bytes_per_block": getattr(nc, "auto_bytes_per_block", None),
        "fault_skip_copy_at": getattr(nc, "fault_skip_copy_at", None), "skip": sorted(native_host.skip_classes_of_env()),
        "weights_hash": leafhash.mode_from_env()}
    assert _digest(doc) == digest
