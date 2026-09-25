"""Every challenge draw takes its seed from its caller: no sample over committed or snapshotted values is drawn from a default seed, and a
command line that draws one refuses to run without `--seed`."""
from __future__ import annotations

import importlib
import inspect

import pytest

SEEDED = [("verity_vllm.check.replay.replay", "tier_a"), ("verity_vllm.check.replay.replay", "tier_chain"),
          ("verity_vllm.program.kernels.relations", "draw_sample"), ("verity_vllm.program.kernels.relations", "ensure_adjacent_pair"),
          ("verity_vllm.check.replay.sampled_replay", "sampled_replay"), ("verity_vllm.check.replay.compiled_kernel_check", "CompiledKernelCheck"),
          ("verity_vllm.correspondence.capture_identities", "run")]


@pytest.mark.parametrize("module, name", SEEDED)
def test_the_seed_has_no_default(module, name):
    if module.endswith("compiled_kernel_check"):
        pytest.importorskip("torch")
    fn = getattr(importlib.import_module(module), name)
    assert inspect.signature(fn).parameters["seed"].default is inspect.Parameter.empty


@pytest.mark.parametrize("tier", ["a", "chain"])
def test_replay_refuses_a_sampled_tier_without_a_seed(tier, tmp_path, capsys):
    from verity_vllm.pipeline import cli

    with pytest.raises(SystemExit) as e:
        cli.main(["replay", "--log", str(tmp_path / "log.jsonl.gz"), "--profile", "unused", "--tier", tier, "--out", str(tmp_path / "out.json")])
    assert e.value.code == 2 and "--seed is required" in capsys.readouterr().err and not (tmp_path / "out.json").exists()


def test_capture_identities_refuses_to_sample_coordinates_without_a_seed(tmp_path, capsys):
    from verity_vllm.pipeline import cli

    with pytest.raises(SystemExit) as e:
        cli.main(["capture-identities", "--commit", str(tmp_path), "--checkpoint", str(tmp_path / "x.safetensors"), "--request", "r0", "--positions", "0",
                 "--out", str(tmp_path / "corr.json")])
    assert e.value.code == 2 and "--seed" in capsys.readouterr().err and not (tmp_path / "corr.json").exists()
