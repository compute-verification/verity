"""Every challenge draw takes its seed from its caller: no sample over committed or snapshotted values is drawn from a default seed, and a
command line that draws one refuses to run without `--seed`."""
from __future__ import annotations

import importlib
import inspect

import pytest

SEEDED = [("verity_vllm.check.replay.driver", "sampled_replay"), ("verity_vllm.check.replay.compiled_kernel_check", "CompiledKernelCheck"),
          ("verity_vllm.correspondence.capture_identities", "run")]


@pytest.mark.parametrize("module, name", SEEDED)
def test_the_seed_has_no_default(module, name):
    if module.endswith("compiled_kernel_check"):
        pytest.importorskip("torch")
    fn = getattr(importlib.import_module(module), name)
    assert inspect.signature(fn).parameters["seed"].default is inspect.Parameter.empty


def test_capture_identities_refuses_to_sample_coordinates_without_a_seed(tmp_path, capsys):
    from verity_vllm.pipeline import cli

    with pytest.raises(SystemExit) as e:
        cli.main(["capture-identities", "--commit", str(tmp_path), "--checkpoint", str(tmp_path / "x.safetensors"), "--request", "r0", "--positions", "0",
                 "--out", str(tmp_path / "corr.json")])
    assert e.value.code == 2 and "--seed" in capsys.readouterr().err and not (tmp_path / "corr.json").exists()


def test_the_derivations_of_record_are_pinned():
    from verity_vllm.commit import challenge as CH

    root = bytes(range(32))
    assert CH.root_seed(root) == 283686952306183 == int.from_bytes(root[:8], "big")
    assert CH.kernel_check_seed(root) == 66051 == int(root.hex()[:8], 16)
    assert CH.challenge_seed("verifier-challenge-1", root) == 2610256019655222637
    assert CH.challenge_seed("", root) == 11482780518489293129
    assert CH.reference_rows_seed(10, 3) == 79193 and CH.case_seed(5, 2) == 5000017


def test_the_generators_are_the_stdlib_and_numpy_ones_over_the_seed_as_given():
    import random

    import numpy as np

    from verity_vllm.commit import challenge as CH

    for seed in (0, 12345, 2610256019655222637):
        assert CH.stream(seed).sample(range(1000), 20) == random.Random(seed).sample(range(1000), 20)
        assert list(CH.generator(seed).permutation(50)) == list(np.random.default_rng(seed).permutation(50))
    assert list(CH.generator([3, 7]).integers(0, 1 << 30, 8)) == list(np.random.default_rng([3, 7]).integers(0, 1 << 30, 8))


def test_sampled_replay_draws_its_challenge_seed_from_the_challenge_module():
    from verity_vllm.check.replay import replay_codes as RC, sample as SM
    from verity_vllm.commit import challenge as CH

    root = bytes(range(32))
    assert SM.challenge_seed("verifier-challenge-1", root) == (CH.challenge_seed("verifier-challenge-1", root),
                                                               RC.challenge_seed_source("verifier-challenge-1"))
