"""A served row's pinned compile artifact (`acquire.served_artifact`): its digest is over the kernels' launched configs and the autotune
winners without their timing; seeding writes the winners into a fresh Inductor dir where torch's loader takes them; any other kernel
config or winner is a named difference; a Build without an artifact is refused; `served-pin check` says whether a dir's winners are
the pin's."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from verity_vllm.acquire import served_artifact as SA
from verity_vllm.config import UsageError

WIN = {"pp/9eef.best_config": {"XBLOCK": 2, "R0_BLOCK": 128, "num_warps": 2, "num_stages": 1, "configs_hash": "d17b",
                               "found_by_coordesc": False, "triton_cache_hash": "7BMP"},
       "xd/c4a1.best_config": {"XBLOCK": 1, "R0_BLOCK": 4096, "num_warps": 16, "num_stages": 1, "configs_hash": "ba27",
                               "found_by_coordesc": False, "triton_cache_hash": "HGZX"}}
KERNELS = {"aa11": {"template": "rms_norm", "config": {"XBLOCK": 2, "R0_BLOCK": 128, "num_warps": 2, "num_stages": 1}},
           "bb22": {"template": "silu_mul", "config": {"XBLOCK": 1024, "num_warps": 4, "num_stages": 1}}}


def _dir(root: Path, winners: dict, timing: int = 280) -> Path:
    for rel, cfg in winners.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(json.dumps({**cfg, "time_taken_ms": timing}))
    return root


def test_the_digest_is_over_configs_and_winners_not_timing(tmp_path: Path) -> None:
    a = SA.record(str(_dir(tmp_path / "a", WIN, 280)), KERNELS)
    b = SA.record(str(_dir(tmp_path / "b", WIN, 1923)), KERNELS)
    assert a == b and len(a["digest"]) == 128
    assert a["winners"] == WIN and a["kernels"] == {k: v["config"] for k, v in KERNELS.items()}
    other = {**WIN, "pp/9eef.best_config": {**WIN["pp/9eef.best_config"], "R0_BLOCK": 64}}
    c = SA.record(str(_dir(tmp_path / "c", other)), KERNELS)
    d = SA.record(str(tmp_path / "a"), {**KERNELS, "bb22": {**KERNELS["bb22"], "config": {"XBLOCK": 512, "num_warps": 4, "num_stages": 1}}})
    assert len({a["digest"], c["digest"], d["digest"]}) == 3
    named = lambda diff: [x.split(":")[0] for x in diff]
    assert named(SA.differences(a, c)) == ["winners pp/9eef.best_config"] and "loaded {'R0_BLOCK': 64" in SA.differences(a, c)[0]
    assert named(SA.differences(a, d)) == ["kernels bb22"]
    absent = SA.differences(a, {})
    assert named(absent) == [f"kernels {k}" for k in KERNELS] + [f"winners {k}" for k in WIN] and all(x.endswith("loaded absent") for x in absent)


def test_a_seeded_dir_holds_the_pinned_winners_torch_loads(tmp_path: Path) -> None:
    from torch._inductor.runtime.autotune_cache import _load_cached_autotuning

    art = SA.record(str(_dir(tmp_path / "match", WIN)), KERNELS)
    assert SA.seed(art, str(tmp_path / "commit")) == 2
    assert SA.winners(str(tmp_path / "commit")) == WIN
    assert SA.record(str(tmp_path / "commit"), KERNELS) == art
    seeded = json.loads((tmp_path / "commit" / "xd/c4a1.best_config").read_text())
    cands = [SimpleNamespace(kwargs={"XBLOCK": x, "R0_BLOCK": r}, num_warps=w, num_stages=1) for x, r, w in ((2, 128, 2), (1, 4096, 16))]
    assert _load_cached_autotuning(seeded, "ba27", cands, {}) is cands[1]
    with pytest.raises(UsageError, match="already holds autotune winners"):
        SA.seed(art, str(tmp_path / "commit"))


def test_a_build_without_an_artifact_is_refused() -> None:
    with pytest.raises(UsageError, match="records no compile artifact"):
        SA.pinned({"program_digest": "ab" * 64})
    with pytest.raises(UsageError, match="records no compile artifact"):
        SA.pinned({"artifact": {"digest": "x", "kernels": {}}})


def test_served_pin_seeds_and_checks_a_dir(tmp_path: Path, capsys) -> None:
    build = tmp_path / "build.json"
    build.write_text(json.dumps({"program_digest": "p", "artifact": SA.record(str(_dir(tmp_path / "m", WIN)), KERNELS)}))
    seed = SA.SeedOptions(build=str(build), inductor_dir=str(tmp_path / "c"))
    check = SA.CheckOptions(build=str(build), inductor_dir=str(tmp_path / "c"))
    assert SA.main(seed) == 0 and SA.main(check) == 0
    (tmp_path / "c" / "zz").mkdir()
    (tmp_path / "c" / "zz" / "new.best_config").write_text(json.dumps({"XBLOCK": 8, "num_warps": 4, "num_stages": 1}))
    assert SA.main(check) == 3
    assert "winners zz/new.best_config: pinned absent" in capsys.readouterr().out
