"""SPOT's ``levels`` in an application's file (`served_table.levels`): absent, it reads one and leaves the configuration
and its hash as they were; stated, it reaches the runner as ``SPOT_LEVELS`` and the dry run shows it; two, any other
value, or a ``SPOT_LEVELS`` unlike the file's is refused by name before anything starts, and never run as one level."""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import pytest

from verity import config as K
from verity.channel import Refused
from verity_cli import run as R

ROOT = Path(R.ROOT)
REG = R.registry()
T = sys.modules["served_table"]
SPOT = T.MANIFESTS["sampled-proofs"].id
APPLICATIONS = sorted(T.DEMOS)
#: a component's parameters where it declares none: refuses any, takes none
NO_PARAMS = next(f.default for f in dataclasses.fields(K.Component) if f.name == "params")


def stating(tmp: Path, demo: str, value: object) -> Path:
    f = tmp / f"{demo}.toml"
    f.write_text((ROOT / "applications" / f"{demo}.toml").read_text() + f'\n[params."{SPOT}"]\nlevels = {json.dumps(value)}\n')
    return f


def hooks(capsys, config: Path) -> tuple[int, dict[str, str], str]:
    rc = R.run_main([str(config), "--dry-run", "--party", T.WORKERS], REG)
    out = capsys.readouterr()
    plan = [json.loads(x.split(" ", 1)[1]) for x in out.out.splitlines() if x.startswith("PLAN ")]
    return rc, {k: v for x in plan for e in x["entries"] for k, v in e["hooks"].items()}, out.err


@pytest.fixture(autouse=True)
def no_spot_levels(monkeypatch):
    monkeypatch.delenv(T.SPOT_LEVELS, raising=False)


@pytest.mark.parametrize("demo", APPLICATIONS)
def test_an_application_that_states_no_levels_hashes_as_before_spot_took_parameters(demo):
    """Only a file that states ``levels`` changes its configuration's hash: SPOT's parameters fill in no default."""
    before = {**REG.components, "sampled-proofs": dataclasses.replace(T.MANIFESTS["sampled-proofs"],
                                                                      params=NO_PARAMS)}
    path = ROOT / "applications" / f"{demo}.toml"
    cfg = R.load(path, REG)
    assert cfg["params"][SPOT] == {}
    assert K.digest(cfg) == K.digest(K.check(K.parse(path.read_text()), before, REG.workloads))


def test_a_file_that_states_one_level_hands_the_runner_spot_levels_and_hashes_otherwise(tmp_path, capsys):
    rc, got, err = hooks(capsys, stating(tmp_path, "onsite_full", 1))
    assert rc == 0, err
    assert got[T.SPOT_LEVELS] == "1"
    assert K.digest(R.load(tmp_path / "onsite_full.toml", REG)) != K.digest(R.load(ROOT / "applications" / "onsite_full.toml", REG))
    rc, got, err = hooks(capsys, ROOT / "applications" / "onsite_full.toml")
    assert rc == 0, err
    assert T.SPOT_LEVELS not in got


@pytest.mark.parametrize("demo", ["onsite_full", "remote"])
def test_two_levels_are_refused_by_name_before_anything_starts_and_never_run_as_one(tmp_path, capsys, monkeypatch, demo):
    config = stating(tmp_path, demo, 2)
    rc, _, err = hooks(capsys, config)
    assert rc == 2 and f"refused levels: {T.TWO_LEVELS}" in err
    assert "not run as one level" in T.TWO_LEVELS
    runner = tmp_path / "runner.sh"
    runner.write_text('touch "$OUT.started"\n')
    monkeypatch.setattr(T, "RUNNER", str(runner))
    state = tmp_path / "state"
    rc = R.run_main([str(config), "--on", T.PROVER_NODE, "--party", T.WORKERS, "--state", str(state)], REG)
    assert rc == 2 and f"refused levels: {T.TWO_LEVELS}" in capsys.readouterr().err
    assert not (state / "out.started").exists()
    with pytest.raises(Refused) as e:
        T.live(R.load(config, REG), state, [T.VERIFIER])
    assert (e.value.name, e.value.detail) == ("levels", T.TWO_LEVELS)


@pytest.mark.parametrize("value", [0, 3, "2", True, 1.0])
def test_levels_other_than_one_or_two_are_refused_by_name(tmp_path, value):
    with pytest.raises(Refused) as e:
        R.load(stating(tmp_path, "onsite_full", value), REG)
    assert e.value.name == "levels" and "levels is 1 or 2" in e.value.detail


def test_a_key_spot_does_not_take_is_refused(tmp_path):
    f = tmp_path / "remote.toml"
    f.write_text((ROOT / "applications" / "remote.toml").read_text() + f'\n[params."{SPOT}"]\nstages = 2\n')
    with pytest.raises(Refused) as e:
        R.load(f, REG)
    assert e.value.name == "unknown-key" and "stages" in e.value.detail and "levels" in e.value.detail


@pytest.mark.parametrize("stated, env", [(None, "2"), (1, "2"), (None, "3"), (1, "")])
def test_a_spot_levels_unlike_the_files_is_refused_by_name_before_anything_starts(tmp_path, capsys, monkeypatch,
                                                                                    stated, env):
    """The levels a run audits are the auditor's, the file's: ``SPOT_LEVELS`` only carries them to the runner."""
    config = ROOT / "applications" / "onsite_full.toml" if stated is None else stating(tmp_path, "onsite_full", stated)
    monkeypatch.setenv(T.SPOT_LEVELS, env)
    rc, got, err = hooks(capsys, config)
    if not env:
        assert rc == 0 and got[T.SPOT_LEVELS] == "1", err
        return
    assert rc == 2 and f"refused levels: {T.SPOT_LEVELS}={env} and the application file's levels is 1" in err
    state = tmp_path / "state"
    with pytest.raises(Refused) as e:
        T.live(R.load(config, REG), state, [T.VERIFIER])
    assert e.value.name == "levels" and not state.exists()


def test_a_spot_levels_like_the_files_runs(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv(T.SPOT_LEVELS, "1")
    rc, got, err = hooks(capsys, ROOT / "applications" / "remote.toml")
    assert rc == 0, err
    assert T.SPOT_LEVELS not in got
    assert T.levels(R.load(stating(tmp_path, "remote", 1), REG)) == 1
