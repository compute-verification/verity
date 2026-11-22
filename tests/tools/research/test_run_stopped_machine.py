"""`research run --on NAME` for a machine the register marks stopped (`stopped_since`): refused before anything is shipped or
started, since a launch there would start the machine again. `--on auto` never reads such a machine."""
from __future__ import annotations

import pytest

from research import auto, cli, remote
from research.pods import registry

NAME = "vy-nebius-cpu-9"
STAMP = "2026-11-18T21:17:28Z"
ENTRY = 'provider = "ssh"\nhost = "stopped.invalid"\nproject = "verity"\n'


@pytest.fixture
def site(tmp_path, monkeypatch):
    reg = tmp_path / "notes" / "machines.d"
    reg.mkdir(parents=True)
    monkeypatch.setenv("RESEARCH_MACHINES", str(tmp_path / "machines.toml"))
    monkeypatch.setenv("RESEARCH_MACHINES_D", str(reg))
    monkeypatch.setenv("RESEARCH_NOTES", str(reg.parent))
    (reg / f"{NAME}.toml").write_text(ENTRY)
    launched = []

    def launch(**kw):
        launched.append(kw)
        raise remote.Refused("reached")

    monkeypatch.setattr(remote, "launch", launch)
    return {"reg": reg, "launched": launched, "runs": tmp_path / "runs"}


def _run(site, *flags: str) -> int:
    return cli.main(["run", "--runs-dir", str(site["runs"]), "--on", NAME, "--project", "verity", *flags, "--", "true"])


def test_a_run_sent_by_name_at_a_machine_marked_stopped_is_refused_and_starts_nothing(site, capsys):
    registry.stopped(NAME, machines_toml=remote.machines_path())
    stamp = registry.recorded(NAME, remote.machines_path())["stopped_since"]
    assert _run(site) == 2
    assert capsys.readouterr().err.strip() == (
        f"research run: refused: {NAME} is marked stopped since {stamp} (the register): it was stopped on purpose, and a run sent at it "
        f"by name would start it. Send the run with --on auto, or, if you mean to start it, run `research pods stopped {NAME} --back` "
        "first.")
    assert site["launched"] == [] and not (site["runs"].exists() and any(site["runs"].iterdir()))


def test_a_machine_without_the_mark_launches_as_before_and_after_back_it_launches_again(site):
    assert _run(site) == 3 and len(site["launched"]) == 1
    (site["reg"] / f"{NAME}.toml").write_text(f'stopped_since = "{STAMP}"\n{ENTRY}')
    assert _run(site) == 2 and len(site["launched"]) == 1
    registry.stopped(NAME, back=True, machines_toml=remote.machines_path())
    assert "stopped_since" not in (site["reg"] / f"{NAME}.toml").read_text()
    assert _run(site) == 3 and len(site["launched"]) == 2, "after --back the launch goes"


def test_a_mark_that_is_no_time_leaves_the_machine_launchable(site):
    (site["reg"] / f"{NAME}.toml").write_text(f'stopped_since = "soon"\n{ENTRY}')
    assert _run(site) == 3 and len(site["launched"]) == 1


def test_on_auto_passes_over_a_machine_marked_stopped_where_it_reads_the_register():
    machines = {NAME: {"projects": ["verity"], "stopped_since": STAMP},
                "vy-nebius-cpu-8": {"projects": ["verity"]}}
    assert auto.spill(machines, now=0) == (("vy-nebius-cpu-8", auto.SPILL_POOL),)
