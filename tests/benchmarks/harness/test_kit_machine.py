"""The harness kit's readers of a machine record's `[properties]`, over the records in `fixtures/machines/`, and of the toolkit
and the card's architecture they and the card records give."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from verity_bench import kit

MACHINES = Path(__file__).parent / "fixtures" / "machines"


def test_a_record_read_whole_gives_each_property_with_the_command_that_read_it():
    props = kit.machine_properties("one-card", directory=MACHINES)
    assert sorted(props) == sorted(kit.MACHINE_PROPERTIES)
    assert props["toolkit"]["value"]["directory"] == "/usr/local/cuda-13.0"
    assert props["profiler"] == {"value": "allowed", "read": "grep RmProfilingAdminOnly /proc/driver/nvidia/params"}
    assert props["clock_lock"]["value"] is True
    assert props["numa"]["value"] == [{"card": 0, "node": 0, "cpus": "0-15"}]


def test_a_record_with_unread_values_and_no_register_entry_is_read():
    props = kit.machine_properties("slurm-node", directory=MACHINES)
    assert props["toolkit"]["value"] == "none"
    assert props["profiler"]["value"] == "unread"
    assert props["job_cap_minutes"]["value"] == 10080


def _record(tmp_path: Path, old: str, new: str) -> Path:
    text = (MACHINES / "one-card.toml").read_text()
    assert old in text
    (tmp_path / "m.toml").write_text(text.replace(old, new))
    return tmp_path


def test_a_record_with_no_file_or_no_table_is_refused_by_name(tmp_path):
    with pytest.raises(SystemExit, match=r"^machine 'gone': .*gone\.toml is unreadable \(FileNotFoundError"):
        kit.machine_properties("gone", directory=MACHINES)
    (tmp_path / "m.toml").write_text('provider = "ssh"\nhost = "dummy.invalid"\n')
    with pytest.raises(SystemExit, match=r"m\.toml has no \[properties\] table"):
        kit.machine_properties("m", directory=tmp_path)


def test_a_missing_or_unknown_property_is_refused_by_name(tmp_path):
    d = _record(tmp_path, 'python = { value = "3.12.3", read = "python3 --version" }\n', "")
    with pytest.raises(SystemExit, match=r"lacks python in \[properties\]: give each \{ value, read \}"):
        kit.machine_properties("m", directory=d)
    # a register key written after the table lands in it
    d = _record(tmp_path, 'read = "no job queue" }\n', 'read = "no job queue" }\nunreachable_since = "2026-11-18T00:00:00Z"\n')
    with pytest.raises(SystemExit, match=r"has unreachable_since in \[properties\], which are none of kind, card"):
        kit.machine_properties("m", directory=d)


@pytest.mark.parametrize("old, new, field", [
    ('cards = { value = 1,', 'cards = { value = 0,', "cards"),
    ('cards = { value = 1,', 'cards = { value = true,', "cards"),
    ('cpus = { value = 16,', 'cpus = { value = "16",', "cpus"),
    ('value = "allowed"', 'value = "yes"', "profiler"),
    ('clock_lock = { value = true,', 'clock_lock = { value = "yes",', "clock_lock"),
    ('loader = "lib64", ', "", "toolkit"),
    ('{ directory = "/usr/local/cuda-12.9", version = "12.9" }', '{ directory = "/usr/local/cuda-12.9" }', "toolkit"),
    ('{ card = 0, node = 0, cpus = "0-15" }', '{ card = 0, cpus = "0-15" }', "numa"),
    ('start = "research run --on one-card", ', "", "launcher"),
    ('value = "none", read = "no job queue"', 'value = 0, read = "no job queue"', "job_cap_minutes"),
])
def test_a_value_its_check_refuses_is_refused_naming_the_field(tmp_path, old, new, field):
    d = _record(tmp_path, old, new)
    with pytest.raises(SystemExit, match=rf"gives {field} the value .*, which its check refuses"):
        kit.machine_properties("m", directory=d)


@pytest.mark.parametrize("old, new", [
    (', read = "python3 --version" }', " }"),
    (', read = "python3 --version" }', ', read = "" }'),
    ('python = { value = "3.12.3", read = "python3 --version" }', 'python = "3.12.3"'),
    (', read = "python3 --version" }', ', read = "python3 --version", note = "the venv" }'),
])
def test_an_entry_that_is_not_a_value_and_its_read_is_refused(tmp_path, old, new):
    d = _record(tmp_path, old, new)
    with pytest.raises(SystemExit, match=re.escape("gives python as ") + r".*not \{ value, read \}"):
        kit.machine_properties("m", directory=d)


def _run(tmp_path: Path, monkeypatch, machine: str | None) -> Path:
    """A run directory as `research run --on` leaves it: the copy of one-card's [properties] in inputs/."""
    (tmp_path / "inputs").mkdir()
    table = (MACHINES / "one-card.toml").read_text().split("[properties]", 1)[1]
    (tmp_path / "inputs" / "one-card.toml").write_text(f'machine = "one-card"\n\n[properties]{table}')
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(tmp_path))
    if machine is None:
        monkeypatch.delenv("RESEARCH_MACHINE", raising=False)
    else:
        monkeypatch.setenv("RESEARCH_MACHINE", machine)
    return tmp_path


def test_unnamed_it_reads_the_copy_the_launch_put_in_the_run(tmp_path, monkeypatch):
    _run(tmp_path, monkeypatch, "one-card")
    assert kit.machine_properties() == kit.machine_properties("one-card", directory=MACHINES)


def test_unnamed_it_refuses_a_run_with_no_machine_or_no_copy_by_name(tmp_path, monkeypatch):
    _run(tmp_path, monkeypatch, None)
    with pytest.raises(SystemExit, match=r"^no machine named: RESEARCH_MACHINE is unset"):
        kit.machine_properties()
    monkeypatch.setenv("RESEARCH_MACHINE", "bare")
    with pytest.raises(SystemExit, match=r"^machine 'bare': .*inputs/bare\.toml is not there: .*only where its record has the table"):
        kit.machine_properties()


OTHERS = '[{ directory = "/usr/local/cuda-12.9", version = "12.9" }]'


def _toolkits(run: Path, value: str | None = None) -> None:
    """The run's copy listing CUDA 13.0 as the toolkit's own with 12.9 and 13.3 beside it, or `value`."""
    copy = run / "inputs" / "one-card.toml"
    text = copy.read_text()
    assert OTHERS in text
    text = text.replace(OTHERS, OTHERS[:-1] + ', { directory = "/usr/local/cuda-13.3", version = "13.3" }]')
    if value is not None:
        start = text.index("toolkit = { value = ")
        text = text[:start] + f"toolkit = {{ value = {value}" + text[text.index(', read = "readlink', start):]
    copy.write_text(text)


def test_the_toolkit_is_the_runs_machines_at_the_version_asked_not_the_first_listed(tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_HOME", raising=False)
    _toolkits(_run(tmp_path, monkeypatch, "one-card"))
    assert [kit.toolkit(v) for v in ("13.0", "13.3", "12.9")] == [
        "/usr/local/cuda-13.0", "/usr/local/cuda-13.3", "/usr/local/cuda-12.9"]


def test_cuda_home_where_given_wins_with_no_machine(monkeypatch):
    monkeypatch.delenv("RESEARCH_MACHINE", raising=False)
    monkeypatch.setenv("CUDA_HOME", "/opt/cuda-13.3")
    assert kit.toolkit("13.3") == "/opt/cuda-13.3"


@pytest.mark.parametrize(("says", "found"), [("13.3.0", "home"), ("13.3", "home"), ("13.0.1", "machine"), ("13.30.0", "machine")])
def test_cuda_home_at_another_version_gives_way_to_the_machines(tmp_path, monkeypatch, says, found):
    home = tmp_path / "home"
    home.mkdir()
    (home / "version.json").write_text(f'{{"cuda": {{"name": "CUDA SDK", "version": "{says}"}}}}')
    _toolkits(_run(tmp_path, monkeypatch, "one-card"))
    monkeypatch.setenv("CUDA_HOME", str(home))
    assert kit.toolkit("13.3") == (str(home) if found == "home" else "/usr/local/cuda-13.3")


@pytest.mark.parametrize(("machine", "says"), [
    (None, r"^no CUDA toolkit 13\.0: set CUDA_HOME to its directory, or run on a machine `research run --on` names: "
           r"no machine named"),
    ("bare", r"^no CUDA toolkit 13\.0: .*machine 'bare': .*inputs/bare\.toml is not there"),
])
def test_no_machine_or_no_copy_is_refused_naming_what_to_set(tmp_path, monkeypatch, machine, says):
    monkeypatch.delenv("CUDA_HOME", raising=False)
    _run(tmp_path, monkeypatch, machine)
    with pytest.raises(SystemExit, match=says):
        kit.toolkit("13.0")


def test_an_unread_toolkit_is_refused_naming_the_property_and_the_machine(tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_HOME", raising=False)
    _toolkits(_run(tmp_path, monkeypatch, "one-card"), '"unread"')
    with pytest.raises(SystemExit, match=r"^machine 'one-card': its \[properties\] have toolkit unread: read it there, or set CUDA_HOME"):
        kit.toolkit("13.0")


@pytest.mark.parametrize(("value", "lists"), [(None, "13.0, 12.9, 13.3"), ('"none"', "none")])
def test_a_version_the_machine_does_not_list_is_named(tmp_path, monkeypatch, value, lists):
    monkeypatch.delenv("CUDA_HOME", raising=False)
    _toolkits(_run(tmp_path, monkeypatch, "one-card"), value)
    with pytest.raises(kit.NotListed, match=rf"list no CUDA toolkit 13\.1 under toolkit \(they list {lists}\): set CUDA_HOME"):
        kit.toolkit("13.1")


@pytest.mark.parametrize(("name", "arch"), [("NVIDIA H200", "sm_90a"), ("NVIDIA H100 80GB HBM3", "sm_90a")])
def test_the_arch_is_the_card_records_whose_driver_holds_the_reported_name(name, arch):
    assert kit.card_arch(name) == arch


def test_a_card_no_record_holds_is_refused_by_name():
    with pytest.raises(SystemExit, match=r"^0 card records under .* have 'NVIDIA A100-SXM4-80GB'"):
        kit.card_arch("NVIDIA A100-SXM4-80GB")


def test_the_machines_arch_is_its_cards_records(tmp_path, monkeypatch):
    _run(tmp_path, monkeypatch, "one-card")
    assert kit.machine_arch() == "sm_90a"


def test_a_machine_whose_card_is_unread_is_refused_by_name(tmp_path, monkeypatch):
    _run(tmp_path, monkeypatch, "slurm-node")
    table = (MACHINES / "slurm-node.toml").read_text().split("[properties]", 1)[1]
    (tmp_path / "inputs" / "slurm-node.toml").write_text(f'machine = "slurm-node"\n\n[properties]{table}')
    with pytest.raises(SystemExit, match=r"^machine 'slurm-node': its \[properties\] have card unread"):
        kit.machine_arch()
