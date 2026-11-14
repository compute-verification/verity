"""The command lines of the commands that write a workload, Build it and Match it, as `verity-vllm` reads them (`cli.parse`):
the field each flag is read into and its type, what a field is when its flag is left out, the flags a line may not leave out,
and each flag's whole name.  Each row of `LINES` is one flag: its field, the flag on a line, what the field then holds, and
what it holds without the flag (`NEEDED`: the line is refused without it)."""
from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from verity_vllm import config
from verity_vllm.pipeline import cli

NEEDED = "a flag the line needs"

LINES: dict[tuple[str, ...], list[tuple[str, list[str], Any, Any]]] = {
    ("build",): [
        ("out", ["--out", "o"], "o", NEEDED),
        ("M", ["--M", "2"], 2, 1),
        ("pos0", ["--pos0", "4"], 4, 8),
        ("profile", ["--profile", "a-profile"], "a-profile", "b1-eager-v3"),
        ("backend", ["--backend", "nccl"], "nccl", "gloo"),
        ("wrapper", ["--wrapper", "padded"], "padded", "step"),
        ("LP", ["--LP", "16"], 16, 8),
        ("T", ["--T", "2"], 2, 4),
        ("cached", ["--cached", "3"], 3, 0),
        ("eos", ["--eos", "5"], 5, 0),
        ("model", ["--model", "org/model"], "org/model", None),
        ("revision", ["--revision", "abc"], "abc", None),
        ("max_model_len", ["--max-model-len", "512"], 512, 2048),
        ("target", ["--target", "{}"], "{}", None),
        ("device_name", ["--device-name", "a device"], "a device", None),
        ("sampling", ["--sampling", "greedy"], "greedy", None),
        ("workload", ["--workload", "w.json"], "w.json", None),
        ("tp", ["--tp", "2"], 2, 1),
        ("rank", ["--rank", "1"], 1, 0),
        ("emit_ranks", ["--emit-ranks", "0,1"], "0,1", None),
        ("emit_root", ["--emit-root", "ranks"], "ranks", None),
        ("dead_free", ["--dead-free"], True, False),
        ("batched", ["--batched"], True, False),
        ("dsa_table_reads", ["--dsa-table-reads"], True, False),
        ("launch_context", ["--launch-context", "7"], "7", None),
        ("stop", ["--stop", "{}"], "{}", None),
    ],
    ("coverage-workloads",): [
        ("case", ["--case", "B0"], "B0", NEEDED),
        ("model", ["--model", "org/model"], "org/model", None),
        ("tokenizer", ["--tokenizer", "tok"], "tok", NEEDED),
        ("tokenizer_revision", ["--tokenizer-revision", "abc"], "abc", None),
        ("profiles", ["--profiles", "B1"], "B1", "B1,B8-static,B8-mixed"),
        ("ctx", ["--ctx", "C256"], "C256", "C1024,C256"),
        ("seed", ["--seed", "7"], 7, 0),
        ("out_dir", ["--out-dir", "o"], "o", str(config.WORKLOADS)),
        ("checkpoints", ["--checkpoints", "c.json"], "c.json", str(config.CHECKPOINTS)),
        ("replicate", ["--replicate", "2"], 2, 0),
        ("variant", ["--variant", "t1", "--variant", "chunked=64"], ["t1", "chunked=64"], []),
        ("sampling_seed", ["--sampling-seed", "5"], 5, 20260917),
        ("dry_run", ["--dry-run"], True, False),
    ],
    ("global-program",): [
        ("workload", ["--workload", "w.json"], "w.json", NEEDED),
        ("derived", ["--derived", "d0", "--derived", "d1"], ["d0", "d1"], []),
        ("out", ["--out", "o"], "o", NEEDED),
        ("checkpoint_dir", ["--checkpoint-dir", "ckpt"], "ckpt", None),
        ("eos_token_ids", ["--eos-token-ids", "1,2"], "1,2", None),
        ("artifact_root", ["--artifact-root", "art"], "art", None),
        ("follow", ["--follow", "0.5"], 0.5, None),
        ("follow_done", ["--follow-done", "done"], "done", None),
        ("follow_stop", ["--follow-stop", "stop"], "stop", None),
        ("theta_registered", ["--theta-registered"], True, False),
    ],
    ("manifest-verify",): [
        ("manifest", ["--manifest", "m.json"], "m.json", NEEDED),
        ("program", ["--program", "p.json"], "p.json", None),
        ("workload", ["--workload", "w.json"], "w.json", None),
        ("fixture", ["--fixture", "f.json"], "f.json", None),
        ("programs_root", ["--programs-root", "programs"], "programs", None),
        ("workload_digest", ["--workload-digest", "ab12"], "ab12", None),
        ("profile", ["--profile", "a-profile"], "a-profile", None),
        ("rank", ["--rank", "1"], 1, 0),
        ("ranks", ["--ranks", "2"], 2, 1),
        ("norm_scales", ["--no-norm-scales"], False, True),
        ("router_softmax", ["--no-router-softmax"], False, True),
        ("vocab_range", ["--no-vocab-range"], False, True),
        ("query", ["--query", "q"], "q", None),
        ("out", ["--out", "o.json"], "o.json", None),
    ],
    ("match",): [
        ("worker", ["--worker", "fold"], "fold", None),
        ("config_id", ["--config-id", "a-config"], "a-config", None),
        ("profile", ["--profile", "a-profile"], "a-profile", None),
        ("workload", ["--workload", "w.json"], "w.json", None),
        ("case", ["--case", "B0"], "B0", None),
        ("out", ["--out", "o"], "o", None),
        ("run_dir", ["--run-dir", "run"], "run", None),
        ("phase", ["--phase", "cpu"], "cpu", "all"),
        ("python_gpu", ["--python-gpu", "py-gpu"], "py-gpu", None),
        ("python_cpu", ["--python-cpu", "py-cpu"], "py-cpu", None),
        ("snapshot_steps", ["--snapshot-steps", "0,1"], "0,1", "all"),
        ("full_replay", ["--full-replay"], True, False),
        ("tier_b_steps", ["--tier-b-steps", "0"], "0", "0,1,2,3"),
        ("chains", ["--chains", "5"], 5, 3),
        ("no_verify_weights", ["--no-verify-weights"], True, False),
        ("parallel", ["--parallel", "2"], 2, None),
        ("max_num_seqs", ["--max-num-seqs", "4"], 4, 8),
        ("engine_arg", ["--engine-arg", "a=1", "--engine-arg", "b=true"], ["a=1", "b=true"], []),
        ("max_tokens", ["--max-tokens", "6"], 6, None),
        ("snapshot_max_bytes", ["--snapshot-max-bytes", "1024"], 1024, None),
        ("snapshot_max_bytes_op", ["--snapshot-max-bytes-op", "512"], 512, None),
        ("checkpoints", ["--checkpoints", "c.json"], "c.json", "manifests/checkpoints.json"),
        ("checkpoint_dir", ["--checkpoint-dir", "ckpt"], "ckpt", None),
        ("hf_config", ["--hf-config", "config.json"], "config.json", None),
        ("family", ["--family", "a-family"], "a-family", None),
        ("log", ["--log", "log.jsonl.gz"], "log.jsonl.gz", None),
        ("snapshots", ["--snapshots", "snaps"], "snaps", None),
        ("other_log", ["--other-log", "other.jsonl.gz"], "other.jsonl.gz", None),
        ("control_tokens", ["--control-tokens", "tokens.json"], "tokens.json", None),
        ("laptop", ["--laptop"], True, False),
        ("max_rss_gb", ["--max-rss-gb", "1.5"], 1.5, None),
        ("stage_timeout", ["--stage-timeout", "90.5"], 90.5, None),
        ("dry_run", ["--dry-run"], True, False),
        ("skip", ["--skip", "census"], "census", ""),
        ("keep_traces", ["--keep-traces"], True, False),
        ("cards_dir", ["--cards-dir", "cards"], "cards", "out/gen/cards"),
        ("tree", ["--tree", "a-tree"], "a-tree", "."),
        ("k", ["--k", "4"], 4, 10),
        ("seed", ["--seed", "9"], 9, None),
        ("no_program", ["--no-program"], True, False),
        ("persist_dir", ["--persist-dir", "persist"], "persist", None),
        ("program_dir", ["--program-dir", "p0", "--program-dir", "p1"], ["p0", "p1"], None),
        ("sampler_construction", ["--sampler-construction", "a-construction"], "a-construction", None),
        ("registered_values", ["--registered-values"], True, False),
        ("match_pipeline", ["--match-pipeline", "shared"], "shared", ""),
        ("moe_construction", ["--moe-construction", "padded"], "padded", None),
    ],
    ("served-match",): [
        ("case", ["--case", "B0"], "B0", NEEDED),
        ("workload", ["--workload", "w.json"], "w.json", NEEDED),
        ("out", ["--out", "o"], "o", NEEDED),
        ("build_of_record", ["--build-of-record", "b.json"], "b.json", None),
        ("max_num_seqs", ["--max-num-seqs", "2"], 2, 8),
        ("gpu_memory_utilization", ["--gpu-memory-utilization", "0.5"], 0.5, 0.3),
        ("kv_cache_gib", ["--kv-cache-gib", "2"], 2, 4),
        ("rows", ["--rows", "4"], 4, 16),
    ],
    ("topp-pin-check", "kernel"): [
        ("out", ["--out", "o"], "o", NEEDED),
    ],
    ("topp-pin-check", "served"): [
        ("workload", ["--workload", "w.json"], "w.json", NEEDED),
        ("out", ["--out", "o"], "o", NEEDED),
    ],
    ("topp-split-probe", "produce"): [
        ("out", ["--out", "o"], "o", NEEDED),
        ("record", ["--record", "r.json"], "r.json", None),
        ("seed", ["--seed", "7"], 7, 20260920),
    ],
    ("topp-split-probe", "check"): [
        ("dir", ["--dir", "d"], "d", NEEDED),
        ("record", ["--record", "r.json"], "r.json", None),
        ("out", ["--out", "o.json"], "o.json", None),
    ],
    ("topp-split-probe", "manifest"): [
        ("record", ["--record"], True, False),
    ],
    ("workload",): [
        ("row_id", ["--row-id", "a-row"], "a-row", NEEDED),
        ("hf_repo", ["--hf-repo", "org/model"], "org/model", NEEDED),
        ("revision", ["--revision", "abc"], "abc", NEEDED),
        ("batch", ["--batch", "8"], 8, NEEDED),
        ("max_input", ["--max-input", "256"], 256, NEEDED),
        ("max_output", ["--max-output", "32"], 32, NEEDED),
        ("sampling", ["--sampling", "stoch-t0.8-p1"], "stoch-t0.8-p1", "greedy"),
        ("arrivals", ["--arrivals", "staggered"], "staggered", "none"),
        ("case", ["--case", "B0"], "B0", None),
        ("tokenizer", ["--tokenizer", "tok"], "tok", None),
        ("lengths_only", ["--lengths-only"], True, False),
        ("check", ["--check"], True, False),
        ("shared_prefix", ["--shared-prefix", "64"], 64, 0),
        ("out", ["--out", "w.json"], "w.json", None),
        ("target", ["--target", "{}"], "{}", None),
        ("reason", ["--reason", "a=b", "--reason", "c=d"], ["a=b", "c=d"], []),
    ],
    ("workload-file",): [
        ("workloads", ["a-row", "w.json"], ["a-row", "w.json"], NEEDED),
    ],
}

COMMANDS = sorted(LINES)
FLAGS = [(command, field) for command in COMMANDS for field, flag, *_ in LINES[command] if flag[0].startswith("-")]
NEEDS = [(command, field) for command in COMMANDS for field, _flag, _read, left_out in LINES[command] if left_out is NEEDED]
NUMBERS = [(command, field) for command in COMMANDS for field, _flag, read, _left_out in LINES[command] if type(read) in (int, float)]


def _id(value: Any) -> str:
    return " ".join(value) if isinstance(value, tuple) else str(value)


@pytest.fixture(autouse=True)
def _in_a_directory_that_stores_no_workload(tmp_path, monkeypatch):
    """A `--workload` is checked against its directory's stored digests as it is read; `w.json` here is in none."""
    monkeypatch.chdir(tmp_path)


def _read(command: tuple[str, ...], argv: list[str]) -> dict[str, Any]:
    options = cli.parse(command[0], [*command[1:], *argv], {})
    assert getattr(options, "cmd", command[-1]) == command[-1]                 # a command's part says which it is, to its `main`
    return {f.name: getattr(options, f.name) for f in dataclasses.fields(options) if f.init}


def _same(got: dict[str, Any], want: dict[str, Any]) -> None:
    assert got == want
    assert {k: type(v) for k, v in got.items()} == {k: type(v) for k, v in want.items()}


def _line(command: tuple[str, ...], *, but: str | None = None, needed_only: bool = False) -> list[str]:
    return [word for field, flag, _read, left_out in LINES[command] if field != but and (left_out is NEEDED or not needed_only)
            for word in flag]


def _row(command: tuple[str, ...], field: str) -> tuple[list[str], Any, Any]:
    return next((flag, read, left_out) for name, flag, read, left_out in LINES[command] if name == field)


def _refusal(command: tuple[str, ...], argv: list[str], capsys: pytest.CaptureFixture[str]) -> str:
    """What the command says of a line it refuses, after its usage: `error: ...`, the exit code 2."""
    with pytest.raises(SystemExit) as refused:
        _read(command, argv)
    assert refused.value.code == 2
    return capsys.readouterr().err.rpartition(f"{' '.join(command)}: error: ")[2]


@pytest.mark.parametrize("command", COMMANDS, ids=_id)
def test_a_line_that_gives_every_flag_is_read_field_by_field_each_in_its_type(command):
    _same(_read(command, _line(command)), {field: read for field, _flag, read, _left_out in LINES[command]})


@pytest.mark.parametrize("command", COMMANDS, ids=_id)
def test_a_flag_that_is_left_out_leaves_its_field_what_the_table_says(command):
    _same(_read(command, _line(command, needed_only=True)),
          {field: read if left_out is NEEDED else left_out for field, _flag, read, left_out in LINES[command]})


@pytest.mark.parametrize(("command", "field"), NEEDS, ids=_id)
def test_a_line_without_a_flag_it_needs_is_refused_for_that_flag(command, field, capsys):
    flag, _, _ = _row(command, field)
    name = flag[0] if flag[0].startswith("-") else "ROW_OR_PATH"
    assert _refusal(command, _line(command, but=field), capsys) == f"the following arguments are required: {name}\n"


@pytest.mark.parametrize(("command", "field"), FLAGS, ids=_id)
def test_a_flag_is_known_by_its_whole_name(command, field, capsys):
    """A flag's first letters are read as the flag, so a line that spells one in full does not show its name; the refusal of a
    flag given without its value (a switch: with one) does."""
    flag, read, left_out = _row(command, field)
    line = _line(command, but=field)
    if type(read) is not bool:
        assert _refusal(command, [*line, flag[0]], capsys) == f"argument {flag[0]}: expected one argument\n"
        return
    name = f"{flag[0].replace('--no-', '--', 1)}/{flag[0]}" if left_out is True else flag[0]      # a switch that is on unless its no form is given
    assert _refusal(command, [*line, f"{flag[0]}=1"], capsys) == f"argument {name}: ignored explicit argument '1'\n"


@pytest.mark.parametrize(("command", "field"), NUMBERS, ids=_id)
def test_a_number_s_flag_refuses_a_value_that_is_not_of_its_type(command, field, capsys):
    flag, read, _ = _row(command, field)
    kind, value = ("int", "1.5") if type(read) is int else ("float", "much")
    assert _refusal(command, [*_line(command, but=field), flag[0], value], capsys) == f"argument {flag[0]}: invalid {kind} value: '{value}'\n"


@pytest.mark.parametrize(("command", "field", "value", "values"), [
    (("build",), "wrapper", "stepwise", "step, request, padded"),
    (("match",), "worker", "census", "fold"),
    (("match",), "phase", "both", "gpu, cpu, all"),
    (("workload",), "sampling", "stoch", "greedy, stoch-t0.8-p0.95, stoch-t0.8-p1"),
    (("workload",), "arrivals", "late", "none, staggered"),
], ids=_id)
def test_a_flag_with_a_closed_list_of_values_refuses_another_and_says_the_list(command, field, value, values, capsys):
    flag, _, _ = _row(command, field)
    said = _refusal(command, [*_line(command, but=field), flag[0], value], capsys)
    assert said.replace("'", "") == f"argument {flag[0]}: invalid choice: {value} (choose from {values})\n"       # quoted or not, by the Python


def test_a_switch_of_the_manifest_s_check_is_on_by_its_own_name_and_off_by_its_no_form():
    line = _line(("manifest-verify",), needed_only=True)
    on = _read(("manifest-verify",), [*line, "--norm-scales", "--router-softmax", "--vocab-range"])
    off = _read(("manifest-verify",), [*line, "--no-norm-scales", "--no-router-softmax", "--no-vocab-range"])
    assert [(on[k], off[k]) for k in ("norm_scales", "router_softmax", "vocab_range")] == [(True, False)] * 3
