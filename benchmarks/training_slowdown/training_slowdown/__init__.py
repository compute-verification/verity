"""training_slowdown: the code that measures a training run's slowdown (README.md). A change to what it measures bumps
VERSION, which every record carries."""

VERSION = "4.23.0"
#: a job's exit status when it was told to stop (SIGTERM) and left its state: sysexits' "temporary failure, try again".
#: Nothing reads it, since what stops a job may report a status of its own (a lease out of time, `timeout`'s 124): a
#: stop is read from the job's record
STOPPED = 75
#: a record's `stopped` when the job was told to stop: the record is the stop's (`stop_record`), not the job's last.
#: A job that stops itself past its budget says by how many seconds instead, in the record it ends with
TOLD = "told to stop"
#: a job's record, `run.json`
SCHEMA = "training-slowdown/run/v1"
#: a job past its budget's minutes by this factor is stopped at its next step
BUDGET_SLACK = 1.2
#: a job's `--continues` when whoever starts it found no state for it to continue from
NOTHING = "nothing"


def stop_record(seed: int, split: str) -> dict:
    """The record of a run of the job of `seed` and `split` that was told to stop: whose it is, and that it was
    (`stopped`: `TOLD`). It holds no step and no checkpoint and is invalid, so nothing scores it; what the job had
    computed is in its state, where it keeps one."""
    return {"schema": SCHEMA, "training_slowdown": VERSION, "seed": seed, "split": split, "stopped": TOLD,
            "steps": [], "checkpoints": [], "invalid": ["the job was told to stop before its last step"]}


def told(record: object) -> bool:
    """Whether `record`, a `run.json` as read, says its job was told to stop."""
    return isinstance(record, dict) and record.get("stopped") == TOLD
