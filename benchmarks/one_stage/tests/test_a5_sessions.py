"""a5's drawn members' sessions run one at a time: outer proofs go through the firewall one at a time (AGENTS.md), so
``--session-jobs`` takes only 1, kept so the commands that pass it still run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import a5  # noqa: E402


def test_session_jobs_takes_one():
    assert a5.one_session("1") == 1


@pytest.mark.parametrize("jobs", ["0", "2", "16"])
def test_session_jobs_refuses_anything_else_naming_the_rule(jobs):
    with pytest.raises(argparse.ArgumentTypeError, match="through the firewall one at a time"):
        a5.one_session(jobs)
