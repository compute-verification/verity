"""The prove path's tests' settings registry (`tests/lint/test_repo_env_reads.py`): each variable a test reads, declared
once here, where a test's new read goes."""

from __future__ import annotations

import os


def zk_self_check() -> str | None:
    """`FC_ZK_SELF_CHECK` as it stands at the call, which is what a prover started then takes for its check of its own
    proofs: ``skip`` and it proves a false witness. test_records_audit reads it where the audit starts its sessions, for
    which of the audit's modes skip that check."""
    return os.environ.get("FC_ZK_SELF_CHECK")
