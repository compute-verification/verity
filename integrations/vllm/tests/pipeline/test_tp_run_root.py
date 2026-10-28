"""The TP run root (`pipeline.tp.run_root`): SHA-512 over the world and every rank's 64-byte rows-sha512 root (v2), a 32-byte vllm-v1
rank root refused by name, the SHA-256 v1 form kept for old records, and the form a record's ``root_tag`` names; tp-commit and the TP
row commit every rank under rows-sha512 by default."""
from __future__ import annotations

import hashlib
import struct

import pytest

from verity_vllm.commit import step_rows
from verity_vllm.pipeline.tp import run_root as RT

ROOTS = {r: hashlib.sha512(b"rank %d" % r).digest() for r in range(4)}
ROOTS_V1 = {r: hashlib.sha256(b"rank %d" % r).digest() for r in range(4)}


def _by_hand(h, tag, roots):
    h.update(tag + struct.pack(">I", len(roots)))
    for r in range(len(roots)):
        h.update(struct.pack(">I", r) + roots[r])
    return h.digest()


def test_the_tp_run_root_is_sha512_over_the_ranks_64_byte_roots_in_rank_order():
    root = RT.tp_run_root(4, ROOTS)
    assert len(root) == 64 == step_rows.ROOT_BYTES and root == _by_hand(hashlib.sha512(), b"verity/tp-run-root/v2", ROOTS)
    assert RT.combine(RT.TP_ROOT_TAG, 4, ROOTS) == RT.combine("verity/tp-run-root/v2", 4, ROOTS) == root
    assert RT.tp_run_root(4, {**ROOTS, 1: ROOTS[2], 2: ROOTS[1]}) != root, "a rank's root is bound to its rank"
    assert RT.tp_run_root(1, {0: ROOTS[0]}) != RT.tp_run_root(2, {0: ROOTS[0], 1: ROOTS[1]})


def test_a_32_byte_rank_root_is_refused_by_name():
    with pytest.raises(ValueError, match="rank 2 root is 32 bytes, a vllm-v1 \\(SHA-256\\) root: verity/tp-run-root/v2 takes rows-sha512's 64-byte"):
        RT.tp_run_root(4, {**ROOTS, 2: ROOTS_V1[2]})
    with pytest.raises(ValueError, match="rank 0 root is 63 bytes, not 64"):
        RT.tp_run_root(1, {0: ROOTS[0][:63]})
    with pytest.raises(ValueError, match="missing rank"):
        RT.tp_run_root(4, {r: ROOTS[r] for r in (0, 1, 3)})


def test_the_v1_form_reads_old_records_over_32_byte_roots_only():
    assert RT.combine(RT.TP_ROOT_TAG_V1.decode(), 4, ROOTS_V1) == _by_hand(hashlib.sha256(), b"verity/tp-run-root/v1", ROOTS_V1)
    with pytest.raises(ValueError, match="rank 0 root is not 32 bytes"):
        RT.combine(RT.TP_ROOT_TAG_V1, 4, ROOTS)
    for tag in (None, "verity/tp-run-root/v0", ""):
        with pytest.raises(ValueError, match="is not a TP run root form"):
            RT.combine(tag, 4, ROOTS)


def test_tp_commit_and_the_tp_row_commit_every_rank_under_rows_sha512(tmp_path):
    from tests.pipeline.test_config_run import _tp
    from verity_vllm.pipeline import cli

    t = _tp(tmp_path, "--config-run", "1")
    cmd = t.commit_cmd([])
    args = cli.parse("tp-commit", cmd[cmd.index("tp-commit") + 1:], {})
    assert (args.commit_scheme, args.committer) == (step_rows.NAME, "native_collect_v2b") and "--commit-scheme" not in cmd
    assert int(t.ck_.window_mb) > 0 and t.ck_.retain != "device", "rows-sha512 commits on the windowed worker"
    for retired in ("vllm-v1", "sha256"):
        with pytest.raises(SystemExit):
            cli.parse("tp-commit", ["--workload", "w", "--out", "o", "--commit-scheme", retired], {})
