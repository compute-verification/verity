"""``bench_result.py``'s post-verification pruning keeps the first ``--keep-proofs`` sub-batches of every rep and the
manifest's identity of everything it deletes; the verifier command carries the bench's soundness target.  No torch, no GPU.

    uv run pytest backends/direct/ligero/bench_result_test.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from backends.direct.ligero.bench_result import prune_proofs, verify_cmd  # noqa: E402


@pytest.mark.parametrize("target, bits", [(-128, "128"), (-80.0, "80"), (-100, "100")])
def test_verify_cmd_passes_the_bench_target_as_soundness_bits(target, bits):
    """ligero-verify defaults to --soundness-bits 128 and rejects a 2^-80 proof as 'parameters give 2^-80.11 ...; requested
    2^-128', so the soundness sweep must hand it the target the bench proved to."""
    cmd = verify_cmd(Path("/bin/lv"), Path("s.bin"), Path("a.stmt"), Path("a.proof"), 16, None, target)
    assert cmd[:2] == ["/bin/lv", "verify"] and "--coins" not in cmd
    assert cmd[cmd.index("--soundness-bits") + 1] == bits and cmd[cmd.index("--threads") + 1] == "16"
    with_coins = verify_cmd(Path("/bin/lv"), Path("s.bin"), Path("a.stmt"), Path("a.proof"), 16, Path("a.coins"), target)
    assert with_coins == cmd + ["--coins", "a.coins"]


def _dump(tmp_path: Path, reps: int, subs: int, coins: bool) -> dict:
    files = []
    for rep in range(1, reps + 1):
        (tmp_path / f"rep{rep}").mkdir()
        for si in range(subs):
            stem = f"rep{rep}/sub_{si:02d}"
            (tmp_path / f"{stem}.stmt").write_bytes(b"s" * 10)
            (tmp_path / f"{stem}.proof").write_bytes(b"p" * 100)
            e = {"rep": rep, "sub": si, "stmt": f"{stem}.stmt", "proof": f"{stem}.proof", "proof_bytes": 100, "proof_sha256": "x" * 64}
            if coins:
                (tmp_path / f"{stem}.coins").write_bytes(b"c" * 128)
                e["coins"] = f"{stem}.coins"
            files.append(e)
    return {"files": files}


@pytest.mark.parametrize("coins", [False, True])
def test_prune_keeps_first_subbatches_of_every_rep(tmp_path, coins):
    man = _dump(tmp_path, reps=2, subs=4, coins=coins)
    out = prune_proofs(tmp_path, man, keep=1)
    assert out == {"kept": 2, "deleted": 6, "bytes_deleted": 6 * (110 + (128 if coins else 0))}
    for e in man["files"]:
        assert e["retained"] == (e["sub"] == 0)
        assert (tmp_path / e["proof"]).is_file() == e["retained"]
        assert (tmp_path / e["stmt"]).is_file() == e["retained"]
        if coins:
            assert (tmp_path / e["coins"]).is_file() == e["retained"]
        assert e["proof_sha256"] == "x" * 64 and e["proof_bytes"] == 100     # identity survives the deletion
    assert json.loads(json.dumps(man))["files"][0]["retained"] is True


def test_prune_keep_all_deletes_nothing(tmp_path):
    man = _dump(tmp_path, reps=1, subs=3, coins=False)
    assert prune_proofs(tmp_path, man, keep=3) == {"kept": 3, "deleted": 0, "bytes_deleted": 0}
    assert all((tmp_path / e["proof"]).is_file() for e in man["files"])
