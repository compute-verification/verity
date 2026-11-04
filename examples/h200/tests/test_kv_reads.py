"""K and V read across steps (``kv_reads.py``) at the small shape: a drawn softmax and pv block of the last step
read their slot's key and value rows of every step, each opened against its own step's root and checked against a
registration naming every step root, and refused under another step-0 root, a tampered path or a changed word; run in
one process or as the GPU's replay and the CPU's open with the prover's state between them."""
from __future__ import annotations

import json
import stat

import kv_reads
import pytest

FLAGS = ["--shape", "small", "--batch", "2", "--steps", "3", "--slots", "1", "--blocks", "2", "--workers", "1"]


def _verdict(out: str) -> dict:
    return json.loads(next(ln for ln in out.splitlines() if ln.startswith("VERDICT "))[8:])


def test_drawn_attention_blocks_read_every_steps_rows_under_its_root(tmp_path, capsys):
    rc = kv_reads.main(["--out", str(tmp_path), *FLAGS])
    verdict = _verdict(capsys.readouterr().out)
    doc = json.loads((tmp_path / "kv_reads.json").read_text())
    assert rc == 0 and verdict["ok"] and doc["ok"]
    assert verdict["check"] == "clean" and verdict["replay_bad"] == 0 and verdict["steps_read"] == [0, 1, 2]
    assert verdict["reads_opened"] == 6 and all(n > 0 for n in verdict["refused"].values())
    assert [b["family"].split(".")[1] for b in doc["blocks"]] == ["softmax", "pv"] and doc["blocks"][0]["n"] == 3
    reg = doc["step_registration"]
    assert [s["step"] for s in reg["steps"]] == [0, 1, 2] and reg["steps"][-1]["root"] == reg["step_root"]
    assert all("row" not in q and "salt" not in q for r in doc["reads"] for q in r)


def test_the_replay_and_the_open_split_across_a_private_state_give_the_same_record(tmp_path, capsys):
    one, out, private = tmp_path / "one", tmp_path / "out", tmp_path / "private"
    assert kv_reads.main(["--out", str(one), *FLAGS]) == 0
    assert kv_reads.main(["--out", str(out), "--phase", "replay", "--private", str(private), *FLAGS]) == 0
    assert not out.exists() and stat.S_IMODE((private / "state.pkl").stat().st_mode) == 0o600
    assert kv_reads.main(["--out", str(out), "--phase", "open", "--private", str(private), *FLAGS]) == 0
    assert _verdict(capsys.readouterr().out)["ok"]
    a, b = (json.loads((d / "kv_reads.json").read_text()) for d in (one, out))
    assert {k: v for k, v in a.items() if k != "seconds"} == {k: v for k, v in b.items() if k != "seconds"}
    with pytest.raises(SystemExit, match="outside --out"):
        kv_reads.main(["--out", str(out), "--phase", "open", "--private", str(out / "p"), *FLAGS])
