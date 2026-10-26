"""K and V read across steps (``kv_reads.py``, CMP-161) at the small shape: a drawn softmax and pv block of the last step
read their slot's key and value rows of every step, each opened against its own step's root and checked against a
registration naming every step root, and refused under another step-0 root, a tampered path or a changed word."""
from __future__ import annotations

import json

import kv_reads


def test_drawn_attention_blocks_read_every_steps_rows_under_its_root(tmp_path, capsys):
    rc = kv_reads.main(["--out", str(tmp_path), "--shape", "small", "--batch", "2", "--steps", "3", "--slots", "1",
                        "--blocks", "2", "--workers", "1"])
    verdict = json.loads(next(ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("VERDICT "))[8:])
    doc = json.loads((tmp_path / "kv_reads.json").read_text())
    assert rc == 0 and verdict["ok"] and doc["ok"]
    assert verdict["check"] == "clean" and verdict["replay_bad"] == 0 and verdict["steps_read"] == [0, 1, 2]
    assert verdict["reads_opened"] == 6 and all(n > 0 for n in verdict["refused"].values())
    assert [b["family"].split(".")[1] for b in doc["blocks"]] == ["softmax", "pv"] and doc["blocks"][0]["n"] == 3
    reg = doc["step_registration"]
    assert [s["step"] for s in reg["steps"]] == [0, 1, 2] and reg["steps"][-1]["root"] == reg["step_root"]
    assert all("row" not in q and "salt" not in q for r in doc["reads"] for q in r)
