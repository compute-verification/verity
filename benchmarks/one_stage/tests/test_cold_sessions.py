"""cold_sessions.py: a window's plan names every file a member's session needs or exits 1 naming the missing one, the ZK
server's part of a session's clock is what its witness, proving and wait leave, and the table has a row a member with
each arm's session or its refusal."""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import cold_sessions as C  # noqa: E402


def window(root: Path, ks: list[int]) -> Path:
    w = root / "audit" / "window"
    for k in ks:
        d = w / f"m{k}"
        d.mkdir(parents=True)
        (d / "circuit.txt").write_text("c")
        (d / "inst-4.bin").write_bytes(b"i")
        (d / "pub-4.bin").write_text(json.dumps({"circuit_sha512": f"pin{k}"}) + "\n")
        (w / f"session-m{k}").mkdir()
        (w / f"session-m{k}" / C.a0.SAMPLE_FILE).write_text(json.dumps({"k": 1, "population": 4, "units": [2]}))
    (w / "summary.json").write_text(json.dumps({"members": [{"k": k, "n": 4, "template": "T_v1{N=1}"} for k in ks]}))
    (w / "partition.json").write_text("{}")
    (w / "program.json").write_text("{}")
    return w


def test_the_plan_names_each_members_statement_and_exits_1_naming_a_missing_file(tmp_path, capsys):
    w = window(tmp_path, [5, 7])
    assert C.main(["plan", "--window", str(w), "--members", "5,7"]) == 0
    plans = [json.loads(x.split(" ", 1)[1]) for x in capsys.readouterr().out.splitlines() if x.startswith("PLAN ")]
    assert [(p["k"], p["n"], p["pin"], p["sample"]["units"]) for p in plans] == [(5, 4, "pin5", [2]), (7, 4, "pin7", [2])]
    (w / "m7" / "inst-4.bin").unlink()
    assert C.main(["plan", "--window", str(w), "--members", "5,7,9"]) == 1
    out = capsys.readouterr().out
    assert "PLAN-MISSING m7: missing" in out and "inst-4.bin" in out and "PLAN-MISSING m9: not in summary.json" in out


def test_the_zk_servers_part_of_the_clock_is_what_witness_proving_and_wait_leave():
    assert C.outside({"e2e_s": 100.0, "witness_s": 10.0, "prove_total_s": 50.0, "wait_e2e_s": 15.0}) == 25.0
    assert C.outside({"e2e_s": 100.0, "prove_total_s": 50.0}) is None


def test_the_table_has_a_row_a_member_with_each_arms_session_or_its_refusal(tmp_path):
    def rec(arm: str, k: int, accepted: bool) -> dict:
        return {"k": k, "arm": arm, "template": "AttnBlockFirst_v1{D=64}", "circuit_bytes": 1234, "accepted": accepted,
                "refusal": None if accepted else "recv: failed to fill whole buffer", "session_wall_s": 9.0,
                "prover_built_s": 1.0, "verifier_built_s": 2.0, "live": {"e2e_s": 5.0}, "outside_s": 1.0,
                "host_peak_gib": 3.0, "device_peak_mib": 4096}
    for arm, k, ok in (("gpu", 8, True), ("cpu", 8, True), ("gpu", 127997, False)):
        d = C.session_dir(tmp_path, arm, k)
        d.mkdir(parents=True)
        (d / "cold.json").write_text(json.dumps(rec(arm, k, ok)))
    (C.session_dir(tmp_path, "gpu", 8) / "verdict.json").write_text(json.dumps({"accepted": True, "verdict_wall_s": 7.0}))
    assert C.main(["table", "--out", str(tmp_path)]) == 0
    rows = (tmp_path / "table.md").read_text().splitlines()[2:]
    assert len(rows) == 2 and rows[0].startswith("| m8 | `AttnBlockFirst_v1` | 1,234 |")
    assert "verdict 7.0 s accepted" in rows[0] and "card 4096 MiB" in rows[0] and "no verdict" in rows[0]
    assert "refused: recv: failed to fill whole buffer" in rows[1] and rows[1].endswith("| not run |")
