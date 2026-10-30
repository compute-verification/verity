"""M3's coin on a withheld session (`m3coin.coin`, `m3check.session_audit`), against the loopback gate (`m3gate`): the
honest client's coin is accepted, a plain session is refused by the client and by the check, and a prover that fixes its
registration after seeing the challenger's root is refused (`m3withheld`'s cases)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import m3coin as COIN  # noqa: E402
import m3reg as REG  # noqa: E402
import m3withheld as MW  # noqa: E402
from verity.core.protocols.sampled_proofs import remote as R  # noqa: E402

PIN = "c0" * 20
UTC = "2026-10-27T00:00:00Z"
RECORD = {"law": {"law": "subset", "k": 3}, "population": 40}


def _setup(tmp_path):
    reg = {"digest": REG.digest(REG.FORMAT, RECORD), "record": RECORD}
    main = tmp_path / "main.txt"
    main.write_text("e1" * 20 + "\n" + PIN + "\n")
    return reg, main


def test_an_honest_withheld_session_is_accepted_and_its_root_comes_only_at_its_close(tmp_path):
    """The open's answer carries no root and the coin's opening, with the root, comes at the signed close; the ledger
    opens the session withheld on the registration's digest, and the check accepts its coin and its draw."""
    reg, main = _setup(tmp_path)
    row = MW.withheld(reg, tmp_path / "w", PIN, main, UTC, "vy-nebius-cpu-4")
    assert row["rc"] == 0 and row["root_at_open"] is None and row["withheld_on"] is True
    assert row["accepted"] and row["audit"]["withheld"] is True
    coin = json.loads((tmp_path / "w" / "coin.json").read_text())
    assert len(coin["units"]) == 3 and coin["opening"]["after"] == reg["digest"] and "failure" not in coin


def test_a_plain_session_is_refused_by_the_client_and_by_the_check(tmp_path):
    """A gate that reserves the session plain would hand out the challenger's root at the hello: `m3coin` refuses it
    before its open, and a coin a prover's own client took from such a session, after its registration, fails the check
    on ``withheld`` alone."""
    reg, main = _setup(tmp_path)
    row = MW.plain(reg, tmp_path / "p", PIN, main, UTC, "vy-nebius-cpu-4")
    assert row["client_rc"] == 2 and row["client_failure"].startswith("refused before open: ")
    assert "is not withheld" in row["client_failure"]
    assert row["root_at_hello"] is not None and not row["accepted"]
    audit = row["audit"]
    assert audit["withheld"] is False and all(audit[k] for k in ("trusted", "pinned", "roots_ok", "commitment_ok",
                                                                    "coin_ok"))


def test_an_open_that_answers_the_root_of_a_withheld_session_is_refused_before_any_coin(tmp_path):
    """The gate reserves the session withheld but its open's answer hands out the challenger's root (or doesn't say it
    is withheld): `m3coin` refuses the open, takes no coin and writes the failure."""
    reg, main = _setup(tmp_path)
    gate, opened = MW._gate(tmp_path, PIN, UTC, True, "leaky"), []

    def leaky(words: list[str]) -> bytes:
        got = gate(words)
        if words[0] != "open":
            return got
        opened.append(o := json.loads(got))
        return json.dumps({**o, "commitment": gate.sessions[o["session"]]["root"]}).encode()
    rc = COIN.coin(leaky, reg, tmp_path, gate_name="leaky", peer="vy-nebius-cpu-4", minutes=20,
                   main=R.main_lines(main))
    (o,) = opened
    R.RemoteChallenger.connect(f"127.0.0.1:{o['port']}", bytes.fromhex(o["secret"])).close()
    gate.join()
    doc = json.loads((tmp_path / "coin.json").read_text())
    assert rc == 2 and doc["failure"].startswith("open: ") and "is not withheld" in doc["failure"]
    assert "opening" not in doc and "units" not in doc


def test_a_prover_that_registers_after_seeing_the_root_is_refused(tmp_path):
    """A prover that reads the root at a plain session's hello and then fixes its registration on it is refused (no
    session the ledger opens on that registration); on a withheld gate the same prover sees no root at the hello."""
    reg, main = _setup(tmp_path)
    row = MW.late(reg, tmp_path / "l", PIN, main, UTC, "vy-nebius-cpu-4")
    assert row["root_at_hello"] is not None and row["root_at_hello_when_withheld"] is None
    assert not row["accepted"] and row["audit"]["trusted"] is False


def test_the_recorded_run_passes_when_every_case_has_its_verdict(tmp_path):
    reg, main = _setup(tmp_path)
    (tmp_path / "reg.json").write_text(json.dumps(reg))
    assert MW.main(["--registration", str(tmp_path / "reg.json"), "--main", str(main), "--pin", PIN,
                    "--out", str(tmp_path / "out")]) == 0
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert [(c["case"], c["accepted"]) for c in summary["cases"]] == [("withheld", True), ("plain", False),
                                                                     ("late", False)]
