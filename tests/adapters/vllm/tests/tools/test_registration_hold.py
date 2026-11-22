"""`registration_hold.py`, the launcher's hold between the registration and the first request: the challenger's matching
signature releases it, any other registration on the lane or a closed period refuses it by name, and the development path
does not wait. The ledger's signature check (`challenger.registration_head`) is handed in, as are the entries."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from verity.commitments import docket_digest as DD
from verity.commitments.live_coin import CoinError
from verity.protocols.proof.spot.verifier.challenger import challenger as Ch

ROOT = Path(__file__).resolve().parents[5]
_spec = importlib.util.spec_from_file_location("registration_hold", ROOT / "adapters/vllm/tools/served_run/registration_hold.py")
RH = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RH)

LANE = "lane-a"


def open_entry(period: str, lane: str = LANE, registration: str | None = None) -> str:
    body = {"period": period, "lane": lane, **({"registration": registration} if registration else {})}
    return json.dumps({"kind": "period-open", "session": None, "body": body})


def close_entry(period: str) -> str:
    return json.dumps({"kind": "period-close", "session": None, "body": {"period": period}})


def head(lines: list[str]) -> str:
    """`registration_head`'s answer on entries already signature-checked: the lane's one registration."""
    regs = [b["registration"] for b in (json.loads(x)["body"] for x in lines)
            if json.loads(x)["kind"] == "period-open" and b.get("lane") == LANE and "registration" in b]
    if not regs:
        raise CoinError(Ch.NO_REGISTRATION)
    if len(regs) > 1:
        raise CoinError(f"{Ch.SECOND_REGISTRATION} (entry 1)")
    return regs[0]


def reads(*ledgers: list[str]):
    """A reader that hands out `ledgers` in turn, and a wait that counts its calls."""
    it, waits = iter(ledgers), []
    return (lambda: next(it)), (lambda: waits.append(1)), waits


@pytest.fixture
def dreg(tmp_path) -> tuple[Path, str]:
    p = tmp_path / "deployment-registration.json"
    p.write_text(json.dumps({"format": "test", "salt": "00"}))
    return p, DD.digest(json.loads(p.read_text()))


def run(dreg: Path, read, wait, *extra: str) -> int:
    return RH.main(["hold", "--registration", str(dreg), "--lane", LANE, *extra], read=read, head=head, wait=wait)


def test_the_matching_signed_registration_releases_the_hold_after_waiting_on_the_ledger(dreg, capsys):
    p, d = dreg
    read, wait, waits = reads([], [open_entry("p0")], [open_entry("p0", registration=d)])
    assert run(p, read, wait) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == (f"REGISTRATION {d} on lane {LANE}; the epoch's circuit is not known before the serve "
                      "(window-commit writes circuit.json after it)")
    assert f"period-open --name <PERIOD> --lane {LANE} --cap <EPOCHS>" in out[1] and out[1].endswith(f"--registration {d}")
    assert out[-1] == f"REGISTRATION SIGNED: period p0 of lane {LANE} signs it; serving" and len(waits) == 2


def test_another_digest_signed_after_ours_was_written_is_refused(dreg, capsys):
    p, _ = dreg
    read, wait, _ = reads([], [open_entry("p0", registration="ab" * 64)])
    assert run(p, read, wait) == 3
    assert capsys.readouterr().out.splitlines()[-1].startswith(RH.OTHER_DIGEST)


def test_a_lane_already_registered_is_refused_before_any_wait(dreg, capsys):
    p, _ = dreg
    read, wait, waits = reads([open_entry("old", registration="cd" * 64), close_entry("old")])
    assert run(p, read, wait) == 3 and not waits
    assert capsys.readouterr().out.splitlines()[-1].startswith(RH.ALREADY_REGISTERED)


def test_a_second_registration_on_the_lane_is_refused(dreg, capsys):
    p, d = dreg
    read, wait, _ = reads([open_entry("p0", registration=d), close_entry("p0"), open_entry("p1", registration=d)])
    assert run(p, read, wait) == 3
    assert capsys.readouterr().out.splitlines()[-1].startswith(RH.SECOND)


def test_a_closed_registering_period_is_refused(dreg, capsys):
    p, d = dreg
    read, wait, _ = reads([], [open_entry("p0", registration=d), close_entry("p0")])
    assert run(p, read, wait) == 3
    assert capsys.readouterr().out.splitlines()[-1] == RH.PERIOD_CLOSED


def test_the_development_path_does_not_wait_and_says_first_that_no_one_signed(capsys):
    def never():
        raise AssertionError("the development path read the ledger")
    assert RH.main(["hold", "--registration", "/dev/null"], read=never, head=never, wait=never) == 0
    assert capsys.readouterr().out.splitlines() == [RH.DEVELOPMENT]


def test_a_named_lane_with_no_ledger_or_key_is_refused_before_reading(dreg, capsys):
    p, _ = dreg
    assert RH.main(["hold", "--registration", str(p), "--lane", LANE]) == 2
    assert capsys.readouterr().out.startswith("REGISTRATION REFUSED preflight")
