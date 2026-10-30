"""The wipe's egress count (`wipe_egress.py`) over a gate warden run built here at the gate's scan schedule and signed
under a small LMS key: the link-windows meeting [t0 − pad, t1 + pad) and none other, each at the schedule's capacity; a
window reaching before the session's origin or past its last window, a certificate missing or failing to verify, a run
the gate's check refused, and an `out/` file other than the one `end.json` hashed each leave the record uncovered, by
name; the CLI reads the gate's `fetch` bundle as it reads the unpacked directory."""

from __future__ import annotations

import hashlib
import json
import sys
import tarfile
from functools import lru_cache
from pathlib import Path

import pytest

from verity.core.protocols.certifier.certificate import Certificate, Direction, LmsKey, certificate_message
from verity.core.protocols.certifier.warden import WardenParams
from verity.core.protocols.pous import streamed_wipe as sw
from verity.ml.crypto import lms

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "live_challenger"))
import auditor as G  # noqa: E402
import certifier_serve as cs  # noqa: E402
import warden_serve as ws  # noqa: E402
import wipe_egress as E  # noqa: E402

SCAN = WardenParams(**{flag[2:].replace("-", "_"): value for flag, value in G.SCAN.items()})
#: the origin of the rehearsal the record was first read off
ORIGIN = 1793035442.0242794
WINDOWS = 6
SESSION = bytes(range(32))
LINK, FAILS = ws.LINK, f"{ws.LINK}-fails"
MS = 1_000_000


@lru_cache(maxsize=None)
def _key() -> lms.PrivateKey:
    return lms.keygen(lms.Params(64, 1, 4), hashlib.sha512(b"wipe-egress test key").digest())


def at(ms: int) -> int:
    """`ms` after the session's origin, in Unix ns."""
    return E.origin_ns(ORIGIN) + ms * MS


def _certificate(link: str, w: int, q: int, session: bytes = SESSION) -> dict:
    key, digest = _key(), hashlib.sha512(b"record %s %d" % (link.encode(), w)).digest()
    message = certificate_message(key.I, session, link, Direction.EGRESS, w, digest)
    c = Certificate(key.I, link, Direction.EGRESS, w, digest, lms.sign_at(key, q, message).encode())
    return {"certificate": cs.certificate_json(c)}


def gate_run(root: Path, *, forged: frozenset = frozenset(), missing: frozenset = frozenset(),
             accepted: bool = True) -> Path:
    """A warden run's `end.json` and `out/` as the gate publishes them: each window's egress and fails certificates
    (signed in another session when (link, w) is in `forged`, left out when in `missing`) and a signer's refusal."""
    run = root / "c-test"
    (run / "out").mkdir(parents=True)
    key = LmsKey(_key().params, lms.public_key(_key()))
    dep = ws.deployment_record(SCAN, key, SESSION, origin_unix=ORIGIN, windows=WINDOWS, listen=("127.0.0.1", 1),
                               upstream=("127.0.0.1", 2), slack=0.025)
    lines = [json.dumps({"refused": [LINK, "ingress", WINDOWS - 1]})]
    for w in range(WINDOWS):
        for i, link in enumerate((LINK, FAILS)):
            if (link, w) not in missing:
                session = bytes(32) if (link, w) in forged else SESSION
                lines.append(json.dumps(_certificate(link, w, 2 * w + i, session), sort_keys=True))
    (run / E.DEPLOYMENT).write_text(json.dumps(dep, indent=1))
    (run / E.CERTIFICATES).write_text("".join(x + "\n" for x in lines))
    files = {Path(n).name: hashlib.sha512((run / n).read_bytes()).hexdigest() for n in (E.DEPLOYMENT, E.CERTIFICATES)}
    (run / E.END).write_text(json.dumps({"device": "warden", "rc": 0, "run": "c-test", "pinned": "0" * 40,
                                         "verdict": {"accepted": accepted}, "files": files}))
    return run


def egress(run: Path, t0: int, t1: int, pad: int = E.PAD_NS) -> dict:
    return E.record(E.read_run(run), t0, t1, pad)


def test_a_link_window_carries_the_scan_schedules_cells() -> None:
    assert E.capacity_bits(SCAN) == (SCAN.lanes * SCAN.slots + SCAN.lanes) * SCAN.window * SCAN.cell_bytes * 8 \
        == 2_621_440
    assert E.window_ns(SCAN) == 3_200_000_000
    assert sw.EGRESS_UNIT_BITS == 1 << 23


def test_a_window_within_the_session_is_covered(tmp_path: Path) -> None:
    doc = egress(gate_run(tmp_path), at(4_000), at(9_000))
    assert doc == {"format": "wipe-egress/v1", "gate_run": "c-test", "pinned": "0" * 40, "link": LINK,
                   "key": cs.public_json(LmsKey(_key().params, lms.public_key(_key()))), "session": SESSION.hex(),
                   "parameters": SCAN.digest.hex(), "origin_unix_ns": E.origin_ns(ORIGIN), "window_ns": 3_200_000_000,
                   "session_windows": WINDOWS, "t0_unix_ns": at(4_000), "t1_unix_ns": at(9_000),
                   "pad_ns": 1_000_000_000, "windows": [0, 4], "certificates": 8, "covered": True, "why": [],
                   "capacity_bits": 2_621_440, "egress_bits": 10_485_760, "egress_mib": 2, "K": 10_485_760,
                   "delta": "0"}


def test_the_pad_takes_the_windows_it_reaches_into_and_no_more(tmp_path: Path) -> None:
    run = gate_run(tmp_path)
    assert egress(run, at(6_900), at(9_100))["windows"] == [1, 4]
    assert egress(run, at(6_900), at(9_100), pad=0)["windows"] == [2, 3]
    assert egress(run, at(4_200), at(5_400))["windows"] == [1, 2], "a window ending at t0 − pad, or starting at t1 + pad"
    assert egress(run, at(4_201), at(5_401))["windows"] == [1, 3]
    assert egress(run, at(4_199), at(5_400))["windows"] == [0, 2]


def test_a_window_past_the_sessions_end_or_before_its_origin_isnt_covered(tmp_path: Path) -> None:
    run = gate_run(tmp_path)
    last = egress(run, at(10_000), at(18_100))
    assert (last["windows"], last["covered"], last["why"]) == ([2, 6], True, [])
    late = egress(run, at(10_000), at(18_500))
    assert (late["windows"], late["covered"], late["egress_bits"], late["egress_mib"], late["K"]) == \
        ([2, 7], False, None, None, None)
    assert late["why"] == [f"the window ends at link-window 7, past the session's {WINDOWS}"]
    early = egress(run, at(500), at(4_000))
    assert not early["covered"] and early["why"] == ["the window starts at link-window -1, before the session's origin"]


def test_a_certificate_that_fails_to_verify_or_is_missing_uncovers_its_window(tmp_path: Path) -> None:
    run = gate_run(tmp_path / "forged", forged=frozenset({(LINK, 2)}), missing=frozenset({(FAILS, 4)}))
    assert egress(run, at(1_000), at(4_000))["covered"], "windows 0 and 1"
    forged = egress(run, at(7_000), at(8_000))
    assert (forged["windows"], forged["covered"], forged["certificates"]) == ([1, 3], False, 3)
    assert forged["why"] == [f"{LINK} egress window 2's certificate doesn't verify under the deployment's key"]
    missing = egress(run, at(14_000), at(15_000))
    assert missing["why"] == [f"{FAILS} egress window 4 has no certificate"]
    fails = gate_run(tmp_path / "fails", forged=frozenset({(FAILS, 0)}))
    assert egress(fails, at(1_000), at(2_000))["why"] == \
        [f"{FAILS} egress window 0's certificate doesn't verify under the deployment's key"]


def test_the_gates_verdict_and_its_hashes_of_out_bind_the_count(tmp_path: Path) -> None:
    refused = egress(gate_run(tmp_path / "refused", accepted=False), at(4_000), at(9_000))
    assert refused["why"] == ["the gate's check didn't accept the run (device warden, rc 0, accepted False)"]
    run = gate_run(tmp_path / "swapped")
    lines = (run / E.CERTIFICATES).read_text().splitlines()
    (run / E.CERTIFICATES).write_text("".join(x + "\n" for x in lines[:-1]))
    assert egress(run, at(1_000), at(2_000))["why"] == [f"{E.CERTIFICATES} isn't the file the gate's end.json hashed"]


def test_the_cli_reads_the_fetch_bundle_and_exits_on_the_cover(tmp_path: Path,
                                                              capsys: pytest.CaptureFixture) -> None:
    run = gate_run(tmp_path / "run")
    (run / "witness").mkdir()
    (run / "witness" / "check.json").write_text("{}")
    bundle = tmp_path / "bundle.tgz"
    with tarfile.open(bundle, "w:gz") as tar:
        tar.add(run, arcname=run.name)
    out = tmp_path / "egress.json"
    assert E.main([str(bundle), "--t0-ns", str(at(4_000)), "--t1-ns", str(at(9_000)), "--out", str(out)]) == 0
    assert json.loads(out.read_text()) == json.loads(capsys.readouterr().out) == egress(run, at(4_000), at(9_000))
    assert E.main([str(bundle), "--t0-ns", str(at(6_900)), "--t1-ns", str(at(9_100)), "--pad-ms", "0"]) == 0
    assert json.loads(capsys.readouterr().out)["windows"] == [2, 3]
    assert E.main([str(run), "--t0-ns", str(at(4_000)), "--t1-ns", str(at(18_500))]) == 1
    assert not json.loads(capsys.readouterr().out)["covered"]
    with pytest.raises(SystemExit) as e:
        E.main([str(run), "--t0-ns", str(at(9_000)), "--t1-ns", str(at(9_000))])
    assert e.value.code == 2 and "is empty: t0 must be before t1" in capsys.readouterr().err
    with pytest.raises(SystemExit) as e:
        E.main([str(out), "--t0-ns", "0", "--t1-ns", "1"])
    assert e.value.code == 2 and "is neither a gate run's `fetch` bundle" in capsys.readouterr().err
    with pytest.raises(SystemExit) as e:
        E.main([str(run / "out"), "--t0-ns", "0", "--t1-ns", "1"])
    assert e.value.code == 2 and "the gate run has no end.json, out/deployment.json" in capsys.readouterr().err
