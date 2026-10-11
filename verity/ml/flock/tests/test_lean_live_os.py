"""M0 on per-round OS coins (`FC_COINS=os`): a proving build's identity then names the coin mode and no coin commitment or
derivation, and the identity is in the statement digest, so such a session passes the Lean verifier's checks only under
`--coins os` (`Tags.withCoins`), and a seeded one only under `--coins seed`. The sessions are the fixture's: one live session
per coin mode, recorded from a proving build of main (`record.sh`, run r20261002-055711-3607), both accepted by its server.
Neither has a draw, which `verify` refuses after every check has passed (§16.15, D12), so passing is that refusal
(`DRAWLESS`). The seeded session, edited one field at a time, is also each of the record's checks' control: every edit is
refused by the check that names it, so deleting any one of them fails a test."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

FLOCK = Path(__file__).resolve().parents[1]
LEAN = Path(__file__).resolve().parents[3] / "core" / "service" / "flock"
BIN = LEAN / ".lake" / "build" / "bin" / "flock-verify"
#: RoPE d64 (4 instances) with a seeded and an FC_COINS=os loopback session of a proving build; the circuit is #306's
FIXTURE = "art:12f95e2a"
CIRCUIT = "art:425f6860/stage/circuit.txt"
STATEMENT = "verity/flock-circuit@967b8d06"
DRAWLESS = "refused: a session without a draw: every soundness claim is about a drawn session (D12)"


def _os_identity_rust() -> dict[str, str]:
    src = (FLOCK / "live" / "src" / "bin" / "flock-circuit.rs").read_text()
    block = src[src.index("if !zk_mode() && !seed_coins() {"):]
    block = block[:block.index("}")]
    return dict(re.findall(r'id(?:\["\w+"\])*\["(\w+)"\] = json!\("([^"]*)"\)', block))


def _os_identity_lean() -> dict[str, str]:
    src = (LEAN / "Flock" / "Tags.lean").read_text()
    body = src[src.index("def liveOsIdentity"):]
    body = body[:body.index("\n\n")]
    return dict(re.findall(r'setObjVal! "(\w+)" \(s "([^"]*)"\)', body))


def test_the_live_os_identity_is_the_provers():
    rust = _os_identity_rust()
    assert set(rust) == {"coins", "coin_commitment", "coin_derivation"}, rust
    assert _os_identity_lean() == rust


@pytest.fixture(scope="module")
def sessions(tmp_path_factory):
    if shutil.which("lake") is None:
        pytest.skip("no Lean toolchain")
    sys.path.insert(0, str(FLOCK / "verifier"))
    import ci

    store = tmp_path_factory.mktemp("store")
    try:
        root, circuit = ci.fetch(FIXTURE, store), ci.fetch(CIRCUIT, store)
    except (subprocess.CalledProcessError, OSError) as e:
        pytest.skip(f"the evidence store is not reachable: {e}")
    pin = (root / "stage" / "circuit.sha512").read_text().split()[0]
    assert hashlib.sha512(circuit.read_bytes()).hexdigest() == pin
    subprocess.run(["lake", "build"], cwd=LEAN, check=True, capture_output=True)
    return root, circuit


def _live(root: Path, mode: str) -> dict:
    line = next(x for x in (root / f"{mode}.prove.txt").read_text().splitlines() if x.startswith("LIVE"))
    return json.loads(line.split(None, 1)[1])


def _session(root: Path, mode: str) -> Path:
    return next(p for p in (root / "sessions" / mode).iterdir() if p.is_dir())


@pytest.fixture(scope="module")
def verdicts(sessions):
    """{(session's coins, --coins): verdict}, each flag's statement set up once over both sessions."""
    root, circuit = sessions
    out = {}
    for coins in ("seed", "os"):
        p = subprocess.run([str(BIN), "verify", "--statement", STATEMENT, "--circuit", str(circuit),
                            "--public", str(root / "stage" / "pub-4.bin"), "--coins", coins,
                            "--session", str(_session(root, "seed")), str(_session(root, "os"))],
                           capture_output=True, text=True, check=False)
        vs = [json.loads(x.split(" ", 1)[1]) for x in p.stdout.splitlines() if x.startswith("VERDICT ")]
        assert len(vs) == 2, p.stdout + p.stderr
        out.update({("seed", coins): vs[0], ("os", coins): vs[1]})
    return out


@pytest.mark.slow
def test_the_fixture_is_one_session_per_coin_mode(sessions):
    root, _ = sessions
    seed, os_ = _live(root, "seed"), _live(root, "os")
    assert seed["accepted"] and os_["accepted"]
    assert seed["coin_seed"] and "coins" not in seed["identity"]
    assert os_["coin_seed"] is None and os_["identity"]["coins"].startswith("live-os")


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["seed", "os"])
def test_a_session_passes_every_check_on_its_own_coins(verdicts, mode):
    v = verdicts[(mode, mode)]
    assert (v["accepted"], v["why"]) == (False, DRAWLESS), v["why"]


@pytest.mark.slow
@pytest.mark.parametrize("mode,coins", [("seed", "os"), ("os", "seed")])
def test_a_session_is_refused_on_the_other_coins(verdicts, mode, coins):
    v = verdicts[(mode, coins)]
    assert not v["accepted"]
    assert v["why"].startswith("S2/R7"), v["why"]


def _flip(h: str) -> str:
    return ("0" if h[0] != "0" else "1") + h[1:]


def _root_f_not_sigma(r: dict) -> None:
    f = _flip(r["link"]["root_f"])
    r["link"]["root_f"], r["link"]["root_f_sha512"] = f, hashlib.sha512(bytes.fromhex(f)).hexdigest()


#: {edit: (the edit to the seeded session's record, the refusal that names it)}, one for each of `Record.decode`'s checks
RECORD_EDITS = {
    "kind": (lambda r: r.update(kind="flock-live-session/v0"), "S1: not a flock-live-session/v1 record"),
    "aborted": (lambda r: r.update(aborted="the server stopped"), "S1: the live session aborted"),
    "require_link": (lambda r: r["config"].update(require_link=False), "S3/F1: require_link is not true"),
    "link_mode": (lambda r: r.update(link_mode="plain"), "S3/E1: link_mode is not exchange"),
    "link.mode": (lambda r: r["link"].update(mode="plain"), "S3/E1: link.mode is not exchange"),
    "sigma": (lambda r: r["link"].update(sigma=_flip(r["link"]["sigma"])), "S4: the record's Σ is not the statement's"),
    "root_b": (lambda r: r["link"]["root_b"].update(circuit=""), "S5: circuit: empty root_B"),
    "root_b_sha512": (lambda r: r["link"]["root_b_sha512"].update(circuit=_flip(r["link"]["root_b_sha512"]["circuit"])),
                      "S5: circuit: root_b differs from root_b_sha512"),
    "root_f": (lambda r: r["link"].update(root_f=""), "S6: empty root_F"),
    "root_f_sha512": (lambda r: r["link"].update(root_f_sha512=_flip(r["link"]["root_f_sha512"])),
                      "S6: root_f differs from root_f_sha512"),
    "root_f not sigma": (_root_f_not_sigma, "S6/D1: root_F is not Σ"),
    "publics_sha512": (lambda r: r["link"]["publics_sha512"].update(circuit="00" * 64),
                       "S7: circuit: publics differ from publics_sha512"),
    "points": (lambda r: r["link"].update(points=r["link"]["points"][:-1]),
               "S9: the link points are not the configured shape"),
    "y": (lambda r: r["link"]["y"].__setitem__(0, "01" + r["link"]["y"][0][2:]),
          "S10: y is not the configured shape, or not zero"),
    "round k": (lambda r: r["streams"][0]["rounds"][1].update(k=5), "S12: circuit/rep0: rounds out of order"),
    "round n": (lambda r: r["streams"][0]["rounds"][1].update(n=0), "S12: circuit/rep0 round 1 is malformed"),
    "round msg null": (lambda r: r["streams"][0]["rounds"][1].update(msg=None),
                       "S12: circuit/rep0 round 1: the statement retains round bytes and the record has none"),
    "round msg": (lambda r: r["streams"][0]["rounds"][1].update(msg=_flip(r["streams"][0]["rounds"][1]["msg"])),
                  "S12: circuit/rep0 round 1: the retained bytes are not the recorded digest's"),
    "stream name": (lambda r: r["streams"][0].update(stream="circuit/rep9"),
                    "S11/D2: the streams are not exactly the tables' rep streams"),
    "stream parent": (lambda r: r["streams"][0].update(parent="circuit/rep1"),
                      "S11: circuit/rep0 is not a root stream of this statement"),
    "stream root": (lambda r: r["streams"][0].update(root_sha512=_flip(r["streams"][0]["root_sha512"])),
                    "S13/R1: circuit/rep0 bound a root other than the committed root_B"),
    "link_sha512": (lambda r: r.update(link_sha512=_flip(r["link_sha512"])), "S14: link_sha512 is not the exchange it covers"),
}


@pytest.fixture(scope="module")
def edited(sessions, tmp_path_factory):
    """{edit: verdict} of the seeded session edited by each of `RECORD_EDITS`, verified in one call."""
    root, circuit = sessions
    seed = _session(root, "seed")
    out = tmp_path_factory.mktemp("edited")
    dirs = []
    for i, (edit, _) in enumerate(RECORD_EDITS.values()):
        d = out / f"{i:02d}"
        shutil.copytree(seed, d)
        record = json.loads((seed / "session.json").read_text())
        edit(record)
        (d / "session.json").write_text(json.dumps(record))
        dirs.append(str(d))
    p = subprocess.run([str(BIN), "verify", "--statement", STATEMENT, "--circuit", str(circuit),
                        "--public", str(root / "stage" / "pub-4.bin"), "--coins", "seed", "--session", *dirs],
                       capture_output=True, text=True, check=False)
    vs = [json.loads(x.split(" ", 1)[1]) for x in p.stdout.splitlines() if x.startswith("VERDICT ")]
    assert len(vs) == len(RECORD_EDITS), p.stdout + p.stderr
    return dict(zip(RECORD_EDITS, vs))


@pytest.mark.slow
@pytest.mark.parametrize("name", RECORD_EDITS)
def test_an_edited_record_is_refused_by_the_check_it_breaks(edited, name):
    v = edited[name]
    assert (v["accepted"], v["why"]) == (False, RECORD_EDITS[name][1])
