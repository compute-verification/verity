"""site-registration (`site_registration.py`): the auditor's registration of its network certifier's deployment, from
its certifier-key/v1 record, is the file the onsite record's auditor-record check holds on, and is refused, saying why,
where it couldn't be."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import composite as CO  # noqa: E402
import onsite_record as OR  # noqa: E402
import site_registration as SR  # noqa: E402
import warden_chain as WC  # noqa: E402
from verity.core import site as S  # noqa: E402
from verity.core.protocols import registration as REG  # noqa: E402

KEY = {"params": [64, 8, 10], "public": "c3" * 64, "key_id": "1d" * 16}
SESSION = "5e" * 32
#: the period's params stating KEY's LMS parameters, key and session, as `scan_period.rows_case` does
PERIOD = {"key": KEY["public"], "lms": {"H": 10, "N": 64, "W": 8, "toy": False}, "session": SESSION}
NODE, GATE = "node-h200", "auditor-gate"
FIGURES = ["--beta", "100", "--latency", "1000000"]


def key_record(public: str = KEY["public"], **fields) -> dict:
    """The auditor's record of its network certifier's run (`warden_chain.RECORD`) of a key ``public``, with the fields
    the gate's run gives it (``peer``, ``host``) where ``fields`` gives them."""
    key = dict(KEY, public=public)
    dep = {"key": key, "session": SESSION}
    return {"schema": WC.RECORD, "key": key, "session": SESSION, "deployment": dep,
            "deployment_sha512": hashlib.sha512(json.dumps(dep).encode()).hexdigest(), **fields}


def run(tmp_path: Path, record: dict, *args: str) -> tuple[int, Path]:
    """`SR.main` on ``record`` written to a file, its registration's file, and ``args``."""
    (tmp_path / "certifier-key.json").write_text(json.dumps(record))
    out = tmp_path / "site-registration.json"
    return SR.main([str(tmp_path / "certifier-key.json"), str(out), *args]), out


def loc(i: str, kind: str, parent: str | None, owner: str = "prover", key: str = "") -> dict:
    return {"bytes": 0, "id": i, "key": key, "kind": kind, "owner": owner, "parent": parent, "sheet": "", "tier": None}


@pytest.mark.parametrize("public, fields", [(KEY["public"], {"peer": NODE, "host": GATE}), (KEY["public"].upper(), {})])
def test_the_registration_places_the_records_key_on_the_nodes_uplink_and_the_records_check_holds_on_it(
        tmp_path, capsys, public, fields):
    """The registration is onsite's params, the auditor holding the record's key under the certifier's id, and
    a log of one site entry placing that certifier, with that key, on the node's uplink, which the log's check
    accepts; the onsite record's auditor-record check holds on it, on parameters the composite of record composed,
    naming the certifier. A record naming its run's peer (`certifier_key`'s) or none is read alike, and a key in
    uppercase hex is registered as the lowercase hex the check and `PeriodLeakRegistered`'s hkey (`Flock.hexOf`)
    compare."""
    rc, out = run(tmp_path, key_record(public, **fields), "--node", NODE, "--certifier", GATE, *FIGURES)
    assert rc == 0
    doc = json.loads(out.read_text())
    assert doc["params"] == {"auditor": {"keys": [{"id": GATE, "key": KEY["public"]}], "sheets": [], "tolerance": "0"},
                             "composite": "onsite"}
    assert (doc["log"]["composite"], [e["previous"] for e in doc["log"]["entries"]]) == (
        "onsite", [S.genesis("onsite")])
    verdict = REG.check_log(REG.params_of(doc["params"]), REG.log_of(doc["log"]))
    assert verdict[0] == "accept"
    assert verdict[1]["site"] == {
        "sheets": [],
        "locs": [loc("site", "site", None), loc(NODE, "host", "site"),
                 loc(GATE, "certifier", "site", "auditor", KEY["public"])],
        "links": [{"beta": 100, "ends": [NODE, GATE], "id": f"{NODE}-uplink", "latency": 1_000_000},
                  {"beta": 100, "ends": [GATE, S.OUTSIDE], "id": f"{NODE}-uplink-outside", "latency": 1_000_000}]}
    head = REG.head(*REG.log_of(doc["log"]))
    assert capsys.readouterr().out == (
        f"SITE-REGISTRATION: {out}, onsite's registration placing network certifier {GATE}, which its auditor "
        f"holds the record's key for, on {NODE}'s uplink (log head {head[:16]}…)\n")
    key = tmp_path / "certifier-key.json"
    composed = hashlib.sha512(CO.canonical(PERIOD)).hexdigest()
    assert OR.auditor_record(PERIOD, composed, key, out) == {
        "is": OR.CHECKED[OR.AUDITOR_RECORD], "holds": True, "certifier": GATE,
        "from": {"--certifier-key": hashlib.sha512(key.read_bytes()).hexdigest(),
                 "--site-registration": hashlib.sha512(out.read_bytes()).hexdigest()}}


NOT_A_RECORD = f"ValueError: the auditor's record isn't a {WC.RECORD} record of a public key"
EMPTY = "ValueError: the node's, the network certifier's and the site's ids can't be empty"
#: each way the registration isn't written, as (the auditor's record, the arguments, why)
REFUSED = {
    "another schema": (key_record() | {"schema": "verity/other/v1"}, (), NOT_A_RECORD),
    "no key": ({k: v for k, v in key_record().items() if k != "key"}, (), NOT_A_RECORD),
    "a key that isn't hex": (key_record("zz" * 64), (), NOT_A_RECORD),
    "an empty key": (key_record(""), (), NOT_A_RECORD),
    "another node than the record's peer": (
        key_record(peer="node-other"), (), "ValueError: the auditor's record is of a run in front of 'node-other', not "
                                           f"--node '{NODE}'"),
    "an empty certifier id": (key_record(), ("--certifier", ""), EMPTY),
    "an empty node id, the record naming no peer": (key_record(), ("--node", ""), EMPTY),
    "no capacity": (key_record(), ("--beta", "0"), "ValueError: its log doesn't check: entry 0: kind"),
    "a negative latency": (key_record(), ("--latency", "-1"),
                           f"Malformed: link {NODE}-uplink's latency: -1 is not a natural number"),
    "the certifier the node": (key_record(), ("--certifier", NODE),
                               "ValueError: its log doesn't check: entry 0: duplicate-id"),
    "the certifier the outside": (key_record(), ("--certifier", S.OUTSIDE),
                                  "ValueError: its log doesn't check: entry 0: duplicate-id"),
    "an id outside ASCII": (key_record(), ("--certifier", "gat\u00e9"),
                            "Malformed: not ASCII without DEL (a byte at or above 0x7f)"),
}


@pytest.mark.parametrize("how", list(REFUSED))
def test_a_registration_that_wouldnt_hold_isnt_written_and_says_why(tmp_path, capsys, how):
    """No registration is written, and its line says why, where the record isn't a certifier-key/v1 record of a public
    key, where it is of a run in front of another node than the one the registration places the certifier on, where an
    id is empty, and where the registration would be one its check refuses or can't read: an uplink of no capacity or a
    negative latency, a certifier id another location's or the outside's, or an id outside ASCII."""
    record, args, why = REFUSED[how]
    given = dict(zip(args[::2], args[1::2]))
    given = {"--node": NODE, "--certifier": GATE, "--beta": "100", "--latency": "1000000"} | given
    argv = [x for k, v in given.items() for x in (k, v)]
    rc, out = run(tmp_path, record, *argv)
    assert (rc, out.exists(), capsys.readouterr().out) == (2, False, f"SITE-REGISTRATION: not written: {why}\n")
