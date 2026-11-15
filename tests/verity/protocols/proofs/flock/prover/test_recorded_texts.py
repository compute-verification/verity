"""The constant text a passing run of the proof system writes into its records, pinned by name against main's bytes.

A word change alters no recorded text: a record key, a line's tag, a run file's name, a format id or a sentence a record
carries keeps its bytes, so a record made on main reads the same here (the prover's ``LIVE`` buckets and ``FIREWALL``
line, the ZK server's record's refusals, the statement identity's ``round_digest`` in Rust and Lean). Each
site is read from the producing file's source, the text being the regex's first group. A fault or refusal message a
failing run writes is no site.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[6]
FC = "verity/protocols/proofs/flock/prover/live/src/bin/flock-circuit.rs"
ZS = "verity/protocols/proofs/flock/prover/python/verity_flock/zk_server.py"
RL = "verity/protocols/proofs/flock/prover/python/verity_flock/rec_live.py"
TAGS = "verity/protocols/proofs/flock/verifier/Flock/Tags.lean"

#: name -> (the producing file, a regex whose first group is the text it writes)
SITES = {
    "flock-circuit.recheck.tag": (FC, r'eprintln!\("([A-Z_-]+)\\t\{\}", json!\(\{"recheck_root"'),
    "flock-circuit.bucket.failed": (FC, r'json!\(\{"([a-z_]+)": r\.failed,'),
    "flock-circuit.bucket.device_rounds": (FC, r'"([a-z_]+)": self\.seen\.load'),
    "flock-circuit.bucket.device_failed": (FC, r'\(None, json!\(\{"([a-z_]+)": why\}\)\)'),
    "flock-circuit.live.outcome": (FC, r'v\["([a-z_]+)"\] = [a-z_]+_outcome\(fw, seen, &v\)'),
    "flock-circuit.round_digest": (FC, r'"round_digest": "([^"]+)"'),
    "zk_server.done.tag": (ZS, r'done=lambda res: print\("([A-Z_-]+)\\t" \+ json\.dumps'),
    "zk_server.stopped": (ZS, r'why = f"recursive session \{session\} stopped at \{s\[\'part\'\]\}: ([^"]+)"'),
    "zk_server.released": (ZS, r'why = f"recursive session \{session\} released all its statements: ([^"]+)"'),
    "rec_live.session.file": (RL, r'\(d / "([^"]+)"\)\.write_text\(json\.dumps\(\{"kind": "[^"]+", "session": sid\}\)\)'),
    "rec_live.session.kind": (RL, r'write_text\(json\.dumps\(\{"kind": "([^"]+)", "session": sid\}\)\)'),
    "tags.round_digest": (TAGS, r'\("round_digest", s s!"\{h\} ([^"]+)"\)'),
}

#: sha256 of each site's text as main's records carry it
PINS = {
    "flock-circuit.recheck.tag": "5eb606913ddd8a32f7c63654582551ac72df6bcf7a90d62e9387209639dfab5a",
    "flock-circuit.bucket.failed": "09305b34acafc95790d01030042cfe6d90920decd312b326f0a7fbd7379b132d",
    "flock-circuit.bucket.device_rounds": "59c4d86d70af4e9057209be6a0508643b240ae8436b433855072aeeb220e4219",
    "flock-circuit.bucket.device_failed": "7d3263ba388cf9583f603ad31eafc53834b237a9f8cf3903b4a1ba4b2c25c484",
    "flock-circuit.live.outcome": "a3863c5d7ce546d24186581bcba0b0a157db9d318d9174d46d0db3f3f38cc1a6",
    "flock-circuit.round_digest": "2c26faaf125a675e231aaedd3368760d4e668cc1e53b4824d384b5471b8d6775",
    "zk_server.done.tag": "5eb606913ddd8a32f7c63654582551ac72df6bcf7a90d62e9387209639dfab5a",
    "zk_server.stopped": "4f91ae0fb283b2b18e04ad068e0476db4d27be0dfd1c0d626cfbffdc4c3cee36",
    "zk_server.released": "4f91ae0fb283b2b18e04ad068e0476db4d27be0dfd1c0d626cfbffdc4c3cee36",
    "rec_live.session.file": "ea2cfb2687bcda0664b885c9eb32583810d3d425f1444ea4ba6fc902c94f19e3",
    "rec_live.session.kind": "8bfb2202df0b8f22ceddc85b15f2a1cf7042734c9dfe7af93b4b157be309f1e7",
    "tags.round_digest": "49945611017e5d4a5d3564bde5a4d8aefd7fee25316dcdd6815a98dd7f825cc8",
}


def text(name: str) -> str:
    path, rx = SITES[name]
    found = re.findall(rx, (ROOT / path).read_text())
    assert found, f"{name}: no site in {path}; a moved site moves its entry here"
    return found[0]


def test_every_site_is_pinned():
    assert sorted(SITES) == sorted(PINS), "a site and its pin are added together, the pin from main's text"


@pytest.mark.parametrize("name", sorted(SITES))
def test_a_recorded_text_keeps_its_bytes(name):
    assert hashlib.sha256(text(name).encode()).hexdigest() == PINS[name], (
        f"{name}: the text {SITES[name][0]} writes into its records changed. A word change alters no recorded text: put "
        f"main's bytes back (`git show origin/main:<file>`). Only a new record format re-pins, with its schema id bumped.")
