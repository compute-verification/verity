"""The messages between the workload and the prover are one list: each protocol declares its own in its
`prover/workload.py`, the developer's side, in `verity.protocol`'s and `verity.channel`'s types, and this test holds
them together. Every message is declared once, named in the Glossary's words, and carried on one channel, its writer the
channel's sender and its reader its receiver. `workload_messages.toml` beside this test lists today's writers and
readers of each message, by lane. Retired words in a name are the retired-terms lint's to refuse."""

from __future__ import annotations

import importlib
import re
import tomllib
from pathlib import Path

from verity.protocol import Manifest

ROOT = Path(__file__).resolve().parents[1]
LIST = Path(__file__).resolve().parent / "workload_messages.toml"
MODULES = {
    "spot": "verity.protocols.proofs.spot.prover.workload",
    "pouw": "verity.protocols.resource_accounting.compute.pouw.prover.workload",
    "pous": "verity.protocols.resource_accounting.memory.pous.prover.workload",
    "certifier": "verity.protocols.resource_accounting.network.prover.workload",
}
#: The two messages declared in today's direction, prover to workload, against the architecture's, each with the
#: direction the architecture gives it.
REVERSED = {
    "RegisteredCompiler": "workload to prover: the adapter hands the prover the registered compiler up front",
    "RegisteredWeights": "workload to prover: the adapter hands the prover the weights up front",
}
WORD = re.compile(r"PoU[WSB]|H1|[A-Z][a-z]+")


def _declared():
    return {p: importlib.import_module(m) for p, m in MODULES.items()}


def _glossary_words() -> set[str]:
    arch = (ROOT / "ARCHITECTURE.md").read_text()
    text = arch[arch.index("\n## Glossary\n"):] + (ROOT / "GLOSSARY.md").read_text()
    return set(re.findall(r"[a-z0-9+]+", text.lower()))


def _channel_of(module) -> dict[str, tuple[str, str]]:
    return {t: (c.sender, c.receiver) for c in module.CHANNELS for t in c.messages}


def test_each_protocol_declares_its_messages_between_the_two_parties():
    spot = _declared()["spot"]
    for p, m in _declared().items():
        Manifest(f"{p}-workload", 1, "", parties=m.PARTIES, channels=m.CHANNELS, messages=m.MESSAGES)
        assert m.PARTIES == spot.PARTIES, f"{MODULES[p]}.PARTIES are SPOT's workload and prover, not {m.PARTIES}"
        on = [t for c in m.CHANNELS for t in c.messages]
        assert sorted(on) == sorted(s.type for s in m.MESSAGES), (
            f"{MODULES[p]}: each message is on one channel, its writer the sender and its reader the receiver")


def test_every_message_is_declared_once():
    seen: dict[str, str] = {}
    for p, m in _declared().items():
        for s in m.MESSAGES:
            assert s.type not in seen, f"{s.type} is declared by both {MODULES[seen[s.type]]} and {MODULES[p]}"
            seen[s.type] = p


def test_every_name_is_the_glossarys_words():
    words = _glossary_words()
    for p, m in _declared().items():
        for s in m.MESSAGES:
            parts = WORD.findall(s.type)
            assert "".join(parts) == s.type, f"{s.type}: a name is words, each capitalized ({WORD.pattern})"
            off = [w for w in parts if w.lower() not in words and w.lower().removesuffix("s") not in words]
            assert not off, f"{MODULES[p]}.{s.type}: {off} are not the Glossary's words; name it as the Glossary does"


def test_only_the_compiler_and_the_weights_are_declared_against_the_architectures_direction():
    spot = _declared()["spot"]
    sent = _channel_of(spot)
    for t, why in REVERSED.items():
        assert sent[t] == (spot.PROVER, spot.WORKLOAD), f"{t} is declared prover to workload today ({why})"


def test_the_list_gives_each_message_its_writers_and_readers():
    declared = {s.type for m in _declared().values() for s in m.MESSAGES}
    entries = tomllib.loads(LIST.read_text())["message"]
    first = [e["type"] for e in entries if not e.get("second")]
    assert sorted(first) == sorted(declared), (
        f"{LIST.name} lists each declared message once without `second`: missing {sorted(declared - set(first))}, "
        f"extra {sorted(set(first) - declared)}")
    for e in entries:
        assert e["type"] in declared, f"{LIST.name}: {e['type']} is no declared message"
        for side in ("writers", "readers"):
            for site in e[side]:
                assert (ROOT / site["file"]).is_file(), f"{LIST.name}: {e['type']}'s {side} name {site['file']}, gone"
    for t in declared:
        sites = [e for e in entries if e["type"] == t]
        for side in ("writers", "readers"):
            assert any(e[side] for e in sites), f"{LIST.name}: {t} has no {side[:-1]} today"
