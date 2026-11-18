"""The Lean executables PoUS's, the bandwidth audit's and the wipe's pieces build are their schemes' declared ones
(`Launch.executables`): served_piece.sh and served_wipe.sh still state each package and target themselves, and are held
to the declaration here until they read it."""
from __future__ import annotations

import re
from pathlib import Path

from verity.protocols.compose import schemes

ROOT = Path(__file__).resolve().parents[3]


def _declared(*names: str) -> set[tuple[str, str]]:
    return {(e.package, e.target) for n in names for e in schemes.get(n).pieces["prover"].executables}


def _assigned(text: str) -> dict[str, str]:
    return dict(re.findall(r"^\s*([A-Z]+)=(verity/\S+)$", text, re.M))


def test_served_piece_builds_what_pous_and_the_bandwidth_audit_declare() -> None:
    text = (ROOT / "benchmarks/pous/served_piece.sh").read_text()
    dirs = _assigned(text)
    built = {(dirs[d], t) for d, t in re.findall(r'"\$([A-Z]+) ([a-z0-9-]+)"', text) if d in dirs}
    assert built == _declared("pous", "bandwidth"), built


def test_served_wipe_builds_what_the_wipe_declares() -> None:
    text = (ROOT / "benchmarks/pous/streamed_wipe/served_wipe.sh").read_text()
    package = _assigned(text)["L"]
    built = {(package, t) for t in re.findall(r"exec lake build ([a-z0-9-]+)\)", text)}
    assert built == _declared("wipe"), built
