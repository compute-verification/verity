"""No served launcher sets the door's templated-statement switch (`proofs.TMPL_SWITCH`): the templated statement becomes the
door's only statement by deleting the switch, never by a launcher turning it on. Outside the door's module, its settings and the
tests, no tracked file names it."""
from __future__ import annotations

import subprocess

from test_repository import ROOT

from verity.protocols.proof.spot.verifier.one_stage import proofs

DOOR = ("verity/protocols/proof/spot/verifier/one_stage/proofs.py", "verity/protocols/proof/spot/verifier/one_stage/settings.py")


def test_no_launcher_sets_the_door_switch() -> None:
    out = subprocess.run(["git", "grep", "-l", "-F", proofs.TMPL_SWITCH], cwd=ROOT, capture_output=True, text=True)
    named = [p for p in out.stdout.split() if p not in DOOR and not p.startswith("tests/")]
    assert not named, (f"{named} name {proofs.TMPL_SWITCH}: a served launcher leaves it unset; the templated statement "
                       "becomes the door's one statement when the switch is deleted from proofs.py")
