"""A protocol keeps a ``PROTOCOL.md`` only while it is experimental (AGENTS.md "Where writing goes"). Once its verifier is
Lean, Lean is the source of truth: the README is an outline, signposts to the Lean verifier and its guarantees and the
core API, and whatever the spec states that the Lean doesn't cover is an issue or an assumption. Every tracked
``PROTOCOL.md`` is listed here with its reason, so a new one needs one.
"""
from __future__ import annotations

import posixpath

from test_repository import ROOT, _tracked

#: Specs that stay: {PROTOCOL.md: why}.
KEPT = {
    "verity/protocols/pouw/verifier/dither/PROTOCOL.md": "experimental: Dither has no Lean verifier",
    "verity/circuits/PROTOCOL.md": "a format: the circuit file",
    "verity/commitments/frame_v3/PROTOCOL.md": "a format: frame-v3 trees",
    "verity/commitments/hm96/PROTOCOL.md": "a format: hm96 rows",
    "adapters/vllm/verity_vllm/commitments/vllm_v1/PROTOCOL.md": "a format: the vLLM v1 commitment",
    "verity/protocols/flock/prover/live/PROTOCOL.md": "a format: the zero-knowledge prover's sessions",
}
#: Specs of protocols whose verifier is already Lean, each to be deleted: {PROTOCOL.md: the Lean that covers it}. An entry
#: only goes, in the PR that deletes its file.
RETIRING = {
    "verity/protocols/certifier/verifier/PROTOCOL.md": "verity/protocols/certifier/device/program/lean",
    "verity/protocols/pous/verifier/PROTOCOL.md": "verity/protocols/pous/lean",
    "verity/protocols/poub/verifier/PROTOCOL.md": "verity/protocols/poub/verifier/lean",
}


def specs(files: list[str]) -> set[str]:
    return {f for f in files if posixpath.basename(f) == "PROTOCOL.md"}


def test_every_spec_has_a_reason():
    unlisted = sorted(specs(_tracked()) - KEPT.keys() - RETIRING.keys())
    assert not unlisted, (
        "a PROTOCOL.md that tests/test_protocol_specs.py doesn't list. A protocol keeps one only while it is experimental: "
        "if its verifier is Lean, write the README as an outline with signposts to the Lean and its guarantees, and file "
        "what the Lean doesn't cover as an issue or an assumption; if it is experimental, add it to KEPT with why:\n"
        + "\n".join(f"  {f}" for f in unlisted))


def test_lists_name_tracked_specs_and_existing_lean():
    tracked = specs(_tracked())
    stale = sorted((KEPT.keys() | RETIRING.keys()) - tracked)
    assert not stale, "listed but no longer tracked; drop the entry:\n" + "\n".join(f"  {f}" for f in stale)
    assert not KEPT.keys() & RETIRING.keys()
    missing = sorted({lean for lean in RETIRING.values() if not (ROOT / lean).is_dir()})
    assert not missing, "RETIRING names Lean that isn't there; name where the verifier now lives:\n" + "\n".join(missing)
