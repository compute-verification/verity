"""A protocol keeps a ``PROTOCOL.md`` only while it is experimental (AGENTS.md "Where writing goes"). Once its verifier is
Lean, Lean is the source of truth: the README is an outline, signposts to the Lean verifier and its guarantees and the
core API, and whatever the spec states that the Lean doesn't cover is an issue or a named premise. Every tracked
``PROTOCOL.md`` is listed here with its reason, so a new one needs one.
"""
from __future__ import annotations

import posixpath

from test_repository import ROOT, _tracked

#: Specs that stay: {PROTOCOL.md: why}.
KEPT = {
    "verity/core/protocols/bandwidth/PROTOCOL.md": "experimental: the bandwidth audit over PoUS's store has no Lean verifier",
    "verity/core/protocols/pouw/dither/PROTOCOL.md": "experimental: Dither has no Lean verifier",
    "verity/ml/circuits/PROTOCOL.md": "a format: the circuit file",
    "verity/ml/commitments/frame_v3/PROTOCOL.md": "a format: frame-v3 trees",
    "verity/ml/commitments/hm96/PROTOCOL.md": "a format: hm96 rows",
    "verity/ml/commitments/vllm_v1/PROTOCOL.md": "a format: the vLLM v1 commitment",
    "verity/ml/flock/live/PROTOCOL.md": "a format: the zero-knowledge prover's sessions",
}
#: Specs of protocols whose verifier is already Lean, each to be deleted: {PROTOCOL.md: the Lean that covers it}. An entry
#: only goes, in the PR that deletes its file.
RETIRING = {
    "verity/core/protocols/certifier/PROTOCOL.md": "verity/core/protocols/certifier/lean",
    "verity/core/protocols/pous/PROTOCOL.md": "verity/core/protocols/pous/lean",
    "verity/core/protocols/pous/p2/PROTOCOL.md": "verity/core/protocols/pous/lean",
    "verity/core/protocols/pouw/PROTOCOL.md": "verity/core/protocols/pouw/lean",
    "verity/core/protocols/remote_bandwidth/PROTOCOL.md": "verity/core/protocols/remote_bandwidth/lean",
    "verity/ml/flock/verifier/PROTOCOL.md": "verity/core/service/flock",
}


def specs(files: list[str]) -> set[str]:
    return {f for f in files if posixpath.basename(f) == "PROTOCOL.md"}


def test_every_spec_has_a_reason():
    unlisted = sorted(specs(_tracked()) - KEPT.keys() - RETIRING.keys())
    assert not unlisted, (
        "a PROTOCOL.md that tests/test_protocol_specs.py doesn't list. A protocol keeps one only while it is experimental: "
        "if its verifier is Lean, write the README as an outline with signposts to the Lean and its guarantees, and file "
        "what the Lean doesn't cover as an issue or a named premise; if it is experimental, add it to KEPT with why:\n"
        + "\n".join(f"  {f}" for f in unlisted))


def test_lists_name_tracked_specs_and_existing_lean():
    tracked = specs(_tracked())
    stale = sorted((KEPT.keys() | RETIRING.keys()) - tracked)
    assert not stale, "listed but no longer tracked; drop the entry:\n" + "\n".join(f"  {f}" for f in stale)
    assert not KEPT.keys() & RETIRING.keys()
    missing = sorted({lean for lean in RETIRING.values() if not (ROOT / lean).is_dir()})
    assert not missing, "RETIRING names Lean that isn't there; name where the verifier now lives:\n" + "\n".join(missing)
