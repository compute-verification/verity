"""No hosting provider's name in a security assumption, claim, guarantee, property or protocol document:
what the auditor trusts is a role, the trusted device (`trusted-device`), and which provider runs it is implementation,
named only in infra and run tooling (machine names such as a node's host name included).

Scanned: every tracked file under `security/`, the claim ids' table (`verity/commitments/claim_ids.json`), and every
`PROTOCOL.md`. A name matches case-insensitively as a word or as one part of an identifier (`trusted-x-host`,
`TrustedXHost`, `x_host`), never inside a longer word (`laws`, `samples`). `lambda` is a keyword in Python and a word in
Lean and mathematics, so the provider of that name matches only in its provider forms (`Lambda Labs`, `Lambda Cloud`).
"""
from __future__ import annotations

import posixpath
import re

from test_repository import ROOT, _tracked

PROVIDERS = ("nebius", "runpod", "coreweave", "azure", "aws", "gcp", r"lambda[ -](?:labs|cloud|gpu cloud)")
_NAME = re.compile(r"(?<![a-z0-9])(?:" + "|".join(PROVIDERS) + r")(?![a-z0-9])", re.I)
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
CLAIM_IDS = "verity/commitments/claim_ids.json"


def scanned(files: list[str]) -> list[str]:
    return sorted(f for f in files if f.startswith("security/") or f == CLAIM_IDS or posixpath.basename(f) == "PROTOCOL.md")


def provider_names(line: str) -> list[str]:
    return [m[0] for m in _NAME.finditer(_CAMEL.sub(" ", line))]


def offenders(files: list[str], read=lambda f: (ROOT / f).read_text(encoding="utf-8", errors="replace")):
    """(file, line, the names found) for each line of a scanned file that names a provider."""
    return [(f, n, names) for f in scanned(files) for n, line in enumerate(read(f).splitlines(), 1)
            if (names := provider_names(line))]


def test_no_provider_name_in_a_security_statement():
    bad = offenders(_tracked())
    assert not bad, (
        "a hosting provider is named where a security assumption, claim, guarantee, property or protocol is stated; say "
        "the role instead (the trusted device, `trusted-device`), and keep provider and machine names in infra and run "
        "tooling:\n" + "\n".join(f"  {f}:{n}: {', '.join(names)}" for f, n, names in bad))


def test_the_scan_reads_each_form():
    """The rules above, on toy files."""
    files = ["security/A.lean", CLAIM_IDS, "verity/commitments/claim_ids.py", "verity/core/protocols/x/PROTOCOL.md",
             "verity/core/protocols/x/README.md"]
    docs = {
        "security/A.lean": "def TrustedNebiusHost := 1\n`trusted-runpod-host` on vy-nebius-2\nfun x => x -- lambda, laws, samples",
        CLAIM_IDS: '{"x": "lambda e: e.id",\n "y": "hosted on Lambda Labs; an AWS_REGION, the gcp_zone, Azure\'s"}',
        "verity/commitments/claim_ids.py": "AWS",
        "verity/core/protocols/x/PROTOCOL.md": "the trusted device (`trusted-device`); a coreweave node",
        "verity/core/protocols/x/README.md": "Nebius",
    }
    assert scanned(files) == ["security/A.lean", "verity/commitments/claim_ids.json", "verity/core/protocols/x/PROTOCOL.md"]
    assert offenders(files, read=docs.__getitem__) == [
        ("security/A.lean", 1, ["Nebius"]), ("security/A.lean", 2, ["runpod", "nebius"]),
        (CLAIM_IDS, 2, ["Lambda Labs", "AWS", "gcp", "Azure"]),
        ("verity/core/protocols/x/PROTOCOL.md", 1, ["coreweave"])]
