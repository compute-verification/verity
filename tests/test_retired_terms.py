"""No retired term in the tree's prose (AGENTS.md "Working principles": a decision rolls out in the change that records it).
A semantically meaningful part of the declared or actual circuit is a block, always named with a prefix (Daniel, 26 Oct
2026, 10:07 AM PDT; `ARCHITECTURE.md`, Glossary, "Block"): a proof unit is a proof block, and a faulty unit a faulty
proof block.

Scanned: every tracked text file but `archive/` (frozen with what it describes) and the Lean (`.lean` files and their
locks' reasons, `lean-audit.json`). A term matches case-insensitively as two words, hyphenated, or wrapped onto the next
line (behind a comment marker or not). Identifiers (`proof_unit`) are code; the Glossary's retired-terms table names
them. What may stay: the table's own rows (``RETIRED_ROWS``), a spelling baked into a record or a digest (``ALLOWED``, one
reason each; an entry that matches nothing fails too), and a file whose lines a PR already queued on `next` rewrites
(``PENDING``), until that PR lands and the rename follows it.
"""
from __future__ import annotations

import re

from test_repository import ROOT, _tracked

#: Each retired term and what to write instead.
TERMS = {"proof unit": "proof block", "faulty unit": "faulty proof block"}
_TERM = re.compile(r"\b(proof|faulty)(?:[ \t]+|-|[ \t]*\n[ \t]*(?:#+|//+|--|\*|;+)?[ \t]*)units?\b", re.I)

GLOSSARY = "ARCHITECTURE.md"
#: The retired-terms table's rows that name the terms, in GLOSSARY.
RETIRED_ROWS = ('| "proof unit"', '| "wrong unit"')

ALLOWED: dict[str, str] = {
    "N, the window's proof units": "a public item's reason in A5's Spec (`benchmarks/one_stage/a5.py`): `Spec.digest` hashes it",
    "N, the round's proof units": "a public item's reason in scan_zk's Spec (`benchmarks/network_traces/scan_zk.py`): "
                                  "`Spec.digest` hashes it",
    "one proof unit per registration: the link-window": "a public item's reason in N2's Spec "
                                                        "(`certifier.service`): `Spec.digest` hashes it",
    "T proof units per registration, unit t the link-window's bucket row t": "a public item's reason in the ingress tie's "
                                                                             "Spec (`certifier.tie_service`): `Spec.digest` hashes it",
    '"proof-unit"': "the kind circuit-check's Boolean export records for a unit circuit (`boolean_export.Registry.type_of`)",
}

PENDING: dict[str, str] = {
    "Security/Proofs/NetworkCertifier/README.md": "#4690 adds lines",
    "benchmarks/census/README.md": "#4731 rewrites its lines",
    "benchmarks/e2e_demo/verity_e2e_demo/report.py": "#4691 adds a line",
    "benchmarks/network_traces/chain_record.py": "#4575 rewrites its lines",
    "benchmarks/network_traces/transfer_zk.py": "#4441 adds it",
    "benchmarks/network_traces/transfer_zk.sh": "#4441 adds it",
    "verity/README.md": "#4502 rewrites its lines",
    "verity/core/protocols/certifier/blocks.py": "#4441 adds it",
    "verity/core/protocols/certifier/boundary.py": "#4721 rewrites its lines",
    "verity/core/protocols/certifier/egress.py": "#4721 rewrites its lines",
    "verity/core/protocols/certifier/tests/test_network_certifier_egress.py": "#4721 rewrites its lines",
    "verity/core/protocols/network/tests/period_case.py": "#4721 rewrites its lines",
    "verity/core/protocols/network/tests/test_network_period_protocol.py": "#4721 adds lines",
    "verity/core/protocols/pouw/accel/PROTOCOL.md": "#4746 adds it",
    "verity/core/protocols/sampled_proofs/README.md": "#4637 rewrites its lines",
    "verity/ml/circuits/ops/boolean/rowwise.py": "#4539 rewrites its lines",
    "verity/ml/windows/interface.py": "#4502 adds lines",
}


def scanned(files: list[str]) -> list[str]:
    """Every file but `archive/`, the Lean, PENDING and this one, which names the terms to find them."""
    return sorted(f for f in files if not f.startswith("archive/") and not f.endswith((".lean", "lean-audit.json"))
                  and f not in PENDING and f != "tests/test_retired_terms.py")


def _read(f: str) -> str | None:
    try:
        return (ROOT / f).read_text(encoding="utf-8")
    except (FileNotFoundError, IsADirectoryError, UnicodeDecodeError):
        return None


def offenders(files: list[str], read=_read) -> tuple[list[tuple[str, int, str]], set[str]]:
    """(file, line, the term as written) for each retired term in a scanned file's prose, and the ALLOWED spellings used."""
    out, used = [], set()
    for f in scanned(files):
        text = read(f)
        if text is None or "unit" not in text.lower():
            continue
        allowed = [(m.start(), m.end(), s) for s in ALLOWED for m in re.finditer(re.escape(s), text)]
        lines = text.splitlines()
        for m in _TERM.finditer(text):
            n = text.count("\n", 0, m.start()) + 1
            if f == GLOSSARY and lines[n - 1].startswith(RETIRED_ROWS):
                continue
            if spelling := next((s for a, b, s in allowed if a <= m.start() and m.end() <= b), None):
                used.add(spelling)
                continue
            out.append((f, n, " ".join(m[0].split())))
    return out, used


def test_no_retired_term_in_prose():
    bad, used = offenders(_tracked())
    assert not bad, (
        "a retired term: say " + " and ".join(f'"{new}" for "{old}"' for old, new in TERMS.items())
        + f' ({GLOSSARY}, Glossary, "Block"); a spelling baked into a record or a digest goes in ALLOWED with its reason:\n'
        + "\n".join(f"  {f}:{n}: {term}" for f, n, term in bad))
    assert not set(ALLOWED) - used, f"ALLOWED entries that match nothing any more: {sorted(set(ALLOWED) - used)}"


def test_the_glossary_records_the_retired_terms():
    rows = [line for line in (ROOT / GLOSSARY).read_text(encoding="utf-8").splitlines() if line.startswith(RETIRED_ROWS)]
    assert len(rows) == len(RETIRED_ROWS), f"{GLOSSARY}'s retired-terms table has no row starting {RETIRED_ROWS}"
    assert all(_TERM.search(row) for row in rows), f"{GLOSSARY}'s rows {RETIRED_ROWS} no longer name the terms"


def test_the_scan_reads_each_form():
    """The rules above, on toy files."""
    docs = {
        "a.py": '"""A proof unit, two Proof-Units and a proof\n    units wrap."""\nx.proof_unit  # a faulty unit; unit tests\n',
        "b.rs": "/// the proof\n/// unit of it; a bulletproof unit\nlet reason = \"N, the round's proof units\";\n",
        "c.lean": "-- a proof unit",
        "archive/d.md": "a proof unit",
        "verity/README.md": "a proof unit",
        GLOSSARY: '| "proof unit", and "unit" alone meaning one | proof block |\n| other | a proof unit |\n',
    }
    found, used = offenders(list(docs), read=docs.__getitem__)
    assert found == [(GLOSSARY, 2, "proof unit"), ("a.py", 1, "proof unit"), ("a.py", 1, "Proof-Units"),
                     ("a.py", 1, "proof units"), ("a.py", 3, "faulty unit"), ("b.rs", 1, "proof /// unit")]
    assert used == {"N, the round's proof units"}
