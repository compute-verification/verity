"""No retired term in the tree's prose (AGENTS.md "Working principles": a decision rolls out in the change that records it).
A semantically meaningful part of the declared or actual circuit is a block, always named with a prefix
(`ARCHITECTURE.md`, Glossary, "Block"): a proof unit is a proof block, and a faulty unit a faulty
proof block.

Scanned: every tracked text file but the Lean (`.lean` files and their locks' reasons, `lean-audit.json`). A term matches case-insensitively as two words, hyphenated, or wrapped onto the next
line (behind a comment marker or not). Identifiers (`proof_unit`) are code; the retired-terms table
(`GLOSSARY.md`) names them. What may stay: the table's own rows (``RETIRED_ROWS``), a spelling baked into a
record or a digest (``ALLOWED``, one reason each; an entry that matches nothing fails too), and a file whose lines a PR
already queued on `next` rewrites (``PENDING``), until that PR lands and the rename follows it.

The maintained docs (every tracked Markdown file but a CHANGELOG) retire more (``DOC_TERMS``:
"block" is reserved for parts of circuits, the subprotocols go by their names, and "leg" is retired). Their code
keeps the old words until its owners rename it, so this scan skips code spans and fences, where a path (`legs/`) or a
record key stays, and a term written as a quoted mention ("leg"); the table's rows name them too.
"""
from __future__ import annotations

import re

from test_repository import ROOT, _tracked

#: Each retired term and what to write instead.
TERMS = {"proof unit": "proof block", "faulty unit": "faulty proof block"}
_TERM = re.compile(r"\b(proof|faulty)(?:[ \t]+|-|[ \t]*\n[ \t]*(?:#+|//+|--|\*|;+)?[ \t]*)units?\b", re.I)

#: The retired-terms table, the Glossary's older names (`ARCHITECTURE.md` links it).
GLOSSARY = "GLOSSARY.md"
#: The table's rows that name the terms, in GLOSSARY.
RETIRED_ROWS = ('| "proof unit"', '| "wrong unit"')

#: Retired in the maintained docs' prose, and what to write instead.
DOC_TERMS = {
    "wrong unit, incorrect unit": "faulty proof block",
    "isolation unit, IU": "isolation block",
    "replay unit": "replay block",
    "physical block": "server-time block",
    "thread block": "CTA",
    "KV block": "KV page",
    "leg": "the subprotocol's name (sampled proofs, PoUW, PoUS, the bandwidth audit, the network certifier, the wipe), or run",
}
_SPACE = r"(?:[ \t]+|-|[ \t]*\n[ \t]*)"
_DOC_TERM = re.compile(rf"(?i:\b(?:(?:wrong|incorrect|isolation|replay){_SPACE}units?|(?:physical|thread|KV){_SPACE}blocks?"
                       rf"|legs?)\b)|\bIUs?\b")
_CODE = re.compile(r"^[ \t]*(`{3,}|~{3,}).*?^[ \t]*\1[^\n]*$|(`+)[\s\S]*?(?<!`)\2(?!`)", re.M | re.S)

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
    "demos/h200/interfaces/README.md": "#4719 rewrites its lines",
    "demos/h200/interfaces/verity_e2e_demo/report.py": "#4691 adds a line",
    "demos/h200/README.md": "#4680 rewrites its lines",
    "benchmarks/network_traces/chain_record.py": "#4575 rewrites its lines",
    "benchmarks/network_traces/transfer_zk.py": "#4441 adds it",
    "benchmarks/network_traces/transfer_zk.sh": "#4441 adds it",
    "benchmarks/stacked_share/README.md": "#4514 rewrites its lines",
    "benchmarks/torch_gpu/p2_stage/README.md": "#4761 adds lines",
    "integrations/vllm/README.md": "#4773 adds lines",
    "verity/README.md": "#4502 rewrites its lines",
    "verity/core/protocols/certifier/blocks.py": "#4441 adds it",
    "verity/core/protocols/certifier/boundary.py": "#4721 rewrites its lines",
    "verity/core/protocols/certifier/egress.py": "#4721 rewrites its lines",
    "verity/core/protocols/certifier/tests/test_network_certifier_egress.py": "#4721 rewrites its lines",
    "verity/core/protocols/network/tests/period_case.py": "#4721 rewrites its lines",
    "verity/core/protocols/network/tests/test_network_period_protocol.py": "#4721 adds lines",
    "verity/core/protocols/pous/README.md": "#4719 rewrites its lines",
    "verity/core/protocols/remote_bandwidth/PROTOCOL.md": "#4620 rewrites its lines",
    "verity/ml/circuits/ops/boolean/rowwise.py": "#4539 rewrites its lines",
    "verity/ml/windows/interface.py": "#4502 adds lines",
}


def scanned(files: list[str]) -> list[str]:
    """Every file but the Lean, PENDING and this one, which names the terms to find them."""
    return sorted(f for f in files if not f.endswith((".lean", "lean-audit.json"))
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


def doc_offenders(files: list[str], read=_read) -> list[tuple[str, int, str]]:
    """(file, line, the term as written) for each of DOC_TERMS in a maintained doc's prose."""
    out = []
    for f in scanned(files):
        if not f.endswith(".md") or f.rsplit("/", 1)[-1] == "CHANGELOG.md" or (text := read(f)) is None:
            continue
        prose = _CODE.sub(lambda m: re.sub(r"[^\n]", " ", m[0]), text)
        lines = text.splitlines()
        for m in _DOC_TERM.finditer(prose):
            n = text.count("\n", 0, m.start()) + 1
            quoted = text[m.start() - 1:m.start()] in ('"', "“") and text[m.end():m.end() + 1] in ('"', "”")
            if quoted or (f == GLOSSARY and lines[n - 1].startswith("|")):
                continue
            out.append((f, n, " ".join(m[0].split())))
    return out


def test_no_retired_term_in_prose():
    bad, used = offenders(_tracked())
    assert not bad, (
        "a retired term: say " + " and ".join(f'"{new}" for "{old}"' for old, new in TERMS.items())
        + f' (ARCHITECTURE.md, Glossary, "Block"; {GLOSSARY}); a spelling baked into a record or a digest goes in ALLOWED'
        + " with its reason:\n" + "\n".join(f"  {f}:{n}: {term}" for f, n, term in bad))
    assert not set(ALLOWED) - used, f"ALLOWED entries that match nothing any more: {sorted(set(ALLOWED) - used)}"


def test_no_retired_term_in_the_maintained_docs():
    bad = doc_offenders(_tracked())
    assert not bad, (
        "a retired term in a maintained doc: say " + "; ".join(f'{new} for "{old}"' for old, new in DOC_TERMS.items())
        + f" (ARCHITECTURE.md's Glossary, {GLOSSARY}). A path or a record key that keeps it goes in backticks:\n"
        + "\n".join(f"  {f}:{n}: {term}" for f, n, term in bad))


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
        "verity/README.md": "a proof unit",
        GLOSSARY: '| "proof unit", and "unit" alone meaning one | proof block |\n| other | a proof unit |\n',
    }
    found, used = offenders(list(docs), read=docs.__getitem__)
    assert found == [(GLOSSARY, 2, "proof unit"), ("a.py", 1, "proof unit"), ("a.py", 1, "Proof-Units"),
                     ("a.py", 1, "proof units"), ("a.py", 3, "faulty unit"), ("b.rs", 1, "proof /// unit")]
    assert used == {"N, the round's proof units"}


def test_the_doc_scan_reads_each_form():
    """The maintained-docs scan, on toy files: prose only, in Markdown only."""
    docs = {
        "a.md": "A wrong unit, two Isolation-Units and a replay\nunit; the IU and IUs; a KV block.\n"
                "The legs ran; each leg's verdict; a three-leg gate. Not legacy, alleged or IUD.\n",
        "b.md": 'The `legs/pous.py` file and `wipe_period.leg(dir)`; ``a `leg` key``; "leg" is retired.\n'
                "```sh\nLEGS=A,B ./run.sh  # each leg\n```\n~~~\na leg\n~~~\nA thread block after the fences.\n",
        "c.py": '"""A wrong unit and a leg stay in code until PR 2."""',
        "benchmarks/pouw/CHANGELOG.md": "a leg",
        "verity/README.md": "a leg",
        GLOSSARY: '| "leg"; a physical block | subprotocol |\nA leg outside the table.\n',
    }
    assert doc_offenders(list(docs), read=docs.__getitem__) == [(GLOSSARY, 2, "leg"),
        ("a.md", 1, "wrong unit"), ("a.md", 1, "Isolation-Units"), ("a.md", 1, "replay unit"), ("a.md", 2, "IU"),
        ("a.md", 2, "IUs"), ("a.md", 2, "KV block"), ("a.md", 3, "legs"), ("a.md", 3, "leg"), ("a.md", 3, "leg"),
        ("b.md", 8, "thread block")]
