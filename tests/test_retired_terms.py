"""No retired term in the tree's prose (AGENTS.md "Working principles": a decision rolls out in the change that records it).
A semantically meaningful part of the declared or actual circuit is a block, always named with a prefix
(`ARCHITECTURE.md`, Glossary, "Block"): a proof unit is a proof block, and a faulty unit a faulty proof block.

Scanned: every tracked text file but the Lean (`.lean` files and their locks' reasons, `lean-audit.json`). A term matches case-insensitively as two words, hyphenated, or wrapped onto the next
line (behind a comment marker or not). Identifiers (`proof_unit`) are code; the retired-terms table
(`GLOSSARY.md`) names them. What may stay: the table's own rows (``RETIRED_ROWS``), a spelling baked into a
record or a digest (``ALLOWED``, one reason each; an entry that matches nothing fails too), and a file whose lines a PR
already queued on `next` rewrites (``PENDING``), until that PR lands and the rename follows it.

The maintained docs (every tracked Markdown file but a CHANGELOG) retire more (``DOC_TERMS``: "block" is reserved for
parts of circuits, the subprotocols go by their names, and "leg" is retired). Their code keeps the old words until its
owners rename it, so this scan skips code spans and fences, where a path (`legs/`) or a record key stays, and a term
written as a quoted mention ("leg"); the table's rows name them too.

``DOC_TERMS`` also holds the overhead lower bound's and the inference-only policy's retired words, each with what
replaced it (`GLOSSARY.md`, Older names). "Slack", "residual" and "faithful" stay out of it, since those words keep
other senses (a tick's slack, the Slack app; the residual stream, the covert residual; a byte-faithful replica); "dark"
alone is in it, since no maintained doc uses the word in another sense. "IU" and "NCI" match in capitals only, so a
path (`benchmarks/nci`) or a Lean name (`Nci`, `NciCheck`) is no match even outside backticks.
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
    "isolation unit, isolation block, IU": "inference block",
    "novel communication intensity, NCI": "the inference-only policy",
    "novel information, novel reads, novel bits, novel data": "fresh input",
    "replay unit": "replay block",
    "physical block, serving block, server-time block, server-window": "server block",
    "certified block": "inference block",
    "thread block": "CTA",
    "KV block": "KV page",
    "leg": "the subprotocol's name (sampled proofs, PoUW, PoUS, the bandwidth audit, the network certifier, the wipe), or run",
    "exclusion bound": "overhead lower bound (the result) or server-block yield (the per-block step)",
    "dampener, compute dampener, training efficiency, pretraining efficiency, hardware multiple": "the overhead lower "
        "bound (the result) or overhead (η is only a success probability)",
    "window cap, per-block cap, per-block limit": "server-block yield",
    "physical class": "feasible set",
    "monitored class, training class, restricted class": "admissible set (a premise set with the yield it gives is a "
                                                         "pricing argument)",
    "work requirement": "necessary subcomputation",
    "reference construction": "reference training algorithm",
    "dark, dark compute, dark fleet, dark leaves, unmonitored compute": "unregistered hardware",
    "monitored compute": "registered hardware",
    "declared compute, accounted compute, certified work, credited work": "verified compute",
    "undeclared compute, unaccounted compute, skippable work, uncertified capacity, uncertified share": "free compute",
    "monitored residual": "free compute on registered hardware (ε_undecl + ε_skip)",
    "faithfulness": "the admissible set, or verified compute",
    "unregistered input bits": "fresh input bits",
    "registered computation": "the declared circuit (registered programs on committed inputs)",
}
_SPACE = r"(?:[ \t]+|-|[ \t]*\n[ \t]*)"
_DOC_TERM = re.compile(rf"(?i:\b(?:(?:wrong|incorrect|isolation|replay){_SPACE}units?"
                       rf"|(?:physical|thread|KV|serving|certified|isolation|server{_SPACE}time){_SPACE}blocks?"
                       rf"|novel{_SPACE}(?:communication{_SPACE}intensit(?:y|ies)|information|reads?|bits?|data)"
                       rf"|server{_SPACE}windows?|exclusion{_SPACE}bounds?|window{_SPACE}caps?"
                       rf"|per{_SPACE}block{_SPACE}(?:caps?|limits?)"
                       rf"|(?:compute{_SPACE})?dampeners?|(?:pre)?training{_SPACE}efficienc(?:y|ies)"
                       rf"|hardware{_SPACE}multiples?"
                       rf"|(?:physical|monitored|training|restricted){_SPACE}class(?:es)?|work{_SPACE}requirements?"
                       rf"|reference{_SPACE}constructions?|dark{_SPACE}(?:compute|fleets?|leaves)|dark"
                       rf"|(?:(?:un)?monitored|(?:un)?declared|(?:un)?accounted){_SPACE}compute"
                       rf"|monitored{_SPACE}residuals?"
                       rf"|(?:skippable|certified|credited){_SPACE}work|uncertified{_SPACE}(?:capacit(?:y|ies)|shares?)"
                       rf"|unregistered{_SPACE}input{_SPACE}bits?|registered{_SPACE}computations?"
                       rf"|faithfulness|legs?)\b)|\b(?:IUs?|NCI)\b")
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
    "examples/h200/interfaces/verity_e2e_demo/report.py": "#4691 adds a line",
    "examples/h200/README.md": "#4680 rewrites its lines",
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
    "verity/core/protocols/sampled_proofs/README.md": "#4637 rewrites its lines",
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
        "d.md": "An exclusion bound on a serving\nblock, a server-time block and a server-window; a certified block.\n"
                "Uncertified capacity, the per-block cap, a window cap, the physical class and a training class.\n"
                "A work requirement, a dark fleet, a reference construction; not uncertified blocks, window capacity.\n",
        "e.md": "Dark compute, the dark leaves; monitored compute, declared compute and undeclared compute.\n"
                "Accounted compute, unaccounted\ncompute, skippable work, certified work; unregistered input bits.\n"
                "It went dark. Not the monitoring period, a declared circuit, accounted for, certified ingress,\n"
                "registered input bits, a tick's slack or Slack.\n",
        "f.md": "A per-block limit, two per-block\nlimits; the registered computation and registered computations.\n"
                "Not a per-block error, per-block keys, a window limit, registered programs or the computation\n"
                "registered.\n",
        "g.md": "Novel communication intensity, the NCI check and NCI's bound; an isolation block, two isolation\n"
                "blocks; novel information, novel reads and novel bits. The compute dampener, a dampener, training\n"
                "efficiency, pretraining efficiency and a hardware multiple; the monitored residual, unmonitored\n"
                "compute, credited work and faithfulness.\n"
                "Not `NciCheck`, `benchmarks/nci`, nci, Nci.Complies, the novel basis, faithful, faithfully, the\n"
                "residual stream, the covert residual, the monitoring period, the work the table credits, fresh input\n"
                'or registered information; "NCI" quoted.\n',
        "benchmarks/numerical/CHANGELOG.md": "a leg",
        "verity/README.md": "a leg",
        GLOSSARY: '| "leg"; a physical block | subprotocol |\nA leg outside the table.\n',
    }
    assert doc_offenders(list(docs), read=docs.__getitem__) == [(GLOSSARY, 2, "leg"),
        ("a.md", 1, "wrong unit"), ("a.md", 1, "Isolation-Units"), ("a.md", 1, "replay unit"), ("a.md", 2, "IU"),
        ("a.md", 2, "IUs"), ("a.md", 2, "KV block"), ("a.md", 3, "legs"), ("a.md", 3, "leg"), ("a.md", 3, "leg"),
        ("b.md", 8, "thread block"), ("d.md", 1, "exclusion bound"), ("d.md", 1, "serving block"),
        ("d.md", 2, "server-time block"), ("d.md", 2, "server-window"), ("d.md", 2, "certified block"),
        ("d.md", 3, "Uncertified capacity"), ("d.md", 3, "per-block cap"), ("d.md", 3, "window cap"),
        ("d.md", 3, "physical class"), ("d.md", 3, "training class"), ("d.md", 4, "work requirement"),
        ("d.md", 4, "dark fleet"), ("d.md", 4, "reference construction"), ("e.md", 1, "Dark compute"),
        ("e.md", 1, "dark leaves"), ("e.md", 1, "monitored compute"), ("e.md", 1, "declared compute"),
        ("e.md", 1, "undeclared compute"), ("e.md", 2, "Accounted compute"), ("e.md", 2, "unaccounted compute"),
        ("e.md", 3, "skippable work"), ("e.md", 3, "certified work"), ("e.md", 3, "unregistered input bits"),
        ("e.md", 4, "dark"), ("f.md", 1, "per-block limit"), ("f.md", 1, "per-block limits"),
        ("f.md", 2, "registered computation"), ("f.md", 2, "registered computations"),
        ("g.md", 1, "Novel communication intensity"), ("g.md", 1, "NCI"), ("g.md", 1, "NCI"),
        ("g.md", 1, "isolation block"), ("g.md", 1, "isolation blocks"), ("g.md", 2, "novel information"),
        ("g.md", 2, "novel reads"), ("g.md", 2, "novel bits"), ("g.md", 2, "compute dampener"), ("g.md", 2, "dampener"),
        ("g.md", 2, "training efficiency"), ("g.md", 3, "pretraining efficiency"), ("g.md", 3, "hardware multiple"),
        ("g.md", 3, "monitored residual"), ("g.md", 3, "unmonitored compute"), ("g.md", 4, "credited work"),
        ("g.md", 4, "faithfulness")]
