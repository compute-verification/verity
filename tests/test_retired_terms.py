"""No retired term in the tree's prose (AGENTS.md "Working principles": a decision rolls out in the change that records it).
A semantically meaningful part of the declared or actual circuit is a block, always named with a prefix
(`GLOSSARY.md`, "Block"): a proof unit is a proof block, and a faulty unit a faulty
proof block.

Scanned: every tracked text file but the Lean (`.lean` files and their locks' reasons, `lean-audit.json`). A term matches case-insensitively as two words, hyphenated, or wrapped onto the next
line (behind a comment marker or not). Identifiers (`proof_unit`) are code; the retired-terms table
(`GLOSSARY.md`) names them. What may stay: the table's own rows (``RETIRED_ROWS``), a spelling baked into a
record or a digest (``ALLOWED``, one reason each; an entry that matches nothing fails too), and a file whose lines a PR
already queued on `next` rewrites (``PENDING``), until that PR lands and the rename follows it.

The maintained docs (every tracked Markdown file but a CHANGELOG) retire more (``DOC_TERMS``:
"block" is reserved for parts of circuits, the subprotocols go by their names, "leg" is retired, the network certifier
is never a warden, a trace never a transcript and the sample never a draw, the inference-only policy keeps none of
NCI's names, and slowdown, in fleet-time, is never overhead). Their code
keeps the old words until its owners rename it, so this scan skips code spans and fences, where a path (`legs/`) or a
record key stays, and a term written as a quoted mention ("leg"); the table's rows name them too.
"""
from __future__ import annotations

import re

from test_repository import ROOT, _tracked

#: Each retired term and what to write instead.
TERMS = {"proof unit": "proof block", "faulty unit": "faulty proof block"}
_TERM = re.compile(r"\b(proof|faulty)(?:[ \t]+|-|[ \t]*\n[ \t]*(?:#+|//+|--|\*|;+)?[ \t]*)units?\b", re.I)

#: The retired-terms table, the glossary's older names.
GLOSSARY = "GLOSSARY.md"
#: The table's rows that name the terms, in GLOSSARY.
RETIRED_ROWS = ('| "proof unit"', '| "wrong unit"')

#: Retired in the maintained docs' prose, and what to write instead.
DOC_TERMS = {
    "wrong unit, incorrect unit": "faulty proof block",
    "isolation unit, isolation block, IU": "inference block",
    "replay unit": "replay block",
    "physical block": "server block",
    "thread block": "CTA",
    "KV block": "KV page",
    "leg": "the subprotocol's name (verity/core/protocols/composition/README.md lists them), or run",
    "warden, active warden": "the network certifier (a warden link is a network certifier link)",
    "component of a composite, the product's components": "subprotocol",
    "declaration block": "execution block",
    "the draw, a draw, its draws": "the sample (the sampling design is the rule)",
    "committed transcript, execution transcript, transcript of the gates": "trace (a transcript is the parties' messages)",
    "NCI, novel communication intensity": "the inference-only policy",
    "baseline for θ": "the registered information θ",
    "novel information, novel reads": "fresh input",
    "overhead lower bound, training overhead, overhead(𝒢, …)": "slowdown, in fleet-time (prover overhead γ keeps its name)",
    "dampener, compute dampener": "slowdown",
}
_SPACE = r"(?:[ \t]+|-|[ \t]*\n[ \t]*)"
#: What a composite goes by. "onsite" and "remote" count only with no other noun after them, so "the components of a
#: remote store" is no match.
_COMPOSITE = (rf"(?:composite{_SPACE}protocols?|composites?|products?"
              rf"|(?:onsite|remote)(?![ \t]+(?!(?:and|or|composites?|products?)\b)[a-z]))")
#: "Draw" is a noun only behind a determiner or a possessive that no verb follows, so "live coins draw proof blocks" and
#: "the challenger that draws them" are no match.
_DRAW = (rf"(?:\b(?:the|a|an|its|their|each|every|this|one|own|first|next)|['’]s?){_SPACE}draw"
         rf"|(?:\b(?:the|its|their|own)|['’]s?){_SPACE}draws")
#: "Overhead" in the training-cost sense only: prover, serving and proving overhead keep the word.
_OVERHEAD = (rf"overhead(?:['’]s)?{_SPACE}(?:lower{_SPACE})?bounds?|overhead{_SPACE}(?:calculators?|boards?|brackets?)"
             rf"|training{_SPACE}overheads?|(?:composition|policy)['’]s{_SPACE}overheads?|overhead[ \t]*\([ \t]*𝒢")
_DOC_TERM = re.compile(rf"(?i:\b(?:(?:wrong|incorrect|isolation|replay){_SPACE}units?"
                       rf"|(?:physical|thread|KV|isolation|declaration){_SPACE}blocks?|legs?|wardens?"
                       rf"|components?{_SPACE}of{_SPACE}(?:(?:the|a|an|each|either|both|its|their){_SPACE})?{_COMPOSITE}"
                       rf"|{_COMPOSITE}(?:'s?|’s?)?{_SPACE}components?"
                       rf"|(?:committed|execution|circuit['’]s|run['’]s){_SPACE}transcripts?"
                       rf"|transcripts?{_SPACE}of{_SPACE}(?:the{_SPACE})?(?:gates?|values|circuits?)"
                       rf"|novel{_SPACE}(?:communication{_SPACE}intensit(?:y|ies)|information|reads?)"
                       rf"|baseline{_SPACE}\(?θ|θ,?{_SPACE}the{_SPACE}baseline|registered{_SPACE}baselines?"
                       rf"|(?:compute{_SPACE})?dampeners?|{_OVERHEAD})\b)|(?i:{_DRAW})\b|\b(?:IUs?|NCI)\b")
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
    "Security/Proofs/Flock/Soundness/DESIGN.md": "#5345 rewrites its lines",
    "Security/Proofs/Flock/Soundness/assumption-notes/a3-uniform-bytes.md": "#5345, #5471 and #5483 rewrite its lines",
    "Security/Proofs/Flock/Soundness/assumption-notes/lock-listed-premises.md": "#5345 rewrites its lines",
    "Security/Proofs/NetworkCertifier/README.md": "#4690 adds lines",
    "benchmarks/README.md": "#5522 adds it",
    "benchmarks/census/README.md": "#4731 rewrites its lines",
    "benchmarks/network_traces/README.md": "#5339 and #5458 rewrite its lines",
    "benchmarks/network_traces/chain_record.py": "#4575 rewrites its lines",
    "benchmarks/network_traces/transfer_zk.py": "#4441 adds it",
    "benchmarks/network_traces/transfer_zk.sh": "#4441 adds it",
    "benchmarks/one_stage/README.md": "#5428, #5436 and #5518 rewrite its lines",
    "benchmarks/pouw/fp_exact/README.md": "#5439 and #5585 rewrite its lines",
    "benchmarks/proof_throughput/README.md": "#5505 rewrites its lines",
    "benchmarks/prove_path/e2e_chains/README.md": "#5085, #5339 and #5637 rewrite its lines",
    "benchmarks/sampling_plan/README.md": "#5505 rewrites its lines",
    "benchmarks/stacked_share/README.md": "#4514 rewrites its lines",
    "benchmarks/torch_gpu/p2_stage/README.md": "#4761 adds lines",
    "benchmarks/verifier_cost/README.md": "#5215 and #5505 rewrite its lines",
    "examples/h200/README.md": "#4680 rewrites its lines",
    "examples/h200/interfaces/README.md": "#4719 rewrites its lines",
    "examples/h200/interfaces/verity_e2e_demo/report.py": "#4691 adds a line",
    "examples/vllm/README.md": "#5637 rewrites its lines",
    "integrations/vllm/README.md": "#4773 adds lines",
    "verity/README.md": "#4502 rewrites its lines",
    "verity/core/protocols/bandwidth/PROTOCOL.md": "#5511 rewrites its lines",
    "verity/core/protocols/certifier/PROTOCOL.md": "#5178 rewrites its lines",
    "verity/core/protocols/certifier/README.md": "#5295 and #5339 rewrite its lines",
    "verity/core/protocols/certifier/blocks.py": "#4441 adds it",
    "verity/core/protocols/certifier/boundary.py": "#4721 rewrites its lines",
    "verity/core/protocols/certifier/egress.py": "#4721 rewrites its lines",
    "verity/core/protocols/certifier/protocol/arbitrary-serving-attacks.md":
        "a section of the certifier's PROTOCOL.md, which #5178 rewrites",
    "verity/core/protocols/certifier/tests/test_network_certifier_egress.py": "#4721 rewrites its lines",
    "verity/core/protocols/network/tests/period_case.py": "#4721 rewrites its lines",
    "verity/core/protocols/network/tests/test_network_period_protocol.py": "#4721 adds lines",
    "verity/core/protocols/pous/PROTOCOL.md": "#4828, #5219 and #5581 rewrite its lines",
    "verity/core/protocols/pous/README.md": "#4719 rewrites its lines",
    "verity/core/protocols/pouw/PROTOCOL.md": "#5046, #5353, #5409 and #5551 rewrite its lines",
    "verity/core/protocols/pouw/dither/PROTOCOL.md": "#5353 and #5637 rewrite its lines",
    "verity/core/protocols/pouw/lean/check/README.md": "#5046 rewrites its lines",
    "verity/core/protocols/remote/README.md": "#5637 rewrites its lines",
    "verity/core/protocols/remote_bandwidth/PROTOCOL.md": "#4620 rewrites its lines",
    "verity/core/protocols/sampled_proofs/README.md": "#5371, #5637 and #5643 rewrite its lines",
    "verity/ml/circuits/ops/boolean/rowwise.py": "#4539 rewrites its lines",
    "verity/ml/flock/live/PROTOCOL.md": "#5139 and #5483 rewrite its lines",
    "verity/ml/flock/verifier/PROTOCOL.md": "#5139 rewrites its lines",
    "verity/ml/flock/verifier/README.md": "#4891, #5139 and #5502 rewrite its lines",
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
        + f' ({GLOSSARY}, "Block"); a spelling baked into a record or a digest goes in ALLOWED'
        + " with its reason:\n" + "\n".join(f"  {f}:{n}: {term}" for f, n, term in bad))
    assert not set(ALLOWED) - used, f"ALLOWED entries that match nothing any more: {sorted(set(ALLOWED) - used)}"


def test_no_retired_term_in_the_maintained_docs():
    bad = doc_offenders(_tracked())
    assert not bad, (
        "a retired term in a maintained doc: say " + "; ".join(f'{new} for "{old}"' for old, new in DOC_TERMS.items())
        + f" ({GLOSSARY}). A path or a record key that keeps it goes in backticks:\n"
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
        "benchmarks/numerical/CHANGELOG.md": "a leg",
        "verity/README.md": "a leg",
        GLOSSARY: '| "leg"; a physical block | subprotocol |\nA leg outside the table.\n',
    }
    assert doc_offenders(list(docs), read=docs.__getitem__) == [(GLOSSARY, 2, "leg"),
        ("a.md", 1, "wrong unit"), ("a.md", 1, "Isolation-Units"), ("a.md", 1, "replay unit"), ("a.md", 2, "IU"),
        ("a.md", 2, "IUs"), ("a.md", 2, "KV block"), ("a.md", 3, "legs"), ("a.md", 3, "leg"), ("a.md", 3, "leg"),
        ("b.md", 8, "thread block")]


def test_the_doc_scan_reads_the_scoped_terms():
    """The terms that keep a sense outside the docs' vocabulary match only in the retired one."""
    docs = {
        "i.md": "The warden signs; the active warden's grid, two wardens and a\nWarden link.\n"
                'Not `warden.py`, `WardenParams`, `verity-network-warden`, warden_rows, the wardenship or "warden" quoted.\n',
        "j.md": "A component of the composite, two components of onsite, the remote composite's components, the\n"
                "product's components, the composites' components and each component of a composite protocol.\n"
                "Not a component of the ZK server, each component's README, `Core.Onsite.components`, the `components`\n"
                "key, a component of the circuit, the components of a remote store or onsite's own components list.\n",
        "k.md": "A declaration block; the draw, A draw and its draws; the verifier's draw.\n"
                "Not: live coins draw proof blocks, the challenger that draws them, the drawn blocks, the drawing.\n"
                "The committed transcript and a transcript of the gates. Not PoUS's transcript hash.\n"
                "NCI and novel communication intensity; novel information and novel reads. Not `Nci`, nci or NCIs.\n"
                "The baseline θ and θ, the baseline. Not a baseline run.\n"
                "The overhead lower bound, training overhead, the policy's overhead and overhead(𝒢, 𝒜, η), a\n"
                "compute dampener. Not prover overhead, serving overhead or the proving overhead γ.\n",
    }
    assert doc_offenders(list(docs), read=docs.__getitem__) == [
        ("i.md", 1, "warden"), ("i.md", 1, "warden"), ("i.md", 1, "wardens"), ("i.md", 2, "Warden"),
        ("j.md", 1, "component of the composite"), ("j.md", 1, "components of onsite"),
        ("j.md", 1, "composite's components"), ("j.md", 2, "product's components"), ("j.md", 2, "composites' components"),
        ("j.md", 2, "component of a composite protocol"),
        ("k.md", 1, "declaration block"), ("k.md", 1, "the draw"), ("k.md", 1, "A draw"), ("k.md", 1, "its draws"),
        ("k.md", 1, "'s draw"), ("k.md", 3, "committed transcript"), ("k.md", 3, "transcript of the gates"),
        ("k.md", 4, "NCI"), ("k.md", 4, "novel communication intensity"), ("k.md", 4, "novel information"),
        ("k.md", 4, "novel reads"), ("k.md", 5, "baseline θ"), ("k.md", 5, "θ, the baseline"),
        ("k.md", 6, "overhead lower bound"), ("k.md", 6, "training overhead"), ("k.md", 6, "policy's overhead"),
        ("k.md", 6, "overhead(𝒢"), ("k.md", 7, "compute dampener")]
