"""The tree's retired terms and what to write instead: `TERMS`, retired everywhere but the Lean, and `DOC_TERMS`, retired in
the maintained docs' prose (GLOSSARY.md's older names). `tests/test_retired_terms.py` holds the tree to them, and
`research msg` warns when a post uses one (`retired`)."""
from __future__ import annotations

import re

#: Each retired term and what to write instead.
TERMS = {"proof unit": "proof block", "faulty unit": "incorrect proof block", "proof service": "proof system"}
_SEP = r"(?:[ \t]+|-|[ \t]*\n[ \t]*(?:#+|//+|--|\*|;+)?[ \t]*)"
TERM = re.compile(rf"\b(?:(?:proof|faulty){_SEP}units?|proof{_SEP}services?)\b", re.I)

#: Retired in the maintained docs' prose, and what to write instead.
DOC_TERMS = {
    "wrong unit, incorrect unit": "incorrect proof block",
    "faulty proof block, faulty block, faulty set": "incorrect proof block; the incorrect set F",
    "isolation unit, IU": "isolation block",
    "replay unit": "replay block",
    "draw, drawn, redraw": "the sample, sampled proof block (verb: sample; coins are flipped)",
    "components, the composite's product": "subprotocols, composition",
    "firewall": "ZK server",
    "flock-verify, remote check, naming the verifier": "verity verify proof, verity verify remote (a command someone runs today "
                                                      "stays in backticks)",
    "builder, advice compiler, circuit builder": "registered compiler",
    "declared advice": "advice",
    "declared circuit": "registered circuit",
    "declared work, declared workload, declares its advice": "registered work; the developer reports its advice, which the "
                                                             "registered compiler turns into the execution block",
    "sampled unit, unit class": "sampled proof block; a size class, or Flock's proof-block class",
    "audit coin": "the challenger's coins (the step that reads them is the verification phase)",
    "declaration block, declaration period": "execution block, execution phase",
    "sampling period, sampling block, PoUW's commit period": "execution phase (work plus its audit: an execution phase and its "
                                                            "verification phase)",
    "hm96 rows, outside the commitments' docs": "hiding commitments",
    "that period's coins": "before any coins are flipped",
    "rejection dominates, reject dominance": "any rejection rejects the whole",
    "physical block": "server-time block",
    "thread block": "CTA",
    "KV block": "KV page",
    "dampener, compute dampener": "the slowdown, slowdown(𝒢, 𝒜, η) (`Nci.Dampener` and `Dampener*` stay in backticks until renamed)",
    "NCI, novel communication intensity": "the inference-only policy (`Nci*`, `NciCheck` and the `nci` directories stay in backticks until "
                                          "renamed)",
    "leg": "the subprotocol's name (sampled proofs, PoUW, PoUS, the bandwidth audit, the network certifier, the wipe), or run",
}
_SPACE = r"(?:[ \t]+|-|[ \t]*\n[ \t]*)"
DOC_TERM = re.compile(rf"(?i:\b(?:(?:wrong|incorrect|isolation|replay){_SPACE}units?|(?:physical|thread|KV){_SPACE}blocks?"
                       rf"|legs?|(?:re)?(?:draws?|drawn|drawing|drew)|components?|firewall(?:s|ed)?|flock-verify"
                       rf"|remote{_SPACE}check|builders?|advice{_SPACE}compilers?|declared{_SPACE}(?:advice|circuits?|work(?:loads?)?)"
                       rf"|declar(?:es|ed|ing){_SPACE}(?:its|their|the){_SPACE}advice|sampled{_SPACE}units?"
                       rf"|units?{_SPACE}class(?:es)?|audit{_SPACE}coins?"
                       rf"|faulty{_SPACE}(?:proof{_SPACE})?blocks?|faulty{_SPACE}sets?"
                       rf"|declaration{_SPACE}(?:blocks?|periods?)|sampling{_SPACE}(?:periods?|blocks?)|hm96{_SPACE}rows?"
                       rf"|period's{_SPACE}(?:audit{_SPACE})?coins|reject(?:ion)?{_SPACE}domina\w*|dampeners?"
                       rf"|novel{_SPACE}communication{_SPACE}intensity)\b)|\bIUs?\b|\bNCI\b"
                       rf"|\bcommit{_SPACE}periods?\b")
#: Code spans and fences, where a path or a record key keeps the old word.
CODE = re.compile(r"^[ \t]*(`{3,}|~{3,}).*?^[ \t]*\1[^\n]*$|(`+)[\s\S]*?(?<!`)\2(?!`)", re.M | re.S)


def retired(text: str) -> list[tuple[str, str]]:
    """(the term as written, what to write instead) for each retired term in `text`'s prose: code spans and fences, and a term
    written as a quoted mention ("leg"), are skipped, as the docs' scan skips them."""
    prose = CODE.sub(lambda m: re.sub(r"[^\n]", " ", m[0]), text)
    out: list[tuple[str, str]] = []
    for m in [*TERM.finditer(prose), *DOC_TERM.finditer(prose)]:
        if prose[m.start() - 1:m.start()] in ('"', "\u201c") and prose[m.end():m.end() + 1] in ('"', "\u201d"):
            continue
        said = " ".join(m[0].replace("-", " ").split())
        if all(said.lower() != s.lower() for s, _ in out):
            out.append((said, instead(said)))
    return out


def instead(term: str) -> str:
    """What to write for a retired `term` as written: the entry of TERMS or DOC_TERMS that names it."""
    t = term.lower()
    for names, write in [*TERMS.items(), *DOC_TERMS.items()]:
        for name in (n.strip().lower() for n in names.split(",")):
            if name and (name in t or t in name or t.startswith(name) or name.startswith(t.rstrip("s"))):
                return write
    return "see GLOSSARY.md's older names"
