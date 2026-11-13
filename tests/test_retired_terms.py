"""No retired term in the tree's prose (AGENTS.md "Working principles": a decision rolls out in the change that records it).
A semantically meaningful part of the registered or actual circuit is a block, always named with a prefix
(`ARCHITECTURE.md`, Glossary, "Block"): a proof unit is a proof block, and a faulty or wrong unit an incorrect
proof block (Glossary, "Incorrect proof block"). The proof service is the proof system (`ARCHITECTURE.md`, Glossary, "Proof system").

Scanned: every tracked text file but the Lean packages' `lakefile.toml` and their locks' reasons (`lean-audit.json`), the
lints that name the terms (``LINTS``: `tests/lint/test_repo_retired_terms.py` holds every term the table retires, in the
whole text of each file, to the file's count, while this one holds its terms at zero in prose outside the Lean) and the
word maps of `tools/move/`, which name the words they retire; and of a `.lean` file only its comments and docstrings
(``lean_prose``, the lane-text lint's reader), where every site is held to its file's count in ``LEAN_LIST``: a file past
its count, or with a site and no entry, fails, and the ratchet gate holds the list's total from rising, so it only shrinks
(``--prune`` lowers it to what the tree holds). A term matches case-insensitively as two words, hyphenated, or wrapped onto
the next line (behind a comment marker or not). Every file under `verity/protocols/inference_only/rho/`, `benchmarks/network_traces/` and the
network tests retires more (``INCORRECT_TERMS``): its proof blocks, their steps, outputs and windows, and K's terms for
them are incorrect, never faulty or wrong, while a faulty node, link or device keeps the word. Identifiers (`proof_unit`,
a key that mirrors a Lean name such as `faulty_bits`) are code; the retired-terms table
(`GLOSSARY.md`) names them. What may stay: the table's own rows (``RETIRED_ROWS``) and a spelling baked into a
record or a digest (``ALLOWED``, one reason each; an entry that matches nothing fails too).

The Lean's declaration names get a pass of their own (``LEAN_WORDS``): a name that carries one of the Glossary's older
names keeps it until it is renamed, and each word's count of names only falls (``RETIRED_NAMES``). SPOT's older name,
sampled proofs, is held by count in every scanned file's whole text (``SPOT_NAMES``): its words until the prose move
(`tools/move/spot_term_map.toml`) and the strings' owners rename them, and the code's names until SPOT's code move does.

The maintained docs (every tracked Markdown file but a CHANGELOG) retire more (``DOC_TERMS``:
"block" is reserved for parts of circuits, a proof block is incorrect or correct, never faulty, the subprotocols go by
their names, and "leg", the dampener, NCI, C-Flock, the netlist, M0, IR and the verifier of record are retired). Their code
keeps the old words until its owners rename it, so this scan skips code spans and fences, where a path (`legs/`) or a
record key stays, and a term written as a quoted mention ("leg"); the table's rows name them too. A retired command
(`verity example`) keeps no path or key, so it is found in code spans and fences as well.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter

from lint.test_repo_lane_text import lean_prose
from test_repository import ROOT, _tracked
from test_lean_packages import RETIRED_NAMES, _held, declarations

#: Each retired term and what to write instead.
TERMS = {"proof unit": "proof block", "faulty unit": "incorrect proof block", "proof service": "proof system",
         "faulty ANDs": "incorrect ANDs"}
_SEP = r"(?:[ \t]+|-|[ \t]*\n[ \t]*(?:#+|//+|--|\*|;+)?[ \t]*)"
_TERM = re.compile(rf"\b(?:(?:proof|faulty){_SEP}units?|proof{_SEP}services?|faulty{_SEP}ANDs)\b", re.I)

#: Retired in every file under INCORRECT too, and what to write instead.
INCORRECT = ("verity/protocols/inference_only/rho/", "benchmarks/network_traces/", "verity/protocols/certifier/verifier/network/tests/")
INCORRECT_TERMS = {"faulty outputs, faulty step, faulty ingress unit, faulty block, faulty set":
                   "incorrect outputs, incorrect step, incorrect ingress unit, incorrect proof block, the incorrect set",
                   "faulty window, wrong unit": "incorrect window, incorrect proof block",
                   "faulty term, faulty bits": "the term for incorrect proof blocks, their bits"}
_INCORRECT_TERM = re.compile(rf"\b(?:faulty{_SEP}(?:outputs?|steps?|ingress{_SEP}units?|(?:proof{_SEP})?blocks?|sets?"
                             rf"|windows?|terms?|bits)|wrong{_SEP}units?)\b", re.I)

#: The retired-terms table, the Glossary's older names (`ARCHITECTURE.md` links it).
GLOSSARY = "GLOSSARY.md"
#: The table's rows that name the terms, in GLOSSARY.
RETIRED_ROWS = ('| "proof unit"', '| "wrong unit"', '| "proof service"', '| "declared advice"', '| "audit coin"')
#: The Lean's sites, a count per file, which only shrinks.
LEAN_LIST = ROOT / "tests/lint/allowlists/lean_retired_terms.json"

#: Retired in the maintained docs' prose, and what to write instead.
DOC_TERMS = {
    "wrong unit, incorrect unit": "incorrect proof block",
    "faulty proof block, faulty block, faulty set": "incorrect proof block; the incorrect set F",
    "isolation unit, IU": "inference block",
    "replay unit": "replay block",
    "draw, drawn, redraw": "the sample, sampled proof block (verb: sample; coins are flipped)",
    "verity example": "verity demo",
    "components, the composite's product": "subprotocols, composition",
    "firewall": "ZK server",
    "warden, network warden": "the network certifier (`warden.py`, `WardenParams` and the `warden_*` modules stay in backticks "
                              "until renamed)",
    "flock-verify, remote check, naming the verifier": "verity verify proof, verity verify remote (a command someone runs today "
                                                      "stays in backticks)",
    "coin server, coin source, coin device, randomness source": "the challenger",
    "salted leaf, salted commitment": "hiding leaf, hiding commitment",
    "acceptance probability P_a, a sampling design's acceptance probability": "miss probability (detection is its "
                                                                              "complement; a prover's acceptance probability "
                                                                              "keeps the word)",
    "egress cone": "reach (E)",
    "link window": "an execution phase's transfer blocks (`link-window` stays in backticks until a format version)",
    "block-scaled, blockwise, scale block": "scale group",
    "production": "serving, or proving",
    "builder, advice compiler, circuit builder": "registered compiler",
    "declared advice": "advice",
    "declared circuit": "registered circuit",
    "declared work, declared workload, declares its advice": "registered circuit; the developer reports its advice, which "
                                                             "the registered compiler turns into the execution block",
    "registered work": "registered circuit (D)",
    "the prover's node, machine, host or site, a machine the prover doesn't control, nothing the prover runs or hosts, "
    "prover-controlled, the prover system":
        "the developer's node, machine, host or site, a machine the developer doesn't control, nothing the developer runs "
        "or hosts, developer-controlled, the developer's hardware (\"prover\" stays for the proving side of a proof: "
        "Flock's prover, a proof's messages, a cheating prover)",
    "onsite deployment, remote deployment, composite protocol": "configuration (onsite or remote, the Lean's `Composite`)",
    "named premise, the card premise": "assumption, or a proposed assumption until a lock lists it; the card assumption (a "
                                       "theorem's hypothesis stays a premise)",
    "sampled unit, unit class": "sampled proof block; a size class, or Flock's proof-block class",
    "C-Flock": "Flock (Succinct's is upstream Flock)",
    "netlist": "a private circuit, or the circuit's description",
    "M0": "Flock's circuit prover",
    "IR, Verity IR": "circuit or Program (`verity.circuits`)",
    "verifier of record": "the verifier (other checking code is named for what it is: the Rust cross-check, a Python mirror)",
    "audit coin": "the challenger's coins (the step that reads them is the verification phase)",
    "declaration block, declaration period": "execution block, execution phase",
    "uplink tap, the auditor's tap": "the network certifier, and its records network certificates",
    "timed verifier": "the memory challenger",
    "network transparency": "network accounting",
    "prover gateway, verifier gateway": "the ZK server; the challenger",
    "batch verifier": "the verifier",
    "GPU workers": "workers",
    "escape probability": "miss probability",
    "post-landing run, postland": "after-landing run",
    "Commit window": "Commit period",
    "accounting period": "the monitoring period",
    "isolation block, certified block": "inference block",
    "transformer block": "layer",
    "audit phase, audit round": "the verification phase",
    "monitored compute, monitored hardware, monitored server": "registered hardware (the monitoring period keeps its name)",
    "staging decode": "the store's load",
    "pinned theorem, pinned statement": "guarantee",
    "trusted text": "spec",
    "sampling period, sampling block, PoUW's commit period": "execution phase (work plus its audit: an execution phase and its "
                                                            "verification phase)",
    "hm96 rows, outside the commitments' docs": "hiding commitments",
    "integrity period": "execution phase (`IntegrityPeriod` stays in backticks until renamed)",
    "execution transcript": "trace (a proof's or a session's messages stay its transcript)",
    "that period's coins": "before any coins are flipped",
    "rejection dominates, reject dominance": "any rejection rejects the whole",
    "physical block": "server block",
    "thread block": "CTA",
    "KV block": "KV page",
    "the accelerator model, the machine model": "the ACCEL model (`benchmarks/machine_model/` and its `machine-model` "
                                                "command keep the name)",
    "coverage premise, coverage table": "the card assumption, the opcode table",
    "envelope ISA, envelope op": "the ACCEL ops",
    "server-time block, serving block, server-window": "server block",
    "overhead lower bound, exclusion bound": "the slowdown lower bound",
    "circuit-size slowdown, training efficiency, hardware multiple": "slowdown(𝒢, 𝒜, η)",
    "per-block cap, per-block limit": "the server-block yield q_S, or the inference-block yield q(c, k)",
    "training class": "pricing argument",
    "physical class, restricted class, monitored class": "the feasible set 𝓕; the admissible set 𝒜",
    "certified work, accounted compute, declared compute": "verified compute, V",
    "certified share, declared utilization": "utilization",
    "uncertified capacity, uncertified gates, undeclared compute, skippable work, residual compute, monitored residual":
        "free compute, U",
    "dark compute, dark fleet, dark leaves, undeclared hardware": "unregistered hardware, ε_unreg",
    "unregistered input, unregistered information, novel information": "fresh input, fresh input bits",
    "registered computation": "the registered circuit",
    "work requirement": "necessary subcomputation",
    "dampener, compute dampener": "the slowdown, slowdown(𝒢, 𝒜, η) (`Nci.Dampener` and `Dampener*` stay in backticks until renamed)",
    "NCI, novel communication intensity": "the inference-only policy (`Nci*`, `NciCheck` and the `nci` directories stay in backticks until "
                                          "renamed)",
    "leg": "the subprotocol's name (SPOT, PoUW, PoUS, PoUB, network certification), the bandwidth audit's or the "
           "memory wipe's, or run",
}
_SPACE = r"(?:[ \t]+|-|[ \t]*\n[ \t]*)"
_DOC_TERM = re.compile(rf"(?i:\b(?:(?:wrong|incorrect|isolation|replay){_SPACE}units?|(?:physical|thread|KV){_SPACE}blocks?"
                       rf"|wardens?"
                       rf"|legs?|(?:re)?(?:draws?|drawn|drawing|drew)|components?|firewall(?:s|ed)?|flock-verify"
                       rf"|coin{_SPACE}(?:servers?|sources?|devices?)|randomness{_SPACE}sources?"
                       rf"|salted{_SPACE}(?:leaf|leaves|commitments?)|egress{_SPACE}cones?"
                       rf"|acceptance{_SPACE}probabilit(?:y|ies){_SPACE}\$?(?:\\bar\{{?\s?)?P\u0304?\}}?_\{{?a\b"
                       rf"|(?:sampling{_SPACE})?design's{_SPACE}acceptance{_SPACE}probabilit(?:y|ies)"
                       rf"|link{_SPACE}windows?|block{_SPACE}scaled|blockwise|scale{_SPACE}blocks?|productions?"
                       rf"|remote{_SPACE}check|builders?|advice{_SPACE}compilers?|declared{_SPACE}(?:advice|circuits?|work(?:loads?)?)"
                       rf"|declar(?:es|ed|ing){_SPACE}(?:its|their|the){_SPACE}advice|sampled{_SPACE}units?"
                       rf"|integrity{_SPACE}periods?|execution{_SPACE}transcripts?"
                       rf"|units?{_SPACE}class(?:es)?|audit{_SPACE}coins?"
                       rf"|registered{_SPACE}work|(?:named|card){_SPACE}premises?|(?:onsite|remote){_SPACE}deployments?|composite{_SPACE}protocols?"
                       rf"|prover{_SPACE}systems?|prover-controlled|prover{_SPACE}runs{_SPACE}or{_SPACE}hosts"
                       rf"|prover's{_SPACE}(?:own{_SPACE})?(?:nodes?|machines?|hosts?|sites?|datacenters?|servers?|GPUs?|hardware)"
                       rf"|prover{_SPACE}(?:doesn't|does{_SPACE}not){_SPACE}control|(?:hosts?|machines?){_SPACE}the{_SPACE}prover{_SPACE}controls"
                       rf"|faulty{_SPACE}(?:proof{_SPACE})?blocks?|faulty{_SPACE}sets?"
                       rf"|declaration{_SPACE}(?:blocks?|periods?)|sampling{_SPACE}(?:periods?|blocks?)|hm96{_SPACE}rows?"
                       rf"|(?:accelerator|machine){_SPACE}models?|coverage{_SPACE}(?:premises?|tables?)|envelope{_SPACE}(?:ISA|ops?)"
                       rf"|server{_SPACE}time{_SPACE}blocks?|serving{_SPACE}blocks?|server-windows?"
                       rf"|(?:overhead{_SPACE}lower|exclusion){_SPACE}bounds?|circuit{_SPACE}size{_SPACE}slowdown"
                       rf"|training{_SPACE}efficiency|hardware{_SPACE}multiples?|per{_SPACE}block{_SPACE}(?:caps?|limits?)"
                       rf"|(?:training|physical|restricted|monitored){_SPACE}class(?:es)?|certified{_SPACE}(?:work|shares?)"
                       rf"|accounted{_SPACE}compute|declared{_SPACE}(?:compute|utilization)|uncertified{_SPACE}(?:capacity|gates)"
                       rf"|undeclared{_SPACE}(?:compute|hardware)|skippable{_SPACE}work|residual{_SPACE}compute|monitored{_SPACE}residuals?"
                       rf"|dark{_SPACE}(?:compute|fleets?|leaves|hardware)|unregistered{_SPACE}(?:inputs?|information)"
                       rf"|novel{_SPACE}information|registered{_SPACE}computations?|work{_SPACE}requirements?"
                       rf"|period's{_SPACE}(?:audit{_SPACE})?coins|reject(?:ion)?{_SPACE}domina\w*|dampeners?"
                       rf"|uplink{_SPACE}taps?|auditor's{_SPACE}taps?"
                       rf"|timed{_SPACE}verifiers?|network{_SPACE}transparency|(?:prover|verifier){_SPACE}gateways?"
                       rf"|batch{_SPACE}verifiers?|GPU{_SPACE}workers?|escape{_SPACE}probabilit(?:y|ies)"
                       rf"|post{_SPACE}landing(?:{_SPACE}runs?)?|postland\w*|commit{_SPACE}windows?|accounting{_SPACE}periods?"
                       rf"|(?:isolation|certified|transformer){_SPACE}blocks?|audit{_SPACE}(?:phases?|rounds?)"
                       rf"|monitored{_SPACE}(?:compute|hardware|servers?)|staging{_SPACE}decodes?"
                       rf"|pinned{_SPACE}(?:theorems?|statements?)|trusted{_SPACE}texts?"
                       rf"|novel{_SPACE}communication{_SPACE}intensity)\b)|\bIUs?\b|\bNCI\b"
                       rf"|(?i:\bC-Flock\b|\bnetlists?\b|\bverifiers?{_SPACE}of{_SPACE}record\b)|\bM0\b|\bIRs?\b"
                       rf"|\bcommit{_SPACE}periods?\b")
_DOC_COMMAND = re.compile(r"\bverity[ \t]+(?:bench\|)?example\b")
#: The commitments' own docs, where hm96 rows keep their name.
HM96_HOME = "verity/commitments/"
_CODE = re.compile(r"^[ \t]*(`{3,}|~{3,}).*?^[ \t]*\1[^\n]*$|(`+)[\s\S]*?(?<!`)\2(?!`)", re.M | re.S)

ALLOWED: dict[str, str] = {
    "N, the window's proof units": "a public item's reason in A5's Spec (`benchmarks/one_stage/a5.py`): `Spec.digest` hashes it",
    "N, the round's proof units": "a public item's reason in scan_zk's Spec (`benchmarks/network_traces/scan_zk.py`): "
                                  "`Spec.digest` hashes it",
    "one proof unit per registration: the link-window": "a public item's reason in N2's Spec "
                                                        "(`certifier.service`): `Spec.digest` hashes it",
    "the client's request commitment, the randomness source": "a line of GLOSSARY.md that a queued PR rewrites; renamed "
                                                              "after it lands",
    "production would publish every event": "infra's Vercel deployment, another sense",
    "Production is `verity-slack-events`": "infra's Vercel deployment, another sense",
    "T proof units per registration, unit t the link-window's bucket row t": "a public item's reason in the ingress tie's "
                                                                             "Spec (`certifier.tie_service`): `Spec.digest` hashes it",
    '"proof-unit"': "the kind circuit-check's Boolean export records for a unit circuit (`boolean_export.Registry.type_of`)",
    "verity/proof-service/": "the stream's and outcome's record formats (`sampled_proofs.service.STREAM`, `OUTCOME`), "
                             "which their digests hash",
    "report-proof-service-": "a cited note's id",
    "Faulty ANDs (`Stratified.harm_bound`": "`sampling_plan/h200.py`'s docstring: a queued PR on next rewrites the line beside "
                                            "it; renamed after it lands",
    "no wrong certificate, faulty block or detection": "a comment beside `Remote.CITES` in `verity/protocols/inference_only/rho/rho/envelopes.py`, "
                                                       "which a queued PR on next rewrites; renamed after it lands",
    "no faulty proof block, every one being drawn": "`benchmarks/network_traces/leak_bounds.py`'s docstring: a queued PR on "
                                                    "next rewrites the line beside it; renamed after it lands",
    "a tied audit no faulty term (its wrong units' outputs are the egress's)": "the network period check's refusal in "
        "`period_case.py`, which mirrors `NetworkCheck.Period`'s Lean message word for word; renamed with the Lean",
    "K charges a tied audit no faulty term": "the refusal `test_network_period_protocol.py` expects from Lean "
                                             "(`NetworkCheck.Period`); renamed with the Lean",
    "an *active warden* in network steganography's sense": "network steganography's own term, which `ARCHITECTURE.md` "
                                                           "names to say it is not the network certifier's name",
    "where a passive warden only watches": "network steganography's own term, beside the one above",
    "Verity's two composite protocols": "`verity/protocols/remote/verifier/composition/README.md`, in a Lake package's "
                                        "directory, which the protocols' word move rewrites; renamed with it",
}

#: the lints that name the terms to find them: this one, and the ratchet over every term `GLOSSARY.md`'s table retires
LINTS = ("tests/test_retired_terms.py", "tests/lint/test_repo_retired_terms.py")
#: the word maps a move renames by, which name the words they retire
MAPS = "tools/move/"


def scanned(files: list[str]) -> list[str]:
    """Every file but the Lean packages' configs and locks, the LINTS and the word maps."""
    return sorted(f for f in files if not f.endswith(("lean-audit.json", "lakefile.toml"))
                  and f not in LINTS and not f.startswith(MAPS))


_CODE_CHAR = re.compile(r"\S")


def _lean_prose(text: str) -> str:
    """A Lean file's comments and docstrings where they stand, and its code's non-blank characters NUL, so that a term
    wraps from one comment line onto the next but never through code."""
    starts = [0] + [m.end() for m in re.finditer("\n", text)]
    out, at = [], 0
    for line, chunk in lean_prose(text):
        start = text.index(chunk, max(at, starts[line - 1]))
        out += [_CODE_CHAR.sub("\0", text[at:start]), chunk]
        at = start + len(chunk)
    return "".join(out) + _CODE_CHAR.sub("\0", text[at:])


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
        incorrect = f.startswith(INCORRECT)
        if text is None or not ("unit" in text.lower() or "service" in text.lower() or "faulty" in text.lower()):
            continue
        if f.endswith(".lean"):
            text = _lean_prose(text)
        allowed = [(m.start(), m.end(), s) for s in ALLOWED for m in re.finditer(re.escape(s), text)]
        lines = text.splitlines()
        terms = (_TERM, _INCORRECT_TERM) if incorrect else (_TERM,)
        for m in sorted((m for t in terms for m in t.finditer(text)), key=lambda m: m.start()):
            n = text.count("\n", 0, m.start()) + 1
            if f == GLOSSARY and lines[n - 1].startswith(RETIRED_ROWS):
                continue
            if spelling := next((s for a, b, s in allowed if a <= m.start() and m.end() <= b), None):
                used.add(spelling)
                continue
            out.append((f, n, " ".join(m[0].split())))
    return out, used


def doc_offenders(files: list[str], read=_read) -> tuple[list[tuple[str, int, str]], set[str]]:
    """(file, line, the term as written) for each of DOC_TERMS in a maintained doc's prose, and the ALLOWED spellings used."""
    out, used = [], set()
    for f in scanned(files):
        if not f.endswith(".md") or f.rsplit("/", 1)[-1] == "CHANGELOG.md" or (text := read(f)) is None:
            continue
        prose = _CODE.sub(lambda m: re.sub(r"[^\n]", " ", m[0]), text)
        allowed = [(m.start(), m.end(), s) for s in ALLOWED for m in re.finditer(re.escape(s), text)]
        lines = text.splitlines()
        for m in sorted([*_DOC_TERM.finditer(prose), *_DOC_COMMAND.finditer(text)], key=lambda m: m.start()):
            n = text.count("\n", 0, m.start()) + 1
            quoted = text[m.start() - 1:m.start()] in ('"', "“") and text[m.end():m.end() + 1] in ('"', "”")
            if quoted or (f == GLOSSARY and lines[n - 1].startswith("|")):
                continue
            if f.startswith(HM96_HOME) and m[0].lower().startswith("hm96"):
                continue
            if spelling := next((s for a, b, s in allowed if a <= m.start() and m.end() <= b), None):
                used.add(spelling)
                continue
            out.append((f, n, " ".join(m[0].split())))
    return out, used


def lean_allowed() -> Counter:
    """Each Lean file's allowed sites, from LEAN_LIST."""
    doc = json.loads(LEAN_LIST.read_text(encoding="utf-8"))
    counts = Counter({e["file"]: e["count"] for e in doc["entries"]})
    assert len(counts) == len(doc["entries"]) and all(type(n) is int and n >= 1 for n in counts.values()), (
        f"{LEAN_LIST.name}: each entry is one file's {{file, count}}, its count a positive int")
    return counts


def over_lean_list(found: list[tuple[str, int, str]], have: Counter) -> list[str]:
    """Each Lean file holding more sites than ``have`` allows, with its sites."""
    lean = Counter(f for f, *_ in found if f.endswith(".lean"))
    return [f"  {f}: {n} found, {have[f]} allowed\n" + "\n".join(f"    {f}:{line}: {term}" for g, line, term in found if g == f)
            for f, n in sorted(lean.items()) if n > have[f]]


def test_no_retired_term_in_prose():
    found, used = offenders(_tracked())
    bad = [o for o in found if not o[0].endswith(".lean")]
    say = ("say " + " and ".join(f'"{new}" for "{old}"' for old, new in TERMS.items())
           + f" (in {', '.join(INCORRECT)}, also " + " and ".join(f'"{new}" for "{old}"' for old, new in INCORRECT_TERMS.items())
           + ")" + f' (ARCHITECTURE.md, Glossary, "Block"; {GLOSSARY})')
    assert not bad, (
        f"a retired term: {say}; a spelling baked into a record or a digest goes in ALLOWED with its reason:\n"
        + "\n".join(f"  {f}:{n}: {term}" for f, n, term in bad))
    over = over_lean_list(found, lean_allowed())
    assert not over, (
        f"a retired term in a Lean comment or docstring, past {LEAN_LIST.relative_to(ROOT)}, which only shrinks: {say}, "
        f"and once a file holds fewer, `python3 tests/test_retired_terms.py --prune` lowers its count:\n" + "\n".join(over))
    unused = set(ALLOWED) - used - doc_offenders(_tracked())[1]
    assert not unused, f"ALLOWED entries that match nothing any more: {sorted(unused)}"


def test_no_retired_term_in_the_maintained_docs():
    bad, _ = doc_offenders(_tracked())
    assert not bad, (
        "a retired term in a maintained doc: say " + "; ".join(f'{new} for "{old}"' for old, new in DOC_TERMS.items())
        + f" (ARCHITECTURE.md's Glossary, {GLOSSARY}). A path or a record key that keeps it goes in backticks, and a word"
        + " in another sense goes in ALLOWED with its reason:\n" + "\n".join(f"  {f}:{n}: {term}" for f, n, term in bad))


def _normal(name: str) -> str:
    return re.sub(r"^(?:(?:the|a|an|or|and)\s+)+", "", " ".join(name.lower().split()))


def older_names(glossary: str) -> set[str]:
    """Each name GLOSSARY's Older names table quotes in its first column: not one qualified by its sense (`"block" for a
    tile`, `"unit" alone meaning one`), nor one it keeps (`"stage", outside "one-stage"`)."""
    table = re.search(r"^## Older names\n(.*?)(?=^## |\Z)", glossary, re.M | re.S)[1]
    names = set()
    for cell in (re.split(r"(?<!\\)\|", line)[1] for line in table.splitlines() if line.startswith("| ")):
        for m in re.finditer(r'"([^"]*)"', cell):
            if not (cell[m.end():].startswith((" for ", " meaning ", " alone ")) or cell[:m.start()].endswith("outside ")):
                names.add(_normal(m[1]))
    return names


def stale_replacements(tables: dict[str, str], old: set[str]) -> list[tuple[str, str]]:
    """(row, the name it prescribes) for each replacement in ``tables`` that is one of ``old``."""
    return [(row, piece) for row, new in tables.items() for piece in map(_normal, re.split(r"[,;:()]", new)) if piece in old]


def test_no_replacement_is_an_older_name():
    old = older_names((ROOT / GLOSSARY).read_text(encoding="utf-8"))
    stale = stale_replacements({**TERMS, **INCORRECT_TERMS, **DOC_TERMS}, old)
    assert not stale, (f"a row prescribes a name {GLOSSARY}'s Older names table retires: write that table's Term instead:\n"
                       + "\n".join(f'  "{row}": "{name}"' for row, name in stale))


def test_the_older_names_are_read_from_the_table():
    """On a toy Glossary: a row that prescribes an older name (`"isolation block"`, `"server-time block"`, one piece of a
    longer replacement) is found, and a sense-qualified or kept name isn't an older name."""
    toy = ('## Older names\n\n| Old name | Term |\n|---|---|\n| "isolation unit" (IU), "isolation block" | inference block |\n'
           '| "server-time block", "window" for a server\'s gates, \\|C\\| | server block |\n'
           '| "stage", outside "one-stage" | phase |\n| "unit" alone meaning one | proof block |\n\n## Next\n\n| "leg" | x |\n')
    assert older_names(toy) == {"isolation unit", "isolation block", "server-time block", "stage"}
    assert stale_replacements({"isolation unit, IU": "isolation block", "physical block": "the server-time block",
                               "replay unit": "replay block", "unit": "a phase, or the stage (a one-stage run)"},
                              older_names(toy)) == [("isolation unit, IU", "isolation block"),
                                                    ("physical block", "server-time block"), ("unit", "stage")]


def test_the_glossary_records_the_retired_terms():
    rows = [line for line in (ROOT / GLOSSARY).read_text(encoding="utf-8").splitlines() if line.startswith(RETIRED_ROWS)]
    assert len(rows) == len(RETIRED_ROWS), f"{GLOSSARY}'s retired-terms table has no row starting {RETIRED_ROWS}"
    assert all(_TERM.search(row) or _DOC_TERM.search(row) for row in rows), (
        f"{GLOSSARY}'s rows {RETIRED_ROWS} no longer name the terms")


#: the Glossary's older names (GLOSSARY.md) a Lean declaration's own name may carry, each word's forms and its term;
#: `session`, `window` and `block` are left out, as each has a sense the Glossary keeps
LEAN_WORDS = {"draw": (("draw", "draws", "drawn"), "sample"), "law": (("law", "laws"), "sampling design"),
              "escape": (("escape",), "miss probability"), "wrong": (("wrong",), "incorrect"),
              "dampener": (("dampener",), "slowdown"), "firewall": (("firewall",), "ZK server"),
              "horizon": (("horizon",), "monitoring period"), "warden": (("warden",), "network certifier"),
              "tap": (("tap", "taps"), "network certifier"), "leg": (("leg", "legs"), "subprotocol")}


def words(name: str) -> list[str]:
    """``name``'s words, lowercased: split at dots, underscores and primes, then at case (`drawCoins_ok` is draw, coins
    and ok; `HTTPLaw` is http and law)."""
    return [w.lower() for part in re.split(r"[._']", name) for w in re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+", part)]


def retired_names(files: dict[str, str]) -> dict[str, list[str]]:
    """{word of LEAN_WORDS: `FILE:LINE NAME` for each declaration of ``files`` ({file: its source}) whose name, as the
    declaration writes it, carries the word}."""
    forms = {f: w for w, (fs, _) in LEAN_WORDS.items() for f in fs}
    out: dict[str, list[str]] = {w: [] for w in LEAN_WORDS}
    for f, src in sorted(files.items()):
        for d in declarations(src):
            for w in sorted({forms[x] for x in words(d.written) if x in forms}):
                out[w].append(f"{f}:{d.line} {d.written}")
    return out


def test_no_retired_word_in_a_new_lean_name():
    """A Lean name says the Glossary's term, not an older name: an existing name keeps its word until it is renamed (with
    `--moved` where a lock lists it), and each word's count only falls (`RETIRED_NAMES`, `test_lean_packages.py`)."""
    files = {f: text for f in _tracked() if f.endswith(".lean") and (text := _read(f)) is not None}
    found = retired_names(files)
    _held("RETIRED_NAMES", RETIRED_NAMES, {w: len(v) for w, v in found.items()}, "the Lean names with a retired word",
          "name the new one with the Glossary's term instead (" + "; ".join(f"{t} for {w}" for w, (_, t) in LEAN_WORDS.items())
          + "; GLOSSARY.md, Older names): `git diff -U0 origin/main -- '*.lean'` shows it")


#: SPOT's older name (GLOSSARY.md, Older names): each spelling's count in the scanned files' whole text, which only falls
#: (`_held`). A sampled proof block, and its proof, keep their words and aren't counted where "block" follows.
SPOT_NAMES = {"sampled proof(s)": 435, "sampled_proofs": 62, "SampledProofs": 15, "sampled-proofs": 507}
_GAP = r"(?:[ \t]+|[ \t]*\n[ \t]*(?:#+|//+|--|\*|;+)?[ \t]*)"
SPOT_SPELLINGS = {"sampled proof(s)": re.compile(rf"\bsampled{_GAP}proofs?\b(?!{_GAP}blocks?\b)", re.I),
                  "sampled_proofs": re.compile("sampled_proofs", re.I), "SampledProofs": re.compile("[Ss]ampledProofs"),
                  "sampled-proofs": re.compile("sampled-proofs", re.I)}


def spot_names(files: list[str], read=_read) -> dict[str, int]:
    """Each of SPOT_SPELLINGS' count in the scanned ``files``."""
    out = dict.fromkeys(SPOT_SPELLINGS, 0)
    for f in scanned(files):
        if (text := read(f)) is not None and "sampled" in text.lower():
            for k, spelling in SPOT_SPELLINGS.items():
                out[k] += len(spelling.findall(text))
    return out


def test_spot_s_older_name_only_falls():
    """The sampled proof system is SPOT: its older name's count in every spelling only falls."""
    _held("SPOT_NAMES", SPOT_NAMES, spot_names(_tracked()), "SPOT's older name, sampled proofs,",
          "write SPOT in prose and strings, `spot` in a new path or Python name and `Spot` in a new Lean name (GLOSSARY.md, "
          "Older names): `git diff -U0 origin/main | grep -i -E 'sampled.?proofs'` shows it")


def test_the_spot_scan_reads_each_form():
    """On toy files: the words count in either case and wrapped behind a comment marker, a sampled proof block doesn't,
    and each code spelling counts under its own key, never under the words."""
    toy = {"a.py": "# Sampled proofs' verifier, a sampled proof, the sampled\n# proofs; sampled proof blocks, a sampled proof block\n",
           "b.lean": "import SampledProofs\n/-- `sampledProofsSound_of`, `core_sampled_proofs`, `SAMPLED_PROOFS` -/\n",
           "c.toml": 'id = "verity/sampled-proofs/v1"\n'}
    assert spot_names(list(toy), read=toy.get) == {"sampled proof(s)": 3, "sampled_proofs": 2, "SampledProofs": 2,
                                                   "sampled-proofs": 1}


def test_the_lean_name_scan_reads_each_form():
    """The words of a name, on toy files: a form at a case or underscore boundary counts, inside a word it doesn't
    (`drawer`, `lawful`, `tapir`, `legal`), a comment's name doesn't, and a private name does."""
    assert words("drawCoins_ok") == ["draw", "coins", "ok"] and words("HTTPLaw.mk'") == ["http", "law", "mk"]
    src = ("namespace Pouw.Draw\n/-- A drawn sample. -/\ndef drawnUnits : Nat := 1\nprivate theorem wrong_set : True := trivial\n"
           "-- def lawOf : Nat := 0\ndef drawer : Nat := 0\nstructure Lawful where\ndef tapir : Nat := 0\n"
           "def legal : Nat := 0\nabbrev TapLegs : Nat := 0\nend Pouw.Draw\n")
    got = {w: v for w, v in retired_names({"a.lean": src}).items() if v}
    assert got == {"draw": ["a.lean:3 drawnUnits"], "wrong": ["a.lean:4 wrong_set"], "tap": ["a.lean:10 TapLegs"],
                   "leg": ["a.lean:10 TapLegs"]}


def test_the_scan_reads_each_form():
    """The rules above, on toy files."""
    docs = {
        "a.py": '"""A proof unit, two Proof-Units and a proof\n    units wrap."""\nx.proof_unit  # a faulty unit; unit tests\n',
        "b.rs": "/// the proof\n/// unit of it; a bulletproof unit\nlet reason = \"N, the round's proof units\";\n",
        "e.lean": ('def s := "a proof unit"  -- the proof\n  -- unit, wrapped\n/-- a faulty\n  unit -/ def proof_unit := 1\n'
                   "-- a proof\ndef unit := 2\n"),
        "d.py": '# the Proof-Service and a proof\n# services; proof_service, verity/proof-service/stream/v0; a faulty service\n',
        "c.lean": "-- a proof unit",
        GLOSSARY: '| "proof unit", and "unit" alone meaning one | proof block |\n| other | a proof unit |\n',
    }
    found, used = offenders(list(docs), read=docs.__getitem__)
    assert found == [(GLOSSARY, 2, "proof unit"), ("a.py", 1, "proof unit"), ("a.py", 1, "Proof-Units"),
                     ("a.py", 1, "proof units"), ("a.py", 3, "faulty unit"), ("b.rs", 1, "proof /// unit"),
                     ("c.lean", 1, "proof unit"), ("d.py", 1, "Proof-Service"), ("d.py", 1, "proof # services"),
                     ("e.lean", 1, "proof -- unit"), ("e.lean", 3, "faulty unit")]
    assert over_lean_list(found, Counter({"c.lean": 1, "e.lean": 2})) == []
    assert over_lean_list(found, Counter({"c.lean": 1, "e.lean": 1})) == [
        "  e.lean: 2 found, 1 allowed\n    e.lean:1: proof -- unit\n    e.lean:3: faulty unit"]
    assert used == {"N, the round's proof units", "verity/proof-service/"}


def test_the_scan_reads_rho_terms():
    """`verity/protocols/inference_only/rho`'s incorrect outputs, steps and ingress units, on toy files: its keys and Lean-mirroring names,
    a faulty node, link or device, and the same words outside it stay."""
    docs = {
        "verity/protocols/inference_only/rho/a.py": ('"""The faulty-outputs channel: each faulty\n    step; a faulty ingress unit and the faulty '
                                'set."""\nx = s["exfil.faulty"], c.faulty_unit_bits  # faultyIngressBits; a faulty node, '
                                "link or device\n"),
        "verity/protocols/inference_only/rho/data/b.toml": "# Faulty outputs, a faulty proof block, two faulty blocks\n[incorrect]\n",
        "c.py": "# faulty outputs and faulty blocks outside it\n",
    }
    assert offenders(list(docs), read=docs.__getitem__)[0] == [
        ("verity/protocols/inference_only/rho/a.py", 1, "faulty-outputs"), ("verity/protocols/inference_only/rho/a.py", 1, "faulty step"),
        ("verity/protocols/inference_only/rho/a.py", 2, "faulty ingress unit"), ("verity/protocols/inference_only/rho/a.py", 2, "faulty set"),
        ("verity/protocols/inference_only/rho/data/b.toml", 1, "Faulty outputs"), ("verity/protocols/inference_only/rho/data/b.toml", 1, "faulty proof block"),
        ("verity/protocols/inference_only/rho/data/b.toml", 1, "faulty blocks")]


def test_the_scan_reads_the_network_terms():
    """The network tests' and `benchmarks/network_traces`' incorrect proof blocks, windows and K's terms for them, on toy
    files: a key or a name that mirrors the Lean's (`faulty_bits`, `wrong_units`, `faultySets`), a faulty certifier, a
    wrong verdict, and the same words outside them stay."""
    net = "verity/protocols/certifier/verifier/network/tests/a.py"
    docs = {
        net: ('"""Its faulty bits, no faulty\n    term and at most 3 wrong units."""\n'
              'class _Faulty(Certifier):  # a faulty certifier; a wrong verdict\n'
              'k = t["faulty_bits"] + t["wrong_units"]  # faultySets\n'),
        "benchmarks/network_traces/b.md": "Each link pays its faulty proof blocks; none of its faulty-windows drawn.\n",
        "c.py": "# faulty bits and wrong units outside them\n",
    }
    assert offenders(list(docs), read=docs.__getitem__)[0] == [
        ("benchmarks/network_traces/b.md", 1, "faulty proof blocks"), ("benchmarks/network_traces/b.md", 1, "faulty-windows"),
        (net, 1, "faulty bits"), (net, 1, "faulty term"), (net, 2, "wrong units")]


def test_the_scan_reads_the_incorrect_ands():
    """`sampling_plan`'s incorrect ANDs, on toy files, in every file: an AND gate in another sense stays."""
    docs = {"a.py": "# 9 faulty ANDs and faulty\n# ANDs; a faulty AND gate, faulty outputs outside rho\n"}
    assert offenders(list(docs), read=docs.__getitem__)[0] == [("a.py", 1, "faulty ANDs"), ("a.py", 1, "faulty # ANDs")]


def test_the_doc_scan_reads_the_trace_and_phase_terms():
    """A run's trace and the execution phase, on toy files: a proof's transcript and `IntegrityPeriod` stay."""
    docs = {"a.md": "An integrity period, two integrity\nperiods and an execution-transcript; a proof's transcript, "
                    "`IntegrityPeriod`.\n"}
    assert doc_offenders(list(docs), read=docs.__getitem__)[0] == [
        ("a.md", 1, "integrity period"), ("a.md", 1, "integrity periods"), ("a.md", 2, "execution-transcript")]


def test_the_doc_scan_reads_each_form():
    """The maintained-docs scan, on toy files: prose only, in Markdown only."""
    docs = {
        "a.md": "A wrong unit, two Isolation-Units and a replay\nunit; the IU and IUs; a KV block.\n"
                "The legs ran; each leg's verdict; a three-leg gate. Not legacy, alleged or IUD.\n",
        "b.md": 'The `legs/pous.py` file and `wipe_period.leg(dir)`; ``a `leg` key``; "leg" is retired.\n'
                "```sh\nLEGS=A,B ./run.sh  # each leg\n```\n~~~\na leg\n~~~\nA thread block after the fences.\n"
                "`verity example h200`, `verity bench|example NAME`; `verity demo h200`, the verity examples.\n"
                "```sh\nverity example vllm\n```\n",
        "c.py": '"""A wrong unit and a leg stay in code until PR 2."""',
        "benchmarks/pouw/CHANGELOG.md": "a leg",
        GLOSSARY: '| "leg"; a physical block | subprotocol |\nA leg outside the table.\n',
    }
    assert doc_offenders(list(docs), read=docs.__getitem__)[0] == [(GLOSSARY, 2, "leg"),
        ("a.md", 1, "wrong unit"), ("a.md", 1, "Isolation-Units"), ("a.md", 1, "replay unit"), ("a.md", 2, "IU"),
        ("a.md", 2, "IUs"), ("a.md", 2, "KV block"), ("a.md", 3, "legs"), ("a.md", 3, "leg"), ("a.md", 3, "leg"),
        ("b.md", 8, "thread block"), ("b.md", 9, "verity example"), ("b.md", 9, "verity bench|example"),
        ("b.md", 11, "verity example")]


def test_the_doc_scan_reads_the_party_terms():
    """The developer, the configurations, the registered circuit and assumptions, on toy files: the proving side's
    prover, a theorem's premise, a deployment record, `verity.configuration.deploy` and a registered workload stay."""
    docs = {
        "a.md": "Nothing the prover runs or hosts; a machine the prover doesn't\ncontrol, the prover's node, a "
                "prover-controlled host and the prover system. The remote deployment's registered work, under a named\n"
                "premise and the card premise, in a composite protocol. Flock's prover, a cheating prover, a theorem's "
                "premise, a deployment record, `verity.configuration.deploy`, the Lean's `Composite` and a registered "
                "workload stay.\n",
    }
    assert doc_offenders(list(docs), read=docs.__getitem__)[0] == [
        ("a.md", 1, "prover runs or hosts"), ("a.md", 1, "prover doesn't control"), ("a.md", 2, "prover's node"),
        ("a.md", 2, "prover-controlled"), ("a.md", 2, "prover system"), ("a.md", 2, "remote deployment"),
        ("a.md", 2, "registered work"), ("a.md", 2, "named premise"), ("a.md", 3, "card premise"),
        ("a.md", 3, "composite protocol")]


def test_the_doc_scan_reads_the_protocol_terms():
    """The protocol's vocabulary (`ARCHITECTURE.md`, "How an audit works"), on toy files: hm96 rows keep their name in the commitments' own docs."""
    docs = {
        "NETWORK.md": "The warden, a network warden's cells, two Wardens.\n"
                      "`warden.py`, `--warden-run`, WardenParams and warden_rows stay.\n",
        "a.md": "The draw, a drawn unit, nothing redraws; the firewall and a sampling\nblock; the declared circuit.\n"
                "Committed as hm96 rows before that period's coins; rejection dominates. A Commit period stays.\n"
                "The commit period, `flock-verify verify`, then flock-verify.\n",
        "verity/commitments/x/PROTOCOL.md": "hm96 rows in frame-v3-sha512 trees, before the draw.\n",
    }
    assert doc_offenders(list(docs), read=docs.__getitem__)[0] == [
        ("NETWORK.md", 1, "warden"), ("NETWORK.md", 1, "warden"), ("NETWORK.md", 1, "Wardens"),
        ("a.md", 1, "draw"), ("a.md", 1, "drawn"), ("a.md", 1, "redraws"), ("a.md", 1, "firewall"),
        ("a.md", 1, "sampling block"), ("a.md", 2, "declared circuit"), ("a.md", 3, "hm96 rows"),
        ("a.md", 3, "period's coins"), ("a.md", 3, "rejection dominates"), ("a.md", 4, "commit period"),
        ("a.md", 4, "flock-verify"), ("verity/commitments/x/PROTOCOL.md", 1, "draw")]


def test_the_doc_scan_reads_the_network_and_process_terms():
    """The network certifier's, the lock's and the landing's older names, on toy files: a Commit period, a pinned key and
    a trusted host stay."""
    docs = {
        "a.md": "The uplink\ntap and the auditor's tap.\n"
                "A Commit window, the isolation blocks, a transformer block; each pinned theorem in the trusted text.\n"
                "The timed verifier, the prover gateway, GPU workers, the escape probability, the post-landing run.\n"
                "A Commit period, a pinned key and a trusted host stay.\n",
    }
    assert doc_offenders(list(docs), read=docs.__getitem__)[0] == [
        ("a.md", 1, "uplink tap"), ("a.md", 2, "auditor's tap"), ("a.md", 3, "Commit window"),
        ("a.md", 3, "isolation blocks"), ("a.md", 3, "transformer block"), ("a.md", 3, "pinned theorem"),
        ("a.md", 3, "trusted text"), ("a.md", 4, "timed verifier"), ("a.md", 4, "prover gateway"), ("a.md", 4, "GPU workers"),
        ("a.md", 4, "escape probability"), ("a.md", 4, "post-landing run")]


def test_the_doc_scan_reads_the_reporting_terms():
    """The advice, the blocks and the challenger's coins (`GLOSSARY.md`), on toy files: Python's declared types, a suite's
    declared inputs, unit tests and units of measure stay."""
    docs = {
        "a.md": "The declared work and a declared\nworkload; it declares its advice. Two sampled units, a unit class, the audit coins.\n"
                "A suite's declared inputs, a declared type, a unit test, SI units and the units of a sampled proof block stay.\n",
    }
    assert doc_offenders(list(docs), read=docs.__getitem__)[0] == [
        ("a.md", 1, "declared work"), ("a.md", 1, "declared workload"), ("a.md", 2, "declares its advice"),
        ("a.md", 2, "sampled units"), ("a.md", 2, "unit class"), ("a.md", 2, "audit coins")]


def test_the_doc_scan_reads_the_training_terms():
    """Training's and accounting's vocabulary, on toy files: the dampener, NCI, the machine model, serving blocks and the
    rest in prose, never their Lean names in code spans or paths; the `machine-model` command, a server block, the
    slowdown lower bound and a window that caps its samples stay."""
    docs = {
        "a.md": "The compute dampener D; Dampeners. NCI's check, novel\ncommunication intensity.\n"
                "`Nci.Dampener.remoteDampener` and `verity/protocols/inference_only/nci`, the nci directory, NciCheck stay.\n"
                "The machine model's coverage\ntable, an envelope op; a serving block, two server-windows, the server time block.\n"
                "The overhead lower bound, an exclusion bound, the training classes, per-block caps; dark compute, certified work.\n"
                "The `machine-model` command, a server block, the slowdown lower bound and a window that caps its samples stay.\n",
    }
    assert doc_offenders(list(docs), read=docs.__getitem__)[0] == [
        ("a.md", 1, "dampener"), ("a.md", 1, "Dampeners"), ("a.md", 1, "NCI"), ("a.md", 1, "novel communication intensity"),
        ("a.md", 4, "machine model"), ("a.md", 4, "coverage table"), ("a.md", 5, "envelope op"), ("a.md", 5, "serving block"),
        ("a.md", 5, "server-windows"), ("a.md", 5, "server time block"), ("a.md", 6, "overhead lower bound"),
        ("a.md", 6, "exclusion bound"), ("a.md", 6, "training classes"), ("a.md", 6, "per-block caps"),
        ("a.md", 6, "dark compute"), ("a.md", 6, "certified work")]


def test_the_doc_scan_reads_the_owners_terms():
    """The owners' renames (`GLOSSARY.md`), on toy files: the challenger, hiding leaves, the miss probability, the reach,
    transfer blocks, scale groups and serving; `link-window` in backticks, a quoted "production", a reproduction, a
    scale group and a prover's acceptance probability stay."""
    docs = {
        "a.md": "The coin server, a coin\nsource, the coin devices and a randomness source; salted leaves and a salted commitment.\n"
                "Acceptance probability $\\bar P_a$, a design's acceptance probabilities, egress cones and a prover's "
                "acceptance probability; the link window and a link-window.\n"
                "Block-scaled FP8, blockwise scales, a scale block; in production, two productions.\n"
                'The `link-window` key, a quoted "production", a reproduction and a scale group stay.\n',
    }
    assert doc_offenders(list(docs), read=docs.__getitem__)[0] == [
        ("a.md", 1, "coin server"), ("a.md", 1, "coin source"), ("a.md", 2, "coin devices"),
        ("a.md", 2, "randomness source"), ("a.md", 2, "salted leaves"), ("a.md", 2, "salted commitment"),
        ("a.md", 3, "Acceptance probability $\\bar P_a"), ("a.md", 3, "design's acceptance probabilities"),
        ("a.md", 3, "egress cones"), ("a.md", 3, "link window"),
        ("a.md", 3, "link-window"), ("a.md", 4, "Block-scaled"), ("a.md", 4, "blockwise"), ("a.md", 4, "scale block"),
        ("a.md", 4, "production"), ("a.md", 4, "productions")]


def test_the_doc_scan_reads_the_incorrect_terms():
    """Incorrect proof blocks and the incorrect set, on toy files: a faulty node, link, device or hardware stays."""
    docs = {
        "a.md": "Two faulty proof blocks, a faulty\nproof block, the faulty-block bound and the faulty set.\n"
                "A faulty node, a faulty link, a faulty device, faulty hardware and an incorrect proof block stay.\n",
    }
    assert doc_offenders(list(docs), read=docs.__getitem__)[0] == [
        ("a.md", 1, "faulty proof blocks"), ("a.md", 1, "faulty proof block"), ("a.md", 2, "faulty-block"),
        ("a.md", 2, "faulty set")]


def test_the_doc_scan_reads_the_flock_terms():
    """C-Flock, the netlist, M0, IR and the verifier of record, on toy files: the stored label `C-Flock`, a
    `flock-netlist` tag, an `ir` key, Flock, upstream Flock, M1, M2, an M00 and the verifier stay."""
    docs = {
        "a.md": "C-Flock's prover proves a netlist; two\nNetlists, and C-FLOCK.\n"
                "M0's session, the Verity IR and two IRs; the verifier of\nrecord and two verifiers-of-record.\n"
                "The stored label `C-Flock`, the `verity/flock-netlist/v1` tag, Flock and upstream Flock stay.\n"
                "The `ir` key, `M0` in a span, M1, M2, an M00, a mirror and the verifier stay.\n",
    }
    assert doc_offenders(list(docs), read=docs.__getitem__)[0] == [
        ("a.md", 1, "C-Flock"), ("a.md", 1, "netlist"), ("a.md", 2, "Netlists"), ("a.md", 2, "C-FLOCK"),
        ("a.md", 3, "M0"), ("a.md", 3, "IR"), ("a.md", 3, "IRs"), ("a.md", 3, "verifier of record"),
        ("a.md", 4, "verifiers-of-record")]


def dump_lean_list(counts: Counter) -> None:
    lines = [f"    {json.dumps({'file': f, 'count': n}, ensure_ascii=False)}" for f, n in sorted(counts.items()) if n > 0]
    LEAN_LIST.write_text('{\n  "rule": "retired terms in Lean comments and docstrings (tests/test_retired_terms.py)",\n'
                         '  "entries": [\n' + ",\n".join(lines) + "\n  ]\n}\n", encoding="utf-8")


if __name__ == "__main__":
    if sys.argv[1:] not in (["--prune"], ["--record"]):
        sys.exit("python3 tests/test_retired_terms.py --prune    # lower LEAN_LIST to what this tree holds\n"
                 "python3 tests/test_retired_terms.py --record   # LEAN_LIST is what this tree holds (the gate holds a rise)")
    lean = Counter(f for f, *_ in offenders(_tracked())[0] if f.endswith(".lean"))
    dump_lean_list(lean_allowed() & lean if sys.argv[1] == "--prune" else lean)
