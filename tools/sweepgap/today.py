"""Today's words that GLOSSARY.md's table does not list yet (the sweep README's small table), as regexes over a name or a
line.  today_words(text) -> [(term, matched text)].  The `row/cell` and `unit` senses need context; here only the spelling."""
import re
WORDS = {
    "certifier": re.compile(r"certifiers?", re.I),
    "memory challenger": re.compile(r"memory[ _-]?challengers?|MemoryChallengers?", re.I),
    "proof table": re.compile(r"proof[ _-]?tables?|ProofTables?", re.I),
    "proof unit": re.compile(r"proof[ _-]?units?|ProofUnits?", re.I),
    "wrong unit": re.compile(r"(?:wrong|incorrect|faulty|correct|drawn|sampled|proved|checked)[ _-]?units?", re.I),
    "row/cell": re.compile(r"(?<![A-Za-z])(?:rows?|cells?)(?![a-z])"),
}
def today_words(text):
    out = []
    for k, rx in WORDS.items():
        if k == "row/cell":
            continue
        for mo in rx.finditer(text):
            out.append((k, mo.group(0)))
    return out
