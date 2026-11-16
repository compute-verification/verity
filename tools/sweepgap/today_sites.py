#!/usr/bin/env python3
"""Task 2: every site of today's words that GLOSSARY.md's table does not list yet (sweep README's small table), in every
tracked text file (not tools/move/, not GLOSSARY.md), by owner lane and kind.
  terms: certifier, memory challenger, proof table, unit (a proof block), row/cell (a benchmark result)
Run from the checkout root:  python3 today_sites.py OUT_DIR
Writes sites2.tsv, names2.tsv (scan.py's columns, `new`/`kind`/`class`/`evidence` filled by rule and marked auto),
phrases.tsv (each phrase counted for row/cell and unit), row_cell_files.tsv (the file-level sense decision), by_owner2.tsv,
excluded.tsv (sites of the spelling left out as another sense, by reason)."""
import collections, csv, json, re, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import scan as S
import kinds as K
m = S.lint(); owner = S.owners()
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
files = [f for f in subprocess.run(["git", "ls-files", "-z"], capture_output=True).stdout.decode().split("\0") if f]
TEXT = (".py", ".pyi", ".md", ".sh", ".bash", ".toml", ".yaml", ".yml", ".service", ".timer", ".conf", ".cfg", ".ini", ".lean", ".rs",
        ".c", ".h", ".cc", ".cpp", ".hpp", ".cu", ".cuh", ".ts", ".js", ".go", ".json", ".jsonl", ".txt", ".rst", ".tex", ".patch", ".csv", ".gitignore")
B = r"(?<![A-Za-z0-9])"
RX = {
    "certifier": re.compile(r"(?i:certifiers?)(?![a-z])"),
    "memory challenger": re.compile(r"(?i:memory[ _-]?challengers?)(?![a-z])"),
    "proof table": re.compile(r"(?i:proof[ _-]?tables?)(?![a-z])"),
    "unit": re.compile(r"(?<![a-z])[Uu]nits?(?![a-z])|(?<![A-Za-z])UNITS?(?![A-Za-z])"),
    "row/cell": re.compile(r"(?<![a-z])(?:[Rr]ows?|ROWS?|[Cc]ells?|CELLS?)(?![a-z])"),
}
IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:[.\-/][A-Za-z_][A-Za-z0-9_]*)*")
def spelling(line, s, e):
    """the identifier (snake, camel, capitals, kebab) holding the match, else the match with one word each side"""
    for mo in IDENT.finditer(line):
        if mo.start() <= s < mo.end():
            whole = mo.group(0)
            for piece in re.split(r"[./]", whole):
                p0 = line.find(piece, mo.start())
                if p0 <= s < p0 + len(piece):
                    return piece if piece != line[s:e] else piece, True
    return line[s:e], False
def phrase(line, s, e):
    a = re.search(r"([A-Za-z'’-]+[ _-])?$", line[:s]).group(0)
    b = re.match(r"([ _-][A-Za-z'’-]+)?", line[e:]).group(0)
    return (a + line[s:e] + b).strip(" _-").lower()
PROOF_BEFORE = re.compile(r"(?i)(?:proof|wrong|incorrect|faulty|correct|drawn|sampled|proved|checked|replay|verification|compile|isolation|coarse|universal|transition)[ _-]?$")
UNIT_CUE = re.compile(r"(?i)proof|proved|sampl|draw|wrong|incorrect|faulty|flock|\bvu\b|_vu|replay")
UNIT_OUT = re.compile(r"(?i)unit[ _-]?tests?|unittest|\bunit[ _-]?(cost|price|norm|vector|step|stride|roundoff|lower|upper|interval|circle|ball|cube)|['\"]unit['\"]\s*[:=]|\bunit\s*=\s*['\"]|rate_unit|time_unit|systemd|\.service|"
                      r"\bin units? of\b|units? of (?:time|measure|work|ulp|account)|\bulps?\b|\bone unit\b.*\bulp|\(unit\)|-> Unit|: Unit\b|PUnit|Unit\.unit|\(\)\s*:=")
UNIT_LIVE_FILES = ("benchmarks/hash/", "benchmarks/native_peak/", "benchmarks/machine_model/", "tools/research/", "infra/")
BENCH_FILE = re.compile(r"^(benchmarks/(?!census/)|tools/research/(src/)?[^ ]*metric|infra/dashboard/|demos/)|(^|/)metrics\.toml$|(^|/)benchmarks/[^ ]*README\.md$")
ROW_LIVE = re.compile(r"(?i)hm96|leaf|leaves|merkle|commit|\bsha|hash|\btile|matrix|gemm|\bkv\b|attention|tensor|torch|numpy|\bnp\.|stride|row[- _]?major|column|\bcols?\b|\bmma\b|\bcta\b|\bsm\b|register|lane|opcode|ulp|\bbits?\b|\bbytes?\b|slot|activation|logit|weights?\b|\bkernel|swizzle|pivot|csv\b|dataframe")
ROW_BENCH = re.compile(r"(?i)benchmark|\bbench\b|panel|result|report|record|metric|measured|per[- ]run|\bruns?\b|proof table|served table|layers? table|steady|ledger|figure|headline|table of record|row of record|\bnumbers?\b|verdict|baseline|overhead|cost\b|throughput|latency|seconds|_s\b")
S1 = re.compile(r"(?i)(served|proof|steady(?:[- _]state)?|summary|layers?|panel|results?|bench(?:mark)?|metrics?|ledger|report|figures?|headline|breakeven|zero-knowledge|recursion|plain)'?s?[ _-]+(rows?|cells?)(?![a-z])")
S2 = re.compile(r"(?i)(rows?|cells?)[ _-]+(of record|per run)|(table|panel|report|figure|grid)'?s?[ _-]+(rows?|cells?)(?![a-z])|(rows?|cells?) (of|in) (the|a|that|this) (table|panel|report|figure|grid)")
DATA_EXT = (".json", ".jsonl", ".csv", ".cu", ".cuh", ".lean", ".patch", ".txt")
rows = []; excluded = collections.Counter(); phr = collections.Counter(); rcfile = collections.defaultdict(lambda: [0, 0, 0, 0])
cache = {}
def read(f):
    try:
        p = Path(f)
        if p.stat().st_size > 1_500_000:
            return None
        return p.read_text(encoding="utf-8")
    except Exception:
        return None
for f in files:
    if f.startswith(("tools/move/", "tools/sweepgap/")) or f == "GLOSSARY.md" or not f.endswith(TEXT):
        continue
    t = read(f)
    if t is None:
        continue
    lines = t.split("\n"); cum = [0]
    for ln in lines:
        cum.append(cum[-1] + len(ln) + 1)
    kind_fn = None
    lean = f.endswith(".lean")
    for term, rx in RX.items():
        if term in ("unit", "row/cell") and not rx.search(t):
            continue
        hits = []
        for mo in rx.finditer(t):
            hits.append(mo)
        if not hits:
            continue
        import bisect
        for mo in hits:
            li = bisect.bisect_right(cum, mo.start()) - 1
            line = lines[li]; s = mo.start() - cum[li]; e = mo.end() - cum[li]
            sp, whole_ident = spelling(line, s, e)
            ph = phrase(line, s, e)
            tier = ""
            if term == "unit":
                if f.endswith((".json", ".jsonl")) or any(d in m.FIXTURE_DIRS for d in f.split("/")[:-1]):
                    excluded[("unit", "a json file or a fixture directory (data a reader holds)")] += 1; continue
                if lean and re.fullmatch(r"Unit|PUnit", mo.group(0)):
                    excluded[("unit", "Lean's Unit type")] += 1; continue
                if UNIT_OUT.search(line):
                    excluded[("unit", "unit test, a unit of measure, a metric's unit field or a numeric unit")] += 1; continue
                if PROOF_BEFORE.search(line[:s]):
                    tier = "A"
                elif UNIT_CUE.search(line):
                    tier = "B"
                else:
                    excluded[("unit", "no proof-block cue on the line")] += 1; continue
                phr[("unit", ph, tier)] += 1
            if term == "row/cell":
                if not BENCH_FILE.search(f):
                    excluded[("row/cell", "outside the benchmark files")] += 1; continue
                if f.endswith(DATA_EXT) or any(d in m.FIXTURE_DIRS for d in f.split("/")[:-1]) or "/data/" in f:
                    excluded[("row/cell", "data (json, csv, CUDA, Lean, fixtures, data directories)")] += 1; continue
                live = bool(ROW_LIVE.search(line)); bench = bool(ROW_BENCH.search(line))
                strict = bool(S1.search(line) or S2.search(line))
                rcfile[f][0] += 1; rcfile[f][1] += live; rcfile[f][2] += bench; rcfile[f][3] += strict
                if live and not strict:
                    excluded[("row/cell", "a live sense on the line (commitment, tile, matrix, table row…)")] += 1; continue
                tier = "S" if strict else ("R" if re.search(r"README|metrics", f) else "F")
                phr[("row/cell", ph, tier)] += 1
            if kind_fn is None:
                kind_fn = K.kind_at(f, t)
            kind = kind_fn(mo.start())
            if kind == "prose" and whole_ident and not f.endswith(".md"):
                pass
            rows.append((term, f, li + 1, mo.group(0), sp, kind, owner(f), tier, ph, int(m.holds(f))))
kept = []
for r in rows:
    if r[0] == "row/cell" and r[7] == "F":
        n, l, b, s = rcfile[r[1]]
        if b / max(n, 1) < 0.4:
            excluded[("row/cell", "file whose lines name a benchmark result on under 40%")] += 1; continue
    kept.append(r)
rows = kept
with open(out / "sites2.tsv", "w") as fh:
    fh.write("term\tfile\tline\tmatched\tspelling\tkind\towner\ttier\tphrase\tlint_reads_file\n")
    for r in rows:
        fh.write("\t".join(map(str, r)) + "\n")
with open(out / "excluded.tsv", "w") as fh:
    fh.write("term\treason\tsites\n")
    for (a, b), n in excluded.most_common():
        fh.write(f"{a}\t{b}\t{n}\n")
with open(out / "phrases.tsv", "w") as fh:
    fh.write("term\tphrase\ttier\tsites\n")
    for (a, b, c), n in sorted(phr.items(), key=lambda kv: -kv[1]):
        if n >= 2:
            fh.write(f"{a}\t{b}\t{c}\t{n}\n")
with open(out / "row_cell_files.tsv", "w") as fh:
    fh.write("file\tsites\twith_a_live_cue\twith_a_benchmark_cue\towner\n")
    for f, (n, l, b, _) in sorted(rcfile.items(), key=lambda kv: -kv[1][0]):
        fh.write(f"{f}\t{n}\t{l}\t{b}\t{owner(f)}\n")
by_own = collections.Counter((r[6], r[0], r[5]) for r in rows)
with open(out / "by_owner2.tsv", "w") as fh:
    fh.write("owner\tterm\tkind\tsites\n")
    for (o, t, k), n in sorted(by_own.items()):
        fh.write(f"{o}\t{t}\t{k}\t{n}\n")
print(len(rows), "sites;", collections.Counter(r[0] for r in rows))

# names2.tsv: scan.py's columns, filled by rule (every filled cell is `auto`: read the evidence before a rename)
SAY = {"certifier": ("attester", "Attester", "ATTESTER"), "memory challenger": ("memory_attester", "MemoryAttester", "MEMORY_ATTESTER"),
       "proof table": ("proof_benchmark", "ProofBenchmark", "PROOF_BENCHMARK"), "unit": ("proof_block", "ProofBlock", "PROOF_BLOCK"),
       "row/cell": ("result", "Result", "RESULT")}
def new_of(term, sp):
    s = sp
    if term == "certifier":
        s = re.sub(r"certifier", "attester", s); s = re.sub(r"Certifier", "Attester", s); s = re.sub(r"CERTIFIER", "ATTESTER", s)
    elif term == "memory challenger":
        s = re.sub(r"memory([ _-]?)challenger", r"memory\1attester", s); s = re.sub(r"Memory([ _-]?)Challenger", r"Memory\1Attester", s)
        s = re.sub(r"MEMORY([ _-]?)CHALLENGER", r"MEMORY\1ATTESTER", s); s = re.sub(r"(?i)memory([ _-]?)challenger", r"memory\1attester", s)
    elif term == "proof table":
        s = re.sub(r"(?i)proof([ _-]?)table", lambda mo: ("proof" if mo.group(0)[0].islower() else "Proof") + mo.group(1) + ("benchmark" if mo.group(0)[-5:].islower() or "_" in mo.group(0) else "Benchmark"), s)
    elif term == "unit":
        pre = {"wrong": "incorrect", "drawn": "sampled", "draw": "sampled"}
        s = re.sub(r"(?i)\b(wrong|drawn)[ _-]units?", lambda mo: pre[mo.group(1).lower()] + " proof block", s)
        s = re.sub(r"(wrong|drawn)_units", lambda mo: pre[mo.group(1)] + "_proof_blocks", s)
        s = re.sub(r"(wrong|drawn)_unit", lambda mo: pre[mo.group(1)] + "_proof_block", s)
        s = re.sub(r"Units", "ProofBlocks", s); s = re.sub(r"Unit", "ProofBlock", s); s = re.sub(r"UNITS", "PROOF_BLOCKS", s); s = re.sub(r"UNIT", "PROOF_BLOCK", s)
        s = re.sub(r"(?<![a-z])units(?![a-z])", "proof_blocks", s); s = re.sub(r"(?<![a-z])unit(?![a-z])", "proof_block", s)
        s = re.sub(r"_units(?![a-z])", "_proof_blocks", s); s = re.sub(r"_unit(?![a-z])", "_proof_block", s); s = re.sub(r"(?<![a-z])units_", "proof_blocks_", s); s = re.sub(r"(?<![a-z])unit_", "proof_block_", s)
    else:
        s = re.sub(r"Rows", "Results", s); s = re.sub(r"Row", "Result", s); s = re.sub(r"ROWS", "RESULTS", s); s = re.sub(r"ROW", "RESULT", s)
        s = re.sub(r"Cells", "Results", s); s = re.sub(r"Cell", "Result", s); s = re.sub(r"CELLS", "RESULTS", s); s = re.sub(r"CELL", "RESULT", s)
        s = re.sub(r"(?<![a-z])(rows|cells)(?![a-z])", "results", s); s = re.sub(r"(?<![a-z])(row|cell)(?![a-z])", "result", s)
        s = re.sub(r"(?<=[_-])(rows|cells)(?![a-z])", "results", s); s = re.sub(r"(?<=[_-])(row|cell)(?![a-z])", "result", s)
        s = re.sub(r"(rows|cells)(?=[_-])", "results", s); s = re.sub(r"(row|cell)(?=[_-])", "result", s)
    return s
# readers: spellings the data files and recorded tags hold
data_tokens = {}
TAGRX = re.compile(r"[\w./:-]*(?:certifier|memory[-_]challenger|proof[-_]table)[\w./:-]*", re.I)
for f in files:
    if f.startswith("tools/move/") or f == "GLOSSARY.md":
        continue
    is_data = f.endswith((".json", ".jsonl", ".u16", ".sha256", ".lock")) or any(d in m.FIXTURE_DIRS for d in f.split("/")[:-1]) or "records" in f.split("/")
    if not is_data:
        continue
    tx = read(f)
    if tx is None:
        continue
    for i, ln in enumerate(tx.split("\n"), 1):
        for mo in TAGRX.finditer(ln):
            data_tokens.setdefault(mo.group(0).lower(), f"{f}:{i}")
tag_lines = {}
for r in rows:
    pass
names = collections.OrderedDict()
for term, f, line, matched, sp, kind, own, tier, ph, lr in rows:
    d = names.setdefault((sp, term), {"sites": 0, "files": set(), "lean": 0, "own": collections.Counter(), "ex": [], "kinds": collections.Counter(), "first_stored": None})
    d["sites"] += 1; d["files"].add(f); d["lean"] += f.endswith(".lean"); d["own"][own] += 1; d["kinds"][kind] += 1
    if len(d["ex"]) < 3:
        d["ex"].append(f"{f}:{line}")
    if kind in ("stored", "key") and f.endswith((".json", ".jsonl")) and d["first_stored"] is None:
        d["first_stored"] = f"{f}:{line}"
with open(out / "names2.tsv", "w") as fh:
    fh.write("spelling\tterm\tsites\tfiles\tlean_sites\towners\texamples\tnew\tkind\tclass\tevidence\n")
    for (sp, term), d in sorted(names.items(), key=lambda kv: -kv[1]["sites"]):
        kind = d["kinds"].most_common(1)[0][0]
        new = new_of(term, sp)
        ev = ""
        low = sp.lower()
        tagged = next((v for k, v in data_tokens.items() if low in k), None)
        if d["lean"] == d["sites"]:
            cls = "LEAN"
        elif tagged:
            cls, ev = "HELD", f"a recorded tag, key or fixture holds it: {tagged}"
        elif d["first_stored"] and any(any(x in f for x in ("fixtures", "records", "lean-audit", "moves.json")) for f in d["files"]):
            cls, ev = "HELD", f"a recorded artifact holds it: {d['first_stored']}"
        elif term in ("unit", "row/cell") and sp.lower() in ("row", "rows", "cell", "cells", "unit", "units") :
            cls, ev = "RENAME", ""
        else:
            cls = "RENAME"
        if term in ("unit", "row/cell") and cls == "RENAME" and sp.lower() not in ("row", "rows", "cell", "cells", "unit", "units") and not re.search(r"(?i)(proof|wrong|drawn|sampled|proved|checked|replay|steady|served|panel|layer|summary|bench)", sp):
            cls, ev = "ASK", f"is `{sp}` the {'proof block' if term == 'unit' else 'benchmark result'} sense? the cue is on its line only"
        fh.write("\t".join(map(str, [sp, term, d["sites"], len(d["files"]), d["lean"], ", ".join(f"{o}:{n}" for o, n in d["own"].most_common(6)), " ".join(d["ex"]),
                                      new + " (auto)" if new != sp else "", kind, cls, ev])) + "\n")
print("names2:", len(names))
