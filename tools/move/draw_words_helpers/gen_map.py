"""Write the draw family's word map from the tree: terms_map.toml's rows for the family, run outside its prose scope, with
the keeps the tree needs today.

    python3 gen_map.py REPO WL.tsv OUT.toml
"""
import re
import subprocess
import sys
import tomllib
from pathlib import Path

repo, wl, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sys.path.insert(0, str(repo / "tools/move"))
import rename as R  # noqa: E402

terms = tomllib.load(open(repo / "tools/move/terms_map.toml", "rb"))
files = subprocess.run(["git", "ls-files"], cwd=repo, capture_output=True, text=True, check=True).stdout.split()

KERNELS = ["benchmarks/hash/**", "adapters/vllm/verity_vllm/definitions/registry/ref_prims.py", "adapters/vllm/verity_vllm/definitions/registry/prims.py", "adapters/vllm/verity_vllm/definitions/registry/b1.py"]
# memory accounting renames the family in its own files by its own maps (`draw_bare_map.toml`, `draw_poub_*`)
KERNELS += ["verity/protocols/accounting/poub/**", "verity/protocols/accounting/pous/**", "benchmarks/pous/**",
            "security/*/Poub/**", "security/*/Poub.lean", "security/*/Pous/**", "security/*/Pous.lean",
            "security/Proofs/Properties/Poub/**", "security/Proofs/Properties/Pous/**"] + [
            l for l in open("/tmp/kit/mem_files.txt").read().split() if not l.startswith("verity/protocols/proofs/")]
import os
OFFERED = ["adapters/vllm/**", "verity/protocols/accounting/pouw/**", "benchmarks/pouw/**", "benchmarks/network_traces/**",
           "benchmarks/network_e2e/**", "benchmarks/slowdown/**", "verity/protocols/accounting/certifier/**",
           "security/**/Certifier/**"]
if os.environ.get("PIECE1"):
    KERNELS += OFFERED
# Lean's names and the caps of the Lean lints wait for the Lean part
KERNELS += ["tests/test_lean_packages.py", "tests/test_lean_vocabulary.py", "tools/verity/lean/**"]
PROOFS = ["verity/protocols/proofs/**", "security/*/Flock/**", "security/*/Flock.lean", "security/*/Spot/**",
          "security/*/Spot.lean", "security/Proofs/Properties/Flock/**", "security/Proofs/Properties/Spot/**",
          "security/Proofs/Properties/Spot.lean", "benchmarks/one_stage/**", "benchmarks/remote_session/**"]
RECORDS = ["tools/move/**", "archive/**", ".agents/skills/friction/SKILL.md", "**/CHANGELOG.md", "fixtures/**",
           "**/fixtures/**", "*.patch", "**/TRUSTED.sha256", "uv.lock", "**/*.lock", "**/lake-manifest.json",
           "GLOSSARY.md", "tests/lint/test_repo_retired_terms.py", "tests/test_retired_terms.py",
           "tests/lint/allowlists/**", "tools/research/tests/test_slack.py",
           "tools/check/tests/test_move_check.py"]
# data: every vector, record and table with the word, by name, but the adapter's lint allowlists, which name code (a
# file, a symbol) and move with it, and `census_roots.txt`, the census's hand-kept list of modules
CODE_DATA = {"adapters/vllm/tests/lint/allowlists/ruff_vllm.json", "adapters/vllm/tests/lint/allowlists/mypy.json"}
DATA = ["*.jsonl", "*.csv", "*.tsv", "*.npz", "*.bin", "*.log", "**/lean-audit.json"]
DATA += sorted(p for p in files if p.endswith(".json") and p not in CODE_DATA
               and re.search(r"(?i)draw|drew", (repo / p).read_text(encoding="utf-8", errors="replace")))
LEAN = ["*.lean", "**/lakefile.toml", "**/lean-toolchain"]

FAMILY = re.compile(r"(?i)(?<!with)(?:draw|drew)")
TOKEN = re.compile(r"[A-Za-z_]\w*")
BARE = {"draw", "draws", "drawn", "drew", "drawing", "redraw", "redraws", "redrawn", "redrawing", "undrawn",
        "Draw", "Draws", "Drawn", "Drew", "Drawing", "Redraw", "Redraws", "Redrawn", "Undrawn", "DRAW", "DRAWS",
        "DRAWN"}


def text(p):
    try:
        return (repo / p).read_text(encoding="utf-8")
    except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError):
        return None


def tokens(t):
    return {m.group(0) for m in TOKEN.finditer(t) if FAMILY.search(m.group(0))}


def code_of(p, t):
    spans, buf, i = R.regions(p, t), [], 0
    for s, e, k in spans:
        buf += [t[i:s], " " * (e - s)]
        i = e
    return "".join(buf + [t[i:]])


# Names another side spells, which this map leaves for that side's map: the Lean's (its code, not its prose) and the
# proof system's tree's
lean_tokens, proofs_tokens = set(), set()
for p in files:
    t = text(p)
    if t is None or not FAMILY.search(t):
        continue
    if p.endswith(".lean"):
        lean_tokens |= tokens(code_of(p, t))
    elif any(R._match(p, g) for g in PROOFS):
        proofs_tokens |= tokens(t)
held = sorted((lean_tokens | proofs_tokens) - BARE, key=lambda s: (-len(s), s))

# Lean's dotted names with a part that is the word alone (`Spot.Draw`, `PouwCheck.Public.drawn`), each suffix of two
# parts or more
dotted = set()
for line in wl.read_text().splitlines():
    f = line.split("\t")
    if f[0] != "draw":
        continue
    parts = f[3].split(".")
    if f[1] == "module":
        parts = f[3].split(".")
    for i in range(len(parts) - 1):
        for j in range(i + 2, len(parts) + 1):
            if any(x in BARE for x in parts[i:j]):
                dotted.add(".".join(parts[i:j]))
dotted = sorted(dotted, key=lambda s: (-len(s), s))

words = [(k, v) for k, v in terms["words"].items() if re.search(r"(?i)draw|drew", k) and "law" not in k.lower()
         and "," not in k]
extra = [("redrawing", "resampling"), ("Redrawing", "Resampling"), ("redrawn", "resampled"), ("Redrawn", "Resampled"),
         ("Redraws", "Resamples"), ("REDRAWN", "RESAMPLED"), ("REDRAW", "RESAMPLE"), ("drew", "sampled"),
         ("Drew", "Sampled")]
at = next(i for i, (k, _) in enumerate(words) if k == "undrawn")
words = words[:at] + extra + words[at:]
FLIPS = [("test_a_seeded_serve_draws_the_same_coins_and_salt_key_and_takes_no_retention_or_live_coins",
          "test_a_seeded_serve_flips_the_same_coins_and_salt_key_and_takes_no_retention_or_live_coins"),
         ("test_each_commit_window_draws_every_coin_from_a_session_of_its_own",
          "test_each_commit_window_flips_every_coin_from_a_session_of_its_own"),
         ("test_each_window_draws_its_coins_from_a_gate_session_of_its_own",
          "test_each_window_flips_its_coins_from_a_gate_session_of_its_own")]
words = FLIPS + words


def q(s):
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def lit(s):
    assert "'''" not in s and not s.endswith("'"), s
    return "'''" + s + "'''"


def alternation(names):
    return "(?:" + "|".join(re.escape(n) for n in names) + ")"


Q = r"\x22\x27"
patterns = [
    # ids: a domain tag or format id, a note, an artifact, a branch, a lane
    r"[\w.-]+(?:/[\w.-]+)*/v\d+(?:\.\d+)*\b", r"note:[\w./-]+", r"art:[0-9a-f]+", r"cursor/[\w./-]+", r"lanes/[\w.-]+",
    # what recorded data or another system reads (terms_map.toml's rows): a quoted lowercase token, escaped too; a
    # two-part id; an upper-case snake-case name; a run's output file; a kebab-case name or a flag and the attribute
    # argparse makes of it; the record keys a Python name builds
    r"(?<![\w\\])(?<!parametrize\()(?<!parametrize\(\()(?<!match=)(?<!\x22symbol\x22: )(?<!\x22rounds\x22, )(?<!r\(coins, )"
    r"(?<!__\(self, )(?<!r\(OR, )(?<!r\(PHC, )(?<!r\(SVC, )(?<!r\(V\.D, )(?<!r\(plan_module, )(?<!\(\x22coins\x22, )"
    r"(?<!reg\.receipt, )(?<!setattr\(VR\.CW, )(?<!setattr\(T\.VR\.CW, )[rbfRBF]{0,2}([\x22\x27])(?![A-Z])(?!drawn\1: tuple\()"
    r"(?!(?:verity|research|circuit_check|cluster|benchmarks|tools|integrations|examples|demos|tests|infra|adapters)\w*\.)"
    r"[^\s\x22\x27/{]*?(?i:draw|drew)[^\s\x22\x27/{]*(?<!\.py)(?<!\.sh)(?<!\.rs)(?<!\.lean)(?<!\.md)(?<!\.toml)\1",
    r"\\[\x22\x27](?![A-Z])[^\s\x22\x27\\/{]*?(?i:draw|drew)[^\s\x22\x27\\/{]*\\[\x22\x27]",
    r"[\x22\x27][a-z_-]+/[\w-]*(?i:draw)[\w-]*[\x22\x27]",
    r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*_(?:DRAWN|DRAWS|DRAW|REDRAWN|REDRAW)(?:_[A-Z0-9]+)*\b"
    r"|\b(?:DRAWN|DRAWS|DRAW|REDRAWN|REDRAW)(?:_[A-Z0-9]+)+\b",
    r"(?<=[/.])(?:draw|drawn|draws)\.(?:json|jsonl|log|bin|lock|txt|npz)\b",
    r"(?<![\w-])-{0,2}(?:\w+-)+\w*(?i:draw|drew)\w*(?:-\w+)*(?![\w-])",
    r"(?<![\w-])-{0,2}\w*(?i:draw|drew)\w*(?:-\w+)+(?![\w-])",
    r"(?<![\w-])--\w*(?i:draw|drew)\w*(?![\w-])",
    r"(?<!a\.sizes, )\ba\.(?:draw|draws|drawn|undrawn)\b(?! = )",
    r"\b(?:draw_k|draw_file|draws_dir|fresh_draws|os_draws|lean_refuses_drawn|lean_refuses_drawless|rec_inner_draw"
    r"|replay_draw)\b",
    r"\b(?:drawn_at|checks_before_draw|unit_draw|draw_counts|draw_coin|draw_after|served_is_lean_draw|draw_reason"
    r"|draws_scan)\b",
    r"\bdraw_salt\b", r"\bdrawIsSubset\b",
    r"(?<=b\x22)draws?(?= [\w{])", r"(?<=f\x22)draw(?= \{len)", r"(?<=a request \x60)draws?\b", r"(?<=\x60)draw(?= LEN)",
    r"\binner\.draw\b", r"\b(?:every|need \d+) draw coins?\b", r"\bdraw coins? \d", r"\bpouw/draw\b",
    r"\bpublic\.draws?\b", r"(?<=window \d, )draw(?= \[)",
    r"(?<=[\x22\x27^`])(?:DRAWN?|DRAWS)(?=[\x22\x27 \\])",
    r"\bRemote\.Draw\b", r"\bRemoteDraw\b", r"\bRemote(?:Held|Audit)[./]Draw(?:Meets)?\b", r"\b(?:to)?DrawnLayer\b",
    r"\bGame\.draw\b", r"\bdrawVec\b", r"\bCoinsDraw\b",
    r"(?<=\.update\()(?:draw|draws|drawn)(?==)", r"\b(draw|draws|drawn)(?==\w+\[\x22\1\x22\])",
    r"(?<=\]\}, )draw(?==union_draw\[)", r"\bdrawn(?==len\(drawn\), exclusion_tiles)",
    r"(?<=subcircuit=sub\.id, )draws(?==)", r"(?<=rounds=len\(got\), )draws(?==)",
    # other senses: a GPU's power draw; a random number a sampler or a generator takes (a seed drawn, random weights,
    # what a generator draws); a resource's draw on CPU slots; hypothesis's `data.draw(` and `DrawFn`; PIL's `ImageDraw`
    r"\bpower[._ -]draws?\b", r"(?i:\bpower draws?\b)",
    r"(?i)\b(?:draw|draws|drawn|drew|drawing)\b(?: [\w\x27-]+){0,2} (?:power|watts?)\b",
    r"(?i)\b(?:re)?(?:draw|draws|drawn|drew|drawing)\b(?: [\w\x27-]+){0,3} seeds?\b",
    r"(?i)\bseeds? (?:[\w\x27-]+ ){0,3}(?:re)?(?:drawn|drew|draws?)\b",
    r"(?i)\b(?:re)?(?:drawn|draws?|drew)\b(?: [\w\x27-]+){0,2} weights?\b",
    r"\bdraw(?= on \{b\[\x27slots\x27\]\} CPU slots)", r"\bdraws a sample\b",
    r"(?i)(?<![\w.])(?:generator|rng|philox|gumbel)s?(?: [\w\x27\x60-]+){0,5} (?:draw|draws|drawn|drew)\b",
    r"(?i)\b(?:draw|draws|drawn|drew)(?: [\w\x27\x60-]+){0,4} (?:generator|rng|philox|gumbel)s?(?![\w.(])",
    r"(?<![\w.])data\.draw(?=\()", r"\bDrawFn\b", r"\bImageDraw\b",
    # a reason's prefix, which the Lean and the proof system's tree spell (`draw: `, `draw: coins: `, `credit draw: `)
    r"[\x22\x27](?:[a-z]+ )?(?:draw|drawn|draws)(?=: )",
    # the proof system's own tree, its paths and modules where another file names them
    r"\bverity/protocols/proofs/[\w./-]*", r"\bbenchmarks/(?:one_stage|remote_session)/[\w./-]*",
    r"\bsecurity/\w+/(?:Properties/)?(?:Flock|Spot)\b[\w./-]*", r"\bverity\.protocols\.proofs\.[\w.]*",
    r"(?<![\w.])(?:one_stage|remote_session)\.[\w.]*",
]
# a flag's attribute where a parsed namespace reads it: argparse names it from the flag, which stays
dests = set()
FLAG = re.compile(r"add_argument\(\s*(?:[\x22\x27]-\w[\x22\x27]\s*,\s*)?[\x22\x27]--([\w-]+)[\x22\x27]([^)]*)")
for p in files:
    if p.endswith(".py"):
        t = text(p)
        for m in FLAG.finditer(t or ""):
            d = re.search(r"dest\s*=\s*[\x22\x27](\w+)", m.group(2))
            name = d.group(1) if d else m.group(1).replace("-", "_")
            if FAMILY.search(name):
                dests.add(name)
patterns.append(r"(?<![\w.])(?:a|ns|args|harness_opts|opts|options)\.(?:" + "|".join(sorted(dests, key=lambda s: (-len(s), s)))
                + r")(?!\w)")
patterns.append(r"(?<=getattr\()(?:a|ns|args), [\x22\x27]\w*(?i:draw)\w*[\x22\x27]")
OTHER_SENSE_IDS = [
    "test_a_periods_salts_are_the_operating_systems_bytes_one_draw_a_port",
    "test_hiding_rows_is_core_commit_rows_with_its_salts_drawn_as_one_stream",
    "test_the_challenger_draws_the_salt_and_the_verdict_derives_the_session_id",
    "test_the_issuer_signs_no_clock_under_a_salt_it_drew_for_one_before",
    "test_h3_draws_each_rows_e_a_from_its_own_digest",
    "test_sampled_replay_draws_its_challenge_seed_from_the_challenge_module",
    "test_bf16_families_keep_the_old_draws_and_add_the_canonical_set",
    "test_sample_blocks_takes_each_tensors_ends_and_a_seeded_draw",
    "test_served_debit_fresh_draws_measure_each_input_under_salts_of_its_own",
    "test_lean_draw_runs_flock_verify_draw_on_the_coins",
]
patterns.append(r"(?<!\w)" + alternation(OTHER_SENSE_IDS) + r"(?!\w)")
# names this map leaves, cited from text it rewrites: Lean's `Draw` namespace and its modules' files, a path's
# segments (files and directories runs write; the moved `h1/` modules excepted), one-stage's `draw.py` (proofs'), record
# files, environment variables, a Lean source line a test parses, and the record keys and subcommands docs cite
patterns += [
    r"(?<![A-Za-z])Draw(?=\.[A-Za-z]|/)", r"[A-Za-z]*Draw[A-Za-z]*(?=\.lean\b)", r"\bDrawIsCoin\b",
    r"(?<=/)(?<!h1/)(?:draw|draws|drawn)(?![\w])", r"(?<![\w/])draw\.py\b", r"\b(?:draw|drawn|draws)\.json\b",
    r"(?<![\w/])draws(?=/w<k>)", r"\bDRAWS?(?==)", r"\$\{?DRAWS?\b",
    r"(?<=Registered\.checkFull registered )draw\b", r"(?<=\{\x22receipts\x22: )draw(?=\.receipts)",
    r"(?<=pouw-check-difftest )draw\b", r"(?<=verify-proof )draw\b", r"(?<=`pass`, `)draw(?=`)",
    r"(?<=`length`, `)drawn(?=`)", r"\bdraw(?=\.parse_design\b)",
    # attributes of the proof system's objects read on this side: its integrity profile's `draws` and `drawn` (and the
    # paths Lean's messages print into it), a `Draw`'s `draw`, a stage's `drawn()`, a template's `draw(key, j)`; and
    # matplotlib's `canvas.draw()`
    r"\bprofile(?:\.[A-Za-z_]+|\[[^\]]*\])*\.(?:draw|drawn|draws)\b", r"(?<=\bpr\.)(?:drawn|draws)\b",
    r"(?<=\brank\.)drawn(?=\()", r"(?<=\bstages\[coins\.at\]\.)drawn(?=\()", r"(?<=\bdraw\.)draw\b",
    r"(?<=\bd\.)draw\b", r"(?<=select\(\x22honest\x22\)\.)draw\b", r"(?<=\[\x22honest\x22\]\.)draw\b",
    r"(?<=\bsecond\.)draw\b", r"(?<=\bsub\.)draw(?=\()", r"(?<=\bcanvas\.)draw\b", r"\bdrawn(?==\{)",
    r"(?<=\x22rounds\x22, \x22)draw(?=\x22\))",
    # refusals the verifiers print, which their tests quote: Lean's `unit {unit} is not drawn`, and the core check's
    # `the {role}'s ...` of the coin role "draw"
    r"(?<=is not )drawn\b", r"(?<=[\x22\x27]not )drawn(?=[\x22\x27])", r"\bdraw(?='s (?:a coin opening|coin is))",
    r"(?<=\{ctx\}\.)(?:draw|drawn|draws)\b", r"\bdraw_out\b", r"(?<=\bWd\.)DRAW\b", r"(?<=\brecord\.)draw\b",
    r"(?<=Definition\(\x22)[^\x22]*", r"\b(?:draw|drawn|draws)(?=/(?:\d|second\b|\{))", r"\bdraw_commitment\b",
    r"(?<=The verifier )draws(?= tiles of the served window)", r"(?<=and each )drawn(?= tile is \x22)",
    r"(?<=for test in )drawn(?=, proved)", r"(?<=D12 \(no )draw(?=\))",
    r"(?<=write_text\(\x22def )drawn(?=\()", r"(?<=\bagree\.)drawn\b", r"(?<=export\.json \()draw\b",
    r"(?<=\x22subset )draw(?=\x22)",
    r"\bFlock_\w+", r"(?<=setup \()drawn(?=\))", r"(?<= \\\()drawn(?=\\\))", r"\btest_pouw_h1_drawn\b", r"(?<=[\x22\x27])Draw(?=[\x22\x27])",
    r"\bdraw(?=\.work_table)", r"(?<=os\.urandom\(64\), )drawn\b", r"(?<=every )drawn(?= instance that passed)",
    r"(?<=no )drawn(?= instance passed)", r"(?<=unseeded )draw\b",
    r"(?<=\bp\.)drawn\b",
]
held_pattern = r"(?<![\w])" + alternation(held) + r"(?![\w])"
dotted_pattern = r"(?<![\w.])" + alternation(dotted) + r"(?![\w])"

keep_in = {
    # terms_map.toml's: a name beside the name it would become in one scope, PoUB's remote audit (its `draws`, the
    # coins' draws, and `samples`, a window's sample cap, are two things), fields set and read by name, a reply named
    "benchmarks/pouw/period_cost.py": ["draws"],
    "benchmarks/pouw/tests/test_period_cost.py": ["draws"],
    "verity/protocols/accounting/pouw/prover/clear/clear_audit.py": ["drawn"],
    "verity/protocols/accounting/certifier/verifier/service.py": ["drawn"],
    "verity/protocols/accounting/certifier/verifier/tests/test_network_certifier_period.py": ["drawn"],
    "verity/protocols/accounting/poub/verifier/**": ["draw", "draws", "Draw", "Draws"],
    "**/p2_v1/**": ["draws"],
    "benchmarks/pous/held_bandwidth.py": ["draws"],
    "adapters/vllm/tests/check/test_pouw_ncp_v2_end_to_end.py": ["draws"],
    "**/served_zk/**": ["DRAWN"],
    "benchmarks/pouw/tests/test_served_zk.py": ["DRAWN"],
    "tools/verity/lean/tests/test_merge.py": ["Draw", "Drawn"],
    "tools/verity/circuit_check/tests/test_ambient.py": ["Drawn"],
}
SAMPLER = ["adapters/vllm/verity_vllm/program/lifting/continuation.py", "adapters/vllm/tests/program/test_continuation.py",
           "adapters/vllm/verity_vllm/check/replay/stoch_recompute.py",
           "adapters/vllm/tests/check/test_stoch_draw_ledger.py",
           "adapters/vllm/tests/check/test_stoch_draw_ledger_of_record.py",
           "adapters/vllm/tests/properties/fp8_cache_replay_gpu.py",
           "adapters/vllm/verity_vllm/pipeline/coverage_workloads.py", "adapters/vllm/tests/pipeline/test_seeded_draws.py",
           "adapters/vllm/verity_vllm/program/predict/reads.py", "adapters/vllm/tests/acquire/glm53_model.py"]
SAMPLER += ["**/test_train_draws.py", "**/desc_fuzz_programs.py", "tools/difftest/tests/test_constants.py",
            "adapters/vllm/tests/pipeline/test_coverage_workloads*.py", "benchmarks/serve_tax/**"]
ALL = ["draw", "draws", "drawn", "drew", "drawing", "redraw", "redraws", "redrawn", "redrawing", "undrawn", "Draw", "Draws",
       "Drawn", "Redrawn", "DRAW", "DRAWS", "DRAWN", "REDRAWN"]
for p in SAMPLER:
    keep_in[p] = ALL

# hypothesis's `draw`: in a module where the retired-words lint finds hypothesis's own (a composite's parameter), the
# bare word is hypothesis's throughout, passed on to its helpers; the one module with both has only `data.draw(`, kept
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("rt", repo / "tests/lint/test_repo_retired_terms.py")
rt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rt)
MIXED = {"verity/protocols/accounting/certifier/verifier/network/tests/test_lean_audit.py"}
excluded = lambda p: any(R._match(p, g) for g in RECORDS + DATA + LEAN + PROOFS + KERNELS)  # noqa: E731
for p in files:
    if p.endswith(".py") and not excluded(p) and p not in MIXED and p not in keep_in:
        t = text(p)
        if t and rt.hypothesis_draws(t, p):
            keep_in[p] = ["draw"]

# a name that would become one its scope already has: the record a run drew beside the tiles it drew, a composite's
# drawn components beside a sampled case
for p, ws in {"benchmarks/pouw/pearl_c_vllm/verify_run.py": ["drew"], "benchmarks/pouw/budget.py": ["draws"],
              "verity/protocols/compose/verifier/lean/tests/test_composition_difftest.py": ["drawn"]}.items():
    keep_in[p] = keep_in.get(p, []) + ws

# a shell script's upper-case words are environment variables its callers set (`DRAW=all`, `${DRAWS:-459}`)
keep_in["*.sh"] = ["DRAW", "DRAWS", "DRAWN"]
# the tests of the proof system's integrity profile read its `draws` and `drawn` throughout
keep_in["verity/protocols/tests/tests/test_profile.py"] = ALL
keep_in["benchmarks/pouw/tests/test_budget.py"] = ["draws"]
# random numbers a generator gives, the word's other sense
keep_in["benchmarks/training_slowdown/tests/test_resume.py"] = ["drawn"]
# the served check's record of `draw` outcomes, read as `record.draw` by its kept caller
keep_in["verity/protocols/accounting/pouw/prover/kernels/served_zk/of_record.py"] = ["draw"]
# seeded random data, another sense
keep_in["benchmarks/multi_unit/stage.py"] = ["draw"]
keep_in["benchmarks/flock_kernels/hm96_leaves.cu"] = ["draw"]
# coins are flipped, not sampled: the runtime's coin sources (`draw(n)`), which memory accounting's and PoUW's tests call
keep_in["verity/runtime/**"] = ["draw", "draws", "drawn"]

# a bare name that keeps its spelling where it is defined (the proof system's tree, a [keep_in] file, a name the proof
# system's tree reads from this side) keeps it where another module imports it or reads it through its module
sys.path.insert(0, "/tmp/kit")
from pyrefs import Refs  # noqa: E402
refs = Refs(repo, files)
proofs_py = [p for p in refs.files if any(R._match(p, g) for g in PROOFS + KERNELS)]


def kept_words(p):
    return {w for g, ws in keep_in.items() if R._match(p, g) for w in ws}


kept = {(p, n) for p in proofs_py for n in refs.defines(p)[0] if n in BARE}
for p in proofs_py:
    for kind, alias, targets, name, _ in refs.uses(p):
        for m in targets:
            if name in BARE and not excluded(m):
                kept.add((m, name))
                if name not in kept_words(m):
                    keep_in[m] = keep_in.get(m, []) + [name]
kept |= {(p, n) for p in refs.files if not excluded(p) for n in refs.defines(p)[0] if n in BARE and n in kept_words(p)}
qualified = set()
for r in refs.files:
    if excluded(r):
        continue
    for kind, alias, targets, name, _ in refs.uses(r):
        if kind == "mod":
            if name in BARE and any(excluded(m) for m in targets) and name not in kept_words(r):
                keep_in[r] = keep_in.get(r, []) + [name]
            continue
        if name in BARE and any((m, name) in kept for m in targets):
            if kind == "from":
                if name not in kept_words(r):
                    keep_in[r] = keep_in.get(r, []) + [name]
            else:
                qualified.add(f"{alias}.{name}")
# a module that keeps a bare word whole passes it on as a parameter and an attribute (`host(draw=16)`), so a module
# importing it keeps the word too
# a name an excluded file reads from a rewritten module keeps its spelling there
NAME_FAMILY = re.compile(r"(?i)draw|drew")
for r in refs.files:
    if not excluded(r):
        continue
    for kind, alias, targets, name, _ in refs.uses(r):
        if kind == "mod" or not NAME_FAMILY.search(name):
            continue
        for m in targets:
            if not excluded(m) and name not in kept_words(m):
                keep_in[m] = keep_in.get(m, []) + [name]
graph = {m: set(ws) & BARE for m, ws in keep_in.items() if m in refs.files and not excluded(m)}
for r in refs.files:
    if excluded(r):
        continue
    for kind, alias, targets, name, _ in refs.uses(r):
        for m in targets:
            for w in graph.get(m, ()):
                if w not in kept_words(r):
                    keep_in[r] = keep_in.get(r, []) + [w]
if qualified:
    patterns.append(r"(?<![\w.])" + alternation(sorted(qualified, key=lambda s: (-len(s), s))) + r"(?!\w)")

import ast  # noqa: E402

# a keyword that becomes a record's key (`rec.update(draw_opening=…)`, `dict(draw=…)`) keeps its bytes, held where it
# stands: by the text before it on its line and the start of its value
RECORD_KW = re.compile(r"(?i)(?:^|_)(?:draw|drawn|draws|drew|redraw|redrawn|undrawn)(?:_|$)")
for p in files:
    if not p.endswith(".py") or excluded(p) or not (t := text(p)) or not re.search(r"(?i)draw|drew", t):
        continue
    try:
        tree = ast.parse(t)
    except SyntaxError:
        continue
    lines = t.splitlines(keepends=True)
    for c in ast.walk(tree):
        if isinstance(c, ast.Call) and ((isinstance(c.func, ast.Attribute) and c.func.attr in ("update", "setdefault"))
                                        or (isinstance(c.func, ast.Name) and c.func.id == "dict")):
            for k in c.keywords:
                if k.arg and RECORD_KW.search(k.arg) and (seg := ast.get_source_segment(t, k.value)):
                    before = lines[k.lineno - 1].encode()[:k.col_offset].decode(errors="ignore")
                    if not before.strip() and k.lineno > 1:
                        before = lines[k.lineno - 2][-16:] + before
                    before = before[-40:]
                    pat = (r"(?<=" + re.escape(before).replace("\\\n", "\\n") + ")" if before.strip() else r"(?<![\w.])") + k.arg \
                        + r"(?=\s*=\s*" + re.escape(seg[:24]) + ")"
                    if pat not in patterns:
                        patterns.append(pat)

# every other name a [keep_in] file keeps, held wherever the tree spells it
extra_held = set()
for p in files:
    kw = kept_words(p)
    if kw and not excluded(p):
        extra_held |= {x for x in tokens(text(p) or "") if x not in BARE and any(w in x for w in kw)}
# a name some file looks up by its quoted spelling (a lazy table, `getattr`, `monkeypatch.setattr`), which keeps its bytes
import ast  # noqa: E402
defined, quoted = set(), set()
for p in refs.files:
    t = refs.tree(p)
    if t is None or not FAMILY.search((repo / p).read_text(encoding="utf-8", errors="replace")):
        continue
    for n in ast.walk(t):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and FAMILY.search(n.name):
            defined.add(n.name)
        elif isinstance(n, ast.Constant) and isinstance(n.value, str) and FAMILY.search(n.value) \
                and re.fullmatch(r"[A-Za-z_]\w*", n.value):
            quoted.add(n.value)
looked_up = (defined & quoted) - BARE
held = sorted(set(held) | extra_held | looked_up, key=lambda s: (-len(s), s))
held_pattern = r"(?<![\w])" + alternation(held) + r"(?![\w])"

# a module whose name is a held name stays where it is
keep_paths = {"benchmarks/pouw/h1_prove/drawn*": ["drawn"]}
for p in refs.files:
    stem = p.rsplit("/", 1)[-1].removesuffix(".py")
    if stem in set(held) and not excluded(p):
        keep_paths[p] = sorted({w for w in BARE if re.search(r"(?<![A-Za-z0-9])" + w + r"(?![a-z])", stem)})

# a string another file spells too keeps its bytes: a piece of a string literal, in a file this map rewrites, that the
# Lean's string literals, the proof system's tree's or a data file (a vector, a record, a fixture) also spell
WORD = re.compile(r"(?<![A-Za-z])(?:[Dd]raw|DRAW|drew|[Rr]edraw|[Uu]ndrawn)")
corpus = []
for p in files:
    t = text(p)
    if t is None or not WORD.search(t):
        continue
    if p.endswith(".lean") or (any(R._match(p, g) for g in PROOFS + KERNELS) and R.language(p, t) != "data"):
        try:
            spans = {"lean": R._lean, "python": R._python, "shell": R._shell, "c": R._c_like}[R.language(p, t)](t)
        except (KeyError, SyntaxError, ValueError):
            continue
        corpus += [t[s:e] for s, e, k in spans if k == R.LITERAL and WORD.search(t[s:e])]
    elif any(R._match(p, g) for g in DATA + ["fixtures/**", "**/fixtures/**"]) and len(t) < 4_000_000 and p not in CODE_DATA \
            and "corpus_coverage" not in p:
        corpus.append(t)
corpus = "\x00".join(corpus)
mirrored = set()
for p in files:
    t = text(p)
    if t is None or excluded(p) or not WORD.search(t):
        continue
    lang = R.language(p, t)
    if lang not in ("python", "shell", "c"):
        continue
    try:
        spans = {"python": R._python, "shell": R._shell, "c": R._c_like}[lang](t)
    except (SyntaxError, ValueError):
        continue
    for s, e, k in spans:
        if k != R.LITERAL or not WORD.search(t[s:e]):
            continue
        for piece in re.split(r"\{[^{}]*\}|\\n|^[rbfRBF]*[\x22\x27]+|[\x22\x27]+$", t[s:e]):
            for seg in [piece.strip()] + [x.strip() for x in re.split(r": |; |, | \u2014 |\(|\)", piece)]:
                if len(seg) >= 12 and "\n" not in seg and WORD.search(seg) and seg in corpus:
                    mirrored.add(seg)
# a string constant the Lean or a data file spells, whole (at least 8 characters) or by a segment of at least 12, keeps
# every piece of its source, an implicitly joined one too (`("... after its " "draw root")`)
import io  # noqa: E402
import tokenize  # noqa: E402
for p in files:
    t = text(p)
    if t is None or excluded(p) or not p.endswith(".py") or not WORD.search(t):
        continue
    try:
        tree = ast.parse(t)
    except SyntaxError:
        continue
    for c in ast.walk(tree):
        if not (isinstance(c, ast.Constant) and isinstance(c.value, str) and WORD.search(c.value)) or "\n" in c.value:
            continue
        v = c.value
        segs = [x.strip() for x in re.split(r": |; |, | \u2014 |\(|\)", v)]
        if not ((len(v) >= 8 and v in corpus) or any(len(x) >= 12 and WORD.search(x) and x in corpus for x in segs)):
            continue
        src = ast.get_source_segment(t, c)
        if not src or src.count("\n") > 6:
            continue
        try:
            toks = [k.string for k in tokenize.generate_tokens(io.StringIO(src).readline) if k.type == tokenize.STRING]
        except (tokenize.TokenError, SyntaxError):
            continue
        for k in toks:
            if WORD.search(k):
                pat = re.escape(k).replace("'", r"\x27").replace('"', r"\x22")
                if pat not in patterns:
                    patterns.append(pat)
mirrored = sorted(mirrored, key=lambda s: (-len(s), s))
mirrored_pattern = alternation(mirrored) if mirrored else None

L = []
L.append("""# The retired word "draw", in the Glossary's words: what `tools/move/rename.py --map tools/move/draw_words_map.toml`
# renames. GLOSSARY.md: "draw" (noun, verb), "drawn proof block", "drawn unit", "sampled unit" -> the sample, a sampled
# proof block; to sample (coins are flipped). Its rows are `terms_map.toml`'s for the family, run here outside that map's
# prose scope, so a name follows its word (`draw_session` -> `sample_session`, `DrawSetup` -> `SampleSetup`) and so does a
# file's path; a drawn unit is a sampled proof block, a draw of coins a flip, a coin family's role a sampling coin.
# Left by this map, each for its owner or its own map:
# - the proof system's own tree (`verity/protocols/proofs/`, Flock's and SPOT's areas of `security/`,
#   `benchmarks/one_stage/`, `benchmarks/remote_session/`), which proofs' workers rename, and every name, path and
#   module of it another file spells, which follows proofs' map;
# - the Lean: its files, and every name the Lean declares or spells wherever another file names it, which the family's
#   Lean map renames after lean's names move, with the Lean lints and tools that spell them (`tests/test_lean_packages.py`,
#   `tests/test_lean_vocabulary.py`, `tools/verity/lean/`);
# - what recorded data or another system reads ([keep]): data files (JSON, JSONL, CSV, TSV, text), record keys and reason
#   codes (a quoted lowercase token), kebab-case names and flags with the attributes argparse makes of them, environment
#   variables and other upper-case snake-case names, domain tags and format ids, coin labels (`draw/0`), a Definition's
#   id, a keyword that becomes a record's key, and a prompt a model reads;
# - the word in its other senses: a random number a sampler or a generator takes (the vLLM adapter's token sampling, its
#   draw ledger, its seeded workload lengths and prompts, a seed a host or a rank draws, random weights or tensors), a
#   GPU's power draw, a resource's draw on CPU slots, hypothesis's `draw` and `DrawFn`, a bare function `draw(`;
# - memory accounting's files (PoUB, PoUS, `benchmarks/pous/` and their Lean), which its own maps rename
#   (`draw_bare_map.toml`, `draw_poub_*`), and every name of theirs another file reads; and the other names terms_map.toml
#   keeps beside the name they would become ([keep_in]);
""" + ("""# - the paths whose owners run their own map over them (`adapters/vllm/`, PoUW's protocol and `benchmarks/pouw/`, the
#   network certifier's files and `benchmarks/network_traces/`, `network_e2e/`, `benchmarks/slowdown/`), and every name
#   of theirs another file reads;
""" if os.environ.get("PIECE1") else "") + """# - the deployment runtime (`verity/runtime/`), whose coin sources' `draw(n)` flips coins, which other lanes' tests call;
# - the kernel Definitions' registry (`prims.py`, `b1.py`, `ref_prims.py`), whose source a manifest's digest hashes;
# - records of what was (earlier maps, changelogs, the archive), the Glossary's table, the retired-words lints and their
#   lists, fixtures, pinned and lock files.
""")
L.append("[words]")
L += [f"{q(k)} = {q(v)}" for k, v in words]
L.append("")
if not os.environ.get("PIECE1"):
    L.append("# the PoUW budget's plan point, which would otherwise be the `Sample` that `Draw` becomes")
    L.append("[[words_in]]")
    L.append('files = ["benchmarks/pouw/budget.py"]')
    L.append('words = { "Sample" = "PlanPoint" }')
    L.append("")
    L.append("# the replay's sampling design, by the vLLM adapter's names for it")
    L.append("[[words_in]]")
    L.append('files = ["adapters/vllm/verity_vllm/pipeline/c2_replay.py", "adapters/vllm/verity_vllm/pipeline/replay_plan.py"]')
    L.append('words = { "draw_of" = "sampling_design_of" }')
    L.append("[[words_in]]")
    L.append('files = ["adapters/vllm/verity_vllm/check/replay/driver.py"]')
    L.append('words = { "DRAWS" = "DESIGNS", "draw" = "design" }')
    L.append("[[words_in]]")
    L.append('files = ["adapters/vllm/tests/pipeline/test_config_run.py"]')
    L.append('words = { "draw_of" = "sampling_design_of", "draw=" = "design=", "replay draw" = "replay design" }')
    L.append("")
    L.append("# the served check's sampled rows, by the vLLM adapter's name")
    L.append("[[words_in]]")
    L.append('files = ["adapters/vllm/verity_vllm/check/replay/served_check.py"]')
    L.append('words = { "_draw(" = "_sample_rows(" }')
    L.append("")
L.append("# the SHA-512 row engine's salts and seed tiles are derived from a key or a seed, not sampled")
L.append("[[words_in]]")
L.append('files = ["benchmarks/torch_gpu/sha512_rows/**", "verity/commitments/cuda/sha512_rows/**"]')
L.append('words = { "draws" = "derives", "drawn" = "generated" }')
L.append("")
L.append("[rewrite]")
L.append("# Never rewritten: records of what was, the term source and the lints that spell the old words on purpose, fixtures,")
L.append("# pinned and lock files; data files, which records and vectors are; the Lean; the proof system's own tree;")
L.append("# `benchmarks/hash/`, whose Dither names kernels' own map renames; the registry files a digest hashes; and memory")
L.append("# accounting's files, which its own maps rename")
ex = [g for g in RECORDS + DATA + LEAN + PROOFS + KERNELS if any(R._match(f, g) for f in files)]
L.append("exclude = [" + ", ".join(q(g) for g in ex) + "]")
L.append("")
L.append("[keep]")
L.append("patterns = [")
for p in patterns:
    L.append(f"    {lit(p)},")
L.append("    # names the Lean or the proof system's tree spells, each renamed by its own side's map")
L.append(f"    {lit(held_pattern)},")
L.append(f"    {lit(dotted_pattern)},")
if mirrored_pattern:
    L.append("    # the pieces of strings that the Lean, the proof system's tree or a data file also spells, which keep their bytes")
    L.append(f"    {lit(mirrored_pattern)},")
L.append("]")
L.append("strings = [\"the PoUW window's keyed draw takes A4 and A5\", \"no faulty proof block, every one being drawn\"]")
L.append("")
L.append("[keep_in]")
for g, ws in keep_in.items():
    L.append(f"{q(g)} = [" + ", ".join(q(w) for w in ws) + "]")
L.append("")
L.append("# paths that stay: the H1 prove benchmark's runners, which the proof system's one-stage audit runs by path, and a")
L.append("# module whose name another side spells")
L.append("[keep_paths]")
for g, ws in keep_paths.items():
    L.append(f"{q(g)} = [" + ", ".join(q(w) for w in ws) + "]")
out.write_text("\n".join(L) + "\n")
tomllib.load(open(out, "rb"))
print(f"{len(words)} words; {len(held)} held names ({len(lean_tokens)} Lean, {len(proofs_tokens)} proofs'); "
      f"{len(dotted)} dotted Lean names; {len(patterns)} patterns; {len(keep_in)} keep_in; {len(mirrored)} mirrored; "
      f"{len(extra_held)} held from keep_in; {len(qualified)} qualified; {len(keep_paths)} keep_paths")
Path("/tmp/kit/mirrored.txt").write_text("\n".join(mirrored) + "\n")
