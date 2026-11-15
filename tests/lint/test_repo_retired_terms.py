"""Retired terms in the tree, held to ``allowlists/repo_retired_terms.json``: no file gains a use of a term that
``GLOSSARY.md``'s table of older names retires.  The table is the term source, so a row added there is enforced here
with no change to this module, and its fix is the row's term: ``'X' is retired; say Y (GLOSSARY.md)``.

What a row retires:

* each quoted old name of its first cell (``"C-Flock"``, ``"draw law"``), as a word or as one part of an identifier, in
  any spelling of its parts: ``draw law``, ``draw_law``, ``drawLaw``, ``DRAW_LAW``.  A lower-case part may start upper
  case; a part spelled with capitals (``IR``, ``C-Flock``) matches as spelled or all upper case.  Plurals count, and so do
  a verb's forms where the row says ``(noun, verb)``.  A quote is not enforced where the row narrows it to a sense
  the lint can't tell apart: a quote followed by ``for``, ``meaning``, ``naming``, ``in``, ``as`` or ``of`` narrows every
  quote of its clause (the cell's ``;``-separated part), ``alone`` narrows its own, and one after ``outside`` is the
  exception it names.  A one-word quote followed by a list of senses (``"dark" (compute, fleet, leaves)``) retires
  each phrase (``dark compute``).  The words that ``GLOSSARY.md``'s narrower senses say still carry two senses
  (**Row**, **Transcript**, ...) and those ``LIVE`` names are not enforced alone, only in the phrases that retire them;
* each identifier written in backticks in a clause that says ``until renamed``, when it is one only code spells
  (``Flock.Firewall``, ``FC_FIREWALL*``, ``DrawIsCoin``: a dot, an underscore, a hyphen, an inner capital or a ``*``,
  which matches any identifier characters); a plain word there (``window``), a path, and a name that owns the next
  (``machine_model``'s ``cover``) are not.  A ``*`` name's site isn't one where the identifier around it already says
  the row's term: its words before a parenthesis, ``;`` or ``,``, joined as ``MonitoringPeriod`` or lowerCamel
  ``monitoringPeriod`` (as spelled otherwise) or ``monitoring_period`` (in any case), so ``MonitoringPeriodChecked``
  and ``liveAt_monitoringPeriod`` say ``monitoring period`` and aren't ``*Period*`` sites.  The identifier is taken
  whole, so one holding an old name beside the term (``ledgerPeriod_le_monitoringPeriod``) passes the glob, and a name
  that is itself a row's old name (``ComposePeriod``) still counts through its own term.

The lint reads the whole text of what the lane-text lint reads (``test_repo_lane_text.py``: code, Lean, shell and
config, and Markdown), identifiers, strings and prose alike, outside what ``SKIP`` names, with each reason, and outside
what the word maps of ``tools/move/`` keep: a domain tag or format id (``verity/flock-netlist/v1``), and ``note:``,
``art:``, branch and lane names.  A goal's id is a name too, kept as written where a TOML file gives it as a ``goal``
key's value (``goal = "vllm/serving-cost-benchmark"``, ``GOAL``); the tree holds no list of goal ids, so the same id
elsewhere still counts.  A word's other senses aren't counted: by the words about it (``OTHER_SENSES``: a power law,
nvidia-smi's ``power.draw``, ``AUDIT_CAP``, ``cap_bits``, a memory cap, a compute capability), by the code around it
(hypothesis's ``draw``), by the file's place (``OWN_CAPS``: PoUS's and PoUB's own caps, a row table's size, the live
proof block's caps, a PoUW session's cap, Flock's caps, a request's length cap, a logit soft cap) and by the name it
stands in (``OWN_NAMES``: PoUS's ``Pous.Period``, its responder's ``B2Period`` and a party's ``Attest.Period``).  A key
a stored record or format still carries is held only in the files its entry of ``HELD_KEYS`` names, where the key stands
whole (the whole identifier, or for a plain word the whole quoted string); the same key in another file, the word in
prose and a longer name still count, and an entry whose key no longer stands in one of its files is refused as stale.
Sites are counted per file: an entry ``{"file", "count"}`` allows ``count`` sites in its file, a file
with more fails, and ``--prune`` lowers what the tree no longer holds.  The ratchet gate (``adapters/vllm/tools/ratchet_gate.py``) refuses a change that adds a site to a file
it touches, grows this list, or leaves a renamed file's entry behind; it loads this module alone from the change's
base, so the module imports only the standard library and reads ``GLOSSARY.md`` from the tree it runs in.

    python3 tests/lint/test_repo_retired_terms.py --move OLD NEW   # a file (or directory) moved: its entry follows it
    python3 tests/lint/test_repo_retired_terms.py --prune          # delete and lower what this tree no longer holds
    python3 tests/lint/test_repo_retired_terms.py --record         # the list is what this tree holds (a new row's sites)
"""
from __future__ import annotations

import io
import json
import re
import subprocess
import sys
import tokenize
from collections import Counter
from functools import lru_cache
from pathlib import Path

NAME = "repo_retired_terms"
ROOT = Path(__file__).resolve().parents[2]
LIST = Path(__file__).resolve().parent / "allowlists" / f"{NAME}.json"
GLOSSARY = "GLOSSARY.md"
#: path prefixes this lint leaves to others (the ratchet gate reads it): none
LEFT_OUT = ()
#: what isn't read, by path prefix (or a whole path), each with its reason
SKIP = {
    GLOSSARY: "the term source: its table of older names says where each old name stays",
    "tools/move/": "the word maps name the words they retire",
    "tests/tools/check/test_move_check.py": "its toy moves rename the words the maps retire",
    "fixtures/": "replicated fixtures and their notes are data, named by what they hold",
    "benchmarks/census/": "data only (AGENTS.md: verity never reads it)",
    f"tests/lint/test_{NAME}.py": "its examples are the terms it refuses",
    "tests/test_retired_terms.py": "the prose lint that holds a few of the terms at zero names them to find them",
}
#: test fixtures: a path with one of these directories is data
FIXTURE_DIRS = ("fixtures", "testdata", "golden")
#: a changelog records what was
HISTORY = ("CHANGELOG.md",)
#: quoted old names the table lists alone that keep another live sense, which the lint can't tell apart
LIVE = {
    "session": "the table retires it by sense (a run, a release, a certifier's attachment, a challenge phase)",
    "kind": "the SP1-era kind; every other kind (a Definition's, a record's, a store's) is live",
    "pin": "a digest pin (`TRUSTED.sha256`, circuit-check's pins) is live; only a pinned guarantee is retired",
    "instance": "Lean's `instance`, Python's `isinstance` and a template instance are live",
    "slack": "the table keeps its other senses, Slack among them",
    "tap": "vLLM's capture taps are live; the certifier's are retired as \"uplink tap\" and \"the auditor's tap\"",
    "remote audit": "PoUB's Lean names keep the words until their rename, and the remote configuration's audit "
                    "(`benchmarks/rho`'s `Remote`) is live; the docs' prose holds it at zero (`tests/test_retired_terms.py`)",
    "M0": "a label the code uses for other things", "M1": "a label the code uses for other things",
    "M2": "a label the code uses for other things",
    "layer 1": "a model's layers", "layer 2": "a model's layers", "layer 3": "a model's layers",
    "commit period": "the Commit period is the term for the steps one Commit covers",
    "IR": "an acronym the code uses for other things (an alias, a cut); \"Verity IR\" and \"IR evaluator\" are retired",
    "reference run": "a run whose output another is compared with is live; the training-cost reference is retired",
}
#: words the narrower senses say carry two senses that this lint tells apart itself, so they stay enforced alone: each
#: kept sense is in ``OTHER_SENSES`` or ``OWN_CAPS``, and the retired one counts everywhere else
SENSED = {"cap": "SPOT's served cap and the certifier's response cap (``max_tokens_cap``) are ``CAP_SENSES``; Pearl's is retired"}
#: other senses of an enforced word, never counted: nvidia-smi's fields (``power.draw``, ``compute_cap``) among them, the
#: names of PoUS's cap on an audit's duration and PoUB's lifetime cap, the stacking property's caps as its Lean names
#: them (``Gates.Caps``, ``FoolingCap``), a kernel's bound on its launch grid (``grid_cap``), the public length ℓ a
#: registered compiler carries (``cap_bits``, ``CAP_BITS``, "the cap ℓ"), nvidia-smi's software power cap
#: (``sw_power_cap``), a logit soft cap (``soft_cap``, ``logits_soft_cap``), cuBLASLt's algorithm capabilities
#: (``cublasLtMatmulAlgoCapGetAttribute``, ``CUBLASLT_ALGO_CAP_*``) and the harness's readers of them (``cap_get``,
#: ``cap_list``), Linux capabilities (``CAP_SYS_PTRACE``) and the hardness theory's ISA capability bound (``ISA-CAP``,
#: ``isa_cap``), none of them Pearl's; and the senses below, each its own pattern
OTHER_NAMES = (r"(?i:power[- ]laws?|laws? of large numbers|(?:Amdahl|Moore|Little|Zipf|Kerckhoffs)'s laws?"
               r"|power[ ._-]draws?)|(?<![^\W_])(?:audit_cap|AUDIT_CAP|auditCap|cap_ns|CAP_NS"
               r"|schedule_cap|SCHEDULE_CAP|cap_bits|CAP_BITS)(?![^\W_])"
               r"|(?<!\w)(?:R_cap(?:_phase|_node(?:_by_class_peak)?)?|caps\.total|FoolingCap|compute_cap"
               r"|grid_cap|cap ℓ)(?!\w)"
               r"|(?<![^\W_])(?:sw_power_cap|soft_cap|isa_cap(?:_check)?|cap_(?:get|list))(?![^\W_])|(?<!\w)ISA-CAP"
               r"|cublasLt\w*Cap\w*|CUBLASLT_\w*CAP\w*"
               r"|(?i:(?<!\w)cap_(?:sys_ptrace|sys_admin|dac_read_search|dac_override|chown|fowner|ipc_lock)(?!\w))")
#: other senses of "cap" that name none of Pearl's, each by its names: a request's length cap, its ``max_tokens`` (the
#: certifier's registered response cap); SPOT's served cap S, in prose; a soft
#: cap in prose (a logit's, or a resource limit's); a cap counted in bytes, MiB, GiB or words, as its name says (a memory
#: or buffer cap); a CUDA compute capability as the tests name one (``cap80``, ``cap90_fa2``); the served-commit
#: profile's cap on the CTAs one launch hashes; FA3's cap on its number of splits
CAP_SENSES = {
    "a request's length cap": r"(?<![^\W_])(?:max_tokens_cap|cap_exhausted|stopped_before_cap)(?![^\W_])",
    "a soft cap": r"(?i:(?<!\w)soft[ -]caps?(?!\w))",
    "a cap in bytes or words": r"(?<![^\W_])(?:host(?:_memory)?_cap|snapshot_cap|SNAPSHOT_CAP|cap_(?:bytes|mb|gb)"
                               r"|CAP_(?:BYTES|MB|GB)|cap_words|cap_exceeded_peak_bytes)(?![^\W_])",
    "a compute capability": r"(?<![^\W_])cap(?:[5-9]\d|1[0-2]\d)(?!\d)",
    "a launch's rows cap": r"(?<![^\W_])(?:rows_cap_(?:blocks|free)|(?:PROF|SERVED_COMMIT)_ROWS(?:_RING8|_4K|_LAYER)?_CAP)"
                           r"(?![^\W_])|(?<!\w)Rows(?:Ring8|4k|Layer)?CapArm(?!\w)",
    "FA3's split cap": r"(?<![^\W_])(?:split_cap|SPLIT_CAP)(?![^\W_])",
    "SPOT's served cap": r"(?<!\w)served cap(?!\w)",
}
OTHER_SENSES = re.compile("|".join((OTHER_NAMES, *CAP_SENSES.values())))
#: the live proof block's files, which name no Pearl term: each request's cap, its public ``max_tokens``, in the table
#: ``caps`` with ports ``cap_<i>``, and in capitals the table's role ``CAPS`` and the request's index ``CAP``; and the
#: tests of a PoUW session's cap, which name none either, with their vectors' cap ``CAP``
LIVE_CAPS = ("verity/protocols/resource_accounting/compute/pouw/prover/circuit/live.py",
             "verity/protocols/resource_accounting/compute/pouw/verifier/live_reads.py",
             "tests/verity/protocols/resource_accounting/compute/pouw/verifier/test_circuit_live.py",
             "tests/verity/protocols/resource_accounting/compute/pouw/verifier/test_pouw_window.py",
             "tests/verity/protocols/resource_accounting/compute/pouw/verifier/window_vectors.py")


def _vllm(path: str) -> str:
    """The vLLM adapter's file at ``path`` as it names it: its tests (``tests/...``) are at ``tests/adapters/vllm/tests``."""
    return f"tests/adapters/vllm/{path}" if path.startswith("tests/") else f"adapters/vllm/{path}"


#: the vLLM adapter's files of a request's length cap, its ``max_tokens``, which name no Pearl term: the continuation
#: controller that lifts a request to its cap, the workload and executed prefix that declare it, and their tests, whose
#: ``CAP`` in capitals is the same cap
REQUEST_CAPS = tuple(_vllm(p) for p in (
    "verity_vllm/program/lifting/continuation.py", "verity_vllm/program/lifting/lifted_compose.py",
    "verity_vllm/program/workload.py", "verity_vllm/commit/executed_prefix.py", "verity_vllm/commit/served_record.py",
    "tests/program/test_continuation.py", "tests/program/test_lifted_request.py",
    "tests/program/test_lifted_request_padrev.py", "tests/program/test_lifted_dense_b1.py",
    "tests/program/test_lifted_dense_b1_padrev.py", "tests/program/test_lifted_mixed.py",
    "tests/program/test_lifted_workload_padrev.py", "tests/program/test_lifted_tiny.py",
    "tests/program/test_lifted_tiny_padrev.py"))
#: the vLLM and torch adapters' files of a logit soft cap, which name no Pearl term: the softcapped attention chain's static ``CAP``
#: (``AttentionSoftcap_v2{...,CAP}``) and the scalar ``cap`` its rules and tests pass
SOFTCAP_CAPS = tuple(_vllm(p) for p in (
    "verity_vllm/program/frontend/targets.py", "verity_vllm/program/frontend/rules/vocab.py",
    "verity_vllm/program/frontend/rules/vllm_bindings/attention.py",
    "verity_vllm/program/frontend/rules/vllm_bindings/observations.py", "verity_vllm/program/predict/build.py",
    "tests/acquire/fa2_attn_oracle.py", "tests/program/test_fa2_softcap.py", "tests/program/test_boolean_softcap.py",
    "tests/program/test_registered_value_chains.py", "tests/program/test_registered_value_rules.py",
    "tests/properties/fa2_softcap_capture_gpu.py")) + ("adapters/torch/verity_torch/frontend/rules/common.py",)
#: the files whose ``CAP`` and ``CAPS`` in capitals are their own too
CAPITALS = LIVE_CAPS + REQUEST_CAPS + SOFTCAP_CAPS
#: the files whose "cap" is their own, by path prefix, with each reason; Pearl's spellings still count there, and
#: ``PEARL_CAP`` too outside ``CAPITALS``
OWN_CAPS = {
    "verity/protocols/resource_accounting/memory/pous/": "PoUS's caps (an audit's duration, its schemes' bounds); Pearl's is PoUW's",
    "verity/protocols/resource_accounting/memory/poub/": "PoUB's lifetime cap on a schedule's bits; Pearl's is PoUW's",
    **dict.fromkeys(tuple(f"security/{tree}/{area}"
                          for tree in ("Definitions", "Properties", "Proofs", "Proofs/Properties")
                          for area in ("Pous", "Poub")),
                    "PoUS's and PoUB's caps in their Lean (an audit's duration, a schedule's bits, a chip's steps, an "
                    "oracle's queries, a side channel's rate); Pearl's is PoUW's"),
    **dict.fromkeys(("security/Definitions/ResourceAccounting/Gates", "security/Definitions/ResourceAccounting/Guarantees/Gates.lean",
                     "security/Proofs/ResourceAccounting/Gates"),
                    "the stacking gates' caps, what a block's outputs cap in units; Pearl's is PoUW's"),
    **dict.fromkeys(tuple(f"benchmarks/{b}/"
                          for b in ("pous", "remote_session", "vllm_encoded", "kv_filler", "codec_tensor")),
                    "PoUS's and PoUB's benchmarks' caps (an RTT allowance, a period's cap, a buffer's rows, a fill's "
                    "CTAs); Pearl's is PoUW's"),
    "verity/protocols/resource_accounting/compute/pouw/prover/kernels/pouw_cuda/row_table": "a row table's `cap` counts its leaves",
    "verity/protocols/resource_accounting/compute/pouw/verifier/window.py": "a session's cap, the most tile instances one session holds",
    "verity/protocols/resource_accounting/compute/pouw/verifier/lean/window/": "`window.py`'s session cap in its Lean port",
    "tests/verity/protocols/resource_accounting/compute/pouw/verifier/lean/window/": "`window.py`'s session cap in its Lean port's tests",
    "verity/protocols/proofs/flock/": "Flock's caps (a Merkle cap, the nodes a proof's rows open to; its bounds on runs and tables)",
    "verity/protocols/proofs/recursion/": "Flock's Merkle caps, as the recursion's verifier checks them",
    "benchmarks/recursion_e2e/": "Flock's Merkle caps, the V* statements the recursion's benchmark proves and times",
    "benchmarks/recursion_figures/": "Flock's Merkle caps, the V* parts the recursion's figures name",
    "security/Proofs/Flock/": "Flock's caps in its proofs: the Merkle cap, and its bounds on runs, tables and samples",
    "security/Definitions/Claims/": "the bound on an extractor's reruns that Flock's claims state (`Deploy.cap`, `Arg.cap`)",
    **dict.fromkeys(LIVE_CAPS, "the live proof block's caps, its requests' public `max_tokens`; a session's cap"),
    **dict.fromkeys(REQUEST_CAPS, "a request's length cap, its `max_tokens`"),
    **dict.fromkeys(SOFTCAP_CAPS, "a logit soft cap, the softcapped attention chain's `CAP`"),
}
#: Pearl's `CAP`, which the table spells as a word, so no longer term holds it ("over-cap", "cap check", `capOK` and
#: `Device.cap` are terms of their own)
PEARL_CAP = re.compile(r"(?<!\w)CAPS?(?!\w)")
#: names that keep a retired term's word in another sense, by the term: a whole identifier one of its patterns matches,
#: in any file or in the files beside it, is never one of the term's sites, for the reason beside it
OWN_NAMES = {
    "*Period*": (
        (r"(?<=Pous[./])Period|Period(?:Held(?:Coins|Sha512)?(?:At)?|Held(?:Deployed|Demo|Honest|Nontrivial)|Deployed"
         r"|Game|At|Capacity|Witnesses)",
         "PoUS's `Pous.Period` and its guarantees' names (`PeriodHeld`, `PeriodHeldAt`, `PeriodGame`, ...) are one "
         "stretch of a joint run: GLOSSARY.md's row of `*Period*` retires the word only as the stretch of time a bound "
         "is about, and its row of \"window\" makes \"period\" the word for a span of time. Open, and lean's: whether "
         "that stretch should read the one shared period type in `Core`; if it should, the names follow by a map with "
         "a moves file", ()),
        (r"B2Period(?:Audit|P2Ring)?",
         "PoUS's responder guarantees' period is the span a location takes to answer one challenge, at most `Δ + RTT` "
         "ticks: a span of time, which GLOSSARY.md's row of \"window\" calls a period, and not the stretch of time a "
         "bound is about that its row of `*Period*` retires", ()),
        (r"(?<=[Aa]ttest[./])Period",
         "an auditor party's certification period (`Attest.Period`): the stretch whose records one signed root "
         "commits, a period by GLOSSARY.md's row of \"window\", and not the stretch of time a bound is about that its "
         "row of `*Period*` retires", ()),
        (r"Period(?:Signed|Sound|Run)",
         "the certification period of `Attest.Period`, in its guarantees' names (`PeriodSigned`, `PeriodSound`, "
         "`PeriodRun`), in their two files only: elsewhere `PeriodRun` is compose's and `PeriodSound` PoUW's",
         ("security/Properties/Attest.lean", "security/Proofs/Properties/Pous/MemoryChallenger/Attest.lean")),
        (r"_PeriodAuditor",
         "the auditor of H1's challenge period in the vLLM adapter's PoUW tests: a span of time, which GLOSSARY.md's row "
         "of \"window\" calls a period, and not the stretch of time a bound is about that its row of `*Period*` retires",
         ("tests/adapters/vllm/tests/pipeline/test_serve_h1.py", "tests/adapters/vllm/tests/protocols/test_pouw_h1_drawn.py")),
    ),
    "row of record": (
        (r"row[\s_]+of[\s_]+record",
         "vLLM's workload of record: GLOSSARY.md's row of \"row\" for a workload keeps \"row of record\" "
         "(`manifests/records.json`) in `adapters/vllm/` until vLLM's workloads move",
         tuple(_vllm(p) for p in (
             "tests/pipeline/test_circuit_db.py", "tests/pipeline/test_done_source.py", "tests/pipeline/test_neutrality.py",
             "tests/pipeline/test_row_chain.py", "tests/program/test_compact.py", "verity_vllm/config.py",
             "verity_vllm/pipeline/circuit_db.py", "verity_vllm/pipeline/config_record.py",
             "verity_vllm/pipeline/neutrality.py", "verity_vllm/pipeline/research.py", "verity_vllm/pipeline/row.py",
             "verity_vllm/pipeline/row_config.py", "verity_vllm/pipeline/workload.py",
             "verity_vllm/program/gemm_target_correspondence.py"))),
    ),
}
OWN_NAME = {term: tuple((re.compile(pattern), files) for pattern, _, files in names)
            for term, names in OWN_NAMES.items()}
#: keys a stored record or format still carries, by the key as written: (the files that read it, by file, never by
#: directory; the reason, naming the record and its owner).  A site in a file its entry names is held where the key stands
#: there whole: the whole run of identifier characters around it (``HELD_CHAR``, a ``*`` in the key any run of word
#: characters and hyphens), and for a key that is a plain word, the whole quoted string too (between two of one quote:
#: ``"``, ``'`` or a backtick).  The same key in a file its entry doesn't name, the same word in prose and a longer name
#: holding the key still count.  An entry leaves with its format (a key that no longer stands held in one of its files is
#: refused as stale); when an owner writes a new key beside the old one, the entry's reason gets the day the old key stops
#: being read.  A file's entry in the list falls by its held sites (``--prune``) and never rises for them.
HELD_KEYS = {
    "firewall": (("benchmarks/recursion_e2e/rec_profile.py",), "reads a stored recursion run, benchmarks'"),
    "flock_firewall": (("benchmarks/recursion_e2e/rec_profile.py",), "reads a stored recursion run, benchmarks'"),
    "flock-firewall": (("benchmarks/recursion_e2e/rec_profile.py", "benchmarks/recursion_e2e/README.md"),
                       "reads a stored recursion run, benchmarks'"),
    "FIREWALL-REFUSED": (("benchmarks/recursion_e2e/rec_profile.py", "benchmarks/recursion_e2e/README.md",
                          "tests/benchmarks/recursion_e2e/test_recursion_e2e.py"),
                         "reads a stored recursion run, benchmarks'"),
    "t.firewall_recheck": (("benchmarks/recursion_e2e/rec_profile.py",), "reads a stored recursion run, benchmarks'"),
    "sampled-proofs:*": (("benchmarks/harness/README.md", "benchmarks/harness/specs/llama31-8b-fp8-1gpu-commit-served.toml",
                          "tests/benchmarks/harness/test_bench.py", "benchmarks/harness/verity_bench/vllm/commit_arm.py"),
                         "a stored run's arm key, vllm's"),
    "coin_source": (("benchmarks/harness/verity_bench/vllm/runner.py", "tests/benchmarks/harness/test_bench.py"),
                    "the PoUW commitment's key, compute-accounting's"),
}
#: a character of an identifier a held key stands in
HELD_CHAR = re.compile(r"[\w.:-]")
QUOTES = "\"'`"


def held_key(key: str) -> tuple[re.Pattern, bool]:
    """``key``'s pattern over a whole identifier, and whether it is a plain word, held only as a whole quoted string."""
    return re.compile(re.escape(key).replace(r"\*", r"[\w-]+")), bool(re.fullmatch(r"[A-Za-z]+", key))


HELD_KEY = {key: held_key(key) for key in HELD_KEYS}
HELD = {f: tuple(HELD_KEY[k] for k, (files, _) in HELD_KEYS.items() if f in files)
        for files, _ in HELD_KEYS.values() for f in files}
#: hypothesis's own ``draw`` in a module importing it: a composite strategy's first parameter, through its body, and the
#: data object's ``data.draw(``, as code where the file is Python; a comment or a string there, a ``"draw"`` key anywhere, a
#: longer name, an attribute (``challenger.draw``) and the other forms (``draws``, ``drawn``) aren't
HYPOTHESIS = re.compile(r"^[ \t]*(?:from|import)[ \t]+hypothesis\b", re.M)
COMPOSITE = re.compile(r"^([ \t]*)@(?:\w+\.)*composite\b.*\n(?:\1@.*\n)*\1(?:async[ \t]+)?def[ \t]+\w+[ \t]*\(\s*draw\b",
                       re.M)
BARE_DRAW = re.compile(r"(?<![\w.'\"-])draw(?![\w'\"])")
DATA_DRAW = re.compile(r"(?<![\w.])data\.draw\(")
#: a line that can end a block: its indent, then neither a comment nor a closing bracket
LINE = re.compile(r"^([ \t]*)[^\s#)\]}]", re.M)
#: what the word maps keep: a domain tag or format id, a note, an artifact, a branch, a lane
KEEP = re.compile(r"[\w.-]+(?:/[\w.-]+)*/v\d+(?:\.\d+)*\b|note:[\w./-]+|art:[0-9a-f]+|cursor/[\w./-]+|lanes/[\w.-]+")
#: a goal's id (``GROUP/slug``) as a TOML line's ``goal`` key gives it: the value is a name, kept as written
GOAL = re.compile(r'[ \t]*goal[ \t]*=[ \t]*"(?P<id>[\w-]+/[\w-]+)"[ \t]*(?:#.*)?')
#: a character of what ``KEEP`` keeps
ID = re.compile(r"[\w./:-]")
#: a character of one identifier
IDENT = re.compile(r"\w")
FIX = (f"Say the term the row gives (GLOSSARY.md's older names). A use the table keeps (a recorded key, a wire format) "
       f"stays only where it already stands. If the file moved, move its entry in the same change: `python3 "
       f"tests/lint/test_{NAME}.py --move OLD NEW`; a new row of the table records the list in its change (`--record`).")

PY = (".py", ".pyi")
LEAN = (".lean",)
MARKDOWN = (".md",)
HASH = (".sh", ".bash", ".toml", ".yaml", ".yml", ".service", ".timer", ".conf", ".cfg", ".ini")
SLASH = (".rs", ".c", ".h", ".cc", ".cpp", ".hpp", ".cu", ".cuh", ".ts", ".js", ".go")

QUOTE = re.compile(r'"([^"\n]+)"')
TICK = re.compile(r"`([^`\n]+)`")
NARROWS = re.compile(r"\b(?:for|meaning|naming|in|as|of)\b")
#: an identifier's parts, as the lint splits a term and a file into words: camel humps, runs of capitals, digits, letters
WORDS = re.compile(r"[A-Z]?[a-zδκ]+|[A-Z]+(?![a-z])|\d+|[δκεΔ]")
INFLECT = {"noun": ("", "s", "es"), "verb": ("", "s", "es", "n", "ing", "ed")}


def holds(path: str) -> bool:
    """Whether this lint reads ``path``: code, Lean, config and Markdown outside ``LEFT_OUT``, ``SKIP``, test fixtures and
    changelogs."""
    if path.startswith(LEFT_OUT) or any(path == s or s.endswith("/") and path.startswith(s) for s in SKIP):
        return False
    if any(d in FIXTURE_DIRS for d in path.split("/")[:-1]) or path.rsplit("/", 1)[-1] in HISTORY:
        return False
    return path.endswith(PY + LEAN + MARKDOWN + HASH + SLASH)


class Term:
    """One retired name: its spelling in the table, what the table says instead, and how it is matched."""

    def __init__(self, old: str, new: str, inflect: str = "noun", code: bool = False):
        self.old, self.new = old, new
        if code:
            body = re.escape(old).replace(r"\.\*", r"\.\w+").replace(r"\*", r"\w*")
            scan = body
        else:
            parts = [p for p in re.split(r"[\s_-]+", old) if p]
            ends = sorted({e for s in INFLECT[inflect] for e in (s, s.upper())}, key=len, reverse=True)
            body = r"[\s_-]?".join(map(self._part, parts)) + "(?:" + "|".join(ends) + ")"
            scan = r"[\s_-]?".join(re.escape(p.lower()) for p in parts) + "(?:" + "|".join(e for e in ends
                                                                                        if e.islower() or not e) + ")"
        self.pattern, self.scan = re.compile(body), re.compile(scan)
        self.anchors = frozenset(w.lower() + s for w in [max(WORDS.findall(old) or [old], key=len)]
                                 for s in INFLECT["verb"])
        self.code = code
        self.says = self._says(new) if code and "*" in old else None

    @staticmethod
    def _says(new: str) -> re.Pattern | None:
        """An identifier that says the term ``new`` gives (its words before a parenthesis, ``;`` or ``,``): those words
        in CamelCase as spelled but for the first letter's case, or in snake_case in any case; None when they aren't
        plain words."""
        words = re.split(r"[(;,]", new, maxsplit=1)[0].split()
        if not words or not all(re.fullmatch(r"[A-Za-z0-9]+", w) for w in words):
            return None
        camel = "".join(w[0].upper() + w[1:] for w in words)
        return re.compile(f"[{camel[0]}{camel[0].lower()}]{re.escape(camel[1:])}|(?i:{re.escape('_'.join(words))})")

    @staticmethod
    def _part(p: str) -> str:
        if p.islower():
            return f"(?:[{p[0]}{p[0].upper()}]{re.escape(p[1:])}|{re.escape(p.upper())})"
        return f"(?:{re.escape(p)}|{re.escape(p.upper())})" if p.upper() != p else re.escape(p)

    def matches(self, text: str, low: str):
        """The spans of this term in ``text`` (``low``, its lower case of the same length) where it stands as a word or as
        one part of an identifier: after no letter or digit but at a camel hump, and before no lower-case letter, nor a
        capital after a capital; a code term before no identifier character.  A word is found in ``low`` by its literal,
        which is fast, and its case then checked in ``text``."""
        found = (self.pattern.finditer(text) if self.code else
                 (m for m in self.scan.finditer(low) if self.pattern.fullmatch(text, *m.span())))
        for m in found:
            s, e = m.span()
            before = text[s - 1] if s else ""
            if before.isalnum() and not (text[s].isupper() and (before.islower() or before.isdigit())):
                continue
            after = text[e] if e < len(text) else ""
            if self.code:
                if after.isalnum() or after == "_":
                    continue
            elif after.isalpha() and (after.islower() or text[e - 1].isupper()):
                continue
            yield s, e


def terms_of(glossary: str) -> list[Term]:
    """The retired terms of ``glossary`` (a ``GLOSSARY.md``'s text), longest first."""
    table = glossary.split("## Older names", 1)[-1].split("\n## ", 1)[0]
    two = glossary.split("still carry two senses", 1)[-1].split("\n## ", 1)[0] if "still carry two senses" in glossary else ""
    narrow = {w.lower() for w in re.findall(r"^\s*- \*\*([^*]+)\*\*", two, re.M)} - SENSED.keys()
    live = {w.lower() for w in LIVE}
    out: dict[str, Term] = {}
    for row in table.splitlines():
        cells = [c.strip() for c in re.split(r"(?<!\\)\|", row)[1:-1]]
        if len(cells) < 2 or set(cells[0]) <= {"-"} or cells[0].startswith("Old name"):
            continue
        new = re.sub(r"\s+", " ", cells[-1])
        for clause in cells[0].split(";"):
            quotes, narrowed, words = list(QUOTE.finditer(clause)), False, []
            for i, m in enumerate(quotes):
                if clause[:m.start()].rstrip().endswith("outside"):
                    continue
                after = clause[m.end():quotes[i + 1].start() if i + 1 < len(quotes) else len(clause)].split("`", 1)[0]
                paren = re.match(r"\s*\(([^)]*)\)", after)
                rest = after[paren.end():] if paren else after
                if re.match(r"\s*alone\b", rest):
                    continue
                narrowed |= bool(NARROWS.search(rest))
                senses = paren.group(1) if paren else ""
                if "," in senses and senses != "noun, verb" and " " not in m.group(1):
                    words += [(f"{m.group(1)} {s.strip()}", "noun") for s in senses.split(",")]
                else:
                    words.append((m.group(1), "verb" if senses == "noun, verb" else "noun"))
            if not narrowed:
                for w, inflect in words:
                    if w.lower() not in narrow | live:
                        out.setdefault(w, Term(w, new, inflect))
            if "until renamed" in clause:
                for m in TICK.finditer(clause):
                    w = m.group(1)
                    owner = clause.startswith("'s", m.end())
                    if not owner and " " not in w and "/" not in w and re.search(r"[._*-]|[a-z][A-Z]", w):
                        out.setdefault(w, Term(w, new, code=True))
    return sorted(out.values(), key=lambda t: -len(t.old))


def glossary_text() -> str:
    """``GLOSSARY.md`` of the tree this module stands in, or (loaded alone, as the ratchet gate loads it) of the tree the
    command runs in."""
    if (ROOT / GLOSSARY).is_file():
        return (ROOT / GLOSSARY).read_text(encoding="utf-8")
    top = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True).stdout.strip()
    return (Path(top) / GLOSSARY).read_text(encoding="utf-8")


@lru_cache(maxsize=None)
def terms() -> tuple[tuple[Term, ...], dict[str, list[Term]]]:
    """The retired terms, and each by the words it is found by."""
    found = tuple(terms_of(glossary_text()))
    by: dict[str, list[Term]] = {}
    for t in found:
        for a in t.anchors:
            by.setdefault(a, []).append(t)
    return found, by


def sites_of(text: str, by: dict[str, list[Term]], path: str = "",
             keys: tuple[tuple[re.Pattern, bool], ...] | None = None) -> list[tuple[int, str]]:
    """(line, ``'X' is retired; say Y``) of each site of a retired term in ``text``, ``by`` its terms by word, ``path`` the
    file's, outside the keys it holds (``keys``, by default ``HELD``'s for ``path``).  Where two terms match one span, the
    one the table spells longer names it, then the first by spelling."""
    keys = HELD.get(path, ()) if keys is None else keys
    words = {w.lower() for w in set(WORDS.findall(text))}
    candidates = sorted({t for w in words & by.keys() for t in by[w]}, key=lambda t: (-len(t.old), t.old))
    if not candidates:
        return []
    low = text.lower()
    if len(low) != len(text):                           # a character whose lower case is longer: lower it alone
        low = "".join(c if len(c.lower()) != 1 else c.lower() for c in text)
    hits = sorted(((s, e, t) for t in candidates for s, e in t.matches(text, low)), key=lambda h: (h[0], -h[1]))
    draws, caps, toml = hypothesis_draws(text, path), path.startswith(tuple(OWN_CAPS)), path.endswith(".toml")
    out, end, line, at = [], -1, 1, 0
    for s, e, t in hits:
        if s < end or text[s:e] == "draw" and s in draws or kept(text, s, e, caps, toml, path in CAPITALS, t.says):
            continue
        if any(n.fullmatch(text, s, e) and (not files or path in files) for n, files in OWN_NAME.get(t.old, ())):
            continue
        if keys and held(text, s, e, keys):
            continue
        end = e
        line, at = line + text.count("\n", at, s), s
        out.append((line, f"{text[s:e]!r} is retired; say {t.new[:120]} (GLOSSARY.md)"))
    return out


def held(text: str, s: int, e: int, keys: tuple[tuple[re.Pattern, bool], ...]) -> bool:
    """Whether the site ``text[s:e]`` stands in one of ``keys`` whole (``HELD_KEYS``): the run of ``HELD_CHAR`` around it
    is the key, and quoted where the key is a plain word."""
    a, b = s, e
    while a and HELD_CHAR.match(text, a - 1):
        a -= 1
    while b < len(text) and HELD_CHAR.match(text, b):
        b += 1
    quoted = 0 < a and b < len(text) and text[a - 1] in QUOTES and text[b] == text[a - 1]
    return any(key.fullmatch(text, a, b) and (quoted or not plain) for key, plain in keys)


def stale(entries: dict[str, tuple[tuple[str, ...], str]]) -> list[str]:
    """Each file of ``entries`` (``HELD_KEYS``' form) that isn't a file this lint reads, or where its key no longer stands
    held at a site of a retired term, with the fix."""
    out = []
    for key, (paths, reason) in entries.items():
        if not paths or not reason:
            out.append(f"{key!r}: an entry names the files that read its key and the reason, naming the record and its owner")
        for f in paths:
            if not (holds(f) and (ROOT / f).is_file()):
                out.append(f"{key!r}: {f} is not a file this lint reads: name each file that reads the key, never a "
                           f"directory, and move its name with the file")
                continue
            text = (ROOT / f).read_text(encoding="utf-8", errors="replace")
            if len(sites_of(text, terms()[1], f, ())) == len(sites_of(text, terms()[1], f, (held_key(key),))):
                out.append(f"{key!r} no longer stands held in {f}: the entry leaves with its format, so take {f} out of "
                           f"HELD_KEYS' entry (or the entry, if it was the last file)")
    return out


def kept(text: str, s: int, e: int, own_caps: bool = False, toml: bool = False, live: bool = False,
         says: re.Pattern | None = None) -> bool:
    """Whether the site ``text[s:e]`` stands in what ``KEEP`` keeps or in ``OTHER_SENSES``, read from the run of id
    characters around it and the few words about it, in a goal's id on a TOML file's ``GOAL`` line (``toml``), or is a
    "cap" in a file ``OWN_CAPS`` names (``own_caps``), other than ``PEARL_CAP`` outside ``CAPITALS`` (``live``), or
    stands in an identifier that already says its row's term (``says``, a ``*`` name's ``Term.says``)."""
    a, b = s, e
    while a and ID.match(text, a - 1):
        a -= 1
    while b < len(text) and ID.match(text, b):
        b += 1
    if any(m.start() <= s - a < m.end() for m in KEEP.finditer(text[a:b])):
        return True
    if says:
        i, j = s, e
        while i and IDENT.match(text, i - 1):
            i -= 1
        while j < len(text) and IDENT.match(text, j):
            j += 1
        if says.search(text, i, j):
            return True
    if toml:
        start = text.rfind("\n", 0, s) + 1
        stop = text.find("\n", e)
        goal = GOAL.fullmatch(text, start, len(text) if stop < 0 else stop)
        if goal and goal.start("id") <= s and e <= goal.end("id"):
            return True
    if own_caps and text[s:e].lower() in ("cap", "caps") and (live or not PEARL_CAP.match(text, s)):
        return True
    a = max(0, s - 40)
    return any(m.start() <= s - a < m.end() for m in OTHER_SENSES.finditer(text[a:e + 40]))


def hypothesis_draws(text: str, path: str = "") -> frozenset[int]:
    """Where ``text``, at ``path``, says hypothesis's own ``draw`` (``HYPOTHESIS``): the start of each.  A composite's body
    runs to the first line after its header, not blank, a comment or a closing bracket, indented no deeper than its
    decorator.  In a Python file only code counts; any other file's text has no code to tell from prose, so every bare
    ``draw`` in the body counts."""
    if "draw" not in text or not HYPOTHESIS.search(text):
        return frozenset()
    out = {m.end() - len("draw(") for m in DATA_DRAW.finditer(text)}
    for m in COMPOSITE.finditer(text):
        after = text.find("\n", m.end()) + 1 or len(text)
        end = next((n.start() for n in LINE.finditer(text, after) if len(n.group(1)) <= len(m.group(1))), len(text))
        out |= {d.start() for d in BARE_DRAW.finditer(text, m.end() - len("draw"), end)}
    if not out or not path.endswith(PY):
        return frozenset(out)
    return frozenset(out) & code_names(text, "draw", path)


def code_names(text: str, name: str, path: str) -> frozenset[int]:
    """Where ``text``, the Python file ``path``, says ``name`` as code, outside comments and strings: the start of each.
    Raises ``ValueError`` naming ``path`` where the text doesn't tokenize."""
    starts = [0]
    for line in text.split("\n"):
        starts.append(starts[-1] + len(line) + 1)
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, SyntaxError) as e:
        raise ValueError(f"{path}: doesn't tokenize as Python ({e}), so the retired-terms lint can't tell hypothesis's "
                         f"`draw` from prose in it. Fix the file's syntax.") from e
    return frozenset(starts[t.start[0] - 1] + t.start[1] for t in tokens if t.type == tokenize.NAME and t.string == name)


def sites(text: str, path: str) -> list[tuple[int, str]]:
    """(line, what) of each site of a retired term in ``path``, ``text`` its content."""
    if not holds(path):
        return []
    return sites_of(text, terms()[1], path)


def parse(doc: object) -> Counter:
    """Each file's allowed sites, from a list's document."""
    assert isinstance(doc, dict) and isinstance(doc.get("entries"), list), f"{LIST.name}: {{'entries': [...]}} expected"
    counts: Counter = Counter()
    for e in doc["entries"]:
        assert isinstance(e, dict) and set(e) == {"file", "count"} and isinstance(e["file"], str), \
            f"{LIST.name}: an entry is {{file, count}}: {e}"
        assert type(e["count"]) is int and e["count"] >= 1, f"{LIST.name}: count must be a positive int: {e}"
        assert e["file"] not in counts, f"{LIST.name}: duplicate entry {e}"
        counts[e["file"]] = e["count"]
    return counts


def files() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout
    return sorted(p for p in out.decode().split("\0") if p and holds(p))


@lru_cache(maxsize=None)
def scan() -> tuple[tuple[str, int, str], ...]:
    """Each site's file, line and text."""
    return tuple((f, line, what) for f in files() if (ROOT / f).is_file()
                 for line, what in sites((ROOT / f).read_text(encoding="utf-8", errors="replace"), f))


def allowed() -> Counter:
    return parse(json.loads(LIST.read_text(encoding="utf-8")))


def found() -> Counter:
    return Counter(f for f, *_ in scan())


def test_no_new_retired_terms() -> None:
    have = allowed()
    where: dict[str, list[tuple[int, str]]] = {}
    for f, line, what in scan():
        where.setdefault(f, []).append((line, what))
    new = [f"  {f}: {n} found, {have[f]} allowed\n" + "\n".join(f"    {f}:{line} {what}" for line, what in where[f])
           for f, n in sorted(found().items()) if n > have[f]]
    assert not new, (f"retired terms: {len(new)} file(s) hold more than tests/lint/allowlists/{NAME}.json allows. {FIX}\n"
                     + "\n".join(new))


GLOSSARY_EXAMPLE = """# Glossary
## Older names

| Old name, where it stays | Term |
|---|---|
| "C-Flock"; the stored label `C-Flock` | Flock |
| "law", "draw law"; `law` record keys, the Lean's `Law` | sampling design |
| "draw" (noun, verb), "drawn unit"; `DrawIsCoin`, `Remote.Draw.Session`, `--draw`, until renamed | the sample |
| "firewall"; `flock-firewall`, `Flock.Firewall`, `FC_FIREWALL*`, until renamed | the ZK server |
| "proof unit", "unit class", and "unit" alone meaning one | proof block |
| "components", "product" for a composite's parts | subprotocols; composition |
| "stage", outside "one-stage" | phase |
| "dark" (compute, fleet) | unregistered hardware |
| "session"; "window" for a span of time; `Verity.Window`, `window`, until renamed | a run |
| "IR", "Verity IR" | circuit |

## Narrower senses

- Two words still carry two senses in the code:
  - **Stage**: to stage a statement.
"""


def test_what_is_a_retired_term() -> None:
    got = {t.old: t for t in terms_of(GLOSSARY_EXAMPLE)}
    assert set(got) == {"C-Flock", "law", "draw law", "draw", "drawn unit", "DrawIsCoin", "Remote.Draw.Session", "--draw",
                        "firewall", "flock-firewall", "Flock.Firewall", "FC_FIREWALL*", "proof unit", "unit class",
                        "dark compute", "dark fleet", "Verity.Window", "Verity IR"}, sorted(got)
    by: dict[str, list[Term]] = {}
    for t in got.values():
        for a in t.anchors:
            by.setdefault(a, []).append(t)

    def olds(text: str) -> list[str]:
        return [w.split(" is retired")[0].strip("'\"") for _, w in sites_of(text, by)]
    assert olds("C-Flock's draw law; its drawLaw, draw_law and DRAW_LAW; drawn units, draws, drawing") == [
        "C-Flock", "draw law", "drawLaw", "draw_law", "DRAW_LAW", "drawn units", "draws", "drawing"]
    assert olds("withdraw, drawback, flaw, lawful, Lawrence, unit, a unit of work, one-stage, stage, units_per_vu") == []
    assert olds("a power law, Amdahl's law, the power draw; parse_law, LAW, Law.stratified, lawOf") == [
        "law", "LAW", "Law", "law"]
    assert olds("Flock.Firewall.Frame, FC_FIREWALL_LEAN, flock-firewall, the firewall's, firewall_on") == [
        "Flock.Firewall", "FC_FIREWALL_LEAN", "flock-firewall", "firewall", "firewall"]
    assert olds("proofUnit, ProofUnits, PROOF_UNIT, proof-unit, unit classes, dark compute, the IR, ir_lower, Verity IR") == [
        "proofUnit", "ProofUnits", "PROOF_UNIT", "proof-unit", "unit classes", "dark compute", "Verity IR"]
    assert olds("DrawIsCoin, Remote.Draw.Session, --draw K, Verity.Window, Verity.Windows") == [
        "DrawIsCoin", "Remote.Draw.Session", "--draw", "Verity.Window"]
    assert olds("the domain verity/flock-firewall/v1 and note:x-draw-law, a session, the window") == []
    assert sites_of("a\nb C-Flock\n", by) == [(2, "'C-Flock' is retired; say Flock (GLOSSARY.md)")]


SENSES_EXAMPLE = """# Glossary
## Older names

| Old name, where it stays | Term |
|---|---|
| "draw" (noun, verb), "drawn unit"; `DrawIsCoin`, `--draw`, until renamed | the sample |
| Pearl's "debit" and "cap": "cap", "cap check", "over-cap" tile; `capOK`, `Device.cap`, `CAP`, until renamed | the bound |
| "production" | serving, or proving, by sense |
"""


def said(text: str, path: str = "") -> list[tuple[int, str]]:
    """(line, the old name) of each site of ``SENSES_EXAMPLE``'s terms in ``text``, at ``path``; the default path isn't a
    Python file's, so hypothesis's ``draw`` is found by the rule for text that isn't Python."""
    by: dict[str, list[Term]] = {}
    for t in terms_of(SENSES_EXAMPLE):
        for a in t.anchors:
            by.setdefault(a, []).append(t)
    return [(line, w.split(" is retired")[0].strip("'\"")) for line, w in sites_of(text, by, path)]


def test_nvidia_smi_s_power_draw_is_another_sense_and_the_sample_beside_it_is_not() -> None:
    assert said("nvidia-smi --query-gpu=clocks.sm,power.draw,power.limit; POWER_DRAW, power_draw, its power draws") == []
    assert said("q = 'power.draw,clocks.sm'  # then challenger.draw(k), --draw 3 and the draws") == [
        (1, "draw"), (1, "--draw"), (1, "draws")]


HYPOTHESIS_EXAMPLE = '''from hypothesis import given, strategies as st


@st.composite
def cases(draw, n: int = 3) -> dict:
    k = draw(st.integers(0, n))
    rows = helper(draw, k)
    return {"draw": challenger.draw(k), "units": drawn, "n": draws, "flag": ["--draw"], "coin": DrawIsCoin}


def helper(draw, k):
    return draw(st.just(k))


@given(st.data())
def test_cases(data) -> None:
    data.draw(cases())
'''


def test_hypothesis_s_draw_is_another_sense_and_the_sample_beside_it_is_not() -> None:
    """A composite's first parameter and its uses in the body, and ``data.draw(``, are hypothesis's; in that body a string,
    an attribute, another form or a longer name is the sample, and so is a ``draw`` past the body (an undecorated helper's
    ``draw`` may be the sample: a test's parameter named for it)."""
    sample = [(8, "draw"), (8, "draw"), (8, "drawn"), (8, "draws"), (8, "--draw"), (8, "DrawIsCoin"), (11, "draw"),
              (12, "draw")]
    assert said(HYPOTHESIS_EXAMPLE) == sample
    bare = HYPOTHESIS_EXAMPLE.replace("from hypothesis import given, strategies as st",
                                      "from hypothesis.strategies import composite").replace("@st.composite", "@composite")
    assert said(bare) == sample
    assert sorted(said(HYPOTHESIS_EXAMPLE.replace("from hypothesis", "from elsewhere"))) == sorted(
        sample + [(5, "draw"), (6, "draw"), (7, "draw"), (17, "draw")])


def test_a_draw_in_a_comment_or_string_of_a_composite_body_is_still_the_sample() -> None:
    """In a Python file only code is hypothesis's: the word in a comment or a docstring inside a composite's body is
    prose."""
    text = ('from hypothesis import strategies as st\n\n\n@st.composite\ndef units(draw):\n'
            '    """Each unit we draw, one by one."""\n'
            '    k = draw(st.integers(0, 3))  # draw the count, then the units\n'
            '    return [draw(st.just(i)) for i in range(k)]  # draw\n')
    assert said(text, "verity/protocols/proofs/spot/tests/test_units.py") == [(6, "draw"), (7, "draw"), (8, "draw")]


def test_a_composite_shown_in_markdown_keeps_every_bare_draw_of_its_body() -> None:
    """A file that isn't Python is never read as Python, so an example in Markdown with prose after it (an apostrophe,
    which doesn't tokenize) doesn't stop the scan. In its body every bare ``draw`` is kept, the comment's too: this is the
    rule before the code-only one, on purpose, since such a text has no code to tell from prose."""
    text = ("# Units\n\n```python\nfrom hypothesis import strategies as st\n\n\n@st.composite\ndef units(draw):\n"
            "    k = draw(st.integers(0, 3))  # draw the count\n    return [draw(st.just(i)) for i in range(k)]\n```\n\n"
            "It's the sample that counts past the example: one draw.\n")
    assert said(text, "verity/protocols/proofs/spot/README.md") == [(13, "draw")]


def test_a_python_file_that_doesn_t_tokenize_stops_the_scan_naming_itself() -> None:
    path = "verity/protocols/proofs/spot/tests/test_broken.py"
    text = 'from hypothesis import given, strategies as st\n\n\n@given(st.data())\ndef test_it(data):\n    data.draw(st.just("x))\n'
    try:
        said(text, path)
    except ValueError as e:
        assert path in str(e) and "Fix the file's syntax" in str(e), e
    else:
        raise AssertionError(f"{path} doesn't tokenize, and the scan read it anyway")


def test_the_stacking_property_s_caps_and_nvidia_smi_s_compute_cap_are_other_senses_and_a_cap_beside_them_is_not() -> None:
    """The stacking property's caps by the names its Lean gives them, and nvidia-smi's field, anywhere; a "cap" beside
    them, and a longer name built on one, still count, in their directories too."""
    theirs = ("R_cap, R_cap_phase, R_cap_node, R_cap_node_by_class_peak, caps.total, `Gates.FoolingCap`; "
              'nvidia-smi --query-gpu=name,compute_cap; {"compute_cap": "12.0"}')
    assert said(theirs, "benchmarks/stacked_share/share.py") == []
    assert said("R_cap is the cap; R_capRows, caps.totals, FoolingCaps, a compute cap, compute_cap_max and its cap",
                "benchmarks/stacked_share/beta.py") == [(1, "cap"), (1, "cap"), (1, "caps"), (1, "Caps"), (1, "cap"),
                                                         (1, "cap"), (1, "cap")]


def test_a_kernel_s_grid_cap_is_another_sense_and_pearl_s_cap_beside_it_is_not() -> None:
    """A kernel's ``grid_cap`` as the identifier it is; Pearl's spellings beside it, and a longer name built on it, still
    count in the same file."""
    path = "verity/protocols/resource_accounting/compute/pouw/prover/kernels/masked_gemm/sgemm.cu"
    theirs = "static void ssetup(SProb& P, int grid_cap) {\n  P.grid = std::min(tiles, grid_cap > 0 ? grid_cap : nsm);\n"
    assert said(theirs, path) == []
    assert said(theirs + "// grid_cap, then Pearl's cap, CAP and capOK; grid_capOK, grid_cap_check\n", path) == [
        (3, "cap"), (3, "CAP"), (3, "capOK"), (3, "capOK"), (3, "cap_check")]


def test_a_registered_compiler_s_cap_bits_and_cap_l_are_another_sense_and_a_cap_beside_them_is_not() -> None:
    """The public length ℓ a registered compiler carries, as ``cap_bits``, ``CAP_BITS`` and "the cap ℓ"; a plain "cap"
    beside them, and another spelling of the name, still count in the same file."""
    path = "adapters/vllm/verity_vllm/pipeline/advice.py"
    theirs = 'CAP_BITS = 4096\nassert len(bits) <= spec.cap_bits, "over the cap ℓ"  # cap_bits=CAP_BITS\n'
    assert said(theirs, path) == []
    assert said(theirs + "# the cap, Pearl's CAP and capBits beside cap_bits\n", path) == [
        (3, "cap"), (3, "CAP"), (3, "cap")]


def test_a_goal_s_id_as_a_toml_goal_key_is_a_name_and_the_title_beside_it_is_not() -> None:
    """A goal's id is a name, kept as written where a TOML line's ``goal`` key gives it. The word in the title beside
    it, in a comment on the goal's line, in another key's value, and in the same line outside a TOML file still
    counts."""
    path = "benchmarks/prod_repro/metrics.toml"
    goal = 'goal = "vllm/production-vllm-reproducible"\n'
    row = '[[metric]]\nid = "vllm/prod-cost"\ntitle = "the production path, reproduced"\n' + goal
    assert said(goal, path) == []
    assert said(row, path) == [(3, "production")]
    assert said(goal.replace("\n", "  # the production path\n"), path) == [(1, "production")]
    assert said(goal.replace("goal", "note"), path) == [(1, "production")]
    assert said(goal, "benchmarks/prod_repro/metrics.py") == [(1, "production")]


def test_a_cap_of_pous_or_poub_is_another_sense_and_pearl_s_beside_it_is_not() -> None:
    """PoUS's and PoUB's caps by their names anywhere, and any "cap" in their protocols' files; Pearl's spellings
    everywhere, in those files too, against the real table as well."""
    theirs = "the cap: `Params.schedule_cap`, SCHEDULE_CAP, `RTT_ALLOWANCE_CAP_NS`, cap_ns, `AUDIT_CAP`, `auditCap`; v.cap, caps"
    pearls = "\nPearl's: capOK, Device.cap, CAP, an over-cap tile, the cap check"
    pearl = [(2, "capOK"), (2, "Device.cap"), (2, "CAP"), (2, "over-cap"), (2, "cap check")]
    for path in ("verity/protocols/resource_accounting/memory/pous/verifier/audit.py", "verity/protocols/resource_accounting/memory/poub/verifier/audit.py"):
        assert said(theirs + pearls, path) == pearl
    assert said(theirs + pearls, "verity/protocols/resource_accounting/compute/pouw/verifier/pearl.py") == [(1, "cap"), (1, "cap"), (1, "caps"), *pearl]
    assert [w.split("'")[1] for _, w in sites(pearls, "verity/protocols/resource_accounting/memory/poub/verifier/audit.py")] == [o for _, o in pearl]


def test_a_row_table_s_cap_is_its_number_of_leaves_and_pearl_s_beside_it_is_not() -> None:
    """A row table's ``cap`` in its kernel's three files; Pearl's spellings there too, and the same word in a kernel
    beside them."""
    table = "uint64_t cap;  // its leaves\nif (p < t->cap) return 2 * cap;"
    pearls = "\n// Pearl's: capOK, CAP, an over-cap tile"
    pearl = [(3, "capOK"), (3, "CAP"), (3, "over-cap")]
    for path in ("verity/protocols/resource_accounting/compute/pouw/prover/kernels/pouw_cuda/row_table.cu",
                 "verity/protocols/resource_accounting/compute/pouw/prover/kernels/pouw_cuda/row_table.cuh",
                 "verity/protocols/resource_accounting/compute/pouw/prover/kernels/pouw_cuda/row_table_host.cpp"):
        assert Path(ROOT, path).is_file(), f"{path} is gone: move its line of OWN_CAPS with it"
        assert said(table + pearls, path) == pearl
    beside = "verity/protocols/resource_accounting/compute/pouw/prover/kernels/pouw_cuda/pearl_tile.cu"
    assert said(table + pearls, beside) == [(1, "cap"), (2, "cap"), (2, "cap"), *pearl]


def test_vendor_kernel_and_isa_names_are_other_senses_and_pearl_s_cap_in_their_shape_is_not() -> None:
    """Each name as the vendor, kernel or theory spells it is kept; Pearl's spellings in the same shape, beside it, are
    counted."""
    cases = {
        "nvidia-smi's software power cap": (
            "BAD_THROTTLE = {0x4: \"sw_power_cap\"}; $HELD.sw_power_cap", "power_cap, hw_power_cap", ["cap", "cap"]),
        "a logit soft cap": (
            "Attention(logits_soft_cap=30.0); LogitsProcessor(soft_cap=30); attn_logits_soft_cap", "soft_capOK, hard_cap",
            ["capOK", "cap"]),
        "cuBLASLt's algorithm capabilities": (
            "cap_list<uint32_t>(&base, CUBLASLT_ALGO_CAP_TILE_IDS); cap_get<int32_t>(&a, CUBLASLT_ALGO_CAP_SPLITK_SUPPORT); "
            "cublasLtMatmulAlgoCapGetAttribute(a, attr); cublasLtMatmulAlgoCapAttributes_t attr",
            "cap_set(&a, ALGO_CAP_TILE_IDS); MatmulAlgoCapOK; cap_getter", ["cap", "CAP", "Cap", "cap"]),
        "Linux capabilities": (
            "CapabilityBoundingSet=CAP_CHOWN CAP_FOWNER CAP_DAC_OVERRIDE CAP_DAC_READ_SEARCH CAP_IPC_LOCK CAP_SYS_ADMIN; "
            "setcap cap_sys_ptrace+ep; needs CAP_SYS_PTRACE", "CAP_CHECK, CAP_TILE, cap_sys", ["CAP_CHECK", "CAP", "cap"]),
        "the ISA capability bound": (
            "ISA-CAP (<= 32/7) in isa_cap.c, checked by isa_cap_check.py", "ISA CAP, isa_capOK, the cap check",
            ["CAP", "capOK", "cap check"]),
    }
    for sense, (theirs, pearls, pearl) in cases.items():
        assert said(theirs, "benchmarks/pouw/harness/native/lt.cu") == [], sense
        assert said(f"{theirs}\n{pearls}\n", "benchmarks/pouw/harness/native/lt.cu") == [(2, o) for o in pearl], sense


def test_the_live_block_s_caps_are_its_own_in_capitals_too_and_pearl_s_other_spellings_there_are_not() -> None:
    """Every "cap" in the live proof block's three files, its table's role ``CAPS`` and its request's index ``CAP``
    among them; Pearl's other spellings there still count, and so does every "cap" of the same text in ``credit.py``,
    which names the table among Pearl's lines."""
    live = ('CAPS = "caps"\nINDEX, ROUND, SLOT, CAP, PROMPT = 0, 1, 2, 3, 4\n'
            'ports |= {f"cap_{i}": known(table(prefix, CAPS)) for i in range(T)}  # each public cap\n')
    pearls = "# Pearl's: capOK, Device.cap, an over-cap tile, the cap check\n"
    pearl = [(4, "capOK"), (4, "Device.cap"), (4, "over-cap"), (4, "cap check")]
    for path in LIVE_CAPS:
        assert Path(ROOT, path).is_file(), f"{path} is gone: move its line of LIVE_CAPS with it"
        assert said(live + pearls, path) == pearl
    credit = "verity/protocols/resource_accounting/compute/pouw/verifier/credit.py"
    assert said(live + pearls, credit) == [(1, "CAPS"), (1, "caps"), (2, "CAP"), (3, "cap"), (3, "CAPS"), (3, "cap"),
                                           *pearl]


def test_a_session_s_cap_is_its_own_and_pearl_s_other_spellings_there_are_not() -> None:
    """A PoUW session's cap, the most tile instances one session holds, in ``window.py`` and its Lean port, and ``CAP``
    too in its two tests, which name no Pearl term; Pearl's other spellings there still count, a ``CAP`` outside those
    tests does, and so does every "cap" of the same text in ``credit.py``."""
    session = ("def sessions(k: int, cap: int) -> int:  # S sessions of `cap` instances: session_cap, sessionCap?\n"
               "CAP = 2\nWd.public_inputs(reg, SAMPLED, s, V.CAP, domain_h3)\n")
    pearls = "# Pearl's: capOK, Device.cap, an over-cap tile, the cap check\n"
    pearl = [(4, "capOK"), (4, "Device.cap"), (4, "over-cap"), (4, "cap check")]
    window = "verity/protocols/resource_accounting/compute/pouw/verifier/"
    for path in ("test_pouw_window.py", "window_vectors.py"):
        assert Path(ROOT, "tests", window, path).is_file(), f"tests/{window}{path} is gone: move its line of LIVE_CAPS with it"
        assert said(session + pearls, f"tests/{window}{path}") == pearl
    for path in (*(window + p for p in ("window.py", "lean/window/PouwWindow/Inputs.lean",
                                        "lean/window/PouwWindow/Difftest.lean", "lean/window/README.md")),
                 f"tests/{window}lean/window/test_difftest.py"):
        assert Path(ROOT, path).is_file(), f"{path} is gone: move its line of OWN_CAPS with it"
        assert said(session + pearls, path) == [(2, "CAP"), (3, "CAP"), *pearl]
    assert said(session + pearls, window + "credit.py") == [(1, "cap"), (1, "cap"), (1, "cap"), (1, "Cap"), (2, "CAP"),
                                                            (3, "CAP"), *pearl]


def test_a_request_s_length_cap_is_its_own_in_capitals_too_and_pearl_s_other_spellings_there_are_not() -> None:
    """A request's length cap, its ``max_tokens``, in the adapter's continuation, workload and executed-prefix files and
    their tests, ``CAP`` among them, and by its names anywhere; Pearl's spellings there still count, and so does every
    "cap" of the same text in a file not listed."""
    request = ("CAP = D_MAX + 1\n"
               "return L.lift_request_component(component.fn, (17,), max_tokens=CAP, lag=0, boundary_prefix=BOUNDARY)\n")
    pearls = "# Pearl's: capOK, Device.cap, an over-cap tile, the cap check\n"
    pearl = [(3, "capOK"), (3, "Device.cap"), (3, "over-cap"), (3, "cap check")]
    for path in REQUEST_CAPS:
        assert Path(ROOT, path).is_file(), f"{path} is gone: move its line of REQUEST_CAPS with it"
        assert said(request + pearls, path) == pearl
    assert said(request + pearls, "adapters/vllm/verity_vllm/program/frontend/compile.py") == [(1, "CAP"), (2, "CAP"),
                                                                                               *pearl]
    names = ('assert (r1["served_output_len"], r1["executed_prefix_len"], r1["cap_exhausted"], r1["eos_stop"]) == '
             '(32, 32, True, False)\nmax_tokens_cap: int | None = None  # stopped_before_cap\n')
    assert said(names, "tests/adapters/vllm/tests/check/test_executed_prefix.py") == []
    assert said(names + "# the response cap, cap_exhaust\n", "tests/adapters/vllm/tests/check/test_executed_prefix.py") == [
        (3, "cap"), (3, "cap")]


def test_a_logit_soft_cap_is_its_own_in_capitals_too_and_pearl_s_other_spellings_there_are_not() -> None:
    """A logit soft cap, the softcapped attention chain's static ``CAP`` and the scalar ``cap`` its rules and tests pass,
    in the adapter's files of it; Pearl's spellings there still count, and so does every "cap" of the same text in a
    file not listed."""
    chain = "v1 = evaluate(bind(dense.AttentionHeadSoftcap, T=T, D=D, BN=BN, CAP=CAP), *_args(q, k, v))\n"
    pearls = "# Pearl's: capOK, Device.cap, an over-cap tile, the cap check\n"
    pearl = [(2, "capOK"), (2, "Device.cap"), (2, "over-cap"), (2, "cap check")]
    for path in SOFTCAP_CAPS:
        assert Path(ROOT, path).is_file(), f"{path} is gone: move its line of SOFTCAP_CAPS with it"
        assert said(chain + pearls, path) == pearl
    assert said(chain + pearls, "tests/adapters/vllm/tests/program/test_fa2.py") == [(1, "CAP"), (1, "CAP"), *pearl]


def test_a_soft_cap_in_prose_is_another_sense_and_pearl_s_cap_beside_it_is_not() -> None:
    """"soft cap", a logit's or a resource limit's, in prose anywhere; a "cap" beside it still counts."""
    theirs = ("ALiBi, a sliding window, a logit soft cap, sinks, non-decoder attention, a quantized KV cache\n"
              "# address-space cap their runtime can't create its threads, so a wrapper lifts the soft cap\n")
    assert said(theirs, "adapters/vllm/verity_vllm/engine/int8_attention_vllm.py") == [(2, "cap")]
    assert said("a soft-cap, soft caps; a hard cap\n", "benchmarks/pouw/h1_prove/drawn.sh") == [(1, "cap")]


def test_a_cap_in_bytes_or_words_is_another_sense_and_a_cap_beside_it_is_not() -> None:
    """A cap its name counts in bytes, MiB, GiB or words (a memory or buffer cap), as an identifier anywhere; the bare
    "cap" beside it, and a name that doesn't say its unit, still count."""
    path = "adapters/vllm/verity_vllm/commit/committer/native_hidden.py"
    theirs = ("tap_cap_bytes = 64 << 20\nRC.snapshot_cap_bytes(wl), SNAPSHOT_CAP_FLOOR, host_memory_cap_source\n"
              "plan_fa2_tap_groups(b, words_per_seq, cap_words); replay_bundle_cap_gb\n")
    assert said(theirs, path) == []
    assert said("cap_mb = self.DEFAULT_TAP_CAP_MB if cap is None else cap\n", path) == [(1, "cap"), (1, "cap")]
    assert said("tap_cap, cap_size, CAP_KB\n", path) == [(1, "cap"), (1, "cap"), (1, "CAP")]


def test_a_compute_capability_is_another_sense_and_a_cap_beside_it_is_not() -> None:
    """A CUDA compute capability as the tests name a manifest's (``cap80``, ``cap90_fa2``); the parameter ``cap`` beside
    it, and a number no capability has, still count."""
    path = "tests/adapters/vllm/tests/program/test_applicability.py"
    line = '@pytest.mark.parametrize("name,cap,sms", [("cap80_108", [8, 0], 108), ("sms132", [8, 9], 132)])\n'
    assert said(line, path) == [(1, "cap")]
    assert said("cap90_fa2, cap100, cap120_site; cap3, cap1000\n", path) == [(1, "cap"), (1, "cap")]


def test_a_launch_s_rows_cap_is_another_sense_and_a_cap_beside_it_is_not() -> None:
    """The served-commit profile's cap on the CTAs one launch hashes, by its names; the arm's flag ``-cap`` beside it,
    and a row cap of another name, still count."""
    path = "benchmarks/served_commit/README.md"
    line = ("staging ring at 32 windows of 256 or 64 MiB; `-cap` arms hash in launches of at most `PROF_ROWS_CAP` CTAs\n"
            "rows_cap_blocks, rows_cap_free, SERVED_COMMIT_ROWS_RING8_CAP, RowsLayerCapArm\n")
    assert said(line, path) == [(1, "cap")]
    assert said("each admitted rows_cap, ROW_CAP\n", path) == [(1, "cap"), (1, "CAP")]


def test_fa3_s_split_cap_is_another_sense_and_a_cap_beside_it_is_not() -> None:
    """FA3's cap on its number of splits, by its names; a "cap" beside it still counts."""
    path = "tests/adapters/vllm/tests/check/test_served_check.py"
    line = "kw = dict(capture_sizes=[1, 2, 4, 8], max_num_seqs=8, split_cap=SV.SPLIT_CAP, device=H200)\n"
    assert said(line, path) == []
    assert said(line + "# the splits' cap, splits_cap\n", path) == [(2, "cap"), (2, "cap")]


def test_pearl_s_debit_cap_still_counts_against_the_real_table() -> None:
    """Pearl's cap, here a row's debit cap in a real line, is counted by the real table wherever no sense above holds."""
    line = ("verdict rejects a run on a sampled row over its debit cap (over-cap rule `reject`, which "
            "`served_passes.py`\n")
    assert [w.split("'")[1] for _, w in sites(line, "benchmarks/one_stage/README.md")] == ["debit", "cap", "over-cap"]


def test_a_word_of_two_kept_senses_the_lint_tells_apart_still_counts_in_its_retired_one() -> None:
    """A word the narrower senses list is enforced only in the phrases that retire it, but one in ``SENSED`` stays
    enforced alone: "cap" keeps SPOT's served cap and the certifier's response cap, by their names, and Pearl's counts."""
    two = "\n## Narrower senses\n\n- Two words still carry two senses in the code:\n  - **Cap**: SPOT's, the certifier's.\n"
    assert "cap" in {t.old for t in terms_of(SENSES_EXAMPLE + two)}
    assert "draw" not in {t.old for t in terms_of(SENSES_EXAMPLE + two.replace("Cap", "Draw"))}
    assert "cap" in {t.old for t in terms()[0]}, "GLOSSARY.md's two-senses list no longer stops Pearl's cap from counting"
    kept = "a served cap S charges each replay block at most F S; `NetworkAttesterParams.max_tokens_cap`\n"
    assert said(kept, "verity/protocols/proofs/spot/prover/plancheck.py") == []
    assert said(kept + "Pearl's cap, a tile over the cap\n", "verity/protocols/proofs/spot/prover/plancheck.py") == [
        (2, "cap"), (2, "cap")]


def test_a_cap_of_flock_s_claims_is_another_sense_and_pearl_s_beside_it_is_not() -> None:
    """The bound on an extractor's reruns and a commitment's Merkle cap: any "cap" where Lean states Flock's claims and
    proves its knowledge soundness and sampling over it, and anywhere in Flock's proofs. Pearl's spellings count there
    too, against the real table as well, and so does every "cap" beside those directories and in PoUW's proofs."""
    theirs = "the cap: `D.cap`, `Numbers.cap`, cap₀, `Model.Cap D`, `IsCapTauZ`, recursiveCaps, capOfRoot, cap_pos; its caps"
    pearls = "\nPearl's: capOK, Device.cap, CAP, an over-cap tile, the cap check"
    pearl = [(2, "capOK"), (2, "Device.cap"), (2, "CAP"), (2, "over-cap"), (2, "cap check")]
    for path in ("security/Definitions/Claims/SamplingTable.lean", "security/Proofs/Flock/Claims/KS/KSTable.lean",
                 "security/Proofs/Flock/Claims/Numbers.lean", "security/Proofs/Flock/Claims/NumbersDeploy.lean",
                 "security/Proofs/Flock/SamplingGeneric/KSTable.lean", "security/Proofs/Flock/Claims/Aggregate.lean",
                 "security/Proofs/Flock/Masked.lean"):
        assert said(theirs + pearls, path) == pearl, path
        assert [w.split("'")[1] for _, w in sites(pearls, path)] == [o for _, o in pearl], path
    counted = [(1, o) for o in ("cap", "cap", "cap", "cap", "Cap", "Cap", "Caps", "cap", "cap", "caps")]
    for path in ("security/Definitions/Flock/Compiled.lean", "security/Definitions/ClaimsOf.lean",
                 "security/Proofs/Pouw/Claims/KS/Bound.lean", "verity/protocols/flock/verifier/Flock/Merkle.lean"):
        assert said(theirs + pearls, path) == counted + pearl, path


def test_two_terms_on_one_span_name_the_same_one_in_every_process() -> None:
    """The ratchet gate compares a base's sites with a head's as strings, each found in its own process, and a set of
    terms iterates in an order that differs between processes."""
    alive, said = [], set()
    for i in range(64):
        pair = [Term("Dampener*", "the slowdown", code=True), Term("*RemotePeriod", "monitoring period", code=True)]
        alive.append(pair)
        by: dict[str, list[Term]] = {}
        for t in pair[::-1] if i % 2 else pair:
            for a in t.anchors:
                by.setdefault(a, []).append(t)
        said |= set(sites_of("the DampenerRemotePeriod\n", by))
    assert said == {(1, "'DampenerRemotePeriod' is retired; say monitoring period (GLOSSARY.md)")}, sorted(said)


def test_two_terms_on_one_site_say_the_same_whatever_order_the_set_holds_them_in() -> None:
    class Placed(Term):
        """A term whose place in a small set is chosen: such a set gives its members in the order of their hashes."""

        def __init__(self, place: int, old: str, new: str):
            super().__init__(old, new, code=True)
            self.place = place

        def __hash__(self) -> int:
            return self.place

    def said(text: str, *rows: tuple[int, str, str]) -> list[tuple[int, str]]:
        by: dict[str, list[Term]] = {}
        for t in (Placed(*row) for row in rows):
            for a in t.anchors:
                by.setdefault(a, []).append(t)
        return sites_of(text, by)

    longer = [(1, "'OldName' is retired; say the longer row's (GLOSSARY.md)")]  # second by spelling: the length decides
    assert said("an OldName", (1, "*Name", "the shorter row's"), (2, "OldNa*", "the longer row's")) == longer
    assert said("an OldName", (2, "*Name", "the shorter row's"), (1, "OldNa*", "the longer row's")) == longer
    first = [(1, "'OldName' is retired; say the first by spelling (GLOSSARY.md)")]  # of one length: the spelling decides
    assert said("an OldName", (1, "OldN*", "the second by spelling"), (2, "*Name", "the first by spelling")) == first
    assert said("an OldName", (2, "OldN*", "the second by spelling"), (1, "*Name", "the first by spelling")) == first


PERIOD_EXAMPLE = """# Glossary
## Older names

| Old name, where it stays | Term |
|---|---|
| "horizon"; the Lean's `*Horizon*` and `*Period*` names (`ComposePeriod`, `CircuitWorkPeriod`), until renamed | monitoring period (`horizon` is its pooled reading) |
| `Dampener*`, `*RemotePeriod`, until renamed | the slowdown |
| the Lean's `*Epoch*` names, until renamed | (`MonitoringEpoch` once renamed) |
"""


def period_sites(text: str) -> list[str]:
    by: dict[str, list[Term]] = {}
    for t in terms_of(PERIOD_EXAMPLE):
        for a in t.anchors:
            by.setdefault(a, []).append(t)
    return [w.split(" is retired")[0].strip("'\"") for _, w in sites_of(text, by)]


def test_a_name_that_already_says_the_row_s_term_is_not_a_site_of_its_glob() -> None:
    """A ``*`` name's site in an identifier that says the row's term (its words before the parenthesis, in CamelCase,
    lowerCamel or snake_case) isn't one; the old names, a name in a dotted path beside one that says the term, another
    row's glob, and a glob whose row gives no plain term still count.  The identifier is read whole, so one holding an old
    name beside the term passes."""
    got = {t.old: t for t in terms_of(PERIOD_EXAMPLE)}
    assert got["*Period*"].says.pattern == "[Mm]onitoringPeriod|(?i:monitoring_period)"
    assert not got["*Period*"].says.search("MONITORINGPeriod monitoringperiod")
    assert got["*Epoch*"].says is None and got["ComposePeriod"].says is None and got["horizon"].says is None
    assert period_sites("MonitoringPeriodChecked, MonitoringPeriod, monitoring_period_checked, MONITORING_PERIOD") == []
    assert period_sites("monitoringPeriodG, monitoringPeriod_every, liveAt_monitoringPeriod") == []
    assert period_sites("ledgerPeriod_le_monitoringPeriod") == []
    assert period_sites("ComposePeriod, CircuitWorkPeriod, DampenerRemotePeriod, MONITORINGPeriod, "
                        "ResourceAccounting.MonitoringPeriod.LedgerPeriod, composePeriod, ledgerPeriod") == [
        "ComposePeriod", "CircuitWorkPeriod", "DampenerRemotePeriod", "MONITORINGPeriod", "LedgerPeriod",
        "composePeriod", "ledgerPeriod"]
    assert period_sites("ComposeHorizon, PooledHorizonBound, the horizon") == [
        "ComposeHorizon", "PooledHorizonBound", "horizon"]
    assert period_sites("DampenerMonitoringPeriod, MonitoringPeriodRemotePeriod") == [
        "DampenerMonitoringPeriod", "MonitoringPeriodRemotePeriod"]
    assert period_sites("ComposeEpoch, MonitoringEpoch") == ["ComposeEpoch", "MonitoringEpoch"]


def test_pous_period_names_keep_the_word_and_the_names_beside_them_do_not() -> None:
    """PoUS's ``Pous.Period``, as a namespace or a module path, and its guarantees' names (``OWN_NAMES``) aren't sites
    of ``*Period*``; a bare ``Period``, another area's (``Core.Period``, Flock's ``Periods``), a longer name holding one
    of PoUS's (``SpaceOfPeriodHeld``), a name only like one of them (``B2PeriodX``, ``PeriodHeldX``) and attest's
    ``PeriodSigned`` outside its files still count."""
    assert Path(ROOT, "security/Properties/Pous/Period.lean").is_file(), \
        "Pous.Period moved: move OWN_NAMES' pattern with it"
    pous = ("Pous.Period.PeriodHeld, Properties/Pous/Period.lean, PeriodHeldAt, PeriodHeldCoinsAt, PeriodHeldSha512, "
            "PeriodHeldDeployed, PeriodHeldDemo, PeriodHeldHonest, PeriodHeldNontrivial, PeriodDeployed, PeriodGame, "
            "PeriodAt, PeriodCapacity, PeriodWitnesses")
    assert period_sites(pous) == []
    beside = "Period, Core.Period, Audit.Periods, SpaceOfPeriodHeld, B2PeriodX, PeriodSigned, PeriodHeldX"
    assert period_sites(beside) == ["Period", "Period", "Periods", "SpaceOfPeriodHeld", "B2PeriodX", "PeriodSigned",
                                    "PeriodHeldX"]


def test_the_responder_s_and_a_party_s_periods_keep_the_word_and_attest_s_names_only_in_its_files() -> None:
    """PoUS's responder's ``B2Period`` names and a party's ``Attest.Period`` aren't sites of ``*Period*`` anywhere;
    ``PeriodSigned``, ``PeriodSound`` and ``PeriodRun`` aren't in attest's two guarantee files, and are elsewhere, where
    ``PeriodRun`` is compose's and ``PeriodSound`` PoUW's."""
    text = "B2Period, B2PeriodAudit, B2PeriodP2Ring, Attest.Period, attest.Period, PeriodSigned, PeriodSound, PeriodRun"
    files = OWN_NAMES["*Period*"][3][2]
    for f in files:
        assert Path(ROOT, f).is_file(), f"{f} is gone: move its line of OWN_NAMES with it"
        assert [w.split("'")[1] for _, w in sites(text, f)] == []
    assert [w.split("'")[1] for _, w in sites(text, "security/Definitions/Compose/JointPeriod.lean")] == [
        "PeriodSigned", "PeriodSound", "PeriodRun"]


def test_the_adapter_s_row_of_record_and_h1_s_period_auditor_keep_their_words_only_in_their_files() -> None:
    """vLLM's "row of record" and H1's ``_PeriodAuditor`` aren't sites in the adapter's files that name them, and are
    elsewhere."""
    text = "the row of record, row_of_record and _PeriodAuditor"
    for term, (_, _, files) in (("row of record", OWN_NAMES["row of record"][0]), ("*Period*", OWN_NAMES["*Period*"][4])):
        for f in files:
            assert Path(ROOT, f).is_file(), f"{f} is gone: move its line of OWN_NAMES with it"
            assert [w.split("'")[1] for _, w in sites(text, f) if (term == "*Period*") == ("Period" in w)] == []
    assert [w.split("'")[1] for _, w in sites(text, "verity/a.py")] == ["row of record", "row_of_record", "_PeriodAuditor"]


RECORDED_KEYS = ('ZK_SERVER_MODE, RECORDED_MODE = "zk_server", "firewall"\n'
                 'LOOP, RECORDED_LOOP = "lean_session_loop", "flock_firewall"\n'
                 'LOOP_PROCESSES = ("flock-zk-server", "flock-firewall")\n'
                 "#: a recorded run's ZK server refused as ``FIREWALL-REFUSED``\n"
                 'STOPS = ("ZK-SERVER-REFUSED", "FIREWALL-REFUSED", "PROVER-STOPPED")\n'
                 'RECORDED_BUCKETS = {"t.firewall_recheck": "t.zk_server_recheck"}\n')
ARM_KEYS = ('bound_arm, _control = resolve_arms("sampled-proofs:hm96,stock-fp8")\n'
            'assert build_bound(bound_arm) == ("sampled-proofs:hm96",)\n'
            'live = _pearl_c_gate(Run(exact_y, coin_source="challenger", session=session, pass_index=0))\n')
RECURSION = "benchmarks/recursion_e2e/"
HARNESS = "benchmarks/harness/"


def retired(text: str, path: str) -> list[tuple[int, str]]:
    return [(line, w.split(" is retired")[0].strip("'\"")) for line, w in sites(text, path)]


def test_a_held_key_in_a_file_its_entry_doesn_t_name_still_counts() -> None:
    """A key is held only in the files its entry names: in a file beside them, and in a file another key's entry names,
    every site of it counts."""
    recorded = [(1, "firewall"), (2, "firewall"), (3, "flock-firewall"), (4, "FIREWALL"), (5, "FIREWALL"), (6, "firewall")]
    assert retired(RECORDED_KEYS, RECURSION + "forge_after.py") == recorded
    assert retired(RECORDED_KEYS, HARNESS + "verity_bench/vllm/runner.py") == recorded
    assert retired(ARM_KEYS, HARNESS + "verity_bench/vllm/runner.py") == [(1, "sampled-proofs"), (2, "sampled-proofs")]
    assert retired(ARM_KEYS, HARNESS + "verity_bench/vllm/commit_arm.py") == [(3, "coin_source")]
    assert retired(ARM_KEYS, "benchmarks/one_stage/run.py") == [(1, "sampled-proofs"), (2, "sampled-proofs"),
                                                                (3, "coin_source")]


def test_the_word_in_prose_or_in_a_new_name_in_a_listed_file_still_counts() -> None:
    """In a file its entry names, a key is held only where it stands whole: the bare word, a plain word unquoted or quoted
    with more beside it, another spelling, and a longer name holding the key still count."""
    recursion = ('# the firewall refused it: mode = firewall, "the firewall", FIREWALL\n'
                 'firewall_mode, "firewall_v2", flock_firewall_loop, "FIREWALL-REFUSED-AGAIN", "t.firewall_recheck2"\n')
    assert retired(recursion, RECURSION + "rec_profile.py") == [
        (1, "firewall"), (1, "firewall"), (1, "firewall"), (1, "FIREWALL"),
        (2, "firewall"), (2, "firewall"), (2, "firewall"), (2, "FIREWALL"), (2, "firewall")]
    harness = ('# the coin source, coin_source_v2, c["coin_sources"]; the sampled proofs, "sampled-proofs", sampled-proofs:\n')
    assert retired(harness, HARNESS + "tests/test_bench.py") == [
        (1, "coin source"), (1, "coin_source"), (1, "coin_sources"), (1, "sampled proofs"), (1, "sampled-proofs"),
        (1, "sampled-proofs")]


def test_a_held_key_in_a_listed_file_is_not_a_site() -> None:
    """Each key where it stands whole in a file its entry names, quoted or as the identifier it is, in code, a comment or
    Markdown; a site beside it on the same line still counts."""
    assert retired(RECORDED_KEYS, RECURSION + "rec_profile.py") == []
    readme = ("  `FIREWALL-REFUSED`) or Lean refused it\n"
              "  cost is read from `flock-zk-server`'s (`flock-firewall`'s in a recorded run) CPU seconds\n")
    assert retired(readme, RECURSION + "README.md") == []
    stop = '(d / "prove.err").write_text("FIREWALL-REFUSED R7: the witness does not open the registered row; no proof\\n")\n'
    assert retired(stop + 'assert "FIREWALL-REFUSED" in outer["note"]\n', RECURSION + "tests/test_recursion_e2e.py") == []
    assert retired(ARM_KEYS, HARNESS + "tests/test_bench.py") == []
    coins = ('check.coins = ({"source": c["coin_source"]} if c.get("coin_source", "in-process") != "in-process"\n'
             "               else {}  # the firewall's coins\n")
    assert retired(coins, HARNESS + "verity_bench/vllm/runner.py") == [(2, "firewall")]
    assert retired("A protocol arm (`pouw:SCHEME`, SPOT's key, `sampled-proofs:SCHEME`) runs; sampled-proofs alone\n",
                   HARNESS + "README.md") == [(1, "sampled-proofs")]


def test_a_held_key_that_no_longer_stands_in_a_file_its_entry_names_is_refused_as_stale() -> None:
    """Every entry of ``HELD_KEYS`` holds its key in each of its files; one that names a file where the key stands only in
    prose or not at all, a directory, or no reason is refused, naming the fix."""
    assert not stale(HELD_KEYS), "\n".join(stale(HELD_KEYS))
    reason = "reads a stored recursion run, benchmarks'"
    gone = stale({"flock_firewall": ((HARNESS + "verity_bench/vllm/runner.py",), reason)})
    assert len(gone) == 1 and "no longer stands held in" in gone[0], gone
    assert "never a directory" in stale({"flock_firewall": ((RECURSION,), reason)})[0]
    assert "the reason" in stale({"flock_firewall": ((RECURSION + "rec_profile.py",), "")})[0]


def test_it_reads_what_lane_text_reads_and_skips_what_it_says() -> None:
    assert holds("verity/a.py") and holds("security/A.lean") and holds("README.md") and holds("a/b.toml")
    assert not holds("x.json") and not holds(GLOSSARY) and not holds("tools/move/x_map.toml")
    assert not holds("a/tests/fixtures/b.py") and not holds("a/CHANGELOG.md") and not holds(f"tests/lint/test_{NAME}.py")
    assert sites("C-Flock", "x.json") == []


def test_the_list_is_well_formed_and_an_entry_moves_with_its_file() -> None:
    # The check that reads the tree's list is a bookkeeping test, whose failed assertion only warns: a list that
    # doesn't parse would switch the scan off with nothing red, so the tree's list is parsed here, where it fails.
    allowed()
    assert parse({"entries": [{"file": "a.py", "count": 2}]}) == Counter({"a.py": 2})
    for bad in ({"file": "a.py", "count": 0}, {"file": "a.py", "count": True}, {"file": "a.py", "count": 1, "rule": "x"}):
        try:
            parse({"entries": [bad]})
        except AssertionError as e:
            assert str(e).startswith(f"{LIST.name}: "), e
        else:
            raise AssertionError(f"{bad} parsed")
    twice = {"entries": [{"file": "a.py", "count": 1}, {"file": "a.py", "count": 1}]}
    try:
        parse(twice)
    except AssertionError as e:
        assert "duplicate entry" in str(e), e
    else:
        raise AssertionError("a file listed twice parsed")
    assert moved(Counter({"a/x.py": 2, "a/y.md": 1, "b.py": 1}), "a", "c") == Counter({"c/x.py": 2, "c/y.md": 1, "b.py": 1})


def moved(counts: Counter, old: str, new: str) -> Counter:
    """``counts`` with the file ``old``'s, or the directory ``old``'s files', under ``new``: a count already there adds."""
    old, new = old.rstrip("/"), new.rstrip("/")
    out: Counter = Counter()
    for f, n in counts.items():
        out[new + f[len(old):] if f == old or f.startswith(f"{old}/") else f] += n
    return out


def dump(counts: Counter) -> None:
    lines = [f"    {json.dumps({'file': f, 'count': n}, ensure_ascii=False)}" for f, n in sorted(counts.items()) if n > 0]
    LIST.write_text(f'{{\n  "rule": "retired terms (tests/lint/test_{NAME}.py)",\n  "entries": [\n'
                    + ",\n".join(lines) + "\n  ]\n}\n", encoding="utf-8")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--move"] and len(sys.argv) == 4:
        have = allowed()
        after = moved(have, *sys.argv[2:])
        dump(after)
        print(f"{sum(1 for f in have if f not in after)} entries moved from {sys.argv[2]} to {sys.argv[3]}")
    elif sys.argv[1:] == ["--prune"]:
        dump(allowed() & found())
    elif sys.argv[1:] == ["--record"]:
        dump(found())
    else:
        sys.exit(__doc__.rsplit("\n\n", 1)[-1])
