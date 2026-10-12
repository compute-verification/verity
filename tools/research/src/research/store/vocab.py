"""The label vocabulary (README §8.1): every label key in use, grouped by who asserts it, with the value form it takes.

The store never validates a label beyond its envelope (§8: any key, any canonicalisable value); this module is the *shared spelling*
of the keys and of the closed value sets, so that a producer, a verifier and a table read the same word.  `research data label`
writes the canonical spelling of an old key or value and REFUSES what the vocabulary cannot place -- an unknown key (with the
nearest known one), a measurement (`MEASUREMENT_KEYS`: it belongs in the result or the artifact's meta), a ref written as a label,
a value outside a declared enum, a word where the form is a bool / int / number.  `--off-vocab` writes it anyway, for a concept the
vocabulary has not learnt yet: bring the word here next.  Readers stay lenient (old spellings are read as the canonical key; the
store keeps whatever older writers wrote).  `research data vocab` prints the table.  Consumers import the constants
(`vocab.PROOF_CLASS`, never the string).

Four groups, by asserter:

    result        the producing lane, about its own result / ledger row (`ledger_to_store`, or `research data label` by hand)
    verification  an independent verifier, never the producer (a table counts it only when `by` is not a producer)
    evidence      a probe or input-set lane, about a capture / dataset (`tc-evidence/v1`, `dataset-snapshot/v1`, `fixture/v1`)
    approach      the owning lane or its coordinator, about an `approach/v1` record (`research notes claim | approach`); `approach`
                  itself goes on a run or artifact: the approach it belongs to

Value forms: `enum` (one of `values`), `bool` (`true` | `false`), `int`, `number`, `text`, `list` (of strings).  Enum and bool
values are compared as the catalog stores them (`index.scalar_text`: strings raw, everything else JSON), so `true` and `"true"`
are the same word and `"True"` / `"yes"` are not.  Only enums and bools are checked; the other forms are documentation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .index import scalar_text

GROUP_RESULT, GROUP_VERIFICATION, GROUP_EVIDENCE, GROUP_APPROACH = "result", "verification", "evidence", "approach"
GROUPS = (GROUP_RESULT, GROUP_VERIFICATION, GROUP_EVIDENCE, GROUP_APPROACH)
CURRENT, DEPRECATED, IGNORED = "current", "deprecated", "ignored"
ENUM, BOOL, INT, NUMBER, TEXT, LIST = "enum", "bool", "int", "number", "text", "list"
TRUE_FALSE = ("true", "false")

# ---- result labels: the producing lane about its own result ---------------------------------------------------------------------------

CANDIDATE = "candidate"
PROOF_CLASS = "proof_class"
SOUNDNESS = "soundness"
K = "K"
B = "B"
HARDWARE = "hardware"
AUTHENTICATION = "authentication"
CAMPAIGN = "campaign"
RELATION = "relation"
MODE = "mode"
ZK = "zk"
OVERHEAD = "overhead"
SECONDS_PER_VU = "seconds_per_vu"
TRACK = "track"
SCOPE = "scope"
LABEL = "label"
NOTE = "note"
ASSUMPTIONS = "assumptions"
OMITTED = "omitted"
DOMAIN = "domain"
DOMAINS = ("total", "finite-only")
BREAKTHROUGH = "breakthrough"
LANE = "lane"
QUESTION = "question"
CITED_BY = "cited-by"
ROW = "row"
SWEEP = "sweep"
ARM = "arm"
STAGE = "stage"
SOURCE = "source"
SUPERSEDED_BY = "superseded_by"
ABANDONED = "abandoned"
REPLAY_SET = "replay_set"
REPLAY_EXPECT = "replay_expect"
INTERACTION_ATTEMPT_OF = "interaction_attempt_of"
HASH = "hash"
SHARING = "sharing"
FOLD = "fold"
PIPELINE = "pipeline"
KERNEL_OUTCOME = "kernel_outcome"
KERNEL_OUTCOMES = ("screened", "frontier", "kept", "dropped", "voided", "superseded")
KERNEL_REASON = "kernel_reason"

# ---- verification labels: an independent verifier, never the producer ----------------------------------------------------------------

VERIFIED = "verified"
VERIFIER = "verifier"
VERIFIER_SECONDS = "verifier_seconds"
INDEPENDENTLY_VERIFIED = "independently_verified"     # the back-fill's spelling of verified=accepted|rejected: accepted, deprecated
VERIFIED_BY = "verified_by"                           # a ledger row's free-text field: carries no verdict, ignored by every table
SAME_DEVICE = "same_device"
BELOW_BAR = "below_bar"                               # verified, but its whole-proof bound misses the tables' 2^-128 bar
OUTCOME = "outcome"
COMMIT_MAP = "commit_map"                             # on a commit-map/v1 artifact: whether the history rewrite it records applies
CUSTODY = "custody"                                   # on a run id: whether `research data custody --triage` needs more of it
CUSTODY_VALUES = ("waived", "never-ran", "needed")
SOURCE_LOST = "source-lost"                             # on a run id: its exact source (commit + dirty bytes) is gone

# ---- evidence labels: probe captures and input sets --------------------------------------------------------------------------------

INSTRUCTION = "instruction"
MODEL = "model"
MODEL_AGREES = "model_agrees"
MODEL_STATUS = "model_status"
MODEL_REJECTED = "model_rejected"
EVIDENCE = "evidence"
FAMILY = "family"
INSTANCES_DATASET = "instances_dataset"
INSTANCES_TIER = "instances_tier"
MISMATCHES = "mismatches"
TARGET = "target"
DEVICE = "device"
INSTANCES_MANIFEST_SHA256 = "instances_manifest_sha256"
COMPONENT = "component"
JOBS = "jobs"
DTYPE = "dtype"

# ---- approach labels: the registry (`research.approaches`), by the owning lane or its coordinator ------------------------------------

APPROACH = "approach"
TITLE = "title"
HYPOTHESIS = "hypothesis"
APPROACH_TYPE = "approach_type"
ATTACKS = "attacks"
OWNER = "owner"
APPROACH_STATUS = "approach_status"
REASON = "reason"
CITES = "cites"
UNCITES = "uncites"

# ---- closed value sets -------------------------------------------------------------------------------------------------------------------

VERIFIED_ACCEPTED, VERIFIED_REJECTED, VERIFIED_NOT_TRANSFERABLE, VERIFIED_NO_DUMPS = "accepted", "rejected", "not-transferable", "no-dumps"
VERIFIED_VALUES = (VERIFIED_ACCEPTED, VERIFIED_REJECTED, VERIFIED_NOT_TRANSFERABLE, VERIFIED_NO_DUMPS)
COMMIT_MAP_VALUES = ("accepted", "retracted")
#: Mirrors `PROOF_CLASSES` in `benchmarks/numerical/python/verity_numerical/bench/contract.py` + the `COMPONENT` that
#: `bench/ledger.py` adds (a bucket measured alone is no proof class).  The store imports neither package (the `research` rule), so
#: the spelling is copied here; `benchmarks/numerical/tests/bench/test_tables.py` checks the copy against both.
PROOF_CLASSES = ("NO_PROOF", "ARITHMETIC_DIAGNOSTIC", "NON_ZK_PROOF_DIAGNOSTIC", "NON_ZK_PROOF", "COMPLETE_HVZK_BACKEND", "COMPLETE_ZK_BACKEND",
                 "COMPONENT")
AUTHENTICATION_VALUES = ("included", "included-hash", "included-hash-shared", "excluded")
MODES = ("interactive", "fiat-shamir")
SCOPES = ("unit", "vu", "component")
TRACKS = ("A", "B", "C", "shared", "scout", "baseline")
#: Mirrors the Glossary's backend families (`README.md`, "Backend family"), which `bench/views.py` spells as `FAMILY_NAMES`; the
#: backends' tests check the copy.  A `candidate` label written today names one of these.
BACKEND_FAMILIES = ("A-GKR", "B-Ligero", "C-Flock", "D-SP1")
#: The backend families, then the values `ledger_to_store.candidate` derives from a frozen ledger lane, which recorded labels carry
#: and so stay valid: `B-AIR`, `C` (the VOLE candidate in `archive/vole`, not C-Flock), `SP1` (D-SP1 as the frozen ledger and
#: tables spell it) and `shared`.  `A` was the ledger import's spelling until 2026-09-23 (`VALUE_ALIASES`): read as `A-GKR`, no longer written.
CANDIDATES = (*BACKEND_FAMILIES, "B-AIR", "C", "SP1", "shared")
APPROACH_STATUSES = ("live", "parked", "killed", "superseded", "merged")
APPROACH_TYPES = ("scheme", "route", "attack")

# ---- old spellings (2026-09-23 hygiene): accepted on read, never written anew ----------------------------------------------------------------

#: label key written -> the key it means.  Writers (`research data label`, `ledger_to_store`) store the canonical key only; readers
#: (`bench.tables`, `bench.store_tables`, `vocab-check`) map an old key to its canonical one with `canonical_key`.
KEY_ALIASES: dict[str, str] = {"notes": NOTE, "verify_note": NOTE, "verifier_note": NOTE, "batch": B, "checker": VERIFIER,
                               "axis": SWEEP, "milestone": ARM, "tree": SOURCE, "release": SOURCE}
#: (key, value written) -> the value it means.
VALUE_ALIASES: dict[tuple[str, str], str] = {(CANDIDATE, "A"): "A-GKR"}
#: Keys whose value names another store object: that is a `ref` (in the manifest at `put`, `--ref name=art:...`), not a label.
#: `research data label` refuses them (without `--off-vocab`); `vocab-check` lists the ones already written.
REF_KEYS = ("path", "result_artifact", "constructed_from_attempt")
#: Measurements and run parameters lanes have written as labels (2026-09-22/23).  The number is already in the result (or goes in
#: the artifact's meta at `put`); a label is what a table selects on or a judgement about the result, so these are refused.
MEASUREMENT_KEYS = ("l", "n", "subbatches", "per_proof_vus", "R_proved", "threads", "window", "rtt_ms_path", "native_peak", "accumulate",
                    "replicate", "encoder")
#: A note in the notes repo, as a label's ref names it: `note:<lane>/<file name without .md>` (`note:kb/<name>` for kb/) or the notes
#: README's `note:<campaign>/<lane>/<name>`, either the same whichever month folder the file has moved to.  Not a bare name (a handoff
#: sent to several lanes has the same name in each), and not a path (`.md`, a `YYYY-MM` or `DD` folder).
NOTE_REF_RE = re.compile(r"^note:(?:(?![\d-]+/)[A-Za-z0-9][\w.-]*/){1,2}(?![\w.-]*\.md$)[\w.-]+$")
#: Keys whose label is only read with its ref: the form that ref must take, and what it names.
REF_FORMS: dict[str, tuple[re.Pattern[str], str]] = {
    ABANDONED: (NOTE_REF_RE, "the note that says why (note:<lane>/<file name without .md>, or note:<campaign>/<lane>/<name>)"),
    SOURCE_LOST: (NOTE_REF_RE, "the note that says how it was lost (note:<lane>/<file name without .md>, or note:<campaign>/<lane>/<name>)")}
#: Label keys that name evidence / findings written by code today (trust dossiers, the ledger's red-team assertions).
FINDING = "finding"
LEVEL = "level"
ANCHOR = "anchor"
#: A reviewer's grant on a pull request at one head (`pr:<n>@<sha>`), which the merge queue's admission rules require.
GRANT = "grant"
GRANT_ROLES = ("red-team", "statement-reviewer", "vllm-coordinator", "review")
#: The owner's word that a pull request is ready at one head (`research queue ready`), which the queue's admission of a single PR
#: asks for; a train member needs none, since the train's own check is the gate (Daniel, Q14, 6 Oct).
READY = "ready"
#: A `check.py --update` report (its audit.json) on a pull request at one head, which stands in for a statement reviewer's grant
#: where it printed the head's whole lean-audit.json change as the same statements (`research.queue.reviewed`).
LEAN_REVIEW = "lean_review"
#: On a run a landing's `Check:` or `Checked:` trailer names: main's commit that landed what it checked (`research merge`).
LANDED = "landed"


def canonical_key(name: str) -> str:
    return KEY_ALIASES.get(name, name)


def canonical_value(name: str, value: Any) -> Any:
    """The value an old spelling means (`candidate=A` -> `A-GKR`); anything else unchanged."""
    if isinstance(value, str):
        return VALUE_ALIASES.get((canonical_key(name), value), value)
    return value


def coerce(name: str, value: Any) -> Any:
    """The value in the form the vocabulary declares for `name`, when a string can be read that way without ambiguity: `"true"` /
    `"false"` -> bool for a `bool` key, `"4096"` -> int for an `int` key, `"1.5"` -> float for a `number` key.  Everything else is
    returned as given (an unknown key, a free-text form, a string that does not parse).  Writers call this so that `zk`, `model_agrees`,
    `independently_verified` are real booleans in the file, not the words."""
    k = VOCAB.get(canonical_key(name))
    if k is None or not isinstance(value, str):
        return value
    s = value.strip()
    if k.form == BOOL and s.lower() in TRUE_FALSE:
        return s.lower() == "true"
    if k.form == INT and _INT_RE.match(s):
        return int(s)
    if k.form == NUMBER and _NUM_RE.match(s):
        return int(s) if _INT_RE.match(s) else float(s)
    return value


_INT_RE = re.compile(r"^[+-]?\d+$")
_NUM_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


@dataclass(frozen=True)
class Key:
    """One vocabulary entry.  `values` is the closed set of an `enum` (and `true`/`false` for a `bool`), in `scalar_text` form."""

    name: str
    group: str
    form: str
    doc: str
    values: tuple[str, ...] = ()
    status: str = CURRENT
    pattern: str = ""

    def check(self, value: Any) -> str | None:
        """Why `value` is outside this key's declared set or `pattern`, as one line, or None (other free forms are never checked)."""
        if self.values and scalar_text(value) not in self.values:
            return f"{self.name}={scalar_text(value)!r} is not one of {', '.join(self.values)}"
        if self.pattern and not re.fullmatch(self.pattern, scalar_text(value)):
            return f"{self.name}={scalar_text(value)!r} does not match {self.pattern}"
        return None


def _k(name: str, group: str, form: str, doc: str, values: tuple[str, ...] = (), status: str = CURRENT, pattern: str = "") -> Key:
    return Key(name, group, form, doc, TRUE_FALSE if form == BOOL else values, status, pattern)


VOCAB: dict[str, Key] = {k.name: k for k in (
    # result
    _k(CANDIDATE, GROUP_RESULT, ENUM, "the backend family the result belongs to (or the older candidate a ledger row names)", CANDIDATES),
    _k(PROOF_CLASS, GROUP_RESULT, ENUM, "the class claimed for the result (a red-team verdict is another proof_class label, by red-team-*)", PROOF_CLASSES),
    _k(SOUNDNESS, GROUP_RESULT, TEXT, "the soundness target or achieved bound as stated, e.g. 2^-128 (qualifiers allowed)"),
    _k(K, GROUP_RESULT, INT, "the relation's K (1536 for the frozen target)"),
    _k(B, GROUP_RESULT, INT, "the batch size the number was measured at"),
    _k(HARDWARE, GROUP_RESULT, TEXT, "the device(s) the prover (and verifier) ran on, as the lane described them"),
    _k(AUTHENTICATION, GROUP_RESULT, ENUM, "whether the boundary words were authenticated against the object's roots", AUTHENTICATION_VALUES),
    _k(CAMPAIGN, GROUP_RESULT, TEXT, "the campaign the result was produced in, e.g. r21-silicon"),
    _k(RELATION, GROUP_RESULT, TEXT, "the relation proved, e.g. fp8-ada-k1536, bf16-k1536-control"),
    _k(MODE, GROUP_RESULT, ENUM, "how the transcript was produced", MODES),
    _k(ZK, GROUP_RESULT, BOOL, "whether the run was zero-knowledge (a COMPLETE_*ZK_BACKEND run)"),
    _k(OVERHEAD, GROUP_RESULT, NUMBER, "R_native / R_proved, the ledger's overhead ratio"),
    _k(SECONDS_PER_VU, GROUP_RESULT, NUMBER, "prover wall seconds per VU"),
    _k(TRACK, GROUP_RESULT, ENUM, "the campaign track the lane worked", TRACKS),
    _k(SCOPE, GROUP_RESULT, ENUM, "what the number measures: a whole VU, a unit, or one component", SCOPES),
    _k(LABEL, GROUP_RESULT, TEXT, "the lane's one-line headline for the result"),
    _k(NOTE, GROUP_RESULT, TEXT, "the lane's free-text note"),
    _k(ASSUMPTIONS, GROUP_RESULT, LIST, "the cryptographic assumptions the number rests on"),
    _k(OMITTED, GROUP_RESULT, LIST, "what the number leaves out (ZK, authentication, key setup, ...)"),
    _k(DOMAIN, GROUP_RESULT, ENUM, "the operand domain the statement covers: total (every bit pattern, NaN and infinity included, as the "
                                   "hardware) or finite-only (non-finite operands asserted away; Daniel 2026-09-26: an explicit, named choice)", DOMAINS),
    _k(BREAKTHROUGH, GROUP_RESULT, BOOL, "the lane's own flag that the row moves the campaign"),
    _k(LANE, GROUP_RESULT, TEXT, "the lane that produced the result, when the artifact's meta does not carry it (tables read meta.lane first)"),
    _k(QUESTION, GROUP_RESULT, TEXT, "the research question the job answers, as its lane's research owner named it (Daniel, 30 Sep: every job names one)"),
    _k(CITED_BY, GROUP_RESULT, TEXT, "one place that cites the run or artifact as evidence (note:<id>, <repo>#<n>, a doc path): a lane marks what it must never lose"),
    _k(ROW, GROUP_RESULT, TEXT, "the table or sweep row the result fills, e.g. 'bf16-hopper int-nonZK 4096 VUs' or a fixture row id"),
    _k(SWEEP, GROUP_RESULT, TEXT, "the parameter a series of results varies (batch, subbatch, l, zk, ...); old spelling `axis`"),
    _k(ARM, GROUP_RESULT, TEXT, "this result's point or variant within its sweep or comparison (pipeline4-window4, 1M-vu); old spelling `milestone`"),
    _k(STAGE, GROUP_RESULT, TEXT, "the pipeline stage the result belongs to (build, match, commit)"),
    _k(SOURCE, GROUP_RESULT, TEXT, "branch@commit the result was produced from, for an artifact no Attempt records; old spellings `tree`, `release`"),
    _k(SUPERSEDED_BY, GROUP_RESULT, TEXT, "the run id or art: id of the result that replaces this one (this one stays in the store); on an "
       "approach record, the <campaign>/<slug> that replaces it"),
    _k(REPLAY_SET, GROUP_RESULT, TEXT, "the replay set a run or artifact belongs to, as AREA/TIER: TIER re-verify (its preserved records "
       "checked again at a new commit, bit for bit) or re-run (our code's timed result run again); one label per area that cites it "
       "(`research data select --label replay_set=network-certifier/re-verify`, `research replay`)", pattern=r"[a-z0-9][a-z0-9-]*/(re-verify|re-run)"),
    _k(REPLAY_EXPECT, GROUP_RESULT, TEXT, "the change a replay at current main should show instead of the recorded verdict, with --ref "
       "naming the ruling: `refused: FORM` (the verifier of record now refuses that form) or `blocked: GATE` (fail-closed refuses the "
       "session kind until GATE, e.g. V3)", pattern=r"(refused|blocked): \S.*"),
    _k(ABANDONED, GROUP_RESULT, TEXT, "the line of work this evidence belongs to was given up, in a few words; --ref names the note that says why "
                                      "(note:<lane>/<file name without .md>), so nobody builds on it unawares"),
    _k(INTERACTION_ATTEMPT_OF, GROUP_RESULT, TEXT, "on an interaction attempt that is not the cell's figure (refused, or run on an earlier pair): the art: id of "
       "the cell it belongs to; the cell lists every attempt with its deviation (IX1)"),
    _k(HASH, GROUP_RESULT, TEXT, "the operand commitment proven in-circuit: none, poseidon2-babybear-w24, or the leaf's name"),
    _k(SHARING, GROUP_RESULT, TEXT, "how operand rows are shared across VUs in-circuit: none, tile64x64"),
    _k(FOLD, GROUP_RESULT, INT, "columns folded per VU (the relation's -xN)"),
    _k(PIPELINE, GROUP_RESULT, INT, "prover pipeline depth (sub-batches in flight; 1 = sequential)"),
    _k(KERNEL_OUTCOME, GROUP_RESULT, ENUM, "on a kernel-attempt/v1 record, by its lane: what became of the attempt (screened: recorded, "
       "not judged; frontier: a distinct route kept for search; kept: adopted; dropped; voided: it didn't measure what it claims; "
       "superseded); the latest label wins", KERNEL_OUTCOMES),
    _k(KERNEL_REASON, GROUP_RESULT, TEXT, "on a kernel-attempt/v1 record: the kernel_outcome's reason, one line"),
    # verification
    _k(COMMIT_MAP, GROUP_VERIFICATION, ENUM, "on a commit-map/v1 artifact, by its approver (research.commitmap.ACCEPTORS): whether the rewrite it "
       "records applies, so that records keyed by an old commit are read for its image; the latest label wins", COMMIT_MAP_VALUES),
    _k(VERIFIED, GROUP_VERIFICATION, ENUM, "an independent verifier's verdict on the result's proof(s)", VERIFIED_VALUES),
    _k(VERIFIER, GROUP_VERIFICATION, TEXT, "which verifier produced the verdict (binary or module + revision)"),
    _k(VERIFIER_SECONDS, GROUP_VERIFICATION, NUMBER, "the verifier's wall seconds for the verdict"),
    _k(INDEPENDENTLY_VERIFIED, GROUP_VERIFICATION, BOOL, "the back-fill's spelling of verified=accepted (true) / rejected (false); read, not written anew",
       status=DEPRECATED),
    _k(VERIFIED_BY, GROUP_VERIFICATION, TEXT, "a ledger row's free-text field: no verdict, ignored by every table (name the verifier under `verifier`)",
       status=IGNORED),
    _k(FINDING, GROUP_VERIFICATION, TEXT, "a red-team or audit finding in prose (DOWNGRADE / EARNED / HOLDS ...), beside the proof_class verdict it explains"),
    _k(SAME_DEVICE, GROUP_VERIFICATION, BOOL, "whether the verifier ran on the prover's machine"),
    _k(CUSTODY, GROUP_VERIFICATION, ENUM, "on a run id, by whoever looked at its records: whether custody triage needs more of it (waived: "
       "nothing more, its logs stay where they are, or are gone; never-ran: it never ran its workload, so it holds no evidence, which "
       "triage believes only while the run dir shows no sign of a start: no pid in status.json, no started, span_start or sampler_start "
       "event, no sampler reading, no output; needed: triage reads its records again, which withdraws an earlier waived or never-ran); "
       "the latest label wins", CUSTODY_VALUES),
    _k(SOURCE_LOST, GROUP_VERIFICATION, BOOL, "on a run id: the exact source it ran is gone (its tree held a dirty source that no record "
       "kept, and it was deleted), so the run cannot be re-run or audited from its source; the ref is the note that says how"),
    _k(BELOW_BAR, GROUP_VERIFICATION, BOOL, "the verified proof's whole-proof bound is below the tables' 2^-128 bar: shown in the below-bar drill-down, never a Table 2 cell"),
    _k(OUTCOME, GROUP_VERIFICATION, TEXT, "an Attempt's failure class as judged (CGROUP_OOM, UNKNOWN_EXIT, ...), restating or correcting the classifier's"),
    _k(GRANT, GROUP_VERIFICATION, ENUM, "a reviewer's grant on pr:<n>@<head>: the role that reviewed that exact head (the merge queue admits on it)",
       GRANT_ROLES),
    _k(READY, GROUP_VERIFICATION, BOOL, "the owner's word that pr:<n>@<head> is ready for the merge queue (`research queue ready`); a new head needs it "
       "again; a train member needs none (Q14)"),
    _k(LEAN_REVIEW, GROUP_VERIFICATION, TEXT, "on pr:<n>@<head>: the art: id of an `audit.py --update` report (its audit.json); a lean-audit.json "
       "change it printed whole as the same statements (moved under --moved, printed differently, rehashed) needs no statement-reviewer grant"),
    _k(LANDED, GROUP_VERIFICATION, TEXT, "on a run (a check, or a part of one), by research-merge: main's commit whose Check: or "
       "Checked: trailer names it (`research merge`; `--label-landed` for past landings)", pattern=r"[0-9a-f]{40}"),
    # evidence
    _k(INSTRUCTION, GROUP_EVIDENCE, TEXT, "the tensor-core instruction probed, as the registry names it (sm89.mma.m16n8k32.e4m3)"),
    _k(MODEL, GROUP_EVIDENCE, TEXT, "the silicon model the capture was compared with"),
    _k(MODEL_AGREES, GROUP_EVIDENCE, BOOL, "whether the capture agrees with `model` on every tile"),
    _k(MODEL_STATUS, GROUP_EVIDENCE, TEXT, "the model's standing after the capture, e.g. hypothesis"),
    _k(MODEL_REJECTED, GROUP_EVIDENCE, TEXT, "model(s) the capture rules out"),
    _k(EVIDENCE, GROUP_EVIDENCE, TEXT, "what the capture consists of: sweep, sass, descriptor, golden_replay, fp8_families (+-joined)"),
    _k(FAMILY, GROUP_EVIDENCE, TEXT, "the numerical family the capture or instance set exercises"),
    _k(INSTANCES_DATASET, GROUP_EVIDENCE, TEXT, "the instance dataset a run or set draws from, e.g. bench-instances/v1"),
    _k(INSTANCES_TIER, GROUP_EVIDENCE, TEXT, "the tier within the dataset, e.g. vu-k1536"),
    _k(MISMATCHES, GROUP_EVIDENCE, INT, "tiles or instances that disagreed with the reference"),
    _k(TARGET, GROUP_EVIDENCE, TEXT, "the verification target the evidence is for (a verity.ml.circuits.targets.target name)"),
    _k(DEVICE, GROUP_EVIDENCE, TEXT, "the physical GPU the capture ran on, as its UUID (nvidia-smi --query-gpu=uuid: GPU-...); the trust framework's device identity"),
    _k(INSTANCES_MANIFEST_SHA256, GROUP_EVIDENCE, TEXT, "sha256 of the instance set's manifest (or its contract-pinned recipe digest) a replay checked"),
    _k(COMPONENT, GROUP_EVIDENCE, TEXT, "the component a measurement isolates (encoder, hash, lookup, ...)"),
    _k(JOBS, GROUP_EVIDENCE, INT, "the number of jobs a battery or sweep ran"),
    _k(DTYPE, GROUP_EVIDENCE, TEXT, "the operand format a measurement is for (bf16, e4m3, e2m1)"),
    _k(LEVEL, GROUP_EVIDENCE, TEXT, "a trust dossier's computed trust level for its target (benchmarks/tc_probe/trust.py)"),
    _k(ANCHOR, GROUP_EVIDENCE, TEXT, "the anchor family a trust dossier rests on"),
    # approach
    _k(APPROACH, GROUP_APPROACH, TEXT, "on a run or artifact: the <campaign>/<slug> of the approach it belongs to"),
    _k(TITLE, GROUP_APPROACH, TEXT, "an approach's title, one line"),
    _k(HYPOTHESIS, GROUP_APPROACH, TEXT, "what would be true if the approach works, one line"),
    _k(APPROACH_TYPE, GROUP_APPROACH, ENUM, "a construction, a way to prove, build or measure something, or an attempt to break a target", APPROACH_TYPES),
    _k(ATTACKS, GROUP_APPROACH, TEXT, "an attack's target: the <campaign>/<slug> it tries to break"),
    _k(OWNER, GROUP_APPROACH, TEXT, "the lane that owns the approach"),
    _k(APPROACH_STATUS, GROUP_APPROACH, ENUM, "the approach's own line of work: worked on now, open to anyone, cannot work, replaced, done and adopted",
       APPROACH_STATUSES),
    _k(REASON, GROUP_APPROACH, TEXT, "why the approach's status changed, one line (required unless live)"),
    _k(CITES, GROUP_APPROACH, TEXT, "one evidence token the approach cites: a run id, art:, note:, <repo>#<n>, <repo>@<sha>, https://..."),
    _k(UNCITES, GROUP_APPROACH, TEXT, "one evidence token the approach no longer cites: replaced, or where nobody else can read it"),
)}

KEYS = tuple(VOCAB)


def key(name: str) -> Key | None:
    return VOCAB.get(name)


def is_known(name: str) -> bool:
    return name in VOCAB


def check(name: str, value: Any) -> str | None:
    """Why (key, value) is outside the vocabulary -- an unknown key, an old spelling of a key or value, a value outside its enum, a
    bool/int/number key holding the word instead of the value, a ref written as a label -- as one line, or None."""
    if name in KEY_ALIASES:
        return f"label key {name!r} is the old spelling of {KEY_ALIASES[name]!r} (read as such; write {KEY_ALIASES[name]!r})"
    if name in REF_KEYS:
        return f"label key {name!r} names another store object: that is a ref (`research data put --ref {name}=art:...`), not a label"
    if name in MEASUREMENT_KEYS:
        return f"label key {name!r} is a measurement or run parameter: it belongs in the result or the artifact's meta, not a label"
    k = VOCAB.get(name)
    if k is None:
        near = suggest(name)
        return f"label key {name!r} is not in the vocabulary" + (f" (did you mean {' or '.join(map(repr, near))}?)" if near else "") + \
            " -- `research data vocab` lists the keys"
    if isinstance(value, str) and (name, value) in VALUE_ALIASES:
        return f"{name}={value!r} is the old spelling of {VALUE_ALIASES[(name, value)]!r} (read as such; write {VALUE_ALIASES[(name, value)]!r})"
    if k.form in (BOOL, INT, NUMBER) and isinstance(value, str) and coerce(name, value) is not value:
        return f"{name}={value!r} is a string; the key's form is {k.form} (`research data label` converts it; other writers must store the {k.form})"
    if k.form in (INT, NUMBER) and isinstance(value, str):
        return f"{name}={value!r} is not {'an int' if k.form == INT else 'a number'}"
    return k.check(value)


def check_ref(name: str, ref: str | None) -> str | None:
    """Why a `name` label's ref is not the form `REF_FORMS` requires (a missing ref included), as one line, or None."""
    form = REF_FORMS.get(name)
    if form is None or (ref is not None and form[0].match(ref)):
        return None
    return f"label key {name!r} needs --ref naming {form[1]}" + (f", not {ref!r}" if ref is not None else "")


def suggest(name: str) -> list[str]:
    """Up to two known keys (or old spellings, as their canonical key) close to `name`, case-insensitively."""
    import difflib  # noqa: PLC0415

    pool = {k.lower(): k for k in KEYS} | {a.lower(): c for a, c in KEY_ALIASES.items()}
    out: list[str] = []
    for hit in difflib.get_close_matches(name.lower(), list(pool), n=4, cutoff=0.6):
        if pool[hit] not in out:
            out.append(pool[hit])
    return out[:2]


def by_group() -> dict[str, list[Key]]:
    return {g: [k for k in VOCAB.values() if k.group == g] for g in GROUPS}


def render() -> str:
    """The vocabulary as `research data vocab` prints it: one block per group, one line per key: name, form, doc, enum values, status."""
    heads = {GROUP_RESULT: "result labels -- the producing lane, about its own result",
             GROUP_VERIFICATION: "verification labels -- an independent verifier, never the producer",
             GROUP_EVIDENCE: "evidence labels -- probe captures and instance sets",
             GROUP_APPROACH: "approach labels -- the registry: the owning lane or its coordinator, about an approach/v1 record"}
    lines: list[str] = []
    for g, keys_ in by_group().items():
        lines.append(heads[g])
        for k in keys_:
            tail = f"  = {' | '.join(k.values)}" if k.form == ENUM else ""
            status = f"  [{k.status}]" if k.status != CURRENT else ""
            lines.append(f"  {k.name:<24} {k.form:<7} {k.doc}{tail}{status}")
        lines.append("")
    return "\n".join(lines).rstrip("\n")
