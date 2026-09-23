"""The label vocabulary (README §8.1): every label key in use, grouped by who asserts it, with the value form it takes.

The store never validates a label beyond its envelope (§8: any key, any canonicalisable value); this module is the *shared spelling*
of the keys and of the closed value sets, so that a producer, a verifier and a table read the same word.  `research data label`
warns (stderr, one line) on a key outside the vocabulary or a value outside a declared enum -- and still writes the label: an
older worktree, a key a lane needs today, a value the enum has not learnt yet all keep working; the warning is the nudge to bring
the word here.  `research data vocab` prints the table.  Consumers import the constants (`vocab.PROOF_CLASS`, never the string).

Three groups, by asserter:

    result        the producing lane, about its own result / ledger row (`ledger_to_store`, or `research data label` by hand)
    verification  an independent verifier, never the producer (a table counts it only when `by` is not a producer)
    evidence      a probe or instance-set lane, about a capture / dataset (`tc-evidence/v1`, `dataset-snapshot/v1`, `fixture/v1`)

Value forms: `enum` (one of `values`), `bool` (`true` | `false`), `int`, `number`, `text`, `list` (of strings).  Enum and bool
values are compared as the catalog stores them (`index.scalar_text`: strings raw, everything else JSON), so `true` and `"true"`
are the same word and `"True"` / `"yes"` are not.  Only enums and bools are checked; the other forms are documentation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .index import scalar_text

GROUP_RESULT, GROUP_VERIFICATION, GROUP_EVIDENCE = "result", "verification", "evidence"
GROUPS = (GROUP_RESULT, GROUP_VERIFICATION, GROUP_EVIDENCE)
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
BREAKTHROUGH = "breakthrough"

# ---- verification labels: an independent verifier, never the producer ----------------------------------------------------------------

VERIFIED = "verified"
VERIFIER = "verifier"
VERIFIER_SECONDS = "verifier_seconds"
INDEPENDENTLY_VERIFIED = "independently_verified"     # the back-fill's spelling of verified=accepted|rejected: accepted, deprecated
VERIFIED_BY = "verified_by"                           # a ledger row's free-text field: carries no verdict, ignored by every table

# ---- evidence labels: probe captures and instance sets --------------------------------------------------------------------------------

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

# ---- closed value sets -------------------------------------------------------------------------------------------------------------------

VERIFIED_ACCEPTED, VERIFIED_REJECTED, VERIFIED_NOT_TRANSFERABLE, VERIFIED_NO_DUMPS = "accepted", "rejected", "not-transferable", "no-dumps"
VERIFIED_VALUES = (VERIFIED_ACCEPTED, VERIFIED_REJECTED, VERIFIED_NOT_TRANSFERABLE, VERIFIED_NO_DUMPS)
#: `verity_numerical.bench.contract.PROOF_CLASSES` + the ledger's `COMPONENT` (a bucket measured alone is no proof class).  The store
#: imports neither package (the `research` rule), so the spelling is pinned here; the ledger's tests cross-check it.
PROOF_CLASSES = ("NO_PROOF", "ARITHMETIC_DIAGNOSTIC", "NON_ZK_PROOF_DIAGNOSTIC", "NON_ZK_PROOF", "COMPLETE_HVZK_BACKEND", "COMPLETE_ZK_BACKEND",
                 "COMPONENT")
AUTHENTICATION_VALUES = ("included", "excluded")
MODES = ("interactive", "fiat-shamir")
SCOPES = ("unit", "vu", "component")
TRACKS = ("A", "B", "C", "shared", "scout", "baseline")
#: As `ledger_to_store.candidate` derives them from the lane name.  (Table 1 of `bench.tables` spells its rows `A-GKR`, `B-Ligero`,
#: `SP1`; a `candidate` label only ever marks its asserter as the producer, so the two spellings have not collided yet.)
CANDIDATES = ("A", "B-AIR", "B-Ligero", "C", "SP1", "shared")


@dataclass(frozen=True)
class Key:
    """One vocabulary entry.  `values` is the closed set of an `enum` (and `true`/`false` for a `bool`), in `scalar_text` form."""

    name: str
    group: str
    form: str
    doc: str
    values: tuple[str, ...] = ()
    status: str = CURRENT

    def check(self, value: Any) -> str | None:
        """Why `value` is outside this key's declared set, as one line, or None (free forms are never checked)."""
        if self.values and scalar_text(value) not in self.values:
            return f"{self.name}={scalar_text(value)!r} is not one of {', '.join(self.values)}"
        return None


def _k(name: str, group: str, form: str, doc: str, values: tuple[str, ...] = (), status: str = CURRENT) -> Key:
    return Key(name, group, form, doc, TRUE_FALSE if form == BOOL else values, status)


VOCAB: dict[str, Key] = {k.name: k for k in (
    # result
    _k(CANDIDATE, GROUP_RESULT, ENUM, "the candidate proof system the result belongs to", CANDIDATES),
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
    _k(BREAKTHROUGH, GROUP_RESULT, BOOL, "the lane's own flag that the row moves the campaign"),
    # verification
    _k(VERIFIED, GROUP_VERIFICATION, ENUM, "an independent verifier's verdict on the result's proof(s)", VERIFIED_VALUES),
    _k(VERIFIER, GROUP_VERIFICATION, TEXT, "which verifier produced the verdict (binary or module + revision)"),
    _k(VERIFIER_SECONDS, GROUP_VERIFICATION, NUMBER, "the verifier's wall seconds for the verdict"),
    _k(INDEPENDENTLY_VERIFIED, GROUP_VERIFICATION, BOOL, "the back-fill's spelling of verified=accepted (true) / rejected (false); read, not written anew",
       status=DEPRECATED),
    _k(VERIFIED_BY, GROUP_VERIFICATION, TEXT, "a ledger row's free-text field: no verdict, ignored by every table (name the verifier under `verifier`)",
       status=IGNORED),
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
    _k(TARGET, GROUP_EVIDENCE, TEXT, "the verification target the evidence is for (a verity.verification.target name)"),
    _k(DEVICE, GROUP_EVIDENCE, TEXT, "the physical GPU the capture ran on, as its UUID (nvidia-smi --query-gpu=uuid: GPU-...); the trust framework's device identity"),
    _k(INSTANCES_MANIFEST_SHA256, GROUP_EVIDENCE, TEXT, "sha256 of the instance set's manifest (or its contract-pinned recipe digest) a replay checked"),
    _k(COMPONENT, GROUP_EVIDENCE, TEXT, "the component a measurement isolates (encoder, hash, lookup, ...)"),
    _k(JOBS, GROUP_EVIDENCE, INT, "the number of jobs a battery or sweep ran"),
)}

KEYS = tuple(VOCAB)


def key(name: str) -> Key | None:
    return VOCAB.get(name)


def is_known(name: str) -> bool:
    return name in VOCAB


def check(name: str, value: Any) -> str | None:
    """Why (key, value) is outside the vocabulary -- an unknown key, or a value outside its enum -- as one line, or None."""
    k = VOCAB.get(name)
    if k is None:
        return f"label key {name!r} is not in the vocabulary (`research data vocab` lists it)"
    return k.check(value)


def by_group() -> dict[str, list[Key]]:
    return {g: [k for k in VOCAB.values() if k.group == g] for g in GROUPS}


def render() -> str:
    """The vocabulary as `research data vocab` prints it: one block per group, one line per key: name, form, doc, enum values, status."""
    heads = {GROUP_RESULT: "result labels -- the producing lane, about its own result",
             GROUP_VERIFICATION: "verification labels -- an independent verifier, never the producer",
             GROUP_EVIDENCE: "evidence labels -- probe captures and instance sets"}
    lines: list[str] = []
    for g, keys_ in by_group().items():
        lines.append(heads[g])
        for k in keys_:
            tail = f"  = {' | '.join(k.values)}" if k.form == ENUM else ""
            status = f"  [{k.status}]" if k.status != CURRENT else ""
            lines.append(f"  {k.name:<24} {k.form:<7} {k.doc}{tail}{status}")
        lines.append("")
    return "\n".join(lines).rstrip("\n")
