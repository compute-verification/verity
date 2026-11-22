"""`credit.py`'s rules on a served window and its period: each mutation of
`pouw_credit_cases` is refused by the rule it aims at, its message naming the fix, and the honest windows and periods are
accepted. The sampled proof blocks are the sampling coin's over the window's own layout, read from the descriptor of the circuit the
period committed (`credit.layout_of`), whose strata are the compiled `h1window.program`'s, and the weights a window
registers are the ones its open root binds, every code and s_w of its B rows. The Lean (`PouwVerifier.Credit`) is held to
the same cases by `lean/check/tests/test_difftest.py`."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import random
import sys
from pathlib import Path

from fractions import Fraction

import pytest

from verity.protocols.accounting.compute.pouw.verifier import core_check as CC
from verity.protocols.accounting.compute.pouw.verifier import credit as Cr
from verity.protocols.accounting.compute.pouw.verifier import salt as PS
from verity.protocols.accounting.compute.pouw.verifier import weights as Wt
from verity.protocols.accounting.compute.pouw.verifier import window as Wd
from verity.protocols.accounting.compute.pouw.prover.circuit import h1window as HW
from verity.protocols.accounting.compute.pouw.prover.circuit import plan
from verity.protocols.proof.spot.verifier.challenger import challenger as Ch
from verity.protocols.proof.spot.verifier.one_stage import partition as P
from verity.protocols.proof.spot.verifier.one_stage import registered as G
from verity.commitments import registered_value as RV
from verity.circuits import codec

PKG = Path(__file__).resolve().parent / "lean" / "check"
for _name, _dir in (("pouw_rounds_cases", PKG), ("pouw_credit_cases", PKG), ("test_circuit_h1window", Path(__file__).parent)):
    if _name not in sys.modules:
        _spec = importlib.util.spec_from_file_location(_name, _dir / f"{_name}.py")
        sys.modules[_name] = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(sys.modules[_name])
cases, rc, h1 = (sys.modules[n] for n in ("pouw_credit_cases", "pouw_rounds_cases", "test_circuit_h1window"))

SEEDS = range(3)
#: a sample root no withheld session opened on: its sampling coin was never sampled after it
NO_DRAW_SESSION = "the draw's session is not trusted: the auditor's ledger opens no challenger session for this window"

#: each `_credited` mutation's refusal: its code and its reason's start
CREDIT_EXPECT = {
    "changed": ("coin", NO_DRAW_SESSION), "withdrawn": ("coin", NO_DRAW_SESSION), "row-changed": ("coin", NO_DRAW_SESSION),
    "stale-root": ("draw", Cr.DRAW_ROOT), "no-bit": ("credit", "manifest row "), "bit-form": ("credit", "manifest row "),
    "no-z": ("credit", "call "), "index": ("manifest", "manifest row "), "row-form": ("manifest", "manifest row "),
    "calls-form": ("manifest", Cr.CALLS_FORM), "salt-form": ("manifest", Cr.SALT_FORM),
    "served-form": ("manifest", Cr.SERVED_FORM), "commit-form": ("commit", Cr.COMMIT_FORM), "rounds-form": ("rounds", ""),
    "draw-form": ("draw", Cr.DRAW_FORM),
    "no-draw": ("draw", Cr.NO_DRAW), "draw-coin": ("coin", "the draw coin: "),
    "key-form": ("coin", "the verifier's challenger key is malformed"),
    "forged-ledger": ("coin", "the draw's session is not trusted: "),
    "unpinned": ("coin", "the draw's session is not trusted: session "),
    "branch-pin": ("coin", "the draw's session is not trusted: session "),
    "unwithheld": ("coin", Ch.WITHHELD), "draw-elsewhere": ("coin", NO_DRAW_SESSION),
    "draw-session": ("coin", "the draw's session is not trusted: "), "draw-after-other": ("draw", Cr.CHANGED),
}
#: a session pinned to a logged commit off main's first-parent line
OFF_MAIN = "which the host's pins log records but which is not on main's first-parent line"
NOT_DRAWN = "has a proof outcome and was not drawn"
NO_SERVED = "states no served rows m_s at most the period's T = "
OTHER_SERVED = "every dense call of a step serves the step's m_s rows"
UNTIED = "whose rows each round's root committed"
NOT_COMMITS = "is not the Commit's "
NOT_RECEIVED = "the round roots are not the roots the challenger's log shows it received, in order"
FEWER = "units are fewer than K = "
NO_K = "commits no draw K (draw_k"
OTHER_REGISTRATION = "'s domain is not the one the window's registration gives it"
OTHER_ROOT = "against another root than the one the window registered for"
OTHER_PERIOD_WEIGHTS = "open signed: a B row's codes or s_w differ from the ones the period registered"
UNSIGNED = "open signs no weights' root (weights_root, a SHA-512 digest"
NO_ROWS = "'s open commits no table rows T (rows, a positive integer"
OTHER_SLOT = "the period's T = "
LIVE_FINDING = "proof opened other ports, values or roots than its statement reads"
#: a tile at a K its proof doesn't cover (`credit.tile_k`), between its K and its fix
TILE_K = "is outside the K its proof covers, multiples of 32 below 131072 (the K its GemmAccumulatorS8 binds): "
DRAWN_EXPECT = {
    "coin-form": ("draw", Cr.COIN_FORM), "args-form": ("draw", Cr.DRAW_ARGS), "layout-form": ("layout", Cr.LAYOUT_FORM),
    "named-twice": ("layout", "is named twice"), "unknown-linear": ("layout", "which is no linear of the window"),
    "uncalled": ("layout", "is never called"), "slot": ("layout", "(h1window.slot)"),
    "kn": ("layout", "n whole blocks of the tile's"), "blocks": ("layout", "is read by some call of K"),
    "k-form": ("budget", Cr.K_FORM), "k-past-units": ("budget", FEWER),
}
WINDOW_DRAWN_EXPECT = {
    "coin-form": ("coin", "the draw coin: " + Ch.OPENING_FORM), "args-form": ("draw", Cr.DRAW_ARGS),
    "descriptor-hex": ("layout", Cr.DESCRIPTOR_HEX), "k-form": ("budget", Cr.K_FORM), "k-past-units": ("budget", FEWER),
}
OUTCOME_EXPECT = {
    "missing": ("missing", "Prove and verify every drawn unit"), "failed": ("failed", "refuses the whole window"),
    "twice": ("outcome", "a retry never reproves it"), "undrawn": ("outcome", NOT_DRAWN), "chosen": ("outcome", NOT_DRAWN),
    "proofs-form": ("outcome", Cr.PROOFS_FORM),
}
WINDOW_EXPECT = {
    "manifest-count": ("manifest", "each call of the window's Program is one manifest row"),
    "chosen": ("outcome", NOT_DRAWN), "failed": ("failed", "no partial credit"), "missing": ("missing", "no partial credit"),
    "descriptor-hex": ("layout", Cr.DESCRIPTOR_HEX), "descriptor-form": ("layout", Cr.DESCRIPTOR_FORM),
    "not-window": ("layout", Cr.NOT_WINDOW), "template": ("layout", "is not h1window's"),
    "tile-ks": ("layout", "h1window binds one template per K"), "body": ("layout", Cr.BODY),
    "layout-form": ("layout", Cr.LAYOUT_FORM), "record-layout": ("layout", "which is no linear of the window"),
    "alt-layout": ("layout", Cr.BODY), "reinterpret": ("layout", Cr.DESCRIPTOR_FORM), "tile-k": ("tile-k", TILE_K),
    "credit-changed": ("coin", NO_DRAW_SESSION), "credit-no-draw": ("draw", Cr.NO_DRAW),
    "credit-forged-ledger": ("coin", "the draw's session is not trusted"),
    "credit-branch-pin": ("coin", OFF_MAIN),
    "k-form": ("budget", Cr.K_FORM), "k-past-units": ("budget", FEWER),
    "never-received": ("rounds", NOT_RECEIVED), "round-coin": ("coin", "coin was not opened after"),
    "a-untied": ("a-link", UNTIED), "no-a": ("a-link", Cr.NO_A),
    "other-registration": ("registration", OTHER_REGISTRATION), "a-call-unbound": ("registration", OTHER_REGISTRATION),
    "proof-other-registration": ("registration", OTHER_ROOT),
    "entry-root": ("weights", Cr.OTHER_WEIGHTS),
    "proof-other-circuit": ("registration", "was verified against another circuit than the window's"),
    "proof-no-a": ("registration", "opened no A value …/a/"),
    "proof-no-b": ("registration", "'s proof opened no weights "),
    "proof-other-call": ("registration", "opened no A value …/a/"),
    "proof-unregistered": ("registration", "which the window registers not"),
    "entry-twice": ("weights", "'s weights twice ("),
    "entry-form": ("registration", "entry is not one-stage's {value, root, leaves, words, domain}"),
    "weights-period": ("weights", OTHER_PERIOD_WEIGHTS), "weights-unsigned": ("weights", UNSIGNED),
    "weights-model": ("weights", Cr.MODEL_FIX),
    "weights-no-period": ("weights", "the window is in no period of the auditor's ledger"),
    "weights-requests": ("weights", Cr.OTHER_WEIGHTS), "weights-no-requests": ("weights", Cr.REQUESTS_FORM),
    "weights-leaves": ("weights", "'s entry is not the registration of K = "),
    "a-zero": ("arow", NOT_COMMITS + "row-table/"), "a-row-off": ("arow", NOT_COMMITS + "row-table/"),
    "ids-off": ("arow", NOT_COMMITS + "input-ids/"), "ids-missing": ("arow", "'s input-ids table "),
    "commit-missing": ("arow", "the Commit record registers "), "commit-twice": ("arow", "the Commit record registers "),
    "commit-missing-expert": ("arow", f"/k{cases.EXPERT_K} not: the served Commit registers each round's A table"),
    "ids0-outside": ("arow", Cr.IDS_OUTSIDE), "ids0-root-off": ("arow", Cr.IDS_OUTSIDE),
    "ids0-other-session": ("registration", "ids@0" + OTHER_REGISTRATION),
    "req0-root-off": ("arow", Cr.REQUESTS_OUTSIDE), "req0-twice": ("arow", Cr.REQUESTS_OUTSIDE),
    "req0-other-prefix": ("arow", "the window registers request table x/"),
    "req0-other-session": ("registration", "requests@0" + OTHER_REGISTRATION),
    "req0-window": ("registration", "requests@0" + OTHER_REGISTRATION),
    "a-call-off": ("arow", NOT_COMMITS + "row-table/"), "a-call-missing": ("a-link", "the window's layout gives it"),
    "commit-unbound": ("draw", Cr.DRAW_ROOT), "credit-commit-form": ("commit", Cr.COMMIT_FORM),
    "rows-past": ("manifest", NO_SERVED), "rows-form": ("manifest", NO_SERVED), "rows-other": ("manifest", OTHER_SERVED),
    "length-none": ("length", "no-fixed-length"), "length-eos": ("length", "eos-keyed-row"),
    "length-form": ("length", "is not {max_tokens, stop}"), "length-past": ("length", "ran past its public max_tokens"),
    "length-early": ("length", "its done source is not the public cap"),
    "length-slots": ("length", "more than the window's"),
    "t-other": ("layout", OTHER_SLOT), "t-form": ("layout", Cr.ROWS_FORM),
    "live-no-slots": ("live", "each registered once"), "live-no-requests": ("live", "each registered once"),
    "live-slots-rows": ("live", "16-bit words its live unit reads"),
    "live-requests-rows": ("live", "16-bit words its live unit reads"),
    "live-form": ("live", "live evidence is not {descriptor, proofs}"),
    "live-descriptor-form": ("live", Cr.LIVE_DESCRIPTOR_FORM), "live-not-live": ("live", Cr.NOT_LIVE),
    "live-body": ("live", Cr.LIVE_BODY), "live-t-other": ("live", "live Program is at T = "),
    "live-s-other": ("live", "rounds, not the window's"), "live-prefix": ("live", Cr.LIVE_PREFIX),
    "live-missing": ("live", "has no proof outcome"), "live-twice": ("live", "has two proof outcomes"),
    "live-past": ("live", "has a proof outcome and the window has"), "live-failed": ("live", "proof was rejected"),
    "live-other-circuit": ("live", "against another circuit than the window's live Program"),
    "live-m-other": ("live", LIVE_FINDING), "live-round-other": ("live", LIVE_FINDING),
    "live-slots-other": ("live", LIVE_FINDING), "live-caps-swapped": ("live", LIVE_FINDING),
    "live-caps-other": ("live", LIVE_FINDING),
    "live-ok-off": ("live", LIVE_FINDING), "live-proofs-form": ("live", Cr.LIVE_PROOFS_FORM),
    "live-v1": ("live", Cr.NOT_LIVE),
}
PERIOD_EXPECT = {
    "no-close": ("period", "present the ledger after its close"), "incomplete": ("period", "closed incomplete"),
    "missing": ("missing", "present them all, in the close's order"),
    "order": ("order", "in the order its close lists them"), "extra": ("period", "only its close's sessions count"),
    "foreign": ("period", "only the sessions its close lists count"), "unopened": ("period", "opens no period"),
    "name": ("period", "the period is not named"), "window-refused": ("window", "so the period earns nothing"),
    "windows-form": ("period", Cr.WINDOWS_FORM), "forged-ledger": ("coin", "the auditor's ledger is not trusted"),
    "key-form": ("coin", "the verifier's challenger key is malformed"),
    "other-circuit": ("circuit", "prove the period's declared circuit"),
    "no-circuit": ("circuit", "commits no cap or circuit"), "descriptor-form": ("circuit", Cr.DESCRIPTOR_HEX),
    "no-k": ("budget", NO_K), "k-form": ("budget", NO_K), "k-past-units": ("window", FEWER),
    "no-rows": ("layout", NO_ROWS), "t-form": ("layout", NO_ROWS),
    "t-other": ("window", "so the period earns nothing: layout: " + OTHER_SLOT),
    "other-k": ("window", "so the period earns nothing"),
    "alt-layout": ("circuit", "prove the period's declared circuit"),
    "bad-circuit": ("window", "so the period earns nothing: layout: "),
    "tile-k": ("window", "so the period earns nothing: tile-k: the window's tile at K = "),
    "never-received": ("window", "so the period earns nothing: rounds: " + NOT_RECEIVED),
    "proof-other-registration": ("window", "so the period earns nothing: registration: drawn unit "),
    "branch-pin": ("circuit", "doesn't count toward it, so the period earns nothing: session s-"),
    "no-weights": ("weights", UNSIGNED), "weights-form": ("weights", UNSIGNED),
    "no-model": ("weights", "open signs a weights' root but no model (model, {name, checkpoint_sha512})"),
    "model-form": ("weights", "open signs a weights' root but no model (model, {name, checkpoint_sha512})"),
    "other-checkpoint": ("weights", "with another checkpoint than the one credit pins for it: " + Cr.MODEL_FIX),
    "other-model": ("weights", "which is no checkpoint role credit pins: " + Cr.MODEL_FIX),
    "other-sw": ("window", "so the period earns nothing: weights: the window's registered weights' root "),
    "zero-codes": ("window", "so the period earns nothing: weights: the window's registered weights' root "),
    "live-refused": ("window", "so the period earns nothing: live: round "),
    "stale-pin": ("pin", "past its freshness floor"), "length-none": ("window", "length: no-fixed-length"),
}
A_LINK_EXPECT = {
    "untied": UNTIED, "late-rows": UNTIED, "rewritten": UNTIED, "round-root": "`credit.round_root`", "gap": "`credit.cover`",
    "extra": "`credit.cover`", "swapped": "`credit.cover`", "misaligned": "`credit.cover`",
    "unknown": "which is no registered A value", "no-a": Cr.NO_A, "twice": " twice", "entry-form": "has no root, domain", "leaves-cap": "is not from 1 to",
    "layout-rows": "the window's layout gives it",
    "blocks": "rounds' equal blocks", "log-form": Cr.LOG_FORM, "round-form": "inner root is not",
    "segments-form": "A segments are not", "entries-form": Cr.ENTRIES_FORM,
    "table-other-round": ", which only round ", "table-past": "and the window has",
    "ids-other-round": "'s input-ids table, which only round ", "ids-past": "is round ",
    "slots-other-round": "'s slot table, which only round ", "slots-past": "registered slot table ",
    "receipt-other-round": "'s receipt table, its round ", "receipt-past": "rounds' equal blocks with a call",
    "receipt-calls": "rounds' equal blocks with a call", "receipt-unknown": "or receipt table of the window",
}
WEIGHTS_EXPECT = {
    "other-root": Cr.OTHER_WEIGHTS, "other-requests": Cr.OTHER_WEIGHTS, "other-session": Cr.OTHER_WEIGHTS,
    "other-window": Cr.OTHER_WEIGHTS, "missing": "the window registers no weights ", "twice": "'s weights twice",
    "foreign": "b/<K> of a K of its circuit", "other-prefix": "b/<K> of a K of its circuit",
    "other-leaves": "'s entry is not the registration of K = ", "other-words": "'s entry is not the registration of K = ",
    "other-domain": "'s entry is not the registration of K = ", "entry-form": "'s entry is not the registration of K = ",
    "root-form": "'s entry is not the registration of K = ", "window-form": Cr.WINDOW_ROOT_FORM,
    "requests-form": Cr.REQUESTS_FORM, "session-form": Cr.SESSION_FORM, "circuit-form": Cr.CIRCUIT_FORM,
    "entries-form": Cr.ENTRIES_FORM,
}


def _credited(name: str, seed: int):
    d = cases.credit_instance(random.Random(f"{name}/{seed}"), name)
    return d, Cr._credited(*cases.credit_args(d))


#: the mutations whose stand-in window's template isn't pinned (`h1_tiles.json`), so `credit.layout_of` refuses it
#: before the refusal each is aimed at; each is owed on a pinned window
PIN_REFUSED_WINDOW = frozenset({"a-call-missing", "a-call-off", "a-call-unbound", "a-row-off", "a-untied", "a-zero",
    "alt-layout", "body", "chosen", "commit-missing", "commit-missing-expert", "commit-twice", "entry-form",
    "entry-root", "entry-twice", "failed", "ids-missing", "ids-off", "ids0-other-session", "ids0-outside",
    "ids0-root-off", "k-form", "k-past-units", "layout-form", "length-early", "length-eos", "length-form",
    "length-none", "length-past", "length-slots", "live-body", "live-caps-other", "live-caps-swapped",
    "live-descriptor-form", "live-failed", "live-form", "live-m-other", "live-missing", "live-no-requests",
    "live-no-slots", "live-not-live", "live-ok-off", "live-other-circuit", "live-past", "live-prefix",
    "live-proofs-form", "live-requests-rows", "live-round-other", "live-s-other", "live-slots-other", "live-slots-rows",
    "live-t-other", "live-twice", "live-v1", "manifest-count", "missing", "no-a", "other-registration", "proof-no-a",
    "proof-no-b", "proof-other-call", "proof-other-circuit", "proof-other-registration", "proof-unregistered",
    "record-layout", "req0-other-prefix", "req0-other-session", "req0-root-off", "req0-twice", "req0-window",
    "rows-form", "rows-other", "rows-past", "t-form", "t-other", "tile-k", "tile-ks", "weights-leaves", "weights-model",
    "weights-no-period", "weights-no-requests", "weights-period", "weights-requests", "weights-unsigned"})
PIN_REFUSED_PERIOD = frozenset({"alt-layout", "branch-pin", "k-past-units", "length-none", "live-refused",
    "never-received", "other-circuit", "other-sw", "proof-other-registration", "t-other", "tile-k", "zero-codes"})
PIN_REFUSED_WINDOW_DRAWN = frozenset({"args-form", "coin-form", "k-form", "k-past-units"})


def test_every_mutation_is_aimed():
    assert PIN_REFUSED_WINDOW <= set(WINDOW_EXPECT) and PIN_REFUSED_PERIOD <= set(PERIOD_EXPECT)
    assert PIN_REFUSED_WINDOW_DRAWN <= set(WINDOW_DRAWN_EXPECT)
    assert set(CREDIT_EXPECT) == set(cases.CREDIT_NAMES)
    assert set(DRAWN_EXPECT) == set(cases.DRAWN_NAMES) - {"budget"}
    assert set(WINDOW_DRAWN_EXPECT) == set(cases.WINDOW_DRAWN_NAMES)
    assert set(OUTCOME_EXPECT) == set(cases.OUTCOME_NAMES)
    assert set(WINDOW_EXPECT) == set(cases.WINDOW_NAMES) - {"j10r"}
    assert set(PERIOD_EXPECT) == set(cases.PERIOD_NAMES)
    assert set(A_LINK_EXPECT) == set(cases.A_LINK_NAMES)
    assert set(WEIGHTS_EXPECT) == set(cases.WEIGHTS_NAMES) - {"layout-form"}


def test_each_pinned_model_is_its_files_digest():
    """`models.json` pins each role's checkpoint as the auditor's registry writes it (`checkpoint_sha512`, the SHA-512 of
    the canonical JSON of each file's SHA-512), so an open the gate signed from a registration of a pinned checkpoint
    is the pin credit counts."""
    doc = json.loads(Cr.MODELS.read_text())
    assert doc
    for role, e in doc.items():
        assert sorted(e) == ["checkpoint_sha512", "files", "recipe", "repo", "revision"], role
        assert e["files"] and all(Cr._digest(h) is not None for h in e["files"].values()), role
        assert e["checkpoint_sha512"] == hashlib.sha512(codec.canonical_json(e["files"])).hexdigest(), role
    assert Cr.PINNED_MODELS == {r: e["checkpoint_sha512"] for r, e in doc.items()}
    assert Cr.MODELS.read_text() == json.dumps(doc, indent=1, sort_keys=True) + "\n"


def test_llama32_1b_int8_and_its_k2048_role_pin_one_checkpoint():
    """LLAMA32_1B_INT8 is vllm's int8 checkpoint of unsloth/Llama-3.2-1B (`verity-int8-channel127/v1`, snapshot 15b0976d):
    the 6 files its `checkpoints-int8.json` entry lists by size and sha256, each by SHA-512, and its checkpoint_sha512
    their canonical JSON's digest. LLAMA32_1B_INT8_K2048 is the same checkpoint under the role whose manifest entry
    declares its K = 2048 linears (a J10-R period's at one K), so both pin one digest."""
    doc = json.loads(Cr.MODELS.read_text())
    e = doc["LLAMA32_1B_INT8"]
    assert (e["repo"], e["revision"], e["recipe"]) == (
        "verity-int8/Llama-3.2-1B-INT8", "15b0976d400d796ee589e7737d03a919526bf800", "verity-int8-channel127/v1")
    assert set(e["files"]) == {"config.json", "generation_config.json", "model.safetensors", "special_tokens_map.json",
                               "tokenizer.json", "tokenizer_config.json"}
    assert e["checkpoint_sha512"] == hashlib.sha512(codec.canonical_json(e["files"])).hexdigest()
    assert doc["LLAMA32_1B_INT8_K2048"] == e
    assert Cr.PINNED_MODELS["LLAMA32_1B_INT8_K2048"] == e["checkpoint_sha512"]


def test_llama32_1b_bf16_pins_the_upstream_checkpoint():
    """LLAMA32_1B_BF16 is unsloth/Llama-3.2-1B as upstream holds it (`upstream-bf16`, snapshot 9535bd9b), the checkpoint
    LLAMA32_1B_INT8 is made from: the 6 files a served engine reads, each by SHA-512. It is the role a remote memory audit
    serves Llama 3.2 1B under, since its store's layout covers BF16 weights; credit counts none of its weights."""
    e = json.loads(Cr.MODELS.read_text())["LLAMA32_1B_BF16"]
    assert (e["repo"], e["revision"], e["recipe"]) == (
        "unsloth/Llama-3.2-1B", "9535bd9b1d1dea6acafbdc4813b728796aeb28da", "upstream-bf16")
    assert set(e["files"]) == {"config.json", "generation_config.json", "model.safetensors", "special_tokens_map.json",
                               "tokenizer.json", "tokenizer_config.json"}
    assert e["checkpoint_sha512"] == hashlib.sha512(codec.canonical_json(e["files"])).hexdigest()
    assert Cr.PINNED_MODELS["LLAMA32_1B_BF16"] == e["checkpoint_sha512"]


def test_glm53_int8_pins_the_v3_checkpoint():
    """GLM53_INT8 is vllm's v3 int8 checkpoint of zai-org/GLM-5.3 (`verity-int8-channel127/v3`, snapshot 7065fe60): the
    146 files its `checkpoints-int8.json` entry lists by size and sha256 (141 shards, their index, the config, the
    generation config and the tokenizer), each by SHA-512, and its checkpoint_sha512 their canonical JSON's digest."""
    e = json.loads(Cr.MODELS.read_text())["GLM53_INT8"]
    assert (e["repo"], e["revision"], e["recipe"]) == (
        "verity-int8/GLM-5.3-INT8-V3", "7065fe60d0a46749ee796e5b3b8fc0ab75d1ce44", "verity-int8-channel127/v3")
    shards = {f"model-{i:05d}-of-00141.safetensors" for i in range(1, 142)}
    assert set(e["files"]) == shards | {"config.json", "generation_config.json", "model.safetensors.index.json",
                                        "tokenizer.json", "tokenizer_config.json"}
    assert e["checkpoint_sha512"] == hashlib.sha512(codec.canonical_json(e["files"])).hexdigest()
    assert Cr.PINNED_MODELS["GLM53_INT8"] == e["checkpoint_sha512"]


@pytest.mark.parametrize("seed", SEEDS)
def test_honest_window_credits_its_bits(seed):
    d, (got, refused) = _credited("honest", seed)
    assert refused is None
    assert got == [i for i, row in enumerate(d["calls"]) if row[Cr.CREDIT]]


@pytest.mark.parametrize("seed", SEEDS)
def test_credit_bit_changed_after_draw_is_refused(seed):
    """A verdict withdrawn (or granted) after the sampling coin, the sample root recomputed over the new bits, is no root the
    ledger opened a withheld sample session on; a sampling coin opened after another root than the sample root is not the one
    the challenger's log shows the sample followed."""
    for name in ("withdrawn", "changed"):
        _, (got, refused) = _credited(name, seed)
        assert got is None and refused == ("coin", NO_DRAW_SESSION)
    _, (got, refused) = _credited("draw-after-other", seed)
    assert got is None and refused == ("draw", Cr.CHANGED)
    assert "Commit every call's credit bit in its manifest row, and the Commit record, before the draw root" in refused[1]


@pytest.mark.parametrize("seed", SEEDS)
def test_credited_call_needs_a_bit_and_a_z_root(seed):
    for name, fix in (("no-bit", "commits each call's verdict as `credit`"), ("no-z", "carries its z table's root")):
        _, (_, refused) = _credited(name, seed)
        assert refused[0] == "credit" and fix in refused[1]


@pytest.mark.parametrize("seed", SEEDS)
def test_failed_or_missing_drawn_unit_refuses_the_window(seed):
    """No partial credit."""
    assert _outcome("honest", seed) is None
    missing, failed = _outcome("missing", seed), _outcome("failed", seed)
    assert missing[0] == "missing" and "Prove and verify every drawn unit" in missing[1]
    assert failed[0] == "failed" and "refuses the whole window" in failed[1]
    assert all("no partial credit" in r[1] for r in (missing, failed))


@pytest.mark.parametrize("seed", SEEDS)
def test_a_tree_not_tied_to_the_round_roots_is_refused(seed):
    """An A tree registered over rows other than those each round's root committed before its coin."""
    assert Cr._a_link(**cases.a_link_instance(random.Random(f"honest/{seed}"))) is None
    for name in ("untied", "late-rows", "rewritten"):
        refused = Cr._a_link(**cases.a_link_instance(random.Random(f"{name}/{seed}"), name))
        assert refused[0] == "a-link" and UNTIED in refused[1]


@pytest.mark.parametrize("name", sorted(CREDIT_EXPECT))
def test_credit_mutation(name):
    code, start = CREDIT_EXPECT[name]
    for seed in SEEDS:
        _, (got, refused) = _credited(name, seed)
        assert got is None and refused[0] == code and refused[1].startswith(start), (seed, refused)


@pytest.mark.parametrize("name", sorted(OUTCOME_EXPECT))
def test_outcome_mutation(name):
    code, part = OUTCOME_EXPECT[name]
    for seed in range(5):
        refused = _outcome(name, seed)
        assert refused[0] == code and part in refused[1], (seed, refused)


@pytest.mark.parametrize("name", sorted(DRAWN_EXPECT))
def test_drawn_mutation(name):
    code, part = DRAWN_EXPECT[name]
    for seed in range(5):
        got, refused = _drawn(name, seed)
        assert got is None and refused[0] == code and part in refused[1], (seed, refused)


@pytest.mark.parametrize("name", sorted(set(WINDOW_EXPECT) - PIN_REFUSED_WINDOW))
def test_window_mutation(name):
    code, part = WINDOW_EXPECT[name]
    for seed in SEEDS:
        got, refused = _window(name, seed)
        assert got is None and refused[0] == code and part in refused[1], (seed, refused)


@pytest.mark.parametrize("name", sorted(set(PERIOD_EXPECT) - PIN_REFUSED_PERIOD))
def test_period_mutation(name):
    code, part = PERIOD_EXPECT[name]
    for seed in SEEDS:
        got, refused = _period(name, seed)
        assert got is None and refused[0] == code and part in refused[1], (seed, refused)


def _drawn(name: str, seed: int):
    d = cases.drawn_instance(random.Random(f"{name}/{seed}"), name)
    return Cr.drawn_twin(*(d[k] for k in cases.DRAWN_KEYS))


def _window_drawn(name: str, seed: int):
    d = cases.window_drawn_instance(random.Random(f"{name}/{seed}"), name)
    if not (type(d["descriptor"]) is str and Cr._HEX.fullmatch(d["descriptor"])):
        return d, (None, ("layout", Cr.DESCRIPTOR_HEX))
    lay, why = Cr.layout_of(bytes.fromhex(d["descriptor"]))
    assert why is None, why
    ss, why = Cr.strata(lay)
    assert why is None, why
    return d, Cr._window_drawn(d["window"], d["draw"], ss, d["k"])


@pytest.mark.parametrize("name", sorted(set(WINDOW_DRAWN_EXPECT) - PIN_REFUSED_WINDOW_DRAWN))
def test_window_drawn_mutation(name):
    code, part = WINDOW_DRAWN_EXPECT[name]
    for seed in range(5):
        _d, (got, refused) = _window_drawn(name, seed)
        assert got is None and refused[0] == code and part in refused[1], (seed, refused)


def _outcome(name: str, seed: int):
    d = cases.outcome_instance(random.Random(f"{name}/{seed}"), name)
    return Cr._outcome(*(d[k] for k in cases.OUTCOME_KEYS))


def _window(name: str, seed: int):
    return Cr._window_outcomes(*cases.window_args(cases.window_instance(random.Random(f"{name}/{seed}"), name)))


def _period(name: str, seed: int):
    d = cases.period_instance(random.Random(f"{name}/{seed}"), name)
    return Cr._period_outcomes(*(d[k] for k in cases.PERIOD_KEYS), rc.lms_verify)


def test_a_draw_at_the_record_k():
    """At a committed K = `plan.RECORD_K` (the least K with (1 − ε)^K ≤ δ at ε = 0.1% and δ = 2⁻⁴⁰) over a larger
    population, the work sampling design samples K units, each stratum's k_s rounded up, so at most one more a stratum."""
    assert cases.PLAN_K == plan.RECORD_K == plan.record_k(Fraction(1, 1000), Fraction(1, 2 ** 40))
    for seed in range(2):
        got, refused = _drawn("budget", seed)
        ss, _ = Cr.strata(cases.drawn_instance(random.Random(f"budget/{seed}"), "budget")["layout"])
        assert refused is None and sum(hi - lo for s in ss for lo, hi in s["units"]) > plan.RECORD_K
        assert plan.RECORD_K <= len(got) < plan.RECORD_K + len(ss) and got == sorted(set(got))


@pytest.mark.parametrize("seed", SEEDS)
def test_the_drawn_units_are_the_draw_coins(seed):
    """`credit.drawn` derives the sampled proof blocks from the sampling coin: each stratum's units are its own, its count the work
    sampling design's k_s, and another coin, stream, registration or receipt samples another set; never a list handed in."""
    d = cases.drawn_instance(random.Random(f"honest/{seed}"))
    args = [d[k] for k in cases.DRAWN_KEYS]
    got, refused = Cr.drawn_twin(*args)
    ss, _ = Cr.strata(d["layout"])
    assert Cr.sampling_design(ss, d["k"]) is None
    assert refused is None and got == sorted(set(got))
    for s in ss:
        assert sum(any(lo <= u < hi for lo, hi in s["units"]) for u in got) == s["k"]
    big = {**d, "layout": cases.layout(random.Random(f"wide/{seed}"), big=True), "k": 100}
    base, _ = Cr.drawn_twin(*(big[k] for k in cases.DRAWN_KEYS))
    for k, v in (("coin", "ab" * 64), ("stream", "another"), ("registration", "cd" * 64), ("receipt", {"receipts": "x"})):
        assert Cr.drawn_twin(*({**big, k: v}[x] for x in cases.DRAWN_KEYS))[0] != base, k


def test_the_strata_are_the_compiled_programs():
    """`credit.strata` over a layout is `partition.strata` of the compiled `h1window.program` and its query: the same
    units in the Program's order, one stratum per K's template, each unit's work its tile's R·C·K MACs."""
    for linears, calls, m in ((h1.LINEARS, h1.CALLS, 8), (h1.LINEARS, h1.CALLS[::-1], 16),
                              ((("k_proj", 32, 16), ("q_proj", 32, 16)), (("k_proj", 0), ("q_proj", 0)), 8)):
        prog = h1.HW.program(linears, calls, m, h1.TILE, unit=h1.unit)
        q = h1.HW.query(prog)
        ids = dict(zip(sorted({k for _n, k, _c in linears}), q["params"]["templates"]))
        ss, refused = Cr.strata({"calls": [list(c) for c in calls], "linears": [list(x) for x in linears], "m": m,
                                 "tile": list(h1.TILE)})
        assert refused is None
        assert [(ids[int(s["template"][1:])], s["units"]) for s in ss] == h1.P.strata(prog, q)
        assert [s["work"] for s in ss] == [h1.R * h1.C * int(s["template"][1:]) for s in ss]
        assert all(s["floor"] == 1 for s in ss)


def test_window_and_period_credit_are_the_only_entries():
    """A caller can't credit a window with a step skipped: the steps `window` composes are internal, and so are the
    forms that take proof outcomes in place of proof files (the Lean's, until it runs the verifier)."""
    for step in ("credited", "outcome", "a_link", "registration", "verified", "asks", "lean_verified",
                 "live_verified", "live_asks", "lean_live_verified", "window_outcomes", "period_outcomes"):
        assert not hasattr(Cr, step) and callable(getattr(Cr, f"_{step}")), step
    assert {"window", "period_credit"} <= set(Cr.__all__)
    assert not {"_credited", "_outcome", "_a_link", "_registration", "_verified", "_asks", "_lean_verified",
                "_live_verified", "_live_asks", "_lean_live_verified", "_window_outcomes",
                "_period_outcomes"} & set(Cr.__all__)


def test_mains_call_tables_are_the_windows_own_and_the_weights_held():
    """The names main's served form registers (`h1window.table`, `h1window.roles`): a call's step's A table of its K
    ``a/<K>@<s>`` is an A value of K and that step, the one `credit.step_tables` gives the call, its z table ``zy@<call>`` a value of the window's own, and the weights' ``b/<K>`` the
    only values held across windows."""
    prog = h1.HW.program(h1.LINEARS, h1.CALLS, 8, h1.TILE, unit=h1.unit, prefix="h1/w-3/")
    lay = {"m": 8, "linears": [list(x) for x in h1.LINEARS], "calls": [list(x) for x in h1.CALLS]}
    for steps, c in ((s, c) for s in range(1, len(h1.CALLS) + 1) if len(h1.CALLS) % s == 0
                     for c in range(len(h1.CALLS))):
        k = dict((name, k) for name, k, _n in h1.LINEARS)[h1.CALLS[c][0]]
        a, zy = h1.HW.table(prog, "a", c, steps), h1.HW.table(prog, "zy", c)
        s, n, at = Cr.step_row(lay, steps, c)
        assert Cr.a_value(a) == (str(k), str(s)) and not Cr.is_held(a)
        assert h1.HW.step_of(prog, c, steps) == (s, n, at) and (s, k) in {(t, j) for t, j, _ in Cr.step_tables(lay, steps)}
        assert not Cr.is_a(zy) and not Cr.is_held(zy)
    for role in h1.HW.roles(k for _n, k, _c in h1.LINEARS):
        v = "h1/w-3/" + role
        assert Cr.is_held(v) == role.startswith("b/") and Cr.is_a(v) == role.startswith("a/"), role
    assert not any(Cr.is_a(v) for v in ("h1/a/x", "h1/a/4096@", "h1/a/@3", "h1/a/4096@3@4", "a/64@-1", 5))


def test_a_glm_window_at_k_6144_registers_its_b_rows_as_2_chunk_leaves_each_past_2_to_the_24():
    """GLM's routed experts at K = 6144 (`weights.chunks` 2: each B row 2 leaves of `weights.leaf_words` (6144) = 1538
    words): Σn = 9 900 032 rows are 19 800 064 leaves, above 2^24. `_weights` accepts b/6144 at those leaves under its
    registered domain, since the layout fixes a weights' value's rows and no `MAX_ROWS` bounds them, and one-stage's
    entry form (`_entry`) takes it; the whole-row registration (Σn leaves of `b_words` (6144)) is refused."""
    sigma = 9_900_032
    lay = {"calls": [["experts", 0]], "linears": [["experts", 6144, sigma]], "m": 32, "tile": [32, 32]}
    digest, requests, session = "ef" * 64, "ab" * 64, "s-1"
    x, w = Wt.chunks(6144), Wt.leaf_words(6144)
    assert Cr.strata(lay)[1] is None and (x, w) == (2, 1538) and sigma < Cr.MAX_ROWS < x * sigma
    root = hashlib.sha512(b"b/6144").digest()
    entry = {"value": "b/6144", "root": root.hex(), "leaves": x * sigma, "words": w,
             "domain": RV.domain(digest, "b/6144", w, x * sigma).domain_id.hex()}
    wr = Wt.weights_root({6144: root})
    window = {"root": PS.window_open_root(wr, bytes.fromhex(requests), session).hex()}
    assert Cr._entry(entry)
    assert Cr._weights(window, lay, [entry], requests, session, digest, "") == (wr, None)
    bw = Wt.b_words(6144)
    whole = {**entry, "leaves": sigma, "words": bw, "domain": RV.domain(digest, "b/6144", bw, sigma).domain_id.hex()}
    assert Cr._weights(window, lay, [whole], requests, session, digest, "") == (
        None, ("weights", Cr.other_weights_entry("b/6144", 6144, sigma)))
    assert Cr._weights(window, {**lay, "linears": [["experts", 6200, sigma]]}, [entry], requests, session, digest,
                       "")[1][0] == "layout"


def test_the_live_program_is_read_from_its_compiled_descriptor():
    """`credit.live_of` reads the compiled `live.program_v2`'s T, S and PREFIX back from its descriptor (canonical JSON
    without annotations, the bytes its program digest hashes), its unit the one template its query samples, unit s round
    s's; the cases' stand-in descriptor (`live_program`) reads the same. A version-1 Program's descriptor, whose units
    don't bind each request's cap to its public max_tokens, is no live Program credit reads."""
    from verity.protocols.accounting.compute.pouw.prover.circuit import live as LV

    def raw(prog):
        return codec.canonical_json({k: v for k, v in codec.encode_program(prog).items() if k != "annotations"})

    for t, s, prefix in ((4, 3, "h1/w-1/"), (1, 1, ""), (32, 2, "x/y/")):
        prog = LV.program_v2(t, s, prefix)
        got, why = Cr.live_of(raw(prog))
        assert why is None and (got["T"], got["S"], got["prefix"]) == (t, s, prefix)
        assert LV.query_v2(t) == {"name": Cr.QUERY[0], "version": Cr.QUERY[1], "params": {"templates": [got["unit"]]}}
        assert hashlib.sha512(raw(prog)).hexdigest() == h1.P.program_digest(prog)
        assert Cr.live_of(bytes.fromhex(cases.live_hex(cases.live_program(t, s, prefix)))) == (got, None)
        assert Cr.live_of(bytes.fromhex(cases.live_hex(cases.live_program(t, s, prefix, version=1)))) == \
            (None, ("live", Cr.NOT_LIVE))


def test_the_commit_record_is_bound_under_calls_root():
    """The Commit record's digest is its format's tagged SHA-512 of its bytes (`registration.digest`), and `calls_root`
    binds it after the manifest and the served trees' digest, as vllm's `pouw_h1.calls_root` must, byte for byte."""
    from verity.commitments import docket_digest as DD
    d = cases.window_instance(random.Random("honest/bound"))
    raw = bytes.fromhex(d["commit"])
    (digest, _reg), _ = Cr.commit_of(d["commit"])
    assert digest.hex() == DD.digest(json.loads(raw)) and codec.canonical_json(json.loads(raw)) == raw
    salt, served = bytes.fromhex(d["salt"]), bytes(64)
    inner = Cr.calls_root(salt, d["calls"], served)
    assert Cr.calls_root(salt, d["calls"], served, digest) == hashlib.sha512(Cr.COMMIT_TAG + inner + digest).digest()
    assert Cr.calls_root(salt, d["calls"], None, digest) != Cr.calls_root(salt, d["calls"], served, digest)


@pytest.mark.parametrize("seed", [2])
def test_a_round_log_the_challenger_never_received_refuses(seed):
    """The composed check's counterexample: a round log the challenger never received, its coins carried as if opened
    after its roots and the sample root over it, has every credit bit fixed before the sampling coin (`_credited` credits its
    calls) and its A trees tied to its roots (`_a_link` accepts). Only `core_check.rounds_twin` holds the log's roots to the
    ones the challenger received, and `window` runs it, so the window, and the period it's in, earn nothing."""
    d = cases.window_instance(random.Random(f"never-received/{seed}"), "never-received")
    got, refused = Cr._credited(*cases.credit_args(d))
    assert refused is None and got == [i for i, row in enumerate(d["calls"]) if row[Cr.CREDIT]]
    assert Cr._a_link(d["rounds"], d["entries"], len(d["calls"])) is None
    rounds = CC.rounds_twin(d["key"], d["ledger"], d["pins"], d["main"], d["window"], d["session"],
                       [x["root"] for x in d["rounds"]], [x["coin"] for x in d["rounds"]], rc.lms_verify)
    assert rounds == ("rounds", NOT_RECEIVED)
    assert Cr._window_outcomes(*cases.window_args(d)) == (None, rounds)
    got, refused = _period("never-received", seed)
    assert got is None and refused[0] == "window" and refused[1].endswith(f"rounds: {NOT_RECEIVED}")


def test_the_reported_eps_is_of_the_credited_work():
    """ε is a share of the credited work, not of the window's: with 8 of 9 equal calls credited (u = 8/9) the K′ of
    ε = 0.1% at `DELTA` is 31,178, and its ε is at most 0.1%, where taking credited = W would report 8/9 of that, too
    optimistic by 9/8. A window credited nothing establishes no ε."""
    lay = {"calls": [["w", 0]] * 9, "linears": [["w", 4096, 1 << 20]], "m": 16, "tile": [16, 16]}
    size, work = Cr.call_work(lay)[0]
    total = 9 * size * work
    k = Wd.k_prime(Fraction(1, 1000), Cr.DELTA, 8 * size * work, total, 1)
    eps = Cr.established(lay, list(range(8)), [16] * 9, k)
    assert k == 31_178 and Fraction(9999, 10_000_000) < eps <= Fraction(1, 1000)
    assert eps == Wd.eps_at(k, Cr.DELTA, 8 * size * work, total, 1)
    assert Fraction(9, 8) * Wd.eps_at(k, Cr.DELTA, total, total, 1) == pytest.approx(eps, rel=1e-5)
    assert Cr.established(lay, [], [16] * 9, k) is None


def test_a_padded_step_is_credited_its_served_rows_not_its_slots():
    """Every call has the window's M slot rows and its step's m_s served rows first; the empty rows' units prove
    trivially. A credited call of a step serving m_s = 2 of M = 16 rows is credited m_s·n·K, not M·n·K, so the ε its
    sample establishes is the one over that work, and the sample's population stays every slot's units."""
    k, n = 4096, 1 << 20
    lay = {"calls": [["w", 0]] * 2, "linears": [["w", k, n]], "m": 16, "tile": [16, 16]}
    total = sum(size * work for size, work in Cr.call_work(lay))
    assert total == 2 * 16 * n * k
    assert Cr.served_work(lay, [2, 16]) == [(2 * n // 16, 16 * k), (16 * n // 16, 16 * k)]
    assert Cr.credited_work(lay, [0], [2, 16]) == 2 * n * k
    assert Cr.credited_work(lay, [0, 1], [16, 16]) == total
    big_k = Wd.k_prime(Fraction(1, 1000), Cr.DELTA, 2 * n * k, total, 1)
    padded = Cr.established(lay, [0], [2, 16], big_k)
    assert padded == Wd.eps_at(big_k, Cr.DELTA, 2 * n * k, total, 1) <= Fraction(1, 1000)
    assert Cr.established(lay, [0], [16, 16], big_k) == Wd.eps_at(big_k, Cr.DELTA, 16 * n * k, total, 1) < padded


def test_an_empty_call_is_credited_nothing():
    """A call whose step served no rows (m_s = 0, an engine step past the last served one) is credited no work, and a
    window credited only such calls establishes no ε."""
    lay = {"calls": [["w", 0]] * 2, "linears": [["w", 64, 1 << 10]], "m": 16, "tile": [16, 16]}
    assert Cr.credited_work(lay, [0], [0, 16]) == 0
    assert Cr.established(lay, [0], [0, 16], 1000) is None
    assert Cr.credited_work(lay, [0, 1], [0, 16]) == Cr.credited_work(lay, [1], [0, 16])


def test_served_rows_are_one_natural_at_most_t_a_step():
    """Each call's m_s is its manifest row's ``served``, a natural at most the period's T table rows (whose slot is the
    A slot M, so T may be below M: a slot row past T is no table's), and every dense call of a step (call c at step
    c // (C/S)) states the same m_s; steps may differ. Else the window is refused (``manifest``)."""
    lay = {"calls": [["w", 0], ["v", 0], ["w", 1], ["v", 1]], "m": 16}
    rows = lambda *ms: [{Cr.SERVED_ROWS: m} for m in ms]  # noqa: E731
    assert Cr.served_rows(rows(3, 3, 16, 16), lay, 2, 16) == ([3, 3, 16, 16], None)
    assert Cr.served_rows(rows(0, 0, 5, 5), lay, 2, 16) == ([0, 0, 5, 5], None)
    assert Cr.served_rows(rows(3, 4, 16, 16), lay, 2, 16) == (None, ("manifest", Cr.other_served(1, 0, 0, 4, 3)))
    assert Cr.served_rows(rows(3, 3, 16, 15), lay, 2, 16) == (None, ("manifest", Cr.other_served(3, 2, 1, 15, 16)))
    assert Cr.served_rows(rows(3, 3, 3, 4), lay, 1, 16) == (None, ("manifest", Cr.other_served(3, 0, 0, 4, 3)))
    assert Cr.served_rows(rows(17, 17, 3, 3), lay, 2, 16) == (None, ("manifest", Cr.no_served(0, 16)))
    assert Cr.served_rows(rows(12, 12, 3, 3), lay, 2, 12) == ([12, 12, 3, 3], None)
    assert Cr.served_rows(rows(3, 3, 13, 13), lay, 2, 12) == (None, ("manifest", Cr.no_served(2, 12)))
    for bad in (True, -1, "3", None, [3], 1.0):
        assert Cr.served_rows(rows(3, bad, 3, 3), lay, 2, 16) == (None, ("manifest", Cr.no_served(1, 16))), bad
    assert Cr.served_rows([{}, *rows(3, 3, 3)], lay, 2, 16) == (None, ("manifest", Cr.no_served(0, 16)))
    assert OTHER_SERVED in Cr.other_served(1, 0, 0, 4, 3) and NO_SERVED in Cr.no_served(0, 16)


def test_the_record_k_establishes_about_0_1125_percent_at_u_eight_ninths():
    """At K = `plan.RECORD_K` = 27,713, the K that establishes 0.1% of a window's work at `DELTA`, a window credited 8
    of its 9 equal calls reports about 0.1125%, 9/8 of it: the miss bound reads only the missed work εC/W, and
    `Properties.Pouw.Credit.EpsAtSound` holds at the ε `eps_at` answers, so its K′ is at most 27,713 and the grid
    point below needs more."""
    lay = {"calls": [["w", 0]] * 9, "linears": [["w", 4096, 1 << 20]], "m": 16, "tile": [16, 16]}
    size, work = Cr.call_work(lay)[0]
    total, credited = 9 * size * work, 8 * size * work
    eps = Cr.established(lay, list(range(8)), [16] * 9, plan.RECORD_K)
    assert eps == Wd.eps_at(plan.RECORD_K, Cr.DELTA, credited, total, 1)
    assert Fraction(1_124, 1_000_000) < eps < Fraction(1_126, 1_000_000)
    assert Wd.k_prime(eps, Cr.DELTA, credited, total, 1) <= plan.RECORD_K
    assert Wd.k_prime(eps - Fraction(1, 1 << Wd.EPS_BITS), Cr.DELTA, credited, total, 1) > plan.RECORD_K
    assert Fraction(8, 9) * eps == pytest.approx(Wd.eps_at(plan.RECORD_K, Cr.DELTA, total, total, 1), rel=1e-5)


def _worst_miss(strata: list[tuple[int, int, int]], eps: Fraction, total: int) -> Fraction:
    """The largest probability that the work sampling design's sample (each stratum (n_s, w_s, k_s) sampled k_s of its n_s units
    without replacement) misses a set holding at least a share `eps` of the work `total`, over two strata: for each
    count b_1 of the second stratum's units in the set, the fewest of the first's that reach the share (a set's miss
    only falls as it grows)."""
    (n0, w0, k0), (n1, w1, k1) = strata

    def miss(n: int, k: int, b: int) -> Fraction:
        return Fraction(math.comb(n - b, k), math.comb(n, k))

    worst = Fraction(0)
    for b1 in range(n1 + 1):
        b0 = max(0, -(-(eps * total - b1 * w1) // w0))
        if b0 <= n0:
            worst = max(worst, miss(n0, k0, b0) * miss(n1, k1, b1))
    return worst


def test_the_eps_holds_at_the_work_sampling_designs_cap():
    """At K = n the work sampling design samples a stratum of little work only in part, so the cap is K >= W, not the unit count:
    1,000 units of work 1 and 1,000 of work 1,000, all credited, sampled at K = n = 2,000. The reported ε, `eps_at` at
    (C, W, 1), about 1.38%, holds: every set holding that share of the work is missed with probability at most δ. The
    unit-count form `eps_at(K, δ, C, n, w_max)` would report 2^-32, yet a set of 500 work-1 units, 0.05% of the work,
    is missed with probability about 1/4."""
    lay = {"calls": [["a", 0], ["b", 0]], "linears": [["a", 1, 1000], ["b", 1000, 1000]], "m": 1, "tile": [1, 1]}
    ss, why = Cr.strata(lay)
    assert why is None and Cr.sampling_design(ss, 2000) is None
    total = sum(s["work"] * (s["units"][0][1] - s["units"][0][0]) for s in ss)
    drawn = [(s["units"][0][1] - s["units"][0][0], s["work"], s["k"]) for s in ss]
    assert total == 1_001_000 and drawn == [(1000, 1, 2), (1000, 1000, 1000)]
    eps = Cr.established(lay, [0, 1], [1, 1], 2000)
    assert eps == Wd.eps_at(2000, Cr.DELTA, total, total, 1) and Fraction(137, 10_000) < eps < Fraction(138, 10_000)
    assert _worst_miss(drawn, eps, total) <= Cr.DELTA
    by_units = Wd.eps_at(2000, Cr.DELTA, total, 2000, 1000)
    assert by_units == Fraction(1, 1 << Wd.EPS_BITS)
    assert _worst_miss(drawn, Fraction(500, total), total) > Fraction(1, 5) > Cr.DELTA


@pytest.mark.parametrize("light, heavy", [((1000, 1), (1000, 1000)), ((400, 1), (100, 100)), ((600, 2), (200, 300)),
                                          ((300, 1), (300, 64)), ((500, 1), (50, 1000))])
def test_the_unit_count_form_is_refuted_at_its_cap(light, heavy):
    """The ε of `eps_at(K, δ, C, n, w_max)`, the unit count n in place of the work W, is refuted wherever the work sampling design
    samples a stratum of light work only in part at K = n: there it reports 2^-32, and a set of light units holding that
    share of the work is missed with probability above 9/10. At the same sample the work form (C, W, 1) holds, every set
    of its share missed with probability at most δ (`Properties.Pouw.Credit.WorkDrawMisses`)."""
    (n_a, w_a), (n_b, w_b) = light, heavy
    lay = {"calls": [["a", 0], ["b", 0]], "linears": [["a", w_a, n_a], ["b", w_b, n_b]], "m": 1, "tile": [1, 1]}
    n = n_a + n_b
    ss, why = Cr.strata(lay)
    assert why is None and Cr.sampling_design(ss, n) is None
    drawn = [(s["units"][0][1] - s["units"][0][0], s["work"], s["k"]) for s in ss]
    total = sum(size * w for size, w, _k in drawn)
    assert drawn[0][0] == n_a and drawn[0][2] < n_a and drawn[1][2] == n_b
    by_units = Wd.eps_at(n, Cr.DELTA, total, n, w_b)
    assert by_units == Fraction(1, 1 << Wd.EPS_BITS) and _worst_miss(drawn, by_units, total) > Fraction(9, 10)
    eps = Cr.established(lay, [0, 1], [1, 1], n)
    assert eps == Wd.eps_at(n, Cr.DELTA, total, total, 1) and _worst_miss(drawn, eps, total) <= Cr.DELTA


@pytest.mark.parametrize("seed", SEEDS)
def test_a_period_open_with_no_k_refuses(seed):
    """K is the proof system's selection, committed in the period-open before any sample: a period whose open commits
    none, or a malformed one, earns nothing."""
    for name in ("no-k", "k-form"):
        got, refused = _period(name, seed)
        assert got is None and refused[0] == "budget" and NO_K in refused[1] and "before any draw" in refused[1]


@pytest.mark.parametrize("name", sorted(A_LINK_EXPECT))
def test_a_link_mutation(name):
    for seed in range(5):
        refused = Cr._a_link(**cases.a_link_instance(random.Random(f"{name}/{seed}"), name))
        assert refused[0] == "a-link" and A_LINK_EXPECT[name] in refused[1], (seed, refused)


def test_a_calls_receipt_table_in_its_rounds_segments_is_tied_and_an_unknown_segment_is_not():
    """vllm's served path lists call c's ``receipt@<c>`` among the segments of c's round (the rounds the calls' equal
    blocks) and registers it: `_a_link` ties it to that round's root as it does a round's A table, and still refuses a
    round segment of no registered A value, input-ids table or receipt table (``zy@<c>``, ``receipt@0<c>``, a call's
    receipt table the window doesn't register), and a receipt table when the window's call count isn't given."""
    tied = 0
    for seed in range(40):
        d = cases.a_link_instance(random.Random(f"receipt/{seed}"))
        receipts = [e["value"] for e in d["entries"] if Cr.receipt_call(e["value"]) is not None]
        tied += len(receipts)
        for v in receipts:
            own = Cr.receipt_call(v) // (d["calls"] // len(d["rounds"]))
            assert [r for r, x in enumerate(d["rounds"]) if v in x["a"]] == [own], v
        assert Cr._a_link(**d) is None, seed
        if receipts:
            refused = Cr._a_link(d["rounds"], d["entries"])
            assert refused[0] == "a-link" and "rounds' equal blocks with a call" in refused[1], seed
        refused = Cr._a_link(**cases.a_link_instance(random.Random(f"receipt-unknown/{seed}"), "receipt-unknown"))
        assert refused[0] == "a-link" and "which is no registered A value, input-ids table or receipt table" in refused[1]
    assert tied


def test_a_rounds_slot_table_in_its_segments_is_tied_and_a_slot_table_it_doesnt_register_is_not():
    """The live unit's ``slot`` port reads round s's ``slots@<s>`` (`circuit.live`), which vllm's served path registers
    and lists whole among round s's segments: `_a_link` ties it to round s's root as it does ``ids@<s>``, and refuses a
    round segment of a slot table the window doesn't register."""
    tied = 0
    for seed in range(40):
        d = cases.a_link_instance(random.Random(f"slots/{seed}"))
        slots = [e["value"] for e in d["entries"] if Cr.slots_step(e["value"]) is not None]
        tied += len(slots)
        for v in slots:
            assert [r for r, x in enumerate(d["rounds"]) if v in x["a"]] == [Cr.slots_step(v)], v
        assert Cr._a_link(**d) is None, seed
        for v in slots:
            gone = [e for e in d["entries"] if e["value"] != v]
            refused = Cr._a_link(d["rounds"], gone, d["calls"])
            assert refused[0] == "a-link" and f"lists segments of {v}, which is no registered" in refused[1], seed
    assert tied
    assert [Cr.slots_step(v) for v in ("h1/slots@3", "slots@0", "h1/slots@03", "h1/slot@1", "h1/slots@", 7)] == \
        [3, 0, None, None, None, None]
    assert Cr.is_tied("h1/slots@2") and not Cr.is_tied("h1/requests@0")


def test_a_templates_receipt_width_is_its_bits_as_the_statements_output_row_holds_them():
    """`shape_of` reads K's receipt words from its template's return as `h1window.statement_rows` counts a row: the bits
    of its leaves, so h1tile's ``Array(2·sets, U32)`` of 1-bit leaves and a stand-in's one ``Value(16)`` word agree with
    the bits the statement's output row o0 holds; another form, or bits of no whole 16-bit words, is 0."""
    bit, rec = ["Value", 1], lambda t: ["Record", [[Cr.RECEIPT_FIELD, t], ["ok", ["Value", 1]]]]
    for t, words in ((["Array", 6, ["Array", 32, bit]], 12), (["Value", 16], 1), (["Array", 3, ["Value", 16]], 3),
                     (["Tuple", [["Value", 16], ["Array", 4, ["Value", 8]]]], 3), (["Array", 16, bit], 1),
                     (["Array", 8, bit], 0), (["Value", 17], 0), (["Array", 2, ["Value", "16"]], 0), (bit, 0)):
        assert Cr._receipt_words(rec(t)) == words, t
    assert Cr._receipt_words(["Value", 16]) == Cr._receipt_words(["Record", [["ok", ["Value", 16]]]]) == 0


def test_cover_is_the_fewest_aligned_subtrees():
    for lo, hi in ((0, 1), (0, 8), (3, 11), (5, 6), (6, 21), (16, 48)):
        segs = Cr.cover(lo, hi)
        at = lo
        for level, index in segs:
            assert index << level == at
            at += 1 << level
        assert at == hi
    assert Cr.cover(0, 8) == [(3, 0)] and Cr.cover(3, 11) == [(0, 3), (2, 1), (1, 4), (0, 10)]


def test_a_window_registering_the_weights_its_open_root_binds_is_credited_its_root():
    for seed in range(20):
        d = cases.weights_instance(random.Random(f"honest/{seed}"))
        w, refused = Cr._weights(*(d[k] for k in cases.WEIGHTS_KEYS))
        held = {e["value"]: bytes.fromhex(e["root"]) for e in d["entries"] if Cr.is_held(e["value"])}
        assert refused is None and Wt.weights_root({int(v.split("/")[-1]): h for v, h in held.items()}) == w, refused
        assert {v.removeprefix(d["prefix"]) for v in held} == {f"b/{k}" for _n, k, _r in d["layout"]["linears"]}


@pytest.mark.parametrize("name", cases.WEIGHTS_NAMES)
def test_weights_mutation(name):
    for seed in range(5):
        d = cases.weights_instance(random.Random(f"{name}/{seed}"), name)
        got, refused = Cr._weights(*(d[k] for k in cases.WEIGHTS_KEYS))
        assert got is None, (seed, name)
        if name == "layout-form":
            assert refused[0] == "layout", (seed, refused)
        else:
            assert refused[0] == "weights" and WEIGHTS_EXPECT[name] in refused[1], (seed, refused)


def _b_rows(rng: random.Random, n: int, k: int) -> list[tuple[list[int], float]]:
    return [([rng.randrange(-128, 128) for _ in range(k)], rng.choice([2.0 ** -9, 0.0123, 3.5])) for _ in range(n)]


def _b_row(codes: list[int], sw: float) -> list[int]:
    """A B row's 16-bit words: its K int8 codes then their f32 scale s_w (`h1zy.ROW`), as the registry commits them."""
    return cases.b_row_words(bytes(c & 0xFF for c in codes), sw)


def test_a_window_whose_s_w_differs_from_the_registered_is_refused():
    """Each K's b/<K>, its linears' B rows (each row its int8 codes then its f32 s_w) committed as the served path
    registers the weights (`registered_value.Registered`, the domain that binds no window, salts sampled once), folded by
    `weights_root` into the window's open root, is accepted. The same window whose b/32 has one row's s_w changed, every
    code and salt kept, is refused, and so is one with a code changed instead: the open root fixes both before the
    salt. So is a b/32 registered under another circuit's domain, or missing a row, though the open root binds it."""
    m = HW.slot(5, h1.TILE)
    prog = HW.program(h1.LINEARS, h1.CALLS, m, h1.TILE, unit=h1.unit)
    digest, rng = P.program_digest(prog), random.Random(7)
    lay = {"calls": [list(x) for x in h1.CALLS], "linears": [list(x) for x in h1.LINEARS], "m": m, "tile": list(h1.TILE)}
    rows = {k: [r for name, kk, n in h1.LINEARS if kk == k for r in _b_rows(rng, n, k)] for k in (32, 64)}

    def commit(k: int, b: list[tuple[list[int], float]], program: str = digest) -> RV.Registered:
        return RV.Registered.commit(program, f"b/{k}", [_b_row(c, sw) for c, sw in b],
                                   RV.supplied_salts(program, f"b/{k}", len(b)), schema=G.SCHEMA_V2)

    reg = {k: commit(k, b) for k, b in rows.items()}
    assert [r.leaves for r in reg.values()] == [24, 8] and all(r.window is None for r in reg.values())
    requests, session = "ab" * 64, "s-1"
    w = Wt.weights_root({k: r.root for k, r in reg.items()})
    window = {"root": PS.window_open_root(w, bytes.fromhex(requests), session).hex()}
    entries = [r.entry() for r in reg.values()]
    assert Cr._weights(window, lay, entries, requests, session, digest, "") == (w, None)
    j = rng.randrange(24)
    codes, sw = rows[32][j]
    for changed in ((codes, sw * 2), ([codes[0] ^ 1, *codes[1:]], sw)):
        other = commit(32, rows[32][:j] + [changed] + rows[32][j + 1:])
        assert other.root != reg[32].root
        got = Cr._weights(window, lay, [other.entry(), entries[1]], requests, session, digest, "")
        assert got == (None, ("weights", Cr.OTHER_WEIGHTS)), changed
    for other in (commit(32, rows[32], "cd" * 64), commit(32, rows[32][:-1])):
        w2 = Wt.weights_root({32: other.root, 64: reg[64].root})
        window2 = {"root": PS.window_open_root(w2, bytes.fromhex(requests), session).hex()}
        got = Cr._weights(window2, lay, [other.entry(), entries[1]], requests, session, digest, "")
        assert got == (None, ("weights", Cr.other_weights_entry("b/32", 32, 24))), other.leaves
