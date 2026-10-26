"""pad -- the PADDED-TABLE #1 tool (`padded_commit_tiny`) pinned: the explicit one-stage padded Commit of the tiny fixture has every
required word (inactive included) as one leaf each under one root, the inactive leaves canonical, the s6 fields present, inactive picks
opened against the root and replayed by relation, and the twin denominators measured -- the criteria of PADDED-TABLE, by name."""
from __future__ import annotations

import json

import pytest

from verity_vllm.program.registry import lifted as L
from tests.program import padded_commit_tiny as PC

pytestmark = pytest.mark.skipif(PC.pos_leaf is None, reason=f"host core unavailable: {PC.CORE}")


@pytest.mark.slow
def test_eos_at_event_1_lag0_every_required_word_committed_inactive_canonical_picks_opened_and_replayed(tmp_path):
    assert PC.main(["--case", "eos-at-event-1", "--lag", "0", "--draws", "48", "--out", str(tmp_path)]) == 0
    row = json.loads((tmp_path / "row_eos-at-event-1_lag0.json").read_text())
    assert row["label"] == "lifted-padded" and row["rules_version"] == L.PADDED_RULES_VERSION == "padded-v1"
    p = row["padded"]
    c, s, cm = p["counts"], p["inactive_sampling"], p["commit"]
    # build: the full padded Program, fixed, digested; the query of record at unlifted_width_max = 64
    assert p["program"]["query"] == "default(64)" and p["program"]["verdict"]["status"] == "ok" and len(p["program"]["digest"]) == 128
    # counts: the s6 fields; inactivity by enablement is whole owners at a time (no mixed VU); the inactive population is real
    assert c["total_population"] == c["active_population"] + c["inactive_population"] + c["empty_population"] and c["mixed_population"] == 0
    assert c["inactive_population"] > 10_000 and c["inv_population"] == 0 and c["sets_unsupported"] == 0
    # commit + materialization: one leaf per required identity, the inactive leaves canonical 1||0^w, no INV, one root, one map
    assert cm["leaves_equal_required"] and cm["n_leaves"] == c["required_identities"] and cm["inactive_leaves_canonical"] and cm["no_inv"]
    assert c["words_by_class"]["bot"] == c["out_words_inactive_sets"] > 0 and len(cm["root"]) == 64 and len(cm["map_digest"]) == 64
    bmap = json.load(__import__("gzip").open(p["files"]["binding_map"], "rt"))
    assert bmap["n_leaves"] == cm["n_leaves"] and bmap["root"] == cm["root"] and len(bmap["entries"]) == cm["n_leaves"]
    assert all(e["leaf"] == i for i, e in enumerate(bmap["entries"]))  # every entry is ONE identity at ONE leaf -- no ranges
    bot_entries = [e for e in bmap["entries"] if e["class"] == "bot"]
    assert bot_entries and all(e["width"] in (17, 33, 2) for e in bot_entries)  # true lifted widths, nothing split
    # G1: prescribed words unwidened
    assert all(e["width"] in (16, 32) for e in bmap["entries"] if e["class"] == "input")
    # inactive sampling: picks land on inactive VUs, every opening verifies against the ONE root, every replay agrees (bot checked)
    assert s["total_samples"] == 48 and s["samples_on_inactive"] > 0 and s["openings_ok"] == s["openings"] > 0
    assert s["replays_ok"] == s["replays"] == 48 and s["inactive_pick_openings"] > 0
    opens = json.loads((tmp_path / "padded_eos-at-event-1_lag0" / "openings.json").read_text())
    inactive = [x for x in opens["picks"] if x["kind"] == "inactive"]
    assert inactive and all(x["bot_checked"] is True and x["replay_equal"] for x in inactive)
    assert all(int.from_bytes(bytes.fromhex(o["word"]), "little") in (L.enc_bot(16), L.enc_bot(32), L.enc_bot(1)) for x in inactive for o in x["openings"])
    # overhead: the twin denominators are measured, the executed prefix is T=1 under LAG 0 (EOS at event 1), the cap twin's population
    # equals the padded one modulo the guard/controller identities (A14.2)
    oe, oc = row["overhead_vs_twin_executed_prefix"], row["overhead_vs_twin_cap"]
    assert oe["twin"]["steps"] == 1 and oc["twin"]["steps"] == 8 and oe["bytes_ratio"] > oc["bytes_ratio"] > 1.0
    assert 0 < oc["extra_leaves"] < 100 and abs(oc["population_ratio"] - 1.0) < 0.002
    # per owner: the inactive suffix is present step by step -- every step after the lag step bot at every word it owns
    po = p["per_owner"]
    assert po["s=1/step"]["bot"] == 0 and po["s=1/step"]["active"] > 0
    assert all(po[f"s={t}/step"]["active"] == 0 and po[f"s={t}/step"]["bot"] == po[f"s={t}/step"]["words"] > 0 for t in range(2, 9))


@pytest.mark.slow
def test_lag1_executes_the_post_eos_step_and_the_twin_is_the_executed_prefix(tmp_path):
    assert PC.main(["--case", "eos-at-event-1", "--lag", "1", "--draws", "16", "--out", str(tmp_path)]) == 0
    row = json.loads((tmp_path / "row_eos-at-event-1_lag1.json").read_text())
    po = row["padded"]["per_owner"]
    assert po["s=2/step"]["active"] == po["s=2/step"]["words"] > 0 and po["s=3/step"]["bot"] == po["s=3/step"]["words"] > 0  # R17-1
    assert row["overhead_vs_twin_executed_prefix"]["twin"]["steps"] == 2 and row["padded"]["events"]["active"] == 2  # served != executed
