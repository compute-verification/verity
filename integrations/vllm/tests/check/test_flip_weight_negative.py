"""A substituted weight input as a LIVE canary negative: commit_delta --flip-weight x canary.sh flipweight.

The canary (verity_vllm/ops/canary.sh) runs `negative flipweight exit3 <regex> ... --flip-weight model.layers.0.self_attn.qkv_proj.weight:7`
on the SmolLM2 positive's derived weights_of_record.json and grades the attack DETECTED when commit_delta exits 3 with a first_fail_reason
matching <regex>.  Two facts have to stay in lock-step and neither is exercised by the GPU-free suites otherwise:

  1. commit_delta declares --flip-weight and applies the flip BEFORE register_weights (else the committed weights root would still be
     the honest one and the pin would compare honest digests: a negative that cannot fail), and records the flip in verdict.json
     (negatives.flip_weight) so an attack record is never mistaken for a row Commit.
  2. the verdict text a pin FAIL produces on the dense path -- `commit_verdict` names the FAILING value_correspondence component:
     "weights_pin: 1 served weight(s) differ from the checkpoint of record: <served name>", or
     "weights of record: ..." when only the verdict's own pin rule fires -- matches the canary's regex; a rename of either side turns the
     live negative into a FAIL of the canary (harness gap), never into a silent PASS.

Torch-free: (1) is read from the source with ast/str.index, (2) runs the real commit_verdict on a synthetic Commit record with the real
weights_of_record.check mismatch text and matches it against the regex extracted from canary.sh (bash ERE == Python re for this pattern).
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

from verity_vllm.check.commit_pass import commit_verdict

BENCH = Path(__file__).resolve().parents[2] / "verity_vllm" / "ops"
DELTA = BENCH.parents[0] / "pipeline" / "commit.py"
CANARY = BENCH / "canary.sh"


def _canary_flipweight_line() -> str:
    lines = CANARY.read_text().splitlines()
    i = next(i for i, l in enumerate(lines) if "negative flipweight exit3" in l)
    return lines[i] + " " + lines[i + 1].strip()


def _canary_regex() -> str:
    m = re.search(r'negative flipweight exit3 "((?:[^"\\]|\\.)*)"', _canary_flipweight_line())
    assert m, "canary.sh: negative flipweight exit3 \"<regex>\" not found"
    return m.group(1)


def test_commit_delta_declares_flip_weight_and_applies_it_before_register_weights():
    src = DELTA.read_text()
    tree = ast.parse(src)
    options = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Options")
    opts = [a.value for node in ast.walk(options) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "option" for a in node.args if isinstance(a, ast.Constant)]
    assert opts.count("--flip-weight") == 1
    # order in the source of main(): record initialised -> flip applied (named_parameters, before register_weights) -> verdict.json carries it
    i_init = src.index("flip_weight_rec: dict | None = None")
    i_flip = src.index("_pmap = dict(eng.model.named_parameters())")
    i_reg = src.index("w = GR.register_weights(_HOT, com, args.weights_hash)")
    i_verdict = src.index('"negatives": ({"flip_weight": flip_weight_rec} if flip_weight_rec else None)')
    assert i_init < i_flip < i_reg < i_verdict
    # the flip refuses to be a no-op (an unchanged element would make the negative un-failable)
    assert "did not change the element" in src
    # the flag is documented as a negative-test-only flag, never a row-procedure flag
    for driver in ("row_stages.py", "row_tp.py"):
        assert "--flip-weight" not in (BENCH.parent / "pipeline" / driver).read_text()


def test_canary_flipweight_regex_matches_the_verdict_text_of_a_pin_fail():
    rx = _canary_regex()
    assert "--flip-weight model.layers.0.self_attn.qkv_proj.weight:" in _canary_flipweight_line()
    assert "--weights-of-record" in _canary_flipweight_line() and "--program-digest" in _canary_flipweight_line()
    # the real mismatch text of weights_of_record.check (one field), as the canary's attack produces it
    why = "1 served weight(s) differ from the checkpoint of record: model.layers.0.self_attn.qkv_proj.weight"
    wp = {"result": False, "why": why, "root_of_record": "ab" * 32, "compared": 303, "mismatched": [{"served_name": "model.layers.0.self_attn.qkv_proj.weight"}],
          "missing": [], "unpinned": []}
    bnd = {"context": {"ok": True}, "coverage": {"ok": True, "expected_n": 1, "bound_n": 1, "missing_n": 0}, "openings_population": {"skipped": "test", "unbound_n": 0}, "population": {"requests": 1}}
    base = {"extra": {"expected_tensors": 3, "committed_tensors": 3, "errors_n": 0}, "openings": {"n": 64, "verified": 64}, "binding": bnd,
            "sampled_replay": {"result": True, "grade": "complete"}, "boundary_linkage": {"result": True, "pairs": 6, "linked": 6}}
    # (a) the dense path: commit_delta sets value_correspondence.result False and names the component -> "weights_pin: <why>"
    vc = {"method": "sampled-exact-replay(complete) + boundary-linkage", "result": False, "components": {"weights_pin": {"result": False, "why": why}}}
    ok, reason = commit_verdict({**base, "value_correspondence": vc, "weights_pin": wp}, True, min_openings=64)
    assert ok is False and reason.startswith("weights_pin: ") and why in reason, reason
    assert re.search(rx, reason), (rx, reason)
    # (b) only the verdict's own pin rule: "weights of record: <why>"
    vc_ok = {"method": "sampled-exact-replay(complete) + boundary-linkage", "result": True}
    ok, reason = commit_verdict({**base, "value_correspondence": vc_ok, "weights_pin": wp}, True, min_openings=64)
    assert ok is False and reason.startswith("weights of record: ") and re.search(rx, reason), reason
    # (c) the honest positive does not match the attack's regex and PASSes
    ok, reason = commit_verdict({**base, "value_correspondence": vc_ok, "weights_pin": {**wp, "result": True, "why": None, "mismatched": []}}, True, min_openings=64)
    assert ok is True and not re.search(rx, reason)
    # (d) a pin FAIL naming ANOTHER field is not this attack's detection (the regex names the flipped field)
    other = why.replace("model.layers.0.self_attn.qkv_proj.weight", "model.layers.3.mlp.down_proj.weight")
    ok, reason = commit_verdict({**base, "value_correspondence": vc_ok, "weights_pin": {**wp, "why": other}}, True, min_openings=64)
    assert ok is False and not re.search(rx, reason)
