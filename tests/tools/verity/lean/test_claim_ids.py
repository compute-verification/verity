"""`tools/verity/lean/claim_ids.py`: the claim ids' table, from the ids the locks' premise entries carry and the rows no
lock carries."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("lean_claim_ids", HERE.parents[3] / "tools" / "verity" / "lean" / "claim_ids.py")
CI = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = CI
spec.loader.exec_module(CI)


def lock(root: Path, rel: str, **sections) -> None:
    (root / rel).mkdir(parents=True)
    (root / rel / "lean-audit.json").write_text(json.dumps({"format": 2} | sections))


def test_each_carried_id_is_an_assumption_with_its_lean_names_and_a_row_no_lock_carries_stays(tmp_path):
    lock(tmp_path, "a", assumptions={"A.CR": {"id": "cr/sha-512", "owner": "@x"}},
         open={"A.Dev": {"id": "trusted-device", "owner": "@x"}, "A.Other": {"owner": "@x"}})
    lock(tmp_path, "b", open={"B.CR": {"id": "cr/sha-512"}, "A.Dev": {"id": "trusted-device"}})
    ids, fails = CI.carried(tmp_path)
    assert fails == [] and ids == {"cr/sha-512": ["A.CR", "B.CR"], "trusted-device": ["A.Dev"]}
    live = {"kind": "model", "lean": [], "name": "Live verifier"}
    old = {"live-verifier": live, "gone": {"kind": "assumption", "lean": ["A.Gone"]},
           "trusted-device": {"kind": "assumption", "lean": [], "name": "now carried"}}
    new = CI.table(old, ids)
    assert new == {"cr/sha-512": {"kind": "assumption", "lean": ["A.CR", "B.CR"]}, "live-verifier": live,
                   "trusted-device": {"kind": "assumption", "lean": ["A.Dev"]}}
    assert json.loads(CI.text(new)) == new and CI.text(new).count("\n") == len(new) + 2, "one row a line"


def test_a_lean_name_with_two_ids_and_an_id_that_is_no_claim_id_are_refused(tmp_path):
    lock(tmp_path, "a", open={"A.Bad": {"id": "Not An Id"}, "A.Dev": {"id": "trusted-device"}})
    lock(tmp_path, "b", domain={"A.Dev": {"id": "faithful", "why": "w"}})
    assert CI.carried(tmp_path)[1] == [
        "a/lean-audit.json: open[A.Bad]'s `id` 'Not An Id' is no claim id",
        "A.Dev carries `id` 'trusted-device' in a/lean-audit.json and 'faithful' in b/lean-audit.json: one Lean name, one id"]


def test_a_row_no_lock_carries_gives_its_kind_and_its_name():
    assert CI.declared_failures({"live-verifier": {"kind": "model", "lean": [], "name": "Live verifier"},
                                 "cr/sha-512": {"kind": "assumption", "lean": ["P.CR"]}}) == []
    for bad in ({"kind": "model", "lean": []}, {"kind": "axiom", "lean": [], "name": "x"},
                {"kind": "model", "lean": [], "name": "x", "label": "X"}, {"kind": "model", "lean": [], "name": " "}):
        assert [f.split(":")[0] for f in CI.declared_failures({"live-verifier": bad})] == ["live-verifier"], bad
    assert CI.declared_failures({"Live Verifier": {"kind": "model", "lean": [], "name": "x"}}) != []
