"""quarantine_lint.py (protected): purity + identity rules on synthetic quarantine modules."""
from __future__ import annotations

from verity_vllm.properties import quarantine_lint

GOOD = '''
from verity.circuits.defs import CompositeDefinition
from verity.circuits.types import Array
SEMANTIC_ID = "gen-x/Gelu@v1"
Gelu = CompositeDefinition("Gelu", 1, ("N",), lambda S: ((("x", Array(S.N, None)),), Array(S.N, None)), lambda B, S, x: x, register=False)
'''
BAD = '''
import os, time
from verity.circuits.defs import CompositeDefinition
Gelu = CompositeDefinition("Gelu", 0, ("N",), lambda S: (), lambda B, S, x: x)
def f():
    global Gelu
    return open("x")
'''
ADAPTER_OK = '''
SEMANTIC_ID = "gen-x/Gelu@v1"
def cases(rng, i): return {}
def production(statics, inputs):
    import torch
    return inputs
def spec(statics): return None
'''
ADAPTER_BAD = '''
import torch
SEMANTIC_ID = "gen-x/Gelu@v1"
'''


def test_good_and_bad(tmp_path):
    root = tmp_path / "q"
    (root / "gen_x").mkdir(parents=True)
    (root / "gen_x" / "gelu.py").write_text(GOOD)
    (root / "gen_x" / "bad.py").write_text(BAD)
    (root / "gen_x" / "gelu_difftest.py").write_text(ADAPTER_OK)
    (root / "gen_x" / "bad_difftest.py").write_text(ADAPTER_BAD)
    rep = quarantine_lint.lint(root)
    by = {r["file"].split("/")[-1]: r for r in rep["files"]}
    assert by["gelu.py"]["status"] == "pass" and by["gelu.py"]["semantic_id"] == "gen-x/Gelu@v1"
    assert by["gelu_difftest.py"]["status"] == "pass"
    bad = by["bad.py"]
    assert bad["status"] == "fail"
    joined = " ".join(bad["problems"])
    assert "forbidden import 'os'" in joined and "version must be" in joined and "global statement" in joined and "open()" in joined and "SEMANTIC_ID" in joined
    assert by["bad_difftest.py"]["status"] == "fail" and "torch" in " ".join(by["bad_difftest.py"]["problems"])
    assert rep["status"] == "fail"


def test_empty_registry_passes():
    rep = quarantine_lint.lint()
    assert rep["status"] == "pass"


def test_an_identity_of_another_form_or_held_twice_fails(tmp_path):
    """`SEMANTIC_ID` is `<area>/<Name>@v<k>`, and no two definitions' modules hold the same one; an adapter holds its
    definition's."""
    root = tmp_path / "q"
    (root / "gen_x").mkdir(parents=True)
    (root / "gen_x" / "gelu.py").write_text(GOOD)
    (root / "gen_x" / "gelu_difftest.py").write_text(ADAPTER_OK)
    (root / "__init__.py").write_text(BAD)
    rep = quarantine_lint.lint(root)
    assert set(rep) == {"schema", "root", "n_files", "status", "duplicate_semantic_ids", "files", "ok", "digest"}
    assert (rep["schema"], rep["root"], rep["status"], rep["n_files"]) == ("verity-gen/quarantine-lint/v1", str(root), "pass", 2)
    (root / "gen_x" / "gelu_again.py").write_text(GOOD)
    (root / "gen_x" / "unversioned.py").write_text(GOOD.replace("gen-x/Gelu@v1", "gen-x/Gelu"))
    rep = quarantine_lint.lint(root)
    by = {r["file"].split("/")[-1]: r for r in rep["files"]}
    assert [by[n]["status"] for n in ("gelu.py", "gelu_again.py", "gelu_difftest.py")] == ["pass"] * 3
    assert by["unversioned.py"]["problems"] == ["SEMANTIC_ID 'gen-x/Gelu' does not match <area>/<Name>@v<k>"]
    assert (rep["status"], rep["ok"], rep["n_files"], rep["duplicate_semantic_ids"]) == ("fail", False, 4, ["gen-x/Gelu@v1"])
    (root / "gen_x" / "unversioned.py").unlink()
    rep = quarantine_lint.lint(root)
    assert (rep["status"], rep["duplicate_semantic_ids"]) == ("fail", ["gen-x/Gelu@v1"])


def test_the_roots_own_init_is_not_read_and_a_packages_is(tmp_path):
    root = tmp_path / "q"
    (root / "gen_x").mkdir(parents=True)
    (root / "__init__.py").write_text(BAD)
    (root / "gen_x" / "gelu.py").write_text(GOOD)
    assert [r["file"] for r in quarantine_lint.lint(root)["files"]] == [str(root / "gen_x" / "gelu.py")]
    (root / "gen_x" / "__init__.py").write_text(BAD)
    rep = quarantine_lint.lint(root)
    assert [(r["file"], r["status"]) for r in rep["files"]] == [(str(root / "gen_x" / "__init__.py"), "fail"),
                                                                (str(root / "gen_x" / "gelu.py"), "pass")]
    assert rep["status"] == "fail"
