"""`tools/migrate/split_attention.py` on a synthetic tree: the move, every import form it rewrites, its idempotence (a rerun is the
rebase) and what it refuses.  The real tree's invariants (ids, digests, circuit-check's tasks) are
`tools/verity/circuit_check/tests/test_attention_split.py`'s."""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "migrate" / "split_attention.py"
_spec = importlib.util.spec_from_file_location("split_attention", SCRIPT)
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)

FA2 = '''"""FA2 per-iteration Check_inf: `AttentionHead_v5` takes `MASKED_FROM` (`fa2_masked_from`)."""
from __future__ import annotations

from verity.ml.circuits.defs import composite

#: FA2's splitkv tile height
KBLOCK_M = 64


def fa2_masked_from(seqlen_k: int, seqlen_q: int, row: int, BN: int, causal: bool) -> int:
    """The lowest guarded key block."""
    nb = -(-seqlen_k // BN)
    return max(0, min(nb, (row // KBLOCK_M + 1) * KBLOCK_M // BN) - 2) if causal else nb - 1


def AttnBlockV5(B, S):
    return composite


__all__ = ["KBLOCK_M", "fa2_masked_from", "AttnBlockV5"]
'''

FA3 = '''"""FA3 per-iteration Check_inf."""
from __future__ import annotations


def fa3_masked_from(T: int, BN: int, row: int, kBM: int, g: int) -> int:
    m_idx_min = (row * g // kBM) * kBM // g
    return (m_idx_min + T - 1 - row) // BN


def AttnBlockV4(B, S):
    return None


__all__ = ["fa3_masked_from", "AttnBlockV4"]
'''

CONSUMERS = {
    # a from-import of a moved and a kept name
    "integrations/vllm/verity_vllm/a.py": (
        "from verity_vllm.program.registry.fa2_check_inf import AttnBlockV5, fa2_masked_from\n\n"
        "X = (AttnBlockV5, fa2_masked_from(1, 1, 0, 64, False))\n"),
    # one statement binding both modules, both still used for their Definitions
    "integrations/vllm/verity_vllm/b.py": (
        "from verity_vllm.program.registry import fa2_check_inf as F2, fa3_check_inf as F3\n\n"
        "X = (F2.AttnBlockV5, F3.AttnBlockV4, F2.fa2_masked_from(1, 1, 0, 64, False), F3.fa3_masked_from(1, 128, 0, 64, 4))\n"),
    # a function-local alias used only for the moved name: the import is replaced
    "integrations/vllm/tests/c.py": (
        "import os\n\n\n"
        "def f():\n"
        "    from verity_vllm.program.registry import fa3_check_inf as F3\n"
        "    return F3.fa3_masked_from(1, 128, 0, 64, 4), os.sep\n"),
    # a relative import from inside the registry, and a docstring mention with a dotted prefix
    "integrations/vllm/verity_vllm/program/registry/targets.py": (
        '"""Reads `registry.fa2_check_inf.fa2_masked_from` and `fa3_check_inf.fa3_masked_from`."""\n'
        "from . import fa2_check_inf as F2\n\n"
        "Y = F2.KBLOCK_M\n"),
    # a plain import of the module, read through its full name
    "tools/d.py": (
        "import verity_vllm.program.registry.fa3_check_inf\n\n"
        "Z = verity_vllm.program.registry.fa3_check_inf.fa3_masked_from(1, 128, 0, 64, 4)\n"),
    "docs/notes.md": "The rule is `fa2_check_inf.fa2_masked_from`; `fa2_check_inf.AttnBlockV5` stays.\n",
}


def _tree(tmp_path: Path, extra: dict[str, str] | None = None) -> Path:
    reg = tmp_path / S.REGISTRY
    reg.mkdir(parents=True)
    (reg / "fa2_check_inf.py").write_text(FA2)
    (reg / "fa3_check_inf.py").write_text(FA3)
    for rel, text in {**CONSUMERS, **(extra or {})}.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text)
    return tmp_path


@pytest.fixture
def files(monkeypatch):
    monkeypatch.setattr(S, "tracked_files", lambda repo: sorted(p for p in repo.rglob("*") if p.is_file()))


def _apply(repo: Path) -> dict[Path, str]:
    writes, _log = S.plan(repo)
    for p, text in writes.items():
        p.write_text(text)
    return writes


def _read(repo: Path, rel: str) -> str:
    return (repo / rel).read_text()


def test_the_move_and_every_import_form(tmp_path, files):
    repo = _tree(tmp_path)
    _apply(repo)
    reg = S.REGISTRY
    vocab = _read(repo, f"{reg}/attention_vocab.py")
    ns: dict = {}
    exec(compile(vocab, "attention_vocab.py", "exec"), ns)
    assert ns["__all__"] == ["KBLOCK_M", "fa2_masked_from", "fa3_masked_from"]
    assert ns["fa2_masked_from"](130, 130, 129, 64, True) == 1 and ns["fa3_masked_from"](130, 128, 129, 128, 4) == 1
    assert "#: FA2's splitkv tile height\nKBLOCK_M = 64" in vocab

    fa2 = _read(repo, f"{reg}/fa2_check_inf.py")
    assert "def fa2_masked_from" not in fa2 and "KBLOCK_M" not in fa2.replace("attention_vocab", "")
    assert '__all__ = ["AttnBlockV5"]' in fa2 and "(`attention_vocab.fa2_masked_from`)" in fa2
    assert "from verity.ml.circuits.defs import composite\n\n\ndef AttnBlockV5" in fa2         # two blank lines where the rules were

    assert _read(repo, "integrations/vllm/verity_vllm/a.py").startswith(
        "from verity_vllm.program.registry.fa2_check_inf import AttnBlockV5\n"
        "from verity_vllm.program.registry.attention_vocab import fa2_masked_from\n")
    b = _read(repo, "integrations/vllm/verity_vllm/b.py")
    assert b.count("import attention_vocab") == 1 and "F2.AttnBlockV5, F3.AttnBlockV4" in b
    assert "attention_vocab.fa2_masked_from(1" in b and "attention_vocab.fa3_masked_from(1" in b
    c = _read(repo, "integrations/vllm/tests/c.py")
    assert "    from verity_vllm.program.registry import attention_vocab\n    return attention_vocab.fa3_masked_from(" in c
    assert "fa3_check_inf" not in c
    t = _read(repo, f"{reg}/targets.py")
    assert "from . import fa2_check_inf as F2\nfrom . import attention_vocab\n" not in t      # F2 is unused after the rewrite: dropped
    assert "from . import attention_vocab\n\nY = attention_vocab.KBLOCK_M" in t
    assert "`registry.attention_vocab.fa2_masked_from` and `attention_vocab.fa3_masked_from`" in t
    d = _read(repo, "tools/d.py")
    assert "Z = attention_vocab.fa3_masked_from(" in d and "import verity_vllm" not in d
    assert _read(repo, "docs/notes.md") == "The rule is `attention_vocab.fa2_masked_from`; `fa2_check_inf.AttnBlockV5` stays.\n"
    for p in repo.rglob("*.py"):
        ast.parse(p.read_text())
    assert S.leftovers(repo, sorted(repo.rglob("*.py")), {}) == []


def test_a_rerun_is_a_no_op_and_the_check_passes(tmp_path, files):
    repo = _tree(tmp_path)
    assert S.main(["--repo", str(repo), "--check"]) == 1
    assert S.main(["--repo", str(repo), "--no-verify"]) == 0
    before = {p: p.read_text() for p in repo.rglob("*") if p.is_file()}
    assert S.plan(repo)[0] == {}
    assert S.main(["--repo", str(repo), "--check"]) == 0
    assert S.main(["--repo", str(repo), "--no-verify"]) == 0
    assert {p: p.read_text() for p in repo.rglob("*") if p.is_file()} == before


def test_a_later_main_with_a_new_consumer_is_rewritten(tmp_path, files):
    repo = _tree(tmp_path)
    _apply(repo)
    new = repo / "integrations/vllm/verity_vllm/e.py"
    new.write_text("from verity_vllm.program.registry.fa3_check_inf import fa3_masked_from as mf\n\nW = mf\n")
    writes = _apply(repo)
    assert list(writes) == [new]
    assert new.read_text().startswith("from verity_vllm.program.registry.attention_vocab import fa3_masked_from as mf\n")


@pytest.mark.parametrize("rel,text,match", [
    ("x.py", "from verity_vllm.program.registry.fa2_check_inf import *\n", "import \\*"),
    ("x.py", "from verity_vllm.program.registry import fa2_check_inf as F2\nf = getattr(F2, 'fa2_masked_from')\n", "by name"),
    ("x.py", "from verity_vllm.program.registry import fa2_check_inf\ndef (:\n", "does not parse"),
])
def test_what_it_cannot_rewrite_fails_loudly(tmp_path, files, rel, text, match):
    repo = _tree(tmp_path, {rel: text})
    before = {p: p.read_text() for p in repo.rglob("*") if p.is_file()}
    with pytest.raises(S.SplitError, match=match):
        S.plan(repo)
    assert S.main(["--repo", str(repo), "--no-verify"]) == 1
    assert {p: p.read_text() for p in repo.rglob("*") if p.is_file()} == before


def test_a_partial_split_is_refused(tmp_path, files):
    repo = _tree(tmp_path)
    (repo / S.REGISTRY / "attention_vocab.py").write_text('def fa3_masked_from():\n    pass\n\n\n__all__ = ["fa3_masked_from"]\n')
    with pytest.raises(S.SplitError, match="partial split"):
        S.plan(repo)


def test_a_moved_rule_reading_a_kept_name_is_refused(tmp_path, files):
    repo = _tree(tmp_path)
    p = repo / S.REGISTRY / "fa3_check_inf.py"
    p.write_text(p.read_text().replace("return (m_idx_min", "return AttnBlockV4 and (m_idx_min"))
    with pytest.raises(S.SplitError, match="AttnBlockV4, which fa3_check_inf keeps"):
        S.plan(repo)


def test_a_missing_module_is_refused(tmp_path, files):
    repo = _tree(tmp_path)
    (repo / S.REGISTRY / "fa3_check_inf.py").unlink()
    with pytest.raises(S.SplitError, match="missing"):
        S.plan(repo)
