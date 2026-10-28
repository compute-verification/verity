"""compare_steps' divergence report is bounded: prefix/suffix in O(n), the matcher over a windowed core."""
from __future__ import annotations

from verity_vllm.check.match import program_compare as C


def test_divergence_report_locates_the_first_divergence_in_full_coordinates():
    a = ["E", "N", "G", "R", "A", "G", "S", "T"]
    b = ["E", "N", "G", "R", "B", "G", "S", "T"]
    d = C._divergence_report(a, b)
    assert d["ops"] == [("replace", 4, 5, 4, 5)]
    assert "window" not in d


def test_divergence_report_equal_sequences_have_no_ops():
    assert C._divergence_report(["x", "y"], ["x", "y"]) == {"ops": []}


def test_divergence_report_insert_and_delete_ranges_are_shifted_by_the_common_prefix():
    a = ["p", "p", "x", "y", "s"]
    b = ["p", "p", "x", "s"]
    d = C._divergence_report(a, b)
    assert d["ops"] == [("delete", 3, 4, 3, 3)]
    d2 = C._divergence_report(b, a)
    assert d2["ops"] == [("insert", 3, 3, 3, 4)]


def test_divergence_report_is_bounded_on_a_prefill_sized_step_and_says_so_by_name(monkeypatch):
    # the #60 shape: one step of LP x layers x ops instances where two kernel families differ throughout (Build for another target)
    row = ["RMSNorm", "Gemm_v2", "RoPE", "Attention_v2", "Gemm_v2", "RMSNorm", "Gemm_v2", "SiluMul", "Gemm_v2"]
    a = ["Embedding"] * 1024 + row * (1024 * 32) + ["TokenSelect"]
    b = [x.replace("Gemm_v2", "Gemm_v1").replace("Attention_v2", "Attention_v3") for x in a]
    assert len(a) > 290_000
    d = C._divergence_report(a, b)   # the record's whole-step matcher takes hours here
    w = d["window"]
    # prefix = 1024 Embedding + the first row's RMSNorm; suffix = the TokenSelect
    assert w["truncated"] is True and w["common_prefix"] == 1025 and w["common_suffix"] == 1 and w["window"] == C._DIVERGENCE_WINDOW
    assert w["core_a"] == w["core_b"] == len(a) - 1026
    # the first divergence is the first Gemm of the first row, in FULL-sequence coordinates
    tag, i1, i2, j1, j2 = d["ops"][0]
    assert tag == "replace" and i1 == 1025 and a[i1] == "Gemm_v2" and b[j1] == "Gemm_v1"
    assert all(i1 >= 1025 for _t, i1, _i2, _j1, _j2 in d["ops"])


def test_compare_steps_carries_the_window_only_when_truncated(monkeypatch):
    monkeypatch.setattr(C, "_DIVERGENCE_WINDOW", 4)

    class P:   # the minimum compare_steps reads
        def __init__(self, label, specs):
            self.label, self.digest, self.inputs = label, label, {}
            self.inst = [{"spec": s, "form": "f", "args": [], "out": "o"} for s in specs]
            self.steps = [list(range(len(specs)))]
            self.splits_values, self.splits_nodes = {}, set()

        def is_pad(self, i):
            return False

        def canon(self, i, unfold=False):
            return []

    monkeypatch.setattr(C, "_canonical_hash", lambda P, seqs, unfold: P.label)
    a = P("a", ["e", "x1", "x2", "x3", "x4", "x5", "x6", "t"])
    b = P("b", ["e", "y1", "y2", "y3", "y4", "y5", "y6", "t"])
    st = C.compare_steps(a, b)["steps"][0]
    assert st["spec_seq_equal"] is False and st["divergence_window"]["truncated"] is True and st["divergence_window"]["core_a"] == 6
    assert st["first_divergences_T_normalised"][0]["a_range"] == [1, 5]
    c = P("c", ["e", "x1", "x2", "t"])
    d = P("d", ["e", "y1", "y2", "t"])
    st2 = C.compare_steps(c, d)["steps"][0]
    assert "divergence_window" not in st2 and st2["n_divergent_opcodes"] == 1 and st2["first_divergences_T_normalised"][0]["a_range"] == [1, 3]
