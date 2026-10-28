"""`CompiledKernelCheck.run`'s per-launch replay and dataflow linkage, pinned on a two-piece, two-step stand-in: an exact launch, a
mismatched one (its diagnostics), one with a committed role missing, one whose relation raises, an RMSNorm launch whose reduction
order misses the default labels, is found by the full search and is then pinned for the next step, and the declared edges.  CPU-only,
no engine: the check is built without a model, the committed tensors come through a stub and the relation is a stand-in."""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import torch

from verity_vllm.check.replay import compiled_kernel_check as CKC

NORM = "triton_red_fused_rms_norm_q"
ROLES = {0: "x", 1: "out"}


def _x(s: int) -> torch.Tensor:
    return torch.arange(12, dtype=torch.float32).reshape(4, 3) + s


def _committed() -> dict:
    out = {}
    for s in (0, 1):
        x = _x(s)
        bad = x * 2
        bad[2] += 1
        out.update({(s, "c00.k_add/x"): x, (s, "c00.k_add/out"): x + 1, (s, "c00.k_bad/x"): x, (s, "c00.k_bad/out"): bad,
                    (s, "c01.k_missing/x"): x, (s, "c01.k_raise/x"): x, (s, "c01.k_raise/out"): x,
                    (s, f"c01.{NORM}/x"): x + 1, (s, f"c01.{NORM}/out"): x + 2})
    return out


def _relation(kernel: str, ins: list, outs_meta: list, eps: float = 1e-5, labels=None) -> dict:
    x = ins[0]
    if kernel == "k_raise":
        raise ValueError("no relation for k_raise")
    if "rms_norm" in kernel:
        orders = labels if labels is not None else ("order-a", "order-b")
        return {"kind": "norm", "variants": {lb: [x + 1 if lb == "order-b" else x] for lb in orders}}
    return {"kind": "add" if kernel == "k_add" else "dbl", "outputs": [x + 1 if kernel == "k_add" else x * 2]}


def _manifest() -> dict:
    first = [{"kernel": "k_add"}, {"kernel": "k_bad"}]
    last = [{"kernel": "k_missing"}, {"kernel": "k_raise"}, {"kernel": NORM}]
    edges = [[{"kernel": "k_add", "edges": [{"role": "x", "src": {"kind": "external"}}]}],
             [{"kernel": NORM, "edges": [{"role": "x", "src": {"kind": "launch", "piece": 0, "kernel": "k_add", "role": "out"}}]},
              {"kernel": "k_missing", "edges": [{"role": "x", "src": {"kind": "launch", "piece": 0, "kernel": "k_bad", "role": "out"}},
                                                {"role": "y", "src": {"kind": "launch", "piece": 0, "kernel": "k_add", "role": "out"}}]}]]
    return {"program": {"launch_table": {"first": first, "middle": [], "last": last},
                        "role_maps": {"first": [{"kernel": "k_add", "roles": {"0": "x", "1": "out"}, "weight_args": []}], "middle": [],
                                      "last": []},
                        "dataflow": {"edges": edges, "n_edges": 4, "n_launch_edges": 3, "wiring_declared": True}}}


def _run(monkeypatch) -> tuple[dict, list[str], list]:
    norm_labels: list = []

    def relation(kernel: str, ins: list, outs_meta: list, eps: float = 1e-5, labels=None) -> dict:
        if kernel == NORM:
            norm_labels.append(labels)
        return _relation(kernel, ins, outs_meta, eps, labels)

    monkeypatch.setattr(CKC, "relation", relation)
    monkeypatch.setattr(CKC, "roles_of", lambda kernel: (dict(ROLES), set()))
    committed = _committed()
    autotuned: list[str] = []
    ck = CKC.CompiledKernelCheck.__new__(CKC.CompiledKernelCheck)
    ck.c = SimpleNamespace(_layouts={0: [], 1: []})
    ck.seed, ck.seed_source, ck.n_random, ck.pieces = 7, "test", 0, 2
    ck.eps, ck.eps_source = 1e-5, "test"
    ck.tokens_by_req, ck.prompts, ck.mods, ck._norm_labels = None, {}, {}, {}
    ck.committed = lambda s, metas, name: committed.get((s, name))                                   # type: ignore[method-assign]
    ck._weights = lambda piece, kernel, wargs=None: {}                                                # type: ignore[method-assign]
    ck._embed_kernel = lambda metas: "k_add"                                                          # type: ignore[method-assign]
    ck._autotune_equivalence = lambda kernel, ins, outs, roles, res: autotuned.append(kernel)         # type: ignore[method-assign]
    res = ck.run(_manifest())
    res.pop("seconds")
    return res, autotuned, norm_labels


def test_run_record_is_pinned(monkeypatch) -> None:
    res, autotuned, norm_labels = _run(monkeypatch)
    assert autotuned == ["k_add", "k_bad", NORM] * 2
    assert norm_labels == [CKC.NORM_LABELS, None, ("order-b",)]
    assert {k: res[k] for k in ("launches_checked", "launches_exact", "launches_missing", "outputs_checked", "outputs_exact")} == {
        "launches_checked": 8, "launches_exact": 4, "launches_missing": 2, "outputs_checked": 6, "outputs_exact": 4}
    assert res["by_kind"] == {
        "add": {"launches": 2, "exact": 2, "outputs": 2, "outputs_exact": 2, "variants_exact": {"single": 2}},
        "dbl": {"launches": 2, "exact": 0, "outputs": 2, "outputs_exact": 0, "variants_exact": {}},
        "norm": {"launches": 2, "exact": 2, "outputs": 2, "outputs_exact": 2, "variants_exact": {"order-b": 2}}}
    assert [(m["step"], m["kernel"], m["outputs_ok"], m["diag"][0]["rows_first_last"]) for m in res["mismatch"]] == [
        (0, "k_bad", [False], [2, 2]), (1, "k_bad", [False], [2, 2])]
    assert res["missing"] == [{"step": s, "piece": 1, "kernel": "k_missing", "roles": ["out"]} for s in (0, 1)]
    assert res["errors"] == [f"step {s} piece 1 k_raise: ValueError('no relation for k_raise')" for s in (0, 1)]
    assert res["norm_orders"] == {f"{NORM}@3": ["order-b"]}
    assert {k: res["edges"][k] for k in ("checked", "equal", "missing", "external", "steps", "ok")} == {
        "checked": 4, "equal": 2, "missing": 2, "external": 2, "steps": 2, "ok": False}
    assert res["ok"] is False
    assert hashlib.sha256(json.dumps(res).encode()).hexdigest() == RECORD_SHA256


#: SHA-256 of the whole record as `json.dumps` writes it (key order included), "seconds" removed.
RECORD_SHA256 = "98b9fba9c7f1d7658c874d6904bd9aa0f55079015c736d1a6de87fab45c6795f"
