"""tp2_match `check` part 2 (provenance over the stored coll.rank<r>.step<s>.pt tensors) BINDS every stored
tensor to its recorder row by digest: sha256(part_k) == row.part_sha256 and sha256(out_k) == row.out_sha256 per (rank, step, k), FAIL by
name; the record exports the recorder files' digests + the count of tensors digested.  Needs torch (skipped without)."""
from __future__ import annotations

import hashlib
import json
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from verity_vllm.pipeline.tp import match as M  # noqa: E402

SITES = [("model.embed_tokens", "all_reduce", 8), ("model.layers.0.self_attn.o_proj", "all_reduce", 8), ("logits_processor", "all_gather", 4)]


def _sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes() if t.dtype == torch.bfloat16 else t.contiguous().numpy().tobytes()).hexdigest()


def _make(run: Path, *, mutate_pt=None, mutate_row=None) -> dict:
    """Two ranks, two steps, three collectives per step; the received values ARE Def(part0, part1); rows digest the stored tensors."""
    cap = run / "capture"
    cap.mkdir(parents=True)
    g = torch.Generator().manual_seed(7)
    rows = {0: [], 1: []}
    for step in (0, 1):
        vals = {0: {"part": [], "out": []}, 1: {"part": [], "out": []}}
        for k, (site, kind, w) in enumerate(SITES):
            M_ = (5 if step == 0 else 1) if kind == "all_reduce" else 1
            a, b = (torch.randn(M_, w, generator=g).to(torch.bfloat16) for _ in range(2))
            out = (a.float() + b.float()).to(torch.bfloat16) if kind == "all_reduce" else torch.cat([a, b], dim=-1)
            for r, part in ((0, a), (1, b)):
                vals[r]["part"].append(part)
                vals[r]["out"].append(out)
                rows[r].append({"rank": r, "step": step, "k": k, "kind": kind, "module": site, "module_class": "Cls", "shape": list(part.shape),
                                "dtype": "bfloat16", "out_shape": list(out.shape), "part_sha256": _sha(part), "out_sha256": _sha(out)})
        for r in (0, 1):
            if mutate_pt:
                mutate_pt(r, step, vals[r])
            torch.save(vals[r], cap / f"coll.rank{r}.step{step}.pt")
    for r in (0, 1):
        if mutate_row:
            rows[r] = [mutate_row(r, x) for x in rows[r]]
        (cap / f"coll.rank{r}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows[r]))
    (cap / "tokens.json").write_text(json.dumps({"requests": [{"request_id": "r0", "prompt_token_ids": [1] * 5, "generated_token_ids": [7, 8]}], "recorder": "test"}))
    for r in (0, 1):
        (run / f"b{r}").mkdir()
        (run / f"b{r}" / "tp_links.json").write_text(json.dumps({"rank": r, "world": 2, "program_digest": f"d{r}",
                                                                 "links": [{"k": k, "kind": kind, "shape": [1, w], "module_path": f"inner.model.{site} (Cls)"}
                                                                           for k, (site, kind, w) in enumerate(SITES)]}))
    M._check_main(types.SimpleNamespace(out=str(run), build_rank0=str(run / "b0"), build_rank1=str(run / "b1")))
    return json.load(open(run / "tp2_match.json"))


def test_provenance_binds_every_stored_tensor_to_its_recorder_row(tmp_path):
    doc = _make(tmp_path / "pos")
    p = doc["provenance"]
    assert p["pass"] is True and p["collectives_checked"] == 6 and p["n_mismatches"] == 0
    assert p["tensors_digested"] == 4 * 6 and p["n_digest_mismatches"] == 0
    cap = tmp_path / "pos" / "capture"
    assert p["recorder_sha256"] == {str(r): hashlib.sha256((cap / f"coll.rank{r}.jsonl").read_bytes()).hexdigest() for r in (0, 1)}
    assert doc["pass"] is True


def test_stored_part_bytes_not_the_recorder_digest_fail_by_name(tmp_path):
    """The arithmetic still holds on the stored bytes (part flipped AND received recomputed from it), but the jsonl digest describes other
    bytes: FAIL named at (rank 1, step 1, k 0, part) and out."""
    def flip(r, step, vals):
        if r == 1 and step == 1:
            vals["part"][0] = vals["part"][0].clone()
            vals["part"][0].view(torch.int16)[0, 0] ^= 1
    doc = _make(tmp_path / "neg_pt", mutate_pt=flip)
    p = doc["provenance"]
    assert p["pass"] is False and doc["pass"] is False
    dm = p["digest_mismatches"]
    assert dm and dm[0]["rank"] == 1 and dm[0]["step"] == 1 and dm[0]["k"] == 0 and dm[0]["tensor"] == "part"
    assert "stored .pt part sha256" in dm[0]["why"] and "!= recorder row" in dm[0]["why"]
    assert p["n_digest_mismatches"] == 1                                # the out tensors still digest to their rows


def test_recorder_row_digest_edited_fails_by_name(tmp_path):
    def edit(r, x):
        return dict(x, out_sha256=("0" * 64 if (r == 0 and x["step"] == 0 and x["k"] == 2) else x["out_sha256"]))
    doc = _make(tmp_path / "neg_row", mutate_row=edit)
    p = doc["provenance"]
    assert p["pass"] is False
    assert [(d["rank"], d["step"], d["k"], d["tensor"]) for d in p["digest_mismatches"]] == [(0, 0, 2, "out")]
