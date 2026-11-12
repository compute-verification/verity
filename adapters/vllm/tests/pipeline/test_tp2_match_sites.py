"""tp2_match STRUCTURE = site identity per k, failing CLOSED by absence -- CPU-only, torch-free
(no stored provenance steps, so `_check_main` runs the structure criterion and skips provenance).

  * positive: every recorded collective's module == the link's module_path (minus 'inner.model.' / ' (Class)') at every k, both ranks
  * NEGATIVE swap: two collectives of the same kind and width recorded in swapped order -> site mismatch at the first swapped k
  * NEGATIVE link None: a tp_links link without module_path -> mismatch at that k (None == None is never 'equal')
  * NEGATIVE record None: a recorded collective without a module name -> mismatch at that k
"""
from __future__ import annotations

import json
import types
from pathlib import Path

from verity_vllm.pipeline.tp import match as M

SITES = [("model.layers.0.self_attn.o_proj", "all_reduce", 64), ("model.layers.0.mlp.down_proj", "all_reduce", 64), ("lm_head", "all_gather", 32)]


def _links(rank: int, *, drop_k: int | None = None, v1: bool = False) -> dict:
    """A rank's tp_links.json as the derive writes it (v2: `model_root` recorded by object identity, so `link.module` is exact) or as a
    frozen v1 Build spelled it (no root: the readers fall back to dotted-suffix identity, `same_module_path`)."""
    ls = []
    for k, (site, kind, w) in enumerate(SITES):
        l = {"k": k, "kind": kind, "shape": [1, w], "module_path": f"inner.model.{site} (Cls)"}
        if k == drop_k:
            l.pop("module_path")
        ls.append(l)
    doc = {"rank": rank, "world": 2, "program_digest": f"d{rank}", "links": ls}
    if not v1:
        doc["model_root"] = "inner.model"
    return doc


def _recs(*, order=None, null_k: int | None = None, tokens: int = 5) -> list[dict]:
    order = list(order if order is not None else range(len(SITES)))
    out = []
    for step in (0, 1):
        for j, k in enumerate(order):
            site, kind, w = SITES[k]
            rows = (tokens if step == 0 else 1) if kind == "all_reduce" else 1
            out.append({"step": step, "k": j, "kind": kind, "shape": [rows, w], "module": None if (null_k == j and step == 0) else site,
                        "module_class": "Cls"})
    return out


def _run(tmp: Path, name: str, recs0, recs1, links0, links1) -> dict:
    run = tmp / name
    cap = run / "capture"
    cap.mkdir(parents=True)
    for r, recs in ((0, recs0), (1, recs1)):
        (cap / f"coll.rank{r}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in recs))
    (cap / "tokens.json").write_text(json.dumps({"requests": [{"request_id": "r0", "prompt_token_ids": [1] * 5, "generated_token_ids": [7, 8]}],
                                                 "recorder": "test"}))
    for r, links in ((0, links0), (1, links1)):
        (run / f"b{r}").mkdir()
        (run / f"b{r}" / "tp_links.json").write_text(json.dumps(links))
    M._check_main(types.SimpleNamespace(out=str(run), build_rank0=str(run / "b0"), build_rank1=str(run / "b1")))
    return json.load(open(run / "tp2_match.json"))["structure"]


def test_positive_every_k_is_a_named_equal_site_pair(tmp_path):
    s = _run(tmp_path, "pos", _recs(), _recs(), _links(0), _links(1))
    assert s["pass"] is True
    for r in ("0", "1"):
        pr = s["per_rank"][r]
        assert pr["n_steps_site_mismatching"] == 0 and pr["links_without_module_path"] == 0
        assert pr["prefill"]["sites_named_equal"] == pr["sites_checked_per_step"] == len(SITES)


def test_same_kind_same_width_order_swap_is_a_site_mismatch_at_the_first_swapped_k(tmp_path):
    s = _run(tmp_path, "swap", _recs(), _recs(order=[1, 0, 2]), _links(0), _links(1))
    assert s["pass"] is False
    assert s["per_rank"]["0"]["n_steps_site_mismatching"] == 0
    bad = s["per_rank"]["1"]["steps_site_mismatching"][0]
    assert bad["step"] == 0 and bad["first_mismatch"]["k"] == 0
    assert bad["first_mismatch"] == {"k": 0, "record": SITES[1][0], "link": SITES[0][0]}
    # the (kind, width) sequence alone would NOT have caught it
    assert s["per_rank"]["1"]["steps_mismatching"] == []


def test_link_without_module_path_fails_closed(tmp_path):
    s = _run(tmp_path, "linknone", _recs(), _recs(), _links(0), _links(1, drop_k=1))
    assert s["pass"] is False
    pr = s["per_rank"]["1"]
    assert pr["links_without_module_path"] == 1 and pr["n_steps_site_mismatching"] == 2      # every step: the link is absent at every step
    assert pr["steps_site_mismatching"][0]["first_mismatch"] == {"k": 1, "record": SITES[1][0], "link": None}
    assert pr["steps_site_mismatching"][0]["sites_named_equal"] == len(SITES) - 1


def test_record_without_module_name_fails_closed(tmp_path):
    s = _run(tmp_path, "recnone", _recs(null_k=2), _recs(), _links(0), _links(1))
    assert s["pass"] is False
    pr = s["per_rank"]["0"]
    assert pr["n_steps_site_mismatching"] == 1                                                # only step 0 carries the unnamed record
    assert pr["steps_site_mismatching"][0] == {"step": 0, "sites_named_equal": len(SITES) - 1,
                                               "first_mismatch": {"k": 2, "record": None, "link": SITES[2][0]}}
    assert s["per_rank"]["1"]["n_steps_site_mismatching"] == 0


def test_v1_links_without_a_recorded_model_root_match_by_dotted_suffix_and_still_catch_a_swap(tmp_path):
    """a frozen v1 tp_links.json spells the traced wrapper's path and records no `model_root`: the site check compares by
    dotted-suffix identity (no spelled prefix) -- the positive passes, an order swap of two same-(kind, width) sites still fails at k 0."""
    s = _run(tmp_path, "v1pos", _recs(), _recs(), _links(0, v1=True), _links(1, v1=True))
    assert s["pass"] is True and s["per_rank"]["0"]["model_root"] is None
    assert "dotted suffix" in s["per_rank"]["0"]["site_rule"]
    s = _run(tmp_path, "v1swap", _recs(), _recs(order=[1, 0, 2]), _links(0, v1=True), _links(1, v1=True))
    assert s["pass"] is False
    bad = s["per_rank"]["1"]["steps_site_mismatching"][0]
    assert bad["first_mismatch"]["k"] == 0 and bad["first_mismatch"]["record"] == SITES[1][0]
