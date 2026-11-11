"""tp-match `check` part 2 at world 8 holds every all-reduce to the routed Definition its rank Program binds (`xrank_dump.route_of` under
the switches every rank's capture log header recorded), not the `AllReduce_v2` chain: DeepSeek-V4-Pro's TP8 Match
receives the f32-accumulated sum at all 3,936 all-reduces and would fail the chain at 3,904.  Needs torch (skipped without)."""
from __future__ import annotations

import hashlib
import json
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from verity_vllm.observe.events import RunHeader  # noqa: E402
from verity_vllm.observe.log import LogWriter  # noqa: E402
from verity_vllm.pipeline.tp import match as M  # noqa: E402

WORLD, N, PROMPT = 8, 7168, 37
SITES = [("model.layers.0.attn.wo_b", "all_reduce", N), ("logits_processor", "all_gather", 4)]


def _sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def f32_once(parts: list[torch.Tensor]) -> torch.Tensor:
    return sum(p.float() for p in parts).to(torch.bfloat16)


def chain(parts: list[torch.Tensor]) -> torch.Tensor:
    acc = parts[WORLD - 1]
    for p in parts[WORLD - 2::-1]:
        acc = (acc.float() + p.float()).to(torch.bfloat16)
    return acc


def _make(run: Path, received, headers=range(WORLD)) -> dict:
    """Eight ranks, a 37-token prefill (Multimem at vLLM's defaults) and one decode step (one-shot MNNVL): wo_b's all-reduce receives
    `received(parts)` on every rank; the logits all-gather is the concatenation.  `headers`: the ranks whose capture log records the
    switches (vLLM's defaults: custom all-reduce on, no switch in the engine flags)."""
    cap = run / "capture"
    cap.mkdir(parents=True)
    g = torch.Generator().manual_seed(3)
    rows: dict[int, list] = {r: [] for r in range(WORLD)}
    for step, m in ((0, PROMPT), (1, 1)):
        vals = {r: {"part": [], "out": []} for r in range(WORLD)}
        for k, (site, kind, w) in enumerate(SITES):
            m_ = m if kind == "all_reduce" else 1
            parts = [torch.randn(m_, w, generator=g).to(torch.bfloat16) for _ in range(WORLD)]
            out = received(parts) if kind == "all_reduce" else torch.cat(parts, dim=-1)
            for r in range(WORLD):
                vals[r]["part"].append(parts[r])
                vals[r]["out"].append(out)
                rows[r].append({"rank": r, "step": step, "k": k, "kind": kind, "module": site, "module_class": "Cls", "shape": [m_, w],
                                "dtype": "bfloat16", "out_shape": list(out.shape), "part_sha256": _sha(parts[r]), "out_sha256": _sha(out)})
        for r in range(WORLD):
            torch.save(vals[r], cap / f"coll.rank{r}.step{step}.pt")
    for r in range(WORLD):
        (cap / f"coll.rank{r}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows[r]))
        if r in headers:
            with LogWriter(cap / f"log.rank{r}.jsonl.gz") as lw:
                lw.write(RunHeader(schema="test", created_utc="", profile_id="", profile={"engine_flags": {}}, model={}, workload={},
                                   notes={"tp2": {"facts": {"parallel_config": {"disable_custom_all_reduce": False}}}}))
        (run / f"b{r}").mkdir()
        (run / f"b{r}" / "tp_links.json").write_text(json.dumps({
            "rank": r, "world": WORLD, "program_digest": f"d{r}",
            "links": [{"k": k, "kind": kind, "shape": [1, w], "module_path": f"inner.model.{site} (Cls)",
                       "peers": [{"rank": p, "input": f"peer{p}", "operand": f"part_rank{p}", "collective_k": k}
                                 for p in range(WORLD) if p != r]}
                      for k, (site, kind, w) in enumerate(SITES)]}))
    request = {"request_id": "r0", "prompt_token_ids": [1] * PROMPT, "generated_token_ids": [7, 8]}
    (cap / "tokens.json").write_text(json.dumps({"requests": [request]}))
    M._check_main(types.SimpleNamespace(out=str(run), build_rank=[str(run / f"b{r}") for r in range(WORLD)], build_rank0=None, build_rank1=None))
    return json.load(open(run / "tp2_match.json"))


def test_a_world_8_all_reduce_is_held_to_its_routed_definition(tmp_path):
    doc = _make(tmp_path / "routed", f32_once)
    p = doc["provenance"]
    assert p["allreduce_settings"] == {"custom_all_reduce": True, "symmetric_memory": True, "flashinfer": True, "batch_invariant": False,
                                       "nccl_transport": None}
    assert p["all_reduce_definitions"] == {"AllReduceMultimem_v1": 1, "AllReduceMnnvl_v1": 1}
    assert p["pass"] is True and p["n_mismatches"] == 0 and p["collectives_checked"] == 4 and doc["pass"] is True
    assert p["alt_f32_once_equal"] == 2 and p["alt_ascending_equal"] < 2, "the partials tell the chain from the f32 sum"


def test_the_chain_is_not_what_a_world_8_rank_receives(tmp_path):
    p = _make(tmp_path / "chain", chain)["provenance"]
    assert p["pass"] is False
    assert [(m["step"], m["definition"].split("{")[0]) for m in p["mismatches"]] == [(0, "AllReduceMultimem_v1"), (1, "AllReduceMnnvl_v1")]
    assert p["mismatches"][0]["n_diff_rank0"] > 0


def test_a_rank_whose_capture_recorded_no_switches_fails_closed_by_name(tmp_path):
    p = _make(tmp_path / "no_header", f32_once, headers=range(WORLD - 1))["provenance"]
    assert p["pass"] is False and p["allreduce_settings"] == {"why": "no all-reduce switches in the capture log header of ranks [7]"}
    assert [m["why"] for m in p["mismatches"]] == [p["allreduce_settings"]["why"]] * 2
