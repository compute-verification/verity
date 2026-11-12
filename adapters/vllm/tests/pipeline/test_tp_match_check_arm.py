"""`verity_vllm.pipeline.tp.match`'s check arm where its other tests do not go: the tokens part against a control arm, a recorder
file that is missing, a rank Build of another rank or world, with a link that names the wrong number of peers or with no links
at all, a capture that names no first forward for a request, a step whose collectives are not the Build's signature, stored
tensors that the recorder's rows do not count or key, a command line that does not give two rank Builds in one spelling, and
what the check says in its log.  Then the Match record a TP Commit reads from the run (`match_record`)."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest
import torch

from verity_vllm.config import UsageError
from verity_vllm.errors import Refused
from verity_vllm.pipeline import cli
from verity_vllm.pipeline.tp import match as M

SITES = [("model.embed_tokens", "all_reduce", 8), ("model.layers.0.self_attn.o_proj", "all_reduce", 8), ("logits_processor", "all_gather", 4)]
PROMPT, GENERATED = [1] * 5, [7, 8]


def _sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def _tokens(generated: list[int], run_s: float) -> str:
    return json.dumps({"requests": [{"request_id": "r0", "prompt_token_ids": PROMPT, "generated_token_ids": generated}], "run_s": run_s})


def _capture(run: Path, *, rows=None, links=None, control: list[int] | None = None, stored: bool = True) -> Path:
    """A two-rank capture under ``run``: two steps of three collectives, each received value the Definition of the two ranks' parts, each
    recorder row the digests of its stored tensors, and each rank's Build links.  ``rows(rank, rows)`` and ``links(rank, doc)`` edit a
    rank's recorder rows and its Build's links; ``control`` is what a control arm generated; ``stored`` False stores no tensors."""
    capture = run / "capture"
    capture.mkdir(parents=True)
    g = torch.Generator().manual_seed(7)
    recorded: dict[int, list[dict]] = {0: [], 1: []}
    for step in (0, 1):
        vals: dict[int, dict[str, list]] = {0: {"part": [], "out": []}, 1: {"part": [], "out": []}}
        for k, (site, kind, w) in enumerate(SITES):
            m = (len(PROMPT) if step == 0 else 1) if kind == "all_reduce" else 1
            a, b = (torch.randn(m, w, generator=g).to(torch.bfloat16) for _ in range(2))
            out = (a.float() + b.float()).to(torch.bfloat16) if kind == "all_reduce" else torch.cat([a, b], dim=-1)
            for r, part in ((0, a), (1, b)):
                vals[r]["part"].append(part)
                vals[r]["out"].append(out)
                recorded[r].append({"rank": r, "step": step, "k": k, "kind": kind, "module": site, "module_class": "Cls",
                                    "shape": list(part.shape), "part_sha256": _sha(part), "out_sha256": _sha(out)})
        for r in (0, 1):
            if stored:
                torch.save(vals[r], capture / f"coll.rank{r}.step{step}.pt")
    for r in (0, 1):
        mine = rows(r, recorded[r]) if rows else recorded[r]
        (capture / f"coll.rank{r}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in mine))
        doc = {"rank": r, "world": 2, "program_digest": f"d{r}",
               "links": [{"k": k, "kind": kind, "shape": [1, w], "module_path": f"inner.model.{site} (Cls)"} for k, (site, kind, w) in enumerate(SITES)]}
        (run / f"b{r}").mkdir()
        (run / f"b{r}" / "tp_links.json").write_text(json.dumps(links(r, doc) if links else doc))
    (capture / "tokens.json").write_text(_tokens(GENERATED, 1.5))
    if control is not None:
        (run / "control").mkdir()
        (run / "control" / "tokens.json").write_text(_tokens(control, 2.5))
    return run


def _check(run: Path) -> tuple[int, dict]:
    """The check arm over ``run`` -> (its exit code, the record it wrote)."""
    rc = M.main(M.Options(arm="check", out=str(run), build_rank0=str(run / "b0"), build_rank1=str(run / "b1")))
    return rc, json.loads((run / "tp2_match.json").read_text())


def test_the_tokens_part_passes_when_the_control_arm_generated_the_same_tokens(tmp_path):
    rc, doc = _check(_capture(tmp_path / "run", control=GENERATED))
    assert doc["tokens"] == {"pass": True, "n_requests": 1, "generated_lengths": [2], "capture_run_s": 1.5, "control_run_s": 2.5}
    assert rc == 0 and doc["pass"] is True


def test_a_control_arm_that_generated_other_tokens_fails_the_check_alone(tmp_path):
    rc, doc = _check(_capture(tmp_path / "run", control=[7, 9]))
    assert doc["tokens"]["pass"] is False and doc["structure"]["pass"] is True and doc["provenance"]["pass"] is True
    assert rc == 1 and doc["pass"] is False


@pytest.mark.parametrize("control, verdict", [(GENERATED, True), ([7, 9], False), (None, True)], ids=["equal tokens", "other tokens", "no control"])
def test_the_check_prints_its_verdict_and_its_three_parts(tmp_path, capsys, control, verdict):
    rc, doc = _check(_capture(tmp_path / "run", control=control))
    said = json.dumps({k: doc[k] for k in ("pass", "structure", "provenance", "tokens")}, indent=1)
    assert capsys.readouterr().out == said + "\n" and len(said) < 6000 and doc["pass"] is verdict and rc == (0 if verdict else 1)


def test_what_the_check_prints_ends_at_6000_characters(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(M, "_tokens", lambda control, tok: {"pass": True, "note": "n" * 7000})
    rc, doc = _check(_capture(tmp_path / "run"))
    said = capsys.readouterr().out
    assert said == json.dumps({k: doc[k] for k in ("pass", "structure", "provenance", "tokens")}, indent=1)[:6000] + "\n" and len(said) == 6001


def test_without_a_control_arm_the_tokens_part_is_not_judged(tmp_path):
    rc, doc = _check(_capture(tmp_path / "run"))
    assert doc["tokens"] == {"pass": None, "note": "no control arm"}
    assert rc == 0 and doc["pass"] is True


def test_a_missing_recorder_file_fails_by_name_before_any_part_is_checked(tmp_path):
    run = _capture(tmp_path / "run")
    (run / "capture" / "coll.rank1.jsonl").unlink()
    rc, doc = _check(run)
    assert rc == 1 and doc["fail"] == f"missing {run / 'capture' / 'coll.rank1.jsonl'}" and doc["world"] == 2
    assert not {"structure", "provenance", "tokens", "pass"} & set(doc)


@pytest.mark.parametrize("edit, said", [({"rank": 1}, "rank 1 world 2 != rank 0 of world 2"), ({"world": 4}, "rank 0 world 4 != rank 0 of world 2")])
def test_a_build_of_another_rank_or_world_is_refused(tmp_path, edit, said):
    run = _capture(tmp_path / "run", links=lambda r, doc: {**doc, **edit} if r == 0 else doc)
    with pytest.raises(Refused, match=said):
        _check(run)
    assert not (run / "tp2_match.json").exists()


@pytest.mark.parametrize("peers", [[], [{"rank": 0}, {"rank": 2}]])
def test_a_link_that_does_not_name_one_peer_per_other_rank_is_refused(tmp_path, peers):
    def edit(r: int, doc: dict) -> dict:
        if r == 1:
            doc["links"][0]["peers"] = peers
        return doc

    run = _capture(tmp_path / "run", links=edit)
    with pytest.raises(Refused, match="tp_links of rank 1: a link does not name exactly world-1 = 1 peers"):
        _check(run)


def _request(first_step, *, listed: bool = True, timing: bool = True, requests: bool = True) -> dict:
    """A capture's tokens of one request (five prompt tokens, two generated) whose prefill the capture's timing puts at ``first_step``."""
    tok = json.loads(_tokens(GENERATED, 1.5))
    if timing:
        tok["timing"] = {"requests": {"r0" if listed else "other": {"first_step": first_step}}} if requests else {"engine_steps": 2}
    return tok


@pytest.mark.parametrize("tok", [_request(0), _request(None), _request(3, listed=False), _request(3, requests=False), _request(3, timing=False),
                                 {**_request(3, timing=False), "timing": None}],
                         ids=["at forward 0", "a step the engine did not report", "a request the timing does not list",
                              "a timing without its requests", "no timing", "a null timing"])
def test_a_request_whose_capture_names_no_first_step_is_scheduled_from_forward_0(tok):
    assert M.step_rows(tok) == {0: (5, 1), 1: (1, 1)}


def test_a_request_is_scheduled_from_the_forward_its_capture_says_ran_its_prefill():
    assert M.step_rows(_request(3)) == {0: (0, 0), 1: (0, 0), 2: (0, 0), 3: (5, 1), 4: (1, 1)}


def test_a_step_whose_collectives_are_not_the_builds_signature_is_named(tmp_path):
    def widen(r: int, rows: list[dict]) -> list[dict]:
        return [{**x, "shape": [1, 16]} if (r, x["step"], x["k"]) == (0, 1, 1) else x for x in rows]

    rc, doc = _check(_capture(tmp_path / "run", rows=widen))
    rank0, rank1 = doc["structure"]["per_rank"]["0"], doc["structure"]["per_rank"]["1"]
    assert rank0["steps_mismatching"] == [1] and rank1["steps_mismatching"] == []
    assert rank0["n_steps_site_mismatching"] == 0 and rank0["n_steps_rows_mismatching"] == 0
    assert doc["structure"]["pass"] is False and doc["provenance"]["pass"] is True
    assert rc == 1 and doc["pass"] is False


def test_stored_tensors_the_recorder_rows_do_not_count_fail_by_name(tmp_path):
    def drop_last(r: int, rows: list[dict]) -> list[dict]:
        return [x for x in rows if (r, x["step"], x["k"]) != (1, 1, 2)]

    rc, doc = _check(_capture(tmp_path / "run", rows=drop_last))
    p = doc["provenance"]
    assert p["digest_mismatches"] == [
        {"rank": -1, "step": 1, "k": -1, "tensor": "count", "why": "stored tensors rank0 3/3 rank1 3/3 vs recorder rows 3 / 2"},
        {"rank": -1, "step": 1, "k": 2, "tensor": "row", "why": "no recorder row for the stored tensor"}]
    assert p["n_digest_mismatches"] == 2 and p["collectives_checked"] == 5 and p["tensors_digested"] == 4 * 5
    assert p["pass"] is False and rc == 1 and doc["pass"] is False


def test_a_recorder_row_keyed_at_another_position_fails_by_name(tmp_path):
    def rekey(r: int, rows: list[dict]) -> list[dict]:
        return [{**x, "k": 5} if (r, x["step"], x["k"]) == (0, 0, 1) else x for x in rows]

    rc, doc = _check(_capture(tmp_path / "run", rows=rekey))
    p = doc["provenance"]
    assert p["digest_mismatches"] == [{"rank": 0, "step": 0, "k": 1, "tensor": "key", "why": "recorder row keyed (step 0, k 5) at position 1 of step 0"}]
    assert p["n_digest_mismatches"] == 1 and p["collectives_checked"] == 6 and p["n_mismatches"] == 0
    assert p["pass"] is False and rc == 1 and doc["pass"] is False


@pytest.mark.parametrize("builds, said", [
    ({"build_rank": ["b0", "b1"], "build_rank0": "b0"}, "give either --build-rank DIR (repeated, rank order) or --build-rank0/--build-rank1, not both"),
    ({"build_rank": ["b0"]}, "--build-rank: 1 rank Build(s); a tensor-parallel world needs at least 2"),
    ({"build_rank0": "b0"}, "check arm: --build-rank DIR (repeated, rank order) or --build-rank0 and --build-rank1 required")])
def test_a_check_not_given_two_rank_builds_in_one_spelling_is_a_usage_error(tmp_path, builds, said):
    with pytest.raises(UsageError, match=re.escape(said)):
        M.main(M.Options(arm="check", out=str(tmp_path), **builds))
    assert not (tmp_path / "tp2_match.json").exists()


def test_the_command_line_ends_a_usage_error_with_exit_2_and_the_reason(tmp_path, capsys):
    with pytest.raises(SystemExit) as end:
        cli.main(["tp-match", "--arm", "check", "--out", str(tmp_path), "--build-rank", "b0"])
    assert end.value.code == 2
    assert "error: --build-rank: 1 rank Build(s); a tensor-parallel world needs at least 2" in capsys.readouterr().err


def test_the_match_record_is_the_captures_tokens_by_the_workloads_request_and_the_checks_verdict(tmp_path):
    run = _capture(tmp_path / "run")
    tokens, record = M.match_record(str(run), {"r0": "r3"}, 2)
    assert tokens == {"r3": GENERATED}
    assert record == {"path": str(run / "capture" / "tokens.json"), "requests": 1, "generated_total": 2, "tp2_match_verdict": None,
                      "capture_dir": str(run / "capture"), "stored_steps_per_rank": {0: 2, 1: 2}}
    assert _check(run)[0] == 0
    assert M.match_record(str(run), {"r0": "r3"}, 2)[1] == {**record, "tp2_match_verdict": True}


def test_a_match_record_with_a_request_the_workload_does_not_declare_is_refused(tmp_path):
    with pytest.raises(Refused, match="fail-closed: Match record request r0 is not a request of this workload"):
        M.match_record(str(_capture(tmp_path / "run")), {"other": "r0"}, 2)


def test_a_match_record_without_a_request_the_workload_declares_is_refused(tmp_path):
    with pytest.raises(Refused, match=re.escape("fail-closed: Match record covers ['r0'] but the workload declares ['r0', 'r1']")):
        M.match_record(str(_capture(tmp_path / "run")), {"r0": "r0", "second": "r1"}, 2)
