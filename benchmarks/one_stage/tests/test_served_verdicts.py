"""`served_verdicts.py`'s window pick, its subprotocols' pieces and its record, on toy serve dirs and answers:
the window it audits holds the most requests and none carried across windows; a subprotocol with no piece is named missing
with the command it lacks and given no verdict, one that refused to run is missing, and one that ran out of its limit or
wrote no answer errs; and sampled proofs' accepted answer composes, with the others missing, to an undecided block record that
`block_record.check` holds and its schema admits, while a refusal or an audit that wrote nothing composes to reject with no
block record. The block record's verdict is the Lean check's answer, so with none here it is undecided (the window's
pending) even when every subprotocol accepted on its node; the record names each one missing, rejecting, malformed or
about another server. An accepted network certifier's clause names the one link K counts and every other
link of the node uncounted and uncertified; one whose certifier ran on the node names no link. The wipe composes only
when its answer accepts on the record's server. Remote's product runs no onsite piece and composes sampled proofs and
PoUW alone, the others dropped. PoUW's development piece checks the window's gate session and credits
nothing, and the gate is refused while the lane has a period
open. The period the served windows join opens, through infra's own gate, on the Program serve-openai declared and
the circuit files built for it, as core's credit reads them, and is refused on files of anything else.
`served_verdicts.sh`'s window circuits gate on every class of CLASSES, and count the class of record only where
CLASSES names it; its endpoint gets one token budget, T, with or without H1. Its preflight refuses a flag the tree's certifier
front doesn't take, a front whose endpoint isn't on the warden's id form, and a front load with no front; its composite
step publishes each case's and each product's verdict lines and nothing else. Its V3c step runs the split network
certifier's chain after the window commit and its period check on V3c's period.json, and fails where no period.json is
written; its preflight refuses the split beside the certifier's whole piece, or without V3c. Under H1_TIE its serve gives
serve-openai the tie's command, which, run as `serve_tie` runs it at an H1 window's close, writes the window's advice and
commits it chained after H1's W, its registration the window's record R, and writes no R when the advice is pending or the
commit refused; its V3c then audits the commit H1 bound and makes none, and has none for a window H1 bound no R for or bound
another, its audit tied to W at stratified:1 on a port for each proof block of R's sample; its preflight refuses H1_TIE
without H1's prover state, without V3c, on a tree whose window-commit takes no --window-registration or whose records_audit
takes no --tie, at a sampling design other than stratified:K, past the sample session's time, or for onsite on a node short
of V3c's audit's memory. Under DAY its serve gives serve-openai the day's declaration and its V3c audit the day and the
registration log, and nothing of either without DAY; its preflight refuses a DAY without its log or program, or on a tree
without them. There
step 6 defers the audit and the composite, leaving the follow-on's settings and its every input, listed and hashed, and
nothing in $OUT but a line; the follow-on holds the copy to those sums, refuses another setting, a node short of the
memory or another tree, under DAY holds the day's log as the deferral copied it to the serve's declaration, and audits,
records about the served node and composes, each saying it composed elsewhere, as the record says where it was composed
and what was deferred. What it publishes holds a load's tokens and span by path, leaving the inference-only policy, its
digest and its verdict line as written. Its load, `serve_load.py
--varied`, writes each request's send, first-token and done times beside its record, on an injected clock and a stand-in
endpoint."""
from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
import importlib.util
import io
import itertools
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tarfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

import jsonschema
import pytest
from verity.hardware import site as SITE
from verity.hardware import registration as RL
from verity.protocols.pous.verifier import wipe_period as W
from verity.protocols.spot.verifier.one_stage import registration as REG
from verity.protocols.poub.verifier import ledger

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import served_verdicts as S
from verity_e2e_demo import block_record as BR

SCHEMA = json.loads((Path(BR.__file__).parent / "schemas" / "block-record.v2.schema.json").read_text())
_SL = importlib.util.spec_from_file_location("serve_load", HERE.parents[2] / "demos" / "vllm" / "serve_load.py")
SL = importlib.util.module_from_spec(_SL)
_SL.loader.exec_module(SL)


def _write(d: Path, name: str, rows: list[dict]) -> None:
    (d / name).write_text("".join(json.dumps(r) + "\n" for r in rows))


def _serve_dir(tmp_path: Path) -> Path:
    """Windows 0-3: w0 one request, w1 three, w2 one carried on into w3, w3 that one and its own; every seal good."""
    reqs = [("a", [0]), ("b", [1]), ("c", [1]), ("d", [1]), ("e", [2, 3]), ("f", [3])]
    by = {k: [u for u, ws in reqs if k in ws] for k in range(4)}
    _write(tmp_path, "seals.jsonl", [{"window": k, "requests": by[k], "row_dir": f"w{k}"} for k in range(4)])
    _write(tmp_path, "windows.jsonl", [{"window": k, "passed": True, "commit": {"steps": 5 + k}} for k in range(4)])
    _write(tmp_path, "requests.jsonl", [{"user": u, "engine_id": u, "windows": ws, "check": {"passed": True}} for u, ws in reqs])
    return tmp_path


def test_the_window_is_the_fullest_one_with_no_request_carried(tmp_path):
    got = S.pick(_serve_dir(tmp_path), slots=4)
    assert (got["window"], got["requests"], got["users"], got["steps"]) == (1, 3, ["b", "c", "d"], [5, 11])
    assert set(got["refused"]) == {"2", "3"} and "carried" in got["refused"]["3"]


def test_the_window_subcommand_prints_the_whole_pick_which_stays_on_the_node(tmp_path):
    """`served_verdicts.py window` prints `pick`'s window whole, for the serve dir's private/window.json: the harness's
    WINDOW line says none of it."""
    got = S.pick(_serve_dir(tmp_path), slots=4)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert S.main(["window", str(tmp_path), "--slots", "4"]) == 0
    assert json.loads(out.getvalue()) == got


REGISTRATION = {"schema": "verity-vllm/advice-registration/v1", "row_bits": 256, "rows": 2, "T": 600, "window": 7,
                "binding": {"compiler": {"name": "c", "digest": "d"}, "advice": {"root": "ab" * 64, "bits": 392}},
                "commit": {"run_id": "r1/w7", "run_root": "ef" * 64},
                "compiler": {"name": "c", "digest": "d", "cap_bits": 8388608}, "bits_by_window": {"7": 392, "8": 7},
                "registered_roots": {"weights/w@16": "cd" * 64}, "requests": 3, "refused": {"q0": "2 of its 3 tokens"}}


def test_the_published_registration_pads_the_advices_length_to_the_cap(tmp_path, capsys):
    """The advice registration's public form states no length of the advice but its compiler's cap, counts and refuses
    none of its requests, names no window, and keeps its roots, compiler and binding's root."""
    got = S.public_registration(REGISTRATION)
    assert got["binding"]["advice"] == {"root": "ab" * 64, "bits": 8388608} and got["rows"] == 32768
    assert got["padded"] == list(S.PADDED) and got["commit"] == {"run_id": "r1/wK", "run_root": "ef" * 64}
    assert not {"requests", "compiled", "refused", "window", "bits_by_window"} & set(got) and "392" not in json.dumps(got)
    assert {k: got[k] for k in ("registered_roots", "compiler", "T", "schema")} == {
        k: REGISTRATION[k] for k in ("registered_roots", "compiler", "T", "schema")}
    (tmp_path / "r.json").write_text(json.dumps(REGISTRATION))
    assert S.main(["public-registration", str(tmp_path / "r.json")]) == 0
    assert json.loads(capsys.readouterr().out) == got


#: a window's number (``w7``, ``window 7``), a step range, a count of windows or requests, or a field of a window's own
#: holding a number, a list or a map of numbers, in text
_OWN = "windows|users|engine_steps|every_slot|population|bits_by_window|advice_bits"
_NUMBERED = "window|steps|requests|refused"


def _window_data(out: Path, k: int) -> list[str]:
    """Each place under ``out``, by path or in a file's text, that names window ``k`` or counts a window's own."""
    pat = re.compile(rf"(?<![A-Za-z0-9])w{k}(?![A-Za-z0-9])|(?i:\bwindows?)[ :=]+{k}\b|(?i:\bsteps?) \[\d|"
                     rf"\b\d+ (?:sealed|closed)\b|\d+ window\(s\)|\d+/\d+ ok|\"(?:{_OWN})\":|"
                     rf"\"(?:{_NUMBERED})\": ?(?:\d|\[\s*\d|\{{\s*\"\d|\[\s*\])")
    hits = []
    for p in sorted(out.rglob("*")):
        if p.is_file():
            rel = p.relative_to(out).as_posix()
            data = p.read_bytes()
            with contextlib.suppress(OSError):
                data = gzip.decompress(data)
            hits += [f"{rel}: {m.group()}" for m in pat.finditer(rel + "\n" + data.decode(errors="replace"))]
    return hits


#: a served run's verdict lines on window 7 of three closed, two sealed, steps [13, 19), windows 5 and 6 refused, and
#: their published form
LINES = [
    ("SERVE stopped (exit 0): 3 window(s) closed, 2 sealed", "SERVE stopped (exit 0): window(s) closed, sealed"),
    ('WINDOW {"engine_steps": 6, "every_slot": true, "refused": [5, 6], "slots": 4, "steps": [13, 19], "window": 7}',
     'WINDOW {"slots": 4}'),
    ('PERIOD closed p: {"body": {"complete": true, "windows": [{"complete": true, "root": "ab", "session": "s1"}]}, "seq": 9}',
     'PERIOD closed p: {"body": {"complete": true}, "seq": 9}'),
    ('THE NETWORK CERTIFIER {"rc": 1, "reason": "window 7: the scan\'s prog is no audit\'s", "verdict": "reject"}',
     'THE NETWORK CERTIFIER {"rc": 1, "reason": "window K: the scan\'s prog is no audit\'s", "verdict": "reject"}'),
    ("SAMPLED-PROOFS (run.sh exit 0): over SiluMul_v2{I=8192} at model.layers.0.mlp. steps [13, 19] (law subset:2), "
     'drawn members [5, 42]: RECORDS {"accepted": true}',
     "SAMPLED-PROOFS (run.sh exit 0): over SiluMul_v2{I=8192} at model.layers.0.mlp. steps (law subset:2), "
     'drawn members [5, 42]: RECORDS {"accepted": true}'),
    ('RECORD {"by": ["PoUS"], "composed": "reject", "node": "n", "steps": [13, 19], "window": 7}',
     'RECORD {"by": ["PoUS"], "composed": "reject", "node": "n"}'),
    ('V3C-AUDIT class 8,32,4,4,16 (records_audit.py exit 0): RECORDS {"ok": true, "circuit": "cd", "population": 37, "dra',
     f"V3C-AUDIT class 8,32,4,4,16 (records_audit.py exit 0): RECORDS {S.CUT}"),
    ("POUW-DEV calls at /srv/h1/w7/pouw-h1-calls.json: drawn unit 52294",
     "POUW-DEV calls at /srv/h1/wK/pouw-h1-calls.json: drawn unit 52294"),
    ("REQUESTS (exit 0): [serve-load] varied: 3/3 ok", "REQUESTS (exit 0): [serve-load] varied: ok"),
    ("CLASSES: the window circuit's 4,32,4; the certifier's scan W63 (w63k4), the one class the gate serves",) * 2,
    ('POUS {"rc": 1, "reason": "the session ended late after 9 of 320 rounds", "verdict": "reject"}',) * 2,
]


def test_a_published_line_names_no_window_and_counts_none_of_a_windows_own(tmp_path, monkeypatch, capsys):
    """Each verdict line's published form (`public_line`) names no window and counts no window closed, sealed or refused, and
    none of a window's steps or requests, in its text or its JSON, and replaces a JSON object cut short that names one; the
    class, the scan class, the sample, a refusal's reason and the protocol's rounds stay. `public-lines` prints that form and
    keeps each line whole in the serve dir's private/verdicts.txt, 0700, where that serve dir exists."""
    for line, public in LINES:
        assert S.public_line(line) == public and S.public_line(public) == public
    d = tmp_path / "serve"
    d.mkdir()
    text = "".join(line + "\n" for line, _ in LINES)
    monkeypatch.setattr(sys, "stdin", io.StringIO(text))
    assert S.main(["public-lines", "--private", str(d / "private" / "verdicts.txt")]) == 0
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "verdicts.txt").write_text(capsys.readouterr().out)
    assert (tmp_path / "out" / "verdicts.txt").read_text().splitlines() == [p for _, p in LINES]
    assert not _window_data(tmp_path / "out", 7)
    assert (d / "private" / "verdicts.txt").read_text() == text and (d / "private").stat().st_mode & 0o777 == 0o700
    monkeypatch.setattr(sys, "stdin", io.StringIO(text))
    assert S.main(["public-lines", "--private", str(tmp_path / "absent" / "private" / "verdicts.txt")]) == 0
    assert capsys.readouterr().out.splitlines() == [p for _, p in LINES] and not (tmp_path / "absent").exists()


def test_publish_holds_each_file_that_names_or_counts_a_window_and_leaves_its_public_form(tmp_path):
    """`publish`, once the run is done: a file whose path names a window is held whole in the serve dir's private/out, one
    whose text or JSON names or counts a window is held there in full with its public form left in out/, and the rest,
    the bench Build's requests by name, a window named ``all`` and the gzipped declared Program, stay byte for byte. Then out/
    names no window and counts none of a window's own, and a second pass holds nothing."""
    out, private = tmp_path / "out", tmp_path / "serve" / "private" / "out"
    files = {
        "verdicts.txt": "SERVE stopped (exit 0): 3 window(s) closed, 2 sealed\nCOMPOSITE CASE verity/pous/p2/v4/run accept\n",
        "record/verdicts.json": json.dumps({"window": 7, "node": "n", "composed": {"verdict": "pending", "reason": "window 7: no"}}),
        "record/block-record.json": json.dumps({"server_time": [{"server": "n", "steps": [13, 19]}], "composed": {"verdict": "pending"}}),
        "registration.json": json.dumps(REGISTRATION),
        "pieces/certifier/answer.json": json.dumps({"reason": "window 7: the scan's prog", "verdict": "reject"}),
        "pieces/bandwidth/answer.json": json.dumps({"profile": {"windows": 6}, "verdict": "accept"}),
        "serve/period.json": json.dumps({"close": {"body": {"windows": [{"root": "ab"}]}}, "rows": 512, "steps": 16}),
        "pieces/pouw/check/check.log": "calls /srv/h1/w7/pouw-h1-calls.json\nPOUW {\"window\": 7, \"verdict\": \"accept\"}\n",
        "sampled-proofs/w7/summary.json": json.dumps({"verdict": "accept"}),
        "predicted.json": json.dumps({"requests": {"build_request_LP11_T8": "ab"}, "row": "r"}),
        "pieces/wipe/served.json": json.dumps({"window": "all", "server": "n"}),
    }
    for rel, body in files.items():
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        (out / rel).write_text(body)
    (out / "serve" / "h1").mkdir()
    (out / "serve" / "h1" / "window-program.json.gz").write_bytes(gzip.compress(json.dumps({"rows": 512}).encode()))
    assert S.main(["publish", str(out), "--private", str(private)]) == 0
    assert not _window_data(out, 7), _window_data(out, 7)
    kept = {"predicted.json", "pieces/wipe/served.json"}
    for rel, body in files.items():
        assert (out / rel).exists() == (rel != "sampled-proofs/w7/summary.json"), rel
        if rel in kept:
            assert (out / rel).read_text() == body and not (private / rel).exists(), rel
        else:
            assert (private / rel).read_text() == body, rel
    assert json.loads((out / "record" / "verdicts.json").read_text()) == {
        "node": "n", "composed": {"verdict": "pending", "reason": "window K: no"}}
    assert json.loads((out / "serve" / "period.json").read_text()) == {"close": {"body": {}}, "rows": 512}
    assert (out / "verdicts.txt").read_text() == ("SERVE stopped (exit 0): window(s) closed, sealed\n"
                                                  "COMPOSITE CASE verity/pous/p2/v4/run accept\n")
    assert gzip.decompress((out / "serve" / "h1" / "window-program.json.gz").read_bytes()) == b'{"rows": 512}'
    assert S.publish(out, private) == []


def test_publish_holds_whole_each_file_that_isnt_text_but_the_declared_program(tmp_path, monkeypatch):
    """`publish` fails closed on a file it can't read as text: bytes that aren't UTF-8, a gzip stream that doesn't
    decompress, and gzipped JSON, whether or not it names a window, are each held whole in private/out. Only the declared
    Program (`PUBLIC_FILES`) stays in out/, byte for byte and unread, even past `PUBLISH_MAX`."""
    monkeypatch.setattr(S, "PUBLISH_MAX", 64)
    out, private = tmp_path / "out", tmp_path / "serve" / "private" / "out"
    program = gzip.compress(json.dumps({"rows": list(range(256))}).encode())
    public = gzip.compress(json.dumps({"rows": 512}).encode())
    files = {"serve/h1/window-program.json.gz": program,
             "blob.bin": b"\xff\x00window 7",
             "broken.json.gz": public[:-6],
             "public.json.gz": public,
             "named.json.gz": gzip.compress(json.dumps({"window": 7}).encode())}
    assert len(program) > S.PUBLISH_MAX
    for rel, body in files.items():
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        (out / rel).write_bytes(body)
    assert S.publish(out, private) == ["blob.bin", "broken.json.gz", "named.json.gz", "public.json.gz"]
    assert (out / "serve" / "h1" / "window-program.json.gz").read_bytes() == program
    assert not (private / "serve" / "h1" / "window-program.json.gz").exists()
    for rel in ("blob.bin", "broken.json.gz", "named.json.gz", "public.json.gz"):
        assert not (out / rel).exists() and (private / rel).read_bytes() == files[rel], rel
    assert S.publish(out, private) == []


#: `served_overhead.py metrics`' aggregate of a composed run's one load: 4 requests of 4 tokens, so 16 tokens, one block
#: a request
AGGREGATE = {"schema": "verity-bench/served-overhead-metrics/v0", "run_id": "r1", "commit": True, "protocols": ["commit"],
             "identity": {"precision": "verity_int8"}, "scheduler": {"max_num_batched_tokens": 512},
             "loads": [{"requests": 4, "tokens": 16, "steps": 9, "span_s": 2.5, "windows": {"n": 1, "open_s": 0.5, "h1": ["tied"]},
                        "time_per_output_token_s": 0.15625, "tokens_per_s": 6.4, "client_tokens_per_s": 6.1}]}
#: an inference-only policy as `inference_only.py policy` writes it, whose ``n_inputs.span_s`` is the lease's, not a load's
POLICY = {"format": "verity-bench/inference-only-policy/v1", "written_utc": "2026-11-06T05:58:00+00:00", "tree": "ab" * 20,
          "class": {"slots": 4, "prompt": 32, "outputs": 4, "positions": 35},
          "serve": {"model": "verity-int8/Llama-3.2-1B-INT8", "revision": "15b0976d", "precision": "verity_int8"},
          "session": {"definition": "cd" * 32, "key": "Session_v1{S=4}", "slots": 4, "positions": 35},
          "c": 123_456_789, "k": 4_096, "n": 25_165_824,
          "theta": "every weight parameter the class's circuit registers (147), held",
          "table": {"sha512": "ef" * 64, "entries": 40, "priced": ["BitAtx32_v1"]},
          "n_inputs": {"n": 25_165_824, "raw": 24_570_799, "grid_bits": 3,
                       "rate": {"macs_per_s": 989_429_760_000_000, "clock_mhz": 1830, "source": "θ"},
                       "utilization": [1, 4], "span_s": 3600, "utilization_source": "the declared share",
                       "span_source": "the serve's GPU lease limit",
                       "work": {"linears": 34_057_748_480, "attention": 41_287_680, "lm_head": 1_050_673_152,
                                "theta_macs": 36_241_670_144, "source": "one slot of the class's fixed load"}}}


def test_publish_holds_the_loads_tokens_and_span_by_path_and_leaves_the_policy_and_its_digest_as_written(tmp_path):
    """The load's tokens over a block's outputs are its exact block count, so `publish` holds each load's ``tokens`` and
    ``span_s`` (`OWN_AT`, by path) with its counts, keeping its rates; the inference-only policy's ``n_inputs.span_s``, a
    ``tokens`` under no load and the verdict citing the policy's digest stay byte for byte, and the digest is the same
    before and after."""
    import inference_only as IO

    out, private = tmp_path / "out", tmp_path / "serve" / "private" / "out"
    (out / "serve").mkdir(parents=True)
    (out / "serve" / "overhead.json").write_text(json.dumps(AGGREGATE) + "\n")
    IO._write(out / "inference-only-policy.json", POLICY)
    before = IO.digest(json.loads((out / "inference-only-policy.json").read_text()))
    verdict = {"format": IO.VERDICT_FORMAT, "verdict": "accept", "reason": None, "complies": True, "failure": None,
               "policy": {"sha512": before, "written_utc": POLICY["written_utc"]}, "c": POLICY["c"], "k": POLICY["k"],
               "n": POLICY["n"], "classes": [["cd" * 32, 1_000, 4_000]], "counts": IO.COUNTS, "slots": 4, "positions": 35}
    IO._write(out / "record" / "inference-only.json", verdict)
    (out / "pieces" / "pearlplus").mkdir(parents=True)
    (out / "pieces" / "pearlplus" / "load.json").write_text(json.dumps({"tokens": 1024, "span_s": 9.5, "served_tokens": 64}))
    kept = {p: p.read_bytes() for p in out.rglob("*.json") if p.name != "overhead.json"}
    assert S.publish(out, private) == ["serve/overhead.json"]
    (ld,) = json.loads((out / "serve" / "overhead.json").read_text())["loads"]
    assert ld == {"time_per_output_token_s": 0.15625, "tokens_per_s": 6.4, "client_tokens_per_s": 6.1}
    assert json.loads((private / "serve" / "overhead.json").read_text()) == AGGREGATE
    assert {p: p.read_bytes() for p in kept} == kept
    after = json.loads((out / "inference-only-policy.json").read_text())
    assert after["n_inputs"]["span_s"] == 3600 and IO.digest(after) == before == IO.digest(POLICY)
    assert json.loads((out / "record" / "inference-only.json").read_text())["policy"]["sha512"] == before
    assert S.public_json(POLICY) == POLICY and S.public_json(verdict) == verdict
    assert S.publish(out, private) == []


def test_publish_holds_pearlpluss_exclusion_tiles_count_and_ids_and_leaves_its_outputs(tmp_path):
    """Which rows a pass gives up is its values' to decide, so `publish` holds what Pearl+'s documents say of its exclusion
    tiles: the profile's count and the tiles it replayed by id (`OWN_AT`, by path), and each verdict's replay by tile, the
    reference's and the Lean's (`_tile_verdicts`). The verdict, its outputs (the work credited, the tie and the class it
    held), the draw's count and source and a record's verdicts by protocol stay as written, and the documents in full are
    under the private dir."""
    work = {"credited": 990, "credited_tiles": 3, "macs": 1000, "rho": "1/400", "epsilon": "1/100", "over_cap_draws": 0}
    answer = {"protocol": "verity/pouw/pearl-plus/v1", "verdict": "accept", "reason": None, "commitment": "ab" * 64,
              "public": {"work": work, "unit": "MAC", "tie": {"record": "cd" * 64, "tables": 10, "load": [2, 8, 2]}},
              "profile": {"delta": "1/100", "draws": 3, "source": "ef" * 32, "over_cap": "uncredited",
                          "exclusion_tiles": 2, "tiles": [[0, 1], [0, 5], [1, 2], [1, 7]]}}
    tiles = [{"pass": i, "tile": t, "ok": True, "reason": None} for i, t in answer["profile"]["tiles"]]
    docs = {"pieces/pearlplus/pearlplus-answer.json": answer,
            "pieces/pearlplus/pearlplus-lean.json": {"verdict": "accept", "drawn": 2, "verdicts": tiles, "work": work,
                                                     "tie": answer["public"]["tie"]},
            "pieces/pearlplus/verify.json": {"answer": answer, "verdicts": tiles, "accepted": True},
            "record/record.json": {"verdicts": {"pouw": "accept", "sampled_proofs": "accept"}, "tiles": 4,
                                   "rows": [{"protocol": "pouw", "answer": answer}]}}
    out, private = tmp_path / "out", tmp_path / "serve" / "private" / "out"
    for rel, doc in docs.items():
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        (out / rel).write_text(json.dumps(doc, indent=1) + "\n")
    assert S.publish(out, private) == sorted(docs)
    pub = {rel: json.loads((out / rel).read_text()) for rel in docs}
    assert {rel: json.loads((private / rel).read_text()) for rel in docs} == docs
    shown = {**answer, "profile": {k: v for k, v in answer["profile"].items() if k not in ("exclusion_tiles", "tiles")}}
    assert pub["pieces/pearlplus/pearlplus-answer.json"] == shown and shown["public"] == answer["public"]
    assert pub["pieces/pearlplus/pearlplus-lean.json"] == {"verdict": "accept", "drawn": 2, "work": work,
                                                           "tie": answer["public"]["tie"]}
    assert pub["pieces/pearlplus/verify.json"] == {"answer": shown, "accepted": True}
    assert pub["record/record.json"] == {**docs["record/record.json"], "rows": [{"protocol": "pouw", "answer": shown}]}
    assert json.loads(S.public_line("POUW answer " + json.dumps(answer))[len("POUW answer "):]) == shown
    assert S.public_line('POUW {"profile": {"draws": 3, "tiles": [[0, 1], [0, 5') == "POUW " + S.CUT
    assert S.publish(out, private) == []


def test_publish_holds_a_class_held_pearlplus_runs_exact_credit_and_leaves_its_grid_work(tmp_path):
    """A Pearl+ run held to its class's fixed load publishes its work in the grid form (`verify_run.grid_work`), and the
    exact credit that form replaces, which the rows the passes exclude decide, is ``exact_work``: `publish` holds it
    wherever a document names it (the reference's profile, the Lean's printout, `served_passes.py lean`'s record of both
    and the answer of record's profile), in a line's JSON, in a JSON string's text and in a JSON object cut short. The
    grid work, its give-up among it, stays as written, and the documents in full are under the private dir."""
    work = {"credited": 134_190_203_114, "credited_tiles": 29_440, "macs": 136_230_993_920, "rho": "1/400",
            "epsilon": "9982889/1000000000", "over_cap_draws": 0, "give_up": 335_544_320}
    exact = {"credited": 134_194_399_455, "credited_tiles": 29_429, "macs": 135_895_449_600}
    tie = {"record": "cd" * 64, "tables": 265, "load": [4, 32, 4]}
    answer = {"protocol": "verity/pouw/pearl-plus/v1", "verdict": "accept", "reason": None, "commitment": "ab" * 64,
              "public": {"work": work, "unit": "MAC", "tie": tie},
              "profile": {"delta": "1/100", "draws": 459, "source": "ef" * 32, "over_cap": "uncredited", "exact_work": exact}}
    lean = {"verdict": "accept", "drawn": 459, "work": work, "exact_work": exact, "tie": tie}
    of_record = {**answer, "profile": {**answer["profile"], "reference_work": work,
                                       "lean": {k: lean[k] for k in ("verdict", "drawn", "work", "tie")}}}
    docs = {"pieces/pearlplus/pearlplus-answer.json": answer, "pieces/pearlplus/pearlplus-lean.json": lean,
            "pieces/pearlplus/pearlplus-lean-check.json": {"lean": lean, "python": {"verdict": "accept", "work": work,
                                                                                    "exact_work": exact}, "differ": []},
            "pieces/pearlplus/verify.json": {"answer": answer, "timed": {"passes": 4, "macs": work["macs"],
                                                                         "credited": work["credited"]}},
            "record/record.json": {"rows": [{"protocol": "pouw", "answer": of_record}]}}
    out, private = tmp_path / "out", tmp_path / "serve" / "private" / "out"
    for rel, doc in docs.items():
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        (out / rel).write_text(json.dumps(doc, indent=1) + "\n")
    assert S.publish(out, private) == sorted(docs)
    pub = {rel: json.loads((out / rel).read_text()) for rel in docs}
    assert {rel: json.loads((private / rel).read_text()) for rel in docs} == docs
    for rel in docs:
        text = (out / rel).read_text()
        assert "exact_work" not in text and str(exact["credited"]) not in text and str(exact["macs"]) not in text, rel
        assert str(work["credited"]) in text and str(work["give_up"]) in text, rel
    shown = {**answer, "profile": {k: v for k, v in answer["profile"].items() if k != "exact_work"}}
    assert pub["pieces/pearlplus/pearlplus-answer.json"] == shown and shown["public"] == answer["public"]
    assert pub["pieces/pearlplus/pearlplus-lean.json"] == {k: v for k, v in lean.items() if k != "exact_work"}
    assert pub["pieces/pearlplus/pearlplus-lean-check.json"]["python"] == {"verdict": "accept", "work": work}
    assert pub["pieces/pearlplus/verify.json"] == {"answer": shown, "timed": docs["pieces/pearlplus/verify.json"]["timed"]}
    assert pub["record/record.json"]["rows"][0]["answer"]["profile"]["reference_work"] == work
    line = S.public_line("POUW answer " + json.dumps(answer))
    assert json.loads(line[len("POUW answer "):]) == shown
    quoted = S.public_line("POUW " + json.dumps({"stdout": json.dumps(lean)}))
    assert "exact_work" not in quoted and str(exact["credited"]) not in quoted and str(work["credited"]) in quoted
    assert S.public_line('POUW {"verdict": "accept", "exact_work": {"credited": 134194399455, "credited_ti') == "POUW " + S.CUT
    assert S.publish(out, private) == []


def test_a_published_line_holds_a_loads_tokens_and_span_and_leaves_the_inference_only_lines(tmp_path):
    """A verdict line's published form has no load's ``tokens`` or ``span_s`` in its JSON, in a JSON string's text or in a
    JSON object cut short, and the INFERENCE-ONLY lines, the policy's (its span in seconds) and the verdict's (the policy's
    digest), are their own published form."""
    import inference_only as IO

    agg = json.dumps(AGGREGATE)
    assert S.public_line(f"OVERHEAD {agg}") == "OVERHEAD " + json.dumps(S.public_json(AGGREGATE))
    assert not re.search(r'"(?:tokens|span_s)"', S.public_line(f"OVERHEAD {agg}"))
    assert S.public_line('OVERHEAD {"loads": [{"tokens": 16, "span_s": 2.5, "tokens_per_s": 6.') == f"OVERHEAD {S.CUT}"
    quoted = json.dumps({"said": agg})
    assert json.loads(json.loads(S.public_line(quoted))["said"]) == S.public_json(AGGREGATE)
    for kept in ('{"n_inputs": {"span_s": 3600, "raw": 24570799}}', '{"tokens": 1024, "span_s": 9.5}',
                 'POLICY {"n_inputs": {"span_s": 3600, "raw": 2457',
                 'PEARL+ {"load": {"requests": "all", "prompt_tokens": 1024}, "served_tokens": 64}'):
        assert S.public_line(kept) == kept
    i = POLICY["n_inputs"]
    policy_line = (f"INFERENCE-ONLY POLICY at class 4,32,4: c {POLICY['c']} gates, k {POLICY['k']} fresh bits, n {POLICY['n']} "
                   f"sessions (⌈{i['utilization'][0]}/{i['utilization'][1]} x {i['rate']['macs_per_s']} MAC/s x {i['span_s']} s / "
                   f"{i['work']['theta_macs']} θ-MACs⌉ = {i['raw']}, on the grid); session {POLICY['session']['definition'][:16]}…, "
                   f"policy {IO.digest(POLICY)[:16]}…")
    pub = {"verdict": "accept", "reason": None, "classes": [["cd" * 32, 1_000, 4_000]], "registered_session": True,
           "c": POLICY["c"], "k": POLICY["k"], "n": POLICY["n"], "policy": {"sha512": IO.digest(POLICY)}}
    for line in (policy_line, f"INFERENCE-ONLY {IO.line(pub)}", f"INFERENCE-ONLY {json.dumps(pub)}"):
        assert S.public_line(line) == line and S.public_line(line + "\n") == line + "\n"


def test_the_spot_line_names_the_partition_and_its_draw_and_no_advice_length_population_or_steps(tmp_path, capsys):
    row = {"ok": True, "accepted": True, "draw": [0, 2], "population": 37,
           "leak": {"advice_bits": 832, "bound": "vacuous", "other_inputs": 786432, "registered": 0}}
    part = {"templates": ["SiluMul_v2{I=8192}"], "law": "subset:2",
            "scope": {"op_path_prefix": "model.layers.0.mlp.", "steps": [5, 11]}}
    line = S.sampled_line(row, part)
    head, _, rec = line.partition(": RECORDS ")
    assert head == "over SiluMul_v2{I=8192} at model.layers.0.mlp. (law subset:2), drawn members [0, 2]"
    assert "steps" not in line and "11" not in line
    assert json.loads(rec)["leak"] == {"bound": "vacuous", "other_inputs": 786432, "registered": 0} and "832" not in line
    assert "population" not in line and "37" not in line and row["population"] == 37
    (tmp_path / "log").write_text("+ set -x\nRECORDS " + json.dumps(row) + "\n")
    (tmp_path / "p.json").write_text(json.dumps(part))
    assert S.main(["sampled-line", str(tmp_path / "log"), "--partition", str(tmp_path / "p.json")]) == 0
    assert capsys.readouterr().out == line + "\n"
    (tmp_path / "log").write_text("refused\n")
    assert S.main(["sampled-line", str(tmp_path / "log")]) == 1


def test_the_front_line_says_the_sessions_public_fields_alone(tmp_path, capsys):
    """The CERTIFIER-FRONT line carries `FRONT_PUBLIC` of the front's session and none of its prompts, request, kept
    fetch, gate or verdict's cells; a refused front's line names no reason, and a missing session says so."""
    session = {"rc": 0, "run": "c1", "prompts": {"count": 4, "ids": 64}, "request": "warden-rehearse --requests 4",
               "keep": "/serve/private/fetch.tar.gz", "gate": {"run": "g"}, "ended": "the rehearsal ended",
               "ended_utc": "t", "session_s": 128.0, "pinned": "ab" * 20, "leaked_removed": [], "origin_unix": 1,
               "verdict": {"cells": [7] * 3, "requests": 4, "wire_bits": 9}}
    (tmp_path / "session.json").write_text(json.dumps(session))
    assert S.main(["front-line", str(tmp_path / "session.json")]) == 0
    said = json.loads(capsys.readouterr().out)
    assert list(said) == list(S.FRONT_PUBLIC) and said == {k: session[k] for k in S.FRONT_PUBLIC}
    (tmp_path / "session.json").write_text(json.dumps({"refused": "4 prompts of 64 ids: past one link-window"}))
    assert S.main(["front-line", str(tmp_path / "session.json")]) == 0
    assert "64" not in capsys.readouterr().out
    assert S.main(["front-line", str(tmp_path / "absent.json")]) == 1
    assert capsys.readouterr().out == "no session (FileNotFoundError)\n"


def test_the_certifiers_chain_files_that_count_requests_move_to_the_serve_dirs_private_after_the_record(tmp_path):
    """`served_verdicts.sh private-pieces` moves CERTIFIER_PRIVATE out of $OUT/pieces/certifier into the serve dir's
    private/, 0700, and leaves the piece's answer, case and verdict where the record and its reader find them."""
    piece, d = tmp_path / "out" / "pieces" / "certifier", tmp_path / "serve"
    moved = ["chain/certs.json", "chain/report/report.json", "chain/tie/summary.json", "chain/n2/summary.json", "piece.log",
             "chain.log", "period.log"]
    kept = ["answer.json", "case.json", "chain/scan/answers.json"]
    for f in moved + kept:
        (piece / f).parent.mkdir(parents=True, exist_ok=True)
        (piece / f).write_text(f)
    env = {"PATH": os.defpath, "HOME": str(tmp_path), "D": str(d), "OUT": str(tmp_path / "out")}
    rc = subprocess.run(["bash", str(HERE.parent / "served_verdicts.sh"), "private-pieces"], cwd=tmp_path, env=env,
                        check=False).returncode
    assert rc == 0
    assert sorted(str(p.relative_to(piece)) for p in piece.rglob("*") if p.is_file()) == sorted(kept)
    private = d / "private" / "pieces" / "certifier"
    assert sorted(str(p.relative_to(private)) for p in private.rglob("*") if p.is_file()) == sorted(moved)
    assert (private / "chain/tie/summary.json").read_text() == "chain/tie/summary.json"
    assert all((d / p).stat().st_mode & 0o777 == 0o700 for p in ("private", "private/pieces", "private/pieces/certifier"))


def test_one_piece_runs_alone_or_all_but_it(tmp_path, capsys):
    """``only`` runs the named pieces alone and ``skip`` all but them, so the certifier's can run beside the others."""
    assert list(S.pieces(tmp_path, 1, tmp_path / "pieces", env={}, only=["certifier"])) == ["certifier"]
    assert "the network certifier" in capsys.readouterr().out
    assert list(S.pieces(tmp_path, 1, tmp_path / "pieces", env={}, skip=["certifier"])) == ["pouw", "pous", "bandwidth"]
    assert "the network certifier" not in capsys.readouterr().out


def test_a_window_past_the_slots_or_named_and_refused_is_refused(tmp_path):
    d = _serve_dir(tmp_path)
    assert S.pick(d, slots=2)["window"] == 0
    with pytest.raises(ValueError, match="carried across windows"):
        S.pick(d, slots=4, want=3)
    with pytest.raises(ValueError, match="no seal"):
        S.pick(d, slots=4, want=9)


def test_a_failed_seal_or_commit_or_check_is_refused(tmp_path):
    d = _serve_dir(tmp_path)
    _write(d, "seals.jsonl", [{"window": 0, "requests": ["a"], "error": "boom"},
                              {"window": 1, "requests": ["b", "c", "d"]}])
    _write(d, "windows.jsonl", [{"window": 0, "passed": True, "commit": {"steps": 5}},
                                {"window": 1, "passed": False, "commit": {"steps": 6}}])
    with pytest.raises(ValueError, match="seal failed.*Commit did not pass|Commit did not pass.*seal failed"):
        S.pick(d, slots=4)


def test_a_subprotocol_with_no_piece_is_named_missing_with_the_command_it_lacks(tmp_path, capsys):
    got = S.pieces(tmp_path, 1, tmp_path / "pieces", env={})
    lines = capsys.readouterr().out.splitlines()
    assert [g["verdict"] for g in got.values()] == ["missing"] * 4
    assert [line.split(" (")[0] for line in lines] == ["MISSING PoUW", "MISSING PoUS", "MISSING the bandwidth audit",
                                                       "MISSING the network certifier"]
    for line, (cmd, owner) in zip(lines, (("POUW_CMD", "compute-accounting"), ("POUS_CMD", "memory-accounting"),
                                          ("BANDWIDTH_CMD", "memory-accounting"), ("CERTIFIER_CMD", "network-accounting"))):
        assert f"({cmd} unset; {owner}): " in line
    assert not (tmp_path / "pieces").exists()


def _reject(protocol: str) -> dict:
    return {"protocol": protocol, "verdict": "reject", "reason": "no", "commitment": None, "public": None, "profile": None}


def test_a_piece_that_ran_gives_its_answer_and_one_that_wrote_none_errs(tmp_path, capsys):
    env = {"POUS_CMD": f"echo '{json.dumps(_reject('verity/pous/p2/v4/run'))}' > \"$3/answer.json\"",
           "CERTIFIER_CMD": "true", "BANDWIDTH_CMD": 'echo \'{"verdict": "reject", "reason": "no"}\' > "$3/answer.json"'}
    got = S.pieces(tmp_path, 1, tmp_path / "pieces", env=env, node="vy-dc-h200")
    assert got["pous"] == {"verdict": "reject", "reason": "no", "rc": 0}
    assert got["certifier"]["verdict"] == "error" and "wrote no answer" in got["certifier"]["reason"]
    assert got["bandwidth"]["verdict"] == "error" and "not a core answer" in got["bandwidth"]["reason"]
    assert got["pouw"]["verdict"] == "missing"
    assert json.loads((tmp_path / "pieces" / "pous" / S.RAN).read_text()) == {"server": "vy-dc-h200", "rc": 0,
                                                                              "limit": S.PIECE_TIMEOUT}
    assert {k: S.piece_row(k, tmp_path / "pieces")["verdict"] for k in S.PIECES} == {
        "work": "missing", "bandwidth": "error", "space": "reject", "communication": "error"}


def test_a_piece_runs_within_its_limit_and_one_out_of_time_or_refused_says_so(tmp_path):
    """Each piece runs under `timeout` at ``<COMPONENT>_TIMEOUT``, else ``PIECE_TIMEOUT``: one that runs out of it (exit
    124, which the fake `timeout` gives for the limit ``0``) errs naming the limit, and one that writes ``refused.txt``
    instead of an answer is missing with the refusal as why."""
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "timeout").write_text(f'#!/bin/sh\necho "$3" >> {tmp_path}/limits\n[ "$3" = 0 ] && exit 124\nshift 3\nexec "$@"\n')
    (bin_ / "timeout").chmod(0o755)
    env = {"PATH": f"{bin_}:{os.defpath}", "PIECE_TIMEOUT": "7m", "BANDWIDTH_TIMEOUT": "0", "BANDWIDTH_CMD": "true",
           "POUS_CMD": 'echo "PoUS REFUSED: the card has a compute process" > "$3/refused.txt"; exit 3'}
    got = S.pieces(tmp_path, 1, tmp_path / "pieces", env=env, node="n")
    assert (tmp_path / "limits").read_text().split() == ["7m", "0"]
    assert got["bandwidth"] == {"verdict": "error", "rc": 124,
                                "reason": "its piece ran (exit 124, out of time at 0) and wrote no answer; "
                                          "its log is bandwidth/piece.log"}
    assert got["pous"] == {"verdict": "missing", "rc": 3,
                           "why": "its piece refused to run: PoUS REFUSED: the card has a compute process"}


def test_a_piece_that_left_a_file_that_is_not_json_errs_about_no_server_and_the_next_piece_still_runs(tmp_path):
    env = {"BANDWIDTH_CMD": f"echo '{json.dumps(_reject('verity/remote-bandwidth/v0'))}' > \"$3/answer.json\"; "
                            "echo '{' > \"$3/served.json\"",
           "POUS_CMD": f"echo '{json.dumps(_reject('verity/pous/p2/v4/run'))}' > \"$3/answer.json\""}
    got = S.pieces(tmp_path, 1, tmp_path / "pieces", env=env, node="n")
    assert got["bandwidth"]["verdict"] == "error"
    assert got["bandwidth"]["reason"].startswith("its piece left a file that isn't JSON: JSONDecodeError")
    assert got["pous"] == {"verdict": "reject", "reason": "no", "rc": 0}
    row, at = S.piece("bandwidth", tmp_path / "pieces")
    assert (row["verdict"], row["owner"], at) == ("error", "memory-accounting", (None, "real"))


def _accepted() -> dict:
    a = {"protocol": "verity/sampled-proofs/one-stage/v1", "verdict": "accept", "reason": None, "commitment": "ab" * 64,
         "public": {"bound": 2, "drawn": [0, 3], "per_template": None, "proved": [0, 3]},
         "profile": {"levels": [{"name": "unit", "size": 4}], "drawn": {"unit": 2}, "delta": "1/100"}}
    return S.row_of(S.INTEGRITY_OWNER, a, Path("/nonexistent"))


def _rows(integrity: dict, pieces: Path) -> dict:
    return {"integrity": integrity, **{k: S.piece_row(k, pieces) for k in S.PIECES}, "wipe": S.piece_row("wipe", pieces)}


def test_spot_alone_composes_to_an_undecided_block_record(tmp_path):
    report, rec, line = S.record(_rows(_accepted(), tmp_path), window=1, steps=(5, 11), node="vy-dc-h200", run="r1")
    jsonschema.Draft202012Validator(SCHEMA).validate(rec)
    assert BR.check(rec) == {}
    assert rec["legs"]["integrity"]["quantity"]["value"] == 2 and rec["server_time"] == [{"server": "vy-dc-h200", "steps": [5, 11]}]
    assert (line["composed"], line["record_missing"]) == ("pending", ["bandwidth", "certifier", "pous", "pouw", "wipe"])
    assert (rec["composed"]["verdict"], rec["composed"]["answer"]) == ("undecided", None)
    assert {name: why.split(" unset: ")[0] for name, why in line["missing"].items()} == {
        "PoUW": "POUW_CMD", "the bandwidth audit": "BANDWIDTH_CMD", "PoUS": "POUS_CMD", "the network certifier": "CERTIFIER_CMD",
        "the wipe": "WIPE_CMD"}
    assert report["protocols"]["work"] == {"verdict": "missing", "owner": "compute-accounting",
                                           "why": f"POUW_CMD unset: {S.PIECES['work'][3]}"}
    assert line["elsewhere"] == {} and "unserved" not in line


def test_an_accepted_pous_answer_says_when_its_weights_were_held_their_share_and_where_its_quantity_is_not_established(
        tmp_path):
    """PoUS's answer in the shape its served piece writes: weights reloaded on the card after the window's serve, 64.9% of
    the checkpoint, at ε = 1/100 + 2^-128, which the block record establishes at neither of its δs. Its accept's clause in
    the composed reason says each, and the ``RECORD`` line carries that reason."""
    eps = Fraction(1, 100) + Fraction(1, 2 ** 128)
    held = "loaded from the served snapshot on the card the window served on, after its serve ended"
    (d := tmp_path / "pous").mkdir()
    (d / "answer.json").write_text(json.dumps({
        "protocol": "verity/pous/p2/v4/run", "verdict": "accept", "reason": None, "commitment": "ef" * 64,
        "public": {"certified": True, "eps": str(eps), "s_bits": 8169621180},
        "profile": {"weights": {"held": held, "share": "237568/366225", "covers": "linears"},
                    "outside": {"bytes": {"the embedding and the head": 525336576, "the norms": 135168,
                                          "the per-row scales": 1507328}}}}))
    (d / S.RAN).write_text(json.dumps({"server": "vy-dc-h200", "rc": 0, "limit": "60m"}))
    rows = _rows(_accepted(), tmp_path)
    report, rec, line = S.record(rows, window=1, steps=(5, 11), node="vy-dc-h200", run="r1",
                                 servers={k: S.piece_server(k, tmp_path) for k in [*S.PIECES, "wipe"]})
    assert rec["legs"]["pous"]["at"] == {d: {"holds": False, "why": f"the run established it except with {eps}, more than {d}"}
                                         for d in ("1/100", "1/1000000")}
    clause = line["reason"].split("; PoUS: ", 1)[1]
    assert clause.startswith(f"accept (its weights {held}, not concurrently with the block; they are 64.9% of the "
                             "checkpoint (linears); the embedding and the head, the norms, the per-row scales are outside "
                             "its claim; its quantity is not established at δ = 1/100 or 1/1000000: the run established "
                             "it except with 1/100 + 2^-128)")
    assert line["reason"] == report["composed"]["reason"] and rec["composed"]["answer"] is None
    assert line["reason"].startswith("sampled proofs: accept; ")


def test_a_bandwidth_accept_says_its_error_with_the_term_it_leaves_out_beside_it(tmp_path):
    """The answer the bandwidth session's piece writes of an accepted session (`served_bandwidth.answer`) names the term
    its eps leaves out, PoUB's ledger's object (`ledger.NAMED_TERMS`). The block record, which `block_record.check`
    holds, carries it beside the error sum under the bandwidth audit's name, the sum's number the answer's eps, and the
    accept's clause in the composed reason says the error with the term beside it in the ledger's own words
    (`ledger.DELTA_FORM` with the error for δ). At an eps the record establishes at neither δ, the clause that says so
    prints the error the same way. The same answer with the term struck from its profile is the bandwidth audit's
    error: the composed verdict is ``reject``, with no block record."""
    import served_bandwidth as SB

    beside = ledger.DELTA_FORM.removeprefix("δ")
    (d := tmp_path / "bandwidth").mkdir()
    (d / S.RAN).write_text(json.dumps({"server": "vy-dc-h200", "rc": 0, "limit": "60m"}))

    def written(delta: str, samples: int) -> dict:
        """The piece's answer of an accepted session at ``delta`` that sampled ``samples`` openings: its eps is
        ``delta`` + ``samples``/2^64."""
        session = {"registration": {"digest": "cd" * 64},
                   "answer": {"verdict": "accept", "delta": delta, "span_ns": 16448, "q_bytes": 8224, "draws": [samples],
                              "receipts": 1, "unit_bytes": 8224, "units": 1, "windows": [0], "wrong_at_most": 0}}
        return SB.answer(session, "beside the served window", 1000)

    def composed(answer: dict) -> tuple:
        (d / "answer.json").write_text(json.dumps(answer))
        return S.record(_rows(_accepted(), tmp_path), window=1, steps=(5, 11), node="vy-dc-h200", run="r1",
                        servers={k: S.piece_server(k, tmp_path) for k in [*S.PIECES, "wipe"]})

    a = written("1/1099511627776", 0)
    assert a["verdict"] == "accept" and a["profile"]["named_terms"] == ledger.NAMED_TERMS
    report, rec, line = composed(a)
    jsonschema.Draft202012Validator(SCHEMA).validate(rec)
    assert BR.check(rec) == {}
    sums = rec["composed"]["joint_horizon"]
    assert sums["terms"]["bandwidth"] == a["public"]["eps"] == "1/1099511627776"
    assert sums["named_terms"] == {"bandwidth": ledger.NAMED_TERMS}
    assert all(x["not_established"] == [] for x in sums["at"].values())
    assert f"; the bandwidth audit: accept (its error reads 1/1099511627776{beside}); " in line["reason"]
    assert line["reason"] == report["composed"]["reason"]

    _, rec, line = composed(written("1/100", 1))
    assert BR.check(rec) == {}
    assert all(x["not_established"] == ["bandwidth"] for x in rec["composed"]["joint_horizon"]["at"].values())
    assert ("; the bandwidth audit: accept (its quantity is not established at δ = 1/100 or 1/1000000: the run established "
            f"it except with 1/100 + 2^-64{beside}); ") in line["reason"]

    silent = {**a, "profile": {k: v for k, v in a["profile"].items() if k != "named_terms"}}
    report, rec, line = composed(silent)
    assert rec is None and (line["composed"], line["by"], line["block_record"]) == ("reject", ["the bandwidth audit"], None)
    assert "ValueError: bandwidth: its answer's profile names no term its error leaves out; its piece writes them in its " \
           "answer's profile as named_terms (PoUB's ledger.NAMED_TERMS)" in report["protocols"]["bandwidth"]["reason"]
    assert "; the bandwidth audit: error: " in line["reason"]


def test_an_accepted_certifier_answer_says_what_its_trusted_host_premise_rests_on(tmp_path):
    """The network certifier's answer as its piece writes it when the warden ran on the gate's host and its checks, samples
    and coins on the served node: its accept's clause says the premise is assumed and why, so the ``RECORD`` line's
    accept isn't read as the auditor's devices having placed it."""
    premise = ("assumed, the auditor's devices not placed: the network certifier ran on 192.0.2.84, not the prover's "
               "vy-dc-h200, but the chain's checks and draws, the period check and their proofs' coins ran on vy-dc-h200")
    (d := tmp_path / "certifier").mkdir()
    (d / "answer.json").write_text(json.dumps({
        "protocol": "verity/certifier/v0", "verdict": "accept", "reason": None, "commitment": "cd" * 64,
        "public": {"K": 15764, "eps": "1/100", "windows": 1},
        "profile": {"placement": {"kind": "not-placed", "trusted-host": premise}}}))
    (d / S.RAN).write_text(json.dumps({"server": "vy-dc-h200", "rc": 0, "limit": "60m"}))
    _report, _rec, line = S.record(_rows(_accepted(), tmp_path), window=1, steps=(5, 11), node="vy-dc-h200", run="r1",
                                   servers={k: S.piece_server(k, tmp_path) for k in [*S.PIECES, "wipe"]})
    assert line["reason"].split("; the network certifier: ", 1)[1].startswith(f"accept (trusted-host {premise}; ")


def test_an_accepted_certifier_answer_names_the_one_link_k_counts_and_a_local_gate_names_none(tmp_path):
    """The network certifier's placement as its piece writes it. On the gate's host, its ``links`` are the relayed link
    K counts and the gate's ssh link it doesn't: the accept's clause names the counted link and says every other link of
    the node is uncounted and uncertified, before the premise. Under ``--gate local`` it has no links, and the clause
    states the premise alone."""
    node, host = "vy-dc-h200", "192.0.2.84"
    gated = {"kind": "not-placed", "parties": {"network_certifier": host, "prover": node, "verifier": node},
             "trusted-host": f"assumed, the auditor's devices not placed: the network certifier ran on {host}, not the "
                             f"prover's {node}",
             "links": [{"name": "certified", "from": host, "to": node, "counted": True,
                        "via": "the warden's tunnel to the endpoint's port"},
                       {"name": "gate", "from": node, "to": host, "counted": False,
                        "via": "ssh, both ways: the session's request and prompts, its status and its fetch"}],
             "uncounted": f"every link of {node}'s but the certified one"}
    local = {"kind": "prover-node", "parties": {"network_certifier": node, "prover": node, "verifier": node},
             "trusted-host": f"unmet: the network certifier ran on the prover's node {node} (the front's --gate local)",
             "links": [], "uncounted": f"every link of {node}'s"}
    clause = {}
    for name, place in (("gated", gated), ("local", local)):
        (d := tmp_path / name / "certifier").mkdir(parents=True)
        (d / "answer.json").write_text(json.dumps({
            "protocol": "verity/certifier/v0", "verdict": "accept", "reason": None, "commitment": "cd" * 64,
            "public": {"K": 15764, "eps": "1/100", "windows": 1}, "profile": {"placement": place}}))
        (d / S.RAN).write_text(json.dumps({"server": node, "rc": 0, "limit": "60m"}))
        _report, _rec, line = S.record(_rows(_accepted(), d.parent), window=1, steps=(5, 11), node=node, run="r1",
                                       servers={k: S.piece_server(k, d.parent) for k in [*S.PIECES, "wipe"]})
        clause[name] = line["reason"].split("; the network certifier: ", 1)[1]
    assert clause["gated"].startswith(
        f"accept (K counts the relayed link alone, {host} to {node} through the warden's tunnel to the endpoint's "
        f"port; every link of {node}'s but the certified one is uncounted and uncertified; "
        f"trusted-host {gated['trusted-host']}; ")
    assert clause["local"].startswith(f"accept (trusted-host {local['trusted-host']}; ")
    assert "K counts" not in clause["local"]


def test_an_audit_that_wrote_nothing_composes_to_reject_with_no_block_record(tmp_path):
    row = S.spot_row(tmp_path)
    assert row["verdict"] == "error" and "no summary" in row["reason"]
    _report, rec, line = S.record(_rows(row, tmp_path), window=1, steps=(5, 11), node="n", run=None)
    assert rec is None and (line["composed"], line["by"]) == ("reject", ["sampled proofs"])


WIPE_ACCEPT = {"protocol": W.PROTOCOL, "verdict": "accept", "reason": None, "commitment": "cd" * 64,
               "public": {"certified": True, "egress_mib": 0, "eps": str(W.EPS), "foreign_bits": 4096, "k": 5500,
                          "memory_bits": 1 << 40, "r": "1/2"},
               "profile": {"delta": str(W.EPS)}}
WIPE_REJECT = {"protocol": W.PROTOCOL, "verdict": "reject", "reason": "a read came back late", "commitment": "cd" * 64,
               "public": None, "profile": None}


@pytest.mark.parametrize("case", ["accept", "elsewhere", "unserved", "refused", "reject"])
def test_the_record_composes_the_wipe_only_when_it_accepted_on_the_records_server(tmp_path, monkeypatch, capsys, case):
    """`record` takes the wipe out of ``record_missing`` only when its answer accepts and its served.json names the
    record's node: served on another server it stays missing (``elsewhere``), an accepting answer with no served.json
    names no server and stays missing (``unserved``), a wipe that wrote no answer (served_wipe.sh refused) stays missing
    with its issue, and one whose answer rejects makes the record reject with no block record."""
    monkeypatch.setenv("RESEARCH_RUN_ID", "r1")
    monkeypatch.setattr(S, "spot_row", lambda _audit_dir, _run=None: _accepted())
    wipe = tmp_path / "pieces" / "wipe"
    wipe.mkdir(parents=True)
    if case == "refused":
        (wipe / "refused.txt").write_text("WIPE REFUSED: the card has a compute process\n")
    else:
        (wipe / "answer.json").write_text(json.dumps(WIPE_REJECT if case == "reject" else WIPE_ACCEPT))
        if case != "unserved":
            (wipe / "served.json").write_text(json.dumps({"server": "vy-dc-h200-2" if case == "elsewhere" else "vy-dc-h200"}))
    (tmp_path / "serve").mkdir()
    rc = S.main(["record", "--serve-run", str(_serve_dir(tmp_path / "serve")), "--window", "1",
                 "--sampled-proofs", str(tmp_path / "audit"), "--pieces", str(tmp_path / "pieces"),
                 "--out", str(tmp_path / "record"), "--node", "vy-dc-h200"])
    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1].removeprefix("RECORD "))
    if case == "reject":
        assert rc == 1 and (line["composed"], line["by"], line["block_record"]) == ("reject", ["the wipe"], None)
        assert not (tmp_path / "record" / "block-record.json").exists()
        return
    rec = json.loads((tmp_path / "record" / "block-record.json").read_text())
    jsonschema.Draft202012Validator(SCHEMA).validate(rec)
    assert rc == 0 and line["check"] == {} and line["composed"] == "pending"
    if case == "accept":
        assert line["record_missing"] == ["bandwidth", "certifier", "pous", "pouw"]
        assert rec["legs"]["wipe"]["server"] == "vy-dc-h200" and rec["legs"]["wipe"]["quantity"]["value"] == 4096
        assert rec["legs"]["wipe"]["runs"] == ["r1"] and rec["composed"]["legs"]["wipe"] == "accept"
        return
    assert line["record_missing"] == ["bandwidth", "certifier", "pous", "pouw", "wipe"]
    if case == "elsewhere":
        assert rec["legs"]["wipe"]["server"] == "vy-dc-h200-2" and rec["composed"]["elsewhere"] == ["wipe"]
    elif case == "unserved":
        assert rec["legs"]["wipe"] is None and "the wipe" not in line["missing"]
    else:
        assert rec["legs"]["wipe"] is None
        assert line["missing"]["the wipe"] == "its piece refused to run: WIPE REFUSED: the card has a compute process"


#: an accepted core answer of each piece's subprotocol, its public outputs the block record's leg certificate, the
#: bandwidth audit's profile naming the term its error leaves out, as its piece writes it
ACCEPTS = {
    "pouw": {"protocol": "verity/pouw/v1", "public": {"work": {"level": 3, "n": 4, "credited_tiles": 2, "credited": 8192}},
             "profile": {"delta": "1/1000000"}},
    "bandwidth": {"protocol": "verity/remote-bandwidth/v0", "public": {"eps": "1/1000000", "q_bytes": 6963200},
                  "profile": {"named_terms": ledger.NAMED_TERMS}},
    "pous": {"protocol": "verity/pous/p2/v4/run", "public": {"certified": True, "eps": "1/1000000", "s_bits": 1 << 30},
             "profile": {}},
    "certifier": {"protocol": "verity/certifier/v0", "public": {"K": 15764, "eps": "1/100", "windows": 1}, "profile": {}},
    "wipe": {k: WIPE_ACCEPT[k] for k in ("protocol", "public", "profile")},
}
COMPONENTS = ("pouw", "bandwidth", "pous", "certifier", "wipe")


@pytest.mark.parametrize("case, piece", [("accept", None)] + [
    (case, piece) for case in ("missing", "reject", "elsewhere", "malformed") for piece in COMPONENTS] + [
    ("unnamed", "bandwidth")])
def test_the_record_is_pending_without_a_lean_answer_and_names_each_subprotocol_that_does_not_accept(
        tmp_path, monkeypatch, capsys, case, piece):
    """Every piece's answer accepted on the record's node (each piece's `RAN`, which a `SERVED` naming no server leaves it
    about, and the wipe's `SERVED`): the composed verdict is ``pending``, the block record's ``undecided``, since its
    verdict is the Lean check's answer and none ran, with nothing missing, and its reason names each subprotocol's
    accept, the bandwidth audit's with its error and the term it leaves out beside it. One piece with no answer leaves it ``pending`` and names that piece missing with
    why; one that rejects, or accepts with outputs that are not its leg's certificate, makes it ``reject`` naming it, with
    no block record, and so does a bandwidth answer whose profile names no term its error leaves out; one about another
    server leaves it ``pending``, carried and named as elsewhere."""
    monkeypatch.setenv("RESEARCH_RUN_ID", "r1")
    monkeypatch.setattr(S, "spot_row", lambda _audit_dir, _run=None: _accepted())
    pieces = tmp_path / "pieces"
    for comp in COMPONENTS:
        (d := pieces / comp).mkdir(parents=True)
        if comp == piece and case == "missing":
            continue
        a = {"verdict": "accept", "reason": None, "commitment": "ef" * 64, **ACCEPTS[comp]}
        if comp == piece and case == "reject":
            a = {**a, "verdict": "reject", "reason": "a read came back late", "public": None, "profile": None}
        if comp == piece and case == "malformed":
            a = {**a, "public": {"unrelated": 1}}
        if comp == piece and case == "unnamed":
            a = {**a, "profile": {}}
        (d / "answer.json").write_text(json.dumps(a))
        server = "vy-dc-h200-2" if comp == piece and case == "elsewhere" else "vy-dc-h200"
        if comp == "wipe" or server != "vy-dc-h200":
            (d / S.SERVED).write_text(json.dumps({"server": server}))
        elif comp == "bandwidth":
            (d / S.SERVED).write_text(json.dumps({"served": {"window": 1}, "scope": "beside the served window"}))
        if comp != "wipe":
            (d / S.RAN).write_text(json.dumps({"server": "vy-dc-h200", "rc": 0, "limit": "60m"}))
    (tmp_path / "serve").mkdir()
    rc = S.main(["record", "--serve-run", str(_serve_dir(tmp_path / "serve")), "--window", "1",
                 "--sampled-proofs", str(tmp_path / "audit"), "--pieces", str(pieces), "--out", str(tmp_path / "record"),
                 "--node", "vy-dc-h200"])
    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1].removeprefix("RECORD "))
    leg, runner = piece, {v[1]: k for k, v in S.PIECES.items()} | {"wipe": "wipe"}
    name = S.NAMES[runner[piece]] if piece else None
    if case in ("reject", "malformed", "unnamed"):
        assert rc == 1 and (line["composed"], line["by"], line["block_record"]) == ("reject", [name], None)
        assert line["verdicts"][name] == ("reject" if case == "reject" else "error")
        assert not (tmp_path / "record" / "block-record.json").exists()
        report = json.loads((tmp_path / "record" / "verdicts.json").read_text())
        row = next(r for k, r in report["protocols"].items() if S.NAMES[k] == name)
        why = {"malformed": "KeyError", "unnamed": "ValueError: bandwidth: its answer's profile names no term its error "
                                                   "leaves out; its piece writes them in its answer's profile as named_terms"}
        assert case == "reject" or row["reason"].startswith(f"its answer is not the block record's {piece} leg: {why[case]}")
        return
    rec = json.loads((tmp_path / "record" / "block-record.json").read_text())
    jsonschema.Draft202012Validator(SCHEMA).validate(rec)
    assert rc == 0 and line["check"] == {} == BR.check(rec)
    if case == "accept":
        assert (line["composed"], line["record_missing"], line["missing"], line["elsewhere"]) == ("pending", [], {}, {})
        assert (rec["composed"]["verdict"], rec["composed"]["answer"]) == ("undecided", None)
        assert rec["composed"]["legs"] == {k: "accept" for k in BR.LEGS}
        late = {"the network certifier": " (its quantity is not established at δ = 1/1000000: the run established it "
                                         "except with 1/100)",
                "the bandwidth audit": f" (its error reads 1/1000000{ledger.DELTA_FORM.removeprefix('δ')})"}
        assert line["reason"] == "; ".join(
            f"{n}: accept{late.get(n, '')}" for n in S.NAMES.values())
        return
    assert line["composed"] == "pending" and line["record_missing"] == [leg]
    if case == "missing":
        what = S.WIPE[3] if piece == "wipe" else S.PIECES[runner[piece]][3]
        assert line["missing"] == {name: f"{piece.upper()}_CMD unset: {what}"}
        assert rec["legs"][leg] is None and line["elsewhere"] == {}
    else:
        assert line["missing"] == {} and line["elsewhere"] == {name: "vy-dc-h200-2"}
        assert rec["legs"][leg]["server"] == "vy-dc-h200-2" and rec["composed"]["elsewhere"] == [leg]


def test_remote_runs_no_onsite_piece_and_composes_spot_and_pouw_alone(tmp_path, monkeypatch, capsys):
    """PRODUCT=remote: `pieces` runs PoUW's piece and no onsite one, whatever its command, printing a DROPPED line for
    each; the record names PoUS, the bandwidth audit, the network certifier and the wipe dropped, neither missing nor
    composed, whatever their pieces left, and sampled proofs' and PoUW's accepts on its node leave it pending with nothing
    missing, the block record undecided until the remote check answers. The same pieces composed for onsite reject on the
    wipe's answer. The report and the line say they were composed on the
    served node, from the answer files it wrote, and the published line keeps that, as it keeps what the run deferred
    (``--deferred``) and the served compare's line (``--served-compare``; none, saying why, where its file is absent or
    can't be read). Off the node from its compose hand-off (``--off-node``) the record takes its engine steps from
    ``--steps``, with no requests file there, says it was composed from the hand-off, and is refused on the served node
    itself."""
    monkeypatch.setenv("RESEARCH_RUN_ID", "r1")
    monkeypatch.setattr(S.socket, "gethostname", lambda: "vy-dc-h200")
    monkeypatch.setattr(S, "spot_row", lambda _audit_dir, _run=None: _accepted())
    pieces, ran = tmp_path / "pieces", tmp_path / "ran"
    pouw = {"verdict": "accept", "reason": None, "commitment": "ef" * 64, **ACCEPTS["pouw"]}
    env = {"POUW_CMD": f"echo '{json.dumps(pouw)}' > \"$3/answer.json\"",
           **{f"{c}_CMD": f"touch {ran}" for c in ("POUS", "BANDWIDTH", "CERTIFIER")}}
    got = S.pieces(tmp_path, 1, pieces, env=env, node="vy-dc-h200", product="remote")
    why = "not run: the remote product composes sampled proofs and PoUW alone"
    assert not ran.exists()
    assert got == {"pouw": {"verdict": "accept", "rc": 0},
                   **{c: {"verdict": "dropped", "why": why} for c in ("pous", "bandwidth", "certifier")}}
    dropped = [x.split(" (")[0] for x in capsys.readouterr().out.splitlines() if x.startswith("DROPPED ")]
    assert dropped == ["DROPPED PoUS", "DROPPED the bandwidth audit", "DROPPED the network certifier"]
    (pieces / "wipe").mkdir()
    (pieces / "wipe" / "answer.json").write_text(json.dumps(WIPE_REJECT))
    (pieces / "wipe" / S.SERVED).write_text(json.dumps({"server": "vy-dc-h200"}))
    (tmp_path / "serve").mkdir()
    argv = ["record", "--serve-run", str(_serve_dir(tmp_path / "serve")), "--window", "1", "--sampled-proofs",
            str(tmp_path / "audit"), "--pieces", str(pieces), "--out", str(tmp_path / "record"), "--node", "vy-dc-h200",
            "--product", "remote"]
    rc = S.main(argv)
    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1].removeprefix("RECORD "))
    rec = json.loads((tmp_path / "record" / "block-record.json").read_text())
    jsonschema.Draft202012Validator(SCHEMA).validate(rec)
    assert rc == 0 and line["check"] == {} == BR.check(rec)
    assert (line["product"], line["composed"], line["record_missing"], line["missing"]) == ("remote", "pending", [], {})
    assert (rec["composed"]["verdict"], rec["composed"]["answer"]) == ("undecided", None)
    names = ("PoUS", "the bandwidth audit", "the network certifier", "the wipe")
    assert line["dropped"] == {n: why for n in names} and all(line["verdicts"][n] == "dropped" for n in names)
    assert line["reason"] == "sampled proofs: accept; PoUW: accept; " + "; ".join(f"{n}: dropped, {why}" for n in names)
    assert (rec["composed"]["product"], rec["composed"]["dropped"]) == ("remote", ["bandwidth", "certifier", "pous", "wipe"])
    assert rec["composed"]["legs"] == {"integrity": "accept", "pouw": "accept"} and rec["legs"]["wipe"] is None
    report = json.loads((tmp_path / "record" / "verdicts.json").read_text())
    assert report["protocols"]["wipe"] == {"verdict": "dropped", "owner": "memory-accounting", "why": why}
    where = {"host": "vy-dc-h200", "served_node": True, "from": S.COMPOSED_FROM}
    assert report["composed_on"] == line["composed_on"] == where and "deferred" not in line and "deferred" not in report
    assert json.loads(S.public_line("RECORD " + json.dumps(line)).removeprefix("RECORD "))["composed_on"] == where
    left = "V3c's audit and the composite, to a follow-on on a node with the memory, which composes the final record"
    assert S.main([*argv, "--deferred", left]) == 0
    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1].removeprefix("RECORD "))
    report = json.loads((tmp_path / "record" / "verdicts.json").read_text())
    assert (line["composed_on"], line["deferred"]) == (report["composed_on"], report["deferred"]) == (where, left)
    assert json.loads(S.public_line("RECORD " + json.dumps(line)).removeprefix("RECORD "))["deferred"] == left
    compare = tmp_path / "served-compare" / "served_compare.json"
    assert S.main([*argv, "--served-compare", str(compare)]) == 0
    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1].removeprefix("RECORD "))
    assert line["served_compare"] == "SERVED-COMPARE: none (served_compare.json absent: served_compare.py wrote no verdict)"
    said = "SERVED-COMPARE: EQUAL (the serve against row r's Build of record: PHI, BUILD and CUT PASS; B-OF-RECORD is the seal's)"
    compare.parent.mkdir()
    compare.write_text(json.dumps({"ok": True, "line": said}))
    assert S.main([*argv, "--served-compare", str(compare)]) == 0
    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1].removeprefix("RECORD "))
    report = json.loads((tmp_path / "record" / "verdicts.json").read_text())
    assert line["served_compare"] == report["served_compare"] == said and "deferred" not in line
    assert json.loads(S.public_line("RECORD " + json.dumps(line)).removeprefix("RECORD "))["served_compare"] == said
    compare.write_text("{")
    assert S.main([*argv, "--served-compare", str(compare)]) == 0
    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1].removeprefix("RECORD "))
    assert line["served_compare"].startswith("SERVED-COMPARE: none (served_compare.json can't be read: JSONDecodeError: ")
    monkeypatch.setattr(S.socket, "gethostname", lambda: "vy-auditor-2")
    assert S.main(argv) == 0
    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1].removeprefix("RECORD "))
    assert line["composed_on"] == {"host": "vy-auditor-2", "served_node": False, "from": S.COPIED_FROM}
    (tmp_path / "serve" / "requests.jsonl").unlink()
    assert S.main([*argv, "--steps", "5,11", "--off-node"]) == 0
    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1].removeprefix("RECORD "))
    report = json.loads((tmp_path / "record" / "verdicts.json").read_text())
    handed = {"host": "vy-auditor-2", "served_node": False, "from": S.HANDED_FROM}
    assert (line["composed_on"], report["composed_on"], line["steps"]) == (handed, handed, [5, 11])
    assert json.loads(S.public_line("RECORD " + json.dumps(line)).removeprefix("RECORD "))["composed_on"] == handed
    with pytest.raises(SystemExit):
        S.main([*argv, "--steps", "5", "--off-node"])
    capsys.readouterr()
    monkeypatch.setattr(S.socket, "gethostname", lambda: "vy-dc-h200")
    assert S.main([*argv, "--steps", "5,11", "--off-node"]) == 1
    assert capsys.readouterr().out.strip() == ("RECORD refused: a record composed off the served node is not composed on "
                                               "it: this host is vy-dc-h200")
    with pytest.raises(ValueError, match="this host is vy-dc-h200"):
        S.record(_rows(_accepted(), pieces), window=1, steps=(5, 11), node="vy-dc-h200", run="r1", off_node=True)
    _serve_dir(tmp_path / "serve")
    monkeypatch.setattr(S.socket, "gethostname", lambda: "vy-auditor-2")
    assert S.main([*argv, "--product", "onsite"]) == 1
    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1].removeprefix("RECORD "))
    assert (line["product"], line["composed"], line["by"]) == ("onsite", "reject", ["the wipe"])


#: remote's sampled proofs Lean (`vstar-recursive`) as `record --sampled-lean` runs it: its arguments and the case files
#: it was handed in $SEEN, and $ANSWER as its answer line (none when empty)
FAKE_SAMPLED_LEAN = f'''#!{sys.executable}
import json, os, sys
from pathlib import Path
a = sys.argv[1:]
json.dump({{"argv": a, "files": {{Path(f).name: Path(f).read_text() for f in a[2:5]}}}}, open(os.environ["SEEN"], "w"))
print("checking")
print(os.environ["ANSWER"])
'''
#: the toy registration record the hand-off's cases commit to
LEAN_RECORD = {"format": "toy-registration", "window": 7}


def _lean_handed(tmp_path: Path) -> tuple[list[str], Path]:
    """A compose host's copy of a hand-off, off the served node vy-dc-h200: sampled proofs' records audit with the
    summary the served node wrote (a refusal of its own), v1's case and v2's naming one proof blob, PoUW's piece's
    accepted answer, and a `FAKE_SAMPLED_LEAN`; then `record --off-node --sampled-lean` on it, remote's product. Its argv
    and the stand-in."""
    from verity.circuits import codec

    d = tmp_path / "serve"
    w = d / "sampled-proofs-served" / "window"
    (w / "m0").mkdir(parents=True)
    (w / "m0" / "p0.bin").write_bytes(b"proof")
    v1 = {"params": {"verifier": "toy"}, "commitment": codec.canonical_json(LEAN_RECORD).hex(),
          "evidence": codec.canonical_json({"draw": [0, 3]}).hex()}
    (w / "case.json").write_text(json.dumps(v1))
    (w / "case-v2.json").write_text(json.dumps({**v1, "params": {"verifier": "toy", "proofs": {}},
                                                "blobs": [str(w / "m0" / "p0.bin")]}))
    (d / "sampled-proofs-served" / "summary.json").write_text(json.dumps({"refused": "the node's own verdict"}))
    (p := tmp_path / "pieces" / "pouw").mkdir(parents=True)
    (p / "answer.json").write_text(json.dumps({"verdict": "accept", "reason": None, "commitment": "ef" * 64,
                                               **ACCEPTS["pouw"]}))
    (p / S.RAN).write_text(json.dumps({"server": "vy-dc-h200", "rc": 0, "limit": "60m"}))
    exe = tmp_path / "vstar-recursive"
    exe.write_text(FAKE_SAMPLED_LEAN)
    exe.chmod(0o755)
    return (["record", "--serve-run", str(d), "--window", "7", "--sampled-proofs", str(d / "sampled-proofs-served"),
             "--pieces", str(tmp_path / "pieces"), "--out", str(tmp_path / "record"), "--node", "vy-dc-h200",
             "--product", "remote", "--steps", "0,3", "--off-node", "--sampled-lean", "--sampled-exe", str(exe)], exe)


def _recorded(tmp_path: Path, argv: list[str], capsys) -> tuple[int, dict, dict]:
    rc = S.main(argv)
    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1].removeprefix("RECORD "))
    return rc, line, json.loads((tmp_path / "record" / "verdicts.json").read_text())["protocols"]


@pytest.mark.parametrize("edit", [
    {"summary.json": json.dumps({"refused": "another verdict of the node's"})},
    {"summary.json": json.dumps({"audit": {"accepted": True}})},
    {"summary.json": "{}", "window/audit.json": json.dumps({"accepted": False, "checks": {"proved": {"ok": False}}})},
    {"summary.json": None},
])
def test_an_edit_to_the_served_nodes_sampled_proofs_answer_leaves_the_row_recomputed_here_unchanged(
        tmp_path, monkeypatch, capsys, edit):
    """Off the served node, `record --sampled-lean` takes sampled proofs' row from this host's own run of its Lean on the
    hand-off's v2 case in the form ``one-stage``, handed the case's files and its proof blob, the answer tied to the
    record the case commits to: the summary and audit record the served node wrote don't enter it, so editing or deleting
    them leaves the row as it was, while without --sampled-lean the row follows them. The row and the line name its
    source, and PoUW's, still read from its piece's answer file, as the served node wrote it, which its edit changes."""
    from verity.protocols.spot.verifier.one_stage import registration as R

    monkeypatch.setattr(S.socket, "gethostname", lambda: "vy-auditor-2")
    argv, _exe = _lean_handed(tmp_path)
    answer = {**_accepted()["answer"], "protocol": "verity/sampled-proofs/v1", "commitment": R.digest(LEAN_RECORD)}
    monkeypatch.setenv("SEEN", str(tmp_path / "seen.json"))
    monkeypatch.setenv("ANSWER", json.dumps(answer))
    rc, line, rows = _recorded(tmp_path, argv, capsys)
    assert rc == 0 and (line["composed"], line["record_missing"]) == ("pending", []), line
    assert rows["integrity"]["answer"] == answer
    pouw_from = f"{S.AS_WRITTEN} (its piece files in pouw/)"
    assert line["answers"] == {"sampled proofs": S.LEAN_HERE, "PoUW": pouw_from}
    assert (rows["integrity"]["answer_from"], rows["work"]["answer_from"]) == (S.LEAN_HERE, pouw_from)
    seen = json.loads((tmp_path / "seen.json").read_text())
    w = tmp_path / "serve" / "sampled-proofs-served" / "window"
    assert seen["argv"][:2] == ["sampled", "check"] and seen["argv"][5:] == [str(w / "m0" / "p0.bin")]
    assert json.loads(seen["files"]["params.json"]) == {"form": "one-stage", "proofs": {}, "verifier": "toy"}
    assert json.loads(seen["files"]["commitment.bin"]) == LEAN_RECORD
    plain = [x for x in argv if x not in ("--sampled-lean", "--sampled-exe", argv[-1])]
    _rc, _line, node_rows = _recorded(tmp_path, plain, capsys)
    for name, body in edit.items():
        target = tmp_path / "serve" / "sampled-proofs-served" / name
        if body is None:
            target.unlink()
        else:
            target.write_text(body)
    _rc, line, edited = _recorded(tmp_path, argv, capsys)
    assert edited["integrity"] == rows["integrity"] and (line["composed"], line["record_missing"]) == ("pending", [])
    _rc, _line, node_edited = _recorded(tmp_path, plain, capsys)
    assert node_edited["integrity"] != node_rows["integrity"]
    assert node_rows["integrity"]["answer_from"] == f"{S.AS_WRITTEN} ({S.WRITTEN['integrity']})"
    (tmp_path / "pieces" / "pouw" / "answer.json").write_text(json.dumps({
        "verdict": "reject", "reason": "the node says so", "commitment": "ef" * 64, "protocol": "verity/pouw/v1",
        "public": None, "profile": None}))
    _rc, line, edited = _recorded(tmp_path, argv, capsys)
    assert edited["integrity"] == rows["integrity"] and edited["work"]["verdict"] == "reject"


@pytest.mark.parametrize("change, verdict, said", [
    ("no-v2", "reject", "refused: \"the node's own verdict\""),
    ("another-record", "error", "commits to"),
    ("no-blob", "error", "blob(s) that aren't files"),
    ("no-line", "error", "answered no line"),
])
def test_a_sampled_proofs_row_the_lean_here_cant_recompute_says_so_and_never_takes_the_nodes_word_for_a_bad_case(
        tmp_path, monkeypatch, capsys, change, verdict, said):
    """With no v2 case the hand-off holds no proofs to recompute from, so the row is the served node's, its answer_from
    saying so; a v2 case naming a missing blob, a Lean answer about another record, or one that answers no line is the
    row's error, recomputed here, never the node's verdict."""
    monkeypatch.setattr(S.socket, "gethostname", lambda: "vy-auditor-2")
    argv, _exe = _lean_handed(tmp_path)
    w = tmp_path / "serve" / "sampled-proofs-served" / "window"
    monkeypatch.setenv("SEEN", str(tmp_path / "seen.json"))
    other = {**_accepted()["answer"], "protocol": "verity/sampled-proofs/v1", "commitment": "00" * 64}
    monkeypatch.setenv("ANSWER", "" if change == "no-line" else json.dumps(other))
    if change == "no-v2":
        (w / "case-v2.json").unlink()
    if change == "no-blob":
        (w / "m0" / "p0.bin").unlink()
    _rc, line, rows = _recorded(tmp_path, argv, capsys)
    row = rows["integrity"]
    assert row["verdict"] == verdict and said in row["reason"], row
    assert row["answer_from"] == (S.LEAN_HERE if change != "no-v2" else (
        f"{S.AS_WRITTEN} ({S.WRITTEN['integrity']}): the records audit wrote no v2 case (window/case-v2.json), so "
        "there are no proofs to recompute it from"))
    assert line["answers"]["sampled proofs"] == row["answer_from"]


SID = "s20261026T221500Z-ab12"
GATE = "ssh -i KEY -o BatchMode=yes research@gate"


def _h1_serve_dir(tmp_path: Path, coin_source: str = "challenger", tied: bool = True) -> Path:
    """A serve dir whose window 1 served H1 (its line's ``pouw_h1`` and its record) on a gate session."""
    d = tmp_path / "serve"
    (d / "h1" / "w1").mkdir(parents=True)
    _write(d, "windows.jsonl", [{"window": 1, "passed": True, "pouw_h1": {
        "window": 1, "tied": tied, "steps_served": 4, "coin_source": coin_source, "roots1": {"reason": None},
        "gate": {"gate_passed": {"rows": 80}}, "session": {"session": SID, "end": {"closed": True, "complete": True}}}}])
    (d / S.H1_CALLS.format(k=1)).write_text(json.dumps({"commitment": {"session": {"session": SID}}}))
    return d


def _gate(calls: list, files: dict[str, str] | None = None):
    """The gate's answers as `served_verdicts._gate_out` gets them: the session's tree for ``fetch`` (its ``out`` holding
    `files` beside the record), the ledger and pins."""
    def out(gate: str, *words: str) -> bytes:
        calls.append((gate, words))
        if words[0] == "fetch":
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w:gz") as t:
                for name, body in ((f"{words[1]}/request.json", b"{}"), (f"{words[1]}/out/record.json", b"{}"),
                                   *((f"{words[1]}/out/{n}", b.encode()) for n, b in (files or {}).items())):
                    info = tarfile.TarInfo(name)
                    info.size = len(body)
                    t.addfile(info, io.BytesIO(body))
            return buf.getvalue()
        if words[0] == "ledger":
            return b'{"kind": "reserve", "session": "%s", "body": {"by": "vllm"}}\n' % SID.encode()
        return b'{"commit": "aa", "main": "bb", "pinned_utc": "2026-10-26T21:35:00Z"}\n'
    return out


def test_a_development_pouw_piece_checks_the_windows_session_and_credits_nothing(tmp_path, monkeypatch, capsys):
    """``pouw``: the window's session fetched from the gate with the ledger and pins, held by the challenger check; it
    accepts, its row in the record stays missing with its document (it credits nothing), the record composes pending, and
    the pieces' line says it is a development run's."""
    calls, seen = [], []
    monkeypatch.setattr(S, "_gate_out", _gate(calls))

    def check(calls_file, tree, ledger, pins, main, key, out, inputs):
        assert inputs == {}, "a serve dir with no declared Program or period gives credit nothing more"
        seen.append((tree, ledger.read_text(), pins.read_text(), main, key))
        assert (tree / "out" / "record.json").is_file()
        return {"ok": True, "reason": None, "manifest": None, "rounds": 16,
                "credit": {"window": ["budget", "the window is in no period"], "manifest": None}}

    d = _h1_serve_dir(tmp_path)
    doc = S.pouw(d, 1, tmp_path / "pieces" / "pouw", gate=GATE, key="ff" * 64, main=tmp_path / "main.txt", check=check)
    assert (doc["verdict"], doc["reasons"], doc["credited"]) == ("accept", [], False)
    assert doc["credit"]["window"] == ["budget", "the window is in no period"] and doc["check"]["rounds"] == 16
    assert doc["SaltIsLiveCoin"] == {"met": True, "why": S.SALT_MET}
    assert doc["AcceptedLive"] == {"met": False, "why": "credit refused the window (budget: the window is in no "
                                                       "period), so it credited no proof"}
    assert [w for _g, w in calls] == [("fetch", SID), ("ledger",), ("pins",)] and all(g == GATE for g, _w in calls)
    (tree, ledger, pins, main, key), = seen
    assert tree == tmp_path / "pieces" / "pouw" / "session" / SID and SID in ledger and '"commit"' in pins
    assert (main, key) == (tmp_path / "main.txt", "ff" * 64)
    row = S.piece_row("work", tmp_path / "pieces")
    why = f"{S.DEV_WHY}; core's credit refuses the window (budget) the window is in no period"
    assert (row["verdict"], row["why"], row["development"]["verdict"]) == ("missing", why, "accept")
    report, rec, line = S.record(_rows(_accepted(), tmp_path / "pieces"), window=1, steps=(0, 4), node="n", run="r")
    assert line["composed"] == "pending" and "pouw" in line["record_missing"] and line["development"] == {"PoUW": "accept"}
    assert line["verdicts"]["PoUW"] == S.DEV_VERDICT == "development run, credits nothing" and line["missing"]["PoUW"] == why
    jsonschema.Draft202012Validator(SCHEMA).validate(rec)
    assert BR.check(rec) == {} and rec["composed"]["answer"] is None
    reason = report["composed"]["reason"]
    assert reason.startswith(f"sampled proofs: accept; PoUW: {why}; ")
    for name, cmd in (("the bandwidth audit", "BANDWIDTH"), ("PoUS", "POUS"), ("the network certifier", "CERTIFIER"),
                      ("the wipe", "WIPE")):
        assert f"; {name}: missing: {cmd}_CMD unset: " in reason
    env = {"POUW_CMD": f"cp {tmp_path}/pieces/pouw/development.json \"$3/\""}
    got = S.pieces(d, 1, tmp_path / "again", env=env)
    out = capsys.readouterr().out
    assert "POUW (development run, credits nothing) " in out and got["pouw"]["verdict"] == "accept"
    assert (got["pouw"]["session"], got["pouw"]["coins"], got["pouw"]["credited"]) == (SID, None, False)
    assert {p: got["pouw"][p]["met"] for p in S.PREMISES} == {"AcceptedLive": False, "SaltIsLiveCoin": True}


def _pearlplus_run(accepted: bool = True, source: str = S.LIVE_SOURCE) -> tuple[dict, dict]:
    """Pearl+'s served run, `served_passes.py`'s two documents: a warm-up pass, then three timed passes, its coins a live
    session's (``source`` ``challenger``, as `served_step.coins_of` records them) or an in-process challenger's, and
    Pearl+'s verdict over all four, accepting (its credit over the timed passes beside it) or not."""
    live = source == S.LIVE_SOURCE
    served = {"arm": "pouw:pearl-c-h100-v1-h3", "spec": "s.toml", "run": "r", "served_tokens": 32, "generate_s": 2.0,
              "prompt_tokens": 1024, "output_tokens": 1, "ok": True, "over_caps": [],
              "coins": {"source": source, "session": {"session": SID} if live else None,
                        "closed": {"closed": True, "complete": True} if live else None},
              "passes": [{"pass": i, "timed": i > 0, "units": 64, "macs": 1000 * i} for i in range(4)]}
    work = {"credited": 5500, "credited_tiles": 40, "macs": 5600, "rho": "1/100", "epsilon": "1/100"}
    verified = {"draws": 459, "delta": "1/100",
                "line": "ACCEPT verity/pouw/pearl-plus/v1 …" if accepted else "REJECT verity/pouw/pearl-plus/v1: pass 1 tile 7: B opening",
                "answer": {"verdict": "accept", "reason": None, "public": {"work": work}} if accepted else
                          {"verdict": "reject", "reason": "pass 1 tile 7: B opening", "public": None},
                **({"timed": {"passes": 3, "macs": 5400, "credited": 5300}} if accepted else {})}
    return served, verified


@pytest.fixture
def live(monkeypatch, tmp_path):
    """`served_verdicts.pearlplus_coins` holding every pass's salt to the live session its serve names; the gate, key pin
    and main's line each call got, and those to give `pearlplus` (``LIVE_ARGS``)."""
    calls = []

    def check(served, out, gate, key, main):
        calls.append((gate, key, main))
        return {"session": SID, "closed": served["coins"]["closed"], "passes": 4, "reason": None}
    monkeypatch.setattr(S, "pearlplus_coins", check)
    return calls, {"gate": GATE, "key": "ff" * 64, "main": tmp_path / "main.txt"}


def test_a_pearlplus_piece_replays_its_own_served_run_and_credits_nothing(tmp_path, capsys, live):
    """``pearlplus``: Pearl+'s own served run, accepted by Pearl+'s verdict over its passes and its epoch's salt its live
    session's, is accepted with that verdict's line and credit and its shares (the timed passes' MACs and their credit at
    θ over the timed call's wall), and its row in the record stays missing, naming the first condition of record it fails
    (a compose with no key pin), so the record composes pending; the pieces' line says it's a development run's, and
    names the session."""
    calls, args = live
    served, verified = _pearlplus_run()
    doc = S.pearlplus(tmp_path / "pieces" / "pouw", served, verified, **args)
    assert (doc["verdict"], doc["reasons"], doc["credited"], doc["why"]) == ("accept", [], False, S.PEARLPLUS_WHY)
    assert calls == [(GATE, "ff" * 64, tmp_path / "main.txt")] and doc["coin_source"] == S.LIVE_SOURCE
    assert doc["coins"] == {**S.PEARLPLUS_COINS, "check": {"session": SID, "closed": served["coins"]["closed"],
                                                           "passes": 4, "reason": None}}
    assert (doc["scheme"], doc["passes"], doc["line"]) == ("pearl-c-h100-v1-h3", 4, verified["line"])
    assert doc["credit"] == verified["answer"]["public"]["work"]
    tensor = doc["shares"]["readings"]["tensor"]
    rho = 4096 * 132 * 1830e6
    assert doc["shares"]["macs"] == {"declared_linears": 6000, "credited": 5300} and doc["shares"]["verdict"] == "accept"
    assert doc["shares"]["credit"] == {"rho": "1/100", "epsilon": "1/100", "delta": "1/100", "draws": 459}
    assert (tensor["declared_linears"], tensor["credited"]) == (6000 / rho / 2.0, 5300 / rho / 2.0)
    row = S.piece_row("work", tmp_path / "pieces")
    why = f"{S.DEV_VERDICT}: the compose holds the coins to no key pin (--key-sha512, H1_KEY_SHA512)"
    assert (row["verdict"], row["why"], row["development"]["verdict"]) == ("missing", why, "accept")
    report, rec, line = S.record(_rows(_accepted(), tmp_path / "pieces"), window=1, steps=(0, 4), node="n", run="r")
    assert line["composed"] == "pending" and line["development"] == {"PoUW": "accept"}
    assert line["missing"]["PoUW"] == why and BR.check(rec) == {}
    assert report["composed"]["reason"].startswith(f"sampled proofs: accept; PoUW: {why}; ")
    got = S.pieces(tmp_path, 1, tmp_path / "again", env={"POUW_CMD": f"cp {tmp_path}/pieces/pouw/development.json \"$3/\""})
    assert "POUW (development run, credits nothing) " in capsys.readouterr().out
    assert (got["pouw"]["verdict"], got["pouw"]["passes"], got["pouw"]["credited"]) == ("accept", 4, False)
    assert (got["pouw"]["coin_source"], got["pouw"]["session"], got["pouw"]["coins"]) == (S.LIVE_SOURCE, SID, None)
    assert got["pouw"]["credit"] == {"credited": 5500, "macs": 5600, "epsilon": "1/100"}
    assert got["pouw"]["shares_at_tensor_clock"]["declared_linears"] == tensor["declared_linears"]


def test_a_pearlplus_piece_sets_the_unretained_serve_of_its_load_beside_its_shares(tmp_path, capsys, live):
    """The same load served with no pass retained (``serve --no-retain``) gives the shares' ``unretained``: the retained
    run's declared MACs over its wall at each reading, and the walls' ratio. One that served another load, or retained its
    passes, is no comparison: the shares say why, and the verdict is the run's and its replay's alone."""
    calls, args = live
    served, verified = _pearlplus_run()
    served["engine"] = {"max_num_batched_tokens": 8192}
    unretained = {**served, "retained": False, "generate_s": 0.5, "passes": []}
    doc = S.pearlplus(tmp_path / "pieces" / "pouw", served, verified, unretained, **args)
    un, rho = doc["shares"]["unretained"], 4096 * 132 * 1830e6
    assert doc["verdict"] == "accept" and (un["wall_s"], un["retained_over_unretained"]) == (0.5, 4.0)
    assert un["readings"]["tensor"]["declared_linears"] == 6000 / rho / 0.5
    assert S.development_line(doc)["unretained"] == {"generate_s": 0.5, "declared_linears_at_tensor_clock": 6000 / rho / 0.5}
    assert S.development_line(S.pearlplus(tmp_path / "plain", served, verified, **args))["unretained"] is None
    for other, why in (({"engine": {"max_num_batched_tokens": 4096}}, "served another load"),
                       ({"prompt_tokens": 512}, "served another load"), ({"retained": True}, "retained its passes")):
        doc = S.pearlplus(tmp_path / "pieces" / "pouw", served, verified, {**unretained, **other}, **args)
        assert doc["verdict"] == "accept" and why in doc["shares"]["unretained"]["error"]
        assert why in S.development_line(doc)["unretained"]
    (tmp_path / "served.json").write_text(json.dumps(served))
    (tmp_path / "v.json").write_text(json.dumps(verified))
    (tmp_path / "un.json").write_text(json.dumps(unretained))
    out, calls[:] = tmp_path / "cli", []
    assert S.main(["pearlplus", str(out), "--served", str(tmp_path / "served.json"), "--verify", str(tmp_path / "v.json"),
                   "--unretained", str(tmp_path / "un.json"), "--gate", GATE, "--key-sha512", "ff" * 64,
                   "--main", str(tmp_path / "main.txt")]) == 0
    line = json.loads(capsys.readouterr().out.removeprefix("POUW-DEV "))
    assert line["verdict"] == "accept" and line["unretained"]["generate_s"] == 0.5
    assert calls == [(GATE, "ff" * 64, tmp_path / "main.txt")]


def test_a_pearlplus_piece_sets_the_provers_replay_beside_its_shares(tmp_path, capsys, live):
    """The prover's replay of every tile before the coin (``served_passes.py volunteer``'s seconds on its processes) is a
    CPU term beside the GPU shares: the cores it takes to keep pace with the timed call's wall, with the node it ran on
    and how many of the passes it finished; its give-up names the CPU's counts. One whose document has no seconds says
    why, and neither decides the verdict."""
    _calls, args = live
    served, verified = _pearlplus_run()
    volunteered = {"s": 110.0, "jobs": 32, "ok": False, "node": "vy-dc-h200", "of": 4, "passes": [{"pass": 0}] * 3}
    doc = S.pearlplus(tmp_path / "pieces" / "pouw", served, verified, volunteered=volunteered, **args)
    assert doc["verdict"] == "accept" and doc["shares"]["replay"] == {"s": 110.0, "cpus": 32, "core_s": 3520.0, "of_wall": 55.0,
                                                                      "cores_to_keep_pace": 1760.0, "device_s": None,
                                                                      "node": "vy-dc-h200", "finished": 3, "of": 4}
    assert S.development_line(doc)["replay"] == {"s": 110.0, "cpus": 32, "cores_to_keep_pace": 1760.0, "device_s": None,
                                                 "node": "vy-dc-h200", "finished": 3, "of": 4}
    assert (doc["give_up"]["counts"], doc["give_up"]["cross_check"], doc["give_up"]["finished"]) == ("cpu", None, 3)
    plain = S.pearlplus(tmp_path / "plain", served, verified, **args)
    assert plain["give_up"] is None and S.development_line(plain)["replay"] is None
    doc = S.pearlplus(tmp_path / "pieces" / "pouw", served, verified, volunteered={"jobs": 32}, **args)
    assert doc["verdict"] == "accept" and "KeyError" in S.development_line(doc)["replay"]
    for name, d in (("served.json", served), ("v.json", verified), ("vol.json", {"s": 110.0, "jobs": 32})):
        (tmp_path / name).write_text(json.dumps(d))
    assert S.main(["pearlplus", str(tmp_path / "cli"), "--served", str(tmp_path / "served.json"), "--verify",
                   str(tmp_path / "v.json"), "--volunteer", str(tmp_path / "vol.json"), "--gate", GATE, "--key-sha512",
                   "ff" * 64, "--main", str(tmp_path / "main.txt")]) == 0
    assert json.loads(capsys.readouterr().out.removeprefix("POUW-DEV "))["replay"]["cores_to_keep_pace"] == 1760.0


def test_a_pearlplus_piece_names_its_provers_give_up_on_the_card_its_cross_check_and_the_rows_given_up(tmp_path, live):
    """With the card's counts (``served_passes.py volunteer --device``) the piece and its line name the give-up: the
    debit cubin, the rate at which tiles were replayed on the CPU too and how many were, the A and B rows given up and
    the tiles over a cap before and left after. Its replay term is its CPU seconds alone, the card's beside them."""
    _calls, args = live
    served, verified = _pearlplus_run()
    device = {"cubin": "/s/debit/debit_sm90.cubin", "cubin_sha256": "ab" * 32, "cross_check": 1024, "replayed": 9,
              "bad_tiles": 1, "form_s": 50.0, "device_s": 30.0, "replay_s": 20.0}
    volunteered = {"s": 110.0, "jobs": 32, "ok": True, "node": "vy-dc-h200", "of": 3, "passes": [{"pass": 0}] * 3,
                   "device": device, "tiles": 8000, "over": 6, "left": 0, "voluntary_a": 4, "voluntary_b": 2}
    doc = S.pearlplus(tmp_path / "pieces" / "pouw", served, verified, volunteered=volunteered, **args)
    line = S.development_line(doc)
    assert doc["verdict"] == "accept" and doc["give_up"] == line["give_up"] == {
        "counts": "device", "cubin_sha256": "ab" * 32, "cross_check": 1024, "replayed": 9, "bad_tiles": 1, "a_rows": 4,
        "b_rows": 2, "tiles": 8000, "over": 6, "left": 0, "finished": 3, "of": 3, "ok": True}
    assert line["replay"] == {"s": 80.0, "cpus": 32, "cores_to_keep_pace": 1280.0, "device_s": 30.0, "node": "vy-dc-h200",
                              "finished": 3, "of": 3}


def test_a_published_serve_dir_carries_none_of_the_give_ups_counts(tmp_path, live):
    """The piece's give-up document (`give_up`: the A and B rows given up, the tiles over a cap before and left after,
    those the card marks non-finite and those replayed on the CPU, the passes finished) is the retained values', and
    `publish` holds it whole: in ``development.json``, in the piece's line (its log's and the verdicts'), in the record's
    row, in a JSON string and in an object cut short. So are the same counts in text: ``served_passes.py volunteer``'s
    line, a pass's with its rows by call, and the tiles a round ``retain`` dropped left over a cap. No published file
    names one of the counts; each file is in full under the private dir. What the grid form publishes of the give-up
    stays: the work's padded ``give_up``, its draws over a cap and its ε. A piece with no give-up says so (None)."""
    served, verified = _pearlplus_run()
    work = {**verified["answer"]["public"]["work"], "over_cap_draws": 2, "give_up": 335_544_320}
    verified = {**verified, "answer": {**verified["answer"], "public": {**verified["answer"]["public"], "work": work}}}
    counts = {"tiles": 800_081, "over": 60_631, "left": 7_013, "voluntary_a": 40_417, "voluntary_b": 20_219}
    device = {"cubin": "/s/debit/debit_sm90.cubin", "cubin_sha256": "ab" * 32, "cross_check": 1024, "replayed": 90_917,
              "bad_tiles": 11_093, "form_s": 50.0, "device_s": 30.0, "replay_s": 20.0}
    volunteered = {"s": 110.0, "jobs": 32, "ok": False, "node": "vy-dc-h200", "of": 3, "passes": [{"pass": 0}] * 3,
                   "device": device, **counts}
    out, private = tmp_path / "out", tmp_path / "serve" / "private" / "out"
    doc = S.pearlplus(out / "pieces" / "pouw", served, verified, volunteered=volunteered, **live[1])
    line = S.development_line(doc)
    own = [str(v) for v in (*counts.values(), device["replayed"], device["bad_tiles"])]
    assert all(str(v) in json.dumps(doc["give_up"]) for v in own) and line["give_up"] == doc["give_up"]
    (out / "pieces" / "pouw" / "piece.log").write_text(
        "POUW volunteer: served_passes.py volunteer --device exit 1 after 12 min: volunteer FAIL: 3 passes, 800081 tiles "
        "counted on the device (90917 replayed on the CPU too) in 110 s on 32 processes, 60631 over a cap; gave up 40417 A "
        "rows and 20219 B rows; 7013 tiles left over a cap\n"
        "POUW volunteer: served_passes.py volunteer on the CPU exit 124 after 60 min: pass-00002: FAIL: 800081 tiles "
        "replayed, 60631 over a cap; gave up 40417 A rows (call 3: [40417, 5]) and 20219 B rows (weight 5: 20219); 7013 "
        "tiles left over a cap; credit kept 0.997531; records root ab -> cd\n"
        "POUW retain: served_passes.py retain exit 0: retain PASS: 2 of 3 passes with calls retained, 1 dropped before "
        "the coin (credit nothing, declared still); round 2: 7013 tiles left over a cap after its give-up\n"
        f"POUW-DEV {json.dumps(line)}\n"
        f"POUW {json.dumps({'stdout': json.dumps({'give_up': doc['give_up'], 'credit': work})})}\n"
        f"POUW {json.dumps(line)[:json.dumps(line).index('a_rows') + 12]}\n")
    (out / "verdicts.txt").write_text(f"POUW (development run, credits nothing) {json.dumps(line, sort_keys=True)}\n")
    (out / "record").mkdir()
    (out / "record" / "record.json").write_text(json.dumps({"rows": [{"protocol": "pouw", "development": doc}]}, default=str))
    full = {p.relative_to(out).as_posix(): p.read_text() for p in out.rglob("*") if p.is_file()}
    assert S.publish(out, private) == sorted(full)
    for rel, text in full.items():
        shown = (out / rel).read_text()
        assert (private / rel).read_text() == text and not [v for v in own if v in shown], rel
        assert "a_rows" not in shown and "bad_tiles" not in shown and "credit kept" not in shown, rel
    pub = json.loads((out / "pieces" / "pouw" / "development.json").read_text())
    assert "give_up" not in pub and pub["credit"] == work and pub == {k: v for k, v in json.loads(
        full["pieces/pouw/development.json"]).items() if k != "give_up"}
    log = (out / "pieces" / "pouw" / "piece.log").read_text().splitlines()
    assert log[0].endswith("volunteer FAIL: " + S.GAVE_UP) and log[1].endswith("pass-00002: FAIL: " + S.GAVE_UP)
    assert log[2].endswith("round 2: tiles left over a cap after its give-up")
    shown = json.loads(log[3].removeprefix("POUW-DEV "))
    assert shown == {k: v for k, v in line.items() if k != "give_up"} and shown["credit"]["epsilon"] == work["epsilon"]
    assert json.loads(json.loads(log[4].removeprefix("POUW "))["stdout"]) == {"credit": work} and log[5] == "POUW " + S.CUT
    assert S.publish(out, private) == []
    plain = S.pearlplus(tmp_path / "plain", served, verified, **live[1])
    assert S.public_json(plain) == plain and plain["give_up"] is None
    assert S.public_json({"work": work}) == {"work": work}


def test_a_pearlplus_piece_names_the_rounds_its_prover_dropped_before_the_coin(tmp_path, live):
    """A verdict over a retention that dropped rounds before the coin (``served_passes.py retain``) names them in the
    piece and its line: the verdict is the retained passes', and the dropped ones stay in the declared share."""
    served, verified = _pearlplus_run()
    doc = S.pearlplus(tmp_path / "pieces" / "pouw", served, {**verified, "rounds": [0, 1, 3], "dropped": [2]}, **live[1])
    assert (doc["verdict"], doc["dropped"], S.development_line(doc)["dropped"]) == ("accept", [2], [2])
    assert doc["shares"]["macs"]["declared_linears"] == 6000
    assert S.development_line(S.pearlplus(tmp_path / "all", served, verified, **live[1]))["dropped"] is None


DSID = "s20261104T090000Z-dr01"


def _draw(after: str = "c1" * 64, seed: int = 7) -> dict:
    """A challenger draw as the coin record carries it (`bench_arm.LiveDraw`): a real draw session's commitment and its
    coin ``pouw/draw`` 0 opened after ``after``, reserved as `DSID`."""
    import random

    from verity.protocols.pouw.verifier import window as Wd

    ch = Wd.draw_challenger(0, random.Random(seed).randbytes)
    opening = ch.open(Wd.DRAW, 0, bytes.fromhex(after))
    return {"from": "challenger", "label": Wd.DRAW, "index": 0,
            "reserved": {"session": DSID, "seq": 9, "by": "vllm", "period": None},
            "session": {"session": DSID, "seq": 10, "root": after}, "commitment": ch.commitment.to_json(),
            "opening": opening.to_json()}


DRAW = _draw()
DRAW_COIN = DRAW["opening"]["coin"]
NODE = {"from": "node", "claim": "uniform/io-getrandombytes"}


def _coin_record(source: str = "5e" * 32, draw: dict | None = None, commitment: str = "c1" * 64) -> bytes:
    """The verifier's coin record as ``served_passes.py verify`` writes it before any replay (``DIR/coin.json``), its
    draw the node's unless ``draw`` names a challenger's."""
    return (json.dumps({"schema": S.COIN_KIND, "commitment": commitment, "source": source, "passes": ["p0", "p1"],
                        "draw": draw or NODE}, indent=1) + "\n").encode()


def _put(answers: dict) -> tuple:
    """A stand-in for ``research data put``, answering by whether a call asks ``--preserve`` with (returncode, stdout,
    stderr) or an exception to raise; every call's argv and keywords in ``calls``."""
    calls = []

    def put(argv, **kw):
        calls.append((argv, kw))
        got = answers["--preserve" in argv]
        if isinstance(got, BaseException):
            raise got
        return subprocess.CompletedProcess(argv, *got)
    return put, calls


NODE_PUT = {True: (1, "", "Traceback (most recent call last):\nStoreError: no remote configured"),
            False: (0, json.dumps({"id": "art:bb", "preserve": None, "pending": "write-through off"}), "")}


@pytest.mark.parametrize("case", ["preserved", "node", "no store"])
def test_coin_puts_the_verifiers_coin_record_in_the_store_the_moment_it_is_drawn(tmp_path, case):
    """`coin` copies the verifier's coin record out of the passes and puts it in the evidence store as its own kind, its
    meta the run and the run commitment: preserved where the put pushes it; on a node with no store write key, in the
    node's own store under the same content-addressed id, with what the preserving put said; and with no id where no put
    answers. Its answer is ``coin-put.json`` beside the copy."""
    (rec := tmp_path / "passes" / "coin.json").parent.mkdir()
    rec.write_bytes(raw := _coin_record())
    put, calls = _put({"preserved": {True: (0, json.dumps({"id": "art:aa", "preserve": {"preserved": True}}), "")},
                       "node": NODE_PUT,
                       "no store": {True: FileNotFoundError(2, "No such file", "research"),
                                    False: FileNotFoundError(2, "No such file", "research")}}[case])
    got = S.coin(rec, out := tmp_path / "pieces" / "pouw", "r1", research=["research"], put=put)
    art, preserved, error = {"preserved": ("art:aa", True, None),
                             "node": ("art:bb", False, "research data put --preserve: StoreError: no remote configured"),
                             "no store": (None, False, "research data put --preserve: FileNotFoundError: [Errno 2] No "
                                                       "such file: 'research'")}[case]
    assert got == {"sha512": hashlib.sha512(raw).hexdigest(), "commitment": "c1" * 64, "source": "5e" * 32, "run": "r1",
                   "draw": "node", "art": art, "preserved": preserved, "error": error, "custody": None}
    assert (out / S.COIN_RECORD).read_bytes() == raw and json.loads((out / S.COIN_PUT).read_text()) == got
    meta = json.dumps({"commitment": "c1" * 64, "run": "r1"})
    assert calls[0] == (["research", "data", "put", "--kind", S.COIN_KIND, "--meta", meta, "--file", str(out / "coin.json"),
                         "--json", "--preserve"],
                        {"capture_output": True, "text": True, "check": False, "timeout": S.PUT_TIMEOUT_S})
    assert [argv for argv, _ in calls[1:]] == ([] if preserved else [calls[0][0][:-1]])


def test_coin_refuses_a_record_not_the_verifiers_and_a_second_coin_in_one_piece(tmp_path):
    """A piece publishes the one coin its verifier drew: a record of another schema, or without its commitment or source,
    is refused before anything is written, and so is a coin other than the one the piece already published, which stays;
    the same coin again is the same put."""
    put, calls = _put({True: (0, json.dumps({"id": "art:aa"}), "")})
    rec, out = tmp_path / "coin.json", tmp_path / "pieces" / "pouw"
    for bad in ({"schema": "other", "commitment": "c1", "source": "5e"}, {"schema": S.COIN_KIND, "source": "5e"}, [1]):
        rec.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match="is no verity-pouw/pearl-plus-coin/v2 record"):
            S.coin(rec, out, "r1", research=["research"], put=put)
    assert not out.exists()
    rec.write_bytes(first := _coin_record())
    assert S.coin(rec, out, "r1", research=["research"], put=put) == S.coin(rec, out, "r1", research=["research"], put=put)
    rec.write_bytes(_coin_record(source="77" * 32))
    with pytest.raises(ValueError, match="already holds another coin"):
        S.coin(rec, out, "r1", research=["research"], put=put)
    assert (out / S.COIN_RECORD).read_bytes() == first and len(calls) == 2


def test_the_coin_command_says_where_the_verifiers_coin_went(tmp_path, capsys, monkeypatch):
    """``served_verdicts.py coin`` prints one COIN line, the run commitment the coin was drawn on, the record's digest and
    where it went: preserved, in the node's store alone (and the put of the run's custody copy that preserves that id),
    or in no store; a record that isn't the verifier's is refused, exit 1."""
    (rec := tmp_path / "coin.json").write_bytes(raw := _coin_record())
    real, head = S.coin, f"COIN drawn on run commitment {'c1' * 8}..., record sha512 {hashlib.sha512(raw).hexdigest()[:16]}..."
    for i, (answers, said) in enumerate((
            ({True: (0, json.dumps({"id": "art:aa"}), "")}, "preserved in the evidence store as art:aa"),
            (NODE_PUT, "art:bb in this node's store, not preserved (research data put --preserve: StoreError: no remote "
                       "configured): a put of the run's custody copy of coin.json as verity-pouw/pearl-plus-coin/v2 with "
                       "meta {run, commitment} preserves that id"),
            ({True: OSError("gone"), False: OSError("gone")},
             "in no store (research data put --preserve: OSError: gone): coin.json goes with the run's custody"))):
        monkeypatch.setattr(S, "coin", lambda *a, answers=answers, **k: real(*a, **k, put=_put(answers)[0]))
        assert S.main(["coin", str(tmp_path / f"out{i}"), "--record", str(rec), "--run", "r1"]) == 0
        assert capsys.readouterr().out == f"{head}: {said}\n"
    rec.write_text("{}")
    assert S.main(["coin", str(tmp_path / "out9"), "--record", str(rec)]) == 1
    assert capsys.readouterr().out.startswith("COIN refused: ValueError: ")


def test_the_coin_names_the_runs_attempt_as_its_custody_off_the_node(tmp_path, capsys, monkeypatch):
    """Where no put preserves the coin (vy-dc-h200's workload shells hold no store key), ``coin-put.json`` names the run's
    Attempt, which carries ``out/``, and the coin's path in it: only for a valid run id and a copy under the run's
    ``out/``; the COIN line says so."""
    (rec := tmp_path / "coin.json").write_bytes(_coin_record(DRAW_COIN, DRAW))
    run_dir, run = tmp_path / "runs" / "r20261104-090000-ab12", "r20261104-090000-ab12"
    put = _put(NODE_PUT)[0]
    got = S.coin(rec, run_dir / "out" / "pieces" / "pouw", run, research=["research"], put=put, run_dir=run_dir)
    assert (got["draw"], got["custody"]) == ("challenger", {"attempt": run, "path": "out/pieces/pouw/coin.json"})
    for other, kw in ((tmp_path / "x" / "pouw", {"run_dir": run_dir}), (run_dir / "scratch", {"run_dir": run_dir}),
                      (run_dir / "out" / "a", {"run_dir": None})):
        assert S.coin(rec, other, run, research=["research"], put=put, **kw)["custody"] is None
    assert S.coin(rec, run_dir / "out" / "b", "local", research=["research"], put=put, run_dir=run_dir)["custody"] is None
    real = S.coin
    monkeypatch.setattr(S, "coin", lambda *a, **k: real(*a, **k, put=put))
    assert S.main(["coin", str(run_dir / "out" / "c"), "--record", str(rec), "--run", run, "--run-dir", str(run_dir)]) == 0
    assert capsys.readouterr().out.rstrip().endswith(f"; the run's Attempt {run} carries it at out/c/coin.json")


def test_a_pearlplus_piece_holds_its_accept_to_the_coin_it_published(tmp_path, capsys, live):
    """``--coin`` (`coin`'s ``coin-put.json``) holds Pearl+'s accept to the coin published at its draw: the piece and its
    line carry the coin's store id, whether the put preserved it and the record's digest, and an accept whose run
    commitment or draw source isn't the published record's, a copy that isn't the one put, or a coin never published
    rejects, while a verdict that rejects keeps its own reasons. The piece names its load."""
    served, verified = _pearlplus_run()
    served.update(requests=16, engine={"max_num_batched_tokens": 8192})
    (rec := tmp_path / "coin.json").write_bytes(_coin_record())
    pub = tmp_path / "pub"
    published = S.coin(rec, pub, "r1", research=["research"], put=_put(NODE_PUT)[0])
    answer = {**verified["answer"], "commitment": "c1" * 64, "profile": {"source": "5e" * 32}}
    held, given = {**verified, "answer": answer}, {"coin": pub / S.COIN_PUT, **live[1]}
    doc = S.pearlplus(tmp_path / "a", served, held, **given)
    assert (doc["verdict"], doc["reasons"]) == ("accept", [])
    coin = {"art": "art:bb", "preserved": False, "error": published["error"], "sha512": published["sha512"]}
    assert doc["coin"] == {**coin, "reason": None} and S.development_line(doc)["coin"] == coin
    load = {"requests": 16, "prompt_tokens": 1024, "output_tokens": 1, "max_num_batched_tokens": 8192}
    assert doc["load"] == S.development_line(doc)["load"] == load
    for name, d in (("served.json", served), ("v.json", held)):
        (tmp_path / name).write_text(json.dumps(d))
    assert S.main(["pearlplus", str(tmp_path / "cli"), "--served", str(tmp_path / "served.json"), "--verify",
                   str(tmp_path / "v.json"), "--gate", GATE, "--key-sha512", "ff" * 64, "--main", str(tmp_path / "main.txt"),
                   "--coin", str(pub / S.COIN_PUT)]) == 0
    line = json.loads(capsys.readouterr().out.removeprefix("POUW-DEV "))
    assert (line["verdict"], line["coin"], line["load"]) == ("accept", coin, load)
    for other in ({"commitment": "c2" * 64}, {"profile": {"source": "77" * 32}}):
        doc = S.pearlplus(tmp_path / "a", served, {**verified, "answer": {**answer, **other}}, **given)
        assert doc["verdict"] == "reject" and doc["reasons"] == [f"coin: {doc['coin']['reason']}"]
        assert doc["coin"]["reason"].startswith("the verdict's run commitment or draw source is not the published coin record's")
    unpublished = {**given, "coin": tmp_path / "none" / S.COIN_PUT}
    doc = S.pearlplus(tmp_path / "a", served, held, **unpublished)
    assert doc["verdict"] == "reject" and doc["reasons"][0].startswith("coin: the verifier's coin was not published: FileNotFoundError")
    doc = S.pearlplus(tmp_path / "a", served, _pearlplus_run(accepted=False)[1], **unpublished)
    assert doc["reasons"] == ["Pearl+'s verdict rejects the run: pass 1 tile 7: B opening"] and doc["coin"]["reason"]
    (pub / S.COIN_RECORD).write_bytes(_coin_record(source="77" * 32))
    doc = S.pearlplus(tmp_path / "a", served, held, **given)
    assert doc["reasons"] == ["coin: the coin record beside the put is not the one it published"]
    assert S.development_line(S.pearlplus(tmp_path / "plain", served, held, **live[1]))["coin"] is None


@pytest.mark.parametrize("served, verified, reason", [
    (True, False, "Pearl+'s verdict rejects the run: pass 1 tile 7: B opening"),
    (True, None, "no verdict: served_passes.py verify wrote no document"),
    (None, None, "no served run: served_passes.py serve wrote no document"),
])
def test_a_pearlplus_piece_rejects_a_run_pearlpluss_verdict_rejects_and_one_that_wrote_nothing(tmp_path, served, verified,
                                                                                               reason, live):
    """A run Pearl+'s verdict rejects, or that has no verdict, still carries its shares, its declared linears beside a
    credit of 0; a run that served nothing has none."""
    s, v = _pearlplus_run(verified is not False)
    doc = S.pearlplus(tmp_path / "pieces" / "pouw", s if served else None, None if verified is None else v, **live[1])
    assert (doc["verdict"], doc["reasons"][0], doc["why"]) == ("reject", reason, S.PEARLPLUS_WHY)
    if served:
        rho = 4096 * 132 * 1830e6
        assert doc["shares"]["macs"] == {"declared_linears": 6000, "credited": 0}
        assert doc["shares"]["verdict"] == (None if verified is None else "reject")
        assert S.development_line(doc)["shares_at_tensor_clock"] == {"declared_linears": 6000 / rho / 2.0, "credited": 0}
    else:
        assert doc["shares"] is None
    row = S.piece_row("work", tmp_path / "pieces")
    assert (row["verdict"], row["reason"]) == ("reject", "; ".join(doc["reasons"]))


def test_the_pearlplus_command_writes_a_reject_where_a_step_wrote_no_document(tmp_path, capsys):
    """`served_pearlplus.sh` names both documents whether or not its steps wrote them: a serve that wrote neither gets a
    piece that rejects, naming both, and its POUW-DEV line."""
    out = tmp_path / "pieces" / "pouw"
    assert S.main(["pearlplus", str(out), "--served", str(tmp_path / "served.json"), "--verify", str(tmp_path / "v.json")]) == 0
    doc = json.loads((out / S.DEVELOPMENT).read_text())
    assert (doc["verdict"], doc["reasons"]) == ("reject", ["no served run: served_passes.py serve wrote no document",
                                                           "no verdict: served_passes.py verify wrote no document"])
    line = json.loads(capsys.readouterr().out.removeprefix("POUW-DEV "))
    assert (line["verdict"], line["credited"], line["passes"]) == ("reject", False, None)


PUBLIC = bytes(range(80))
LEDGER_LINE = '{"kind": "reserve", "session": "%s", "body": {"by": "vllm"}}' % SID
PINS_LINE = '{"commit": "aa", "main": "bb", "pinned_utc": "2026-10-26T21:35:00Z"}'


@pytest.mark.parametrize("case, reason", [
    ("honest", None),
    ("refused", "pass 2: coin: the open's pin is not on main's first-parent line"),
    ("other key", "CoinError: the challenger key at "),
    ("no gate", "ValueError: no gate, key pin or main's first-parent line to check the session by (--gate, "
                "--key-sha512, --main)"),
    ("no session", "ValueError: the served run names no live session"),
    ("no calls", "the served run has no pass with calls to check"),
    ("gate refuses", "RuntimeError: the gate refused fetch (exit 1): no such session"),
])
def test_a_pearlplus_piece_holds_each_pass_salt_to_the_live_session_the_gate_holds(tmp_path, monkeypatch, case, reason):
    """`pearlplus_coins`: the serve's live session fetched through the gate with the auditor's ledger and the host's pins,
    the session's key held to the deployment's pin, and every pass with calls held by `core_check.live_salt` to them, to
    its own manifest and to main's first-parent line, so the piece accepts; the first refusal names its pass and no pass
    after it is checked, and a key other than the pin's, no gate, key pin or line, a run that names no session or has no
    pass with calls, and a gate that refuses the fetch each reject, naming why."""
    from verity.protocols.pouw.verifier import core_check as CC
    from verity.protocols.spot.device import challenger as Ch
    from verity.protocols.spot.device import remote

    files = {remote.KEY: json.dumps({"H": 5, "W": 4, "public": PUBLIC.hex()}), remote.RELEASES: '{"event": "root"}\n',
             remote.CLOSE: "{}"}
    calls, seen = [], []

    def refusing(gate: str, *words: str) -> bytes:
        raise RuntimeError("the gate refused fetch (exit 1): no such session")
    monkeypatch.setattr(S, "_gate_out", refusing if case == "gate refuses" else _gate(calls, files))

    def live_salt(key, ledger, pins, main, session, manifest, verify):
        assert Ch.ChallengerKey.from_json(key).public == PUBLIC and verify is CC.lms_verify
        assert (ledger, pins, main) == ([LEDGER_LINE], [PINS_LINE], ["cc" * 20, "aa" * 20])
        assert session == {"close": {}, "hellos": None, "log": ['{"event": "root"}'], "record": {}}
        seen.append(manifest["pass_index"])
        if case == "refused" and manifest["pass_index"] == 2:
            return None, ("coin", "the open's pin is not on main's first-parent line")
        return b"s" * 32, None
    monkeypatch.setattr(CC, "live_salt", live_salt)
    served, verified = _pearlplus_run()
    for p in served["passes"]:
        (m := tmp_path / f"manifest-{p['pass']}.json").write_text(json.dumps({"pass_index": p["pass"]}))
        p.update(manifest=str(m), units=0 if case == "no calls" else p["units"])
    served["passes"].append({"pass": 4, "timed": True, "units": 0, "macs": 0})
    if case == "no session":
        served["coins"]["session"] = None
    (main := tmp_path / "main.txt").write_text("cc" * 20 + "\n" + "aa" * 20 + "\n")
    pin = "00" * 64 if case == "other key" else hashlib.sha512(PUBLIC).hexdigest()
    out = tmp_path / "pieces" / "pouw"
    doc = S.pearlplus(out, served, verified, gate=None if case == "no gate" else GATE, key=pin, main=main)
    got = doc["coins"]["check"]
    assert (got["session"], got["closed"]) == (None if case == "no session" else SID, served["coins"]["closed"])
    assert got["reason"] == reason if case != "other key" else got["reason"].startswith(reason), got
    checked = {"honest": [0, 1, 2, 3], "refused": [0, 1, 2]}.get(case, [])
    assert (seen, got["passes"]) == (checked, len(checked))
    assert (doc["verdict"], doc["reasons"]) == (("accept", []) if reason is None else ("reject", [f"coins: {got['reason']}"]))
    assert S.development_line(doc)["coins"] == got["reason"] and S.development_line(doc)["session"] == got["session"]
    if case in ("honest", "refused", "other key", "no calls"):
        assert [w for _g, w in calls] == [("fetch", SID), ("ledger",), ("pins",)] and all(g == GATE for g, _w in calls)
        assert (out / "ledger.jsonl").read_text() == LEDGER_LINE + "\n" and (out / "pins.jsonl").read_text() == PINS_LINE + "\n"
        assert json.loads((out / "session" / SID / "out" / remote.KEY).read_text())["public"] == PUBLIC.hex()
    else:
        assert calls == []


def test_a_pearlplus_piece_on_in_process_coins_claims_no_verdict(tmp_path, monkeypatch, capsys):
    """A Pearl+ run whose epoch's coins are an in-process challenger's, which the serving process holds (its serve without
    PEARLC_CHALLENGER_GATE), is held to nothing: where nothing failed its verdict is unclaimed, its row missing with
    `PEARLPLUS_WHY_IN_PROCESS`, the record's line naming it, and `pieces` counting it no failure; the same run given the
    gate, which its serve then didn't take its coins from, rejects."""
    served, verified = _pearlplus_run(source="in-process")
    doc = S.pearlplus(tmp_path / "pieces" / "pouw", served, verified)
    assert (doc["verdict"], doc["reasons"], doc["credited"]) == (S.UNCLAIMED, [], False)
    assert (doc["why"], doc["coin_source"], doc["coins"]) == (S.PEARLPLUS_WHY_IN_PROCESS, "in-process",
                                                              S.PEARLPLUS_COINS_IN_PROCESS)
    assert doc["credit"] == verified["answer"]["public"]["work"] and doc["shares"]["verdict"] == "accept"
    row = S.piece_row("work", tmp_path / "pieces")
    assert (row["verdict"], row["why"], row["development"]["verdict"]) == ("missing", S.PEARLPLUS_WHY_IN_PROCESS,
                                                                           S.UNCLAIMED)
    _report, rec, line = S.record(_rows(_accepted(), tmp_path / "pieces"), window=1, steps=(0, 4), node="n", run="r")
    assert (line["composed"], line["development"]) == ("pending", {"PoUW": S.UNCLAIMED})
    assert line["verdicts"]["PoUW"] == S.DEV_VERDICT and line["missing"]["PoUW"] == S.PEARLPLUS_WHY_IN_PROCESS
    assert BR.check(rec) == {}
    monkeypatch.setenv("POUW_CMD", f"cp {tmp_path}/pieces/pouw/development.json \"$3/\"")
    assert S.main(["pieces", "--serve-run", str(tmp_path), "--window", "1", "--out", str(tmp_path / "again"),
                   "--only", "pouw"]) == 0
    got = json.loads(capsys.readouterr().out.strip().removeprefix("POUW (development run, credits nothing) "))
    assert (got["verdict"], got["coin_source"], got["session"], got["coins"]) == (S.UNCLAIMED, "in-process", None,
                                                                                  "unchecked")
    doc = S.pearlplus(tmp_path / "gated", served, verified, gate=GATE, key="ff" * 64, main=tmp_path / "main.txt")
    assert (doc["verdict"], doc["why"]) == ("reject", S.PEARLPLUS_WHY_IN_PROCESS)
    assert doc["reasons"] == ["coins: in-process, not a live session on the gate, so nothing checks them"]


def test_a_pearlplus_piece_over_a_commit_windows_epoch_names_its_arm_and_no_shares(tmp_path):
    """Under PEARLPLUS_TIE the piece reads the window's own epoch, serve_pearl's ``pearl/w<k>/served.json``: it names the
    arm by its ``entry``, and since that epoch times no generate call of its own it has no shares, which is no reason to
    reject; the same doc short of ok still rejects."""
    served, verified = _pearlplus_run(source="in-process")
    window = {"schema": served.get("schema"), "entry": served["arm"], "run": "r", "window": 0, "retained": True,
              "commit_steps": 3, "passes": [{**p, "timed": True} for p in served["passes"][1:]], "passes_with_calls": 3,
              "coins": served["coins"], "ok": True, "error": None}
    doc = S.pearlplus(tmp_path / "w", window, verified)
    assert (doc["verdict"], doc["reasons"], doc["arm"], doc["scheme"]) == (S.UNCLAIMED, [], served["arm"], "pearl-c-h100-v1-h3")
    assert list(doc["shares"]) == ["none"] and doc["passes"] == 3
    assert S.development_line(doc)["shares_at_tensor_clock"] == {"declared_linears": None, "credited": None}
    bad = S.pearlplus(tmp_path / "bad", {**window, "ok": False, "error": "pass 2 retained no manifest"}, verified)
    assert bad["verdict"] == "reject" and bad["shares"] is None


def _tied_epoch(tmp_path: Path, source: str = "in-process",
                reserved: dict[str, str] | None = None) -> tuple[dict, dict, Path]:
    """Window 7's own epoch under PEARLPLUS_TIE: serve_pearl's served.json for it (one round each of its three Commit
    steps, its coins ``source``'s, the gate's reserve answer ``reserved``), Pearl+'s verdict accepting it, and W's file
    (`PEARL_WREC`)."""
    served, verified = _pearlplus_run(source=source)
    window = {"entry": served["arm"], "run": "r", "window": 7, "retained": True, "commit_steps": 3,
              "passes": [{**p, "timed": True} for p in served["passes"][1:]], "passes_with_calls": 3,
              "coins": {**served["coins"], "reserved": reserved}, "ok": True, "error": None}
    (w := tmp_path / "window-record.json").write_text(json.dumps(PEARL_WREC))
    return window, verified, w


def test_a_tied_pearlplus_piece_is_the_windows_own_epoch_and_says_so(tmp_path, capsys):
    """Under PEARLPLUS_TIE (``--tie W``) the piece is the window's own Pearl+ epoch: on in-process coins it claims no
    verdict and says so of the window's epoch (`PEARLPLUS_WHY_TIED_IN_PROCESS`, its epoch
    `PEARLPLUS_EPOCH_TIED_IN_PROCESS`), not of a seeded load of its own after the window's serve; its load is the
    window's steps and its ``tie`` W's digest, which its line names. Its arm's document short of ok, or missing, rejects,
    naming the window's arm."""
    window, verified, w = _tied_epoch(tmp_path)
    doc = S.pearlplus(tmp_path / "a", window, verified, tie=w)
    assert (doc["verdict"], doc["reasons"], doc["why"]) == (S.UNCLAIMED, [], S.PEARLPLUS_WHY_TIED_IN_PROCESS)
    assert doc["coins"] == {**S.PEARLPLUS_COINS_IN_PROCESS, "epoch": S.PEARLPLUS_EPOCH_TIED_IN_PROCESS}
    assert (doc["load"], doc["tie"]) == ({"window_steps": 3}, {"record": REG.digest(PEARL_WREC)})
    for name, d in (("served.json", window), ("v.json", verified)):
        (tmp_path / name).write_text(json.dumps(d))
    assert S.main(["pearlplus", str(tmp_path / "cli"), "--served", str(tmp_path / "served.json"), "--verify",
                   str(tmp_path / "v.json"), "--tie", str(w)]) == 0
    line = json.loads(capsys.readouterr().out.removeprefix("POUW-DEV "))
    assert (line["verdict"], line["tie"], line["load"]) == (S.UNCLAIMED, doc["tie"], {"window_steps": 3})
    assert S.pearlplus(tmp_path / "b", window, verified, tie=tmp_path / "none.json")["tie"]["record"] is None
    bad = S.pearlplus(tmp_path / "c", {**window, "ok": False, "error": "pass 2 retained no manifest"}, verified, tie=w)
    assert bad["reasons"] == ["the window's Pearl+ epoch isn't one round a Commit step: pass 2 retained no manifest"]
    none = S.pearlplus(tmp_path / "d", None, verified, tie=w)
    assert none["reasons"][0] == "no served run: serve-openai's Pearl+ arm wrote no served.json for the window"


@pytest.mark.parametrize("reserved, joined", [
    ({"session": SID, "by": "vllm", "period": "pearlplus-r1"},
     "reserved under lane vllm's key into its Pearl-C period pearlplus-r1"),
    ({"session": SID, "by": "vllm"}, "in no period, its reserve having joined none, so the record can't take it"),
])
def test_a_tied_pearlplus_piece_on_a_live_salt_session_names_the_period_it_joined(tmp_path, live, reserved, joined):
    """Under PEARLPLUS_TIE the window's salt coin is a live session's reserved as the window opened
    (`PEARLPLUS_EPOCH_TIED`), and why the piece credits nothing by itself names the Pearl-C period that session joined,
    as the gate's reserve answer in the window's served.json names it, or says it joined none."""
    _calls, args = live
    window, verified, w = _tied_epoch(tmp_path, S.LIVE_SOURCE, reserved)
    doc = S.pearlplus(tmp_path / "a", window, verified, tie=w, **args)
    assert (doc["verdict"], doc["reasons"], doc["coins"]["epoch"]) == ("accept", [], S.PEARLPLUS_EPOCH_TIED)
    assert doc["why"] == S.pearlplus_why_tied(reserved) and doc["why"].endswith(f"on vy-challenger-1, {joined}")
    assert doc["why"].startswith("development piece, which the record takes as PoUW's verdict of record, the prover "
                                 "side's, only where")


PEARL_W = "d7" * 64
PEARL_LEAN_WORK = {"credited": 5400, "credited_tiles": 40, "macs": 5600, "rho": "1/100", "epsilon": "1/100"}


#: the window-circuit record V3c held R's circuit to, kept beside R (`S.CIRCUIT_RECORD`), which Pearl+'s period opens on
PEARL_RECORD = b'{"name":"toy window circuit cut at its linears"}\n'


def _pearlplus_piece(tmp_path: Path, monkeypatch, live, case: str = "of record") -> tuple[Path, Path, str, str]:
    """Pearl+'s accepted piece as `served_pearlplus.sh` leaves it, every condition of record met unless ``case`` breaks
    one: its live session checked (`live`), the session's tree and the auditor's ledger as the gate gave them (the lane's
    own reserve in its open Pearl-C period, whose circuit is the SHA-512 of the window-circuit record beside R), its coin
    published, Pearl+'s verdict tied to W, the window's registration R chained after W and its audit tied to W, and the
    Lean verdict on the same run. Returns the pieces dir, the sampled
    proofs dir, the compose's key pin and lane."""
    from verity.protocols.spot.device import remote

    served, verified = _pearlplus_run()
    draw = {"node draw": NODE, "draw off C": _draw("c2" * 64)}.get(case, DRAW)
    source = {"node draw": "5e" * 32, "source not coin": "77" * 64}.get(case, DRAW_COIN)
    answer = {"protocol": S.PEARLPLUS_PROTOCOL, "verdict": "accept", "reason": None, "commitment": "c1" * 64,
              "public": {"work": verified["answer"]["public"]["work"], "unit": "MAC",
                         **({} if case == "untied" else {"tie": {"record": PEARL_W, "tables": 6}})},
              "profile": {"delta": "1/100", "draws": 459, "source": source}}
    piece = tmp_path / "pieces" / "pouw"
    piece.mkdir(parents=True)
    (rec := tmp_path / "coin.json").write_bytes(_coin_record(source, draw))
    S.coin(rec, piece, "r1", research=["research"], put=_put(NODE_PUT)[0])
    checked = {"from": draw["from"], "session": DSID if draw is not NODE else None, "reason": None}
    monkeypatch.setattr(S, "pearlplus_draw", lambda *a, **k: dict(checked))
    doc = S.pearlplus(piece, served, {**verified, "answer": answer}, coin=piece / S.COIN_PUT, **live[1])
    assert (doc["verdict"], doc["reasons"], doc["coins"]["draw"]) == ("accept", [], checked)
    doc.update({"in-process coins": {"coin_source": "in-process"},
                "session refused": {"coins": {**doc["coins"], "check": {**doc["coins"]["check"],
                                                                         "reason": "pass 2: coin: bad"}}},
                "draw refused": {"coins": {**doc["coins"], "draw": {**checked,
                                                                     "reason": "ValueError: the record's draw coin: bad"}}},
                "unpublished coin": {"coin": None}}.get(case, {}))
    (piece / S.DEVELOPMENT).write_text(json.dumps(doc))
    (piece / S.PEARLPLUS_ANSWER).write_text(json.dumps(answer))
    (piece / S.RAN).write_text(json.dumps({"server": "vy-dc-h200", "rc": 0, "limit": "120m"}))
    for sid, public in ((SID, PUBLIC), (DSID, bytes(80) if case == "draw key" else PUBLIC)):
        (piece / "session" / sid / "out").mkdir(parents=True)
        (piece / "session" / sid / "out" / remote.KEY).write_text(json.dumps({"H": 5, "W": 4, "public": public.hex()}))
    opens = {"no period": [], "H1 period": [{"kind": "period-open", "body": {"period": "p-h1", "lane": "vllm", "cap": 4,
                                                                            "circuit": "c1" * 64, "draw_k": 27, "rows": 64}}]}
    circuit = hashlib.sha512(b"another circuit" if case == "period on another circuit" else PEARL_RECORD).hexdigest()
    ledger = opens.get(case, [{"kind": "period-open", "body": {"period": "p-pc", "lane": "vllm", "cap": 1,
                                                                "circuit": circuit}}])
    by = {"reserved for": {"by": "vy-admin", "for": "vllm"}}.get(case, {"by": "vllm"})
    ledger += [] if case == "no reserve" else [{"kind": "reserve", "session": SID, "body": by}]
    if case != "period open":
        ledger += [{"kind": "period-close", "session": None, "body": {"period": "p-pc", "lane": "vllm"}}]
    reserve = {"by": "vllm", "families": [["pouw/draw", 1]], **({} if case == "draw not withheld" else {"withheld": True})}
    if case == "drawn again":
        ledger += [{"kind": "reserve", "session": "s-first", "body": {**reserve}}]
    ledger += [{"kind": "reserve", "session": DSID, "body": reserve},
               {"kind": "open", "session": DSID, "body": {"window": {"root": "c1" * 64}, "withheld": True}}]
    if case == "second open":
        ledger += [{"kind": "open", "session": "s-other", "body": {"window": {"root": "c1" * 64}, "withheld": True}}]
    (piece / "ledger.jsonl").write_text("".join(json.dumps(e) + "\n" for e in ledger))
    lean = {"verdict": "accept", "reason": None, "refused": None, "commitment": "c1" * 64, "source": source,
            "drawn": 459, "verdicts": [], "work": PEARL_LEAN_WORK, "tie": {"record": PEARL_W}}
    lean.update({"Lean rejects": {"verdict": "reject", "reason": "the live session's coins are out of scope"},
                 "Lean other run": {"commitment": "c2" * 64}, "Lean untied": {"tie": None},
                 "Lean credits nothing": {"work": {**PEARL_LEAN_WORK, "credited": 0}},
                 "Lean other epsilon": {"work": {**PEARL_LEAN_WORK, "epsilon": "1/50"}},
                 "class held": {"work": {**PEARL_LEAN_WORK, "give_up": 96}, "exact_work": {
                     "credited": 5490, "credited_tiles": 39, "macs": 5590}}}.get(case, {}))
    if case != "no Lean":
        (piece / S.PEARLPLUS_LEAN).write_text(json.dumps(lean))
    audit = tmp_path / "audit"
    (audit / S.WINDOW_REGISTRATION).parent.mkdir(parents=True)
    (audit / S.WINDOW_REGISTRATION).write_text(json.dumps(
        {"format": REG.FORMAT_V2, "previous": "ab" * 64 if case == "R not after W" else PEARL_W, "registered": []}))
    (audit / S.AUDIT_SUMMARY).write_text(json.dumps({} if case == "audit untied" else {"tie": {"record": PEARL_W}}))
    if case != "no circuit record":
        (audit / S.CIRCUIT_RECORD).write_bytes(PEARL_RECORD)
    pin = "00" * 64 if case == "other key" else None if case == "no pin" else hashlib.sha512(PUBLIC).hexdigest()
    return piece.parent, audit, pin, "kernels" if case == "other lane" else "vllm"


def test_pearlpluss_accepted_piece_under_every_condition_of_record_is_pouws_verdict_of_record(tmp_path, monkeypatch,
                                                                                             capsys, live):
    """Pearl+'s accepted piece whose live coins the compose holds to its pin, whose session the lane reserved under its own
    gate key in its open Pearl-C period, whose passes are tied to the window sampled proofs audited, and which the Lean
    `PouwCheck.PearlPlus` verdict accepts on the same run with nonzero work, is PoUW's verdict of record: its row is
    Pearl+'s answer with the Lean's work, and remote's record composes it with W_cert that work, nothing missing, pending
    the remote check's answer."""
    monkeypatch.setenv("RESEARCH_RUN_ID", "r1")
    monkeypatch.setattr(S.socket, "gethostname", lambda: "vy-dc-h200")
    monkeypatch.setattr(S, "spot_row", lambda _audit_dir, _run=None: _accepted())
    pieces, audit, pin, lane = _pearlplus_piece(tmp_path, monkeypatch, live)
    row = S.piece_row("work", pieces, sampled_proofs=audit, key_sha512=pin, lane=lane)
    assert (row["protocol"], row["verdict"], row["commitment"], row["delta"]) == (S.PEARLPLUS_PROTOCOL, "accept",
                                                                                  "c1" * 64, "1/100")
    assert row["quantity"]["work"] == PEARL_LEAN_WORK and row["quantity"]["tie"]["record"] == PEARL_W
    profile = row["answer"]["profile"]
    assert (profile["lane"], profile["key_sha512"], profile["period"], profile["session"]) == ("vllm", pin, "p-pc", SID)
    assert profile["draw_session"] == DSID and profile["source"] == DRAW_COIN
    assert profile["lean"]["work"] == PEARL_LEAN_WORK and row["piece"]["piece"] == "pearlplus" and "development" not in row
    assert row["answer_from"] == S.PEARLPLUS_FROM and "the prover side's" in S.PEARLPLUS_FROM
    (tmp_path / "serve").mkdir()
    assert S.main(["record", "--serve-run", str(_serve_dir(tmp_path / "serve")), "--window", "1", "--sampled-proofs",
                   str(audit), "--pieces", str(pieces), "--out", str(tmp_path / "record"), "--node", "vy-dc-h200",
                   "--product", "remote", "--key-sha512", pin, "--lane", lane]) == 0
    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1].removeprefix("RECORD "))
    rec = json.loads((tmp_path / "record" / "block-record.json").read_text())
    jsonschema.Draft202012Validator(SCHEMA).validate(rec)
    assert (line["composed"], line["verdicts"]["PoUW"], line["check"]) == ("pending", "accept", {})
    assert (line["record_missing"], rec["composed"]["verdict"], rec["composed"]["answer"]) == ([], "undecided", None)
    assert line["answers"]["PoUW"] == S.PEARLPLUS_FROM
    reason = line["reason"]
    assert reason.count(S.PEARLPLUS_SAID) == 1 and reason.index(S.PEARLPLUS_SAID) > reason.index("PoUW: accept"), reason
    assert rec["legs"]["pouw"]["protocol"] == S.PEARLPLUS_PROTOCOL
    assert rec["legs"]["pouw"]["quantity"] == {"symbol": "W_cert", "value": 5400, "unit": "work", "tiles": 40}


def test_pearlpluss_verdict_of_record_keeps_a_class_held_runs_give_up_and_not_its_exact_credit(tmp_path, monkeypatch, live):
    """A run held to its class's fixed load publishes its work in the grid form, and the answer of record's work is the
    Lean's with that form's ``give_up``; the exact credit the form replaces (``exact_work``) is not in the row."""
    monkeypatch.setenv("RESEARCH_RUN_ID", "r1")
    monkeypatch.setattr(S.socket, "gethostname", lambda: "vy-dc-h200")
    monkeypatch.setattr(S, "spot_row", lambda _audit_dir, _run=None: _accepted())
    pieces, audit, pin, lane = _pearlplus_piece(tmp_path, monkeypatch, live, "class held")
    row = S.piece_row("work", pieces, sampled_proofs=audit, key_sha512=pin, lane=lane)
    assert row["verdict"] == "accept" and "development" not in row
    assert row["answer"]["public"]["work"] == {**PEARL_LEAN_WORK, "give_up": 96}
    assert "exact_work" not in json.dumps(row)


def test_the_record_reads_pearlpluss_tie_from_the_window_tied_to_w_under_window_tie(tmp_path, monkeypatch, capsys, live):
    """Sampled proofs' records audit (``--sampled-proofs``) is the step units' audit, chained after no W; the window tied
    to W, its registration chained right after W and its audit tied to W, is another (``--window-tie``), which the record
    reads Pearl+'s tie from. Without ``--window-tie`` it reads the step units' and composes pending, naming the chain."""
    monkeypatch.setenv("RESEARCH_RUN_ID", "r1")
    monkeypatch.setattr(S.socket, "gethostname", lambda: "vy-dc-h200")
    monkeypatch.setattr(S, "spot_row", lambda _audit_dir, _run=None: _accepted())
    pieces, tied, pin, lane = _pearlplus_piece(tmp_path, monkeypatch, live)
    (steps := tmp_path / "steps" / S.WINDOW_REGISTRATION).parent.mkdir(parents=True)
    steps.write_text(json.dumps({"format": REG.FORMAT_V2, "previous": None, "registered": []}))
    (tmp_path / "steps" / S.AUDIT_SUMMARY).write_text(json.dumps({}))
    (tmp_path / "serve").mkdir()
    base = ["record", "--serve-run", str(_serve_dir(tmp_path / "serve")), "--window", "1", "--sampled-proofs",
            str(tmp_path / "steps"), "--pieces", str(pieces), "--node", "vy-dc-h200", "--product", "remote",
            "--key-sha512", pin, "--lane", lane]
    S.main([*base, "--out", str(tmp_path / "untied")])
    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1].removeprefix("RECORD "))
    assert (line["composed"], line["verdicts"]["PoUW"]) == ("pending", S.DEV_VERDICT)
    assert "the window's registration R is chained after None..." in line["missing"]["PoUW"]
    assert S.main([*base, "--out", str(tmp_path / "tied"), "--window-tie", str(tied)]) == 0
    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1].removeprefix("RECORD "))
    assert (line["composed"], line["record_missing"], line["verdicts"]["PoUW"], line["check"]) == ("pending", [], "accept", {})


@pytest.mark.parametrize("case, why", [
    ("in-process coins", "its epoch's coins come from 'in-process', not a live session on the auditor host's challenger"),
    ("session refused", "the live session's check refused its passes: pass 2: coin: bad"),
    ("no pin", "the compose holds the coins to no key pin (--key-sha512, H1_KEY_SHA512)"),
    ("other key", "the live session's challenger key is not the compose's pin: the challenger key at "),
    ("no reserve", f"the piece's ledger holds no reserve of the live session {SID}"),
    ("reserved for", "the live session was reserved by vy-admin for lane vllm, not under the lane's own gate key"),
    ("other lane", "the live session was reserved under lane vllm's gate key, not lane kernels's own"),
    ("no period", "the live session joined no period of lane vllm"),
    ("H1 period", "the live session joined period p-h1, H1's (its open commits draw_k, rows): a Pearl-C salt session is "
                  "credited in a Pearl-C period alone"),
    ("node draw", S.PEARLPLUS_NODE_DRAW),
    ("draw refused", "the draw session's check refused it: ValueError: the record's draw coin: bad"),
    ("draw key", "the draw session's challenger key is not the compose's pin: the challenger key at "),
    ("period open", "the piece's ledger doesn't close the Pearl-C period p-pc, so no draw follows it"),
    ("drawn again", f"the draw session {DSID} is not lane vllm's first reserve after its Pearl-C period p-pc closed "
                    "(that is s-first): a draw is taken once"),
    ("draw not withheld", f"the draw session {DSID} was not reserved withheld with a draw session's layout alone"),
    ("second open", f"the ledger opens ['{DSID}', 's-other'] on the run's commitment, not the draw session {DSID} alone"),
    ("draw off C", "the record's draw coin: "),
    ("source not coin", "the record's source is not its draw session's coin"),
    ("unpublished coin", "the verdict is not held to the coin published at its draw: none published"),
    ("untied", "Pearl+'s verdict holds its passes to no window record W (served_passes.py verify --tie W), so nothing "
               "ties them to the window"),
    ("R not after W", f"the window's registration R is chained after {'ab' * 8}..., not right after Pearl+'s window "
                      f"record W {PEARL_W[:16]}..."),
    ("audit untied", f"the window's records audit is not tied to Pearl+'s window record W {PEARL_W[:16]}... "
                     "(records_audit --tie W)"),
    ("no circuit record", f"no window-circuit record V3c held R's circuit to ({S.CIRCUIT_RECORD} beside R: "
                          "FileNotFoundError), so nothing holds period p-pc's circuit to the window served"),
    ("period on another circuit", f"period p-pc committed circuit {hashlib.sha512(b'another circuit').hexdigest()[:16]}"
                                  f"..., not {hashlib.sha512(PEARL_RECORD).hexdigest()[:16]}..., the SHA-512 of the "
                                  "window-circuit record V3c held R's circuit to: its salt session counts toward another "
                                  "circuit than the window served"),
    ("no Lean", f"no Lean PouwCheck.PearlPlus verdict ({S.PEARLPLUS_LEAN}: FileNotFoundError"),
    ("Lean rejects", "the Lean PouwCheck.PearlPlus verdict is reject: the live session's coins are out of scope"),
    ("Lean other run", f"the Lean verdict is on run {'c2' * 8}... at source {DRAW_COIN[:16]}..., not Pearl+'s verdict's "
                       "run and source"),
    ("Lean untied", f"the Lean verdict doesn't hold the passes to Pearl+'s window record W {PEARL_W[:16]}..."),
    ("Lean credits nothing", "the Lean verdict credits 0 MACs over 40 tiles: nothing"),
    ("Lean other epsilon", "the Lean verdict's ε '1/50' is not the reference's '1/100'"),
])
def test_pearlpluss_accepted_piece_that_fails_a_condition_of_record_stays_a_development_piece_naming_it(
        tmp_path, monkeypatch, live, case, why):
    """Each condition of record broken alone leaves Pearl+'s accepted piece a development one: its row missing with its
    document, its ``why`` the development verdict and the condition, in `pearlplus_of_record`'s order, so remote's record
    composes pending and names it."""
    pieces, audit, pin, lane = _pearlplus_piece(tmp_path, monkeypatch, live, case)
    row = S.piece_row("work", pieces, sampled_proofs=audit, key_sha512=pin, lane=lane)
    assert (row["verdict"], row["development"]["piece"]) == ("missing", "pearlplus"), row
    assert row["why"].startswith(f"{S.DEV_VERDICT}: {why}"), row["why"]
    _report, rec, line = S.record({"integrity": _accepted(), "work": row}, window=1, steps=(0, 4), node="vy-dc-h200",
                                  run="r", servers={"work": ("vy-dc-h200", "real")}, product="remote")
    assert (line["composed"], line["verdicts"]["PoUW"], line["missing"]["PoUW"]) == ("pending", S.DEV_VERDICT, row["why"])
    assert rec["legs"]["pouw"] is None


def _draw_answers(calls: list, public: bytes = PUBLIC):
    """The gate's answers for the run's draw session, as `pearlplus_draw` asks them: its tree for ``fetch`` (its key),
    the ledger (its reserve and its open on the run's commitment) and the pins."""
    def ask(*words: str) -> bytes:
        calls.append(words)
        if words[0] == "fetch":
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w:gz") as t:
                for name, body in (("key.json", json.dumps({"H": 5, "W": 4, "public": public.hex()}).encode()),
                                   ("record.json", b"{}"), ("releases.jsonl", b'{"event": "root"}\n'), ("close.json", b"{}")):
                    info = tarfile.TarInfo(f"{words[1]}/out/{name}")
                    info.size = len(body)
                    t.addfile(info, io.BytesIO(body))
            return buf.getvalue()
        if words[0] == "ledger":
            return ("".join(json.dumps(e) + "\n" for e in (
                {"kind": "reserve", "session": DSID, "body": {"by": "vllm", "withheld": True, "families": [["pouw/draw", 1]]}},
                {"kind": "open", "session": DSID, "body": {"window": {"root": "c1" * 64}, "withheld": True}}))).encode()
        return (PINS_LINE + "\n").encode()
    return ask


@pytest.mark.parametrize("case, reason", [
    ("honest", None),
    ("node", None),
    ("no gate", "ValueError: no gate, key pin or main's first-parent line to check the draw session by"),
    ("other key", "CoinError: the challenger key at "),
    ("untrusted", "CoinError: the auditor's ledger opens no challenger session for this window"),
    ("not withheld", "ValueError: the run's draw session: the session the auditor's ledger opens on the window does not "
                     "withhold"),
    ("other commitment", "ValueError: the record's draw commitment is not the one the draw session's challenger signed"),
    ("other root", "ValueError: the draw session's log does not show the run's commitment as the one root its draw coin "
                   "followed"),
    ("off C", "ValueError: the record's draw coin: "),
    ("other source", "ValueError: the record's source is not its draw session's coin"),
])
def test_the_pearlplus_piece_holds_the_verifiers_draw_to_its_draw_session(tmp_path, monkeypatch, case, reason):
    """`pearlplus_draw`: a challenger draw's session fetched through the gate with the ledger and pins, its key held to
    the pin, the ledger opening it alone on the run's commitment C (`challenger.trusted_session_of`, its signed session),
    withheld, its signed commitment the record's, its log showing C as its coin's one root, and the record's opening its
    coin after C with the record's source that coin; a node's draw is checked by none of it. Each failure names why, and
    the piece rejects on it."""
    from verity.protocols.pouw.verifier import window as Wd
    from verity.protocols.spot.device import challenger as Ch

    other = _draw(seed=8)
    signed = {"other commitment": other["commitment"]}.get(case, DRAW["commitment"])
    root = bytes.fromhex("c2" * 64 if case == "other root" else "c1" * 64)
    seen = []

    def trusted(ledger, window, key, pins, main, session, verify):
        seen.append((window, key.public, pins, main))
        if case == "untrusted":
            raise Ch.CoinError("the auditor's ledger opens no challenger session for this window")
        return Ch.LiveCommitment.from_json(signed), {Wd.DRAW: (root,)}
    monkeypatch.setattr(Ch, "trusted_session_of", trusted)

    def withheld_on(ledger, window):
        assert [json.loads(x)["kind"] for x in ledger] == ["reserve", "open"] and window == "c1" * 64
        return case != "not withheld"
    monkeypatch.setattr(Ch, "withheld_on", withheld_on)
    draw = {"off C": _draw("c2" * 64)}.get(case, DRAW)
    (rec := tmp_path / "coin.json").write_bytes(_coin_record({"other source": "77" * 64}.get(case, DRAW_COIN),
                                                             NODE if case == "node" else draw))
    (main := tmp_path / "main.txt").write_text("aa" * 20 + "\n")
    calls = []
    ask = _draw_answers(calls, public=bytes(80) if case == "other key" else PUBLIC)
    out = tmp_path / "pieces" / "pouw"
    out.mkdir(parents=True)
    got = S.pearlplus_draw(rec, out, None if case == "no gate" else GATE, hashlib.sha512(PUBLIC).hexdigest(), main,
                           fetch=None if case == "no gate" else ask)
    assert got["reason"] == reason if reason is None else got["reason"].startswith(reason), got
    assert (got["from"], got["session"]) == (("node", None) if case == "node" else ("challenger", DSID))
    assert calls == ([] if case in ("node", "no gate") else [("fetch", DSID), ("ledger",), ("pins",)])
    if case in ("honest", "untrusted", "other commitment", "other root", "off C", "other source", "not withheld"):
        assert seen == [({"root": "c1" * 64}, PUBLIC, [PINS_LINE], ["aa" * 20])]
    if case != "no gate":
        return
    served, verified = _pearlplus_run()
    monkeypatch.setattr(S, "pearlplus_coins", lambda *a: {"session": SID, "closed": None, "passes": 4, "reason": None})
    monkeypatch.setattr(S, "_gate_out", lambda gate, *words: ask(*words))
    answer = {**verified["answer"], "commitment": "c1" * 64, "profile": {"source": DRAW_COIN}}
    S.coin(rec, out, "r1", research=["research"], put=_put(NODE_PUT)[0])
    doc = S.pearlplus(out, served, {**verified, "answer": answer}, gate=GATE, key="00" * 64, main=main,
                      coin=out / S.COIN_PUT)
    assert doc["verdict"] == "reject" and doc["reasons"] == [f"draw: {doc['coins']['draw']['reason']}"]
    assert doc["coins"]["draw"]["reason"].startswith("CoinError: the challenger key at ")
    assert doc["coins"]["draws"] == S.PEARLPLUS_COINS["draws"]


def test_the_pearlplus_period_opens_around_the_retained_serve_and_closes_before_the_draw(tmp_path, monkeypatch, capsys):
    """``pearlplus-period open``: the lane's Pearl-C period opened at the gate with cap 1 (the serve's one salt session)
    and circuit the SHA-512 of the piece's descriptor (Pearl+'s protocol, the scheme and the cubin's SHA-512), its answer
    in ``period.json``; ``close`` closes it by that name. A gate that refuses is a PERIOD refused line, exit 1. With
    ``--circuit-record`` (a tied window's PEARLPLUS-CIRCUIT) the descriptor is that record byte for byte, and the
    circuit its SHA-512; it and ``--cubin`` are exclusive."""
    calls = []

    def gate_out(gate: str, *words: str) -> bytes:
        calls.append((gate, words))
        if "refuse" in words:
            return b'{"refused": "lane vllm has period p0 open: close it first"}'
        return json.dumps({"period": words[2], "verb": words[0]}).encode()
    monkeypatch.setattr(S, "_gate_out", gate_out)
    (cubin := tmp_path / "k.cubin").write_bytes(b"sass")
    piece = tmp_path / "pieces" / "pouw"
    assert S.main(["pearlplus-period", "open", str(piece), "--gate", GATE, "--name", "pearlplus-r1", "--cubin",
                   str(cubin)]) == 0
    raw = (piece / S.PEARLPLUS_DESCRIPTOR).read_bytes()
    assert json.loads(raw) == {"schema": S.PEARLPLUS_DESCRIPTOR_SCHEMA, "protocol": S.PEARLPLUS_PROTOCOL,
                               "scheme": "pearl-c-h100-v1-h3", "cubin_sha512": hashlib.sha512(b"sass").hexdigest()}
    circuit = hashlib.sha512(raw).hexdigest()
    assert capsys.readouterr().out == f"PERIOD opened pearlplus-r1: Pearl-C, cap 1, circuit {circuit[:16]}... " \
                                      f"({S.PEARLPLUS_DESCRIPTOR})\n"
    assert calls == [(GATE, ("period-open", "--name", "pearlplus-r1", "--cap", "1", "--circuit", circuit))]
    assert S.main(["pearlplus-period", "close", str(piece), "--gate", GATE]) == 0
    assert calls[1] == (GATE, ("period-close", "--name", "pearlplus-r1"))
    doc = json.loads((piece / S.PERIOD).read_text())
    assert (doc["name"], doc["cap"], doc["circuit"], doc["close"]["verb"]) == ("pearlplus-r1", 1, circuit, "period-close")
    assert capsys.readouterr().out.startswith("PERIOD closed pearlplus-r1: ")
    assert S.main(["pearlplus-period", "open", str(tmp_path / "other"), "--gate", GATE, "--name", "refuse"]) == 1
    assert capsys.readouterr().out.startswith("PERIOD refused: RuntimeError: the gate refused period-open: ")
    assert S.main(["pearlplus-period", "open", str(tmp_path / "other"), "--gate", GATE]) == 1
    capsys.readouterr()
    (record := tmp_path / "circuit-record.json").write_bytes(PEARL_RECORD)
    tied = tmp_path / "serve" / "pearlplus"
    assert S.main(["pearlplus-period", "open", str(tied), "--gate", GATE, "--name", "pearlplus-r2", "--circuit-record",
                   str(record)]) == 0
    circuit = hashlib.sha512(PEARL_RECORD).hexdigest()
    assert (tied / S.PEARLPLUS_DESCRIPTOR).read_bytes() == PEARL_RECORD
    assert calls[-1] == (GATE, ("period-open", "--name", "pearlplus-r2", "--cap", "1", "--circuit", circuit))
    doc = json.loads((tied / S.PERIOD).read_text())
    assert (doc["circuit"], doc["circuit_record"], doc["descriptor"]) == (circuit, str(record), S.PEARLPLUS_DESCRIPTOR)
    assert capsys.readouterr().out == (f"PERIOD opened pearlplus-r2: Pearl-C, cap 1, circuit {circuit[:16]}... (the "
                                       f"window's circuit record {record})\n")
    with pytest.raises(SystemExit):
        S.main(["pearlplus-period", "open", str(tied), "--gate", GATE, "--name", "pearlplus-r3", "--circuit-record",
                str(record), "--cubin", str(cubin)])


@pytest.mark.parametrize("case", ["held", "stray", "unread", "unsalted", "unparsed"])
def test_the_tied_period_closes_holding_its_reserves_to_the_windows_salt_sessions(tmp_path, monkeypatch, capsys, case):
    """``pearlplus-period close --lane L --windows PEARL_DIR`` (PEARLPLUS_TIE's, in serve()): the ledger read before the
    close, the sessions lane L reserved (``for``, else ``by``) since the period's open held to the windows' salt sessions
    (each pearl/w<k>/served.json's ``coins.reserved.session``, in window order), kept as period.json's ``reserves`` and
    said held or REFUSED in a line that names no session and counts none; a window's served.json that doesn't parse
    refuses them, naming it; the period closes either way."""
    calls = []
    ledger = [{"kind": "reserve", "session": "s-before", "body": {"by": "vllm"}},
              {"kind": "period-open", "session": None, "body": {"period": "pp-r1", "lane": "vllm"}},
              {"kind": "reserve", "session": "s-other", "body": {"by": "kernels"}},
              {"kind": "reserve", "session": "s-salt", "body": {"by": "vllm"}}]
    if case == "stray":
        ledger.insert(2, {"kind": "reserve", "session": "s-stray", "body": {"by": "infra", "for": "vllm"}})

    def gate_out(gate: str, *words: str) -> bytes:
        calls.append(words[0])
        if words[0] == "ledger":
            if case == "unread":
                raise RuntimeError("the gate refused ledger (exit 255): ssh: connect to host: timed out")
            return "".join(json.dumps(e) + "\n" for e in ledger).encode()
        return json.dumps({"period": words[2], "verb": words[0]}).encode()
    monkeypatch.setattr(S, "_gate_out", gate_out)
    serve = tmp_path / "serve"
    (tied := serve / "pearlplus").mkdir(parents=True)
    (tied / S.PERIOD).write_text(json.dumps({"name": "pp-r1", "cap": 1}))
    for k, reserved in ((0, {"session": "s-salt", "period": "pp-r1"}), (1, None)):
        (serve / "pearl" / f"w{k}").mkdir(parents=True)
        if case != "unsalted":
            (serve / "pearl" / f"w{k}" / "served.json").write_text(json.dumps({"coins": {"reserved": reserved}}))
    if case == "unparsed":
        (serve / "pearl" / "w1" / "served.json").write_text("{")
    assert S.main(["pearlplus-period", "close", str(tied), "--gate", GATE, "--lane", "vllm", "--windows",
                   str(serve / "pearl")]) == 0
    assert calls == (["period-close"] if case == "unparsed" else ["ledger", "period-close"])
    doc, out = json.loads((tied / S.PERIOD).read_text()), capsys.readouterr().out
    salt = [] if case in ("unsalted", "unparsed") else ["s-salt"]
    assert doc["close"]["verb"] == "period-close" and doc["reserves"]["salt_sessions"] == salt
    assert doc["reserves"]["ok"] is (case == "held") and not any(f"s-{s}" in out for s in ("salt", "stray", "other"))
    said = {"held": "its reserves held (lane vllm's since its open the windows' salt sessions alone)",
            "stray": "its reserves REFUSED (lane vllm reserved in it other sessions than the windows' salt sessions; "
                     "period.json's reserves), so no draw follows it",
            "unread": "its reserves REFUSED (the ledger or a window's served.json can't be read: RuntimeError: the gate "
                      "refused ledger (exit 255)",
            "unsalted": "its reserves REFUSED (no window's served.json names a salt session; ",
            "unparsed": "its reserves REFUSED (the ledger or a window's served.json can't be read: ValueError: "
                        "w1/served.json: JSONDecodeError: "}[case]
    assert out.startswith(f"PERIOD closed pp-r1, {said}"), out
    if case not in ("unread", "unparsed"):
        assert doc["reserves"]["since_open"] == (["s-stray", "s-salt"] if case == "stray" else ["s-salt"])
    assert S.main(["pearlplus-period", "close", str(tied), "--gate", GATE, "--lane", "vllm"]) == 1
    assert capsys.readouterr().out.startswith(
        "PERIOD refused: ValueError: pearlplus-period close takes --lane and --windows together")


@pytest.mark.parametrize("polls", [0, 1, 3])
def test_the_tied_close_waits_for_the_ledger_to_end_each_windows_salt_session(tmp_path, monkeypatch, capsys, polls):
    """``--wait S``: the gate closes a period only once each of its windows has ended, and a salt session's ``end`` follows
    its client's close by the time its challenger takes to exit, so the close reads the ledger again every PERIOD_POLL_S
    seconds, for at most S, until it ends each window's salt session, then closes; past S it closes anyway, and the gate's
    refusal is the close's."""
    ledger = [{"kind": "period-open", "session": None, "body": {"period": "pp-r1", "lane": "vllm"}},
              {"kind": "reserve", "session": "s-salt", "body": {"by": "vllm"}},
              {"kind": "open", "session": "s-salt", "body": {"by": "vllm"}}]
    calls, slept, reads = [], [], []

    def gate_out(gate: str, *words: str) -> bytes:
        calls.append(words[0])
        if words[0] == "ledger":
            reads.append(1)
            ended = [{"kind": "end", "session": "s-salt", "body": {"complete": True}}] if len(reads) > polls else []
            return "".join(json.dumps(e) + "\n" for e in ledger + ended).encode()
        if len(reads) <= polls:
            return b'{"refused": "period pp-r1\'s window s-salt still runs"}'
        return json.dumps({"period": words[2], "verb": words[0]}).encode()
    monkeypatch.setattr(S, "_gate_out", gate_out)
    serve = tmp_path / "serve"
    (tied := serve / "pearlplus").mkdir(parents=True)
    (tied / S.PERIOD).write_text(json.dumps({"name": "pp-r1", "cap": 1}))
    (serve / "pearl" / "w0").mkdir(parents=True)
    (serve / "pearl" / "w0" / "served.json").write_text(json.dumps({"coins": {"reserved": {"session": "s-salt"}}}))
    if polls < 3:
        doc = S.pearlplus_period_close(tied, GATE, "vllm", serve / "pearl", wait_s=2 * S.PERIOD_POLL_S,
                                       sleep=slept.append)
        assert calls == ["ledger"] * (polls + 1) + ["period-close"] and slept == [S.PERIOD_POLL_S] * polls
        assert doc["reserves"]["ok"] is True and doc["close"]["verb"] == "period-close"
        return
    with pytest.raises(RuntimeError, match="the gate refused period-close: .*still runs"):
        S.pearlplus_period_close(tied, GATE, "vllm", serve / "pearl", wait_s=2 * S.PERIOD_POLL_S, sleep=slept.append)
    assert calls == ["ledger"] * 3 + ["period-close"] and slept == [S.PERIOD_POLL_S] * 2
    assert json.loads((tied / S.PERIOD).read_text())["reserves"]["ok"] is True
    assert S.main(["pearlplus-period", "close", str(tied), "--gate", GATE, "--lane", "vllm", "--windows",
                   str(serve / "pearl"), "--wait", "0"]) == 0
    assert calls[4:] == ["ledger", "period-close"] and slept == [S.PERIOD_POLL_S] * 2
    assert capsys.readouterr().out.startswith("PERIOD closed pp-r1, its reserves held")


@pytest.mark.parametrize("case", ["held", "between", "open", "node"])
def test_the_rule_line_reads_whether_the_draw_was_the_lanes_first_reserve_after_the_close(tmp_path, monkeypatch,
                                                                                          capsys, case):
    """``pearlplus-period rule``, once the coin is drawn: HELD where the coin record's draw session is the lane's first
    reserve after the period in DIR's period.json closed (`pouw_h1_window.after_close`), else BROKEN, exit 1, saying
    why; kept as period.json's ``rule``. A reserve of another lane between them leaves it held."""
    ledger = [{"kind": "period-open", "session": None, "body": {"period": "pp-r1", "lane": "vllm"}},
              {"kind": "reserve", "session": "s-salt", "body": {"by": "vllm"}}]
    if case != "open":
        ledger.append({"kind": "period-close", "session": None, "body": {"period": "pp-r1"}})
    ledger.append({"kind": "reserve", "session": "s-other", "body": {"by": "kernels"}})
    if case == "between":
        ledger.append({"kind": "reserve", "session": "s-between", "body": {"by": "vllm"}})
    ledger.append({"kind": "reserve", "session": "s-draw", "body": {"by": "vllm", "withheld": True}})
    monkeypatch.setattr(S, "_gate_out", lambda gate, *words: "".join(json.dumps(e) + "\n" for e in ledger).encode())
    (piece := tmp_path / "pearlplus").mkdir()
    (piece / S.PERIOD).write_text(json.dumps({"name": "pp-r1"}))
    draw = {"from": "node"} if case == "node" else {"from": "challenger", "reserved": {"session": "s-draw"}}
    (coin := tmp_path / "coin.json").write_text(json.dumps({"draw": draw}))
    argv = ["pearlplus-period", "rule", str(piece), "--gate", GATE, "--lane", "vllm", "--coin", str(coin)]
    assert S.main(argv) == (0 if case == "held" else 1)
    why = {"held": "HELD (the draw session s-draw is the lane's first reserve after the close)",
           "between": "BROKEN (the lane's first reserve after the close is s-between, not the draw session s-draw)",
           "open": "BROKEN (the ledger doesn't close period pp-r1)",
           "node": "BROKEN (the verifier's coin record names no draw session)"}[case]
    assert capsys.readouterr().out == f"RULE no other reserve on lane vllm between period pp-r1's close and the draw: {why}\n"
    assert json.loads((piece / S.PERIOD).read_text())["rule"]["held"] is (case == "held")
    assert S.main(argv[:-2]) == 1
    assert capsys.readouterr().out.startswith("RULE not read: ValueError: pearlplus-period rule needs --lane and --coin")


def test_the_pouw_piece_fetches_the_windows_sample_session_and_gives_the_check_its_tree(tmp_path, monkeypatch):
    """A window whose record names its sample's session (``draw_session``, which the coin check refuses without): the
    piece fetches it beside the window's session, and the check gets its tree as ``--draw-dir``."""
    calls, got = [], []
    monkeypatch.setattr(S, "_gate_out", _gate(calls))
    d, dsid = _h1_serve_dir(tmp_path), "s20261026T221503Z-cd34"
    (d / S.H1_CALLS.format(k=1)).write_text(json.dumps({"commitment": {"session": {"session": SID},
                                                                       "draw_session": {"session": dsid}}}))

    def check(calls_file, tree, ledger, pins, main, key, out, inputs):
        got.append(inputs)
        assert (tree / "out" / "record.json").is_file() and (Path(inputs["--draw-dir"]) / "out" / "record.json").is_file()
        return {"ok": True, "reason": None, "manifest": None, "credit": {}}

    S.pouw(d, 1, tmp_path / "pieces" / "pouw", gate=GATE, key="ff" * 64, main=None, check=check)
    assert [w for _g, w in calls] == [("fetch", SID), ("fetch", dsid), ("ledger",), ("pins",)]
    assert got == [{"--draw-dir": str(tmp_path / "pieces" / "pouw" / "session" / dsid)}]


def test_the_pouw_piece_gives_credit_the_declared_program_the_period_and_the_proofs_it_has(tmp_path, monkeypatch):
    """With the serve dir's declared Program (``h1/window-program.json.gz``) and its period (``period.json``), and the
    sampled statements and live evidence given, the challenger check gets each as its ``--check-challenger`` flag, and the
    piece's document carries them and credit's calls."""
    monkeypatch.setattr(S, "_gate_out", _gate([]))
    d = _h1_serve_dir(tmp_path)
    (d / "h1" / "window-program.json.gz").write_bytes(b"")
    (d / S.PERIOD).write_text(json.dumps({"name": "served-r1"}))
    st, lv = tmp_path / "statements.json", tmp_path / "live.json"
    got = []

    def check(*a):
        got.append(a[-1])
        return {"ok": True, "reason": None, "manifest": None, "credit": {"window": None, "calls": [3], "credit": [3]}}

    doc = S.pouw(d, 1, tmp_path / "pieces" / "pouw", gate=GATE, key="ff" * 64, main=None, check=check, statements=st,
                 live=lv)
    want = {"--descriptor": str(d / "h1" / "window-program.json.gz"), "--period": "served-r1", "--statements": str(st),
            "--live-evidence": str(lv)}
    assert got == [want] and doc["credit_inputs"] == want
    assert (doc["credit"]["window"], doc["credit"]["calls"], doc["credit"]["credit"]) == (None, [3], [3])
    assert (doc["AcceptedLive"], doc["SaltIsLiveCoin"]) == ({"met": True, "why": S.LIVE_MET},
                                                            {"met": True, "why": S.SALT_MET})


def test_the_pouw_piece_hands_the_verifiers_memo_to_the_check_and_records_what_it_took(tmp_path, monkeypatch):
    """``--lean-answers`` reaches the challenger check as its own flag, never among credit's inputs, and the piece's
    document carries the check's counts of answers it ran and took by key."""
    monkeypatch.setattr(S, "_gate_out", _gate([]))
    d = _h1_serve_dir(tmp_path)
    memo, got = tmp_path / "verifier" / "answers", []
    counts = {"ran": 33, "taken": {}, "unkeyed": 0}

    def check(*a):
        got.append(a[-1])
        return {"ok": True, "reason": None, "manifest": None, "credit": {"credit": [3]}, "lean_answers": counts}

    doc = S.pouw(d, 1, tmp_path / "pieces" / "pouw", gate=GATE, key="ff" * 64, main=None, check=check,
                 lean_answers=memo)
    assert got[0]["--lean-answers"] == str(memo) and "--lean-answers" not in doc["credit_inputs"]
    assert doc["lean_answers"] == counts


def test_a_record_that_isnt_one_manifest_under_its_salt_coin_doesnt_meet_salt_is_live_coin(tmp_path, monkeypatch):
    """Coins the challenger check accepts with a record that isn't one manifest under them: ``SaltIsLiveCoin`` isn't
    met, and names the manifest's refusal."""
    monkeypatch.setattr(S, "_gate_out", _gate([]))
    got = {"ok": True, "reason": None, "manifest": "round 2's tables are not under the salt", "credit": {}}
    doc = S.pouw(_h1_serve_dir(tmp_path), 1, tmp_path / "pieces" / "pouw", gate=GATE, key="ff" * 64, main=None,
                 check=lambda *a: got)
    assert doc["SaltIsLiveCoin"] == {"met": False, "why": "the window's record isn't one manifest under its salt coin: "
                                                          "round 2's tables are not under the salt"}
    assert doc["AcceptedLive"]["met"] is False and doc["verdict"] == "reject"


@pytest.mark.parametrize("case", ["coins", "untied", "in-process", "no gate"])
def test_a_development_pouw_piece_rejects_what_its_checks_refuse(tmp_path, monkeypatch, case):
    """Coins the challenger check refuses, H1 tables untied to the Commit, coins held in the serving process, or no gate
    to check the session by: the piece rejects, naming why, and so does the record."""
    monkeypatch.setattr(S, "_gate_out", _gate([]))
    d = _h1_serve_dir(tmp_path, "in-process" if case == "in-process" else "challenger", tied=case != "untied")
    refused = {"ok": False, "reason": "session s1 is not the one the ledger opened", "manifest": None, "credit": {}}
    ok = {"ok": True, "reason": None, "manifest": None, "credit": {}}
    doc = S.pouw(d, 1, tmp_path / "pieces" / "pouw", gate=None if case == "no gate" else GATE, key="ff" * 64, main=None,
                 check=lambda *a: refused if case == "coins" else ok)
    why = {"coins": "coins: session s1 is not the one the ledger opened", "untied": "ROOTS1:",
           "in-process": "coins: in-process, not the auditor host's challenger's", "no gate": "coins: ValueError: no gate"}
    assert doc["verdict"] == "reject" and doc["reasons"][0].startswith(why[case]), doc["reasons"]
    salt = {"coins": "the challenger check refused the window's coins, its salt among them: session s1",
            "untied": S.SALT_MET, "in-process": "the challenger check didn't run on the window: coins: in-process",
            "no gate": "the challenger check didn't run on the window: coins: ValueError: no gate"}
    assert doc["SaltIsLiveCoin"]["met"] is (case == "untied") and doc["SaltIsLiveCoin"]["why"].startswith(salt[case])
    assert doc["AcceptedLive"]["met"] is False
    row = S.piece_row("work", tmp_path / "pieces")
    assert row["verdict"] == "reject" and row["reason"].startswith(why[case])
    _report, rec, line = S.record(_rows(_accepted(), tmp_path / "pieces"), window=1, steps=(0, 4), node="n", run=None)
    assert rec is None and (line["composed"], line["by"], line["development"]) == ("reject", ["PoUW"], {"PoUW": "reject"})


def test_the_gate_is_refused_while_the_lane_has_a_period_open(tmp_path, monkeypatch):
    """``gate``: a reserve joins the lane's open period, so a lane with one open is refused before serving; the line
    names the gate's latest pin and whether it is on the verifier's first-parent line of main."""
    ledger = [{"kind": "period-open", "session": None, "body": {"period": "p1", "lane": "vllm"}}]
    answers = {"ledger": ledger, "pins": [{"commit": "aa" * 20, "main": "bb" * 20, "pinned_utc": "2026-10-26T21:35:00Z"}]}
    monkeypatch.setattr(S, "_gate_out", lambda gate, verb: "".join(json.dumps(e) + "\n" for e in answers[verb]).encode())
    (main := tmp_path / "main.txt").write_text("cc" * 20 + "\n" + "aa" * 20 + "\n")
    ok, line = S.gate_ready(GATE, "vllm", main)
    assert not ok and line.startswith("GATE refused: lane vllm's open period p1 (a reserve would join it)")
    assert "on the verifier's first-parent line of main" in line and "NOT" not in line
    answers["ledger"].append({"kind": "period-close", "session": None, "body": {"period": "p1"}})
    main.write_text("cc" * 20 + "\n")
    ok, line = S.gate_ready(GATE, "vllm", main)
    assert ok and line.startswith("GATE ready: lane vllm's open period None;") and "NOT on the verifier's" in line


def test_the_gate_refuses_a_root_an_open_in_its_ledger_names(monkeypatch):
    """``gate --root``: the gate opens a root once, ever, so a run root an open in the ledger names (v2's window root) is
    refused before a session of the run reserves, and another root is ready; the lane's open period still refuses."""
    opened = {"kind": "open", "session": "s1", "schema": "verity/challenger-auditor/v2", "body": {"window": {"root": "ab" * 64}}}
    answers = {"ledger": [{"kind": "reserve", "session": "s1", "body": {"by": "vllm"}}, opened], "pins": []}
    monkeypatch.setattr(S, "_gate_out", lambda gate, verb: "".join(json.dumps(e) + "\n" for e in answers[verb]).encode())
    ok, line = S.gate_ready(GATE, "vllm", root="AB" * 64)
    assert not ok and line.startswith("GATE refused: lane vllm's open period None; root ABABABABABABABAB… already has its "
                                      "session s1 (one session a root, ever);")
    ok, line = S.gate_ready(GATE, "vllm", root="cd" * 64)
    assert ok and "; root cdcdcdcdcdcdcdcd… has no session yet;" in line
    answers["ledger"].append({"kind": "period-open", "session": None, "body": {"period": "p1", "lane": "vllm"}})
    ok, line = S.gate_ready(GATE, "vllm", root="cd" * 64)
    assert not ok and line.startswith("GATE refused: lane vllm's open period p1 (a reserve would join it); root cdcd")


FAKE_PROVE_PATH = '''import json, sys
a = sys.argv[1:]
assert a[0] == "window-circuit"
open("argv.jsonl", "a").write(json.dumps(a) + "\\n")
c = (a[a.index("--class") + 1] if "--class" in a else "record") + ("+tie" if "--window-registration" in a else "")
answer = json.load(open("answers.json"))[c]
if answer != "count":
    print("prove_path.py window-circuit: refused at", c, file=sys.stderr)
    sys.exit({"refuse": 2, "expire": 124}[answer])
open(a[a.index("--out") + 1], "w").write("{}")
print(json.dumps({"bits": 772, "count": {"free": 3, "inputs": 10, "registered": 7, "units": 1}}))
'''


def _circuits(tmp_path: Path, classes: str, answers: dict, **extra: str) -> tuple[int, list[str], list[str]]:
    """`served_verdicts.sh circuits 0` in a toy tree whose prove_path.py counts a class, refuses it, or exits 124 as
    `timeout` does when a count runs out of time (``answers``, by class, ``C+tie`` for class C reading H1's tables), under
    a `timeout` that records the limit it got (``extra`` adds to the environment): the exit status, the verdict lines,
    and the limits in order."""
    tree, bin_, venv = tmp_path / "tree", tmp_path / "bin", tmp_path / "venv" / "bin"
    for d in (tree / "benchmarks" / "prove_path", bin_, venv, tmp_path / "serve"):
        d.mkdir(parents=True)
    (tree / "benchmarks" / "prove_path" / "prove_path.py").write_text(FAKE_PROVE_PATH)
    (tree / "answers.json").write_text(json.dumps(answers))
    (tmp_path / "serve" / "weights_registration.json").write_text("{}")
    for path, body in ((venv / "python", f'exec {sys.executable} "$@"'),
                       (bin_ / "timeout", f'echo "$1" >> {tmp_path}/limits; shift; exec "$@"')):
        path.write_text(f"#!/bin/sh\n{body}\n")
        path.chmod(0o755)
    env = {"PATH": f"{bin_}:{os.defpath}", "HOME": str(tmp_path), "VENV": str(venv.parent),
           "D": str(tmp_path / "serve"), "OUT": str(tmp_path / "out"), "CLASSES": classes, **extra}
    rc = subprocess.run(["bash", str(HERE.parent / "served_verdicts.sh"), "circuits", "0"], cwd=tree, env=env,
                        check=False).returncode
    return rc, (tmp_path / "out" / "verdicts.txt").read_text().splitlines(), (tmp_path / "limits").read_text().split()


def test_a_class_that_counts_passes_and_the_class_of_record_is_not_counted_unasked(tmp_path):
    """CLASSES' one class counts under WC_TIMEOUT, its published count without the advice's length; the class of record,
    which CLASSES doesn't name and which would run out of time here, is never started."""
    rc, lines, limits = _circuits(tmp_path, "4,32,4", {"4,32,4": "count", "record": "expire"})
    assert rc == 0 and limits == ["30m"], (rc, lines)
    assert len(lines) == 1, lines
    assert lines[0].startswith("WINDOW-CIRCUIT class 4,32,4: registered 7 bits, free 3 bits of 10 input bits at the "
                               "weights registration's pins; ")
    assert not list(tmp_path.rglob("window-circuit*record*"))
    published = json.loads((tmp_path / "out" / "window-circuit-4-32-4.json").read_text())
    kept = json.loads((tmp_path / "serve" / "window-circuit-4-32-4.count.json").read_text())
    assert "bits" not in published and '"bits"' not in lines[0] and kept["bits"] == 772 and published["count"] == kept["count"]


@pytest.mark.parametrize("classes, answers, refused", [
    ("4,32,4 8,64,4", {"4,32,4": "count", "8,64,4": "refuse", "record": "count"}, "WINDOW-CIRCUIT class 8,64,4 refused (exit 2): "),
    ("4,32,4", {"4,32,4": "expire", "record": "count"}, "WINDOW-CIRCUIT class 4,32,4 timed out after WC_TIMEOUT=30m"),
    ("record", {"record": "expire"}, "WINDOW-CIRCUIT class record timed out after WC_TIMEOUT=30m"),
])
def test_any_class_of_classes_that_does_not_count_fails_the_run(tmp_path, classes, answers, refused):
    """Each class of CLASSES gates, not just one of them; CLASSES that names `record` gates on the class of record
    under WC_TIMEOUT, and no class CLASSES doesn't name is counted."""
    rc, lines, limits = _circuits(tmp_path, classes, answers)
    assert rc != 0 and sum(line.startswith(refused) for line in lines) == 1, lines
    assert any("prove_path.py window-circuit: refused at" in line for line in lines) == (answers.get("8,64,4") == "refuse")
    assert len(lines) == len(classes.split()) and limits == ["30m"] * len(classes.split()), (lines, limits)


@pytest.mark.parametrize("tied", ["count", "refuse"])
def test_under_h1_tie_step_3_writes_v3cs_record_reading_h1s_tables_after_the_classes_and_gates_nothing_on_it(tmp_path, tied):
    """Under H1_TIE the commit V3c holds is the tie's, whose circuit reads H1's W's tables, so after the classes CLASSES
    gates on, step 3 writes the verifier's record at V3C_CLASS reading them too (window-circuit --window-registration W,
    the window's H1 registration; its tag's suffix -tie), the record V3c holds that commit to. It gates nothing, and
    the class of record, which CLASSES doesn't name, is not counted after it."""
    rc, lines, limits = _circuits(tmp_path, "4,32,4", {"4,32,4": "count", "4,32,4+tie": tied}, H1_TIE="1", V3C_CLASS="4,32,4")
    d = tmp_path / "serve"
    argv = [json.loads(x) for x in (tmp_path / "tree" / "argv.jsonl").read_text().splitlines()]
    assert rc == 0 and limits == ["30m", "30m"], (rc, lines)
    assert [x[x.index("--out") + 1] for x in argv] == [str(d / f"window-circuit-w0-{t}.json")
                                                       for t in ("4-32-4", "4-32-4-tie")]
    assert [x[x.index("--window-registration") + 1] if "--window-registration" in x else None for x in argv] == [
        None, str(d / "h1" / "w0" / "registration.json")]
    head = "WINDOW-CIRCUIT class 4,32,4 reading H1's tables"
    assert lines[1].startswith(f"{head}: registered 7 bits" if tied == "count" else f"{head} refused (exit 2): "), lines


FAKE_INFERENCE_ONLY = '''import json, os, sys
open(os.environ["ORDER"], "a").write(json.dumps(sys.argv[1:]) + "\\n")
if os.environ.get("IO_FAIL"):
    print("inference_only.py: no torch", file=sys.stderr)
    sys.exit(2)
print("INFERENCE-ONLY POLICY at class 4,32,4: c 7 gates, k 3 fresh bits, n 16 sessions" if sys.argv[1] == "policy" else
      "INFERENCE-ONLY accept: session abab… (the registered one), 7 gates and 3 fresh bits a block, against c 7, k 3, n 16 "
      "(no count published: n is judged on the blocks' exact count, held on the node)")
'''
NCI_CHECK = "verity/protocols/inference_only/verifier/nci/lean/.lake/build/bin/nci-check"


def _inference_only(tmp_path: Path, *args: str, lean: bool = False, **extra: str) -> tuple[int, list[str], list[list[str]]]:
    """`served_verdicts.sh ARGS` in a toy tree whose inference_only.py notes its arguments and says a line (or, with
    IO_FAIL, exits 2 saying nothing), nci-check built where ``lean`` says (``extra`` adds to the environment): the exit
    status, the verdict lines, and each call's arguments."""
    tree, venv, d = tmp_path / "tree", tmp_path / "venv" / "bin", tmp_path / "serve"
    _script(venv / "python", f'exec {sys.executable} "$@"')
    for p in (tree / "benchmarks" / "one_stage", d):
        p.mkdir(parents=True)
    (tree / "benchmarks" / "one_stage" / "inference_only.py").write_text(FAKE_INFERENCE_ONLY)
    if lean:
        _script(tree / NCI_CHECK, "exit 0")
    env = {"PATH": os.defpath, "HOME": str(tmp_path), "VENV": str(venv.parent), "D": str(d), "OUT": str(tmp_path / "out"),
           "ORDER": str(tmp_path / "order"), "RESEARCH_SOURCE_SHA": SHA, "GIT_CEILING_DIRECTORIES": str(tmp_path), **extra}
    rc = subprocess.run(["bash", str(HERE.parent / "served_verdicts.sh"), *args], cwd=tree, env=env, check=False).returncode
    said, order = tmp_path / "out" / "verdicts.txt", tmp_path / "order"
    return (rc, said.read_text().splitlines() if said.exists() else [],
            [json.loads(x) for x in order.read_text().splitlines()] if order.exists() else [])


@pytest.mark.parametrize("extra, cls, lease", [
    ({}, "4,32,4", "60"),
    ({"CLASSES": "record 8,64,4", "LEASE_MIN": "150"}, "8,64,4", "150"),
    ({"CLASSES": "4,32,4", "IO_CLASS": "8,64,4"}, "8,64,4", "60"),
])
def test_the_inference_only_policy_is_written_at_the_first_class_but_record_from_the_runs_settings_alone(
        tmp_path, extra, cls, lease):
    """The policy reads the class (IO_CLASS, else the first of CLASSES but `record`), the model at its revision, the lease's
    limit (LEASE_MIN, the GPU lease's own) and the tree's commit, and nothing of a serve."""
    rc, lines, calls = _inference_only(tmp_path, "inference-policy", **extra)
    assert rc == 0 and lines == ["INFERENCE-ONLY POLICY at class 4,32,4: c 7 gates, k 3 fresh bits, n 16 sessions"]
    assert calls == [["policy", "--class", cls, "--model", "verity-int8/Llama-3.2-1B-INT8", "--revision",
                      "15b0976d400d796ee589e7737d03a919526bf800", "--lease-min", lease, "--tree", SHA, "--out",
                      str(tmp_path / "out" / "inference-only-policy.json")]]


def test_with_no_class_but_record_there_is_no_inference_only_policy_or_verdict(tmp_path):
    assert _inference_only(tmp_path / "policy", "inference-policy", CLASSES="record") == (
        0, ["INFERENCE-ONLY POLICY: none, since CLASSES names no class but record"], [])
    assert _inference_only(tmp_path / "verdict", "inference-only", "7", CLASSES="record") == (0, [], [])


@pytest.mark.parametrize("lean", [False, True])
def test_the_inference_only_verdict_judges_the_windows_record_at_the_class_with_nci_check_where_the_build_built_it(
        tmp_path, lean):
    """The verdict reads the policy, the serve dir, window K's circuit record at the class, its registration and the
    weights registration's pins, and writes the public verdict into the record and the full one in the serve dir; it
    runs `nci-check --count` only where the build built it. $OUT names no window."""
    rc, lines, calls = _inference_only(tmp_path, "inference-only", "7", lean=lean)
    d, out = tmp_path / "serve", tmp_path / "out"
    assert rc == 0 and lines == ["INFERENCE-ONLY accept: session abab… (the registered one), 7 gates and 3 fresh bits a "
                                 "block, against c 7, k 3, n 16 (no count published: n is judged on the blocks' exact "
                                 "count, held on the node)"]
    assert calls == [["verdict", "--policy", str(out / "inference-only-policy.json"), "--serve-run", str(d), "--circuit",
                      str(d / "window-circuit-w7-4-32-4.json"), "--registration", str(d / "advice" / "w7" / "registration.json"),
                      "--pins", str(d / "weights_registration.json"),
                      *(["--lean", str(tmp_path / "tree" / NCI_CHECK), "--lean-timeout", "1800"] if lean else []),
                      "--out", str(out / "record" / "inference-only.json"), "--private", str(d / "private" / "inference-only.json")]]
    assert not _window_data(out, 7)


def test_an_inference_only_verdict_that_fails_says_so_and_keeps_its_log_in_the_serve_dir(tmp_path):
    rc, lines, _calls = _inference_only(tmp_path, "inference-only", "7", IO_FAIL="1")
    assert (rc, lines) == (2, ["INFERENCE-ONLY failed (exit 2): its log in the serve dir, on the node"])
    assert (tmp_path / "serve" / "inference-only.err").read_text() == "inference_only.py: no torch\n"
    rc, lines, _calls = _inference_only(tmp_path / "policy", "inference-policy", IO_FAIL="1")
    assert (rc, lines) == (2, ["INFERENCE-ONLY POLICY failed (exit 2): inference_only.py: no torch"])


@pytest.mark.parametrize("n, said", [
    (40, "BLOCK-COUNTS class 4,32,4, the picked window alone: [block-counts] 2025-11-01: accept, against c "),
    (1, "BLOCK-COUNTS class 4,32,4, the picked window alone: [block-counts] 2025-11-01: refuse: the blocks' exact count "
        "is more than n = 1, against c "),
    (None, "BLOCK-COUNTS: none, since the run declares no c, k and n"),
])
def test_block_counts_publishes_the_judgment_against_the_policy_and_keeps_the_count_private(tmp_path, n, said):
    """`served_verdicts.sh block-counts 0` on a window of two slots (two blocks) at the class runs the real
    `verity-vllm block-counts` against the policy at IO_POLICY's default: its line and its public judgment hold n and no
    count, and the exact count is in the serve dir's private/ alone (0600); with no policy it says so and judges nothing."""
    from verity.definitions.registry.catalog import load
    from verity.commitments.windows import interface as IF

    load()
    gemm = 'Gemm_v2{K=64,N=8,DOT={"fn":"HopperBF16WgmmaDot16_v1"}}'
    entries = (IF.InferenceEntry(gemm, 0, (2, 2), ("slot", "-"), 0),)
    record = IF.descriptor_record(IF.Circuit("test/blocks", ((IF.resolve(gemm), 4),),
                                             blocks=IF.Blocks(slots=2, steps=3, inference=entries)), 1)
    tree, venv, d, out = tmp_path / "tree", tmp_path / "venv" / "bin", tmp_path / "serve", tmp_path / "out"
    _script(venv / "python", f'exec {sys.executable} "$@"')
    for p in (tree, d / "advice" / "w0", out):
        p.mkdir(parents=True)
    (d / "window-circuit-w0-4-32-4.json").write_text(json.dumps(record))
    (d / "advice" / "w0" / "registration.json").write_text("{}")
    (d / "serve.json").write_text(json.dumps({"clocks": {"time": 1_762_000_000}}))
    if n is not None:
        (out / "inference-only-policy.json").write_text(json.dumps({"c": 10 ** 12, "k": 10 ** 12, "n": n}))
    env = {"PATH": os.defpath, "HOME": str(tmp_path), "VENV": str(venv.parent), "D": str(d), "OUT": str(out),
           "CLASSES": "4,32,4", "RESEARCH_SOURCE_SHA": SHA, "GIT_CEILING_DIRECTORIES": str(tmp_path)}
    rc = subprocess.run(["bash", str(HERE.parent / "served_verdicts.sh"), "block-counts", "0"], cwd=tree, env=env,
                        check=False).returncode
    (line,) = (out / "verdicts.txt").read_text().splitlines()
    assert rc == 0 and line.startswith(said), line
    public, private = out / "block-counts-4-32-4.json", d / "private" / "block-counts-4-32-4.json"
    if n is None:
        assert not public.exists() and not private.exists()
        return
    doc, held = json.loads(public.read_text()), json.loads(private.read_text())
    assert (doc["verdict"], doc["n"]) == ("accept" if n == 40 else "refuse", n)
    assert not {"blocks", "total", "windows"} & set(doc) and not any("blocks" in c for c in doc["classes"])
    assert (held["total"], held["windows"], private.stat().st_mode & 0o777) == (2, 1, 0o600)
    assert " 2 blocks" not in line and "count 2" not in line


W63K4 = "8,32,4,4,16"
FAKE_V3C_PROVE_PATH = f'''import json, os, sys
from pathlib import Path
a = sys.argv[1:]
open(os.environ["ORDER"], "a").write(a[0] + "\\n")
if a[0] == "compiler":
    print(json.dumps({{"inner": {{"name": "toy", "digest": "ab" * 64}}}}))
    sys.exit(0)
out = Path(a[a.index("--out") + 1])
assert a[a.index("--class") + 1] == "{W63K4}"
# with W the circuit reads H1's tables, in the verifier's record and the commit alike
tables = {{"tables": "h1"}} if "--window-registration" in a else {{}}
if a[0] == "window-circuit":
    if tables:
        open(os.environ["ORDER"] + ".circuit", "w").write(json.dumps([out.name, *a[a.index("--window-registration"):]]))
    else:
        open(os.environ["ORDER"] + ".circuit", "w").write("\\n".join(a))
    out.write_text(json.dumps(tables))
    print(json.dumps({{"bits": 772, "count": {{"free": 3, "inputs": 10, "registered": 7, "units": 1}}}}))
else:
    assert a[0] == "window-commit"
    if os.environ.get("FAIL_COMMIT"):
        print("prove_path.py window-commit: the row dir holds no window 7", file=sys.stderr)
        sys.exit(2)
    out.mkdir(parents=True)
    (out / "circuit.json").write_text(json.dumps({{"run": "ab", **tables}}))
    open(os.environ["ORDER"] + ".law", "w").write(a[a.index("--law") + 1] if "--law" in a else "")
    if "--window-registration" in a:
        tail = a[a.index("--window-registration"):]
        open(os.environ["ORDER"] + ".commit", "w").write("\\n".join(tail))
        w = json.load(open(tail[1]))
        if "--pearl-passes" in a:
            from verity.protocols.spot.verifier.one_stage import registration as REG
            r = {{"format": "toy", "previous": "ef" * 64 if os.environ.get("BREAK_CHAIN") else REG.digest(w)}}
        else:
            r = {{"format": "toy", "after": w}}
        (out / "registration.json").write_text(json.dumps(r))
    print(json.dumps({{"run": "ab", "circuit": "cd", "population": 37, "sources": {{"pad": 3}}, "window": 7}}))
'''
FAKE_RECORDS_AUDIT = '''import json, os, sys
from pathlib import Path
a = sys.argv[1:]
open(os.environ["ORDER"], "a").write(a[0] + "\\n")
if a[0] == "warm":
    open(os.environ["ORDER"] + ".warm", "w").write(json.dumps(a))
    if os.environ.get("FAIL_WARM"):
        print("records_audit.py warm: a template's compose was killed", file=sys.stderr)
        sys.exit(1)
    print(json.dumps({"template": "a@0", "seconds": 1.5, "kept": False, "pin": "ab"}))
    print("WARM " + json.dumps({"templates": 3, "kept": 0, "seconds": 2.0}, sort_keys=True))
    sys.exit(0)
out = Path(a[a.index("--out") + 1])
if a[0] == "audit":
    assert "--tie" in a or "--tie-prefix" not in a
    open(os.environ["ORDER"] + ".audit", "w").write(json.dumps(a))
    if "--tie" in a:
        print("TIE " + json.dumps({"record": "cd" * 64, "chained": True, "previous": "cd" * 64, "values": []}))
    open(os.environ["ORDER"] + ".main", "w").write(open(a[a.index("--main") + 1]).read())
    out.mkdir(parents=True)
    (out / "summary.json").write_text("{}")
    print("RECORDS " + json.dumps({"verdict": "accept", "population": 37, "window": 7}))
    sys.exit(0)
assert a[0] == "period" and a[a.index("--audit") + 1] == str(out.parent / "audit")
if os.environ.get("FAIL_PERIOD"):
    print("records_audit.py period: the audit under it wrote no Program", file=sys.stderr)
    sys.exit(1)
out.write_text(json.dumps({"free": 0}))
print("PERIOD " + json.dumps({"verdict": "accept"}))
'''
TOY_MAIN = f"{'ef' * 20}\n{'01' * 20}\n"
CHAIN = 'echo chain >> "$ORDER"; printf "%s\\n" "$@" > "$ORDER.args"; echo "chain on window $2"'
PERIOD = 'echo period-check >> "$ORDER"; cp "$4/period.json" "$3/"; echo \'{"verdict": "accept"}\' > "$3/answer.json"; echo checked'


def _v3c(tmp_path: Path, args: tuple[str, ...] = ("v3c",), *, prove_path: str = FAKE_V3C_PROVE_PATH,
         records_audit: str = FAKE_RECORDS_AUDIT, lease: bool = True,
         **extra: str | None) -> tuple[int, list[str], list[str], list[str]]:
    """`served_verdicts.sh v3c` (or ``args``) on window 7 of a toy serve dir at the certifier's class, in a toy tree whose
    prove_path.py (``prove_path``) and records_audit.py (``records_audit``) note each step they run, its builds stubbed,
    main's first-parent line a toy MAIN_FILE, the split certifier's two commands set, a development run's local coins,
    no floor on the serve dir's free inodes and no composite, so the default COMPOSE_OFF_NODE=1 composes the final record
    alone off the node (``extra`` adds to the environment, None unsetting; ``lease`` False, no gpu-lease on the PATH): the
    exit status, the verdict lines, the steps in order and the limits."""
    tree, bin_, venv, d = tmp_path / "tree", tmp_path / "bin", tmp_path / "venv" / "bin", tmp_path / "serve"
    for p in (tree / "benchmarks" / "prove_path", tree / ".venv" / "bin", tree / "verity" / "protocols" / "flock" / "verifier",
              tree / "verity" / "protocols" / "spot" / "verifier" / "lean", tree / "verity" / "protocols" / "flock" / "pod", bin_, venv,
              d / "advice" / "w7" / "private", tmp_path / "row"):
        p.mkdir(parents=True, exist_ok=True)
    (tree / "benchmarks" / "prove_path" / "prove_path.py").write_text(prove_path)
    (tree / "benchmarks" / "prove_path" / "records_audit.py").write_text(records_audit)
    (tree / "verity" / "protocols" / "flock" / "pod" / "60-circuit.sh").write_text("exit 0\n")
    for f in (d / "advice" / "w7" / "private" / "advice.bin", tmp_path / "row" / "manifest.json",
              tmp_path / "row" / "build_summary.json", tmp_path / "h.so", tmp_path / "n.so"):
        f.write_text("")
    (d / "weights_registration.json").write_text("{}")
    if not (d / "windows.jsonl").exists():
        _write(d, "windows.jsonl", [{"window": 7, "run_root": "ab" * 64}])
    (tmp_path / "main.txt").write_text(TOY_MAIN)
    # the toy tree's verity/ directories, on the PYTHONPATH the script sets and as `python -`'s working directory, would
    # shadow the installed verity
    py = f'unset PYTHONPATH\nexport PYTHONSAFEPATH=1\n[ "$1 $2" = "-m research" ] && exit 0\nexec {sys.executable} "$@"'
    for path, body in ((venv / "python", py), (tree / ".venv" / "bin" / "python", py),
                       (bin_ / "timeout", f'[ "$1" = -k ] && shift 2\necho "$1" >> {tmp_path}/limits; shift; exec "$@"'),
                       *([(bin_ / "gpu-lease", "exit 99")] if lease else []), (bin_ / "uv", "exit 0"), (bin_ / "lake", "exit 0")):
        _script(path, body)
    env = {"PATH": f"{bin_}:{os.defpath}", "HOME": str(tmp_path), "VENV": str(venv.parent), "D": str(d), "K": "7",
           "OUT": str(tmp_path / "out"), "SERVE_ROOT": str(tmp_path / "root"), "FLOCK_WORK": str(tmp_path / "fw"),
           "ROW": str(tmp_path / "row"), "HIDDEN_SO": str(tmp_path / "h.so"), "NORM_TAP_SO": str(tmp_path / "n.so"),
           "ORDER": str(tmp_path / "order"), "H1_GATE": "", "V3C_CLASS": W63K4, "CLASSES": W63K4,
           "CERTIFIER_CHAIN_CMD": CHAIN, "CERTIFIER_PERIOD_CMD": PERIOD, "NODE_MEM_GB": "512",
           "MAIN_FILE": str(tmp_path / "main.txt"), "DEVELOPMENT": "1", "COMPOSITE": "", "PEARLPLUS_INODES": "0"}
    env = {k: v for k, v in {**env, **extra}.items() if v is not None}
    rc = subprocess.run(["bash", str(HERE.parent / "served_verdicts.sh"), *args], cwd=tree, env=env, check=False,
                        capture_output=True).returncode
    read = lambda f: (tmp_path / f).read_text().split() if (tmp_path / f).exists() else []  # noqa: E731
    return rc, (tmp_path / "out" / "verdicts.txt").read_text().splitlines(), read("order"), read("limits")


#: where V3c's audit's coins come from in `_v3c`'s development run, as `served_verdicts.sh v3c`'s first line says
V3C_DEV = "a development challenger session where it runs (V3C_CHALLENGER local), in a development run (DEVELOPMENT=1)"
V3C_ENTRY = f"COINS V3c's audit: {V3C_DEV}"


def test_v3c_runs_the_certifiers_chain_beside_its_audit_and_its_period_check_on_the_period_json_after(tmp_path):
    """After its COINS line, the window commit, then the split certifier's chain (it reads the commit's scan state)
    beside V3c's audit and its period.json, and the certifier's period check on it once both are done, each within its
    own limit; the check gets the serve dir, the window, the certifier's piece dir and V3c's dir, and writes the answer
    `pieces` reads there. The two steps' logs, which count the window's requests, end in the serve dir's
    private/pieces/certifier, not in $OUT, which leaves the node. $OUT then names no window and counts none of its own (the
    window commit's population, the audit's, the chain's window): each line is whole in the serve dir's
    private/verdicts.txt, each file in its private/out."""
    rc, lines, order, limits = _v3c(tmp_path)
    assert rc == 0, lines
    assert lines.pop(0) == V3C_ENTRY
    assert order[:2] == ["window-circuit", "window-commit"] and order[-1] == "period-check"
    assert sorted(order[2:-1]) == ["audit", "chain", "period"] and order.index("audit") < order.index("period")
    assert limits[:2] == ["30m", "120m"] and sorted(limits[2:-1]) == ["180m", "4h"] and limits[-1] == "60m"
    vw, pieces = tmp_path / "serve" / f"v3c-w7-{W63K4.replace(',', '-')}", tmp_path / "out" / "pieces" / "certifier"
    assert (tmp_path / "order.args").read_text().split() == [str(tmp_path / "serve"), "7", str(pieces), str(vw)]
    assert not _window_data(tmp_path / "out", 7), _window_data(tmp_path / "out", 7)
    held = (tmp_path / "serve" / "private" / "verdicts.txt").read_text()
    assert "chain on window 7" in held and '"population": 37' in held and "chain on window K" in "\n".join(lines)
    assert '"population": 37' in (tmp_path / "serve" / "private" / "out" / "v3c-audit" / "audit.log").read_text()
    assert json.loads((pieces / "answer.json").read_text()) == {"verdict": "accept"}
    assert json.loads((tmp_path / "out" / "v3c-audit" / "period.json").read_text()) == {"free": 0}
    heads = [line.split(" (")[0].split(":")[0] for line in lines]
    assert heads[:3] == [f"WINDOW-CIRCUIT class {W63K4}", f"WINDOW-COMMIT class {W63K4}",
                         f"WINDOW-CIRCUIT-HELD class {W63K4}"] and heads[-1] == "CERTIFIER-PERIOD"
    assert sorted(heads[3:-1]) == ["CERTIFIER-CHAIN", f"V3C-AUDIT class {W63K4}", "V3C-PERIOD"]
    assert lines[2].endswith("PASS") and lines[heads.index("V3C-PERIOD")].endswith('PERIOD {"verdict": "accept"}')
    private = tmp_path / "serve" / "private" / "pieces" / "certifier"
    assert (private / "chain.log").read_text().strip() == "chain on window 7"
    assert (private / "period.log").read_text().strip() == "checked"
    assert not (pieces / "chain.log").exists() and not (pieces / "period.log").exists()


def test_v3c_with_no_period_json_runs_no_period_check_and_fails(tmp_path):
    """V3c's audit wrote no period.json: no period check, and `served_verdicts.sh v3c` exits 1 once it has moved the
    chain's log to the serve dir's private/pieces/certifier."""
    rc, lines, order, _limits = _v3c(tmp_path, FAIL_PERIOD="1")
    assert rc == 1 and order[:2] == ["window-circuit", "window-commit"]
    assert sorted(order[2:]) == ["audit", "chain", "period"] and order.index("audit") < order.index("period")
    assert any("the audit under it wrote no Program" in x for x in lines[-3:-1])
    assert lines[-1] == "CERTIFIER-PERIOD: not run: V3c wrote no period.json"
    pieces = tmp_path / "out" / "pieces" / "certifier"
    assert not (pieces / "answer.json").exists() and not (pieces / "chain.log").exists()
    assert (tmp_path / "serve" / "private" / "pieces" / "certifier" / "chain.log").exists()


@pytest.mark.parametrize("args, extra, line", [
    (("v3c",), {"CERTIFIER_CMD": "true", "CERTIFIER_PERIOD_CMD": None},
     "PREFLIGHT: CERTIFIER_CMD or the split's CERTIFIER_CHAIN_CMD and CERTIFIER_PERIOD_CMD, not both"),
    ((), {"V3C_CLASS": "", "CERTIFIER_CHAIN_CMD": None},
     "PREFLIGHT: the split certifier's period check reads V3c's period.json, and V3C_CLASS is empty"),
])
def test_the_split_certifier_needs_v3c_and_no_whole_piece_beside_it(tmp_path, args, extra, line):
    """The split's period check writes the certifier's answer from V3c's period.json, so the preflight refuses it beside
    CERTIFIER_CMD, the whole piece's command, and in a run that leaves V3c out, before anything runs."""
    rc, lines, order, _limits = _v3c(tmp_path, args, **extra)
    assert (rc, lines, order) == (3, [line], [])


FAKE_ADVICE = '''import json, os, sys
from pathlib import Path
a = sys.argv[1:]
open(os.environ["ORDER"], "a").write("advice\\n")
k, out = a[a.index("--window") + 1], Path(a[a.index("--out") + 1])
assert a[a.index("--compiler-name") + 1] == "toy" and "--weights-of-record" in a
if os.environ.get("ADVICE_RC") == "3":
    print(f"[advice] commit window {k}: pending: request r1 of window {k} has no Commit record yet")
    sys.exit(3)
(out / f"w{k}" / "private").mkdir(parents=True)
(out / f"w{k}" / "registration.json").write_text("{}")
(out / f"w{k}" / "private" / "advice.bin").write_bytes(b"\\x01")
(out / "windows.json").write_text("{}")
'''
W_OF_7 = {"format": "verity/registration/v2", "window": 7, "previous": None}


def _tie(tmp_path: Path, **extra: str) -> tuple[subprocess.CompletedProcess, list[str]]:
    """The ``--h1-window-commit`` that `served_verdicts.sh serve` gives serve-openai under H1_TIE, run as `serve_tie` runs
    it at window 7's close (``bash -c CMD _ SERVE_DIR K W_FILE R_FILE PROVER_DIR``, from elsewhere), in a toy tree whose
    prove_path.py and advice note each step they run (``extra`` adds to the environment): the run and the steps in order."""
    rc, argv = _serve_argv(tmp_path, {"H1_GATE": GATE, "H1_CIRCUITS": str(tmp_path / "circuits"), "H1_TIE": "1"})
    assert rc == 10 and "--h1-prover" in argv and argv.count("--h1-window-commit") == 1, (rc, argv)
    cmd = argv[argv.index("--h1-window-commit") + 1]
    tree, d, venv = tmp_path / "tree", tmp_path / "serve", tmp_path / "venv" / "bin"
    (tree / "benchmarks" / "prove_path").mkdir(parents=True)
    (tree / "benchmarks" / "prove_path" / "prove_path.py").write_text(FAKE_V3C_PROVE_PATH)
    (tree / "benchmarks" / "one_stage").symlink_to(HERE.parent, target_is_directory=True)
    (tmp_path / "advice.py").write_text(FAKE_ADVICE)
    _script(venv / "python", f'[ "$1 $2 $3" = "-m verity_vllm.pipeline.cli advice" ] && {{ shift 3; exec {sys.executable} '
                             f'{tmp_path}/advice.py "$@"; }}\nunset PYTHONPATH\nexport PYTHONSAFEPATH=1\nexec {sys.executable} "$@"')
    for p in (d / "h1" / "w7", d / "private" / "h1" / "w7"):
        p.mkdir(parents=True, exist_ok=True)
    (d / "weights_registration.json").write_text("{}")
    (d / "h1" / "w7" / "registration.json").write_text(json.dumps(W_OF_7))
    env = {"PATH": os.defpath, "HOME": str(tmp_path), "VENV": str(venv.parent), "D": str(d), "OUT": str(tmp_path / "tie-out"),
           "ORDER": str(tmp_path / "order"), "H1_TIE": "1", "V3C_CLASS": W63K4, **extra}
    run = subprocess.run(["bash", "-c", cmd, "_", str(d), "7", str(d / "h1" / "w7" / "registration.json"),
                          str(d / "h1" / "w7" / "record.json"), str(d / "private" / "h1" / "w7")],
                         cwd=tmp_path, env=env, check=False, capture_output=True, text=True)
    return run, (tmp_path / "order").read_text().split() if (tmp_path / "order").exists() else []


def test_under_h1_tie_each_h1_windows_close_writes_its_advice_and_commits_it_chained_after_its_w_as_its_record_r(tmp_path):
    """serve-openai's tie command, from any directory, writes the window's advice where step 3 reads it, then runs
    window-commit at V3C_CLASS on W, the window's H1 prover key and V3c's law, and copies the registration it wrote to R.
    It says nothing in $OUT (its window is H1's, which no published line names); its commit's answer stays in the serve dir."""
    run, order = _tie(tmp_path)
    d = tmp_path / "serve"
    assert run.returncode == 0 and order == ["compiler", "advice", "window-commit"], (run.stdout, run.stderr, order)
    assert run.stdout.splitlines()[-1] == f"TIE: window-commit class {W63K4}: its registration is the window's record R"
    vw = d / f"v3c-w7-{W63K4.replace(',', '-')}" / "window"
    assert json.loads((d / "h1" / "w7" / "record.json").read_text()) == json.loads((vw / "registration.json").read_text()) \
        == {"format": "toy", "after": W_OF_7}
    assert (tmp_path / "order.commit").read_text().split() == [
        "--window-registration", str(d / "h1" / "w7" / "registration.json"), "--h1-key", str(d / "private" / "h1" / "w7" / "key"),
        "--law", "stratified:1"]
    assert (d / "advice" / "w7" / "private" / "advice.bin").exists() and (d / "advice" / ".w7" / "windows.json").exists()
    assert json.loads((d / "h1" / "w7" / "window-commit.json").read_text())["population"] == 37
    assert not (tmp_path / "tie-out" / "verdicts.txt").exists()


@pytest.mark.parametrize("extra, rc, steps, said", [
    ({"ADVICE_RC": "3"}, 14, ["compiler", "advice"],
     "TIE: the window's advice is pending: a request of it runs on past its close, with no Commit record yet"),
    ({"FAIL_COMMIT": "1"}, 16, ["compiler", "advice", "window-commit"],
     f"TIE: window-commit class {W63K4} refused (exit 2): prove_path.py window-commit: the row dir holds no window 7"),
])
def test_a_tie_whose_advice_is_pending_or_whose_commit_is_refused_writes_no_record_r(tmp_path, extra, rc, steps, said):
    """A request carried past the window's close has no Commit record until a later close, so its advice can't be written
    before the window's sample coin: the tie fails at once, never polls; a refused window-commit fails it too. Either way
    no R, so `serve_tie` refuses the window's record and H1 draws no sample coin."""
    run, order = _tie(tmp_path, **extra)
    assert (run.returncode, order, run.stdout.splitlines()[-1]) == (rc, steps, said), run.stderr
    assert not (tmp_path / "serve" / "h1" / "w7" / "record.json").exists()


#: R's sample in `_tied_commit`: stratified:1 over three tables' readers, a proof block of each
TIED_LAW = {"law": "stratified", "k": 1, "strata": [{"template": t, "units": [[3 * i, 3 * i + 3]], "k": 1}
                                                   for i, t in enumerate(("a@0", "ids@0", "ids@1"))]}


def _tied_commit(tmp_path: Path, bound: str | None, law: dict | None = TIED_LAW) -> None:
    """Window 7's commit as the tie left it in the toy serve dir: its registration R (its sample ``law``; None, none) and
    circuit, which reads H1's tables, its answer, and H1's log naming W and, unless ``bound`` is None, the record H1 bound
    (``same``: R; ``other``: another)."""
    d = tmp_path / "serve"
    vw = d / f"v3c-w7-{W63K4.replace(',', '-')}" / "window"
    vw.mkdir(parents=True)
    (d / "h1" / "w7").mkdir(parents=True)
    r = {"format": "toy", "after": W_OF_7, **({"law": law} if law else {})}
    (vw / "registration.json").write_text(json.dumps(r))
    (vw / "circuit.json").write_text(json.dumps({"run": "ab", "tables": "h1"}))
    (d / "h1" / "w7" / "window-commit.json").write_text(json.dumps({"run": "ab", "circuit": "cd", "population": 37}))
    log = [{"window": 7, "kind": "registration", "file": "w7/registration.json", "digest": "aa" * 64}]
    if bound is not None:
        log.append({"window": 7, "kind": "record", "file": "w7/record.json",
                    "digest": REG.digest(r if bound == "same" else {**r, "after": None})})
    _write(d / "h1", "log.jsonl", log)


def test_under_h1_tie_v3c_audits_the_commit_h1_bound_and_makes_none_of_its_own(tmp_path):
    """The window's one commit is the one its close made: V3c holds its registration to the record H1 bound (the log's
    digest) and its circuit to the verifier's, which reads H1's W's tables as the commit's does (window-circuit
    --window-registration W, its own record beside the untied one), runs no window-commit, and goes on to its audit."""
    _tied_commit(tmp_path, "same")
    rc, lines, order, _limits = _v3c(tmp_path, H1_TIE="1")
    assert rc == 0 and "window-commit" not in order and order[0] == "window-circuit" and "audit" in order, (lines, order)
    d = tmp_path / "serve"
    assert json.loads((tmp_path / "order.circuit").read_text()) == [
        f"window-circuit-w7-{W63K4.replace(',', '-')}-tie.json", "--window-registration", str(d / "h1" / "w7" / "registration.json")]
    assert not (d / f"window-circuit-w7-{W63K4.replace(',', '-')}.json").exists()
    assert lines.pop(0) == V3C_ENTRY
    assert lines[0].startswith(f"WINDOW-CIRCUIT class {W63K4} reading H1's tables: registered 7 bits"), lines
    assert lines[1].startswith(f"WINDOW-COMMIT class {W63K4} at H1's close, its registration the Commit record H1 bound "
                               "before the sample coin (R ") and '"run": "ab"' in lines[1] and "population" not in lines[1]
    assert lines[2] == (f"WINDOW-CIRCUIT-HELD class {W63K4}: the committed circuit.json less its run fields is the "
                        "verifier's own window-circuit record: PASS")
    assert "--pearl-tables" not in (tmp_path / "order.circuit").read_text().split("\n")


#: the flags naming the challenger whose coins records_audit's audit takes
COIN_FLAGS = ("--challenger", "--peer", "--challenger-key", "--gate-key")


def _audited(tmp_path: Path, flags: tuple[str, ...] = ("--law", "--tie", "--tie-prefix", "--ports")) -> dict[str, str | None]:
    """The flags `flags` the toy records_audit.py's audit got (by default the sampling design's, the tie's and the ports')."""
    a = json.loads((tmp_path / "order.audit").read_text())
    return {f: a[a.index(f) + 1] if f in a else None for f in flags}


@pytest.mark.parametrize("tied, extra, ports", [(False, {}, "16"), (True, {}, "3"), (True, {"V3C_PORTS": "40"}, "40")])
def test_under_h1_tie_v3cs_audit_is_tied_to_h1s_w_at_a_stratum_of_each_tables_readers_on_a_port_a_sampled_block(
        tmp_path, tied, extra, ports):
    """Under H1_TIE the tables the window circuit reads are H1's W's, so V3c's audit is tied to W (records_audit's --tie,
    with H1's names' prefix h1/) at stratified:1, a stratum of each table's readers, so every table has a sampled reader,
    on as many ports as R's sample has proof blocks (its strata's k summed) unless V3C_PORTS says; an untied window's
    audit is at subset:1 on 16 ports."""
    if tied:
        _tied_commit(tmp_path, "same")
    rc, lines, order, _limits = _v3c(tmp_path, **({"H1_TIE": "1"} if tied else {}), **extra)
    assert rc == 0 and "audit" in order, (lines, order)
    w = str(tmp_path / "serve" / "h1" / "w7" / "registration.json")
    assert _audited(tmp_path) == ({"--law": "stratified:1", "--tie": w, "--tie-prefix": "h1/", "--ports": ports} if tied
                                  else {"--law": "subset:1", "--tie": None, "--tie-prefix": None, "--ports": ports})


DAYS = ("--day", "--day-log", "--day-params")


def _day_tree(tmp_path: Path) -> str:
    """A toy tree with serve-openai's --day (`serve_day`), and its records_audit.py, which takes --day."""
    p = tmp_path / "tree" / "adapters" / "vllm" / "verity_vllm" / "pipeline" / "serve_day.py"
    p.parent.mkdir(parents=True)
    p.write_text("")
    return FAKE_RECORDS_AUDIT + '# "--day"\n'


@pytest.mark.parametrize("extra, flags", [
    ({}, {"--day": None, "--day-log": None, "--day-params": None}),
    ({"DAY": "3", "DAY_LOG": "/l.json", "DAY_PROGRAM": "5d" * 64}, {"--day": "3", "--day-log": "/l.json", "--day-params": None}),
    ({"DAY": "3", "DAY_LOG": "/l.json", "DAY_PROGRAM": "5d" * 64, "DAY_PARAMS": "/p.json"},
     {"--day": "3", "--day-log": "/l.json", "--day-params": "/p.json"}),
], ids=["off", "day", "day-params"])
def test_under_day_v3cs_audit_holds_the_window_to_the_days_declaration(tmp_path, extra, flags):
    """DAY hands V3c's audit the day and the registration log (records_audit's --day, --day-log, --day-params), whose
    declarations R2 holds the window's registration to; without DAY the audit gets none of them."""
    rc, lines, order, _limits = _v3c(tmp_path, records_audit=_day_tree(tmp_path), **extra)
    assert rc == 0 and "audit" in order, (lines, order)
    assert _audited(tmp_path, DAYS) == flags


NO_DAY = ("PREFLIGHT: DAY=N declares day N's circuit: it needs DAY_LOG (the deployment's registration log) and DAY_PROGRAM "
          "(the SHA-512, lowercase hex, of the program its windows register)")


@pytest.mark.parametrize("extra, tree, line", [
    ({"DAY": "3", "DAY_PROGRAM": "5d" * 64}, True, NO_DAY),
    ({"DAY": "3", "DAY_LOG": "/l.json", "DAY_PROGRAM": "5D" * 64}, True, NO_DAY),
    ({"DAY": "x", "DAY_LOG": "/l.json", "DAY_PROGRAM": "5d" * 64}, True, NO_DAY),
    ({"DAY": "3", "DAY_LOG": "/l.json", "DAY_PROGRAM": "5d" * 64}, False,
     "PREFLIGHT: DAY needs serve-openai's --day and records_audit's --day, which this tree hasn't"),
], ids=["no-log", "upper-hex", "not-a-day", "old-tree"])
def test_the_preflight_refuses_a_day_it_cant_declare_or_audit(tmp_path, extra, tree, line):
    rc, lines, order, _limits = _v3c(tmp_path, records_audit=_day_tree(tmp_path) if tree else FAKE_RECORDS_AUDIT, **extra)
    assert (rc, lines, order) == (3, [line], [])


@pytest.mark.parametrize("extra, law", [({}, "subset:1"), ({"V3C_LAW": "stratified:1"}, "stratified:1")])
def test_an_untied_v3c_registers_its_window_at_the_sampling_design_its_audit_samples_by(tmp_path, extra, law):
    """Untied, window-commit registers the window at V3C_LAW's sampling design and the audit samples by that one: R1 holds
    a sample to the registration's design, so a commit at subset:1 under a stratified:1 audit is refused before its sample."""
    rc, lines, order, _limits = _v3c(tmp_path, **extra)
    assert rc == 0 and order.index("window-commit") < order.index("audit"), (lines, order)
    assert (tmp_path / "order.law").read_text() == law and _audited(tmp_path, ("--law",)) == {"--law": law}


def test_v3c_on_an_existing_serve_dir_needs_no_row_dir_taps_or_gpu_lease(tmp_path):
    """`served_verdicts.sh v3c` serves nothing, so a CPU node with the audit's memory runs it: its preflight asks for the
    venv's python alone, not the row dir, the FlashAttention and norm-tap builds or gpu-lease that serving reads."""
    gone = {"ROW": str(tmp_path / "no-row"), "HIDDEN_SO": str(tmp_path / "no-h.so"), "NORM_TAP_SO": str(tmp_path / "no-n.so")}
    rc, lines, order, _limits = _v3c(tmp_path, lease=False, **gone)
    assert rc == 0 and "audit" in order and not any(x.startswith("PREFLIGHT") for x in lines), (lines, order)
    rc, lines, _order, _limits = _v3c(tmp_path / "no-venv", lease=False, VENV=str(tmp_path / "no-venv" / "none"), **gone)
    assert rc == 3 and lines[-1].startswith(f"PREFLIGHT: no {tmp_path / 'no-venv' / 'none'}/bin/python"), lines


@pytest.mark.parametrize("tied, extra, limit", [
    (False, {}, None), (False, {"V3C_WARM_JOBS": "2"}, "180m"),
    (True, {"V3C_WARM_JOBS": "3", "V3C_WARM_TIMEOUT": "5h"}, "5h"),
])
def test_v3c_warm_composes_the_windows_templates_before_its_audit_which_then_takes_their_cache(tmp_path, tied, extra, limit):
    """With V3C_WARM_JOBS, records_audit's warm composes every template of the committed window into its compose cache,
    that many at a time, at the audit's sampling design, within V3C_WARM_TIMEOUT (default V3C_AUDIT_TIMEOUT), before the
    audit, which takes the cache and so composes none; without it, no warm and no cache."""
    if tied:
        _tied_commit(tmp_path, "same")
    rc, lines, order, limits = _v3c(tmp_path, **({"H1_TIE": "1"} if tied else {}), **extra)
    assert rc == 0 and "audit" in order, (lines, order)
    audit = json.loads((tmp_path / "order.audit").read_text())
    if limit is None:
        assert "warm" not in order and "--compose-cache" not in audit and not (tmp_path / "order.warm").exists()
        assert not any(line.startswith("V3C-WARM") for line in lines)
        return
    vw, jobs = tmp_path / "serve" / f"v3c-w7-{W63K4.replace(',', '-')}", extra["V3C_WARM_JOBS"]
    assert order.count("warm") == 1 and order.index("warm") < order.index("audit"), order
    assert json.loads((tmp_path / "order.warm").read_text()) == [
        "warm", "--window", str(vw / "window"), "--law", "stratified:1" if tied else "subset:1",
        "--compose-cache", str(vw / "compose-cache"), "--jobs", jobs]
    assert audit[audit.index("--compose-cache") + 1] == str(vw / "compose-cache")
    assert limits.count(limit) == 1 + (limit == "180m"), limits
    assert f'V3C-WARM class {W63K4} (records_audit.py warm exit 0, {jobs} at a time): WARM ' \
           '{"kept": 0, "seconds": 2.0, "templates": 3}' in lines


def test_a_warm_that_fails_says_why_and_the_audit_composes_what_it_did_not_keep(tmp_path):
    """The warm only makes the audit's composes early: one that fails is said, and the audit, taking the cache, composes the
    templates it holds no circuit of."""
    rc, lines, order, _limits = _v3c(tmp_path, V3C_WARM_JOBS="2", FAIL_WARM="1")
    assert rc == 0 and order.index("warm") < order.index("audit"), (lines, order)
    assert f"V3C-WARM class {W63K4} (records_audit.py warm exit 1, 2 at a time): records_audit.py warm: a template's " \
           "compose was killed; the audit composes the templates it didn't keep" in lines
    audit = json.loads((tmp_path / "order.audit").read_text())
    assert "--compose-cache" in audit


@pytest.mark.parametrize("jobs, records_audit, line", [
    ("0", FAKE_RECORDS_AUDIT, "PREFLIGHT: V3C_WARM_JOBS is how many of the window's templates warm composes at once, a "
                              "positive integer, not 0"),
    ("two", FAKE_RECORDS_AUDIT, "PREFLIGHT: V3C_WARM_JOBS is how many of the window's templates warm composes at once, a "
                                "positive integer, not two"),
    ("2", "import sys\n", "PREFLIGHT: V3C_WARM_JOBS composes the window's templates before its audit with records_audit.py "
                          "warm, which this tree's records_audit.py doesn't have"),
])
def test_the_preflight_refuses_a_warm_at_no_positive_number_of_jobs_or_on_a_tree_without_it(tmp_path, jobs, records_audit,
                                                                                             line):
    """V3C_WARM_JOBS is a count of composes at once, and a tree whose records_audit.py has no warm is refused, not run
    with every compose inside the audit's limit: before anything runs."""
    rc, lines, order, _limits = _v3c(tmp_path, records_audit=records_audit, V3C_WARM_JOBS=jobs)
    assert (rc, lines, order) == (3, [line], [])


@pytest.mark.parametrize("given", ["MAIN_FILE", "MAIN_SHA"])
def test_v3cs_audit_samples_on_a_local_challenger_session_pinned_on_mains_first_parent_line(tmp_path, given):
    """records_audit samples only on the auditor's challenger's coins, so a development run's V3c audit takes a local
    development session's (--challenger local, no peer), pinned on main's first-parent line: MAIN_FILE's, else MAIN_SHA's
    in the tree, kept as $OUT/v3c-audit/main.txt, the --main the audit reads."""
    extra, line = {}, TOY_MAIN
    if given == "MAIN_SHA":
        tree = tmp_path / "tree"
        tree.mkdir()
        git = lambda *a: subprocess.run(["git", "-C", str(tree), "-c", "user.name=t", "-c", "user.email=t@t", *a],  # noqa: E731
                                        check=True, capture_output=True, text=True).stdout
        git("init", "-q")
        for m in ("one", "two", "three"):
            git("commit", "-q", "--allow-empty", "-m", m)
        extra, line = {"MAIN_FILE": None, "MAIN_SHA": git("rev-parse", "HEAD~1").strip()}, git("rev-list", "--first-parent", "HEAD~1")
    rc, lines, order, _limits = _v3c(tmp_path, **extra)
    assert rc == 0 and "audit" in order, (lines, order)
    assert _audited(tmp_path, COIN_FLAGS) == {"--challenger": "local", "--peer": None, "--challenger-key": None,
                                              "--gate-key": None}
    assert len(line.split()) == 2 and (tmp_path / "order.main").read_text() == line
    assert (tmp_path / "out" / "v3c-audit" / "main.txt").read_text() == line


@pytest.mark.parametrize("main, head, tail", [
    ("gone.txt", "V3C-AUDIT: MAIN_FILE ", ", main's first-parent line its challenger session pins on, is missing or empty"),
    (None, "V3C-AUDIT: no first-parent line of main for its challenger session to pin on (MAIN_FILE unset, and HEAD has "
           "none in this tree): set MAIN_FILE", ""),
])
def test_v3c_runs_no_audit_without_a_line_of_main_for_its_session_to_pin_on(tmp_path, main, head, tail):
    """A MAIN_FILE that isn't there, or none in a tree without MAIN_SHA's line, leaves the session nothing to pin on: V3c
    says so and runs no audit, which records_audit would refuse."""
    rc, lines, order, _limits = _v3c(tmp_path, MAIN_FILE=main and str(tmp_path / main),
                                     GIT_CEILING_DIRECTORIES=str(tmp_path))
    assert rc == 1 and "audit" not in order, (lines, order)
    assert sum(x.startswith(head) and x.endswith(tail) for x in lines) == 1, lines


LOCAL_COINS = "a local development challenger session on vy-dc-h200, the served node, not an auditor's (PP_CHALLENGER local)"
V3C_COINS = f" V3c's audit: {V3C_DEV};"
LIVE = "gate:research@auditor"
LIVE_COINS = f"the challenger {LIVE} (peer vy-dc-h200)"
V3C_LIVE = f"the challenger {LIVE} (peer vy-dc-h200), the window's run root's one session, the sampled-proofs audit of record"
STEP3_LOCAL = ("a local development challenger session on vy-dc-h200, the served node, for step 3, a development check, since "
               "V3c's audit's live session takes the window's run root, which the gate opens once, ever")
NO_V3C = {"V3C_CLASS": "", "CERTIFIER_CHAIN_CMD": None, "CERTIFIER_PERIOD_CMD": None}
#: in a parametrized environment, the path of the challenger key record `_key` writes
KEYED = "<challenger key record>"


def _key(tmp_path: Path) -> Path:
    """A challenger key record for the preflight to find readable (the toy records_audit.py reads none)."""
    (key := tmp_path / "challenger-key.json").write_text('{"schema": "challenger-key/v1"}')
    return key


@pytest.mark.parametrize("extra, coins, v3c", [
    ({}, LOCAL_COINS, V3C_COINS),
    ({"PP_CHALLENGER": LIVE, "PP_PEER": "vy-dc-h200", "V3C_CHALLENGER": "local"}, LIVE_COINS, V3C_COINS),
    ({"PP_CHALLENGER": LIVE, "PP_PEER": "vy-dc-h200", "PP_CHALLENGER_KEY": KEYED}, STEP3_LOCAL, f" V3c's audit: {V3C_LIVE};"),
    (NO_V3C, LOCAL_COINS, ""),
    ({**NO_V3C, "PP_CHALLENGER": LIVE, "PP_PEER": "vy-dc-h200"}, LIVE_COINS, ""),
    ({**NO_V3C, "DEVELOPMENT": None, "PP_CHALLENGER": LIVE, "PP_PEER": "vy-dc-h200"}, LIVE_COINS, ""),
])
def test_the_coins_line_says_whose_coins_each_sample_is_on(tmp_path, extra, coins, v3c):
    """Before step 0 the run says whose coins each sample is on: sampled proofs' (run.sh's local development session
    unless PP_CHALLENGER names a challenger, and then too while V3c's audit takes its coins there, since both open on the
    window's run root), PoUW H1's, and V3c's audit's when it runs (V3C_CHALLENGER, by default PP_CHALLENGER). Here step
    0's `predict row` then fails, and the run stops there."""
    venv = tmp_path / "toy-venv" / "bin"
    _script(venv / "python", '[ "$1 $2" = "-m research" ] && exit 0\n[ "$1 $2" = "-m verity_vllm.pipeline.cli" ] && exit 9\n'
                             f'unset PYTHONPATH\nexport PYTHONSAFEPATH=1\nexec {sys.executable} "$@"')
    extra = {k: str(_key(tmp_path)) if v == KEYED else v for k, v in extra.items()}
    rc, lines, _order, _limits = _v3c(tmp_path, (), VENV=str(venv.parent), HOST="vy-dc-h200",
                                      PATH=f"{_hosts(tmp_path)}:{tmp_path / 'bin'}:{os.defpath}", **extra)
    assert rc == 17 and lines[-1] == "STOP at predict (exit 17)", lines
    assert f"COINS sampled proofs: {coins}; PoUW H1: not served (H1_GATE empty);{v3c} the bandwidth audit's: " \
           "vy-auditor-2's bw_party, inside PoUS's piece" in lines, lines


#: infra's gate behind `ssh`, for V3c's live challenger: its ledger the test's ($HOME/ledger.jsonl) and its pins pinning
#: TOY_MAIN's head; each word sent noted in $ORDER (gate-WORD) and each call's arguments in $ORDER.ssh
FAKE_SSH = ('for w; do :; done\necho "gate-$w" >> "$ORDER"\nprintf "%s\\n" "$*" >> "$ORDER.ssh"\ncase "$w" in\n'
            'ledger) cat "$HOME/ledger.jsonl" ;;\n'
            f"""pins) echo '{{"commit": "{"ef" * 20}", "main": "{"ef" * 20}", "pinned_utc": "2026-10-26T21:35:00Z"}}' ;;\n"""
            "*) exit 2 ;;\nesac")
OPENED = json.dumps({"kind": "open", "session": "s1", "schema": "verity/challenger-auditor/v2",
                     "body": {"window": {"root": "ab" * 64}}})
PERIOD_OPEN = json.dumps({"kind": "period-open", "session": None, "body": {"period": "p1", "lane": "vllm"}})


def _live(tmp_path: Path, ledger: list[str], hosts: bool = False) -> dict[str, str | None]:
    """`_v3c`'s environment for a run of record (DEVELOPMENT unset) whose V3c audit takes its coins from the live
    challenger LIVE from peer vy-dc-h200 under a key record and a gate key, the gate a fake `ssh` (FAKE_SSH) whose ledger
    is `ledger`; with `hosts`, `hostname` says $HOST."""
    _script(tmp_path / "gate" / "ssh", FAKE_SSH)
    (tmp_path / "ledger.jsonl").write_text("".join(x + "\n" for x in ledger))
    (tmp_path / "gate_ed25519").write_text("")
    path = [*([str(_hosts(tmp_path))] if hosts else []), str(tmp_path / "gate"), str(tmp_path / "bin"), os.defpath]
    return {"DEVELOPMENT": None, "PP_CHALLENGER": LIVE, "PP_PEER": "vy-dc-h200", "PP_CHALLENGER_KEY": str(_key(tmp_path)),
            "PP_GATE_KEY": str(tmp_path / "gate_ed25519"), "PATH": ":".join(path)}


def test_v3cs_audit_samples_on_the_live_challenger_once_the_gate_holds_no_session_on_its_run_root(tmp_path):
    """V3c's audit is the sampled-proofs audit of record: with PP_CHALLENGER the live challenger it takes its coins
    there (records_audit's --challenger, --peer PP_PEER, --challenger-key and --gate-key), as its COINS line says, once
    `served_verdicts.py gate --root`, through the same ssh and key, finds no session on the window's run root and no
    period of the lane open, which its V3C-GATE line says, and when, before the audit reserves anything."""
    env = _live(tmp_path, [LEDGER_LINE])
    rc, lines, order, _limits = _v3c(tmp_path, **env)
    assert rc == 0 and order.index("gate-ledger") < order.index("gate-pins") < order.index("audit"), (lines, order)
    assert lines[0] == f"COINS V3c's audit: {V3C_LIVE}"
    gate = [x for x in lines if x.startswith("V3C-GATE")]
    assert len(gate) == 1 and gate[0].startswith(
        f"V3C-GATE ready: lane vllm's open period None; root {'ab' * 8}… has no session yet; the ledger's 1 entries; "
        f"pinned {'ef' * 6} (main {'ef' * 6}, 2026-10-26T21:35:00Z), on the verifier's first-parent line of main (checked "), gate
    assert re.fullmatch(r".* \(checked \d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ\)", gate[0]), gate
    assert _audited(tmp_path, COIN_FLAGS) == {"--challenger": LIVE, "--peer": "vy-dc-h200",
                                              "--challenger-key": env["PP_CHALLENGER_KEY"], "--gate-key": env["PP_GATE_KEY"]}
    assert (tmp_path / "order.ssh").read_text().splitlines() == [
        f"-o BatchMode=yes -o IdentitiesOnly=yes -i {env['PP_GATE_KEY']} research@auditor {w}" for w in ("ledger", "pins")]


@pytest.mark.parametrize("ledger, said", [
    ([LEDGER_LINE, OPENED], f"V3C-GATE refused: lane vllm's open period None; root {'ab' * 8}… already has its session s1 "
                            "(one session a root, ever); the ledger's 2 entries;"),
    ([PERIOD_OPEN], f"V3C-GATE refused: lane vllm's open period p1 (a reserve would join it); root {'ab' * 8}… has no "
                    "session yet; the ledger's 1 entries;"),
])
def test_v3c_reserves_no_session_on_a_run_root_the_gate_opened_or_while_its_lane_has_a_period_open(tmp_path, ledger, said):
    """The gate opens a root once, ever, and a reserve joins the lane's open period: V3c's audit runs on neither, so a run
    root with its session already (step 3's, or an earlier audit's) or a lane with a period open gets no audit, and no
    retry, and its V3C-GATE line says why."""
    rc, lines, order, _limits = _v3c(tmp_path, **_live(tmp_path, ledger))
    assert rc == 1 and "audit" not in order and order.count("gate-ledger") == 1, (lines, order)
    assert sum(x.startswith(said) for x in lines) == 1, lines


def _not_development(challenger: str) -> str:
    return ("PREFLIGHT: V3c's audit is the sampled-proofs audit of record, and V3C_CHALLENGER (default PP_CHALLENGER) is "
            f"{challenger}, a development session, which only a development run takes: set PP_CHALLENGER=gate:USER@HOST "
            "with PP_PEER, PP_CHALLENGER_KEY and PP_GATE_KEY, or DEVELOPMENT=1")


@pytest.mark.parametrize("extra, line", [
    ({"DEVELOPMENT": None}, _not_development("local")),
    ({"DEVELOPMENT": None, "V3C_CHALLENGER": "loopback:127.0.0.1:7000"}, _not_development("loopback:127.0.0.1:7000")),
    ({"DEVELOPMENT": "yes"}, "PREFLIGHT: DEVELOPMENT is 1, a development run, or unset, not yes"),
    ({"V3C_CHALLENGER": "gate:auditor"}, "PREFLIGHT: V3C_CHALLENGER (default PP_CHALLENGER) is local, loopback:HOST:PORT or "
                                         "gate:USER@HOST, not gate:auditor"),
    ({"PP_CHALLENGER": LIVE}, f"PREFLIGHT: V3c's audit's session on {LIVE} listens on its peer's 127.0.0.1, and V3C_PEER "
                              "(default PP_PEER), this node's name in the gate's /etc/vy/peers, is unset"),
    ({"PP_CHALLENGER": LIVE, "PP_PEER": "vy-dc-h200"}, f"PREFLIGHT: V3c's audit's session on {LIVE} is held to the "
                                                       "challenger key record PP_CHALLENGER_KEY, which is unset and not a "
                                                       "readable file"),
])
def test_the_preflight_refuses_v3c_on_a_development_session_outside_a_development_run_or_a_live_one_it_cannot_hold(
        tmp_path, extra, line):
    """V3c's audit is the sampled-proofs audit of record, so its coins are the live challenger's: a local or loopback
    session is refused unless the run is a development run (DEVELOPMENT=1, which is otherwise unset), and a live one with
    no peer for its session to listen on or no key record to hold it to; each before anything runs."""
    rc, lines, order, _limits = _v3c(tmp_path, **extra)
    assert (rc, lines, order) == (3, [line], [])


@pytest.mark.parametrize("challenger", [None, "loopback:127.0.0.1:7000"])
def test_without_v3c_the_preflight_refuses_step_3s_development_session_outside_a_development_run(tmp_path, challenger):
    """With V3c off step 3's audit is the sampled-proofs audit of record, whose sampled proofs the compose gate's Lean
    holds to a challenger, so its local or loopback session is refused unless the run is a development run, before
    anything runs."""
    rc, lines, order, _limits = _v3c(tmp_path, (), HOST="vy-dc-h200", DEVELOPMENT=None, PP_CHALLENGER=challenger,
                                     PATH=f"{_hosts(tmp_path)}:{tmp_path / 'bin'}:{os.defpath}", **NO_V3C)
    assert (rc, lines, order) == (3, [
        "PREFLIGHT: with V3c off (V3C_CLASS empty) step 3's audit is the sampled-proofs audit of record, and PP_CHALLENGER "
        f"is {challenger or 'local'}, a development session, which only a development run takes: set "
        "PP_CHALLENGER=gate:USER@HOST with PP_PEER, PP_CHALLENGER_KEY and PP_GATE_KEY, or DEVELOPMENT=1"], [])


@pytest.mark.parametrize("extra, other", [
    ({"H1_GATE": GATE, "H1_PEER": "vy-dc-h200-3"}, "H1_PEER"),
    ({"POUW": "pearlplus", "H1_PEER": "vy-dc-h200-3"}, "PEARLC_CHALLENGER_PEER"),
])
def test_the_preflight_refuses_a_v3c_peer_other_than_the_one_the_runs_other_live_session_names(tmp_path, extra, other):
    """On the served node V3c's audit's session and H1's (H1_PEER), or Pearl+'s (PEARLC_CHALLENGER_PEER, default
    H1_PEER), each listen on their peer's 127.0.0.1, the node the run serves on: a V3C_PEER naming another is refused
    before anything runs."""
    rc, lines, order, _limits = _v3c(tmp_path, (), **_live(tmp_path, [], hosts=True), HOST="vy-dc-h200", **extra)
    assert (rc, lines, order) == (3, [f"PREFLIGHT: V3C_PEER (default PP_PEER) is vy-dc-h200, and {other} is vy-dc-h200-3: "
                                      "each names the node the run serves on, where V3c's audit's session listens as the "
                                      "other one does"], [])


PEARLPLUS_RUN = "PoUW: Pearl+'s own served run on this card after the window's (POUW=pearlplus)"
FAKE_GATE = ('case "$1" in\n'
             f"ledger) echo '{LEDGER_LINE}' ;;\n"
             f"""pins) echo '{{"commit": "{"ef" * 20}", "main": "{"ef" * 20}", "pinned_utc": "2026-10-26T21:35:00Z"}}' ;;\n"""
             "*) exit 2 ;;\nesac")


@pytest.mark.parametrize("live, tie", [(True, False), (True, True), (False, False)])
def test_under_pouw_pearlplus_the_coins_line_says_whether_pearlpluss_epoch_coins_are_live(tmp_path, live, tie):
    """Under POUW=pearlplus the run checks the gate Pearl+'s serve reserves its session through, PEARLC_CHALLENGER_GATE
    (a lane with a period open would be refused), and the COINS line says Pearl+'s epoch's salt coin is a live session's
    of its own there, its piece crediting nothing, or under PEARLPLUS_TIE, where Pearl+ serves the window itself, taken as
    PoUW's verdict of record only where `pearlplus_of_record` holds it; with PEARLC_CHALLENGER_GATE empty no gate is
    checked, and the COINS line, the record's first, leads with the coins being in-process and the record claiming no
    Pearl+ verdict."""
    venv = tmp_path / "toy-venv" / "bin"
    _script(venv / "python", '[ "$1 $2" = "-m research" ] && exit 0\n[ "$1 $2" = "-m verity_vllm.pipeline.cli" ] && exit 9\n'
                             f'unset PYTHONPATH\nexport PYTHONSAFEPATH=1\nexec {sys.executable} "$@"')
    _script(gate := tmp_path / "gate", FAKE_GATE)
    (one := tmp_path / "tree" / "benchmarks" / "one_stage").mkdir(parents=True)
    (one / "served_verdicts.py").symlink_to(HERE.parent / "served_verdicts.py")
    env = {**(PEARL_TIED if tie else {}), "POUW": "pearlplus", "PEARLC_CHALLENGER_GATE": str(gate) if live else ""}
    rc, lines, _order, _limits = _v3c(tmp_path, (), VENV=str(venv.parent), HOST="vy-dc-h200",
                                      PATH=f"{_hosts(tmp_path)}:{tmp_path / 'bin'}:{os.defpath}", **env)
    assert rc == 17 and lines[-1] == "STOP at predict (exit 17)", lines
    tail = f";{V3C_COINS} the bandwidth audit's: vy-auditor-2's bw_party, inside PoUS's piece"
    if live:
        run, credit = PEARLPLUS_RUN, "it credits nothing"
        if tie:
            run = "PoUW: Pearl+ on the window's own engine, its passes the window's linears (POUW=pearlplus, PEARLPLUS_TIE)"
            credit = ("which the record takes as PoUW's verdict of record, the prover side's, only where "
                      "pearlplus_of_record holds it")
        assert lines[0] == (f"GATE ready: lane vllm's open period None; the ledger's 1 entries; pinned {'ef' * 6} (main "
                            f"{'ef' * 6}, 2026-10-26T21:35:00Z), on the verifier's first-parent line of main"), lines
        assert lines[1] == (f"COINS sampled proofs: {LOCAL_COINS}; {run}, its epoch's salt coin a live session's "
                            "of its own on vy-challenger-1's gate (peer vy-dc-h200, lane vllm), each pass's salt checked by "
                            "core_check.live_salt in its piece, its serve inside a Pearl-C period of that lane's own, and "
                            "its replay's draws a draw session's coin on that gate, reserved withheld once the period "
                            "closed and opened on the run's commitment, taken once before the replay, checked against the "
                            f"signed ledger in its piece and cited by store id in the POUW line's coin; a development piece, "
                            f"{credit}{tail}"), lines
    else:
        assert lines[0] == ("COINS Pearl+'s epoch coins are in-process and the record claims no Pearl+ verdict; sampled "
                            f"proofs: {LOCAL_COINS}; {PEARLPLUS_RUN}, its epoch's coins an in-process challenger's "
                            "(PEARLC_CHALLENGER_GATE empty), which the serving process holds, and its replay's draws the "
                            "Lean verifier's IO.getRandomBytes on this node, taken once before the replay and cited by "
                            f"store id in the POUW line's coin; a development piece, it credits nothing{tail}"), lines
        assert not any(x.startswith("GATE") for x in lines), lines


def test_the_script_parses_whole():
    """bash parses a top-level command only once a run reaches it, and no toy run here reaches the serve's later steps."""
    run = subprocess.run(["bash", "-n", str(HERE.parent / "served_verdicts.sh")], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr


def test_under_h1_tie_v3c_runs_no_audit_whose_ports_rs_sample_cannot_size(tmp_path):
    """A tied audit's ports default to the size of R's sample, so an R with no strata gets no audit, and V3c says why."""
    _tied_commit(tmp_path, "same", law=None)
    rc, lines, order, _limits = _v3c(tmp_path, H1_TIE="1")
    assert rc == 1 and "audit" not in order, (lines, order)
    assert "V3C-AUDIT: R, the window's registration, has no strata to size the audit's ports by: set V3C_PORTS" in lines


@pytest.mark.parametrize("bound, why", [
    (None, "H1 bound no record R at the window's close, so V3c has no commit of it (the serve dir's h1/w<k>/window-commit.log)"),
    ("other", "H1 bound a record R other than window-commit's registration"),
])
def test_under_h1_tie_a_window_h1_bound_no_r_for_or_another_r_has_no_v3c(tmp_path, bound, why):
    """No retries: a window whose tie failed isn't committed again for V3c, and a commit that isn't the record H1 bound
    isn't audited as if it were."""
    _tied_commit(tmp_path, bound)
    rc, lines, order, _limits = _v3c(tmp_path, H1_TIE="1")
    assert (rc, order, lines[-1]) == (16, ["window-circuit"], f"WINDOW-COMMIT class {W63K4}: {why}"), lines


#: Pearl+'s window record W of window 7 (`retain --window-record`'s), the record a Pearl+-tied commit is chained after
PEARL_WREC = {"format": "verity/registration/v2", "window": {"scheme": "pearl-c-h100-v1-h3"}, "previous": None,
           "registered": {"pearl-a/0/0": {}}}
PEARL_TIES = {"POUW": "pearlplus", "PEARLPLUS_TIE": "1", "PEARLC_CHALLENGER_GATE": ""}


def _pearl_tie(tmp_path: Path, **extra: str) -> tuple[subprocess.CompletedProcess, list[str]]:
    """`served_verdicts.sh pearl-tie SERVE_DIR 7 W PASSES` as Pearl+'s piece runs it (PEARLPLUS_TIE_CMD) once W is written,
    in a toy tree whose prove_path.py and advice note each step they run (``extra`` adds to the environment): the run and
    the steps in order."""
    tree, d, venv = tmp_path / "tree", tmp_path / "serve", tmp_path / "venv" / "bin"
    for p in (tree / "benchmarks" / "prove_path", tree / "adapters" / "vllm", tmp_path / "piece"):
        p.mkdir(parents=True)
    (tree / "benchmarks" / "prove_path" / "prove_path.py").write_text(FAKE_V3C_PROVE_PATH)
    (tree / "benchmarks" / "one_stage").symlink_to(HERE.parent, target_is_directory=True)
    (tmp_path / "advice.py").write_text(FAKE_ADVICE)
    _script(venv / "python", f'[ "$1 $2 $3" = "-m verity_vllm.pipeline.cli advice" ] && {{ shift 3; exec {sys.executable} '
                             f'{tmp_path}/advice.py "$@"; }}\nunset PYTHONPATH\nexport PYTHONSAFEPATH=1\nexec {sys.executable} "$@"')
    d.mkdir()
    (d / "weights_registration.json").write_text("{}")
    (tmp_path / "piece" / "window-record.json").write_text(json.dumps(PEARL_WREC))
    env = {"PATH": os.defpath, "HOME": str(tmp_path), "VENV": str(venv.parent), "D": str(d), "OUT": str(tmp_path / "tie-out"),
           "ORDER": str(tmp_path / "order"), "V3C_CLASS": W63K4, **PEARL_TIES, **extra}
    run = subprocess.run(["bash", "benchmarks/one_stage/served_verdicts.sh", "pearl-tie", str(d), "7",
                          str(tmp_path / "piece" / "window-record.json"), str(tmp_path / "piece" / "passes")],
                         cwd=tree, env=env, check=False, capture_output=True, text=True)
    return run, (tmp_path / "order").read_text().split() if (tmp_path / "order").exists() else []


def test_under_pearlplus_tie_the_pieces_tie_commits_the_window_cut_at_pearls_linears_chained_right_after_its_w(tmp_path):
    """Pearl+'s piece, once `retain --window-record` wrote W and before its coin, runs `pearl-tie`: the window's advice
    where step 3 reads it, W kept in the serve dir's pearlplus/w<k>/ (where V3c's audit reads it), and window-commit at
    V3C_CLASS on W with the window's passes (``--pearl-passes``, its circuit cut at Pearl+'s linears) and V3c's law, whose
    registration R it holds chained right after W. It says nothing in $OUT; its commit's answer stays in the serve dir."""
    run, order = _pearl_tie(tmp_path)
    d = tmp_path / "serve"
    w = REG.digest(PEARL_WREC)
    assert run.returncode == 0 and order == ["compiler", "advice", "window-commit"], (run.stdout, run.stderr, order)
    assert run.stdout.splitlines()[-1].startswith(
        f"PEARL-TIE: window-commit class {W63K4}, its circuit cut at Pearl+'s linears: W {w[:16]}…, R ")
    assert json.loads((d / "pearlplus" / "w7" / "registration.json").read_text()) == PEARL_WREC
    vw = d / f"v3c-w7-{W63K4.replace(',', '-')}" / "window"
    assert json.loads((vw / "registration.json").read_text())["previous"] == w
    assert (tmp_path / "order.commit").read_text().split() == [
        "--window-registration", str(d / "pearlplus" / "w7" / "registration.json"), "--pearl-passes",
        str(tmp_path / "piece" / "passes"), "--law", "stratified:1"]
    assert json.loads((d / "pearlplus" / "w7" / "window-commit.json").read_text())["population"] == 37
    assert not (tmp_path / "tie-out" / "verdicts.txt").exists()


@pytest.mark.parametrize("extra, rc, said", [
    ({"FAIL_COMMIT": "1"}, 16,
     f"PEARL-TIE: window-commit class {W63K4} refused (exit 2): prove_path.py window-commit: the row dir holds no window 7"),
    ({"BREAK_CHAIN": "1"}, 16, f"PEARL-TIE: window-commit's registration R is chained after {'ef' * 8}…, not right after "
                               f"Pearl+'s W {REG.digest(PEARL_WREC)[:16]}…"),
    ({"PEARLPLUS_TIE": ""}, 3, "PEARL-TIE: PEARLPLUS_TIE or V3C_CLASS is empty"),
])
def test_a_pearl_tie_whose_commit_is_refused_or_not_chained_right_after_w_fails(tmp_path, extra, rc, said):
    """A refused window-commit, or one whose registration isn't chained right after W, fails the tie, which the piece
    says before its coin; V3c then finds no commit tied to W and audits none."""
    run, _order = _pearl_tie(tmp_path, **extra)
    assert (run.returncode, run.stdout.splitlines()[-1]) == (rc, said), run.stderr


#: `served_compare.py` as the served compare runs it: its arguments and CUDA_VISIBLE_DEVICES in $ORDER.compare; at
#: FAKE_COMPARE_RC 0 or 1 a CUT line naming W, its verdicts in --out and its SERVED-COMPARE line, and above 1 it dies
#: before a verdict
FAKE_COMPARE = '''import json, os, sys
from pathlib import Path
rc = int(os.environ.get("FAKE_COMPARE_RC") or 0)
Path(os.environ["ORDER"] + ".compare").write_text(json.dumps({"argv": sys.argv[1:], "cuda": os.environ.get("CUDA_VISIBLE_DEVICES")}))
if rc > 1:
    sys.stderr.write("served_compare.py: died before a verdict\\n")
    sys.exit(rc)
at = dict(zip(sys.argv[1::2], sys.argv[2::2]))
print(f"CUT: W ({at['--window-record']}) registers 2 values: {'FAIL' if rc else 'PASS'}", flush=True)
line = f"SERVED-COMPARE: {'NOT EQUAL' if rc else 'EQUAL'} (the serve against row {at['--row']}'s Build of record)"
for name, doc in (("served_compare.json", {"ok": not rc, "line": line}), ("circuit_facts.json", {})):
    Path(at["--out"], name).write_text(json.dumps(doc))
print(line, flush=True)
sys.exit(rc)
'''
PEARL_ROW = "llama32-1b__bf16__h200__tp1__b4__i256__o32__mixed__greedy__bi-eager"


@pytest.mark.parametrize("case", ["equal", "unequal", "died", "untied"])
def test_the_served_compare_runs_on_the_cpu_against_the_rows_build_of_record_and_w(tmp_path, case):
    """`served_verdicts.sh served-compare K`, the run's step under PEARLPLUS_TIE once Pearl+'s piece wrote W and before the
    record, runs served_compare.py with no GPU on the serve dir, the row of record, V3C's class, Pearl+'s scheme, PRED and
    the row's Build of record, against W in pearlplus/w<k>/: its verdicts in the serve dir's served-compare/, copied to
    $OUT, its lines published (W's path naming no window) and its exit the step's. One that dies before a verdict says
    NOT EQUAL with its exit and error, and no earlier compare's verdict stands for it; without PEARLPLUS_TIE it isn't run."""
    tree, d, venv = tmp_path / "tree", tmp_path / "serve", tmp_path / "venv" / "bin"
    (tree / "benchmarks").mkdir(parents=True)
    (tree / "benchmarks" / "one_stage").symlink_to(HERE.parent, target_is_directory=True)
    (tmp_path / "compare.py").write_text(FAKE_COMPARE)
    _script(venv / "python", f'[ "$1" = benchmarks/one_stage/served_compare.py ] && {{ shift; exec {sys.executable} '
                             f'{tmp_path}/compare.py "$@"; }}\nexec {sys.executable} "$@"')
    (d / "served-compare").mkdir(parents=True)
    stale = d / "served-compare" / "served_compare.json"
    stale.write_text(json.dumps({"ok": True, "line": "SERVED-COMPARE: EQUAL (an earlier compare's)"}))
    rc = {"unequal": 1, "died": 2}.get(case, 0)
    env = {"PATH": os.defpath, "HOME": str(tmp_path), "VENV": str(venv.parent), "D": str(d), "OUT": str(tmp_path / "out"),
           "ORDER": str(tmp_path / "order"), "V3C_CLASS": W63K4, **PEARL_TIES, "FAKE_COMPARE_RC": str(rc),
           **({"PEARLPLUS_TIE": ""} if case == "untied" else {})}
    run = subprocess.run(["bash", "benchmarks/one_stage/served_verdicts.sh", "served-compare", "7"], cwd=tree, env=env,
                         check=False, capture_output=True, text=True)
    lines = run.stdout.splitlines()
    if case == "untied":
        assert (run.returncode, lines) == (
            3, ["SERVED-COMPARE: not run: PEARLPLUS_TIE is empty, so no W ties the serve to the row"]), run.stderr
        assert not (tmp_path / "order.compare").exists()
        return
    called = json.loads((tmp_path / "order.compare").read_text())
    at = dict(zip(called["argv"][::2], called["argv"][1::2], strict=True))
    assert re.fullmatch("[0-9a-f]{128}:[0-9a-f]{128}", at.pop("--of-record"))
    assert (called["cuda"], at) == ("", {
        "--serve": str(d), "--row": PEARL_ROW, "--class": W63K4, "--pearl-tables": "pearl-c-h100-v1-h3",
        "--pred": f"{d}/predicted", "--window-record": f"{d}/pearlplus/w7/registration.json", "--out": f"{d}/served-compare"})
    published = tmp_path / "out" / "served-compare" / "served_compare.json"
    if case == "died":
        assert (run.returncode, lines) == (2, ["SERVED-COMPARE: NOT EQUAL (served_compare.py exit 2: served_compare.py: died "
                                               "before a verdict)"]), run.stderr
        assert not stale.exists() and not published.exists()
        return
    said = f"SERVED-COMPARE: {'NOT EQUAL' if rc else 'EQUAL'} (the serve against row {PEARL_ROW}'s Build of record)"
    assert (run.returncode, lines) == (rc, [f"CUT: W ({d}/pearlplus/wK/registration.json) registers 2 values: "
                                            f"{'FAIL' if rc else 'PASS'}", said]), run.stderr
    assert json.loads(published.read_text()) == json.loads(stale.read_text()) == {"ok": not rc, "line": said}
    assert (tmp_path / "out" / "served-compare" / "circuit_facts.json").is_file()
    assert (d / "served-compare" / "compare.err").is_file()


def test_under_pearlplus_tie_the_window_serve_serves_pearl_c_on_its_one_engine(tmp_path):
    """Under PEARLPLUS_TIE serve-openai serves Pearl-C's arm on the window's own engine (`serve_pearl`), its
    --protocols-config the cubin and kernel PEARLPLUS-CUBIN built, at the arm's precision bfloat16 on Llama 3.2 1B's BF16
    checkpoint, one fixed window of the class's outputs, no H1. Each window's salt coin is a live salt-only session's
    (`serve_pearl.live`): the gate, peer and lane go to serve-openai, the lane's gate by default; with
    PEARLC_CHALLENGER_GATE empty none goes, and the arm's coins are in-process."""
    cubin = {"PEARLC_KERNEL": "pearl_c_sm90", "PEARLC_CUBIN": "/c/pearl_c_sm90.cubin"}
    rc, argv = _serve_argv(tmp_path, {**PEARL_TIES, **cubin})
    assert rc == 10 and argv[:3] == ["-m", "verity_vllm.pipeline.cli", "serve-openai"], (rc, argv)
    at = {f: argv[argv.index(f) + 1] for f in ("--protocols", "--protocols-config", "--model", "--revision", "--precision",
                                               "--window-steps")}
    assert at == {"--protocols": "pouw:pearl-c-h100-v1-h3", "--model": "unsloth/Llama-3.2-1B",
                  "--revision": "9535bd9b1d1dea6acafbdc4813b728796aeb28da", "--precision": "bfloat16", "--window-steps": "4",
                  "--protocols-config": json.dumps({"pouw": {"device": "cuda", "kernel": "pearl_c_sm90",
                                                             "cubin": "/c/pearl_c_sm90.cubin"}})}
    assert argv.count("--protocols") == 1 and "--h1-prover" not in argv and "--h1-window-commit" not in argv
    assert not {"--challenger-gate", "--challenger-peer", "--challenger-lane"} & set(argv)
    live = {"PEARLC_CHALLENGER_GATE": GATE, "PEARLC_CHALLENGER_PEER": "vy-dc-h200", "PEARLC_CHALLENGER_LANE": "compute-accounting"}
    _rc, gated = _serve_argv(tmp_path / "gated", {**PEARL_TIES, **cubin, **live})
    assert {f: gated[gated.index(f) + 1] for f in ("--challenger-gate", "--challenger-peer", "--challenger-lane")} == {
        "--challenger-gate": GATE, "--challenger-peer": "vy-dc-h200", "--challenger-lane": "compute-accounting"}
    assert gated.count("--protocols") == 1 and "--h1-prover" not in gated
    calls, d = (tmp_path / "calls").read_text().splitlines(), tmp_path / "serve" / "pearlplus"
    assert [c.split("prove_path/prove_path.py ", 1)[-1] for c in calls[-4:-2]] == [
        f"serve-phi --model unsloth/Llama-3.2-1B --revision 9535bd9b1d1dea6acafbdc4813b728796aeb28da --out {d}/serve-phi.json",
        f"class-circuit --serve {d}/serve-phi.json --class 4,32,4 --pearl-tables pearl-c-h100-v1-h3 --out "
        f"{d}/circuit-record.json"] and "inference_only.py policy" in calls[-2] and "serve-openai" in calls[-1], calls
    _rc, default = _serve_argv(tmp_path / "default", {k: v for k, v in {**PEARL_TIES, **cubin}.items()
                                                      if k != "PEARLC_CHALLENGER_GATE"})
    assert default[default.index("--challenger-gate") + 1] == (
        f"ssh -i {tmp_path / 'default'}/.ssh/vllm_lane_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes research@204.12.178.84")
    assert (default[default.index("--challenger-peer") + 1], default[default.index("--challenger-lane") + 1]) == (
        "vy-dc-h200", "vllm")


def test_under_pearlplus_tie_with_a_gate_the_window_serve_runs_inside_a_pearl_c_period_on_its_circuit_record(tmp_path):
    """With a gate, serve() opens the lane's Pearl-C period on PEARLPLUS-CIRCUIT's record (``pearlplus-period open
    --circuit-record``) once it is built and before serve-openai starts, so the window's salt session joins it, and
    closes it once serve-openai is done, here when it never listens: before Pearl+'s piece draws, its reserves first
    held to the windows' salt sessions (``--lane``, ``--windows`` the serve dir's pearl/), once the ledger ends them
    (``--wait`` PEARLPLUS_CLOSE_WAIT, default 300 s). Between the record and the
    open the inference-only policy is written, before serve-openai's start, from serve-phi's φ. In-process coins
    (PEARLC_CHALLENGER_GATE empty) open no period, and the policy is written all the same."""
    cubin = {"PEARLC_KERNEL": "pearl_c_sm90", "PEARLC_CUBIN": "/c/pearl_c_sm90.cubin"}
    live = {"PEARLC_CHALLENGER_GATE": GATE, "PEARLC_CHALLENGER_PEER": "vy-dc-h200", "PEARLC_CHALLENGER_LANE": "vllm",
            "RESEARCH_RUN_ID": "r20261104-121500-abcd"}
    rc, argv = _serve_argv(tmp_path, {**PEARL_TIES, **cubin, **live})
    assert rc == 10 and argv[:3] == ["-m", "verity_vllm.pipeline.cli", "serve-openai"], (rc, argv)
    d = tmp_path / "serve" / "pearlplus"

    def steps(calls: Path) -> list[str]:
        return [c.split("one_stage/served_verdicts.py ", 1)[-1] if "served_verdicts.py" in c
                else c.split("prove_path/prove_path.py ", 1)[-1].split()[0] if "prove_path.py" in c
                else "inference_only.py " + c.split()[1] if "inference_only.py" in c else c.split()[2]
                for c in calls.read_text().splitlines()]
    assert steps(tmp_path / "calls")[-6:] == [
        "serve-phi", "class-circuit", "inference_only.py policy",
        f"pearlplus-period open {d} --gate {GATE} --name pearlplus-r20261104-121500-abcd --circuit-record "
        f"{d}/circuit-record.json",
        "serve-openai", f"pearlplus-period close {d} --gate {GATE} --lane vllm --windows {d.parent}/pearl --wait 300"]
    _rc, _argv = _serve_argv(tmp_path / "in-process", {**PEARL_TIES, **cubin})
    assert steps(tmp_path / "in-process" / "calls")[-4:] == [
        "serve-phi", "class-circuit", "inference_only.py policy", "serve-openai"]


def test_under_pearlplus_tie_the_inference_only_policy_states_serve_phis_phi_cut_at_pearl_plus_linears(tmp_path):
    """Under PEARLPLUS_TIE the policy reads φ as serve-phi states it on the card (``--serve``, PEARLPLUS-CIRCUIT's
    pearlplus/serve-phi.json, which serve() writes under the lease before serve-openai starts) and cuts the class's
    circuit at Pearl+'s linears (``--pearl-tables``, PEARLPLUS_ARM's scheme): the BF16 checkpoint's φ, which an H200 int8
    serve's facts don't state."""
    rc, lines, calls = _inference_only(tmp_path, "inference-policy", **PEARL_TIES)
    assert rc == 0 and lines == ["INFERENCE-ONLY POLICY at class 4,32,4: c 7 gates, k 3 fresh bits, n 16 sessions"]
    assert calls == [["policy", "--class", "4,32,4", "--model", "unsloth/Llama-3.2-1B", "--revision",
                      "9535bd9b1d1dea6acafbdc4813b728796aeb28da", "--lease-min", "60", "--tree", SHA, "--serve",
                      str(tmp_path / "serve" / "pearlplus" / "serve-phi.json"), "--pearl-tables", "pearl-c-h100-v1-h3",
                      "--out", str(tmp_path / "out" / "inference-only-policy.json")]]


def _pearl_tied_commit(tmp_path: Path, chained: bool = True) -> None:
    """Window 7's commit as Pearl+'s piece left it: W in the serve dir's pearlplus/w7/, its answer beside it, and the
    registration R (chained right after W unless not ``chained``) and circuit in V3c's window dir."""
    d = tmp_path / "serve"
    vw = d / f"v3c-w7-{W63K4.replace(',', '-')}" / "window"
    vw.mkdir(parents=True)
    (d / "pearlplus" / "w7").mkdir(parents=True)
    (d / "pearlplus" / "w7" / "registration.json").write_text(json.dumps(PEARL_WREC))
    (d / "pearlplus" / "w7" / "window-commit.json").write_text(json.dumps({"run": "ab", "circuit": "cd", "population": 37}))
    r = {"format": "toy", "previous": REG.digest(PEARL_WREC) if chained else "ef" * 64, "law": TIED_LAW}
    (vw / "registration.json").write_text(json.dumps(r))
    (vw / "circuit.json").write_text(json.dumps({"run": "ab"}))


def test_under_pearlplus_tie_v3c_audits_the_commit_tied_to_pearls_w_with_records_audits_tie(tmp_path):
    """V3c makes no commit of its own: it holds the piece's R to W (chained right after it), and audits it tied to W with
    no prefix (Pearl+'s values are W's own names) at stratified:1 on a port a sampled block, saying the audit's tie. The
    verifier's own window-circuit record it holds the commit's circuit to is the circuit cut at Pearl+'s linears, and it
    leaves that record beside R in the tie the record reads, for the period's circuit."""
    _pearl_tied_commit(tmp_path)
    rc, lines, order, _limits = _v3c(tmp_path, **PEARL_TIES, CERTIFIER_CHAIN_CMD=None, CERTIFIER_PERIOD_CMD=None)
    assert rc == 0 and "window-commit" not in order and "audit" in order, (lines, order)
    circuit = (tmp_path / "order.circuit").read_text().split("\n")
    assert circuit[circuit.index("--pearl-tables") + 1] == "pearl-c-h100-v1-h3", circuit
    assert any(x.startswith(f"WINDOW-CIRCUIT class {W63K4}, cut at Pearl+'s linears: ") for x in lines), lines
    w = REG.digest(PEARL_WREC)
    assert lines[2].startswith(f"WINDOW-COMMIT class {W63K4} in Pearl+'s piece, its registration chained right after "
                               f"Pearl+'s window record W before Pearl+'s coin (W {w[:16]}…, R "), lines
    assert _audited(tmp_path) == {"--law": "stratified:1", "--tie": str(tmp_path / "serve" / "pearlplus" / "w7" / "registration.json"),
                                  "--tie-prefix": None, "--ports": "3"}
    assert f"V3C-TIE: the audit read its records tied to W {'cd' * 8}…, R chained right after it: PASS" in lines
    d, tag = tmp_path / "serve", W63K4.replace(",", "-")
    assert any(x.endswith("window-circuit record: PASS") for x in lines), lines
    assert (d / "pearlplus" / "w7" / "tie" / S.CIRCUIT_RECORD).read_bytes() == \
        (d / f"window-circuit-w7-{tag}.json").read_bytes(), "the record reads the period's circuit against V3c's record"


def test_under_pearlplus_tie_v3c_audits_no_commit_not_chained_right_after_w(tmp_path):
    """No retries: a commit whose R isn't chained right after Pearl+'s W isn't audited as if it were."""
    _pearl_tied_commit(tmp_path, chained=False)
    rc, lines, order, _limits = _v3c(tmp_path, **PEARL_TIES, CERTIFIER_CHAIN_CMD=None, CERTIFIER_PERIOD_CMD=None)
    assert (rc, order) == (16, ["window-circuit"]), lines
    assert lines[-1] == (f"WINDOW-COMMIT class {W63K4}: window-commit's registration R is chained after {'ef' * 8}…, not "
                         f"right after Pearl+'s W {REG.digest(PEARL_WREC)[:16]}…")


@pytest.mark.parametrize("extra, line", [
    ({"POUW": "h1"}, "PREFLIGHT: PEARLPLUS_TIE ties the window to Pearl+'s W, which POUW=pearlplus's piece writes: POUW is h1"),
    ({"H1_TIE": "1", "H1_GATE": ""}, "PREFLIGHT: PEARLPLUS_TIE or H1_TIE, not both: the window is tied to one W"),
    ({}, "PREFLIGHT: under PEARLPLUS_TIE V3c runs once Pearl+'s piece wrote W, after step 4, and the split certifier's "
         "period check needs it before: set CERTIFIER_CMD instead"),
    ({"CERTIFIER_CHAIN_CMD": None, "CERTIFIER_PERIOD_CMD": None, "LOAD": "varied"},
     "PREFLIGHT: under PEARLPLUS_TIE round s of Pearl+'s W is the window's s-th step, the class's s-th output, so the window "
     "is one full window of fixed length: LOAD=fixed (not varied) and WINDOW_STEPS the class's outputs, 4 (not 4)"),
    ({"CERTIFIER_CHAIN_CMD": None, "CERTIFIER_PERIOD_CMD": None, "WINDOW_STEPS": "16"},
     "PREFLIGHT: under PEARLPLUS_TIE round s of Pearl+'s W is the window's s-th step, the class's s-th output, so the window "
     "is one full window of fixed length: LOAD=fixed (not fixed) and WINDOW_STEPS the class's outputs, 4 (not 16)"),
    ({"CERTIFIER_CHAIN_CMD": None, "CERTIFIER_PERIOD_CMD": None, "PROMPTS": "16:24"},
     "PREFLIGHT: under PEARLPLUS_TIE the circuit cut at Pearl+'s linears reads its tables at the class's fixed load, its "
     "slots' requests of its prompt's ids each: SLOTS 8 (not 8) and PROMPTS 32:32 (not 16:24)"),
    ({"CERTIFIER_CHAIN_CMD": None, "CERTIFIER_PERIOD_CMD": None, "SLOTS": "4"},
     "PREFLIGHT: under PEARLPLUS_TIE the circuit cut at Pearl+'s linears reads its tables at the class's fixed load, its "
     "slots' requests of its prompt's ids each: SLOTS 8 (not 4) and PROMPTS 32:32 (not 32:32)"),
])
def test_the_preflight_refuses_pearlplus_tie_without_pearls_piece_beside_h1s_tie_or_the_split_or_off_one_fixed_window(
        tmp_path, extra, line):
    """PEARLPLUS_TIE ties the window to the W Pearl+'s piece writes, so it needs that piece (POUW=pearlplus) and not H1's
    tie, V3c after step 4 (not the split certifier's, before it), and one fixed window whose s-th step is the class's s-th
    output at the class's load: each refused before anything runs."""
    rc, lines, order, _limits = _v3c(tmp_path, (), **{**PEARL_TIES, **extra})
    assert (rc, lines, order) == (3, [line], [])


#: the toy tree's compiler of record for the BF16 1B, as prove_path.py lists it: its RoPE heads hold transfer gates
TOY_GATES = 'GATES_OF_RECORD: dict = {"unsloth/Llama-3.2-1B": ("RoPEHead_v1{D=64}",)}\n'
TG_PARTITION = {"templates": ["RoPEHead_v1{D=64}", "SiluMul_v2{I=8192}"], "law": "work:2",
                "scope": {"op_path_prefix": "model.layers.0.", "steps": [1, 2]}}
#: PEARLPLUS_TIE with V3c after step 4, the certifier unsplit, on the BF16 row
PEARL_TIED = {**PEARL_TIES, "CERTIFIER_CHAIN_CMD": None, "CERTIFIER_PERIOD_CMD": None,
              "ROW_ID": "llama32-1b__bf16__h200__tp1__b4__i256__o32__mixed__greedy__bi-eager"}


def _no_transfer_gates(part: str, law: str, templates: list[str]) -> str:
    return (f"PREFLIGHT: unsloth/Llama-3.2-1B's compiler of record lists ['RoPEHead_v1{{D=64}}'] as transfer gates "
            f"(prove_path.GATES_OF_RECORD), every unit of which step 3 proves beside work:K of the rest, and PP_PARTITION "
            f"{part} is {law} over {templates}: give a work:K partition that holds them, and in PP_SETS their input sets")


def _partition_preflight(tmp_path: Path, tie: bool, **extra: str) -> tuple[int, list[str], list[str], list[str]]:
    """`served_verdicts.sh` under PREFLIGHT_ONLY on vy-dc-h200-2, in `_v3c`'s toy tree whose prove_path.py lists the BF16
    1B's RoPE heads as its transfer gates, under PEARLPLUS_TIE if ``tie``."""
    return _v3c(tmp_path, (), prove_path=FAKE_V3C_PROVE_PATH + TOY_GATES,
                PATH=f"{_hosts(tmp_path)}:{tmp_path / 'bin'}:{os.defpath}", HOST="vy-dc-h200-2", PREFLIGHT_ONLY="1",
                **(PEARL_TIED if tie else {}), **extra)


@pytest.mark.parametrize("given, tie, refusal", [
    (None, True, None),
    ({"templates": ["SiluMul_v2{I=8192}"], "law": "subset:2"}, True, ("subset:2", ["SiluMul_v2{I=8192}"])),
    ({**TG_PARTITION, "law": "stratified:2"}, True, ("stratified:2", TG_PARTITION["templates"])),
    ({**TG_PARTITION, "templates": ["SiluMul_v2{I=8192}"]}, True, ("work:2", ["SiluMul_v2{I=8192}"])),
    ({"templates": ["SiluMul_v2{I=8192}"], "law": "subset:2"}, False, None),
])
def test_the_preflight_refuses_a_step_3_partition_that_leaves_out_the_served_compilers_transfer_gates(
        tmp_path, given, tie, refusal):
    """Under PEARLPLUS_TIE the served BF16 1B's compiler of record lists its RoPE heads as transfer gates, every unit of
    which step 3 proves beside the drawn rest (`serving_rows.work_table_of`), so step 3's partition defaults to a work:2
    one holding them, and the preflight refuses another law or a partition without them, before anything runs. The int8
    checkpoint's compiler lists none, so its subset:2 partition passes."""
    (pp := tmp_path / "tree" / "benchmarks" / "prove_path").mkdir(parents=True)
    (pp / "llama32_1b_tg_l0_step1.json").write_text(json.dumps(TG_PARTITION))
    extra = {}
    if given is not None:
        (part := tmp_path / "partition.json").write_text(json.dumps(given))
        extra["PP_PARTITION"] = str(part)
    rc, lines, order, _limits = _partition_preflight(tmp_path, tie, **extra)
    if refusal is None:
        assert (rc, order) == (0, []) and lines[0].startswith("PREFLIGHT passed"), lines
    else:
        assert (rc, lines, order) == (3, [_no_transfer_gates(str(tmp_path / "partition.json"), *refusal)], [])


def test_the_preflight_refuses_a_step_3_partition_it_cant_read_where_the_compiler_lists_transfer_gates(tmp_path):
    """A partition file that isn't there, where the served compiler lists transfer gates, is refused, not passed."""
    rc, lines, order, _limits = _partition_preflight(tmp_path, True)
    assert (rc, order) == (3, []) and len(lines) == 1, lines
    assert lines[0].startswith("PREFLIGHT: step 3's partition PP_PARTITION benchmarks/prove_path/llama32_1b_tg_l0_step1.json "
                               "can't be read: FileNotFoundError"), lines


@pytest.mark.parametrize("extra, line", [
    ({"PRECISION": "verity_int8"}, "PREFLIGHT: PEARLPLUS_TIE serves the BF16 checkpoint's linears as Pearl+'s FP8 product, "
                                   "and PRECISION is verity_int8: set it bfloat16"),
    ({"ROW_ID": None}, "PREFLIGHT: PEARLPLUS_TIE serves the BF16 checkpoint, and ROW_ID row is no BF16 row's (its id names "
                       "__bf16__): set ROW to the BF16 row's dir"),
    ({"CLASSES": "4,32,4"}, f"PREFLIGHT: PEARLPLUS_TIE commits the window at V3C_CLASS {W63K4}, whose window circuit "
                            "CLASSES (4,32,4) doesn't count: add it"),
    ({"SERVE_ARGS": "--prefix-caching"}, "PREFLIGHT: under PEARLPLUS_TIE each request's prefill computes every row of its "
     "prompt, at which the cut circuit reads Pearl+'s tables, and SERVE_ARGS turns vLLM's prefix cache on "
     "(--prefix-caching): drop it"),
    ({"CERTIFIER_FRONT": "true"}, "PREFLIGHT: under PEARLPLUS_TIE the window is V3C_CLASS's fixed load, and "
     "CERTIFIER_FRONT's group closes into a window of its own, which the run's pick takes on a tie in requests: unset "
     "CERTIFIER_FRONT"),
])
def test_the_preflight_refuses_pearlplus_tie_off_the_bf16_row_its_class_uncounted_with_the_prefix_cache_or_a_front(
        tmp_path, extra, line):
    """PEARLPLUS_TIE serves the BF16 checkpoint, whose linears Pearl+'s FP8 product reads, and V3c's commit is held to
    V3C_CLASS's window circuit; the cut circuit reads Pearl+'s tables at every row of the class's fixed load. So another
    PRECISION or row, a V3C_CLASS CLASSES doesn't count, vLLM's prefix cache (a hit leaves a prompt's rows uncomputed) and
    a certifier front's window of its own are each refused before anything runs."""
    rc, lines, order, _limits = _v3c(tmp_path, (), **{**PEARL_TIED, **extra})
    assert (rc, lines, order) == (3, [line], [])


def test_the_preflight_refuses_pearlplus_tie_under_a_live_gate_without_main_file(tmp_path):
    """Under PEARLPLUS_TIE with a live Pearl+ gate the piece holds the gate's pin to main's first-parent line, which
    MAIN_SHA's (by default the tree's own HEAD) need not hold: without MAIN_FILE the run is refused before anything runs,
    naming where the line comes from."""
    rc, lines, order, _limits = _v3c(tmp_path, (), **{**PEARL_TIED, "PEARLC_CHALLENGER_GATE": GATE, "MAIN_FILE": None})
    assert (rc, lines, order) == (3, [
        "PREFLIGHT: under PEARLPLUS_TIE the gate's pin is held to main's first-parent line, which MAIN_SHA's (default this "
        "tree's HEAD) need not hold: set MAIN_FILE to `git rev-list --first-parent origin/main` of a fresh fetch, sent "
        "with the run"], [])


#: `df` on a toy filesystem: its arguments in $ORDER.df, the line under its header DF_LINE
FAKE_DF = 'echo "$@" > "$ORDER.df"\necho "Inodes IFree Mounted on"\necho "$DF_LINE"'
SHORT_INODES = ("PREFLIGHT: the serve dir's filesystem /workspace has 100 inodes free, 772288 short of PEARLPLUS_INODES "
                "(772388), two commits of the tied window at three files a member: delete earlier runs' Pearl+ windows "
                "with `research retention rm`, or set SERVE_ROOT on another filesystem")


@pytest.mark.parametrize("env, df, refusal", [
    (PEARL_TIED, "2000000 100 /workspace", SHORT_INODES),
    (PEARL_TIED, "- - /workspace", "PREFLIGHT: the serve dir's free inodes can't be held to PEARLPLUS_INODES (772388): df "
                                   "says - free of - on {at}"),
    (PEARL_TIED, "2000000 772388 /workspace", None),
    (PEARL_TIED, "0 0 /workspace", None),
    ({"POUW": "pearlplus", "PEARLC_CHALLENGER_GATE": ""}, "2000000 100 /workspace", None),
    ({}, "2000000 100 /workspace", None),
])
def test_the_preflight_refuses_a_pearlplus_tie_where_the_serve_dirs_filesystem_has_fewer_inodes_free_than_two_commits(
        tmp_path, env, df, refusal):
    """Under PEARLPLUS_TIE Pearl+'s piece commits the window on the served node, three files a member, into the serve dir,
    so the preflight refuses to serve where df counts fewer than PEARLPLUS_INODES free on its filesystem (by default two
    commits' at the class of record) or counts none, naming the filesystem and the shortfall, before anything runs. It
    asks df about the serve dir's nearest existing directory, since the serve dir is made after the preflight. A
    filesystem with no inode table makes its inodes as it needs them, and an untied serve commits no tied window."""
    _script(tmp_path / "bin" / "df", FAKE_DF)
    rc, lines, order, _limits = _v3c(tmp_path, (), PATH=f"{_hosts(tmp_path)}:{tmp_path / 'bin'}:{os.defpath}",
                                     HOST="vy-dc-h200-2", PREFLIGHT_ONLY="1", D=str(tmp_path / "root" / "r1"),
                                     DF_LINE=df, PEARLPLUS_INODES=None, **env)
    asked = (tmp_path / "order.df").read_text().split() if (tmp_path / "order.df").exists() else None
    assert asked == (["--output=itotal,iavail,target", str(tmp_path)] if "PEARLPLUS_TIE" in env else None)
    if refusal is None:
        assert (rc, order) == (0, []) and len(lines) == 1 and lines[0].startswith("PREFLIGHT passed"), lines
    else:
        assert (rc, lines, order) == (3, [refusal.format(at=tmp_path)], [])


TIES = {"H1_TIE": "1", "H1_GATE": GATE, "H1_CIRCUITS": "circuits"}


@pytest.mark.parametrize("extra, prove_path, line", [
    ({"H1_TIE": "1"}, FAKE_V3C_PROVE_PATH,
     "PREFLIGHT: H1_TIE ties H1's windows, and H1_GATE or H1_CIRCUITS (whose prover state window-commit reads) is unset"),
    ({**TIES, "V3C_CLASS": "", "CERTIFIER_CHAIN_CMD": None, "CERTIFIER_PERIOD_CMD": None}, FAKE_V3C_PROVE_PATH,
     "PREFLIGHT: H1_TIE commits each H1 window at V3C_CLASS, and it is empty"),
    (TIES, "import sys\n", "PREFLIGHT: H1_TIE needs window-commit's --window-registration, its registration chained right "
                           "after H1's, which this tree's prove_path.py doesn't take"),
    ({**TIES, "V3C_TIMEOUT": "120m"}, FAKE_V3C_PROVE_PATH, "PREFLIGHT: under H1_TIE the window's sample coin waits for "
     "window-commit, and its session's reserve lasts the gate's 120 minutes from the serve's start: V3C_TIMEOUT is at most "
     "90m, not 120m"),
    ({**TIES, "V3C_TIMEOUT": "1h"}, FAKE_V3C_PROVE_PATH, "PREFLIGHT: under H1_TIE the window's sample coin waits for "
     "window-commit, and its session's reserve lasts the gate's 120 minutes from the serve's start: V3C_TIMEOUT is at most "
     "90m, not 1h"),
    ({**TIES, "NODE_MEM_GB": "196"}, FAKE_V3C_PROVE_PATH, "PREFLIGHT: under H1_TIE, where this node has 196 GB, under the "
     "290 GB V3c's audit needs (V3C_AUDIT_MEM_GB), its audit and the composite are deferred to a follow-on, and onsite's "
     "composite reads PoUS's case, which needs this node's card: PRODUCT=remote, or a node with the memory"),
    ({**TIES, "V3C_AUDIT_MEM_GB": "600"}, FAKE_V3C_PROVE_PATH, "PREFLIGHT: under H1_TIE, where this node has 512 GB, under "
     "the 600 GB V3c's audit needs (V3C_AUDIT_MEM_GB), its audit and the composite are deferred to a follow-on, and onsite's "
     "composite reads PoUS's case, which needs this node's card: PRODUCT=remote, or a node with the memory"),
])
def test_the_preflight_refuses_h1_tie_without_h1s_prover_or_v3c_or_window_commits_tie_or_past_the_sessions_time(
        tmp_path, extra, prove_path, line):
    """H1_TIE reads H1's prover state and commits at V3C_CLASS with window-commit's --window-registration, the flag that
    chains its registration after H1's W (a tree without it is refused, not run with an untied commit), and its commit runs
    while the window's sample session's reserve, the gate's 120 minutes, runs out; on a node with less memory than V3c's
    audit needs (V3C_AUDIT_MEM_GB) the audit and the composite go to a follow-on, which onsite's PoUS case can't: each
    refused before anything runs."""
    rc, lines, order, _limits = _v3c(tmp_path, (), prove_path=prove_path, **extra)
    assert (rc, lines, order) == (3, [line], [])


NO_TIE = ("PREFLIGHT: under H1_TIE V3c's audit is tied to H1's W, whose tables the circuit reads, with records_audit's "
          "--tie, which this tree's records_audit.py doesn't take")


def _no_reader(law: str) -> str:
    return ("PREFLIGHT: under H1_TIE every table H1's W registers needs a sampled reader, which only stratified:K, a "
            f"stratum of each table's readers, gives: V3C_LAW is {law}")


@pytest.mark.parametrize("args, extra, records_audit, line", [
    ((), TIES, "import sys\n", NO_TIE),
    (("v3c",), {"H1_TIE": "1"}, "import sys\n", NO_TIE),
    ((), {**TIES, "V3C_LAW": "subset:1"}, FAKE_RECORDS_AUDIT, _no_reader("subset:1")),
    (("v3c",), {"H1_TIE": "1", "V3C_LAW": "work:4"}, FAKE_RECORDS_AUDIT, _no_reader("work:4")),
])
def test_the_preflight_refuses_h1_tie_on_a_tree_whose_audit_takes_no_tie_or_at_a_design_other_than_stratified(
        tmp_path, args, extra, records_audit, line):
    """Under H1_TIE the audit reads the tables H1's W registered, so a tree whose records_audit.py can't tie it to W is
    refused, not run untied, and so is a sampling design that can leave a table with no sampled reader, which Record.checks
    refuses: in the serve's run and in `served_verdicts.sh v3c` alike, before anything runs."""
    rc, lines, order, _limits = _v3c(tmp_path, args, records_audit=records_audit, **extra)
    assert (rc, lines, order) == (3, [line], [])


VTAG = W63K4.replace(",", "-")
SHA = "5a" * 20
SHORT = "this node has 196 GB, under the 290 GB V3c's audit needs (V3C_AUDIT_MEM_GB)"
#: what a deferral's follow-on reads in `_deferred`'s serve dir, but for V3c's window dir and the deferral's own files
DEFERRED_READS = ("weights_registration.json", f"window-circuit-w7-{VTAG}-tie.json", "windows.jsonl", "seals.jsonl",
                  "requests.jsonl", "h1/log.jsonl", "h1/w7/registration.json", "h1/w7/record.json",
                  "h1/w7/window-commit.json", "sampled-proofs-served/summary.json", "sampled-proofs-served/window/audit.json",
                  "sampled-proofs-served/window/registration.json", "sampled-proofs-served/runs.json",
                  "private/pouw/period/case.json", "private/pouw/period/answer.json")
FAKE_RECORD = '''import json, os, sys
a = sys.argv[1:]
open(os.environ["ORDER"], "a").write(a[0] + "\\n")
open(os.environ["ORDER"] + ".record", "w").write(json.dumps(a))
print("RECORD " + json.dumps({"composed": "pending", "window": 7}))
'''
FAKE_SERVED_COMPOSITE = '''echo composite >> "$ORDER"
printf "%s\\n" "$SERVED_NODE" "$@" > "$ORDER.composite"
echo "COMPOSITE: composed on $(hostname -s)" >> "$2/verdicts.txt"'''


def _hosts(tmp_path: Path) -> Path:
    """A directory, for PATH, whose `hostname` says $HOST."""
    _script(tmp_path / "hosts" / "hostname", 'echo "$HOST"')
    return tmp_path / "hosts"


def _deferred(tmp_path: Path, **extra: str | None) -> subprocess.CompletedProcess:
    """Window 7 of a toy serve dir under H1's tie: its commit as the tie left it, with a value file, H1's records, the
    record's and the composite's inputs, files no step after the serve reads (a row dir, a key, an audit session), and
    the pieces the record read in $OUT; then step 6's deferral (`served_verdicts.sh defer WHY`) on vy-dc-h200, a node short
    of V3c's audit's memory, at the tree of commit SHA, in a development run (``extra`` adds to the environment, None
    unsetting)."""
    d, out = tmp_path / "serve", tmp_path / "served-out"
    _tied_commit(tmp_path, "same")
    vw = d / f"v3c-w7-{VTAG}" / "window"
    files = {"h1/w7/registration.json": W_OF_7, "h1/w7/record.json": json.loads((vw / "registration.json").read_text()),
             f"window-circuit-w7-{VTAG}.json": {}, f"window-circuit-w7-{VTAG}-tie.json": {"tables": "h1"},
             "weights_registration.json": {},
             "seals.jsonl": {"window": 7, "requests": ["a"]},
             "requests.jsonl": {"user": "a", "engine_id": "a", "windows": [7], "check": {"passed": True}},
             "sampled-proofs-served/summary.json": {"audit": {}},
             "sampled-proofs-served/window/audit.json": {"accepted": True},
             "sampled-proofs-served/window/registration.json": {"format": "toy"},
             "sampled-proofs-served/runs.json": ["r0"], "private/pouw/period/case.json": {"evidence": "beef"},
             "private/pouw/period/answer.json": {"protocol": "verity/pouw/period/v1"},
             "windows/w7/rows.bin": {}, "private/weights.key": {}, "sampled-proofs-served/window/sessions/0.json": {}}
    for f, body in files.items():
        (d / f).parent.mkdir(parents=True, exist_ok=True)
        (d / f).write_text(json.dumps(body))
    _write(d, "windows.jsonl", [{"window": 7, "run_root": "ab" * 64, "passed": True, "commit": {"steps": 3}}])
    (vw / "values").mkdir()
    (vw / "values" / "a.bin").write_bytes(b"\x01")
    (out / "pieces" / "pouw").mkdir(parents=True)
    (out / "pieces" / "pouw" / "answer.json").write_text(json.dumps({"verdict": "accept"}))
    (tmp_path / "served-tree").mkdir()
    env = {"PATH": f"{_hosts(tmp_path)}:{os.defpath}", "HOME": str(tmp_path), "VENV": str(tmp_path / "served-venv"),
           "D": str(d), "K": "7", "OUT": str(out), "SERVE_ROOT": str(tmp_path / "root"), "V3C_CLASS": W63K4,
           "PRODUCT": "remote", "H1_TIE": "1", "HOST": "vy-dc-h200", "RESEARCH_SOURCE_SHA": SHA,
           "GIT_CEILING_DIRECTORIES": str(tmp_path), "DEVELOPMENT": "1"}
    env = {k: v for k, v in {**env, **extra}.items() if v is not None}
    return subprocess.run(["bash", str(HERE.parent / "served_verdicts.sh"), "defer", SHORT], cwd=tmp_path / "served-tree",
                          env=env, check=False, capture_output=True, text=True)


def _deferral_settings(tmp_path: Path) -> dict[str, str]:
    """The settings `_deferred`'s deferral recorded for its follow-on (private/deferred/deferred.env), as bash reads them."""
    names = ("K PRODUCT V3C_CLASS V3C_LAW V3C_CHALLENGER DEVELOPMENT H1_TIE COMPOSITE ALPHA GAMMA SERVED_NODE SERVED_PEER "
             "DEFERRED_TREE COMPOSE_OFF_NODE COMPOSE_NODE DAY")
    said = subprocess.run(["bash", "-c", f'. "$1"; for v in {names}; do printf "%s=%s\\n" "$v" "${{!v}}"; done', "_",
                           str(tmp_path / "serve" / "private" / "deferred" / "deferred.env")],
                          check=True, capture_output=True, text=True).stdout
    return dict(x.split("=", 1) for x in said.splitlines())


DEFERRED_LINE = {
    "1": (f"DEFERRED V3c's audit: {SHORT}. A follow-on on a node with the memory runs it there and writes the compose "
          "hand-off, from the serve dir's audit inputs copied prover-side, outside any run dir, each file its "
          "private/deferred/inputs.txt lists: D={d} bash benchmarks/one_stage/served_verdicts.sh v3c; the final record "
          "and the composite are composed off both nodes, on COMPOSE_NODE (vy-auditor-2), from that hand-off"),
    "0": (f"DEFERRED V3c's audit and the composite: {SHORT}. A follow-on on a node with the memory runs them and composes "
          "the final record there, from the serve dir's audit inputs copied prover-side, outside any run dir, each file "
          "its private/deferred/inputs.txt lists: D={d} bash benchmarks/one_stage/served_verdicts.sh v3c")}


@pytest.mark.parametrize("off", ["1", "0"])
def test_a_node_short_of_v3cs_audits_memory_defers_it_and_the_composite_listing_and_hashing_each_input(tmp_path, off):
    """Step 6's deferral leaves in the serve dir's private/deferred the pieces the record read, the settings the
    follow-on runs at (the window, the product, V3c's class, law and coins, H1_TIE, the composite, the served node, its
    peer, the tree's commit and where the final record is composed), and the follow-on's every input, listed for the
    prover's copy and hashed for the follow-on to hold it to: V3c's window dir whole, H1's log and each record it names,
    the tie's commit answer, the record's and the composite's inputs and its own files, and nothing else of the serve dir.
    Its one line in $OUT names what was deferred, why and the follow-on's command (under COMPOSE_OFF_NODE=1, the default,
    V3c's audit alone, the record and the composite composed off both nodes), and no window; nothing else lands in $OUT."""
    run = _deferred(tmp_path, **({} if off == "1" else {"COMPOSE_OFF_NODE": off}))
    assert run.returncode == 0, run.stderr
    d, dd, out = tmp_path / "serve", tmp_path / "serve" / "private" / "deferred", tmp_path / "served-out"
    assert _deferral_settings(tmp_path) == {
        "K": "7", "PRODUCT": "remote", "V3C_CLASS": W63K4, "V3C_LAW": "stratified:1", "V3C_CHALLENGER": "local",
        "DEVELOPMENT": "1", "H1_TIE": "1", "COMPOSITE": "1", "ALPHA": "", "GAMMA": "", "SERVED_NODE": "vy-dc-h200",
        "SERVED_PEER": "", "DEFERRED_TREE": SHA, "COMPOSE_OFF_NODE": off, "COMPOSE_NODE": "vy-auditor-2", "DAY": ""}
    listed = (dd / "inputs.txt").read_text().splitlines()
    window = [f"v3c-w7-{VTAG}/window/{f}" for f in ("circuit.json", "registration.json", "values/a.bin")]
    own = ["private/deferred/deferred.env", "private/deferred/pieces/pouw/answer.json"]
    assert listed == [*sorted([*DEFERRED_READS, *window, *own]), "private/deferred/inputs.sha256"]
    sums = {f: h for h, f in (x.split("  ", 1) for x in (dd / "inputs.sha256").read_text().splitlines())}
    assert sums == {f: hashlib.sha256((d / f).read_bytes()).hexdigest() for f in listed[:-1]}
    assert (dd / "pieces" / "pouw" / "answer.json").read_text() == (out / "pieces" / "pouw" / "answer.json").read_text()
    assert (out / "verdicts.txt").read_text().splitlines() == [DEFERRED_LINE[off].replace("{d}", str(d))]
    assert sorted(f.relative_to(out).as_posix() for f in out.rglob("*") if f.is_file()) == [
        "pieces/pouw/answer.json", "verdicts.txt"]
    assert not _window_data(out, 7) and oct(dd.stat().st_mode & 0o777) == "0o700"


def _follow(tmp_path: Path, **extra: str | None) -> tuple[int, list[str], list[str], list[str]]:
    """The follow-on of `_deferred`'s deferral: `served_verdicts.sh v3c` with D alone (no K, class or certifier) on
    cpu-6, a node with V3c's audit's memory, at the served node's tree, in `_v3c`'s toy tree with a record and a
    composite that note their arguments (``extra`` adds to the environment, None unsetting): `_v3c`'s outcome."""
    _script(tmp_path / "tree" / "benchmarks" / "one_stage" / "served_composite.sh", FAKE_SERVED_COMPOSITE)
    (tmp_path / "tree" / "benchmarks" / "one_stage" / "served_verdicts.py").write_text(FAKE_RECORD)
    env = {"K": None, "V3C_CLASS": None, "CERTIFIER_CHAIN_CMD": None, "CERTIFIER_PERIOD_CMD": None, "COMPOSITE": None,
           "PATH": f"{_hosts(tmp_path)}:{tmp_path / 'bin'}:{os.defpath}", "HOST": "cpu-6", "NODE_MEM_GB": "755",
           "RESEARCH_SOURCE_SHA": SHA, "GIT_CEILING_DIRECTORIES": str(tmp_path)}
    return _v3c(tmp_path, **{**env, **extra})


@pytest.mark.parametrize("extra, tree", [
    ({}, "(this tree's)"),
    ({"RESEARCH_SOURCE_SHA": "cd" * 20, "FOLLOW_ANY_TREE": "1"}, f"(this tree is {'cd' * 20}: FOLLOW_ANY_TREE)"),
])
def test_the_follow_on_holds_the_copy_to_the_served_nodes_sums_then_audits_and_composes_the_final_record_there(
        tmp_path, extra, tree):
    """The follow-on of a deferral made under COMPOSE_OFF_NODE=0, at the settings the deferral recorded: the copy held to
    the served node's sums, then no window circuit or commit of its own (the served node's record at the class, its
    tree's, and the commit H1 bound, held as there), V3c's audit and period, the final record about the served node from
    the pieces it read and V3c's audit, the sampled-proofs audit of record, and the composite, told the served node, so
    each says it composed elsewhere. Another tree runs only under FOLLOW_ANY_TREE, which its line names. $OUT names no
    window."""
    assert _deferred(tmp_path, COMPOSE_OFF_NODE="0").returncode == 0
    rc, lines, order, limits = _follow(tmp_path, **extra)
    d, out = tmp_path / "serve", tmp_path / "out"
    assert rc == 0 and order == ["audit", "period", "record", "composite"], (lines, order)
    assert lines.pop(0) == V3C_ENTRY
    assert lines[:2] == [
        "FOLLOW-ON on cpu-6, 755 GB: V3c's audit and the composite, which the served node vy-dc-h200 deferred, from its "
        "serve dir's audit inputs copied here prover-side, each as it hashed them (the serve dir's "
        "private/deferred/inputs.sha256): PASS",
        f"WINDOW-CIRCUIT class {W63K4}: the served node's record at the class (step 3's), from its tree's commit {SHA} "
        f"{tree}; this node holds no row dir to count it from"]
    assert lines[2].startswith(f"WINDOW-COMMIT class {W63K4} at H1's close") and lines[3].endswith("PASS")
    assert [x.split(" ")[0] for x in lines[4:]] == ["V3C-AUDIT", "V3C-PERIOD", "RECORD", "COMPOSITE:"]
    assert lines[-1] == "COMPOSITE: composed on cpu-6" and limits == ["180m"]
    assert _audited(tmp_path) == {"--law": "stratified:1", "--tie": str(d / "h1" / "w7" / "registration.json"),
                                  "--tie-prefix": "h1/", "--ports": "3"}
    argv = json.loads((tmp_path / "order.record").read_text())
    assert argv[:-4] == [
        "record", "--serve-run", str(d), "--window", "7", "--sampled-proofs", str(d / f"v3c-w7-{VTAG}" / "audit"),
        "--pieces", str(d / "private" / "deferred" / "pieces"), "--out", str(out / "record"), "--product", "remote",
        "--node", "vy-dc-h200"]
    assert (argv[-4], len(argv[-3]), argv[-2:]) == ("--key-sha512", 128, ["--lane", "vllm"])
    assert (tmp_path / "order.composite").read_text().split() == ["vy-dc-h200", str(d), str(out)]
    assert not _window_data(out, 7), _window_data(out, 7)


FOLLOW_FAIL = ("FOLLOW-ON on cpu-6, 755 GB: V3c's audit, which the served node vy-dc-h200 deferred, "
               "from its serve dir's audit inputs copied here prover-side, each as it hashed them (the serve dir's "
               "private/deferred/inputs.sha256): FAIL (the serve dir's private/deferred/check.log)")
CLASH = ("PREFLIGHT: the follow-on runs at the deferral's settings (the serve dir's private/deferred/deferred.env), and "
         "the environment sets another")


@pytest.mark.parametrize("change, extra, said", [
    ("values/a.bin", {}, [V3C_ENTRY, FOLLOW_FAIL]),
    ("case.json", {}, [V3C_ENTRY, FOLLOW_FAIL]),
    (None, {"K": "8"}, [f"{CLASH} K: unset it"]),
    (None, {"PRODUCT": "onsite", "COMPOSITE": ""}, [f"{CLASH} PRODUCT COMPOSITE: unset it"]),
    (None, {"NODE_MEM_GB": "196"}, [f"PREFLIGHT: the follow-on runs the V3c audit the served node deferred, and {SHORT}"]),
    (None, {"RESEARCH_SOURCE_SHA": "cd" * 20}, ["PREFLIGHT: the follow-on reads the served node's window-circuit record, "
     f"which its tree {SHA} wrote, and this tree is {'cd' * 20}: run it at that commit, or set FOLLOW_ANY_TREE=1"]),
])
def test_a_follow_on_refuses_a_copy_unlike_the_served_nodes_another_setting_a_node_short_of_the_memory_or_another_tree(
        tmp_path, change, extra, said):
    """No retries, and no follow-on unlike the deferral: a copied value file changed or a listed input missing fails the
    copy's check before anything runs, and a setting other than the deferral's, a node short of the audit's memory (the
    served node itself, after its deferral) or another tree is refused at the preflight."""
    assert _deferred(tmp_path).returncode == 0
    d = tmp_path / "serve"
    if change == "values/a.bin":
        (d / f"v3c-w7-{VTAG}" / "window" / "values" / "a.bin").write_bytes(b"\x02")
    elif change == "case.json":
        (d / "private" / "pouw" / "period" / "case.json").unlink()
    rc, lines, order, _limits = _follow(tmp_path, **extra)
    assert (rc, lines, order) == (3, said, []), lines


def _day_log(path: Path, program: str) -> RL.Declaration:
    """A remote deployment's registration log at ``path`` declaring day 3's circuit ``program``: the declaration."""
    log = RL.chain("remote", [RL.Declaration(3, program)])
    path.write_bytes(RL.committed("remote", log))
    return log[-1]


def _day_deferred(tmp_path: Path, **extra: str | None) -> RL.Declaration:
    """`_deferred`'s deferral under DAY=3: the serve dir's declaration.json as serve-openai's --day wrote it, its log
    DAY_LOG and its parameters DAY_PARAMS outside the serve dir (``extra`` adds to the environment): the declaration."""
    d, log, params = tmp_path / "serve", tmp_path / "day-log.json", tmp_path / "day-params.json"
    entry = _day_log(log, "5d" * 64)
    params.write_text(json.dumps(RL.params("remote", SITE.Auditor())))
    d.mkdir()
    (d / "declaration.json").write_text(json.dumps({"day": 3, "program": "5d" * 64, "blocks": [],
                                                    "declaration": _named(entry), "head": entry.digest(),
                                                    "log": str(log), "declared": "now", "entry": entry.to_json(),
                                                    "file": "declaration.json"}))
    run = _deferred(tmp_path, DAY="3", DAY_LOG=str(log), DAY_PARAMS=str(params), **extra)
    assert run.returncode == 0, run.stderr
    return entry


def _named(entry: RL.Declaration) -> str:
    """The digest serve-openai's declaration.json names ``entry`` by: its Declaration with no ``previous``."""
    return RL.Declaration(entry.day, entry.program, entry.blocks).digest()


def _held(entry: RL.Declaration, verdict: str) -> str:
    return ("DAY-HELD: the serve dir's private/deferred/day-log.json, the day's registration log as the deferral copied "
            f"it, holds the declaration the serve's declaration.json names: {verdict}")


@pytest.mark.parametrize("extra", [{}, {"DAY_LOG": "/elsewhere.json", "DAY_PARAMS": "/elsewhere-params.json"}],
                         ids=["copy", "this-nodes-ignored"])
def test_a_deferral_under_day_carries_the_days_log_and_its_follow_on_audits_r2_against_that_copy(tmp_path, extra):
    """The onsite run of record defers V3c's audit, so DAY rides the deferral: DAY in its settings, the day's log and
    parameters copied into private/deferred and hashed with the serve's declaration.json, and the follow-on, once the
    copy holds to the served node's sums and to the declaration declaration.json names by digest, hands records_audit
    the copy (--day-log, --day-params), never a log of its own node's environment."""
    entry = _day_deferred(tmp_path, COMPOSE_OFF_NODE="0")
    d, dd = tmp_path / "serve", tmp_path / "serve" / "private" / "deferred"
    assert _deferral_settings(tmp_path)["DAY"] == "3"
    assert (dd / "day-log.json").read_bytes() == (tmp_path / "day-log.json").read_bytes()
    listed = (dd / "inputs.txt").read_text().splitlines()
    assert {"declaration.json", "private/deferred/day-log.json", "private/deferred/day-params.json"} <= set(listed)
    sums = {f: h for h, f in (x.split("  ", 1) for x in (dd / "inputs.sha256").read_text().splitlines())}
    assert sums["private/deferred/day-log.json"] == hashlib.sha256((d / "private/deferred/day-log.json").read_bytes()).hexdigest()
    rc, lines, order, _limits = _follow(tmp_path, records_audit=_day_tree(tmp_path), **extra)
    assert rc == 0 and order == ["audit", "period", "record", "composite"], (lines, order)
    assert lines[1].endswith("(the serve dir's private/deferred/inputs.sha256): PASS")
    assert lines[2] == _held(entry, f"PASS (day 3, program {'5d' * 8}…, declaration {_named(entry)[:16]}…)")
    assert _audited(tmp_path, DAYS) == {"--day": "3", "--day-log": str(dd / "day-log.json"),
                                        "--day-params": str(dd / "day-params.json")}


@pytest.mark.parametrize("change", ["edited", "rehashed", "clash", "old-tree"])
def test_a_follow_on_under_day_refuses_a_day_log_unlike_the_one_the_serve_declared_in(tmp_path, change):
    """The follow-on audits no day the serve didn't declare: an edited copy of the day's log fails the served node's
    sums before anything runs; one rehashed to pass them, declaring the day another program, fails DAY-HELD, the
    declaration's digest the serve wrote; another DAY is a clash with the deferral's settings; and a tree whose
    records_audit takes no --day is refused at the preflight."""
    entry = _day_deferred(tmp_path)
    dd = tmp_path / "serve" / "private" / "deferred"
    extra: dict[str, str | None] = {}
    tree = _day_tree(tmp_path)
    if change == "edited":
        (dd / "day-log.json").write_bytes((dd / "day-log.json").read_bytes() + b"\n")
        said = [V3C_ENTRY, FOLLOW_FAIL]
    elif change == "rehashed":
        other = _day_log(dd / "day-log.json", "6e" * 64)
        sums = (dd / "inputs.sha256").read_text().splitlines()
        h = hashlib.sha256((dd / "day-log.json").read_bytes()).hexdigest()
        (dd / "inputs.sha256").write_text("".join(
            f"{h}  {f}\n" if f == "private/deferred/day-log.json" else f"{x}\n"
            for x, f in ((x, x.split("  ", 1)[1]) for x in sums)))
        assert _named(other) != _named(entry)
        said = [V3C_ENTRY, FOLLOW_FAIL.replace("FAIL (the serve dir's private/deferred/check.log)", "PASS"),
                _held(entry, f"FAIL (it holds no declaration of day 3 with digest {_named(entry)[:16]}…)")]
    elif change == "clash":
        extra = {"DAY": "4"}
        said = [f"{CLASH} DAY: unset it"]
    else:
        tree = FAKE_RECORDS_AUDIT
        said = ["PREFLIGHT: the deferral declared day 3, whose audit runs R2 with records_audit's --day, which this "
                "tree's records_audit.py doesn't take"]
    rc, lines, order, _limits = _follow(tmp_path, records_audit=tree, **extra)
    assert (rc, lines, order) == (3, said, []), lines


def _live_follow(tmp_path: Path, **extra: str | None) -> tuple[int, list[str], list[str], list[str]]:
    """`_deferred`'s deferral in a run of record whose V3c audit takes its coins from the live challenger LIVE from the
    served node's peer vy-dc-h200, then its follow-on on cpu-6 under `_live`'s gate with V3C_PEER cpu-6 (``extra`` adds
    to the environment, None unsetting): `_v3c`'s outcome."""
    assert _deferred(tmp_path, DEVELOPMENT=None, PP_CHALLENGER=LIVE, PP_PEER="vy-dc-h200").returncode == 0
    env = {**_live(tmp_path, [LEDGER_LINE], hosts=True), "PP_CHALLENGER": None, "PP_PEER": None, "V3C_PEER": "cpu-6"}
    return _follow(tmp_path, **{**env, **extra})


def test_a_follow_on_of_a_deferral_of_record_samples_on_the_challenger_it_recorded_from_its_own_node(tmp_path):
    """A deferral in a run of record records V3c's audit's coins (V3C_CHALLENGER, DEVELOPMENT unset) and the served
    node's peer, so its follow-on takes its coins from that challenger, whatever its own environment's PP_CHALLENGER,
    from its own node (V3C_PEER, default PP_PEER), as its COINS line says, once the gate holds no session on the window's
    run root."""
    rc, lines, order, _limits = _live_follow(tmp_path)
    assert {k: v for k, v in _deferral_settings(tmp_path).items() if k in ("V3C_CHALLENGER", "DEVELOPMENT", "SERVED_PEER")} == {
        "V3C_CHALLENGER": LIVE, "DEVELOPMENT": "", "SERVED_PEER": "vy-dc-h200"}
    assert rc == 0 and order == ["gate-ledger", "gate-pins", "audit", "period"], (lines, order)
    assert lines[-1].startswith("HANDOFF"), lines
    assert lines[0] == (f"COINS V3c's audit: the challenger {LIVE} (peer cpu-6), the window's run root's one session, the "
                        "sampled-proofs audit of record")
    assert sum(x.startswith("V3C-GATE ready: lane vllm's open period None; root ") for x in lines) == 1, lines
    assert _audited(tmp_path, COIN_FLAGS) == {
        "--challenger": LIVE, "--peer": "cpu-6", "--challenger-key": str(tmp_path / "challenger-key.json"),
        "--gate-key": str(tmp_path / "gate_ed25519")}


@pytest.mark.parametrize("extra, line", [
    ({"V3C_PEER": "vy-dc-h200"}, "PREFLIGHT: V3C_PEER (default PP_PEER) is vy-dc-h200, the served node's (SERVED_PEER), "
     "which deferred the audit: its session would listen on that node's 127.0.0.1, not this one's; name this node as the "
     "gate's /etc/vy/peers does"),
    ({"V3C_PEER": None}, f"PREFLIGHT: V3c's audit's session on {LIVE} listens on its peer's 127.0.0.1, and V3C_PEER "
     "(default PP_PEER), this node's name in the gate's /etc/vy/peers, is unset"),
    ({"DEVELOPMENT": "1"}, f"{CLASH} DEVELOPMENT: unset it"),
    ({"V3C_CHALLENGER": "local"}, f"{CLASH} V3C_CHALLENGER: unset it"),
])
def test_a_follow_on_of_a_deferral_of_record_refuses_the_served_nodes_peer_no_peer_or_other_coins(tmp_path, extra, line):
    """The follow-on's session listens on its own node, never the served node's, and its coins are the deferral's: a
    development run or another challenger is a setting other than the deferral's. Each is refused before anything runs."""
    rc, lines, order, _limits = _live_follow(tmp_path, **extra)
    assert (rc, lines, order) == (3, [line], []), lines


ONLY = "PREFLIGHT_ONLY: nothing kept, built or served"
DEFERS = "so step 6 defers V3c's audit to a follow-on"
OFF = "the final record composed off this node, on COMPOSE_NODE (vy-auditor-2), from its compose hand-off"
SHORT_REMOTE = {**TIES, "NODE_MEM_GB": "196", "PRODUCT": "remote", "CERTIFIER_CHAIN_CMD": None, "CERTIFIER_PERIOD_CMD": None}


@pytest.mark.parametrize("extra, said", [
    ({}, f"512 GB: the serve's lease 60 min, capped at 128 GB; {OFF}"),
    ({"COMPOSE_OFF_NODE": "0"}, "512 GB: the serve's lease 60 min, capped at 128 GB"),
    (TIES, f"512 GB: the serve's lease 150 min, capped at 160 GB; {OFF}"),
    ({**TIES, "LEASE_MEM_GB": "64"}, f"512 GB: the serve's lease 150 min, capped at 64 GB; {OFF}"),
    (SHORT_REMOTE, f"196 GB: the serve's lease 150 min, capped at 160 GB; {SHORT}, {DEFERS}; {OFF}"),
    ({**SHORT_REMOTE, "COMPOSE_OFF_NODE": "0", "COMPOSITE": None},
     f"196 GB: the serve's lease 150 min, capped at 160 GB; {SHORT}, so step 6 defers V3c's audit and the composite to a "
     "follow-on"),
    ({"COMPOSITE": None, "PRODUCT": "remote", "CERTIFIER_CHAIN_CMD": None, "CERTIFIER_PERIOD_CMD": None},
     f"512 GB: the serve's lease 60 min, capped at 128 GB; {OFF.replace('record', 'record and the composite')}"),
])
def test_preflight_only_says_what_the_run_would_do_and_stops_before_anything_is_kept_built_or_served(
        tmp_path, extra, said):
    """PREFLIGHT_ONLY=1 runs the preflight and stops there, exit 0, with one line: the serve's lease (under H1_TIE 150
    minutes capped at 160 GB, the serve's 128 and window-commit's peak, within a 196 GB node), on a node short of V3c's
    audit's memory its deferral, and under COMPOSE_OFF_NODE=1, the default, what is composed off the node. Nothing runs,
    and neither the serve root nor C-Flock's work dir is made."""
    rc, lines, order, limits = _v3c(tmp_path, (), PATH=f"{_hosts(tmp_path)}:{tmp_path / 'bin'}:{os.defpath}",
                                    HOST="vy-dc-h200-2", PREFLIGHT_ONLY="1", **extra)
    assert (rc, lines, order, limits) == (0, [f"PREFLIGHT passed on vy-dc-h200-2, {said}. {ONLY}"], [], [])
    assert not (tmp_path / "root").exists() and not (tmp_path / "fw").exists()


def test_preflight_only_on_a_follow_on_names_the_deferral_it_would_finish(tmp_path):
    """A follow-on under PREFLIGHT_ONLY passes its own preflight (the deferral's settings, the audit's memory, the served
    node's tree) and stops before its copy's check or V3c's audit."""
    assert _deferred(tmp_path).returncode == 0
    rc, lines, order, _limits = _follow(tmp_path, PREFLIGHT_ONLY="1")
    assert (rc, lines, order) == (0, ["PREFLIGHT passed on cpu-6, 755 GB: the follow-on of vy-dc-h200's deferral, its "
                                      f"compose hand-off for COMPOSE_NODE (vy-auditor-2). {ONLY}"], [])


def test_a_deferral_carries_the_served_nodes_inference_only_verdict_into_the_follow_ons_record(tmp_path):
    """The inference-only verdict the served node judged goes with the deferral, listed and hashed with the follow-on's
    inputs, and into the follow-on's record as the served node wrote it, said before the record: the follow-on judges it
    no second time (under COMPOSE_OFF_NODE=0; under 1 the hand-off carries it)."""
    verdict = {"format": "verity-bench/inference-only-verdict/v3", "verdict": "accept", "classes": [["ab", 7, 3]]}
    (tmp_path / "served-out" / "record").mkdir(parents=True)
    (tmp_path / "served-out" / "record" / "inference-only.json").write_text(json.dumps(verdict))
    assert _deferred(tmp_path, COMPOSE_OFF_NODE="0").returncode == 0
    dd = tmp_path / "serve" / "private" / "deferred"
    assert "private/deferred/inference-only.json" in (dd / "inputs.txt").read_text().splitlines()
    assert json.loads((dd / "inference-only.json").read_text()) == verdict
    rc, lines, order, _limits = _follow(tmp_path)
    assert rc == 0 and order == ["audit", "period", "record", "composite"], (lines, order)
    assert lines.pop(0) == V3C_ENTRY
    assert [x.split(" ")[0] for x in lines[4:]] == ["V3C-AUDIT", "V3C-PERIOD", "INFERENCE-ONLY:", "RECORD", "COMPOSITE:"]
    assert json.loads((tmp_path / "out" / "record" / "inference-only.json").read_text()) == verdict


#: a5's v2 case's blobs in `_handed`'s serve dir: its circuit, public file, lookup table, and a session's record and proof
BLOBS = tuple(f"sampled-proofs-served/window/m0/{f}" for f in (
    "circuit.txt", "pub-0.bin", "tables/t0.bin", "session-m0/sessions/s0/session.json", "session-m0/sessions/s0/p0.bin"))
#: what `_handed`'s serve dir holds that no hand-off may carry: the a5 member's prover files beside its blobs, a value
#: file, V3c's index salts, H1's prover state, a key, a row dir, the advice and the requests
HELD_BACK = ("sampled-proofs-served/window/m0/inst-0.bin", "sampled-proofs-served/window/m0/prover/rows.bin",
             f"v3c-w7-{VTAG}/window/values/a.bin", f"v3c-w7-{VTAG}/window/index-salt.bin", "private/h1/key",
             "private/weights.key", "windows/w7/rows.bin",
             "advice/w7/private/advice.bin", "requests.jsonl")
#: what the composite and the final record read in `_handed`'s serve dir, but for the blobs and the hand-off's own files
HANDED_READS = ("h1/log.jsonl", "h1/w7/record.json", "h1/w7/registration.json", "private/pouw/period/answer.json",
                "private/pouw/period/case.json", "sampled-proofs-served/runs.json", "sampled-proofs-served/summary.json",
                "sampled-proofs-served/window/audit.json", "sampled-proofs-served/window/case-v2.json",
                "sampled-proofs-served/window/case.json", "sampled-proofs-served/window/registration.json",
                f"v3c-w7-{VTAG}/audit/window/case.json")
HANDED_OWN = ("private/compose/compose.env", "private/compose/inference-only.json",
              "private/compose/pieces/pouw/answer.json", "private/compose/pieces/pouw/served.json")
HANDOFF_LINE = ("HANDOFF the final record and the composite, composed off this node: on COMPOSE_NODE (vy-auditor-2), "
                "from the serve dir's verifier-side files and no witness, each file its private/compose/inputs.txt lists, "
                "copied to the same path there: D={d} bash benchmarks/one_stage/served_verdicts.sh compose")


def _env_of(f: Path, names: str) -> dict[str, str]:
    said = subprocess.run(["bash", "-c", f'. "$1"; for v in {names}; do printf "%s=%s\\n" "$v" "${{!v}}"; done', "_",
                           str(f)], check=True, capture_output=True, text=True).stdout
    return dict(x.split("=", 1) for x in said.splitlines())


def _handed(tmp_path: Path, blobs: tuple[str, ...] = BLOBS, verdict: bool = True, files: Mapping[str, Any] | None = None,
            cmd: Sequence[str] = ("handoff",), pouw: Mapping[str, Any] | None = None, piece: Path | None = None,
            **extra: str | None) -> subprocess.CompletedProcess:
    """Window 7 of a toy serve dir as the served node leaves it at its record under H1's tie: its records audits' cases
    (a5's v2 case naming ``blobs``, relative to the serve dir unless absolute), H1's log and records, PoUW's period case,
    the window's seal, Commit line and request, every file of `HELD_BACK`, and in $OUT the record's pieces and (if
    ``verdict``) inference-only verdict; then `served_verdicts.sh handoff` (or ``cmd``) on vy-dc-h200 at the tree of
    commit SHA (``files`` replaces a file's body, by its path, None leaving it unwritten; ``pouw`` PoUW's answer, or
    ``piece`` PoUW's piece dir, copied whole; ``extra`` adds to the environment, None unsetting)."""
    d, out = tmp_path / "serve", tmp_path / "served-out"
    sp = "sampled-proofs-served"
    served = {"seals.jsonl": {"window": 7, "requests": ["a"]},
             "windows.jsonl": {"window": 7, "run_root": "ab" * 64, "passed": True, "commit": {"steps": 3}},
             "requests.jsonl": {"user": "a", "engine_id": "a", "windows": [7], "check": {"passed": True}},
             "h1/log.jsonl": {"window": 7, "kind": "registration", "file": "w7/registration.json"},
             "h1/w7/registration.json": W_OF_7, "h1/w7/record.json": {"format": "toy"}, "h1/w7/window-commit.json": {},
             f"{sp}/summary.json": {"audit": {}}, f"{sp}/window/audit.json": {"accepted": True},
             f"{sp}/window/registration.json": {"format": "toy"}, f"{sp}/runs.json": ["r0"],
             f"{sp}/window/case.json": {"evidence": "beef"},
             f"{sp}/window/case-v2.json": {"blobs": [b if b.startswith("/") else str(d / b) for b in blobs]},
             f"v3c-w7-{VTAG}/audit/window/case.json": {"evidence": "cafe"},
             "private/pouw/period/case.json": {"evidence": "beef"},
             "private/pouw/period/answer.json": {"protocol": "verity/pouw/period/v1"},
             **{f: {} for f in (*BLOBS, *HELD_BACK) if f != "requests.jsonl"}}
    files = {**served, **(files or {})}
    for f, body in files.items():
        if body is None:
            continue
        (d / f).parent.mkdir(parents=True, exist_ok=True)
        (d / f).write_text(json.dumps(body) + "\n")
    with open(d / "h1" / "log.jsonl", "a") as log:
        log.write(json.dumps({"window": 7, "kind": "record", "file": "w7/record.json"}) + "\n")
    if piece is not None:
        shutil.copytree(piece, out / "pieces" / "pouw")
    else:
        (out / "pieces" / "pouw").mkdir(parents=True)
        for f, body in (("answer.json", pouw or {"verdict": "accept"}), ("served.json", {"server": "vy-dc-h200"}),
                        ("piece.log", "")):
            (out / "pieces" / "pouw" / f).write_text(json.dumps(body))
    (out / "record").mkdir()
    if verdict:
        (out / "record" / "inference-only.json").write_text(json.dumps({"verdict": "accept"}))
    (tmp_path / "served-tree").mkdir(exist_ok=True)
    env = {"PATH": f"{_hosts(tmp_path)}:{os.defpath}", "HOME": str(tmp_path), "VENV": str(tmp_path / "served-venv"),
           "D": str(d), "K": "7", "OUT": str(out), "SERVE_ROOT": str(tmp_path / "root"), "V3C_CLASS": W63K4,
           "PRODUCT": "remote", "H1_TIE": "1", "HOST": "vy-dc-h200", "RESEARCH_SOURCE_SHA": SHA,
           "GIT_CEILING_DIRECTORIES": str(tmp_path)}
    env = {k: v for k, v in {**env, **extra}.items() if v is not None}
    return subprocess.run(["bash", str(HERE.parent / "served_verdicts.sh"), *cmd], cwd=tmp_path / "served-tree",
                          env=env, check=False, capture_output=True, text=True)


COMPOSES = "K PRODUCT COMPOSITE ALPHA GAMMA SERVED_NODE SERVED_DIR COMPOSE_NODE COMPOSE_TREE AUDITED_ON SAMPLED_AUDIT STEPS"


def test_the_served_nodes_compose_hand_off_lists_and_hashes_the_verifier_side_files_alone(tmp_path):
    """The compose hand-off (COMPOSE_OFF_NODE=1, the default) leaves in the serve dir's private/compose the files the
    final record reads of each piece (no log), the inference-only verdict, the settings its follow-on composes at (the
    window and its engine steps, picked here, the product, the composite, the served node, COMPOSE_NODE and the tree's
    commit), and its every input, listed for the copy and hashed for the follow-on: the records audits' cases and the
    blobs a5's v2 case names, H1's log and the records it names, PoUW's period case and sampled proofs' summary and
    registration. It lists no witness: no value, instance or prover file, H1's prover state, key, row dir, advice or
    request, and nothing `served_verdicts.py witness` refuses. Its one line names the follow-on's command and no window."""
    run = _handed(tmp_path)
    assert run.returncode == 0, run.stderr
    d, dc, out = tmp_path / "serve", tmp_path / "serve" / "private" / "compose", tmp_path / "served-out"
    assert _env_of(dc / "compose.env", COMPOSES) == {
        "K": "7", "PRODUCT": "remote", "COMPOSITE": "1", "ALPHA": "", "GAMMA": "", "SERVED_NODE": "vy-dc-h200",
        "SERVED_DIR": str(d), "COMPOSE_NODE": "vy-auditor-2", "COMPOSE_TREE": SHA, "AUDITED_ON": "",
        "SAMPLED_AUDIT": "sampled-proofs-served", "STEPS": "0,3"}
    listed = (dc / "inputs.txt").read_text().splitlines()
    assert listed == [*sorted([*HANDED_READS, *BLOBS, *HANDED_OWN]), "private/compose/inputs.sha256"]
    assert not set(listed) & set(HELD_BACK) and S.witness(listed) == []
    sums = {f: h for h, f in (x.split("  ", 1) for x in (dc / "inputs.sha256").read_text().splitlines())}
    assert sums == {f: hashlib.sha256((d / f).read_bytes()).hexdigest() for f in listed[:-1]}
    assert sorted(p.name for p in (dc / "pieces" / "pouw").iterdir()) == ["answer.json", "served.json"]
    assert (out / "verdicts.txt").read_text().splitlines() == [HANDOFF_LINE.replace("{d}", str(d))]
    assert not _window_data(out, 7) and oct(dc.stat().st_mode & 0o777) == "0o700"


def test_a_hand_off_of_a_record_with_no_inference_only_verdict_lists_the_rest(tmp_path):
    """A record with no inference-only verdict still hands off: its list is every other file, and nothing is refused."""
    run = _handed(tmp_path, verdict=False)
    assert run.returncode == 0, run.stderr
    listed = (tmp_path / "serve" / "private" / "compose" / "inputs.txt").read_text().splitlines()
    own = [f for f in HANDED_OWN if f != "private/compose/inference-only.json"]
    assert listed == [*sorted([*HANDED_READS, *BLOBS, *own]), "private/compose/inputs.sha256"]


def test_a_hand_off_after_v3cs_audit_gives_the_record_that_audit_and_carries_its_files_not_step_3s(tmp_path):
    """Where V3c audited the window before the hand-off (its summary there), the hand-off names V3c's audit, the
    sampled-proofs audit of record, as the record's (SAMPLED_AUDIT) and carries its summary, audit record, registration
    and runs; of step 3's, on local coins while V3c's are live, only the cases the composite reads. The compose's record
    then reads V3c's audit, its Lean check on that audit's coins."""
    v3c = f"v3c-w7-{VTAG}/audit"
    files = {f"{v3c}/summary.json": {"tie": {"record": "d7" * 64}}, f"{v3c}/window/audit.json": {"accepted": True},
             f"{v3c}/window/registration.json": {"format": "toy"}, f"{v3c}/runs.json": ["r1"]}
    assert _handed(tmp_path, files=files).returncode == 0
    d, dc = tmp_path / "serve", tmp_path / "serve" / "private" / "compose"
    assert _env_of(dc / "compose.env", "SAMPLED_AUDIT") == {"SAMPLED_AUDIT": v3c}
    step3 = {f"sampled-proofs-served/{f}" for f in ("summary.json", "window/audit.json", "window/registration.json",
                                                    "runs.json")}
    listed = (dc / "inputs.txt").read_text().splitlines()
    assert listed == [*sorted([*(set(HANDED_READS) - step3), *files, *BLOBS, *HANDED_OWN]),
                      "private/compose/inputs.sha256"]
    rc, lines, order = _compose(tmp_path)
    assert rc == 0 and order == ["record", "composite"], (lines, order)
    assert lines[0] == COMPOSED.format(audit="no V3c audit", sampled=f"v3c-wK-{VTAG}/audit", verdict="PASS")
    argv = json.loads((tmp_path / "order.record").read_text())
    assert argv[argv.index("--sampled-proofs") + 1] == str(d / v3c) and "--sampled-lean" in argv


@pytest.mark.parametrize("blob, said", [
    ("sampled-proofs-served/window/m0/inst-0.bin", "it would carry the served run's witness (WITNESS "
     "sampled-proofs-served/window/m0/inst-0.bin: a prover's instance file (its rows))"),
    ("sampled-proofs-served/window/m0/prover/rows.bin", "it would carry the served run's witness (WITNESS "
     "sampled-proofs-served/window/m0/prover/rows.bin: a prover's input file (its rows and salts))"),
    ("/elsewhere/pub-0.bin", "{case} names a blob outside the serve dir: /elsewhere/pub-0.bin"),
])
def test_a_hand_off_that_would_carry_a_witness_or_a_blob_outside_the_serve_dir_is_refused(tmp_path, blob, said):
    """No witness leaves the served node: a case naming a prover's file as a blob, or a blob outside the serve dir, refuses
    the hand-off, which writes no sums for a copy to be held to."""
    run = _handed(tmp_path, (*BLOBS, blob))
    case = tmp_path / "serve" / "sampled-proofs-served" / "window" / "case-v2.json"
    assert run.returncode == 1
    assert (tmp_path / "served-out" / "verdicts.txt").read_text().splitlines() == [
        "HANDOFF refused: " + said.replace("{case}", str(case))]
    assert not (tmp_path / "serve" / "private" / "compose" / "inputs.sha256").exists()


def test_witness_names_each_path_a_served_run_holds_as_its_witness_and_why(tmp_path, capsys, monkeypatch):
    """`served_verdicts.py witness` refuses each of `HELD_BACK` (by a directory part, the serve dir's top directory, H1's
    prover state or a name) and passes the verifier-side files a hand-off carries, exiting 1 iff it named one."""
    found = dict(S.witness([*HELD_BACK, *HANDED_READS, *BLOBS, *HANDED_OWN, "private/h1/w0/state.json"]))
    assert sorted(found) == sorted([*HELD_BACK, "private/h1/w0/state.json"])
    assert found["private/h1/key"] == "H1's prover state (its key)" and found["windows/w7/rows.bin"].startswith("a sealed row")
    monkeypatch.setattr(sys, "stdin", io.StringIO("".join(f"{x}\n" for x in (*HANDED_READS, *BLOBS))))
    assert S.main(["witness"]) == 0 and capsys.readouterr().out == ""
    monkeypatch.setattr(sys, "stdin", io.StringIO("h1/log.jsonl\nrequests.jsonl\n"))
    assert S.main(["witness"]) == 1
    assert capsys.readouterr().out == "WITNESS requests.jsonl: the requests' prompt and served ids, rows and seeds\n"


@pytest.mark.parametrize("extra, line", [
    ({"COMPOSITE": None}, "PREFLIGHT: COMPOSE_OFF_NODE=1 composes the composite off this node from its hand-off, and "
     "onsite's reads PoUS's and the wipe's cases, which only this node's card writes: PRODUCT=remote, COMPOSITE set empty, "
     "or COMPOSE_OFF_NODE=0 for a development composite here"),
    ({"COMPOSITE": None, "COMPOSE_OFF_NODE": "0"}, None),
    ({}, None),
    ({"COMPOSE_OFF_NODE": "yes"}, "PREFLIGHT: COMPOSE_OFF_NODE is 1 (the default) or 0, not yes"),
])
def test_the_preflight_refuses_an_onsite_composite_off_the_node_whose_cases_only_its_card_writes(tmp_path, extra, line):
    """Onsite's composite reads PoUS's and the wipe's cases, which `served_case.sh` writes under the served card's lease,
    so they can't be composed off the node: under COMPOSE_OFF_NODE=1 the preflight refuses onsite with a composite, before
    anything runs, and passes it with none or under COMPOSE_OFF_NODE=0."""
    rc, lines, order, _limits = _v3c(tmp_path, (), PATH=f"{_hosts(tmp_path)}:{tmp_path / 'bin'}:{os.defpath}",
                                     HOST="vy-dc-h200-2", PREFLIGHT_ONLY="1", **extra)
    assert (rc, order) == ((3, []) if line else (0, [])) and (lines == [line] if line else len(lines) == 1), lines
    assert line or lines[0].startswith("PREFLIGHT passed")


FAKE_COMPOSE_COMPOSITE = '''echo composite >> "$ORDER"
printf "%s\\n" "$SERVED_NODE" "$COMPOSE_HANDOFF" "$@" > "$ORDER.composite"
echo "COMPOSITE: composed on $(hostname -s)" >> "$2/verdicts.txt"'''


def _compose(tmp_path: Path, keep: tuple[str, ...] = (), **extra: str | None) -> tuple[int, list[str], list[str]]:
    """The compose follow-on of the hand-off in `tmp_path`'s serve dir: the files its inputs.txt lists (and ``keep``)
    copied to the same path on vy-auditor-2, the rest of the serve dir moved away, then `served_verdicts.sh compose` with
    D alone in a toy tree whose record and composite note their arguments (``extra`` adds to the environment, None
    unsetting): the exit status, the verdict lines and the steps in order."""
    d, held = tmp_path / "serve", tmp_path / "served-node"
    listed = (d / "private" / "compose" / "inputs.txt").read_text().splitlines()
    d.rename(held)
    for f in (*listed, *keep):
        (d / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(held / f, d / f)
    tree, bin_, venv = tmp_path / "compose-tree", tmp_path / "compose-bin", tmp_path / "compose-venv" / "bin"
    py = f'unset PYTHONPATH\nexport PYTHONSAFEPATH=1\n[ "$1 $2" = "-m research" ] && exit 0\nexec {sys.executable} "$@"'
    for path, body in ((venv / "python", py), (tree / ".venv" / "bin" / "python", py), (bin_ / "uv", "exit 0"),
                       (bin_ / "lake", "exit 0"),
                       (tree / "benchmarks" / "one_stage" / "served_composite.sh", FAKE_COMPOSE_COMPOSITE)):
        _script(path, body)
    (tree / "benchmarks" / "one_stage" / "served_verdicts.py").write_text(FAKE_RECORD)
    env = {"PATH": f"{_hosts(tmp_path)}:{bin_}:{os.defpath}", "HOME": str(tmp_path), "VENV": str(venv.parent),
           "D": str(d), "OUT": str(tmp_path / "compose-out"), "SERVE_ROOT": str(tmp_path / "root"),
           "FLOCK_WORK": str(tmp_path / "fw"), "ORDER": str(tmp_path / "order"), "HOST": "vy-auditor-2",
           "RESEARCH_SOURCE_SHA": SHA, "GIT_CEILING_DIRECTORIES": str(tmp_path)}
    env = {k: v for k, v in {**env, **extra}.items() if v is not None}
    rc = subprocess.run(["bash", str(HERE.parent / "served_verdicts.sh"), "compose"], cwd=tree, env=env, check=False,
                        capture_output=True).returncode
    said = tmp_path / "compose-out" / "verdicts.txt"
    order = tmp_path / "order"
    return rc, said.read_text().splitlines() if said.exists() else [], order.read_text().split() if order.exists() else []


COMPOSED = ("COMPOSE on vy-auditor-2 (COMPOSE_NODE vy-auditor-2): the final record and the composite of the served "
            "node vy-dc-h200, from its compose hand-off (verifier-side files, no witness; {audit}; the record's sampled "
            "proofs {sampled}), each file as it hashed it (the serve dir's private/compose/inputs.sha256): {verdict}")
STEP3 = "sampled-proofs-served"


def test_the_compose_follow_on_holds_the_hand_off_to_its_sums_and_composes_the_record_and_composite_off_the_node(tmp_path):
    """The compose follow-on, on a node of neither the serve nor the audit, from the hand-off alone: the copy held to the
    served node's sums, the inference-only verdict carried as the served node wrote it, then the final record about the
    served node from the hand-off's pieces at the steps it picked (`record --steps --off-node`, so its composed_on is this
    host, served_node false, from the hand-off), and the composite, told the served node and the hand-off. $OUT names no
    window."""
    assert _handed(tmp_path).returncode == 0
    rc, lines, order = _compose(tmp_path)
    d, out = tmp_path / "serve", tmp_path / "compose-out"
    assert rc == 0 and order == ["record", "composite"], (lines, order)
    assert lines[0] == COMPOSED.format(audit="no V3c audit", sampled=STEP3, verdict="PASS")
    assert [x.split(" ")[0] for x in lines[1:]] == ["INFERENCE-ONLY:", "RECORD", "COMPOSITE:"]
    assert lines[-1] == "COMPOSITE: composed on vy-auditor-2"
    argv = json.loads((tmp_path / "order.record").read_text())
    assert argv[:17] == [
        "record", "--serve-run", str(d), "--window", "7", "--sampled-proofs", str(d / "sampled-proofs-served"), "--pieces",
        str(d / "private" / "compose" / "pieces"), "--out", str(out / "record"), "--product", "remote", "--node",
        "vy-dc-h200", "--steps", "0,3"]
    assert (argv[17], len(argv[18]), argv[19:]) == ("--key-sha512", 128, ["--lane", "vllm", "--off-node", "--sampled-lean",
                                                                         "--tree", SHA])
    assert (tmp_path / "order.composite").read_text().split() == ["vy-dc-h200", "1", str(d), str(out)]
    assert json.loads((out / "record" / "inference-only.json").read_text()) == {"verdict": "accept"}
    assert not (d / "requests.jsonl").exists() and not _window_data(out, 7)


COMPOSE_CLASH = ("PREFLIGHT: the compose follow-on runs at the hand-off's settings (the serve dir's "
                 "private/compose/compose.env), and the environment sets another")


@pytest.mark.parametrize("keep, change, extra, line", [
    ((), "sampled-proofs-served/window/m0/pub-0.bin", {}, COMPOSED.format(
        audit="no V3c audit", sampled=STEP3, verdict="FAIL (the serve dir's private/compose/check.log)")),
    ((), None, {"HOST": "vy-dc-h200"}, "PREFLIGHT: this is the served node vy-dc-h200, which the prover controls, so "
     "nothing composed here claims a record: compose on COMPOSE_NODE (vy-auditor-2)"),
    (("advice/w7/private/advice.bin",), None, {}, "PREFLIGHT: the compose follow-on holds no witness of the served run, "
     "and this node's serve dir does (WITNESS advice/wK/private/advice.bin: the advice and its opening): compose on a "
     "node that holds the hand-off alone"),
    ((f"v3c-w7-{VTAG}/window/values/a.bin",), None, {}, "PREFLIGHT: the compose follow-on holds no witness of the served "
     f"run, and this node's serve dir does (WITNESS v3c-wK-{VTAG}/window/values/a.bin: a committed value file): compose "
     "on a node that holds the hand-off alone"),
    ((), None, {"PRODUCT": "onsite"}, f"{COMPOSE_CLASH} PRODUCT: unset it"),
    ((), None, {"RESEARCH_SOURCE_SHA": "cd" * 20}, "PREFLIGHT: the compose follow-on composes with the served node's "
     f"tree {SHA}, and this tree is {'cd' * 20}: run it at that commit, or set FOLLOW_ANY_TREE=1"),
    ((), "private/compose/compose.env", {}, "PREFLIGHT: the compose follow-on composes from the served node's hand-off, "
     "and the serve dir {d} holds none (private/compose/compose.env): copy each file its private/compose/inputs.txt "
     "lists to the same path here first"),
])
def test_a_compose_follow_on_refuses_the_served_node_a_witness_a_changed_copy_another_setting_or_tree(
        tmp_path, keep, change, extra, line):
    """No record is composed on the served node, which the prover controls, nor where a witness of the served run is
    held (the follow-on's serve dir holds the hand-off alone), and no retries: a copied file changed fails the copy's check
    before anything runs, and a setting other than the hand-off's, another tree or no hand-off is refused."""
    assert _handed(tmp_path).returncode == 0
    d = tmp_path / "serve"
    if change == "private/compose/compose.env":
        (d / change).unlink()
        listed = d / "private" / "compose" / "inputs.txt"
        listed.write_text("".join(f"{x}\n" for x in listed.read_text().splitlines() if x != change))
    elif change:
        (d / change).write_text("changed")
    rc, lines, order = _compose(tmp_path, keep, **extra)
    assert (rc, lines, order) == (3, [line.replace("{d}", str(d))], []), lines


def test_a_follow_on_under_compose_off_node_audits_then_hands_off_and_composes_nothing(tmp_path):
    """Under COMPOSE_OFF_NODE=1 the deferral's follow-on is prover-side (it holds the copy's value files), so after V3c's
    audit it composes neither the record nor the composite: it writes the compose hand-off, naming itself as where V3c's
    audit ran and that audit, the sampled-proofs audit of record, as the record's (step 3's, on local coins while V3c's
    are live, isn't carried), with the deferral's pieces and inference-only verdict. A compose follow-on on a third node
    then composes both from that hand-off alone, the record from V3c's audit, and refuses the follow-on's serve dir, which
    holds the value files."""
    verdict = {"format": "verity-bench/inference-only-verdict/v3", "verdict": "accept"}
    (tmp_path / "served-out" / "record").mkdir(parents=True)
    (tmp_path / "served-out" / "record" / "inference-only.json").write_text(json.dumps(verdict))
    assert _deferred(tmp_path).returncode == 0
    rc, lines, order, _limits = _follow(tmp_path)
    d, dc = tmp_path / "serve", tmp_path / "serve" / "private" / "compose"
    assert rc == 0 and order == ["audit", "period"], (lines, order)
    assert lines.pop(0) == V3C_ENTRY
    assert [x.split(" ")[0] for x in lines[4:]] == ["V3C-AUDIT", "V3C-PERIOD", "HANDOFF"]
    assert lines[-1] == HANDOFF_LINE.replace("{d}", str(d))
    v3c = f"v3c-w7-{VTAG}/audit"
    assert _env_of(dc / "compose.env", COMPOSES) == {
        "K": "7", "PRODUCT": "remote", "COMPOSITE": "1", "ALPHA": "", "GAMMA": "", "SERVED_NODE": "vy-dc-h200",
        "SERVED_DIR": str(d), "COMPOSE_NODE": "vy-auditor-2", "COMPOSE_TREE": SHA, "AUDITED_ON": "cpu-6",
        "SAMPLED_AUDIT": v3c, "STEPS": "0,3"}
    listed = (dc / "inputs.txt").read_text().splitlines()
    assert listed == [*sorted([
        "h1/log.jsonl", "h1/w7/record.json", "h1/w7/registration.json", "private/compose/compose.env",
        "private/compose/inference-only.json", "private/compose/pieces/pouw/answer.json", "private/pouw/period/answer.json",
        "private/pouw/period/case.json", f"{v3c}/summary.json"]),
        "private/compose/inputs.sha256"]
    assert S.witness(listed) == [] and json.loads((dc / "inference-only.json").read_text()) == verdict
    (tmp_path / "order").unlink()
    rc, lines, order = _compose(tmp_path, HOST="vy-auditor-2")
    assert rc == 0 and order == ["record", "composite"], (lines, order)
    assert lines[0] == COMPOSED.format(audit="V3c's audit from cpu-6", sampled=f"v3c-wK-{VTAG}/audit", verdict="PASS")
    argv = json.loads((tmp_path / "order.record").read_text())
    assert argv[argv.index("--sampled-proofs") + 1] == str(d / v3c) and "--sampled-lean" in argv
    for x in ("serve", "compose-out"):
        shutil.rmtree(tmp_path / x)
    (tmp_path / "served-node").rename(tmp_path / "serve")
    (tmp_path / "order").unlink()
    rc, lines, order = _compose(tmp_path, keep=(f"v3c-w7-{VTAG}/window/values/a.bin",), HOST="cpu-6")
    assert rc == 3 and order == [] and lines[0].startswith("PREFLIGHT: the compose follow-on holds no witness")


#: what the compose hand-off carries of `_pearlplus_piece`'s piece: the record's files of a piece, and what
#: `pearlplus_of_record` reads, each challenger session's public key at its path
PEARLPLUS_HANDED = ("coin.json", "development.json", "ledger.jsonl", "pearlplus-answer.json", "pearlplus-lean.json",
                    "piece.json", f"session/{SID}/out/key.json", f"session/{DSID}/out/key.json")


def test_the_hand_off_carries_pearlpluss_piece_of_record_which_the_compose_follow_on_composes_as_pouws_verdict(
        tmp_path, monkeypatch, capsys, live):
    """Pearl+'s piece goes with the compose hand-off with what `pearlplus_of_record` reads: the auditor's ledger, the coin
    record, Pearl+'s answer, the Lean verdict and each challenger session's public key, and no other file of the piece or
    its sessions, none a witness. The compose follow-on's record, from the hand-off alone with the window tied to W, its
    argv the follow-on's, composes accept with PoUW's W_cert the Lean's work; from a piece's files alone (no session key)
    it composes pending, naming the challenger key it can't read."""
    monkeypatch.setenv("RESEARCH_RUN_ID", "r1")
    monkeypatch.setattr(S.socket, "gethostname", lambda: "vy-auditor-2")
    monkeypatch.setattr(S, "lean_sampled_row", lambda *_a, **_k: _accepted())
    pieces, tie, pin, lane = _pearlplus_piece(tmp_path / "pp", monkeypatch, live)
    for f in ("piece.log", "verify.json", "session.tgz", f"session/{SID}/request.json", f"session/{SID}/out/record.json"):
        (pieces / "pouw" / f).write_text("{}")
    shutil.copytree(tie, tmp_path / "serve" / "pearlplus" / "w7" / "tie")
    run = _handed(tmp_path, piece=pieces / "pouw", POUW="pearlplus", PEARLPLUS_TIE="1", H1_TIE=None)
    assert run.returncode == 0, run.stderr
    d, dc = tmp_path / "serve", tmp_path / "serve" / "private" / "compose"
    handed = dc / "pieces" / "pouw"
    assert sorted(p.relative_to(handed).as_posix() for p in handed.rglob("*") if p.is_file()) == sorted(PEARLPLUS_HANDED)
    listed = (dc / "inputs.txt").read_text().splitlines()
    tied = [f"pearlplus/w7/tie/{f}" for f in (S.WINDOW_REGISTRATION, S.AUDIT_SUMMARY, S.CIRCUIT_RECORD)]
    assert set(listed) >= {*tied, *(f"private/compose/pieces/pouw/{f}" for f in PEARLPLUS_HANDED)}
    assert S.witness(listed) == []
    rc, lines, order = _compose(tmp_path, H1_KEY_SHA512=pin)
    assert rc == 0 and order == ["record", "composite"], (lines, order)
    assert lines[0] == COMPOSED.format(audit="no V3c audit", sampled=STEP3, verdict="PASS")
    argv = json.loads((tmp_path / "order.record").read_text())
    assert argv[argv.index("--window-tie") + 1] == str(d / "pearlplus" / "w7" / "tie")
    assert (argv[argv.index("--key-sha512") + 1], argv[argv.index("--lane") + 1]) == (pin, lane)
    argv[argv.index("--out") + 1] = str(tmp_path / "record")
    assert S.main(argv) == 0
    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1].removeprefix("RECORD "))
    rec = json.loads((tmp_path / "record" / "block-record.json").read_text())
    assert (line["composed"], line["record_missing"], line["verdicts"]["PoUW"], line["check"]) == ("pending", [], "accept", {})
    assert line["composed_on"]["from"] == S.HANDED_FROM
    assert rec["legs"]["pouw"]["quantity"] == {"symbol": "W_cert", "value": 5400, "unit": "work", "tiles": 40}
    alone = tmp_path / "alone" / "pouw"
    alone.mkdir(parents=True)
    for f in ("answer.json", "development.json", "piece.json", "refused.txt", "served.json", "runs.json"):
        if (handed / f).exists():
            shutil.copy2(handed / f, alone / f)
    argv[argv.index("--pieces") + 1], argv[argv.index("--out") + 1] = str(alone.parent), str(tmp_path / "alone-record")
    S.main(argv)
    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1].removeprefix("RECORD "))
    assert (line["composed"], line["verdicts"]["PoUW"]) == ("pending", S.DEV_VERDICT)
    assert "the live session's challenger key is not the compose's pin: no challenger key at " in line["missing"]["PoUW"]


def test_handoff_host_writes_a_deferrals_hand_off_again_at_its_settings_naming_where_v3cs_audit_ran(tmp_path):
    """`served_verdicts.sh handoff HOST` on a deferral's serve dir (its follow-on's copy, V3c's audit there) writes the
    compose hand-off again from the deferral's pieces and inference-only verdict at the settings it recorded, HOST where
    V3c's audit ran and the served node the record's. Without HOST where the serve dir holds V3c's audit, or with a
    setting other than the deferral's, it is refused and writes none."""
    verdict = {"format": "verity-bench/inference-only-verdict/v3", "verdict": "accept"}
    (tmp_path / "served-out" / "record").mkdir(parents=True)
    (tmp_path / "served-out" / "record" / "inference-only.json").write_text(json.dumps(verdict))
    assert _deferred(tmp_path).returncode == 0
    d, dc = tmp_path / "serve", tmp_path / "serve" / "private" / "compose"
    (d / f"v3c-w7-{VTAG}" / "audit" / "window").mkdir(parents=True)
    (d / f"v3c-w7-{VTAG}" / "audit" / "window" / "case.json").write_text(json.dumps({"evidence": "cafe"}))

    def handoff(out: str, *host: str, **extra: str) -> tuple[int, list[str]]:
        env = {"PATH": f"{_hosts(tmp_path)}:{os.defpath}", "HOME": str(tmp_path), "D": str(d), "OUT": str(tmp_path / out),
               "SERVE_ROOT": str(tmp_path / "root"), "HOST": "cpu-6", "RESEARCH_SOURCE_SHA": SHA,
               "GIT_CEILING_DIRECTORIES": str(tmp_path), **extra}
        rc = subprocess.run(["bash", str(HERE.parent / "served_verdicts.sh"), "handoff", *host],
                            cwd=tmp_path / "served-tree", env=env, check=False, capture_output=True).returncode
        said = tmp_path / out / "verdicts.txt"
        return rc, said.read_text().splitlines() if said.exists() else []

    assert handoff("no-host") == (3, ["HANDOFF refused: the serve dir holds V3c's audit, so the hand-off names the host it "
                                      "ran on: served_verdicts.sh handoff HOST"])
    assert handoff("clash", "cpu-6", K="8") == (3, [
        "HANDOFF refused: the hand-off is written at the deferral's settings (the serve dir's "
        "private/deferred/deferred.env), and the environment sets another K: unset it"])
    assert not dc.exists()
    assert handoff("out", "cpu-6") == (0, [HANDOFF_LINE.replace("{d}", str(d))])
    assert _env_of(dc / "compose.env", COMPOSES) == {
        "K": "7", "PRODUCT": "remote", "COMPOSITE": "1", "ALPHA": "", "GAMMA": "", "SERVED_NODE": "vy-dc-h200",
        "SERVED_DIR": str(d), "COMPOSE_NODE": "vy-auditor-2", "COMPOSE_TREE": SHA, "AUDITED_ON": "cpu-6",
        "SAMPLED_AUDIT": STEP3, "STEPS": "0,3"}
    listed = (dc / "inputs.txt").read_text().splitlines()
    assert {"private/compose/pieces/pouw/answer.json", f"v3c-w7-{VTAG}/audit/window/case.json"} <= set(listed)
    assert S.witness(listed) == [] and json.loads((dc / "inference-only.json").read_text()) == verdict


#: a compose gate that answers that it composed the record, accepted, at its pin on remote, keeping in $GATE_SEEN its
#: argv and each file of the tar it was sent
STUB_GATE = '''import json, os, sys, tarfile
with tarfile.open(fileobj=sys.stdin.buffer, mode="r|") as t:
    sent = {m.name: t.extractfile(m).read().decode(errors="replace") for m in t if m.isreg()}
open(os.environ["GATE_SEEN"], "w").write(json.dumps({"argv": sys.argv[1:], "sent": sent}))
print(json.dumps({"pinned": "c0" * 20, "pin": {"on": "remote"},
                  "compose": {"rc": 0, "files": {"verdicts.txt": 'RECORD {"composed": "accept"}\\n'}}}))
'''


def test_gate_compose_sends_a_deferrals_hand_off_at_its_settings_naming_where_v3cs_audit_ran(tmp_path):
    """`served_verdicts.sh gate-compose PIECES VERDICT HOST` on a deferral's serve dir (its follow-on's copy, V3c's audit
    there) writes the hand-off at the settings the deferral recorded, not the environment's, HOST where V3c's audit ran,
    and sends it to the compose gate. Without HOST where the serve dir holds V3c's audit, or with a setting other than
    the deferral's, it is refused and sends nothing."""
    verdict = {"format": "verity-bench/inference-only-verdict/v3", "verdict": "accept"}
    (tmp_path / "served-out" / "record").mkdir(parents=True)
    (tmp_path / "served-out" / "record" / "inference-only.json").write_text(json.dumps(verdict))
    assert _deferred(tmp_path).returncode == 0
    d, dd, dc = tmp_path / "serve", tmp_path / "serve" / "private" / "deferred", tmp_path / "serve" / "private" / "compose"
    (d / f"v3c-w7-{VTAG}" / "audit" / "window").mkdir(parents=True)
    (d / f"v3c-w7-{VTAG}" / "audit" / "window" / "case.json").write_text(json.dumps({"evidence": "cafe"}))
    gate, seen = tmp_path / "stub-gate.py", tmp_path / "gate-seen.json"
    gate.write_text(STUB_GATE)

    def gate_compose(out: str, *host: str, **extra: str) -> tuple[int, list[str]]:
        env = {"PATH": f"{_hosts(tmp_path)}:{os.defpath}", "HOME": str(tmp_path), "D": str(d), "OUT": str(tmp_path / out),
               "SERVE_ROOT": str(tmp_path / "root"), "HOST": "cpu-6", "RESEARCH_SOURCE_SHA": SHA,
               "GIT_CEILING_DIRECTORIES": str(tmp_path), "COMPOSE_GATE": f"{sys.executable} {gate}",
               "GATE_SEEN": str(seen), **extra}
        rc = subprocess.run(["bash", str(HERE.parent / "served_verdicts.sh"), "gate-compose", str(dd / "pieces"),
                             str(dd / "inference-only.json"), *host], cwd=tmp_path / "served-tree", env=env, check=False,
                            capture_output=True).returncode
        said = tmp_path / out / "verdicts.txt"
        return rc, said.read_text().splitlines() if said.exists() else []

    assert gate_compose("no-host") == (3, ["GATE-COMPOSE refused: the serve dir holds V3c's audit, so the hand-off names "
                                           "the host it ran on: served_verdicts.sh gate-compose PIECES VERDICT HOST"])
    assert gate_compose("clash", "cpu-6", K="8") == (3, [
        "GATE-COMPOSE refused: the hand-off is written at the deferral's settings (the serve dir's "
        "private/deferred/deferred.env), and the environment sets another K: unset it"])
    assert not seen.exists() and not dc.exists()
    assert gate_compose("out", "cpu-6") == (0, [
        HANDOFF_LINE.replace("{d}", str(d)), f"GATE-COMPOSE at the compose gate's tree {'c0' * 6} (on remote): exit 0, "
        f"the record accept, 1 files of its $OUT in {tmp_path / 'out' / 'compose'}"])
    assert _env_of(dc / "compose.env", COMPOSES) == {
        "K": "7", "PRODUCT": "remote", "COMPOSITE": "1", "ALPHA": "", "GAMMA": "", "SERVED_NODE": "vy-dc-h200",
        "SERVED_DIR": str(d), "COMPOSE_NODE": "vy-auditor-2", "COMPOSE_TREE": SHA, "AUDITED_ON": "cpu-6",
        "SAMPLED_AUDIT": STEP3, "STEPS": "0,3"}
    sent, listed = json.loads(seen.read_text()), (dc / "inputs.txt").read_text().splitlines()
    assert sent["argv"] == ["compose"] and sorted(sent["sent"]) == sorted(listed)
    assert sent["sent"]["private/compose/compose.env"] == (dc / "compose.env").read_text()
    assert f"v3c-w7-{VTAG}/audit/window/case.json" in listed and S.witness(listed) == []


#: the compose gate as `vy-compose-gate` serves it over ssh, its words its argv: `host.gate` on a compose host under
#: $GATE_ROOT with lane vllm's key, its compose `host.compose_handoff` in-process with $GATE_ENV added (no runuser); the
#: hand-off in transit with $TAMPER's member rewritten or a $CARRY member added
GATE_DRIVER = '''import io, json, os, sys, tarfile
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, os.environ["LIVE_VERIFIER"])
import host as H
root = Path(os.environ["GATE_ROOT"])
host = H.Host(root=root / "var", config=root / "compose.json", pins=root / "pins.jsonl", main=root / "main.txt",
              compose=True)
src = sys.stdin.buffer
if os.environ.get("TAMPER") or os.environ.get("CARRY"):
    buf = io.BytesIO()
    with tarfile.open(fileobj=src, mode="r|") as t, tarfile.open(fileobj=buf, mode="w") as w:
        for m in t:
            body = t.extractfile(m).read() if m.isreg() else b""
            if m.name == os.environ.get("TAMPER"):
                body = b"tampered"
                m.size = len(body)
            w.addfile(m, io.BytesIO(body))
        if os.environ.get("CARRY"):
            w.addfile(tarfile.TarInfo(os.environ["CARRY"]), io.BytesIO(b""))
    buf.seek(0)
    src = buf


def compose(argv, scratch):
    Path(os.environ["GATE_ARGV"]).write_text(json.dumps(argv))
    i = argv.index("compose")
    return H.compose_handoff(host, scratch, src, pinned=argv[i + 2], on=(argv[i + 3:] or [None])[0],
                             env=json.loads(os.environ["GATE_ENV"]))


sys.exit(H.gate(["--ssh", " ".join(sys.argv[1:]), "--key", "vllm"], host, sink=sys.stdout.buffer,
                pinned=os.environ["PIN"], busy=lambda: [], own=lambda p: None, compose=compose,
                now=datetime(2026, 11, 4, 12, tzinfo=timezone.utc), token="ab12"))
'''
GATE_PIN = "c0" * 20
#: the compose gate's directory (its `host.py`), which the driver imports from
LIVE_VERIFIER = HERE.parents[2] / "verity" / "protocols" / "poub" / "device" / "live_verifier"


def _gate_composed(tmp_path: Path, edit: Mapping[str, str | None] | None = None,
                   **extra: str) -> tuple[subprocess.CompletedProcess, Path]:
    """`_handed`'s serve dir with `_lean_handed`'s sampled proofs cases (v2's naming a5's ``BLOBS``), sent by
    `served_verdicts.sh gate-compose` to `GATE_DRIVER`, pinned at GATE_PIN on remote, where the compose runs this tree's
    `served_verdicts.sh compose` and record on vy-auditor-2 with a `FAKE_SAMPLED_LEAN`, no composite and no git (as the
    pinned archive has none). ``edit`` replaces the served node's sampled-proofs files before the hand-off (None deletes);
    ``extra`` adds to the sender's environment. The sender's run and $OUT."""
    from verity.circuits import codec

    d, out, gate = tmp_path / "serve", tmp_path / "served-out", tmp_path / "gate-host"
    v1 = {"params": {"verifier": "toy"}, "commitment": codec.canonical_json(LEAN_RECORD).hex(),
          "evidence": codec.canonical_json({"draw": [0, 3]}).hex()}
    sp = "sampled-proofs-served"
    cases = {f"{sp}/window/case.json": v1, f"{sp}/summary.json": {"refused": "the node's own verdict"},
             f"{sp}/window/case-v2.json": {**v1, "params": {"verifier": "toy", "proofs": {}},
                                           "blobs": [str(d / b) for b in BLOBS]}}
    gate.mkdir()
    H = _live_host()
    H.record_pin(gate / "pins.jsonl", GATE_PIN, datetime(2026, 11, 4, 11, tzinfo=UTC), on="remote", seal=lambda p: None)
    driver = tmp_path / "gate-driver.py"
    driver.write_text(GATE_DRIVER)
    exe = tmp_path / "vstar-recursive"
    exe.write_text(FAKE_SAMPLED_LEAN)
    exe.chmod(0o755)
    answer = {**_accepted()["answer"], "protocol": "verity/sampled-proofs/v1", "commitment": REG.digest(LEAN_RECORD)}
    gate_env = {"PATH": f"{_hosts(tmp_path)}:{Path(sys.executable).parent}:/usr/bin:/bin", "HOST": "vy-auditor-2",
                "SAMPLED_EXE": str(exe), "SEEN": str(tmp_path / "seen.json"), "ANSWER": json.dumps(answer),
                "GIT_DIR": str(tmp_path / "no-git")}
    cases.update({f"{sp}/{name}": None if body is None else json.loads(body) for name, body in (edit or {}).items()})
    run = _handed(tmp_path, files=cases, cmd=("gate-compose", str(out / "pieces"), str(out / "record" / "inference-only.json")),
                  pouw={"verdict": "accept", "reason": None, "commitment": "ef" * 64, **ACCEPTS["pouw"]},
                  COMPOSITE="", COMPOSE_GATE=f"{sys.executable} {driver}", LIVE_VERIFIER=str(LIVE_VERIFIER),
                  GATE_ROOT=str(gate), PIN=GATE_PIN, GATE_ENV=json.dumps(gate_env), GATE_ARGV=str(tmp_path / "gate.argv"),
                  **extra)
    return run, out


def _live_host():
    spec = importlib.util.spec_from_file_location("live_host", LIVE_VERIFIER / "host.py")
    mod = sys.modules.setdefault(spec.name, importlib.util.module_from_spec(spec))
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("edit", [
    None,
    {"summary.json": json.dumps({"audit": {"accepted": True}})},
    {"summary.json": None},
])
def test_gate_compose_sends_the_hand_off_to_the_compose_gate_which_composes_the_record_at_its_pin_and_answers_it(
        tmp_path, edit):
    """`gate-compose` sends the hand-off to the compose gate (lane vllm's key, the word `compose`, the tar on stdin), which
    receives it in a scratch of its own, holds it to the served node's sums, rebases the cases' blobs onto its copy and
    composes the record there with the pinned tree, off the served node: sampled proofs' row is its Lean's answer on the
    copy's v2 case and blobs, whatever summary the served node wrote, and the record, pending the remote check's answer
    with both its subprotocols accepted, names the gate's pin and the branch infra pinned it on. The answer lands in the sender's $OUT/compose, and the gate removes its scratch."""
    run, out = _gate_composed(tmp_path, edit)
    gate = tmp_path / "gate-host"
    lines = (out / "verdicts.txt").read_text().splitlines()
    assert run.returncode == 0, (run.stderr, lines)
    assert lines[0].startswith("HANDOFF ")
    assert lines[-1].startswith(f"GATE-COMPOSE at the compose gate's tree {GATE_PIN[:12]} (on remote): exit 0, the record "
                                "pending, ") and lines[-1].endswith(f" files of its $OUT in {out / 'compose'}"), lines
    argv = json.loads((tmp_path / "gate.argv").read_text())
    scratch = gate / "var" / "scratch" / "compose-20261104T120000Z-ab12"
    assert argv[:3] == ["runuser", "-u", "auditor"] and argv[-4:] == ["compose", str(scratch), GATE_PIN, "remote"]
    composed = json.loads((out / "compose" / "record" / "verdicts.json").read_text())
    assert composed["composed"]["verdict"] == "pending" and all(
        composed["protocols"][k]["verdict"] == "accept" for k in ("integrity", "work"))
    assert composed["composed_on"] == {"host": socket.gethostname(), "served_node": False, "from": S.HANDED_FROM,
                                       "tree": GATE_PIN, "tree_on": "remote"}
    row = composed["protocols"]["integrity"]
    assert row["answer_from"] == S.LEAN_HERE and row["answer"]["commitment"] == REG.digest(LEAN_RECORD), row
    seen = json.loads((tmp_path / "seen.json").read_text())
    assert seen["argv"][5:] == [str(scratch / "serve" / b) for b in BLOBS]
    assert not scratch.exists() and list((gate / "var" / "scratch").iterdir()) == []
    said = json.loads((out / "compose" / "gate.json").read_text())
    assert (said["pinned"], said["pin"], said["by"], said["compose"]["rc"]) == (
        GATE_PIN, {"line": 0, "utc": "2026-11-04T11:00:00Z", "on": "remote"}, "vllm", 0)
    logged = [json.loads(x) for x in (gate / "var" / "composes.jsonl").read_text().splitlines()]
    assert [(x["by"], x["rc"], x["refused"]) for x in logged] == [("vllm", 0, None)]
    assert logged[0]["sums_sha512"] == hashlib.sha512(
        (tmp_path / "serve" / "private" / "compose" / "inputs.sha256").read_bytes()).hexdigest() == said["compose"]["sums_sha512"]
    assert not _window_data(out, 7)


@pytest.mark.parametrize("extra, said", [
    ({"TAMPER": "sampled-proofs-served/summary.json"}, "sampled-proofs-served/summary.json isn't the file the served node "
     "hashed: its SHA-256 isn't its sum (1 such)"),
    ({"TAMPER": "sampled-proofs-served/window/case-v2.json"}, "sampled-proofs-served/window/case-v2.json isn't the file the "
     "served node hashed"),
    ({"CARRY": "private/h1/key"}, "the hand-off carries the served run's witness: private/h1/key (H1's prover state (its "
     "key))"),
    ({"CARRY": "sampled-proofs-served/window/m0/inst-0.bin"}, "the hand-off carries the served run's witness: "
     "sampled-proofs-served/window/m0/inst-0.bin"),
    ({"CARRY": "notes.txt"}, "the hand-off carries notes.txt, which its sums don't list"),
])
def test_the_compose_gate_refuses_a_hand_off_off_its_sums_or_carrying_a_witness_and_composes_nothing(tmp_path, extra, said):
    """No retries, and nothing composed from bytes the served node didn't hash or beside a witness: a file changed in
    transit (even the summary a Lean row would never read) or a file added is refused before anything runs, the refusal
    the sender's one line and exit 3, the gate's scratch removed and its log naming the refusal."""
    run, out = _gate_composed(tmp_path, **extra)
    lines = (out / "verdicts.txt").read_text().splitlines()
    assert run.returncode == 3 and lines[-1].startswith("GATE-COMPOSE refused by the compose gate: "), lines
    assert said in lines[-1] and not (tmp_path / "seen.json").exists()
    assert not (out / "compose" / "record").exists()
    gate = tmp_path / "gate-host" / "var"
    assert list((gate / "scratch").iterdir()) == []
    logged = [json.loads(x) for x in (gate / "composes.jsonl").read_text().splitlines()]
    assert len(logged) == 1 and logged[0]["rc"] == 3 and said in logged[0]["refused"]


def test_no_answer_from_the_compose_gate_names_the_gate_commands_last_error(tmp_path):
    """A gate that never answers (ssh refused the key, the host unreachable) fails with ssh's own last line, so the
    sender's one line says why without a second look at the serve dir."""
    answer, err = tmp_path / "answer.json", tmp_path / "gate.err"
    answer.write_text("")
    err.write_text("Warning: Permanently added 'h' (ED25519) to the list of known hosts.\n"
                   "research@h: Permission denied (publickey).\n\n")
    assert S.composed(answer, tmp_path / "out", 255, err) == (
        1, "GATE-COMPOSE failed (ssh exit 255): no answer from the compose gate (IndexError): research@h: Permission "
           "denied (publickey).")
    assert S.composed(answer, tmp_path / "out", 255) == (
        1, "GATE-COMPOSE failed (ssh exit 255): no answer from the compose gate (IndexError)")


#: rsync as `served_verdicts.sh copy` calls it: each file --files-from lists, from the source dir to $COPIED/HOST/PATH
#: for the destination HOST:PATH, its arguments in $COPIED.args, exiting $RSYNC_EXIT
FAKE_RSYNC = f'''exec {sys.executable} - "$@" <<'EOF'
import json, os, shutil, sys
a = sys.argv[1:]
json.dump(a, open(os.environ["COPIED"] + ".args", "w"))
listed = next(x.split("=", 1)[1] for x in a if x.startswith("--files-from="))
host, dest = a[-1].split(":", 1)
for f in open(listed).read().split():
    to = os.path.join(os.environ["COPIED"], host.split("@")[-1], dest.lstrip("/"), f)
    os.makedirs(os.path.dirname(to), exist_ok=True)
    shutil.copy2(os.path.join(a[-2], f), to)
sys.exit(int(os.environ.get("RSYNC_EXIT", "0")))
EOF'''


def _copy(tmp_path: Path, kind: str, to: str, **extra: str) -> tuple[int, list[str], list[str]]:
    """`served_verdicts.sh copy KIND TO` with D alone on the node `extra`'s HOST names (default vy-dc-h200), through
    `FAKE_RSYNC`: the exit status, the lines it said and the files that reached TO's serve dir."""
    d, copied = tmp_path / "serve", tmp_path / "copied"
    _script(tmp_path / "copy-bin" / "rsync", FAKE_RSYNC)
    (tmp_path / "served-tree").mkdir(exist_ok=True)
    env = {"PATH": f"{_hosts(tmp_path)}:{tmp_path / 'copy-bin'}:{os.defpath}", "HOME": str(tmp_path), "D": str(d),
           "OUT": str(tmp_path / "copy-out"), "SERVE_ROOT": str(tmp_path / "root"), "HOST": "vy-dc-h200",
           "COPIED": str(copied), "RESEARCH_SOURCE_SHA": SHA, "GIT_CEILING_DIRECTORIES": str(tmp_path), **extra}
    rc = subprocess.run(["bash", str(HERE.parent / "served_verdicts.sh"), "copy", kind, to], cwd=tmp_path / "served-tree",
                        env=env, check=False, capture_output=True).returncode
    said = tmp_path / "copy-out" / "verdicts.txt"
    there = copied / to.split("@")[-1] / str(d).lstrip("/")
    return (rc, said.read_text().splitlines() if said.exists() else [],
            sorted(str(p.relative_to(there)) for p in there.rglob("*") if p.is_file()) if there.exists() else [])


@pytest.mark.parametrize("kind, to, host, what", [
    ("compose", "vy-auditor-2", "vy-dc-h200", "the hand-off's verifier-side"),
    ("deferred", "research@vy-nebius-cpu-6", "vy-dc-h200", "the deferral's prover-side"),
    ("compose", "vy-auditor-2", "cpu-6", "the hand-off's verifier-side"),
])
def test_copy_sends_each_file_a_list_names_and_nothing_else_to_the_same_path_on_its_host(tmp_path, kind, to, host, what):
    """`served_verdicts.sh copy deferred|compose HOST` sends the files a deferral's or a hand-off's inputs.txt lists, its
    sums among them, and nothing else of the serve dir, to the same path on HOST, creating it there; the copy passes the
    sums its follow-on holds it to. A hand-off goes to COMPOSE_NODE from the served node or from a deferral's follow-on
    (cpu-6, which holds the window's witness and sends none of it)."""
    if kind == "compose" and host == "vy-dc-h200":
        assert _handed(tmp_path).returncode == 0
    else:
        assert _deferred(tmp_path).returncode == 0
        if kind == "compose":
            assert _follow(tmp_path)[0] == 0
    d = tmp_path / "serve"
    listed = (d / "private" / kind / "inputs.txt").read_text().splitlines()
    rc, lines, there = _copy(tmp_path, kind, to, HOST=host)
    assert rc == 0 and there == sorted(listed), (lines, there)
    assert lines == [f"COPY {what} list's {len(listed)} files to {to.split('@')[-1]}, each to the same path there: PASS"]
    assert kind == "deferred" or S.witness(there) == []
    args = json.loads((tmp_path / "copied.args").read_text())
    assert args[-2:] == [f"{d}/", f"{to}:{d}/"] and f"--rsync-path=mkdir -p {d} && rsync" in args
    copy = tmp_path / "copied" / to.split("@")[-1] / str(d).lstrip("/")
    subprocess.run(["sha256sum", "--check", "--strict", "--quiet", f"private/{kind}/inputs.sha256"], cwd=copy, check=True)


@pytest.mark.parametrize("to, sub", [("research@vy-nebius-cpu-6:", "serve"), ("research@vy-nebius-cpu-6:incoming/", "incoming/serve")])
def test_copy_to_a_write_only_rsync_sends_the_list_under_the_serve_dirs_name_making_no_directory(tmp_path, to, sub):
    """`copy deferred HOST:DIR` sends the same files under DIR/<the serve dir's name>/ in the directory HOST's rsync writes
    to, without the remote mkdir a write-only rsync (`rrsync -wo`) refuses to run, and the copy passes its sums there."""
    assert _deferred(tmp_path).returncode == 0
    d = tmp_path / "serve"
    listed = (d / "private" / "deferred" / "inputs.txt").read_text().splitlines()
    rc, lines, _ = _copy(tmp_path, "deferred", to)
    there = tmp_path / "copied" / "vy-nebius-cpu-6" / sub
    assert rc == 0 and sorted(str(p.relative_to(there)) for p in there.rglob("*") if p.is_file()) == sorted(listed), lines
    assert lines == [f"COPY the deferral's prover-side list's {len(listed)} files to vy-nebius-cpu-6, under {sub}/ in its "
                     "rsync's directory, which its node moves to the serve dir's path: PASS"]
    args = json.loads((tmp_path / "copied.args").read_text())
    assert args[-2:] == [f"{d}/", f"research@vy-nebius-cpu-6:{sub}/"] and not any(a.startswith("--rsync-path") for a in args)
    subprocess.run(["sha256sum", "--check", "--strict", "--quiet", "private/deferred/inputs.sha256"], cwd=there, check=True)


@pytest.mark.parametrize("made, kind, to, extra, rc, line", [
    ("deferred", "deferred", "vy-auditor-2", {}, 3, "COPY refused: the deferral's list holds the window's witness, and "
     "vy-auditor-2 is COMPOSE_NODE, an auditor host where no witness goes: copy it to a prover node"),
    ("deferred", "deferred", "research@vy-auditor-3.example", {"COMPOSE_NODE": "vy-auditor-3"}, 3, "COPY refused: the "
     "deferral's list holds the window's witness, and vy-auditor-3.example is COMPOSE_NODE, an auditor host where no "
     "witness goes: copy it to a prover node"),
    ("deferred", "deferred", "research@vy-auditor-2:incoming", {}, 3, "COPY refused: the deferral's list holds the "
     "window's witness, and vy-auditor-2 is COMPOSE_NODE, an auditor host where no witness goes: copy it to a prover node"),
    ("handed", "compose", "vy-dc-h200.example", {"HOST": "cpu-6"}, 3, "COPY refused: vy-dc-h200.example is the served "
     "node, which the prover controls: copy the hand-off to COMPOSE_NODE (vy-auditor-2)"),
    ("handed+witness", "compose", "vy-auditor-2", {}, 3, "COPY refused: the hand-off's list names the served run's "
     "witness (WITNESS requests.jsonl: the requests' prompt and served ids, rows and seeds)"),
    ("deferred", "compose", "vy-auditor-2", {}, 3, "COPY refused: the serve dir {d} holds no compose list "
     "(private/compose/inputs.txt and compose.env)"),
    ("handed", "values", "vy-auditor-2", {}, 3, "COPY refused: served_verdicts.sh copy deferred|compose HOST[:DIR], not values"),
    ("handed", "compose", "vy-auditor-2", {"RSYNC_EXIT": "23"}, 23, "COPY the hand-off's verifier-side list's 22 files "
     "to vy-auditor-2, each to the same path there: FAIL (exit 23, the serve dir's private/copy-compose.log)"),
])
def test_copy_never_sends_a_deferrals_witness_to_compose_node_nor_a_hand_off_to_the_served_node_or_with_a_witness(
        tmp_path, made, kind, to, extra, rc, line):
    """The deferral's list is prover-side (the window's witness), so it never goes to COMPOSE_NODE, the deferral's or the
    environment's, however its host is written; a hand-off never goes back to the served node, and its list is held to
    `served_verdicts.py witness` again before it leaves, so a list edited to name a witness sends nothing. No list, or
    another kind, is refused, and a failed transfer says so and where its log is."""
    if made == "deferred":
        assert _deferred(tmp_path).returncode == 0
    else:
        assert _handed(tmp_path).returncode == 0
    d = tmp_path / "serve"
    if made == "handed+witness":
        with open(d / "private" / "compose" / "inputs.txt", "a") as listed:
            listed.write("requests.jsonl\n")
    got, lines, there = _copy(tmp_path, kind, to, **extra)
    assert (got, lines) == (rc, [line.replace("{d}", str(d))]), lines
    assert there == [] or rc != 3


ONSITE_PIECES = HERE.parents[2] / "benchmarks" / "pous" / "onsite_pieces.sh"


def _script(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(0o755)


FRONT = '"$PY" benchmarks/network_traces/certifier_piece.py front "$1" "$2" "$3" --bucket-ms 4000 --warmup 0'


IDS = "--protocol ids --id-payload 503"


@pytest.mark.parametrize("load, front, takes, serve_args, line", [
    ("varied", FRONT, "--gate --warmup --bucket-ms", IDS, None),
    ("front", FRONT, "--gate --warmup --bucket-ms", IDS, None),
    ("varied", "", "", "", None),
    ("varied", FRONT, "--gate --warmup --bucket-ms-x", IDS, "PREFLIGHT: CERTIFIER_FRONT passes --bucket-ms, which this "
                                                             "tree's certifier front doesn't take"),
    ("front", "", "", "", "PREFLIGHT: LOAD=front sends the front's group alone, and CERTIFIER_FRONT is unset"),
    ("front", FRONT, "--gate --warmup --bucket-ms", "--seal-hold-gb 8", "PREFLIGHT: the certifier front's warden sends "
     "id-form requests, which serve-openai answers only under SERVE_ARGS='--protocol ids --id-payload 503'"),
])
def test_the_preflight_refuses_a_front_flag_the_trees_front_does_not_take_and_a_front_load_with_no_front(
        tmp_path, load, front, takes, serve_args, line):
    tree, venv = tmp_path / "tree", tmp_path / "venv" / "bin"
    (tree / "benchmarks" / "network_traces").mkdir(parents=True)
    (tree / "benchmarks" / "network_traces" / "certifier_piece.py").write_text(
        f"import sys\nassert sys.argv[1:] == ['front', '--help']\nprint('usage: certifier_piece.py front PORT [{takes}]')\n")
    _script(venv / "python", f'exec {sys.executable} "$@"')
    env = {"PATH": os.defpath, "HOME": str(tmp_path), "VENV": str(venv.parent), "OUT": str(tmp_path / "out"),
           "LOAD": load, "CERTIFIER_FRONT": front, "SERVE_ARGS": serve_args}
    rc = subprocess.run(["bash", str(HERE.parent / "served_verdicts.sh"), "front-flags"], cwd=tree, env=env,
                        check=False).returncode
    said = (tmp_path / "out" / "verdicts.txt").read_text().splitlines() if (tmp_path / "out" / "verdicts.txt").exists() else []
    assert (rc, said) == ((3, [line]) if line else (0, []))


@pytest.mark.parametrize("product, extra, line", [
    ("remote", {}, None),
    ("remote", {"POUS_CMD": "true"}, "PREFLIGHT: PRODUCT=remote runs no onsite piece, and POUS_CMD is set: unset it"),
    ("remote", {"CERTIFIER_CHAIN_CMD": "true"},
     "PREFLIGHT: PRODUCT=remote runs no onsite piece, and CERTIFIER_CHAIN_CMD is set: unset it"),
    ("remote", {"CERTIFIER_FRONT": FRONT},
     "PREFLIGHT: PRODUCT=remote runs no onsite piece, and CERTIFIER_FRONT is set: unset it"),
    ("onsite", {}, None),
    ("offsite", {}, "PREFLIGHT: PRODUCT is onsite or remote, not offsite"),
])
def test_remote_sets_no_onsite_default_and_its_preflight_refuses_an_onsite_command(tmp_path, product, extra, line):
    """PRODUCT=remote drops the onsite pieces: in a tree that has the wipe's piece, with ONSITE_PIECES naming the file of
    onsite's defaults, its default WIPE_CMD isn't set, and the preflight refuses any onsite piece's command set, before
    anything runs."""
    tree, venv = tmp_path / "tree", tmp_path / "venv" / "bin"
    _script(tree / "benchmarks" / "pous" / "streamed_wipe" / "served_wipe.sh", "exit 0")
    _script(venv / "python", f'exec {sys.executable} "$@"')
    env = {"PATH": os.defpath, "HOME": str(tmp_path), "VENV": str(venv.parent), "OUT": str(tmp_path / "out"),
           "PRODUCT": product, "ONSITE_PIECES": str(ONSITE_PIECES), **extra}
    rc = subprocess.run(["bash", str(HERE.parent / "served_verdicts.sh"), "front-flags"], cwd=tree, env=env,
                        check=False).returncode
    said = (tmp_path / "out" / "verdicts.txt").read_text().splitlines() if (tmp_path / "out" / "verdicts.txt").exists() else []
    assert (rc, said) == ((3, [line]) if line else (0, []))


@pytest.mark.parametrize("product, pieces, called", [
    ("onsite", "pieces.sh", ["onsite"]),
    ("onsite", None, []),
    ("remote", "pieces.sh", []),
])
def test_onsite_defaults_come_from_the_file_onsite_pieces_names(tmp_path, product, pieces, called):
    """Under PRODUCT=onsite, ONSITE_PIECES (a path from the tree) names the file whose `onsite_piece_commands` sets the
    onsite pieces' defaults; unset, nothing is sourced, and under PRODUCT=remote the file is read but sets none."""
    tree, venv = tmp_path / "tree", tmp_path / "venv" / "bin"
    tree.mkdir()
    (tree / "pieces.sh").write_text(f'onsite_piece_commands() {{ echo "$PRODUCT" >> {tmp_path}/called; }}\n')
    _script(venv / "python", f'exec {sys.executable} "$@"')
    env = {"PATH": os.defpath, "HOME": str(tmp_path), "VENV": str(venv.parent), "OUT": str(tmp_path / "out"),
           "PRODUCT": product, **({"ONSITE_PIECES": pieces} if pieces else {})}
    rc = subprocess.run(["bash", str(HERE.parent / "served_verdicts.sh"), "front-flags"], cwd=tree, env=env,
                        check=False).returncode
    assert (rc, (tmp_path / "called").read_text().split() if (tmp_path / "called").exists() else []) == (0, called)


def test_a_missing_onsite_pieces_file_is_refused(tmp_path):
    tree, venv = tmp_path / "tree", tmp_path / "venv" / "bin"
    tree.mkdir()
    _script(venv / "python", f'exec {sys.executable} "$@"')
    env = {"PATH": os.defpath, "HOME": str(tmp_path), "VENV": str(venv.parent), "OUT": str(tmp_path / "out"),
           "ONSITE_PIECES": "nowhere/pieces.sh"}
    got = subprocess.run(["bash", str(HERE.parent / "served_verdicts.sh"), "front-flags"], cwd=tree, env=env,
                         check=False, capture_output=True, text=True)
    assert (got.returncode, got.stderr.strip()) == (3, f"PREFLIGHT: ONSITE_PIECES={tree}/nowhere/pieces.sh is not a file")


@pytest.mark.parametrize("files, keys, want", [
    ((), (), ("", "", "")),
    (("benchmarks/pous/streamed_wipe/served_wipe.sh",), (), ("wipe", "", "")),
    (("benchmarks/pous/served_piece.sh",), ("vy-auditor2-pous",), ("", "pous", "collect")),
    (("benchmarks/one_stage/served_bandwidth.sh", "store/setup.json"), ("vy-auditor2-bw",), ("", "", "session")),
    (("benchmarks/one_stage/served_bandwidth.sh",), ("vy-auditor2-bw",), ("", "", "")),
])
def test_onsite_pieces_defaults_are_set_where_the_tree_has_each_piece_and_the_node_its_key(tmp_path, files, keys, want):
    """`onsite_piece_commands` (benchmarks/pous/onsite_pieces.sh) sets WIPE_CMD where the tree has the wipe's piece,
    POUS_CMD where it has PoUS's and the node the memory challenger's key, and BANDWIDTH_CMD to collect PoUS's session
    then, else to the bandwidth session where the tree has it and the node the verifier's key and its registered store."""
    tree, home = tmp_path / "tree", tmp_path / "home"
    for f in files:
        (tree / f).parent.mkdir(parents=True, exist_ok=True)
        (tree / f).write_text("")
    (home / ".ssh").mkdir(parents=True)
    for k in keys:
        (home / ".ssh" / k).write_text("")
    got = subprocess.run(["bash", "-c", f'. {ONSITE_PIECES}; onsite_piece_commands; '
                          'printf "%s\\n" "${WIPE_CMD-}" "${POUS_CMD-}" "${BANDWIDTH_CMD-}"'],
                         env={"PATH": os.defpath, "HOME": str(home), "TREE": str(tree), "BW_STORE": str(tree / "store")},
                         check=True, capture_output=True, text=True).stdout.split("\n")[:3]
    marks = {"wipe": "streamed_wipe/served_wipe.sh", "pous": "served_piece.sh", "collect": "served_bandwidth.py collect",
             "session": "served_bandwidth.sh"}
    assert all((marks[w] in g) if w else g == "" for w, g in zip(want, got)), got


def test_settings_are_read_before_the_defaults_and_an_unreadable_file_is_refused(tmp_path):
    """SETTINGS names the file a load's counts are launched from, so the job record's command holds only its path."""
    tree, venv = tmp_path / "tree", tmp_path / "venv" / "bin"
    tree.mkdir()
    _script(venv / "python", f'exec {sys.executable} "$@"')
    settings = tmp_path / "private" / "settings.env"
    settings.parent.mkdir()
    settings.write_text("LOAD=front\nCERTIFIER_FRONT=\n")
    env = {"PATH": os.defpath, "HOME": str(tmp_path), "VENV": str(venv.parent), "OUT": str(tmp_path / "out"),
           "SETTINGS": str(settings)}
    rc = subprocess.run(["bash", str(HERE.parent / "served_verdicts.sh"), "front-flags"], cwd=tree, env=env,
                        check=False).returncode
    said = (tmp_path / "out" / "verdicts.txt").read_text().splitlines()
    assert (rc, said) == (3, ["PREFLIGHT: LOAD=front sends the front's group alone, and CERTIFIER_FRONT is unset"])
    got = subprocess.run(["bash", str(HERE.parent / "served_verdicts.sh"), "front-flags"], cwd=tree,
                         env={**env, "SETTINGS": str(tmp_path / "missing.env")}, check=False, capture_output=True, text=True)
    assert (got.returncode, got.stderr.strip()) == (3, f"PREFLIGHT: SETTINGS={tmp_path / 'missing.env'} is not a readable file")


@pytest.mark.parametrize("product, extra, line", [
    ("remote", {}, None),
    ("remote", {"POUS_CMD": "true"}, "PREFLIGHT: PRODUCT=remote runs no onsite piece, and POUS_CMD is set: unset it"),
    ("remote", {"CERTIFIER_CHAIN_CMD": "true"},
     "PREFLIGHT: PRODUCT=remote runs no onsite piece, and CERTIFIER_CHAIN_CMD is set: unset it"),
    ("remote", {"CERTIFIER_FRONT": FRONT},
     "PREFLIGHT: PRODUCT=remote runs no onsite piece, and CERTIFIER_FRONT is set: unset it"),
    ("onsite", {}, None),
    ("offsite", {}, "PREFLIGHT: PRODUCT is onsite or remote, not offsite"),
])
def test_remote_sets_no_onsite_default_and_its_preflight_refuses_an_onsite_command(tmp_path, product, extra, line):
    """PRODUCT=remote drops the onsite pieces: in a tree that has the wipe's piece its default WIPE_CMD isn't set, and
    the preflight refuses any onsite piece's command set, before anything runs."""
    tree, venv = tmp_path / "tree", tmp_path / "venv" / "bin"
    _script(tree / "benchmarks" / "pous" / "streamed_wipe" / "served_wipe.sh", "exit 0")
    _script(venv / "python", f'exec {sys.executable} "$@"')
    env = {"PATH": os.defpath, "HOME": str(tmp_path), "VENV": str(venv.parent), "OUT": str(tmp_path / "out"),
           "PRODUCT": product, **extra}
    rc = subprocess.run(["bash", str(HERE.parent / "served_verdicts.sh"), "front-flags"], cwd=tree, env=env,
                        check=False).returncode
    said = (tmp_path / "out" / "verdicts.txt").read_text().splitlines() if (tmp_path / "out" / "verdicts.txt").exists() else []
    assert (rc, said) == ((3, [line]) if line else (0, []))


FAKE_COMPOSITE = '''import json, sys
from pathlib import Path
COMPONENTS = {"onsite": ("verity/pous/p2/v4/run", "verity/wipe/v0"),
              "remote": ("verity/pouw/v1", "verity/sampled-proofs/one-stage/v2")}
if __name__ == "__main__":
    a = sys.argv[1:]
    Path(a[a.index("--private-out") + 1]).mkdir(parents=True)
    Path(a[a.index("--public-out") + 1]).mkdir(parents=True)
    Path(a[a.index("--public-out") + 1], "composite.json").write_text(json.dumps({"product": a[0]}))
    cases = [a[i + 1] for i, x in enumerate(a) if x == "--cases"]
    Path(a[a.index("--private-out") + 1], "argv.json").write_text(json.dumps(cases))
    Path(a[a.index("--private-out") + 1], "development.json").write_text(
        json.dumps(a[a.index("--development") + 1] if "--development" in a else None))
    print("PIECE verity/pous/p2/v4/run accept 0123456789abcdef0123")
    print("NOT_ESTABLISHED missing verity/wipe/v0")
    print(f"VERDICT verity/{a[0]}/v0 reject components")
    sys.exit(1)
'''


def test_the_composite_step_publishes_only_each_cases_verdict_and_each_products_lean_verdict(tmp_path):
    """served_composite.sh with a stand-in lake, served_cases.py and composite.py: PoUS's case, which its piece wrote
    under the serve dir's private dir beside its Lean's answer, is composed as written, with no card asked for and no
    Lean built but the two products'; the wipe's, which no piece wrote, is named not written and none is made; and
    served_cases.py's are written under the private dir, given no α or Γ while ALPHA and GAMMA are unset; each product
    gets the cases of its own subprotocols; OUT/verdicts.txt gets each
    CASE line's protocol and verdict, each case not written by its protocol, each product's NOT_ESTABLISHED and
    VERDICT lines and the host it composed on, never a PIECE line, a reason, a log or a path; each product's public
    output is OUT/composite/PRODUCT;
    and without composite.py in the tree, or onsite's cases without ONSITE_PIECES (benchmarks/pous/onsite_pieces.sh), it
    says so and fails, since no Lean answered. PRODUCT=remote reads neither PoUS's case nor the wipe's, and runs
    remote's Lean alone. A serve dir with no V3c
    audit hands served_cases.py step 3's records audit (``--audit SERVE_DIR/sampled-proofs-served``) once it holds a
    case, but not under H1's tie (a ``SERVE_DIR/h1/log.jsonl``), whose cases are on the record H1 named; and no product
    is handed a case under ``SERVE_DIR/private/pouw``, tied or not, whose period case reaches a product only as
    served_cases.py writes it, tied to that record. On the served node (SERVED_NODE unset) it composes only under COMPOSE_OFF_NODE=0, a
    development check its first line and each product's composite.py say claims no record. Where SERVED_NODE names
    another host it says it composed there, from the compose hand-off (COMPOSE_HANDOFF) or a deferral's prover-side copy
    of the serve dir's audit inputs."""
    tree, bin_, d, out = tmp_path / "tree", tmp_path / "bin", tmp_path / "serve", tmp_path / "out"
    out.mkdir()
    _script(bin_ / "lake", f'mkdir -p .lake/build/bin && touch ".lake/build/bin/$2" && echo "$2" >> {tmp_path}/built')
    _script(bin_ / "systemd-run", "exit 1")
    _script(bin_ / "gpu-lease", f'echo "$@" >> {tmp_path}/card')
    for w, protocol in (("pous/w0", "verity/pous/p2/v4/run"), ("other", "verity/certifier/v9")):
        (d / "private" / w).mkdir(parents=True)
        (d / "private" / w / "case.json").write_text(json.dumps({"evidence": "beef"}))
        (d / "private" / w / "answer.json").write_text(json.dumps({"protocol": protocol, "verdict": "accept"}))
    (out / "pieces" / "pous").mkdir(parents=True)
    (out / "pieces" / "pous" / "pous.json").write_text(json.dumps(
        {"window": 0, "verdict": "accept", "decided_by": {"program": "pous-run-protocol", "host": "the-served-node"}}))
    for pkg in ("protocols/remote/verifier", "protocols/onsite/verifier"):
        (tree / "verity" / pkg).mkdir(parents=True)
    (tree / "benchmarks" / "one_stage").mkdir(parents=True)
    (tree / "benchmarks" / "one_stage" / "composite.py").write_text(FAKE_COMPOSITE)
    v2 = d / "private" / "sampled-proofs-v2"
    (tree / "benchmarks" / "one_stage" / "served_cases.py").write_text(f'''import json, sys
from pathlib import Path
assert sys.argv[1:3] in (["--serve-dir", "{d}"], ["--audit", "{d}/sampled-proofs-served"]), sys.argv
assert sys.argv[3:] == ["--private", "{d}/private", "--from-division"], sys.argv
Path("{tmp_path}", "served_cases.argv").write_text(json.dumps(sys.argv[1:3]))
Path("{v2}").mkdir(parents=True)
Path("{v2}", "case.json").write_text(json.dumps({{"evidence": "beef"}}))
Path("{v2}", "answer.json").write_text(json.dumps({{"protocol": "verity/sampled-proofs/one-stage/v2", "verdict": "accept"}}))
print("CASE " + json.dumps({{"protocol": "verity/sampled-proofs/one-stage/v2", "verdict": "accept", "dir": "{v2}"}}))
print("CASE " + json.dumps({{"protocol": "verity/registration/v0", "verdict": "accept", "dir": "{d}/private/reg", "alpha_gamma": "235330880/1544757928"}}))
print("NOT_WRITTEN " + json.dumps({{"protocol": "verity/inference-only/v0", "reason": "no alpha 0123456789abcdef"}}))
sys.exit(1)
''')
    env = {"PATH": f"{bin_}:{os.defpath}", "HOME": str(tmp_path), "PY": sys.executable, "TPY": sys.executable,
           "COMPOSE_OFF_NODE": "0"}
    run = lambda: subprocess.run(["bash", str(HERE.parent / "served_composite.sh"), str(d), str(out)], cwd=tree, env=env,  # noqa: E731
                                 check=False, capture_output=True, text=True)
    host = subprocess.run(["hostname", "-s"], check=True, capture_output=True, text=True).stdout.strip()
    on = (f"COMPOSITE: composed on {host}, which holds the serve dir, from the cases the served node wrote there, not "
          "from the parties that produced them")
    why = (f"composed on the served node {host}, which the prover controls: a development check, no claim of record "
           "(COMPOSE_OFF_NODE=0)")
    dev = f"COMPOSITE DEVELOPMENT: {why}"
    assert run().returncode == 1
    assert (out / "verdicts.txt").read_text() == (f"{dev}\nCOMPOSITE: not run, onsite's own cases come from "
                                                  "ONSITE_PIECES's file, and it is unset\n")
    (out / "verdicts.txt").unlink()
    env["ONSITE_PIECES"] = str(ONSITE_PIECES)
    r = run()
    assert r.returncode == 0, r.stderr
    assert (out / "verdicts.txt").read_text().splitlines() == [
        dev, "COMPOSITE CASE verity/pous/p2/v4/run accept", "COMPOSITE NOT_WRITTEN verity/wipe/v0",
        "COMPOSITE CASES: PoUS's and the wipe's as their pieces wrote them, none made here",
        "COMPOSITE CASE verity/sampled-proofs/one-stage/v2 accept",
        "COMPOSITE CASE verity/registration/v0 accept",
        "COMPOSITE NOT_WRITTEN verity/inference-only/v0",
        "COMPOSITE CASES: served_cases.py exit 1, its log on the node",
        "COMPOSITE remote: NOT_ESTABLISHED missing verity/wipe/v0", "COMPOSITE remote: VERDICT verity/remote/v0 reject components",
        "COMPOSITE onsite: NOT_ESTABLISHED missing verity/wipe/v0", "COMPOSITE onsite: VERDICT verity/onsite/v0 reject components",
        on]
    p = d / "private" / "composite"
    assert json.loads((p / "remote" / "private" / "argv.json").read_text()) == [str(v2)]
    assert json.loads((p / "onsite" / "private" / "argv.json").read_text()) == [str(d / "private" / "pous" / "w0")]
    assert all(json.loads((p / x / "private" / "development.json").read_text()) == why for x in ("remote", "onsite"))
    assert (tmp_path / "built").read_text().split() == ["remote", "onsite"] and not (tmp_path / "card").exists()
    assert sorted(f.name for f in (d / "private" / "pous" / "w0").iterdir()) == ["answer.json", "case.json"]
    assert not (d / "private" / "wipe").exists()
    assert oct((d / "private").stat().st_mode & 0o777) == "0o700"
    assert sorted(out.iterdir()) == [out / "composite", out / "pieces", out / "verdicts.txt"]
    assert sorted(f.relative_to(out).as_posix() for f in (out / "composite").rglob("*") if f.is_file()) == [
        "composite/onsite/composite.json", "composite/remote/composite.json"]
    (tree / "benchmarks" / "one_stage" / "composite.py").unlink()
    (out / "verdicts.txt").unlink()
    shutil.rmtree(out / "composite")
    assert run().returncode == 1
    assert (out / "verdicts.txt").read_text() == f"{dev}\nCOMPOSITE: not run, this tree has no benchmarks/one_stage/composite.py\n"
    (tree / "benchmarks" / "one_stage" / "composite.py").write_text(FAKE_COMPOSITE)
    (out / "verdicts.txt").unlink()
    shutil.rmtree(d / "private")
    env["PRODUCT"] = "remote"
    r = run()
    assert r.returncode == 0, r.stderr
    assert (out / "verdicts.txt").read_text().splitlines() == [
        dev, "COMPOSITE CASE verity/sampled-proofs/one-stage/v2 accept", "COMPOSITE CASE verity/registration/v0 accept",
        "COMPOSITE NOT_WRITTEN verity/inference-only/v0", "COMPOSITE CASES: served_cases.py exit 1, its log on the node",
        "COMPOSITE remote: NOT_ESTABLISHED missing verity/wipe/v0", "COMPOSITE remote: VERDICT verity/remote/v0 reject components",
        on]
    assert sorted(x.name for x in (out / "composite").iterdir()) == ["remote"] and not (p / "onsite").exists()
    assert json.loads((tmp_path / "served_cases.argv").read_text()) == ["--serve-dir", str(d)]
    (d / "sampled-proofs-served" / "window").mkdir(parents=True)
    (d / "sampled-proofs-served" / "window" / "case.json").write_text("{}")
    (out / "verdicts.txt").unlink()
    shutil.rmtree(d / "private")
    assert run().returncode == 0
    assert json.loads((tmp_path / "served_cases.argv").read_text()) == ["--audit", str(d / "sampled-proofs-served")]
    (d / "h1").mkdir()
    (d / "h1" / "log.jsonl").write_text("")
    (out / "verdicts.txt").unlink()
    shutil.rmtree(d / "private")
    shutil.rmtree(out / "composite")
    piece = d / "private" / "pouw" / "period"
    piece.mkdir(parents=True)
    (piece / "case.json").write_text(json.dumps({"evidence": "beef"}))
    (piece / "answer.json").write_text(json.dumps({"protocol": "verity/pouw/v1", "verdict": "accept"}))
    assert run().returncode == 0
    assert json.loads((tmp_path / "served_cases.argv").read_text()) == ["--serve-dir", str(d)]
    assert json.loads((p / "remote" / "private" / "argv.json").read_text()) == [str(v2)]
    (d / "h1" / "log.jsonl").unlink()
    (out / "verdicts.txt").unlink()
    for x in (out / "composite", p, v2):
        shutil.rmtree(x)
    assert run().returncode == 0
    assert json.loads((p / "remote" / "private" / "argv.json").read_text()) == [str(v2)]
    (out / "verdicts.txt").unlink()
    for x in (out / "composite", p, v2):
        shutil.rmtree(x)
    env["SERVED_NODE"] = "the-served-node"
    assert run().returncode == 0
    said = (out / "verdicts.txt").read_text().splitlines()
    assert said[-1] == (
        f"COMPOSITE: composed on {host}, not the served node the-served-node: from a prover-side copy of its serve dir's "
        "audit inputs and the cases written here from it, not from the parties that produced them")
    assert not any("DEVELOPMENT" in x for x in said)
    assert json.loads((p / "remote" / "private" / "development.json").read_text()) is None
    (out / "verdicts.txt").unlink()
    for x in (out / "composite", p, v2):
        shutil.rmtree(x)
    env |= {"COMPOSE_HANDOFF": "1", "COMPOSE_OFF_NODE": "1"}
    assert run().returncode == 0
    assert (out / "verdicts.txt").read_text().splitlines()[-1] == (
        f"COMPOSITE: composed on {host}, not the served node the-served-node: from its compose hand-off (verifier-side "
        "files, no witness) and the cases written and checked here from it, not from the parties that produced them")
    (out / "verdicts.txt").unlink()
    for x in (out / "composite", p, v2):
        shutil.rmtree(x)
    del env["SERVED_NODE"], env["COMPOSE_OFF_NODE"]
    assert run().returncode == 1
    assert (out / "verdicts.txt").read_text().splitlines() == [
        f"COMPOSITE refused: this is the served node {host}, which the prover controls, so a composite here claims no "
        "record; served_verdicts.sh composes it off the node from its compose hand-off (COMPOSE_OFF_NODE=0 composes it "
        "here as a development check)"]
    assert not (out / "composite").exists() and not p.exists()


#: onsite's own protocols: the piece's private directory, its public summary, its Lean, and a directory that isn't its
ONSITE_OWN = {"verity/pous/p2/v4/run": ("pous/w0", "pous/pous.json", "pous-run-protocol", "pous/w1"),
              "verity/wipe/v0": ("wipe", "wipe/summary.json", "wipe-protocol", "earlier/wipe")}
#: A stand-in for a product's Lean executable: it reads nothing it is handed and rejects under its own name, so the
#: VERDICT line it leaves says nothing of what the product's Lean answers
NO_LEAN = '''import json, sys
from pathlib import Path
print(json.dumps({"protocol": f"verity/{Path(sys.argv[0]).name}/v0", "reason": "stand-in: no Lean ran",
                  "verdict": "reject"}))
sys.exit(1)
'''


@pytest.mark.parametrize("protocol", list(ONSITE_OWN))
@pytest.mark.parametrize("killed", [True, False])
def test_the_composite_leaves_out_a_pous_or_wipe_case_that_is_not_its_pieces_and_composes_the_rest(tmp_path, protocol,
                                                                                                    killed):
    """served_composite.sh with onsite_pieces.sh's `onsite_cases` and composite.py as they are, a stand-in lake, and a
    stand-in for each product's Lean, which reads nothing and rejects under its own name (`NO_LEAN`); the serve dir,
    named through a link, holds a case of each of onsite's subprotocols, and SERVED_NODE names another host than this
    one, where the composite is composed and not refused. A piece killed once it wrote its case and its
    answer and before it published its summary (PoUS's pous.json, the wipe's summary.json): the protocol is
    NOT_WRITTEN and its directory LEFT_OUT; the case composite.py makes for onsite's Lean names every other
    subprotocol and not that one, in its params and in its evidence; OUT/verdicts.txt and onsite's composite.json name
    that protocol, and it alone, as missing (NOT_ESTABLISHED), before the Lean's verdict line; remote, which composes
    neither protocol, is handed its three cases and misses none; and the exit is 0, each product's Lean having
    answered. A case of the protocol in another directory beside the piece's own (a second window's; the wipe's left
    by an earlier run): the piece's own is composed, the other is not, and nothing is missing. No path reaches
    OUT/verdicts.txt. What onsite's Lean answers on a case that lacks a subprotocol is not shown here, where its Lean
    is the stand-in."""
    import composite as C
    tree, bin_, d, out = tmp_path / "tree", tmp_path / "bin", tmp_path / "link", tmp_path / "out"
    (tmp_path / "serve").mkdir()
    d.symlink_to(tmp_path / "serve", target_is_directory=True)
    (tmp_path / "lean").write_text(f"#!{sys.executable}\n{NO_LEAN}")
    (tmp_path / "lean").chmod(0o755)
    _script(bin_ / "lake", f'mkdir -p .lake/build/bin && cp {tmp_path}/lean ".lake/build/bin/$2"')
    _script(bin_ / "systemd-run", "exit 1")
    for pkg in ("protocols/remote/verifier", "protocols/onsite/verifier"):
        (tree / "verity" / pkg).mkdir(parents=True)
    (tree / "benchmarks" / "one_stage").mkdir(parents=True)
    shutil.copy(HERE.parent / "composite.py", tree / "benchmarks" / "one_stage")

    def case(where: str, of: str) -> None:
        (d / "private" / where).mkdir(parents=True)
        (d / "private" / where / "case.json").write_text(json.dumps(
            {"commitment": "c0", "evidence": "beef", "params": {"in": where}}))
        (d / "private" / where / "answer.json").write_text(json.dumps({"protocol": of, "verdict": "accept"}))

    own = {p: ONSITE_OWN[p][0] if p in ONSITE_OWN else f"piece{n}" for n, p in enumerate(C.COMPONENTS["onsite"])}
    for p, where in own.items():
        case(where, p)
    for _, summary, lean, _ in ONSITE_OWN.values():
        (out / "pieces" / summary).parent.mkdir(parents=True)
        (out / "pieces" / summary).write_text(json.dumps(
            {"window": 0, "verdict": "accept", "decided_by": {"program": lean, "host": "the-served-node"}}))
    if killed:
        (out / "pieces" / ONSITE_OWN[protocol][1]).unlink()
    else:
        case(ONSITE_OWN[protocol][3], protocol)
    env = {"PATH": f"{bin_}:{os.defpath}", "HOME": str(tmp_path), "PY": sys.executable, "TPY": sys.executable,
           "ONSITE_PIECES": str(ONSITE_PIECES), "SERVED_NODE": "the-served-node"}
    r = subprocess.run(["bash", str(HERE.parent / "served_composite.sh"), str(d), str(out)], cwd=tree, env=env,
                       check=False, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    said = (out / "verdicts.txt").read_text()
    lines = {p: [f"COMPOSITE NOT_WRITTEN {p}" if killed and p == protocol else f"COMPOSITE CASE {p} accept"]
             for p in ONSITE_OWN}
    lines[protocol].append(f"COMPOSITE LEFT_OUT {protocol}: a case under the serve dir's private that is not its "
                           "piece's with its Lean's answer (onsite_cases.log on the node)")
    assert said.splitlines()[:-1] == [
        *lines["verity/pous/p2/v4/run"], *lines["verity/wipe/v0"],
        "COMPOSITE CASES: PoUS's and the wipe's as their pieces wrote them, none made here",
        "COMPOSITE remote: VERDICT verity/remote/v0 reject stand-in",
        *([f"COMPOSITE onsite: NOT_ESTABLISHED missing {protocol}"] if killed else []),
        "COMPOSITE onsite: VERDICT verity/onsite/v0 reject stand-in"]
    assert str(tmp_path) not in said
    for x in ("remote", "onsite"):
        want = [p for p in C.COMPONENTS[x] if not (killed and p == protocol)]
        made = d / "private" / "composite" / x / "private"
        assert [(c["id"], c["params"]) for c in json.loads((made / "params.json").read_text())["components"]] == [
            (p, {"in": own[p]}) for p in want]
        assert [c["id"] for c in json.loads((made / "evidence.json").read_text())["components"]] == want
        assert json.loads((out / "composite" / x / "composite.json").read_text())["missing"] == [
            p for p in C.COMPONENTS[x] if p not in want]


DIGEST = "5e" * 64


def _auditor():
    """Infra's gate, ``verity/composite/device/live_challenger/auditor.py``, as a module of its own."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("sv_auditor", HERE.parents[2] / "verity" / "composite" / "device" / "live_challenger" / "auditor.py")
    module = sys.modules.setdefault(spec.name, importlib.util.module_from_spec(spec))
    spec.loader.exec_module(module)
    return module


def _declared(tmp_path: Path, files: dict | None = None, live=(32, 2)) -> tuple[Path, Path, dict]:
    """A serve dir whose serve-openai declared a window Program of digest `DIGEST` at T 32, S 2 (serve.json and
    ``h1/window-program.meta.json``, its layout `served.declared_layout`'s over two linears at K 32 and 64), and a
    circuit-files directory for it whose manifest holds `files` (default the live file and k32's, not k64's)."""
    from verity_vllm.protocols.pouw.h1 import served as SV
    d, c = tmp_path / "serve", tmp_path / "circuits"
    (d / "h1").mkdir(parents=True)
    c.mkdir()
    layout = SV.declared_layout([(64, 32), (32, 64)], {0: "up", 1: "down"}, 32, 2)
    (d / "serve.json").write_text(json.dumps({"protocols": {"program": DIGEST, "rows": 32, "steps": 2,
                                                            "weights_root": "77" * 64}}))
    (d / S.H1_META).write_text(json.dumps({"program_digest": DIGEST, "rows": 32, "steps": 2, "layout": layout}))
    files = {"live": "11" * 64, "k32": "22" * 64} if files is None else files
    (c / S.CIRCUITS_MANIFEST).write_text(json.dumps({"program": DIGEST, "live_bind": {"T": live[0], "S": live[1]},
                                                     "files": files}))
    return d, c, layout


def test_the_period_opens_on_the_served_programs_digest_and_files_as_cores_credit_reads_them(tmp_path, monkeypatch):
    """``period open``, through infra's own gate (`auditor.gate` and `auditor.period`, as lane vllm's key): its open
    commits the Program digest serve-openai wrote, its T, the sample K, the fixed length and the circuit files, each read
    back by core's credit as the window's period commits it; a stratum with no ``k<K>`` file yet is named and left out.
    It names no weights without ``--model`` (the gate refuses a root its registry lacks). ``period close`` closes it."""
    from verity.protocols.pouw.verifier import credit as CR
    from verity.commitments.crypto import lms
    from verity_vllm.protocols.pouw.h1 import challenger as PHC

    A = _auditor()
    (tmp_path / "var" / "sessions").mkdir(parents=True)
    (tmp_path / "peers").write_text("vy-dc-h200 research@10.0.0.2\n")
    (tmp_path / "pins.jsonl").write_text(A.pin_line("c" * 40, "d" * 40, "2026-10-27T09:00:00Z") + "\n")
    host = A.Host(root=tmp_path / "var", peers=tmp_path / "peers", pins=tmp_path / "pins.jsonl")
    signer = A.key_signer(host, lms.Params(64, 4, 5), workers=1)
    now = datetime(2026, 10, 27, 19, 0, tzinfo=UTC)
    hooks = dict(now=now, token="ab12", own=lambda d: None, start=lambda s: None, active=lambda s: False,
                 secret=bytes(32), ready=lambda s: None,
                 period=lambda args: A.period(host, args, signer=signer, pinned="c" * 40, now=now, active=lambda s: False))
    sent = []

    def gate_out(gate: str, *words: str) -> bytes:
        sent.append(words)
        rc, out = A.gate(["--ssh", shlex.join(words), "--key", "vllm"], host, **hooks)
        if rc:
            raise RuntimeError(f"the gate refused {words[0]} (exit {rc}): {out.decode()}")
        return out

    monkeypatch.setattr(S, "_gate_out", gate_out)
    d, c, layout = _declared(tmp_path)
    doc = S.period_open(d, GATE, "served-r1", c, slots=4, max_tokens=8)
    assert doc["missing"] == [64] and doc["weights"] == [] and doc["draw_k"] == PHC.draw_k(layout)
    assert "--weights-root" not in sent[0] and "--model" not in sent[0]
    ledger = (host.root / "windows.jsonl").read_text().splitlines()
    assert CR.committed_k(ledger, "served-r1") == (PHC.draw_k(layout), None)
    assert CR.committed_rows(ledger, "served-r1") == (32, None)
    assert CR.committed_files(ledger, "served-r1") == ({"live": "11" * 64, "k32": "22" * 64}, None)
    assert CR.committed_length(ledger, "served-r1") == ([8] * 4, None)
    assert json.loads(ledger[-1])["body"]["circuit"] == DIGEST
    assert json.loads((d / S.PERIOD).read_text())["open"]["body"]["period"] == "served-r1"
    argv, _ = S.period_open_argv(d, "served-r2", c, slots=4, max_tokens=8, draw_k=3, model="LLAMA32_1B_INT8")
    assert argv[argv.index("--draw-k") + 1] == "3" and argv[argv.index("--model") + 1] == "LLAMA32_1B_INT8"
    S.period_close(d, GATE)
    kinds = [json.loads(x)["kind"] for x in (host.root / "windows.jsonl").read_text().splitlines()]
    assert kinds[-1] == "period-close" and json.loads((d / S.PERIOD).read_text())["close"]["body"]["windows"] == []


def test_a_period_opens_before_serving_and_holds_the_served_program_to_it(tmp_path, monkeypatch, capsys):
    """Before serve-openai starts (no serve.json; its startup reserves window 0's session, which joins only an open
    period) the open goes out on the offline Program (`program`'s); ``period served`` then holds the Program
    serve-openai declared to the period's, refusing another, whose windows credit nothing."""
    d, c, _ = _declared(tmp_path)
    served = (d / "serve.json").read_text()
    (d / "serve.json").unlink()
    monkeypatch.setattr(S, "_gate_out", lambda gate, *words: json.dumps({"body": {"period": words[2]}}).encode())
    assert S.main(["period", "open", str(d), "--gate", GATE, "--name", "p", "--circuits", str(c), "--slots", "4",
                   "--max-tokens", "8"]) == 0
    assert json.loads((d / S.PERIOD).read_text())["circuit"] == DIGEST
    (d / "serve.json").write_text(served)
    assert S.main(["period", "served", str(d)]) == 0
    assert f"PERIOD served p: serve-openai declared its Program {DIGEST[:16]}" in capsys.readouterr().out
    (d / "serve.json").write_text(json.dumps({"protocols": {"program": "6f" * 64}}))
    assert S.main(["period", "served", str(d)]) == 1
    assert "its windows credit nothing" in capsys.readouterr().out


@pytest.mark.parametrize("case", ["other program", "other live T,S", "no live", "no meta", "no manifest"])
def test_a_period_open_is_refused_on_files_that_are_not_the_served_programs(tmp_path, case):
    """Circuit files of another Program, a live file for another T or S, none for the live template, a serve dir with no
    declared Program, or no circuit files: refused before anything is sent, naming the build to run."""
    d, c, _ = _declared(tmp_path, files={"k32": "22" * 64} if case == "no live" else None,
                        live=(64, 2) if case == "other live T,S" else (32, 2))
    if case == "other program":
        m = json.loads((c / S.CIRCUITS_MANIFEST).read_text())
        (c / S.CIRCUITS_MANIFEST).write_text(json.dumps({**m, "program": "6f" * 64}))
    if case == "no meta":
        (d / S.H1_META).unlink()
    if case == "no manifest":
        (c / S.CIRCUITS_MANIFEST).unlink()
    why = {"other program": "not the served", "other live T,S": "not the window's 32, 2", "no live": "not the window's",
           "no meta": "has no declared window Program", "no manifest": "no circuit files at"}[case]
    with pytest.raises(ValueError, match=why):
        S.period_open_argv(d, "p", c, slots=4, max_tokens=8)


def test_the_program_is_built_off_the_server_as_serve_openai_declares_and_writes_it(tmp_path, monkeypatch):
    """``program`` declares the served model's linears (`DECLARED`) in vLLM's order and naming, ``(n, K)`` each, at T and
    S, and writes them with serve-openai's own `served.write_program`; ``--like`` takes model, T and S from a serve
    dir and refuses a digest other than the one that run declared. The declaration and its digest are stand-ins here
    (a real one takes minutes); the node job checks the real one against a recorded run's digest."""
    from verity_vllm.protocols.pouw import h1 as PH
    from verity_vllm.protocols.pouw.h1 import served as SV

    seen = []
    monkeypatch.setattr(SV, "declared", lambda shapes, modules, rows, steps, unit=None: ("program", shapes, modules, rows,
                                                                                           steps))
    monkeypatch.setattr(SV, "declared_layout", lambda shapes, modules, rows, steps: {"tile": [32, 32]})

    def write(prog, layout, out, *, rows, steps):
        seen.append((prog, layout, out, rows, steps))
        out.mkdir(parents=True, exist_ok=True)
        (out / PH.PROGRAM_META).write_text(json.dumps({"program_digest": DIGEST, "rows": rows, "steps": steps,
                                                       "bytes": 9, "layout": layout}))
        return DIGEST
    monkeypatch.setattr(SV, "write_program", write)
    meta = S.program(tmp_path / "h1", "verity-int8/Llama-3.2-1B-INT8", 512, 16)
    (_tag, shapes, modules, rows, steps), _layout, out, *_ = seen[0]
    assert meta["program_digest"] == DIGEST and out == tmp_path / "h1" and (rows, steps) == (512, 16)
    assert len(shapes) == 64 and shapes[:4] == [(3072, 2048), (2048, 2048), (16384, 2048), (2048, 8192)]
    assert modules[0] == "model.layers.0.self_attn.qkv_proj" and modules[63] == "model.layers.15.mlp.down_proj"

    run = tmp_path / "serve"
    run.mkdir()
    (run / "serve.json").write_text(json.dumps({"served_model_name": "verity-int8/Llama-3.2-1B-INT8",
                                                "protocols": {"program": DIGEST, "rows": 32, "steps": 2}}))
    assert S.main(["program", str(tmp_path / "like"), "--like", str(run)]) == 0 and seen[-1][3:] == (32, 2)
    (run / "serve.json").write_text(json.dumps({"served_model_name": "verity-int8/Llama-3.2-1B-INT8",
                                                "protocols": {"program": "6f" * 64, "rows": 32, "steps": 2}}))
    with pytest.raises(ValueError, match="DECLARED's linears aren't the ones that run served"):
        S.program(tmp_path / "like", "x", 0, 0, like=run)
    with pytest.raises(ValueError, match="no declared linears for 'other'"):
        S.program(tmp_path / "other", "other", 512, 16)


def _llama(torch, model: str):
    """`DECLARED`'s model as serve-openai's H1 arm finds it: its int8 linears on the meta device under vLLM's module names,
    each on a stand-in of the int8_exact plain arm."""
    from types import SimpleNamespace

    class Linear(torch.nn.Module):
        def __init__(self, n, k):
            super().__init__()
            self.weight = torch.empty(n, k, dtype=torch.int8, device="meta")
            self.weight_scale = torch.empty(n, 1, device="meta")
            self.quant_method = SimpleNamespace(int8_arm="plain", observe=lambda *a: None)

    layers, linears = S.DECLARED[model]
    root = torch.nn.Module()
    root.model = torch.nn.Module()
    root.model.layers = torch.nn.ModuleList()
    for _ in range(layers):
        layer = torch.nn.Module()
        for name, n, k in linears:
            parent, leaf = name.split(".")
            if not hasattr(layer, parent):
                layer.add_module(parent, torch.nn.Module())
            getattr(layer, parent).add_module(leaf, Linear(n, k))
        root.model.layers.append(layer)
    return root, Linear


def test_the_offline_program_is_the_one_serve_openai_declares_under_the_same_protocols_config(tmp_path, monkeypatch):
    """``program --protocols-config C`` declares what serve-openai's H1 arm (`serve_h1.H1Windows`) declares under
    ``--protocols-config C`` over the served model: under ``{"pouw": {"unit": "j10r", "ks": [2048]}}`` the same linears
    at K 2,048 alone, named alike, over the same J10-R tile unit (`served.unit_of`), at the same T and S, so ``period
    served`` finds the Program the period opened on; ``--like`` takes C from the run (serve.json's ``protocols.config``).
    Unset on both sides it is v1's over every linear, another Program, and ``--like`` refuses a run whose Program is not
    the one its recorded config declares. The declaration is a stand-in that records what it was given (a real one takes
    minutes)."""
    from types import SimpleNamespace

    torch = pytest.importorskip("torch")
    from verity.protocols.pouw.prover.circuit import h1tile as HT, h1window as HW
    from verity_vllm.pipeline import serve_h1 as SH
    from verity_vllm.protocols import pouw
    from verity_vllm.protocols.pouw import h1 as PH
    from verity_vllm.protocols.pouw.h1 import served as SV

    model, j10r = "verity-int8/Llama-3.2-1B-INT8", {"pouw": {"unit": "j10r", "ks": [2048]}}
    llama, linear = _llama(torch, model)
    monkeypatch.setattr(pouw, "linear_types", lambda: (linear,))
    monkeypatch.setattr(SV, "weights_of", lambda layers: None)
    monkeypatch.setattr(SV, "register_weights", lambda *a, **k: SimpleNamespace(root=lambda: bytes(64)))
    monkeypatch.setattr(SV, "declared", lambda shapes, modules, rows, steps, unit=None: (
        [tuple(s) for s in shapes], dict(modules), rows, steps,
        None if unit is None else (unit.func, unit.args, unit.keywords)))
    programs = []

    def write(prog, layout, out, *, rows, steps):
        if (prog, layout) not in programs:
            programs.append((prog, layout))
        digest = f"{programs.index((prog, layout)):0128x}"
        out.mkdir(parents=True, exist_ok=True)
        (out / PH.PROGRAM_META).write_text(json.dumps({"program_digest": digest, "rows": rows, "steps": steps, "bytes": 0}))
        return digest
    monkeypatch.setattr(SV, "write_program", write)

    def served(config):
        arm = SH.H1Windows(llama, None, 512, 16, engine=object(), out=tmp_path / f"serve{len(programs)}", config=config)
        run = tmp_path / f"run{len(programs)}"
        run.mkdir()
        (run / "serve.json").write_text(json.dumps({"served_model_name": model, "protocols": arm.describe()}))
        return arm, run

    arm, run = served(j10r)
    meta = S.program(tmp_path / "offline", model, 512, 16, config=json.dumps(j10r))
    assert meta["program_digest"] == arm.digest and len(programs) == 1
    shapes, modules, _r, _s, unit = programs[0][0]
    assert len(shapes) == 48 and {k for _n, k in shapes} == {2048} and modules[0] == "model.layers.0.self_attn.qkv_proj"
    assert unit[0] is HW.tile_unit and unit[2]["definition"] is HT.PouwH1Tile5
    assert S.main(["program", str(tmp_path / "like"), "--like", str(run)]) == 0 and len(programs) == 1
    assert S.main(["program", str(tmp_path / "flag"), "--rows", "512", "--steps", "16", "--protocols-config",
                   json.dumps(j10r)]) == 0 and len(programs) == 1

    v1, v1_run = served(None)
    assert S.program(tmp_path / "v1", model, 512, 16)["program_digest"] == v1.digest != arm.digest
    assert len(programs[1][0][0]) == 64 and programs[1][0][4] is None
    (v1_run / "serve.json").write_text(json.dumps({"served_model_name": model,
                                                   "protocols": {**v1.describe(), "program": arm.digest}}))
    with pytest.raises(ValueError, match="DECLARED's linears aren't the ones that run served"):
        S.program(tmp_path / "mixed", model, 0, 0, like=v1_run)
    with pytest.raises(ValueError, match=r"no linear at K \[4096\]"):
        S.program(tmp_path / "none", model, 512, 16, config=json.dumps({"pouw": {"ks": [4096]}}))



def _serve_argv(tmp_path: Path, env: dict[str, str]) -> tuple[int, list[str]]:
    """`served_verdicts.sh serve` in a toy tree whose venv python records serve-openai's argv and exits before listening,
    under a no-op `sleep`: the exit status and that argv."""
    tree, bin_, venv = tmp_path / "tree", tmp_path / "bin", tmp_path / "venv" / "bin"
    for d in (tree / "adapters" / "vllm", bin_, venv):
        d.mkdir(parents=True)
    python = (f"printf '%s\\n' \"$*\" >> {tmp_path}/calls; case \"$*\" in *prove_path.py\\ class-circuit*) "
              f"while [ $# -gt 0 ]; do [ \"$1\" = --out ] && printf '{{}}\\n' > \"$2\"; shift; done; exit 0;; "
              f"*prove_path.py\\ serve-phi*|*served_verdicts.py\\ pearlplus-period*) exit 0;; esac; "
              f"printf '%s\\n' \"$@\" > {tmp_path}/argv; exit 1")
    for path, body in ((venv / "python", python), (bin_ / "sleep", "exit 0")):
        path.write_text(f"#!/bin/sh\n{body}\n")
        path.chmod(0o755)
    env = {"PATH": f"{bin_}:{os.defpath}", "HOME": str(tmp_path), "VENV": str(venv.parent), "D": str(tmp_path / "serve"),
           "OUT": str(tmp_path / "out"), **env}
    rc = subprocess.run(["bash", str(HERE.parent / "served_verdicts.sh"), "serve"], cwd=tree, env=env, check=False,
                        capture_output=True).returncode
    return rc, (tmp_path / "argv").read_text().splitlines()


@pytest.mark.parametrize("env, budget", [
    ({"H1_GATE": ""}, "512"),
    ({"H1_GATE": GATE}, "512"),
    ({"H1_GATE": GATE, "H1_T": "8192"}, "8192"),
    ({"H1_GATE": "", "H1_T": "8192"}, "8192"),
    ({"T": "2048", "H1_T": "8192", "H1_GATE": GATE}, "2048"),
])
def test_the_engine_gets_one_token_budget_with_or_without_h1(tmp_path, env, budget):
    """serve-openai gets ``--max-num-batched-tokens`` once, T (default H1_T, else 512), whether H1 serves or not (H1_GATE
    empty turns H1 off); without it the engine would take serve-openai's own 2048 with H1 off."""
    rc, argv = _serve_argv(tmp_path, env)
    assert rc == 10 and argv[:3] == ["-m", "verity_vllm.pipeline.cli", "serve-openai"], (rc, argv)
    assert argv.count("--max-num-batched-tokens") == 1 and argv[argv.index("--max-num-batched-tokens") + 1] == budget
    assert ("--protocols" in argv) == bool(env["H1_GATE"]) and "--h1-window-commit" not in argv


DAY_PROGRAM = "5d" * 64


@pytest.mark.parametrize("env, flags", [
    ({}, []),
    ({"DAY": "3", "DAY_LOG": "/l.json", "DAY_PROGRAM": DAY_PROGRAM},
     ["--day", "3", "--day-log", "/l.json", "--day-program", DAY_PROGRAM]),
    ({"DAY": "3", "DAY_LOG": "/l.json", "DAY_PROGRAM": DAY_PROGRAM, "DAY_BLOCKS": "a,b", "DAY_PARAMS": "/p.json"},
     ["--day", "3", "--day-log", "/l.json", "--day-program", DAY_PROGRAM, "--day-blocks", "a,b", "--day-params", "/p.json"]),
], ids=["off", "day", "day-blocks-params"])
def test_serve_openai_declares_the_day_only_under_day(tmp_path, env, flags):
    """DAY unset (the default) gives serve-openai no day option, so it writes what it wrote before; DAY gives it the day,
    the log, the program and, where set, the blocks and the log's parameters."""
    rc, argv = _serve_argv(tmp_path, {"H1_GATE": "", **env})
    assert rc == 10 and [x for x in argv if x.startswith("--day")] == flags[::2], (rc, argv)
    assert all(argv[argv.index(f) + 1] == v for f, v in zip(flags[::2], flags[1::2]))


class _Response:
    """`urlopen`'s response, read whole or line by line; reading line i sets ``now[0]`` to ``times[i]``, and reading past the
    last line to ``end``."""

    def __init__(self, lines: list[bytes], now: list[float] | None = None, times: list[float] = (), end: float = 0.0) -> None:
        self.lines, self.now, self.times, self.end = lines, now, list(times), end

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def read(self) -> bytes:
        return b"".join(self.lines)

    def __iter__(self):
        for i, line in enumerate(self.lines):
            if self.now is not None:
                self.now[0] = self.times[i]
            yield line
        if self.now is not None:
            self.now[0] = self.end


def _sse(*events: dict | bytes) -> list[bytes]:
    """serve-openai's stream of ``events``, as `urlopen`'s response yields it: each event's data line, then its blank line."""
    return [x for e in events for x in (b"data: " + (e if isinstance(e, bytes) else json.dumps(e).encode()) + b"\n", b"\n")]


def test_serve_load_times_a_request_at_its_send_its_first_output_id_and_its_end(monkeypatch):
    """A streamed request's first token is the first event carrying output ids, not the stream's opening event, and its end
    comes once the stream is read past [DONE]; its ids and finish are what they were. An unstreamed one has no first token."""
    now = [1.0]
    lines = _sse({"choices": [{"text": "", "token_ids": None}]}, {"choices": [{"token_ids": [7]}]},
                 {"choices": [{"token_ids": [8], "finish_reason": "length"}]}, b"[DONE]")
    stream = _Response(lines, now, times=[2, 2, 3, 3, 4, 4, 5, 5], end=6.0)
    monkeypatch.setattr(SL.urllib.request, "urlopen", lambda req, timeout=None: stream)
    times: dict = {}
    assert SL.post(0, {"stream": True}, times=times, clock=lambda: now[0]) == {"token_ids": [7, 8], "finish_reason": "length"}
    assert times == {"send": 1.0, "first_token": 3.0, "done": 6.0}
    whole = _Response([json.dumps({"choices": [{"token_ids": [7]}]}).encode()])
    monkeypatch.setattr(SL.urllib.request, "urlopen", lambda req, timeout=None: whole)
    times = {}
    assert SL.post(0, {}, times=times, clock=lambda: now[0]) == {"choices": [{"token_ids": [7]}]} and set(times) == {"send", "done"}


def test_serve_load_varied_writes_each_requests_timings_beside_its_record(tmp_path, monkeypatch):
    """A varied run's timings file beside its record: its release run and wall-clock start, and one entry a request in
    release order, its times on the injected clock from the run's start (send, then first token, then done) and its
    streamed ids counted; the record's lines are what they were."""
    ticks = itertools.count()

    def clock() -> float:
        return float(next(ticks))

    def urlopen(req, timeout=None):
        b = json.loads(req.data)
        if req.full_url.endswith("/tokenize"):
            return _Response([json.dumps({"tokens": [128000] + [1000 + ord(c) for c in b["prompt"]]}).encode()])
        ids = list(range(500, 500 + b["max_tokens"]))
        return _Response(_sse({"choices": [{"text": "", "token_ids": None}]},
                              *({"choices": [{"token_ids": [t], "finish_reason": "length" if t == ids[-1] else None}]} for t in ids),
                              b"[DONE]"))

    monkeypatch.setattr(SL.urllib.request, "urlopen", urlopen)
    out = tmp_path / "load.jsonl"
    a = argparse.Namespace(port=0, model="m", varied=3, fixed=0, rate=2.0, prompt_range="16:24", max_tokens=4, seed=1,
                           release="run", jitter_ms=0.0, out=str(out))
    assert SL.run_varied(a, clock=clock, wall=lambda: 1.7e9, sleep=lambda s: None) == 0
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert SL.timings_path(out) == tmp_path / "load.timings.json"
    doc = json.loads(SL.timings_path(out).read_text())
    assert set(doc) == {"clock", "release_run", "start_unix", "requests"} and (doc["release_run"], doc["start_unix"]) == ("run", 1.7e9)
    assert [set(r) for r in rows] == [{"user", "release", "at_s", "delay_s", "prompt", "max_tokens", "response"}] * 3
    assert [(t["user"], t["release"], t["at_s"]) for t in doc["requests"]] == [(r["user"], r["release"], r["at_s"]) for r in rows]
    assert [r["release"] for r in rows] == [0, 1, 2]
    for t, r in zip(doc["requests"], rows):
        assert set(t) == {"user", "release", "at_s", "send_s", "first_token_s", "done_s", "tokens", "ok"} and t["ok"]
        assert 0 < t["send_s"] < t["first_token_s"] < t["done_s"]
        assert r["response"] == {"token_ids": list(range(500, 500 + r["max_tokens"])), "finish_reason": "length"}
        assert t["tokens"] == r["max_tokens"]
