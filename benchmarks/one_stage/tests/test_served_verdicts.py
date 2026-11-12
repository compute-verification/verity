"""`served_verdicts.py`'s window pick, its subprotocols' pieces and its record, on toy serve dirs and answers:
the window it audits holds the most requests and none carried across windows; a subprotocol with no piece is named missing
with the command it lacks and given no verdict, one that refused to run is missing, and one that ran out of its limit or
wrote no answer errs; and sampled proofs' accepted answer composes, with the others missing, to an undecided block record that
`block_record.check` holds and its schema admits, while a refusal or an audit that wrote nothing composes to reject with no
block record. The block record's verdict is the Lean check's answer, so with none here it is undecided (the window's
pending) even when every subprotocol accepted on its node; the record names each one missing, rejecting, malformed or
about another server. The wipe composes only when its answer accepts on the record's server. Remote's
product runs no onsite piece and composes sampled proofs and PoUW alone, the others dropped. PoUW's
development piece checks the window's gate session and credits nothing, and the gate is refused while the lane has a period
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
of V3c's audit's memory. There
step 6 defers the audit and the composite, leaving the follow-on's settings and its every input, listed and hashed, and
nothing in $OUT but a line; the follow-on holds the copy to those sums, refuses another setting, a node short of the
memory or another tree, and audits, records about the served node and composes, each saying it composed elsewhere, as
the record says where it was composed and what was deferred. What it publishes holds a load's tokens and span by
path, leaving the inference-only policy, its digest and its verdict line as written. Its load, `serve_load.py
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
import subprocess
import sys
import tarfile
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path

import jsonschema
import pytest
from verity.protocols.pous.verifier import wipe_period as W
from verity.protocols.spot.verifier.one_stage import registration as REG

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


def _digest(doc: dict) -> str:
    """A policy's digest as its verdict cites it: SHA-512 of its sorted, compact JSON."""
    return hashlib.sha512(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def test_publish_holds_the_loads_tokens_and_span_by_path_and_leaves_the_policy_and_its_digest_as_written(tmp_path):
    """The load's tokens over a block's outputs are its exact block count, so `publish` holds each load's ``tokens`` and
    ``span_s`` (`OWN_AT`, by path) with its counts, keeping its rates; an inference-only policy's ``n_inputs.span_s``, a
    ``tokens`` under no load and the verdict citing the policy's digest stay byte for byte, and the digest is the same
    before and after."""
    out, private = tmp_path / "out", tmp_path / "serve" / "private" / "out"
    (out / "serve").mkdir(parents=True)
    (out / "serve" / "overhead.json").write_text(json.dumps(AGGREGATE) + "\n")
    (out / "inference-only-policy.json").write_text(json.dumps(POLICY, indent=1, sort_keys=True) + "\n")
    before = _digest(json.loads((out / "inference-only-policy.json").read_text()))
    verdict = {"verdict": "accept", "reason": None, "complies": True, "failure": None,
               "policy": {"sha512": before, "written_utc": POLICY["written_utc"]}, "c": POLICY["c"], "k": POLICY["k"],
               "n": POLICY["n"], "classes": [["cd" * 32, 1_000, 4_000]], "slots": 4, "positions": 35}
    (out / "record").mkdir()
    (out / "record" / "inference-only.json").write_text(json.dumps(verdict, indent=1, sort_keys=True) + "\n")
    (out / "pieces" / "pearlplus").mkdir(parents=True)
    (out / "pieces" / "pearlplus" / "load.json").write_text(json.dumps({"tokens": 1024, "span_s": 9.5, "served_tokens": 64}))
    kept = {p: p.read_bytes() for p in out.rglob("*.json") if p.name != "overhead.json"}
    assert S.publish(out, private) == ["serve/overhead.json"]
    (ld,) = json.loads((out / "serve" / "overhead.json").read_text())["loads"]
    assert ld == {"time_per_output_token_s": 0.15625, "tokens_per_s": 6.4, "client_tokens_per_s": 6.1}
    assert json.loads((private / "serve" / "overhead.json").read_text()) == AGGREGATE
    assert {p: p.read_bytes() for p in kept} == kept
    after = json.loads((out / "inference-only-policy.json").read_text())
    assert after["n_inputs"]["span_s"] == 3600 and _digest(after) == before == _digest(POLICY)
    assert json.loads((out / "record" / "inference-only.json").read_text())["policy"]["sha512"] == before
    assert S.public_json(POLICY) == POLICY and S.public_json(verdict) == verdict
    assert S.publish(out, private) == []


def test_a_published_line_holds_a_loads_tokens_and_span_and_leaves_the_inference_only_lines(tmp_path):
    """A verdict line's published form has no load's ``tokens`` or ``span_s`` in its JSON, in a JSON string's text or in a
    JSON object cut short, and the INFERENCE-ONLY lines, the policy's (its span in seconds) and the verdict's (the policy's
    digest), are their own published form."""
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
    i, digest = POLICY["n_inputs"], _digest(POLICY)
    policy_line = (f"INFERENCE-ONLY POLICY at class 4,32,4: c {POLICY['c']} gates, k {POLICY['k']} fresh bits, n {POLICY['n']} "
                   f"sessions (⌈{i['utilization'][0]}/{i['utilization'][1]} x {i['rate']['macs_per_s']} MAC/s x {i['span_s']} s / "
                   f"{i['work']['theta_macs']} θ-MACs⌉ = {i['raw']}, on the grid); session {POLICY['session']['definition'][:16]}…, "
                   f"policy {digest[:16]}…")
    verdict_line = (f"INFERENCE-ONLY accept: session {'cd' * 8}… (the registered one), 1000 gates and 4000 fresh bits a block, "
                    f"against c {POLICY['c']}, k {POLICY['k']}, n {POLICY['n']} (no count published: n is judged on the blocks' "
                    "exact count, held on the node)")
    pub = {"verdict": "accept", "c": POLICY["c"], "k": POLICY["k"], "n": POLICY["n"], "policy": {"sha512": digest}}
    for line in (policy_line, verdict_line, f"INFERENCE-ONLY {json.dumps(pub)}"):
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


def test_an_accepted_certifier_answer_says_what_its_trusted_host_premise_rests_on(tmp_path):
    """The network certifier's answer as its piece writes it when the warden ran on the gate's host and its checks, samples
    and coins on the served node: its accept's clause says the premise is assumed and why, so the ``RECORD`` line's
    accept isn't read as the auditor's devices having placed it."""
    premise = ("assumed, the auditor's devices not placed: the network certifier ran on 204.12.178.84, not the prover's "
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


#: an accepted core answer of each piece's subprotocol, its public outputs the block record's leg certificate
ACCEPTS = {
    "pouw": {"protocol": "verity/pouw/v1", "public": {"work": {"level": 3, "n": 4, "credited_tiles": 2, "credited": 8192}},
             "profile": {"delta": "1/1000000"}},
    "bandwidth": {"protocol": "verity/remote-bandwidth/v0", "public": {"eps": "1/1000000", "q_bytes": 6963200}, "profile": {}},
    "pous": {"protocol": "verity/pous/p2/v4/run", "public": {"certified": True, "eps": "1/1000000", "s_bits": 1 << 30},
             "profile": {}},
    "certifier": {"protocol": "verity/certifier/v0", "public": {"K": 15764, "eps": "1/100", "windows": 1}, "profile": {}},
    "wipe": {k: WIPE_ACCEPT[k] for k in ("protocol", "public", "profile")},
}
COMPONENTS = ("pouw", "bandwidth", "pous", "certifier", "wipe")


@pytest.mark.parametrize("case, piece", [("accept", None)] + [
    (case, piece) for case in ("missing", "reject", "elsewhere", "malformed") for piece in COMPONENTS])
def test_the_record_is_pending_without_a_lean_answer_and_names_each_subprotocol_that_does_not_accept(
        tmp_path, monkeypatch, capsys, case, piece):
    """Every piece's answer accepted on the record's node (each piece's `RAN`, which a `SERVED` naming no server leaves it
    about, and the wipe's `SERVED`): the composed verdict is ``pending``, the block record's ``undecided``, since its
    verdict is the Lean check's answer and none ran, with nothing missing, and its reason names each subprotocol's
    accept. One piece with no answer leaves it ``pending`` and names that piece missing with
    why; one that rejects, or accepts with outputs that are not its leg's certificate, makes it ``reject`` naming it, with
    no block record; one about another server leaves it ``pending``, carried and named as elsewhere."""
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
    if case in ("reject", "malformed"):
        assert rc == 1 and (line["composed"], line["by"], line["block_record"]) == ("reject", [name], None)
        assert line["verdicts"][name] == ("reject" if case == "reject" else "error")
        assert not (tmp_path / "record" / "block-record.json").exists()
        report = json.loads((tmp_path / "record" / "verdicts.json").read_text())
        row = next(r for k, r in report["protocols"].items() if S.NAMES[k] == name)
        assert case == "reject" or row["reason"].startswith(f"its answer is not the block record's {leg} leg: KeyError")
        return
    rec = json.loads((tmp_path / "record" / "block-record.json").read_text())
    jsonschema.Draft202012Validator(SCHEMA).validate(rec)
    assert rc == 0 and line["check"] == {} == BR.check(rec)
    if case == "accept":
        assert (line["composed"], line["record_missing"], line["missing"], line["elsewhere"]) == ("pending", [], {}, {})
        assert (rec["composed"]["verdict"], rec["composed"]["answer"]) == ("undecided", None)
        assert rec["composed"]["legs"] == {k: "accept" for k in BR.LEGS}
        late = {"the network certifier": " (its quantity is not established at δ = 1/1000000: the run established it "
                                         "except with 1/100)"}
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
    wipe's answer. The report and the line say they were composed on the served node, from the answer files it wrote, and
    the published line keeps that."""
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
    monkeypatch.setattr(S.socket, "gethostname", lambda: "vy-auditor-2")
    assert S.main(argv) == 0
    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1].removeprefix("RECORD "))
    assert line["composed_on"] == {"host": "vy-auditor-2", "served_node": False, "from": S.COPIED_FROM}
    assert S.main([*argv, "--product", "onsite"]) == 1
    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1].removeprefix("RECORD "))
    assert (line["product"], line["composed"], line["by"]) == ("onsite", "reject", ["the wipe"])


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


def _gate(calls: list):
    """The gate's answers as `served_verdicts._gate_out` gets them: the session's tree for ``fetch``, the ledger and pins."""
    def out(gate: str, *words: str) -> bytes:
        calls.append((gate, words))
        if words[0] == "fetch":
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w:gz") as t:
                for name, body in ((f"{words[1]}/request.json", b"{}"), (f"{words[1]}/out/record.json", b"{}")):
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


FAKE_PROVE_PATH = '''import json, sys
a = sys.argv[1:]
assert a[0] == "window-circuit"
c = a[a.index("--class") + 1] if "--class" in a else "record"
answer = json.load(open("answers.json"))[c]
if answer != "count":
    print("prove_path.py window-circuit: refused at", c, file=sys.stderr)
    sys.exit({"refuse": 2, "expire": 124}[answer])
open(a[a.index("--out") + 1], "w").write("{}")
print(json.dumps({"bits": 772, "count": {"free": 3, "inputs": 10, "registered": 7, "units": 1}}))
'''


def _circuits(tmp_path: Path, classes: str, answers: dict) -> tuple[int, list[str], list[str]]:
    """`served_verdicts.sh circuits 0` in a toy tree whose prove_path.py counts a class, refuses it, or exits 124 as
    `timeout` does when a count runs out of time (``answers``, by class), under a `timeout` that records the limit it got:
    the exit status, the verdict lines, and the limits in order."""
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
           "D": str(tmp_path / "serve"), "OUT": str(tmp_path / "out"), "CLASSES": classes}
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
if a[0] == "window-circuit":
    out.write_text("{{}}")
    print(json.dumps({{"bits": 772, "count": {{"free": 3, "inputs": 10, "registered": 7, "units": 1}}}}))
else:
    assert a[0] == "window-commit"
    if os.environ.get("FAIL_COMMIT"):
        print("prove_path.py window-commit: the row dir holds no window 7", file=sys.stderr)
        sys.exit(2)
    out.mkdir(parents=True)
    (out / "circuit.json").write_text(json.dumps({{"run": "ab"}}))
    if "--window-registration" in a:
        tail = a[a.index("--window-registration"):]
        open(os.environ["ORDER"] + ".commit", "w").write("\\n".join(tail))
        (out / "registration.json").write_text(json.dumps({{"format": "toy", "after": json.load(open(tail[1]))}}))
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
    assert ("--tie" in a) == ("--tie-prefix" in a) and a[a.index("--challenger") + 1] == "local"
    open(os.environ["ORDER"] + ".audit", "w").write(json.dumps(a))
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
         records_audit: str = FAKE_RECORDS_AUDIT, **extra: str | None) -> tuple[int, list[str], list[str], list[str]]:
    """`served_verdicts.sh v3c` (or ``args``) on window 7 of a toy serve dir at the certifier's class, in a toy tree whose
    prove_path.py (``prove_path``) and records_audit.py (``records_audit``) note each step they run, its builds stubbed,
    main's first-parent line a toy MAIN_FILE and the split certifier's two commands set (``extra`` adds to the
    environment, None unsetting): the exit status, the verdict lines, the steps in order and the limits."""
    tree, bin_, venv, d = tmp_path / "tree", tmp_path / "bin", tmp_path / "venv" / "bin", tmp_path / "serve"
    for p in (tree / "benchmarks" / "prove_path", tree / ".venv" / "bin", tree / "verity" / "protocols" / "flock" / "verifier",
              tree / "verity" / "protocols" / "spot" / "verifier" / "lean", tree / "verity" / "protocols" / "flock" / "pod", bin_, venv,
              d / "advice" / "w7" / "private", tmp_path / "row"):
        p.mkdir(parents=True)
    (tree / "benchmarks" / "prove_path" / "prove_path.py").write_text(prove_path)
    (tree / "benchmarks" / "prove_path" / "records_audit.py").write_text(records_audit)
    (tree / "verity" / "protocols" / "flock" / "pod" / "60-circuit.sh").write_text("exit 0\n")
    for f in (d / "advice" / "w7" / "private" / "advice.bin", tmp_path / "row" / "manifest.json",
              tmp_path / "row" / "build_summary.json", tmp_path / "h.so", tmp_path / "n.so"):
        f.write_text("")
    (d / "weights_registration.json").write_text("{}")
    _write(d, "windows.jsonl", [{"window": 7, "run_root": "ab" * 64}])
    (tmp_path / "main.txt").write_text(TOY_MAIN)
    # the toy tree's verity/ directories, on the PYTHONPATH the script sets and as `python -`'s working directory, would
    # shadow the installed verity
    py = f'unset PYTHONPATH\nexport PYTHONSAFEPATH=1\n[ "$1 $2" = "-m research" ] && exit 0\nexec {sys.executable} "$@"'
    for path, body in ((venv / "python", py), (tree / ".venv" / "bin" / "python", py),
                       (bin_ / "timeout", f'[ "$1" = -k ] && shift 2\necho "$1" >> {tmp_path}/limits; shift; exec "$@"'),
                       (bin_ / "gpu-lease", "exit 99"), (bin_ / "uv", "exit 0"), (bin_ / "lake", "exit 0")):
        _script(path, body)
    env = {"PATH": f"{bin_}:{os.defpath}", "HOME": str(tmp_path), "VENV": str(venv.parent), "D": str(d), "K": "7",
           "OUT": str(tmp_path / "out"), "SERVE_ROOT": str(tmp_path / "root"), "FLOCK_WORK": str(tmp_path / "fw"),
           "ROW": str(tmp_path / "row"), "HIDDEN_SO": str(tmp_path / "h.so"), "NORM_TAP_SO": str(tmp_path / "n.so"),
           "ORDER": str(tmp_path / "order"), "H1_GATE": "", "V3C_CLASS": W63K4, "CLASSES": W63K4,
           "CERTIFIER_CHAIN_CMD": CHAIN, "CERTIFIER_PERIOD_CMD": PERIOD, "NODE_MEM_GB": "512",
           "MAIN_FILE": str(tmp_path / "main.txt")}
    env = {k: v for k, v in {**env, **extra}.items() if v is not None}
    rc = subprocess.run(["bash", str(HERE.parent / "served_verdicts.sh"), *args], cwd=tree, env=env, check=False,
                        capture_output=True).returncode
    read = lambda f: (tmp_path / f).read_text().split() if (tmp_path / f).exists() else []  # noqa: E731
    return rc, (tmp_path / "out" / "verdicts.txt").read_text().splitlines(), read("order"), read("limits")


def test_v3c_runs_the_certifiers_chain_beside_its_audit_and_its_period_check_on_the_period_json_after(tmp_path):
    """The window commit, then the split certifier's chain (it reads the commit's scan state) beside V3c's audit and its
    period.json, and the certifier's period check on it once both are done, each within its own limit; the check gets
    the serve dir, the window, the certifier's piece dir and V3c's dir, and writes the answer `pieces` reads there. The
    two steps' logs, which count the window's requests, end in the serve dir's private/pieces/certifier, not in $OUT,
    which leaves the node. $OUT then names no window and counts none of its own (the window commit's population, the
    audit's, the chain's window): each line is whole in the serve dir's private/verdicts.txt, each file in its private/out."""
    rc, lines, order, limits = _v3c(tmp_path)
    assert rc == 0, lines
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
    circuit, its answer, and H1's log naming W and, unless ``bound`` is None, the record H1 bound (``same``: R; ``other``:
    another)."""
    d = tmp_path / "serve"
    vw = d / f"v3c-w7-{W63K4.replace(',', '-')}" / "window"
    vw.mkdir(parents=True)
    (d / "h1" / "w7").mkdir(parents=True)
    r = {"format": "toy", "after": W_OF_7, **({"law": law} if law else {})}
    (vw / "registration.json").write_text(json.dumps(r))
    (vw / "circuit.json").write_text(json.dumps({"run": "ab"}))
    (d / "h1" / "w7" / "window-commit.json").write_text(json.dumps({"run": "ab", "circuit": "cd", "population": 37}))
    log = [{"window": 7, "kind": "registration", "file": "w7/registration.json", "digest": "aa" * 64}]
    if bound is not None:
        log.append({"window": 7, "kind": "record", "file": "w7/record.json",
                    "digest": REG.digest(r if bound == "same" else {**r, "after": None})})
    _write(d / "h1", "log.jsonl", log)


def test_under_h1_tie_v3c_audits_the_commit_h1_bound_and_makes_none_of_its_own(tmp_path):
    """The window's one commit is the one its close made: V3c holds its registration to the record H1 bound (the log's
    digest) and its circuit to the verifier's, runs no window-commit, and goes on to its audit."""
    _tied_commit(tmp_path, "same")
    rc, lines, order, _limits = _v3c(tmp_path, H1_TIE="1")
    assert rc == 0 and "window-commit" not in order and order[0] == "window-circuit" and "audit" in order, (lines, order)
    assert lines[1].startswith(f"WINDOW-COMMIT class {W63K4} at H1's close, its registration the Commit record H1 bound "
                               "before the sample coin (R ") and '"run": "ab"' in lines[1] and "population" not in lines[1]
    assert lines[2] == (f"WINDOW-CIRCUIT-HELD class {W63K4}: the committed circuit.json less its run fields is the "
                        "verifier's own window-circuit record: PASS")


def _audited(tmp_path: Path) -> dict[str, str | None]:
    """The sampling design's, the tie's and the ports' flags the toy records_audit.py's audit got."""
    a = json.loads((tmp_path / "order.audit").read_text())
    return {f: a[a.index(f) + 1] if f in a else None for f in ("--law", "--tie", "--tie-prefix", "--ports")}


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
    """records_audit samples only on the auditor's challenger's coins, so V3c's audit takes a local development session's
    (--challenger local), pinned on main's first-parent line: MAIN_FILE's, else MAIN_SHA's in the tree, kept as
    $OUT/v3c-audit/main.txt, the --main the audit reads."""
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
V3C_COINS = " V3c's audit: a local development challenger session where it runs;"


@pytest.mark.parametrize("extra, coins, v3c", [
    ({}, LOCAL_COINS, V3C_COINS),
    ({"PP_CHALLENGER": "gate:research@auditor", "PP_PEER": "vy-dc-h200"},
     "the challenger gate:research@auditor (peer vy-dc-h200)", V3C_COINS),
    ({"V3C_CLASS": "", "CERTIFIER_CHAIN_CMD": None, "CERTIFIER_PERIOD_CMD": None}, LOCAL_COINS, ""),
])
def test_the_coins_line_says_whose_coins_each_sample_is_on(tmp_path, extra, coins, v3c):
    """Before step 0 the run says whose coins each sample is on: sampled proofs' (run.sh's local development session
    unless PP_CHALLENGER names a challenger), PoUW H1's, and V3c's audit's when it runs. Here step 0's `predict row` then
    fails, and the run stops there."""
    venv = tmp_path / "toy-venv" / "bin"
    _script(venv / "python", '[ "$1 $2" = "-m research" ] && exit 0\n[ "$1 $2" = "-m verity_vllm.pipeline.cli" ] && exit 9\n'
                             f'unset PYTHONPATH\nexport PYTHONSAFEPATH=1\nexec {sys.executable} "$@"')
    rc, lines, _order, _limits = _v3c(tmp_path, (), VENV=str(venv.parent), HOST="vy-dc-h200",
                                      PATH=f"{_hosts(tmp_path)}:{tmp_path / 'bin'}:{os.defpath}", **extra)
    assert rc == 17 and lines[-1] == "STOP at predict (exit 17)", lines
    assert f"COINS sampled proofs: {coins}; PoUW H1: not served (H1_GATE empty);{v3c} the bandwidth audit's: " \
           "vy-auditor-2's bw_party, inside PoUS's piece" in lines, lines


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
DEFERRED_READS = ("weights_registration.json", f"window-circuit-w7-{VTAG}.json", "windows.jsonl", "seals.jsonl",
                  "requests.jsonl", "h1/log.jsonl", "h1/w7/registration.json", "h1/w7/record.json",
                  "h1/w7/window-commit.json", "sampled-proofs-served/summary.json", "sampled-proofs-served/window/audit.json",
                  "sampled-proofs-served/windows/rank0/registration.json", "sampled-proofs-served/runs.json",
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


def _deferred(tmp_path: Path, **extra: str) -> subprocess.CompletedProcess:
    """Window 7 of a toy serve dir under H1's tie: its commit as the tie left it, with a value file, H1's records, the
    record's and the composite's inputs, files no step after the serve reads (a row dir, a key, an audit session), and
    the pieces the record read in $OUT; then step 6's deferral (`served_verdicts.sh defer WHY`) on vy-dc-h200, a node short
    of V3c's audit's memory, at the tree of commit SHA (``extra`` adds to the environment)."""
    d, out = tmp_path / "serve", tmp_path / "served-out"
    _tied_commit(tmp_path, "same")
    vw = d / f"v3c-w7-{VTAG}" / "window"
    files = {"h1/w7/registration.json": W_OF_7, "h1/w7/record.json": json.loads((vw / "registration.json").read_text()),
             f"window-circuit-w7-{VTAG}.json": {}, "weights_registration.json": {}, "seals.jsonl": {"window": 7},
             "requests.jsonl": {"windows": [7]}, "sampled-proofs-served/summary.json": {"audit": {}},
             "sampled-proofs-served/window/audit.json": {"accepted": True},
             "sampled-proofs-served/windows/rank0/registration.json": {"format": "toy"},
             "sampled-proofs-served/runs.json": ["r0"], "private/pouw/period/case.json": {"evidence": "beef"},
             "private/pouw/period/answer.json": {"protocol": "verity/pouw/period/v1"},
             "windows/w7/rows.bin": {}, "private/weights.key": {}, "sampled-proofs-served/window/sessions/0.json": {}}
    for f, body in files.items():
        (d / f).parent.mkdir(parents=True, exist_ok=True)
        (d / f).write_text(json.dumps(body))
    _write(d, "windows.jsonl", [{"window": 7, "run_root": "ab" * 64}])
    (vw / "values").mkdir()
    (vw / "values" / "a.bin").write_bytes(b"\x01")
    (out / "pieces" / "pouw").mkdir(parents=True)
    (out / "pieces" / "pouw" / "answer.json").write_text(json.dumps({"verdict": "accept"}))
    (tmp_path / "served-tree").mkdir()
    env = {"PATH": f"{_hosts(tmp_path)}:{os.defpath}", "HOME": str(tmp_path), "VENV": str(tmp_path / "served-venv"),
           "D": str(d), "K": "7", "OUT": str(out), "SERVE_ROOT": str(tmp_path / "root"), "V3C_CLASS": W63K4,
           "PRODUCT": "remote", "H1_TIE": "1", "HOST": "vy-dc-h200", "RESEARCH_SOURCE_SHA": SHA,
           "GIT_CEILING_DIRECTORIES": str(tmp_path), **extra}
    return subprocess.run(["bash", str(HERE.parent / "served_verdicts.sh"), "defer", SHORT], cwd=tmp_path / "served-tree",
                          env=env, check=False, capture_output=True, text=True)


def test_a_node_short_of_v3cs_audits_memory_defers_it_and_the_composite_listing_and_hashing_each_input(tmp_path):
    """Step 6's deferral leaves in the serve dir's private/deferred the pieces the record read, the settings the
    follow-on runs at (the window, the product, V3c's class and law, H1_TIE, the composite, the served node and the tree's
    commit), and the follow-on's every input, listed for the prover's copy and hashed for the follow-on to hold it to:
    V3c's window dir whole, H1's log and each record it names, the tie's commit answer, the record's and the composite's
    inputs and its own files, and nothing else of the serve dir. Its one line in $OUT names what was deferred, why and the
    follow-on's command, and no window; nothing else lands in $OUT."""
    run = _deferred(tmp_path)
    assert run.returncode == 0, run.stderr
    d, dd, out = tmp_path / "serve", tmp_path / "serve" / "private" / "deferred", tmp_path / "served-out"
    names = "K PRODUCT V3C_CLASS V3C_LAW H1_TIE COMPOSITE ALPHA GAMMA SERVED_NODE DEFERRED_TREE"
    said = subprocess.run(["bash", "-c", f'. "$1"; for v in {names}; do printf "%s=%s\\n" "$v" "${{!v}}"; done', "_",
                           str(dd / "deferred.env")], check=True, capture_output=True, text=True).stdout
    assert dict(x.split("=", 1) for x in said.splitlines()) == {
        "K": "7", "PRODUCT": "remote", "V3C_CLASS": W63K4, "V3C_LAW": "stratified:1", "H1_TIE": "1", "COMPOSITE": "1",
        "ALPHA": "", "GAMMA": "", "SERVED_NODE": "vy-dc-h200", "DEFERRED_TREE": SHA}
    listed = (dd / "inputs.txt").read_text().splitlines()
    window = [f"v3c-w7-{VTAG}/window/{f}" for f in ("circuit.json", "registration.json", "values/a.bin")]
    own = ["private/deferred/deferred.env", "private/deferred/pieces/pouw/answer.json"]
    assert listed == [*sorted([*DEFERRED_READS, *window, *own]), "private/deferred/inputs.sha256"]
    sums = {f: h for h, f in (x.split("  ", 1) for x in (dd / "inputs.sha256").read_text().splitlines())}
    assert sums == {f: hashlib.sha256((d / f).read_bytes()).hexdigest() for f in listed[:-1]}
    assert (dd / "pieces" / "pouw" / "answer.json").read_text() == (out / "pieces" / "pouw" / "answer.json").read_text()
    assert (out / "verdicts.txt").read_text().splitlines() == [
        f"DEFERRED V3c's audit and the composite: {SHORT}. A follow-on on a node with the memory runs them and composes the "
        "final record there, from the serve dir's audit inputs copied prover-side, outside any run dir, each file its "
        f"private/deferred/inputs.txt lists: D={d} bash benchmarks/one_stage/served_verdicts.sh v3c"]
    assert sorted(f.relative_to(out).as_posix() for f in out.rglob("*") if f.is_file()) == [
        "pieces/pouw/answer.json", "verdicts.txt"]
    assert not _window_data(out, 7) and oct(dd.stat().st_mode & 0o777) == "0o700"


def _follow(tmp_path: Path, **extra: str | None) -> tuple[int, list[str], list[str], list[str]]:
    """The follow-on of `_deferred`'s deferral: `served_verdicts.sh v3c` with D alone (no K, class or certifier) on
    cpu-6, a node with V3c's audit's memory, at the served node's tree, in `_v3c`'s toy tree with a record and a
    composite that note their arguments (``extra`` adds to the environment, None unsetting): `_v3c`'s outcome."""
    _script(tmp_path / "tree" / "benchmarks" / "one_stage" / "served_composite.sh", FAKE_SERVED_COMPOSITE)
    (tmp_path / "tree" / "benchmarks" / "one_stage" / "served_verdicts.py").write_text(FAKE_RECORD)
    env = {"K": None, "V3C_CLASS": None, "CERTIFIER_CHAIN_CMD": None, "CERTIFIER_PERIOD_CMD": None,
           "PATH": f"{_hosts(tmp_path)}:{tmp_path / 'bin'}:{os.defpath}", "HOST": "cpu-6", "NODE_MEM_GB": "755",
           "RESEARCH_SOURCE_SHA": SHA, "GIT_CEILING_DIRECTORIES": str(tmp_path)}
    return _v3c(tmp_path, **{**env, **extra})


@pytest.mark.parametrize("extra, tree", [
    ({}, "(this tree's)"),
    ({"RESEARCH_SOURCE_SHA": "cd" * 20, "FOLLOW_ANY_TREE": "1"}, f"(this tree is {'cd' * 20}: FOLLOW_ANY_TREE)"),
])
def test_the_follow_on_holds_the_copy_to_the_served_nodes_sums_then_audits_and_composes_the_final_record_there(
        tmp_path, extra, tree):
    """The follow-on, at the settings the deferral recorded: the copy held to the served node's sums, then no window
    circuit or commit of its own (the served node's record at the class, its tree's, and the commit H1 bound, held as
    there), V3c's audit and period, the final record about the served node from the pieces it read, and the composite,
    told the served node, so each says it composed elsewhere. Another tree runs only under FOLLOW_ANY_TREE, which its
    line names. $OUT names no window."""
    assert _deferred(tmp_path).returncode == 0
    rc, lines, order, limits = _follow(tmp_path, **extra)
    d, out = tmp_path / "serve", tmp_path / "out"
    assert rc == 0 and order == ["audit", "period", "record", "composite"], (lines, order)
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
    assert json.loads((tmp_path / "order.record").read_text()) == [
        "record", "--serve-run", str(d), "--window", "7", "--sampled-proofs", str(d / "sampled-proofs-served"), "--pieces",
        str(d / "private" / "deferred" / "pieces"), "--out", str(out / "record"), "--product", "remote", "--node",
        "vy-dc-h200"]
    assert (tmp_path / "order.composite").read_text().split() == ["vy-dc-h200", str(d), str(out)]
    assert not _window_data(out, 7), _window_data(out, 7)


FOLLOW_FAIL = ("FOLLOW-ON on cpu-6, 755 GB: V3c's audit and the composite, which the served node vy-dc-h200 deferred, "
               "from its serve dir's audit inputs copied here prover-side, each as it hashed them (the serve dir's "
               "private/deferred/inputs.sha256): FAIL (the serve dir's private/deferred/check.log)")
CLASH = ("PREFLIGHT: the follow-on runs at the deferral's settings (the serve dir's private/deferred/deferred.env), and "
         "the environment sets another")


@pytest.mark.parametrize("change, extra, line", [
    ("values/a.bin", {}, FOLLOW_FAIL),
    ("case.json", {}, FOLLOW_FAIL),
    (None, {"K": "8"}, f"{CLASH} K: unset it"),
    (None, {"PRODUCT": "onsite", "COMPOSITE": ""}, f"{CLASH} PRODUCT COMPOSITE: unset it"),
    (None, {"NODE_MEM_GB": "196"}, f"PREFLIGHT: the follow-on runs the V3c audit the served node deferred, and {SHORT}"),
    (None, {"RESEARCH_SOURCE_SHA": "cd" * 20}, "PREFLIGHT: the follow-on reads the served node's window-circuit record, "
     f"which its tree {SHA} wrote, and this tree is {'cd' * 20}: run it at that commit, or set FOLLOW_ANY_TREE=1"),
])
def test_a_follow_on_refuses_a_copy_unlike_the_served_nodes_another_setting_a_node_short_of_the_memory_or_another_tree(
        tmp_path, change, extra, line):
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
    assert (rc, lines, order) == (3, [line], []), lines


ONLY = "PREFLIGHT_ONLY: nothing kept, built or served"
DEFERS = "so step 6 defers V3c's audit and the composite to a follow-on"


@pytest.mark.parametrize("extra, said", [
    ({}, "512 GB: the serve's lease 60 min, capped at 128 GB"),
    (TIES, "512 GB: the serve's lease 150 min, capped at 160 GB"),
    ({**TIES, "LEASE_MEM_GB": "64"}, "512 GB: the serve's lease 150 min, capped at 64 GB"),
    ({**TIES, "NODE_MEM_GB": "196", "PRODUCT": "remote", "CERTIFIER_CHAIN_CMD": None, "CERTIFIER_PERIOD_CMD": None},
     f"196 GB: the serve's lease 150 min, capped at 160 GB; {SHORT}, {DEFERS}"),
])
def test_preflight_only_says_what_the_run_would_do_and_stops_before_anything_is_kept_built_or_served(
        tmp_path, extra, said):
    """PREFLIGHT_ONLY=1 runs the preflight and stops there, exit 0, with one line: the serve's lease (under H1_TIE 150
    minutes capped at 160 GB, the serve's 128 and window-commit's peak, within a 196 GB node) and, on a node short of
    V3c's audit's memory, its deferral. Nothing runs, and neither the serve root nor C-Flock's work dir is made."""
    rc, lines, order, limits = _v3c(tmp_path, (), PATH=f"{_hosts(tmp_path)}:{tmp_path / 'bin'}:{os.defpath}",
                                    HOST="vy-dc-h200-2", PREFLIGHT_ONLY="1", **extra)
    assert (rc, lines, order, limits) == (0, [f"PREFLIGHT passed on vy-dc-h200-2, {said}. {ONLY}"], [], [])
    assert not (tmp_path / "root").exists() and not (tmp_path / "fw").exists()


def test_preflight_only_on_a_follow_on_names_the_deferral_it_would_finish(tmp_path):
    """A follow-on under PREFLIGHT_ONLY passes its own preflight (the deferral's settings, the audit's memory, the served
    node's tree) and stops before its copy's check or V3c's audit."""
    assert _deferred(tmp_path).returncode == 0
    rc, lines, order, _limits = _follow(tmp_path, PREFLIGHT_ONLY="1")
    assert (rc, lines, order) == (0, [f"PREFLIGHT passed on cpu-6, 755 GB: the follow-on of vy-dc-h200's deferral. {ONLY}"],
                                  [])


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
    case, but not under H1's tie (a ``SERVE_DIR/h1/log.jsonl``), whose cases are on the record H1 named; and under it no
    product is handed a case under ``SERVE_DIR/private/pouw``, whose period case reaches a product only tied to that
    record, while an untied serve dir's goes to remote as its own. Where SERVED_NODE names another host (a deferral's follow-on), it says it composed
    there, from a prover-side copy of the serve dir's audit inputs."""
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
    env = {"PATH": f"{bin_}:{os.defpath}", "HOME": str(tmp_path), "PY": sys.executable, "TPY": sys.executable}
    run = lambda: subprocess.run(["bash", str(HERE.parent / "served_composite.sh"), str(d), str(out)], cwd=tree, env=env,  # noqa: E731
                                 check=False, capture_output=True, text=True)
    assert run().returncode == 1
    assert (out / "verdicts.txt").read_text() == ("COMPOSITE: not run, onsite's own cases come from ONSITE_PIECES's file, "
                                                  "and it is unset\n")
    (out / "verdicts.txt").unlink()
    env["ONSITE_PIECES"] = str(ONSITE_PIECES)
    r = run()
    assert r.returncode == 0, r.stderr
    host = subprocess.run(["hostname", "-s"], check=True, capture_output=True, text=True).stdout.strip()
    on = (f"COMPOSITE: composed on {host}, which holds the serve dir, from the cases the served node wrote there, not "
          "from the parties that produced them")
    assert (out / "verdicts.txt").read_text().splitlines() == [
        "COMPOSITE CASE verity/pous/p2/v4/run accept", "COMPOSITE NOT_WRITTEN verity/wipe/v0",
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
    assert (out / "verdicts.txt").read_text() == "COMPOSITE: not run, this tree has no benchmarks/one_stage/composite.py\n"
    (tree / "benchmarks" / "one_stage" / "composite.py").write_text(FAKE_COMPOSITE)
    (out / "verdicts.txt").unlink()
    shutil.rmtree(d / "private")
    env["PRODUCT"] = "remote"
    r = run()
    assert r.returncode == 0, r.stderr
    assert (out / "verdicts.txt").read_text().splitlines() == [
        "COMPOSITE CASE verity/sampled-proofs/one-stage/v2 accept", "COMPOSITE CASE verity/registration/v0 accept",
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
    assert json.loads((p / "remote" / "private" / "argv.json").read_text()) == [str(piece), str(v2)]
    (out / "verdicts.txt").unlink()
    for x in (out / "composite", p, v2):
        shutil.rmtree(x)
    env["SERVED_NODE"] = "the-served-node"
    assert run().returncode == 0
    assert (out / "verdicts.txt").read_text().splitlines()[-1] == (
        f"COMPOSITE: composed on {host}, not the served node the-served-node: from a prover-side copy of its serve dir's "
        "audit inputs and the cases written here from it, not from the parties that produced them")


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
    named through a link, holds a case of each of onsite's subprotocols. A piece killed once it wrote its case and its
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
           "ONSITE_PIECES": str(ONSITE_PIECES)}
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
    """Infra's gate, ``verity/protocols/spot/device/live_challenger/auditor.py``, as a module of its own."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("sv_auditor", HERE.parents[2] / "verity" / "protocols" / "spot" / "device" / "live_challenger" / "auditor.py")
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
    for path, body in ((venv / "python", f"printf '%s\\n' \"$@\" > {tmp_path}/argv; exit 1"), (bin_ / "sleep", "exit 0")):
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
