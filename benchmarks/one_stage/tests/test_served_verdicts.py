"""`served_verdicts.py`'s window pick, its subprotocols' pieces and its record, on toy serve dirs and answers:
the window it audits holds the most requests and none carried across windows; a subprotocol with no piece is named missing
with the command it lacks and given no verdict, one that refused to run is missing, and one that ran out of its limit or
wrote no answer errs; and sampled proofs' accepted answer composes, with the others missing, to a pending block record that
`block_record.check` holds and its schema admits, while a refusal or an audit that wrote nothing composes to reject with no
block record. The record accepts only when every subprotocol accepted on its node, and names each one missing, rejecting,
malformed or about another server. An accepted network certifier's clause names the one link K counts and every other
link of the node uncounted and uncertified; one whose certifier ran on the node names no link. The wipe composes only
when its answer accepts on the record's server. Remote's product runs no onsite piece and composes sampled proofs and
PoUW alone, the others dropped. PoUW's development piece checks the window's gate session and credits
nothing, and the gate is refused while the lane has a period
open. The period the served windows join opens, through infra's own gate, on the Program serve-openai declared and
the circuit files built for it, as core's credit reads them, and is refused on files of anything else.
`served_verdicts.sh`'s window circuits gate on every class of CLASSES, and the class of record's count after them
gates nothing; its endpoint gets one token budget, T, with or without H1. Its preflight refuses a flag the tree's certifier
front doesn't take, a front whose endpoint isn't on the network certifier's id form, and a front load with no front; its
composite step publishes each case's and each product's verdict lines and nothing else. Its V3c step, sampled proofs'
audit of record, draws on a challenger's coins with no advice opening, runs the split network certifier's chain after the
window commit and its period check on V3c's period.json, and fails where no period.json is written; its preflight refuses
the split beside the certifier's whole piece, and a run without V3c. A V3c audit's row commits to the registration it is
given. Under H1_TIE its serve gives
serve-openai the tie's command, which, run as `serve_tie` runs it at an H1 window's close, writes the window's advice and
commits it chained after H1's W, its registration the window's record R, and writes no R when the advice is pending or the
commit refused; its V3c then audits the commit H1 bound and makes none, and has none for a window H1 bound no R for or bound
another, its audit tied to W at stratified:1 on a port for each proof block of R's sample; its preflight refuses H1_TIE
without H1's prover state, without V3c, on a tree whose window-commit takes no --window-registration or whose records_audit
takes no --tie, at a sampling design other than stratified:K, past the sample session's time, or for onsite on a node short
of V3c's audit's memory. There
step 6 defers the audit and the composite, leaving the follow-on's settings and its every input, listed and hashed, and
nothing in $OUT but a line; the follow-on holds the copy to those sums, refuses another setting, a node short of the
memory or another tree, and audits, records about the served node on V3c's audit and composes, each saying it composed
elsewhere, as the record says where it was composed and what was deferred, with sampled proofs missing on the served
node. Its load, `serve_load.py
--varied`, writes each request's send, first-token and done times beside its record, on an injected clock and a stand-in
endpoint. Its ADVICE-LINE holds a serve-openai ``--advice`` run's B to the compiler the verifier recomputes and every
closed window's advice line to after B's registration and before the first coin."""
from __future__ import annotations

import argparse
import contextlib
import fnmatch
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
import textwrap
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path

import jsonschema
import pytest
from verity.core.protocols.pous import wipe_period as W
from verity.core.protocols.sampled_proofs.one_stage import registration as REG

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import served_verdicts as S
from verity_e2e_demo import block_record as BR

SCHEMA = json.loads((Path(BR.__file__).parent / "schemas" / "block-record.v1.schema.json").read_text())
_SL = importlib.util.spec_from_file_location("serve_load", HERE.parents[2] / "examples" / "vllm" / "serve_load.py")
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


#: a window's advice registration as `verity-vllm advice` writes it, its advice padded to the cap ℓ_a = 2^23: 32,768 rows
REGISTRATION = {"schema": "verity-vllm/advice-registration/v1", "row_bits": 256, "rows": 32768, "window": 7,
                "binding": {"compiler": {"name": "c", "digest": "d"}, "advice": {"root": "ab" * 64, "bits": 8388608}},
                "commit": {"run_id": "r1/w7", "run_root": "ef" * 64},
                "compiler": {"name": "c", "digest": "d", "cap_bits": 8388608},
                "registered_roots": {"weights/w@16": "cd" * 64}, "requests": 3, "refused": {"q0": "2 of its 3 tokens"}}


def test_the_published_registration_is_the_padded_one_and_one_stating_the_advices_length_is_refused(tmp_path, capsys):
    """A window's advice registration is published as committed, its advice padded to its compiler's cap so its length and
    rows are ℓ_a's, less the count and refusals of its requests and its window's number. One that states the advice's own
    length (a length or a row count other than ℓ_a's, or its bits per egress window) is refused, by the command too, which
    prints nothing on stdout."""
    got = S.public_registration(REGISTRATION)
    assert got["binding"] == REGISTRATION["binding"] and got["rows"] == 32768 and S.padded(REGISTRATION)
    assert got["commit"] == {"run_id": "r1/wK", "run_root": "ef" * 64} and not {"requests", "compiled", "refused", "window"} & set(got)
    assert {k: got[k] for k in ("registered_roots", "compiler", "schema", "row_bits")} == {
        k: REGISTRATION[k] for k in ("registered_roots", "compiler", "schema", "row_bits")}
    (tmp_path / "r.json").write_text(json.dumps(REGISTRATION))
    assert S.main(["public-registration", str(tmp_path / "r.json")]) == 0
    assert json.loads(capsys.readouterr().out) == got
    advice = REGISTRATION["binding"]["advice"]
    for bad in ({**REGISTRATION, "binding": {**REGISTRATION["binding"], "advice": {**advice, "bits": 392}}},
                {**REGISTRATION, "rows": 2}, {**REGISTRATION, "bits_by_window": {"7": 392}},
                {k: v for k, v in REGISTRATION.items() if k != "rows"},
                {**REGISTRATION, "compiler": {**REGISTRATION["compiler"], "cap_bits": 4096}}):
        assert not S.padded(bad)
        with pytest.raises(ValueError, match="states the advice's own length"):
            S.public_registration(bad)
        (tmp_path / "r.json").write_text(json.dumps(bad))
        assert S.main(["public-registration", str(tmp_path / "r.json")]) == 1
        out = capsys.readouterr()
        assert out.out == "" and "wasn't committed padded to its compiler's cap" in out.err


B = {"name": "verity/prove-path/capped-compiler/v1", "digest": "b0" * 64, "cap_bits": 4096,
     "inner": {"name": "verity-vllm/i2-compile-steps/v0", "digest": "1a" * 64}}


def _advised(d: Path, *, cap: int = 4096, t0: float = 10.0, day: int = 3, declared: tuple | None = (3, B["digest"])) -> Path:
    """A serve-openai ``--advice`` serve dir of day ``day``: B registered at ``t0`` under a log whose builder is B's item at
    ``cap`` and whose last entry, given ``declared`` (a day and a program), is that declaration; windows 0-2 closed, 1
    empty, each with its advice line, 0's and 2's committed under B."""
    from verity.core.protocols import registration as REG

    item = REG.item_of({"kind": "builder", "digest": B["inner"]["digest"], "advice": "verity-vllm/window-advice/v1", "cap": cap})
    log = REG.chain("onsite", [REG.Information((item,)), *([REG.Declaration(*declared)] if declared else [])])
    d.mkdir(parents=True)
    (d / S.ADVICE_REGISTRATION).write_text(json.dumps({
        "compiler": B, "advice": "verity-vllm/window-advice/v1", "cap_bits": 4096, "row_bits": 256, "day": day, "t": t0,
        "registration": {"composite": "onsite", "head": REG.head("onsite", log), "log": json.loads(REG.committed("onsite", log))}}))
    _write(d, "windows.jsonl", [{"window": k} for k in range(3)])
    named = {"name": B["name"], "digest": B["digest"]}
    _write(d, S.ADVICE_LINES, [
        {"window": 0, "compiler": named, "advice": {"root": "a0" * 64, "bits": 4096}, "rows": 16, "t": 11.0},
        {"window": 1, "empty": True, "t": 12.0},
        {"window": 2, "compiler": named, "advice": {"root": "a2" * 64, "bits": 4096}, "rows": 16, "t": 13.0}])
    for k in (0, 2):
        (d / "advice" / f"w{k}").mkdir(parents=True)
        (d / "advice" / f"w{k}" / "registration.json").write_text(json.dumps(_window_registration(named, f"a{k}" * 64)))
    return d


def _window_registration(compiler: dict, root: str, bits: int = 4096, rows: int = 16) -> dict:
    """A window's advice registration under B, its advice committed padded to ℓ_a = 4096 bits (16 rows) unless ``bits`` and
    ``rows`` say its own length."""
    return {"binding": {"compiler": compiler, "advice": {"root": root, "bits": bits}}, "rows": rows, "row_bits": 256,
            "compiler": {**compiler, "cap_bits": 4096}}


def test_the_advice_line_holds_the_servers_b_and_each_windows_line_before_the_first_coin(tmp_path, capsys):
    """ADVICE-LINE passes on a run whose server registered B, the compiler the verifier recomputes, as its log's builder at
    ℓ_a with the run's day's declaration of B as the log's last entry, and wrote every closed window's line under B at ℓ_a
    and its rows after that and before the first coin, the audited window's registration its line's root, its advice padded
    to ℓ_a; it names no window and states no advice's own length. It fails, saying why, for another compiler, a log of
    another ℓ_a, a log that declares no day, another day or another program, a closed window with no line, a line of
    another length or row count, a line after the coins or before B, an audited window whose registration roots other
    advice, states the advice's own length or has no advice, and a run that registered no B."""
    ok, line = S.advice_line(_advised(tmp_path / "ok"), 2, B, coins_after=20.0)
    assert ok and line.endswith(": PASS") and S.public_line(line) == line and "4096 bits" in line
    assert "declared day 3's one program" in line
    assert S.main(["advice-line", str(tmp_path / "ok"), "--window", "2", "--compiler", str(_dump(tmp_path, B)),
                   "--coins-after", "20"]) == 0 and capsys.readouterr().out.strip() == line

    def fails(d: Path, window: int = 2, compiler: dict = B, coins_after: float = 20.0) -> str:
        ok, line = S.advice_line(d, window, compiler, coins_after)
        assert not ok and ": FAIL" in line and S.public_line(line) == line
        return line

    assert "isn't the compiler prove_path.py recomputes" in fails(_advised(tmp_path / "b"), compiler={**B, "digest": "00" * 64})
    assert "builder isn't B's" in fails(_advised(tmp_path / "cap", cap=8192))
    for k, declared in enumerate((None, (4, B["digest"]), (3, "00" * 64))):
        assert "isn't the run's day's declaration of B" in fails(_advised(tmp_path / f"day{k}", declared=declared))
    assert "isn't the run's day's declaration of B" in fails(_advised(tmp_path / "day-b", day=True, declared=(1, B["digest"])))
    assert "after sampled proofs' or the network certifier's first coin" in fails(_advised(tmp_path / "late"), coins_after=12.5)
    assert "before B's registration" in fails(_advised(tmp_path / "early", t0=11.5))
    assert "has no committed advice" in fails(_advised(tmp_path / "empty"), window=1)
    d = _advised(tmp_path / "gap")
    _write(d, S.ADVICE_LINES, [x for x in S._lines(d / S.ADVICE_LINES) if x["window"] != 0])
    assert "a closed window has no advice line" in fails(d)
    named = {"name": B["name"], "digest": B["digest"]}
    for k, (field, value) in enumerate((("rows", 2), ("advice", {"root": "a0" * 64, "bits": 392}))):
        d = _advised(tmp_path / f"line{k}")
        _write(d, S.ADVICE_LINES, [{**x, field: value} if x["window"] == 0 else x for x in S._lines(d / S.ADVICE_LINES)])
        assert "a length or row count other than ℓ_a's" in fails(d)
    d = _advised(tmp_path / "root")
    (d / "advice" / "w2" / "registration.json").write_text(json.dumps(_window_registration(named, "ff" * 64)))
    assert "isn't its advice line's root under B" in fails(d)
    for k, (bits, rows) in enumerate(((392, 16), (4096, 2), (392, 2))):
        d = _advised(tmp_path / f"own{k}")
        (d / "advice" / "w2" / "registration.json").write_text(json.dumps(_window_registration(named, "a2" * 64, bits, rows)))
        assert "wasn't committed padded to ℓ_a" in fails(d)
    (tmp_path / "none").mkdir()
    assert "the server registered no B" in fails(tmp_path / "none")


def _dump(d: Path, doc: dict) -> Path:
    (d / "doc.json").write_text(json.dumps(doc))
    return d / "doc.json"


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
    class, the scan class, the draw, a refusal's reason and the protocol's rounds stay. `public-lines` prints that form and
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


@pytest.mark.parametrize("scanned", [False, True])
def test_publish_holds_each_file_that_names_or_counts_a_window_and_leaves_its_public_form(tmp_path, capsys, scanned):
    """`publish`, once the run is done: a file whose path names a window is held whole in the serve dir's private/out, one
    whose text or JSON names or counts a window is held there in full with its public form left in out/, and the rest,
    the bench Build's requests by name and a window named ``all`` among them, stay byte for byte; a binary file, which no
    scan reads, is held whole. What a verifier reads (`RECEIVED`: the advice registration, the certifier's answer and
    case, and V3c's period audit) is held whole unless the kind scan found it clean (``--received-scanned``), and then
    keeps every key and value, the period's ``steps`` and the case's ``windows`` among them, only its text in public form.
    Then out/ names no window, counts none of a window's own but what a verifier reads, and a second pass holds
    nothing."""
    out, private = tmp_path / "out", tmp_path / "serve" / "private" / "out"
    case = {"params": {"windows": 4, "link": "certified"}, "commitment": "ab" * 8}
    files = {
        "verdicts.txt": "SERVE stopped (exit 0): 3 window(s) closed, 2 sealed\nCOMPOSITE CASE verity/pous/p2/v4/run accept\n",
        "record/verdicts.json": json.dumps({"window": 7, "node": "n", "composed": {"verdict": "pending", "reason": "window 7: no"}}),
        "record/block-record.json": json.dumps({"server_time": [{"server": "n", "steps": [13, 19]}], "composed": {"verdict": "pending"}}),
        "registration.json": json.dumps(REGISTRATION),
        "pieces/certifier/answer.json": json.dumps({"reason": "window 7: the scan's prog", "verdict": "reject"}),
        "pieces/certifier/case.json": json.dumps(case),
        "v3c-audit/period.json": json.dumps({"free": 0, "steps": 16}),
        "pieces/bandwidth/answer.json": json.dumps({"profile": {"windows": 6}, "verdict": "accept"}),
        "serve/period.json": json.dumps({"close": {"body": {"windows": [{"root": "ab"}]}}, "rows": 512, "steps": 16}),
        "pieces/pouw/check/check.log": "calls /srv/h1/w7/pouw-h1-calls.json\nPOUW {\"window\": 7, \"verdict\": \"accept\"}\n",
        "sampled-proofs/w7/summary.json": json.dumps({"verdict": "accept"}),
        "predicted.json": json.dumps({"requests": {"build_request_LP11_T8": "ab"}, "row": "r"}),
        "pieces/wipe/served.json": json.dumps({"window": "all", "server": "n"}),
    }
    if scanned:
        files["registration.json"] = json.dumps(S.public_registration(REGISTRATION))
    for rel, body in files.items():
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        (out / rel).write_text(body)
    (out / "serve" / "h1").mkdir()
    (out / "serve" / "h1" / "window-program.json.gz").write_bytes(gzip.compress(json.dumps({"rows": 512}).encode()))
    capsys.readouterr()
    assert S.main(["publish", str(out), "--private", str(private), *(["--received-scanned"] * scanned)]) == 0
    said = capsys.readouterr().out
    assert ("(RECEIVED) wasn't found clean by the kind scan: held whole" in said) == (not scanned)
    received = {"registration.json", "pieces/certifier/answer.json", "pieces/certifier/case.json", "v3c-audit/period.json"}
    assert {r for r in files if any(fnmatch.fnmatchcase(r, g) for g in S.RECEIVED)} == received
    whole = {"sampled-proofs/w7/summary.json", *(() if scanned else received)}
    kept = {"predicted.json", "pieces/wipe/served.json", *(received - {"pieces/certifier/answer.json"} if scanned else ())}
    assert _window_data(out, 7) == (['pieces/certifier/case.json: "windows":', 'v3c-audit/period.json: "steps": 1']
                                    if scanned else []), _window_data(out, 7)
    for rel, body in files.items():
        assert (out / rel).exists() == (rel not in whole), rel
        if rel in kept:
            assert (out / rel).read_text() == body and not (private / rel).exists(), rel
        else:
            assert (private / rel).read_text() == body, rel
    if scanned:
        assert json.loads((out / "pieces" / "certifier" / "answer.json").read_text()) == {
            "reason": "window K: the scan's prog", "verdict": "reject"}
    assert json.loads((out / "record" / "verdicts.json").read_text()) == {
        "node": "n", "composed": {"verdict": "pending", "reason": "window K: no"}}
    assert json.loads((out / "serve" / "period.json").read_text()) == {"close": {"body": {}}, "rows": 512}
    assert (out / "verdicts.txt").read_text() == ("SERVE stopped (exit 0): window(s) closed, sealed\n"
                                                  "COMPOSITE CASE verity/pous/p2/v4/run accept\n")
    assert gzip.decompress((private / "serve" / "h1" / "window-program.json.gz").read_bytes()) == b'{"rows": 512}'
    assert not (out / "serve" / "h1" / "window-program.json.gz").exists()
    assert S.publish(out, private, scanned) == []


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
    moved = ["chain/certs.json", "chain/report/report.json", "chain/n2/summary.json", "piece.log", "chain.log", "period.log"]
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
    assert (private / "chain/n2/summary.json").read_text() == "chain/n2/summary.json"
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


def test_publish_holds_whole_each_file_it_cant_scan_because_it_isnt_text_and_names_it_on_stderr(tmp_path, capsys):
    """A file that isn't UTF-8 text, which no scan reads, leaves out/: it is held whole in private at its path, listed
    among the held, and `publish` says so on stderr, with why."""
    out, private = tmp_path / "out", tmp_path / "private"
    (out / "serve").mkdir(parents=True)
    (out / "serve" / "program.json.gz").write_bytes(gzip.compress(b'{"rows": 512}'))
    assert S.publish(out, private) == ["serve/program.json.gz"]
    assert gzip.decompress((private / "serve" / "program.json.gz").read_bytes()) == b'{"rows": 512}'
    assert not (out / "serve" / "program.json.gz").exists()
    assert capsys.readouterr().err == "PUBLISH: serve/program.json.gz held whole, unscanned: not UTF-8 text\n"


def test_sampled_proofs_alone_composes_to_a_pending_block_record(tmp_path):
    report, rec, line = S.record(_rows(_accepted(), tmp_path), window=1, steps=(5, 11), node="vy-dc-h200", run="r1")
    jsonschema.Draft202012Validator(SCHEMA).validate(rec)
    assert BR.check(rec) == {}
    assert rec["legs"]["integrity"]["quantity"]["value"] == 2 and rec["server_time"] == [{"server": "vy-dc-h200", "steps": [5, 11]}]
    assert (line["composed"], line["record_missing"]) == ("pending", ["bandwidth", "certifier", "pous", "pouw", "wipe"])
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
    assert line["reason"] == report["composed"]["reason"] == rec["composed"]["onsite"]["reason"]
    assert line["reason"].startswith("sampled proofs: accept; ")


def test_an_accepted_certifier_answer_says_what_its_trusted_host_premise_rests_on(tmp_path):
    """The network certifier's answer as its piece writes it when the warden ran on the gate's host and its checks, draws
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


def test_an_accepted_certifier_answer_names_the_one_link_k_counts_and_a_local_gate_names_none(tmp_path):
    """The network certifier's placement as its piece writes it. On the gate's host, its ``links`` are the relayed link
    K counts and the gate's ssh link it doesn't: the accept's clause names the counted link and says every other link of
    the node is uncounted and uncertified, before the premise. Under ``--gate local`` it has no links, and the clause
    states the premise alone."""
    node, host = "vy-dc-h200", "204.12.178.84"
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
    (tmp_path / "registration.json").write_text(json.dumps({"window": {"circuit": "cd" * 64}}))
    row = S.sampled_proofs_row(tmp_path, None, tmp_path / "registration.json")
    assert row["verdict"] == "error" and "no summary" in row["reason"]
    _report, rec, line = S.record(_rows(row, tmp_path), window=1, steps=(5, 11), node="n", run=None)
    assert rec is None and (line["composed"], line["by"]) == ("reject", ["sampled proofs"])


def test_a_v3c_audit_refused_before_staging_commits_to_the_registration_it_is_given(tmp_path, monkeypatch):
    """V3c's audit dir holds no registration of its own until it stages its members; ``--registration`` (the window
    commit's) is the one its row commits to. A registration file that isn't there is an ``error`` row, and so is an
    accept whose audit record registers another window than ``--registration``."""
    from verity.core.protocols.sampled_proofs.one_stage import registration as R

    audit, reg = tmp_path / "v3c" / "audit", {"format": R.FORMAT, "window": {"circuit": "cd" * 64}}
    audit.mkdir(parents=True)
    (audit / "summary.json").write_text(json.dumps({"ok": False, "refused": {"records": "ValueError: no"}}))
    (tmp_path / "v3c" / "registration.json").write_text(json.dumps(reg))
    row = S.sampled_proofs_row(audit, None, tmp_path / "v3c" / "registration.json")
    assert (row["verdict"], row["commitment"]) == ("reject", R.digest(reg)) and "refused" in row["reason"]
    missing = S.sampled_proofs_row(audit, None, audit / "registration.json")
    assert missing["verdict"] == "error" and "registration.json" in missing["reason"] and "commitment" not in missing
    import served_integrity as SI

    for registered, want in ((reg, "accept"), ({**reg, "window": {"circuit": "ab" * 64}}, "error")):
        answered = {**_accepted()["answer"], "commitment": R.digest(registered)}
        monkeypatch.setattr(SI, "answer", lambda _d, _r, a=answered: a)
        row = S.sampled_proofs_row(audit, None, tmp_path / "v3c" / "registration.json")
        assert row["verdict"] == want and (want == "accept" or "commits to another registration" in row["reason"])
    seen = []
    monkeypatch.setattr(S, "sampled_proofs_row", lambda d, _run=None, r=None: seen.append((d, r)) or _accepted())
    (tmp_path / "serve").mkdir()
    S.main(["record", "--serve-run", str(_serve_dir(tmp_path / "serve")), "--window", "1", "--sampled-proofs", str(audit),
            "--registration", str(tmp_path / "v3c" / "registration.json"), "--pieces", str(tmp_path / "pieces"),
            "--out", str(tmp_path / "record")])
    assert seen == [(audit, tmp_path / "v3c" / "registration.json")]


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
    monkeypatch.setattr(S, "sampled_proofs_row", lambda _audit_dir, _run=None, _registration=None: _accepted())
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
                 "--sampled-proofs", str(tmp_path / "audit"), "--registration", str(tmp_path / "registration.json"),
                 "--pieces", str(tmp_path / "pieces"), "--out", str(tmp_path / "record"), "--node", "vy-dc-h200"])
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
def test_the_record_accepts_only_when_every_subprotocol_accepts_on_its_node_and_names_each_that_does_not(
        tmp_path, monkeypatch, capsys, case, piece):
    """Every piece's answer accepted on the record's node (each piece's `RAN`, which a `SERVED` naming no server leaves it
    about, and the wipe's `SERVED`): the composed verdict is ``accept``, with nothing missing, and its reason names each
    subprotocol's accept. One piece with no answer leaves it ``pending`` and names that piece missing with
    why; one that rejects, or accepts with outputs that are not its leg's certificate, makes it ``reject`` naming it, with
    no block record; one about another server leaves it ``pending``, carried and named as elsewhere."""
    monkeypatch.setenv("RESEARCH_RUN_ID", "r1")
    monkeypatch.setattr(S, "sampled_proofs_row", lambda _audit_dir, _run=None, _registration=None: _accepted())
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
                 "--sampled-proofs", str(tmp_path / "audit"), "--registration", str(tmp_path / "registration.json"),
                 "--pieces", str(pieces), "--out", str(tmp_path / "record"), "--node", "vy-dc-h200"])
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
        assert (line["composed"], line["record_missing"], line["missing"], line["elsewhere"]) == ("accept", [], {}, {})
        assert rec["composed"]["legs"] == {k: "accept" for k in BR.LEGS}
        late = {"the network certifier": " (its quantity is not established at δ = 1/1000000: the run established it "
                                         "except with 1/100)"}
        assert rec["composed"]["onsite"]["reason"] == line["reason"] == "; ".join(
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


def test_remote_runs_no_onsite_piece_and_composes_sampled_proofs_and_pouw_alone(tmp_path, monkeypatch, capsys):
    """PRODUCT=remote: `pieces` runs PoUW's piece and no onsite one, whatever its command, printing a DROPPED line for
    each; the record names PoUS, the bandwidth audit, the network certifier and the wipe dropped, neither missing nor
    composed, whatever their pieces left, and composes accept from sampled proofs' and PoUW's accepts on its node. The
    same pieces composed for onsite reject on the wipe's answer. The report and the line say they were composed on the
    served node, from the answer files it wrote, and the published line keeps that. A certifier front's kept fetch and
    session in the serve dir leave remote's record as it was. With ``--deferred`` the record and its line carry what was
    deferred, and sampled proofs, whose audit is V3c's, is missing for it, so the record is pending."""
    monkeypatch.setenv("RESEARCH_RUN_ID", "r1")
    monkeypatch.setattr(S.socket, "gethostname", lambda: "vy-dc-h200")
    monkeypatch.setattr(S, "sampled_proofs_row", lambda _audit_dir, _run=None, _registration=None: _accepted())
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
            str(tmp_path / "audit"), "--registration", str(tmp_path / "registration.json"), "--pieces", str(pieces),
            "--out", str(tmp_path / "record"), "--node", "vy-dc-h200", "--product", "remote"]
    rc = S.main(argv)
    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1].removeprefix("RECORD "))
    rec = json.loads((tmp_path / "record" / "block-record.json").read_text())
    jsonschema.Draft202012Validator(SCHEMA).validate(rec)
    assert rc == 0 and line["check"] == {} == BR.check(rec)
    assert (line["product"], line["composed"], line["record_missing"], line["missing"]) == ("remote", "accept", [], {})
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
    front = tmp_path / "serve" / "private"
    (front / "certifier" / "c20260101T000000Z-ab12").mkdir(parents=True)
    (front / "certifier" / "c20260101T000000Z-ab12" / "fetch.tar.gz").write_bytes(b"the front's kept fetch")
    (front / "certifier-front").mkdir()
    (front / "certifier-front" / "session.json").write_text(json.dumps({"run": "c20260101T000000Z-ab12", "rc": 0}))
    assert S.main(argv) == 0
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1].removeprefix("RECORD ")) == line
    assert json.loads((tmp_path / "record" / "block-record.json").read_text()) == rec
    left = "V3c's audit and the composite, to a follow-on on a node with the memory, which composes the final record"
    assert S.main([*argv, "--deferred", left]) == 0
    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1].removeprefix("RECORD "))
    report = json.loads((tmp_path / "record" / "verdicts.json").read_text())
    assert (line["composed_on"], line["deferred"]) == (report["composed_on"], report["deferred"]) == (where, left)
    assert json.loads(S.public_line("RECORD " + json.dumps(line)).removeprefix("RECORD "))["deferred"] == left
    assert (line["composed"], line["missing"]) == ("pending", {"sampled proofs": f"deferred: {left}"})
    monkeypatch.setattr(S.socket, "gethostname", lambda: "vy-auditor-2")
    assert S.main(argv) == 0
    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1].removeprefix("RECORD "))
    assert line["composed_on"] == {"host": "vy-auditor-2", "served_node": False, "from": S.COPIED_FROM}
    assert S.main([*argv, "--product", "onsite"]) == 1
    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1].removeprefix("RECORD "))
    assert (line["product"], line["composed"], line["by"]) == ("onsite", "reject", ["the wipe"])


#: what `served_verdicts.sh` defers on a node short of V3c's audit's memory, as its record and the split certifier's
#: step-4 piece say it
DEFERRED_AUDIT = ("V3c's audit and the composite, to a follow-on on a node with the memory, which composes the final "
                  "record: this node has 196 GB, under the 290 GB V3c's audit needs (V3C_AUDIT_MEM_GB)")
DEFERRED_CHAIN = ("its chain and period check, which read V3c's period.json, go with V3c's audit to a follow-on on a node "
                  "with the memory, which composes the final record: this node has 196 GB, under the 290 GB V3c's audit "
                  "needs (V3C_AUDIT_MEM_GB)")


@pytest.mark.parametrize("product", ["remote-network", "remote-network-no-pouw"])
@pytest.mark.parametrize("case", ["deferred", "failed", "follow-on", "follow-on failed"])
def test_a_deferred_split_certifier_leaves_the_served_record_pending_and_one_that_ran_and_failed_rejects(
        tmp_path, monkeypatch, capsys, product, case):
    """The split certifier's step-4 piece (`true`, which reads the answer step 6's period check wrote) on a node that
    defers V3c's audit, whose period.json that check reads, isn't run (`pieces --deferred`): it leaves its refusal
    naming the deferral, so the certifier is missing beside sampled proofs, the served node's record composes
    ``pending`` and `record` exits 0, as `served_verdicts.sh` needs. One that ran and wrote no answer (its period check
    failed) is the certifier's ``error``, and the record rejects with no block record, on the served node and in the
    deferral's follow-on, which runs the piece about the served node (`pieces --served`): its answer there is composed
    about the served node, and the record accepts."""
    monkeypatch.setenv("RESEARCH_RUN_ID", "r1")
    monkeypatch.setattr(S, "sampled_proofs_row", lambda _audit_dir, _run=None, _registration=None: _accepted())
    pieces, ran, cert = tmp_path / "pieces", tmp_path / "ran", tmp_path / "pieces" / "certifier"
    pouw = {"verdict": "accept", "reason": None, "commitment": "ef" * 64, **ACCEPTS["pouw"]}
    env = {"POUW_CMD": f"echo '{json.dumps(pouw)}' > \"$3/answer.json\"", "CERTIFIER_CMD": f"touch {ran}; true"}
    if case == "deferred":
        got = S.pieces(tmp_path, 1, pieces, env=env, node="vy-dc-h200", product=product, deferred=DEFERRED_CHAIN)
        refused = f"its piece refused to run: deferred: {DEFERRED_CHAIN}"
        assert not ran.exists() and not (cert / S.RAN).exists() and got["certifier"] == {"verdict": "missing", "why": refused}
    elif case == "failed":
        got = S.pieces(tmp_path, 1, pieces, env=env, node="vy-dc-h200", product=product)
        assert got["certifier"] == {"verdict": "error", "rc": 0,
                                    "reason": "its piece ran (exit 0) and wrote no answer; its log is certifier/piece.log"}
    else:
        S.pieces(tmp_path, 1, pieces, env=env, node="vy-dc-h200", product=product, deferred=DEFERRED_CHAIN)
        shutil.rmtree(cert)
        cert.mkdir()
        if case == "follow-on":
            a = {"verdict": "accept", "reason": None, "commitment": "ef" * 64, **ACCEPTS["certifier"]}
            (cert / "answer.json").write_text(json.dumps(a))
        got = S.pieces(tmp_path, 1, pieces, env={"CERTIFIER_CMD": "true"}, node="cpu-6", only=["certifier"],
                       product=product, served="vy-dc-h200")
        assert json.loads((cert / S.RAN).read_text())["server"] == "cpu-6"
        assert json.loads((cert / S.SERVED).read_text()) == {"server": "vy-dc-h200"}
        assert got["certifier"]["verdict"] == ("accept" if case == "follow-on" else "error")
    capsys.readouterr()
    (tmp_path / "serve").mkdir()
    rc = S.main(["record", "--serve-run", str(_serve_dir(tmp_path / "serve")), "--window", "1", "--sampled-proofs",
                 str(tmp_path / "audit"), "--registration", str(tmp_path / "registration.json"), "--pieces", str(pieces),
                 "--out", str(tmp_path / "record"), "--node", "vy-dc-h200", "--product", product,
                 *(["--deferred", DEFERRED_AUDIT] if case == "deferred" else [])])
    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1].removeprefix("RECORD "))
    report = json.loads((tmp_path / "record" / "verdicts.json").read_text())
    assert report["servers"]["communication"] == (None if case == "deferred" else "vy-dc-h200")
    if case in ("failed", "follow-on failed"):
        assert rc == 1 and (line["composed"], line["by"], line["block_record"]) == ("reject", ["the network certifier"], None)
        assert line["verdicts"]["the network certifier"] == "error"
        assert not (tmp_path / "record" / "block-record.json").exists()
        return
    rec = json.loads((tmp_path / "record" / "block-record.json").read_text())
    jsonschema.Draft202012Validator(SCHEMA).validate(rec)
    assert rc == 0 and line["check"] == {} == BR.check(rec) and line["elsewhere"] == {}
    if case == "deferred":
        assert (line["composed"], line["deferred"]) == ("pending", DEFERRED_AUDIT)
        assert line["missing"] == {"sampled proofs": f"deferred: {DEFERRED_AUDIT}", "the network certifier": refused}
        assert sorted(line["record_missing"]) == ["certifier", "integrity"]
    else:
        assert (line["composed"], line["missing"], line["record_missing"]) == ("accept", {}, [])
        assert rec["legs"]["certifier"]["server"] == "vy-dc-h200"


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
    assert BR.check(rec) == {} and rec["composed"]["onsite"] == report["composed"]
    reason = rec["composed"]["onsite"]["reason"]
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
    θ over the timed call's wall), and its row in the record stays missing with its own reason, its passes not the window
    sampled proofs audited, so the record composes pending; the pieces' line says it's a development run's, and names
    the session."""
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
    assert (row["verdict"], row["why"], row["development"]["verdict"]) == ("missing", S.PEARLPLUS_WHY, "accept")
    report, rec, line = S.record(_rows(_accepted(), tmp_path / "pieces"), window=1, steps=(0, 4), node="n", run="r")
    assert line["composed"] == "pending" and line["development"] == {"PoUW": "accept"}
    assert line["missing"]["PoUW"] == S.PEARLPLUS_WHY and BR.check(rec) == {}
    assert rec["composed"]["onsite"]["reason"].startswith(f"sampled proofs: accept; PoUW: {S.PEARLPLUS_WHY}; ")
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


def test_a_pearlplus_piece_names_the_rounds_its_prover_dropped_before_the_coin(tmp_path, live):
    """A verdict over a retention that dropped rounds before the coin (``served_passes.py retain``) names them in the
    piece and its line: the verdict is the retained passes', and the dropped ones stay in the declared share."""
    served, verified = _pearlplus_run()
    doc = S.pearlplus(tmp_path / "pieces" / "pouw", served, {**verified, "rounds": [0, 1, 3], "dropped": [2]}, **live[1])
    assert (doc["verdict"], doc["dropped"], S.development_line(doc)["dropped"]) == ("accept", [2], [2])
    assert doc["shares"]["macs"]["declared_linears"] == 6000
    assert S.development_line(S.pearlplus(tmp_path / "all", served, verified, **live[1]))["dropped"] is None


def _coin_record(source: str = "5e" * 32) -> bytes:
    """The verifier's coin record as ``served_passes.py verify`` writes it before any replay (``DIR/coin.json``)."""
    return (json.dumps({"schema": S.COIN_KIND, "commitment": "c1" * 64, "source": source, "passes": ["p0", "p1"]},
                       indent=1) + "\n").encode()


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
                   "art": art, "preserved": preserved, "error": error}
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
        with pytest.raises(ValueError, match="is no verity-pouw/pearl-plus-coin/v1 record"):
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
                       "configured): a put of the run's custody copy of coin.json as verity-pouw/pearl-plus-coin/v1 with "
                       "meta {run, commitment} preserves that id"),
            ({True: OSError("gone"), False: OSError("gone")},
             "in no store (research data put --preserve: OSError: gone): coin.json goes with the run's custody"))):
        monkeypatch.setattr(S, "coin", lambda *a, answers=answers, **k: real(*a, **k, put=_put(answers)[0]))
        assert S.main(["coin", str(tmp_path / f"out{i}"), "--record", str(rec), "--run", "r1"]) == 0
        assert capsys.readouterr().out == f"{head}: {said}\n"
    rec.write_text("{}")
    assert S.main(["coin", str(tmp_path / "out9"), "--record", str(rec)]) == 1
    assert capsys.readouterr().out.startswith("COIN refused: ValueError: ")


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
    from verity.core.protocols.pouw import core_check as CC
    from verity.core.protocols.sampled_proofs import challenger as Ch
    from verity.core.protocols.sampled_proofs import remote

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
    drawn statements and live evidence given, the challenger check gets each as its ``--check-challenger`` flag, and the
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


#: the registered compiler a toy tree's ``prove_path.py compiler`` prints (`prove_path.compiler_doc`'s fields)
COMPILER_DOC = {"name": "c", "digest": "d", "cap_bits": REGISTRATION["compiler"]["cap_bits"],
                "inner": {"name": "i2", "digest": "e"}, "served": {"name": "i2", "digest": "e"}, "transfer_gates": [],
                "default_seed": 0, "phi": {"schema": "phi/v1", "model": {"repo": "m", "revision": "r"}, "workload": {},
                                           "sampling": {"kind": "greedy"}, "execution": "bi-eager"}}
#: a toy tree's window circuit: the verifier's, from the run's published compiler and registration alone
WINDOW_CIRCUIT = '''
if a[0] == "compiler":
    print(%r)
    sys.exit(0)
assert a[0] == "window-circuit" and not {"--serve-run", "--window", "--row-dir", "--advice"} & set(a), a
assert [a[a.index(f) + 1].rsplit("/", 1)[-1] for f in ("--compiler", "--registration")] == ["compiler.json",
                                                                                          "registration.json"], a
''' % json.dumps(COMPILER_DOC)
FAKE_PROVE_PATH = f'''import json, os, sys
a = sys.argv[1:]
{WINDOW_CIRCUIT}
open("argv.jsonl", "a").write(json.dumps(a) + "\\n")
c = (a[a.index("--class") + 1] if "--class" in a else "record") + ("+tie" if "--window-registration" in a else "")
answer = json.load(open("answers.json"))[c]
if answer != "count":
    print("prove_path.py window-circuit: refused at", c, file=sys.stderr)
    sys.exit({{"refuse": 2, "expire": 124}}[answer])
open(a[a.index("--out") + 1], "w").write("{{}}")
print(json.dumps({{"count": {{"free": 3, "inputs": 10, "registered": 7, "units": 1}}}}))
'''


def _circuits(tmp_path: Path, classes: str, answers: dict, **extra: str) -> tuple[int, list[str], list[str]]:
    """`served_verdicts.sh circuits 0` in a toy tree whose prove_path.py counts a class, refuses it, or exits 124 as
    `timeout` does when a count runs out of time (``answers``, by class, ``C+tie`` for class C reading H1's tables), under
    a `timeout` that records the limit it got (``extra`` adds to the environment; under H1_TIE the serve dir holds H1's
    window Program's meta): the exit status, the verdict lines, and the limits in order."""
    tree, bin_, venv = tmp_path / "tree", tmp_path / "bin", tmp_path / "venv" / "bin"
    for d in (tree / "benchmarks" / "prove_path", bin_, venv, tmp_path / "serve" / "advice" / "w0"):
        d.mkdir(parents=True)
    (tree / "benchmarks" / "prove_path" / "prove_path.py").write_text(FAKE_PROVE_PATH)
    (tree / "answers.json").write_text(json.dumps(answers))
    (tmp_path / "serve" / "weights_registration.json").write_text("{}")
    (tmp_path / "serve" / "advice" / "w0" / "registration.json").write_text(json.dumps(REGISTRATION))
    if extra.get("H1_TIE"):
        (tmp_path / "serve" / "h1").mkdir()
        (tmp_path / "serve" / "h1" / "window-program.meta.json").write_text(json.dumps({"program_digest": "p" * 128}))
    for path, body in ((venv / "python", f'exec {sys.executable} "$@"'),
                       (bin_ / "timeout", f'echo "$1" >> {tmp_path}/limits; shift; exec "$@"')):
        path.write_text(f"#!/bin/sh\n{body}\n")
        path.chmod(0o755)
    env = {"PATH": f"{bin_}:{os.defpath}", "HOME": str(tmp_path), "VENV": str(venv.parent),
           "D": str(tmp_path / "serve"), "OUT": str(tmp_path / "out"), "CLASSES": classes, **extra}
    rc = subprocess.run(["bash", str(HERE.parent / "served_verdicts.sh"), "circuits", "0"], cwd=tree, env=env,
                        check=False).returncode
    return rc, (tmp_path / "out" / "verdicts.txt").read_text().splitlines(), (tmp_path / "limits").read_text().split()


def test_the_class_of_record_is_measured_after_the_gating_classes_and_its_timeout_fails_nothing(tmp_path):
    rc, lines, limits = _circuits(tmp_path, "4,32,4", {"4,32,4": "count", "record": "expire"})
    assert rc == 0 and limits == ["30m", "45m"], (rc, lines)
    assert lines[0].startswith("WINDOW-CIRCUIT class 4,32,4: registered 7 bits, free 3 bits of 10 input bits at the "
                               "weights registration's pins; ")
    assert lines[1:] == ["WINDOW-CIRCUIT class record (measurement) timed out after WC_RECORD_TIMEOUT=45m (exit 124): no count"]
    out = tmp_path / "out"
    published = json.loads((out / "window-circuit-4-32-4.json").read_text())
    assert published == json.loads((tmp_path / "serve" / "window-circuit-4-32-4.count.json").read_text())
    assert json.loads((out / "compiler.json").read_text()) == COMPILER_DOC
    assert json.loads((out / "registration.json").read_text()) == S.public_registration(REGISTRATION)
    assert (out / "weights_registration.json").read_text() == "{}"


@pytest.mark.parametrize("classes, answers, refused", [
    ("4,32,4 8,64,4", {"4,32,4": "count", "8,64,4": "refuse", "record": "count"}, "WINDOW-CIRCUIT class 8,64,4 refused (exit 2): "),
    ("4,32,4", {"4,32,4": "expire", "record": "count"}, "WINDOW-CIRCUIT class 4,32,4 timed out after WC_TIMEOUT=30m"),
    ("record", {"record": "expire"}, "WINDOW-CIRCUIT class record timed out after WC_TIMEOUT=30m"),
])
def test_any_class_of_classes_that_does_not_count_fails_the_run(tmp_path, classes, answers, refused):
    """Each class of CLASSES gates, not just one of them, whatever the class of record's measurement says; CLASSES that
    names `record` gates on it under WC_TIMEOUT and doesn't measure it again."""
    rc, lines, limits = _circuits(tmp_path, classes, answers)
    assert rc != 0 and sum(line.startswith(refused) for line in lines) == 1, lines
    assert any("prove_path.py window-circuit: refused at" in line for line in lines) == (answers.get("8,64,4") == "refuse")
    measured = [line for line in lines if line.startswith("WINDOW-CIRCUIT class record (measurement)")]
    assert len(measured) == (classes != "record") and limits[-1] == ("30m" if classes == "record" else "45m")


@pytest.mark.parametrize("tied", ["count", "refuse"])
def test_under_h1_tie_step_3_writes_v3cs_record_reading_h1s_tables_after_the_classes_and_gates_nothing_on_it(tmp_path, tied):
    """Under H1_TIE the commit V3c holds is the tie's, whose circuit reads H1's W's tables, so after the classes CLASSES
    gates on, step 3 writes the verifier's record at V3C_CLASS reading them too (window-circuit --window-registration W,
    the window's H1 registration, with --h1-meta its Program's meta as the run publishes it; its tag's suffix -tie), the
    record V3c holds that commit to. It gates nothing."""
    rc, lines, limits = _circuits(tmp_path, "4,32,4", {"4,32,4": "count", "4,32,4+tie": tied, "record": "count"},
                                  H1_TIE="1", V3C_CLASS="4,32,4")
    d, meta = tmp_path / "serve", tmp_path / "out" / "serve" / "h1" / "window-program.meta.json"
    argv = [json.loads(x) for x in (tmp_path / "tree" / "argv.jsonl").read_text().splitlines()]
    assert rc == 0 and limits == ["30m", "30m", "45m"], (rc, lines)
    assert [x[x.index("--out") + 1] for x in argv] == [str(d / f"window-circuit-w0-{t}.json")
                                                       for t in ("4-32-4", "4-32-4-tie", "record")]
    assert [x[x.index("--window-registration"):] if "--window-registration" in x else None for x in argv] == [
        None, ["--window-registration", str(d / "h1" / "w0" / "registration.json"), "--h1-meta", str(meta)], None]
    assert meta.read_bytes() == (d / "h1" / "window-program.meta.json").read_bytes()
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
NCI_CHECK = "verity/core/protocols/network/nci/lean/.lake/build/bin/nci-check"


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
    from verity.ml.definitions.registry.catalog import load
    from verity.ml.windows import interface as IF

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
if a[0] != "window-commit":
{textwrap.indent(WINDOW_CIRCUIT, "    ")}
out = Path(a[a.index("--out") + 1])
assert a[a.index("--class") + 1] == "{W63K4}"
# with W the circuit reads H1's tables, in the verifier's record and the commit alike
tables = {{"tables": "h1"}} if "--window-registration" in a else {{}}
if a[0] == "window-circuit":
    if tables:
        open(os.environ["ORDER"] + ".circuit", "w").write(json.dumps([out.name, *a[a.index("--window-registration"):]]))
    out.write_text(json.dumps(tables))
    c = {{"count": {{"free": 3, "inputs": 10, "registered": 7, "units": 1}}}}
    print(json.dumps({{"bits": 772, **c}} if os.environ.get("ADVICE_BITS") else c))
else:
    assert a[0] == "window-commit"
    if os.environ.get("FAIL_COMMIT"):
        print("prove_path.py window-commit: the row dir holds no window 7", file=sys.stderr)
        sys.exit(2)
    open(os.environ["ORDER"] + ".cap", "w").write(a[a.index("--cap-bits") + 1])
    out.mkdir(parents=True)
    (out / "circuit.json").write_text(json.dumps({{"run": "ab", **tables}}))
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
    assert "--opening" not in a and ("--tie" in a) == ("--tie-prefix" in a), a
    assert a[a.index("--challenger") + 1] == "local", a
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
PERIOD = ('echo period-check >> "$ORDER"; test -s "$4/period.json" || exit 1; '
          'echo \'{"verdict": "accept"}\' > "$3/answer.json"; echo checked')


def _default_cap(requests: int = 4, steps: int = 256, max_model_len: int = 512, id_bytes: int = 24) -> str:
    """ℓ_a's default in `served_verdicts.sh` at those limits (SLOTS, WINDOW_STEPS, MAX_MODEL_LEN, ID_BYTES; their defaults
    with H1 off)."""
    from verity_vllm.pipeline import advice_windows as AW
    from verity_vllm.pipeline import window_advice as WA

    return str(AW.least_cap(WA.max_bits(requests, steps, max_model_len, id_bytes)))


def _v3c(tmp_path: Path, args: tuple[str, ...] = ("v3c",), *, prove_path: str = FAKE_V3C_PROVE_PATH,
         records_audit: str = FAKE_RECORDS_AUDIT, fresh: bool = False, **extra: str | None
         ) -> tuple[int, list[str], list[str], list[str]]:
    """`served_verdicts.sh v3c` (or ``args``) on window 7 of a toy serve dir at the certifier's class, in a toy tree whose
    prove_path.py (``prove_path``) and records_audit.py (``records_audit``) note each step they run, its builds stubbed
    (``fresh``: the tree a fresh clone, with no venv until its `uv sync` makes one, which notes each call in ``uv`` and
    fails under FAIL_SYNC), main's first-parent line a toy MAIN_FILE and the split certifier's two commands set
    (``extra`` adds to the environment, None unsetting): the exit status, the verdict lines, the steps in order and the
    limits."""
    tree, bin_, venv, d = tmp_path / "tree", tmp_path / "bin", tmp_path / "venv" / "bin", tmp_path / "serve"
    for p in (tree / "benchmarks" / "prove_path", tree / ".venv" / "bin", tree / "verity" / "core" / "service" / "flock",
              tree / "verity" / "core" / "service" / "sampled_proofs", tree / "verity" / "ml" / "flock" / "pod", bin_, venv,
              d / "advice" / "w7" / "private", tmp_path / "row"):
        p.mkdir(parents=True)
    (tree / "benchmarks" / "prove_path" / "prove_path.py").write_text(prove_path)
    (tree / "benchmarks" / "prove_path" / "records_audit.py").write_text(records_audit)
    (tree / "verity" / "ml" / "flock" / "pod" / "60-circuit.sh").write_text("exit 0\n")
    for f in (d / "advice" / "w7" / "private" / "advice.bin", tmp_path / "row" / "manifest.json",
              tmp_path / "row" / "build_summary.json", tmp_path / "h.so", tmp_path / "n.so"):
        f.write_text("")
    (d / "weights_registration.json").write_text("{}")
    (d / "advice" / "w7" / "registration.json").write_text(json.dumps(REGISTRATION))
    _write(d, "windows.jsonl", [{"window": 7, "run_root": "ab" * 64}])
    (tmp_path / "main.txt").write_text(TOY_MAIN)
    # the toy tree's verity/ directories, on the PYTHONPATH the script sets and as `python -`'s working directory, would
    # shadow the installed verity
    py = f'unset PYTHONPATH\nexport PYTHONSAFEPATH=1\n[ "$1 $2" = "-m research" ] && exit 0\nexec {sys.executable} "$@"'
    for path, body in ((venv / "python", py), (tree / ".venv" / "bin" / "python", py),
                       (bin_ / "timeout", f'[ "$1" = -k ] && shift 2\necho "$1" >> {tmp_path}/limits; shift; exec "$@"'),
                       (bin_ / "gpu-lease", "exit 99"), (bin_ / "uv", "exit 0"), (bin_ / "lake", "exit 0")):
        _script(path, body)
    if fresh:
        shutil.rmtree(tree / ".venv")
        _script(bin_ / "uv", f'echo "$*" >> {tmp_path}/uv\n[ -z "${{FAIL_SYNC:-}}" ] || exit 1\n'
                             f'[ -e {tree}/.venv/bin/python ] || {{ mkdir -p {tree}/.venv/bin '
                             f'&& cp -p {venv}/python {tree}/.venv/bin/python; }}')
    env = {"PATH": f"{bin_}:{os.defpath}", "HOME": str(tmp_path), "VENV": str(venv.parent), "D": str(d), "K": "7",
           "OUT": str(tmp_path / "out"), "SERVE_ROOT": str(tmp_path / "root"), "FLOCK_WORK": str(tmp_path / "fw"),
           "ROW": str(tmp_path / "row"), "HIDDEN_SO": str(tmp_path / "h.so"), "NORM_TAP_SO": str(tmp_path / "n.so"),
           "ORDER": str(tmp_path / "order"), "H1_GATE": "", "V3C_CLASS": W63K4, "CLASSES": W63K4,
           "CERTIFIER_CHAIN_CMD": CHAIN, "CERTIFIER_PERIOD_CMD": PERIOD, "NODE_MEM_GB": "512", "MIN_FREE_INODES": "0",
           "MAIN_FILE": str(tmp_path / "main.txt"), "GIT_CEILING_DIRECTORIES": str(tmp_path)}
    env = {k: v for k, v in {**env, **extra}.items() if v is not None}
    rc = subprocess.run(["bash", str(HERE.parent / "served_verdicts.sh"), *args], cwd=tree, env=env, check=False,
                        capture_output=True).returncode
    read = lambda f: (tmp_path / f).read_text().split() if (tmp_path / f).exists() else []  # noqa: E731
    return rc, (tmp_path / "out" / "verdicts.txt").read_text().splitlines(), read("order"), read("limits")


@pytest.mark.parametrize("given", [None, "0.25"])
def test_the_certifiers_chain_proves_its_scan_at_2_to_the_minus_64_unless_the_run_sets_cr_scan_delta(tmp_path, given):
    """The split certifier's chain gets CR_SCAN_DELTA, which chain_record.sh hands scan_zk.py as --delta: by default
    2^-64 exactly (every scan block is drawn, so no δ costs proving), else the run's own."""
    chain = CHAIN + '; printf "%s" "$CR_SCAN_DELTA" > "$ORDER.delta"'
    rc, lines, _, _ = _v3c(tmp_path, CERTIFIER_CHAIN_CMD=chain, CR_SCAN_DELTA=given)
    assert rc == 0, lines
    got = (tmp_path / "order.delta").read_text()
    assert got == given if given else Fraction(float(got)) == Fraction(1, 2 ** 64)


def test_v3c_runs_the_certifiers_chain_beside_its_audit_and_its_period_check_on_the_period_json_after(tmp_path):
    """The window commit, then the split certifier's chain (it reads the commit's scan state) beside V3c's audit and its
    period.json, and the certifier's period check on it once both are done, each within its own limit; the check gets
    the serve dir, the window, the certifier's piece dir and V3c's dir, and writes the answer `pieces` reads there. The
    two steps' logs, which count the window's requests, end in the serve dir's private/pieces/certifier, and V3c's
    audit's in its private/v3c-audit, not in $OUT, which leaves the node; $OUT/v3c-audit holds what the record reads of
    the audit alone, and what a verifier reads of $OUT scans clean (`kind_scan.py --received`) and leaves as it is. $OUT
    then names no window and counts none of its own (the window commit's population, the audit's, the chain's window):
    each line is whole in the serve dir's private/verdicts.txt, each file in its private/out."""
    rc, lines, order, limits = _v3c(tmp_path)
    assert rc == 0, lines
    assert order[:3] == ["compiler", "window-circuit", "window-commit"] and order[-1] == "period-check"
    assert sorted(order[3:-1]) == ["audit", "chain", "period"] and order.index("audit") < order.index("period")
    assert limits[:2] == ["30m", "120m"] and sorted(limits[2:-1]) == ["180m", "4h"] and limits[-1] == "12h"
    vw, pieces = tmp_path / "serve" / f"v3c-w7-{W63K4.replace(',', '-')}", tmp_path / "out" / "pieces" / "certifier"
    assert (tmp_path / "order.args").read_text().split() == [str(tmp_path / "serve"), "7", str(pieces), str(vw)]
    assert not _window_data(tmp_path / "out", 7), _window_data(tmp_path / "out", 7)
    held = (tmp_path / "serve" / "private" / "verdicts.txt").read_text()
    assert "chain on window 7" in held and '"population": 37' in held and "chain on window K" in "\n".join(lines)
    assert '"population": 37' in (tmp_path / "serve" / "private" / "v3c-audit" / "audit.log").read_text()
    assert sorted(p.relative_to(tmp_path / "out").as_posix() for p in (tmp_path / "out" / "v3c-audit").rglob("*")) == [
        "v3c-audit/period.json", "v3c-audit/summary.json"]
    scan = (tmp_path / "serve" / "private" / "received-scan.txt").read_text().splitlines()
    assert scan[-1].startswith("KIND-SCAN clean: ") and not (tmp_path / "serve" / "private" / "out" / "compiler.json").exists()
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


@pytest.mark.parametrize("extra", [{"NODE_MEM_GB": "320"}, {"CERTIFIER_CHAIN_MEM_GB": "300"}])
def test_v3c_runs_the_certifiers_chain_after_its_audit_on_a_node_that_holds_only_one(tmp_path, extra):
    """On a node with less memory than V3c's audit and the chain take together (NODE_MEM_GB under V3C_AUDIT_MEM_GB plus
    CERTIFIER_CHAIN_MEM_GB), the chain runs after the audit and its period.json rather than beside them, a line saying
    why before either starts, and the period check still runs last, on both."""
    rc, lines, order, limits = _v3c(tmp_path, **extra)
    assert rc == 0, lines
    assert order[3:] == ["audit", "period", "chain", "period-check"] and limits[2:] == ["180m", "4h", "12h"]
    node, chain = extra.get("NODE_MEM_GB", "512"), extra.get("CERTIFIER_CHAIN_MEM_GB", "96")
    heads = [line.split(" (")[0].split(":")[0] for line in lines]
    assert heads[3:] == ["CERTIFIER-CHAIN after V3c's audit", f"V3C-AUDIT class {W63K4}", "V3C-PERIOD", "CERTIFIER-CHAIN",
                         "CERTIFIER-PERIOD"]
    assert lines[3] == (f"CERTIFIER-CHAIN after V3c's audit: this node has {node} GB, under the 290 GB V3c's audit needs "
                        f"(V3C_AUDIT_MEM_GB) plus the {chain} GB of the chain beside it (CERTIFIER_CHAIN_MEM_GB)")
    assert json.loads((tmp_path / "out" / "pieces" / "certifier" / "answer.json").read_text()) == {"verdict": "accept"}


def test_what_a_verifier_reads_that_reads_the_advice_fails_the_run_and_stays_on_the_node(tmp_path):
    """A window circuit's count that states the advice's length (as one counted from the advice would): the kind scan
    of what a verifier reads of $OUT finds it, the run fails though every step passed, and each file the scan reads is
    held whole in the serve dir's private/out, none of it left in $OUT."""
    rc, lines, _order, _limits = _v3c(tmp_path, ADVICE_BITS="1")
    assert rc == 1 and lines[-1].startswith("RECEIVED-SCAN failed (exit 1, 1 findings, in the serve dir's "), lines
    out, private = tmp_path / "out", tmp_path / "serve" / "private"
    assert "window-circuit-8-32-4-4-16.json bits: a int of no declared kind" in (private / "received-scan.txt").read_text()
    left = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    assert not [p for p in left if any(fnmatch.fnmatchcase(p, g) for g in S.RECEIVED)], left
    assert json.loads((private / "out" / "window-circuit-8-32-4-4-16.json").read_text())["bits"] == 772
    assert (private / "out" / "compiler.json").exists() and (private / "out" / "pieces" / "certifier" / "answer.json").exists()


def test_v3c_with_no_period_json_runs_no_period_check_and_fails(tmp_path):
    """V3c's audit wrote no period.json: no period check, and `served_verdicts.sh v3c` exits 1 once it has moved the
    chain's log to the serve dir's private/pieces/certifier."""
    rc, lines, order, _limits = _v3c(tmp_path, FAIL_PERIOD="1")
    assert rc == 1 and order[:3] == ["compiler", "window-circuit", "window-commit"]
    assert sorted(order[3:]) == ["audit", "chain", "period"] and order.index("audit") < order.index("period")
    assert any("the audit under it wrote no Program" in x for x in lines[-3:-1])
    assert lines[-1] == "CERTIFIER-PERIOD: not run: V3c wrote no period.json"
    pieces = tmp_path / "out" / "pieces" / "certifier"
    assert not (pieces / "answer.json").exists() and not (pieces / "chain.log").exists()
    assert (tmp_path / "serve" / "private" / "pieces" / "certifier" / "chain.log").exists()


@pytest.mark.parametrize("given", [True, False])
def test_pp_main_comes_before_main_file_as_the_line_v3cs_session_pins_on(tmp_path, given):
    """PP_MAIN, run.sh's name for the verifier's copy of main's first-parent line, comes before MAIN_FILE: V3c's audit's
    session pins on it, and one that isn't there leaves the session nothing to pin on, so V3c runs no audit and
    `served_verdicts.sh v3c` fails."""
    pp, line = tmp_path / "pp-main.txt", f"{'cd' * 20}\n"
    if given:
        pp.write_text(line)
    rc, lines, order, _limits = _v3c(tmp_path, PP_MAIN=str(pp))
    if given:
        assert rc == 0 and (tmp_path / "order.main").read_text() == line, lines
    else:
        assert rc == 1 and "audit" not in order and "window-commit" in order, (lines, order)
        assert f"V3C-AUDIT: PP_MAIN {pp}, main's first-parent line its challenger session pins on, is missing or " \
               "empty" in lines, lines


@pytest.mark.parametrize("args, extra, line", [
    (("v3c",), {"CERTIFIER_CMD": "true", "CERTIFIER_PERIOD_CMD": None},
     "PREFLIGHT: CERTIFIER_CMD or the split's CERTIFIER_CHAIN_CMD and CERTIFIER_PERIOD_CMD, not both"),
    ((), {"V3C_CLASS": "", "CERTIFIER_CHAIN_CMD": None, "CERTIFIER_PERIOD_CMD": None},
     "PREFLIGHT: sampled proofs' audit of record is V3c's (and the split certifier reads its period.json), and "
     "V3C_CLASS is empty"),
    ((), {"CERTIFIER_PERIOD_CMD": None},
     "PREFLIGHT: the split certifier runs its chain and its period check, and CERTIFIER_PERIOD_CMD is unset"),
    (("v3c",), {"CERTIFIER_CHAIN_CMD": None, "PRODUCT": "remote-network"},
     "PREFLIGHT: the split certifier runs its chain and its period check, and CERTIFIER_CHAIN_CMD is unset"),
])
def test_the_split_certifier_needs_v3c_and_no_whole_piece_beside_it(tmp_path, args, extra, line):
    """The split's period check writes the certifier's answer from V3c's period.json, so the preflight refuses it beside
    CERTIFIER_CMD, the whole piece's command, and with either of its two commands unset (a chain with no period check
    answers nothing, and a period check with no chain refuses); and V3c's audit is sampled proofs' of record, so it
    refuses a run that leaves V3c out, split or not, before anything runs."""
    rc, lines, order, _limits = _v3c(tmp_path, args, **extra)
    assert (rc, lines, order) == (3, [line], [])


W_OF_7 = {"format": "verity/registration/v2", "window": 7, "previous": None}


def _tie(tmp_path: Path, committed: bool = True, **extra: str) -> tuple[subprocess.CompletedProcess, list[str]]:
    """The ``--h1-window-commit`` that `served_verdicts.sh serve` gives serve-openai under H1_TIE, run as `serve_tie` runs
    it at window 7's close (``bash -c CMD _ SERVE_DIR K W_FILE R_FILE PROVER_DIR``, from elsewhere), in a toy tree whose
    prove_path.py notes each step it runs, on a serve dir where the server committed window 7's advice (unless not
    ``committed``) padded to the ℓ_a the run exports (CAP, 1,008 bits; ``extra`` adds to the environment): the run and the
    steps in order."""
    rc, argv = _serve_argv(tmp_path, {"H1_GATE": GATE, "H1_CIRCUITS": str(tmp_path / "circuits"), "H1_TIE": "1"})
    assert rc == 10 and "--h1-prover" in argv and argv.count("--h1-window-commit") == 1, (rc, argv)
    cmd = argv[argv.index("--h1-window-commit") + 1]
    tree, d, venv = tmp_path / "tree", tmp_path / "serve", tmp_path / "venv" / "bin"
    (tree / "benchmarks" / "prove_path").mkdir(parents=True)
    (tree / "benchmarks" / "prove_path" / "prove_path.py").write_text(FAKE_V3C_PROVE_PATH)
    (tree / "benchmarks" / "one_stage").symlink_to(HERE.parent, target_is_directory=True)
    _script(venv / "python", f'unset PYTHONPATH\nexport PYTHONSAFEPATH=1\nexec {sys.executable} "$@"')
    if committed:
        (d / "advice" / "w7" / "private").mkdir(parents=True)
        (d / "advice" / "w7" / "registration.json").write_text("{}")
        (d / "advice" / "w7" / "private" / "advice.bin").write_bytes(b"\x01")
    for p in (d / "h1" / "w7", d / "private" / "h1" / "w7"):
        p.mkdir(parents=True, exist_ok=True)
    (d / "weights_registration.json").write_text("{}")
    (d / "h1" / "w7" / "registration.json").write_text(json.dumps(W_OF_7))
    env = {"PATH": os.defpath, "HOME": str(tmp_path), "VENV": str(venv.parent), "D": str(d), "OUT": str(tmp_path / "tie-out"),
           "ORDER": str(tmp_path / "order"), "H1_TIE": "1", "V3C_CLASS": W63K4, "CAP": "1008", **extra}
    run = subprocess.run(["bash", "-c", cmd, "_", str(d), "7", str(d / "h1" / "w7" / "registration.json"),
                          str(d / "h1" / "w7" / "record.json"), str(d / "private" / "h1" / "w7")],
                         cwd=tmp_path, env=env, check=False, capture_output=True, text=True)
    return run, (tmp_path / "order").read_text().split() if (tmp_path / "order").exists() else []


def test_under_h1_tie_each_h1_windows_close_commits_it_on_the_servers_advice_chained_after_its_w_as_its_record_r(tmp_path):
    """serve-openai's tie command, from any directory, runs window-commit on the window's advice as the server committed
    it, read at the ℓ_a the server padded it to (CAP), at V3C_CLASS on W, the window's H1 prover key and V3c's law, and
    copies the registration it wrote to R. It commits no advice of its own, so the window's advice has one commitment, and
    says nothing in $OUT (its window is H1's, which no published line names); its commit's answer stays in the serve
    dir."""
    run, order = _tie(tmp_path)
    d = tmp_path / "serve"
    assert run.returncode == 0 and order == ["window-commit"], (run.stdout, run.stderr, order)
    assert (tmp_path / "order.cap").read_text() == "1008"
    assert run.stdout.splitlines()[-1] == f"TIE: window-commit class {W63K4}: its registration is the window's record R"
    vw = d / f"v3c-w7-{W63K4.replace(',', '-')}" / "window"
    assert json.loads((d / "h1" / "w7" / "record.json").read_text()) == json.loads((vw / "registration.json").read_text()) \
        == {"format": "toy", "after": W_OF_7}
    assert (tmp_path / "order.commit").read_text().split() == [
        "--window-registration", str(d / "h1" / "w7" / "registration.json"), "--h1-key", str(d / "private" / "h1" / "w7" / "key"),
        "--law", "stratified:1"]
    assert sorted(p.relative_to(d / "advice").as_posix() for p in (d / "advice").rglob("*")) == [
        "w7", "w7/private", "w7/private/advice.bin", "w7/registration.json"]
    assert json.loads((d / "h1" / "w7" / "window-commit.json").read_text())["population"] == 37
    assert not (tmp_path / "tie-out" / "verdicts.txt").exists()


@pytest.mark.parametrize("committed, extra, rc, steps, said", [
    (False, {}, 14, [], "TIE: the server committed no advice for the window (advice.jsonl): it refused it, or a request of "
     "it runs on past its close"),
    (True, {"FAIL_COMMIT": "1"}, 16, ["window-commit"],
     f"TIE: window-commit class {W63K4} refused (exit 2): prove_path.py window-commit: the row dir holds no window 7"),
])
def test_a_tie_on_a_window_the_server_committed_no_advice_for_or_whose_commit_is_refused_writes_no_record_r(
        tmp_path, committed, extra, rc, steps, said):
    """The server commits a window's advice once it is whole, and `serve_tie`'s flush commits every whole window before the
    command runs. A window it committed none for (it refused it, or a request carried past the close has no Commit record
    until a later close) fails the tie at once, which never polls and commits no advice of its own; a refused
    window-commit fails it too. Either way no R, so `serve_tie` refuses the window's record and H1 draws no sample coin."""
    run, order = _tie(tmp_path, committed, **extra)
    assert (run.returncode, order, run.stdout.splitlines()[-1]) == (rc, steps, said), run.stderr
    assert not (tmp_path / "serve" / "h1" / "w7" / "record.json").exists()


#: R's sample in `_tied_commit`: stratified:1 over three tables' readers, a proof block of each
TIED_LAW = {"law": "stratified", "k": 1, "strata": [{"template": t, "units": [[3 * i, 3 * i + 3]], "k": 1}
                                                   for i, t in enumerate(("a@0", "ids@0", "ids@1"))]}


def _tied_commit(tmp_path: Path, bound: str | None, law: dict | None = TIED_LAW) -> None:
    """Window 7's commit as the tie left it in the toy serve dir: its registration R (its sample ``law``; None, none) and
    circuit, which reads H1's tables, its answer, H1's window Program's meta, and H1's log naming W and, unless ``bound``
    is None, the record H1 bound (``same``: R; ``other``: another)."""
    d = tmp_path / "serve"
    vw = d / f"v3c-w7-{W63K4.replace(',', '-')}" / "window"
    vw.mkdir(parents=True)
    (d / "h1" / "w7").mkdir(parents=True)
    r = {"format": "toy", "previous": REG.digest(W_OF_7), **({"law": law} if law else {})}
    (vw / "registration.json").write_text(json.dumps(r))
    (vw / "circuit.json").write_text(json.dumps({"run": "ab", "tables": "h1"}))
    (d / "h1" / "window-program.meta.json").write_text(json.dumps({"program_digest": "p" * 128}))
    (d / "h1" / "w7" / "window-commit.json").write_text(json.dumps({"run": "ab", "circuit": "cd", "population": 37}))
    log = [{"window": 7, "kind": "registration", "file": "w7/registration.json", "digest": "aa" * 64}]
    if bound is not None:
        log.append({"window": 7, "kind": "record", "file": "w7/record.json",
                    "digest": REG.digest(r if bound == "same" else {**r, "previous": None})})
    _write(d / "h1", "log.jsonl", log)


def test_under_h1_tie_v3c_audits_the_commit_h1_bound_and_makes_none_of_its_own(tmp_path):
    """The window's one commit is the one its close made: V3c holds its registration to the record H1 bound (the log's
    digest) and its circuit to the verifier's, which reads H1's W's tables as the commit's does (window-circuit
    --window-registration W, with its Program's meta as the run publishes it; its own record beside the untied one), runs
    no window-commit, and goes on to its audit."""
    _tied_commit(tmp_path, "same")
    rc, lines, order, _limits = _v3c(tmp_path, H1_TIE="1")
    assert rc == 0 and "window-commit" not in order and order[:2] == ["compiler", "window-circuit"] and "audit" in order, \
        (lines, order)
    d = tmp_path / "serve"
    assert json.loads((tmp_path / "order.circuit").read_text()) == [
        f"window-circuit-w7-{W63K4.replace(',', '-')}-tie.json", "--window-registration", str(d / "h1" / "w7" / "registration.json"),
        "--h1-meta", str(tmp_path / "out" / "serve" / "h1" / "window-program.meta.json")]
    assert not (d / f"window-circuit-w7-{W63K4.replace(',', '-')}.json").exists()
    assert lines[0].startswith(f"WINDOW-CIRCUIT class {W63K4} reading H1's tables: registered 7 bits"), lines
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
    (--challenger local), pinned on main's first-parent line: MAIN_FILE's, else MAIN_SHA's in the tree, kept as the
    serve dir's private/v3c-audit/main.txt, the --main the audit reads, never in $OUT/v3c-audit, which holds only what
    the record reads (`kind_scan.V3C_DOCS`)."""
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
    assert (tmp_path / "serve" / "private" / "v3c-audit" / "main.txt").read_text() == line
    assert not (tmp_path / "out" / "v3c-audit" / "main.txt").exists()


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
FOLLOW_COINS = ("a local development challenger session on cpu-6, the follow-on's node, where it runs V3c's audit, "
                "not an auditor's (PP_CHALLENGER local)")
@pytest.mark.parametrize("extra, coins", [
    ({}, LOCAL_COINS),
    ({"PP_CHALLENGER": "gate:research@auditor", "PP_PEER": "vy-dc-h200"},
     "the challenger gate:research@auditor (peer vy-dc-h200)"),
])
def test_the_coins_line_says_whose_coins_each_sample_is_on(tmp_path, extra, coins):
    """Before step 0 the run says whose coins each sample is on: sampled proofs' (V3c's audit's local development session
    unless PP_CHALLENGER names a challenger) and PoUW H1's. Here step 0's `predict row` then fails, and the run stops
    there."""
    venv = tmp_path / "toy-venv" / "bin"
    _script(venv / "python", '[ "$1 $2" = "-m research" ] && exit 0\n[ "$1 $2" = "-m verity_vllm.pipeline.cli" ] && exit 9\n'
                             f'unset PYTHONPATH\nexport PYTHONSAFEPATH=1\nexec {sys.executable} "$@"')
    rc, lines, _order, _limits = _v3c(tmp_path, (), VENV=str(venv.parent), HOST="vy-dc-h200",
                                      PATH=f"{_hosts(tmp_path)}:{tmp_path / 'bin'}:{os.defpath}", **extra)
    assert rc == 17 and lines[-1] == "STOP at predict (exit 17)", lines
    assert f"COINS sampled proofs: {coins}; PoUW H1: not served (H1_GATE empty); the bandwidth audit's: " \
           "vy-auditor-2's bw_party, inside PoUS's piece" in lines, lines


PEARLPLUS_RUN = "PoUW: Pearl+'s own served run on this card after the window's (POUW=pearlplus)"
FAKE_GATE = ('case "$1" in\n'
             f"ledger) echo '{LEDGER_LINE}' ;;\n"
             f"""pins) echo '{{"commit": "{"ef" * 20}", "main": "{"ef" * 20}", "pinned_utc": "2026-10-26T21:35:00Z"}}' ;;\n"""
             "*) exit 2 ;;\nesac")


@pytest.mark.parametrize("live", [True, False])
def test_under_pouw_pearlplus_the_coins_line_says_whether_pearlpluss_epoch_coins_are_live(tmp_path, live):
    """Under POUW=pearlplus the run checks the gate Pearl+'s serve reserves its session through, PEARLC_CHALLENGER_GATE
    (a lane with a period open would be refused), and the COINS line says Pearl+'s epoch's salt coin is a live session's
    of its own there; with PEARLC_CHALLENGER_GATE empty no gate is checked, and the COINS line, the record's first,
    leads with the coins being in-process and the record claiming no Pearl+ verdict."""
    venv = tmp_path / "toy-venv" / "bin"
    _script(venv / "python", '[ "$1 $2" = "-m research" ] && exit 0\n[ "$1 $2" = "-m verity_vllm.pipeline.cli" ] && exit 9\n'
                             f'unset PYTHONPATH\nexport PYTHONSAFEPATH=1\nexec {sys.executable} "$@"')
    _script(gate := tmp_path / "gate", FAKE_GATE)
    (one := tmp_path / "tree" / "benchmarks" / "one_stage").mkdir(parents=True)
    (one / "served_verdicts.py").symlink_to(HERE.parent / "served_verdicts.py")
    rc, lines, _order, _limits = _v3c(tmp_path, (), VENV=str(venv.parent), HOST="vy-dc-h200",
                                      PATH=f"{_hosts(tmp_path)}:{tmp_path / 'bin'}:{os.defpath}", POUW="pearlplus",
                                      PEARLC_CHALLENGER_GATE=str(gate) if live else "")
    assert rc == 17 and lines[-1] == "STOP at predict (exit 17)", lines
    tail = "; the bandwidth audit's: vy-auditor-2's bw_party, inside PoUS's piece"
    if live:
        assert lines[0] == (f"GATE ready: lane vllm's open period None; the ledger's 1 entries; pinned {'ef' * 6} (main "
                            f"{'ef' * 6}, 2026-10-26T21:35:00Z), on the verifier's first-parent line of main"), lines
        assert lines[1] == (f"COINS sampled proofs: {LOCAL_COINS}; {PEARLPLUS_RUN}, its epoch's salt coin a live session's "
                            "of its own on vy-challenger-1's gate (peer vy-dc-h200, lane vllm), each pass's salt checked by "
                            "core_check.live_salt in its piece, and its replay's draws the Lean verifier's "
                            "IO.getRandomBytes on this node, taken once before the replay and cited by store id in the "
                            "POUW line's coin; a development piece under the lane's development key, it credits "
                            f"nothing{tail}"), lines
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
    assert (rc, order, lines[-1]) == (16, ["compiler", "window-circuit"], f"WINDOW-COMMIT class {W63K4}: {why}"), lines


TIES = {"H1_TIE": "1", "H1_GATE": GATE, "H1_CIRCUITS": "circuits"}


@pytest.mark.parametrize("extra, prove_path, line", [
    ({"H1_TIE": "1"}, FAKE_V3C_PROVE_PATH,
     "PREFLIGHT: H1_TIE ties H1's windows, and H1_GATE or H1_CIRCUITS (whose prover state window-commit reads) is unset"),
    (TIES, "import sys\n", "PREFLIGHT: H1_TIE needs window-commit's --window-registration, its registration chained right "
                           "after H1's, which this tree's prove_path.py doesn't take"),
    ({**TIES, "V3C_TIMEOUT": "120m"}, FAKE_V3C_PROVE_PATH, "PREFLIGHT: under H1_TIE the window's sample coin waits for "
     "window-commit, and its session's reserve lasts the gate's 120 minutes from the serve's start: V3C_TIMEOUT is at most "
     "90m, not 120m"),
    ({**TIES, "V3C_TIMEOUT": "1h"}, FAKE_V3C_PROVE_PATH, "PREFLIGHT: under H1_TIE the window's sample coin waits for "
     "window-commit, and its session's reserve lasts the gate's 120 minutes from the serve's start: V3C_TIMEOUT is at most "
     "90m, not 1h"),
    ({**TIES, "NODE_MEM_GB": "196"}, FAKE_V3C_PROVE_PATH, "PREFLIGHT: where this node has 196 GB, under the 290 GB "
     "V3c's audit needs (V3C_AUDIT_MEM_GB), its audit and the composite are deferred to a follow-on, and onsite's "
     "composite reads PoUS's and the wipe's cases, which only this node's card writes: PRODUCT=remote, "
     "remote-network or remote-network-no-pouw, COMPOSITE set empty, or a node with the memory"),
    ({**TIES, "V3C_AUDIT_MEM_GB": "600"}, FAKE_V3C_PROVE_PATH, "PREFLIGHT: where this node has 512 GB, under the 600 GB "
     "V3c's audit needs (V3C_AUDIT_MEM_GB), its audit and the composite are deferred to a follow-on, and onsite's "
     "composite reads PoUS's and the wipe's cases, which only this node's card writes: PRODUCT=remote, "
     "remote-network or remote-network-no-pouw, COMPOSITE set empty, or a node with the memory"),
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


def test_an_untied_onsite_run_on_a_node_short_of_v3cs_audits_memory_is_refused_as_a_tied_one_is(tmp_path):
    """Untied too, a node short of V3c's audit's memory defers the audit and the composite, so onsite, whose composite
    reads the card's PoUS and wipe cases, is refused before anything runs."""
    rc, lines, order, _limits = _v3c(tmp_path, (), NODE_MEM_GB="196")
    assert (rc, lines, order) == (3, [
        f"PREFLIGHT: where {SHORT}, its audit and the composite are deferred to a follow-on, and onsite's composite reads "
        "PoUS's and the wipe's cases, which only this node's card writes: PRODUCT=remote, remote-network or "
        "remote-network-no-pouw, COMPOSITE set empty, or a node with the memory"], [])


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
                  "h1/w7/window-commit.json", "private/pouw/period/case.json", "private/pouw/period/answer.json")
FAKE_RECORD = '''import json, os, sys
a = sys.argv[1:]
open(os.environ["ORDER"], "a").write(a[0] + "\\n")
open(os.environ["ORDER"] + "." + a[0], "w").write(json.dumps(a))
if a[0] == "pieces":
    open(os.environ["ORDER"] + ".pieces-cmd", "w").write(os.environ["CERTIFIER_CMD"])
else:
    print("RECORD " + json.dumps({"composed": "pending", "window": 7}))
'''
FAKE_SERVED_COMPOSITE = '''echo composite >> "$ORDER"
printf "%s\\n" "$SERVED_NODE" "$PRODUCT" "$@" > "$ORDER.composite"
echo "COMPOSITE: composed on $(hostname -s)" >> "$2/verdicts.txt"'''


def _hosts(tmp_path: Path) -> Path:
    """A directory, for PATH, whose `hostname` says $HOST."""
    _script(tmp_path / "hosts" / "hostname", 'echo "$HOST"')
    return tmp_path / "hosts"


def _untied_commit(tmp_path: Path) -> None:
    """Window 7's commit as untied step 6 made it on the served node: its window dir's registration and circuit, and its
    answer beside the serve dir's other records."""
    d = tmp_path / "serve"
    vw = d / f"v3c-w7-{VTAG}" / "window"
    vw.mkdir(parents=True)
    (vw / "registration.json").write_text(json.dumps({"format": "toy", "previous": None}))
    (vw / "circuit.json").write_text(json.dumps({"run": "ab"}))
    (d / f"window-commit-{VTAG}.json").write_text(json.dumps({"run": "ab", "circuit": "cd", "population": 37}))


def _deferred(tmp_path: Path, tied: bool = True, **extra: str) -> subprocess.CompletedProcess:
    """Window 7 of a toy serve dir under H1's tie (untied, ``tied`` False: no H1 records, and its commit step 6's): its
    commit as the tie left it, with a value file, H1's records, the record's and the composite's inputs, files no step
    after the serve reads (a row dir, a key, run.sh's records audit), and the pieces the record read in $OUT; then step 6's
    deferral (`served_verdicts.sh defer WHY`) on vy-dc-h200, a node short of V3c's audit's memory, at the tree of commit
    SHA (``extra`` adds to the environment)."""
    d, out = tmp_path / "serve", tmp_path / "served-out"
    if tied:
        _tied_commit(tmp_path, "same")
    else:
        _untied_commit(tmp_path)
    vw = d / f"v3c-w7-{VTAG}" / "window"
    h1 = {"h1/w7/registration.json": W_OF_7, "h1/w7/record.json": json.loads((vw / "registration.json").read_text())}
    files = {**(h1 if tied else {}),
             f"window-circuit-w7-{VTAG}.json": {}, f"window-circuit-w7-{VTAG}-tie.json": {"tables": "h1"},
             "weights_registration.json": {}, "seals.jsonl": {"window": 7},
             "requests.jsonl": {"windows": [7]}, "private/pouw/period/case.json": {"evidence": "beef"},
             "private/pouw/period/answer.json": {"protocol": "verity/pouw/period/v1"},
             "windows/w7/rows.bin": {}, "private/weights.key": {}, "sampled-proofs-served/window/audit.json": {}}
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
           "PRODUCT": "remote", **({"H1_TIE": "1"} if tied else {}), "HOST": "vy-dc-h200", "RESEARCH_SOURCE_SHA": SHA,
           "GIT_CEILING_DIRECTORIES": str(tmp_path), **extra}
    return subprocess.run(["bash", str(HERE.parent / "served_verdicts.sh"), "defer", SHORT], cwd=tmp_path / "served-tree",
                          env=env, check=False, capture_output=True, text=True)


def test_a_node_short_of_v3cs_audits_memory_defers_it_and_the_composite_listing_and_hashing_each_input(tmp_path):
    """Step 6's deferral leaves in the serve dir's private/deferred the pieces the record read, the settings the
    follow-on runs at (the window, the product, V3c's class and law, H1_TIE, the composite, the served node, the tree's
    commit, WINDOW_STEPS and MAX_TOKENS, which label the onsite record, and OF_RECORD), and the follow-on's every
    input, listed for the prover's copy and hashed for the follow-on to hold it to: V3c's window dir whole, H1's log
    and each record it names, the tie's commit answer, the record's and the composite's inputs and its own files, and
    nothing else of the serve dir. Its one line in $OUT names what was deferred, why and the follow-on's command, and no
    window; nothing else lands in $OUT."""
    run = _deferred(tmp_path)
    assert run.returncode == 0, run.stderr
    d, dd, out = tmp_path / "serve", tmp_path / "serve" / "private" / "deferred", tmp_path / "served-out"
    names = ("K PRODUCT V3C_CLASS V3C_LAW H1_TIE COMPOSITE ALPHA GAMMA SERVED_NODE DEFERRED_TREE CR_SCAN_DELTA ONSITE "
             "CERTIFIER_CHAIN_CMD CERTIFIER_PERIOD_CMD WINDOW_STEPS MAX_TOKENS OF_RECORD")
    said = subprocess.run(["bash", "-c", f'. "$1"; for v in {names}; do printf "%s=%s\\n" "$v" "${{!v}}"; done', "_",
                           str(dd / "deferred.env")], check=True, capture_output=True, text=True).stdout
    assert dict(x.split("=", 1) for x in said.splitlines()) == {
        "K": "7", "PRODUCT": "remote", "V3C_CLASS": W63K4, "V3C_LAW": "stratified:1", "H1_TIE": "1", "COMPOSITE": "1",
        "ALPHA": "", "GAMMA": "", "SERVED_NODE": "vy-dc-h200", "DEFERRED_TREE": SHA,
        "CR_SCAN_DELTA": "5.421010862427522e-20", "ONSITE": "", "CERTIFIER_CHAIN_CMD": "", "CERTIFIER_PERIOD_CMD": "",
        "WINDOW_STEPS": "16", "MAX_TOKENS": "4", "OF_RECORD": ""}
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


def _follow(tmp_path: Path, composite: str = FAKE_SERVED_COMPOSITE, fresh: bool = False, **extra: str | None
            ) -> tuple[int, list[str], list[str], list[str]]:
    """The follow-on of `_deferred`'s deferral: `served_verdicts.sh v3c` with D alone (no K, class or certifier) on
    cpu-6, a node with V3c's audit's memory, at the served node's tree, in `_v3c`'s toy tree (``fresh``: a fresh
    clone's) with a record and a composite step (``composite``'s body) that note their arguments (``extra`` adds to the
    environment, None unsetting): `_v3c`'s outcome."""
    _script(tmp_path / "tree" / "benchmarks" / "one_stage" / "served_composite.sh", composite)
    (tmp_path / "tree" / "benchmarks" / "one_stage" / "served_verdicts.py").write_text(FAKE_RECORD)
    env = {"K": None, "V3C_CLASS": None, "CERTIFIER_CHAIN_CMD": None, "CERTIFIER_PERIOD_CMD": None,
           "PATH": f"{_hosts(tmp_path)}:{tmp_path / 'bin'}:{os.defpath}", "HOST": "cpu-6", "NODE_MEM_GB": "755",
           "RESEARCH_SOURCE_SHA": SHA, "GIT_CEILING_DIRECTORIES": str(tmp_path)}
    return _v3c(tmp_path, fresh=fresh, **{**env, **extra})


@pytest.mark.parametrize("extra, tree", [
    ({}, "(this tree's)"),
    ({"RESEARCH_SOURCE_SHA": "cd" * 20, "FOLLOW_ANY_TREE": "1"}, f"(this tree is {'cd' * 20}: FOLLOW_ANY_TREE)"),
])
def test_the_follow_on_holds_the_copy_to_the_served_nodes_sums_then_audits_and_composes_the_final_record_there(
        tmp_path, extra, tree):
    """The follow-on, at the settings the deferral recorded: the copy held to the served node's sums, then no window
    circuit or commit of its own (the served node's record at the class, its tree's, and the commit H1 bound, held as
    there), the COINS line of its own audit's coins (its $OUT holds no other), V3c's audit and period, the final record
    about the served node from the pieces it read, and the composite, told the served node, so each says it composed
    elsewhere; no onsite record, which the served node's run didn't write. Another tree runs only under
    FOLLOW_ANY_TREE, which its line names. $OUT names no window."""
    assert _deferred(tmp_path).returncode == 0
    rc, lines, order, limits = _follow(tmp_path, **extra)
    d, out = tmp_path / "serve", tmp_path / "out"
    assert rc == 0 and order == ["audit", "period", "record", "composite"], (lines, order)
    assert lines[:3] == [
        "FOLLOW-ON on cpu-6, 755 GB: V3c's audit and the composite, which the served node vy-dc-h200 deferred, from its "
        "serve dir's audit inputs copied here prover-side, each as it hashed them (the serve dir's "
        "private/deferred/inputs.sha256): PASS",
        f"WINDOW-CIRCUIT class {W63K4}: the served node's record at the class (step 3's), from its tree's commit {SHA} "
        f"{tree}; this node holds no row dir to count it from",
        f"COINS sampled proofs: {FOLLOW_COINS}; the other protocols' as the served node's COINS line names them"]
    assert lines[3].startswith(f"WINDOW-COMMIT class {W63K4} at H1's close") and lines[4].endswith("PASS")
    assert [x.split(" ")[0] for x in lines[5:]] == ["V3C-AUDIT", "V3C-PERIOD", "RECORD", "COMPOSITE:"]
    assert lines[-1] == "COMPOSITE: composed on cpu-6" and limits == ["180m"]
    assert _audited(tmp_path) == {"--law": "stratified:1", "--tie": str(d / "h1" / "w7" / "registration.json"),
                                  "--tie-prefix": "h1/", "--ports": "3"}
    assert json.loads((tmp_path / "order.record").read_text()) == [
        "record", "--serve-run", str(d), "--window", "7", "--sampled-proofs", str(d / f"v3c-w7-{VTAG}" / "audit"),
        "--registration", str(d / f"v3c-w7-{VTAG}" / "window" / "registration.json"), "--pieces",
        str(d / "private" / "deferred" / "pieces"), "--out", str(out / "record"), "--product", "remote", "--node",
        "vy-dc-h200"]
    assert (tmp_path / "order.composite").read_text().split() == ["vy-dc-h200", "remote", str(d), str(out)]
    assert not (out / "registration.json").exists() and not _window_data(out, 7), _window_data(out, 7)


UNTIED_READS = ("weights_registration.json", f"window-circuit-w7-{VTAG}.json", f"window-commit-{VTAG}.json",
                "windows.jsonl", "seals.jsonl", "requests.jsonl", "private/pouw/period/case.json",
                "private/pouw/period/answer.json")


def test_an_untied_deferral_hands_the_follow_on_the_window_commit_step_6_made_on_the_served_node(tmp_path):
    """Untied, a node short of V3c's audit's memory defers it as a tied one does, at the untied law (subset:1): the
    follow-on's inputs are the window commit step 6 made here (its answer and its window dir) and the verifier's untied
    record, in place of the tie's commit and H1's records, listed and hashed alike."""
    run = _deferred(tmp_path, tied=False)
    assert run.returncode == 0, run.stderr
    dd = tmp_path / "serve" / "private" / "deferred"
    said = subprocess.run(["bash", "-c", '. "$1"; printf "%s|%s" "$H1_TIE" "$V3C_LAW"', "_", str(dd / "deferred.env")],
                          check=True, capture_output=True, text=True).stdout
    assert said == "|subset:1"
    window = [f"v3c-w7-{VTAG}/window/{f}" for f in ("circuit.json", "registration.json", "values/a.bin")]
    own = ["private/deferred/deferred.env", "private/deferred/pieces/pouw/answer.json"]
    assert (dd / "inputs.txt").read_text().splitlines() == [*sorted([*UNTIED_READS, *window, *own]),
                                                            "private/deferred/inputs.sha256"]


@pytest.mark.parametrize("copied", [True, False])
def test_an_untied_follow_on_holds_the_served_nodes_step_6_commit_and_audits_it_untied(tmp_path, copied):
    """Untied, the follow-on makes no window commit either, since its node holds no row dir or advice: it holds the
    copy's, the one step 6 made on the served node, to the verifier's untied record, then audits it at subset:1 with no
    tie and composes the final record there. A copy without that commit fails its check before any audit."""
    assert _deferred(tmp_path, tied=False).returncode == 0
    if not copied:
        (tmp_path / "serve" / f"window-commit-{VTAG}.json").unlink()
    rc, lines, order, limits = _follow(tmp_path)
    if not copied:
        assert (rc, lines, order) == (3, [FOLLOW_FAIL], []), lines
        return
    assert rc == 0 and order == ["audit", "period", "record", "composite"] and limits == ["180m"], (lines, order)
    assert lines[3].startswith(f"WINDOW-COMMIT class {W63K4} on the served node vy-dc-h200 at step 6, its copy held "
                               "to that node's sums: ") and '"run": "ab"' in lines[3], lines
    assert lines[4] == (f"WINDOW-CIRCUIT-HELD class {W63K4}: the committed circuit.json less its run fields is the "
                        "verifier's own window-circuit record: PASS")
    assert _audited(tmp_path) == {"--law": "subset:1", "--tie": None, "--tie-prefix": None, "--ports": "16"}


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
    (None, {"WINDOW_STEPS": "1", "MAX_TOKENS": "4"}, f"{CLASH} WINDOW_STEPS: unset it"),
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
    ({**TIES, "NODE_MEM_GB": "196", "PRODUCT": "remote-network"},
     f"196 GB: the serve's lease 150 min, capped at 160 GB; {SHORT}, {DEFERS}"),
    ({**TIES, "NODE_MEM_GB": "196", "COMPOSITE": ""},
     f"196 GB: the serve's lease 150 min, capped at 160 GB; {SHORT}, so step 6 defers V3c's audit to a follow-on"),
    ({"NODE_MEM_GB": "196", "PRODUCT": "remote-network"},
     f"196 GB: the serve's lease 60 min, capped at 128 GB; {SHORT}, {DEFERS}"),
])
def test_preflight_only_says_what_the_run_would_do_and_stops_before_anything_is_kept_built_or_served(
        tmp_path, extra, said):
    """PREFLIGHT_ONLY=1 runs the preflight and stops there, exit 0, with one line: the serve's lease (under H1_TIE 150
    minutes capped at 160 GB, the serve's 128 and window-commit's peak, within a 196 GB node) and, on a node short of
    V3c's audit's memory, tied or not, its deferral: of the audit alone where COMPOSITE is set empty, under which onsite, whose
    composite reads the card's cases, defers too. Nothing runs, and neither the serve root nor C-Flock's work dir is
    made."""
    rc, lines, order, limits = _v3c(tmp_path, (), PATH=f"{_hosts(tmp_path)}:{tmp_path / 'bin'}:{os.defpath}",
                                    HOST="vy-dc-h200-2", PREFLIGHT_ONLY="1", **extra)
    assert (rc, lines, order, limits) == (0, [f"PREFLIGHT passed on vy-dc-h200-2, {said}. {ONLY}"], [], [])
    assert not (tmp_path / "root").exists() and not (tmp_path / "fw").exists()


@pytest.mark.parametrize("total, free, refused", [
    ("2000000", "499999", True), ("2000000", "500000", False), ("0", "0", False), ("-", "-", False)])
def test_the_preflight_refuses_a_serve_whose_serve_dirs_filesystem_has_fewer_free_inodes_than_min_free_inodes(
        tmp_path, total, free, refused):
    """Step 6 commits the class window into the serve dir, an inode a file, after the whole serve, so a filesystem short
    of inodes would fail the run only there: the preflight reads the free inodes of the serve dir's nearest existing
    directory and refuses under MIN_FREE_INODES (default 500000), the fix in its line, before anything is made. A
    filesystem with no fixed inode table (a total of 0 or -) has no floor."""
    _script(tmp_path / "df" / "df", f'echo "$2" >> {tmp_path}/df-at\nprintf " Inodes IFree\\n%s %s\\n" "{total}" "{free}"')
    rc, lines, order, _limits = _v3c(
        tmp_path, (), PATH=f"{tmp_path / 'df'}:{_hosts(tmp_path)}:{tmp_path / 'bin'}:{os.defpath}", HOST="vy-dc-h200-2",
        PREFLIGHT_ONLY="1", D=str(tmp_path / "fresh" / "run"), MIN_FREE_INODES=None)
    assert (tmp_path / "df-at").read_text().split() == [str(tmp_path)]
    line = (f"PREFLIGHT: the serve commits its class window into the serve dir, an inode a file, and {tmp_path}'s "
            "filesystem has 499999 free inodes, under MIN_FREE_INODES 500000: remove old serve dirs' class windows (their "
            "v3c-w* dirs, once each serve dir has its copy and its run its custody), or serve on a node with the inodes")
    passed = f"PREFLIGHT passed on vy-dc-h200-2, 512 GB: the serve's lease 60 min, capped at 128 GB. {ONLY}"
    assert (rc, lines, order) == ((3, [line], []) if refused else (0, [passed], []))
    assert not (tmp_path / "root").exists() and not (tmp_path / "fw").exists()


def test_preflight_only_on_a_follow_on_names_the_deferral_it_would_finish(tmp_path):
    """A follow-on under PREFLIGHT_ONLY passes its own preflight (the deferral's settings, the audit's memory, the served
    node's tree, its venv synced for the checks that run it) and stops before its copy's check or V3c's audit."""
    assert _deferred(tmp_path).returncode == 0
    rc, lines, order, _limits = _follow(tmp_path, PREFLIGHT_ONLY="1")
    assert (rc, lines, order) == (0, [f"PREFLIGHT passed on cpu-6, 755 GB: the follow-on of vy-dc-h200's deferral, its "
                                      f"venv synced for the checks. {ONLY}"], [])


SYNC = "sync --all-packages --extra torch-cpu"
SYNC_FAIL = (f"PREFLIGHT: the follow-on's venv, whose python its checks run, didn't sync (uv {SYNC}): its log is "
             "sync.log")


@pytest.mark.parametrize("synced", [True, False])
def test_a_follow_on_in_a_fresh_clone_syncs_its_venv_before_its_preflight_runs_its_python(tmp_path, synced):
    """A follow-on runs in a fresh clone of the served node's tree, whose venv (VENV unset: the tree's) doesn't exist
    until it syncs, and its preflight runs that venv's python (ℓ_a, `advice_cap`): it syncs the venv first, then
    audits and composes as any follow-on does, the background build's sync after the preflight's. A sync that fails
    stops it in its preflight, exit 3, its first line naming the sync, before anything else runs."""
    assert _deferred(tmp_path).returncode == 0
    rc, lines, order, _limits = _follow(tmp_path, fresh=True, VENV=None, FAIL_SYNC=None if synced else "1")
    syncs = (tmp_path / "uv").read_text().splitlines() if (tmp_path / "uv").exists() else []
    if not synced:
        assert (rc, lines[:1], order, syncs) == (3, [SYNC_FAIL], [], [SYNC]), lines
        assert not [x for x in lines if x.startswith(("FOLLOW-ON", "PREFLIGHT: no"))], lines
        return
    assert rc == 0 and order == ["audit", "period", "record", "composite"], (lines, order)
    assert syncs == [SYNC, SYNC] and lines[0].startswith("FOLLOW-ON on cpu-6") and lines[-1] == (
        "COMPOSITE: composed on cpu-6")


def test_a_deferral_carries_the_served_nodes_inference_only_verdict_into_the_follow_ons_record(tmp_path):
    """The inference-only verdict the served node judged goes with the deferral, listed and hashed with the follow-on's
    inputs, and into the follow-on's record as the served node wrote it, said before the record: the follow-on judges it
    no second time."""
    verdict = {"format": "verity-bench/inference-only-verdict/v3", "verdict": "accept", "classes": [["ab", 7, 3]]}
    (tmp_path / "served-out" / "record").mkdir(parents=True)
    (tmp_path / "served-out" / "record" / "inference-only.json").write_text(json.dumps(verdict))
    assert _deferred(tmp_path).returncode == 0
    dd = tmp_path / "serve" / "private" / "deferred"
    assert "private/deferred/inference-only.json" in (dd / "inputs.txt").read_text().splitlines()
    assert json.loads((dd / "inference-only.json").read_text()) == verdict
    rc, lines, order, _limits = _follow(tmp_path)
    assert rc == 0 and order == ["audit", "period", "record", "composite"], (lines, order)
    assert [x.split(" ")[0] for x in lines[5:]] == ["V3C-AUDIT", "V3C-PERIOD", "INFERENCE-ONLY:", "RECORD",
                                                    "COMPOSITE:"]
    assert json.loads((tmp_path / "out" / "record" / "inference-only.json").read_text()) == verdict


def test_a_deferral_carries_the_advice_registrations_public_form_into_the_follow_ons_out(tmp_path):
    """The advice registration's public form the served node published ($OUT/registration.json, whose cap the onsite
    record charges) goes with the deferral, listed and hashed with the follow-on's inputs, and into the follow-on's $OUT
    as the served node wrote it."""
    public = S.public_registration(REGISTRATION)
    (tmp_path / "served-out").mkdir()
    (tmp_path / "served-out" / "registration.json").write_text(json.dumps(public))
    assert _deferred(tmp_path).returncode == 0
    dd = tmp_path / "serve" / "private" / "deferred"
    assert "private/deferred/registration.json" in (dd / "inputs.txt").read_text().splitlines()
    assert "private/deferred/registration.json" in (dd / "inputs.sha256").read_text()
    rc, lines, order, _limits = _follow(tmp_path)
    assert rc == 0 and order == ["audit", "period", "record", "composite"], (lines, order)
    assert json.loads((tmp_path / "out" / "registration.json").read_text()) == public


FAKE_ONSITE_RECORD = '''import json, os, sys
open(os.environ["ORDER"], "a").write("onsite\\n")
open(os.environ["ORDER"] + ".onsite", "w").write(json.dumps(sys.argv[1:]))
print("ONSITE " + json.dumps({"written": "record/onsite.json"}))
'''
REPROOF = 'echo reproof >> "$ORDER"; printf "%s\\n" "$@" > "$ORDER.reproof"; echo "REPROOF COPY: {said}"; exit {rc}'


@pytest.mark.parametrize("rc, said", [(0, "PASS"), (4, "FAIL (rsync)")])
def test_the_follow_on_runs_reproof_cmd_after_the_audit_then_the_onsite_record_with_its_copy_where_the_served_run_would(
        tmp_path, rc, said):
    """A deferral of a run that writes the onsite record (its certifier's front or piece set) records ONSITE, and its
    follow-on runs REPROOF_CMD once V3c's audit is done, with the serve dir, the window and V3c's dir, its log in the
    serve dir's private/ and its last line the REPROOF line; then the record and the composite, and last the onsite
    record at the served run's WINDOW_STEPS and MAX_TOKENS, not the follow-on's defaults, with REPROOF_COPY and
    REPROOF_LISTING, the auditor's inputs its checks read (CERTIFIER_KEY and SITE_REGISTRATION, in the follow-on's
    environment), and --h1-tie, the deferral's being H1-tied, before $OUT is published. A copy that fails fails the
    follow-on, and every record is still written."""
    assert _deferred(tmp_path, CERTIFIER_FRONT="front", WINDOW_STEPS="1", MAX_TOKENS="1").returncode == 0
    d, out, listing = tmp_path / "serve", tmp_path / "out", tmp_path / "serve" / "private" / "reproof-listing.txt"
    key, site = tmp_path / "auditor" / "certifier-key.json", tmp_path / "auditor" / "site-registration.json"
    (tmp_path / "tree" / "benchmarks" / "one_stage").mkdir(parents=True, exist_ok=True)
    (tmp_path / "tree" / "benchmarks" / "one_stage" / "onsite_record.py").write_text(FAKE_ONSITE_RECORD)
    got, lines, order, _limits = _follow(tmp_path, REPROOF_CMD=REPROOF.format(said=said, rc=rc),
                                         REPROOF_COPY="vy-nebius-cpu-3:/p/r1", REPROOF_LISTING=str(listing),
                                         CERTIFIER_KEY=str(key), SITE_REGISTRATION=str(site))
    assert (got == 0) == (rc == 0) and order == ["audit", "period", "reproof", "record", "composite", "onsite"], order
    assert (tmp_path / "order.reproof").read_text().split() == [str(d), "7", str(d / f"v3c-w7-{VTAG}")]
    assert f"REPROOF (exit {rc}): REPROOF COPY: {said}" in lines and lines[-1].startswith("ONSITE ")
    assert oct((d / "private" / "reproof.log").stat().st_mode & 0o777) == "0o600"
    assert json.loads((tmp_path / "order.onsite").read_text()) == [
        str(out), "--window-steps", "1", "--max-tokens", "1", "--reproof-copy", "vy-nebius-cpu-3:/p/r1",
        "--reproof-listing", str(listing), "--certifier-key", str(key), "--site-registration", str(site), "--h1-tie"]


def test_the_onsite_record_is_handed_the_inference_only_cases_policy_where_the_composite_wrote_the_case(tmp_path):
    """Where the composite step wrote the inference-only case under the serve dir's private/, the onsite record is
    handed the case's private policy.json, where served_cases.py --from-division says α and Γ came from, whether or not
    it exists (with ALPHA and GAMMA set it doesn't); without the case it is handed none (above)."""
    assert _deferred(tmp_path, CERTIFIER_FRONT="front", WINDOW_STEPS="1", MAX_TOKENS="1").returncode == 0
    d, out = tmp_path / "serve", tmp_path / "out"
    (tmp_path / "tree" / "benchmarks" / "one_stage").mkdir(parents=True, exist_ok=True)
    (tmp_path / "tree" / "benchmarks" / "one_stage" / "onsite_record.py").write_text(FAKE_ONSITE_RECORD)
    case = 'mkdir -p "$1/private/inference-only" && echo {} > "$1/private/inference-only/case.json"'
    rc, lines, order, _limits = _follow(tmp_path, f"{FAKE_SERVED_COMPOSITE}\n{case}")
    assert rc == 0 and order == ["audit", "period", "record", "composite", "onsite"], (lines, order)
    assert json.loads((tmp_path / "order.onsite").read_text()) == [
        str(out), "--window-steps", "1", "--max-tokens", "1",
        "--inference-only-case-policy", str(d / "private" / "inference-only" / "policy.json"), "--h1-tie"]


#: V3c's audit of `FAKE_RECORDS_AUDIT` with its circuits composed: template A's in the warm's cache, linked at its first
#: member m0 and at m2, a drawn member of it; template B's at m1, which V2_CASE makes the file a v2 case names; and m0's
#: opening, which isn't a circuit
COMPOSED_AUDIT = FAKE_RECORDS_AUDIT.replace('    (out / "summary.json").write_text("{}")\n', '''\
    (out / "summary.json").write_text("{}")
    w, cache = out / "window", out.parent / "compose-cache" / "a"
    for p in (w / "m0", w / "m1", w / "m2", cache):
        p.mkdir(parents=True)
    (cache / "circuit.txt").write_text("A" * 10)
    for m in ("m0", "m2"):
        os.link(cache / "circuit.txt", w / m / "circuit.txt")
    (w / "m1" / "circuit.txt").write_text("B" * 4)
    (w / "m0-private.bin").write_bytes(b"\\x01")
    if os.environ.get("V2_CASE"):
        blobs = [str(w / "m1" / "circuit.txt"), str(w / "m0-private.bin")]
        (w / "case-v2.json").write_text(json.dumps({"blobs": blobs}))
        (w / "public-pin.json").write_text("{}")
''')
#: a step that notes the circuit files left under V3c's dir when it runs
LEFT = '(cd "{at}" && find . -name circuit.txt | sort) > "$ORDER.left-{step}"'


@pytest.mark.parametrize("v2, first, last", [
    (None, "4 files, 14 bytes freed", None),
    ("1", "3 files, 10 bytes freed; 1 the v2 case names, after the composite", "1 file, 4 bytes freed"),
])
def test_the_follow_on_removes_the_windows_circuit_files_after_its_period_check_before_the_reproof_copy(
        tmp_path, v2, first, last):
    """Once the period check has read them, the follow-on removes the window's every circuit file under V3c's dir (each
    member's, links and all, and the warm's cache), so the re-proof copy finds none and the node's next window has the
    disk, saying how many and the bytes freed; the openings stay. A file the audit's v2 case names waits for the
    composite, which reads it, and goes after."""
    assert _deferred(tmp_path).returncode == 0
    vw = tmp_path / "serve" / f"v3c-w7-{VTAG}"
    composite = f"{FAKE_SERVED_COMPOSITE}\n{LEFT.format(at=vw, step='composite')}"
    reproof = f'echo reproof >> "$ORDER"; {LEFT.format(at=vw, step="reproof")}; echo "REPROOF COPY: PASS"'
    rc, lines, order, _limits = _follow(tmp_path, composite, records_audit=COMPOSED_AUDIT, V2_CASE=v2,
                                        REPROOF_CMD=reproof)
    assert rc == 0 and order == ["audit", "period", "reproof", "record", "composite"], (lines, order)
    said = "V3C-CIRCUITS removed, the window's circuit files, recomposable and named by their pins: "
    heads = [x.split(" ")[0] for x in lines[5:]]
    assert heads == ["V3C-AUDIT", "V3C-PERIOD", "V3C-CIRCUITS", "REPROOF", "RECORD", "COMPOSITE:",
                     *["V3C-CIRCUITS"] * bool(last)], lines
    assert lines[7] == said + first and (not last or lines[-1] == said + last), lines
    kept = ["./audit/window/m1/circuit.txt"] if v2 else []
    left = [(tmp_path / f"order.left-{s}").read_text().split() for s in ("reproof", "composite")]
    assert left == [kept, kept]
    assert not list(vw.rglob("circuit.txt")) and (vw / "audit" / "window" / "m0-private.bin").read_bytes() == b"\x01"


def test_the_follow_on_hands_the_onsite_record_remotes_sampled_proofs_case_before_removing_the_files_it_registers(
        tmp_path):
    """Where the composite step wrote remote's sampled proofs case under the serve dir's private/, the onsite record is
    handed it (--sampled-proofs-case), whose registered files it hashes, and the audit's pin of its public files at
    staging, before the draw (--public-pin, its window/public-pin.json), so it runs before the follow-on removes the
    circuit file the audit's v2 case names: it finds that file there, and the file goes after it."""
    assert _deferred(tmp_path, CERTIFIER_FRONT="front", WINDOW_STEPS="1", MAX_TOKENS="1").returncode == 0
    d, out, vw = tmp_path / "serve", tmp_path / "out", tmp_path / "serve" / f"v3c-w7-{VTAG}"
    left = LEFT.format(at=vw, step="onsite")
    (tmp_path / "tree" / "benchmarks" / "one_stage").mkdir(parents=True, exist_ok=True)
    (tmp_path / "tree" / "benchmarks" / "one_stage" / "onsite_record.py").write_text(
        f"{FAKE_ONSITE_RECORD}import subprocess\nsubprocess.run(['bash', '-c', {left!r}], check=True)\n")
    case = 'mkdir -p "$1/private/sampled-proofs" && echo {} > "$1/private/sampled-proofs/case.json"'
    rc, lines, order, _limits = _follow(tmp_path, f"{FAKE_SERVED_COMPOSITE}\n{case}", records_audit=COMPOSED_AUDIT,
                                        V2_CASE="1")
    assert rc == 0 and order == ["audit", "period", "record", "composite", "onsite"], (lines, order)
    assert json.loads((tmp_path / "order.onsite").read_text()) == [
        str(out), "--window-steps", "1", "--max-tokens", "1",
        "--sampled-proofs-case", str(d / "private" / "sampled-proofs" / "case.json"),
        "--public-pin", str(vw / "audit" / "window" / "public-pin.json"), "--h1-tie"]
    assert (tmp_path / "order.left-onsite").read_text().split() == ["./audit/window/m1/circuit.txt"]
    assert [x.split(" ")[0] for x in lines[-2:]] == ["ONSITE", "V3C-CIRCUITS"] and not list(vw.rglob("circuit.txt"))


#: the network certifier's kept run in `_deferred`'s serve dir: the gate's fetch and the front's session
KEPT = ("private/certifier/r1/fetch.tar.gz", "private/certifier/r1/session.json")
#: the split certifier's period check, writing the network certifier's case and its answer in its piece dir
NETWORK_PERIOD = ('echo period-check >> "$ORDER"; test -s "$4/period.json" || exit 1; '
                  'echo \'{"commitment": "7b7d", "evidence": "7b7d", "params": {}}\' > "$3/case.json"; '
                  'echo \'{"protocol": "verity/certifier/v9", "verdict": "accept"}\' > "$3/answer.json"; echo checked')


def _kept(tmp_path: Path) -> None:
    """The network certifier's kept run (`KEPT`) and an attack log beside it in `_deferred`'s serve dir."""
    for f in (*KEPT, "private/certifier/attacks/relay.log"):
        (tmp_path / "serve" / f).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "serve" / f).write_text(f)


def _composing_tree(tmp_path: Path) -> Path:
    """`_v3c`'s toy tree with the onsite record's and composite.py's stand-ins and the composites' Lean packages, and a
    directory, for PATH, whose systemd-run fails, so `served_composite.sh` runs each composite outside a scope."""
    tree = tmp_path / "tree"
    (tree / "benchmarks" / "one_stage").mkdir(parents=True, exist_ok=True)
    (tree / "benchmarks" / "one_stage" / "onsite_record.py").write_text(FAKE_ONSITE_RECORD)
    (tree / "benchmarks" / "one_stage" / "composite.py").write_text(FAKE_COMPOSITE)
    for pkg in ("remote", "onsite"):
        (tree / "verity" / "core" / "protocols" / pkg).mkdir(parents=True)
    _script(tmp_path / "stubs" / "systemd-run", "exit 1")
    return tmp_path / "stubs"


#: a run of record's settings on its served node (`served_verdicts.sh`'s OF_RECORD), untied
OF_SERVED = {"OF_RECORD": "1", "PRODUCT": "remote-network-no-pouw", "CERTIFIER_FRONT": "front"}
OF = "of record (OF_RECORD): every input its of_record reads is set"
OF_LACKS = "PREFLIGHT: OF_RECORD makes this run one of record, and it lacks what its onsite record's of_record reads: "
OF_PRODUCT = "PRODUCT=remote-network-no-pouw untied (H1_TIE empty) at V3C_LAW subset:1, not PRODUCT="
OF_SPLIT = ("the split certifier behind its front (CERTIFIER_FRONT, CERTIFIER_CHAIN_CMD and CERTIFIER_PERIOD_CMD), "
            "whose network case the composite reads")
OF_COMPOSITE = "COMPOSITE unset, not empty: the composite is the record's verdict"
OF_COINS = ("PP_CHALLENGER=gate:USER@HOST with PP_PEER and PP_CHALLENGER_KEY (a non-empty file), the live challenger "
            "V3c's audit draws on, without which of_record is conditional on live coins")
OF_KEY = ("CERTIFIER_KEY and SITE_REGISTRATION, each a non-empty file (the auditor's record of its network certifier "
          "and its registration), without which certifier-key stays open")
OF_DRAWS = ("CR_SCAN_DELTA its default 2^-64, and CERTIFIER_CHAIN_CMD naming none of CR_N2_DRAW, CR_N2_CMD, "
            "CR_SCAN_DELTA, CR_SCAN_ARGS, CR_SCAN_CMD, --draw, --chain-cmd, the chain's draws of record, without which "
            "a chain draw can be open (of_record then conditional on live coins) or the scan's bound at another δ")
OF_CHAIN = ("CR_N2_DRAW, CR_N2_CMD and CR_SCAN_CMD unset and no --draw in CR_SCAN_ARGS where the chain runs, without "
            "which a chain draw can be open and of_record conditional on live coins")
#: `FAKE_RECORDS_AUDIT` on the live challenger gate:research@auditor, its session on cpu-6, under `_auditors`' key
#: record
GATE_AUDIT = FAKE_RECORDS_AUDIT.replace(
    '    assert a[a.index("--challenger") + 1] == "local", a\n',
    '    assert [a[a.index(f) + 1] for f in ("--challenger", "--peer")] == ["gate:research@auditor", "cpu-6"], a\n'
    '    assert open(a[a.index("--challenger-key") + 1]).read() == "{}", a\n')


def _auditors(tmp_path: Path, peer: str, **files: str) -> dict[str, str]:
    """What a run of record reads on the node that runs V3c's audit and writes the onsite record: the live challenger,
    its session on ``peer``, and the auditor's records (its challenger's key record, its network certifier's key and its
    site registration), each a file written here holding ``{}`` unless ``files`` names it ``missing`` or ``empty``."""
    auditor = tmp_path / "auditor"
    auditor.mkdir(exist_ok=True)
    env = {"PP_CHALLENGER": "gate:research@auditor", "PP_PEER": peer}
    for v in ("PP_CHALLENGER_KEY", "CERTIFIER_KEY", "SITE_REGISTRATION"):
        f = auditor / f"{v.lower()}.json"
        if files.get(v) != "missing":
            f.write_text("" if files.get(v) == "empty" else "{}")
        env[v] = str(f)
    return env


@pytest.mark.parametrize("extra, said", [
    ({}, f"512 GB: the serve's lease 60 min, capped at 128 GB; {OF}"),
    ({"NODE_MEM_GB": "196", "PP_CHALLENGER": None, "CERTIFIER_KEY": None, "CR_N2_DRAW": "8"},
     f"196 GB: the serve's lease 60 min, capped at 128 GB; {SHORT}, {DEFERS}; {OF}, but the follow-on's, which its own "
     "preflight checks"),
])
def test_a_run_of_record_passes_its_preflight_with_every_input_its_onsite_records_of_record_reads(
        tmp_path, extra, said):
    """OF_RECORD makes a run one of record: untied remote-network-no-pouw behind the network certifier's front, with the
    split certifier drawing everything, the composite, the live challenger and the auditor's records on the node that
    runs V3c's audit, the chain and the onsite record. A node short of V3c's audit's memory defers them to a follow-on,
    which holds the challenger, the auditor's records and the chain's environment in its own preflight, so this node
    needs none of them; PREFLIGHT_ONLY's line says which."""
    env = {**OF_SERVED, **_auditors(tmp_path, "vy-dc-h200-2"), **extra}
    rc, lines, order, limits = _v3c(tmp_path, (), PATH=f"{_hosts(tmp_path)}:{tmp_path / 'bin'}:{os.defpath}",
                                    HOST="vy-dc-h200-2", PREFLIGHT_ONLY="1", **env)
    assert (rc, lines, order, limits) == (0, [f"PREFLIGHT passed on vy-dc-h200-2, {said}. {ONLY}"], [], [])


@pytest.mark.parametrize("extra, files, said", [
    ({"PRODUCT": "remote-network"}, {}, f"{OF_PRODUCT}remote-network at subset:1"),
    (TIES, {}, f"{OF_PRODUCT}remote-network-no-pouw under H1_TIE at stratified:1"),
    ({"V3C_LAW": "stratified:1"}, {}, f"{OF_PRODUCT}remote-network-no-pouw at stratified:1"),
    ({"CERTIFIER_FRONT": None}, {}, OF_SPLIT),
    ({"CERTIFIER_CHAIN_CMD": None, "CERTIFIER_PERIOD_CMD": None}, {}, OF_SPLIT),
    ({"COMPOSITE": ""}, {}, OF_COMPOSITE),
    ({"PP_CHALLENGER": "local"}, {}, OF_COINS),
    ({"PP_PEER": None}, {}, OF_COINS),
    ({}, {"PP_CHALLENGER_KEY": "missing"}, OF_COINS),
    ({"CERTIFIER_KEY": None}, {}, OF_KEY),
    ({}, {"SITE_REGISTRATION": "empty"}, OF_KEY),
    ({"CR_SCAN_DELTA": "1e-06"}, {}, OF_DRAWS),
    ({"CERTIFIER_CHAIN_CMD": f"{CHAIN} CR_N2_DRAW=8"}, {}, OF_DRAWS),
    ({"CERTIFIER_CHAIN_CMD": f"{CHAIN} --chain-cmd true"}, {}, OF_DRAWS),
    ({"CR_N2_DRAW": "8"}, {}, OF_CHAIN),
    ({"CR_SCAN_ARGS": "--group 2 --draw 8"}, {}, OF_CHAIN),
    ({"CR_SCAN_CMD": "true"}, {}, OF_CHAIN),
    ({"COMPOSITE": "", "PP_PEER": None}, {"CERTIFIER_KEY": "empty"}, f"{OF_COMPOSITE}; {OF_COINS}; {OF_KEY}"),
])
def test_the_preflight_refuses_a_run_of_record_lacking_any_input_its_onsite_records_of_record_reads(
        tmp_path, extra, files, said):
    """A run of record lacking any input its onsite record's of_record reads is refused before anything is kept, built
    or served, with one line naming each it lacks: another product, H1's tie or another law; the network certifier's
    front or the split certifier; the composite set empty; a chain draw short of everything, by CR_SCAN_DELTA or
    CERTIFIER_CHAIN_CMD's text, or by the chain's environment here; the live challenger, its peer or its key record; or
    the auditor's record of its network certifier or its registration, missing or empty."""
    env = {**OF_SERVED, **_auditors(tmp_path, "vy-dc-h200-2", **files), **extra}
    rc, lines, order, limits = _v3c(tmp_path, (), PATH=f"{_hosts(tmp_path)}:{tmp_path / 'bin'}:{os.defpath}",
                                    HOST="vy-dc-h200-2", PREFLIGHT_ONLY="1", **env)
    assert (rc, lines, order, limits) == (3, [OF_LACKS + said], [], [])
    assert not (tmp_path / "root").exists() and not (tmp_path / "fw").exists()


def _deferred_of_record(tmp_path: Path) -> subprocess.CompletedProcess:
    """`_deferred`'s deferral of a run of record: untied remote-network-no-pouw behind the network certifier's front,
    its kept run in the serve dir, the split certifier's commands set, at one engine step a window and one token a
    request."""
    _kept(tmp_path)
    return _deferred(tmp_path, tied=False, **OF_SERVED, CERTIFIER_CHAIN_CMD=CHAIN, CERTIFIER_PERIOD_CMD=NETWORK_PERIOD,
                     WINDOW_STEPS="1", MAX_TOKENS="1")


@pytest.mark.parametrize("extra, files, line", [
    ({"PP_CHALLENGER": None}, {}, OF_LACKS + OF_COINS),
    ({"PP_PEER": None}, {}, OF_LACKS + OF_COINS),
    ({}, {"PP_CHALLENGER_KEY": "missing"}, OF_LACKS + OF_COINS),
    ({"CERTIFIER_KEY": None}, {}, OF_LACKS + OF_KEY),
    ({"SITE_REGISTRATION": None}, {}, OF_LACKS + OF_KEY),
    ({"CR_N2_DRAW": "8"}, {}, OF_LACKS + OF_CHAIN),
    ({"CR_SCAN_ARGS": "--draw 8"}, {}, OF_LACKS + OF_CHAIN),
    ({"OF_RECORD": ""}, {}, f"{CLASH} OF_RECORD: unset it"),
])
def test_a_run_of_records_follow_on_refuses_to_run_without_the_live_challenger_or_the_auditors_records(
        tmp_path, extra, files, line):
    """The follow-on of a run of record runs V3c's audit, the certifier's chain and the onsite record, so its own
    preflight refuses it without the live challenger, its peer or its key record, or the auditor's record of its
    network certifier or its registration, or with a chain environment that draws short of everything, before its
    copy's check or V3c's audit; and it runs of record as the deferral was, so an environment that says otherwise is
    refused as any other setting is."""
    assert _deferred_of_record(tmp_path).returncode == 0
    rc, lines, order, _limits = _follow(tmp_path, **{**_auditors(tmp_path, "cpu-6", **files), **extra})
    assert (rc, lines, order) == (3, [line], []), lines


def test_a_run_of_record_defers_untied_and_its_follow_on_writes_the_onsite_record_at_remote_network_no_pouw(tmp_path):
    """Untied remote-network-no-pouw behind the network certifier's front, of record, on a node short of V3c's audit's
    memory: the deferral records OF_RECORD with the product, the untied law and the split certifier's commands, and the
    follow-on, on the live challenger with the auditor's records, says whose coins its audit draws on (the
    challenger's), runs V3c's audit at subset:1 with no tie on the challenger through its peer under its key record,
    the chain and the certifier's period check after it, the record at the product on the served node's pieces and the
    certifier's piece run here, remote-network-no-pouw's composite alone over the network certifier's case checked
    here, and the onsite record last, at the served run's WINDOW_STEPS and MAX_TOKENS with the auditor's records and no
    --h1-tie."""
    run = _deferred_of_record(tmp_path)
    assert run.returncode == 0, run.stderr
    d, dd, out = tmp_path / "serve", tmp_path / "serve" / "private" / "deferred", tmp_path / "out"
    names = "OF_RECORD PRODUCT H1_TIE V3C_LAW ONSITE"
    said = subprocess.run(["bash", "-c", f'. "$1"; for v in {names}; do printf "%s=%s\\n" "$v" "${{!v}}"; done', "_",
                           str(dd / "deferred.env")], check=True, capture_output=True, text=True).stdout
    assert dict(x.split("=", 1) for x in said.splitlines()) == {
        "OF_RECORD": "1", "PRODUCT": "remote-network-no-pouw", "H1_TIE": "", "V3C_LAW": "subset:1", "ONSITE": "1"}
    stubs = _composing_tree(tmp_path)
    auditors = _auditors(tmp_path, "cpu-6")
    rc, lines, order, limits = _follow(tmp_path, composite=(HERE.parent / "served_composite.sh").read_text(),
                                       records_audit=GATE_AUDIT,
                                       PATH=f"{_hosts(tmp_path)}:{stubs}:{tmp_path / 'bin'}:{os.defpath}", **auditors)
    pieces = out / "pieces" / "certifier"
    assert rc == 0 and order == ["audit", "period", "chain", "period-check", "pieces", "record",
                                 "composite-remote-network-no-pouw", "onsite"], (lines, order)
    assert limits == ["180m", "4h", "12h", "60m"]
    assert ("COINS sampled proofs: the challenger gate:research@auditor (peer cpu-6); the other protocols' as the "
            "served node's COINS line names them") in lines
    assert _audited(tmp_path) == {"--law": "subset:1", "--tie": None, "--tie-prefix": None, "--ports": "16"}
    record = json.loads((tmp_path / "order.record").read_text())
    assert (record[record.index("--pieces") + 1], record[record.index("--product") + 1]) == (
        str(pieces.parent), "remote-network-no-pouw")
    assert json.loads((pieces / "answer.json").read_text()) == {"protocol": "verity/certifier/v9", "verdict": "accept"}
    handed = json.loads((d / "private" / "composite" / "remote-network-no-pouw" / "private" / "argv.json").read_text())
    assert handed == [str(pieces)]
    assert sorted(f.relative_to(out).as_posix() for f in (out / "composite").rglob("*") if f.is_file()) == [
        "composite/remote-network-no-pouw/composite.json"]
    assert json.loads((tmp_path / "order.onsite").read_text()) == [
        str(out), "--window-steps", "1", "--max-tokens", "1", "--certifier-key", auditors["CERTIFIER_KEY"],
        "--site-registration", auditors["SITE_REGISTRATION"]]
    assert lines[-1].startswith("ONSITE ") and not _window_data(out, 7), lines



@pytest.mark.parametrize("product", ["remote", "remote-network"])
def test_a_deferral_with_the_split_certifier_behind_its_front_leaves_the_chain_and_its_period_check_to_the_follow_on(
        tmp_path, product):
    """With the network certifier in front, remote takes the split certifier's two commands, whose period case the onsite
    record reads (remote's record drops the certifier), and so does remote-network, whose composite is the onsite
    record's verdict. Its deferral carries both and the kept run the chain reads,
    listed and hashed, and nothing else of the certifier's private/; the follow-on runs the chain after V3c's audit and
    period rather than beside them (its node has the audit's memory alone), then the certifier's period check on that
    period.json into its own $OUT/pieces/certifier, the certifier's piece there as step 4 runs it (`true`, reading that
    answer), about the served node (``--served``), the record on the served node's pieces with that piece (remote-network's
    record reads it), then `served_composite.sh` after that period check, told the served
    node, and the onsite record last. remote-network's follow-on writes $OUT/composite/remote-network/composite.json,
    the onsite record's verdict, from the network certifier's case checked here; remote's writes remote's alone."""
    d, tree = tmp_path / "serve", tmp_path / "tree"
    for f in (*KEPT, "private/certifier/attacks/relay.log"):
        (d / f).parent.mkdir(parents=True, exist_ok=True)
        (d / f).write_text(f)
    period = ('echo period-check >> "$ORDER"; test -s "$4/period.json" || exit 1; '
              'echo \'{"commitment": "7b7d", "evidence": "7b7d", "params": {}}\' > "$3/case.json"; '
              'echo \'{"protocol": "verity/certifier/v9", "verdict": "accept"}\' > "$3/answer.json"; echo checked')
    run = _deferred(tmp_path, PRODUCT=product, CERTIFIER_FRONT="front", CERTIFIER_CHAIN_CMD=CHAIN,
                    CERTIFIER_PERIOD_CMD=period)
    assert run.returncode == 0, run.stderr
    dd = d / "private" / "deferred"
    assert [f for f in (dd / "inputs.txt").read_text().splitlines() if f.startswith("private/certifier/")] == list(KEPT)
    assert all(f in (dd / "inputs.sha256").read_text() for f in KEPT)
    (tree / "benchmarks" / "one_stage").mkdir(parents=True, exist_ok=True)
    (tree / "benchmarks" / "one_stage" / "onsite_record.py").write_text(FAKE_ONSITE_RECORD)
    (tree / "benchmarks" / "one_stage" / "composite.py").write_text(FAKE_COMPOSITE)
    for pkg in ("remote", "onsite"):
        (tree / "verity" / "core" / "protocols" / pkg).mkdir(parents=True)
    _script(tmp_path / "stubs" / "systemd-run", "exit 1")
    rc, lines, order, limits = _follow(tmp_path, composite=(HERE.parent / "served_composite.sh").read_text(),
                                       PATH=f"{_hosts(tmp_path)}:{tmp_path / 'stubs'}:{tmp_path / 'bin'}:{os.defpath}")
    out, pieces = tmp_path / "out", tmp_path / "out" / "pieces" / "certifier"
    products = {"remote": ["remote"], "remote-network": ["remote", "remote-network"]}[product]
    assert rc == 0 and order == ["audit", "period", "chain", "period-check", "pieces", "record",
                                 *(f"composite-{x}" for x in products), "onsite"], (lines, order)
    assert limits == ["180m", "4h", "12h", *["60m"] * len(products)]
    assert (tmp_path / "order.args").read_text().split() == [str(d), "7", str(pieces), str(d / f"v3c-w7-{VTAG}")]
    assert json.loads((tmp_path / "order.pieces").read_text()) == [
        "pieces", "--serve-run", str(d), "--window", "7", "--out", str(pieces.parent), "--only", "certifier",
        "--served", "vy-dc-h200", "--product", product]
    assert (tmp_path / "order.pieces-cmd").read_text() == "true"
    assert json.loads((pieces / "answer.json").read_text()) == {"protocol": "verity/certifier/v9", "verdict": "accept"}
    record = json.loads((tmp_path / "order.record").read_text())
    assert (record[record.index("--pieces") + 1], record[record.index("--product") + 1]) == (
        str(pieces.parent), product)
    assert (pieces.parent / "pouw" / "answer.json").read_text() == (dd / "pieces" / "pouw" / "answer.json").read_text()
    assert sorted(f.relative_to(out).as_posix() for f in (out / "composite").rglob("*") if f.is_file()) == sorted(
        f"composite/{x}/composite.json" for x in products)
    handed = {x: json.loads((d / "private" / "composite" / x / "private" / "argv.json").read_text()) for x in products}
    assert handed == ({"remote": [], "remote-network": [str(pieces)]} if product == "remote-network" else {"remote": []})
    assert ("COMPOSITE: composed on cpu-6, not the served node vy-dc-h200: from a prover-side copy of its serve dir's "
            "audit inputs and the cases written here from it, not from the parties that produced them") in lines
    assert not (pieces / "chain.log").exists() and (d / "private" / "pieces" / "certifier" / "period.log").exists()


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
    ("front", FRONT, "--gate --warmup --bucket-ms", "--seal-hold-gb 8", "PREFLIGHT: the network certifier in front of the "
     "endpoint sends id-form requests, which serve-openai answers only under SERVE_ARGS='--protocol ids --id-payload 503'"),
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


#: the toy tree's prove_path.py with `window-class`: its argv to ORDER.class, refused unless the class is the one whose
#: window the chain's scan reads at the scan class (the front's w63k4 unless given)
WINDOW_CLASS = f'''import os, sys
a = sys.argv[1:]
if a[0] == "window-class":
    open(os.environ["ORDER"] + ".class", "w").write(" ".join(a[1:]))
    fit = {{"w63k4": "{W63K4}", "w63": "8,1024,64,4,16"}}[a[a.index("--scan-class") + 1] if "--scan-class" in a else "w63k4"]
    if a[a.index("--class") + 1] != fit:
        sys.exit(f"prove_path.py window-class: not the class of the scan's link-window: set the class {{fit}}")
    print("{{}}")
    sys.exit(0)
''' + FAKE_V3C_PROVE_PATH
PREFLIGHT_PASSED = f"PREFLIGHT passed on vy-dc-h200-2, 512 GB: the serve's lease 60 min, capped at 128 GB. {ONLY}"
UNLINKED = ("PREFLIGHT: V3C_CLASS {} behind the certifier front: prove_path.py window-class: not the class of the scan's "
            "link-window: set the class {}")


@pytest.mark.parametrize("extra, argv, line", [
    ({"CERTIFIER_FRONT": FRONT}, f"--class {W63K4}", PREFLIGHT_PASSED),
    ({"CERTIFIER_FRONT": FRONT, "V3C_CLASS": "4,32,4", "CLASSES": "4,32,4"}, "--class 4,32,4",
     UNLINKED.format("4,32,4", W63K4)),
    ({"CERTIFIER_FRONT": f"{FRONT} --scan-class w63", "V3C_CLASS": "8,1024,64,4,16", "CLASSES": "8,1024,64,4,16"},
     "--class 8,1024,64,4,16 --scan-class w63", PREFLIGHT_PASSED),
    ({"CERTIFIER_FRONT": f"{FRONT} --scan-class=w63"}, f"--class {W63K4} --scan-class w63",
     UNLINKED.format(W63K4, "8,1024,64,4,16")),
    ({"CERTIFIER_FRONT": f'{FRONT} --scan-class "$SCAN"'}, None,
     "PREFLIGHT: CERTIFIER_FRONT passes --scan-class with no class this preflight reads: name it"),
    ({"CERTIFIER_FRONT": "front", "V3C_CLASS": "4,32,4", "CLASSES": "4,32,4"}, None, PREFLIGHT_PASSED),
    ({"CLASSES": "4,32,4"}, None, "PREFLIGHT: untied, V3c's window commit is held to the window-circuit record step 3 "
     f"writes at each class of CLASSES, and CLASSES '4,32,4' doesn't hold V3C_CLASS {W63K4}: set CLASSES to hold it"),
    ({"CLASSES": f"record 4,32,4 {W63K4}"}, None, PREFLIGHT_PASSED),
    ({**TIES, "CLASSES": "4,32,4"}, None,
     f"PREFLIGHT passed on vy-dc-h200-2, 512 GB: the serve's lease 150 min, capped at 160 GB. {ONLY}"),
])
def test_the_preflight_holds_v3c_class_to_the_fronts_link_window_and_an_untied_runs_classes_to_hold_it(
        tmp_path, extra, argv, line):
    """Behind this tree's certifier front (CERTIFIER_FRONT naming `certifier_piece.py front`), V3C_CLASS is held to the
    one class whose window the chain's scan reads at the front's scan class, its `--scan-class` or w63k4
    (`prove_path.py window-class`), before the lease: window-commit's scan refuses any other only after the serve. A
    front that isn't this tree's is not held to it, and one whose --scan-class names no class this preflight reads is
    refused. An untied run's CLASSES must hold V3C_CLASS, since step 6 holds V3c's commit to the record step 3 writes at
    each of them; a tied run writes its own (`v3c_circuit`)."""
    (tmp_path / "tree" / "benchmarks" / "network_traces").mkdir(parents=True)
    (tmp_path / "tree" / "benchmarks" / "network_traces" / "certifier_piece.py").write_text(
        "print('usage: certifier_piece.py front PORT [--gate --warmup --bucket-ms --scan-class]')\n")
    rc, lines, order, _limits = _v3c(tmp_path, (), prove_path=WINDOW_CLASS,
                                     PATH=f"{_hosts(tmp_path)}:{tmp_path / 'bin'}:{os.defpath}", HOST="vy-dc-h200-2",
                                     PREFLIGHT_ONLY="1", SERVE_ARGS=IDS, **extra)
    assert (rc, lines, order) == (0 if ONLY in line else 3, [line], [])
    noted = tmp_path / "order.class"
    assert (noted.read_text() if noted.exists() else None) == argv


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


SPLIT_NO_FRONT = ("PREFLIGHT: PRODUCT=remote runs the split certifier's CERTIFIER_CHAIN_CMD only with the network certifier "
                  "in front, whose period case it writes: set CERTIFIER_FRONT, or unset it")


@pytest.mark.parametrize("product, extra, line", [
    ("remote", {}, None),
    ("remote", {"POUS_CMD": "true"}, "PREFLIGHT: PRODUCT=remote runs no onsite piece, and POUS_CMD is set: unset it"),
    ("remote", {"CERTIFIER_CHAIN_CMD": "true"}, SPLIT_NO_FRONT),
    ("remote", {"CERTIFIER_CMD": "true"},
     "PREFLIGHT: PRODUCT=remote runs no onsite piece, and CERTIFIER_CMD is set: unset it"),
    ("remote", {"LOAD": "front", "CERTIFIER_FRONT": FRONT, "SERVE_ARGS": IDS}, None),
    ("remote", {"LOAD": "front", "CERTIFIER_FRONT": FRONT, "SERVE_ARGS": IDS, "CERTIFIER_CHAIN_CMD": "true",
                "CERTIFIER_PERIOD_CMD": "true"}, None),
    ("remote", {"LOAD": "front", "CERTIFIER_FRONT": FRONT, "SERVE_ARGS": IDS, "CERTIFIER_CMD": "true"},
     "PREFLIGHT: PRODUCT=remote runs no onsite piece, and CERTIFIER_CMD is set: unset it"),
    ("remote-network", {}, None),
    ("remote-network", {"CERTIFIER_CHAIN_CMD": "true", "CERTIFIER_PERIOD_CMD": "true"}, None),
    ("remote-network", {"POUS_CMD": "true"},
     "PREFLIGHT: PRODUCT=remote-network runs no onsite piece but the network certifier's, and POUS_CMD is set: unset it"),
    ("remote-network", {"CERTIFIER_CMD": "true"},
     "PREFLIGHT: PRODUCT=remote-network holds every compute audit of the network certifier's period to the run's "
     "registration, which only the split certifier's (V3c's) is: set CERTIFIER_CHAIN_CMD and CERTIFIER_PERIOD_CMD in "
     "place of CERTIFIER_CMD"),
    ("remote-network-no-pouw", {}, None),
    ("remote-network-no-pouw", {"CERTIFIER_CHAIN_CMD": "true", "CERTIFIER_PERIOD_CMD": "true"}, None),
    ("remote-network-no-pouw", {"WIPE_CMD": "true"},
     "PREFLIGHT: PRODUCT=remote-network-no-pouw runs no onsite piece but the network certifier's, and WIPE_CMD is set: "
     "unset it"),
    ("remote-network-no-pouw", {"CERTIFIER_CMD": "true"},
     "PREFLIGHT: PRODUCT=remote-network-no-pouw holds every compute audit of the network certifier's period to the run's "
     "registration, which only the split certifier's (V3c's) is: set CERTIFIER_CHAIN_CMD and CERTIFIER_PERIOD_CMD in "
     "place of CERTIFIER_CMD"),
    ("onsite", {"CERTIFIER_CMD": "true"}, None),
    ("onsite", {}, None),
    ("offsite", {}, "PREFLIGHT: PRODUCT is onsite, remote, remote-network or remote-network-no-pouw, not offsite"),
])
def test_remote_sets_no_onsite_default_and_its_preflight_refuses_an_onsite_command(tmp_path, product, extra, line):
    """PRODUCT=remote drops the onsite pieces: in a tree that has the wipe's piece its default WIPE_CMD isn't set, and
    the preflight refuses any onsite piece's command set, before anything runs. The certifier's front is how a run's
    requests are served, not a piece, so remote takes it, and with it the split certifier's chain and period, which
    write the period case the onsite record reads; without the front, remote refuses them. PRODUCT=remote-network takes
    the split certifier's commands and refuses the other onsite pieces' as remote does, and the whole certifier piece's
    too: its egress tie holds every compute audit of the certifier's period to the run's registration, and the whole
    piece's are its scan's own. PRODUCT=remote-network-no-pouw holds the same."""
    tree, venv = tmp_path / "tree", tmp_path / "venv" / "bin"
    _script(tree / "benchmarks" / "pous" / "streamed_wipe" / "served_wipe.sh", "exit 0")
    (tree / "benchmarks" / "network_traces").mkdir(parents=True)
    (tree / "benchmarks" / "network_traces" / "certifier_piece.py").write_text(
        "print('usage: certifier_piece.py front PORT [--gate --warmup --bucket-ms]')\n")
    _script(venv / "python", f'exec {sys.executable} "$@"')
    env = {"PATH": os.defpath, "HOME": str(tmp_path), "VENV": str(venv.parent), "OUT": str(tmp_path / "out"),
           "PRODUCT": product, **extra}
    rc = subprocess.run(["bash", str(HERE.parent / "served_verdicts.sh"), "front-flags"], cwd=tree, env=env,
                        check=False).returncode
    said = (tmp_path / "out" / "verdicts.txt").read_text().splitlines() if (tmp_path / "out" / "verdicts.txt").exists() else []
    assert (rc, said) == ((3, [line]) if line else (0, []))


@pytest.mark.parametrize("extra, line", [
    ({"PRODUCT": "onsite"}, "PREFLIGHT: advice-charge serves a test run alone: PRODUCT=remote and H1_GATE empty"),
    ({"H1_GATE": "ssh gate"}, "PREFLIGHT: advice-charge serves a test run alone: PRODUCT=remote and H1_GATE empty"),
    ({"SERVE_ARGS": "--h1-receipts --h1-model m"}, "PREFLIGHT: advice-charge serves with H1_GATE empty, and serve-openai "
     "refuses SERVE_ARGS' --h1-receipts or --h1-prover without H1's --protocols: drop them"),
    ({"SERVE_ARGS": f"{IDS} --h1-prover"}, "PREFLIGHT: advice-charge serves with H1_GATE empty, and serve-openai "
     "refuses SERVE_ARGS' --h1-receipts or --h1-prover without H1's --protocols: drop them"),
    ({"SERVE_ARGS": IDS}, None),
    ({"SERVE_ARGS": f"{IDS} --advice-tau 4"}, "PREFLIGHT: the server commits the advice in buckets of SERVE_ARGS' "
     "--advice-tau (4 s), and advice_charge.py charges it in buckets of ADVICE_TAU (0.1 s): make them equal"),
    ({"SERVE_ARGS": IDS, "ADVICE_TAU": "4"}, "PREFLIGHT: the server commits the advice in buckets of SERVE_ARGS' "
     "--advice-tau (0.1 s), and advice_charge.py charges it in buckets of ADVICE_TAU (4 s): make them equal"),
    ({"SERVE_ARGS": f"{IDS} --advice-tau 4", "ADVICE_TAU": "4"}, None),
    ({"SERVE_ARGS": f"--advice-tau=4.0 {IDS}", "ADVICE_TAU": "4"}, None),
])
def test_the_advice_charge_runs_preflight_refuses_what_serve_openai_or_its_report_would_refuse_after_the_lease(
        tmp_path, extra, line):
    """advice-charge serves a test run with H1_GATE empty, which passes serve-openai no H1 --protocols, and charges the
    server's advice in buckets of ADVICE_TAU. So its preflight refuses an onsite product, an H1 gate, an H1 option in
    SERVE_ARGS, which serve-openai refuses only once the node is leased, and a server --advice-tau other than ADVICE_TAU,
    which advice_charge.py refuses only once the run has served; a run it lets through goes on to the row's files (here
    absent)."""
    tree, venv, row = tmp_path / "tree", tmp_path / "venv" / "bin", tmp_path / "row"
    tree.mkdir()
    _script(venv / "python", f'exec {sys.executable} "$@"')
    env = {"PATH": os.defpath, "HOME": str(tmp_path), "VENV": str(venv.parent), "OUT": str(tmp_path / "out"),
           "SERVE_ROOT": str(tmp_path / "serve"), "ROW": str(row), "PRODUCT": "remote", "H1_GATE": "", **extra}
    (tmp_path / "out").mkdir()
    rc = subprocess.run(["bash", str(HERE.parent / "served_verdicts.sh"), "advice-charge"], cwd=tree, env=env,
                        check=False, capture_output=True).returncode
    said = (tmp_path / "out" / "verdicts.txt").read_text().splitlines()
    assert (rc, said) == (3, [line or f"PREFLIGHT: no {row}/manifest.json"])


FAKE_COMPOSITE = '''import json, os, sys
from pathlib import Path
COMPONENTS = {"onsite": ("verity/pous/p2/v4/run", "verity/wipe/v0"),
              "remote": ("verity/pouw/v1", "verity/sampled-proofs/one-stage/v2"),
              "remote-network": ("verity/certifier/v9", "verity/pouw/v1", "verity/sampled-proofs/one-stage/v2"),
              "remote-network-no-pouw": ("verity/certifier/v9", "verity/sampled-proofs/one-stage/v2")}
if __name__ == "__main__":
    a = sys.argv[1:]
    if "ORDER" in os.environ:
        open(os.environ["ORDER"], "a").write(f"composite-{a[0]}\\n")
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
    """served_composite.sh with a stand-in lake, served_case.sh, served_cases.py and composite.py: PoUS's case is written
    under the serve dir's private dir with both Lean answers built and passed, and served_cases.py's under it, given no α
    or Γ while ALPHA and GAMMA are unset; each product gets the cases of its own subprotocols; OUT/verdicts.txt gets each
    CASE line's protocol and verdict, each case not written by its protocol, each product's NOT_ESTABLISHED and
    VERDICT lines and the host it composed on, never a PIECE line, a reason, a log or a path; each product's public
    output is OUT/composite/PRODUCT;
    and without composite.py in the tree it says so and fails, since no Lean answered. PRODUCT=remote needs no
    served_case.sh, writes neither PoUS's case nor the wipe's, and runs remote's Lean alone. A serve dir holding
    run.sh's records audit, which opened the advice (SERVE_DIR/sampled-proofs-served), and no V3c audit still hands
    served_cases.py the serve dir alone, whose V3c audit is the only one it reads, as under H1's tie (a
    ``SERVE_DIR/h1/log.jsonl``), whose cases are on the record H1 named; and no product is handed a case under
    ``SERVE_DIR/private/pouw``, whose period case reaches a product only as served_cases.py writes it, tied to that
    record. PRODUCT=remote-network runs remote's Lean and remote-network's, the latter built in onsite's package and
    given the network certifier's case where its piece released it, OUT/pieces/certifier. PRODUCT=remote-network-no-pouw
    runs its own Lean alone, built in onsite's package, given the network certifier's case and sampled proofs' and never
    PoUW's period case, even one served_cases.py wrote on R. Where SERVED_NODE names another host (a deferral's
    follow-on), it says it composed there, from a prover-side copy of the serve dir's audit inputs."""
    tree, bin_, d, out = tmp_path / "tree", tmp_path / "bin", tmp_path / "serve", tmp_path / "out"
    out.mkdir()
    _script(bin_ / "lake", 'mkdir -p .lake/build/bin && touch ".lake/build/bin/$2"')
    _script(bin_ / "systemd-run", "exit 1")
    _script(tree / "benchmarks" / "pous" / "served_case.sh", f'''set -e
[ -x "$POUS_EXE" ] && [ -x "$WIPE_EXE" ] && [ "$1" = {out}/pieces ] && [ "$MEM_GB" = 170 ]
for w in pous/w0 other; do mkdir -p {d}/private/$w; echo '{{"evidence": "beef"}}' > {d}/private/$w/case.json; done
echo '{{"protocol": "verity/pous/p2/v4/run", "verdict": "accept"}}' > {d}/private/pous/w0/answer.json
echo '{{"protocol": "verity/certifier/v9", "verdict": "accept"}}' > {d}/private/other/answer.json
echo "CASE: evidence beef"
echo 'CASE {{"piece": "{d}/private/pous/w0", "protocol": "verity/pous/p2/v4/run", "verdict": "accept"}}' ''')
    for pkg in ("protocols/pous/check", "devices/memory_challenger/lean", "protocols/remote", "protocols/onsite"):
        (tree / "verity" / "core" / pkg).mkdir(parents=True)
    (tree / "benchmarks" / "one_stage").mkdir(parents=True)
    (tree / "benchmarks" / "one_stage" / "composite.py").write_text(FAKE_COMPOSITE)
    v2 = d / "private" / "sampled-proofs-v2"
    (tree / "benchmarks" / "one_stage" / "served_cases.py").write_text(f'''import json, sys
from pathlib import Path
assert sys.argv[1:3] == ["--serve-dir", "{d}"], sys.argv
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
    r = run()
    assert r.returncode == 0, r.stderr
    host = subprocess.run(["hostname", "-s"], check=True, capture_output=True, text=True).stdout.strip()
    on = (f"COMPOSITE: composed on {host}, which holds the serve dir, from the cases the served node wrote there, not "
          "from the parties that produced them")
    assert (out / "verdicts.txt").read_text().splitlines() == [
        "COMPOSITE CASE verity/pous/p2/v4/run accept", "COMPOSITE CASES: served_case.sh exit 0",
        "COMPOSITE CASE verity/sampled-proofs/one-stage/v2 accept",
        "COMPOSITE CASE verity/registration/v0 accept",
        "COMPOSITE NOT_WRITTEN verity/inference-only/v0",
        "COMPOSITE CASES: served_cases.py exit 1, its log on the node",
        "COMPOSITE remote: NOT_ESTABLISHED missing verity/wipe/v0", "COMPOSITE remote: VERDICT verity/remote/v0 reject components",
        "COMPOSITE remote-network: NOT_ESTABLISHED missing verity/wipe/v0",
        "COMPOSITE remote-network: VERDICT verity/remote-network/v0 reject components",
        "COMPOSITE onsite: NOT_ESTABLISHED missing verity/wipe/v0", "COMPOSITE onsite: VERDICT verity/onsite/v0 reject components",
        on]
    p = d / "private" / "composite"
    assert json.loads((p / "remote" / "private" / "argv.json").read_text()) == [str(v2)]
    assert json.loads((p / "remote-network" / "private" / "argv.json").read_text()) == [str(d / "private" / "other"),
                                                                                        str(v2)]
    assert json.loads((p / "onsite" / "private" / "argv.json").read_text()) == [str(d / "private" / "pous" / "w0")]
    assert (tree / "verity" / "core" / "protocols" / "onsite" / ".lake" / "build" / "bin" / "remote-network").is_file()
    assert oct((d / "private").stat().st_mode & 0o777) == "0o700"
    assert sorted(out.iterdir()) == [out / "composite", out / "verdicts.txt"]
    assert sorted(f.relative_to(out).as_posix() for f in (out / "composite").rglob("*") if f.is_file()) == [
        "composite/onsite/composite.json", "composite/remote-network/composite.json", "composite/remote/composite.json"]
    (tree / "benchmarks" / "one_stage" / "composite.py").unlink()
    (out / "verdicts.txt").unlink()
    shutil.rmtree(out / "composite")
    assert run().returncode == 1
    assert (out / "verdicts.txt").read_text() == "COMPOSITE: not run, this tree has no benchmarks/one_stage/composite.py\n"
    (tree / "benchmarks" / "one_stage" / "composite.py").write_text(FAKE_COMPOSITE)
    (tree / "benchmarks" / "pous" / "served_case.sh").unlink()
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
    assert json.loads((tmp_path / "served_cases.argv").read_text()) == ["--serve-dir", str(d)]
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
    (out / "verdicts.txt").unlink()
    for x in (out / "composite", p, v2):
        shutil.rmtree(x)
    cert = out / "pieces" / "certifier"
    cert.mkdir(parents=True)
    (cert / "case.json").write_text('{"evidence": "beef"}')
    (cert / "answer.json").write_text('{"protocol": "verity/certifier/v9", "verdict": "accept"}')
    env["PRODUCT"] = "remote-network"
    r = run()
    assert r.returncode == 0, r.stderr
    assert (out / "verdicts.txt").read_text().splitlines() == [
        "COMPOSITE CASE verity/sampled-proofs/one-stage/v2 accept", "COMPOSITE CASE verity/registration/v0 accept",
        "COMPOSITE NOT_WRITTEN verity/inference-only/v0", "COMPOSITE CASES: served_cases.py exit 1, its log on the node",
        "COMPOSITE remote: NOT_ESTABLISHED missing verity/wipe/v0", "COMPOSITE remote: VERDICT verity/remote/v0 reject components",
        "COMPOSITE remote-network: NOT_ESTABLISHED missing verity/wipe/v0",
        "COMPOSITE remote-network: VERDICT verity/remote-network/v0 reject components",
        on]
    assert sorted(x.name for x in (out / "composite").iterdir()) == ["remote", "remote-network"]
    assert not (p / "onsite").exists()
    assert json.loads((p / "remote" / "private" / "argv.json").read_text()) == [str(v2)]
    assert json.loads((p / "remote-network" / "private" / "argv.json").read_text()) == [str(v2), str(cert)]
    (out / "verdicts.txt").unlink()
    for x in (out / "composite", p, v2):
        shutil.rmtree(x)
    period = d / "private" / "period-on-r"
    period.mkdir()
    (period / "case.json").write_text(json.dumps({"evidence": "beef"}))
    (period / "answer.json").write_text(json.dumps({"protocol": "verity/pouw/v1", "verdict": "accept"}))
    env["PRODUCT"] = "remote-network-no-pouw"
    r = run()
    assert r.returncode == 0, r.stderr
    assert (out / "verdicts.txt").read_text().splitlines() == [
        "COMPOSITE CASE verity/sampled-proofs/one-stage/v2 accept", "COMPOSITE CASE verity/registration/v0 accept",
        "COMPOSITE NOT_WRITTEN verity/inference-only/v0", "COMPOSITE CASES: served_cases.py exit 1, its log on the node",
        "COMPOSITE remote-network-no-pouw: NOT_ESTABLISHED missing verity/wipe/v0",
        "COMPOSITE remote-network-no-pouw: VERDICT verity/remote-network-no-pouw/v0 reject components",
        on]
    assert sorted(x.name for x in (out / "composite").iterdir()) == ["remote-network-no-pouw"]
    assert sorted(x.name for x in p.iterdir() if x.is_dir()) == ["remote-network-no-pouw"]
    assert json.loads((p / "remote-network-no-pouw" / "private" / "argv.json").read_text()) == [str(v2), str(cert)]
    assert (tree / "verity" / "core" / "protocols" / "onsite" / ".lake" / "build" / "bin" / "remote-network-no-pouw").is_file()
    (out / "verdicts.txt").unlink()
    for x in (out / "composite", p, v2, period):
        shutil.rmtree(x)
    env["PRODUCT"] = "remote-network"
    env["SERVED_NODE"] = "the-served-node"
    assert run().returncode == 0
    assert (out / "verdicts.txt").read_text().splitlines()[-1] == (
        f"COMPOSITE: composed on {host}, not the served node the-served-node: from a prover-side copy of its serve dir's "
        "audit inputs and the cases written here from it, not from the parties that produced them")


DIGEST = "5e" * 64


def _auditor():
    """Infra's gate, ``benchmarks/live_challenger/auditor.py``, as a module of its own."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("sv_auditor", HERE.parents[2] / "benchmarks" / "live_challenger" / "auditor.py")
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
    commits the Program digest serve-openai wrote, its T, the draw K, the fixed length and the circuit files, each read
    back by core's credit as the window's period commits it; a stratum with no ``k<K>`` file yet is named and left out.
    It names no weights without ``--model`` (the gate refuses a root its registry lacks). ``period close`` closes it."""
    from verity.core.protocols.pouw import credit as CR
    from verity.ml.crypto import lms
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
    from verity.core.protocols.pouw.circuit import h1tile as HT, h1window as HW
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
    """`served_verdicts.sh serve` in a toy tree whose venv python runs a ``-c`` program (ℓ_a's default) and otherwise records
    serve-openai's argv and exits before listening, under a no-op `sleep`: the exit status and that argv."""
    tree, bin_, venv = tmp_path / "tree", tmp_path / "bin", tmp_path / "venv" / "bin"
    for d in (tree / "integrations" / "vllm", bin_, venv):
        d.mkdir(parents=True)
    py = f'[ "$1" = -c ] && exec {sys.executable} "$@"\nprintf \'%s\\n\' "$@" > {tmp_path}/argv; exit 1'
    for path, body in ((venv / "python", py), (bin_ / "sleep", "exit 0")):
        path.write_text(f"#!/bin/sh\n{body}\n")
        path.chmod(0o755)
    env = {"PATH": f"{bin_}:{os.defpath}", "HOME": str(tmp_path), "VENV": str(venv.parent), "D": str(tmp_path / "serve"),
           "OUT": str(tmp_path / "out"), **env}
    rc = subprocess.run(["bash", str(HERE.parent / "served_verdicts.sh"), "serve"], cwd=tree, env=env, check=False,
                        capture_output=True).returncode
    argv = tmp_path / "argv"
    return rc, argv.read_text().splitlines() if argv.exists() else []


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


@pytest.mark.parametrize("env, cap, composite", [
    ({}, (4, 256, 512, 24), "onsite"),
    ({"WINDOW_STEPS": "1"}, "1328", "onsite"),
    ({"SLOTS": "8", "WINDOW_STEPS": "16", "MAX_MODEL_LEN": "2048", "ID_BYTES": "12"}, (8, 16, 2048, 12), "onsite"),
    ({"CAP": "65536", "PRODUCT": "remote"}, "65536", "remote"),
    ({"CAP": "65536", "SERVE_ARGS": "--prefix-caching"}, "65536", "onsite"),
])
def test_the_server_registers_b_at_cap_and_commits_each_windows_advice_under_the_weights_registration(tmp_path, env, cap,
                                                                                                      composite):
    """serve-openai gets ``--advice`` once, B capped at CAP (ℓ_a), by default the least cap that pads every Commit period
    within the run's limits (SLOTS requests and WINDOW_STEPS steps a period, MAX_MODEL_LEN, ID_BYTES), its registration log
    the run's PRODUCT's, and each window's advice recording the serve dir's weights registration, which the run writes once
    the server listens."""
    cap = _default_cap(*cap) if isinstance(cap, tuple) else cap
    rc, argv = _serve_argv(tmp_path, {"H1_GATE": "", **env})
    assert rc == 10 and argv.count("--advice") == 1, (rc, argv)
    got = {f: argv[argv.index(f) + 1] for f in ("--advice-cap", "--advice-composite", "--advice-weights")}
    assert got == {"--advice-cap": cap, "--advice-composite": composite,
                   "--advice-weights": str(tmp_path / "serve" / "weights_registration.json")}


def test_a_server_with_the_prefix_cache_and_no_cap_is_refused_before_it_starts(tmp_path):
    """ℓ_a's default leaves out the prefix cache's KV sources, so a server serving with it and no CAP is a PREFLIGHT refusal
    before serve-openai runs, never a server whose every period past the default is refused."""
    rc, argv = _serve_argv(tmp_path, {"H1_GATE": "", "SERVE_ARGS": "--prefix-caching"})
    lines = (tmp_path / "out" / "verdicts.txt").read_text().splitlines()
    assert rc == 3 and argv == [], (rc, argv)
    assert lines == ["PREFLIGHT: SERVE_ARGS serves with the prefix cache, whose KV sources ℓ_a's default leaves out: set CAP"]


@pytest.mark.parametrize("env, cap", [
    ({}, (4, 256, 512, 24)),
    ({"WINDOW_STEPS": "1"}, "1328"),
    ({"CAP": "65536"}, "65536"),
    ({"SERVE_ARGS": "--prefix-caching"}, None),
])
def test_charge_on_a_serve_dir_whose_server_committed_no_advice_writes_it_under_b_at_cap(tmp_path, env, cap):
    """`served_verdicts.sh charge` alone, on a serve dir with no advice.jsonl, writes each window's advice with
    `verity-vllm advice` under prove_path's compiler, both at CAP, by default ℓ_a of the run's limits as `serve` derives
    it, then charges it (advice_charge.py); with the prefix cache and no CAP it is the same PREFLIGHT refusal as `serve`'s,
    before any of them runs."""
    tree, venv, calls = tmp_path / "tree", tmp_path / "venv" / "bin", tmp_path / "calls"
    (tree / "integrations" / "vllm").mkdir(parents=True)
    _script(venv / "python", f'[ "$1" = -c ] && exec {sys.executable} "$@"\nprintf \'%s\\n\' "$*" >> {calls}\n'
                             'case $1 in *prove_path.py) echo \'{"inner": {"name": "n", "digest": "d"}}\' ;; esac')
    env = {"PATH": os.defpath, "HOME": str(tmp_path), "VENV": str(venv.parent), "D": str(tmp_path / "serve"),
           "OUT": str(tmp_path / "out"), "H1_GATE": "", **env}
    rc = subprocess.run(["bash", str(HERE.parent / "served_verdicts.sh"), "charge"], cwd=tree, env=env, check=False,
                        capture_output=True).returncode
    ran = [c.split() for c in calls.read_text().splitlines()] if calls.exists() else []
    if cap is None:
        assert (rc, ran) == (3, []) and (tmp_path / "out" / "verdicts.txt").read_text().splitlines() == [
            "PREFLIGHT: SERVE_ARGS serves with the prefix cache, whose KV sources ℓ_a's default leaves out: set CAP"]
        return
    cap = _default_cap(*cap) if isinstance(cap, tuple) else cap
    compiler, advice, charge = ran
    assert rc == 0 and compiler[:2] == ["benchmarks/prove_path/prove_path.py", "compiler"], (rc, ran)
    assert compiler[compiler.index("--cap-bits") + 1] == cap and advice[advice.index("--cap") + 1] == cap
    assert advice[:4] == ["-m", "verity_vllm.pipeline.cli", "advice", str(tmp_path / "serve")]
    assert [advice[advice.index(f) + 1] for f in ("--compiler-name", "--compiler-digest")] == ["n", "d"]
    assert charge[0] == str(tree / "benchmarks" / "one_stage" / "advice_charge.py")


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
