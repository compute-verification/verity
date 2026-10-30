"""`flock-verify rec-compose` (`RecCompose.check`) and its Python mirror (`verity_flock.rec_compose`) agree, verdict and
reason, on an honest part set at a real inner shape and on that set with one thing changed: each shared input in one part
(another file, or none where the compiled verifier reads one), an inner shape or rows file other than the composition's,
a session without `--zk`, with an identifier or refused, a reused or missing coin key, and a part missing, repeated, out
of range or in a log with two `VERDICT`s or none; and, against the challenger's interactive-run record (written by
`challenger.RunRecord`), a part retried, a part whose session the record lacks, an opened session that is not the composed
one, a session for a part V* lacks, a staging after the first Hello or a second one, other rows or another shared set
staged, a record not closed or going on past its close, and a record that is none, of another kind or with an event out
of its form. Each `rec-compose` builds V*'s statements at the inner shape (about 5 s), so the module is slow; it catches
what `Flock/Test/RecCompose.lean`'s #guards can't: the binary's own reading of its flags and files (the inner shape's,
rows file's and record's), the challenger's writer read by the Lean, and the mirror drifting from the Lean."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from verity.core.protocols.sampled_proofs import challenger as C
from verity_flock import rec_compose as RC

BIN = Path(__file__).resolve().parents[3] / "core" / "service" / "flock" / ".lake" / "build" / "bin" / "flock-verify"

pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(not BIN.exists() or shutil.which("lake") is None, reason="the Lean verifier is not built")]

#: the fixture session's table shape (`test_rec_vstar_checks.SHAPES`)
INNER = {"format": "verity/rec-inner/v2", "m": 25, "k_log": 22, "claims": 8, "regions": 3}
ROWS = b'{"ports":{"acc0":{}}}'
FILES = {"inner_circuit": b"circuit", "inner_public": b"public", "inner_partition": b"{}", "inner_session": b"session",
         "inner_transcript": b"transcript", "inner_registration": b"registration",
         "inner_statement": "verity/flock-circuit+plain-leaves", "inner_statement_coins": "os",
         "inner_tables": [("range8", b"table")], "inner_draw": b"draw"}


@pytest.fixture(scope="module")
def setup(tmp_path_factory) -> tuple[Path, int, bytes]:
    d = tmp_path_factory.mktemp("compose")
    (d / "inner.json").write_text(json.dumps(INNER))
    (d / "rows.json").write_bytes(ROWS)
    p = subprocess.run([str(BIN), "rec-statements", "--rec-inner", str(d / "inner.json")], capture_output=True, text=True)
    [line] = [json.loads(ln[9:]) for ln in p.stdout.splitlines() if ln.startswith("RECVSTAR ")]
    return d, len(line["statements"]), (d / "inner.json").read_bytes()


def _shared(inner: bytes, **change) -> dict:
    return RC.shared(rec_inner=inner, **{**FILES, **change})


def key(j: int) -> str:
    """A session's coin key, 256 bytes in lowercase hex."""
    return f"{j:08x}" * 64


def _honest(n: int, inner: bytes) -> list[tuple[str, str]]:
    return [(f"p{j}", f"setup: 3 ms\n{RC.verdict(j, ROWS, _shared(inner), key(j))}\n") for j in range(n)]


def _record(n: int, inner: bytes, opens=None, rows: bytes = ROWS, shared: dict | None = None) -> dict:
    """The challenger's record of the run (`challenger.RunRecord`): the staging, the sessions ``opens`` ((session, part,
    key), by default one per part on its own key), the close."""
    r = C.RunRecord()
    r.stage(RC.digest(rows), shared or _shared(inner))
    for o in opens if opens is not None else [(f"s{j}", j, key(j)) for j in range(n)]:
        r.open(*o)
    return r.close()


def _text(doc: dict) -> str:
    return json.dumps(doc)


def _events(doc: dict, events: list) -> str:
    return _text({**doc, "events": events})


def _set(parts: list, i: int, line: str) -> list:
    return [*parts[:i], (parts[i][0], f"{line}\n"), *parts[i + 1:]]


def _cases(n: int, inner: bytes) -> dict[str, list[tuple[str, str]]]:
    h = _honest(n, inner)
    v = lambda j, **kw: RC.verdict(j, kw.pop("rows", ROWS), kw.pop("shared", _shared(inner)), kw.pop("key", key(j)), **kw)  # noqa: E731
    other = {k: (b"other" if isinstance(x, bytes) else [("range8", b"other")] if k == "inner_tables" else
                 "seed" if k == "inner_statement_coins" else "verity/flock-circuit") for k, x in FILES.items()}
    cases = {"honest": h}
    for k in ("rec_inner", *FILES, "inner_coins"):
        sh = _shared(inner, **{k: other.get(k, b"other")}) if k != "rec_inner" else RC.shared(**{**FILES, "rec_inner": b"x"})
        cases[f"{k}-one-part"] = _set(h, 1, v(1, shared=sh))
    for k in FILES:
        sh = _shared(inner, **{k: None})
        cases[f"{k}-none"] = [(name, f"{v(j, shared=sh)}\n") for j, (name, _) in enumerate(h)]
    cases["inner_coins-every-part"] = [(name, f"{v(j, shared=_shared(inner, inner_coins=b'c'))}\n")
                                       for j, (name, _) in enumerate(h)]
    cases["other-inner-every-part"] = [(name, f"{v(j, shared=RC.shared(**{**FILES, 'rec_inner': b'x'}))}\n")
                                       for j, (name, _) in enumerate(h)]
    cases["other-rows-one-part"] = _set(h, 1, v(1, rows=b"other"))
    cases["other-rows-every-part"] = [(name, f"{v(j, rows=b'other')}\n") for j, (name, _) in enumerate(h)]
    cases["not-zk"] = _set(h, 1, v(1, zk=False))
    cases["identifier"] = _set(h, 1, v(1, identifier={"id": 1}))
    cases["refused"] = _set(h, 1, v(1, accepted=False, why="S2/R7"))
    cases["coin-key-reused"] = _set(h, 1, v(1, key=key(0)))
    cases["no-rec-shared"] = _set(h, 1, v(1).replace(',"rec_shared":', ',"x":'))
    cases["part-missing"] = h[:-1]
    cases["part-twice"] = _set(h, 1, v(0, key=key(1)))
    cases["part-out-of-range"] = _set(h, 1, v(n, key=key(1)))
    cases["one-part-too-many"] = [*h, ("extra", f"{v(0, key=key(9000))}\n")]
    cases["two-verdicts"] = _set(h, 1, f"{v(1)}\n{v(1, key=key(9001))}")
    cases["no-verdict"] = _set(h, 1, "setup: refused")
    cases["no-parts"] = []
    return cases


def _runs(n: int, inner: bytes) -> dict[str, str]:
    """The challenger's record of the honest parts' run with one thing changed."""
    honest = _record(n, inner)
    ev = honest["events"]
    one = [(f"s{j}", j, key(j)) for j in range(n)]
    return {
        "run-part-retried": _text(_record(n, inner, [*one[:2], ("s1-retry", 1, key(8000)), *one[2:]])),
        "run-part-retried-composed-second": _text(_record(n, inner, [one[0], ("s1-first", 1, key(8000)), *one[1:]])),
        "run-part-unrecorded": _text(_record(n, inner, one[:-1])),
        "run-opened-not-composed": _text(_record(n, inner, [one[0], ("s1", 1, key(8000)), *one[2:]])),
        "run-part-out-of-range": _text(_record(n, inner, [*one, (f"s{n}", n, key(n))])),
        "run-staged-after-hello": _events(honest, [ev[1], ev[0], *ev[2:]]),
        "run-staged-twice": _events(honest, [ev[0], ev[1], ev[0], *ev[2:]]),
        "run-other-rows-staged": _text(_record(n, inner, rows=b"other")),
        "run-other-shared-staged": _text(_record(n, inner, shared=_shared(inner, inner_transcript=b"other"))),
        "run-not-closed": _events(honest, ev[:-1]),
        "run-event-after-close": _events(honest, [*ev, {"event": "open", "session": "late", "rec_part": 0,
                                                        "coin_key": key(8001)}]),
        "run-none": "",
        "run-other-kind": _text({**honest, "kind": "verity/rec-run/v0"}),
        "run-no-events": _text({"kind": RC.RUN_KIND}),
        "run-event-out-of-form": _events(honest, [ev[0], {"event": "open", "session": "s0", "rec_part": "0",
                                                          "coin_key": key(0)}, *ev[2:]]),
    }


CASES = ["honest", *(f"{k}-one-part" for k in ("rec_inner", *FILES, "inner_coins")), *(f"{k}-none" for k in FILES),
         "inner_coins-every-part", "other-inner-every-part", "other-rows-one-part", "other-rows-every-part", "not-zk",
         "identifier", "refused", "coin-key-reused", "no-rec-shared", "part-missing", "part-twice", "part-out-of-range",
         "one-part-too-many", "two-verdicts", "no-verdict", "no-parts"]
RUNS = ["run-part-retried", "run-part-retried-composed-second", "run-part-unrecorded", "run-opened-not-composed",
        "run-part-out-of-range", "run-staged-after-hello", "run-staged-twice", "run-other-rows-staged",
        "run-other-shared-staged", "run-not-closed", "run-event-after-close", "run-none", "run-other-kind",
        "run-no-events", "run-event-out-of-form"]
#: the compiled verifier leaves these optional: none in every part composes
OPTIONAL = {"inner_registration-none", "inner_draw-none"}


def _compose(d: Path, tmp_path: Path, n: int, inner: bytes, parts: list[tuple[str, str]], run: str) -> tuple[bool, str]:
    logs = []
    for name, text in parts:
        (tmp_path / f"{name}.log").write_text(text)
        logs.append(str(tmp_path / f"{name}.log"))
    (tmp_path / "run.json").write_text(run)
    p = subprocess.run([str(BIN), "rec-compose", "--rec-inner", str(d / "inner.json"), "--rec-rows", str(d / "rows.json"),
                        "--rec-run", str(tmp_path / "run.json"), *logs], capture_output=True, text=True)
    [line] = [json.loads(ln[12:]) for ln in p.stdout.splitlines() if ln.startswith("REC-COMPOSE ")]
    want = RC.check(n, RC.digest(ROWS), RC.digest(inner), run, [(lg, t) for lg, (_, t) in zip(logs, parts)])
    assert (line["accepted"], line["why"]) == want, line
    assert p.returncode == (0 if want[0] else 1)
    return want


@pytest.mark.parametrize("case", CASES)
def test_rec_compose_agrees_with_its_mirror(setup, tmp_path, case):
    d, n, inner = setup
    parts = _cases(n, inner)[case]
    shared = None
    if case in OPTIONAL:
        shared = _shared(inner, **{case.removesuffix("-none"): None})
    want = _compose(d, tmp_path, n, inner, parts, _text(_record(n, inner, shared=shared)))
    assert want[0] == (case == "honest" or case in OPTIONAL), (case, want)


@pytest.mark.parametrize("case", RUNS)
def test_rec_compose_holds_the_parts_to_one_run(setup, tmp_path, case):
    """The honest parts, each accepted, against the challenger's record of a run they are not exactly one of: refused,
    by the binary and its mirror alike."""
    d, n, inner = setup
    want = _compose(d, tmp_path, n, inner, _honest(n, inner), _runs(n, inner)[case])
    assert not want[0] and want[1], (case, want)


def test_rec_compose_needs_the_run_record(setup, tmp_path):
    """Without the challenger's record there is no composition: a usage error, not a verdict."""
    d, n, inner = setup
    p = subprocess.run([str(BIN), "rec-compose", "--rec-inner", str(d / "inner.json"), "--rec-rows", str(d / "rows.json"),
                        str(tmp_path / "p0.log")], capture_output=True, text=True)
    assert p.returncode == 2 and "--rec-run F" in p.stderr
