"""`flock-verify rec-compose` (`RecCompose.check`) and its Python mirror (`verity_flock.rec_compose`) agree, verdict and
reason, on an honest part set at a real inner shape and on that set with one thing changed: each shared input in one part
(another file, or none where the compiled verifier reads one), an inner shape or rows file other than the composition's,
a session without `--zk`, with an identifier or refused, a reused or missing coin key, and a part missing, repeated, out
of range or in a log with two `VERDICT`s or none. Each `rec-compose` builds V*'s statements at the inner shape (about
5 s), so the module is slow; it catches what `Flock/Test/RecCompose.lean`'s #guards can't: the binary's own reading of
its flags and files (the inner shape's and rows file's SHA-512, V*'s count from the inner shape) and the mirror drifting
from the Lean."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
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


def _honest(n: int, inner: bytes) -> list[tuple[str, str]]:
    return [(f"p{j}", f"setup: 3 ms\n{RC.verdict(j, ROWS, _shared(inner), f'k{j}')}\n") for j in range(n)]


def _set(parts: list, i: int, line: str) -> list:
    return [*parts[:i], (parts[i][0], f"{line}\n"), *parts[i + 1:]]


def _cases(n: int, inner: bytes) -> dict[str, list[tuple[str, str]]]:
    h = _honest(n, inner)
    v = lambda j, **kw: RC.verdict(j, kw.pop("rows", ROWS), kw.pop("shared", _shared(inner)), kw.pop("key", f"k{j}"), **kw)  # noqa: E731
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
    cases["coin-key-reused"] = _set(h, 1, v(1, key="k0"))
    cases["no-rec-shared"] = _set(h, 1, v(1).replace(',"rec_shared":', ',"x":'))
    cases["part-missing"] = h[:-1]
    cases["part-twice"] = _set(h, 1, v(0, key="k1"))
    cases["part-out-of-range"] = _set(h, 1, v(n, key="k1"))
    cases["one-part-too-many"] = [*h, ("extra", f"{v(0, key='kx')}\n")]
    cases["two-verdicts"] = _set(h, 1, f"{v(1)}\n{v(1, key='ky')}")
    cases["no-verdict"] = _set(h, 1, "setup: refused")
    cases["no-parts"] = []
    return cases


CASES = ["honest", *(f"{k}-one-part" for k in ("rec_inner", *FILES, "inner_coins")), *(f"{k}-none" for k in FILES),
         "inner_coins-every-part", "other-inner-every-part", "other-rows-one-part", "other-rows-every-part", "not-zk",
         "identifier", "refused", "coin-key-reused", "no-rec-shared", "part-missing", "part-twice", "part-out-of-range",
         "one-part-too-many", "two-verdicts", "no-verdict", "no-parts"]
#: the compiled verifier leaves these optional: none in every part composes
OPTIONAL = {"inner_registration-none", "inner_draw-none"}


@pytest.mark.parametrize("case", CASES)
def test_rec_compose_agrees_with_its_mirror(setup, tmp_path, case):
    d, n, inner = setup
    parts = _cases(n, inner)[case]
    logs = []
    for name, text in parts:
        (tmp_path / f"{name}.log").write_text(text)
        logs.append(str(tmp_path / f"{name}.log"))
    p = subprocess.run([str(BIN), "rec-compose", "--rec-inner", str(d / "inner.json"), "--rec-rows", str(d / "rows.json"),
                        *logs], capture_output=True, text=True)
    [line] = [json.loads(ln[12:]) for ln in p.stdout.splitlines() if ln.startswith("REC-COMPOSE ")]
    want = RC.check(n, RC.digest(ROWS), RC.digest(inner), [(lg, t) for lg, (_, t) in zip(logs, parts)])
    assert (line["accepted"], line["why"]) == want, (case, line)
    assert p.returncode == (0 if want[0] else 1)
    assert want[0] == (case == "honest" or case in OPTIONAL), (case, want)
