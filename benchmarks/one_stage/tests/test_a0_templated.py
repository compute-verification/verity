"""a0 hands a templated statement to both sides and to the verifier (`a0.session`, `a0.lean_verify`, `a0.firewall_env`):
the server and the prover read its placed unit (``st["tmpl_unit"]``, `--tmpl-unit`), and the verifier and a firewalled
prover set up its statement (``st["statement"]``); a flat stage passes no placed unit and keeps `a0.STATEMENT_ID`. The
server, prover and verifier are one stand-in that records each call's argv."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import a0  # noqa: E402

TMPL = "verity/flock-circuit/tmpl"

#: a stand-in for flock-circuit and flock-verify: each call appends its argv to ``calls``; ``serve`` records an accepted
#: session, ``prove`` prints its LIVE line, ``archive-put`` keys each file by its name and ``verify`` accepts
FAKE = '''#!{python}
import json, pathlib, sys
args = sys.argv[1:]
with open({calls!r}, "a") as f:
    f.write(json.dumps(args) + "\\n")
if args[0] == "serve":
    s = pathlib.Path(args[args.index("--out") + 1]) / "s0"
    s.mkdir(parents=True, exist_ok=True)
    (s / "session.json").write_text(json.dumps({{"verdict": {{"accepted": True}}}}))
    print("SERVING", flush=True)
elif args[0] == "prove":
    print("LIVE\\t" + json.dumps({{"accepted": True, "roots": {{}}}}))
elif args[0] == "archive-put":
    for p in args[2:]:
        print(pathlib.Path(p).name + "-key  " + p)
elif args[0] == "verify":
    print("VERDICT " + json.dumps({{"accepted": True, "why": None}}))
'''


@pytest.mark.parametrize("templated", [True, False])
def test_a_templated_stage_reaches_both_sides_and_the_verifier(tmp_path, monkeypatch, templated):
    monkeypatch.delenv("FLOCK_CHALLENGER", raising=False)
    calls, bin_ = tmp_path / "calls.jsonl", tmp_path / "flock"
    bin_.write_text(FAKE.format(python=sys.executable, calls=str(calls)))
    bin_.chmod(0o755)
    d = tmp_path / "stage"
    d.mkdir()
    (d / "circuit.txt").write_text("circuit")
    (d / "pub-4.bin").write_bytes(b"pub")
    st = {"dir": d, "pin": "00", **({"statement": TMPL, "tmpl_unit": d / "unit.placed"} if templated else {})}
    out = tmp_path / "session"
    record, live = a0.session(str(bin_), st, 4, "subset:2", 0, out, zk=True)
    assert record["verdict"]["accepted"] and live["accepted"]
    assert a0.lean_verify(str(bin_), st, 4, out, zk=True)["accepted"]
    argv = {a[0]: a for a in map(json.loads, calls.read_text().splitlines())}
    for side in ("serve", "prove"):
        got = argv[side][argv[side].index("--tmpl-unit") + 1] if "--tmpl-unit" in argv[side] else None
        assert got == (str(d / "unit.placed") if templated else None)
    want = TMPL if templated else a0.STATEMENT_ID
    assert argv["verify"][argv["verify"].index("--statement") + 1] == want
    assert a0.firewall_env(st)["FC_FIREWALL_STATEMENT"] == want


#: a stand-in for flock-tmpl-unit: appends its argv to ``calls`` and writes the placed unit at ``--out``
PLACE = '''#!{python}
import json, pathlib, sys
args = sys.argv[1:]
with open({calls!r}, "a") as f:
    f.write(json.dumps(args) + "\\n")
pathlib.Path(args[args.index("--out") + 1]).write_text("ORDER\\n")
'''


def test_templated_places_a_templated_statements_unit_once_and_passes_a_flat_one(tmp_path):
    """`a0.templated`: a templated statement's template section and layout mode go beside it for flock-tmpl-unit, once, and
    its placed unit and statement are what a session and the verifier read; a flat statement adds neither, and a templated
    one without flock-tmpl-unit is refused."""
    from types import SimpleNamespace

    from verity_flock import circuit as FC

    calls, bin_ = tmp_path / "calls.jsonl", tmp_path / "flock-tmpl-unit"
    bin_.write_text(PLACE.format(python=sys.executable, calls=str(calls)))
    bin_.chmod(0o755)
    st = SimpleNamespace(meta={"statement": TMPL, "layout": "iv/hybrid/4"},
                         text=f"{FC.TMPL_FORMAT} v1\nMETA {{}}\n{FC.TMPL_SECTION} 2\ndef a\nend\nUNIT\n")
    d = tmp_path / "stage"
    assert a0.templated(SimpleNamespace(meta={"statement": a0.STATEMENT_ID}, text=""), d, None) == {}
    with pytest.raises(SystemExit, match="--tmpl-unit-bin"):
        a0.templated(st, d, None)
    for _ in range(2):
        assert a0.templated(st, d, str(bin_)) == {"statement": TMPL, "tmpl_unit": d / "unit.placed"}
    assert (d / "unit.tmpl").read_text() == "def a\nend\n" and (d / "unit.layout").read_text() == "iv/hybrid/4"
    assert [json.loads(x) for x in calls.read_text().splitlines()] == [
        [str(d / "unit.tmpl"), "--layout", "iv/hybrid/4", "--out", str(d / "unit.placed")]]
