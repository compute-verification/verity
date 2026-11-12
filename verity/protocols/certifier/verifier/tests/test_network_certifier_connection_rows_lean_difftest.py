"""The difftest of `connection_rows.py` against the Lean package `network_check`: the Python half.

`verity/protocols/certifier/verifier/lean/scripts/connection_rows_vectors.py` writes random instances and this
package's results on each (a schedule's digest and tables, `request_room`, `request_bytes` and `response_bytes`, a table's domain,
a head's encoding and digest, and `check_registration`'s head or refusal). `check` runs the Lean half in its Lean audit
(`runs` in that package's `lean-audit.json`): `NetworkCheck/ConnectionRowsDifftestMain.lean` computes every result with
`NetworkCheck.ConnectionRows` and fails on any that differs. The audit checks the data against the record's SHA-256,
and its pass is cached on the Lean package's files alone, so the test here is what says the data is still what this
package computes: a change to the head, a table or the registration's read changes the data, fails this test, and
`check.py --update` records the new data, which makes `check` run the Lean half on it. The other test checks that the
generator reaches what its docstring says.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

LEAN = Path(__file__).resolve().parents[1] / "lean"
SCRIPT = LEAN / "scripts" / "connection_rows_vectors.py"
_spec = importlib.util.spec_from_file_location("connection_rows_difftest_vectors", SCRIPT)
vectors = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vectors)
RUN = json.loads((LEAN / "lean-audit.json").read_text())["runs"]["NetworkCheck.ConnectionRowsDifftestMain"]


def test_the_data_is_what_the_audit_pins(tmp_path):
    """Every instance and the Python's results on it, byte for byte as the record pins them."""
    made = {n: hashlib.sha256(p.read_bytes()).hexdigest() for n, p in vectors.write(tmp_path).items()}
    assert made == RUN["data"]
    assert RUN["args"] == ["{connection-rows/instances.jsonl}", "{connection-rows/python.jsonl}", str(vectors.COUNT)]


def test_the_generator_covers_what_it_says():
    pairs = [vectors.instance(s) for s in range(vectors.COUNT)]
    insts, results = [i for i, _ in pairs], [r for _, r in pairs]
    listed = [(i, r) for i, r in pairs if r["tables"] is not None]
    assert len(pairs) // 2 < len(listed) < len(pairs)  # most schedules are taken, and some are refused
    assert {i["lengths"] for i in insts} > {None, 1, 2}  # and a form that is none
    assert {i["lengths"] for i, _ in listed} == {1, 2}
    assert any(i["content"] is None for i, _ in listed) and any(len(r["tables"]) == 14 for _, r in listed)
    assert any("max_tokens_cap" in i["schedule"] for i, _ in listed)
    for field in ("rooms", "ids", "heads"):  # each query both answered and refused
        answers = [x for r in results for x in r[field]]
        assert len(answers) // 4 < answers.count(None) < 3 * len(answers) // 4, field
    assert all(len(d) == 128 for r in results for d in r["domains"])
    assert any(q[1] == 0 for i, _ in listed for q in i["domains"])  # a domain of no row
    links = {i["link"] for i in insts}
    assert "" in links and any(not link.isascii() for link in links)
    assert any(len(h["link"]) > 65_535 for i in insts for h in i["heads"])
    # the first registration is the honest one, and it is read; of the others some are, and most are refused
    assert all(r["registrations"][0] is not None for _, r in listed)
    later = [x for _, r in listed for x in r["registrations"][1:]]
    assert 0 < len(later) - later.count(None) < len(later) // 3
    # form 2: a class of no row and a class of the most rows, each read
    held = [(i, r) for i, r in listed if i["lengths"] == 2 and i["content"]]
    counts = {(t["rows"] == 0, t["rows"] == t0[1]) for i, r in held
              for t, t0 in zip(i["registrations"][0]["tables"][2:], r["tables"][2:])}
    assert (True, False) in counts and (False, True) in counts
    # and refused: a count past the most, a class's tables of two counts, a table of no row with a root
    refused = [d for i, r in held for d, x in zip(i["registrations"], r["registrations"]) if x is None]
    assert any(t["rows"] > t0[1] for i, r in held for d, x in zip(i["registrations"], r["registrations"]) if x is None
               and isinstance(d.get("tables"), list) for t, t0 in zip(d["tables"][2:], r["tables"][2:])
               if isinstance(t, dict) and isinstance(t.get("rows"), int))
    tabled = [d["tables"] for d in refused if isinstance(d.get("tables"), list)
              and all(isinstance(t, dict) and isinstance(t.get("rows"), int) for t in d["tables"])]
    assert any(len({t["rows"] for t in ts[k:k + 3]}) > 1 for ts in tabled for k in range(2, len(ts), 3))
    assert any(t["rows"] == 0 and t.get("root") != "00" * 64 for ts in tabled for t in ts[2:])
