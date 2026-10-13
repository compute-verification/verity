"""The difftest of `certificate.py`'s encodings against the Lean package `network_check`: the Python half.

`verity/core/protocols/network/lean/scripts/difftest_vectors.py` writes 2,000 random instances and this
package's results on each (the frame tree, `parameters_digest` and its canonical JSON, `record_head`, `RecordHead`'s
encoding and digest, `certificate_message`). `check` runs the Lean half in its Lean audit (`runs` in that package's
`lean-audit.json`): `NetworkCheck/DifftestMain.lean` computes every result with `NetworkCheck.Certificate` and fails on any
byte that differs. The audit checks the data against the record's SHA-256, and its pass is cached on the Lean package's
files alone, so the test here is what says the data is still what this package computes: a change to an encoding changes
the data, fails this test, and `check.py --update` records the new data, which makes `check` run the Lean half on it. The
other test checks, on the first 300 instances, that the generator reaches what its docstring says.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

LEAN = Path(__file__).resolve().parents[2] / "network" / "lean"
SCRIPT = LEAN / "scripts" / "difftest_vectors.py"
_spec = importlib.util.spec_from_file_location("network_check_difftest_vectors", SCRIPT)
vectors = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vectors)
RUN = json.loads((LEAN / "lean-audit.json").read_text())["runs"]["NetworkCheck.DifftestMain"]


def test_the_data_is_what_the_audit_pins(tmp_path):
    """All 2,000 instances and the Python's results on them, byte for byte as the record pins them."""
    made = {n: hashlib.sha256(p.read_bytes()).hexdigest() for n, p in vectors.write(tmp_path).items()}
    assert made == RUN["data"]
    assert RUN["args"] == ["{difftest/instances.jsonl}", "{difftest/python.jsonl}", str(vectors.COUNT)]


def test_the_generator_covers_what_it_says():
    pairs = [vectors.instance(s) for s in range(300)]
    insts, results = [i for i, _ in pairs], [r for _, r in pairs]
    depths = {r["depth"] for r in results}
    assert 0 in depths and max(depths) >= 5
    assert any(i["params"]["B"] > 128 - 38 for i in insts)  # a leaf's frame spans two SHA-512 blocks
    assert any(i["params"]["tau"] >= 2 ** 32 for i in insts) and any(i["params"]["Q"] >= 2 ** 64 for i in insts)
    assert any(i["sync_set"] is None for i in insts)
    rhos = [d for i in insts if i["sync_set"] for _, d in i["sync_set"]["rhos"]]
    assert 1 in rhos and any(d > 1 for d in rhos)
    assert any(r["real_frames"] is None for r in results) and any(r["real_frames"] == [] for r in results)
    assert any(r["root"] is None for r in results) and any(None in r["leaves"] for r in results)
    opens = [o for r in results for o in r["opens"]]
    assert None in opens and any(o for o in opens)
    verdicts = [v for r in results for v in r["verifies"]]
    assert verdicts.count(True) > 100 and verdicts.count(False) > 100
    assert any(r["record_head"] is None and r["real_frames"] is not None for r in results)  # refused by encode
    heads = [h for r in results for h in r["heads"]]
    assert None in heads and heads.count(None) < len(heads) // 2
    messages = [m for r in results for m in r["messages"]]
    assert None in messages and messages.count(None) < len(messages) // 2
    by_session = [(len(q[1]) // 2, m) for i, r in zip(insts, results) for q, m in zip(i["messages"], r["messages"])]
    for n in (31, 33):  # the session-length refusal, pinned to agree with Lean's
        assert any(k == n for k, _ in by_session) and all(m is None for k, m in by_session if k == n)
    assert any(k == 32 and m is not None for k, m in by_session)
    assert any(r["nodes"] for r in results)
    links = {i["link"] for i in insts} | {h["link"] for i in insts for h in i["heads"]}
    assert any(not link.isascii() for link in links) and "" in links and any("\x00" in link for link in links)
    assert {i["direction"] for i in insts} == {"egress", "ingress"}
    assert any(i["window"] == 2 ** 64 - 1 for i in insts)
    assert {i["status"] for i in insts} == set(vectors.REASONS)
