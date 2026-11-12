"""A history table read as a reader's `root` and `domain`: a table whose rows the verifier holds (`{"held": ...}` runs, no
writer units) resolves into a supplied value of `Digest`s (`{"history": "H/root"}`, `{"history": "H/domain"}`), whose
rows R7 lays out as a circuit's `Digest` holds them (`registered.digest_leaves`, `gates.rowleaf.digest_bits`), in both
verifiers. `fixtures/history_read_vector.json` is the bytes a committer reproduces at capacity 8; regenerate it with
`uv run python verity/protocols/spot/verifier/tests/test_history_read.py`."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from verity.circuits.defs import CompositeDefinition, bind
from verity.circuits.program import Program
from verity.circuits.refs import coll_concat
from verity.circuits.types import Array, Value
from verity.commitments import history as HT
from verity.commitments.gates.rowleaf import digest_bits
from verity.protocols.spot.verifier.one_stage import partition as P, registered as G, registration as R, verifier as V

from test_registered import LAW, MIX, POS, Q, SCHEME, SERVED, TABLE

VECTOR = Path(__file__).parent / "fixtures" / "history_read_vector.json"
TABLE_NAME = "layer"
#: the reads' supplied value: row 0 the table's root, row 1 its domain id, each a `Digest`
SUPPLIED = {"hist": [[{"history": f"{TABLE_NAME}/root"}], [{"history": f"{TABLE_NAME}/domain"}]]}


def _salts(label: bytes, n: int) -> list[bytes]:
    return [b"".join(hashlib.sha512(label + i.to_bytes(4, "big") + c.to_bytes(4, "big")).digest() for c in range(3))
            for i in range(n)]


def _rows(n: int, words: int) -> list[list[int]]:
    return [[(7919 * (r * words + k) + 13) % 65536 for k in range(words)] for r in range(n)]


def _table(program: str, n: int = 5, words: int = 4, cap: int = 8):
    """A table of `n` rows of `words` words at capacity `cap` under `program`, each row under its own salt: its rows,
    salts, `b ‖ c` strings, domain and root."""
    rows, salts = _rows(n, words), _salts(b"history-read-vector/salt", n)
    bcs = [HT.string(r, words, s) for r, s in zip(rows, salts)]
    dom = HT.domain(program, TABLE_NAME, words, cap)
    return rows, salts, bcs, dom, HT.root(dom, bcs, HT.placeholder(words))


def _history(program: str, bcs: list[bytes], root: bytes, words: int = 4, cap: int = 8) -> dict:
    """The verifier's `history` for a table whose rows are one held run of the strings `bcs`, registered at `root`."""
    table = {"name": TABLE_NAME, "program": program, "writer": "none", "out": "none", "words": words, "capacity": cap,
             "first": 0, "units": 0, "members": [],
             "runs": [{"start": 0, "count": len(bcs), "held": "strings", "row": 0}]}
    return {"digests": {}, "tables": [table], "held": {TABLE_NAME: {"strings": [b.hex() for b in bcs]}},
            "registered": {TABLE_NAME: root.hex()}}


def _program() -> Program:
    def sig(S):
        return ((("x", Array(len(POS), Array(2, Value(8)))), ("W", Array(len(TABLE), Array(2, Value(8)))),
                 ("hist", Array(2, G.DIGEST))), Array(len(POS), Value(8)))

    def body(B, S, x, W, hist):
        return coll_concat(sig(S)[1], [B.call(MIX, x[i], W[p]) for i, p in enumerate(POS)])

    return Program(bind(CompositeDefinition("HistoryReadTestProgram", 0, (), sig, body, register=False)))


def _case(hist_rows=None):
    """A record of `_program()` registering W and `hist` (as the prover commits the reads' supplied value, unless
    `hist_rows` says which rows it commits instead), a held-run table, and R7's arguments with its `history`."""
    prog = _program()
    pd = P.program_digest(prog)
    _, _, bcs, dom, root = _table(pd)
    history = _history(pd, bcs, root)
    leaf = HT.leaf_layer({})
    part, n = P.partition(prog, Q), P.units(prog, Q)
    ports = {p: {"leaves": n, "domain": R.window_domain(part["program"], P.digest(part), p, s, n, "r-registered")}
             for p, s in (("x", "hm96-sha512/row/v1"), ("out", "u16"))}
    served = [{"port": p, "root": "ab" * 64, **ports[p]} for p in ports]
    w = G.Registered.commit(pd, "W", TABLE)
    if hist_rows is None:
        resolved = HT.resolve(HT.roots_of(leaf, history), SUPPLIED)
        base = R.record(prog, Q, LAW, SCHEME, served, SERVED, registered=[w.entry()], leaf_layer=leaf)
        hist = G.supplied_value(prog, "hist", G.sources(resolved)["hist"], base, ["hist"])
    else:
        hist = G.Registered.commit(pd, "hist", hist_rows, G.supplied_salts(pd, "hist", 2), word_bits=16)
    rec = R.record(prog, Q, LAW, SCHEME, served, SERVED, registered=[w.entry(), hist.entry()], leaf_layer=leaf)
    check = {"program": prog, "query": Q, "law": LAW, "scheme": SCHEME, "ports": ports, "leaf_layer": leaf,
             "registered": {"W": 2, "hist": 512}, "supplied": SUPPLIED, "history": history}
    return rec, check, hist, root, dom


def test_a_digest_parameter_reads_its_bytes_as_a_circuits_digest():
    """A parameter of whole `Digest`s reads each 64 bytes as a circuit's `Digest` holds them: eight big-endian 64-bit
    words, each least significant bit first, as `digest_bits` does. Every other parameter keeps `bytes_leaves`' 32-bit
    halves, which swap each word's halves against it: PoUW's digests, 16 leaves of 32 bits or 512 of one, are unchanged."""
    data = hashlib.sha512(b"a").digest() + hashlib.sha512(b"b").digest()
    assert G.digest_leaves(data) == digest_bits(data[:64]) + digest_bits(data[64:])
    assert G.digest_leaves(bytes.fromhex("00" * 7 + "01" + "80" + "00" * 55))[:128] == [1] + [0] * 126 + [1]
    halves = G.bytes_leaves(data, 1)
    assert halves != G.digest_leaves(data)
    assert all(halves[64 * i:64 * i + 32] == G.digest_leaves(data)[64 * i + 32:64 * i + 64] for i in range(16))
    assert G.digests(G.DIGEST) and G.digests(Array(3, G.DIGEST))
    for t in (Array(512, Value(1)), Array(16, Value(32)), Array(2, Array(512, Value(1))), Array(8, Array(32, Value(1))),
              Array(2, Array(2, G.DIGEST))):
        assert not G.digests(t), t
    for bad in (b"\0" * 63, b"\0" * 65):
        with pytest.raises(ValueError):
            G.digest_leaves(bad)


def test_a_held_run_table_resolves_into_a_reads_root_and_domain():
    """A history table of one held run (the strings a window publishes for it, no writer units) is held to its registered
    root by `checkAt`, and its root and domain id fill the reads' `hist` rows as `Digest`s: the rows a reader's `root`
    and `domain` ports hold. The prover's commitment of those rows passes in both verifiers. Rows in `bytes_leaves`'
    order, or a domain id of another table, are `R7-supplied`; a held string moved or dropped, or a root that is not the
    table's, is `R7-history`, each verifier's outcome that refusal beside its own reason; in both verifiers."""
    rec, check, hist, root, dom = _case()
    assert [list(r) for r in hist.rows] == [digest_bits(root), digest_bits(dom.domain_id)]
    assert R.check(rec, **check).ok
    got = V.check(rec, **check)
    assert got.ok and got.codes == (), got

    other = HT.domain(P.program_digest(check["program"]), TABLE_NAME, 5, 8).domain_id
    for rows in ([G.bytes_leaves(root, 1), G.bytes_leaves(dom.domain_id, 1)], [digest_bits(root), digest_bits(other)]):
        bad = _case(hist_rows=rows)[0]
        assert R.check(bad, **check).codes == ("R7-supplied",)
        got = V.check(bad, **check)
        assert got.codes == ("R7-supplied",) and got.verifiers["lean"]["codes"] == ["R7-supplied"], got

    history = check["history"]
    strings = history["held"][TABLE_NAME]["strings"]
    for change in ({**history, "held": {TABLE_NAME: {"strings": [strings[1], strings[0], *strings[2:]]}}},
                   {**history, "registered": {TABLE_NAME: "00" * 64}},
                   {**history, "held": {TABLE_NAME: {"strings": strings[:-1]}}}):
        args = {**check, "history": change}
        with pytest.raises(R.HistoryRefused):
            R.check(rec, **args)
        got = V.check(rec, **args)
        assert not got.ok and got.codes == (R.HISTORY,), got
        assert got.verifiers["lean"] == got.verifiers["python"] == {"ok": False, "codes": [R.HISTORY]}, got
        assert got.detail.keys() == {R.HISTORY} and got.detail[R.HISTORY].keys() == {"lean", "python"}, got
        assert all(isinstance(reason, str) and reason for reason in got.detail[R.HISTORY].values()), got


def test_a_history_table_is_r7_history_only_where_the_lean_itself_refused(monkeypatch):
    """A history table the Python refuses is `R7-history` only where the Lean's own answer is a refusal, whatever its
    reason: the detail holds both reasons and nothing holds one to the other. A Lean that can't run, whatever its error
    says, and one that accepts each disagree, and what stood in the Lean's place is recorded: a named refusal is never
    one side's alone."""
    rec, check, _, _, _ = _case()
    args = {**check, "history": {**check["history"], "registered": {TABLE_NAME: "00" * 64}}}
    with pytest.raises(R.HistoryRefused) as refused:
        R.check(rec, **args)
    why = str(refused.value)
    got = V.check(rec, **args)
    assert got.codes == (R.HISTORY,) and got.detail.keys() == {R.HISTORY}, got
    assert got.detail[R.HISTORY]["python"] == why and isinstance(got.detail[R.HISTORY]["lean"], str), got
    both = {"ok": False, "codes": [R.HISTORY]}
    assert got.verifiers == {"of_record": V.RECORD_VERDICT, "lean": both, "python": both}, got

    def answers(answer):
        def call(fn, value):
            if isinstance(answer, Exception):
                raise answer
            return answer
        monkeypatch.setattr(V._server, "call", call)
        return V.check(rec, **args)

    for error in ("the Lean cannot run", why):
        got = answers(V.LeanError(error))
        assert not got.ok and got.codes == (V.DISAGREE,), got
        assert got.verifiers == {"of_record": V.RECORD_VERDICT, "lean": {"refused": error}, "python": {"refused": why}}
        assert got.detail[V.DISAGREE]["lean"] == {"refused": error, "answered": False}, got
    got = answers(("output", {"codes": []}))
    assert not got.ok and got.codes == (V.DISAGREE,) and got.verifiers["lean"] == {"ok": True, "codes": []}, got
    for reason in (why, "the keys of the parameters are not the ones the Lean reads"):
        got = answers(("reject", reason))
        assert not got.ok and got.codes == (R.HISTORY,), got
        assert got.detail == {R.HISTORY: {"lean": reason, "python": why}}, got
        assert got.verifiers == {"of_record": V.RECORD_VERDICT, "lean": both, "python": both}, got


def vector() -> dict:
    """The bytes a committer reproduces: a table at capacity 8 (its rows, salts, `b ‖ c` strings, domain id and root),
    the domain id at `2^17` and 576 words, and the reads' supplied value of the table's root and domain id (each row's
    512 leaves as bits, and its registered entry)."""
    program = hashlib.sha512(b"history-read-vector/program").hexdigest()
    rows, salts, bcs, dom, root = _table(program)
    full = HT.domain(program, TABLE_NAME, 576)
    leaves = [digest_bits(root), digest_bits(dom.domain_id)]
    hist = G.Registered.commit(program, "hist", leaves, G.supplied_salts(program, "hist", 2), word_bits=16)
    return {
        "generator": "verity/protocols/spot/verifier/tests/test_history_read.py",
        "program": program,
        "table": {"name": TABLE_NAME, "words": 4, "capacity": 8, "rows": rows, "salts": [s.hex() for s in salts],
                  "strings": [b.hex() for b in bcs], "placeholder": HT.placeholder(4).hex(),
                  "domain_id": dom.domain_id.hex(), "root": root.hex()},
        "domain_2_17": {"name": TABLE_NAME, "words": 576, "capacity": HT.CAPACITY, "domain_id": full.domain_id.hex()},
        "supplied": {"value": "hist", "type": "Array<2, Digest>", "sources": SUPPLIED["hist"], "word_bits": 16,
                     "rows_bits": ["".join(map(str, r)) for r in leaves],
                     "salts": [s.hex() for s in G.supplied_salts(program, "hist", 2)], "entry": hist.entry()},
    }


def test_the_vector_is_what_the_code_commits():
    """`fixtures/history_read_vector.json` is `vector()`: the bytes a committer tests against are the verifier's."""
    assert json.loads(VECTOR.read_text()) == vector(), (
        "regenerate with `uv run python verity/protocols/spot/verifier/tests/test_history_read.py`")


if __name__ == "__main__":
    VECTOR.write_text(json.dumps(vector(), indent=1) + "\n")
    sys.exit(0)
