"""`trace_cache`: a build that reuses traced circuits gives the gates and the Program digest a build without the cache gives, and
an entry that isn't whole, or is another key's, is traced again rather than read."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean import trace as TR
from verity.ml.circuits.ops.boolean import trace_cache
from verity.ml.circuits.types import Array

W8 = Array(8, BL.BIT)


def _mask(S):
    return lambda C, a: [C.AND(a[i], a[(i + S.K) % 8]) for i in range(8)]


def _pair(C, a, b):
    """Bit i is ``(a_i a_{i+1}) b_i ^ a_i``: its ``a_i a_{i+1}`` read only ``a``, the shared operand."""
    return [C.AND(C.AND(a[i], a[(i + 1) % 8]), b[i]) ^ a[i] for i in range(8)]


def build(directory: str) -> dict:
    """A Program over a catalog `specialized` (RowAbsMax_v2), a catalog `definition` (Bf16Add) and this file's `specialized` and
    `shared_operand`, built with the cache at ``directory`` ("" for none): its digest, gates, each traced circuit's hash and the
    cache's counts."""
    cache = trace_cache.use(directory or None)
    from verity.ml.circuits.codec import program_digest
    from verity.ml.circuits.defs import CompositeDefinition, bind
    from verity.ml.circuits.ops.boolean import elementwise as EW, quant as BQ
    from verity.ml.circuits.program import Program
    from verity.ml.circuits.refs import tuple_of
    from verity.ml.circuits.types import Tuple

    mask = EW.specialized("TraceCacheMask", 1, ("K",), _mask, lambda S: ((("a", W8),), W8), register=False)
    own, rest = EW.shared_operand(("TraceCachePairA", "TraceCachePair"), 1, _pair, (("a", W8), ("b", W8)), W8, 0, ("", ""),
                                  register=False)
    amax, add, m3 = bind(BQ.RowAbsMax, N=4, RELU=False), bind(EW.Bf16Add), bind(mask, K=3)
    x_t, bf16 = amax.signature()[0][0][1], amax.signature()[1]
    params = (("x", x_t), ("y", bf16), ("a", W8), ("b", W8))

    def body(B, S, x, y, a, b):
        s = B.call(add, B.call(amax, x), y)
        r = B.call(bind(rest), a, b, B.call(bind(own), a))
        return tuple_of(s, B.call(m3, r))

    prog = Program(bind(CompositeDefinition("TraceCacheProgram", 1, (), lambda S: (params, Tuple(bf16, W8)), body,
                                            register=False)))
    traced = [amax.defn.traced(amax.S()), EW.Bf16Add.traced(), m3.defn.traced(m3.S()), own.traced(), rest.traced()]
    return {"digest": program_digest(prog), "gates": prog.gates,
            "traced": [hashlib.sha256(repr((T.n_inputs, T.gates, T.outputs, T.counts)).encode()).hexdigest() for T in traced],
            "counts": [cache.hits, cache.misses, len(cache.bad), len(cache.unkeyed)] if cache else None}


def _run(directory: str) -> dict:
    p = subprocess.run([sys.executable, __file__, directory], capture_output=True, text=True, check=True)
    return json.loads(p.stdout.strip().splitlines()[-1])


def test_a_warm_build_gives_the_cold_builds_gates_and_digest(tmp_path: Path) -> None:
    plain = _run("")
    cold = _run(str(tmp_path / "traces"))
    warm = _run(str(tmp_path / "traces"))
    assert plain["counts"] is None
    assert cold["counts"] == [0, 4, 0, 0] and warm["counts"] == [4, 0, 0, 0]
    for k in ("digest", "gates", "traced"):
        assert cold[k] == plain[k] and warm[k] == plain[k], k
    assert len(list((tmp_path / "traces").glob("*/*.trace"))) == 4


def test_an_entry_is_read_only_whole_and_under_its_own_key(tmp_path: Path) -> None:
    T = TR.trace(_pair, [W8, W8], W8)
    key = {"name": "TraceCachePair", "version": 1}
    path = tmp_path / "e.trace"
    trace_cache.store(path, key, T)
    back = trace_cache.load(path, key)
    assert (back.n_inputs, back.gates, back.outputs, back.counts) == (T.n_inputs, T.gates, T.outputs, T.counts)
    assert trace_cache.load(path, {**key, "version": 2}) == "another key's entry"
    data = path.read_bytes()
    path.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    assert trace_cache.load(path, key) == "its payload's SHA-512 isn't its header's"
    path.write_bytes(data[:-3])
    assert trace_cache.load(path, key) == "its payload's SHA-512 isn't its header's"
    path.write_bytes(trace_cache.MAGIC + b"{\n")
    assert trace_cache.load(path, key).startswith("unreadable: JSONDecodeError: ")
    assert trace_cache.load(tmp_path / "none.trace", key).startswith("unreadable: FileNotFoundError: ")


def test_a_bad_entry_is_traced_again_and_rewritten(tmp_path: Path, monkeypatch) -> None:
    cache = trace_cache.TraceCache(tmp_path)
    monkeypatch.setattr(TR, "CACHE", cache)
    key = ("TraceCachePair", 1, {})
    T = TR.trace(_pair, [W8, W8], W8, key=key)
    (path,) = tmp_path.glob("*/*.trace")
    path.write_bytes(path.read_bytes()[:-1])
    again = TR.trace(_pair, [W8, W8], W8, key=key)
    assert (cache.hits, cache.misses, cache.bad) == (0, 2, [f"{path.name}: its payload's SHA-512 isn't its header's"])
    assert again.gates == T.gates and isinstance(trace_cache.load(path, cache.key(key, _pair, [W8, W8], W8, True)), TR.Traced)


def test_a_trace_it_cannot_key_is_traced_and_recorded(tmp_path: Path, monkeypatch) -> None:
    cache = trace_cache.TraceCache(tmp_path)
    monkeypatch.setattr(TR, "CACHE", cache)
    T = TR.trace(_pair, [W8, W8], W8, key=("TraceCachePair", 1, {"F": object()}))
    assert T.gates == TR.trace(_pair, [W8, W8], W8).gates
    assert (cache.hits, cache.misses, cache.bad) == (0, 0, [])
    assert len(cache.unkeyed) == 1 and cache.unkeyed[0].startswith("TraceCachePair: a binding has no static form (")
    assert not list(tmp_path.iterdir())


def test_a_builder_the_code_digest_skips_is_keyed_by_its_file() -> None:
    """`code_digest` leaves out tests, so a builder in one (as here) names its own file in the key."""
    from verity.ml.circuits.ops.boolean import elementwise as EW
    assert trace_cache._source(_pair) == _pair.__code__.co_filename
    assert trace_cache._source(EW._split) == ""


if __name__ == "__main__":
    print(json.dumps(build(sys.argv[1])))
