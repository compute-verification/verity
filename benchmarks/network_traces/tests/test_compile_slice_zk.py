"""compile_slice_zk's scan of what the verifier received is `zk_scan`'s (tests/test_zk_scan.py), and each found piece's
classes are the witness fields it sets (`classes`, `scanned`). Nothing here lowers or proves (`compile_slice_zk.prove`
does, on a node)."""

from __future__ import annotations

import importlib.util
import itertools
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parents[1]


def _load():
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location("network_traces_compile_slice_zk", HERE / "compile_slice_zk.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


Z = _load()


def test_classes_are_the_witness_fields_a_piece_sets():
    """On the demonstration class's honest unit, which takes real prefix hits, the layout tiles each row; a piece holding
    a nonzero hit sets ``wit.hit``, one holding a nonzero walk entry ``wit.cache.pfx``, and an all-zero piece none."""
    CS = Z.CS
    st, sl = CS.demo()
    S = SimpleNamespace(**st)
    c = CS.case(S, sl)
    unit = SimpleNamespace(words={r: CS.words(c.rows[r]) for r in CS.ROWS})
    for r in ("wit", "srv"):
        lay = Z.layout(S, r)
        assert [s for _c, s, _w in lay] == list(itertools.accumulate([0] + [w for *_, w in lay[:-1]]))
        assert sum(w for *_, w in lay) == CS.row_bits(S)[r]
    lay, bits = Z.layout(S, "wit"), c.rows["wit"]
    b = b"".join(x.to_bytes(2, "little") for x in unit.words["wit"])
    at = {name: next(s // 128 * 16 for c_, s, w in lay if c_ == name and any(bits[s:s + w]))
          for name in ("wit.hit", "wit.cache.pfx")}
    got = Z.classes(S, [unit], {b[o:o + 16] for o in at.values()} | {bytes(16)})
    assert got[bytes(16).hex()] == []
    for name, o in at.items():
        assert name in got[b[o:o + 16].hex()]


def test_scanned_gives_each_found_piece_its_classes():
    """A found piece, excluded or leaked, is named with the witness classes it sets; a salt sets none."""
    CS = Z.CS
    st, sl = CS.demo()
    S = SimpleNamespace(**st)
    c = CS.case(S, sl)
    unit = SimpleNamespace(words={r: CS.words(c.rows[r]) for r in CS.ROWS})
    blobs, _held = Z.blobs_of([unit], {r: [bytes(range(16))] for r in ("adv", "cert", "srv", "wit")})
    ps = Z.ZS.pieces(blobs)[0]
    hit = next(k for k, x in ps.items() if k.startswith("wit 0 ") and "wit.hit" in Z.classes(S, [unit], {x})[x.hex()])
    got = Z.scanned(S, [unit], {"verifier_found": [hit, "adv salt 0 @0"], "leaked": [hit]}, blobs)
    assert "wit.hit" in got["found_classes"][hit] and got["leaked"] == [hit]
    assert got["found_classes"]["adv salt 0 @0"] == []
