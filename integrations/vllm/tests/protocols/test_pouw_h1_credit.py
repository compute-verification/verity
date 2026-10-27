"""`pouw_h1_credit`, `pouw_h1_gpu.py --credit`'s check, on a small window served with the real tile unit and receipts from
`h1tile.reference` (slow: the window Program's real tile unit takes half a minute to declare): core credit's statement
steps accept the honest window, and refuse each prover-side negative by name."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from verity.core.protocols.pouw import credit as CR  # noqa: E402
from verity.core.protocols.pouw.circuit import h1tile as HT  # noqa: E402
from verity.core.protocols.pouw.circuit import h1window as HW  # noqa: E402

from tests.protocols import pouw_h1_credit as PC  # noqa: E402

SALT = hashlib.sha512(b"salt").digest()


def _window(j10r: bool = False):
    """Two steps over q, k (K = 64) and o (K = 128), 20 rows served at step 0 and none at step 1; each call's receipts
    `h1tile.reference`'s (with ``j10r``, a window over `h1tile.PouwH1Tile5` and `h1tile.j10r_reference`'s) over its
    padded codes and weight, seeded from the run's salt, its units' positions and its registered roots. The run, each
    call's operands by index and each call's receipt words by index."""
    from verity_vllm.engine import int8_ops as OPS
    from verity_vllm.protocols.pouw import h1 as PH
    from verity_vllm.protocols.pouw.h1 import served as SV
    shapes, blocks = [(64, 64), (32, 64), (32, 128)], (0, 0, 1)
    served = SV.Served(blocks=blocks, window=hashlib.sha512(b"window").digest(),
                       unit=(lambda k: HW.tile_unit(k, (32, 32), HT.PouwH1Tile5)) if j10r else None)
    held: dict = {}
    words: dict = {}

    def receipts(call, codes, s_a, w, s_w):
        u0, n = HW.call_units(served.program)[call.index]
        pre = served.program.fn.bindings["PREFIX"]
        roots = (served.registered[served.a_table(call.index)].root, served.registered[pre + f"b/{call.k}"].root)
        a = np.zeros((served.m, call.k), np.int8)
        a[:len(codes)] = codes
        held[call.index] = (a, np.asarray(w, np.int8))
        ncb = call.n // 32

        def one(u):
            i, j = divmod(u, ncb)
            ta = a[32 * i:32 * (i + 1)].view(np.uint8).tolist()
            tb = held[call.index][1][32 * j:32 * (j + 1)].view(np.uint8).tolist()
            if j10r:
                return HT.j10r_reference(ta, tb, run.salt, u0 + u, roots, u0,
                                         HT.j10r_blocks(served.m, call.k, (32, 32), i, j), pairs=[(0, 0)])[1]
            return HT.reference(ta, tb, run.salt, u0 + u, roots, pairs=[(0, 0)])[1]
        out = np.array([one(u) for u in range(n)], np.uint32)
        words[call.index] = SV.receipt_words(out)
        return out

    served.receipts = receipts
    run = PH.H1Run("t", SALT, {0: "q", 1: "k", 2: "o"}, shapes, 64, 2, key=bytes([7]) * 32, served=served)
    rng, g = np.random.default_rng(5), torch.Generator().manual_seed(5)
    ws = [(torch.from_numpy(rng.integers(-127, 128, (n, k), dtype=np.int8)), torch.rand(n, 1, generator=g) + 0.5)
          for n, k in shapes]
    for m in (20, 0):
        xs = {}
        for w, ((n, k), (wt, sw)) in enumerate(zip(shapes, ws)):
            x = xs.setdefault(blocks[w], torch.from_numpy(rng.integers(-127, 128, (m, k), dtype=np.int8)))
            s_a = torch.rand(m, generator=g) + 0.5
            run.record(w, x, s_a, wt, sw, z=OPS.scaled_mm_zy(x, s_a, wt, sw, 64)[1])
    run.close()
    assert not served.refused, served.refused
    assert HW.is_j10r(served.program) == j10r
    return run, held, words


@pytest.mark.slow
@pytest.mark.parametrize("j10r", [False, True], ids=["v1", "j10r"])
def test_credit_accepts_the_honest_window_and_refuses_each_negative_by_name(j10r):
    run, held, words = _window(j10r)
    w = PC.of_run(run, run.served)
    assert PC.bc_of(PC.CC.OK_ROW, PC.CC.ok_salt(w.window, 0)) == w.ok.bcs[0]
    units = [w.u0(0), w.u0(0) + 3, w.u0(2) + 1, w.u0(5)]
    got = PC.check(w, units, lambda c: held[c], lambda c: words[c], run._key)
    print("\n".join(PC.lines(got)))
    assert got["honest"]["accepted"], got["honest"]
    negs = got["negatives"]
    assert negs["receipt-word"]["a_link"] is None and negs["receipt-word"]["refused"][:2] == ["verified", "failed"]
    assert "o0 row is not the receipt it computes" in negs["receipt-word"]["rejected"][0]
    assert negs["receipt-word-after-round"]["refused"][:2] == ["a_link", "a-link"]
    assert "receipt table" in negs["receipt-word-after-round"]["refused"][2]
    for kind, port in (("pub-other-root", CR.PUB_PORT), ("ok-zero", CR.OK_PORT)):
        assert negs[kind]["refused"][:2] == ["verified", "failed"], negs[kind]
        assert f"{port} row does not open" in negs[kind]["rejected"][0], negs[kind]
    assert got["ok"]
