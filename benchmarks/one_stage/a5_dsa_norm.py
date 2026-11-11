"""One-stage audit, run A5, on a DSA RMSNorm stand-in: two ``DsaRMSNorm_v1{N=512,EPS=1e-05}`` rows in decoder layers, the
row of `adapters/vllm/tests/pipeline/test_serving_rows_norm_split.py`, cut at their scale as serving cuts them
(`pipeline.serving_templates`' ``PARTS``): per row its sum of squares once and its apply in eight blocks of 64 columns, each
block's scale row wired to the output row of its row's sum of squares (the fixture's ``wired``). The units are the Boolean bodies
written as Calls, ``DsaRMSNormSumSq_v3`` and ``DsaRMSNormApply_v3``, at statics GLM-5.3 serves. Its run is ``<ROOT>/rank0``,
committed as serving commits it (`pipeline.serving_rows.read_all` on the fixture's served values, each row's scale by its
twin, `ml.windows.rows.commit`) and audited by `a5_sparse_mla.main`, as that stand-in is: rank 0 of a one-rank step on a
local challenger's coins under ``stratified-rate:1/64``, at least one unit of each template; the wiring check, each template's
flat circuit, M0's `--zk` session of each sampled proof block, the Lean verifier, the stream's outcome and the negatives. The
run exits 0 only when the outcome is accepted and every negative refused. Its arguments are `a5_sparse_mla.py`'s.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "adapters" / "vllm"))

import a5  # noqa: E402
import a5_sparse_mla as SM  # noqa: E402
from tests.pipeline import test_serving_rows_norm_split as TN  # noqa: E402
from verity.circuits.codec import descriptor_id  # noqa: E402
from verity.circuits.defs import bind  # noqa: E402
from verity.commitments.windows import rows as CS  # noqa: E402
from verity.definitions.registry import boolean_dsa_norm as BD  # noqa: E402
from verity.protocols.spot.verifier.one_stage import registration as R  # noqa: E402
from verity_vllm.commit import index as IX  # noqa: E402
from verity_vllm.pipeline import serving_rows as SRW  # noqa: E402

FAMILY = "DsaRMSNorm_v1"
ROOT = hashlib.sha512(b"verity/a5/dsa-norm-stand-in/v1").hexdigest()
#: the sum of squares and the apply block written as Calls, and each one's units a chunk: a row's sum of squares, its 8 blocks
TMPLS = [descriptor_id(bind(BD.DsaRMSNormSumSqV3, N=TN.N, EPS=TN.EPS)), descriptor_id(bind(BD.DsaRMSNormApplyV3, G=TN.G))]
CHUNKS = dict(zip(TMPLS, (1, TN.N // TN.G)))
OF = {"fixture": "adapters/vllm/tests/pipeline/test_serving_rows_norm_split.py", "row": descriptor_id(TN.FAMILIES[FAMILY][0]()),
      "templates": TMPLS}


class Served:
    """``n`` rows of the family, row p in decoder layer p (its x and w are arguments 2p and 2p + 1, Gaussian bf16 rows), behind a
    ProgramIndex stand-in (``index``), and the store of their served outputs, the row's word Definition's."""

    def __init__(self, n: int = 2, seed: int = 7):
        fn, groups, members, _sumsq, _apply, layer = TN.FAMILIES[FAMILY]
        fn = fn()
        self.ops, self.out, described = {}, {}, {}
        self.index = SimpleNamespace(pnames=["weights"], by_i={}, describe=lambda r: described[r["i"]])
        for p in range(n):
            rng = np.random.default_rng(seed + p)
            vals = [TN.bf16s(rng, k, 0.1 if name == "w" else 1.0) for name, k in groups]
            op = layer.format(L=p)
            self.out[op] = dict(zip(members, TN.ev(fn, [v[None] for v in vals]).astype(np.uint16).reshape(len(members), TN.N)))
            self.ops.update({2 * p + g: v for g, v in enumerate(vals)})
            r = {"i": 7 + p, "spec": descriptor_id(fn), "args": [[["x", 2 * p + g, 0, v.size]] for g, v in enumerate(vals)]}
            d = {"statics": dict(fn.bindings), "op_path": op, "members": [(m, TN.N, "bf16") for m in members], "row": 0, "step": 16}
            self.index.by_i[r["i"]], described[r["i"]] = r, d

    def slices(self, _pi, _store, _weights, _rid, _arrive, group, **_kw):
        """`serving_rows._slices`: an argument group's elements."""
        return [self.ops[group[0][1]][:group[0][3]]]

    def words(self, rid, step, op, member, lo, hi):
        return self.out[op][member][lo:hi], None


def window(s: Served) -> tuple[CS.Binding, list]:
    """The window of ``s``'s rows over ``TMPLS`` under ``<ROOT>/rank0`` and the stand-in's rate, cut and wired as the fixture
    cuts it (``wired``: each apply block's scale to its row's sum of squares), and each template's rows."""
    SRW.index_of, SRW._slices = (lambda _program, _pool, **_kw: s.index), s.slices
    ms, rows = TN.wired(TMPLS, CHUNKS)
    defs = [(SRW.template(t).fn, sum(m.n for m in ms if m.template == t)) for t in TMPLS]
    return a5.rated(CS.Binding(*CS.population_program(defs, several=True), {}, f"{ROOT}/rank0", tuple(ms)), SM.RULE), rows


def commit(out: Path) -> CS.Binding:
    """The fixture's window under ``<ROOT>/rank0`` committed into ``out`` as serving commits it, its salts fresh."""
    s = Served()
    w, rows = window(s)
    got = SRW.read_all(w.members, rows, s, None)
    c = CS.commit(w, [g for g, _e in got])
    CS.write(c, out, {"of": OF, "committer": {"module": "a5_dsa_norm.py"}}, IX.index_doc(w.members, [e for _g, e in got], w.run),
             kind="stand-in", registrar=R)
    return w


if __name__ == "__main__":
    raise SystemExit(SM.main(commit, ROOT, OF, "A5-DSA-NORM"))
