"""What serving commits on a served Program is the canonical partition's committed set (`verity.ml.circuits.partition_object`).

The manifest names the Program's partition of record as a `verity/partition/v1` object (`required.partition_of`: Q_word v1,
X = 16, W = 32), and the object derives what each Call commits: every gate another unit of the Call's cut reads and every gate the
Call returns.  Serving decides the Call outputs (the boundary of `query.units` with every Call a unit, plus the policy:
`Population`) and the word cut the interior words (`verity.ml.definitions.word`).  Those commits, per Call in `CallGraph`'s
numbering, are what `partition_object.verify` holds to the object the manifest records, on TINY's derived greedy serve.

The boundary alone is not the canonical set: a Value no Call reads and the Program does not return is no boundary Value, while the
object commits every Call output.  On this Program four are (three fused norms' residuals nothing adds to, and the last layer's down
projection of the first prompt row, whose final norm the serve skips), and the policy (TENSOR-RETURN) requires each.  A Call the
policy leaves out too is `Population.unreturned`: the word check skips it, the manifest lists it, and the object would commit it."""
from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from verity.ml.circuits import partition_object as PO  # noqa: E402
from verity.ml.circuits.codec import descriptor_id  # noqa: E402
from verity.ml.definitions import word as W  # noqa: E402
from verity_vllm.correspondence.reader_for_query import Correspondence  # noqa: E402
from verity_vllm.pipeline.manifest import recorded_program_digest  # noqa: E402
from verity_vllm.query.program_view import from_instances  # noqa: E402
from verity_vllm.query.required import partition_of, request_manifest  # noqa: E402
from verity_vllm.query.units import ValueRef  # noqa: E402
from tests.program.test_serve3 import TINY  # noqa: E402
from tests.query.test_boolean_manifest import _write  # noqa: E402


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    """TINY's greedy serve, derived as serving derives it and written as the Build writes a request Program dir."""
    from verity_vllm.program.frontend import derive_examples as ex
    from verity_vllm.program.frontend.derive import derive_report

    with torch.device("meta"):
        m = ex.GreedyServe(TINY, 1)
    rep = derive_report(m, (torch.empty(2, dtype=torch.int32, device="meta"),), mode="profile", profile="b1-eager", config=TINY,
                        name="Serve", registered_values=True)
    assert rep.ok, rep.unsupported
    d = tmp_path_factory.mktemp("tiny")
    _write(d, rep)
    V = from_instances(str(d / "instances.json.gz"))
    res = json.loads((d / "result.json").read_text())
    return rep.program, V, Correspondence.of(V, str(d)), res, recorded_program_digest(str(d), res)


def _manifest(served, **kw):
    _program, V, corr, res, digest = served
    return request_manifest(V, corr, res, request_id="r0", program_digest=digest, **kw)


def _commits(program, V, required) -> dict[int, list[int]]:
    """Root-body index -> the gates serving commits for that Call, as `CallGraph` numbers them: the word cut's committed interior
    gates and the returned gates of every port `required` holds."""
    out = {}
    for (k, fn), c in zip(PO.calls(program), V.calls):
        assert descriptor_id(fn) == c.definition
        G = W.Graph(fn)
        assert len(G.outputs) == c.port_table[-1].hi and not G.returned_inputs, f"{c.definition}: not one returned gate per output leaf"
        _owner, _kinds, head = W.partition(G, 16)
        interior = set(np.flatnonzero(head >= 0).tolist()) - set(G.outputs)
        ports = [o for p in c.port_table if ValueRef(c.id, p.name) in required for o in G.outputs[p.lo:p.hi]]
        out[k] = sorted(interior | set(ports))
    return out


def test_what_serving_commits_is_the_canonical_partitions_committed_set(served):
    program, V, _corr, _res, digest = served
    r = _manifest(served)
    part = r.manifest["query"]["partition"]
    assert part == partition_of(digest) and part["object"]["program"] == digest
    assert not r.unreturned and len(V.calls) == len(PO.calls(program)) == 75
    v = PO.verify(part["object"], program, _commits(program, V, r.required.values))
    assert v.ok, (v.codes, v.detail)


def test_the_boundary_alone_leaves_out_the_outputs_nothing_reads_which_the_policy_requires(served):
    program, V, _corr, _res, digest = served
    plain = _manifest(served, return_slots=False)
    unread = {ValueRef(c.id, p.name) for c in V.calls for p in c.port_table} - plain.required.values
    assert sorted((V.call(u.call).family, u.port) for u in unread) == [("Gemm_v1", "out")] + [("RMSNormFusedCuda_v4", "1")] * 3
    v = PO.verify(partition_of(digest)["object"], program, _commits(program, V, plain.required.values))
    root = {c.id: k for (k, _fn), c in zip(PO.calls(program), V.calls)}
    assert v.codes == ("cut",) and {f["call"] for f in v.detail["cut"]["gates"]} == {root[u.call] for u in unread}
    assert all("output-uncommitted" in f["codes"] for f in v.detail["cut"]["gates"])
    assert unread <= _manifest(served).required.protocol
