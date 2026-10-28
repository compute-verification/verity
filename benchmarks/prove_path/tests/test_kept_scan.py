"""`prove_path.kept_scan`: a served window's scan state and slots off the warden run the certifier front kept under the
serve dir (`certifier_piece.PRIVATE`), on a synthetic warden run at the scan schedule of the class 8,32,4,4,16 (4 lanes,
boundaries at buckets 0 and 16 of a 32-bucket link-window), published as `warden_serve.Publisher` publishes a served one
and kept as the front keeps its fetch. Three requests admitted at bucket 0 are anchored at bucket 16 in lanes 0, 1 and
2, so link-window 0's class Programs 1, 3 and 5 start them (slot nb·j + b), each one's row at its anchor its prompt; a
fourth, admitted at bucket 5 and released in link-window 0, is anchored at 32, so link-window 1's Program 6 starts it.
Each window's state rows are the scan's state tree's for its link-window. A window whose requests start in two link-windows, a request the warden didn't certify, a class whose window isn't the
warden's link-window or whose scan doesn't serve the warden, and a prompt past the class's prompt cap are refused by
name; a serve dir with no kept run gives None, and the unpacked fetch is removed. Reading the run leaves one_stage's
``replay`` (a5's) the process's, not network_traces' (`prove_path.network_traces`)."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tarfile
from pathlib import Path

import pytest

from verity.core.protocols.certifier import ingress_scan as IS
from verity.core.protocols.certifier import warden as W
from verity.core.protocols.certifier.active import id_request
from verity.core.protocols.certifier.certificate import Issuer, LmsSigner
from verity.ml.crypto import lms
from verity_vllm.program.predict.window_circuit import WindowClass

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
import prove_path as PP  # noqa: E402

with PP.network_traces():
    import certifier_piece as CP
    import warden_serve as WS

_spec = importlib.util.spec_from_file_location("scan_traffic", ROOT / "verity/core/protocols/certifier/tests/scan_traffic.py")
ST = sys.modules["scan_traffic"] = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ST)

CLASS = WindowClass(8, 32, 4, lanes=4, every=16)
PROMPTS = ((17, 3, 250, 9), (5, 77, 42), tuple(range(1000, 1032)), (9, 8, 7, 6, 5))
ARRIVALS = (0, 0, 0, 5)
RESPONSES = ((100, 101, 102, 103), (200,), (300, 301), (400, 401, 402))
USERS = [f"user-{i}" for i in range(len(PROMPTS))]
SESSION = bytes(range(32))


def _served() -> ST.Served:
    """`PROMPTS`' warden-form id requests (``user-i``) arriving at `ARRIVALS` and answered `RESPONSES`, served through
    the warden over 2 link-windows at `CLASS`'s scan schedule."""
    scan = CLASS.scan
    params = W.WardenParams(cell_bytes=scan.B, slots=scan.r, bucket_ms=100, delay=scan.D, lanes=scan.lanes,
                            queue=scan.lanes, window=CLASS.buckets, ingress_cells=scan.C, ingress_buckets=scan.WITHIN,
                            egress_cells=scan.KO + 1, every=scan.EVERY)
    reqs = [id_request(model="org/m", user=u, prompt=list(p), max_tokens=4, seed=i, temperature=0.0, top_p=1.0,
                       frame_bytes=scan.B, offset=0) for i, (u, p) in enumerate(zip(USERS, PROMPTS))]
    return ST.serve(params, reqs, ARRIVALS, [1 << 20] * len(reqs), RESPONSES, 2)


@pytest.fixture(scope="module")
def serve_dir(tmp_path_factory) -> Path:
    """A serve dir whose certifier front kept the warden run of `_served`."""
    d, served = tmp_path_factory.mktemp("kept"), _served()
    params = served.params
    run = d / "run"
    run.mkdir()
    signer = LmsSigner.create(lms.keygen(lms.Params(64, 1, 6), hashlib.sha512(b"kept-scan").digest()), d / "signer.json")
    pub = WS.Publisher(run, Issuer(signer, SESSION), params)
    for w in range(2):
        for rec, grid in ((served.ingress[w], params.grid), (served.egress[w], params.grid),
                          (served.failed[w], params.fails_grid)):
            pub.keep(rec, grid)
    dep = WS.deployment_record(params, signer.public_key(), SESSION, origin_unix=0.0, windows=2,
                               listen=("127.0.0.1", 1), upstream=("127.0.0.1", 2), slack=0.025, run_id=bytes(32))
    (run / "deployment.json").write_text(json.dumps(dep))
    serve = d / "serve"
    (serve / CP.PRIVATE / "RUN").mkdir(parents=True)
    with tarfile.open(serve / CP.PRIVATE / "RUN" / "fetch.tar.gz", "w:gz") as tar:
        tar.add(run, arcname="RUN")
    return serve


def test_a_windows_slots_are_the_requests_the_scan_starts_there(serve_dir):
    scan = CLASS.scan
    rows, starts = PP.kept_scan(serve_dir, CLASS, USERS[:3])
    assert starts == [None, "user-0", None, "user-1", None, "user-2", None, None]
    assert len(rows) == CLASS.buckets and rows[0] == IS.encode(scan, IS.empty(scan))
    for k, u in enumerate(starts):
        p = PROMPTS[USERS.index(u)] if u else ()
        want = (1, len(p), p) if u else (0, 0, ())
        assert IS.token_row(scan, IS.decode(scan, rows[CLASS.anchor(k)]))[CLASS.lane(k)] == want
    _, late = PP.kept_scan(serve_dir, CLASS, USERS[3:])
    assert late == [None] * 6 + ["user-3", None]
    assert sorted(p.name for p in (serve_dir / CP.PRIVATE / "RUN").iterdir()) == ["fetch.tar.gz"]
    import replay
    assert Path(replay.__file__).parent.name == "one_stage" and str(ROOT / "benchmarks" / "network_traces") not in sys.path


def test_a_windows_state_rows_are_the_rows_of_the_scans_state_tree(serve_dir):
    """The rows window-commit registers as the window's `SCAN_STATE` are the ones the network certifier's scan puts in
    its state tree for the link-window (`scan_traffic.chain` over the warden's own rows, then
    `ingress_scan.window_state`, as scan_zk's `link` builds it), in link-window 0 and in link-window 1: what a period's
    stateTied needs before it can hold the two at one root."""
    served, T = _served(), CLASS.buckets
    states = ST.chain(CLASS.scan, *served.rows(), served.outputs).states[1:]
    for w, users in ((0, USERS[:3]), (1, USERS[3:])):
        rows, _ = PP.kept_scan(serve_dir, CLASS, users)
        assert [list(r) for r in rows] == IS.window_state(CLASS.scan, states, w * T, T)


def test_a_window_the_kept_run_doesnt_give_is_refused_by_name(serve_dir, tmp_path):
    with pytest.raises(ValueError, match=r"start in the scan's link-windows \['0', '1'\], not one"):
        PP.kept_scan(serve_dir, CLASS, USERS)
    with pytest.raises(ValueError, match="1 of the window's 2 requests are no certified request"):
        PP.kept_scan(serve_dir, CLASS, ["user-0", "user-9"])
    with pytest.raises(ValueError, match="link-windows are 32 buckets, not a window of class .*'s 16"):
        PP.kept_scan(serve_dir, WindowClass(4, 32, 4, lanes=4, every=16), USERS[:3])
    with pytest.raises(ValueError, match="doesn't serve the warden at these parameters"):
        PP.kept_scan(serve_dir, WindowClass(8, 32, 2, lanes=4, every=16), USERS[:3])
    with pytest.raises(ValueError, match="refuses the warden's ingress rows"):
        PP.kept_scan(serve_dir, WindowClass(8, 4, 4, lanes=4, every=16), USERS[:3])
    assert PP.kept_scan(tmp_path, CLASS, USERS) is None
