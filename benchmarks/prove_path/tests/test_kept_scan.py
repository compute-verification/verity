"""`prove_path.kept_scan`: a served window's scan state and slots off the network certifier's run the certifier front kept
under the serve dir (`certifier_piece.PRIVATE`), on a synthetic run at the scan schedule of the class 8,32,4,4,16 (4 lanes,
boundaries at buckets 0 and 16 of a 32-bucket link-window), published as `warden_serve.Publisher` publishes a served one
and kept as the front keeps its fetch. Three requests admitted at bucket 0 are anchored at bucket 16 in lanes 0, 1 and
2, so link-window 0's class Programs 1, 3 and 5 start them (slot nb·j + b), each one's row at its anchor its prompt; a
fourth, admitted at bucket 5 and released in link-window 0, is anchored at 32, so link-window 1's Program 6 starts it.
Each window's state rows are the scan's state tree's for its link-window. A window whose requests start in two
link-windows, a request the network certifier didn't certify, a class whose window isn't its link-window or whose scan
doesn't serve it, and a prompt past the class's prompt cap are refused by name; so are a kept run that won't read, one
whose certified I/O isn't accepted or whose ingress link-windows aren't 0, 1, …, and certified I/O read with a request
that has no END or without a request the scan starts. A serve dir with no kept run gives None, and the unpacked fetch
is removed. Reading the run leaves ``sys.path`` and the module ``replay`` as they were (`prove_path.network_traces`),
so a5's ``import replay`` still finds one_stage's where it did before. `prove_path.place_slots` places the window's
slots off the scan's starts and rows without serving, and refuses by name a request started in two slots, a served
request started in none, a started request that wasn't served, and a slot whose prompt or ``max_tokens`` isn't its
served request's."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
import tarfile
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

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
    import warden_chain as WCH
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


def _served(users: Sequence[str] = USERS) -> ST.Served:
    """`PROMPTS`' id requests (of ``users``, `USERS` unless given) arriving at `ARRIVALS` and answered `RESPONSES`,
    served through the network certifier over 2 link-windows at `CLASS`'s scan schedule."""
    scan = CLASS.scan
    params = W.WardenParams(cell_bytes=scan.B, slots=scan.r, bucket_ms=100, delay=scan.D, lanes=scan.lanes,
                            queue=scan.lanes, window=CLASS.buckets, ingress_cells=scan.C, ingress_buckets=scan.WITHIN,
                            egress_cells=scan.KO + 1, every=scan.EVERY)
    reqs = [id_request(model="org/m", user=u, prompt=list(p), max_tokens=4, seed=i, temperature=0.0, top_p=1.0,
                       frame_bytes=scan.B, offset=0) for i, (u, p) in enumerate(zip(users, PROMPTS, strict=True))]
    return ST.serve(params, reqs, ARRIVALS, [1 << 20] * len(reqs), RESPONSES, 2)


def _kept(d: Path, served: ST.Served, records: Callable[[str, Any], Any] = lambda kind, rec: rec,
          deployment: Callable[[dict], dict] = lambda dep: dep) -> Path:
    """A serve dir under ``d`` whose certifier front kept the network certifier's run of ``served``, each of its records
    as ``records`` gives it (by its kind, ``ingress``, ``egress`` or ``failed``) and its deployment.json as
    ``deployment`` gives it."""
    params = served.params
    run = d / "run"
    run.mkdir(parents=True)
    signer = LmsSigner.create(lms.keygen(lms.Params(64, 1, 6), hashlib.sha512(b"kept-scan").digest()), d / "signer.json")
    pub = WS.Publisher(run, Issuer(signer, SESSION), params)
    for w in range(2):
        for kind, rec, grid in (("ingress", served.ingress[w], params.grid), ("egress", served.egress[w], params.grid),
                                ("failed", served.failed[w], params.fails_grid)):
            pub.keep(records(kind, rec), grid)
    dep = WS.deployment_record(params, signer.public_key(), SESSION, origin_unix=0.0, windows=2,
                               listen=("127.0.0.1", 1), upstream=("127.0.0.1", 2), slack=0.025, run_id=bytes(32))
    (run / "deployment.json").write_text(json.dumps(deployment(dep)))
    serve = d / "serve"
    (serve / CP.PRIVATE / "RUN").mkdir(parents=True)
    with tarfile.open(serve / CP.PRIVATE / "RUN" / "fetch.tar.gz", "w:gz") as tar:
        tar.add(run, arcname="RUN")
    return serve


@pytest.fixture(scope="module")
def serve_dir(tmp_path_factory) -> Path:
    """A serve dir whose certifier front kept the network certifier's run of `_served`."""
    return _kept(tmp_path_factory.mktemp("kept"), _served())


def _imports() -> tuple:
    """``sys.path``, the module ``replay`` and the file ``import replay`` finds."""
    return list(sys.path), sys.modules.get("replay"), getattr(importlib.util.find_spec("replay"), "origin", None)


def test_a_windows_slots_are_the_requests_the_scan_starts_there(serve_dir):
    scan, imports = CLASS.scan, _imports()
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
    assert _imports() == imports


def test_a_windows_state_rows_are_the_rows_of_the_scans_state_tree(serve_dir):
    """The rows window-commit registers as the window's `SCAN_STATE` are the ones the network certifier's scan puts in
    its state tree for the link-window (`scan_traffic.chain` over its own rows, then `ingress_scan.window_state`, as
    scan_zk's `link` builds it), in link-window 0 and in link-window 1: what a period's check needs before it can tie
    the two to one root."""
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
    with pytest.raises(ValueError, match="refuses the network certifier's ingress rows"):
        PP.kept_scan(serve_dir, WindowClass(8, 4, 4, lanes=4, every=16), USERS[:3])
    assert PP.kept_scan(tmp_path, CLASS, USERS) is None


def test_a_kept_run_that_wont_read_or_whose_io_or_link_windows_arent_whole_is_refused_by_name(serve_dir, tmp_path):
    """The kept run edited before it's packed: two kept runs, or a deployment.json whose digest isn't its parameters',
    don't read; two requests naming one user leave its certified I/O unaccepted; and its second link-window's ingress
    record kept as link-window 2 leaves its ingress link-windows 0 and 2."""
    two = shutil.copytree(serve_dir, tmp_path / "two")
    shutil.copytree(two / CP.PRIVATE / "RUN", two / CP.PRIVATE / "RUN2")
    with pytest.raises(ValueError, match="the certifier front's kept run: 2 warden runs kept"):
        PP.kept_scan(two, CLASS, USERS[:3])
    undigested = _kept(tmp_path / "digest", _served(), deployment=lambda dep: dep | {"digest": "00" * 64})
    with pytest.raises(ValueError, match="the certifier front's kept run: .*deployment.json's digest isn't its param"):
        PP.kept_scan(undigested, CLASS, USERS[:3])
    twice = _kept(tmp_path / "twice", _served([*USERS[:3], USERS[0]]))
    with pytest.raises(ValueError, match="the kept run's certified I/O doesn't read whole off its cells"):
        PP.kept_scan(twice, CLASS, USERS[:3])
    skipped = _kept(tmp_path / "skipped", _served(),
                    records=lambda kind, rec: replace(rec, window=2) if kind == "ingress" and rec.window == 1 else rec)
    with pytest.raises(ValueError, match=r"the kept run's ingress records are link-windows \[0, 2\], not 0, 1, …"):
        PP.kept_scan(skipped, CLASS, USERS[:3])
    assert not list((tmp_path / "skipped" / "serve" / CP.PRIVATE / "RUN").glob("unpacked-*"))


def test_an_accepted_ios_request_with_no_end_or_a_start_it_doesnt_hold_is_refused_by_name(serve_dir, monkeypatch):
    """A request of the window with no END, and a connection the scan starts whose request the certified I/O doesn't
    hold. No kept run whose certified I/O `warden_chain.io` accepts has either: an accepted request is a whole id
    request, whose last bytes the network certifier sends in an END cell (an id request never fills its last cell),
    and accepted I/O holds every connection whose cells carry a request. So it is `io`'s reading of the kept run that is
    edited."""
    io = WCH.io
    monkeypatch.setattr(WCH, "io", lambda run: (got := io(run)) | {
        "records": [r | {"release": None} if r["user"] == USERS[0] else r for r in got["records"]]})
    with pytest.raises(ValueError, match="a request of the window has no END in the kept run's ingress"):
        PP.kept_scan(serve_dir, CLASS, USERS[:3])
    monkeypatch.setattr(WCH, "io", lambda run: (got := io(run)) | {
        "records": [r for r in got["records"] if r["user"] != USERS[2]]})
    with pytest.raises(ValueError, match=r"the scan starts connections \[3\], whose requests the kept run's certified"):
        PP.kept_scan(serve_dir, CLASS, USERS[:2])


def _heads(starts: Sequence[str | None]) -> list[tuple[list[int], list[int]]]:
    """Each slot's row and head as `window_rows.scan_slots` reads them off the scan's state, for a scan that starts
    ``starts`` (the I1 ids of `USERS`, None for an empty slot) with their prompts and ``max_tokens`` 4."""
    out = []
    for u in starts:
        p = PROMPTS[USERS.index(u)] if u else ()
        out.append(([int(u is not None), len(p), *p, *[0] * (CLASS.prompt - len(p))], [len(p), 4 if u else 0]))
    return out


def test_a_windows_slots_hold_the_requests_the_scan_starts_or_are_refused_by_name():
    """`prove_path.place_slots`, window-commit's slots, without serving. With the scan's starts and rows, each slot
    holds the request the scan starts there, with the scan's row and ``max_tokens``; without them, slot k holds the
    k-th served request, with its own. Refused by name: one request started in two slots; a served request started in
    none; a certified request started that wasn't served; and a slot whose prompt or ``max_tokens`` isn't its served
    request's."""
    ingress = {u: {"prompt": list(p), "max_tokens": 4} for u, p in zip(USERS, PROMPTS, strict=True)}
    starts = [None, "user-0", None, "user-1", None, "user-2", None, None]
    heads = _heads(starts)

    def slots(placed: Sequence[str | None]) -> list[tuple[str | None, list[int], int]]:
        return [(u, h[0], h[1][1]) for u, h in zip(placed, _heads(placed), strict=True)]

    assert PP.place_slots(CLASS, USERS[:3], ingress, (starts, heads)) == slots(starts)
    assert PP.place_slots(CLASS, USERS[:3], ingress) == slots([*USERS[:3], *[None] * 5])
    with pytest.raises(ValueError, match="the scan starts one request in two slots of the window"):
        PP.place_slots(CLASS, USERS[:3], ingress, ([*starts[:7], "user-0"], _heads([*starts[:7], "user-0"])))
    with pytest.raises(ValueError, match="1 of the window's 4 served requests start in none of its slots"):
        PP.place_slots(CLASS, USERS, ingress, (starts, heads))
    with pytest.raises(ValueError, match=r"starts certified requests in the window that it didn't serve: \['user-2'\]"):
        PP.place_slots(CLASS, USERS[:2], ingress, (starts, heads))
    other = [list(h[0]) for h in heads]
    other[3][3] += 1
    with pytest.raises(ValueError, match="slot 3: the scan's request isn't the served one"):
        PP.place_slots(CLASS, USERS[:3], ingress, (starts, [(r, h[1]) for r, h in zip(other, heads, strict=True)]))
    longer = [(h[0], [h[1][0], 5]) if k == 5 else h for k, h in enumerate(heads)]
    with pytest.raises(ValueError, match="slot 5: the scan's request isn't the served one"):
        PP.place_slots(CLASS, USERS[:3], ingress, (starts, longer))
