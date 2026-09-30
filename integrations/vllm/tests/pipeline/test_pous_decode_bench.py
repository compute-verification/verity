"""`verity-vllm pous-decode-bench` takes the scheme (``--scheme``, #188's ``--codec`` the same flag), and `pous-dense-audit`, the
verifier its ``--audit`` starts, audits a live responder under that scheme through `pipeline/pous_verifier.run`: here one band
segment of pseudo-random `C` at the served geometry, served by `test_pous_verifier`'s stand-in responder.  No GPU."""
from __future__ import annotations

import hashlib
import json
import socket
import threading

from verity_pous.verifier import leaf_hash

from tests.pipeline.test_pous_verifier import _serve
from verity_vllm.engine.pous import twins
from verity_vllm.pipeline import cli, pous_dense_audit

BAND = "band-chain/d12/v1"


def test_the_bench_takes_the_scheme_and_codec_is_the_same_flag():
    common = ["--out", "/o", "--shapes", "shapes.json", "--build-dir", "/b"]
    assert cli.parse("pous-decode-bench", common).scheme == BAND
    assert cli.parse("pous-decode-bench", [*common, "--codec", "dense-chain/v1"]).scheme == "dense-chain/v1"
    assert cli.parse("pous-decode-bench", [*common, "--scheme", "dense-chain/v1"]).scheme == "dense-chain/v1"


def test_the_audit_verifier_audits_the_live_responder_under_the_scheme(tmp_path):
    params = twins.reference(BAND)[1].with_segments(1)
    salt = bytes(range(24))
    blocks = [hashlib.shake_256(b"c/%d" % i).digest(params.block_bytes) for i in range(params.blocks)]
    (tmp_path / "leaves.bin").write_bytes(b"".join(leaf_hash(params, salt, i, b) for i, b in enumerate(blocks)))

    def audit(served: list[bytes]) -> tuple[int, dict]:
        sock = socket.create_server(("127.0.0.1", 0))
        t = threading.Thread(target=_serve, args=(sock, served), daemon=True)
        t.start()
        (tmp_path / "ready.json").write_text(json.dumps({"port": sock.getsockname()[1], "salt": salt.hex(), "scheme": BAND,
                                                         "segments": 1}))
        opts = cli.parse("pous-dense-audit", ["--ready", str(tmp_path / "ready.json"), "--leaves", str(tmp_path / "leaves.bin"),
                                              "--out", str(tmp_path / "audit.json"), "--audits", "1", "--pings", "20"])
        try:
            rc = pous_dense_audit.main(opts)
        finally:
            t.join()
            sock.close()
        return rc, json.loads((tmp_path / "audit.json").read_text())

    rc, doc = audit(blocks)
    assert rc == 0 and doc["scheme"] == BAND and doc["k"] == 111 and doc["audit_of_record"]["accepted"]
    rc, doc = audit([bytes(len(b)) if i % 2 else b for i, b in enumerate(blocks)])
    assert rc == 1 and not doc["audit_of_record"]["accepted"]
