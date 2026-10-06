"""The inner session's proxy and coin server (`verity_flock.rec_live`): the wire codec, how a round's frames are sorted into
public caps and the private free message, the message rows' commitments, and that the coin server is shown nothing of the
hidden statement while the proxy's session record keeps what the live server's would."""
from __future__ import annotations

import hashlib
import json
import struct

import pytest
from verity_flock import rec_live as RL
from verity.protocols.verification.sampled_proofs.one_stage import registered as G


def _counter_rand():
    state = {"i": 0}

    def rand(n: int) -> bytes:
        out = b""
        while len(out) < n:
            out += hashlib.sha512(b"rec-live-test" + state["i"].to_bytes(8, "little")).digest()
            state["i"] += 1
        return out[:n]

    return rand


def _salts():
    state = {"i": 0}

    def salt() -> bytes:
        state["i"] += 1
        return hashlib.shake_256(b"salt" + state["i"].to_bytes(8, "little")).digest(192)

    return salt


def _hdr(op, kind, n):
    return bytes([op, kind]) + bytes(6) + struct.pack("<Q", n)


def _pad(b):
    return b + bytes(-len(b) % 16)


def _domain(d):
    return _hdr(RL.OP_DOMAIN, 0, len(d)) + _pad(d)


def _label(b):
    return _hdr(RL.OP_LABEL, 0, len(b)) + _pad(b)


def _bytes(b):
    return _hdr(RL.OP_BYTES, 0, len(b)) + _pad(b)


def _obs(vs, kind=RL.KIND_SLICE):
    return _hdr(RL.OP_OBSERVE, kind, len(vs)) + b"".join(RL.f128_bytes(v) for v in vs)


def _sq(n, kind=RL.KIND_SLICE):
    return _hdr(RL.OP_SQUEEZE, kind, n)


DIGEST = bytes(range(64))
CAP0 = hashlib.sha512(b"cap0").digest() + hashlib.sha512(b"cap0'").digest()
CAP1 = hashlib.sha512(b"cap1").digest()


def _binding_round():
    return _domain(b"flock-circuit/fast100/rep0") + _label(b"flock-r1cs-v0") + _bytes(DIGEST) + _bytes(CAP0) + \
        _label(b"flock-zerocheck-v0") + _sq(6)


def test_requests_round_trip_through_the_codec():
    cases = [(RL.REQ_HELLO, '{"a":1}'), (RL.REQ_LINK, b"\x01\x02"), (RL.REQ_OPEN, "circuit/rep0", "", b"dom", 0),
             (RL.REQ_ROUND, "circuit/rep0", b"xyz", 3), (RL.REQ_PROOF, "circuit", 1, b"proof"), (RL.REQ_FINISH,),
             (RL.REQ_REGISTER,), (RL.REQ_PING, bytes(64)), (RL.REQ_COMMIT, b"rf", [("circuit", b"cap")], [])]
    for c in cases:
        assert RL.decode_req(RL.encode_req(*c)) == c
    assert RL.decode_resp(RL.resp_coins([1, (5 << 64) | 7])) == (RL.RESP_COINS, [1, (5 << 64) | 7])


def test_a_reps_first_round_drops_the_statement_digest_and_publishes_the_cap():
    s = RL.split_round(_binding_round(), first_of_rep=True)
    assert s.caps == [CAP0] and s.digest == DIGEST and s.free == [] and s.n == 6


def test_a_round_sorts_caps_nonces_and_the_free_message():
    vs = [3, (1 << 127) | 9, 12345678901234567890]
    content = _obs(vs[:2]) + _obs([vs[2]], RL.KIND_SCALAR) + _bytes(struct.pack("<Q", 77)) + _bytes(CAP1) + _sq(1, RL.KIND_SCALAR)
    s = RL.split_round(content, first_of_rep=False)
    assert s.free == vs and s.caps == [CAP1] and s.nonces == [struct.pack("<Q", 77)] and s.n == 1 and s.digest is None


def test_a_round_that_is_not_a_round_is_refused():
    with pytest.raises(ValueError, match="without its squeeze"):
        RL.split_round(_obs([1]), first_of_rep=False)
    with pytest.raises(ValueError, match="binds no statement"):
        RL.split_round(_obs([1]) + _sq(1), first_of_rep=True)
    with pytest.raises(ValueError, match="neither a cap nor a nonce"):
        RL.split_round(_bytes(b"12345") + _sq(1), first_of_rep=False)


def test_message_rows_hold_eight_elements_each_and_commit_as_registered_rows():
    vs = [(i << 64) | (i * 3 + 1) for i in range(11)]
    rows = RL.message_rows(vs)
    assert len(rows) == 2 and all(len(r) == RL.ROW_WORDS for r in rows)
    assert RL.row_elements(rows[0]) + RL.row_elements(rows[1])[:3] == vs and RL.row_elements(rows[1])[3:] == [0] * 5
    assert RL.message_rows([]) == []
    salts = [bytes([i]) * 192 for i in range(1, 3)]
    reg = G.Registered.commit("00" * 64, "msgs", rows, salts)
    assert RL.commit_rows(rows, salts) == list(reg.bcs)


def _session(proxy):
    m = 4
    hello = json.dumps({"profile": "fast100", "reps": 2, "flavor": "rs", "tables": ["circuit"],
                        "link": {"sigma": "ab" * 64, "m": m, "points": 2}, "rounds": "bytes retained", "digests": "sha512"},
                       separators=(",", ":"))
    out = [proxy.handle(RL.encode_req(RL.REQ_REGISTER)), proxy.handle(RL.encode_req(RL.REQ_HELLO, hello)),
           proxy.handle(RL.encode_req(RL.REQ_COMMIT, b"sigma-root-f", [("circuit", CAP0)], [])),
           proxy.handle(RL.encode_req(RL.REQ_LINK, bytes(32))),
           proxy.handle(RL.encode_req(RL.REQ_OPEN, "circuit/rep0", "", b"flock-circuit/fast100/rep0", 0)),
           proxy.handle(RL.encode_req(RL.REQ_ROUND, "circuit/rep0", _binding_round(), 6)),
           proxy.handle(RL.encode_req(RL.REQ_ROUND, "circuit/rep0", _obs([5, 6, 7]) + _sq(1, RL.KIND_SCALAR), 1)),
           proxy.handle(RL.encode_req(RL.REQ_PROOF, "circuit", 0, b"the proof")),
           proxy.handle(RL.encode_req(RL.REQ_FINISH))]
    return [RL.decode_resp(b) for b in out]


def test_the_coin_server_sees_only_public_data_and_the_record_keeps_the_session(tmp_path):
    coins = RL.CoinServer(_counter_rand())
    proxy = RL.Proxy(coins, _salts())
    resp = _session(proxy)
    assert resp[0] == (RL.RESP_DRAW, "null") and resp[1] == (RL.RESP_OK,)
    assert resp[2][0] == RL.RESP_COINS and len(resp[2][1]) == 8
    assert resp[5][0] == RL.RESP_COINS and len(resp[5][1]) == 6 and resp[6][0] == RL.RESP_COINS and len(resp[6][1]) == 1
    assert resp[-1][0] == RL.RESP_VERDICT
    pub = coins.record()
    text = json.dumps(pub)
    for secret in ("ab" * 64, DIGEST.hex(), b"sigma-root-f".hex()):
        assert secret not in text
    assert pub["link"]["roots"] == {"circuit": CAP0.hex()}
    assert [r["caps"] for r in pub["rounds"]] == [[CAP0.hex()], []]
    assert [len(r["commits"]) for r in pub["rounds"]] == [0, 1] and pub["message_rows"] == 1
    priv = proxy.private[1]
    assert RL.commit_rows(priv["rows"], [bytes.fromhex(s) for s in priv["salts"]]) == [bytes.fromhex(c) for c in pub["rounds"][1]["commits"]]
    assert RL.row_elements(priv["rows"][0])[:3] == [5, 6, 7]
    rec = proxy.session_record()
    st = rec["streams"][0]
    assert st["root_sha512"] == hashlib.sha512(CAP0).hexdigest()
    assert [r["coins"] for r in st["rounds"]] == [r["coins"] for r in pub["rounds"]]
    assert bytes.fromhex(st["rounds"][1]["msg"]) == _obs([5, 6, 7]) + _sq(1, RL.KIND_SCALAR)
    pts = b"".join(RL.f128_bytes(int(c, 16)) for p in rec["link"]["points"] for c in p)
    assert rec["link_sha512"] == hashlib.sha512(b"sigma-root-f" + CAP0 + pts + bytes(32)).hexdigest()
    assert rec["proof_sha512"] == [{"stream": "circuit/rep0", "bytes": 9, "sha512": hashlib.sha512(b"the proof").hexdigest()}]
    sess = proxy.write(tmp_path)
    assert (sess / "circuit.rep0.bin").read_bytes() == b"the proof"
    assert json.loads((tmp_path / "public.json").read_text())["kind"] == RL.KIND


def test_serve_writes_each_of_several_sessions_to_its_own_directory(tmp_path):
    import socket
    import threading

    class Wire:
        """`_session`'s proxy, as a client of the served proxy over one connection."""

        def __init__(self, addr):
            host, port = addr.rsplit(":", 1)
            self.sock = socket.create_connection((host, int(port)))
            self.rf, self.wf = self.sock.makefile("rb"), self.sock.makefile("wb")

        def handle(self, req):
            RL.write_frame(self.wf, req)
            return RL.read_frame(self.rf)

    bound, got = threading.Event(), {}
    t = threading.Thread(target=lambda: got.update(res=RL.serve("127.0.0.1:0", tmp_path, _counter_rand(), sessions=2,
                                                                    ready=lambda a: (got.update(addr=a), bound.set()))))
    t.start()
    bound.wait()
    for _ in range(2):
        w = Wire(got["addr"])
        assert _session(w)[-1][0] == RL.RESP_VERDICT
        w.sock.close()
    t.join()
    assert [r["session"] for r in got["res"]] == [str(tmp_path / f"s{k}" / "session") for k in range(2)]
    for k in range(2):
        assert (tmp_path / f"s{k}" / "session" / "circuit.rep0.bin").read_bytes() == b"the proof"
        assert json.loads((tmp_path / f"s{k}" / "public.json").read_text())["message_rows"] == 1
    assert not (tmp_path / "session").exists()


def test_coins_follow_the_link_exchange():
    proxy = RL.Proxy(RL.CoinServer(_counter_rand()), _salts())
    hello = json.dumps({"profile": "fast100", "reps": 2, "flavor": "rs", "tables": ["circuit"], "link": {"sigma": "00", "m": 2, "points": 2}})
    proxy.handle(RL.encode_req(RL.REQ_HELLO, hello))
    proxy.handle(RL.encode_req(RL.REQ_OPEN, "circuit/rep0", "", b"d", 0))
    got = RL.decode_resp(proxy.handle(RL.encode_req(RL.REQ_ROUND, "circuit/rep0", _binding_round(), 6)))
    assert got[0] == RL.RESP_REFUSED and "before the link exchange" in got[1]
