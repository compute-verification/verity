"""compile_slice_zk proves w212 at its run's engine class, scheduler included; the slice it proves (CompileSlice_v7)
charges each certified arrival's length term as code_v1 sends it; and its scan of what the verifier received judges
every found witness piece and carry row by the lane's one rule (`leak_scan.scan`), with no byte-count exemption, each
row shorter than the scan's 16-byte piece found whole. Nothing here lowers or proves (`compile_slice_zk.prove` does,
on a node, under ``--zk`` or ``--plain``)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parents[1]
#: `a8.find`'s answer, by substring: each piece (named by its needles) found in `data`
A8 = SimpleNamespace(find=lambda data, table: {v: 1 for k, v in table.items() if k in data})


def _load():
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location("network_traces_compile_slice_zk", HERE / "compile_slice_zk.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_w212_is_proved_at_its_runs_engine_class(monkeypatch):
    """8eb1's engine class is its trace header's, AsyncScheduler (AS = 1) included, and w212 is loaded at it. A fixture
    at another scheduler, the sync class's (AS = 0), is refused, so every window of the run is proved at its one AS."""
    Z = _load()
    x = Z.load("w212")
    assert Z.engine_class()["async_scheduling"] is True
    assert x.fx["engine"] == Z.engine_class() and x.st["AS"] == 1 and x.sl.steps == 21
    monkeypatch.setattr(Z, "W212", Z.FIXTURES / "compile_slice_w212_class.json")
    with pytest.raises(SystemExit, match="isn't the run's engine class"):
        Z.load("w212")


def _term_widths(max_tokens) -> list[int]:
    """code_v1's widths for these requests' length terms (its `lengths.widths`): each output length in
    width(max_tokens + 1) bits, then its finish in 1."""
    from verity_vllm.pipeline.advice_code.arith import width

    return [n for m in max_tokens for n in (width(m + 1), 1)]


def _length_widths(max_tokens) -> int:
    return sum(_term_widths(max_tokens))


def test_the_proved_slices_charge_covers_its_length_terms():
    """The demo's honest unit charges its decisions as CompileSlice_v1 does from the session's first carry, plus each
    certified arrival's length term at code_v1's widths, max_tokens 8 (its request's output length, a power of two),
    16 and 255 among them. ``tails``, declaring the decisions' charge alone, is refused by its charge check.
    ``max-tokens``, request 2's max_tokens lowered from 255 to its output length 3, passes every check at 6 bits less,
    with a ``cert`` row other than the honest one, so the verifier's cert binding alone refuses it (`prove` holds
    ``cert_bound`` to `CERT`)."""
    Z = _load()
    x = Z.load("demo")
    x.carry = Z.chain(x, "compile-slice-zk-test", "00" * 64).carry
    honest, tails, low = Z.cases(x, 3, ("honest", "tails", "max-tokens"))
    v1 = Z.CS.case(x.S, x.sl, Z.BOOK, served=x.rows).length
    assert (x.sl.max_tokens, _length_widths(x.sl.max_tokens), Z.tails(x.sl)) == ((8, 16, 255), 20, 20)
    assert (honest.charge, tails.charge, low.charge) == (v1 + 20, v1, v1 + 14)
    assert all(honest.case.flags.values()) and all(low.case.flags.values())
    assert [f for f, ok in tails.case.flags.items() if not ok] == ["charge"]
    assert low.words["cert"] != honest.words["cert"] and tails.words["cert"] == honest.words["cert"]
    assert "max-tokens" in Z.CERT and "tails" not in Z.CERT
    with pytest.raises(ValueError, match="stops at its cap"):
        Z.lowered(Z.V7.Slice(((0, 9, 8),), (), 1, (), (8,)))


def test_w212s_unit_charges_code_v1s_window_with_its_length_tails():
    """w212's unit, from the chain's carry over the run's windows 0 … 211, charges window 212 what code_v1 does with its
    requests' length terms as tails (`window_code.encode`'s ``lens`` on the run's windows, each with its requests'
    output lengths and finishes at code_v1's widths), which is the chain's (`charged`); each of its three requests
    stops at its max_tokens, so no ``max-tokens`` kind is offered there."""
    from verity.protocols.certifier.prover.window_charge import Advice
    from verity_vllm.pipeline.advice_code import window_code

    Z = _load()
    x = Z.load("w212")
    run = Z.run_windows()
    tails = []
    for _d, rq in run:
        values = [v for r in rq for v in (r.output_len, int(r.output_len >= r.max_tokens))]
        tails.append(list(zip(values, _term_widths([r.max_tokens for r in rq]))))
    lens = window_code.encode([Advice(arrived=d) for d, _rq in run], window_code.Book.from_json(Z.BOOK), tails).lens
    ch = Z.chain(x, "compile-slice-zk-test", "00" * 64)
    x.carry = ch.carry
    honest, = Z.cases(x, 1, ("honest",))
    assert (honest.charge, ch.window, Z.charged(x, ch, [honest])["bits"]) == (lens[Z.WINDOW],) * 3
    assert Z.tails(x.sl) == _length_widths(x.sl.max_tokens) == 14
    assert [a[2] for a in x.sl.arrivals] == list(x.sl.max_tokens) and "max-tokens" not in Z.REFUSED["w212"]
    with pytest.raises(ValueError, match="stops at its cap"):
        Z.lowered(x.sl)


def test_a_piece_with_one_nonzero_byte_is_judged_by_the_rule(tmp_path):
    """No piece is left out for its byte count: a witness piece with one nonzero byte found in a file whose format isn't
    read is leaked, and one whose bytes the public rows hold is structured."""
    Z = _load()
    low, pub = b"\x05" + bytes(15), bytes(15) + b"\x80"
    vdir = tmp_path / "verifier"
    vdir.mkdir()
    (vdir / "openings.json").write_bytes(b"[" + low + b"]")
    (vdir / "pub-2.bin").write_bytes(b"\x01" + pub)
    (tmp_path / "inst.bin").write_bytes(low + pub)
    got = Z.received(A8, vdir, tmp_path / "inst.bin", {"wit 0 @0": low, "wit 8 @0": pub}, {"wit 0 @0", "wit 8 @0"})
    assert got["verifier_found"] == ["wit 0 @0", "wit 8 @0"]
    assert got["leaked"] == ["wit 0 @0"] and got["structured"] == {"wit 8 @0": "public"}
    assert got["control"]["holds_all"]


def test_the_carry_rows_are_each_scanned_whole_under_the_rule(tmp_path):
    """Each unit's carry rows (`window_charge`'s, 20 bytes) are scanned whole, the public first carry's ``cin`` not
    among them. A restart's zero ``cin`` row is structured where the public rows hold its bytes, and leaked where they
    don't; a carry row found off them is leaked; the control holds every row."""
    Z = _load()
    assert Z.WC.ROW_BITS == 160
    zero, live, sent = [0] * 10, [58, 50, 58, 0, 392, 48, 0, 0, 0, 0], list(range(1, 11))
    units = [SimpleNamespace(words={"cin": zero, "cout": live}, chained=False),
             SimpleNamespace(words={"cin": live, "cout": sent}, chained=True)]
    assert list(Z.carry_rows(units, SimpleNamespace(public=True))) == ["cout 0", "cin 0", "cout 1"]
    carry = Z.carry_rows(units, SimpleNamespace(public=False))
    assert carry == {"cout 0": Z._u16(live), "cin 0": bytes(20), "cout 1": Z._u16(sent), "cin 1": Z._u16(live)}
    vdir = tmp_path / "verifier"
    vdir.mkdir()
    (vdir / "openings.json").write_bytes(bytes(64) + Z._u16(sent))
    (vdir / "pub-2.bin").write_bytes(b"\x07" + bytes(28))
    (tmp_path / "inst.bin").write_bytes(b"".join(carry.values()))
    got = Z.received(A8, vdir, tmp_path / "inst.bin", carry, set(carry))
    assert got["verifier_found"] == ["cin 0", "cout 1"]
    assert got["structured"] == {"cin 0": "public"} and got["leaked"] == ["cout 1"] and got["control"]["holds_all"]

    (vdir / "pub-2.bin").write_bytes(bytes(range(1, 30)))
    assert Z.received(A8, vdir, tmp_path / "inst.bin", carry, set(carry))["leaked"] == ["cin 0", "cout 1"]


def test_the_rows_shorter_than_a_piece_are_found_whole_under_the_rule(tmp_path):
    """`scan_pieces` cuts each row of at least 16 bytes whole (`leak_scan.public_pieces`) and keeps each shorter row
    whole: the ``srv`` rows of a few steps (10 bytes at the demo's G = 2, 6 at G = 1), which `leak_scan.cut` refuses
    and the aligned cutter before it never scanned. Such a row is found raw or as hex, a raw one under 8 bytes too
    (`a8.find` keys each needle on its first 8 bytes, so `_find` searches those itself), and judged by the rule: leaked
    off the statement's bytes, structured once a ``pub`` row holds them. The instance file lays the rows back to back
    (`circuit.write`), so the control holds a row's last 16 bytes and not its zero-padded last chunk, which the scan
    still looks for."""
    Z = _load()
    a8 = SimpleNamespace(PIECE=16,
                         needles=lambda pieces: {nd: k for k, x in pieces.items() for nd in (x, x.hex().encode())},
                         find=lambda data, table: {k: 1 for nd, k in table.items() if len(nd) >= 8 and nd in data})
    wit, srv0, srv1 = list(range(1, 19)), [7, 9, 11, 13, 15], [21, 23, 25]
    units = [SimpleNamespace(words={"wit": wit, "srv0": srv0, "srv1": srv1}, chained=False)]
    with pytest.raises(ValueError, match="shorter than the scan's 16-byte piece"):
        Z.LS.public_pieces(a8, {"srv0 0": Z._u16(srv0)}, set(), b"")
    pieces, held, public = Z.scan_pieces(a8, units, {}, SimpleNamespace(public=False), ("srv0", "srv1"), b"")
    assert pieces == {"wit 0 @0": Z._u16(wit)[:16], "wit 0 @16": Z._u16(wit)[16:32], "wit 0 @20": Z._u16(wit)[20:],
                      "wit 0 @32": Z._u16(wit)[32:].ljust(16, b"\0"), "srv0 0": Z._u16(srv0), "srv1 0": Z._u16(srv1)}
    assert held == set(pieces) - {"wit 0 @32"} and public == []
    vdir = tmp_path / "verifier"
    vdir.mkdir()
    (vdir / "openings.json").write_bytes(Z._u16(srv0).hex().encode() + b"," + Z._u16(srv1) + b"\x01\x02")
    (tmp_path / "inst.bin").write_bytes(Z._u16(wit) + Z._u16(srv0) + Z._u16(srv1))
    got = Z.received(a8, vdir, tmp_path / "inst.bin", pieces, held)
    assert got["verifier_found"] == ["srv0 0", "srv1 0"] and got["leaked"] == ["srv0 0", "srv1 0"]
    assert got["control"]["holds_all"] and got["control"]["pieces_found"] == 5
    (vdir / "pub-2.bin").write_bytes(Z._u16(srv1) + bytes(4))
    got = Z.received(a8, vdir, tmp_path / "inst.bin", pieces, held)
    assert got["leaked"] == ["srv0 0"] and got["structured"] == {"srv1 0": "public"}
