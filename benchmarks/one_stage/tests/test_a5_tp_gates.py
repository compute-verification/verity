"""a5_tp_gates' two ranks' windows, before anything is lowered or proved (`a5_tp_gates.main` does, through `a5.audit`, on a
node): the transfer-gate unit is the registered AllReduce2_v1 with a port a rank, its sums bit for bit; each rank's chunk
c<k> reads both ranks' registered partials of request q's collective k at the token rows, under their owners' salts, and
they open the roots each
owner's registration names at the positions the verifier's own reads name; the registration passes the verifier's check with every
transfer-gate unit proved; each rank's provenance is the verifier's read of both registrations; the negatives (a peer's
partial under another root, the rank's own as its peer's, another request's as its peer's, a partial under another TP
run's domain, the other rank's registration, ports built before the peer registered, and each way of moving a bit through
the received partial) are refused by name; and each refused run's window changes only rank 1's received port of q0c0."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from verity.ml.circuits import evaluation as E
from verity.ml.circuits.defs import bind
from verity.core.protocols.certifier import warden_rows as WR
from verity.core.protocols.sampled_proofs.one_stage import draw as D, verifier as V
from verity.ml.definitions.registry.b1_tp2 import AllReduce2
from verity.ml.windows import rows as SR

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import a5_tp_gates as AT  # noqa: E402

H, T, Q, REST = 8, 2, 2, 4
K = len(AT.COLLECTIVES)


@pytest.fixture(scope="module")
def parts():
    return AT.partials(H, T, Q, seed=3)


def committed(tmp: Path, parts: dict, altered: bool = False, made: str | None = None):
    mine = {r: AT.window(r, H, T, Q, REST) for r in AT.RANKS}
    c = {r: AT.commit(tmp / f"rank{r}", mine[r], parts, r, altered and r == AT.RECEIVER, made if r == AT.RECEIVER else None)
         for r in AT.RANKS}
    wins = {r: json.loads((tmp / f"rank{r}" / "window.json").read_text()) for r in AT.RANKS}
    return mine, c, AT.registrations(tmp), wins


def table(c, m, port: str) -> list[bytes]:
    b = c.bc[m.port(port)]
    return [b[i * AT.CS.BC_BYTES:(i + 1) * AT.CS.BC_BYTES] for i in range(len(b) // AT.CS.BC_BYTES)]


def header(win: dict, m) -> dict:
    return SR.registered_header(win, next(e["k"] for e in win["members"] if e["name"] == m.key))


def test_the_unit_is_the_registered_allreduce2_with_a_port_a_rank() -> None:
    """The transfer-gate template is the registered `AllReduce2_v1{N}`, its ports ``part0`` and ``part1`` its input's rows
    ``parts[0]`` and ``parts[1]``; it is `sums` of ``[part0, part1]`` on every word pair drawn, the special words among them
    (zeros of both signs, infinities, NaNs, subnormals, the largest finite), and on the activations' rows `sums` is
    ``bf16_rn(f32(a) + f32(b))``."""
    unit = AT.templates(H)[0]
    assert unit == bind(AllReduce2, N=H) and unit.id == f"AllReduce2_v1{{N={H}}}"
    assert AT.PORT_PARAMS == {"part0": "parts[0]", "part1": "parts[1]"}
    rng = np.random.default_rng(7)
    special = np.array([0x0000, 0x8000, 0x7F80, 0xFF80, 0x7FC0, 0xFFC1, 0x0001, 0x807F, 0x7F7F, 0xFF7F, 0x3F80, 0xBF80], np.uint16)
    a = rng.integers(0, 1 << 16, (400, H)).astype(np.uint16)
    b = rng.integers(0, 1 << 16, (400, H)).astype(np.uint16)
    a[:len(special)], b[:len(special)] = special[:, None], np.roll(special, 5)[:, None]
    a[len(special):len(special) + 40] = AT.bf16(rng.normal(0, 0.05, (40, H)))
    b[len(special):len(special) + 40] = AT.bf16(rng.normal(0, 0.05, (40, H)))
    got = np.asarray(E.evaluate_batch(unit, np.stack([a, b], 1))).reshape(a.shape)
    assert np.array_equal(got, AT.sums(a, b)) and np.array_equal(AT.sums(a, b), AT.sums(b, a))
    act = slice(len(special), len(special) + 40)
    f32 = lambda w: (w.astype(np.uint32) << 16).view(np.float32)  # noqa: E731
    assert np.array_equal(got[act], AT.bf16(f32(a[act]) + f32(b[act])))


def test_bf16_rounds_to_nearest_even() -> None:
    x = np.array([1.0, 1.0 + 2 ** -8, 1.0 + 3 * 2 ** -8, -2.5, 1.0 + 2 ** -8 + 2 ** -20], np.float32)
    assert AT.bf16(x).tolist() == [0x3F80, 0x3F80, 0x3F82, 0xC020, 0x3F81]


def test_each_chunk_opens_both_ranks_registered_partials_at_the_token_rows(tmp_path, parts) -> None:
    mine, c, regs, wins = committed(tmp_path, parts)
    for r in AT.RANKS:
        assert regs[r]["window"]["of"] == AT.provenance(parts, r) and regs[r]["window"]["run"] == f"{AT.ROOT}/rank{r}"
        files = AT.reads(mine[r], r, regs)
        assert [(q, k) for q, k, _m in AT.chunks(mine[r])] == [(q, k) for q in range(Q) for k in range(K)]
        for u, (q, k, m) in enumerate(AT.chunks(mine[r])):
            assert (m.n, m.base, dict(m.tables), m.name) == (T, u * T, {"part0": T, "part1": T}, f"{m.template}/q{q}c{k}")
            for j in AT.RANKS:
                tree, p = parts[j][q][k], f"part{j}"
                port, entry = files[m.key]["ports"][p], header(wins[r], m)[p]
                assert port["root"] == regs[j]["window"]["of"]["partials"][AT.value_name(j, q, k)]["root"] == tree.root.hex()
                assert port["value"] == entry["value"] == AT.value_name(j, q, k) and port["reads"] == entry["positions"] == list(range(T))
                assert port["domain"] == tree.domain.domain_id.hex() and port.get("reads_from", 0) == m.base
                assert table(c[r], m, p) == list(tree.bcs) == [AT.bc(x, s) for x, s in zip(tree.rows, tree.salts)]
                assert c[r].salts[m.port(p)] == b"".join(tree.salts)
                assert WR.refusal(port, AT.domain(j, q, k, H, T), entry, table(c[r], m, p)) is None
            want = AT.sums(np.asarray(parts[0][q][k].rows), np.asarray(parts[1][q][k].rows))
            assert np.array_equal(c[r].rows[m.port("o0")], want)
        rest = mine[r].members[-1]
        assert "registered" not in wins[r]["members"][-1]
        assert not set(table(c[r], rest, "row")) & {b for t in parts.values() for req in t for v in req for b in v.bcs}
    assert all(np.array_equal(c[0].rows[m0.port("o0")], c[1].rows[m1.port("o0")])
               for (_q0, _k0, m0), (_q1, _k1, m1) in zip(AT.chunks(mine[0]), AT.chunks(mine[1])))
    assert len({b for t in parts.values() for req in t for v in req for b in v.bcs}) == len(AT.RANKS) * Q * K * T


def test_the_registration_proves_every_transfer_gate_unit_beside_the_drawn_rest(tmp_path, parts) -> None:
    mine, c, regs, _wins = committed(tmp_path, parts)
    for r in AT.RANKS:
        tg, rid = mine[r].members[0].template, mine[r].members[-1].template
        assert tg != rid and mine[r].work_table == {tg: {"work": 0, "floor": Q * K * T}, rid: {"work": 1, "floor": 1}}
        publics = {m.key: c[r].public_body(k) for k, m in enumerate(mine[r].members)}
        v = V.check(regs[r], mine[r].program, mine[r].query, mine[r].law, AT.CS.SCHEME, AT.CS.expected_ports(mine[r]),
                    leaf_layer=AT.CS.leaf_layer(publics), work_table=mine[r].work_table)
        assert v.ok, v.codes
        strata = {s["template"]: s for s in D.derive(mine[r].program, mine[r].query, mine[r].law, mine[r].work_table)["strata"]}
        assert strata[tg]["units"] == [[0, Q * K * T]] and strata[tg]["k"] == Q * K * T
    assert regs[0]["roots"] != regs[1]["roots"]


def test_the_provenance_is_the_verifiers_read_and_the_negatives_are_refused_by_name(tmp_path, parts) -> None:
    mine, _c, regs, _wins = committed(tmp_path, parts)
    for r in AT.RANKS:
        source = AT.source_of(mine[r], r, regs)
        assert source(regs[r]["window"])["ok"]
        assert source({**regs[r]["window"], "run": "00" * 64})["codes"] == ["run"]
        neg = AT.negatives(mine[r], parts, r, regs)
        assert all(x["refused"] for x in neg.values()), neg
        assert len(neg) == 6 + len(AT.TRANSFERS)
        assert neg["registration-naming-the-peers-partial-under-another-root (source)"]["codes"] == ["peers"]
        assert neg["registration-naming-its-own-partial-as-the-peers (source)"]["codes"] == ["peers"]
        assert neg["registration-naming-another-requests-partial-as-the-peers (source)"]["codes"] == ["peers"]
        assert neg["partial-under-another-tp-runs-domain (source)"]["codes"] == ["partials"]
        assert neg["registration-of-the-other-rank (source)"]["codes"] == ["run", "step", "partials", "peers"]
        assert "ranks [" + str(1 - r) + "]" in neg["ports-before-the-peers-registration (reads)"]["why"]
        sent = AT.value_name(1 - r, 0, 0)
        assert "table row 0 does not open" in neg["received-partial-altered (registered reads)"]["why"]
        assert "unit 0 reads position 1, and the verifier's program reads 0" in neg["received-partial-swapped (registered reads)"]["why"]
        for kind, value in (("other-collective", AT.value_name(1 - r, 0, 1)), ("other-request", AT.value_name(1 - r, 1, 0)),
                            ("own", AT.value_name(r, 0, 0))):
            x = neg[f"received-partial-{kind} (registered reads)"]
            assert x["value"] == value and f"reads {sent}, not another value ({value})" in x["why"]


def test_ports_need_every_ranks_registration(tmp_path, parts) -> None:
    mine, _c, regs, _wins = committed(tmp_path, parts)
    with pytest.raises(ValueError, match="before any rank's coin"):
        AT.reads(mine[0], 0, {0: regs[0]})
    with pytest.raises(ValueError, match=r"ranks \[0, 1\]"):
        AT.reads(mine[0], 0, {})


@pytest.mark.parametrize("kind", sorted(AT.TRANSFERS))
def test_a_refused_runs_window_moves_the_bit_through_rank_1s_received_partial_only(tmp_path, parts, kind) -> None:
    """``--altered`` / ``--made-for``'s windows: rank 1's q0c0 ``part0`` (rank 0's embedding partial of request 0, as rank 1
    received it) made for that read, its sums the rows it read (the circuit holds), the registration's checks passing; only the verifier's
    port refuses it, by `TRANSFERS`' reason, and every other port of both ranks opens."""
    mine, c, regs, wins = committed(tmp_path, parts, altered=kind == "altered", made=None if kind == "altered" else kind)
    for r in AT.RANKS:
        assert AT.source_of(mine[r], r, regs)(regs[r]["window"])["ok"]
        files = AT.reads(mine[r], r, regs)
        for q, k, m in AT.chunks(mine[r]):
            got = {}
            for j in AT.RANKS:
                p = f"part{j}"
                got[p] = WR.refusal(files[m.key]["ports"][p], AT.domain(j, q, k, H, T), header(wins[r], m)[p], table(c[r], m, p))
            rows = [c[r].rows[m.port(p)] for p in ("part0", "part1")]
            assert np.array_equal(c[r].rows[m.port("o0")], AT.sums(*rows))
            if (r, q, k) != (AT.RECEIVER, 0, 0):
                assert got == {"part0": None, "part1": None}
                continue
            assert got["part1"] is None and got["part0"].startswith("registered reads:") and AT.TRANSFERS[kind] in got["part0"]
            sent = parts[0][0][0]
            if kind == "altered":
                assert c[r].salts[m.port("part0")] == b"".join(sent.salts)
                diff = np.flatnonzero(rows[0] != np.asarray(sent.rows, np.uint16))
                assert diff.tolist() == [0] and rows[0][0, 0] ^ sent.rows[0][0] == 1
            else:
                assert header(wins[r], m)["part0"]["value"] == {
                    "swapped": AT.value_name(0, 0, 0), "other-collective": AT.value_name(0, 0, 1),
                    "other-request": AT.value_name(0, 1, 0), "own": AT.value_name(1, 0, 0)}[kind]


def test_a5s_registered_read_negatives_alter_one_of_a_chunks_two_ports_and_keep_the_other(tmp_path, monkeypatch) -> None:
    """A chunk reads both ranks' partials, so each of a5's registered-read negatives alters its first port's read and holds
    the other port's as registered: Lean is to refuse the altered read by its reason, not a port the reads leave out."""
    import a5

    ports = {f"part{j}": {"value": AT.value_name(j, 0, 0), "root": f"{j}" * 128, "leaves": T, "reads": list(range(T))}
             for j in AT.RANKS}
    (tmp_path / "registered.json").write_text(json.dumps({"format": "verity/registered-reads/v1", "ports": ports}))
    seen = {}

    def lean_verify(lean, st, n, sdir, zk, log):
        seen[log] = json.loads(st["registered_file"].read_text())["ports"] if "registered_file" in st else None
        return {"accepted": False, "digest": "d", "why": "setup: registered reads: none"}

    monkeypatch.setattr(a5.a0, "lean_verify", lean_verify)
    a5.registered_negatives(SimpleNamespace(lean="lean"), {"st": {"registered_file": tmp_path / "registered.json"},
                                                            "m": SimpleNamespace(n=T)}, tmp_path)
    assert seen == {"chunk-without-registered-reads": None,
                    "registered-read-against-another-root": {**ports, "part0": {**ports["part0"], "root": "00" * 64}},
                    "registered-reads-another-row": {**ports, "part0": {**ports["part0"], "reads": [*range(1, T), 0]}}}
