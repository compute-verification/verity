"""n2_advised_zk's link-windows on the advised tests' traffic (`advised_traffic`). At the driver's T = 512 the session
is varied, every window receives, its honest windows' output words are verdict 1 and deadline 1, and each negative (a
moved release bucket, an output length one short, extra egress) fails in some window on the advised counts alone, its
head checks passing and every window's unit taking an instance; the verifier's own ``pub`` row, from public fields,
the certificates and its own view of the public advice, is the prover's, in whole u16 words. The --e2e link's advice
(`joined`) numbers each Commit window's requests after the windows' before it. At the tests' T = 8 the
proof block (`LinkWindows.proof_unit`), evaluated on the rows the prover stages, gives the output words the gates'
references do (`LinkWindows.outputs`) in every window of every member and on a head the prover changed. ``--jobs`` is 1
unless given and 2 is refused, naming the rule (`n2_zk.ONE_SESSION`). Nothing here lowers or proves
(`n2_advised_zk.prove` does, on a node)."""

from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from dataclasses import replace
from functools import cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from verity.ml.circuits.evaluation.bits import evaluate_bits
from verity.core.protocols.certifier import advice as ADV, head_check as HC

HERE = Path(__file__).resolve().parents[1]


def _load():
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location("network_traces_n2_advised_zk", HERE / "n2_advised_zk.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


n2 = _load()
AT = n2.AT


@cache
def _link(T: int):
    shape = n2.SHAPE if T == n2.T else AT.SHAPE
    sim = n2.simulated(n2.SEED, shape)
    return sim, {name: n2.link_windows(sim.lw, link, pairs) for name, (link, pairs, _) in sim.members.items()}


def test_the_fixture_is_varied_and_every_window_receives():
    sim, _ = _link(n2.T)
    s, a = sim.session, sim.session.link.advice
    rows = [r for st in a.steps for r in st.rows]
    assert {AT.FRAMING[q.prompt_len - 1] for q in a.requests} == {1, 2, 3} and len({q.output_len for q in a.requests}) > 3
    assert any(r.kind == "prefill" and r.start + r.n < a.requests[r.request].prompt_len for r in rows)
    assert any(r.cache_hit and not r.preempted for r in rows) and any(r.preempted and r.cache_hit for r in rows)
    assert any([r.request for r in st.rows] != sorted(r.request for r in st.rows) for st in a.steps)
    assert any(b // n2.T > h // n2.T for h, b in zip(s.hand, a.release))
    assert max(len({b // n2.T for o, b in zip(ADV.outputs(a), a.release) if o.request == j})
               for j in range(len(a.requests))) > 1
    assert all(s.link.received) and len(s.link.received) == n2.WINDOWS
    assert sim.lw.unit.id == "NetworkAdvisedLinkWindow_v2{EGRESS=(9,512,0),INGRESS=(9,512,0),ADVISED=(512,4,1)}"
    assert sim.lw.proof_unit.id == sim.lw.unit.id and sim.lw.advised_key() == (512, 4, 1)
    assert all(HC.pinned(g) for g in sim.lw.gates())


def test_honest_passes_and_each_negative_fails_on_the_advised_counts_alone():
    _, rounds = _link(n2.T)
    assert list(rounds) == list(n2.MEMBERS)
    for name, rd in rounds.items():
        fails = [n2.failed(x) for x in rd.windows]
        assert n2.expected(name, fails), (name, fails)
        assert [x.expect for x in rd.windows] == [int(not f) for f in fails], name
    assert all(x.word == 0b11 for x in rounds["honest"].windows)
    assert not n2.expected("moved", [[]] * n2.WINDOWS)
    assert not n2.expected("extra", [["egress/signed", "advised"]] + [[]] * (n2.WINDOWS - 1))


def test_a_head_the_prover_changed_fails_its_signed_digest_in_window_0_alone():
    """The served link's ``tampered`` (`n2_zk.one_count` on the honest records, its certificate the original's), here
    on the simulated link: window 0's egress head fails its signed digest, the advised counts maybe the count it
    changed, and no other window fails; `expected` holds tampered to that and no other member to it. The tie the
    summary names is the stand-in, since the link's ingress isn't IoTieRequest's pinned size class."""
    sim, _ = _link(n2.T)
    lw = sim.lw
    link, pairs, _ = sim.members["honest"]
    rd = n2.link_windows(lw, link, [n2.Z.one_count(lw, w, p) for w, p in enumerate(pairs)])
    fails = [n2.failed(x) for x in rd.windows]
    assert n2.expected("tampered", fails) and "egress/signed" in fails[0], fails
    assert [x.expect for x in rd.windows] == [0] + [1] * (n2.WINDOWS - 1)
    assert not n2.expected("tampered", [[]] * n2.WINDOWS)
    assert not n2.expected("tampered", [["advised"]] + [[]] * (n2.WINDOWS - 1))
    assert not n2.expected("tampered", [["egress/signed"]] * n2.WINDOWS)
    assert not n2.expected("moved", fails) and not n2.expected("honest", fails)
    assert sim.tie["source"] == "stand-in" and tuple(sim.tie["ingress"]) != n2.TIE_PINNED


def _proxy_log(run, syncs=AT.SYNCS):
    with gzip.open(run / "proxy.json.gz", "wt") as fh:
        json.dump({"syncs": [[s.delta, str(s.rho)] for s in syncs]}, fh)


def test_a_served_run_without_clocks_jsonl_is_refused_unless_the_stand_in_is_asked_for_and_then_labelled_one(tmp_path):
    """`served_clocks` fails closed: a run without ``clocks.jsonl`` is refused even beside its certifier's proxy log,
    and only ``--clock-stand-in`` (`stand_in`) takes `CertifierClocks`, the syncs that log holds for its link alone,
    which the summary labels a stand-in; with neither file the flag is refused too. On the simulated link at T = 8
    every window has its view under the log; a window whose logged sync isn't the one it declares is refused alone."""
    sim, _ = _link(AT.T)
    lw, (link, pairs, _) = sim.lw, sim.members["honest"]
    egress, ingress = ([list(p[d].head.counts) for p in pairs] for d in (0, 1))
    with pytest.raises(SystemExit, match="no stand-in clock source"):
        n2.served_clocks(tmp_path, lw.link, stand_in=True)
    _proxy_log(tmp_path)
    with pytest.raises(SystemExit, match="no clocks.jsonl.*--clock-stand-in"):
        n2.served_clocks(tmp_path, lw.link)
    logged = replace(lw, certified_clocks=None, **n2.served_clocks(tmp_path, lw.link, stand_in=True))
    assert isinstance(logged.clocks, n2.CertifierClocks) and logged.clocks.syncs("another") == ()
    label = n2.clocks_label(logged, AT.SYNCS)
    assert label["source"] == "stand-in" and "--clock-stand-in" in label["by"] and label["as_declared"]
    assert not n2.clocks_label(logged, AT.SYNCS[::-1])["as_declared"]
    assert None not in n2.advised_refs(logged, link, egress, ingress)
    off = n2.CertifierClocks(lw.link, (replace(AT.SYNCS[0], delta=AT.SYNCS[0].delta + 1), *AT.SYNCS[1:]))
    refs = n2.advised_refs(replace(lw, certified_clocks=None, clocks=off), link, egress, ingress)
    assert refs[0] is None and None not in refs[1:]


def test_a_served_runs_clock_source_is_its_certifiers_signed_syncs_even_beside_a_log(tmp_path):
    """`served_clocks`, the --e2e link's clock source: the run's ``clocks.jsonl`` (the certifier's salted clock
    certificates with their openings, `LinkWindows.certified_clocks`), even beside a proxy log and with the stand-in
    asked for. On the simulated link at T = 8, whose certifier signed every window's sync, every window has its view
    under them, and a window whose opening fails (here window 0's sync, swapped after signing) has no sync, nor has any
    window after it, which the summary's label names without showing a sync. A ``clocks.jsonl`` of unsalted lines, or
    the client's public copy, is no source: the run is refused, the stand-in asked for or not."""
    sim, _ = _link(AT.T)
    lw, (link, pairs, _) = sim.lw, sim.members["honest"]
    egress, ingress = ([list(p[d].head.counts) for p in pairs] for d in (0, 1))
    signed = tuple(lw.certified_clocks)
    assert sim.clocks["source"] == "signed" and sim.clocks["as_declared"] and len(signed) == len(pairs)
    assert sim.clocks["windows"] == len(pairs) and sim.clocks["refused"] == [] and "syncs" not in sim.clocks
    _proxy_log(tmp_path)
    (tmp_path / "clocks.jsonl").write_text("".join(json.dumps(n2.cs.clock_json(c)) + "\n" for c in signed))
    for stand_in in (False, True):
        read = n2.served_clocks(tmp_path, lw.link, stand_in)
        assert list(read) == ["certified_clocks"] and read["certified_clocks"] == signed
    again = replace(lw, certified_clocks=read["certified_clocks"])
    assert n2.clocks_label(again, AT.SYNCS)["source"] == "signed"
    assert None not in n2.advised_refs(again, link, egress, ingress)
    forged = (replace(signed[0], sync=AT.SYNCS[1]), *signed[1:])
    off = replace(lw, certified_clocks=forged)
    assert n2.advised_refs(off, link, egress, ingress) == [None] * len(pairs)
    label = n2.clocks_label(off, AT.SYNCS)
    assert label["windows"] == 0 and not label["as_declared"] and \
        label["refused"] == [f"{AT.LINK} window 0's egress clock (key {lw.key.key_id.hex()[:16]}…): its sync and salt "
                             "don't open the salted digest it signs"]
    for doc in ({"sync": [1, "2"], "certificate": n2.cs.certificate_json(signed[0].certificate)},
                n2.cs.certificate_json(signed[0].certificate)):
        (tmp_path / "clocks.jsonl").write_text(json.dumps(doc) + "\n")
        for stand_in in (False, True):
            with pytest.raises(SystemExit, match="without its opening's salt.*so the run is refused"):
                n2.served_clocks(tmp_path, lw.link, stand_in)


def test_a_clocks_jsonl_that_isnt_a_regular_file_is_the_signed_source_not_a_run_without_one(tmp_path):
    """A run's ``clocks.jsonl`` that is there but isn't a regular file (a dangling symlink, a directory) is the
    certifier's clock certificates, which can't be read, so the run is refused, the stand-in asked for or not; it is
    never taken for a run without them, whose unsigned proxy log beside it could hold δ = 0."""
    _proxy_log(tmp_path, [replace(s, delta=0) for s in AT.SYNCS])
    (tmp_path / "clocks.jsonl").symlink_to(tmp_path / "gone" / "clocks.jsonl")
    for stand_in in (False, True):
        with pytest.raises(SystemExit, match="not a file of clock certificates"):
            n2.served_clocks(tmp_path, AT.LINK, stand_in)
    (tmp_path / "clocks.jsonl").unlink()
    (tmp_path / "clocks.jsonl").mkdir()
    for stand_in in (False, True):
        with pytest.raises(SystemExit, match="not a file of clock certificates"):
            n2.served_clocks(tmp_path, AT.LINK, stand_in)


def test_joined_advice_numbers_each_windows_requests_after_the_windows_before_and_releases_by_id():
    """`joined` (the --e2e link's advice from its Commit windows'): the second window's rows and hit sources name its
    requests after the first's, and each kept output is released where the certified frames of its request say."""
    a = _link(n2.T)[0].session.link.advice
    b = ADV.Advice(tuple(replace(q, id=f"b{q.id}") for q in a.requests), a.steps, a.release)
    frames = {q.id: [1000 * k + 10 * j + i for i in range(q.output_len)]
              for k, x in enumerate((a, b)) for j, q in enumerate(x.requests)}
    got, n = n2.joined([(w, n2.WA.Advice(x.requests, x.steps, x.release)) for w, x in enumerate((a, b))], frames), \
        len(a.requests)
    assert got.requests == a.requests + b.requests and got.steps[:len(a.steps)] == a.steps
    rows = [r for st in got.steps[len(a.steps):] for r in st.rows]
    assert [(r.request, [s.request for s in r.sources]) for r in rows] == \
        [(r.request + n, [s.request + n for s in r.sources]) for st in a.steps for r in st.rows]
    assert any(r.sources for r in rows)
    assert all(b_ == frames[got.requests[o.request].id][o.index] for o, b_ in zip(ADV.outputs(got), got.release))


def test_advice_from_io_is_the_certified_requests_each_kept_output_released_at_its_frame():
    """`from_io` (--advice-from-io, a gate rehearsal's advice): the certified requests in arrival order, and each kept
    output of each released in the bucket its certified frame went out."""
    a = _link(n2.T)[0].session.link.advice
    frames = {q.id: [] for q in a.requests}
    for o, b in zip(ADV.outputs(a), a.release):
        frames[a.requests[o.request].id].append(b)
    io = [{"user": q.id, "prompt": [0] * q.prompt_len, "output": [0] * q.output_len, "finish": q.finish,
           "frames": [[i, b] for i, b in enumerate(frames[q.id])]} for q in a.requests]
    got = n2.from_io(io)
    assert got.requests == a.requests
    assert sorted((o.request, o.index) for o in ADV.outputs(got)) == \
        sorted((r, i) for r, q in enumerate(a.requests) for i in range(q.output_len))
    assert all(b == frames[got.requests[o.request].id][o.index] for o, b in zip(ADV.outputs(got), got.release))


def test_the_verifier_derives_the_provers_public_row():
    """The verifier's own ``pub`` row is the prover's in every window of every member, and holds the window's view of
    the member's advice: moved and short state another view than honest's in some window, extra the same."""
    _, rounds = _link(n2.T)
    for name, rd in rounds.items():
        lw = rd.lw
        for x in rd.windows:
            mine = lw.public_arguments(x.window, x.root, x.certificates, rd.syncs, rd.advice)
            assert lw.row(mine, "pub") == lw.row(x.args, "pub"), (name, x.window)
            assert all(len(lw.row(x.args, r)) * 8 == lw.bits(r) and lw.bits(r) % 16 == 0 for r in n2.SVC.ROWS)
            assert set(n2.SVC.ADVICE) <= set(lw.layout["pub"])

    def views(name):
        return [{n: x.args[n] for n in n2.SVC.ADVICE} for x in rounds[name].windows]

    assert views("moved") != views("honest") and views("short") != views("honest") and views("extra") == views("honest")


def _evaluate(lw, windows) -> list[int]:
    names = [n for n, _ in lw.proof_unit.signature()[0]]
    args = [np.asarray([x.args[n] for x in windows], dtype=np.uint8) for n in names]
    out = np.asarray(evaluate_bits(lw.proof_unit, *args)).reshape(len(windows), -1)
    return [int(r[0]) | int(r[1]) << 1 for r in out]


@pytest.mark.slow
def test_the_proof_unit_on_the_staged_rows_is_the_gates_references():
    """At T = 8: every member's windows, and the honest records with window 0's egress head one count off, committed by
    the prover (its certificate the original's), which the egress head check refuses. About 85 s, nearly all of it
    building the head check's evaluation; it is the one test that evaluates the advised unit as the node lowers it,
    so a head's counts cut at the wrong offset or the ends swapped into the advised counts fail here and pass every
    other test."""
    sim, rounds = _link(AT.T)
    lw = sim.lw
    link, pairs, _ = sim.members["honest"]
    tampered = n2.link_windows(lw, link, [n2.Z.one_count(lw, w, p) for w, p in enumerate(pairs)])
    assert "egress/signed" in n2.failed(tampered.windows[0])
    windows = [x for rd in (*rounds.values(), tampered) for x in rd.windows]
    assert _evaluate(lw, windows) == [x.word for x in windows]
    assert [x.word & 1 for x in rounds["honest"].windows] == [1] * AT.WINDOWS
    assert all(0 in [x.word & 1 for x in rd.windows] for name, rd in rounds.items() if name != "honest")


def test_the_scan_names_both_ends_pieces_and_leaves_out_what_the_public_rows_hold():
    """The scan looks for every piece of both ends' heads, salts and siblings, and leaves out the pieces the windows'
    ``pub`` rows hold: on an advised link the egress head's counts in an accepted window are its public ``released``,
    so a piece of them with frames in it (an egress head's piece holding a bucket's frames) is public, while no piece of a salt or sibling is."""
    _, rounds = _link(n2.T)
    rd = rounds["honest"]
    blobs, held = n2.Z.blobs_of(rd)
    for end in n2.ENDS:
        assert {f"head {end} {w}" for w in range(n2.WINDOWS)} <= held
        assert f"certifier salt {end} 0" in blobs and f"sibling 1 {end} 0" in blobs
    assert HC.Shape(9, n2.T).HL == len(blobs["head egress 0"])
    scan = SimpleNamespace(PIECE=16, needles=lambda pieces: {v: n for n, v in pieces.items()},
                           find=lambda data, table: {n: 1 for v, n in table.items() if v in data})
    _, _, out = n2.Z.public_pieces(scan, blobs, held, b"".join(rd.lw.row(x.args, "pub") for x in rd.windows))
    head = blobs["head egress 0"]
    assert head[2128:2144].hex() in out and any(head[2128:2144])
    secret = {b[o:o + 16].hex() for k, b in blobs.items() if "salt" in k or "sibling" in k
              for o in range(0, len(b) - 15, 16)}
    assert not secret & set(out)


def test_the_prover_s_other_subset_is_another_k_subset_of_the_period():
    """`n2_zk.other_subset`, the ``other_subset`` member's windows: K of the period's, ascending, not the sample's units
    (the undrawn windows first), and the sample's own at K = n, where no other subset exists."""
    for n, k, units in ((26, 8, [3, 8, 10, 11, 15, 18, 19, 23]), (4, 3, [0, 1, 3]), (5, 1, [0]), (3, 2, [1, 2])):
        got = n2.Z.other_subset({"population": n, "k": k, "units": units})
        assert len(got) == len(set(got)) == k and got == sorted(got) and got != units and set(got) <= set(range(n))
    assert n2.Z.other_subset({"population": 26, "k": 8, "units": [3, 8, 10, 11, 15, 18, 19, 23]}) == [0, 1, 2, 4, 5,
                                                                                                    6, 7, 9]
    assert n2.Z.other_subset({"population": 4, "k": 4, "units": [0, 1, 2, 3]}) == [0, 1, 2, 3]


def test_a_sampled_period_adds_the_other_subset_member_and_an_unsampled_one_refuses_it(tmp_path):
    """``--draw K`` (K < n) adds ``other_subset``, the honest records, whose expected window verdicts are honest's; a
    period with every window proved has no other K-subset to prove, so asking for it is refused."""
    assert n2.main(["prove", "--dry", "--draw", "2", "--members", "honest,other_subset", "--out", str(tmp_path)]) == 0
    s = json.loads((tmp_path / "summary.json").read_text())
    assert set(s["members"]) == {"honest", n2.OTHER} and "draw" in s["changed"][n2.OTHER]
    assert s["members"][n2.OTHER]["reference"] == s["members"]["honest"]["reference"]
    assert s["draw"]["law"] == "subset:2" and s["draw"]["population"] == n2.WINDOWS
    with pytest.raises(SystemExit, match="needs --draw"):
        n2.main(["prove", "--dry", "--members", "honest,other_subset", "--out", str(tmp_path / "all")])


def test_the_provers_instance_holds_every_piece_the_control_expects_of_a_head_but_its_zero_padded_tail():
    """The control (`a8.received` on the prover's own instance files) expects every held piece of each head, but not its
    zero-padded last chunk: the advised unit's head is 157 + L + 4T bytes, 2214 at T = 512, so its last 6 bytes end
    off a 16-byte boundary, and the wit row holds the head followed by its salt's field, never by zeros, so a
    control on the instance files fails on exactly those pieces. Here the verification phases' wit rows, the rows the
    instance files hold as they are, stand in for them."""
    _, rounds = _link(n2.T)
    rd = rounds["honest"]
    blobs, held = n2.Z.blobs_of(rd)
    scan = SimpleNamespace(PIECE=16, needles=lambda pieces: {v: n for n, v in pieces.items()},
                           find=lambda data, table: {n: 1 for v, n in table.items() if v in data})
    kept, held_kept, _ = n2.Z.public_pieces(scan, blobs, held, b"")
    wit = b"".join(rd.lw.row(x.args, "wit") for x in rd.windows)
    head = blobs["head egress 0"]
    tail = f"head egress 0 @{len(head) - len(head) % 16}"
    assert len(head) == 2214 and tail == "head egress 0 @2208" and tail in kept and tail not in held_kept
    assert kept[tail] not in wit and wit.startswith(head) and wit[len(head):len(head) + 10] != bytes(10)
    assert held_kept and all(kept[k] in wit for k in held_kept)
    assert {k for k in kept if k.rsplit(" @", 1)[0] in held} - held_kept == {
        f"head {end} {w} @2208" for end in n2.ENDS for w in range(n2.WINDOWS)}


def test_jobs_is_one_firewalled_session_at_a_time(monkeypatch):
    monkeypatch.setattr(n2, "prove", lambda a: a.jobs)
    argv = ["prove", "--out", "o", "--dry"]
    assert n2.main(argv) == 1 and n2.main([*argv, "--jobs", "1"]) == 1
    with pytest.raises(SystemExit, match="outer proofs go through the firewall one at a time"):
        n2.main([*argv, "--jobs", "2"])
