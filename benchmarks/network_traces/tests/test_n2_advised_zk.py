"""n2_advised_zk's link-windows on the advised tests' traffic (`advised_traffic`). At the driver's T = 512 the session
is varied, every window receives, its honest windows' output words are verdict 1 and deadline 1, and each negative (a
moved release bucket, an output length one short, extra egress) fails in some window on the advised counts alone, its
head checks passing and every window's unit taking an instance; the verifier's own ``pub`` row, from public fields,
the certificates and its own view of the public advice, is the prover's, in whole u16 words. The --e2e link's advice
(`joined`) numbers each Commit window's requests after the windows' before it. At the tests' T = 8 the
proof unit (`LinkWindows.proof_unit`), evaluated on the rows the prover stages, gives the output words the gates'
references do (`LinkWindows.outputs`) in every window of every member and on a head the prover changed. Each builder
`n2_advised_zk.builders` attaches, called as C-Flock's lowering calls it, traces to its part's own gates. Nothing here
lowers or proves (`n2_advised_zk.prove` does, on a node)."""

from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from functools import cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from verity.ml.circuits.boolean import trace as TR
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


def test_a_served_runs_clock_source_is_its_certifiers_signed_syncs_else_its_log_as_a_stand_in(tmp_path):
    """`served_clocks`, the --e2e link's clock source: the run's ``clocks.jsonl`` (the certifier's clock certificates,
    `LinkWindows.clock_certificates`) when it has one, even beside a proxy log; else `CertifierClocks`, the syncs its
    certifier logged (``proxy.json.gz``) for its link alone, labelled a stand-in; else no source, and the run is
    refused. On the simulated link at T = 8, whose certifier signed every window's sync, every window has its view
    under either source. Under the certificates, a window whose sync is wrongly signed (here window 0's, swapped after
    signing) has no sync, and neither has any window after it. Under the log, a window whose logged sync isn't the one
    it declares is refused alone."""
    sim, _ = _link(AT.T)
    lw, (link, pairs, _) = sim.lw, sim.members["honest"]
    egress, ingress = ([list(p[d].head.counts) for p in pairs] for d in (0, 1))
    signed = tuple(lw.clock_certificates)
    assert sim.clocks["source"] == "signed" and sim.clocks["as_declared"] and len(signed) == len(pairs)
    with pytest.raises(SystemExit, match="no clock source"):
        n2.served_clocks(tmp_path, lw.link)
    with gzip.open(tmp_path / "proxy.json.gz", "wt") as fh:
        json.dump({"syncs": [[s.delta, str(s.rho)] for s in AT.SYNCS]}, fh)
    logged = n2.replace(lw, clock_certificates=None, **n2.served_clocks(tmp_path, lw.link))
    assert isinstance(logged.clocks, n2.CertifierClocks) and logged.clocks.syncs("another") == ()
    label = n2.clocks_label(logged, AT.SYNCS)
    assert label["source"] == "stand-in" and "proxy.json.gz" in label["by"] and label["as_declared"]
    assert not n2.clocks_label(logged, AT.SYNCS[::-1])["as_declared"]
    assert None not in n2.advised_refs(logged, link, egress, ingress)
    (tmp_path / "clocks.jsonl").write_text("".join(json.dumps(n2.cs.clock_json(c)) + "\n" for c in signed))
    read = n2.served_clocks(tmp_path, lw.link)
    assert list(read) == ["clock_certificates"] and read["clock_certificates"] == signed
    again = n2.replace(lw, clock_certificates=read["clock_certificates"])
    assert n2.clocks_label(again, AT.SYNCS)["source"] == "signed"
    assert None not in n2.advised_refs(again, link, egress, ingress)
    forged = (n2.replace(signed[0], sync=AT.SYNCS[1]), *signed[1:])
    assert n2.advised_refs(n2.replace(lw, clock_certificates=forged), link, egress, ingress) == [None] * len(pairs)
    off = n2.CertifierClocks(lw.link, (n2.replace(AT.SYNCS[0], delta=AT.SYNCS[0].delta + 1), *AT.SYNCS[1:]))
    refs = n2.advised_refs(n2.replace(lw, clock_certificates=None, clocks=off), link, egress, ingress)
    assert refs[0] is None and None not in refs[1:]


def test_joined_advice_numbers_each_windows_requests_after_the_windows_before_and_releases_by_id():
    """`joined` (the --e2e link's advice from its Commit windows'): the second window's rows and hit sources name its
    requests after the first's, and each kept output is released where the certified frames of its request say."""
    a = _link(n2.T)[0].session.link.advice
    b = ADV.Advice(tuple(n2.replace(q, id=f"b{q.id}") for q in a.requests), a.steps, a.release)
    frames = {q.id: [1000 * k + 10 * j + i for i in range(q.output_len)]
              for k, x in enumerate((a, b)) for j, q in enumerate(x.requests)}
    got, n = n2.joined([a, b], frames), len(a.requests)
    assert got.requests == a.requests + b.requests and got.steps[:len(a.steps)] == a.steps
    rows = [r for st in got.steps[len(a.steps):] for r in st.rows]
    assert [(r.request, [s.request for s in r.sources]) for r in rows] == \
        [(r.request + n, [s.request + n for s in r.sources]) for st in a.steps for r in st.rows]
    assert any(r.sources for r in rows)
    assert all(b_ == frames[got.requests[o.request].id][o.index] for o, b_ in zip(ADV.outputs(got), got.release))


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


@pytest.mark.slow
def test_each_advised_builder_traces_to_its_parts_gates_called_as_c_flock_calls_it():
    parts = n2.builders()
    assert {"NetworkAdvisedOwn", "NetworkAdvisedBucket", "NetworkAdvisedEnd"} <= {d.name for d in parts}
    for d in parts:
        if not d.name.startswith("NetworkAdvised"):
            continue
        key = (n2.T, 4)
        S = SimpleNamespace(**dict(zip(d.statics, key)))
        params, ret = d.signature(S)
        got, want = TR.trace(lambda C, *e, d=d, S=S: d.build(C, S, *e), [t for _, t in params], ret), d.traced(*key)
        assert (got.gates, got.outputs) == (want.gates, want.outputs), d.name


def test_the_scan_names_both_ends_pieces_and_leaves_out_what_the_public_rows_hold():
    """The scan looks for every piece of both ends' heads, salts and siblings, and leaves out the pieces the windows'
    ``pub`` rows hold: on an advised link the egress head's counts in an accepted window are its public ``released``,
    so a piece of them with frames in it (run r20261024-084443-2f66's ``egress head 0 @2128``, bucket 509's two
    frames) is public, while no piece of a salt or sibling is."""
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
