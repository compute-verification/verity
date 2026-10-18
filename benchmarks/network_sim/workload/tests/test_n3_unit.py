"""N3's proof unit for one port-window (`n3_unit`): a descriptor puts its declared frame on its outbox rows, an outbox
holds its unit's messages' bytes, every opening the unit makes is its tree's, N2's frames root in gates is N2's, and the
circuit equals its Python reference. An honest window gives ok; a flipped bit, an extra frame, written fill, a wrong
outbox row (first or second of a segment, or a nonzero row for an empty segment), descriptor salt, position or record
salt, a head whose status isn't complete, or a wrong salt frame gives not-ok, on the demo's, the simulator's and the
lab's windows."""
import base64
import copy
import dataclasses
import random

import numpy as np
import pytest

from verity.core.protocols.certifier import certificate as CT
from verity.core.protocols.certifier.active import DATA, HEADER, LAST, encode_frame
from verity.core.protocols.certifier.grid import GridParams, Reason, Record, Status
from verity.ml.circuits.boolean.forms import Circuit
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation.bits import evaluate_bits
from verity.ml.commitments.gates.frame_v3 import root_ref
from verity.ml.commitments.identity import tagged_sha512
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as SCHEMA
from workload import framing as F
from workload import lab as L
from workload import n3
from workload import n3_unit as U
from workload import values as V
from workload.block import Context
from workload.run import run_memory, statement

def evaluate_forms(build, inputs: dict[str, list[bytes]], widths: dict[str, int]) -> list[list[int]]:
    """A forms builder's output bits on each instance, the inputs' bytes least significant bit first."""
    C = Circuit()
    ins = {k: C.input(k, w) for k, w in widths.items()}
    C.output("out", build(C, ins))
    T = len(next(iter(inputs.values())))
    vals = {k: [sum(((int.from_bytes(x, "little") >> i) & 1) << t for t, x in enumerate(xs)) for i in range(widths[k])]
            for k, xs in inputs.items()}
    outs, _ = C.evaluate(vals, T)
    return [[(o >> t) & 1 for o in outs["out"]] for t in range(T)]


def as_bytes(bits: list[int]) -> bytes:
    return bytes(sum(b << j for j, b in enumerate(bits[i:i + 8])) for i in range(0, len(bits), 8))


def bits(data: bytes, n: int) -> list[int]:
    x = int.from_bytes(data, "little")
    return [(x >> j) & 1 for j in range(n)]


def flip(f: bytes, at: int, bit: int = 1) -> bytes:
    return f[:at] + bytes([f[at] ^ bit]) + f[at + 1:]


def test_base64_characters_are_the_alphabet():
    got = evaluate_forms(lambda C, x: U.b64_char(C, x["v"]), {"v": [bytes([v]) for v in range(64)]}, {"v": 6})
    assert b"".join(as_bytes(b) for b in got) == U.B64_ALPHABET


@pytest.mark.parametrize("n", range(1, 8))
def test_base64_text_is_base64(n):
    rng = random.Random(n)
    data = [bytes(rng.randrange(256) for _ in range(n)) for _ in range(32)] + [bytes(n), b"\xff" * n]
    got = evaluate_forms(lambda C, x: U.b64(C, x["d"], n), {"d": data}, {"d": 8 * n})
    assert [as_bytes(b) for b in got] == [base64.b64encode(d) for d in data]


# --- descriptors and N3Frame -----------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def demo():
    return U.demo()


def test_a_descriptor_puts_its_declared_frame_on_its_outbox_rows(demo):
    f = demo.layout.fmt
    for pieces, d in zip(U.DEMO_FRAMES, demo.layout.descriptors["p0"]):
        row = U.encode(f, d)
        assert U.decode(f, row) == d and len(row) == -(-f.LW // 8)
        rows = [demo.units.pair(s, f.B)[0] for s in d.segs]
        assert U.assemble(f, row, rows) == U.expand(pieces, f.B, demo.units.outbox)
    assert [len(d.segs) for d in demo.layout.descriptors["p0"]] == [0, 1, 2, 2]
    assert demo.layout.descriptor("p1", 0) == U.NONE and U.fields(f, U.encode(f, U.NONE)).mode == 0


@pytest.mark.parametrize("pieces, error", [
    ([U.const(HEADER.pack(DATA, 1, 3)), ("r", ("u",), 0, 8), U.const(b"\x01"), ("r", ("u",), 8, 8), U.zeros(4)],
     "nonzero constant"),
    ([U.const(HEADER.pack(DATA, 1, 3)), ("r", ("u",), 0, 12), ("k", 4, 0), U.zeros(5)], "whole bytes"),
    ([U.const(HEADER.pack(DATA, 1, 3)), ("r", ("u",), 4, 8), U.zeros(6)], "whole bytes"),
], ids=["nonzero-constant", "text-past-a-byte", "outbox-bits-past-a-byte"])
def test_a_frame_a_segment_cant_hold_has_no_descriptor(pieces, error):
    with pytest.raises(ValueError, match=error):
        U.describe(pieces, 16, lambda k: 0)


def test_an_outbox_is_its_bytes_in_rows_up_to_a_power_of_two_with_a_row_after_them():
    data = bytes(range(1, 40))
    assert U.outbox_rows(b"", 16) == [bytes(16)]
    assert U.outbox_rows(data[:16], 16) == [data[:16], bytes(16)]
    assert U.outbox_rows(data[:17], 16) == [data[:16], data[16:17].ljust(16, b"\0")] + [bytes(16)] * 2
    assert U.outbox_rows(data, 16) == [data[:16], data[16:32], data[32:].ljust(16, b"\0"), bytes(16)]


def frame_cases(inst: U.Instance):
    """Each descriptor of the window with its G segments' two rows, and its frame (the salt frame skipped)."""
    a = inst.args
    frames = list(a["frames"])
    if inst.salted:
        del frames[a["slot"]]
    yield from zip(a.get("descriptors", []), a.get("rows", []), frames)


def n3frame(f: U.Format, G: int, cases) -> list[int]:
    """`N3Frame{G}`'s output on each (descriptor, rows, frame)."""
    descs, rows, frames = zip(*cases)
    args = [[bits(x, f.LW) for x in descs],
            [[b for two in rr[:G] for r in two for b in bits(r, 8 * f.B)] for rr in rows],
            [bits(x, 8 * f.B) for x in frames]]
    out = evaluate_bits(bind(U.N3Frame, FMT=tuple(f), G=G), *(np.asarray(a, dtype=np.uint8) for a in args))
    return np.asarray(out).reshape(-1).tolist()


def assembled(f: U.Format, G: int, cases) -> list[int]:
    return [int(U.fields(f, x).mode in (U.RAW, U.B64) and not any(s.kind for s in U.fields(f, x).segs[G:])
                and U.assemble(f, x, rows, G) == frame) for x, rows, frame in cases]


def by_g(windows):
    """The frame cases of the windows' instances, by their G."""
    out: dict[int, list] = {}
    for w in windows:
        if w.instance is not None:
            out.setdefault(w.instance.G, []).extend(frame_cases(w.instance))
    return out


@pytest.mark.parametrize("G", [1, 2])
def test_n3frame_is_assemble_on_valid_modes_and_refuses_the_others(demo, G):
    f = demo.layout.fmt
    honest = list(frame_cases(demo.window.instance)) if G == f.G else []
    rng = random.Random(G)
    cases = honest + [(x, r, flip(fr, i % f.B, 1 << i % 8)) for i, (x, r, fr) in enumerate(honest)]
    for i in range(32):
        x = bytearray(rng.randbytes(-(-f.LW // 8)))
        if i % 2:
            for g in range(G, f.G):
                x[(f.HW + g * f.SW) // 8] &= ~(1 << (f.HW + g * f.SW) % 8)
        x = bytes(x)
        rows = [(rng.randbytes(f.B), rng.randbytes(f.B)) for _ in range(G)]
        frame = U.assemble(f, x, rows, G)
        cases += [(x, rows, frame), (x, rows, flip(frame, rng.randrange(f.B), 1 << rng.randrange(8)))]
    want = assembled(f, G, cases)
    assert want[:len(honest)] == [1] * len(honest) and not any(want[len(honest):2 * len(honest)])
    assert {U.fields(f, x).mode for x, _, _ in cases} == {0, 1, 2, 3} and 1 in want
    assert n3frame(f, G, cases) == want


def test_n3seg_is_segment():
    B, rng = 16, random.Random(0)
    ob, lb = U._seg_widths(B)
    cases = [(rng.randbytes(B), rng.randbytes(B), rng.randrange(1 << ob), rng.randrange(1 << lb),
              rng.randrange(1 << lb)) for _ in range(200)]
    cases += [(b"\xff" * B, b"\xfe" * B, 15, 16, 0), (b"\xff" * B, b"\xfe" * B, 0, 31, 0),
              (b"\xff" * B, b"\xfe" * B, 15, 31, 20)]
    args = [[bits(r0 + r1, 16 * B) for r0, r1, *_ in cases], [bits(bytes([c[2]]), ob) for c in cases],
            [bits(bytes([c[3]]), lb) for c in cases], [bits(bytes([c[4]]), lb) for c in cases]]
    out = np.asarray(evaluate_bits(bind(U.N3Seg, B=B), *(np.asarray(a, dtype=np.uint8) for a in args)))
    want = [bits(U.segment(B, (1, 0, 0, o, n, d), r0, r1), 8 * B) for r0, r1, o, n, d in cases]
    assert out.reshape(len(cases), -1).tolist() == want


# --- openings and the demo window ------------------------------------------------------------------------------------

def test_every_row_opens_in_its_tree_at_its_position(demo):
    trees = [(t, 8 * demo.layout.fmt.B) for t in demo.units.trees.values()]
    trees += [(demo.layout.tree, demo.layout.fmt.LW)]
    for t, n_bits in trees:
        for i, (row, salt) in enumerate(zip(t.reg.rows, t.reg.salts)):
            assert root_ref(t.domain_id, SCHEMA, U.hiding_value(row, n_bits, salt), i, t.path(i)) == t.root
            other = U.hiding_value(row, n_bits, flip(salt, 0))
            assert root_ref(t.domain_id, SCHEMA, other, i, t.path(i)) != t.root


def test_the_record_commitment_is_n2s_and_its_head_leaf_holds_the_frames_root(demo):
    """The demo's record commitment is `LinkWindows`'s own root over its four leaves, and the egress head leaf is the
    unit's head bytes followed by N2's frames root of the window's frames."""
    rc = demo.record
    assert rc.root.hex() == rc.lw.root(rc.opening)
    for i, (row, salt) in enumerate(zip(rc.rows, rc.opening.salts)):
        assert root_ref(rc.domain_id, SCHEMA, U.hiding_value(row, 8 * len(row), salt), i, rc.path(i)) == rc.root
    root = CT.frames_root(demo.window.frames, GridParams(2, 2, demo.layout.fmt.B))
    assert rc.rows[U.N2.HEADS] == rc.head(0) + root and rc.rows[U.N2.HEADS + 1].endswith(CT.empty(2))


@pytest.mark.parametrize("B, N, T, R", [(16, 1, 1, 1), (16, 3, 2, 2), (24, 5, 3, 3)])
def test_n3framesroot_is_n2s_frames_root(B, N, T, R):
    rng = random.Random(N)
    frames = [rng.randbytes(B) for _ in range(N)]
    x = np.asarray([[b for f in frames for b in bits(f, 8 * B)]], dtype=np.uint8)
    out = np.asarray(evaluate_bits(bind(U.N3FramesRoot, B=B, N=N, D=(T * R - 1).bit_length()), x)).reshape(-1)
    want = U._digest_bits(CT.frames_root(frames, GridParams(T, R, B)))
    assert out.tolist() == want


@pytest.fixture(scope="module")
def demo_cases(demo):
    """The demo's window, honest and with one witness or public input changed."""
    inst = demo.window.instance
    a = inst.args

    def edit(**kw):
        x = copy.deepcopy(inst)
        x.args.update(kw)
        return x

    def row(i, g, k, at):
        rows = copy.deepcopy(a["rows"])
        rows[i][g][k] = flip(rows[i][g][k], at)
        return edit(rows=rows)

    fr = a["frames"]
    s = demo.setup
    overflow = dataclasses.replace(demo.window, reason=Reason.OVERFLOW.value, instance=None)
    rc = s.certifier.commit("demo", 0, [s.head("p0", 0, overflow), s.head("p1", 0, None)], s.grids["p0"],
                            s.grids["p1"], s.sync_set)
    return {
        "honest": inst,
        "frame-bit": edit(frames=[fr[0], flip(fr[1], 9), fr[2], fr[3]]),
        "suffix-bit": edit(frames=[fr[0], fr[1], flip(fr[2], 14), fr[3]]),
        "outbox-row": row(1, 0, 0, 14),
        "second-row": row(1, 0, 1, 2),
        "zero-row": row(0, 1, 0, 0),
        "descriptor-salt": edit(descriptor_salts=[flip(a["descriptor_salts"][0], 0)] + a["descriptor_salts"][1:]),
        "at": edit(at=a["at"] + 1),
        "record-salt": edit(record_salt=flip(a["record_salt"], 3)),
        "direction": edit(direction=1),
        "head-count": edit(head=flip(a["head"], len(a["head"]) - 1)),
        "incomplete": edit(record=rc.root, head=rc.head(0), record_salt=rc.opening.salts[U.N2.HEADS],
                           record_path=rc.path(U.N2.HEADS)),
    }


def test_the_demo_windows_reference_holds_on_the_honest_window_alone(demo, demo_cases):
    assert demo.window.instance.key == (tuple(demo.layout.fmt), 4, 2, False, (4, 2, 2))
    assert [U.reference(x) for x in demo_cases.values()] == [True] + [False] * (len(demo_cases) - 1)


def test_an_incomplete_head_opens_and_only_its_status_refuses_it(demo_cases):
    x = demo_cases["incomplete"]
    a, rec = x.args, x.rec
    head = a["head"] + CT.frames_root(a["frames"], GridParams(rec.T, rec.R, x.fmt.B))
    assert head[rec.ST] == CT.STATUS_CODES[Reason.OVERFLOW]
    value = U.hiding_value(head, 8 * len(head), a["record_salt"])
    assert root_ref(a["record_domain"], SCHEMA, value, U.N2.HEADS, a["record_path"]) == a["record"]


@pytest.mark.slow
def test_the_demo_window_is_its_reference(demo_cases):
    """The whole window at its smallest format (B = 16, four frames of two segments), on one honest and twelve broken
    instances. It is the one test of the window's wiring, which the cheaper tests can't catch: the descriptors' reads at
    at, at + 1, ..., the segments' identities cut from them, both outbox rows of each segment (zero rows among them),
    N2's frames root, the head's status and its opening at leaf 2 + direction. About two minutes, nearly all of it
    tracing and evaluating the openings' SHA-512 compressions."""
    assert U.evaluate(*demo_cases.values()) == [int(U.reference(x)) for x in demo_cases.values()]
    size = U.size(demo_cases["honest"])
    assert size["ands"] == sum(v for k, v in size.items() if k != "ands") and size["row_reads"] > size["frames"]


@pytest.mark.slow
def test_n3outboxpair_is_its_reference(demo):
    """Both rows of a segment opened, on the demo's outboxes: honest of either unit, a second row past the identity's
    row bits (q = 3, where q + 1 wraps to row 0 and both rows open), another unit's rows, and kind 0 with zero and with
    nonzero rows. It is the one test that reaches the q + 1 carry, which no honest layout sets; about a minute."""
    f, units = demo.layout.fmt, demo.units
    u0 = units.trees["u0"]
    wrap = ([u0.reg.rows[3], u0.reg.rows[0]], [u0.reg.salts[3], u0.reg.salts[0]], [u0.path(3), u0.path(0)])
    zero = units.pair(U.Segment(), f.B)
    cases = [(U.Segment(1, 0, 1), units.pair(U.Segment(1, 0, 1), f.B)),
             (U.Segment(1, 1, 0), units.pair(U.Segment(1, 1, 0), f.B)),
             (U.Segment(1, 0, 3), wrap), (U.Segment(1, 1, 1), units.pair(U.Segment(1, 0, 1), f.B)),
             (U.Segment(), zero), (U.Segment(), ([zero[0][0], flip(zero[0][1], 5)], zero[1], zero[2]))]
    want = [int(U.pair_ok(f, s, *r, units.roots, units.domains)) for s, r in cases]
    assert want == [1, 1, 0, 0, 1, 0]
    ident = [bits((s.kind | s.u << 1 | s.q << (1 + f.UB)).to_bytes(2, "little"), f.IDW) for s, _ in cases]
    rows = [[b for r in rr[0] for b in bits(r, 8 * f.B)] for _, rr in cases]
    salts = [[b for s in rr[1] for b in U._digest_bits(s)] for _, rr in cases]
    paths = [[b for p in rr[2] for s in list(p) + [bytes(64)] * (f.DB - len(p)) for b in U._digest_bits(s)]
             for _, rr in cases]
    roots = [[b for r in units.roots for b in U._digest_bits(r)]] * len(cases)
    domains = [[b for r in units.domains for b in U._digest_bits(r)]] * len(cases)
    args = [np.asarray(x, dtype=np.uint8) for x in (ident, rows, salts, paths, roots, domains)]
    out = evaluate_bits(bind(U.N3OutboxPair, DEPTHS=f.DEPTHS, N_BITS=8 * f.B), *args)
    assert np.asarray(out).reshape(-1).tolist() == want


@pytest.mark.parametrize("fmt, N, G, salted, rec", [
    ((16, 13, 2, 2, (2, 0), 4), 4, 3, 0, (4, 2, 2)),
    ((16, 13, 2, 2, (2, 0), 4), 4, 0, 0, (4, 2, 2)),
    ((16, 13, 2, 0, (2, 0), 4), 4, 1, 0, (4, 2, 2)),
    ((16, 13, 2, 2, (2, 0), 4), 0, 2, 0, (4, 2, 2)),
    ((16, 13, 2, 2, (2, 0), 4), 2, 2, 1, (4, 2, 2)),
    ((96, 13, 2, 2, (2, 0), 4), 2, 2, 2, (4, 2, 2)),
    ((16, 0, 2, 2, (2, 0), 4), 1, 1, 0, (4, 2, 2)),
    ((16, 13, 2, 2, (), 4), 1, 1, 0, (4, 2, 2)),
    ((16, 13, 2, 2, (2, 0), 4), 5, 2, 0, (4, 2, 2)),
    ((16, 13, 2, 2, (2, 0), 4), 1, 1, 0, (4, 0, 2)),
    ((16, 13, 2, 2, (2, 0), 4), 1, 1, 0, (4, 2)),
], ids=["segments-past-the-format", "no-segments", "a-format-of-no-segments", "no-frames", "salt-frame-past-B",
        "salted-not-a-bit", "no-prefix", "no-units", "frames-past-the-grid", "no-buckets", "no-slots"])
def test_a_size_class_the_unit_doesnt_cover_is_refused(fmt, N, G, salted, rec):
    with pytest.raises(ValueError):
        bind(U.N3Window, FMT=fmt, N=N, G=G, SALTED=salted, REC=rec).signature()


# --- the simulator ---------------------------------------------------------------------------------------------------

SIM = {"spec": {"d": 4, "layers": 3, "requests": 1}, "blocks": 3, "devices": 2, "frame_bytes": 64, "seed": 0}
BUCKETS = 4
TAMPERS = ("b0:flip-bit", "b1:extra-frame", "b0:fill")


def sim_run(tamper: str | None):
    """A run in one process: its setup and windows, and the in-the-clear check's report."""
    config = dict(SIM, **({"tamper": {"unit": tamper.split(":")[0], "mode": tamper.split(":")[1]}} if tamper else {}))
    ctx = Context(config)
    results = run_memory(ctx)
    public, witness = statement(ctx, results)
    records, _ = n3.merge(results)
    return U.sim_windows(public, witness, records, BUCKETS), n3.check(public, witness, records)


@pytest.fixture(scope="module")
def sim():
    return {t: sim_run(t) for t in (None, *TAMPERS)}


def not_ok(windows) -> set[tuple[str, int]]:
    return {(w.port, w.window) for w in windows if not w.check(circuit=False)["ok"]}


def test_the_simulators_windows_are_ok_exactly_where_n3_in_the_clear_holds(sim):
    (_, honest), report = sim[None]
    assert report.ok and not not_ok(honest) and all(w.instance is not None for w in honest if w.frames)
    for t in TAMPERS:
        (_, windows), report = sim[t]
        bad = {p for p, _ in not_ok(windows)}
        assert bad and bad == {v.where for v in report.violations}, t


def test_a_tampered_window_runs_the_unit_and_the_position_of_its_honest_one(sim):
    """The unit and its public position depend on the size class and the occupancy alone, not on what was recorded
    (each run's commitments have their own salts, so the roots differ)."""
    (setup, honest), _ = sim[None]
    before = {(w.port, w.window): w.instance for w in honest if w.instance is not None}
    for t in ("b0:flip-bit", "b0:fill"):
        (other, windows), _ = sim[t]
        assert other.layout.descriptors == setup.layout.descriptors and other.layout.cap == setup.layout.cap
        for w in windows:
            if w.instance is not None:
                h = before[(w.port, w.window)]
                assert (w.instance.key, w.instance.args["at"]) == (h.key, h.args["at"]), (t, w.port, w.window)


def test_n3frame_is_assemble_on_the_simulators_frames(sim):
    """φ's layouts through `N3Frame`: HELLOs and values split across frames, honest and tampered."""
    (setup, _), _ = sim[None]
    groups = by_g([w for t in (None, *TAMPERS) for w in sim[t][0][1]])
    want = {G: assembled(setup.fmt, G, cases) for G, cases in groups.items()}
    assert any(0 in v for v in want.values())
    assert {G: n3frame(setup.fmt, G, cases) for G, cases in groups.items()} == want


# --- the lab ---------------------------------------------------------------------------------------------------------

BLOCK = "b1"
SALTED_B = 96  # a salt frame is 81 bytes; at 54 an event's chunk is 3 bytes, so the MLP's messages take several


def cut(stream: int, request: bytes, B: int) -> list[bytes]:
    room = B - HEADER.size
    chunks = [request[i:i + room] for i in range(0, len(request), room)]
    return [encode_frame(LAST if j == len(chunks) - 1 else DATA, stream, c, B) for j, c in enumerate(chunks)]


def commitment(block: str, window: int) -> bytes:
    return tagged_sha512("verity-network-certifier/salt/v1", f"{block}/{window}".encode().ljust(64, b"\0"))


def lab_records(block: str, egress: list[bytes], ingress: list[bytes], B: int, salted: bool) -> list[Record]:
    """Two frames a window in two buckets of two slots; with `salted`, each ingress window's salt frame first."""
    out = []
    for link, frames in ((L.EGRESS, egress), (L.INGRESS, ingress)):
        for w in range(0, len(frames), 2):
            fs = frames[w:w + 2]
            if salted and link == L.INGRESS:
                fs = [U.salt_frame(w // 2, commitment(block, w // 2), B)] + fs
            slots = fs + [None] * (4 - len(fs))
            out.append(Record(link, w // 2, (tuple(slots[:2]), tuple(slots[2:])), Status()))
    return out


def lab_setup(B: int):
    """An honest run's statement and witness, each block's honest egress and ingress frames, B, and each message's
    payload by stream."""
    spec = {"d": 4, "layers": 3, "requests": 2}
    ctx = Context({"spec": spec, "blocks": 3, "devices": 2, "frame_bytes": B, "seed": 1})
    program = L.Lab.of(ctx, B)
    values = dict(ctx.inputs) | ctx.constants
    ctx.model.evaluate(values)
    results = {}
    for u in ctx.hw.unit_names():
        c = V.commit(u, program.held[u], values, ctx.model.value_bits)
        results[u] = {"values": {"root": c.root.hex(), "count": len(c.gates),
                                 "rows": [r.hex() for r in c.registration.rows],
                                 "salts": [s.hex() for s in c.registration.salts]}}
    public, witness = statement(ctx, results)
    payloads = {s: F.encode(m, values) for s, m in program.messages.items()}
    frames = {}
    for b in program.blocks:
        egress = [encode_frame(DATA, j + 1, program.event(b, p, payloads[m.stream]), B)
                  for p, (j, m, _) in enumerate(program.events[b])]
        frames[b] = (egress, [f for s in program.sessions[b] for f in cut(s.stream, program.request(s, payloads), B)])
    return public, witness, frames, B, payloads


@pytest.fixture(scope="module")
def lab():
    return lab_setup(54)


@pytest.fixture(scope="module")
def salted_lab():
    return lab_setup(SALTED_B)


def lab_windows(lab, edit=None, salts=None):
    """The setup and windows of the lab's records, `BLOCK`'s frames edited; salted when `salts` is given."""
    public, witness, frames, B, _ = lab
    published = {}
    for b, (e, i) in frames.items():
        if b == BLOCK and edit is not None:
            e, i = edit(e, i)
        published[b] = lab_records(b, e, i, B, salts is not None)
    return U.lab_windows(public, witness, published, salts)


EGRESS, INGRESS = f"{BLOCK}:{L.EGRESS}", f"{BLOCK}:{L.INGRESS}"


def test_an_honest_lab_runs_windows_are_ok(lab):
    setup, windows = lab_windows(lab)
    assert windows and not not_ok(windows) and all(w.instance is not None for w in windows)
    assert {w.port.split(":")[1] for w in windows} == {L.EGRESS, L.INGRESS}
    assert all(d.mode == U.B64 for p, ds in setup.layout.descriptors.items() if p.endswith(L.EGRESS) for d in ds)
    assert all(d.mode == U.RAW for p, ds in setup.layout.descriptors.items() if p.endswith(L.INGRESS) for d in ds)
    assert setup.layout.segments(EGRESS) == setup.layout.segments(INGRESS) == 1


def test_an_outbox_holds_its_units_messages_back_to_back_in_send_order(lab):
    """Each unit's outbox is the payloads of the messages it sends, as the lab puts them on the wire, in φ's send
    order; and each message's run starts at its offset."""
    public, witness, _, _, payloads = lab
    S = U.statement(public, witness)
    data, base = S.outboxes
    order = S.phi.send_order()
    assert set(data) == set(S.hw.unit_names()) and sum(map(len, data.values())) == sum(map(len, payloads.values()))
    for u, x in data.items():
        assert x == b"".join(payloads[m.stream] for m in order if m.src == u)
    for m in order:
        assert data[m.src][base[m.stream]:base[m.stream] + len(payloads[m.stream])] == payloads[m.stream]


def data_end(f: bytes) -> int:
    """The index in frame f of its event's last data byte, before the closing blank line."""
    return HEADER.size + HEADER.unpack_from(f)[2] - 3


@pytest.mark.parametrize("edit, refused", [
    (lambda e, i: ([flip(e[0], data_end(e[0]))] + e[1:], i), {(EGRESS, 0)}),
    (lambda e, i: (e, [flip(i[0], HEADER.size)] + i[1:]), {(INGRESS, 0)}),
    (lambda e, i: (e, i[:-1] + [flip(i[-1], HEADER.size + 4)]), {(INGRESS, -1)}),
    (lambda e, i: ([e[0][:-1] + b"\x01"] + e[1:], i), {(EGRESS, 0)}),
], ids=["flipped-event", "flipped-request", "flipped-last-request", "fill"])
def test_a_tampered_lab_window_is_not_ok(lab, edit, refused):
    last = (len(lab[2][BLOCK][1]) - 1) // 2
    assert not_ok(lab_windows(lab, edit)[1]) == {(p, last if w == -1 else w) for p, w in refused}


def test_an_extra_lab_frame_shifts_every_window_after_it_onto_the_wrong_descriptors(lab):
    setup, windows = lab_windows(lab, lambda e, i: (e[:1] + e, i))
    egress = sorted(w.window for w in windows if w.port == EGRESS)
    assert not_ok(windows) == {(EGRESS, w) for w in egress}
    declared = len(setup.layout.descriptors[EGRESS])
    last = next(w for w in windows if w.port == EGRESS and w.window == egress[-1])
    assert last.instance is not None and declared + 1 <= setup.layout.cap
    assert setup.layout.descriptor(EGRESS, declared) == U.NONE


def test_a_lab_record_whose_coverage_wasnt_established_is_refused_with_no_frames(lab):
    public, witness, frames, B, _ = lab
    published = {b: lab_records(b, e, i, B, False) for b, (e, i) in frames.items()}
    lost = Record(L.EGRESS, 0, ((None, None), (None, None)), Status(Reason.COVERAGE_NEVER_ESTABLISHED))
    published[BLOCK] = [lost if (r.link, r.window) == (L.EGRESS, 0) else r for r in published[BLOCK]]
    w = next(w for w in U.lab_windows(public, witness, published)[1] if (w.port, w.window) == (EGRESS, 0))
    assert w.instance is None and not w.check(circuit=False)["ok"]
    assert w.why == "the record's status is coverage-never-established"


def revealed(lab) -> dict[tuple[str, int], bytes]:
    return {(b, w): commitment(b, w) for b in lab[2] for w in range(len(lab[2][b][1]))}


def test_a_salted_ingress_window_holds_against_its_revealed_commitment_alone(salted_lab):
    salts = revealed(salted_lab)
    assert not not_ok(lab_windows(salted_lab, salts=salts)[1])
    unsalted = not_ok(lab_windows(salted_lab, salts={})[1])
    assert unsalted and all(p.endswith(L.INGRESS) for p, _ in unsalted)
    wrong = salts | {(BLOCK, 0): commitment(BLOCK, 99)}
    assert not_ok(lab_windows(salted_lab, salts=wrong)[1]) == {(INGRESS, 0)}


def test_n3frame_is_assemble_on_the_labs_frames(lab):
    """The lab's layouts through `N3Frame`: base64 events and requests across frames, honest and tampered."""
    setup, honest = lab_windows(lab)
    _, flipped = lab_windows(lab, lambda e, i: ([flip(e[0], data_end(e[0]))] + e[1:],
                                                [flip(i[0], HEADER.size)] + i[1:]))
    groups = by_g(honest + flipped)
    want = {G: assembled(setup.fmt, G, cases) for G, cases in groups.items()}
    assert all(0 in v for v in want.values())
    assert {G: n3frame(setup.fmt, G, cases) for G, cases in groups.items()} == want


@pytest.mark.slow
def test_a_salted_window_is_its_reference(salted_lab):
    """The salted window (B = 96): honest, with a wrong revealed commitment, and with the salt frame's slot pointing
    at a data frame. It is the one test of the salt path, which the demo can't reach (a salt frame needs B >= 81):
    the slot's one-hot selection of the salt frame, its comparison with the revealed commitment, and the routing of
    the other frames onto the descriptors."""
    salts = revealed(salted_lab)
    honest = next(w for w in lab_windows(salted_lab, salts=salts)[1] if w.port == INGRESS and w.window == 0).instance
    wrong = copy.deepcopy(honest)
    wrong.args["salt"] = wrong.args["salt"][:8] + commitment(BLOCK, 99)
    moved = copy.deepcopy(honest)
    moved.args["slot"] = 1
    cases = [honest, wrong, moved]
    assert honest.salted and honest.args["slot"] == 0
    assert U.evaluate(*cases) == [int(U.reference(x)) for x in cases] == [1, 0, 0]
