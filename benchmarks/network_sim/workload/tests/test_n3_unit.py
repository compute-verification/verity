"""N3's proof unit for one port-window (`n3_unit`): the circuit equals its Python reference; an honest window gives
ok, and a flipped bit, an extra frame, written fill or a wrong salt gives not-ok, on the simulator's windows and the
lab's."""
import base64
import random

import pytest

from verity.core.protocols.certifier.active import DATA, HEADER, LAST, encode_frame
from verity.core.protocols.certifier.grid import Record, Status
from verity.ml.circuits.boolean.forms import Circuit
from verity.ml.commitments.identity import tagged_sha512
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


def test_base64_characters_are_the_alphabet():
    got = evaluate_forms(lambda C, x: U.b64_char(C, x["v"]), {"v": [bytes([v]) for v in range(64)]}, {"v": 6})
    assert b"".join(as_bytes(bits) for bits in got) == U.B64_ALPHABET


@pytest.mark.parametrize("n", range(1, 8))
def test_base64_text_is_base64(n):
    rng = random.Random(n)
    data = [bytes(rng.randrange(256) for _ in range(n)) for _ in range(32)] + [bytes(n), b"\xff" * n]
    got = evaluate_forms(lambda C, x: U.b64(C, x["d"], n), {"d": data}, {"d": 8 * n})
    assert [as_bytes(bits) for bits in got] == [base64.b64encode(d) for d in data]


def layout(**kw) -> dict:
    return dict(U.DEMO, **kw)


@pytest.mark.parametrize("bad", [
    layout(counts=[1, 0, 1]),
    layout(counts=[0, 0, 0], frames=[]),
    layout(counts=[1, 0, 1 << 16]),
    layout(rows=1),
    layout(frames=[U.DEMO["frames"][0], U.DEMO["frames"][1][:-1] + [U.zeros(3)], U.DEMO["frames"][2]]),
    layout(frames=[U.DEMO["frames"][0], [U.const(bytes(9)), ("r", 0, 250, 8), U.zeros(6)], U.DEMO["frames"][2]]),
    layout(frames=[U.DEMO["frames"][0], U.DEMO["frames"][1], [("b", (("k", 8, 0),), 2), U.zeros(12)]]),
], ids=["counts-not-frames", "no-frames", "count-past-16-bits", "row-out-of-range", "short-frame", "past-the-row",
        "base64-of-the-wrong-length"])
def test_a_layout_it_doesnt_cover_is_refused(bad):
    assert U.why(U.DEMO) is None and U.why(bad) is not None
    with pytest.raises(ValueError, match="refuses this binding"):
        U.unit(bad)


def flip(f: bytes, at: int, bit: int = 1) -> bytes:
    return f[:at] + bytes([f[at] ^ bit]) + f[at + 1:]


@pytest.mark.slow
def test_the_demo_unit_holds_on_the_honest_window_alone():
    """The whole Definition, its SHA-512 compression included: about 7 s, most of it tracing the compression."""
    rng = random.Random(0)
    rows = [rng.randbytes(32) for _ in range(2)]
    frames = U.expand(U.DEMO, rows)
    digest, counts = U.record_digest(frames), U.DEMO["counts"]
    other = [rows[0], flip(rows[1], 0, 0x80)]
    flipped = [frames[0], flip(frames[1], 9), frames[2]]
    instances = [
        (digest, counts, frames, rows),
        (U.record_digest(flipped), counts, flipped, rows),
        (flip(digest, 63), counts, frames, rows),
        (digest, [2, 0, 1], frames, rows),
        (digest, counts, frames, other),
        (U.record_digest(frames[:2] + [flip(frames[2], 15)]), counts, frames[:2] + [flip(frames[2], 15)], rows),
    ]
    assert U.evaluate(U.DEMO, *instances) == [1, 0, 0, 0, 0, 0]
    assert [U.reference(U.DEMO, *x) for x in instances] == [True] + [False] * 5
    size = U.size(U.DEMO)
    assert size["compressions"] == 1 and size["ands"] == size["ands_match"] + size["ands_sha512"] + 511 + 1


# --- the simulator ---------------------------------------------------------------------------------------------------

SIM = {"spec": {"d": 4, "layers": 3, "requests": 1}, "blocks": 3, "devices": 2, "frame_bytes": 64, "seed": 0}
BUCKETS = 4


def sim_run(tamper: str | None):
    """A run in one process: its public statement and witness, each link direction's records, the windows and the
    in-the-clear check's report."""
    config = dict(SIM, **({"tamper": {"unit": tamper.split(":")[0], "mode": tamper.split(":")[1]}} if tamper else {}))
    ctx = Context(config)
    results = run_memory(ctx)
    public, witness = statement(ctx, results)
    records, _ = n3.merge(results)
    return U.sim_windows(public, witness, records, BUCKETS), n3.check(public, witness, records)


@pytest.fixture(scope="module")
def sim():
    return {t: sim_run(t) for t in (None, "b0:flip-bit", "b1:extra-frame", "b0:fill")}


def test_the_simulators_windows_are_ok_exactly_where_n3_in_the_clear_holds(sim):
    honest, report = sim[None]
    assert report.ok and honest and all(w.check(circuit=False)["ok"] for w in honest)
    assert all(w.layout is not None for w in honest if w.frames)
    for t in ("b0:flip-bit", "b1:extra-frame", "b0:fill"):
        windows, report = sim[t]
        bad = {w.port for w in windows if not w.check(circuit=False)["ok"]}
        assert bad and bad == {v.where for v in report.violations}, t


@pytest.mark.slow
def test_the_circuit_is_the_reference_on_the_simulators_windows(sim):
    """The Definition on φ's layouts (HELLOs, values across frames), each window compiled once."""
    for t, (windows, _) in sim.items():
        for w in windows:
            if w.layout is not None and (t is None or not w.check(circuit=False)["ok"]):
                r = w.check()
                assert r["circuit"] == r["ok"] == (t is None), (t, w.port, w.window)


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
    """The lab's Program, an honest run's statement and witness, and each block's honest egress and ingress frames."""
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
    return public, witness, frames, B


@pytest.fixture(scope="module")
def lab():
    return lab_setup(54)


@pytest.fixture(scope="module")
def salted_lab():
    return lab_setup(SALTED_B)


def lab_windows(lab, edit=None, salts=None):
    """The windows of the lab's records, `BLOCK`'s frames edited; salted when `salts` is given."""
    public, witness, frames, B = lab
    published = {}
    for b, (e, i) in frames.items():
        if b == BLOCK and edit is not None:
            e, i = edit(e, i)
        published[b] = lab_records(b, e, i, B, salts is not None)
    return U.lab_windows(public, witness, published, salts)


def not_ok(windows) -> set[tuple[str, int]]:
    return {(w.port, w.window) for w in windows if not w.check(circuit=False)["ok"]}


EGRESS, INGRESS = f"{BLOCK}:{L.EGRESS}", f"{BLOCK}:{L.INGRESS}"


def test_an_honest_lab_runs_windows_are_ok(lab):
    windows = lab_windows(lab)
    assert windows and not not_ok(windows) and all(w.layout is not None for w in windows)
    assert {w.port.split(":")[1] for w in windows} == {L.EGRESS, L.INGRESS}
    assert all(any(p[0] == "b" for p in f) for w in windows if w.port.endswith(L.EGRESS) for f in w.layout["frames"])


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
    assert not_ok(lab_windows(lab, edit)) == {(p, last if w == -1 else w) for p, w in refused}


def test_an_extra_lab_frame_shifts_every_window_after_it_and_runs_past_the_program(lab):
    windows = lab_windows(lab, lambda e, i: (e[:1] + e, i))
    egress = sorted(w.window for w in windows if w.port == EGRESS)
    assert not_ok(windows) == {(EGRESS, w) for w in egress}
    assert next(w for w in windows if w.port == EGRESS and w.window == egress[-1]).why.startswith("frames past")


def revealed(lab) -> dict[tuple[str, int], bytes]:
    return {(b, w): commitment(b, w) for b in lab[2] for w in range(len(lab[2][b][1]))}


def test_a_salted_ingress_window_holds_against_its_revealed_commitment_alone(salted_lab):
    salts = revealed(salted_lab)
    assert not not_ok(lab_windows(salted_lab, salts=salts))
    unsalted = not_ok(lab_windows(salted_lab, salts={}))
    assert unsalted and all(p.endswith(L.INGRESS) for p, _ in unsalted)
    wrong = salts | {(BLOCK, 0): commitment(BLOCK, 99)}
    assert not_ok(lab_windows(salted_lab, salts=wrong)) == {(INGRESS, 0)}


@pytest.mark.slow
def test_the_circuit_is_the_reference_on_the_labs_windows(lab, salted_lab):
    """The Definition on the lab's layouts (base64 events, requests across frames, a salt frame), each compiled once."""
    honest = [w for w in lab_windows(lab) if w.port.startswith(BLOCK) and w.window == 0]
    flipped = [w for w in lab_windows(lab, lambda e, i: ([flip(e[0], data_end(e[0]))] + e[1:], i))
               if w.port == EGRESS and w.window == 0]
    salts = revealed(salted_lab)
    salted = [w for w in lab_windows(salted_lab, salts=salts) if w.port == INGRESS and w.window == 0]
    wrong = [w for w in lab_windows(salted_lab, salts=salts | {(BLOCK, 0): bytes(64)})
             if w.port == INGRESS and w.window == 0]
    for windows, ok in ((honest, 1), (flipped, 0), (salted, 1), (wrong, 0)):
        assert windows
        for w in windows:
            assert w.check()["circuit"] == ok, (w.port, w.window)
