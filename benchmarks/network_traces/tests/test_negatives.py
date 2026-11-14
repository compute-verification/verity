"""The negatives harness (`negatives.py`): each negative names the check that refuses it by its code, and each
advice or Commit tampering is a pure edit.

The advice's tamperings run on I1 v0 (the advice module's classes when they're
importable, a stand-in of their fields until then): each changes exactly what its check reads and leaves the advice
valid under v0's schedule rules (`_check`; the real classes refuse invalid advice on construction). The Commit's runs on
a stand-in of I3 as drafted. The I/O tie's (`io_tie`, when importable) run on a fake upstream's cells and
Commit port of the advice, under the map the verifier computes from it; `frame` also runs on the network certifier's own
committed cells (`warden_rows`, when importable), by the verifier's registered read."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest
from verity.protocols.accounting.certifier.verifier.certificate import Direction

HERE = Path(__file__).resolve().parents[1]
SESSION = bytes(range(32))


def _load(name: str):
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location(f"network_traces_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # its dataclasses look their module up
    spec.loader.exec_module(mod)
    return mod


NG = _load("negatives")


STRATEGIES = ("honest", "late", "extra", "drop", "token")
PROFILES = ("planned", "served")


def test_every_negative_names_a_check_and_a_code_and_the_table_lists_each():
    rows = NG.table().splitlines()
    assert len(rows) == 2 + len(NG.NEGATIVES)
    for name, n in NG.NEGATIVES.items():
        assert n.family in ("io", "advice", "token", "circuit") and n.tampers in ("plan", "certified", "frames",
                                                                                   "advice", "commit")
        assert re.fullmatch(r"[A-Za-z0-9_]+(\.[a-z_-]+)?", n.code), n.code
        assert any(row.startswith(f"| `{name}` |") for row in rows)
    assert {n.family for n in NG.NEGATIVES.values()} == {"io", "advice", "token", "circuit"}
    assert [n for n in NG.NEGATIVES.values() if not n.waits and n.tampers in ("advice", "commit")] == []


def test_a_frame_with_one_byte_changed_no_longer_opens_under_the_records_frames_root():
    """On the certificate's own tree (`certificate.frames_root`, `open_frame`, `verify_frame`): every honest frame opens
    under the root, the changed one doesn't, and no other frame changes."""
    from verity.protocols.accounting.certifier.verifier.certificate import frames_root, open_frame, verify_frame
    from verity.protocols.accounting.certifier.verifier.grid import GridParams

    params = GridParams(buckets=4, allocation=2, frame_bytes=64, bucket_ms=100, queue_capacity=8)
    frames = [bytes([k]) * 64 for k in range(5)]
    root = frames_root(frames, params)
    bad = NG.frame_byte(frames, 2)
    assert [k for k, (a, b) in enumerate(zip(frames, bad)) if a != b] == [2] and bad[2][-1] == 3
    assert all(verify_frame(root, params, k, f, open_frame(frames, params, k)) for k, f in enumerate(frames))
    assert not verify_frame(root, params, 2, bad[2], open_frame(frames, params, 2))
    assert not verify_frame(root, params, 2, bad[2], open_frame(bad, params, 2))


# --- the advice: I1 v0, or a stand-in of its fields without the advice module ------------------------------------------

@dataclasses.dataclass(frozen=True)
class RequestAdvice:
    id: str
    prompt_len: int
    output_len: int
    finish: str


@dataclasses.dataclass(frozen=True)
class Source:
    request: int
    n: int


@dataclasses.dataclass(frozen=True)
class Row:
    request: int
    kind: str
    start: int
    n: int
    cache_hit: int = 0
    preempted: bool = False
    sources: tuple = ()


@dataclasses.dataclass(frozen=True)
class StepAdvice:
    rows: tuple


@dataclasses.dataclass(frozen=True)
class Advice:
    requests: tuple
    steps: tuple
    release: tuple


V = NG.advice_module() or sys.modules[__name__]


def _advice(rows=None, release=(3, 3, 3, 3, 5)):
    """r0: a prompt of 5 in two prefill chunks (3, then 2 beside r1's whole prefill), 3 tokens out, the last at EOS;
    r1: a prompt of 2, 2 tokens out to its cap. Released: the first four outputs in bucket 3, r0's last in 5."""
    rows = rows or [[(0, "prefill", 0, 3)], [(0, "prefill", 3, 2), (1, "prefill", 0, 2)],
                    [(0, "decode", 5, 1), (1, "decode", 2, 1)], [(0, "decode", 6, 1)]]
    return V.Advice((V.RequestAdvice("r0", 5, 3, "eos"), V.RequestAdvice("r1", 2, 2, "length")),
                    tuple(V.StepAdvice(tuple(V.Row(*r) for r in step)) for step in rows), tuple(release))


def _check(a) -> list[tuple[int, str, int]]:
    """The advice's kept outputs (step, request id, index), once it's held to I1 v0's schedule rules (`advice._replay`,
    which the real module also runs on construction): requests in order of first row, a request's first row at its
    cache hit, each later row where the last left off (a resume at its hit), positions within the known tokens, `kind`
    decode exactly when n = 1 past the prompt, a cache hit's sources KV their requests computed by its step, every
    request keeping its `output_len`, and a release bucket a kept output, in order within each request."""
    R = len(a.requests)
    computed, known, kept, first, out = [0] * R, [q.prompt_len for q in a.requests], [0] * R, [], []
    made: list[set[int]] = [set() for _ in range(R)]
    for s, st in enumerate(a.steps):
        assert st.rows and len({r.request for r in st.rows}) == len(st.rows)
        for row in st.rows:
            r, q = row.request, a.requests[row.request]
            assert sum(x.n for x in row.sources) == row.cache_hit
            if r == len(first):
                first.append(s)
                assert not row.preempted and row.start == row.cache_hit
            else:
                assert r < len(first) and row.start == (row.cache_hit if row.preempted else computed[r])
                assert row.preempted or not row.cache_hit
            end = row.start + row.n
            assert row.n >= 1 and end <= known[r]
            assert row.kind == ("decode" if row.n == 1 and row.start >= q.prompt_len else "prefill")
            computed[r] = end
            made[r] |= set(range(row.start, end))
            if end == known[r] and kept[r] < q.output_len:
                out.append((s, q.id, kept[r]))
                kept[r], known[r] = kept[r] + 1, known[r] + 1
        for row in st.rows:
            at = 0
            for x in row.sources:
                assert x.request < len(first) and set(range(at, at + x.n)) <= made[x.request]
                assert x.request != row.request or row.preempted
                at += x.n
    assert len(first) == R and kept == [q.output_len for q in a.requests] and len(a.release) == len(out)
    for rid in {q.id for q in a.requests}:
        mine = [b for (_, i, _), b in zip(out, a.release) if i == rid]
        assert mine == sorted(mine)
    return out


def _served(a) -> dict[str, list[int]]:
    """Each request's positions, in the order its rows schedule them."""
    out: dict[str, list[int]] = {q.id: [] for q in a.requests}
    for st in a.steps:
        for r in st.rows:
            out[a.requests[r.request].id] += list(range(r.start, r.start + r.n))
    return out


def _tokens(a) -> dict[str, int]:
    """Each request's kept outputs by its rows (`_check`)."""
    out = {q.id: 0 for q in a.requests}
    for _, q, _ in _check(a):
        out[q] += 1
    return out


def _shapes(a) -> list[list[tuple]]:
    """What a step's compiled Program reads of the advice: each row's kind, start, n and cache hit, by request id."""
    return [sorted((a.requests[r.request].id, r.kind, r.start, r.n, r.cache_hit) for r in st.rows) for st in a.steps]


def test_the_advice_is_whole():
    a = _advice()
    assert _check(a) == [(1, "r0", 0), (1, "r1", 0), (2, "r0", 1), (2, "r1", 1), (3, "r0", 2)]
    assert _served(a) == {"r0": list(range(7)), "r1": list(range(3))}


def test_every_advice_negative_is_valid_advice_that_round_trips_through_the_codec():
    a = _advice()
    for name, n in NG.NEGATIVES.items():
        if n.tampers == "advice":
            b = n.edit(a)
            assert b != a and _check(b), name
            if V is not sys.modules[__name__]:
                assert V.decode(V.encode(b)) == b, name
    assert a == _advice()


def test_advice_length_keeps_one_output_more_or_fewer_and_stays_whole():
    """One more: a decode row past r0's last token, in a step of its own (r0 keeps its last in the last step), released
    with the output before it; r1's, a decode row beside r0's last. One fewer: r1's last kept sample discarded, its
    row kept. A row past an EOS keeps the sample that one more needs, and goes with the one fewer."""
    a = _advice()
    b = NG.output_length(a)
    assert [q.output_len for q in b.requests] == [4, 2] and b.steps[:4] == a.steps
    assert b.steps[4].rows == (V.Row(0, "decode", 7, 1),) and b.release == (3, 3, 3, 3, 5, 5)
    assert _tokens(b) == {"r0": 4, "r1": 2}
    c = NG.output_length(a, 1, -1)
    assert c.requests[1].output_len == 1 and c.steps == a.steps and c.release == (3, 3, 3, 5)
    assert _tokens(c) == {"r0": 3, "r1": 1}
    d = NG.output_length(a, 1)
    assert d.steps[3].rows == (V.Row(0, "decode", 6, 1), V.Row(1, "decode", 3, 1)) and _tokens(d) == {"r0": 3, "r1": 3}
    assert d.release == (3, 3, 3, 3, 5, 5)
    past = _advice([[(0, "prefill", 0, 3)], [(0, "prefill", 3, 2), (1, "prefill", 0, 2)],
                    [(0, "decode", 5, 1), (1, "decode", 2, 1)], [(0, "decode", 6, 1), (1, "decode", 3, 1)]])
    assert _check(past) == _check(a)
    e = NG.output_length(past, 1)
    assert e.steps == past.steps and _tokens(e) == {"r0": 3, "r1": 3} and e.release == (3, 3, 3, 3, 5, 5)
    f = NG.output_length(past, 1, -1)
    assert f.steps == c.steps and f.release == c.release
    with pytest.raises(NG.NotApplicable):
        NG.output_length(c, 1, -1)
    assert a == _advice()


def test_advice_missing_leaves_out_one_request_and_its_rows_and_keeps_the_others_whole():
    a = _advice()
    for i, gone, release in ((-1, "r1", (3, 3, 5)), (0, "r0", (3, 3))):
        b = NG.missing_request(a, i)
        assert [q.id for q in b.requests] == [q.id for q in a.requests if q.id != gone] and b.release == release
        assert {k: v for k, v in _served(a).items() if k != gone} == _served(b)
        assert _tokens(b) == {q.id: q.output_len for q in b.requests}
    with pytest.raises(NG.NotApplicable, match="reads request 0's KV"):
        NG.missing_request(NG.start_shifted(a), 0)


def test_advice_extra_serves_one_request_more_beside_the_original_and_is_whole():
    a = _advice()
    b = NG.extra_request(a)
    assert [q.id for q in b.requests] == ["r0", "r0~extra", "r1"]
    assert _served(b)["r0~extra"] == _served(a)["r0"] and _tokens(b) == {q.id: q.output_len for q in b.requests}
    assert b.release == (3, 3, 3, 3, 3, 3, 5, 5)
    assert [q.id for q in NG.extra_request(a, 1).requests] == ["r0", "r1", "r1~extra"]
    shifted = NG.start_shifted(a)
    c = NG.extra_request(shifted, 1)
    assert c.steps[1].rows[2].sources == shifted.steps[1].rows[1].sources == (V.Source(0, 1),) and _check(c)


def test_advice_release_moves_one_output_one_bucket_and_keeps_the_order():
    """By default the first kept output (r0's, at step 1), the outputs after it moved up with it."""
    a = _advice()
    b = NG.release_moved(a)
    assert b.release == (4, 4, 4, 4, 5) and b.steps == a.steps and b.requests == a.requests
    assert NG.release_moved(a, -1).release == (3, 3, 3, 3, 6)
    assert NG.release_moved(a, 2, by=3).release == (3, 3, 6, 6, 6)


def test_advice_start_claims_a_cache_hit_with_its_frames_kept_and_changes_one_steps_shape():
    """r0's first row has no KV of another request to read at step 0, so r1's moves: its first position read from r0's
    KV. N2 reads a row's end only, so it accepts the shifted advice (the same outputs); the step's Program changes. One
    position fewer undoes it."""
    a = _advice()
    b = NG.start_shifted(a)
    assert [k for k, (x, y) in enumerate(zip(_shapes(a), _shapes(b))) if x != y] == [1]
    assert b.steps[1].rows[1] == V.Row(1, "prefill", 1, 1, 1, False, (V.Source(0, 1),))
    assert b.requests == a.requests and b.release == a.release and _check(b) == _check(a)
    assert NG.start_shifted(b, 1, by=-1) == a
    with pytest.raises(NG.NotApplicable):
        NG.start_shifted(a, 0)


def test_advice_chunk_serves_the_same_positions_in_other_step_shapes():
    """The circuit negative: every request is served the same positions with the same outputs in the same buckets, so
    the I/O agrees; only the chunks' steps compile to other Programs. An unchunked prefill is split in two."""
    a = _advice()
    b = NG.chunk_split(a)
    assert _served(b) == _served(a) and _check(b) == _check(a) and b.release == a.release
    assert [k for k, (x, y) in enumerate(zip(_shapes(a), _shapes(b))) if x != y] == [0, 1]
    assert (b.steps[0].rows[0].n, b.steps[1].rows[0].start, b.steps[1].rows[0].n) == (2, 2, 3)
    whole = _advice([[(0, "prefill", 0, 5)], [(1, "prefill", 0, 2)], [(0, "decode", 5, 1), (1, "decode", 2, 1)],
                     [(0, "decode", 6, 1)]], (2, 3, 3, 3, 5))
    c = NG.chunk_split(whole)
    assert [st.rows for st in c.steps[:2]] == [(V.Row(0, "prefill", 0, 4),), (V.Row(0, "prefill", 4, 1),)]
    assert c.steps[2:] == whole.steps[1:] and c.release == whole.release and _served(c) == _served(whole)
    one = V.Advice((V.RequestAdvice("x", 1, 1, "eos"),), (V.StepAdvice((V.Row(0, "prefill", 0, 1),)),), (0,))
    with pytest.raises(NG.NotApplicable, match="no prefill row of two tokens"):
        NG.chunk_split(one)


def test_advice_prompt_split_swaps_two_prompts_and_keeps_every_row_and_output_where_it_was():
    """The prompt split on the stand-in: r0's prompt of 5 advised as 2, cut from the front of its two chunks
    (each keeping a token), r1's of 2 as 5, its one chunk longer, each decode moved with its prompt; every row stays in
    its step, so each output is kept where it was and released in its bucket. Under the tests' framing r0's end moves a
    frame earlier and the frames' total stays. Without a pair whose first prompt is the longer, the first whose
    prompts differ. It needs two prompts of different lengths, rows that read no cached KV, and as many tokens as the
    shorter prompt has chunks."""
    a = _advice()
    b = NG.prompt_split(a)
    assert [(q.id, q.prompt_len) for q in b.requests] == [("r0", 2), ("r1", 5)] and b.release == a.release
    assert [[(r.request, r.kind, r.start, r.n) for r in st.rows] for st in b.steps] == [
        [(0, "prefill", 0, 1)], [(0, "prefill", 1, 1), (1, "prefill", 0, 5)],
        [(0, "decode", 2, 1), (1, "decode", 5, 1)], [(0, "decode", 3, 1)]]
    assert _check(b) == _check(a)

    def ends(x):
        return [sum(FRAMING[q.prompt_len - 1] for q in x.requests[:k + 1]) for k in range(len(x.requests))]

    assert ends(a) == [2, 3] and ends(b) == [1, 3]
    back = NG.prompt_split(b)
    assert [q.prompt_len for q in back.requests] == [5, 2] and _check(back) == _check(a) and ends(back) == [2, 3]
    one = V.Advice((V.RequestAdvice("x", 1, 1, "eos"),), (V.StepAdvice((V.Row(0, "prefill", 0, 1),)),), (0,))
    with pytest.raises(NG.NotApplicable, match="no request's prompt length differs"):
        NG.prompt_split(one)
    with pytest.raises(NG.NotApplicable, match="KV"):
        NG.prompt_split(NG.start_shifted(a))
    three = _advice([[(0, "prefill", 0, 2)], [(0, "prefill", 2, 2)], [(0, "prefill", 4, 1), (1, "prefill", 0, 2)],
                     [(0, "decode", 5, 1), (1, "decode", 2, 1)], [(0, "decode", 6, 1)]])
    with pytest.raises(NG.NotApplicable, match="3 chunks"):
        NG.prompt_split(three)


FRAMING = (1, 1, 2, 2, 2)  # the link's request frames by prompt length: r0's prompt of 5 two, r1's of 2 one


def test_the_declared_received_is_the_advices_requests_frames_under_the_links_framing():
    """`received` as a prover declares it beside the advice: the honest run's, less or more the frames of a request the
    advice leaves out or adds (a copy's are its original's), in the first window that receives any."""
    a = _advice()
    assert NG.link_received(a, FRAMING, [3]) == [3] and NG.link_received(a, FRAMING, [0, 3]) == [0, 3]
    assert NG.link_received(NG.extra_request(a), FRAMING, [3]) == [5]
    assert NG.link_received(NG.missing_request(a), FRAMING, [3]) == [2]


def test_the_advice_suite_ties_each_advice_negative_by_the_ingress_tie(tmp_path):
    """`negatives.py advice` on a run's advice, here the stand-in's through I1's own codec (skipped until I1 is
    importable), IoTieRequest's reference on the certified ingress of the advice's id requests at 404-byte frames, whose
    framing is the tests': the honest advice's requests are the cells', so every window is accepted; each advice
    negative is valid advice that round-trips; missing, extra and prompt-split are refused by IoTieRequest's verdict,
    their requests not the cells' (prompt-split's frames honest's); length and release, their requests honest's, are
    the I/O tie's, and start and chunk the step unit's, so the ingress tie accepts them. `advice_run` on its encoding
    at the link's cells: every request's frame received in window 0, honest accepted and every negative as named."""
    if V is sys.modules[__name__]:
        pytest.skip(f"{NG.ADVICE_MODULE} (I1) isn't importable")
    rows = NG.advice_suite(_advice(), T=8, request_frames=FRAMING, received=[3], frame_bytes=404)
    assert rows["honest"] == {"bits": V.total_bits(_advice()), "codes": [], "request_tie": [True], "ok": True}
    made = {k: r for k, r in rows.items() if k != "honest"}
    assert set(made) == {k for k, n in NG.NEGATIVES.items() if n.tampers == "advice"}
    assert all(r["ran"] and r["valid"] and r["ok"] for r in made.values()), made
    tie = NG.REQUEST_TIE
    assert {k: (r["codes"], r["request_tie"]) for k, r in made.items()} == {
        "advice-length": ([], [True]), "advice-release": ([], [True]),
        "advice-missing": ([tie], [False]), "advice-extra": ([tie], [False]),
        "prompt-split": ([tie], [False]), "advice-start": ([], [True]), "advice-chunk": ([], [True])}
    assert made["prompt-split"]["prompt_lens"] == {0: [5, 2], 1: [2, 5]}
    (tmp_path / "advice.bin").write_bytes(V.encode(_advice()))
    doc = NG.advice_run(tmp_path / "advice.bin", T=8, R=8)
    assert doc["frame_bytes"] == NG.FRAME_BYTES and doc["received"] == [2] and doc["negatives"]["honest"]["ok"]
    assert all(r["ran"] and r["ok"] for k, r in doc["negatives"].items() if k != "honest"), doc


def test_the_ingress_tie_reads_each_requests_record_by_its_user_and_states_the_rest(tmp_path):
    """serve-openai's rows by user (by http id without one) where they carry the prompt; each advice request's id
    request is its record's (its prompt ids, user and what it brought, seed 0 when it brought none), and one without a
    record a prompt of its length under its own id; a record whose prompt isn't the advice's length can't be tied."""
    from verity.protocols.accounting.certifier.verifier.id_form import id_request

    brought = {"max_tokens": 3, "temperature": 0.5, "top_p": 0.9}
    (tmp_path / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in [
        {"http_id": "cmpl-0", "user": "r0", "prompt_ids": [1, 2, 3, 4, 5], "streamed_ids": [7, 8, 9],
         "ingress": {"brought": brought}},
        {"http_id": "cmpl-1", "user": None, "prompt_ids": [6, 6]},
        {"request": "q2", "user": "r1", "output_ids": [21, 22]}]))
    records = NG.read_requests(tmp_path / "requests.jsonl")
    assert sorted(records) == ["cmpl-1", "r0"]
    a = _advice()
    assert NG.id_requests(a, records, 404, "m") == [
        id_request(model="m", user="r0", prompt=[1, 2, 3, 4, 5], frame_bytes=404, seed=0, offset=0, **brought),
        id_request(model="m", user="r1", prompt=[0, 0], max_tokens=2, frame_bytes=404, offset=0)]
    with pytest.raises(NG.NotApplicable, match="2-token prompt, the advice 5"):
        NG.id_requests(a, {"r0": records["cmpl-1"]}, 404)
    assert NG.link_framing(404, 5) == FRAMING


def test_phase_6s_split_of_339_and_16_token_prompts_is_refused_by_the_ingress_tie_alone():
    """The prompt split at the link's 4096-byte cells: prompts of 339 and 16 tokens served (two request cells,
    then one; 338 is one cell at the relay's cut), advised as 16 and 339. The cells' total is honest's and
    the first request's end a cell earlier, so no count of the link's frames refuses it; IoTieRequest's reference on the
    warden's ingress does."""
    if V is sys.modules[__name__]:
        pytest.skip(f"{NG.ADVICE_MODULE} (I1) isn't importable")
    a = V.Advice((V.RequestAdvice("u0", 339, 2, "length"), V.RequestAdvice("u1", 16, 1, "length")),
                 (V.StepAdvice((V.Row(0, "prefill", 0, 339),)), V.StepAdvice((V.Row(0, "decode", 339, 1),
                                                                             V.Row(1, "prefill", 0, 16)))),
                 (3, 3, 3))
    framing = NG.link_framing(NG.FRAME_BYTES, 339)
    assert (framing[15], framing[337], framing[338]) == (1, 1, 2)
    rows = NG.advice_suite(a, T=8, request_frames=framing, received=[3], frame_bytes=NG.FRAME_BYTES)
    split = rows["prompt-split"]
    assert rows["honest"]["ok"] and rows["honest"]["request_tie"] == [True]
    assert split["prompt_lens"] == {0: [339, 16], 1: [16, 339]}
    assert split["codes"] == [NG.REQUEST_TIE] and split["request_tie"] == [False] and split["ok"]


# --- the I/O tie on a fake upstream (`io_tie`) -------------------------------------------------------------------------

IDS = {"r0": [11, 12, 13], "r1": [21, 22]}


def test_the_tie_reads_each_requests_sampled_ids_by_its_advice_id(tmp_path):
    (tmp_path / "requests.jsonl").write_text(
        "".join(json.dumps({"request": f"q{k}", "user": u, "output_ids": x}) + "\n" for k, (u, x) in enumerate(IDS.items())))
    assert NG.read_ids(tmp_path / "requests.jsonl") == IDS
    (tmp_path / "served.jsonl").write_text("".join(json.dumps({"http_id": f"cmpl-{k}", "user": u, "streamed_ids": x})
                                                   + "\n" for k, (u, x) in enumerate([(None, [7]), *IDS.items()])))
    assert NG.read_ids(tmp_path / "served.jsonl") == {"cmpl-0": [7]} | IDS


def test_the_tie_reads_the_requests_whose_clients_aborted_them(tmp_path):
    """serve-openai writes a request its client aborted with ``finish`` null (or of reason ``abort``), an ``output_ids``
    row with ``finish_reason`` ``abort``; a row with another finish, or with no finish field, isn't one."""
    rows = [{"http_id": "cmpl-0", "user": "a", "streamed_ids": [1], "finish": None},
            {"http_id": "cmpl-1", "user": None, "streamed_ids": [1], "finish": {"reason": "abort"}},
            {"http_id": "cmpl-2", "user": "c", "streamed_ids": [1], "finish": {"reason": "length", "stop_reason": None}},
            {"http_id": "cmpl-3", "user": "d", "streamed_ids": [1]},
            {"request": "q0", "user": "e", "output_ids": [1], "finish_reason": "abort"},
            {"request": "q1", "user": "f", "output_ids": [1], "finish_reason": "stop"}]
    (tmp_path / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert NG.read_aborted(tmp_path / "requests.jsonl") == {"a", "cmpl-1", "e"}


def test_the_ties_view_of_the_advice_is_each_kept_outputs_step_row_request_index_bucket_and_id():
    """Each kept output as advice v0 has it (step, batch row, request, index), its release bucket and its sampled id, in
    the advice's order; an output past its request's ids (a request the run never served) has the id 0."""
    IT = pytest.importorskip(NG.IO_TIE_MODULE)
    a = _advice()
    assert NG.tie_outputs(IT, a, IDS) == ([IT.Output(1, 0, 0, 0), IT.Output(1, 1, 1, 0), IT.Output(2, 0, 0, 1),
                                           IT.Output(2, 1, 1, 1), IT.Output(3, 0, 0, 2)], [3, 3, 3, 3, 5],
                                          [11, 21, 12, 22, 13])
    outputs, _, ids = NG.tie_outputs(IT, NG.extra_request(a), IDS)
    assert [(o, x) for o, x in zip(outputs, ids) if o.request == 1] == [
        (IT.Output(1, 1, 1, 0), 0), (IT.Output(2, 1, 1, 1), 0), (IT.Output(3, 1, 1, 2), 0)]
    _, _, ids = NG.tie_outputs(IT, NG.output_length(a), IDS)
    assert ids[-1] == 0 and len(ids) == 6


def test_the_io_tie_refuses_each_negative_it_reads_by_its_part():
    """`io_tie`'s reference (`io_tie.reference_parts`, skipped until it's importable) on a fake upstream's cells and
    Commit port of the advice served through a network certifier of r = 2 slots a connection (r0's last output released
    at bucket 4, the network certifier's, not 5), one shape for every advice: the honest window passes; the changed cell
    under the kept root fails the cells' root, the changed port id under the Commit's kept root the port's; a certified
    cell of another id, a dropped cell, and the map of a length or missing advice negative fail the content; an extra
    request, which no connection carries, and a release the network certifier didn't make fail the map; start and chunk,
    whose outputs and buckets are honest's, pass, leaving them to the step proof block. Two lanes of r = 2 slots hold r0
    and r1 live together; one lane can't."""
    pytest.importorskip(NG.IO_TIE_MODULE)
    rows = NG.tie_suite(_advice(), IDS, T=8)
    assert rows["honest"]["ok"] and rows["honest"]["shape"] == {"B": 256, "T": 8, "R": 4, "P": 3, "S": 3, "DS": 3,
                                                                 "N": 32, "M": 16}
    assert set(rows) == {"honest", *NG.TIE_PARTS}
    assert all(r["ran"] and r["ok"] and "moved" not in r for k, r in rows.items() if k != "honest"), rows
    want = {k: [f"{NG.TIE}.{p}"] if p else [] for k, p in NG.TIE_PARTS.items()}
    want["frame"] = [f"{NG.TIE}.cells_root", f"{NG.TIE}.content"]
    want["wrong-token"] = [f"{NG.TIE}.content", f"{NG.TIE}.port"]
    assert {k: r["codes"] for k, r in rows.items() if k != "honest"} == want
    with pytest.raises(NG.NotApplicable, match="more than R / r = 1 lanes"):
        NG.tie_suite(_advice(), IDS, T=8, R=2)


def test_a_chunk_split_that_moves_outputs_to_later_steps_is_refused_by_the_ties_content():
    """With no chunked prefill, chunk splits a prefill row in two, so every later step moves one on: the map the
    verifier computes from that advice names port entries of other steps than the Commit sampled each output in, and the
    content refuses it; a map the verifier can't compute (a request no connection carries) refuses every window, by
    `map`, and an advice the network certifier can't serve in R slots (two connections live in one lane) has no tie at
    all."""
    pytest.importorskip(NG.IO_TIE_MODULE)
    a = _advice([[(0, "prefill", 0, 5), (1, "prefill", 0, 2)], [(0, "decode", 5, 1), (1, "decode", 2, 1)],
                 [(0, "decode", 6, 1)]])
    rows = NG.tie_suite(a, IDS, T=8)
    assert rows["honest"]["ok"] and rows["advice-start"]["ran"] and rows["advice-start"]["codes"] == []
    assert rows["advice-chunk"] | {"failing": None} == {"code": f"{NG.TIE}.content", "moved": True, "ran": True,
                                                       "failing": None, "codes": [f"{NG.TIE}.content"], "ok": True}
    assert rows["advice-extra"]["codes"] == [f"{NG.TIE}.map"] and "permutation" in rows["advice-extra"]["why"]
    assert rows["advice-extra"]["ok"]
    with pytest.raises(NG.NotApplicable, match="2 live requests, more than R / r = 1 lanes"):
        NG.tie_suite(a, IDS, T=8, R=2)


# --- the Commit record: a stand-in of I3 as drafted --------------------------------------------------------------------

def test_wrong_token_changes_one_committed_sampled_id_and_keeps_the_registered_root():
    commit = {"root": "ab" * 64, "steps": [{"rows": [{"user": "r0", "positions": [5]}, {"user": "r1", "positions": [2]}],
                                            "c00.input/ids": [11, 21], NG.SAMPLED: [12, 22]}]}
    out = NG.wrong_token(commit, step=0, row=1)
    assert out["steps"][0][NG.SAMPLED] == [12, 23] and out["root"] == commit["root"]
    assert commit["steps"][0][NG.SAMPLED] == [12, 22]
    assert NG.wrong_token(commit, row=0, vocab=13)["steps"][0][NG.SAMPLED] == [0, 22]


def test_a_changed_byte_of_a_wardens_committed_cell_doesnt_read_under_its_root():
    """On the relay's bucket rows: connection 1's data cell and pad in bucket 0's row, bucket 1 empty. The
    row reads under the signed root at its position; with one byte of the data cell changed it doesn't, and neither
    does the honest row at another position."""
    WR = pytest.importorskip(NG.WARDEN_ROWS_MODULE)
    from verity.protocols.accounting.certifier.verifier.link_frame import DATA, HEADER, PAD
    from verity.protocols.accounting.certifier.verifier.grid import COMPLETE, BlockGrid
    from verity.protocols.accounting.certifier.verifier.warden import WardenParams, cell

    params = WardenParams(cell_bytes=64, slots=2, lanes=2, window=2)
    B, S = params.cell_bytes, params.grid.allocation
    grid = ((cell(DATA, 1, b"abc", B), cell(PAD, 1, b"", B)) + (None,) * (S - 2), (None,) * S)
    salts = [bytes([k + 1]) * WR.SALT_BYTES for k in range(params.window)]
    cells = WR.Cells.of("tie", Direction.EGRESS, BlockGrid("tie", 0, grid, COMPLETE), params, SESSION, salts, bytes(64))
    assert cells.wiring.connections == ((1, 0, 2),) and len(cells.rows[0]) == S * B
    row = cells.rows[0]
    bad = bytearray(row)
    bad[HEADER.size] ^= 1
    assert NG.cell_reads(WR, cells, 0, row)
    assert not NG.cell_reads(WR, cells, 0, bytes(bad))
    assert not NG.cell_reads(WR, cells, 1, row)


def test_frame_on_a_warden_run_reads_its_data_cells_row_and_refuses_the_changed_byte(tmp_path):
    """`warden_frame` on a warden-serve run driven tick by tick (`test_warden_serve._run`): its egress registration
    (v2) is read under the network certifier's key and the schedule, its bucket rows and signed wiring are committed
    under the run's salts to the registered head, and its first data cell, at its bucket's row and slot, reads honest
    and not with a payload byte changed; a run that kept no salts is not applicable."""
    pytest.importorskip(NG.WARDEN_ROWS_MODULE)
    sys.path.insert(0, str(HERE / "tests"))
    import test_warden_serve as tws

    run, key = tmp_path / "run", tmp_path / "key"
    run.mkdir()
    tws._run(run, key, [tws.ws.http_request({"prompt": "x" * 10}, "h:1")], [tws._response([" word"] * 8)])
    got = NG.warden_frame(run)
    p = tws.PARAMS
    assert got["ok"] and got["honest_reads"] and not got["tampered_reads"]
    assert got["rows"] == p.window and got["row_bytes"] == p.grid.allocation * p.cell_bytes
    assert got["cell_bytes"] == p.cell_bytes and 0 <= got["slot"] < p.grid.allocation
    assert NG.warden_frame(run, at=-1)["ok"]
    (run / "cells.salts").unlink()
    with pytest.raises(NG.NotApplicable, match="cells.salts"):
        NG.warden_frame(run)
