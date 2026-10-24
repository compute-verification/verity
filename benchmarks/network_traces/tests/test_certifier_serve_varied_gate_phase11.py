"""Phase 11 red team of NEG's varied gate (#3709): can a server use the forwarded `user`, or the varied plan's own
fields, to move anything the certifier signs?

The certifier (`active.Proxy`) signs records of the frames on the link: the client's ingress requests and the shim's
egress responses. The endpoint is beyond the shim, which forwards it the request body, `user` and all (`varied_body`),
and frames the stream it gets back. These cases hold the boundary the gate rests on:

- the forwarded `user` and every varied field (`model`, `prompt`, `max_tokens`, `temperature`, `top_p`, `seed`) are the
  client's, bound canonically in the certified ingress request (`id_request_fields` reconstructs them, and nothing in
  an id request is free), which the endpoint never produces;
- the egress carries only (ordinal, token id) per id event (`id_event`): there is no `user` on the egress for a server
  to smuggle or re-attribute by, and a response is read by its request's ingress ordinal, which the proxy assigns, not
  by anything the endpoint returns;
- so the server's only lever is the egress content it streams back, which the certified I/O's finish rule gates for
  shape and the I/O tie (#3701) binds to the committed sampled tokens; the forwarded `user` buys it no more.
"""
from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import pytest
from verity.core.protocols.certifier import active

import test_certifier_serve as cst

cs = cst.cs
ar = cst.ar
B = cs.FRAME_BYTES
EOS, VARIED_TOKENS = cst.EOS, cst.VARIED_TOKENS


# The ingress binding: the forwarded `user` and the varied fields are the client's, pinned canonically.

def test_the_forwarded_body_and_the_certified_request_carry_the_same_client_user_and_fields():
    """The shim forwards the endpoint `varied_body` with the request's `user`; the certified ingress request is the id
    request of the same fields and the same `user`, and `id_request_fields` recovers them exactly. The two are one
    client-set bundle: the endpoint receives it, it does not author it."""
    q = {"prompt": [1, 2, 3, 4, 5], "max_tokens": 5, "seed": 123456789}
    fields = cs.varied_fields("a-model", q)
    user = "varied.00042"
    forwarded = cs.varied_body("a-model", q) | {"user": user}
    raw = active.id_request(**fields, user=user, frame_bytes=B)
    assert json.loads(raw.partition(b"\r\n\r\n")[2]) == forwarded
    assert active.id_request_fields(raw, B) == fields | {"user": user}


@pytest.mark.parametrize("find,replace", [
    (b'"max_tokens":', b'"max_tokens": '),  # a space the canonical form does not have
    (b'"seed":', b'"seed" :'),              # whitespace before a colon
    (b'"top_p":', b'"top_p" :'),
    (b'"stream":true', b'"stream":fals'),   # a changed literal, length kept
])
def test_nothing_in_an_id_request_is_free_so_a_server_cannot_move_a_field_the_certifier_signs(find, replace):
    """Every field is at a place the prompt length and the frame size fix, with no free byte: an id request with any
    byte the canonical form does not have is not an id request (`id_request_fields` reconstructs the request from the
    fields and returns None when it differs), so the certified I/O refuses it ("a request that isn't an id request"). A
    server could not slip a moved field past the certifier even if it reached the ingress, which it does not."""
    raw = active.id_request(model="a-model", user="varied.00007", prompt=[1, 2, 3], max_tokens=4, seed=9, frame_bytes=B)
    assert active.id_request_fields(raw, B) is not None and find in raw
    moved = raw.replace(find, replace, 1)
    assert active.id_request_fields(moved, B) is None


# The egress has no `user` channel: an id event is only an ordinal and a token.

def test_an_egress_id_event_is_only_an_ordinal_and_a_token_with_no_user_channel():
    """The server's egress frames are id events: a request's ingress ordinal and one sampled token's id, and nothing
    else is free (`ID_EVENT_BYTES` long whatever the two). There is no `user` on the egress for a server to smuggle or
    to re-attribute a response by."""
    ev = active.id_event(7, 42)
    assert active.id_event_data(ev) == (7, 42)
    assert len(ev) == active.ID_EVENT_BYTES and b"user" not in ev and b"varied" not in ev
    assert active.id_event_data(ev[:-1] + b"X") is None  # any other byte is not an id event
    assert active.id_event_data(ev + b" ") is None


# Real records: the endpoint keyed on the forwarded `user` moves only its egress content.

@pytest.fixture(scope="module")
def honest(tmp_path_factory):
    """One honest varied run against the stub endpoint: its plan, calibration and the records the certified I/O reads
    (`requests.records`, `responses.records`), on an injected clock."""
    endpoint = cst._Varied()
    args = argparse.Namespace(model=None, api_key_env=None, traffic="varied", requests=12, rate=2.0,
                              prompt_range="16:512", tokens=VARIED_TOKENS, seed=5, probes=4, eos_ids=str(EOS))
    tmp, _probed, report, plan, runs = cst._varied_runs(
        tmp_path_factory.mktemp("varied-gate"), endpoint,
        lambda url: cs.plan_of(args, url, lambda: 0.0, cst._no_sleep), ["honest"])
    return tmp, report, plan, runs


def _records(run, setup):
    e, i = setup.egress_params, setup.ingress_params
    ins = [ar.unpack_record(b, i.frame_bytes) for b in ar.read_blobs(run / "requests.records")]
    outs = [ar.unpack_record(b, e.frame_bytes) for b in ar.read_blobs(run / "responses.records")]
    return ins, outs


def _retoken(outs, frame_bytes, change):
    """A copy of the egress records with each id event's token id passed through `change(ordinal, token)`; padding and
    non-id-event frames are left as they are."""
    new = []
    for rec in outs:
        grid = []
        for row in rec.grid:
            built = []
            for slot in row:
                if slot is not None:
                    kind, st, payload = active.decode_frame(slot)
                    got = active.id_event_data(payload) if kind == active.DATA else None
                    if got is not None and (nt := change(*got)) != got[1]:
                        slot = active.encode_frame(active.DATA, st, active.id_event(got[0], nt), frame_bytes)
                built.append(slot)
            grid.append(built)
        new.append(SimpleNamespace(window=rec.window, grid=grid))
    return new


def test_the_honest_run_is_certified_and_its_io_is_the_clients_requests_read_from_the_frames(honest):
    """Baseline: every link-window's records are certified under the certifier's key and session, the audit and the salt
    accept, and the certified I/O's requests are the plan's, user by user — read from the signed ingress frames, which
    the endpoint never authored."""
    tmp, _, plan, runs = honest
    check = runs["honest"]["check"]
    assert check["verdict"] == "accepted", cs.table(check)
    assert check["certified_accepted"] and check["audit_accepted"] and check["salt_accepted"]
    io = check["io"]  # `check` pops the per-request records into io.json; what stays is the summary
    assert io["accepted"] and io["requests_are_the_plans"] and io["ids_are_the_shims"]
    assert io["violations"] == [] and io["requests"] == len(plan["requests"])
    records = {r["user"]: r for r in json.loads((tmp / "honest" / "io.json").read_text())["requests"]}
    assert set(records) == {q["session"] for q in plan["requests"]}


def test_a_server_keyed_on_the_user_that_keeps_the_finish_rule_moves_only_egress_content(honest):
    """A server that, from the forwarded `user`, serves other tokens than the model's (every non-EOS id changed, EOS
    kept so each response's length and finish stand) moves only the egress content: the certified I/O still accepts,
    the certified requests are still the client's (their users and fields unchanged), and only the ids differ. `check`
    does not read the ids against the committed sampled ones; the I/O tie does. The forwarded `user` buys the server no
    more than the content it already streams."""
    tmp, report, plan, _ = honest
    setup = ar.Setup(report)
    ins, outs = _records(tmp / "honest", setup)
    honest_io = cs.certified_io(plan, setup, ins, outs)
    assert honest_io["accepted"]
    tampered = _retoken(outs, setup.egress_params.frame_bytes,
                        lambda o, t: t if t == EOS else (t + 1 if t + 1 != EOS else t + 2))
    moved = cs.certified_io(plan, setup, ins, tampered)
    assert moved["accepted"] and moved["violations"] == []
    by_user = {r["user"]: r for r in moved["records"]}
    for r in honest_io["records"]:
        m = by_user[r["user"]]
        assert {k: m[k] for k in ("user", "stream", "prompt", "max_tokens", "seed")} == \
            {k: r[k] for k in ("user", "stream", "prompt", "max_tokens", "seed")}
        assert m["finish"] == r["finish"] and len(m["output"]) == len(r["output"])
    changed = [r for r in honest_io["records"] if any(t != EOS for t in r["output"])]
    assert changed and any(by_user[r["user"]]["output"] != r["output"] for r in changed)


def test_a_server_that_drops_an_eos_is_refused_by_the_finish_rule_whatever_the_user(honest):
    """A server that drops a response's EOS (its last token changed to another id) so the response ends short of its
    `max_tokens` with no EOS is refused by the certified I/O's finish rule, the response its own. The forwarded `user`
    grants no escape from the finish rule."""
    tmp, report, plan, _ = honest
    setup = ar.Setup(report)
    ins, outs = _records(tmp / "honest", setup)
    honest_io = cs.certified_io(plan, setup, ins, outs)
    target = next(r for r in honest_io["records"]
                  if r["output"] and r["output"][-1] == EOS and len(r["output"]) < r["max_tokens"])
    stream = target["stream"]
    dropped = _retoken(outs, setup.egress_params.frame_bytes,
                       lambda o, t, s=stream: (EOS + 1) if (o == s and t == EOS) else t)
    refused = cs.certified_io(plan, setup, ins, dropped)
    assert not refused["accepted"]
    assert [stream, "short of max_tokens without EOS"] in refused["violations"]
