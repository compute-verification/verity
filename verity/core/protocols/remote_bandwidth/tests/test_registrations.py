"""One live registration a node, the red team's KB-4: each
registration draws a fresh codec seed, so a prover that registered R times and kept the key that suits it would choose
log₂ R bits of it, a retry that redraws.  The verifier's `Registry` refuses a second registration while one is live,
and a registration ends only when the verifier closes it, counting its sessions, a failed or unended one as failed."""

from __future__ import annotations

import json

import pytest

from test_appends import N, NODE, P512, SOURCE, Run, payload, salts
from verity.core.protocols import registration as log
from verity.core.protocols.remote_bandwidth import audit
from verity.core.protocols.remote_bandwidth.codec import BLOCK_BYTES


def registered_in(registry: audit.Registry, nonce: bytes = bytes(32), node: bytes = NODE) -> audit.Store:
    payloads = [payload(b"weights %d" % j) for j in range(N)]
    ys = salts(b"payload salts", N)
    root = audit.payload_tree(node, payloads, ys).commitment.root
    return audit.Store(registry.register(SOURCE, node, N, root, P512, nonce=nonce), payloads, ys)


def accepted(store: audit.Store, registry: audit.Registry) -> Run:
    r = Run(store, windows=1, registry=registry)
    assert r.run() is None
    r.boundary()
    assert r.finish() == [audit.Verdict(0, True)]
    return r


def rejected(store: audit.Store, registry: audit.Registry) -> Run:
    """A session whose second pass names the first's step: rejected with `step`."""
    r = Run(store, windows=1, registry=registry)
    assert r.run() is None
    assert r.run(step=0) == audit.Verdict(0, False, "step")
    return r


def unended(store: audit.Store, registry: audit.Registry) -> Run:
    r = Run(store, windows=1, registry=registry)
    assert r.run() is None
    return r


def test_a_second_registration_of_a_live_node_is_refused_and_draws_nothing(monkeypatch) -> None:
    draws, draw = [], audit.register

    def counted(*args, **kw):
        draws.append(args[1])
        return draw(*args, **kw)

    monkeypatch.setattr(audit, "register", counted)
    registry = audit.Registry()
    st = registered_in(registry)
    assert draws == [NODE] and registry.live(NODE) == st.registration.digest and registry.drawn(NODE) == 1
    for nonce in (bytes(32), bytes(range(32))):
        with pytest.raises(audit.RegistryError, match="live registration"):
            registered_in(registry, nonce)
    assert draws == [NODE] and registry.refused == {NODE: 2} and registry.drawn(NODE) == 1
    assert registry.live(NODE) == st.registration.digest
    other = registered_in(registry, node=b"another node")
    assert draws == [NODE, b"another node"] and registry.live(b"another node") == other.registration.digest


def test_a_registration_ends_only_when_closed_with_every_session_counted() -> None:
    """Three sessions over one registration, accepted, rejected and unended: the registration stays live through the
    rejection, so no fresh draw can replace it, and its close counts all three, two of them failed.  Only then does the
    node register again, under a fresh codec seed, R = 2."""
    registry = audit.Registry()
    st = registered_in(registry)
    runs = [accepted(st, registry), rejected(st, registry), unended(st, registry)]
    assert [r.s.commitment.epoch for r in runs] == [0, 1, 2]
    assert [r.s.outcome for r in runs] == ["accepted", "step", None]
    with pytest.raises(audit.RegistryError):
        registered_in(registry, bytes(range(32)))

    closed = registry.close(NODE)
    assert closed == audit.Closed(NODE, st.registration.digest, ("accepted", "step", audit.UNENDED))
    assert closed.failed == 2 and registry.live(NODE) is None and registry.closed == [closed]
    with pytest.raises(audit.RegistryError, match="live registration"):
        Run(st, windows=1, registry=registry)
    with pytest.raises(audit.RegistryError, match="no live registration"):
        registry.close(NODE)

    again = registered_in(registry, bytes(range(32)))
    assert again.registration.codec_seed != st.registration.codec_seed and registry.drawn(NODE) == 2
    assert registry.live(NODE) == again.registration.digest
    assert accepted(again, registry).s.commitment.epoch == 0


def test_a_session_runs_only_over_its_node_s_live_registration_drawn_by_the_registry() -> None:
    registry = audit.Registry()
    st = registered_in(registry)
    elsewhere = registered_in(audit.Registry(), bytes(range(32)))
    with pytest.raises(audit.RegistryError):
        Run(elsewhere, windows=1, registry=registry)
    r = Run(st, windows=1, registry=registry)
    p = r.p
    with pytest.raises(audit.RegistryError, match="next session is 1, not 0"):
        registry.session(p, st.registration, None, not_before_ns=0, index=0)
    with pytest.raises(ValueError):
        registry.session(audit.Params(N + 1, p.deadline_ns, 1, codec=P512), st.registration, None, not_before_ns=0)
    assert registry.session(p, st.registration, None, not_before_ns=0, index=1).commitment.epoch == 1
    assert registry.close(NODE).sessions == (audit.UNENDED, audit.UNENDED)


def test_the_registry_outlives_the_verifier_s_process_without_its_secret() -> None:
    """Read back, a live registration still refuses a second, runs no session (the verifier that drew it is gone),
    and closes with its sessions as they stood; the record holds no secret and no codec seed."""
    registry = audit.Registry()
    st = registered_in(registry)
    accepted(st, registry)
    registry.close(NODE)
    st = registered_in(registry, bytes(range(32)))
    accepted(st, registry)
    unended(st, registry)
    with pytest.raises(audit.RegistryError):
        registered_in(registry, bytes(range(1, 33)))
    text = json.dumps(registry.to_json())
    assert SOURCE.hex() not in text and st.registration.codec_seed.hex() not in text

    back = audit.Registry.from_json(json.loads(text))
    assert back.to_json() == registry.to_json()
    assert back.live(NODE) == st.registration.digest and back.drawn(NODE) == 2 and back.refused == {NODE: 1}
    with pytest.raises(audit.RegistryError):
        registered_in(back, bytes(range(2, 34)))
    with pytest.raises(audit.RegistryError):
        Run(st, windows=1, registry=back)
    assert back.close(NODE) == audit.Closed(NODE, st.registration.digest, ("accepted", audit.UNENDED))
    assert [c.sessions for c in back.closed] == [("accepted",), ("accepted", audit.UNENDED)]

    good = registry.to_json()
    for bad in ({**good, "schema": "other"}, {**good, "closed": [{**good["closed"][0], "registration": "00"}]},
                {**good, "live": {NODE.hex(): {"registration": st.registration.digest.hex(), "sessions": [""]}}},
                {**good, "refused": {NODE.hex(): 0}}):
        with pytest.raises(ValueError):
            audit.Registry.from_json(bad)


def logged(node: bytes, root: bytes, item: dict | None = None, store_bytes: int = N * BLOCK_BYTES,
           device: bool = False) -> log.View:
    """The view of a remote site's registration log holding the node's store, `store`: a memory leaf in the node (or
    in a device of it), and θ's value for its payload (`audit.payload_item`, or `item`)."""
    n, card = node.decode(), log.Card("a card", 1, (1, 1, 1, 1, 1), 1, 1, 1 << 40)
    parent = [log.Loc("gpu", "device", n, card=card.digest())] if device else []
    site = log.Hardware(cards=(card,), locs=(log.Loc("site", "site"), log.Loc(n, "host", "site"), *parent,
                                             log.Loc("store", "memory", "gpu" if device else n, tier="hbm" if device else
                                                     "dram", bytes=store_bytes)))
    theta = log.Information((audit.payload_item("store", node, N, root) if item is None else item,))
    return log.check(log.Auditor(sheets=(card,)), "remote", log.chain("remote", [site, theta])).view()


def test_a_store_drawn_from_the_registration_log_is_the_registration_of_its_node_size_and_payload() -> None:
    """`register_stored` reads the node, N and the payload root from the log, and the registration it draws is the one
    `register` draws of them, byte for byte, so its digest, the root a session's first pass follows, is unchanged.  A
    log that doesn't register the store so draws nothing: no such leaf, a leaf not whole blocks, or θ's value for it of
    other leaves, words or node."""
    root = audit.payload_tree(NODE, [payload(b"weights %d" % j) for j in range(N)], salts(b"y", N)).commitment.root
    registry = audit.Registry()
    reg = registry.register_stored(SOURCE, logged(NODE, root), "store", P512, nonce=bytes(32))
    assert reg == audit.register(SOURCE, NODE, N, root, P512, nonce=bytes(32)) and registry.live(NODE) == reg.digest
    assert audit.stored(logged(NODE, root, device=True), "store") == (NODE, N, root)
    good = audit.payload_item("store", NODE, N, root)
    for view, store in ((logged(NODE, root), "elsewhere"), (logged(NODE, root), "site"),
                        (logged(NODE, root, store_bytes=N * BLOCK_BYTES + 2), "store"),
                        (logged(NODE, root, good | {"leaves": N + 1}), "store"),
                        (logged(NODE, root, good | {"words": BLOCK_BYTES // 4, "word_bits": 32}), "store"),
                        (logged(NODE, root, audit.payload_item("store", b"another node", N, root)), "store")):
        with pytest.raises(ValueError):
            audit.Registry().register_stored(SOURCE, view, store, P512, nonce=bytes(32))
