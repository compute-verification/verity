"""drive.toml's `signed_verdicts` in the verified queue's loop (`research.vqueue_loop`), on the loop's throwaway origin with the
labels in a store, read as the live loop reads them (`StoreLabels`): on "shadow" a mark or verdict that doesn't authenticate
counts and is logged once, and with the flag on it doesn't count.  No network, no clock: the issuer's keys are seeded RSA keys
and every time is given."""
from __future__ import annotations

import json

import pytest

import research.queue as Q
from research import labelauth
from research.executor import GRAPH_V2
from research.queue import DRIVE_FLAGS, StoreLabels, label_target
from research.store import vocab
from test_labelauth import HOLDER, OTHER, READER, T, TS, _jwks, _mint
from test_vqueue_loop import Env, _git, env, policy  # noqa: F401 -- env is a fixture

#: ci staffs the readers, and lane-1, whose holder marks its PR, owns #1
REGISTRY = {"handles": [{"handle": "ci", "holder": HOLDER, "staffs": ["review-*"]}, {"handle": "lane-1", "holder": OTHER}]}
FLAG = {"off": "false", "shadow": '"shadow"', "on": "true"}
NO_DELEGATION = (f"its token is {READER!r}'s, which doesn't hold review-36 in main's registry, and it carries no delegation")


def _main(env: Env, mode: str, registry: str = json.dumps(REGISTRY)) -> str:
    """`main` moved to a commit whose drive.toml sets `signed_verdicts` to ``mode`` with the low-risk lane off, so every PR
    needs a land verdict, and whose registry is ``registry``; its check listed with no task.  The origin's store gets its
    remote, which `StoreLabels` and a signer's read of its delegation pull from."""
    env.o.with_store_remote()
    main = env.o.change({DRIVE_FLAGS: f"signed_verdicts = {FLAG[mode]}\nlowrisk_lane = false\n", str(Q.REGISTRY): registry})
    _git(env.o.origin, "update-ref", "refs/heads/main", main)
    env.graph2.list_graph(f"ready:{main}", {"schema": GRAPH_V2, "commit": main, "tasks": []})
    return main


def _loop(env: Env, monkeypatch, keys=lambda cache, log=print: labelauth.jwks(_jwks())):
    """A loop reading its labels from the origin's store, the issuer's keys ``keys``."""
    monkeypatch.setattr(labelauth, "issuer_keys", keys)
    loop = env.loop(policy)
    loop.world.labels_at = lambda: StoreLabels(env.o.store, log=env.out.append)
    return loop


def _marked(env: Env, n: int, head: str) -> str:
    """#``n``'s ready mark at ``head``, signed by its owner's holder."""
    target = label_target(n, head)
    labelauth.Signer(env.o.store, _mint(OTHER), now=lambda: TS).add_label(target, vocab.READY, True, "lane-1")
    return target


def _unauthenticated(recs: list[dict]) -> list[tuple]:
    return [(r["number"], r["heads"], r["label"], r["by"], r["written"], r["why"]) for r in recs if r["action"] == "unauthenticated"]


@pytest.mark.parametrize("mode", ["shadow", "on", "off"])
def test_a_readers_verdict_no_one_registered_counts_in_shadow_logged_once_and_not_with_the_flag_on(env, monkeypatch, mode):
    """A reader signs its land verdict with its own identity token, but ci's holder never registered its agent as
    `review-36`.  On "shadow" the PR is built on that verdict as with the flag off, and the loop logs the verdict once with
    why it wouldn't count; with the flag on it doesn't count; the authenticated mark is never logged."""
    _main(env, mode)
    head = env.put(1, {"a.txt": "1\n"})
    target = _marked(env, 1, head)
    labelauth.Signer(env.o.store, _mint(READER), now=lambda: TS).add_label(target, vocab.REVIEWED, "land", "review-36")
    loop = _loop(env, monkeypatch)
    recs = loop.tick()
    assert [r["members"] for r in recs if r["action"] == "build"] == ([] if mode == "on" else [[1]])
    assert _unauthenticated(recs) == ([(1, [head], vocab.REVIEWED, "review-36", TS, NO_DELEGATION)] if mode == "shadow" else [])
    assert (loop.world.left.get(1) or ("", ""))[1] == ("no land verdict, and it isn't low-risk" if mode == "on" else "")
    assert _unauthenticated(loop.tick()) == [] and _unauthenticated(env.lines()) == _unauthenticated(recs)


def test_in_shadow_an_unsigned_mark_and_verdict_are_each_logged_and_a_registered_readers_verdict_is_not(env, monkeypatch):
    """Labels written with no token count on "shadow" and are logged, each once; once ci's holder registers the reader's agent
    (`research queue delegate`), the verdict that reader signs authenticates and isn't logged, and a verdict whose reason
    changes is logged again."""
    _main(env, "shadow")
    one, two = env.put(1, {"a.txt": "1\n"}), env.put(2, {"b.txt": "2\n"})
    store = env.o.store
    store.add_label(label_target(1, one), vocab.READY, True, "lane-1", ts="2026-11-02T02:58:00Z")
    store.add_label(label_target(1, one), vocab.REVIEWED, "land", "review-36", ts="2026-11-02T02:59:00Z")
    Q.delegate(store, READER, "review-37", 1, {HOLDER: "ci"}, mint=_mint(HOLDER, iat=T - 3600), now=lambda: T - 3600,
               staffs={"ci": ("review-*",)})
    labelauth.Signer(store, _mint(READER), now=lambda: TS).add_label(_marked(env, 2, two), vocab.REVIEWED, "land", "review-37")
    loop = _loop(env, monkeypatch)
    recs = loop.tick()
    assert [r["members"] for r in recs if r["action"] == "build"] == [[1, 2]]
    assert _unauthenticated(recs) == [(1, [one], vocab.READY, "lane-1", "2026-11-02T02:58:00Z", "unsigned"),
                                      (1, [one], vocab.REVIEWED, "review-36", "2026-11-02T02:59:00Z", "unsigned")]
    assert _unauthenticated(loop.tick()) == []
    # a companion appears beside the unsigned verdict, by an agent that isn't that reader: the reason changes
    lab = next(x for x in store.labels(label_target(1, one), key=vocab.REVIEWED))
    store.add_label(lab.target, vocab.LABEL_AUTH, labelauth.Signer(store, _mint(OTHER)).sign(lab), "review-36", ts=lab.ts)
    (again,) = _unauthenticated(loop.tick())
    assert again[:5] == (1, [one], vocab.REVIEWED, "review-36", "2026-11-02T02:59:00Z") and "which doesn't hold review-36" in again[5]


@pytest.mark.parametrize("broken", ["keys", "registry"])
def test_in_shadow_a_tick_that_cant_read_the_keys_or_the_registry_judges_nothing_says_so_and_goes_on(env, monkeypatch, broken):
    """With the flag on such a tick stops (`label_verifier`); the shadow changes nothing the queue lands, so its tick counts
    every label as written and says why none was judged."""
    def down(cache, log=print):
        raise labelauth.AuthError(f"{labelauth.JWKS_URL}: connection refused, and none are kept at {cache}")

    _main(env, "shadow", **({"registry": "{"} if broken == "registry" else {}))
    head = env.put(1, {"a.txt": "1\n"})
    env.o.store.add_label(label_target(1, head), vocab.READY, True, "lane-1", ts=TS)
    env.o.store.add_label(label_target(1, head), vocab.REVIEWED, "land", "review-36", ts=TS)
    loop = _loop(env, monkeypatch, **({"keys": down} if broken == "keys" else {}))
    recs = loop.tick()
    assert [r["members"] for r in recs if r["action"] == "build"] == [[1]] and _unauthenticated(recs) == []
    why = "connection refused" if broken == "keys" else "registry.json at refs/vqueue/main can't be parsed"
    said = [s for s in env.out if "`signed_verdicts` is \"shadow\" and no mark or verdict is judged now" in s]
    assert len(said) == 1 and why in said[0], env.out
