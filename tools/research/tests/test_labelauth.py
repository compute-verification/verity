"""Authenticated ready marks and land verdicts (`research.labelauth`, drive.toml's `signed_verdicts`).  A landing label counts
only beside a `label_auth` companion whose identity token, signed under a key generated here, is over that label's nonce and
was minted at its time by its handle's holder in the registry, or by a worker that holder delegated to, or, for a PR reader's
name, by an agent the holder of the handle that staffs the name delegated it to; ejections and holds count unsigned.  On
"shadow" each counts and why one wouldn't is kept.  No network, no clock: the issuer's keys are seeded RSA keys and every
time is given."""
from __future__ import annotations

import base64
import functools
import hashlib
import json
import math
import random
import subprocess
from pathlib import Path

import pytest

from research import labelauth as L
from research import queue as Q
from research import slack
from research.queue import PR, DriveFlags, QueueError, StoreLabels, label_target, unreviewed
from research.store import vocab
from research.store.local import LocalStore
from research.store.model import Label

HOLDER = "bc-0000000a-0000-4000-8000-00000000000a"
OTHER = "bc-0000000b-0000-4000-8000-00000000000b"
WORKER = "bc-0000000c-0000-4000-8000-00000000000c"
READER = "bc-0000000d-0000-4000-8000-00000000000d"
HOLDERS = {HOLDER: "ci", OTHER: "proofs"}
STAFFS = {"ci": ("review-*",)}
REAL_REGISTRY = Path(__file__).resolve().parents[3] / Q.REGISTRY
TS = "2026-11-02T03:00:00Z"
T = L.epoch(TS)
HEAD = "a" * 40
TARGET = label_target(7, HEAD)
KID = "test-key"
SMALL_PRIMES = [p for p in range(3, 2000, 2) if all(p % q for q in range(3, math.isqrt(p) + 1, 2))]


def _probable_prime(n: int, rng: random.Random) -> bool:
    if any(n % p == 0 for p in SMALL_PRIMES):
        return False
    d, s = n - 1, 0
    while d % 2 == 0:
        d, s = d // 2, s + 1
    for _ in range(16):
        x = pow(rng.randrange(2, n - 1), d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def _prime(bits: int, rng: random.Random) -> int:
    while True:
        c = rng.getrandbits(bits) | (1 << (bits - 1)) | (1 << (bits - 2)) | 1
        if _probable_prime(c, rng):
            return c


@functools.lru_cache(maxsize=None)
def _key(seed: int) -> tuple[int, int, int]:
    """(n, e, d): an RSA-2048 key from ``seed``, the issuer's for seed 1."""
    rng, e = random.Random(seed), 65537
    while True:
        p, q = _prime(1024, rng), _prime(1024, rng)
        if p != q and math.gcd(e, (p - 1) * (q - 1)) == 1:
            return p * q, e, pow(e, -1, (p - 1) * (q - 1))


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _jwt(claims: dict, seed: int = 1, kid: str = KID, alg: str = "RS256") -> str:
    n, _, d = _key(seed)
    signing = f"{_b64(json.dumps({'alg': alg, 'kid': kid, 'typ': 'JWT'}).encode())}.{_b64(json.dumps(claims).encode())}"
    size = (n.bit_length() + 7) // 8
    digest = hashlib.sha256(signing.encode()).digest()
    info = bytes.fromhex("3031300d060960864801650304020105000420")
    block = b"\x00\x01" + b"\xff" * (size - 3 - len(info) - len(digest)) + b"\x00" + info + digest
    return f"{signing}.{_b64(pow(int.from_bytes(block, 'big'), d, n).to_bytes(size, 'big'))}"


def _claims(agent: str, nonce: str, iat: int = T, **kw) -> dict:
    return {"iss": L.ISSUER, "aud": L.AUDIENCE, "iat": iat, "nbf": iat - 5, "exp": iat + 300, "nonce": nonce,
            "cloud_agent_id": agent, "agent_runtime": "managed", **kw}


def _mint(agent: str, iat: int = T, seed: int = 1, kid: str = KID, **kw):
    return lambda nonce: _jwt(_claims(agent, nonce, iat, **kw), seed=seed, kid=kid)


def _jwks(seed: int = 1) -> dict:
    n, e, _ = _key(seed)
    return {"keys": [{"kty": "RSA", "kid": KID, "alg": "RS256", "use": "sig", "n": _b64(n.to_bytes((n.bit_length() + 7) // 8, "big")),
                      "e": _b64(e.to_bytes(3, "big"))}]}


def _verifier(holders: dict[str, str] = HOLDERS, staffs: dict[str, tuple[str, ...]] | None = None) -> L.Verifier:
    return L.Verifier(L.jwks(_jwks()), holders, staffs)


def _land(by: str = "ci", ts: str = TS, target: str = TARGET, ref: str | None = None) -> Label:
    return Label(target=target, key=vocab.REVIEWED, value="land", by=by, ts=ts, ref=ref)


def _auth(label: Label, mint, delegation: dict | None = None) -> Label:
    value = {"for": L.label_nonce(label), "token": mint(L.label_nonce(label))}
    if delegation is not None:
        value["delegation"] = delegation
    return Label(target=label.target, key=vocab.LABEL_AUTH, value=value, by=label.by, ts=label.ts)


def _delegation(lane: str = "ci", agent: str = WORKER, until: str = "2026-11-09T02:00:00Z", minter: str = HOLDER,
                iat: int = T - 3600) -> dict:
    return L.delegation(lane, agent, until, _mint(minter, iat=iat))


@pytest.fixture
def store(tmp_path):
    """A store with a remote (a directory): `StoreLabels` answers only from labels it pulled from one."""
    (tmp_path / "store.toml").write_text(f'[remote]\ntype = "fs"\npath = "{tmp_path / "vault"}"\n')
    return LocalStore(tmp_path / "store", config_path=tmp_path / "store.toml")


def test_the_issuers_keys_parse_and_the_holders_token_over_the_labels_nonce_authenticates_it():
    n, e, _ = _key(1)
    assert L.jwks(_jwks()) == {KID: (n, e)}
    lab = _land()
    assert _verifier().why(lab, [_auth(lab, _mint(HOLDER))]) is None
    assert _verifier().why(lab, [_auth(lab, _mint(HOLDER, aud=["other", L.AUDIENCE]))]) is None
    assert _verifier().why(lab, []) == "unsigned"


def test_a_label_by_a_handle_its_tokens_agent_doesnt_hold_doesnt_count():
    """The `by` a writer types is checked against the agent the issuer names: another lane's holder, or an agent holding
    nothing, can't write as `ci`."""
    lab = _land()
    for agent in (OTHER, WORKER):
        why = _verifier().why(lab, [_auth(lab, _mint(agent))])
        assert f"its token is {agent!r}'s, which doesn't hold ci in main's registry, and it carries no delegation" == why
    assert _verifier().why(_land(by="@CI"), [_auth(_land(by="@CI"), _mint(HOLDER))]) is None


def test_a_token_bound_to_another_label_doesnt_count():
    """The nonce names the label's target, key, value, author, time and ref: a companion copied beside another label, or one
    whose `for` names this label over a token minted for another, authenticates nothing."""
    lab, other = _land(), _land(target=label_target(7, "b" * 40))
    assert _verifier().why(lab, [_auth(other, _mint(HOLDER))]) == "unsigned"
    moved = _auth(other, _mint(HOLDER))
    assert _verifier().why(other, [Label(**{**moved.__dict__, "target": lab.target})]) == "unsigned"
    crossed = {"for": L.label_nonce(lab), "token": _mint(HOLDER)(L.label_nonce(other))}
    assert _verifier().why(lab, [Label(TARGET, vocab.LABEL_AUTH, crossed, "ci", TS)]) == "its token's nonce isn't this label's"
    ready = Label(target=TARGET, key=vocab.READY, value=True, by="ci", ts=TS)
    assert _verifier().why(ready, [_auth(lab, _mint(HOLDER))]) == "unsigned"
    assert _verifier().why(_land(ref="read it"), [_auth(lab, _mint(HOLDER))]) == "unsigned"


@pytest.mark.parametrize("mint, why", [
    (_mint(HOLDER, seed=2), "the token's signature doesn't verify under the issuer's key"),
    (_mint(HOLDER, kid="retired"), "the token is signed with key 'retired', not one of the issuer's (test-key)"),
    (lambda n: _jwt(_claims(HOLDER, n), alg="HS256"), "the token is signed with 'HS256', not RS256"),
    (_mint(HOLDER, iss="https://elsewhere.example"), "the token's issuer is 'https://elsewhere.example'"),
    (_mint(HOLDER, aud="urn:other"), "the token's audience is 'urn:other'"),
    (lambda n: _mint(HOLDER)(n).rpartition(".")[0] + "." + _b64(b"\x00" * 256), "doesn't verify"),
    (lambda n: "not-a-jwt", "no token, or not a JWT"),
])
def test_a_token_not_signed_by_the_issuers_key_for_this_audience_doesnt_count(mint, why):
    lab = _land()
    assert why in _verifier().why(lab, [_auth(lab, mint)])


def test_a_token_whose_claims_were_swapped_after_signing_doesnt_count():
    lab = _land()
    good = _mint(HOLDER)(L.label_nonce(lab))
    h, _, sig = good.split(".")
    forged = f"{h}.{_b64(json.dumps(_claims(OTHER, L.label_nonce(lab))).encode())}.{sig}"
    v = {"for": L.label_nonce(lab), "token": forged}
    assert _verifier(holders={OTHER: "ci"}).why(lab, [Label(TARGET, vocab.LABEL_AUTH, v, "ci", TS)]) == \
        "the token's signature doesn't verify under the issuer's key"


def test_a_token_minted_more_than_its_lifetime_from_the_labels_time_doesnt_count():
    """A label's time is in its nonce and the token's `iat` is the issuer's, so a token minted now can't vouch for a label
    dated an hour ago."""
    lab = _land()
    assert _verifier().why(lab, [_auth(lab, _mint(HOLDER, iat=T - L.SKEW_S))]) is None
    assert _verifier().why(lab, [_auth(lab, _mint(HOLDER, iat=T + L.SKEW_S + 1))]) == \
        f"its token was minted +{L.SKEW_S + 1} s from the label's time, more than {L.SKEW_S} s"


def test_one_good_companion_among_bad_ones_authenticates_and_the_reasons_name_each_bad_one():
    lab = _land()
    bad, good = _auth(lab, _mint(OTHER)), _auth(lab, _mint(HOLDER))
    assert _verifier().why(lab, [bad, good]) is None
    assert _verifier().why(lab, [bad, _auth(lab, _mint(HOLDER, seed=2))]).count("; ") == 1


def test_a_workers_label_counts_under_its_holders_delegation_and_not_outside_it():
    lab = _land()
    v = _verifier()
    assert v.why(lab, [_auth(lab, _mint(WORKER), _delegation())]) is None
    cases = {
        "the label's time is outside its delegation": _delegation(until="2026-11-02T02:59:59Z"),
        f"its delegation was minted by {OTHER!r}, which doesn't hold ci": _delegation(minter=OTHER),
        f"its delegation is of 'ci' to {OTHER!r}, not of ci to its token's {WORKER!r}": _delegation(agent=OTHER),
        f"its delegation is of 'proofs' to {WORKER!r}": _delegation(lane="proofs", minter=OTHER),
        f"its delegation runs past {L.DELEGATION_MAX_DAYS} days": _delegation(until="2026-11-17T03:00:00Z"),
        "its delegation's token isn't over the delegation": {**_delegation(), "until": "2026-11-20T00:00:00Z"},
        f"its delegation isn't a {L.DELEGATION_SCHEMA} statement": {**_delegation(), "until": 7},
    }
    for why, d in cases.items():
        assert why in v.why(lab, [_auth(lab, _mint(WORKER), d)]), why
    late = _land(ts="2026-11-02T01:00:00Z")
    assert "outside its delegation" in v.why(late, [_auth(late, _mint(WORKER, iat=L.epoch(late.ts)), _delegation())])


def test_a_readers_verdict_counts_only_under_a_delegation_of_its_name_by_the_holder_of_the_handle_that_staffs_it():
    """A reader's name is no handle's, so no agent holds it: its land verdict counts when its token's agent carries a delegation
    of that name minted by the holder of the handle whose `staffs` in the registry matches it, and under nothing else."""
    lab, v = _land(by="review-36"), _verifier(staffs=STAFFS)
    d = _delegation(lane="review-36", agent=READER)
    assert v.why(lab, [_auth(lab, _mint(READER), d)]) is None
    assert v.why(_land(by="@Review-36"), [_auth(_land(by="@Review-36"), _mint(READER), d)]) is None
    minted_by = "its delegation was minted by {!r}, which doesn't hold review-36 in main's registry, nor a handle that staffs it"
    assert _verifier().why(lab, [_auth(lab, _mint(READER), d)]) == minted_by.format(HOLDER)      # a registry staffing no one
    cases = {
        "unsigned": [],
        f"its token is {READER!r}'s, which doesn't hold review-36 in main's registry, and it carries no delegation":
            [_auth(lab, _mint(READER))],
        f"its token is {HOLDER!r}'s, which doesn't hold review-36 in main's registry, and it carries no delegation":
            [_auth(lab, _mint(HOLDER))],                                # the staffer registers a reader; it isn't one
        minted_by.format(OTHER): [_auth(lab, _mint(READER), _delegation(lane="review-36", agent=READER, minter=OTHER))],
        minted_by.format(READER): [_auth(lab, _mint(READER), _delegation(lane="review-36", agent=READER, minter=READER))],
        f"its delegation is of 'review-37' to {READER!r}, not of review-36 to its token's {READER!r}":
            [_auth(lab, _mint(READER), _delegation(lane="review-37", agent=READER))],
        f"its delegation is of 'review-36' to {READER!r}, not of review-36 to its token's {WORKER!r}":
            [_auth(lab, _mint(WORKER), d)],
        "the label's time is outside its delegation":
            [_auth(lab, _mint(READER), _delegation(lane="review-36", agent=READER, until="2026-11-02T02:59:59Z"))],
    }
    for why, beside in cases.items():
        assert v.why(lab, beside) is not None and why in v.why(lab, beside), why
    # staffing a reader's name reaches no handle, and no name its globs don't match
    for name in ("proofs", "reviewer", "top-review-1"):
        other = _land(by=name)
        got = v.why(other, [_auth(other, _mint(READER), _delegation(lane=name, agent=READER))])
        assert got == f"its delegation was minted by {HOLDER!r}, which doesn't hold {name} in main's registry, nor a handle " \
                      "that staffs it", name


def test_the_staffer_registers_a_readers_agent_which_then_signs_its_verdicts_as_that_reader(store, capsys):
    """`research queue delegate AGENT --by review-N`, by the holder of the handle that staffs the name, is a reader's
    registration: the reader's own `review` then writes a verdict that authenticates, and no other handle's holder can."""
    for holders, staffs, minter in ((HOLDERS, STAFFS, OTHER), (HOLDERS, None, HOLDER), (HOLDERS, {"proofs": ("review-*",)}, HOLDER)):
        with pytest.raises(QueueError, match="doesn't hold review-36 in main's registry .*nor a handle whose `staffs` there names it"):
            Q.delegate(store, READER, "review-36", 1, holders, mint=_mint(minter, iat=T - 3600), now=lambda: T - 3600, staffs=staffs)
    assert store.labels(L.delegation_target(READER)) == []
    signer = L.Signer(store, _mint(READER), required=True, holders=HOLDERS, now=lambda: TS)
    with pytest.raises(L.AuthError, match=f"`research queue delegate {READER} --by review-36`"):
        signer.add_label(TARGET, vocab.REVIEWED, "land", "review-36")
    assert Q.delegate(store, READER, "review-36", 1, HOLDERS, mint=_mint(HOLDER, iat=T - 3600), now=lambda: T - 3600,
                      staffs=STAFFS) == 0
    assert f"{READER} signs as review-36 until 2026-11-03T02:00:00Z, while this agent holds ci" in capsys.readouterr().out
    (written,) = store.labels(L.delegation_target(READER))
    assert (written.by, written.value["lane"], written.value["agent"]) == ("ci", "review-36", READER)
    signer.add_label(TARGET, vocab.REVIEWED, "land", "review-36", ref="read it")
    (auth,) = store.labels(TARGET, key=vocab.LABEL_AUTH)
    assert (auth.value["delegation"]["lane"], auth.value["delegation"]["agent"]) == ("review-36", READER)
    counted = StoreLabels(store, log=lambda s: None, verify=_verifier(staffs=STAFFS))
    assert unreviewed(counted.reviews(7, HEAD), {"vllm"}) is None and counted.refusals(7, HEAD, vocab.REVIEWED) == []
    store.add_label(TARGET, vocab.REVIEWED, "land", "review-37", ts="2026-11-02T03:01:00Z")       # written as a reader by anyone
    assert [lab.by for lab in counted.reviews(7, HEAD)] == ["review-36"]
    assert counted.refusals(7, HEAD, vocab.REVIEWED) == ["reviewed = land by review-37 at 2026-11-02T03:01:00Z: unsigned"]


def _registry(tmp_path, handles: list[dict] | None) -> None:
    """A commit of ``tmp_path`` whose registry holds ``handles`` (none: no registry)."""
    if not (tmp_path / ".git").exists():
        subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    if handles is not None:
        (tmp_path / Q.REGISTRY).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / Q.REGISTRY).write_text(json.dumps({"handles": handles}))
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty",
                    "-m", "x"], check=True)


def test_the_registry_names_the_names_each_handle_staffs_and_a_glob_never_reaches_a_handle(tmp_path, monkeypatch):
    """A handle's `staffs` in `main`'s registry is the reader names its holder registers agents under; one that matched a
    handle would let that holder sign as it, so the registry is refused, as one that can't be parsed is."""
    ci, proofs = {"handle": "ci", "holder": HOLDER, "staffs": ["review-*"]}, {"handle": "proofs", "holder": OTHER}
    _registry(tmp_path, None)
    assert Q.staffs_at(tmp_path, "HEAD") == {}
    _registry(tmp_path, [ci, proofs])
    assert Q.staffs_at(tmp_path, "HEAD") == STAFFS and Q.holders_at(tmp_path, "HEAD") == HOLDERS
    monkeypatch.setattr(L, "issuer_keys", lambda cache, log=print: L.jwks(_jwks()))
    v = Q.label_verifier(tmp_path, "HEAD", tmp_path / "keys.json")
    lab = _land(by="review-36")
    assert v.why(lab, [_auth(lab, _mint(READER), _delegation(lane="review-36", agent=READER))]) is None
    for staffs, why in ((["pro*"], "it matches proofs, and a handle is delegated only by its own holder"),
                        (["review-*", "*"], "it matches ci, proofs, and a handle is delegated only by its own holder"),
                        (["ci"], "it matches ci, and"),
                        ("review-*", "it is a list of names or globs of names"),
                        (["Review-*"], "it is a list of names or globs of names"),
                        (["review *"], "it is a list of names or globs of names"),
                        ([""], "it is a list of names or globs of names"),
                        ([7], "it is a list of names or globs of names")):
        _registry(tmp_path, [{**ci, "staffs": staffs}, proofs])
        with pytest.raises(QueueError, match="registry.json at HEAD: ci's staffs = ") as e:
            Q.label_verifier(tmp_path, "HEAD", tmp_path / "keys.json")
        assert why in str(e.value), staffs


def test_the_repositorys_registry_staffs_readers_under_ci_and_reaches_no_handle():
    """ci staffs the PR readers (the team skill's "Landing"), so its holder registers each reader's agent."""
    handles = json.loads(REAL_REGISTRY.read_text())["handles"]
    staffs = Q.staffs_of(handles, str(Q.REGISTRY))
    assert [h for h in staffs if L.speaks_for(h, "review-36", staffs)] == ["ci"]
    assert not any(L.speaks_for(h, o["handle"], staffs) for h in staffs for o in handles if o["handle"] != h)


def test_in_shadow_every_mark_and_verdict_counts_and_why_one_wouldnt_is_kept(store):
    """`signed_verdicts = "shadow"`: the labels read exactly as with the flag off, and each ready mark and land verdict the
    verifier wouldn't count is kept with why, by its PR, head, key, author and time; an authenticated one, an ejection and a
    hold aren't."""
    def labels() -> StoreLabels:
        out = StoreLabels(store, log=lambda s: None, verify=_verifier(staffs=STAFFS))
        out.shadow = True
        return out

    def read(lab: StoreLabels) -> tuple:
        return lab.ready(7, HEAD), lab.ready_by(7, HEAD), unreviewed(lab.reviews(7, HEAD), {"vllm"}), lab.lands(7, HEAD)

    store.add_label(TARGET, vocab.READY, True, "vllm", ts="2026-11-02T02:58:00Z")
    store.add_label(TARGET, vocab.REVIEWED, "land", "review-36", ts="2026-11-02T02:59:00Z")
    L.Signer(store, _mint(READER), now=lambda: TS).add_label(TARGET, vocab.REVIEWED, "land", "review-37")
    shadow = labels()
    assert read(shadow) == read(StoreLabels(store, log=lambda s: None)) == (True, "vllm", None, True)
    assert shadow.refusals(7, HEAD, vocab.READY) == shadow.refusals(7, HEAD, vocab.REVIEWED) == []
    assert shadow.unauthenticated == {
        (7, HEAD, vocab.READY, "vllm", "2026-11-02T02:58:00Z"): "unsigned",
        (7, HEAD, vocab.REVIEWED, "review-36", "2026-11-02T02:59:00Z"): "unsigned",
        (7, HEAD, vocab.REVIEWED, "review-37", TS):
            f"its token is {READER!r}'s, which doesn't hold review-37 in main's registry, and it carries no delegation"}
    strict = StoreLabels(store, log=lambda s: None, verify=_verifier(staffs=STAFFS))
    assert not strict.ready(7, HEAD) and strict.unauthenticated == {}                       # on: refused, as `refusals` says
    Q.delegate(store, READER, "review-38", 1, HOLDERS, mint=_mint(HOLDER, iat=T - 3600), now=lambda: T - 3600, staffs=STAFFS)
    L.Signer(store, _mint(READER), now=lambda: "2026-11-02T03:01:00Z").add_label(TARGET, vocab.REVIEWED, "land", "review-38")
    store.add_label(TARGET, vocab.REVIEWED, "hold", "proofs", ref="wait", ts="2026-11-02T03:02:00Z")
    store.add_label(TARGET, vocab.READY, False, "merge-queue", ts="2026-11-02T03:03:00Z")
    shadow = labels()
    assert read(shadow) == read(StoreLabels(store, log=lambda s: None)) == (False, "vllm", "held by proofs (wait)", False)
    assert sorted(k[3] for k in shadow.unauthenticated) == ["review-36", "review-37", "vllm"]


def test_in_shadow_a_label_the_verifier_cant_judge_still_counts_and_says_so(store):
    class Broken:
        def why(self, label, beside):
            raise RuntimeError("no keys")

    store.add_label(TARGET, vocab.READY, True, "vllm", ts=TS)
    labels = StoreLabels(store, log=lambda s: None, verify=Broken())
    with pytest.raises(RuntimeError):
        labels.ready(7, HEAD)
    labels.shadow = True
    assert labels.ready(7, HEAD)
    assert labels.unauthenticated == {(7, HEAD, vocab.READY, "vllm", TS): "it can't be judged: RuntimeError: no keys"}


def test_with_a_verifier_only_authenticated_marks_count_and_an_ejection_holds_unsigned(store):
    labels = StoreLabels(store, log=lambda s: None, verify=_verifier())
    store.add_label(TARGET, vocab.READY, True, "ci", ts="2026-11-02T02:59:00Z")
    assert not labels.ready(7, HEAD) and labels.ready_by(7, HEAD) is None
    assert labels.refusals(7, HEAD, vocab.READY) == ["ready = True by ci at 2026-11-02T02:59:00Z: unsigned"]
    assert StoreLabels(store, log=lambda s: None).ready(7, HEAD)                 # with the flag off, as written
    L.Signer(store, _mint(HOLDER), now=lambda: TS).add_label(TARGET, vocab.READY, True, "ci")
    assert labels.ready(7, HEAD) and labels.ready_by(7, HEAD) == "ci" and labels.marked_at(7, HEAD) == TS
    store.add_label(TARGET, vocab.READY, False, "merge-queue", ref="broke next", ts="2026-11-02T03:01:00Z")
    assert not labels.ready(7, HEAD) and labels.marked_by(7, HEAD) == "merge-queue"
    store.add_label(TARGET, vocab.READY, True, "ci", ts="2026-11-02T03:02:00Z")      # written as ci by anyone
    assert not labels.ready(7, HEAD)


def test_with_a_verifier_only_authenticated_land_verdicts_count_and_any_hold_holds(store):
    labels, owners = StoreLabels(store, log=lambda s: None, verify=_verifier()), {"vllm"}
    store.add_label(TARGET, vocab.REVIEWED, "land", "ci", ts="2026-11-02T02:59:00Z")
    assert unreviewed(labels.reviews(7, HEAD), owners) == "no land verdict"
    assert unreviewed(StoreLabels(store, log=lambda s: None).reviews(7, HEAD), owners) is None
    L.Signer(store, _mint(HOLDER), now=lambda: TS).add_label(TARGET, vocab.REVIEWED, "land", "ci", ref="read it")
    assert unreviewed(labels.reviews(7, HEAD), owners) is None
    store.add_label(TARGET, vocab.REVIEWED, "hold", "proofs", ref="wait", ts="2026-11-02T03:01:00Z")
    store.add_label(TARGET, vocab.REVIEWED, "land", "proofs", ts="2026-11-02T03:02:00Z")  # not proofs' holder's
    assert unreviewed(labels.reviews(7, HEAD), owners) == "held by proofs (wait)"


def test_the_signer_signs_landing_labels_only_and_refuses_what_it_cant_sign_once_required(store):
    def down(nonce: str) -> str:
        raise L.AuthError("the Cursor identity socket is unavailable")

    said: list[str] = []
    L.Signer(store, down, now=lambda: TS, log=said.append).add_label(TARGET, vocab.REVIEWED, "hold", "ci", ref="wait")
    assert said == [] and [lab.key for lab in store.labels(TARGET)] == [vocab.REVIEWED]
    L.Signer(store, down, now=lambda: TS, log=said.append).add_label(TARGET, vocab.READY, True, "ci")
    assert "is written unsigned (the Cursor identity socket is unavailable)" in said[0]
    assert [lab.key for lab in store.labels(TARGET)] == [vocab.REVIEWED, vocab.READY]
    with pytest.raises(L.AuthError, match="unavailable"):
        L.Signer(store, down, required=True, now=lambda: TS).add_label(TARGET, vocab.REVIEWED, "land", "ci")
    with pytest.raises(L.AuthError, match=f"`research queue delegate {WORKER} --by ci`"):
        L.Signer(store, _mint(WORKER), required=True, holders=HOLDERS, now=lambda: TS).add_label(TARGET, vocab.REVIEWED, "land", "ci")
    assert len(store.labels(TARGET)) == 2


def test_a_delegated_worker_signs_with_its_delegation_which_only_the_holder_can_write(store, capsys):
    with pytest.raises(QueueError, match=f"this agent \\({OTHER}\\) doesn't hold ci"):
        Q.delegate(store, WORKER, "ci", 7, HOLDERS, mint=_mint(OTHER, iat=T - 3600), now=lambda: T - 3600)
    with pytest.raises(QueueError, match="--days 15: 1 to 14"):
        Q.delegate(store, WORKER, "ci", 15, HOLDERS, mint=_mint(HOLDER))
    with pytest.raises(QueueError, match="isn't a Cursor agent id"):
        Q.delegate(store, "worker", "ci", 7, HOLDERS, mint=_mint(HOLDER))
    assert store.labels(L.delegation_target(WORKER)) == []
    assert Q.delegate(store, WORKER, "ci", 7, HOLDERS, mint=_mint(HOLDER, iat=T - 3600), now=lambda: T - 3600) == 0
    assert f"{WORKER} signs as ci until 2026-11-09T02:00:00Z" in capsys.readouterr().out
    L.Signer(store, _mint(WORKER), required=True, holders=HOLDERS, now=lambda: TS).add_label(TARGET, vocab.READY, True, "ci")
    (auth,) = store.labels(TARGET, key=vocab.LABEL_AUTH)
    assert auth.value["delegation"]["agent"] == WORKER
    assert StoreLabels(store, log=lambda s: None, verify=_verifier()).ready(7, HEAD)


def test_a_mark_or_verdict_the_signer_refuses_is_not_written(store, capsys):
    def down(nonce: str) -> str:
        raise L.AuthError("the Cursor identity socket is unavailable")

    pr = PR(7, HEAD, body="Lane vllm")
    assert Q._mark(L.Signer(store, down, required=True, now=lambda: TS), pr, "ci", None, "ready") == 1
    assert "can't be signed: the Cursor identity socket is unavailable" in capsys.readouterr().err
    assert store.labels(TARGET) == []
    signer = L.Signer(store, _mint(HOLDER), required=True, holders=HOLDERS, now=lambda: TS)
    assert Q._mark(signer, pr, "ci", None, "ready") == 0
    assert StoreLabels(store, log=lambda s: None, verify=_verifier()).ready(7, HEAD)
    assert Q._mark(store, PR(8, HEAD, body="Lane vllm"), "ci", None, "ready") == 0          # no signer: as before
    assert [lab.key for lab in store.labels(label_target(8, HEAD))] == [vocab.READY]


def test_the_flag_is_off_by_default_and_on_only_while_the_low_risk_lane_cant_change_who_holds_a_handle():
    assert DriveFlags().signed_verdicts == "off" and DriveFlags.parse("signed_verdicts = false\n").signed_verdicts == "off"
    excludes = '["*.lean", "landing.py", "queue.py"]'
    with pytest.raises(QueueError, match="registry.json or tools/research/src/research/labelauth.py land on its ready mark"):
        DriveFlags.parse(f"signed_verdicts = true\nlowrisk_excludes = {excludes}\n")
    full = '["*.lean", "landing.py", "queue.py", "labelauth.py", ".agents/skills/using-slack/"]'
    assert DriveFlags.parse(f"signed_verdicts = true\nlowrisk_excludes = {full}\n").signed_verdicts == "on"
    without_landing = '["*.lean", "drive.py", "queue.py", "labelauth.py", ".agents/skills/using-slack/"]'
    with pytest.raises(QueueError, match="a change to tools/research/src/research/landing.py land on its ready mark"):
        DriveFlags.parse(f"signed_verdicts = true\nlowrisk_excludes = {without_landing}\n")
    assert DriveFlags.parse("signed_verdicts = true\nlowrisk_lane = false\n").signed_verdicts == "on"
    for bad in ('"yes"', '"on"', '"off"', "1", '"Shadow"'):
        with pytest.raises(QueueError, match='signed_verdicts = .*false, "shadow" .*or true'):
            DriveFlags.parse(f"signed_verdicts = {bad}\n")


def test_the_shadow_refuses_nothing_so_it_runs_beside_the_low_risk_lane_as_it_is():
    """ "shadow" counts every label, so no lane gains by changing who holds a handle: it needs none of what "on" needs."""
    flags = DriveFlags.parse('signed_verdicts = "shadow"\nlowrisk_excludes = ["*.lean"]\n')
    assert flags.signed_verdicts == "shadow" and flags.lowrisk_lane
    assert "signed_verdicts=shadow" in flags.line()


def test_a_client_reads_only_the_flag_at_main(tmp_path):
    def commit(text: str) -> str:
        (tmp_path / "tools/verity/check").mkdir(parents=True, exist_ok=True)
        (tmp_path / Q.DRIVE_FLAGS).write_text(text)
        subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
        subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "x"], check=True)
        return subprocess.run(["git", "-C", str(tmp_path), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    assert Q.signed_at(tmp_path, "HEAD") == "off"                                           # no drive.toml there
    assert Q.signed_at(tmp_path, commit("on = [\"m\"]\n")) == "off"
    assert Q.signed_at(tmp_path, commit("signed_verdicts = false\n")) == "off"
    assert Q.signed_at(tmp_path, commit("signed_verdicts = true\nsome_newer_key = 1\n")) == "on"
    assert Q.signed_at(tmp_path, commit('signed_verdicts = "shadow"\nsome_newer_key = 1\n')) == "shadow"
    with pytest.raises(QueueError, match="signed_verdicts = 'yes': false, \"shadow\" or true"):
        Q.signed_at(tmp_path, commit('signed_verdicts = "yes"\n'))


def test_the_issuers_keys_are_fetched_and_the_last_set_covers_an_outage(tmp_path):
    cache, said = tmp_path / "q" / "keys.json", []
    keys = L.issuer_keys(cache, get=lambda url: json.dumps(_jwks()).encode())
    assert keys == L.jwks(_jwks()) and json.loads(cache.read_text()) == _jwks()

    def down(url: str) -> bytes:
        raise slack.SlackError(f"{url}: connection refused")

    assert L.issuer_keys(cache, get=down, log=said.append) == keys and "connection refused" in said[0]
    assert L.issuer_keys(cache, get=lambda url: json.dumps(_jwks(2)).encode()) == L.jwks(_jwks(2))   # replaced, not merged
    with pytest.raises(L.AuthError, match="none are kept"):
        L.issuer_keys(tmp_path / "none.json", get=down)
    with pytest.raises(L.AuthError, match="lists no RSA signing key"):
        L.issuer_keys(tmp_path / "none.json", get=lambda url: b'{"keys": []}')


def test_drives_verifier_refuses_the_tick_when_it_cant_read_the_issuers_keys_or_the_registry(tmp_path, monkeypatch):
    """Judging every mark absent would take every PR off `next`: drive does nothing instead, saying why."""
    def commit() -> None:
        subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
        subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty",
                        "-m", "x"], check=True)

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    commit()

    def down(cache, log=print):
        raise L.AuthError(f"{L.JWKS_URL}: connection refused, and none are kept at {cache}")

    monkeypatch.setattr(L, "issuer_keys", down)
    with pytest.raises(QueueError, match="no ready mark or land verdict can be authenticated .*connection refused"):
        Q.label_verifier(tmp_path, "HEAD", tmp_path / "keys.json")
    (tmp_path / Q.REGISTRY).parent.mkdir(parents=True)
    (tmp_path / Q.REGISTRY).write_text('{"handles": [{"handle": "ci", "holder": "' + HOLDER + '"}]}')
    commit()
    assert Q.holders_at(tmp_path, "HEAD") == {HOLDER: "ci"}
    (tmp_path / Q.REGISTRY).write_text("{")
    commit()
    with pytest.raises(QueueError, match="registry.json at HEAD can't be parsed"):
        Q.label_verifier(tmp_path, "HEAD", tmp_path / "keys.json")
