"""The scan's ok bits at public content (`scan_ok`): the salt rule and each schedule's constants are the vectors
`NetworkCheck.ScanOk`'s guards hold Lean to; the members are the schedule's 2T proof blocks with an ok bit; a
registration is held at its own schedule's constants and at no other's, its start unit's port at any root; two members of one template and one count are at one domain and root;
a schedule of fewer than two ticks has no constants; and the hashes are the constants' alone, counted, none in the
comparison."""

from __future__ import annotations

import hashlib

import pytest

from verity.commitments import advice as ADV
from verity.commitments import hm96, scheme
from verity.commitments.rowleaf import bit_row
from verity.protocols.accounting.certifier.verifier import scan_ok as K

# `NetworkCheck.ScanOk`'s guards' vectors: (port, leaves, domain, root) at T = 16, and the roots at T = 3
T16 = (
    ("ingress-0/o0", 15,
     "d85575918fa7f47e3539edee110443d3b8a548c64c291c23b45b3043ce7b9283"
     "11f74c668f91d782ee01485a04c20797e0001cf41ab5942bded8aa4dc566deef",
     "970dca820db7995e32ba5fd9cf48f02c8cafe07b9b6a6d8d28f64d35a67f7478"
     "669096a154227e1a8f3b1b8773ac2e887626ec60301aecce4535f32790376fa3"),
    ("last/o0", 1,
     "fd35975d6bafaf188590e3dd6a0648bc174821e81a7593ee9346bd01fc5546fe"
     "b2e3d4a45bb7cf9fe6a363b31a5e5c8088ec201d487a39d25f07c0a288236a91",
     "a343d2d50d845bb751143f81fe6b59d8532cdb3238cb04903b63e2362c8dde6b"
     "f12a60a5c5d18a83f7c0b7d961de2c1a9228e2f31894d796a19a3cc5b4da1f8f"),
    ("egress-0/o0", 16,
     "8190bf324e277745d8364f69d58f89ea895eb371ea9f19ce571e4b6b2f9ab181"
     "8b1721da96991ce2600c6346e63535aa87ed9a63b589aa81425f1f53b1055a31",
     "fed250918695d50d116e2e5f263a898fbf82c26d392c02c1cf26ded06c2ef588"
     "0c6b93cc47827632e333a23251c142ebf53526fc904e3a7395b48e8b8ed42419"),
)
T3 = (
    ("ingress-0/o0", 2,
     "3ff491826e963c3a7caa145c82f587b7591900c650757544118ca82b11fc9254"
     "8e258b7a12cf2d3794ca09d48abc4b6b4bf1b7f2cac48389f63699a17b111a4e"),
    ("last/o0", 1, T16[1][3]),
    ("egress-0/o0", 3,
     "f9b478b82e457b566f32ac528e447b185d1df3ba42eb792fed539e61867b9272"
     "e2161f725b83ad09aeb6fd947d5a14c9e5caacff5129b9c4f6ef73e5cd788a53"),
)


def entry(c: K.Constant, **over) -> dict:
    return {"port": c.port, "leaves": c.leaves, "domain": c.domain, "root": c.root} | over


def registration(entries: list) -> dict:
    return {"roots": [*entries, {"port": "scan/state", "leaves": 3, "domain": "d", "root": "r"}]}


@pytest.fixture
def sha512_calls(monkeypatch):
    """Every SHA-512 call counted, through the commitments' registry and through hashlib: the two names the
    verifier's Python hashes by."""
    calls, real = [], hashlib.sha512

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setitem(scheme._SHA, "sha512", counting)
    monkeypatch.setattr(hashlib, "sha512", counting)
    return calls


def test_the_salt_is_sha512_in_counter_mode_over_the_tag_and_the_rows_index():
    assert K.SALT_BYTES == 192 and K.SALT_TAG == b"verity-network-certifier/scan-ok-row-salt/v1\0"
    assert K.salt(0)[:8].hex() == "f61f80289735b3b6" and K.salt(7)[184:].hex() == "27e9fc9788b62bae"
    for i in (0, 7, 2**40):
        head = K.SALT_TAG + i.to_bytes(8, "big")
        assert K.salt(i) == b"".join(hashlib.sha512(head + bytes([0, 0, 0, c])).digest() for c in range(3))
    assert len({K.salt(i) for i in range(64)}) == 64


def test_the_known_content_is_the_ok_row_under_the_public_salts():
    assert K.known(3) == {"rows": [[1]] * 3, "salts": [K.salt(i).hex() for i in range(3)]}
    assert K.known(0) == {"rows": [], "salts": []}


@pytest.mark.parametrize("T", [2, 3, 4, 16, 32])
def test_the_members_are_the_schedules_proof_blocks(T):
    members = K.members(T)
    assert sum(n for _, _, n in members) == 2 * T and all(n >= 1 for _, _, n in members)
    assert [name for name, _, _ in members] == ["ingress-0", "last", "egress-0"]
    assert {name: template for name, template, _ in members} == {
        "ingress-0": "ingress", "last": "ingress", "egress-0": "egress"}
    assert [c.port for c in K.constants(T)] == [f"{name}/o0" for name, _, _ in members]
    assert [c.leaves for c in K.constants(T)] == [n for _, _, n in members]


@pytest.mark.parametrize("T", [1, 0, -3])
def test_a_schedule_of_fewer_than_two_ticks_has_no_constants(T, sha512_calls):
    """At T = 1 the scan's one bucket would be its last, and `ingress-0` a member of no proof block, a shape no scan
    is staged in, so the constants refuse it, in the words `NetworkCheck.ScanOk`'s guard holds Lean's refusal to,
    before any hash; T = 2, the least a scan is staged at, has its three members' constants."""
    with pytest.raises(K.FewTicks) as refused:
        K.constants(T)
    assert str(refused.value) == f"T = {T}: a scan has at least two ticks, its first bucket and its last"
    assert isinstance(refused.value, ValueError) and not sha512_calls
    assert [c.port for c in K.constants(2)] == ["ingress-0/o0", "last/o0", "egress-0/o0"]


def test_the_constants_are_the_vectors_lean_is_held_to():
    assert tuple((c.port, c.leaves, c.domain, c.root) for c in K.constants(16)) == T16
    assert tuple((c.port, c.leaves, c.root) for c in K.constants(3)) == T3


def test_a_constant_is_the_frame_root_of_its_ok_rows_under_the_public_salts():
    """A constant is what committing the member's rows gives: its n rows of the one word 1, row i under salt i, at
    the frame domain of the template's input set over n instances, and of no other row, salt, count or template."""
    for name, template, n in K.members(5):
        const = next(c for c in K.constants(5) if c.port == f"{name}/o0")
        row = bit_row(((1,), 16))[0]
        tree = ADV.commit([row] * n, 16, K.binding(template, n), salts=[K.salt(i) for i in range(n)],
                          key=hm96.SHA512.default_key)
        assert (tree.root.hex(), ADV.domain(tree.binding, n).domain_id.hex()) == (const.root, const.domain)
        zero = bit_row(((0,), 16))[0]
        others = [
            ADV.commit([zero, *[row] * (n - 1)], 16, K.binding(template, n), salts=[K.salt(i) for i in range(n)]),
            ADV.commit([row] * n, 16, K.binding(template, n), salts=[K.salt(i + 1) for i in range(n)]),
            ADV.commit([row] * n, 16, K.binding("other", n), salts=[K.salt(i) for i in range(n)]),
            ADV.commit([row] * (n + 1), 16, K.binding(template, n + 1), salts=[K.salt(i) for i in range(n + 1)]),
        ]
        assert const.root not in {t.root.hex() for t in others}


def test_two_members_of_one_template_and_count_are_at_one_domain_and_root():
    """At T = 2 `ingress-0` and `last` are one proof block of the ingress template each: one input set, one count,
    so one frame domain and one root, under two ports. The rule holds each port to its own constant, which here are
    equal; at T = 3 they differ, and swapping them is refused."""
    ingress, last, egress = K.constants(2)
    assert (ingress.domain, ingress.root, ingress.leaves) == (last.domain, last.root, last.leaves)
    assert ingress.port != last.port and ingress.root != egress.root
    assert K.held(K.constants(2), registration([entry(c) for c in K.constants(2)]))
    assert K.held(K.constants(2), registration([entry(last, port=ingress.port), entry(ingress, port=last.port),
                                                entry(egress)]))
    ingress, last, egress = three = K.constants(3)
    assert ingress.root != last.root and ingress.domain != last.domain
    assert not K.held(three, registration([entry(last, port=ingress.port), entry(ingress, port=last.port),
                                           entry(egress)]))


def test_a_registration_is_held_at_its_schedules_constants_and_no_other():
    cs, other, two = K.constants(3), K.constants(4), K.constants(2)
    es = [entry(c) for c in cs]
    assert K.held(cs, registration(es)) and K.held(cs, registration(es + es))
    assert K.held(cs, registration([*es, entry(cs[0], port="other/o1")]))
    refused = {
        "another schedule's constants": (other, es),
        "another schedule's registration": (cs, [entry(c) for c in other]),
        "a member at another's root": (cs, [*es[:2], entry(cs[2], root=cs[0].root)]),
        "a member at other leaves": (cs, [*es[:2], entry(cs[2], leaves=4)]),
        "a member at another's domain": (cs, [*es[:2], entry(cs[2], domain=cs[0].domain)]),
        "a member missing": (cs, es[1:]),
        "a member missing whose root another member is at": (two, [entry(two[0]), entry(two[2])]),
        "a member named twice, once at another root": (cs, [*es, entry(cs[2], root=cs[0].root)]),
        "a port ending /o0 that is no member's": (cs, [*es, entry(cs[0], port="other/o0")]),
        "the start unit's port and a member missing": (cs, [*es[1:], entry(cs[0], port=K.START)]),
        "leaves a boolean": (two, [entry(two[0], leaves=True), *map(entry, two[1:])]),
        "a root in upper case": (cs, [*es[:2], entry(cs[2], root=cs[2].root.upper())]),
        "leaves a string": (cs, [*es[:2], entry(cs[2], leaves="3")]),
    }
    for what, (consts, entries) in refused.items():
        assert not K.held(consts, registration(entries)), what
    for no_roots in ({}, {"roots": None}, {"roots": {}}, {"registered": es}):
        assert not K.held(cs, no_roots)
    assert K.held(cs, {"roots": [*es, "o0", 7, None, {"port": 3}, {"root": "r"}]})


def test_the_start_units_port_is_held_to_no_constant():
    """The start unit's `o0` is the state it copies, a row of the scan's state and no ok bit: a registration names
    it at any root, leaves and domain, once or twice, and is held as it is without it."""
    cs = K.constants(3)
    es = [entry(c) for c in cs]
    assert K.START == "start/o0" and K.ok_port(K.START) and K.START not in {c.port for c in cs}
    for start in ({"root": "r", "domain": "d", "leaves": 1}, {"root": cs[0].root, "leaves": 7}, {"root": None}):
        assert K.held(cs, registration([*es, entry(cs[0], port=K.START) | start]))
        assert K.held(cs, registration([entry(cs[1], port=K.START) | start, *es, entry(cs[0], port=K.START)]))


def test_the_ports_the_rule_reads_end_in_the_public_word():
    assert all(K.ok_port(p) for p in ("last/o0", "egress-0/o0", "a/b/o0", "/o0"))
    assert not any(K.ok_port(p) for p in ("o0", "last/o1", "last/o00", "last/O0", "last/o0/", "", None, 3))


@pytest.mark.parametrize("T", [2, 3, 16, 32])
def test_the_hashes_are_the_constants_alone(T, sha512_calls):
    """The cost: the constants hash a number of times that is the schedule's alone, the same on every call; reading
    a registration against them hashes nothing, held or refused."""
    cs = K.constants(T)
    once = len(sha512_calls)
    assert once > 0
    K.constants(T)
    assert len(sha512_calls) == 2 * once
    good = registration([entry(c) for c in cs])
    bad = registration([entry(c, root=cs[0].domain) for c in cs])
    assert K.held(cs, good) and not K.held(cs, bad) and not K.held(cs, {})
    assert len(sha512_calls) == 2 * once


def test_the_trees_are_511_hashes_at_32_ticks():
    """Lean's count (`NetworkCheck.ScanOk.rootCalls`): a row is its salt's three hashes, its row digest, its salt
    digest, its hm96 leaf and its frame leaf; then the frame's pads and nodes. The three trees at T = 32 are 511."""
    def root_calls(n: int) -> int:
        width = 1 << (n - 1).bit_length()
        return 7 * n + (width - n) + (width - 1)

    assert [root_calls(n) for n in (1, 15, 16, 31, 32)] == [7, 121, 127, 249, 255]
    assert sum(root_calls(n) for _, _, n in K.members(16)) == 255
    assert sum(root_calls(n) for _, _, n in K.members(32)) == 511
