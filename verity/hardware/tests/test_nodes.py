"""The node files: every number an entry under the cards' rules, each naming a GPU card and a host card and no
machine, address or provider. A node whose file states its host's DRAM and disks gives a log accepted for `remote`
against the auditor the card files give; a file without them gives no entry. `onsite` refuses a node whose file states
an interface, by `uncertified`: the interface is the node's own way out, which no certifier splits, with a network
certifier standing in front of the node or without one. It refuses a node whose file states none until a memory
challenger is registered."""

from __future__ import annotations

import math
import re
from dataclasses import replace

import pytest

from verity.errors import InvalidArtifact
from verity.hardware import cards, nodes
from verity.hardware.cards import MARKS
from verity.hardware.site import (OUTSIDE, TICK, TIERS, Auditor, Entry, Link, Loc, Malformed, Refused, Registration,
                                  Sheet, check, refusal)

HOST = "host0"
"""The id a test registers a node's host under: a node file names none."""
CHALLENGER = "challenger0"
CERTIFIER = ("certifier0", "dummy-certifier0-key")
"""A network certifier's id and key, as whoever registers a node gives them: a node file names neither."""
READ = [kind for kind in nodes.kinds()
        if {"dram", "disk"} <= {leaf["tier"] for leaf in nodes.load(kind).get("memory", {}).values()}]
"""The nodes whose file states the host's DRAM and a disk: the ones an entry can be built of."""
WIRED = [kind for kind in READ if nodes.load(kind).get("interface")]
"""Those whose file also states an interface that carries traffic off the node."""
NAMED = re.compile(r"\bvy-[a-z0-9-]+|\bcomputeinstance-\w+|\b(?:\d{1,3}\.){3}\d{1,3}\b|\b(?:nebius|runpod)\b", re.I)
"""What a node file never holds: a machine's name or instance id, an address, or a provider's name. A machine's name
is matched by the prefix every one of the team's machines carries, and a provider by the names of the two that host
them."""


def entries(table: dict, at: str = "") -> dict[str, dict]:
    """Every number entry of a node file, by its dotted key."""
    out = {}
    for key, item in table.items():
        if isinstance(item, dict) and "value" in item:
            out[at + key] = item
        elif isinstance(item, dict):
            out |= entries(item, f"{at}{key}.")
    return out


def auditor(**keys: str) -> Auditor:
    """The auditor whose data sheets are the GPU card files', with these keys of its own devices."""
    return Auditor.of_files(map(cards.load, cards.skus("gpu")), keys=keys)


def log_of(kind: str, composite: str, certifier: tuple[str, str] | None = None) -> Registration:
    """The composite's log of one entry, the node's, with this network certifier standing in front of the node."""
    node = nodes.load(kind)
    log = Registration(composite)
    return log.append(log.node_entry(node, cards.load(node["device"]["sku"]), HOST, certifier=certifier))


def reached(entry: Entry) -> Entry:
    """`entry` with a memory challenger of the auditor's linked to the host."""
    return replace(entry, locs=(*entry.locs, Loc(CHALLENGER, "challenger", "site", "auditor", key="dummy-key")),
                   links=(*entry.links, Link(f"{CHALLENGER}-link", (CHALLENGER, HOST), 1, 1)))


def test_the_nodes_we_run() -> None:
    """Each states its host's DRAM and disks, so the acceptance tests below hold each one's log. The interfaces of the
    eight-card node are read; no run has read the H200 node's."""
    assert nodes.kinds() == READ == ["h200-sxm-141gb-x1", "rtx-pro-6000-server-x8"]
    assert WIRED == ["rtx-pro-6000-server-x8"]


@pytest.mark.parametrize("kind", nodes.kinds())
def test_a_node_file_states_what_no_card_does(kind: str) -> None:
    """Its kind is its card and count; it names a GPU card and a host card; and its only numbers are the count, each
    memory leaf's bytes, in bytes and whole, under a tier of the host's own, and each interface's line rate, in bits a
    second. A leaf and an interface share no name, since each is registered as `host/NAME`."""
    node = nodes.load(kind)
    device, count = node["device"], cards.number(node["device"]["count"])
    assert kind == f"{device['sku']}-x{count}" and count >= 1
    assert cards.load(device["sku"])["kind"] == "gpu" and cards.load(node["host"]["sku"])["kind"] == "host"
    assert set(node) <= {"node", "name", "device", "host", "memory", "interface"}
    leaves, ports = node.get("memory", {}), node.get("interface", {})
    assert sorted(entries(node)) == sorted(["device.count", *(f"memory.{name}.bytes" for name in leaves),
                                            *(f"interface.{name}.rate" for name in ports)])
    for name, leaf in leaves.items():
        assert sorted(leaf) == ["bytes", "tier"] and leaf["tier"] in set(TIERS) - {"hbm"}, name
        assert leaf["bytes"]["unit"] == "B" and cards.number(leaf["bytes"]) > 0, name
    for name, port in ports.items():
        assert sorted(port) == ["rate"] and port["rate"]["unit"] == "bit/s" and cards.number(port["rate"]) > 0, name
    assert not set(leaves) & set(ports) and not any(re.fullmatch(r"gpu\d+", name) for name in (*leaves, *ports))


@pytest.mark.parametrize("kind", nodes.kinds())
def test_every_number_is_sourced_as_a_cards_is(kind: str) -> None:
    """The cards' rules: a mark and a source on every number, evidence cited by its `art:` id, never by a run id, and
    a measurement citing the record of what measured it."""
    numbers = entries(nodes.load(kind))
    assert all(e["mark"] in MARKS and e["source"].strip() for e in numbers.values())
    text = (nodes.NODES / f"{kind}.toml").read_text(encoding="utf-8")
    assert not re.findall(r"\br\d{8}-\d{6}-[0-9a-f]{4}\b", text), \
        "cite the run's record (`research data show RUN`: outputs.run_record) as art:<12 hex>"
    uncited = sorted(key for key, e in numbers.items()
                     if e["mark"] == "measurement" and not re.search(r"\bart:[0-9a-f]{12}", e["source"]))
    assert not uncited, (f"{kind}: marked `measurement` with no record cited: {uncited}; add the record of the run "
                         "or the read that measured each to its source as art:<12 hex>, or leave the leaf out until "
                         "one reads it")


@pytest.mark.parametrize("kind", nodes.kinds())
def test_a_node_file_names_no_machine_address_or_provider(kind: str) -> None:
    """The node files are in the public tree, and a kind of node is not a machine: a file says "a VM of this kind" and
    "the node", and cites what read it by its `art:` id."""
    text = (nodes.NODES / f"{kind}.toml").read_text(encoding="utf-8")
    assert not NAMED.findall(text), (f"{kind}.toml names {sorted(set(NAMED.findall(text)))}; name the node by its "
                                     "kind and the read by its art:<12 hex>, and leave the machine, its address and "
                                     "its provider out")


@pytest.mark.parametrize("text", ["read on vy-example-9", "computeinstance-e00example", "ssh 192.0.2.7", "a Nebius VM"])
def test_what_a_node_file_never_holds_is_matched(text: str) -> None:
    assert NAMED.search(text)


@pytest.mark.parametrize("kind", READ)
def test_remote_accepts_a_nodes_log(kind: str) -> None:
    """The site holds the node's count of devices, each naming the sheet its card file gives and holding that sheet's
    device memory, whole, and the host's leaves at the file's bytes, by tier. Its links are the file's interfaces, each
    from the host to the outside at its rate over a tick, rounded up: boundary links, none of them certified."""
    node = nodes.load(kind)
    sheet = Sheet.of_file(cards.load(node["device"]["sku"]))
    count, leaves = int(cards.number(node["device"]["count"])), node.get("memory", {}).values()
    site = check(log_of(kind, "remote"), auditor())
    assert site.units == tuple(f"{HOST}/gpu{i}" for i in range(count))
    assert all(site.sheet(d) == sheet and site.held(d) == sheet.mem for d in site.units)
    assert site.tiers(HOST) == {"hbm": count * sheet.mem} | {
        tier: sum(int(cards.number(leaf["bytes"])) for leaf in leaves if leaf["tier"] == tier) for tier in ("dram", "disk")}
    assert site.links == tuple(Link(f"{HOST}/{name}", (HOST, OUTSIDE), math.ceil(cards.number(port["rate"]) * TICK), 0)
                               for name, port in node.get("interface", {}).items())
    assert site.boundary(HOST) == list(site.links) and not any(site.certified(k) for k in site.links)


@pytest.mark.parametrize("kind", sorted(set(READ) - set(WIRED)))
def test_onsite_refuses_a_node_with_no_interface_on_record_until_a_challenger_is_registered(kind: str) -> None:
    """A file with no `interface` table registers no link, so `onsite` has no way out of the node's to refuse: its
    entry is the one it gave before any file stated an interface. No node file has a memory challenger's row, so no
    challenger reaches the host that holds the node's memory; with one of the auditor's linked to the host, the entry
    is accepted, and it says nothing of how the node reaches the outside."""
    log = log_of(kind, "onsite")
    assert log.entries[0].links == ()
    with pytest.raises(Refused) as refused:
        check(log, auditor())
    assert refused.value.name == "challenger-reach" and HOST in refused.value.detail
    site = check(Registration("onsite", (reached(log.entries[0]),)), auditor(**{CHALLENGER: "dummy-key"}))
    assert site.units == check(log_of(kind, "remote"), auditor()).units and site.boundary() == []
    with pytest.raises(Malformed, match="states no interface, so a certifier's links have no rate"):
        log_of(kind, "onsite", CERTIFIER)


@pytest.mark.parametrize("certifier", [None, CERTIFIER])
@pytest.mark.parametrize("kind", WIRED)
def test_onsite_refuses_a_nodes_own_interface(kind: str, certifier: tuple[str, str] | None) -> None:
    """The node's interface is a boundary link that no certifier splits, so `onsite` refuses the node by `uncertified`,
    before and once a challenger reaches the host, and with a network certifier standing in front of the node as
    without one: the certifier's links are registered beside the interface's, which stays."""
    keys = {CHALLENGER: "dummy-key", **({} if certifier is None else {certifier[0]: certifier[1]})}
    entry = log_of(kind, "onsite", certifier).entries[0]
    bare = sorted(f"{HOST}/{name}" for name in nodes.load(kind)["interface"])
    for e in (entry, reached(entry)):
        with pytest.raises(Refused) as refused:
            check(Registration("onsite", (e,)), auditor(**keys))
        assert refused.value.name == "uncertified"
        assert sorted(re.findall(rf"{HOST}/\w+", refused.value.detail)) == bare


@pytest.mark.parametrize("kind", WIRED)
def test_a_certifier_links_to_one_host_and_the_outside(kind: str) -> None:
    """The stand-in's two links are the Site's shape of a certified way out, at the interfaces' rates summed. A
    certifier with its link to the outside missing, or linked to a second host, is refused by `kind`."""
    log = log_of(kind, "remote", CERTIFIER)
    cert, entry = CERTIFIER[0], log.entries[0]
    beta = math.ceil(sum(cards.number(port["rate"]) for port in nodes.load(kind)["interface"].values()) * TICK)
    assert entry.links[-2:] == (Link(f"{HOST}/{cert}", (HOST, cert), beta, 0),
                                Link(f"{cert}/{OUTSIDE}", (cert, OUTSIDE), beta, 0))
    assert check(log, auditor(**dict([CERTIFIER]))).far(cert) == [HOST, OUTSIDE]
    blind = replace(entry, links=entry.links[:-1])
    second = replace(entry, locs=(*entry.locs, Loc("host1", "host", "site")),
                     links=(*entry.links, Link(f"host1/{cert}", ("host1", cert), beta, 0)))
    for e in (blind, second):
        assert refusal(Registration("remote"), e, auditor(**dict([CERTIFIER]))) == "kind"


@pytest.mark.parametrize("kind", WIRED)
def test_only_onsites_uncertified_refuses_a_host_with_a_bare_link_beside_its_certifier(kind: str) -> None:
    """No rule of the Site refuses a host that has a bare link out beside a certified one: `remote` accepts the entry
    with both. `onsite` refuses it by `uncertified`, and accepts the same entry without the bare links, so the bare
    link is all it refuses."""
    keys = {CHALLENGER: "dummy-key", **dict([CERTIFIER])}
    site = check(log_of(kind, "remote", CERTIFIER), auditor(**keys))
    assert [site.certified(k) for k in site.boundary(HOST)] == [False] * len(nodes.load(kind)["interface"]) + [True]
    entry = reached(log_of(kind, "onsite", CERTIFIER).entries[0])
    assert refusal(Registration("onsite"), entry, auditor(**keys)) == "uncertified"
    certified = replace(entry, links=tuple(k for k in entry.links if OUTSIDE not in k.ends or CERTIFIER[0] in k.ends))
    assert refusal(Registration("onsite"), certified, auditor(**keys)) is None


@pytest.mark.parametrize("kind", WIRED)
@pytest.mark.parametrize("held", [{}, {CERTIFIER[0]: "dummy-another-key"}, {"certifier1": CERTIFIER[1]}])
def test_a_certifiers_key_is_the_one_the_auditor_holds_for_its_id(kind: str, held: dict[str, str]) -> None:
    """An auditor that holds no key for the certifier's id, another key for it, or the key under another id refuses the
    entry by `owner`; with the caller's key under the caller's id it is accepted."""
    log = log_of(kind, "remote", CERTIFIER)
    with pytest.raises(Refused) as refused:
        check(log, auditor(**held))
    assert refused.value.name == "owner" and CERTIFIER[0] in refused.value.detail
    assert check(log, auditor(**dict([CERTIFIER]))).registered(CERTIFIER[0])


def test_the_quoted_bytes() -> None:
    """Each node's tiers, so a changed entry is a changed number here."""
    assert check(log_of("rtx-pro-6000-server-x8", "remote"), auditor()).tiers(HOST) == {
        "hbm": 8 * 103_079_215_104, "dram": 1_872_605_741_056, "disk": 274_877_906_944 + 1_048_576 + 5_991_479_377_920}
    assert check(log_of("h200-sxm-141gb-x1", "remote"), auditor()).tiers(HOST) == {
        "hbm": 154_618_822_656, "dram": 214_748_364_800, "disk": 274_877_906_944 + 1_048_576 + 499_289_948_160}


def test_the_quoted_rates() -> None:
    """Each node's own way out, in bits a tick of a nanosecond: the eight-card node's one interface at 400 Gbit/s."""
    assert [(k.id, k.beta, k.latency) for k in check(log_of("rtx-pro-6000-server-x8", "remote"), auditor()).links] == [
        (f"{HOST}/eth0", 400, 0)]
    assert check(log_of("h200-sxm-141gb-x1", "remote"), auditor()).links == ()


def test_an_unknown_node_names_the_nodes() -> None:
    with pytest.raises(KeyError, match="h200-sxm-141gb-x1.*rtx-pro-6000-server-x8"):
        nodes.load("rtx-pro-6000-server")


COUNT = 'count = { value = 2, unit = "card", mark = "measurement", source = "s (art:0123456789ab)" }'
DEVICE = f'[device]\nsku = "h200-sxm-141gb"\n{COUNT}\n'
HOST_TABLE = '[host]\nsku = "xeon-platinum-8468-2s"\n'


@pytest.mark.parametrize(("body", "refusal"), [
    (f'node = "y"\n{DEVICE}{HOST_TABLE}', "set it to the file's name"),
    (f'node = "x"\n[device]\nsku = "h200-sxm-141gb"\ncount = 2\n{HOST_TABLE}', "device: count is a bare number"),
    (f'node = "x"\n{DEVICE}{HOST_TABLE}[memory.dram]\ntier = "dram"\nbytes = {{ value = 1, unit = "B" }}\n',
     "an entry has exactly"),
    (f'node = "x"\n[device]\nsku = "xeon-platinum-8468-2s"\n{COUNT}\n{HOST_TABLE}', "device.sku = .*name a gpu card"),
    (f'node = "x"\n{DEVICE}[host]\nsku = "h200-sxm-141gb"\n', "host.sku = .*name a host card"),
    (f'node = "x"\n{DEVICE}', "host.sku = None"),
])
def test_a_node_file_that_breaks_a_rule_is_refused(tmp_path, monkeypatch, body: str, refusal: str) -> None:
    (tmp_path / "x.toml").write_text(body, encoding="utf-8")
    monkeypatch.setattr(nodes, "NODES", tmp_path)
    with pytest.raises(InvalidArtifact, match=refusal):
        nodes.load("x")


def test_a_node_file_within_the_rules_loads(tmp_path, monkeypatch) -> None:
    """The bodies above differ from this one in the one thing each has wrong."""
    (tmp_path / "x.toml").write_text(f'node = "x"\n{DEVICE}{HOST_TABLE}', encoding="utf-8")
    monkeypatch.setattr(nodes, "NODES", tmp_path)
    assert nodes.load("x")["device"]["count"]["value"] == 2 and nodes.kinds() == ["x"]


DRAM = '[memory.dram]\ntier = "dram"\nbytes = { value = 1, unit = "B", mark = "measurement", source = "s (art:0123456789ab)" }\n'


@pytest.mark.parametrize(("leaves", "refusal"), [
    ("", "no dram and no disk leaf.*lsmem -b --summary=only; lsblk -b -d -n -o NAME,SIZE"),
    (DRAM, "no disk leaf"),
])
@pytest.mark.parametrize("composite", ["remote", "onsite"])
def test_a_node_file_without_its_hosts_memory_gives_no_entry(tmp_path, monkeypatch, composite: str, leaves: str,
                                                             refusal: str) -> None:
    """A file may state the card, the count and the host card alone, before anything has read its host: it loads. An
    entry of it would pass `check` for `remote` with a host holding less than the node has, so the builder refuses it
    and says what to read."""
    (tmp_path / "x.toml").write_text(f'node = "x"\n{DEVICE}{HOST_TABLE}{leaves}', encoding="utf-8")
    monkeypatch.setattr(nodes, "NODES", tmp_path)
    with pytest.raises(Malformed, match=refusal):
        log_of("x", composite)
