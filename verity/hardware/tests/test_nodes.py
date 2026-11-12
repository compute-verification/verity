"""The node files: every number an entry under the cards' rules, each naming a GPU card and a host card and no
machine, address or provider. Every memory leaf states the most its bytes are read back at, a ceiling, which
`nodes.leaves` gives in whole bits a second beside the rate that bounds a tier's leaves together. `nodes.ways_out`
gives a node's ways out: its interfaces, each with its line rate or with none where it reads none, and its storage
path, at the rate that bounds its disks together. A node whose file states its host's DRAM and disks and an interface
gives a log accepted for `remote` against the auditor the card files give; a file without them gives no entry. Every
entry holds a link for each way out, none of which a certifier splits. The storage path's link and that of an interface
that reads no line rate say why (`Link.unmediated`), so neither node's site is mediated; an interface with a rate says
nothing, so `onsite` refuses the eight-card node by `uncertified`, with a network certifier standing in front of the
node or without one, and takes the H200 node once a challenger reaches its host."""

from __future__ import annotations

import math
import re
from dataclasses import replace

import pytest

from verity.errors import InvalidArtifact
from verity.hardware import cards, nodes
from verity.hardware.cards import MARKS
from verity.hardware.site import (NO_LINE_RATE, OUTSIDE, STORAGE_PATH, TIERS, Auditor, Entry, Link, Loc, Malformed,
                                  Refused, Registration, Sheet, check, refusal)

HOST = "host0"
"""The id a test registers a node's host under: a node file names none."""
CHALLENGER = "challenger0"
CERTIFIER = ("certifier0", "dummy-certifier0-key")
"""A network certifier's id and key, as whoever registers a node gives them: a node file names neither."""
READ = [kind for kind in nodes.kinds()
        if {"dram", "disk"} <= {leaf["tier"] for leaf in nodes.load(kind).get("memory", {}).values()}]
"""The nodes whose file states the host's DRAM and a disk: the ones an entry can be built of."""
WIRED = [kind for kind in READ if any(way.split and way.bw for way in nodes.ways_out(nodes.load(kind), "host0"))]
"""Those with an interface that states a line rate: the ones a network certifier can be registered in front of."""
ART = re.compile(r"\bart:[0-9a-f]{12}")
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
    return log.append(log.node_entry(node, cards.load(node["device"]["sku"]), HOST, certifier=certifier,
                                     ways=nodes.ways_out(node, HOST)))


def reached(entry: Entry) -> Entry:
    """`entry` with a memory challenger of the auditor's linked to the host."""
    return replace(entry, locs=(*entry.locs, Loc(CHALLENGER, "challenger", "site", "auditor", key="dummy-key")),
                   links=(*entry.links, Link(f"{CHALLENGER}-link", (CHALLENGER, HOST), 1, 1)))


def test_the_nodes_we_run() -> None:
    """Each states its host's DRAM and disks, so the acceptance tests below hold each one's log. Each states one
    interface: the eight-card node's reads a line rate, and the H200 node's reads none."""
    assert nodes.kinds() == READ == ["h200-sxm-141gb-x1", "rtx-pro-6000-server-x8"]
    assert WIRED == ["rtx-pro-6000-server-x8"]
    assert all(list(nodes.load(kind)["interface"]) == ["eth0"] for kind in READ)


@pytest.mark.parametrize("kind", nodes.kinds())
def test_a_node_file_states_what_no_card_does(kind: str) -> None:
    """Its kind is its card and count; it names a GPU card and a host card; and its only numbers are the count, each
    memory leaf's bytes, in bytes and whole, under a tier of the host's own, with the leaf's read rate, and the line
    rate, in bits a second, of each interface that reads one. An interface that reads none states its driver and the
    read, and no number. The storage path is stated once, naming every disk leaf, each a network disk on both nodes,
    with no rate of its own. A leaf, an interface and the storage path share no name, since each is registered as
    `host/NAME`."""
    node = nodes.load(kind)
    device, count = node["device"], cards.number(node["device"]["count"])
    assert kind == f"{device['sku']}-x{count}" and count >= 1
    assert cards.load(device["sku"])["kind"] == "gpu" and cards.load(node["host"]["sku"])["kind"] == "host"
    assert set(node) <= {"node", "name", "device", "host", "memory", "interface", "storage"}
    leaves, ports = node.get("memory", {}), node.get("interface", {})
    rated = [name for name, port in ports.items() if "rate" in port]
    assert sorted(entries(node)) == sorted(["device.count", *(f"memory.{name}.{key}" for name in leaves
                                                              for key in ("bytes", "rate")),
                                            *(f"interface.{name}.rate" for name in rated)])
    for name, leaf in leaves.items():
        assert sorted(leaf) == ["bytes", "rate", "tier"] and leaf["tier"] in set(TIERS) - {"hbm"}, name
        assert leaf["bytes"]["unit"] == "B" and cards.number(leaf["bytes"]) > 0, name
    for name, port in ports.items():
        if name in rated:
            assert sorted(port) == ["rate"] and port["rate"]["unit"] == "bit/s" and cards.number(port["rate"]) > 0, name
        else:
            assert sorted(port) == ["driver", "source"] and port["driver"].strip() and ART.search(port["source"]), name
    path = node["storage"]
    assert sorted(path) == ["leaves", "source"] and ART.search(path["source"])
    assert path["leaves"] == [name for name, leaf in leaves.items() if leaf["tier"] == "disk"]
    names = (*leaves, *ports, nodes.STORAGE)
    assert len(set(names)) == len(names) and not any(re.fullmatch(r"gpu\d+", name) for name in names)


@pytest.mark.parametrize("kind", nodes.kinds())
def test_every_number_is_sourced_as_a_cards_is(kind: str) -> None:
    """The cards' rules: a mark and a source on every number, evidence cited by its `art:` id, never by a run id, and
    a measurement citing the record of what measured it."""
    numbers = entries(nodes.load(kind))
    assert all(e["mark"] in MARKS and e["source"].strip() for e in numbers.values())
    text = (nodes.NODES / f"{kind}.toml").read_text(encoding="utf-8")
    assert not re.findall(r"\br\d{8}-\d{6}-[0-9a-f]{4}\b", text), \
        "cite the run's record (`research data show RUN`: outputs.run_record) as art:<12 hex>"
    uncited = sorted(key for key, e in numbers.items() if e["mark"] == "measurement" and not ART.search(e["source"]))
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
    device memory, whole, and the host's leaves at the file's bytes, by tier. Its links are the file's ways out, each
    from the host to the outside at its rate in bits a second: an interface at the line rate its table states, or at
    none, saying so, where it reads none, then the storage path at the rate that bounds the node's disks together,
    saying it is the storage path. They are boundary links, none of them certified, and neither site is mediated."""
    node = nodes.load(kind)
    sheet = Sheet.of_file(cards.load(node["device"]["sku"]))
    count, leaves = int(cards.number(node["device"]["count"])), node.get("memory", {}).values()
    site = check(log_of(kind, "remote"), auditor())
    assert site.units == tuple(f"{HOST}/gpu{i}" for i in range(count))
    assert all(site.sheet(d) == sheet and site.held(d) == sheet.mem for d in site.units)
    assert site.tiers(HOST) == {"hbm": count * sheet.mem} | {
        tier: sum(int(cards.number(leaf["bytes"])) for leaf in leaves if leaf["tier"] == tier) for tier in ("dram", "disk")}
    ports = [Link(f"{HOST}/{name}", (HOST, OUTSIDE), math.ceil(cards.number(port["rate"])), 0) if "rate" in port
             else Link(f"{HOST}/{name}", (HOST, OUTSIDE), None, 0, NO_LINE_RATE)
             for name, port in node["interface"].items()]
    path = Link(f"{HOST}/storage", (HOST, OUTSIDE), nodes.leaves(node, HOST)[1]["disk"], 0, STORAGE_PATH)
    assert site.links == (*ports, path) and len(ports) == 1
    assert site.boundary(HOST) == list(site.links) and not any(site.certified(k) for k in site.links)
    assert not site.mediated


@pytest.mark.parametrize("kind", sorted(set(READ) - set(WIRED)))
def test_a_node_whose_interface_reads_no_rate_says_why_no_certifier_splits_it(kind: str) -> None:
    """The H200 node's one interface reads no line rate, so its link states no capacity and says so, and the storage
    path's link says it is the storage path. Each says why no certifier splits it, so `onsite` refuses neither: it
    refuses the node by `challenger-reach` alone, and accepts it once a challenger reaches the host, with a site that
    is not mediated. A network certifier named on the node is refused under either composite: its links would have
    no rate."""
    assert [way.bw for way in nodes.ways_out(nodes.load(kind), HOST) if way.split] == [None]
    entry = log_of(kind, "onsite").entries[0]
    assert [(k.id, k.beta, k.unmediated) for k in entry.links] == [
        (f"{HOST}/eth0", None, NO_LINE_RATE), (f"{HOST}/storage", 5_293_292_307_696, STORAGE_PATH)]
    keys = auditor(**{CHALLENGER: "dummy-key"})
    assert refusal(Registration("onsite"), entry, keys) == "challenger-reach"
    assert not check(Registration("onsite", (reached(entry),)), keys).mediated
    for composite in ("remote", "onsite"):
        with pytest.raises(Malformed, match=rf"no interface the node file states has a rate \(\['{HOST}/eth0'\]\).*"
                                            "takes no certifier"):
            log_of(kind, composite, CERTIFIER)


@pytest.mark.parametrize("certifier", [None, CERTIFIER])
@pytest.mark.parametrize("kind", WIRED)
def test_onsite_refuses_a_nodes_own_ways_out(kind: str, certifier: tuple[str, str] | None) -> None:
    """The node's interface is a boundary link that no certifier splits and that says no reason, so `onsite` refuses
    the node by `uncertified`, naming it and not the storage path, which says why, before and once a challenger
    reaches the host, and with a network certifier standing in front of the node as without one: the certifier's
    links are registered beside the node's own, which stay."""
    keys = {CHALLENGER: "dummy-key", **({} if certifier is None else {certifier[0]: certifier[1]})}
    entry = log_of(kind, "onsite", certifier).entries[0]
    bare = [f"{HOST}/{name}" for name in nodes.load(kind)["interface"]]
    for e in (entry, reached(entry)):
        with pytest.raises(Refused) as refused:
            check(Registration("onsite", (e,)), auditor(**keys))
        assert refused.value.name == "uncertified"
        assert sorted(re.findall(rf"{HOST}/\w+", refused.value.detail)) == bare


@pytest.mark.parametrize("kind", WIRED)
def test_a_certifier_links_to_one_host_and_the_outside(kind: str) -> None:
    """The stand-in's two links are the Site's shape of a certified way out, at the interfaces' rates summed: the
    storage path's rate is not in the sum, and no certifier is at an end of its link. A certifier with its link to
    the outside missing, or linked to a second host, is refused by `kind`."""
    log = log_of(kind, "remote", CERTIFIER)
    cert, entry = CERTIFIER[0], log.entries[0]
    beta = sum(way.bw for way in nodes.ways_out(nodes.load(kind), HOST) if way.split)
    assert entry.links[-2:] == (Link(f"{HOST}/{cert}", (HOST, cert), beta, 0),
                                Link(f"{cert}/{OUTSIDE}", (cert, OUTSIDE), beta, 0))
    assert check(log, auditor(**dict([CERTIFIER]))).far(cert) == [HOST, OUTSIDE]
    assert [k.id for k in entry.links if cert in k.ends] == [f"{HOST}/{cert}", f"{cert}/{OUTSIDE}"]
    blind = replace(entry, links=entry.links[:-1])
    second = replace(entry, locs=(*entry.locs, Loc("host1", "host", "site")),
                     links=(*entry.links, Link(f"host1/{cert}", ("host1", cert), beta, 0)))
    for e in (blind, second):
        assert refusal(Registration("remote"), e, auditor(**dict([CERTIFIER]))) == "kind"


@pytest.mark.parametrize("kind", WIRED)
def test_only_onsites_uncertified_refuses_a_host_with_a_bare_link_beside_its_certifier(kind: str) -> None:
    """No rule of the Site refuses a host that has a bare link out beside a certified one: `remote` accepts the entry
    with both. `onsite` refuses it by `uncertified`. With a certifier named and the interface's bare link taken out,
    it accepts the eight-card node, storage path and all, since that link says why no certifier splits it: a
    certifier on the interface does not mediate the disks, so the site is not mediated. Without the path's link too,
    it is."""
    keys = {CHALLENGER: "dummy-key", **dict([CERTIFIER])}
    ports = [f"{HOST}/{name}" for name in nodes.load(kind)["interface"]]
    site = check(log_of(kind, "remote", CERTIFIER), auditor(**keys))
    assert [(k.id, site.certified(k)) for k in site.boundary(HOST)] == [
        *((port, False) for port in ports), (f"{HOST}/storage", False), (f"{HOST}/{CERTIFIER[0]}", True)]
    entry = reached(log_of(kind, "onsite", CERTIFIER).entries[0])
    assert refusal(Registration("onsite"), entry, auditor(**keys)) == "uncertified"
    stored = replace(entry, links=tuple(k for k in entry.links if k.id not in ports))
    assert not check(Registration("onsite", (stored,)), auditor(**keys)).mediated
    certified = replace(entry, links=tuple(k for k in entry.links if OUTSIDE not in k.ends or CERTIFIER[0] in k.ends))
    assert check(Registration("onsite", (certified,)), auditor(**keys)).mediated


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
        "hbm": 151_397_597_184, "dram": 214_748_364_800, "disk": 274_877_906_944 + 1_048_576 + 499_289_948_160}


def test_the_quoted_rates() -> None:
    """Each node's links out, in bits a second: the eight-card node's one interface at 400 Gbit/s and its storage path
    at the 5,584,738,461,544 bit/s that bound its disks together, and the H200 node's interface at none, since it
    reads no rate, and its storage path at 5,293,292,307,696 bit/s. The storage paths and the interface that reads no
    rate say why no certifier splits them."""
    assert check(log_of("rtx-pro-6000-server-x8", "remote"), auditor()).links == (
        Link(f"{HOST}/eth0", (HOST, OUTSIDE), 400_000_000_000, 0),
        Link(f"{HOST}/storage", (HOST, OUTSIDE), 5_584_738_461_544, 0, "storage path"))
    assert check(log_of("h200-sxm-141gb-x1", "remote"), auditor()).links == (
        Link(f"{HOST}/eth0", (HOST, OUTSIDE), None, 0, "no line rate stated"),
        Link(f"{HOST}/storage", (HOST, OUTSIDE), 5_293_292_307_696, 0, "storage path"))


def test_the_quoted_ways_out() -> None:
    """Each node's ways out as `nodes.ways_out` gives them, so a changed table is a changed line here: the interface,
    which a certifier may split, with its line rate in whole bits a second or with none, then the storage path, which
    none may, at the one rate for the node's disks together that `nodes.leaves` returns."""
    assert nodes.ways_out(nodes.load("rtx-pro-6000-server-x8"), HOST) == (
        nodes.Way(f"{HOST}/eth0", 400_000_000_000, True), nodes.Way(f"{HOST}/storage", 5_584_738_461_544, False))
    assert nodes.ways_out(nodes.load("h200-sxm-141gb-x1"), HOST) == (
        nodes.Way(f"{HOST}/eth0", None, True), nodes.Way(f"{HOST}/storage", 5_293_292_307_696, False))
    for kind in nodes.kinds():
        node = nodes.load(kind)
        assert nodes.ways_out(node, HOST)[-1].bw == nodes.leaves(node, HOST)[1]["disk"]
        assert "rate" not in node["storage"]


@pytest.mark.parametrize("kind", nodes.kinds())
def test_every_leaf_states_a_ceiling_on_its_read(kind: str) -> None:
    """Every memory leaf of every node file states a rate in bytes a second, above 0 and marked `ceiling`: the Site's
    check of a leaf's rate refuses 0, and a measured rate or a limit somebody sets bounds nothing. The DRAM leaf's is
    its host card's `memory.bandwidth_max`, and a disk's is at most its host card's `disk.read_max`, the lanes every
    disk's bytes enter the server by."""
    node = nodes.load(kind)
    host = cards.load(node["host"]["sku"])
    assert node["memory"]
    for name, leaf in node["memory"].items():
        rate = leaf["rate"]
        assert (rate["unit"], rate["mark"]) == ("B/s", "ceiling") and cards.number(rate) > 0, name
        bound = host["memory"]["bandwidth_max"] if leaf["tier"] == "dram" else host["disk"]["read_max"]
        assert bound["mark"] == "ceiling" and cards.number(rate) <= cards.number(bound), name
        assert leaf["tier"] != "dram" or cards.number(rate) == cards.number(bound), name


def test_the_quoted_leaf_rates() -> None:
    """Each node's leaves as `nodes.leaves` gives them, in the file's order, and the rate of each tier's leaves
    together, in whole bits a second, so a changed entry is a changed number here. Each disk states its host's lanes:
    2 x (88 lanes at 32 GT/s and four USB ports at 5 GT/s) on the eight-card node's host, and 2 x (80 lanes at
    32 GT/s and a chipset link of 8 at 16 GT/s) on the H200 node's, less the coding, rounded up to a whole byte a
    second."""
    assert nodes.leaves(nodes.load("rtx-pro-6000-server-x8"), HOST) == ((
        nodes.Leaf(f"{HOST}/dram", "dram", 1_872_605_741_056, 9_011_200_000_000),
        nodes.Leaf(f"{HOST}/vda", "disk", 274_877_906_944, 5_584_738_461_544),
        nodes.Leaf(f"{HOST}/vdb", "disk", 1_048_576, 5_584_738_461_544),
        nodes.Leaf(f"{HOST}/vdc", "disk", 5_991_479_377_920, 5_584_738_461_544),
    ), {"dram": 9_011_200_000_000, "disk": 5_584_738_461_544})
    assert nodes.leaves(nodes.load("h200-sxm-141gb-x1"), HOST) == ((
        nodes.Leaf(f"{HOST}/dram", "dram", 214_748_364_800, 4_915_200_000_000),
        nodes.Leaf(f"{HOST}/vda", "disk", 274_877_906_944, 5_293_292_307_696),
        nodes.Leaf(f"{HOST}/vdb", "disk", 1_048_576, 5_293_292_307_696),
        nodes.Leaf(f"{HOST}/vdc", "disk", 499_289_948_160, 5_293_292_307_696),
    ), {"dram": 4_915_200_000_000, "disk": 5_293_292_307_696})
    assert 5_584_738_461_544 == 8 * math.ceil(2 * (88 * 32 + 4 * 5) * 10 ** 9 * 128 / 130 / 8)
    assert 5_293_292_307_696 == 8 * math.ceil(2 * (80 * 32 + 8 * 16) * 10 ** 9 * 128 / 130 / 8)


@pytest.mark.parametrize("kind", READ)
def test_the_leaves_are_the_ones_the_entry_registers(kind: str) -> None:
    """`nodes.leaves` and the Site's entry read one file: the same leaves under the host, by id, tier and bytes, in
    the same order, and each tier's rate is the host card's ceiling in bits."""
    node = nodes.load(kind)
    each, together = nodes.leaves(node, HOST)
    site = check(log_of(kind, "remote"), auditor())
    registered = [(loc.id, loc.tier, loc.bytes) for loc in site.locs if loc.kind == "memory" and loc.parent == HOST]
    assert [(leaf.id, leaf.tier, leaf.bytes) for leaf in each] == registered
    host = cards.load(node["host"]["sku"])
    assert together == {"dram": math.ceil(8 * cards.number(host["memory"]["bandwidth_max"])),
                        "disk": math.ceil(8 * cards.number(host["disk"]["read_max"]))}
    assert all(0 < leaf.bw <= together[leaf.tier] for leaf in each)


def test_a_ceiling_is_no_less_than_the_measured_rate() -> None:
    """Where a host card has a rate a run measured, the node's ceilings are at or above it: every disk leaf's rate
    and the disks' together are above the disks' measured read, which sums the disks, and the DRAM leaf's is above
    DRAM's measured read. The eight-card node has both on record."""
    seen = []
    for kind in nodes.kinds():
        node = nodes.load(kind)
        host = cards.load(node["host"]["sku"])
        each, together = nodes.leaves(node, HOST)
        for tier, (table, key) in {"dram": ("memory", "read"), "disk": ("disk", "read")}.items():
            measured = host.get(table, {}).get(key)
            if measured is None or measured["mark"] != "measurement":
                continue
            assert measured["unit"] == "B/s"
            bits = 8 * cards.number(measured)
            assert 0 < bits <= together[tier] and all(bits <= leaf.bw for leaf in each if leaf.tier == tier), (kind, tier)
            seen.append((kind, tier))
    assert seen == [("rtx-pro-6000-server-x8", "dram"), ("rtx-pro-6000-server-x8", "disk")]


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


def leaf(name: str, tier: str, rate: str | None) -> str:
    """A leaf's table of one byte, with this text as its rate."""
    return (f'[memory.{name}]\ntier = "{tier}"\n{DRAM.splitlines()[2]}\n' + (f"rate = {rate}\n" if rate else ""))


def ceiling(value: float | int, unit: str = "B/s", mark: str = "ceiling") -> str:
    return f'{{ value = {value}, unit = "{unit}", mark = "{mark}", source = "s (art:0123456789ab)" }}'


HOST_DRAM = 6.144e11
"""`memory.bandwidth_max` of the host card `HOST_TABLE` names, in B/s."""


def test_a_drive_of_the_servers_own_states_its_link(tmp_path, monkeypatch) -> None:
    """A leaf's rate is the leaf's: a drive behind its own link (four lanes at 16 GT/s, less the coding) states it,
    under its host's lanes, and the tier's rate beside it stays the host's. A rate is rounded up to a whole bit."""
    drive = math.ceil(4 * 16 * 10 ** 9 * 128 / 130 / 8)
    body = (f'node = "x"\n{DEVICE}{HOST_TABLE}{leaf("dram", "dram", ceiling(HOST_DRAM))}'
            f'{leaf("nvme0n1", "disk", ceiling(drive))}{leaf("vda", "disk", ceiling(0.3))}')
    (tmp_path / "x.toml").write_text(body, encoding="utf-8")
    monkeypatch.setattr(nodes, "NODES", tmp_path)
    each, together = nodes.leaves(nodes.load("x"), HOST)
    assert each == (nodes.Leaf(f"{HOST}/dram", "dram", 1, 4_915_200_000_000),
                    nodes.Leaf(f"{HOST}/nvme0n1", "disk", 1, 8 * drive), nodes.Leaf(f"{HOST}/vda", "disk", 1, 3))
    assert together == {"dram": 4_915_200_000_000, "disk": 5_293_292_307_696} and 8 * drive < together["disk"]


def test_the_host_card_given_is_the_one_read(tmp_path, monkeypatch) -> None:
    """A caller that holds the host card's table gives it, and the tiers' rates are that table's."""
    (tmp_path / "x.toml").write_text(f'node = "x"\n{DEVICE}{HOST_TABLE}{leaf("dram", "dram", ceiling(16))}'
                                     f'{leaf("vda", "disk", ceiling(2))}', encoding="utf-8")
    monkeypatch.setattr(nodes, "NODES", tmp_path)
    entry = {"unit": "B/s", "mark": "ceiling", "source": "s"}
    card = {"memory": {"bandwidth_max": entry | {"value": 16}}, "disk": {"read_max": entry | {"value": 2.5}}}
    each, together = nodes.leaves(nodes.load("x"), HOST, card)
    assert [leaf.bw for leaf in each] == [128, 16] and together == {"dram": 128, "disk": 20}
    with pytest.raises(InvalidArtifact, match="the host card has memory.bandwidth_max and disk.read_max"):
        nodes.leaves(nodes.load("x"), HOST, {"memory": card["memory"]})
    with pytest.raises(InvalidArtifact, match="the host card's disk.read_max is in 'B/s', marked 'measurement'"):
        nodes.leaves(nodes.load("x"), HOST, card | {"disk": {"read_max": entry | {"value": 2.5, "mark": "measurement"}}})


@pytest.mark.parametrize(("leaves", "refusal"), [
    (leaf("dram", "dram", None), "memory.dram states no rate.*state memory.dram.rate.*memory.bandwidth_max"),
    (leaf("dram", "dram", ceiling(HOST_DRAM)) + leaf("vda", "disk", None),
     "memory.vda states no rate.*disk.read_max, or for a drive of the server's own its link's lanes"),
    (leaf("vda", "disk", ceiling(4.4e9, mark="measurement")), "marked 'measurement'.*no ceiling"),
    (leaf("vda", "disk", ceiling(4.4e9, mark="placeholder")), "marked 'placeholder'"),
    (leaf("vda", "disk", ceiling(4.4e9, unit="bit/s")), "memory.vda.rate is in 'bit/s'.*unit = \"B/s\""),
    (leaf("vdb", "disk", ceiling(0)), "memory.vdb.rate is 0.*above 0"),
    (leaf("vda", "disk", ceiling(661661538463)), "memory.vda.rate is 5293292307704 bit/s.*disk.read_max.*5293292307696"),
    (leaf("dram", "dram", ceiling(HOST_DRAM - 1)), "memory.dram.rate is .*memory.bandwidth_max.*state that figure"),
    (leaf("dram", "dram", ceiling(HOST_DRAM + 1)), "memory.dram.rate is .*memory.bandwidth_max.*state that figure"),
    (leaf("hbm", "hbm", ceiling(1)), "memory.hbm.tier is 'hbm'.*one of \\['disk', 'dram'\\]"),
    (leaf("vda", "disk", ceiling(1)).replace("value = 1, unit = \"B\"", "value = 1.5, unit = \"B\""),
     "memory.vda.bytes is 1.5; state a whole number of bytes"),
    (leaf("vda", "disk", ceiling(1)).replace("value = 1, unit = \"B\"", "value = 1, unit = \"GiB\""),
     "memory.vda.bytes is .*state it as an entry in B"),
])
def test_a_leaf_without_a_ceiling_on_its_read_is_refused(tmp_path, monkeypatch, leaves: str, refusal: str) -> None:
    """A leaf that states no rate, a rate that is no ceiling in bytes a second above 0, a rate above its host's for
    the tier, a DRAM rate that is not its host's, a tier that is no host's own, or a size that is no whole number of
    bytes gives no leaves: the file loads, as one that states no memory does, and the reader of its rates refuses it
    and says what to state."""
    (tmp_path / "x.toml").write_text(f'node = "x"\n{DEVICE}{HOST_TABLE}{leaves}', encoding="utf-8")
    monkeypatch.setattr(nodes, "NODES", tmp_path)
    with pytest.raises(InvalidArtifact, match=refusal):
        nodes.leaves(nodes.load("x"), HOST)


def test_a_file_with_no_memory_has_no_leaves(tmp_path, monkeypatch) -> None:
    """A file that states no leaf gives none, with its host's rates for the tiers: the entry's builder is what
    refuses a node whose host's memory nothing has read."""
    (tmp_path / "x.toml").write_text(f'node = "x"\n{DEVICE}{HOST_TABLE}', encoding="utf-8")
    monkeypatch.setattr(nodes, "NODES", tmp_path)
    assert nodes.leaves(nodes.load("x"), HOST) == ((), {"dram": 4_915_200_000_000, "disk": 5_293_292_307_696})


SOURCE = "s (art:0123456789ab)"
PORT = f'[interface.eth0]\nrate = {{ value = 25000000000, unit = "bit/s", mark = "measurement", source = "{SOURCE}" }}\n'
"""An interface that reads a line rate, 25 Gbit/s."""
UNRATED = f'[interface.eth0]\ndriver = "virtio_net"\nsource = "{SOURCE}"\n'
"""An interface that reads none: its driver and the read."""
PATH = f'[storage]\nleaves = ["vda"]\nsource = "{SOURCE}"\n'
"""The storage path of a file whose disk `vda` is not the server's own."""
DISKS = 5_293_292_307_696
"""The rate that bounds the disks together on the host card `HOST_TABLE` names, in bits a second."""


def body(*tables: str, disk: str = "vda") -> str:
    """A node file's text: two cards, the host, its DRAM and one disk, and these tables."""
    return (f'node = "x"\n{DEVICE}{HOST_TABLE}{leaf("dram", "dram", ceiling(HOST_DRAM))}'
            f'{leaf(disk, "disk", ceiling(2))}' + "".join(tables))


def test_a_server_whose_disks_are_its_own_has_no_storage_link(tmp_path, monkeypatch) -> None:
    """A file with no `storage` table has no storage path: its ways out are its interfaces, and its entry's links are
    theirs alone. With the table, the path is a way out after them, at the disks' rate together."""
    (tmp_path / "x.toml").write_text(body(PORT, disk="nvme0n1"), encoding="utf-8")
    monkeypatch.setattr(nodes, "NODES", tmp_path)
    assert nodes.ways_out(nodes.load("x"), HOST) == (nodes.Way(f"{HOST}/eth0", 25 * 10 ** 9, True),)
    assert log_of("x", "remote").entries[0].links == (Link(f"{HOST}/eth0", (HOST, OUTSIDE), 25 * 10 ** 9, 0),)
    (tmp_path / "x.toml").write_text(body(PORT, PATH), encoding="utf-8")
    assert nodes.ways_out(nodes.load("x"), HOST) == (nodes.Way(f"{HOST}/eth0", 25 * 10 ** 9, True),
                                                     nodes.Way(f"{HOST}/storage", DISKS, False))
    assert log_of("x", "remote").entries[0].links == (Link(f"{HOST}/eth0", (HOST, OUTSIDE), 25 * 10 ** 9, 0),
                                                      Link(f"{HOST}/storage", (HOST, OUTSIDE), DISKS, 0, STORAGE_PATH))


def test_the_storage_paths_rate_is_the_host_card_givens(tmp_path, monkeypatch) -> None:
    """A caller that holds the host card's table gives it, and the storage path's rate is that table's for the disks
    together: the file writes none."""
    (tmp_path / "x.toml").write_text(body(UNRATED, PATH), encoding="utf-8")
    monkeypatch.setattr(nodes, "NODES", tmp_path)
    entry = {"unit": "B/s", "mark": "ceiling", "source": "s"}
    card = {"memory": {"bandwidth_max": entry | {"value": HOST_DRAM}}, "disk": {"read_max": entry | {"value": 2.5}}}
    assert nodes.ways_out(nodes.load("x"), HOST, card) == (nodes.Way(f"{HOST}/eth0", None, True),
                                                           nodes.Way(f"{HOST}/storage", 20, False))


@pytest.mark.parametrize(("tables", "refusal"), [
    ((), "states no interface, so its entry would not say how the node reaches the outside.*ip route"),
    ((PATH,), "states no interface, so its entry would not say how the node reaches the outside"),
])
@pytest.mark.parametrize("composite", ["remote", "onsite"])
def test_a_node_file_with_no_link_out_gives_no_entry(tmp_path, monkeypatch, composite: str, tables: tuple[str, ...],
                                                     refusal: str) -> None:
    """An entry always registers a link out. A file that states no interface gives none, with a storage path or
    without one: its entry would not say how the node reaches the outside. Each loads, and the builder refuses it and
    says what to state. The same file with an interface gives an entry, one whose only way out has no rate too: its
    link states no capacity and says why no certifier splits it."""
    (tmp_path / "x.toml").write_text(body(*tables), encoding="utf-8")
    monkeypatch.setattr(nodes, "NODES", tmp_path)
    with pytest.raises(Malformed, match=refusal):
        log_of("x", composite)
    (tmp_path / "x.toml").write_text(body(UNRATED), encoding="utf-8")
    assert log_of("x", composite).entries[0].links == (Link(f"{HOST}/eth0", (HOST, OUTSIDE), None, 0, NO_LINE_RATE),)


RATE = {"value": 25 * 10 ** 9, "unit": "bit/s", "mark": "measurement", "source": SOURCE}


@pytest.mark.parametrize(("port", "refusal"), [
    ({}, "interface.eth0 is {}; state its rate alone.*or, where the interface reads none, its driver and the source"),
    ({"rate": 25 * 10 ** 9}, "interface.eth0.rate is 25000000000, not an entry"),
    ({"rate": RATE | {"value": 25, "unit": "Gbit/s"}}, "interface.eth0.rate is 25 Gbit/s.*in bit/s"),
    ({"rate": RATE | {"value": 3125000000, "unit": "B/s"}}, "interface.eth0.rate is 3125000000 B/s.*in bit/s"),
    ({"rate": RATE | {"value": 0}}, "interface.eth0.rate is 0 bit/s.*above 0"),
    ({"rate": RATE | {"value": -1}}, "interface.eth0.rate is -1 bit/s.*above 0"),
    ({"rate": RATE, "driver": "mlx5_core"}, "state its rate alone"),
    ({"rate": RATE, "source": SOURCE}, "state its rate alone"),
    ({"driver": "virtio_net"}, "its driver and the source of that read alone"),
    ({"source": SOURCE}, "its driver and the source of that read alone"),
    ({"driver": "virtio_net", "source": " "}, "its driver and the source of that read alone"),
    ({"driver": 7, "source": SOURCE}, "each a string"),
    ("virtio_net", "interface.eth0 is 'virtio_net'"),
])
def test_an_interface_states_its_rate_or_what_was_read_of_it(port: object, refusal: str) -> None:
    """An interface's table is its line rate, an entry in bits a second above 0, and nothing else, or, where the
    interface reads none, its driver and the source of that read, and nothing else: a table that is neither is
    refused by the reader of the ways out, which says what to state. A rate is rounded up to a whole bit."""
    with pytest.raises(InvalidArtifact, match=refusal):
        nodes.ways_out({"interface": {"eth0": port}}, HOST)
    assert nodes.ways_out({"interface": {"eth0": {"rate": RATE | {"value": "1000000000.5"}},
                                         "eth1": {"driver": "virtio_net", "source": SOURCE}}}, HOST) == (
        nodes.Way(f"{HOST}/eth0", 1_000_000_001, True), nodes.Way(f"{HOST}/eth1", None, True))


@pytest.mark.parametrize(("path", "more", "refusal"), [
    ('[storage]\nleaves = ["vda"]\n', "", "storage is .*state the storage path as leaves.*and source"),
    (f'[storage]\nsource = "{SOURCE}"\n', "", r"state the storage path as leaves.*disk leaves are \['vda'\]"),
    (f'[storage]\nleaves = []\nsource = "{SOURCE}"\n', "", r"storage.leaves is \[\]"),
    (f'[storage]\nleaves = "vda"\nsource = "{SOURCE}"\n', "", "storage.leaves is 'vda'"),
    (f'[storage]\nleaves = ["dram"]\nsource = "{SOURCE}"\n', "", r"storage.leaves is \['dram'\].*disk leaves"),
    (f'[storage]\nleaves = ["vdz"]\nsource = "{SOURCE}"\n', "", r"storage.leaves is \['vdz'\]"),
    (f'[storage]\nleaves = ["vda", "vda"]\nsource = "{SOURCE}"\n', "", r"storage.leaves is \['vda', 'vda'\]"),
    ('[storage]\nleaves = ["vda"]\nsource = ""\n', "", "storage.source is ''"),
    (f'{PATH}rate = {ceiling(2)}\n', "", "storage is .*state the storage path as leaves"),
    (PATH, leaf("storage", "disk", ceiling(2)), f"memory.storage would register as {HOST}/storage"),
    (PATH, PORT.replace("eth0", "storage"), f"interface.storage would register as {HOST}/storage"),
    ("", leaf("storage", "disk", ceiling(2)), f"memory.storage would register as {HOST}/storage"),
    ("", PORT.replace("eth0", "storage"), f"interface.storage would register as {HOST}/storage"),
    (PATH, leaf("vdb", "disk", None), "memory.vdb states no rate"),
])
def test_a_storage_path_that_breaks_a_rule_is_refused(tmp_path, monkeypatch, path: str, more: str,
                                                      refusal: str) -> None:
    """The storage path names, each once, disk leaves of the file's and says why they leave, and states nothing
    else: no rate of its own, since its rate is the disks' together. No leaf and no interface takes its name, in a
    file that states the path or in one that does not. Its rate is read through `nodes.leaves`, so a disk leaf that
    breaks a rule there refuses it too. The file loads, and the reader of the ways out refuses it and says what to
    state."""
    (tmp_path / "x.toml").write_text(body(PORT, more, path), encoding="utf-8")
    monkeypatch.setattr(nodes, "NODES", tmp_path)
    with pytest.raises(InvalidArtifact, match=refusal):
        nodes.ways_out(nodes.load("x"), HOST)
