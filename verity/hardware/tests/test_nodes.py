"""The node files: every number an entry under the cards' rules, each naming a GPU card and a host card and no
machine, address or provider. A node whose file states its host's DRAM and disks gives a log accepted for `remote`
against the auditor the card files give, and refused for `onsite` until a memory challenger is registered; a file
without them gives no entry."""

from __future__ import annotations

import re
from dataclasses import replace

import pytest

from verity.errors import InvalidArtifact
from verity.hardware import cards, nodes
from verity.hardware.cards import MARKS
from verity.hardware.site import TIERS, Auditor, Link, Loc, Malformed, Refused, Registration, Sheet, check

HOST = "host0"
"""The id a test registers a node's host under: a node file names none."""
CHALLENGER = "challenger0"
READ = [kind for kind in nodes.kinds()
        if {"dram", "disk"} <= {leaf["tier"] for leaf in nodes.load(kind).get("memory", {}).values()}]
"""The nodes whose file states the host's DRAM and a disk: the ones an entry can be built of."""
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


def log_of(kind: str, composite: str) -> Registration:
    """The composite's log of one entry, the node's."""
    node = nodes.load(kind)
    log = Registration(composite)
    return log.append(log.node_entry(node, cards.load(node["device"]["sku"]), HOST))


def test_the_nodes_we_run() -> None:
    """Each states its host's DRAM and disks, so the acceptance tests below hold each one's log."""
    assert nodes.kinds() == READ == ["h200-sxm-141gb-x1", "rtx-pro-6000-server-x8"]


@pytest.mark.parametrize("kind", nodes.kinds())
def test_a_node_file_states_what_no_card_does(kind: str) -> None:
    """Its kind is its card and count; it names a GPU card and a host card; and its only numbers are the count and each
    memory leaf's bytes, in bytes and whole, under a tier of the host's own."""
    node = nodes.load(kind)
    device, count = node["device"], cards.number(node["device"]["count"])
    assert kind == f"{device['sku']}-x{count}" and count >= 1
    assert cards.load(device["sku"])["kind"] == "gpu" and cards.load(node["host"]["sku"])["kind"] == "host"
    leaves = node.get("memory", {})
    assert sorted(entries(node)) == sorted(["device.count", *(f"memory.{name}.bytes" for name in leaves)])
    for name, leaf in leaves.items():
        assert sorted(leaf) == ["bytes", "tier"] and leaf["tier"] in set(TIERS) - {"hbm"}, name
        assert leaf["bytes"]["unit"] == "B" and cards.number(leaf["bytes"]) > 0, name


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
    device memory, whole, and the host's leaves at the file's bytes, by tier."""
    node = nodes.load(kind)
    sheet = Sheet.of_file(cards.load(node["device"]["sku"]))
    count, leaves = int(cards.number(node["device"]["count"])), node.get("memory", {}).values()
    site = check(log_of(kind, "remote"), auditor())
    assert site.units == tuple(f"{HOST}/gpu{i}" for i in range(count))
    assert all(site.sheet(d) == sheet and site.held(d) == sheet.mem for d in site.units)
    assert site.tiers(HOST) == {"hbm": count * sheet.mem} | {
        tier: sum(int(cards.number(leaf["bytes"])) for leaf in leaves if leaf["tier"] == tier) for tier in ("dram", "disk")}
    assert site.links == () and site.boundary() == []


@pytest.mark.parametrize("kind", READ)
def test_onsite_refuses_a_nodes_log_until_a_challenger_is_registered(kind: str) -> None:
    """No node file has a memory challenger's row, so no challenger reaches the host that holds the node's memory. With
    one of the auditor's linked to the host, the same entry is accepted."""
    log = log_of(kind, "onsite")
    with pytest.raises(Refused) as refused:
        check(log, auditor())
    assert refused.value.name == "challenger-reach" and HOST in refused.value.detail
    entry = log.entries[0]
    reached = replace(entry, locs=(*entry.locs, Loc(CHALLENGER, "challenger", "site", "auditor", key="dummy-key")),
                      links=(Link(f"{CHALLENGER}-link", (CHALLENGER, HOST), 1, 1),))
    assert check(Registration("onsite", (reached,)), auditor(**{CHALLENGER: "dummy-key"})).units == check(
        log_of(kind, "remote"), auditor()).units


def test_the_quoted_bytes() -> None:
    """Each node's tiers, so a changed entry is a changed number here."""
    assert check(log_of("rtx-pro-6000-server-x8", "remote"), auditor()).tiers(HOST) == {
        "hbm": 8 * 103_079_215_104, "dram": 1_872_605_741_056, "disk": 274_877_906_944 + 1_048_576 + 5_991_479_377_920}
    assert check(log_of("h200-sxm-141gb-x1", "remote"), auditor()).tiers(HOST) == {
        "hbm": 151_397_597_184, "dram": 214_748_364_800, "disk": 274_877_906_944 + 1_048_576 + 499_289_948_160}


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
