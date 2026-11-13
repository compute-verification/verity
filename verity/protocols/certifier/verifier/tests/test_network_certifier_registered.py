"""The network certifier's link and key read from the registered hardware (`registered.py`): the accepting case, each
refusal by its message, a site that is no site, and the Python held to the Lean (`NetworkCheck.Registered`) on the same
vectors: the cases below and seeded sites around them, each answered by `tests/lean/RegisteredVectors.lean` in the
certifier check's Lake package and by the Python, which agree on the registered link, on each refusal's message and on
which sites are malformed."""

from __future__ import annotations

import copy
import fcntl
import json
import random
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

import pytest

from verity.protocols.certifier.verifier import registered as R

PACKAGE = Path(R.__file__).resolve().parent / "lean"
SCRIPT = Path(__file__).resolve().parent / "lean" / "RegisteredVectors.lean"
# `verity_difftest.build`'s lock: no two `lake build`s of a package overlap, across processes and suites.
BUILD_LOCK = Path(__file__).resolve().parents[5] / "verity/difftest/lean" / ".lake" / "harness-build.lock"

KEY0, KEY1 = "a" * 96, "b" * 96
LINKS = "the network certifiers' links there are [up0, up2]"


def loc(i: str, kind: str, parent: str | None = "site0", **more) -> dict:
    return {"bytes": 0, "id": i, "key": "", "kind": kind, "owner": "prover", "parent": parent, "sheet": "", "tier": None,
            **more}


def link(i: str, a: str, b: str, beta: int) -> dict:
    return {"beta": beta, "ends": [a, b], "id": i, "latency": 1}


def view_site() -> dict:
    """A site as a registration log's view writes it: a host behind a network certifier, which splits its uplink, and
    a memory challenger; beside them a second host behind a second certifier, a link between the two hosts and the
    second host's bare uplink."""
    return {
        "sheets": [],
        "locs": [loc("site0", "site", None), loc("node0", "host"),
                 loc("gpu0", "device", "node0", sheet="s"), loc("hbm0", "memory", "gpu0", tier="hbm", bytes=1 << 30),
                 loc("cert0", "certifier", owner="auditor", key=KEY0), loc("chal0", "challenger", owner="auditor", key="k1"),
                 loc("node1", "host"), loc("cert1", "certifier", owner="auditor", key=KEY1)],
        "links": [link("up0", "node0", "cert0", 100), link("up1", "cert0", "outside", 100), link("mc0", "chal0", "node0", 10),
                  link("up2", "cert1", "node1", 40), link("up3", "cert1", "outside", 40), link("hh0", "node0", "node1", 400),
                  link("up4", "node1", "outside", 10)],
    }


def read(site: dict, name: str, key: str, head: str = "h") -> R.Registered:
    return R.registered(R.site_of(site), head, name, key)


def refusal(site: dict, name: str, key: str) -> str:
    with pytest.raises(R.Refused) as e:
        read(site, name, key)
    assert type(e.value) is R.Refused
    return str(e.value)


def with_key(key: str) -> dict:
    site = view_site()
    next(x for x in site["locs"] if x["id"] == "cert0")["key"] = key
    return site


def test_the_certifiers_link_and_key_as_registered():
    assert read(view_site(), "up0", KEY0, head="head0") == R.Registered("head0", "up0", "node0", "cert0", KEY0, 100)
    assert read(view_site(), "up2", KEY1) == R.Registered("h", "up2", "node1", "cert1", KEY1, 40)
    assert R.certifier_links(R.site_of(view_site())) == ["up0", "up2"]


def test_a_link_that_is_no_link_of_the_site_is_refused():
    assert refusal(view_site(), "up9", KEY0) == f"link up9: no link of the registered hardware; {LINKS}"
    assert refusal(view_site(), "", KEY0) == f"link : no link of the registered hardware; {LINKS}"


@pytest.mark.parametrize("name, joins", [
    ("up1", "certifier cert0 and the outside"),
    ("up3", "certifier cert1 and the outside"),
    ("up4", "host node1 and the outside"),
    ("hh0", "host node0 and host node1"),
    ("mc0", "challenger chal0 and host node0"),
])
def test_a_link_that_joins_no_certifier_and_host_is_refused(name, joins):
    """The certifier's half to the outside, a host's bare uplink, a link between two hosts, the memory challenger's."""
    assert refusal(view_site(), name, KEY0) == (f"link {name}: it joins {joins}, not a network certifier and a host; "
                                                 f"{LINKS}")


def test_a_link_to_an_end_nothing_registers_is_refused():
    site = view_site()
    site["links"] = [link("up0", "cert0", "node9", 1)]
    assert refusal(site, "up0", KEY0) == ("link up0: it joins certifier cert0 and node9, which is not registered, not a "
                                          "network certifier and a host; the network certifiers' links there are []")


def outside_named() -> dict:
    """A site with a host named as the outside is, and the first certifier's link to that name."""
    site = view_site()
    site["locs"].append(loc("outside", "host"))
    site["links"].insert(0, link("up5", "cert0", "outside", 9))
    return site


def test_the_outside_is_the_outside_whatever_is_registered_under_its_name():
    assert refusal(outside_named(), "up5", KEY0) == ("link up5: it joins certifier cert0 and the outside, not a network "
                                                    f"certifier and a host; {LINKS}")


def repeated() -> dict:
    """A site that names a link and a certifier twice: the first of each is read."""
    site = view_site()
    site["locs"].append(loc("cert0", "certifier", owner="auditor", key=KEY1))
    site["links"].append(link("up0", "node1", "cert1", 7))
    return site


def test_the_first_link_and_location_of_a_name_are_read():
    assert read(repeated(), "up0", KEY0) == R.Registered("h", "up0", "node0", "cert0", KEY0, 100)
    assert refusal(repeated(), "up0", KEY1).startswith(f"link up0: the key {KEY1} is not network certifier cert0's")


def test_a_key_that_is_not_the_certifiers_registered_key_is_refused():
    assert refusal(view_site(), "up0", KEY1) == (f"link up0: the key {KEY1} is not network certifier cert0's registered "
                                                 f"key, {KEY0}")
    assert refusal(view_site(), "up2", "00") == f"link up2: the key 00 is not network certifier cert1's registered key, {KEY1}"
    assert refusal(view_site(), "up0", KEY0.upper()).startswith(f"link up0: the key {KEY0.upper()} is not")


@pytest.mark.parametrize("bad", ["k0", "AB", "abc", "", "0g", "0x", "١٢"])
def test_a_registered_key_that_is_not_lower_case_hex_is_refused(bad):
    """Whatever key is asked for, the same string included."""
    assert refusal(with_key(bad), "up0", bad) == (
        f"link up0: network certifier cert0 registers the key {bad}, which is not lower-case hex; a network certifier "
        "registers its LMS public key in lower-case hex")
    assert not R.lower_hex(bad)


def test_lower_case_hex():
    assert R.lower_hex("00ff19ae") and R.lower_hex(KEY0)
    assert not any(R.lower_hex(s) for s in ("00FF", "0", "", "0x", "00 ", "00\n"))


def malformed() -> list[dict]:
    """Sites that are no site of a view, each one change from `view_site()`."""
    out = []

    def change(edit) -> None:
        site = view_site()
        edit(site)
        out.append(site)

    change(lambda s: s.pop("sheets"))
    change(lambda s: s.update(more=[]))
    change(lambda s: s.update(sheets=None))
    change(lambda s: s.update(locs={}))
    change(lambda s: s.update(links="up0"))
    change(lambda s: s["locs"][1].pop("tier"))
    change(lambda s: s["locs"][1].update(more=1))
    change(lambda s: s["locs"][1].update(kind="router"))
    change(lambda s: s["locs"][1].update(owner="developer"))
    change(lambda s: s["locs"][1].update(tier="tape"))
    change(lambda s: s["locs"][1].update(parent=5))
    change(lambda s: s["locs"][1].update(bytes=True))
    change(lambda s: s["locs"][1].update(bytes=-1))
    change(lambda s: s["locs"][1].update(bytes=1.5))
    change(lambda s: s["locs"][1].update(id=7))
    change(lambda s: s["locs"][4].update(key=None))
    change(lambda s: s["locs"][1].update(sheet=None))
    change(lambda s: s["locs"].append("node2"))
    change(lambda s: s["links"][0].pop("latency"))
    change(lambda s: s["links"][0].update(beta="100"))
    change(lambda s: s["links"][0].update(beta=-1))
    change(lambda s: s["links"][0].update(latency=None))
    change(lambda s: s["links"][0].update(ends=["node0"]))
    change(lambda s: s["links"][0].update(ends=["node0", "cert0", "outside"]))
    change(lambda s: s["links"][0].update(ends=["node0", 1]))
    change(lambda s: s["links"][0].update(ends="nc"))
    change(lambda s: s["links"][0].update(id=None))
    change(lambda s: s["links"].append([]))
    return out + [[], None, "site", 1]


@pytest.mark.parametrize("site", malformed())
def test_a_site_that_is_no_site_of_a_view_is_malformed(site):
    with pytest.raises(R.Malformed):
        R.site_of(site)


IDS = ("site0", "node0", "node1", "cert0", "cert1", "chal0", "gpu0", "outside", "node9", "é")
NAMES = ("up0", "up1", "up2", "mc0", "hh0", "up9", "é")
KEYS = (KEY0, KEY1, "00", "k0", "AB", "abc", "", "0g", "١٢")


def seeded(rng: random.Random) -> dict:
    """A vector near the example's: its site with locations and links dropped, repeated, re-kinded, re-keyed and
    re-ended, or a site made whole at random, and a link and a key that are likely ones of it: the key of a certifier at the
    link's end, most often."""
    if rng.random() < 0.6:
        site = view_site()
        for _ in range(rng.randrange(4)):
            edit = rng.randrange(7)
            if edit == 0 and site["locs"]:
                site["locs"].pop(rng.randrange(len(site["locs"])))
            elif edit == 1 and site["links"]:
                site["links"].pop(rng.randrange(len(site["links"])))
            elif edit == 2 and site["locs"]:
                rng.choice(site["locs"])["kind"] = rng.choice(R.KINDS)
            elif edit == 3 and site["locs"]:
                rng.choice(site["locs"])["key"] = rng.choice(KEYS)
            elif edit == 4 and site["links"]:
                rng.choice(site["links"])["ends"] = [rng.choice(IDS), rng.choice(IDS)]
            elif edit == 5 and site["locs"]:
                site["locs"].insert(rng.randrange(len(site["locs"]) + 1), copy.deepcopy(rng.choice(site["locs"])))
            elif edit == 6 and site["links"]:
                again = copy.deepcopy(rng.choice(site["links"]))
                again["id"] = rng.choice(NAMES)
                site["links"].insert(rng.randrange(len(site["links"]) + 1), again)
    else:
        site = {"sheets": [],
                "locs": [loc(rng.choice(IDS), rng.choice(R.KINDS), key=rng.choice(KEYS)) for _ in range(rng.randrange(7))],
                "links": [link(rng.choice(NAMES), rng.choice(IDS), rng.choice(IDS), rng.randrange(1 << 70))
                          for _ in range(rng.randrange(6))]}
    which = rng.random()
    name = ("up0" if which < 0.5 else rng.choice([k["id"] for k in site["links"]] + ["up2"]) if which < 0.85
            else rng.choice(NAMES))
    ends = next((k["ends"] for k in site["links"] if k["id"] == name), [])
    certifiers = [x for x in site["locs"] if x["id"] in ends and x["kind"] == "certifier"]
    if rng.random() < 0.25:
        for x in certifiers:
            x["key"] = rng.choice(KEYS)
    keys = [x["key"] for x in certifiers]
    key = rng.choice(keys) if keys and rng.random() < 0.7 else rng.choice(KEYS)
    return {"head": rng.choice(("h", "", "é")), "key": key, "link": name, "site": site}


def vectors() -> list[dict]:
    """The named cases above, every malformed site, and 600 seeded vectors."""
    named = [{"head": "head0", "key": key, "link": name, "site": view_site()}
             for name in ("up0", "up1", "up2", "up3", "up4", "hh0", "mc0", "up9", "") for key in (KEY0, KEY1, "00")]
    named += [{"head": "h", "key": bad, "link": "up0", "site": with_key(bad)} for bad in ("k0", "AB", "abc", "", "0g", "١٢")]
    named += [{"head": "h", "key": key, "link": name, "site": site}
              for site, name in ((outside_named(), "up5"), (repeated(), "up0")) for key in (KEY0, KEY1)]
    bad = [{"head": "h", "key": KEY0, "link": "up0", "site": site} for site in malformed()]
    rng = random.Random(0)
    return named + bad + [seeded(rng) for _ in range(600)]


def python(vector: dict) -> dict:
    """The Python's answer on a vector, as the Lean script writes its own."""
    try:
        site = R.site_of(vector["site"])
    except R.Malformed as e:
        return {"malformed": str(e)}
    try:
        return {"ok": asdict(R.registered(site, vector["head"], vector["link"], vector["key"]))}
    except R.Refused as e:
        return {"refused": str(e)}


def lean(file: Path) -> list[dict]:
    """The Lean's answers on a file of vectors: `NetworkCheck.Registered` built in its package, one build at a time,
    then the script run against that build."""
    BUILD_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with open(BUILD_LOCK, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        built = subprocess.run(["lake", "build", "NetworkCheck.Registered"], cwd=PACKAGE, capture_output=True, text=True)
    assert built.returncode == 0, f"lake build NetworkCheck.Registered in {PACKAGE} failed:\n{built.stdout}{built.stderr}"
    ran = subprocess.run(["lake", "env", "lean", "--run", str(SCRIPT), str(file)], cwd=PACKAGE, capture_output=True,
                         text=True, encoding="utf-8")
    assert ran.returncode == 0, f"{SCRIPT.name} failed ({ran.returncode}):\n{ran.stdout}{ran.stderr}"
    return json.loads(ran.stdout)


def test_the_vectors_reach_every_answer():
    """So that the comparison below is of what it says: registered links, each refusal and malformed sites, many of
    each."""
    answers = [python(v) for v in vectors()]
    refused = [a["refused"] for a in answers if "refused" in a]
    count = {"ok": sum("ok" in a for a in answers), "malformed": sum("malformed" in a for a in answers),
             "no link": sum("no link of the registered hardware" in r for r in refused),
             "joins": sum("not a network certifier and a host" in r for r in refused),
             "not hex": sum("which is not lower-case hex" in r for r in refused),
             "key": sum("registered key, " in r for r in refused)}
    assert all(n >= 25 for n in count.values()), count
    assert sum(count.values()) == len(answers)
    assert any("which is not registered" in r for r in refused)
    assert len({a["ok"]["host"] for a in answers if "ok" in a}) > 1


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_the_python_is_the_lean_on_the_same_vectors(tmp_path):
    """Vector by vector: the same registered link, the same refusal word for word, and the same sites malformed (the
    two sides word a malformed site on their own)."""
    cases = vectors()
    file = tmp_path / "vectors.json"
    file.write_text(json.dumps(cases), encoding="utf-8")
    theirs = lean(file)
    assert len(theirs) == len(cases)
    for i, (case, lean_answer) in enumerate(zip(cases, theirs)):
        mine = python(case)
        if "malformed" in mine or "malformed" in lean_answer:
            assert mine.keys() == lean_answer.keys() == {"malformed"}, (i, case, mine, lean_answer)
        else:
            assert mine == lean_answer, (i, case, mine, lean_answer)
