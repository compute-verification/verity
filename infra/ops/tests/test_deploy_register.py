"""`research deploy`'s files made from the register of machines: an install fills the entry's template from the register and
writes that, drift compares the node with what the register gives now, nothing is written from a register that can't be read,
no machine leaves a list unasked, the list's key is taken from every machine it names, and a roll puts one file on every node
one at a time. The register is a directory of this test's, and each node is a directory of this machine: the runner turns
`/NODE/` into the node's own directory, so one manifest path is a different file on each."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable

import pytest
from test_deploy import _commit, _git, local

from verity_ops import deploy

PEERS = ("# how each machine reaches the others\n"
         "{{vy-n-[0-9]*}} -i /k research@{{vy-n-[0-9]*:host}}\n"
         "{{vy-cpu-*}} -i /k research@{{vy-cpu-*:host}}\n")
CONF = '{"gate": "research@{{vy-gate:host}}",\n "peers": {"vy-n-2": {"bytes": 1}}}\n'
# documentation addresses and a name: what a register's `host` holds
HOSTS = {"vy-n-2": "192.0.2.2", "vy-cpu-1": "192.0.2.11", "vy-cpu-2": "192.0.2.12", "vy-cpu-10": "192.0.2.20",
         "vy-gate": "gate.example"}
LIST = ("# how each machine reaches the others\n"
        "vy-n-2 -i /k research@192.0.2.2\n"
        "vy-cpu-1 -i /k research@192.0.2.11\n"
        "vy-cpu-2 -i /k research@192.0.2.12\n"
        "vy-cpu-10 -i /k research@192.0.2.20\n")
KEY = "ssh-ed25519 AAAAdummyclusterkey vy-cluster"
OTHER = 'command="/bin/gate a,b" ssh-ed25519 AAAAdummyotherkey someone'
SCRIPTS = {deploy.PROBE: "probe", deploy.INSTALL: "install", deploy.ACCEPT: "accept", deploy.RECORD: "record"}


def machine(reg: Path, name: str, host: str | None, **more: str) -> None:
    """A machine's register file; with no host, a released machine's."""
    table = {"provider": "ssh", "project": "verity", **({"host": host} if host else {"released_host": "192.0.2.1"}), **more}
    (reg / f"{name}.toml").write_text("".join(f"{k} = {json.dumps(v)}\n" for k, v in table.items()))


def node(tmp: Path, name: str, key: bool = True) -> None:
    """A node, as a directory: its /etc, and its login's ssh directory with another key's line and, with `key`, the list's key."""
    (tmp / name / "etc").mkdir(parents=True)
    (tmp / name / "ssh").mkdir(mode=0o700)
    (tmp / name / "ssh" / "authorized_keys").write_text(OTHER + "\n")
    if key:
        (tmp / name / "ssh" / "k.pub").write_text(KEY + "\n")


@pytest.fixture
def fleet(tmp_path, monkeypatch):
    origin, repo, reg = tmp_path / "origin.git", tmp_path / "repo", tmp_path / "machines.d"
    (tmp_path / "certs").mkdir()
    monkeypatch.setattr(deploy, "CERTS", str(tmp_path / "certs"))
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(origin)], check=True)
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    _git(repo, "remote", "add", "origin", str(origin))
    _commit(repo, {"src/peers": PEERS, "src/conf.json": CONF}, "the templates")
    _git(repo, "push", "-q", "origin", "main")
    reg.mkdir()
    for name, host in HOSTS.items():
        machine(reg, name, host)
    machine(reg, "vy-n-1", None)
    monkeypatch.setenv("RESEARCH_MACHINES_D", str(reg))
    for n in "abc":
        node(tmp_path, n)
    manifest = tmp_path / "deploy.toml"
    manifest.write_text("""
[nodes.a]
record = "/NODE/research/deploy/record.jsonl"
managed = ["/NODE/etc/*"]
[nodes.b]
record = "/NODE/research/deploy/record.jsonl"
managed = ["/NODE/etc/*"]
[nodes.c]
record = "/NODE/research/deploy/record.jsonl"
managed = ["/NODE/etc/*"]

[[file]]
path = "/NODE/etc/peers"
template = "src/peers"
register = true
by = "research deploy, from the template and the register of machines"
accept = "/NODE/ssh/k.pub"
nodes = ["a", "b", "c"]

[[file]]
path = "/NODE/etc/conf.json"
template = "src/conf.json"
register = true
by = "research deploy, from the template and the register of machines"
nodes = ["a"]
""")
    calls: list[tuple[str, str]] = []
    faults: dict[tuple[str, str], tuple[int, str]] = {}
    then: dict[tuple[str, str], Callable[[], None]] = {}

    def run(node: str, argv: list[str], stdin: bytes | None) -> subprocess.CompletedProcess:
        root = f"{tmp_path}/{node}/"
        kind = next(SCRIPTS.get(a, "older") for a in argv if "\n" in a) + ("" if argv[0] == "sudo" or deploy.PROBE in argv else "?")
        calls.append((node, kind))
        if (node, kind) in faults:
            rc, err = faults[(node, kind)]
            return subprocess.CompletedProcess(argv, rc, b"", err.encode())
        r = local(node, [a if "\n" in a else a.replace("/NODE/", root) for a in argv],
                  stdin.replace(b"/NODE/", root.encode()) if stdin else stdin)
        then.get((node, kind), lambda: None)()
        return subprocess.CompletedProcess(argv, r.returncode, *(o.replace(root.encode(), b"/NODE/") for o in (r.stdout, r.stderr)))

    return {"repo": repo, "reg": reg, "tmp": tmp_path, "manifest": manifest, "nodes": deploy.load(manifest), "run": run,
            "calls": calls, "faults": faults, "then": then}


def _main(fleet: dict, *argv: str) -> int:
    return deploy.main([*argv, "--no-fetch"], run=fleet["run"], manifest=fleet["manifest"], repo=fleet["repo"])


def _install(fleet: dict, node: str = "a", **how: bool) -> dict:
    return deploy.install(fleet["nodes"][node], "origin/main", "lane-x", ["/NODE/etc/peers"], fleet["run"], repo=fleet["repo"],
                          **{"apply": True, **how})


def _read(fleet: dict, node: str = "a", path: str = "/NODE/etc/peers") -> dict:
    rep = deploy.drift(fleet["nodes"][node], "origin/main", fleet["run"], fleet["repo"])
    return next(e for e in rep["entries"] if e["path"] == path)


def test_an_install_writes_what_the_register_gives_in_one_order_and_drift_then_reads_the_node_clean(fleet, capsys):
    """The path's drift is clean after one install, and a second install changes nothing. A machine's line follows its name with
    numbers by value (cpu-2 before cpu-10), whatever order the directory lists the register's files in."""
    a, run, repo, etc = fleet["nodes"]["a"], fleet["run"], fleet["repo"], fleet["tmp"] / "a" / "etc"
    assert _main(fleet, "install", "a", "--by", "lane-x") == 0
    out = capsys.readouterr().out
    assert "would install /NODE/etc/peers  <- src/peers" in out and "isn't installed, 4 after" in out
    assert not (etc / "peers").exists()

    done = deploy.install(a, "origin/main", "lane-x", [], run, apply=True, repo=repo)
    assert [f["changed"] for f in done["files"]] == [True, True]
    assert (etc / "peers").read_text() == LIST
    assert json.loads((etc / "conf.json").read_text()) == {"gate": "research@gate.example", "peers": {"vy-n-2": {"bytes": 1}}}
    rep = deploy.drift(a, "origin/main", run, repo)
    assert {e["path"]: e["state"] for e in rep["entries"]} == {"/NODE/etc/peers": "ok", "/NODE/etc/conf.json": "ok"}
    assert deploy.clean(rep) and _main(fleet, "drift", "--on", "a") == 0
    record = [json.loads(ln) for ln in (fleet["tmp"] / "a/research/deploy/record.jsonl").read_text().splitlines()]
    blob = _git(repo, "hash-object", str(etc / "peers"))
    assert [r["blob"] for r in record if r["path"] == str(etc / "peers")] == [blob]

    assert [f["changed"] for f in deploy.install(a, "origin/main", "lane-x", [], run, apply=True, repo=repo)["files"]] == [False, False]
    reg = deploy.register()
    assert deploy.fill(PEERS, reg, "src/peers") == (LIST, ["vy-cpu-1", "vy-cpu-2", "vy-cpu-10", "vy-n-2"])
    assert deploy.fill(PEERS, reg, "src/peers") == deploy.fill(PEERS, deploy.register(), "src/peers")


def test_drift_compares_the_node_with_what_the_register_gives_now_never_with_the_tracked_file(fleet):
    path = fleet["tmp"] / "a" / "etc" / "peers"
    path.write_text(PEERS)                                  # the tracked file, byte for byte: not what a node should hold
    e = _read(fleet)
    assert e["state"] == "unknown" and e["detail"].endswith("; it names 0 machine(s) and the register gives 4; it lacks "
                                                            "vy-cpu-1, vy-cpu-2, vy-cpu-10, vy-n-2")
    _install(fleet)
    assert _read(fleet)["state"] == "ok"
    machine(fleet["reg"], "vy-cpu-3", "192.0.2.13")        # a machine joins: every installed list is behind the register
    e = _read(fleet)
    assert e["state"] == "stale" and "by lane-x" in e["detail"] and e["detail"].endswith(
        "the register or its template has changed since; it names 4 machine(s) and the register gives 5; it lacks vy-cpu-3: install it")
    path.write_text(path.read_text() + "vy-cpu-3 research@192.0.2.13\n")
    e = _read(fleet)
    assert e["state"] == "unknown" and e["detail"].endswith(
        "written by hand or edited in place; it names the 5 machine(s) the register gives, written otherwise")
    path.write_text(path.read_text() + "vy-cpu-77 research@192.0.2.9\n")       # a machine the register never held
    assert _read(fleet)["detail"].endswith("the register gives, written otherwise (6 line(s) for the register's 5)")
    path.write_text(path.read_text().replace("vy-cpu-77 research@192.0.2.9\n", "vy-gate research@192.0.2.9\n"))
    assert _read(fleet)["detail"].endswith("; it names 6 machine(s) and the register gives 5; it has vy-gate (live in the register)")
    path.write_text(path.read_text().replace("vy-gate research@192.0.2.9\n", ""))
    path.write_text(path.read_text().replace("vy-cpu-1 ", "vy-n-1 "))
    assert _read(fleet)["detail"].endswith("; it names 5 machine(s) and the register gives 5; it lacks vy-cpu-1; it has vy-n-1 "
                                           "(released in the register)")
    path.write_text(path.read_text().replace("vy-n-1 ", "vy-cpu-1 "))
    _install(fleet)
    assert _read(fleet)["state"] == "ok" and "vy-cpu-3 -i /k research@192.0.2.13\n" in path.read_text()


def test_an_install_never_takes_a_machine_out_of_a_list_unasked_and_says_which_would_go(fleet, capsys):
    """The tree's list was shorter than the machines': an install of it would have cut machines off from each other."""
    path = fleet["tmp"] / "a" / "etc" / "peers"
    hand = LIST + "vy-n-1 research@192.0.2.1\n# vy-gate, in a comment, is no line\n"
    path.write_text(hand)
    for apply in (True, False):
        with pytest.raises(deploy.Refused, match=r"names 5 machines in 5 lines, and the register gives 4 in 4: an install would "
                                                 r"take out vy-n-1 \(released in the register\)"):
            _install(fleet, apply=apply)
    assert _main(fleet, "install", "a", "--by", "lane-x", "--apply", "/NODE/etc/peers") == 3
    assert "`--allow-shrink` installs it all the same" in capsys.readouterr().err and path.read_text() == hand

    path.write_text(LIST + "vy-cpu-77 research@192.0.2.77\n")       # a line of a machine the register never held
    with pytest.raises(deploy.Refused, match=r"would take out 1 line\(s\) that name no machine of the register"):
        _install(fleet)
    (fleet["reg"] / "vy-cpu-2.toml").unlink()                       # and one whose file is gone: its line would go with it
    with pytest.raises(deploy.Refused, match="would take out 2 line"):
        _install(fleet)
    machine(fleet["reg"], "vy-cpu-2", HOSTS["vy-cpu-2"])

    path.write_text(hand)
    assert _main(fleet, "install", "a", "--by", "lane-x", "--allow-shrink", "/NODE/etc/peers") == 0
    assert ("names 5 machines now, 4 after; drops vy-n-1 (released in the register); the lines that go: line 6 (vy-n-1)"
            in capsys.readouterr().out)
    assert path.read_text() == hand
    path.write_text(LIST.replace("-i /k ", "", 1) + "vy-cpu-77 research@192.0.2.77\n192.0.2.78 research\n")    # each line that
    assert _main(fleet, "install", "a", "--by", "lane-x", "--allow-shrink", "/NODE/etc/peers") == 0    # goes, by its number and
    out = capsys.readouterr().out                                                                      # name, never its address
    assert "the lines that go: line 2 (vy-n-2), line 6 (vy-cpu-77), line 7\n" in out and "192.0.2.7" not in out
    path.write_text(hand)
    assert _install(fleet, allow_shrink=True)["files"][0]["machines"]["dropped"] == ["vy-n-1"] and path.read_text() == LIST

    path.write_text(LIST.replace("vy-cpu-10 -i /k research@192.0.2.20\n", ""))       # a list that grows needs no leave
    assert _install(fleet)["files"][0]["machines"] == {"now": 3, "after": 4, "added": ["vy-cpu-10"], "dropped": [], "says": {},
                                                       "gone": []}
    assert path.read_text() == LIST


def test_an_install_of_the_whole_node_leaves_a_list_written_by_hand_or_one_it_would_shorten_and_installs_the_rest(fleet, capsys):
    """Landing changes no list: until one is named or rolled, a node's other files install around it, and an install of them
    is never refused for a list."""
    a, run, repo, etc = fleet["nodes"]["a"], fleet["run"], fleet["repo"], fleet["tmp"] / "a" / "etc"
    hand = LIST.replace("-i /k ", "")                       # the register's machines, written another way
    (etc / "peers").write_text(hand)
    for apply in ((), ("--apply",)):
        assert _main(fleet, "install", "a", "--by", "lane-x", *apply) == 0
        assert ("left as it is /NODE/etc/peers  <- src/peers  (written by hand, which an install of the whole node leaves: name it, "
                "or `research deploy roll /NODE/etc/peers`)") in capsys.readouterr().out
    assert (etc / "peers").read_text() == hand and (etc / "conf.json").exists() and _keys(fleet) == [OTHER]
    assert _install(fleet)["files"][0]["changed"] is True and (etc / "peers").read_text() == LIST       # named: the register's now

    machine(fleet["reg"], "vy-cpu-10", None)                # a machine is released: the list deploy wrote would lose its line
    left = [f for f in deploy.install(a, "origin/main", "lane-x", [], run, apply=True, repo=repo)["files"] if f["changed"] == "left"]
    assert [f["path"] for f in left] == ["/NODE/etc/peers"] and (etc / "peers").read_text() == LIST
    assert "would take out vy-cpu-10 (released in the register)" in left[0]["why"]
    assert "`research deploy roll /NODE/etc/peers --allow-shrink`, installs it all the same" in left[0]["why"]
    with pytest.raises(deploy.Refused, match="would take out vy-cpu-10"):
        _install(fleet)
    done = deploy.install(a, "origin/main", "lane-x", [], run, apply=True, repo=repo, allow_shrink=True)
    assert "left" not in [f["changed"] for f in done["files"]] and "vy-cpu-10" not in (etc / "peers").read_text()

    machine(fleet["reg"], "vy-cpu-11", "192.0.2.21")        # a machine joins: the whole node's install keeps deploy's list current
    done = deploy.install(a, "origin/main", "lane-x", [], run, apply=True, repo=repo)
    assert "left" not in [f["changed"] for f in done["files"]] and "vy-cpu-11 " in (etc / "peers").read_text()


def test_the_machines_a_file_names_are_whole_words_of_its_lines_comments_apart():
    """A line ends at a newline only: a carriage return or a form feed inside one is no second line."""
    text = "# vy-cpu-1 in a comment\n\nvy-cpu-10 -i /k research@192.0.2.20\r\n  Host n2 vy-n-2 x\x0cvy-x\n{\"vy-gate\": 1}\n"
    digest = [hashlib.sha256(ln.encode()).hexdigest() for ln in text.split("\n")]
    assert deploy.named(text, ["vy-cpu-1", "vy-cpu-10", "vy-n-2", "vy-gate", "research", "absent"]) == \
        (3, ["research", "vy-cpu-10", "vy-gate", "vy-n-2"],
         [[3, digest[2], ["research", "vy-cpu-10"]], [4, digest[3], ["vy-n-2"]], [5, digest[4], ["vy-gate"]]])
    assert deploy.named("192.0.2.9 x\nvy-old y\n", [])[2] == [[1, hashlib.sha256(b"192.0.2.9 x").hexdigest(), []],
                                                             [2, hashlib.sha256(b"vy-old y").hexdigest(), ["vy-old"]]]


def test_a_released_machine_leaves_a_pattern_and_is_refused_by_name_and_a_stopped_one_keeps_its_line(fleet):
    reg = fleet["reg"]
    machine(reg, "vy-cpu-2", HOSTS["vy-cpu-2"], unreachable_since="2026-01-01T00:00:00Z")      # stopped: still its address
    got = deploy.register()
    assert got.gone == {"vy-n-1"} and "vy-cpu-2" in got.live
    assert deploy.fill(PEERS, got, "src/peers")[0] == LIST
    with pytest.raises(deploy.NoRegister, match=r"src/x:1 names vy-n-1, which the register marks released \(.*/vy-n-1.toml\)"):
        deploy.fill("{{vy-n-1}} research@{{vy-n-1:host}}\n", got, "src/x")
    machine(reg, "vy-cpu-2", None)                          # released: the pattern passes it by, and its line goes
    assert "vy-cpu-2" not in deploy.fill(PEERS, deploy.register(), "src/peers")[0]


UNREADABLE = {
    "off": (lambda f, m: m.setenv("RESEARCH_MACHINES_D", "none"), r"the register of machines is off \(\$RESEARCH_MACHINES_D='none'\)"),
    "no directory": (lambda f, m: m.setenv("RESEARCH_MACHINES_D", str(f["tmp"] / "nowhere")), "{tmp}/nowhere is no directory"),
    "empty": (lambda f, m: [p.unlink() for p in f["reg"].glob("*.toml")], "holds no machine: {tmp}/machines.d has no"),
    "a broken file a pattern matches": (lambda f, m: (f["reg"] / "vy-cpu-5.toml").write_text("provider = "),
                                        "{tmp}/machines.d/vy-cpu-5.toml is broken and matches"),
    "a file that is no machine's": (lambda f, m: (f["reg"] / "vy-cpu-5.toml").write_text('provider = "ssh"\nhost = "a b"\n'),
                                    "{tmp}/machines.d/vy-cpu-5.toml is broken and matches"),
    "no machine of a pattern": (lambda f, m: [p.unlink() for p in f["reg"].glob("vy-cpu-*.toml")],
                                r"{tmp}/machines.d holds no live machine matching vy-cpu-\*"),
    "a machine the template names": (lambda f, m: (f["reg"] / "vy-gate.toml").unlink(),
                                     r"src/conf.json:1 names vy-gate, which the register doesn't hold \(no {tmp}/machines.d/vy-gate.toml\)"),
}


@pytest.mark.parametrize("case", UNREADABLE)
def test_nothing_is_written_from_a_register_that_cannot_be_read_and_the_refusal_names_the_path(fleet, monkeypatch, capsys, case):
    """Never an empty or a shorter list: the install refuses before it reaches the node, and drift says why it can't tell."""
    a, run, repo, etc = fleet["nodes"]["a"], fleet["run"], fleet["repo"], fleet["tmp"] / "a" / "etc"
    deploy.install(a, "origin/main", "lane-x", [], run, apply=True, repo=repo)
    before = {p.name: p.read_bytes() for p in etc.iterdir() if p.is_file()}
    fleet["calls"].clear()
    spoil, why = UNREADABLE[case]
    spoil(fleet, monkeypatch)
    why = why.replace("{tmp}", str(fleet["tmp"]))
    for apply in (True, False):
        with pytest.raises(deploy.NoRegister, match=why):
            deploy.install(a, "origin/main", "lane-x", [], run, apply=apply, repo=repo)
    assert _main(fleet, "install", "a", "--by", "lane-x", "--apply", "--allow-shrink") == 3
    assert fleet["calls"] == [] and {p.name: p.read_bytes() for p in etc.iterdir() if p.is_file()} == before
    with pytest.raises(deploy.NoRegister, match=why):
        deploy.roll(fleet["nodes"], "/NODE/etc/peers" if case != "a machine the template names" else "/NODE/etc/conf.json",
                    "origin/main", "lane-x", run, apply=True, repo=repo, allow_shrink=True)
    capsys.readouterr()
    rep = deploy.drift(a, "origin/main", run, repo)
    told = [e for e in rep["entries"] if e["state"] == "unknown" and "the register can't give it" in e["detail"]]
    assert told and not deploy.clean(rep)


def test_a_register_value_is_one_plain_word_or_the_file_is_not_made_and_the_refusal_never_prints_it(fleet):
    """A list's line is ssh's arguments on every machine: a value with a space would be an option there."""
    machine(fleet["reg"], "vy-gate", "gate.example", root="/w -oProxyCommand=sh", lane="infra")
    reg = deploy.register()
    assert deploy.fill("{{vy-gate}} {{vy-gate:lane}}@{{vy-gate:host}}\n", reg, "src/x") == ("vy-gate infra@gate.example\n", ["vy-gate"])
    with pytest.raises(deploy.NoRegister, match="vy-gate.toml: its `root` isn't one plain word") as e:
        deploy.fill("{{vy-gate:root}}\n", reg, "src/x")
    assert "ProxyCommand" not in str(e.value)
    with pytest.raises(deploy.NoRegister, match=r"machines.d/vy-gate.toml has no `private_host`, which src/x:2 asks of it"):
        deploy.fill("# its own line\nvy-gate research@{{vy-gate:private_host}}\n", reg, "src/x")
    with pytest.raises(deploy.NoRegister, match="src/x:1 names more than one machine or pattern"):
        deploy.fill("{{vy-gate}} {{vy-n-2:host}}\n", reg, "src/x")


def _keys(fleet: dict, node: str = "a") -> list[str]:
    return (fleet["tmp"] / node / "ssh" / "authorized_keys").read_text().splitlines()


def test_the_lists_key_is_taken_from_every_machine_it_names_and_no_other_line_changes(fleet, capsys):
    ak = fleet["tmp"] / "a" / "ssh" / "authorized_keys"
    every = "192.0.2.11,192.0.2.12,192.0.2.20,192.0.2.2"            # in the order of the machines' names
    assert _main(fleet, "install", "a", "--by", "lane-x", "/NODE/etc/peers") == 0       # a dry run reads and says
    assert "its key is taken from 4 more address(es), 4 in all" in capsys.readouterr().out and _keys(fleet) == [OTHER]

    assert _install(fleet)["files"][0]["accept"] == {"line": "new", "added": 4, "sources": 4, "anywhere": False}
    assert _keys(fleet) == [OTHER, f'from="{every}" {KEY}']
    assert _install(fleet)["files"][0]["accept"] == {"line": "kept", "added": 0, "sources": 4, "anywhere": False}

    # the key's line as a machine has it: sources no register holds, an option, another key before and after it
    ak.write_text(f'{OTHER}\nrestrict,from="198.51.100.5,192.0.2.11",pty {KEY}\n# a comment\n{OTHER} again\n')
    assert _install(fleet)["files"][0]["accept"] == {"line": "extended", "added": 3, "sources": 5, "anywhere": False}
    assert _keys(fleet) == [OTHER, f'restrict,from="198.51.100.5,192.0.2.11,192.0.2.12,192.0.2.20,192.0.2.2",pty {KEY}',
                            "# a comment", f"{OTHER} again"]
    assert {p.name.split(".")[0] for p in (ak.parent / ".deploy-prev").iterdir()} == {"authorized_keys"}
    record = [json.loads(ln) for ln in (fleet["tmp"] / "a/research/deploy/record.jsonl").read_text().splitlines()]
    assert [(r["path"], r["added"], r["by"]) for r in record if "accept" in r] == [(str(ak), 4, "lane-x"), (str(ak), 3, "lane-x")]

    ak.write_text(f"{KEY}\n")                               # no from=: taken from anywhere already, and never narrowed
    assert _install(fleet)["files"][0]["accept"] == {"line": "kept", "added": 0, "sources": 0, "anywhere": True} and _keys(fleet) == [KEY]
    assert _main(fleet, "install", "a", "--by", "lane-x", "/NODE/etc/peers") == 0
    assert "its key is taken from anywhere already (its line has no from=), and stays so" in capsys.readouterr().out

    if os.geteuid():                                        # a login that can't read the key's directory (vy-admin on a
        ak.parent.chmod(0)                                  # locked-out node): the dry run says so, and --apply reads as root
        try:
            assert _main(fleet, "install", "a", "--by", "lane-x", "/NODE/etc/peers") == 0
        finally:
            ak.parent.chmod(0o700)
        assert "its key can't be read here (PermissionError: --apply reads it as its owner)" in capsys.readouterr().out

    (ak.parent / "k.pub").unlink()                          # no key here: refused before the list is touched
    (fleet["tmp"] / "a" / "etc" / "peers").write_text("by hand\n")
    for apply in (True, False):
        with pytest.raises(deploy.Refused, match="a: no public key at /NODE/ssh/k.pub"):
            _install(fleet, apply=apply, allow_shrink=True)
    assert (fleet["tmp"] / "a" / "etc" / "peers").read_text() == "by hand\n"


# Put before a script: records each open and each other call that names a path (with whether it is relative to a directory's
# descriptor), and each id call a root process makes, stood in for so the script runs as any user.
SPY = r"""
import builtins, json, os, sys
_log = os.open(os.environ["SPY_LOG"], os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
def _say(*what):
    f = sys._getframe(1)
    while f is not None:
        if f.f_code.co_filename.endswith("verity_suite_reads.py"):
            return                                          # the suite's read tracer resolving a name, not the script
        f = f.f_back
    os.write(_log, (json.dumps([os.getpid(), *what]) + "\n").encode())
_os_open, _open = os.open, builtins.open
def _spy_os_open(path, flags, mode=0o777, *, dir_fd=None):
    _say("open", os.fspath(path), flags, mode, dir_fd is not None)
    return _os_open(path, flags, mode, dir_fd=dir_fd)
def _spy_open(file, *args, **kwargs):
    if not isinstance(file, int):
        _say("open", os.fspath(file), None, None, False)
    return _open(file, *args, **kwargs)
def _spy_path(name):
    real = getattr(os, name)
    def spy(*args, **kwargs):
        _say(name, *(os.fspath(a) for a in args if isinstance(a, (str, os.PathLike))),
             any(kwargs.get(k) is not None for k in ("dir_fd", "src_dir_fd", "dst_dir_fd")))
        return real(*args, **kwargs)
    setattr(os, name, spy)
for _name in ("stat", "lstat", "chown", "chmod", "mkdir", "makedirs", "unlink", "rmdir", "link", "symlink", "rename", "replace",
              "fchown"):
    _spy_path(_name)
os.open, builtins.open = _spy_os_open, _spy_open
os.geteuid = os.getuid = os.getegid = os.getgid = lambda: 0
os.setgroups = lambda groups: _say("setgroups", list(groups))
os.setgid = lambda gid: _say("setgid", gid)
os.setuid = lambda uid: _say("setuid", uid)
"""
NOFOLLOW_DIR = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def _accept(fleet: dict, tmp_path: Path, *before: str, **more) -> tuple[subprocess.CompletedProcess, list[list]]:
    """ACCEPT under --apply on node a, run as root runs it (SPY), with `before` put ahead of it: its result and its calls."""
    ssh, log = fleet["tmp"] / "a" / "ssh", tmp_path / "spy.jsonl"
    log.unlink(missing_ok=True)
    req = {"key": str(ssh / "k.pub"), "authorized": str(ssh / "authorized_keys"), "sources": ["192.0.2.11"], "apply": True,
           "record": str(fleet["tmp"] / "a" / "research" / "deploy" / "record.jsonl"), "commit": "c" * 40, "ref": "origin/main",
           "by": "lane-x", **more}
    r = subprocess.run([sys.executable, "-c", "".join(before) + SPY + deploy.ACCEPT], input=json.dumps(req).encode(),
                       capture_output=True, env={**os.environ, "SPY_LOG": str(log)})
    return r, [json.loads(ln) for ln in log.read_text().splitlines()] if log.exists() else []


def test_the_keys_line_is_written_by_the_directorys_owner_and_root_appends_only_the_record(fleet, tmp_path):
    """ACCEPT as root runs it under --apply: the child takes the directory's groups, gid and uid before it opens anything and
    then names nothing but by the directory's descriptor; root opens the home, the directory in it and the record's directory
    above, and names nothing else by a path (the directory's owner is read from its descriptor, never by its name again); the
    temporary file is made new with mode 0600 and never through a link, and the record's line is root's alone, through the
    descriptors it opened, never a link."""
    home = fleet["tmp"] / "a"
    ssh, up = home / "ssh", home / "research"
    up.mkdir()                                              # as /workspace/research is on a node
    r, calls = _accept(fleet, tmp_path)
    assert (r.returncode, r.stderr) == (0, b"")
    assert json.loads(r.stdout) == {"line": "new", "added": 1, "sources": 1, "anywhere": False}
    assert _keys(fleet) == [OTHER, f'from="192.0.2.11" {KEY}']
    root = calls[0][0]
    child = [c[1:] for c in calls if c[0] != root]
    own = ssh.stat()
    tmp = ".authorized_keys.deploy-tmp"
    assert child[:3] == [["setgroups", []], ["setgid", own.st_gid], ["setuid", own.st_uid]]
    read = os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW
    assert child[3:] == [["open", "k.pub", read, 0o600, True], ["open", "authorized_keys", read, 0o600, True],
                         ["open", ".deploy-prev", NOFOLLOW_DIR, 0o600, True], ["stat", tmp, True],
                         ["open", tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, True],
                         ["mkdir", ".deploy-prev", True], ["open", ".deploy-prev", NOFOLLOW_DIR, 0o600, True],
                         ["link", "authorized_keys", next(p.name for p in (ssh / ".deploy-prev").iterdir()), True],
                         ["replace", tmp, "authorized_keys", True]]
    assert [c[1:] for c in calls if c[0] == root] == [
        ["open", str(home), os.O_RDONLY | os.O_DIRECTORY, 0o777, False], ["open", "ssh", NOFOLLOW_DIR, 0o777, True],
        ["open", str(up), NOFOLLOW_DIR, 0o777, False], ["mkdir", "deploy", True], ["open", "deploy", NOFOLLOW_DIR, 0o777, True],
        ["open", "record.jsonl", os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o644, True],
        ["fchown", False], ["fchown", False]]
    assert [json.loads(ln)["added"] for ln in (up / "deploy" / "record.jsonl").read_text().splitlines()] == [1]


def _entries(*tops: Path) -> dict[str, tuple]:
    """Every entry under `tops`, links never followed: a link as its target, a directory as its mode, a fifo as its mode (never
    read), a file as its bytes too."""
    out = {}
    for top in tops:
        for d, dirs, files in os.walk(top):
            for p in (Path(d) / n for n in dirs + files):
                out[str(p)] = (("link", os.readlink(p)) if p.is_symlink() else ("dir", p.stat().st_mode) if p.is_dir()
                               else ("fifo", p.stat().st_mode) if p.is_fifo() else ("file", p.read_bytes(), p.stat().st_mode))
    return out


def _move_away(ssh: Path, away: Path) -> None:
    ssh.rename(away / "ssh")
    ssh.symlink_to(away / "ssh")


LINKED = {   # what the directory's owner puts there, and the refusal
    "authorized_keys is a link": (lambda ssh, away: ((ssh / "authorized_keys").unlink(), (ssh / "authorized_keys").symlink_to(away / "f")),
                                  "/NODE/ssh/authorized_keys is a link"),
    "a link at the temporary name": (lambda ssh, away: (ssh / ".authorized_keys.deploy-tmp").symlink_to(away / "f"),
                                     "/NODE/ssh/.authorized_keys.deploy-tmp is a link or not a regular file"),
    "the key is a fifo": (lambda ssh, away: ((ssh / "k.pub").unlink(), os.mkfifo(ssh / "k.pub")), "/NODE/ssh/k.pub is not a regular file"),
    "authorized_keys is a fifo": (lambda ssh, away: ((ssh / "authorized_keys").unlink(), os.mkfifo(ssh / "authorized_keys")),
                                  "/NODE/ssh/authorized_keys is not a regular file"),
    "authorized_keys is a directory": (lambda ssh, away: ((ssh / "authorized_keys").unlink(), (ssh / "authorized_keys").mkdir()),
                                       "/NODE/ssh/authorized_keys is not a regular file"),
    ".deploy-prev is a link": (lambda ssh, away: ((away / "prev").mkdir(), (ssh / ".deploy-prev").symlink_to(away / "prev")),
                               "/NODE/ssh/.deploy-prev is a link or no directory"),
    "the directory is a link": (_move_away, "/NODE/ssh is a link or no directory: nothing in it is written"),
    "the group can write the directory": (lambda ssh, away: ssh.chmod(0o770), "/NODE/ssh is writable by group or others (mode 770)"),
    "others can write the directory": (lambda ssh, away: ssh.chmod(0o702), "/NODE/ssh is writable by group or others (mode 702)"),
}


@pytest.mark.parametrize("case", LINKED)
def test_a_link_in_the_keys_directory_or_a_directory_others_can_write_is_refused_and_nothing_is_written(fleet, capsys, case):
    """A link the directory's user puts there would have root's write land on a file of its choosing; a directory others can
    write is one ssh refuses and one whose files another user can swap; a fifo would hold the step on its read forever, so
    anything but a regular file is refused, unread. The record may be opened (and made, empty) before the child refuses; it
    gets no line."""
    ssh, away = fleet["tmp"] / "a" / "ssh", fleet["tmp"] / "away"
    away.mkdir()
    (away / "f").write_text("another file\n")
    spoil, why = LINKED[case]
    spoil(ssh, away)
    record = fleet["tmp"] / "a" / "research"
    before = _entries(fleet["tmp"] / "a", away)
    assert _main(fleet, "install", "a", "--by", "lane-x", "--apply", "/NODE/etc/peers") == 3
    assert f"research deploy: a: {why}" in capsys.readouterr().err
    assert {p: v for p, v in _entries(fleet["tmp"] / "a", away).items() if not p.startswith(str(record))} == before
    assert not (record / "deploy" / "record.jsonl").exists() or (record / "deploy" / "record.jsonl").read_text() == ""


RECORD_SPOILED = {   # what the record directory's user puts at the record, and the refusal
    "a link at the record": (lambda deploy_d, away: (deploy_d / "record.jsonl").symlink_to(away / "f"),
                             "/NODE/research/deploy/record.jsonl is a link or not a regular file"),
    "a link at its directory": (lambda deploy_d, away: ((away / "d").mkdir(), deploy_d.rmdir(), deploy_d.symlink_to(away / "d")),
                                "/NODE/research/deploy is a link or no directory"),
    "a fifo at the record": (lambda deploy_d, away: os.mkfifo(deploy_d / "record.jsonl"),
                             "/NODE/research/deploy/record.jsonl is a link or not a regular file"),
    "a second name of the record": (lambda deploy_d, away: ((deploy_d / "record.jsonl").write_text(""),
                                                            os.link(deploy_d / "record.jsonl", away / "g")),
                                    "/NODE/research/deploy/record.jsonl is not a regular file of one name"),
}


@pytest.mark.parametrize("route", ["the key's step", "an install", "a restart's record", "the attic"])
@pytest.mark.parametrize("case", RECORD_SPOILED)
def test_the_record_is_appended_by_root_never_through_a_link_and_only_to_a_regular_file_of_its_own(fleet, capsys, case, route):
    """The record's directory and the record are the research user's, and root appends to it: a link there would have root
    write, and hand the user, a file or directory of its choosing (a link at the record to /etc/shadow, at its directory to
    /etc/sudoers.d), a second name the same, and a fifo would hold the step forever. Each script that appends (the key's
    step, an install, a restart's record, the attic) opens the record first and refuses, exit 3, before it changes anything."""
    tmp, away = fleet["tmp"], fleet["tmp"] / "away"
    deploy_d = tmp / "a" / "research" / "deploy"
    deploy_d.mkdir(parents=True)
    away.mkdir()
    (away / "f").write_text("another file\n")
    (tmp / "a" / "etc" / "stray").write_text("a stray\n")
    spoil, why = RECORD_SPOILED[case]
    spoil(deploy_d, away)
    before = _entries(tmp / "a", away)
    if route == "a restart's record":
        req = {"record": str(deploy_d / "record.jsonl"), "lines": [{"unit": "u", "restart": "restarted"}]}
        r = subprocess.run([sys.executable, "-c", deploy.RECORD], input=json.dumps(req).encode(), capture_output=True)
        assert (r.returncode, why.replace("/NODE/", f"{tmp}/a/") in r.stderr.decode()) == (3, True)
    elif route == "the attic":
        assert _main(fleet, "attic", "a", "--by", "lane-x", "--apply", "/NODE/etc/stray") == 3
        assert f"research deploy: a: {why}" in capsys.readouterr().err
    else:
        path = "/NODE/etc/peers" if route == "the key's step" else "/NODE/etc/conf.json"
        assert _main(fleet, "install", "a", "--by", "lane-x", "--apply", path) == 3
        assert f"research deploy: a: {why}" in capsys.readouterr().err
    assert _entries(tmp / "a", away) == before


def test_the_probe_reads_the_record_only_when_it_is_a_regular_file(fleet):
    """The probe reads the record as the login, which may be vy-admin, from a directory the research user owns: a link there to
    a device that never ends (here /dev/urandom) would hold drift forever or break it. It is passed by, unread."""
    deploy_d = fleet["tmp"] / "a" / "research" / "deploy"
    deploy_d.mkdir(parents=True)
    (deploy_d / "record.jsonl").symlink_to("/dev/urandom")
    assert _read(fleet)["state"] == "missing"


def test_the_keys_line_is_written_only_as_the_user_whose_home_holds_the_directory_and_the_key_is_read_beside_it(fleet, tmp_path):
    """A directory root owns (one renamed onto the name, say, or root's own home and directory) would leave the child root, and
    one another user owns would have the line written as that user: each is refused before the child starts. The key is read
    in the directory, by its descriptor, never by a path of its own."""
    home = fleet["tmp"] / "a"
    ssh = home / "ssh"
    keys = (ssh / "authorized_keys").read_bytes()
    for uid, also_home in ((0, False), (ssh.stat().st_uid + 1, False), (0, True)):
        inos = {ssh.stat().st_ino, *([home.stat().st_ino] if also_home else [])}
        owner = ("import os as _o\n_fstat = _o.fstat\n"
                 f"def _f(fd, _inos={inos}):\n    st = _fstat(fd)\n"
                 f"    return _o.stat_result((st.st_mode, st.st_ino, st.st_dev, st.st_nlink, {uid}, *st[5:10])) "
                 "if st.st_ino in _inos else st\n_o.fstat = _f\n")
        r, calls = _accept(fleet, tmp_path, owner)
        assert r.returncode == 3 and f"{ssh} is owned by uid {uid}, not by the user whose home" in r.stderr.decode()
        assert not [c for c in calls if c[1] in ("setgroups", "setgid", "setuid")] and (ssh / "authorized_keys").read_bytes() == keys
    r, _ = _accept(fleet, tmp_path, key=str(fleet["tmp"] / "a" / "k.pub"))
    assert r.returncode == 3 and "is not in" in r.stderr.decode() and (ssh / "authorized_keys").read_bytes() == keys


def test_a_run_that_fails_once_its_temporary_file_is_made_removes_it_and_the_next_run_writes(fleet, tmp_path):
    """A failure after the temporary file is made (a full disk, a link the kernel refuses) takes the file away, and one a run
    killed mid-way left is removed by the next run as its own, so no later run is refused for it."""
    ssh = fleet["tmp"] / "a" / "ssh"
    tmp = ssh / ".authorized_keys.deploy-tmp"
    keys = (ssh / "authorized_keys").read_bytes()
    full = "import errno as _e, os as _o\ndef _full(*a, **k):\n    raise OSError(_e.ENOSPC, 'No space left on device')\n_o.replace = _full\n"
    r, _ = _accept(fleet, tmp_path, full)
    assert r.returncode == 1 and b"No space left on device" in r.stderr
    assert not tmp.exists() and (ssh / "authorized_keys").read_bytes() == keys
    assert not (fleet["tmp"] / "a" / "research" / "deploy" / "record.jsonl").read_text()
    tmp.write_text("half a file a killed run left\n")
    r, _ = _accept(fleet, tmp_path)
    assert r.returncode == 0 and not tmp.exists() and _keys(fleet) == [OTHER, f'from="192.0.2.11" {KEY}']


def test_every_other_line_of_authorized_keys_is_written_back_as_it_was_whatever_it_holds(fleet, tmp_path):
    """Lines end at a newline only: a carriage return, a form feed or a line separator inside a line is part of it."""
    ak = fleet["tmp"] / "a" / "ssh" / "authorized_keys"
    odd = [f"{OTHER}\r", "# a comment\x0cwith a form feed\u2028and a line separator", "", f"{OTHER} \x1c again"]
    ak.write_bytes(("\n".join(odd) + "\n").encode())
    r, _ = _accept(fleet, tmp_path)
    assert r.returncode == 0 and ak.read_bytes().decode().split("\n") == [*odd, f'from="192.0.2.11" {KEY}', ""]


def test_a_roll_brings_a_new_machine_into_every_nodes_list_one_node_at_a_time_keys_first(fleet, capsys):
    """Bringing up a machine: once the register holds it, one roll has every node take the key from it and then name it."""
    tmp, calls = fleet["tmp"], fleet["calls"]
    assert _main(fleet, "roll", "/NODE/etc/peers", "--by", "lane-x", "--apply") == 0
    assert all((tmp / n / "etc" / "peers").read_text() == LIST for n in "abc")
    capsys.readouterr()

    machine(fleet["reg"], "vy-cpu-11", "192.0.2.21")
    calls.clear()
    assert _main(fleet, "roll", "/NODE/etc/peers", "--by", "lane-x") == 0
    out = capsys.readouterr().out
    assert out.count("names 4 machines now, 5 after; adds vy-cpu-11; its key is taken from 1 more address(es), 5 in all") == 3
    assert "nothing changed" in out and all((tmp / n / "etc" / "peers").read_text() == LIST for n in "abc")
    assert {k for _, k in calls} == {"probe", "accept?"}

    calls.clear()
    assert _main(fleet, "roll", "/NODE/etc/peers", "--by", "lane-x", "--apply") == 0
    wrote = [c for c in calls if c[1] in ("accept", "install")]
    assert wrote == [("a", "accept"), ("b", "accept"), ("c", "accept"), ("a", "install"), ("b", "install"), ("c", "install")]
    at = {n: calls.index((n, "install")) for n in "abc"}
    assert all((x, "probe") in calls[at[x] + 1:at[y]] for x, y in ("ab", "bc"))      # each is read back before the next is written
    for n in "abc":
        assert (tmp / n / "etc" / "peers").read_text() == LIST + "vy-cpu-11 -i /k research@192.0.2.21\n"
        assert "192.0.2.21" in _keys(fleet, n)[1] and _read(fleet, n)["state"] == "ok"


def test_bringing_up_a_machine_gives_it_the_list_and_the_key_and_puts_it_in_every_other_nodes_list(fleet, capsys):
    """A machine from its launch to a node every other reaches. The nodes here are the machines the list names: two that serve
    and one the manifest lists ahead of its launch. Registering it is the launch's; the key pair is copied to it; one roll then
    has every node take the key from it and it from every node, and writes each node's list, one node at a time."""
    tmp, calls, reg, new = fleet["tmp"], fleet["calls"], fleet["reg"], "vy-cpu-3"
    nodes = ["vy-cpu-1", "vy-cpu-2", new]
    for n in nodes:
        node(tmp, n, key=n != new)
    text = fleet["manifest"].read_text().replace('nodes = ["a", "b", "c"]', f"nodes = {json.dumps(nodes)}")
    for old, name in zip("abc", nodes):
        text = text.replace(f"[nodes.{old}]\n", f"[nodes.{name}]\n" + ('staged = "its quota is asked for"\n' if name == new else ""))
    fleet["manifest"].write_text(text.replace('nodes = ["a"]', 'nodes = ["vy-cpu-1"]'))
    fleet["nodes"] = deploy.load(fleet["manifest"])
    roll = ("roll", "/NODE/etc/peers", "--by", "lane-x")

    def lists() -> list[str | None]:
        return [p.read_text() if (p := tmp / n / "etc" / "peers").exists() else None for n in nodes]

    assert _main(fleet, *roll, "--apply") == 0              # before its launch it is passed by, and the others hold the list
    assert f"{new}  passed by: staged (its quota is asked for) and not in the register" in capsys.readouterr().out
    assert lists() == [LIST, LIST, None]

    machine(reg, new, "192.0.2.13")                         # the launch registered it: its key pair isn't on it yet
    calls.clear()
    assert _main(fleet, *roll, "--apply") == 3
    err = capsys.readouterr().err
    assert f"{new}: no public key at /NODE/ssh/k.pub" in err and "Nothing changed on any node (1 of 3 refuse)" in err
    assert lists() == [LIST, LIST, None] and not [c for c in calls if c[1] in ("accept", "install")]

    (tmp / new / "ssh" / "k.pub").write_text(KEY + "\n")    # the pair is copied to it: the dry run says what each node gains
    assert _main(fleet, *roll) == 0
    out = capsys.readouterr().out
    assert out.count(f"names 4 machines now, 5 after; adds {new}; its key is taken from 1 more address(es), 5 in all") == 2
    assert f"{new}  would install  isn't installed, 5 after; its key is taken from 5 more address(es), 5 in all" in out
    assert lists() == [LIST, LIST, None]

    calls.clear()
    assert _main(fleet, *roll, "--apply") == 0
    assert [c for c in calls if c[1] in ("accept", "install")] == [*((n, "accept") for n in nodes), *((n, "install") for n in nodes)]
    after = LIST.replace("vy-cpu-10 ", f"{new} -i /k research@192.0.2.13\nvy-cpu-10 ")
    assert lists() == [after] * 3
    every = {*HOSTS.values()} - {"gate.example"} | {"192.0.2.13"}
    for n in nodes:
        line = _keys(fleet, n)[1]
        assert line.endswith(KEY) and set(line.split('"')[1].split(",")) == every and _read(fleet, n)["state"] == "ok"
    assert _main(fleet, *roll, "--apply") == 0 and "unchanged, reads ok" in capsys.readouterr().out


def test_keys_only_has_every_node_take_the_key_and_writes_no_list_whatever_a_list_would_lose(fleet, capsys):
    """The first step over lists made by hand. It adds to what each node takes the key from and takes nothing away, so the
    rule that holds a list back doesn't hold it; one node's list can then be rolled alone and tried before the rest."""
    tmp, calls, roll = fleet["tmp"], fleet["calls"], ("roll", "/NODE/etc/peers", "--by", "lane-x")
    hand = LIST + "vy-n-1 research@192.0.2.1\n"
    for n in "abc":
        (tmp / n / "etc" / "peers").write_text(hand)
    assert _main(fleet, *roll, "--apply") == 3 and "Nothing changed on any node (3 of 3 refuse)" in capsys.readouterr().err

    assert _main(fleet, *roll, "--keys-only") == 0
    out = capsys.readouterr().out
    assert out.count("  would take  its key is taken from 4 more address(es), 4 in all") == 3 and "names 5 machines" not in out
    assert out.endswith("nothing changed: --apply has each node take the key\n") and all(_keys(fleet, n) == [OTHER] for n in "abc")

    calls.clear()
    assert _main(fleet, *roll, "--keys-only", "--apply") == 0
    out = capsys.readouterr().out
    assert [out.count(f"  {w}  its key is taken from 4 more address(es), 4 in all") for w in ("would take", "taken")] == [3, 3]
    assert {k for _, k in calls} == {"probe", "accept?", "accept"}
    assert _main(fleet, *roll, "--keys-only", "--apply") == 0
    assert capsys.readouterr().out.count("  unchanged  its key is taken from each of them already (4 address(es))") == 6
    for n in "abc":
        assert len(_keys(fleet, n)) == 2 and "192.0.2.20" in _keys(fleet, n)[1] and (tmp / n / "etc" / "peers").read_text() == hand

    assert _main(fleet, *roll, "--on", "b", "--allow-shrink", "--apply") == 0
    assert [(tmp / n / "etc" / "peers").read_text() for n in "abc"] == [hand, LIST, hand]
    assert _main(fleet, "roll", "/NODE/etc/conf.json", "--by", "lane-x", "--keys-only") == 3
    assert "names no key for a node to take" in capsys.readouterr().err


def test_a_roll_changes_nothing_when_a_node_refuses_stops_at_the_first_that_fails_and_passes_by_one_it_cannot_reach(fleet, capsys):
    tmp, calls, faults = fleet["tmp"], fleet["calls"], fleet["faults"]
    assert _main(fleet, "roll", "/NODE/etc/peers", "--by", "lane-x", "--apply") == 0
    hand = LIST + "vy-n-1 research@192.0.2.1\n"
    (tmp / "b" / "etc" / "peers").write_text(hand)          # b would lose a machine: no node is touched, a included
    machine(fleet["reg"], "vy-cpu-11", "192.0.2.21")
    calls.clear()
    assert _main(fleet, "roll", "/NODE/etc/peers", "--by", "lane-x", "--apply") == 3
    assert "b names 5 machines" in (err := capsys.readouterr().err) and "Nothing changed on any node (1 of 3 refuse)" in err
    assert not [c for c in calls if c[1] in ("accept", "install")] and (tmp / "a" / "etc" / "peers").read_text() == LIST

    faults[("b", "install")] = (1, "disk full")             # b's install fails: a is done, c is not started
    assert _main(fleet, "roll", "/NODE/etc/peers", "--by", "lane-x", "--apply", "--allow-shrink") == 1
    assert "b: install failed (exit 1): disk full. The roll stopped there: done a; not done b, c" in capsys.readouterr().err
    assert "vy-cpu-11" in (tmp / "a" / "etc" / "peers").read_text() and (tmp / "c" / "etc" / "peers").read_text() == LIST
    assert (tmp / "b" / "etc" / "peers").read_text() == hand

    faults.clear()                                          # b's file isn't the register's once written: c is not started
    fleet["then"][("b", "install")] = lambda: (tmp / "b" / "etc" / "peers").write_text("written over\n")
    assert _main(fleet, "roll", "/NODE/etc/peers", "--by", "lane-x", "--apply", "--allow-shrink") == 1
    assert "b: /NODE/etc/peers reads unknown after its install" in (err := capsys.readouterr().err) and "not done b, c" in err
    assert (tmp / "c" / "etc" / "peers").read_text() == LIST
    fleet["then"].clear()
    (tmp / "b" / "etc" / "peers").write_text(hand)

    faults.clear()                                          # b is stopped: passed by and named, and c is done
    faults[("b", "probe")] = (255, "ssh: connect to host b port 22: Connection timed out")
    assert _main(fleet, "roll", "/NODE/etc/peers", "--by", "lane-x", "--apply", "--allow-shrink") == 1
    out = capsys.readouterr().out
    assert "b  passed by: unreachable" in out and "not done: b: `research deploy install NODE /NODE/etc/peers --by lane-x --apply`" in out
    assert "vy-cpu-11" in (tmp / "c" / "etc" / "peers").read_text() and (tmp / "b" / "etc" / "peers").read_text() == hand

    faults.clear()
    assert _main(fleet, "roll", "/NODE/etc/peers", "--by", "lane-x", "--apply", "--allow-shrink", "--on", "b") == 0
    assert all("vy-cpu-11" in (tmp / n / "etc" / "peers").read_text() and "vy-n-1" not in (tmp / n / "etc" / "peers").read_text() for n in "abc")
    assert _main(fleet, "roll", "/NODE/etc/nothing", "--by", "lane-x") == 3
    assert "lists /NODE/etc/nothing as a file made from the register" in capsys.readouterr().err


def test_a_roll_passes_by_a_node_that_has_no_machine_and_an_admin_only_node_unless_named(fleet, capsys):
    """A node staged ahead of its machine, and one whose machine was released, have nothing to install: no failure of the roll."""
    manifest = fleet["manifest"]
    manifest.write_text(manifest.read_text().replace("[nodes.b]\n", '[nodes.b]\nstaged = "created when the quota lands"\n')
                        .replace("[nodes.c]\n", "[nodes.c]\nadmin = true\n"))
    assert _main(fleet, "roll", "/NODE/etc/peers", "--by", "lane-x", "--apply") == 0
    assert "b  passed by: staged (created when the quota lands) and not in the register" in capsys.readouterr().out
    assert [(fleet["tmp"] / n / "etc" / "peers").exists() for n in "abc"] == [True, False, False]
    machine(fleet["reg"], "b", "192.0.2.30")                # the staged machine exists now: it is installed as any other
    assert _main(fleet, "roll", "/NODE/etc/peers", "--by", "lane-x", "--apply", "--on", "b", "c") == 0
    assert [(fleet["tmp"] / n / "etc" / "peers").read_text() for n in "bc"] == [LIST, LIST]

    machine(fleet["reg"], "b", None)                        # its machine is released: the others go on without it
    machine(fleet["reg"], "vy-cpu-11", "192.0.2.21")
    fleet["calls"].clear()
    assert _main(fleet, "roll", "/NODE/etc/peers", "--by", "lane-x", "--apply") == 0
    assert "b  passed by: released in the register" in capsys.readouterr().out and "b" not in {n for n, _ in fleet["calls"]}
    assert ["vy-cpu-11" in (fleet["tmp"] / n / "etc" / "peers").read_text() for n in "abc"] == [True, False, False]


def test_the_launchers_own_notes_clone_is_pulled_before_the_register_is_read_and_another_is_left_alone(tmp_path, monkeypatch):
    from research import notes_sync

    pulls, clone = [], tmp_path / ".research" / "notes"
    (clone / ".git").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(notes_sync, "pull", lambda root: pulls.append(root) or subprocess.CompletedProcess([], len(pulls) - 1, "", "no route"))
    for name in ("RESEARCH_MACHINES_D", "RESEARCH_NOTES"):
        monkeypatch.delenv(name, raising=False)
    assert deploy.stale_register() is None and pulls == [clone]
    assert f"the notes clone {clone} wasn't pulled, so the register of machines may be behind: no route" == deploy.stale_register()
    for name, where in (("RESEARCH_MACHINES_D", tmp_path / "kept" / "machines.d"), ("RESEARCH_NOTES", tmp_path / "kept")):
        monkeypatch.setenv(name, str(where))                # the control pod's: the steward keeps that clone
        assert deploy.stale_register() is None and len(pulls) == 2
        monkeypatch.delenv(name)


def test_register_is_a_copied_files_and_accept_a_register_made_files(tmp_path):
    head = '[nodes.n]\nrecord = "/r"\n[[{kind}]]\npath = "/p"\nnodes = ["n"]\n'
    made = 'template = "f"\nregister = true\nby = "deploy"\n'
    for kind, rest, why in (("file", 'by = "setup.sh"\nregister = true\n', "is made from the register: name its template `template`"),
                            ("file", 'by = "setup.sh"\nregister = true\nhold = "a field"\n', "only an entry install copies"),
                            ("tree", 'src = "d"\nregister = true\n', "`register` is true or false"),
                            ("file", 'src = "f"\nregister = "yes"\n', "`register` is true or false"),
                            ("file", 'src = "f"\naccept = "/k.pub"\n', "`accept` is the path, on the node, of the public key"),
                            ("file", made + 'accept = "k.pub"\n', "`accept` is the path")):
        with pytest.raises(ValueError, match=why):
            deploy.load(text=head.format(kind=kind) + rest)
    e = deploy.load(text=head.format(kind="file") + made + 'accept = "/k.pub"\n')["n"].entries[0]
    assert (e.src, e.by, e.register, e.accept) == ("f", None, True, "/k.pub")


@pytest.mark.parametrize("rest", ['src = "f"\nregister = true\n', 'template = "f"\nregister = true\n', 'template = "f"\nby = "deploy"\n',
                                  'template = "f"\nregister = true\nby = "deploy"\nsrc = "f"\n',
                                  'template = "f"\nregister = true\nby = "deploy"\nhold = "a field"\n'])
def test_a_file_made_from_the_register_names_its_template_as_older_deploy_code_never_copies(rest):
    """A `src` is what older deploy code copies onto a node as it is; its `by` it leaves alone; a held entry it loads only with
    a `src`. So an entry made from the register has `template` and `by`, and keeps `src` only while it is held."""
    with pytest.raises(ValueError, match="deploy code from before templates"):
        deploy.load(text='[nodes.n]\nrecord = "/r"\n[[file]]\npath = "/p"\nnodes = ["n"]\n' + rest)
    e = deploy.load(text='[nodes.n]\nrecord = "/r"\n[[file]]\npath = "/p"\nnodes = ["n"]\nsrc = "f"\nregister = true\nhold = "a field"\n')
    assert [(x.src, x.register, x.hold) for x in e["n"].entries] == [("f", True, "a field")]


#: `main` as this change found it: its deploy code knows no file made from the register, and a checkout behind main or the
#: control pod's watchers tree can still run it against main's deploy.toml
BEFORE = "245da271367f1fc0bea03b63b24408d18a4e8862"
MADE_BY = "research deploy, from the template and the register of machines"
BY = f'by = "{MADE_BY}"\n'


@pytest.fixture
def before(tmp_path, monkeypatch):
    """`main`'s deploy module from before templates, loaded by path from `git show`."""
    path = tmp_path / "deploy_before.py"
    path.write_bytes(subprocess.run(["git", "-C", str(deploy.SRC), "show", f"{BEFORE}:infra/ops/verity_ops/deploy.py"],
                                    capture_output=True, check=True).stdout)
    spec = importlib.util.spec_from_file_location("deploy_before", path)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "deploy_before", mod)
    spec.loader.exec_module(mod)
    assert not hasattr(mod, "fill"), "deploy code from before templates"
    monkeypatch.setattr(mod, "CERTS", str(tmp_path / "certs"))
    return mod


def test_deploy_code_from_before_templates_copied_a_src_template_onto_the_node_and_read_it_clean(fleet, before, capsys):
    """Why a template is never a `src`: in the shape this change first had (`src` the template, `register = true`), main's
    install of a whole node writes the template's text over the node's list, and main's drift then reads the node clean."""
    former = fleet["tmp"] / "former.toml"
    former.write_text(fleet["manifest"].read_text().replace("template = ", "src = ").replace(BY, ""))
    etc = fleet["tmp"] / "a" / "etc"
    _install(fleet)                                         # the list the register gives, as this change's code installs it

    def old(*argv: str) -> int:
        return before.main([*argv, "--no-fetch"], run=fleet["run"], manifest=former, repo=fleet["repo"])

    assert old("drift", "--on", "a") == 1
    assert "stale    /NODE/etc/peers  (installed" in capsys.readouterr().out          # "install it"
    assert old("install", "a", "--by", "lane-x", "--apply") == 0
    assert "installed     /NODE/etc/peers  <- src/peers" in capsys.readouterr().out
    assert (etc / "peers").read_text() == PEERS and (etc / "conf.json").read_text() == CONF
    assert old("drift", "--on", "a") == 0 and "ok       2 of 2" in capsys.readouterr().out


def test_deploy_code_from_before_templates_writes_nothing_a_template_makes_and_drift_reads_it_as_made_on_the_node(fleet, before, capsys):
    """This change's shape (`template` and `by`, no `src`): main's install of a whole node, or of the path by name, leaves
    the list as it is and never writes the template; its drift reads the list as a file made on the node."""
    etc = fleet["tmp"] / "a" / "etc"
    _install(fleet)
    nodes = before.load(fleet["manifest"])
    assert [(e.path, e.src, e.by) for e in nodes["a"].entries] == [
        ("/NODE/etc/peers", None, MADE_BY), ("/NODE/etc/conf.json", None, MADE_BY)]

    def old(*argv: str) -> int:
        return before.main([*argv, "--no-fetch"], run=fleet["run"], manifest=fleet["manifest"], repo=fleet["repo"])

    node = _entries(fleet["tmp"] / "a")
    assert old("install", "a", "--by", "lane-x", "--apply") == 0 and "peers" not in capsys.readouterr().out
    assert old("install", "a", "--by", "lane-x", "--apply", "/NODE/etc/peers") == 3
    assert "not in deploy.toml for a: /NODE/etc/peers" in capsys.readouterr().err
    assert _entries(fleet["tmp"] / "a") == node and (etc / "peers").read_text() == LIST
    assert old("drift", "--on", "a") == 1
    out = capsys.readouterr().out
    assert "ok       1 of 2" in out and f"missing  /NODE/etc/conf.json  (by {MADE_BY}: re-run it)" in out


def test_deploy_code_from_before_templates_copies_no_template_from_the_manifest(before):
    """The tree's deploy.toml as main's deploy code reads it: each file made from the register is, on every node that lists
    it, a file made on the node (`by`) or a held one, and no entry it would copy is a template."""
    made = {(n, e.path): e for n, node in deploy.load().items() for e in node.entries if e.register}
    old = {(n, e.path): e for n, node in before.load(deploy.MANIFEST).items() for e in node.entries}
    assert made and set(made) <= set(old)
    assert {k: "by" if old[k].by else "held" if old[k].hold else "copied" for k in made} == \
        {k: "held" if e.hold else "by" for k, e in made.items()}
    templates = {e.src for e in made.values()}
    assert not [k for k, e in old.items() if e.src in templates and not e.hold]


MADE = {"/etc/vy/peers": {"tools/research/src/research/pods/nebius/peers", "tools/research/src/research/pods/nebius/peers-challenger"},
        "/etc/vy/verifier.json": {"tools/research/src/research/pods/nebius/verifier/verifier.json"},
        "/data/watchers/ssh_config": {"infra/control/watchers_ssh_config"}}


def _stand_in(tmp_path: Path, private: bool) -> deploy.Register:
    """A register holding every machine the manifest's templates name, each at a documentation address."""
    asked = {w for srcs in MADE.values() for s in srcs for w, _ in deploy.PLACE.findall((deploy.SRC / s).read_text())}
    names = {w for w in asked if not deploy.WILD.search(w)} | {"vy-nebius-2", "vy-nebius-cpu-1", "vy-nebius-cpu-2", "vy-nebius-cpu-10"}
    for i, n in enumerate(sorted(names)):
        machine(tmp_path, n, f"192.0.2.{i + 1}", **({"private_host": f"198.51.100.{i + 1}"} if private else {}))
    machine(tmp_path, "vy-nebius-1", None)
    return deploy.register({"RESEARCH_MACHINES_D": str(tmp_path)})


def test_the_manifests_address_files_are_made_from_the_register_and_each_reads_as_its_reader_reads_it(tmp_path):
    entries = {(e.path, e.src): e for node in deploy.load().values() for e in node.entries if e.register}
    assert {p: {s for q, s in entries if q == p} for p in MADE} == MADE and {p for p, _ in entries} == set(MADE)
    reg = _stand_in(tmp_path, private=True)
    text = {s: deploy.fill((deploy.SRC / s).read_text(), reg, s) for srcs in MADE.values() for s in srcs}
    assert all("{{" not in t for t, _ in text.values())

    peers, names = text["tools/research/src/research/pods/nebius/peers"]
    lines = [ln.split() for ln in peers.splitlines() if ln and not ln.startswith("#")]
    cpus = sorted(int(n.rsplit("-", 1)[1]) for n in reg.live if n.startswith("vy-nebius-cpu-"))
    assert {1, 2, 10} <= set(cpus) and [ln[0] for ln in lines] == ["vy-nebius-2", *(f"vy-nebius-cpu-{i}" for i in cpus)] == names
    assert all(ln[-1] == f"research@{reg.live[ln[0]]['host']}" and "/home/research/.ssh/vy-cluster" in ln for ln in lines)
    shared = {n for n, node in deploy.load().items() for e in node.entries if e.src and e.src.endswith("/peers")}
    assert {f"vy-nebius-cpu-{i}" for i in range(1, 18)} | {"vy-nebius-2"} <= shared
    assert {e.accept for e in entries.values() if e.src.endswith("/peers")} == {"/home/research/.ssh/vy-cluster.pub"}

    challenger, names = text["tools/research/src/research/pods/nebius/peers-challenger"]
    lines = [ln.split() for ln in challenger.splitlines() if ln and not ln.startswith("#")]
    assert lines[0] == ["vy-challenger-1", f"research@{reg.live['vy-challenger-1']['private_host']}"]
    assert all(ln[-1].endswith(f"@{reg.live[ln[0]]['host']}") for ln in lines[1:]) and len(lines) == len(names) > 5

    verifier = json.loads(text["tools/research/src/research/pods/nebius/verifier/verifier.json"][0])
    assert verifier["challenger"]["ssh"].endswith(f"research@{reg.live['vy-challenger-1']['host']}")

    config = text["infra/control/watchers_ssh_config"][0].splitlines()
    hosts = {ln.split()[1:][1]: nxt.split()[1] for ln, nxt in zip(config, config[1:]) if ln.startswith("Host ") and "*" not in ln}
    assert hosts and all(reg.live[n]["host"] == h for n, h in hosts.items())


def test_the_challengers_list_is_not_made_until_the_register_holds_its_hosts_private_address(tmp_path):
    """Its own line names the host by the address its interface has, which no register file holds: the fact isn't invented."""
    reg = _stand_in(tmp_path, private=False)
    src = "tools/research/src/research/pods/nebius/peers-challenger"
    with pytest.raises(deploy.NoRegister, match=r"vy-challenger-1.toml has no `private_host`, which .*peers-challenger:\d+ asks"):
        deploy.fill((deploy.SRC / src).read_text(), reg, src)
    held = [e.hold for e in deploy.load()["vy-challenger-1"].entries if e.src == src]
    assert held and "`private_host`" in held[0], "held, so the host's other files install and its drift says why"


def test_the_verifiers_file_is_held_while_its_template_gives_no_peer_its_addresses():
    """The verifier's gate admits a party only from one of a peer's `addresses` in its verifier.json (live_verifier/host.py's
    `node_of`): an install of a template that gives a peer none would shut that peer's parties out, so the entry is held, on
    both auditor hosts, until the template gives every peer its addresses."""
    src = "tools/research/src/research/pods/nebius/verifier/verifier.json"
    peers = json.loads((deploy.SRC / src).read_text())["peers"]
    entries = [e for n in ("vy-auditor-1", "vy-auditor-2") for e in deploy.load()[n].entries if e.path == "/etc/vy/verifier.json"]
    assert len(entries) == 2 and all(e.register and e.src == src for e in entries)
    assert all("addresses" in p for p in peers.values()) or all(e.hold for e in entries)


def test_an_install_from_a_ref_whose_manifest_copies_a_file_this_code_makes_from_the_register_is_refused(fleet, monkeypatch, capsys):
    """An install reads REF's deploy.toml. One from before templates copies the list of addresses as it is, over the node's,
    and no shrink rule reads a copied file; one whose template is a `src` would copy the template: each is refused, exit 3,
    before the node is read. A REF with the present shape installs as before."""
    repo, tmp = fleet["repo"], fleet["tmp"]
    manifest = repo / "deploy.toml"
    monkeypatch.setattr(deploy, "MANIFEST", manifest)
    monkeypatch.setattr(deploy, "SRC", repo)
    now = fleet["manifest"].read_text()
    copies = now.replace('template = "src/peers"\nregister = true\nby = "research deploy, from the template and the register of '
                         'machines"\naccept = "/NODE/ssh/k.pub"\n', 'src = "src/peers.list"\n')
    as_src = now.replace('template = "src/peers"\nregister = true\nby = "research deploy, from the template and the register of '
                         'machines"\n', 'src = "src/peers"\nregister = true\n')
    assert copies != now != as_src
    old = _commit(repo, {"deploy.toml": copies, "src/peers.list": LIST}, "before templates").strip()
    early = _commit(repo, {"deploy.toml": as_src}, "the template as a src").strip()
    _commit(repo, {"deploy.toml": now}, "templates")
    _git(repo, "push", "-q", "origin", "main")
    fleet["calls"].clear()

    def main(*argv: str) -> int:
        return deploy.main([*argv, "--no-fetch"], run=fleet["run"], manifest=manifest, repo=repo)

    for paths in ((), ("/NODE/etc/peers",)):
        assert main("install", "a", "--by", "lane-x", "--apply", "--ref", old, *paths) == 3
        assert (f"research deploy: {old}: its deploy.toml copies /NODE/etc/peers to a as it is, and this code makes it from the "
                "register of machines") in capsys.readouterr().err
    assert main("install", "a", "--by", "lane-x", "--apply", "--ref", early) == 3
    assert "'/NODE/etc/peers' is made from the register: name its template `template`" in capsys.readouterr().err
    assert fleet["calls"] == [] and not (tmp / "a" / "etc" / "peers").exists()
    assert main("install", "a", "--by", "lane-x", "--ref", "origin/main") == 0
    assert "would install /NODE/etc/peers  <- src/peers" in capsys.readouterr().out
