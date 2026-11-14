"""`research deploy`'s files made from the register of machines: an install fills the entry's template from the register and
writes that, drift compares the node with what the register gives now, nothing is written from a register that can't be read,
no machine leaves a list unasked, the list's key is taken from every machine it names, and a roll puts one file on every node
one at a time. The register is a directory of this test's, and each node is a directory of this machine: the runner turns
`/NODE/` into the node's own directory, so one manifest path is a different file on each."""
from __future__ import annotations

import json
import subprocess
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
    (tmp / name / "ssh").mkdir()
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
src = "src/peers"
register = true
accept = "/NODE/ssh/k.pub"
nodes = ["a", "b", "c"]

[[file]]
path = "/NODE/etc/conf.json"
src = "src/conf.json"
register = true
nodes = ["a"]
""")
    calls: list[tuple[str, str]] = []
    faults: dict[tuple[str, str], tuple[int, str]] = {}
    then: dict[tuple[str, str], Callable[[], None]] = {}

    def run(node: str, argv: list[str], stdin: bytes | None) -> subprocess.CompletedProcess:
        root = f"{tmp_path}/{node}/"
        kind = next(SCRIPTS[a] for a in argv if a in SCRIPTS) + ("" if argv[0] == "sudo" or deploy.PROBE in argv else "?")
        calls.append((node, kind))
        if (node, kind) in faults:
            rc, err = faults[(node, kind)]
            return subprocess.CompletedProcess(argv, rc, b"", err.encode())
        r = local(node, [a if a in SCRIPTS else a.replace("/NODE/", root) for a in argv],
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
    assert _read(fleet)["state"] == "unknown"
    _install(fleet)
    assert _read(fleet)["state"] == "ok"
    machine(fleet["reg"], "vy-cpu-3", "192.0.2.13")        # a machine joins: every installed list is behind the register
    e = _read(fleet)
    assert e["state"] == "stale" and "by lane-x" in e["detail"] and "the register or its template has changed since" in e["detail"]
    path.write_text(path.read_text() + "vy-cpu-3 research@192.0.2.13\n")
    e = _read(fleet)
    assert e["state"] == "unknown" and "written by hand" in e["detail"]
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
    assert "names 5 machines now, 4 after; drops vy-n-1 (released in the register)" in capsys.readouterr().out
    assert path.read_text() == hand
    assert _install(fleet, allow_shrink=True)["files"][0]["machines"]["dropped"] == ["vy-n-1"] and path.read_text() == LIST

    path.write_text(LIST.replace("vy-cpu-10 -i /k research@192.0.2.20\n", ""))       # a list that grows needs no leave
    assert _install(fleet)["files"][0]["machines"] == {"now": 3, "after": 4, "added": ["vy-cpu-10"], "dropped": [], "says": {}}
    assert path.read_text() == LIST


def test_the_machines_a_file_names_are_whole_words_of_its_lines_comments_apart():
    text = "# vy-cpu-1 in a comment\n\nvy-cpu-10 -i /k research@192.0.2.20\n  Host n2 vy-n-2 x\n{\"vy-gate\": 1}\n"
    assert deploy.named(text, ["vy-cpu-1", "vy-cpu-10", "vy-n-2", "vy-gate", "research", "absent"]) == \
        (3, ["research", "vy-cpu-10", "vy-gate", "vy-n-2"])


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

    assert _install(fleet)["files"][0]["accept"] == {"line": "new", "added": 4, "sources": 4}
    assert _keys(fleet) == [OTHER, f'from="{every}" {KEY}']
    assert _install(fleet)["files"][0]["accept"] == {"line": "kept", "added": 0, "sources": 4}

    # the key's line as a machine has it: sources no register holds, an option, another key before and after it
    ak.write_text(f'{OTHER}\nrestrict,from="10.0.0.5,192.0.2.11",pty {KEY}\n# a comment\n{OTHER} again\n')
    assert _install(fleet)["files"][0]["accept"] == {"line": "extended", "added": 3, "sources": 5}
    assert _keys(fleet) == [OTHER, f'restrict,from="10.0.0.5,192.0.2.11,192.0.2.12,192.0.2.20,192.0.2.2",pty {KEY}',
                            "# a comment", f"{OTHER} again"]
    assert {p.name.split(".")[0] for p in (ak.parent / ".deploy-prev").iterdir()} == {"authorized_keys"}
    record = [json.loads(ln) for ln in (fleet["tmp"] / "a/research/deploy/record.jsonl").read_text().splitlines()]
    assert [(r["path"], r["added"], r["by"]) for r in record if "accept" in r] == [(str(ak), 4, "lane-x"), (str(ak), 3, "lane-x")]

    ak.write_text(f"{KEY}\n")                               # no from=: taken from anywhere already, and never narrowed
    assert _install(fleet)["files"][0]["accept"]["line"] == "kept" and _keys(fleet) == [KEY]

    (ak.parent / "k.pub").unlink()                          # no key here: refused before the list is touched
    (fleet["tmp"] / "a" / "etc" / "peers").write_text("by hand\n")
    for apply in (True, False):
        with pytest.raises(deploy.Refused, match="a: no public key at /NODE/ssh/k.pub"):
            _install(fleet, apply=apply, allow_shrink=True)
    assert (fleet["tmp"] / "a" / "etc" / "peers").read_text() == "by hand\n"


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
    for kind, rest, why in (("file", 'by = "setup.sh"\nregister = true\n', "`register` is true or false, for a file whose `src`"),
                            ("tree", 'src = "d"\nregister = true\n', "`register` is true or false"),
                            ("file", 'src = "f"\nregister = "yes"\n', "`register` is true or false"),
                            ("file", 'src = "f"\naccept = "/k.pub"\n', "`accept` is the path, on the node, of the public key"),
                            ("file", 'src = "f"\nregister = true\naccept = "k.pub"\n', "`accept` is the path")):
        with pytest.raises(ValueError, match=why):
            deploy.load(text=head.format(kind=kind) + rest)
    e = deploy.load(text=head.format(kind="file") + 'src = "f"\nregister = true\naccept = "/k.pub"\n')["n"].entries[0]
    assert (e.register, e.accept) == (True, "/k.pub")


MADE = {"/etc/vy/peers": {"tools/research/src/research/pods/nebius/peers", "tools/research/src/research/pods/nebius/peers-challenger"},
        "/etc/vy/verifier.json": {"tools/research/src/research/pods/nebius/verifier/verifier.json"},
        "/data/watchers/ssh_config": {"infra/control/watchers_ssh_config"}}


def _stand_in(tmp_path: Path, private: bool) -> deploy.Register:
    """A register holding every machine the manifest's templates name, each at a documentation address."""
    asked = {w for srcs in MADE.values() for s in srcs for w, _ in deploy.PLACE.findall((deploy.SRC / s).read_text())}
    names = {w for w in asked if not deploy.WILD.search(w)} | {"vy-nebius-2", "vy-nebius-cpu-1", "vy-nebius-cpu-2", "vy-nebius-cpu-10"}
    for i, n in enumerate(sorted(names)):
        machine(tmp_path, n, f"192.0.2.{i + 1}", **({"private_host": f"10.0.0.{i + 1}"} if private else {}))
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
