"""The verifier host's packaging (`pods/nebius/verifier/`) against the gate it deploys
(`verity/protocols/accounting/memory/poub/attester/live_verifier/host.py`): the sshd and tunnel files open the gate's ports and no
others, `pin.sh` runs no unpinned code, `setup.sh` gives a tunnel key a pty only where its client forces one, the
deployed `verifier.json` is one the gate runs a peer from, and each peer's figures are what this tree's data sheets give
(`figures.py`, which `pin.sh` runs)."""
from __future__ import annotations

import importlib.util
import json
import re
import sys
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
PACKAGING = REPO / "tools/research/src/research/pods/nebius/verifier"
LIVE_VERIFIER = REPO / "verity/protocols/accounting/memory/poub/attester/live_verifier"
_spec = importlib.util.spec_from_file_location("live_verifier_host", LIVE_VERIFIER / "host.py")
H = sys.modules[_spec.name] = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)
_spec = importlib.util.spec_from_file_location("verifier_figures", PACKAGING / "figures.py")
F = sys.modules[_spec.name] = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(F)
TEMPLATE = json.loads((PACKAGING / "verifier.json").read_text())


def test_the_runs_ports_are_the_tunnels_and_the_challengers_are_the_gates():
    conf = (PACKAGING / "12-vy-verifier.conf").read_text()
    prover = conf.split("Match User prover")[1].split("Match User")[0]
    opened = re.search(r"^\s*PermitOpen (.*)$", prover, re.M).group(1).split()
    assert opened == [f"127.0.0.1:{p}" for p in H.PORTS]
    challenger = conf.split("Match User challenger")[1]
    listened = re.search(r"^\s*PermitListen (.*)$", challenger, re.M).group(1).split()
    assert listened == [f"127.0.0.1:{p}" for p in H.GATE_PORTS]
    tunnel = (PACKAGING / "vy-verifier-tunnel").read_text()
    pattern = re.search(r"\$4 ~ /\^(.*)\$/", tunnel).group(1).replace("\\.", ".")
    assert [p for p in range(7000, 8000) if re.fullmatch(pattern, f"127.0.0.1:{p}")] == list(H.PORTS)


def test_pin_sh_runs_no_unpinned_code_and_logs_each_pin_before_current_names_it():
    """Infra's review: elan is a release, its archive checked against a pinned sha256 before it runs (never a
    script fetched from a branch), and the pin is logged (`host.py pinned`) before `current` moves to it."""
    code = "\n".join(ln for ln in (PACKAGING / "pin.sh").read_text().splitlines() if not ln.lstrip().startswith("#"))
    assert "raw.githubusercontent.com" not in code and not re.search(r"\|\s*(ba)?sh\b", code)
    assert re.search(r"^ELAN=v\d+\.\d+\.\d+ ELAN_SHA256=[0-9a-f]{64}$", code, re.M)
    assert "https://github.com/leanprover/elan/releases/download/$ELAN/" in code
    assert code.index('echo "$ELAN_SHA256  $e/elan.tar.gz" | sha256sum -c') < code.index('"$e/elan-init"')
    assert code.index('host.py" pinned "$c"') < code.index('ln -sfn "$c" "$O/current.tmp"')
    assert code.index('vy-verifier-figures "$T" || die') < code.index('host.py" pinned "$c"')


def test_the_pinned_archive_holds_what_the_figures_are_made_from():
    archive = re.search(r"git archive COMMIT (.*?)\|", (PACKAGING / "pin.sh").read_text(), re.S).group(1)
    assert {"verity", "benchmarks/poub_session", "calibration/hardware", "calibration/rate_squaring/gamma.py"} <= set(archive.replace("#", " ").replace("\\", " ").split())


def test_a_tunnel_key_gets_a_pty_only_while_the_challengers_pinned_client_forces_one():
    """ssh ends a session whose forced pty (`-tt`) the server refuses, so the prover's key, whose client asks none
    (`ssh -T`), has no `pty`, and the challenger's keeps it only while serve.py forces one."""
    setup = (PACKAGING / "setup.sh").read_text()
    assert "opts=restrict,port-forwarding\n" in setup and '[ "$acct" = prover ] || opts+=,pty\n' in setup
    serve = (REPO / "verity/protocols/run/verifier/live_challenger/serve.py").read_text()
    assert re.search(r'^SSH = \["ssh", "-tt"', serve, re.M), (
        "serve.py no longer forces a pty: drop `pty` from the challenger's tunnel key in verifier/setup.sh, and this check")


def test_the_deployed_verifier_json_has_the_shape_the_gates_tests_run_on_and_the_gate_runs_its_peers(tmp_path):
    def shape(v):
        if isinstance(v, dict):
            return {k: shape(x) for k, x in v.items()}
        return float if isinstance(v, (int, float)) and not isinstance(v, bool) else type(v)

    tested = json.loads((LIVE_VERIFIER.parents[6] / "tests" / "verity" / "protocols" / "accounting" / "memory" / "poub" / "attester" / "live_verifier" / "verifier.json").read_text())
    assert shape({k: v for k, v in TEMPLATE.items() if k != "peers"}) == shape({k: v for k, v in tested.items() if k != "peers"})
    host = H.Host(root=tmp_path, main=tmp_path / "main-first-parent.txt")
    host.main.write_text("")
    assert TEMPLATE["peers"]
    for peer, entry in TEMPLATE["peers"].items():
        req = {"peer": peer, "port": H.PORTS.start, "period": "p", "pin": "0" * 128, "sessions": 1}
        argv = H.verifier_argv(host, "r", req, TEMPLATE)
        assert argv[argv.index("--figures") + 1] == entry["figures"]
        assert argv[argv.index("--figures-sha512") + 1] == entry["figures_sha512"]


@pytest.mark.parametrize("peer", sorted(TEMPLATE["peers"]))
def test_each_peers_figures_sha512_is_the_digest_of_the_figures_this_trees_data_sheets_give(peer):
    """A change to a data sheet that moves a node kind's figures fails here: run `python benchmarks/poub_session/sheets.py
    NODE`, and put `timing.Figures.of_json(...).sha512()` of what it writes in the template's `figures_sha512`."""
    entry = TEMPLATE["peers"][peer]
    try:
        files = F.plan(REPO, {"peers": {peer: entry}}, python=sys.executable)
    except F.Refused as e:
        pytest.fail(str(e))
    assert list(files) == [F.DIR / f"{entry['node']}.json"]


def test_each_peers_figures_file_is_one_the_deploy_table_names_on_the_templates_nodes():
    rows = tomllib.loads((PACKAGING.parent / "deploy.toml").read_text())["file"]
    on = next(e["nodes"] for e in rows if e.get("src", "").endswith("/verifier/verifier.json"))
    made = {e["path"]: e for e in rows if e.get("by", "").endswith("/verifier/figures.py")}
    assert {e["figures"] for e in TEMPLATE["peers"].values()} == set(made)
    assert all(e["nodes"] == on for e in made.values())
    assert next(e["nodes"] for e in rows if e["path"] == "/usr/local/sbin/vy-verifier-figures") == on


def _settings(**peers):
    node = "rtx-pro-6000-server-x8"
    good = {"node": node, "figures": str(F.DIR / f"{node}.json"), "figures_sha512": TEMPLATE["peers"]["vy-nebius-2"]["figures_sha512"],
            "ingress_bytes_per_s": 4.4e9}
    return {"peers": {name: {**good, **change} for name, change in peers.items()}}


def test_figures_writes_what_sheets_writes_root_readable_once_every_peer_agrees(tmp_path):
    conf, out = tmp_path / "verifier.json", tmp_path / "figures"
    settings = _settings(a={}, b={})
    for e in settings["peers"].values():
        e["figures"] = str(out / f"{e['node']}.json")
    conf.write_text(json.dumps(settings))
    assert F.main([str(REPO)], config=conf, out=out, python=sys.executable) == 0
    written = out / "rtx-pro-6000-server-x8.json"
    assert [p.name for p in out.iterdir()] == [written.name] and written.stat().st_mode & 0o777 == 0o644
    text, digest = F.figures(REPO, "rtx-pro-6000-server-x8", sys.executable)
    assert written.read_text() == text and digest == settings["peers"]["a"]["figures_sha512"]


@pytest.mark.parametrize(("change", "why"), [
    ({"figures_sha512": "0" * 128}, "the peer 'b' has figures_sha512 " + "0" * 128 + ", but the figures"),
    ({"node": None}, "the peer 'b' names no node kind"),
    ({"node": "../x"}, "the peer 'b' names no node kind"),
    ({"figures": "/etc/vy/elsewhere.json"}, "the peer 'b' has `figures` '/etc/vy/elsewhere.json'"),
])
def test_figures_refuses_a_peer_it_cant_make_or_whose_digest_moved_and_writes_nothing(tmp_path, capsys, change, why):
    conf, out = tmp_path / "verifier.json", tmp_path / "figures"
    settings = _settings(a={}, b=change)
    for e in settings["peers"].values():
        if e.get("figures") == str(F.DIR / f"{e.get('node')}.json"):
            e["figures"] = str(out / f"{e['node']}.json")
    conf.write_text(json.dumps(settings))
    assert F.main([str(REPO)], config=conf, out=out, python=sys.executable) == 3
    err = capsys.readouterr().err
    assert why in err and not out.exists()
    if "figures_sha512" in change:
        assert settings["peers"]["a"]["figures_sha512"] in err and "the template" in err and "the data sheets" in err
