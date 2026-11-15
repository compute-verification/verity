"""The verifier host's packaging (`pods/nebius/verifier/`) against the gate it deploys
(`verity/protocols/resource_accounting/memory/poub/attester/live_verifier/host.py`): the sshd and tunnel files open the gate's ports and no
others, `pin.sh` runs no unpinned code, `setup.sh` gives a tunnel key a pty only where its client forces one, and the
deployed `verifier.json` has the shape the gate's own tests run on."""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PACKAGING = REPO / "tools/research/src/research/pods/nebius/verifier"
LIVE_VERIFIER = REPO / "verity/protocols/resource_accounting/memory/poub/attester/live_verifier"
_spec = importlib.util.spec_from_file_location("live_verifier_host", LIVE_VERIFIER / "host.py")
H = sys.modules[_spec.name] = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)


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


def test_a_tunnel_key_gets_a_pty_only_while_the_challengers_pinned_client_forces_one():
    """ssh ends a session whose forced pty (`-tt`) the server refuses, so the prover's key, whose client asks none
    (`ssh -T`), has no `pty`, and the challenger's keeps it only while serve.py forces one."""
    setup = (PACKAGING / "setup.sh").read_text()
    assert "opts=restrict,port-forwarding\n" in setup and '[ "$acct" = prover ] || opts+=,pty\n' in setup
    serve = (REPO / "verity/protocols/compose/verifier/live_challenger/serve.py").read_text()
    assert re.search(r'^SSH = \["ssh", "-tt"', serve, re.M), (
        "serve.py no longer forces a pty: drop `pty` from the challenger's tunnel key in verifier/setup.sh, and this check")


def test_the_deployed_verifier_json_has_the_shape_the_gates_tests_run_on():
    def shape(v):
        if isinstance(v, dict):
            return {k: shape(x) for k, x in v.items()}
        return float if isinstance(v, (int, float)) and not isinstance(v, bool) else type(v)

    deployed = json.loads((PACKAGING / "verifier.json").read_text())
    tested = json.loads((LIVE_VERIFIER / "tests/verifier.json").read_text())
    peer = {k: v for k, v in next(iter(tested["peers"].values())).items() if k != "addresses"}
    assert shape({k: v for k, v in deployed.items() if k != "peers"}) == shape({k: v for k, v in tested.items() if k != "peers"})
    assert deployed["peers"] and all(shape(p) == shape(peer) for p in deployed["peers"].values())
