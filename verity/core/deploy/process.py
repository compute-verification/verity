"""The local-process adapter: a party run in a process of its own on this machine, over TCP to the run that spawned it
(`note:memory-accounting/20261026T2305Z-draft-deployment-api-v3` §4, the addendum of 00:08Z).

`spawn(spec)` starts `python -m verity.core.deploy.process` and writes `spec` to its stdin as JSON; the child calls
`runtime.run` for its parties and prints the run's summary (`RunResult.summary`) as its last line of stdout. Core
imports no protocol, so the spec names the code the child loads, each as `module:attr`:

- `protocols`: `{id: "module:attr"}`, each a `ProtocolDef`;
- `keys`: a callable that gives a party's key (`keys(party)`), and `verify`, the peer-key check;
- `path`: directories to put on `sys.path` first (a protocol defined in a test suite, say);

and the run's own fields: `config`, `parties_here`, `mode`, `state_dir`, `seed` (hex, or null), `clock_step_ns`, and
`dial` (`{peer: "HOST:PORT"}`, where each peer listens). The child exits 0 when its run concluded, refused or not, and
nonzero when it couldn't run; the spawning run counts either way (`runtime.run`).
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Mapping
from typing import Any

from .runtime import _load, run


def spawn(spec: Mapping[str, Any]) -> subprocess.Popen[bytes]:
    """The child process running `spec`, its stdout a pipe for the summary."""
    child = subprocess.Popen([sys.executable, "-m", "verity.core.deploy.process"], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE)
    assert child.stdin is not None
    child.stdin.write(json.dumps(dict(spec)).encode())
    child.stdin.close()
    return child


def main() -> int:
    spec = json.loads(sys.stdin.read())
    sys.path[:0] = list(spec.get("path", ()))
    protocols = {pid: _load(ref) for pid, ref in spec["protocols"].items()}
    keys = _load(spec["keys"]) if spec.get("keys") else None
    seed = spec.get("seed")
    result = run(spec["config"], spec["parties_here"], spec["mode"], spec.get("state_dir"), protocols=protocols,
                 keys=None if keys is None else {p: keys(p) for p in spec["parties_here"]},
                 verify=_load(spec["verify"]) if spec.get("verify") else None,
                 seed=None if seed is None else bytes.fromhex(seed), clock_step_ns=spec.get("clock_step_ns", 0),
                 dial_to=spec.get("dial", {}), authenticate=spec.get("authenticate", True))
    print(json.dumps(result.summary(), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
