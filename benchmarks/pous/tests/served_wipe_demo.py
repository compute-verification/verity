"""A stand-in for `streamed_wipe/demo.sh` in `test_pous_served_wipe`, taking its options and writing its outputs under
`$OUT` (period.json, wipe.json, wipe.case.json, summary.json) with its exit: one 32 MiB segment kept in host memory
(`wipe_store.HostMemory`, period.json's `wipe.store` `host`) in place of the card, against `wipe_side.py`, the challenger's
stand-in on a host of its own, in place of the side `--challenger-command` starts on `--challenger`. The challenger
period.json names is `wipe_party.Party.of`'s, written to `$OUT/party.json` before the stand-in replaces it; the arguments
and the lease it ran under go to `$OUT/demo-argv.json`. The Python judges alone: a `--wipe-protocol` is only recorded.
`python served_wipe_demo.py OUT LEASE [DEMO_OPTIONS]`: the test's shim passes `$OUT` and `$GPU_LEASE_UUID`."""

from __future__ import annotations

import argparse
import contextlib
import json
import shlex
import sys
import tempfile
from pathlib import Path

from verity.core.protocols.pous import wipe_party as WP
from verity.core.protocols.pous import wipe_period as W

SIDE = Path(__file__).resolve().parents[3] / "verity" / "core" / "protocols" / "pous" / "tests" / "wipe_side.py"


def main(argv: list[str]) -> int:
    out, lease, *argv = argv
    ap = argparse.ArgumentParser()
    for o in ("--node", "--challenger", "--challenger-address", "--challenger-command", "--reads", "--wipe-protocol"):
        ap.add_argument(o)
    ap.add_argument("--no-lean", action="store_true")
    a = ap.parse_args(argv)
    out = Path(out)
    (out / "demo-argv.json").write_text(json.dumps({"argv": argv, "lease": lease or None}))
    doc = {"tools": {}, "challenger_host": a.challenger,
           "challenger": {"wipe": {"command": shlex.split(a.challenger_command), "address": a.challenger_address}},
           "wipe": {"store": "host", "reads": int(a.reads)}}
    of = WP.Party.of

    def stand_in(cls, period, node, device, extra=()):
        named = of(period, node, device, extra)
        (out / "party.json").write_text(json.dumps({"host": named.host, "command": named.command,
                                                    "address": named.address}))
        return WP.Party(named.host, (sys.executable, str(SIDE), str(device)))

    WP.Party.of = classmethod(stand_in)  # type: ignore[method-assign]
    with tempfile.TemporaryDirectory() as t:
        period = Path(t) / "periods" / "p0"
        period.mkdir(parents=True)
        (period / "period.json").write_text(json.dumps(doc, indent=1, sort_keys=True))
        (out / "period.json").write_text(json.dumps(doc, indent=1, sort_keys=True))
        (Path(t) / "periods" / "wipe-challenger").mkdir()
        with open(out / "summary.json", "w") as f, contextlib.redirect_stdout(f):
            return W.main(["--period", str(period), "--out", str(out), "--node", a.node])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
