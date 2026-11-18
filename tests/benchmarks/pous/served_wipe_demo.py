"""A stand-in for `streamed_wipe/demo.sh` in `test_pous_served_wipe`, taking its options and writing its outputs under
`$OUT` (period.json, wipe.json, wipe.case.json, summary.json) with its exit: one 32 MiB segment kept in host memory
(`wipe_store.HostMemory`, period.json's `wipe.store` `host`) in place of the card, against `wipe_side.py`, the challenger's
stand-in on a host of its own, in place of the side `--challenger-command` starts on `--challenger`, its key trusted
only when it is the one `--trusted-sha512` pins (period.json's `wipe.trusted_sha512`), as demo.sh has it. The
challenger period.json names is `Side.of`'s, written to `$OUT/party.json` before the stand-in replaces it;
the arguments and the lease it ran under go to `$OUT/demo-argv.json`. The `--wipe-protocol` the wrapper always gives is
the period's Lean judge, stood in by one that answers each case as the Python answered it.
`python served_wipe_demo.py OUT LEASE [DEMO_OPTIONS]`: the test's shim passes `$OUT` and `$GPU_LEASE_UUID`."""

from __future__ import annotations

import argparse
import contextlib
import json
import shlex
import sys
import tempfile
from pathlib import Path

from verity.protocols.resource_accounting.memory.pous.prover import wipe_run
from verity.runtime.stream import Side

SIDE = Path(__file__).resolve().parents[2] / "verity" / "protocols" / "resource_accounting" / "memory" / "pous" / "verifier" / "wipe_side.py"


def main(argv: list[str]) -> int:
    out, lease, *argv = argv
    ap = argparse.ArgumentParser()
    for o in ("--node", "--challenger", "--challenger-address", "--challenger-command", "--trusted-sha512", "--reads",
              "--wipe-protocol"):
        ap.add_argument(o)
    a = ap.parse_args(argv)
    out = Path(out)
    (out / "demo-argv.json").write_text(json.dumps({"argv": argv, "lease": lease or None}))
    judge = a.wipe_protocol
    doc = {"tools": {} if judge is None else {"wipe-protocol": judge}, "challenger_host": a.challenger,
           "challenger": {"wipe": {"command": shlex.split(a.challenger_command), "address": a.challenger_address}},
           "wipe": {"store": "host", "reads": int(a.reads), "trusted_sha512": a.trusted_sha512}}

    def stand_in(period, node, key, local, what=None):
        assert (key, what) == ("wipe", "the wipe"), "the wipe's side is period.json's challenger.wipe"
        named = Side.of(period, node, key, local, what)
        (out / "party.json").write_text(json.dumps({"host": named.host, "command": named.command,
                                                    "address": named.address}))
        return Side(named.host, (sys.executable, str(SIDE), local[local.index("--device") + 1]))

    wipes, run = [], wipe_run.run

    def ran(*args, **kw):
        wipes.append(run(*args, **kw))
        return wipes[-1]

    def judged(exe, cases):
        assert exe == judge and list(cases) == [wipes[-1].case]
        return [wipes[-1].answer]

    wipe_run.run, wipe_run.serve = ran, judged
    with tempfile.TemporaryDirectory() as t:
        period = Path(t) / "periods" / "p0"
        period.mkdir(parents=True)
        (period / "period.json").write_text(json.dumps(doc, indent=1, sort_keys=True))
        (out / "period.json").write_text(json.dumps(doc, indent=1, sort_keys=True))
        (Path(t) / "periods" / "wipe-challenger").mkdir()
        with open(out / "summary.json", "w") as f, contextlib.redirect_stdout(f):
            return wipe_run.main(["--period", str(period), "--out", str(out), "--node", a.node], side_of=stand_in)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
