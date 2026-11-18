"""A stand-in for `streamed_wipe/demo.sh` in `test_pous_served_wipe`, taking its options, attaching the wipe as it does
(`Exchange.attach`, the case to `$WIPE_PRIVATE/wipe`) and writing its outputs under `$OUT` (period.json, wipe.json,
summary.json) with its exit: one 32 MiB segment kept in host memory
(`wipe_store.HostMemory`, period.json's `wipe.store` `host`) in place of the card, against `wipe_side.py`, the challenger's
stand-in on a host of its own, in place of the side `--challenger-command` starts on `--challenger`, its key trusted
only when it is the one `--trusted-sha512` pins (the attach's ``trusted``), as demo.sh has it. The
challenger period.json names is `Side.of`'s, written to `$OUT/party.json` before the stand-in replaces it;
the arguments and the lease it ran under go to `$OUT/demo-argv.json`. The `--wipe-protocol` the wrapper always gives is
the period's Lean judge, stood in by one that answers each case as the Python answered it.
`python served_wipe_demo.py OUT LEASE PRIVATE PINS CONFIGURATION [DEMO_OPTIONS]`: the test's shim passes `$OUT`,
`$GPU_LEASE_UUID`, `$WIPE_PRIVATE`, `$VERITY_PINS` and `$VERITY_CONFIGURATION`."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import sys
from pathlib import Path

from verity import hooks as H
from verity.protocols.compose import schemes
from verity.protocols.resource_accounting.memory.pous.prover import wipe_run
from verity.runtime.stream import Side
from verity_vllm.messages import Exchange

SIDE = Path(__file__).resolve().parents[2] / "verity" / "protocols" / "resource_accounting" / "memory" / "pous" / "verifier" / "wipe_side.py"


def main(argv: list[str]) -> int:
    out, lease, private, pins, configuration, *argv = argv
    ap = argparse.ArgumentParser()
    for o in ("--node", "--challenger", "--challenger-address", "--challenger-command", "--trusted-sha512", "--reads",
              "--wipe-protocol"):
        ap.add_argument(o)
    a = ap.parse_args(argv)
    out = Path(out)
    (out / "demo-argv.json").write_text(json.dumps({"argv": argv, "lease": lease or None}))
    judge = a.wipe_protocol
    settings = {"store": "host", "streamer": "python", "reads": int(a.reads), "host": a.challenger,
                "address": a.challenger_address, "command": shlex.split(a.challenger_command)}

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
    private = Path(private)
    got = Exchange().attach(schemes.get("wipe"), settings, private=private, process=H.ENGINE, side_of=stand_in,
                            pins=json.loads(pins), configuration=configuration or None, trusted=a.trusted_sha512,
                            node=a.node, judge=judge)
    shutil.copy(private / "wipe" / "period.json", out / "period.json")
    (out / "wipe.json").write_text(json.dumps(got.record(), sort_keys=True, indent=1))
    (out / "summary.json").write_text(json.dumps(got.summary(), indent=1, sort_keys=True))
    return 0 if got.answer["verdict"] == "accept" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
