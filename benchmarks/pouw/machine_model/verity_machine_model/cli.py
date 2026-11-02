"""`machine-model`: re-solve a named price vector from the data and check every attack against it.

Each command is a module of `verity_machine_model.commands`, whose docstring is its usage; `machine-model` with no command
prints them all. This module only dispatches.

`--device sm_NN` (before the command, or on `eps`, `redteam` and `margin`) reads that device's directory,
`data/devices/sm_NN/` (README "Devices"); sm_120, `data/` itself, is the default. `--data DIR` (before the command) reads
another data directory, such as a copy with a candidate row. `solve` exits 1 when the vector fails its check or doesn't
reproduce a published number, and `solve` and `attacks` exit 1 on a device no price vector is solved on.
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

from verity_machine_model.commands import commands
from verity_machine_model.commands._shared import device_help
from verity_machine_model.model import DATA, device


def usage() -> str:
    head, _, tail = __doc__.partition("\n\n")
    return "\n\n".join([head, "\n".join(textwrap.indent(m.__doc__.strip("\n"), "    ") for m in commands().values()),
                        tail.split("\n\n", 1)[1]])


def main(argv: list[str] | None = None) -> int:
    cmds = commands()
    p = argparse.ArgumentParser(prog="machine-model", description=__doc__.split("\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=DATA, help="the data directory (default: %(default)s)")
    p.add_argument("--device", help=device_help())
    sub = p.add_subparsers(dest="cmd")
    for m in cmds.values():
        m.add(sub)
    args = p.parse_args(argv)
    if args.device is not None:
        if args.data != DATA:
            raise SystemExit("--device or --data, not both")
        args.data = device(args.device)
    if args.cmd is None:
        print(usage())
        return 0
    return cmds[args.cmd].run(args)


if __name__ == "__main__":
    sys.exit(main())
