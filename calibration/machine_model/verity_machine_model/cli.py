"""`machine-model`: re-solve a named price vector from the data and check every attack against it.

Each command is a module of `verity_machine_model.commands`, whose docstring is its usage; `machine-model` with no command
prints them all. This module only dispatches.

A command that reads one device's model is given the device, and refused without one: `--device sm_NN` (before the
command, or on `eps`, `redteam` and `margin`) reads that device's directory (DESIGN.md "Devices"), and `--card SKU`
(before the command) the device whose model is that card's (its `model.toml`'s `card`), refusing a card no device's
model is. `--data DIR` (before the command) reads another data directory, such as a copy of a device's with a candidate
row. `cover`, `headline` and `lean-headline` read every device they cover and take none. `solve` exits 1 when the vector fails its check or doesn't
reproduce a published number, and `solve` and `attacks` exit 1 on a device no price vector is solved on.
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

from verity_machine_model.commands import commands
from verity_machine_model.commands._shared import device_help
from verity_machine_model.model import device, device_cards, device_of


def usage() -> str:
    head, _, tail = __doc__.partition("\n\n")
    return "\n\n".join([head, "\n".join(textwrap.indent(m.__doc__.strip("\n"), "    ") for m in commands().values()),
                        tail.split("\n\n", 1)[1]])


def main(argv: list[str] | None = None) -> int:
    cmds = commands()
    p = argparse.ArgumentParser(prog="machine-model", description=__doc__.split("\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, help="a data directory, such as a copy of a device's with a candidate row")
    p.add_argument("--device", help=device_help())
    # its own dest: `cover --card sm_NN` names a device
    p.add_argument("--card", dest="sku", help="the device whose model is this card's, by its SKU "
                                              "(calibration/hardware/cards/)")
    sub = p.add_subparsers(dest="cmd")
    for m in cmds.values():
        m.add(sub)
    args = p.parse_args(argv)
    if sum(x is not None for x in (args.device, args.sku, args.data)) > 1:
        raise SystemExit("one of --device, --card or --data")
    if args.device is not None:
        args.data = device(args.device)
    if args.sku is not None:
        args.data = device_of(args.sku)
    if args.cmd is None:
        print(usage())
        return 0
    if args.data is None and getattr(cmds[args.cmd], "READS_DEVICE", True):
        held = "; ".join(f"{d}, the {sku}" for d, sku in device_cards().items())
        raise SystemExit(f"machine-model {args.cmd} reads one device's model and was given none: name it with "
                         f"--device sm_NN or --card SKU ({held})")
    return cmds[args.cmd].run(args)


if __name__ == "__main__":
    sys.exit(main())
