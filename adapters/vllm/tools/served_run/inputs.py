"""The launchers' command line of the adapter's input datasets (`verity_vllm.inputs`): its statement, its door and its one
sender.

    inputs.py statement                  # MODEL REVISION PRECISION CLASS, one line
    inputs.py door NAME [TIMING]         # the door over the input dataset NAME at TIMING (default at-once): its identity
    inputs.py application FILE           # MODEL REVISION PRECISION CLASS DATASET TIMING of applications/NAME.toml, then the door's line
    inputs.py send --port P --dataset NAME --timing T [--release RUN] --out FILE    # the one sender

A refusal prints ``REFUSED NAME: TEXT`` (``[send] REFUSED`` from the sender) and exits 3, before anything is sent.
"""
from __future__ import annotations

import argparse

import input_datasets
from verity_vllm.inputs import sender, serves


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("statement")
    door = sub.add_parser("door")
    door.add_argument("dataset")
    door.add_argument("timing", nargs="?", default="at-once")
    sub.add_parser("application").add_argument("file")
    send = sub.add_parser("send")
    send.add_argument("--port", type=int, required=True)
    send.add_argument("--dataset", required=True, help="the input dataset's name (input_datasets.open)")
    send.add_argument("--timing", required=True, help="at-once, or rate:R requests a second")
    send.add_argument("--release", default=None, help="number the requests in order under this release run id")
    send.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    s = serves.SERVES
    if a.command == "statement":
        print(s.model, s.revision, s.precision, s.window_class)
        return 0
    if a.command == "send":
        return sender.send(a.port, a.dataset, a.timing, release=a.release, out=a.out)
    try:
        if a.command == "application":
            name, timing = serves.application(a.file)
            door_said = serves.door_line(name, timing)
            said = [" ".join((s.model, s.revision, s.precision, s.window_class, name, timing)), door_said]
        else:
            said = [serves.door_line(a.dataset, a.timing)]
    except (serves.Refused, input_datasets.Refused) as e:
        print("REFUSED {}: {}".format(*serves.refusal(e)))
        return 3
    print("\n".join(said))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
