"""The command line for Verity's shared benchmark harnesses."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from verity_bench.vllm import cli as vllm_cli
from verity_bench.vllm import result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="verity-bench")
    commands = parser.add_subparsers(dest="command", required=True, metavar="BACKEND")

    vllm = commands.add_parser("vllm", help="measure serving overhead through the vLLM integration")
    vllm_cli.add_arguments(vllm)
    vllm.set_defaults(run=vllm_cli.run, command_parser=vllm)

    judge = commands.add_parser("judge", help="compare two benchmark results whose identities agree")
    result.add_judge_arguments(judge)
    judge.set_defaults(run=result.run_judge, command_parser=judge)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return int(args.run(args))
    except ValueError as error:
        args.command_parser.error(str(error))
        raise AssertionError("argparse.error returned") from error


if __name__ == "__main__":
    sys.exit(main())
