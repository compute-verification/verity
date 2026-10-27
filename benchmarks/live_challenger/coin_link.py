"""The network certifier's coin link for one window's session, on the challenger's host
(`verity.core.protocols.certifier.coin_link`): the one path the session's coins take to its peer, counted and re-clocked
on the link's grids.

    python3 -I benchmarks/live_challenger/coin_link.py --out DIR --port PORT [--peer NAME [--peers FILE]] [--minutes M]
        [--cell-bytes B] [--coin-cells R] [--request-cells R] [--bucket-ms MS] [--window T] [--slack-ms MS]

`auditor.py run` starts it once `serve.py` listens on this host's 127.0.0.1:PORT, after the window's `open` entry and the
challenger's signed record (`--out`, the session's `out/`), and stops it with SIGTERM once serve.py exits. It reads the
session's coin commitment from that record, listens on a port of this host's loopback that the system picks, LISTEN, and
only then opens `ssh -N -R 127.0.0.1:PORT:127.0.0.1:LISTEN` to `--peer`'s line of `--peers` (/etc/vy/peers), so the
prover's client on the peer connects to its own 127.0.0.1:PORT; without `--peer` (a test) no tunnel opens and the client
connects to LISTEN. It prints `listening 127.0.0.1:LISTEN` and relays each connection to serve.py's port, one at a time,
until SIGTERM, the tunnel's end or `--minutes`; then it runs to the end of its link-window and writes `coin-link.json` to
`--out` (`CoinLink.summary`, with PORT, LISTEN, the peer and how the tunnel ended), which the window's `end` entry seals.
The defaults are D7's parameters (`CoinLinkParams`).

The tunnel goes straight to the peer: `ProxyJump` and `ProxyCommand` are `none`, and a `--peers` line that names a jump
or a proxy is refused, so the coins leave this host encrypted to the peer's own sshd and are read on no other host
(WitnessPathPrivate).

Exit 0 when every link-window is complete and no cell was dropped, 1 otherwise, 2 when it refuses to start.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import signal
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

from serve import peer_ssh  # noqa: E402
from verity.core.protocols.certifier.coin_link import CoinLink, CoinLinkParams  # noqa: E402
from verity.core.protocols.sampled_proofs import remote  # noqa: E402
from verity.core.protocols.sampled_proofs.challenger import ChallengerRecord, CoinError, LiveCommitment  # noqa: E402

SUMMARY = "coin-link.json"
TUNNEL = ["ssh", "-N", "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=15",
          "-o", "ProxyJump=none", "-o", "ProxyCommand=none"]


def parse(argv: list[str] | None) -> argparse.Namespace:
    d = CoinLinkParams()
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--peer")
    p.add_argument("--peers", type=Path, default=Path("/etc/vy/peers"))
    p.add_argument("--minutes", type=int)
    p.add_argument("--cell-bytes", type=int, default=d.cell_bytes)
    p.add_argument("--coin-cells", type=int, default=d.coin_cells)
    p.add_argument("--request-cells", type=int, default=d.request_cells)
    p.add_argument("--bucket-ms", type=int, default=d.bucket_ms)
    p.add_argument("--window", type=int, default=d.window)
    p.add_argument("--slack-ms", type=int, default=25)
    return p.parse_args(argv)


def jump(words: list[str]) -> str | None:
    """The option of `words` (an ssh command line's) that routes through another host, None when none does."""
    for i, w in enumerate(words):
        if w.startswith(("-J", "-W")):
            return w
        option = w[2:] if w.startswith("-o") and w != "-o" else words[i + 1] if w == "-o" and i + 1 < len(words) else ""
        if re.split(r"[=\s]", option.strip(), maxsplit=1)[0].lower() in ("proxyjump", "proxycommand"):
            return option
    return None


def commitment(out: Path) -> LiveCommitment:
    """The session's coin commitment, as the challenger's signed record in `out` states it."""
    try:
        return ChallengerRecord.from_json(json.loads((out / remote.RECORD).read_text())).commitment
    except FileNotFoundError:
        raise CoinError(f"no challenger record at {out / remote.RECORD}: serve.py writes it before it listens") from None
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise CoinError(f"the challenger record at {out / remote.RECORD} is unreadable: {e}") from e


async def serve(args: argparse.Namespace, params: CoinLinkParams, coins: LiveCommitment,
                ssh: list[str] | None) -> dict:
    loop = asyncio.get_running_loop()
    link = CoinLink(params, coins, lambda: asyncio.open_connection("127.0.0.1", args.port), origin=time.monotonic(),
                    slack=args.slack_ms / 1000,
                    buckets=None if args.minutes is None else args.minutes * 60_000 // params.bucket_ms)
    server = await asyncio.start_server(link.handle, "127.0.0.1", 0)
    listen = server.sockets[0].getsockname()[1]
    tunnel, ended_first = None, False
    if ssh is not None:
        tunnel = await asyncio.create_subprocess_exec(*TUNNEL, "-R", f"127.0.0.1:{args.port}:127.0.0.1:{listen}", *ssh,
                                                      stdin=asyncio.subprocess.DEVNULL,
                                                      stdout=asyncio.subprocess.DEVNULL)

        async def watch() -> None:
            nonlocal ended_first
            await tunnel.wait()
            ended_first = not link.stopping
            link.stop()

        watching = asyncio.create_task(watch())
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, link.stop)
    print(f"listening 127.0.0.1:{listen}", flush=True)
    try:
        await link.run()
    finally:
        server.close()
        if tunnel is not None:
            if tunnel.returncode is None:
                tunnel.terminate()
            await watching
    return {**link.summary(), "port": args.port, "listen": listen, "peer": args.peer,
            "tunnel": None if tunnel is None else {"rc": tunnel.returncode, "ended_first": ended_first}}


def main(argv: list[str] | None = None) -> int:
    args = parse(argv)
    try:
        params = CoinLinkParams(args.cell_bytes, args.coin_cells, args.request_cells, args.bucket_ms, args.window)
        coins = commitment(args.out)
        ssh = None
        if args.peer is not None:
            ssh = peer_ssh(args.peers, args.peer)
            if (option := jump(ssh)) is not None:
                raise ValueError(f"{args.peers}'s line for {args.peer} routes through another host ({option}): the coins "
                                 f"go straight to the peer's own sshd (WitnessPathPrivate), so name the peer itself")
    except (CoinError, ValueError, OSError) as e:
        print(f"the coin link refused to start: {e}", file=sys.stderr)
        return 2
    summary = asyncio.run(serve(args, params, coins, ssh))
    (args.out / SUMMARY).write_text(json.dumps(summary, sort_keys=True, indent=1) + "\n")
    print("ended:", "complete" if summary["complete"] else "not complete", file=sys.stderr)
    return 0 if summary["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
