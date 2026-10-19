"""The proof service's challenger on its auditor host (`verity.core.protocols.sampled_proofs.remote`): one epoch's coins for
one client session, its record signed with the host's LMS key and written before the first coin opens.

    python3 benchmarks/live_challenger/serve.py --out DIR --port PORT --scheme SCHEME --families JSON [--epoch E]
        [--peer NAME [--peers FILE]] [--key DIR] [--params W,H] [--keygen-workers N] [--host NAME]

- **The key** is the challenger's LMS key over SHA-512, on its host in `--key DIR` (default
  `~/.local/state/live-challenger/key`), in the memory challenger's layout (`benchmarks/pous/memory_challenger/live.py`):
  `seed` (secret, 0600), `public` (`I || T[1]`), `tree.bin` (the key's tree, a cache checked against `public` and
  `tree.sha512`) and `state.json` (`lms.LeafState`, which refuses a used leaf across runs).  A missing key is made there
  first, at `lms.DEPLOYED` (about 1.1G SHA-512 hashes over `--keygen-workers`) or at `--params W,H` (tests).  A key inside
  `--out` is refused, since a run's `out/` is published.
- **The coins** are the operating system's bytes as the Lean verifier of record reads them (`one_stage.verifier.coins`:
  `IO.getRandomBytes`, A3), its `sampled-proofs-verifier` built with `lake` on first use; a host where the Lean can't
  answer starts no session.
- **The session.**  It commits to the layout's coins (`--scheme`, `--epoch`, `--families` as `[[label, count], ...]`),
  signs and writes `record.json`, `key.json` and the head of `releases.jsonl` to `--out` (`remote.open_session`), and only
  then listens on 127.0.0.1:PORT and prints `listening 127.0.0.1:PORT` (PORT 0: one the system picks).  With `--peer` it
  then opens `ssh -R PORT` to that machine of `--peers` (`/etc/vy/peers`, as `benchmarks/remote_session/run.sh` does for
  its verifier), whose own 127.0.0.1:PORT the client connects to.  It serves the one client that connects
  (`remote.serve`), and writes `challenger.json`: the commitment, the host, the coins released by family and how the
  session ended.

Exit 0 when the client closed the session, 1 when it ended otherwise (a refusal, a broken stream, a tunnel that ended
first), 2 on arguments or a key it refuses.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import selectors
import shlex
import socket
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from verity.core.protocols.sampled_proofs import remote  # noqa: E402
from verity.core.protocols.sampled_proofs.challenger import ChallengerKey, CoinError, LiveChallenger  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import verifier as V  # noqa: E402
from verity.ml.crypto import lms  # noqa: E402

KEY = Path.home() / ".local" / "state" / "live-challenger" / "key"
SSH = ["ssh", "-tt", "-o", "ObscureKeystrokeTiming=no", "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes",
       "-o", "ServerAliveInterval=15"]


def load_key(directory: Path, params: lms.Params, workers: int) -> tuple[lms.PrivateKey, lms.LeafState]:
    """The LMS key in `directory` with its leaf state, made there first when absent."""
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    seed_p, public_p, tree_p, check_p, state_p = (directory / n for n in ("seed", "public", "tree.bin", "tree.sha512",
                                                                          "state.json"))
    if not seed_p.exists():
        fd = os.open(seed_p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(fd, os.urandom(params.N))
            os.fsync(fd)
        finally:
            os.close(fd)
    seed = seed_p.read_bytes()
    if tree_p.exists() and public_p.exists() and check_p.exists():
        public, raw = public_p.read_bytes(), tree_p.read_bytes()
        tree = (b"",) + tuple(raw[params.N * r:params.N * (r + 1)] for r in range(len(raw) // params.N))
        key = lms.PrivateKey(params, public[:16], seed, tree)
        if (len(tree) != 2 ** (params.H + 1) or lms.public_key(key) != public
                or hashlib.sha512(public + raw).digest() != check_p.read_bytes()):
            raise ValueError(f"{tree_p} is not the tree of {public_p} at {params}")
    else:
        key = lms.keygen(params, seed, workers)
        raw, public = b"".join(key.tree[1:]), lms.public_key(key)
        for path, data in ((tree_p, raw), (public_p, public), (check_p, hashlib.sha512(public + raw).digest())):
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_bytes(data)
            os.replace(tmp, path)
    if not state_p.exists():
        lms.LeafState.create(state_p, lms.public_key(key), params.H)
    return key, lms.LeafState(state_p, lms.public_key(key), params.H)


def peer_ssh(peers: Path, peer: str) -> list[str]:
    """ssh's options and destination for `peer`, its line of `peers`."""
    for line in peers.read_text().splitlines():
        name, _, rest = line.partition(" ")
        if name == peer and rest.strip():
            return shlex.split(rest)
    raise ValueError(f"no peer {peer} in {peers}")


def parse(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--scheme", required=True)
    p.add_argument("--families", type=json.loads, required=True)
    p.add_argument("--epoch", type=int, default=0)
    p.add_argument("--peer")
    p.add_argument("--peers", type=Path, default=Path("/etc/vy/peers"))
    p.add_argument("--key", type=Path, default=KEY)
    p.add_argument("--params", default=None, metavar="W,H")
    p.add_argument("--keygen-workers", type=int, default=len(os.sched_getaffinity(0)))
    p.add_argument("--host", default=socket.gethostname())
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse(argv)
    out, key_dir = args.out.resolve(), args.key.resolve()
    if key_dir == out or out in key_dir.parents:
        print(f"the challenger's key {key_dir} is inside its published --out {out}", file=sys.stderr)
        return 2
    try:
        params = lms.DEPLOYED if args.params is None else lms.Params(64, *map(int, args.params.split(",")))
        challenger = LiveChallenger(args.scheme, args.epoch, args.families, V.coins)
        ssh = peer_ssh(args.peers, args.peer) if args.peer else None
        out.mkdir(parents=True, exist_ok=True)
        key, state = load_key(key_dir, params, args.keygen_workers)
        try:
            remote.open_session(challenger, ChallengerKey(params.W, params.H, lms.public_key(key)), args.host,
                                lambda m: lms.sign(key, state, m).encode(), out)
        finally:
            state.close()
    except (CoinError, V.LeanError, ValueError, TypeError, OSError, lms.StateError) as e:
        print(f"the challenger refused to start: {e}", file=sys.stderr)
        return 2

    listener = socket.create_server(("127.0.0.1", args.port))
    address = "127.0.0.1:%d" % listener.getsockname()[1]
    print("listening", address, flush=True)
    sel = selectors.DefaultSelector()
    sel.register(listener, selectors.EVENT_READ)
    tunnel = None
    if ssh is not None:
        port = listener.getsockname()[1]
        with open(out / "tunnel.log", "w") as log:
            tunnel = subprocess.Popen([*SSH, "-R", f"127.0.0.1:{port}:127.0.0.1:{port}", *ssh, "sleep 86400"],
                                      stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        sel.register(os.pidfd_open(tunnel.pid), selectors.EVENT_READ)
    (out / "address.json").write_text(json.dumps({"challenger": args.host, "listen": address, "peer": args.peer,
                                                  "address": address}) + "\n")
    try:
        if any(k.fileobj is not listener for k, _ in sel.select()):
            ended = f"the tunnel to {args.peer} ended before a client connected"
        else:
            conn, _ = listener.accept()
            listener.close()
            with conn:
                ended = remote.serve(conn, challenger, out)
    finally:
        if tunnel is not None:
            tunnel.terminate()
            tunnel.wait()
    released = {label: len(r) for label, r in remote.released_roots(out / remote.RELEASES, challenger.commitment).items()}
    (out / "challenger.json").write_text(json.dumps({
        "commitment": challenger.commitment.to_json(), "host": args.host, "released": released,
        "ended": "closed by the client" if ended is None else ended}, indent=1) + "\n")
    print("ended:", "closed by the client" if ended is None else ended, file=sys.stderr)
    return 0 if ended is None else 1


if __name__ == "__main__":
    sys.exit(main())
