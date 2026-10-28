"""The proof system's challenger on its auditor host (`verity.core.protocols.sampled_proofs.remote`): one epoch's coins for
one client session, its record signed with the host's LMS key and written before the first coin opens, and its release
log signed when the session ends.

    python3 benchmarks/live_challenger/serve.py --out DIR --port PORT --scheme SCHEME --families JSON --session ID
        [--epoch E] [--peer NAME [--peers FILE]] [--key DIR] [--params W,H] [--keygen-workers N] [--host NAME]
        [--root HEX] [--secret-file FILE] [--sign-lock FILE]

- **The key** is the challenger's LMS key over SHA-512, on its host in `--key DIR` (default
  `~/.local/state/live-challenger/key`), in the memory challenger's layout (`benchmarks/pous/memory_challenger/live.py`):
  `seed` (secret, 0600), `public` (`I || T[1]`), `tree.bin` (the key's tree, a cache checked against `public` and
  `tree.sha512`) and `state.json` (`lms.LeafState`, which refuses a used leaf across runs).  A missing key is made there
  first, at `lms.DEPLOYED` (about 1.1G SHA-512 hashes over `--keygen-workers`) or at `--params W,H` (tests).  A key inside
  `--out` is refused, since a run's `out/` is published.
- **The coins** are the operating system's bytes as the Lean verifier of record reads them (`one_stage.verifier.coins`:
  `IO.getRandomBytes`, A3), its `sampled-proofs-verifier` built with `lake` on first use; a host where the Lean can't
  answer starts no session, and neither does one whose source no claim instance names (`challenger.served_claim`: A3's
  is `uniform/io-getrandombytes`).
- **The session.**  `--session` is the id the auditor launched it under, which the auditor's signed ledger opens for
  its window; the key's leaf state records it, so the key signs no id twice.  It commits to the layout's coins (`--scheme`,
  `--epoch`, `--families` as `[[label, count], ...]`), signs and writes `record.json`, `key.json` and the head of
  `releases.jsonl` to `--out` (`remote.open_session`), makes `hellos.jsonl` (empty), and only then listens on
  127.0.0.1:PORT and prints `listening 127.0.0.1:PORT` (PORT 0: one the system picks).  A window's session (`--root`)
  opens no tunnel: PORT is one of its peer's certifier coin link, which carries it to the peer (`auditor.py`, the coins'
  path), and `--peer` with `--root` is refused (`CoinsConfidential`).  Only a development session (`run.sh`) takes
  `--peer`: it then opens `ssh -R PORT` to that machine of `--peers` (`/etc/vy/peers`, as
  `benchmarks/remote_session/run.sh` does for its verifier), whose own 127.0.0.1:PORT the client connects to.  It
  serves the first client whose hello it accepts (`remote.greet`, then `remote.serve`): a refused hello releases nothing, so it is logged in `hellos.jsonl` (`{"n",
  "reason"}`, fsynced) and the next connection is taken, each hello read within 10 s; once one is accepted it stops
  listening, so every later connection is refused.  However the session ends, a tunnel that ends first included, it
  signs its release log and how it ended and writes `challenger.json` (`remote.close_session`).  On the auditor host
  (`auditor.py`) the session has a secret, `--secret-file` (64 hex digits, given to the window's opener by the gate),
  which the client's hello must carry, and `--root`, the window's Commit root: a salt family's coin (label `salt` or
  `.../salt`) opens only after that root, and any other is refused, which ends the session.
- **Signing.**  The key's leaf state is opened only for each signature (the session id's start with the record's, then
  the close's), under `--sign-lock` when given (the auditor host's `sign.lock`, which every signer of the key takes), so
  the host's other sessions and the ledger sign between them.

Exit 0 when the client closed the session, 1 when it ended otherwise (a refusal, a broken stream, a tunnel that ended
first), 2 on arguments or a key it refuses.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import selectors
import shlex
import socket
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from verity.core.protocols.sampled_proofs import remote  # noqa: E402
from verity.core.protocols.sampled_proofs.challenger import (ChallengerKey, CoinError, LiveChallenger,  # noqa: E402
                                                             served_claim)
from verity.core.protocols.sampled_proofs.one_stage import verifier as V  # noqa: E402
from verity.ml.crypto import lms  # noqa: E402

KEY = Path.home() / ".local" / "state" / "live-challenger" / "key"
HELLOS = "hellos.jsonl"
HELLO_SECONDS = 10.0
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
    p.add_argument("--session", required=True)
    p.add_argument("--epoch", type=int, default=0)
    p.add_argument("--peer")
    p.add_argument("--peers", type=Path, default=Path("/etc/vy/peers"))
    p.add_argument("--key", type=Path, default=KEY)
    p.add_argument("--params", default=None, metavar="W,H")
    p.add_argument("--keygen-workers", type=int, default=len(os.sched_getaffinity(0)))
    p.add_argument("--host", default=socket.gethostname())
    p.add_argument("--root", default=None, help="the window's Commit root, 128 lowercase hex digits")
    p.add_argument("--secret-file", type=Path, default=None)
    p.add_argument("--sign-lock", type=Path, default=None)
    return p.parse_args(argv)


def is_salt(label: str) -> bool:
    return label == "salt" or label.endswith("/salt")


class WindowChallenger:
    """The challenger as one window's session serves it: a salt family's coin opens only after the window's Commit root."""

    def __init__(self, challenger: LiveChallenger, root: bytes) -> None:
        self._challenger, self._root = challenger, root

    def __getattr__(self, name: str):
        return getattr(self._challenger, name)

    def open(self, label: str, index: int, after: bytes):
        if is_salt(label) and bytes(after) != self._root:
            raise CoinError(f"a {label} coin opens only after the window's Commit root {self._root.hex()[:16]}..., not "
                            f"{bytes(after).hex()[:16]}...")
        return self._challenger.open(label, index, after)


def log_hello(out: Path, n: int, why: str) -> None:
    with open(out / HELLOS, "a", encoding="utf-8") as f:
        f.write(json.dumps({"n": n, "reason": why}, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_secret(path: Path | None) -> bytes | None:
    if path is None:
        return None
    text = path.read_text().strip()
    if not re.fullmatch(r"[0-9a-f]{64}", text):
        raise ValueError(f"{path}: a session's secret is 64 lowercase hex digits")
    return bytes.fromhex(text)


@contextmanager
def leaf_state(key: lms.PrivateKey, key_dir: Path, lock: Path | None) -> Iterator[lms.LeafState]:
    """The key's leaf state for one use, under `lock` (held exclusively, waiting for its holder) when given."""
    with open(lock, "a") if lock is not None else nullcontext() as held:
        if held is not None:
            fcntl.flock(held, fcntl.LOCK_EX)
        with lms.LeafState(key_dir / "state.json", lms.public_key(key), key.params.H) as state:
            yield state


def main(argv: list[str] | None = None) -> int:
    args = parse(argv)
    out, key_dir = args.out.resolve(), args.key.resolve()
    if key_dir == out or out in key_dir.parents:
        print(f"the challenger's key {key_dir} is inside its published --out {out}", file=sys.stderr)
        return 2
    if args.root is not None and not re.fullmatch(r"[0-9a-f]{128}", args.root):
        print("--root: the window's Commit root, 128 lowercase hex digits", file=sys.stderr)
        return 2
    if args.root is not None and args.peer is not None:
        print("the challenger refused to start: a window's session (--root) reaches its peer through the peer's "
              "certifier coin link, never its own ssh -R (--peer; CoinsConfidential)", file=sys.stderr)
        return 2
    try:
        secret = read_secret(args.secret_file)
        params = lms.DEPLOYED if args.params is None else lms.Params(64, *map(int, args.params.split(",")))
        served_claim(V.coins)
        challenger = LiveChallenger(args.scheme, args.epoch, args.families, V.coins)
        ssh = peer_ssh(args.peers, args.peer) if args.peer else None
        out.mkdir(parents=True, exist_ok=True)
        key, state = load_key(key_dir, params, args.keygen_workers)
        state.close()
    except (CoinError, V.LeanError, ValueError, TypeError, OSError, lms.StateError) as e:
        print(f"the challenger refused to start: {e}", file=sys.stderr)
        return 2

    def sign(message: bytes) -> bytes:
        with leaf_state(key, key_dir, args.sign_lock) as st:
            return lms.sign(key, st, message).encode()

    served = challenger if args.root is None else WindowChallenger(challenger, bytes.fromhex(args.root))
    try:
        with leaf_state(key, key_dir, args.sign_lock) as st:
            st.start(args.session.encode())
        session = remote.open_session(served, ChallengerKey(params.W, params.H, lms.public_key(key)), args.host, sign,
                                      out, args.session)
        with open(out / HELLOS, "x", encoding="utf-8") as f:
            os.fsync(f.fileno())
    except (CoinError, ValueError, OSError, lms.StateError) as e:
        print(f"the challenger refused to start: {e}", file=sys.stderr)
        return 2
    ended = run(args, out, ssh, session, secret)
    print("ended:", "closed by the client" if ended is None else ended, file=sys.stderr)
    return 0 if ended is None else 1


def run(args: argparse.Namespace, out: Path, ssh: list[str] | None, session: remote.Session,
        secret: bytes | None) -> str | None:
    """Listen, open the tunnel, serve the first client whose hello carries `secret` and close the session: None when its
    client closed it, else why it ended."""
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
    refused = 0
    try:
        while True:
            if any(k.fileobj is not listener for k, _ in sel.select()):
                ended = f"the tunnel to {args.peer} ended before a client's hello was accepted"
                break
            conn, _ = listener.accept()
            conn.settimeout(HELLO_SECONDS)
            why = remote.greet(conn, secret)
            if why is not None:
                conn.close()
                log_hello(out, refused, why)
                refused += 1
                continue
            conn.settimeout(None)
            sel.unregister(listener)
            listener.close()
            with conn:
                ended = remote.serve(conn, session, secret=secret, greeted=True)
            break
    finally:
        if tunnel is not None:
            tunnel.terminate()
            tunnel.wait()
    if session.close is not None:
        return ended
    try:
        remote.close_session(session, ended)
    except (CoinError, OSError, lms.StateError) as e:
        return f"{ended or 'the client closed the session'}, and the challenger's close went unwritten ({e})"
    return ended


if __name__ == "__main__":
    sys.exit(main())
