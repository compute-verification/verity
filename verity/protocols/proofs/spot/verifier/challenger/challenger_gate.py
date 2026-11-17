"""Infra's challenger gate, the SSH host that reserves, opens and publishes the auditor's challenger host's sessions, as
its clients call it: ``ssh GATE VERB ...`` or a gate command on this machine (`gate_call`) with the lane's gate key (`ssh_args`); where a session it
publishes (``fetch SESSION``) keeps the challenger's out/ (`published`); and the challenger's key from its deployment
record (`key_record`). The gate is a transport: what it answers its callers check (`challenger_session.GateClient`,
`remote.trusted_session`)."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from verity.protocols.proofs.spot.verifier.challenger import challenger as Ch

SSH = ("ssh", "-o", "BatchMode=yes")


def ssh_args(key: str | None = None) -> list[str]:
    return [*SSH, *(["-o", "IdentitiesOnly=yes", "-i", key] if key else [])]


#: what a refusal shows of a gate answer it can't read: no run of hex digits long enough to be a session's secret or a root
HEX, HEX_WITHHELD = re.compile(r"[0-9a-fA-F]{32,}"), "(hex withheld)"


def command_of(challenger: str) -> list[str] | None:
    """The gate command a ``--challenger command:WORDS`` names (WORDS shell-split), or None for any other challenger."""
    if not challenger.startswith("command:"):
        return None
    return shlex.split(challenger.removeprefix("command:")) or None


def gate_key(doc: Any) -> Ch.ChallengerKey:
    """The key a gate's ``pubkey`` answers (`W`, `H`, `public`, `public_sha512`), refused unless `public` hashes to
    `public_sha512`: the gate's own word for its key, which only a gate whose coins are simulated is taken on."""
    key = Ch.ChallengerKey.from_json({k: doc.get(k) for k in ("W", "H", "public")} if isinstance(doc, dict) else None)
    if hashlib.sha512(key.public).hexdigest() != doc.get("public_sha512"):
        raise Ch.CoinError("the gate's pubkey answer names a public key whose SHA-512 is not the one it gives")
    return key


def gate_argv(gate: str | Sequence[str], args: Sequence[str], ssh: Sequence[str] = SSH) -> list[str]:
    """The command asking the gate `args`: a ``USER@HOST`` gate over `ssh`, or a gate that is a command (its words, as
    Pearl+'s gate is) run on this machine with `args` as its words, as one quoted line when that command is itself ssh,
    whose remote side reads it with shlex."""
    if isinstance(gate, str):
        return [*ssh, gate, shlex.join(args)]
    return [*gate, shlex.join(args)] if Path(gate[0]).name == "ssh" else [*gate, *args]


def gate_call(gate: str | Sequence[str], args: list[str], ssh: Sequence[str] = SSH,
              run: Callable[..., Any] = subprocess.run) -> dict:
    """The gate's answer to `args` (`gate_argv`); a refusal raises. It
    quotes the gate's reason, else ssh's stderr, else the answer, whichever it is without its long hex runs (`HEX`): an
    open's answer holds its session's secret, which a gate failing mid-open can put in any of the three. The filter
    knows only hex: a secret in another encoding is quoted as it stands."""
    p = run(gate_argv(gate, args, ssh), capture_output=True, text=True)

    def refused(why: object = None) -> Ch.CoinError:
        quoted = str(why or p.stderr.strip() or p.stdout.strip() or f"exit {p.returncode}")
        return Ch.CoinError(f"infra's gate refused {args[0]}: {HEX.sub(HEX_WITHHELD, quoted)}")

    try:
        got = json.loads(p.stdout)
    except ValueError:
        raise refused() from None
    if p.returncode or not isinstance(got, dict) or "refused" in got:
        raise refused(got.get("refused") if isinstance(got, dict) else None)
    return got


def published(directory: str | os.PathLike) -> Path:
    """The challenger's out/ of a session as infra's gate publishes it (``fetch SESSION``): `directory`'s out/, beside
    its reserve's request.json and its open's open.json, or `directory` when it is that out/."""
    d = Path(directory)
    return d / "out" if (d / "out").is_dir() else d


def key_record(path: str | os.PathLike) -> Ch.ChallengerKey:
    """The challenger's key from its deployment record at `path` (infra's challenger-key/v1, read by core's
    `challenger.key_of_record`), the key a verifier holds a session to, never the session's own key.json nor the
    ledger's key_sha512."""
    return Ch.key_of_record(json.loads(Path(path).read_text()))
