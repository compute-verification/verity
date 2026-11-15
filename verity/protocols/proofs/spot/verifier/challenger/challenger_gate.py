"""Infra's challenger gate, the SSH host that reserves, opens and publishes the auditor's challenger host's sessions, as
its clients call it: ``ssh GATE VERB ...`` (`gate_call`) with the lane's gate key (`ssh_args`); where a session it
publishes (``fetch SESSION``) keeps the challenger's out/ (`published`); and the challenger's key from its deployment
record (`key_record`). The gate is a transport: what it answers its callers check (`challenger_session.GateClient`,
`remote.trusted_session`)."""
from __future__ import annotations

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


def gate_call(gate: str, args: list[str], ssh: Sequence[str] = SSH, run: Callable[..., Any] = subprocess.run) -> dict:
    """The gate's answer to `args` (``ssh GATE VERB ...``, quoted for the gate's shell words); a refusal raises. It
    quotes the gate's reason, else ssh's stderr, else the answer, whichever it is without its long hex runs (`HEX`): an
    open's answer holds its session's secret, which a gate failing mid-open can put in any of the three. The filter
    knows only hex: a secret in another encoding is quoted as it stands."""
    p = run([*ssh, gate, shlex.join(args)], capture_output=True, text=True)

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
