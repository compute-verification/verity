"""The development challenger on this machine (`challenger_session`'s ``local`` session): the auditor device's own
service, `verity/protocols/compose/verifier/live_challenger/auditor.py`'s reserve, admit and run, then `serve.py`, with no
tunnel, run in this process. It is compose's service, which no SPOT file the deployed challenger runs loads: the trusted
`challenger_session.start` and `challenger_party.Host` take `Local` from their caller (``local``), and a development
command passes it from here.
"""
from __future__ import annotations

import importlib.util
import json
import secrets
import shutil
import sys
import threading
import time
from datetime import datetime, UTC
from pathlib import Path

from verity.commitments.crypto import lms
from verity.protocols.proofs.spot.verifier.challenger import challenger as Ch, challenger_session as AC

TREE = Path(__file__).resolve().parents[6]


def _module(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = sys.modules[name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Local:
    """A development session on this machine (module docstring), its auditor device's root in scratch `root`, pinned on the
    verifier's copy of main's first-parent line `main`, its key at `params` (W, H), made in `workers` processes. A
    `root` that holds a host already is reopened: its key, with the leaves it has signed at, its ledger and its pins
    log are kept, so a session opened on it is the same challenger's as the ones before it, under the same key (`key`)."""

    def __init__(self, root: Path, main: list[str], by: str = "circuits", params: tuple[int, int] = AC.LOCAL_PARAMS,
                 workers: int = 1) -> None:
        A = _module("a5_auditor", TREE / "verity" / "protocols" / "compose" / "verifier" / "live_challenger" / "auditor.py")
        self.A, self.by = A, by
        self.host = A.Host(root=root / "var", peers=root / "peers", name=AC.LOCAL_HOST, pins=root / "pins.jsonl")
        (root / "var" / "sessions").mkdir(parents=True, exist_ok=True)
        (root / "peers").write_text(f"{AC.LOCAL_HOST} research@127.0.0.1\n{AC.LOCAL_PEER} research@127.0.0.1\n")
        (root / "coin-ports").write_text(f"{AC.LOCAL_PEER} {A.PORTS.start}-{A.PORTS.stop - 1}\n")
        self.pin = AC.on_main(main)
        pins = AC.lines(self.host.pins) if self.host.pins.exists() else []
        if not pins or json.loads(pins[-1]).get("commit") != self.pin:  # the pins log only grows, as the host's does
            with open(self.host.pins, "a", encoding="utf-8") as log:
                log.write(A.pin_line(self.pin, main[0] if main else self.pin, A.utc(datetime.now(UTC))) + "\n")
        self.params = lms.Params(64, *params)
        self.signer = A.key_signer(self.host, self.params, workers=workers)
        self.key = Ch.ChallengerKey(self.params.W, self.params.H, (self.host.key / "public").read_bytes())
        self.thread: threading.Thread | None = None
        self.session: str | None = None

    def _serve(self, session: str) -> None:
        A, host = self.A, self.host
        body = A.sessions_of(A.read_ledger(host.ledger))[session]["open"]["body"]
        argv = A.serve_argv(host, session, body)
        argv[argv.index("--port") + 1] = "0"
        argv += ["--params", f"{self.params.W},{self.params.H}", "--keygen-workers", "1"]
        A.run(host, session, signer=self.signer, pinned=self.pin, now=lambda: datetime.now(UTC), argv=argv)

    def open(self, root: str, layout: list, minutes: int = AC.MINUTES, scheme: str = AC.SCHEME,
             withhold: bool = False) -> tuple[str, str, str | None]:
        """Reserve and open the session on `root` with the coins `layout` of `scheme`, as the gate does, a withheld one
        when `withhold`: (address, secret, commitment), the commitment None on a withheld session (the gate holds its
        root until the session's end)."""
        A, host, now = self.A, self.host, datetime.now(UTC)
        hooks = dict(by=self.by, now=now, own=lambda p: None, token=secrets.token_hex(2),
                     reserve=lambda s: A.reserve(host, s, signer=self.signer, pinned=self.pin, now=now))
        rc, out = A.gate(["reserve", "--peer", AC.LOCAL_PEER, "--scheme", scheme, "--families", json.dumps(layout),
                          "--minutes", str(minutes), *(["--withhold"] if withhold else [])], host, **hooks)
        if rc:
            raise Ch.CoinError(f"the local gate refused the reserve: {out.decode().strip()}")
        self.session = json.loads(out)["session"]

        def start(s: str) -> None:
            self.thread = threading.Thread(target=self._serve, args=(s,), daemon=True)
            self.thread.start()

        def alive(_: str) -> bool:
            return self.thread is not None and self.thread.is_alive()
        rc, out = A.gate(["open", "--session", self.session, "--root", root], host, **hooks, secret=secrets.token_bytes(32),
                         admit=lambda s: A.admit(host, s, signer=self.signer, pinned=self.pin, now=now, active=alive),
                         start=start, active=alive, ready=lambda s: A.await_commitment(host, s, alive))
        got = json.loads(out)
        secret = got.pop("secret", None)  # an open the gate answered holds the hello's secret: no error quotes it
        if rc or AC.opened_reason(got, withhold) is not None:
            raise Ch.CoinError(f"the local gate refused the open, or {AC.opened_reason(got, withhold)}: {json.dumps(got)}")
        address = host.sessions / self.session / "out" / "address.json"
        # serve.py writes the file in place, its newline last: read it only once it is whole
        while not (address.exists() and address.read_text().endswith("\n")) and alive(self.session):
            time.sleep(0.1)
        return json.loads(address.read_text())["address"], secret, got["commitment"]

    def finish(self, dest: Path) -> tuple[Path, Path, Path, Ch.ChallengerKey]:
        """Once the session has ended (its end entry in the ledger): its out/, the ledger and the pins log copied to `dest`, and
        its key."""
        assert self.thread is not None and self.session is not None
        self.thread.join()
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copytree(self.host.sessions / self.session / "out", dest / "session", dirs_exist_ok=True)
        shutil.copyfile(self.host.ledger, dest / "windows.jsonl")
        shutil.copyfile(self.host.pins, dest / "pins.jsonl")
        return dest / "session", dest / "windows.jsonl", dest / "pins.jsonl", self.key


__all__ = ["Local"]
