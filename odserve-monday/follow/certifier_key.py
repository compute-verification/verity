"""certifier_key.py: the auditor's key record of the network certifier's run, read from the certifier's host by the
follow-on (the auditor's side), never through the prover's node.

    python3 certifier_key.py D OUT_JSON --gate USER@HOST --key SSH_KEY --pin COMMIT --peer NAME

Each warden rehearsal makes a fresh key and session on the certifier's host (vy-auditor-2), which writes them to its
`certifier/RUN/out/deployment.json` and the SHA-512 of every `out/` file to `end.json` beside it, unsigned: they hold only
as read from that host, under trusted-host and its pinned code. RUN is the one run whose fetch the serve dir D kept
(D/private/certifier/RUN/fetch.tar.gz, the prover's copy). This reads `status RUN` and `fetch RUN`'s
`RUN/out/deployment.json` over ssh (BatchMode: an unknown host key refuses), and writes OUT_JSON (`certifier-key/v1`:
the key and session, the whole deployment and its SHA-512, the run's request and end) only when the deployment's SHA-512
is the end's, the run exited 0 at the commit `--pin` names and its peer is `--peer`; else it exits 3 naming why. It then
says whether the prover's copy carries the same deployment.json (exit 4 when it doesn't, the record still written).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path

DEPLOYMENT = "deployment.json"


def refuse(why: str, code: int = 3) -> None:
    print(f"CERTIFIER-KEY: not read: {why}")
    sys.exit(code)


def member(tar: tarfile.TarFile, name: str) -> bytes | None:
    for m in tar:
        if m.name == name and m.isfile():
            f = tar.extractfile(m)
            return f.read() if f else None
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("d", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--gate", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--pin", required=True)
    ap.add_argument("--peer", required=True)
    a = ap.parse_args()
    kept = sorted(a.d.glob("private/certifier/*/fetch.tar.gz"))
    if len(kept) != 1:
        refuse(f"the serve dir keeps {len(kept)} network certifier runs, not one")
    run = kept[0].parent.name
    ssh = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", "-o", "ConnectTimeout=30", "-i", a.key, a.gate]
    st = subprocess.run([*ssh, "status", run], stdin=subprocess.DEVNULL, capture_output=True)
    try:
        status = json.loads(st.stdout)
    except ValueError:
        status = None
    if st.returncode != 0 or not isinstance(status, dict) or "refused" in status:
        why = status.get("refused") if isinstance(status, dict) else None
        refuse(f"{a.gate} status {run} exited {st.returncode}: "
               f"{why or st.stderr.decode(errors='replace').strip()[-300:]}")
    req, end = status.get("request") or {}, status.get("end")
    if not isinstance(end, dict):
        refuse(f"{a.gate} has no end for {run}")
    pinned = str(end.get("pinned") or "")
    want = (end.get("files") or {}).get(DEPLOYMENT)
    problems = [p for p in (
        end.get("rc") != 0 and f"it exited {end.get('rc')}",
        not (len(pinned) >= 11 and (pinned.startswith(a.pin) or a.pin.startswith(pinned)))
        and f"it ran at {pinned or '?'}, not {a.pin}",
        req.get("peer") != a.peer and f"its peer is {req.get('peer')}, not {a.peer}",
        req.get("device") != "warden" and f"its device is {req.get('device')}, not warden",
        not want and f"its end names no {DEPLOYMENT}") if p]
    if problems:
        refuse(f"{a.gate}'s run {run}: " + "; ".join(problems))
    with subprocess.Popen([*ssh, "fetch", run], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE) as p:
        try:
            with tarfile.open(fileobj=p.stdout, mode="r|gz") as tar:
                dep = member(tar, f"{run}/out/{DEPLOYMENT}")
        except (tarfile.TarError, OSError) as e:
            dep = None
            print(f"CERTIFIER-KEY: {a.gate} fetch {run}: {type(e).__name__}: {e}")
        while p.stdout.read(1 << 20):
            pass
        err = p.stderr.read()
    if p.returncode != 0 or dep is None:
        refuse(f"{a.gate} fetch {run} exited {p.returncode} with no {DEPLOYMENT}: "
               f"{err.decode(errors='replace').strip()[-300:]}")
    got = hashlib.sha512(dep).hexdigest()
    if got != want:
        refuse(f"{a.gate}'s {run}/out/{DEPLOYMENT} has SHA-512 {got[:16]}…, not its end's {want[:16]}…")
    deployment = json.loads(dep)
    record = {"schema": "certifier-key/v1", "host": a.gate.split("@")[-1], "run": run, "pinned": pinned,
              "peer": req.get("peer"), "key": deployment.get("key"), "session": deployment.get("session"),
              "deployment_sha512": got, "deployment": deployment, "request": req, "end": end}
    a.out.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    with tarfile.open(kept[0], "r:gz") as tar:
        theirs = member(tar, f"{run}/out/{DEPLOYMENT}")
    same = theirs is not None and hashlib.sha512(theirs).hexdigest() == got
    print(f"CERTIFIER-KEY: {a.out}, the network certifier's key and session for run {run} as {a.gate} recorded them "
          f"(deployment.json SHA-512 {got[:16]}…, at {pinned[:11]}, peer {req.get('peer')}); the serve dir's kept fetch "
          + ("carries the same deployment.json" if same else
             "carries " + ("no deployment.json" if theirs is None else "a different deployment.json")
             + ": the network verdict is not established"))
    sys.exit(0 if same else 4)


if __name__ == "__main__":
    main()
