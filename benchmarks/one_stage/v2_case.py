"""Sampled proofs v2's case (`verity/sampled-proofs/one-stage/v2`) of a served window from the files its run kept, for
`composite.py --cases`, and with ``--exe`` its own Lean answer on it.

    v2_case.py --window DIR --out DIR --scratch DIR [--challenger-key RECORD] [--exe PATH]

``--window`` is the window directory `prove_path/run.sh` keeps (``out/sampled-proofs/served/window``): its v1 case
(``case.json``), the verifier's partition (``partition.json``), the drawn member's ``m<k>/`` with the public file and
the circuit file, its template's ``compose.log`` (its own, or its template's first member's, by ``summary.json``),
``session-m<k>/sessions/*/`` with each session's ``session.json`` and ``*.bin``, the drawn member the one whose sessions
the window kept, and the challenger's release
of the draw's coins (``release.json``, `a5.RELEASE`) where the audit drew on a challenger's session. The case is a5.py's
`case_v2` of those: v1's params with the partition, ``proofs: {circuit, public, registered: null}`` (``public`` the
drawn member's public file's SHA-512 as a5 pinned it, ``public-pin.json``, `a5.PIN`: as completed once the tie held, the
file its session proved, else as staged before the draw; never the kept file's read after the draw, which is refused
where it isn't the pinned one, as a window with no pin is), the
challenger and the release's window root, v1's commitment, the evidence with the sessions' proofs and the release's
coins, and the blobs' paths. The challenger's key
is the auditor's own record's (``--challenger-key``, challenger-key/v1, `Ch.key_of_record`), never the one the release
names (`drawn_of`): a release naming another key is refused, and leaves only ``missing.json``.
A window with no release, or no record given, names no challenger, which v2 refuses (`OneStageZk.NO_CHALLENGER`).

A run that didn't keep the circuit file gets it rebuilt in ``--scratch`` by the command that ``compose.log`` records
(`compose_circuit.py` over the verifier's synthetic input set, `run.sh`'s ``verity_numerical.bench.generate --n 4 --seed
1``): the circuit is public structure, and the file is used only if its SHA-512 is the log's pin, which the check reads
anyway. The case goes to ``--out`` private as `composite.py` reads it (0700, its files 0600): ``case.json``,
``params.json``, ``commitment.bin``, ``evidence.json`` and, with ``--exe`` (``sampled-proofs-one-stage-zk``),
``answer.json``. A blob with no file leaves only ``missing.json`` saying why, so `composite.py` leaves the protocol out.

Prints one ``V2CASE`` line (the pin, where the circuit file came from, which challenger it names or why none, the
sessions and blobs; or why it was refused) and, with ``--exe``, one ``V2`` line with the answer's verdict only: the whole
answer is witness-adjacent (its query is the partition's digest) and stays in ``--out``. Exits as the check does (0
accept, 1 reject, 2 undecided), 2 for a missing or refused case, 0 without ``--exe``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import a5
from composite import canonical, write_private

from verity.core.protocols.sampled_proofs import challenger as Ch
from verity.core.protocols.sampled_proofs.one_stage import proofs as PF

ID = "verity/sampled-proofs/one-stage/v2"
TREE = Path(__file__).resolve().parents[2]
CASE_FILES = ("case.json", "params.json", "commitment.bin", "evidence.json", "answer.json")


class Refused(ValueError):
    """The window's kept files make no case: why."""


def compose_command(log: str) -> dict:
    """What a member's ``compose.log`` records: the subcircuit, the pin, the input set's name and the partition and program
    digests `compose_circuit.py` was given."""
    cmd = log.splitlines()[0]
    arg = {k: (m.group(1) if (m := re.search(rf"--{k} ([0-9a-f]+)", cmd)) else None) for k in ("partition", "program")}
    return {"subcircuit": re.search(r'"subcircuit": "([^"]+)"', log).group(1),
            "pin": re.search(r'"pin": "([0-9a-f]{128})"', log).group(1),
            "set": re.search(r"--input-set \S*/sets/([^/\s]+)(\s|$)", cmd).group(1), **arg}


def rebuild(c: dict, scratch: Path) -> tuple[Path | None, str]:
    """The circuit file `compose_command` `c` names, rebuilt in `scratch`, and how; None unless its SHA-512 is the pin."""
    scratch.mkdir(parents=True, exist_ok=True)
    sets, f = scratch / "sets", scratch / "circuit.txt"
    py = sys.executable
    try:
        subprocess.run([py, "-m", "verity_numerical.bench.generate", "--subcircuit", c["subcircuit"], "--n", "4",
                        "--seed", "1", "--out", str(sets)], check=True, capture_output=True, cwd=TREE)
        argv = [py, "benchmarks/one_stage/compose_circuit.py", "--input-set", str(sets / "sets" / c["set"]), "--out", str(f)]
        argv += [x for k in ("partition", "program") if c[k] for x in (f"--{k}", c[k])]
        subprocess.run(argv, check=True, capture_output=True, cwd=TREE)
    except subprocess.CalledProcessError as x:
        return None, f"rebuild failed (exit {x.returncode}): {x.stderr.decode().strip()[-400:]}"
    got = hashlib.sha512(f.read_bytes()).hexdigest()
    if got != c["pin"]:
        return None, f"rebuilt from compose.log's command, SHA-512 {got[:16]}…, not the pin {c['pin'][:16]}…"
    return f, f"rebuilt from compose.log's command, its SHA-512 the pin ({f.stat().st_size} bytes)"


def drawn_of(window: Path, key: Ch.ChallengerKey | None) -> tuple[dict | None, str]:
    """The draw `a5.case_v2` names (`a5.drawn_by`) from `window`'s kept release (`a5.RELEASE`), and which challenger:
    the release's coins and window root under `key`, the auditor's own record's. The key the release names is the served
    node's to write, and with it the prover would name its own key and sign its own coins, so it is never read into the
    case, and a release naming another key than `key` is refused (`Refused`). With no release or no `key`, None."""
    release = window / a5.RELEASE
    if not release.is_file():
        return None, "none: the window kept no release"
    if key is None:
        return None, "none: no --challenger-key, the auditor's record of its challenger's key"
    try:
        r = json.loads(release.read_text())
        named, coins, root = r["params"]["challenger"]["key"], dict(r["evidence"]["coins"]), r["params"]["window_root"]
    except (KeyError, TypeError, ValueError):
        raise Refused(f"the kept {a5.RELEASE} is not a5.drawn_by's form") from None
    if named != key.to_json():
        raise Refused(f"the kept {a5.RELEASE} names another challenger key than the auditor's record's")
    return a5.drawn_by({**coins, "key": key.to_json()}, root), "the auditor's record's key"


def drawn_member(window: Path) -> int:
    """The drawn member's k: the one whose ``session-m<k>/sessions`` the window kept a session of. A window that kept
    none, or several members' (`a5.v2_proof`: v2 verifies one table's sessions), is refused (`Refused`)."""
    ks = sorted(int(d.name.removeprefix("session-m")) for d in window.glob("session-m*")
                if d.name.removeprefix("session-m").isdigit() and any((d / "sessions").glob("*/session.json")))
    if len(ks) != 1:
        raise Refused(f"the window kept sessions of {len(ks)} members, and v2 verifies one drawn member's")
    return ks[0]


def layout_of(window: Path) -> list[list]:
    """The verifier's layout of the window (`a5.layout`) from its ``summary.json`` ``members``, in unit order: each
    member's key and unit count, the key as a5 recorded it or, in a summary from before a5 recorded keys, as
    `interface._members` names a template's chunks (``<template>/c<j>``, ``j`` counting that template's members from 0).
    v2 holds the layout to the record's population and the drawn member's public file to its key's slice of the
    registration, so a wrong key refuses. Members that don't tile the window from unit 0 are refused (`Refused`)."""
    try:
        rows = sorted((m["base"], m["k"], m["n"], m["template"], m.get("key"))
                      for m in json.loads((window / "summary.json").read_text())["members"])
    except (OSError, KeyError, TypeError, ValueError):
        raise Refused("the window's summary.json names no members, the layout v2 finds the drawn member by") from None
    at, seen, out = 0, {}, []
    for base, k, n, template, key in rows:
        if base != at:
            raise Refused(f"the window's summary.json members don't tile it: m{k} starts at unit {base}, not {at}")
        j = seen[template] = seen.get(template, -1) + 1
        out.append([key or f"{template}/c{j}", n])
        at += n
    return out


def compose_log(window: Path, k: int) -> Path:
    """Member `k`'s ``compose.log``: its own, or, where a5 composed its template's circuit once at the template's first
    member (`a5.audit`), that member's, by the window's ``summary.json`` ``members``. None kept is refused."""
    log = window / f"m{k}" / "compose.log"
    if log.is_file():
        return log
    try:
        members = json.loads((window / "summary.json").read_text())["members"]
        template = next(m["template"] for m in members if m["k"] == k)
        log = window / f"m{min(m['k'] for m in members if m['template'] == template)}" / "compose.log"
    except (OSError, KeyError, StopIteration, TypeError, ValueError):
        raise Refused(f"m{k} kept no compose.log, and the window's summary.json names no member of its template") from None
    if not log.is_file():
        raise Refused(f"neither m{k} nor its template's first member kept a compose.log")
    return log


def pinned(window: Path, pub: Path, k: int = 0) -> str:
    """Member `k`'s public file's SHA-512 as a5 pinned it (`a5.PIN`): as completed once the tie held, the file its session
    proved (``completed``, `a5.pin_completed`), else as staged before the draw (`a5.public_pin`), which the kept file
    `pub` must be: a window with no pin naming m<k>, or whose kept public file isn't the pinned one, is refused
    (`Refused`)."""
    try:
        pin = json.loads((window / a5.PIN).read_text())
        m = next(m for m in pin["members"].values() if m["k"] == k)
        file = (m.get("completed") or m)["file"]
    except (OSError, AttributeError, KeyError, StopIteration, TypeError, ValueError):
        raise Refused(f"the window kept no {a5.PIN} naming m{k}, its public file's SHA-512 pinned at staging before "
                      "the draw") from None
    if PF.file_sha512(pub) != file:
        raise Refused(f"the kept public file isn't, by SHA-512, the one {a5.PIN} pinned for m{k}, at staging before the "
                      "draw or completed once the tie held")
    return file


def case(window: Path, scratch: Path, key: Ch.ChallengerKey | None = None) -> tuple[dict, str, int, str]:
    """`case_v2` of `window`'s kept files under the auditor's challenger key `key` (`drawn_of`), of its drawn member
    (`drawn_member`), its public file the pinned one (`pinned`): the case, where its circuit file came from, its
    sessions' count and which challenger it names."""
    drawn, challenger = drawn_of(window, key)
    k = drawn_member(window)
    log = compose_log(window, k)
    c = compose_command(log.read_text())
    v1 = json.loads((window / "case.json").read_text())
    v1["params"]["partition"] = json.loads((window / "partition.json").read_text())
    names = json.loads(bytes.fromhex(v1["evidence"]))["sessions"]
    sessions = tuple(sorted(f.parent for f in (window / f"session-m{k}" / "sessions").glob("*/session.json")))
    pub = next(iter(sorted((window / f"m{k}").glob("pub-*.bin"))), None)
    if pub is None:
        raise Refused(f"the drawn member m{k} kept no public file")
    public = pinned(window, pub, k)
    kept = [f for f in (window / f"m{k}" / "circuit.txt", log.parent / "circuit.txt") if f.is_file()]
    circuit, how = (kept[0], "kept") if kept else rebuild(c, scratch)
    circuit = circuit or window / f"m{k}" / "circuit.txt"
    v2 = a5.case_v2(v1, names, proof=PF.Proof(circuit, pub, sessions), circuit=c["pin"], public=public,
                    registered=None, tables=[], drawn=drawn, members=layout_of(window))
    return v2, how, len(sessions), challenger


def missing(out: Path, why: str) -> None:
    """Only ``missing.json`` in `out`, saying `why`, so `composite.py` leaves the protocol out."""
    for f in (*CASE_FILES, "missing.json"):
        (out / f).unlink(missing_ok=True)
    write_private(out, {"missing.json": (json.dumps({"protocol": ID, "why": why}) + "\n").encode()})


def write(out: Path, v2: dict, how: str) -> list[str]:
    """The case in `out`, or only ``missing.json`` when a blob has no file; the blobs with none."""
    gone = [b for b in v2["blobs"] if not Path(b).is_file()]
    if gone:
        missing(out, f"no file for {', '.join(gone)} (circuit: {how})")
        return gone
    for f in (*CASE_FILES, "missing.json"):
        (out / f).unlink(missing_ok=True)
    write_private(out, {"case.json": json.dumps(v2, sort_keys=True).encode(), "params.json": canonical(v2["params"]),
                        "commitment.bin": bytes.fromhex(v2["commitment"]),
                        "evidence.json": bytes.fromhex(v2["evidence"])})
    return []


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--window", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scratch", type=Path, required=True, help="where a circuit file the run didn't keep is rebuilt")
    ap.add_argument("--challenger-key", type=Path, metavar="RECORD",
                    help="the auditor's record of its challenger's key (challenger-key/v1), the case's only key")
    ap.add_argument("--exe", help="sampled-proofs-one-stage-zk, for the case's own Lean answer")
    a = ap.parse_args(argv)
    try:
        key = Ch.key_of_record(json.loads(a.challenger_key.read_text())) if a.challenger_key else None
    except (OSError, ValueError) as e:
        ap.error(f"--challenger-key: {e}")
    try:
        v2, how, n, challenger = case(a.window, a.scratch, key)
    except Refused as e:
        missing(a.out, f"refused: {e}")
        print("V2CASE " + json.dumps({"refused": str(e)}), flush=True)
        return 2
    gone = write(a.out, v2, how)
    print("V2CASE " + json.dumps({"circuit": v2["params"]["proofs"]["circuit"], "circuit_file": how,
                                  "challenger": challenger, "sessions": n, "blobs": len(v2["blobs"]), "missing": gone}),
          flush=True)
    if gone:
        return 2
    if not a.exe:
        return 0
    p = subprocess.run([a.exe, "check", *(str(a.out / f) for f in ("params.json", "commitment.bin", "evidence.json")),
                        *v2["blobs"]], capture_output=True, text=True, check=False)
    line = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
    if not line:
        print(f"V2 no answer (exit {p.returncode})", flush=True)
        return p.returncode or 1
    write_private(a.out, {"answer.json": (line + "\n").encode()})
    print(f"V2 {ID} {json.loads(line)['verdict']}", flush=True)
    return p.returncode


if __name__ == "__main__":
    sys.exit(main())
