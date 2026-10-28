"""A pod job's Lean step, fail closed (``85-rec-reprice.sh`` and ``86-zk-cpu-steps.sh``, STEP=lean): each part's
flock-verify exit, as its ``/usr/bin/time -v`` record has it (``time-PART.txt``'s ``Exit status``), and the ``VERDICT``
lines of its log (``PART.log``, one per session). An honest part passes when it exited 0 and has a VERDICT and every VERDICT
accepts. A forgery is the parts named ``forged-KIND-PART``, PART one of the honest parts named (else the whole name is its
kind): V* accepts only when every part does, so a forgery is refused when one of its parts exited 1, flock-verify's
refusal, with a VERDICT and none accepting, and every other part of it has a VERDICT (a part the forgery leaves as the
honest one is accepted). ``flock-verify statement``'s line is its part's VERDICT: accepted when Lean derived the units from
the verifier's program (``partition_check``'s ``units`` ``derived``), refused as ``setup: …``. A part that never ran has
neither file and fails. A forgery's part the verifier never checked is "no proof produced (not exercised)", never a
refusal: one whose session Lean found missing its files (``session files: …``), and one the firewall held back (its
prover's ``LIVE`` lines, ``PART.live``, the session verdicts oprove kept, stop at ``FIREWALL-REFUSED: …``, and no session
reached Lean), since the firewall is the prover's own check, which a dishonest prover doesn't run. A forgery no part of
which Lean refused, one of them not exercised, fails the step as not exercised. A forged inner session
(``forged-inner-W-PART``, 85's NEG_INNER) is read only by the parts ``COVERS`` names for it: without one of them among the
parts run, nothing could refuse it, and it is "not covered by the parts run", which fails the step like a forgery no part
refused. One line per part, then the count; the exit is 0 only when there is a part, every honest part passes and every
forgery is refused.

With ``--compose BIN`` (85's V* parts), the recursive statement's verdict is Lean's, not this script's:
``BIN rec-compose`` (`RecCompose.check`) over DIR's ``rec-inner.json``, its one shared rows file ``rows.json``, the
challenger's interactive-run record ``rec-run.json`` and the V* parts' logs (the honest parts 85 gave V*'s inputs,
``own-PART.txt``) accepts only when there is one part per V* statement, each one ``--zk`` session accepted at the SHA-512
of that one rows file and of that inner shape, every part having read one set of the other inputs V*'s parts share and
no two sessions one coin key (`RecCompose.Accepts`), and the record shows the parts are one run: the staging first, one
session opened per part, each the composed one, the close last (`RecCompose.OneRun`); the step fails unless it does.
Each forgery of a V* part, put in that part's place, must be refused by it too. A run of some of V*'s parts (``LEVELS``)
is not composed, and says so; nor is a run with no ``rec-run.json`` (85's loopback ``serve`` keeps no interactive-run
record, and runs WARM + RUNS sessions a part), whose parts are not shown to be one run.

    python3 verity/ml/flock/pod/lean_verdicts.py [--compose BIN] DIR PART...
    python3 verity/ml/flock/pod/lean_verdicts.py --cover KIND PART...   # exit 0 when a PART covers the forgery KIND
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REFUSED = 1  # flock-verify's exit when a session is refused (2: its inputs, with no VERDICT)

# The V* parts that read each forged inner coin (rec_inner.py forge, at level 0): the challenge coin after the binding round
# is the algebra's rec-v rows (its every part, alg-p<i>), the first query coin of Ligerito level 0 is RecOpen L0's rec-dirs
# rows. A part covers KIND when it is one of these or one of its parts.
COVERS = {"inner-challenge": ("alg",), "inner-query": ("L0",)}


def covered_by(kind: str, parts: list[str]) -> list[str] | None:
    """The parts among ``parts`` that read the forgery ``kind``'s coin; None when ``COVERS`` has no entry for it."""
    needs = COVERS.get(kind)
    if needs is None:
        return None
    return [p for p in parts if any(p == n or re.fullmatch(re.escape(n) + r"-p\d+", p) for n in needs)]


def exit_status(time_file: Path) -> int | None:
    m = re.search(r"^\s*Exit status: (\d+)\s*$", time_file.read_text(errors="replace"), re.MULTILINE) if time_file.exists() else None
    return int(m.group(1)) if m else None


def verdicts(log: Path) -> list[dict]:
    if not log.exists():
        return []
    out = []
    for line in log.read_text(errors="replace").splitlines():
        if line.startswith("VERDICT "):
            try:
                out.append(json.JSONDecoder(strict=False).decode(line[8:]))
            except ValueError:
                out.append({"accepted": None, "why": "a VERDICT line that is not JSON"})
        elif line.startswith("{") and '"partition_check"' in line:
            try:
                pc = json.loads(line).get("partition_check") or {}
                out.append({"accepted": pc.get("units") == "derived", "why": pc.get("note", "")})
            except (ValueError, AttributeError):
                out.append({"accepted": None, "why": "a statement line that is not JSON"})
        elif line.startswith("setup: "):
            out.append({"accepted": False, "why": line})
    return out


def firewall_refusal(live: Path) -> str | None:
    """The firewall's refusal among the prover's ``LIVE`` verdicts (one JSON per line), if one stopped at it."""
    if not live.exists():
        return None
    for line in live.read_text(errors="replace").splitlines():
        try:
            stopped = json.loads(line).get("prover_stopped")
        except (ValueError, AttributeError):
            continue
        if isinstance(stopped, str) and stopped.startswith("FIREWALL-REFUSED"):
            return stopped
    return None


def no_proof(v: dict) -> bool:
    """Lean refused the session before any check because a file of it is missing (FlockVerify's ``session files: …``): the
    prover produced no proof, so the verifier checked nothing."""
    return str(v.get("why") or "").startswith("session files:")


def check(d: Path, part: str, kind: str | None = None) -> tuple[bool, bool, bool, str]:
    """``part``'s (passes, refused, not exercised, line); a forgery's part (``kind``) passes when it has a VERDICT or the
    firewall held it back, and is refused when it exited 1 with none accepting and Lean checked a session of it. A part
    whose every session is missing its files, or that the firewall held back, is not exercised, never refused."""
    rc, vs = exit_status(d / f"time-{part}.txt"), verdicts(d / f"{part}.log")
    fw = None if vs or rc is not None else firewall_refusal(d / f"{part}.live")
    if fw:
        line = (f"lean {part}{f' (forgery {kind}: one of its parts must be refused)' if kind else ''}: refused by the "
                f"firewall before Lean, no proof produced (not exercised), why {fw[:300]!r}{'' if kind else '  <- FAILS'}")
        return bool(kind), False, True, line
    accepted = [v.get("accepted") is True for v in vs]
    unexercised = bool(vs) and all(no_proof(v) for v in vs)
    refused = bool(vs) and rc == REFUSED and not any(accepted) and not unexercised
    ok = bool(vs) and (refused or unexercised or rc == 0 and all(accepted) if kind else rc == 0 and all(accepted))
    why = "; ".join(dict.fromkeys(str(v.get("why") or "") for v in vs if v.get("why"))) or ("" if vs else "no VERDICT")
    shown = [v.get("accepted") for v in vs]
    said = "no proof produced (not exercised)" if unexercised else f"accepted {json.dumps(shown[0] if len(shown) == 1 else shown)}"
    line = (f"lean {part}{f' (forgery {kind}: one of its parts must be refused)' if kind else ''}: exit {rc}, "
            f"{said}, why {why[:300]!r}{'' if ok else '  <- FAILS'}")
    return ok, refused, unexercised, line


def kind_of(part: str, honest: list[str]) -> str:
    """The forgery ``forged-KIND-PART`` is KIND, PART the longest honest part it ends with; else the name without ``forged-``."""
    name = part.removeprefix("forged-")
    ends = [h for h in honest if name.endswith("-" + h)]
    return name[: -len(max(ends, key=len)) - 1] if ends else name


def statements(d: Path) -> int:
    """How many statements V* has: ``rec-statements.txt``'s ``RECVSTAR`` line (85's), 0 without one."""
    f = d / "rec-statements.txt"
    lines = [ln for ln in f.read_text(errors="replace").splitlines() if ln.startswith("RECVSTAR ")] if f.exists() else []
    return len(json.loads(lines[0][9:])["statements"]) if lines else 0


def compose(binary: str, d: Path, logs: list[Path]) -> tuple[bool, str]:
    """``binary rec-compose`` over ``logs`` at DIR's ``rec-inner.json``, ``rows.json`` and ``rec-run.json``: (accepted, its
    line)."""
    r = subprocess.run([binary, "rec-compose", "--rec-inner", str(d / "rec-inner.json"), "--rec-rows", str(d / "rows.json"),
                        "--rec-run", str(d / "rec-run.json"), *map(str, logs)], capture_output=True, text=True, check=False)
    line = next((ln for ln in r.stdout.splitlines() if ln.startswith("REC-COMPOSE ")), None)
    return r.returncode == 0 and line is not None, line or f"exit {r.returncode}: {r.stderr.strip()[-300:]!r}"


def composed(binary: str, d: Path, honest: list[str], groups: dict[str, list[str]],
             results: dict[str, tuple[bool, bool, bool, str]]) -> bool:
    """The V* parts composed by Lean (`compose`): the honest ones must compose, and each forgery of one of them, put in
    its place, must not; a forgery Lean never checked (not exercised) is judged above, not here. True when a run of only
    some of V*'s parts is not composed."""
    vparts = [p for p in honest if (d / f"own-{p}.txt").exists()]
    n = statements(d)
    if len(vparts) < n or n == 0:
        print(f"lean compose: {len(vparts)} of V*'s {n} parts run, so the recursive statement is not composed")
        return True
    if not (d / "rec-run.json").exists():
        print("lean compose: no interactive-run record (rec-run.json), so the recursive statement is not composed: its "
              "parts are not shown to be one run (C10 OneRun)")
        return True
    ok, line = compose(binary, d, [d / f"{p}.log" for p in vparts])
    print(f"lean compose: {line}{'' if ok else '  <- FAILS'}")
    for k, ps in groups.items():
        for p in ps:
            part = p.removeprefix(f"forged-{k}-")
            if part not in vparts or results[p][2]:
                continue
            accepted, fline = compose(binary, d, [d / f"{p if q == part else q}.log" for q in vparts])
            print(f"lean compose {p}: {fline}{'  <- FAILS' if accepted else ''}")
            ok = ok and not accepted
    return ok


def main(argv: list[str]) -> int:
    binary = None
    if argv[:1] == ["--compose"]:
        if len(argv) < 2:
            print("lean_verdicts.py --compose BIN DIR PART...", file=sys.stderr)
            return 2
        binary, argv = argv[1], argv[2:]
    if not argv:
        print(__doc__.strip().splitlines()[-1].strip(), file=sys.stderr)
        return 2
    if argv[0] == "--cover":
        if len(argv) < 2:
            print("lean_verdicts.py --cover KIND PART...", file=sys.stderr)
            return 2
        kind, parts = argv[1], argv[2:]
        by = covered_by(kind, parts)
        if by == []:
            print(f"lean forgery {kind}: not covered by the parts run (needs one of {list(COVERS[kind])}, or its parts; "
                  f"ran {parts})")
            return 1
        return 0
    d, parts = Path(argv[0]), list(dict.fromkeys(argv[1:]))
    honest = [p for p in parts if not p.startswith("forged-")]
    kinds = {p: kind_of(p, honest) for p in parts if p.startswith("forged-")}
    results = {p: check(d, p, kinds.get(p)) for p in parts}
    for *_, line in results.values():
        print(line)
    groups: dict[str, list[str]] = {}
    for p, k in kinds.items():
        groups.setdefault(k, []).append(p)
    uncovered = {k for k, ps in groups.items() if covered_by(k, [p.removeprefix(f"forged-{k}-") for p in ps]) == []}
    refused = {k: k not in uncovered and all(results[p][0] for p in ps) and any(results[p][1] for p in ps)
               for k, ps in groups.items()}
    unexercised = {k: [p for p in ps if results[p][2]] for k, ps in groups.items()}
    unexercised = {k: ps for k, ps in unexercised.items() if ps and not refused[k] and k not in uncovered}
    for k, ps in groups.items():
        if k in uncovered:
            print(f"lean forgery {k}: not covered by the parts run (needs one of {list(COVERS[k])}, or its parts; ran "
                  f"{[p.removeprefix(f'forged-{k}-') for p in ps]})  <- FAILS")
        elif k in unexercised:
            print(f"lean forgery {k}: no proof produced (not exercised) at {unexercised[k]} of {len(ps)}, so the verifier "
                  f"never checked it  <- FAILS")
        elif len(ps) > 1 or not refused[k]:
            by = [p for p in ps if results[p][1]]
            print(f"lean forgery {k}: refused by {by or 'no part'} of {len(ps)}{'' if refused[k] else '  <- FAILS'}")
    count = f"lean: {sum(results[p][0] for p in honest)}/{len(honest)} accepted"
    if groups:
        count += f", {sum(refused.values())}/{len(groups)} forgeries refused"
    if uncovered:
        count += f", {len(uncovered)} not covered by the parts run"
    if unexercised:
        count += f", {len(unexercised)} not exercised (no proof produced)"
    ok = bool(parts) and all(results[p][0] for p in honest) and all(refused.values())
    if binary:
        ok = composed(binary, d, honest, groups, results) and ok
    print(count + ("" if ok else ": the step fails" if parts else ": no part to check, the step fails"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
