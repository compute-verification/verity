"""A pod job's Lean step, fail closed (``85-rec-reprice.sh`` and ``86-zk-cpu-steps.sh``, STEP=lean): each part's
flock-verify exit, as its ``/usr/bin/time -v`` record has it (``time-PART.txt``'s ``Exit status``), and the ``VERDICT``
lines of its log (``PART.log``, one per session). An honest part passes when it exited 0 and has a VERDICT and every VERDICT
accepts. A forgery is the parts named ``forged-KIND-PART``, PART one of the honest parts named (else the whole name is its
kind): V* accepts only when every part does, so a forgery is refused when one of its parts exited 1, flock-verify's
refusal, with a VERDICT and none accepting, and every other part of it has a VERDICT (a part the forgery leaves as the
honest one is accepted). ``flock-verify statement``'s line is its part's VERDICT: accepted when Lean derived the units from
the verifier's program (``partition_check``'s ``units`` ``derived``), refused as ``setup: …``. A part that never ran has
neither file and fails, except a forgery's part the firewall held back: its prover's ``LIVE`` lines (``PART.live``, the
session verdicts oprove kept) stop at ``FIREWALL-REFUSED: …``, so no session reached Lean, and the part counts as
refused by the firewall. One line per part, then the count; the exit is 0 only when there is a part, every honest part
passes and every forgery is refused.

    python3 verity/ml/flock/pod/lean_verdicts.py DIR PART...
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REFUSED = 1  # flock-verify's exit when a session is refused (2: its inputs, with no VERDICT)


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


def check(d: Path, part: str, kind: str | None = None) -> tuple[bool, bool, str]:
    """``part``'s (passes, refused, line); a forgery's part (``kind``) passes when it has a VERDICT, and is refused when it
    exited 1 with none accepting, or when no session reached Lean because the firewall refused it."""
    rc, vs = exit_status(d / f"time-{part}.txt"), verdicts(d / f"{part}.log")
    fw = None if vs or rc is not None else firewall_refusal(d / f"{part}.live")
    if fw:
        line = (f"lean {part}{f' (forgery {kind}: one of its parts must be refused)' if kind else ''}: refused by the "
                f"firewall before Lean, why {fw[:300]!r}{'' if kind else '  <- FAILS'}")
        return bool(kind), True, line
    accepted = [v.get("accepted") is True for v in vs]
    refused = bool(vs) and rc == REFUSED and not any(accepted)
    ok = bool(vs) and (refused or rc == 0 and all(accepted) if kind else rc == 0 and all(accepted))
    why = "; ".join(dict.fromkeys(str(v.get("why") or "") for v in vs if v.get("why"))) or ("" if vs else "no VERDICT")
    shown = [v.get("accepted") for v in vs]
    line = (f"lean {part}{f' (forgery {kind}: one of its parts must be refused)' if kind else ''}: exit {rc}, "
            f"accepted {json.dumps(shown[0] if len(shown) == 1 else shown)}, why {why[:300]!r}{'' if ok else '  <- FAILS'}")
    return ok, refused, line


def kind_of(part: str, honest: list[str]) -> str:
    """The forgery ``forged-KIND-PART`` is KIND, PART the longest honest part it ends with; else the name without ``forged-``."""
    name = part.removeprefix("forged-")
    ends = [h for h in honest if name.endswith("-" + h)]
    return name[: -len(max(ends, key=len)) - 1] if ends else name


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__.strip().splitlines()[-1].strip(), file=sys.stderr)
        return 2
    d, parts = Path(argv[0]), list(dict.fromkeys(argv[1:]))
    honest = [p for p in parts if not p.startswith("forged-")]
    kinds = {p: kind_of(p, honest) for p in parts if p.startswith("forged-")}
    results = {p: check(d, p, kinds.get(p)) for p in parts}
    for _, _, line in results.values():
        print(line)
    groups: dict[str, list[str]] = {}
    for p, k in kinds.items():
        groups.setdefault(k, []).append(p)
    refused = {k: all(results[p][0] for p in ps) and any(results[p][1] for p in ps) for k, ps in groups.items()}
    for k, ps in groups.items():
        if len(ps) > 1 or not refused[k]:
            by = [p for p in ps if results[p][1]]
            print(f"lean forgery {k}: refused by {by or 'no part'} of {len(ps)}{'' if refused[k] else '  <- FAILS'}")
    count = f"lean: {sum(results[p][0] for p in honest)}/{len(honest)} accepted"
    if groups:
        count += f", {sum(refused.values())}/{len(groups)} forgeries refused"
    ok = bool(parts) and all(results[p][0] for p in honest) and all(refused.values())
    print(count + ("" if ok else ": the step fails" if parts else ": no part to check, the step fails"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
