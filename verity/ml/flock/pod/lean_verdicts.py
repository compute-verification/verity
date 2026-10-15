"""A pod job's Lean step, fail closed (``85-rec-reprice.sh`` and ``86-zk-cpu-steps.sh``, STEP=lean): each part's
flock-verify exit, as its ``/usr/bin/time -v`` record has it (``time-PART.txt``'s ``Exit status``), and the ``VERDICT``
lines of its log (``PART.log``, one per session). An honest part passes when it exited 0 and has a VERDICT and every VERDICT
accepts; a forgery (a part named ``forged-*``) when it exited 1, flock-verify's refusal, with a VERDICT and none accepting.
``flock-verify statement``'s line is its part's VERDICT: accepted when Lean derived the units from the verifier's program
(``partition_check``'s ``units`` ``derived``), refused as ``setup: …``. A part that never ran has neither file and fails.
One line per part, then the count; the exit is 0 only when there is a part and every part passes.

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


def check(d: Path, part: str) -> tuple[bool, str]:
    rc, vs = exit_status(d / f"time-{part}.txt"), verdicts(d / f"{part}.log")
    accepted = [v.get("accepted") is True for v in vs]
    forged = part.startswith("forged-")
    ok = bool(vs) and (rc == REFUSED and not any(accepted) if forged else rc == 0 and all(accepted))
    why = "; ".join(dict.fromkeys(str(v.get("why") or "") for v in vs if v.get("why"))) or ("" if vs else "no VERDICT")
    shown = [v.get("accepted") for v in vs]
    line = (f"lean {part}{' (a forgery: Lean must refuse it)' if forged else ''}: exit {rc}, "
            f"accepted {json.dumps(shown[0] if len(shown) == 1 else shown)}, why {why[:300]!r}{'' if ok else '  <- FAILS'}")
    return ok, line


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__.strip().splitlines()[-1].strip(), file=sys.stderr)
        return 2
    d, parts = Path(argv[0]), list(dict.fromkeys(argv[1:]))
    results = {p: check(d, p) for p in parts}
    for _, line in results.values():
        print(line)
    honest = [p for p in parts if not p.startswith("forged-")]
    forged = [p for p in parts if p.startswith("forged-")]
    count = f"lean: {sum(results[p][0] for p in honest)}/{len(honest)} accepted"
    if forged:
        count += f", {sum(results[p][0] for p in forged)}/{len(forged)} forgeries refused"
    ok = bool(parts) and all(r[0] for r in results.values())
    print(count + ("" if ok else ": the step fails" if parts else ": no part to check, the step fails"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
