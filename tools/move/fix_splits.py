#!/usr/bin/env python3
"""`tools/move/fix_splits.py MAP CERT.json|--text OUTPUT [--log FILE]` (STAGING ONLY): make a map's `[split]` rows what `layout.py` asked
for when `stage.py --through-refusals` could not run the map: drop each row it called stale ("[split] acknowledges what
no reference needs any more: KEY") and add each reference it called unacknowledged. Prints and logs every row it drops or
adds, for the map's owner; exits 1 when the certificate names nothing to fix."""
import json
import re
import sys
from pathlib import Path

REASON = "added by the staging run: `layout.py` names this reference and the map's own scan did not"


def main(argv):
    path = Path(argv[0])
    log = Path(argv[argv.index("--log") + 1]) if "--log" in argv else None
    if argv[1] == "--text":                 # the script's own output (stage.py's STAGE_KEEP_OUTPUT), every line of it
        text = Path(argv[2]).read_text()
    else:
        text = "\n".join(m.get("refused") or "" for m in json.loads(Path(argv[1]).read_text())["moves"] if not m.get("commit"))
    stale = re.findall(r"\[split\] acknowledges what no reference needs any more: (.+)", text)
    new = [x.split(" (", 1)[0] for x in re.findall(r"^  (?!\[split\])(\S[^\n]*: [^\n]+ \([^\n]*\))$", text, re.M)]
    if not stale and not new:
        return 1
    lines, out, dropped, in_split = path.read_text().split("\n"), [], [], False
    for line in lines:
        if line.startswith("["):
            if in_split:
                out += [f"{json.dumps(k)} = {json.dumps(REASON)}" for k in new]
            in_split = line.strip() == "[split]"
        if in_split and (m := re.match(r'^("(?:[^"\\]|\\.)*")\s*=', line)) and json.loads(m.group(1)) in stale:
            dropped.append(json.loads(m.group(1)))
            continue
        out.append(line)
    if new and "[split]" not in lines:
        raise SystemExit(f"{path}: no [split] table to add {len(new)} rows to")
    path.write_text("\n".join(out))
    said = [f"dropped (stale): {k}" for k in dropped] + [f"added (unacknowledged): {k}" for k in new]
    said += [f"NOT FOUND in [split]: {k}" for k in stale if k not in dropped]
    print("\n".join(said))
    if log:
        with log.open("a") as f:
            f.write("\n".join(said) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
