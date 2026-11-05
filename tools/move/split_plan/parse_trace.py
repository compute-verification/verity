"""Per-declaration times from trace.profiler logs (one per module): [Elab.command] (statement and sequential elaboration),
[Elab.async] elaborating proof of N, [Kernel] typechecking declarations [N], [Compiler] compiling: [N].

    python3 parse_trace.py PROF_DIR TIMES_JSONL OUT_JSON"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HEAD = re.compile(r"^(\s*)\[([\w.]+)\] \[([\d.e-]+)\] (?:✅️|❌️|💥️)?\s*(.*)$")
DECL = re.compile(r"(?:@\[[^\]]*\]\s*)*(?:(?:private|protected|noncomputable|partial|unsafe|public|nonrec|scoped|local)\s+)*"
                  r"(theorem|lemma|def|abbrev|instance|structure|inductive|class|opaque|example)\s+([^\s:(\[{]+)?")


def _cmd_name(lines: list[str], i: int, rest: str) -> str:
    text = rest
    for j in range(i + 1, min(i + 12, len(lines))):
        if HEAD.match(lines[j]):
            break
        text += "\n" + lines[j]
    if re.match(r"\s*set_option\s+\S+\s+\S+\s+in\b", text):
        return None
    body = re.sub(r"(?s)^\s*/--.*?-/\s*", "", text)
    d = DECL.search(body)
    if d:
        return d.group(2) or f"<{d.group(1)}>"
    return body.strip().splitlines()[0][:60] if body.strip() else rest.strip()[:60]


def parse(text: str):
    out = []  # (kind, name, secs)
    cur_top = None
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = HEAD.match(line)
        if not m:
            continue
        ind, cls, secs, rest = len(m.group(1)), m.group(2), float(m.group(3)), m.group(4)
        if ind == 0:
            cur_top = (cls, rest)
            if cls == "Elab.command":
                nm = _cmd_name(lines, i, rest)
                if nm is not None:
                    out.append(("cmd", nm, secs))
            elif cls == "Elab.async" and rest.startswith("elaborating proof of "):
                out.append(("proof", rest.removeprefix("elaborating proof of ").strip(), secs))
            elif cls == "Elab.async":
                out.append(("async", rest.strip()[:80], secs))
            else:
                out.append((cls, rest.strip()[:80], secs))
        elif ind == 2 and cur_top and cur_top[0] == "Elab.async":
            if cls == "Kernel" and "typechecking declarations" in rest:
                out.append(("kernel", rest.split("[", 1)[-1].rstrip("]").strip(), secs))
            elif cls == "Compiler":
                out.append(("compile", rest.split("[", 1)[-1].rstrip("]").strip(), secs))
        elif ind == 2 and cur_top and cur_top[0] == "Elab.command" and cls == "Kernel":
            out.append(("kernel_sync", cur_top[1], secs))
    return out


if __name__ == "__main__":
    prof, times, dst = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    walls = {}
    for line in times.read_text().splitlines():
        r = json.loads(line)
        walls[r["module"]] = r
    res = {}
    for f in sorted(prof.glob("*.log")):
        mod = f.name[:-4]
        ents = parse(f.read_text(errors="replace"))
        decl = {}
        for kind, name, secs in ents:
            if kind == "kernel_sync":
                d = DECL.match(name.strip())
                key = ((d.group(2) or "") if d else name.strip()[:60]).split(".")[-1]
                decl.setdefault(key, {"cmd": 0.0, "proof": 0.0, "kernel": 0.0, "compile": 0.0})["kernel"] += secs
            elif kind in ("cmd", "proof", "kernel", "compile"):
                key = name.split(".")[-1] if kind != "cmd" else name.split(".")[-1]
                d = decl.setdefault(key, {"cmd": 0.0, "proof": 0.0, "kernel": 0.0, "compile": 0.0})
                d[kind] += secs
        res[mod] = {"wall": walls.get(mod, {}).get("wall_s"), "user": walls.get(mod, {}).get("user_s"),
                    "rc": walls.get(mod, {}).get("rc"), "decls": decl,
                    "other": [(k, n, s) for k, n, s in ents if k not in ("cmd", "proof", "kernel", "compile", "kernel_sync")][:20]}
    json.dump(res, open(dst, "w"), indent=0)
    rows = [(v["cmd"] + v["proof"], mod, name, v) for mod, r in res.items() for name, v in r["decls"].items()]
    rows.sort(reverse=True)
    for tot, mod, name, v in rows[:40]:
        print(f"{tot:7.1f} s  {mod}  {name}  (cmd {v['cmd']:.1f}, proof {v['proof']:.1f}, kernel {v['kernel']:.1f}, compile {v['compile']:.1f})")
