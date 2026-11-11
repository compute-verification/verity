"""People's posts waiting on an agent: run at every quarterback tick.

Prints each people's chat thread (other than #chat-daniel, which top's subscription already wakes it for) whose last post
is the person's, and each router doorbell to @top from a person since the last run. Exit 1 when anything waits, so the
tick answers it (or routes it to the lane that owns it) before doing anything else.
"""
import json
import pathlib
import re
import subprocess
import sys
import time

STATE = pathlib.Path("/cursor/stores/self/people_check.json")
TOP_INBOX = "1790877429.351069"
CHATS = {"chat-douglas": "douglas", "chat-luke": "luke"}
LINE = re.compile(r"^(\d{10}\.\d+) \d\d:\d\d:\d\dZ (\S+?)(?: \(\(quiet\)\))?: ")
DOORBELL = re.compile(r"^from ([a-z0-9][a-z0-9._-]*)(?: \(URGENT\))?, (#\S+ )?thread (\d+\.\d+)")
AGENTS = re.compile(r"^@")
WEEK = 7 * 86400


def read(*args: str) -> list[tuple[str, str, str]]:
    out = subprocess.run(["research", "msg", "read", *args, "--limit", "400"], capture_output=True, text=True,
                         timeout=120).stdout
    posts = []
    for ln in out.splitlines():
        if m := LINE.match(ln):
            posts.append((m[1], m[2], ln[m.end():]))
        elif posts:
            ts, who, text = posts[-1]
            posts[-1] = (ts, who, text + "\n" + ln)
    return posts


def main() -> int:
    now = time.time()
    state = json.loads(STATE.read_text()) if STATE.exists() else {"inbox_since": f"{now - 3600:.6f}"}
    waiting = []
    for ch, person in CHATS.items():
        roots = [p for p in read("--channel", ch, "--since", f"{now - WEEK:.6f}") if p[1] == person]
        for ts, _, text in roots:
            thread = read("--channel", ch, "--thread", ts)
            said = [p for p in thread if p[1] != "@router" and "has joined the channel" not in p[2]]
            if said and not AGENTS.match(said[-1][1]):
                age = (now - float(said[-1][0])) / 60
                waiting.append(f"OPEN #{ch} thread {ts}: {person} spoke last, {age:.0f} min ago: "
                               f"{said[-1][2][:160]!r} (research msg read --channel {ch} --thread {ts})")
    for ts, who, text in read("--channel", "agent-coordination", "--thread", TOP_INBOX, "--since", state["inbox_since"]):
        if who == "@router" and (m := DOORBELL.match(text)) and m[1] in CHATS.values():
            waiting.append(f"DOORBELL {ts}: from {m[1]}, {m[2] or ''}thread {m[3]}")
    state["inbox_since"] = f"{now:.6f}"
    STATE.write_text(json.dumps(state))
    print("\n".join(waiting) if waiting else "people: nothing waiting")
    return 1 if waiting else 0


if __name__ == "__main__":
    sys.exit(main())
