#!/usr/bin/env python3
"""Launch and archive Verity's coordinators, each a plain Cursor cloud agent on one environment.

Run it on Daniel's laptop with ``CURSOR_API_KEY`` set to his user key. The key never goes to a VM, and the GitHub
broker trusts only managed, single-repo agents created by their owner (no service accounts), so a coordinator is
always created here, by Daniel's key, on ``ENVIRONMENT`` alone.

A successor (``--predecessor``) starts from the outgoing lead's handoff, ``artifacts/handoff-HANDLE.md`` in that lead's
own artifacts (it writes the file to ``/opt/cursor/artifacts/``). A new slice (``--charter-from``) starts from a charter,
``artifacts/charter-HANDLE.md`` in its author's artifacts. Either way the text goes under the standard brief, and the
agent gets a fixed id, ``uuid5`` of the handle and the generation (1 for a new slice, 2 for a successor), so the
registry can name it before launch and a repeated launch gets the existing agent back (409 ``agent_id_conflict``)
instead of a second one.

    launch_coordinator.py ids                                     # every coordinator's id and branch suffix
    launch_coordinator.py launch HANDLE --predecessor BC_ID --pairing-thread TS [--model ID] [--dry-run]
    launch_coordinator.py launch HANDLE --charter-from BC_ID --pairing-thread TS [--model ID] [--dry-run]
    launch_coordinator.py archive BC_ID                           # idempotent
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

API = os.environ.get("CURSOR_API_URL", "https://api.cursor.com")
ENVIRONMENT = "6542f951-c746-11f1-ac31-5e2d0494121f"
NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "https://github.com/compute-verification/verity/coordinators")
REGISTRY = Path(os.environ.get("RESEARCH_SLACK_REGISTRY")
                or Path(__file__).resolve().parents[2] / ".agents" / "skills" / "using-slack" / "registry.json")
SUCCESSORS = ("top", "compute-accounting", "network-accounting", "lean", "memory-accounting", "proofs", "circuits",
              "comms", "infra", "ci")
NEW_SLICES = ("vllm", "benchmarks")

HEAD = """\
You are the {handle} coordinator for Verity (compute-verification/verity), {role}. You are a plain Cursor cloud agent \
on "{env_name}". Read origin/main's AGENTS.md and the lane contract first.
Your slice: {remit} Your paths: {owns}. Your Linear team: {team}. {scope}
First hour, in order:
1. Launch check. Run each, and post the first failure in the {thread_kind} thread, then stop:
   git push --dry-run origin HEAD:refs/heads/cursor/launch-check-{suffix}
   gh api repos/compute-verification/verity --jq .permissions.push
   research msg linear check
   research msg names read --as {handle} --peek
   plus the lane checks your {source} names (Lean, pod ssh, the Nebius CLI, gpu-lease, slot.py).
2. Copy the {source} below into the notes repo as lanes/{handle}/<UTC YYYYMMDDTHHMMZ>-report-{source}.md and push it \
with research notes sync --path lanes/{handle}. This is also the notes-token check.
3. File each item under "{items}" as a Linear issue in {team}, linked to the note.
4. {inbox} Then make every subscribe call that research msg subscriptions --as {handle} prints.
5. Post "up" in the {thread_kind} thread (#agent-coordination, thread {thread}) with your link.
6. {acting}
Keep nothing only in a Cursor agent store: it's scratch. Reports and logs go to the notes repo, work items to Linear, \
decisions to Notion, runs to the evidence store.
--- {source} from {author} ---
"""

SUCCESSOR = dict(
    role="the successor to {author}",
    scope='Your predecessor finishes only the items under "Finishes with me" in its handoff; everything else, and '
          "everything new, is yours.",
    thread_kind="pairing", source="handoff", items="Hands over",
    inbox="Take the inbox: research msg names open --as {handle} --succeeds.",
    acting='Start an acting loop (a timer that acts, a drive, a relay) only after your predecessor posts "stopped LOOP" '
           'in the pairing thread; then post "started LOOP".',
)
NEW_SLICE = dict(
    role="a new slice, with no predecessor (charter by {author})",
    scope='A path in your charter that another lead owns today becomes yours when that lead posts "moved PATH" in the '
          "launch thread; until then, ask it, and build nothing there.",
    thread_kind="launch", source="charter", items="First outcomes",
    inbox="Open your inbox: research msg names open --as {handle}.",
    acting="Agree the path split your charter leaves open with the leads it names, in the launch thread, and ask comms "
           "to record it in the registry.",
)


def agent_id(handle: str, generation: int) -> str:
    return f"bc-{uuid.uuid5(NAMESPACE, f'{handle}/{generation}')}"


def call(method: str, path: str, key: str, body: dict | None = None) -> tuple[int, dict]:
    auth = base64.b64encode(f"{key}:".encode()).decode()
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(API + path, data=data, method=method,
                                 headers={"Authorization": f"Basic {auth}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as err:
        raw = err.read()
        try:
            return err.code, json.loads(raw or b"{}")
        except ValueError:
            return err.code, {"raw": raw.decode(errors="replace")}


def ok(status: int, payload: dict, what: str) -> dict:
    if not 200 <= status < 300:
        sys.exit(f"{what}: HTTP {status} {json.dumps(payload)}")
    return payload


def entry(handle: str) -> dict:
    for h in json.loads(REGISTRY.read_text())["handles"]:
        if h["handle"] == handle:
            return h
    sys.exit(f"{handle} is not a handle in {REGISTRY}; a coordinator needs one before launch (comms adds it)")


def environment_name(key: str) -> str:
    env = ok(*call("GET", f"/v1/environments/{ENVIRONMENT}", key), f"environment {ENVIRONMENT}")
    if len(env.get("repos", [])) != 1:
        sys.exit(f"environment {ENVIRONMENT} has {len(env.get('repos', []))} repos; the broker accepts single-repo agents only")
    return env["name"]


def fetch_artifact(author: str, name: str, key: str) -> str:
    want = f"artifacts/{name}"
    items = ok(*call("GET", f"/v1/agents/{author}/artifacts", key), f"artifacts of {author}")["items"]
    if not any(i["path"] == want for i in items):
        sys.exit(f"{author} has no {want}: it copies the file to /opt/cursor/artifacts/{name} and ends its turn, "
                 f"then rerun this")
    query = urllib.parse.urlencode({"path": want})
    url = ok(*call("GET", f"/v1/agents/{author}/artifacts/download?{query}", key), f"download {want}")["url"]
    with urllib.request.urlopen(url) as resp:
        return resp.read().decode()


def brief(handle: str, author: str, new_slice: bool, thread: str, env_name: str, new_id: str, text: str) -> str:
    h = entry(handle)
    parts = {k: v.format(author=author, handle=handle) for k, v in (NEW_SLICE if new_slice else SUCCESSOR).items()}
    return HEAD.format(handle=handle, author=author, env_name=env_name, remit=h["remit"],
                       owns=", ".join(h.get("owns", [])) or "as your charter or handoff says",
                       team=h.get("linear") or "none", suffix=new_id[-4:], thread=thread, **parts) + text


def launch(args: argparse.Namespace) -> None:
    key = os.environ.get("CURSOR_API_KEY", "")
    if not key and not (args.dry_run and args.text_file):
        sys.exit("set CURSOR_API_KEY (Daniel's user key; a service-account key is refused by the broker)")
    entry(args.handle)
    new_slice = args.charter_from is not None
    author = args.charter_from or args.predecessor
    generation = args.generation or (1 if new_slice else 2)
    new_id = agent_id(args.handle, generation)
    env_name = environment_name(key) if key else "Verity compute-verification"
    kind = "charter" if new_slice else "handoff"
    text = (Path(args.text_file).read_text() if args.text_file
            else fetch_artifact(author, f"{kind}-{args.handle}.md", key))
    body = {
        "agentId": new_id,
        "name": f"verity: {args.handle}",
        "prompt": {"text": brief(args.handle, author, new_slice, args.pairing_thread, env_name, new_id, text)},
        "env": {"type": "cloud", "name": env_name},
        "autoCreatePR": False,
    }
    if args.model:
        body["model"] = {"id": args.model}
    if args.dry_run:
        print(json.dumps(body, indent=2))
        return
    status, payload = call("POST", "/v1/agents", key, body)
    if status == 409 and "agent_id_conflict" in json.dumps(payload):
        print(f"{args.handle}: already launched, https://cursor.com/agents/{new_id}")
        return
    agent = ok(status, payload, f"create {args.handle}")["agent"]
    print(f"{args.handle}: launched {agent['id']}, {agent['url']}")


def archive(args: argparse.Namespace) -> None:
    key = os.environ.get("CURSOR_API_KEY") or sys.exit("set CURSOR_API_KEY")
    ok(*call("POST", f"/v1/agents/{args.agent}/archive", key, {}), f"archive {args.agent}")
    print(f"archived {args.agent}")


def ids(_: argparse.Namespace) -> None:
    for handle, generation in [(h, 2) for h in SUCCESSORS] + [(h, 1) for h in NEW_SLICES]:
        new_id = agent_id(handle, generation)
        print(f"{handle}\t{generation}\t{new_id}\t{new_id[-4:]}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ids", help="every coordinator's generation, id and branch suffix").set_defaults(fn=ids)
    q = sub.add_parser("launch", help="create a coordinator: a successor from a handoff, or a new slice from a charter")
    q.add_argument("handle")
    src = q.add_mutually_exclusive_group(required=True)
    src.add_argument("--predecessor", help="a successor: the outgoing lead's agent id (reads artifacts/handoff-HANDLE.md)")
    src.add_argument("--charter-from", help="a new slice: the charter author's agent id (reads artifacts/charter-HANDLE.md)")
    q.add_argument("--pairing-thread", required=True, help="ts of the handover or launch thread in #agent-coordination")
    q.add_argument("--generation", type=int, help="default: 2 for a successor, 1 for a new slice")
    q.add_argument("--model", help="a model id from GET /v1/models; default: Daniel's default model")
    q.add_argument("--text-file", help="read the handoff or charter from this file instead of the author's artifacts")
    q.add_argument("--dry-run", action="store_true", help="print the create request and send nothing")
    q.set_defaults(fn=launch)
    q = sub.add_parser("archive", help="archive an agent once it has stopped")
    q.add_argument("agent")
    q.set_defaults(fn=archive)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
