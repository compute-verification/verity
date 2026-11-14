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
    launch_coordinator.py tell (--text TEXT | --file PATH) [--only HANDLE ...] [--skip HANDLE ...] [--agent BC_ID ...]
                          [--no-steer] [--dry-run]                # one message to every coordinator's chat but top's

``tell`` sends the text to the agent holding each coordinator handle (the registry's ``holder``; a handle held by a
desktop agent gets nothing) with one ``POST /v1/agents/{id}/runs`` each, as Cursor's "Send A Follow-Up Or Steer"
documents it (https://cursor.com/docs/cloud-agent/api/endpoints): ``delivery: steer`` by default, which a running turn
takes and an idle agent gets as its next follow-up; ``--no-steer`` sends a plain follow-up, which starts a new run and is
refused (409 ``agent_busy``) while one runs. An archived agent refuses either (409 ``agent_archived``).
"""

from __future__ import annotations

import argparse
import base64
import http.client
import json
import os
import re
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
on "{env_name}". Read origin/main's AGENTS.md, its team skill \
(`.agents/skills/team/SKILL.md`) and the lane contract first.
Your slice: {remit} Your paths: {owns}. Your Linear team: {team}. {scope}
First hour, in order:
0. Put the agent guard and research on this VM (the environment's start does so since 18 Oct; this covers a VM booted \
from an older version, and is harmless twice): \
git -C /workspace fetch -q origin main && git -C /workspace show origin/main:tools/agent-guard/boot.sh | bash -s -- /workspace \
&& export PATH=/workspace/.git/verity-auth/bin:/workspace/.venv/bin:$PATH
1. Launch check. Posting needs the name, so take it first (step 4's names open). Run each; a failure that your \
{source} or the {thread_kind} thread's first post already names as known goes in your "up" post and doesn't stop you. \
Post any other first failure in the {thread_kind} thread, then stop:
   git push --dry-run origin HEAD:refs/heads/cursor/launch-check-{suffix}   (the push check)
   gh pr list --repo compute-verification/verity --limit 1   (the broker's app token always reports .permissions.push \
false, so don't probe that)
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
decisions to Notion, runs to the evidence store. Daniel's decisions so far, including what still waits on him, are \
in Notion's "Daniel's decisions log" (https://app.notion.com/p/3f9399515d9e8179bf21e021353fad5b), which the admin keeps.
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


MODEL = "claude-fable-5-2"
EFFORT = "xhigh"
FOLLOW_UPS = "https://cursor.com/docs/cloud-agent/api/endpoints"
CLOUD_AGENT = re.compile(r"bc-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}")


def agent_id(handle: str, generation: int) -> str:
    return f"bc-{uuid.uuid5(NAMESPACE, f'{handle}/{generation}')}"


def user_key() -> str:
    return os.environ.get("CURSOR_API_KEY", "")


def call(method: str, path: str, key: str, body: dict | None = None) -> tuple[int, dict]:
    auth = base64.b64encode(f"{key}:".encode()).decode()
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(API + path, data=data, method=method,
                                 headers={"Authorization": f"Basic {auth}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, decoded(resp.read())
    except urllib.error.HTTPError as err:
        return err.code, decoded(err.read())


def decoded(raw: bytes) -> dict:
    """A response body as a JSON object: ``{}`` when empty, ``{"raw": TEXT}`` when it is not a JSON object."""
    try:
        doc = json.loads(raw or b"{}")
    except ValueError:
        return {"raw": raw.decode(errors="replace")}
    return doc if isinstance(doc, dict) else {"raw": raw.decode(errors="replace")}


def ok(status: int, payload: dict, what: str) -> dict:
    if not 200 <= status < 300:
        sys.exit(f"{what}: HTTP {status} {json.dumps(payload)}")
    if "raw" in payload:
        sys.exit(f"{what}: HTTP {status}, but the body is not a JSON object: {json.dumps(payload['raw'][:200])}")
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
    key = user_key()
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
        # No PR gate on coordinators: leave PR creation to the team's
        # "Create PRs" setting instead of pinning it off per run.
    }
    body["model"] = {"id": args.model, "params": [{"id": "effort", "value": args.effort}]}
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
    key = user_key() or sys.exit("set CURSOR_API_KEY")
    ok(*call("POST", f"/v1/agents/{args.agent}/archive", key, {}), f"archive {args.agent}")
    print(f"archived {args.agent}")


def ids(_: argparse.Namespace) -> None:
    for handle, generation in [(h, 2) for h in SUCCESSORS] + [(h, 1) for h in NEW_SLICES]:
        new_id = agent_id(handle, generation)
        print(f"{handle}\t{generation}\t{new_id}\t{new_id[-4:]}")


def coordinators() -> dict[str, str]:
    """Each coordinator handle and the cloud agent that holds it, from the registry's ``holder``."""
    return {h["handle"]: h["holder"] for h in json.loads(REGISTRY.read_text())["handles"]
            if CLOUD_AGENT.fullmatch(h.get("holder", ""))}


def targets(args: argparse.Namespace) -> list[tuple[str, str]]:
    held = coordinators()
    unknown = [h for h in args.only + args.skip if h not in held]
    if unknown:
        sys.exit(f"not a coordinator handle: {', '.join(unknown)}; nothing sent. The handles {REGISTRY} gives a cloud "
                 f"agent: {', '.join(held)}")
    bad = [a for a in args.agent if not CLOUD_AGENT.fullmatch(a)]
    if bad:
        sys.exit(f"not a cloud agent id (bc-<uuid>): {', '.join(bad)}; nothing sent")
    chosen = [(h, held[h]) for h in dict.fromkeys(args.only)] if args.only else [(h, a) for h, a in held.items() if h != "top"]
    out = [(h, a) for h, a in chosen if h not in args.skip]
    out += [("--agent", a) for a in dict.fromkeys(args.agent) if a not in {b for _, b in out}]
    if not out:
        sys.exit("no targets left; nothing sent")
    return out


def tell(args: argparse.Namespace) -> None:
    text = args.text if args.text is not None else Path(args.file).read_text()
    if not text.strip():
        sys.exit("the message is empty; nothing sent")
    to = targets(args)
    if args.dry_run:
        for handle, agent in to:
            print(f"{handle}\t{agent}\thttps://cursor.com/agents/{agent}")
        print(f"\n{'a plain follow-up' if args.no_steer else 'a steer'} to each of these {len(to)}, saying:\n{text}")
        return
    key = user_key() or sys.exit("set CURSOR_API_KEY (Daniel's user key); nothing sent")
    body = {"prompt": {"text": text}} if args.no_steer else {"prompt": {"text": text}, "delivery": "steer"}
    failed = []
    for handle, agent in to:
        sent, what = send(agent, key, body)
        print(f"{handle}\t{agent}\t{'sent (' + what + ')' if sent else 'FAILED ' + what}")
        if not sent:
            failed.append(f"{handle} {agent}: {what}")
    if failed:
        sys.exit(f"{len(failed)} of {len(to)} not sent:\n" + "\n".join(failed))


def send(agent: str, key: str, body: dict) -> tuple[bool, str]:
    """One target's request: whether the API accepted it, and what came back. Whatever goes wrong for one target is
    reported here, so the others are still sent."""
    try:
        status, payload = call("POST", f"/v1/agents/{agent}/runs", key, body)
    except (OSError, ValueError, http.client.HTTPException) as err:
        return False, f"not reached: {type(err).__name__}: {err}"
    if not 200 <= status < 300:
        return False, f"HTTP {status} {json.dumps(payload)}"
    if "raw" in payload:
        return True, f"HTTP {status}, body not a JSON object: {json.dumps(payload['raw'][:80])}"
    run = payload.get("run")
    return True, str(payload.get("delivery") or "") or f"run {run.get('id', '?') if isinstance(run, dict) else '?'}"


def main(argv: list[str] | None = None) -> None:
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
    q.add_argument("--model", default=MODEL, help=f"a canonical model id from GET /v1/models; default {MODEL} (coordinators run Fable 5.2 at xhigh, never the account default)")
    q.add_argument("--effort", default=EFFORT, help=f"the model's effort parameter; default {EFFORT}")
    q.add_argument("--text-file", help="read the handoff or charter from this file instead of the author's artifacts")
    q.add_argument("--dry-run", action="store_true", help="print the create request and send nothing")
    q.set_defaults(fn=launch)
    q = sub.add_parser("archive", help="archive an agent once it has stopped")
    q.add_argument("agent")
    q.set_defaults(fn=archive)
    q = sub.add_parser("tell", help=f"send one message to every coordinator but top, a steer each ({FOLLOW_UPS}, "
                                    "Send A Follow-Up Or Steer)")
    src = q.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", help="the message")
    src.add_argument("--file", help="read the message from this file")
    q.add_argument("--only", nargs="+", action="extend", default=[], metavar="HANDLE",
                   help="only these handles (top included if named)")
    q.add_argument("--skip", nargs="+", action="extend", default=[], metavar="HANDLE", help="leave these handles out")
    q.add_argument("--agent", nargs="+", action="extend", default=[], metavar="BC_ID", help="also these agent ids")
    q.add_argument("--no-steer", action="store_true",
                   help="a plain follow-up: starts a new run, refused with 409 agent_busy while the agent runs")
    q.add_argument("--dry-run", action="store_true", help="print each target and the message; send nothing, need no key")
    q.set_defaults(fn=tell)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
