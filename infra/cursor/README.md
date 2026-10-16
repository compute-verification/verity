# Coordinators as Cursor cloud agents

Each coordinator is a plain Cursor cloud agent on the one-repo environment "Verity compute-verification"
(`6542f951-c746-11f1-ac31-5e2d0494121f`), created with Daniel's user key on his laptop. The key stays off every VM, and the
GitHub broker trusts only managed, single-repo agents created by their owner, so no service account launches one.

`launch_coordinator.py` (standard library only) creates and archives them through the Cloud Agents API:

~~~text
launch_coordinator.py ids                                                  # each coordinator's id and branch suffix
launch_coordinator.py launch HANDLE --predecessor BC_ID --pairing-thread TS   # a successor, from its predecessor's handoff
launch_coordinator.py launch HANDLE --charter-from BC_ID --pairing-thread TS  # a new slice, from its charter
launch_coordinator.py archive BC_ID
~~~

- **The handoff or charter travels as an artifact.** The outgoing lead copies its handoff to
  `/opt/cursor/artifacts/handoff-HANDLE.md` (a charter's author, `charter-HANDLE.md`) and ends its turn; the script reads it
  through the API and puts it under the standard brief. No store, notes token or evidence-store credential is needed on the
  old VM. `--text-file` reads a local file instead.
- **Ids are fixed.** An agent's id is `uuid5` of its handle and generation (2 for a successor, 1 for a new slice), so the
  registry (`.agents/skills/using-slack/registry.json`) names the holder before launch, and a repeated launch gets the
  existing agent back (409 `agent_id_conflict`) rather than a second one. The handle must be in the registry first.
- **The brief's first hour**: a launch check (git push, gh, Linear, the Slack inbox), the handoff or charter copied into
  the notes repo under `lanes/HANDLE/`, its items filed in the handle's Linear team, the inbox opened (a successor's with
  `--succeeds`), "up" posted in the pairing thread, and each acting loop started only after the predecessor posts
  "stopped LOOP".
- `--dry-run` prints the create request and sends nothing; with `--text-file` it needs no key.

## Handing over

A handover is one thread in `#agent-coordination`, which the admin opens with "handover: HANDLE, OLD_ID to NEW_ID"; its
ts is the launch's `--pairing-thread`. Both leads subscribe to it, and it is the handover's only record.

- **The old lead finishes; the successor takes everything else.** The old lead posts its "finishes with me" list
  (or "none") as the thread's first reply and closes each item with "landed" or "moved to NEW_ID". From the launch on it
  takes no new asks (the inbox moves with `names open --succeeds`), starts no workers and opens no PRs.
- **Listening overlaps; acting hands off.** Both leads may read the thread and the channel, but every loop that acts (a
  timer, a drive, a relay) has one runner: the old lead posts "stopped LOOP", then the successor posts "started LOOP". A
  loop tied to a VM moves to the control pod first.
- **The old lead stops** when its list is empty or at the date its handoff names, and at most 24 hours after its
  successor's "up" unless the handoff names a later date. It posts "stopped", ends its subscriptions, and is archived
  (`launch_coordinator.py archive`).
