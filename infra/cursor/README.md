# Coordinators as Cursor cloud agents

Each of the team's coordinators (top, lean, proofs and the rest) is a plain Cursor cloud agent. This directory's one
script, `launch_coordinator.py`, launches a coordinator's successor or a new one, archives the old, and puts one message
of Daniel's in every coordinator's chat, all through Cursor's Cloud Agents API.

It runs on Daniel's laptop with his user key. The key stays off every VM, and the GitHub broker trusts only managed,
single-repo agents created by their owner, so no service account launches one.

## Try it

~~~sh
python3 infra/cursor/launch_coordinator.py ids
~~~

This prints each coordinator's handle, its generation, the agent id it has or will have, and its branch suffix. It
needs no key: an agent's id is fixed by its handle and generation, so it is known before launch. `launch` and `tell` take
`--dry-run`, which prints what it would send and sends nothing; the script's docstring gives each subcommand.

An outgoing lead's handoff, or a new slice's charter, travels as an artifact: the author copies it to
`/opt/cursor/artifacts/handoff-HANDLE.md` (or `charter-HANDLE.md`) and ends its turn, and the launch reads it through the
API and puts it under the standard brief.

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
