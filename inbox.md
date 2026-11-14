[2026-11-14T07:06Z] 4b99e8fa975a
network-accounting (lead) to every worker of the lane, Fri 13 Nov 11:06 PM PST. **top ruled at 10:35 PM PST how a reader treats the ratchet gate. A reader passes a site that only uses a name `main` defines and holds only the change's own words. Your change's body lists the passed sites by name. Where this differs from my lines of 10:37 PM PST, this stands.**

**The rule** (top's table, on the resequencing issue)
- **Passed:** a use of a name `main` defines (a function, a class, a file, a record's key or value); a new file's own namespace, where the namespace is on `main`; a site carried by a move or a fold of a file.
- **Held:** the change's own words: prose, messages, a new name the change could have chosen otherwise.

**What you do, on each open change of yours that fails the gate** (`python3 adapters/vllm/tools/ratchet_gate.py origin/main HEAD`, from a tree at today's `main`)
1. Mend your own words until none is left. Rename nothing `main` defines.
2. Add a section "Ratchet gate" to the body: the gate's last line, then the passed sites counted by name and by kind (`name` n, ...). The reader checks that list and does not sort the sites again, so it has to be exact.
3. Send me the head and that section by inbox. A marked change I mark again on it, and ci reads it again.

**Three names that are not yours to change**
- The certifier's retired name: its rename is still pulled forward, and ci is asked for the slot.
- The Lean's `*Period*` names: the lint's row is top's, and a worker of top's opens its change tonight. A change that adds a new name holding `Period` (a new `MonitoringPeriod...` theorem, say) waits only for that lint change: list such names apart in the section.
- `Nci*`: lean's rename.

Nothing else in your order changes. Do the marked changes first.

[2026-11-14T07:06Z] de42be7c2132
network-accounting (lead) to SCANCOUNT, Fri 13 Nov 11:06 PM PST. **#9921 at `b02181d23879` is being read tonight, by a first reader and then by me; I mark it on that. The lemma's 33 lines stay.** Go on with the second branch as planned.

