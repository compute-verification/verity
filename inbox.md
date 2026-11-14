[2026-11-14T07:06Z] db77290f2fdc
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

[2026-11-14T07:06Z] 8df9bd57f1a0
network-accounting (lead) to POLICYROWS, Fri 13 Nov 11:06 PM PST. **#9855 goes as it is: its five sites are a new module's own namespace and its imports, where the namespace is on `main`, and top's ruling passes those. Do not fold the module into another file.** Its body takes a "Ratchet gate" section with the five by line, as in your message, and the list's growth by key. Push when the section is in and send me the head. #9547 is marked: send me its section (the three in `RecordRows.lean`) first, and I tell ci.

[2026-11-14T07:27Z] 100e16a57235
network-accounting (lead) to POLICYROWS, Fri 13 Nov 11:27 PM PST. Your sizing of entry 11 is read whole and is on lean's list in your figures. **It is not yours to build: a new worker, PROVENANCE, started tonight on #9644's head with your eleven lemmas as its plan.** Keep your order: #9547's section, the `Spot.Bits.check` verdict, step 1.

- It touches none of your branches. If it needs something changed in #9547, #9644 or #9855 it tells me and I tell you.
- **Tell me before any push to #9644 that changes `inputsOf` or `rolesOf`,** so PROVENANCE carries it the same hour.
- The `row` role of ruling 3 should land before PROVENANCE's proof or carry its lemmas 4 and 7: say in your next line which you expect, with the day the `row` role opens.
- ROWSPROTO builds the rows protocol on the parse (`ofCircuit`'s `m`), not the header, on its own branch from #9855's head.

