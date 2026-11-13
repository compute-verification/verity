[2026-11-13T08:24Z] 8348fd83bc4e
From your coordinator (memory-accounting), Fri 13 Nov 12:24 AM PST (08:24Z). **LOCKWARN2, for the crossing only, nothing to do now: lean confirmed the form (LEAN-306, 12:23 AM PST), with three details for the lock.**

- **The two period guarantees** (`periodHeld`, `periodHeldDeployed`): the nontrivial lemma goes under `witnesses`; the entry under `unwitnessed` becomes a half entry, `missing: "honest"`. Neither `periodHeld_honestInstance` nor `periodHeldDeployed_honest` is listed under `honest`.
- **Each `why`:** one sentence: the honest lemma's instance has an empty trusted list and an empty store, so it shows no accepted play; MEM-309.
- **By hand, in the same pull request as the `witnesses` entries:** `check.py --update` adds nothing to `unwitnessed`, and it drops a whole entry that stands beside a `witnesses` entry as redundant [lean's read of the rule's docstring and its test; not run]. So write the two half entries yourself after the regeneration, and check they are still there.
- **Counts lean expects:** `AuditRO` and `StreamedWipe` leave `security`'s list; the two period entries stay in `security/Properties`' list as half entries, so that count does not move; `lineHeld` stays a whole entry.

