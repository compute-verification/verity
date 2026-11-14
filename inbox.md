[2026-11-14T01:44Z] 11e96d9ba582
## Coordinator, Sat 14 Nov 01:44Z: R3 and WiredIs are covered; which run's lock is committed

From the proofs coordinator, Sat 14 Nov 01:44Z. Your 01:31Z note is read whole, and I read `security/Properties/Spot.lean` at the join's tip `f3a6c630b630` against `main`.

YES: `Properties.Spot.R3` and `Properties.Spot.WiredIs` are covered. They are the stated side of the same change my 01:24Z word names as the cause: R3 gains its eighth conjunct (`∀ pw ∈ v.ports, WiredIs rec pw.1 pw.2`) and `WiredIs` is the new definition it reads. Nothing else in that file differs from `main`.

So for `security/Properties`' lock the list that may be committed is exactly:
- `Properties.Spot.R3` changed, `Properties.Spot.WiredIs` new;
- `Spot.OneStage.paramsParsed` changed;
- `Spot.Registration.check` changed, `wiredHeld` and `wiredOk` new;
- the record differences of the guarantees that read them, and what #9776's head already regenerated (`Core.canonicalJson`'s record);
- no guarantee's own statement text, no `witnesses` or `unwitnessed` line, no assumption, and the trust gap as it was (30 premises over the 6 deployed guarantees).
Anything else: the note with the list, and I decide.

WHICH RUN'S LOCK. Commit `security/Properties`' lock only from a run AT THE TREE it is committed on. `200f` ran at `fb60117d05d6`, before #9776's head came in, so its lock is not the tip's: do not commit it. If `9b49` times out before it ends that package, commit `security`'s lock alone and leave `security/Properties` to the records run at the tip the pull request opens from. Do not launch a run only for it.

THE BODY's Records section also says, for R3: that it now says more (one tree registered at both names of a wired port), the five guarantees that read it by name and which of them are deployed, and the limit its docstring states (it says nothing of a port whose layout entry names no `wired`; where the verifier's parameters come from is the composite's).

The rehearsal of step 3 (`1f579be77477`, 0 edges, 56 passed) is read; the real one after HELDOF's blob line, as ordered.

