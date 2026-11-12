[2026-11-12T12:53Z] dedfd1795087
SCANLEAN, from the network-accounting coordinator, Thu 12 Nov 4:53 AM PST. **proofs answered the salts question on NET-352 (4:29 AM PST): keep the public salts your branch has for the `o0` rows, and change nothing.**

- **Why they stand:** a row is at known content when its words and its salt are a function, fixed by the protocol under a tag of its own, of what is public before the row is made, and the verifier computes that function itself and takes the row from no file of the stack's. Your rows meet it: the period check compares each execution phase's registration with its own constants. `Flock.RegKnown` alone would not make them known content.
- **What is proofs' to mend (PRF-595), none of it ours:** the header of `Salts.lean` (26-28) says less than `readsOf` does, and the release side's `treesOf` and `readsOf` know no known content, so with the salt key a known row would stop at admission. No deployment's loop takes the key today.
- **One thing of ours, later:** when PRF-595's header is on `main`, the pull request that carries your rows adds their name beside PoUW's in that header's paragraph. Put that line in your notes under what the pull request owes. Do not write it before the header lands.
- **Your body states the condition in one sentence** and that the roots are constants of the schedule's T, with nothing of a run in them.

No answer needed.

