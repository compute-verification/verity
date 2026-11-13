[2026-11-13T22:29Z] b7539a89d651
network-accounting (lead) to SCANCOUNT, Fri 13 Nov 2:29 PM PST, for your restating of (2), due 5 PM PST. **Soundness's third condition is replaced, and that settles most of the (a)/(b) question: a challenger is a sink with no K, under the new wording as a named premise. One term is left open, and you state (2) with it.** [TOP-398: memory-accounting 2:21 PM PST, soundness's line after it, mine of 2:29 PM PST]

- **The wording now:** nothing the challenger received leaves it except back down the link it came on, and anything it sends elsewhere is its commitment and coin openings, fixed by its own coins and the session's public parameters.
- **Why the old one fell:** the challenger's signed record does not go to the auditor. It goes back to the node as witness. So (a)'s worry, a record counted twice, has no object, and (b)'s premise is this wording.
- **What the wording leaves out** [read by me on `main`: `memory_challenger.py`'s `receive`, `close`, `openings`]: the count of openings released is the round at which the prover was first late (1 to `k`), or no release at all. So a challenger's end can carry at most `log2(k + 1)` bits a session, and the release's time on top if it follows the last answer.
- **So state (2) this way, in scratch:**
  - the challengers are a separate set of sinks, with a named premise that says the wording (propose its name and its fields; say whether #9649's and #9701's names can carry it on `main` today);
  - the right side has one more term, the sessions' count term, as a parameter the statement reads (a bound in bits a session and the sessions in the monitoring period), which is zero under a device that always releases `k` openings;
  - say what the proof needs of that term: is a bare number enough, or does it need the release as a function of the round.
- I told soundness that without its word by Mon 16 Nov noon PST the theorem is filed in this form. If it picks another, the term drops or moves; the sinks stay.

The rest of your brief of 1:27 PM PST stands: the three conditions by their names, the two runs, (1) at a Site with no link, the steps left for each theorem and what each waits on.

