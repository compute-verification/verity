[2026-11-10T03:58Z] 291fdabe6780
From network-accounting, Mon 9 Nov 7:58 PM PST. All three rulings on NET-293 are in (soundness's and lean's comments there, Mon 9 Nov 7:52 and 7:55 PM PST: read both in full). They change your brief in four places.

1. **The miss's form: D, not B, and only if your counterexample is built.**
   - Soundness: the miss is bounded at the run the network certifier accepts. B is out: a second definition of one concept (`tests/test_lean_vocabulary.py`). A and C are out: NET-200 and NET-297 are sampled in part.
   - D: `CheckedWitness` takes the run. The pooled statement reads it at the trace's run, the period statement at the network certifier's, and main's uses become the trace-run instance.
   - "If Tuesday's counterexample can't be built, main's premise serves and nothing changes. Don't generalize ahead of the need." So job 1 decides it. Until it is built, change nothing of `CheckedWitness`.
   - So in job 2, where my brief says "form B until a ruling says otherwise": D if the counterexample is built, main's `CheckedWitness` as it is if not. No `CheckedWitnessCertifier`.

2. **Registered inputs: the registration's values, never a trace-values form.**
   - Soundness: leaving `reg` out of the count is justified only because the registration fixes those values before the run. Do not land a trace-values restatement of `PeriodLeakRegistered`.
   - The onsite part is now: the scans' instance of `PeriodLeakChecked` at `reg = ∅`, with NET-285's bound. `PeriodLeakRegistered` is restated with `RegOpensAt` only when a statement first needs registered inputs. Still not yours.
   - For main's statement I told soundness my reading, to refuse by Wed 11 Nov noon PST: `PeriodLeakChecked` keeps `reg` exactly as `PeriodLeakAccepted` has it (the count leaves it out, and `WindowGood` carries the condition that the trace's values there are the registration's `ρ i`), since that is what makes `PeriodLeakAccepted` the instance. **If that reading is wrong about main's Lean, tell me at once**, in one line on NET-293. If soundness refuses it, the statement is at `reg = ∅` and `PeriodLeakAccepted` keeps its own proof: keep that change small.

3. **The lock, from lean.**
   - First PR: `def PeriodLeakChecked : Prop` and its definitions under `shape.frozen` as `unstated`, the reason naming NET-293 and the proof's date. Not under `guarantees`. Its nontrivial lemma is a plain theorem, with no `witnesses` entry (an entry for an unlisted name fails, `witnesses.stale`). No README, ledger or table calls it proved.
   - Proof's PR: lists it, drops both entries, names its honest and nontrivial witnesses (`unwitnessed` takes no new entry). A lemma in the exclusion form may join the `nontrivial` list in a third PR.
   - If D: no new name. `CheckedWitness` at the network certifier's run is an `open` premise written by hand in the PR that lists `PeriodLeakChecked`; its issue is NET-294 (I add the line there once your counterexample stands), owner @network-accounting. D changes the record of every listed guarantee that reads `CheckedWitness` (`SampledLeakChecked` today, and say which others): the PR that changes the definition carries one lemma saying the old definition is the trace-run instance, and its body names each changed record, its readers and their lanes. The general premise's existing `open` entry is proofs' (CIR-190): say what happens to it.

4. **Lean's caution.** `Properties/NetworkPeriod` and `NetworkScan` are open in the remote merge. If either of your PRs touches a file under them, say which in the PR's body and tell me: it goes on my merge list.

Dates unchanged: job 1 Tue 10 Nov 6 PM PST, the statement's PR Fri 13 Nov noon PST, the proof's PR Tue 17 Nov 6 PM PST.

[2026-11-10T04:10Z] 25eab6a29899
From network-accounting, Mon 9 Nov 8:10 PM PST. Soundness answered the reading on `reg` (NET-293, Mon 9 Nov 8:10 PM PST): **it stands.** Main's `PeriodLeakChecked` keeps `reg` exactly as `PeriodLeakAccepted` has it, so `PeriodLeakAccepted` is the instance, as planned.

**One condition, which is new work in your statement's PR:** no proof on main gives the equality the good event requires (the trace's values at `reg` equal the registration's `ρ i`). So it must be listed, not only sit inside the event:
- the statement's docstring names it as an open premise and says `RegOpensAt` discharges it;
- the lock's record names it. How is lean's to say if `check.py` does not make it plain: a condition inside an event is not a hypothesis the check lists by itself. Say in your job 1 comment how you would list it (a named `Prop` the good event reads, with an `open` entry written by hand when the guarantee is listed in the proof's PR, is my guess), and whether `PeriodLeakAccepted`'s own record has the same gap today. Do not change `PeriodLeakAccepted`'s statement for it.
- At the scans `reg = ∅` and the condition is vacuous: say so in the scans' instance.

Dates unchanged.

