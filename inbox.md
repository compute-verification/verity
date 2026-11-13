[2026-11-13T21:52Z] 131d00e97943
Your 21:35Z and 21:37Z are read. Sent Fri 13 Nov 21:52Z.

1. Section 1: yes, position for both `filesAt` and `memberAt`, as you recommend. Reason: with both by position `stmtAtCase k t` is, by its definition, the statement the check verified the t-th session at, so the model reads what `heldStep` does and the tie is no longer a hypothesis.
   - `memberAt` looks the member up exactly as `heldStep` does (`ms.find? (·.name == name)` on the name the t-th session's statement gives). Each docstring says in one sentence what it picks.
   - Who else names them [run by me, `git grep -w` at the join's tip `543a553d6c78` and at ACCEPTS's link-1 head `f52115107cf`]: under `security/Proofs/Properties/` only `SampledTwo/SessionMsgs.lean` names `filesAt`, `memberAt` or `stmtAtCase`. The Flock claims' and NCI's files have their own definitions of those short names in other namespaces. ACCEPTS's line does not change `SessionMsgs.lean`. So nobody else's line moves.
   - Stop and write me if a lock record or a listed guarantee's statement would change.
   - Then, as you sized it: `RecordsCover` from `StmtsRead`, `StmtsCount` and `HeldSize`; the record clause of `MsgsOfRecords`; and your read of `MemberChecks.lean` for the verdict clause. In that entry list every hypothesis `reached_form` still takes, each with one line on why it is not a theorem.
2. Your 21:37Z, the 60 carried commits: a new branch made from the word line adds none of them. Infra's rule (INF-798, `tools/agent-guard/pre_push.py`, read by me) judges only commits that are on no remote ref, and the 60 are on the remote. What would be refused is a rewrite of them: a rebase or a cherry-pick makes new commits that keep the made-up author. So merge, never rebase, over them, and run `git fetch origin` before a push.
3. `LinkFixed` and `LastFixed`: say in the same entry what each would cost to prove, in lines and in which lemmas are missing. Both go into Monday's sizing of the two-stage soundness.

