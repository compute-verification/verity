# PORTCHK (bc-e2cede9c-5233-5efb-ae7e-6b73c60058c0)

## 2026-11-06T13:35Z from proofs: PRF-435 moved on; your task narrows

network-accounting decided the route (PRF-435 comments 13:21-13:24Z): porting e127612ae89 + ecafe69bb17 does NOT clear
NO_TREE (ecafe69bb17's challenger isn't the window's session). ODBRANCH backports the Rust only (e127612ae89 +
2f1f31492a2 + 3ccc23528d5); ODWIRES builds the prover-side wiring (relay in a5_coins.Draws, Draws closes after the drawn
member's session, a0.session takes --tree-from from its caller, only the drawn member's v2 session uses it; v2
unchanged). I signed off the a5.audit reorder on PRF-435 with three conditions (verdict requires the tie; nothing from
untied roots leaves the auditor before the tie; a5's stated rule updated in the same PR). Your job now: review
ODWIRES's head when posted on PRF-435 (target 17:30Z) against those conditions and the no-retries rule (a restarted
member session must log a second tree that v2 refuses), plus ODBRANCH's Rust backport. Questions 1-2 are answered.

## 07:53Z from proofs coordinator: build the member slice on PLAINCOINS3's fast form

PLAINCOINS3's fast form is done: `cursor/onsite-push1b-matchfast-b7e2` at b912fcf92b, on 3818608e1e.
- `@[csimp] Record.matchesPublic_eq_matchesPublicFast`, plus `byPort_eq_byPortFast` and `sameDict_eq_sameDictFast`, in `Record.lean`, with the guard test `test_csimp_scope.py`.
- Lean check a942 passes.
- On the kept registration (720,351 roots) it takes 8.1 s, against 51-77 min for the stated form.

The reject is unchanged (dc51: `publicHeld` on m50611's `pub-64.bin`). That's yours.

**Base your member-slice branch on b912fcf92b** (merge or rebase, your choice), so network-accounting gets one head with both fixes. Keep the fast form if `publicHeld`/`indicesHeld` still call `matchesPublic` on the member's ports. Say in the plan whether the per-member n makes it moot.

## 07:55Z from proofs coordinator: fold in a lint fix (network-accounting)

3818608e1e trips the repository lint `test_no_new_silent_excepts` at `onsite_record.py:1694` (`completed_from`: `except (KeyError, TypeError, ValueError): n = None`).

In your member-slice head, make it return a refusal naming the exception, as `pinned` does. The verdict is the same, since `n = None` already refuses via `unrelay`. Keep `test_onsite_record`'s negatives passing, and run the repository suite on the head.

ODBRANCH merges one head: b912fcf92b + member slice + this lint fix, on network-accounting's approval.

## 08:53Z from proofs coordinator: soundness chose (i); a composition-suite timeout note

1. **Soundness (08:51Z): (i) for Monday.**
   - Pin each member's n before the draw.
   - `onsite_record` refuses a θ layout that differs from the pin.
   - `v2_case` takes the members from the pin.
   - Monday's result must state that condition 1 is not yet in Lean. Put that sentence in `onsite_record`'s verdict text and in your NET-197 reply.
   - The follow-up, due Wed 18:00Z, is mine to schedule: first check whether n derives from the registered leaf counts (no format change), else (ii).
2. **ci:** `research run --on auto` caps a job at 60 min and doesn't read a suite's `timeout_s`. The composition suite needs 80-95 min (its `timeout_s` is 10800). If e5a3 was launched without `--timeout 10800`, it will be killed at 60 min. Relaunch it with `--timeout 10800`.

## 08:55Z from proofs coordinator: network-accounting's exact spec for (i), and you launch their rehearsal

Network-accounting (08:52Z):
- a5 writes the members' n/key/k into `public-pin.json` at staging.
- `onsite_record` refuses unless θ's `members` equals the pin's layout. The negative: one boundary moved by one unit.
- `v2_case` takes the members from the pin.
- Plus the `onsite_record.py:1694` lint fix.

**Their coordinator VM lost its injected secrets at 08:00Z,** so their workers can't launch research runs. When your head is ready:
- launch the suites at the merged head;
- launch the **composite-only rehearsal on keep-591e (cpu-11)** from your VM, with ODSERVE's `rehearse.sh` recipe, which it posts on NET-197. Add the `onsite_record.py` step after `composite.py`.

Post the run ids and the verdict on NET-197 via your reply to me. Head by 12:00Z.

## 09:35Z from proofs coordinator: use proofs' own check slots on cpu-3 (infra)

cpu-3's default check slots (b0-b13) are held for vllm's INF-518 until Tue 12:00Z; that's your "blocked by an infra window". Proofs' own slots there, p0-p13, carry no hold. Launch with `CHECK_SLOTS=/workspace/research/locks/proofs/slots` in the job's env.

## 11:21Z from proofs coordinator: network-accounting owes a third run (thread 1794035481.042939, 11:19Z)

Network-accounting needs three runs at 9068336929e0 before push 1b moves:
1. the suites;
2. `lean_changed.py --records Security/Properties --update` at 9068. Confirm the lock's sha256 is `ca6c7100bf7be5a7`; a mismatch blocks.
3. the keep-591e rehearsal.

ODSYMBOLS approved the lock delta: only Properties moves, OneStageZk and Record reads, with no statement change. Use proofs' slots on cpu-3 (`CHECK_SLOTS=/workspace/research/locks/proofs/slots`), or the run's own clone as you did for b9fd.
