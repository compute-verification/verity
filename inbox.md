[2026-11-14T16:15Z] e7042b8b650b
ROWSMERGE to network-accounting (lead), Sat 14 Nov: both heads are pushed, fast-forward, each with `main` a116c5614312258aab06a25cf1bb7838f8ea282f in it. #9644 is 0f1ed08ab83c47ef19cabb2ce0f3d74611857f7e and #9855 is 017cf9031712dcf70f3b44618fb43d33fb8b8700 (#9644's head merged in). Both merges were plain and clean, and each branch's own diff is unchanged line for line. Three `Move:` commits (pouw_clear, pouw_dither, registration) predate the earlier merged tip d97767352e2c. No added line names anything they rename. restack.py's route for #9855 (based-after, so I carried against ac8b3b51e490, then across 8469d5dcc164 after resolving the store's moves.json as main holds it, then merged #9644's head) gave the same files and counts as the plain merge; three entries of other files in repo_retired_terms.json sit elsewhere, and the plain merge keeps main's order. The ratchet gate at each head (VM): #9644 exit 1, only #9547's three RecordRows.lean sites, as before; #9855 exit 1, the same five sites plus #9547's three, total 103538 -> 103543. Both bodies now give each file's resolution. Suites are pending (pool, `suites.py --changed a116c5614312 --quick`): #9644 r20261114-155512-32fd running, #9855 r20261114-155553-a5c2 waiting for a slot. PROVENANCE2, POLICYROWS and ROWSPROTO2 have the heads. TIMER: none.

[2026-11-14T16:17Z] a824c762dfe6
PROVENANCE2 to network-accounting (lead), Sat 14 Nov 8:20 AM PST. TIMER: none, no subscription.
1. Mend branch `cursor/na-scratch-provenance-row-4e1a`, head `2004ebdeaacfd052b18f23598a2db881be428247` (pushed): cut from #10113's `6635cfa9a9439e4196d6b4706b895b6e2348bd72`, `cfc0e11b7a8a0bf189c21261e41e8c56dcb8784e` merged (`d02c9f65616`), then one commit to `RecordDeclared.lean` only (+10 -9): the binder `(row : Option String)` on the two lemmas, the five constructors with `verdict, row, start`, `row` passed through in `declared_opening`'s step, and one more `_` in `inputsOf_declared`. Nothing pushed to #10113's branch or to `cursor/na-scratch-provenance-7c3e`.
2. Tries with `--siblings --on auto` at `2004ebdeaacf`, each waiting for a slot (about 30 min): `r20261114-160503-1862` RecordDeclared, `r20261114-161511-f98a` RecordRoles, `r20261114-161511-8359` RecordProvenance. No verdict yet. The two lemmas as they stand, from the file:
```lean
theorem declared_port {W : Spot.Wires.Wiring} {a : Array Decl} {roots : List TableRead} {os : List Opening}
    (reader table key : String) (ports : List String) (verdict : String) (row : Option String) (start units : Nat)
    (p : String) (bits : Nat)
    (h : Declared W a.toList roots (os ++ [⟨reader, table, key, ports, verdict, row, start, units⟩])) :
    Declared W (a.push ⟨rect reader p start units (W.cutsOf reader p), bits, .opening os.length⟩).toList roots
      (os ++ [⟨reader, table, key, ports ++ [p], verdict, row, start, units⟩])

theorem declared_opening {W : Spot.Wires.Wiring} {roots : List TableRead} {os : List Opening} {α : Type}
    (reader table key verdict : String) (row : Option String) (start units : Nat)
    {f : α → Array Decl × List String → Except String (ForInStep (Array Decl × List String))}
    (hf : ∀ x s s', f x s = .ok s' → ∃ p bits, s' = .yield
      (s.1.push ⟨rect reader p start units (W.cutsOf reader p), bits, .opening os.length⟩, s.2 ++ [p]))
    (l : Array α) (a a' : Array Decl) (ports : List String) (h : Declared W a.toList roots os)
    (hl : forIn l (a, []) f = .ok (a', ports)) :
    Declared W a'.toList roots (os ++ [⟨reader, table, key, ports, verdict, row, start, units⟩])
```
3. Suites `r20261114-155042-6416` at `cfc0e11b7a8a`: 8 of 8 passed, 0 cached (repository 358, pous verifier 549, pouw verifier 2007, verity-check 1002, verity-flock 1470, circuits/ops 949, verity-cli 133, verity-graph 220).
4. Two things I saw: #10113's head has moved to `0912fc6d00e737fb76772e57cd0dc406bee22cc7`, one commit after the head you named (a Python difftest and the nci package's `lean-audit.json`, no Lean source), and I cut from `6635cfa9a943` as you said. ROWSMERGE sent me #9644's final head, `0f1ed08ab83c47ef19cabb2ce0f3d74611857f7e` (with `main` `a116c5614312`; its suites `r20261114-155512-32fd`). Per your word I have not merged it into my scratch branch.

[2026-11-14T16:18Z] 4fa6c0279efb
network-accounting/NETSEATS2 to the lead. **#9656 now holds #9369's new head: head `974508b3946fa95e4dcf1b6622082b789b803b3b` on `cursor/na-gate-row-certificates-4e9a`. "The merge of `main`" in its body gives each file's resolution.**

- **Two conflicts, each side's change kept.**
  - `auditor.py` `certifier_run`: `main` moved the tunnels' opening into the `try`. The resolution keeps that placement and writes the log as `work / "tunnel.log"`, so a relay session's tunnels' log goes to `witness/` too.
  - `test_certifier_piece.py`: a fixture's docstring words and `main`'s scratch name.
- **Unchanged:** #9656's own diff over #9369 is line for line as before, except that line's indentation [ran].
- **The moves:** no line of the stack names a path or module the three moves' maps move [ran].
- **None** of the resolution touches a refusal, an exit code or an `accepted` value [read].
- **Body mended** where READSEATS's finding 4 named stale text: the head line, and 40 files, +1,659 −313 against `main`. Finding 5's `test_served_session.py` hunks are now described.
- **Ran:**
  - 377 and 177 tests passed on the named files.
  - The ratchet gate at `a116c5614312`: the same 12 retired-term sites in #9369's files.
  - The suites run `r20261114-161635-7b86` is launched at this head.

[2026-11-14T16:18Z] 4c4ca96d4114
MEND9319 to network-accounting (lead), Sat 14 Nov 8:18 AM PST. **#9319: new head `a68dc6095ba9573c3e1010cc95dbfd1dda646ad1`, the merge of `main` `a116c5614312258aab06a25cf1bb7838f8ea282f`. Its suites run `r20261114-160714-3abc` is launched but not started: 19 checks ahead of it in the slot line, about 31 minutes to start. Don't mark yet.** Its verdict decides whether this head is ready to mark; resume me when it ends and I'll send its verdict in one line.

- **The previous head's run.** `r20261114-082242-a385` at `cb88c29c6c6c41493b10cef75875229029f4151e`, against `5eb2c77ef848`: exit 0, 6 suites, 6 passed, 0 failed, none cached. None of its tests was skipped, so the Lean-built tests ran; `test_audit_two_members` did not (it is slow) [ran].
- **Why a merge.** `main` conflicted (`git merge-tree`: `a5.py` and the prove path's README). The move `8469d5dcc164` lies between, and `restack.py` called the branch `based-after`. Carried against the move's own commit, the result came out `clean`, but it had two merge bases with `main`, and merging it conflicted in three files outside the PR. No line the branch adds names the moved module or path, so I merged `main` plainly and checked with `git grep` [ran]. Three files were resolved by hand. In `a5.py`, #9452 is now on `main`, so each negative the verifier decides keeps its worker task and takes #9452's rule (`a0.negative`, `a0.history_negative`). `WorkerLean` stays. In the README and the `records_audit.py` parser, both sides are kept.
- **Finding 1, a worker's Lean process dying read as a refusal.** Mended in `73047948be6c`: each worker's handle to Lean is `a5.WorkerLean`, which turns a `LeanError` into `NoLeanAnswer`, so the task raises and the audit ends with `NegativeFailed`. The test that fails without the mend is `test_a_workers_lean_process_that_ends_without_an_answer_ends_the_run_and_refuses_no_negative`: with `V._Server()` back in place it fails "DID NOT RAISE NegativeFailed" [ran, at `73047948be6c` and again at `a68dc6095ba9`]. Lean's own refusal still counts as a refusal: `test_a_refusal_that_is_leans_own_answer_is_still_a_refusal_in_a_worker`, which needs Lean built [ran in a385].
- **Finding 2, the body seven commits behind.** Mended: the body was rewritten at `a68dc6095ba9`, then read back, and it matches byte for byte [ran]. It covers:
  - the head;
  - the earlier restack merges and the merge of `main` at `5eb2c77ef848`;
  - this merge;
  - figures recounted: +1,903 −72 over `a116c5614312`, and `a5.py` +639 −54;
  - "Nothing here changes a verdict" corrected: the two new ways an audit ends, and `verifier_checks` above one worker;
  - the fixtures sentence, now the true one (cab5 ran the real prover and Lean at `b82b94e92007`);
  - every run with its head;
  - sections "Ratchet gate" and "Not in this change".

  The title now says "each start admitted by a stated memory estimate" (finding 5).
- **The LeanPool hang.** A reading of free memory that raised inside a check's gate never gave up that check's turn, so every later check waited and the pool's shutdown never returned. What holds it now (`9a449491fbcd`): the wait is in a `try` that advances the turn, notifies the rest and raises again. The test is `test_a_reading_that_raises_in_a_checks_gate_is_its_outcome_and_the_next_check_takes_its_turn`, which uses no clock; without the `try` it hangs [ran: exit 124 under a 20 s shell timeout].
- **Ratchet gate**, run from a tree at `main` `a116c5614312`, exit 1. Its last line is `benchmarks/one_stage/tests/test_a5_negative_workers.py:316 'law' is retired; say sampling design (GLOSSARY.md)`. That is one site, an attribute read of `rows.Binding.law`, which `main` defines; it is listed in the body.
- **One instruction I did not follow as written.** Your brief asked the body to say the audit's own process keeps `main`'s hole. Since #9452 landed, that is false for the negatives the verifier decides. The body says what still stands as on `main`: `records_audit.py`'s `--forged` takes a member check with no verdict as Lean's refusal.

Least sure of: the hand resolution of `a5.py` has run only on fixtures, and no audit of a window has run at this head. Did not look at: the three conflicts in the restack result outside the PR, beyond why they arise. No timer and no subscription held. TIMER: none.

[2026-11-14T16:21Z] 3861666095e6
CLOCKGONE to network-accounting (lead), Sat 14 Nov 8:24 AM PST. This answers your 8:00 message; it replaces my 7:54 draft, which I can't confirm reached you.

Branch `cursor/na-clock-form-gone-b51f`, head `6bef2f142fa4abd35d5c6a3ff16bf47debb5914d` (pushed). The new commits over `030ba64ef533f8d5f9ff507cd951ae6bf5f43bb3` are:
- `6ca535fd98709f6120ebd82d7e4e3b60c8d5b540`: the twin;
- `2fd973c4334eb007e508a9e8a2bdca006d086ffe`: `clock_terms` and N2;
- `e2c0e592df624a95da00b4ae7ab1dbc1d83d8f8a`: `block_certifier` docstrings;
- `6bef2f142fa4…`: the docs.

**1. Runs (ran; read).** `r20261114-155118-f9ba` ran at `030ba64e…` and ended rc 0. In it, the verifier package's setup and build passed (159 jobs, no warnings). The tries on `NetworkCheck.lean`, `Clocks.lean` and `Period.lean` each passed, and each axiom probe reported "no theorems". `r20261114-155204-01f3` (the security Proofs build and tries) is waiting for a slot. `r20261114-155205-c551` (the records run) is running. I send `NetworkCheck.Period`'s new digest when c551 ends; I expect the dropped `let audits` to move that digest and nothing else (argued).

**2. Readers that stay (read; the build shows the Lean ones).**
- Lean readers of `NetworkCheck.Period`, all built:
  - `security/Properties/NetworkPeriod.lean`;
  - `security/Proofs/Properties/NetworkPeriod.lean`;
  - `security/Properties/NetworkScan.lean` l.4, 71, 82, and `NetworkScan/Flattens.lean` l.3;
  - `security/Proofs/Properties/NetworkScan.lean` l.84, 118, 139, and `NetworkScan/{Accept,Causal,Fits}.lean`;
  - `security/Properties/Verity/Game.lean` l.190;
  - `verity/protocols/compose/verifier/onsite/CoreOnsite.lean` l.35, 48, 64.
- Docstrings that name it: `security/Definitions/Pouw/OnsiteFreed.lean` l.15 and `Flock/WindowRecord.lean` l.13.
- Every honest witness that reaches it builds, and none can use the clock constructor, because it no longer exists. None moved.

**The piece that stops: `benchmarks/network_traces/chain_record.py`.**
- Its `period_binding` (about l.983) refuses any answer whose params lack `sync_set` ("a rows-form answer: …"). So `block_entry` binds only clock-form answers.
- `benchmarks/network_traces/tests/test_chain_record.py` l.365 and l.393 build their cases with `period.case("accept", 1|0, "clock")`. They now fail with a TypeError, since the twin has no form argument (argued, not run; b238 will show it).
- I left both untouched. Making `chain_record` bind a rows-form answer is a new binding, not a deletion. It is yours to rule on.

**3. Guarantees whose `reads` move (records run `r20261114-093213-8486`, at `00de0ee05b60…`, against main's records at `f28bdad2463…`).**
- Every kept guarantee's own record is byte-identical before and after: 1114 in `security` and 542 in `security/Properties`.
- Removed: 22 guarantees in `security` (the twelve counts and their ten witnesses) and `Properties.NetworkPeriod.clockKBoundsCount`.
- Module digests that moved (before → after):

  | Module | Before | After |
  |---|---|---|
  | `NetworkCheck.Period` | `506ace8a05e7c298e99f30ffc86cb4e74e26684f3ba4dfc8fe0a344c12c3133b` | `1f39dec10982839b527d16e0a5d06f28a8dbcf5734235131f993036b3fa2065b` |
  | `NetworkCheck.Clocks` | `7d8366ab7c72af1a13724cefdbaeaee3d2b7e3f140b2b1664c3ec4293e052598` | `0a011649aaa3c4fa17bb1e6b5527fdd09a825ceb60b3480bae7ffad36ad367c1` |
  | `NetworkCheck.Check` | `514de3f180defa15768790d890a0aa0230dda148d8d2b983fad772c208fe2e41` | `15a8833821ffb3407f91ddf43b7e66894079484cf81a08047bba27b096b1b5a7` |
  | `CoreProtocol.Profile` (security) | `27371a89…` | `b05ae84b…` |
  | `CoreProtocol.Profile` (Properties) | `e6b76f38…` | `467aa8dc…` |
  | `Properties.NetworkPeriod` | `21e5ca9e9d34eeea3af07c0c123b73f33c95558dbb6bbbf66d2f152b70558462` | `c2e96ae45e3595bd56599136fae5cba79ba1521623c11356758b347a1e5cd219` |

- Readers in `security`: `Flock.SecurityProofs.RecursiveAuditC3` and `Verity.Sampled.remote_soundness_deployedReady` (with `_honest` and `_nontrivial`).
- Readers in Properties:
  - `Properties.NetworkPeriod.rowsCapPublished` (with `_honest` and `_nontrivial`), `rowsKBoundsCount`, `rowsProgWired`, `rowsStateTied` and `rowsTiedCharged`;
  - `Verity.remote_soundness_of_subprotocols` (deployed), `coins_couple`, `soundness_at_of_componentBounds`, `soundness_at_of_targets`, `soundness_of_subprotocols`, `soundness_of_upFront`, `target_of_periods` and `wipe_target`, each with its witnesses.
- No PoUS guarantee and no other deployed guarantee moves.

**4.** #10088 has not landed. My frozen reasons that carry DEVICEGONE's text stay as they are, and the body will say this change lands after #10088.

**5. Done (written; the named test files ran locally, the suites are in runs).**
- **The twin** (`period_case.py`) is now rows-only. A params object with `sync_set` is refused first with "sync_set: the clock form is not one this check reads". `test_the_clock_form_is_refused` holds the twin and Lean (over the difftest port) to that one refusal. The old clock-form mirror test now runs on `rows-cert-forged`.
- **`clock_terms`:** `leak_bounds.py` drops it along with the N2 row, `--syncs` and `sync_bits`. `test_certifier_term.py` passed 34 tests locally.
- **The N2 row also reached the harness:** `verity_bench/exfiltration.py`'s `TOGETHER` lists it. I removed that entry and re-pinned `test_exfiltration.py`'s `NO_VERDICTS`; it passed 44 tests locally. The certifier_term, exfiltration and benchmarks READMEs lost their N2 lines.
- **Docs:**
  - `capacity.py`, `test_network_certifier_grid.py`, the verifier's `PROTOCOL.md` (the Capacity table, its records paragraph, the egress paragraph and Count) and `security/Proofs/NetworkCertifier/README.md` now name the kept lemmas (`NetTiming.card_free` and the rest) in place of the removed guarantees.
  - Rows and sentences for the per-block egress and wire-decoder bounds are gone, since nothing proves them now.
  - The README's counts now match the lock: 6 statements in `Guarantees.lean`, 16 frozen, 2 calibration and 14 others, and 9 witnesses.
  - The grid, capacity, `test_doc_paths` and `test_lean_vocabulary` tests passed 69 locally. `test_lean_vocabulary` warns on its "Record" bookkeeping, which I haven't checked against main.
  - `ARCHITECTURE.md` has no row for the two forms, at this head or at main `a116c5614312…` (grep for rows form, clock form and `sync_set`), so I touched nothing there.

**Left as they are, for your word:**
- `test_block_certifier.py`'s fixture `certifier-period-answers.json` holds a past clock-form run's answers. `block_leg` reads either form, and the fixture is counted in `tests/test_literal_addresses.py`.
- The N2 clock-sync pipeline (`replay.py`, `active_replay.py`, `network_e2e`, `network_sim`, and the N2 row in the `network_traces` README) is the certifier's, not the period check's.
- Lean's `Audit.bits` is still written (`Period.lean` l.600), but by grep nothing reads it now. The twin keeps it as a mirror.

**Runs open:**
- `r20261114-160439-b238`, at `6ca535fd…`, is waiting. It runs the network suite, `verity-core-composition` and `verity-network-traces`, and decides whether the twin and Lean agree, including on the clock case. I expect `test_chain_record` to fail there, at the reader named above.
- `r20261114-161905-78fe`, at `6bef2f14…`, is waiting. It runs `verity-bench`, `benchmarks/certifier_term`, the certifier verifier suite and `repository` (the root tests), and decides the N2 removal and the docs.

TIMER: none.

[2026-11-14T16:22Z] 8440f2c133d9
ONSITEMAIN2 to network-accounting (lead), Sat 14 Nov 8:24 AM PST. #9736: at a78100cf5cae5ecb83b8abed1ddc12a574257ada the Lean check r20261114-092257-75e4 exit 0, AUDIT: PASS in all four packages (certifier verifier, security, security/Properties, security/Proofs), and the suites r20261114-092344-dbb9 exit 0, 87 of 87 passed. But main a116c5614312 conflicts with it (git merge-tree: test_scan_zk.py), and the registration layout move 8469d5dcc164 lies between, so: restack.py --main 8469d5dcc164 (clean, 0d837cef07ab), then one merge of main. New head 29a70d9b1b72a83d56b26b2ad118551e56cb727f. By hand: test_scan_zk.py keeps our scan_ok ports with main's network_certifier key; import_order.toml, hardware/README.md and hardware/tests/test_package.py are main's (the move rerun met main's own lines, which we do not change). Tree vs main = our 24 files, same line counts; no Lean file of ours changed text, so no second Lean check. Suites at the new head r20261114-155150-3f1d running. Body updated, read back equal. PERIODPROOF2 told the new head.

[2026-11-14T16:22Z] a0e833cff83e
ONSITEMAIN2 to network-accounting (lead), Sat 14 Nov 8:25 AM PST. #9826: at 63f4cddfb058d9f7ca07b7cb24d2bd12f11cc24b the suites r20261114-093531-2a53 exit 0, 87 of 87 passed; the records r20261114-080923-9177 (at 18d369230828) exit 0, outcome regenerated, every failure a record difference, and its two analysis locks are 63f4cddfb058's byte for byte (sha256). It now takes #9736's new head: one merge, new head 484f76448fc77b699b2c33bf9453a913c3d548ca. By hand: test_certifier_piece.py's imports, our warden_serve and test_scan_zk with infra's CHALLENGER_HOST, LIVE_GATE line. The move's map names nothing in our own lines; tree vs main = our and #9736's files alone. Suites r20261114-155344-b224 waiting for a slot. Body updated, read back equal.

[2026-11-14T16:22Z] c03093b5a1ff
ONSITEMAIN2 to network-accounting (lead), Sat 14 Nov 8:26 AM PST. The stack holds main a116c5614312 by ci's recipe (restack.py --main e4a8682800d6, then ac8b3b51e490, then 8469d5dcc164, all 12 carries clean; then #9670 merges main and each above merges #9670's new head). Heads, suites not ended, so not for a mark yet: #9670 0b85bd55707c962c38a2b12e75a2645fcc16672b (suites r20261114-160141-6318; full Lean check r20261114-162021-e646, owed by the mend's two Lean files, certifier program+verifier, security, security/Properties, 120 GB); #9677 e2b24cd5f2ee5868731c3e4810474a34890b3cf0 (suites r20261114-161413-a450); #9678 9aafdbfcf8027388071ca033f1c4fcfac48db332 (suites r20261114-161445-5218; Lean check of the onsite package r20261114-162103-277d, since main moved under the Lean it imports); #9679 a982e353604c3d7be77189a1b4fd0ef46634a8f7 (suites r20261114-161517-63b1). By hand: public_boundary.toml deleted as main has it (nothing else names it); the retired-terms allowlist main's whole on all four (the reruns re-added entries main had dropped); #9677 certifier_piece.decide keeps our docstring plus main's import chain_record as CR, and test_certifier_piece.py the same import union as #9826. Each tree vs main = its own files plus #9670's. Bodies updated, read back equal.

