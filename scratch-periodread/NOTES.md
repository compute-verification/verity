# PERIODREAD scratch (PRF-600). Not for merging; no pull request opens this branch.

A resumed worker starts on a new machine: everything here is what I would otherwise lose.

## Stack

- PR 0 #9416 merged (`4175265019` on main).
- PR 1 #9537 `cursor/proofs-prf600-period-ledger-3b1f` (ledger's half, `Spot/Period.lean`). Records run `r20261112-231708-ddb2`.
- PR 2 `cursor/proofs-prf600-period-audit-9c4e` (`Spot/PeriodAudit.lean`), not opened. Body: `pr2-body.md` (has a `QUICK`
  placeholder; update for the list-match change, runs `e334` build+difftests and `a005` quick tier).
- PR 3 `cursor/proofs-prf600-period-statements-e2a7` (`security/Properties/Spot/Period.lean`, `PeriodLedger.lean`).
- PR 4 `cursor/proofs-prf600-period-proofs-4c9a` (`security/Proofs/Properties/Spot/Period.lean`, `Window.lean`'s
  `line_of_device`).
- #9546 `cursor/proofs-a5-case-retired-terms-6d1c`: the six retired terms of #9416. Quick tier `r20261112-235319-090e`.

## Launch words

    cd /agent/repos/wt-pr0   # detached origin/main, for `uv run -q research ...`
    uv run -q research run --on vy-nebius-2 --project verity --source WT --cwd source|clone --pool proofs --as proofs \
      --mem-gb N --timeout S -- bash -lc "$(cat SCRIPT)"
    verity lean try FILE --on vy-nebius-2 --pool proofs --as proofs      # from the worktree; ships uncommitted files too
    python3 tools/verity/lean/lean_changed.py --build FILE...            # inside a run: several files, one Lean slot

Launch inside tmux (`tmux -f /exec-daemon/tmux.portal.conf ...`). `research fetch RUN --all`, `research status RUN`.

## Lean facts found the hard way

- `read` is ambiguous with `MonadReader.read`: write `_root_.Spot.Period.read`.
- `open Spot` inside `namespace Properties.Spot.Period` resolves to `Properties.Spot`: write `_root_.Spot`.
- mvcgen and `split` cannot split a match on an array literal (`#[k]`): the code matches `xs.toList` instead.
- mvcgen stalls at `Array.mapM` without a spec: give it `ok_spec`-style `mapM_spec`.
- Unknown constants (probe `6f65`): `Array.partition_eq_filter_filter`, `List.Nodup.map_on`,
  `List.nodup_iff_injective_getElem`, `Array.foldlM_eq_foldlM_toList`, `Option.orElse_eq_none`.
- Core has no lemma about `Array.partition` (toolchain source, `Init/Data/Array`): prove
  `xs.partition p = (xs.filter p, xs.filter fun a => !p a)` by `Id.of_wp_run_eq` + `mvcgen invariants`.
- The toolchain's source is on the VM: `~/.elan/toolchains/leanprover--lean4---v4.34.0/src/lean` (read only; no
  Lean runs on the VM unless the coordinator says so).

## Proof plan for `security/Proofs/Properties/Spot/Period.lean`

Header: `module`; `public import Properties.Spot.Period`; `public import Proofs.Properties.Pouw.LiveSalt.Ledger`;
`@[expose] public section`; `set_option mvcgen.warning false`; `namespace Properties.Spot.Period`; open `Lean (Json)`,
`Std.Do`, `_root_.Spot _root_.Spot.Challenger _root_.Spot.Period`,
`Properties.Pouw.LiveSalt.Ledger (bind_ok of_triple throw_spec ok_spec checked_ok verifyLedger_inv LedgerInv)`.
Final theorems in `Properties.Spot.SecurityProofs`.

1. Helpers: `ite_some`, `none_of_forall₂`, `foldlM_prefix` (list fold keeps `P` of the prefix), `pairwise_of_distinct`,
   `pairwise_filterMap_of`, `nodup_split` (filter and its complement, mapped, are Nodup together), `list_mapM_ok_getElem`
   and `array_mapM_ok_getElem` (copied shape from `Proofs/Flock/Soundness/Discharge/Exec/Setup.lean`), `partition_eq`.
2. `sessionAt p ledger periods stop : Win → Option Session` (the code's lambda); `read_ok`: `checked … = .ok (r.wins,
   r.periods)`, `reaches … p.stop = some stop`, `r.sessions = r.wins.filterMap (sessionAt …)`; `sessionAt_some`.
3. `wins_distinct` from `checked_ok` + `verifyLedger_inv` + `LedgerInv.distinct`.
4. `windowsIn_ok` by mvcgen with `mapM_spec`: others, nonempty, `∃ g, mapM g … = .ok ws.windows ∧ …`, the four
   `find?`/`findSome?` = `none` facts; then the semantic lemmas.
5. `ledgerMeans : LedgerMeans`.
6. `caseAt_ok`, `windowAt_ok` (accept / undecided), `run_ok`, `checkWith_accept`, the fold invariant
   (`st.undecided = none → Good pre st`), `acceptMeans : AcceptMeans`, `committedOnce : CommittedOnce`.
7. A24: `SeqInv`/`verifyLedger_seq` after `verifyLedger_sig`; `LedgerLinesOfDevice` from `line_of_device` per line;
   `LedgerOfDevice` from it plus `Numbered` and `entry_fields_equiv` on `"seq"`.

## State at Fri 01:02Z (for whoever resumes on a new machine)

- PR 4's branch `cursor/proofs-prf600-period-proofs-4c9a` holds both proofs files, written blind, never compiled:
  `security/Proofs/Properties/Spot/Period.lean` (`ledgerMeans`, `acceptMeans`, `committedOnce`) and
  `security/Proofs/Properties/Spot/PeriodLedger.lean` (`ledgerLinesOfDevice`, `ledgerOfDevice`), both roots in
  `security/Proofs/lakefile.toml`; `Window.lean` has `line_of_device` carved out and `sig_lms` added.
- The compile loop: `run-pr4-try.sh` in a `research run --cwd clone --source <PR 4 worktree> --pool proofs --as proofs
  --mem-gb 16 --timeout 3600`. `--try` prints every error of the file; `--build` only each file's first.
- Every mvcgen closer ends in `fail "<name>: a goal mvcgen left is not closed"`, which prints the goal and its
  hypotheses: read those to write the closer.
- Known risks, in the order I expect them: `partition_toList` (invariant syntax and `simp_all`; fallback is two
  `filter`s in `windowsIn`, a PR 2 change: say before doing it); `read_ok`'s `rfl` (the copy of `read`'s lambda in
  `sessionAt`); the leaf closers of `windowsIn_ok`, `caseAt_ok`, `windowAt_raw`, `run_ok`; `checked_lines`'s invariant
  (copied from `checked_ok`); `recordOfJson_seq`'s `by assumption` for `getD j "seq" = .num n`.
- Machines: vy-nebius-2 and vy-nebius-cpu-11 were both full at 01:00Z (cpu-11: free slots blocked by memory
  reserved by live runs). Coordinator's rule: stay on vy-nebius-2; after 15 min in line, cancel and resend to cpu-11;
  never `--on auto`, never cpu-13.
- PR 2 (`…-period-audit-9c4e` at `5dacc95eab`): build + 24 difftests pass (`e334`); opens when its quick tier
  (`a005`) passes; then a records run.
- #9537 waits on its records run `ddb2`; #9546 on its quick tier `090e`.

## State at Fri 02:00Z

- PR 1 #9537 at `44b84839217d`: READY line given 01:45Z (records run `ddb2` pass, `regenerated` empty).
- #9546 at `ee245a981c3e`: READY line given 01:28Z.
- PR 2 `cursor/proofs-prf600-period-audit-9c4e` at `5dacc95eab`: quick tier `r20261112-232219-a005` still running
  (slot since 00:56:36Z). Open it when `SUITES exit 0`; body in /tmp is lost on a new machine: rewrite from the
  README section and PR 1's body shape.
- PR 3 `cursor/proofs-prf600-period-statements-e2a7` at `06ac8f46e7`: statements + lock (roots, layer for
  PeriodLedger, five `unstated` lines under `shape.frozen`). Not opened. Needs its own records run
  (`--records security/Properties --update`) after `d333` shows the lock edits are right.
- PR 4 `cursor/proofs-prf600-period-proofs-4c9a` at `36dc4d568ff4`: all three proof files elaborate, three axioms
  only (`3281`: Period.lean 47/47; `b335`: Lines.lean 12/12, PeriodLedger.lean 4/4). Locks: Proofs roots +3,
  frozen lines name the theorems. Records run `r20261113-015344-d333` (vy-nebius-2, launched 01:53:44Z,
  `--records verity/protocols/spot/verifier/lean --update`, mem 32, timeout 10800): expect ~04:00Z.
  Read its four `audit.json` (`regenerated`, `failed`) and stdout for `shape.`/`layers.`/`orphan.` failures.
- Lock facts learned (check.py): `shape.unstated` (3462) fails for a `def X : Prop` in the properties tree that no
  guarantee states and `shape.frozen` lacks; a guarantee with `shape.properties` {function, outcome: accept} is in
  `witness_scope` (2006) and needs `witnesses` honest + nontrivial (`unwitnessed` adds none); `Properties` and
  `Proofs.Properties` modules are seeds for the unread ratchet (unread.py SEEDED_PREFIXES), so no cap entry.
  `layers.import` (1511): a module's import inside the package must be under its layer's `imports`.
- Asked the coordinator (01:55Z entry): unstated lines vs guarantee entries with witnesses; whether
  `LedgerOfDevice`'s unstated line naming `Numbered` is within "no lock entry reads Numbered".
- Mailbox last read at `23c3e423be` (01:55Z).

## State at Fri 02:20Z (read this first on a new machine)

Heads (all pushed):
- PR 1 #9537 `44b84839217d`: marked ready by the coordinator 01:57Z.
- #9546 `ee245a981c3e`: ready line given 01:28Z.
- PR 2 `cursor/proofs-prf600-period-audit-9c4e` at `5dacc95eab9e`: NOT opened. Opens when its quick tier
  `r20261112-232219-a005` shows `SUITES exit 0` (script's rc is always 0; read stdout; timeout 03:26Z). Body:
  `pr2-body.md` here, placeholders `RECORDS` and `QUICK`. Per the lean-proofs skill it needs no records run of its own
  (no hand edit to a lock); name the records that regenerate from `d333`'s `regenerated` lists.
  Open: `uv run -q research queue open --head cursor/proofs-prf600-period-audit-9c4e --issue PRF-600 --title T
  --body-file F`.
- PR 3 = #9581 `cursor/proofs-prf600-period-statements-e2a7` at `43fde9d04680`: statements, lock (roots, layer,
  five `unstated` lines), the executable `sampled-proofs-period` (+ `exempt` line, README, difftest asserts), the premise
  paragraph in `IntegrityPeriod`'s docstring, SPOT in titles. Body `pr3-body.md` (as posted). Exe run `0c67` rc 0
  (24 passed). Quick tier `r20261113-021125-efae` launched 02:11Z. Needs its own records run (hand edits to locks):
  launch when `d333` ends (one tree at a time), from a worktree at its head:
  `uv run -q research run --on vy-nebius-2 --project verity --source WT --cwd clone --pool proofs --as proofs
  --mem-gb 32 --timeout 10800 --declared-output 'lean-audit/*' -- python3 tools/verity/lean/lean_changed.py
  --records verity/protocols/spot/verifier/lean --update`.
  Then: edit the body's Checks (`research queue edit 9581 --body-file F`), ready line.
- PR 4 `cursor/proofs-prf600-period-proofs-4c9a` at `41cbecec62af` (PR 3's head merged, title says SPOT): NOT opened.
  Body draft `pr4-body.md` (placeholders `RECORDS`, `CHECKS`). `d333` runs its tree BEFORE the merge
  (`36dc4d568f`): it shows whether the lock edits pass and which records regenerate. Final records run on the final
  head after PR 3's. Quick tier too. Second reader on the statements: STOPS (coordinator assigned 02:00Z).
- Scratch: this branch.

Coordinator's words since 02:00Z:
- 02:00Z: `lean` on the VM is allowed on one scratch file importing core only, under a minute and 1 GB.
- 02:13Z: the lock default stands: five `unstated` lines, no guarantee entry in PR 3; `LedgerOfDevice`'s line may name
  `Numbered`; no table or page cites the five as guarantees; witnesses only if he asks on Friday.
- Open ask of mine (02:12Z entry): rename the executable to `spot-period`? Default: keep `sampled-proofs-period`.

Mailbox last read at `bea8cf9117` (02:11Z), plus the hook's 02:13Z word.
Entries: last `9be616f60c` (02:12Z).

## State at Fri 02:52Z (read this first on a new machine)

- PR 2 = #9595 (unmarked) at `5dacc95eab9e`. Quick tier `a005`: 60 of 61; `research` red = the node's hold file
  (`/etc/vy/auto-hold`, `auto.HOLD_FILE`) read by two tests of `tools/research/tests/test_auto.py`; red on main's tree
  too (`b5e6`), on PR 2's (`7179`). Body `pr2-body.md`; its Records section says two records, the run says four in
  `security` (`RecursiveAuditC3`, `remote_soundness_deployedReady` and its `_honest`, `_nontrivial`): correct it when
  `d333` ends, then give #9595's ready line.
- Fix = #9596 (unmarked) `cursor/proofs-auto-hold-tests-7d2e` at `7764f46a6233`; `2380`: 76 passed on the node. Whole
  `research` suite at that head: `r20261113-024735-2bee` (script `run-fix-suite.sh`). Ready line after it.
- PR 3 = #9581 at `43fde9d04680`; quick tier `efae` running (research red there too). Records run NOT started.
- Coordinator's rulings of 02:43Z on STOPS's second reading (`stops-0222-second-reading.md`), placed in my 02:50Z
  entry (`d5f9b8e69b`):
  1. corollary by name (accept ∧ `pub.others = #[]` → every committed root is one published window's, audited
     `AuditedAt`); `CommittedOnce`'s title and the opening sentences reworded ("one audit entry", "every root on a
     session with the sample's family"). Statement PR 3, proof PR 4.
  2. decoder lemma: `AuditParams.of j = .ok p` → `p.keyJ` reads as `p.ledger.key`, `p.audit` names none of the five
     keys. Statement PR 3, proof PR 4.
  3. lemma `au.session = w.session` (from `checked` refusing a second open of a root). Statement PR 3, proof PR 4.
  4. (h) answered in the 02:50Z entry.
  5. (i) (j) (b) (c) (d) (e): one sentence each in "what this does not say", STOPS's words. PR 3.
  6. (l): not PR 3's; size it (lines' whole texts among the strings the finder hashes; the last line's signature
     stays free).
  7. (o): weaken `Numbered` to "a line of the device's that is an entry numbered i is at place i". PR 3 + PR 4 proof.
  8. remove `periods` from `CommittedOnce`. PR 3 + PR 4.
  A23's day: firm word due Fri 12:00Z.
- After the statement edits: one try of the two Properties files and PR 4's two proof files; then PR 3's records run
  (after `d333`), then PR 4's.

## State at Fri 03:46Z (read this first on a new machine)

Governing words: coordinator 03:00Z (`1616d4b1a91a`: the refusal goes into #9595; PR 3 stated once against it) and
03:35Z (`8401237762b7`: leave `auditedOpen` word for word as `main`, `auditedIn` its own copy of the 24 lines, tied by
a lemma proved by `rfl`; (i) to (iv) stand; wording for soundness under `To post` once the refusal has a run).

Heads:
- #9595 `cursor/proofs-prf600-period-audit-9c4e` at `41a71836c3`: the refusal (`registeredIn`, parameter
  `registration`), `auditedOpen` as `main`, `auditedIn` + `auditedOpen_eq_auditedIn := rfl`. Nothing under
  `security/` differs from PR 1's head. Body NOT updated (`/tmp/periodread/pr2-body.md` is gone on a new machine:
  rewrite from `gh pr view 9595 --json body`).
- #9581 `cursor/proofs-prf600-period-statements-e2a7` at `a441d99476`: `CommittedOnce` restated, `EveryRootAudited`
  gone, `OthersElsewhere` added (my addition, the coordinator has not ruled on it). Not compiled.
- PR 4 `cursor/proofs-prf600-period-proofs-4c9a` at `4ee128adc1`: proofs redone, not compiled. Lock lines are still
  PR 3's (`unstated … it takes no premise`): once a try passes, write "proved with no premise in security_proofs by
  Properties.Spot.Period.<name>" on each (LedgerOfDevice: "proved … from Numbered").
- #9596 marked at `7764f46a62`. #9537 marked at `44b84839217d`, still open. #9546 at `ee245a981c`.

Runs:
- `r20261113-031857-3061` done rc 0 at `4f6a8f3d81` (the refusal, with the split): build ok, 28 passed.
- `r20261113-034210-2101` at `41a71836c3` on vy-nebius-2 (build Spot + difftest exe, both period difftests): waiting.
- `r20261113-034343-e626` on vy-nebius-cpu-11: try of PR 4's `security/Proofs/Properties/Spot/Period.lean` at
  `4ee128adc1` (fetch with `--on vy-nebius-cpu-11`). Then try `PeriodLedger.lean` the same way.
- Cancelled: `d333` (53 records moved with the split: 4 in `security`, 49 in `security/Properties`), `efae`, `cd66`.

Next: read 2101 and e626; fix; #9595's body and ready line (quick tier first: `run-pr2-quick.sh` = suites.py
--quick --fresh over the changed suites); PR 3's body; PR 4 lock lines, records, open PR 4; R8 sentence PR (#9375
merged 03:01Z); (l) sizing; A23 word by Fri 12:00Z.

Scratch facts: on core-only stand-ins `registeredIn_ok` and the `rfl` tie both elaborate (`Reg2.lean`, `Tie.lean`
under `/tmp/periodread/scratch`, gone on a new machine); `match` on `(xs.find? …).map (·.circuit)` with
`some (.str d)` does not elaborate, `.bind fun mp => match mp.circuit with …` does.

## State at Fri 04:10Z (read this first on a new machine)

Governing words unchanged (03:00Z, 03:35Z). Mailbox last read at `4adc7e8915`.

Heads:
- #9595 at `41a71836c3` (body in `pr2-body.md` here, `CHECKS` placeholder still to fill; not posted).
- #9581 at `a441d99476` (body in `pr3-body.md` here, `CHECKS` placeholder; not posted).
- PR 4 = #9625 `cursor/proofs-prf600-period-proofs-4c9a` at `73ab10a29a`, opened unmarked 03:59Z (body `pr4-body.md`).
- R8 sentence: `cursor/proofs-r8-rows-sentence-5d1c` at `73a3f2fea2` (from main `8ef93f0196`), pushed, NO PR yet.
  The coordinator's word was "before you push: say whether the edit moves any lock record": I pushed first. No
  record moves [derived: the lock's read hash is `type.hash ++ valueHash`, Facts.lean:1106; the lock holds no line
  numbers]. Needs a try of `security/Properties/Spot.lean`, then `research queue open` and a ready line.
- A23: `CommittedOnce` elaborates on the three axioms, run `r20261113-035213-962c` (cpu-11), entry 03:57Z.

Runs on vy-nebius-2:
- `r20261113-034210-2101` (#9595 difftests at `41a71836c3`): took a slot 04:04:18Z after 21m41s.
- `r20261113-040405-1199` (#9595 quick tier).
- PR 3's exe build + difftests at `a441d99476`: launched 04:08Z in tmux `pr3-exe` (id in `/tmp/periodread/pr3-exe-a441.log`).

(l) sized (in the 04:1xZ entry): a third ledger statement, every presented line but the last is the device's whole
line; needs `Chained` (the device's `prev` hashes its own line before) beside `Numbered`; the finder must also hash
the lines' whole texts. Changing `readHashed`/`deviceHashed` moves the records of what reads `findBreak`
(`security/lean-audit.json` reads it), so: a second finder in the period's own file, tied to `Break` by a lemma.
Reuses `hexOf_inj` (Proofs/Flock/Bytes.lean:161), `collisionIn_ne_none`, `break_of_forgery`
(Proofs/Properties/Challenger/Break.lean), and `Lines.lean`'s `verifyLedger_seq` pattern for `prev`.

Next: read 2101, 1199, PR 3's run; fill the bodies; ready lines #9595 then #9581; try R8, open, ready line;
#9625's quick tier and Lean check as the coordinator says.

## State at Fri 05:02Z (read this first on a new machine)

Heads (all pushed):

- #9595 `cursor/proofs-prf600-period-audit-9c4e` `41a71836c34823cc2f7dcc54396ed776c7c8c5a9`. Build and difftests: `r20261113-034210-2101` rc 0. Quick tier `r20261113-040405-1199` on vy-nebius-2, in its slot since 04:31:00Z. Ready line when it ends.
- #9581 `cursor/proofs-prf600-period-statements-e2a7` `e230e76f63423740f73b9e69bff0cfe6a32e3376` (`a441d99476` plus the docstring sentence of the 04:50Z word). `87ae` rc 0 at `a441d99476`. Ready line after #9595's.
- #9625 `cursor/proofs-prf600-period-proofs-4c9a` `2f3c3e6022e189a2f94be90f1d3d067eda6995d6` (`73ab10a29a` plus the merge plus the restored lock clause). `Drawn` try `r20261113-043328-5c31` rc 0 at `73ab10a29a`. Ready line after #9581's.
- #9637 `cursor/proofs-r8-rows-sentence-5d1c` `73a3f2fea226cb0ddff70fc2a687678bfcf1ce8e`. Ready line written 04:34Z.
- (l) `cursor/proofs-prf600-period-signed-7d2e` `ade5505e0edd4a46ae8729cabcc924f1ae623e26`, worktree `wt-l`, not compiled. First try `r20261113-050027-499f` on vy-nebius-2, launched 05:00:25Z. Body in `l-body.md` (has a `CHECKS` placeholder). No pull request yet: open it unmarked once the file elaborates on the three axioms, with `research queue open --head cursor/proofs-prf600-period-signed-7d2e --issue PRF-600`.

Words since 04:10Z:

- 04:24Z: (l) YES as sized (`Chained` an explicit hypothesis, own finder, `findBreak` and `readHashed` untouched, a pull request of its own on #9625); `OthersElsewhere` stays; `Drawn` built before #9625's ready line.
- 04:50Z: the device's ledger facts stay hypotheses until Tue 17 Nov; no law into `History`; the docstrings say "a lemma of the modelled device once the model builds that device's whole ledger". The tree leaves out the issue id and the lane's name; the bodies name PRF-490. No `unwitnessed` lock entry without asking.

Things that bit:

- `verity lean try`'s local waiter leaves with exit 79 after about 1200 s while the run goes on: follow with `research fetch RUN --all` and read `status.json`.
- Merging up the stack, the lock's merge driver takes theirs on an `unstated` line both sides changed: diff the lock against the old head after every merge.
- One try at a time. A try that waits 15 minutes on vy-nebius-2 is cancelled and sent to vy-nebius-cpu-11.

## State at Fri 05:30Z (read this first on a new machine)

Only one run is open: `r20261113-040405-1199`, #9595's quick tier on vy-nebius-2, in its slot since 04:31:00Z, timeout 06:01:00Z. At 05:28Z: 58 of 61 suites written, 57 passed, 1 failed (`research`: "4769 passed, 2 failed", `tests/test_auto.py:616` `KeyError: 'error'`, the base's red that #9596 fixes). Left: `verity-flock`, `verity/protocols/pouw/verifier`, `verity/protocols/recursion/verifier/vstar_cli`. Expected end about 05:45Z.

On a new machine: `uv run -q research fetch r20261113-040405-1199 --all --on vy-nebius-2` from a checkout, then read `~/.research/runs/r20261113-040405-1199/out/suites-quick.txt` and `status.json`.

When it ends:

1. Name the two failing `research` tests; say they are the base's and #9596's. Add the result to #9595's body (`pr2-body.md`, then `research queue edit 9595 --body-file`).
2. One entry whose first content is the ready lines, in order: #9595 `41a71836c34823cc2f7dcc54396ed776c7c8c5a9`; #9581 `e230e76f63423740f73b9e69bff0cfe6a32e3376`; #9625 `2f3c3e6022e189a2f94be90f1d3d067eda6995d6`; #9665 `5edf7e30ee80bb013e16e4fe8386ef6332aee99e`. #9637's is written (04:34Z).
3. Ask the coordinator who starts the kernel replay (a full Lean check) on the proofs' heads and who reads the statements as a second reader.

(l) is done as far as a try goes: `r20261113-051324-3c1a` rc 0 at `5edf7e30ee`, 12 theorems on the three axioms; #9665 open, unmarked. The device's chain is `verity/protocols/spot/device/challenger.py:1502` with `prev_of` in `verity/hardware/record.py:71`.
