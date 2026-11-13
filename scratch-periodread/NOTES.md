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
