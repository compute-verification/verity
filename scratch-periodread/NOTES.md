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
