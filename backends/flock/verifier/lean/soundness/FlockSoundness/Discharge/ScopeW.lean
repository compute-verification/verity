import FlockSoundness.Discharge.ScopeW.Scope
import FlockSoundness.Discharge.ScopeW.RowEnc
import FlockSoundness.Discharge.ScopeW.Rows
import FlockSoundness.Discharge.ScopeW.Keyed
import FlockSoundness.Discharge.ScopeW.KeyH
import FlockSoundness.Discharge.ScopeW.Sites
import FlockSoundness.Discharge.ScopeW.Headline
import FlockSoundness.Discharge.ScopeW.HeadlineExec
import FlockSoundness.Discharge.ScopeW.HdecCor
import FlockSoundness.Discharge.ScopeW.Delta
import FlockSoundness.Discharge.ScopeW.CopiesAt
import FlockSoundness.Discharge.ScopeW.Copies
import FlockSoundness.Discharge.ScopeW.Chain
import FlockSoundness.Discharge.ScopeW.Commit
import FlockSoundness.Discharge.ScopeW.RowVal
import FlockSoundness.Discharge.ScopeW.ZkHidden
import FlockSoundness.Discharge.ScopeW.ZkHiddenSound
import FlockSoundness.Discharge.ScopeW.ZkHiddenJ
import FlockSoundness.Discharge.ScopeW.ZkAccepted
import FlockSoundness.Discharge.ScopeW.ZkComposed

/-!
# The headlines' scope widened to PoUW's rows: frozen statements

`Layout.scopeOk` admits only circuits whose every row is a `sha512/row/v1` row (`Layout.v1Ports`), and a manifest with
`row_schemas` makes every row a bit row, so no PoUW circuit is in scope of any headline that takes it. This area freezes
the widened scope and the headlines at it, and proves them: no `sorry` is left.

* `Scope`: `scopeOkW` (`Layout.scopeOk`'s circuit conditions, every row one of four shapes, `rowShapeW`),
  `scopeOkW_spec`, `scopeOkW_ports`, `scopeOkW_of_scopeOk`, and PoUW's rows in scope.
* `RowEnc`: the bytes each schema's digest hashes (`rowEncOf`: `rowEncV2`, `rowEncSeg`), their extractors
  (`rowPairOf`), and both at `keyProg`'s commit strings (`encW`, `pairW`), with `rowEncV2_inj` and `segPair_spec`.
* `Rows`: `HmRowsP`, rows each under its own leaf scheme, and `HmRowsP.toHmRows`, the same binding as an `HmRows`.
* `Keyed`: the public sites' rows under their own schemas (`keyedRowsW`, `keyedRowsOutsW`, `Layout.keyedRows`' copies
  and pins), and `hHm` at them (`keyedRowsOutsW_computes`).
* `KeyH`: the hidden sites' readers at output rows of runs (`outBitRowW`), `keyLayoutHW`, `keyRowsHW`, with their
  copies and pins (`keyProgHW_copies`, `keyProgHW_bcPinned`, `keyProgHW_valPinned`) and `hHm` (`keyRowsHW_computes`).
* `Sites`: `sitesAtHW`, with the unit slots' rows (`unitRowsFlatW`).
* `Headline`: `flock_headline_execW`, `Integrate.flock_headline_exec` at `scopeOkW`.
* `HeadlineExec`: `flock_headline_exec_hiddenW`, `HiddenExec.flock_headline_exec_hidden` at `scopeOkW`.
* `HdecCor`: `flock_headline_exec_decW`, `flock_headline_exec_hidden_decW`, `HdecCor`'s corollaries at `scopeOkW`.
* `Delta`: Δ at rows of any shape (`flatPairsW`, `delta_flat_eqW`), and what `HmRow.msgBit` and `HmRow.chainStart`
  give at each shape, with the final block in the `hm96` slot (`tailBit`).
* `CopiesAt`: a row's input positions (`inPos`), their cases, and the copies Δ's flat pairs hold at each.
* `Copies`: Δ's copies at the rows, once at `RowDelta` (setupH's, setupHidden's and `HmIn.delta`'s Δ) with the plain
  forms as corollaries.
* `Chain`: the bytes a row's digest hashes (`rowMsgOf`) and its commit string from its slots at a short and a segmented
  row (`chain_bits`, `sha_stepW`, `hm_commitW`, `port_commit_shortW`, `port_commit_segW`).
* `Commit`: a row's `b ‖ c` at every shape (`row_bc_flatW`, at `RowDelta`) and at the public and hidden sites
  (`bc_site_commitW`, `bc_site_commitHW`).
* `RowVal`: a row's value's encoding is its row's message (`rowEncOf_row`), from `rowPorts_bits`, `parse_bits` and
  `bitShaped_row`.
* `ZkHidden`, `ZkHiddenSound`, `ZkHiddenJ`: the hidden-output `--zk` headlines at `scopeOkW` (`zk_session_soundHW`,
  `zk_session_soundH_custodyW`, `zk_session_soundHJW`, `zk_session_soundHJ_custodyW`), their rows `rsZHW`, `rsZHJW`.
* `ZkAccepted`: `AcceptsZKAcceptedHW`, `ZkHidden.AcceptsZKAcceptedH` at the widened rows.
* `ZkComposed`: the public-output `--zk` headlines at `scopeOkW` (`zk_session_soundW`, `zk_session_sound_custodyW`),
  their rows `rsZAtW`.

Each headline is proved through the base headline's own chain, at `HmRowsP.toHmRows`.
-/
