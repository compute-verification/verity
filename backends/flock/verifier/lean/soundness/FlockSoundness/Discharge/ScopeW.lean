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
import FlockSoundness.Discharge.ScopeW.Copies
import FlockSoundness.Discharge.ScopeW.Chain
import FlockSoundness.Discharge.ScopeW.Commit
import FlockSoundness.Discharge.ScopeW.RowVal

/-!
# The headlines' scope widened to PoUW's rows: frozen statements

`Layout.scopeOk` admits only circuits whose every row is a `sha512/row/v1` row (`Layout.v1Ports`), and a manifest with
`row_schemas` makes every row a bit row, so no PoUW circuit is in scope of any headline that takes it. This area freezes
the widened scope and the headlines at it; the proofs left open are named obligations on `sorry` stubs, to be proved in
their own lanes before anything here reaches `main`.

* `Scope`: `scopeOkW` (`Layout.scopeOk`'s circuit conditions, every row one of four shapes, `rowShapeW`),
  `scopeOkW_spec`, `scopeOkW_ports`, `scopeOkW_of_scopeOk`, and PoUW's rows in scope.
* `RowEnc`: the bytes each schema's digest hashes (`rowEncOf`: `rowEncV2`, `rowEncSeg`), their extractors
  (`rowPairOf`), and both at `keyProg`'s commit strings (`encW`, `pairW`); open: `rowEncV2_inj`, `segPair_spec`.
* `Rows`: `HmRowsP`, rows each under its own leaf scheme, and `HmRowsP.toHmRows`, the same binding as an `HmRows`.
* `Keyed`: the public sites' rows under their own schemas (`keyedRowsW`, `keyedRowsOutsW`, `Layout.keyedRows`' copies
  and pins); open: `keyedRowsOutsW_computes`.
* `KeyH`: the hidden sites' readers at output rows of runs (`outBitRowW`), `keyLayoutHW`, `keyRowsHW`; open:
  `keyProgHW_copies`, `keyProgHW_bcPinned`, `keyProgHW_valPinned`, `keyRowsHW_computes`.
* `Sites`: `sitesAtHW`; open: `unitRowsFlatW`.
* `Headline`: `flock_headline_execW`, `Integrate.flock_headline_exec` at `scopeOkW`.
* `HeadlineExec`: `flock_headline_exec_hiddenW`, `HiddenExec.flock_headline_exec_hidden` at `scopeOkW`.
* `HdecCor`: `flock_headline_exec_decW`, `flock_headline_exec_hidden_decW`, `HdecCor`'s corollaries at `scopeOkW`.
* `Delta`: Δ at rows of any shape (`flatPairsW`; open: `delta_flat_eqW`), and what `HmRow.msgBit` and
  `HmRow.chainStart` give at each shape, with the final block in the `hm96` slot (`tailBit`; open: each shape's lemmas).
* `Copies`: Δ's copies at the rows, once at `RowDelta` (setupH's, setupHidden's and `HmIn.delta`'s Δ) with the plain
  forms as corollaries; open: `inBit_copyLW`, `row_copies_flatW` and its three shapes, `msg_copyW`, the instances.
* `Chain`: the bytes a row's digest hashes (`rowMsgOf`) and its commit string from its slots at a short and a segmented
  row; open: `chain_bits`, `sha_stepW`, `hm_commitW`, `port_commit_shortW`, `port_commit_segW`.
* `Commit`: a row's `b ‖ c` at every shape (`row_bc_flatW`, at `RowDelta`) and at the public and hidden sites
  (`bc_site_commitW`, `bc_site_commitHW`); open: `row_bc_flatW`, `outRowsS_of_checkRows`, `AcceptedH.row_bcW`.
* `RowVal`: a row's value's encoding is its row's message (`rowEncOf_row`); open: `rowPorts_bits`, `parse_bits`,
  `bitShaped_row`, `rowEncOf_row`.

Each headline is proved from the obligations through the base headline's own chain, at `HmRowsP.toHmRows`.
-/
