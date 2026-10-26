import Proofs.Flock.Claims.KS.PlainDrawRColsE
import Proofs.Flock.Claims.KS.PlainKSLin
import Proofs.Flock.Claims.Recursion.FlockKSDeployE

/-!
# The plain record at the deployed reader `VR rd`, and its code facts

`deployPlainR rel rd` is the plain record with the deployed reader `VR rd` (`nextR rd` at `V false`'s rounds) as its
verifier and the reader's columns on a view (`colsRV rd`) as its opened columns; `deployPlainTrimR rel rd` is it as the
claims read it, its columns and code trimmed of trailing zeros. The bound record's element-level record at
`nextE := nextR rd`, `colsE := (deployPlainTrimR rel rd).cols` and `wE := (deployPlainTrimR rel rd).weight` is it
(`RecDeploy.deE_R_trim`, as `deE_VE_trim`).

Its code facts, each from `VR rd`'s decision at SHA-512 (`decideR_sha512`), by `VE`'s proofs (`FlockKSDeployE`):

* it accepts only served tables in scope (`nextR_acc_check`, `accR_inScope`), so at one table (`plainR_hone`) and
  with `k ≥ 8` (`kR_range`, `plainR_kSplits`);
* an accepted run's statement is its last message's (`accR_last`), so the reader's columns are table 0's of the
  statement and have the zero column's width (`accR_cols`, `plainR_colsOk`);
* `plainR_KSFlockLin`: `KSFlockLin` at the trimmed record and `CpDep`, from conflicts tied to collisions and the table
  bound at the untrimmed record (`plainE_KSFlockLin`'s assembly).
-/

namespace FlockDeploy

open Flock

local notation "𝔽" => FlockLevel3.GF128

/-- **The plain record at the deployed reader**: `deployPlain rel` with `VR rd` as its verifier and the reader's
columns on a view (`colsRV rd`) as its opened columns. -/
noncomputable def deployPlainR (rel : Readable.Parsed → Readable.Relation) (rd : Claims.Msg → Option FinalP) :
    Claims.Flock.Deploy Coin (List 𝔽) Claims.Flock.PlainTrace :=
  { deployPlain rel with V := VR rd, cols := colsRV rd }

/-- **The plain record at the deployed reader, as the claims read it**: `deployPlainR rel rd` with its opened columns
and its code trimmed of trailing zeros. -/
noncomputable def deployPlainTrimR (rel : Readable.Parsed → Readable.Relation) (rd : Claims.Msg → Option FinalP) :
    Claims.Flock.Deploy Coin (List 𝔽) Claims.Flock.PlainTrace :=
  (deployPlainR rel rd).mapCols trimCol

end FlockDeploy

namespace RecDeploy

/-- **The element-level record at the deployed reader is the trimmed plain record there.** -/
theorem deE_R_trim (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (rd : Msg → Option FlockDeploy.FinalP) :
    deE rel (FlockDeploy.nextR rd) (FlockDeploy.deployPlainTrimR rel rd).cols
        (FlockDeploy.deployPlainTrimR rel rd).weight = FlockDeploy.deployPlainTrimR rel rd :=
  rfl

end RecDeploy

namespace FlockDeploy.KS

open Flock FlockSoundness.Discharge.Exec
open FlockSoundness.Refine (ebind_ok)
open Claims.Bridge.Custody (checksAtJ_ok checksAtJ_scope)
open Claims.Bridge.PlainDraw (decideR_sha512)
open Claims.Recursion

local notation "𝔽" => FlockLevel3.GF128

/-! ## Scope -/

/-- **An accepting step of `VR rd` at SHA-512** is at a statement that sets up, serves only `fast100`'s
`22 ≤ m ≤ 35` and passed the scope check (`decideR_sha512`, `verifyJWithMK_tables_cap`, `checksAtJ_scope`). -/
theorem nextR_acc_check {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    (h : ((nextR rd x as cs).eval sha512).1 = .acc) :
    ∃ ins spec sts, setupOf false x = some (ins, spec, sts) ∧ 0 < sts.size ∧
      (∀ (i : ℕ) (hi : i < sts.size), 22 ≤ sts[i].m ∧ sts[i].m ≤ 35) ∧ ins.read.check = .ok () := by
  cases ha : actR rd x as cs with
  | rej => revert h; simp [nextR, ha, Claims.OC.eval]
  | coin c => revert h; simp [nextR, ha, Claims.OC.eval]
  | decide s spec sts link c₁ mids f =>
    obtain ⟨hv, hc, -, -, -⟩ := decideR_sha512 ha h
    obtain ⟨_, schs, -, -, hpos, -, hall⟩ := verifyJWithMK_tables_cap id With.std hv
    have hsz : (tabsOf sts).size = sts.size := Array.size_map ..
    refine ⟨s, spec, sts, actR_setupOf ha, hsz ▸ hpos, fun i hi => ?_, checksAtJ_scope hc⟩
    obtain ⟨_, _, hf, -⟩ := hall i (hsz ▸ hi)
    have hm : (tabsOf sts)[i].m = sts[i].m := by simp [tabsOf, Setup.ofCircuit]
    exact hm ▸ fast100_range hf

theorem accR_check {rd : Claims.Msg → Option FinalP} {x : Claims.Msg}
    (hacc : ¬ Claims.Flock.NoAcc sha512 (VR rd) x) :
    ∃ ins spec sts, setupOf false x = some (ins, spec, sts) ∧ 0 < sts.size ∧
      (∀ (i : ℕ) (hi : i < sts.size), 22 ≤ sts[i].m ∧ sts[i].m ≤ 35) ∧ ins.read.check = .ok () := by
  unfold Claims.Flock.NoAcc at hacc
  push Not at hacc
  obtain ⟨Ω, P, ω, r, hr⟩ := hacc
  obtain ⟨as, has⟩ := FlockDeploy.KS.stepWith_acc _ _ _ _ _ _ _ _ _ hr
  exact nextR_acc_check has

/-- **`VR rd` accepts only in scope**, at SHA-512: a statement some run accepts parses, in scope (`accE_inScope`'s
proof, at `nextR_acc_check`). -/
theorem accR_inScope (rel : Readable.Parsed → Readable.Relation) {rd : Claims.Msg → Option FinalP} {x : Claims.Msg}
    (hacc : ¬ Claims.Flock.NoAcc sha512 (VR rd) x) :
    ∃ s, parse false rel x = some s ∧ Claims.Flock.InScope false s := by
  obtain ⟨ins, spec, sts, hset, h0, hm, hc⟩ := accR_check hacc
  obtain ⟨hi, hs⟩ := setupOf_some hset
  have hr : readOf false x = some ins.read := by unfold readOf; rw [hi]; rfl
  have hzk : ins.read.zk = false := readOf_zk hr
  have hrp := read_parse_of_setup hs
  have hparse : parse false rel x = some (Readable.toReadable rel { ins.read with sts }) := by
    unfold parse Readable.parseOf Readable.parseBytes
    rw [hr, Option.bind_some, hrp, Option.map_some]
  refine ⟨_, hparse, ?_⟩
  have hin := (Readable.inScope_iff rel (Readable.parses_of_parse hrp)).2 ⟨hc, ?_⟩
  · exact hzk ▸ hin
  · simpa only [Readable.toReadable, Array.getElem?_eq_getElem h0, Option.map_some, Option.getD_some] using
      (hm 0 h0).1

/-- **`VR rd` accepts only at one table**, at SHA-512: the plain scope admits one (`accR_inScope`). -/
theorem plainR_hone (rel : Readable.Parsed → Readable.Relation) (rd : Claims.Msg → Option FinalP) (x : Claims.Msg) :
    (∃ s, (deployPlainR rel rd).parse x = some s ∧ s.tables = 1) ∨
      Claims.Flock.NoAcc sha512 (deployPlainR rel rd).V x := by
  by_cases hno : Claims.Flock.NoAcc sha512 (VR rd) x
  · exact .inr hno
  · obtain ⟨s, hs, hin⟩ := accR_inScope rel hno
    obtain ⟨-, -, -, -, -, -, -, -, ht, -⟩ := hin
    exact .inl ⟨s, hs, (Claims.Flock.Stmt.tables_eq_one_iff s).2 (by simpa using ht)⟩

/-- **`VR rd`'s `k` range**: at every statement, `k ≥ 8` or `VR rd` accepts no run at SHA-512 (`kE_range`'s proof). -/
theorem kR_range (rel : Readable.Parsed → Readable.Relation) (rd : Claims.Msg → Option FinalP) (x : Claims.Msg) :
    8 ≤ (deployPlainR rel rd).k x ∨ Claims.Flock.NoAcc sha512 (deployPlainR rel rd).V x := by
  by_cases hacc : Claims.Flock.NoAcc sha512 (VR rd) x
  · exact .inr hacc
  obtain ⟨s, spec, sts, hs, h0, hm, -⟩ := accR_check hacc
  obtain ⟨h22, h35⟩ := hm 0 h0
  have hd := (Claims.Flock.plainShape_table ⟨sts[0].m - 22, by omega⟩)
  have hi : Claims.Flock.PlainIdx.m ⟨sts[0].m - 22, by omega⟩ = sts[0].m := by
    simp only [Claims.Flock.PlainIdx.m]
    omega
  have hlen : 10 ≤ (Claims.Flock.plainLevel0 sts[0].m).d := by
    unfold Claims.Flock.plainLen Claims.Flock.plainCols at hd
    rw [hi] at hd
    omega
  refine .inl (eight_le_k ?_)
  show 2 ^ 10 ≤ N₀ false x
  exact le_trans (two_pow_le hlen) (N₀_ge_of_setup hs h0)

/-- **The trimmed record at `VR rd`'s `k` range, with the relation split**, where the statement has one table wherever
`VR rd` accepts some run (`plainR_hone`). -/
theorem plainR_kSplits (rel : Readable.Parsed → Readable.Relation) (rd : Claims.Msg → Option FinalP)
    (x : Claims.Msg) :
    (8 ≤ (deployPlainTrimR rel rd).k x ∧ Claims.Flock.Splits (deployPlainTrimR rel rd) x) ∨
      Claims.Flock.NoAcc sha512 (deployPlainTrimR rel rd).V x :=
  (kR_range rel rd x).elim
    (fun hk => ((plainR_hone rel rd x).imp_left fun ⟨_, hs, h1⟩ =>
      splits_of_one (deployPlainTrimR rel rd) rfl rfl hs h1).imp (fun h => ⟨hk, h⟩) id) .inr

/-! ## Columns -/

/-- **An accepted run of `VR rd` ends on a message the reader reads at the statement**, at SHA-512: `rd` reads its last
message, on the run's own statement (`decR_rd`, `checksAtJ_ok`). -/
theorem accR_last {rd : Claims.Msg → Option FinalP} {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω}
    {r : Fin ((VR rd).k x) → Coin} (h : (Claims.run sha512 P (VR rd) x ω r).out = .acc) :
    ∃ fl, (((Claims.run sha512 P (VR rd) x ω r).msgs.map Prod.fst).getLast?).bind rd = some fl ∧ fl.stmt = x := by
  have ha := FlockDeploy.KS.run_acc_next sha512 P (VR rd) x ω r h
  generalize (List.range _).map (Claims.coinSeq r) = cs at ha
  replace ha : ((nextR rd x ((Claims.run sha512 P (VR rd) x ω r).msgs.map Prod.fst) cs).eval sha512).1 = .acc := ha
  cases hact : actR rd x ((Claims.run sha512 P (VR rd) x ω r).msgs.map Prod.fst) cs with
  | rej => revert ha; simp [nextR, hact, Claims.OC.eval]
  | coin c => revert ha; simp [nextR, hact, Claims.OC.eval]
  | decide s spec sts link c₁ mids f =>
    obtain ⟨-, hc, -⟩ := decideR_sha512 hact ha
    obtain ⟨fl, hfl, hfs⟩ := decR_rd (actR_decide hact)
    exact ⟨fl, hfl, hfs.symm.trans (checksAtJ_ok hc).1⟩

/-- **`runOfDecR`'s run is at its table's `m`.** -/
theorem runOfDecR_m {d : DecR} {j : ℕ} {rr : Claims.Flock.ZkRun} (h : runOfDecR d j = some rr) :
    ∃ st, d.sts[j]? = some st ∧ rr.m = st.m := by
  unfold runOfDecR at h
  simp only [Option.bind_eq_bind, Option.pure_def] at h
  obtain ⟨st, hst, h⟩ := Option.bind_eq_some_iff.1 h
  obtain ⟨_, -, h⟩ := Option.bind_eq_some_iff.1 h
  obtain ⟨_, -, h⟩ := Option.bind_eq_some_iff.1 h
  refine ⟨st, hst, ?_⟩
  rw [← Option.some.inj h]
  simp [Setup.ofCircuit]

/-- **An accepted run of `VR rd` opens columns of the statement's table 0**: each has its level-0 width
(`accE_cols`'s proof, at the reader's columns). -/
theorem accR_cols {rd : Claims.Msg → Option FinalP} {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω}
    {r : Fin ((VR rd).k x) → Coin} (h : (Claims.run sha512 P (VR rd) x ω r).out = .acc) {q : ℕ} {u : List 𝔽}
    (hu : Claims.Flock.lookCol (colsRV rd (Claims.run sha512 P (VR rd) x ω r).seen 0) q = some u) :
    ∃ spec st, (tables false x).head? = some (spec, st) ∧
      u.length = 2 ^ (if false then Claims.Flock.zkLevel0 st.m else Claims.Flock.plainLevel0 st.m).lanes := by
  obtain ⟨fl, hfl, hx⟩ := accR_last h
  unfold colsRV at hu
  rw [show (Claims.run sha512 P (VR rd) x ω r).seen.msgs = (Claims.run sha512 P (VR rd) x ω r).msgs from rfl, hfl]
    at hu
  dsimp only at hu
  rw [hx] at hu
  split at hu
  · rename_i d hd
    obtain ⟨⟨s, spec, sts⟩, -, -, -, -, -, -, hv, -, -, -, -, -, rfl⟩ := decR_some hd
    unfold colsOf at hu
    split at hu
    · rename_i rr hrr
      obtain ⟨st, hst, hm⟩ := runOfDecR_m hrr
      refine ⟨spec, st, tables_head hv hst, ?_⟩
      obtain ⟨p, hp, rfl⟩ := exists_of_lookCol hu
      simp only [Bool.false_eq_true, ↓reduceIte]
      rw [plainRunCols_width hp, hm]
    · simp [Claims.Flock.lookCol] at hu
  · simp [Claims.Flock.lookCol] at hu

/-- **`ColsOk` at the plain record at `VR rd`**: an accepted run opens only columns of the zero column's width. -/
theorem plainR_colsOk (rel : Readable.Parsed → Readable.Relation) (rd : Claims.Msg → Option FinalP) :
    Claims.Flock.ColsOk sha512 (deployPlainR rel rd) (wordLen false) (wide false) := by
  intro x Ω P ω r q u hacc _ hu
  obtain ⟨spec, st, hh, hl⟩ := accR_cols hacc hu
  unfold wide
  rw [hl, zeroCol_length hh]

/-! ## `KSFlockLin` at the trimmed record -/

/-- **`KSFlockLin` at the plain record at `VR rd`**, at SHA-512, at the trimmed record and the deployed caps `CpDep`:
`KSFlockLin_of_tableBoundAbortLin` from conflicts tied to collisions and the table bound at the untrimmed record,
carried to the trimmed columns (`tableBoundAbortLin_mapCols`), with the record's numbers, one table wherever some run
accepts (`plainR_hone`) and `ColsOk` (`plainR_colsOk`), as `plainE_KSFlockLin`. -/
theorem plainR_KSFlockLin (rel : Readable.Parsed → Readable.Relation) (rd : Claims.Msg → Option FinalP)
    (hconf : Claims.Flock.ConflictTied sha512 (deployPlainR rel rd) (wordLen false))
    (htab : Claims.Recursion.TableBoundAbortLin sha512 (deployPlainR rel rd) (wordLen false) (wide false) CpDep) :
    Claims.Recursion.KSFlockLin sha512 (deployPlainTrimR rel rd) CpDep :=
  Claims.Recursion.KSFlockLin_of_tableBoundAbortLin sha512 (deployPlainTrimR rel rd) (wordLen false)
    (trimWide false) CpDep (wordLen_le_N₀ false) (fun x => εtable_nonneg false x) (fun _ => cap_pos)
    (plainR_kSplits rel rd) (plainR_hone rel rd) (trimWide_nil false)
    (Claims.Flock.colsOk_mapCols sha512 (deployPlainR rel rd) (wordLen false) trimCol (wide false)
      (plainR_colsOk rel rd))
    (Claims.Flock.conflictTied_mapCols sha512 (deployPlainR rel rd) (wordLen false) trimCol hconf)
    (Claims.Recursion.tableBoundAbortLin_mapCols sha512 (deployPlainR rel rd) (wordLen false) trimCol rfl
      (wide false) CpDep (plainR_colsOk rel rd) (wide_trimCol_inj false) htab)

end FlockDeploy.KS
