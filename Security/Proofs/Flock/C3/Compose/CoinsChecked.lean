import Proofs.Flock.C3.Compose.Decoder
import Proofs.Flock.C3.Compose.HoloStageCoins
import Proofs.Flock.C3.Compose.HoloSparseWords
import Proofs.Flock.C3.Compose.HoloStreamSizes
import Proofs.Flock.C3.Reps.DrawnRounds

/-!
# The stages' coins at the checked sessions

`stageCoins_checked`: `hcoins` of `srcOK_holo_rs` and `srcOK_repsOfVals_pR` (`StageCoins` at `holoCTs`, at the tied
source of `repsOfVals`) at the checked sessions, the guarantee's hold of the inner transcript and reps whose rounds are
the record's (`hrd`, `rfl` at the decoder with its reads, `RepMaps.withReads`), from the guarantee's custody at
`holoMap` (`_hIC`). It is `stageCoins_holo` at the map's one-round draws (`holoMapOne`), where the map's coins are the
history's (`coinsEncoded_holoMap`), as they are not at all of `holoMap`'s (a sparse phase's round of `n > 1` coins);
custody at `holoMap` holds there (`custody_holoMapOne`), and the reps' rounds are held by `hrd`
(`repRoundsHeld_of_rounds`). `RepsDrawn` there is `repsDrawn_checked`'s, at the class's rep forms (`repFormsAt_holoAt`),
from the inner shape check the event carries (`AcceptsC3`) and the claims tie that V*'s list refuses without
(`shape_of_statements`); the session's streams exist wherever V*'s list builds at a RecSparse shape
(`holoStreams_of_list`).
-/

namespace FlockC3.Compose

noncomputable section

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockC3.Reps (repsOfVals UnitVals ProofOfVals SetupOfVals FramesOfVals SrcRepTie)
open FlockSoundness.Assumptions (InnerRecordCustody)
open Flock.Guarantees (holoCTs holoMap)

/-- **V*'s shape and the claims tie are the verifier's**: V*'s list refuses an inner shape off V*'s
(`RecResiduals.shape?`: a `fast100` schedule, `k_log ≥ 6`, two claims or more) and one whose claims are not two and
two per region (`RecVStar.statements`). -/
theorem shape_of_statements {i : RecVStar.Inner} {ss : Array VStar.Stmt} (h : RecVStar.statements i = .ok ss) :
    (∃ sh, RecResiduals.shape? i.m i.kLog i.claims = some sh) ∧ i.claims = 2 + 2 * i.regions := by
  unfold RecVStar.statements at h
  dsimp only at h
  split at h
  · cases h
  split at h
  rotate_left
  · cases h
  rename_i sh hsh
  split at h
  · cases h
  rename_i hc
  exact ⟨⟨sh, hsh⟩, by simpa using hc⟩

/-- **`z_partial`'s round of a rep's head** (`zpRound`) is a message round of one coin, whatever follows the head. -/
theorem repHead_zp {m kLog claims : ℕ} (hm : 5 ≤ m) (hk : 6 ≤ kLog) (T : Array RecVStar.SchedRound) :
    (repHead m kLog claims ++ T)[Flock.Guarantees.zpRound m kLog]? = some (1, #[], 0, true) := by
  have e : ∀ j, j = 0 → (#[(1, #[], 0, true)] : Array RecVStar.SchedRound)[j]? = some (1, #[], 0, true) := by
    rintro _ rfl
    rfl
  unfold repHead Flock.Guarantees.zpRound
  rw [Array.getElem?_append_left ?_, Array.getElem?_append_left ?_, Array.getElem?_append_left ?_,
    Array.getElem?_append_right ?_, e]
  all_goals
    simp only [Array.size_append, Array.size_replicate, List.size_toArray, List.length_cons, List.length_nil]
    omega

/-- **The session's streams exist** wherever V*'s list builds at a RecSparse shape: the inner shape is at a
`fast100` schedule with `k_log ≥ 6` (`shape_of_statements`), so `repRounds` passes at every `k_log` and claims
(`repRounds_tail`, by evaluating it at each `m` below 36), its head has `z_partial`'s round where `holoRounds` reads it
(`repHead_zp`), and R's shape is `⟨tableM sp, 2⟩` (`stacked_of_candidate3`) at a `fast100` schedule (`stacked_of_list`),
where R's rounds pass (by evaluation). -/
theorem holoStreams_of_list {i : RecVStar.Inner} {ss : Array VStar.Stmt} (hss : RecVStar.statements i = .ok ss)
    (hsp : i.sparse.isSome = true) :
    ∃ Ls, Flock.Guarantees.holoStreams i.m i.kLog i.claims i.sparse i.stacked = some Ls := by
  obtain ⟨⟨_, hsh⟩, -⟩ := shape_of_statements hss
  obtain ⟨hf, -, -, -, hk, -⟩ := FlockVBridge.Family.shape?_eq hsh
  obtain ⟨sp, hsp⟩ := Option.isSome_iff_exists.1 hsp
  obtain ⟨_, _, _, -, hC, -, -⟩ := FlockC3.VStarList.statements_ok hss
  have hst := FlockC3.VStarList.stacked_of_candidate3 hC hsp
  obtain ⟨sch, hf', -⟩ := FlockC3.VStarList.stacked_of_list hss hst
  replace hf' : Flock.fast100 (RecVStar.tableM sp) = some sch := hf'
  have key : ∀ m < 36, (Flock.fast100 m).isSome → (RecVStar.repRounds m 6 1).isSome ∧
      (RecVStar.stackedRounds ⟨m, RecVStar.STACKED_CLAIMS⟩).isSome := by
    decide +kernel
  obtain ⟨hm, hm'⟩ := FlockVBridge.Family.fast100_range hf
  obtain ⟨-, hm''⟩ := FlockVBridge.Family.fast100_range hf'
  obtain ⟨_, hout⟩ := Option.isSome_iff_exists.1 (key i.m (by omega) (Option.isSome_iff_exists.2 ⟨_, hf⟩)).1
  obtain ⟨T, -, hall⟩ := repRounds_tail hout
  obtain ⟨rs, hrs⟩ := Option.isSome_iff_exists.1
    (key (RecVStar.tableM sp) (by omega) (Option.isSome_iff_exists.2 ⟨_, hf'⟩)).2
  have hzp := repHead_zp (m := i.m) (claims := i.claims) (by omega) hk T
  refine Option.isSome_iff_exists.1 ?_
  unfold Flock.Guarantees.holoStreams Flock.Guarantees.holoRounds
  simp only [hall, hsp, hst, Option.bind_eq_bind, Option.bind_some, hzp, hrs, bne_self_eq_false]
  all_goals rfl

/-- **`holoMap` at its one-round draws**: its records and coins, drawing a stream round only where it takes one history
round (`holoSelOne`). -/
def holoMapOne (st : Flock.Stmt) (hp : HoloParams) (Cm : Type) (ish : RecVStar.Inner) (tbl : String)
    (bc : Cm → String → Nat → List UInt8) :
    FlockSoundness.Assumptions.InnerRecordMap Cm
      ((k : Fin (1 + (Exec.kindCoinTys (Flock.Guarantees.holoKinds (holoAt st hp))).length)) → holoCTs st hp k) :=
  { holoMap st hp Cm ish tbl bc with selC := holoSelOne ish tbl }

/-- **Custody at `holoMap` holds at its one-round draws**, each one of the map's (`selC_of_holoSelOne`). -/
theorem custody_holoMapOne {st : Flock.Stmt} {hp : HoloParams} {Cm : Type} {ish : RecVStar.Inner} {tbl : String}
    {bc : Cm → String → Nat → List UInt8}
    {wr : List (Cm × ((k : Fin (1 + (Exec.kindCoinTys (Flock.Guarantees.holoKinds (holoAt st hp))).length)) →
      holoCTs st hp k)) → Lean.Json × String}
    {held : Lean.Json × String → Except String RecVStar.InnerCommits} {sparse : Option RecSparse.Shape}
    (h : InnerRecordCustody wr held sparse (holoMap st hp Cm ish tbl bc)) :
    InnerRecordCustody wr held sparse (holoMapOne st hp Cm ish tbl bc) := fun t =>
  ⟨fun i e he => ⟨((h t).1 i e he).1, fun _ _ hs => ((h t).1 i e he).2 _ _ (selC_of_holoSelOne hs)⟩, (h t).2⟩

variable {nT : ℕ} {T : Fin nT → Type} {st : Flock.Stmt} {hp : HoloParams} {n : ℕ} {L : Law n}
  {Reg Cm Reg' Reg₂ : Type} {S : ℕ} {Vs : Fin S → VStmt}

local notation "ℐ" => holoInnerAt T (holoCTs st hp) st hp

variable {ish : RecVStar.Inner} {reg : Lean.Json} {wr : List (Cm × (ℐ).Coin) → Lean.Json × String}
  {coins : List (Cm × (ℐ).Coin) → Option Flock.RecCoins.Source} {bind : Fin S → VStar.Bindings}
  {rows : Option (Array Flock.Registered.Port)}
  {zr : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).OuterR Reg'}
  {hC : Flock.Premises.RecordCustody zr} {h : Typed zr} {sh : Flock.Shape}

local notation "𝒵" => Flock.Guarantees.checkedC3 ish (Flock.Guarantees.innerHeld wr st reg) coins bind rows zr hC h

-- elaborating this signature at `holoInnerAt` runs past the default
set_option diagnostics true in
set_option maxHeartbeats 1600000 in
/-- **The stages' coins at the checked sessions** (`hcoins` of `srcOK_holo_rs` and `srcOK_repsOfVals_pR`), at reps
whose rounds are the record's (`hrd`; `fun _ _ _ _ _ => rfl` at the decoder with its reads, `RepMaps.withReads`): the
reps' from the guarantee's custody (`_hIC`) at the map's one-round draws, where the reps are drawn
(`repsDrawn_checked`) wherever V*'s list builds (`hss`), the class checks pass (`hcls`) and the inner shape is the
statement's (`hi`, `AcceptsC3`'s), and the link points' and the later stages' from `LaterCoins` (`laterCoins_of_held`).
`halg` and `hacc` are E's (`withReads_algebraDecodes`), `hpr`, `hsv` and `hf` the value tie's. -/
theorem stageCoins_checked {ss : Array VStar.Stmt} {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    {stp : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    {rep : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData}
    {vals : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → UnitVals}
    {tbl : String} {bc : Cm → String → ℕ → List UInt8} {cap₀ : Cap Dig} (hS₀ : 0 < S)
    (hss : RecVStar.statements ish = .ok ss) (hcls : ClassOk (shapeOf st))
    (hi : (RecVStar.checkInnerStmt ish st).isOk = true)
    (halg : AlgebraDecodes (I := ℐ) 𝒵 dg N k Rw sh rep) (hacc : AccChains (I := ℐ) 𝒵 dg N k Rw sh rep)
    (hpr : ProofOfVals 𝒵 dg N k Rw sh rep vals)
    (hsv : SetupOfVals 𝒵 dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) sh rep vals)
    (hf : FramesOfVals 𝒵 dg N k Rw sh rep vals)
    (hcus : InnerRecordCustody wr (fun w => innerCommitsAt w.1 w.2 st reg) ish.sparse (holoMap st hp Cm ish tbl bc))
    (htbl : ∀ i : Fin 2, (((Flock.Guarantees.HoloStream.rep i).name tbl).startsWith s!"{RecVStar.SPARSE}/" ||
      ((Flock.Guarantees.HoloStream.rep i).name tbl).startsWith s!"{RecVStar.STACKED}/") = false)
    (hrd : ∀ R ω t Lb i, (rep R ω t Lb i).rounds =
      FlockC3.Reps.heldRounds (innerCommitsAt (wr t).1 (wr t).2 st reg) ((Flock.Guarantees.HoloStream.rep i).name tbl))
    (hrt : SrcRepTie 𝒵 dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)))
    (hst : FlockC3.Sparse.SrcSparseTie 𝒵 dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)))
    (hot : AtSrc 𝒵 dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) fun _ u => ∃ sh w ops,
      FlockC3.Opening.OpenTie sh w ops u ∧ OpeningReadAt (runSteps u) u)
    (hlt : FlockC3.RegLevels.SrcLevelTie 𝒵 dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) cap₀)
    (hlater : AtSrc 𝒵 dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) fun t u =>
      LaterCoins (CT := holoCTs st hp) (t.map Prod.snd) u) :
    AtSrc 𝒵 dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) fun t u =>
      StageCoins (CT := holoCTs st hp) (t.map Prod.snd) u := by
  have htie := (shape_of_statements hss).2
  obtain ⟨⟨⟩, hi'⟩ := FlockC3.VStarList.ok_of_isOk hi
  obtain ⟨_, hS⟩ := holoStreams_of_list hss (FlockC3.VStarList.checkInnerStmt_ok hi').1
  exact stageCoins_holo (m := holoMapOne st hp Cm ish tbl bc) (hLeafC3_checked hS₀ dg N k Rw) halg hacc hpr hsv hf
    (custody_holoMapOne hcus)
    (FlockC3.Reps.repRoundsHeld_of_rounds (wr := wr) (held := fun w => innerCommitsAt w.1 w.2 st reg)
      (s := fun i => (Flock.Guarantees.HoloStream.rep i).name tbl) hrd)
    (coinsEncoded_holoMap (ish := ish) (tbl := tbl) (bc := bc))
    (FlockC3.Reps.repsDrawn_checked hS₀ hS (by omega) (repFormsAt_holoAt hcls hi' (by omega)) halg hacc hpr hsv hf
      hcus hrd htbl) hrt hst hot hlt hlater

end

end FlockC3.Compose

#print axioms FlockC3.Compose.holoStreams_of_list
#print axioms FlockC3.Compose.custody_holoMapOne
#print axioms FlockC3.Compose.stageCoins_checked
