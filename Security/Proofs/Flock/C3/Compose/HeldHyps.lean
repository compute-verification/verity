import Proofs.Flock.C3.Compose.AtHoloHeld
import Proofs.Flock.C3.Sparse.PubChecked
import Proofs.Flock.Recursive.HoloSchedCheck

/-!
# The verifier's hold at the map read: `HeldOk`, `C3Held`, `SparseStagesAt` and `SparseCountedAt`

`stageCoins_holo_held`'s record hypotheses on the sparse side, at the guarantee's custody `_hIC` at `holoMap`:

* `heldOk_of_selC`, `heldOk_holo`: the verifier holds the record at every history long enough for the map's rounds:
  sparse phase 0's first coin round is drawn below them (`sparseDrawnAt_holo` at `holoSelWords`), and custody holds what
  a drawn round reads, which an error holds nowhere (`heldCoins`);
* `c3Held_heldC3`: `C3Held` at the `C3` built of the hold (`heldC3`, `c3Of`'s at a hold), the one check 5 reads at the
  guarantee; `held_commits`: with `HeldOk`, the record held is the run's `C3`'s commits (`stageHeldW_sparse`'s `hh`);
* `sparseStagesAt_sparseOf`: `SparseStagesAt` at the source whose sparse stages are the steps of `w` at `c` (`sparseOf`),
  from V*'s sessions at the run's `C3` (`SparseSessionAt`'s coins conjunct, `c.cs = sparseCs sc`) and `C3Held`;
* `sparseCountedAt_of_rounds`: `SparseCountedAt` from the round-for-round count of the held sparse streams, which
  `checkCommitRounds` makes at `sparsePhaseRounds sh` under `some sh := i.sparse` (`sparseCoins_of_checkCommitRounds`,
  as `coins_of_checkCommitRounds` reads the reps'; `sparseCoins` counts only the rounds that draw coins), so
  `sparseCountedAt_of_check` gives it from every hold's check.
-/

namespace FlockC3.Compose

noncomputable section

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockC3.Reps (repsOfVals UnitVals)
open Flock.Guarantees (holoCTs holoKinds holoRounds heldBefore coinRounds sparseRounds c3Of)
open Flock.Guarantees.HoloSchedCheck (checkBind_ok checkThrow_ne forIn_each)

/-- **Custody at a drawn round holds the record**: where round `i` of the history draws a stream's round (`m.selC`),
custody holds its coins there, which an error holds nowhere. -/
theorem heldOk_of_selC {Cm Coin : Type} {wr : List (Cm × Coin) → Lean.Json × String}
    {held : Lean.Json × String → Except String RecVStar.InnerCommits} {sparse : Option RecSparse.Shape}
    {m : FlockSoundness.Assumptions.InnerRecordMap Cm Coin}
    (hcus : FlockSoundness.Assumptions.InnerRecordCustody wr held sparse m) {t : List (Cm × Coin)} {i : ℕ}
    (hi : i < t.length) {s : String} {q : ℕ} (hq : m.selC i s q) : HeldOk wr held t := by
  have hc := ((hcus t).1 i t[i] (List.getElem?_eq_getElem hi)).2 s q hq
  cases h : held (wr t) with
  | error _ =>
    rw [h] at hc
    simp [FlockSoundness.Assumptions.heldCoins] at hc
  | ok cs => exact ⟨cs, h⟩

/-- **`HeldOk` at the map read**: sparse phase 0's first coin round (ζ_c's, of `PRE_ROUNDS` coins) is drawn at a
history round below the map's (`sparseDrawnAt_holo`), which a history of `hlen`'s length has. -/
theorem heldOk_holo {st : Flock.Stmt} {hp : HoloParams} {Cm : Type}
    {wr : List (Cm × ((k : Fin (1 + (Exec.kindCoinTys (holoKinds (holoAt st hp))).length)) → holoCTs st hp k)) →
      Lean.Json × String}
    {held : Lean.Json × String → Except String RecVStar.InnerCommits} {sparse : Option RecSparse.Shape}
    {ish : RecVStar.Inner} {tbl : String} {bc : Cm → String → Nat → List UInt8} {sp : RecSparse.Shape}
    {rr rs : Array RecVStar.SchedRound} {shR : RecVStar.StackedShape}
    (hcus : FlockSoundness.Assumptions.InnerRecordCustody wr held sparse
      (Flock.Guarantees.holoMap st hp Cm ish tbl bc))
    (hrep : holoRounds ish.m ish.kLog ish.claims = some rr) (hsp : ish.sparse = some sp)
    (hsh : ish.stacked = some shR) (hrs : RecVStar.stackedRounds shR = some rs) (hc : 0 < ish.claims)
    (hA : repFormsAt (holoAt st hp) = repForms ish.m ish.kLog ish.claims)
    (hshape : (holoAt st hp).cls.shape = sparseOf sp) (htblS : tbl ≠ s!"{RecVStar.SPARSE}/")
    {t : List (Cm × ((k : Fin (1 + (Exec.kindCoinTys (holoKinds (holoAt st hp))).length)) → holoCTs st hp k))}
    (hlen : 1 + (Exec.kindCoinTys (holoKinds (holoAt st hp))).length + 1 ≤ t.length) : HeldOk wr held t := by
  have hd := FlockC3.Sparse.sparseDrawnAt_holo (st := st) (hp := hp) 0 hsp hrep
    ⟨_, FlockC3.RegLevels.holoStreams_four hrep hsp hsh hrs⟩ hc hA hshape htblS
  have hg : ((sparseRounds sp).getD 0 default).1 = RecSparse.PRE_ROUNDS := by
    simp [sparseRounds]
  have h0 : 0 ∈ coinRounds (sparseRounds sp) := by
    unfold coinRounds
    simp [RecSparse.PRE_ROUNDS, sparseRounds]
  have hb : heldBefore (.sparse 0) (sparseRounds sp) 0 = 0 := by simp [heldBefore]
  have hnc : RecSparse.PRE_ROUNDS ≤ FlockVBridge.Sparse.nc (sparseOf sp) := by
    unfold FlockVBridge.Sparse.nc RecSparse.PRE_ROUNDS
    omega
  refine heldOk_of_selC hcus ?_ (selC_of_holoSelWords (bc := bc) (hd.2 0 h0))
  have := hd.1
  rw [hb, hg]
  omega

/-- `heldOk_holo` at every accepted outer proof. -/
theorem heldOk_atSrc {nT : ℕ} {T : Fin nT → Type} {st : Flock.Stmt} {hp : HoloParams} {n : ℕ} {L : Law n}
    {Reg Cm Reg' Reg₂ : Type} {S : ℕ} {Vs : Fin S → VStmt}
    {zs : Reg → L.Ω → List (Cm × (holoInnerAt T (holoCTs st hp) st hp).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    {src : Reg → L.Ω → List (Cm × (holoInnerAt T (holoCTs st hp) st hp).Coin) → ((j : Fin S) → (Vs j).Bits) →
      C3Run (holoAt st hp)}
    {wr : List (Cm × (holoInnerAt T (holoCTs st hp) st hp).Coin) → Lean.Json × String}
    {held : Lean.Json × String → Except String RecVStar.InnerCommits} {sparse : Option RecSparse.Shape}
    {ish : RecVStar.Inner} {tbl : String} {bc : Cm → String → Nat → List UInt8} {sp : RecSparse.Shape}
    {rr rs : Array RecVStar.SchedRound} {shR : RecVStar.StackedShape}
    (hcus : FlockSoundness.Assumptions.InnerRecordCustody wr held sparse
      (Flock.Guarantees.holoMap st hp Cm ish tbl bc))
    (hrep : holoRounds ish.m ish.kLog ish.claims = some rr) (hsp : ish.sparse = some sp)
    (hsh : ish.stacked = some shR) (hrs : RecVStar.stackedRounds shR = some rs) (hc : 0 < ish.claims)
    (hA : repFormsAt (holoAt st hp) = repForms ish.m ish.kLog ish.claims)
    (hshape : (holoAt st hp).cls.shape = sparseOf sp) (htblS : tbl ≠ s!"{RecVStar.SPARSE}/")
    (hlen : AtSrc zs dg N k Rw src fun t _ => 1 + (Exec.kindCoinTys (holoKinds (holoAt st hp))).length + 1 ≤ t.length) :
    AtSrc zs dg N k Rw src fun t _ => HeldOk wr held t := fun R ω t R₂ τ o hA' =>
  heldOk_holo hcus hrep hsp hsh hrs hc hA hshape htblS (hlen R ω t R₂ τ o hA')

/-- **The `C3` the verifier builds of its hold** (`c3Of` at a hold, at the inner shape's RecSparse shape `sp`). -/
def heldC3 (ish : RecVStar.Inner) (sp : RecSparse.Shape) : Except String RecVStar.InnerCommits → RecCoins.C3
  | .ok cs => { sparse := sp, stacked := ish.stacked.or cs.stacked, commits := cs }
  | .error _ => { sparse := sp, stacked := ish.stacked, commits := { reps := #[] } }

section src

variable {nT : ℕ} {T : Fin nT → Type} {st : Flock.Stmt} {hp : HoloParams} {n : ℕ} {L : Law n}
  {Reg Cm Reg' Reg₂ : Type} {S : ℕ} {Vs : Fin S → VStmt}

local notation "ℐ" => holoInnerAt T (holoCTs st hp) st hp

/-- **`C3Held` at the `C3` built of the hold**, where the inner shape names its RecSparse shape. -/
theorem c3Held_heldC3 {wr : List (Cm × (ℐ).Coin) → Lean.Json × String}
    {held : Lean.Json × String → Except String RecVStar.InnerCommits} {ish : RecVStar.Inner} {sp : RecSparse.Shape}
    (hsp : ish.sparse = some sp) :
    C3Held wr held ish fun (_ : Reg) (_ : L.Ω) (t : List (Cm × (ℐ).Coin)) (_ : (j : Fin S) → (Vs j).Bits) =>
      heldC3 ish sp (held (wr t)) := by
  intro R ω t Lb cs h
  show c3Of ish (some cs) = some (heldC3 ish sp (held (wr t)))
  rw [h]
  unfold Flock.Guarantees.c3Of
  rw [hsp]
  rfl

/-- **The record held is the run's `C3`'s commits** (`stageHeldW_sparse`'s `hh`), under `C3Held`. -/
theorem held_commits {wr : List (Cm × (ℐ).Coin) → Lean.Json × String}
    {held : Lean.Json × String → Except String RecVStar.InnerCommits} {ish : RecVStar.Inner}
    {c3 : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → RecCoins.C3}
    (hc3 : C3Held wr held ish c3) {t : List (Cm × (ℐ).Coin)} (hh : HeldOk wr held t) (R : Reg) (ω : L.Ω)
    (Lb : (j : Fin S) → (Vs j).Bits) : held (wr t) = .ok (c3 R ω t Lb).commits := by
  obtain ⟨cs, h⟩ := hh
  rw [FlockC3.RegLevels.commits_c3Of (hc3 R ω t Lb cs h)]
  exact h

/-- **`SparseStagesAt` at `sparseOf`'s source**: its sparse stages are the steps of `w` at `c`'s coins by definition
(`repsOfVals` and `tie` keep them), and V*'s sessions at the run's `C3` give `c.cs = sparseCs sc` at the coins
`sparseCoins` accepts there (`SparseSessionAt`), which is the `C3` of the hold (`C3Held`). -/
theorem sparseStagesAt_sparseOf {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    {stp : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    {sh : Flock.Shape} {rep : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData}
    {vals : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → UnitVals}
    {wr : List (Cm × (ℐ).Coin) → Lean.Json × String}
    {held : Lean.Json × String → Except String RecVStar.InnerCommits} {ish : RecVStar.Inner} {unit : String}
    {sp : RecSparse.Shape} {c3 : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → RecCoins.C3}
    {w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → Array F128}
    {c : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → FlockVBridge.Sparse.SCoins}
    (hc3 : C3Held wr held ish c3) (hshape : (holoAt st hp).cls.shape = sparseOf sp)
    (hs : ∀ R ω t R₂ τ o, FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o → ∀ i : Fin 2,
      FlockC3.Sparse.SparseSessionAt (zs R ω t R₂) (layers zs dg N k Rw R ω t R₂ τ) (holoAt st hp)
        (c3 R ω t (layers zs dg N k Rw R ω t R₂ τ)) unit i (w R ω t (layers zs dg N k Rw R ω t R₂ τ) i)
        (c R ω t (layers zs dg N k Rw R ω t R₂ τ) i)) :
    AtSrc zs dg N k Rw (tiedSrc (repsOfVals (FlockC3.Sparse.sparseOf stp w c) sh rep vals)) fun t u =>
      SparseStagesAt wr held ish unit sp t u := by
  intro R ω t R₂ τ o hA cs c3' hh hc3' x
  obtain rfl : c3' = c3 R ω t (layers zs dg N k Rw R ω t R₂ τ) :=
    Option.some.inj (hc3'.symm.trans (hc3 R ω t _ cs hh))
  obtain ⟨⟨sc, hsc, -, hcs⟩, -⟩ := hs R ω t R₂ τ o hA x
  refine ⟨w R ω t (layers zs dg N k Rw R ω t R₂ τ) x, sc, hsc, ?_⟩
  show FlockC3.Sparse.stepsOf (holoAt st hp).cls.shape _ (c R ω t (layers zs dg N k Rw R ω t R₂ τ) x).cs = _
  rw [hshape, hcs]

end src

/-- **`SparseCountedAt` from the round-for-round count of the held sparse streams** (the check `checkCommitRounds`
makes of every other held stream, at `sparsePhaseRounds`): at every hold, each held `sparse/…` stream draws, round by
round, the coins of the sparse phase's schedule. -/
theorem sparseCountedAt_of_rounds {Cm Coin : Type} {wr : List (Cm × Coin) → Lean.Json × String}
    {held : Lean.Json × String → Except String RecVStar.InnerCommits} {sp : RecSparse.Shape} {t : List (Cm × Coin)}
    (h : ∀ cs, held (wr t) = .ok cs → ∀ st ∈ cs.streams, st.name.startsWith s!"{RecVStar.SPARSE}/" = true →
      st.coins.map (·.size) = (RecVStar.sparsePhaseRounds sp).map (·.1)) :
    SparseCountedAt wr held sp t := by
  intro cs hh x st hst
  have hs' : cs.streams.find? (·.name == s!"{RecVStar.SPARSE}/rep{x.val}") = some st := hst
  have hname : st.name = s!"{RecVStar.SPARSE}/rep{x.val}" := by
    first
    | exact beq_iff_eq.1 (Array.find?_some hs')
    | simpa using Array.find?_some hs'
  refine h cs hh st (Array.mem_of_find?_eq_some hs') ?_
  rw [hname, String.startsWith_string_iff]
  simp only [FlockVBridge.toString_str, String.toList_append, List.append_assoc]
  exact List.prefix_append _ _

/-- **`checkCommitRounds` accepting counts the sparse phases' coins**: under the verifier's RecSparse shape `sh`, every
held stream named `sparse/…` draws `sparsePhaseRounds sh`'s coins round for round (its loop after the refusal of any
other `sparse/…` name). -/
theorem sparseCoins_of_checkCommitRounds {i : RecVStar.Inner} {cs : RecVStar.InnerCommits}
    {rep : Array RecVStar.SchedRound} {sh : RecSparse.Shape}
    (hsch : (if i.holo then RecVStar.holoRepRounds else RecVStar.repRounds) i.m i.kLog i.claims = some rep)
    (hsp : i.sparse = some sh) (h : RecVStar.checkCommitRounds i cs = .ok ()) :
    ∀ st ∈ cs.streams, st.name.startsWith s!"{RecVStar.SPARSE}/" = true →
      st.coins.map (·.size) = (RecVStar.sparsePhaseRounds sh).map (·.1) := by
  unfold RecVStar.checkCommitRounds at h
  rw [hsch] at h
  try simp only at h
  obtain ⟨_, -, h⟩ := checkBind_ok h
  split at h
  · exact absurd h (checkThrow_ne _ _)
  rcases hf : cs.streams.find? (fun st => st.name.startsWith s!"{RecVStar.SPARSE}/" &&
      st.name != s!"{RecVStar.SPARSE}/rep0" && st.name != s!"{RecVStar.SPARSE}/rep1") with _ | stx
  rotate_left
  · rw [hf] at h
    try simp only at h
    exact absurd h (checkThrow_ne _ _)
  rw [hf] at h
  try simp only at h
  obtain ⟨_, h, -⟩ := checkBind_ok h
  unfold RecVStar.checkSparseCoins at h
  rw [hsp] at h
  try simp only at h
  obtain ⟨_, h, -⟩ := checkBind_ok h
  have hall := forIn_each (fun st : RecVStar.InnerStream => st.name.startsWith s!"{RecVStar.SPARSE}/" = true →
      st.coins.map (·.size) = (RecVStar.sparsePhaseRounds sh).map (·.1)) ?_ cs.streams.toList
    (Array.forIn_toList.trans h)
  · exact fun st hst => hall st (Array.mem_toList_iff.2 hst)
  intro a r hb
  try simp only at hb
  split at hb
  · exact absurd hb (checkThrow_ne _ _)
  · rename_i hne
    exact ⟨fun hn => by simpa [hn] using hne, (Except.ok.inj hb).symm⟩

/-- **`SparseCountedAt` from the verifier's check**: where every hold at `t` passes `checkCommitRounds` at an inner shape
with a reps' schedule and RecSparse shape `sp` (`hchk_of_held`'s `hchk`), its sparse streams are counted. -/
theorem sparseCountedAt_of_check {Cm Coin : Type} {wr : List (Cm × Coin) → Lean.Json × String}
    {held : Lean.Json × String → Except String RecVStar.InnerCommits} {sp : RecSparse.Shape} {t : List (Cm × Coin)}
    {i : RecVStar.Inner} {rep : Array RecVStar.SchedRound}
    (hsch : (if i.holo then RecVStar.holoRepRounds else RecVStar.repRounds) i.m i.kLog i.claims = some rep)
    (hsp : i.sparse = some sp) (hchk : ∀ cs, held (wr t) = .ok cs → RecVStar.checkCommitRounds i cs = .ok ()) :
    SparseCountedAt wr held sp t :=
  sparseCountedAt_of_rounds fun cs hh => sparseCoins_of_checkCommitRounds hsch hsp (hchk cs hh)

end

end FlockC3.Compose

#print axioms FlockC3.Compose.sparseCoins_of_checkCommitRounds
#print axioms FlockC3.Compose.sparseCountedAt_of_check
#print axioms Flock.Guarantees.HoloSchedCheck.commitRounds_of_check
