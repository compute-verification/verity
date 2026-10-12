import Proofs.Flock.Recursive.HoloExec
import Proofs.Flock.Recursive.ClashFree
import Proofs.Flock.C3.Fork.CouplingLock

/-!
# a1l, step 5: no accepted run in `W` disagrees with `W`'s tables

The coupling's oracle prover sends `W`'s tables (`chH`): at a rep's slot, the rows the reps' part of an outcome of `W`
opens there (`tabRW` at `WC W`); at R's, the rows R's part opens (`tabRW` at `WR W x`, R's stream as rep 0 of a
compiled table at the registered cap). `holo_bad_zero`: where the tables a run records are those, an accepted run in
`W` that disagrees with them clashes with a run in `W` (`ClashHL`): on the reps as in `bad_zero_of_fixed`, on R's
levels `ℓ ≥ 1` the same way at R's part, and on R's level 0, whose table is the registered word, by a `RegClash`.
-/

namespace FlockC3.Fork

open FlockSoundness FlockSoundness.Game FlockSoundness.Model FlockSoundness.Accounting
open FlockSoundness.Discharge.Exec FlockSoundness.Discharge.Recursion

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {D Sl : Type}

/-- The reps' parts of `W`'s outcomes, as compiled-table outcomes. -/
def WC (W : OutHL F D Sl → Prop) (c : OutCL F D Sl) : Prop := ∃ o, W o ∧ o.circ = c

/-- R's parts of `W`'s outcomes, as compiled-table outcomes whose rep 0 is R's stream at the registered cap. -/
def WR (W : OutHL F D Sl → Prop) (x : HoloReg F D) (c : OutCL F D Sl) : Prop := ∃ o, W o ∧ o.reg x.cap = c

/-- **`W`'s tables**, at the lockstep's slots: a rep's at the even ones, R's at the odd ones. -/
noncomputable def chH (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H)
    (sch rsch : Schedule) (W : OutHL F D Sl → Prop) (x : HoloReg F D) (s : ℕ) (d : Cap D) : Oracle F :=
  if s % 2 = 0 then tabRW H E Lf sch (WC W) (s / 2) d else tabRW H E Lf rsch (WR W x) (s / 2) d

theorem not_of_ind_le_zero {α : Type} {p : α → Prop} {a : α} (h : ind p a ≤ 0) : ¬ p a := fun hp => by
  unfold ind at h
  rw [if_pos hp] at h
  exact not_le.2 zero_lt_one h

/-- **The reps agree with `W`'s tables**: `bad_zero_of_fixed` at the reps' part, whose clashes are `ClashHL`'s first. -/
theorem holo_reps_agree (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H)
    (sch rsch : Schedule) (x : HoloReg F D) (W : OutHL F D Sl → Prop)
    (hW : ∀ o o', W o → W o' → ¬ ClashHL H E Lf sch rsch x o o') (cap₀ : Cap D)
    (i : (IRep F D × IRep F D) × IRep F D) (b : OutHL F D Sl) (hb : W b) (hacc : b.accepted = true)
    (hcap : b.cap₀ = cap₀) (hok₀ : OpensOKL H E Lf cap₀ 0 b.pend.1 b.opens b.salts)
    (hok₁ : OpensOKL H E Lf cap₀ 1 b.pend.2 b.opens b.salts) (hs₀ : RecStruct sch i.1.1 b.pend.1)
    (hs₁ : RecStruct sch i.1.2 b.pend.2)
    (hfix : ∀ s t d, getEO (Lock.getPair (getRep 0) (getRep 1)) (getRep 0) i s = some (t, d) →
      t = chH H E Lf sch rsch W x s d) :
    AgreeRep 0 (tab₀W H E Lf sch (WC W) cap₀) i.1.1 b.pend.1.reqs b.opens ∧
      AgreeRep 1 (tab₀W H E Lf sch (WC W) cap₀) i.1.2 b.pend.2.reqs b.opens := by
  have hz := bad_zero_of_fixed H E Lf sch (WC W)
    (by rintro o ⟨o₁, ho₁, rfl⟩ hcl; exact hW o₁ o₁ ho₁ ho₁ (Or.inl (Or.inl hcl)))
    (by
      rintro o o' ⟨o₁, ho₁, rfl⟩ ⟨o₂, ho₂, rfl⟩ p hc
      exact hW o₁ o₂ ho₁ ho₂ (Or.inl (Or.inr (Or.inr (Or.inl ⟨p, hc⟩)))))
    (by
      rintro o o' ⟨o₁, ho₁, rfl⟩ ⟨o₂, ho₂, rfl⟩ r hr ℓ hℓ cap hc p hconf
      exact hW o₁ o₂ ho₁ ho₂ (Or.inl (Or.inr (Or.inr (Or.inr ⟨r, hr, ℓ, hℓ, cap, hc, p, hconf⟩)))))
    cap₀ i.1 (true, tab₀W H E Lf sch (WC W) cap₀) b.circ
    ⟨rfl, hcap, fun _ _ _ => rfl, fun _ => ⟨hok₀, hok₁⟩, hs₀, hs₁⟩
    (fun s t d hs => by
      have h := hfix (2 * s) t d (by rw [getEO_even]; exact hs)
      have h1 : 2 * s % 2 = 0 := by omega
      have h2 : 2 * s / 2 = s := by omega
      rw [h, chH, if_pos h1, h2])
  by_contra hn
  exact not_of_ind_le_zero hz ⟨hacc, ⟨b, hb, rfl⟩, hn⟩

/-- **R's level 0**: an opening of R's level 0 off the registered word, in R's part of a run, is a `RegClash`. R's part
has no rep 1, so its level-0 openings are R's own. -/
theorem regClash_of_off₀ (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H)
    (rsch : Schedule) (x : HoloReg F D) (b : OutHL F D Sl) {p : Fin (2 ^ (lvOf rsch 0).logLen)}
    (hoff : Rewinding.Off (op₀L H E Lf rsch x.cap) (rowT _ _ x.w) (b.reg x.cap) p) :
    RegClash H E Lf rsch.level₀ x b := by
  obtain ⟨y, hy, hyp, hver, hop⟩ := opOfL_some hoff.1
  have hlv : rsch.level₀ = lvOf rsch 0 := by
    rcases rsch with ⟨m, k0, _ | ⟨l, ls⟩, ext⟩ <;> rfl
  have h1 : slotOpensL (b.reg x.cap) 1 0 (lvOf rsch 0).queries = [] := rfl
  unfold opens₀L at hy
  rw [h1, List.append_nil] at hy
  rw [hlv]
  refine ⟨y, hy, by rw [hyp]; exact p.isLt, hver, fun he => hoff.2 (hop.trans (congrArg some ?_))⟩
  funext lane
  simp only [rowFin, rowT, he, HoloReg.column, hyp]
  rw [List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range lane.isLt]
  rfl

/-- **R agrees with the registered word and `W`'s tables**: `rep_badL` at R's part, rep 0 of a compiled table at the
registered cap. A clash there is `ClashHL`'s second; an opening of level 0 off `x.w` is a `RegClash`; an opening of a
level `ℓ ≥ 1` off the table recorded there, which `hfix` makes `W`'s, conflicts with a run of `W`. -/
theorem holo_regR_agree (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H)
    (sch rsch : Schedule) (x : HoloReg F D) (W : OutHL F D Sl → Prop)
    (hW : ∀ o o', W o → W o' → ¬ ClashHL H E Lf sch rsch x o o')
    (i : (IRep F D × IRep F D) × IRep F D) (b : OutHL F D Sl) (hb : W b)
    (hokR : OpensOKL H E Lf x.cap 2 b.pendR b.opens b.salts) (hsR : RecStruct rsch i.2 b.pendR)
    (hfix : ∀ s t d, getEO (Lock.getPair (getRep 0) (getRep 1)) (getRep 0) i s = some (t, d) →
      t = chH H E Lf sch rsch W x s d) :
    AgreeRep 2 x.w i.2 b.pendR.reqs b.opens := by
  by_contra hn
  rcases rep_badL H E Lf rsch x.w x.cap (b.reg x.cap) 0 (by norm_num) i.2 (by rw [if_pos rfl]; exact hsR)
      (by rw [if_pos rfl]; exact hokR) (by rw [if_pos rfl]; exact hn) with
    hcl | ⟨p, hoff⟩ | ⟨ℓ, hℓ, t, d, hi, hc, ⟨p, hoff⟩ | hcl⟩
  · exact hW b b hb hb (Or.inr (Or.inl (Or.inl (Or.inl hcl))))
  · exact hW b b hb hb (Or.inr (Or.inr (Or.inl (regClash_of_off₀ H E Lf rsch x b hoff))))
  · have hℓ1 := (Finset.mem_Icc.1 hℓ).1
    have ht : t = tabRW H E Lf rsch (WR W x) (slotOf 0 ℓ) d := by
      have h := hfix (2 * slotOf 0 ℓ + 1) t d (by rw [getEO_odd, getRep_slot 0 (by norm_num) i.2 ℓ hℓ1]; exact hi)
      have h1 : ¬ (2 * slotOf 0 ℓ + 1) % 2 = 0 := by omega
      have h2 : (2 * slotOf 0 ℓ + 1) / 2 = slotOf 0 ℓ := by omega
      rw [h, chH, if_neg h1, h2]
    subst ht
    obtain ⟨o, ⟨o₁, ho₁, rfl⟩, hconf⟩ :=
      conflictR_of_off H E Lf rsch (WR W x) 0 (by norm_num) ℓ d ⟨b, hb, rfl⟩ hoff
    exact hW b o₁ hb ho₁ (Or.inr (Or.inl (Or.inr (Or.inr (Or.inr ⟨0, by norm_num, ℓ, hℓ1, d, hc, p, hconf⟩)))))
  · exact hW b b hb hb
      (Or.inr (Or.inl (Or.inl (Or.inr ⟨0, by norm_num, ℓ, (Finset.mem_Icc.1 hℓ).1, d, hc, hcl⟩))))

/-- **An accepted run in `W` agrees with `W`'s tables**: at the level-0 table `W`'s at the run's cap, R's level 0 the
registered word, and the tables the run records `W`'s (`chH`), disagreeing would be a clash of the run with a run of
`W`. -/
theorem holo_bad_zero (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H)
    (sch rsch : Schedule) (x : HoloReg F D) (W : OutHL F D Sl → Prop)
    (hW : ∀ o o', W o → W o' → ¬ ClashHL H E Lf sch rsch x o o') (cap₀ : Cap D)
    (i : (IRep F D × IRep F D) × IRep F D) (v : Bool) (b : OutHL F D Sl)
    (hR : TopRelH H E Lf sch rsch (tab₀W H E Lf sch (WC W) cap₀) x.w cap₀ x.cap i v b)
    (hfix : ∀ s t d, getEO (Lock.getPair (getRep 0) (getRep 1)) (getRep 0) i s = some (t, d) →
      t = chH H E Lf sch rsch W x s d) :
    ind (fun out : OutHL F D Sl => out.accepted = true ∧ W out ∧
      ¬ (AgreeRep 0 (tab₀W H E Lf sch (WC W) cap₀) i.1.1 out.pend.1.reqs out.opens ∧
        AgreeRep 1 (tab₀W H E Lf sch (WC W) cap₀) i.1.2 out.pend.2.reqs out.opens ∧
        AgreeRep 2 x.w i.2 out.pendR.reqs out.opens)) b ≤ 0 := by
  have hnot : ¬ (b.accepted = true ∧ W b ∧
      ¬ (AgreeRep 0 (tab₀W H E Lf sch (WC W) cap₀) i.1.1 b.pend.1.reqs b.opens ∧
        AgreeRep 1 (tab₀W H E Lf sch (WC W) cap₀) i.1.2 b.pend.2.reqs b.opens ∧
        AgreeRep 2 x.w i.2 b.pendR.reqs b.opens)) := by
    rintro ⟨hacc, hb, hbad⟩
    obtain ⟨hcap, -, hok, hs₀, hs₁, hsR⟩ := hR
    obtain ⟨hok₀, hok₁, hokR⟩ := hok hacc
    obtain ⟨h₀, h₁⟩ := holo_reps_agree H E Lf sch rsch x W hW cap₀ i b hb hacc hcap hok₀ hok₁ hs₀ hs₁ hfix
    exact hbad ⟨h₀, h₁, holo_regR_agree H E Lf sch rsch x W hW i b hb hokR hsR hfix⟩
  refine le_of_eq ?_
  unfold ind
  rw [if_neg hnot]

end FlockC3.Fork
