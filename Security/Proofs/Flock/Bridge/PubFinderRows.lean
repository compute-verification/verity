import Proofs.Flock.Bridge.PubFinder

/-!
# `SampledKSLivePub`'s finder at the registrant's rows

`SampledKSLivePub` reads the claims at the rows the registrant opens, so its finder `B^τ` (`pubFinderR`) runs, besides
`pubFinder`'s members (each table's knowledge finder slots, every link finder, and the finder that outputs nothing):

* each committed position's rows finder (`rowsComp`, `rowsFinderPrZC`'s: one trial at the position, at `t'` per run of
  the prover, and the binding's collision between a value the trial reads there and the registrant's opening);
* the registered reads' finder `B_reg`, a parameter: `SampledKSLivePub`'s is one play of the audit and the pair the
  registered check's conflict gives (`Registered.ReadsX.pick`).

`B^τ` is their uniform mixture (`Claims.Finder.mix`), `K = card PubIxR` of them, at most `20·J` table slots, two per
committed position and two (`card_PubIxR_le`), so each succeeds at most `K` times as often as `B^τ` (`pubFamR_le`).
`pubFam_le_R`: `pubFinder`'s members are among them. `rowsFinderPrZC_le_mix`: each rows finder's collision probability
is at most what bounds every member.
-/

namespace Flock.Bridge

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden FlockSoundness.LinkFinder
open scoped ENNReal

section FamilyRows

variable {n mPts : ℕ} {Reg : Type} {I : Inputs} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
  {d₀ : Lean.Json} (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀) {C : FlockSoundness.Audit.Circuit}
  (P : FlockSoundness.Audit.Partition C n)
  {decode : Discharge.ZkSession.DecoderZK (C := C) (planZHJ (Reg := Reg) dj y₀) (tabZHJ dj y₀)}
  (vb : ValueBindingZ execArith P (planZHJ dj y₀) (tabZHJ dj y₀) decode H512) {L : Law n}

/-- **Position `c`'s rows finder**, `rowsFinderPrZC`'s: one trial at `c`, at cost `t'` per run of the prover, and the
binding's collision between a value it reads at `c` and the registrant's opening `wo c` (`rowsOutZC`). -/
noncomputable def rowsComp (wo : vb.Pos → vb.Opening) (k Rw : ℕ) (t' : ℝ) (R : Reg)
    (τ : Cont L Reg (batchedSessionZC execArith hs512 (planZHJ dj y₀)) R) (c : LinkPos dj y₀ P vb L) :
    Claims.Finder :=
  haveI := c.2
  Game.tapeFinder (trialGZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb R τ c.1 k Rw)
    (trialStratZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb R τ c.1 k Rw)
    (rowsOutZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb R τ wo c.1)
    (fun x => ⌈cTZC (planZHJ dj y₀) R t' x⌉₊)

variable (L) in
/-- **C-Flock's finders at the registrant's rows**: `PubIx`'s, every rows finder, and the registered reads' finder. -/
abbrev PubIxR : Type :=
  Option ((Fin (nTab I) × TabIx 7) ⊕ LinkPos dj y₀ P vb L ⊕ LinkPos dj y₀ P vb L ⊕ Unit)

variable (L) in
instance instFinitePubIxR : Finite (PubIxR dj y₀ P vb L) := by
  classical
  infer_instance

variable (L) in
noncomputable instance instFintypePubIxR : Fintype (PubIxR dj y₀ P vb L) := Fintype.ofFinite _

variable (L) in
/-- **The number of finders at the registrant's rows**: at most `20` slots per table, two per committed position, and
two. No draw, registration or strategy is in the bound. -/
theorem card_PubIxR_le : Fintype.card (PubIxR dj y₀ P vb L) ≤ 20 * nTab I + 2 * Fintype.card vb.Pos + 2 := by
  classical
  have hT : Nat.card (TabIx 7) = 20 := by
    rw [Nat.card_eq_fintype_card]
    simp only [TabIx, Fintype.card_sum, Fintype.card_prod, Fintype.card_fin, Fintype.card_unit]
  have hL : Nat.card (LinkPos dj y₀ P vb L) ≤ Fintype.card vb.Pos := by
    rw [Nat.card_eq_fintype_card]
    convert Fintype.card_subtype_le _
  rw [Fintype.card_eq_nat_card, Finite.card_option, Nat.card_sum, Nat.card_sum, Nat.card_sum, Nat.card_prod,
    Nat.card_eq_fintype_card (α := Fin (nTab I)), Fintype.card_fin, hT, Nat.card_eq_fintype_card (α := Unit),
    Fintype.card_unit]
  omega

/-- C-Flock's finders at the prover's continuation `τ`, at the registrant's openings `wo`, with the registered reads'
finder `Breg`. -/
noncomputable def pubFamR (wo : vb.Pos → vb.Opening) (k Rw M : ℕ) (qF qS t' : ℝ) (R : Reg)
    (τ : Cont L Reg (batchedSessionZC execArith hs512 (planZHJ dj y₀)) R) (Breg : Claims.Finder) :
    PubIxR dj y₀ P vb L → Claims.Finder
  | none => Claims.Finder.null
  | some (.inl (j, s)) => tabMix dj y₀ qF qS R τ j s
  | some (.inr (.inl c)) => linkComp dj y₀ P vb k Rw M t' R τ c
  | some (.inr (.inr (.inl c))) => rowsComp dj y₀ P vb wo k Rw t' R τ c
  | some (.inr (.inr (.inr _))) => Breg

/-- **`SampledKSLivePub`'s finder `B^τ`**: one of C-Flock's finders at `τ` and the registrant's openings, picked
uniformly with its own coins, run on its own coins (`Claims.Finder.mix`). -/
noncomputable def pubFinderR (wo : vb.Pos → vb.Opening) (k Rw M : ℕ) (qF qS t' : ℝ) (R : Reg)
    (τ : Cont L Reg (batchedSessionZC execArith hs512 (planZHJ dj y₀)) R) (Breg : Claims.Finder) : Claims.Finder :=
  Claims.Finder.mix (pubFamR dj y₀ P vb wo k Rw M qF qS t' R τ Breg)

/-- **Each finder succeeds at most `K` times as often as `B^τ`**, `K = card PubIxR` (`Claims.Finder.le_mix`). -/
theorem pubFamR_le (wo : vb.Pos → vb.Opening) (k Rw M : ℕ) (qF qS t' : ℝ) (R : Reg)
    (τ : Cont L Reg (batchedSessionZC execArith hs512 (planZHJ dj y₀)) R) (Breg : Claims.Finder)
    (i : PubIxR dj y₀ P vb L) :
    (pubFamR dj y₀ P vb wo k Rw M qF qS t' R τ Breg i).Pr Claims.Sampling.sha512Msg ≤
      Fintype.card (PubIxR dj y₀ P vb L) *
        (pubFinderR dj y₀ P vb wo k Rw M qF qS t' R τ Breg).Pr Claims.Sampling.sha512Msg :=
  Claims.Finder.le_mix _ _ i

/-- **`pubFinder`'s members are `pubFinderR`'s**: where every member of `pubFamR` succeeds with probability at most
`p`, so does every member of `pubFam`. -/
theorem pubFam_le_R (wo : vb.Pos → vb.Opening) (k Rw M : ℕ) (qF qS t' : ℝ) (R : Reg)
    (τ : Cont L Reg (batchedSessionZC execArith hs512 (planZHJ dj y₀)) R) (Breg : Claims.Finder) {p : ℝ}
    (hfam : ∀ i, (pubFamR dj y₀ P vb wo k Rw M qF qS t' R τ Breg i).Pr Claims.Sampling.sha512Msg ≤ p)
    (i : PubIx dj y₀ P vb L) : (pubFam dj y₀ P vb k Rw M qF qS t' R τ i).Pr Claims.Sampling.sha512Msg ≤ p := by
  rcases i with _ | ⟨j, s⟩ | c
  · exact hfam none
  · exact hfam (some (.inl (j, s)))
  · exact hfam (some (.inr (.inl c)))

/-- **The rows finders from the family**: where every member succeeds with probability at most `p ≥ 0`, each committed
position's rows finder outputs a collision of `H512` with probability at most `p`. -/
theorem rowsFinderPrZC_le_mix (wo : vb.Pos → vb.Opening) (k Rw M : ℕ) (qF qS t' : ℝ) (R : Reg)
    (τ : Cont L Reg (batchedSessionZC execArith hs512 (planZHJ dj y₀)) R) (Breg : Claims.Finder) {p : ℝ}
    (hp : 0 ≤ p)
    (hfam : ∀ i, (pubFamR dj y₀ P vb wo k Rw M qF qS t' R τ Breg i).Pr Claims.Sampling.sha512Msg ≤ p) (c : vb.Pos) :
    rowsFinderPrZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb wo R τ c k Rw ≤ p := by
  unfold rowsFinderPrZC
  split_ifs with hc
  · have h : (rowsComp dj y₀ P vb wo k Rw t' R τ ⟨c, hc⟩).Pr Claims.Sampling.sha512Msg ≤ p :=
      hfam (some (.inr (.inr (.inl ⟨c, hc⟩))))
    rw [rowsComp, Game.tapeFinder_Pr] at h
    simpa only [Claims.Collision, Hash.Collision, H512_eq_iff] using h
  · exact hp

end FamilyRows

end Flock.Bridge
