import Proofs.Flock.Bridge.KSLive
import Proofs.Flock.Claims.GameFinder
import Proofs.Flock.Soundness.Discharge.FailClosed.ZkHiddenM1

/-!
# `SampledKSLivePub`'s finder `B^τ`: C-Flock's collision finders as one `Claims.Finder`

`SampledKSLivePub` bounds a cheating prover's success by a public error plus a loss in the probability that one finder
built from the prover, `B^τ` (`pubFinder`), outputs a SHA-512 collision. `B^τ` is built from the finders C-Flock's chain
reads, and no others:

* each table's knowledge finders after each draw (`TableCRZ`'s: the rewinding finders, at cost `qF`, and the
  self-clash finder, at cost `qS`) and each committed position's link finder (`linkFinderPrZC`'s, at `t'` per run of
  the prover), each as a `Claims.Finder` on its strategy's coin tape (`Game.tapeFinder`), and the finder that outputs
  nothing;
* `B^τ` is their uniform mixture (`Claims.Finder.mix`): it picks one of the `K = card PubIx` uniformly with its own
  coins and runs it, so each succeeds at most `K` times as often as `B^τ` (`pubFam_le`, from `Claims.Finder.le_mix`).

`tableCRZ_of_mix`: where every member succeeds with probability at most `p`, each table's finders meet `cr/sha-512`'s
strict bound at the budget `√(p·2^513)`, whose `q²/2^513` is `p`. `linkFinderPrZC_le_mix`: each link finder's collision
probability is at most `p`.
-/

namespace Flock.Bridge

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden FlockSoundness.LinkFinder
open Properties.NetworkCertifier.Bytes (bs bs_inj)
open scoped ENNReal

/-- `H512` and the sampling claims' SHA-512 agree on collisions. -/
theorem H512_eq_iff (u v : List UInt8) :
    H512 u = H512 v ↔ Claims.Sampling.sha512Msg u = Claims.Sampling.sha512Msg v := by
  unfold Claims.Sampling.sha512Msg Claims.Sampling.bytes
  constructor
  · intro h
    exact congrArg (fun d : Dig => bs d.1) h
  · intro h
    exact Subtype.ext (bs_inj h)

/-- A strict-time finder of C-Flock's on its tape, at the constant cost `⌈q⌉`. -/
noncomputable def strictComp (Fd : StrictCR.Finder StrictCR.Pair) (q : ℝ) : Claims.Finder :=
  Game.tapeFinder Fd.g Fd.s id (fun _ => ⌈q⌉₊)

/-- **A strict-time finder's premise at a threshold**: where its tape finder outputs a SHA-512 collision with
probability at most `p ≥ 0`, it meets `cr/sha-512`'s strict bound at the budget `√(p·2^513)`, whose `q²/2^513` is `p`. -/
theorem strictCR_of_pr (Fd : StrictCR.Finder StrictCR.Pair) (c : ℝ) {p : ℝ} (hp : 0 ≤ p)
    (hB : (strictComp Fd c).Pr Claims.Sampling.sha512Msg ≤ p) :
    Fd.CR H512 (Real.sqrt (p * 2 ^ (513 : ℝ))) := by
  unfold StrictCR.Finder.CR FlockSoundness.Assumptions.SHA512CRStrict
  intro _
  unfold strictComp at hB
  rw [Game.tapeFinder_Pr] at hB
  have hq : (fun b : StrictCR.Pair => ∃ x y, id b = some (x, y) ∧ x ≠ y ∧ H512 x = H512 y) =
      fun b => ∃ u v, id b = some (u, v) ∧ Claims.Collision Claims.Sampling.sha512Msg u v :=
    funext fun b => by simp only [Claims.Collision, H512_eq_iff]
  rw [hq, Real.sq_sqrt (by positivity), mul_div_assoc, div_self (by positivity), mul_one]
  exact hB

/-- **A table's knowledge finders**: level `ℓ + 1`'s rewinding finder of rep `r`, the pads' and `τ`'s of rep `r`, level
0's and the self-clash finder. -/
abbrev TabIx (levels : ℕ) : Type := (Fin 2 × Fin levels) ⊕ Fin 2 ⊕ Fin 2 ⊕ Unit ⊕ Unit

section Family

variable {n mPts : ℕ} {Reg : Type} {I : Inputs} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
  {d₀ : Lean.Json} (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀) {C : FlockSoundness.Audit.Circuit}
  (P : FlockSoundness.Audit.Partition C n)
  {decode : Discharge.ZkSession.DecoderZK (C := C) (planZHJ (Reg := Reg) dj y₀) (tabZHJ dj y₀)}
  (vb : ValueBindingZ execArith P (planZHJ dj y₀) (tabZHJ dj y₀) decode H512) {L : Law n}

/-- **Table `j`'s knowledge finders after draw `ω`**, each `SessCRZ`'s (through `TableCRZ`), the rewinding ones at cost
`qF` and the self-clash finder at `qS`. -/
noncomputable def tabComp (qF qS : ℝ) (R : Reg) (τ : Cont L Reg (batchedSessionZC execArith hs512 (planZHJ dj y₀)) R)
    (ω : L.Ω) (j : Fin (planZHJ dj y₀ (L.draw ω) R).length) :
    TabIx ((planZHJ dj y₀ (L.draw ω) R).get j).sch.levels.length → Claims.Finder
  | .inl (r, ℓ) => strictComp (advRSFinderZ execArith hs512 ((planZHJ dj y₀ (L.draw ω) R).get j)
      ((gamesZC execArith hs512 (planZHJ dj y₀ (L.draw ω) R)
        (zcStrat execArith hs512 (planZHJ dj y₀) (τ ω)).1).take j)
      ((gamesZC execArith hs512 (planZHJ dj y₀ (L.draw ω) R)
        (zcStrat execArith hs512 (planZHJ dj y₀) (τ ω)).1).drop (j + 1))
      ((zcStrat execArith hs512 (planZHJ dj y₀) (τ ω)).1 j) r.val (ℓ.val + 1)
      (stratZ execArith hs512 (planZHJ dj y₀ (L.draw ω) R) j (zcStrat execArith hs512 (planZHJ dj y₀) (τ ω)))) qF
  | .inr (.inl r) => strictComp (advPadsSFinderZ execArith hs512 ((planZHJ dj y₀ (L.draw ω) R).get j)
      ((gamesZC execArith hs512 (planZHJ dj y₀ (L.draw ω) R)
        (zcStrat execArith hs512 (planZHJ dj y₀) (τ ω)).1).take j)
      ((gamesZC execArith hs512 (planZHJ dj y₀ (L.draw ω) R)
        (zcStrat execArith hs512 (planZHJ dj y₀) (τ ω)).1).drop (j + 1))
      ((zcStrat execArith hs512 (planZHJ dj y₀) (τ ω)).1 j) r.val
      (stratZ execArith hs512 (planZHJ dj y₀ (L.draw ω) R) j (zcStrat execArith hs512 (planZHJ dj y₀) (τ ω)))) qF
  | .inr (.inr (.inl r)) => strictComp (advTauSFinderZ execArith hs512 ((planZHJ dj y₀ (L.draw ω) R).get j)
      ((gamesZC execArith hs512 (planZHJ dj y₀ (L.draw ω) R)
        (zcStrat execArith hs512 (planZHJ dj y₀) (τ ω)).1).take j)
      ((gamesZC execArith hs512 (planZHJ dj y₀ (L.draw ω) R)
        (zcStrat execArith hs512 (planZHJ dj y₀) (τ ω)).1).drop (j + 1))
      ((zcStrat execArith hs512 (planZHJ dj y₀) (τ ω)).1 j) r.val
      (stratZ execArith hs512 (planZHJ dj y₀ (L.draw ω) R) j (zcStrat execArith hs512 (planZHJ dj y₀) (τ ω)))) qF
  | .inr (.inr (.inr (.inl _))) => strictComp (adv₀SFinderZ execArith hs512 ((planZHJ dj y₀ (L.draw ω) R).get j)
      ((gamesZC execArith hs512 (planZHJ dj y₀ (L.draw ω) R)
        (zcStrat execArith hs512 (planZHJ dj y₀) (τ ω)).1).take j)
      ((gamesZC execArith hs512 (planZHJ dj y₀ (L.draw ω) R)
        (zcStrat execArith hs512 (planZHJ dj y₀) (τ ω)).1).drop (j + 1))
      ((zcStrat execArith hs512 (planZHJ dj y₀) (τ ω)).1 j)
      (stratZ execArith hs512 (planZHJ dj y₀ (L.draw ω) R) j (zcStrat execArith hs512 (planZHJ dj y₀) (τ ω)))) qF
  | .inr (.inr (.inr (.inr _))) => strictComp (selfSFinderZ execArith hs512 ((planZHJ dj y₀ (L.draw ω) R).get j)
      ((gamesZC execArith hs512 (planZHJ dj y₀ (L.draw ω) R)
        (zcStrat execArith hs512 (planZHJ dj y₀) (τ ω)).1).take j)
      ((gamesZC execArith hs512 (planZHJ dj y₀ (L.draw ω) R)
        (zcStrat execArith hs512 (planZHJ dj y₀) (τ ω)).1).drop (j + 1))
      ((zcStrat execArith hs512 (planZHJ dj y₀) (τ ω)).1 j)
      (stratZ execArith hs512 (planZHJ dj y₀ (L.draw ω) R) j (zcStrat execArith hs512 (planZHJ dj y₀) (τ ω)))) qS

variable (L) in
/-- The committed positions some draw reads, where a link finder exists. -/
abbrev LinkPos : Type :=
  {c : vb.Pos // Nonempty (ΩcZ execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb L c)}

/-- **Position `c`'s link finder**, `linkFinderPrZC`'s: a trial and up to `M` more, at cost `t'` per run of the
prover. -/
noncomputable def linkComp (k Rw M : ℕ) (t' : ℝ) (R : Reg)
    (τ : Cont L Reg (batchedSessionZC execArith hs512 (planZHJ dj y₀)) R) (c : LinkPos dj y₀ P vb L) :
    Claims.Finder :=
  haveI := c.2
  Game.tapeFinder
    (finderG (trialGZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb R τ c.1 k Rw) M)
    (finderStrat _ (trialStratZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb R τ c.1 k Rw) M)
    (outCZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb R τ c.1)
    (fun x => ⌈finderCost (rdTZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb R τ c.1)
      (cTZC (planZHJ dj y₀) R t') x⌉₊)

variable (L) in
/-- **C-Flock's finders at a registration `R`**: every table's knowledge finders after every draw, every link finder,
and the finder that outputs nothing. -/
abbrev PubIx (R : Reg) : Type :=
  Option ((Σ ω : L.Ω, Σ j : Fin (planZHJ dj y₀ (L.draw ω) R).length,
    TabIx ((planZHJ dj y₀ (L.draw ω) R).get j).sch.levels.length) ⊕ LinkPos dj y₀ P vb L)

variable (L) in
instance instFinitePubIx (R : Reg) : Finite (PubIx dj y₀ P vb L R) := by
  classical
  infer_instance

variable (L) in
noncomputable instance instFintypePubIx (R : Reg) : Fintype (PubIx dj y₀ P vb L R) := Fintype.ofFinite _

/-- C-Flock's finders at the prover's continuation `τ`. -/
noncomputable def pubFam (k Rw M : ℕ) (qF qS t' : ℝ) (R : Reg)
    (τ : Cont L Reg (batchedSessionZC execArith hs512 (planZHJ dj y₀)) R) : PubIx dj y₀ P vb L R → Claims.Finder
  | none => Claims.Finder.null
  | some (.inl ⟨ω, j, i⟩) => tabComp dj y₀ qF qS R τ ω j i
  | some (.inr c) => linkComp dj y₀ P vb k Rw M t' R τ c

/-- **`SampledKSLivePub`'s finder `B^τ`**: one of C-Flock's finders at `τ`, picked uniformly with its own coins, run
on its own coins (`Claims.Finder.mix`). -/
noncomputable def pubFinder (k Rw M : ℕ) (qF qS t' : ℝ) (R : Reg)
    (τ : Cont L Reg (batchedSessionZC execArith hs512 (planZHJ dj y₀)) R) : Claims.Finder :=
  Claims.Finder.mix (pubFam dj y₀ P vb k Rw M qF qS t' R τ)

/-- **Each of C-Flock's finders succeeds at most `K` times as often as `B^τ`**, `K = card PubIx`
(`Claims.Finder.le_mix`). -/
theorem pubFam_le (k Rw M : ℕ) (qF qS t' : ℝ) (R : Reg)
    (τ : Cont L Reg (batchedSessionZC execArith hs512 (planZHJ dj y₀)) R) (i : PubIx dj y₀ P vb L R) :
    (pubFam dj y₀ P vb k Rw M qF qS t' R τ i).Pr Claims.Sampling.sha512Msg ≤
      Fintype.card (PubIx dj y₀ P vb L R) *
        (pubFinder dj y₀ P vb k Rw M qF qS t' R τ).Pr Claims.Sampling.sha512Msg :=
  Claims.Finder.le_mix _ _ i

/-- **The table finders' premise from the family**: where every member succeeds with probability at most `p ≥ 0`, every
table's knowledge finders after every draw meet `cr/sha-512`'s strict bound at the budget `√(p·2^513)`. -/
theorem tableCRZ_of_mix (k Rw M : ℕ) (qF qS t' : ℝ) (R : Reg)
    (τ : Cont L Reg (batchedSessionZC execArith hs512 (planZHJ dj y₀)) R) {p : ℝ} (hp : 0 ≤ p)
    (hfam : ∀ i, (pubFam dj y₀ P vb k Rw M qF qS t' R τ i).Pr Claims.Sampling.sha512Msg ≤ p) (ω : L.Ω)
    (j : Fin (planZHJ dj y₀ (L.draw ω) R).length) :
    TableCRZ execArith hs512 (planZHJ dj y₀ (L.draw ω) R) j (zcStrat execArith hs512 (planZHJ dj y₀) (τ ω))
      (Real.sqrt (p * 2 ^ (513 : ℝ))) (Real.sqrt (p * 2 ^ (513 : ℝ))) := by
  refine ⟨fun r hr ℓ hℓ => ?_, fun r hr => ?_, fun r hr => ?_, ?_, ?_⟩
  · have hr' : r < 2 := Finset.mem_range.1 hr
    obtain ⟨hℓ1, hℓ2⟩ := Finset.mem_Icc.1 hℓ
    obtain ⟨ℓ', rfl⟩ : ∃ ℓ', ℓ = ℓ' + 1 := ⟨ℓ - 1, by omega⟩
    exact strictCR_of_pr _ qF hp (hfam (some (.inl ⟨ω, j, .inl (⟨r, hr'⟩, ⟨ℓ', Nat.lt_of_succ_le hℓ2⟩)⟩)))
  · exact strictCR_of_pr _ qF hp (hfam (some (.inl ⟨ω, j, .inr (.inl ⟨r, Finset.mem_range.1 hr⟩)⟩)))
  · exact strictCR_of_pr _ qF hp (hfam (some (.inl ⟨ω, j, .inr (.inr (.inl ⟨r, Finset.mem_range.1 hr⟩))⟩)))
  · exact strictCR_of_pr _ qF hp (hfam (some (.inl ⟨ω, j, .inr (.inr (.inr (.inl ())))⟩)))
  · exact strictCR_of_pr _ qS hp (hfam (some (.inl ⟨ω, j, .inr (.inr (.inr (.inr ())))⟩)))

/-- **The link finders from the family**: where every member succeeds with probability at most `p ≥ 0`, each
committed position's link finder outputs a collision of `H512` with probability at most `p`. -/
theorem linkFinderPrZC_le_mix (k Rw M : ℕ) (qF qS t' : ℝ) (R : Reg)
    (τ : Cont L Reg (batchedSessionZC execArith hs512 (planZHJ dj y₀)) R) {p : ℝ} (hp : 0 ≤ p)
    (hfam : ∀ i, (pubFam dj y₀ P vb k Rw M qF qS t' R τ i).Pr Claims.Sampling.sha512Msg ≤ p) (c : vb.Pos) :
    linkFinderPrZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb R τ c k Rw M ≤ p := by
  unfold linkFinderPrZC
  split_ifs with hc
  · have h : (linkComp dj y₀ P vb k Rw M t' R τ ⟨c, hc⟩).Pr Claims.Sampling.sha512Msg ≤ p :=
      hfam (some (.inr ⟨c, hc⟩))
    rw [linkComp, Game.tapeFinder_Pr] at h
    simpa only [Claims.Collision, H512_eq_iff] using h
  · exact hp

end Family

end Flock.Bridge
