import Proofs.Flock.Bridge.KSLive
import Proofs.Flock.Claims.GameFinder
import Proofs.Flock.Soundness.Discharge.FailClosed.ZkHiddenM1

/-!
# `SampledKSLivePub`'s finder `B^τ`: C-Flock's collision finders as one `Claims.Finder`

`SampledKSLivePub` bounds a cheating prover's success by a public error plus a loss in the probability that one finder
built from the prover, `B^τ` (`pubFinder`), outputs a SHA-512 collision. `B^τ` is built from the finders C-Flock's chain
reads, and no others:

* each table's knowledge finders (`TableCRZ`'s: the rewinding finders, at cost `qF`, and the self-clash finder, at cost
  `qS`), indexed by the table `j < J` (`J = nTab I`) and the finder's slot, not by the draw: slot `s` of table `j`
  draws the verifier's draw `ω` uniformly with its own coins and runs table `j`'s slot-`s` finder after `ω`
  (`tabMix`), or outputs nothing where that table has no such slot (`tabAt`);
* each committed position's link finder (`linkFinderPrZC`'s, at `t'` per run of the prover);
* the finder that outputs nothing.

Each is a `Claims.Finder` on its strategy's coin tape (`Game.tapeFinder`). `B^τ` is their uniform mixture
(`Claims.Finder.mix`): it picks one of the `K = card PubIx` uniformly with its own coins and runs it, so each succeeds
at most `K` times as often as `B^τ` (`pubFam_le`, from `Claims.Finder.le_mix`). `K` counts no draw: at most `20·J` table
slots, the committed positions and one (`card_PubIx_le`).

`tabAvg_of_mix`: where every member succeeds with probability at most `p`, each table slot's finder succeeds with
probability at most `p` averaged over the draw (`Claims.Finder.mix_Pr`), the averaged premise `Bridge.PubAvg` reads.
`linkFinderPrZC_le_mix`: each link finder's collision probability is at most `p`.
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
  have e : ∀ m, Claims.Sampling.sha512Msg m = bs (H512 m).1 := fun m => by
    unfold Claims.Sampling.sha512Msg Claims.Sampling.bytes
    rw [H512_val]
  rw [e u, e v]
  constructor
  · intro h
    rw [h]
  · intro h
    exact Subtype.ext (bs_inj h)

/-- A strict-time finder of C-Flock's on its tape, at the constant cost `⌈q⌉`. -/
noncomputable def strictComp (Fd : StrictCR.Finder StrictCR.Pair) (q : ℝ) : Claims.Finder :=
  Game.tapeFinder Fd.g Fd.s id (fun _ => ⌈q⌉₊)

/-- A finder's output is a SHA-512 collision exactly when it is one of `H512`. -/
theorem isColl_sha512 :
    (fun b : StrictCR.Pair => ∃ u v, (id : StrictCR.Pair → Option (Claims.Msg × Claims.Msg)) b = some (u, v) ∧
      Claims.Collision Claims.Sampling.sha512Msg u v) = StrictCR.IsColl H512 := by
  funext b
  simp only [StrictCR.IsColl, Claims.Collision, Hash.Collision, H512_eq_iff, id]

/-- **A strict-time finder succeeds as its tape finder does**: its probability of outputting a collision of `H512` is
its tape finder's of outputting a SHA-512 collision. -/
theorem strictComp_Pr (Fd : StrictCR.Finder StrictCR.Pair) (c : ℝ) :
    (strictComp Fd c).Pr Claims.Sampling.sha512Msg = Fd.ev (Game.ind (StrictCR.IsColl H512)) := by
  unfold strictComp
  rw [Game.tapeFinder_Pr, StrictCR.Finder.ev, ← Game.prob_toReal]
  -- `congr` here would try to close the predicates' equality by unfolding SHA-512.
  exact congrArg (fun q => (Game.prob q Fd.g Fd.s).toReal) isColl_sha512

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

/-- **Slot `s` in a table of `l` rewinding levels**: `s` itself, where the table has it. -/
def slotAt {m : ℕ} (l : ℕ) : TabIx m → Option (TabIx l)
  | .inl (r, ℓ) => if h : ℓ.val < l then some (.inl (r, ⟨ℓ.val, h⟩)) else none
  | .inr x => some (.inr x)

/-- Table `j` of the plan after draw `ω`. -/
noncomputable abbrev tabOf (R : Reg) (ω : L.Ω) (j : Fin (nTab I)) : Fin (planZHJ dj y₀ (L.draw ω) R).length :=
  Fin.cast (planZHJ_length dj y₀ (L.draw ω) R).symm j

/-- **Table `j`'s slot-`s` knowledge finder after draw `ω`**: `tabComp`'s, where table `j` has slot `s` (C-Flock ZK's
tables have at most 7 rewinding levels), else the finder that outputs nothing. -/
noncomputable def tabAt (qF qS : ℝ) (R : Reg) (τ : Cont L Reg (batchedSessionZC execArith hs512 (planZHJ dj y₀)) R)
    (ω : L.Ω) (j : Fin (nTab I)) (s : TabIx 7) : Claims.Finder :=
  match slotAt ((planZHJ dj y₀ (L.draw ω) R).get (tabOf dj y₀ R ω j)).sch.levels.length s with
  | some s' => tabComp dj y₀ qF qS R τ ω (tabOf dj y₀ R ω j) s'
  | none => Claims.Finder.null

theorem tabAt_inl (qF qS : ℝ) (R : Reg) (τ : Cont L Reg (batchedSessionZC execArith hs512 (planZHJ dj y₀)) R)
    (ω : L.Ω) (j : Fin (nTab I)) (r : Fin 2) (ℓ : Fin 7)
    (h : ℓ.val < ((planZHJ dj y₀ (L.draw ω) R).get (tabOf dj y₀ R ω j)).sch.levels.length) :
    tabAt dj y₀ qF qS R τ ω j (.inl (r, ℓ)) =
      tabComp dj y₀ qF qS R τ ω (tabOf dj y₀ R ω j) (.inl (r, ⟨ℓ.val, h⟩)) := by
  simp only [tabAt, slotAt, dif_pos h]

theorem tabAt_inr (qF qS : ℝ) (R : Reg) (τ : Cont L Reg (batchedSessionZC execArith hs512 (planZHJ dj y₀)) R)
    (ω : L.Ω) (j : Fin (nTab I)) (x : Fin 2 ⊕ Fin 2 ⊕ Unit ⊕ Unit) :
    tabAt dj y₀ qF qS R τ ω j (.inr x) = tabComp dj y₀ qF qS R τ ω (tabOf dj y₀ R ω j) (.inr x) := by
  simp only [tabAt, slotAt]

/-- **Table `j`'s slot-`s` finder over the draw**: it draws `ω` uniformly with its own coins and runs `tabAt`'s. -/
noncomputable def tabMix (qF qS : ℝ) (R : Reg) (τ : Cont L Reg (batchedSessionZC execArith hs512 (planZHJ dj y₀)) R)
    (j : Fin (nTab I)) (s : TabIx 7) : Claims.Finder :=
  Claims.Finder.mix fun ω : L.Ω => tabAt dj y₀ qF qS R τ ω j s

/-- `tabMix` succeeds with `tabAt`'s probability averaged over the draw (`Claims.Finder.mix_Pr`). -/
theorem tabMix_Pr (qF qS : ℝ) (R : Reg) (τ : Cont L Reg (batchedSessionZC execArith hs512 (planZHJ dj y₀)) R)
    (j : Fin (nTab I)) (s : TabIx 7) :
    (tabMix dj y₀ qF qS R τ j s).Pr Claims.Sampling.sha512Msg =
      Claims.Ex fun ω : L.Ω => (tabAt dj y₀ qF qS R τ ω j s).Pr Claims.Sampling.sha512Msg :=
  Claims.Finder.mix_Pr _ _

variable (L) in
/-- **C-Flock's finders**: each table's knowledge finder slots, every link finder, and the finder that outputs
nothing. It indexes no draw and no registration. -/
abbrev PubIx : Type := Option ((Fin (nTab I) × TabIx 7) ⊕ LinkPos dj y₀ P vb L)

variable (L) in
instance instFinitePubIx : Finite (PubIx dj y₀ P vb L) := by
  classical
  infer_instance

variable (L) in
noncomputable instance instFintypePubIx : Fintype (PubIx dj y₀ P vb L) := Fintype.ofFinite _

variable (L) in
/-- **The number of C-Flock's finders**: `K = card PubIx` is at most `20` slots per table, one per committed position,
and one. No draw, registration or strategy is in the bound. -/
theorem card_PubIx_le : Fintype.card (PubIx dj y₀ P vb L) ≤ 20 * nTab I + Fintype.card vb.Pos + 1 := by
  classical
  have hT : Nat.card (TabIx 7) = 20 := by
    rw [Nat.card_eq_fintype_card]
    simp only [TabIx, Fintype.card_sum, Fintype.card_prod, Fintype.card_fin, Fintype.card_unit]
  have hL : Nat.card (LinkPos dj y₀ P vb L) ≤ Fintype.card vb.Pos := by
    rw [Nat.card_eq_fintype_card]
    convert Fintype.card_subtype_le _
  rw [Fintype.card_eq_nat_card, Finite.card_option, Nat.card_sum, Nat.card_prod,
    Nat.card_eq_fintype_card (α := Fin (nTab I)), Fintype.card_fin, hT]
  omega

/-- C-Flock's finders at the prover's continuation `τ`. -/
noncomputable def pubFam (k Rw M : ℕ) (qF qS t' : ℝ) (R : Reg)
    (τ : Cont L Reg (batchedSessionZC execArith hs512 (planZHJ dj y₀)) R) : PubIx dj y₀ P vb L → Claims.Finder
  | none => Claims.Finder.null
  | some (.inl (j, s)) => tabMix dj y₀ qF qS R τ j s
  | some (.inr c) => linkComp dj y₀ P vb k Rw M t' R τ c

/-- **`SampledKSLivePub`'s finder `B^τ`**: one of C-Flock's finders at `τ`, picked uniformly with its own coins, run
on its own coins (`Claims.Finder.mix`). -/
noncomputable def pubFinder (k Rw M : ℕ) (qF qS t' : ℝ) (R : Reg)
    (τ : Cont L Reg (batchedSessionZC execArith hs512 (planZHJ dj y₀)) R) : Claims.Finder :=
  Claims.Finder.mix (pubFam dj y₀ P vb k Rw M qF qS t' R τ)

/-- **Each of C-Flock's finders succeeds at most `K` times as often as `B^τ`**, `K = card PubIx`
(`Claims.Finder.le_mix`). -/
theorem pubFam_le (k Rw M : ℕ) (qF qS t' : ℝ) (R : Reg)
    (τ : Cont L Reg (batchedSessionZC execArith hs512 (planZHJ dj y₀)) R) (i : PubIx dj y₀ P vb L) :
    (pubFam dj y₀ P vb k Rw M qF qS t' R τ i).Pr Claims.Sampling.sha512Msg ≤
      Fintype.card (PubIx dj y₀ P vb L) *
        (pubFinder dj y₀ P vb k Rw M qF qS t' R τ).Pr Claims.Sampling.sha512Msg :=
  Claims.Finder.le_mix _ _ i

/-- **The table slots' averaged premise from the family**: where every member succeeds with probability at most `p`,
each table slot's finder succeeds with probability at most `p` averaged over the draw (`tabMix_Pr`). -/
theorem tabAvg_of_mix (k Rw M : ℕ) (qF qS t' : ℝ) (R : Reg)
    (τ : Cont L Reg (batchedSessionZC execArith hs512 (planZHJ dj y₀)) R) {p : ℝ}
    (hfam : ∀ i, (pubFam dj y₀ P vb k Rw M qF qS t' R τ i).Pr Claims.Sampling.sha512Msg ≤ p) (j : Fin (nTab I))
    (s : TabIx 7) :
    Claims.Ex (fun ω : L.Ω => (tabAt dj y₀ qF qS R τ ω j s).Pr Claims.Sampling.sha512Msg) ≤ p :=
  (tabMix_Pr dj y₀ qF qS R τ j s).symm.trans_le (hfam (some (.inl (j, s))))

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
    simpa only [Claims.Collision, Hash.Collision, H512_eq_iff] using h
  · exact hp

end Family

end Flock.Bridge
