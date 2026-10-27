import Proofs.Properties.Verity.SampledDeploy
import Proofs.Flock.Claims.GameFinder
import Proofs.Flock.Soundness.CROnly.Link

/-!
# Sampled proofs' finder: C-Flock's collision finders as one `Claims.Finder`

`SampledDeploy.cr` asks that the vocabulary's finder premise (`Vocab.FinderCR .sampled`: the finder `B^τ` outputs a
SHA-512 collision with probability at most the published `ε_cr` at the strategy's calls) give C-Flock's
(`LiveAt.CR τ`: each of the finders C-Flock builds from `τ` meets its own premise). `B^τ` is built from those finders,
the ones `LiveAt.CR` names (`TableCRZAt`, `LinkCRZCAt`), and no others:

* each table's knowledge finders after each draw (`SessCRZ`'s: the rewinding finders at `qF τ = 2 (t' τ + v)`, the
  self-clash finder at `qS τ = t' τ + v`), and each committed position's link finder (`LinkCRZC`'s), each as a
  `Claims.Finder` on its strategy's coin tape (`Game.tapeFinder`, with the finder's success);
* `B^τ` (`LiveAt.crFinder`) is their sequential composition (`Claims.Finder.seq`): it runs each of them on its own
  coins in a fixed order and outputs the first SHA-512 collision found. It succeeds at least as often as each of them,
  and costs their sum plus two SHA-512 calls a check (`crFinder_cost_le`).

`LiveAt.CR_of_crFinder`: where `B^τ` succeeds with probability at most `β`, and `β` is at most each kind's bound
(`qF τ²/2^513`, `qS τ²/2^513`, and `t' τ / 2^256` for the link finders), `LiveAt.CR τ` holds. A link finder plays `τ`
at least once, so its expected cost is at least `t' τ` (`linkCost_ge`). `Verity.SampledDeploy.cr_of` gives
`SampledDeploy`'s `cr` field at a vocabulary whose sampled finder is `crFinder` at the parse, with the three
inequalities at `β = crBound θ .sampled (cost σ)`.
-/

namespace Flock.Guarantees.LiveAt

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden FlockSoundness.LinkFinder
open scoped ENNReal

/-- `H512` and the compiled SHA-512 agree on collisions. -/
theorem H512_eq_iff (u v : List UInt8) : H512 u = H512 v ↔ Verity.sha512Msg u = Verity.sha512Msg v := by
  constructor
  · intro h
    exact congrArg (fun d : Dig => d.1.data.toList) h
  · intro h
    have hd : (Flock.Sha512.hash ⟨u.toArray⟩).data = (Flock.Sha512.hash ⟨v.toArray⟩).data := Array.toList_inj.mp h
    exact Subtype.ext (congrArg ByteArray.mk hd)

/-- A strict-time finder of C-Flock's on its tape, at the constant cost `⌈q⌉`. -/
noncomputable def strictComp (Fd : StrictCR.Finder StrictCR.Pair) (q : ℝ) : Claims.Finder :=
  Game.tapeFinder Fd.g Fd.s id (fun _ => ⌈q⌉₊)

/-- **A strict-time finder's premise from the threshold**: where its tape finder succeeds with probability at most
`β ≤ q²/2^513`, C-Flock's finder meets `SHA512CRStrict` at the budget `q`. -/
theorem strictCR_of (Fd : StrictCR.Finder StrictCR.Pair) (q : ℝ) {β : ℝ}
    (hB : (strictComp Fd q).Pr Verity.sha512Msg ≤ β) (hb : β ≤ q ^ 2 / 2 ^ (513 : ℝ)) :
    Fd.CR H512 q := by
  unfold StrictCR.Finder.CR Assumptions.SHA512CRStrict
  intro _
  unfold strictComp at hB
  rw [Game.tapeFinder_Pr] at hB
  have hp : (fun b : StrictCR.Pair => ∃ x y, id b = some (x, y) ∧ x ≠ y ∧ H512 x = H512 y) =
      fun b => ∃ u v, id b = some (u, v) ∧ Claims.Collision Verity.sha512Msg u v :=
    funext fun b => by simp only [Claims.Collision, H512_eq_iff]
  rw [hp]
  exact hB.trans hb

/-- **An expected-time finder's premise from the threshold**: where every outcome costs at least `t` and its tape
finder succeeds with probability at most `β ≤ t/2^256`, `SHA512CRExpected` holds. -/
theorem expectedCR_of {γ : Type} (g : Game γ) (s : Strategy g) (out : γ → Option (Claims.Msg × Claims.Msg))
    (cost : γ → ℝ) (cst : γ → ℕ) {t β : ℝ} (hcost : ∀ x, t ≤ cost x)
    (hB : (Game.tapeFinder g s out cst).Pr Verity.sha512Msg ≤ β) (hb : β ≤ t / 2 ^ (256 : ℝ)) :
    Assumptions.SHA512CRExpected H512 g s out cost := by
  unfold Assumptions.SHA512CRExpected
  rw [Game.tapeFinder_Pr] at hB
  have hp : (fun b : γ => ∃ x y, out b = some (x, y) ∧ x ≠ y ∧ H512 x = H512 y) =
      fun b => ∃ u v, out b = some (u, v) ∧ Claims.Collision Verity.sha512Msg u v :=
    funext fun b => by simp only [Claims.Collision, H512_eq_iff]
  rw [hp]
  have ht : t ≤ expect cost g s := by
    have := expect_mono hcost g s
    rwa [expect_const] at this
  exact (hB.trans hb).trans (div_le_div_of_nonneg_right ht (by positivity))

variable {n : ℕ} {D : Flock.Desc} {hU : Built D.unroll.unit} {kd K₀ : ℕ} {desc : Claims.Msg}
  (A : LiveAt n D hU kd K₀ desc)

/-- The plan after draw `ω`, at `τ`. -/
noncomputable abbrev tsAt (τ : Strategy A.G) (ω : A.L.Ω) := planZHJ A.dj A.y₀ (A.L.draw ω) (reg (A.sim τ))

/-- The session strategy after draw `ω`, at `τ`. -/
noncomputable abbrev zcAt (τ : Strategy A.G) (ω : A.L.Ω) :=
  zcStrat execArith hs512 (planZHJ A.dj A.y₀) (cont (A.sim τ) ω)

/-- **A table's knowledge finders**: level `ℓ + 1`'s rewinding finder of rep `r`, the pads' and `τ`'s of rep `r`,
level 0's and the self-clash finder. -/
abbrev TabIx (levels : ℕ) : Type := (Fin 2 × Fin levels) ⊕ Fin 2 ⊕ Fin 2 ⊕ Unit ⊕ Unit

/-- **Table `j`'s knowledge finders after draw `ω`**, each `SessCRZ`'s (through `TableCRZ`). -/
noncomputable def tabComp (τ : Strategy A.G) (ω : A.L.Ω) (j : Fin (A.tsAt τ ω).length) :
    TabIx ((A.tsAt τ ω).get j).sch.levels.length → Claims.Finder
  | .inl (r, ℓ) => strictComp (advRSFinderZ execArith hs512 ((A.tsAt τ ω).get j)
      ((gamesZC execArith hs512 (A.tsAt τ ω) (A.zcAt τ ω).1).take j)
      ((gamesZC execArith hs512 (A.tsAt τ ω) (A.zcAt τ ω).1).drop (j + 1)) ((A.zcAt τ ω).1 j) r.val (ℓ.val + 1)
      (stratZ execArith hs512 (A.tsAt τ ω) j (A.zcAt τ ω))) (A.qF τ)
  | .inr (.inl r) => strictComp (advPadsSFinderZ execArith hs512 ((A.tsAt τ ω).get j)
      ((gamesZC execArith hs512 (A.tsAt τ ω) (A.zcAt τ ω).1).take j)
      ((gamesZC execArith hs512 (A.tsAt τ ω) (A.zcAt τ ω).1).drop (j + 1)) ((A.zcAt τ ω).1 j) r.val
      (stratZ execArith hs512 (A.tsAt τ ω) j (A.zcAt τ ω))) (A.qF τ)
  | .inr (.inr (.inl r)) => strictComp (advTauSFinderZ execArith hs512 ((A.tsAt τ ω).get j)
      ((gamesZC execArith hs512 (A.tsAt τ ω) (A.zcAt τ ω).1).take j)
      ((gamesZC execArith hs512 (A.tsAt τ ω) (A.zcAt τ ω).1).drop (j + 1)) ((A.zcAt τ ω).1 j) r.val
      (stratZ execArith hs512 (A.tsAt τ ω) j (A.zcAt τ ω))) (A.qF τ)
  | .inr (.inr (.inr (.inl _))) => strictComp (adv₀SFinderZ execArith hs512 ((A.tsAt τ ω).get j)
      ((gamesZC execArith hs512 (A.tsAt τ ω) (A.zcAt τ ω).1).take j)
      ((gamesZC execArith hs512 (A.tsAt τ ω) (A.zcAt τ ω).1).drop (j + 1)) ((A.zcAt τ ω).1 j)
      (stratZ execArith hs512 (A.tsAt τ ω) j (A.zcAt τ ω))) (A.qF τ)
  | .inr (.inr (.inr (.inr _))) => strictComp (selfSFinderZ execArith hs512 ((A.tsAt τ ω).get j)
      ((gamesZC execArith hs512 (A.tsAt τ ω) (A.zcAt τ ω).1).take j)
      ((gamesZC execArith hs512 (A.tsAt τ ω) (A.zcAt τ ω).1).drop (j + 1)) ((A.zcAt τ ω).1 j)
      (stratZ execArith hs512 (A.tsAt τ ω) j (A.zcAt τ ω))) (A.qS τ)

/-- The committed positions some draw reads, where a link finder exists. -/
abbrev LinkPos : Type :=
  {c : A.vb.Pos // Nonempty (ΩcZ execArith A.P (planZHJ A.dj A.y₀) (tabZHJ A.dj A.y₀) A.vb A.L c)}

/-- **Position `c`'s link finder**, `LinkCRZC`'s: a trial and up to `M` more, at cost `t' τ` per run of `τ`. -/
noncomputable def linkComp (τ : Strategy A.G) (c : A.LinkPos) : Claims.Finder :=
  haveI := c.2
  (Game.tapeFinder
    (finderG (trialGZC execArith hs512 A.P (planZHJ A.dj A.y₀) (tabZHJ A.dj A.y₀) A.vb (reg (A.sim τ))
      (cont (A.sim τ)) c.1 A.k A.Rw) A.M)
    (finderStrat _ (trialStratZC execArith hs512 A.P (planZHJ A.dj A.y₀) (tabZHJ A.dj A.y₀) A.vb (reg (A.sim τ))
      (cont (A.sim τ)) c.1 A.k A.Rw) A.M)
    (outCZC execArith hs512 A.P (planZHJ A.dj A.y₀) (tabZHJ A.dj A.y₀) A.vb (reg (A.sim τ)) (cont (A.sim τ)) c.1)
    (fun x => ⌈finderCost (rdTZC execArith hs512 A.P (planZHJ A.dj A.y₀) (tabZHJ A.dj A.y₀) A.vb (reg (A.sim τ))
      (cont (A.sim τ)) c.1) (cTZC (planZHJ A.dj A.y₀) (reg (A.sim τ)) (A.t' τ)) x⌉₊)

/-- **C-Flock's finders at `τ`**: every table's knowledge finders after every draw, and every link finder. -/
abbrev Ix (τ : Strategy A.G) : Type :=
  (Σ ω : A.L.Ω, Σ j : Fin (A.tsAt τ ω).length, TabIx ((A.tsAt τ ω).get j).sch.levels.length) ⊕ A.LinkPos

instance instFiniteIx (τ : Strategy A.G) : Finite (A.Ix τ) := by
  classical
  infer_instance

noncomputable instance instFintypeIx (τ : Strategy A.G) : Fintype (A.Ix τ) := Fintype.ofFinite _

/-- C-Flock's finders at `τ`. -/
noncomputable def crFam (τ : Strategy A.G) : A.Ix τ → Claims.Finder
  | .inl ⟨ω, j, i⟩ => A.tabComp τ ω j i
  | .inr c => A.linkComp τ c

/-- **Sampled proofs' finder `B^τ`**: C-Flock's finders at `τ` run in turn, each on its own coins, outputting the first
SHA-512 collision found. -/
noncomputable def crFinder (τ : Strategy A.G) : Claims.Finder :=
  Claims.Finder.seq Verity.sha512Msg (A.crFam τ)

theorem strictComp_cost_le (Fd : StrictCR.Finder StrictCR.Pair) (q : ℝ) {b : ℕ} (h : ⌈q⌉₊ ≤ b)
    (x : (strictComp Fd q).S) : (strictComp Fd q).cost x ≤ b :=
  Game.tapeFinder_cost_le _ _ _ _ b (fun _ => h) x

/-- **`B^τ`'s cost on every coin** is at most `b + 2` per finder at `τ` where the rewinding and self-clash budgets are
at most `b` and every link finder's run is. -/
theorem crFinder_cost_le (τ : Strategy A.G) {b : ℕ} (hF : ⌈A.qF τ⌉₊ ≤ b) (hS : ⌈A.qS τ⌉₊ ≤ b)
    (hL : ∀ (c : A.vb.Pos) (x : TrialOutZC FlockLevel3.GF128 A.L Dig (List UInt8) ×
        List (TrialOutZC FlockLevel3.GF128 A.L Dig (List UInt8))),
      finderCost (rdTZC execArith hs512 A.P (planZHJ A.dj A.y₀) (tabZHJ A.dj A.y₀) A.vb (reg (A.sim τ))
        (cont (A.sim τ)) c) (cTZC (planZHJ A.dj A.y₀) (reg (A.sim τ)) (A.t' τ)) x ≤ b)
    (x : (A.crFinder τ).S) : (A.crFinder τ).cost x ≤ Fintype.card (A.Ix τ) * (b + 2) := by
  refine Claims.Finder.seq_cost_le _ _ b (fun i => ?_) x
  rcases i with ⟨ω, j, ⟨r, ℓ⟩ | r | r | _ | _⟩ | c
  · exact strictComp_cost_le _ _ hF
  · exact strictComp_cost_le _ _ hF
  · exact strictComp_cost_le _ _ hF
  · exact strictComp_cost_le _ _ hF
  · exact strictComp_cost_le _ _ hS
  · exact Game.tapeFinder_cost_le _ _ _ _ b (fun _ => Nat.ceil_le.mpr (hL c.1 _))

/-- A link trial plays `τ` at least once: its cost is at least `t' τ`. -/
theorem cTZC_ge (τ : Strategy A.G) (e : TrialOutZC FlockLevel3.GF128 A.L Dig (List UInt8)) :
    A.t' τ ≤ cTZC (planZHJ A.dj A.y₀) (reg (A.sim τ)) (A.t' τ) e := by
  unfold cTZC
  have h0 := A.ht' τ
  split_ifs
  · refine le_mul_of_one_le_right h0 (le_add_of_nonneg_right (List.sum_nonneg ?_))
    simp only [List.mem_map]
    rintro _ ⟨b, -, rfl⟩
    exact Extract.waitCost_nonneg _ b
  · simp

/-- **A link finder's cost is at least one run's**, `t' τ`. -/
theorem linkCost_ge (τ : Strategy A.G) (c : A.vb.Pos) (x : TrialOutZC FlockLevel3.GF128 A.L Dig (List UInt8) ×
    List (TrialOutZC FlockLevel3.GF128 A.L Dig (List UInt8))) :
    A.t' τ ≤ finderCost (rdTZC execArith hs512 A.P (planZHJ A.dj A.y₀) (tabZHJ A.dj A.y₀) A.vb (reg (A.sim τ))
      (cont (A.sim τ)) c) (cTZC (planZHJ A.dj A.y₀) (reg (A.sim τ)) (A.t' τ)) x := by
  have h1 := A.cTZC_ge τ x.1
  unfold finderCost
  split_ifs
  · exact le_add_of_le_of_nonneg h1 (CROnly.loopCost_nonneg _ (fun e => (A.ht' τ).trans (A.cTZC_ge τ e)) _)
  · simpa using h1

/-- **C-Flock's premises from the threshold.** Where `B^τ` (`crFinder τ`) outputs a SHA-512 collision with probability
at most `β`, and `β` is at most each kind's bound (`qF τ²/2^513` for the rewinding finders, `qS τ²/2^513` for the
self-clash finder, `t' τ/2^256` for the link finders), every finder `LiveAt.CR τ` names meets its premise. -/
theorem CR_of_crFinder (τ : Strategy A.G) {β : ℝ}
    (hB : (A.crFinder τ).Pr Verity.sha512Msg ≤ β)
    (hbF : β ≤ A.qF τ ^ 2 / 2 ^ (513 : ℝ)) (hbS : β ≤ A.qS τ ^ 2 / 2 ^ (513 : ℝ))
    (hbL : β ≤ A.t' τ / 2 ^ (256 : ℝ)) :
    A.CR τ := by
  have hfam : ∀ i, (A.crFam τ i).Pr Verity.sha512Msg ≤ β := fun i =>
    (Claims.Finder.le_seq Verity.sha512Msg (A.crFam τ) i).trans hB
  unfold Flock.Guarantees.LiveAt.CR
  refine ⟨fun ω j => ⟨fun r hr ℓ hℓ => ?_, fun r hr => ?_, fun r hr => ?_, ?_, ?_⟩, ?_⟩
  · have hr' : r < 2 := Finset.mem_range.1 hr
    obtain ⟨hℓ1, hℓ2⟩ := Finset.mem_Icc.1 hℓ
    obtain ⟨ℓ', rfl⟩ : ∃ ℓ', ℓ = ℓ' + 1 := ⟨ℓ - 1, by omega⟩
    exact strictCR_of _ _ (hfam (.inl ⟨ω, j, .inl (⟨r, hr'⟩, ⟨ℓ', Nat.lt_of_succ_le hℓ2⟩)⟩)) hbF
  · exact strictCR_of _ _ (hfam (.inl ⟨ω, j, .inr (.inl ⟨r, Finset.mem_range.1 hr⟩)⟩)) hbF
  · exact strictCR_of _ _ (hfam (.inl ⟨ω, j, .inr (.inr (.inl ⟨r, Finset.mem_range.1 hr⟩))⟩)) hbF
  · exact strictCR_of _ _ (hfam (.inl ⟨ω, j, .inr (.inr (.inr (.inl ())))⟩)) hbF
  · exact strictCR_of _ _ (hfam (.inl ⟨ω, j, .inr (.inr (.inr (.inr ())))⟩)) hbS
  · intro c hc
    exact expectedCR_of _ _ _ _ _ (A.linkCost_ge τ c) (hfam (.inr ⟨c, hc⟩)) hbL

end Flock.Guarantees.LiveAt

namespace Verity

open Accounting Core

variable {ι α : Type} [Fintype α] {d : Device ι}

/-- **`SampledDeploy`'s `cr` field.** At a vocabulary whose sampled finder, at registered files that parse, is
`crFinder` at the session's strategy (`hfinder`), and whose published `ε_cr` at the strategy's calls is at most each
kind's bound (`hbF`, `hbS`, `hbL`), the vocabulary's finder premise gives C-Flock's. -/
theorem SampledDeploy.cr_of (V : Vocab d α) (kd : ByteArray → ℕ) (desc : ByteArray → Claims.Msg)
    (A : ∀ (θ F c : ByteArray) (_σ : Game.Strategy (V.game θ F c)) (r : Flock.Bridge.Parsed), V.parsed F = some r →
      Flock.Guarantees.LiveAt r.n r.D r.hU (kd θ) (⌊(V.certOf θ).ε * r.n⌋₊ + 1) (desc θ))
    (τOf : ∀ (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)) (r : Flock.Bridge.Parsed)
      (hr : V.parsed F = some r), FlockSoundness.Game.Strategy (A θ F c σ r hr).G)
    (hfinder : ∀ θ F c σ r hr, V.Registered θ F →
      V.finder .sampled θ F c σ = (A θ F c σ r hr).crFinder (τOf θ F c σ r hr))
    (hbF : ∀ θ F c σ r hr, V.crBound θ .sampled (V.cost θ F c σ) ≤
      (A θ F c σ r hr).qF (τOf θ F c σ r hr) ^ 2 / 2 ^ (513 : ℝ))
    (hbS : ∀ θ F c σ r hr, V.crBound θ .sampled (V.cost θ F c σ) ≤
      (A θ F c σ r hr).qS (τOf θ F c σ r hr) ^ 2 / 2 ^ (513 : ℝ))
    (hbL : ∀ θ F c σ r hr, V.crBound θ .sampled (V.cost θ F c σ) ≤
      (A θ F c σ r hr).t' (τOf θ F c σ r hr) / 2 ^ (256 : ℝ)) :
    ∀ (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)) (r : Flock.Bridge.Parsed)
      (hr : V.parsed F = some r), V.Registered θ F →
      V.FinderCR .sampled θ F c σ → (A θ F c σ r hr).CR (τOf θ F c σ r hr) := by
  intro θ F c σ r hr hreg hcr
  have hB : ((A θ F c σ r hr).crFinder (τOf θ F c σ r hr)).Pr sha512Msg ≤
      V.crBound θ .sampled (V.cost θ F c σ) := by
    rw [← hfinder θ F c σ r hr hreg]
    exact hcr
  exact (A θ F c σ r hr).CR_of_crFinder _ hB (hbF θ F c σ r hr) (hbS θ F c σ r hr) (hbL θ F c σ r hr)

end Verity
