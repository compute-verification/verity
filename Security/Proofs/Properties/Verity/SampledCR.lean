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
* each diluted (`Claims.Finder.dilute`) by its kind's weight `m / (N + 1)` (`Weights`), so that one threshold carries
  the three premises (`q²/2^513` at two budgets, and the expected cost over `2^256`);
* `B^τ` (`LiveAt.crFinder`) is the one among them, with the finder that outputs nothing, of largest success
  (`Claims.Finder.worst`).

`LiveAt.CR_of_crFinder`: where `B^τ` succeeds with probability at most `β` and `(N + 1) β ≤ m · bound` for each kind,
`LiveAt.CR τ` holds. A link finder plays `τ` at least once, so its expected cost is at least `t' τ`
(`linkCost_ge`), and `t' τ / 2^256` is the link kind's bound. `Verity.SampledDeploy.cr_of` gives `SampledDeploy`'s
`cr` field at a vocabulary whose sampled finder is `crFinder` at the parse, with the three inequalities at
`β = crBound θ .sampled (cost σ)`.
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

/-- **The dilution of each kind of finder**, `m` in `N + 1`: the rewinding finders' (`F`), the self-clash finder's
(`S`) and the link finders' (`L`). -/
structure Weights where
  F : ℕ × ℕ
  S : ℕ × ℕ
  L : ℕ × ℕ
  hF : 0 < F.1 ∧ F.1 ≤ F.2 + 1
  hS : 0 < S.1 ∧ S.1 ≤ S.2 + 1
  hL : 0 < L.1 ∧ L.1 ≤ L.2 + 1

/-- A strict-time finder of C-Flock's on its tape, at the constant cost `⌈q⌉`, diluted by `w`. -/
noncomputable def strictFinder (Fd : StrictCR.Finder StrictCR.Pair) (q : ℝ) (w : ℕ × ℕ) : Claims.Finder :=
  (Game.tapeFinder Fd.g Fd.s id (fun _ => ⌈q⌉₊)).dilute w.1 w.2

theorem weighted_le {m N : ℕ} (hm : 0 < m) {p β b : ℝ} (hB : (m : ℝ) / (N + 1) * p ≤ β)
    (hb : ((N : ℝ) + 1) * β ≤ m * b) : p ≤ b := by
  have hN : (0 : ℝ) < N + 1 := by positivity
  have h1 : (m : ℝ) * p ≤ (N + 1) * β := by
    rw [div_mul_eq_mul_div, div_le_iff₀ hN, mul_comm β] at hB
    exact hB
  exact le_of_mul_le_mul_left (h1.trans hb) (by exact_mod_cast hm)

/-- **A diluted strict-time finder's premise from the threshold**: where it succeeds with probability at most `β` and
`(N + 1) β ≤ m q²/2^513`, C-Flock's finder meets `SHA512CRStrict` at the budget `q`. -/
theorem strictCR_of (Fd : StrictCR.Finder StrictCR.Pair) (q : ℝ) (w : ℕ × ℕ) (hw : 0 < w.1 ∧ w.1 ≤ w.2 + 1) {β : ℝ}
    (hB : (strictFinder Fd q w).Pr Verity.sha512Msg ≤ β)
    (hb : ((w.2 : ℝ) + 1) * β ≤ w.1 * (q ^ 2 / 2 ^ (513 : ℝ))) :
    Fd.CR H512 q := by
  unfold StrictCR.Finder.CR Assumptions.SHA512CRStrict
  intro _
  unfold strictFinder at hB
  rw [Claims.Finder.dilute_Pr _ _ hw.2, Game.tapeFinder_Pr] at hB
  have hp : (fun b : StrictCR.Pair => ∃ x y, id b = some (x, y) ∧ x ≠ y ∧ H512 x = H512 y) =
      fun b => ∃ u v, id b = some (u, v) ∧ Claims.Collision Verity.sha512Msg u v :=
    funext fun b => by simp only [Claims.Collision, H512_eq_iff]
  rw [hp]
  exact weighted_le hw.1 hB hb

/-- **A diluted expected-time finder's premise from the threshold**: where every outcome costs at least `t` and the
diluted finder succeeds with probability at most `β`, `(N + 1) β ≤ m t/2^256` gives `SHA512CRExpected`. -/
theorem expectedCR_of {γ : Type} (g : Game γ) (s : Strategy g) (out : γ → Option (Claims.Msg × Claims.Msg))
    (cost : γ → ℝ) (cst : γ → ℕ) (w : ℕ × ℕ) (hw : 0 < w.1 ∧ w.1 ≤ w.2 + 1) {t β : ℝ} (hcost : ∀ x, t ≤ cost x)
    (hB : ((Game.tapeFinder g s out cst).dilute w.1 w.2).Pr Verity.sha512Msg ≤ β)
    (hb : ((w.2 : ℝ) + 1) * β ≤ w.1 * (t / 2 ^ (256 : ℝ))) :
    Assumptions.SHA512CRExpected H512 g s out cost := by
  unfold Assumptions.SHA512CRExpected
  rw [Claims.Finder.dilute_Pr _ _ hw.2, Game.tapeFinder_Pr] at hB
  have hp : (fun b : γ => ∃ x y, out b = some (x, y) ∧ x ≠ y ∧ H512 x = H512 y) =
      fun b => ∃ u v, out b = some (u, v) ∧ Claims.Collision Verity.sha512Msg u v :=
    funext fun b => by simp only [Claims.Collision, H512_eq_iff]
  rw [hp]
  have ht : t ≤ expect cost g s := by
    have := expect_mono hcost g s
    rwa [expect_const] at this
  exact (weighted_le hw.1 hB hb).trans (div_le_div_of_nonneg_right ht (by positivity))

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

/-- **Table `j`'s knowledge finders after draw `ω`**, each `SessCRZ`'s (through `TableCRZ`), diluted by its kind's
weight. -/
noncomputable def tabComp (τ : Strategy A.G) (w : Weights) (ω : A.L.Ω) (j : Fin (A.tsAt τ ω).length) :
    TabIx ((A.tsAt τ ω).get j).sch.levels.length → Claims.Finder
  | .inl (r, ℓ) => strictFinder (advRSFinderZ execArith hs512 ((A.tsAt τ ω).get j)
      ((gamesZC execArith hs512 (A.tsAt τ ω) (A.zcAt τ ω).1).take j)
      ((gamesZC execArith hs512 (A.tsAt τ ω) (A.zcAt τ ω).1).drop (j + 1)) ((A.zcAt τ ω).1 j) r.val (ℓ.val + 1)
      (stratZ execArith hs512 (A.tsAt τ ω) j (A.zcAt τ ω))) (A.qF τ) w.F
  | .inr (.inl r) => strictFinder (advPadsSFinderZ execArith hs512 ((A.tsAt τ ω).get j)
      ((gamesZC execArith hs512 (A.tsAt τ ω) (A.zcAt τ ω).1).take j)
      ((gamesZC execArith hs512 (A.tsAt τ ω) (A.zcAt τ ω).1).drop (j + 1)) ((A.zcAt τ ω).1 j) r.val
      (stratZ execArith hs512 (A.tsAt τ ω) j (A.zcAt τ ω))) (A.qF τ) w.F
  | .inr (.inr (.inl r)) => strictFinder (advTauSFinderZ execArith hs512 ((A.tsAt τ ω).get j)
      ((gamesZC execArith hs512 (A.tsAt τ ω) (A.zcAt τ ω).1).take j)
      ((gamesZC execArith hs512 (A.tsAt τ ω) (A.zcAt τ ω).1).drop (j + 1)) ((A.zcAt τ ω).1 j) r.val
      (stratZ execArith hs512 (A.tsAt τ ω) j (A.zcAt τ ω))) (A.qF τ) w.F
  | .inr (.inr (.inr (.inl _))) => strictFinder (adv₀SFinderZ execArith hs512 ((A.tsAt τ ω).get j)
      ((gamesZC execArith hs512 (A.tsAt τ ω) (A.zcAt τ ω).1).take j)
      ((gamesZC execArith hs512 (A.tsAt τ ω) (A.zcAt τ ω).1).drop (j + 1)) ((A.zcAt τ ω).1 j)
      (stratZ execArith hs512 (A.tsAt τ ω) j (A.zcAt τ ω))) (A.qF τ) w.F
  | .inr (.inr (.inr (.inr _))) => strictFinder (selfSFinderZ execArith hs512 ((A.tsAt τ ω).get j)
      ((gamesZC execArith hs512 (A.tsAt τ ω) (A.zcAt τ ω).1).take j)
      ((gamesZC execArith hs512 (A.tsAt τ ω) (A.zcAt τ ω).1).drop (j + 1)) ((A.zcAt τ ω).1 j)
      (stratZ execArith hs512 (A.tsAt τ ω) j (A.zcAt τ ω))) (A.qS τ) w.S

/-- The committed positions some draw reads, where a link finder exists. -/
abbrev LinkPos : Type :=
  {c : A.vb.Pos // Nonempty (ΩcZ execArith A.P (planZHJ A.dj A.y₀) (tabZHJ A.dj A.y₀) A.vb A.L c)}

/-- **Position `c`'s link finder**, `LinkCRZC`'s: a trial and up to `M` more, at cost `t' τ` per run of `τ`, diluted
by the link weight. -/
noncomputable def linkComp (τ : Strategy A.G) (w : Weights) (c : A.LinkPos) : Claims.Finder :=
  haveI := c.2
  (Game.tapeFinder
    (finderG (trialGZC execArith hs512 A.P (planZHJ A.dj A.y₀) (tabZHJ A.dj A.y₀) A.vb (reg (A.sim τ))
      (cont (A.sim τ)) c.1 A.k A.Rw) A.M)
    (finderStrat _ (trialStratZC execArith hs512 A.P (planZHJ A.dj A.y₀) (tabZHJ A.dj A.y₀) A.vb (reg (A.sim τ))
      (cont (A.sim τ)) c.1 A.k A.Rw) A.M)
    (outCZC execArith hs512 A.P (planZHJ A.dj A.y₀) (tabZHJ A.dj A.y₀) A.vb (reg (A.sim τ)) (cont (A.sim τ)) c.1)
    (fun x => ⌈finderCost (rdTZC execArith hs512 A.P (planZHJ A.dj A.y₀) (tabZHJ A.dj A.y₀) A.vb (reg (A.sim τ))
      (cont (A.sim τ)) c.1) (cTZC (planZHJ A.dj A.y₀) (reg (A.sim τ)) (A.t' τ)) x⌉₊)).dilute w.L.1 w.L.2

/-- **C-Flock's finders at `τ`**: every table's knowledge finders after every draw, every link finder, and the finder
that outputs nothing. -/
abbrev Ix (τ : Strategy A.G) : Type :=
  Option ((Σ ω : A.L.Ω, Σ j : Fin (A.tsAt τ ω).length, TabIx ((A.tsAt τ ω).get j).sch.levels.length) ⊕ A.LinkPos)

instance instFiniteIx (τ : Strategy A.G) : Finite (A.Ix τ) := by
  classical
  infer_instance

/-- C-Flock's finders at `τ`, each diluted by its kind's weight. -/
noncomputable def crFam (τ : Strategy A.G) (w : Weights) : A.Ix τ → Claims.Finder
  | none => Claims.Finder.null
  | some (.inl ⟨ω, j, i⟩) => A.tabComp τ w ω j i
  | some (.inr c) => A.linkComp τ w c

/-- **Sampled proofs' finder `B^τ`**: among C-Flock's finders at `τ`, each diluted by its kind's weight, one of
largest probability of outputting a SHA-512 collision. -/
noncomputable def crFinder (τ : Strategy A.G) (w : Weights) : Claims.Finder :=
  Claims.Finder.worst Verity.sha512Msg (A.crFam τ w)

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

/-- **C-Flock's premises from the threshold.** Where `B^τ` (`crFinder τ w`) outputs a SHA-512 collision with
probability at most `β`, and `(N + 1) β ≤ m · bound` for each kind's weight `m / (N + 1)` and bound (`qF τ²/2^513`
for the rewinding finders, `qS τ²/2^513` for the self-clash finder, `t' τ/2^256` for the link finders), every finder
`LiveAt.CR τ` names meets its premise. -/
theorem CR_of_crFinder (τ : Strategy A.G) (w : Weights) {β : ℝ}
    (hB : (A.crFinder τ w).Pr Verity.sha512Msg ≤ β)
    (hbF : ((w.F.2 : ℝ) + 1) * β ≤ w.F.1 * (A.qF τ ^ 2 / 2 ^ (513 : ℝ)))
    (hbS : ((w.S.2 : ℝ) + 1) * β ≤ w.S.1 * (A.qS τ ^ 2 / 2 ^ (513 : ℝ)))
    (hbL : ((w.L.2 : ℝ) + 1) * β ≤ w.L.1 * (A.t' τ / 2 ^ (256 : ℝ))) :
    A.CR τ := by
  have hfam : ∀ i, (A.crFam τ w i).Pr Verity.sha512Msg ≤ β := fun i =>
    (Claims.Finder.le_worst Verity.sha512Msg (A.crFam τ w) i).trans hB
  unfold Flock.Guarantees.LiveAt.CR
  refine ⟨fun ω j => ⟨fun r hr ℓ hℓ => ?_, fun r hr => ?_, fun r hr => ?_, ?_, ?_⟩, ?_⟩
  · have hr' : r < 2 := Finset.mem_range.1 hr
    obtain ⟨hℓ1, hℓ2⟩ := Finset.mem_Icc.1 hℓ
    obtain ⟨ℓ', rfl⟩ : ∃ ℓ', ℓ = ℓ' + 1 := ⟨ℓ - 1, by omega⟩
    exact strictCR_of _ _ w.F w.hF
      (hfam (some (.inl ⟨ω, j, .inl (⟨r, hr'⟩, ⟨ℓ', Nat.lt_of_succ_le hℓ2⟩)⟩))) hbF
  · exact strictCR_of _ _ w.F w.hF
      (hfam (some (.inl ⟨ω, j, .inr (.inl ⟨r, Finset.mem_range.1 hr⟩)⟩))) hbF
  · exact strictCR_of _ _ w.F w.hF
      (hfam (some (.inl ⟨ω, j, .inr (.inr (.inl ⟨r, Finset.mem_range.1 hr⟩))⟩))) hbF
  · exact strictCR_of _ _ w.F w.hF (hfam (some (.inl ⟨ω, j, .inr (.inr (.inr (.inl ())))⟩))) hbF
  · exact strictCR_of _ _ w.S w.hS (hfam (some (.inl ⟨ω, j, .inr (.inr (.inr (.inr ())))⟩))) hbS
  · intro c hc
    exact expectedCR_of _ _ _ _ _ w.L w.hL (A.linkCost_ge τ c) (hfam (some (.inr ⟨c, hc⟩))) hbL

end Flock.Guarantees.LiveAt

namespace Verity

open Accounting Core

variable {ι α : Type} [Fintype α] {d : Device ι}

/-- **`SampledDeploy`'s `cr` field.** At a vocabulary whose sampled finder, at registered files that parse, is
`crFinder` at the session's strategy (`hfinder`), with weights `w` under which the published `ε_cr` at the strategy's
calls meets each kind's inequality (`hbF`, `hbS`, `hbL`), the vocabulary's finder premise gives C-Flock's. -/
theorem SampledDeploy.cr_of (V : Vocab d α) (kd : ByteArray → ℕ) (desc : ByteArray → Claims.Msg)
    (A : ∀ (θ F c : ByteArray) (_σ : Game.Strategy (V.game θ F c)) (r : Flock.Bridge.Parsed), V.parsed F = some r →
      Flock.Guarantees.LiveAt r.n r.D r.hU (kd θ) (⌊(V.certOf θ).ε * r.n⌋₊ + 1) (desc θ))
    (τOf : ∀ (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)) (r : Flock.Bridge.Parsed)
      (hr : V.parsed F = some r), FlockSoundness.Game.Strategy (A θ F c σ r hr).G)
    (w : ∀ (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)) (r : Flock.Bridge.Parsed),
      V.parsed F = some r → Flock.Guarantees.LiveAt.Weights)
    (hfinder : ∀ θ F c σ r hr, V.Registered θ F →
      V.finder .sampled θ F c σ = (A θ F c σ r hr).crFinder (τOf θ F c σ r hr) (w θ F c σ r hr))
    (hbF : ∀ θ F c σ r hr, (((w θ F c σ r hr).F.2 : ℝ) + 1) * V.crBound θ .sampled (V.cost θ F c σ) ≤
      (w θ F c σ r hr).F.1 * ((A θ F c σ r hr).qF (τOf θ F c σ r hr) ^ 2 / 2 ^ (513 : ℝ)))
    (hbS : ∀ θ F c σ r hr, (((w θ F c σ r hr).S.2 : ℝ) + 1) * V.crBound θ .sampled (V.cost θ F c σ) ≤
      (w θ F c σ r hr).S.1 * ((A θ F c σ r hr).qS (τOf θ F c σ r hr) ^ 2 / 2 ^ (513 : ℝ)))
    (hbL : ∀ θ F c σ r hr, (((w θ F c σ r hr).L.2 : ℝ) + 1) * V.crBound θ .sampled (V.cost θ F c σ) ≤
      (w θ F c σ r hr).L.1 * ((A θ F c σ r hr).t' (τOf θ F c σ r hr) / 2 ^ (256 : ℝ))) :
    ∀ (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)) (r : Flock.Bridge.Parsed)
      (hr : V.parsed F = some r), V.Registered θ F →
      V.FinderCR .sampled θ F c σ → (A θ F c σ r hr).CR (τOf θ F c σ r hr) := by
  intro θ F c σ r hr hreg hcr
  have hB : ((A θ F c σ r hr).crFinder (τOf θ F c σ r hr) (w θ F c σ r hr)).Pr sha512Msg ≤
      V.crBound θ .sampled (V.cost θ F c σ) := by
    rw [← hfinder θ F c σ r hr hreg]
    exact hcr
  exact (A θ F c σ r hr).CR_of_crFinder _ _ hB (hbF θ F c σ r hr) (hbS θ F c σ r hr) (hbL θ F c σ r hr)

end Verity
