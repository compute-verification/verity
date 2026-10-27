import Proofs.Flock.Bridge.HiddenReturns

/-!
# The hidden-output count curve is nontrivial at its layer

`EndToEndHidden` is the compiled `--zk` count curve with "wrong" measured against the registration
(`ZkLink.zk_flock_countPr_reg`, over any partition and value binding), instantiated at the hidden-output tables, the
key program and the public file's commitments. At a parsed circuit file no constructed prover is wrong: the rows'
commitments hide, the single gates' leaves hash values of many bits, and the value layer `XplurZC` is a maximizer
nobody can name (`HiddenReturns`). So the instance that shows the count curve's bound is nontrivial is at the layer,
over the same tables `planZHJ dj y₀`:

* **The instance** (`toyBinding`): `Bridge.toyCircuit` (an input and its negation, returned) in one unit, under a
  perfectly binding commitment that registers `1` at both wires, a decoder that reads the registered bits, and a law
  that draws the unit.
* **Wrong at every registration and transcript** (`toy_mem_wrongRegZ`): the input's registered `1` is its value in
  `Xv`, so the circuit computes `0` at the negation, which is registered `1`.
* **The layer's hypotheses hold** where the tables are unsatisfiable (`toy_lowering`, `toy_readsOwn`): a satisfying
  message would open the negation's commitment to the input's negation, which is not registered.
* **Nontrivial** (`countPrReg_nontrivial`): the count curve's bound at the deployed parameters is below 1, while the
  event with the verifier that accepts everything holds with probability 1, at every prover whose finder `B^τ` has
  `J²·K·Pr[B^τ] ≤ 2^-200` (`boundPr_le_pub`).
-/

namespace Flock.Bridge

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Model FlockSoundness.Accounting
  FlockSoundness.Refine
  FlockSoundness.Discharge.ZkSession FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec
  FlockSoundness.Discharge.ZkHidden FlockSoundness.Discharge.Composed
open scoped ENNReal

section Layer

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}
  (A : Arith F K) {Reg : Type} (planZ : Finset (Fin 1) → Reg → List (TabZK F mPts))
  (tab : (S : Finset (Fin 1)) → (R : Reg) → Fin 1 → Fin (planZ S R).length)

/-- **The decoder that reads the registered bits**: `1` at every wire. -/
def toyDecode : DecoderZK (C := toyCircuit) planZ tab := fun _ _ _ _ _ => true

/-- **A perfectly binding commitment registering `1` at both wires**: each wire its own commit string, a bit its own
opening and commitment. -/
def toyBinding {Dh : Type} (Hc : List UInt8 → Dh) :
    ValueBindingZ A toyPartition planZ tab (toyDecode planZ tab) Hc where
  Pos := Fin toyCircuit.N
  Val := Bool
  pos g := g
  bit v _ := v
  Opening := Bool
  Com := Bool
  val o := o
  com o := o
  cs _ _ := true
  opening _ _ _ _ _ := true
  opening_spec _ _ _ _ _ _ _ _ := ⟨rfl, rfl⟩
  collide _ _ := ([], [])
  collide_spec _ _ h hne := absurd h hne

theorem toy_mem_io0 : (⟨0, by decide⟩ : Fin toyCircuit.N) ∈ toyPartition.io 0 := by
  classical
  simp only [Partition.io, Finset.mem_filter]
  exact ⟨toyPartition.mem_committed_of_input (by decide), Or.inr ⟨⟨1, by decide⟩, by decide, by decide⟩⟩

theorem toy_mem_io1 : (⟨1, by decide⟩ : Fin toyCircuit.N) ∈ toyPartition.io 0 := by
  classical
  simp only [Partition.io, Finset.mem_filter]
  exact ⟨toyPartition.mem_committed_of_output (by decide), Or.inl (by decide)⟩

omit [Fintype K] [DecidableEq K] in
/-- **The one unit is wrong at every registration and transcript**: an opening of the input's commit string carries
`1`, the input's value in `Xv`; then the circuit computes `0` at the negation, and its commit string opens only to
`1`. -/
theorem toy_mem_wrongRegZ {Dh : Type} (Hc : List UInt8 → Dh) (R : Reg) (Xv : Fin toyCircuit.N → Bool) :
    (0 : Fin 1) ∈ wrongRegZ A toyPartition planZ tab (toyBinding A planZ tab Hc) R Xv := by
  refine (mem_wrongRegZ A toyPartition planZ tab (toyBinding A planZ tab Hc)).2 ?_
  rintro ⟨-, op, hop⟩
  obtain ⟨hc0, hb0⟩ := hop _ toy_mem_io0
  obtain ⟨hc1, hb1⟩ := hop _ toy_mem_io1
  have a0 : op ⟨0, by decide⟩ = true := hc0
  have a1 : op ⟨1, by decide⟩ = true := hc1
  have b0 : op ⟨0, by decide⟩ = toyCircuit.eval Xv ⟨0, by decide⟩ := hb0
  have b1 : op ⟨1, by decide⟩ = toyCircuit.eval Xv ⟨1, by decide⟩ := hb1
  have e1 : toyCircuit.eval Xv ⟨1, by decide⟩ = !toyCircuit.eval Xv ⟨0, by decide⟩ :=
    toyCircuit.eval_op Xv (g := ⟨1, by decide⟩) (by decide)
  have h0 : toyCircuit.eval Xv ⟨0, by decide⟩ = true := b0.symm.trans a0
  have h1 : toyCircuit.eval Xv ⟨1, by decide⟩ = true := b1.symm.trans a1
  rw [e1, h0] at h1
  exact absurd h1 (by decide)

omit [Fintype K] [DecidableEq K] in
/-- **The layer's lowering holds where no message satisfies the tables**: there is nothing to decode. -/
theorem toy_lowering
    (hunsat : ∀ S R u M, ¬ (tabOfZ planZ tab S R u).S.Satisfies (wit A (tabOfZ planZ tab S R u).S (tabOfZ planZ tab S R u).sch M)) :
    LoweringSoundZK A toyPartition planZ tab (toyDecode planZ tab) :=
  fun S R u _ M hM => absurd hM (hunsat S R u M)

/-- **The unit reads only its own gates and the input.** -/
theorem toy_readsOwn : ∀ u, toyPartition.ReadsOwn u := by
  intro u h x _ _
  by_cases hx : x.val = 0
  · exact Or.inr (by simp [toyPartition, hx])
  · exact Or.inl (by simp [toyPartition, hx, Fin.fin_one_eq_zero u])

end Layer

open FlockSoundness.Game in
/-- **An event that always holds holds with probability 1**, under every strategy of every game: so it exceeds any
`x < 1`. -/
theorem exceeds_of {α : Type} {G : Game α} (s : Strategy G) {R : Prop} (hR : R) {x : ℝ} (hx : x < 1) :
    ENNReal.ofReal x < prob (fun _ => True ∧ R) G s := by
  have h1 : prob (fun _ => True ∧ R) G s = 1 :=
    le_antisymm (prob_le_one _ _ _) ((prob_true _ _).symm.le.trans (prob_mono (fun _ _ => ⟨trivial, hR⟩) _ _))
  rw [h1]
  exact ENNReal.ofReal_lt_one.2 hx

open Classical in
/-- **The hidden-output count curve is nontrivial at its layer**: at the hidden-output tables of any untyped statement
(`planZHJ dj y₀`), the toy instance (`toyBinding`) with a law that draws its one unit, and the deployed parameters
(`k = ⌈2^24/e⌉`, `R_w = M = 2^192`, `ρ = 1/2`), the bound of `ZkLink.zk_flock_countPr_reg` (its knowledge term at
`ksBoundAccZ`) is below the probability, 1, that the verifier that accepts everything accepts with the unit wrong, at
every prover whose finder `B^τ` (at a per-run time `t'` and a verification `v`) has `J²·K·Pr[B^τ] ≤ 2^-200`, `J` the
plan's tables. That premise only places the prover, as in `sampledKSLiveV2Z_unopened`. -/
theorem countPrReg_nontrivial {mPts : ℕ} {Reg : Type} {I : Discharge.Exec.Inputs} (dj : Finset (Fin 1) → Lean.Json)
    {S₀ : Finset (Fin 1)} {d₀ : Lean.Json} (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀) (L : Law 1)
    (hL : ∀ ω, L.draw ω = Finset.univ)
    (σ : Strategy (audit L Reg (batchedSessionZC execArith hs512 (planZHJ dj y₀)))) (t' v : ℝ)
    (hB : (nTab I : ℝ) ^ 2 * Fintype.card (PubIx (Reg := Reg) dj y₀ toyPartition
        (toyBinding execArith (planZHJ dj y₀) (tabZHJ dj y₀) H512) L) *
      (pubFinder (Reg := Reg) dj y₀ toyPartition (toyBinding execArith (planZHJ dj y₀) (tabZHJ dj y₀) H512) kDep
        (2 ^ 192) (2 ^ 192) (2 * (t' + v)) (t' + v) t' (reg σ) (cont σ)).Pr Claims.Sampling.sha512Msg ≤ 1 / 2 ^ 200) :
    L.miss 1 +
        ksAvgZC execArith hs512 (planZHJ dj y₀) kDep
          (fun S R τ j => ksBoundAccZ execArith hs512 (planZHJ dj y₀ S R) j kDep τ) (reg σ) (cont σ) +
        linkPrZC execArith hs512 toyPartition (planZHJ dj y₀) (tabZHJ dj y₀)
          (toyBinding execArith (planZHJ dj y₀) (tabZHJ dj y₀) H512) kDep (2 ^ 192) (2 ^ 192) (1 / 2) (reg σ) (cont σ) <
      prob (fun _ => True ∧ 1 ≤ (wrongRegZ execArith toyPartition (planZHJ dj y₀) (tabZHJ dj y₀)
          (toyBinding execArith (planZHJ dj y₀) (tabZHJ dj y₀) H512) (reg σ)
          (XplurZC execArith hs512 toyPartition (planZHJ dj y₀) (tabZHJ dj y₀)
            (toyBinding execArith (planZHJ dj y₀) (tabZHJ dj y₀) H512) (reg σ) (cont σ) kDep (2 ^ 192))).card)
        (audit L Reg (batchedSessionZC execArith hs512 (planZHJ dj y₀))) σ := by
  have hle := boundPr_le_pub dj y₀ toyPartition (toyBinding execArith (planZHJ dj y₀) (tabZHJ dj y₀) H512)
    (kd := 1) (K₀ := 1) le_rfl (k := kDep) (Rw := 2 ^ 192) (M := 2 ^ 192) (ρ := 1 / 2) (by norm_num) rateM1_kDep
    (2 * (t' + v)) (t' + v) t' (reg σ) (cont σ)
  have hlt := deltaPub_lt_one (n := 1) (kd := 1) (K₀ := 1) (W := ioSum toyPartition) one_pos le_rfl (by norm_num)
    toy_ioSum_le (mul_nonneg (mul_nonneg (sq_nonneg _) (Nat.cast_nonneg _)) (Claims.Pr_nonneg _)) hB
  refine lt_of_le_of_lt ?_ (exceeds_of σ (Finset.one_le_card.2
    ⟨0, toy_mem_wrongRegZ execArith (planZHJ dj y₀) (tabZHJ dj y₀) H512 (reg σ) _⟩) hlt)
  rw [L.miss_eq_zero_of_draw_univ hL le_rfl]
  exact le_trans (add_le_add (add_le_add zero_le (ksAvgZC_le_pr execArith hs512 (planZHJ dj y₀) kDep (reg σ)
    (cont σ))) le_rfl) hle

end Flock.Bridge
