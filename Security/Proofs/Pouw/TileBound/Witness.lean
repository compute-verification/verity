import Proofs.Pouw.TileBound.Proofs

/-!
# The chain's satisfiability witness (a later bundle; outside the 36 merged pins)

The honest one-gate program meets `tile_cost_ge_of_gates`' whole three-input hypothesis on one live pair. This module
proves it:
* `oneGateP` and `oneGateQ` carry `(e_(0,0), 0)` and `(0, e_(0,0))`, and are exact on all 128 words (`decide +kernel`);
* `oneGateProg_eval`: the honest one-gate program computes `X·Y` at every input;
* `oneGateProg_gates` is the hypothesis `hg` of `tile_cost_ge_of_gates` for `oneGateProg`, with live entries
  `x = (0, 0)` and `y = (0, 0)`;
* `tile_chain_satisfiable` (the pin) packages it: some representable tile program meets the chain's hypothesis on
  one live product, so the chain covers a real program. The pin states it existentially, so that it reads only
  `Defs.lean`.
-/

namespace Pouw.TileBound

open Finset Pouw.PearlC.Fp4

variable {τ : Type*} [Fintype τ] {β : Type*} [Fintype β] {γ : Type*} [Fintype γ]

/-- Code 1.0 at `P[0, 0]`, every other code 0, every scale 1.0: the firing that carries `(e_(0,0), 0)`. -/
def oneGateP : Gate where
  p i l := if i.val = 0 ∧ l.val = 0 then 2 else 0
  ps _ _ := 0x38
  q _ _ := 0
  qs _ _ := 0x38
  acc _ _ := 0

/-- Code 1.0 at `Q[0, 0]`, every other code 0, every scale 1.0: the firing that carries `(0, e_(0,0))`. -/
def oneGateQ : Gate where
  p _ _ := 0
  ps _ _ := 0x38
  q l j := if l.val = 0 ∧ j.val = 0 then 2 else 0
  qs _ _ := 0x38
  acc _ _ := 0

theorem oneGateP_exact : oneGateP.Exact := by
  unfold Gate.Exact
  decide +kernel

theorem oneGateQ_exact : oneGateQ.Exact := by
  unfold Gate.Exact
  decide +kernel

theorem oneGateP_p : ∀ i l, elemValue (oneGateP.p i l) (oneGateP.ps i (scaleIdx l)) =
    if (i, l) = ((0 : Fin 16), (0 : Fin 64)) then 1 else 0 := by decide +kernel

theorem oneGateP_q : ∀ l j, elemValue (oneGateP.q l j) (oneGateP.qs j (scaleIdx l)) = 0 := by decide +kernel

theorem oneGateQ_p : ∀ i l, elemValue (oneGateQ.p i l) (oneGateQ.ps i (scaleIdx l)) = 0 := by decide +kernel

theorem oneGateQ_q : ∀ l j, elemValue (oneGateQ.q l j) (oneGateQ.qs j (scaleIdx l)) =
    if (l, j) = ((0 : Fin 64), (0 : Fin 8)) then 1 else 0 := by decide +kernel

/-- The honest one-gate program computes `X·Y` at every input. -/
theorem oneGateProg_eval (X : Fin 16 × Fin 64 → ℚ) (Y : Fin 64 × Fin 8 → ℚ) (o : Fin 16 × Fin 8) :
    oneGateProg.toBilin.toQuad.eval X Y o = matMul X Y o := by
  obtain ⟨o1, o2⟩ := o
  simp only [QuadProg.eval, BilinProg.toQuad, TileProg.toBilin, oneGateProg, Pi.zero_apply, zero_mul,
    Finset.sum_const_zero, add_zero, zero_add, Fintype.sum_prod_type, unitForm, ite_mul, one_mul,
    Finset.sum_ite_eq', Finset.mem_univ, ite_true, matMul, Prod.mk.injEq]
  simp [ite_and]

theorem oneGateProg_realizesAt_P :
    oneGateProg.RealizesAt (fun _ => oneGateP) (unitForm (0, 0)) 0 := by
  intro _
  refine ⟨oneGateP_exact, fun _ _ => rfl, fun i l => ?_, fun l j => ?_⟩
  · rw [oneGateP_p]; simp only [oneGateProg, sum_unitForm_mul]
  · rw [oneGateP_q]; simp [oneGateProg]

theorem oneGateProg_realizesAt_Q :
    oneGateProg.RealizesAt (fun _ => oneGateQ) 0 (unitForm (0, 0)) := by
  intro _
  refine ⟨oneGateQ_exact, fun _ _ => rfl, fun i l => ?_, fun l j => ?_⟩
  · rw [oneGateQ_p]; simp [oneGateProg]
  · rw [oneGateQ_q]; simp only [oneGateProg, sum_unitForm_mul]

/-- **The chain's hypothesis holds for a real program.** `tile_cost_ge_of_gates`' three-input hypothesis `hg` for the
honest one-gate program, with live entries `x = (0, 0)` and `y = (0, 0)`: exact `sm120Nvf4` firings realize it at
`(e_x, e_y)`, `(e_x, 0)` and `(0, e_y)`, and output `X·Y` there. -/
theorem oneGateProg_gates : ∀ x : Fin 16 × Fin 64, x = (0, 0) → ∀ y : Fin 64 × Fin 8, y = (0, 0) →
    ∃ g11 g10 g01 : Fin 1 → Gate,
      oneGateProg.RealizesAt g11 (unitForm x) (unitForm y) ∧ oneGateProg.RealizesAt g10 (unitForm x) 0 ∧
        oneGateProg.RealizesAt g01 0 (unitForm y) ∧
      ∀ o, oneGateProg.outAt g11 o = matMul (unitForm x) (unitForm y) o ∧
        oneGateProg.outAt g10 o = matMul (unitForm x) 0 o ∧
        oneGateProg.outAt g01 o = matMul 0 (unitForm y) o := by
  rintro x rfl y rfl
  refine ⟨fun _ => oneGate, fun _ => oneGateP, fun _ => oneGateQ, oneGateProg_realizesAt, oneGateProg_realizesAt_P,
    oneGateProg_realizesAt_Q, fun o => ⟨?_, ?_, ?_⟩⟩
  · rw [realizes_out_at _ _ _ _ oneGateProg_realizesAt, oneGateProg_eval]
  · rw [realizes_out_at _ _ _ _ oneGateProg_realizesAt_P, oneGateProg_eval]
  · rw [realizes_out_at _ _ _ _ oneGateProg_realizesAt_Q, oneGateProg_eval]

/-- The chain on the honest one-gate program (a lemma; the pinned form is `tile_chain_satisfiable`). -/
theorem tile_cost_ge_of_gates_witness :
    price .fp4 * nLive (m := 16) (k := 64) (n := 8) (fun x => x = (0, 0)) (fun y => y = (0, 0)) ≤
      oneGateProg.cost :=
  tile_cost_ge_of_gates oneGateProg (nvfp4X 16 64) (fun _ _ _ => by simp [oneGateProg, XBlocks.touches_unit])
    oneGateProg_gates

omit [Fintype τ] [Fintype β] [Fintype γ] in
/-- One live X-entry and one live Y-entry that meet: one live product. -/
theorem nLive_pair {m k n : ℕ} (a : Fin m × Fin k) (b : Fin k × Fin n) (h : a.2 = b.1) :
    nLive (fun x => x = a) (fun y => y = b) = 1 := by
  unfold nLive
  let U : Unique {x : Fin m × Fin k × Fin n // (fun x => x = a) (x.1, x.2.1) ∧ (fun y => y = b) (x.2.1, x.2.2)} :=
    { default := ⟨(a.1, a.2, b.2), rfl, by show (a.2, b.2) = b; rw [h]⟩
      uniq := fun ⟨x, hx⟩ => Subtype.ext (by
        obtain ⟨h1, h2⟩ := hx
        show x = (a.1, a.2, b.2)
        rw [← h1, ← h2]) }
  convert @Fintype.card_unique _ U _

omit [Fintype τ] [Fintype β] [Fintype γ] in
theorem nLive_single : nLive (m := 16) (k := 64) (n := 8) (fun x => x = (0, 0)) (fun y => y = (0, 0)) = 1 :=
  nLive_pair _ _ rfl

/-- **The chain's hypotheses are satisfiable, non-vacuously.** Some tile program is representable on NVFP4's blocks
and meets `tile_cost_ge_of_gates`' three-input hypothesis on a live pair, and that pair is one live product. So the
chain from exact `sm120Nvf4` firings to the cost bound applies to a real program. -/
theorem tile_chain_satisfiable : ∃ P : TileProg ℚ 16 64 8 1, P.Representable (nvfp4X 16 64) ∧
    (∀ x : Fin 16 × Fin 64, x = (0, 0) → ∀ y : Fin 64 × Fin 8, y = (0, 0) →
      ∃ g11 g10 g01 : Fin 1 → Gate,
        P.RealizesAt g11 (unitForm x) (unitForm y) ∧ P.RealizesAt g10 (unitForm x) 0 ∧
          P.RealizesAt g01 0 (unitForm y) ∧
        ∀ o, P.outAt g11 o = matMul (unitForm x) (unitForm y) o ∧ P.outAt g10 o = matMul (unitForm x) 0 o ∧
          P.outAt g01 o = matMul 0 (unitForm y) o) ∧
    nLive (m := 16) (k := 64) (n := 8) (fun x => x = (0, 0)) (fun y => y = (0, 0)) = 1 :=
  ⟨oneGateProg, fun _ _ _ => by simp [oneGateProg, XBlocks.touches_unit], oneGateProg_gates, nLive_single⟩

end Pouw.TileBound
