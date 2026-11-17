module

public import Definitions.Claims.SamplingTwoRegX

/-! Claims: what the two-stage audit's committed trace serves, fixed before the sample. -/

@[expose] public section

namespace Claims.Sampling

/-- The values a trace holds at the program's input gates, `none` at every other gate (`inputsOf` at a trace). -/
def traceInputs (s : AuditStatement) (val : ℕ → Option Bool) (g : ℕ) : Option Bool :=
  if s.prog.gates[g]? = some .input then val g else none

/-- **What a trace serves**: its values at the program's returned gates, a constant's its own (`Prog.read`), a gate
it holds no value at `false` (`served` at a trace). -/
def traceServed (s : AuditStatement) (val : ℕ → Option Bool) : Fin s.prog.outs.length → Bool :=
  fun i => (s.prog.read val s.prog.outs[i]).getD false

open Classical in
/-- **The influence set of traces** at `δ` and input values `a`: what every trace with inputs `a` serves whose
footprint is in `M̄_δ` at some interiors (`influenceTwo` at a trace). -/
noncomputable def influenceTwoAt (s : AuditStatement) (Pt : Partition) (d : Design.Two) (δ : ℝ)
    (a : ℕ → Option Bool) : Finset (Fin s.prog.outs.length → Bool) :=
  Finset.univ.filter fun y => ∃ val : ℕ → Option Bool, traceInputs s val = a ∧
    (∃ ι : ℕ → ℕ → Bool, footprintAt s Pt d.part val ι ∈ d.M Pt δ) ∧ traceServed s val = y

end Claims.Sampling

namespace Claims.SamplingGeneric

open Claims
open Claims.Sampling (AuditStatement Partition InScopeTwo footprintAt admissibleTwo bitsOf traceInputs traceServed
  influenceTwoAt)

namespace CompiledTwo

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-- **The exfiltration bound at the committed trace**, read at the statement and fixed before the sample
(`Compiled.regValX`): the committed form holds (`KSRegTwoX`); on a statement in scope, at every prover and `δ ≥ 0`, a
run accepts while what that trace serves is outside the influence set at its own inputs with probability at most
`δ + ε_reg,2,X`; and the influence set at any input values has at most `Σ_{T admissible} 2^{b(T)}` members. -/
def ExfilRegTwoX (H : Msg → Msg) (L : CompiledTwo Coin A) : Prop :=
  L.KSRegTwoX H ∧
    (∀ x, L.Scope x → ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω),
      ∀ s Pt d, L.parse x = some s → L.partition s = some Pt → L.two s = some d → ∀ δ : ℝ, 0 ≤ δ →
        Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
            traceServed s (L.regValX H P x) ∉ influenceTwoAt s Pt d δ (traceInputs s (L.regValX H P x))) ≤
          δ + L.εregTwoX H x P) ∧
    ∀ x s Pt d (a : ℕ → Option Bool), L.parse x = some s → L.partition s = some Pt → L.two s = some d →
      InScopeTwo Pt d s → ∀ δ : ℝ,
        (influenceTwoAt s Pt d δ a).card ≤ ∑ T ∈ admissibleTwo Pt d δ, 2 ^ bitsOf s Pt T

end CompiledTwo

end Claims.SamplingGeneric
