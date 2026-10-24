import Definitions.Pouw.PearlC.Game

/-! The proof-only part of `Definitions.Pouw.PearlC.Game`: the definitions and lemmas that no guarantee
statement or assumption reads. -/

namespace Pouw.PearlC

open Finset Pouw.Fp8Atom

variable {Q R S : Type}

open Classical in
/-- An FP32 word with value `v` (some word, when one exists; `0` otherwise). -/
noncomputable def wordOf (v : Option ℚ) : ℕ := if h : ∃ w, wordQ w = v then Classical.choose h else 0

/-- The honest transcript on activations `act`. -/
noncomputable def honestTranscript (P : Protocol Q R S) (H : Q → R) (s : S) (U : Workload) (act : ℕ → Codes) :
    Transcript :=
  fun u => if u < U.layout.N then
    some ⟨act u, fun i j => wordOf (P.ticket H s U u (act u) i j), fun i j => wordOf (P.useful H s U u (act u) i j)⟩
  else none

/-- The cost model runs the honest reference: one program outputs the honest transcript on every in-domain workload
whose weights pass the rules and admissible activations, with no preprocessing, at cost at most `∑_u W_ref(u)` and at
most `qref U` calls. -/
def AdmitsRef (CM : CostModel Q R S) (P : Protocol Q R S) (D : Layout → Prop) (qref : Workload → ℕ) : Prop :=
  ∃ prog : CM.Prog, ∀ U : Workload, U.InDomain P D → U.WeightsOK P → ∀ act : ℕ → Codes,
    (∀ u < U.layout.N, P.actOK (U.layout.shape u) (act u)) → ∀ (H : Q → R) (s : S),
      CM.run prog H U act [] s = honestTranscript P H s U act ∧
      CM.cost prog H U act [] s ≤ ∑ u ∈ range U.layout.N, P.Wref U.layout u ∧
      CM.calls prog H U act [] s ≤ qref U

/-- As `adopted`, with the model's quantizer also emitting ρ. -/
def Costs.adoptedFull : Costs := ⟨64, 0⟩

/-- Under the model's current plain max scale: `α·x` (an FP32 multiply, 32) and ρ (4) are A-only protocol work. -/
def Costs.current : Costs := ⟨64, 36⟩

/-- The credited work when forming is not credited at all. -/
def creditChainOnly (p : Params) (c : Costs) (s : Shape) : ℚ := creditOf p c s - (c.fs : ℚ) * s.m * s.k

/-- The reduction's gap: `γ = 1 − (1 − γ₀)/Ω`. -/
def gammaOf (γ₀ Ω : ℚ) : ℚ := 1 - (1 - γ₀) / Ω

/-- `Π`'s costs are Pearl-C's accounting at `(p, c)` with cap `ρ`: `W_ref` is `wrefOf`, and a unit within the cap is
credited at least `(1 − ρ)·creditOf`. -/
def PearlCAccounting (P : Protocol Q R S) (p : Params) (c : Costs) (ρ : ℚ) : Prop :=
  (∀ (L : Layout) (u : ℕ), P.Wref L u = (wrefOf p c (L.shape u) : ℝ)) ∧
    ∀ (H : Q → R) (s : S) (U : Workload) (u : ℕ) (act : Codes), P.capOK H s U u act →
      (((1 - ρ) * creditOf p c (U.layout.shape u) : ℚ) : ℝ) ≤ P.credit H s U u act

/-- The domain restricted to one shape. -/
def pearlCDomainAt (p : Params) (s : Shape) (L : Layout) : Prop := pearlCDomain p L ∧ ∀ u < L.N, L.shape u = s

end Pouw.PearlC
