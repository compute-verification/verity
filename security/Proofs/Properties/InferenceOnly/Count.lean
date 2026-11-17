module

public import Properties.InferenceOnly.Count
public import Proofs.Slowdown.Cut

/-! Once `DescCountsCK` holds, a published `complies` of the count of `(c, k)`-inference blocks gives the day's bound
under (H) and (E). -/

@[expose] public section

namespace InferenceOnlyCheck.Properties.SecurityProofs

open Lean (Json)
open Flock.Program (Prog)

theorem complies_countHE (hD : DescCountsCK) {j : Json}
    (hj : (Count.check j).getObjVal? "complies" = .ok (.bool true)) (i : Count.Instance) (P : Prog)
    (d : Verity.RegisteredCircuit) (lw : Lowering P i.gatesOf d.D) (hp : CountParses j i)
    (hdec : Flock.Program.decode i.program.toUTF8 = .ok P) (hB : d.D.Boolean)
    (hθ : ∀ b, b ∈ d.θ ↔ ∃ g, b ∈ lw.L g ∧ InTheta P i.theta g)
    (ι V : Type) [Fintype ι] [DecidableEq ι] [DecidableEq V] (W : Slowdown.Cut.Work d.D ι)
    (val : Finset (Fin d.D.N) → V) (βw βx b : ℝ) (hw : 0 < βw) (hx : 0 < βx) (hb : 0 < b)
    (hval : Slowdown.Cut.ValuesInjective W val) (hshare : Slowdown.Cut.NoSharedPair W val)
    (hH : Slowdown.Cut.Assumptions.Hardness d ![W.allW, W.allX] val ![βw, βx] i.k) (hE : Slowdown.Cut.Assumptions.Exports W b) :
    (W.N : ℝ) ≤ Slowdown.Cut.workHE i.n i.k (βw * βx) b W.O ∧
      (i.c / d.D.size d.D.computed : ℝ) * (W.N / (i.k + Slowdown.Cut.alphaHE i.k (βw * βx) b) -
          i.k / 2 * √(W.O * W.N / (βw * βx)) / (i.k + Slowdown.Cut.alphaHE i.k (βw * βx) b) ^ (3 / 2 : ℝ)) ≤
        i.n * i.c / d.D.size d.D.computed :=
  Slowdown.Cut.SecurityProofs.CountHE ι V d W val i.c i.k i.n βw βx b hw hx hb hval hshare hH hE
    (hD j hj i P d lw hp hdec hB hθ)

end InferenceOnlyCheck.Properties.SecurityProofs
