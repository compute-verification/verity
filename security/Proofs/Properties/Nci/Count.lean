module

public import Properties.Nci.Count
public import Proofs.Nci.Cut

/-!
# A published count bounds the day's work

`complies_countHE`: once `DescCountsCK` holds, a published `complies` of the count of `(c, k)`-inference blocks on the
day's instance gives the Boolean declaration the Program lowers to `Nci.CompliesAtMost c k n`, and `CountHE` turns
that verdict, with (H) and (E) on the declaration's work, into the day's bound: `N ≤ n (k + α) + (k/2) √(n O/(β_w β_x))`
and its slowdown form.
-/

@[expose] public section

namespace NciCheck.Properties.SecurityProofs

open Lean (Json)
open Flock.Program (Prog)

theorem complies_countHE (hD : DescCountsCK) {j : Json}
    (hj : (Count.check j).getObjVal? "complies" = .ok (.bool true)) (i : Count.Instance) (P : Prog)
    (d : Verity.RegisteredCircuit) (lw : Lowering P i.gatesOf d.D) (hp : CountParses j i)
    (hdec : Flock.Program.decode i.program.toUTF8 = .ok P) (hB : d.D.Boolean)
    (hθ : ∀ b, b ∈ d.θ ↔ ∃ g, b ∈ lw.L g ∧ InTheta P i.theta g)
    (ι V : Type) [Fintype ι] [DecidableEq ι] [DecidableEq V] (W : Nci.Cut.Work d.D ι)
    (val : Finset (Fin d.D.N) → V) (βw βx b : ℝ) (hw : 0 < βw) (hx : 0 < βx) (hb : 0 < b)
    (hval : Nci.Cut.ValuesInjective W val) (hshare : Nci.Cut.NoSharedPair W val)
    (hH : Nci.Cut.Assumptions.Hardness d ![W.allW, W.allX] val ![βw, βx] i.k) (hE : Nci.Cut.Assumptions.Exports W b) :
    (W.N : ℝ) ≤ Nci.Cut.workHE i.n i.k (βw * βx) b W.O ∧
      (i.c / d.D.size d.D.computed : ℝ) * (W.N / (i.k + Nci.Cut.alphaHE i.k (βw * βx) b) -
          i.k / 2 * √(W.O * W.N / (βw * βx)) / (i.k + Nci.Cut.alphaHE i.k (βw * βx) b) ^ (3 / 2 : ℝ)) ≤
        i.n * i.c / d.D.size d.D.computed :=
  Nci.Cut.SecurityProofs.CountHE ι V d W val i.c i.k i.n βw βx b hw hx hb hval hshare hH hE
    (hD j hj i P d lw hp hdec hB hθ)

end NciCheck.Properties.SecurityProofs
