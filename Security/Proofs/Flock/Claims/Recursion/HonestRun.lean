import Proofs.Flock.Claims.Recursion.DeployHonest

/-!
# The honest compiled run's inner rounds are the inner prover's messages (`run_msg_inner`)

On the compiled verifier's run of the reference prover `refP` at a statement that reads, each inner round's message
is the inner prover's (`b.ref.PI w`) at the inner statement, after the run's earlier inner rounds, on the inner
prover's tape `ω.1`: `refP` hands the inner rounds to the inner prover, and the compiled verifier answers each with its
coin, so the run gets past them (`run_length_ge`).
-/

namespace RecDeploy

open Claims (Msg Out Prover run)
open Flock
open FlockDeploy (Coin)

/-- **Before the staging round, the reference prover is the inner prover** at the inner statement, on its tape. -/
theorem refP_inner (IV : Claims.Verifier Coin) (b : Brings) (w : Claims.Flock.Wit) (x : Msg)
    (hist : List (Msg × Msg)) (ω : Tape b) (h : hist.length < IV.k (xIOf x)) :
    (refP IV b w).next x hist ω = (b.ref.PI w).next (xIOf x) hist ω.1 := by
  simp only [refP, h, ↓reduceIte]

/-- **The run lemma**: on the compiled verifier's run of `refP` at a statement that reads, inner round `i`'s message
is the inner prover's at the inner statement, after the run's first `i` inner rounds (`hist`), on tape `ω.1`. -/
theorem run_msg_inner (H : Msg → Msg) (IV : Claims.Verifier Coin) (b : Brings) (w : Claims.Flock.Wit) {x : Msg}
    {xr : XR} (hx : readX x = some xr) (ω : Tape b) (r : Fin ((V IV b.st).k x) → Coin) :
    ∀ i < IV.k (xIOf x), ((run H (refP IV b w) (V IV b.st) x ω r).msgs.map Prod.fst).getD i [] =
      (((b.ref.PI w).next (xIOf x)
        ((((run H (refP IV b w) (V IV b.st) x ω r).msgs.take (IV.k (xIOf x))).map histOf).take i) ω.1).eval H).1 := by
  intro i hi
  have hlen := run_length_ge H IV b.st hx (refP IV b w) ω r
  obtain ⟨new, hnew, hget⟩ := run_get H (refP IV b w) (V IV b.st) x ω r i (by omega)
  have hh : (((run H (refP IV b w) (V IV b.st) x ω r).msgs.take (IV.k (xIOf x))).map histOf).take i =
      new.map histOf := by
    rw [hnew, ← List.map_take, List.take_take, Nat.min_eq_left hi.le]
  have hnl : (new.map histOf).length = i := by
    rw [List.length_map, hnew, List.length_take]
    omega
  rw [hh, ← refP_inner IV b w x (new.map histOf) ω (by omega), List.getD_eq_getElem?_getD, List.getElem?_map, hget]
  rfl

/-- **The run lemma at the deployed record's verifier**: `run_msg_inner` at `(deployRec IV ivScope b).V`, which is
`V IV b.st`. -/
theorem run_msg_inner_deploy (H : Msg → Msg) (IV : Claims.Verifier Coin) (ivScope : Msg → Bool) (b : Brings)
    (w : Claims.Flock.Wit) {x : Msg} {xr : XR} (hx : readX x = some xr) (ω : Tape b)
    (r : Fin ((deployRec IV ivScope b).V.k x) → Coin) :
    ∀ i < IV.k (xIOf x), ((run H (refP IV b w) (deployRec IV ivScope b).V x ω r).msgs.map Prod.fst).getD i [] =
      (((b.ref.PI w).next (xIOf x)
        ((((run H (refP IV b w) (deployRec IV ivScope b).V x ω r).msgs.take (IV.k (xIOf x))).map histOf).take i)
          ω.1).eval H).1 :=
  run_msg_inner H IV b w hx ω r

end RecDeploy
