module

public import Proofs.Flock.Claims.Recursion.DeployHonest

/-! The honest compiled run's inner rounds are the inner prover's messages (`run_msg_inner`). -/

@[expose] public section

namespace RecDeploy

open Claims (Msg Out Prover run)
open Flock
open FlockDeploy (Coin)

theorem refP_inner (IV : Claims.Verifier Coin) (b : Brings) (w : Claims.Flock.Wit) (x : Msg)
    (hist : List (Msg × Msg)) (ω : Tape b) (h : hist.length < IV.k (xIOf x)) :
    (refP IV b w).next x hist ω = (b.ref.PI w).next (xIOf x) hist ω.1 := by
  simp only [refP, h, ↓reduceIte]

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

end RecDeploy
