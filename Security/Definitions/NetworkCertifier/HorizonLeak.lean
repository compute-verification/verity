import Definitions.NetworkCertifier.SampledCharge

/-!
# A monitoring period of sampled audits: each window's leak relation at its witness's faulty set

Definitions only. `RLeak δ` puts a window's observations among the sequences accepted with some faulty set that `M_δ`
allows (`AccSeqsSampled`), so a monitoring period charged by it pays `log2 #faultySets(δ)` and `δ` in every window. Over
a monitoring period the faulty set that matters is the one of the witness computed from each window's registration,
before its draw: the draws bound how many units those sets hold in total, and the count pays for which units they are
once. `LeakAt` is the leak relation at one faulty set, and `HorizonGood` is every window's at its own set, with the
sets' total below a bound.
-/

namespace NetTiming

open Claims Claims.Sampling

/-- **A window's leak relation at the faulty set `F`**: at the window registered as `s` (partition `P`, values `ρ` at
the registered inputs, operator `op`), the program fits its layout over the audit's scope `k s`, and the observations
`o` have that length, jitter below `Dj`, and are accepted with the faulty set `F` and the registered inputs pinned to
`ρ` (`AcceptedFaulty` at `AsReg`). -/
def LeakAt {L K S : ℕ} (lay : Stmt → Layout L S) (k : Stmt → ℕ) (reg : Stmt → Finset ℕ) (Dj : ℕ) (s : Stmt)
    (P : Partition) (ρ : ℕ → Bool) (op : ℕ → List (Obs L K S) → List Bool) (F : Finset ℕ)
    (o : List (Obs L K S)) : Prop :=
  (lay s).Fits s.prog (k s) ∧ o.length = k s ∧ (∀ y ∈ o, y.jitter < Dj) ∧
    AcceptedFaulty s.prog P (lay s) (k s) (AsReg s.prog (lay s) (k s) (reg s) ρ) F op o

/-- **A monitoring period's good event**: over the windows registered as `sw i`, the faulty sets `W i` hold fewer than
`Fmax` units in total, and every window's observations `o i` meet its leak relation at its own set (`LeakAt`). -/
def HorizonGood {L K S m : ℕ} (lay : Stmt → Layout L S) (k : Stmt → ℕ) (reg : Stmt → Finset ℕ) (Dj : ℕ)
    (sw : Fin m → Stmt) (Pw : Fin m → Partition) (ρ : Fin m → ℕ → Bool) (op : Fin m → ℕ → List (Obs L K S) → List Bool)
    (Fmax : ℕ) (W : Fin m → Finset ℕ) (o : Fin m → List (Obs L K S)) : Prop :=
  ∑ i, (W i).card < Fmax ∧ ∀ i, LeakAt lay k reg Dj (sw i) (Pw i) (ρ i) (op i) (W i) (o i)

end NetTiming
