module

public import Proofs.Flock.SamplingGeneric
public import Definitions.Claims.SamplingTable

/-! The sampled compiler at Flock's extractor at one table of a session (`Arg.ofFlockTable`). -/

@[expose] public section

namespace Claims

section AtStmt

variable {Ω Coin : Type} [Inhabited Coin]

def Prover.atStmt (P : Prover Ω) (y : Msg) : Prover Ω := ⟨fun _ hist ω => P.next y hist ω⟩

omit [Inhabited Coin] in
theorem stepWith_atStmt (H : Msg → Msg) (P : Prover Ω) (y x : Msg) (ω : Ω) (vnext : List Msg → OC Out) :
    ∀ (n : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t : Transcript),
      stepWith H (P.atStmt y) x ω vnext n hist as t = stepWith H P y ω vnext n hist as t := by
  intro n
  induction n with
  | zero => intro _ _ _; rfl
  | succ n ih =>
    intro hist as t
    simp only [stepWith]
    have hP : (P.atStmt y).next x hist ω = P.next y hist ω := rfl
    rw [hP]
    generalize (vnext (as ++ [((P.next y hist ω).eval H).1])).eval H = vb
    obtain ⟨b, l⟩ := vb
    cases b with
    | msg m => exact ih _ _ _
    | acc => rfl
    | rej => rfl

theorem run_atStmt (H : Msg → Msg) (P : Prover Ω) (V : Verifier Coin) (f : Msg → Msg) (y : Msg) (ω : Ω) :
    run H (P.atStmt y) V (f y) ω = run H P ⟨fun y => V.k (f y), fun y => V.next (f y)⟩ y ω :=
  funext fun _ => stepWith_atStmt H P y (f y) ω _ _ _ _ _

end AtStmt

end Claims

namespace Claims.SamplingGeneric

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]

theorem flockArgKSTable : FlockArgKSTable (Coin := Coin) (Col := Col) (T := T) := by
  intro H D _ N ε names hKS hdec y Ω _ _ P
  have hrun : run H (P.atStmt y) D.V (names y).2 = run H P (Arg.ofFlockTable D N ε names).V y :=
    funext fun ω => run_atStmt H P D.V (fun y => (names y).2) y ω
  have h := hKS D.decode hdec (names y).1 (names y).2 Ω (P.atStmt y)
  unfold PrBad PrB B Extractor.eval at h
  rw [hrun] at h
  exact h

theorem flockKSTable : FlockKSTable (Coin := Coin) (Col := Col) (T := T) := by
  intro H D _ N ε names L hKS hdec hB hS
  exact compilerKS Coin (Arg.ofFlockTable D N ε names) L H (flockArgKSTable H D N ε names hKS hdec) hB hS

theorem flockKSRegTable : FlockKSRegTable (Coin := Coin) (Col := Col) (T := T) := by
  intro H D _ N ε names L hKS hdec hB hS
  exact compilerKSReg Coin (Arg.ofFlockTable D N ε names) L H (flockArgKSTable H D N ε names hKS hdec) hB hS

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] in
theorem flockRerunsTable : FlockRerunsTable (Coin := Coin) (Col := Col) (T := T) := fun _ _ _ _ _ => ⟨rfl, rfl⟩

end Claims.SamplingGeneric
