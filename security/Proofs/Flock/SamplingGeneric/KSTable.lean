module

public import Proofs.Flock.SamplingGeneric
public import Definitions.Claims.SamplingTable

/-!
# The sampled compiler at Flock's extractor at one table of a session

`Arg.ofFlockTable` is Flock at one table of a session: its statement names a table and a Flock statement, its verifier
is Flock's at the named statement and its extractor `Flock.ETable` at the named table and the deployed decoder.

* **A verifier read at named statements** runs as itself against the prover that sends, at every statement, the
  messages it sends at the naming statement (`Prover.atStmt`, `stepWith_atStmt`, `run_atStmt`): a run reads the
  prover's statement only through its next message.
* **Its knowledge soundness is `Flock.KSTable`'s** at the named table and statement and the deployed decoder
  (`flockArgKSTable`): `Flock.KSTable` at that prover, with the runs rewritten.
* **The compiler at it** (`flockKSTable`, `flockKSRegTable`): `CompilerKS` and `CompilerKSReg` at `Arg.ofFlockTable`, as
  `flockKS` and `flockKSReg` are at `Arg.ofFlock`.
* **Its waits** (`flockRerunsTable`): at every statement's shape its waits are the session's and each stops at the named
  table's `k_j`, by unfolding. What `Compiled.es` over it is then: `OneVerifierTables.lean`.
-/

@[expose] public section

namespace Claims

section AtStmt

variable {Ω Coin : Type} [Inhabited Coin]

/-- **The prover at one statement**: at every statement it sends `P`'s messages at the statement `y`. -/
def Prover.atStmt (P : Prover Ω) (y : Msg) : Prover Ω := ⟨fun _ hist ω => P.next y hist ω⟩

omit [Inhabited Coin] in
/-- A run reads the prover's statement only through its next message: `P` at `y` runs as `P.atStmt y` at any `x`. -/
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

/-- **A verifier read at named statements runs as itself**: `P` against `V` read at `f y` for the statement `y` runs
as `P.atStmt y` against `V` at `f y`. -/
theorem run_atStmt (H : Msg → Msg) (P : Prover Ω) (V : Verifier Coin) (f : Msg → Msg) (y : Msg) (ω : Ω) :
    run H (P.atStmt y) V (f y) ω = run H P ⟨fun y => V.k (f y), fun y => V.next (f y)⟩ y ω :=
  funext fun _ => stepWith_atStmt H P y (f y) ω _ _ _ _ _

end AtStmt

end Claims

namespace Claims.SamplingGeneric

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]

/-- **`Arg.ofFlockTable` is knowledge sound from `Flock.KSTable`**: at a statement naming table `j` of the Flock
statement `x`, `Flock.KSTable` at `j`, `x` and the deployed decoder, read at the prover that sends this statement's
messages. -/
theorem flockArgKSTable : FlockArgKSTable (Coin := Coin) (Col := Col) (T := T) := by
  intro H D _ N ε names hKS hdec y Ω _ _ P
  have hrun : run H (P.atStmt y) D.V (names y).2 = run H P (Arg.ofFlockTable D N ε names).V y :=
    funext fun ω => run_atStmt H P D.V (fun y => (names y).2) y ω
  have h := hKS D.decode hdec (names y).1 (names y).2 Ω (P.atStmt y)
  unfold PrBad PrB B Extractor.eval at h
  rw [hrun] at h
  exact h

/-- **Flock at one table of a session is an instance**: the compiler at `Arg.ofFlockTable`, from `Flock.KSTable` and its
decoder's list bound. -/
theorem flockKSTable : FlockKSTable (Coin := Coin) (Col := Col) (T := T) := by
  intro H D _ N ε names L hKS hdec hB hS
  exact compilerKS Coin (Arg.ofFlockTable D N ε names) L H (flockArgKSTable H D N ε names hKS hdec) hB hS

/-- **Its committed form**: the compiler's at `Arg.ofFlockTable`. -/
theorem flockKSRegTable : FlockKSRegTable (Coin := Coin) (Col := Col) (T := T) := by
  intro H D _ N ε names L hKS hdec hB hS
  exact compilerKSReg Coin (Arg.ofFlockTable D N ε names) L H (flockArgKSTable H D N ε names hKS hdec) hB hS

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] in
/-- **Its waits**: the session's waits, each stopping at the named table's `k_j`. -/
theorem flockRerunsTable : FlockRerunsTable (Coin := Coin) (Col := Col) (T := T) := fun _ _ _ _ _ => ⟨rfl, rfl⟩

end Claims.SamplingGeneric
