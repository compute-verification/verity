import Definitions.Core.Interaction

/-!
# Claims: zero knowledge with a public outcome

The claims core's one zero-knowledge notion for a protocol whose outcome is public (each verdict, and where a session
stopped): per outcome, the honest transcript's probability of the outcome and an event is at most a simulator's,
given the outcome, of the event, plus an error of the statement, the witness, the verifier and the outcome.

* `SimOut`: a simulator given the outcome, fixed before the witness and the verifier.
* `ZKOut`: zero knowledge per outcome (`ZKmalRx`'s one-sided form at each outcome).

Its leak form over finitely many outcomes, a lemma of every `ZKOut`, is `Claims.zkOutLeak`
(`Proofs.Flock.Claims.ZKOut`).
-/

namespace Claims

/-- **A simulator given the outcome**: its coins, and its program, which also receives the protocol's public outcome
`o ∈ St` (each verdict, and where a session stopped). -/
structure SimOut (St : Type) where
  C : Msg → Type
  [fin : ∀ x, Fintype (C x)]
  [ne : ∀ x, Nonempty (C x)]
  prog : (x : Msg) → St → C x → SP (List (Msg × Out))

attribute [instance] SimOut.fin SimOut.ne

/-- The simulator given the outcome `o`, as the spec's `Simulator`. -/
def SimOut.given {St : Type} (S : SimOut St) (o : St) : Simulator where
  C := S.C
  prog x c := S.prog x o c

/-- **Zero knowledge with a public outcome**, per outcome (`ZKmalRx`'s one-sided form at each outcome): for `(x, w) ∈ R`,
every verifier `V*` of the protocol's `k(x)` rounds and tape `θ`, every outcome `o` and every event, the honest
transcript's probability of showing the outcome `o` and the event is at most the simulator's, given `o`, of the event,
plus `ε(x, w, V*, θ, o)`. `out` reads the outcome off the transcript, so the outcome is public; the simulator is fixed
before the witness and the verifier; `ε` is a function of the statement, the witness, the verifier and the outcome, such
as a collision finder's success built from them (`Fork`), never an assumed bound. -/
def ZKOut {W St : Type} (N Ω : Msg → Type) [∀ x, Fintype (N x)] [∀ x, Nonempty (N x)] [∀ x, Fintype (Ω x)]
    [∀ x, Nonempty (Ω x)] (H : Msg → Msg) (k : Msg → ℕ) (P : W → (x : Msg) → Prover (N x × Ω x)) (R : Rel W)
    (out : Msg → List (Msg × Out) → St) (S : SimOut St) (ε : (x : Msg) → W → (Θ : Type) → VStar Θ → Θ → St → ℝ) :
    Prop :=
  ∀ x w, R x w → ∀ (Θ : Type) (Vs : VStar Θ) (θ : Θ), Vs.k x = k x → ∀ (o : St) (Ev : List (Msg × Out) → Prop),
    Pr (fun ω : N x × Ω x =>
        out x (runStar H (P w x) Vs x ω θ).msgs = o ∧ Ev (runStar H (P w x) Vs x ω θ).msgs) ≤
      Pr (fun c : S.C x => Ev ((S.given o).out H Vs θ x c)) + ε x w Θ Vs θ o

end Claims
