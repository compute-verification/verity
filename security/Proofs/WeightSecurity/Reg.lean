module

public import Definitions.Claims.SamplingGeneric
public import Proofs.WeightSecurity.Stop

/-!
# A period of one-stage windows at the committed traces

The period over windows each audited by a compiled sampled verifier (`Compiled`), in the form `period_undetected`
takes. A window (`oneGame`): the datacenter commits what leaves on the egress slots, its timing bucket and its advice,
then sends the window's statement and prover, and the verifier flips the run's live coins. Its record reads the
committed trace (`Compiled.regVal`): its sample holds an incorrect proof block when it holds one of the committed
incorrect set (`Compiled.Freg`), and it counts that set, which the statement and prover its strategy sends fix before
the run's coins.

* `oneRegCount`: that count at a window's statement and prover, `0` out of scope.
* `oneRegRec`, `oneRegRun`: the record, and the deployment's run over `T` windows recorded by it.
* `PeriodReg`: the run leaves `M` incorrect proof blocks undetected with probability at most `Σ_t ε_t + x^M`, each
  `ε_t` bounding window `t`'s `ε_reg` at every strategy the run plays it with. It is `period_undetected` at
  `oneRegRec`: `hacc` is `Compiled.KSReg`'s first conjunct and `hstage` is the sample's miss of the committed incorrect
  set, at most `P_a(Freg)` by the contract's sample (`Compiled.Sessions.draws`).
* `CompilerPeriodReg`: `PeriodReg` at every run of window verifiers that meet `CompilerKSReg`'s premises, at sampling
  designs that miss every set `B` of proof blocks with probability at most `x^{|B|}` at every statement in scope.

Statements only: definitions, no proofs.
-/

@[expose] public section

namespace WeightSecurity.SecurityProofs

open Claims Claims.Sampling Claims.SamplingGeneric
open FlockSoundness.Game
open scoped ENNReal

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin} {O V Tick Adv : Type}

/-- A window's play at a one-stage audit: the statement and the prover the datacenter sent, and the run's coins. -/
abbrev OnePlay (L : Compiled Coin A) := (x : Msg) × Prover Unit × (Fin (L.V.k x) → Coin)

/-- **A window at a one-stage audit**: the datacenter commits the window's release, then sends the statement and the
prover, and the verifier flips the run's live coins. -/
noncomputable def oneGame (L : Compiled Coin A) (Rls : Type) : Verity.Game (Rls × OnePlay L) :=
  .send Rls fun m => .send (Msg × Prover Unit) fun xp =>
    .coin (Fin (L.V.k xp.1) → Coin) fun r => .ret (m, ⟨xp.1, xp.2, r⟩)

/-- The run accepts. -/
def OnePlay.accepted (H : Msg → Msg) {L : Compiled Coin A} (p : OnePlay L) : Prop :=
  (run H p.2.1 L.V p.1 () p.2.2).out = .acc

open Classical in
/-- **The incorrect proof blocks a window fixes**: at the statement `x` and the prover `P` its strategy sends, the
committed trace's incorrect set (`Compiled.Freg`); `0` on a statement out of scope. -/
noncomputable def oneRegCount (H : Msg → Msg) (L : Compiled Coin A) (x : Msg) (P : Prover Unit) : ℕ :=
  if L.Scope x then (L.Freg H P x ()).card else 0

/-- **A one-stage window's record at the committed traces**: accepted when the run accepts; sampled incorrect when its
sample holds a proof block of the committed incorrect set; incorrect, the count `n` its strategy fixes; and the
release. -/
noncomputable def oneRegRec (H : Msg → Msg) (L : Compiled Coin A) (n : ℕ)
    (b : ((O → V) × Tick × Adv) × OnePlay L) : Served O V Tick Adv where
  accepted := b.2.accepted H
  sampledIncorrect := ∃ u ∈ L.checked b.2.1 b.2.2.2, u ∈ L.Freg H b.2.2.1 b.2.1 ()
  incorrect := n
  egress := b.1.1
  tick := b.1.2.1
  adv := b.1.2.2

/-- **The deployment's run over `T` one-stage windows** at the datacenter's strategy `s`, each window recorded by
`oneRegRec` at the count its strategy fixes (`oneRegCount`). -/
noncomputable def oneRegRun (H : Msg → Msg) (Lv : ℕ → Compiled Coin A) (T : ℕ)
    (s : Strategy (periodG (fun t => oneGame (Lv t) ((O → V) × Tick × Adv)) 0 T)) :
    Verity.Game (List (Served O V Tick Adv)) :=
  periodRun (fun t => oneGame (Lv t) ((O → V) × Tick × Adv))
    (fun t τ => oneRegRec H (Lv t) (oneRegCount H (Lv t) τ.2.1.1 τ.2.1.2)) 0 T s

/-- **The period at the committed traces**: over `T` one-stage windows, at every strategy whose windows' committed-form
errors (`Compiled.εreg` at the statement and prover each sends) are at most `ε_t`, the run leaves `M` incorrect proof
blocks undetected with probability at most `Σ_t ε_t + x^M`: some window is accepted with its sample holding a proof
block of its committed incorrect set, or the windows before the first such sample hold at least `M` such blocks. -/
def PeriodReg (H : Msg → Msg) (Lv : ℕ → Compiled Coin A) (x : ℝ) : Prop :=
  ∀ (T : ℕ) (ε : ℕ → ℝ) (s : Strategy (periodG (fun t => oneGame (Lv t) ((O → V) × Tick × Adv)) 0 T)),
    (∀ t τ, PeriodPlays (fun t => oneGame (Lv t) ((O → V) × Tick × Adv)) 0 T s t τ →
      (Lv t).εreg H τ.2.1.1 τ.2.1.2 ≤ ε t) →
    ∀ M : ℕ, Verity.Game.value (Served.Undetected M) (oneRegRun H Lv T s) ≤
      ∑ t ∈ Finset.range T, ENNReal.ofReal (ε t) + ENNReal.ofReal (x ^ M)

/-- **The period bound from the windows' verifiers**: when every window's verifier meets `CompilerKSReg`'s premises and
every statement in its scope has a sampling design that misses each set `B` of its proof blocks with probability at
most `x^{|B|}`, `PeriodReg` holds (`period_undetected` at `oneRegRec`, from `CompilerKSReg` and that miss bound). -/
def CompilerPeriodReg : Prop :=
  ∀ (Coin : Type) [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (A : Arg Coin) (O V Tick Adv : Type)
    (H : Msg → Msg) (Lv : ℕ → Compiled Coin A) (x : ℝ), A.KS H →
    (∀ t, (Lv t).commit.Binds H ∧ (Lv t).Sessions H) → 0 ≤ x → x ≤ 1 →
    (∀ t y s P, (Lv t).parse y = some s → (Lv t).partition s = some P → InScope P s →
      ∀ B ⊆ Finset.range P.blocks.length, s.design.Pa P B ≤ x ^ B.card) →
    PeriodReg (O := O) (V := V) (Tick := Tick) (Adv := Adv) H Lv x

end WeightSecurity.SecurityProofs
