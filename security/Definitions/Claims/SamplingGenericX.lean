module

public import Definitions.Claims.SamplingGeneric

/-!
# Claims: the committed form at a trace read at the statement

`Compiled.KSReg`'s committed trace, `Compiled.regVal`, is read at the prover's tape: tape by tape, the plurality of what
the runs' extractions open over the run's coins and the extractor's. So whatever the tape holds reaches it. A prover
whose tape holds values fixed only after its registration has a committed trace, and an incorrect set, that those values
move, and no trace read at one tape is fixed by the registration: a prover that opens a gate at such a value in few
of its runs has that value as the gate's plurality at every tape, and two runs at one tape never disagree there.

This file states the committed form at a trace read at the statement, beside `Compiled.KSReg`, which it leaves as it
stands:

1. **The trace** (`Compiled.regValX`): at each gate, the value the runs' extractions open there most often over the
   tape, the run's coins and the extractor's coins together. It reads the statement, which holds the roots every run
   opens under, and the prover's strategy; it reads no tape.
2. **Its incorrect set** (`Compiled.FregX`): `Compiled.Freg`'s form at that trace, one set at every tape.
3. **Its finder** (`Compiled.BconfX`): two independent runs, each at its own tape. Two runs at one statement open under
   one roots, so two that open one gate two ways give a collision, whatever their tapes.
4. **Its error** (`Compiled.εregX`): `Compiled.εreg` at `B_conf,X` in both binding terms.
5. **The claim** (`Compiled.KSRegX`): `Compiled.KSReg`'s two conjuncts at `FregX` and `εregX`.
6. **The compiler's statement** (`CompilerKSRegX`): at `CompilerKSReg`'s premises.

Statements only: definitions, no proofs.
-/

@[expose] public section

namespace Claims.SamplingGeneric

open Claims
open Claims.Sampling (AuditStatement ownLog)

namespace Compiled

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-! ## 1. The trace -/

open Classical in
/-- **The committed trace read at the statement**, a definition nobody runs: at gate `g`, the value the prover's runs'
extractions open there most often (`opensAt`) over the prover's tape, the run's coins and the extractor's coins together
(`true` on a tie); `none` where no run's extraction opens `g` at any tape. It is `regVal` with the tape moved into the
table: it reads the statement, under whose roots every run opens, and the prover's strategy, and no tape, so nothing
the tape holds moves it. -/
noncomputable def regValX (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} [Fintype Ω] (P : Prover Ω) (x : Msg)
    (g : ℕ) : Option Bool :=
  let p : Bool → ℝ := fun b =>
    Pr fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => L.opensAt H P x q.1 q.2.1 q.2.2 g = some b
  if p true = 0 ∧ p false = 0 then none else some (decide (p false ≤ p true))

/-! ## 2. Its incorrect set -/

open Classical in
/-- **The incorrect set of the committed trace read at the statement**: the proof blocks that no values agreeing with
that trace on the program's gates, and no interior values, make correct (`Freg`'s form at `regValX`); empty on a
statement that doesn't parse. It is one set at every tape: it reads the statement and the prover's strategy only. -/
noncomputable def FregX (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} [Fintype Ω] (P : Prover Ω) (x : Msg) :
    Finset ℕ :=
  match L.parse x with
  | none => ∅
  | some s => (Finset.range (L.part s).blocks.length).filter fun v =>
      ¬ ∃ b, (L.part s).blocks[v]? = some b ∧ ∃ (val : ℕ → Option Bool) (ι : ℕ → Bool),
        (∀ g < s.prog.gates.length, val g = L.regValX H P x g) ∧ s.prog.Correct (s.prog.refIn b val ι) b

/-! ## 3. Its finder -/

/-- **`B_conf,X`**: two independent runs, each at its own tape with its coins and the extractor's coins there, and the
first collision among the calls the two runs' extractions make themselves (the openings they check, `A`'s own). Both
runs open under the statement's roots, so two whose extractions open one gate two ways give one
(`CommitmentScheme.Binds`), whatever their tapes; `Bconf`'s two runs share a tape. Its cost is `Bconf`'s at those two
runs: the SHA-512 calls of both runs, both parties', and of both runs' extractions, their reruns' included. -/
noncomputable def BconfX (H : Msg → Msg) (L : Compiled Coin A) : FinderFamily := fun Ω _ _ P x =>
  match L.parse x with
  | none => { S := Unit, out := fun _ => none, cost := fun _ => 0 }
  | some s =>
    { S := (Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) × (Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x))
      ne := ⟨((Classical.arbitrary _, (fun _ => default, Classical.arbitrary _)),
        (Classical.arbitrary _, (fun _ => default, Classical.arbitrary _)))⟩
      out := fun q =>
        findColl (ownLog H (run H P L.V x q.1.1)
            (L.runExt x s q.1.2.1 (run H P L.V x q.1.1 q.1.2.1).seen q.1.2.2) ++
          ownLog H (run H P L.V x q.2.1) (L.runExt x s q.2.2.1 (run H P L.V x q.2.1 q.2.2.1).seen q.2.2.2))
      cost := fun q =>
        (run H P L.V x q.1.1 q.1.2.1).pLog.length + (run H P L.V x q.1.1 q.1.2.1).vLog.length +
          (run H P L.V x q.2.1 q.2.2.1).pLog.length + (run H P L.V x q.2.1 q.2.2.1).vLog.length +
          ((L.runExt x s q.1.2.1 (run H P L.V x q.1.1 q.1.2.1).seen q.1.2.2).eval H (run H P L.V x q.1.1)).2.1 +
          ((L.runExt x s q.2.2.1 (run H P L.V x q.2.1 q.2.2.1).seen q.2.2.2).eval H (run H P L.V x q.2.1)).2.1 }

/-! ## 4. Its error -/

/-- **The committed form's error at the trace read at the statement**:
`ε_reg,X = κ_s + √(k(x)·γ_s·Pr[B]) + N·√Pr[B_conf,X] + Pr[B_conf,X]`, `εreg` at `B_conf,X` in both binding terms. The
rewinding lemma's table is over the tape as well as the coins, so its conflicts are between two runs at two tapes
(`BconfX`). The last term, one run's extractions that open a position two ways, is at `BconfX` too, whose first run
it is: the error then reads one finder, so a reader takes binding at one finder family, not at `Bconf` beside it. -/
noncomputable def εregX (H : Msg → Msg) (L : Compiled Coin A) (x : Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) : ℝ :=
  L.κs x + Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) + L.nG x * Real.sqrt ((BconfX H L Ω P x).Pr H) +
    (BconfX H L Ω P x).Pr H

/-! ## 5. The claim -/

/-- **KS, the committed form at the trace read at the statement**, on the statements in scope, at every prover: the
run accepts with a sampled proof block in that trace's incorrect set (`FregX`) with probability at most `ε_reg,X`
(`εregX`); so, by the sample, it accepts with that set outside `M_δ` with probability at most `δ + ε_reg,X`, at every
`δ ≥ 0`. It is `KSReg` at `FregX`, which is one set at every tape: whatever fixes the statement and the prover's
strategy fixes it, so a value the tape holds that is fixed only after the registration doesn't move it, as it can
move `Freg`. -/
def KSRegX (H : Msg → Msg) (L : Compiled Coin A) : Prop :=
  ∀ x, L.Scope x → ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω),
    Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
        ∃ u ∈ L.checked x p.2, u ∈ L.FregX H P x) ≤ L.εregX H x P ∧
      ∀ s, L.parse x = some s → ∀ δ : ℝ, 0 ≤ δ →
        Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
          L.FregX H P x ∉ s.design.M (L.part s) δ) ≤ δ + L.εregX H x P

end Compiled

/-! ## 6. The compiler's statement -/

/-- **The committed form at the trace read at the statement**: as `CompilerKSReg`, `KSRegX`, at the same premises. The
rewinding lemma (`FlockSoundness.Rewinding.prBad_le`) is read at the table over the tape, the run's coins and the
extractor's coins: a run's extractions open a gate off its plurality with probability at most `N·√Pr[B_conf,X]`. -/
def CompilerKSRegX : Prop :=
  ∀ (Coin : Type) [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (A : Arg Coin) (L : Compiled Coin A)
    (H : Msg → Msg), A.KS H → L.commit.Binds H → L.Sessions H → L.KSRegX H

end Claims.SamplingGeneric
