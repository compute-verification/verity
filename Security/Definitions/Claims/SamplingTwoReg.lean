module

public import Definitions.Claims.SamplingTwo

/-!
# Claims: the two-stage audit's committed form

`Compiled.KSReg` at two stages: one event on the run's own coins, about traces the prover fixes before the coins that
sample them, so that the misses of a monitoring period's windows multiply. `CompiledTwo.KS` is about the extracted
boundary trace, which the extractor outputs after it has seen the judged run's coins; this file is about the committed
ones.

1. **The footprint at values** (`wrongInAt`, `footprintAt`): `wrongIn` and `footprint` read a witness only through its
   values (`Wit.val`); these read the boundary values themselves, so a trace with no openings has a footprint.
   `wrongAtBoundaryAt`: the replay blocks wrong at their boundary under those values, each holding an incorrect proof
   block at every interior. It reads the boundary values only.
2. **The committed traces.** The boundary's is `Compiled.regVal`: at each gate, what the run's extractions open there
   under the statement's roots most often, over the run's coins and the extractor's; it reads the statement, the prover
   and its tape. Replay block `ρ`'s interior's is `iregVal`: at each gate, what the extractions open under the interior
   roots the run shows for `ρ` most often (`iopensAt`), over the second stage's coins and the extractor's, at the run's
   coins off the second stage. The prover commits it after the first stage's coins and before the second stage's
   (`Sessions.interior`), so it reads the tape and the first stage's coins.
3. **The committed incorrect set** (`FregTwo`): over the grid of replay blocks and strata, the proof blocks incorrect at
   the committed boundary trace and their replay block's committed interior.
4. **The error** (`εregTwo`): `Compiled.εreg`'s four terms (the sessions' knowledge error, and the boundary trace's
   binding at `B_conf`, whose `Pr[B_conf]` also pays for one run's extractions opening an interior gate two ways) and
   the interior's binding across runs, `N·√Pr[B_conf,2]`, at the finder whose two runs share the first stage's coins
   (`BconfTwo`).
5. **The claims**: `CompiledTwo.KSRegTwo`, two conjuncts on the run's coins, the event and the threshold form;
   `CompiledTwo.HeadlineTwo`, the acceptance probability at the committed boundary trace's best interiors;
   `CompiledTwo.MissTwo`, the miss at the replay blocks wrong at the committed boundary; and the compiler's statements.

Statements only: definitions, no proofs.
-/

@[expose] public section

namespace Claims.Sampling

/-! ## 1. The footprint at values -/

open Classical in
/-- **The incorrect proof blocks of cell `(ρ, t)` at boundary values `val` and interior values `ι`**: each is incorrect
on replay block `ρ`'s values, its boundary's `val` and its interior's `ι` (`Prog.refIn` at the replay block). `wrongIn`
is its case at a witness's values, `val = w.val H Cm s`. -/
noncomputable def wrongInAt (s : AuditStatement) (Pt Pρ : Partition) (val : ℕ → Option Bool) (ρ t : ℕ)
    (ι : ℕ → Bool) : Finset ℕ :=
  (cell Pt Pρ ρ t).filter fun v => ¬ ∃ u r, Pt.blocks[v]? = some u ∧ Pρ.blocks[ρ]? = some r ∧
    s.prog.Correct (s.prog.refIn r val ι) u

/-- **The footprint at boundary values `val`** and interior values `ι`, one interior per replay block: cell `(ρ, t)`'s
incorrect proof blocks (`wrongInAt`). `footprint` is its case at a witness's values. -/
noncomputable def footprintAt (s : AuditStatement) (Pt Pρ : Partition) (val : ℕ → Option Bool) (ι : ℕ → ℕ → Bool)
    (ρ t : ℕ) : ℕ :=
  (wrongInAt s Pt Pρ val ρ t (ι ρ)).card

open Classical in
/-- **The replay blocks wrong at their boundary** under boundary values `val`: each holds an incorrect proof block at
every interior, so no interior committed after the first stage's coins hides it from the second stage. -/
noncomputable def wrongAtBoundaryAt (s : AuditStatement) (Pt Pρ : Partition) (val : ℕ → Option Bool) : Finset ℕ :=
  (Finset.range Pρ.blocks.length).filter fun ρ =>
    ∀ i : ℕ → Bool, ∃ t < Pt.strata.length, (wrongInAt s Pt Pρ val ρ t i).Nonempty

end Claims.Sampling

namespace Claims.SamplingGeneric

open Claims
open Claims.Sampling (AuditStatement Partition InScopeTwo forkOn ownLog wrongInAt footprintAt wrongAtBoundaryAt)

namespace CompiledTwo

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-! ## 2. The committed traces -/

open Classical in
/-- **What a run's extractions open under replay block `ρ`'s interior roots**, at the prover's run `(ω, r)` and the
extractor's coins `c` there: at gate `g`, the first opening there that passes under the interior roots the run shows
for `ρ` (`iroots`), among what `runExt` finds; `none` where none opens `g`, and on a statement that doesn't parse. -/
noncomputable def iopensAt (H : Msg → Msg) (L : CompiledTwo Coin A) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (r : Fin (L.V.k x) → Coin) (c : L.SlotEC x) (ρ g : ℕ) : Option Bool :=
  match L.parse x with
  | none => none
  | some s =>
    (((L.runExt x s r (run H P L.V x ω r).seen c).eval H (run H P L.V x ω)).1.find? fun p =>
      decide (p.1 = L.iroots x (run H P L.V x ω r).seen ρ) && p.2.pos == g && L.commit.Opens H p.1 p.2).map
        (·.2.val)

open Classical in
/-- **The committed interior** of replay block `ρ` at the prover's tape `ω` and the run's coins `r`, a definition nobody
runs: at gate `g`, the value the extractions open there under `ρ`'s interior roots most often (`iopensAt`), over the
second stage's coins and the extractor's coins, at `r`'s coins off the second stage (`true` on a tie, and where none
opens `g`). It reads the statement, the prover, `ω` and `r` off `stage2` only, so the prover fixes it after the first
stage's coins and before the second stage's. -/
noncomputable def iregVal (H : Msg → Msg) (L : CompiledTwo Coin A) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (r : Fin (L.V.k x) → Coin) (ρ g : ℕ) : Bool :=
  let p : Bool → ℝ := fun b =>
    Pr fun ec : (Fin (L.V.k x) → Coin) × L.SlotEC x =>
      L.iopensAt H P x ω (forkOn (L.stage2 x) r ec.1) ec.2 ρ g = some b
  decide (p false ≤ p true)

/-! ## 3. The committed incorrect set -/

open Classical in
/-- **The committed incorrect set** at the prover's tape `ω` and the run's coins `r`: over the grid of replay blocks and
strata, the proof blocks incorrect at the committed boundary trace (`Compiled.regVal`) and their replay block's
committed interior (`iregVal`); empty on a statement that doesn't parse or whose partitions don't evaluate. It reads
`r` off the second stage only. -/
noncomputable def FregTwo (H : Msg → Msg) (L : CompiledTwo Coin A) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (r : Fin (L.V.k x) → Coin) : Finset ℕ :=
  match L.parse x with
  | none => ∅
  | some s =>
    match L.partition s, L.two s with
    | some Pt, some d =>
      (Finset.range d.part.blocks.length ×ˢ Finset.range Pt.strata.length).biUnion fun p =>
        wrongInAt s Pt d.part (L.regVal H P x ω) p.1 p.2 (L.iregVal H P x ω r p.1)
    | _, _ => ∅

/-! ## 4. The error -/

/-- **`B_conf,2`**: the prover's tape, the run's coins, two independent second stages at them, each with the extractor's
coins there, and the first collision among the calls the two runs' extractions make themselves. Two runs that share
the first stage's coins show one interior root for each replay block (`Sessions.interior`), so two extractions that
open one interior gate two ways give one; two independent runs (`Compiled.Bconf`'s) show different interior roots and
give none. Its cost is `Compiled.Bconf`'s at those two runs. -/
noncomputable def BconfTwo (H : Msg → Msg) (L : CompiledTwo Coin A) : FinderFamily := fun Ω _ _ P x =>
  match L.parse x with
  | none => { S := Unit, out := fun _ => none, cost := fun _ => 0 }
  | some s =>
    { S := Ω × (Fin (L.V.k x) → Coin) × ((Fin (L.V.k x) → Coin) × L.SlotEC x) ×
        ((Fin (L.V.k x) → Coin) × L.SlotEC x)
      ne := ⟨(Classical.arbitrary _, fun _ => default, (fun _ => default, Classical.arbitrary _),
        (fun _ => default, Classical.arbitrary _))⟩
      out := fun q =>
        findColl (ownLog H (run H P L.V x q.1) (L.runExt x s (forkOn (L.stage2 x) q.2.1 q.2.2.1.1)
            (run H P L.V x q.1 (forkOn (L.stage2 x) q.2.1 q.2.2.1.1)).seen q.2.2.1.2) ++
          ownLog H (run H P L.V x q.1) (L.runExt x s (forkOn (L.stage2 x) q.2.1 q.2.2.2.1)
            (run H P L.V x q.1 (forkOn (L.stage2 x) q.2.1 q.2.2.2.1)).seen q.2.2.2.2))
      cost := fun q =>
        (run H P L.V x q.1 (forkOn (L.stage2 x) q.2.1 q.2.2.1.1)).pLog.length +
          (run H P L.V x q.1 (forkOn (L.stage2 x) q.2.1 q.2.2.1.1)).vLog.length +
          (run H P L.V x q.1 (forkOn (L.stage2 x) q.2.1 q.2.2.2.1)).pLog.length +
          (run H P L.V x q.1 (forkOn (L.stage2 x) q.2.1 q.2.2.2.1)).vLog.length +
          ((L.runExt x s (forkOn (L.stage2 x) q.2.1 q.2.2.1.1)
            (run H P L.V x q.1 (forkOn (L.stage2 x) q.2.1 q.2.2.1.1)).seen q.2.2.1.2).eval H
              (run H P L.V x q.1)).2.1 +
          ((L.runExt x s (forkOn (L.stage2 x) q.2.1 q.2.2.2.1)
            (run H P L.V x q.1 (forkOn (L.stage2 x) q.2.1 q.2.2.2.1)).seen q.2.2.2.2).eval H
              (run H P L.V x q.1)).2.1 }

/-- **The committed form's error at two stages**: `Compiled.εreg`'s four terms and the committed interior's binding
across runs, `ε_reg,2 = κ_s + √(k(x)·γ_s·Pr[B]) + N·√Pr[B_conf] + Pr[B_conf] + N·√Pr[B_conf,2]`, `N` the program's
gates. Each gate is in at most one replay block, whose interior alone `Prog.refIn` reads there, so `N` bounds the
interior positions as well as the boundary's. -/
noncomputable def εregTwo (H : Msg → Msg) (L : CompiledTwo Coin A) (x : Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) : ℝ :=
  L.εreg H x P + L.nG x * Real.sqrt ((BconfTwo H L Ω P x).Pr H)

/-! ## 5. The claims -/

/-- **KS, the committed form at two stages**, on the statements in scope, at every prover: the run accepts with a
sampled proof block in the committed incorrect set (`FregTwo`, which the prover's tape and the first stage's coins fix
before the second stage's) with probability at most `ε_reg,2` (`εregTwo`); so, by the two stages' samples, at every
`δ ≥ 0` it accepts with no interiors putting the committed boundary trace's footprint in `M̄_δ` with probability at
most `δ + ε_reg,2`. -/
def KSRegTwo (H : Msg → Msg) (L : CompiledTwo Coin A) : Prop :=
  ∀ x, L.Scope x → ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω),
    Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
        ∃ u ∈ L.checked x p.2, u ∈ L.FregTwo H P x p.1 p.2) ≤ L.εregTwo H x P ∧
      ∀ s Pt d, L.parse x = some s → L.partition s = some Pt → L.two s = some d → ∀ δ : ℝ, 0 ≤ δ →
        Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
          ∀ ι : ℕ → ℕ → Bool, footprintAt s Pt d.part (L.regVal H P x p.1) ι ∉ d.M Pt δ) ≤
          δ + L.εregTwo H x P

/-- **The headline at two stages**: on a statement in scope, at every prover, the run accepts with probability at most
the expectation over the prover's tape of the two-stage design's `P̄_a` at the committed boundary trace's footprint at
its best interiors, plus `ε_reg,2`. A bound `pb ω` on `P̄_a` at every interior stands for the best interiors'. -/
def HeadlineTwo (H : Msg → Msg) (L : CompiledTwo Coin A) : Prop :=
  ∀ x s Pt d, L.Scope x → L.parse x = some s → L.partition s = some Pt → L.two s = some d →
    ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (pb : Ω → ℝ),
      (∀ ω ι, d.Pa Pt (footprintAt s Pt d.part (L.regVal H P x ω) ι) ≤ pb ω) →
        Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc) ≤ Ex pb + L.εregTwo H x P

/-- **The miss at the replay blocks wrong at the committed boundary**: on a statement in scope whose first stage
catches each replay block at rate `r₁` and whose second stage samples each cell at rate `r₂`, at every prover and tape,
the run's sample holds no proof block of the committed incorrect set with probability at most `(1 − r₁·r₂)^{|W|}`, `W`
the replay blocks wrong at the committed boundary trace (`wrongAtBoundaryAt`), which the tape fixes before any coin. -/
def MissTwo (H : Msg → Msg) (L : CompiledTwo Coin A) (r₁ r₂ : ℝ) : Prop :=
  ∀ x s Pt d, L.Scope x → L.parse x = some s → L.partition s = some Pt → L.two s = some d →
    d.FirstRate r₁ → d.SecondRate Pt r₂ → ∀ (Ω : Type) (P : Prover Ω) (ω : Ω),
      Pr (fun r : Fin (L.V.k x) → Coin => ∀ u ∈ L.checked x r, u ∉ L.FregTwo H P x ω r) ≤
        (1 - r₁ * r₂) ^ (wrongAtBoundaryAt s Pt d.part (L.regVal H P x ω)).card

end CompiledTwo

/-- **The two-stage compiler's committed form**: at `CompilerKSTwo`'s premises, every compiled two-stage verifier meets
`CompiledTwo.KSRegTwo`. -/
def CompilerKSRegTwo : Prop :=
  ∀ (Coin : Type) [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (A : Arg Coin) (L : CompiledTwo Coin A)
    (H : Msg → Msg), A.KS H → L.commit.Binds H → L.Sessions H → L.KSRegTwo H

/-- **The two-stage miss bound**: every compiled two-stage verifier that meets the contract meets
`CompiledTwo.MissTwo` at all rates `0 ≤ r₁ ≤ 1` and `0 ≤ r₂ ≤ 1`, from its draw laws (`replayDraws`, `secondDraws`). -/
def CompilerMissTwo : Prop :=
  ∀ (Coin : Type) [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (A : Arg Coin) (L : CompiledTwo Coin A)
    (H : Msg → Msg), L.Sessions H → ∀ r₁ r₂ : ℝ, 0 ≤ r₁ → r₁ ≤ 1 → 0 ≤ r₂ → r₂ ≤ 1 → L.MissTwo H r₁ r₂

/-- **The two-stage headline from the committed form**: every compiled two-stage verifier that meets the contract and
`CompiledTwo.KSRegTwo` meets `CompiledTwo.HeadlineTwo`, by its draw laws and the best interiors. -/
def CompilerHeadlineTwo : Prop :=
  ∀ (Coin : Type) [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (A : Arg Coin) (L : CompiledTwo Coin A)
    (H : Msg → Msg), L.Sessions H → L.KSRegTwo H → L.HeadlineTwo H

end Claims.SamplingGeneric
