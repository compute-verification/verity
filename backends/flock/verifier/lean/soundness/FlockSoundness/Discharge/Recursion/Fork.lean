import FlockSoundness.Discharge.Recursion.Defs
import FlockSoundness.StrictCR

/-!
# Recursion: binding a stage to a reference opening fixed before its coin

The stage-`k` commitment `c_k` is a vector of hiding commitments, one digest per position `pos`. The outer session's
extraction reads openings of them (a **trial** at each leaf, `T t s pos`, which outputs the openings it read at `pos`).
The binding hypothesis Flock proves is a **transfer**: for every candidate value vector `v`, the decoded round-`k` message
differs from `msg v` with probability at most a term `εtr` plus, at each position, `κ pos` times the probability that the
trial reads an opening of `c_k`'s digest at `pos` to a value other than `v pos`.

The reference opening has to be a function of the prover's state when it sends `c_k`, so that the simulated inner prover
fixes its round-`k` message before `ρ_k`. A plurality over the extraction's openings at the leaf is not one: it depends on
`ρ_k`. This file fixes it at the node instead:

* `refv p c pos` is the value the rest of the run from the node (`contF`: the remaining coins, then the trial) reads
  **alone** at `pos` most often, the readings `A ⊆ V` counted when `A = {v}` (`alone_plurality_sq`);
* off that value, two independent continuations from the node conflict: if a run reads some `u ∈ A`, every run whose
  readings are nonempty and not `{u}` reads a value other than `u`, and by plurality those runs weigh at least the
  off-reference mass `b`; so `b² ≤ Pr₂[conflict]`;
* a conflict is two openings of one digest to different values, a collision of `H` that the **fork finder** (`forkF`:
  the coins to the node, then two continuations) outputs (`Openings.collide`);
* `(Σ b_node)² ≤ Σ b_node²` over the coins to the node (Cauchy–Schwarz), so with `cr/sha-512` for the fork finder at
  budget `q` the off-reference mass is at most `√(q²/2^513) = q/2^256.5`: the rewinding costs a square root.

`stage_bind` puts the two together: the stage's binding error is `leafAvg εtr + (Σ_pos κ pos)·√(q²/2^513)`.
-/

namespace FlockSoundness.Discharge.Recursion

open Game Audit StrictCR
open scoped ENNReal

/-- **Openings of a binding commitment**, as the stage binding reads them: an opening `o` opens the digest `com o` to the
value `val o`, and two openings of one digest to different values are a collision of `H` (`collide`), as
`ValueBindingZ`'s are. -/
structure Openings {D : Type} (H : List UInt8 → D) (V : Type) : Type 1 where
  Op : Type
  Dg : Type
  com : Op → Dg
  val : Op → V
  collide : Op → Op → List UInt8 × List UInt8
  collide_spec : ∀ o o', com o = com o' → val o ≠ val o' →
    (collide o o').1 ≠ (collide o o').2 ∧ H (collide o o').1 = H (collide o o').2

variable {D : Type} {H : List UInt8 → D} {V : Type} (ob : Openings H V)

/-- The openings `os` open the digest `d` to a value other than `v`. -/
def OffAt (d : ob.Dg) (v : V) (os : Finset ob.Op) : Prop := ∃ o ∈ os, ob.com o = d ∧ ob.val o ≠ v

variable {Cm C Pos β γ : Type} [Fintype C] [Nonempty C]

/-- **The rest of the run from a node**: the remaining round coins, the prover's messages read from its strategy, and then
the finder `T` at the leaf reached. -/
noncomputable def contF : (r : ℕ) → (f : List (Cm × C) → Game β) →
    ((t : List (Cm × C)) → Strategy (f t) → Finder γ) → Strategy (rounds Cm C r f) → Finder γ
  | 0, _, T, s => T [] s
  | r + 1, f, T, s =>
      ⟨.coin C fun ρ => (contF r (fun t => f ((s.1, ρ) :: t)) (fun t => T ((s.1, ρ) :: t)) (s.2 ρ)).g,
        fun ρ => (contF r (fun t => f ((s.1, ρ) :: t)) (fun t => T ((s.1, ρ) :: t)) (s.2 ρ)).s⟩

open Classical in
/-- **A fork's output**: an opening from each run, of the digest `d` to different values, as a collision. -/
noncomputable def forkPick (d : ob.Dg) (os os' : Finset ob.Op) : Pair :=
  if h : ∃ o ∈ os, ∃ o' ∈ os', ob.com o = d ∧ ob.com o' = d ∧ ob.val o ≠ ob.val o' then
    some (ob.collide h.choose h.choose_spec.2.choose)
  else none

/-- **The fork finder at stage `k` and position `pos`**: the round coins to the node where the prover sends its stage-`k`
commitment `c`, then the rest of the run twice, independently (`contF`, the trial `T … pos` at each leaf), and the two
runs' openings of `c`'s digest at `pos` to different values (`forkPick`). With no stage `k` it finds nothing. -/
noncomputable def forkF (cmt : Cm → Pos → ob.Dg) (pos : Pos) : (k r : ℕ) → (f : List (Cm × C) → Game β) →
    ((t : List (Cm × C)) → Strategy (f t) → Pos → Finder (Finset ob.Op)) → Strategy (rounds Cm C r f) → Finder Pair
  | _, 0, _, _, _ => ⟨.ret none, PUnit.unit⟩
  | 0, r + 1, f, T, s =>
      twoRuns (contF (r + 1) f (fun t s' => T t s' pos) s).g (contF (r + 1) f (fun t s' => T t s' pos) s).s
        (forkPick ob (cmt s.1 pos))
  | k + 1, r + 1, f, T, s =>
      ⟨.coin C fun ρ => (forkF cmt pos k r (fun t => f ((s.1, ρ) :: t)) (fun t => T ((s.1, ρ) :: t)) (s.2 ρ)).g,
        fun ρ => (forkF cmt pos k r (fun t => f ((s.1, ρ) :: t)) (fun t => T ((s.1, ρ) :: t)) (s.2 ρ)).s⟩

/-- **The probability, at a leaf, that the trial at `pos` reads `c_k`'s digest off the reference value.** -/
noncomputable def offProb (cmt : Cm → Pos → ob.Dg) (k : ℕ) (refv : List (Cm × C) → Cm → Pos → V)
    {f : List (Cm × C) → Game β} (T : (t : List (Cm × C)) → Strategy (f t) → Pos → Finder (Finset ob.Op))
    (t : List (Cm × C)) (s : Strategy (f t)) (pos : Pos) : ℝ≥0∞ :=
  match t[k]? with
  | some e => prob (OffAt ob (cmt e.1 pos) (refv (t.take k) e.1 pos)) (T t s pos).g (T t s pos).s
  | none => 0

/-- **The plurality reference and its fork**: some reference values, fixed at the node where each stage-`k` commitment is
sent, are read off by the trials with a mass whose square is at most the fork finder's success. -/
theorem fork_ref [Fintype V] [DecidableEq V] [Nonempty V] (cmt : Cm → Pos → ob.Dg) :
    ∀ (k r : ℕ) (f : List (Cm × C) → Game β)
      (T : (t : List (Cm × C)) → Strategy (f t) → Pos → Finder (Finset ob.Op)) (s : Strategy (rounds Cm C r f)),
    ∃ refv : List (Cm × C) → Cm → Pos → V, ∀ pos,
      (leafAvg r f (fun t s' => offProb ob cmt k refv T t s' pos) s).toReal ^ 2 ≤
        (forkF ob cmt pos k r f T s).ev (ind (IsColl H)) := by
  sorry

/-- **Stage binding.** Given the transfer `htr` at stage `k` and `cr/sha-512` for the fork finders at budget `q`, some
reference values fixed at the stage's nodes bind the decoded round-`k` message: `E t s m` is the leaf's probability that
the decoded message is not `m`. -/
theorem stage_bind [Fintype V] [DecidableEq V] [Nonempty V] [Fintype Pos] {Msg : Type}
    (cmt : Cm → Pos → ob.Dg) (msg : (Pos → V) → Msg) {k r : ℕ} (hk : k < r) (f : List (Cm × C) → Game β)
    (T : (t : List (Cm × C)) → Strategy (f t) → Pos → Finder (Finset ob.Op)) (s : Strategy (rounds Cm C r f))
    (E : (t : List (Cm × C)) → Strategy (f t) → Option Msg → ℝ≥0∞)
    (εtr : (t : List (Cm × C)) → Strategy (f t) → ℝ≥0∞) (κ : Pos → ℝ≥0∞)
    (htr : AllLeaves r f (fun t s' => ∀ (v : Pos → V) (e : Cm × C), t[k]? = some e →
      E t s' (some (msg v)) ≤ εtr t s' + ∑ pos, κ pos * prob (OffAt ob (cmt e.1 pos) (v pos)) (T t s' pos).g (T t s' pos).s) s)
    {q : ℝ} (hCR : ∀ pos, (forkF ob cmt pos k r f T s).CR H q) :
    ∃ refv : List (Cm × C) → Cm → Pos → V,
      leafAvg r f (fun t s' => E t s' ((t[k]?).map fun e => msg (refv (t.take k) e.1))) s ≤
        leafAvg r f εtr s + (∑ pos, κ pos) * ENNReal.ofReal (Real.sqrt (q ^ 2 / 2 ^ (513 : ℝ))) := by
  sorry

end FlockSoundness.Discharge.Recursion
