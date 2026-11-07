module

public import Proofs.Accounting.ReadUnit
public import Proofs.Accounting.ReadUnit.Tie
public import Proofs.Flock.Bridge.UnitRows
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init

/-!
# The read's gates refine its reference at a unit of the parsed circuit file

`gatesRefine_partsOf`: at unit `u` of the parsed file's partition (`Flock.Bridge.partsOf`), let the unit net's rows hold
the read's program (`HoldsRead`) and compute its reference wherever they hold (`hsound`). Then, where the stack's
constant gate is registered as 1 and is among `u`'s inputs, and copy `j`'s read outputs are committed, the read's gates
at copy `j` refine `Hash.EmbeddingRead` (`ReadUnit.GatesRefine`), at the ports through the copy's gates
(`Flock.Bridge.copyGate`). The values are those a correct unit's copy holds (`Flock.Bridge.rows_of_correct`).

`le_pow_depth` and `depth_le` give the tree bounds the read's statements take, for a table of at most `2 ^ 32` rows.

`readUnitDrawn_partsOf` is the read's binding at the parsed file: `Accounting.SecurityProofs.readUnitDrawnCR` at
`partsOf`, over the reads at copy `j`'s ports (`readsOf`), with its gate premise discharged by `gatesRefine_partsOf`
at every unit.
-/

@[expose] public section

namespace Accounting.ReadUnit.Gates

open Flock.HmNets hiding and
open FlockSoundness.Discharge.Hm

/-- A table's rows fit its tree's leaves. -/
theorem le_pow_depth (V : ℕ) : V ≤ 2 ^ Hash.Tree.depth V := by
  unfold Hash.Tree.depth
  split
  · rename_i h
    simpa using h
  · have h := @Nat.lt_log2_self (V - 1)
    generalize 2 ^ (Nat.log2 (V - 1) + 1) = p at h ⊢
    omega

/-- A table of at most `2 ^ 32` rows has a tree of depth at most 32. -/
theorem depth_le {V : ℕ} (hV : V ≤ 2 ^ 32) : Hash.Tree.depth V ≤ 32 := by
  unfold Hash.Tree.depth
  split
  · omega
  · rename_i h
    have hlt : V - 1 < 2 ^ 32 := Nat.lt_of_lt_of_le (Nat.sub_lt (by omega) Nat.one_pos) hV
    have := (Nat.log2_lt (by omega)).2 hlt
    omega

/-- **The read's gates refine its reference at a unit of the parsed circuit file.** At unit `u` of `partsOf`, let the
unit net's rows hold the read's program for a table of `V` rows of `H` words (`hr`), computing its reference wherever
they hold with their constant set (`hsound`). If the stack's constant gate is registered as 1 and is among `u`'s inputs,
and copy `j`'s read outputs are committed, then the read's gates at copy `j` refine `Hash.EmbeddingRead` at SHA-512 under
the hm96 key `key`, at the ports through copy `j`'s gates. -/
theorem gatesRefine_partsOf (key : Hash.Msg) {V H : ℕ} (D : Flock.Desc) (hU : FlockSoundness.Built D.unroll.unit)
    (n : ℕ) {run : (List Lin × Lin) × St}
    (hr : HoldsRead (FlockSoundness.Rows.ofNet hU) (readIn H (Hash.Tree.depth V)) run.2 (run.1.1 ++ [run.1.2]))
    (hsound : ∀ z : Fin (FlockSoundness.Rows.ofNet hU).w → Bool, (FlockSoundness.Rows.ofNet hU).Sat z →
      z (FlockSoundness.Rows.ofNet hU).one = true →
      evs (fun v => z (hr.φ v)) run.1.1 =
          Hash.EmbeddingRead.h V H (readTok fun v => z (hr.φ v))
            (readOpening H (Hash.Tree.depth V) fun v => z (hr.φ v)) ∧
        (ev (fun v => z (hr.φ v)) run.1.2 = true →
          Hash.EmbeddingRead.ok Verity.sha512 Verity.sha512 ⟨readDomain fun v => z (hr.φ v), V, fun r => r⟩ key
            (readRoot fun v => z (hr.φ v)) V (readTok fun v => z (hr.φ v))
            (readOpening H (Hash.Tree.depth V) fun v => z (hr.φ v))))
    (consts : Fin (Flock.Bridge.circuitOf D hU n).N → Option Bool) (u : Fin n) {j : ℕ} (hj : j < D.unroll.upv)
    (hone : consts (Flock.Bridge.constGate D hU n u) = some true)
    (hone_io : Flock.Bridge.constGate D hU n u ∈ (Flock.Bridge.partsOf D hU n).io u)
    (hcom : ∀ i < 16 * H + 1, Flock.Bridge.copyGate D hU n u hj (hr.out i) ∈ (Flock.Bridge.partsOf D hU n).committed) :
    ReadUnit.GatesRefine Verity.sha512 Verity.sha512 key consts (Flock.Bridge.partsOf D hU n) u
      (readPorts (Flock.Bridge.copyGate D hU n u hj) hr V H) := by
  refine gatesRefine_of_vals key (Flock.Bridge.partsOf D hU n) u (Flock.Bridge.copyGate D hU n u hj) hr hsound consts
    fun tr hcorrect hconsts => ?_
  obtain ⟨z, hsat, hz1, hz⟩ := Flock.Bridge.rows_of_correct D hU n hcorrect hj
  exact ⟨fun c => z c.val, hsat, hz1.trans (hconsts _ hone_io true hone), fun c hc => hz c (Or.inl hc),
    fun i hi => hz _ (Or.inr (hcom i hi))⟩

/-- **The parsed circuit file's reads at copy `j`**: the units `units` read tables, each at the ports through its copy
`j`'s gates, where the unit net's rows hold the read's program (`hr`); unit `u`'s table is under the root `root u` and
the domain id `domain u`, with the opening `table u i` of its row `i`. -/
noncomputable def readsOf {V H : ℕ} (D : Flock.Desc) (hU : FlockSoundness.Built D.unroll.unit) (n : ℕ)
    {run : (List Lin × Lin) × St}
    (hr : HoldsRead (FlockSoundness.Rows.ofNet hU) (readIn H (Hash.Tree.depth V)) run.2 (run.1.1 ++ [run.1.2]))
    {j : ℕ} (hj : j < D.unroll.upv) (units : Finset (Fin n)) (root domain : Fin n → Verity.Digest)
    (table : Fin n → ℕ → Hash.RowLeaf.Opening) : ReadUnit.Reads (Flock.Bridge.circuitOf D hU n) n where
  units := units
  port u := readPorts (Flock.Bridge.copyGate D hU n u hj) hr V H
  root := root
  domain := domain
  table := table

/-- **A drawn read unit of the parsed circuit file returns its table's row at its index**, but with probability at
most the layer below's `ε` plus the chance of finding a SHA-512 collision. At the parsed file's partition (`partsOf`),
let the unit net's rows hold the read's program and compute its reference wherever they hold (`hr`, `hsound`), and let
each unit's constant gate be registered as 1, among its inputs, and its copy `j`'s read outputs committed. Then over the
outcomes `ω`, a run that accepts while a drawn read unit's copy `j` returns, at an index inside its table, a row other
than its table's has probability at most `ε`, the layer below's bound on accepting while a drawn read unit is wrong or
its registered rows fail, plus `E[cost] / 2^256`, where `cost` is what the collision search behind the read's checks
costs (`Claims.ExpectedCR`). -/
theorem readUnitDrawn_partsOf {Ω : Type} [Fintype Ω] [Nonempty Ω] (key : Hash.Msg) {V H : ℕ} (D : Flock.Desc)
    (hU : FlockSoundness.Built D.unroll.unit) (n : ℕ) {run : (List Lin × Lin) × St}
    (hr : HoldsRead (FlockSoundness.Rows.ofNet hU) (readIn H (Hash.Tree.depth V)) run.2 (run.1.1 ++ [run.1.2]))
    (hsound : ∀ z : Fin (FlockSoundness.Rows.ofNet hU).w → Bool, (FlockSoundness.Rows.ofNet hU).Sat z →
      z (FlockSoundness.Rows.ofNet hU).one = true →
      evs (fun v => z (hr.φ v)) run.1.1 =
          Hash.EmbeddingRead.h V H (readTok fun v => z (hr.φ v))
            (readOpening H (Hash.Tree.depth V) fun v => z (hr.φ v)) ∧
        (ev (fun v => z (hr.φ v)) run.1.2 = true →
          Hash.EmbeddingRead.ok Verity.sha512 Verity.sha512 ⟨readDomain fun v => z (hr.φ v), V, fun r => r⟩ key
            (readRoot fun v => z (hr.φ v)) V (readTok fun v => z (hr.φ v))
            (readOpening H (Hash.Tree.depth V) fun v => z (hr.φ v))))
    {j : ℕ} (hj : j < D.unroll.upv) (consts : Fin (Flock.Bridge.circuitOf D hU n).N → Option Bool)
    (hone : ∀ u, consts (Flock.Bridge.constGate D hU n u) = some true)
    (hone_io : ∀ u, Flock.Bridge.constGate D hU n u ∈ (Flock.Bridge.partsOf D hU n).io u)
    (hcom : ∀ u, ∀ i < 16 * H + 1,
      Flock.Bridge.copyGate D hU n u hj (hr.out i) ∈ (Flock.Bridge.partsOf D hU n).committed)
    (units : Finset (Fin n)) (root domain : Fin n → Verity.Digest) (table : Fin n → ℕ → Hash.RowLeaf.Opening)
    (acc : Ω → Prop) (drawn : Ω → Finset (Fin n)) (X : Ω → (Flock.Bridge.circuitOf D hU n).Transcript)
    (cost : Ω → ℕ) (ε : ℝ)
    (hcommitted : (readsOf D hU n hr hj units root domain table).Committed Verity.sha512 Verity.sha512 key)
    (below : Claims.Pr (fun ω => acc ω ∧ ∃ u ∈ drawn ω, u ∈ units ∧
      ¬ ((Flock.Bridge.partsOf D hU n).Correct (X ω) u ∧
        ReadUnit.RegisteredRows consts (Flock.Bridge.partsOf D hU n) u
          (readPorts (Flock.Bridge.copyGate D hU n u hj) hr V H) (root u) (domain u) (X ω))) ≤ ε)
    (hcr : Claims.ExpectedCR (fun m => (Verity.sha512 m).bytes)
      ((readsOf D hU n hr hj units root domain table).finder Verity.sha512 key drawn X cost)) :
    Claims.Pr (fun ω => acc ω ∧ ∃ u ∈ drawn ω, u ∈ units ∧
      (readsOf D hU n hr hj units root domain table).index u (X ω) < V ∧
      (readPorts (Flock.Bridge.copyGate D hU n u hj) hr V H).hOf (X ω) ≠
        (table u ((readsOf D hU n hr hj units root domain table).index u (X ω))).row) ≤
      ε + Claims.Ex (fun ω => (cost ω : ℝ)) / 2 ^ (256 : ℝ) :=
  Accounting.SecurityProofs.readUnitDrawnCR Verity.sha512 key consts (Flock.Bridge.partsOf D hU n)
    (readsOf D hU n hr hj units root domain table)
    acc drawn X cost ε
    (fun u _ => gatesRefine_partsOf key D hU n hr hsound consts u hj (hone u) (hone_io u) (hcom u))
    hcommitted below hcr

end Accounting.ReadUnit.Gates
