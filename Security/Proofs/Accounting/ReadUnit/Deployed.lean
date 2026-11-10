module

public import Proofs.Accounting.ReadUnit
public import Proofs.Accounting.ReadUnit.Check
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
constant gate is registered as 1 and is among `u`'s inputs, and copy `j`'s verdict gate is committed, the read's gates
at copy `j` refine `Hash.EmbeddingRead` (`ReadUnit.GatesRefine`), at the ports through the copy's gates
(`Flock.Bridge.copyGate`). The values are the unit's local evaluation, where every copy holds the unit net's rows
(`Flock.Bridge.rows_localEval`), so the read's row need not be committed: a block that consumes it inside itself binds
it as a block that commits it does.

`readUnitDrawn_partsOf` is the read's binding at the parsed file: `Accounting.SecurityProofs.readUnitDrawnCR` at
`partsOf`, for reads each at one copy's gates in its unit, with its gate premise discharged by `gatesRefine_partsOf` at
every read. `readUnitDrawn_layout` discharges its constant and verdict premises from the parsed file's layout, and
`readUnitDrawn_check` the rest from the file's parse and the read's template check (`unrolled_leaves_of_parse`,
`read_of_check`): no premise about the read's gates remains.
-/

@[expose] public section

namespace Accounting.ReadUnit.Gates

open Flock.HmNets hiding and
open FlockSoundness.Discharge.Hm

/-- **The read's gates refine its reference at a unit of the parsed circuit file.** At unit `u` of `partsOf`, let the
unit net's rows hold the read's program for a table of `V` rows of `H` words (`hr`), computing its reference wherever
they hold with their constant set (`hsound`). If the stack's constant gate is registered as 1 and is among `u`'s inputs,
and copy `j`'s verdict gate is committed, then the read's gates at copy `j` refine `Hash.EmbeddingRead` at SHA-512
under the hm96 key `key`, at the ports through copy `j`'s gates. -/
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
    (hok : Flock.Bridge.copyGate D hU n u hj (hr.out (16 * H)) ∈ (Flock.Bridge.partsOf D hU n).committed) :
    ReadUnit.GatesRefine Verity.sha512 Verity.sha512 key consts (Flock.Bridge.partsOf D hU n) u
      (readPorts (Flock.Bridge.copyGate D hU n u hj) hr V H) := by
  refine gatesRefine_of_vals key (Flock.Bridge.partsOf D hU n) u (Flock.Bridge.copyGate D hU n u hj) hr hsound consts
    (fun tr _ hconsts => ?_) (fun c hc hu => absurd hu (Flock.Bridge.copyGate_input D hU n u hj c hc)) (fun _ => hok)
  obtain ⟨z, hsat, hz1, hz⟩ := Flock.Bridge.rows_localEval D hU n tr u hj
  exact ⟨fun c => z c.val, hsat, hz1.trans (hconsts _ hone_io true hone), fun c _ => hz c, fun i _ => hz _⟩

/-- **Each read of a drawn unit of the parsed circuit file returns its table's row at its index**, but with probability
at most the layer below's `ε` plus the chance of finding a SHA-512 collision. At the parsed file's partition
(`partsOf`), let the unit net's rows hold the read's program and compute its reference wherever they hold (`hr`,
`hsound`), and let each unit's constant gate be registered as 1, among its inputs, and each copy's verdict gate
committed. Take reads `R`, each at the ports through one copy's gates in its unit (`hR`), with tables committed under
their roots: a unit of `units_per_vu` stacked copies holds a read a copy. Then over the outcomes `ω`, a run that accepts
while a read of a drawn unit returns, in its unit's local evaluation and at an index inside its table, a row other than
its table's has probability at most `ε`, the layer below's bound on accepting while a drawn read's unit is wrong or the
read's registered rows fail, plus `E[cost] / 2^256`, where `cost` is what the collision search behind the reads' checks
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
    (consts : Fin (Flock.Bridge.circuitOf D hU n).N → Option Bool)
    (hone : ∀ u, consts (Flock.Bridge.constGate D hU n u) = some true)
    (hone_io : ∀ u, Flock.Bridge.constGate D hU n u ∈ (Flock.Bridge.partsOf D hU n).io u)
    (hok : ∀ u j (hj : j < D.unroll.upv),
      Flock.Bridge.copyGate D hU n u hj (hr.out (16 * H)) ∈ (Flock.Bridge.partsOf D hU n).committed)
    {ι : Type} (R : ReadUnit.Reads (Flock.Bridge.circuitOf D hU n) n ι)
    (hR : ∀ i ∈ R.reads, ∃ j, ∃ hj : j < D.unroll.upv,
      R.port i = readPorts (Flock.Bridge.copyGate D hU n (R.unit i) hj) hr V H)
    (acc : Ω → Prop) (drawn : Ω → Finset (Fin n)) (tr : Ω → (Flock.Bridge.circuitOf D hU n).Trace)
    (cost : Ω → ℕ) (ε : ℝ) (hcommitted : R.Committed Verity.sha512 Verity.sha512 key)
    (below : Claims.Pr (fun ω => acc ω ∧ ∃ i ∈ R.reads, R.unit i ∈ drawn ω ∧
      ¬ ((Flock.Bridge.partsOf D hU n).Correct (tr ω) (R.unit i) ∧
        ReadUnit.RegisteredRows consts (Flock.Bridge.partsOf D hU n) (R.unit i) (R.port i) (R.root i) (R.domain i)
          (tr ω))) ≤ ε)
    (hcr : Claims.ExpectedCR (fun m => (Verity.sha512 m).bytes)
      (R.finder (Flock.Bridge.partsOf D hU n) Verity.sha512 key drawn tr cost)) :
    Claims.Pr (fun ω => acc ω ∧ ∃ i ∈ R.reads, R.unit i ∈ drawn ω ∧
      R.index i ((Flock.Bridge.partsOf D hU n).localEval (tr ω) (R.unit i)) < (R.port i).count ∧
      (R.port i).hOf ((Flock.Bridge.partsOf D hU n).localEval (tr ω) (R.unit i)) ≠
        (R.table i (R.index i ((Flock.Bridge.partsOf D hU n).localEval (tr ω) (R.unit i)))).row) ≤
      ε + Claims.Ex (fun ω => (cost ω : ℝ)) / 2 ^ (256 : ℝ) := by
  refine Accounting.SecurityProofs.readUnitDrawnCR Verity.sha512 key consts (Flock.Bridge.partsOf D hU n) R acc drawn
    tr cost ε (fun i hi => ?_) hcommitted below hcr
  obtain ⟨j, hj, hp⟩ := hR i hi
  rw [hp]
  exact gatesRefine_partsOf key D hU n hr hsound consts (R.unit i) hj (hone _) (hone_io _) (hok _ j hj)

/-- **`readUnitDrawn_partsOf` at the parsed file's layout.** Its premises that each unit's constant gate is among its
inputs and each copy's verdict gate is committed hold where the stack has a copy (`hupv`), the read has more than
`16 H` outputs (`hlen`), and its verdict, output `16 H`, sits on bit 0 of every copy's returned leaf `H` (`hout`,
`hleaves`): the unit's constant gate is opened at the unit (`Flock.Bridge.constGate_mem_io`), and a copy's gate on a
returned leaf is committed (`Flock.Bridge.copyGate_mem_committed`). -/
theorem readUnitDrawn_layout {Ω : Type} [Fintype Ω] [Nonempty Ω] (key : Hash.Msg) {V H : ℕ} (D : Flock.Desc)
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
    (hupv : 0 < D.unroll.upv) (hlen : 16 * H < (run.1.1 ++ [run.1.2]).length)
    (hout : (hr.out (16 * H)).val = FlockSoundness.Discharge.Layout.outCol D.unroll H 0)
    (hleaves : ∀ j < D.unroll.upv, H < (D.unroll.leavesOut.getD j #[]).size)
    (consts : Fin (Flock.Bridge.circuitOf D hU n).N → Option Bool)
    (hone : ∀ u, consts (Flock.Bridge.constGate D hU n u) = some true)
    {ι : Type} (R : ReadUnit.Reads (Flock.Bridge.circuitOf D hU n) n ι)
    (hR : ∀ i ∈ R.reads, ∃ j, ∃ hj : j < D.unroll.upv,
      R.port i = readPorts (Flock.Bridge.copyGate D hU n (R.unit i) hj) hr V H)
    (acc : Ω → Prop) (drawn : Ω → Finset (Fin n)) (tr : Ω → (Flock.Bridge.circuitOf D hU n).Trace)
    (cost : Ω → ℕ) (ε : ℝ) (hcommitted : R.Committed Verity.sha512 Verity.sha512 key)
    (below : Claims.Pr (fun ω => acc ω ∧ ∃ i ∈ R.reads, R.unit i ∈ drawn ω ∧
      ¬ ((Flock.Bridge.partsOf D hU n).Correct (tr ω) (R.unit i) ∧
        ReadUnit.RegisteredRows consts (Flock.Bridge.partsOf D hU n) (R.unit i) (R.port i) (R.root i) (R.domain i)
          (tr ω))) ≤ ε)
    (hcr : Claims.ExpectedCR (fun m => (Verity.sha512 m).bytes)
      (R.finder (Flock.Bridge.partsOf D hU n) Verity.sha512 key drawn tr cost)) :
    Claims.Pr (fun ω => acc ω ∧ ∃ i ∈ R.reads, R.unit i ∈ drawn ω ∧
      R.index i ((Flock.Bridge.partsOf D hU n).localEval (tr ω) (R.unit i)) < (R.port i).count ∧
      (R.port i).hOf ((Flock.Bridge.partsOf D hU n).localEval (tr ω) (R.unit i)) ≠
        (R.table i (R.index i ((Flock.Bridge.partsOf D hU n).localEval (tr ω) (R.unit i)))).row) ≤
      ε + Claims.Ex (fun ω => (cost ω : ℝ)) / 2 ^ (256 : ℝ) := by
  obtain ⟨i, hi⟩ := hr.output (16 * H) hlen
  have hcol : (hr.out (16 * H)).val = (FlockSoundness.Rows.ofNet hU).nIn + i.val := by
    rw [hi]
    rfl
  have hlt := i.isLt
  exact readUnitDrawn_partsOf key D hU n hr hsound consts hone
    (fun u => Flock.Bridge.constGate_mem_io D hU n u hupv)
    (fun u j hj => Flock.Bridge.copyGate_mem_committed D hU n u hj (hr.out (16 * H)) (hleaves j hj) (by omega) hout
      (by omega) (by omega))
    R hR acc drawn tr cost ε hcommitted below hcr

/-- **A drawn read unit of a parsed circuit file returns its table's row, at a template the read's check accepts.**
Let the circuit file parse into `ct` (`hparse`, `Flock.HmOut.parseTmpl`), with its unit template passing the read's
check (`hcheck`, `Flock.ReadNet.check V H`) and its unit returning the circuit's outputs (`ho`). The parse gives a copy
in the stack and one leaf per output port in each copy (`unrolled_leaves_of_parse`), which the check makes the read's
`H + 1` outputs (`unrolled_outPorts`). Then the unrolled unit's rows run the read's program, its input `x` on column
`x` and its outputs on the unit's returned output group, and for any reads each at one copy's ports of that run,
committed under the pinned hm96 key: over the outcomes `ω`, a run that accepts while a read of a drawn unit returns, in
its unit's local evaluation and at an index inside its table, a row other than its table's has probability at most the
layer below's `ε` plus `E[cost] / 2^256` (`readUnitDrawn_layout`, `read_of_check`). -/
theorem readUnitDrawn_check {Ω : Type} [Fintype Ω] [Nonempty Ω] {V H : ℕ} {tags : Flock.Tags} {bytes : ByteArray}
    {tables : List (String × ByteArray)} {ct : Flock.Circuit} {tu : Flock.TmplUnit}
    (hparse : Flock.HmOut.parseTmpl tags bytes tables = .ok ct) (ht : ct.tmpl = some tu)
    (hcheck : Flock.ReadNet.check V H tu.t = true) (ho : ct.outNet = ct.unitNet)
    (hB : FlockSoundness.Built (Flock.HmOut.unrolled ct).unit) (n : ℕ) :
    ∃ (run : (List Lin × Lin) × St)
      (hr : HoldsRead (FlockSoundness.Rows.ofNet hB) (readIn H (Hash.Tree.depth V)) run.2 (run.1.1 ++ [run.1.2])),
      (∀ x < readIn H (Hash.Tree.depth V), (hr.φ x).val = x) ∧
      (∀ j < 16 * H + 1,
        (hr.out j).val = FlockSoundness.Discharge.Layout.outCol (Flock.HmOut.unrolled ct) (j / 16) (j % 16)) ∧
      ∀ (consts : Fin (Flock.Bridge.circuitOf (.tmpl ct) hB n).N → Option Bool),
        (∀ u, consts (Flock.Bridge.constGate (.tmpl ct) hB n u) = some true) →
        ∀ {ι : Type} (R : ReadUnit.Reads (Flock.Bridge.circuitOf (.tmpl ct) hB n) n ι),
        (∀ i ∈ R.reads, ∃ j, ∃ hj : j < (Flock.HmOut.unrolled ct).upv,
          R.port i = readPorts (Flock.Bridge.copyGate (.tmpl ct) hB n (R.unit i) hj) hr V H) →
        ∀ (acc : Ω → Prop) (drawn : Ω → Finset (Fin n))
          (tr : Ω → (Flock.Bridge.circuitOf (.tmpl ct) hB n).Trace) (cost : Ω → ℕ) (ε : ℝ),
        R.Committed Verity.sha512 Verity.sha512 pinnedKey →
        Claims.Pr (fun ω => acc ω ∧ ∃ i ∈ R.reads, R.unit i ∈ drawn ω ∧
          ¬ ((Flock.Bridge.partsOf (.tmpl ct) hB n).Correct (tr ω) (R.unit i) ∧
            ReadUnit.RegisteredRows consts (Flock.Bridge.partsOf (.tmpl ct) hB n) (R.unit i) (R.port i) (R.root i)
              (R.domain i) (tr ω))) ≤ ε →
        Claims.ExpectedCR (fun m => (Verity.sha512 m).bytes)
          (R.finder (Flock.Bridge.partsOf (.tmpl ct) hB n) Verity.sha512 pinnedKey drawn tr cost) →
        Claims.Pr (fun ω => acc ω ∧ ∃ i ∈ R.reads, R.unit i ∈ drawn ω ∧
          R.index i ((Flock.Bridge.partsOf (.tmpl ct) hB n).localEval (tr ω) (R.unit i)) < (R.port i).count ∧
          (R.port i).hOf ((Flock.Bridge.partsOf (.tmpl ct) hB n).localEval (tr ω) (R.unit i)) ≠
            (R.table i (R.index i ((Flock.Bridge.partsOf (.tmpl ct) hB n).localEval (tr ω) (R.unit i)))).row) ≤
          ε + Claims.Ex (fun ω => (cost ω : ℝ)) / 2 ^ (256 : ℝ) := by
  have hsz : ct.unitNet < ct.nets.size := by
    obtain ⟨_, _, P, -, rfl, -, -⟩ := FlockSoundness.TmplValue.parseTmpl_facts hparse
    exact P.unitNet
  obtain ⟨hupv, hleaves⟩ := unrolled_leaves_of_parse hparse ho
  obtain ⟨run, hr, hsound, hrow, hφ, hout⟩ := read_of_check ht hsz hcheck hB
  have hlen : 16 * H < (run.1.1 ++ [run.1.2]).length := by
    simp only [List.length_append, List.length_singleton, hrow]
    omega
  have hH : ∀ j < (Flock.HmOut.unrolled ct).upv, H < ((Flock.HmOut.unrolled ct).leavesOut.getD j #[]).size :=
    fun j hj => by
      rw [hleaves j hj, unrolled_outPorts ht hsz hcheck]
      omega
  have hv : (hr.out (16 * H)).val = FlockSoundness.Discharge.Layout.outCol (Flock.HmOut.unrolled ct) H 0 := by
    have h := hout (16 * H) (by omega)
    rwa [Nat.mul_div_cancel_left H (by norm_num : 0 < 16), Nat.mul_mod_right] at h
  exact ⟨run, hr, hφ, hout, fun consts hone {ι} R hR acc drawn tr cost ε hcommitted below hcr =>
    readUnitDrawn_layout pinnedKey (.tmpl ct) hB n hr hsound hupv hlen hv hH consts hone R hR acc drawn tr cost
      ε hcommitted below hcr⟩

end Accounting.ReadUnit.Gates
