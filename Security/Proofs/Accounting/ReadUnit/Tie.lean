module

public import Proofs.Accounting.ReadUnit.Net
public import Proofs.Core.Circuit
public import Proofs.Flock.Bridge.CircuitOf
public import Proofs.Flock.Soundness.Audit.Circuit
public import Proofs.Flock.Soundness.Lowering
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
# The read unit's gates refine its reference, where its rows hold the read's program

`gatesRefine_of_holdsRead`: let unit `u` of a game partition be an instance of the rows `R` (`IsRowsUnit`), and let `R`
hold the read's program (`HoldsRead`: the program's input bits at `R`'s input columns and its outputs at computed ones,
and wherever `R`'s rows hold with their constant set, the program's rows hold and its outputs are those columns'). Then,
where the four parts' statements hold, the read's gates refine `Hash.EmbeddingRead` at SHA-512
(`ReadUnit.GatesRefine`) in core's view of the partition, at the ports through the unit's wires (`readPorts`), wherever
the constant's gate is registered as 1.

A unit's rows hold at its local evaluation whatever the transcript (`sat_localEval`). A correct unit's committed gates
hold that evaluation, and its input columns are gates outside it, so the evaluation is the transcript on the read's
ports. The core, `gatesRefine_of_vals`, needs only that: values satisfying `R` that are the transcript's bits at the
read's ports wherever the unit is correct, so it serves as well a unit whose rows hold the read as one copy of a stack.
What remains of the read's gate refinement is `HoldsRead` for the registered template, which a per-Definition check of
the template against the parts' generators gives.
-/

@[expose] public section

namespace Accounting.ReadUnit.Gates

open Flock.HmNets hiding and
open FlockSoundness.Discharge.Hm

/-! ## A unit's rows hold at its local evaluation -/

/-- **A unit's rows hold at its local evaluation**, whatever the transcript: each computed row's gate is the row of its
columns' gates, which the local evaluation applies. -/
theorem sat_localEval {C : FlockSoundness.Audit.Circuit} {n : ℕ} {P : FlockSoundness.Audit.Partition C n} {u : Fin n}
    {R : FlockSoundness.Rows} (I : P.IsRowsUnit u R) (tr : Fin C.N → Bool) :
    R.Sat fun c => P.localEval tr u (I.wire c) := by
  intro i
  show P.localEval tr u (I.wire (R.comp i)) = _
  rw [P.localEval_of_mem tr (I.mem i), I.op i]
  simp only [FlockSoundness.Audit.Op.apply, FlockSoundness.Audit.xorSum_map, Function.comp_def]

theorem ofBits_bit {C : FlockSoundness.Audit.Circuit} (tr : (Verity.Circuit.ofAudit C).Transcript) :
    Verity.Circuit.ofBits (fun g => ReadUnit.bit tr g) = tr := by
  funext g
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  have hi0 : i = 0 := by
    have : i < 1 := hi
    omega
  subst hi0
  simp [Verity.Circuit.ofBits, ReadUnit.bit]

/-- A unit correct in core's view of a game partition is correct in the game's, at its bits. -/
theorem correct_bits {C : FlockSoundness.Audit.Circuit} {n : ℕ} (P : FlockSoundness.Audit.Partition C n) {u : Fin n}
    {tr : (Verity.Circuit.ofAudit C).Transcript} (h : (Verity.Partition.ofAudit P).Correct tr u) :
    P.Correct (fun g => ReadUnit.bit tr g) u := by
  by_contra hc
  have hw := Verity.Partition.wrong_ofAudit P (fun g => ReadUnit.bit tr g)
  rw [ofBits_bit] at hw
  have hm : u ∈ P.wrong fun g => ReadUnit.bit tr g := (P.mem_wrong).2 hc
  rw [← hw] at hm
  exact ((Verity.Partition.ofAudit P).mem_wrong.1 hm) h

/-! ## The read's program -/

/-- The read's input bits, in `EmbeddingRead_v1`'s port order: the token's 32, the root's and the domain id's 512 each,
the row's `16 H`, the salt's 1536 and the path's `512 D`. -/
def readIn (H D : ℕ) : ℕ := 1056 + 16 * H + 1536 + 512 * D

/-- The `n` input bits from bit `k`, as forms. -/
def ins (k n : ℕ) : List Lin := (List.range n).map fun i => Lin.var (k + i)

/-- `m` input words from bit `k`. -/
def insW (k m : ℕ) : List Word := (List.range m).map fun j => ins (k + 64 * j) 64

/-- **The read's program**: its input bits in port order, then the read's net over them. -/
def readProg (D H : ℕ) (rowSeg : List Lin → M (List Word)) (leaf : List Word → List Word → M (List Word))
    (frame : List Word → List Word → List Word → List Lin → List (List Word) → M Lin)
    (select : List Lin → Lin → List Lin → M (List Lin × Lin)) : M (List Lin × Lin) := do
  let _ ← input (readIn H D)
  readNet D rowSeg leaf frame select (ins 0 32) (insW 32 8) (insW 544 8) (ins 1056 (16 * H))
    (insW (1056 + 16 * H) 24) ((List.range D).map fun j => insW (2592 + 16 * H + 512 * j) 8)

/-- **A unit's rows hold a program** of `N` input bits, built to `st` with outputs `outs`: under the column map `φ`, the
program's input bits sit at the rows' input columns and its outputs at computed ones (`out`), and wherever the rows hold
with their constant set, every row of the program holds and output `j`'s column holds output form `j`. -/
structure HoldsRead (R : FlockSoundness.Rows) (N : ℕ) (st : St) (outs : List Lin) where
  φ : Var → Fin R.w
  out : ℕ → Fin R.w
  input : ∀ v < N, (φ v).val < R.nIn
  output : ∀ j < outs.length, ∃ i, out j = R.comp i
  rows : ∀ z : Fin R.w → Bool, R.Sat z → z R.one = true → stHolds (fun v => z (φ v)) st
  outs_eq : ∀ z : Fin R.w → Bool, R.Sat z → z R.one = true → ∀ j (hj : j < outs.length),
    z (out j) = ev (fun v => z (φ v)) outs[j]

/-- **The read's ports** at the gates `wire` puts the rows' columns on, for a table of `V` rows of `H` words: its input
bits' columns in port order, then its outputs' (the row, then the verdict). -/
def readPorts {C : Verity.Circuit} {R : FlockSoundness.Rows} (wire : Fin R.w → Fin C.N) {N : ℕ} {st : St}
    {outs : List Lin} (hr : HoldsRead R N st outs) (V H : ℕ) : ReadUnit.Ports C where
  count := V
  words := H
  position r := r
  tok i := wire (hr.φ i)
  root i := wire (hr.φ (32 + i))
  domain i := wire (hr.φ (544 + i))
  row i := wire (hr.φ (1056 + i))
  salt i := wire (hr.φ (1056 + 16 * H + i))
  path j i := wire (hr.φ (2592 + 16 * H + 512 * j + i))
  h i := wire (hr.out i)
  ok := wire (hr.out (16 * H))

/-! ## Lists -/

theorem ofFn_eq_map {α : Type} {n : ℕ} (f : Fin n → α) (g : ℕ → α) (h : ∀ i : Fin n, f i = g i) :
    List.ofFn f = (List.range n).map g :=
  List.ext_getElem (by simp) fun i h₁ h₂ => by simp [h ⟨i, by simpa using h₁⟩]

theorem length_ins (k n : ℕ) : (ins k n).length = n := by simp [ins]

theorem evs_ins (z : Var → Bool) (k n : ℕ) : evs z (ins k n) = (List.range n).map fun i => z (k + i) := by
  simp [evs, ins]

theorem shaped_insW (k m : ℕ) : Shaped m (insW k m) :=
  ⟨by simp [insW], fun w hw => by
    obtain ⟨j, -, rfl⟩ := List.mem_map.1 hw
    exact length_ins _ _⟩

theorem ins_append (k a b : ℕ) : ins k a ++ ins (k + a) b = ins k (a + b) := by
  refine List.ext_getElem (by simp [ins]) fun i h₁ h₂ => ?_
  rw [List.getElem_append]
  simp only [ins, List.getElem_map, List.getElem_range, List.length_map, List.length_range]
  split
  · rfl
  · rw [show k + a + (i - a) = k + i by omega]

theorem flatten_insW (k : ℕ) : ∀ m, (insW k m).flatten = ins k (64 * m)
  | 0 => by simp [insW, ins]
  | m + 1 => by
    rw [show insW k (m + 1) = insW k m ++ [ins (k + 64 * m) 64] by simp [insW, List.range_succ]]
    rw [List.flatten_append, flatten_insW k m, List.flatten_singleton, ins_append,
      show 64 * m + 64 = 64 * (m + 1) by ring]

theorem length_fill (H : ℕ) : (Hash.EmbeddingRead.fill H).length = 16 * H := by
  have h16 : Hash.EmbeddingRead.FILL_WORD.length = 16 := by simp [Hash.EmbeddingRead.FILL_WORD]
  simp only [Hash.EmbeddingRead.fill, List.length_flatten, List.map_replicate, h16, List.sum_replicate, smul_eq_mul]
  ring

/-! ## The read at its rows' values, whatever the unit's boundary -/

/-- The read's token at the program's variable values `z`. -/
def readTok (z : Var → Bool) : ℕ := num (evs z (ins 0 32))

/-- The read's root at the program's variable values `z`. -/
def readRoot (z : Var → Bool) : Verity.Digest := wordsDigest z (insW 32 8)

/-- The read's domain id at the program's variable values `z`. -/
def readDomain (z : Var → Bool) : Verity.Digest := wordsDigest z (insW 544 8)

/-- The read's opening at the program's variable values `z`: its row, salt and `D` sibling digests. -/
def readOpening (H D : ℕ) (z : Var → Bool) : Hash.RowLeaf.Opening :=
  openingOf z (ins 1056 (16 * H)) (insW (1056 + 16 * H) 24) ((List.range D).map fun j => insW (2592 + 16 * H + 512 * j) 8)

/-- **Where rows hold the read's program, they compute the read's reference**, whatever the unit's boundary: if the
program's rows hold at the rows' values under the column map `φ` wherever the rows hold with their constant set
(`HoldsRead.rows`), then at any such values `z`, the program's row output is
`Hash.EmbeddingRead.h` of its token and opening, and where its verdict is set, `Hash.EmbeddingRead.ok` accepts them at
SHA-512. The read's own unit reads it through its ports (`gatesRefine_of_holdsRead`); a block the read is fused into
reads it at its interior columns. -/
theorem holdsRead_sound (key : Hash.Msg) {V H D : ℕ} (hD : D = Hash.Tree.depth V) (hVD : V ≤ 2 ^ D) (hD32 : D ≤ 32)
    (hH : 16 * H < 2 ^ 64) {rowSeg : List Lin → M (List Word)} {leaf : List Word → List Word → M (List Word)}
    {frame : List Word → List Word → List Word → List Lin → List (List Word) → M Lin}
    {select : List Lin → Lin → List Lin → M (List Lin × Lin)} (hA : RowSegSpec (16 * H) rowSeg)
    (hB : Hm96LeafSpec key leaf) (hC : FrameReadSpec D Hash.RowLeaf.Format.seg.schema frame)
    (hS : SelectSpec V select) {R : FlockSoundness.Rows} {φ : Var → Fin R.w}
    (hrows : ∀ z : Fin R.w → Bool, R.Sat z → z R.one = true →
      stHolds (fun v => z (φ v)) ((readProg D H rowSeg leaf frame select).run St.empty).2)
    (z : Fin R.w → Bool) (hsat : R.Sat z) (h1 : z R.one = true) :
    evs (fun v => z (φ v)) ((readProg D H rowSeg leaf frame select).run St.empty).1.1 =
        Hash.EmbeddingRead.h V H (readTok fun v => z (φ v)) (readOpening H D fun v => z (φ v)) ∧
      (ev (fun v => z (φ v)) ((readProg D H rowSeg leaf frame select).run St.empty).1.2 = true →
        Hash.EmbeddingRead.ok Verity.sha512 Verity.sha512 ⟨readDomain fun v => z (φ v), V, fun r => r⟩ key
          (readRoot fun v => z (φ v)) V (readTok fun v => z (φ v)) (readOpening H D fun v => z (φ v))) := by
  have hrun : (readProg D H rowSeg leaf frame select).run St.empty =
      (readNet D rowSeg leaf frame select (ins 0 32) (insW 32 8) (insW 544 8) (ins 1056 (16 * H))
        (insW (1056 + 16 * H) 24) ((List.range D).map fun j => insW (2592 + 16 * H + 512 * j) 8)).run
        ((input (readIn H D)).run St.empty).2 :=
    rfl
  have hst := hrows z hsat h1
  rw [hrun] at hst ⊢
  exact (readNet_sound key hD hVD hD32 hH hA hB hC hS (ins 0 32) (insW 32 8) (insW 544 8) (ins 1056 (16 * H))
    (insW (1056 + 16 * H) 24) ((List.range D).map fun j => insW (2592 + 16 * H + 512 * j) 8)
    ((input (readIn H D)).run St.empty).2).2 _ hst (length_ins 0 32) (shaped_insW 32 8) (shaped_insW 544 8)
    (length_ins 1056 (16 * H)) (shaped_insW _ 24) (by simp)
    (fun s hs => by obtain ⟨j, -, rfl⟩ := List.mem_map.1 hs; exact shaped_insW _ 8)

/-! ## The tie -/

/-- The row the read returns has the table's row length, whichever branch of the reference returns it. -/
theorem length_h (V H D : ℕ) (z : Var → Bool) :
    (Hash.EmbeddingRead.h V H (readTok z) (readOpening H D z)).length = 16 * H := by
  unfold Hash.EmbeddingRead.h
  split
  · simp [readOpening, openingOf, length_evs, length_ins]
  · exact length_fill H

/-- **The read's gate refinement from its rows' values.** Let the rows `R` hold the read's program for a run (`hr`)
that computes the read's reference at the rows' values (`hsound`, which `holdsRead_sound` gives), and let `wire` put
`R`'s columns on gates of `C`. If wherever unit `u` of `P` is correct and matches the registered constants, some values
satisfy `R`, set its constant to 1, and equal the transcript's bits at `R`'s input columns and at the read's outputs
(`hvals`), then the read's gates refine `Hash.EmbeddingRead` at SHA-512 under the hm96 key `key`, at the read's ports
through `wire`. -/
theorem gatesRefine_of_vals (key : Hash.Msg) {V H : ℕ} {C : Verity.Circuit} {n : ℕ} (P : Verity.Partition C n)
    (u : Fin n) {R : FlockSoundness.Rows} (wire : Fin R.w → Fin C.N) {run : (List Lin × Lin) × St}
    (hr : HoldsRead R (readIn H (Hash.Tree.depth V)) run.2 (run.1.1 ++ [run.1.2]))
    (hsound : ∀ z : Fin R.w → Bool, R.Sat z → z R.one = true →
      evs (fun v => z (hr.φ v)) run.1.1 =
          Hash.EmbeddingRead.h V H (readTok fun v => z (hr.φ v))
            (readOpening H (Hash.Tree.depth V) fun v => z (hr.φ v)) ∧
        (ev (fun v => z (hr.φ v)) run.1.2 = true →
          Hash.EmbeddingRead.ok Verity.sha512 Verity.sha512 ⟨readDomain fun v => z (hr.φ v), V, fun r => r⟩ key
            (readRoot fun v => z (hr.φ v)) V (readTok fun v => z (hr.φ v))
            (readOpening H (Hash.Tree.depth V) fun v => z (hr.φ v))))
    (consts : Fin C.N → Option Bool)
    (hvals : ∀ tr : C.Transcript, P.Correct tr u →
      (∀ g ∈ P.io u, ∀ b, consts g = some b → ReadUnit.bit tr g = b) →
      ∃ z : Fin R.w → Bool, R.Sat z ∧ z R.one = true ∧
        (∀ c : Fin R.w, c.val < R.nIn → z c = ReadUnit.bit tr (wire c)) ∧
        ∀ j < 16 * H + 1, z (hr.out j) = ReadUnit.bit tr (wire (hr.out j))) :
    ReadUnit.GatesRefine Verity.sha512 Verity.sha512 key consts P u (readPorts wire hr V H) := by
  intro tr hcorrect hconsts
  obtain ⟨z, hsat, hz1, hzin, hzout⟩ := hvals tr hcorrect hconsts
  have hin : ∀ v < readIn H (Hash.Tree.depth V), z (hr.φ v) = ReadUnit.bit tr (wire (hr.φ v)) := fun v hv =>
    hzin _ (hr.input v hv)
  obtain ⟨hh, hok⟩ := hsound z hsat hz1
  set z' : Var → Bool := fun v => z (hr.φ v)
  have hlen1 : run.1.1.length = 16 * H := by
    have := congrArg List.length hh
    rwa [length_evs, length_h] at this
  have hob : ∀ j (hj : j < 16 * H + 1),
      ReadUnit.bit tr (wire (hr.out j)) = ev z' ((run.1.1 ++ [run.1.2])[j]'(by simp [hlen1]; omega)) := by
    intro j hj
    exact (hzout j hj).symm.trans (hr.outs_eq z hsat hz1 j (by simp [hlen1]; omega))
  have hN : readIn H (Hash.Tree.depth V) = 1056 + 16 * H + 1536 + 512 * Hash.Tree.depth V := rfl
  have hbits : ∀ (k m : ℕ), k + m ≤ readIn H (Hash.Tree.depth V) → ∀ f : Fin m → Fin C.N,
      (∀ i : Fin m, f i = wire (hr.φ (k + i))) → ReadUnit.bits tr f = evs z' (ins k m) := by
    intro k m hk f hf
    rw [evs_ins]
    refine ofFn_eq_map _ _ fun i => ?_
    show ReadUnit.bit tr (f i) = z' (k + i)
    rw [hf i]
    exact (hin _ (by have := i.isLt; omega)).symm
  have hwords : ∀ (k m : ℕ), k + 64 * m ≤ readIn H (Hash.Tree.depth V) → ∀ f : Fin (64 * m) → Fin C.N,
      (∀ i : Fin (64 * m), f i = wire (hr.φ (k + i))) → ReadUnit.bits tr f = evs z' (insW k m).flatten := by
    intro k m hk f hf
    rw [flatten_insW]
    exact hbits k (64 * m) hk f hf
  have htok : (readPorts wire hr V H).tokOf tr = readTok z' :=
    congrArg num (hbits 0 32 (by omega) (readPorts wire hr V H).tok fun i => by simp [readPorts])
  have hroot : (readPorts wire hr V H).rootOf tr = readRoot z' :=
    congrArg ReadUnit.digestOf (hwords 32 8 (by omega) (readPorts wire hr V H).root fun i => rfl)
  have hdomain : (readPorts wire hr V H).domainOf tr = readDomain z' :=
    congrArg ReadUnit.digestOf (hwords 544 8 (by omega) (readPorts wire hr V H).domain fun i => rfl)
  have hopen : (readPorts wire hr V H).opening tr = readOpening H (Hash.Tree.depth V) z' := by
    have e1 : ReadUnit.bits tr (readPorts wire hr V H).row = evs z' (ins 1056 (16 * H)) :=
      hbits 1056 (16 * H) (by omega) (readPorts wire hr V H).row fun i => rfl
    have e2 : ReadUnit.bits tr (readPorts wire hr V H).salt = evs z' (insW (1056 + 16 * H) 24).flatten :=
      hwords (1056 + 16 * H) 24 (by omega) (readPorts wire hr V H).salt fun i => rfl
    have e3 : (List.ofFn fun j => ReadUnit.digestOf (ReadUnit.bits tr ((readPorts wire hr V H).path j))) =
        ((List.range (Hash.Tree.depth V)).map fun j => insW (2592 + 16 * H + 512 * j) 8).map (wordsDigest z') := by
      rw [List.map_map]
      refine ofFn_eq_map _ _ fun j => ?_
      exact congrArg ReadUnit.digestOf (hwords (2592 + 16 * H + 512 * j) 8 (by have hj : (j : ℕ) < Hash.Tree.depth V := j.isLt; omega)
        ((readPorts wire hr V H).path j) fun i => rfl)
    show (⟨ReadUnit.bits tr (readPorts wire hr V H).row, ReadUnit.bytesOf (ReadUnit.bits tr (readPorts wire hr V H).salt),
      List.ofFn fun j => ReadUnit.digestOf (ReadUnit.bits tr ((readPorts wire hr V H).path j))⟩ : Hash.RowLeaf.Opening) =
      ⟨evs z' (ins 1056 (16 * H)), wordsBytes z' (insW (1056 + 16 * H) 24),
        ((List.range (Hash.Tree.depth V)).map fun j => insW (2592 + 16 * H + 512 * j) 8).map (wordsDigest z')⟩
    rw [e1, e2, e3]
    rfl
  refine ⟨?_, fun hokX => ?_⟩
  · show ReadUnit.bits tr (readPorts wire hr V H).h =
      Hash.EmbeddingRead.h V H ((readPorts wire hr V H).tokOf tr) ((readPorts wire hr V H).opening tr)
    rw [htok, hopen, ← hh]
    refine List.ext_getElem (by simp [ReadUnit.bits, length_evs, hlen1, readPorts]) fun i h₁ h₂ => ?_
    have hi : i < 16 * H := by simpa [ReadUnit.bits, readPorts] using h₁
    simp only [ReadUnit.bits, List.getElem_ofFn]
    change ReadUnit.bit tr (wire (hr.out i)) = _
    rw [hob i (by omega)]
    simp only [evs, List.getElem_map]
    have hil : i < run.1.1.length := by omega
    rw [List.getElem_append_left hil]
  · have e := hob (16 * H) (by omega)
    have hlast : (run.1.1 ++ [run.1.2])[16 * H]'(by simp [hlen1]) = run.1.2 := by
      have hle : run.1.1.length ≤ 16 * H := by omega
      rw [List.getElem_append_right hle]
      simp [hlen1]
    rw [hlast] at e
    have hok2 : ev z' run.1.2 = true := e.symm.trans hokX
    show Hash.EmbeddingRead.ok Verity.sha512 Verity.sha512 ⟨(readPorts wire hr V H).domainOf tr, V, fun r => r⟩ key
      ((readPorts wire hr V H).rootOf tr) V ((readPorts wire hr V H).tokOf tr) ((readPorts wire hr V H).opening tr)
    rw [hdomain, hroot, htok, hopen]
    exact hok hok2


/-- **The read's own unit gives the read's gate refinement at its ports**, for any run of the read's program whose
program the rows `R` hold (`hr`) and which computes the read's reference at the rows' values (`hsound`): where the
constant's gate is registered as 1 and among `u`'s inputs, and the read's output gates are committed. A correct unit's
committed gates hold its local evaluation, where its rows hold, and its input columns are gates outside it, so the
evaluation gives `gatesRefine_of_vals` its values. -/
theorem gatesRefine_of_run (key : Hash.Msg) {V H : ℕ} {C : FlockSoundness.Audit.Circuit} {n : ℕ}
    (P : FlockSoundness.Audit.Partition C n) (u : Fin n) {R : FlockSoundness.Rows} (I : P.IsRowsUnit u R)
    {run : (List Lin × Lin) × St} (hr : HoldsRead R (readIn H (Hash.Tree.depth V)) run.2 (run.1.1 ++ [run.1.2]))
    (hsound : ∀ z : Fin R.w → Bool, R.Sat z → z R.one = true →
      evs (fun v => z (hr.φ v)) run.1.1 =
          Hash.EmbeddingRead.h V H (readTok fun v => z (hr.φ v))
            (readOpening H (Hash.Tree.depth V) fun v => z (hr.φ v)) ∧
        (ev (fun v => z (hr.φ v)) run.1.2 = true →
          Hash.EmbeddingRead.ok Verity.sha512 Verity.sha512 ⟨readDomain fun v => z (hr.φ v), V, fun r => r⟩ key
            (readRoot fun v => z (hr.φ v)) V (readTok fun v => z (hr.φ v))
            (readOpening H (Hash.Tree.depth V) fun v => z (hr.φ v))))
    (consts : Fin C.N → Option Bool) (hone : consts (I.wire R.one) = some true)
    (hone_io : I.wire R.one ∈ (Verity.Partition.ofAudit P).io u)
    (hcom : ∀ j < 16 * H + 1, I.wire (hr.out j) ∈ P.committed) :
    ReadUnit.GatesRefine Verity.sha512 Verity.sha512 key consts (Verity.Partition.ofAudit P) u
      (readPorts I.wire hr V H) := by
  refine gatesRefine_of_vals key (Verity.Partition.ofAudit P) u I.wire hr hsound consts
    fun tr hcorrect hconsts => ?_
  have hc := correct_bits P hcorrect
  let xs : Fin C.N → Bool := fun g => ReadUnit.bit tr g
  let z : Fin R.w → Bool := fun c => P.localEval xs u (I.wire c)
  have hsat : R.Sat z := sat_localEval I xs
  have hnot : ∀ c : Fin R.w, c.val < R.nIn ∨ c = R.one → z c = xs (I.wire c) := by
    intro c hc'
    refine P.localEval_of_not_mem xs (I.outside c fun i hi => ?_)
    have h3 : (R.comp i).val = R.nIn + i.val := rfl
    rcases hc' with hlt | hc1
    · have h2 : (R.comp i).val < R.nIn := hi ▸ hlt
      omega
    · have h1 : R.nIn + R.nComp = (R.comp i).val := congrArg Fin.val (hc1.symm.trans hi)
      have h2 := i.isLt
      omega
  have hz1 : z R.one = true := by
    rw [hnot R.one (Or.inr rfl)]
    exact hconsts _ hone_io true hone
  refine ⟨z, hsat, hz1, fun c hc' => hnot c (Or.inl hc'), fun j hj => ?_⟩
  have hlen1 : run.1.1.length = 16 * H := by
    have := congrArg List.length (hsound z hsat hz1).1
    rwa [length_evs, length_h] at this
  obtain ⟨i, hi⟩ := hr.output j (by simp [hlen1]; omega)
  exact (hc _ (hcom j hj) (by rw [hi]; exact I.mem i)).symm

/-- **The read's gates refine its reference, where its rows hold the read's program.** Let unit `u` of the game
partition `P` be an instance of the rows `R` (`I`), and let `R` hold the read's program over the four parts' generators
(`hr`), for a table of `V ≤ 2 ^ D` rows of `H` words with `D` its tree's depth. If the four parts' statements hold, the
constant's gate is registered as 1 and is among `u`'s inputs, and the read's output gates are committed, then the read's
gates refine `Hash.EmbeddingRead` at SHA-512 under the hm96 key `key`, in core's view of the partition, at the read's
ports through `u`'s wires. -/
theorem gatesRefine_of_holdsRead (key : Hash.Msg) {V H D : ℕ} (hD : D = Hash.Tree.depth V) (hVD : V ≤ 2 ^ D)
    (hD32 : D ≤ 32) (hH : 16 * H < 2 ^ 64) {rowSeg : List Lin → M (List Word)}
    {leaf : List Word → List Word → M (List Word)}
    {frame : List Word → List Word → List Word → List Lin → List (List Word) → M Lin}
    {select : List Lin → Lin → List Lin → M (List Lin × Lin)} (hA : RowSegSpec (16 * H) rowSeg)
    (hB : Hm96LeafSpec key leaf) (hC : FrameReadSpec D Hash.RowLeaf.Format.seg.schema frame)
    (hS : SelectSpec V select) {C : FlockSoundness.Audit.Circuit} {n : ℕ} (P : FlockSoundness.Audit.Partition C n)
    (u : Fin n) {R : FlockSoundness.Rows} (I : P.IsRowsUnit u R)
    (hr : HoldsRead R (readIn H D) ((readProg D H rowSeg leaf frame select).run St.empty).2
      (((readProg D H rowSeg leaf frame select).run St.empty).1.1 ++
        [((readProg D H rowSeg leaf frame select).run St.empty).1.2]))
    (consts : Fin C.N → Option Bool) (hone : consts (I.wire R.one) = some true)
    (hone_io : I.wire R.one ∈ (Verity.Partition.ofAudit P).io u)
    (hcom : ∀ j < 16 * H + 1, I.wire (hr.out j) ∈ P.committed) :
    ReadUnit.GatesRefine Verity.sha512 Verity.sha512 key consts (Verity.Partition.ofAudit P) u
      (readPorts I.wire hr V H) := by
  subst hD
  exact gatesRefine_of_run key P u I hr
    (fun z hsat h1 => holdsRead_sound key rfl hVD hD32 hH hA hB hC hS hr.rows z hsat h1) consts hone hone_io hcom

end Accounting.ReadUnit.Gates
