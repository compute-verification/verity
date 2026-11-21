module

public import Proofs.Flock.Bridge.Canon
public import Proofs.Flock.Claims.Bridge.Transports
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
import Mathlib.Tactic.ReduceModChar
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.C3.RegLevels.ValueTie
import Proofs.Flock.Soundness.Discharge.PlainReg.Checked
import Std.Tactic.BVDecide.Normalize.BitVec

/-!
# The deployed commitment's pieces at a session's rows

`commitDep` (the deployed trace commitment at a session's rows) reads a claims position's registered row at a port of
`16 w` bits, `sha512/row/v1`, `sha512/row/v2` or `sha512/row-seg/v1`. The deployed row message in the port's format is
the game's bytes of the value (`rowMsg_eq_encPortH`, `rowDep_eq`), and the registered leaf is framed at the port's
schema (`schemaOf`).

The deployed check refuses other values: a misshaped value's leaf is the leaf of an opening `commitDep` passes only by
a SHA-512 collision (`leaf_misshaped_dep`, `encPortH_ne_misshaped`), and so is a leaf under a salt of another length
(`leaf_salt`): `commitDep` refuses data shorter than the salt (`pathOf`). Wires and positions: `posOf` is injective
and ordered (`posOf_inj`, `posOf_lt`), and `wireOf` inverts it (`wireOf_posOf`). The game's openings at a Flock session:
`vbOf`.
-/

@[expose] public section

namespace Flock.Bridge

open Claims Claims.Sampling
open FlockSoundness.Refine FlockSoundness.Binding
open FlockSoundness.Discharge.Layout (BitVal RowShaped rowEncP)
open Properties.Certifier.Bytes (bs)

/-- A row's bits as the deployed data carries them, one byte each. -/
def bitMsg (b : List Bool) : Msg := b.map fun x => if x then 1 else 0

/-- The length of a deployed opening's salt: `saltLen` bytes wherever its path is whole levels, since data shorter
than the salt has no such path (`saltLen_le_of_pathOf`). -/
theorem saltOf_length {d k : ℕ} {o : Opening Bool} (h : (pathOf d o).length = 64 * k) :
    (saltOf o).length = saltLen := by
  have hl := saltLen_le_of_pathOf h
  simp only [saltOf, List.length_take]
  omega

section Layout

variable {C : Verity.Boolean.Circuit} {rows : Fin C.N → Option RowAt}

theorem placeOf_eq {i : ℕ} {r : RowAt} (hr : rowOfPos C rows i = some r) (hpl : r.place < r.width) :
    placeOf C rows i = ⟨r.width, ⟨r.place, hpl⟩⟩ := by
  simp only [placeOf, hr, hpl, ↓reduceDIte]

/-- Port `q` as the verifier registers it (`portsOf`'s `q`-th). -/
def portAt (c : Flock.Circuit) (pub : Flock.Public) (nl : ℕ) (roots : List Msg) (q : ℕ) : Flock.Registered.Port :=
  { name := (c.ports[q]?.map (·.name)).getD "", value := "", root := bytes (roots.getD q []), leaves := nl,
    words := none, wordBits := 0, domain := domOf c pub q, readsRel := #[], schema := schemaOf c q }

theorem portsOf_eq (c : Flock.Circuit) (pub : Flock.Public) (nl : ℕ) (roots : List Msg) :
    portsOf c pub nl roots = (Array.range c.ports.size).map (portAt c pub nl roots) := rfl

/-- **Under `RowsOk` a port's rows are `16 · words` bits**: its row width in the game (`portRowW`) is its bits. -/
theorem rowsOk_bits {c : Flock.Circuit} (h : RowsOk c) (q : Fin c.ports.size) :
    FlockSoundness.Discharge.Layout.portRowW c q = c.ports[q.val]!.bits := by
  rw [getElem!_pos c.ports q.val q.isLt]
  unfold FlockSoundness.Discharge.Layout.portRowW
  have hm : c.ports[q.val] ∈ c.ports.toList := by simp
  rcases FlockSoundness.Discharge.Layout.rowsH_spec h with h1 | ⟨h2, -, -⟩
  · exact (h1 _ hm).2.1.symm
  · rcases h2 _ hm with h | h
    · exact (FlockSoundness.Discharge.Layout.v2Row_spec h).2.2.1.symm
    · exact (FlockSoundness.Discharge.Layout.segRow_spec h).2.1.symm

/-- **A row's message in its port's format is the game's bytes of its value** (`HiddenExec.encPortH`), at a row of the
port's bits, `16 w` of them: `sha512/row-seg/v1`'s outer message is `encSeg`'s, `sha512/row/v2`'s and
`sha512/row/v1`'s are `rowEncP` under their first blocks. -/
theorem rowMsg_eq_encPortH {W : ℕ} {place : ℕ → (w : ℕ) × Fin w} {bits : Msg → ℕ → Bool} {o : Opening Bool}
    {p : Flock.Port} (hb : (place o.pos).1 = p.bits) (h16 : (place o.pos).1 % 16 = 0) (hW : (place o.pos).1 ≤ W) :
    rowMsg p (SamplingDeploy.rowBits place bits o) =
      FlockSoundness.Discharge.HiddenExec.encPortH p (SamplingDeploy.valOf place bits o W) := by
  obtain ⟨hs, hw⟩ := SamplingDeploy.valOf_shaped (bits := bits) h16 hW
  have hpk : Flock.HmRow.packBits (List.ofFn (SamplingDeploy.rowFn place bits o)) =
      FlockSoundness.Discharge.Layout.rowBytes (SamplingDeploy.valOf place bits o W) :=
    SamplingDeploy.packBits_eq_rowBytes _ h16 hW
  rw [SamplingDeploy.rowBits_eq]
  unfold rowMsg FlockSoundness.Discharge.HiddenExec.encPortH
  by_cases hseg : p.segs.isSome
  · simp only [hseg, ↓reduceIte, List.length_ofFn]
    unfold FlockSoundness.Discharge.Layout.encSeg
    rw [if_pos ⟨hs, hw.trans hb⟩, hpk, hb]
  · simp only [hseg, Bool.false_eq_true, ↓reduceIte]
    unfold FlockSoundness.Discharge.Layout.rowEncP
    rw [if_pos hs, hw]
    unfold FlockSoundness.Discharge.Hidden.rowPre
    cases p.v2
    · simp only [Bool.false_eq_true, ↓reduceIte, List.length_ofFn, hpk]
    · simp only [↓reduceIte]
      unfold FlockLevel3.RowV2.enc
      rw [List.length_ofFn, show 16 * ((place o.pos).1 / 16) = (place o.pos).1 by omega, hpk]

/-- **A deployed opening's row message is the game's bytes of its value** (`rowMsg_eq_encPortH` at its row's port), at
a position whose row has its port's bits. -/
theorem rowDep_eq {c : Flock.Circuit} {W : ℕ} {bits : Msg → ℕ → Bool} (o : Opening Bool) {r : RowAt}
    (hr : rowOfPos C rows o.pos = some r) (hq : r.port < c.ports.size) (hb : r.width = c.ports[r.port]!.bits)
    (hpl : r.place < r.width) (h16 : r.width % 16 = 0) (hW : r.width ≤ W) :
    rowDep C rows c (placeOf C rows) bits o =
      FlockSoundness.Discharge.HiddenExec.encPortH c.ports[r.port]! (SamplingDeploy.valOf (placeOf C rows) bits o W) := by
  have hp1 : (placeOf C rows o.pos).1 = r.width := by rw [placeOf_eq hr hpl]
  have hport : portOfPos C rows c o.pos = some c.ports[r.port]! := by
    rw [getElem!_pos c.ports r.port hq]
    simp [portOfPos, hr, hq]
  unfold rowDep
  rw [hport]
  dsimp only
  exact rowMsg_eq_encPortH (by rw [hp1, hb]) (by rw [hp1]; exact h16) (by rw [hp1]; exact hW)

/-- **A misshaped value's bytes are no shaped value's** at a port: a value that isn't `RowShaped` of the port's bits
has other bytes (`encPortH`) than one that is. -/
theorem encPortH_ne_misshaped {p : Flock.Port} {W : ℕ} {v v' : BitVal W} (hv' : RowShaped v' ∧ v'.1.val = p.bits)
    (hv : ¬ (RowShaped v ∧ v.1.val = p.bits)) :
    FlockSoundness.Discharge.HiddenExec.encPortH p v ≠ FlockSoundness.Discharge.HiddenExec.encPortH p v' := by
  intro h
  unfold FlockSoundness.Discharge.HiddenExec.encPortH at h
  split_ifs at h
  · unfold FlockSoundness.Discharge.Layout.encSeg at h
    rw [if_neg hv, if_pos hv'] at h
    have e := congrArg List.length h
    rw [List.length_append, List.length_replicate, FlockSoundness.Discharge.Layout.bitCode_length,
      FlockSoundness.Discharge.Layout.length_rowSegEnc p.bits (FlockSoundness.Discharge.Layout.rowBytes v') []] at e
    omega
  · have e := FlockSoundness.Discharge.Layout.rowEncP_inj (FlockSoundness.Discharge.Hidden.size_rowPre _) h
    subst e
    exact hv hv'

end Layout

/-! ## The deployed check refuses other salts -/

/-- **hm96's extractor outputs a SHA-512 collision** whenever two different salts have one leaf, under any rows. -/
theorem hm96Pair_spec_salt (r y r' y' : List UInt8) (hy : y ≠ y') (h : H512 (hm96Input r y) = H512 (hm96Input r' y')) :
    Collides (hm96Pair r y r' y') := by
  by_cases hr : r = r'
  · subst hr
    unfold Collides hm96Pair
    by_cases ho : hm96Input r y = hm96Input r y'
    · simp only [ho, hy, ↓reduceIte]
      have hcs : K512.leafPrefix ++ commitString r y = K512.leafPrefix ++ commitString r y' := ho
      exact ⟨fun e => hy (List.append_cancel_left e),
        enc512.dig_inj (commitString_split (List.append_cancel_left hcs)).2⟩
    · simp only [ho, ↓reduceIte]
      exact ⟨ho, h⟩
  · exact hm96Pair_spec r y r' y' hr h

/-- **An opening the deployed check passes carries a full path**: `64 · log2ceil nl` bytes after its salt. -/
theorem pathOf_length_of_opens (C : Verity.Boolean.Circuit) (rows : Fin C.N → Option RowAt) (c : Flock.Circuit)
    (pub : Flock.Public) (nl : ℕ) (roots : List Msg) (o : Opening Bool)
    (h : (commitDep C rows c pub nl roots).Opens sha512Msg roots o = true) :
    (pathOf (Flock.log2ceil nl) o).length = 64 * Flock.log2ceil nl := by
  unfold commitDep at h
  rw [SamplingDeploy.opens_pathCommitRow] at h
  obtain ⟨ri, reg, -, hf, ho⟩ := h
  have hm := Array.mem_of_find?_eq_some hf
  rw [portsOf_eq, Array.mem_map] at hm
  obtain ⟨q, -, rfl⟩ := hm
  unfold Flock.Registered.opensLeaf at ho
  simp only [Bool.and_eq_true, decide_eq_true_eq] at ho
  have h2 := ho.1.2
  rw [size_bytes] at h2
  exact h2

/-- **The deployed check refuses a salt of another length**: the leaf of a game opening `(r, y)` whose salt isn't
`saltLen` bytes is the leaf of an opening the deployed check passes only by a SHA-512 collision. -/
theorem leaf_salt (C : Verity.Boolean.Circuit) (rows : Fin C.N → Option RowAt) (c : Flock.Circuit)
    (pub : Flock.Public) (nl : ℕ) (roots : List Msg) (o : Opening Bool)
    (h : (commitDep C rows c pub nl roots).Opens sha512Msg roots o = true)
    {r y : Msg} (hy : y.length ≠ saltLen)
    (hl : H512 (hm96Input r y) =
      H512 (hm96Input (rowDep C rows c (placeOf C rows) (bitsOf (Flock.log2ceil nl)) o) (saltOf o))) :
    Collides (hm96Pair r y (rowDep C rows c (placeOf C rows) (bitsOf (Flock.log2ceil nl)) o) (saltOf o)) := by
  refine hm96Pair_spec_salt _ _ _ _ (fun e => hy ?_) hl
  rw [e]
  exact saltOf_length (pathOf_length_of_opens C rows c pub nl roots o h)

/-- **The deployed check refuses a misshaped value**: at a position whose row has its port's bits, `16 w` of them, a
game opening of a value that isn't `RowShaped` of the row's width has the leaf of a deployed opening's row only by a
SHA-512 collision. -/
theorem leaf_misshaped_dep (C : Verity.Boolean.Circuit) (rows : Fin C.N → Option RowAt) (c : Flock.Circuit)
    (d : ℕ) (o : Opening Bool) {r : RowAt} (hr : rowOfPos C rows o.pos = some r) (hq : r.port < c.ports.size)
    (hb : r.width = c.ports[r.port]!.bits) (hpl : r.place < r.width) (h16 : r.width % 16 = 0) {W : ℕ}
    (hW : r.width ≤ W) {v : BitVal W} (hv : ¬ (RowShaped v ∧ v.1.val = r.width)) (y : Msg)
    (h : H512 (hm96Input (FlockSoundness.Discharge.HiddenExec.encPortH c.ports[r.port]! v) y) =
      H512 (hm96Input (rowDep C rows c (placeOf C rows) (bitsOf d) o) (saltOf o))) :
    Collides (hm96Pair (FlockSoundness.Discharge.HiddenExec.encPortH c.ports[r.port]! v) y
      (rowDep C rows c (placeOf C rows) (bitsOf d) o) (saltOf o)) := by
  have hp1 : (placeOf C rows o.pos).1 = r.width := by rw [placeOf_eq hr hpl]
  obtain ⟨hs, hw⟩ := SamplingDeploy.valOf_shaped (bits := bitsOf d) (o := o) (place := placeOf C rows) (W := W)
    (by rw [hp1]; exact h16) (by rw [hp1]; exact hW)
  rw [hp1, hb] at hw
  rw [hb] at hv
  refine hm96Pair_spec _ _ _ _ (fun e => encPortH_ne_misshaped ⟨hs, hw⟩ hv ?_) h
  rw [e, rowDep_eq o hr hq hb hpl h16 hW]

/-! ## Wires and positions -/

theorem sum_filter_map {α : Type} (l : List α) (p : α → Bool) (f : α → ℕ) :
    ((l.filter p).map f).sum = (l.map fun x => if p x then f x else 0).sum := by
  induction l with
  | nil => rfl
  | cons a l ih => by_cases hp : p a <;> simp [hp, ih]

theorem off_eq_sum_ite (C : Verity.Boolean.Circuit) (g : ℕ) :
    off C g = ∑ h : Fin C.N, if h.val < g then gateSize (C.op h) else 0 := by
  rw [Fin.sum_univ_def, off, sum_filter_map]
  simp

theorem gateSize_pos {N : ℕ} (o : Verity.Boolean.Op N) : 0 < gateSize o := by
  cases o <;> simp [gateSize]

theorem off_add_le_val (C : Verity.Boolean.Circuit) {g g' : Fin C.N} (h : g.val < g'.val) :
    off C g.val + gateSize (C.op g) ≤ off C g'.val := by
  rw [off_eq_sum_ite, off_eq_sum_ite]
  calc (∑ h : Fin C.N, if h.val < g.val then gateSize (C.op h) else 0) + gateSize (C.op g)
      = ∑ h : Fin C.N, ((if h.val < g.val then gateSize (C.op h) else 0) +
          if h = g then gateSize (C.op g) else 0) := by
        rw [Finset.sum_add_distrib, Finset.sum_ite_eq' Finset.univ g]
        simp
    _ ≤ ∑ h : Fin C.N, if h.val < g'.val then gateSize (C.op h) else 0 := by
        refine Finset.sum_le_sum fun x _ => ?_
        rcases eq_or_ne x g with rfl | hx
        · simp [h]
        · simp only [hx, ↓reduceIte, Nat.add_zero]
          split_ifs <;> omega

theorem posOf_lt (C : Verity.Boolean.Circuit) {g g' : Fin C.N} (h : g.val < g'.val) : posOf C g < posOf C g' := by
  have h1 := off_add_le_val C h
  have h2 := gateSize_pos (C.op g)
  have h3 := gateSize_pos (C.op g')
  unfold posOf
  omega

theorem posOf_inj (C : Verity.Boolean.Circuit) {g g' : Fin C.N} (h : posOf C g = posOf C g') : g = g' := by
  rcases lt_trichotomy g.val g'.val with hl | he | hl
  · exact absurd h (posOf_lt C hl).ne
  · exact Fin.ext he
  · exact absurd h (posOf_lt C hl).ne'

theorem wireOf_posOf (C : Verity.Boolean.Circuit) (g : Fin C.N) : wireOf C (posOf C g) = some g := by
  unfold wireOf
  rw [List.find?_eq_some_iff_getElem]
  refine ⟨by simp, g.val, by simp, by simp, fun j hj => ?_⟩
  have hjN : j < C.N := by have := g.isLt; omega
  simp only [List.getElem_finRange, Bool.not_eq_true', decide_eq_false_iff_not]
  intro e
  have := congrArg Fin.val (posOf_inj C e)
  simp at this
  omega

theorem rowOfPos_posOf {C : Verity.Boolean.Circuit} (rows : Fin C.N → Option RowAt) (g : Fin C.N) :
    rowOfPos C rows (posOf C g) = rows g := by
  simp only [rowOfPos, wireOf_posOf, Option.bind_some]

/-! ## The game's openings at a Flock session -/

section Game

open FlockSoundness FlockSoundness.Game Spot.Sampling FlockSoundness.Discharge.Exec FlockSoundness.Discharge.ZkLink
  FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed FlockSoundness.Discharge.ZkHidden
  FlockSoundness.Discharge.ZkBind FlockSoundness.Discharge FlockSoundness.Discharge.Composed

variable {n mPts : ℕ} {Reg : Type} {I : Inputs} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
  {d₀ : Lean.Json} (y₀ : SampleSetupZHJ (inputsJ I 0) mPts S₀ d₀) (ht : I.tags.typed = false) {c : Flock.Circuit}
  (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hscope : Discharge.Layout.scopeOkH c = true)
  (hU : Built c.unit) {pub : Flock.Public} (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub)
  (hpt : pub.tables = none)

/-- **The game's value binding at a Flock session** (`SampledKSLive`'s `vb`). -/
noncomputable abbrev vbOf := (rsZHJ (Reg := Reg) dj y₀ ht hc hscope hU hpub hpt).bindingZ
  (rsZHJ_computes dj y₀ ht hc hscope hU hpub hpt)

/-- The session's commit strings: unit `u`'s row of port `q`, or a single gate. -/
abbrev PosH (c : Flock.Circuit) (hU : Built c.unit) (n : ℕ) : Type :=
  (Fin n × Fin c.ports.size) ⊕ Fin ((keyProg c hU n).circuit (Discharge.Layout.outsOf c hU n)).N

/-- The session's values: bit strings of at most the widest row's bits, with their width. -/
abbrev ValH (c : Flock.Circuit) : Type := BitVal (Finset.univ.sup (Discharge.Layout.portRowW c) + 1)

variable {dj y₀ ht hc hscope hU hpub hpt}
variable {g : Fin ((keyProg c hU n).circuit (Discharge.Layout.outsOf c hU n)).N} {u : Fin n} {q : Fin c.ports.size}
  {k : Fin (Discharge.Layout.portRowW c q)}

theorem vbOf_cs_fst (R : Reg) (p : PosH c hU n) :
    ((vbOf dj y₀ ht hc hscope hU hpub hpt (Reg := Reg)).cs R p).1 = p :=
  rfl

end Game


end Flock.Bridge
