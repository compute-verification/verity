module

public import Proofs.Flock.Bridge.Deployed
public import Proofs.Flock.Recursive.TopRoot
import all Init.Data.String.Defs
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
import Proofs.Flock.Soundness.Discharge.ZkReg.Checked

/-!
# A port's registered root is SHA-512 of its top frame

Each port's root is `FV3.root` over its own domain, the public file's published one (`domOf`), and that root is SHA-512
of one top frame (the top node's, or with one leaf that leaf's, or with none the pad's; `root_top`), whose first part
is the domain. With one leaf the frame ends in the port's leaf schema and its payload, and two frames at different row
schemas differ (`step_schema`), so two frames at one domain's place are at one schema (`topIn_ne`).

* `pairs m`, `rootsCost c nl`: the ordered pairs of ports, and the SHA-512 calls that recompute every port's root;
* `finderPr_one`, `finderPr_zero`: a finder whose every output is a collision finds one with probability 1, and one
  with none with probability 0.
-/

@[expose] public section

namespace Flock.Bridge.PortRoots

open Claims Claims.Sampling
open Flock.SecurityProofs
open Properties.Certifier.Bytes (bs bs_inj be64_size)
open scoped ENNReal

/-! ## Frames -/

theorem foldl_step_size : ∀ (ps : List ByteArray) (a : ByteArray),
    (ps.foldl Registered.step a).size = a.size + (ps.map fun p => 8 + p.size).sum
  | [], a => by simp
  | p :: ps, a => by
    rw [List.foldl_cons, foldl_step_size ps, Registered.step_size, be64_size, List.map_cons, List.sum_cons]
    omega

theorem sum_size_eq {ps ps' : List ByteArray} (h : List.Forall₂ (fun p p' : ByteArray => p.size = p'.size) ps ps') :
    (ps.map fun p => 8 + p.size).sum = (ps'.map fun p => 8 + p.size).sum := by
  induction h with
  | nil => rfl
  | cons hp _ ih => simp only [List.map_cons, List.sum_cons, hp, ih]

theorem foldl_step_inj {ps ps' : List ByteArray} (h : List.Forall₂ (fun p p' : ByteArray => p.size = p'.size) ps ps') :
    ∀ {a a' : ByteArray}, a.size = a'.size → ps.foldl Registered.step a = ps'.foldl Registered.step a' →
      a = a' ∧ ps = ps' := by
  induction h with
  | nil => intro a a' _ he; exact ⟨he, rfl⟩
  | cons hp _ ih =>
    intro a a' ha he
    obtain ⟨h1, rfl⟩ := ih (by rw [Registered.step_size, Registered.step_size, ha, hp]) he
    obtain ⟨rfl, rfl⟩ := Registered.step_inj ha hp h1
    exact ⟨rfl, rfl⟩

/-- **A frame binds its first part**, when its other parts have sizes part by part. -/
theorem frame_head_inj {tag : String} {d d' : ByteArray} {ps ps' : List ByteArray}
    (h : Registered.frame tag (d :: ps) = Registered.frame tag (d' :: ps'))
    (hps : List.Forall₂ (fun p p' : ByteArray => p.size = p'.size) ps ps') : d = d' := by
  have hs := congrArg ByteArray.size h
  simp only [Registered.frame, foldl_step_size, List.map_cons, List.sum_cons, sum_size_eq hps] at hs
  have hd : d.size = d'.size := by omega
  exact (List.cons.inj (foldl_step_inj (.cons hd hps) rfl h).2).1

/-- A frame's last part, of one size, and the rest. -/
theorem step_inj' {a a' p p' : ByteArray} (hp : p.size = p'.size) (h : Registered.step a p = Registered.step a' p') :
    a = a' ∧ p = p' := by
  have hs := congrArg ByteArray.size h
  rw [Registered.step_size, Registered.step_size, hp] at hs
  exact Registered.step_inj (by omega) hp h

theorem drop_of_append_eq {l₁ l₂ s t : List UInt8} (h : l₁ ++ s = l₂ ++ t) (hl : t.length ≤ s.length) :
    s.drop (s.length - t.length) = t := by
  have hs : s = s.take (s.length - t.length) ++ s.drop (s.length - t.length) := (List.take_append_drop _ _).symm
  rw [hs, ← List.append_assoc] at h
  exact (List.append_inj' h (by simp only [List.length_drop]; omega)).2

/-- The row schemas a port's leaves are framed at. -/
def schemas : List String := [HmRow.ROW_SCHEMA, HmRow.ROW_SCHEMA_V2, HmRow.ROW_SCHEMA_SEG]

theorem schemaOf_mem (c : Flock.Circuit) (q : ℕ) : schemaOf c q ∈ schemas := by
  unfold schemaOf schemas
  cases c.ports[q]? with
  | none => simp
  | some p =>
    simp only [Option.map_some, Option.getD_some, Registered.portSchema]
    split_ifs <;> simp

/-- **A frame's schema part ends it apart from its payload**: two frames that end in two row schemas are one only at
one schema (the schemas' last bytes differ). -/
theorem step_schema {a a' : ByteArray} {s s' : String} (hs : s ∈ schemas) (hs' : s' ∈ schemas)
    (h : Registered.step a s.toUTF8 = Registered.step a' s'.toUTF8) : s = s' := by
  have e : ∀ (b : ByteArray) (t : String), (Registered.step b t.toUTF8).data.toList =
      (b ++ Flock.be64 t.toUTF8.size.toUInt64).data.toList ++ t.toUTF8.data.toList := fun b t => by
    simp [Registered.step, ByteArray.data_append]
  have hl := (e a s).symm.trans ((congrArg (fun b : ByteArray => b.data.toList) h).trans (e a' s'))
  simp only [schemas, List.mem_cons, List.not_mem_nil, or_false] at hs hs'
  rcases hs with rfl | rfl | rfl <;> rcases hs' with rfl | rfl | rfl
  all_goals first
    | rfl
    | exact absurd (drop_of_append_eq hl (by decide)) (by decide)
    | exact absurd (drop_of_append_eq hl.symm (by decide)) (by decide)

/-! ## A port's root and its top frame -/

/-- Port `q`'s registered tree leaves, as `rootsOf` builds them. -/
def leaves (c : Flock.Circuit) (pub : Flock.Public) (nl q : ℕ) : Array ByteArray :=
  (Array.range nl).map fun u => Flock.Registered.leafAs (schemaOf c q) (domOf c pub q) u (regLeafOf c pub q u)

theorem leaves_size (c : Flock.Circuit) (pub : Flock.Public) (nl q : ℕ) : (leaves c pub nl q).size = nl := by
  simp [leaves]

theorem leaves_width (c : Flock.Circuit) (pub : Flock.Public) (nl q : ℕ) : ∀ x ∈ leaves c pub nl q, x.size = 64 := by
  intro x hx
  simp only [leaves, Array.mem_map] at hx
  obtain ⟨_, -, rfl⟩ := hx
  exact Flock.Sha512.hash_size _

theorem regLeafOf_size (c : Flock.Circuit) (pub : Flock.Public) (q u : ℕ) : (regLeafOf c pub q u).size = 64 := by
  unfold regLeafOf Flock.Hm96.Default512.treeLeaf
  exact Flock.Sha512.hash_size _

theorem one_le_log2ceil {n : ℕ} (h : 2 ≤ n) : 1 ≤ Flock.log2ceil n := by
  unfold Flock.log2ceil
  rw [ite_eq_right (by omega)]
  omega

/-- **What port `q`'s root hashes last**: the frame of its tree's top node; with one leaf, that leaf's frame; with
none, the lone pad's. -/
def topIn (c : Flock.Circuit) (pub : Flock.Public) (nl q : ℕ) : ByteArray :=
  if nl = 0 then Registered.frame "pad" [domOf c pub q, Flock.FV3.uint 0]
  else if nl = 1 then Registered.leafInAs (schemaOf c q) (domOf c pub q) 0 (regLeafOf c pub q 0)
  else Registered.nodeIn (domOf c pub q) (Flock.log2ceil nl - 1) 0
    (TopRoot.lev (domOf c pub q) (leaves c pub nl q) (Flock.log2ceil nl - 1))[0]!
    (TopRoot.lev (domOf c pub q) (leaves c pub nl q) (Flock.log2ceil nl - 1))[1]!

theorem lev_top_size (c : Flock.Circuit) (pub : Flock.Public) {nl : ℕ} (h2 : 2 ≤ nl) (q : ℕ) {i : ℕ} (hi : i < 2) :
    ((TopRoot.lev (domOf c pub q) (leaves c pub nl q) (Flock.log2ceil nl - 1))[i]!).size = 64 := by
  have h1 := one_le_log2ceil h2
  refine TopRoot.lev_width _ _ (leaves_width c pub nl q) _ _ (TopRoot.get!_mem ?_)
  rw [TopRoot.lev_size _ _ _ (by rw [leaves_size]; omega), leaves_size,
    show Flock.log2ceil nl - (Flock.log2ceil nl - 1) = 1 by omega]
  simpa using hi

/-- **A port's root is SHA-512 of its top frame.** -/
theorem root_top (c : Flock.Circuit) (pub : Flock.Public) (nl q : ℕ) :
    Flock.FV3.root (domOf c pub q) (leaves c pub nl q) Flock.Sha512.hash = Flock.Sha512.hash (topIn c pub nl q) := by
  rw [TopRoot.root_eq, leaves_size]
  rcases Nat.lt_or_ge nl 2 with hlt | h2
  · interval_cases nl
    · simp [TopRoot.lev, TopRoot.base, leaves, topIn, Flock.nextPow2, Flock.log2ceil, Registered.hash_frame]
    · simp [TopRoot.lev, TopRoot.base, leaves, topIn, Flock.nextPow2, Flock.log2ceil, Registered.leafAs_eq]
  · obtain ⟨k, hk⟩ : ∃ k, Flock.log2ceil nl = k + 1 := ⟨Flock.log2ceil nl - 1, by have := one_le_log2ceil h2; omega⟩
    have hk' : Flock.log2ceil nl - 1 = k := by omega
    have hsz : 0 < (TopRoot.lev (domOf c pub q) (leaves c pub nl q) k).size / 2 := by
      rw [TopRoot.lev_size _ _ _ (by rw [leaves_size]; omega), leaves_size, hk, show k + 1 - k = 1 by omega]
      decide
    rw [hk, TopRoot.lev, TopRoot.next_get _ _ _ hsz]
    simp only [topIn, ite_eq_right (show nl ≠ 0 by omega), ite_eq_right (show nl ≠ 1 by omega), hk', Nat.mul_zero,
      Nat.zero_add]

/-- **Two top frames over different domains differ.** -/
theorem topIn_ne {c : Flock.Circuit} {pub : Flock.Public} {nl q q' : ℕ} (hd : domOf c pub q' ≠ domOf c pub q) :
    topIn c pub nl q' ≠ topIn c pub nl q := by
  intro h
  apply hd
  unfold topIn at h
  split_ifs at h with h0 h1
  · exact frame_head_inj h (.cons rfl .nil)
  · have hs : schemaOf c q' = schemaOf c q := by
      have h' := h
      simp only [Registered.leafInAs, Registered.frame, List.foldl] at h'
      exact step_schema (schemaOf_mem c q') (schemaOf_mem c q)
        (step_inj' (by rw [regLeafOf_size, regLeafOf_size]) h').1
    rw [hs] at h
    exact frame_head_inj h (.cons rfl (.cons rfl (.cons rfl (.cons (by rw [regLeafOf_size, regLeafOf_size]) .nil))))
  · exact frame_head_inj h (.cons rfl (.cons rfl
      (.cons (by rw [lev_top_size c pub (by omega) q' (by omega), lev_top_size c pub (by omega) q (by omega)])
      (.cons (by rw [lev_top_size c pub (by omega) q' (by omega), lev_top_size c pub (by omega) q (by omega)])
        .nil))))

/-! ## The guard the deployed check needs -/

/-! ## The finder, and the term in `ecr/sha-512` -/

/-- Pairs `(q', q)` of ports, `q' < q < m`. -/
def pairs (m : ℕ) : List (ℕ × ℕ) := (List.range m).flatMap fun q => (List.range q).map fun q' => (q', q)

theorem mem_pairs {m q' q : ℕ} : (q', q) ∈ pairs m ↔ q' < q ∧ q < m := by
  simp only [pairs, List.mem_flatMap, List.mem_range, List.mem_map, Prod.mk.injEq]
  constructor
  · rintro ⟨a, ha, b, hb, rfl, rfl⟩
    exact ⟨hb, ha⟩
  · rintro ⟨h1, h2⟩
    exact ⟨q, h2, q', h1, rfl, rfl⟩

/-- **SHA-512 calls to recompute every port's root**: per port, `nl` hm96 tree leaves, `nl` frame-v3 leaves, and at
most `2 · nextPow2 nl` pads and nodes. -/
def rootsCost (c : Flock.Circuit) (nl : ℕ) : ℕ := c.ports.size * (2 * nl + 2 * Flock.nextPow2 nl)

theorem finderPr_one {H : Msg → Msg} {F : Claims.Finder}
    (h : ∀ s, ∃ u v, F.out s = some (u, v) ∧ Claims.Collision H u v) : F.Pr H = 1 := by
  have hc : (Fintype.card F.S : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr Fintype.card_ne_zero
  unfold Claims.Finder.Pr Claims.Pr Claims.prUniform
  simp [h, hc]

theorem finderPr_zero {H : Msg → Msg} {F : Claims.Finder}
    (h : ∀ s, ¬ ∃ u v, F.out s = some (u, v) ∧ Claims.Collision H u v) : F.Pr H = 0 := by
  unfold Claims.Finder.Pr Claims.Pr Claims.prUniform
  simp [h]

end Flock.Bridge.PortRoots
