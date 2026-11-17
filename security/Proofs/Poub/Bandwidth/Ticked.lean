module

public import Proofs.Poub.Bandwidth.Interactive
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Algebra.Order.Ring.Star
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.Int.Star
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.RingTheory.WittVector.IsPoly
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

/-! B3 for the chip as a tick-by-tick transducer over the links' signals, in the reads form and the capacity form. -/

@[expose] public section

namespace Bandwidth

open Pous (Bits pr)
open scoped ENNReal

namespace Ticked

abbrev Obs (h : ℕ) := List Bool × List Bool × List (Bits h)

structure Chip (N b h S : ℕ) where
  key : Bits S → List (Obs h) → ℕ
  width : Bits S → List (Obs h) → ℕ
  out : Bits S → List (Obs h) → List Bool
  ask : Bits S → List (Obs h) → List (Fin N × Bits b)
  tags : Bits S → List (Obs h) → Fin N → Bits h

abbrev Hbm (N b : ℕ) := Store N b → List ℕ → ℕ → Bool

abbrev Rest (N b h : ℕ) := Store N b → Oracle N b h → List Bool → ℕ → ℕ → Bool

def hbmBits {h : ℕ} (o : List (Obs h)) : ℕ := (o.map fun x => x.1.length).sum

variable {N b h S : ℕ} {α : Type}

def readMem (a : ℕ → ℕ) : ℕ → (List Bool → IChip N b h α) → IChip N b h α
  | 0, k => k []
  | n + 1, k => IChip.ingress (a n) fun β => readMem a n fun l => k (β :: l)

def readSide (a : ℕ → ℕ) : ℕ → (List Bool → IChip N b h α) → IChip N b h α
  | 0, k => k []
  | n + 1, k => IChip.side (a n) fun β => readSide a n fun l => k (β :: l)

def askAll : List (Fin N × Bits b) → (List (Bits h) → IChip N b h α) → IChip N b h α
  | [], k => k []
  | x :: xs, k => IChip.tag x fun y => askAll xs fun ys => k (y :: ys)

def sendAll : List Bool → IChip N b h α → IChip N b h α
  | [], p => p
  | β :: bs, p => IChip.send β (sendAll bs p)

def memAddr (ks : List ℕ) (j : ℕ) : ℕ := Encodable.encode (ks, j)

def sideAddr (u j : ℕ) : ℕ := Encodable.encode (u, j)

def prog (C : Chip N b h S) (βs : ℕ) (st : Bits S) (fin : List (Obs h) → α) :
    ℕ → List (Obs h) → List ℕ → IChip N b h α
  | 0, o, _ => IChip.done (fin o)
  | n + 1, o, ks =>
    sendAll (C.out st o) <|
      readMem (memAddr (C.key st o :: ks)) (C.width st o) fun m =>
        readSide (sideAddr o.length) βs fun r =>
          askAll (C.ask st o) fun ys => prog C βs st fin n ((m, r, ys) :: o) (C.key st o :: ks)

def mem (hbm : Hbm N b) (E : Store N b) (a : ℕ) : Bool :=
  match (Encodable.decode a : Option (List ℕ × ℕ)) with
  | some (ks, j) => hbm E ks j
  | none => false

def side (rest : Rest N b h) (E : Store N b) (O : Oracle N b h) (v : List Bool) (a : ℕ) : Bool :=
  match (Encodable.decode a : Option (ℕ × ℕ)) with
  | some (u, j) => rest E O v u j
  | none => false

def world (hbm : Hbm N b) (rest : Rest N b h) (E : Store N b) (O : Oracle N b h) : IWorld N b h :=
  ⟨O, mem hbm E, side rest E O, E⟩

def observe (C : Chip N b h S) (βs : ℕ) (st : Bits S) (hbm : Hbm N b) (rest : Rest N b h) (E : Store N b)
    (O : Oracle N b h) : ℕ → List (Obs h) → List ℕ → List Bool → List (Obs h)
  | 0, o, _, _ => o
  | n + 1, o, ks, v =>
    observe C βs st hbm rest E O n
      (((List.range (C.width st o)).reverse.map (hbm E (C.key st o :: ks)),
        (List.range βs).reverse.map (rest E O ((C.out st o).reverse ++ v) o.length), (C.ask st o).map O) :: o)
      (C.key st o :: ks) ((C.out st o).reverse ++ v)

section Blocks

variable (w : IWorld N b h)

theorem readMem_eq (a : ℕ → ℕ) (n : ℕ) (k : List Bool → IChip N b h α) (v : List Bool) :
    (readMem a n k).run w v = (k ((List.range n).reverse.map fun j => w.mem (a j))).run w v ∧
      (readMem a n k).ingressBits w v = (k ((List.range n).reverse.map fun j => w.mem (a j))).ingressBits w v + n ∧
      (readMem a n k).sideBits w v = (k ((List.range n).reverse.map fun j => w.mem (a j))).sideBits w v ∧
      (readMem a n k).regens w v = (k ((List.range n).reverse.map fun j => w.mem (a j))).regens w v ∧
      (readMem a n k).queries w v = (k ((List.range n).reverse.map fun j => w.mem (a j))).queries w v := by
  induction n generalizing k with
  | zero => simp [readMem]
  | succ n ih =>
    obtain ⟨h1, h2, h3, h4, h5⟩ := ih fun l => k (w.mem (a n) :: l)
    simp only [readMem, IChip.run, IChip.ingressBits, IChip.sideBits, IChip.regens, IChip.queries, List.range_succ,
      List.reverse_append, List.reverse_singleton, List.singleton_append, List.map_cons]
    exact ⟨h1, by rw [h2]; omega, h3, h4, h5⟩

theorem readSide_eq (a : ℕ → ℕ) (n : ℕ) (k : List Bool → IChip N b h α) (v : List Bool) :
    (readSide a n k).run w v = (k ((List.range n).reverse.map fun j => w.side v (a j))).run w v ∧
      (readSide a n k).ingressBits w v = (k ((List.range n).reverse.map fun j => w.side v (a j))).ingressBits w v ∧
      (readSide a n k).sideBits w v = (k ((List.range n).reverse.map fun j => w.side v (a j))).sideBits w v + n ∧
      (readSide a n k).regens w v = (k ((List.range n).reverse.map fun j => w.side v (a j))).regens w v ∧
      (readSide a n k).queries w v = (k ((List.range n).reverse.map fun j => w.side v (a j))).queries w v := by
  induction n generalizing k with
  | zero => simp [readSide]
  | succ n ih =>
    obtain ⟨h1, h2, h3, h4, h5⟩ := ih fun l => k (w.side v (a n) :: l)
    simp only [readSide, IChip.run, IChip.ingressBits, IChip.sideBits, IChip.regens, IChip.queries, List.range_succ,
      List.reverse_append, List.reverse_singleton, List.singleton_append, List.map_cons]
    exact ⟨h1, h2, by rw [h3]; omega, h4, h5⟩

theorem askAll_eq (xs : List (Fin N × Bits b)) (k : List (Bits h) → IChip N b h α) (v : List Bool) :
    (askAll xs k).run w v = (k (xs.map w.oracle)).run w v ∧
      (askAll xs k).ingressBits w v = (k (xs.map w.oracle)).ingressBits w v ∧
      (askAll xs k).sideBits w v = (k (xs.map w.oracle)).sideBits w v ∧
      (askAll xs k).regens w v = (k (xs.map w.oracle)).regens w v ∧
      (askAll xs k).queries w v = xs ++ (k (xs.map w.oracle)).queries w v := by
  induction xs generalizing k with
  | nil => simp [askAll]
  | cons x xs ih =>
    obtain ⟨h1, h2, h3, h4, h5⟩ := ih fun ys => k (w.oracle x :: ys)
    simp only [askAll, IChip.run, IChip.ingressBits, IChip.sideBits, IChip.regens, IChip.queries, List.map_cons,
      List.cons_append]
    exact ⟨h1, h2, h3, h4, by rw [h5]⟩

theorem sendAll_eq (bs : List Bool) (p : IChip N b h α) (v : List Bool) :
    (sendAll bs p).run w v = p.run w (bs.reverse ++ v) ∧
      (sendAll bs p).ingressBits w v = p.ingressBits w (bs.reverse ++ v) ∧
      (sendAll bs p).sideBits w v = p.sideBits w (bs.reverse ++ v) ∧
      (sendAll bs p).regens w v = p.regens w (bs.reverse ++ v) ∧
      (sendAll bs p).queries w v = p.queries w (bs.reverse ++ v) := by
  induction bs generalizing v with
  | nil => simp [sendAll]
  | cons β bs ih =>
    obtain ⟨h1, h2, h3, h4, h5⟩ := ih (β :: v)
    simp only [sendAll, IChip.run, IChip.ingressBits, IChip.sideBits, IChip.regens, IChip.queries,
      List.reverse_cons, List.append_assoc, List.singleton_append]
    exact ⟨h1, h2, h3, h4, h5⟩

end Blocks

theorem mem_memAddr (hbm : Hbm N b) (E : Store N b) (ks : List ℕ) (j : ℕ) : mem hbm E (memAddr ks j) = hbm E ks j := by
  simp [mem, memAddr]

theorem side_sideAddr (rest : Rest N b h) (E : Store N b) (O : Oracle N b h) (v : List Bool) (u j : ℕ) :
    side rest E O v (sideAddr u j) = rest E O v u j := by
  simp [side, sideAddr]

theorem world_mem (hbm : Hbm N b) (rest : Rest N b h) (E : Store N b) (O : Oracle N b h) (ks : List ℕ) (j : ℕ) :
    (world hbm rest E O).mem (memAddr ks j) = hbm E ks j :=
  mem_memAddr hbm E ks j

theorem world_side (hbm : Hbm N b) (rest : Rest N b h) (E : Store N b) (O : Oracle N b h) (v : List Bool) (u j : ℕ) :
    (world hbm rest E O).side v (sideAddr u j) = rest E O v u j :=
  side_sideAddr rest E O v u j

theorem prog_run (C : Chip N b h S) (βs : ℕ) (st : Bits S) (fin : List (Obs h) → α) (hbm : Hbm N b)
    (rest : Rest N b h) (E : Store N b) (O : Oracle N b h) (n : ℕ) (o : List (Obs h)) (ks : List ℕ) (v : List Bool) :
    (prog C βs st fin n o ks).run (world hbm rest E O) v = fin (observe C βs st hbm rest E O n o ks v) := by
  induction n generalizing o ks v with
  | zero => rfl
  | succ n ih =>
    simp only [prog, observe]
    rw [(sendAll_eq _ _ _ _).1, (readMem_eq _ _ _ _ _).1, (readSide_eq _ _ _ _ _).1, (askAll_eq _ _ _ _).1, ih]
    simp only [world_mem, world_side]
    rfl

theorem prog_ingress (C : Chip N b h S) (βs : ℕ) (st : Bits S) (fin : List (Obs h) → α) (hbm : Hbm N b)
    (rest : Rest N b h) (E : Store N b) (O : Oracle N b h) (n : ℕ) (o : List (Obs h)) (ks : List ℕ) (v : List Bool) :
    (prog C βs st fin n o ks).ingressBits (world hbm rest E O) v + hbmBits o =
      hbmBits (observe C βs st hbm rest E O n o ks v) := by
  induction n generalizing o ks v with
  | zero => simp [prog, observe, IChip.ingressBits]
  | succ n ih =>
    simp only [prog, observe]
    rw [(sendAll_eq _ _ _ _).2.1, (readMem_eq _ _ _ _ _).2.1, (readSide_eq _ _ _ _ _).2.1, (askAll_eq _ _ _ _).2.1]
    simp only [world_mem, world_side]
    rw [← ih]
    simp only [hbmBits, List.map_cons, List.sum_cons, List.length_map, List.length_reverse, List.length_range]
    exact (by omega : ∀ x c d : ℕ, x + c + d = x + (c + d)) _ _ _

theorem prog_counts (C : Chip N b h S) (βs q : ℕ) (hq : ∀ st o, (C.ask st o).length ≤ q) (st : Bits S)
    (fin : List (Obs h) → α) (w : IWorld N b h) (n : ℕ) (o : List (Obs h)) (ks : List ℕ) (v : List Bool) :
    (prog C βs st fin n o ks).sideBits w v = n * βs ∧ (prog C βs st fin n o ks).regens w v = 0 ∧
      ((prog C βs st fin n o ks).queries w v).length ≤ n * q := by
  induction n generalizing o ks v with
  | zero => simp [prog, IChip.sideBits, IChip.regens, IChip.queries]
  | succ n ih =>
    simp only [prog]
    rw [(sendAll_eq _ _ _ _).2.2.1, (sendAll_eq _ _ _ _).2.2.2.1, (sendAll_eq _ _ _ _).2.2.2.2,
      (readMem_eq _ _ _ _ _).2.2.1, (readMem_eq _ _ _ _ _).2.2.2.1, (readMem_eq _ _ _ _ _).2.2.2.2,
      (readSide_eq _ _ _ _ _).2.2.1, (readSide_eq _ _ _ _ _).2.2.2.1, (readSide_eq _ _ _ _ _).2.2.2.2,
      (askAll_eq _ _ _ _).2.2.1, (askAll_eq _ _ _ _).2.2.2.1, (askAll_eq _ _ _ _).2.2.2.2]
    obtain ⟨i1, i2, i3⟩ := ih (((List.range (C.width st o)).reverse.map fun j => w.mem (memAddr (C.key st o :: ks) j),
        (List.range βs).reverse.map fun j => w.side ((C.out st o).reverse ++ v) (sideAddr o.length j),
        (C.ask st o).map w.oracle) :: o) (C.key st o :: ks) ((C.out st o).reverse ++ v)
    refine ⟨by rw [i1]; ring, i2, ?_⟩
    rw [List.length_append]
    have := hq st o
    nlinarith [i3]

theorem hbmBits_le (C : Chip N b h S) (βs βm : ℕ) (hw : ∀ st o, C.width st o ≤ βm) (st : Bits S) (hbm : Hbm N b)
    (rest : Rest N b h) (E : Store N b) (O : Oracle N b h) (n : ℕ) (o : List (Obs h)) (ks : List ℕ) (v : List Bool) :
    hbmBits (observe C βs st hbm rest E O n o ks v) ≤ n * βm + hbmBits o := by
  induction n generalizing o ks v with
  | zero => simp [observe]
  | succ n ih =>
    simp only [observe]
    refine (ih _ _ _).trans ?_
    simp only [hbmBits, List.map_cons, List.sum_cons, List.length_map, List.length_reverse, List.length_range]
    have := hw st o
    nlinarith

def device (C : Chip N b h S) (βs T : ℕ) (state : Store N b → Bits S) (hbm : Hbm N b) (rest : Rest N b h)
    (c q : ℕ) : InteractiveDevice N b h S c 0 q :=
  InteractiveDevice.ofRuns state (mem hbm) (side rest) fun st => prog C βs st (C.tags st) T [] []

theorem device_run (C : Chip N b h S) (βs qt T : ℕ) (hq : ∀ st o, (C.ask st o).length ≤ qt)
    (state : Store N b → Bits S) (hbm : Hbm N b) (rest : Rest N b h) (E : Store N b) (O : Oracle N b h) :
    (device C βs T state hbm rest (T * βs) (T * qt)).tags E O =
        C.tags (state E) (observe C βs (state E) hbm rest E O T [] [] []) ∧
      (device C βs T state hbm rest (T * βs) (T * qt)).ingress E O =
        hbmBits (observe C βs (state E) hbm rest E O T [] [] []) := by
  obtain ⟨c2, c3, c4⟩ := prog_counts C βs qt hq (state E) (C.tags (state E)) (world hbm rest E O) T [] [] []
  obtain ⟨ht, hi⟩ := InteractiveDevice.ofRuns_run (c := T * βs) (G := 0) (q := T * qt) state (mem hbm) (side rest)
    (fun st => prog C βs st (C.tags st) T [] []) E O ⟨c2.le, c3.le, c4⟩
  exact ⟨ht.trans (prog_run C βs (state E) (C.tags (state E)) hbm rest E O T [] [] []),
    hi.trans ((add_zero _).symm.trans (prog_ingress C βs (state E) (C.tags (state E)) hbm rest E O T [] [] []))⟩

end Ticked

namespace SecurityProofs.Ticked

open Bandwidth.Ticked

variable {N b h S : ℕ}

theorem audit_reads (C : Ticked.Chip N b h S) (βs qt T : ℕ) (hq : ∀ st o, (C.ask st o).length ≤ qt)
    (state : Store N b → Bits S) (hbm : Hbm N b) (rest : Rest N b h) (s k m lam : ℕ) :
    pr (fun x : Store N b × Oracle N b h × (Fin s → Fin N) =>
        (∀ i, C.tags (state x.1) (observe C βs (state x.1) hbm rest x.1 x.2.1 T [] [] []) (x.2.2 i) =
            x.2.1 (x.2.2 i, x.1 (x.2.2 i))) ∧
          hbmBits (observe C βs (state x.1) hbm rest x.1 x.2.1 T [] [] []) + S + βs * T + m * b + lam < k * b)
      ≤ (((k - 1 : ℕ) : ℝ≥0∞) / N) ^ s + (N.choose m : ℝ≥0∞) * 2 ^ (T * βs) / 2 ^ (h * m)
        + (((T * qt).choose (k - m) : ℕ) : ℝ≥0∞) / 2 ^ lam := by
  refine (pr_mono fun x hx => ?_).trans
    (AuditInteractive N b h S (T * βs) 0 (T * qt) s k m lam (device C βs T state hbm rest (T * βs) (T * qt)))
  obtain ⟨ht, hi⟩ := device_run C βs qt T hq state hbm rest x.1 x.2.1
  refine ⟨fun i => ?_, ?_⟩
  · rw [ht]
    exact hx.1 i
  · rw [hi]
    have := hx.2
    nlinarith

theorem audit (C : Ticked.Chip N b h S) (βm βs qt T : ℕ) (hw : ∀ st o, C.width st o ≤ βm)
    (hq : ∀ st o, (C.ask st o).length ≤ qt) (state : Store N b → Bits S) (hbm : Hbm N b) (rest : Rest N b h)
    (s k m lam : ℕ) :
    pr (fun x : Store N b × Oracle N b h × (Fin s → Fin N) =>
        (∀ i, C.tags (state x.1) (observe C βs (state x.1) hbm rest x.1 x.2.1 T [] [] []) (x.2.2 i) =
            x.2.1 (x.2.2 i, x.1 (x.2.2 i))) ∧
          βm * T + S + βs * T + m * b + lam < k * b)
      ≤ (((k - 1 : ℕ) : ℝ≥0∞) / N) ^ s + (N.choose m : ℝ≥0∞) * 2 ^ (T * βs) / 2 ^ (h * m)
        + (((T * qt).choose (k - m) : ℕ) : ℝ≥0∞) / 2 ^ lam := by
  refine (pr_mono fun x hx => ⟨hx.1, ?_⟩).trans (audit_reads C βs qt T hq state hbm rest s k m lam)
  have := hbmBits_le C βs βm hw (state x.1) hbm rest x.1 x.2.1 T [] [] []
  have := hx.2
  simp only [hbmBits, List.map_nil, List.sum_nil] at *
  nlinarith

theorem audit_reads_ro (C : Ticked.Chip N b h S) (βs qt T : ℕ) (hq : ∀ st o, (C.ask st o).length ≤ qt)
    (state : Store N b → Bits S) (hbm : Hbm N b) (rest : Rest N b h) (hrest : ∀ E O O', rest E O = rest E O')
    (s k lam : ℕ) :
    pr (fun x : Store N b × Oracle N b h × (Fin s → Fin N) =>
        (∀ i, C.tags (state x.1) (observe C βs (state x.1) hbm rest x.1 x.2.1 T [] [] []) (x.2.2 i) =
            x.2.1 (x.2.2 i, x.1 (x.2.2 i))) ∧
          hbmBits (observe C βs (state x.1) hbm rest x.1 x.2.1 T [] [] []) + S + βs * T + lam < k * b)
      ≤ (((k - 1 : ℕ) : ℝ≥0∞) / N) ^ s + (s : ℝ≥0∞) / 2 ^ h + (((T * qt).choose k : ℕ) : ℝ≥0∞) / 2 ^ lam := by
  have hside : ∀ E O O', (device C βs T state hbm rest (T * βs) (T * qt)).side E O =
      (device C βs T state hbm rest (T * βs) (T * qt)).side E O' := by
    intro E O O'
    funext v a
    simp only [device, InteractiveDevice.ofRuns, side, hrest E O O']
  refine (pr_mono fun x hx => ?_).trans
    (AuditInteractiveRO N b h S (T * βs) 0 (T * qt) s k lam (device C βs T state hbm rest (T * βs) (T * qt)) hside)
  obtain ⟨ht, hi⟩ := device_run C βs qt T hq state hbm rest x.1 x.2.1
  refine ⟨fun i => ?_, ?_⟩
  · rw [ht]
    exact hx.1 i
  · rw [hi]
    have := hx.2
    nlinarith

theorem audit_ro (C : Ticked.Chip N b h S) (βm βs qt T : ℕ) (hw : ∀ st o, C.width st o ≤ βm)
    (hq : ∀ st o, (C.ask st o).length ≤ qt) (state : Store N b → Bits S) (hbm : Hbm N b) (rest : Rest N b h)
    (hrest : ∀ E O O', rest E O = rest E O') (s k lam : ℕ) :
    pr (fun x : Store N b × Oracle N b h × (Fin s → Fin N) =>
        (∀ i, C.tags (state x.1) (observe C βs (state x.1) hbm rest x.1 x.2.1 T [] [] []) (x.2.2 i) =
            x.2.1 (x.2.2 i, x.1 (x.2.2 i))) ∧
          βm * T + S + βs * T + lam < k * b)
      ≤ (((k - 1 : ℕ) : ℝ≥0∞) / N) ^ s + (s : ℝ≥0∞) / 2 ^ h + (((T * qt).choose k : ℕ) : ℝ≥0∞) / 2 ^ lam := by
  refine (pr_mono fun x hx => ⟨hx.1, ?_⟩).trans (audit_reads_ro C βs qt T hq state hbm rest hrest s k lam)
  have := hbmBits_le C βs βm hw (state x.1) hbm rest x.1 x.2.1 T [] [] []
  have := hx.2
  simp only [hbmBits, List.map_nil, List.sum_nil] at *
  nlinarith

end SecurityProofs.Ticked

end Bandwidth
