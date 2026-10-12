import Mathlib
import Proofs.Accounting.Bandwidth.Interactive

/-!
# B3 for the cores as a tick-by-tick transducer over the links' alphabets

The bandwidth audit's model prover (`InteractiveDevice`) from the cores in the window, read tick by tick as v2.3 of
`note:console/20261012T0645Z-draft-actual-circuit-model` describes them: causal from the window's start, the cores'
behaviour is a function of their state then and of what they observe at each tick. Each link delivers one signal of
its alphabet a tick, so a signal is its index in the alphabet: `βm` bits a tick from HBM and `βs` from the rest of the
prover. The oracle answers the tick's queries. HBM hears only the cores: its signal is a function of the store and of the
keys the cores sent it. The rest may answer as any function of the store, the oracle and every bit the cores sent it.

`Ticked.Cores` is such a transducer, `Ticked.observe` what it observes over `T` ticks, and `Ticked.prog` the chip
program that reads each tick's signals bit by bit: `βm T` bits of device memory (addressed by the keys sent so far) and
`βs T` of side sources, with no regeneration (`prog_counts`). Its tags are the transducer's (`prog_run`).

`SecurityProofs.Ticked.audit` is then B3 in the capacity form: the probability that the tags pass while
`βm T + S + βs T + m b + λ < k b` is at most `AuditInteractive`'s bound at `c = βs T`. A width bound per run would not
do: an alphabet's signals can differ only in their width-0 wires, so the information a link carries in a tick is its
alphabet index, at most `β` bits, not the width delivered. `audit_ro` is `AuditInteractiveRO`'s bound when the rest
ignores the oracle.
-/

namespace Bandwidth

open Pous (Bits pr)
open scoped ENNReal

namespace Ticked

/-- What the cores observe in a tick: the bits of HBM's signal and of the rest's (each its index in its link's alphabet)
and the oracle's answers to the tick's queries. -/
abbrev Obs (h : ℕ) := List Bool × List Bool × List (Bits h)

/-- **The cores in the window, tick by tick**: from their state `st` at the window's start and what they observed (newest
first), the key they send HBM, the bits they send the rest and the oracle queries they make at a tick, and at the end
the tags they commit to. -/
structure Cores (N b h S : ℕ) where
  key : Bits S → List (Obs h) → ℕ
  out : Bits S → List (Obs h) → List Bool
  ask : Bits S → List (Obs h) → List (Fin N × Bits b)
  tags : Bits S → List (Obs h) → Fin N → Bits h

/-- HBM: bit `j` of its signal at a tick, from the store and the keys the cores sent it so far (newest first). -/
abbrev Hbm (N b : ℕ) := Store N b → List ℕ → ℕ → Bool

/-- The rest of the prover: bit `j` of its signal at tick `u`, from the store, the oracle and the bits the cores sent it
so far (newest first). -/
abbrev Rest (N b h : ℕ) := Store N b → Oracle N b h → List Bool → ℕ → ℕ → Bool

variable {N b h S : ℕ} {α : Type}

/-- Read `n` bits of device memory at addresses `a (n − 1), …, a 0`. -/
def readMem (a : ℕ → ℕ) : ℕ → (List Bool → IChip N b h α) → IChip N b h α
  | 0, k => k []
  | n + 1, k => IChip.ingress (a n) fun β => readMem a n fun l => k (β :: l)

/-- Read `n` bits of side sources at addresses `a (n − 1), …, a 0`. -/
def readSide (a : ℕ → ℕ) : ℕ → (List Bool → IChip N b h α) → IChip N b h α
  | 0, k => k []
  | n + 1, k => IChip.side (a n) fun β => readSide a n fun l => k (β :: l)

/-- Ask the oracle each query of a list. -/
def askAll : List (Fin N × Bits b) → (List (Bits h) → IChip N b h α) → IChip N b h α
  | [], k => k []
  | x :: xs, k => IChip.tag x fun y => askAll xs fun ys => k (y :: ys)

/-- Send each bit of a list. -/
def sendAll : List Bool → IChip N b h α → IChip N b h α
  | [], p => p
  | β :: bs, p => IChip.send β (sendAll bs p)

/-- Device memory's address of bit `j` of HBM's signal after the keys `ks`. -/
def memAddr (ks : List ℕ) (j : ℕ) : ℕ := Encodable.encode (ks, j)

/-- The side sources' address of bit `j` of the rest's signal at tick `u`. -/
def sideAddr (u j : ℕ) : ℕ := Encodable.encode (u, j)

/-- **The chip program** for the remaining `n` ticks, having observed `o` and sent HBM the keys `ks`: at each tick it
sends the rest its bits, reads HBM's signal from device memory and the rest's from the side sources, and asks the
oracle; at the end it commits to the transducer's tags. -/
def prog (C : Cores N b h S) (βm βs : ℕ) (st : Bits S) : ℕ → List (Obs h) → List ℕ → IChip N b h (Fin N → Bits h)
  | 0, o, _ => IChip.done (C.tags st o)
  | n + 1, o, ks =>
    sendAll (C.out st o) <|
      readMem (memAddr (C.key st o :: ks)) βm fun m =>
        readSide (sideAddr o.length) βs fun r =>
          askAll (C.ask st o) fun ys => prog C βm βs st n ((m, r, ys) :: o) (C.key st o :: ks)

/-- HBM as device memory: the bit at `memAddr ks j` is bit `j` of its signal after the keys `ks`. -/
def mem (hbm : Hbm N b) (E : Store N b) (a : ℕ) : Bool :=
  match (Encodable.decode a : Option (List ℕ × ℕ)) with
  | some (ks, j) => hbm E ks j
  | none => false

/-- The rest as side sources: the bit at `sideAddr u j` is bit `j` of its signal at tick `u`. -/
def side (rest : Rest N b h) (E : Store N b) (O : Oracle N b h) (v : List Bool) (a : ℕ) : Bool :=
  match (Encodable.decode a : Option (ℕ × ℕ)) with
  | some (u, j) => rest E O v u j
  | none => false

/-- The world the chip runs in, for store `E` and oracle `O`. -/
def world (hbm : Hbm N b) (rest : Rest N b h) (E : Store N b) (O : Oracle N b h) : IWorld N b h :=
  ⟨O, mem hbm E, side rest E O, E⟩

/-- **What the cores observe** over `n` more ticks, having observed `o`, sent HBM the keys `ks` and the rest the bits
`v` (all newest first). -/
def observe (C : Cores N b h S) (βm βs : ℕ) (st : Bits S) (hbm : Hbm N b) (rest : Rest N b h) (E : Store N b)
    (O : Oracle N b h) : ℕ → List (Obs h) → List ℕ → List Bool → List (Obs h)
  | 0, o, _, _ => o
  | n + 1, o, ks, v =>
    observe C βm βs st hbm rest E O n
      (((List.range βm).reverse.map (hbm E (C.key st o :: ks)),
        (List.range βs).reverse.map (rest E O ((C.out st o).reverse ++ v) o.length), (C.ask st o).map O) :: o)
      (C.key st o :: ks) ((C.out st o).reverse ++ v)

/-! ### What the building blocks do -/

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

/-- **The chip runs the transducer**: its tags are the cores' tags on what they observe. -/
theorem prog_run (C : Cores N b h S) (βm βs : ℕ) (st : Bits S) (hbm : Hbm N b) (rest : Rest N b h) (E : Store N b)
    (O : Oracle N b h) (n : ℕ) (o : List (Obs h)) (ks : List ℕ) (v : List Bool) :
    (prog C βm βs st n o ks).run (world hbm rest E O) v = C.tags st (observe C βm βs st hbm rest E O n o ks v) := by
  induction n generalizing o ks v with
  | zero => rfl
  | succ n ih =>
    simp only [prog, observe]
    rw [(sendAll_eq _ _ _ _).1, (readMem_eq _ _ _ _ _).1, (readSide_eq _ _ _ _ _).1, (askAll_eq _ _ _ _).1, ih]
    simp only [world, mem_memAddr, side_sideAddr]

/-- **The chip's reads**: over `n` ticks it reads `n βm` bits of device memory and `n βs` of side sources, regenerates no
block, and asks at most `n q` queries when the cores ask at most `q` a tick. -/
theorem prog_counts (C : Cores N b h S) (βm βs q : ℕ) (hq : ∀ st o, (C.ask st o).length ≤ q) (st : Bits S)
    (w : IWorld N b h) (n : ℕ) (o : List (Obs h)) (ks : List ℕ) (v : List Bool) :
    (prog C βm βs st n o ks).ingressBits w v = n * βm ∧ (prog C βm βs st n o ks).sideBits w v = n * βs ∧
      (prog C βm βs st n o ks).regens w v = 0 ∧ ((prog C βm βs st n o ks).queries w v).length ≤ n * q := by
  induction n generalizing o ks v with
  | zero => simp [prog, IChip.ingressBits, IChip.sideBits, IChip.regens, IChip.queries]
  | succ n ih =>
    simp only [prog]
    obtain ⟨s1, s2, s3, s4, s5⟩ := sendAll_eq w (C.out st o) (readMem (memAddr (C.key st o :: ks)) βm fun m =>
        readSide (sideAddr o.length) βs fun r =>
          askAll (C.ask st o) fun ys => prog C βm βs st n ((m, r, ys) :: o) (C.key st o :: ks)) v
    rw [s2, s3, s4, s5]
    obtain ⟨-, m2, m3, m4, m5⟩ := readMem_eq w (memAddr (C.key st o :: ks)) βm (fun m =>
        readSide (sideAddr o.length) βs fun r =>
          askAll (C.ask st o) fun ys => prog C βm βs st n ((m, r, ys) :: o) (C.key st o :: ks))
      ((C.out st o).reverse ++ v)
    rw [m2, m3, m4, m5]
    obtain ⟨-, r2, r3, r4, r5⟩ := readSide_eq w (sideAddr o.length) βs (fun r =>
        askAll (C.ask st o) fun ys => prog C βm βs st n
          (((List.range βm).reverse.map fun j => w.mem (memAddr (C.key st o :: ks) j), r, ys) :: o)
          (C.key st o :: ks))
      ((C.out st o).reverse ++ v)
    rw [r2, r3, r4, r5]
    obtain ⟨-, a2, a3, a4, a5⟩ := askAll_eq w (C.ask st o) (fun ys => prog C βm βs st n
          (((List.range βm).reverse.map fun j => w.mem (memAddr (C.key st o :: ks) j),
            (List.range βs).reverse.map fun j => w.side ((C.out st o).reverse ++ v) (sideAddr o.length j), ys) :: o)
          (C.key st o :: ks))
      ((C.out st o).reverse ++ v)
    rw [a2, a3, a4, a5]
    obtain ⟨i1, i2, i3, i4⟩ := ih (((List.range βm).reverse.map fun j => w.mem (memAddr (C.key st o :: ks) j),
        (List.range βs).reverse.map fun j => w.side ((C.out st o).reverse ++ v) (sideAddr o.length j),
        (C.ask st o).map w.oracle) :: o) (C.key st o :: ks) ((C.out st o).reverse ++ v)
    refine ⟨by rw [i1]; ring, by rw [i2]; ring, i3, ?_⟩
    rw [List.length_append]
    have := hq st o
    nlinarith [i4]

/-- **The transducer as an interactive device** over `T` ticks: its state from the store, HBM as device memory, the
rest as side sources, and `prog` from the state, capped at `c` side bits and `q` queries. -/
def device (C : Cores N b h S) (βm βs T : ℕ) (state : Store N b → Bits S) (hbm : Hbm N b) (rest : Rest N b h)
    (c q : ℕ) : InteractiveDevice N b h S c 0 q :=
  InteractiveDevice.ofRuns state (mem hbm) (side rest) fun st => prog C βm βs st T [] []

theorem device_run (C : Cores N b h S) (βm βs qt T : ℕ) (hq : ∀ st o, (C.ask st o).length ≤ qt)
    (state : Store N b → Bits S) (hbm : Hbm N b) (rest : Rest N b h) (E : Store N b) (O : Oracle N b h) :
    (device C βm βs T state hbm rest (T * βs) (T * qt)).tags E O =
        C.tags (state E) (observe C βm βs (state E) hbm rest E O T [] [] []) ∧
      (device C βm βs T state hbm rest (T * βs) (T * qt)).ingress E O = T * βm := by
  obtain ⟨c1, c2, c3, c4⟩ := prog_counts C βm βs qt hq (state E) (world hbm rest E O) T [] [] []
  obtain ⟨ht, hi⟩ := InteractiveDevice.ofRuns_run (c := T * βs) (G := 0) (q := T * qt) state (mem hbm) (side rest)
    (fun st => prog C βm βs st T [] []) E O ⟨c2.le, c3.le, c4⟩
  exact ⟨ht.trans (prog_run C βm βs (state E) hbm rest E O T [] [] []), hi.trans c1⟩

end Ticked

namespace SecurityProofs.Ticked

open Bandwidth.Ticked

variable {N b h S : ℕ}

/-- **B3 for the cores as a tick transducer**, in the capacity form: with HBM's link delivering `βm` bits of its alphabet
a tick and the rest's `βs`, the cores' state at the window's start `S` bits fixed by the store, HBM hearing only the
cores and at most `qt` queries a tick, the probability over the store, the oracle and the sample that the cores' tags
pass while `βm T + S + βs T + m b + λ < k b` is at most `AuditInteractive`'s bound at `c = βs T`. -/
theorem audit (C : Cores N b h S) (βm βs qt T : ℕ) (hq : ∀ st o, (C.ask st o).length ≤ qt)
    (state : Store N b → Bits S) (hbm : Hbm N b) (rest : Rest N b h) (s k m lam : ℕ) :
    pr (fun x : Store N b × Oracle N b h × (Fin s → Fin N) =>
        (∀ i, C.tags (state x.1) (observe C βm βs (state x.1) hbm rest x.1 x.2.1 T [] [] []) (x.2.2 i) =
            x.2.1 (x.2.2 i, x.1 (x.2.2 i))) ∧
          βm * T + S + βs * T + m * b + lam < k * b)
      ≤ (((k - 1 : ℕ) : ℝ≥0∞) / N) ^ s + (N.choose m : ℝ≥0∞) * 2 ^ (T * βs) / 2 ^ (h * m)
        + (((T * qt).choose (k - m) : ℕ) : ℝ≥0∞) / 2 ^ lam := by
  refine (pr_mono fun x hx => ?_).trans
    (AuditInteractive N b h S (T * βs) 0 (T * qt) s k m lam (device C βm βs T state hbm rest (T * βs) (T * qt)))
  obtain ⟨ht, hi⟩ := device_run C βm βs qt T hq state hbm rest x.1 x.2.1
  refine ⟨fun i => ?_, ?_⟩
  · rw [ht]
    exact hx.1 i
  · rw [hi]
    have := hx.2
    nlinarith

/-- **B3 for a tick transducer whose rest ignores the oracle**: `AuditInteractiveRO`'s bound, with no relayed-tag
term. -/
theorem audit_ro (C : Cores N b h S) (βm βs qt T : ℕ) (hq : ∀ st o, (C.ask st o).length ≤ qt)
    (state : Store N b → Bits S) (hbm : Hbm N b) (rest : Rest N b h) (hrest : ∀ E O O', rest E O = rest E O')
    (s k lam : ℕ) :
    pr (fun x : Store N b × Oracle N b h × (Fin s → Fin N) =>
        (∀ i, C.tags (state x.1) (observe C βm βs (state x.1) hbm rest x.1 x.2.1 T [] [] []) (x.2.2 i) =
            x.2.1 (x.2.2 i, x.1 (x.2.2 i))) ∧
          βm * T + S + βs * T + lam < k * b)
      ≤ (((k - 1 : ℕ) : ℝ≥0∞) / N) ^ s + (s : ℝ≥0∞) / 2 ^ h + (((T * qt).choose k : ℕ) : ℝ≥0∞) / 2 ^ lam := by
  have hside : ∀ E O O', (device C βm βs T state hbm rest (T * βs) (T * qt)).side E O =
      (device C βm βs T state hbm rest (T * βs) (T * qt)).side E O' := by
    intro E O O'
    funext v a
    simp only [device, InteractiveDevice.ofRuns, side, hrest E O O']
  refine (pr_mono fun x hx => ?_).trans
    (AuditInteractiveRO N b h S (T * βs) 0 (T * qt) s k lam (device C βm βs T state hbm rest (T * βs) (T * qt)) hside)
  obtain ⟨ht, hi⟩ := device_run C βm βs qt T hq state hbm rest x.1 x.2.1
  refine ⟨fun i => ?_, ?_⟩
  · rw [ht]
    exact hx.1 i
  · rw [hi]
    have := hx.2
    nlinarith

end SecurityProofs.Ticked

end Bandwidth
