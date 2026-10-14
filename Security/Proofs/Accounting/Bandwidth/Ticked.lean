import Mathlib
import Proofs.Accounting.Bandwidth.Interactive

/-!
# B3 for the chip as a tick-by-tick transducer over the links' signals

The bandwidth audit's model prover (`InteractiveDevice`) from the chip in the window, read tick by tick as v2.4 of
`note:console/20261012T0645Z-draft-actual-circuit-model` describes it: causal from the window's start, the chip's
behaviour is a function of its state then and of what it observes at each tick. The audited location is the chip, and
HBM's only link is to it, so HBM hears only the chip: its reply is a function of the store and of the requests the chip
sent it, and it computes no tag. The rest of the prover may answer as any function of the store, the oracle and every bit
the chip sent it; its link delivers one signal of its alphabet a tick, so a signal is its index there, `βs` bits.

`Ticked.Chip` is such a transducer: at each tick it sends a request to HBM and fixes how many bits HBM's reply has, its
width. `Ticked.observe` is what it observes over `T` ticks, and `Ticked.prog` the chip program that reads each tick's
replies bit by bit: as many bits of device memory as the widths (addressed by the requests so far, `prog_ingress`) and
`βs T` of side sources, with no regeneration (`prog_counts`). Its tags are the transducer's (`prog_run`).

Two forms of B3 follow, as v2.4's B3 row has them:
- **the reads form**, `SecurityProofs.Ticked.audit_reads`: the probability that the tags pass while
  `hbmBits + S + βs T + m b + λ < k b` is at most `AuditInteractive`'s bound at `c = βs T`, where `hbmBits` is the bits
  HBM replied with. It needs passive memory: given the chip's requests, a reply carries only its values, so its width is
  the chip's to fix;
- **the capacity form**, `SecurityProofs.Ticked.audit`, without passive memory: HBM's link delivers one signal of its
  alphabet a tick, `βm` bits as its index, so every tick's reply has width at most `βm`, and the bound holds at
  `βm T + S + βs T + m b + λ < k b`. A width bound per run would not do here: an alphabet's signals can differ only in
  their width-0 wires, so the information a link carries in a tick is its alphabet index, not the width delivered.

`audit_reads_ro` and `audit_ro` are `AuditInteractiveRO`'s bounds when the rest ignores the oracle.
-/

namespace Bandwidth

open Pous (Bits pr)
open scoped ENNReal

namespace Ticked

/-- What the chip observes in a tick: the bits of HBM's reply and of the rest's signal (its index in the link's alphabet)
and the oracle's answers to the tick's queries. -/
abbrev Obs (h : ℕ) := List Bool × List Bool × List (Bits h)

/-- **The chip in the window, tick by tick**: from its state `st` at the window's start and what it observed (newest
first), the request it sends HBM and the width of HBM's reply, the bits it sends the rest and the oracle queries it makes
at a tick, and at the end the tags it commits to. -/
structure Chip (N b h S : ℕ) where
  key : Bits S → List (Obs h) → ℕ
  width : Bits S → List (Obs h) → ℕ
  out : Bits S → List (Obs h) → List Bool
  ask : Bits S → List (Obs h) → List (Fin N × Bits b)
  tags : Bits S → List (Obs h) → Fin N → Bits h

/-- HBM: bit `j` of its reply at a tick, from the store and the requests the chip sent it so far (newest first). -/
abbrev Hbm (N b : ℕ) := Store N b → List ℕ → ℕ → Bool

/-- The rest of the prover: bit `j` of its signal at tick `u`, from the store, the oracle and the bits the chip sent it
so far (newest first). -/
abbrev Rest (N b h : ℕ) := Store N b → Oracle N b h → List Bool → ℕ → ℕ → Bool

/-- The bits HBM replied with over the ticks observed. -/
def hbmBits {h : ℕ} (o : List (Obs h)) : ℕ := (o.map fun x => x.1.length).sum

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

/-- Device memory's address of bit `j` of HBM's reply after the requests `ks`. -/
def memAddr (ks : List ℕ) (j : ℕ) : ℕ := Encodable.encode (ks, j)

/-- The side sources' address of bit `j` of the rest's signal at tick `u`. -/
def sideAddr (u j : ℕ) : ℕ := Encodable.encode (u, j)

/-- **The chip program** for the remaining `n` ticks, having observed `o` and sent HBM the requests `ks`: at each tick
it sends the rest its bits, reads HBM's reply from device memory and the rest's signal from the side sources, and asks
the oracle; at the end it outputs `fin` of what it observed (the transducer's tags, `C.tags st`, for the audit). -/
def prog (C : Chip N b h S) (βs : ℕ) (st : Bits S) (fin : List (Obs h) → α) :
    ℕ → List (Obs h) → List ℕ → IChip N b h α
  | 0, o, _ => IChip.done (fin o)
  | n + 1, o, ks =>
    sendAll (C.out st o) <|
      readMem (memAddr (C.key st o :: ks)) (C.width st o) fun m =>
        readSide (sideAddr o.length) βs fun r =>
          askAll (C.ask st o) fun ys => prog C βs st fin n ((m, r, ys) :: o) (C.key st o :: ks)

/-- HBM as device memory: the bit at `memAddr ks j` is bit `j` of its reply after the requests `ks`. -/
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

/-- **What the chip observes** over `n` more ticks, having observed `o`, sent HBM the requests `ks` and the rest the
bits `v` (all newest first). -/
def observe (C : Chip N b h S) (βs : ℕ) (st : Bits S) (hbm : Hbm N b) (rest : Rest N b h) (E : Store N b)
    (O : Oracle N b h) : ℕ → List (Obs h) → List ℕ → List Bool → List (Obs h)
  | 0, o, _, _ => o
  | n + 1, o, ks, v =>
    observe C βs st hbm rest E O n
      (((List.range (C.width st o)).reverse.map (hbm E (C.key st o :: ks)),
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

theorem world_mem (hbm : Hbm N b) (rest : Rest N b h) (E : Store N b) (O : Oracle N b h) (ks : List ℕ) (j : ℕ) :
    (world hbm rest E O).mem (memAddr ks j) = hbm E ks j :=
  mem_memAddr hbm E ks j

theorem world_side (hbm : Hbm N b) (rest : Rest N b h) (E : Store N b) (O : Oracle N b h) (v : List Bool) (u j : ℕ) :
    (world hbm rest E O).side v (sideAddr u j) = rest E O v u j :=
  side_sideAddr rest E O v u j

/-- **The chip runs the transducer**: its output is `fin` of what it observes. -/
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

/-- **The chip's device-memory reads are HBM's replies**: over `n` ticks it reads from device memory exactly the bits
HBM replies with. -/
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

/-- **The chip's other reads**: over `n` ticks it reads `n βs` bits of side sources, regenerates no block, and asks at
most `n q` queries when it asks at most `q` a tick. -/
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

/-- Every tick's reply has width at most `βm`, so `n` ticks add at most `n βm` bits. -/
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

/-- **The transducer as an interactive device** over `T` ticks: its state from the store, HBM as device memory, the
rest as side sources, and `prog` from the state, capped at `c` side bits and `q` queries. -/
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

/-- **B3 for the chip as a tick transducer, in the reads form** (passive memory): with the rest's link delivering `βs`
bits of its alphabet a tick, the chip's state at the window's start `S` bits fixed by the store, HBM hearing only the
chip and replying at the width the chip fixes, and at most `qt` queries a tick, the probability over the store, the
oracle and the sample that the chip's tags pass while `hbmBits + S + βs T + m b + λ < k b` is at most
`AuditInteractive`'s bound at `c = βs T`. -/
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

/-- **B3 for the chip as a tick transducer, in the capacity form**: without passive memory, HBM's link delivers one
signal of its alphabet a tick, so every reply has width at most `βm`, and the probability that the chip's tags pass while
`βm T + S + βs T + m b + λ < k b` is at most `AuditInteractive`'s bound at `c = βs T`. -/
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

/-- **The reads form when the rest ignores the oracle**: `AuditInteractiveRO`'s bound, with no relayed-tag term. -/
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

/-- **The capacity form when the rest ignores the oracle**. -/
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
