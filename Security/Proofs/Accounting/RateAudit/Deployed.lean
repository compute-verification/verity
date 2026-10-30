import Mathlib
import Definitions.Accounting.Guarantees.RateAuditDeployed
import Proofs.Accounting.RateAudit.Session

/-!
# The remote bandwidth audit as deployed: the proof

**Completions as side reads.** A pass device with completions is simulated by an interactive device of
`RatePaidNamedRO`'s class (`PIDevice.toInteractive`): its chip reads a completion's `x` bits from the side sources, at
odd addresses that name the word, the width, the function and the bit, and the side sources answer them from the
store; an original side read at `a` goes to `2a`. The simulating chip keeps count of its link bits and queries and
stops at its bounds, so it is bounded in every world; in the device's own worlds the counts are the device's link bits
plus its completed bits and its queries, within `L + X` and `q` once `X` bounds the completed bits, and there it
returns the same receipts with the same tier reads, sends what the device sends, and rebuilds nothing.
`WorkHardness` and `SquaringCeiling` give `X = b·F/(rounds·c_root)` (`PIDevice.completed_le`).

**A window and a session.** A deployed window maps to a `GrindChain` of the simulated devices, whose union over the
`G` candidates (`GrindChain.pr_unpaid_le`) and `RatePaidNamedRO` bound each window's unpaid words; the session's pooled
draws are causal over the monitoring period (`Pooled.pr_causal_le'`, `Pooled.pr_floor_le`), as in `Session.lean`.

**The readings.** `K = ⌊(1 − d̄)·n·P·N⌋` and `K_w = ⌊(1 − d_w)·P·N⌋` turn the counts into the fractions of `γ`.
-/

namespace Accounting.Rate

open Pous (Bits pr)
open scoped ENNReal
open Bandwidth (pr_mono pr_or_le pr_exists_le)

namespace IChip

variable {N b h ρ : ℕ} {α : Type}

/-- Read `n` side bits at the addresses `addr`, then continue with them. -/
def readSide : (n : ℕ) → (Fin n → ℕ) → (Bits n → IChip N b h ρ α) → IChip N b h ρ α
  | 0, _, k => k Fin.elim0
  | n + 1, addr, k => side (addr 0) (fun β => readSide n (fun i => addr i.succ) (fun z => k (Fin.cons β z)))

theorem readSide_run (w : IWorld N b h ρ) (v : List Bool) :
    ∀ (n : ℕ) (addr : Fin n → ℕ) (k : Bits n → IChip N b h ρ α),
      (readSide n addr k).run w v = (k (fun i => w.side v (addr i))).run w v
  | 0, addr, k => by
    simp only [readSide]
    congr 2
    funext i
    exact i.elim0
  | n + 1, addr, k => by
    simp only [readSide, run]
    rw [readSide_run w v n]
    congr 2
    funext i
    refine Fin.cases rfl (fun i => rfl) i

theorem readSide_tierBits {τ : ℕ} (tier : ℕ → Fin τ) (t : Fin τ) (w : IWorld N b h ρ) (v : List Bool) :
    ∀ (n : ℕ) (addr : Fin n → ℕ) (k : Bits n → IChip N b h ρ α),
      (readSide n addr k).tierBits tier t w v = (k (fun i => w.side v (addr i))).tierBits tier t w v
  | 0, addr, k => by
    simp only [readSide]
    congr 2
    funext i
    exact i.elim0
  | n + 1, addr, k => by
    simp only [readSide, tierBits]
    rw [readSide_tierBits tier t w v n]
    congr 2
    funext i
    refine Fin.cases rfl (fun i => rfl) i

theorem readSide_queries (w : IWorld N b h ρ) (v : List Bool) :
    ∀ (n : ℕ) (addr : Fin n → ℕ) (k : Bits n → IChip N b h ρ α),
      (readSide n addr k).queries w v = (k (fun i => w.side v (addr i))).queries w v
  | 0, addr, k => by
    simp only [readSide]
    congr 2
    funext i
    exact i.elim0
  | n + 1, addr, k => by
    simp only [readSide, queries]
    rw [readSide_queries w v n]
    congr 2
    funext i
    refine Fin.cases rfl (fun i => rfl) i

theorem readSide_rebuilds (w : IWorld N b h ρ) (v : List Bool) :
    ∀ (n : ℕ) (addr : Fin n → ℕ) (k : Bits n → IChip N b h ρ α),
      (readSide n addr k).rebuilds w v = (k (fun i => w.side v (addr i))).rebuilds w v
  | 0, addr, k => by
    simp only [readSide]
    congr 2
    funext i
    exact i.elim0
  | n + 1, addr, k => by
    simp only [readSide, rebuilds]
    rw [readSide_rebuilds w v n]
    congr 2
    funext i
    refine Fin.cases rfl (fun i => rfl) i

/-- Reading `n` side bits takes `n` bits over the link, and whatever the continuation takes, in every world. -/
theorem readSide_linkBits (w : IWorld N b h ρ) (v : List Bool) :
    ∀ (n : ℕ) (addr : Fin n → ℕ) (k : Bits n → IChip N b h ρ α),
      ∃ z, (readSide n addr k).linkBits w v = (k z).linkBits w v + n ∧
        (readSide n addr k).queries w v = (k z).queries w v
  | 0, addr, k => ⟨Fin.elim0, by simp [readSide]⟩
  | n + 1, addr, k => by
    obtain ⟨z, hz, hq⟩ := readSide_linkBits w v n (fun i => addr i.succ) (fun z => k (Fin.cons (w.side v (addr 0)) z))
    exact ⟨Fin.cons (w.side v (addr 0)) z, by simp only [readSide, linkBits]; omega,
      by simp only [readSide, queries]; exact hq⟩

end IChip

namespace PIChip

variable {N b h ρ : ℕ} {α : Type}

/-- The address of bit `i` of the completion of `x` bits `g` of word `j`. -/
noncomputable def code (j : Fin N) (x : ℕ) (g : Bits b → Bits x) (i : Fin x) : ℕ :=
  Nat.pair j.1 (Nat.pair x (Nat.pair (Fintype.equivFin (Bits b → Bits x) g).1 i.1))

/-- Bit `i` of the `x`-bit function numbered `gi` of word `j` of the store, or `false` if they name none. -/
noncomputable def decodeAt (E : Store N b) (j x gi i : ℕ) : Bool :=
  if hj : j < N then
    if hg : gi < Fintype.card (Bits b → Bits x) then
      if hi : i < x then ((Fintype.equivFin (Bits b → Bits x)).symm ⟨gi, hg⟩ (E ⟨j, hj⟩)) ⟨i, hi⟩ else false
    else false
  else false

/-- The bit an address names, read from the store: bit `i` of `g (E_j)` at `code j x g i`. -/
noncomputable def decode (E : Store N b) (c : ℕ) : Bool :=
  decodeAt E c.unpair.1 c.unpair.2.unpair.1 c.unpair.2.unpair.2.unpair.1 c.unpair.2.unpair.2.unpair.2

theorem decode_code (E : Store N b) (j : Fin N) (x : ℕ) (g : Bits b → Bits x) (i : Fin x) :
    decode E (code j x g i) = g (E j) i := by
  simp only [decode, code, Nat.unpair_pair]
  unfold decodeAt
  rw [dite_eq_left_of_eq_true (eq_true j.2), dite_eq_left_of_eq_true (eq_true (Fintype.equivFin (Bits b → Bits x) g).2),
    dite_eq_left_of_eq_true (eq_true i.2)]
  rw [Fin.eta, Equiv.symm_apply_apply]

/-- Side sources that also answer completions: an original read at `a` is at `2a`, and `code j x g i` at
`2·code j x g i + 1`. -/
noncomputable def sideOf (E : Store N b) (f : List Bool → ℕ → Bool) (v : List Bool) (a : ℕ) : Bool :=
  if a % 2 = 0 then f v (a / 2) else decode E (a / 2)

/-- The simulating chip: link bits within `ℓ` and queries within `m`, a completion read from the side sources. -/
noncomputable def sim (dflt : α) : ℕ → ℕ → PIChip N b h ρ α → IChip N b h ρ α
  | _, _, done a => .done a
  | ℓ, m, tag y k => if m = 0 then .done dflt else .tag y (fun t => sim dflt ℓ (m - 1) (k t))
  | ℓ, m, ingress a k => .ingress a (fun β => sim dflt ℓ m (k β))
  | ℓ, m, side a k => if ℓ = 0 then .done dflt else .side (2 * a) (fun β => sim dflt (ℓ - 1) m (k β))
  | ℓ, m, stream i k => if ℓ = 0 then .done dflt else .stream i (fun β => sim dflt (ℓ - 1) m (k β))
  | ℓ, m, complete j x _ g k =>
    if x ≤ ℓ then IChip.readSide x (fun i => 2 * code j x g i + 1) (fun z => sim dflt (ℓ - x) m (k z))
    else .done dflt
  | ℓ, m, send β p => .send β (sim dflt ℓ m p)

/-- The simulating chip is within its counts in every world. -/
theorem sim_bounded (dflt : α) (w : IWorld N b h ρ) :
    ∀ (c : PIChip N b h ρ α) (ℓ m : ℕ) (v : List Bool),
      (sim dflt ℓ m c).linkBits w v ≤ ℓ ∧ ((sim dflt ℓ m c).queries w v).length ≤ m := by
  intro c
  induction c with
  | done a => intro ℓ m v; simp [sim, IChip.linkBits, IChip.queries]
  | tag y k ih =>
    intro ℓ m v
    by_cases hm : m = 0
    · simp [sim, hm, IChip.linkBits, IChip.queries]
    · simp only [sim, hm, ↓reduceIte, IChip.linkBits, IChip.queries, List.length_cons]
      have := ih (w.oracle y) ℓ (m - 1) v
      omega
  | ingress a k ih => intro ℓ m v; simpa [sim, IChip.linkBits, IChip.queries] using ih (w.mem a) ℓ m v
  | side a k ih =>
    intro ℓ m v
    by_cases hl : ℓ = 0
    · simp [sim, hl, IChip.linkBits, IChip.queries]
    · simp only [sim, hl, ↓reduceIte, IChip.linkBits, IChip.queries]
      have := ih (w.side v (2 * a)) (ℓ - 1) m v
      omega
  | stream i k ih =>
    intro ℓ m v
    by_cases hl : ℓ = 0
    · simp [sim, hl, IChip.linkBits, IChip.queries]
    · simp only [sim, hl, ↓reduceIte, IChip.linkBits, IChip.queries]
      have := ih (w.stream i) (ℓ - 1) m v
      omega
  | complete j x hx g k ih =>
    intro ℓ m v
    by_cases hl : x ≤ ℓ
    · simp only [sim, hl, ↓reduceIte]
      obtain ⟨z, hz, hq⟩ := IChip.readSide_linkBits w v x (fun i => 2 * code j x g i + 1)
        (fun z => sim dflt (ℓ - x) m (k z))
      have := ih z (ℓ - x) m v
      rw [hz, hq]
      omega
    · simp [sim, hl, IChip.linkBits, IChip.queries]
  | send β p ih => intro ℓ m v; simpa [sim, IChip.linkBits, IChip.queries] using ih ℓ m (β :: v)

/-- The world the simulating chip runs in, for the device's world `w`. -/
noncomputable def simWorld (w : IWorld N b h ρ) : IWorld N b h ρ :=
  { w with side := sideOf w.store w.side }

@[simp] theorem simWorld_oracle (w : IWorld N b h ρ) : (simWorld w).oracle = w.oracle := rfl
@[simp] theorem simWorld_mem (w : IWorld N b h ρ) : (simWorld w).mem = w.mem := rfl
@[simp] theorem simWorld_stream (w : IWorld N b h ρ) : (simWorld w).stream = w.stream := rfl
@[simp] theorem simWorld_store (w : IWorld N b h ρ) : (simWorld w).store = w.store := rfl

theorem simWorld_side_even (w : IWorld N b h ρ) (v : List Bool) (a : ℕ) :
    (simWorld w).side v (2 * a) = w.side v a := by
  simp [simWorld, sideOf]

theorem simWorld_side_code (w : IWorld N b h ρ) (v : List Bool) (j : Fin N) (x : ℕ) (g : Bits b → Bits x) :
    (fun i => (simWorld w).side v (2 * code j x g i + 1)) = g (w.store j) := by
  funext i
  simp only [simWorld, sideOf]
  have h2 : ((2 * code j x g i + 1) % 2 = 0) = False := by simp only [eq_iff_iff, iff_false]; omega
  simp only [h2, ↓reduceIte, show (2 * code j x g i + 1) / 2 = code j x g i by omega, decode_code]

/-- Within its counts, the simulating chip returns what the chip does, reads what it reads and queries what it
queries, in the chip's world. -/
theorem sim_eq (dflt : α) {τ : ℕ} (tier : ℕ → Fin τ) (w : IWorld N b h ρ) :
    ∀ (c : PIChip N b h ρ α) (ℓ m : ℕ) (v : List Bool),
      c.linkBits w v + c.completed w v ≤ ℓ → (c.queries w v).length ≤ m →
      (sim dflt ℓ m c).run (simWorld w) v = c.run w v ∧
        (∀ t, (sim dflt ℓ m c).tierBits tier t (simWorld w) v = c.tierBits tier t w v) ∧
        (sim dflt ℓ m c).queries (simWorld w) v = c.queries w v ∧
        (sim dflt ℓ m c).rebuilds (simWorld w) v = 0 := by
  intro c
  induction c with
  | done a => intro ℓ m v _ _; simp [sim, run, IChip.run, IChip.tierBits, tierBits, IChip.queries, queries,
      IChip.rebuilds]
  | tag y k ih =>
    intro ℓ m v hl hq
    simp only [queries, List.length_cons, linkBits, completed] at hq hl
    have hm : m ≠ 0 := by omega
    obtain ⟨h₁, h₂, h₃, h₄⟩ := ih (w.oracle y) ℓ (m - 1) v hl (by omega)
    simp only [sim, hm, ↓reduceIte, IChip.run, run, IChip.tierBits, tierBits, IChip.queries, queries,
      IChip.rebuilds, simWorld_oracle]
    exact ⟨h₁, h₂, by rw [h₃], h₄⟩
  | ingress a k ih =>
    intro ℓ m v hl hq
    obtain ⟨h₁, h₂, h₃, h₄⟩ := ih (w.mem a) ℓ m v hl hq
    simp only [sim, IChip.run, run, IChip.tierBits, tierBits, IChip.queries, queries, IChip.rebuilds, simWorld_mem]
    exact ⟨h₁, fun t => by rw [h₂ t], h₃, h₄⟩
  | side a k ih =>
    intro ℓ m v hl hq
    simp only [linkBits, completed] at hl
    simp only [queries] at hq
    have hℓ : ℓ ≠ 0 := by omega
    obtain ⟨h₁, h₂, h₃, h₄⟩ := ih (w.side v a) (ℓ - 1) m v (by omega) hq
    simp only [sim, hℓ, ↓reduceIte, IChip.run, run, IChip.tierBits, tierBits, IChip.queries, queries,
      IChip.rebuilds, simWorld_side_even]
    exact ⟨h₁, h₂, h₃, h₄⟩
  | stream i k ih =>
    intro ℓ m v hl hq
    simp only [linkBits, completed] at hl
    simp only [queries] at hq
    have hℓ : ℓ ≠ 0 := by omega
    obtain ⟨h₁, h₂, h₃, h₄⟩ := ih (w.stream i) (ℓ - 1) m v (by omega) hq
    simp only [sim, hℓ, ↓reduceIte, IChip.run, run, IChip.tierBits, tierBits, IChip.queries, queries,
      IChip.rebuilds, simWorld_stream]
    exact ⟨h₁, h₂, h₃, h₄⟩
  | complete j x hx g k ih =>
    intro ℓ m v hl hq
    simp only [linkBits, completed] at hl
    simp only [queries] at hq
    have hxl : x ≤ ℓ := by omega
    obtain ⟨h₁, h₂, h₃, h₄⟩ := ih (g (w.store j)) (ℓ - x) m v (by omega) hq
    simp only [sim, hxl, ↓reduceIte, run, tierBits, queries, IChip.readSide_run, IChip.readSide_tierBits,
      IChip.readSide_queries, IChip.readSide_rebuilds, simWorld_side_code]
    exact ⟨h₁, h₂, h₃, h₄⟩
  | send β p ih =>
    intro ℓ m v hl hq
    simp only [linkBits, completed] at hl
    simp only [queries] at hq
    obtain ⟨h₁, h₂, h₃, h₄⟩ := ih ℓ m (β :: v) hl hq
    simp only [sim, IChip.run, run, IChip.tierBits, tierBits, IChip.queries, queries, IChip.rebuilds]
    exact ⟨h₁, h₂, h₃, h₄⟩

/-- `WorkHardness`, one completion at a time: the bits completed, at `rounds·c_root` per `b` of them, cost at most the
completions' compute. -/
theorem completed_mul_le {rounds cRoot : ℕ} {cost : ℕ → ℕ} (hW : Assumptions.WorkHardness b rounds cRoot cost)
    (w : IWorld N b h ρ) : ∀ (c : PIChip N b h ρ α) (v : List Bool),
      c.completed w v * (rounds * cRoot) ≤ b * c.work cost w v := by
  intro c
  induction c with
  | done a => intro v; simp [completed, work]
  | tag y k ih => intro v; exact ih _ v
  | ingress a k ih => intro v; exact ih _ v
  | side a k ih => intro v; exact ih _ v
  | stream i k ih => intro v; exact ih _ v
  | complete j x hx g k ih =>
    intro v
    simp only [completed, work]
    have := hW x hx
    have := ih (g (w.store j)) v
    nlinarith
  | send β p ih => intro v; exact ih (β :: v)

end PIChip

namespace PIDevice

variable {N b h ρ S L q : ℕ}

/-- The interactive device that simulates a pass device, its link carrying `L + X` bits: its side sources also answer
the completed bits from the store. -/
noncomputable def toInteractive (A : PIDevice N b h ρ S L q) (X : ℕ) : InteractiveDevice N b h ρ S (L + X) q where
  state := A.state
  mem := A.mem
  side E R O := PIChip.sideOf E (A.side E R O)
  chip st := PIChip.sim (fun _ _ => false) (L + X) q (A.chip st)
  bounded st w := PIChip.sim_bounded _ w (A.chip st) (L + X) q []

theorem toInteractive_world (A : PIDevice N b h ρ S L q) (X : ℕ) (E : Store N b) (R : Bits ρ) (O : Oracle N b h ρ) :
    (A.toInteractive X).world E R O = PIChip.simWorld (A.world E R O) := rfl

theorem toInteractive_side (A : PIDevice N b h ρ S L q) (X : ℕ) (hA : ∀ E R O O', A.side E R O = A.side E R O') :
    ∀ E R O O', (A.toInteractive X).side E R O = (A.toInteractive X).side E R O' := fun E R O O' => by
  simp only [toInteractive, hA E R O O']

/-- When `X` covers the bits completed, the simulating device commits the device's receipts with its tier reads, and
rebuilds nothing. -/
theorem toInteractive_eq {τ : ℕ} (A : PIDevice N b h ρ S L q) (X : ℕ) (tier : ℕ → Fin τ) (E : Store N b) (R : Bits ρ)
    (O : Oracle N b h ρ) (hX : A.completed E R O ≤ X) :
    (A.toInteractive X).correct E R O = A.correct E R O ∧
      (∀ t, (A.toInteractive X).tierReads tier t E R O = A.tierReads tier t E R O) ∧
      (A.toInteractive X).rebuilt E R O = 0 := by
  have hb := A.bounded (A.state E) (A.world E R O)
  obtain ⟨h₁, h₂, -, h₄⟩ := PIChip.sim_eq (fun _ _ => false) tier (A.world E R O) (A.chip (A.state E)) (L + X) q []
    (by unfold completed at hX; omega) hb.2
  refine ⟨?_, fun t => h₂ t, h₄⟩
  have ht : (A.toInteractive X).tags E R O = A.tags E R O := h₁
  simp only [InteractiveDevice.correct, PIDevice.correct, ht]

namespace TimeBudget

/-- A pass whose work is split across concurrent devices uses at most their summed rate times the pass duration. -/
theorem work_le (A : PIDevice N b h ρ S L q) (cost : ℕ → ℕ) (k duration : ℕ) (rate : Fin k → ℕ)
    (B : A.TimeBudget cost k duration rate) (st : Pous.Bits S) (w : IWorld N b h ρ) :
    (A.chip st).work cost w [] ≤ (∑ i, rate i) * duration := by
  rw [B.work_eq]
  calc
    ∑ i, B.workAt st w i ≤ ∑ i, rate i * duration := Finset.sum_le_sum fun i _ => B.workAt_le st w i
    _ = (∑ i, rate i) * duration := by rw [Finset.sum_mul]

/-- The explicit time budget implies the per-pass squaring ceiling used by the composed remote guarantee. -/
theorem squaringCeiling (A : PIDevice N b h ρ S L q) (cost : ℕ → ℕ) (k duration : ℕ) (rate : Fin k → ℕ)
    (B : A.TimeBudget cost k duration rate) :
    Assumptions.SquaringCeiling cost ((∑ i, rate i) * duration) A :=
  fun st w => work_le A cost k duration rate B st w

end TimeBudget

/-- `WorkHardness` and the squaring ceiling: a pass completes at most `b·F/(rounds·c_root)` bits. -/
theorem completed_le (A : PIDevice N b h ρ S L q) {rounds cRoot F : ℕ} {cost : ℕ → ℕ} (hrc : 0 < rounds * cRoot)
    (hW : Assumptions.WorkHardness b rounds cRoot cost) (hF : Assumptions.SquaringCeiling cost F A) (E : Store N b)
    (R : Bits ρ) (O : Oracle N b h ρ) : A.completed E R O ≤ Deployed.rebuildBits b rounds cRoot F := by
  unfold Deployed.rebuildBits
  rw [Nat.le_div_iff_mul_le hrc]
  calc A.completed E R O * (rounds * cRoot) ≤ b * A.work cost E R O :=
        PIChip.completed_mul_le hW (A.world E R O) (A.chip (A.state E)) []
    _ ≤ b * F := Nat.mul_le_mul_left b (hF (A.state E) (A.world E R O))

/-- **A pass device's correct receipts are paid for**: `RatePaidNamedRO` at the simulating device, the completed bits
in its link; `toInteractive_eq` carries it to the pass device once `X` covers them. -/
theorem pr_unpaid_le {τ : ℕ} (A : PIDevice N b h ρ S L q) (tier : ℕ → Fin τ) (X lam : ℕ)
    (hA : ∀ E R O O', A.side E R O = A.side E R O') :
    pr (fun x : Store N b × Bits ρ × Oracle N b h ρ =>
        ∑ t, (A.toInteractive X).tierReads tier t x.1 x.2.1 x.2.2 + (A.toInteractive X).rebuilt x.1 x.2.1 x.2.2 * b +
            (S + (L - ρ) + X + lam) < ((A.toInteractive X).correct x.1 x.2.1 x.2.2).card * b)
      ≤ (N : ℝ≥0∞) / 2 ^ h + (q.choose 2 : ℝ≥0∞) / 2 ^ h + (2 ^ N : ℝ≥0∞) / 2 ^ lam :=
  (pr_mono (fun _ hx => by omega)).trans
    (SecurityProofs.RatePaidNamedRO N b h ρ S (L + X) q lam τ tier (A.toInteractive X) (toInteractive_side A X hA))

end PIDevice

namespace DeployedWindow

variable {P N b h ρ S L q r s G : ℕ}

/-- The grinding chain of the simulating devices. -/
noncomputable def toGrind (A : DeployedWindow P N b h ρ S L q r s G) (X : ℕ) : GrindChain P N b h ρ S (L + X) q r s G where
  device j η c := (A.device j η c).toInteractive X
  cand := A.cand
  pick := A.pick

/-- A window's correct receipts are its passes' correct receipts, counted together. -/
theorem card_correctAll (A : DeployedWindow P N b h ρ S L q r s G) (E : Store N b)
    (ω : Fin P → PassCoins N b h ρ r s) :
    (A.correctAll E ω).card = ∑ j, ((A.dev j E ω).correct E (ω j).1 (A.oracle j E ω)).card := by
  classical
  unfold correctAll
  rw [Finset.card_filter, Fintype.sum_prod_type]
  refine Finset.sum_congr rfl (fun j _ => ?_)
  rw [Finset.sum_boole]
  simp

/-- **A window's unpaid words**, its draw beside: with `X` covering every pass's completed bits, the probability that
its reads and `P` slacks fall short of its correct receipts' words is at most `P·G·ε`. -/
theorem pr_unpaid_draw_le {τ : ℕ} (A : DeployedWindow P N b h ρ S L q r s G) (tier : ℕ → Fin τ) (X slack d : ℕ)
    (ε : ℝ≥0∞) (hX : ∀ j η c E R O, (A.device j η c).completed E R O ≤ X)
    (hε : ∀ j η c, pr (fun x : Store N b × Bits ρ × Oracle N b h ρ =>
        ∑ t, ((A.device j η c).toInteractive X).tierReads tier t x.1 x.2.1 x.2.2 +
            ((A.device j η c).toInteractive X).rebuilt x.1 x.2.1 x.2.2 * b + slack <
          (((A.device j η c).toInteractive X).correct x.1 x.2.1 x.2.2).card * b) ≤ ε) :
    pr (fun y : Store N b × WindowCoins P N b h ρ r s d =>
        A.reads tier y.1 y.2.1 + P * slack < (A.correctAll y.1 y.2.1).card * b) ≤ (P : ℝ≥0∞) * (G * ε) := by
  classical
  let e : Store N b × WindowCoins P N b h ρ r s d ≃
      (Store N b × (Fin P → PassCoins N b h ρ r s)) × (Fin d → Fin P × Fin N) :=
    (Equiv.prodAssoc _ _ _).symm
  rw [pr_equiv e]
  refine pr_le_of_fiber _ _ (fun D => ?_)
  refine (pr_mono (fun x hx => ?_)).trans ((A.toGrind X).pr_unpaid_le tier slack ε hε)
  have hx' : A.reads tier x.1 x.2 + P * slack < (A.correctAll x.1 x.2).card * b := hx
  have heq := fun j => (A.dev j x.1 x.2).toInteractive_eq X tier x.1 (x.2 j).1 (A.oracle j x.1 x.2)
    (hX j _ _ x.1 (x.2 j).1 (A.oracle j x.1 x.2))
  show ∑ j, (∑ t, ((A.dev j x.1 x.2).toInteractive X).tierReads tier t x.1 (x.2 j).1 (A.oracle j x.1 x.2) +
      ((A.dev j x.1 x.2).toInteractive X).rebuilt x.1 (x.2 j).1 (A.oracle j x.1 x.2) * b) + P * slack <
    ∑ j, (((A.dev j x.1 x.2).toInteractive X).correct x.1 (x.2 j).1 (A.oracle j x.1 x.2)).card * b
  simp only [fun j => (heq j).1, fun j t => (heq j).2.1 t, fun j => (heq j).2.2, zero_mul, add_zero]
  rw [card_correctAll, Finset.sum_mul] at hx'
  exact hx'

end DeployedWindow

namespace DeployedSession

variable {n P N b h ρ S L q r s d G : ℕ}

/-- **Sequencing over windows**: an event of window `w`'s prover and its coins has at most its bound for the provers the
earlier windows' coins pick. -/
theorem pr_window_le (A : DeployedSession n P N b h ρ S L q r s d G) (w : Fin n)
    (Q : DeployedWindow P N b h ρ S L q r s G → Store N b × WindowCoins P N b h ρ r s d → Prop) (ε : ℝ≥0∞)
    (hQ : ∀ η, pr (Q (A.window w η)) ≤ ε) :
    pr (fun x : Store N b × (Fin n → WindowCoins P N b h ρ r s d) => Q (A.win w x.2) (x.1, x.2 w)) ≤ ε := by
  classical
  let e : Store N b × (Fin n → WindowCoins P N b h ρ r s d) ≃
      (Store N b × WindowCoins P N b h ρ r s d) × ({v : Fin n // v ≠ w} → WindowCoins P N b h ρ r s d) :=
    ((Equiv.refl _).prodCongr (Equiv.funSplitAt w _)).trans (Equiv.prodAssoc _ _ _).symm
  rw [pr_equiv e]
  refine pr_le_of_fiber _ _ (fun ω' => ?_)
  let η : {v : Fin n // v < w} → WindowCoins P N b h ρ r s d := fun v => ω' ⟨v.1, v.2.ne⟩
  refine (pr_mono (fun y hy => ?_)).trans (hQ η)
  have hω : ((Equiv.funSplitAt w (WindowCoins P N b h ρ r s d)).symm (y.2, ω')) w = y.2 := by simp
  have hη : (fun v : {v : Fin n // v < w} =>
      ((Equiv.funSplitAt w (WindowCoins P N b h ρ r s d)).symm (y.2, ω')) v.1) = η := by
    funext v
    simp [η, v.2.ne]
  have hc : A.win w ((Equiv.funSplitAt w (WindowCoins P N b h ρ r s d)).symm (y.2, ω')) = A.window w η := by
    simp only [DeployedSession.win]
    rw [hη]
  simpa [e, hω, hc] using hy

/-- A session's draws over one store as a causal family over the monitoring period (`Rate.Causal`): any event of the
openings and correct receipts that the family's own bound covers has at most that bound, on any store `store ω` drawn
before the session's coins. -/
theorem pr_draws_map_of_causal {Ω : Type} [Fintype Ω] (store : Ω → Store N b)
    (A : DeployedSession n P N b h ρ S L q r s d G) (Ev : Store N b →
      (Fin n → WindowCoins P N b h ρ r s d) → Prop) (B : ℝ≥0∞)
    (hB : ∀ (X : Fin n → Type) [∀ w, Fintype (X w)]
      (Gs : (w : Fin n) → ((v : Fin n) → X v × (Fin d → Fin (P * N))) → Finset (Fin (P * N))),
      Causal (s := fun _ => d) Gs → ∀ (f : ((v : Fin n) → X v × (Fin d → Fin (P * N))) → Fin n →
        WindowCoins P N b h ρ r s d) (E : Store N b),
        (∀ ω w t, (f ω w).2 t ∈ (A.win w (f ω)).correctAll E (f ω w).1 → (ω w).2 t ∈ Gs w ω) →
        (∀ ω w, (Gs w ω).card = ((A.win w (f ω)).correctAll E (f ω w).1).card) →
        pr (fun ω => Ev E (f ω)) ≤ B) :
    pr (fun x : Ω × (Fin n → WindowCoins P N b h ρ r s d) => Ev (store x.1) x.2) ≤ B := by
  classical
  rw [pr_equiv (Equiv.prodComm _ _)]
  refine pr_le_of_fiber _ _ (fun ω => ?_)
  show pr (fun y => Ev (store ω) y) ≤ B
  generalize store ω = E
  let X : Fin n → Type := fun _ => Fin P → PassCoins N b h ρ r s
  let e : (Fin n → WindowCoins P N b h ρ r s d) ≃ ((w : Fin n) → X w × (Fin d → Fin (P * N))) :=
    Equiv.piCongrRight (fun _ => (Equiv.refl _).prodCongr (Equiv.arrowCongr (Equiv.refl _) finProdFinEquiv))
  rw [pr_equiv e]
  let Gs : (w : Fin n) → ((v : Fin n) → X v × (Fin d → Fin (P * N))) → Finset (Fin (P * N)) :=
    fun w ω => ((A.win w (e.symm ω)).correctAll E (e.symm ω w).1).map finProdFinEquiv.toEmbedding
  have hG : Causal (s := fun _ => d) Gs := by
    intro w ω ω' hlt hw
    have hη : (fun v : {v : Fin n // v < w} => e.symm ω v.1) = (fun v => e.symm ω' v.1) :=
      funext fun v => by simp [e, hlt v.1 v.2]
    have h₁ : (e.symm ω w).1 = (e.symm ω' w).1 := by simp [e, hw]
    simp only [Gs, DeployedSession.win]
    rw [hη, h₁]
  have hmem : ∀ ω w t, (e.symm ω w).2 t ∈ (A.win w (e.symm ω)).correctAll E (e.symm ω w).1 →
      (ω w).2 t ∈ Gs w ω := by
    intro ω w t h
    simp only [Gs, Finset.mem_map_equiv]
    exact h
  have hcard : ∀ ω w, (Gs w ω).card = ((A.win w (e.symm ω)).correctAll E (e.symm ω w).1).card :=
    fun ω w => Finset.card_map _
  exact hB X Gs hG e.symm E hmem hcard

/-- `pr_draws_map_of_causal` on a uniform store. -/
theorem pr_draws_of_causal (A : DeployedSession n P N b h ρ S L q r s d G) (Ev : Store N b →
      (Fin n → WindowCoins P N b h ρ r s d) → Prop) (B : ℝ≥0∞)
    (hB : ∀ (X : Fin n → Type) [∀ w, Fintype (X w)]
      (Gs : (w : Fin n) → ((v : Fin n) → X v × (Fin d → Fin (P * N))) → Finset (Fin (P * N))),
      Causal (s := fun _ => d) Gs → ∀ (f : ((v : Fin n) → X v × (Fin d → Fin (P * N))) → Fin n →
        WindowCoins P N b h ρ r s d) (E : Store N b),
        (∀ ω w t, (f ω w).2 t ∈ (A.win w (f ω)).correctAll E (f ω w).1 → (ω w).2 t ∈ Gs w ω) →
        (∀ ω w, (Gs w ω).card = ((A.win w (f ω)).correctAll E (f ω w).1).card) →
        pr (fun ω => Ev E (f ω)) ≤ B) :
    pr (fun x : Store N b × (Fin n → WindowCoins P N b h ρ r s d) => Ev x.1 x.2) ≤ B :=
  pr_draws_map_of_causal id A Ev B hB

/-- **A session's pooled draws over the monitoring period**: every window's openings land on correct receipts while the
session holds at most `K`. -/
theorem pr_draws_pooled_le_map {Ω : Type} [Fintype Ω] (store : Ω → Store N b)
    (A : DeployedSession n P N b h ρ S L q r s d G) (K : ℕ) :
    pr (fun x : Ω × (Fin n → WindowCoins P N b h ρ r s d) =>
        (∀ w t, (x.2 w).2 t ∈ (A.win w x.2).correctAll (store x.1) (x.2 w).1) ∧
          ∑ w, ((A.win w x.2).correctAll (store x.1) (x.2 w).1).card ≤ K)
      ≤ ((K : ℝ≥0∞) / (n * P * N)) ^ (d * n) := by
  refine pr_draws_map_of_causal store A (fun E y => (∀ w t, (y w).2 t ∈ (A.win w y).correctAll E (y w).1) ∧
    ∑ w, ((A.win w y).correctAll E (y w).1).card ≤ K) _ (fun X _ Gs hG f E hmem hcard => ?_)
  refine (pr_mono (fun ω hω => ?_)).trans
    ((Pooled.pr_causal_le' n (P * N) (d * K) X (fun _ => d) Gs hG).trans (le_of_eq ?_))
  · obtain ⟨hall, hK⟩ := hω
    refine ⟨fun w t => hmem ω w t (hall w t), ?_⟩
    show ∑ i, d * (Gs i ω).card ≤ d * K
    simp only [hcard]
    rw [← Finset.mul_sum]
    exact Nat.mul_le_mul_left d hK
  · simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, smul_eq_mul]
    rcases Nat.eq_zero_or_pos d with rfl | hd
    · simp
    rw [mul_comm n d]
    congr 1
    have hd' : (d : ℝ≥0∞) ≠ 0 := by exact_mod_cast hd.ne'
    push_cast
    rw [show (P : ℝ≥0∞) * N * (d * n) = d * (n * P * N) by ring]
    exact ENNReal.mul_div_mul_left _ _ hd' (ENNReal.natCast_ne_top d)

/-- `pr_draws_pooled_le_map` on a uniform store. -/
theorem pr_draws_pooled_le (A : DeployedSession n P N b h ρ S L q r s d G) (K : ℕ) :
    pr (fun x : Store N b × (Fin n → WindowCoins P N b h ρ r s d) =>
        (∀ w t, (x.2 w).2 t ∈ (A.win w x.2).correctAll x.1 (x.2 w).1) ∧
          ∑ w, ((A.win w x.2).correctAll x.1 (x.2 w).1).card ≤ K)
      ≤ ((K : ℝ≥0∞) / (n * P * N)) ^ (d * n) :=
  pr_draws_pooled_le_map id A K

/-- **A session's pooled draws in every window**: every window's openings land on correct receipts while some window
holds at most `K_w`. -/
theorem pr_draws_window_le_map {Ω : Type} [Fintype Ω] (store : Ω → Store N b)
    (A : DeployedSession n P N b h ρ S L q r s d G) (Kw : ℕ) :
    pr (fun x : Ω × (Fin n → WindowCoins P N b h ρ r s d) =>
        (∀ w t, (x.2 w).2 t ∈ (A.win w x.2).correctAll (store x.1) (x.2 w).1) ∧
          ∃ w, ((A.win w x.2).correctAll (store x.1) (x.2 w).1).card ≤ Kw)
      ≤ ((((n - 1) * (P * N) + Kw : ℕ) : ℝ≥0∞) / (n * P * N)) ^ (d * n) := by
  refine pr_draws_map_of_causal store A (fun E y => (∀ w t, (y w).2 t ∈ (A.win w y).correctAll E (y w).1) ∧
    ∃ w, ((A.win w y).correctAll E (y w).1).card ≤ Kw) _ (fun X _ Gs hG f E hmem hcard => ?_)
  refine (pr_mono (fun ω hω => ?_)).trans ((Pooled.pr_floor_le n (P * N) Kw d X Gs hG).trans (le_of_eq ?_))
  · obtain ⟨hall, w, hw⟩ := hω
    exact ⟨fun w t => hmem ω w t (hall w t), w, (hcard ω w).trans_le hw⟩
  · simp only [Nat.cast_mul, mul_assoc]

/-- `pr_draws_window_le_map` on a uniform store. -/
theorem pr_draws_window_le (A : DeployedSession n P N b h ρ S L q r s d G) (Kw : ℕ) :
    pr (fun x : Store N b × (Fin n → WindowCoins P N b h ρ r s d) =>
        (∀ w t, (x.2 w).2 t ∈ (A.win w x.2).correctAll x.1 (x.2 w).1) ∧
          ∃ w, ((A.win w x.2).correctAll x.1 (x.2 w).1).card ≤ Kw)
      ≤ ((((n - 1) * (P * N) + Kw : ℕ) : ℝ≥0∞) / (n * P * N)) ^ (d * n) :=
  pr_draws_window_le_map id A Kw

/-- **A session's pooled draws**, on any store `store ω` drawn before the session's coins: every window's openings
land on correct receipts while the session holds at most `K` in all or some window at most `K_w`. -/
theorem pr_draws_le_map {Ω : Type} [Fintype Ω] (store : Ω → Store N b)
    (A : DeployedSession n P N b h ρ S L q r s d G) (K Kw : ℕ) :
    pr (fun x : Ω × (Fin n → WindowCoins P N b h ρ r s d) =>
        (∀ w t, (x.2 w).2 t ∈ (A.win w x.2).correctAll (store x.1) (x.2 w).1) ∧
          (∑ w, ((A.win w x.2).correctAll (store x.1) (x.2 w).1).card ≤ K ∨
            ∃ w, ((A.win w x.2).correctAll (store x.1) (x.2 w).1).card ≤ Kw))
      ≤ ((K : ℝ≥0∞) / (n * P * N)) ^ (d * n) +
        ((((n - 1) * (P * N) + Kw : ℕ) : ℝ≥0∞) / (n * P * N)) ^ (d * n) :=
  (pr_mono (fun _ ⟨hall, h⟩ => h.imp (fun hK => ⟨hall, hK⟩) (fun hw => ⟨hall, hw⟩))).trans
    ((pr_or_le _ _).trans (add_le_add (pr_draws_pooled_le_map store A K) (pr_draws_window_le_map store A Kw)))

/-- **A session's pooled draws**: every window's openings land on correct receipts while the session holds at most
`K` in all or some window at most `K_w`. -/
theorem pr_draws_le (A : DeployedSession n P N b h ρ S L q r s d G) (K Kw : ℕ) :
    pr (fun x : Store N b × (Fin n → WindowCoins P N b h ρ r s d) =>
        (∀ w t, (x.2 w).2 t ∈ (A.win w x.2).correctAll x.1 (x.2 w).1) ∧
          (∑ w, ((A.win w x.2).correctAll x.1 (x.2 w).1).card ≤ K ∨
            ∃ w, ((A.win w x.2).correctAll x.1 (x.2 w).1).card ≤ Kw))
      ≤ ((K : ℝ≥0∞) / (n * P * N)) ^ (d * n) +
        ((((n - 1) * (P * N) + Kw : ℕ) : ℝ≥0∞) / (n * P * N)) ^ (d * n) :=
  pr_draws_le_map id A K Kw

/-- A session's unpaid words from a per-window bound: the probability that some window's reads and `P` slacks fall
short of its correct receipts' words is at most `n·ε`. -/
theorem pr_unpaid_le {τ : ℕ} (A : DeployedSession n P N b h ρ S L q r s d G) (tier : ℕ → Fin τ) (slack : ℕ)
    (ε : ℝ≥0∞) (hε : ∀ w η, pr (fun y : Store N b × WindowCoins P N b h ρ r s d =>
        (A.window w η).reads tier y.1 y.2.1 + P * slack < ((A.window w η).correctAll y.1 y.2.1).card * b) ≤ ε) :
    pr (fun x : Store N b × (Fin n → WindowCoins P N b h ρ r s d) =>
        ∃ w, (A.win w x.2).reads tier x.1 (x.2 w).1 + P * slack <
          ((A.win w x.2).correctAll x.1 (x.2 w).1).card * b)
      ≤ (n : ℝ≥0∞) * ε := by
  classical
  calc _ ≤ ∑ w, pr (fun x : Store N b × (Fin n → WindowCoins P N b h ρ r s d) =>
          (A.win w x.2).reads tier x.1 (x.2 w).1 + P * slack <
            ((A.win w x.2).correctAll x.1 (x.2 w).1).card * b) := pr_exists_le _
    _ ≤ ∑ _w : Fin n, ε := Finset.sum_le_sum (fun w _ => pr_window_le A w
          (fun C y => C.reads tier y.1 y.2.1 + P * slack < (C.correctAll y.1 y.2.1).card * b) _ (hε w))
    _ = (n : ℝ≥0∞) * ε := by rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]

/-- **The session's words, in counts, from an unpaid bound**, on any store `store ω` drawn before the session's coins:
every opening passes while the session's reads and `n·P` slacks fall short of `K + 1` words, or some window's reads
and `P` slacks of `K_w + 1`, at most the draws' bound and `ε_u`, a bound on some window's unpaid words. -/
theorem pr_paid_of_unpaid_map {τ : ℕ} {Ω : Type} [Fintype Ω] (store : Ω → Store N b)
    (A : DeployedSession n P N b h ρ S L q r s d G) (tier : ℕ → Fin τ) (slack K Kw : ℕ) (εu : ℝ≥0∞)
    (hu : pr (fun x : Ω × (Fin n → WindowCoins P N b h ρ r s d) =>
        ∃ w, (A.win w x.2).reads tier (store x.1) (x.2 w).1 + P * slack <
          ((A.win w x.2).correctAll (store x.1) (x.2 w).1).card * b) ≤ εu) :
    pr (fun x : Ω × (Fin n → WindowCoins P N b h ρ r s d) =>
        (∀ w t, (x.2 w).2 t ∈ (A.win w x.2).correctAll (store x.1) (x.2 w).1) ∧
          (∑ w, (A.win w x.2).reads tier (store x.1) (x.2 w).1 + n * (P * slack) < (K + 1) * b ∨
            ∃ w, (A.win w x.2).reads tier (store x.1) (x.2 w).1 + P * slack < (Kw + 1) * b))
      ≤ ((K : ℝ≥0∞) / (n * P * N)) ^ (d * n) +
        ((((n - 1) * (P * N) + Kw : ℕ) : ℝ≥0∞) / (n * P * N)) ^ (d * n) + εu := by
  have h₁ := pr_draws_le_map store A K Kw
  refine (pr_mono (fun x hx => ?_)).trans ((pr_or_le _ _).trans (add_le_add h₁ hu))
  obtain ⟨hall, hT | ⟨w, hw⟩⟩ := hx
  · by_cases hD : ∑ w, ((A.win w x.2).correctAll (store x.1) (x.2 w).1).card ≤ K ∨
        ∃ w, ((A.win w x.2).correctAll (store x.1) (x.2 w).1).card ≤ Kw
    · exact Or.inl ⟨hall, hD⟩
    refine Or.inr ?_
    by_contra hno
    push Not at hno hD
    have hle := Finset.sum_le_sum (fun w (_ : w ∈ Finset.univ) => hno w)
    rw [Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ, Fintype.card_fin, smul_eq_mul,
      ← Finset.sum_mul] at hle
    have hK : (K + 1) * b ≤ (∑ w, ((A.win w x.2).correctAll (store x.1) (x.2 w).1).card) * b :=
      Nat.mul_le_mul_right _ hD.1
    omega
  · by_cases hD : ∑ w, ((A.win w x.2).correctAll (store x.1) (x.2 w).1).card ≤ K ∨
        ∃ w, ((A.win w x.2).correctAll (store x.1) (x.2 w).1).card ≤ Kw
    · exact Or.inl ⟨hall, hD⟩
    push Not at hD
    exact Or.inr ⟨w, hw.trans_le (Nat.mul_le_mul_right _ (hD.2 w))⟩

/-- **The session's words, in counts, from a per-window bound**: every opening passes while the session's reads and
`n·P` slacks fall short of `K + 1` words, or some window's reads and `P` slacks of `K_w + 1`, at most the draws' bound
and `n·ε`. -/
theorem pr_paid_of_window {τ : ℕ} (A : DeployedSession n P N b h ρ S L q r s d G) (tier : ℕ → Fin τ) (slack K Kw : ℕ)
    (ε : ℝ≥0∞) (hε : ∀ w η, pr (fun y : Store N b × WindowCoins P N b h ρ r s d =>
        (A.window w η).reads tier y.1 y.2.1 + P * slack < ((A.window w η).correctAll y.1 y.2.1).card * b) ≤ ε) :
    pr (fun x : Store N b × (Fin n → WindowCoins P N b h ρ r s d) =>
        (∀ w t, (x.2 w).2 t ∈ (A.win w x.2).correctAll x.1 (x.2 w).1) ∧
          (∑ w, (A.win w x.2).reads tier x.1 (x.2 w).1 + n * (P * slack) < (K + 1) * b ∨
            ∃ w, (A.win w x.2).reads tier x.1 (x.2 w).1 + P * slack < (Kw + 1) * b))
      ≤ ((K : ℝ≥0∞) / (n * P * N)) ^ (d * n) +
        ((((n - 1) * (P * N) + Kw : ℕ) : ℝ≥0∞) / (n * P * N)) ^ (d * n) + (n : ℝ≥0∞) * ε :=
  pr_paid_of_unpaid_map id A tier slack K Kw _ (pr_unpaid_le A tier slack _ hε)

/-- **The session's words, in counts**: `pr_paid_of_window` at `pr_unpaid_draw_le`'s `P·G·ε` a window. -/
theorem pr_paid_le {τ : ℕ} (A : DeployedSession n P N b h ρ S L q r s d G) (tier : ℕ → Fin τ) (X slack K Kw : ℕ)
    (ε : ℝ≥0∞) (hX : ∀ w η j ι c E R O, ((A.window w η).device j ι c).completed E R O ≤ X)
    (hε : ∀ w η j ι c, pr (fun x : Store N b × Bits ρ × Oracle N b h ρ =>
        ∑ t, (((A.window w η).device j ι c).toInteractive X).tierReads tier t x.1 x.2.1 x.2.2 +
            (((A.window w η).device j ι c).toInteractive X).rebuilt x.1 x.2.1 x.2.2 * b + slack <
          ((((A.window w η).device j ι c).toInteractive X).correct x.1 x.2.1 x.2.2).card * b) ≤ ε) :
    pr (fun x : Store N b × (Fin n → WindowCoins P N b h ρ r s d) =>
        (∀ w t, (x.2 w).2 t ∈ (A.win w x.2).correctAll x.1 (x.2 w).1) ∧
          (∑ w, (A.win w x.2).reads tier x.1 (x.2 w).1 + n * (P * slack) < (K + 1) * b ∨
            ∃ w, (A.win w x.2).reads tier x.1 (x.2 w).1 + P * slack < (Kw + 1) * b))
      ≤ ((K : ℝ≥0∞) / (n * P * N)) ^ (d * n) +
        ((((n - 1) * (P * N) + Kw : ℕ) : ℝ≥0∞) / (n * P * N)) ^ (d * n) + (n : ℝ≥0∞) * (P * (G * ε)) :=
  pr_paid_of_window A tier slack K Kw _ (fun w η =>
    (A.window w η).pr_unpaid_draw_le tier X slack d ε (hX w η) (hε w η))

end DeployedSession

end Accounting.Rate

namespace Accounting.Rate.Deployed

open scoped ENNReal

/-- The named terms of a pass's `γ` sum to its slack in bits, a fraction of its `N·b`. -/
theorem gammaPass_mul (N b ρ S L lam rounds cRoot F : ℕ) (hN : 0 < N) (hb : 0 < b) :
    gammaPass N b ρ S L lam rounds cRoot F * (N * b) =
      ((S + (L - ρ) + rebuildBits b rounds cRoot F + lam : ℕ) : ℝ) := by
  have hNb : (N : ℝ) * b ≠ 0 := by positivity
  unfold gammaPass muChip muLink muRebuild muLambda
  rw [← add_div, ← add_div, ← add_div, div_mul_cancel₀ _ hNb]
  push_cast
  ring

/-- A count at most `t·c` of `c` items, as a fraction in `ℝ≥0∞`. -/
theorem div_le_ofReal {a c : ℕ} (hc : 0 < c) {t : ℝ} (h : (a : ℝ) ≤ t * c) :
    (a : ℝ≥0∞) / c ≤ ENNReal.ofReal t := by
  have hc' : (0 : ℝ) < c := by exact_mod_cast hc
  rw [← ENNReal.ofReal_natCast a, ← ENNReal.ofReal_natCast c, ← ENNReal.ofReal_div_of_pos hc']
  exact ENNReal.ofReal_le_ofReal ((div_le_iff₀ hc').2 h)

/-- The monitoring period's draw term at `K = ⌊(1 − d̄)·M⌋`: at most `(1 − d̄)^e`. -/
theorem pooled_le (dbar : ℝ) (M e : ℕ) (hM : 0 < M) :
    ((⌊(1 - dbar) * M⌋₊ : ℝ≥0∞) / M) ^ e ≤ ENNReal.ofReal (1 - dbar) ^ e := by
  refine pow_le_pow_left₀ zero_le ?_ e
  by_cases h : 0 ≤ 1 - dbar
  · exact div_le_ofReal hM (Nat.floor_le (by positivity))
  · rw [Nat.floor_eq_zero.2 (by nlinarith [show (0 : ℝ) < M by exact_mod_cast hM])]
    simp

/-- The window's draw term at `K_w = ⌊(1 − d_w)·P·N⌋`: at most `(1 − d_w/n)^e`. -/
theorem window_le (dw : ℝ) (n M e : ℕ) (hn : 0 < n) (hM : 0 < M) (hdw : dw ≤ 1) :
    ((((n - 1) * M + ⌊(1 - dw) * M⌋₊ : ℕ) : ℝ≥0∞) / (n * M)) ^ e ≤ ENNReal.ofReal (1 - dw / n) ^ e := by
  refine pow_le_pow_left₀ zero_le ?_ e
  have hM' : (0 : ℝ) < M := by exact_mod_cast hM
  have hn' : (0 : ℝ) < n := by exact_mod_cast hn
  have hfl : (⌊(1 - dw) * M⌋₊ : ℝ) ≤ (1 - dw) * M := Nat.floor_le (by nlinarith)
  have := div_le_ofReal (a := (n - 1) * M + ⌊(1 - dw) * M⌋₊) (c := n * M) (by positivity) (t := 1 - dw / n) (by
    push_cast [Nat.cast_sub (show 1 ≤ n by omega)]
    rw [show (1 - dw / n) * (n * M) = (n - 1) * M + (1 - dw) * M by field_simp; ring]
    linarith)
  simpa using this

/-- A reading's shortfall as a count: reads short of `(1 − (t + γ))` of `c` passes' `N·b` bits, with `γ·N·b` the
pass's slack, are short of `⌊(1 − t)·c·N⌋ + 1` words with the `c` slacks. -/
theorem count_lt {R t g c Nn b sl : ℝ} (hb : 0 < b) (hg : g * (Nn * b) = sl)
    (h : R < (1 - (t + g)) * (c * Nn * b)) : R + c * sl < (⌊(1 - t) * (c * Nn)⌋₊ + 1) * b := by
  have e : (1 - (t + g)) * (c * Nn * b) = (1 - t) * (c * Nn) * b - c * (g * (Nn * b)) := by ring
  rw [e, hg] at h
  have := mul_lt_mul_of_pos_right (Nat.lt_floor_add_one ((1 - t) * (c * Nn))) hb
  linarith

/-- `e^(−x) ≤ 2^-40` once `x ≥ 40·0.6931471808 > 40·log 2`. -/
theorem exp_neg_le (x : ℝ) (hx : 40 * 0.6931471808 ≤ x) : Real.exp (-x) ≤ 1 / 2 ^ 40 := by
  have h2 := Real.log_two_lt_d9
  have e : Real.exp (40 * Real.log 2) = 2 ^ 40 := by
    rw [show (40 : ℝ) = ((40 : ℕ) : ℝ) by norm_num, Real.exp_nat_mul, Real.exp_log two_pos]
  rw [one_div, ← e, ← Real.exp_neg]
  exact Real.exp_le_exp.2 (by linarith)

/-- `(1 − u)^m ≤ e^(−u·m)`. -/
theorem one_sub_pow_le (u : ℝ) (m : ℕ) (hu : u ≤ 1) : (1 - u) ^ m ≤ Real.exp (-(u * m)) := by
  have h := Real.add_one_le_exp (-u)
  calc (1 - u) ^ m ≤ Real.exp (-u) ^ m := pow_le_pow_left₀ (by linarith) (by linarith) m
    _ = Real.exp (-(u * m)) := by rw [← Real.exp_nat_mul]; ring_nf

/-- A real bound as an `ℝ≥0∞` one, at `2^-40`. -/
theorem ofReal_pow_le {x : ℝ} {m : ℕ} (hx : 0 ≤ x) (h : x ^ m ≤ 1 / 2 ^ 40) :
    ENNReal.ofReal x ^ m ≤ 1 / 2 ^ 40 := by
  rw [← ENNReal.ofReal_pow hx]
  have e : (1 : ℝ≥0∞) / 2 ^ 40 = ENNReal.ofReal (1 / 2 ^ 40) := by
    rw [ENNReal.ofReal_div_of_pos (by positivity), ENNReal.ofReal_one, ENNReal.ofReal_pow (by norm_num),
      ENNReal.ofReal_ofNat]
  rw [e]
  exact ENNReal.ofReal_le_ofReal h

/-- The monitoring period's term at the deployed draw: `(1 − 0.0015)^(2773·n) ≤ 2^-40` from `n = 7`. -/
theorem pooled_deployed (n : ℕ) (hn : 7 ≤ n) : ENNReal.ofReal (1 - 0.0015) ^ (2773 * n) ≤ 1 / 2 ^ 40 := by
  refine ofReal_pow_le (by norm_num) ((one_sub_pow_le 0.0015 _ (by norm_num)).trans (exp_neg_le _ ?_))
  have : (7 : ℝ) ≤ n := by exact_mod_cast hn
  push_cast
  nlinarith

/-- The window's term at the deployed draw: `(1 − 0.01/n)^(2773·n) ≤ e^(−27.73) ≤ 2^-40` at every `n ≥ 1`. -/
theorem window_deployed (n : ℕ) (hn : 0 < n) : ENNReal.ofReal (1 - 0.01 / n) ^ (2773 * n) ≤ 1 / 2 ^ 40 := by
  have hn' : (1 : ℝ) ≤ n := by exact_mod_cast hn
  have hu : 0.01 / (n : ℝ) ≤ 0.01 := div_le_self (by norm_num) hn'
  have hu0 : 0 ≤ 0.01 / (n : ℝ) := by positivity
  refine ofReal_pow_le (by linarith) ((one_sub_pow_le _ _ (by linarith)).trans (exp_neg_le _ ?_))
  have : 0.01 / (n : ℝ) * ((2773 * n : ℕ) : ℝ) = 27.73 := by push_cast; field_simp; norm_num
  rw [this]
  norm_num

end Accounting.Rate.Deployed

namespace Accounting.SecurityProofs

open Accounting.Rate
open Pous (pr)
open scoped ENNReal
open Bandwidth (pr_mono)

/-- **The remote bandwidth audit as deployed: an accepted session conserves its words**
(`Accounting.Guarantees.RemoteSessionConservation`). -/
theorem RemoteSessionConservation : Accounting.Guarantees.RemoteSessionConservation := by
  intro n P N b h ρ S L q r s d G lam τ rounds cRoot F cost tier A dbar dw hn hP hN hb hrc hdw hW hF hA
  have hX : ∀ w η j ι c E R O, ((A.window w η).device j ι c).completed E R O ≤ Deployed.rebuildBits b rounds cRoot F :=
    fun w η j ι c E R O => PIDevice.completed_le _ hrc hW (hF w η j ι c) E R O
  have hε := fun w η j ι c =>
    PIDevice.pr_unpaid_le ((A.window w η).device j ι c) tier (Deployed.rebuildBits b rounds cRoot F) lam (hA w η j ι c)
  have main := DeployedSession.pr_paid_le A tier _ _ ⌊(1 - dbar) * ((n : ℝ) * P * N)⌋₊ ⌊(1 - dw) * ((P : ℝ) * N)⌋₊ _
    hX hε
  have hγ := Deployed.gammaPass_mul N b ρ S L lam rounds cRoot F hN hb
  have hb' : (0 : ℝ) < b := by exact_mod_cast hb
  refine (pr_mono (fun x hx => ?_)).trans (main.trans ?_)
  · obtain ⟨hall, hT | ⟨w, hw⟩⟩ := hx
    · refine ⟨hall, Or.inl ?_⟩
      have := Deployed.count_lt (c := (n : ℝ) * P) hb' hγ hT
      rw [← mul_assoc]
      exact_mod_cast this
    · refine ⟨hall, Or.inr ⟨w, ?_⟩⟩
      have := Deployed.count_lt (c := (P : ℝ)) hb' hγ hw
      exact_mod_cast this
  · have e1 : ((n : ℝ) * P * N) = ((n * P * N : ℕ) : ℝ) := by push_cast; ring
    have e2 : ((P : ℝ) * N) = ((P * N : ℕ) : ℝ) := by push_cast; ring
    rw [e1, e2]
    refine add_le_add (add_le_add ?_ ?_) (le_of_eq (by ring))
    · rw [show ((n : ℝ≥0∞) * P * N) = ((n * P * N : ℕ) : ℝ≥0∞) by push_cast; ring]
      exact Deployed.pooled_le dbar (n * P * N) (d * n) (by positivity)
    · rw [show ((n : ℝ≥0∞) * P * N) = (n : ℝ≥0∞) * ((P * N : ℕ) : ℝ≥0∞) by push_cast; ring]
      exact Deployed.window_le dw n (P * N) (d * n) hn (by positivity) hdw

/-- **The remote bandwidth audit at its deployed parameters**
(`Accounting.Guarantees.RemoteSessionConservationDeployed`). -/
theorem RemoteSessionConservationDeployed : Accounting.Guarantees.RemoteSessionConservationDeployed := by
  intro n P N ρ S L q r s G lam τ cRoot F cost tier A hn hP hN hc hq hG hW hF hA
  have h := RemoteSessionConservation n P N 32768 512 ρ S L q r s 2773 G lam τ 3 cRoot F cost tier A 0.0015 0.01
    (by omega) hP hN (by norm_num) (by positivity) (by norm_num) hW hF hA
  refine h.trans (add_le_add ?_ ?_)
  · rw [show (2 : ℝ≥0∞) / 2 ^ 40 = 1 / 2 ^ 40 + 1 / 2 ^ 40 by rw [ENNReal.div_add_div_same]; norm_num]
    exact add_le_add (Deployed.pooled_deployed n hn) (Deployed.window_deployed n (by omega))
  · have hq' : (q.choose 2 : ℝ≥0∞) ≤ (((2 ^ 55 : ℕ).choose 2 : ℕ) : ℝ≥0∞) := by
      exact_mod_cast Nat.choose_le_choose 2 hq
    have hG' : (G : ℝ≥0∞) ≤ 2 ^ 55 := by exact_mod_cast hG
    gcongr

end Accounting.SecurityProofs
