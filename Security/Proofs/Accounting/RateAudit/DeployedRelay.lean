import Mathlib
import Proofs.Accounting.RateAudit.Conservation
import Proofs.Accounting.RateAudit.DeployedRun

/-!
# PoUB as deployed, with side sources that may relay receipts: the proof

**Completions as memory reads.** `Deployed.lean` simulates a pass device by reading its completions over the side
link, which costs nothing while the side sources ignore the oracle. When they may relay receipts, every bit of the link
is a bit a relayed receipt can come from (`card_relay_le`), so the completions go to device memory instead
(`PIDevice.toInteractiveMem`): a completion's `x` bits are read at odd addresses that name the word, the width, the
function and the bit, which the memory answers from the store, and an original read at `a` goes to `2a`, in the tier
of `a`. The link and the side sources are the device's, the receipts and queries are the device's, nothing is rebuilt,
and the tier reads are the device's plus its completed bits.

**A pass, a window, a session.** `RatePaid` bounds each simulated pass at `m` relayed words; a window's unpaid words
are its passes' at each candidate (`GrindChain.pr_unpaid_le`), with the completed bits, at most `X`, paid from the
slack; the session is `DeployedSession.pr_paid_of_window` and the draws, as in `Deployed.lean`.
-/

namespace Accounting.Rate

open Pous (Bits pr)
open scoped ENNReal
open Bandwidth (pr_mono pr_or_le pr_exists_le)

namespace IChip

variable {N b h ρ : ℕ} {α : Type}

/-- Read `n` bits of device memory at the addresses `addr`, then continue with them. -/
def readMem : (n : ℕ) → (Fin n → ℕ) → (Bits n → IChip N b h ρ α) → IChip N b h ρ α
  | 0, _, k => k Fin.elim0
  | n + 1, addr, k => ingress (addr 0) (fun β => readMem n (fun i => addr i.succ) (fun z => k (Fin.cons β z)))

theorem cons_mem (w : IWorld N b h ρ) {n : ℕ} (addr : Fin (n + 1) → ℕ) :
    (Fin.cons (w.mem (addr 0)) (fun i : Fin n => w.mem (addr i.succ)) : Bits (n + 1)) = fun i => w.mem (addr i) :=
  funext fun i => Fin.cases rfl (fun _ => rfl) i

theorem elim0_mem (w : IWorld N b h ρ) (addr : Fin 0 → ℕ) :
    (Fin.elim0 : Bits 0) = fun i => w.mem (addr i) :=
  funext fun i => i.elim0

theorem readMem_run (w : IWorld N b h ρ) (v : List Bool) :
    ∀ (n : ℕ) (addr : Fin n → ℕ) (k : Bits n → IChip N b h ρ α),
      (readMem n addr k).run w v = (k (fun i => w.mem (addr i))).run w v
  | 0, addr, k => by simp only [readMem]; rw [← elim0_mem w addr]
  | n + 1, addr, k => by
    simp only [readMem, run]
    rw [readMem_run w v n, cons_mem]

theorem readMem_linkBits (w : IWorld N b h ρ) (v : List Bool) :
    ∀ (n : ℕ) (addr : Fin n → ℕ) (k : Bits n → IChip N b h ρ α),
      (readMem n addr k).linkBits w v = (k (fun i => w.mem (addr i))).linkBits w v
  | 0, addr, k => by simp only [readMem]; rw [← elim0_mem w addr]
  | n + 1, addr, k => by
    simp only [readMem, linkBits]
    rw [readMem_linkBits w v n, cons_mem]

theorem readMem_queries (w : IWorld N b h ρ) (v : List Bool) :
    ∀ (n : ℕ) (addr : Fin n → ℕ) (k : Bits n → IChip N b h ρ α),
      (readMem n addr k).queries w v = (k (fun i => w.mem (addr i))).queries w v
  | 0, addr, k => by simp only [readMem]; rw [← elim0_mem w addr]
  | n + 1, addr, k => by
    simp only [readMem, queries]
    rw [readMem_queries w v n, cons_mem]

theorem readMem_rebuilds (w : IWorld N b h ρ) (v : List Bool) :
    ∀ (n : ℕ) (addr : Fin n → ℕ) (k : Bits n → IChip N b h ρ α),
      (readMem n addr k).rebuilds w v = (k (fun i => w.mem (addr i))).rebuilds w v
  | 0, addr, k => by simp only [readMem]; rw [← elim0_mem w addr]
  | n + 1, addr, k => by
    simp only [readMem, rebuilds]
    rw [readMem_rebuilds w v n, cons_mem]

/-- Reading `n` bits of device memory reads `n` bits over the tiers, whichever they are. -/
theorem readMem_tierBits {τ : ℕ} (tier : ℕ → Fin τ) (w : IWorld N b h ρ) (v : List Bool) :
    ∀ (n : ℕ) (addr : Fin n → ℕ) (k : Bits n → IChip N b h ρ α),
      ∑ t, (readMem n addr k).tierBits tier t w v = ∑ t, (k (fun i => w.mem (addr i))).tierBits tier t w v + n
  | 0, addr, k => by simp only [readMem, add_zero]; rw [← elim0_mem w addr]
  | n + 1, addr, k => by
    simp only [readMem, tierBits]
    rw [Finset.sum_add_distrib, readMem_tierBits tier w v n, cons_mem]
    simp only [Finset.sum_ite_eq, Finset.mem_univ, ↓reduceIte]
    ring

end IChip

namespace PIChip

variable {N b h ρ : ℕ} {α : Type}

/-- Device memory that also answers completions: an original read at `a` is at `2a`, and `code j x g i` at
`2·code j x g i + 1`. -/
noncomputable def memOf (E : Store N b) (f : ℕ → Bool) (a : ℕ) : Bool :=
  if a % 2 = 0 then f (a / 2) else decode E (a / 2)

/-- The simulating chip with completions read from device memory: link bits within `ℓ` and queries within `m`. -/
noncomputable def simMem (dflt : α) : ℕ → ℕ → PIChip N b h ρ α → IChip N b h ρ α
  | _, _, done a => .done a
  | ℓ, m, tag y k => if m = 0 then .done dflt else .tag y (fun t => simMem dflt ℓ (m - 1) (k t))
  | ℓ, m, ingress a k => .ingress (2 * a) (fun β => simMem dflt ℓ m (k β))
  | ℓ, m, side a k => if ℓ = 0 then .done dflt else .side a (fun β => simMem dflt (ℓ - 1) m (k β))
  | ℓ, m, stream i k => if ℓ = 0 then .done dflt else .stream i (fun β => simMem dflt (ℓ - 1) m (k β))
  | ℓ, m, complete j x _ g k => IChip.readMem x (fun i => 2 * code j x g i + 1) (fun z => simMem dflt ℓ m (k z))
  | ℓ, m, send β p => .send β (simMem dflt ℓ m p)

/-- The simulating chip is within its counts in every world. -/
theorem simMem_bounded (dflt : α) (w : IWorld N b h ρ) :
    ∀ (c : PIChip N b h ρ α) (ℓ m : ℕ) (v : List Bool),
      (simMem dflt ℓ m c).linkBits w v ≤ ℓ ∧ ((simMem dflt ℓ m c).queries w v).length ≤ m := by
  intro c
  induction c with
  | done a => intro ℓ m v; simp [simMem, IChip.linkBits, IChip.queries]
  | tag y k ih =>
    intro ℓ m v
    by_cases hm : m = 0
    · simp [simMem, hm, IChip.linkBits, IChip.queries]
    · simp only [simMem, hm, ↓reduceIte, IChip.linkBits, IChip.queries, List.length_cons]
      have := ih (w.oracle y) ℓ (m - 1) v
      omega
  | ingress a k ih => intro ℓ m v; simpa [simMem, IChip.linkBits, IChip.queries] using ih (w.mem (2 * a)) ℓ m v
  | side a k ih =>
    intro ℓ m v
    by_cases hl : ℓ = 0
    · simp [simMem, hl, IChip.linkBits, IChip.queries]
    · simp only [simMem, hl, ↓reduceIte, IChip.linkBits, IChip.queries]
      have := ih (w.side v a) (ℓ - 1) m v
      omega
  | stream i k ih =>
    intro ℓ m v
    by_cases hl : ℓ = 0
    · simp [simMem, hl, IChip.linkBits, IChip.queries]
    · simp only [simMem, hl, ↓reduceIte, IChip.linkBits, IChip.queries]
      have := ih (w.stream i) (ℓ - 1) m v
      omega
  | complete j x hx g k ih =>
    intro ℓ m v
    simp only [simMem, IChip.readMem_linkBits, IChip.readMem_queries]
    exact ih _ ℓ m v
  | send β p ih => intro ℓ m v; simpa [simMem, IChip.linkBits, IChip.queries] using ih ℓ m (β :: v)

/-- The world the simulating chip runs in, for the device's world `w`. -/
noncomputable def memWorld (w : IWorld N b h ρ) : IWorld N b h ρ :=
  { w with mem := memOf w.store w.mem }

@[simp] theorem memWorld_oracle (w : IWorld N b h ρ) : (memWorld w).oracle = w.oracle := rfl
@[simp] theorem memWorld_side (w : IWorld N b h ρ) : (memWorld w).side = w.side := rfl
@[simp] theorem memWorld_stream (w : IWorld N b h ρ) : (memWorld w).stream = w.stream := rfl
@[simp] theorem memWorld_store (w : IWorld N b h ρ) : (memWorld w).store = w.store := rfl

theorem memWorld_mem_even (w : IWorld N b h ρ) (a : ℕ) : (memWorld w).mem (2 * a) = w.mem a := by
  simp [memWorld, memOf]

theorem memWorld_mem_code (w : IWorld N b h ρ) (j : Fin N) (x : ℕ) (g : Bits b → Bits x) :
    (fun i => (memWorld w).mem (2 * code j x g i + 1)) = g (w.store j) := by
  funext i
  simp only [memWorld, memOf]
  have h2 : ((2 * code j x g i + 1) % 2 = 0) = False := by simp only [eq_iff_iff, iff_false]; omega
  simp only [h2, ↓reduceIte, show (2 * code j x g i + 1) / 2 = code j x g i by omega, decode_code]

/-- Within its counts, the simulating chip returns what the chip does and queries what it queries, in the chip's
world; its reads over the tiers (an address `a` in the tier of `a/2`) are the chip's plus its completed bits, and it
rebuilds nothing. -/
theorem simMem_eq (dflt : α) {τ : ℕ} (tier : ℕ → Fin τ) (w : IWorld N b h ρ) :
    ∀ (c : PIChip N b h ρ α) (ℓ m : ℕ) (v : List Bool),
      c.linkBits w v ≤ ℓ → (c.queries w v).length ≤ m →
      (simMem dflt ℓ m c).run (memWorld w) v = c.run w v ∧
        ∑ t, (simMem dflt ℓ m c).tierBits (fun a => tier (a / 2)) t (memWorld w) v =
          ∑ t, c.tierBits tier t w v + c.completed w v ∧
        (simMem dflt ℓ m c).rebuilds (memWorld w) v = 0 := by
  intro c
  induction c with
  | done a => intro ℓ m v _ _; simp [simMem, run, IChip.run, IChip.tierBits, tierBits, completed, IChip.rebuilds]
  | tag y k ih =>
    intro ℓ m v hl hq
    simp only [queries, List.length_cons, linkBits] at hq hl
    have hm : m ≠ 0 := by omega
    obtain ⟨h₁, h₂, h₃⟩ := ih (w.oracle y) ℓ (m - 1) v hl (by omega)
    simp only [simMem, hm, ↓reduceIte, IChip.run, run, IChip.tierBits, tierBits, completed, IChip.rebuilds,
      memWorld_oracle]
    exact ⟨h₁, h₂, h₃⟩
  | ingress a k ih =>
    intro ℓ m v hl hq
    obtain ⟨h₁, h₂, h₃⟩ := ih (w.mem a) ℓ m v hl hq
    simp only [simMem, IChip.run, run, IChip.tierBits, tierBits, completed, IChip.rebuilds, memWorld_mem_even,
      show 2 * a / 2 = a by omega]
    refine ⟨h₁, ?_, h₃⟩
    rw [Finset.sum_add_distrib, Finset.sum_add_distrib, h₂]
    ring
  | side a k ih =>
    intro ℓ m v hl hq
    simp only [linkBits] at hl
    simp only [queries] at hq
    have hℓ : ℓ ≠ 0 := by omega
    obtain ⟨h₁, h₂, h₃⟩ := ih (w.side v a) (ℓ - 1) m v (by omega) hq
    simp only [simMem, hℓ, ↓reduceIte, IChip.run, run, IChip.tierBits, tierBits, completed, IChip.rebuilds,
      memWorld_side]
    exact ⟨h₁, h₂, h₃⟩
  | stream i k ih =>
    intro ℓ m v hl hq
    simp only [linkBits] at hl
    simp only [queries] at hq
    have hℓ : ℓ ≠ 0 := by omega
    obtain ⟨h₁, h₂, h₃⟩ := ih (w.stream i) (ℓ - 1) m v (by omega) hq
    simp only [simMem, hℓ, ↓reduceIte, IChip.run, run, IChip.tierBits, tierBits, completed, IChip.rebuilds,
      memWorld_stream]
    exact ⟨h₁, h₂, h₃⟩
  | complete j x hx g k ih =>
    intro ℓ m v hl hq
    simp only [linkBits] at hl
    simp only [queries] at hq
    obtain ⟨h₁, h₂, h₃⟩ := ih (g (w.store j)) ℓ m v hl hq
    refine ⟨?_, ?_, ?_⟩
    · simp only [simMem, IChip.readMem_run, memWorld_mem_code, run]
      exact h₁
    · simp only [simMem, tierBits, completed]
      rw [IChip.readMem_tierBits, memWorld_mem_code, h₂]
      ring
    · simp only [simMem, IChip.readMem_rebuilds, memWorld_mem_code]
      exact h₃
  | send β p ih =>
    intro ℓ m v hl hq
    simp only [linkBits] at hl
    simp only [queries] at hq
    obtain ⟨h₁, h₂, h₃⟩ := ih ℓ m (β :: v) hl hq
    simp only [simMem, IChip.run, run, IChip.tierBits, tierBits, completed, IChip.rebuilds]
    exact ⟨h₁, h₂, h₃⟩

end PIChip

namespace PIDevice

variable {N b h ρ S L q : ℕ}

/-- The interactive device that simulates a pass device on its own link: its device memory also answers the completed
bits from the store. -/
noncomputable def toInteractiveMem (A : PIDevice N b h ρ S L q) : InteractiveDevice N b h ρ S L q where
  state := A.state
  mem E := PIChip.memOf E (A.mem E)
  side := A.side
  chip st := PIChip.simMem (fun _ _ => false) L q (A.chip st)
  bounded st w := PIChip.simMem_bounded _ w (A.chip st) L q []

/-- The simulating device commits the device's receipts, reads over the tiers what the device reads and completes, and
rebuilds nothing. -/
theorem toInteractiveMem_eq {τ : ℕ} (A : PIDevice N b h ρ S L q) (tier : ℕ → Fin τ) (E : Store N b) (R : Bits ρ)
    (O : Oracle N b h ρ) :
    A.toInteractiveMem.correct E R O = A.correct E R O ∧
      ∑ t, A.toInteractiveMem.tierReads (fun a => tier (a / 2)) t E R O =
        ∑ t, A.tierReads tier t E R O + A.completed E R O ∧
      A.toInteractiveMem.rebuilt E R O = 0 := by
  have hb := A.bounded (A.state E) (A.world E R O)
  obtain ⟨h₁, h₂, h₃⟩ := PIChip.simMem_eq (fun _ _ => false) tier (A.world E R O) (A.chip (A.state E)) L q []
    hb.1 hb.2
  refine ⟨?_, h₂, h₃⟩
  have ht : A.toInteractiveMem.tags E R O = A.tags E R O := h₁
  simp only [InteractiveDevice.correct, PIDevice.correct, ht]

/-- **A pass device's correct receipts are paid for, with any side sources**: `RatePaid` at the simulating device. -/
theorem pr_unpaid_relay_le {τ : ℕ} (A : PIDevice N b h ρ S L q) (tier : ℕ → Fin τ) (m lam : ℕ) :
    pr (fun x : Store N b × Bits ρ × Oracle N b h ρ =>
        ∑ t, A.toInteractiveMem.tierReads (fun a => tier (a / 2)) t x.1 x.2.1 x.2.2 +
            A.toInteractiveMem.rebuilt x.1 x.2.1 x.2.2 * b + (S + (L - ρ) + lam + m * b) <
          (A.toInteractiveMem.correct x.1 x.2.1 x.2.2).card * b)
      ≤ (N.choose m : ℝ≥0∞) * 2 ^ L / 2 ^ (h * m) + (2 ^ q : ℝ≥0∞) / 2 ^ lam :=
  (pr_mono (fun _ hx => by omega)).trans
    (SecurityProofs.RatePaid N b h ρ S L q m lam τ (fun a => tier (a / 2)) A.toInteractiveMem)

end PIDevice

namespace DeployedWindow

variable {P N b h ρ S L q r s G : ℕ}

/-- The grinding chain of the memory-simulating devices. -/
noncomputable def toGrindMem (A : DeployedWindow P N b h ρ S L q r s G) : GrindChain P N b h ρ S L q r s G where
  device j η c := (A.device j η c).toInteractiveMem
  cand := A.cand
  pick := A.pick

/-- **A window's unpaid words with any side sources**, its draw beside: with `X` covering every pass's completed bits,
the probability that its reads and `P` slacks `X + slack` fall short of its correct receipts' words is at most
`P·G·ε`. -/
theorem pr_unpaid_draw_mem_le {τ : ℕ} (A : DeployedWindow P N b h ρ S L q r s G) (tier : ℕ → Fin τ) (X slack d : ℕ)
    (ε : ℝ≥0∞) (hX : ∀ j η c E R O, (A.device j η c).completed E R O ≤ X)
    (hε : ∀ j η c, pr (fun x : Store N b × Bits ρ × Oracle N b h ρ =>
        ∑ t, (A.device j η c).toInteractiveMem.tierReads (fun a => tier (a / 2)) t x.1 x.2.1 x.2.2 +
            (A.device j η c).toInteractiveMem.rebuilt x.1 x.2.1 x.2.2 * b + slack <
          ((A.device j η c).toInteractiveMem.correct x.1 x.2.1 x.2.2).card * b) ≤ ε) :
    pr (fun y : Store N b × WindowCoins P N b h ρ r s d =>
        A.reads tier y.1 y.2.1 + P * (X + slack) < (A.correctAll y.1 y.2.1).card * b) ≤ (P : ℝ≥0∞) * (G * ε) := by
  classical
  let e : Store N b × WindowCoins P N b h ρ r s d ≃
      (Store N b × (Fin P → PassCoins N b h ρ r s)) × (Fin d → Fin P × Fin N) :=
    (Equiv.prodAssoc _ _ _).symm
  rw [pr_equiv e]
  refine pr_le_of_fiber _ _ (fun D => ?_)
  refine (pr_mono (fun x hx => ?_)).trans (A.toGrindMem.pr_unpaid_le (fun a => tier (a / 2)) slack ε hε)
  have hx' : A.reads tier x.1 x.2 + P * (X + slack) < (A.correctAll x.1 x.2).card * b := hx
  have heq := fun j => (A.dev j x.1 x.2).toInteractiveMem_eq tier x.1 (x.2 j).1 (A.oracle j x.1 x.2)
  show ∑ j, (∑ t, (A.dev j x.1 x.2).toInteractiveMem.tierReads (fun a => tier (a / 2)) t x.1 (x.2 j).1
        (A.oracle j x.1 x.2) +
      (A.dev j x.1 x.2).toInteractiveMem.rebuilt x.1 (x.2 j).1 (A.oracle j x.1 x.2) * b) + P * slack <
    ∑ j, ((A.dev j x.1 x.2).toInteractiveMem.correct x.1 (x.2 j).1 (A.oracle j x.1 x.2)).card * b
  have e1 : ∑ j, (∑ t, (A.dev j x.1 x.2).toInteractiveMem.tierReads (fun a => tier (a / 2)) t x.1 (x.2 j).1
        (A.oracle j x.1 x.2) +
      (A.dev j x.1 x.2).toInteractiveMem.rebuilt x.1 (x.2 j).1 (A.oracle j x.1 x.2) * b) =
      A.reads tier x.1 x.2 + ∑ j, (A.dev j x.1 x.2).completed x.1 (x.2 j).1 (A.oracle j x.1 x.2) := by
    rw [DeployedWindow.reads, ← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl fun j _ => by rw [(heq j).2.1, (heq j).2.2, zero_mul, add_zero]
  have e2 : ∑ j, ((A.dev j x.1 x.2).toInteractiveMem.correct x.1 (x.2 j).1 (A.oracle j x.1 x.2)).card * b =
      (A.correctAll x.1 x.2).card * b := by
    rw [card_correctAll, Finset.sum_mul]
    exact Finset.sum_congr rfl fun j _ => by rw [(heq j).1]
  have hsum : ∑ j, (A.dev j x.1 x.2).completed x.1 (x.2 j).1 (A.oracle j x.1 x.2) ≤ P * X :=
    (Finset.sum_le_sum fun j _ => hX j _ _ x.1 (x.2 j).1 (A.oracle j x.1 x.2)).trans (by simp)
  rw [e1, e2]
  have : P * (X + slack) = P * X + P * slack := by ring
  omega

end DeployedWindow

end Accounting.Rate

namespace Accounting.Poub

open Accounting.Rate
open scoped ENNReal

variable {n P N b h ρ S L q r s d G τ : ℕ}

/-- The relay error, finite. -/
theorem relayErr_ne_top (N h L m q lam : ℕ) :
    (N.choose m : ℝ≥0∞) * 2 ^ L / 2 ^ (h * m) + (2 ^ q : ℝ≥0∞) / 2 ^ lam ≠ ⊤ :=
  ENNReal.add_ne_top.2 ⟨ENNReal.div_ne_top (ENNReal.mul_ne_top (by simp) (by simp)) (by simp),
    ENNReal.div_ne_top (by simp) (by simp)⟩

theorem relayErr_toReal (P G N h L m q lam : ℕ) :
    ((P : ℝ≥0∞) * (G * ((N.choose m : ℝ≥0∞) * 2 ^ L / 2 ^ (h * m) + (2 ^ q : ℝ≥0∞) / 2 ^ lam))).toReal =
      (P * G : ℝ) * ((N.choose m : ℝ) * 2 ^ L / 2 ^ (h * m) + (2 ^ q : ℝ) / 2 ^ lam) := by
  have h1 : (N.choose m : ℝ≥0∞) * 2 ^ L / 2 ^ (h * m) ≠ ⊤ :=
    ENNReal.div_ne_top (ENNReal.mul_ne_top (by simp) (by simp)) (by simp)
  have h2 : (2 ^ q : ℝ≥0∞) / 2 ^ lam ≠ ⊤ := ENNReal.div_ne_top (by simp) (by simp)
  rw [ENNReal.toReal_mul, ENNReal.toReal_mul, ENNReal.toReal_add h1 h2, ENNReal.toReal_div, ENNReal.toReal_div,
    ENNReal.toReal_mul]
  simp only [ENNReal.toReal_natCast, ENNReal.toReal_pow, ENNReal.toReal_ofNat]
  ring

/-- Window `w`'s bound with any side sources, at the slack `S + (L − ρ) + X + (λ + m·b)` a pass. -/
theorem window_relay_le (A : DeployedSession n P N b h ρ S L q r s d G) (tier : ℕ → Fin τ) {lam m rounds cRoot F : ℕ}
    {cost : ℕ → ℕ} (hrc : 0 < rounds * cRoot) (hW : Accounting.Assumptions.WorkHardness b rounds cRoot cost)
    (hF : ∀ w η j ι c, Accounting.Assumptions.SquaringCeiling cost F ((A.window w η).device j ι c)) (w : Fin n) (η) :
    Pous.pr (fun y : Store N b × WindowCoins P N b h ρ r s d =>
        (A.window w η).reads tier y.1 y.2.1 + P * (S + (L - ρ) + Deployed.rebuildBits b rounds cRoot F + (lam + m * b)) <
          ((A.window w η).correctAll y.1 y.2.1).card * b)
      ≤ (P : ℝ≥0∞) * (G * ((N.choose m : ℝ≥0∞) * 2 ^ L / 2 ^ (h * m) + (2 ^ q : ℝ≥0∞) / 2 ^ lam)) := by
  set X := Deployed.rebuildBits b rounds cRoot F
  refine (Bandwidth.pr_mono (fun y hy => ?_)).trans ((A.window w η).pr_unpaid_draw_mem_le tier X
    (S + (L - ρ) + lam + m * b) d _ (fun j ι c E R O => PIDevice.completed_le _ hrc hW (hF w η j ι c) E R O)
    (fun j ι c => PIDevice.pr_unpaid_relay_le _ tier m lam))
  have : P * (S + (L - ρ) + X + (lam + m * b)) = P * (X + (S + (L - ρ) + lam + m * b)) := by ring
  omega

/-- **Paid** for the deployed session's run with any side sources, at the run's slack `λ + m·b`. -/
theorem ofDeployed_paid_relay (A : DeployedSession n P N b h ρ S L q r s d G) (tier : ℕ → Fin τ)
    (Nd : Node (Fin τ)) (a : Audit) {lam m rounds cRoot F : ℕ} {cost : ℕ → ℕ} (hrc : 0 < rounds * cRoot)
    (ha : a.P = n * P) (hab : a.b = b) (hal : a.lam = lam + m * b)
    (hW : Accounting.Assumptions.WorkHardness b rounds cRoot cost)
    (hF : ∀ w η j ι c, Accounting.Assumptions.SquaringCeiling cost F ((A.window w η).device j ι c)) :
    (Run.ofDeployed A tier (Deployed.rebuildBits b rounds cRoot F) Nd a).Paid
      ((n * P * G : ℝ) * ((N.choose m : ℝ) * 2 ^ L / 2 ^ (h * m) + (2 ^ q : ℝ) / 2 ^ lam)) := by
  classical
  set X := Deployed.rebuildBits b rounds cRoot F
  have h₂ := DeployedSession.pr_unpaid_le A tier (S + (L - ρ) + X + (lam + m * b)) _
    (window_relay_le A tier hrc hW hF)
  have hT : (n : ℝ≥0∞) * (P * (G * ((N.choose m : ℝ≥0∞) * 2 ^ L / 2 ^ (h * m) + (2 ^ q : ℝ≥0∞) / 2 ^ lam))) ≠ ⊤ :=
    ENNReal.mul_ne_top (ENNReal.natCast_ne_top n) (ENNReal.mul_ne_top (ENNReal.natCast_ne_top P)
      (ENNReal.mul_ne_top (ENNReal.natCast_ne_top G) (relayErr_ne_top N h L m q lam)))
  have eT : ((n : ℝ≥0∞) * (P * (G * ((N.choose m : ℝ≥0∞) * 2 ^ L / 2 ^ (h * m) +
      (2 ^ q : ℝ≥0∞) / 2 ^ lam)))).toReal =
      (n * P * G : ℝ) * ((N.choose m : ℝ) * 2 ^ L / 2 ^ (h * m) + (2 ^ q : ℝ) / 2 ^ lam) := by
    rw [ENNReal.toReal_mul, relayErr_toReal]
    simp only [ENNReal.toReal_natCast]
    ring
  unfold Run.Paid Run.Holds
  rw [← eT]
  refine pouw_pr_le ((Bandwidth.pr_mono fun x hx => ?_).trans h₂) hT
  obtain ⟨-, hlt⟩ := hx
  have hc : (Run.ofDeployed A tier X Nd a).correct x = ∑ w, ((A.win w x.2).correctAll x.1 (x.2 w).1).card := rfl
  rw [not_le, ofDeployed_paid, ofDeployed_reads, hc, ha, hab, hal] at hlt
  by_contra hno
  push Not at hno
  have hle := Finset.sum_le_sum (fun w (_ : w ∈ Finset.univ) => hno w)
  rw [Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ, Fintype.card_fin, smul_eq_mul,
    ← Finset.sum_mul] at hle
  have e : n * (P * (S + (L - ρ) + X + (lam + m * b))) =
      n * P * (L - ρ) + n * P * S + n * P * X + n * P * (lam + m * b) := by ring
  omega

/-- **Paid** for window `w`'s run with any side sources, at the run's slack `λ + m·b`. -/
theorem ofDeployedWindow_paid_relay (A : DeployedSession n P N b h ρ S L q r s d G) (w : Fin n) (tier : ℕ → Fin τ)
    (Nd : Node (Fin τ)) (a : Audit) {lam m rounds cRoot F : ℕ} {cost : ℕ → ℕ} (hrc : 0 < rounds * cRoot)
    (ha : a.P = P) (hab : a.b = b) (hal : a.lam = lam + m * b)
    (hW : Accounting.Assumptions.WorkHardness b rounds cRoot cost)
    (hF : ∀ w η j ι c, Accounting.Assumptions.SquaringCeiling cost F ((A.window w η).device j ι c)) :
    (Run.ofDeployedWindow A w tier (Deployed.rebuildBits b rounds cRoot F) Nd a).Paid
      ((P * G : ℝ) * ((N.choose m : ℝ) * 2 ^ L / 2 ^ (h * m) + (2 ^ q : ℝ) / 2 ^ lam)) := by
  classical
  set X := Deployed.rebuildBits b rounds cRoot F
  have h₂ := DeployedSession.pr_window_le A w
    (fun C y => C.reads tier y.1 y.2.1 + P * (S + (L - ρ) + X + (lam + m * b)) < (C.correctAll y.1 y.2.1).card * b) _
    (window_relay_le A tier hrc hW hF w)
  have hT : (P : ℝ≥0∞) * (G * ((N.choose m : ℝ≥0∞) * 2 ^ L / 2 ^ (h * m) + (2 ^ q : ℝ≥0∞) / 2 ^ lam)) ≠ ⊤ :=
    ENNReal.mul_ne_top (ENNReal.natCast_ne_top P)
      (ENNReal.mul_ne_top (ENNReal.natCast_ne_top G) (relayErr_ne_top N h L m q lam))
  unfold Run.Paid Run.Holds
  rw [← relayErr_toReal]
  refine pouw_pr_le ((Bandwidth.pr_mono fun x hx => ?_).trans h₂) hT
  obtain ⟨-, hlt⟩ := hx
  have hc : (Run.ofDeployedWindow A w tier X Nd a).correct x = ((A.win w x.2).correctAll x.1 (x.2 w).1).card := rfl
  rw [not_le, ofDeployedWindow_paid, ofDeployedWindow_reads, hc, ha, hab, hal] at hlt
  show (A.win w x.2).reads tier x.1 (x.2 w).1 + P * (S + (L - ρ) + X + (lam + m * b)) <
    ((A.win w x.2).correctAll x.1 (x.2 w).1).card * b
  have e : P * (S + (L - ρ) + X + (lam + m * b)) = P * (L - ρ) + P * S + P * X + P * (lam + m * b) := by ring
  omega

/-- `PoubSessionConservationRelay`'s reading on any store `store ω` drawn before the session's coins, from a bound
`ε_u` on some window's unpaid words at the slack `S + (L − ρ) + X + (λ + m·b)` a pass: the draws' terms hold at every
store, and only the unpaid words read the store's distribution. -/
theorem session_relay_of_unpaid {Ω : Type} [Fintype Ω] (store : Ω → Store N b)
    (A : DeployedSession n P N b h ρ S L q r s d G) (tier : ℕ → Fin τ) (lam m rounds cRoot F : ℕ) (dbar dw : ℝ)
    (hn : 0 < n) (hP : 0 < P) (hN : 0 < N) (hb : 0 < b) (hdw : dw ≤ 1) (εu : ℝ≥0∞)
    (hu : Pous.pr (fun x : Ω × (Fin n → WindowCoins P N b h ρ r s d) =>
        ∃ w, (A.win w x.2).reads tier (store x.1) (x.2 w).1 +
            P * (S + (L - ρ) + Deployed.rebuildBits b rounds cRoot F + (lam + m * b)) <
          ((A.win w x.2).correctAll (store x.1) (x.2 w).1).card * b) ≤ εu) :
    Pous.pr (fun x : Ω × (Fin n → WindowCoins P N b h ρ r s d) =>
        (∀ w t, (x.2 w).2 t ∈ (A.win w x.2).correctAll (store x.1) (x.2 w).1) ∧
          ((((∑ w, (A.win w x.2).reads tier (store x.1) (x.2 w).1 : ℕ)) : ℝ) <
              (1 - (dbar + Deployed.gammaPass N b ρ S L (lam + m * b) rounds cRoot F)) * (n * P * N * b) ∨
            ∃ w, (((A.win w x.2).reads tier (store x.1) (x.2 w).1 : ℕ) : ℝ) <
              (1 - (dw + Deployed.gammaPass N b ρ S L (lam + m * b) rounds cRoot F)) * (P * N * b)))
      ≤ ENNReal.ofReal (1 - dbar) ^ (d * n) + ENNReal.ofReal (1 - dw / n) ^ (d * n) + εu := by
  have main := DeployedSession.pr_paid_of_unpaid_map store A tier
    (S + (L - ρ) + Deployed.rebuildBits b rounds cRoot F + (lam + m * b))
    ⌊(1 - dbar) * ((n : ℝ) * P * N)⌋₊ ⌊(1 - dw) * ((P : ℝ) * N)⌋₊ εu hu
  have hγ := Deployed.gammaPass_mul N b ρ S L (lam + m * b) rounds cRoot F hN hb
  have hb' : (0 : ℝ) < b := by exact_mod_cast hb
  refine (Bandwidth.pr_mono (fun x hx => ?_)).trans (main.trans ?_)
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
    refine add_le_add (add_le_add ?_ ?_) le_rfl
    · rw [show ((n : ℝ≥0∞) * P * N) = ((n * P * N : ℕ) : ℝ≥0∞) by push_cast; ring]
      exact Deployed.horizon_le dbar (n * P * N) (d * n) (by positivity)
    · rw [show ((n : ℝ≥0∞) * P * N) = (n : ℝ≥0∞) * ((P * N : ℕ) : ℝ≥0∞) by push_cast; ring]
      exact Deployed.window_le dw n (P * N) (d * n) hn (by positivity) hdw

end Accounting.Poub

namespace Accounting.SecurityProofs

open Accounting.Rate Accounting.Poub
open Pous (pr)
open scoped ENNReal
open Bandwidth (pr_mono)

/-- **PoUB as deployed, with side sources that may relay receipts**
(`Accounting.Guarantees.PoubSessionConservationRelay`). -/
theorem PoubSessionConservationRelay : Accounting.Guarantees.PoubSessionConservationRelay := by
  intro n P N b h ρ S L q r s d G lam m τ rounds cRoot F cost tier A dbar dw hn hP hN hb hrc hdw hW hF
  refine (session_relay_of_unpaid id A tier lam m rounds cRoot F dbar dw hn hP hN hb hdw _
    (DeployedSession.pr_unpaid_le A tier _ _ (window_relay_le A tier hrc hW hF))).trans (le_of_eq ?_)
  ring

/-- **The deployed session with relaying side sources conserves its words as a run of PoUB**
(`Accounting.Guarantees.PoubSessionRunConservationRelay`), from `Accounting.SecurityProofs.Poub.Conservation`. -/
theorem PoubSessionRunConservationRelay : Accounting.Guarantees.PoubSessionRunConservationRelay := by
  intro n P N b h ρ S L q r s d G lam m τ rounds cRoot F cost tier A Nd a dbar hn hP hN hb hrc hdbar ha hab hal hC hW hF
  have hs := ofDeployed_sampled A tier (Deployed.rebuildBits b rounds cRoot F) Nd a dbar hn hP hN hb hdbar hab hC
  have hp := ofDeployed_paid_relay A tier Nd a hrc ha hab hal hW hF
  have hC0 : 0 < Nd.declared a.T := by rw [hC]; positivity
  refine Poub.holds_mono ?_ (Poub.Conservation (Fin τ) _ Nd a _ dbar _ _ hC0 hs hp)
  rintro x - ⟨h1, h2⟩
  refine ⟨h1, ?_⟩
  rw [ofDeployed_paid] at h2
  have hCr : (Nd.declared a.T : ℝ) = n * P * N * b := by rw [hC]; push_cast; ring
  have eΛ : muLambda Nd a * Nd.declared a.T = a.P * a.lam := by
    have : (0 : ℝ) < Nd.declared a.T := by exact_mod_cast hC0
    unfold muLambda; field_simp
  rw [ha, hal] at eΛ
  have hγ := Deployed.gammaPass_mul N b ρ S L (lam + m * b) rounds cRoot F hN hb
  have k : Deployed.gammaPass N b ρ S L (lam + m * b) rounds cRoot F * Nd.declared a.T =
      n * P * ((S + (L - ρ) + Deployed.rebuildBits b rounds cRoot F + (lam + m * b) : ℕ) : ℝ) := by
    rw [hCr, ← hγ]; ring
  push_cast at h2 eΛ k
  rw [hCr] at h2 eΛ k ⊢
  nlinarith [h2, eΛ, k]

/-- **Each window with relaying side sources conserves its words as a run of PoUB**
(`Accounting.Guarantees.PoubWindowRunConservationRelay`), from `Accounting.SecurityProofs.Poub.Conservation`. -/
theorem PoubWindowRunConservationRelay : Accounting.Guarantees.PoubWindowRunConservationRelay := by
  intro n P N b h ρ S L q r s d G lam m τ rounds cRoot F cost tier A w Nd a dw hP hN hb hrc hdw ha hab hal hC hW hF
  have hs := ofDeployedWindow_sampled A w tier (Deployed.rebuildBits b rounds cRoot F) Nd a dw hP hN hb hdw hab hC
  have hp := ofDeployedWindow_paid_relay A w tier Nd a hrc ha hab hal hW hF
  have hC0 : 0 < Nd.declared a.T := by rw [hC]; positivity
  refine Poub.holds_mono ?_ (Poub.Conservation (Fin τ) _ Nd a _ dw _ _ hC0 hs hp)
  rintro x - ⟨h1, h2⟩
  refine ⟨h1, ?_⟩
  rw [ofDeployedWindow_paid] at h2
  have hCr : (Nd.declared a.T : ℝ) = P * N * b := by rw [hC]; push_cast; ring
  have eΛ : muLambda Nd a * Nd.declared a.T = a.P * a.lam := by
    have : (0 : ℝ) < Nd.declared a.T := by exact_mod_cast hC0
    unfold muLambda; field_simp
  rw [ha, hal] at eΛ
  have hγ := Deployed.gammaPass_mul N b ρ S L (lam + m * b) rounds cRoot F hN hb
  have k : Deployed.gammaPass N b ρ S L (lam + m * b) rounds cRoot F * Nd.declared a.T =
      P * ((S + (L - ρ) + Deployed.rebuildBits b rounds cRoot F + (lam + m * b) : ℕ) : ℝ) := by
    rw [hCr, ← hγ]; ring
  push_cast at h2 eΛ k
  rw [hCr] at h2 eΛ k ⊢
  nlinarith [h2, eΛ, k]

end Accounting.SecurityProofs
