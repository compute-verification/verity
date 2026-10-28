import Mathlib
import Definitions.Accounting.Guarantees.RateAudit
import Proofs.Accounting.Bandwidth.Honest

/-!
# The honest device: the rate audit is not vacuous

A device with no on-chip state, no free side sources and nothing rebuilt, whose device memory holds the store as it is
(`Bandwidth.Completeness.layout`), reads the certified stream, then every word once, and queries the oracle at each
word under the stream. It is accepted on every store, stream, oracle and sample; its footprint and ingress are exactly
the store, `N·b` bits, its link carries exactly the stream, and its compute is `N` tags (`honest_accepts`; the
guarantee `Accounting.SecurityProofs.RateHonest`).
-/

namespace Accounting.Rate.Completeness

open Pous (Bits)

variable {N b h ρ : ℕ} {α : Type}

/-- Read the listed bits of the certified stream into `acc`, then continue with it. -/
def readStream : List (Fin ρ) → Bits ρ → (Bits ρ → Chip N b h ρ α) → Chip N b h ρ α
  | [], acc, k => k acc
  | i :: is, acc, k => .stream i (fun x => readStream is (Function.update acc i x) k)

theorem readStream_eq (w : World N b h ρ) (is : List (Fin ρ)) (acc : Bits ρ) (k : Bits ρ → Chip N b h ρ α) :
    let R : Bits ρ := fun i => if i ∈ is then w.stream i else acc i
    (readStream is acc k).run w = (k R).run w ∧ (readStream is acc k).ingressBits w = (k R).ingressBits w ∧
    (readStream is acc k).addrs w = (k R).addrs w ∧ (readStream is acc k).linkBits w = (k R).linkBits w + is.length ∧
    (readStream is acc k).rebuilds w = (k R).rebuilds w ∧ (readStream is acc k).queries w = (k R).queries w := by
  induction is generalizing acc with
  | nil => simp [readStream]
  | cons i is ih =>
    have hR : (fun i' => if i' ∈ is then w.stream i' else Function.update acc i (w.stream i) i') =
        fun i' => if i' ∈ i :: is then w.stream i' else acc i' := by
      funext i'
      by_cases h : i' = i
      · subst h; simp
      · simp [h]
    obtain ⟨h1, h2, h3, h4, h5, h6⟩ := ih (Function.update acc i (w.stream i))
    simp only [readStream, Chip.run, Chip.ingressBits, Chip.addrs, Chip.linkBits, Chip.rebuilds, Chip.queries]
    rw [h1, h2, h3, h4, h5, h6, hR, List.length_cons]
    exact ⟨rfl, rfl, rfl, by omega, rfl, rfl⟩

/-- Read `n` bits of device memory from address `a`, then continue with them. -/
def readBits (a : ℕ) : (n : ℕ) → (Bits n → Chip N b h ρ α) → Chip N b h ρ α
  | 0, k => k Fin.elim0
  | n + 1, k => .ingress a (fun x => readBits (a + 1) n (fun v => k (Fin.cons x v)))

theorem readBits_eq (w : World N b h ρ) (a n : ℕ) (k : Bits n → Chip N b h ρ α) :
    let v : Bits n := fun i => w.mem (a + i)
    (readBits a n k).run w = (k v).run w ∧ (readBits a n k).ingressBits w = (k v).ingressBits w + n ∧
    (readBits a n k).addrs w = Finset.Ico a (a + n) ∪ (k v).addrs w ∧
    (readBits a n k).linkBits w = (k v).linkBits w ∧ (readBits a n k).rebuilds w = (k v).rebuilds w ∧
    (readBits a n k).queries w = (k v).queries w := by
  induction n generalizing a with
  | zero =>
    have : (fun i : Fin 0 => w.mem (a + i)) = Fin.elim0 := funext (fun i => i.elim0)
    simp [readBits, this]
  | succ n ih =>
    have hc : (Fin.cons (w.mem a) (fun i : Fin n => w.mem (a + 1 + i)) : Bits (n + 1)) =
        fun i : Fin (n + 1) => w.mem (a + i) := by
      funext i
      refine Fin.cases ?_ (fun i => ?_) i
      · simp
      · simp only [Fin.cons_succ, Fin.val_succ]
        congr 1
        omega
    have hI : insert a (Finset.Ico (a + 1) (a + 1 + n)) = Finset.Ico a (a + (n + 1)) := by
      ext x
      simp only [Finset.mem_insert, Finset.mem_Ico]
      omega
    obtain ⟨h1, h2, h3, h4, h5, h6⟩ := ih (a + 1) (fun v => k (Fin.cons (w.mem a) v))
    simp only [readBits, Chip.run, Chip.ingressBits, Chip.addrs, Chip.linkBits, Chip.rebuilds, Chip.queries]
    rw [h1, h2, h3, h4, h5, h6, hc, ← Finset.insert_union, hI]
    exact ⟨rfl, by omega, rfl, rfl, rfl, rfl⟩

/-- Query the oracle at each listed word of `V` under the stream `R`, recording the tag. -/
def tagAll (R : Bits ρ) (V : Store N b) : List (Fin N) → (Fin N → Bits h) → Chip N b h ρ (Fin N → Bits h)
  | [], acc => .done acc
  | j :: js, acc => .tag (R, j, V j) (fun y => tagAll R V js (Function.update acc j y))

theorem tagAll_eq (w : World N b h ρ) (R : Bits ρ) (V : Store N b) (js : List (Fin N)) (acc : Fin N → Bits h) :
    (tagAll R V js acc).run w = (fun j => if j ∈ js then w.oracle (R, j, V j) else acc j) ∧
    (tagAll R V js acc).ingressBits w = 0 ∧ (tagAll R V js acc).addrs w = ∅ ∧
    (tagAll R V js acc).linkBits w = 0 ∧ (tagAll R V js acc).rebuilds w = 0 ∧
    (tagAll R V js acc).queries w = js.map (fun j => (R, j, V j)) := by
  induction js generalizing acc with
  | nil => simp [tagAll, Chip.run, Chip.ingressBits, Chip.addrs, Chip.linkBits, Chip.rebuilds, Chip.queries]
  | cons j js ih =>
    obtain ⟨i1, i2, i3, i4, i5, i6⟩ := ih (Function.update acc j (w.oracle (R, j, V j)))
    simp only [tagAll, Chip.run, Chip.ingressBits, Chip.addrs, Chip.linkBits, Chip.rebuilds, Chip.queries]
    refine ⟨?_, i2, i3, i4, i5, by rw [i6, List.map_cons]⟩
    rw [i1]
    funext j'
    by_cases hj : j' = j
    · subst hj; simp
    · by_cases hm : j' ∈ js <;> simp [hm, hj]

theorem word_lt {j : Fin N} {i : Fin b} : (j : ℕ) * b + i < N * b := by
  have := j.2; have := i.2
  calc (j : ℕ) * b + i < j * b + b := by omega
    _ = (j + 1) * b := by ring
    _ ≤ N * b := Nat.mul_le_mul_right _ (by omega)

/-- The store's words in `N·b` bits laid out word by word. -/
def words (v : Bits (N * b)) : Store N b := fun j i => v ⟨j * b + i, word_lt⟩

/-- The honest chip: read the stream, then device memory's `N·b` bits, then tag every word under the stream. -/
def chip (N b h ρ : ℕ) : Chip N b h ρ (Fin N → Bits h) :=
  readStream (List.finRange ρ) (fun _ => false) (fun R =>
    readBits 0 (N * b) (fun v => tagAll R (words v) (List.finRange N) (fun _ _ => false)))

theorem chip_eq (w : World N b h ρ) :
    (chip N b h ρ).run w = (fun j => w.oracle (w.stream, j, words (fun i => w.mem (0 + i)) j)) ∧
    (chip N b h ρ).ingressBits w = N * b ∧ (chip N b h ρ).addrs w = Finset.range (N * b) ∧
    (chip N b h ρ).linkBits w = ρ ∧ (chip N b h ρ).rebuilds w = 0 ∧
    ((chip N b h ρ).queries w).length = N := by
  have hR : (fun i => if i ∈ List.finRange ρ then w.stream i else false) = w.stream := by
    funext i; simp
  obtain ⟨h1, h2, h3, h4, h5, h6⟩ := readStream_eq (N := N) (b := b) (h := h) w (List.finRange ρ)
    (fun _ => false) (fun R => readBits 0 (N * b) (fun v => tagAll R (words v) (List.finRange N) (fun _ _ => false)))
  simp only [hR] at h1 h2 h3 h4 h5 h6
  obtain ⟨g1, g2, g3, g4, g5, g6⟩ := readBits_eq w 0 (N * b)
    (fun v => tagAll (h := h) w.stream (words v) (List.finRange N) (fun _ _ => false))
  obtain ⟨i1, i2, i3, i4, i5, i6⟩ := tagAll_eq w w.stream (words (fun i : Fin (N * b) => w.mem (0 + i)))
    (List.finRange N) (fun _ _ => false)
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [chip, h1, g1, i1]
    simp
  · rw [chip, h2, g2, i2]; simp
  · rw [chip, h3, g3, i3]
    ext a; simp
  · rw [chip, h4, g4, i4]; simp
  · rw [chip, h5, g5, i5]
  · rw [chip, h6, g6, i6]; simp

/-- The honest device: no on-chip state and no free side sources; device memory holds the store as it is, and the
chip reads the stream, then every word once, and tags it under the stream. -/
def honest (N b h ρ : ℕ) : Device N b h ρ 0 ρ N where
  state _ := Fin.elim0
  mem := Bandwidth.Completeness.layout
  side _ _ := false
  chip _ := chip N b h ρ
  bounded _ w := by
    obtain ⟨-, -, -, h4, -, h6⟩ := chip_eq (h := h) w
    exact ⟨h4.le, h6.le⟩

/-- **Completeness.** The honest device is accepted on every store, stream, oracle and sample; its footprint and
ingress are the store, its compute `N` tags. -/
theorem honest_accepts (cTag cEnc : ℕ) (E : Store N b) (R : Bits ρ) (O : Oracle N b h ρ) {s : ℕ}
    (I : Fin s → Fin N) :
    (honest N b h ρ).Accepts E R O I ∧ (honest N b h ρ).footprint E R O = N * b ∧
      (honest N b h ρ).ingress E R O = N * b ∧ (honest N b h ρ).compute cTag cEnc E R O = N * cTag := by
  obtain ⟨h1, h2, h3, -, h5, h6⟩ := chip_eq (h := h) ((honest N b h ρ).world E R O)
  have hw : words (fun i : Fin (N * b) => Bandwidth.Completeness.layout E (0 + i)) = E := by
    funext j i
    simp only [words, zero_add]
    exact Bandwidth.Completeness.layout_block E j i
  refine ⟨fun i => ?_, ?_, ?_, ?_⟩
  · have := congrFun h1 (I i)
    simp only [Device.tags, honest, Device.world] at this ⊢
    rw [this, hw]
  · simp only [Device.footprint, honest] at h3 ⊢
    rw [h3, Finset.card_range]
  · simpa [Device.ingress, honest] using h2
  · simp only [Device.compute, honest] at h5 h6 ⊢
    rw [h5, h6, zero_mul, add_zero]

end Accounting.Rate.Completeness

/-- **Completeness of the rate audit** (`Accounting.Guarantees.RateHonest`). -/
theorem Accounting.SecurityProofs.RateHonest : Accounting.Guarantees.RateHonest :=
  fun N b h ρ => ⟨Accounting.Rate.Completeness.honest N b h ρ,
    fun cTag cEnc E R O _ I => Accounting.Rate.Completeness.honest_accepts cTag cEnc E R O I⟩
