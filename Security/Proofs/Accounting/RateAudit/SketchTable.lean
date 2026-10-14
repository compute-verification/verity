import Mathlib
import Models.Accounting.Guarantees.RateAuditSketch

/-!
# The table: outside the verbatim class, a prover reads only the digest

A prover whose memory is laid out before the signs as any function of the store lays out, for every signs `σ` and tile
`t`, the `d` bits of an encoding of the digest `dig σ (tile t)` at the address `(j·N + t)·d`, `j` the index of `σ`. Its
chip takes the certified stream in over the link, then for each tile reads those `d` bits at the window's signs,
decodes the digest and queries the oracle at it. Every tag it commits is right, so it is accepted at every sample; it
reads `N·d` bits and its memory is `|Sig|·N·d` bits (`table_run`, `table_mem`; the guarantees
`Accounting.SecurityProofs.RateSketchTable` and `RateTensorSketchTable`).
-/

namespace Accounting.RateSketch.Table

open Pous (Bits)

variable {N ρ h : ℕ} {X Dg α : Type}

/-- Read the `d` bits at addresses `a, a + 1, …, a + d − 1`, then continue with them. -/
def readBits : (a d : ℕ) → (Bits d → LayoutChip N ρ h X Dg α) → LayoutChip N ρ h X Dg α
  | _, 0, k => k fun i => Fin.elim0 i
  | a, d + 1, k => .read a fun b => readBits (a + 1) d fun v => k (Fin.cons b v)

theorem readBits_eq (O : Oracle N ρ h X Dg) (M : ℕ → Bool) (R : Fin ρ → X) (a d : ℕ)
    (k : Bits d → LayoutChip N ρ h X Dg α) :
    (readBits a d k).run O M R = (k fun i => M (a + i)).run O M R ∧
    (readBits a d k).queries O M R = (k fun i => M (a + i)).queries O M R ∧
    (readBits a d k).reads O M R = (k fun i => M (a + i)).reads O M R + d ∧
    (readBits a d k).linkSteps O M R = (k fun i => M (a + i)).linkSteps O M R := by
  induction d generalizing a with
  | zero =>
    have hv : (fun i : Fin 0 => Fin.elim0 i : Bits 0) = fun i : Fin 0 => M (a + (i : ℕ)) := funext fun i => Fin.elim0 i
    simp [readBits, hv]
  | succ d ih =>
    obtain ⟨h1, h2, h3, h4⟩ := ih (a + 1) (fun v => k (Fin.cons (M a) v))
    have hv : (Fin.cons (M a) (fun i : Fin d => M (a + 1 + (i : ℕ))) : Bits (d + 1)) = fun i : Fin (d + 1) => M (a + (i : ℕ)) := by
      funext i
      refine Fin.cases ?_ (fun j => ?_) i
      · simp
      · simp only [Fin.cons_succ, Fin.val_succ]
        congr 1
        omega
    simp only [readBits, LayoutChip.run, LayoutChip.queries, LayoutChip.reads, LayoutChip.linkSteps]
    rw [h1, h2, h3, h4, hv]
    exact ⟨rfl, rfl, by omega, rfl⟩

/-- Take the listed sectors of the certified stream in over the link into `acc`, then continue with it. -/
def linkAll : List (Fin ρ) → (Fin ρ → X) → ((Fin ρ → X) → LayoutChip N ρ h X Dg α) → LayoutChip N ρ h X Dg α
  | [], acc, k => k acc
  | r :: rs, acc, k => .link r fun x => linkAll rs (Function.update acc r x) k

theorem linkAll_eq (O : Oracle N ρ h X Dg) (M : ℕ → Bool) (R : Fin ρ → X) (rs : List (Fin ρ)) (acc : Fin ρ → X)
    (k : (Fin ρ → X) → LayoutChip N ρ h X Dg α) :
    (linkAll rs acc k).run O M R = (k fun r => if r ∈ rs then R r else acc r).run O M R ∧
    (linkAll rs acc k).queries O M R = (k fun r => if r ∈ rs then R r else acc r).queries O M R ∧
    (linkAll rs acc k).reads O M R = (k fun r => if r ∈ rs then R r else acc r).reads O M R ∧
    (linkAll rs acc k).linkSteps O M R = (k fun r => if r ∈ rs then R r else acc r).linkSteps O M R + rs.length := by
  induction rs generalizing acc with
  | nil => simp [linkAll]
  | cons r rs ih =>
    have hR : (fun q => if q ∈ rs then R q else Function.update acc r (R r) q) =
        fun q => if q ∈ r :: rs then R q else acc q := by
      funext q
      by_cases hq : q = r
      · subst hq; simp
      · simp [hq]
    obtain ⟨h1, h2, h3, h4⟩ := ih (Function.update acc r (R r))
    simp only [linkAll, LayoutChip.run, LayoutChip.queries, LayoutChip.reads, LayoutChip.linkSteps]
    rw [h1, h2, h3, h4, hR]
    exact ⟨rfl, rfl, rfl, by simp; omega⟩

/-- For each listed tile `t`: read the `d` bits at `addr t`, query the oracle at the digest `dec` decodes from them under
the stream `R`, and record the tag. -/
def tagAll (R : Fin ρ → X) (addr : Fin N → ℕ) (d : ℕ) (dec : Bits d → Dg) :
    List (Fin N) → (Fin N → Bits h) → LayoutChip N ρ h X Dg (Fin N → Bits h)
  | [], acc => .done acc
  | t :: ts, acc => readBits (addr t) d fun v =>
      .tag (R, t, dec v) fun y => tagAll R addr d dec ts (Function.update acc t y)

theorem tagAll_eq (O : Oracle N ρ h X Dg) (M : ℕ → Bool) (R' R : Fin ρ → X) (addr : Fin N → ℕ) (d : ℕ)
    (dec : Bits d → Dg) (ts : List (Fin N)) (acc : Fin N → Bits h) :
    (tagAll R addr d dec ts acc).run O M R' =
      (fun t => if t ∈ ts then O (R, t, dec fun i => M (addr t + i)) else acc t) ∧
    (tagAll R addr d dec ts acc).queries O M R' = ts.map (fun t => (R, t, dec fun i => M (addr t + i))) ∧
    (tagAll R addr d dec ts acc).reads O M R' = ts.length * d ∧
    (tagAll R addr d dec ts acc).linkSteps O M R' = 0 := by
  induction ts generalizing acc with
  | nil => simp [tagAll, LayoutChip.run, LayoutChip.queries, LayoutChip.reads, LayoutChip.linkSteps]
  | cons t ts ih =>
    obtain ⟨i1, i2, i3, i4⟩ := ih (Function.update acc t (O (R, t, dec fun i => M (addr t + i))))
    obtain ⟨r1, r2, r3, r4⟩ := readBits_eq O M R' (addr t) d
      (fun v => .tag (R, t, dec v) fun y => tagAll R addr d dec ts (Function.update acc t y))
    simp only [tagAll]
    rw [r1, r2, r3, r4]
    simp only [LayoutChip.run, LayoutChip.queries, LayoutChip.reads, LayoutChip.linkSteps]
    refine ⟨?_, by rw [i2, List.map_cons], by rw [i3, List.length_cons]; ring, i4⟩
    rw [i1]
    funext t'
    by_cases ht : t' = t
    · subst ht; simp
    · by_cases hm : t' ∈ ts <;> simp [hm, ht]

/-- The table's memory: at address `a`, with `a = ((j·N + t)·d + i)` for signs of index `j < |Sig|`, tile `t < N` and
bit `i < d`, bit `i` of the encoding `e` of the digest of tile `t` at the signs of index `j`; `false` past the table. -/
noncomputable def mem (N m d : ℕ) {X Sig Dg : Type} [Fintype Sig] (dig : Sig → (Fin m → X) → Dg) (e : Dg ↪ Bits d)
    (E : Fin N × Fin m → X) (a : ℕ) : Bool :=
  if hb : a / d / N < Fintype.card Sig ∧ a / d % N < N ∧ a % d < d then
    e (dig ((Fintype.equivFin Sig).symm ⟨a / d / N, hb.1⟩) fun u => E (⟨a / d % N, hb.2.1⟩, u)) ⟨a % d, hb.2.2⟩
  else false

/-- Where tile `t`'s entry at signs `σ` starts: `(j·N + t)·d`, `j` the index of `σ`. -/
noncomputable def addr {Sig : Type} [Fintype Sig] (N d : ℕ) (σ : Sig) (t : Fin N) : ℕ :=
  ((Fintype.equivFin Sig σ : ℕ) * N + t) * d

theorem mem_entry (N m d : ℕ) {X Sig Dg : Type} [Fintype Sig] (dig : Sig → (Fin m → X) → Dg) (e : Dg ↪ Bits d)
    (E : Fin N × Fin m → X) (σ : Sig) (t : Fin N) :
    (fun i : Fin d => mem N m d dig e E (addr N d σ t + i)) = e (dig σ fun u => E (t, u)) := by
  funext i
  have hd : 0 < d := Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt
  have hN : 0 < N := Nat.lt_of_le_of_lt (Nat.zero_le _) t.isLt
  set j := Fintype.equivFin Sig σ with hj
  have h1 : (addr N d σ t + i) / d = (j : ℕ) * N + t := by
    rw [addr, ← hj, Nat.add_comm, Nat.add_mul_div_right _ _ hd, Nat.div_eq_of_lt i.isLt, Nat.zero_add]
  have h2 : ((j : ℕ) * N + t) / N = j := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hN, Nat.div_eq_of_lt t.isLt, Nat.zero_add]
  have h3 : ((j : ℕ) * N + t) % N = t := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt t.isLt]
  have h4 : (addr N d σ t + i) % d = i := by
    rw [addr, ← hj, Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt i.isLt]
  have hb : (addr N d σ t + i) / d / N < Fintype.card Sig ∧ (addr N d σ t + i) / d % N < N ∧
      (addr N d σ t + i) % d < d := by
    rw [h1, h2, h3, h4]
    exact ⟨j.isLt, t.isLt, i.isLt⟩
  rw [mem, dite_eq_left hb]
  have hσ : (Fintype.equivFin Sig).symm ⟨(addr N d σ t + i) / d / N, hb.1⟩ = σ := by
    rw [Equiv.symm_apply_eq]
    ext
    change (addr N d σ t + i) / d / N = _
    rw [h1, h2]
  have ht : (⟨(addr N d σ t + i) / d % N, hb.2.1⟩ : Fin N) = t := by
    ext
    change (addr N d σ t + i) / d % N = _
    rw [h1, h3]
  have hi : (⟨(addr N d σ t + i) % d, hb.2.2⟩ : Fin d) = i := by
    ext
    change (addr N d σ t + i) % d = _
    rw [h4]
  rw [hσ, ht, hi]

theorem mem_cap (N m d : ℕ) {X Sig Dg : Type} [Fintype Sig] (dig : Sig → (Fin m → X) → Dg) (e : Dg ↪ Bits d)
    (E : Fin N × Fin m → X) (a : ℕ) (ha : Fintype.card Sig * (N * d) ≤ a) : mem N m d dig e E a = false := by
  rw [mem, dite_eq_right]
  rintro ⟨hb1, hb2, hb3⟩
  have hd : 0 < d := Nat.lt_of_le_of_lt (Nat.zero_le _) hb3
  have hN : 0 < N := Nat.lt_of_le_of_lt (Nat.zero_le _) hb2
  have h1 : a / d < Fintype.card Sig * N := (Nat.div_lt_iff_lt_mul hN).1 hb1
  have h2 : a < Fintype.card Sig * N * d := (Nat.div_lt_iff_lt_mul hd).1 h1
  rw [Nat.mul_assoc] at h2
  omega

/-- **The table.** Memory `mem`; the chip takes the stream in, then for each tile reads its entry at the window's
signs, decodes the digest and queries the oracle at it. -/
noncomputable def table (N m ρ h d : ℕ) (X Sig Dg : Type) [Inhabited X] [Fintype Sig]
    (dig : Sig → (Fin m → X) → Dg) (e : Dg ↪ Bits d) : LayoutDevice N m ρ h X Sig Dg where
  mem := mem N m d dig e
  chip σ :=
    haveI : Nonempty Dg := ⟨dig σ default⟩
    linkAll (List.finRange ρ) (fun _ => default) fun R =>
      tagAll R (addr N d σ) d (Function.invFun e) (List.finRange N) fun _ _ => false

theorem table_run (N m ρ h d : ℕ) (X Sig Dg : Type) [Inhabited X] [Fintype Sig]
    (dig : Sig → (Fin m → X) → Dg) (e : Dg ↪ Bits d) (σ : Sig) (O : Oracle N ρ h X Dg) (D : Data N m ρ X) :
    (∀ t, (table N m ρ h d X Sig Dg dig e).tags σ O D t = O (D.stream, t, dig σ (D.tile t))) ∧
      ((table N m ρ h d X Sig Dg dig e).queries σ O D).length = N ∧
      (table N m ρ h d X Sig Dg dig e).reads σ O D = N * d ∧
      (table N m ρ h d X Sig Dg dig e).linkSteps σ O D = ρ := by
  have : Nonempty Dg := ⟨dig σ default⟩
  have hR : (fun r => if r ∈ List.finRange ρ then D.stream r else (fun _ => (default : X)) r) = D.stream := by
    funext r; simp
  have hdec : ∀ t : Fin N, Function.invFun e (fun i => mem N m d dig e D.store (addr N d σ t + i)) =
      dig σ (D.tile t) := by
    intro t
    rw [mem_entry]
    exact Function.leftInverse_invFun e.injective _
  obtain ⟨l1, l2, l3, l4⟩ := linkAll_eq (h := h) O (mem N m d dig e D.store) D.stream (List.finRange ρ)
    (fun _ => default) fun R => tagAll R (addr N d σ) d (Function.invFun e) (List.finRange N) fun _ _ => false
  rw [hR] at l1 l2 l3 l4
  obtain ⟨t1, t2, t3, t4⟩ := tagAll_eq O (mem N m d dig e D.store) D.stream D.stream (addr N d σ) d
    (Function.invFun e) (List.finRange N) fun _ _ => false
  refine ⟨fun t => ?_, ?_, ?_, ?_⟩
  · simp only [LayoutDevice.tags, table]
    rw [l1, t1]
    simp [hdec]
  · simp only [LayoutDevice.queries, table]
    rw [l2, t2]
    simp
  · simp only [LayoutDevice.reads, table]
    rw [l3, t3, List.length_finRange]
  · simp only [LayoutDevice.linkSteps, table]
    rw [l4, t4, List.length_finRange, Nat.zero_add]

end Accounting.RateSketch.Table

open Accounting.RateSketch

/-- **The table** (`Accounting.Guarantees.RateSketchTable`). -/
theorem Accounting.SecurityProofs.RateSketchTable : Accounting.Guarantees.RateSketchTable := by
  intro N m ρ h d X Sig Dg _ _ _ dig hd
  obtain ⟨e⟩ := Function.Embedding.nonempty_of_card_le (α := Dg) (β := Pous.Bits d) (by simpa using hd)
  exact ⟨Table.table N m ρ h d X Sig Dg dig e, Table.mem_cap N m d dig e,
    Table.table_run N m ρ h d X Sig Dg dig e⟩

/-- **The table at the deployed tag** (`Accounting.Guarantees.RateTensorSketchTable`). -/
theorem Accounting.SecurityProofs.RateTensorSketchTable : Accounting.Guarantees.RateTensorSketchTable := by
  intro N m ρ h ks
  have hS : Fintype.card (SketchSigns ks m) = 2 ^ (ks * m * 32) := by
    simp only [SketchSigns, Fintype.card_fun, Fintype.card_fin, Fintype.card_bool, ← pow_mul]
    congr 1
    ring
  have hD : Fintype.card (Fin ks → ZMod (2 ^ 32)) ≤ 2 ^ (32 * ks) := by
    simp only [Fintype.card_fun, Fintype.card_fin, ZMod.card, ← pow_mul, le_refl]
  obtain ⟨A, hmem, hrun⟩ := Accounting.SecurityProofs.RateSketchTable N m ρ h (32 * ks) Sector (SketchSigns ks m)
    (Fin ks → ZMod (2 ^ 32)) sketch hD
  exact ⟨A, fun E a ha => hmem E a (by rwa [hS]), hrun⟩
