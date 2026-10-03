import Pous.SecurityProofs.Bandwidth.Digest

/-!
# The bulk read pins the challenged blocks to device memory

`Pous.Guarantees.HbmPlacement` and `Pous.Guarantees.HbmPlacementFStar` from `TwoTierBandwidth`:

1. A response within `T` reads fewer than `h` host-resident blocks, so it leaves a challenged block `j` unread
   (`exists_unread`).
2. A program's output depends only on the blocks it reads (`run_reads_congr`).
3. Changing the last word of `j` by `y` changes the digest by `y · z^r` (`digest_setLast_sub`): for `z ≠ 0` at most
   one value of that word makes an answer that ignores `j` right. Counting the stores along that word
   (`card_line_le`) gives probability `1/|F|` for each `j`, and `z = 0` adds `1/|F|`.
-/

namespace PousBandwidth

open Pous Pous.Bandwidth Pous.Assumptions
open scoped ENNReal NNReal

/-- A program's output and reads depend only on the blocks it reads. -/
theorem run_reads_congr {F : Type} [Fintype F] [Nonempty F] {N W : ℕ} {α : Type} (p : Prog (memory N W F) α)
    (C C' : Store N W F) (h : ∀ i ∈ reads C p, C' i = C i) :
    p.run ((memory N W F).answer C') = p.run ((memory N W F).answer C) ∧ reads C' p = reads C p := by
  induction p with
  | ret a => exact ⟨rfl, rfl⟩
  | round qs k ih =>
    have hq : qs.map ((memory N W F).answer C') = qs.map ((memory N W F).answer C) :=
      List.map_congr_left fun i hi => h i (by simp [reads, hi])
    have h' : ∀ i ∈ reads C (k (qs.map ((memory N W F).answer C))), C' i = C i :=
      fun i hi => h i (by simp only [reads, Finset.mem_union]; exact Or.inr hi)
    obtain ⟨h1, h2⟩ := ih _ h'
    refine ⟨?_, ?_⟩
    · show Prog.run _ (k (qs.map ((memory N W F).answer C'))) = Prog.run _ (k (qs.map ((memory N W F).answer C)))
      rw [hq, h1]
    · show qs.toFinset ∪ reads C' (k (qs.map ((memory N W F).answer C'))) =
        qs.toFinset ∪ reads C (k (qs.map ((memory N W F).answer C)))
      rw [hq, h2]

section Line

variable {F : Type}

/-- The last word of a block. -/
def lastWord {W : ℕ} (hW : 0 < W) : Fin W := ⟨W - 1, by omega⟩

/-- The store with the last word of block `j` set to `y`. -/
def setLast {N W : ℕ} (hW : 0 < W) (j : Fin N) (C : Store N W F) (y : F) : Store N W F :=
  Function.update C j (Function.update (C j) (lastWord hW) y)

theorem setLast_ne {N W : ℕ} (hW : 0 < W) {j i : Fin N} (C : Store N W F) (y : F) (h : i ≠ j) :
    setLast hW j C y i = C i :=
  Function.update_of_ne h _ _

theorem setLast_involutive {N W : ℕ} (hW : 0 < W) (j : Fin N) :
    Function.Involutive fun x : Store N W F × F => (setLast hW j x.1 x.2, x.1 j (lastWord hW)) := by
  intro x
  simp [setLast]

end Line

variable {F : Type} [Field F] [Fintype F] [DecidableEq F]

omit [Fintype F] [DecidableEq F] in
/-- The last word enters a block's hash with coefficient `g^0 = 1`. -/
theorem blockHash_update_last_sub {W : ℕ} (hW : 0 < W) (g : F) (x : Blk W F) (y y' : F) :
    blockHash g (Function.update x (lastWord hW) y) - blockHash g (Function.update x (lastWord hW) y') = y - y' := by
  rw [blockHash_sub, Finset.sum_eq_single (lastWord hW)]
  · simp [lastWord]
  · intro i _ hi
    simp [Function.update_of_ne hi]
  · simp

omit [Fintype F] [DecidableEq F] in
/-- Setting the last word of the challenged block `S r0` moves the digest by `y · z^r0`. -/
theorem digest_setLast_sub {N R W : ℕ} (hW : 0 < W) (S : Fin R → Fin N) (hS : Function.Injective S) (r0 : Fin R)
    (C : Store N W F) (g z y y' : F) :
    digest g z (fun r => setLast hW (S r0) C y (S r)) - digest g z (fun r => setLast hW (S r0) C y' (S r)) =
      (y - y') * z ^ (r0 : ℕ) := by
  rw [digest_sub, Finset.sum_eq_single r0]
  · simp only [setLast, Function.update_self]
    rw [blockHash_update_last_sub]
  · intro r _ hr
    rw [setLast_ne hW C y (hS.ne hr), setLast_ne hW C y' (hS.ne hr), sub_self, zero_mul]
  · simp

/-- Along the last word of the challenged block `S r0`, at most one store has a right answer that leaves the block
unread, when `z ≠ 0`. -/
theorem card_line_le {N R W : ℕ} (hW : 0 < W) (S : Fin R → Fin N) (hS : Function.Injective S) (r0 : Fin R)
    (p : Prog (memory N W F) F) (g z : F) (hz : z ≠ 0) (C : Store N W F) :
    (Finset.univ.filter fun y : F =>
      S r0 ∉ reads (setLast hW (S r0) C y) p ∧
        p.run ((memory N W F).answer (setLast hW (S r0) C y)) =
          digest g z (fun r => setLast hW (S r0) C y (S r))).card ≤ 1 := by
  refine Finset.card_le_one.2 fun y hy y' hy' => ?_
  simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hy hy'
  have hsame : p.run ((memory N W F).answer (setLast hW (S r0) C y')) =
      p.run ((memory N W F).answer (setLast hW (S r0) C y)) := by
    refine (run_reads_congr p _ _ fun i hi => ?_).1
    have hij : i ≠ S r0 := fun h => hy.1 (h ▸ hi)
    rw [setLast_ne hW C y' hij, setLast_ne hW C y hij]
  have hd := digest_setLast_sub hW S hS r0 C g z y y'
  rw [← hy.2, ← hy'.2, hsame, sub_self] at hd
  have := (mul_eq_zero.1 hd.symm).resolve_right (pow_ne_zero _ hz)
  exact (sub_eq_zero.1 this)

/-- At a key with `z ≠ 0`, the stores on which a program is right while leaving the challenged block `S r0` unread are
at most a `1/|F|` fraction. -/
theorem card_unread_right_le {N R W : ℕ} (hW : 0 < W) (S : Fin R → Fin N) (hS : Function.Injective S) (r0 : Fin R)
    (p : Prog (memory N W F) F) (g z : F) (hz : z ≠ 0) :
    Fintype.card F * (Finset.univ.filter fun C : Store N W F =>
      S r0 ∉ reads C p ∧ p.run ((memory N W F).answer C) = digest g z (fun r => C (S r))).card ≤
        Fintype.card (Store N W F) := by
  set E : Store N W F → Prop := fun C =>
    S r0 ∉ reads C p ∧ p.run ((memory N W F).answer C) = digest g z (fun r => C (S r)) with hE
  set Ψ : Store N W F × F → Store N W F × F := fun x => (setLast hW (S r0) x.1 x.2, x.1 (S r0) (lastWord hW))
  have hΨ : Function.Involutive Ψ := setLast_involutive hW (S r0)
  have h1 : (Finset.univ.filter fun x : Store N W F × F => E x.1).card = (Finset.univ.filter E).card * Fintype.card F := by
    rw [show (Finset.univ.filter fun x : Store N W F × F => E x.1) = (Finset.univ.filter E) ×ˢ Finset.univ by
      ext x; simp, Finset.card_product, Finset.card_univ]
  have h2 : (Finset.univ.filter fun x : Store N W F × F => E x.1).card =
      (Finset.univ.filter fun x : Store N W F × F => E (Ψ x).1).card := by
    refine Finset.card_bij' (fun x _ => Ψ x) (fun x _ => Ψ x) (fun x hx => ?_) (fun x hx => ?_)
      (fun x _ => hΨ x) (fun x _ => hΨ x)
    · simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hx ⊢
      rwa [hΨ x]
    · simpa using hx
  have h3 : (Finset.univ.filter fun x : Store N W F × F => E (Ψ x).1).card ≤ Fintype.card (Store N W F) := by
    rw [card_filter_prod]
    calc ∑ C : Store N W F, (Finset.univ.filter fun y : F => E (Ψ (C, y)).1).card
        ≤ ∑ _C : Store N W F, 1 := Finset.sum_le_sum fun C _ => card_line_le hW S hS r0 p g z hz C
      _ = Fintype.card (Store N W F) := by simp
  show Fintype.card F * (Finset.univ.filter E).card ≤ Fintype.card (Store N W F)
  rw [mul_comm, ← h1, h2]
  exact h3

omit [DecidableEq F] in
/-- A response within the deadline reads fewer host-resident blocks than the challenge holds, so it leaves one unread. -/
theorem exists_unread {N R W : ℕ} {host : Finset (Fin N)} {b BWpcie T : ℝ≥0} {A : BulkResponder N R W F}
    (hTTB : TwoTierBandwidth host b BWpcie A) {S : Fin R → Fin N}
    (hT : BWpcie * T < b * ((Finset.univ.image S ∩ host).card : ℝ≥0)) {C : Store N W F} {k : Key F}
    (hacc : Accepts A T S C k) : ∃ j ∈ Finset.univ.image S ∩ host, j ∉ reads C (A.prog k S) := by
  by_contra hcon
  push Not at hcon
  have hsub : Finset.univ.image S ∩ host ⊆ reads C (A.prog k S) ∩ host :=
    fun j hj => Finset.mem_inter.2 ⟨hcon j hj, (Finset.mem_inter.1 hj).2⟩
  have hcard : ((Finset.univ.image S ∩ host).card : ℝ≥0) ≤ (reads C (A.prog k S) ∩ host).card := by
    exact_mod_cast Finset.card_le_card hsub
  have := calc b * ((Finset.univ.image S ∩ host).card : ℝ≥0)
      ≤ b * (reads C (A.prog k S) ∩ host).card := by gcongr
    _ ≤ BWpcie * A.time C k S := hTTB C k S
    _ ≤ BWpcie * T := by gcongr; exact hacc.2
    _ < b * ((Finset.univ.image S ∩ host).card : ℝ≥0) := hT
  exact lt_irrefl _ this

theorem hbmPlacement : Pous.Guarantees.HbmPlacement := by
  intro N R W F _ _ hW host b BWpcie T A hTTB S hS hT
  classical
  set H := Finset.univ.image S ∩ host with hH
  set acc := Finset.univ.filter fun x : Store N W F × Key F => Accepts A T S x.1 x.2
  set Z := Finset.univ.filter fun x : Store N W F × Key F => x.2.2 = 0
  set B : Fin N → Finset (Store N W F × Key F) := fun j => Finset.univ.filter fun x =>
    x.2.2 ≠ 0 ∧ j ∉ reads x.1 (A.prog x.2 S) ∧
      (A.prog x.2 S).run ((memory N W F).answer x.1) = digest x.2.1 x.2.2 (fun r => x.1 (S r))
  have hsub : acc ⊆ Z ∪ H.biUnion B := by
    intro x hx
    simp only [acc, Finset.mem_filter, Finset.mem_univ, true_and] at hx
    by_cases hz : x.2.2 = 0
    · exact Finset.mem_union_left _ (by simp [Z, hz])
    · obtain ⟨j, hj, hjr⟩ := exists_unread hTTB hT hx
      exact Finset.mem_union_right _ (Finset.mem_biUnion.2 ⟨j, hj, by simp [B, hz, hjr, hx.1]⟩)
  have hZ : Z.card = Fintype.card (Store N W F) * Fintype.card F := by
    rw [show Z = Finset.univ ×ˢ (Finset.univ ×ˢ {0}) by ext ⟨C, g, z⟩; simp [Z, eq_comm]]
    simp
  have hB : ∀ j ∈ H, (B j).card ≤ Fintype.card (Store N W F) * Fintype.card F := by
    intro j hj
    obtain ⟨r0, -, rfl⟩ := Finset.mem_image.1 (Finset.mem_inter.1 hj).1
    have hslice : ∀ k : Key F, Fintype.card F * (Finset.univ.filter fun C : Store N W F =>
        k.2 ≠ 0 ∧ S r0 ∉ reads C (A.prog k S) ∧
          (A.prog k S).run ((memory N W F).answer C) = digest k.1 k.2 (fun r => C (S r))).card ≤
            Fintype.card (Store N W F) := by
      intro k
      by_cases hz : k.2 = 0
      · simp [hz]
      · simpa [hz] using card_unread_right_le hW S hS r0 (A.prog k S) k.1 k.2 hz
    have hsum : Fintype.card F * (B (S r0)).card ≤ Fintype.card (Key F) * Fintype.card (Store N W F) := by
      rw [card_filter_prod_right, Finset.mul_sum]
      calc ∑ k : Key F, Fintype.card F * (Finset.univ.filter fun C : Store N W F =>
            (C, k).2.2 ≠ 0 ∧ S r0 ∉ reads (C, k).1 (A.prog (C, k).2 S) ∧
              (A.prog (C, k).2 S).run ((memory N W F).answer (C, k).1) =
                digest (C, k).2.1 (C, k).2.2 (fun r => (C, k).1 (S r))).card
          ≤ ∑ _k : Key F, Fintype.card (Store N W F) := Finset.sum_le_sum fun k _ => hslice k
        _ = Fintype.card (Key F) * Fintype.card (Store N W F) := by simp
    rw [Fintype.card_prod] at hsum
    have hF : 0 < Fintype.card F := Fintype.card_pos
    nlinarith
  have hcount : acc.card ≤ (H.card + 1) * (Fintype.card F * Fintype.card (Store N W F)) := by
    calc acc.card ≤ (Z ∪ H.biUnion B).card := Finset.card_le_card hsub
      _ ≤ Z.card + (H.biUnion B).card := Finset.card_union_le _ _
      _ ≤ Z.card + ∑ j ∈ H, (B j).card := by gcongr; exact Finset.card_biUnion_le
      _ ≤ Fintype.card (Store N W F) * Fintype.card F +
            ∑ _j ∈ H, Fintype.card (Store N W F) * Fintype.card F := by
          rw [hZ]; gcongr with j hj; exact hB j hj
      _ = (H.card + 1) * (Fintype.card F * Fintype.card (Store N W F)) := by
          rw [Finset.sum_const, smul_eq_mul]; ring
  refine pr_le_of_card _ _ _ _ (Nat.mul_ne_zero Fintype.card_ne_zero Fintype.card_ne_zero) ?_ hcount
  rw [Fintype.card_prod, Fintype.card_prod]
  ring

theorem hbmPlacementFStar : Pous.Guarantees.HbmPlacementFStar := by
  intro N R W F _ _ hW host b BWpcie BWdev T A hTTB hb hdev hTdev S hS hf
  refine hbmPlacement N R W F hW host b BWpcie T A hTTB S hS ?_
  set h : ℝ≥0 := ((Finset.univ.image S ∩ host).card : ℝ≥0)
  have hR : (R : ℝ≥0) ≠ 0 := by
    rintro hR
    rw [hR, div_zero] at hf
    exact (not_lt.2 zero_le) hf
  have hcross : BWpcie * R < h * BWdev := (div_lt_div_iff₀ hdev (pos_iff_ne_zero.2 hR)).1 hf
  have : BWpcie * T * BWdev < b * h * BWdev := by
    calc BWpcie * T * BWdev = BWpcie * (T * BWdev) := by ring
      _ ≤ BWpcie * (R * b) := by gcongr
      _ = BWpcie * R * b := by ring
      _ < h * BWdev * b := mul_lt_mul_of_pos_right hcross hb
      _ = b * h * BWdev := by ring
  exact lt_of_mul_lt_mul_right this zero_le

end PousBandwidth
