import Proofs.Flock.Claims.Complete.Zk.Run
import Proofs.Flock.Claims.Complete.Coins
import Proofs.Flock.Claims.Complete.Runs

/-!
# δcZk: where the honest masked prover can fail

The masked verifier refuses an honest proof at three kinds of coin, and nowhere else. A masked check is an affine
constraint on the pads, and at the honest pads (`padsOf`: `h_ab = h_a·h_b` and `w = x·y` set) each constraint is the
plain check on the true values: the masks cancel at every tape, whatever the pads, so no mask can make an honest proof
fail.

* **Zerocheck's outer coins** (`outerSlotsZk`), as in the plain case (`Bad`): at `r_i = 1` the replay's `(1 + r)⁻¹` is
  `0` and the zerocheck's final constraint need not hold at the true values. Rep `q`'s second round, elements `0` to
  `m − 14`.
* **Level 0's pair coins** (`rhoSlotsZk`, `rhoDep`): `ρ_0, ρ_1`, read as `(c_0, c_1)` from Ligerito's rounds `k_0 + 2`
  and `k_0 + 3`, are refused when dependent over F128 (`Zk.rhoIndependent`), by the verifier and the prover alike. At
  most `2·2^-128` (`pr_rho_le`): `ρ_0`'s `c_0` is `0`, or it is not and `ρ_1`'s `c_1` is the one value that makes them
  dependent.
* **The inner proof's `β`** (`betaSlotsZk`): refused at `0` (round `2` of the inner proof), by the verifier and the
  prover alike.
* **The mask slot's rank** (`rankSlotsZk`, `MaskRank.MaskRankAt`): the prover refuses coins at which the mask slot's
  weights at its claim points so far are rank-deficient (`rankCheckP`), one event per table, at its rep `1`'s four
  points. At most `4·m·2^-128` per table: the open premise `MaskRank.MaskRankBound`.

`εZk` is their count, each pair coin twice and each table's rank checks `4·m` times, over `2^128` (`Complete.Ready`,
with the slots and `δcZk`); `pr_badZk_le` is the union bound. The slots read the statement's `m`, `k_log`, regions and
points, as Deploy's `repRounds true`, and the
`--zk` schedule's `k_0` and levels; rep `q` is table `q / 2`'s rep `q % 2`, its first coin after the link's and every
earlier rep's rounds (`repStartZk`).
-/

namespace FlockComplete

open Flock FlockDeploy

/-! ## The probability of the bad coins -/

/-- **A coin element is a given value with probability at most `2^-128`**, at an element index below `65536`. -/
theorem pr_slot_eq_le {k : ℕ} (j : ℕ) {i : ℕ} (hi : i < 65536) (c : F128) :
    Claims.Pr (fun r : Fin k → Coin => j < k ∧ (Claims.coinSeq r j).at i = c) ≤ 1 / 2 ^ 128 := by
  by_cases hj : j < k
  · let e1 : (Fin k → Coin) ≃ Coin × ({j' // j' ≠ (⟨j, hj⟩ : Fin k)} → Coin) :=
      Equiv.piSplitAt (⟨j, hj⟩ : Fin k) fun _ => Coin
    let e2 : Coin ≃ FlockLevel3.GF128 × ({i' // i' ≠ (⟨i, hi⟩ : Fin 65536)} → FlockLevel3.GF128) :=
      Equiv.piSplitAt (⟨i, hi⟩ : Fin 65536) fun _ => FlockLevel3.GF128
    have : Nonempty FlockLevel3.GF128 := ⟨(0 : F128)⟩
    have hev : ∀ r : Fin k → Coin, (j < k ∧ (Claims.coinSeq r j).at i = c) ↔
        (e1.trans (Equiv.prodCongr e2 (Equiv.refl _)) r).1.1 = c := by
      intro r
      simp only [hj, true_and, Claims.coinSeq, dite_true, Coin.at, hi]
      rfl
    rw [Claims.Pr_congr hev, Pr_equiv (e1.trans (Equiv.prodCongr e2 (Equiv.refl _))) (fun s => s.1.1 = c),
      Claims.Pr_fst (fun s : FlockLevel3.GF128 × _ => s.1 = c), Claims.Pr_fst (fun g : FlockLevel3.GF128 => g = c)]
    exact le_of_eq ((Pr_eq_const (α := FlockLevel3.GF128) c).trans (by rw [FlockLevel3.card_F]; norm_num))
  · refine le_trans (le_of_eq (Claims.Pr_congr (q := fun _ => False) fun r => ?_)) ?_
    · simp only [hj, false_and]
    · rw [Claims.Pr_false]; positivity

/-- **An event with at most one point in each fiber** of `e`'s second coordinate has probability at most
`1/|A|`. -/
theorem Pr_fiber_le {α A B : Type} [Fintype α] [Fintype A] [Fintype B] [Nonempty A] [Nonempty B] (e : α ≃ A × B)
    (p : α → Prop) (h : ∀ x y, p x → p y → (e x).2 = (e y).2 → (e x).1 = (e y).1) :
    Claims.Pr p ≤ 1 / Fintype.card A := by
  classical
  unfold Claims.Pr
  have hc : (Finset.univ.filter p).card ≤ Fintype.card B := by
    rw [← Finset.card_univ (α := B)]
    refine Finset.card_le_card_of_injOn (fun x => (e x).2) (fun _ _ => Finset.mem_coe.2 (Finset.mem_univ _)) ?_
    intro x hx y hy hxy
    have hx' : p x := by simpa using hx
    have hy' : p y := by simpa using hy
    exact e.injective (Prod.ext (h x y hx' hy' hxy) hxy)
  have hA : (0 : ℝ) < Fintype.card A := by exact_mod_cast Fintype.card_pos
  have hB : (0 : ℝ) < Fintype.card B := by exact_mod_cast Fintype.card_pos
  have hc' : ((Finset.univ.filter p).card : ℝ) ≤ Fintype.card B := by exact_mod_cast hc
  rw [Fintype.card_congr e, Fintype.card_prod, Nat.cast_mul, div_le_div_iff₀ (mul_pos hA hB) hA]
  nlinarith

/-- **`ρ_0, ρ_1` dependent**: read from the coins `a` and `a + 1` as `(c_0, c_1)`,
`ρ_0.c_0 ρ_1.c_1 + ρ_0.c_1 ρ_1.c_0 = 0` (`Zk.rhoIndependent` false). -/
def rhoDep {k : ℕ} (r : Fin k → Coin) (a : ℕ) : Prop :=
  a + 1 < k ∧ (Claims.coinSeq r a).at 0 * (Claims.coinSeq r (a + 1)).at 1 +
    (Claims.coinSeq r a).at 1 * (Claims.coinSeq r (a + 1)).at 0 = 0

theorem gf_cancel {x d d' c : F128} (hx : x ≠ 0) (h : x * d + c = 0) (h' : x * d' + c = 0) : d = d' := by
  have e : FlockLevel3.GF128.of x * FlockLevel3.GF128.of d + FlockLevel3.GF128.of c =
      FlockLevel3.GF128.of x * FlockLevel3.GF128.of d' + FlockLevel3.GF128.of c := by
    rw [FlockLevel3.GF128.mul_eq, FlockLevel3.GF128.mul_eq, FlockLevel3.GF128.add_eq, FlockLevel3.GF128.add_eq, h, h']
  have hx' : FlockLevel3.GF128.of x ≠ 0 := by
    rw [FlockLevel3.GF128.zero_eq]
    exact hx
  have e' : FlockLevel3.GF128.of d = FlockLevel3.GF128.of d' := mul_left_cancel₀ hx' (add_right_cancel e)
  exact e'

/-- **The pair coins are dependent with probability at most `2·2^-128`.** -/
theorem pr_rho_le {k : ℕ} (a : ℕ) : Claims.Pr (fun r : Fin k → Coin => rhoDep r a) ≤ 2 / 2 ^ 128 := by
  by_cases ha : a + 1 < k
  · have h1 := pr_slot_eq_le (k := k) a (i := 0) (by norm_num) 0
    let e1 : (Fin k → Coin) ≃ Coin × ({j' // j' ≠ (⟨a + 1, ha⟩ : Fin k)} → Coin) :=
      Equiv.piSplitAt (⟨a + 1, ha⟩ : Fin k) fun _ => Coin
    let e2 : Coin ≃ FlockLevel3.GF128 × ({i' // i' ≠ (⟨1, by norm_num⟩ : Fin 65536)} → FlockLevel3.GF128) :=
      Equiv.piSplitAt (⟨1, by norm_num⟩ : Fin 65536) fun _ => FlockLevel3.GF128
    let e := (e1.trans (Equiv.prodCongr e2 (Equiv.refl _))).trans (Equiv.prodAssoc _ _ _)
    have : Nonempty FlockLevel3.GF128 := ⟨(0 : F128)⟩
    have h2 : Claims.Pr (fun r : Fin k → Coin => (Claims.coinSeq r a).at 0 ≠ 0 ∧ rhoDep r a) ≤ 1 / 2 ^ 128 := by
      refine le_trans (Pr_fiber_le e _ ?_) (le_of_eq (by rw [FlockLevel3.card_F]; norm_num))
      have key : ∀ r : Fin k → Coin, e r = (r ⟨a + 1, ha⟩ ⟨1, by norm_num⟩,
          (fun i' => r ⟨a + 1, ha⟩ i'.1, fun j' => r j'.1)) := fun _ => rfl
      intro x y hx hy hxy
      rw [key x, key y] at hxy
      rw [key x, key y]
      have ha' : a < k := by omega
      have hne : (⟨a, ha'⟩ : Fin k) ≠ ⟨a + 1, ha⟩ := by simp
      have h10 : (⟨0, by norm_num⟩ : Fin 65536) ≠ ⟨1, by norm_num⟩ := by simp
      have hxy1 : x ⟨a + 1, ha⟩ ⟨0, by norm_num⟩ = y ⟨a + 1, ha⟩ ⟨0, by norm_num⟩ :=
        congrArg (fun s => s.1 ⟨⟨0, by norm_num⟩, h10⟩) hxy
      have hxy2 : x ⟨a, ha'⟩ = y ⟨a, ha'⟩ := congrArg (fun s => s.2 ⟨⟨a, ha'⟩, hne⟩) hxy
      obtain ⟨hx0, -, hx⟩ := hx
      obtain ⟨-, -, hy⟩ := hy
      simp only [Claims.coinSeq, ha, ha', dite_true, Coin.at, show (0 : ℕ) < 65536 by norm_num,
        show (1 : ℕ) < 65536 by norm_num] at hx hy hx0
      rw [← hxy2, ← hxy1] at hy
      exact gf_cancel hx0 hx hy
    calc Claims.Pr (fun r : Fin k → Coin => rhoDep r a)
        ≤ Claims.Pr (fun r : Fin k → Coin => (a < k ∧ (Claims.coinSeq r a).at 0 = 0) ∨
            ((Claims.coinSeq r a).at 0 ≠ 0 ∧ rhoDep r a)) := by
          refine Claims.Pr_mono fun r hr => ?_
          by_cases h0 : (Claims.coinSeq r a).at 0 = 0
          · exact Or.inl ⟨by omega, h0⟩
          · exact Or.inr ⟨h0, hr⟩
      _ ≤ _ := Pr_or_le _ _
      _ ≤ 1 / 2 ^ 128 + 1 / 2 ^ 128 := add_le_add h1 h2
      _ = 2 / 2 ^ 128 := by ring
  · refine le_trans (le_of_eq (Claims.Pr_congr (q := fun _ => False) fun r => ?_)) ?_
    · simp only [rhoDep, ha, false_and]
    · rw [Claims.Pr_false]; positivity

/-! ## The bad coins -/

/-- **The bad coins** of the masked verifier: a zerocheck outer coin is `1`, an inner `β` is `0`, a pair of
level-0 coins is dependent, or a table's rank checks refuse at coins inside the verifier's. -/
def BadZk (x : Claims.Msg) {k : ℕ} (r : Fin k → Coin) : Prop :=
  (∃ p ∈ outerSlotsZk x, (Claims.coinSeq r p.1).at p.2 = 1) ∨
    (∃ b ∈ betaSlotsZk x, b < k ∧ (Claims.coinSeq r b).at 0 = 0) ∨ (∃ a ∈ rhoSlotsZk x, rhoDep r a) ∨
    ∃ s ∈ rankSlotsZk x, s.2.2 + (s.1.m + s.1.c.kLog) ≤ k ∧ ¬ MaskRank.MaskRankAt s.1 r s.2.1 s.2.2

theorem pr_list_le {k : ℕ} {β : Type} (E : β → (Fin k → Coin) → Prop) (L : List β) (c : ℝ)
    (h : ∀ b ∈ L, Claims.Pr (E b) ≤ c) : Claims.Pr (fun r => ∃ b ∈ L, E b r) ≤ L.length * c := by
  refine le_trans (Pr_exists_le E L) (le_trans (List.sum_le_length_nsmul _ c ?_) (le_of_eq ?_))
  · intro y hy
    obtain ⟨b, hb, rfl⟩ := List.mem_map.1 hy
    exact h b hb
  · rw [List.length_map, nsmul_eq_mul]

/-- A table's rank checks: a table of the session that sets up, rep `1`'s coins after rep `0`'s claim points. -/
theorem mem_rankSlotsZk {x : Claims.Msg} {s : Flock.Stmt × ℕ × ℕ} (hs : s ∈ rankSlotsZk x) :
    ∃ ins spec sts, setupOf true x = some (ins, spec, sts) ∧ s.1 ∈ sts ∧ s.2.1 + (s.1.m + s.1.c.kLog) ≤ s.2.2 := by
  unfold rankSlotsZk at hs
  split at hs
  · rename_i ins spec sts hx
    obtain ⟨i, hi, hs⟩ := List.mem_flatMap.1 hs
    rw [List.mem_range] at hi
    split at hs
    · rename_i v hv
      rw [List.mem_singleton] at hs
      subst hs
      refine ⟨ins, spec, sts, hx, by rw [getElem!_pos sts i hi]; exact Array.getElem_mem hi, ?_⟩
      have h : repStartZk spec sts (2 * i + 1) = repStartZk spec sts (2 * i) + repRounds true spec sts[i]! := by
        unfold repStartZk
        rw [List.range_succ, List.map_append, List.sum_append]
        simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, show 2 * i / 2 = i by omega]
        omega
      rw [h]
      unfold repRounds
      rcases v with ⟨sch, tp⟩
      simp only [hv, ite_true]
      omega
    · simp at hs
  · simp at hs

/-- **`Pr[BadZk] ≤ εZk`**: the union bound over the outer coins, the `β`s, the pairs and, at the rank bound
(`MaskRank.MaskRankBound`), each table's rank checks. -/
theorem pr_badZk_le (hMR : MaskRank.MaskRankBound) (x : Claims.Msg) {k : ℕ} :
    Claims.Pr (fun r : Fin k → Coin => BadZk x r) ≤ εZk x := by
  have ho := pr_list_le (fun (p : ℕ × ℕ) (r : Fin k → Coin) => (Claims.coinSeq r p.1).at p.2 = 1) (outerSlotsZk x)
    (1 / 2 ^ 128) fun p _ => pr_slot_le p.1 p.2
  have hb := pr_list_le (fun (b : ℕ) (r : Fin k → Coin) => b < k ∧ (Claims.coinSeq r b).at 0 = 0) (betaSlotsZk x)
    (1 / 2 ^ 128) fun b _ => pr_slot_eq_le b (by norm_num) 0
  have hr := pr_list_le (fun (a : ℕ) (r : Fin k → Coin) => rhoDep r a) (rhoSlotsZk x) (2 / 2 ^ 128)
    fun a _ => pr_rho_le a
  have hm : Claims.Pr (fun r : Fin k → Coin => ∃ s ∈ rankSlotsZk x, s.2.2 + (s.1.m + s.1.c.kLog) ≤ k ∧
      ¬ MaskRank.MaskRankAt s.1 r s.2.1 s.2.2) ≤ (((rankSlotsZk x).map fun s => 4 * s.1.m).sum : ℕ) / 2 ^ 128 := by
    refine le_trans (Pr_exists_le _ _) (le_of_le_of_eq
      (List.sum_le_sum (g := fun s : Flock.Stmt × ℕ × ℕ => (4 * s.1.m : ℝ) / 2 ^ 128) fun s hs => ?_) ?_)
    · obtain ⟨ins, spec, sts, hx, hst, hn⟩ := mem_rankSlotsZk hs
      exact hMR x ins spec sts hx s.1 hst k s.2.1 s.2.2 hn
    · simp only [div_eq_mul_inv, List.sum_map_mul_right, Nat.cast_list_sum, List.map_map]
      congr 2
      refine List.map_congr_left fun s _ => ?_
      simp
  refine le_trans (Pr_or_le _ _) (le_trans (add_le_add ho (Pr_or_le _ _)) ?_)
  refine le_trans (add_le_add le_rfl (add_le_add hb (Pr_or_le _ _))) ?_
  refine le_trans (add_le_add le_rfl (add_le_add le_rfl (add_le_add hr hm))) (le_of_eq ?_)
  unfold εZk
  ring

/-! ## The composition -/

/-- **Completeness from acceptance at the good coins**, at the masked verifier: if, on every statement in scope that it
can accept and every witness in the relation, the honest masked prover is accepted at every tape and every coin outside
`BadZk`, then Flock's masked verifier is complete for `honestPZk` with error `δcZk H`. -/
theorem complete_zk_of (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation)
    (hMR : MaskRank.MaskRankBound)
    (hacc : ∀ x w, Claims.Flock.Scope (deployZk rel) x → Claims.Flock.R (deployZk rel).parse x w →
      readyZk H x = true → ∀ (ω : TapeZk) (r : Fin ((V true).k x) → Coin), ¬ BadZk x r →
        (Claims.run H (honestPZk w) (V true) x ω r).out = .acc) :
    CompleteWith H (deployZk rel) honestPZk (δcZk H) := by
  intro _ x w hs hr ω
  unfold δcZk
  split
  · rename_i hready
    calc 1 - εZk x ≤ 1 - Claims.Pr (fun r : Fin ((V true).k x) → Coin => BadZk x r) := by
          linarith [pr_badZk_le hMR x (k := (V true).k x)]
      _ = Claims.Pr (fun r : Fin ((V true).k x) → Coin => ¬ BadZk x r) := (Pr_not _).symm
      _ ≤ Claims.Pr (fun r : Fin ((V true).k x) → Coin =>
            (Claims.run H (honestPZk w) (V true) x ω r).out = .acc) :=
          Claims.Pr_mono fun r hr' => hacc x w hs hr hready ω r hr'
  · rw [sub_self]
    exact Claims.Pr_nonneg _

end FlockComplete
