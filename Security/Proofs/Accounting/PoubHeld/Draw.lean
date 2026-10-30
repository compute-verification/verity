import Definitions.Accounting.Guarantees.PoubDraw
import Proofs.Accounting.PoubHeld.OverTime
import Proofs.Accounting.RateAudit.Horizon
import Proofs.Accounting.Prob
import Proofs.Accounting.Actual.Sampler
import Proofs.Hash.RowBinding

/-!
# The openings' pooled bound: the proof

Once the coin commitment the prover holds is fixed, window `t`'s size `U_t` and correct receipts `C_t` are fixed before
its openings coin (`Live` of `Committed.view c`), so given the other coins and the earlier windows' coins the window's
draw is `Key.subset` on the first blocks `prf t` gives a uniform coin.

* **On uniform first blocks** (`Draw.pr_test_le`). Step `i` of the partial Fisher–Yates reads its first word, uniform
  (`Draw.pr_word_le`: a first word is the top bits of a uniform block). The sampler rejects it with probability at
  most `2^-64 − 2^-512` (`Draw.pr_reject_le`, `2^w mod N < N ≤ 2^(w − 64)` and `w ≤ 512`), so some step is rejected
  with probability at most `s'_t·(2^-64 − 2^-512)` (`Draw.pr_not_accepted_le`). An accepted word gives each of the
  `U_t − i` offsets at most a `1/(U_t − i)` share (`Draw.pr_hit_le`, the wipe's `card_fiber_le`). Once the earlier
  steps took receipts of `C_t`, `g_t − i` of the positions left hold receipts of `C_t` (`Draw.card_arrange_mem`), so
  step `i` takes one with probability at most `(g_t − i)/(U_t − i) ≤ g_t/U_t`, and the draw lands in `C_t` with
  probability at most `(g_t/U_t)^(s'_t)`, `0` when `g_t < s'_t` (`Draw.pr_window_le`, the steps as rounds of
  `Draw.pr_rounds_le`, the adaptive product bound `Draw.pr_live_le`): `bound t` (`Session.bound`).
* **The pooled bound** (`Draw.prod_le_of_pooled_lt`). When the wrong receipts pass the pooled bound, the wrong shares
  `f_t = 1 − g_t/U_t` are no point of its knapsack, so a window has fewer correct receipts than draws, or
  `Σ_t s'_t·f_t` passes the budget `S·(1 − δ^(1/S))` and weighted AM–GM puts `∏_t bound t` at most `δ`.
* **The hybrid over windows** (`Draw.pr_potential_le`). The potential after `t` windows (`Session.potential`) is the
  event that windows `j < t` land on `prf j` and a uniform 512-bit threshold coin `u` falls below `P = ∏_{j<t} bound j`
  (`below`, of probability `Draw.coinWeight δ P`, `min(1, δ/P)` rounded down to a multiple of `2^-512`,
  `Draw.pr_below`). After no window it has probability at most `δ`. Window `t`'s step (`Draw.pr_potential_succ_le`,
  the arithmetic `Draw.hybrid_step`) first trades the deployed commitment for `Committed.junk com t`, whose leaf `t`
  commits zeros, at the potentials after `t + 1` and after `t` windows: that is `Committed.hide com prf δ t`. In
  the junk session windows `j ≤ t` read nothing of coin `t` (`Draw.SwapFree`, from `Live` at each fixed commitment,
  `Draw.swapFree_junk`). There window `t`'s distinguisher on `prf t` under a uniform key is the potential after
  `t + 1` windows, its key read as window `t`'s coin (`Draw.pr_potential_succ`, by `Draw.swapCoin`), so it is at most
  its probability on uniform blocks plus its advantage. On uniform
  blocks it is at most the potential after `t` windows plus `s_t/2^64` (`Draw.pr_distinguisher_le`): given the
  earlier coins, window `t` lands with probability at most `s'_t·(2^-64 − 2^-512) + bound t`, independently of `u`
  falling below `P·bound t`, and `coinWeight δ (P·bound t)·(s'_t·(2^-64 − 2^-512) + bound t)` is at most
  `coinWeight δ P + s_t/2^64` (`Draw.window_step_le`: rounding costs at most `2^-512`, which the sharper rejection
  bound pays when `s'_t ≥ 1`; a window of no draws has bound `1`).
* **On the coin** (`Draw.pooledDraw`). Off the binding event, where no opened receipt was presented with an accepted
  row other than the one its root holds, a receipt `Refused` lets through is correct (`Draw.opened_sub`); the binding
  event costs its probability, `Receipts.binding`. An accepted draw ignores the fallbacks (`Draw.openings_congr`), so on an accepting outcome past
  the pooled bound and off the binding event every window lands on `prf t` while `∏_t bound t ≤ δ`, where `u` always
  falls below: the potential after `n` windows (`Draw.pr_landed_le`). `PooledDraw` is that bound, `δ` plus each
  window's `s_t/2^64`, its distinguisher's advantage at the junk commitment and its hiding term, plus the binding term
  (`Draw.pooledDraw`, `Draw.error_toReal`); `LiveCoinsDrawn` and the memory line at the record's pooled bound
  (`HeldDrawn` and its time-held forms) follow from it as `Held` does from `LiveCoins`.

**What the term is.** Window `t`'s term, `prfAdvantage (prf t) ((F.play (Committed.junk com t)).distinguisher prf δ t)`,
is one distinguisher's advantage, fixed by the session: it reads at most `s_t` first blocks of window `t`'s stream
(`Draw.distinguisher_congr`) and otherwise runs the session's own windows at a commitment that reads no bit of coin
`t`, so efficient provers and verifiers give an efficient distinguisher with `s_t` queries, C3's form. An honest
session, every receipt correct, gains nothing (`Draw.prfAdvantage_honest`). The
term it replaces, the supremum of the advantage over every window test `test U k C`, was at least `1 − 2^-43` whatever
the session (at `U = 2^300`, `k = 1`, `C` the first opening under each of the `2^256` keys, which the
test lands in on `prf` always and on uniform blocks with probability at most `2^-44 + 2^-64`). That test is a window's
distinguisher only in a session whose correct receipts are those `2^256` openings, which a prover can't compute; it
passes, and C3 bounds no such distinguisher.
-/

namespace Accounting.SecurityProofs.Poub

open Finset Accounting.Rate Accounting.Rate.Horizon
open Pous (pr)
open scoped ENNReal
open Accounting.Poub.Held Accounting.Poub.Held.Assumptions Accounting.Poub.Draw
  Accounting.Poub.Draw.Assumptions

namespace Draw

/-! ## Adaptive rounds -/

/-- Rounds `0, …, n − 1`, each with coins `Y i` and then `Z i`. Round `i`'s event `A i` reads the coins of rounds up to
`i`, and `h i`, read off the earlier rounds and round `i`'s first coins, bounds the probability of `A i` over its second
coins. Then every event holds while the bounds multiply to at most `c` with probability at most `c`. -/
theorem pr_live_le : ∀ (n : ℕ) (Y Z : Fin n → Type) [∀ i, Fintype (Y i)] [∀ i, Fintype (Z i)]
    (A : (i : Fin n) → ((j : Fin n) → Y j × Z j) → Prop) (h : (i : Fin n) → ((j : Fin n) → Y j × Z j) → ℝ≥0∞),
    (∀ i ω ω', (∀ j, j ≤ i → ω j = ω' j) → (A i ω ↔ A i ω')) →
    (∀ i ω ω', (∀ j, j < i → ω j = ω' j) → (ω i).1 = (ω' i).1 → h i ω = h i ω') →
    (∀ i ω, pr (fun z : Z i => A i (Function.update ω i ((ω i).1, z))) ≤ h i ω) →
    ∀ c, pr (fun ω => (∀ i, A i ω) ∧ ∏ i, h i ω ≤ c) ≤ c := by
  intro n
  induction n with
  | zero =>
    intro Y Z _ _ A h _ _ _ c
    by_cases hc : 1 ≤ c
    · exact (pr_le_one _).trans hc
    · refine (Bandwidth.pr_mono (fun ω hω => (hc (by simpa using hω.2) : False))).trans ?_
      rw [pr_false]
      exact zero_le
  | succ n ih =>
    intro Y Z _ _ A h hA hh hb c
    classical
    let T : Type := (i : Fin n) → Y i.succ × Z i.succ
    let e : ((j : Fin (n + 1)) → Y j × Z j) ≃ (Z 0 × T) × Y 0 :=
      { toFun := fun ω => (((ω 0).2, Fin.tail ω), (ω 0).1)
        invFun := fun q => Fin.cons (q.2, q.1.1) q.1.2
        left_inv := fun ω => by simp
        right_inv := fun q => by simp }
    rw [pr_equiv e]
    refine pr_le_of_fiber _ _ (fun y₀ => ?_)
    simp only [e, Equiv.coe_fn_symm_mk]
    rcases isEmpty_or_nonempty (Z 0 × T) with hE | ⟨⟨z₀, t₀⟩⟩
    · exact (le_of_eq (Bandwidth.pr_eq_zero_of_card _ Fintype.card_eq_zero)).trans zero_le
    set H := h 0 (Fin.cons (y₀, z₀) t₀ : (j : Fin (n + 1)) → Y j × Z j) with hHdef
    have hH : ∀ z t, h 0 (Fin.cons (y₀, z) t : (j : Fin (n + 1)) → Y j × Z j) = H :=
      fun z t => hh 0 _ _ (fun j hj => absurd hj (Fin.not_lt_zero j)) rfl
    let A₀ : Z 0 → Prop := fun z => A 0 (Fin.cons (y₀, z) t₀ : (j : Fin (n + 1)) → Y j × Z j)
    have hA₀ : ∀ z t, A 0 (Fin.cons (y₀, z) t : (j : Fin (n + 1)) → Y j × Z j) ↔ A₀ z := fun z t =>
      hA 0 _ _ (fun j hj => by
        obtain rfl : j = 0 := Fin.le_zero_iff.1 hj
        rfl)
    have hpA : pr A₀ ≤ H := by
      have := hb 0 (Fin.cons (y₀, z₀) t₀)
      simpa [Fin.update_cons_zero, A₀] using this
    let A' : Z 0 → (i : Fin n) → T → Prop := fun z i t => A i.succ (Fin.cons (y₀, z) t)
    let h' : Z 0 → (i : Fin n) → T → ℝ≥0∞ := fun z i t => h i.succ (Fin.cons (y₀, z) t)
    have hrest : ∀ z c', pr (fun t : T => (∀ i, A' z i t) ∧ ∏ i, h' z i t ≤ c') ≤ c' := by
      intro z
      refine ih (fun i => Y i.succ) (fun i => Z i.succ) (A' z) (h' z) ?_ ?_ ?_
      · intro i ω ω' hω
        refine hA i.succ _ _ (fun j hj => ?_)
        cases j using Fin.cases with
        | zero => rfl
        | succ j => simpa using hω j (Fin.succ_le_succ_iff.1 hj)
      · intro i ω ω' hω h1
        refine hh i.succ _ _ (fun j hj => ?_) (by simpa using h1)
        cases j using Fin.cases with
        | zero => rfl
        | succ j => simpa using hω j (Fin.succ_lt_succ_iff.1 hj)
      · intro i ω
        have := hb i.succ (Fin.cons (y₀, z) ω)
        simpa [A', h', Fin.cons_update] using this
    let B : Z 0 × T → Prop := fun p => (∀ i, A' p.1 i p.2) ∧ (∏ i, h' p.1 i p.2) * H ≤ c
    refine (Bandwidth.pr_mono (fun p hp => (?_ : A₀ p.1 ∧ B p))).trans ?_
    · obtain ⟨hall, hprod⟩ := hp
      refine ⟨(hA₀ p.1 p.2).1 (hall 0), fun i => hall i.succ, ?_⟩
      rw [Fin.prod_univ_succ, hH, mul_comm] at hprod
      exact hprod
    by_cases hc : c = ⊤
    · rw [hc]; exact le_top
    by_cases hH0 : H = 0
    · have hA0 : pr A₀ = 0 := le_antisymm (hpA.trans hH0.le) zero_le
      refine (pr_and_le_mul A₀ B ⊤ (fun _ _ => le_top)).trans ?_
      rw [hA0, zero_mul]
      exact zero_le
    refine (pr_and_le_mul A₀ B (c / H) (fun z _ => ?_)).trans ?_
    · refine (Bandwidth.pr_mono (fun t ht => ⟨ht.1, ?_⟩)).trans (hrest z (c / H))
      exact (ENNReal.le_div_iff_mul_le (Or.inl hH0) (Or.inr hc)).2 ht.2
    calc pr A₀ * (c / H) ≤ H * (c / H) := by gcongr
      _ ≤ c := ENNReal.mul_div_le

/-! ## One coordinate -/

/-- An event that reads one coordinate of a product has at most that coordinate's probability. -/
theorem pr_coord_le {ι : Type} [Fintype ι] [DecidableEq ι] {β : ι → Type} [∀ i, Fintype (β i)] (i : ι)
    (P : β i → Prop) : pr (fun z : (j : ι) → β j => P (z i)) ≤ pr P := by
  rw [pr_equiv (Equiv.piSplitAt i β)]
  refine pr_le_of_fiber _ _ (fun y => ?_)
  simp [Equiv.piSplitAt_symm_apply]

/-! ## The first word -/

theorem width_le {N : ℕ} (h : N < 2 ^ 448) : width N ≤ 512 := by
  have : Nat.size N ≤ 448 := Nat.size_le.2 h
  unfold width
  omega

theorem mul_le_width (N : ℕ) : N * 2 ^ 64 ≤ 2 ^ width N := by
  have h1 : N < 2 ^ Nat.size N := Nat.lt_size_self N
  have h2 : Nat.size N + 64 ≤ width N := by unfold width; omega
  calc N * 2 ^ 64 ≤ 2 ^ Nat.size N * 2 ^ 64 := Nat.mul_le_mul_right _ h1.le
    _ = 2 ^ (Nat.size N + 64) := by rw [pow_add]
    _ ≤ 2 ^ width N := Nat.pow_le_pow_right (by norm_num) h2

theorem bitsVal_word (N : ℕ) {L : ℕ} (b : Pous.Bits L) :
    Pous.bitsVal (word N b) = Pous.bitsVal b / 2 ^ (L - width N) := by
  simp [word, Pous.bitsVal]

open Classical in
/-- **A first word has at most `2^(512 − w)` blocks.** -/
theorem card_word_fiber_le (N : ℕ) {L : ℕ} (u : Pous.Bits (width N)) :
    (univ.filter fun z : Pous.Bits L => word N z = u).card ≤ 2 ^ (L - width N) := by
  have hinj : Set.InjOn (fun z : Pous.Bits L => Pous.bitsVal z % 2 ^ (L - width N))
      ↑(univ.filter fun z : Pous.Bits L => word N z = u) := by
    intro z hz z' hz' h
    simp only [coe_filter, mem_univ, true_and, Set.mem_ofPred_eq] at hz hz'
    have e : Pous.bitsVal z / 2 ^ (L - width N) = Pous.bitsVal z' / 2 ^ (L - width N) := by
      rw [← bitsVal_word, ← bitsVal_word, hz, hz']
    apply Actual.bitsVal_injective L
    rw [← Nat.div_add_mod (Pous.bitsVal z) (2 ^ (L - width N)),
      ← Nat.div_add_mod (Pous.bitsVal z') (2 ^ (L - width N)), e]
    exact congrArg _ h
  calc _ ≤ (range (2 ^ (L - width N))).card :=
        card_le_card_of_injOn _ (fun z _ => mem_range.2 (Nat.mod_lt _ (by positivity))) hinj
    _ = _ := card_range _

open Classical in
/-- **The first word is uniform**: an event of the first word has at most its probability over uniform words. -/
theorem pr_word_le {N L : ℕ} (hw : width N ≤ L) (P : Pous.Bits (width N) → Prop) :
    pr (fun z : Pous.Bits L => P (word N z)) ≤ pr P := by
  have hc : (univ.filter fun z : Pous.Bits L => P (word N z)).card ≤
      (univ.filter P).card * 2 ^ (L - width N) := by
    calc _ = ∑ u ∈ univ.filter P,
          ((univ.filter fun z : Pous.Bits L => P (word N z)).filter fun z => word N z = u).card :=
          card_eq_sum_card_fiberwise (fun z hz => mem_filter.2 ⟨mem_univ _, (mem_filter.1 hz).2⟩)
      _ ≤ ∑ _u ∈ univ.filter P, 2 ^ (L - width N) := sum_le_sum fun u _ =>
          (card_le_card fun z hz => by
            simp only [mem_filter, mem_univ, true_and] at hz ⊢
            exact hz.2).trans (card_word_fiber_le N u)
      _ = _ := by rw [sum_const, smul_eq_mul]
  refine (Actual.pr_le_ofReal _ (by positivity) ?_).trans (Actual.ofReal_le_pr P)
  have h512 : ((2 : ℝ) ^ L) = 2 ^ width N * 2 ^ (L - width N) := by
    rw [← pow_add, Nat.add_sub_cancel' hw]
  simp only [Fintype.card_fun, Fintype.card_bool, Fintype.card_fin, Nat.cast_pow, Nat.cast_ofNat]
  rw [h512, div_mul_eq_mul_div, mul_div_assoc, mul_div_cancel_left₀ _ (by positivity)]
  exact_mod_cast hc

/-- **The sampler rejects a first word of an `L`-bit block with probability at most `2^-64 − 2^-L`.** -/
theorem pr_reject_le {N L : ℕ} (h0 : 0 < N) (hw : width N ≤ L) :
    pr (fun z : Pous.Bits L => Actual.wordIndex N (word N z) = none) ≤ ENNReal.ofReal (1 / 2 ^ 64 - 1 / 2 ^ L) := by
  classical
  refine (pr_word_le hw (fun u => Actual.wordIndex N u = none)).trans ?_
  have hc : (univ.filter fun u : Pous.Bits (width N) => Actual.wordIndex N u = none).card = 2 ^ width N % N := by
    have h1 := card_filter_add_card_filter_not (s := (univ : Finset (Pous.Bits (width N))))
      (fun u : Pous.Bits (width N) => (Actual.wordIndex N u).isSome)
    rw [Actual.card_accepted, card_univ, Fintype.card_fun, Fintype.card_bool, Fintype.card_fin] at h1
    have h2 : (univ.filter fun u : Pous.Bits (width N) => ¬ (Actual.wordIndex N u).isSome).card =
        (univ.filter fun u : Pous.Bits (width N) => Actual.wordIndex N u = none).card := by
      congr 1
      ext u
      simp
    have h3 : 2 ^ width N % N ≤ 2 ^ width N := Nat.mod_le _ _
    omega
  have h64 : 64 ≤ width N := by unfold width; omega
  have hβ : (0 : ℝ) ≤ 1 / 2 ^ 64 - 1 / 2 ^ L :=
    sub_nonneg.2 (one_div_le_one_div_of_le (by positivity) (pow_le_pow_right₀ (by norm_num) (h64.trans hw)))
  have key : ((univ.filter fun u : Pous.Bits (width N) => Actual.wordIndex N u = none).card : ℝ) ≤
      (1 / 2 ^ 64 - 1 / 2 ^ L) * Fintype.card (Pous.Bits (width N)) := by
    rw [hc]
    simp only [Fintype.card_fun, Fintype.card_bool, Fintype.card_fin, Nat.cast_pow, Nat.cast_ofNat]
    have hm : 2 ^ width N % N + 1 ≤ N := Nat.mod_lt _ h0
    have h1 : ((2 ^ width N % N : ℕ) : ℝ) + 1 ≤ N := by exact_mod_cast hm
    have h2 : (N : ℝ) * 2 ^ 64 ≤ 2 ^ width N := by exact_mod_cast mul_le_width N
    have h3 : (2 : ℝ) ^ width N / 2 ^ L ≤ 1 := div_le_one_of_le₀ (pow_le_pow_right₀ (by norm_num) hw) (by positivity)
    have h4 : (N : ℝ) ≤ 2 ^ width N / 2 ^ 64 := (le_div_iff₀ (by positivity)).2 h2
    rw [sub_mul, one_div_mul_eq_div, one_div_mul_eq_div]
    linarith
  refine Actual.pr_le_ofReal _ hβ ?_
  convert key using 3
  ext u
  simp

/-- **Each offset has at most a `1/N` share**: the first word gives an offset in `P` with probability at most
`|P ∩ [0, N)| / N`. -/
theorem pr_hit_le {N L : ℕ} (hw : width N ≤ L) (P : ℕ → Prop) [DecidablePred P] :
    pr (fun z : Pous.Bits L => ∃ o : Fin N, Actual.wordIndex N (word N z) = some o ∧ P o) ≤
      ENNReal.ofReal (((range N).filter P).card / N) := by
  classical
  refine (pr_word_le hw (fun u => ∃ o : Fin N, Actual.wordIndex N u = some o ∧ P o)).trans ?_
  have hsub : (univ.filter fun u : Pous.Bits (width N) => ∃ o : Fin N, Actual.wordIndex N u = some o ∧ P o) ⊆
      (univ.filter fun o : Fin N => P o).biUnion
        (fun o => univ.filter fun u : Pous.Bits (width N) => Actual.wordIndex N u = some o) := by
    intro u hu
    simp only [mem_filter, mem_univ, true_and, mem_biUnion] at hu ⊢
    obtain ⟨o, h1, h2⟩ := hu
    exact ⟨o, h2, h1⟩
  have hP : (univ.filter fun o : Fin N => P o).card = ((range N).filter P).card := by
    rw [card_filter, card_filter, Fin.sum_univ_eq_sum_range (fun o => if P o then 1 else 0) N]
  have hc : (univ.filter fun u : Pous.Bits (width N) => ∃ o : Fin N, Actual.wordIndex N u = some o ∧ P o).card ≤
      ((range N).filter P).card * (2 ^ width N / N) := by
    refine (card_le_card hsub).trans ((card_biUnion_le).trans ?_)
    rw [← hP, ← smul_eq_mul, ← sum_const]
    exact sum_le_sum fun o _ => Actual.card_fiber_le o
  have key : ((univ.filter fun u : Pous.Bits (width N) =>
      ∃ o : Fin N, Actual.wordIndex N u = some o ∧ P o).card : ℝ) ≤
      (((range N).filter P).card / N : ℝ) * Fintype.card (Pous.Bits (width N)) := by
    simp only [Fintype.card_fun, Fintype.card_bool, Fintype.card_fin, Nat.cast_pow, Nat.cast_ofNat]
    calc (((univ.filter fun u : Pous.Bits (width N) => ∃ o : Fin N,
            Actual.wordIndex N u = some o ∧ P o).card : ℕ) : ℝ)
        ≤ (((range N).filter P).card : ℝ) * ((2 ^ width N / N : ℕ) : ℝ) := by exact_mod_cast hc
      _ ≤ (((range N).filter P).card : ℝ) * ((2 : ℝ) ^ width N / N) := by
          gcongr
          exact_mod_cast Nat.cast_div_le
      _ = (((range N).filter P).card / N : ℝ) * 2 ^ width N := by ring
  refine Actual.pr_le_ofReal _ (by positivity) ?_
  convert key using 3
  ext u
  simp

/-! ## The arrangement -/

theorem arrange_succ (j : ℕ → ℕ) (i : ℕ) : arrange j (i + 1) = arrange j i * Equiv.swap i (i + j i) := rfl

/-- Later swaps leave the earlier positions alone. -/
theorem arrange_fix (j : ℕ → ℕ) {p i : ℕ} (hp : p < i) : ∀ i', i ≤ i' → arrange j i' p = arrange j i p := by
  intro i' hi
  induction i', hi using Nat.le_induction with
  | base => rfl
  | succ i' hi ih =>
    rw [arrange_succ, Equiv.Perm.mul_apply, Equiv.swap_apply_of_ne_of_ne (by omega) (by omega), ih]

theorem arrange_self (j : ℕ → ℕ) (i : ℕ) : arrange j (i + 1) i = arrange j i (i + j i) := by
  rw [arrange_succ, Equiv.Perm.mul_apply, Equiv.swap_apply_left]

theorem arrange_congr {j j' : ℕ → ℕ} : ∀ {k : ℕ}, (∀ l, l < k → j l = j' l) → arrange j k = arrange j' k
  | 0, _ => rfl
  | k + 1, h => by
    rw [arrange_succ, arrange_succ, arrange_congr (fun l hl => h l (by omega)), h k (by omega)]

/-- Swaps inside `[0, U)` keep `[0, U)`. -/
theorem arrange_lt {j : ℕ → ℕ} {U : ℕ} : ∀ {i : ℕ}, i ≤ U → (∀ l, l < i → l + j l < U) →
    ∀ p, p < U → arrange j i p < U
  | 0, _, _, p, hp => hp
  | i + 1, hi, hj, p, hp => by
    rw [arrange_succ, Equiv.Perm.mul_apply]
    refine arrange_lt (by omega) (fun l hl => hj l (by omega)) _ ?_
    rw [Equiv.swap_apply_def]
    have := hj i (by omega)
    split_ifs <;> omega

theorem image_arrange {j : ℕ → ℕ} {U i : ℕ} (hi : i ≤ U) (hj : ∀ l, l < i → l + j l < U) :
    (range U).image (arrange j i) = range U := by
  refine eq_of_subset_of_card_le (fun q hq => ?_) ?_
  · obtain ⟨p, hp, rfl⟩ := mem_image.1 hq
    exact mem_range.2 (arrange_lt hi hj p (mem_range.1 hp))
  · rw [card_image_of_injective _ (Equiv.injective _)]

/-- The positions of `[0, U)` holding receipts of `C` number `|C ∩ [0, U)|`. -/
theorem card_arrange_mem {j : ℕ → ℕ} {U i : ℕ} (hi : i ≤ U) (hj : ∀ l, l < i → l + j l < U) (C : Finset ℕ) :
    ((range U).filter fun p => arrange j i p ∈ C).card = ((range U).filter (· ∈ C)).card := by
  calc ((range U).filter fun p => arrange j i p ∈ C).card
      = (((range U).filter fun p => arrange j i p ∈ C).image (arrange j i)).card :=
        (card_image_of_injective _ (Equiv.injective _)).symm
    _ = (((range U).image (arrange j i)).filter (· ∈ C)).card := by rw [filter_image]
    _ = _ := by rw [image_arrange hi hj]

/-! ## One window's draw on uniform first blocks -/

theorem card_split (U i : ℕ) (hi : i ≤ U) (Q : ℕ → Prop) [DecidablePred Q] :
    ((range U).filter Q).card = ((range i).filter Q).card + ((range (U - i)).filter fun o => Q (i + o)).card := by
  rw [card_filter, card_filter, card_filter, ← sum_range_add_sum_Ico _ hi, sum_Ico_eq_sum_range]

/-- **Rounds of uniform coins**: the adaptive product bound `pr_live_le` with no first coins. -/
theorem pr_rounds_le (n : ℕ) (Z : Fin n → Type) [∀ i, Fintype (Z i)]
    (A : (i : Fin n) → ((j : Fin n) → Z j) → Prop) (h : (i : Fin n) → ((j : Fin n) → Z j) → ℝ≥0∞)
    (hA : ∀ i ω ω', (∀ j, j ≤ i → ω j = ω' j) → (A i ω ↔ A i ω'))
    (hh : ∀ i ω ω', (∀ j, j < i → ω j = ω' j) → h i ω = h i ω')
    (hb : ∀ i ω, pr (fun z : Z i => A i (Function.update ω i z)) ≤ h i ω) (c : ℝ≥0∞) :
    pr (fun ω => (∀ i, A i ω) ∧ ∏ i, h i ω ≤ c) ≤ c := by
  classical
  let e : ((j : Fin n) → Unit × Z j) ≃ ((j : Fin n) → Z j) :=
    Equiv.piCongrRight fun j => Equiv.punitProd (Z j)
  have he : ∀ ω j, e ω j = (ω j).2 := fun _ _ => rfl
  rw [pr_equiv e.symm]
  simp only [Equiv.symm_symm]
  refine pr_live_le n (fun _ => Unit) Z (fun i ω => A i (e ω)) (fun i ω => h i (e ω)) ?_ ?_ ?_ c
  · intro i ω ω' hω
    exact hA i _ _ (fun j hj => by rw [he, he, hω j hj])
  · intro i ω ω' hω _
    exact hh i _ _ (fun j hj => by rw [he, he, hω j hj])
  · intro i ω
    have hu : ∀ z : Z i, e (Function.update ω i ((ω i).1, z)) = Function.update (e ω) i z := by
      intro z
      funext j
      by_cases hj : j = i
      · subst hj
        simp [he]
      · simp [he, Function.update_of_ne hj]
    simp only [hu]
    exact hb i (e ω)

theorem offsets_lt {m U k : ℕ} {b : Blocks m} (r : ℕ → ℕ) (hk : k ≤ m) (ha : accepted U k b) :
    ∀ l, l < k → l + offsets U b r l < U := by
  intro l hl
  have hlm : l < m := by omega
  have h := ha ⟨l, hlm⟩ hl
  simp only at h
  obtain ⟨o, ho⟩ := Option.isSome_iff_exists.1 h
  have hU : l < U := by
    by_contra hU
    have : o.val < U - l := o.isLt
    omega
  simp only [offsets, hlm, ↓reduceDIte, offset, ho, Option.map_some, Option.getD_some]
  have : o.val < U - l := o.isLt
  omega

theorem offsets_congr {m U k : ℕ} {b : Blocks m} (r r' : ℕ → ℕ) (hk : k ≤ m) (ha : accepted U k b) :
    ∀ l, l < k → offsets U b r l = offsets U b r' l := by
  intro l hl
  have hlm : l < m := by omega
  have h := ha ⟨l, hlm⟩ hl
  simp only at h
  obtain ⟨o, ho⟩ := Option.isSome_iff_exists.1 h
  simp only [offsets, hlm, ↓reduceDIte, offset, ho, Option.map_some, Option.getD_some]

theorem openings_congr {m U k : ℕ} {b : Blocks m} (r r' : ℕ → ℕ) (hk : k ≤ m) (ha : accepted U k b) :
    openings U k b r = openings U k b r' := by
  unfold openings
  rw [arrange_congr (offsets_congr r r' hk ha)]

open Classical in
/-- **One window's draw lands in `C`** with probability at most `(g/U)^k`, `g = |C ∩ [0, U)|`, and never when
`g < k`: step `i` takes one of the `U − i` positions left, `g − i` of which hold receipts of `C` once the earlier steps
took receipts of `C`, each with at most a `1/(U − i)` share. -/
theorem pr_window_le {m U k : ℕ} (hm : m < 2 ^ 448) (hU : U ≤ m) (hk : k ≤ U) (C : Finset ℕ) (r : ℕ → ℕ) :
    pr (fun b : Blocks m => accepted U k b ∧ openings U k b r ⊆ C) ≤
      ENNReal.ofReal (if k ≤ ((range U).filter (· ∈ C)).card
        then ((((range U).filter (· ∈ C)).card : ℝ) / U) ^ k else 0) := by
  set g := ((range U).filter (· ∈ C)).card with hg
  let σ : Blocks m → ℕ → Equiv.Perm ℕ := fun b => arrange (offsets U b r)
  let cnt : Blocks m → ℕ → ℕ := fun b i => ((range (U - i)).filter fun o => σ b i (i + o) ∈ C).card
  let A : (i : Fin m) → Blocks m → Prop := fun i b =>
    (i : ℕ) < k → (Actual.wordIndex (U - i) (word (U - i) (b i))).isSome = true ∧ σ b (i + 1) i ∈ C
  let h : (i : Fin m) → Blocks m → ℝ≥0∞ := fun i b =>
    if (i : ℕ) < k then ENNReal.ofReal ((cnt b i : ℝ) / ((U - i : ℕ) : ℝ)) else 1
  have hoff : ∀ (b b' : Blocks m) (i : ℕ), (∀ l : Fin m, (l : ℕ) < i → b l = b' l) →
      ∀ l, l < i → offsets U b r l = offsets U b' r l := by
    intro b b' i hb l hl
    unfold offsets
    split_ifs with hlm
    · rw [hb ⟨l, hlm⟩ hl]
    · rfl
  have hσ : ∀ (b b' : Blocks m) (i : ℕ), (∀ l : Fin m, (l : ℕ) < i → b l = b' l) → σ b i = σ b' i :=
    fun b b' i hb => arrange_congr (hoff b b' i hb)
  have hkm : k ≤ m := hk.trans hU
  refine (Bandwidth.pr_mono (fun b hb => (?_ : (∀ i, A i b) ∧ ∏ i, h i b ≤ _))).trans
    (pr_rounds_le m (fun _ => Pous.Bits 512) A h ?_ ?_ ?_ _)
  · obtain ⟨ha, hsub⟩ := hb
    have hA : ∀ i, A i b := by
      intro i hi
      refine ⟨ha i hi, hsub ?_⟩
      refine mem_image.2 ⟨i, mem_range.2 hi, ?_⟩
      exact arrange_fix _ (Nat.lt_succ_self _) k hi
    refine ⟨hA, ?_⟩
    have hlt := offsets_lt (b := b) r hkm ha
    have hcnt : ∀ i, i ≤ k → cnt b i + i = g := by
      intro i hi
      have h1 := card_split U i (hi.trans hk) (fun p => σ b i p ∈ C)
      rw [show ((range U).filter fun p => σ b i p ∈ C).card = g from
        card_arrange_mem (hi.trans hk) (fun l hl => hlt l (by omega)) C] at h1
      have h2 : ((range i).filter fun p => σ b i p ∈ C) = range i := by
        refine filter_true_of_mem fun p hp => ?_
        have hp' := mem_range.1 hp
        have := (hA ⟨p, by omega⟩ (by simp only; omega)).2
        simp only at this
        rwa [show σ b i p = σ b (p + 1) p from arrange_fix _ (Nat.lt_succ_self _) i (by omega)]
      rw [h2, card_range] at h1
      simp only [cnt]
      omega
    have hkg : k ≤ g := by have := hcnt k le_rfl; omega
    simp only [hkg, ↓reduceIte]
    have hh : ∀ i : Fin m, h i b = ENNReal.ofReal (if (i : ℕ) < k then ((g - i : ℕ) : ℝ) / ((U - i : ℕ) : ℝ) else 1) := by
      intro i
      simp only [h]
      split_ifs with hi
      · have := hcnt i hi.le
        rw [show cnt b i = g - i by omega]
      · simp
    simp only [hh]
    rw [← ENNReal.ofReal_prod_of_nonneg (fun i _ => by split_ifs <;> positivity)]
    refine ENNReal.ofReal_le_ofReal ?_
    rw [Fin.prod_univ_eq_prod_range (fun i => if i < k then ((g - i : ℕ) : ℝ) / ((U - i : ℕ) : ℝ) else 1) m,
      ← prod_range_mul_prod_Ico _ hkm]
    have hIco : ∏ i ∈ Ico k m, (if i < k then ((g - i : ℕ) : ℝ) / ((U - i : ℕ) : ℝ) else 1) = 1 :=
      prod_eq_one fun i hi => by simp [not_lt.2 (mem_Ico.1 hi).1]
    rw [hIco, mul_one]
    refine (prod_le_prod₀ (g := fun _ => (g : ℝ) / U) (fun i _ => by split_ifs <;> positivity)
      (fun i hi => ?_)).trans (le_of_eq (by rw [prod_const, card_range]))
    have hik := mem_range.1 hi
    simp only [hik, ↓reduceIte]
    have hgU : g ≤ U := by
      have := card_le_card (filter_subset (· ∈ C) (range U))
      rwa [card_range] at this
    rw [Nat.cast_sub (by omega), Nat.cast_sub (by omega)]
    have hU0 : (0 : ℝ) < U - i := by
      have : (i : ℝ) < U := by exact_mod_cast (show i < U by omega)
      linarith
    have hU1 : (0 : ℝ) < U := by exact_mod_cast (show 0 < U by omega)
    rw [div_le_div_iff₀ hU0 hU1]
    have : (g : ℝ) ≤ U := by exact_mod_cast hgU
    nlinarith [(Nat.cast_nonneg i : (0 : ℝ) ≤ i)]
  · intro i b b' hb
    have hbi : b i = b' i := hb i le_rfl
    have hs : σ b (i + 1) = σ b' (i + 1) := hσ b b' (i + 1) (fun l hl => hb l (by
      rw [Fin.le_def]
      omega))
    simp only [A, hbi, hs]
  · intro i b b' hb
    have hs : σ b i = σ b' i := hσ b b' i (fun l hl => hb l (by
      rw [Fin.lt_def]
      exact hl))
    simp only [h, cnt, hs]
  · intro i b
    by_cases hi : (i : ℕ) < k
    swap
    · simp only [h, hi, ↓reduceIte]
      exact pr_le_one _
    simp only [h, hi, ↓reduceIte]
    have hiU : (i : ℕ) < U := by omega
    have hs : ∀ z, σ (Function.update b i z) i = σ b i := fun z =>
      hσ _ _ i (fun l hl => Function.update_of_ne (Fin.ne_of_lt (by rw [Fin.lt_def]; exact hl)) _ _)
    refine (Bandwidth.pr_mono (fun z hz => (?_ : ∃ o : Fin (U - i),
      Actual.wordIndex (U - i) (word (U - i) z) = some o ∧ σ b i (i + o) ∈ C))).trans
      (pr_hit_le (width_le ((Nat.sub_le U i).trans_lt (hU.trans_lt hm))) (fun o => σ b i (i + o) ∈ C))
    obtain ⟨h1, h2⟩ := hz hi
    simp only [Function.update_self] at h1
    obtain ⟨o, ho⟩ := Option.isSome_iff_exists.1 h1
    refine ⟨o, ho, ?_⟩
    have e1 : σ (Function.update b i z) (i + 1) i = σ b i (i + offset (U - i) (r i) z) := by
      simp only [σ]
      rw [arrange_self]
      have := hs z
      simp only [σ] at this
      rw [this]
      simp [offsets, i.isLt]
    rw [e1] at h2
    simpa [offset, ho] using h2


open Classical in
/-- A window's draw is rejected somewhere with probability at most `k·(2^-64 − 2^-512)`. -/
theorem pr_not_accepted_le {m U k : ℕ} (hm : m < 2 ^ 448) (hU : U ≤ m) (hk : k ≤ U) :
    pr (fun b : Blocks m => ¬ accepted U k b) ≤ ENNReal.ofReal (k * (1 / 2 ^ 64 - 1 / 2 ^ 512)) := by
  refine (Bandwidth.pr_mono (fun b hb => (?_ : ∃ i : Fin m,
    (i : ℕ) < k ∧ Actual.wordIndex (U - i) (word (U - i) (b i)) = none))).trans
    ((Bandwidth.pr_exists_le _).trans ?_)
  · unfold accepted at hb
    push Not at hb
    obtain ⟨i, hi, h⟩ := hb
    exact ⟨i, hi, by simpa using h⟩
  calc ∑ i : Fin m, pr (fun b : Blocks m => (i : ℕ) < k ∧ Actual.wordIndex (U - i) (word (U - i) (b i)) = none)
      ≤ ∑ i : Fin m, (if (i : ℕ) < k then ENNReal.ofReal (1 / 2 ^ 64 - 1 / 2 ^ 512) else 0) := sum_le_sum fun i _ => by
        split_ifs with hi
        · refine (Bandwidth.pr_mono (fun b hb => hb.2)).trans ?_
          refine (pr_coord_le (β := fun _ : Fin m => Pous.Bits 512) i
            (fun z => Actual.wordIndex (U - i) (word (U - i) z) = none)).trans ?_
          exact pr_reject_le (by omega) (width_le (by omega))
        · exact (Bandwidth.pr_mono (Q := fun _ => False) (fun b hb => absurd hb.1 hi)).trans (le_of_eq pr_false)
    _ = ((univ.filter fun i : Fin m => (i : ℕ) < k).card : ℝ≥0∞) * ENNReal.ofReal (1 / 2 ^ 64 - 1 / 2 ^ 512) := by
        rw [sum_ite, sum_const_zero, add_zero, sum_const, nsmul_eq_mul]
    _ ≤ (k : ℝ≥0∞) * ENNReal.ofReal (1 / 2 ^ 64 - 1 / 2 ^ 512) := by
        gcongr
        exact_mod_cast (card_le_card_of_injOn (fun i : Fin m => (i : ℕ))
          (fun i hi => mem_range.2 (mem_filter.1 hi).2) (fun a _ b _ h => Fin.ext h)).trans_eq (card_range k)
    _ = ENNReal.ofReal (k * (1 / 2 ^ 64 - 1 / 2 ^ 512)) := by
        rw [← ENNReal.ofReal_natCast, ← ENNReal.ofReal_mul (by positivity)]

open Classical in
/-- **A window test on uniform first blocks** outputs true with probability at most `k·(2^-64 − 2^-512)` (some first
word rejected) plus `(g/U)^k`, `g = |C ∩ [0, U)|`, the latter `0` when `g < k`. -/
theorem pr_test_le {m U k : ℕ} (hm : m < 2 ^ 448) (hU : U ≤ m) (hk : k ≤ U) (C : Finset ℕ) :
    pr (test (m := m) U k C) ≤ ENNReal.ofReal (k * (1 / 2 ^ 64 - 1 / 2 ^ 512)) +
      ENNReal.ofReal (if k ≤ ((range U).filter (· ∈ C)).card
        then ((((range U).filter (· ∈ C)).card : ℝ) / U) ^ k else 0) := by
  refine (Bandwidth.pr_mono (fun b hb => (?_ : ¬ accepted U k b ∨
      (accepted U k b ∧ openings U k b (fun _ => 0) ⊆ C)))).trans
    ((Bandwidth.pr_or_le _ _).trans (add_le_add (pr_not_accepted_le hm hU hk) (pr_window_le hm hU hk C _)))
  by_cases ha : accepted U k b
  · exact Or.inr ⟨ha, hb ha⟩
  · exact Or.inl ha

/-- The test reads the first `k` blocks only. -/
theorem test_congr {m U k : ℕ} (C : Finset ℕ) {b b' : Blocks m} (h : ∀ i : Fin m, (i : ℕ) < k → b i = b' i) :
    test U k C b ↔ test U k C b' := by
  have ha : accepted U k b ↔ accepted U k b' :=
    forall_congr' fun i => imp_congr_right fun hi => by rw [h i hi]
  have hoff : ∀ l, l < k → offsets U b (fun _ => 0) l = offsets U b' (fun _ => 0) l := by
    intro l hl
    unfold offsets
    split_ifs with hlm
    · rw [h ⟨l, hlm⟩ hl]
    · rfl
  have ho : openings U k b (fun _ => 0) = openings U k b' (fun _ => 0) := by
    unfold openings
    rw [arrange_congr hoff]
  unfold test
  rw [ha, ho]

/-- An accepted draw opens receipts among the window's `U`. -/
theorem openings_subset {m U k : ℕ} {b : Blocks m} (r : ℕ → ℕ) (hk : k ≤ U) (hkm : k ≤ m)
    (ha : accepted U k b) : openings U k b r ⊆ range U := by
  intro q hq
  obtain ⟨p, hp, rfl⟩ := mem_image.1 hq
  exact mem_range.2 (arrange_lt hk (offsets_lt r hkm ha) p (by have := mem_range.1 hp; omega))

/-! ## The pooled bound -/

/-- Wrong receipts beyond the pooled bound leave the windows' draws, all landing on correct receipts, a probability
of at most `δ`: `∏_t (g_t/U_t)^(d_t) ≤ δ`, and `0` when a window has fewer correct receipts than draws. -/
theorem prod_le_of_pooled_lt {n : ℕ} (U g d : Fin n → ℕ) {δ : ℝ} (hδ : 0 ≤ δ) (hg : ∀ t, g t ≤ U t)
    (h : pooled n U d δ < ∑ t, ((U t - g t : ℕ) : ℝ)) :
    ∏ t, (if d t ≤ g t then ((g t : ℝ) / U t) ^ d t else 0) ≤ δ := by
  by_cases hdg : ∀ t, d t ≤ g t
  swap
  · push Not at hdg
    obtain ⟨t, ht⟩ := hdg
    rw [Finset.prod_eq_zero (Finset.mem_univ t) (by simp [not_le.2 ht])]
    exact hδ
  simp only [hdg, ↓reduceIte]
  set f : Fin n → ℝ := fun t => ((U t - g t : ℕ) : ℝ) / U t with hf
  have hfU : ∀ t, f t * U t = ((U t - g t : ℕ) : ℝ) := by
    intro t
    rcases Nat.eq_zero_or_pos (U t) with h0 | hpos
    · have : g t = 0 := by have := hg t; omega
      simp [hf, h0, this]
    · have hU : (U t : ℝ) ≠ 0 := by positivity
      simp only [hf]
      field_simp
  have hbox : ∀ t, 0 ≤ f t ∧ f t ≤ 1 - (d t : ℝ) / U t := by
    intro t
    refine ⟨by positivity, ?_⟩
    rcases Nat.eq_zero_or_pos (U t) with h0 | hpos
    · simp [hf, h0]
    · have hU : (0 : ℝ) < U t := by exact_mod_cast hpos
      have hd : (d t : ℝ) ≤ g t := by exact_mod_cast hdg t
      simp only [hf]
      rw [Nat.cast_sub (hg t), sub_div, div_self hU.ne']
      gcongr
  have hbdd : BddAbove ((fun f : Fin n → ℝ => ∑ t, f t * U t) ''
      {f | (∀ t, 0 ≤ f t ∧ f t ≤ 1 - (d t : ℝ) / U t) ∧ ∑ t, (d t : ℝ) * f t ≤ budget (∑ t, d t) δ}) := by
    refine ⟨∑ t, (U t : ℝ), ?_⟩
    rintro _ ⟨f', ⟨hf', -⟩, rfl⟩
    refine Finset.sum_le_sum (fun t _ => ?_)
    have h1 : f' t ≤ 1 := (hf' t).2.trans (by have : 0 ≤ (d t : ℝ) / U t := by positivity
                                              linarith)
    nlinarith [(hf' t).1, (Nat.cast_nonneg (U t) : (0 : ℝ) ≤ U t)]
  have hover : budget (∑ t, d t) δ < ∑ t, (d t : ℝ) * f t := by
    by_contra hle
    push Not at hle
    have := le_csSup hbdd ⟨f, ⟨hbox, hle⟩, rfl⟩
    simp only [hfU] at this
    exact absurd h (not_lt.2 this)
  set S := ∑ t, d t with hS
  have hd0 : S = 0 → ∀ t, d t = 0 := fun h0 t => by
    have := Finset.single_le_sum (fun i _ => Nat.zero_le (d i)) (Finset.mem_univ t)
    omega
  have hSpos : 0 < S := by
    rcases Nat.eq_zero_or_pos S with h0 | h0
    · have hz : ∑ t, (d t : ℝ) * f t = 0 := Finset.sum_eq_zero (fun t _ => by simp [hd0 h0 t])
      rw [hz, budget, h0] at hover
      simp at hover
    · exact h0
  set x : Fin n → ℝ := fun t => (g t : ℝ) / U t with hx
  have hx0 : ∀ t, 0 ≤ x t := fun t => by positivity
  have hdx : ∀ t, (d t : ℝ) * x t = d t - d t * f t := by
    intro t
    rcases Nat.eq_zero_or_pos (U t) with h0 | hpos
    · have : d t = 0 := by have := hdg t; have := hg t; omega
      simp [this]
    · have hU : (U t : ℝ) ≠ 0 := by positivity
      simp only [hx, hf]
      rw [Nat.cast_sub (hg t)]
      field_simp
      ring
  have hSR0 : (0 : ℝ) < S := by exact_mod_cast hSpos
  have hsum : ∑ t, (d t : ℝ) * x t < S * δ ^ ((S : ℝ)⁻¹) := by
    rw [Finset.sum_congr rfl (fun t _ => hdx t), Finset.sum_sub_distrib, ← Nat.cast_sum, ← hS]
    unfold budget at hover
    linarith
  have hamgm := Real.geom_mean_le_arith_mean Finset.univ (fun t => (d t : ℝ)) x (fun t _ => Nat.cast_nonneg _)
    (by rw [← Nat.cast_sum, ← hS]; exact hSR0) (fun t _ => hx0 t)
  have hSR : (∑ t, (d t : ℝ)) = S := by rw [hS, Nat.cast_sum]
  have hP : ∏ t, x t ^ (d t : ℝ) = ∏ t, x t ^ d t :=
    Finset.prod_congr rfl (fun t _ => Real.rpow_natCast _ _)
  rw [hSR, hP] at hamgm
  have hP0 : 0 ≤ ∏ t, x t ^ d t := Finset.prod_nonneg (fun t _ => pow_nonneg (hx0 t) _)
  have hroot : (∏ t, x t ^ d t) ^ ((S : ℝ)⁻¹) ≤ δ ^ ((S : ℝ)⁻¹) := by
    refine hamgm.trans ?_
    rw [div_le_iff₀ hSR0]
    linarith
  calc ∏ t, x t ^ d t = ((∏ t, x t ^ d t) ^ ((S : ℝ)⁻¹)) ^ S :=
        (Real.rpow_inv_natCast_pow hP0 hSpos.ne').symm
    _ ≤ (δ ^ ((S : ℝ)⁻¹)) ^ S := pow_le_pow_left₀ (Real.rpow_nonneg hP0 _) hroot S
    _ = δ := Real.rpow_inv_natCast_pow hδ hSpos.ne'

/-! ## Pairs -/

/-- A probability over a pair is the average over the first coordinate of its fibers' probabilities. -/
theorem pr_eq_sum_div₁ {X Y : Type} [Fintype X] [Fintype Y] (Q : X × Y → Prop) :
    pr Q = (∑ x, pr fun y => Q (x, y)) / Fintype.card X := by
  classical
  unfold pr
  have e : (univ.filter Q).card = ∑ x, (univ.filter fun y => Q (x, y)).card := by
    rw [card_filter, Fintype.sum_prod_type]
    simp only [card_filter]
  rw [e, Nat.cast_sum, Fintype.card_prod, Nat.cast_mul, ← ENNReal.sum_div]
  simp only [ENNReal.div_eq_inv_mul]
  rw [ENNReal.mul_inv (Or.inr (ENNReal.natCast_ne_top _)) (Or.inl (ENNReal.natCast_ne_top _))]
  ring

/-- A bound that holds at every value of the first coordinate, up to `e`, holds for the pair. -/
theorem pr_le_add_of_fiber {X Y Z : Type} [Fintype X] [Fintype Y] [Fintype Z] (P : X × Y → Prop)
    (Q : X × Z → Prop) (e : ℝ≥0∞) (h : ∀ x, pr (fun y => P (x, y)) ≤ pr (fun z => Q (x, z)) + e) :
    pr P ≤ pr Q + e := by
  rw [pr_eq_sum_div₁ P, pr_eq_sum_div₁ Q]
  calc (∑ x, pr fun y => P (x, y)) / Fintype.card X
      ≤ (∑ x, (pr (fun z => Q (x, z)) + e)) / Fintype.card X := ENNReal.div_le_div_right (sum_le_sum fun x _ => h x) _
    _ = (∑ x, pr fun z => Q (x, z)) / Fintype.card X + Fintype.card X * e / Fintype.card X := by
        rw [sum_add_distrib, ENNReal.add_div, sum_const, card_univ, nsmul_eq_mul]
    _ ≤ _ := add_le_add le_rfl (ENNReal.div_le_of_le_mul (mul_comm _ _).le)

/-- An event of the first coordinate has its own probability. -/
theorem pr_fst_eq {α β : Type} [Fintype α] [Fintype β] [Nonempty β] (P : α → Prop) :
    pr (fun z : α × β => P z.1) = pr P := by
  classical
  unfold pr
  have e : (univ.filter fun z : α × β => P z.1) = (univ.filter P) ×ˢ (univ : Finset β) := by
    ext z
    simp
  rw [e, card_product, card_univ, Fintype.card_prod, Nat.cast_mul, Nat.cast_mul]
  exact ENNReal.mul_div_mul_right _ _ (by exact_mod_cast Fintype.card_ne_zero) (ENNReal.natCast_ne_top _)

/-- An event of the second coordinate has at most its own probability. -/
theorem pr_snd_le {α β : Type} [Fintype α] [Fintype β] (P : β → Prop) : pr (fun z : α × β => P z.2) ≤ pr P := by
  rcases isEmpty_or_nonempty α with hα | hα
  · exact (le_of_eq (Bandwidth.pr_eq_zero_of_card _ Fintype.card_eq_zero)).trans zero_le
  · rw [pr_equiv (Equiv.prodComm α β)]
    exact le_of_eq (pr_fst_eq P)

theorem pr_true {α : Type} [Fintype α] [Nonempty α] : pr (fun _ : α => True) = 1 := by
  classical
  unfold pr
  rw [filter_true, card_univ]
  exact ENNReal.div_self (by exact_mod_cast Fintype.card_ne_zero) (ENNReal.natCast_ne_top _)

/-! ## The threshold coin -/

/-- The probability of `below δ P`: `1` at `P ≤ δ`, else `⌊δ·2^512/P⌋/2^512`. -/
noncomputable def coinWeight (δ P : ℝ) : ℝ := if P ≤ δ then 1 else ⌊δ * 2 ^ 512 / P⌋₊ / 2 ^ 512

theorem coinWeight_nonneg (δ P : ℝ) : 0 ≤ coinWeight δ P := by
  unfold coinWeight
  split_ifs <;> positivity

theorem coinWeight_le_div {δ P : ℝ} (hδ : 0 ≤ δ) (h : δ < P) : coinWeight δ P ≤ δ / P := by
  have hP : 0 < P := hδ.trans_lt h
  rw [coinWeight, ite_eq_right (not_le.2 h), div_le_iff₀ (by positivity)]
  calc (⌊δ * 2 ^ 512 / P⌋₊ : ℝ) ≤ δ * 2 ^ 512 / P := Nat.floor_le (by positivity)
    _ = δ / P * 2 ^ 512 := by ring

theorem lt_coinWeight_add {δ P : ℝ} (h : δ < P) : δ / P < coinWeight δ P + 1 / 2 ^ 512 := by
  rw [coinWeight, ite_eq_right (not_le.2 h), ← add_div, lt_div_iff₀ (by positivity)]
  calc δ / P * 2 ^ 512 = δ * 2 ^ 512 / P := by ring
    _ < _ := Nat.lt_floor_add_one _

theorem coinWeight_le_one {δ : ℝ} (hδ : 0 ≤ δ) (P : ℝ) : coinWeight δ P ≤ 1 := by
  by_cases h : P ≤ δ
  · rw [coinWeight, ite_eq_left h]
  · exact (coinWeight_le_div hδ (not_le.1 h)).trans ((div_le_one (hδ.trans_lt (not_le.1 h))).2 (not_le.1 h).le)

theorem coinWeight_one_le {δ : ℝ} (hδ : 0 ≤ δ) : coinWeight δ 1 ≤ δ := by
  by_cases h : 1 ≤ δ
  · rw [coinWeight, ite_eq_left h]
    exact h
  · simpa using coinWeight_le_div hδ (not_le.1 h)

theorem pr_below {δ : ℝ} (hδ : 0 ≤ δ) (P : ℝ) : pr (below δ P) = ENNReal.ofReal (coinWeight δ P) := by
  by_cases hP : P ≤ δ
  · have e : below δ P = fun _ => True := funext fun u => propext ⟨fun _ => trivial, fun _ => Or.inl hP⟩
    rw [e, pr_true, coinWeight, ite_eq_left hP, ENNReal.ofReal_one]
  · have hP0 : 0 < P := hδ.trans_lt (not_le.1 hP)
    have hF : ⌊δ * 2 ^ 512 / P⌋₊ ≤ 2 ^ 512 := by
      have : (⌊δ * 2 ^ 512 / P⌋₊ : ℝ) ≤ 2 ^ 512 :=
        (Nat.floor_le (by positivity)).trans ((div_le_iff₀ hP0).2 (by
          rw [mul_comm δ]; exact mul_le_mul_of_nonneg_left (not_le.1 hP).le (by positivity)))
      exact_mod_cast this
    have e : pr (below δ P) = pr (fun i : Fin (2 ^ 512) => (i : ℕ) < ⌊δ * 2 ^ 512 / P⌋₊) := by
      rw [← Actual.pr_comp_equiv (Pous.bitsEquiv 512)]
      congr 1
      funext u
      exact propext (by simp [below, hP, Pous.bitsVal])
    rw [e, coinWeight, ite_eq_right hP]
    classical
    unfold pr
    rw [filter_congr_decidable, Fin.card_filter_val_lt, min_eq_right hF, Fintype.card_fin,
      ENNReal.ofReal_div_of_pos (by positivity), ENNReal.ofReal_natCast, ENNReal.ofReal_pow (by norm_num),
      ENNReal.ofReal_ofNat, Nat.cast_pow, Nat.cast_ofNat]

/-- The step's rounding: `ρ·coinWeight(Pρ) ≤ coinWeight(P) + 2^-512`. -/
theorem mul_coinWeight_le {δ P ρ : ℝ} (hδ : 0 ≤ δ) (hP : 0 ≤ P) (hρ0 : 0 ≤ ρ) (hρ1 : ρ ≤ 1) :
    ρ * coinWeight δ (P * ρ) ≤ coinWeight δ P + 1 / 2 ^ 512 := by
  by_cases h : P * ρ ≤ δ
  · rw [coinWeight, ite_eq_left h, mul_one]
    by_cases hPδ : P ≤ δ
    · rw [coinWeight, ite_eq_left hPδ]
      linarith [show (0 : ℝ) ≤ 1 / 2 ^ 512 by positivity]
    · have hP0 : 0 < P := hδ.trans_lt (not_le.1 hPδ)
      refine le_of_lt (lt_of_le_of_lt ?_ (lt_coinWeight_add (not_le.1 hPδ)))
      rw [le_div_iff₀ hP0, mul_comm]
      exact h
  · have h' : δ < P * ρ := not_le.1 h
    have hPρ : P * ρ ≤ P := mul_le_of_le_one_right hP hρ1
    have hρ : 0 < ρ := by
      rcases hρ0.lt_or_eq with h1 | h1
      · exact h1
      · rw [← h1, mul_zero] at h'; linarith
    have hP0 : 0 < P := (hδ.trans_lt h').trans_le hPρ
    calc ρ * coinWeight δ (P * ρ) ≤ ρ * (δ / (P * ρ)) := mul_le_mul_of_nonneg_left (coinWeight_le_div hδ h') hρ0
      _ = δ / P := by field_simp
      _ ≤ coinWeight δ P + 1 / 2 ^ 512 := (lt_coinWeight_add (h'.trans_le hPρ)).le

/-- **One window's step**: `coinWeight(Pρ)·(k(2^-64 − 2^-512) + ρ) ≤ coinWeight(P) + s/2^64`. -/
theorem window_step_le {δ P ρ : ℝ} {k s : ℕ} (hδ : 0 ≤ δ) (hP : 0 ≤ P) (hρ0 : 0 ≤ ρ) (hρ1 : ρ ≤ 1) (hks : k ≤ s)
    (h0 : k = 0 → ρ = 1) :
    coinWeight δ (P * ρ) * (k * (1 / 2 ^ 64 - 1 / 2 ^ 512) + ρ) ≤ coinWeight δ P + s / 2 ^ 64 := by
  rcases Nat.eq_zero_or_pos k with rfl | hk
  · rw [h0 rfl, mul_one, Nat.cast_zero, zero_mul, zero_add, mul_one]
    linarith [show (0 : ℝ) ≤ s / 2 ^ 64 by positivity]
  · have ha : (0 : ℝ) ≤ 1 / 2 ^ 64 - 1 / 2 ^ 512 :=
      sub_nonneg.2 (one_div_le_one_div_of_le (by positivity) (pow_le_pow_right₀ (by norm_num) (by norm_num)))
    have ht : (0 : ℝ) ≤ 1 / 2 ^ 512 := by positivity
    have h1 : coinWeight δ (P * ρ) * (k * (1 / 2 ^ 64 - 1 / 2 ^ 512)) ≤ k * (1 / 2 ^ 64 - 1 / 2 ^ 512) :=
      mul_le_of_le_one_left (mul_nonneg (Nat.cast_nonneg _) ha) (coinWeight_le_one hδ _)
    have h2 := mul_coinWeight_le hδ hP hρ0 hρ1
    have hk1 : (1 : ℝ) ≤ k := by exact_mod_cast hk
    have hks' : (k : ℝ) ≤ s := by exact_mod_cast hks
    rw [mul_add]
    nlinarith [mul_nonneg (sub_nonneg.2 hk1) ht]

theorem gap_nonneg : (0 : ℝ) ≤ 1 / 2 ^ 64 - 1 / 2 ^ 512 :=
  sub_nonneg.2 (one_div_le_one_div_of_le (by positivity) (pow_le_pow_right₀ (by norm_num) (by norm_num)))

/-! ## The hybrid over windows -/

section Session

variable {n m : ℕ} {s : Fin n → ℕ} {κ : ℕ} {X : Type} (D : Session n m s κ X)

theorem draws_le (t : Fin n) (ω : Coins n κ X) : D.draws t ω ≤ m :=
  (min_le_right _ _).trans (Nat.lt_succ_iff.1 (D.size t ω).isLt)

theorem good_le (t : Fin n) (ω : Coins n κ X) : D.good t ω ≤ D.size t ω :=
  (card_le_card (filter_subset _ _)).trans_eq (card_range _)

theorem bound_nonneg (t : Fin n) (ω : Coins n κ X) : 0 ≤ D.bound t ω := by
  unfold Session.bound
  split_ifs
  · positivity
  · exact le_rfl

theorem bound_le_one (t : Fin n) (ω : Coins n κ X) : D.bound t ω ≤ 1 := by
  unfold Session.bound
  split_ifs
  · exact pow_le_one₀ (by positivity) (div_le_one_of_le₀ (by exact_mod_cast good_le D t ω) (by positivity))
  · exact zero_le_one

theorem lands_eq {t : Fin n} {ω ω' : Coins n κ X} (e1 : D.size t ω = D.size t ω')
    (e2 : D.correct t ω = D.correct t ω') : D.lands t ω = D.lands t ω' := by
  funext b
  simp only [Session.lands, Session.draws]
  rw [e1, e2]

theorem bound_eq {t : Fin n} {ω ω' : Coins n κ X} (e1 : D.size t ω = D.size t ω')
    (e2 : D.correct t ω = D.correct t ω') : D.bound t ω = D.bound t ω' := by
  have e3 : D.draws t ω = D.draws t ω' := by simp only [Session.draws, e1]
  have e4 : D.good t ω = D.good t ω' := by
    unfold Session.good
    rw [e1, e2]
  unfold Session.bound
  rw [e3, e4, e1]

/-- Windows `j ≤ t` read nothing of window `t`'s openings coin: replacing it leaves their sizes and correct receipts
as they are. -/
def SwapFree (t : Fin n) : Prop :=
  ∀ j, j ≤ t → ∀ (x : X) (c : Fin n → Pous.Bits κ) (K : Pous.Bits κ),
    D.size j (x, Function.update c t K) = D.size j (x, c) ∧
      D.correct j (x, Function.update c t K) = D.correct j (x, c)

/-- Window `t`'s coin traded for the key: `((x, c), u), K ↦ ((x, c[t ↦ K]), u), c_t`. -/
def swapCoin (t : Fin n) : (Coins n κ X × Pous.Bits 512) × Pous.Bits κ ≃ (Coins n κ X × Pous.Bits 512) × Pous.Bits κ where
  toFun z := (((z.1.1.1, Function.update z.1.1.2 t z.2), z.1.2), z.1.1.2 t)
  invFun z := (((z.1.1.1, Function.update z.1.1.2 t z.2), z.1.2), z.1.1.2 t)
  left_inv z := by
    obtain ⟨⟨⟨x, c⟩, u⟩, K⟩ := z
    simp
  right_inv z := by
    obtain ⟨⟨⟨x, c⟩, u⟩, K⟩ := z
    simp

theorem distinguisher_swap_iff (prf : Fin n → Pous.Bits κ → Blocks m) (δ : ℝ) (t : Fin n) (hS : SwapFree D t)
    (x : (Coins n κ X × Pous.Bits 512) × Pous.Bits κ) :
    D.distinguisher prf δ t (x.1, prf t x.2) ↔ D.potential prf δ ((t : ℕ) + 1) (swapCoin t x).1 := by
  obtain ⟨⟨⟨y, c⟩, u⟩, K⟩ := x
  have hl : ∀ j : Fin n, j ≤ t → D.lands j (y, Function.update c t K) = D.lands j (y, c) := fun j hj =>
    lands_eq D (hS j hj y c K).1 (hS j hj y c K).2
  have hb : ∀ j : Fin n, j ≤ t → D.bound j (y, Function.update c t K) = D.bound j (y, c) := fun j hj =>
    bound_eq D (hS j hj y c K).1 (hS j hj y c K).2
  have hf : univ.filter (fun j : Fin n => (j : ℕ) < (t : ℕ) + 1) = univ.filter (· ≤ t) :=
    filter_congr fun j _ => by rw [Fin.le_def]; omega
  simp only [Session.distinguisher, Session.potential, swapCoin, Equiv.coe_fn_mk, hf]
  rw [prod_congr rfl fun j hj => hb j (mem_filter.1 hj).2]
  constructor
  · rintro ⟨h1, h2, h3⟩
    refine ⟨fun j hj => ?_, h3⟩
    rcases Nat.lt_or_ge (j : ℕ) t with hjt | hjt
    · have hjt' : j < t := Fin.lt_def.2 hjt
      rw [hl j hjt'.le, Function.update_of_ne hjt'.ne]
      exact h1 j hjt'
    · have hjt' : j = t := Fin.ext (by omega)
      rw [hjt', hl t le_rfl, Function.update_self]
      exact h2
  · rintro ⟨h1, h3⟩
    refine ⟨fun j hjt => ?_, ?_, h3⟩
    · have := h1 j (by have := Fin.lt_def.1 hjt; omega)
      rwa [hl j hjt.le, Function.update_of_ne hjt.ne] at this
    · have := h1 t (Nat.lt_succ_self _)
      rwa [hl t le_rfl, Function.update_self] at this

/-- **The real side**: window `t`'s distinguisher on `prf t` under a uniform key is the potential after `t + 1`
windows, its key read as window `t`'s coin, in a session whose windows `j ≤ t` read nothing of that coin. -/
theorem pr_potential_succ [Fintype X] (prf : Fin n → Pous.Bits κ → Blocks m) (δ : ℝ) (t : Fin n)
    (hS : SwapFree D t) :
    pr (D.potential prf δ ((t : ℕ) + 1)) = pr (fun x : (Coins n κ X × Pous.Bits 512) × Pous.Bits κ =>
      D.distinguisher prf δ t (x.1, prf t x.2)) := by
  calc pr (D.potential prf δ ((t : ℕ) + 1))
      = pr (fun z : (Coins n κ X × Pous.Bits 512) × Pous.Bits κ => D.potential prf δ ((t : ℕ) + 1) z.1) :=
        (pr_fst_eq _).symm
    _ = pr (fun x => (fun z : (Coins n κ X × Pous.Bits 512) × Pous.Bits κ =>
          D.potential prf δ ((t : ℕ) + 1) z.1) (swapCoin t x)) :=
        (Actual.pr_comp_equiv (swapCoin t) _).symm
    _ = _ := by
        congr 1
        funext x
        exact propext (distinguisher_swap_iff D prf δ t hS x).symm

/-- **The ideal side**: on uniform blocks, window `t`'s distinguisher outputs true with probability at most the
potential after `t` windows plus `s_t/2^64`. -/
theorem pr_distinguisher_le [Fintype X] (prf : Fin n → Pous.Bits κ → Blocks m) {δ : ℝ}
    (hδ : 0 ≤ δ) (hm : m < 2 ^ 448) (t : Fin n) :
    pr (D.distinguisher prf δ t) ≤ pr (D.potential prf δ t) + ENNReal.ofReal (s t / 2 ^ 64) := by
  rw [pr_equiv (Equiv.prodAssoc (Coins n κ X) (Pous.Bits 512) (Blocks m))]
  refine pr_le_add_of_fiber _ _ _ (fun ω => ?_)
  show pr (fun y : Pous.Bits 512 × Blocks m => D.distinguisher prf δ t ((ω, y.1), y.2)) ≤
    pr (fun u => D.potential prf δ t (ω, u)) + _
  set P := ∏ j ∈ univ.filter (fun j : Fin n => (j : ℕ) < t), D.bound j ω with hP
  have hprod : ∏ j ∈ univ.filter (· ≤ t), D.bound j ω = P * D.bound t ω := by
    have e : univ.filter (· ≤ t) = insert t (univ.filter fun j : Fin n => (j : ℕ) < t) := by
      ext j
      simp only [mem_filter, mem_univ, true_and, mem_insert]
      rw [Fin.le_def, Fin.ext_iff]
      omega
    rw [e, prod_insert (by simp), mul_comm]
  by_cases hA : ∀ j : Fin n, j < t → D.lands j ω (prf j (ω.2 j))
  · have hU : (D.size t ω : ℕ) ≤ m := Nat.lt_succ_iff.1 (D.size t ω).isLt
    have hk : D.draws t ω ≤ D.size t ω := min_le_right _ _
    have hP0 : 0 ≤ P := prod_nonneg fun j _ => bound_nonneg D j ω
    calc pr (fun y : Pous.Bits 512 × Blocks m => D.distinguisher prf δ t ((ω, y.1), y.2))
        ≤ pr (fun y : Pous.Bits 512 × Blocks m => below δ (P * D.bound t ω) y.1 ∧ D.lands t ω y.2) :=
          Bandwidth.pr_mono fun y hy => ⟨hprod ▸ hy.2.2, hy.2.1⟩
      _ ≤ pr (below δ (P * D.bound t ω)) * (ENNReal.ofReal (D.draws t ω * (1 / 2 ^ 64 - 1 / 2 ^ 512)) +
            ENNReal.ofReal (D.bound t ω)) :=
          pr_and_le_mul (below δ (P * D.bound t ω)) (fun p => D.lands t ω p.2) _
            fun _ _ => pr_test_le hm hU hk (D.correct t ω)
      _ = ENNReal.ofReal (coinWeight δ (P * D.bound t ω) *
            (D.draws t ω * (1 / 2 ^ 64 - 1 / 2 ^ 512) + D.bound t ω)) := by
          rw [pr_below hδ, ← ENNReal.ofReal_add (mul_nonneg (Nat.cast_nonneg _) gap_nonneg) (bound_nonneg D t ω),
            ← ENNReal.ofReal_mul (coinWeight_nonneg _ _)]
      _ ≤ ENNReal.ofReal (coinWeight δ P + s t / 2 ^ 64) :=
          ENNReal.ofReal_le_ofReal (window_step_le hδ hP0 (bound_nonneg D t ω) (bound_le_one D t ω) (min_le_left _ _)
            fun h0 => by simp [Session.bound, h0])
      _ = pr (fun u => D.potential prf δ t (ω, u)) + ENNReal.ofReal (s t / 2 ^ 64) := by
          rw [ENNReal.ofReal_add (coinWeight_nonneg _ _) (by positivity), ← pr_below hδ]
          congr 2
          funext u
          exact propext ⟨fun h => ⟨hA, h⟩, fun h => h.2⟩
  · calc pr (fun y : Pous.Bits 512 × Blocks m => D.distinguisher prf δ t ((ω, y.1), y.2))
        ≤ pr (fun _ : Pous.Bits 512 × Blocks m => False) := Bandwidth.pr_mono fun y hy => hA hy.1
      _ = 0 := pr_false
      _ ≤ _ := zero_le

/-! ## What the distinguisher is -/

/-- **Window `t`'s distinguisher reads at most `s_t` first blocks.** -/
theorem distinguisher_congr (prf : Fin n → Pous.Bits κ → Blocks m) (δ : ℝ) (t : Fin n)
    (y : Coins n κ X × Pous.Bits 512) {b b' : Blocks m} (h : ∀ i : Fin m, (i : ℕ) < s t → b i = b' i) :
    D.distinguisher prf δ t (y, b) ↔ D.distinguisher prf δ t (y, b') := by
  have e : D.lands t y.1 b ↔ D.lands t y.1 b' :=
    test_congr _ fun i hi => h i (lt_of_lt_of_le hi (min_le_left _ _))
  simp only [Session.distinguisher, e]

/-- **An honest session's distinguishers gain nothing.** -/
theorem prfAdvantage_honest [Fintype X] (hc : ∀ t ω, range (D.size t ω : ℕ) ⊆ D.correct t ω)
    (prf : Fin n → Pous.Bits κ → Blocks m) (δ : ℝ) (t : Fin n) :
    Pous.StreamedWipe.prfAdvantage (prf t) (D.distinguisher prf δ t) = 0 := by
  have hl : ∀ j ω b, D.lands j ω b := fun j ω b ha =>
    (openings_subset _ (min_le_right _ _) (draws_le D j ω) ha).trans (hc j ω)
  let E : Coins n κ X × Pous.Bits 512 → Prop := fun y => below δ (∏ j ∈ univ.filter (· ≤ t), D.bound j y.1) y.2
  have e : ∀ y b, D.distinguisher prf δ t (y, b) ↔ E y :=
    fun y b => ⟨fun h => h.2.2, fun h => ⟨fun j _ => hl _ _ _, hl _ _ _, h⟩⟩
  have h1 : pr (fun x : (Coins n κ X × Pous.Bits 512) × Pous.Bits κ => D.distinguisher prf δ t (x.1, prf t x.2)) =
      pr E := by
    rw [← pr_fst_eq (β := Pous.Bits κ) E]
    congr 1
    funext x
    exact propext (e x.1 (prf t x.2))
  have h2 : pr (D.distinguisher prf δ t) = pr E := by
    rw [← pr_fst_eq (β := Blocks m) E]
    congr 1
    funext x
    exact propext (e x.1 x.2)
  unfold Pous.StreamedWipe.prfAdvantage
  rw [h1, h2, tsub_self]

end Session

/-! ## The junk-commitment hybrid -/

/-- One window's step, as arithmetic: the potential after `t + 1` windows at the deployed commitment `p1`, at the
junk one `j1`, the potentials after `t` windows at the junk commitment `jt` and the deployed one `pt`. -/
theorem hybrid_step {p1 j1 jt pt β e a : ℝ≥0∞} (hj : j1 ≤ jt + e + a) (h : pt ≤ β) :
    p1 ≤ β + (e + (a + ((p1 - j1) + (jt - pt)))) :=
  calc p1 ≤ j1 + (p1 - j1) := le_add_tsub
    _ ≤ jt + e + a + (p1 - j1) := add_le_add hj le_rfl
    _ ≤ pt + (jt - pt) + e + a + (p1 - j1) :=
        add_le_add (add_le_add (add_le_add le_add_tsub le_rfl) le_rfl) le_rfl
    _ ≤ β + (jt - pt) + e + a + (p1 - j1) :=
        add_le_add (add_le_add (add_le_add (add_le_add h le_rfl) le_rfl) le_rfl) le_rfl
    _ = β + (e + (a + ((p1 - j1) + (jt - pt)))) := by ring

section Committed

variable {n m : ℕ} {s : Fin n → ℕ} {κ : ℕ} {X Cm : Type} (F : Committed n m s κ X Cm)

theorem play_size (com : Coins n κ X → Cm) (t : Fin n) (ω : Coins n κ X) :
    (F.play com).size t ω = (F.view (com ω)).size t ω := rfl

theorem play_correct (com : Coins n κ X → Cm) (t : Fin n) (ω : Coins n κ X) :
    (F.play com).correct t ω = (F.view (com ω)).correct t ω := rfl

/-- **The junk session's windows `j ≤ t` read nothing of window `t`'s coin**: its commitment reads none of coin `t`,
and at each fixed commitment windows `j ≤ t` read only the other coins and the earlier openings coins. -/
theorem swapFree_junk (hL : ∀ c, Live (F.view c)) (com : Coins n κ X → Cm) (t : Fin n) :
    SwapFree (F.play (Committed.junk com t)) t := by
  intro j hj x c K
  have hc : Committed.junk com t (x, Function.update c t K) = Committed.junk com t (x, c) := by
    simp only [Committed.junk, Function.update_idem]
  have hω : ∀ i, i < j → Function.update c t K i = c i := fun i hi =>
    Function.update_of_ne (ne_of_lt (lt_of_lt_of_le hi hj)) _ _
  rw [play_size, play_size, play_correct, play_correct, hc]
  exact hL (Committed.junk com t (x, c)) j (x, Function.update c t K) (x, c) rfl hω

theorem hide_le [Fintype X] (com : Coins n κ X → Cm) (prf : Fin n → Pous.Bits κ → Blocks m) (δ : ℝ) (t : Fin n) :
    F.hide com prf δ t ≤ 1 + 1 := by
  unfold Committed.hide
  exact add_le_add (tsub_le_self.trans (pr_le_one _)) (tsub_le_self.trans (pr_le_one _))

/-- One window of the hybrid: the potential gains at most `s_t/2^64`, window `t`'s distinguisher's advantage at the
junk commitment, and window `t`'s hiding term. -/
theorem pr_potential_succ_le [Fintype X] (hL : ∀ c, Live (F.view c)) (com : Coins n κ X → Cm)
    (prf : Fin n → Pous.Bits κ → Blocks m) {δ : ℝ} (hδ : 0 ≤ δ) (hm : m < 2 ^ 448) (t : Fin n) {β : ℝ≥0∞}
    (h : pr ((F.play com).potential prf δ t) ≤ β) :
    pr ((F.play com).potential prf δ ((t : ℕ) + 1)) ≤ β + (ENNReal.ofReal (s t / 2 ^ 64) +
      (Pous.StreamedWipe.prfAdvantage (prf t) ((F.play (Committed.junk com t)).distinguisher prf δ t) +
        F.hide com prf δ t)) := by
  have hJ : pr ((F.play (Committed.junk com t)).potential prf δ ((t : ℕ) + 1)) ≤
      pr ((F.play (Committed.junk com t)).potential prf δ t) + ENNReal.ofReal (s t / 2 ^ 64) +
        Pous.StreamedWipe.prfAdvantage (prf t) ((F.play (Committed.junk com t)).distinguisher prf δ t) := by
    rw [pr_potential_succ _ prf δ t (swapFree_junk F hL com t)]
    exact (Actual.pr_le_add_prfAdvantage (prf t) _ le_rfl).trans
      (add_le_add (pr_distinguisher_le _ prf hδ hm t) le_rfl)
  exact hybrid_step hJ h

/-- **The hybrid**: the potential after `t` windows is at most `δ` plus the first `t` windows' errors. -/
theorem pr_potential_le [Fintype X] (hL : ∀ c, Live (F.view c)) (com : Coins n κ X → Cm)
    (prf : Fin n → Pous.Bits κ → Blocks m) {δ : ℝ} (hδ : 0 ≤ δ) (hm : m < 2 ^ 448) :
    ∀ t, t ≤ n → pr ((F.play com).potential prf δ t) ≤ ENNReal.ofReal δ +
      ∑ j ∈ univ.filter (fun j : Fin n => (j : ℕ) < t),
        (ENNReal.ofReal (s j / 2 ^ 64) + (Pous.StreamedWipe.prfAdvantage (prf j)
          ((F.play (Committed.junk com j)).distinguisher prf δ j) + F.hide com prf δ j))
  | 0, _ => by
    have h0 : univ.filter (fun j : Fin n => (j : ℕ) < 0) = ∅ := filter_false_of_mem fun j _ => Nat.not_lt_zero _
    rw [h0, sum_empty, add_zero]
    calc pr ((F.play com).potential prf δ 0) ≤ pr (fun z : Coins n κ X × Pous.Bits 512 => below δ 1 z.2) :=
          Bandwidth.pr_mono fun z hz => by
            have := hz.2
            rwa [h0, prod_empty] at this
      _ ≤ pr (below δ 1) := pr_snd_le _
      _ = ENNReal.ofReal (coinWeight δ 1) := pr_below hδ 1
      _ ≤ ENNReal.ofReal δ := ENNReal.ofReal_le_ofReal (coinWeight_one_le hδ)
  | t + 1, ht => by
    have hf : univ.filter (fun j : Fin n => (j : ℕ) < t + 1) =
        insert ⟨t, ht⟩ (univ.filter fun j : Fin n => (j : ℕ) < t) := by
      ext j
      simp only [mem_filter, mem_univ, true_and, mem_insert, Fin.ext_iff]
      omega
    rw [hf, sum_insert (by simp)]
    refine (pr_potential_succ_le F hL com prf hδ hm ⟨t, ht⟩ (pr_potential_le hL com prf hδ hm t (by omega))).trans
      (le_of_eq ?_)
    ring

/-- Every window landing on `prf` while the bounds multiply to at most `δ` has probability at most `δ` plus the
windows' errors. -/
theorem pr_landed_le [Fintype X] (hL : ∀ c, Live (F.view c)) (com : Coins n κ X → Cm)
    (prf : Fin n → Pous.Bits κ → Blocks m) {δ : ℝ} (hδ : 0 ≤ δ) (hm : m < 2 ^ 448) (E : Coins n κ X → Prop)
    (hE : ∀ ω, E ω → (∀ t, (F.play com).lands t ω (prf t (ω.2 t))) ∧ ∏ t, (F.play com).bound t ω ≤ δ) :
    pr E ≤ ENNReal.ofReal δ + ∑ t, (ENNReal.ofReal (s t / 2 ^ 64) + (Pous.StreamedWipe.prfAdvantage (prf t)
      ((F.play (Committed.junk com t)).distinguisher prf δ t) + F.hide com prf δ t)) := by
  rw [← pr_fst_eq (β := Pous.Bits 512) E]
  refine (Bandwidth.pr_mono fun z hz => (?_ : (F.play com).potential prf δ n z)).trans ?_
  · obtain ⟨h1, h2⟩ := hE z.1 hz
    refine ⟨fun j _ => h1 j, Or.inl ?_⟩
    rwa [filter_true_of_mem fun j _ => j.isLt]
  · have := pr_potential_le F hL com prf hδ hm n le_rfl
    rwa [filter_true_of_mem fun j _ => j.isLt] at this

end Committed

/-- The hybrid's error as the guarantees state it: `δ`, the windows' `s_t/2^64`, and the advantages, hiding terms and
binding term summed apart. -/
theorem error_toReal {n : ℕ} (s : Fin n → ℕ) {δ : ℝ} (hδ : 0 ≤ δ) (a h : Fin n → ℝ≥0∞) (ha : ∀ t, a t ≤ 1)
    (hh : ∀ t, h t ≤ 1 + 1) (b : ℝ≥0∞) (hb : b ≤ 1) {p : ℝ≥0∞}
    (hp : p ≤ ENNReal.ofReal δ + ∑ t, (ENNReal.ofReal ((s t : ℝ) / 2 ^ 64) + (a t + h t)) + b) :
    p.toReal ≤ δ + ((∑ t, s t : ℕ) : ℝ) / 2 ^ 64 + (∑ t, a t).toReal + (∑ t, h t).toReal + b.toReal := by
  have ha' : ∑ t, a t ≠ ⊤ := ENNReal.sum_ne_top.2 fun t _ => ne_top_of_le_ne_top ENNReal.one_ne_top (ha t)
  have hh' : ∑ t, h t ≠ ⊤ := ENNReal.sum_ne_top.2 fun t _ =>
    ne_top_of_le_ne_top (ENNReal.add_ne_top.2 ⟨ENNReal.one_ne_top, ENNReal.one_ne_top⟩) (hh t)
  have hb' : b ≠ ⊤ := ne_top_of_le_ne_top ENNReal.one_ne_top hb
  have hpos : 0 ≤ δ + ((∑ t, s t : ℕ) : ℝ) / 2 ^ 64 := by positivity
  have h1 : ENNReal.ofReal (((∑ t, s t : ℕ) : ℝ) / 2 ^ 64) = ∑ t, ENNReal.ofReal ((s t : ℝ) / 2 ^ 64) := by
    rw [Nat.cast_sum, sum_div]
    exact ENNReal.ofReal_sum_of_nonneg fun t _ => by positivity
  have hε : ENNReal.ofReal δ + ∑ t, (ENNReal.ofReal ((s t : ℝ) / 2 ^ 64) + (a t + h t)) + b =
      ENNReal.ofReal (δ + ((∑ t, s t : ℕ) : ℝ) / 2 ^ 64) + ∑ t, a t + ∑ t, h t + b := by
    rw [ENNReal.ofReal_add hδ (by positivity), h1, sum_add_distrib, sum_add_distrib]
    ring
  rw [hε] at hp
  have hq : ENNReal.ofReal (δ + ((∑ t, s t : ℕ) : ℝ) / 2 ^ 64) + ∑ t, a t ≠ ⊤ :=
    ENNReal.add_ne_top.2 ⟨ENNReal.ofReal_ne_top, ha'⟩
  have hq' : ENNReal.ofReal (δ + ((∑ t, s t : ℕ) : ℝ) / 2 ^ 64) + ∑ t, a t + ∑ t, h t ≠ ⊤ :=
    ENNReal.add_ne_top.2 ⟨hq, hh'⟩
  calc p.toReal ≤ (ENNReal.ofReal (δ + ((∑ t, s t : ℕ) : ℝ) / 2 ^ 64) + ∑ t, a t + ∑ t, h t + b).toReal :=
        ENNReal.toReal_mono (ENNReal.add_ne_top.2 ⟨hq', hb'⟩) hp
    _ = _ := by
        rw [ENNReal.toReal_add hq' hb', ENNReal.toReal_add hq hh', ENNReal.toReal_add ENNReal.ofReal_ne_top ha',
          ENNReal.toReal_ofReal hpos]

/-- An opened receipt that `Refused` lets through on an accepting outcome off the binding event is correct: a moved one
is in the event. -/
theorem opened_sub {n m : ℕ} {s : Fin n → ℕ} {κ : ℕ} {X : Type} [Fintype X] {D : Session n m s κ X}
    {prf : Fin n → Pous.Bits κ → Blocks m} {rest : Coins n κ X → Fin n → ℕ → ℕ} {T : Trace (Coins n κ X)}
    {R : Receipts n κ X} (hR : Refused D prf rest T R) {ω : Coins n κ X} (ha : T.accepts ω)
    (hb : ¬ ∃ t, ∃ p ∈ D.opened prf rest t ω, R.Moved D prf rest t ω p) (t : Fin n) :
    D.opened prf rest t ω ⊆ D.correct t ω :=
  fun p hp => (hR ω ha t p hp).resolve_right fun hmv => hb ⟨t, p, hp, hmv⟩

/-- `PooledDraw` at one committed session. -/
theorem pooledDraw {n m : ℕ} {s : Fin n → ℕ} {κ : ℕ} {X Cm : Type} [Fintype X]
    (prf : Fin n → Pous.Bits κ → Blocks m) (rest : Coins n κ X → Fin n → ℕ → ℕ) (F : Committed n m s κ X Cm)
    (com : Coins n κ X → Cm) (T : Trace (Coins n κ X)) (R : Receipts n κ X) {δ : ℝ} (hδ : 0 ≤ δ) (hm : m < 2 ^ 448)
    (hL : ∀ c, Live (F.view c)) (hR : Refused (F.play com) prf rest T R) (hC : Counted (F.play com) T) :
    T.Holds (δ + ((∑ t, s t : ℕ) : ℝ) / 2 ^ 64 +
        (∑ t, Pous.StreamedWipe.prfAdvantage (prf t) ((F.play (Committed.junk com t)).distinguisher prf δ t)).toReal +
        (∑ t, F.hide com prf δ t).toReal + (R.binding (F.play com) prf rest).toReal)
      (fun ω => (T.wrong ω : ℝ) ≤ (F.play com).pooled δ ω) := by
  classical
  have hreal : ∀ ω, (T.accepts ω ∧ ¬ (T.wrong ω : ℝ) ≤ (F.play com).pooled δ ω) ∧
      ¬ (∃ t, ∃ p ∈ (F.play com).opened prf rest t ω, R.Moved (F.play com) prf rest t ω p) →
      (∀ t, (F.play com).lands t ω (prf t (ω.2 t))) ∧ ∏ t, (F.play com).bound t ω ≤ δ := by
    intro ω ⟨⟨hacc, hw⟩, hb⟩
    refine ⟨fun t ha => ?_, ?_⟩
    · have h := opened_sub hR hacc hb t
      simp only [Session.opened] at h
      rwa [openings_congr (fun _ => 0) (rest ω t) (draws_le (F.play com) t ω) ha]
    · refine prod_le_of_pooled_lt (fun t => (F.play com).size t ω) (fun t => (F.play com).good t ω)
        (fun t => (F.play com).draws t ω) hδ (fun t => good_le (F.play com) t ω) ?_
      have hw' : (F.play com).pooled δ ω < T.wrong ω := not_le.1 hw
      have hc : (T.wrong ω : ℝ) ≤ (F.play com).wrong ω := by exact_mod_cast hC ω hacc
      calc pooled n (fun t => (F.play com).size t ω) (fun t => (F.play com).draws t ω) δ =
            (F.play com).pooled δ ω := rfl
        _ < (F.play com).wrong ω := hw'.trans_le hc
        _ = ∑ t, (((F.play com).size t ω - (F.play com).good t ω : ℕ) : ℝ) := by
            rw [Session.wrong, Nat.cast_sum]
  have hall := pr_landed_le F hL com prf hδ hm
    (fun ω => (T.accepts ω ∧ ¬ (T.wrong ω : ℝ) ≤ (F.play com).pooled δ ω) ∧
      ¬ (∃ t, ∃ p ∈ (F.play com).opened prf rest t ω, R.Moved (F.play com) prf rest t ω p)) hreal
  have hsplit : pr (fun ω => T.accepts ω ∧ ¬ (T.wrong ω : ℝ) ≤ (F.play com).pooled δ ω) ≤
      pr (fun ω => (T.accepts ω ∧ ¬ (T.wrong ω : ℝ) ≤ (F.play com).pooled δ ω) ∧
        ¬ (∃ t, ∃ p ∈ (F.play com).opened prf rest t ω, R.Moved (F.play com) prf rest t ω p)) +
      R.binding (F.play com) prf rest :=
    (Bandwidth.pr_mono (Q := fun ω => ((T.accepts ω ∧ ¬ (T.wrong ω : ℝ) ≤ (F.play com).pooled δ ω) ∧
        ¬ (∃ t, ∃ p ∈ (F.play com).opened prf rest t ω, R.Moved (F.play com) prf rest t ω p)) ∨
        (∃ t, ∃ p ∈ (F.play com).opened prf rest t ω, R.Moved (F.play com) prf rest t ω p))
      fun ω h => (em (∃ t, ∃ p ∈ (F.play com).opened prf rest t ω, R.Moved (F.play com) prf rest t ω p)).elim Or.inr
        fun hb => Or.inl ⟨h, hb⟩).trans (Bandwidth.pr_or_le _ _)
  unfold Trace.Holds
  rw [Accounting.pr_eq_toReal]
  exact error_toReal s hδ
    (fun t => Pous.StreamedWipe.prfAdvantage (prf t) ((F.play (Committed.junk com t)).distinguisher prf δ t))
    (fun t => F.hide com prf δ t) (fun t => tsub_le_self.trans (pr_le_one _)) (fun t => hide_le F com prf δ t)
    (R.binding (F.play com) prf rest) (pr_le_one _) (hsplit.trans (add_le_add hall le_rfl))

end Draw

/-- **The openings' pooled bound** (`Accounting.Guarantees.Poub.PooledDraw`). -/
theorem PooledDraw : Accounting.Guarantees.Poub.PooledDraw := by
  intro n m k κ X Cm _ prf rest F com T R δ hδ hm hL hR hC
  exact Draw.pooledDraw prf rest F com T R hδ hm hL hR hC

/-- **Live coins, derived** (`Accounting.Guarantees.Poub.LiveCoinsDrawn`). -/
theorem LiveCoinsDrawn : Accounting.Guarantees.Poub.LiveCoinsDrawn := by
  intro n m k κ X Cm _ prf rest F com T R W δ hδ hm hL hR hC hW
  exact holdsMono (fun ω ha h => h.trans (hW ω ha)) (Draw.pooledDraw prf rest F com T R hδ hm hL hR hC)

/-- **The snapshot held at the record's pooled bound** (`Accounting.Guarantees.Poub.HeldDrawn`). -/
theorem HeldDrawn : Accounting.Guarantees.Poub.HeldDrawn := by
  intro n m k κ X Cm _ prf rest F com T R S s₀ e s E β Br Bd δs δc hE hs hβ hδ hm hI hsq hd hS hr hc hk hL hR hC
  refine holdsMono (fun ω ha h => ?_) (holdsAnd hk (Draw.pooledDraw prf rest F com T R hδ hm hL hR hC))
  obtain ⟨hm', h0, hP⟩ := hc ω ha
  exact held_of ω hE hs hβ hI hsq hd hS hr hm' h0 hP h.1 h.2

/-- **Held over time with the links charged, at the record's pooled bound**
(`Accounting.Guarantees.Poub.HeldOverTimeLinksDrawn`). -/
theorem HeldOverTimeLinksDrawn : Accounting.Guarantees.Poub.HeldOverTimeLinksDrawn := by
  intro n m k κ X Cm _ prf rest F com T R K S s E β βl Br Bd Bl δs δc hE hs hβ hβl hδ hm hl hsq hd hS hr hp hcov hk
    hL hR hC
  refine holdsMono (fun ω ha h => ?_) (holdsAnd hk (Draw.pooledDraw prf rest F com T R hδ hm hL hR hC))
  exact heldOverLinks_of K ω hE hs hβ hβl hl hsq hd hS hr hp hcov ha h.1 h.2

/-- **Held over time at the record's pooled bound, under Identity** (`Accounting.Guarantees.Poub.HeldOverTimeDrawn`):
the link-charged form at links of rate 0 and burst 0. -/
theorem HeldOverTimeDrawn : Accounting.Guarantees.Poub.HeldOverTimeDrawn := by
  intro n m k κ X Cm _ prf rest F com T R K S s E β Br Bd δs δc hE hs hβ hδ hm hI hsq hd hS hr hp hcov hk hL hR hC
  simpa only [add_zero] using HeldOverTimeLinksDrawn n m k κ X Cm prf rest F com T R K S s E β 0 Br Bd 0 δs δc hE hs hβ
    le_rfl hδ hm (links_of_identity hI) hsq hd hS hr hp hcov hk hL hR hC

end Accounting.SecurityProofs.Poub
