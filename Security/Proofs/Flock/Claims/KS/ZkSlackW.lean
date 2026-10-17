import Proofs.Flock.Claims.KS.ZkFitsW

/-!
# The masked weight's room for one more event per round

The fork bound's slack (`Claims.Flock.forkSlack`) adds one fork event per round. The plain weight pays for it with
level 0's unused `2·2^{d₀}·Q₀` (`FlockDeploy.KS.plain_fork_room`). The masked weight already spends level 0's on its
fork event (`Claims.Flock.cZW`), but counted per round (`Claims.Escapes.FitsForkR`) only round `0`'s events are charged
twice, and where round `0` holds only level 0's event of each table (`hr0`: the deployed level caps, pads and `τ` are
absorbed after the link message), the weight's room past the joined escapes (`Claims.Flock.joinCW`) and the fork of
weight `k` is every level cap's and both pads roots' weight once more (`Claims.Flock.roomW`, `zk_fork_room`): each
table's weight pays twice for its fork events but `τ`'s `2` and level 0's, and `k`'s ceiling fits in the `2` per table.
-/

namespace Claims.Flock

/-- **The joined masked weight at round `r`**: every table's fork events there (`joinEscW`'s weights). -/
noncomputable def joinCW {K : ℕ} (n : ℕ) (sch : ℕ → FlockSoundness.Accounting.Schedule) (Np : ℕ → ℕ)
    (rd : (j : ℕ) → IZW (sch j).levels.length → Fin K) (r : Fin K) : ℝ :=
  ∑ j ∈ Finset.range n, ∑ i ∈ Finset.univ.filter (fun i => rd j i = r), cZW (sch j) (Np j) i

theorem joinCW_nonneg {K : ℕ} (n : ℕ) (sch : ℕ → FlockSoundness.Accounting.Schedule) (Np : ℕ → ℕ)
    (rd : (j : ℕ) → IZW (sch j).levels.length → Fin K) (r : Fin K) : 0 ≤ joinCW n sch Np rd r :=
  Finset.sum_nonneg fun _ _ => Finset.sum_nonneg fun _ _ => cZW_nonneg _ _ _

/-- **The masked weight's room**: every table's level caps' weights `Σ_{r,ℓ} 2^{d_{ℓ+1}}·Q_{ℓ+1}` and pads roots'
`2·N_p²`. -/
noncomputable def roomW (n : ℕ) (sch : ℕ → FlockSoundness.Accounting.Schedule) (Np : ℕ → ℕ) : ℝ :=
  ∑ j ∈ Finset.range n,
    ((∑ a : Fin 2 × Fin (sch j).levels.length, cOf (sch j) (a.2.val + 1)) + 2 * ((Np j : ℝ) * Np j))

theorem roomW_nonneg (n : ℕ) (sch : ℕ → FlockSoundness.Accounting.Schedule) (Np : ℕ → ℕ) : 0 ≤ roomW n sch Np :=
  Finset.sum_nonneg fun _ _ => add_nonneg (Finset.sum_nonneg fun _ _ => cOf_nonneg _ _) (by positivity)

/-- **The joined weights are every table's.** -/
theorem joinCW_sum {K : ℕ} (n : ℕ) (sch : ℕ → FlockSoundness.Accounting.Schedule) (Np : ℕ → ℕ)
    (rd : (j : ℕ) → IZW (sch j).levels.length → Fin K) :
    ∑ r, joinCW n sch Np rd r = ∑ j ∈ Finset.range n, ∑ i, cZW (sch j) (Np j) i := by
  unfold joinCW
  rw [Finset.sum_comm]
  exact Finset.sum_congr rfl fun j _ => Finset.sum_fiberwise _ _ _

/-- **Round `0`'s joined weight is at most level 0's events'**, where only level 0's event of a table is at round
`0`. -/
theorem joinCW_zero_le {K : ℕ} (n : ℕ) (sch : ℕ → FlockSoundness.Accounting.Schedule) (Np : ℕ → ℕ)
    (rd : (j : ℕ) → IZW (sch j).levels.length → Fin K) (hr0 : ∀ j i, (rd j i).val = 0 → i = .inr ())
    (r : Fin K) (hr : r.val = 0) :
    joinCW n sch Np rd r ≤ ∑ j ∈ Finset.range n, cZW (sch j) (Np j) (.inr ()) := by
  refine Finset.sum_le_sum fun j _ => ?_
  have hsub : Finset.univ.filter (fun i => rd j i = r) ⊆ {.inr ()} := fun i hi => by
    rw [Finset.mem_singleton]
    exact hr0 j i ((congrArg Fin.val (Finset.mem_filter.1 hi).2).trans hr)
  refine (Finset.sum_le_sum_of_subset_of_nonneg hsub fun i _ _ => cZW_nonneg _ _ i).trans ?_
  rw [Finset.sum_singleton]

/-- **The room and level 0's events are every table's weight but `2`.** -/
theorem roomW_add (n : ℕ) (sch : ℕ → FlockSoundness.Accounting.Schedule) (Np : ℕ → ℕ) :
    roomW n sch Np + ∑ j ∈ Finset.range n, cZW (sch j) (Np j) (.inr ()) =
      ∑ j ∈ Finset.range n, ∑ i, cZW (sch j) (Np j) i - 2 * n := by
  have hper : ∀ j, (∑ a : Fin 2 × Fin (sch j).levels.length, cOf (sch j) (a.2.val + 1)) +
      2 * ((Np j : ℝ) * Np j) + cZW (sch j) (Np j) (.inr ()) = ∑ i, cZW (sch j) (Np j) i - 2 := fun j => by
    have hz : cZW (sch j) (Np j) (.inr ()) =
        2 ^ (FlockSoundness.lvOf (sch j) 0).logLen * (2 * (FlockSoundness.lvOf (sch j) 0).queries) := rfl
    rw [cZW_sum, hz]
    ring
  unfold roomW
  rw [← Finset.sum_add_distrib, Finset.sum_congr rfl fun j _ => hper j, Finset.sum_sub_distrib, Finset.sum_const,
    Finset.card_range, nsmul_eq_mul]
  ring

end Claims.Flock

namespace FlockDeploy.KS

open Flock FlockSoundness

/-- **The masked weight's room for one more event per round**: at a statement whose tables `st j` are C-Flock ZK's,
when each table's pads' weight is at most its deployed pads' domain and only level 0's event of a table is at round
`0`, the fork of weight `k` and the joined escapes, round `0`'s charged twice, leave every table's level caps' and pads
roots' weights (`Claims.Flock.roomW`) of twice the deployed weight: each table's weight pays for its events
(`cZW_sum_le_tableWeight`), round `0` holds at most level 0's, the room and level 0's are every table's but `2`
(`Claims.Flock.roomW_add`), and `2·k < 8·N₀/e + 2`. -/
theorem zk_fork_room (rel : Readable.Parsed → Readable.Relation) (x : Claims.Msg) {spec : SessionSpec}
    (st : ℕ → Stmt) (hst : ∀ j < (tables true x).length, (tables true x)[j]? = some (spec, st j))
    (sch : ℕ → Accounting.Schedule) (hsch : ∀ j < (tables true x).length, sch j = Accounting.m1Schedule (st j).m)
    (Np : ℕ → ℕ) (hNp : ∀ j < (tables true x).length,
      Np j ≤ Claims.Flock.Numbers.padDomain (Zk.PadLayout.new (st j).m (st j).c.kLog
        (Claims.Flock.zkSched (st j).m).1.k0))
    (rd : (j : ℕ) → Claims.Flock.IZW (sch j).levels.length → Fin ((deployZk rel).V.k x))
    (hr0 : ∀ j i, (rd j i).val = 0 → i = .inr ()) (r₀ : Fin ((deployZk rel).V.k x)) (hr₀ : r₀.val = 0) :
    2 * ((deployZk rel).k x : ℝ) + Claims.Flock.joinCW (tables true x).length sch Np rd r₀ +
        ∑ r, Claims.Flock.joinCW (tables true x).length sch Np rd r +
        Claims.Flock.roomW (tables true x).length sch Np ≤
      2 * (deployZk rel).weight x := by
  set n := (tables true x).length with hn
  have hz := Claims.Flock.joinCW_zero_le n sch Np rd hr0 r₀ hr₀
  have hroom := Claims.Flock.roomW_add n sch Np
  rw [Claims.Flock.joinCW_sum]
  have hw : weight true x = ∑ j ∈ Finset.range n, tableWeight true (st j) :=
    sum_map_eq_range (tables true x) (fun j => (spec, st j)) hst _
  have hN : N₀ true x = ∑ j ∈ Finset.range n, tableLen true (st j) :=
    sum_map_eq_range (tables true x) (fun j => (spec, st j)) hst _
  have hk : ((deployZk rel).k x : ℝ) = (⌈4 * N₀ true x / Real.exp 1⌉₊ : ℝ) := rfl
  show 2 * ((deployZk rel).k x : ℝ) + _ + _ + _ ≤ 2 * weight true x
  rcases Nat.eq_zero_or_pos n with h0 | hpos
  · have := k_le_weight true x
    rw [h0, Finset.sum_range_zero] at hz hroom
    rw [h0, Finset.sum_range_zero, hk]
    push_cast at hroom
    linarith
  have hper : ∀ j ∈ Finset.range n, 4 * tableLen true (st j) / Real.exp 1 +
      ∑ i, Claims.Flock.cZW (sch j) (Np j) i ≤ tableWeight true (st j) := fun j hj => by
    have hj := Finset.mem_range.1 hj
    exact cZW_sum_le_tableWeight (hsch j hj) (Np j) (hNp j hj)
  have hs := Finset.sum_le_sum hper
  rw [Finset.sum_add_distrib, ← Finset.sum_div, ← Finset.mul_sum, ← hN, ← hw] at hs
  have he := Real.exp_pos 1
  have hceil := Nat.ceil_lt_add_one (div_nonneg (mul_nonneg (by norm_num : (0 : ℝ) ≤ 4) (N₀_nonneg true x)) he.le)
  have hn1 : (1 : ℝ) ≤ n := by exact_mod_cast hpos
  rw [hk]
  linarith

end FlockDeploy.KS
