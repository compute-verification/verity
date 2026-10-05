import Proofs.Pouw.PearlC.Cap.IntForm

/-!
# rev1's `capOK` is the cap circuit's verdict

`tileDebitRev1` (`DeviceRev1.lean`) is `debitDev ⟨atoms, 0⟩ + fs·(|C|/N)·elems + add·zeros` in three counts over the tile's
passing rows and every column: the flagged atom-words (`tileAtoms`), the forming elements (`tileElems`) and the later
promotions into a +0 total (`tileZeros`), as Lean writes them (`tileDebitRev1_eq`). Each stays within what a tile of
`tm × tn` words reaches (`tileAtoms_le`, `tileZeros_le`, `tileElems_le`), so under `_cap_constants`' bounds the word tests
on these counts decide `pearlCTilesDevRev1`'s `capOK` at `ρ = 1/RHO` (`capOK_iff`).
-/

namespace Pouw.PearlC.Cap

open Finset Pouw.PearlC Pouw.Fp8Atom

variable {Q R S : Type}

open Classical in
/-- `tileDebitDev`'s `atoms`: the flagged atom-words of the tile's words. -/
noncomputable def tileAtoms (sem : PearlCSem Q R S) (d : PearlCDevice) (H : Q → R) (s : S) (U : Workload) (u : ℕ)
    (rows cols : Finset ℕ) (act : Codes) : ℕ :=
  let sh := U.layout.shape u
  let AB := sem.unitChain H s U u act
  let fl := d.flags AB.1 AB.2
  ∑ e ∈ (univ : Finset (Fin sh.m × Fin sh.n)).filter (fun e => (e.1 : ℕ) ∈ rows ∧ (e.2 : ℕ) ∈ cols),
    ((range (sh.k / 32)).filter fun t => fl e.1 e.2 t = true).card

open Classical in
/-- `tileDebitDev`'s `elems`: per row, the elements salt-dead or flagged in every column of the tile. -/
noncomputable def tileElems (sem : PearlCSem Q R S) (d : PearlCDevice) (H : Q → R) (s : S) (U : Workload) (u : ℕ)
    (rows cols : Finset ℕ) (act : Codes) : ℕ :=
  let sh := U.layout.shape u
  let X := wordsVal sh.m sh.k act
  let AB := sem.unitChain H s U u act
  let fl := d.flags AB.1 AB.2
  ((univ : Finset (Fin sh.m × Fin sh.k)).filter fun e => (e.1 : ℕ) ∈ rows ∧
    (d.saltDead sem.alpha sem.beta sh.k (X e.1) e.2 ∨
      ∀ j : Fin sh.n, (j : ℕ) ∈ cols → fl e.1 j ((e.2 : ℕ) / 32) = true)).card

open Classical in
/-- `tileDebitRev1`'s later promotions into a +0 total over the tile's words. -/
noncomputable def tileZeros (sem : PearlCSem Q R S) (d : PearlCDevice) (H : Q → R) (s : S) (U : Workload) (u : ℕ)
    (rows cols : Finset ℕ) (act : Codes) : ℕ :=
  let sh := U.layout.shape u
  let AB := sem.unitChain H s U u act
  ∑ e ∈ (univ : Finset (Fin sh.m × Fin sh.n)).filter (fun e => (e.1 : ℕ) ∈ rows ∧ (e.2 : ℕ) ∈ cols),
    laterZeroPromos d AB.1 AB.2 e.1 e.2

theorem tileDebitRev1_eq (sem : PearlCSem Q R S) (d : PearlCDevice) (H : Q → R) (s : S) (U : Workload) (u : ℕ)
    (rows cols : Finset ℕ) (act : Codes) :
    sem.tileDebitRev1 d H s U u rows cols act =
      debitDev d ⟨tileAtoms sem d H s U u rows cols act, 0⟩ +
        (d.prices.costs.fs : ℚ) * (cols.card / (U.layout.shape u).n) * tileElems sem d H s U u rows cols act +
        d.prices.add * tileZeros sem d H s U u rows cols act := by
  simp only [PearlCSem.tileDebitRev1, PearlCSem.tileDebitDev, tileAtoms, tileElems, tileZeros]
  push_cast
  rfl

theorem card_tileWords_le {m n : ℕ} (rows cols : Finset ℕ) :
    ((univ : Finset (Fin m × Fin n)).filter (fun e => (e.1 : ℕ) ∈ rows ∧ (e.2 : ℕ) ∈ cols)).card ≤
      rows.card * cols.card := by
  rw [← card_product]
  apply card_le_card_of_injOn (fun e => ((e.1 : ℕ), (e.2 : ℕ)))
  · intro e he
    simp only [coe_filter, mem_univ, true_and, Set.mem_ofPred_eq] at he
    simp [he.1, he.2]
  · intro e1 _ e2 _ h
    simp only [Prod.mk.injEq] at h
    exact Prod.ext (Fin.ext h.1) (Fin.ext h.2)

theorem tileAtoms_le (sem : PearlCSem Q R S) (d : PearlCDevice) (H : Q → R) (s : S) (U : Workload) (u : ℕ)
    (rows cols : Finset ℕ) (act : Codes) :
    tileAtoms sem d H s U u rows cols act ≤ rows.card * cols.card * ((U.layout.shape u).k / 32) := by
  unfold tileAtoms
  refine (sum_le_sum fun e _ => (card_filter_le _ _).trans (card_range _).le).trans ?_
  rw [sum_const, smul_eq_mul]
  exact Nat.mul_le_mul_right _ (card_tileWords_le rows cols)

theorem laterZeroPromos_le (d : PearlCDevice) {g : ℕ} (hG : d.G = g) {m n k : ℕ} (A : Matrix (Fin m) (Fin k) Code)
    (B : Matrix (Fin n) (Fin k) Code) (i : Fin m) (j : Fin n) : laterZeroPromos d A B i j ≤ k / (32 * g) := by
  unfold laterZeroPromos
  split_ifs
  · exact Nat.zero_le _
  · rw [hG]
    refine (card_filter_le _ _).trans ?_
    rw [Nat.card_Ico]
    exact Nat.sub_le _ _

theorem tileZeros_le (sem : PearlCSem Q R S) (d : PearlCDevice) {g : ℕ} (hG : d.G = g) (H : Q → R) (s : S)
    (U : Workload) (u : ℕ) (rows cols : Finset ℕ) (act : Codes) :
    tileZeros sem d H s U u rows cols act ≤ rows.card * cols.card * ((U.layout.shape u).k / (32 * g)) := by
  unfold tileZeros
  refine (sum_le_sum fun e _ => laterZeroPromos_le d hG _ _ e.1 e.2).trans ?_
  rw [sum_const, smul_eq_mul]
  exact Nat.mul_le_mul_right _ (card_tileWords_le rows cols)

theorem tileElems_le (sem : PearlCSem Q R S) (d : PearlCDevice) (H : Q → R) (s : S) (U : Workload) (u : ℕ)
    (rows cols : Finset ℕ) (act : Codes) :
    tileElems sem d H s U u rows cols act ≤ rows.card * (U.layout.shape u).k := by
  unfold tileElems
  calc _ ≤ (rows ×ˢ range (U.layout.shape u).k).card := by
        apply card_le_card_of_injOn (fun e => ((e.1 : ℕ), (e.2 : ℕ)))
        · intro e he
          simp only [coe_filter, mem_univ, true_and, Set.mem_ofPred_eq] at he
          simp [he.1, e.2.isLt]
        · intro e1 _ e2 _ h
          simp only [Prod.mk.injEq] at h
          exact Prod.ext (Fin.ext h.1) (Fin.ext h.2)
    _ = rows.card * (U.layout.shape u).k := by rw [card_product, card_range]

/-- **A tile's cap is the word tests on its counts**: on `_cap_constants`' domain and within its bounds, at a promoted
record with prices `p`, a unit of depth `k` and `n` columns, a tile of `tn` columns and at most `tm` rows, rev1's tile
debit is at most `ρ·creditDevRev1·tileShare` (ρ = 1/RHO) iff `capTestsW` holds on the tile's counts. -/
theorem tileCap_iff (p : IntPrices) {d : PearlCDevice} {g k tm tn n rho : ℕ} (hD : CapDomain g k tm tn n rho)
    (hF : CapFits p g k tm tn n rho) (hG : d.G = g) (hp : d.prices = p.toPrices) (sem : PearlCSem Q R S) (H : Q → R)
    (s : S) (U : Workload) (u : ℕ) (rows cols : Finset ℕ) (act : Codes) (hk : (U.layout.shape u).k = k)
    (hn : (U.layout.shape u).n = n) (hm : 0 < (U.layout.shape u).m) (hc : cols.card = tn) (hr : rows.card ≤ tm) :
    sem.tileDebitRev1 d H s U u rows cols act ≤
        1 / (rho : ℚ) * creditDevRev1 d (U.layout.shape u) * tileShare (U.layout.shape u) rows cols ↔
      capTestsW (capConstants p g k tm tn n rho) tm (tileAtoms sem d H s U u rows cols act)
        (tileZeros sem d H s U u rows cols act) (tileElems sem d H s U u rows cols act) rows.card = true := by
  have hk1 : 32 * g ≤ k := by
    have := Nat.div_mul_le_self k (32 * g)
    have := hD.two
    nlinarith
  have hn0 : 0 < n := lt_of_lt_of_le hD.tn_pos hD.tn_le
  have ha := tileAtoms_le sem d H s U u rows cols act
  have hz := tileZeros_le sem d hG H s U u rows cols act
  have he := tileElems_le sem d H s U u rows cols act
  rw [hk, hc] at ha hz
  rw [hk] at he
  rw [tileDebitRev1_eq, cap_iff_int p hG hD.g_pos hp _ (hk ▸ hD.window) (hk ▸ hk1) hm (hn ▸ hn0) hD.rho_pos,
    capTestsW_iff p hD hF (ha.trans (by gcongr)) (hz.trans (by gcongr)) (he.trans (by gcongr)) hr, hk, hn, hc]

/-- **rev1's `capOK` is the circuit's verdict**: `pearlCTilesDevRev1`'s `capOK` at `ρ = 1/RHO` on tile `t` holds iff the word
tests hold on the counts of its passing rows and every column, under the hypotheses of `tileCap_iff` at that tile. -/
theorem capOK_iff (p : IntPrices) {d : PearlCDevice} {g k tm tn n rho : ℕ} (hD : CapDomain g k tm tn n rho)
    (hF : CapFits p g k tm tn n rho) (hG : d.G = g) (hp : d.prices = p.toPrices) (sem : PearlCSem Q R S) (H : Q → R)
    (s : S) (U : Workload) (t : ℕ) (act : Codes)
    (hk : (U.layout.shape ((auditTiling 64 64 U.layout).unit t)).k = k)
    (hn : (U.layout.shape ((auditTiling 64 64 U.layout).unit t)).n = n)
    (hm : 0 < (U.layout.shape ((auditTiling 64 64 U.layout).unit t)).m)
    (hc : ((auditTiling 64 64 U.layout).cols t).card = tn)
    (hr : (sem.passRows (U.layout.shape ((auditTiling 64 64 U.layout).unit t)) act
      ((auditTiling 64 64 U.layout).rows t)).card ≤ tm) :
    (pearlCTilesDevRev1 d sem (1 / (rho : ℚ))).capOK H s U t act ↔
      let tl := auditTiling 64 64 U.layout
      let rows := sem.passRows (U.layout.shape (tl.unit t)) act (tl.rows t)
      capTestsW (capConstants p g k tm tn n rho) tm (tileAtoms sem d H s U (tl.unit t) rows (tl.cols t) act)
        (tileZeros sem d H s U (tl.unit t) rows (tl.cols t) act) (tileElems sem d H s U (tl.unit t) rows (tl.cols t) act)
        rows.card = true :=
  tileCap_iff p hD hF hG hp sem H s U _ _ _ act hk hn hm hc hr

end Pouw.PearlC.Cap
