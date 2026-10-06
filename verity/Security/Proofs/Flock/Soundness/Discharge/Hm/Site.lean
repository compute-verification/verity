/-
`HmRowComputes` at e2e-layout's rows (`keyedRowsOuts_computes`). At a drawn unit's site in its table's accepted
statement, the `b ‖ c` a satisfying witness holds at row port `p`'s `hm96` output (`bcCol`, the address the regions pin
`Layout.bcAddr`) is the commit string of the row prefix and the bytes at the message columns, under the salt at `saltCol`
(`bc_site_col`, `bc_site_commit`): `row_bc_setupH`, with each column in the block (`bcCol_lt`, `saltCol_lt`,
`Layout.msgCol_lt_of_parse`), so the site's addresses read the block's bits (`bit_siteAddr`).
-/
import Proofs.Flock.Soundness.Discharge.Hm.Bc
import Proofs.Flock.Soundness.Discharge.Layout

open Flock FlockLevel3
open Flock.HmNets hiding and

namespace FlockSoundness.Discharge.Hm

open Model

/-- A laid-out net's output bits lie below its constant column. -/
theorem outStart_add_lt (st : St) (inBits outBits : Array ℕ) (outs : Array Lin) {t : ℕ}
    (ht : t < outBits.foldl (· + ·) 0) : outStart st inBits outBits + t < (layout st inBits outBits outs).constPos := by
  obtain ⟨h1, -⟩ := HmNets.layout_rows st inBits outBits outs
  have hob := HmNets.le_groupWords (outBits.foldl (· + ·) 0)
  rw [h1]
  unfold outStart inRowsOf spanOf
  omega

/-- The `hm96` net's output bits lie below its constant column. -/
theorem hmOut_lt {c : Circuit} (h : HmNets.check c = .ok ()) {t : ℕ} (ht : t < 1024) :
    hmOut c + t < (HmNets.hmLaid ()).constPos := by
  have hsum : HM_OUT.foldl (· + ·) 0 = 1024 := by decide
  rw [hmOut_eq h, HmNets.hmLaid_eq]
  exact outStart_add_lt _ _ _ _ (by rw [hsum]; exact ht)

section Cols

variable {c : Circuit} (R : RowRanges c)
include R

/-- `b ‖ c`'s columns are in the block. -/
theorem bcCol_lt {g p j t : ℕ} (hg : g < c.g) (hp : p < c.ports.size) (hj : j < 128) (ht : t < 8) :
    bcCol c g p j t < 2 ^ c.kLog := by
  have hn := R.range R.hmNet
  obtain ⟨-, -, -, -, hcp, -⟩ := Net.ofRows_ok (hmNet_ofRows R.rowNets)
  have hfit := R.fit c.hmNet
  have hlt := hmOut_lt R.rowNets (t := 8 * j + t) (by omega)
  refine AtSlot.lt R.bound hn.1 (n := c.hmNet) (s := g * c.perVu.getD c.hmNet 0 + p) (off := hmOut c + (8 * j + t))
    ⟨R.slot_lt_count (Or.inr rfl) hg (by rw [R.perVu_hm]; exact lt_of_lt_of_le hp (le_nextPow2 _)), ?_, ?_⟩
  · rw [hn.2.2.2]
    omega
  · unfold bcCol Circuit.slotCol
    omega

/-- The salt's columns are in the block. -/
theorem saltCol_lt {g p j t : ℕ} (hg : g < c.g) (hp : p < c.ports.size) (hj : j < 192) (ht : t < 8) :
    saltCol c g p j t < 2 ^ c.kLog := by
  have h2 := (hmIn_eq R.rowNets).2.2
  have hat := R.hmIn_at hg hp (off := hmIn c 2 + (8 * j + t)) (by omega)
  have := hat.lt R.bound (R.range R.hmNet).1
  unfold saltCol
  omega

/-- The `hm96` net's slots per template instance are a power of two, as `Layout.bcAddr` reads them. -/
theorem perVu_hm_pow2 : c.perVu.getD c.hmNet 0 = 2 ^ (c.perVu.getD c.hmNet 1).log2 := by
  have h0 := R.perVu_hm
  have hlt : c.hmNet < c.perVu.size := by
    by_contra hge
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_none (by omega), Option.getD_none] at h0
    exact absurd h0.symm (Nat.pos_iff_ne_zero.1 (Nat.two_pow_pos _))
  have h1 : c.perVu.getD c.hmNet 1 = c.perVu.getD c.hmNet 0 := by
    simp only [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hlt, Option.getD_some]
  rw [h1, h0]
  unfold nextPow2
  rw [Nat.log2_two_pow]

end Cols

section Site

variable {T : Model.Statement}

/-- An accepted statement's circuit has `RowRanges`. -/
theorem acc_rowRanges (a : Placed.Accepted T) : RowRanges a.st.c :=
  RowRanges.ofL (setupH_layout a.ht a.h) (Aliased.setupH_unitFacts a.ht a.h).parse
    (Aliased.setupH_unitFacts a.ht a.h).check

/-- `row_bc_setupH` in an accepted statement `T`. -/
theorem acc_row_bc (a : Placed.Accepted T) (ho : a.st.c.outNet = a.st.c.unitNet)
    (hv1 : ∀ p ∈ a.st.c.ports.toList, p.v2 = false ∧ p.bits = 16 * p.words ∧ 0 < p.chunks)
    {z : Witness T.m} (hz : T.Satisfies z) (o : Fin (2 ^ T.nbl)) {g p : ℕ} (hg : g < a.st.c.g)
    (hp : p < a.st.c.ports.size) :
    bytesOf (blockBit T z o) 128 (bcCol a.st.c g p) = Binding.commitString
      ((HmRow.rowPrefix a.st.c.ports[p]!.words).data.toList ++
        bytesOf (blockBit T z o) (2 * a.st.c.ports[p]!.words) (Typed.msgCol a.st.c g p))
      (bytesOf (blockBit T z o) 192 (saltCol a.st.c g p)) := by
  obtain ⟨tags, cf, pf, coins, tables, draw, partition, program, st, ht, h, regs, hT⟩ := a
  subst hT
  exact row_bc_setupH ht h ho hv1 hz o hg hp

/-- **A site address reads the block's bit**: in-block column `y` of block `o`, as an address of the accepted
statement `T`. -/
theorem bit_siteAddr (a : Placed.Accepted T) (z : Witness T.m) (o : Fin (2 ^ T.nbl)) {y : ℕ}
    (hy : y < 2 ^ a.st.c.kLog) : FlockSoundness.bit (z (Layout.siteAddr a o y)) = blockBit T z o y := by
  have hy' : y < 2 ^ T.kLog := by rw [← a.kLog_eq]; exact hy
  have hin : Layout.inBlockOr a.st.c.kLog y = ⟨y, hy⟩ := by
    simp only [Layout.inBlockOr, hy, ↓reduceDIte]
  unfold blockBit Layout.siteAddr
  simp only [hy', ↓reduceDIte]
  rw [hin]
  rfl

/-- Bytes read at in-block site addresses are the block's (`bytesOf`). -/
theorem ofFn_site (a : Placed.Accepted T) (z : Witness T.m) (o : Fin (2 ^ T.nbl)) (N : ℕ) (col : ℕ → ℕ → ℕ)
    (hcol : ∀ j < N, ∀ t < 8, col j t < 2 ^ a.st.c.kLog) :
    (List.ofFn fun j : Fin N => Binding.byteOf fun t => FlockSoundness.bit (z (Layout.siteAddr a o (col j.val t.val)))) =
      bytesOf (blockBit T z o) N col := by
  unfold bytesOf
  congr 1
  funext j
  congr 1
  funext t
  exact bit_siteAddr a z o (hcol j j.isLt t t.isLt)

/-- `ofFn_site` as `Binding.bytesAt`. -/
theorem bytesAt_site (a : Placed.Accepted T) (z : Witness T.m) (o : Fin (2 ^ T.nbl)) (N : ℕ) (col : ℕ → ℕ → ℕ)
    (hcol : ∀ j < N, ∀ t < 8, col j t < 2 ^ a.st.c.kLog) :
    Binding.bytesAt z (fun (j : Fin N) (t : Fin 8) => Layout.siteAddr a o (col j.val t.val)) =
      bytesOf (blockBit T z o) N col :=
  ofFn_site a z o N col hcol

/-- **A row's `b ‖ c` at a drawn unit's site**: in a satisfying witness of an accepted statement whose unit returns the
outputs and whose rows are `sha512/row/v1` rows (`hv1`), at block `o`, template instance `g` and row port `p`, the 128
bytes at the `hm96` slot's output (`bcCol`) are the commit string of the row prefix and the `2 · words` bytes at the
message columns (`Typed.msgCol`), under the 192 bytes at `saltCol`. -/
theorem bc_site_col (a : Placed.Accepted T) (ho : a.st.c.outNet = a.st.c.unitNet)
    (hv1 : ∀ p ∈ a.st.c.ports.toList, p.v2 = false ∧ p.bits = 16 * p.words ∧ 0 < p.chunks)
    {z : Witness T.m} (hz : T.Satisfies z) (o : Fin (2 ^ T.nbl)) {g p : ℕ} (hg : g < a.st.c.g)
    (hp : p < a.st.c.ports.size) :
    Binding.bytesAt z (fun (j : Fin 128) (t : Fin 8) => Layout.siteAddr a o (bcCol a.st.c g p j.val t.val)) =
      Binding.commitString ((HmRow.rowPrefix a.st.c.ports[p]!.words).data.toList ++
        List.ofFn fun j : Fin (2 * a.st.c.ports[p]!.words) => Binding.byteOf fun t =>
          FlockSoundness.bit (z (Layout.siteAddr a o (Typed.msgCol a.st.c g p j.val t.val))))
        (Binding.bytesAt z fun (j : Fin 192) (t : Fin 8) => Layout.siteAddr a o (saltCol a.st.c g p j.val t.val)) := by
  have R := acc_rowRanges a
  rw [bytesAt_site a z o 128 (bcCol a.st.c g p) (fun j hj t ht => bcCol_lt R hg hp hj ht),
    bytesAt_site a z o 192 (saltCol a.st.c g p) (fun j hj t ht => saltCol_lt R hg hp hj ht),
    ofFn_site a z o _ (Typed.msgCol a.st.c g p)
      (fun j _ t ht => Layout.msgCol_lt_of_parse (setupH_spec a.ht a.h).1 hg hp j t ht)]
  exact acc_row_bc a ho hv1 hz o hg hp

/-- `bc_site_col` at the address the regions pin (`Layout.bcAddr`). -/
theorem bc_site_commit (a : Placed.Accepted T) (ho : a.st.c.outNet = a.st.c.unitNet)
    (hv1 : ∀ p ∈ a.st.c.ports.toList, p.v2 = false ∧ p.bits = 16 * p.words ∧ 0 < p.chunks)
    {z : Witness T.m} (hz : T.Satisfies z) (o : Fin (2 ^ T.nbl)) {g p : ℕ} (hg : g < a.st.c.g)
    (hp : p < a.st.c.ports.size) :
    Binding.bytesAt z (fun (j : Fin 128) (t : Fin 8) => Layout.siteAddr a o (Layout.bcAddr a.st.c g p j.val t.val)) =
      Binding.commitString ((HmRow.rowPrefix a.st.c.ports[p]!.words).data.toList ++
        List.ofFn fun j : Fin (2 * a.st.c.ports[p]!.words) => Binding.byteOf fun t =>
          FlockSoundness.bit (z (Layout.siteAddr a o (Typed.msgCol a.st.c g p j.val t.val))))
        (Binding.bytesAt z fun (j : Fin 192) (t : Fin 8) => Layout.siteAddr a o (saltCol a.st.c g p j.val t.val)) := by
  have hbc : (fun (j : Fin 128) (t : Fin 8) => Layout.siteAddr a o (Layout.bcAddr a.st.c g p j.val t.val)) =
      fun j t => Layout.siteAddr a o (bcCol a.st.c g p j.val t.val) := by
    funext j t
    rw [Layout.bcAddr_eq_slotCol _ _ _ _ _ (perVu_hm_pow2 (acc_rowRanges a))]
    rfl
  rw [hbc]
  exact bc_site_col a ho hv1 hz o hg hp

end Site

/-! ## At e2e-layout's rows -/

section Keyed

open FlockSoundness.Discharge.Layout Placed

variable {c : Flock.Circuit} {hU : Built c.unit} {n : ℕ} {Reg : Type} {mPts : ℕ}
  {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
  {si : Placed.Sites plan tab}
  (hparse : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S),
    HmRow.parse (si.acc S R u h).tags (si.acc S R u h).cf (si.acc S R u h).tables = .ok c)
  {popOf : Reg → Public}

/-- **`HmRowComputes` at e2e-layout's rows** (`hHm` for `Layout.keyedRowsOuts` at `saltCol`): at every drawn unit's
committed wire held by a row port `q` (each a `sha512/row/v1` row of `16 · words` bits in at least one chunk, `hv1`,
which `Layout.scopeOk_ports` gives), in a satisfying witness of the unit's table, the `b ‖ c` at the port's `hm96`
output is the commit string of the row's bytes (`rowEnc`) and the salt the witness holds at `saltCol`. -/
theorem keyedRowsOuts_computes {tags : Tags} {bytes : ByteArray} {tables : List (String × ByteArray)}
    (hc : HmRow.parse tags bytes tables = .ok c) (h1 : c.unit.inGroups.size = 1) (hpr : PortReads c hU)
    (ho : c.outNet = c.unitNet) (hv1 : ∀ p ∈ c.ports.toList, p.v2 = false ∧ p.bits = 16 * p.words ∧ 0 < p.chunks)
    (hsite : ∀ S R u (h : u ∈ S) M, Binding.Sat plan tab S R u M →
      BcSite (si.acc S R u h) (popOf R) (si.o S R u h) (si.vu S R u h) u.val) :
    (keyedRowsOuts hparse saltCol hc h1 hpr ho hsite (popOf := popOf)).Computes := by
  unfold Binding.HmRows.Computes Assumptions.HmRowComputes
  rintro ⟨S, R, u, M, g⟩ ⟨hu, hM, -, hrow⟩
  dsimp only at hu hM hrow ⊢
  revert hrow
  generalize (keyedLayout hparse (outsOf c hU n) saltCol popOf).pos g = pos
  rcases pos with ⟨u', q⟩ | g'
  · intro _
    have hce := (si.acc S R u hu).c_eq (hparse S R u hu)
    have hq : q.val < (si.acc S R u hu).st.c.ports.size := by rw [hce]; exact q.isLt
    have hwq : portRowW c q = 16 * (si.acc S R u hu).st.c.ports[q.val]!.words := by
      rw [hce, getElem!_pos c.ports q.val q.isLt]
      rfl
    have hb := bc_site_commit (si.acc S R u hu) (by rw [hce]; exact ho) (by rw [hce]; exact hv1) hM (si.o S R u hu)
      (si.hvu S R u hu) hq
    show Binding.bytesAt _ (keyBcAddr si S R u q) = Binding.commitString
      (rowEnc ((keyLayout (portRowW c) hparse (outsOf c hU n) saltCol popOf (outBitOf c hU n (outsOf c hU n) popOf)
        bitCode).row S R u M (.inl (u', q))))
      (Binding.bytesAt _ (keySaltAddr si saltCol S R u (.inl (u', q)) (hU := hU)))
    have e1 : keyBcAddr si S R u q = fun (j : Fin 128) (t : Fin 8) => siteAddr (si.acc S R u hu) (si.o S R u hu)
        (bcAddr (si.acc S R u hu).st.c (si.vu S R u hu) q.val j.val t.val) := by
      funext j t
      simp only [keyBcAddr, hu, dite_true]
    have e2 : keySaltAddr si saltCol S R u (.inl (u', q)) (hU := hU) = fun (j : Fin 192) (t : Fin 8) =>
        siteAddr (si.acc S R u hu) (si.o S R u hu) (saltCol (si.acc S R u hu).st.c (si.vu S R u hu) q.val j.val t.val) := by
      funext j t
      simp only [keySaltAddr, hu, dite_true]
    rw [keyRow_enc hparse (outsOf c hU n) saltCol popOf (outBitOf c hU n (outsOf c hU n) popOf) bitCode hu M u' q hwq,
      e1, e2]
    exact hb
  · intro h
    exact (Bool.false_ne_true h).elim

end Keyed

end FlockSoundness.Discharge.Hm
