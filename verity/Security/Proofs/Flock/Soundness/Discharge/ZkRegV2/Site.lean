import Proofs.Flock.Soundness.Discharge.Hm.Site
import Proofs.Flock.Soundness.Discharge.Hidden.Wide.RowBc
import Proofs.Flock.Soundness.Discharge.ZkRegV2.RowEnc

set_option Elab.async false

/-!
# `HmRowComputes` at e2e-layout's rows, for `sha512/row/v2` rows

`Hm/Site.lean` for whole-block `sha512/row/v2` rows: `row_bc_setupH_tgt` is `Hidden.row_bc_tgt` at the untyped
statement `Stmt.setupH` accepted (a row's `b ‖ c` under its own prefix `HmRow.prefixOf`), and `bc_site_col_tgt`,
`bc_site_commit_tgt` carry it to a drawn unit's site. A v2 row of `words` words has the prefix `preV2 words`, P(16 ·
words) (`prefixOf_v2`), so `keyedRowsOutsV2_computes` is `keyedRowsOuts_computes` at `Layout.keyedRowsOutsP preV2`.
-/

open Flock FlockLevel3
open Flock.HmNets hiding and

namespace FlockSoundness.Discharge.Hm

open Model

/-- The `sha512/row/v2` prefix of a row of `w` u16 words: `P(16 · w)`. -/
def preV2 (w : ℕ) : ByteArray := HmRow.rowPrefixV2 (16 * w)

theorem preV2_size (w : ℕ) : (preV2 w).size = 128 := Hidden.size_rowPrefixV2 _

/-- An unsegmented bit row of `16 · words` bits has the prefix `preV2 words`. -/
theorem prefixOf_v2 {p : Flock.Port} (hv : p.v2 = true) (hs : p.segs = none) (hb : p.bits = 16 * p.words) :
    HmRow.prefixOf p = preV2 p.words := by
  unfold HmRow.prefixOf preV2
  simp [hs, hv, hb]

variable {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
  {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt}

/-- **A whole-block row's `b ‖ c` at the untyped statement `Stmt.setupH` accepted**, under the row's own prefix
(`Hidden.row_bc_tgt`), when the unit returns the outputs (`ho`) and no row is segmented (`hu`). -/
theorem row_bc_setupH_tgt (ht : tags.typed = false)
    (h : Stmt.setupH tags cf pf coins tables draw partition program = .ok st)
    (ho : st.c.outNet = st.c.unitNet) (hu : ∀ p ∈ st.c.ports.toList, p.segs = none)
    {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))} {z : Model.Witness ((setupH_layout ht h).model regs).m}
    (hz : ((setupH_layout ht h).model regs).Satisfies z)
    (o : Fin (2 ^ ((setupH_layout ht h).model regs).nbl)) {g p : ℕ} (hg : g < st.c.g) (hp : p < st.c.ports.size)
    (hb : st.c.ports[p]!.bits = 16 * st.c.ports[p]!.words)
    (hw : 2 * st.c.ports[p]!.words = 128 * st.c.ports[p]!.chunks) :
    bytesOf (blockBit ((setupH_layout ht h).model regs) z o) 128 (bcCol st.c g p) = Binding.commitString
      ((HmRow.prefixOf st.c.ports[p]!).data.toList ++
        bytesOf (blockBit ((setupH_layout ht h).model regs) z o) (2 * st.c.ports[p]!.words) (Typed.msgCol st.c g p))
      (bytesOf (blockBit ((setupH_layout ht h).model regs) z o) 192 (saltCol st.c g p)) :=
  Hidden.row_bc_tgt (setupH_layout ht h) (Aliased.setupH_unitFacts ht h) (setupH_spec ht h).2.2.1
    (upv_pos (Aliased.setupH_unitFacts ht h).check ho) hz o hg hp hb hw hu

section Site

variable {T : Model.Statement}

/-- `row_bc_setupH_tgt` in an accepted statement `T`. -/
theorem acc_row_bc_tgt (a : Placed.Accepted T) (ho : a.st.c.outNet = a.st.c.unitNet)
    (hu : ∀ p ∈ a.st.c.ports.toList, p.segs = none)
    {z : Witness T.m} (hz : T.Satisfies z) (o : Fin (2 ^ T.nbl)) {g p : ℕ} (hg : g < a.st.c.g)
    (hp : p < a.st.c.ports.size) (hb : a.st.c.ports[p]!.bits = 16 * a.st.c.ports[p]!.words)
    (hw : 2 * a.st.c.ports[p]!.words = 128 * a.st.c.ports[p]!.chunks) :
    bytesOf (blockBit T z o) 128 (bcCol a.st.c g p) = Binding.commitString
      ((HmRow.prefixOf a.st.c.ports[p]!).data.toList ++
        bytesOf (blockBit T z o) (2 * a.st.c.ports[p]!.words) (Typed.msgCol a.st.c g p))
      (bytesOf (blockBit T z o) 192 (saltCol a.st.c g p)) := by
  obtain ⟨tags, cf, pf, coins, tables, draw, partition, program, st, ht, h, regs, hT⟩ := a
  subst hT
  exact row_bc_setupH_tgt ht h ho hu hz o hg hp hb hw

/-- **A whole-block row's `b ‖ c` at a drawn unit's site**, under its own prefix: `bc_site_col` with
`acc_row_bc_tgt`. -/
theorem bc_site_col_tgt (a : Placed.Accepted T) (ho : a.st.c.outNet = a.st.c.unitNet)
    (hu : ∀ p ∈ a.st.c.ports.toList, p.segs = none)
    {z : Witness T.m} (hz : T.Satisfies z) (o : Fin (2 ^ T.nbl)) {g p : ℕ} (hg : g < a.st.c.g)
    (hp : p < a.st.c.ports.size) (hb : a.st.c.ports[p]!.bits = 16 * a.st.c.ports[p]!.words)
    (hw : 2 * a.st.c.ports[p]!.words = 128 * a.st.c.ports[p]!.chunks) :
    Binding.bytesAt z (fun (j : Fin 128) (t : Fin 8) => Layout.siteAddr a o (bcCol a.st.c g p j.val t.val)) =
      Binding.commitString ((HmRow.prefixOf a.st.c.ports[p]!).data.toList ++
        List.ofFn fun j : Fin (2 * a.st.c.ports[p]!.words) => Binding.byteOf fun t =>
          FlockSoundness.bit (z (Layout.siteAddr a o (Typed.msgCol a.st.c g p j.val t.val))))
        (Binding.bytesAt z fun (j : Fin 192) (t : Fin 8) => Layout.siteAddr a o (saltCol a.st.c g p j.val t.val)) := by
  have R := acc_rowRanges a
  rw [bytesAt_site a z o 128 (bcCol a.st.c g p) (fun j hj t ht => bcCol_lt R hg hp hj ht),
    bytesAt_site a z o 192 (saltCol a.st.c g p) (fun j hj t ht => saltCol_lt R hg hp hj ht),
    ofFn_site a z o _ (Typed.msgCol a.st.c g p)
      (fun j _ t ht => Layout.msgCol_lt_of_parse (setupH_spec a.ht a.h).1 hg hp j t ht)]
  exact acc_row_bc_tgt a ho hu hz o hg hp hb hw

/-- `bc_site_col_tgt` at the address the regions pin (`Layout.bcAddr`). -/
theorem bc_site_commit_tgt (a : Placed.Accepted T) (ho : a.st.c.outNet = a.st.c.unitNet)
    (hu : ∀ p ∈ a.st.c.ports.toList, p.segs = none)
    {z : Witness T.m} (hz : T.Satisfies z) (o : Fin (2 ^ T.nbl)) {g p : ℕ} (hg : g < a.st.c.g)
    (hp : p < a.st.c.ports.size) (hb : a.st.c.ports[p]!.bits = 16 * a.st.c.ports[p]!.words)
    (hw : 2 * a.st.c.ports[p]!.words = 128 * a.st.c.ports[p]!.chunks) :
    Binding.bytesAt z (fun (j : Fin 128) (t : Fin 8) => Layout.siteAddr a o (Layout.bcAddr a.st.c g p j.val t.val)) =
      Binding.commitString ((HmRow.prefixOf a.st.c.ports[p]!).data.toList ++
        List.ofFn fun j : Fin (2 * a.st.c.ports[p]!.words) => Binding.byteOf fun t =>
          FlockSoundness.bit (z (Layout.siteAddr a o (Typed.msgCol a.st.c g p j.val t.val))))
        (Binding.bytesAt z fun (j : Fin 192) (t : Fin 8) => Layout.siteAddr a o (saltCol a.st.c g p j.val t.val)) := by
  have hbc : (fun (j : Fin 128) (t : Fin 8) => Layout.siteAddr a o (Layout.bcAddr a.st.c g p j.val t.val)) =
      fun j t => Layout.siteAddr a o (bcCol a.st.c g p j.val t.val) := by
    funext j t
    rw [Layout.bcAddr_eq_slotCol _ _ _ _ _ (perVu_hm_pow2 (acc_rowRanges a))]
    rfl
  rw [hbc]
  exact bc_site_col_tgt a ho hu hz o hg hp hb hw

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

/-- **`HmRowComputes` at e2e-layout's `sha512/row/v2` rows** (`hHm` for `Layout.keyedRowsOutsP preV2` at `saltCol`):
`keyedRowsOuts_computes` for rows that are whole-block, unsegmented `sha512/row/v2` bit rows of `16 · words` bits
(`hv2`): at every drawn unit's committed wire held by a row port `q`, in a satisfying witness of the unit's table, the
`b ‖ c` at the port's `hm96` output is the commit string of the row's `sha512/row/v2` bytes (`rowEncP preV2`) and the
salt the witness holds at `saltCol`. -/
theorem keyedRowsOutsV2_computes {tags : Tags} {bytes : ByteArray} {tables : List (String × ByteArray)}
    (hc : HmRow.parse tags bytes tables = .ok c) (h1 : c.unit.inGroups.size = 1) (hpr : PortReads c hU)
    (ho : c.outNet = c.unitNet)
    (hv2 : ∀ p ∈ c.ports.toList, p.v2 = true ∧ p.segs = none ∧ p.bits = 16 * p.words ∧ 2 * p.words = 128 * p.chunks)
    (hsite : ∀ S R u (h : u ∈ S) M, Binding.Sat plan tab S R u M →
      BcSite (si.acc S R u h) (popOf R) (si.o S R u h) (si.vu S R u h) u.val) :
    (keyedRowsOutsP hparse saltCol preV2 preV2_size hc h1 hpr ho hsite (popOf := popOf)).Computes := by
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
    have hP : ∀ p ∈ (si.acc S R u hu).st.c.ports.toList,
        p.v2 = true ∧ p.segs = none ∧ p.bits = 16 * p.words ∧ 2 * p.words = 128 * p.chunks := by
      rw [hce]; exact hv2
    obtain ⟨hvq, hsq, hbq, hwq'⟩ := hP _ (getElem_mem_toList' hq)
    have hb := bc_site_commit_tgt (si.acc S R u hu) (by rw [hce]; exact ho) (fun p hp => (hP p hp).2.1) hM
      (si.o S R u hu) (si.hvu S R u hu) hq hbq hwq'
    rw [prefixOf_v2 hvq hsq hbq] at hb
    show Binding.bytesAt _ (keyBcAddr si S R u q) = Binding.commitString
      (rowEncP preV2 ((keyLayout (portRowW c) hparse (outsOf c hU n) saltCol popOf
        (outBitOf c hU n (outsOf c hU n) popOf) bitCode).row S R u M (.inl (u', q))))
      (Binding.bytesAt _ (keySaltAddr si saltCol S R u (.inl (u', q)) (hU := hU)))
    have e1 : keyBcAddr si S R u q = fun (j : Fin 128) (t : Fin 8) => siteAddr (si.acc S R u hu) (si.o S R u hu)
        (bcAddr (si.acc S R u hu).st.c (si.vu S R u hu) q.val j.val t.val) := by
      funext j t
      simp only [keyBcAddr, hu, dite_true]
    have e2 : keySaltAddr si saltCol S R u (.inl (u', q)) (hU := hU) = fun (j : Fin 192) (t : Fin 8) =>
        siteAddr (si.acc S R u hu) (si.o S R u hu) (saltCol (si.acc S R u hu).st.c (si.vu S R u hu) q.val j.val t.val) := by
      funext j t
      simp only [keySaltAddr, hu, dite_true]
    rw [keyRow_encP hparse (outsOf c hU n) saltCol popOf (outBitOf c hU n (outsOf c hU n) popOf) preV2 bitCode hu M
      u' q hwq, e1, e2]
    exact hb
  · intro h
    exact (Bool.false_ne_true h).elim

end Keyed

end FlockSoundness.Discharge.Hm
