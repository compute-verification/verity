import Properties.NetworkPeriod.ForcedZero
import Proofs.Properties.NetworkPeriod.PadCols
import Proofs.Flock.Claims.Scope.Setup
import Proofs.Flock.Soundness.ExecCircuit
import Proofs.Flock.Soundness.ExecSetup
import Proofs.Flock.Soundness.ExecFlatCopies
import Proofs.Flock.Soundness.Discharge.Exec.Setup
import Proofs.Flock.Soundness.Discharge.Composed.TagsJ
import Proofs.Flock.Soundness.Discharge.FailClosed.Scope
import Proofs.Flock.Soundness.Discharge.HiddenExec.Accepts
import Proofs.Flock.Soundness.Discharge.HiddenExec.CopiesP
import Proofs.Flock.Soundness.Discharge.HiddenExec.Flat
import Proofs.Flock.Soundness.Discharge.HiddenExec.Setup
import Proofs.Flock.Soundness.Discharge.Hidden.Commit
import Proofs.Flock.Soundness.Discharge.Hidden.Delta
import Proofs.Flock.Soundness.Discharge.Hidden.LayoutP
import Proofs.Flock.Soundness.Discharge.Hidden.PortsP
import Proofs.Flock.Soundness.Discharge.Hidden.SpecP
import Proofs.Flock.Soundness.Discharge.Integrate.Bridge
import Proofs.Flock.Soundness.Discharge.Aliased.Copies
import Proofs.Flock.Soundness.Refine.Setup
import Proofs.Flock.Soundness.Refine.StmtOf
import Proofs.Flock.Statement
import Flock.Witness

/-!
# No pair of Δ fills a forced-zero column, and its row of `A₀` is zero

`Properties.NetworkPeriod.ForcedZeroUnfilled` at the verifier's statement check and setup (`forcedZeroUnfilled`), its
step to the model statement (`forced_A₀_zero`), and that step composed with `Flock.Guarantees.ZeroRowForced`
(`forced_block_zero`):
- `forced_mem`: a forced-zero column is an input column below the unit's constant that no port holds, with empty rows in
  A and B;
- `placed_forced`: the placed rows of A and B there are zero where no pair of Δ has its row there, since the unit's
  slot reads its net's empty rows (`Layout.rangesEntry_unit`);
- `delta_unheld`: no pair of an untyped statement's flat Δ (`FlatDelta`) has its row at a unit slot's input column no
  port holds. Its constants are at or past the unit's constant (`Layout.constPairs_not_below`), its row bits and its
  wires into other nets are in other nets' slots (`Layout.ne_unit_atT`), and its leaf, cut and wire bits into the unit
  are at its port columns;
- `table_forced`: every table of an accepted session, at either setup. `Stmt.setupH`'s Δ is flat; `Stmt.setupHidden`'s
  is its flat twin's then the output rows' pairs (`Hidden.hidden_delta_unpinned`: the setup refuses pinned outputs),
  less the public leaves' columns, and the output rows' pairs are in other nets' slots on the rows the statement check
  admits (`HiddenExec.outPairs_ne_unitP`, from the scope's `v1Ports || bitRows`);
- `forced_A₀_zero`: that row of `Refine.stmtOf`'s `A₀`, the model statement's, is zero
  (`Integrate.placedA_eq`, `Integrate.placedA_eqW`);
- `forced_block_zero`: every satisfying witness holds 0 there in every block (`Flock.SecurityProofs.ZeroRowForced`);
- `hmRowParse_identity`, `parseTmpl_identity`, `hmOutParse_identity`, `circuitOf_circuit_identity`: no parse of a
  circuit file reads the tags' identity;
- `withCoins_identity`, `zkTags_identity`, `circuitOf_zkTags`: `--zk`'s tags (`Flock.Witness.zkTags`) are the
  statement's with another identity, so at `Tags.circuit` and no lookup tables a session's circuit is the one
  `ProvedScope.circuitOf Tags.circuit cf []` parses;
- `session_circuit`, `session_block_zero`: so is every table's of an accepted `--zk` session there
  (`Flock.Witness.scope`, `Flock.Witness.setupOf`), and every satisfying witness holds 0 at that circuit's forced-zero
  columns;
- `forcedZeroUnfilled_nontrivial`: `ForcedZeroUnfilledOf` fails at `acceptAllScope` and `fillForced`.
-/

namespace Properties.NetworkPeriod

open Flock FlockLevel3 FlockSoundness

/-- A forced-zero column is an input column below the unit's constant, no port holds it, and its rows of A and B are
empty. -/
theorem forced_mem {c : Flock.Circuit} {p : ℕ} (hp : p ∈ NetworkCheck.Period.forcedCols c) :
    p < c.unit.inWords * Flock.WORD ∧ p < c.unit.constPos ∧
      (∀ q ∈ (Flock.portCols c.unit.inGroups).toList, ¬ (q.1 ≤ p ∧ p < q.1 + q.2)) ∧
      NetworkCheck.Period.forcedZero c p = true := by
  simp only [forcedCols_eq, List.mem_filter, List.mem_range, Bool.and_eq_true, Option.isNone_iff_eq_none,
    List.find?_eq_none, decide_eq_true_eq] at hp
  obtain ⟨hlt, hnone, hz⟩ := hp
  refine ⟨lt_of_lt_of_le hlt (Nat.min_le_left _ _), lt_of_lt_of_le hlt (Nat.min_le_right _ _), ?_, hz⟩
  intro q hq hcov
  obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.1 (Array.mem_toList_iff.1 hq)
  have hg : (Flock.portCols c.unit.inGroups).getD i (0, 0) = (Flock.portCols c.unit.inGroups)[i] := by
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hi, Option.getD_some]
  exact hnone i hi (by rw [hg]; exact hcov)

/-- An empty row of a sparse matrix has no entries. -/
theorem sparseEntry_empty {m : Sparse} {i : ℕ} (h : (m.row i).isEmpty = true) (j : ℕ) :
    sparseEntry m i j = 0 := by
  have h0 : m.len.getD i 0 = 0 := by
    unfold Sparse.row at h
    simpa only [Array.isEmpty_iff_size_eq_zero, Array.size_map, Array.size_range] using h
  have hl : m.len[i]! = 0 := by
    by_cases hi : i < m.len.size
    · rw [getElem!_pos m.len i hi]
      rwa [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hi, Option.getD_some] at h0
    · rw [getElem!_neg m.len i hi]; rfl
  unfold sparseEntry
  split_ifs
  all_goals first
    | (rw [hl, List.range_zero, List.filter_nil, List.length_nil, Nat.cast_zero])
    | rfl

/-- **The placed rows at a forced-zero column are zero** where no pair of Δ has its row there: the unit's slot reads
its net's rows, which are empty there. -/
theorem placed_forced {st : Stmt} (L : FlockSoundness.Layout st) {s₀ p : ℕ} (hs₀ : s₀ < st.c.unitRange.count)
    (hb : p < 2 ^ st.c.unit.unitLog) (hz : NetworkCheck.Period.forcedZero st.c p = true)
    (hd : ∀ q ∈ st.da.toList ++ st.db.toList, q.1 ≠ st.c.unitRange.slot s₀ + p)
    (r j : Fin (2 ^ st.c.kLog)) (hr : r.val = st.c.unitRange.slot s₀ + p) :
    placedA st r j = 0 ∧ placedB st r j = 0 := by
  unfold NetworkCheck.Period.forcedZero at hz
  simp only [Bool.and_eq_true] at hz
  obtain ⟨⟨-, hA⟩, hB⟩ := hz
  have hd0 : ∀ d : Array (ℕ × ℕ), (∀ q ∈ d.toList, q.1 ≠ st.c.unitRange.slot s₀ + p) →
      deltaEntry d r.val j.val = 0 := fun d h => by
    have hn : (r.val, j.val) ∉ d.toList := fun hm => h _ hm hr
    rw [deltaEntry_eq_count, List.count_eq_zero.2 hn, Nat.cast_zero]
  constructor
  · show rangesEntry st (slotEntryA st) r.val j.val + deltaEntry st.da r.val j.val = 0
    rw [hd0 st.da fun q hq => hd q (List.mem_append_left _ hq), add_zero, hr, L.rangesEntry_unit hs₀ hb]
    split_ifs
    all_goals first
      | exact sparseEntry_empty hA _
      | rfl
  · show rangesEntry st (slotEntryB st) r.val j.val + deltaEntry st.db r.val j.val = 0
    rw [hd0 st.db fun q hq => hd q (List.mem_append_right _ hq), add_zero, hr, L.rangesEntry_unit hs₀ hb]
    split_ifs
    all_goals first
      | (show sparseEntry st.c.unit.b p _ + lookupEntry st.c.unit.lookup p _ = 0
         simp only [sparseEntry_empty hB, L.unit_lookup, lookupEntry, add_zero])
      | rfl

/-- **No pair of a flat Δ has its row at a unit slot's input column that no port holds**: its constants are at or past
the unit's constant, its row bits and its wires into other nets are in other nets' slots, and its leaf, cut and wire
bits into the unit are at the unit's port columns. -/
theorem delta_unheld {tags : Tags} {st : Stmt} (L : FlockSoundness.Layout st)
    (F : Discharge.Aliased.UnitFacts tags st) {s₀ p : ℕ} (hs₀ : s₀ < st.c.unitRange.count)
    (hin : p < st.c.unit.inWords * Flock.WORD)
    (hun : ∀ q ∈ (Flock.portCols st.c.unit.inGroups).toList, ¬ (q.1 ≤ p ∧ p < q.1 + q.2))
    {x : ℕ × ℕ} (hx : x ∈ st.da.toList) : x.1 ≠ st.c.unitRange.slot s₀ + p := by
  obtain ⟨rows, hd, hrows⟩ := F.delta
  have C := F.check
  have P := F.parse
  have hok : ∀ n, NetOK (st.c.nets.getD n default).2 := fun n => (F.nets n).1
  have hc : p < st.c.unit.constPos := lt_of_lt_of_le hin L.unit_in
  have hb : p < 2 ^ st.c.unit.unitLog := lt_of_lt_of_le hin (L.fit st.c.unitNet).2
  have hpW : p < 2 ^ st.c.unitRange.slotLog := by rw [L.unit_slotLog]; exact hb
  have hport : ∀ k jj tt, jj < (Flock.portCols st.c.unit.inGroups).size →
      tt < (Flock.portCols st.c.unit.inGroups)[jj]!.2 →
      st.c.unitRange.slot k + ((Flock.portCols st.c.unit.inGroups)[jj]!.1 + tt) ≠
        st.c.unitRange.slot s₀ + p := by
    intro k jj tt hjj htt heq
    obtain ⟨-, h2⟩ := slot_add_inj (L.port_lt_slot hjj htt) hpW heq
    exact hun _ (getElem_mem_toList' hjj) ⟨by omega, by omega⟩
  rw [hd] at hx
  simp only [List.mem_append] at hx
  rcases hx with (((hx | hx) | hx) | hx) | hx
  · intro he
    have hm : (st.c.unitRange.slot s₀ + p, x.2) ∈ constPairs st.c st.pin := by rw [← he]; exact hx
    exact L.constPairs_not_below L.unit_mem L.unit_kind hs₀ hc x.2 hm
  · intro heq
    have hws := rowSlots_spec st.c.ports
    have hall := rowSlots_all st.c.ports
    rcases hrows x hx with ⟨g, hg, pp, hpp, k, hk, i, hi, tt, htt, he⟩ | ⟨g, hg, pp, hpp, i, hi, tt, htt, he⟩
    · obtain ⟨h1, h2⟩ := hall _ (getElem_mem_toList' hpp) _ (getElem_mem_toList' hk)
      have hidx : 2 * ((HmRow.rowSlots st.c.ports).2[pp]!)[k]!.2 + i < 6 := by omega
      refine L.ne_unit_atT hok C P.shaNet (Ne.symm F.sha) hg (by rw [C.perVu_sha]; exact h1)
        (getElem_mem_toList' (by rw [C.sha_in.1]; exact hidx)) ?_ hs₀ hb (he ▸ heq)
      rw [C.sha_in.2 _ hidx, sha_width _ _ hi]
      exact htt
    · obtain ⟨hil, hw⟩ := hm_width C.hm_in hi
      refine L.ne_unit_atT hok C P.hmNet (Ne.symm F.hm) hg ?_ (getElem_mem_toList' hil)
        (by rw [hw]; exact htt) hs₀ hb (he ▸ heq)
      rw [C.perVu_hm, ← hws]
      exact lt_of_lt_of_le hpp (FlockSoundness.le_nextPow2 _)
  · unfold Flat.leafPairs at hx
    simp only [List.mem_flatMap, List.mem_range, List.mem_cons, List.not_mem_nil, or_false] at hx
    obtain ⟨k, -, jj, hjj, tt, htt, hx⟩ := hx
    have hx1 : x.1 = Flat.leafCol st.c k jj tt := by rcases hx with rfl | rfl <;> rfl
    have hupv : 0 < st.c.upv := by
      rcases Nat.eq_zero_or_pos st.c.upv with h0 | h0
      · rw [L.unit_count, h0, Nat.mul_zero] at hs₀; omega
      · exact h0
    have hmod := Nat.mod_lt k hupv
    have hlt : k % st.c.upv < st.c.leavesIn.size := by rw [C.leaves.1]; exact hmod
    have hli : (st.c.leavesIn.getD (k % st.c.upv) #[]).size = (st.c.unit.inGroups.getD 0 default).bits.size := by
      rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hlt, Option.getD_some,
        ← getElem!_pos st.c.leavesIn _ hlt]
      exact C.leaves.2 _ hmod
    rw [hli] at hjj
    obtain ⟨hjs, hcol, hw⟩ := firstGroup_port C.unit_groups.1 hjj
    have h16 : (st.c.unit.inGroups.getD 0 default).bits[jj]! = 16 :=
      C.unit_groups.2 _ (getElem_mem_toList' (xs := (st.c.unit.inGroups.getD 0 default).bits) hjj)
    have hlc : Flat.leafCol st.c k jj tt =
        st.c.unitRange.slot k + ((Flock.portCols st.c.unit.inGroups)[jj]!.1 + tt) := by
      unfold Flat.leafCol
      rw [hcol]
      omega
    rw [hx1, hlc]
    exact hport k jj tt hjs (by rw [hw, h16]; exact htt)
  · unfold Flat.cutPairs at hx
    simp only [List.mem_flatMap, List.mem_range, List.mem_cons, List.not_mem_nil, or_false] at hx
    obtain ⟨g, -, e, he, tt, htt, hx⟩ := hx
    have hx1 : x.1 = Flat.cutCol st.c g e + tt := by rcases hx with rfl | rfl <;> rfl
    obtain ⟨-, he2⟩ := C.cuts e he
    have hpv := L.perVu_eq C hs₀
    have hcol : Flat.cutCol st.c g e + tt =
        st.c.unitRange.slot (g * st.c.upv + e.1) + ((Flock.portCols st.c.unit.inGroups)[e.2.1]!.1 + tt) := by
      unfold Flat.cutCol Circuit.slotCol
      rw [hpv, Nat.add_assoc]
      rfl
    rw [hx1, hcol]
    exact hport _ _ _ he2 htt
  · unfold Flat.wirePairs at hx
    simp only [List.mem_flatMap, List.mem_range, List.mem_cons, List.not_mem_nil, or_false] at hx
    obtain ⟨g, hg, w, hw, tt, htt, hx⟩ := hx
    have hx1 : x.1 = Flat.wireDst st.c g w + tt := by rcases hx with rfl | rfl <;> rfl
    rw [hx1]
    obtain ⟨hd1, hd2, hd3⟩ := C.wires w hw
    by_cases hdu : w.dst.1 = st.c.unitNet
    · have hpv := L.perVu_eq C hs₀
      obtain ⟨⟨dn, di, dp⟩, src⟩ := w
      simp only at hdu hd2 hd3 htt ⊢
      subst hdu
      have hq : Flat.wireWidth st.c ⟨(st.c.unitNet, di, dp), src⟩ =
          (Flock.portCols st.c.unit.inGroups)[dp]!.2 := by
        unfold Flat.wireWidth
        exact congrArg Prod.snd (FlockSoundness.Layout.getD_portCols hd3)
      rw [hq] at htt
      have hcol : Flat.wireDst st.c g ⟨(st.c.unitNet, di, dp), src⟩ + tt =
          st.c.unitRange.slot (g * st.c.upv + di) + ((Flock.portCols st.c.unit.inGroups)[dp]!.1 + tt) := by
        unfold Flat.wireDst Circuit.slotCol
        simp only
        rw [hpv, FlockSoundness.Layout.getD_portCols hd3, Nat.add_assoc]
        rfl
      rw [hcol]
      exact hport _ _ _ hd3 htt
    · exact L.ne_unit_atT hok C hd1 hdu hg hd2 (getD_mem_toList hd3) htt hs₀ hb

/-- **Every table of an accepted session**: its circuit is the one `ProvedScope.circuitOf` parses, it has the layout
the placement reads, its placed A is `Refine.matA`, and no pair of its Δ has its row at a forced-zero column of a unit
slot. -/
theorem table_forced {tags : Tags} {zk : Bool} {cf pf : ByteArray} {tables : List (String × ByteArray)}
    {count : ℕ} {registered : Bool} {coins : Coins} {draw partition : Option Lean.Json}
    {program : Option ByteArray} {work : Option (ℕ × Array (String × ℕ × ℕ))} {stratified : Option ℕ}
    {closure : Option (Std.HashMap ℕ (Array ℕ))} {spec : SessionSpec} {sts : Array Stmt}
    (hck : ProvedScope.check tags zk cf pf tables count false registered = .ok ())
    (hset : Stmt.setupTables tags cf pf coins count tables draw partition program work stratified closure none =
      .ok (spec, sts)) {j : ℕ} (hj : j < sts.size) :
    ProvedScope.circuitOf tags cf tables = .ok sts[j].c ∧ FlockSoundness.Layout sts[j] ∧ Refine.StmtWF sts[j] ∧
      (∀ a b : Fin (2 ^ sts[j].c.kLog), placedA sts[j] a b = Refine.matA sts[j] a.val b.val) ∧
      ∀ s₀ < sts[j].c.unitRange.count, ∀ p ∈ NetworkCheck.Period.forcedCols sts[j].c,
        ∀ q ∈ sts[j].da.toList ++ sts[j].db.toList, q.1 ≠ sts[j].c.unitRange.slot s₀ + p := by
  have hok := Discharge.FailClosed.check_ok hck
  have ht : tags.typed = false := hok.1.2.1
  obtain ⟨-, hall⟩ := Discharge.HiddenExec.setupTables_setupOne hset
  obtain ⟨d, hd, -⟩ := hall j hj
  have ht' := (Discharge.Exec.tagsAt_typed tags count j).trans ht
  unfold Discharge.HiddenExec.setupOne at hd
  rw [Flock.Readable.tagsAt_hiddenOutputs] at hd
  cases hh : tags.hiddenOutputs
  · rw [hh] at hd
    simp only [Bool.false_eq_true, ↓reduceIte] at hd
    have L := setupH_layout ht' hd
    have F := Discharge.Aliased.setupH_unitFacts ht' hd
    have hp := (setupH_spec ht' hd).1
    refine ⟨?_, L, (Refine.setupH_wf hd).1, Discharge.Integrate.placedA_eq (parse_facts hp) L
      (Discharge.Integrate.setupH_leaf hd), fun s₀ hs₀ p hp' q hq => ?_⟩
    · rw [Discharge.Composed.parse_tagsAt] at hp
      simp only [ProvedScope.circuitOf, hh, Bool.false_eq_true, ↓reduceIte, hp]
    · obtain ⟨hin, -, hun, -⟩ := forced_mem hp'
      rw [L.delta_b, List.mem_append, or_self] at hq
      exact delta_unheld L F hs₀ hin hun hq
  · rw [hh] at hd
    simp only [↓reduceIte] at hd
    obtain ⟨c, _, hc, -, -, hscope⟩ := hok.2
    obtain ⟨pi, s⟩ := Discharge.Hidden.setupHidden_specP hd
    have hpin := Discharge.HiddenExec.setupTables_pinnedH hh hset j hj
    have L := Discharge.Hidden.hidden_layoutP ht' s hpin
    have Lf := Discharge.HiddenExec.flat_layoutP ht' s
    have F := Discharge.HiddenExec.flat_unitFactsP ht' s
    have hC := Discharge.Hidden.committed_of_commit s.parse
    have hcirc : ProvedScope.circuitOf tags cf tables = .ok sts[j].c := by
      rw [ProvedScope.circuitOf_hidden hh, ← Flock.Readable.parseCommit_tagsAt ht count j]
      exact s.parse
    obtain rfl : c = sts[j].c := Except.ok.inj (hc.symm.trans hcirc)
    have hrb : (Discharge.Layout.v1Ports sts[j].c || Discharge.Layout.bitRows sts[j].c) = true := by
      rcases hscope with ⟨-, hs, -⟩ | ⟨tu, -, hT⟩
      · rw [hh] at hs
        simp only [↓reduceIte] at hs
        unfold Discharge.Layout.scopeOkI at hs
        simp only [Bool.and_eq_true] at hs
        exact hs.2
      · unfold ProvedScope.scopeOkT at hT
        simp only [Bool.and_eq_true] at hT
        have e : ProvedScope.v2Row = Discharge.Layout.v2Row := rfl
        have e' : ProvedScope.segRow = Discharge.Layout.segRow := rfl
        have hr := hT.2
        simp only [ProvedScope.v1Ports, Discharge.Layout.v1Ports, ProvedScope.v1Rows, Discharge.Layout.v1Rows,
          ProvedScope.unpinned, Discharge.Layout.unpinned, ProvedScope.bitRows, Discharge.Layout.bitRows,
          ProvedScope.bitRowsU, Discharge.Layout.bitRowsU, e, e'] at hr ⊢
        exact hr
    have hrows := (Discharge.HiddenExec.rowsFit_of_committed ht' hC hrb).1.toP
    refine ⟨hcirc, L, (Discharge.HiddenExec.setupHidden_wf hd).1,
      Discharge.Integrate.placedA_eqW (Discharge.Hidden.Committed.netWidths ht' hC)
        (Discharge.Hidden.Committed.netKinds ht' hC) L s.leaf, fun s₀ hs₀ p hp q hq => ?_⟩
    obtain ⟨hin, -, hun, -⟩ := forced_mem hp
    have hb : p < 2 ^ sts[j].c.unit.unitLog := lt_of_lt_of_le hin (L.fit sts[j].c.unitNet).2
    have hdb : (HmRow.delta sts[j].c sts[j].pin).2 = (HmRow.delta sts[j].c sts[j].pin).1 := Lf.delta_b
    have hmem : q ∈ (HmRow.delta sts[j].c sts[j].pin).1.toList ∨ q ∈ Discharge.Hidden.outPairs sts[j].c := by
      rw [s.da, s.db, (Discharge.Hidden.inDelta_toList _ _ _).1, (Discharge.Hidden.inDelta_toList _ _ _).2,
        Discharge.Hidden.hidden_delta_unpinned hpin] at hq
      simp only [Array.toList_append, List.mem_append, List.mem_filter] at hq
      rcases hq with ⟨hq | hq, -⟩ | ⟨hq | hq, -⟩
      · exact Or.inl hq
      · exact Or.inr hq
      · rw [hdb] at hq; exact Or.inl hq
      · exact Or.inr hq
    rcases hmem with hq | hq
    · exact delta_unheld Lf F hs₀ hin hun hq
    · exact Discharge.HiddenExec.outPairs_ne_unitP Lf F hrows hq hs₀ hb

/-- **No pair of Δ fills a forced-zero column**, at the verifier's statement check and setup. -/
theorem forcedZeroUnfilled : ForcedZeroUnfilled := by
  unfold ForcedZeroUnfilled ForcedZeroUnfilledOf
  intro tags zk cf pf tables count registered coins draw partition program work stratified closure spec sts
    hck hset st hst
  obtain ⟨j, hj, rfl⟩ := Array.mem_iff_getElem.1 (Array.mem_toList_iff.1 hst)
  obtain ⟨hc, -, -, -, hd⟩ := table_forced hck hset hj
  exact ⟨hc, hd⟩

/-- **A forced-zero column's row of the model statement's `A₀` is zero**, at every unit slot of every table of an
accepted session: the step from `ForcedZeroUnfilled` to `Refine.stmtOf`. -/
theorem forced_A₀_zero {tags : Tags} {zk : Bool} {cf pf : ByteArray} {tables : List (String × ByteArray)}
    {count : ℕ} {registered : Bool} {coins : Coins} {draw partition : Option Lean.Json}
    {program : Option ByteArray} {work : Option (ℕ × Array (String × ℕ × ℕ))} {stratified : Option ℕ}
    {closure : Option (Std.HashMap ℕ (Array ℕ))} {spec : SessionSpec} {sts : Array Stmt}
    (hck : ProvedScope.check tags zk cf pf tables count false registered = .ok ())
    (hset : Stmt.setupTables tags cf pf coins count tables draw partition program work stratified closure none =
      .ok (spec, sts)) {j : ℕ} (hj : j < sts.size) (hwf : Refine.StmtWF sts[j])
    (regions : List (Model.Region sts[j].c.kLog (sts[j].m - sts[j].c.kLog))) {s₀ p : ℕ}
    (hs₀ : s₀ < sts[j].c.unitRange.count) (hp : p ∈ NetworkCheck.Period.forcedCols sts[j].c)
    (r : Fin (2 ^ sts[j].c.kLog)) (hr : r.val = sts[j].c.unitRange.slot s₀ + p) :
    (Refine.stmtOf sts[j] hwf regions).A₀ r = 0 := by
  obtain ⟨-, L, -, hmat, hd⟩ := table_forced hck hset hj
  obtain ⟨hin, -, -, hz⟩ := forced_mem hp
  have hb : p < 2 ^ sts[j].c.unit.unitLog := lt_of_lt_of_le hin (L.fit sts[j].c.unitNet).2
  funext c
  show Refine.matA sts[j] r.val c.val = 0
  rw [← hmat r c]
  exact (placed_forced L hs₀ hb hz (hd s₀ hs₀ p hp) r c hr).1

/-- **Every satisfying witness holds 0 at a forced-zero column**, in every block, at every unit slot of every table of
an accepted session: `forced_A₀_zero` with `Flock.Guarantees.ZeroRowForced`. -/
theorem forced_block_zero {tags : Tags} {zk : Bool} {cf pf : ByteArray} {tables : List (String × ByteArray)}
    {count : ℕ} {registered : Bool} {coins : Coins} {draw partition : Option Lean.Json}
    {program : Option ByteArray} {work : Option (ℕ × Array (String × ℕ × ℕ))} {stratified : Option ℕ}
    {closure : Option (Std.HashMap ℕ (Array ℕ))} {spec : SessionSpec} {sts : Array Stmt}
    (hck : ProvedScope.check tags zk cf pf tables count false registered = .ok ())
    (hset : Stmt.setupTables tags cf pf coins count tables draw partition program work stratified closure none =
      .ok (spec, sts)) {j : ℕ} (hj : j < sts.size) (hwf : Refine.StmtWF sts[j])
    (regions : List (Model.Region sts[j].c.kLog (sts[j].m - sts[j].c.kLog)))
    {z : Model.Witness (Refine.stmtOf sts[j] hwf regions).m} (hz : (Refine.stmtOf sts[j] hwf regions).Satisfies z)
    {s₀ p : ℕ} (hs₀ : s₀ < sts[j].c.unitRange.count) (hp : p ∈ NetworkCheck.Period.forcedCols sts[j].c)
    (r : Fin (2 ^ sts[j].c.kLog)) (hr : r.val = sts[j].c.unitRange.slot s₀ + p)
    (o : Fin (2 ^ (Refine.stmtOf sts[j] hwf regions).nbl)) :
    (Refine.stmtOf sts[j] hwf regions).block z o r = 0 :=
  Flock.SecurityProofs.ZeroRowForced _ z hz r (forced_A₀_zero hck hset hj hwf regions hs₀ hp r hr) o

set_option maxRecDepth 20000 in
-- `dsimp` reduces the tags' projections through the whole bodies of `parse` and its check, a few hundred binds each
set_option maxHeartbeats 1000000 in
/-- **`HmRow.parse` does not read the tags' identity**: it reads their file format, statement, bindings, block keyword,
typed flag and Merkle leaf, and its check (`HmRow.check`, `HmRow.checkTyped`) their hidden outputs and pinned leaf
scheme. -/
theorem hmRowParse_identity (t : Tags) (x : Lean.Json) (cf : ByteArray) (tbl : List (String × ByteArray))
    (tm : Option Typed.Template) (pre : Array (String × Net)) :
    HmRow.parse { t with identity := x } cf tbl tm pre = HmRow.parse t cf tbl tm pre := by
  unfold HmRow.parse HmRow.check HmRow.checkTyped
  dsimp -zeta only

/-- **A templated file's parse does not read the tags' identity**: only its `HmRow.parse` reads the tags. -/
theorem parseTmpl_identity (t : Tags) (x : Lean.Json) (cf : ByteArray) (tbl : List (String × ByteArray)) :
    HmOut.parseTmpl { t with identity := x } cf tbl = HmOut.parseTmpl t cf tbl := by
  unfold HmOut.parseTmpl
  simp only [hmRowParse_identity]

/-- **`HmOut.parse` of an untyped statement does not read the tags' identity**: it reads their name and typed flag,
then parses with `HmRow.parse`. -/
theorem hmOutParse_identity {t : Tags} (ht : t.typed = false) (x : Lean.Json) (cf : ByteArray)
    (tbl : List (String × ByteArray)) : HmOut.parse { t with identity := x } cf tbl = HmOut.parse t cf tbl := by
  unfold HmOut.parse Tags.isTmpl
  simp only [hmRowParse_identity]
  simp only [ht, Bool.not_false, ↓reduceIte]

/-- **`ProvedScope.circuitOf` at `Tags.circuit` does not read the identity**: `Tags.circuit` is untyped, so each of
`circuitOf`'s parses is one of the three above. -/
theorem circuitOf_circuit_identity (x : Lean.Json) (cf : ByteArray) (tbl : List (String × ByteArray)) :
    ProvedScope.circuitOf { Tags.circuit with identity := x } cf tbl = ProvedScope.circuitOf Tags.circuit cf tbl := by
  unfold ProvedScope.circuitOf HmOut.parseCommit Tags.isTmpl
  simp only [parseTmpl_identity, hmOutParse_identity (rfl : Tags.circuit.typed = false), hmRowParse_identity]

/-- **A session's coins change only the tags' identity** (`Tags.withCoins`). -/
theorem withCoins_identity (t : Tags) (coins : Coins) : ∃ x, t.withCoins coins = { t with identity := x } := by
  unfold Tags.withCoins
  split
  · exact ⟨_, rfl⟩
  · exact ⟨_, rfl⟩

/-- **`--zk`'s tags are the statement's with another identity**: `Flock.Witness.zkTags` sets the coins
(`Tags.withCoins`) and the identity's `hooks` (`Zk.tags`). -/
theorem zkTags_identity (I : Witness.Inputs) : ∃ x, Witness.zkTags I = { I.tags with identity := x } := by
  obtain ⟨y, hy⟩ := withCoins_identity I.tags (Witness.zkCoins I)
  unfold Witness.zkTags Zk.tags
  rw [hy]
  exact ⟨_, rfl⟩

/-- **A `--zk` session of `Tags.circuit` without lookup tables parses the circuit file as `Tags.circuit` does.** -/
theorem circuitOf_zkTags {I : Witness.Inputs} (ht : I.tags = Tags.circuit) (hT : I.tables = []) :
    ProvedScope.circuitOf (Witness.zkTags I) I.circuitFile I.tables =
      ProvedScope.circuitOf Tags.circuit I.circuitFile [] := by
  obtain ⟨x, hx⟩ := zkTags_identity I
  rw [hx, hT, ht]
  exact circuitOf_circuit_identity x _ _

/-- **Every table of an accepted `--zk` session of `Tags.circuit` without lookup tables has the circuit
`ProvedScope.circuitOf Tags.circuit` parses**, where its statement check (`Flock.Witness.scope`) accepts and it sets up
(`Flock.Witness.setupOf`). -/
theorem session_circuit {I : Witness.Inputs} (ht : I.tags = Tags.circuit) (hT : I.tables = []) {reg : Bool}
    {record : String} {spec : SessionSpec} {sts : Array Stmt} (hck : Witness.scope I reg = .ok ())
    (hset : Witness.setupOf I record = .ok (spec, sts)) {j : ℕ} (hj : j < sts.size) :
    ProvedScope.circuitOf Tags.circuit I.circuitFile [] = .ok sts[j].c :=
  (circuitOf_zkTags ht hT).symm.trans (table_forced hck hset hj).1

/-- **Every satisfying witness holds 0 at a forced-zero column of the circuit `Tags.circuit` parses**, in every block,
at every unit slot of every table of an accepted `--zk` session of that circuit file without lookup tables:
`session_circuit` with `forced_block_zero`. -/
theorem session_block_zero {I : Witness.Inputs} (ht : I.tags = Tags.circuit) (hT : I.tables = []) {reg : Bool}
    {record : String} {spec : SessionSpec} {sts : Array Stmt} (hck : Witness.scope I reg = .ok ())
    (hset : Witness.setupOf I record = .ok (spec, sts)) {c : Flock.Circuit}
    (hc : ProvedScope.circuitOf Tags.circuit I.circuitFile [] = .ok c) {j : ℕ} (hj : j < sts.size)
    (hwf : Refine.StmtWF sts[j]) (regions : List (Model.Region sts[j].c.kLog (sts[j].m - sts[j].c.kLog)))
    {z : Model.Witness (Refine.stmtOf sts[j] hwf regions).m} (hz : (Refine.stmtOf sts[j] hwf regions).Satisfies z)
    {s₀ p : ℕ} (hs₀ : s₀ < c.unitRange.count) (hp : p ∈ NetworkCheck.Period.forcedCols c)
    (r : Fin (2 ^ sts[j].c.kLog)) (hr : r.val = c.unitRange.slot s₀ + p)
    (o : Fin (2 ^ (Refine.stmtOf sts[j] hwf regions).nbl)) :
    (Refine.stmtOf sts[j] hwf regions).block z o r = 0 := by
  have e : c = sts[j].c := Except.ok.inj (hc.symm.trans (session_circuit ht hT hck hset hj))
  rw [e] at hs₀ hp hr
  exact forced_block_zero hck hset hj hwf regions hz hs₀ hp r hr o

/-- **`ForcedZeroUnfilledOf` is not trivially true**: at a check that accepts everything and a setup whose one table's
Δ has the pair `(0, 0)`, at the forced-zero column 0 of `forcedStmt`'s one unit slot, it fails. -/
theorem forcedZeroUnfilled_nontrivial : ¬ ForcedZeroUnfilledOf acceptAllScope fillForced := by
  intro h
  have hu : forcedStmt.c.unit = forcedNet := rfl
  have h0 : 0 ∈ NetworkCheck.Period.forcedCols forcedStmt.c := by
    simp only [forcedCols_eq, List.mem_filter, List.mem_range, NetworkCheck.Period.forcedZero, Sparse.row,
      Bool.and_eq_true, decide_eq_true_eq, Array.isEmpty_iff_size_eq_zero, Array.size_map, Array.size_range, hu]
    all_goals decide
  have hm : ((0 : ℕ), (0 : ℕ)) ∈ forcedStmt.da.toList ++ forcedStmt.db.toList :=
    List.mem_append_left _ (by decide)
  exact (h default false ByteArray.empty ByteArray.empty [] 0 false default none none none none none none default
    #[forcedStmt] rfl rfl forcedStmt (by simp)).2 0 (by decide) 0 h0 (0, 0) hm (by decide)

end Properties.NetworkPeriod
