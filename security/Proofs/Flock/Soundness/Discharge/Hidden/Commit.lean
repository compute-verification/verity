module

public import Proofs.Flock.Soundness.Discharge.Hidden.Flat
public import Proofs.Flock.Soundness.Discharge.Hidden.ParseCommit
public import Proofs.Flock.Soundness.Discharge.Hidden.SpecOf
public import Proofs.Flock.Soundness.Discharge.Hidden.Unpin
public import Proofs.Flock.Soundness.Discharge.Integrate.Bridge
public import Proofs.Flock.Soundness.TmplLayout
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Proofs.Flock.Soundness.Discharge.Exec.Merkle
import Proofs.Flock.Soundness.Discharge.Hidden.Delta
import Proofs.Flock.Soundness.Discharge.Hidden.Facts
import Proofs.Flock.Soundness.Discharge.Hidden.Layout
import Proofs.Flock.Soundness.Discharge.Hidden.Rows
import Proofs.Flock.Soundness.Discharge.Hidden.Spec
import Proofs.Flock.Soundness.ParseStmt
import Proofs.Flock.Soundness.ParseUnitPre
import Proofs.Flock.Soundness.Refine.Setup
import Proofs.Flock.Soundness.TmplNetOK
import Proofs.Flock.Soundness.TmplUnitAt
import Proofs.Flock.Soundness.TmplUnitNet

/-! A hidden-output session on the circuit it commits, `HmOut.parseCommit`'s, flat or templated. `ParseFactsPre` and
`Aliased.UnitFactsPre` hold of the circuit a templated circuit was read from, never of the committed circuit itself. -/

@[expose] public section

open Flock FlockLevel3
open Lean (Json)

namespace FlockSoundness.Discharge.Hidden

open Refine

variable {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
  {draw partition : Option Json} {program : Option ByteArray} {st : Stmt} {c : Circuit}

theorem parseCommit_untyped (ht : tags.typed = false) (h : HmOut.parseCommit tags cf tables = .ok c) :
    ∃ pre c₀, HmRow.parse tags cf tables (pre := pre) = .ok c₀ ∧ ParseFactsPre tags pre c₀ ∧
      HmOut.checkRows c₀ = .ok () ∧ c = { c₀ with tmpl := c.tmpl } ∧
      ((tags.isTmpl = false ∧ pre = #[] ∧ c.tmpl = none) ∨
        ∃ tu, tags.isTmpl = true ∧ pre = #[("unit", tu.net)] ∧ c.tmpl = some tu) := by
  cases hi : tags.isTmpl
  · rw [parseCommit_flat hi] at h
    obtain ⟨hp, hc⟩ := parse_untyped ht h
    have P := (parse_factsPre_nets hp).1
    exact ⟨#[], c, hp, P, hc, rfl, .inl ⟨rfl, rfl, P.tmpl⟩⟩
  · rw [parseCommit_tmpl hi] at h
    obtain ⟨tu, c₀, hp, hc, -, rfl⟩ := TmplValue.parseTmpl_cases h
    exact ⟨_, c₀, hp, (parse_factsPre_nets hp).1, hc, rfl, .inr ⟨tu, rfl, rfl, rfl⟩⟩

theorem parseCommit_check (ht : tags.typed = false) (h : HmOut.parseCommit tags cf tables = .ok c) :
    ∃ ls, HmRow.check c tags ls = .ok () := by
  obtain ⟨pre, c₀, -, P, -, he, -⟩ := parseCommit_untyped ht h
  obtain ⟨ls, hl⟩ := P.check
  exact ⟨ls, by rw [he]; exact hl⟩

theorem parseCommit_checkFacts (ht : tags.typed = false) (h : HmOut.parseCommit tags cf tables = .ok c) :
    CheckFacts c := by
  obtain ⟨pre, c₀, -, P, -, he, -⟩ := parseCommit_untyped ht h
  obtain ⟨ls, hl⟩ := parseCommit_check ht h
  have hty : c.typed = none := by rw [he]; exact P.typed
  exact check_facts (tags := tags) (ls := ls) hty hl

theorem parseCommit_leafScheme (h : HmOut.parseCommit tags cf tables = .ok c) : c.leafScheme = tags.merkleLeaf := by
  cases hi : tags.isTmpl
  · rw [parseCommit_flat hi] at h
    exact parse_leafScheme h
  · rw [parseCommit_tmpl hi] at h
    obtain ⟨tu, c₀, hp, -, -, rfl⟩ := TmplValue.parseTmpl_cases h
    have := Exec.parse_leafScheme_pre hp
    exact this

theorem parseCommit_slots (h : HmOut.parseCommit tags cf tables = .ok c) :
    ∀ r ∈ c.ranges.toList, 7 ≤ r.slotLog ∧ r.slotLog ≤ c.kLog := by
  cases hi : tags.isTmpl
  · rw [parseCommit_flat hi] at h
    exact parse_slots h
  · rw [parseCommit_tmpl hi] at h
    obtain ⟨tu, c₀, hp, -, -, rfl⟩ := TmplValue.parseTmpl_cases h
    obtain ⟨_, hls⟩ := hm_parse_checkPre hp
    have := hm_check_slots hls
    exact this

theorem setupHidden_parseCommit (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) :
    HmOut.parseCommit tags cf tables = .ok st.c :=
  (setupHidden_spec' h).parse

theorem setupHidden_specC (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) :
    SpecOf (HmOut.parseCommit tags cf tables) tags pf coins draw st :=
  setupHidden_specOf h

theorem setupHiddenOf_specC (hi : tags.isTmpl = true)
    (h : Stmt.setupHiddenOf (HmOut.parseTmpl tags cf tables) tags pf coins draw partition program = .ok st) :
    SpecOf (HmOut.parseCommit tags cf tables) tags pf coins draw st := by
  rw [parseCommit_tmpl hi]
  exact setupHiddenOf_spec h

theorem specC_hm96 (hl : tags.merkleLeaf = "hm96-sha512/v1")
    (s : SpecOf (HmOut.parseCommit tags cf tables) tags pf coins draw st) : st.merkle = MerkleScheme.hm96Sha512 := by
  have hf := s.merkle
  rw [parseCommit_leafScheme s.parse, hl, Exec.find_hm96] at hf
  exact (Option.some.inj hf).symm

theorem specC_plain (hl : tags.merkleLeaf = "flock-leaf/sha512-unsalted")
    (s : SpecOf (HmOut.parseCommit tags cf tables) tags pf coins draw st) :
    st.merkle = MerkleScheme.sha512 "flock-leaf/sha512-unsalted" := by
  have hf := s.merkle
  rw [parseCommit_leafScheme s.parse, hl, Exec.find_plain] at hf
  exact (Option.some.inj hf).symm

theorem specC_wf (s : SpecOf (HmOut.parseCommit tags cf tables) tags pf coins draw st) : StmtWF st ∧ RegionsWF st :=
  ⟨wf_of_facts (parseCommit_slots s.parse) (hm_pin_ok s.pin) s.m, s.m, s.mPts, regions_wf s.regions⟩

theorem outPairs_inBit_of
    (hports : ∀ n, ∀ q ∈ (portCols (c.nets.getD n default).2.inGroups).toList,
      q.1 + q.2 ≤ (c.nets.getD n default).2.inWords * WORD)
    (hu : c.unitNet < c.nets.size) (hs : c.shaNet < c.nets.size) (hh : c.hmNet < c.nets.size) (C : CheckFacts c)
    (hc : HmOut.checkRows c = .ok ()) : ∀ x ∈ outPairs c, InBit c x.1 := by
  intro x hx
  simp only [outPairs, List.mem_flatMap, List.mem_range, List.mem_cons, List.not_mem_nil, or_false] at hx
  obtain ⟨g, hg, j, hj, k, -, t, -, hx⟩ := hx
  have hlt := (checkRows_out hc hj).1
  have hr := deltaMsgCol_rowBit hg hlt (byte := 2 * k + t / 8) (Nat.mod_lt t (by norm_num))
  rcases hx with rfl | rfl <;> exact deltaIn_inBit_of hports hu hs hh C (hr.imp id Or.inl)

theorem hidden_layout_of
    (L : Layout { st with da := (HmRow.delta st.c st.pin).1, db := (HmRow.delta st.c st.pin).2 })
    (hty : st.c.typed = none) (hda : st.da = (HmOut.delta st.c st.pin).1) (hdb : st.db = (HmOut.delta st.c st.pin).2)
    (hout : ∀ x ∈ outPairs st.c, InBit st.c x.1) (hpin : HmRow.pinnedCount st.c.metaJ = 0) : Layout st := by
  have hd := hidden_delta_unpinned hpin st.pin
  obtain ⟨rest, h1, h2⟩ := L.delta
  exact {
    disj := L.disj
    bound := L.bound
    slotLog := L.slotLog
    fit := L.fit
    unit_mem := L.unit_mem
    unit_kind := L.unit_kind
    unit_count := L.unit_count
    unit_built := L.unit_built
    unit_at := L.unit_at
    unit_lookup := L.unit_lookup
    unit_in := L.unit_in
    unit_const := L.unit_const
    pin := L.pin
    delta_b := by rw [hdb, hda, hd, delta_snd _ _ hty]
    delta := by
      refine ⟨rest ++ outPairs st.c, ?_, fun x hx => ?_⟩
      · rw [hda, hd, Array.toList_append, List.toList_toArray]
        exact (congrArg (· ++ outPairs st.c) h1).trans (List.append_assoc _ _ _)
      · rcases List.mem_append.1 hx with hx | hx
        · exact h2 x hx
        · exact hout x hx
    kLog := L.kLog
    kLog6 := L.kLog6
    m := L.m }

theorem hidden_layoutC (ht : tags.typed = false) (s : SpecOf (HmOut.parseCommit tags cf tables) tags pf coins draw st)
    (hpin : HmRow.pinnedCount st.c.metaJ = 0) : Layout st := by
  cases hi : tags.isTmpl
  · have hP := s.parse
    rw [parseCommit_flat hi] at hP
    obtain ⟨hp, hc⟩ := parse_untyped ht hP
    have P := parse_facts hp
    exact hidden_layout_of (layout_of P s.pin rfl rfl s.m s.kLog) P.typed s.da s.db (outPairs_inBit hp hc) hpin
  · have hT := s.parse
    rw [parseCommit_tmpl hi] at hT
    obtain ⟨tu, c₀, P, hck, he, -, -⟩ := TmplValue.parseTmpl_facts hT
    have hok := parseTmpl_nets_ok hT
    obtain ⟨ls, hcheck⟩ := P.check
    have hty : st.c.typed = none := by rw [he]; exact P.typed
    have C : CheckFacts st.c := check_facts (tags := tags) (ls := ls) hty (by rw [he]; exact hcheck)
    have hun : st.c.unitNet < st.c.nets.size := by rw [he]; exact P.unitNet
    have hsh : st.c.shaNet < st.c.nets.size := by rw [he]; exact P.shaNet
    have hhm : st.c.hmNet < st.c.nets.size := by rw [he]; exact P.hmNet
    have hc : HmOut.checkRows st.c = .ok () := by rw [he]; exact hck
    exact hidden_layout_of (layout_of_tmpl hT s.pin rfl rfl s.m s.kLog) hty s.da s.db
      (outPairs_inBit_of (fun n => (hok n).1.2.2) hun hsh hhm C hc) hpin

theorem hidden_layout' : HiddenLayout := fun ht h hpin => hidden_layoutC ht (setupHidden_specC h) hpin

theorem unpin_layout_of
    (L : Layout { st with da := (HmRow.delta st.c st.pin).1, db := (HmRow.delta st.c st.pin).2 })
    (hty : st.c.typed = none) (hout : ∀ x ∈ outPairs st.c, InBit st.c x.1) : Layout (unpin st) := by
  obtain ⟨rest, h1, h2⟩ := L.delta
  exact {
    disj := L.disj
    bound := L.bound
    slotLog := L.slotLog
    fit := L.fit
    unit_mem := L.unit_mem
    unit_kind := L.unit_kind
    unit_count := L.unit_count
    unit_built := L.unit_built
    unit_at := L.unit_at
    unit_lookup := L.unit_lookup
    unit_in := L.unit_in
    unit_const := L.unit_const
    pin := L.pin
    delta_b := by
      show (HmRow.delta st.c st.pin).2 ++ (outPairs st.c).toArray = (HmRow.delta st.c st.pin).1 ++ (outPairs st.c).toArray
      rw [delta_snd _ _ hty]
    delta := by
      refine ⟨rest ++ outPairs st.c, ?_, fun x hx => ?_⟩
      · show ((HmRow.delta st.c st.pin).1 ++ (outPairs st.c).toArray).toList = constPairs st.c st.pin ++ (rest ++ outPairs st.c)
        rw [Array.toList_append, List.toList_toArray]
        exact (congrArg (· ++ outPairs st.c) h1).trans (List.append_assoc _ _ _)
      · rcases List.mem_append.1 hx with hx | hx
        · exact h2 x hx
        · exact hout x hx
    kLog := L.kLog
    kLog6 := L.kLog6
    m := L.m }

theorem unpin_layoutC (ht : tags.typed = false) (s : SpecOf (HmOut.parseCommit tags cf tables) tags pf coins draw st) :
    Layout (unpin st) := by
  cases hi : tags.isTmpl
  · have hP := s.parse
    rw [parseCommit_flat hi] at hP
    obtain ⟨hp, hc⟩ := parse_untyped ht hP
    have P := parse_facts hp
    exact unpin_layout_of (layout_of P s.pin rfl rfl s.m s.kLog) P.typed (outPairs_inBit hp hc)
  · have hT := s.parse
    rw [parseCommit_tmpl hi] at hT
    obtain ⟨tu, c₀, P, hck, he, -, -⟩ := TmplValue.parseTmpl_facts hT
    have hok := parseTmpl_nets_ok hT
    obtain ⟨ls, hcheck⟩ := P.check
    have hty : st.c.typed = none := by rw [he]; exact P.typed
    have C : CheckFacts st.c := check_facts (tags := tags) (ls := ls) hty (by rw [he]; exact hcheck)
    have hun : st.c.unitNet < st.c.nets.size := by rw [he]; exact P.unitNet
    have hsh : st.c.shaNet < st.c.nets.size := by rw [he]; exact P.shaNet
    have hhm : st.c.hmNet < st.c.nets.size := by rw [he]; exact P.hmNet
    have hc : HmOut.checkRows st.c = .ok () := by rw [he]; exact hck
    exact unpin_layout_of (layout_of_tmpl hT s.pin rfl rfl s.m s.kLog) hty
      (outPairs_inBit_of (fun n => (hok n).1.2.2) hun hsh hhm C hc)

theorem unpin_layout (ht : tags.typed = false)
    (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) : Layout (unpin st) :=
  unpin_layoutC ht (setupHidden_specC h)

theorem setupHidden_hm96 (hl : tags.merkleLeaf = "hm96-sha512/v1")
    (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) :
    st.merkle = MerkleScheme.hm96Sha512 :=
  specC_hm96 hl (setupHidden_specC h)

theorem setupHidden_plain (hl : tags.merkleLeaf = "flock-leaf/sha512-unsalted")
    (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) :
    st.merkle = MerkleScheme.sha512 "flock-leaf/sha512-unsalted" :=
  specC_plain hl (setupHidden_specC h)

theorem setupHidden_wf (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) :
    StmtWF st ∧ RegionsWF st :=
  specC_wf (setupHidden_specC h)

theorem unitFacts_tmpl (hT : HmOut.parseTmpl tags cf tables = .ok st.c) (hda : st.da = (HmRow.delta st.c st.pin).1) :
    Aliased.UnitFacts tags st := by
  obtain ⟨tu, c₀, hp, -, -, he⟩ := TmplValue.parseTmpl_cases hT
  obtain ⟨P, -, -, hsn, hhn⟩ := parse_factsPre_nets hp
  obtain ⟨hu0, h0, hun⟩ := unit_of_parsePre hp
  obtain ⟨ls, hcheck⟩ := P.check
  have hty : st.c.typed = none := by rw [he]; exact P.typed
  have C : CheckFacts st.c := check_facts (tags := tags) (ls := ls) hty (by rw [he]; exact hcheck)
  have htu : st.c.tmpl = some tu := by rw [he]
  have hu : st.c.unit = tu.net := by rw [he]; exact hun
  have hname : (c₀.nets.getD c₀.unitNet default).1 = "unit" := by rw [hu0, h0]
  have R := tmpl_net_rows tu
  refine ⟨(by rw [unitPre_tmpl htu, he]; exact P.toU.retmpl (some tu)), parseTmpl_nets_ok hT,
    fun i hi => (parseTmpl_net_width hT hi).2.2.2, C, (by rw [hda]; exact delta_flat st.c st.pin hty),
    fun e => ?_, fun e => ?_, (by rw [hu]; exact R.1.le), (by rw [hu]; exact R.2.1.le), (by rw [hu]; exact R.2.2)⟩
  · have e' : c₀.unitNet = c₀.shaNet := by rw [he] at e; exact e
    rw [← e', hname] at hsn
    exact absurd hsn (by unfold HmRow.SHA_NET; decide)
  · have e' : c₀.unitNet = c₀.hmNet := by rw [he] at e; exact e
    rw [← e', hname] at hhn
    exact absurd hhn (by unfold HmRow.HM_NET; decide)

theorem flat_layoutC (ht : tags.typed = false) (s : SpecOf (HmOut.parseCommit tags cf tables) tags pf coins draw st) :
    Layout (flatTwin st) := by
  cases hi : tags.isTmpl
  · have hP := s.parse
    rw [parseCommit_flat hi] at hP
    exact layout_of (st := flatTwin st) (parse_facts (parse_untyped ht hP).1) s.pin rfl rfl s.m s.kLog
  · have hT := s.parse
    rw [parseCommit_tmpl hi] at hT
    exact layout_of_tmpl (st := flatTwin st) hT s.pin rfl rfl s.m s.kLog

theorem flat_unitFactsC (ht : tags.typed = false)
    (s : SpecOf (HmOut.parseCommit tags cf tables) tags pf coins draw st) : Aliased.UnitFacts tags (flatTwin st) := by
  cases hi : tags.isTmpl
  · have hP := s.parse
    rw [parseCommit_flat hi] at hP
    exact unitFacts_of_parse (st := flatTwin st) (parse_untyped ht hP).1 rfl
  · have hT := s.parse
    rw [parseCommit_tmpl hi] at hT
    exact unitFacts_tmpl (st := flatTwin st) hT rfl

/-- **A committed circuit**: `HmOut.parse`'s (a flat file, or a flat session at any tags) or `HmOut.parseTmpl`'s. -/
def Committed (tags : Tags) (cf : ByteArray) (tables : List (String × ByteArray)) (c : Circuit) : Prop :=
  HmOut.parse tags cf tables = .ok c ∨ HmOut.parseTmpl tags cf tables = .ok c

theorem committed_of_commit (h : HmOut.parseCommit tags cf tables = .ok c) : Committed tags cf tables c := by
  cases hi : tags.isTmpl
  · rw [parseCommit_flat hi] at h
    exact .inl h
  · rw [parseCommit_tmpl hi] at h
    exact .inr h

theorem Committed.unpinned_of_rep (ht : tags.typed = false) (hC : Committed tags cf tables c)
    (h : c.outRowPorts = Array.replicate c.outPorts.size 1) : HmRow.pinnedCount c.metaJ = 0 := by
  rcases hC with hC | hC
  · obtain ⟨hp, hc⟩ := parse_untyped ht hC
    exact FlockSoundness.Discharge.Hidden.unpinned_of_rep hp hc h
  · obtain ⟨tu, c₀, hp, hck, -, rfl⟩ := TmplValue.parseTmpl_cases hC
    exact FlockSoundness.Discharge.Hidden.unpinned_of_rep (c := c₀) hp hck h

theorem setupHidden_unpinned (ht : tags.typed = false)
    (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st)
    (hr : st.c.outRowPorts = Array.replicate st.c.outPorts.size 1) : HmRow.pinnedCount st.c.metaJ = 0 :=
  (committed_of_commit (setupHidden_parseCommit h)).unpinned_of_rep ht hr

/-- **What a hidden-output session accepted, on its committed circuit**: `SpecOf` at `HmOut.parseCommit`
(`setupHidden_specC`, `setupHiddenOf_specC`), or at `HmOut.parse` (`Stmt.setupHidden`, at any tags). -/
def SpecC (tags : Tags) (cf pf : ByteArray) (coins : Coins) (tables : List (String × ByteArray)) (draw : Option Json)
    (st : Stmt) : Prop :=
  SpecOf (HmOut.parseCommit tags cf tables) tags pf coins draw st ∨
    SpecOf (HmOut.parse tags cf tables) tags pf coins draw st

theorem setupHidden_specCC (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) :
    SpecC tags cf pf coins tables draw st :=
  .inl (setupHidden_specOf h)

theorem SpecC.self (s : SpecC tags cf pf coins tables draw st) : SpecOf (.ok st.c) tags pf coins draw st :=
  Or.elim s (fun s => { s with parse := rfl }) (fun s => { s with parse := rfl })

theorem SpecC.committed (s : SpecC tags cf pf coins tables draw st) : Committed tags cf tables st.c :=
  Or.elim s (fun s => committed_of_commit s.parse) (fun s => .inl s.parse)

theorem SpecC.flatLayout (ht : tags.typed = false) (s : SpecC tags cf pf coins tables draw st) :
    Layout (flatTwin st) := by
  have s0 := s.self
  rcases s.committed with hP | hT
  · exact layout_of (st := flatTwin st) (parse_facts (parse_untyped ht hP).1) s0.pin rfl rfl s0.m s0.kLog
  · exact layout_of_tmpl (st := flatTwin st) hT s0.pin rfl rfl s0.m s0.kLog

theorem SpecC.unitFacts (ht : tags.typed = false) (s : SpecC tags cf pf coins tables draw st) :
    Aliased.UnitFacts tags (flatTwin st) := by
  rcases s.committed with hP | hT
  · exact unitFacts_of_parse (st := flatTwin st) (parse_untyped ht hP).1 rfl
  · exact unitFacts_tmpl (st := flatTwin st) hT rfl

theorem Committed.unrolled_unit (hty : tags.typed = false) (h : Committed tags cf tables c) :
    (HmOut.unrolled c).unit = lunit c ∧ (HmOut.unrolled c).tmpl = none ∧ (HmOut.unrolled c).upv = c.upv := by
  rcases h with h | h
  · have hf := parse_tmpl_none hty h
    rw [unrolled_flat hf, lunit_flat hf]
    exact ⟨rfl, hf, rfl⟩
  · obtain ⟨_tu, _n, -, -, -, -, -, -, he⟩ := unrolled_of_parseTmpl h
    exact ⟨FlockSoundness.unrolled_unit h, unrolled_tmpl h, by rw [he]⟩

theorem Committed.unrolled_eq (hty : tags.typed = false) (h : Committed tags cf tables c) :
    ∃ nets, HmOut.unrolled c = { c with nets := nets, tmpl := none } := by
  rcases h with h | h
  · have hf := parse_tmpl_none hty h
    refine ⟨c.nets, ?_⟩
    rw [unrolled_flat hf]
    cases c
    simp only at hf
    subst hf
    rfl
  · obtain ⟨_tu, _n, -, -, -, -, -, -, he⟩ := unrolled_of_parseTmpl h
    exact ⟨_, he⟩

theorem Committed.check (hty : tags.typed = false) (h : Committed tags cf tables c) :
    c.typed = none ∧ (∃ ls, HmRow.check c tags ls = .ok ()) ∧ HmOut.checkRows c = .ok () := by
  rcases h with h | h
  · obtain ⟨hp, hc⟩ := parse_untyped hty h
    have P := parse_facts hp
    exact ⟨P.typed, P.check, hc⟩
  · obtain ⟨tu, c₀, P, hck, he, -, -⟩ := TmplValue.parseTmpl_facts h
    obtain ⟨ls, hcheck⟩ := P.check
    exact ⟨by rw [he]; exact P.typed, ⟨ls, by rw [he]; exact hcheck⟩, by rw [he]; exact hck⟩

theorem Committed.netWidths (hty : tags.typed = false) (h : Committed tags cf tables c) : Integrate.NetWidths c := by
  rcases h with h | h
  · exact Integrate.netWidths_of (parse_facts (parse_untyped hty h).1)
  · exact fun _ hn => have w := parseTmpl_net_width h hn; ⟨w.1, w.2.1, w.2.2.1⟩

theorem Committed.netKinds (hty : tags.typed = false) (h : Committed tags cf tables c) : Integrate.NetKinds c := by
  rcases h with h | h
  · exact (parse_facts (parse_untyped hty h).1).kinds
  · obtain ⟨tu, c₀, P, -, he, -, -⟩ := TmplValue.parseTmpl_facts h
    subst he
    exact P.kinds

theorem Committed.factsU (hty : tags.typed = false) (h : Committed tags cf tables c) :
    ∃ pre, ParseFactsU tags pre c := by
  rcases h with h | h
  · exact ⟨#[], (parse_facts (parse_untyped hty h).1).toU⟩
  · obtain ⟨tu, c₀, P, -, rfl, -, -⟩ := TmplValue.parseTmpl_facts h
    exact ⟨_, P.toU.retmpl _⟩

theorem Committed.outRows (hty : tags.typed = false) (h : Committed tags cf tables c)
    (hv1 : ∀ p ∈ c.ports.toList, p.v2 = false) : OutRows c := by
  rcases h with h | h
  · obtain ⟨hp, hc⟩ := parse_untyped hty h
    exact out_rows_v1 hp hc hv1
  · obtain ⟨tu, c₀, hp, hck, -, rfl⟩ := TmplValue.parseTmpl_cases h
    have h := out_rows_v1Pre hp (parse_factsPre_nets hp).1 hck hv1
    exact h

theorem SpecC.layout (ht : tags.typed = false) (s : SpecC tags cf pf coins tables draw st)
    (hpin : HmRow.pinnedCount st.c.metaJ = 0) : Layout st :=
  Or.elim s (fun s => hidden_layoutC ht s hpin) fun s => by
    obtain ⟨hp, hc⟩ := parse_untyped ht s.parse
    have P := parse_facts hp
    exact hidden_layout_of (layout_of P s.pin rfl rfl s.m s.kLog) P.typed s.da s.db (outPairs_inBit hp hc) hpin

theorem SpecC.unpinLayout (ht : tags.typed = false) (s : SpecC tags cf pf coins tables draw st) : Layout (unpin st) :=
  Or.elim s (unpin_layoutC ht) fun s => by
    obtain ⟨hp, hc⟩ := parse_untyped ht s.parse
    have P := parse_facts hp
    exact unpin_layout_of (layout_of P s.pin rfl rfl s.m s.kLog) P.typed (outPairs_inBit hp hc)

theorem SpecC.model_eq (ht : tags.typed = false) (s : SpecC tags cf pf coins tables draw st) (L : Layout st)
    (hwf : StmtWF st) (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))) : L.model regs = stmtOf st hwf regs :=
  Integrate.model_eqW (Committed.netWidths ht s.committed) (Committed.netKinds ht s.committed) L s.self.leaf hwf regs

theorem SpecC.model_eq_unpin (ht : tags.typed = false) (s : SpecC tags cf pf coins tables draw st)
    (L : Layout (unpin st)) (hwf : StmtWF (unpin st)) (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))) :
    L.model regs = stmtOf (unpin st) hwf regs :=
  Integrate.model_eqW (st := unpin st) (Committed.netWidths ht s.committed) (Committed.netKinds ht s.committed) L
    s.self.leaf hwf regs

theorem SpecC.parseStmt_eq (ht : tags.typed = false) (s : SpecC tags cf pf coins tables draw st) :
    HmOut.parseStmt tags cf tables = .ok (HmOut.unrolled st.c) :=
  Or.elim s (fun s => parseStmt_of_commit ht s.parse) fun s => by
    rw [unrolled_flat (parse_tmpl_none ht s.parse)]
    exact parseStmt_of_parse (HmOut.parse_isTmpl s.parse) s.parse

theorem SpecC.parse_flat (hi : tags.isTmpl = false) (s : SpecC tags cf pf coins tables draw st) :
    HmOut.parse tags cf tables = .ok st.c :=
  Or.elim s (fun s => by rw [← parseCommit_flat hi]; exact s.parse) (fun s => s.parse)

end FlockSoundness.Discharge.Hidden
