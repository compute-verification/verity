import Proofs.Flock.Soundness.Discharge.Hidden.Layout
import Proofs.Flock.Soundness.Discharge.Hidden.ParseCommit
import Proofs.Flock.Soundness.Discharge.Hidden.SpecOf
import Proofs.Flock.Soundness.TmplLayout

/-!
# A hidden-output session on the committed circuit (PRF-133, ASK-112 option A)

A session commits the circuit `HmOut.parseCommit` reads: `HmOut.parse`'s for a flat file, `HmOut.parseTmpl`'s, which
carries its template, for a templated one (`Tags.isTmpl`). What the session side reads of `HmOut.parse … = .ok st.c` it
reads here of `HmOut.parseCommit … = .ok st.c`: the circuit is `HmRow.parse`'s with nets `pre` first, then
`HmOut.checkRows` (`parseCommit_untyped`), its leaf scheme and slots, the statement's well-formedness, and its layout
(`hidden_layoutC`). `setupHidden_parseCommit` and `setupHiddenOf_specC` give the session's `SpecOf` at
`HmOut.parseCommit`, flat and templated.

A templated circuit carries its template, so `ParseFactsPre` (whose `tmpl` is `none`) and `Aliased.UnitFactsPre` hold of
the circuit `c₀` it was read from (`parseCommit_untyped`), never of the committed circuit itself.
-/

open Flock FlockLevel3
open Lean (Json)

namespace FlockSoundness.Discharge.Hidden

open Refine

variable {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
  {draw partition : Option Json} {program : Option ByteArray} {st : Stmt} {c : Circuit}

/-! ## The committed circuit -/

/-- **`parse_untyped`'s `HmOut.parseCommit` form**: an untyped committed circuit is `HmRow.parse`'s with nets `pre`
first (none for a flat file, the template's unit net for a templated one), with its `ParseFactsPre`, then
`HmOut.checkRows`, and carries the template a templated file read. -/
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

/-- A committed circuit has the tags' leaf scheme. -/
theorem parseCommit_leafScheme (h : HmOut.parseCommit tags cf tables = .ok c) : c.leafScheme = tags.merkleLeaf := by
  cases hi : tags.isTmpl
  · rw [parseCommit_flat hi] at h
    exact parse_leafScheme h
  · rw [parseCommit_tmpl hi] at h
    obtain ⟨tu, c₀, hp, -, -, rfl⟩ := TmplValue.parseTmpl_cases h
    exact Exec.parse_leafScheme_pre hp

/-- A committed circuit's slots are `HmRow.check`'s: `slot_log` in 7..k_log. -/
theorem parseCommit_slots (h : HmOut.parseCommit tags cf tables = .ok c) :
    ∀ r ∈ c.ranges.toList, 7 ≤ r.slotLog ∧ r.slotLog ≤ c.kLog := by
  cases hi : tags.isTmpl
  · rw [parseCommit_flat hi] at h
    exact parse_slots h
  · rw [parseCommit_tmpl hi] at h
    obtain ⟨tu, c₀, hp, -, -, rfl⟩ := TmplValue.parseTmpl_cases h
    obtain ⟨_, hls⟩ := hm_parse_checkPre hp
    exact hm_check_slots hls

/-! ## The session -/

/-- **A flat session commits `HmOut.parseCommit`'s circuit**: `Stmt.setupHidden` reads `HmOut.parse`, which is
`HmOut.parseCommit` when the tags are not templated. -/
theorem setupHidden_parseCommit (hi : tags.isTmpl = false)
    (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) :
    HmOut.parseCommit tags cf tables = .ok st.c := by
  rw [parseCommit_flat hi]
  exact (setupHidden_spec' h).parse

/-- **A flat session's `SpecOf` at `HmOut.parseCommit`.** -/
theorem setupHidden_specC (hi : tags.isTmpl = false)
    (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) :
    SpecOf (HmOut.parseCommit tags cf tables) tags pf coins draw st := by
  rw [parseCommit_flat hi]
  exact setupHidden_specOf h

/-- **A templated session's `SpecOf` at `HmOut.parseCommit`**: `Stmt.setupHiddenOf` on `HmOut.parseTmpl`'s circuit. -/
theorem setupHiddenOf_specC (hi : tags.isTmpl = true)
    (h : Stmt.setupHiddenOf (HmOut.parseTmpl tags cf tables) tags pf coins draw partition program = .ok st) :
    SpecOf (HmOut.parseCommit tags cf tables) tags pf coins draw st := by
  rw [parseCommit_tmpl hi]
  exact setupHiddenOf_spec h

/-- **The leaf scheme**: under `hm96-sha512/v1` tags, a session on the committed circuit commits its rows with
`hm96Sha512`. -/
theorem specC_hm96 (hl : tags.merkleLeaf = "hm96-sha512/v1")
    (s : SpecOf (HmOut.parseCommit tags cf tables) tags pf coins draw st) : st.merkle = MerkleScheme.hm96Sha512 := by
  have hf := s.merkle
  rw [parseCommit_leafScheme s.parse, hl, Exec.find_hm96] at hf
  exact (Option.some.inj hf).symm

/-- **The leaf scheme, plain**, for a session on the committed circuit. -/
theorem specC_plain (hl : tags.merkleLeaf = "flock-leaf/sha512-unsalted")
    (s : SpecOf (HmOut.parseCommit tags cf tables) tags pf coins draw st) :
    st.merkle = MerkleScheme.sha512 "flock-leaf/sha512-unsalted" := by
  have hf := s.merkle
  rw [parseCommit_leafScheme s.parse, hl, Exec.find_plain] at hf
  exact (Option.some.inj hf).symm

/-- **The statement is well formed** (`StmtWF`, `RegionsWF`), for a session on the committed circuit. -/
theorem specC_wf (s : SpecOf (HmOut.parseCommit tags cf tables) tags pf coins draw st) : StmtWF st ∧ RegionsWF st :=
  ⟨wf_of_facts (parseCommit_slots s.parse) (hm_pin_ok s.pin) s.m, s.m, s.mPts, regions_wf s.regions⟩

/-! ## The layout -/

/-- **Every first column of `outPairs` is an input bit of a net slot**, for a circuit whose nets' ports lie in their
input rows, whose unit, `sha512x3` and `hm96` nets are among its nets, with `CheckFacts`, which `HmOut.checkRows`
passed. -/
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

/-- **A hidden-output statement's layout from its flat twin's**: `HmOut.delta` appends `outPairs` to both sides of
`HmRow.delta`, and every first column of `outPairs` is an input bit. -/
theorem hidden_layout_of
    (L : Layout { st with da := (HmRow.delta st.c st.pin).1, db := (HmRow.delta st.c st.pin).2 })
    (hty : st.c.typed = none) (hda : st.da = (HmOut.delta st.c st.pin).1) (hdb : st.db = (HmOut.delta st.c st.pin).2)
    (hout : ∀ x ∈ outPairs st.c, InBit st.c x.1) : Layout st := by
  have hd := hidden_delta' st.c st.pin
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

/-- **`HiddenLayout` on the committed circuit**: an untyped session on `HmOut.parseCommit`'s circuit has the layout the
placement reads, its unit the flat file's or, templated, the template's logical net placed at `posMap`
(`layout_of_tmpl`). -/
theorem hidden_layoutC (ht : tags.typed = false) (s : SpecOf (HmOut.parseCommit tags cf tables) tags pf coins draw st) :
    Layout st := by
  cases hi : tags.isTmpl
  · have hP := s.parse
    rw [parseCommit_flat hi] at hP
    obtain ⟨hp, hc⟩ := parse_untyped ht hP
    have P := parse_facts hp
    exact hidden_layout_of (layout_of P s.pin rfl rfl s.m s.kLog) P.typed s.da s.db (outPairs_inBit hp hc)
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
      (outPairs_inBit_of (fun n => (hok n).1.2.2) hun hsh hhm C hc)

/-- **`hidden_layout'` on a flat session, through `hidden_layoutC`.** -/
theorem hidden_layout_flat (ht : tags.typed = false) (hi : tags.isTmpl = false)
    (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) : Layout st :=
  hidden_layoutC ht (setupHidden_specC hi h)

end FlockSoundness.Discharge.Hidden
