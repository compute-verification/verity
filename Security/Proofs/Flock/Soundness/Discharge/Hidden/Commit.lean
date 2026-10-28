import Proofs.Flock.Soundness.Discharge.Hidden.Flat
import Proofs.Flock.Soundness.Discharge.Hidden.Layout
import Proofs.Flock.Soundness.Discharge.Hidden.ParseCommit
import Proofs.Flock.Soundness.Discharge.Hidden.SpecOf
import Proofs.Flock.Soundness.Discharge.Hidden.Unpin
import Proofs.Flock.Soundness.TmplLayout

/-!
# A hidden-output session on the committed circuit

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

/-- **A committed untyped circuit passed `HmRow.check`.** -/
theorem parseCommit_check (ht : tags.typed = false) (h : HmOut.parseCommit tags cf tables = .ok c) :
    ∃ ls, HmRow.check c tags ls = .ok () := by
  obtain ⟨pre, c₀, -, P, -, he, -⟩ := parseCommit_untyped ht h
  obtain ⟨ls, hl⟩ := P.check
  exact ⟨ls, by rw [he]; exact hl⟩

/-- `parseCommit_check`'s `CheckFacts`. -/
theorem parseCommit_checkFacts (ht : tags.typed = false) (h : HmOut.parseCommit tags cf tables = .ok c) :
    CheckFacts c := by
  obtain ⟨pre, c₀, -, P, -, he, -⟩ := parseCommit_untyped ht h
  obtain ⟨ls, hl⟩ := parseCommit_check ht h
  have hty : c.typed = none := by rw [he]; exact P.typed
  exact check_facts (tags := tags) (ls := ls) hty hl

/-- A committed circuit has the tags' leaf scheme. -/
theorem parseCommit_leafScheme (h : HmOut.parseCommit tags cf tables = .ok c) : c.leafScheme = tags.merkleLeaf := by
  cases hi : tags.isTmpl
  · rw [parseCommit_flat hi] at h
    exact parse_leafScheme h
  · rw [parseCommit_tmpl hi] at h
    obtain ⟨tu, c₀, hp, -, -, rfl⟩ := TmplValue.parseTmpl_cases h
    have := Exec.parse_leafScheme_pre hp
    exact this

/-- A committed circuit's slots are `HmRow.check`'s: `slot_log` in 7..k_log. -/
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

/-! ## The session -/

/-- **A session commits `HmOut.parseCommit`'s circuit** (`Spec.parse`). -/
theorem setupHidden_parseCommit (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) :
    HmOut.parseCommit tags cf tables = .ok st.c :=
  (setupHidden_spec' h).parse

/-- **A session's `SpecOf` at `HmOut.parseCommit`.** -/
theorem setupHidden_specC (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) :
    SpecOf (HmOut.parseCommit tags cf tables) tags pf coins draw st :=
  setupHidden_specOf h

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

/-- **A hidden-output statement's layout from its flat twin's**, where it pins no output: `HmOut.delta` appends
`outPairs` to both sides of `HmRow.delta` (`hidden_delta_unpinned`), and every first column of `outPairs` is an input
bit. -/
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

/-- **`HiddenLayout` on the committed circuit**: an untyped session on `HmOut.parseCommit`'s circuit has the layout the
placement reads, its unit the flat file's or, templated, the template's logical net placed at `posMap`
(`layout_of_tmpl`). -/
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

/-- **`unpin st`'s layout from its flat twin's**, pinned outputs or not: its Δ is `HmRow.delta`'s, then `outPairs`, by
definition (`unpin`), and every first column of `outPairs` is an input bit. -/
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

/-- **`unpin` of an untyped session on the committed circuit has the layout the placement reads**, pinned outputs or
not: `layout_of`'s for a flat file, `layout_of_tmpl`'s for a templated one, with `outPairs` past its Δ. -/
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

/-- **`unpin` of an untyped statement `Stmt.setupHidden` accepts has the layout the placement reads**, pinned outputs
or not (`unpin_layoutC`). -/
theorem unpin_layout (ht : tags.typed = false)
    (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) : Layout (unpin st) :=
  unpin_layoutC ht (setupHidden_specC h)

/-! ## The facts read off `Spec` on the committed circuit -/

/-- **The leaf scheme**: under `hm96-sha512/v1` tags, a statement `Stmt.setupHidden` accepts commits its rows with
`hm96Sha512`. -/
theorem setupHidden_hm96 (hl : tags.merkleLeaf = "hm96-sha512/v1")
    (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) :
    st.merkle = MerkleScheme.hm96Sha512 :=
  specC_hm96 hl (setupHidden_specC h)

/-- **The leaf scheme, plain**: under tags with plain SHA-512 leaves, a statement `Stmt.setupHidden` accepts commits its
rows with `MerkleScheme.sha512 "flock-leaf/sha512-unsalted"`. -/
theorem setupHidden_plain (hl : tags.merkleLeaf = "flock-leaf/sha512-unsalted")
    (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) :
    st.merkle = MerkleScheme.sha512 "flock-leaf/sha512-unsalted" :=
  specC_plain hl (setupHidden_specC h)

/-- **The statement is well formed** (`StmtWF`, `RegionsWF`), as `setupH_wf` for `Stmt.setupH`. -/
theorem setupHidden_wf (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) :
    StmtWF st ∧ RegionsWF st :=
  specC_wf (setupHidden_specC h)

/-! ## The flat twin -/

/-- **What the copy positions read, of a templated circuit** whose Δ is `HmRow.delta`'s: the unit is the template's
committed net, first and named `unit`, whose rows end at its useful rows, the constant's the last (`tmpl_net_rows`). -/
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

/-- **The flat twin of an untyped session on the committed circuit has the layout the placement reads**: `layout_of`'s
for a flat file, `layout_of_tmpl`'s for a templated one. -/
theorem flat_layoutC (ht : tags.typed = false) (s : SpecOf (HmOut.parseCommit tags cf tables) tags pf coins draw st) :
    Layout (flatTwin st) := by
  cases hi : tags.isTmpl
  · have hP := s.parse
    rw [parseCommit_flat hi] at hP
    exact layout_of (st := flatTwin st) (parse_facts (parse_untyped ht hP).1) s.pin rfl rfl s.m s.kLog
  · have hT := s.parse
    rw [parseCommit_tmpl hi] at hT
    exact layout_of_tmpl (st := flatTwin st) hT s.pin rfl rfl s.m s.kLog

/-- **The flat twin's unit facts**, of an untyped session on the committed circuit (`unitFacts_of_parse`,
`unitFacts_tmpl`). -/
theorem flat_unitFactsC (ht : tags.typed = false)
    (s : SpecOf (HmOut.parseCommit tags cf tables) tags pf coins draw st) : Aliased.UnitFacts tags (flatTwin st) := by
  cases hi : tags.isTmpl
  · have hP := s.parse
    rw [parseCommit_flat hi] at hP
    exact unitFacts_of_parse (st := flatTwin st) (parse_untyped ht hP).1 rfl
  · have hT := s.parse
    rw [parseCommit_tmpl hi] at hT
    exact unitFacts_tmpl (st := flatTwin st) hT rfl

/-! ## A session from either setup -/

/-- **A committed circuit**: `HmOut.parse`'s (a flat file, or a flat session at any tags) or `HmOut.parseTmpl`'s. -/
def Committed (tags : Tags) (cf : ByteArray) (tables : List (String × ByteArray)) (c : Circuit) : Prop :=
  HmOut.parse tags cf tables = .ok c ∨ HmOut.parseTmpl tags cf tables = .ok c

theorem committed_of_commit (h : HmOut.parseCommit tags cf tables = .ok c) : Committed tags cf tables c := by
  cases hi : tags.isTmpl
  · rw [parseCommit_flat hi] at h
    exact .inl h
  · rw [parseCommit_tmpl hi] at h
    exact .inr h

/-- **A committed circuit of one output row per output port pins no output** (`unpinned_of_rep`, on the circuit
`HmRow.parse` read, after a templated file's unit net). -/
theorem Committed.unpinned_of_rep (ht : tags.typed = false) (hC : Committed tags cf tables c)
    (h : c.outRowPorts = Array.replicate c.outPorts.size 1) : HmRow.pinnedCount c.metaJ = 0 := by
  rcases hC with hC | hC
  · obtain ⟨hp, hc⟩ := parse_untyped ht hC
    exact FlockSoundness.Discharge.Hidden.unpinned_of_rep hp hc h
  · obtain ⟨tu, c₀, hp, hck, -, rfl⟩ := TmplValue.parseTmpl_cases hC
    exact FlockSoundness.Discharge.Hidden.unpinned_of_rep (c := c₀) hp hck h

/-- **An untyped hidden statement of one output row per output port pins no output** (`Committed.unpinned_of_rep`). -/
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

/-- `Stmt.setupHidden`'s accept is a `SpecC`. -/
theorem setupHidden_specCC (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) :
    SpecC tags cf pf coins tables draw st :=
  .inl (setupHidden_specOf h)

/-- `SpecOf` at the committed circuit itself. -/
theorem SpecC.self (s : SpecC tags cf pf coins tables draw st) : SpecOf (.ok st.c) tags pf coins draw st :=
  Or.elim s (fun s => { s with parse := rfl }) (fun s => { s with parse := rfl })

theorem SpecC.committed (s : SpecC tags cf pf coins tables draw st) : Committed tags cf tables st.c :=
  Or.elim s (fun s => committed_of_commit s.parse) (fun s => .inl s.parse)

/-- **The flat twin's layout**, of an untyped session from either setup. -/
theorem SpecC.flatLayout (ht : tags.typed = false) (s : SpecC tags cf pf coins tables draw st) :
    Layout (flatTwin st) := by
  have s0 := s.self
  rcases s.committed with hP | hT
  · exact layout_of (st := flatTwin st) (parse_facts (parse_untyped ht hP).1) s0.pin rfl rfl s0.m s0.kLog
  · exact layout_of_tmpl (st := flatTwin st) hT s0.pin rfl rfl s0.m s0.kLog

/-- **The flat twin's unit facts**, of an untyped session from either setup. -/
theorem SpecC.unitFacts (ht : tags.typed = false) (s : SpecC tags cf pf coins tables draw st) :
    Aliased.UnitFacts tags (flatTwin st) := by
  rcases s.committed with hP | hT
  · exact unitFacts_of_parse (st := flatTwin st) (parse_untyped ht hP).1 rfl
  · exact unitFacts_tmpl (st := flatTwin st) hT rfl

/-- **The statement circuit's unit is the committed circuit's logical net**, with no template, at the committed
circuit's units per VU. -/
theorem Committed.unrolled_unit (hty : tags.typed = false) (h : Committed tags cf tables c) :
    (HmOut.unrolled c).unit = lunit c ∧ (HmOut.unrolled c).tmpl = none ∧ (HmOut.unrolled c).upv = c.upv := by
  rcases h with h | h
  · have hf := parse_tmpl_none hty h
    rw [unrolled_flat hf, lunit_flat hf]
    exact ⟨rfl, hf, rfl⟩
  · obtain ⟨_tu, _n, -, -, -, -, -, -, he⟩ := unrolled_of_parseTmpl h
    exact ⟨FlockSoundness.unrolled_unit h, unrolled_tmpl h, by rw [he]⟩

/-- **The statement circuit is the committed one with other nets and no template**: every other field (ports, leaves,
ranges, `units_per_vu`, the net indices) is the committed circuit's. -/
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

/-- **A committed circuit is untyped and passes `HmRow.check` and `HmOut.checkRows`.** -/
theorem Committed.check (hty : tags.typed = false) (h : Committed tags cf tables c) :
    c.typed = none ∧ (∃ ls, HmRow.check c tags ls = .ok ()) ∧ HmOut.checkRows c = .ok () := by
  rcases h with h | h
  · obtain ⟨hp, hc⟩ := parse_untyped hty h
    have P := parse_facts hp
    exact ⟨P.typed, P.check, hc⟩
  · obtain ⟨tu, c₀, P, hck, he, -, -⟩ := TmplValue.parseTmpl_facts h
    obtain ⟨ls, hcheck⟩ := P.check
    exact ⟨by rw [he]; exact P.typed, ⟨ls, by rw [he]; exact hcheck⟩, by rw [he]; exact hck⟩

/-- **A committed circuit's nets store `A` and `B` at their slots' widths**, flat (`Integrate.net_width`) or templated
(`parseTmpl_net_width`). -/
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

/-- **What a committed circuit's parse accepted, whatever its template** (`ParseFactsU`): `HmRow.parse`'s facts for a
flat file, its parsed file's with the template's committed unit net first for a templated one. -/
theorem Committed.factsU (hty : tags.typed = false) (h : Committed tags cf tables c) :
    ∃ pre, ParseFactsU tags pre c := by
  rcases h with h | h
  · exact ⟨#[], (parse_facts (parse_untyped hty h).1).toU⟩
  · obtain ⟨tu, c₀, P, -, rfl, -, -⟩ := TmplValue.parseTmpl_facts h
    exact ⟨_, P.toU.retmpl _⟩

/-- **A committed circuit's `sha512/row/v1` output rows** (`out_rows_v1`): unsegmented, one per output port, each after
the input rows and of `2 · words = 128 · chunks` bytes; a templated file's are its parsed file's. -/
theorem Committed.outRows (hty : tags.typed = false) (h : Committed tags cf tables c)
    (hv1 : ∀ p ∈ c.ports.toList, p.v2 = false) : OutRows c := by
  rcases h with h | h
  · obtain ⟨hp, hc⟩ := parse_untyped hty h
    exact out_rows_v1 hp hc hv1
  · obtain ⟨tu, c₀, hp, hck, -, rfl⟩ := TmplValue.parseTmpl_cases h
    have h := out_rows_v1Pre hp (parse_factsPre_nets hp).1 hck hv1
    exact h

/-- **A session's layout**, flat or templated, where it pins no output (`hidden_layoutC`). -/
theorem SpecC.layout (ht : tags.typed = false) (s : SpecC tags cf pf coins tables draw st)
    (hpin : HmRow.pinnedCount st.c.metaJ = 0) : Layout st :=
  Or.elim s (fun s => hidden_layoutC ht s hpin) fun s => by
    obtain ⟨hp, hc⟩ := parse_untyped ht s.parse
    have P := parse_facts hp
    exact hidden_layout_of (layout_of P s.pin rfl rfl s.m s.kLog) P.typed s.da s.db (outPairs_inBit hp hc) hpin

/-- **A session's statement without Δ's pinned copies has a layout**, flat or templated, pinned outputs or not
(`unpin_layoutC`). -/
theorem SpecC.unpinLayout (ht : tags.typed = false) (s : SpecC tags cf pf coins tables draw st) : Layout (unpin st) :=
  Or.elim s (unpin_layoutC ht) fun s => by
    obtain ⟨hp, hc⟩ := parse_untyped ht s.parse
    have P := parse_facts hp
    exact unpin_layout_of (layout_of P s.pin rfl rfl s.m s.kLog) P.typed (outPairs_inBit hp hc)

/-- **A session's layout model is `stmtOf` of its statement**, flat or templated (`Integrate.model_eqW`). -/
theorem SpecC.model_eq (ht : tags.typed = false) (s : SpecC tags cf pf coins tables draw st) (L : Layout st)
    (hwf : StmtWF st) (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))) : L.model regs = stmtOf st hwf regs :=
  Integrate.model_eqW (Committed.netWidths ht s.committed) (Committed.netKinds ht s.committed) L s.self.leaf hwf regs

/-- **A session's layout model without Δ's pinned copies is `stmtOf` of `unpin st`**, flat or templated
(`Integrate.model_eqW`). -/
theorem SpecC.model_eq_unpin (ht : tags.typed = false) (s : SpecC tags cf pf coins tables draw st)
    (L : Layout (unpin st)) (hwf : StmtWF (unpin st)) (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))) :
    L.model regs = stmtOf (unpin st) hwf regs :=
  Integrate.model_eqW (st := unpin st) (Committed.netWidths ht s.committed) (Committed.netKinds ht s.committed) L
    s.self.leaf hwf regs

/-- **A session's statement circuit is `HmOut.parseStmt`'s**: its committed circuit, unrolled. -/
theorem SpecC.parseStmt_eq (ht : tags.typed = false) (s : SpecC tags cf pf coins tables draw st) :
    HmOut.parseStmt tags cf tables = .ok (HmOut.unrolled st.c) :=
  Or.elim s (fun s => parseStmt_of_commit ht s.parse) fun s => by
    rw [unrolled_flat (parse_tmpl_none ht s.parse)]
    exact parseStmt_of_parse (HmOut.parse_isTmpl s.parse) s.parse

/-- **A flat session's circuit is `HmOut.parse`'s.** -/
theorem SpecC.parse_flat (hi : tags.isTmpl = false) (s : SpecC tags cf pf coins tables draw st) :
    HmOut.parse tags cf tables = .ok st.c :=
  Or.elim s (fun s => by rw [← parseCommit_flat hi]; exact s.parse) (fun s => s.parse)

end FlockSoundness.Discharge.Hidden
