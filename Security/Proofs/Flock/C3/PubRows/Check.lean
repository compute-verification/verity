import Proofs.Flock.C3.PubRows.Prefix
import Proofs.Flock.C3.Compose.Honest
import Proofs.Flock.VBridge.Family.Pub
import Proofs.Flock.C3.Compose.RowsReg

/-!
# n1: what the verifier's check of the public rows gives

`RecVStar.checkPubRows` compares each table row of each registered port whose value `fixedRow` knows with
`fixedRow`'s row at the position the port's entry gives it (`checkPubRows_rows`). At an accepted session, a drawn unit
that the drawn statement labels itself reads its row of the port at one of those positions (`Registered.check`'s
`checkPort`, whose positions are one per table row: `checkPort_size`), so the public file's row there is the row
`fixedRow` holds (`pubRow_of_checked`, the end of `opens_of_acceptsZKR` at the check rather than at the root).

`HPubRows` reads every unit and every port of a name, at the verifier's own RecSparse shape; the check reads the drawn
units, the first port of a name, and a rec-sparse unit's own shape. What closes the gap is the family's:

* `RowsHeld`: a session accepting a transcript it writes draws every unit that reads a value `fixedRow` knows (G's
  `RowsDrawn` at every such value, not only `rec-c<k>`), whose label is the unit by the verifier's label check
  (`label_of_checked`); every family holds it (`rowsHeld`: the record's custody and the full draw);
* `PortsNamed`: each circuit names each port once (G's `PortNamesUnique`);
* `SparseAtShape`: a rec-sparse unit's shape is the verifier's own.
-/

namespace FlockC3.PubRows

open FlockSoundness FlockSoundness.Game FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open FlockSoundness.Discharge.Recursion FlockSoundness.Refine FlockSoundness.Model FlockSoundness.Audit
open Flock FlockLevel3 FlockVBridge.Compiled FlockVBridge.InnerAccepts FlockVBridge.Stages
open FlockSoundness.Discharge.ZkReg FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.ZkLink
open FlockSoundness.Discharge.ZkBind FlockSoundness.Discharge.Integrate FlockSoundness.Discharge.DecodeZL
open FlockSoundness.Discharge.Composed (drawJ partOf)
open FlockC3.Compose

/-! ## The executable's check -/

theorem rows_forIn_each {α ε : Type} (P : α → Prop) {f : α → Unit → Except ε (ForInStep Unit)}
    (hf : ∀ a r, f a () = .ok r → P a ∧ r = .yield ()) :
    ∀ (l : List α) {u : Unit}, forIn l () f = .ok u → ∀ a ∈ l, P a
  | [], _, _, a, ha => by cases ha
  | b :: l, u, h, a, ha => by
    rw [List.forIn_cons] at h
    obtain ⟨st, hst, h⟩ := Registered.except_bind_ok h
    obtain ⟨hb, rfl⟩ := hf b st hst
    cases ha with
    | head => exact hb
    | tail _ ha => exact rows_forIn_each P hf l h a ha

theorem rows_range_size (n : ℕ) : [:n].size = n := by simp [Std.Legacy.Range.size]

/-- Whether `fixedRow` knows a value is the value's prefix alone. -/
theorem fixedRow_isSome (cs cs' : RecVStar.InnerCommits) (v : String) (x x' : ℕ)
    (sp sp' : Option RecSparse.Shape) :
    (RecVStar.fixedRow cs v x sp).isSome = (RecVStar.fixedRow cs' v x' sp').isSome := by
  unfold RecVStar.fixedRow
  split <;> (try split) <;> simp_all

/-- A row `fixedRow` holds with no RecSparse shape is held at every shape: only the sparse values read it. -/
theorem fixedRow_sparse {cs : RecVStar.InnerCommits} {v : String} {x : ℕ} {a : String × ByteArray}
    (h : RecVStar.fixedRow cs v x none = some (.ok a)) (sp : Option RecSparse.Shape) :
    RecVStar.fixedRow cs v x sp = some (.ok a) := by
  unfold RecVStar.fixedRow at h ⊢
  split <;> split at h <;> simp_all

/-- A row `fixedRow` holds at the shape `sp₀` is held at `sp.or sp₀` where `sp` is none or `sp₀`'s. -/
theorem fixedRow_or {cs : RecVStar.InnerCommits} {v : String} {x : ℕ} {sp sp₀ : Option RecSparse.Shape}
    {a : String × ByteArray} (hsp : ∀ sh sh₀, sp = some sh → sp₀ = some sh₀ → sh = sh₀)
    (h : RecVStar.fixedRow cs v x sp₀ = some (.ok a)) : RecVStar.fixedRow cs v x (sp.or sp₀) = some (.ok a) := by
  cases sp with
  | none => exact h
  | some sh =>
    cases sp₀ with
    | none => exact fixedRow_sparse h _
    | some sh₀ => rw [hsp sh sh₀ rfl rfl] at *; exact h

/-- **`checkPort` accepting has one position per table row.** -/
theorem checkPort_size {c : Flock.Circuit} {pub : Public} {t : Flock.Tables} {indices : Array ℕ}
    {reg : Registered.Port} (h : Registered.checkPort c pub t indices reg = .ok ()) :
    ∃ p positions paths, (List.range c.ports.size).find? (fun p => c.ports[p]!.name == reg.name) = some p ∧
      Registered.entry pub reg = .ok (positions, paths) ∧ positions.size = t.counts.getD p 0 := by
  unfold Registered.checkPort at h
  simp only at h
  split at h
  · rename_i p hp
    split at h
    · exact absurd (Registered.except_bind_ok h) fun ⟨_, h1, _⟩ => Registered.throw_ne_ok _ _ h1
    · split at h
      · exact absurd (Registered.except_bind_ok h) fun ⟨_, h1, _⟩ => Registered.throw_ne_ok _ _ h1
      · obtain ⟨⟨positions, paths⟩, hent, h⟩ := Registered.except_bind_ok h
        simp only at h
        split at h
        · exact absurd (Registered.except_bind_ok h) fun ⟨_, h1, _⟩ => Registered.throw_ne_ok _ _ h1
        · rename_i hsz
          refine ⟨p, positions, paths, hp, hent, ?_⟩
          simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, not_or, Classical.not_not] at hsz
          exact hsz.1
  · exact absurd h (Registered.throw_ne_ok _ _)

/-- **`checkPubRows` accepting**: at each port of the reads whose value `fixedRow` knows, the public file is a table,
the port is the circuit's first of its name, and each table row is `fixedRow`'s row at the position the port's entry
gives it, at the unit's RecSparse shape or else `sparse`. -/
theorem checkPubRows_rows {st : Flock.Stmt} {own : Array Registered.Port} {cs : RecVStar.InnerCommits}
    {sparse : Option RecSparse.Shape} (h : RecVStar.checkPubRows st (some own) (some cs) sparse = .ok ())
    {reg : Registered.Port} (hmem : reg ∈ own)
    (hfix : (RecVStar.fixedRow { reps := #[] } reg.value 0).isSome = true) :
    ∃ t, st.pub.tables = some t ∧ ∃ p positions paths,
      (List.range st.c.ports.size).find? (fun p => st.c.ports[p]!.name == reg.name) = some p ∧
      Registered.entry st.pub reg = .ok (positions, paths) ∧
      ∀ r < positions.size, ∃ w c, RecVStar.fixedRow cs reg.value (positions.getD r 0)
        ((RecSparse.params st.c.unit.name).or sparse) = some (.ok (w, c)) ∧ Registered.tableRow st.pub t p r = c := by
  have hreg : reg ∈ own.filter (fun reg => (RecVStar.fixedRow { reps := #[] } reg.value 0).isSome) :=
    Array.mem_filter.2 ⟨hmem, hfix⟩
  unfold RecVStar.checkPubRows at h
  simp only [Option.getD_some] at h
  obtain ⟨_, -, h⟩ := Registered.except_bind_ok h
  split at h
  · obtain ⟨_, -, h⟩ := Registered.except_bind_ok h
    obtain ⟨_, -, h⟩ := Registered.except_bind_ok h
    split at h
    · obtain ⟨_, h1, -⟩ := Registered.except_bind_ok h
      exact absurd h1 (Registered.throw_ne_ok _ _)
    split at h
    · rename_i t ht
      obtain ⟨_, hloop, -⟩ := Registered.except_bind_ok h
      rw [← Array.forIn_toList] at hloop
      refine ⟨t, ht, rows_forIn_each (fun a : Registered.Port => ∃ p positions paths,
          (List.range st.c.ports.size).find? (fun p => st.c.ports[p]!.name == a.name) = some p ∧
          Registered.entry st.pub a = .ok (positions, paths) ∧
          ∀ r < positions.size, ∃ w c, RecVStar.fixedRow cs a.value (positions.getD r 0)
            ((RecSparse.params st.c.unit.name).or sparse) = some (.ok (w, c)) ∧ Registered.tableRow st.pub t p r = c)
        (fun a r hb => ?_) _ hloop reg (Array.mem_toList_iff.2 hreg)⟩
      split at hb
      · rename_i p hp
        obtain ⟨⟨positions, paths⟩, hent, hb⟩ := Registered.except_bind_ok hb
        obtain ⟨_, hrows, hb⟩ := Registered.except_bind_ok hb
        refine ⟨⟨p, positions, paths, hp, hent, fun r hr => ?_⟩, (Except.ok.inj hb).symm⟩
        rw [Std.Legacy.Range.forIn_eq_forIn_range', rows_range_size] at hrows
        refine rows_forIn_each (fun i => ∃ w c, RecVStar.fixedRow cs a.value (positions.getD i 0)
            ((RecSparse.params st.c.unit.name).or sparse) = some (.ok (w, c)) ∧
            Registered.tableRow st.pub t p i = c) (fun i s hi => ?_) _ hrows r
          (List.mem_range'_1.2 ⟨Nat.zero_le _, by simpa using hr⟩)
        have hget : positions[i]! = positions.getD i 0 := by
          simp only [getElem!_def, Array.getD_eq_getD_getElem?]
          cases positions[i]? <;> rfl
        split at hi
        · rename_i w c hc
          split at hi
          · obtain ⟨_, h1, -⟩ := Registered.except_bind_ok hi
            exact absurd h1 (Registered.throw_ne_ok _ _)
          · rename_i hne
            have he : (Registered.tableRow st.pub t p i == c) = true := by simpa [bne] using hne
            exact ⟨⟨w, c, by rw [← hget]; exact hc, ByteArray.ext (beq_iff_eq.1 he)⟩, (Except.ok.inj hi).symm⟩
        · obtain ⟨_, h1, -⟩ := Registered.except_bind_ok hi
          exact absurd h1 (Registered.throw_ne_ok _ _)
        · obtain ⟨_, h1, -⟩ := Registered.except_bind_ok hi
          exact absurd h1 (Registered.throw_ne_ok _ _)
      · obtain ⟨_, h1, -⟩ := Registered.except_bind_ok hb
        exact absurd h1 (Registered.throw_ne_ok _ _)
    · exact absurd h (Registered.throw_ne_ok _ _)
  · rename_i hnone
    exact absurd (Array.getElem?_eq_getElem (Array.size_pos_of_mem hreg)) (hnone _)

theorem bytes_toList_loop (b : ByteArray) (i : ℕ) (r : List UInt8) :
    ByteArray.toList.loop b i r = r.reverse ++ b.data.toList.drop i := by
  fun_induction ByteArray.toList.loop b i r with
  | case1 i r h ih =>
    rw [ih, List.reverse_cons, List.append_assoc, List.singleton_append]
    congr 1
    conv_rhs => rw [List.drop_eq_getElem_cons (by simpa using h)]
    simp [ByteArray.get!, h]
  | case2 i r h =>
    rw [List.drop_eq_nil_of_le (by simp; simpa using h), List.append_nil]

theorem bytes_toList_eq (b : ByteArray) : b.toList = b.data.toList := by
  unfold ByteArray.toList
  rw [bytes_toList_loop]
  simp

/-- **A row `heldRow` holds** is `fixedRow`'s at the verifier's hold of the inner transcript. -/
theorem heldRow_some {cs : Except String RecVStar.InnerCommits} {sparse : Option RecSparse.Shape} {v : String}
    {x : ℕ} {b : List UInt8} (h : heldRow cs sparse v x = some b) :
    ∃ cs' w c, cs = .ok cs' ∧ RecVStar.fixedRow cs' v x sparse = some (.ok (w, c)) ∧
      (⟨b.toArray⟩ : ByteArray) = c := by
  unfold heldRow at h
  split at h
  · rename_i cs'
    split at h
    · rename_i w c hc
      cases h
      refine ⟨cs', w, c, rfl, hc, ?_⟩
      rw [bytes_toList_eq]
      try simp
    · cases h
  · cases h

theorem port_find_of_unique {c : Flock.Circuit}
    (hu : ∀ q q' : Fin c.ports.size, c.ports[q].name = c.ports[q'].name → q = q') (q : Fin c.ports.size) :
    (List.range c.ports.size).find? (fun p => c.ports[p]!.name == c.ports[q].name) = some q.val := by
  rw [List.find?_eq_some_iff_getElem]
  refine ⟨by rw [getElem!_pos c.ports q.val q.isLt]; simp, q.val, by simp, by simp, fun k hk => ?_⟩
  have hk' : k < c.ports.size := by omega
  simp only [List.getElem_range]
  rw [getElem!_pos c.ports k hk']
  have hne : c.ports[k].name ≠ c.ports[q].name := fun he =>
    Nat.ne_of_lt hk (congrArg Fin.val (hu ⟨k, hk'⟩ q he))
  simpa using hne

/-! ## At an accepted session -/

/-- **An accepted session's checked public row**: where the session accepts at `tr` and the verifier's check of its
public rows passes at `tr`'s setup, a drawn unit `u` that its table's drawn statement labels `idx` reads row `x` of a
port `reg` of `own` whose value `fixedRow` knows (the circuit's port `q`), and the public file's `b ‖ c` of `u`'s row
`q` is `fixedRow`'s row `x` of the value, at the unit's RecSparse shape or else `sparse`. -/
theorem pubRow_of_checked {mPts nv J : ℕ} {Reg' : Type} {c : Flock.Circuit} {hU : Built c.unit}
    {hscope : Layout.scopeOk c = true} {Lv : Law nv} (zo : ZkOuter mPts nv J Reg' c hU hscope Lv)
    {tr : Exec.Transcript} (hacc : AcceptsZKR zo.I zo.own tr) {cs : RecVStar.InnerCommits}
    {sparse : Option RecSparse.Shape}
    (hchk : ∃ spec sts, setupOf (zkInputs zo.I) tr.record = .ok (spec, sts) ∧
      sts.forM (fun s => RecVStar.checkPubRows s (some zo.own) (some cs) sparse) = .ok ())
    {reg : Registered.Port} (hmem : reg ∈ zo.own)
    (hfix : (RecVStar.fixedRow { reps := #[] } reg.value 0).isSome = true) {u : Fin nv}
    {q : Fin c.ports.size}
    (hfind : (List.range c.ports.size).find? (fun p => c.ports[p]!.name == reg.name) = some q.val)
    {d : Lean.Json} {D : Draw.UnitDraw} {dp : Public} {idx : Array ℕ} (hd : recordDraw tr.record = some d)
    (hD : Draw.UnitDraw.ofJson (drawJ zo.I d (partOf zo.I d u.val)) = .ok D)
    (hdp : HmOut.drawn c zo.pub (drawJ zo.I d (partOf zo.I d u.val)) = .ok dp)
    (hidx : Registered.unitIndices dp = .ok idx) (hu : u.val ∈ D.proved) {x : ℕ}
    (hread : reg.reads[idx.getD (D.proved.idxOf u.val) 0]? = some (some x)) :
    ∃ w b, RecVStar.fixedRow cs reg.value x ((RecSparse.params c.unit.name).or sparse) = some (.ok (w, b)) ∧
      zo.pub.digest u.val q.val c.ports.size = b := by
  obtain ⟨spec, sts, hs, hreg, -, -⟩ := hacc
  obtain ⟨spec', sts', hs', hpr⟩ := hchk
  have hst : sts' = sts := congrArg Prod.snd (Except.ok.inj (hs'.symm.trans hs))
  rw [hst] at hpr
  obtain ⟨hsz, hstJ⟩ := ZkHidden.setupOf_tableJH zo.hTags.1 hd hs
  have hjs : (partOf zo.I d u.val).val < sts.size := by rw [hsz]; exact (partOf zo.I d u.val).isLt
  have s := Hidden.setupHidden_spec' (hstJ _ hjs)
  have hcst : sts[(partOf zo.I d u.val).val].c = c := by
    have h1 := s.parseF ((FlockSoundness.Discharge.Exec.tagsAt_isTmpl _ _ _).trans (HmOut.parse_isTmpl zo.hcH))
    rw [ZkHidden.parseH_tagsAt ((zkTags_typed zo.I).trans zo.ht), zo.hcH] at h1
    exact (Except.ok.inj h1).symm
  obtain ⟨pop', hpop', -, hdr⟩ := s.pub
  rw [hcst, ZkHidden.loadPublicH_tagsAt, zo.hpub] at hpop'
  cases hpop'
  have hdr' := hdr _ rfl
  rw [hcst, hdp] at hdr'
  have hpd : sts[(partOf zo.I d u.val).val].pub = dp := (Except.ok.inj hdr').symm
  have hdpR : HmRow.drawn { c with outPorts := #[] } zo.pub (drawJ zo.I d (partOf zo.I d u.val)) = .ok dp := hdp
  have hck := checkRegistered_ok (array_forM_ok hreg _ hjs)
  rw [hcst, hpd] at hck
  obtain ⟨tb, idx', htb, hidx', hports⟩ := check_tables hck
  rw [hidx] at hidx'
  cases hidx'
  obtain ⟨p', positions, paths, hp', hent, hall⟩ := checkPort_at (hports _ hmem)
  obtain ⟨p'', positions', paths', hp'', hent', hsize⟩ := checkPort_size (hports _ hmem)
  rw [hfind] at hp' hp''
  cases hp'
  cases hp''
  have hpos : positions' = positions := congrArg Prod.fst (Except.ok.inj (hent'.symm.trans hent))
  rw [hpos] at hsize
  obtain ⟨D', hD', -, hn, -, -⟩ := Layout.drawn_spec hdpR
  rw [hD] at hD'
  cases hD'
  have hi : D.proved.idxOf u.val < D.proved.size := Array.idxOf_lt_length_of_mem hu
  obtain ⟨hr, hreads, -⟩ := hall _ (by rw [hn]; exact hi)
  rw [hread] at hreads
  have hx : x = positions.getD (Registered.ref tb c.ports.size q.val (D.proved.idxOf u.val)) 0 := by
    simpa using hreads
  have hget : D.proved[D.proved.idxOf u.val]'hi = u.val := by
    have := Array.findIdx_getElem (xs := D.proved) (p := (· == u.val)) (w := hi)
    simp only [beq_iff_eq] at this
    exact this
  have hdig : zo.pub.digest u.val q.val c.ports.size =
      Registered.tableRow dp tb q.val (Registered.ref tb c.ports.size q.val (D.proved.idxOf u.val)) := by
    have h1 := Hidden.drawn_digestW (Hidden.loadPublic_nil zo.hpub) hdpR hD (Layout.proved_lt_population hD) hi q.isLt
    rw [hget] at h1
    rw [← h1]
    simp only [Public.digest, htb, Registered.tableRow, Registered.ref]
  obtain ⟨t', ht', p, positions'', paths'', hp, hent'', hrows⟩ :=
    checkPubRows_rows (array_forM_ok hpr _ hjs) hmem hfix
  rw [hcst] at hp hrows
  rw [hpd] at ht' hent'' hrows
  rw [htb] at ht'
  cases ht'
  rw [hfind] at hp
  cases hp
  have hpos' : positions'' = positions := congrArg Prod.fst (Except.ok.inj (hent''.symm.trans hent))
  rw [hpos'] at hrows
  obtain ⟨w, b, hb, hrow⟩ := hrows (Registered.ref tb c.ports.size q.val (D.proved.idxOf u.val))
    (by rw [hsize]; exact hr)
  rw [← hx] at hb
  exact ⟨w, b, hb, hdig.trans hrow⟩

theorem getD_idxOf {a : Array ℕ} {u : ℕ} (h : u ∈ a) : a.getD (a.idxOf u) 0 = u := by
  rcases a with ⟨l⟩
  have hl : u ∈ l := by simpa using h
  have hlt := List.idxOf_lt_length_of_mem hl
  simp [Array.getD, hlt, List.getElem_idxOf hlt]

theorem checkLabel_ok {dj : Lean.Json} {pub : Public} {j : ℕ} (h : RecVStar.checkLabel dj pub j = .ok ())
    {D : Draw.UnitDraw} (hD : Draw.UnitDraw.ofJson dj = .ok D) {idx : Array ℕ}
    (hidx : Registered.unitIndices pub = .ok idx) : idx = D.proved := by
  simp only [RecVStar.checkLabel, hD, hidx, Bind.bind, Except.bind] at h
  split at h
  · simp [throw, throwThe, MonadExceptOf.throw] at h
  · rename_i hb
    simpa using hb

theorem checkLabels_at {d : Lean.Json} {count : ℕ} {sts : Array Stmt}
    (h : RecVStar.checkLabels (some d) count sts = .ok ()) (j : ℕ) (hj : j < sts.size) :
    RecVStar.checkLabel (RecVStar.tableDraw d count j) sts[j].pub j = .ok () := by
  have := array_forM_ok (show (Array.range sts.size).forM
    (fun j => RecVStar.checkLabel (RecVStar.tableDraw d count j) sts[j]!.pub j) = .ok () from h) j (by simpa using hj)
  simpa [Array.getElem_range, getElem!_pos sts j hj] using this

/-- The verifier's table draw is the one the model's tables are set up from. -/
theorem drawJ_tableDraw (I : Inputs) (d : Lean.Json) (j : ℕ) : drawJ I d j = RecVStar.tableDraw d I.count j := by
  unfold drawJ RecVStar.tableDraw DecodeZLJ.partJson Tables.partDraw
  rfl

/-- **The label conjunct, from the verifier's label check** (`RecVStar.checkLabels`): at an accepted transcript whose
tables label their instances with the units their draws prove, the drawn statement's label of a drawn unit is the
unit. -/
theorem label_of_checked {mPts nv J : ℕ} {Reg' : Type} {c : Flock.Circuit} {hU : Built c.unit}
    {hscope : Layout.scopeOk c = true} {Lv : Law nv} (zo : ZkOuter mPts nv J Reg' c hU hscope Lv)
    {tr : Exec.Transcript} (hacc : AcceptsZKR zo.I zo.own tr)
    (hlb : ∃ spec sts, setupOf (zkInputs zo.I) tr.record = .ok (spec, sts) ∧
      RecVStar.checkLabels (recordDraw tr.record) zo.I.count sts = .ok ())
    {u : Fin nv} {d : Lean.Json} {D : Draw.UnitDraw} {dp : Public} {idx : Array ℕ}
    (hd : recordDraw tr.record = some d)
    (hD : Draw.UnitDraw.ofJson (drawJ zo.I d (partOf zo.I d u.val)) = .ok D)
    (hdp : HmOut.drawn c zo.pub (drawJ zo.I d (partOf zo.I d u.val)) = .ok dp)
    (hidx : Registered.unitIndices dp = .ok idx) (hu : u.val ∈ D.proved) :
    idx.getD (D.proved.idxOf u.val) 0 = u.val := by
  obtain ⟨spec, sts, hs, -⟩ := hacc
  obtain ⟨spec', sts', hs', hlb⟩ := hlb
  have hst : sts' = sts := congrArg Prod.snd (Except.ok.inj (hs'.symm.trans hs))
  rw [hst, hd] at hlb
  obtain ⟨hsz, hstJ⟩ := ZkHidden.setupOf_tableJH zo.hTags.1 hd hs
  have hjs : (partOf zo.I d u.val).val < sts.size := by rw [hsz]; exact (partOf zo.I d u.val).isLt
  have s := Hidden.setupHidden_spec' (hstJ _ hjs)
  have hcst : sts[(partOf zo.I d u.val).val].c = c := by
    have h1 := s.parseF ((FlockSoundness.Discharge.Exec.tagsAt_isTmpl _ _ _).trans (HmOut.parse_isTmpl zo.hcH))
    rw [ZkHidden.parseH_tagsAt ((zkTags_typed zo.I).trans zo.ht), zo.hcH] at h1
    exact (Except.ok.inj h1).symm
  obtain ⟨pop', hpop', -, hdr⟩ := s.pub
  rw [hcst, ZkHidden.loadPublicH_tagsAt, zo.hpub] at hpop'
  cases hpop'
  have hdr' := hdr _ rfl
  rw [hcst, hdp] at hdr'
  have hpd : sts[(partOf zo.I d u.val).val].pub = dp := (Except.ok.inj hdr').symm
  have hl := checkLabels_at hlb _ hjs
  rw [← drawJ_tableDraw, hpd] at hl
  rw [checkLabel_ok hl hD hidx]
  exact getD_idxOf hu

/-! ## At the family -/

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {xR : HoloReg GF128 Dig} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

/-- **(n1) A session accepting a transcript it writes draws the units that read a value `fixedRow` knows**: G's
`RowsDrawn` at every such value, at the outcomes the session's game reaches (`ZkOuter.Writes`). `AcceptsZKR` checks the
drawn units only (U4, `checkFull`, is not among its checks), so a unit it does not draw has a public row no check reads;
a transcript it writes draws every unit (`rowsHeld`). The drawn statement's label of the unit
(`Registered.unitIndices`) is the unit by the verifier's label check (`label_of_checked`), not by this. -/
def RowsHeld (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂) : Prop :=
  ∀ R ω t R₂ j R' tr, (F.zs R ω t R₂ j).Writes R' tr → AcceptsZKR (F.zs R ω t R₂ j).I (F.zs R ω t R₂ j).own tr →
    ∀ reg ∈ (F.zs R ω t R₂ j).own,
    (RecVStar.fixedRow { reps := #[] } reg.value 0).isSome = true →
    ∀ (u : Fin (F.Vs j).nv) x, reg.reads[u.val]? = some (some x) →
      ∃ d D dp idx, recordDraw tr.record = some d ∧
        Draw.UnitDraw.ofJson (drawJ (F.zs R ω t R₂ j).I d (partOf (F.zs R ω t R₂ j).I d u.val)) = .ok D ∧
        HmOut.drawn (F.Vs j).c (F.zs R ω t R₂ j).pub
          (drawJ (F.zs R ω t R₂ j).I d (partOf (F.zs R ω t R₂ j).I d u.val)) = .ok dp ∧
        Registered.unitIndices dp = .ok idx ∧ u.val ∈ D.proved

/-- **(n1) Every family holds its rows**: a transcript a session writes and accepts records the law's draw file (the
record's custody, `hRec`), the table each unit is drawn into has a drawn statement (`drawn_of_acceptsZKR`), and that
table's draw proves the unit (the coin server's draw, `hDraw`), since V*'s law, the full draw, draws every unit
(`execOS_subset_full_mem`). -/
theorem rowsHeld (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂) : RowsHeld F := by
  rintro R ω t R₂ j R' tr ⟨ω', o, rfl⟩ hacc _ _ _ u _ _
  have hd := ((F.zs R ω t R₂ j).hRec _ _ _ (acceptsZK_of_R hacc)).1
  obtain ⟨D, dp, idx, hD, hdp, hidx⟩ := FlockSoundness.Discharge.ZkReg.drawn_of_acceptsZKR (F.zs R ω t R₂ j).I
    (F.zs R ω t R₂ j).ht (F.zs R ω t R₂ j).hcH (F.zs R ω t R₂ j).hpub (F.zs R ω t R₂ j).hTags hacc hd u.val
  exact ⟨_, D, dp, idx, hd, hD, hdp, hidx, ((F.zs R ω t R₂ j).hDraw ω' (partOf (F.zs R ω t R₂ j).I _ u.val)).1 D hD u
    (FlockSoundness.Discharge.Composed.mem_partS (execOS_subset_full_mem (F.Vs j).law.src ω' u))⟩

/-- **(n1) Each of V*'s circuits names each port once** (G's `PortNamesUnique`): the check reads the first port of a
name. -/
def PortsNamed (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂) : Prop :=
  ∀ j (q q' : Fin (F.Vs j).c.ports.size), (F.Vs j).c.ports[q].name = (F.Vs j).c.ports[q'].name → q = q'

/-- **(n1) A rec-sparse unit of V*'s is at the verifier's RecSparse shape**: the check reads its rows at the unit's own
shape, `F.row` at the verifier's. -/
def SparseAtShape (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂) : Prop :=
  ∀ j sh sh₀, RecSparse.params (F.Vs j).c.unit.name = some sh → F.ish.sparse = some sh₀ → sh = sh₀

end FlockC3.PubRows
