import FlockSoundness.Discharge.Hidden.Layout

/-!
# A hidden-output statement's flat twin

`Stmt.setupHidden`'s Δ is `HmRow.delta`'s, then `Hidden.outPairs` (`Hidden.HiddenDelta`): each output row's message
bits, copied from the output bits the circuit computes. `flat st` is the statement with `HmRow.delta`'s Δ alone, the one
`Stmt.setupH` would have built on the same circuit. It has `layout_of`'s layout (`flat_layout`) and the unit facts the
copy positions read (`flat_unitFacts`), as setupH's statement does.

Every first column of `outPairs` is a message bit of a row's `sha512x3` slot (`deltaMsgCol`), which no unit slot's row
is (`outPairs_ne_unit`, for `sha512/row/v1` rows). So at every unit slot's row the hidden statement's level-0 matrices
are its flat twin's (`unitRowsFlat`): the rows that place a VU and copy its inputs read the same in both.
-/

set_option Elab.async false

namespace FlockSoundness.Discharge.HiddenExec

open Flock FlockLevel3 Refine
open FlockSoundness.Discharge.Exec
open Lean (Json)

/-- **The flat twin**: the statement with `HmRow.delta`'s Δ for its own. -/
abbrev flat (st : Stmt) : Stmt :=
  { st with da := (HmRow.delta st.c st.pin).1, db := (HmRow.delta st.c st.pin).2 }

/-- **The hidden statement reads its flat twin's rows at every unit slot's row**, in `A` and in `B`. -/
def UnitRowsFlat (st : Stmt) : Prop :=
  ∀ s₀ < st.c.unitRange.count, ∀ b < 2 ^ st.c.unit.unitLog, ∀ r j : Fin (2 ^ st.c.kLog),
    r.val = st.c.unitRange.slot s₀ + b → placedA st r j = placedA (flat st) r j ∧ placedB st r j = placedB (flat st) r j

/-! ## What `HmOut.checkRows` checks -/

/-- `HmOut.checkRows` passes only when the rows are the input rows, then one per output port. -/
theorem checkRows_le {c : Circuit} (h : HmOut.checkRows c = .ok ()) : c.outPorts.size ≤ c.ports.size := by
  unfold HmOut.checkRows at h
  obtain ⟨k, -, h⟩ := ebind_ok h
  by_contra hlt
  have hne : (k + c.outPorts.size != c.ports.size) = true := by
    simp only [bne_iff_ne, ne_eq]
    omega
  simp only [hne, Bool.true_or, ↓reduceIte] at h
  cases h

/-! ## The unit facts, from a parse and `HmRow.delta`'s Δ -/

/-- **What the copy positions read**, for a statement whose circuit `HmRow.parse` accepted and whose Δ is
`HmRow.delta`'s: `Aliased.setupH_unitFacts`, from those two facts alone. -/
theorem unitFacts_of_parse {tags : Tags} {cf : ByteArray} {tables : List (String × ByteArray)} {st : Stmt}
    (hp : HmRow.parse tags cf tables = .ok st.c) (hda : st.da = (HmRow.delta st.c st.pin).1) :
    Aliased.UnitFacts tags st := by
  obtain ⟨P, ⟨more, hm, hmore⟩, hunit, hsn, hhn⟩ := parse_facts_nets (pre := #[]) (fun _ hx => by simp at hx) hp
  obtain ⟨ls, hcheck⟩ := P.check
  have C := check_facts P.typed hcheck
  have hname : (st.c.nets.getD st.c.unitNet default).1 = "unit" := by
    obtain ⟨hlt, hp', -⟩ := Array.findIdx?_eq_some_iff_getElem.1 hunit
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hlt, Option.getD_some]
    simpa using hp'
  have hmem : st.c.nets.getD st.c.unitNet default ∈ more := by
    have : st.c.nets.getD st.c.unitNet default ∈ st.c.nets.toList := getD_mem_toList P.unitNet
    rw [hm] at this
    simpa using this
  have hrows : st.c.unit.a.rows = st.c.unit.useful ∧ st.c.unit.b.rows = st.c.unit.useful := by
    rcases hmore _ hmem with ⟨lines, hl⟩ | ⟨_, _, _, _, _, hb⟩
    · exact Aliased.parse_rows hl
    · have h1 := (build_facts hb).1
      have h2 : (st.c.nets.getD st.c.unitNet default).2.lookup.isSome = false := C.unit_lookup
      rw [h1] at h2
      cases h2
  refine ⟨P, C, by rw [hda]; exact delta_flat st.c st.pin P.typed, fun he => ?_, fun he => ?_,
    hrows.1.le, hrows.2.le⟩
  · rw [← he, hname] at hsn
    exact absurd hsn (by unfold HmRow.SHA_NET; decide)
  · rw [← he, hname] at hhn
    exact absurd hhn (by unfold HmRow.HM_NET; decide)

variable {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
  {draw partition : Option Json} {program : Option ByteArray} {st : Stmt}

/-- **The flat twin of an untyped hidden statement has the layout the placement reads** (`layout_of`). -/
theorem flat_layout (ht : tags.typed = false)
    (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) : Layout (flat st) :=
  have s := Hidden.setupHidden_spec' h
  layout_of (st := flat st) (parse_facts (Hidden.parse_untyped ht s.parse).1) s.pin rfl rfl s.m s.kLog

/-- **The flat twin's unit facts** (`unitFacts_of_parse`). -/
theorem flat_unitFacts (ht : tags.typed = false)
    (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) :
    Aliased.UnitFacts tags (flat st) :=
  unitFacts_of_parse (st := flat st) (Hidden.parse_untyped ht (Hidden.setupHidden_spec' h).parse).1 rfl

/-! ## Δ's output pairs are at the rows' `sha512x3` slots -/

/-- The first column of each of Δ's output pairs is `deltaMsgCol` at a VU, output row, word and bit. -/
theorem outPairs_fst {c : Circuit} {q : ℕ × ℕ} (hq : q ∈ Hidden.outPairs c) :
    ∃ g < c.g, ∃ j < c.outPorts.size, ∃ k < (c.ports.getD (HmOut.nIn c + j) default).words, ∃ t < 16,
      q.1 = Hidden.deltaMsgCol c g (HmOut.nIn c + j) (2 * k + t / 8) (t % 8) := by
  simp only [Hidden.outPairs, List.mem_flatMap, List.mem_range, List.mem_cons, List.not_mem_nil, or_false] at hq
  obtain ⟨g, hg, j, hj, k, hk, t, ht, hq⟩ := hq
  exact ⟨g, hg, j, hj, k, hk, t, ht, by rcases hq with rfl | rfl <;> rfl⟩

/-- **A row's message bit at its `sha512x3` slot is no unit slot's row**: compression `byte / 128` of row `p` is one of
the VU's `sha512x3` slots, its message input 1024 bits wide, and the bit is inside it. -/
theorem deltaMsgCol_ne_unit (L : Layout st) (P : ParseFacts tags st.c) (C : CheckFacts st.c)
    (hsha : st.c.unitNet ≠ st.c.shaNet) {g p byte t : ℕ} (hg : g < st.c.g) (hp : p < st.c.ports.size)
    (hbyte : byte / 128 < st.c.ports[p]!.chunks) (ht : t < 8) {s₀ b : ℕ} (hs₀ : s₀ < st.c.unitRange.count)
    (hb : b < 2 ^ st.c.unit.unitLog) : Hidden.deltaMsgCol st.c g p byte t ≠ st.c.unitRange.slot s₀ + b := by
  obtain ⟨h1, h2⟩ := Hm.rowSlot_lt hp hbyte
  have hsz := C.sha_in.1
  have hidx : 2 * ((HmRow.rowSlots st.c.ports).2[p]!)[byte / 128]!.2 + 1 < 6 := by omega
  have hw := C.sha_in.2 _ hidx
  rw [sha_width _ _ (by norm_num : 1 < 2)] at hw
  have hne := L.ne_unit_at P C P.shaNet hsha.symm hg (by rw [C.perVu_sha]; exact h1)
    (getElem_mem_toList' (by rw [hsz]; exact hidx)) (tt := 64 * (byte % 128 / 8) + 56 - 8 * (byte % 128 % 8) + t)
    (by rw [hw]; omega) hs₀ hb
  dsimp only [Hidden.deltaMsgCol]
  intro he
  refine hne (Eq.trans ?_ he)
  omega

/-- **No output pair of Δ is at a unit slot's row**, for `sha512/row/v1` rows: every output row's bytes are within its
compressions (`2 · words = 128 · chunks`). -/
theorem outPairs_ne_unit (L : Layout st) (P : ParseFacts tags st.c) (C : CheckFacts st.c)
    (hsha : st.c.unitNet ≠ st.c.shaNet) (hv1 : ∀ p ∈ st.c.ports.toList, p.v2 = false)
    (hle : st.c.outPorts.size ≤ st.c.ports.size) {q : ℕ × ℕ} (hq : q ∈ Hidden.outPairs st.c) {s₀ b : ℕ}
    (hs₀ : s₀ < st.c.unitRange.count) (hb : b < 2 ^ st.c.unit.unitLog) : q.1 ≠ st.c.unitRange.slot s₀ + b := by
  obtain ⟨g, hg, j, hj, k, hk, t, ht, hq1⟩ := outPairs_fst hq
  rw [hq1]
  have hp : HmOut.nIn st.c + j < st.c.ports.size := by unfold HmOut.nIn; omega
  have hw := P.portWords _ (getElem_mem_toList' hp) (hv1 _ (getElem_mem_toList' hp))
  have e : st.c.ports.getD (HmOut.nIn st.c + j) default = st.c.ports[HmOut.nIn st.c + j]! := by
    rw [getElem!_pos st.c.ports _ hp, Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hp, Option.getD_some]
  rw [e] at hk
  exact deltaMsgCol_ne_unit L P C hsha hg hp (by omega) (Nat.mod_lt _ (by norm_num)) hs₀ hb

/-- Δ's count at a row with no output pair is `HmRow.delta`'s. -/
theorem deltaEntry_append_out {d : Array (ℕ × ℕ)} {c : Circuit} {r : ℕ}
    (hr : ∀ q ∈ Hidden.outPairs c, q.1 ≠ r) (j : ℕ) :
    deltaEntry (d ++ (Hidden.outPairs c).toArray) r j = deltaEntry d r j := by
  rw [deltaEntry_eq_count, deltaEntry_eq_count, Array.toList_append, List.count_append, List.toList_toArray,
    List.count_eq_zero.2 fun hm => hr _ hm rfl, Nat.add_zero]

/-- **An untyped hidden statement reads its flat twin's rows at every unit slot's row**, for `sha512/row/v1` rows, when
its Δ being `HmRow.delta`'s, then `outPairs` ([wave2-hidden]'s `Hidden.hidden_delta'`). -/
theorem unitRowsFlat (ht : tags.typed = false)
    (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st)
    (hv1 : ∀ p ∈ st.c.ports.toList, p.v2 = false) : UnitRowsFlat st := by
  have s := Hidden.setupHidden_spec' h
  obtain ⟨hp, hcr⟩ := Hidden.parse_untyped ht s.parse
  have L := flat_layout ht h
  have F := flat_unitFacts ht h
  intro s₀ hs₀ b hb r j hr
  have hout : ∀ q ∈ Hidden.outPairs st.c, q.1 ≠ r.val := fun q hq => by
    rw [hr]; exact outPairs_ne_unit L F.parse F.check F.sha hv1 (checkRows_le hcr) hq hs₀ hb
  have hda : st.da = (HmRow.delta st.c st.pin).1 ++ (Hidden.outPairs st.c).toArray := by rw [s.da, Hidden.hidden_delta']
  have hdb : st.db = (HmRow.delta st.c st.pin).2 ++ (Hidden.outPairs st.c).toArray := by rw [s.db, Hidden.hidden_delta']
  refine ⟨?_, ?_⟩
  · show rangesEntry st (slotEntryA st) r.val j.val + deltaEntry st.da r.val j.val =
      rangesEntry st (slotEntryA st) r.val j.val + deltaEntry (HmRow.delta st.c st.pin).1 r.val j.val
    rw [hda, deltaEntry_append_out hout]
  · show rangesEntry st (slotEntryB st) r.val j.val + deltaEntry st.db r.val j.val =
      rangesEntry st (slotEntryB st) r.val j.val + deltaEntry (HmRow.delta st.c st.pin).2 r.val j.val
    rw [hdb, deltaEntry_append_out hout]

end FlockSoundness.Discharge.HiddenExec
