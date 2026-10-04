import FlockSoundness.Discharge.Canonical.Outputs
import FlockSoundness.Discharge.Canonical.Reads
import FlockSoundness.Discharge.Canonical.Shared

set_option Elab.async false

/-!
# `lay` and `rs` at `keyProg`'s places in canonical tables

`HiddenExec/KeyH.lean` at the canonical format. Every row is a `sha512/row/v2` bit row of its own length
`rowW c q = c.ports[q].bits`, input or output, and its bytes are `RowV2.enc`'s, `P(ℓ) ‖ packBits(bits)`:

* `rowEncC`, `rowEncC_inj`, `rowEncC_row`: the layout's values' encoding, `RowV2.enc` at a value of width `ℓ < 2^64`
  with nothing set past it (`CanonShaped`), and a longer code of every other value (`Layout.rowEnc`'s shape at `v2`);
* `outRowOfC`, `rowOfC`: `outRowOf`, `rowOfH` at `outBitRowC` (runs of output ports, exact length);
* `keyReadersC`: `Layout.Readers.ofRows` at `rowW c`;
* `keyProgC_copies`, `keyProgC_bcPinned`, `keyProgC_valPinned`: `keyProgH_*` with `out_copyC`, `bc_site_ofC` (any public
  file) and `outBitRowC_ne_none`, without `outNet = unitNet` or `sha512/row/v1` rows;
* `keyLayoutC`, `keyRowsC`, `keyRowC_enc`, `keyRowsC_computes`: `rowEncC` for the rows, `bitCode` elsewhere, and
  `HmRowComputes` at them from `bc_site_commitC`.

Every declaration here is proved. Lengths separate `rowEncC`'s branches (`128 + ⌈ℓ/8⌉ ≤ 128 + W < 128 + W + 2 W + 1`);
with a tail `HmRow.parse` requires every `leavesOut` empty (`check_tail`), so `OutColAt` is vacuous there; and a `v2` row
is below `2^64` bits because `HmRow.rowPorts` refuses more than `MAX_COMPS` compression blocks (`parse_bits_lt`).
-/

namespace FlockSoundness.Discharge.Canonical

open Flock FlockLevel3 Model Audit Binding Placed
open FlockSoundness.Discharge.Layout (keyN keyRowOf OutColAt outDecode OutsOut KeysRow BitVal widthOf padBits bitCode)
open FlockSoundness.Discharge.HiddenExec (AcceptedH SitesH siteAddrH keyRowAddrH keyBcAddrH keySaltAddrH keyPlaceH
  outWordOf unit_wire constCols_keyProgH zeroCols_keyProgH)

/-! ## The canonical row encoding -/

section Enc

variable {W : ℕ}

/-- **A canonical row's value**: a width `ℓ < 2^64` (a u64 bit length) and no bit set past it. -/
def CanonShaped (v : BitVal W) : Prop := v.1.val < 2 ^ 64 ∧ ∀ i : Fin W, v.1.val ≤ i.val → v.2 i = false

instance : DecidablePred (CanonShaped (W := W)) := fun _ => inferInstanceAs (Decidable (_ ∧ _))

/-- A value's `ℓ` bits. -/
def rowBitsOf (v : BitVal W) : List Bool := List.ofFn fun i : Fin v.1.val => Layout.BitVal.get v i.val

/-- **The bytes `sha512/row/v2` hashes for a value**: `P(ℓ) ‖ packBits(bits)` (`RowV2.enc`) for a canonical row, and
`128 + W` zero bytes then `bitCode` for any other value, longer than every row. -/
def rowEncC (v : BitVal W) : List UInt8 :=
  if CanonShaped v then RowV2.enc (rowBitsOf v) else List.replicate (128 + W) 0 ++ bitCode v

theorem rowBitsOf_length (v : BitVal W) : (rowBitsOf v).length = v.1.val := List.length_ofFn

theorem rowEncC_canon_length {v : BitVal W} (h : CanonShaped v) : (rowEncC v).length = 128 + (v.1.val + 7) / 8 := by
  simp only [rowEncC, h, ↓reduceIte, RowV2.enc, List.length_append, RowV2.prefix_length, RowV2.packBits_length,
    rowBitsOf_length]

theorem rowEncC_other_length {v : BitVal W} (h : ¬ CanonShaped v) : (rowEncC v).length = 128 + W + (2 * W + 1) := by
  simp only [rowEncC, h, ↓reduceIte, List.length_append, List.length_replicate, Layout.bitCode_length]

/-- **`rowEncC` is injective**: two rows by `RowV2.enc_inj` and nothing set past `ℓ`; every other value is longer than
every row, and `bitCode` is injective. -/
theorem rowEncC_inj : Function.Injective (rowEncC (W := W)) := by
  intro v w h
  have hv1 := v.1.isLt
  have hw1 := w.1.isLt
  by_cases hv : CanonShaped v <;> by_cases hw : CanonShaped w
  · have he : RowV2.enc (rowBitsOf v) = RowV2.enc (rowBitsOf w) := by
      simpa only [rowEncC, hv, hw, ↓reduceIte] using h
    have hb : rowBitsOf v = rowBitsOf w := congrArg Subtype.val
      (@RowV2.enc_inj ⟨rowBitsOf v, by rw [rowBitsOf_length]; exact hv.1⟩
        ⟨rowBitsOf w, by rw [rowBitsOf_length]; exact hw.1⟩ he)
    have hk : v.1 = w.1 := Fin.ext (by simpa only [rowBitsOf_length] using congrArg List.length hb)
    refine Prod.ext hk (funext fun i => ?_)
    by_cases hi : i.val < v.1.val
    · have e := List.getElem_of_eq hb (by rw [rowBitsOf_length]; exact hi)
      simp only [rowBitsOf, List.getElem_ofFn, Layout.BitVal.get, i.isLt, ↓reduceDIte] at e
      exact e
    · rw [hv.2 i (by omega), hw.2 i (by rw [← hk]; omega)]
  · have hl := congrArg List.length h
    rw [rowEncC_canon_length hv, rowEncC_other_length hw] at hl
    omega
  · have hl := congrArg List.length h
    rw [rowEncC_other_length hv, rowEncC_canon_length hw] at hl
    omega
  · simp only [rowEncC, hv, hw, ↓reduceIte] at h
    exact Layout.bitCode_inj (List.append_inj h rfl).2

/-- **`rowEncC` of a row of `K < 2^64` bits** is `P(K)` and the row's packed bits. -/
theorem rowEncC_row {K : ℕ} (hKW : K ≤ W) (hK : K < 2 ^ 64) (b : Fin K → Bool) :
    rowEncC (widthOf W K, (padBits b : Fin W → Bool)) =
      (HmRow.rowPrefixV2 K).data.toList ++ HmRow.packBits (List.ofFn b) := by
  have hw : (widthOf W K).val = K := by simp only [widthOf]; omega
  have hs : CanonShaped (widthOf W K, (padBits b : Fin W → Bool)) :=
    ⟨by change (widthOf W K).val < 2 ^ 64; rw [hw]; exact hK, fun i hi => by
      change (widthOf W K).val ≤ i.val at hi
      rw [hw] at hi
      simp only [padBits, show ¬ i.val < K by omega, ↓reduceDIte]⟩
  have hb : rowBitsOf (widthOf W K, (padBits b : Fin W → Bool)) = List.ofFn b := by
    refine List.ext_getElem (by simp only [rowBitsOf, List.length_ofFn, hw]) fun j h1 h2 => ?_
    have hj : j < K := by simpa using h2
    simp only [rowBitsOf, List.getElem_ofFn, Layout.BitVal.get, show j < W by omega, ↓reduceDIte, padBits, hj]
  simp only [rowEncC, hs, ↓reduceIte, RowV2.enc, hb, List.length_ofFn]

end Enc

set_option maxHeartbeats 1000000 in
/-- **`HmRow.rowPorts`' bit rows have at most `MAX_COMPS` compressions** (`Hidden.rowPorts_v2`'s proof, to its bound). -/
theorem rowPorts_v2_comps {mj : Lean.Json} {typed : Bool} {ports : Array Port} (h : HmRow.rowPorts mj typed = .ok ports) :
    ∀ p ∈ ports.toList, p.v2 = true → p.chunks ≤ HmRow.MAX_COMPS := by
  unfold HmRow.rowPorts at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) v2 one at h
  obtain ⟨s, -, h⟩ := except_bind_ok h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp0 at h
  replace h := ite_throw_ok h
  simp (config := { zeta := false }) only [jp0] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp1 at h
  replace h := ite_throw_ok h
  simp (config := { zeta := false }) only [jp1] at h
  obtain ⟨listedPorts, -, h⟩ := except_bind_ok h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp2 at h
  obtain ⟨lengths, h⟩ : ∃ l, jp2 l = .ok ports := by
    split at h
    · obtain ⟨s, -, h⟩ := except_bind_ok h
      extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpv at h
      replace h := ite_throw_ok h
      simp (config := { zeta := false }) only [jpv] at h
      obtain ⟨l, -, h⟩ := except_bind_ok h
      exact ⟨l, h⟩
    · exact ⟨_, h⟩
  simp (config := { zeta := false }) only [jp2] at h
  obtain ⟨ports', hmap, h⟩ := except_bind_ok h
  obtain ⟨listed, -, h⟩ := except_bind_ok h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) bad jp3 at h
  replace h := ite_throw_ok h
  simp (config := { zeta := false }) only [jp3] at h
  obtain ⟨u, -, h⟩ := except_bind_ok h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp4 at h
  replace h := ite_throw_ok h
  simp (config := { zeta := false }) only [jp4] at h
  cases h
  intro p hp hv
  obtain ⟨i, -, hf⟩ := array_mapM_ok hmap p hp
  obtain ⟨a, -, hf⟩ := except_bind_ok hf
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpa at hf
  replace hf := ite_throw_ok hf
  simp (config := { zeta := false }) only [jpa] at hf
  obtain ⟨name, -, hf⟩ := except_bind_ok hf
  obtain ⟨words, -, hf⟩ := except_bind_ok hf
  obtain ⟨comps, -, hf⟩ := except_bind_ok hf
  split at hf
  · extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpn at hf
    replace hf := ite_throw_ok hf
    simp (config := { zeta := false }) only [jpn] at hf
    cases hf
    cases hv
  · rename_i len seg _
    extract_lets (config := { underBinder := false, onlyGivenNames := true }) serial segs want jpw at hf
    obtain ⟨hbad, hf⟩ := ite_throw_ok' hf
    simp (config := { zeta := false }) only [jpw] at hf
    cases hf
    simp only [want, Bool.or_eq_true, decide_eq_true_eq, bne_iff_ne, ne_eq, not_or, Decidable.not_not,
      Prod.mk.injEq] at hbad
    obtain ⟨hmax, -, hc⟩ := hbad
    show comps ≤ _
    rw [hc]
    exact Nat.le_of_not_lt hmax

/-- **A canonical row is shorter than `2^64` bits**: its compressions, at most `MAX_COMPS`, fix its length's bytes. -/
theorem parse_bits_lt {tags : Tags} {bytes : ByteArray} {tables : List (String × ByteArray)} {c : Flock.Circuit}
    (hU : Built c.unit) (h : HmRow.parse tags bytes tables = .ok c) (hfmt : formatOk c = true) (q : Fin c.ports.size) :
    c.ports[q].bits < 2 ^ 64 := by
  obtain ⟨⟨mj, typed, hrp⟩, -⟩ := Hidden.parse_rowPorts h
  have hmem : c.ports[q] ∈ c.ports.toList := Array.getElem_mem_toList _
  obtain ⟨hv, hs⟩ := (formatOk_spec hU hfmt).1 _ hmem
  have hc := (Hidden.rowPorts_v2 hrp _ hmem hv).2.1 hs
  have hm := rowPorts_v2_comps hrp _ hmem hv
  simp only [HmRow.MAX_COMPS] at hm
  have h64 : (2 : ℕ) ^ 64 = 18446744073709551616 := by norm_num
  rw [h64]
  omega

/-! ## The row bits of `keyProg` at canonical rows -/

variable {c : Flock.Circuit} {hU : Built c.unit} {n : ℕ}

/-- **A tail that returns the outputs leaves the units none**: `HmRow.check` refuses a nonempty `leavesOut` when the
out net is not the unit's. -/
theorem check_tail {tags : Tags} {ls : Lean.Json} (ht : c.typed = none) (ho : c.outNet ≠ c.unitNet)
    (h : HmRow.check c tags ls = .ok ()) (k : ℕ) : (c.leavesOut.getD k #[]).size = 0 := by
  unfold HmRow.check at h
  obtain ⟨_spans, -, h⟩ := except_bind_ok h
  obtain ⟨_u0, -, h⟩ := except_bind_ok h
  obtain ⟨-, h⟩ := Layout.ite_throw_cond h
  obtain ⟨-, h⟩ := Layout.ite_throw_cond h
  obtain ⟨-, h⟩ := Layout.ite_throw_cond h
  simp only [ht] at h
  obtain ⟨-, h⟩ := Layout.ite_throw_cond h
  obtain ⟨-, h⟩ := Layout.ite_throw_cond h
  rw [ite_eq_left (bne_iff_ne.2 ho)] at h
  obtain ⟨hc, -⟩ := Layout.ite_throw_cond h
  simp only [Bool.or_eq_true, not_or, Array.any_eq_true', not_exists, not_and, Bool.not_eq_true'] at hc
  by_cases hu : k < c.leavesOut.size
  · have := hc.2 _ (Array.getElem_mem hu)
    simp only [Bool.not_eq_false, Array.isEmpty_iff_size_eq_zero] at this
    simpa [Array.getD, hu] using this
  · simp [Array.getD, hu]

/-- **A returned output's word is an output word**, with or without a tail. -/
theorem outWord_lt_of_parse {tags : Tags} {bytes : ByteArray} {tables : List (String × ByteArray)}
    (h : HmRow.parse tags bytes tables = .ok c) {x u' j t : ℕ} (hoc : OutColAt c hU x u' j t) :
    outWordOf c (u', j, t) < c.outWords := by
  have P := parse_facts h
  obtain ⟨ls, hls⟩ := P.check
  obtain ⟨hu', hj, -⟩ := hoc
  by_cases ho : c.outNet = c.unitNet
  · obtain ⟨-, -, hlo⟩ := Layout.check_out P.typed ho hls
    have hget : c.leavesOut.getD u' #[] = c.leavesOut[u']! := by
      rw [Hidden.getElem!_eq_getD']; rfl
    rw [show outWordOf c (u', j, t) = (c.leavesOut.getD u' #[])[j] from getElem!_pos _ j hj]
    refine (hlo u' hu').2 _ ?_
    rw [← hget]
    exact Array.getElem_mem_toList _
  · have := check_tail P.typed ho hls u'
    omega

variable (c hU n) in
open Classical in
/-- **The canonical output-row bit of a unit's computed gate** (`HiddenExec.outRowOf` at `outBitRowC`). -/
noncomputable def outRowOfC (outs : Finset (Fin (keyN c hU n))) (g : Fin (keyN c hU n)) :
    Option (Fin n × (q : Fin c.ports.size) × Fin (rowW c q)) :=
  match (keyProg c hU n).unit g with
  | some u =>
    if h : ∃ x, (Const.Prog.instC (keyProg c hU n) outs u).wire x = g then
      outBitRowC c u (outDecode c hU (Classical.choose h).val)
    else none
  | none => none

theorem outRowOfC_wire (outs : Finset (Fin (keyN c hU n))) {u : Fin n} (x : Fin ((keyProg c hU n).rows u).w)
    (hgu : (keyProg c hU n).unit ((Const.Prog.instC (keyProg c hU n) outs u).wire x) = some u)
    (hx : ((keyProg c hU n).rows u).nIn ≤ x.val) :
    outRowOfC c hU n outs ((Const.Prog.instC (keyProg c hU n) outs u).wire x) =
      outBitRowC c u (outDecode c hU x.val) := by
  have hc : ∃ x', (Const.Prog.instC (keyProg c hU n) outs u).wire x' =
      (Const.Prog.instC (keyProg c hU n) outs u).wire x := ⟨x, rfl⟩
  have hch : Classical.choose hc = x := by
    rcases (Const.Prog.instC (keyProg c hU n) outs u).shared _ x (Classical.choose_spec hc) with he | ⟨-, h2⟩
    · exact he
    · omega
  unfold outRowOfC
  rw [hgu]
  dsimp only
  rw [dite_eq_left hc, hch]

theorem outRowOfC_wire_some (outs : Finset (Fin (keyN c hU n))) {u u' : Fin n} (x : Fin ((keyProg c hU n).rows u).w)
    {q : Fin c.ports.size} {k : Fin (rowW c q)}
    (h : outRowOfC c hU n outs ((Const.Prog.instC (keyProg c hU n) outs u).wire x) = some (u', ⟨q, k⟩)) :
    (keyProg c hU n).unit ((Const.Prog.instC (keyProg c hU n) outs u).wire x) = some u ∧
      ((keyProg c hU n).rows u).nIn ≤ x.val ∧ outBitRowC c u (outDecode c hU x.val) = some (u', ⟨q, k⟩) := by
  rcases hgu : (keyProg c hU n).unit ((Const.Prog.instC (keyProg c hU n) outs u).wire x) with _ | u''
  · unfold outRowOfC at h
    rw [hgu] at h
    exact absurd h (by simp)
  · obtain ⟨rfl, hx⟩ := unit_wire outs x hgu
    exact ⟨rfl, hx, (outRowOfC_wire outs x hgu hx).symm.trans h⟩

variable (c hU n) in
/-- **The row bits of `keyProg` at a canonical statement**: an input row's bit (`Layout.keyRowOf` at `rowW c`), else a
canonical output row's (`outRowOfC`). -/
noncomputable def rowOfC (outs : Finset (Fin (keyN c hU n))) (g : Fin (keyN c hU n)) :
    Option (Fin n × (q : Fin c.ports.size) × Fin (rowW c q)) :=
  match keyRowOf c hU n (rowW c) g with
  | some r => some r
  | none => outRowOfC c hU n outs g

theorem rowOfC_of_key (outs : Finset (Fin (keyN c hU n))) {g : Fin (keyN c hU n)}
    {r : Fin n × (q : Fin c.ports.size) × Fin (rowW c q)} (h : keyRowOf c hU n (rowW c) g = some r) :
    rowOfC c hU n outs g = some r := by
  unfold rowOfC
  rw [h]

theorem rowOfC_of_none (outs : Finset (Fin (keyN c hU n))) {g : Fin (keyN c hU n)}
    (h : keyRowOf c hU n (rowW c) g = none) : rowOfC c hU n outs g = outRowOfC c hU n outs g := by
  unfold rowOfC
  rw [h]

theorem rowOfC_none (outs : Finset (Fin (keyN c hU n))) {g : Fin (keyN c hU n)} (h : rowOfC c hU n outs g = none) :
    keyRowOf c hU n (rowW c) g = none ∧ outRowOfC c hU n outs g = none := by
  rcases hk : keyRowOf c hU n (rowW c) g with _ | r
  · exact ⟨rfl, (rowOfC_of_none outs hk).symm.trans h⟩
  · rw [rowOfC_of_key outs hk] at h
    cases h

theorem rowOfC_wire_unit (outs : Finset (Fin (keyN c hU n))) {u u' : Fin n} (x : Fin ((keyProg c hU n).rows u).w)
    {q : Fin c.ports.size} {k : Fin (rowW c q)}
    (h : rowOfC c hU n outs ((Const.Prog.instC (keyProg c hU n) outs u).wire x) = some (u', ⟨q, k⟩)) : u' = u := by
  rcases hk : keyRowOf c hU n (rowW c) ((Const.Prog.instC (keyProg c hU n) outs u).wire x) with _ | r
  · rw [rowOfC_of_none outs hk] at h
    exact (outBitRowC_spec (outRowOfC_wire_some outs x h).2.2).1
  · rw [rowOfC_of_key outs hk] at h
    rw [h] at hk
    exact (Layout.keyRowOf_wire outs x hk).1

/-! ## The readers -/

section places

variable {Reg : Type} {mPts : ℕ} {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length} {si : SitesH plan tab}
  (hparse : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S),
    HmRow.parse (si.acc S R u h).tags (si.acc S R u h).cf (si.acc S R u h).tables = .ok c)
  (outs : Finset (Fin (keyN c hU n))) (popOf : Reg → Public)

/-- **The readers at `keyProg`'s sites in canonical tables** (`keyReadersH` at `rowW c` and `rowOfC`). -/
noncomputable def keyReadersC : Layout.Readers ((keyProg c hU n).circuit outs) plan tab :=
  Layout.Readers.ofRows (keyPlaceH hparse outs) (Layout.keyN_pos c hU n) (rowW c)
    (W := Finset.univ.sup (rowW c) + 1)
    (fun q => (Finset.le_sup (Finset.mem_univ q)).trans (Nat.le_succ _)) (Nat.le_add_left 1 _) (rowOfC c hU n outs)
    (keyRowAddrH si) (keySaltAddrH si (hU := hU)) (keyBcAddrH si)
    (fun R u q => ((popOf R).digest u.val q.val c.ports.size).data.toList)
    (Layout.keyGate (c := c) (hU := hU) (n := n) fun _ _ => false)

variable (hfmt : formatOk c = true) (houts : OutsOut c hU n outs)

include hparse hfmt houts in
/-- **Δ's copies at `keyProg`'s sites in canonical tables** (`keyProgH_copies` with `out_copyC`). -/
theorem keyProgC_copies : (keyReadersC hparse outs popOf).Copies (keyPlaceH hparse outs) := by
  refine Layout.ofRows_copies _ fun S R u hu M hM x hx u' q k hr => ?_
  have hc := (si.acc S R u hu).c_eq (hparse S R u hu)
  have hrw := SitesH.rowsOf_keyProg hU hparse S R u hu
  rcases hk : keyRowOf c hU n (rowW c) ((Const.Prog.instC (keyProg c hU n) outs u).wire x) with _ | r
  · have ho' : outRowOfC c hU n outs ((Const.Prog.instC (keyProg c hU n) outs u).wire x) = some (u', ⟨q, k⟩) :=
      (rowOfC_of_none outs hk).symm.trans hr
    obtain ⟨hgu, -, hb⟩ := outRowOfC_wire_some outs x ho'
    obtain ⟨u'', j, t, hoc⟩ := houts u _ hx hgu x rfl
    rw [Layout.outDecode_of hoc] at hb
    have := out_copyC (si.acc S R u hu) hc hfmt hM (si.o S R u hu) (si.hvu S R u hu)
      (Fin.cast (congrArg FlockSoundness.Rows.w hrw) x) hoc hb
    simp only [keyRowAddrH, hu, ↓reduceDIte]
    exact this
  · have hr' : keyRowOf c hU n (rowW c) ((Const.Prog.instC (keyProg c hU n) outs u).wire x) =
        some (u', ⟨q, k⟩) := hk.trans ((rowOfC_of_key outs hk).symm.trans hr)
    obtain ⟨-, hxn, hkey⟩ := Layout.keyRowOf_wire outs x hr'
    have hk' : Aliased.srcKey (si.acc S R u hu).st.c x.val = .msg q.val (k.val / 8) (k.val % 8) := by
      rw [hc]; exact hkey
    have hin' : Typed.msgCol (si.acc S R u hu).st.c (si.vu S R u hu) q.val (k.val / 8) (k.val % 8) <
        2 ^ (si.acc S R u hu).st.c.kLog := by
      have hv := si.hvu S R u hu
      rw [hc] at hv ⊢
      exact Layout.msgCol_lt_of_parse (hparse S R u hu) hv q.isLt _ _ (Nat.mod_lt _ (by norm_num))
    have := HiddenExec.copy_siteH (si.acc S R u hu) hM (si.o S R u hu) (si.hvu S R u hu)
      (Fin.cast (congrArg FlockSoundness.Rows.w hrw) x) (lt_of_lt_of_eq hxn (congrArg FlockSoundness.Rows.nIn hrw))
      hk' hin'
    simp only [keyRowAddrH, hu, ↓reduceDIte]
    exact this

include hparse in
/-- **The rows' `b ‖ c` at `keyProg`'s sites in canonical tables is the registration's** (`keyProgH_bcPinned` at a
`BcSiteC`: any public file, shared rows or not). -/
theorem keyProgC_bcPinned
    (hsite : ∀ S R u (h : u ∈ S) M, Binding.Sat plan tab S R u M →
      BcSiteC (si.acc S R u h) (popOf R) (si.o S R u h) (si.vu S R u h) u.val) :
    (keyReadersC hparse outs popOf).BcPinned (P := (keyProg c hU n).partition outs) := by
  refine Layout.ofRows_bcPinned _ fun S R u hu M hM g hg u' q k hr => ?_
  obtain ⟨x, rfl⟩ := Layout.exists_col_of_io ((keyPlaceH hparse outs) S R u hu).inst hg
  rw [rowOfC_wire_unit outs x hr]
  have hc := (si.acc S R u hu).c_eq (hparse S R u hu)
  have hnp : (si.acc S R u hu).st.c.ports.size = c.ports.size := by rw [hc]
  have := bc_site_ofC (hsite S R u hu M hM) hM (si.hvu S R u hu) (p := q.val) (by rw [hnp]; exact q.isLt)
  rw [hnp] at this
  unfold keyBcAddrH
  simp only [hu, ↓reduceDIte]
  exact this

include hparse hfmt houts in
/-- **The single gates at `keyProg`'s sites in canonical tables are the constant and the zero**
(`keyProgH_valPinned` with `outBitRowC_ne_none` and `outWord_lt_of_parse`). -/
theorem keyProgC_valPinned (hkeys : KeysRow c hU (rowW c)) :
    (keyReadersC hparse outs popOf).ValPinned (P := (keyProg c hU n).partition outs) := by
  refine Layout.ofRows_valPinned _ fun S R u hu M hM g hg hr => ?_
  have hpub := Layout.gate_pub (keyPlaceH hparse outs) (constCols_keyProgH hU hparse outs)
    (zeroCols_keyProgH hU hparse outs) hu hM hg
  obtain ⟨x, hx⟩ := Layout.exists_col_of_io ((keyPlaceH hparse outs) S R u hu).inst hg
  change FlockSoundness.bit _ = Layout.keyGate (c := c) (hU := hU) (n := n) (fun _ _ => false) R g
  obtain ⟨hk, hno⟩ := rowOfC_none outs hr
  rcases Layout.single_gate outs hkeys ⟨x, hx⟩ hg hk with rfl | rfl | ⟨hm, -, -⟩
  · rw [hpub.2 (Finset.mem_singleton_self _)]
    simp only [Layout.keyGate, Layout.zeroG_ne_oneG, ↓reduceIte]
  · rw [hpub.1 (Finset.mem_singleton_self _)]
    simp only [Layout.keyGate, ↓reduceIte]
  · exfalso
    subst hx
    obtain ⟨u'', j, t, hoc⟩ := houts u _ hg hm x rfl
    have hno' : outRowOfC c hU n outs ((Const.Prog.instC (keyProg c hU n) outs u).wire x) = none := hno
    rw [outRowOfC_wire outs x hm (unit_wire outs x hm).2, Layout.outDecode_of hoc] at hno'
    have hc := (si.acc S R u hu).c_eq (hparse S R u hu)
    have hcr : HmOut.checkRows c = .ok () := hc ▸ (si.acc S R u hu).checkRows
    exact outBitRowC_ne_none hcr hfmt u (outWord_lt_of_parse (hparse S R u hu) hoc) hoc.2.2.1 hno'

/-- **`lay` at `keyProg`'s places in canonical tables**: the readers' layout, `bitCode` for the single gates. -/
noncomputable def keyLayoutC :
    Binding.Layout ((keyProg c hU n).partition outs) plan tab
      (((keyProg c hU n).partition outs).placeDecoder plan tab (keyPlaceH hparse outs)) :=
  (keyReadersC hparse outs popOf).layout (keyPlaceH hparse outs) Layout.bitCode
    (keyProgC_copies hparse outs popOf hfmt houts)

/-- **`rs` at `keyProg`'s places in canonical tables**: the readers' rows, `rowEncC` at the rows (the bytes
`sha512/row/v2` hashes). -/
noncomputable def keyRowsC
    (hsite : ∀ S R u (h : u ∈ S) M, Binding.Sat plan tab S R u M →
      BcSiteC (si.acc S R u h) (popOf R) (si.o S R u h) (si.vu S R u h) u.val)
    (hkeys : KeysRow c hU (rowW c)) :
    Binding.HmRows (keyLayoutC hparse outs popOf hfmt houts) :=
  (keyReadersC hparse outs popOf).rows (keyPlaceH hparse outs) rowEncC rowEncC_inj Layout.bitCode
    Layout.bitCode_inj (keyProgC_copies hparse outs popOf hfmt houts) (keyProgC_bcPinned hparse outs popOf hsite)
    (keyProgC_valPinned hparse outs popOf hfmt houts hkeys)

/-- **The bytes `sha512/row/v2` hashes for a row at `keyProg`'s places in canonical tables**: at a drawn unit `u`'s
site, the value a message holds at row `q` of unit `u'` has `rowEncC` the prefix `P(ℓ)` of the row's `ℓ` bits and the
bits the witness holds at the unit's VU's message columns (`rowBits`), packed. -/
theorem keyRowC_enc {S : Finset (Fin n)} {R : Reg} {u : Fin n} (hu : u ∈ S) (M : Binding.Msg plan tab S R u)
    (u' : Fin n) (q : Fin c.ports.size) :
    rowEncC ((keyLayoutC hparse outs popOf hfmt houts).row S R u M (.inl (u', q))) =
      (HmRow.rowPrefixV2 (si.acc S R u hu).st.c.ports[q.val]!.bits).data.toList ++
        HmRow.packBits (rowBits (fun x => FlockSoundness.bit (Binding.wit plan tab S R u M
          (siteAddrH (si.acc S R u hu) (si.o S R u hu) x))) (si.acc S R u hu).st.c (si.vu S R u hu) q.val) := by
  have hle : rowW c q ≤ Finset.univ.sup (rowW c) + 1 := (Finset.le_sup (Finset.mem_univ q)).trans (Nat.le_succ _)
  have hc := (si.acc S R u hu).c_eq (hparse S R u hu)
  have hlt : rowW c q < 2 ^ 64 := parse_bits_lt hU (hparse S R u hu) hfmt q
  show rowEncC (W := Finset.univ.sup (rowW c) + 1)
    (widthOf (Finset.univ.sup (rowW c) + 1) (rowW c q),
      padBits (bitsAt (Binding.wit plan tab S R u M) fun k : Fin (rowW c q) => keyRowAddrH si S R u q k.val)) = _
  rw [rowEncC_row hle hlt]
  have hb : (si.acc S R u hu).st.c.ports[q.val]!.bits = rowW c q := by
    rw [hc]; exact congrArg Port.bits (getElem!_pos c.ports q.val q.isLt)
  rw [hb]
  congr 2
  refine List.ext_getElem (by simp only [List.length_ofFn, rowBits, hb]) fun j h1 h2 => ?_
  simp only [List.getElem_ofFn, rowBits, bitsAt]
  simp only [keyRowAddrH, hu, ↓reduceDIte]

/-- **`HmRowComputes` at `keyProg`'s rows in canonical tables** (`hHm` for `keyRowsC`): at every drawn unit's
committed wire held by a row, input or output, in a satisfying witness of the unit's table, the `b ‖ c` at the row's
`hm96` output is the commit string of the row's bytes (`rowEncC`) and the salt (`bc_site_commitC`, `keyRowC_enc`). -/
theorem keyRowsC_computes
    (hsite : ∀ S R u (h : u ∈ S) M, Binding.Sat plan tab S R u M →
      BcSiteC (si.acc S R u h) (popOf R) (si.o S R u h) (si.vu S R u h) u.val)
    (hkeys : KeysRow c hU (rowW c)) :
    (keyRowsC hparse outs popOf hfmt houts hsite hkeys).Computes := by
  unfold Binding.HmRows.Computes Assumptions.HmRowComputes
  rintro ⟨S, R, u, M, g⟩ ⟨hu, hM, -, hrow⟩
  dsimp only at hu hM hrow ⊢
  revert hrow
  generalize (keyLayoutC hparse outs popOf hfmt houts).pos g = pos
  rcases pos with ⟨u', q⟩ | g'
  · intro _
    have hce := (si.acc S R u hu).c_eq (hparse S R u hu)
    have hq : q.val < (si.acc S R u hu).st.c.ports.size := by rw [hce]; exact q.isLt
    have hb := bc_site_commitC (si.acc S R u hu) (by rw [hce]; exact hfmt) hM (si.o S R u hu) (si.hvu S R u hu) hq
    show Binding.bytesAt _ (keyBcAddrH si S R u q) = Binding.commitString
      (rowEncC ((keyLayoutC hparse outs popOf hfmt houts).row S R u M (.inl (u', q))))
      (Binding.bytesAt _ (keySaltAddrH si S R u (.inl (u', q)) (hU := hU)))
    have e1 : keyBcAddrH si S R u q = fun (j : Fin 128) (t : Fin 8) => siteAddrH (si.acc S R u hu) (si.o S R u hu)
        (Layout.bcAddr (si.acc S R u hu).st.c (si.vu S R u hu) q.val j.val t.val) := by
      funext j t
      simp only [keyBcAddrH, hu, dite_true]
    have e2 : keySaltAddrH si S R u (.inl (u', q)) (hU := hU) = fun (j : Fin 192) (t : Fin 8) =>
        siteAddrH (si.acc S R u hu) (si.o S R u hu)
          (Hm.saltCol (si.acc S R u hu).st.c (si.vu S R u hu) q.val j.val t.val) := by
      funext j t
      simp only [keySaltAddrH, hu, dite_true]
    rw [keyRowC_enc hparse outs popOf hfmt houts hu M u' q, e1, e2]
    exact hb
  · intro h
    exact (Bool.false_ne_true h).elim

end places

end FlockSoundness.Discharge.Canonical
