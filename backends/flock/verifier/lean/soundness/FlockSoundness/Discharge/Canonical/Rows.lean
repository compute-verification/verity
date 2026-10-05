import FlockSoundness.Discharge.Canonical.Outputs
import FlockSoundness.Discharge.Canonical.Reads
import FlockSoundness.Discharge.Canonical.Shared
import FlockSoundness.Protocol.Discharge.Canonical.Rows
import FlockSoundness.Discharge.Canonical.Commit
import FlockSoundness.Discharge.HiddenExec.KeyH

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

/-! ## The readers -/

section places

variable {Reg : Type} {mPts : ℕ} {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length} {si : SitesH plan tab}
  (hparse : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S),
    HmRow.parse (si.acc S R u h).tags (si.acc S R u h).cf (si.acc S R u h).tables = .ok c)
  (outs : Finset (Fin (keyN c hU n))) (popOf : Reg → Public)

variable (hfmt : formatOk c = true) (houts : OutsOut c hU n outs)

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
