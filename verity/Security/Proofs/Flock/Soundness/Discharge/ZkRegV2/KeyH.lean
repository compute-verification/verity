import Proofs.Flock.Soundness.Discharge.ZkRegV2.HiddenSite
import Proofs.Flock.Soundness.Discharge.ZkRegV2.RowEnc

set_option Elab.async false

/-!
# `lay` and `rs` at `keyProg`'s places in hidden-output tables, for `sha512/row/v2` rows

`HiddenExec/KeyH.lean` at unsegmented `sha512/row/v2` rows of `16 · words` bits in whole SHA-512 blocks with one output
port per output row (`hv2`, `hor`): the readers are `keyReadersH`'s, unchanged; `out_copy_keyV2` and
`outBitRow_ne_noneV2` are `out_copy_key` and `outBitRow_ne_none` from `out_copyHV2` and `checkRows_wordsW`, so
`keyProgH_copiesV2` and `keyProgH_valPinnedV2` are `keyProgH_copies` and `keyProgH_valPinned`; `keyLayoutHV2` is
`keyLayoutH`; `keyRowsHV2` is `keyRowsH` with the rows' `sha512/row/v2` bytes (`Layout.rowEncP Hm.preV2`), and
`keyRowsH_computesV2` is `keyRowsH_computes` from `bc_site_commitHV2`, a row's `b ‖ c` under its own prefix
(`Hm.prefixOf_v2`).
-/

namespace FlockSoundness.Discharge.HiddenExec

open Flock FlockLevel3 Model Audit Binding Placed
open FlockSoundness.Discharge.Layout (keyN keyRowOf portRowW OutColAt outDecode OutsOut KeysRow)

variable {c : Flock.Circuit} {hU : Built c.unit} {n : ℕ}

/-- **An output word's bit is an output row's**, at one output port per output row and rows of `16 · words` bits:
`outBitRow_ne_none` from `checkRows_wordsW`. -/
theorem outBitRow_ne_noneV2 (hcr : HmOut.checkRows c = .ok ()) (hrows : Hidden.OutRows c)
    (hb : ∀ p ∈ c.ports.toList, p.bits = 16 * p.words) (u : Fin n)
    {d : ℕ × ℕ × ℕ} (hw : outWordOf c d < c.outWords) (ht : d.2.2 < 16) : outBitRow c u d ≠ none := by
  obtain ⟨hjo, -, hk⟩ := outPortOf_spec c hw
  have hle := checkRows_wordsW hcr hrows.2.1 hb hjo
  obtain ⟨hq, -⟩ := hrows.2.2 _ hjo
  have hpw : portRowW c ⟨_, hq⟩ = 16 * (c.ports.getD (HmOut.nIn c + outPortOf c (outWordOf c d)) default).words := by
    unfold portRowW
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hq, Option.getD_some]
    rfl
  unfold outBitRow
  rw [dite_eq_left hq, dite_eq_left (by rw [hpw]; omega)]
  exact Option.some_ne_none _

/-- **A returned output's column holds its output row's bit**, at `sha512/row/v2` rows: `out_copy_key` from
`out_copyHV2`. -/
theorem out_copy_keyV2 {T : Model.Statement} (a : AcceptedH T) (hce : a.st.c = c) (ho : c.outNet = c.unitNet)
    (hv2 : ∀ p ∈ c.ports.toList, p.v2 = true ∧ p.segs = none ∧ p.bits = 16 * p.words ∧ 2 * p.words = 128 * p.chunks)
    (hor : c.outRowPorts = Array.replicate c.outPorts.size 1)
    {z : Witness T.m} (hz : T.Satisfies z) (o : Fin (2 ^ T.nbl)) {g : ℕ} (hg : g < a.st.c.g) (x : Fin a.rows.w)
    {u' j t : ℕ} (hoc : OutColAt c hU x.val u' j t) {u v : Fin n} {q : Fin c.ports.size} {k : Fin (portRowW c q)}
    (hb : outBitRow c u (u', j, t) = some (v, ⟨q, k⟩)) :
    T.block z o (a.col hg x) = z (siteAddrH a o (Typed.msgCol a.st.c g q.val (k.val / 8) (k.val % 8))) := by
  subst hce
  obtain ⟨-, hq, hk⟩ := outBitRow_spec hb
  obtain ⟨hu', hj, ht, h1, h2, hx⟩ := hoc
  have P := parse_facts a.parse
  obtain ⟨ls, hls⟩ := P.check
  obtain ⟨-, -, hlo⟩ := Layout.check_out P.typed ho hls
  have hget : a.st.c.leavesOut.getD u' #[] = a.st.c.leavesOut[u']! := by
    rw [Hidden.getElem!_eq_getD']; rfl
  have hwj : outWordOf a.st.c (u', j, t) = (a.st.c.leavesOut.getD u' #[])[j] := getElem!_pos _ j hj
  have hwlt : outWordOf a.st.c (u', j, t) < a.st.c.outWords := by
    rw [hwj]
    refine (hlo u' hu').2 _ ?_
    rw [← hget]
    exact Array.getElem_mem_toList _
  obtain ⟨hjo, hle, hkk⟩ := outPortOf_spec a.st.c hwlt
  have hw : Hidden.outAt a.st.c (outPortOf a.st.c (outWordOf a.st.c (u', j, t))) +
      (outWordOf a.st.c (u', j, t) - Hidden.outAt a.st.c (outPortOf a.st.c (outWordOf a.st.c (u', j, t)))) =
      (a.st.c.leavesOut.getD u' #[])[j] := by
    rw [← hwj]
    omega
  have := out_copyHV2 a ho hv2 hor hz o hg x hu' hj ht h1 h2 hx hjo hkk hw
  rw [hq, hk]
  exact this

section places

variable {Reg : Type} {mPts : ℕ} {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
  {si : SitesH plan tab}
  (hparse : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S),
    HmRow.parse (si.acc S R u h).tags (si.acc S R u h).cf (si.acc S R u h).tables = .ok c)
  (outs : Finset (Fin (keyN c hU n))) (popOf : Reg → Public)
  (ho : c.outNet = c.unitNet)
  (hv2 : ∀ p ∈ c.ports.toList, p.v2 = true ∧ p.segs = none ∧ p.bits = 16 * p.words ∧ 2 * p.words = 128 * p.chunks)
  (hor : c.outRowPorts = Array.replicate c.outPorts.size 1)
  (houts : OutsOut c hU n outs)

include hparse ho hv2 hor houts in
/-- **Δ's copies at `keyProg`'s sites in hidden-output tables, at `sha512/row/v2` rows**: `keyProgH_copies` with
`out_copy_keyV2`. -/
theorem keyProgH_copiesV2 : (keyReadersH hparse outs popOf).Copies (keyPlaceH hparse outs) := by
  refine Layout.ofRows_copies _ fun S R u hu M hM x hx u' q k hr => ?_
  have hc := (si.acc S R u hu).c_eq (hparse S R u hu)
  have hrw := SitesH.rowsOf_keyProg hU hparse S R u hu
  rcases hk : keyRowOf c hU n (portRowW c) ((Const.Prog.instC (keyProg c hU n) outs u).wire x) with _ | r
  · have ho' : outRowOf c hU n outs ((Const.Prog.instC (keyProg c hU n) outs u).wire x) = some (u', ⟨q, k⟩) :=
      (rowOfH_of_none outs hk).symm.trans hr
    obtain ⟨hgu, -, hb⟩ := outRowOf_wire_some outs x ho'
    obtain ⟨u'', j, t, hoc⟩ := houts u _ hx hgu x rfl
    rw [Layout.outDecode_of hoc] at hb
    have := out_copy_keyV2 (si.acc S R u hu) hc ho hv2 hor hM (si.o S R u hu) (si.hvu S R u hu)
      (Fin.cast (congrArg FlockSoundness.Rows.w hrw) x) hoc hb
    simp only [keyRowAddrH, hu, ↓reduceDIte]
    exact this
  · have hr' : keyRowOf c hU n (portRowW c) ((Const.Prog.instC (keyProg c hU n) outs u).wire x) =
        some (u', ⟨q, k⟩) := hk.trans ((rowOfH_of_key outs hk).symm.trans hr)
    obtain ⟨-, hxn, hkey⟩ := Layout.keyRowOf_wire outs x hr'
    have hk' : Aliased.srcKey (si.acc S R u hu).st.c x.val = .msg q.val (k.val / 8) (k.val % 8) := by
      rw [hc]; exact hkey
    have hin' : Typed.msgCol (si.acc S R u hu).st.c (si.vu S R u hu) q.val (k.val / 8) (k.val % 8) <
        2 ^ (si.acc S R u hu).st.c.kLog := by
      have hv := si.hvu S R u hu
      rw [hc] at hv ⊢
      exact Layout.msgCol_lt_of_parse (hparse S R u hu) hv q.isLt _ _ (Nat.mod_lt _ (by norm_num))
    have := copy_siteH (si.acc S R u hu) hM (si.o S R u hu) (si.hvu S R u hu)
      (Fin.cast (congrArg FlockSoundness.Rows.w hrw) x) (lt_of_lt_of_eq hxn (congrArg FlockSoundness.Rows.nIn hrw))
      hk' hin'
    simp only [keyRowAddrH, hu, ↓reduceDIte]
    exact this

include hparse ho hv2 hor houts in
/-- **The single gates at `keyProg`'s sites in hidden-output tables are the constant and the zero**, at
`sha512/row/v2` rows: `keyProgH_valPinned` with `outBitRow_ne_noneV2`. -/
theorem keyProgH_valPinnedV2 (hkeys : KeysRow c hU (portRowW c)) :
    (keyReadersH hparse outs popOf).ValPinned (P := (keyProg c hU n).partition outs) := by
  refine Layout.ofRows_valPinned _ fun S R u hu M hM g hg hr => ?_
  have hpub := Layout.gate_pub (keyPlaceH hparse outs) (constCols_keyProgH hU hparse outs)
    (zeroCols_keyProgH hU hparse outs) hu hM hg
  obtain ⟨x, hx⟩ := Layout.exists_col_of_io ((keyPlaceH hparse outs) S R u hu).inst hg
  change bit _ = Layout.keyGate (c := c) (hU := hU) (n := n) (fun _ _ => false) R g
  obtain ⟨hk, hno⟩ := rowOfH_none outs hr
  rcases Layout.single_gate outs hkeys ⟨x, hx⟩ hg hk with rfl | rfl | ⟨hm, -, -⟩
  · rw [hpub.2 (Finset.mem_singleton_self _)]
    simp only [Layout.keyGate, Layout.zeroG_ne_oneG, ↓reduceIte]
  · rw [hpub.1 (Finset.mem_singleton_self _)]
    simp only [Layout.keyGate, ↓reduceIte]
  · exfalso
    subst hx
    obtain ⟨u'', j, t, hoc⟩ := houts u _ hg hm x rfl
    have hno' : outRowOf c hU n outs ((Const.Prog.instC (keyProg c hU n) outs u).wire x) = none := hno
    rw [outRowOf_wire outs x hm (unit_wire outs x hm).2, Layout.outDecode_of hoc] at hno'
    have hc := (si.acc S R u hu).c_eq (hparse S R u hu)
    have hcr : HmOut.checkRows c = .ok () := hc ▸ (si.acc S R u hu).checkRows
    have P := parse_facts (hparse S R u hu)
    obtain ⟨ls, hls⟩ := P.check
    obtain ⟨-, -, hlo⟩ := Layout.check_out P.typed ho hls
    obtain ⟨hu', hj, ht, -⟩ := hoc
    have hget : c.leavesOut.getD u'' #[] = c.leavesOut[u'']! := by
      rw [Hidden.getElem!_eq_getD']; rfl
    have hwlt : outWordOf c (u'', j, t) < c.outWords := by
      rw [show outWordOf c (u'', j, t) = (c.leavesOut.getD u'' #[])[j] from getElem!_pos _ j hj]
      refine (hlo u'' hu').2 _ ?_
      rw [← hget]
      exact Array.getElem_mem_toList _
    exact outBitRow_ne_noneV2 hcr
      (outRowsW_of_v2 (hparse S R u hu) hcr hor fun p hp => ⟨(hv2 p hp).2.1, (hv2 p hp).2.2.2⟩).1
      (fun p hp => (hv2 p hp).2.2.1) u hwlt ht hno'

/-- **`lay` at `keyProg`'s places in hidden-output tables, at `sha512/row/v2` rows**: the readers' layout with Δ's
copies from `keyProgH_copiesV2`. -/
noncomputable def keyLayoutHV2 :
    Binding.Layout ((keyProg c hU n).partition outs) plan tab
      (((keyProg c hU n).partition outs).placeDecoder plan tab (keyPlaceH hparse outs)) :=
  (keyReadersH hparse outs popOf).layout (keyPlaceH hparse outs) Layout.bitCode
    (keyProgH_copiesV2 hparse outs popOf ho hv2 hor houts)

/-- **`rs` at `keyProg`'s places in hidden-output tables, at `sha512/row/v2` rows**: the readers' rows with
`rowEncP Hm.preV2` at the rows (the bytes `sha512/row/v2` hashes). -/
noncomputable def keyRowsHV2
    (hsite : ∀ S R u (h : u ∈ S) M, Binding.Sat plan tab S R u M →
      BcSiteH (si.acc S R u h) (popOf R) (si.o S R u h) (si.vu S R u h) u.val)
    (hkeys : KeysRow c hU (portRowW c)) :
    Binding.HmRows (keyLayoutHV2 hparse outs popOf ho hv2 hor houts) :=
  (keyReadersH hparse outs popOf).rows (keyPlaceH hparse outs) (Layout.rowEncP Hm.preV2)
    (Layout.rowEncP_inj Hm.preV2_size) Layout.bitCode Layout.bitCode_inj
    (keyProgH_copiesV2 hparse outs popOf ho hv2 hor houts) (keyProgH_bcPinned hparse outs popOf hsite)
    (keyProgH_valPinnedV2 hparse outs popOf ho hv2 hor houts hkeys)

/-- **The bytes `sha512/row/v2` hashes for a row at `keyProg`'s places in hidden-output tables**: `keyRowH_enc` under
`Hm.preV2`. -/
theorem keyRowH_encV2 {S : Finset (Fin n)} {R : Reg} {u : Fin n} (hu : u ∈ S) (M : Binding.Msg plan tab S R u)
    (u' : Fin n) (q : Fin c.ports.size) {words : ℕ} (hrw : portRowW c q = 16 * words) :
    Layout.rowEncP Hm.preV2 ((keyLayoutHV2 hparse outs popOf ho hv2 hor houts).row S R u M (.inl (u', q))) =
      (Hm.preV2 words).data.toList ++
        List.ofFn fun j : Fin (2 * words) => byteOf fun t => bit (Binding.wit plan tab S R u M
          (siteAddrH (si.acc S R u hu) (si.o S R u hu)
            (Typed.msgCol (si.acc S R u hu).st.c (si.vu S R u hu) q.val j.val t.val))) := by
  have hle : portRowW c q ≤ Finset.univ.sup (portRowW c) + 1 :=
    (Finset.le_sup (Finset.mem_univ q)).trans (Nat.le_succ _)
  show Layout.rowEncP (W := Finset.univ.sup (portRowW c) + 1) Hm.preV2
    (Layout.widthOf (Finset.univ.sup (portRowW c) + 1) (portRowW c q),
      Layout.padBits (bitsAt (Binding.wit plan tab S R u M) fun k : Fin (portRowW c q) =>
        keyRowAddrH si S R u q k.val)) = _
  rw [Layout.rowEncP_row Hm.preV2 hrw hle]
  congr 1
  refine List.ext_getElem (by simp) fun j h1 h2 => ?_
  simp only [List.getElem_ofFn]
  congr 1
  funext t
  have hj : (8 * j + t.val) / 8 = j := by omega
  have ht : (8 * j + t.val) % 8 = t.val := by omega
  show bit (Binding.wit plan tab S R u M (keyRowAddrH si S R u q (8 * j + t.val))) = _
  simp only [keyRowAddrH, hu, ↓reduceDIte, hj, ht]

/-- **`HmRowComputes` at `keyProg`'s rows in hidden-output tables, at `sha512/row/v2` rows** (`hHm` for
`keyRowsHV2`): `keyRowsH_computes` from `bc_site_commitHV2`, the row's prefix `Hm.preV2 words` (`Hm.prefixOf_v2`). -/
theorem keyRowsH_computesV2
    (hsite : ∀ S R u (h : u ∈ S) M, Binding.Sat plan tab S R u M →
      BcSiteH (si.acc S R u h) (popOf R) (si.o S R u h) (si.vu S R u h) u.val)
    (hkeys : KeysRow c hU (portRowW c)) :
    (keyRowsHV2 hparse outs popOf ho hv2 hor houts hsite hkeys).Computes := by
  unfold Binding.HmRows.Computes Assumptions.HmRowComputes
  rintro ⟨S, R, u, M, g⟩ ⟨hu, hM, -, hrow⟩
  dsimp only at hu hM hrow ⊢
  revert hrow
  generalize (keyLayoutHV2 hparse outs popOf ho hv2 hor houts).pos g = pos
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
    obtain ⟨hvq, hsq, hbq, -⟩ := hP _ (getElem_mem_toList' hq)
    have hb := bc_site_commitHV2 (si.acc S R u hu) (by rw [hce]; exact ho) hP (by rw [hce]; exact hor) hM
      (si.o S R u hu) (si.hvu S R u hu) hq
    rw [Hm.prefixOf_v2 hvq hsq hbq] at hb
    show Binding.bytesAt _ (keyBcAddrH si S R u q) = Binding.commitString
      (Layout.rowEncP Hm.preV2 ((keyLayoutHV2 hparse outs popOf ho hv2 hor houts).row S R u M (.inl (u', q))))
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
    rw [keyRowH_encV2 hparse outs popOf ho hv2 hor houts hu M u' q hwq, e1, e2]
    exact hb
  · intro h
    exact (Bool.false_ne_true h).elim

end places

end FlockSoundness.Discharge.HiddenExec
