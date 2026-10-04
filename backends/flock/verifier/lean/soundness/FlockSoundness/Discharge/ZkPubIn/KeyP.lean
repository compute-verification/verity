import FlockSoundness.Discharge.ZkPubIn.KeyWire

set_option Elab.async false

/-!
# `lay` and `rs` at `keyProgP`'s places, with the verifier's public inputs

HiddenExec's `keyReadersH`, `keyLayoutH`, `keyRowsH` and `keyRowsH_computes` (`HiddenExec/KeyH.lean`) at
`SitesHP.progPlaces_keyProgP`. The rows are `keyProg`'s (`HiddenExec.rowOfH`: `keyProgP` puts every input that is a
row bit on `keyProg`'s gate for it, and computes `keyProg`'s gates); a public gate is a single gate, and the
registration's bit there is the verifier's own (`Layout.keyGate` at `pubBit`):

* `siteAddrHP`, `keyRowAddrHP`, `keyBcAddrHP`, `keySaltAddrHP`: `siteAddrH`'s addresses at `AcceptedHP` sites;
* `keyReadersHP`: `Layout.Readers.ofRows` at those rows, single gates with `Layout.keyGate (fun _ g => pubBit …)`;
* `keyProgHP_copies`, `keyProgHP_bcPinned`, `keyProgHP_valPinned`: Δ's copies, the registered `b ‖ c`, and the single
  gates' values: the constant, the zero, and at a public gate the `in` region's bit, the verifier's own word's
  (`OwnWords`, at the unit's rank in the draw);
* `keyLayoutHP`, `keyRowsHP`, `keyRowsHP_computes`.
-/

namespace FlockSoundness.Discharge.ZkPubIn

open Flock FlockLevel3 Model Audit Binding Placed HiddenExec
open FlockSoundness.Discharge.Layout (keyN keyRowOf portRowW OutsOut KeysRow)

variable {c : Flock.Circuit} {hU : Built c.unit} {n : ℕ}

variable {T : Model.Statement}

section places

variable {Reg : Type} {mPts : ℕ} {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
  (si : SitesHP plan tab)

/-- **Bit `k` of row `q` at a drawn unit's site** (`keyRowAddrH`). -/
noncomputable def keyRowAddrHP (S : Finset (Fin n)) (R : Reg) (u : Fin n) (q : Fin c.ports.size) (k : ℕ) :
    Fin (2 ^ (Partition.tabOf plan tab S R u).S.m) :=
  if h : u ∈ S then
    siteAddrHP (si.acc S R u h) (si.o S R u h)
      (Typed.msgCol (si.acc S R u h).st.c (si.vu S R u h) q.val (k / 8) (k % 8))
  else ⟨0, Nat.two_pow_pos _⟩

/-- **Row `q`'s `b ‖ c` at a drawn unit's site** (`keyBcAddrH`). -/
noncomputable def keyBcAddrHP (S : Finset (Fin n)) (R : Reg) (u : Fin n) (q : Fin c.ports.size) (j : Fin 128)
    (t : Fin 8) : Fin (2 ^ (Partition.tabOf plan tab S R u).S.m) :=
  if h : u ∈ S then
    siteAddrHP (si.acc S R u h) (si.o S R u h)
      (Layout.bcAddr (si.acc S R u h).st.c (si.vu S R u h) q.val j.val t.val)
  else ⟨0, Nat.two_pow_pos _⟩

/-- **Row `q`'s salt at a drawn unit's site** (`keySaltAddrH`). -/
noncomputable def keySaltAddrHP (S : Finset (Fin n)) (R : Reg) (u : Fin n)
    (q : (Fin n × Fin c.ports.size) ⊕ Fin (keyN c hU n)) (j : Fin 192) (t : Fin 8) :
    Fin (2 ^ (Partition.tabOf plan tab S R u).S.m) :=
  if h : u ∈ S then
    match q with
    | .inl (_, q) =>
      siteAddrHP (si.acc S R u h) (si.o S R u h) (Hm.saltCol (si.acc S R u h).st.c (si.vu S R u h) q.val j.val t.val)
    | .inr _ => ⟨0, Nat.two_pow_pos _⟩
  else ⟨0, Nat.two_pow_pos _⟩

variable {si}
variable (hparse : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S),
    HmRow.parse (si.acc S R u h).tags (si.acc S R u h).cf (si.acc S R u h).tables = .ok c)
  (pi : HmIn.Ports) (hpi : HmIn.parse c = .ok pi) (own : ByteArray) (outs : Finset (Fin (keyN c hU n)))

/-- `SitesHP.progPlaces_keyProgP`'s places. -/
noncomputable abbrev keyPlaceHP := (SitesHP.progPlaces_keyProgP hU hparse pi hpi outs).derived.place

variable (popOf : Reg → Public)

/-- **The readers at `keyProgP`'s sites**: `keyReadersH`'s rows (`rowOfH`), addresses at the `AcceptedHP` sites, the
registration's `b ‖ c` the population's public file's (`popOf R`), and single gates with `Layout.keyGate` at the
public gates' bits (`pubBit`, the verifier's own words). -/
noncomputable def keyReadersHP : Layout.Readers ((keyProgP c hU n pi).circuit outs) plan tab :=
  Layout.Readers.ofRows (keyPlaceHP hparse pi hpi outs) (Layout.keyN_pos c hU n) (portRowW c)
    (W := Finset.univ.sup (portRowW c) + 1)
    (fun q => (Finset.le_sup (Finset.mem_univ q)).trans (Nat.le_succ _)) (Nat.le_add_left 1 _) (rowOfH c hU n outs)
    (keyRowAddrHP si) (keySaltAddrHP si (hU := hU)) (keyBcAddrHP si)
    (fun R u q => ((popOf R).digest u.val q.val c.ports.size).data.toList)
    (Layout.keyGate (c := c) (hU := hU) (n := n) fun _ g => (pubBit c hU n pi own g).getD false)

variable (ho : c.outNet = c.unitNet) (hv1 : ∀ p ∈ c.ports.toList, p.v2 = false ∧ p.bits = 16 * p.words ∧ 0 < p.chunks)
  (houts : OutsOut c hU n outs)

include hparse ho hv1 houts in
/-- **Δ's copies at `keyProgP`'s sites** (`keyProgH_copies`): a column on an input row's bit holds the witness's bit
at its key's message column; a column on an output row's bit, the bit at the output row's message column. -/
theorem keyProgHP_copies : (keyReadersHP hparse pi hpi own outs popOf).Copies (keyPlaceHP hparse pi hpi outs) := by
  refine Layout.ofRows_copies _ fun S R u hu M hM x hx u' q k hr => ?_
  have hc := (si.acc S R u hu).c_eq (hparse S R u hu)
  have hrw := SitesHP.rowsOf_keyProgP hU hparse pi S R u hu
  rcases hk : keyRowOf c hU n (portRowW c) ((Const.Prog.instC (keyProgP c hU n pi) outs u).wire x) with _ | r
  · have ho' : outRowOf c hU n outs ((Const.Prog.instC (keyProgP c hU n pi) outs u).wire x) = some (u', ⟨q, k⟩) :=
      (rowOfH_of_none outs hk).symm.trans hr
    obtain ⟨hgu, -, hb⟩ := outRowOf_wire_someP outs x ho'
    obtain ⟨u'', j, t, hoc⟩ := outsOut_keyProgP houts u _ hx hgu x rfl
    rw [Layout.outDecode_of hoc] at hb
    have := out_copy_keyP (si.acc S R u hu) hc ho hv1 hM (si.o S R u hu) (si.hvu S R u hu)
      (Fin.cast (congrArg FlockSoundness.Rows.w hrw) x) hoc hb
    simp only [keyRowAddrHP, hu, ↓reduceDIte]
    exact this
  · have hr' : keyRowOf c hU n (portRowW c) ((Const.Prog.instC (keyProgP c hU n pi) outs u).wire x) =
        some (u', ⟨q, k⟩) := hk.trans ((rowOfH_of_key outs hk).symm.trans hr)
    obtain ⟨-, hxn, hkey⟩ := keyRowOf_wireP outs x hr'
    have hpi' : HmIn.parse (si.acc S R u hu).st.c = .ok pi := by rw [hc]; exact hpi
    have hk' : Aliased.srcKey (si.acc S R u hu).st.c x.val = .msg q.val (k.val / 8) (k.val % 8) := by
      rw [hc]; exact hkey
    have hpub : pubLeaf (si.acc S R u hu).st.c pi x.val = none := by
      rcases hp : pubLeaf (si.acc S R u hu).st.c pi x.val with _ | v
      · rfl
      · have := srcKey_of_pubLeaf ((si.acc S R u hu).leavesPublic hpi') hp
        rw [hk'] at this
        cases this
    have hin' : Typed.msgCol (si.acc S R u hu).st.c (si.vu S R u hu) q.val (k.val / 8) (k.val % 8) <
        2 ^ (si.acc S R u hu).st.c.kLog := by
      have hv := si.hvu S R u hu
      rw [hc] at hv ⊢
      exact Layout.msgCol_lt_of_parse (hparse S R u hu) hv q.isLt _ _ (Nat.mod_lt _ (by norm_num))
    have := copy_siteHP (si.acc S R u hu) hpi' hM (si.o S R u hu) (si.hvu S R u hu)
      (Fin.cast (congrArg FlockSoundness.Rows.w hrw) x) (lt_of_lt_of_eq hxn (congrArg FlockSoundness.Rows.nIn hrw))
      hpub hk' hin'
    simp only [keyRowAddrHP, hu, ↓reduceDIte]
    exact this

include hparse in
/-- **The rows' `b ‖ c` at `keyProgP`'s sites is the registration's** (`keyProgH_bcPinned`), at every site whose
rank is the unit (`BcSiteHP`). -/
theorem keyProgHP_bcPinned
    (hsite : ∀ S R u (h : u ∈ S) M, Binding.Sat plan tab S R u M →
      BcSiteHP (si.acc S R u h) pi (popOf R) (si.o S R u h) (si.vu S R u h) u.val) :
    (keyReadersHP hparse pi hpi own outs popOf).BcPinned (P := (keyProgP c hU n pi).partition outs) := by
  refine Layout.ofRows_bcPinned _ fun S R u hu M hM g hg u' q k hr => ?_
  obtain ⟨x, rfl⟩ := Layout.exists_col_of_io ((keyPlaceHP hparse pi hpi outs) S R u hu).inst hg
  rw [rowOfH_wire_unitP outs x hr]
  have hc := (si.acc S R u hu).c_eq (hparse S R u hu)
  have hnp : (si.acc S R u hu).st.c.ports.size = c.ports.size := by rw [hc]
  have := bc_site_ofHP (hsite S R u hu M hM) hM (si.hvu S R u hu) (p := q.val) (by rw [hnp]; exact q.isLt)
  rw [hnp] at this
  unfold keyBcAddrHP
  simp only [hu, ↓reduceDIte]
  exact this

include hparse ho hv1 houts in
/-- **The single gates at `keyProgP`'s sites hold the registration's bits** (`keyProgH_valPinned`): the constant 1,
the zero 0, and a public gate the bit of the `in` region at its column (`HmIn.regions`), which is the verifier's own
word's at the unit's instance (`OwnWords`, at the unit's rank in the draw, `BcSiteHP`). -/
theorem keyProgHP_valPinned (hkeys : KeysRow c hU (portRowW c))
    (hsite : ∀ S R u (h : u ∈ S) M, Binding.Sat plan tab S R u M →
      BcSiteHP (si.acc S R u h) pi (popOf R) (si.o S R u h) (si.vu S R u h) u.val) :
    (keyReadersHP hparse pi hpi own outs popOf).ValPinned (P := (keyProgP c hU n pi).partition outs) := by
  sorry

/-- **`lay` at `keyProgP`'s places** (`keyLayoutH`). -/
noncomputable def keyLayoutHP :
    Binding.Layout ((keyProgP c hU n pi).partition outs) plan tab
      (((keyProgP c hU n pi).partition outs).placeDecoder plan tab (keyPlaceHP hparse pi hpi outs)) :=
  (keyReadersHP hparse pi hpi own outs popOf).layout (keyPlaceHP hparse pi hpi outs) Layout.bitCode
    (keyProgHP_copies hparse pi hpi own outs popOf ho hv1 houts)

/-- **`rs` at `keyProgP`'s places** (`keyRowsH`). -/
noncomputable def keyRowsHP
    (hsite : ∀ S R u (h : u ∈ S) M, Binding.Sat plan tab S R u M →
      BcSiteHP (si.acc S R u h) pi (popOf R) (si.o S R u h) (si.vu S R u h) u.val)
    (hkeys : KeysRow c hU (portRowW c)) :
    Binding.HmRows (keyLayoutHP hparse pi hpi own outs popOf ho hv1 houts) :=
  (keyReadersHP hparse pi hpi own outs popOf).rows (keyPlaceHP hparse pi hpi outs) Layout.rowEnc Layout.rowEnc_inj
    Layout.bitCode Layout.bitCode_inj (keyProgHP_copies hparse pi hpi own outs popOf ho hv1 houts)
    (keyProgHP_bcPinned hparse pi hpi own outs popOf hsite)
    (keyProgHP_valPinned hparse pi hpi own outs popOf ho hv1 houts hkeys hsite)

/-- **The bytes `sha512/row/v1` hashes for a row at `keyProgP`'s places** (`keyRowH_enc`). -/
theorem keyRowHP_enc {S : Finset (Fin n)} {R : Reg} {u : Fin n} (hu : u ∈ S) (M : Binding.Msg plan tab S R u)
    (u' : Fin n) (q : Fin c.ports.size) {words : ℕ} (hrw : portRowW c q = 16 * words) :
    Layout.rowEnc ((keyLayoutHP hparse pi hpi own outs popOf ho hv1 houts).row S R u M (.inl (u', q))) =
      (HmRow.rowPrefix words).data.toList ++
        List.ofFn fun j : Fin (2 * words) => byteOf fun t => bit (Binding.wit plan tab S R u M
          (siteAddrHP (si.acc S R u hu) (si.o S R u hu)
            (Typed.msgCol (si.acc S R u hu).st.c (si.vu S R u hu) q.val j.val t.val))) := by
  have hle : portRowW c q ≤ Finset.univ.sup (portRowW c) + 1 :=
    (Finset.le_sup (Finset.mem_univ q)).trans (Nat.le_succ _)
  show Layout.rowEnc (W := Finset.univ.sup (portRowW c) + 1)
    (Layout.widthOf (Finset.univ.sup (portRowW c) + 1) (portRowW c q),
      Layout.padBits (bitsAt (Binding.wit plan tab S R u M) fun k : Fin (portRowW c q) =>
        keyRowAddrHP si S R u q k.val)) = _
  rw [Layout.rowEnc_row hrw hle]
  congr 1
  refine List.ext_getElem (by simp) fun j h1 h2 => ?_
  simp only [List.getElem_ofFn]
  congr 1
  funext t
  have hj : (8 * j + t.val) / 8 = j := by omega
  have ht : (8 * j + t.val) % 8 = t.val := by omega
  show bit (Binding.wit plan tab S R u M (keyRowAddrHP si S R u q (8 * j + t.val))) = _
  simp only [keyRowAddrHP, hu, ↓reduceDIte, hj, ht]

/-- **`HmRowComputes` at `keyProgP`'s rows** (`keyRowsH_computes`). -/
theorem keyRowsHP_computes
    (hsite : ∀ S R u (h : u ∈ S) M, Binding.Sat plan tab S R u M →
      BcSiteHP (si.acc S R u h) pi (popOf R) (si.o S R u h) (si.vu S R u h) u.val)
    (hkeys : KeysRow c hU (portRowW c)) :
    (keyRowsHP hparse pi hpi own outs popOf ho hv1 houts hsite hkeys).Computes := by
  unfold Binding.HmRows.Computes Assumptions.HmRowComputes
  rintro ⟨S, R, u, M, g⟩ ⟨hu, hM, -, hrow⟩
  dsimp only at hu hM hrow ⊢
  revert hrow
  generalize (keyLayoutHP hparse pi hpi own outs popOf ho hv1 houts).pos g = pos
  rcases pos with ⟨u', q⟩ | g'
  · intro _
    have hce := (si.acc S R u hu).c_eq (hparse S R u hu)
    have hq : q.val < (si.acc S R u hu).st.c.ports.size := by rw [hce]; exact q.isLt
    have hwq : portRowW c q = 16 * (si.acc S R u hu).st.c.ports[q.val]!.words := by
      rw [hce, getElem!_pos c.ports q.val q.isLt]
      rfl
    have hb := bc_site_commitHP (si.acc S R u hu) (by rw [hce]; exact ho) (by rw [hce]; exact hv1) hM
      (si.o S R u hu) (si.hvu S R u hu) hq
    show Binding.bytesAt _ (keyBcAddrHP si S R u q) = Binding.commitString
      (Layout.rowEnc ((keyLayoutHP hparse pi hpi own outs popOf ho hv1 houts).row S R u M (.inl (u', q))))
      (Binding.bytesAt _ (keySaltAddrHP si S R u (.inl (u', q)) (hU := hU)))
    have e1 : keyBcAddrHP si S R u q = fun (j : Fin 128) (t : Fin 8) => siteAddrHP (si.acc S R u hu) (si.o S R u hu)
        (Layout.bcAddr (si.acc S R u hu).st.c (si.vu S R u hu) q.val j.val t.val) := by
      funext j t
      simp only [keyBcAddrHP, hu, dite_true]
    have e2 : keySaltAddrHP si S R u (.inl (u', q)) (hU := hU) = fun (j : Fin 192) (t : Fin 8) =>
        siteAddrHP (si.acc S R u hu) (si.o S R u hu)
          (Hm.saltCol (si.acc S R u hu).st.c (si.vu S R u hu) q.val j.val t.val) := by
      funext j t
      simp only [keySaltAddrHP, hu, dite_true]
    rw [keyRowHP_enc hparse pi hpi own outs popOf ho hv1 houts hu M u' q hwq, e1, e2]
    exact hb
  · intro h
    exact (Bool.false_ne_true h).elim

end places

end FlockSoundness.Discharge.ZkPubIn
