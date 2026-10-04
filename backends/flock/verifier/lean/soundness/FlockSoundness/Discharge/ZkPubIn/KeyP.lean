import FlockSoundness.Discharge.ZkPubIn.KeyWire
import FlockSoundness.Discharge.ZkPubIn.InPin

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
  (`OwnWords`, at the unit's rank in the draw; `pub_siteHP`), at sites whose acceptances check the readers' own words
  (`hown`);
* `AcceptedHP.zero_blockP`, `single_gateP`, `constCols_keyProgHP`, `zeroCols_keyProgHP`: the zero input's 0, the four
  kinds of single gate, and `hConst`, `hZero` at `keyProgP`'s places;
* `keyLayoutHP`, `keyRowsHP`, `keyRowsHP_computes`.
-/

namespace FlockSoundness.Discharge.ZkPubIn

open Flock FlockLevel3 Model Audit Binding Placed HiddenExec
open FlockSoundness.Discharge.Layout (keyN keyRowOf portRowW OutsOut KeysRow)

variable {c : Flock.Circuit} {hU : Built c.unit} {n : ℕ}

variable {T : Model.Statement}

/-- **A VU input that is no public leaf's bit and reads the zero carries 0** (`AcceptedH.zero_block`), at the VU's
columns, in every block of a satisfying witness (`copyPos_blockHP`). -/
theorem AcceptedHP.zero_blockP (a : AcceptedHP T) {pi : HmIn.Ports} (hpi : HmIn.parse a.st.c = .ok pi) {g : ℕ}
    (hg : g < a.st.c.g) (x : Fin a.rows.w) (hx : x.val < a.rows.nIn) (hpl : pubLeaf a.st.c pi x.val = none)
    (hz : Aliased.srcKey a.st.c x.val = .zero) {z : Model.Witness T.m} (hsat : T.Satisfies z) (o : Fin (2 ^ T.nbl)) :
    T.block z o (a.col hg x) = 0 := by
  have lp := a.leavesPublic hpi
  obtain ⟨tags, cf, pf, coins, tables, draw, partition, program, own, st, ht, h, L, hR, regs, hT⟩ := a
  subst hT
  have hx'' : x.val < st.c.upv * (st.c.unit.inWords * WORD) := by
    have : x.val < st.c.upv * (unitRows st L.unit_built).nIn := hx
    rwa [L.unitRows_nIn] at this
  have Lf := flat_layoutP ht h
  have F := flat_unitFactsP ht h
  obtain ⟨s, hs, he⟩ := Aliased.copyPos_zero Lf regs F hg hx'' hz
  have e := copyPos_blockHP L Lf regs F hR hpi lp hg x hx hpl hsat o
  rw [he, zeroPos_blockH L Lf regs F hs hsat o] at e
  exact e

section gates

variable {pi : HmIn.Ports}

/-- `KeysRow` at a unit of `keyProgP`, whose rows are the circuit's (`Layout.keysRow_keyProg`). -/
theorem keysRow_keyProgP {rowW : Fin c.ports.size → ℕ} (hkeys : KeysRow c hU rowW) (u : Fin n)
    (i : Fin ((keyProgP c hU n pi).rows u).nComp) (x : Fin ((keyProgP c hU n pi).rows u).w)
    (hx : x ∈ ((keyProgP c hU n pi).rows u).a i ++ ((keyProgP c hU n pi).rows u).b i)
    (hxn : x.val < ((keyProgP c hU n pi).rows u).nIn) : Layout.RowKey c rowW x.val := by
  have e : (keyProgP c hU n pi).rows u = Placed.circuitRows c hU := Placed.rows_circuitProg c hU n _ _ u
  revert i x
  generalize (keyProgP c hU n pi).rows u = R at e ⊢
  subst e
  exact hkeys

/-- **A single committed gate of a unit's own instance of `keyProgP`** (`Layout.single_gate`) is the zero, the constant,
one of the unit's computed gates, or an input's public gate `2 + u · B + x` at a public leaf's bit `x` (`pubLeaf`), when
every input a row reads has a row key (`KeysRow`): any other input's gate is a Δ key's, read by a row, so a row bit. -/
theorem single_gateP (outs : Finset (Fin (keyN c hU n))) {rowW : Fin c.ports.size → ℕ} (hkeys : KeysRow c hU rowW)
    {u : Fin n} (x : Fin ((keyProgP c hU n pi).rows u).w)
    (hg : (Const.Prog.instC (keyProgP c hU n pi) outs u).wire x ∈ ((keyProgP c hU n pi).partition outs).io u)
    (hr : keyRowOf c hU n rowW ((Const.Prog.instC (keyProgP c hU n pi) outs u).wire x) = none) :
    (Const.Prog.instC (keyProgP c hU n pi) outs u).wire x = Layout.zeroG ∨
      (Const.Prog.instC (keyProgP c hU n pi) outs u).wire x = Layout.oneG ∨
      ((keyProgP c hU n pi).unit ((Const.Prog.instC (keyProgP c hU n pi) outs u).wire x) = some u ∧
        ((keyProgP c hU n pi).rows u).nIn ≤ x.val) ∨
      (x.val < Layout.keyB c hU ∧ (pubLeaf c pi x.val).isSome ∧
        ((Const.Prog.instC (keyProgP c hU n pi) outs u).wire x).val = 2 + u.val * Layout.keyB c hU + x.val) := by
  by_cases hx : x.val < ((keyProgP c hU n pi).rows u).nIn
  · have hxB : x.val < Layout.keyB c hU := lt_of_lt_of_eq hx (nIn_keyProgP u)
    have hv : (Const.Prog.insAt (keyProgP c hU n pi) u x.val).val =
        (Placed.insOf n (Layout.keyB c hU) (keyIdxP c pi (Layout.keyB c hU)) u.val ⟨x.val, hxB⟩).val :=
      Placed.insAt_uniform n u x.val hxB
    have hw : ((Const.Prog.instC (keyProgP c hU n pi) outs u).wire x).val =
        (Placed.srcGate n (Layout.keyB c hU) u.isLt (keyIdxP c pi (Layout.keyB c hU) ⟨x.val, hxB⟩)).val := by
      rw [Const.Prog.instC_wire_input _ _ _ x hx, hv,
        Placed.insOf_of_lt (nU := n) (B := Layout.keyB c hU) (src := keyIdxP c pi (Layout.keyB c hU)) u.isLt]
    rcases hb : keyIdxP c pi (Layout.keyB c hU) ⟨x.val, hxB⟩ with _ | b
    · rw [hb] at hw
      exact .inl (Fin.ext hw)
    · rw [hb] at hw
      simp only [Placed.srcGate] at hw
      by_cases hp : (pubLeaf c pi x.val).isSome
      · have hself : keyIdxP c pi (Layout.keyB c hU) ⟨x.val, hxB⟩ = some ⟨x.val, hxB⟩ := by
          unfold keyIdxP
          simp [hp]
        rw [hself] at hb
        obtain rfl := Option.some.inj hb
        exact .inr (.inr (.inr ⟨hxB, hp, hw⟩))
      · exfalso
        have hpl : pubLeaf c pi x.val = none := Option.not_isSome_iff_eq_none.1 hp
        have hnz : Aliased.srcKey c x.val ≠ .zero := fun h => by
          rw [((keyIdxP_none c pi _ ⟨x.val, hxB⟩ hpl).1).2 h] at hb
          cases hb
        have hkb := keyIdxP_some hb
        have hm : (2 + u.val * Layout.keyB c hU + b.val - 2) % Layout.keyB c hU = b.val := by
          rw [show 2 + u.val * Layout.keyB c hU + b.val - 2 = b.val + u.val * Layout.keyB c hU by omega,
            Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt b.isLt]
        have hnone : ((keyProgP c hU n pi).partition outs).unit
            ((Const.Prog.instC (keyProgP c hU n pi) outs u).wire x) = none :=
          keyProgP_reads outs u x fun i e => by
            have := congrArg Fin.val e
            simp only [FlockSoundness.Rows.comp] at this
            omega
        obtain ⟨i, c', hc', hwc⟩ := Layout.exists_read_of_io (Const.Prog.instC (keyProgP c hU n pi) outs u) hg
          (fun h => by rw [hnone] at h; cases h)
        have hc'n : c'.val < ((keyProgP c hU n pi).rows u).nIn := by
          rcases (Const.Prog.instC (keyProgP c hU n pi) outs u).shared c' x hwc with rfl | ⟨h, -⟩
          · exact hx
          · exact h
        have hkey' : Aliased.srcKey c c'.val = Aliased.srcKey c x.val := by
          rcases keyProgP_wire c hU pi outs u c' x hc'n hx hwc with he | ⟨-, -, he⟩
          · rw [he]
          · exact he
        rcases keysRow_keyProgP hkeys u i c' hc' hc'n with h0 | ⟨q, byte, bt, hk, hq, hlw, hlb⟩
        · exact hnz (hkey'.symm.trans h0)
        · refine absurd hr (Layout.keyRowOf_ne_none (by rw [hw]; omega)
            (by rw [hw]; exact Placed.source_lt n _ u.isLt b) (q := q) (byte := byte) (bit := bt) ?_ hq hlw hlb)
          rw [hw, hm, hkb, ← hkey']
          exact hk
  · by_cases hone : x = ((keyProgP c hU n pi).rows u).one
    · refine .inr (.inl ?_)
      rw [hone, Const.Prog.instC_wire_one]
      exact Fin.ext (Layout.oneAt_uniform_val (k := Placed.inGates n (Layout.keyB c hU)) (R := Placed.circuitRows c hU)
        (ins := Placed.insOf n (Layout.keyB c hU) (keyIdxP c pi (Layout.keyB c hU)))
        (one := Placed.oneGate n (Layout.keyB c hU))
        (hone := Placed.insOf_ne_one n (Layout.keyB c hU) (keyIdxP c pi (Layout.keyB c hU))) n u)
    · have hne : x.val ≠ ((keyProgP c hU n pi).rows u).nIn + ((keyProgP c hU n pi).rows u).nComp := fun he =>
        hone (Fin.ext he)
      have hxw := x.isLt
      have hi : x.val - ((keyProgP c hU n pi).rows u).nIn < ((keyProgP c hU n pi).rows u).nComp := by
        change x.val < ((keyProgP c hU n pi).rows u).nIn + ((keyProgP c hU n pi).rows u).nComp + 1 at hxw
        omega
      have hx' : x = ((keyProgP c hU n pi).rows u).comp ⟨_, hi⟩ :=
        Fin.ext (by simp only [FlockSoundness.Rows.comp]; omega)
      have hm : (keyProgP c hU n pi).unit ((Const.Prog.instC (keyProgP c hU n pi) outs u).wire x) = some u := by
        have := (Const.Prog.instC (keyProgP c hU n pi) outs u).mem ⟨_, hi⟩
        rw [← hx'] at this
        exact this
      exact .inr (.inr (.inl ⟨hm, by omega⟩))

end gates

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

/-- **`hConst` at `keyProgP`'s places** (`constCols_keyProgH`): only a unit's constant column sits on the one gate. -/
theorem constCols_keyProgHP :
    ((keyProgP c hU n pi).partition outs).ConstCols plan tab (keyPlaceHP hparse pi hpi outs)
      {Layout.oneG (c := c) (hU := hU) (n := n)} :=
  Const.constCols_of_instC (SitesHP.progPlaces_keyProgP hU hparse pi hpi outs) (Zero.oneGate_uniform n) fun _ => rfl

/-- **`hZero` at `keyProgP`'s places** (`zeroCols_keyProgH`): an input on the zero gate has no `keyIdxP` gate, so it is
no public leaf's bit and reads the zero (`keyIdxP_none`), and carries 0 (`AcceptedHP.zero_blockP`). -/
theorem zeroCols_keyProgHP :
    ((keyProgP c hU n pi).partition outs).ZeroCols plan tab (keyPlaceHP hparse pi hpi outs)
      {Layout.zeroG (c := c) (hU := hU) (n := n)} := by
  refine Zero.Prog.zeroCols_singleton (SitesHP.progPlaces_keyProgP hU hparse pi hpi outs)
    (one := Layout.oneG (c := c) (hU := hU) (n := n))
    (Zero.unit_uniform n _ (Placed.zeroGate n (Layout.keyB c hU)).isLt) Layout.zeroG_ne_oneG (fun u => ?_) ?_
  · show (Const.Prog.instC (keyProgP c hU n pi) outs u).wire _ = _
    rw [Const.Prog.instC_wire_one]
    exact Zero.oneGate_uniform n u
  · intro S R u h x hx hw z hsat
    change (Const.Prog.instC (keyProgP c hU n pi) outs u).wire x = _ at hw
    rw [Const.Prog.instC_wire_input _ _ _ x hx] at hw
    have hxB : x.val < Layout.keyB c hU := lt_of_lt_of_eq hx (nIn_keyProgP u)
    have hv := congrArg Fin.val hw
    have hu' : (Const.Prog.insAt (keyProgP c hU n pi) u x.val).val =
        (Placed.insOf n (Layout.keyB c hU) (keyIdxP c pi (Layout.keyB c hU)) u.val ⟨x.val, hxB⟩).val :=
      Placed.insAt_uniform n u x.val hxB
    rw [hu'] at hv
    have hnone : keyIdxP c pi (Layout.keyB c hU) ⟨x.val, hxB⟩ = none :=
      (Placed.insOf_eq_zero_iff u.isLt _).1 (Fin.ext hv)
    have hpl : pubLeaf c pi x.val = none := by
      rcases hp : pubLeaf c pi x.val with _ | v
      · rfl
      · unfold keyIdxP at hnone
        simp [hp] at hnone
    have hz := ((keyIdxP_none c pi _ ⟨x.val, hxB⟩ hpl).1).1 hnone
    have hc := (si.acc S R u h).c_eq (hparse S R u h)
    have hpi' : HmIn.parse (si.acc S R u h).st.c = .ok pi := by rw [hc]; exact hpi
    have hrw := SitesHP.rowsOf_keyProgP hU hparse pi S R u h
    exact (si.acc S R u h).zero_blockP hpi' (si.hvu S R u h) (Fin.cast (congrArg FlockSoundness.Rows.w hrw) x)
      (lt_of_lt_of_eq hx (congrArg FlockSoundness.Rows.nIn hrw)) (by rw [hc]; exact hpl) (by rw [hc]; exact hz) hsat _

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
      BcSiteHP (si.acc S R u h) pi (popOf R) (si.o S R u h) (si.vu S R u h) u.val)
    (hown : ∀ S R u (h : u ∈ S), (si.acc S R u h).own = own) :
    (keyReadersHP hparse pi hpi own outs popOf).ValPinned (P := (keyProgP c hU n pi).partition outs) := by
  refine Layout.ofRows_valPinned _ fun S R u hu M hM g hg hr => ?_
  have hpub := Layout.gate_pub (keyPlaceHP hparse pi hpi outs) (constCols_keyProgHP hparse pi hpi outs)
    (zeroCols_keyProgHP hparse pi hpi outs) hu hM hg
  obtain ⟨x, hx⟩ := Layout.exists_col_of_io ((keyPlaceHP hparse pi hpi outs) S R u hu).inst hg
  change bit _ = Layout.keyGate (c := c) (hU := hU) (n := n) (fun _ g => (pubBit c hU n pi own g).getD false) R g
  obtain ⟨hk, hno⟩ := rowOfH_none outs hr
  have hx' : (Const.Prog.instC (keyProgP c hU n pi) outs u).wire x = g := hx
  subst hx'
  rcases single_gateP outs hkeys x hg hk with h0 | h1 | ⟨hm, hxn⟩ | ⟨hxB, hp, hwv⟩
  · rw [hpub.2 (by rw [h0]; exact Finset.mem_singleton_self _), h0]
    simp only [Layout.keyGate, Layout.zeroG_ne_oneG, ↓reduceIte]
  · rw [hpub.1 (by rw [h1]; exact Finset.mem_singleton_self _), h1]
    simp only [Layout.keyGate, ↓reduceIte]
  · exfalso
    obtain ⟨u'', j, t, hoc⟩ := outsOut_keyProgP houts u _ hg hm x rfl
    have hno' : outRowOf c hU n outs ((Const.Prog.instC (keyProgP c hU n pi) outs u).wire x) = none := hno
    rw [outRowOf_wireP outs x hm hxn, Layout.outDecode_of hoc] at hno'
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
    exact outBitRow_ne_none hcr (Hidden.out_rows_v1 (hparse S R u hu) hcr fun p hp => (hv1 p hp).1) u hwlt ht hno'
  · have hrw := SitesHP.rowsOf_keyProgP hU hparse pi S R u hu
    have hc := (si.acc S R u hu).c_eq (hparse S R u hu)
    obtain ⟨⟨q, t⟩, hqt⟩ := Option.isSome_iff_exists.1 hp
    have hpl : pubLeaf (si.acc S R u hu).st.c pi x.val = some (q, t) := by rw [hc]; exact hqt
    refine Eq.trans (congrArg bit (Layout.block_gateAt (keyPlaceHP hparse pi hpi outs) hu hM x)).symm ?_
    have hxa : (Fin.cast (congrArg FlockSoundness.Rows.w hrw) x).val < (si.acc S R u hu).rows.nIn :=
      lt_of_lt_of_eq (lt_of_lt_of_eq hxB (nIn_keyProgP u).symm) (congrArg FlockSoundness.Rows.nIn hrw)
    refine Eq.trans (pub_siteHP (si.acc S R u hu) (hsite S R u hu M hM) hM (si.hvu S R u hu)
      (Fin.cast (congrArg FlockSoundness.Rows.w hrw) x) hxa hpl) ?_
    rw [hown S R u hu]
    have hB : 0 < Layout.keyB c hU := lt_of_le_of_lt (Nat.zero_le _) hxB
    have hg2 : 2 ≤ ((Const.Prog.instC (keyProgP c hU n pi) outs u).wire x).val := by rw [hwv]; omega
    have hlt : ((Const.Prog.instC (keyProgP c hU n pi) outs u).wire x).val < 2 + n * Layout.keyB c hU := by
      rw [hwv]; exact Placed.source_lt n _ u.isLt ⟨x.val, hxB⟩
    have hne1 : (Const.Prog.instC (keyProgP c hU n pi) outs u).wire x ≠ Layout.oneG := fun h => by
      have := congrArg Fin.val h
      simp only [Layout.oneG, Fin.val_castLE, Placed.oneGate] at this
      omega
    have hne0 : (Const.Prog.instC (keyProgP c hU n pi) outs u).wire x ≠ Layout.zeroG := fun h => by
      have := congrArg Fin.val h
      simp only [Layout.zeroG, Fin.val_castLE, Placed.zeroGate] at this
      omega
    have hd : (((Const.Prog.instC (keyProgP c hU n pi) outs u).wire x).val - 2) / Layout.keyB c hU = u.val := by
      rw [hwv, show 2 + u.val * Layout.keyB c hU + x.val - 2 = x.val + u.val * Layout.keyB c hU by omega,
        Nat.add_mul_div_right _ _ hB, Nat.div_eq_of_lt hxB, Nat.zero_add]
    have hmo : (((Const.Prog.instC (keyProgP c hU n pi) outs u).wire x).val - 2) % Layout.keyB c hU = x.val := by
      rw [hwv, show 2 + u.val * Layout.keyB c hU + x.val - 2 = x.val + u.val * Layout.keyB c hU by omega,
        Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hxB]
    simp only [Layout.keyGate, pubBit, hg2, hlt, and_self, ↓reduceIte, hmo, hqt, hd, Option.getD_some]
    split_ifs with e1 e0
    · exact absurd e1 hne1
    · exact absurd e0 hne0
    · rfl

include hparse in
/-- **A public gate decodes to the verifier's own bit at `keyProgP`'s places**: at a drawn unit's committed gate with
a value (`pubBit … = some b`), a satisfying message decodes `b`. The gate is an input gate `2 + u · B + x` of the unit
whose source `x` is a public leaf's bit, so `keyIdxP` puts that input alone on it, and the input's column holds the
verifier's own word's bit (`pub_siteHP`, at the site's `BcSiteHP` and own words `hown`). -/
theorem keyPlaceHP_decode_pub
    (hsite : ∀ S R u (h : u ∈ S) M, Binding.Sat plan tab S R u M →
      BcSiteHP (si.acc S R u h) pi (popOf R) (si.o S R u h) (si.vu S R u h) u.val)
    (hown : ∀ S R u (h : u ∈ S), (si.acc S R u h).own = own)
    {S : Finset (Fin n)} {R : Reg} {u : Fin n} (hu : u ∈ S) {M : Binding.Msg plan tab S R u}
    (hM : Binding.Sat plan tab S R u M) {g : Fin (keyN c hU n)} (hg : g ∈ ((keyProgP c hU n pi).partition outs).io u)
    {b : Bool} (hb : pubBit c hU n pi own g = some b) :
    (keyPlaceHP hparse pi hpi outs S R u hu).decode (Binding.wit plan tab S R u M) g = b := by
  obtain ⟨x, hx⟩ := Layout.exists_col_of_io ((keyPlaceHP hparse pi hpi outs) S R u hu).inst hg
  subst hx
  unfold UnitPlace.decode
  rw [((keyPlaceHP hparse pi hpi outs) S R u hu).inst.decode_wire
    (((keyPlaceHP hparse pi hpi outs) S R u hu).agree hM) x]
  have hb' : pubBit c hU n pi own ((Const.Prog.instC (keyProgP c hU n pi) outs u).wire x) = some b := hb
  clear hb
  unfold pubBit at hb'
  split_ifs at hb' with hrange
  rcases hpl : pubLeaf c pi
      ((((Const.Prog.instC (keyProgP c hU n pi) outs u).wire x).val - 2) % Layout.keyB c hU) with _ | ⟨q, t⟩
  · rw [hpl] at hb'
    cases hb'
  rw [hpl] at hb'
  obtain rfl := Option.some.inj hb'
  by_cases hxi : x.val < ((keyProgP c hU n pi).rows u).nIn
  · have hxB : x.val < Layout.keyB c hU := lt_of_lt_of_eq hxi (nIn_keyProgP u)
    have hv : (Const.Prog.insAt (keyProgP c hU n pi) u x.val).val =
        (Placed.insOf n (Layout.keyB c hU) (keyIdxP c pi (Layout.keyB c hU)) u.val ⟨x.val, hxB⟩).val :=
      Placed.insAt_uniform n u x.val hxB
    have hw : ((Const.Prog.instC (keyProgP c hU n pi) outs u).wire x).val =
        (Placed.srcGate n (Layout.keyB c hU) u.isLt (keyIdxP c pi (Layout.keyB c hU) ⟨x.val, hxB⟩)).val := by
      rw [Const.Prog.instC_wire_input _ _ _ x hxi, hv,
        Placed.insOf_of_lt (nU := n) (B := Layout.keyB c hU) (src := keyIdxP c pi (Layout.keyB c hU)) u.isLt]
    rcases hk : keyIdxP c pi (Layout.keyB c hU) ⟨x.val, hxB⟩ with _ | b'
    · rw [hk] at hw
      simp only [Placed.srcGate, Placed.zeroGate] at hw
      omega
    rw [hk] at hw
    simp only [Placed.srcGate] at hw
    have hB : 0 < Layout.keyB c hU := lt_of_le_of_lt (Nat.zero_le _) hxB
    have hd : (((Const.Prog.instC (keyProgP c hU n pi) outs u).wire x).val - 2) / Layout.keyB c hU = u.val := by
      rw [hw, show 2 + u.val * Layout.keyB c hU + b'.val - 2 = b'.val + u.val * Layout.keyB c hU by omega,
        Nat.add_mul_div_right _ _ hB, Nat.div_eq_of_lt b'.isLt, Nat.zero_add]
    have hm : (((Const.Prog.instC (keyProgP c hU n pi) outs u).wire x).val - 2) % Layout.keyB c hU = b'.val := by
      rw [hw, show 2 + u.val * Layout.keyB c hU + b'.val - 2 = b'.val + u.val * Layout.keyB c hU by omega,
        Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt b'.isLt]
    rw [hm] at hpl
    have hbx : b' = ⟨x.val, hxB⟩ := by
      by_cases hp : (pubLeaf c pi x.val).isSome
      · have hself : keyIdxP c pi (Layout.keyB c hU) ⟨x.val, hxB⟩ = some ⟨x.val, hxB⟩ := by
          unfold keyIdxP
          simp [hp]
        rw [hself] at hk
        exact (Option.some.inj hk).symm
      · have := ((keyIdxP_none c pi _ ⟨x.val, hxB⟩ (Option.not_isSome_iff_eq_none.1 hp)).2 b' hk).1
        rw [hpl] at this
        cases this
    subst hbx
    rw [hd, ← hown S R u hu]
    have hrw := SitesHP.rowsOf_keyProgP hU hparse pi S R u hu
    have hc := (si.acc S R u hu).c_eq (hparse S R u hu)
    have hxa : (Fin.cast (congrArg FlockSoundness.Rows.w hrw) x).val < (si.acc S R u hu).rows.nIn :=
      lt_of_lt_of_eq hxi (congrArg FlockSoundness.Rows.nIn hrw)
    exact pub_siteHP (si.acc S R u hu) (hsite S R u hu M hM) hM (si.hvu S R u hu)
      (Fin.cast (congrArg FlockSoundness.Rows.w hrw) x) hxa (by rw [hc]; exact hpl)
  · exfalso
    by_cases hone : x = ((keyProgP c hU n pi).rows u).one
    · subst hone
      have h0 : (Const.Prog.oneAt (keyProgP c hU n pi) u).val = (Placed.oneGate n (Layout.keyB c hU)).val :=
        Layout.oneAt_uniform_val (k := Placed.inGates n (Layout.keyB c hU)) (R := Placed.circuitRows c hU)
          (ins := Placed.insOf n (Layout.keyB c hU) (keyIdxP c pi (Layout.keyB c hU)))
          (one := Placed.oneGate n (Layout.keyB c hU))
          (hone := Placed.insOf_ne_one n (Layout.keyB c hU) (keyIdxP c pi (Layout.keyB c hU))) n u
      have h2 := hrange.1
      rw [Const.Prog.instC_wire_one, h0] at h2
      simp [Placed.oneGate] at h2
    · have hxw := x.isLt
      have hne : x.val ≠ ((keyProgP c hU n pi).rows u).nIn + ((keyProgP c hU n pi).rows u).nComp := fun he =>
        hone (Fin.ext he)
      have hi : x.val - ((keyProgP c hU n pi).rows u).nIn < ((keyProgP c hU n pi).rows u).nComp := by
        change x.val < ((keyProgP c hU n pi).rows u).nIn + ((keyProgP c hU n pi).rows u).nComp + 1 at hxw
        omega
      have hx' : x = ((keyProgP c hU n pi).rows u).comp ⟨_, hi⟩ :=
        Fin.ext (by simp only [FlockSoundness.Rows.comp]; omega)
      have hm : (keyProgP c hU n pi).unit ((Const.Prog.instC (keyProgP c hU n pi) outs u).wire x) = some u := by
        have := (Const.Prog.instC (keyProgP c hU n pi) outs u).mem ⟨_, hi⟩
        rw [← hx'] at this
        exact this
      have hnone : (keyProgP c hU n pi).unit ((Const.Prog.instC (keyProgP c hU n pi) outs u).wire x) = none :=
        Layout.unit_uniform_input (k := Placed.inGates n (Layout.keyB c hU)) (R := Placed.circuitRows c hU)
          (ins := Placed.insOf n (Layout.keyB c hU) (keyIdxP c pi (Layout.keyB c hU)))
          (one := Placed.oneGate n (Layout.keyB c hU))
          (hone := Placed.insOf_ne_one n (Layout.keyB c hU) (keyIdxP c pi (Layout.keyB c hU))) n _ hrange.2
      rw [hnone] at hm
      cases hm

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
    (hkeys : KeysRow c hU (portRowW c)) (hown : ∀ S R u (h : u ∈ S), (si.acc S R u h).own = own) :
    Binding.HmRows (keyLayoutHP hparse pi hpi own outs popOf ho hv1 houts) :=
  (keyReadersHP hparse pi hpi own outs popOf).rows (keyPlaceHP hparse pi hpi outs) Layout.rowEnc Layout.rowEnc_inj
    Layout.bitCode Layout.bitCode_inj (keyProgHP_copies hparse pi hpi own outs popOf ho hv1 houts)
    (keyProgHP_bcPinned hparse pi hpi own outs popOf hsite)
    (keyProgHP_valPinned hparse pi hpi own outs popOf ho hv1 houts hkeys hsite hown)

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
    (hkeys : KeysRow c hU (portRowW c)) (hown : ∀ S R u (h : u ∈ S), (si.acc S R u h).own = own) :
    (keyRowsHP hparse pi hpi own outs popOf ho hv1 houts hsite hkeys hown).Computes := by
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
