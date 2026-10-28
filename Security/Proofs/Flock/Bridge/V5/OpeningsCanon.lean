import Proofs.Flock.Bridge.V5.Statement
import Proofs.Flock.Bridge.OpeningsCanon

/-!
# A drawn unit's openings are canonical at public inputs (`Bridge.OpeningsCanonTargetV5`, proved)

`OpeningsCanon.lean` at `keyProgP`, `rsZHJP` and `constsOfP`, whose constants are the constant, the zero and each
unit's public gates:

* `HiddenExec.rowOfHP_ne_none_of_io`: a committed wire of a drawn unit other than the constant, the zero and a public
  gate has a row (`Layout.single_gateP`; a computed one is a returned output's, `OutsOutP`, so an output row's bit);
* `Bridge.openingsCanonV5_row`: at a wire with a row `(u', q, k)`, the opening is canonical (`canonOpen`) at the row
  the public file's leaf map registers (`LeafMap.ref`), as in `openingsCanonH`;
* `Bridge.openingsCanonV5_const`: at a constant wire, the opening's bit is the decoder's (`opening_spec`): the constant
  column's 1 and a zero column's 0 at the public place (`constCols_keyProgHP`, `zeroCols_keyProgHP`), and at a public
  gate, which has no row (`HiddenExec.rowOfHP_pub`), the bit its place holds (`Layout.block_gateAt`), the public input
  word's bit in the population's public file (`HiddenExec.keyProgHP_gate`, `keyGateP_pub`);
* `Bridge.openingsCanonV5 : Bridge.OpeningsCanonTargetV5`.
-/

namespace FlockSoundness.Discharge.HiddenExec

open Flock FlockLevel3 Model Audit Binding Placed
open FlockSoundness.Discharge.Layout (keyN portRowW OutColAt outDecode OutsOutP KeysRowP keyRowOfP keyPubOf)

variable {c : Flock.Circuit} {hU : Built c.unit} {n : ℕ} {pi : HmIn.Ports} {Reg : Type} {mPts : ℕ}
  {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length} {si : SitesHP plan tab}

/-- **A committed wire of a drawn unit other than the constant, the zero and a public gate has a row** at public
inputs (`rowOfH_ne_none_of_io` at `keyProgP`): a committed gate with no row is the zero, the constant, a public gate or
a computed gate of the unit (`Layout.single_gateP`), and a computed one is a returned output's (`OutsOutP`), so an
output row's bit (`outBitRow_ne_none`). -/
theorem rowOfHP_ne_none_of_io
    (hparse : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S),
      HmRow.parse (si.acc S R u h).tags (si.acc S R u h).cf (si.acc S R u h).tables = .ok c)
    (hpi : HmIn.parse c = .ok pi)
    (outs : Finset (Fin (keyN c hU n))) (ho : c.outNet = c.unitNet) (hrf : Hidden.Any.RowsFitS c)
    (houts : OutsOutP c hU n pi outs) (hkeys : KeysRowP c hU pi (portRowW c)) {S : Finset (Fin n)} {R : Reg}
    {u : Fin n} (hu : u ∈ S) {g : Fin (keyN c hU n)} (hg : g ∈ ((keyProgP c pi hU n).partition outs).io u)
    (h1 : g ≠ Layout.oneG) (h0 : g ≠ Layout.zeroG) (hp : keyPubOf c hU n pi g = none) :
    rowOfHP c hU n pi outs g ≠ none := by
  intro hr
  obtain ⟨x, hx⟩ := Layout.exists_col_of_io ((keyPlaceHP hparse hpi outs) S R u hu).inst hg
  obtain ⟨hk, hno⟩ := rowOfHP_none outs hr
  rcases Layout.single_gateP outs hkeys ⟨x, hx⟩ hg hk with rfl | rfl | ⟨q, t, hpq⟩ | ⟨hm, -, -⟩
  · exact h0 rfl
  · exact h1 rfl
  · rw [hpq] at hp
    cases hp
  subst hx
  obtain ⟨u'', j, t, hoc⟩ := houts u _ hg hm x rfl
  have hno' : outRowOfP c hU n pi outs ((Const.Prog.instC (keyProgP c pi hU n) outs u).wire x) = none := hno
  rw [outRowOfP_wire outs x hm (unit_wireP outs x hm).2, Layout.outDecode_of hoc] at hno'
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
  exact outBitRow_ne_none hcr hrf u hwlt ht hno'

end FlockSoundness.Discharge.HiddenExec

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
    FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden FlockSoundness.Discharge.ZkSession
open FlockSoundness.Discharge.Tables (LiveOutJ)

/-- **At a wire with a row, a drawn unit's opening is canonical** at public inputs (`canonOpen` at the row the public
file's leaf map registers): its commit string is `.inl (u', q)` (`Layout.ofRows_pos_some`), whose value is
`(widthOf W (portRowW q), padBits b)`, canonical by `rowShaped_rowOf`, and whose salt is `saltLen` bytes. -/
theorem _root_.Flock.Bridge.openingsCanonV5_row
    (n : ℕ) (Reg : Type)
    (I : Inputs) (own : Option ByteArray) (ht : I.tags.typed = false)
    (D : Flock.Desc) (hD : Flock.Desc.parse (zkTags I) I.circuitFile I.tables = .ok D) (hU : Built D.unroll.unit)
    (pi : HmIn.Ports) (hpi : HmIn.parse D.unroll = .ok pi)
    (pub : Public) (hpub : HmIn.loadPublic (zkTags I) D.unroll pi I.publicFile = .ok pub)
    (hscope : Discharge.Layout.scopeOkH D.unroll = true)
    (kd : ℕ) (dj : Finset (Fin n) → Lean.Json) (S₀ : Finset (Fin n)) (d₀ : Lean.Json)
    (y₀ : DrawSetupZHJP (inputsJ I 0) own (mPtsOf D.unroll.g (kd / nTab I)) S₀ d₀) :
    let vb := (rsZHJP (Reg := Reg) dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpi hpub).bindingZ
      (rsZHJP_computes dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpi hpub)
    let Cc := (keyProgP D.unroll pi hU n).circuit (Discharge.Layout.outsOfP D.unroll hU n pi)
    let Lm := Bridge.LeafMap.ofPublic D.unroll pub n
    let rows : Fin Cc.N → Option Bridge.RowAt := fun g =>
      (Discharge.HiddenExec.rowOfHP D.unroll hU n pi (Discharge.Layout.outsOfP D.unroll hU n pi) g).map fun r =>
        ⟨r.2.1.val, Lm.ref r.2.1.val r.1.val, Discharge.Layout.portRowW D.unroll r.2.1, r.2.2.val⟩
    ∀ S R u M (g : Fin Cc.N), rows g ≠ none →
      Bridge.canonOpen rows g (vb.val (vb.opening S R u M (vb.pos g))) (vb.opening S R u M (vb.pos g)).2.2 := by
  intro vb Cc Lm rows S R u M g hr
  rcases hx : Discharge.HiddenExec.rowOfHP D.unroll hU n pi (Discharge.Layout.outsOfP D.unroll hU n pi) g
    with _ | ⟨u', q, k⟩
  · exact absurd (by simp [rows, hx] : rows g = none) hr
  have hpos : vb.pos g = Sum.inl (u', q) :=
    Discharge.Layout.ofRows_pos_some
      (Discharge.HiddenExec.keyPlaceHP (sitesShHJP_parse (Reg := Reg) dj y₀ ht (Flock.Desc.parse_flat hD) hscope) hpi
        (Discharge.Layout.outsOfP D.unroll hU n pi))
      (hN := Discharge.Layout.keyN_pos D.unroll hU n) (W := Finset.univ.sup (Discharge.Layout.portRowW D.unroll) + 1)
      (hW := fun q => (Finset.le_sup (Finset.mem_univ q)).trans (Nat.le_succ _)) (h1 := Nat.le_add_left 1 _)
      (rowAddr := fun _ _ _ _ _ => ⟨0, Nat.two_pow_pos _⟩)
      (saltAddr := fun _ _ _ _ _ _ => ⟨0, Nat.two_pow_pos _⟩) (bcAddr := fun _ _ _ _ _ _ => ⟨0, Nat.two_pow_pos _⟩)
      (regBc := fun _ _ _ => []) (regGate := fun _ _ => false) hx
  obtain ⟨b, hb⟩ : ∃ b, vb.val (vb.opening S R u M (Sum.inl (u', q))) =
      (Discharge.Layout.widthOf (Finset.univ.sup (Discharge.Layout.portRowW D.unroll) + 1)
        (Discharge.Layout.portRowW D.unroll q), Discharge.Layout.padBits b) := ⟨_, rfl⟩
  have hlen : (vb.opening S R u M (Sum.inl (u', q))).2.2.length = Bridge.saltLen :=
    Bridge.bytesAt_length _ _
  rw [hpos, hb]
  refine ⟨⟨⟨q.val, Lm.ref q.val u'.val, Discharge.Layout.portRowW D.unroll q, k.val⟩, by simp [rows, hx], ?_⟩, hlen⟩
  exact Bridge.rowShaped_rowOf rfl
    ((Finset.le_sup (f := Discharge.Layout.portRowW D.unroll) (Finset.mem_univ q)).trans (Nat.le_succ _)) b

/-- **At a constant wire, a drawn unit's opening carries the constant** at public inputs: the opening's bit is the
decoder's (`opening_spec`), which at the public place reads the unit's constant column, `1` (`UnitPlace.decode_one`,
`constCols_keyProgHP`), a zero column, `0` (`zeroCols_keyProgHP`), or, at a public gate, the bit its place holds
(`Layout.block_gateAt`), which is its public input word's bit in the population's public file
(`HiddenExec.keyProgHP_gate` at a gate with no row, `HiddenExec.rowOfHP_pub`, and `keyGateP_pub`). -/
theorem _root_.Flock.Bridge.openingsCanonV5_const
    (n : ℕ) (Reg : Type)
    (I : Inputs) (own : Option ByteArray) (ht : I.tags.typed = false)
    (D : Flock.Desc) (hD : Flock.Desc.parse (zkTags I) I.circuitFile I.tables = .ok D) (hU : Built D.unroll.unit)
    (pi : HmIn.Ports) (hpi : HmIn.parse D.unroll = .ok pi)
    (pub : Public) (hpub : HmIn.loadPublic (zkTags I) D.unroll pi I.publicFile = .ok pub)
    (hscope : Discharge.Layout.scopeOkH D.unroll = true)
    (kd : ℕ) (dj : Finset (Fin n) → Lean.Json) (S₀ : Finset (Fin n)) (d₀ : Lean.Json)
    (y₀ : DrawSetupZHJP (inputsJ I 0) own (mPtsOf D.unroll.g (kd / nTab I)) S₀ d₀) :
    let P := (keyProgP D.unroll pi hU n).partition (Discharge.Layout.outsOfP D.unroll hU n pi)
    let vb := (rsZHJP (Reg := Reg) dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpi hpub).bindingZ
      (rsZHJP_computes dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpi hpub)
    ∀ S R u, u ∈ S → ∀ M, (tabOfZ (planZHJP dj y₀) (tabZHJP dj y₀) S R u).S.Satisfies
        (wit execArith (tabOfZ (planZHJP dj y₀) (tabZHJP dj y₀) S R u).S
          (tabOfZ (planZHJP dj y₀) (tabZHJP dj y₀) S R u).sch M) →
      ∀ g ∈ P.io u, ∀ b, Bridge.constsOfP D.unroll hU n pi pub g = some b →
        vb.bit (vb.val (vb.opening S R u M (vb.pos g))) g = b := by
  intro P vb S R u hu M hM g hg b hb
  rw [(vb.opening_spec S R u hu M hM g hg).2]
  simp only [decodeOfPlace, hu, ↓reduceDIte, placeZHJP, decode_placeCast]
  have hz := FlockSoundness.Discharge.ZkBind.sat_witOf (planZHJP_S dj y₀ S R u) hM
  obtain ⟨x, hx⟩ := Discharge.Layout.exists_col_of_io
    ((Discharge.HiddenExec.keyPlaceHP (sitesShHJP_parse (Reg := Reg) dj y₀ ht (Flock.Desc.parse_flat hD) hscope) hpi
      (Discharge.Layout.outsOfP D.unroll hU n pi)) S R u hu).inst hg
  rcases hcs : Bridge.constsOf D.unroll hU n g with _ | b'
  · simp only [Bridge.constsOfP, hcs] at hb
    obtain ⟨⟨u', q, t⟩, hpq, rfl⟩ := Option.map_eq_some_iff.1 hb
    subst hx
    unfold UnitPlace.decode
    rw [FlockSoundness.Audit.Partition.IsRowsUnit.decode_wire _ (UnitPlace.agree _ hz),
      Discharge.Layout.block_gateAt
        (Discharge.HiddenExec.keyPlaceHP (sitesShHJP_parse (Reg := Reg) dj y₀ ht (Flock.Desc.parse_flat hD) hscope)
          hpi (Discharge.Layout.outsOfP D.unroll hU n pi)) hu hz x]
    obtain ⟨M', hM'⟩ := FlockSoundness.Discharge.ZkBind.wit_surj
      (Partition.tabOf (planShHJP (Reg := Reg) dj y₀) (tabShHJP dj y₀) S R u)
      (FlockSoundness.Discharge.ZkBind.witOf (planZHJP_S dj y₀ S R u)
        (wit execArith (tabOfZ (planZHJP dj y₀) (tabZHJP dj y₀) S R u).S
          (tabOfZ (planZHJP dj y₀) (tabZHJP dj y₀) S R u).sch M))
    have hw : Binding.wit (planShHJP (Reg := Reg) dj y₀) (tabShHJP dj y₀) S R u M' =
        FlockSoundness.Discharge.ZkBind.witOf (planZHJP_S dj y₀ S R u)
          (wit execArith (tabOfZ (planZHJP dj y₀) (tabZHJP dj y₀) S R u).S
            (tabOfZ (planZHJP dj y₀) (tabZHJP dj y₀) S R u).sch M) := hM'
    have hsat : Binding.Sat (planShHJP (Reg := Reg) dj y₀) (tabShHJP dj y₀) S R u M' := by
      unfold Binding.Sat
      rw [hw]
      exact hz
    have hgate := Discharge.HiddenExec.keyProgHP_gate
      (sitesShHJP_parse (Reg := Reg) dj y₀ ht (Flock.Desc.parse_flat hD) hscope) hpi
      (Discharge.Layout.outsOfP D.unroll hU n pi) (fun _ => pub) (Discharge.Layout.scopeOkH_spec hU hscope).2.1
      (Discharge.HiddenExec.rowsFit_of_scopeH (FlockSoundness.Discharge.ZkBind.zkTyped ht) (Flock.Desc.parse_flat hD)
        hscope)
      (Discharge.Layout.outsOutP_outsOfP D.unroll hU n pi)
      (keysRowP_of_scopeZ ht (Flock.Desc.parse_flat hD) hscope hU)
      (sitesShHJP_bcSite dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpi hpub) hu hsat hg
      (Discharge.HiddenExec.rowOfHP_pub _ hpq)
    rw [Discharge.HiddenExec.keyGateP_pub _ _ R hpq, hw] at hgate
    exact hgate
  · have hbb : b' = b := by simpa [Bridge.constsOfP, hcs] using hb
    subst hbb
    simp only [Bridge.constsOf] at hcs
    split_ifs at hcs with h1 h2
    · cases hcs
      have hone := Discharge.HiddenExec.constCols_keyProgHP hU
        (sitesShHJP_parse (Reg := Reg) dj y₀ ht (Flock.Desc.parse_flat hD) hscope) hpi
        (Discharge.Layout.outsOfP D.unroll hU n pi) S R u hu x (by rw [hx, h1]; exact Finset.mem_singleton_self _)
      rw [← hx, hone]
      exact UnitPlace.decode_one _ hz
    · cases hcs
      have hzero := Discharge.HiddenExec.zeroCols_keyProgHP hU
        (sitesShHJP_parse (Reg := Reg) dj y₀ ht (Flock.Desc.parse_flat hD) hscope) hpi
        (Discharge.Layout.outsOfP D.unroll hU n pi) S R u hu x (by rw [hx, h2]; exact Finset.mem_singleton_self _) _ hz
      rw [← hx]
      unfold UnitPlace.decode
      rw [FlockSoundness.Audit.Partition.IsRowsUnit.decode_wire _ (UnitPlace.agree _ hz), hzero]
      rfl

/-- **`OpeningsCanonTargetV5` holds**: at a constant wire by `openingsCanonV5_const`; at any other committed wire the
wire has a row (`HiddenExec.rowOfHP_ne_none_of_io`, under `scopeOkH`'s circuit conditions), so the opening is canonical
there (`openingsCanonV5_row`). -/
theorem _root_.Flock.Bridge.openingsCanonV5 : Bridge.OpeningsCanonTargetV5 := by
  intro n Reg I own ht D hD hU pi hpi pub hpub hscope kd dj S₀ d₀ y₀ P vb Cc Lm rows κ S R u hu M hM g hg
  refine ⟨fun b hb => Bridge.openingsCanonV5_const n Reg I own ht D hD hU pi hpi pub hpub hscope kd dj S₀ d₀ y₀ S R u
    hu M hM g hg b hb, fun hk => ?_⟩
  have hk' : Bridge.constsOfP D.unroll hU n pi pub g = none := hk
  rcases hcs : Bridge.constsOf D.unroll hU n g with _ | b'
  · have hp : Discharge.Layout.keyPubOf D.unroll hU n pi g = none := by
      simpa [Bridge.constsOfP, hcs] using hk'
    simp only [Bridge.constsOf] at hcs
    split_ifs at hcs with h1 h2
    refine Bridge.openingsCanonV5_row n Reg I own ht D hD hU pi hpi pub hpub hscope kd dj S₀ d₀ y₀ S R u M g ?_
    have hr := Discharge.HiddenExec.rowOfHP_ne_none_of_io (R := R)
      (sitesShHJP_parse (Reg := Reg) dj y₀ ht (Flock.Desc.parse_flat hD) hscope) hpi
      (Discharge.Layout.outsOfP D.unroll hU n pi) (Discharge.Layout.scopeOkH_spec hU hscope).2.1
      (Discharge.HiddenExec.rowsFit_of_scopeH (FlockSoundness.Discharge.ZkBind.zkTyped ht) (Flock.Desc.parse_flat hD)
        hscope)
      (Discharge.Layout.outsOutP_outsOfP D.unroll hU n pi)
      (keysRowP_of_scopeZ ht (Flock.Desc.parse_flat hD) hscope hU) hu hg h1 h2 hp
    simpa [rows] using hr
  · exact absurd hk' (by simp [Bridge.constsOfP, hcs])

end Flock.Guarantees
