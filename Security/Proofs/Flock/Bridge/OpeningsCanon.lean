import Proofs.Flock.Bridge.Canon

/-!
# A drawn unit's openings are canonical (`Bridge.OpeningsCanonTarget`, proved)

`OpeningsCanonTarget` asks `canonAt` at every committed wire of a drawn unit: at the constant `oneG` and the zero
`zeroG` (`Bridge.constsOf`), the opening's bit is the constant; at any other wire, the opening is canonical
(`canonOpen`), which asks the wire to have a row.

* `Bridge.canonOpenH rows g v y`: `g` has no row, or the opening is canonical there (`canonOpen`);
* `Bridge.OpeningsCanonTargetH`: `OpeningsCanonTarget`'s second clause at `canonOpenH`;
* `Bridge.openingsCanonH : Bridge.OpeningsCanonTargetH`. At a wire with a row `(u', q, k)`, its commit string is
  `.inl (u', q)` (`Layout.ofRows_pos_some`), whose value is `(widthOf W (portRowW q), padBits b)` for the bits `b` the
  witness holds there (`Readers.ofBits`'s `rowOf`), canonical by `rowShaped_rowOf` at `portRowW q = 16 · words ≤ W`,
  and whose salt is the witness's bytes at `saltAt`, `List.ofFn` over `Fin 192`;
* `Bridge.openingsCanon_row`: the same, as `canonOpen`, at every wire that has a row;
* `HiddenExec.rowOfH_ne_none_of_io`: a committed wire of a drawn unit other than the constant and the zero has a row
  (`Layout.single_gate`; a computed one is a returned output's, `OutsOut`, so an output row's bit);
* `Bridge.openingsCanon_const`: at the constant and the zero, the opening's bit is the decoder's (`opening_spec`), which
  reads the unit's constant column, 1, or a zero column, 0, at the public place (`UnitPlace.decode_one`,
  `constCols_keyProgH`, `zeroCols_keyProgH`);
* `Bridge.openingsCanon : Bridge.OpeningsCanonTarget`.
-/

namespace Flock.Bridge

/-- **A canonical opening, or a wire with no row**: `g` has no row (`rows g = none`), or the opening's value and salt
are canonical at `g` (`canonOpen`). -/
def canonOpenH {N W : ℕ} (rows : Fin N → Option RowAt) (g : Fin N) (v : FlockSoundness.Discharge.Layout.BitVal W)
    (y : List UInt8) : Prop :=
  rows g = none ∨ canonOpen rows g v y

/-- A witness's bytes at `N` addresses are `N` bytes. -/
theorem bytesAt_length {m N : ℕ} (z) (a : Fin N → Fin 8 → Fin (2 ^ m)) :
    (FlockSoundness.Binding.bytesAt (m := m) z a).length = N := by
  simp [FlockSoundness.Binding.bytesAt]

end Flock.Bridge

namespace FlockSoundness.Discharge.HiddenExec

open Flock FlockLevel3 Model Audit Binding Placed
open FlockSoundness.Discharge.Layout (keyN keyRowOf portRowW OutColAt outDecode OutsOut KeysRow)

variable {c : Flock.Circuit} {hU : Built c.unit} {n : ℕ} {Reg : Type} {mPts : ℕ}
  {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length} {si : SitesH plan tab}

/-- **A committed wire of a drawn unit other than the constant and the zero has a row**: a committed gate with no row
is the zero, the constant or a computed gate of the unit (`Layout.single_gate`), and a computed one is a returned
output's (`OutsOut`), so an output row's bit (`outBitRow_ne_none`), as in `keyProgH_valPinned`. -/
theorem rowOfH_ne_none_of_io
    (hparse : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S),
      HmRow.parse (si.acc S R u h).tags (si.acc S R u h).cf (si.acc S R u h).tables = .ok c)
    (outs : Finset (Fin (keyN c hU n))) (ho : c.outNet = c.unitNet) (hrf : Hidden.Any.RowsFitS c)
    (houts : OutsOut c hU n outs) (hkeys : KeysRow c hU (portRowW c)) {S : Finset (Fin n)} {R : Reg} {u : Fin n}
    (hu : u ∈ S) {g : Fin (keyN c hU n)} (hg : g ∈ ((keyProg c hU n).partition outs).io u)
    (h1 : g ≠ Layout.oneG) (h0 : g ≠ Layout.zeroG) : rowOfH c hU n outs g ≠ none := by
  intro hr
  obtain ⟨x, hx⟩ := Layout.exists_col_of_io ((keyPlaceH hparse outs) S R u hu).inst hg
  obtain ⟨hk, hno⟩ := rowOfH_none outs hr
  rcases Layout.single_gate outs hkeys ⟨x, hx⟩ hg hk with rfl | rfl | ⟨hm, -, -⟩
  · exact h0 rfl
  · exact h1 rfl
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
  exact outBitRow_ne_none hcr hrf u hwlt ht hno'

end FlockSoundness.Discharge.HiddenExec

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden FlockSoundness.Discharge.ZkSession
open FlockSoundness.Discharge.Tables (LiveOutJ)

/-- **A drawn unit's openings are canonical where its wires have rows**, at `EndToEndHidden`'s session:
`OpeningsCanonTarget` at `Bridge.canonOpenH`. -/
def _root_.Flock.Bridge.OpeningsCanonTargetH : Prop :=
  ∀ (n : ℕ) (Reg : Type)
    (I : Inputs) (ht : I.tags.typed = false)
    (D : Flock.Desc) (hD : Flock.Desc.parse (zkTags I) I.circuitFile I.tables = .ok D) (hU : Built D.unroll.unit)
    (pub : Public) (hpub : HmOut.loadPublic (zkTags I) D.unroll I.publicFile = .ok pub) (hpt : pub.tables = none)
    (hscope : Discharge.Layout.scopeOkH D.unroll = true)
    (kd : ℕ) (dj : Finset (Fin n) → Lean.Json) (S₀ : Finset (Fin n)) (d₀ : Lean.Json)
    (y₀ : DrawSetupZHJ (inputsJ I 0) (mPtsOf D.unroll.g (kd / nTab I)) S₀ d₀),
    let P := (keyProg D.unroll hU n).partition (Discharge.Layout.outsOf D.unroll hU n)
    let vb := (rsZHJ (Reg := Reg) dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpub hpt).bindingZ
      (rsZHJ_computes dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpub hpt)
    let Cc := (keyProg D.unroll hU n).circuit (Discharge.Layout.outsOf D.unroll hU n)
    let rows : Fin Cc.N → Option Bridge.RowAt := fun g =>
      (FlockSoundness.Discharge.HiddenExec.rowOfH D.unroll hU n (Discharge.Layout.outsOf D.unroll hU n) g).map fun r =>
        ⟨r.2.1.val, r.1.val, Discharge.Layout.portRowW D.unroll r.2.1, r.2.2.val⟩
    OpeningsCanon execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb (fun g o => Bridge.canonOpenH rows g (vb.val o) o.2.2)

/-- **`OpeningsCanonTargetH` holds.** -/
theorem _root_.Flock.Bridge.openingsCanonH : Bridge.OpeningsCanonTargetH := by
  intro n Reg I ht D hD hU pub hpub hpt hscope kd dj S₀ d₀ y₀ P vb Cc rows S R u _ M _ g _
  show Bridge.canonOpenH rows g (vb.val (vb.opening S R u M (vb.pos g))) (vb.opening S R u M (vb.pos g)).2.2
  rcases hx : FlockSoundness.Discharge.HiddenExec.rowOfH D.unroll hU n (Discharge.Layout.outsOf D.unroll hU n) g
    with _ | ⟨u', q, k⟩
  · exact Or.inl (by simp [rows, hx])
  have hpos : vb.pos g = Sum.inl (u', q) :=
    Discharge.Layout.ofRows_pos_some
      (FlockSoundness.Discharge.HiddenExec.keyPlaceH (sitesShHJ_parse (Reg := Reg) dj y₀ ht (Flock.Desc.parse_flat hD) hscope)
        (Discharge.Layout.outsOf D.unroll hU n))
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
  refine Or.inr ⟨⟨⟨q.val, u'.val, Discharge.Layout.portRowW D.unroll q, k.val⟩, by simp [rows, hx], ?_⟩, hlen⟩
  exact Bridge.rowShaped_rowOf rfl
    ((Finset.le_sup (f := Discharge.Layout.portRowW D.unroll) (Finset.mem_univ q)).trans (Nat.le_succ _)) b

/-- **At a wire with a row, a drawn unit's opening is canonical** (`canonOpen`), at `OpeningsCanonTargetH`'s session. -/
theorem _root_.Flock.Bridge.openingsCanon_row
    (n : ℕ) (Reg : Type)
    (I : Inputs) (ht : I.tags.typed = false)
    (D : Flock.Desc) (hD : Flock.Desc.parse (zkTags I) I.circuitFile I.tables = .ok D) (hU : Built D.unroll.unit)
    (pub : Public) (hpub : HmOut.loadPublic (zkTags I) D.unroll I.publicFile = .ok pub) (hpt : pub.tables = none)
    (hscope : Discharge.Layout.scopeOkH D.unroll = true)
    (kd : ℕ) (dj : Finset (Fin n) → Lean.Json) (S₀ : Finset (Fin n)) (d₀ : Lean.Json)
    (y₀ : DrawSetupZHJ (inputsJ I 0) (mPtsOf D.unroll.g (kd / nTab I)) S₀ d₀) :
    let P := (keyProg D.unroll hU n).partition (Discharge.Layout.outsOf D.unroll hU n)
    let vb := (rsZHJ (Reg := Reg) dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpub hpt).bindingZ
      (rsZHJ_computes dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpub hpt)
    let Cc := (keyProg D.unroll hU n).circuit (Discharge.Layout.outsOf D.unroll hU n)
    let rows : Fin Cc.N → Option Bridge.RowAt := fun g =>
      (FlockSoundness.Discharge.HiddenExec.rowOfH D.unroll hU n (Discharge.Layout.outsOf D.unroll hU n) g).map fun r =>
        ⟨r.2.1.val, r.1.val, Discharge.Layout.portRowW D.unroll r.2.1, r.2.2.val⟩
    ∀ S R u, u ∈ S → ∀ M, (tabOfZ (planZHJ dj y₀) (tabZHJ dj y₀) S R u).S.Satisfies
        (wit execArith (tabOfZ (planZHJ dj y₀) (tabZHJ dj y₀) S R u).S (tabOfZ (planZHJ dj y₀) (tabZHJ dj y₀) S R u).sch
          M) →
      ∀ g ∈ P.io u, rows g ≠ none →
        Bridge.canonOpen rows g (vb.val (vb.opening S R u M (vb.pos g))) (vb.opening S R u M (vb.pos g)).2.2 := by
  intro P vb Cc rows S R u hu M hM g hg hr
  exact (Bridge.openingsCanonH n Reg I ht D hD hU pub hpub hpt hscope kd dj S₀ d₀ y₀ S R u hu M hM g hg).resolve_left hr

/-- **At the constant and the zero, a drawn unit's opening carries the constant**, at `OpeningsCanonTarget`'s session:
the opening's bit is the decoder's (`opening_spec`), which at the public place reads the unit's constant column, `1`
(`UnitPlace.decode_one`, `constCols_keyProgH`), or a zero column, `0` (`zeroCols_keyProgH`). -/
theorem _root_.Flock.Bridge.openingsCanon_const
    (n : ℕ) (Reg : Type)
    (I : Inputs) (ht : I.tags.typed = false)
    (D : Flock.Desc) (hD : Flock.Desc.parse (zkTags I) I.circuitFile I.tables = .ok D) (hU : Built D.unroll.unit)
    (pub : Public) (hpub : HmOut.loadPublic (zkTags I) D.unroll I.publicFile = .ok pub) (hpt : pub.tables = none)
    (hscope : Discharge.Layout.scopeOkH D.unroll = true)
    (kd : ℕ) (dj : Finset (Fin n) → Lean.Json) (S₀ : Finset (Fin n)) (d₀ : Lean.Json)
    (y₀ : DrawSetupZHJ (inputsJ I 0) (mPtsOf D.unroll.g (kd / nTab I)) S₀ d₀) :
    let P := (keyProg D.unroll hU n).partition (Discharge.Layout.outsOf D.unroll hU n)
    let vb := (rsZHJ (Reg := Reg) dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpub hpt).bindingZ
      (rsZHJ_computes dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpub hpt)
    ∀ S R u, u ∈ S → ∀ M, (tabOfZ (planZHJ dj y₀) (tabZHJ dj y₀) S R u).S.Satisfies
        (wit execArith (tabOfZ (planZHJ dj y₀) (tabZHJ dj y₀) S R u).S (tabOfZ (planZHJ dj y₀) (tabZHJ dj y₀) S R u).sch
          M) →
      ∀ g ∈ P.io u, ∀ b, Bridge.constsOf D.unroll hU n g = some b →
        vb.bit (vb.val (vb.opening S R u M (vb.pos g))) g = b := by
  intro P vb S R u hu M hM g hg b hb
  rw [(vb.opening_spec S R u hu M hM g hg).2]
  simp only [decodeOfPlace, hu, ↓reduceDIte, placeZHJ, decode_placeCast]
  have hz := FlockSoundness.Discharge.ZkBind.sat_witOf (planZHJ_S dj y₀ S R u) hM
  obtain ⟨c, hc⟩ := Discharge.Layout.exists_col_of_io
    ((FlockSoundness.Discharge.HiddenExec.keyPlaceH
      (sitesShHJ_parse (Reg := Reg) dj y₀ ht (Flock.Desc.parse_flat hD) hscope)
      (Discharge.Layout.outsOf D.unroll hU n)) S R u hu).inst hg
  simp only [Bridge.constsOf] at hb
  split_ifs at hb with h1 h2
  · cases hb
    have hone := FlockSoundness.Discharge.HiddenExec.constCols_keyProgH hU
      (sitesShHJ_parse (Reg := Reg) dj y₀ ht (Flock.Desc.parse_flat hD) hscope)
      (Discharge.Layout.outsOf D.unroll hU n) S R u hu c (by rw [hc, h1]; exact Finset.mem_singleton_self _)
    rw [← hc, hone]
    exact UnitPlace.decode_one _ hz
  · cases hb
    have hzero := FlockSoundness.Discharge.HiddenExec.zeroCols_keyProgH hU
      (sitesShHJ_parse (Reg := Reg) dj y₀ ht (Flock.Desc.parse_flat hD) hscope)
      (Discharge.Layout.outsOf D.unroll hU n) S R u hu c (by rw [hc, h2]; exact Finset.mem_singleton_self _) _ hz
    rw [← hc]
    unfold UnitPlace.decode
    rw [FlockSoundness.Audit.Partition.IsRowsUnit.decode_wire _ (UnitPlace.agree _ hz), hzero]
    rfl

/-- **`OpeningsCanonTarget` holds**: at the constant and the zero by `openingsCanon_const`; at any other committed wire
the wire has a row (`HiddenExec.rowOfH_ne_none_of_io`, under `scopeOkH`'s circuit conditions), so `openingsCanonH`'s
opening is canonical there. -/
theorem _root_.Flock.Bridge.openingsCanon : Bridge.OpeningsCanonTarget := by
  intro n Reg I ht D hD hU pub hpub hpt hscope kd dj S₀ d₀ y₀ P vb Cc rows κ S R u hu M hM g hg
  refine ⟨fun b hb => Bridge.openingsCanon_const n Reg I ht D hD hU pub hpub hpt hscope kd dj S₀ d₀ y₀ S R u hu M hM g
    hg b hb, fun hk => ?_⟩
  rcases Bridge.openingsCanonH n Reg I ht D hD hU pub hpub hpt hscope kd dj S₀ d₀ y₀ S R u hu M hM g hg with h | h
  · exfalso
    have hk' : Bridge.constsOf D.unroll hU n g = none := hk
    simp only [Bridge.constsOf] at hk'
    split_ifs at hk' with h1 h2
    refine FlockSoundness.Discharge.HiddenExec.rowOfH_ne_none_of_io (R := R)
      (sitesShHJ_parse (Reg := Reg) dj y₀ ht (Flock.Desc.parse_flat hD) hscope) (Discharge.Layout.outsOf D.unroll hU n)
      (Discharge.Layout.scopeOkH_spec hU hscope).2.1
      (FlockSoundness.Discharge.HiddenExec.rowsFit_of_scopeH (FlockSoundness.Discharge.ZkBind.zkTyped ht)
        (Flock.Desc.parse_flat hD) hscope)
      (Discharge.Layout.outsOut_outsOf D.unroll hU n)
      (Discharge.Layout.keysRow_of_parse
        (FlockSoundness.Discharge.Hidden.parse_untyped (FlockSoundness.Discharge.ZkBind.zkTyped ht)
          (Flock.Desc.parse_flat hD)).1
        (Discharge.Layout.scopeOkH_spec hU hscope).1 (Discharge.Layout.scopeOkH_spec hU hscope).2.2)
      hu hg h1 h2 ?_
    simpa [rows] using h
  · exact h

end Flock.Guarantees
