import Proofs.Flock.Bridge.Canon

/-!
# A drawn unit's openings are canonical (proved, at wires with a row)

`OpeningsCanonTarget` asks `canonOpen` at every committed wire of a drawn unit, and `canonOpen` asks the wire to have a
row. The constant `oneG` and the zero `zeroG` are committed wires every unit may read (`Layout.single_gate`) and have no
row (`rowOfH = none`: they are the game's single-gate commit strings `.inr g`), so `OpeningsCanonTarget` fails at any
drawn unit that reads one. The claims side needs no opening there (`Claims.Sampling.Faulty` fills an unopened position
by `ι`, and `commitDep` opens nothing where `atOf = none`).

* `Bridge.canonOpenH rows g v y`: `g` has no row, or the opening is canonical there (`canonOpen`);
* `Bridge.OpeningsCanonTargetH`: `OpeningsCanonTarget` at `canonOpenH`;
* `Bridge.openingsCanonH : Bridge.OpeningsCanonTargetH`. At a wire with a row `(u', q, k)`, its commit string is
  `.inl (u', q)` (`Layout.ofRows_pos_some`), whose value is `(widthOf W (portRowW q), padBits b)` for the bits `b` the
  witness holds there (`Readers.ofBits`'s `rowOf`), canonical by `rowShaped_rowOf` at `portRowW q = 16 · words ≤ W`,
  and whose salt is the witness's bytes at `saltAt`, `List.ofFn` over `Fin 192`;
* `Bridge.openingsCanon_row`: the same, as `canonOpen`, at every wire that has a row.
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

end Flock.Guarantees
