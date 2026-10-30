import Proofs.Flock.Soundness.Discharge.Layout.KeyProg
import Proofs.Flock.Soundness.Discharge.Placed.KeyedP

set_option Elab.async false

/-!
# The row bits and public gates of the circuit's program on the keys with public inputs

`KeyProg.lean`'s gates at `Placed.keyProgP c pi hU n`, whose unit inputs sit on the gate of their key with public inputs
(`Aliased.keyIdxP`):

* `keyRowOfP`: the row bits, input gates whose key is a row's message bit (as `keyRowOf`);
* `keyPubOf`: the public gates, input gates `2 + u·B + b` whose key is bit `t` of public word `q` (`Aliased.KeyP.pub`),
  unit `u`'s;
* `keyGateP`: the registration's bit at a single gate: 1 at the constant, 0 at the zero, unit `u`'s public word `q`'s bit
  `t` at a public gate (`pubBit`), `outBit`'s elsewhere;
* `wire_inGateP`: a column of a unit's own instance on an input gate is the unit's input, keyed as the gate;
* `single_gateP`: a single committed gate of a unit's own instance is the zero, the constant, a public gate of the
  unit, or one of its computed gates, when every input a row reads has the zero, a row bit or a public word bit as its
  key (`KeysRowP`).
-/

namespace FlockSoundness.Discharge.Layout

open Flock FlockLevel3 Model Audit Binding Placed

variable (c : Flock.Circuit) (hU : Built c.unit) (n : ℕ) (pi : HmIn.Ports) (rowW : Fin c.ports.size → ℕ)

/-- **The row bits of `keyProgP`**: input gate `2 + u·B + b` is bit `8·byte + bit` of unit `u`'s row port `q` when input
`b`'s key is message bit `bit < 8` of byte `byte` of row port `q`, within the row's `rowW q` bits. -/
noncomputable def keyRowOfP (g : Fin (keyN c hU n)) : Option (Fin n × (q : Fin c.ports.size) × Fin (rowW q)) :=
  if h : 2 ≤ g.val ∧ g.val < 2 + n * keyB c hU then
    match Aliased.srcKeyP c pi ((g.val - 2) % keyB c hU) with
    | .src (.msg q byte bit) =>
      if hq : q < c.ports.size then
        if hk : 8 * byte + bit < rowW ⟨q, hq⟩ ∧ bit < 8 then
          some (⟨(g.val - 2) / keyB c hU, (Nat.div_lt_iff_lt_mul (Nat.pos_of_ne_zero fun h0 => by
            rw [h0, Nat.mul_zero] at h; omega)).2 (by omega)⟩, ⟨⟨q, hq⟩, ⟨8 * byte + bit, hk.1⟩⟩)
        else none
      else none
    | _ => none
  else none

/-- **The public gates of `keyProgP`**: input gate `2 + u·B + b` is unit `u`'s bit `t` of public word `q` when input
`b`'s key is that bit (`Aliased.KeyP.pub q t`). -/
noncomputable def keyPubOf (g : Fin (keyN c hU n)) : Option (Fin n × ℕ × ℕ) :=
  if h : 2 ≤ g.val ∧ g.val < 2 + n * keyB c hU then
    match Aliased.srcKeyP c pi ((g.val - 2) % keyB c hU) with
    | .pub q t => some (⟨(g.val - 2) / keyB c hU, (Nat.div_lt_iff_lt_mul (Nat.pos_of_ne_zero fun h0 => by
        rw [h0, Nat.mul_zero] at h; omega)).2 (by omega)⟩, q, t)
    | _ => none
  else none

/-- **The registration's bit at a single gate of `keyProgP`**: 1 at the constant, 0 at the zero, `pubBit R u q t` at
unit `u`'s public gate of bit `t` of public word `q`, `outBit R g` elsewhere. -/
noncomputable def keyGateP {Reg : Type} (pubBit : Reg → Fin n → ℕ → ℕ → Bool)
    (outBit : Reg → Fin (keyN c hU n) → Bool) (R : Reg) (g : Fin (keyN c hU n)) : Bool :=
  if g = oneG (c := c) (hU := hU) (n := n) then true
  else if g = zeroG (c := c) (hU := hU) (n := n) then false
  else match keyPubOf c hU n pi g with
    | some (u, q, t) => pubBit R u q t
    | none => outBit R g

/-- **An input's key with public inputs is the zero, a row bit or a public word bit**: as `RowKey`, with `.pub`. -/
def RowKeyP (x : ℕ) : Prop :=
  Aliased.srcKeyP c pi x = .src .zero ∨
    (∃ q byte bit, Aliased.srcKeyP c pi x = .src (.msg q byte bit) ∧
      ∃ hq : q < c.ports.size, 8 * byte + bit < rowW ⟨q, hq⟩ ∧ bit < 8) ∨
    ∃ q t, Aliased.srcKeyP c pi x = .pub q t

/-- **Every input a computed row reads has a key with public inputs** (`RowKeyP`), as `KeysRow`. -/
def KeysRowP : Prop :=
  ∀ (i : Fin (Placed.circuitRows c hU).nComp) (x : Fin (Placed.circuitRows c hU).w),
    x ∈ (Placed.circuitRows c hU).a i ++ (Placed.circuitRows c hU).b i → x.val < (Placed.circuitRows c hU).nIn →
      RowKeyP c pi rowW x.val

variable {c hU n pi rowW}

/-- **`KeysRowP` from `KeysRow`**: an input on a public leaf has a public word bit as its key, and one off every public
leaf keeps its key (`Aliased.srcKeyP_of_neg`). -/
theorem keysRowP_of_keysRow (h : KeysRow c hU rowW) : KeysRowP c hU pi rowW := by
  intro i x hx hlt
  by_cases hp : 0 ≤ Aliased.pubLeaf c pi x.val
  · obtain ⟨t, ht⟩ := Aliased.srcKeyP_of_nonneg hp
    exact .inr (.inr ⟨_, t, ht⟩)
  · unfold RowKeyP
    rw [Aliased.srcKeyP_of_neg (by omega)]
    rcases h i x hx hlt with h0 | ⟨q, byte, bit, hk, hq⟩
    · exact .inl (by rw [h0])
    · exact .inr (.inl ⟨q, byte, bit, by rw [hk], hq⟩)

theorem keyRowOfP_spec {g : Fin (keyN c hU n)} {u : Fin n} {q : Fin c.ports.size} {k : Fin (rowW q)}
    (h : keyRowOfP c hU n pi rowW g = some (u, ⟨q, k⟩)) :
    2 ≤ g.val ∧ g.val < 2 + n * keyB c hU ∧ u.val = (g.val - 2) / keyB c hU ∧
      Aliased.srcKeyP c pi ((g.val - 2) % keyB c hU) = .src (.msg q.val (k.val / 8) (k.val % 8)) := by
  unfold keyRowOfP at h
  split at h
  · rename_i hg
    split at h
    · rename_i q₀ byte bit hkey
      split at h
      · rename_i hq
        split at h
        · rename_i hk
          simp only [Option.some.injEq, Prod.mk.injEq] at h
          obtain ⟨hu, hs⟩ := h
          subst hu
          cases hs
          refine ⟨hg.1, hg.2, rfl, ?_⟩
          rw [hkey]
          congr 2 <;> dsimp only <;> omega
        · exact absurd h (by simp)
      · exact absurd h (by simp)
    all_goals exact absurd h (by simp)
  · exact absurd h (by simp)

theorem keyRowOfP_ne_none {g : Fin (keyN c hU n)} (h2 : 2 ≤ g.val) (hlt : g.val < 2 + n * keyB c hU)
    {q byte bit : ℕ} (hk : Aliased.srcKeyP c pi ((g.val - 2) % keyB c hU) = .src (.msg q byte bit))
    (hq : q < c.ports.size) (hw : 8 * byte + bit < rowW ⟨q, hq⟩) (hb : bit < 8) :
    keyRowOfP c hU n pi rowW g ≠ none := by
  unfold keyRowOfP
  simp only [h2, hlt, and_self, ↓reduceDIte, hk, hq, hw, hb]
  exact Option.some_ne_none _

theorem keyPubOf_spec {g : Fin (keyN c hU n)} {u : Fin n} {q t : ℕ} (h : keyPubOf c hU n pi g = some (u, q, t)) :
    2 ≤ g.val ∧ g.val < 2 + n * keyB c hU ∧ u.val = (g.val - 2) / keyB c hU ∧
      Aliased.srcKeyP c pi ((g.val - 2) % keyB c hU) = .pub q t := by
  unfold keyPubOf at h
  split at h
  · rename_i hg
    split at h
    · rename_i q₀ t₀ hkey
      simp only [Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨hu, rfl, rfl⟩ := h
      subst hu
      exact ⟨hg.1, hg.2, rfl, hkey⟩
    · exact absurd h (by simp)
  · exact absurd h (by simp)

theorem keyPubOf_of {g : Fin (keyN c hU n)} (h2 : 2 ≤ g.val) (hlt : g.val < 2 + n * keyB c hU) {q t : ℕ}
    (hk : Aliased.srcKeyP c pi ((g.val - 2) % keyB c hU) = .pub q t) :
    ∃ u : Fin n, u.val = (g.val - 2) / keyB c hU ∧ keyPubOf c hU n pi g = some (u, q, t) := by
  unfold keyPubOf
  simp only [h2, hlt, and_self, ↓reduceDIte, hk]
  exact ⟨_, rfl, rfl⟩

/-- **A column of a unit's own instance on an input gate** `2 ≤ g < 2 + n·B` is an input column of the unit, the
gate's unit index is the unit's, and the gate's key is the input's (`Aliased.keyIdxP_some`). -/
theorem wire_inGateP (outs : Finset (Fin (keyN c hU n))) {u : Fin n} (x : Fin ((keyProgP c pi hU n).rows u).w)
    (h2 : 2 ≤ ((Const.Prog.instC (keyProgP c pi hU n) outs u).wire x).val)
    (hlt : ((Const.Prog.instC (keyProgP c pi hU n) outs u).wire x).val < 2 + n * keyB c hU) :
    x.val < ((keyProgP c pi hU n).rows u).nIn ∧
      (((Const.Prog.instC (keyProgP c pi hU n) outs u).wire x).val - 2) / keyB c hU = u.val ∧
      Aliased.srcKeyP c pi ((((Const.Prog.instC (keyProgP c pi hU n) outs u).wire x).val - 2) % keyB c hU) =
        Aliased.srcKeyP c pi x.val := by
  have e := congrArg FlockSoundness.Rows.nIn
    (rows_circuitProg c hU n (keyB c hU) (Aliased.keyIdxP c pi (keyB c hU)) u)
  have hx : x.val < ((keyProgP c pi hU n).rows u).nIn := by
    by_contra hx
    by_cases hone : x = ((keyProgP c pi hU n).rows u).one
    · rw [hone, Const.Prog.instC_wire_one] at h2
      have h0 : (Const.Prog.oneAt (keyProgP c pi hU n) u).val = 0 :=
        oneAt_uniform_val (k := Placed.inGates n (keyB c hU)) (R := Placed.circuitRows c hU)
          (ins := Placed.insOf n (keyB c hU) (Aliased.keyIdxP c pi (keyB c hU)))
          (one := Placed.oneGate n (keyB c hU))
          (hone := Placed.insOf_ne_one n (keyB c hU) (Aliased.keyIdxP c pi (keyB c hU))) n u
      omega
    · have hne : x.val ≠ ((keyProgP c pi hU n).rows u).nIn + ((keyProgP c pi hU n).rows u).nComp := fun he =>
        hone (Fin.ext he)
      have hxw := x.isLt
      have hi : x.val - ((keyProgP c pi hU n).rows u).nIn < ((keyProgP c pi hU n).rows u).nComp := by
        change x.val < ((keyProgP c pi hU n).rows u).nIn + ((keyProgP c pi hU n).rows u).nComp + 1 at hxw
        omega
      have hx' : x = ((keyProgP c pi hU n).rows u).comp ⟨_, hi⟩ :=
        Fin.ext (by simp only [FlockSoundness.Rows.comp]; omega)
      have hm := (Const.Prog.instC (keyProgP c pi hU n) outs u).mem ⟨_, hi⟩
      rw [← hx'] at hm
      have hnone : (keyProgP c pi hU n).unit ((Const.Prog.instC (keyProgP c pi hU n) outs u).wire x) = none :=
        unit_uniform_input (k := Placed.inGates n (keyB c hU)) (R := Placed.circuitRows c hU)
          (ins := Placed.insOf n (keyB c hU) (Aliased.keyIdxP c pi (keyB c hU)))
          (one := Placed.oneGate n (keyB c hU))
          (hone := Placed.insOf_ne_one n (keyB c hU) (Aliased.keyIdxP c pi (keyB c hU))) n _ (by
            show _ < 2 + n * keyB c hU; omega)
      have : (keyProgP c pi hU n).unit ((Const.Prog.instC (keyProgP c pi hU n) outs u).wire x) = some u := hm
      rw [hnone] at this
      cases this
  have hxB : x.val < keyB c hU := lt_of_lt_of_eq hx e
  refine ⟨hx, ?_⟩
  have hv : (Const.Prog.insAt (keyProgP c pi hU n) u x.val).val =
      (Placed.insOf n (keyB c hU) (Aliased.keyIdxP c pi (keyB c hU)) u.val ⟨x.val, hxB⟩).val :=
    Placed.insAt_uniform n u x.val hxB
  rw [Const.Prog.instC_wire_input _ _ _ x hx, hv,
    Placed.insOf_of_lt (nU := n) (B := keyB c hU) (src := Aliased.keyIdxP c pi (keyB c hU)) u.isLt] at h2 ⊢
  rcases hb : Aliased.keyIdxP c pi (keyB c hU) ⟨x.val, hxB⟩ with _ | b
  · rw [hb] at h2
    exact absurd h2 (by simp [Placed.srcGate, Placed.zeroGate])
  · simp only [Placed.srcGate]
    have hB : 0 < keyB c hU := lt_of_le_of_lt (Nat.zero_le _) hxB
    have hd : (2 + u.val * keyB c hU + b.val - 2) / keyB c hU = u.val := by
      rw [show 2 + u.val * keyB c hU + b.val - 2 = b.val + u.val * keyB c hU by omega,
        Nat.add_mul_div_right _ _ hB, Nat.div_eq_of_lt b.isLt, Nat.zero_add]
    have hm : (2 + u.val * keyB c hU + b.val - 2) % keyB c hU = b.val := by
      rw [show 2 + u.val * keyB c hU + b.val - 2 = b.val + u.val * keyB c hU by omega,
        Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt b.isLt]
    exact ⟨hd, by rw [hm, Aliased.keyIdxP_some hb]⟩

/-- **A column of a unit's own instance on a public gate** of unit `u'` is an input column of `u = u'` with that key. -/
theorem keyPubOf_wire (outs : Finset (Fin (keyN c hU n))) {u u' : Fin n} (x : Fin ((keyProgP c pi hU n).rows u).w)
    {q t : ℕ} (h : keyPubOf c hU n pi ((Const.Prog.instC (keyProgP c pi hU n) outs u).wire x) = some (u', q, t)) :
    u' = u ∧ x.val < ((keyProgP c pi hU n).rows u).nIn ∧ Aliased.srcKeyP c pi x.val = .pub q t := by
  obtain ⟨h2, hlt, hu, hkey⟩ := keyPubOf_spec h
  obtain ⟨hx, hd, hk⟩ := wire_inGateP outs x h2 hlt
  exact ⟨Fin.ext (hu.trans hd), hx, hk.symm.trans hkey⟩

/-- **A column of a unit's own instance on a row bit** of unit `u'` is an input column of `u = u'` whose key is the row
bit's message bit (`keyRowOf_wire` at `keyProgP`). -/
theorem keyRowOfP_wire (outs : Finset (Fin (keyN c hU n))) {u u' : Fin n} (x : Fin ((keyProgP c pi hU n).rows u).w)
    {q : Fin c.ports.size} {k : Fin (rowW q)}
    (h : keyRowOfP c hU n pi rowW ((Const.Prog.instC (keyProgP c pi hU n) outs u).wire x) = some (u', ⟨q, k⟩)) :
    u' = u ∧ x.val < ((keyProgP c pi hU n).rows u).nIn ∧
      Aliased.srcKeyP c pi x.val = .src (.msg q.val (k.val / 8) (k.val % 8)) := by
  obtain ⟨h2, hlt, hu, hkey⟩ := keyRowOfP_spec h
  obtain ⟨hx, hd, hk⟩ := wire_inGateP outs x h2 hlt
  exact ⟨Fin.ext (hu.trans hd), hx, hk.symm.trans hkey⟩

/-- `KeysRowP` at a unit of `keyProgP`, whose rows are the circuit's. -/
theorem keysRowP_keyProgP (hkeys : KeysRowP c hU pi rowW) (u : Fin n) (i : Fin ((keyProgP c pi hU n).rows u).nComp)
    (x : Fin ((keyProgP c pi hU n).rows u).w)
    (hx : x ∈ ((keyProgP c pi hU n).rows u).a i ++ ((keyProgP c pi hU n).rows u).b i)
    (hxn : x.val < ((keyProgP c pi hU n).rows u).nIn) : RowKeyP c pi rowW x.val := by
  have e : (keyProgP c pi hU n).rows u = Placed.circuitRows c hU :=
    rows_circuitProg c hU n (keyB c hU) (Aliased.keyIdxP c pi (keyB c hU)) u
  revert i x
  generalize (keyProgP c pi hU n).rows u = R at e ⊢
  subst e
  exact hkeys

/-- **A single committed gate of a unit's own instance of `keyProgP`** is the zero, the constant, a public gate of the
unit, or one of the unit's computed gates, when every input a row reads has a key with public inputs (`KeysRowP`). -/
theorem single_gateP (outs : Finset (Fin (keyN c hU n))) (hkeys : KeysRowP c hU pi rowW) {u : Fin n}
    {g : Fin (keyN c hU n)}
    (hgx : ∃ x : Fin ((keyProgP c pi hU n).rows u).w, (Const.Prog.instC (keyProgP c pi hU n) outs u).wire x = g)
    (hg : g ∈ ((keyProgP c pi hU n).partition outs).io u) (hr : keyRowOfP c hU n pi rowW g = none) :
    g = zeroG ∨ g = oneG ∨ (∃ q t, keyPubOf c hU n pi g = some (u, q, t)) ∨
      ((keyProgP c pi hU n).unit g = some u ∧ g ≠ oneG ∧ g ≠ zeroG) := by
  obtain ⟨x, rfl⟩ := hgx
  have e := congrArg FlockSoundness.Rows.nIn
    (rows_circuitProg c hU n (keyB c hU) (Aliased.keyIdxP c pi (keyB c hU)) u)
  have hgate_in : ∀ {g : Fin (keyN c hU n)}, g.val < 2 + n * keyB c hU → (keyProgP c pi hU n).unit g = none :=
    fun {g} hg => unit_uniform_input (k := Placed.inGates n (keyB c hU)) (R := Placed.circuitRows c hU)
      (ins := Placed.insOf n (keyB c hU) (Aliased.keyIdxP c pi (keyB c hU))) (one := Placed.oneGate n (keyB c hU))
      (hone := Placed.insOf_ne_one n (keyB c hU) (Aliased.keyIdxP c pi (keyB c hU))) n g hg
  by_cases hx : x.val < ((keyProgP c pi hU n).rows u).nIn
  · have hxB : x.val < keyB c hU := lt_of_lt_of_eq hx e
    have hw : ((Const.Prog.instC (keyProgP c pi hU n) outs u).wire x).val =
        (Placed.srcGate n (keyB c hU) u.isLt (Aliased.keyIdxP c pi (keyB c hU) ⟨x.val, hxB⟩)).val := by
      have hv : (Const.Prog.insAt (keyProgP c pi hU n) u x.val).val =
          (Placed.insOf n (keyB c hU) (Aliased.keyIdxP c pi (keyB c hU)) u.val ⟨x.val, hxB⟩).val :=
        Placed.insAt_uniform n u x.val hxB
      rw [Const.Prog.instC_wire_input _ _ _ x hx, hv,
        Placed.insOf_of_lt (nU := n) (B := keyB c hU) (src := Aliased.keyIdxP c pi (keyB c hU)) u.isLt]
    rcases hb : Aliased.keyIdxP c pi (keyB c hU) ⟨x.val, hxB⟩ with _ | b
    · rw [hb] at hw
      exact .inl (Fin.ext hw)
    · rw [hb] at hw
      simp only [Placed.srcGate] at hw
      have hnz : Aliased.srcKeyP c pi x.val ≠ .src .zero := fun h => by
        rw [(Aliased.keyIdxP_eq_none (c := c) (pi := pi) (⟨x.val, hxB⟩ : Fin (keyB c hU))).2 h] at hb
        cases hb
      have hB : 0 < keyB c hU := lt_of_le_of_lt (Nat.zero_le _) hxB
      have hm : (2 + u.val * keyB c hU + b.val - 2) % keyB c hU = b.val := by
        rw [show 2 + u.val * keyB c hU + b.val - 2 = b.val + u.val * keyB c hU by omega,
          Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt b.isLt]
      have hd : (2 + u.val * keyB c hU + b.val - 2) / keyB c hU = u.val := by
        rw [show 2 + u.val * keyB c hU + b.val - 2 = b.val + u.val * keyB c hU by omega,
          Nat.add_mul_div_right _ _ hB, Nat.div_eq_of_lt b.isLt, Nat.zero_add]
      have hnone : (keyProgP c pi hU n).unit ((Const.Prog.instC (keyProgP c pi hU n) outs u).wire x) = none :=
        hgate_in (by rw [hw]; exact Placed.source_lt n _ u.isLt b)
      obtain ⟨i, c', hc', hwc⟩ := exists_read_of_io (Const.Prog.instC (keyProgP c pi hU n) outs u) hg
        (fun h => by
          have h' : (keyProgP c pi hU n).unit ((Const.Prog.instC (keyProgP c pi hU n) outs u).wire x) = some u := h
          rw [hnone] at h'
          cases h')
      have hc'n : c'.val < ((keyProgP c pi hU n).rows u).nIn := by
        rcases (Const.Prog.instC (keyProgP c pi hU n) outs u).shared c' x hwc with rfl | ⟨h, -⟩
        · exact hx
        · exact h
      have hkey' := Placed.keyProgP_wire_key hU outs u x c' hx hc'n hwc.symm
      have hgk : Aliased.srcKeyP c pi
          ((((Const.Prog.instC (keyProgP c pi hU n) outs u).wire x).val - 2) % keyB c hU) =
            Aliased.srcKeyP c pi c'.val := by
        rw [hw, hm, Aliased.keyIdxP_some hb, hkey']
      have h2 : 2 ≤ ((Const.Prog.instC (keyProgP c pi hU n) outs u).wire x).val := by rw [hw]; omega
      have hlt : ((Const.Prog.instC (keyProgP c pi hU n) outs u).wire x).val < 2 + n * keyB c hU := by
        rw [hw]; exact Placed.source_lt n _ u.isLt b
      rcases keysRowP_keyProgP hkeys u i c' hc' hc'n with h0 | ⟨q, byte, bt, hk, hq, hlw, hlb⟩ | ⟨q, t, hk⟩
      · exact absurd (hkey'.trans h0) hnz
      · exact absurd hr (keyRowOfP_ne_none h2 hlt (q := q) (byte := byte) (bit := bt) (hgk.trans hk) hq hlw hlb)
      · obtain ⟨u', hu', hp⟩ := keyPubOf_of h2 hlt (hgk.trans hk)
        have hu : u' = u := Fin.ext (by rw [hu', hw, hd])
        subst hu
        exact .inr (.inr (.inl ⟨q, t, hp⟩))
  · by_cases hone : x = ((keyProgP c pi hU n).rows u).one
    · refine .inr (.inl ?_)
      rw [hone, Const.Prog.instC_wire_one]
      exact Fin.ext (oneAt_uniform_val (k := Placed.inGates n (keyB c hU)) (R := Placed.circuitRows c hU)
        (ins := Placed.insOf n (keyB c hU) (Aliased.keyIdxP c pi (keyB c hU))) (one := Placed.oneGate n (keyB c hU))
        (hone := Placed.insOf_ne_one n (keyB c hU) (Aliased.keyIdxP c pi (keyB c hU))) n u)
    · have hne : x.val ≠ ((keyProgP c pi hU n).rows u).nIn + ((keyProgP c pi hU n).rows u).nComp := fun he =>
        hone (Fin.ext he)
      have hxw := x.isLt
      have hi : x.val - ((keyProgP c pi hU n).rows u).nIn < ((keyProgP c pi hU n).rows u).nComp := by
        change x.val < ((keyProgP c pi hU n).rows u).nIn + ((keyProgP c pi hU n).rows u).nComp + 1 at hxw
        omega
      have hx' : x = ((keyProgP c pi hU n).rows u).comp ⟨_, hi⟩ :=
        Fin.ext (by simp only [FlockSoundness.Rows.comp]; omega)
      have hm : (keyProgP c pi hU n).unit ((Const.Prog.instC (keyProgP c pi hU n) outs u).wire x) = some u := by
        have := (Const.Prog.instC (keyProgP c pi hU n) outs u).mem ⟨_, hi⟩
        rw [← hx'] at this
        exact this
      refine .inr (.inr (.inr ⟨hm, fun h => ?_, fun h => ?_⟩))
      · rw [h, hgate_in (by show 0 < 2 + n * keyB c hU; omega)] at hm
        cases hm
      · rw [h, hgate_in (by show 1 < 2 + n * keyB c hU; omega)] at hm
        cases hm

end FlockSoundness.Discharge.Layout
