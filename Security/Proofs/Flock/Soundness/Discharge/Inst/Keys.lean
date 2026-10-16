import Proofs.Flock.Soundness.Discharge.Inst.Prog
import Proofs.Flock.Soundness.Discharge.Layout.KeyProg

/-!
# `lay` and `rs` at the program of the whole template instance

`Layout.keyProg_copies`, `keyProg_bcPinned` and `keyProg_valPinned` at `instProg c n` (`Inst/Prog.lean`) in place of
`keyProg`. A stacked unit input a wire fills is no input of `instProg`: it is the computed gate of its column, whose row
copies the wire's source (`Inst/Rows.lean`). So a composite input's key is never `Src.wire` (`keyOf_ne_wire`), and
`InstKeysRow`, the hypothesis `instProg_valPinned` takes in place of `KeysRow`, asks a row key (`RowKey`) only of the
stacked inputs no wire fills: it holds where a computed row reads a stacked input with `srcKey c x = .wire n i col`.

* `row_of_wire_key`: `copies_of_key`'s `Src.wire n i col`, resolved: that stacked input's column copies column `col` of
  net `n`'s slot `i` (`feed_of_wire_key`), which sits where the key's column does (`pos_wire_key`);
* `instRowOf_wire`: a column of unit `u`'s own instance on a row bit of unit `u'` is a composite input of `u = u'` with
  that message key;
* `instProg_copies`: Δ's copies (`copy_site_inst` at each drawn unit's template instance and block);
* `instProg_bcPinned`: the rows' `b ‖ c` is the registration's (`bc_site_of`), at each drawn unit's site (`BcSite`, which
  `Layout.bcSite_keyProg` gives here too);
* `instProg_valPinned`: the single gates hold the registration's bits, when every composite input a row reads has a row
  key (`InstKeysRow`).

`instLayout` and `instHmRows` are `lay` and `rs` there.
-/

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Inst

open Model Audit Binding Placed _root_.FlockSoundness.Discharge.Layout

/-! ## A wire-fed input is no input -/

/-- **A stacked unit input no wire fills has no wire key**: `srcKey`'s `Src.wire` is the first wire into the input's
port, which `feed` reads too. -/
theorem srcKey_ne_wire {c : Flock.Circuit} {x : ℕ}
    (hf : feed c c.unitNet (x / netIn c c.unitNet) (x % netIn c c.unitNet) = none) (n i col : ℕ) :
    Aliased.srcKey c x ≠ .wire n i col := by
  intro h
  unfold Aliased.srcKey at h
  split at h
  · rename_i hm
    unfold Aliased.srcAt at h
    split at h
    · split at h
      · cases h
      · simp only at h
        split at h <;> cases h
    · split at h
      · simp only at h
        split at h <;> cases h
      · split at h
        · rename_i w l hw
          have hwa : wireAt c c.unitNet (x / netIn c c.unitNet) (x % netIn c c.unitNet) = (w :: l).head? :=
            congrArg List.head? hw
          unfold feed at hf
          rw [ite_eq_left (show x % netIn c c.unitNet ∈ portBits (netOf c c.unitNet).inGroups from hm), hwa] at hf
          cases hf
        · cases h
  · cases h

/-- **A composite input's key is never a wire's.** -/
theorem keyOf_ne_wire {c : Flock.Circuit} {ℓ : ℕ} (hℓ : ℓ < (instRows c).nIn) (n i col : ℕ) :
    keyOf c ℓ ≠ .wire n i col := by
  obtain ⟨-, hm⟩ := inCols_getD (c := c) hℓ
  exact srcKey_ne_wire (Option.isNone_iff_eq_none.1 (List.mem_filter.1 hm).2) n i col

/-- **A wire-keyed stacked input is fed its key's column**: `srcKey`'s `Src.wire n i col` is column `col` of net `n`'s
slot `i`, which `feed` copies in. -/
theorem feed_of_wire_key {c : Flock.Circuit} {x n i col : ℕ} (h : Aliased.srcKey c x = .wire n i col) :
    feed c c.unitNet (x / netIn c c.unitNet) (x % netIn c c.unitNet) = some (code n i col) := by
  unfold Aliased.srcKey at h
  split at h
  · rename_i hm
    unfold Aliased.srcAt at h
    split at h
    · split at h
      · cases h
      · simp only at h
        split at h <;> cases h
    · split at h
      · simp only at h
        split at h <;> cases h
      · split at h
        · rename_i w l hw
          have hwa : wireAt c c.unitNet (x / netIn c c.unitNet) (x % netIn c c.unitNet) = (w :: l).head? :=
            congrArg List.head? hw
          injection h with h1 h2 h3
          unfold feed
          rw [ite_eq_left (show x % netIn c c.unitNet ∈ portBits (netOf c c.unitNet).inGroups from hm), hwa]
          subst h1 h2 h3
          rfl
        · cases h
  · cases h

/-- **A wire-keyed stacked input's column copies the wire's source** (`copies_of_key`'s `Src.wire`, resolved): its row
is the unit's local row there with Δ's pairs `(x, x)` and `(x, code n i col)`, in every slot whose rows `rowAt` has. -/
theorem row_of_wire_key {c : Flock.Circuit} {x n i col : ℕ} (h : Aliased.srcKey c x = .wire n i col)
    (hd : c.unitNet < c.nets.size ∧ c.unitNet ≠ c.shaNet ∧ c.unitNet ≠ c.hmNet ∧
      x / netIn c c.unitNet < c.perVu.getD c.unitNet 0 ∧ x % netIn c c.unitNet < netW c c.unitNet)
    (hW : 0 < netIn c c.unitNet) (hc : netIn c c.unitNet ≤ (netOf c c.unitNet).constPos) :
    row c (inCode c x) = some (some
      (reads c c.unitNet (x / netIn c c.unitNet)
          (addSelf (entA c c.unitNet (x % netIn c c.unitNet)) (x % netIn c c.unitNet)) ++ [code n i col],
        reads c c.unitNet (x / netIn c c.unitNet)
          (addSelf (entB c c.unitNet (x % netIn c c.unitNet)) (x % netIn c c.unitNet)) ++ [code n i col])) := by
  have ho : x % netIn c c.unitNet < netIn c c.unitNet := Nat.mod_lt _ hW
  unfold row inCode
  simp only [codeNet_code, codeSlot_code, codeOff_code]
  unfold rowAt
  rw [ite_eq_left hd, ite_eq_right (by omega), ite_eq_right (by omega), feed_of_wire_key h]

/-- **The source column sits at the key's place**: `pos` of `code n i col` is `Aliased.realize` of `Src.wire n i col`,
the column `copies_of_key` aliases a wire-keyed stacked input to. -/
theorem pos_wire_key (c : Flock.Circuit) (g n i col : ℕ) :
    pos c g (code n i col) = Aliased.realize c g (.wire n i col) := by
  rw [pos_code]
  rfl

/-! ## The program's gates -/

variable (c : Flock.Circuit) (n : ℕ)

/-- The number of composite inputs. -/
abbrev instB : ℕ := (instRows c).nIn

/-- The gates of `instProg c n`. -/
abbrev instN : ℕ := gates (inGates n (instB c)) (instRows c) n

theorem instN_pos : 0 < instN c n :=
  lt_of_lt_of_le (by show 0 < 2 + n * instB c; omega) (le_gates _ _ n)

variable (rowW : Fin c.ports.size → ℕ)

/-- **The row bits of `instProg`**: input gate `2 + u·B + b` is bit `8·byte + bit` of unit `u`'s row port `q` when
composite input `b`'s Δ key is message bit `bit < 8` of byte `byte` of row port `q`, within the row's `rowW q` bits. -/
def instRowOf (g : Fin (instN c n)) : Option (Fin n × (q : Fin c.ports.size) × Fin (rowW q)) :=
  if h : 2 ≤ g.val ∧ g.val < 2 + n * instB c then
    match keyOf c ((g.val - 2) % instB c) with
    | .msg q byte bit =>
      if hq : q < c.ports.size then
        if hk : 8 * byte + bit < rowW ⟨q, hq⟩ ∧ bit < 8 then
          some (⟨(g.val - 2) / instB c, (Nat.div_lt_iff_lt_mul (Nat.pos_of_ne_zero fun h0 => by
            rw [h0, Nat.mul_zero] at h; omega)).2 (by omega)⟩, ⟨⟨q, hq⟩, ⟨8 * byte + bit, hk.1⟩⟩)
        else none
      else none
    | _ => none
  else none

variable {c n rowW}

theorem instRowOf_spec {g : Fin (instN c n)} {u : Fin n} {q : Fin c.ports.size} {k : Fin (rowW q)}
    (h : instRowOf c n rowW g = some (u, ⟨q, k⟩)) :
    2 ≤ g.val ∧ g.val < 2 + n * instB c ∧ u.val = (g.val - 2) / instB c ∧
      keyOf c ((g.val - 2) % instB c) = .msg q.val (k.val / 8) (k.val % 8) := by
  unfold instRowOf at h
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
          congr 1 <;> dsimp only <;> omega
        · exact absurd h (by simp)
      · exact absurd h (by simp)
    all_goals exact absurd h (by simp)
  · exact absurd h (by simp)

/-- The first composite input with an input's key has that key. -/
theorem instIdx_some {m : ℕ} {x b : Fin m} (h : instIdx c m x = some b) : keyOf c b.val = keyOf c x.val := by
  unfold instIdx at h
  split at h
  · cases h
  · simpa using List.find?_some h

theorem rows_instProg (u : Fin n) : (instProg c n).rows u = instRows c :=
  rows_uniform _ _ _ _ (insOf_ne_one n _ (instIdx c (instB c))) n u

/-- **A column of a unit's own instance on a row bit** is a composite input of that unit, whose Δ key is the row bit's
message bit. -/
theorem instRowOf_wire (outs : Finset (Fin (instN c n))) {u u' : Fin n} (x : Fin ((instProg c n).rows u).w)
    {q : Fin c.ports.size} {k : Fin (rowW q)}
    (h : instRowOf c n rowW ((Const.Prog.instC (instProg c n) outs u).wire x) = some (u', ⟨q, k⟩)) :
    u' = u ∧ x.val < ((instProg c n).rows u).nIn ∧ keyOf c x.val = .msg q.val (k.val / 8) (k.val % 8) := by
  obtain ⟨h2, hlt, hu, hkey⟩ := instRowOf_spec h
  have e := congrArg FlockSoundness.Rows.nIn (rows_instProg (c := c) u)
  have hx : x.val < ((instProg c n).rows u).nIn := by
    by_contra hx
    by_cases hone : x = ((instProg c n).rows u).one
    · rw [hone, Const.Prog.instC_wire_one] at h2
      have h0 : (Const.Prog.oneAt (instProg c n) u).val = 0 :=
        oneAt_uniform_val (k := inGates n (instB c)) (R := instRows c)
          (ins := insOf n (instB c) (instIdx c (instB c))) (one := oneGate n (instB c))
          (hone := insOf_ne_one n (instB c) (instIdx c (instB c))) n u
      omega
    · have hne : x.val ≠ ((instProg c n).rows u).nIn + ((instProg c n).rows u).nComp := fun he =>
        hone (Fin.ext he)
      have hxw := x.isLt
      have hi : x.val - ((instProg c n).rows u).nIn < ((instProg c n).rows u).nComp := by
        change x.val < ((instProg c n).rows u).nIn + ((instProg c n).rows u).nComp + 1 at hxw
        omega
      have hx' : x = ((instProg c n).rows u).comp ⟨_, hi⟩ := Fin.ext (by simp only [FlockSoundness.Rows.comp]; omega)
      have hm := (Const.Prog.instC (instProg c n) outs u).mem ⟨_, hi⟩
      rw [← hx'] at hm
      have hnone : (instProg c n).unit ((Const.Prog.instC (instProg c n) outs u).wire x) = none :=
        unit_uniform_input (k := inGates n (instB c)) (R := instRows c)
          (ins := insOf n (instB c) (instIdx c (instB c))) (one := oneGate n (instB c))
          (hone := insOf_ne_one n (instB c) (instIdx c (instB c))) n _ (by
            show _ < 2 + n * instB c; omega)
      have : (instProg c n).unit ((Const.Prog.instC (instProg c n) outs u).wire x) = some u := hm
      rw [hnone] at this
      cases this
  have hxB : x.val < instB c := lt_of_lt_of_eq hx e
  refine ⟨?_, hx, ?_⟩ <;>
  · rw [Const.Prog.instC_wire_input _ _ _ x hx] at h2 hu hkey
    have hv : (Const.Prog.insAt (instProg c n) u x.val).val =
        (insOf n (instB c) (instIdx c (instB c)) u.val ⟨x.val, hxB⟩).val :=
      insAt_uniform n u x.val hxB
    rw [hv, insOf_of_lt (nU := n) (B := instB c) (src := instIdx c (instB c)) u.isLt] at h2 hu hkey
    rcases hb : instIdx c (instB c) ⟨x.val, hxB⟩ with _ | b
    · rw [hb] at h2
      exact absurd h2 (by simp [srcGate, zeroGate])
    · rw [hb] at hu hkey
      simp only [srcGate] at hu hkey
      have hB : 0 < instB c := lt_of_le_of_lt (Nat.zero_le _) hxB
      have hd : (2 + u.val * instB c + b.val - 2) / instB c = u.val := by
        rw [show 2 + u.val * instB c + b.val - 2 = b.val + u.val * instB c by omega,
          Nat.add_mul_div_right _ _ hB, Nat.div_eq_of_lt b.isLt, Nat.zero_add]
      have hm : (2 + u.val * instB c + b.val - 2) % instB c = b.val := by
        rw [show 2 + u.val * instB c + b.val - 2 = b.val + u.val * instB c by omega,
          Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt b.isLt]
      first
        | exact Fin.ext (hu.trans hd)
        | rw [hm, instIdx_some hb] at hkey; exact hkey

/-- **A message-keyed composite input's row bit is in the row.** -/
theorem instRowOf_ne_none {g : Fin (instN c n)} (h2 : 2 ≤ g.val) (hlt : g.val < 2 + n * instB c) {q byte bit : ℕ}
    (hk : keyOf c ((g.val - 2) % instB c) = .msg q byte bit) (hq : q < c.ports.size)
    (hw : 8 * byte + bit < rowW ⟨q, hq⟩) (hb : bit < 8) : instRowOf c n rowW g ≠ none := by
  unfold instRowOf
  simp only [h2, hlt, and_self, ↓reduceDIte, hk, hq, hw, hb]
  exact Option.some_ne_none _

variable (c rowW) in
/-- **Every composite input a computed row reads has a row key** (`RowKey`, at the stacked input it is). `KeysRow` at
the template instance's rows: a stacked input a wire fills is no composite input (`keyOf_ne_wire`), so it is not asked
here. -/
def InstKeysRow : Prop :=
  ∀ (i : Fin (instRows c).nComp) (x : Fin (instRows c).w), x ∈ (instRows c).a i ++ (instRows c).b i →
    x.val < (instRows c).nIn → RowKey c rowW ((inX c).getD x.val 0)

/-- `InstKeysRow` at a unit of `instProg`, whose rows are the template instance's. -/
theorem keysRow_instProg (hkeys : InstKeysRow c rowW) (u : Fin n) (i : Fin ((instProg c n).rows u).nComp)
    (x : Fin ((instProg c n).rows u).w) (hx : x ∈ ((instProg c n).rows u).a i ++ ((instProg c n).rows u).b i)
    (hxn : x.val < ((instProg c n).rows u).nIn) : RowKey c rowW ((inX c).getD x.val 0) := by
  have e : (instProg c n).rows u = instRows c := rows_instProg u
  revert i x
  generalize (instProg c n).rows u = R at e ⊢
  subst e
  exact hkeys

/-- The constant's gate of `instProg`. -/
def instOneG : Fin (instN c n) := Fin.castLE (le_gates _ _ n) (oneGate n (instB c))

/-- The zero's gate of `instProg`. -/
def instZeroG : Fin (instN c n) := Fin.castLE (le_gates _ _ n) (zeroGate n (instB c))

/-- **The registration's bit at a single gate**: 1 at the constant, 0 at the zero, `outBit R g` elsewhere. -/
def instGate {Reg : Type} (outBit : Reg → Fin (instN c n) → Bool) (R : Reg) (g : Fin (instN c n)) : Bool :=
  if g = instOneG (c := c) (n := n) then true
  else if g = instZeroG (c := c) (n := n) then false else outBit R g

theorem instZeroG_ne_instOneG : instZeroG (c := c) (n := n) ≠ instOneG := fun h =>
  absurd (congrArg Fin.val h) (by simp [instZeroG, instOneG, zeroGate, oneGate])

/-- **A single committed gate of a unit's own instance** is the zero, the constant, or one of the unit's computed gates,
when every composite input a row reads has a row key (`InstKeysRow`). -/
theorem inst_single_gate (outs : Finset (Fin (instN c n))) (hkeys : InstKeysRow c rowW) {u : Fin n}
    {g : Fin (instN c n)}
    (hgx : ∃ x : Fin ((instProg c n).rows u).w, (Const.Prog.instC (instProg c n) outs u).wire x = g)
    (hg : g ∈ ((instProg c n).partition outs).io u) (hr : instRowOf c n rowW g = none) :
    g = instZeroG ∨ g = instOneG ∨ ((instProg c n).unit g = some u ∧ g ≠ instOneG ∧ g ≠ instZeroG) := by
  obtain ⟨x, rfl⟩ := hgx
  have e := congrArg FlockSoundness.Rows.nIn (rows_instProg (c := c) u)
  have hgate_in : ∀ {g : Fin (instN c n)}, g.val < 2 + n * instB c → (instProg c n).unit g = none :=
    fun {g} hg => unit_uniform_input (k := inGates n (instB c)) (R := instRows c)
      (ins := insOf n (instB c) (instIdx c (instB c))) (one := oneGate n (instB c))
      (hone := insOf_ne_one n (instB c) (instIdx c (instB c))) n g hg
  by_cases hx : x.val < ((instProg c n).rows u).nIn
  · have hxB : x.val < instB c := lt_of_lt_of_eq hx e
    have hw : ((Const.Prog.instC (instProg c n) outs u).wire x).val =
        (srcGate n (instB c) u.isLt (instIdx c (instB c) ⟨x.val, hxB⟩)).val := by
      have hv : (Const.Prog.insAt (instProg c n) u x.val).val =
          (insOf n (instB c) (instIdx c (instB c)) u.val ⟨x.val, hxB⟩).val :=
        insAt_uniform n u x.val hxB
      rw [Const.Prog.instC_wire_input _ _ _ x hx, hv,
        insOf_of_lt (nU := n) (B := instB c) (src := instIdx c (instB c)) u.isLt]
    rcases hb : instIdx c (instB c) ⟨x.val, hxB⟩ with _ | b
    · rw [hb] at hw
      exact .inl (Fin.ext hw)
    · rw [hb] at hw
      simp only [srcGate] at hw
      have hnz : keyOf c x.val ≠ .zero := fun h => by
        rw [(instIdx_eq_none (c := c) (⟨x.val, hxB⟩ : Fin (instB c))).2 h] at hb
        cases hb
      have hB : 0 < instB c := lt_of_le_of_lt (Nat.zero_le _) hxB
      have hm : (2 + u.val * instB c + b.val - 2) % instB c = b.val := by
        rw [show 2 + u.val * instB c + b.val - 2 = b.val + u.val * instB c by omega,
          Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt b.isLt]
      have hnone : (instProg c n).unit ((Const.Prog.instC (instProg c n) outs u).wire x) = none :=
        hgate_in (by rw [hw]; exact source_lt n _ u.isLt b)
      obtain ⟨i, c', hc', hwc⟩ := exists_read_of_io (Const.Prog.instC (instProg c n) outs u) hg
        (fun h => by
          have h' : (instProg c n).unit ((Const.Prog.instC (instProg c n) outs u).wire x) = some u := h
          rw [hnone] at h'
          cases h')
      have hc'n : c'.val < ((instProg c n).rows u).nIn := by
        rcases (Const.Prog.instC (instProg c n) outs u).shared c' x hwc with rfl | ⟨h, -⟩
        · exact hx
        · exact h
      have hkey' := instProg_wire_key c outs u x c' hx hc'n hwc.symm
      rcases keysRow_instProg hkeys u i c' hc' hc'n with h0 | ⟨q, byte, bt, hk, hq, hlw, hlb⟩
      · exact absurd (hkey'.trans h0) hnz
      · refine absurd hr (instRowOf_ne_none (by rw [hw]; omega) (by rw [hw]; exact source_lt n _ u.isLt b)
          (q := q) (byte := byte) (bit := bt) ?_ hq hlw hlb)
        rw [hw, hm, instIdx_some hb, hkey']
        exact hk
  · by_cases hone : x = ((instProg c n).rows u).one
    · refine .inr (.inl ?_)
      rw [hone, Const.Prog.instC_wire_one]
      exact Fin.ext (oneAt_uniform_val (k := inGates n (instB c)) (R := instRows c)
        (ins := insOf n (instB c) (instIdx c (instB c))) (one := oneGate n (instB c))
        (hone := insOf_ne_one n (instB c) (instIdx c (instB c))) n u)
    · have hne : x.val ≠ ((instProg c n).rows u).nIn + ((instProg c n).rows u).nComp := fun he =>
        hone (Fin.ext he)
      have hxw := x.isLt
      have hi : x.val - ((instProg c n).rows u).nIn < ((instProg c n).rows u).nComp := by
        change x.val < ((instProg c n).rows u).nIn + ((instProg c n).rows u).nComp + 1 at hxw
        omega
      have hx' : x = ((instProg c n).rows u).comp ⟨_, hi⟩ :=
        Fin.ext (by simp only [FlockSoundness.Rows.comp]; omega)
      have hm : (instProg c n).unit ((Const.Prog.instC (instProg c n) outs u).wire x) = some u := by
        have := (Const.Prog.instC (instProg c n) outs u).mem ⟨_, hi⟩
        rw [← hx'] at this
        exact this
      refine .inr (.inr ⟨hm, fun h => ?_, fun h => ?_⟩)
      · rw [h, hgate_in (by show 0 < 2 + n * instB c; omega)] at hm
        cases hm
      · rw [h, hgate_in (by show 1 < 2 + n * instB c; omega)] at hm
        cases hm

/-! ## At a drawn unit's site -/

/-- **A drawn unit's message-keyed composite input holds its row's bit**: in a satisfying witness of an accepted
statement, at block `o` and template instance `g`, the composite input `x` whose Δ key is message bit `bit` of byte
`byte` of row port `q` holds the witness's bit at that message column (`Typed.msgCol`), when the column is in the block.
`Layout.copy_site` at the template instance's rows. -/
theorem copy_site_inst {T : Model.Statement} (a : Accepted T) {z : Witness T.m} (hz : T.Satisfies z)
    (o : Fin (2 ^ T.nbl)) {g : ℕ} (hg : g < a.st.c.g) (x : Fin (instRows a.st.c).w) (hx : x.val < (instRows a.st.c).nIn)
    {q byte bit : ℕ} (hk : keyOf a.st.c x.val = .msg q byte bit)
    (hin : Typed.msgCol a.st.c g q byte bit < 2 ^ a.st.c.kLog) :
    T.block z o (Fin.cast a.kLog_eq (composePlace (a.layout.model a.regs) (logical a.st.c) (logical_ordered a.st.c)
        (pos a.st.c g) (inst_hb a.layout (Aliased.setupH_unitFacts a.ht a.h) a.regs hg) x)) =
      z (siteAddr a o (Typed.msgCol a.st.c g q byte bit)) := by
  obtain ⟨tags, cf, pf, coins, tables, draw, partition, program, st, ht, h, regs, hT⟩ := a
  subst hT
  simp only [Fin.cast_eq_self]
  rw [inst_block (setupH_layout ht h) (Aliased.setupH_unitFacts ht h) regs hg x hx hz o]
  obtain ⟨-, hm⟩ := inCols_getD (c := st.c) hx
  have hk' : Aliased.srcKey st.c ((inX st.c).getD x.val 0) = .msg q byte bit := hk
  rcases Aliased.copyPos_key (setupH_layout ht h) regs (Aliased.setupH_unitFacts ht h) hg (inX_lt hm)
      (by rw [hk']; simp) (fun k off => by rw [hk']; simp) with he | ⟨hn, -⟩
  · simp only [siteAddr]
    show z _ = z _
    congr 2
    apply Fin.ext
    rw [he, hk']
    simp [Aliased.realize, inBlockOr, hin]
    rfl
  · exact absurd (by rw [hk']; exact hin) hn

section places

variable {Reg : Type} {mPts : ℕ} {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
  (si : Sites plan tab)

/-- **Row port `q`'s salt at a drawn unit's site**: `saltCol` at the unit's template instance, in its block
(`Layout.keySaltAddr` on `instProg`'s gates). -/
noncomputable def instSaltAddr (saltCol : Flock.Circuit → ℕ → ℕ → ℕ → ℕ → ℕ) (S : Finset (Fin n)) (R : Reg)
    (u : Fin n) (q : (Fin n × Fin c.ports.size) ⊕ Fin (instN c n)) (j : Fin 192) (t : Fin 8) :
    Fin (2 ^ (Partition.tabOf plan tab S R u).S.m) :=
  if h : u ∈ S then
    match q with
    | .inl (_, q) =>
      siteAddr (si.acc S R u h) (si.o S R u h) (saltCol (si.acc S R u h).st.c (si.vu S R u h) q.val j.val t.val)
    | .inr _ => ⟨0, Nat.two_pow_pos _⟩
  else ⟨0, Nat.two_pow_pos _⟩

variable {si}
variable (hparse : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S),
    HmRow.parse (si.acc S R u h).tags (si.acc S R u h).cf (si.acc S R u h).tables = .ok c)
  (outs : Finset (Fin (instN c n)))

/-- The places of the program of the whole template instance (`Sites.progPlaces_instProg`). -/
noncomputable abbrev instPlace := (Sites.progPlaces_instProg hparse outs).derived.place

variable (rowW) (saltCol : Flock.Circuit → ℕ → ℕ → ℕ → ℕ → ℕ) (popOf : Reg → Public)
  (outBit : Reg → Fin (instN c n) → Bool)

/-- **The readers at `instProg`'s sites**: `Layout.keyReaders` on `instProg`'s row bits (`instRowOf`) and single gates
(`instGate`). -/
noncomputable def instReaders : Readers ((instProg c n).circuit outs) plan tab :=
  Readers.ofRows (instPlace hparse outs) (instN_pos c n) rowW (W := Finset.univ.sup rowW + 1)
    (fun q => (Finset.le_sup (Finset.mem_univ q)).trans (Nat.le_succ _)) (Nat.le_add_left 1 _) (instRowOf c n rowW)
    (keyRowAddr si) (instSaltAddr si saltCol) (keyBcAddr si)
    (fun R u q => ((popOf R).digest u.val q.val c.ports.size).data.toList) (instGate outBit)

variable {rowW}

include hparse in
/-- **Δ's copies at `instProg`'s sites**: each column of a drawn unit on a row bit holds the witness's bit at the
message column of its key (`copy_site_inst`), in the block by the parse (`msgCol_lt_of_parse`). -/
theorem instProg_copies :
    (instReaders rowW hparse outs saltCol popOf outBit).Copies (instPlace hparse outs) := by
  refine ofRows_copies _ fun S R u hu M hM x hx u' q k hr => ?_
  obtain ⟨-, hxn, hkey⟩ := instRowOf_wire outs x hr
  have hc := (si.acc S R u hu).c_eq (hparse S R u hu)
  have hrw := Sites.rows_instProg hparse S R u hu
  have hk' : keyOf (si.acc S R u hu).st.c x.val = .msg q.val (k.val / 8) (k.val % 8) := by
    rw [hc]; exact hkey
  have hin' : Typed.msgCol (si.acc S R u hu).st.c (si.vu S R u hu) q.val (k.val / 8) (k.val % 8) <
      2 ^ (si.acc S R u hu).st.c.kLog := by
    have hv := si.hvu S R u hu
    rw [hc] at hv ⊢
    exact msgCol_lt_of_parse (hparse S R u hu) hv q.isLt _ _ (Nat.mod_lt _ (by norm_num))
  have := copy_site_inst (si.acc S R u hu) hM (si.o S R u hu) (si.hvu S R u hu)
    (Fin.cast (congrArg FlockSoundness.Rows.w hrw) x) (lt_of_lt_of_eq hxn (congrArg FlockSoundness.Rows.nIn hrw))
    hk' hin'
  simp only [keyRowAddr, hu, ↓reduceDIte]
  exact this

include hparse in
/-- **The rows' `b ‖ c` at `instProg`'s sites is the registration's** (`bc_site_of`, at every site whose rank is the
unit). -/
theorem instProg_bcPinned
    (hsite : ∀ S R u (h : u ∈ S) M, Binding.Sat plan tab S R u M →
      BcSite (si.acc S R u h) (popOf R) (si.o S R u h) (si.vu S R u h) u.val) :
    (instReaders rowW hparse outs saltCol popOf outBit).BcPinned (P := (instProg c n).partition outs) := by
  refine ofRows_bcPinned _ fun S R u hu M hM g hg u' q k hr => ?_
  obtain ⟨x, rfl⟩ := exists_col_of_io ((instPlace hparse outs) S R u hu).inst hg
  obtain ⟨hu', -, -⟩ := instRowOf_wire outs x hr
  rw [hu']
  have hc := (si.acc S R u hu).c_eq (hparse S R u hu)
  have hnp : (si.acc S R u hu).st.c.ports.size = c.ports.size := by rw [hc]
  have := bc_site_of (hsite S R u hu M hM) hM (si.hvu S R u hu) (p := q.val) (by rw [hnp]; exact q.isLt)
  rw [hnp] at this
  unfold keyBcAddr
  simp only [hu, ↓reduceDIte]
  exact this

include hparse in
/-- **The single gates hold the registration's bits** at `instProg`'s sites: 1 at the constant (`hc1`), 0 at the zero
(`hz1`), and at a unit's committed gate the registration's bit (`hout`), when every composite input a row reads has a
row key (`InstKeysRow`). -/
theorem instProg_valPinned (hkeys : InstKeysRow c rowW)
    (hc1 : ((instProg c n).partition outs).ConstCols plan tab (instPlace hparse outs) {instOneG (c := c) (n := n)})
    (hz1 : ((instProg c n).partition outs).ZeroCols plan tab (instPlace hparse outs) {instZeroG (c := c) (n := n)})
    (hout : ∀ S R u, u ∈ S → ∀ M, Binding.Sat plan tab S R u M → ∀ g ∈ ((instProg c n).partition outs).io u,
      (instProg c n).unit g = some u →
        bit (Binding.wit plan tab S R u M (gateAt (instPlace hparse outs) S R u g)) = outBit R g) :
    (instReaders rowW hparse outs saltCol popOf outBit).ValPinned (P := (instProg c n).partition outs) := by
  refine ofRows_valPinned _ fun S R u hu M hM g hg hr => ?_
  have hpub := gate_pub (instPlace hparse outs) hc1 hz1 hu hM hg
  obtain ⟨x, hx⟩ := exists_col_of_io ((instPlace hparse outs) S R u hu).inst hg
  change bit _ = instGate outBit R g
  rcases inst_single_gate outs hkeys ⟨x, hx⟩ hg hr with rfl | rfl | ⟨hm, h1, h2⟩
  · rw [hpub.2 (Finset.mem_singleton_self _)]
    simp only [instGate, instZeroG_ne_instOneG, ↓reduceIte]
  · rw [hpub.1 (Finset.mem_singleton_self _)]
    simp only [instGate, ↓reduceIte]
  · rw [hout S R u hu M hM g hg hm]
    simp only [instGate, h1, h2, ↓reduceIte]

variable (rowW) in
/-- **`lay` at `instProg`'s places**: the readers' layout (`Readers.layout`), with Δ's copies from `instProg_copies`. -/
noncomputable def instLayout (encO : BitVal (Finset.univ.sup rowW + 1) → List UInt8) :
    Binding.Layout ((instProg c n).partition outs) plan tab
      (((instProg c n).partition outs).placeDecoder plan tab (instPlace hparse outs)) :=
  (instReaders rowW hparse outs saltCol popOf outBit).layout (instPlace hparse outs) encO
    (instProg_copies hparse outs saltCol popOf outBit)

variable (rowW) in
/-- **`rs` at `instProg`'s places**: the readers' rows (`Readers.rows`), from `instProg_copies`, `instProg_bcPinned`
and `instProg_valPinned`. -/
noncomputable def instHmRows (enc : BitVal (Finset.univ.sup rowW + 1) → List UInt8)
    (enc_inj : Function.Injective enc) (encO : BitVal (Finset.univ.sup rowW + 1) → List UInt8)
    (encO_inj : Function.Injective encO)
    (hsite : ∀ S R u (h : u ∈ S) M, Binding.Sat plan tab S R u M →
      BcSite (si.acc S R u h) (popOf R) (si.o S R u h) (si.vu S R u h) u.val)
    (hkeys : InstKeysRow c rowW)
    (hc1 : ((instProg c n).partition outs).ConstCols plan tab (instPlace hparse outs) {instOneG (c := c) (n := n)})
    (hz1 : ((instProg c n).partition outs).ZeroCols plan tab (instPlace hparse outs) {instZeroG (c := c) (n := n)})
    (hout : ∀ S R u, u ∈ S → ∀ M, Binding.Sat plan tab S R u M → ∀ g ∈ ((instProg c n).partition outs).io u,
      (instProg c n).unit g = some u →
        bit (Binding.wit plan tab S R u M (gateAt (instPlace hparse outs) S R u g)) = outBit R g) :
    Binding.HmRows (instLayout rowW hparse outs saltCol popOf outBit encO) :=
  (instReaders rowW hparse outs saltCol popOf outBit).rows (instPlace hparse outs) enc enc_inj encO encO_inj
    (instProg_copies hparse outs saltCol popOf outBit)
    (instProg_bcPinned hparse outs saltCol popOf outBit hsite)
    (instProg_valPinned hparse outs saltCol popOf outBit hkeys hc1 hz1 hout)

end places

end FlockSoundness.Discharge.Inst
