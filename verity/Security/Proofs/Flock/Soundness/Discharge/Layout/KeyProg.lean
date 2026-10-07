import Proofs.Flock.Soundness.Discharge.Layout.Site
import Proofs.Flock.Soundness.Discharge.Layout.OfRows
import Proofs.Flock.Soundness.Discharge.Layout.QSort

set_option Elab.async false

/-!
# `lay` and `rs` at the circuit's program on Δ's source keys

`Placed.keyProg c hU n` is the program of `n` units of the circuit `c`'s template instance rows, each unit's input `x` on the input gate
of its Δ key (`Aliased.keyIdx`): gate `2 + u·B + b` for unit `u` and the first input `b` with `x`'s key, gate 1 for the
zero, gate 0 for the constant. Its row bits (`keyRowOf`) are the input gates whose key is message bit `bit` of byte
`byte` of a row port `q` (`Aliased.Src.msg`): bit `8·byte + bit` of unit `u`'s row `q`. Every other wire is a single
gate. At [e2e-placed]'s places (`Placed.Sites.progPlaces_keyProg`):

* `keyRowOf_wire`: a column of unit `u`'s own instance on a row bit of unit `u'` is an input column of `u = u'` with that
  message key;
* `keyProg_copies`: Δ's copies (`copy_site` at each drawn unit's template instance and block; each message column is in
  the block by the parse, `msgCol_lt_of_parse`);
* `keyProg_bcPinned`: the rows' `b ‖ c` is the registration's, the population's public file (`bc_site`), at each drawn
  unit's site (`BcSite`: the site's rank is the unit, and `bc_site`'s conditions; `bcSite_keyProg` gives the template
  instance's bound from the parse when the unit returns the outputs, `vu_lt_pow2`);
* `keyProg_valPinned`: the single gates hold the registration's bits: 1 at the constant, 0 at the zero, and at the units'
  committed gates the registration's bit (`hout`, the returned outputs' pin), when every input a computed row reads has
  the zero or a row bit as its key (`KeysRow`).

`keyLayout` and `keyRows` are `lay` and `rs` there.
-/

namespace FlockSoundness.Discharge.Layout

open Flock FlockLevel3 Model Audit Binding Placed

/-! ## The program's gates -/

section uniform

variable {k : ℕ} {R : FlockSoundness.Rows} {ins : ℕ → Fin R.nIn → Fin k} {one : Fin k} {hone : ∀ u c, ins u c ≠ one}

/-- A program input of `uniform` is no unit's gate. -/
theorem unit_uniform_input : ∀ (n : ℕ) (g : Fin (Placed.gates k R n)), g.val < k →
    (Placed.uniform k R ins one hone n).unit g = none
  | 0, _, _ => rfl
  | n + 1, g, hg => by
    have h' : g.val < Placed.gates k R n := lt_of_lt_of_le hg (Placed.le_gates k R n)
    change (FlockSoundness.Prog.snoc (Placed.uniform k R ins one hone n) R _ _ _).unit g = none
    simp only [FlockSoundness.Prog.unit, h', ↓reduceDIte, unit_uniform_input n ⟨g.val, h'⟩ hg, Option.map_none]

/-- `uniform` wires every unit's constant to the input gate `one`. -/
theorem oneAt_uniform_val : ∀ (n : ℕ) (u : Fin n),
    (Const.Prog.oneAt (Placed.uniform k R ins one hone n) u).val = one.val
  | 0, u => u.elim0
  | n + 1, u => by
    induction u using Fin.lastCases with
    | last =>
      change (Const.Prog.oneAt (FlockSoundness.Prog.snoc (Placed.uniform k R ins one hone n) R _ _ _)
        (Fin.last n)).val = _
      rw [Const.Prog.oneAt_last]
      simp
    | cast v =>
      change (Const.Prog.oneAt (FlockSoundness.Prog.snoc (Placed.uniform k R ins one hone n) R _ _ _)
        v.castSucc).val = _
      rw [Const.Prog.oneAt_castSucc]
      simp [oneAt_uniform_val n v]

end uniform

/-- **A message column is in the block**, in a circuit `HmRow.parse` accepted: message bit `b < 8` of byte `byte` of row
port `p` at template instance `g` (`Typed.msgCol`), the untyped twin of the typed path's `msgCol_lt`. -/
theorem msgCol_lt_of_parse {tags : Tags} {bytes : ByteArray} {tables : List (String × ByteArray)} {c : Flock.Circuit}
    (h : HmRow.parse tags bytes tables = .ok c) {g : ℕ} (hg : g < c.g) {p : ℕ} (hp : p < c.ports.size) (byte b : ℕ)
    (hb : b < 8) : Typed.msgCol c g p byte b < 2 ^ c.kLog := by
  have P := parse_facts h
  obtain ⟨ls, hls⟩ := P.check
  have C := check_facts P.typed hls
  have hws := rowSlots_spec c.ports
  have hall := rowSlots_all c.ports
  have hpv := C.perVu_sha
  unfold Typed.msgCol
  rw [show HmRow.byteAt c.ports[p]! byte = ((HmRow.byteAt c.ports[p]! byte).1, byte % 128) from
    Prod.ext rfl (Typed.byteAt_snd _ _)]
  generalize (HmRow.byteAt c.ports[p]! byte).1 = kk
  generalize hrs : HmRow.rowSlots c.ports = rs at hws hall hpv
  obtain ⟨nSha, where_⟩ := rs
  simp only at hws hall hpv ⊢
  split
  swap
  · obtain ⟨hm, hk, -⟩ := rangeOf_spec (C.one_range _ P.hmNet)
    have hc := C.count _ hm _ hk
    have hsl := ((C.bound_slotLog _ hm).2 _ hk).symm
    obtain ⟨-, hW, hports⟩ := (nets_ok P c.hmNet).1
    have hin := C.hm_in
    have hsz : (portCols (c.nets.getD c.hmNet default).2.inGroups).size = 3 := by
      simpa using congrArg Array.size hin
    have hq : ((portCols (c.nets.getD c.hmNet default).2.inGroups)[1]!).2 = 1024 := by
      have h1 := congrArg (fun a => a[1]!) hin
      rw [getElem!_pos _ 1 (by rw [Array.size_map, hsz]; decide), Array.getElem_map] at h1
      rw [getElem!_pos _ 1 (by rw [hsz]; decide), h1]
      rfl
    have hqm := hports _ (getElem_mem_toList' (xs := portCols (c.nets.getD c.hmNet default).2.inGroups) (i := 1)
      (by rw [hsz]; decide))
    have hat : AtSlot c c.hmNet (g * c.perVu.getD c.hmNet 0 + p)
        (((portCols (c.nets.getD c.hmNet default).2.inGroups)[1]!).1 + 64 * (byte % 128 / 8) +
          (56 - 8 * (byte % 128 % 8)) + b)
        (c.slotCol c.hmNet g p + ((portCols (c.nets.getD c.hmNet default).2.inGroups)[1]!).1 +
          64 * (byte % 128 / 8) + 56 - 8 * (byte % 128 % 8) + b) := by
      refine ⟨by rw [hc, C.perVu_hm]; exact Rows.mul_add_lt hg (lt_of_lt_of_le hp (le_nextPow2 _)), ?_, ?_⟩
      · rw [hsl]
        have h1 : 64 * (byte % 128 / 8) ≤ 960 := by omega
        have h2 : 8 * (byte % 128 % 8) ≤ 56 := by omega
        omega
      · unfold Circuit.slotCol
        have h2 : 8 * (byte % 128 % 8) ≤ 56 := by omega
        omega
    exact hat.lt (fun r hr => (C.bound_slotLog r hr).1) hm
  rename_i hkw
  have hent : (where_[p]!)[kk]!.1 < nSha ∧ (where_[p]!)[kk]!.2 < 3 :=
    hall _ (getElem_mem_toList' (by rw [hws]; exact hp)) _ (getElem_mem_toList' hkw)
  obtain ⟨hm, hk, -⟩ := rangeOf_spec (C.one_range _ P.shaNet)
  have hc := C.count _ hm _ hk
  have hsl := ((C.bound_slotLog _ hm).2 _ hk).symm
  obtain ⟨-, hW, hports⟩ := (nets_ok P c.shaNet).1
  obtain ⟨hsz, hsin⟩ := C.sha_in
  generalize hE : (where_[p]!)[kk]! = e at hent ⊢
  obtain ⟨slot, sub⟩ := e
  simp only at hent ⊢
  have hsub : 2 * sub + 1 < 6 := by omega
  have hq : ((portCols (c.nets.getD c.shaNet default).2.inGroups)[2 * sub + 1]!).2 = 1024 := by
    rw [hsin _ hsub]
    have : (2 * sub + 1) % 2 = 1 := by omega
    simp [this]
  have hqm := hports _ (getElem_mem_toList' (by rw [hsz]; exact hsub))
  have hat : AtSlot c c.shaNet (g * c.perVu.getD c.shaNet 0 + slot)
      (((portCols (c.nets.getD c.shaNet default).2.inGroups)[2 * sub + 1]!).1 + 64 * (byte % 128 / 8) +
        (56 - 8 * (byte % 128 % 8)) + b)
      (c.slotCol c.shaNet g slot + ((portCols (c.nets.getD c.shaNet default).2.inGroups)[2 * sub + 1]!).1 +
        64 * (byte % 128 / 8) + 56 - 8 * (byte % 128 % 8) + b) := by
    refine ⟨by rw [hc, hpv]; exact Rows.mul_add_lt hg hent.1, ?_, ?_⟩
    · rw [hsl]
      have h1 : 64 * (byte % 128 / 8) ≤ 960 := by omega
      have h2 : 8 * (byte % 128 % 8) ≤ 56 := by omega
      omega
    · unfold Circuit.slotCol
      have h2 : 8 * (byte % 128 % 8) ≤ 56 := by omega
      omega
  exact hat.lt (fun r hr => (C.bound_slotLog r hr).1) hm

private theorem ite_throw_ok {α β : Type} {c : Prop} [Decidable c] {e : String} {f : α → Except String β}
    {g : Except String β} {v : β} (h : (if c then (throw e >>= f) else g) = .ok v) : ¬c ∧ g = .ok v := by
  by_cases hc : c
  · rw [ite_eq_left hc] at h; cases h
  · rw [ite_eq_right hc] at h; exact ⟨hc, h⟩

/-- `HmRow.check` takes the unit range's count to be a power of two when the unit returns the outputs. -/
theorem unit_count_pow2 {c : Flock.Circuit} {tags : Tags} {ls : Lean.Json} (ht : c.typed = none)
    (h : HmRow.check c tags ls = .ok ()) (ho : c.outNet = c.unitNet) :
    isPow2 (c.ranges.filter (·.kind == Kind.net c.unitNet))[0]!.count = true := by
  unfold HmRow.check at h
  obtain ⟨spans, -, h⟩ := except_bind_ok h
  obtain ⟨u, -, h⟩ := except_bind_ok h
  obtain ⟨-, h⟩ := ite_throw_ok h
  obtain ⟨-, h⟩ := ite_throw_ok h
  obtain ⟨-, h⟩ := ite_throw_ok h
  simp only [ht] at h
  obtain ⟨hur, -⟩ := ite_throw_ok h
  simp only [ho, beq_self_eq_true, Bool.true_and, Bool.or_eq_true, bne_iff_ne, ne_eq, Bool.not_eq_true',
    not_or, Bool.not_eq_false] at hur
  exact hur.2

/-- **Every template instance is below `2^⌊log₂ c.g⌋`** in a circuit `HmRow.parse` accepted whose unit returns the
outputs: there the unit range's `c.g · c.upv` slots are a power of two, so `c.g` is one. (A tail that returns the
outputs has no such check.) -/
theorem vu_lt_pow2 {tags : Tags} {bytes : ByteArray} {tables : List (String × ByteArray)} {c : Flock.Circuit}
    (h : HmRow.parse tags bytes tables = .ok c) (ho : c.outNet = c.unitNet) {g : ℕ} (hg : g < c.g) :
    g < 2 ^ c.g.log2 := by
  have P := parse_facts h
  obtain ⟨ls, hls⟩ := P.check
  have C := check_facts P.typed hls
  have hp := unit_count_pow2 P.typed hls ho
  rw [C.unit_range.2] at hp
  simp only [isPow2, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hp
  have hd : c.g ∣ 2 ^ (c.g * c.upv).log2 := by rw [hp.2]; exact Dvd.intro c.upv rfl
  obtain ⟨i, -, hi⟩ := (Nat.dvd_prime_pow Nat.prime_two).1 hd
  rw [hi, Nat.log2_two_pow, ← hi]
  exact hg

variable (c : Flock.Circuit) (hU : Built c.unit) (n : ℕ)

/-- The number of input columns of the circuit's template instance rows. -/
noncomputable abbrev keyB : ℕ := (Placed.circuitRows c hU).nIn

/-- The gates of `keyProg c hU n`. -/
noncomputable abbrev keyN : ℕ := Placed.gates (Placed.inGates n (keyB c hU)) (Placed.circuitRows c hU) n

theorem keyN_pos : 0 < keyN c hU n :=
  lt_of_lt_of_le (by show 0 < 2 + n * keyB c hU; omega) (Placed.le_gates _ _ n)

variable (rowW : Fin c.ports.size → ℕ)

/-- **The row bits of `keyProg`**: input gate `2 + u·B + b` is bit `8·byte + bit` of unit `u`'s row port `q` when input
`b`'s Δ key is message bit `bit < 8` of byte `byte` of row port `q`, within the row's `rowW q` bits. -/
noncomputable def keyRowOf (g : Fin (keyN c hU n)) : Option (Fin n × (q : Fin c.ports.size) × Fin (rowW q)) :=
  if h : 2 ≤ g.val ∧ g.val < 2 + n * keyB c hU then
    match Aliased.srcKey c ((g.val - 2) % keyB c hU) with
    | .msg q byte bit =>
      if hq : q < c.ports.size then
        if hk : 8 * byte + bit < rowW ⟨q, hq⟩ ∧ bit < 8 then
          some (⟨(g.val - 2) / keyB c hU, (Nat.div_lt_iff_lt_mul (Nat.pos_of_ne_zero fun h0 => by
            rw [h0, Nat.mul_zero] at h; omega)).2 (by omega)⟩, ⟨⟨q, hq⟩, ⟨8 * byte + bit, hk.1⟩⟩)
        else none
      else none
    | _ => none
  else none

variable {c hU n rowW}

theorem keyRowOf_spec {g : Fin (keyN c hU n)} {u : Fin n} {q : Fin c.ports.size} {k : Fin (rowW q)}
    (h : keyRowOf c hU n rowW g = some (u, ⟨q, k⟩)) :
    2 ≤ g.val ∧ g.val < 2 + n * keyB c hU ∧ u.val = (g.val - 2) / keyB c hU ∧
      Aliased.srcKey c ((g.val - 2) % keyB c hU) = .msg q.val (k.val / 8) (k.val % 8) := by
  unfold keyRowOf at h
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

/-- The first input with an input's key has that key. -/
theorem keyIdx_some {m : ℕ} {x b : Fin m} (h : Aliased.keyIdx c m x = some b) :
    Aliased.srcKey c b.val = Aliased.srcKey c x.val := by
  unfold Aliased.keyIdx at h
  split at h
  · cases h
  · simpa using List.find?_some h

/-- **A column of a unit's own instance on a row bit** is an input column of that unit, whose Δ key is the row bit's
message bit. -/
theorem keyRowOf_wire (outs : Finset (Fin (keyN c hU n))) {u u' : Fin n} (x : Fin ((keyProg c hU n).rows u).w)
    {q : Fin c.ports.size} {k : Fin (rowW q)}
    (h : keyRowOf c hU n rowW ((Const.Prog.instC (keyProg c hU n) outs u).wire x) = some (u', ⟨q, k⟩)) :
    u' = u ∧ x.val < ((keyProg c hU n).rows u).nIn ∧ Aliased.srcKey c x.val = .msg q.val (k.val / 8) (k.val % 8) := by
  obtain ⟨h2, hlt, hu, hkey⟩ := keyRowOf_spec h
  have e := congrArg FlockSoundness.Rows.nIn
    (rows_circuitProg c hU n (keyB c hU) (Aliased.keyIdx c (keyB c hU)) u)
  have hx : x.val < ((keyProg c hU n).rows u).nIn := by
    by_contra hx
    by_cases hone : x = ((keyProg c hU n).rows u).one
    · rw [hone, Const.Prog.instC_wire_one] at h2
      have h0 : (Const.Prog.oneAt (keyProg c hU n) u).val = 0 :=
        oneAt_uniform_val (k := Placed.inGates n (keyB c hU)) (R := Placed.circuitRows c hU)
          (ins := Placed.insOf n (keyB c hU) (Aliased.keyIdx c (keyB c hU))) (one := Placed.oneGate n (keyB c hU))
          (hone := Placed.insOf_ne_one n (keyB c hU) (Aliased.keyIdx c (keyB c hU))) n u
      omega
    · have hne : x.val ≠ ((keyProg c hU n).rows u).nIn + ((keyProg c hU n).rows u).nComp := fun he =>
        hone (Fin.ext he)
      have hxw := x.isLt
      have hi : x.val - ((keyProg c hU n).rows u).nIn < ((keyProg c hU n).rows u).nComp := by
        change x.val < ((keyProg c hU n).rows u).nIn + ((keyProg c hU n).rows u).nComp + 1 at hxw
        omega
      have hx' : x = ((keyProg c hU n).rows u).comp ⟨_, hi⟩ := Fin.ext (by simp only [FlockSoundness.Rows.comp]; omega)
      have hm := (Const.Prog.instC (keyProg c hU n) outs u).mem ⟨_, hi⟩
      rw [← hx'] at hm
      have hnone : (keyProg c hU n).unit ((Const.Prog.instC (keyProg c hU n) outs u).wire x) = none :=
        unit_uniform_input (k := Placed.inGates n (keyB c hU)) (R := Placed.circuitRows c hU)
          (ins := Placed.insOf n (keyB c hU) (Aliased.keyIdx c (keyB c hU))) (one := Placed.oneGate n (keyB c hU))
          (hone := Placed.insOf_ne_one n (keyB c hU) (Aliased.keyIdx c (keyB c hU))) n _ (by
            show _ < 2 + n * keyB c hU; omega)
      have : (keyProg c hU n).unit ((Const.Prog.instC (keyProg c hU n) outs u).wire x) = some u := hm
      rw [hnone] at this
      cases this
  have hxB : x.val < keyB c hU := lt_of_lt_of_eq hx e
  refine ⟨?_, hx, ?_⟩ <;>
  · rw [Const.Prog.instC_wire_input _ _ _ x hx] at h2 hu hkey
    have hv : (Const.Prog.insAt (keyProg c hU n) u x.val).val =
        (Placed.insOf n (keyB c hU) (Aliased.keyIdx c (keyB c hU)) u.val ⟨x.val, hxB⟩).val :=
      Placed.insAt_uniform n u x.val hxB
    rw [hv, Placed.insOf_of_lt (nU := n) (B := keyB c hU) (src := Aliased.keyIdx c (keyB c hU)) u.isLt] at h2 hu hkey
    rcases hb : Aliased.keyIdx c (keyB c hU) ⟨x.val, hxB⟩ with _ | b
    · rw [hb] at h2
      exact absurd h2 (by simp [Placed.srcGate, Placed.zeroGate])
    · rw [hb] at hu hkey
      simp only [Placed.srcGate] at hu hkey
      have hB : 0 < keyB c hU := lt_of_le_of_lt (Nat.zero_le _) hxB
      have hd : (2 + u.val * keyB c hU + b.val - 2) / keyB c hU = u.val := by
        rw [show 2 + u.val * keyB c hU + b.val - 2 = b.val + u.val * keyB c hU by omega,
          Nat.add_mul_div_right _ _ hB, Nat.div_eq_of_lt b.isLt, Nat.zero_add]
      have hm : (2 + u.val * keyB c hU + b.val - 2) % keyB c hU = b.val := by
        rw [show 2 + u.val * keyB c hU + b.val - 2 = b.val + u.val * keyB c hU by omega,
          Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt b.isLt]
      first
        | exact Fin.ext (hu.trans hd)
        | rw [hm, keyIdx_some hb] at hkey; exact hkey

/-- **A message-keyed input's row bit is in the row**: when an input's key is message bit `bit < 8` of byte `byte` of a
row port `q` within `rowW q` bits, its gate is a row bit. -/
theorem keyRowOf_ne_none {g : Fin (keyN c hU n)} (h2 : 2 ≤ g.val) (hlt : g.val < 2 + n * keyB c hU) {q byte bit : ℕ}
    (hk : Aliased.srcKey c ((g.val - 2) % keyB c hU) = .msg q byte bit) (hq : q < c.ports.size)
    (hw : 8 * byte + bit < rowW ⟨q, hq⟩) (hb : bit < 8) : keyRowOf c hU n rowW g ≠ none := by
  unfold keyRowOf
  simp only [h2, hlt, and_self, ↓reduceDIte, hk, hq, hw, hb]
  exact Option.some_ne_none _

variable (c rowW) in
/-- **An input's key is the zero or a row bit**: message bit `bit < 8` of byte `byte` of a row port `q`, within the
row's `rowW q` bits. -/
def RowKey (x : ℕ) : Prop :=
  Aliased.srcKey c x = .zero ∨ ∃ q byte bit, Aliased.srcKey c x = .msg q byte bit ∧
    ∃ hq : q < c.ports.size, 8 * byte + bit < rowW ⟨q, hq⟩ ∧ bit < 8

variable (c hU rowW) in
/-- **Every input a computed row reads has a row key** (`RowKey`). A circuit condition: a unit input copied from another
net's output (`Src.wire`) or a fixed column (`Src.lit`) has no registered commit string. An input no row reads, such as
an input padding row (`Src.pad`), is no committed wire and needs none. -/
def KeysRow : Prop :=
  ∀ (i : Fin (Placed.circuitRows c hU).nComp) (x : Fin (Placed.circuitRows c hU).w),
    x ∈ (Placed.circuitRows c hU).a i ++ (Placed.circuitRows c hU).b i → x.val < (Placed.circuitRows c hU).nIn →
      RowKey c rowW x.val

/-- `KeysRow` at a unit of `keyProg`, whose rows are the circuit's. -/
theorem keysRow_keyProg (hkeys : KeysRow c hU rowW) (u : Fin n) (i : Fin ((keyProg c hU n).rows u).nComp)
    (x : Fin ((keyProg c hU n).rows u).w) (hx : x ∈ ((keyProg c hU n).rows u).a i ++ ((keyProg c hU n).rows u).b i)
    (hxn : x.val < ((keyProg c hU n).rows u).nIn) : RowKey c rowW x.val := by
  have e : (keyProg c hU n).rows u = Placed.circuitRows c hU :=
    rows_circuitProg c hU n (keyB c hU) (Aliased.keyIdx c (keyB c hU)) u
  revert i x
  generalize (keyProg c hU n).rows u = R at e ⊢
  subst e
  exact hkeys

/-- The constant's gate of `keyProg`. -/
noncomputable def oneG : Fin (keyN c hU n) := Fin.castLE (Placed.le_gates _ _ n) (Placed.oneGate n (keyB c hU))

/-- The zero's gate of `keyProg`. -/
noncomputable def zeroG : Fin (keyN c hU n) := Fin.castLE (Placed.le_gates _ _ n) (Placed.zeroGate n (keyB c hU))

/-- **The registration's bit at a single gate**: 1 at the constant, 0 at the zero, `outBit R g` elsewhere. -/
noncomputable def keyGate {Reg : Type} (outBit : Reg → Fin (keyN c hU n) → Bool) (R : Reg) (g : Fin (keyN c hU n)) :
    Bool :=
  if g = oneG (c := c) (hU := hU) (n := n) then true
  else if g = zeroG (c := c) (hU := hU) (n := n) then false else outBit R g

/-! ## At the program's places -/

section places

variable {Reg : Type} {mPts : ℕ} {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
  (si : Placed.Sites plan tab)

/-- **Bit `k` of row port `q` at a drawn unit's site**: message bit `k % 8` of byte `k / 8` (`Typed.msgCol`) at the
unit's template instance, in its block. -/
noncomputable def keyRowAddr (S : Finset (Fin n)) (R : Reg) (u : Fin n) (q : Fin c.ports.size) (k : ℕ) :
    Fin (2 ^ (Partition.tabOf plan tab S R u).S.m) :=
  if h : u ∈ S then
    siteAddr (si.acc S R u h) (si.o S R u h)
      (Typed.msgCol (si.acc S R u h).st.c (si.vu S R u h) q.val (k / 8) (k % 8))
  else ⟨0, Nat.two_pow_pos _⟩

/-- **Row port `q`'s `b ‖ c` at a drawn unit's site**: its `hm96` slot's output (`bcAddr`) at the unit's template
instance, in its block. -/
noncomputable def keyBcAddr (S : Finset (Fin n)) (R : Reg) (u : Fin n) (q : Fin c.ports.size) (j : Fin 128)
    (t : Fin 8) : Fin (2 ^ (Partition.tabOf plan tab S R u).S.m) :=
  if h : u ∈ S then
    siteAddr (si.acc S R u h) (si.o S R u h) (bcAddr (si.acc S R u h).st.c (si.vu S R u h) q.val j.val t.val)
  else ⟨0, Nat.two_pow_pos _⟩

/-- **Row port `q`'s salt at a drawn unit's site**: `saltCol` (the `hm96` slot's salt input, [e2e-hm]'s
`Discharge.Hm.saltCol`) at the unit's template instance, in its block. -/
noncomputable def keySaltAddr (saltCol : Flock.Circuit → ℕ → ℕ → ℕ → ℕ → ℕ) (S : Finset (Fin n)) (R : Reg)
    (u : Fin n) (q : (Fin n × Fin c.ports.size) ⊕ Fin (keyN c hU n)) (j : Fin 192) (t : Fin 8) :
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
  (outs : Finset (Fin (keyN c hU n)))

/-- [e2e-placed]'s places of the circuit's program on Δ's source keys. -/
noncomputable abbrev keyPlace := (Placed.Sites.progPlaces_keyProg hU hparse outs).derived.place

variable (rowW) (saltCol : Flock.Circuit → ℕ → ℕ → ℕ → ℕ → ℕ) (popOf : Reg → Public)
  (outBit : Reg → Fin (keyN c hU n) → Bool)

/-- **The readers at `keyProg`'s sites**: rows `(u, q)` of `rowW q` bits at the message columns (`keyRowAddr`), their
`b ‖ c` at the `hm96` outputs (`keyBcAddr`) and salt at `saltCol` (`keySaltAddr`), the registration's `b ‖ c` the
population's public file's (`popOf R`), and single gates with `keyGate outBit`. -/
noncomputable def keyReaders : Readers ((keyProg c hU n).circuit outs) plan tab :=
  Readers.ofRows (keyPlace hparse outs) (keyN_pos c hU n) rowW (W := Finset.univ.sup rowW + 1)
    (fun q => (Finset.le_sup (Finset.mem_univ q)).trans (Nat.le_succ _)) (Nat.le_add_left 1 _) (keyRowOf c hU n rowW)
    (keyRowAddr si) (keySaltAddr si saltCol) (keyBcAddr si)
    (fun R u q => ((popOf R).digest u.val q.val c.ports.size).data.toList) (keyGate outBit)

variable {rowW}

include hparse in
/-- **Δ's copies at `keyProg`'s sites**: each column of a drawn unit on a row bit holds the witness's bit at the
message column of its key (`copy_site`), in the block by the parse (`msgCol_lt_of_parse`). -/
theorem keyProg_copies :
    (keyReaders rowW hparse outs saltCol popOf outBit).Copies (keyPlace hparse outs) := by
  refine ofRows_copies _ fun S R u hu M hM x hx u' q k hr => ?_
  obtain ⟨-, hxn, hkey⟩ := keyRowOf_wire outs x hr
  have hc := (si.acc S R u hu).c_eq (hparse S R u hu)
  have hrw := Placed.Sites.rowsOf_keyProg hU hparse S R u hu
  have hxB : x.val < keyB c hU :=
    lt_of_lt_of_eq hxn (congrArg FlockSoundness.Rows.nIn
      (rows_circuitProg c hU n (keyB c hU) (Aliased.keyIdx c (keyB c hU)) u))
  have hk' : Aliased.srcKey (si.acc S R u hu).st.c x.val = .msg q.val (k.val / 8) (k.val % 8) := by
    rw [hc]; exact hkey
  have hin' : Typed.msgCol (si.acc S R u hu).st.c (si.vu S R u hu) q.val (k.val / 8) (k.val % 8) <
      2 ^ (si.acc S R u hu).st.c.kLog := by
    have hv := si.hvu S R u hu
    rw [hc] at hv ⊢
    exact msgCol_lt_of_parse (hparse S R u hu) hv q.isLt _ _ (Nat.mod_lt _ (by norm_num))
  have := copy_site (si.acc S R u hu) hM (si.o S R u hu) (si.hvu S R u hu)
    (Fin.cast (congrArg FlockSoundness.Rows.w hrw) x) (lt_of_lt_of_eq hxn (congrArg FlockSoundness.Rows.nIn hrw))
    hk' hin'
  simp only [keyRowAddr, hu, ↓reduceDIte]
  exact this

include hparse in
/-- **The rows' `b ‖ c` at `keyProg`'s sites is the registration's**: at each row port a drawn unit reads, the bytes at
its `hm96` output are the population's `b ‖ c` of the unit (`bc_site`), at every site whose rank is the unit
(`BcSite`, at the sites whose table some witness satisfies). -/
theorem keyProg_bcPinned
    (hsite : ∀ S R u (h : u ∈ S) M, Binding.Sat plan tab S R u M →
      BcSite (si.acc S R u h) (popOf R) (si.o S R u h) (si.vu S R u h) u.val) :
    (keyReaders rowW hparse outs saltCol popOf outBit).BcPinned (P := (keyProg c hU n).partition outs) := by
  refine ofRows_bcPinned _ fun S R u hu M hM g hg u' q k hr => ?_
  obtain ⟨x, rfl⟩ := exists_col_of_io ((keyPlace hparse outs) S R u hu).inst hg
  obtain ⟨hu', -, -⟩ := keyRowOf_wire outs x hr
  rw [hu']
  have hc := (si.acc S R u hu).c_eq (hparse S R u hu)
  have hnp : (si.acc S R u hu).st.c.ports.size = c.ports.size := by rw [hc]
  have := bc_site_of (hsite S R u hu M hM) hM (si.hvu S R u hu) (p := q.val) (by rw [hnp]; exact q.isLt)
  rw [hnp] at this
  unfold keyBcAddr
  simp only [hu, ↓reduceDIte]
  exact this

include hparse in
set_option linter.unusedVariables false in
/-- **`BcSite` at a site of `keyProg`'s places** in a circuit whose unit returns the outputs: the template instance is
below `2^⌊log₂ c.g⌋` by the parse (`vu_lt_pow2`), the proved units are in the population by the draw's decoding
(`proved_lt_population`), and the other conditions are `BcSite`'s. -/
theorem bcSite_keyProg (ho : c.outNet = c.unitNet) {S : Finset (Fin n)} {R : Reg} {u : Fin n} (h : u ∈ S)
    {pop : Public} {d : Lean.Json} {D : Draw.UnitDraw} (hd : (si.acc S R u h).draw = some d)
    (hD : Draw.UnitDraw.ofJson d = .ok D)
    (hregs : ∀ r ∈ (si.acc S R u h).st.regions.toList,
      Refine.regionOf (si.acc S R u h).st.c.kLog ((si.acc S R u h).st.m - (si.acc S R u h).st.c.kLog) r ∈
        (si.acc S R u h).regs)
    (hpop : HmRow.loadPublic (si.acc S R u h).tags (si.acc S R u h).st.c (si.acc S R u h).pf = .ok pop)
    (hpt : pop.tables = none) (hu : D.proved[(si.o S R u h).val * c.g + si.vu S R u h]? = some u.val) :
    BcSite (si.acc S R u h) pop (si.o S R u h) (si.vu S R u h) u.val := by
  have hc := (si.acc S R u h).c_eq (hparse S R u h)
  have hv := si.hvu S R u h
  rw [hc] at hv
  refine ⟨d, D, hd, hD, proved_lt_population hD, hregs, hpop, ?_, ?_⟩
  · rw [hc]; exact vu_lt_pow2 (hparse S R u h) ho hv
  · rw [hc]; exact hu

theorem zeroG_ne_oneG : zeroG (c := c) (hU := hU) (n := n) ≠ oneG := fun h =>
  absurd (congrArg Fin.val h) (by simp [zeroG, oneG, Placed.zeroGate, Placed.oneGate])

/-- **A single committed gate of a unit's own instance** is the zero, the constant, or one of the unit's computed gates,
when every input a row reads has a row key (`KeysRow`). -/
theorem single_gate (outs : Finset (Fin (keyN c hU n))) (hkeys : KeysRow c hU rowW) {u : Fin n}
    {g : Fin (keyN c hU n)}
    (hgx : ∃ x : Fin ((keyProg c hU n).rows u).w, (Const.Prog.instC (keyProg c hU n) outs u).wire x = g)
    (hg : g ∈ ((keyProg c hU n).partition outs).io u) (hr : keyRowOf c hU n rowW g = none) :
    g = zeroG ∨ g = oneG ∨ ((keyProg c hU n).unit g = some u ∧ g ≠ oneG ∧ g ≠ zeroG) := by
  obtain ⟨x, rfl⟩ := hgx
  have e := congrArg FlockSoundness.Rows.nIn
    (rows_circuitProg c hU n (keyB c hU) (Aliased.keyIdx c (keyB c hU)) u)
  have hgate_in : ∀ {g : Fin (keyN c hU n)}, g.val < 2 + n * keyB c hU → (keyProg c hU n).unit g = none :=
    fun {g} hg => unit_uniform_input (k := Placed.inGates n (keyB c hU)) (R := Placed.circuitRows c hU)
      (ins := Placed.insOf n (keyB c hU) (Aliased.keyIdx c (keyB c hU))) (one := Placed.oneGate n (keyB c hU))
      (hone := Placed.insOf_ne_one n (keyB c hU) (Aliased.keyIdx c (keyB c hU))) n g hg
  by_cases hx : x.val < ((keyProg c hU n).rows u).nIn
  · have hxB : x.val < keyB c hU := lt_of_lt_of_eq hx e
    have hw : ((Const.Prog.instC (keyProg c hU n) outs u).wire x).val =
        (Placed.srcGate n (keyB c hU) u.isLt (Aliased.keyIdx c (keyB c hU) ⟨x.val, hxB⟩)).val := by
      have hv : (Const.Prog.insAt (keyProg c hU n) u x.val).val =
          (Placed.insOf n (keyB c hU) (Aliased.keyIdx c (keyB c hU)) u.val ⟨x.val, hxB⟩).val :=
        Placed.insAt_uniform n u x.val hxB
      rw [Const.Prog.instC_wire_input _ _ _ x hx, hv,
        Placed.insOf_of_lt (nU := n) (B := keyB c hU) (src := Aliased.keyIdx c (keyB c hU)) u.isLt]
    rcases hb : Aliased.keyIdx c (keyB c hU) ⟨x.val, hxB⟩ with _ | b
    · rw [hb] at hw
      exact .inl (Fin.ext hw)
    · rw [hb] at hw
      simp only [Placed.srcGate] at hw
      have hnz : Aliased.srcKey c x.val ≠ .zero := fun h => by
        rw [(Aliased.keyIdx_eq_none (c := c) (⟨x.val, hxB⟩ : Fin (keyB c hU))).2 h] at hb
        cases hb
      have hB : 0 < keyB c hU := lt_of_le_of_lt (Nat.zero_le _) hxB
      have hm : (2 + u.val * keyB c hU + b.val - 2) % keyB c hU = b.val := by
        rw [show 2 + u.val * keyB c hU + b.val - 2 = b.val + u.val * keyB c hU by omega,
          Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt b.isLt]
      have hnone : (keyProg c hU n).unit ((Const.Prog.instC (keyProg c hU n) outs u).wire x) = none :=
        hgate_in (by rw [hw]; exact Placed.source_lt n _ u.isLt b)
      obtain ⟨i, c', hc', hwc⟩ := exists_read_of_io (Const.Prog.instC (keyProg c hU n) outs u) hg
        (fun h => by
          have h' : (keyProg c hU n).unit ((Const.Prog.instC (keyProg c hU n) outs u).wire x) = some u := h
          rw [hnone] at h'
          cases h')
      have hc'n : c'.val < ((keyProg c hU n).rows u).nIn := by
        rcases (Const.Prog.instC (keyProg c hU n) outs u).shared c' x hwc with rfl | ⟨h, -⟩
        · exact hx
        · exact h
      have hkey' := Placed.keyProg_wire_key hU outs u x c' hx hc'n hwc.symm
      rcases keysRow_keyProg hkeys u i c' hc' hc'n with h0 | ⟨q, byte, bt, hk, hq, hlw, hlb⟩
      · exact absurd (hkey'.trans h0) hnz
      · refine absurd hr (keyRowOf_ne_none (by rw [hw]; omega) (by rw [hw]; exact Placed.source_lt n _ u.isLt b)
          (q := q) (byte := byte) (bit := bt) ?_ hq hlw hlb)
        rw [hw, hm, keyIdx_some hb, hkey']
        exact hk
  · by_cases hone : x = ((keyProg c hU n).rows u).one
    · refine .inr (.inl ?_)
      rw [hone, Const.Prog.instC_wire_one]
      exact Fin.ext (oneAt_uniform_val (k := Placed.inGates n (keyB c hU)) (R := Placed.circuitRows c hU)
        (ins := Placed.insOf n (keyB c hU) (Aliased.keyIdx c (keyB c hU))) (one := Placed.oneGate n (keyB c hU))
        (hone := Placed.insOf_ne_one n (keyB c hU) (Aliased.keyIdx c (keyB c hU))) n u)
    · have hne : x.val ≠ ((keyProg c hU n).rows u).nIn + ((keyProg c hU n).rows u).nComp := fun he =>
        hone (Fin.ext he)
      have hxw := x.isLt
      have hi : x.val - ((keyProg c hU n).rows u).nIn < ((keyProg c hU n).rows u).nComp := by
        change x.val < ((keyProg c hU n).rows u).nIn + ((keyProg c hU n).rows u).nComp + 1 at hxw
        omega
      have hx' : x = ((keyProg c hU n).rows u).comp ⟨_, hi⟩ :=
        Fin.ext (by simp only [FlockSoundness.Rows.comp]; omega)
      have hm : (keyProg c hU n).unit ((Const.Prog.instC (keyProg c hU n) outs u).wire x) = some u := by
        have := (Const.Prog.instC (keyProg c hU n) outs u).mem ⟨_, hi⟩
        rw [← hx'] at this
        exact this
      refine .inr (.inr ⟨hm, fun h => ?_, fun h => ?_⟩)
      · rw [h, hgate_in (by show 0 < 2 + n * keyB c hU; omega)] at hm
        cases hm
      · rw [h, hgate_in (by show 1 < 2 + n * keyB c hU; omega)] at hm
        cases hm

include hparse in
/-- **The single gates hold the registration's bits** at `keyProg`'s sites: 1 at the constant (`hc1`, [e2e-const]'s
`hConst`), 0 at the zero (`hz1`, [e2e-zero]'s `hZero`), and at a unit's committed gate the registration's bit
(`hout`), when every input a row reads has the zero or a row bit as its key (`KeysRow`), so that no other input is a
committed single gate. -/
theorem keyProg_valPinned (hkeys : KeysRow c hU rowW)
    (hc1 : ((keyProg c hU n).partition outs).ConstCols plan tab (keyPlace hparse outs)
      {oneG (c := c) (hU := hU) (n := n)})
    (hz1 : ((keyProg c hU n).partition outs).ZeroCols plan tab (keyPlace hparse outs)
      {zeroG (c := c) (hU := hU) (n := n)})
    (hout : ∀ S R u, u ∈ S → ∀ M, Binding.Sat plan tab S R u M → ∀ g ∈ ((keyProg c hU n).partition outs).io u,
      (keyProg c hU n).unit g = some u →
        bit (Binding.wit plan tab S R u M (gateAt (keyPlace hparse outs) S R u g)) = outBit R g) :
    (keyReaders rowW hparse outs saltCol popOf outBit).ValPinned (P := (keyProg c hU n).partition outs) := by
  refine ofRows_valPinned _ fun S R u hu M hM g hg hr => ?_
  have hpub := gate_pub (keyPlace hparse outs) hc1 hz1 hu hM hg
  obtain ⟨x, hx⟩ := exists_col_of_io ((keyPlace hparse outs) S R u hu).inst hg
  change bit _ = keyGate outBit R g
  rcases single_gate outs hkeys ⟨x, hx⟩ hg hr with rfl | rfl | ⟨hm, h1, h2⟩
  · rw [hpub.2 (Finset.mem_singleton_self _)]
    simp only [keyGate, zeroG_ne_oneG, ↓reduceIte]
  · rw [hpub.1 (Finset.mem_singleton_self _)]
    simp only [keyGate, ↓reduceIte]
  · rw [hout S R u hu M hM g hg hm]
    simp only [keyGate, h1, h2, ↓reduceIte]

variable (rowW) in
/-- **`lay` at `keyProg`'s places**: the readers' layout (`Readers.layout`), with Δ's copies from `keyProg_copies`. -/
noncomputable def keyLayout (encO : BitVal (Finset.univ.sup rowW + 1) → List UInt8) :
    Binding.Layout ((keyProg c hU n).partition outs) plan tab
      (((keyProg c hU n).partition outs).placeDecoder plan tab (keyPlace hparse outs)) :=
  (keyReaders rowW hparse outs saltCol popOf outBit).layout (keyPlace hparse outs) encO
    (keyProg_copies hparse outs saltCol popOf outBit)

variable (rowW) in
/-- **`rs` at `keyProg`'s places**: the readers' rows (`Readers.rows`), from `keyProg_copies`, `keyProg_bcPinned` and
`keyProg_valPinned`. -/
noncomputable def keyRows (enc : BitVal (Finset.univ.sup rowW + 1) → List UInt8)
    (enc_inj : Function.Injective enc) (encO : BitVal (Finset.univ.sup rowW + 1) → List UInt8)
    (encO_inj : Function.Injective encO)
    (hsite : ∀ S R u (h : u ∈ S) M, Binding.Sat plan tab S R u M →
      BcSite (si.acc S R u h) (popOf R) (si.o S R u h) (si.vu S R u h) u.val)
    (hkeys : KeysRow c hU rowW)
    (hc1 : ((keyProg c hU n).partition outs).ConstCols plan tab (keyPlace hparse outs)
      {oneG (c := c) (hU := hU) (n := n)})
    (hz1 : ((keyProg c hU n).partition outs).ZeroCols plan tab (keyPlace hparse outs)
      {zeroG (c := c) (hU := hU) (n := n)})
    (hout : ∀ S R u, u ∈ S → ∀ M, Binding.Sat plan tab S R u M → ∀ g ∈ ((keyProg c hU n).partition outs).io u,
      (keyProg c hU n).unit g = some u →
        bit (Binding.wit plan tab S R u M (gateAt (keyPlace hparse outs) S R u g)) = outBit R g) :
    Binding.HmRows (keyLayout rowW hparse outs saltCol popOf outBit encO) :=
  (keyReaders rowW hparse outs saltCol popOf outBit).rows (keyPlace hparse outs) enc enc_inj encO encO_inj
    (keyProg_copies hparse outs saltCol popOf outBit)
    (keyProg_bcPinned hparse outs saltCol popOf outBit hsite)
    (keyProg_valPinned hparse outs saltCol popOf outBit hkeys hc1 hz1 hout)

end places

end FlockSoundness.Discharge.Layout
