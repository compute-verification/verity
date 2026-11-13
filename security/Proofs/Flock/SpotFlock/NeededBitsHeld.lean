module

public import Proofs.Flock.SpotFlock.NeededBitsOf
public import Proofs.Flock.SpotFlock.Facts
public import Proofs.Flock.SpotFlock.Placement
public import Proofs.Flock.SpotFlock.WiringOk
public import Proofs.Flock.Bridge.BitsRows
public import Proofs.Flock.Bridge.BitsKeys
public import Proofs.Flock.Soundness.Discharge.Hidden.Unrolled
public import Proofs.Flock.Soundness.Discharge.Hidden.Commit
public import Proofs.Flock.Soundness.Discharge.Hidden.ParseCommit
public import Proofs.Flock.Soundness.Discharge.HiddenExec.Flat
public import Proofs.Flock.Soundness.Discharge.FailClosed.Scope
public import Proofs.Flock.Soundness.Discharge.Layout.OutShape
public import Flock.Desc

/-! # Every member of a held code is rowed, at an accepted run

The checks an accepted run passed make every member's bits rowed (`MemberRowed`), so `NeededBits h` holds with no
premise (`SpotFlock.neededBits_held`). A member's statement parses (`templateOk`) and is in the hidden-output scope
(`membersScoped`): its unit reads one input group, and every input a row reads is a port bit (`Layout.KeysRow`), so each
source gate a unit owns is a port's message bit and has a commitment position (`Bits.rowBit`); padding, a wire and a
literal never reach one. Every returned output word is an output word of an output row (`RowsFitS`, `Layout.check_out`),
so each returned output bit has one too (`Bits.outRowBit`).

The same facts give `RowsArePorts h` (each source and written bit's row is one of its member's ports), and `WrittenOnce h`
holds of every held code: a written gate fixes its member, unit, instance, leaf and bit. -/

@[expose] public section

namespace Flock.Bridge.NeededBitsOf

open FlockSoundness

theorem mod_units {N u i : ℕ} (hi : i < N) : (u * N + i) % N = i := by
  have h := Nat.div_add_mod (u * N + i) N
  rw [div_units hi, Nat.mul_comm N u] at h
  omega

theorem inst_lt {N0 upv k d : ℕ} (hk : k < upv) (hd : d < N0) : k * (N0 + 1) + (1 + d) < upv * (N0 + 1) := by
  have hle := Nat.mul_le_mul_right (N0 + 1) (show k + 1 ≤ upv by omega)
  rw [Nat.add_mul, Nat.one_mul] at hle
  omega

theorem srcOf_zero {s : Spot.Bits.Src} (h : srcOf s = .zero) : s = .zero := by
  cases s with
  | zero => rfl
  | _ => simp [srcOf] at h

theorem srcOf_msg {s : Spot.Bits.Src} {q byte bit : ℕ} (h : srcOf s = .msg q byte bit) : s = .msg q byte bit := by
  cases s with
  | msg p byte' bit' =>
    simp only [srcOf] at h
    cases h
    rfl
  | _ => simp [srcOf] at h

theorem name_getD {c : Flock.Circuit} {q : ℕ} (hq : q < c.ports.size) :
    c.ports[q]!.name = (c.ports.getD q default).name := by
  rw [getElem!_pos c.ports q hq]
  simp [Array.getD_eq_getD_getElem?, hq]

section rowed

variable (c : Flock.Circuit) (n : ℕ)

/-- **A column's gate below the rows is a key's source gate**: a column of unit `u` whose gate is a source gate is an
input column with a key. -/
theorem keyWire_src {u col : ℕ} (h2 : 2 ≤ keyWire c n u col) (hlt : keyWire c n u col < 2 + n * (bitsRows c).nIn) :
    col < (bitsRows c).nIn ∧ ∃ b, Spot.Bits.keyIdx c (bitsRows c).nIn col = some b ∧
      keyWire c n u col = 2 + u * (bitsRows c).nIn + b := by
  unfold keyWire at h2 hlt ⊢
  by_cases h1 : col < (bitsRows c).nIn
  · rw [if_pos h1] at h2 hlt ⊢
    have hk : ((Array.range (bitsRows c).nIn).map (Spot.Bits.keyIdx c (bitsRows c).nIn)).getD col none =
        Spot.Bits.keyIdx c (bitsRows c).nIn col := by
      simp [Array.getD_eq_getD_getElem?, h1]
    rw [hk] at h2 hlt ⊢
    cases hb : Spot.Bits.keyIdx c (bitsRows c).nIn col with
    | none =>
      rw [hb] at h2
      simp at h2
    | some b => exact ⟨h1, b, rfl, rfl⟩
  · rw [if_neg h1] at h2 hlt
    by_cases h3 : col < (bitsRows c).nIn + (bitsRows c).nComp
    · rw [if_pos h3] at hlt
      omega
    · rw [if_neg h3] at h2
      omega

/-- **A source gate a unit's row reads** is one of the unit's keyed inputs, at an input column the row reads. -/
theorem src_read {g u w : ℕ} (hg : Spot.Bits.unitOf (Spot.Bits.keyProg c n) g = some u)
    (hw : w ∈ Spot.Bits.args (Spot.Bits.keyProg c n) g) (h2 : 2 ≤ w) (hlt : w < 2 + n * (bitsRows c).nIn) :
    ∃ i < (bitsRows c).nComp, ∃ col ∈ (bitsRows c).a.getD i [] ++ (bitsRows c).b.getD i [],
      col < (bitsRows c).nIn ∧ ∃ b, Spot.Bits.keyIdx c (bitsRows c).nIn col = some b ∧
        w = 2 + u * (bitsRows c).nIn + b := by
  obtain ⟨hu, i, hi, rfl⟩ := unitOf_some c n hg
  simp only [Spot.Bits.args, keyProg_row?_row c n hu hi, List.mem_append, List.mem_map] at hw
  rcases hw with ⟨col, hc, rfl⟩ | ⟨col, hc, rfl⟩
  · obtain ⟨hcol, b, hb, he⟩ := keyWire_src c n h2 hlt
    exact ⟨i, hi, col, List.mem_append.2 (Or.inl hc), hcol, b, hb, he⟩
  · obtain ⟨hcol, b, hb, he⟩ := keyWire_src c n h2 hlt
    exact ⟨i, hi, col, List.mem_append.2 (Or.inr hc), hcol, b, hb, he⟩

/-- **Unit `u`'s source gate at its own key `b`** is unit `u`'s read of `b`'s source. -/
theorem rowBit_key {u b : ℕ} (hu : u < n) (hbB : b < (bitsRows c).nIn)
    (hb : Spot.Bits.keyIdx c (bitsRows c).nIn b = some b) :
    Spot.Bits.rowBit c n (2 + u * (bitsRows c).nIn + b) =
      Spot.Bits.srcPos c u (Spot.Bits.srcKey c (Spot.Bits.Index.of c) b) := by
  have he : ((Spot.Bits.Rows.ofNet c.unit).stack c.upv).nIn = (bitsRows c).nIn := rfl
  have hlt := src_lt hu hbB
  unfold Spot.Bits.rowBit
  dsimp only
  rw [he, show 2 + u * (bitsRows c).nIn + b - 2 = u * (bitsRows c).nIn + b by omega, mod_units hbB,
    div_units hbB, hb]
  split
  · rename_i hc
    simp only [Bool.or_eq_true, decide_eq_true_eq, beq_iff_eq, bne_self_eq_false, Bool.false_eq_true,
      or_false] at hc
    omega
  · rfl

/-- **A keyed input a row reads is a port's message bit** (`KeysRow`): its source gate's commitment position is some,
on a port's row. -/
theorem rowBit_of_read {hU : Built c.unit} (hu32 : c.unit.unitLog ≤ 32)
    (hK : FlockSoundness.Discharge.Layout.KeysRow c hU (FlockSoundness.Discharge.Layout.portRowW c))
    {u i col b : ℕ} (hu : u < n) (hi : i < (bitsRows c).nComp)
    (hc : col ∈ (bitsRows c).a.getD i [] ++ (bitsRows c).b.getD i []) (hcol : col < (bitsRows c).nIn)
    (hb : Spot.Bits.keyIdx c (bitsRows c).nIn col = some b) :
    ∃ q < c.ports.size, ∃ rp, Spot.Bits.rowBit c n (2 + u * (bitsRows c).nIn + b) = some rp ∧
      rp.row = (c.ports.getD q default).name := by
  obtain ⟨hIn, hComp, hab⟩ := rowsAgree_circuitRows c hU hu32
  have hi' : i < (FlockSoundness.Discharge.Placed.circuitRows c hU).nComp := by rw [← hComp]; exact hi
  obtain ⟨ha, hb'⟩ := hab ⟨i, hi'⟩
  have hc' : col ∈ ((FlockSoundness.Discharge.Placed.circuitRows c hU).a ⟨i, hi'⟩ ++
      (FlockSoundness.Discharge.Placed.circuitRows c hU).b ⟨i, hi'⟩).map Fin.val := by
    rw [List.map_append, ← ha, ← hb']
    exact hc
  have hb0 := hb
  unfold Spot.Bits.keyIdx at hb0
  dsimp only at hb0
  split_ifs at hb0 with h0
  have hbe := List.find?_some hb0
  simp only [decide_eq_true_eq] at hbe
  obtain ⟨x, hx, rfl⟩ := List.mem_map.1 hc'
  have hkey := hK ⟨i, hi'⟩ x hx (by rw [← hIn]; exact hcol)
  unfold FlockSoundness.Discharge.Layout.RowKey at hkey
  rw [srcKey_eq] at hkey
  rcases hkey with hz | ⟨q, byte, bit, hm, hq, -, -⟩
  · exact absurd (srcOf_zero hz) h0
  · have hsb : Spot.Bits.srcKey c (Spot.Bits.Index.of c) b = .msg q byte bit := by
      rw [hbe]
      exact srcOf_msg hm
    rw [rowBit_key c n hu (keyIdx_lt hb) (keyIdx_self hb), hsb]
    exact ⟨q, hq, _, rfl, rfl⟩

/-- **Every source gate a member's constants leave free has a commitment position on a port's row**, at `KeysRow`. -/
theorem rowBit_of_free {hU : Built c.unit} (hu32 : c.unit.unitLog ≤ 32)
    (hK : FlockSoundness.Discharge.Layout.KeysRow c hU (FlockSoundness.Discharge.Layout.portRowW c))
    {g : ℕ} (hg : g < (Spot.Bits.keyProg c n).inputs) (hκ : Spot.Bits.consts c n g = none) :
    ∃ q < c.ports.size, ∃ rp, Spot.Bits.rowBit c n g = some rp ∧ rp.row = (c.ports.getD q default).name := by
  have h0 : g ≠ 0 := fun h => by rw [h, consts_zero] at hκ; cases hκ
  have h1 : g ≠ 1 := fun h => by rw [h, consts_one] at hκ; cases hκ
  cases hk : Spot.Bits.owner (Spot.Bits.keyProg c n) g with
  | none => simp [Spot.Bits.consts, hg, hk, h0, h1] at hκ
  | some k =>
    obtain ⟨g', hg', hw⟩ := exists_of_owner hk
    rw [keyProg_inputs] at hg
    obtain ⟨i, hi, col, hc, hcol, b, hb, rfl⟩ := src_read c n hg' hw (by omega) hg
    exact rowBit_of_read c n hu32 hK (unitOf_some c n hg').1 hi hc hcol hb

/-- **A returned output bit's commitment position is an output word's**, when every returned output is an output word. -/
theorem outRowBit_of_outGate (hsz : c.leavesOut.size = c.upv)
    (hl : ∀ u < c.upv, ∀ o ∈ (c.leavesOut[u]!).toList, o < c.outWords)
    {u k j t x : ℕ} (hx : Spot.Bits.outGate c n u k j t = some x) :
    ∃ w < c.outWords, Spot.Bits.outRowBit c u k j t = Spot.Bits.outPos c u w t := by
  obtain ⟨-, hk, hj, -, -⟩ := outGate_some c n hx
  have hkl : k < c.leavesOut.size := by rw [hsz]; exact hk
  have hg : c.leavesOut.getD k #[] = c.leavesOut[k]! := by
    rw [getElem!_pos c.leavesOut k hkl]
    simp [Array.getD_eq_getD_getElem?, hkl]
  rw [hg] at hj
  refine ⟨(c.leavesOut[k]!)[j], hl k hk _ (Array.getElem_mem_toList hj), ?_⟩
  unfold Spot.Bits.outRowBit
  rw [hg, getElem?_pos (c.leavesOut[k]!) j hj]
  rfl

/-- **Every returned output bit has a commitment position on a port's row**, at `RowsFitS` with every returned output
an output word. -/
theorem outRowBit_some (hR : FlockSoundness.Discharge.Hidden.Any.RowsFitS c) (hsz : c.leavesOut.size = c.upv)
    (hl : ∀ u < c.upv, ∀ o ∈ (c.leavesOut[u]!).toList, o < c.outWords)
    {u k j t x : ℕ} (hx : Spot.Bits.outGate c n u k j t = some x) :
    ∃ q < c.ports.size, ∃ rp, Spot.Bits.outRowBit c u k j t = some rp ∧ rp.row = (c.ports.getD q default).name := by
  obtain ⟨w, hw, he⟩ := outRowBit_of_outGate c n hsz hl hx
  rw [he, outPos_eq hR.1.1 hw]
  exact ⟨_, (hR.1.2 _ (FlockSoundness.Discharge.HiddenExec.outPortOf_spec c hw).1).1, _, rfl, rfl⟩

/-- **What the hidden-output scope gives a parsed statement**: every input a row reads is a port bit (`KeysRow`), its
rows fit (`RowsFitS`), and every returned output is an output word. -/
theorem scoped_facts {tags : Flock.Tags} {cf : ByteArray} {tables : List (String × ByteArray)}
    (hty : tags.typed = false) (hcS : HmOut.parseStmt tags cf tables = .ok c) (hU : Built c.unit)
    (hs : ProvedScope.scopeOkH c = true) :
    FlockSoundness.Discharge.Layout.KeysRow c hU (FlockSoundness.Discharge.Layout.portRowW c) ∧
      FlockSoundness.Discharge.Hidden.Any.RowsFitS c ∧ c.leavesOut.size = c.upv ∧
        ∀ u < c.upv, ∀ o ∈ (c.leavesOut[u]!).toList, o < c.outWords := by
  rw [FlockSoundness.Discharge.FailClosed.scopeOkH_eq] at hs
  obtain ⟨h1, ho, hpr⟩ := FlockSoundness.Discharge.Layout.scopeOkH_spec hU hs
  refine ⟨FlockSoundness.Discharge.Hidden.keysRow_stmt hty hcS h1 hpr,
    FlockSoundness.Discharge.HiddenExec.rowsFit_of_stmt hty hcS hs, ?_⟩
  obtain ⟨cs, hcs, hce⟩ := FlockSoundness.commit_of_parseStmt hty hcS
  have hC := FlockSoundness.Discharge.Hidden.committed_of_commit hcs
  obtain ⟨nets, he⟩ := FlockSoundness.Discharge.Hidden.Committed.unrolled_eq hty hC
  obtain ⟨hty', ⟨ls, hls⟩, -⟩ := FlockSoundness.Discharge.Hidden.Committed.check hty hC
  subst hce
  rw [he] at ho ⊢
  have hout := FlockSoundness.Discharge.Layout.check_out hty' ho hls
  exact ⟨hout.2.1, fun u hu => (hout.2.2 u hu).2⟩

/-- **A member in the hidden-output scope is rowed** (`MemberRowed`). -/
theorem memberRowed_of {tags : Flock.Tags} {cf : ByteArray} {tables : List (String × ByteArray)}
    (hty : tags.typed = false) (hcS : HmOut.parseStmt tags cf tables = .ok c) (hU : Built c.unit)
    (hu32 : c.unit.unitLog ≤ 32) (hs : ProvedScope.scopeOkH c = true) : _root_.SpotFlock.MemberRowed c n := by
  obtain ⟨hK, hR, hsz, hl⟩ := scoped_facts c hty hcS hU hs
  refine ⟨fun g hg hκ => ?_, fun u k j t x hx => ?_⟩
  · obtain ⟨-, -, rp, he, -⟩ := rowBit_of_free c n hu32 hK hg hκ
    rw [he]
    rfl
  · obtain ⟨-, -, rp, he, -⟩ := outRowBit_some c n hR hsz hl hx
    rw [he]
    rfl

/-- **One gate is written once**: two returned output bits on one gate are the same unit's same instance's same leaf
bit. -/
theorem outGate_inj {u k j t u' k' j' t' x : ℕ} (h : Spot.Bits.outGate c n u k j t = some x)
    (h' : Spot.Bits.outGate c n u' k' j' t' = some x) : u = u' ∧ k = k' ∧ j = j' ∧ t = t' := by
  have hP : ((Spot.Bits.Rows.ofNet c.unit).stack c.upv).nComp = c.upv * ((Spot.Bits.Rows.ofNet c.unit).nComp + 1) :=
    rfl
  unfold Spot.Bits.outGate at h h'
  dsimp only at h h'
  split at h
  · rename_i hc
    split at h'
    · rename_i hc'
      simp only [Bool.and_eq_true, decide_eq_true_eq] at hc hc'
      obtain ⟨⟨⟨⟨⟨-, hk⟩, -⟩, ht⟩, h1⟩, h2⟩ := hc
      obtain ⟨⟨⟨⟨⟨-, hk'⟩, -⟩, ht'⟩, h1'⟩, h2'⟩ := hc'
      have he := (Option.some.inj h).trans (Option.some.inj h').symm
      unfold Spot.Bits.rowGate at he
      have hA := inst_lt (N0 := (Spot.Bits.Rows.ofNet c.unit).nComp) hk
        (show Spot.Bits.outCol c j t - (Spot.Bits.Rows.ofNet c.unit).nIn < (Spot.Bits.Rows.ofNet c.unit).nComp by
          omega)
      have hA' := inst_lt (N0 := (Spot.Bits.Rows.ofNet c.unit).nComp) hk'
        (show Spot.Bits.outCol c j' t' - (Spot.Bits.Rows.ofNet c.unit).nIn < (Spot.Bits.Rows.ofNet c.unit).nComp by
          omega)
      rw [← hP] at hA hA'
      have hu := eq_units (u := u) (u' := u') hA hA' (by omega)
      subst hu
      have hd1 : 1 + (Spot.Bits.outCol c j t - (Spot.Bits.Rows.ofNet c.unit).nIn) <
          (Spot.Bits.Rows.ofNet c.unit).nComp + 1 := by omega
      have hd1' : 1 + (Spot.Bits.outCol c j' t' - (Spot.Bits.Rows.ofNet c.unit).nIn) <
          (Spot.Bits.Rows.ofNet c.unit).nComp + 1 := by omega
      have hk2 := eq_units (u := k) (u' := k') hd1 hd1' (by omega)
      subst hk2
      have hoc : Spot.Bits.outCol c j t = Spot.Bits.outCol c j' t' := by omega
      unfold Spot.Bits.outCol at hoc
      exact ⟨rfl, rfl, by omega, by omega⟩
    · cases h'
  · cases h

end rowed

end Flock.Bridge.NeededBitsOf

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy FlockSoundness FlockSoundness.Discharge Exec Flock Flock.Readable

/-- **Each member's statement parses** as an untyped hidden-output statement, at an accepted design (`templateOk`). -/
theorem member_parsed (h : Held) {s : AuditStatement} {d : Design.Two} (hd : two h s = some d) {t : ℕ}
    (ht : t < h.ms.length) :
    ∃ (tags : _root_.Flock.Tags) (cf : ByteArray) (tables : List (String × ByteArray)), tags.typed = false ∧
      _root_.Flock.HmOut.parseStmt tags cf tables = .ok h.ms[t].1 := by
  have hok := templateOk_of_two h hd ht
  have hms : h.ms[t].1 = stmtOf (h.xS₀ t) := by simp [Held.ms]
  unfold templateOk at hok
  cases hr : FlockDeploy.readOf true (h.xS₀ t) with
  | none => rw [hr] at hok; cases hok
  | some r =>
    rw [hr] at hok
    have hst : h.ms[t].1 = r.stmt := by
      rw [hms]
      unfold stmtOf
      rw [hr]
      rfl
    simp only [Bool.and_eq_true, Bool.not_eq_true', decide_eq_true_eq] at hok
    obtain ⟨⟨⟨⟨⟨hh, hty⟩, hD⟩, -⟩, -⟩, -⟩ := hok
    split at hD
    · next D hD' =>
      rw [hst, stmt_of_parseStmt hh hty hD']
      exact ⟨_, _, _, hty, _root_.Flock.Desc.parseStmt_stmt hD'⟩
    · cases hD

/-- **What an accepted design gives member `t`**: its unit is built, of at most 2³² columns, every input a row reads is
a port bit, its rows fit, and every returned output is an output word. -/
theorem member_facts (h : Held) {s : AuditStatement} {d : Design.Two} (hd : two h s = some d) {t : ℕ}
    (ht : t < h.ms.length) :
    ∃ hU : _root_.FlockSoundness.Built h.ms[t].1.unit, h.ms[t].1.unit.unitLog ≤ 32 ∧
      FlockSoundness.Discharge.Layout.KeysRow h.ms[t].1 hU (FlockSoundness.Discharge.Layout.portRowW h.ms[t].1) ∧
        FlockSoundness.Discharge.Hidden.Any.RowsFitS h.ms[t].1 ∧
        h.ms[t].1.leavesOut.size = h.ms[t].1.upv ∧
          ∀ u < h.ms[t].1.upv, ∀ o ∈ (h.ms[t].1.leavesOut[u]!).toList, o < h.ms[t].1.outWords := by
  obtain ⟨tags, cf, tables, hty, hcS⟩ := member_parsed h hd ht
  obtain ⟨-, hU, hu32⟩ := templateOk_facts h (templateOk_of_two h hd ht) ht
  obtain ⟨hK, hR, hsz, hl⟩ :=
    _root_.Flock.Bridge.NeededBitsOf.scoped_facts _ hty hcS hU (scoped_of_two h hd _ (List.getElem_mem ht))
  exact ⟨hU, hu32, hK, hR, hsz, hl⟩

/-- **Every member is rowed at an accepted design** (`MembersRowed`). -/
theorem membersRowed_of_two (h : Held) {s : AuditStatement} {d : Design.Two} (hd : two h s = some d) :
    MembersRowed h := by
  intro ci hci
  obtain ⟨t, ht, rfl⟩ := List.getElem_of_mem hci
  obtain ⟨tags, cf, tables, hty, hcS⟩ := member_parsed h hd ht
  obtain ⟨-, hU, hu32⟩ := templateOk_facts h (templateOk_of_two h hd ht) ht
  exact _root_.Flock.Bridge.NeededBitsOf.memberRowed_of _ _ hty hcS hU hu32
    (scoped_of_two h hd _ (List.getElem_mem ht))

/-- **Every gate a checked unit needs is one of its own unit's bits** (`NeededBits`), of every held code: an accepted
run's design is accepted, so every member is rowed. -/
theorem neededBits_held (h : Held) : NeededBits h := by
  intro r hr
  obtain ⟨_, _, _, _, _, h2, _⟩ := id hr
  exact neededBits_of_rowed (membersRowed_of_two h h2) r hr

theorem memberCircuit_eq (h : Held) {t : ℕ} (ht : t < h.ms.length) : memberCircuit h t = h.ms[t].1 := by
  simp [memberCircuit, Held.ms]

/-- **Each source and written bit's row is one of its member's ports** (`RowsArePorts`), at an accepted design. -/
theorem rowsArePorts_of_two (h : Held) {s : AuditStatement} {d : Design.Two} (hd : two h s = some d) :
    RowsArePorts h := by
  refine ⟨fun e he => ?_, fun e he => ?_⟩
  · obtain ⟨m, hm, g, hg, hκ, r, hr, rfl⟩ := _root_.Flock.Bridge.NeededBitsOf.of_mem_sourceBits he
    obtain ⟨hU, hu32, hK, -⟩ := member_facts h hd hm
    obtain ⟨q, hq, rp, he', hrow⟩ := _root_.Flock.Bridge.NeededBitsOf.rowBit_of_free _ _ hu32 hK hg hκ
    rw [hr] at he'
    cases he'
    dsimp only
    rw [memberCircuit_eq h hm]
    exact ⟨q, hq, by rw [hrow]; exact _root_.Flock.Bridge.NeededBitsOf.name_getD hq⟩
  · obtain ⟨m, hm, u, k, j, t, x, r, hx, hr, rfl⟩ := _root_.Flock.Bridge.NeededBitsOf.of_mem_writtenBits he
    obtain ⟨hU, hu32, -, hR, hsz, hl⟩ := member_facts h hd hm
    obtain ⟨q, hq, rp, he', hrow⟩ := _root_.Flock.Bridge.NeededBitsOf.outRowBit_some _ _ hR hsz hl hx
    rw [hr] at he'
    cases he'
    dsimp only
    rw [memberCircuit_eq h hm]
    exact ⟨q, hq, by rw [hrow]; exact _root_.Flock.Bridge.NeededBitsOf.name_getD hq⟩

/-- **Each written gate is written once** (`WrittenOnce`), of every held code. -/
theorem writtenOnce_held (h : Held) : WrittenOnce h := by
  intro a ha b hb hab
  obtain ⟨m, hm, u, k, j, t, x, r, hx, hr, rfl⟩ := _root_.Flock.Bridge.NeededBitsOf.of_mem_writtenBits ha
  obtain ⟨m', hm', u', k', j', t', x', r', hx', hr', rfl⟩ := _root_.Flock.Bridge.NeededBitsOf.of_mem_writtenBits hb
  obtain ⟨-, -, -, -, i, hi, hxe⟩ := _root_.Flock.Bridge.NeededBitsOf.outGate_some _ _ hx
  obtain ⟨-, -, -, -, i', hi', hxe'⟩ := _root_.Flock.Bridge.NeededBitsOf.outGate_some _ _ hx'
  have hxu := _root_.Flock.Bridge.NeededBitsOf.unitOf_row _ _ (_root_.Flock.Bridge.NeededBitsOf.outGate_some _ _ hx).1 hi
  have hxu' :=
    _root_.Flock.Bridge.NeededBitsOf.unitOf_row _ _ (_root_.Flock.Bridge.NeededBitsOf.outGate_some _ _ hx').1 hi'
  rw [← hxe] at hxu
  rw [← hxe'] at hxu'
  obtain ⟨a₁, b₁, hab₁⟩ := _root_.Flock.Bridge.NeededBitsOf.row?_of_unitOf hxu
  obtain ⟨a₂, b₂, hab₂⟩ := _root_.Flock.Bridge.NeededBitsOf.row?_of_unitOf hxu'
  have hxs := (_root_.Flock.Bridge.NeededBitsOf.unitOf_eq_some hxu).2.1
  have hxs' := (_root_.Flock.Bridge.NeededBitsOf.unitOf_eq_some hxu').2.1
  have hp : Spot.Bits.posOf (Spot.Bits.keyProg h.ms[m].1 h.ms[m].2.size) x <
      (Properties.Spot.memberProg h.ms[m].1 h.ms[m].2.size).gates.length := by
    rw [_root_.Flock.Bridge.NeededBitsOf.memberProg_eq, Properties.Spot.SecurityProofs.bitsProg_length]
    exact _root_.Flock.Bridge.NeededBitsOf.posOf_lt_off hxs
  have hp' : Spot.Bits.posOf (Spot.Bits.keyProg h.ms[m'].1 h.ms[m'].2.size) x' <
      (Properties.Spot.memberProg h.ms[m'].1 h.ms[m'].2.size).gates.length := by
    rw [_root_.Flock.Bridge.NeededBitsOf.memberProg_eq, Properties.Spot.SecurityProofs.bitsProg_length]
    exact _root_.Flock.Bridge.NeededBitsOf.posOf_lt_off hxs'
  obtain ⟨rfl, hpq⟩ := _root_.Flock.Bridge.NeededBitsOf.pos_eq h.ms hm hm' hp hp' hab
  rw [_root_.Flock.Bridge.NeededBitsOf.posOf_of_some hab₁, _root_.Flock.Bridge.NeededBitsOf.posOf_of_some hab₂] at hpq
  have hxx := _root_.Flock.Bridge.NeededBitsOf.eq_of_off hxs hxs'
    (lt_of_lt_of_eq (by decide : 2 < 3) (_root_.Flock.Bridge.NeededBitsOf.gateSize_of_some hab₁).symm)
    (lt_of_lt_of_eq (by decide : 2 < 3) (_root_.Flock.Bridge.NeededBitsOf.gateSize_of_some hab₂).symm) hpq
  subst hxx
  obtain ⟨rfl, rfl, rfl, rfl⟩ := _root_.Flock.Bridge.NeededBitsOf.outGate_inj _ _ hx hx'
  rw [hr] at hr'
  cases hr'
  rfl

end SpotFlock
