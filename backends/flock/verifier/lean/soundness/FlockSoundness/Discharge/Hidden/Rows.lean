import FlockSoundness.Discharge.Hidden.Facts

/-!
# An output row's compressions

`HmOut.delta` writes output row `j`'s message bytes `0 … 2 · words` into the row's `sha512x3` slots, byte `b` in
compression `b / 128`. Each is one of the row's compressions (`out_chunks`: `2 · words = 128 · chunks`): a
`sha512/row/v1` row's by `ParseFacts.portWords`; a `sha512/row/v2` row's because `HmRow.rowPorts` derives its
compressions from its length in bits (`rowPorts_v2`), which `HmOut.checkRows` holds to `16 · words`, its words whole
SHA-512 blocks (`checkRows_out`).
-/

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Hidden

set_option maxHeartbeats 1000000 in
/-- `HmRow.rowPorts`' `sha512/row/v2` ports: a row of `bits` bits is `(bits + 15) / 16` u16 leaves and the message's
blocks after the prefix but the last. -/
theorem rowPorts_v2 {mj : Lean.Json} {typed : Bool} {ports : Array Port} (h : HmRow.rowPorts mj typed = .ok ports) :
    ∀ p ∈ ports.toList, p.v2 = true →
      p.words = (p.bits + 15) / 16 ∧ p.chunks = ((p.bits + 7) / 8 + 17 + 127) / 128 - 1 := by
  unfold HmRow.rowPorts at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) v2 one at h
  obtain ⟨s, -, h⟩ := except_bind_ok h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp0 at h
  replace h := ite_throw_ok h
  simp (config := { zeta := false }) only [jp0] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp1 at h
  replace h := ite_throw_ok h
  simp (config := { zeta := false }) only [jp1] at h
  obtain ⟨listedPorts, -, h⟩ := except_bind_ok h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp2 at h
  obtain ⟨lengths, h⟩ : ∃ l, jp2 l = .ok ports := by
    split at h
    · obtain ⟨s, -, h⟩ := except_bind_ok h
      extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpv at h
      replace h := ite_throw_ok h
      simp (config := { zeta := false }) only [jpv] at h
      obtain ⟨l, -, h⟩ := except_bind_ok h
      exact ⟨l, h⟩
    · exact ⟨_, h⟩
  simp (config := { zeta := false }) only [jp2] at h
  obtain ⟨ports', hmap, h⟩ := except_bind_ok h
  obtain ⟨listed, -, h⟩ := except_bind_ok h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) bad jp3 at h
  replace h := ite_throw_ok h
  simp (config := { zeta := false }) only [jp3] at h
  obtain ⟨u, -, h⟩ := except_bind_ok h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp4 at h
  replace h := ite_throw_ok h
  simp (config := { zeta := false }) only [jp4] at h
  cases h
  intro p hp hv
  obtain ⟨i, -, hf⟩ := array_mapM_ok hmap p hp
  obtain ⟨a, -, hf⟩ := except_bind_ok hf
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpa at hf
  replace hf := ite_throw_ok hf
  simp (config := { zeta := false }) only [jpa] at hf
  obtain ⟨name, -, hf⟩ := except_bind_ok hf
  obtain ⟨words, -, hf⟩ := except_bind_ok hf
  obtain ⟨comps, -, hf⟩ := except_bind_ok hf
  split at hf
  · extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpn at hf
    replace hf := ite_throw_ok hf
    simp (config := { zeta := false }) only [jpn] at hf
    cases hf
    cases hv
  · rename_i l _
    extract_lets (config := { underBinder := false, onlyGivenNames := true }) want jpw at hf
    obtain ⟨hbad, hf⟩ := ite_throw_ok' hf
    simp (config := { zeta := false }) only [jpw] at hf
    cases hf
    simp only [want, Bool.or_eq_true, decide_eq_true_eq, bne_iff_ne, ne_eq, not_or, Decidable.not_not,
      Prod.mk.injEq] at hbad
    exact ⟨hbad.2.1, hbad.2.2⟩

set_option maxHeartbeats 4000000 in
/-- **`HmRow.parse`'s rows are `HmRow.rowPorts`'** of some META. -/
theorem parse_rowPorts {tags : Tags} {bytes : ByteArray} {tables : List (String × ByteArray)} {pre : Array (String × Net)}
    {c : Circuit} (h : HmRow.parse tags bytes tables (pre := pre) = .ok c) :
    ∃ mj typed, HmRow.rowPorts mj typed = .ok c.ports := by
  unfold HmRow.parse at h
  simp (config := { zeta := false }) only [Option.isSome_none, Bool.false_eq_true, ↓reduceIte, Option.mapM_none] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) sha jp0 at h
  split at h
  swap
  · obtain ⟨_, hte, -⟩ := except_bind_ok h; cases hte
  (have hh := except_bind_ok h; clear h; obtain ⟨text, hte, h⟩ := hh)
  cases hte
  simp (config := { zeta := false }) only [jp0] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) lines0 lines head jp1 at h
  replace h := ite_throw_ok h
  simp (config := { zeta := false }) only [jp1] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) name metaLine jp2 at h
  replace h := ite_throw_ok h
  simp (config := { zeta := false }) only [jp2] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp3 at h
  split at h
  swap
  · obtain ⟨_, hte, -⟩ := except_bind_ok h; cases hte
  (have hh := except_bind_ok h; clear h; obtain ⟨mj, hte, h⟩ := hh)
  cases hte
  simp (config := { zeta := false }) only [jp3] at h
  (have hh := except_bind_ok h; clear h; obtain ⟨f1, hf1, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨f2, hf2, h⟩ := hh)
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp4 at h
  replace h := ite_throw_ok h
  simp (config := { zeta := false }) only [jp4] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp5 jp6 at h
  obtain ⟨classSha, h⟩ : ∃ classSha, jp5 classSha = .ok c := by
    split at h
    · replace h := ite_throw_ok h
      simp (config := { zeta := false }) only [jp6] at h
      extract_lets (config := { underBinder := false, onlyGivenNames := true }) part rule jp7 at h
      replace h := ite_throw_ok h
      simp (config := { zeta := false }) only [jp7] at h
      obtain ⟨cs, -, h⟩ := except_bind_ok h
      exact ⟨cs, h⟩
    · obtain ⟨cs, -, h⟩ := except_bind_ok h
      exact ⟨cs, h⟩
  simp (config := { zeta := false }) only [jp5] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) nets0 at0 at h
  (have hh := except_bind_ok h; clear h; obtain ⟨s1, hwhile, h⟩ := hh)
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) nets1 at h
  (have hh := except_bind_ok h; clear h; obtain ⟨u1, -, h⟩ := hh)
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpT at h
  simp (config := { zeta := false }) only [jpT] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpL at h
  obtain ⟨lks, h⟩ : ∃ lks, jpL lks = .ok c := by
    split at h <;> (obtain ⟨l, -, h⟩ := except_bind_ok h; exact ⟨l, h⟩)
  simp (config := { zeta := false }) only [jpL] at h
  (have hh := except_bind_ok h; clear h; obtain ⟨s2, hlook, h⟩ := hh)
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) nets2 at h
  (have hh := except_bind_ok h; clear h; obtain ⟨ports, hrp, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpK at h
  replace h := ite_throw_ok h
  simp (config := { zeta := false }) only [jpK] at h
  (have hh := except_bind_ok h; clear h; obtain ⟨g, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨upv, -, h⟩ := hh)
  split at h
  swap
  · cases h
  rename_i unitNet hunit
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) netOf at h)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨ranges, hranges, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨leavesIn, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨leavesOut, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨dummy, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨dummyDigests, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨dummyOutputs, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨outPorts, -, h⟩ := hh)
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) perVu0 jpP at h
  obtain ⟨pv, h⟩ : ∃ pv, jpP () pv = .ok c := by
    split at h
    · obtain ⟨p', -, h⟩ := except_bind_ok h
      exact ⟨_, h⟩
    · exact ⟨_, h⟩
  simp (config := { zeta := false }) only [jpP] at h
  (have hh := except_bind_ok h; clear h; obtain ⟨shaNet, hsha, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨hmNet, hhm, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨outNet, -, h⟩ := hh)
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpO at h)
  obtain ⟨onp, h⟩ : ∃ x, jpO x = .ok c := by
    split at h <;> (obtain ⟨l, -, h⟩ := except_bind_ok h; exact ⟨l, h⟩)
  simp (config := { zeta := false }) only [jpO] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpC at h
  obtain ⟨lc, h⟩ : ∃ x, jpC x = .ok c := by
    split at h <;> (obtain ⟨l, -, h⟩ := except_bind_ok h; exact ⟨l, h⟩)
  simp (config := { zeta := false }) only [jpC] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpW at h
  obtain ⟨wires, h⟩ : ∃ x, jpW x = .ok c := by
    split at h
    · obtain ⟨_, -, h⟩ := except_bind_ok h
      obtain ⟨l, -, h⟩ := except_bind_ok h
      exact ⟨l, h⟩
    · obtain ⟨l, -, h⟩ := except_bind_ok h
      exact ⟨l, h⟩
  simp (config := { zeta := false }) only [jpW] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) derived wiresT rowNet at h
  obtain ⟨hwires, h⟩ := ite_throw_ok' h
  (have hh := except_bind_ok h; clear h; obtain ⟨ls, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨leafId, -, h⟩ := hh)
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpF at h
  replace h := ite_throw_ok h
  simp (config := { zeta := false }) only [jpF] at h
  (have hh := except_bind_ok h; clear h; obtain ⟨kLog, -, h⟩ := hh)
  simp (config := { zeta := false }) only [pure_bind] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) c0 at h
  (have hh := except_bind_ok h; clear h; obtain ⟨w, hcheck, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨w', hrows, h⟩ := hh)
  have hc : c0 = c := Except.ok.inj h
  subst hc
  exact ⟨_, tags.typed, hrp⟩

/-- **What `HmOut.checkRows` checks of output row `j`**: it is row `nIn + j`, a row of the circuit, its words whole
SHA-512 blocks (a multiple of 64) and its bits all of them. -/
theorem checkRows_out {c : Circuit} (h : HmOut.checkRows c = .ok ()) {j : ℕ} (hj : j < c.outPorts.size) :
    HmOut.nIn c + j < c.ports.size ∧ (c.ports.getD (HmOut.nIn c + j) default).words % 64 = 0 ∧
      (c.ports.getD (HmOut.nIn c + j) default).bits = 16 * (c.ports.getD (HmOut.nIn c + j) default).words := by
  unfold HmOut.checkRows at h
  obtain ⟨k, -, h⟩ := except_bind_ok h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) names jp at h
  obtain ⟨hbad, -⟩ := ite_throw_ok' h
  simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, not_or, List.any_eq_true, List.mem_range, not_exists,
    not_and] at hbad
  obtain ⟨⟨hk, -⟩, hrow⟩ := hbad
  have hk' : HmOut.nIn c = k := by unfold HmOut.nIn; omega
  have := hrow j hj
  rw [hk']
  generalize c.outPorts[j]! = e at this
  obtain ⟨nm, w⟩ := e
  simp only [Decidable.not_not] at this
  refine ⟨by omega, ?_, this.2⟩
  rw [this.1.2]
  omega

/-- **Output row `j`'s message bytes are its compressions'**: `2 · words = 128 · chunks`, for a `sha512/row/v1` or a
`sha512/row/v2` row. -/
theorem out_chunks {tags : Tags} {cf : ByteArray} {tables : List (String × ByteArray)} {c : Circuit}
    (hp : HmRow.parse tags cf tables = .ok c) (hc : HmOut.checkRows c = .ok ()) {j : ℕ} (hj : j < c.outPorts.size) :
    HmOut.nIn c + j < c.ports.size ∧
      2 * (c.ports.getD (HmOut.nIn c + j) default).words = 128 * (c.ports.getD (HmOut.nIn c + j) default).chunks := by
  obtain ⟨hlt, h64, hbits⟩ := checkRows_out hc hj
  refine ⟨hlt, ?_⟩
  have hmem : c.ports.getD (HmOut.nIn c + j) default ∈ c.ports.toList := by
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hlt, Option.getD_some]
    exact Array.getElem_mem_toList hlt
  generalize c.ports.getD (HmOut.nIn c + j) default = p at hmem h64 hbits ⊢
  cases hv : p.v2
  · exact (parse_facts hp).portWords p hmem hv
  · obtain ⟨mj, typed, hr⟩ := parse_rowPorts hp
    obtain ⟨-, hch⟩ := rowPorts_v2 hr p hmem hv
    rw [hch, hbits]
    omega

end FlockSoundness.Discharge.Hidden
