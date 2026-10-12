import Proofs.Flock.Soundness.Discharge.Hidden.Facts

/-!
# An output row's shape

`HmOut.checkRows` holds output row `j` to one of two shapes (`checkRows_out`): a lone output port's words zero-padded to
whole SHA-512 blocks, all of them bits, or a bit row (`sha512/row/v2` or `sha512/row-seg/v1`) of exactly its run's words.
A `sha512/row/v1` row is the first (`out_rows_v1`), so its output rows are one per output port (`outRows_single`,
`outGroup_single`) and whole blocks (`ParseFacts.portWords`). `HmRow.rowPorts` derives a bit row's words and compressions
from its length in bits (`rowPorts_v2`).
-/

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Hidden

set_option maxHeartbeats 1000000 in
/-- `HmRow.rowPorts`' bit rows: a row of `bits` bits is `(bits + 15) / 16` u16 leaves; a `sha512/row/v2` row's compressions
its message's blocks after the prefix but the last, a `sha512/row-seg/v1` row's its segments' (`HmRow.segShape`) and its
outer message's but the last. -/
theorem rowPorts_v2 {mj : Lean.Json} {typed : Bool} {ports : Array Port} (h : HmRow.rowPorts mj typed = .ok ports) :
    ∀ p ∈ ports.toList, p.v2 = true →
      p.words = (p.bits + 15) / 16 ∧ (p.segs = none → p.chunks = ((p.bits + 7) / 8 + 17 + 127) / 128 - 1) ∧
        ∀ s, p.segs = some s → s = (HmRow.segShape p.bits).1 ∧
          p.chunks = s.foldl (· + ·.2) 0 + (HmRow.segShape p.bits).2 - 1 := by
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
  · rename_i len seg _
    extract_lets (config := { underBinder := false, onlyGivenNames := true }) serial segs want jpw at hf
    obtain ⟨hbad, hf⟩ := ite_throw_ok' hf
    simp (config := { zeta := false }) only [jpw] at hf
    cases hf
    simp only [want, Bool.or_eq_true, decide_eq_true_eq, bne_iff_ne, ne_eq, not_or, Decidable.not_not,
      Prod.mk.injEq] at hbad
    obtain ⟨hmax, hw, hc⟩ := hbad
    refine ⟨hw, ?_, ?_⟩
    · intro hn
      simp only [Option.map_eq_none_iff] at hn
      simp only [hn] at hc hmax
      cases seg
      · exact hc
      · simp at hmax hc
    · intro s hs
      simp only [Option.map_eq_some_iff] at hs
      obtain ⟨⟨s', o⟩, hso, rfl⟩ := hs
      simp only [hso] at hc
      simp only [segs] at hso
      split at hso <;> (try cases hso) <;> (try exact ⟨rfl, hc⟩)

theorem foldl_replicate_one' (n : ℕ) : ∀ a, List.foldl (· + ·) a (List.replicate n 1) = a + n := by
  induction n with
  | zero => intro a; rfl
  | succ n ih => intro a; rw [List.replicate_succ, List.foldl_cons, ih]; omega

theorem foldl_replicate_one (n : ℕ) : (Array.replicate n 1).foldl (· + ·) 0 = n := by
  rw [← Array.foldl_toList, Array.toList_replicate, foldl_replicate_one', Nat.zero_add]

theorem foldl_snd_eq {α : Type} {xs : List (α × ℕ)} :
    ∀ a, List.foldl (· + ·.2) a xs = a + (xs.map (·.2)).sum := by
  induction xs with
  | nil => intro a; rfl
  | cons x xs ih => intro a; rw [List.foldl_cons, ih, List.map_cons, List.sum_cons]; omega

set_option maxHeartbeats 4000000 in
/-- **`HmRow.parse`'s rows are `HmRow.rowPorts`'** of some META, and its output rows' runs cover its output ports. -/
theorem parse_rowPorts {tags : Tags} {bytes : ByteArray} {tables : List (String × ByteArray)} {pre : Array (String × Net)}
    {c : Circuit} (h : HmRow.parse tags bytes tables (pre := pre) = .ok c) :
    (∃ mj typed, HmRow.rowPorts mj typed = .ok c.ports) ∧ c.outRowPorts.foldl (· + ·) 0 = c.outPorts.size := by
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
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpR at h)
  obtain ⟨outRowPorts, hsum, h⟩ : ∃ r, r.foldl (· + ·) 0 = outPorts.size ∧ jpR r = .ok c := by
    split at h
    · exact ⟨_, foldl_replicate_one _, h⟩
    · obtain ⟨_, -, h⟩ := except_bind_ok h
      obtain ⟨r, -, h⟩ := except_bind_ok h
      obtain ⟨hb, h⟩ := ite_throw_ok' h
      exact ⟨r, by simpa only [bne_iff_ne, ne_eq, Decidable.not_not] using hb, h⟩
  simp (config := { zeta := false }) only [jpR] at h
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
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  have hc : c0 = c := Except.ok.inj h
  subst hc
  exact ⟨⟨_, tags.typed, hrp⟩, hsum⟩

/-- **What `HmOut.checkRows` checks of output row `j`**: it is row `nIn + j`, a row of the circuit, and either a padded
lone port (its run one port, its words that port's zero-padded to whole SHA-512 blocks, a multiple of 64, and its bits
all of them; unsegmented) or an exact run (a bit row, `sha512/row/v2` or `sha512/row-seg/v1`, of exactly its run's
words). -/
theorem checkRows_out {c : Circuit} (h : HmOut.checkRows c = .ok ()) {j : ℕ} (hj : j < c.outRowPorts.size) :
    HmOut.nIn c + j < c.ports.size ∧
      (((HmOut.outGroup c j).2.1 = 1 ∧
          (c.ports.getD (HmOut.nIn c + j) default).words = ((HmOut.outGroup c j).2.2.2 + 63) / 64 * 64 ∧
          (c.ports.getD (HmOut.nIn c + j) default).bits = 16 * (c.ports.getD (HmOut.nIn c + j) default).words ∧
          (c.ports.getD (HmOut.nIn c + j) default).segs = none) ∨
        (((c.ports.getD (HmOut.nIn c + j) default).segs.isSome ∨ (c.ports.getD (HmOut.nIn c + j) default).v2) ∧
          (c.ports.getD (HmOut.nIn c + j) default).bits = 16 * (HmOut.outGroup c j).2.2.2)) := by
  unfold HmOut.checkRows at h
  obtain ⟨k, -, h⟩ := except_bind_ok h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) names jp at h
  obtain ⟨hbad, -⟩ := ite_throw_ok' h
  simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, not_or, List.any_eq_true, List.mem_range, not_exists,
    not_and] at hbad
  obtain ⟨⟨hk, -⟩, hrow⟩ := hbad
  have hk' : HmOut.nIn c = k := by unfold HmOut.nIn; omega
  have := (hrow j hj).2
  rw [hk']
  generalize c.ports.getD (k + j) default = q at this ⊢
  refine ⟨by omega, ?_⟩
  cases hs : q.segs with
  | some s =>
    simp only [hs, Option.isSome_some, ↓reduceIte, beq_iff_eq, Bool.not_eq_eq_eq_not, Bool.not_true,
      Bool.not_eq_false] at this
    exact Or.inr ⟨Or.inl rfl, this⟩
  | none =>
    cases hv : q.v2
    · simp only [hs, hv, Option.isSome_none, Bool.false_eq_true, ↓reduceIte, Bool.and_eq_true, beq_iff_eq,
        Bool.not_eq_eq_eq_not, Bool.not_true, Bool.not_eq_false] at this
      exact Or.inl ⟨this.1.1, this.1.2, this.2, rfl⟩
    · simp only [hs, hv, Option.isSome_none, Bool.false_eq_true, ↓reduceIte, Bool.or_eq_true, Bool.and_eq_true,
        beq_iff_eq, Bool.not_eq_eq_eq_not, Bool.not_true, Bool.not_eq_false] at this
      rcases this with h16 | ⟨⟨h1, hw⟩, hb⟩
      · exact Or.inr ⟨Or.inr rfl, h16⟩
      · exact Or.inl ⟨h1, hw, hb, rfl⟩

/-- **Output rows each of one port are one per output port.** -/
theorem outRows_single {c : Circuit} (hsum : c.outRowPorts.foldl (· + ·) 0 = c.outPorts.size)
    (h1' : ∀ j < c.outRowPorts.size, (HmOut.outGroup c j).2.1 = 1) :
    c.outRowPorts = Array.replicate c.outPorts.size 1 := by
  have h1 : ∀ i (hi : i < c.outRowPorts.size), c.outRowPorts[i] = 1 := fun i hi => by
    have := h1' i hi
    simp only [HmOut.outGroup, Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hi, Option.getD_some] at this
    exact this
  have hr : c.outRowPorts = Array.replicate c.outRowPorts.size 1 :=
    Array.ext (by simp) fun i hi _ => by rw [h1 i hi, Array.getElem_replicate]
  rw [hr, foldl_replicate_one] at hsum
  rw [← hsum]
  exact hr

/-- **Output row `j` of one-port runs** is output port `j`'s: its first word `outAt c j`, its words the port's. -/
theorem outGroup_single {c : Circuit} (hr : c.outRowPorts = Array.replicate c.outPorts.size 1) {j : ℕ}
    (hj : j < c.outPorts.size) : HmOut.outGroup c j = (j, 1, outAt c j, (c.outPorts.getD j default).2) := by
  have e1 : (c.outRowPorts.extract 0 j).foldl (· + ·) 0 = j := by
    rw [hr, ← Array.foldl_toList, Array.toList_extract, List.extract_eq_take_drop, Array.toList_replicate,
      List.drop_zero, List.take_replicate, Nat.sub_zero, foldl_replicate_one']
    omega
  have e2 : c.outRowPorts.getD j 0 = 1 := by
    rw [hr, Array.getD_eq_getD_getElem?, Array.getElem?_replicate]
    simp [hj]
  have hl : j < c.outPorts.toList.length := by simpa using hj
  simp only [HmOut.outGroup, e1, e2, Prod.mk.injEq, true_and]
  refine ⟨?_, ?_⟩
  · rw [← Array.foldl_toList, Array.toList_extract, List.extract_eq_take_drop, List.drop_zero, Nat.sub_zero,
      foldl_snd_eq, Nat.zero_add, outAt]
  · rw [← Array.foldl_toList, Array.toList_extract, List.extract_eq_take_drop, Nat.add_sub_cancel_left,
      foldl_snd_eq, Nat.zero_add, List.drop_eq_getElem_cons hl, Array.getD_eq_getD_getElem?,
      Array.getElem?_eq_getElem hj, Option.getD_some]
    simp

/-- **`sha512/row/v1` rows' output rows**: unsegmented, one per output port, each after the input rows and of
`2 · words = 128 · chunks` bytes (`ParseFacts.portWords`): a `sha512/row/v1` row is no bit row, so each output row is a
padded lone port (`checkRows_out`). -/
theorem out_rows_v1 {tags : Tags} {cf : ByteArray} {tables : List (String × ByteArray)} {c : Circuit}
    (hp : HmRow.parse tags cf tables = .ok c) (hc : HmOut.checkRows c = .ok ())
    (hv1 : ∀ p ∈ c.ports.toList, p.v2 = false) :
    (∀ p ∈ c.ports.toList, p.segs = none) ∧ c.outRowPorts = Array.replicate c.outPorts.size 1 ∧
      ∀ j < c.outPorts.size, HmOut.nIn c + j < c.ports.size ∧
        2 * (c.ports.getD (HmOut.nIn c + j) default).words = 128 * (c.ports.getD (HmOut.nIn c + j) default).chunks := by
  have P := parse_facts hp
  have hu : ∀ p ∈ c.ports.toList, p.segs = none := fun p hm => P.v1Segs p hm (hv1 p hm)
  have hr := outRows_single (parse_rowPorts hp).2 fun j hj => by
    obtain ⟨hlt, h1 | ⟨hs, -⟩⟩ := checkRows_out hc hj
    · exact h1.1
    · rw [hu _ (getD_mem_toList (d := default) hlt), hv1 _ (getD_mem_toList (d := default) hlt)] at hs
      simp at hs
  refine ⟨hu, hr, fun j hj => ?_⟩
  have hj' : j < c.outRowPorts.size := by rw [hr, Array.size_replicate]; exact hj
  have hlt := (checkRows_out hc hj').1
  exact ⟨hlt, P.portWords _ (getD_mem_toList (d := default) hlt) (hv1 _ (getD_mem_toList (d := default) hlt))⟩

/-- **Unsegmented output rows of whole blocks**: when the output rows are one per output port, each a multiple of 64
words, each is after the input rows, of `2 · words = 128 · chunks` bytes (`ParseFacts.portWords`; a `sha512/row/v2`
row's by `rowPorts_v2`) and `16 · words` bits (`checkRows_out`: a padded lone port's, or a bit row's of exactly its
port's words). -/
theorem out_rows_whole {tags : Tags} {cf : ByteArray} {tables : List (String × ByteArray)} {c : Circuit}
    (hp : HmRow.parse tags cf tables = .ok c) (hc : HmOut.checkRows c = .ok ())
    (hu : ∀ p ∈ c.ports.toList, p.segs = none) (hr : c.outRowPorts = Array.replicate c.outPorts.size 1)
    (h64 : ∀ j < c.outPorts.size, (c.ports.getD (HmOut.nIn c + j) default).words % 64 = 0) :
    ∀ j < c.outPorts.size, HmOut.nIn c + j < c.ports.size ∧
      2 * (c.ports.getD (HmOut.nIn c + j) default).words = 128 * (c.ports.getD (HmOut.nIn c + j) default).chunks ∧
      (c.ports.getD (HmOut.nIn c + j) default).bits = 16 * (c.ports.getD (HmOut.nIn c + j) default).words := by
  have P := parse_facts hp
  obtain ⟨⟨mj, typed, hrp⟩, -⟩ := parse_rowPorts hp
  intro j hj
  have hj' : j < c.outRowPorts.size := by rw [hr, Array.size_replicate]; exact hj
  have h6 := h64 j hj
  obtain ⟨hlt, hrow⟩ := checkRows_out hc hj'
  have hmem := getD_mem_toList (d := default) hlt
  generalize c.ports.getD (HmOut.nIn c + j) default = p at hmem hrow h6 ⊢
  have hb : p.bits = 16 * p.words := by
    rcases hrow with ⟨-, -, hb, -⟩ | ⟨hs, hb⟩
    · exact hb
    · have hv : p.v2 = true := by
        rcases hs with hs | hs
        · rw [hu p hmem] at hs; cases hs
        · exact hs
      rw [(rowPorts_v2 hrp p hmem hv).1, hb]
      omega
  refine ⟨hlt, ?_, hb⟩
  cases hv : p.v2
  · exact P.portWords p hmem hv
  · rw [(rowPorts_v2 hrp p hmem hv).2.1 (hu p hmem), hb]
    omega

end FlockSoundness.Discharge.Hidden
