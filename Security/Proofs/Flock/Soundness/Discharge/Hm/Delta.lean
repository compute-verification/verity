/-
Δ in a typed template statement, pair by pair (`delta_rows`): every slot's constant from the pin (`constPairs`), then
each row's constant bits (`rowPairs`: its first `cv` the row prefix's midstate, its `hm96` slot's `pad` the padding
block, bit 1 from the pin and bit 0 from the unit range's forced-zero row), the units' leaf and cut bits, every wire's
destination bits from its source's, then the typed tail. Δ writes each row's constant bits as `rowPairsOf`, which for
`sha512/row/v1` rows (`16 · words` bits in at least one chunk) are `rowPairs`'s (`rowPairsAll_v1`).
-/
import Proofs.Flock.Soundness.Discharge.Hm.Accept
import Proofs.Flock.Soundness.ExecFlatCopies

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Hm

/-- The unit range's first slot's forced-zero row. -/
def zeroCol (c : Circuit) : ℕ := c.unitRange.slot 0 + c.unit.useful

/-- Template instance `g`, port `p`: its first compression's `cv` input's first column. -/
def cvCol (c : Circuit) (g p : ℕ) : ℕ :=
  c.slotCol c.shaNet g ((HmRow.rowSlots c.ports).2[p]!)[0]!.1 +
    (portCols (c.nets.getD c.shaNet default).2.inGroups)[2 * ((HmRow.rowSlots c.ports).2[p]!)[0]!.2]!.1

/-- Template instance `g`, port `p`: its `hm96` slot's `pad` input's first column. -/
def padCol (c : Circuit) (g p : ℕ) : ℕ := c.slotCol c.hmNet g p + (portCols (c.nets.getD c.hmNet default).2.inGroups)[1]!.1

/-- A constant bit's source: the pin for bit 1, the forced-zero row for bit 0. -/
def constSrc (c : Circuit) (pin : ℕ) (ws : Array UInt64) (t : ℕ) : ℕ := if wordsBit ws t then pin else zeroCol c

/-- Δ's pairs at the rows' constant bits. -/
def rowPairs (c : Circuit) (pin : ℕ) : List (ℕ × ℕ) :=
  (List.range c.g).flatMap fun g => (List.range (HmRow.rowSlots c.ports).2.size).flatMap fun p =>
    (List.range 512).flatMap (fun t => [(cvCol c g p + t, cvCol c g p + t),
        (cvCol c g p + t, constSrc c pin (HmRow.midstate c.ports[p]!.words) t)]) ++
      (List.range 1024).flatMap fun t => [(padCol c g p + t, padCol c g p + t),
        (padCol c g p + t, constSrc c pin (HmRow.padWords c.ports[p]!.words) t)]

/-- Template instance `g`, port `p`: its row's first `cv` input's first column, its first compression's or, for a row of
no compression, its `hm96` slot's. -/
def cvColOf (c : Circuit) (g p : ℕ) : ℕ :=
  if 0 < ((HmRow.rowSlots c.ports).2[p]!).size then cvCol c g p
  else c.slotCol c.hmNet g p + (portCols (c.nets.getD c.hmNet default).2.inGroups)[0]!.1

/-- Template instance `g`, port `p`: its compression `k`'s `msg` input's first column. -/
def msgColOf (c : Circuit) (g p k : ℕ) : ℕ :=
  c.slotCol c.shaNet g ((HmRow.rowSlots c.ports).2[p]!)[k]!.1 +
    (portCols (c.nets.getD c.shaNet default).2.inGroups)[2 * ((HmRow.rowSlots c.ports).2[p]!)[k]!.2 + 1]!.1

/-- Δ's pairs at column `x` for a constant bit `HmRow.constBit` gives, none at a row bit. -/
def bitPairs (c : Circuit) (pin x : ℕ) : Option Bool → List (ℕ × ℕ)
  | some v => [(x, x), (x, if v then pin else zeroCol c)]
  | none => []

/-- Δ's pairs at template instance `g`'s row `p`'s constant bits, as `HmRow.delta` writes them: its first `cv` the row
prefix's midstate, its `hm96` slot's `pad` the constant bits of its last block, and each compression's `msg` from the
first that holds a bit at or past the row's length its constant bits. -/
def rowPairsOf (c : Circuit) (pin g p : ℕ) : List (ℕ × ℕ) :=
  (List.range 512).flatMap (fun t => [(cvColOf c g p + t, cvColOf c g p + t),
      (cvColOf c g p + t, constSrc c pin (HmRow.midstateOf c.ports[p]!) t)]) ++
    (List.range 1024).flatMap (fun t => bitPairs c pin (padCol c g p + t)
      (HmRow.constBit c.ports[p]! (HmRow.tailOf c.ports[p]!) ((HmRow.rowSlots c.ports).2[p]!).size t)) ++
    (List.range' (c.ports[p]!.bits / 1024) (((HmRow.rowSlots c.ports).2[p]!).size - c.ports[p]!.bits / 1024)).flatMap
      fun k => (List.range 1024).flatMap fun t => bitPairs c pin (msgColOf c g p k + t)
        (HmRow.constBit c.ports[p]! (HmRow.tailOf c.ports[p]!) k t)

/-- Δ's pairs at the rows' constant bits, as `HmRow.delta` writes them. -/
def rowPairsAll (c : Circuit) (pin : ℕ) : List (ℕ × ℕ) :=
  (List.range c.g).flatMap fun g => (List.range (HmRow.rowSlots c.ports).2.size).flatMap fun p => rowPairsOf c pin g p

theorem flatMap_congr' {α β : Type} {l : List α} {f f' : α → List β} (h : ∀ x ∈ l, f x = f' x) :
    l.flatMap f = l.flatMap f' := by
  induction l with
  | nil => rfl
  | cons a l ih => rw [List.flatMap_cons, List.flatMap_cons, h a List.mem_cons_self,
      ih fun x hx => h x (List.mem_cons_of_mem _ hx)]

theorem forIn_range'_append {γ : Type} (G : ℕ → List γ) (a b : ℕ)
    {f : ℕ → Array γ → Id (ForInStep (Array γ))}
    (hf : ∀ k, a ≤ k → k < b → ∀ d, ∃ d', f k d = ForInStep.yield d' ∧ d'.toList = d.toList ++ G k)
    (d : Array γ) :
    (forIn [a:b] d f).toList = d.toList ++ (List.range' a (b - a)).flatMap G := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  have := forIn_id_append G (List.range' a [a:b].size) (fun k hk d => hf k
    (by simp only [Std.Legacy.Range.size, List.mem_range'_1] at hk; omega)
    (by simp only [Std.Legacy.Range.size, List.mem_range'_1] at hk; omega) d) d
  simpa [Std.Legacy.Range.size] using this

theorem padTail_eq {total : ℕ} (h : total % 128 = 0) : HmRow.padTail total = HmRow.padBlock total := by
  simp only [HmRow.padTail, HmRow.padBlock, h]

/-- A `sha512/row/v1` row's (`2 · words = 128 n` row bytes, of 16 bits each) constant bits past its `n` blocks are the
padding block's. -/
theorem constBit_v1 {p : Port} {n t : ℕ} (hv : p.v2 = false) (hs : p.segs = none) (hb : p.bits = 16 * p.words)
    (hw : 2 * p.words = 128 * n) (ht : t < 1024) :
    HmRow.constBit p (HmRow.tailOf p) n t = some (wordsBit (HmRow.padWords p.words) t) := by
  have hby : p.bytes = 2 * p.words := by simp [Port.bytes, hv]
  have hj : (HmRow.blockBit t).1 = 8 * (t / 64) + 7 - t % 64 / 8 := rfl
  have hb8 : (HmRow.blockBit t).2 = t % 8 := rfl
  unfold HmRow.constBit HmRow.tailOf
  simp only [hby, hb, hs]
  rw [if_neg (by rw [hj, hb8]; omega), padTail_eq (by omega), HmRow.padWords,
    wordsBit_eq _ (n := 16) (by omega), decide_eq_true (by omega), Bool.true_and,
    show 128 * n + (HmRow.blockBit t).1 - 2 * p.words = 8 * (t / 64) + 7 - t % 64 / 8 by rw [hj]; omega, hb8, wbit,
    List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range (by omega), Option.map_some, Option.getD_some,
    wordsBE_getElem! (by omega), getLsbD_wordL _ _ _ (by omega), get!_eq_getD,
    show 0 + 8 * (t / 64) + (7 - t % 64 / 8) = 8 * (t / 64) + 7 - t % 64 / 8 by omega,
    show t % 64 % 8 = t % 8 by omega]
  congr 1
  generalize ((HmRow.padBlock (128 + 2 * p.words)).data.toList.getD (8 * (t / 64) + 7 - t % 64 / 8) 0).toNat = x
  rw [Nat.testBit, Nat.one_and_eq_mod_two]
  rcases Nat.mod_two_eq_zero_or_one (x >>> (t % 8)) with h | h <;> simp [h]

/-- **A `sha512/row/v1` row's constant bits**: with `2 · words = 128 · chunks` 16-bit row bytes in `chunks > 0`
compressions, Δ's pairs at its constant bits are its first `cv`'s midstate and its `pad`'s padding block. -/
theorem rowPairsOf_v1 (c : Circuit) (pin g : ℕ) {p : ℕ} (hp : p < c.ports.size) (hv : c.ports[p]!.v2 = false)
    (hs : c.ports[p]!.segs = none)
    (hb : c.ports[p]!.bits = 16 * c.ports[p]!.words) (hn : 0 < c.ports[p]!.chunks)
    (hw : 2 * c.ports[p]!.words = 128 * c.ports[p]!.chunks) :
    rowPairsOf c pin g p = (List.range 512).flatMap (fun t => [(cvCol c g p + t, cvCol c g p + t),
        (cvCol c g p + t, constSrc c pin (HmRow.midstate c.ports[p]!.words) t)]) ++
      (List.range 1024).flatMap fun t => [(padCol c g p + t, padCol c g p + t),
        (padCol c g p + t, constSrc c pin (HmRow.padWords c.ports[p]!.words) t)] := by
  have hsz := rowSlots_size hp
  have hcv : cvColOf c g p = cvCol c g p := by rw [cvColOf, hsz, if_pos hn]
  have hms : HmRow.midstateOf c.ports[p]! = HmRow.midstate c.ports[p]!.words := by
    simp only [HmRow.midstateOf, HmRow.prefixOf, hv, hs, Option.isSome_none, Bool.false_eq_true, ite_false,
      HmRow.midstate]
  have hk : c.ports[p]!.bits / 1024 = c.ports[p]!.chunks := by omega
  unfold rowPairsOf
  simp only [hcv, hms, hsz, hk, Nat.sub_self, List.range'_zero, List.flatMap_nil, List.append_nil]
  rw [List.append_right_inj]
  exact flatMap_congr' fun t ht => by simp only [constBit_v1 hv hs hb hw (List.mem_range.1 ht), bitPairs, constSrc]

theorem rowPairsAll_v1 (c : Circuit) (pin : ℕ)
    (hv1 : ∀ p ∈ c.ports.toList, p.v2 = false ∧ p.bits = 16 * p.words ∧ 0 < p.chunks)
    (hw : ∀ p ∈ c.ports.toList, p.v2 = false → 2 * p.words = 128 * p.chunks)
    (hs : ∀ p ∈ c.ports.toList, p.v2 = false → p.segs = none) :
    rowPairsAll c pin = rowPairs c pin := by
  have hS : (HmRow.rowSlots c.ports).2.size = c.ports.size := by
    rw [← Array.length_toList, rowSlots_eq, List.length_map, List.length_range]
  unfold rowPairsAll rowPairs
  refine flatMap_congr' fun g _ => flatMap_congr' fun p hp => ?_
  have hp' : p < c.ports.size := hS ▸ List.mem_range.1 hp
  have hm := getElem_mem_toList' hp'
  obtain ⟨hv, hb, hn⟩ := hv1 _ hm
  exact rowPairsOf_v1 c pin g hp' hv (hs _ hm hv) hb hn (hw _ hm hv)

/-- A row of a statement of `sha512/row/v1` rows is not segmented. -/
theorem segs_none_of_v1 {c : Circuit} (hv1 : ∀ p ∈ c.ports.toList, p.v2 = false ∧ p.bits = 16 * p.words ∧ 0 < p.chunks)
    (hs : ∀ p ∈ c.ports.toList, p.v2 = false → p.segs = none) {p : ℕ} (hp : p < (HmRow.rowSlots c.ports).2.size) :
    c.ports[p]!.segs = none := by
  have hS : (HmRow.rowSlots c.ports).2.size = c.ports.size := by
    rw [← Array.length_toList, rowSlots_eq, List.length_map, List.length_range]
  have hm := getElem_mem_toList' (hS ▸ hp)
  exact hs _ hm (hv1 _ hm).1

/-- Δ's pairs, before the typed tail. -/
def flatPairs (c : Circuit) (pin : ℕ) : List (ℕ × ℕ) :=
  constPairs c pin ++ rowPairs c pin ++ Flat.leafPairs c ++ Flat.cutPairs c ++ Flat.wirePairs c

theorem delta_rows (c : Circuit) (pin : ℕ) {tb : TypedBlock} (ht : c.typed = some tb)
    (hv1 : ∀ p ∈ c.ports.toList, p.v2 = false ∧ p.bits = 16 * p.words ∧ 0 < p.chunks)
    (hw : ∀ p ∈ c.ports.toList, p.v2 = false → 2 * p.words = 128 * p.chunks)
    (hs : ∀ p ∈ c.ports.toList, p.v2 = false → p.segs = none) :
    (HmRow.delta c pin).1.toList = flatPairs c pin ++ Typed.tail c tb true ∧
      (HmRow.delta c pin).2.toList = flatPairs c pin ++ Typed.tail c tb false := by
  unfold HmRow.delta
  simp only
  refine FlockSoundness.run_bind_pair (Q := fun r => r.1.toList = flatPairs c pin ++ Typed.tail c tb true ∧
      r.2.toList = flatPairs c pin ++ Typed.tail c tb false) (P := fun d => d.toList = constPairs c pin) ?_ ?_
  · rw [← Array.forIn_toList]
    refine (forIn_id_append (slotPairs c pin) _ (fun r _ d => ?_) #[]).trans ?_
    rotate_left
    · simp [constPairs]
    · split
      · rename_i i hk
        refine ⟨_, rfl, ?_⟩
        simp only [slotPairs, hk]
        rw [Std.Legacy.Range.forIn_eq_forIn_range', forIn_id_append (fun s =>
          if (r.slot s + (c.nets.getD i default).2.constPos != pin) = true then
            [(r.slot s + (c.nets.getD i default).2.constPos, r.slot s + (c.nets.getD i default).2.constPos),
              (r.slot s + (c.nets.getD i default).2.constPos, pin)] else []) _ (fun s _ d => ?_) d]
        · simp [List.range_eq_range']
        · split
          · exact ⟨_, rfl, by simp⟩
          · exact ⟨_, rfl, by simp⟩
      · rename_i hk
        refine ⟨_, rfl, ?_⟩
        unfold slotPairs
        split
        · rename_i i hi; exact absurd hi (hk i)
        · simp
  intro d hd
  -- the rows' constant bits
  refine FlockSoundness.run_bind_pair (Q := fun r => r.1.toList = flatPairs c pin ++ Typed.tail c tb true ∧
      r.2.toList = flatPairs c pin ++ Typed.tail c tb false) (P := fun d' => d'.toList = d.toList ++ rowPairs c pin)
    ?_ ?_
  · rw [← rowPairsAll_v1 c pin hv1 hw hs]
    refine forIn_range_append _ _ (fun g _ d => ⟨_, rfl, ?_⟩) d
    refine forIn_range_append _ _ (fun p hp d => ?_) d
    rw [segs_none_of_v1 hv1 hs hp]
    refine ⟨_, rfl, ?_⟩
    rw [rowPairsOf, ← List.append_assoc, ← List.append_assoc]
    refine Eq.trans (forIn_range'_append (fun k => (List.range 1024).flatMap fun t => bitPairs c pin
        (msgColOf c g p k + t) (HmRow.constBit c.ports[p]! (HmRow.tailOf c.ports[p]!) k t)) _ _
        (fun k _ _ d => ⟨_, rfl, ?_⟩) _) ?_
    · refine forIn_range_append _ _ (fun t _ d => ?_) d
      split
      · rename_i v hv
        have hv' : HmRow.constBit c.ports[p]! (HmRow.tailOf c.ports[p]!) k t = some v := hv
        refine ⟨_, rfl, ?_⟩
        rw [hv']
        simp only [bitPairs, Array.toList_push, List.append_assoc, List.cons_append, List.nil_append]
        rfl
      · rename_i hv
        have hv' : HmRow.constBit c.ports[p]! (HmRow.tailOf c.ports[p]!) k t = none :=
          Option.eq_none_iff_forall_ne_some.mpr hv
        refine ⟨_, rfl, ?_⟩
        rw [hv']
        simp only [bitPairs, List.append_nil]
    refine (List.append_left_inj _).mpr ?_
    refine Eq.trans (forIn_range_append (fun t => bitPairs c pin (padCol c g p + t)
        (HmRow.constBit c.ports[p]! (HmRow.tailOf c.ports[p]!) ((HmRow.rowSlots c.ports).2[p]!).size t)) 1024
        (fun t _ d => ?_) _) ?_
    · split
      · rename_i v hv
        have hv' : HmRow.constBit c.ports[p]! (HmRow.tailOf c.ports[p]!) ((HmRow.rowSlots c.ports).2[p]!).size t =
          some v := hv
        refine ⟨_, rfl, ?_⟩
        rw [hv']
        simp only [bitPairs, Array.toList_push, List.append_assoc, List.cons_append, List.nil_append]
        rfl
      · rename_i hv
        have hv' : HmRow.constBit c.ports[p]! (HmRow.tailOf c.ports[p]!) ((HmRow.rowSlots c.ports).2[p]!).size t =
          none := Option.eq_none_iff_forall_ne_some.mpr hv
        refine ⟨_, rfl, ?_⟩
        rw [hv']
        simp only [bitPairs, List.append_nil]
    refine (List.append_left_inj _).mpr ?_
    refine forIn_range_append _ _ (fun t _ d => ⟨_, rfl, ?_⟩) d
    simp only [Array.toList_push, List.append_assoc, List.cons_append, List.nil_append]
    rfl
  intro d0 hd0
  -- the units' leaf bits
  refine FlockSoundness.run_bind_pair (Q := fun r => r.1.toList = flatPairs c pin ++ Typed.tail c tb true ∧
      r.2.toList = flatPairs c pin ++ Typed.tail c tb false) (P := fun d' => d'.toList = d0.toList ++ Flat.leafPairs c)
    ?_ ?_
  · refine forIn_range_append _ _ (fun k _ d => ⟨_, rfl, ?_⟩) d0
    refine forIn_range_append _ _ (fun j _ d => ⟨_, rfl, ?_⟩) d
    refine forIn_range_append _ _ (fun t _ d => ⟨_, rfl, ?_⟩) d
    simp only [Array.toList_push, List.append_assoc, List.cons_append, List.nil_append]
    rfl
  intro d1 hd1
  -- the units' leaf cuts
  refine FlockSoundness.run_bind_pair (Q := fun r => r.1.toList = flatPairs c pin ++ Typed.tail c tb true ∧
      r.2.toList = flatPairs c pin ++ Typed.tail c tb false) (P := fun d' => d'.toList = d1.toList ++ Flat.cutPairs c)
    ?_ ?_
  · refine forIn_range_append _ _ (fun g _ d => ⟨_, rfl, ?_⟩) d1
    refine forIn_array_append _ _ (fun e _ d => ⟨_, rfl, ?_⟩) d
    refine forIn_range_append _ _ (fun t _ d => ⟨_, rfl, ?_⟩) d
    simp only [Array.toList_push, List.append_assoc, List.cons_append, List.nil_append]
    rfl
  intro d2 hd2
  -- the wires
  refine FlockSoundness.run_bind_pair (Q := fun r => r.1.toList = flatPairs c pin ++ Typed.tail c tb true ∧
      r.2.toList = flatPairs c pin ++ Typed.tail c tb false) (P := fun d' => d'.toList = d2.toList ++ Flat.wirePairs c)
    ?_ ?_
  · refine forIn_range_append _ _ (fun g _ d => ⟨_, rfl, ?_⟩) d2
    refine forIn_array_append _ _ (fun w _ d => ⟨_, rfl, ?_⟩) d
    refine forIn_range_append _ _ (fun t _ d => ⟨_, rfl, ?_⟩) d
    simp only [Array.toList_push, List.append_assoc, List.cons_append, List.nil_append]
    rfl
  intro d3 hd3
  have hflat : d3.toList = flatPairs c pin := by
    rw [hd3, hd2, hd1, hd0, hd]
    rfl
  -- the typed tail
  simp only [ht]
  rw [← hflat]
  have H := fun f hf => forIn_range_append2 (f := f) (Typed.tailVU c tb true) (Typed.tailVU c tb false) c.g hf (d3, d3)
  refine ⟨(H _ ?hf).1, (H _ ?hf).2⟩
  intro g _ s
  refine ⟨_, rfl, ?_⟩
  have HW := fun f hf => forIn_range_append2 (f := f)
    (fun w => [(Typed.col c tb g tb.inCols[w]!, Typed.col c tb g tb.inCols[w]!),
      (Typed.col c tb g tb.inCols[w]!, Typed.copySrc c g w)])
    (fun w => [(Typed.col c tb g tb.inCols[w]!, Typed.col c tb g tb.inCols[w]!),
      (Typed.col c tb g tb.inCols[w]!, Typed.copySrc c g w)]) tb.inCols.size hf
  have HB := fun f hf => forIn_array_append2 (f := f)
    (fun rs : ℕ × Array ℕ => (Typed.col c tb g rs.1, Typed.col c tb g rs.1) ::
      rs.2.toList.map fun x => (Typed.col c tb g rs.1, Typed.col c tb g x))
    (fun rs : ℕ × Array ℕ => (Typed.col c tb g rs.1, Typed.col c tb g rs.1) ::
      rs.2.toList.map fun x => (Typed.col c tb g rs.1, Typed.col c tb g x)) tb.bind hf
  have HC := fun f hf => forIn_array_append2 (f := f)
    (fun e : ℕ × Array ℕ × Array ℕ => e.2.1.toList.map fun x => (Typed.col c tb g e.1, Typed.col c tb g x))
    (fun e : ℕ × Array ℕ × Array ℕ => e.2.2.toList.map fun x => (Typed.col c tb g e.1, Typed.col c tb g x)) tb.cross hf
  constructor
  all_goals first
    | refine (HC _ ?_ _).1.trans ?_
    | refine (HC _ ?_ _).2.trans ?_
  all_goals first
    | (intro e _ t
       exact ⟨_, rfl, by (simp only [Array.toList_append, Array.toList_map]; rfl),
         by (simp only [Array.toList_append, Array.toList_map]; rfl)⟩)
    | skip
  all_goals first
    | (dsimp only; rw [(HB _ ?_ _).1, (HW _ ?_ _).1])
    | (dsimp only; rw [(HB _ ?_ _).2, (HW _ ?_ _).2])
  all_goals first
    | (simp [Typed.tailVU]; done)
    | (intro rs _ t
       exact ⟨_, rfl,
         by (simp only [Array.toList_append, Array.toList_push, Array.toList_map, List.append_assoc]; rfl),
         by (simp only [Array.toList_append, Array.toList_push, Array.toList_map, List.append_assoc]; rfl)⟩)

end FlockSoundness.Discharge.Hm
