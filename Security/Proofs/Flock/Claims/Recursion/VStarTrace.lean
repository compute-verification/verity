import Proofs.Flock.Claims.Recursion.VStarBuild
import VStar.Trace

/-!
# The trace assigner places the port inputs (`places_assign`)

The builder `VStarBuild.vstarWit` runs an assigner `A` on a part's circuit at its port inputs. Here `A` is the
trace the deployed C-Flock prover assigns (`VStar.Trace.assign`, difftested bit for bit against `flock-circuit`'s
`Stmt::witness` by `verity/core/service/vstar/tests/test_vstar_trace_difftest.py`), the empty witness where the assigner
refuses, and `places_assign` is the port lemma for it: under `TraceOk`, each port's row bits sit at their unit input
columns (`rowHidden`) and each salt at its port's hm96 slot (`hmSalt`), whatever the mask. `placesR_assign` adds what
the last message's readers take besides (`PlacesR`): the `rec-open` copies' columns hold the stager's row bits
(`copies`, each below the unit's constant, so `OpenRows.colForm`'s hidden value), and each placed row is 16 hidden
values a word (`width`), so its opening is the bytes `VStarBuild.rowMsg` lays after its prefix (`PlacesR.rowMsg_eq`).

`TraceOk` is what the proof reads of the circuit and the inputs: the assigner mirrors the circuit (`supported`), the
circuits' ranges are disjoint and each slot is its circuit's size, the unit's first input group is read as 16-bit leaves
into self rows, no leaf cut feeds unit 0, the hm96 salt's columns are self rows, each port has one ref per instance, and
each row has its port's words and a 192-byte salt. The parse checks most of these (`Net.ofRows_inputs`, `checkLayout`);
`TraceOk` names them for the builder's proof to discharge or carry.
-/

namespace RecDeploy.VStarTrace

open _root_.Flock (Circuit Net Range WORD portCols)
open Claims.Recursion.VStarB (formBit)
open RecDeploy.VStarBuild (Places portRow saltBits)
open _root_.VStar.Trace (assign bitAt slots evalNet unitIns rowIns hmSlot instRow portIn wordBytes saltOf putBytes
  putWords orU16 u16At rowAt supported nbl)

/-! ## Folds of writes, read at one column -/

theorem ite_t {α : Sort _} {p : Prop} [Decidable p] {a b : α} (h : p) : (if p then a else b) = a :=
  ite_cond_eq_true a b (eq_true h)

theorem ite_f {α : Sort _} {p : Prop} [Decidable p] {a b : α} (h : ¬ p) : (if p then a else b) = b :=
  ite_cond_eq_false a b (eq_false h)

theorem getD_set!_ne {acc : Array Bool} {i xc : Nat} {v d : Bool} (h : i ≠ xc) :
    (acc.set! i v).getD xc d = acc.getD xc d := by
  simp only [Array.getD_eq_getD_getElem?, Array.set!_eq_setIfInBounds, Array.getElem?_setIfInBounds_ne h]

theorem getD_set! {acc : Array Bool} {i xc : Nat} {v : Bool} (hX : xc < acc.size) :
    (acc.set! i v).getD xc false = if xc = i then v else acc.getD xc false := by
  by_cases h : i = xc
  · subst h
    simp [Array.getD_eq_getD_getElem?, Array.set!_eq_setIfInBounds, hX]
  · rw [getD_set!_ne h, ite_eq_right_iff.mpr fun e => absurd e.symm h]

theorem size_set! {acc : Array Bool} {i : Nat} {v : Bool} : (acc.set! i v).size = acc.size := by
  simp [Array.set!_eq_setIfInBounds, Array.size_setIfInBounds]

/-- A fold over `range m` whose step `t` writes at most column `col + t`, read at one column. -/
theorem foldl_range_spec {f : Array Bool → Nat → Array Bool} {col : Nat} {cond : Nat → Bool} {val : Nat → Bool}
    (hf : ∀ acc t, (f acc t).size = acc.size ∧ ∀ xc < acc.size,
      (f acc t).getD xc false = if xc = col + t ∧ cond t = true then val t else acc.getD xc false) :
    ∀ m (w : Array Bool), ((List.range m).foldl f w).size = w.size ∧ ∀ xc < w.size,
      ((List.range m).foldl f w).getD xc false =
        if col ≤ xc ∧ xc < col + m ∧ cond (xc - col) = true then val (xc - col) else w.getD xc false := by
  intro m w
  induction m with
  | zero => exact ⟨rfl, fun xc _ => by simp only [List.range_zero, List.foldl_nil]; split <;> first | omega | rfl⟩
  | succ m ih =>
    rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]
    obtain ⟨hs, hg⟩ := ih
    obtain ⟨hs', hg'⟩ := hf ((List.range m).foldl f w) m
    refine ⟨hs'.trans hs, fun xc hX => ?_⟩
    rw [hg' xc (hs ▸ hX), hg xc hX]
    by_cases hx : xc = col + m
    · subst hx
      simp only [Nat.add_sub_cancel_left]
      by_cases hc : cond m = true <;> simp [hc]
    · have : (xc < col + (m + 1)) ↔ xc < col + m := by omega
      simp only [hx, false_and, ite_false, this]

theorem set_step_spec {cond : Nat → Bool} {val : Nat → Bool} {col : Nat} (acc : Array Bool) (t : Nat) :
    (if cond t = true then acc.set! (col + t) (val t) else acc).size = acc.size ∧ ∀ xc < acc.size,
      (if cond t = true then acc.set! (col + t) (val t) else acc).getD xc false =
        if xc = col + t ∧ cond t = true then val t else acc.getD xc false := by
  by_cases hc : cond t = true
  · simp only [hc, ite_true, and_true]
    exact ⟨size_set!, fun xc hX => getD_set! hX⟩
  · simp [hc]

/-- A fold that writes only at columns other than `t` leaves column `t`. -/
theorem foldl_keeps {α : Type} {F : Array Bool → α → Array Bool} {t : Nat} {p : α → Prop}
    (hF : ∀ z i, p i → (F z i).getD t false = z.getD t false) :
    ∀ (l : List α) (z : Array Bool), (∀ i ∈ l, p i) → (l.foldl F z).getD t false = z.getD t false
  | [], _, _ => rfl
  | i :: l, z, h => by
    rw [List.foldl_cons, foldl_keeps hF l _ fun j hj => h j (List.mem_cons_of_mem _ hj),
      hF z i (h i List.mem_cons_self)]

/-! ## Lists -/

theorem findSome?_one {α β : Type} {f : α → Option β} {a : α} {v : β} :
    ∀ {l : List α}, a ∈ l → f a = some v → (∀ x ∈ l, x ≠ a → f x = none) → l.findSome? f = some v
  | [], h, _, _ => by cases h
  | x :: l, h, ha, hn => by
    rw [List.findSome?_cons]
    by_cases hx : x = a
    · subst hx; rw [ha]
    · rw [hn x (List.mem_cons_self) hx]
      exact findSome?_one ((List.mem_cons.mp h).resolve_left (Ne.symm hx)) ha fun y hy => hn y (List.mem_cons_of_mem _ hy)

/-- Element `k a + e` of a list of `k`-element blocks. -/
theorem getElem?_flatMap_range {α β : Type} (g : α → Nat → β) (k e : Nat) (he : e < k) :
    ∀ (l : List α) (a : Nat), (l.flatMap fun u => (List.range k).map (g u))[k * a + e]? = l[a]?.map (g · e)
  | [], _ => by simp
  | u :: l, 0 => by
    simp [List.flatMap_cons, List.getElem?_append_left (by simp; omega : e < ((List.range k).map (g u)).length),
      List.getElem?_range he]
  | u :: l, a + 1 => by
    rw [List.flatMap_cons, List.getElem?_append_right (by simp; rw [Nat.mul_add]; omega)]
    have : k * (a + 1) + e - ((List.range k).map (g u)).length = k * a + e := by
      simp [Nat.mul_add]; omega
    rw [this, getElem?_flatMap_range g k e he l a]
    simp

theorem length_flatMap_range {α β : Type} (g : α → Nat → β) (k : Nat) :
    ∀ (l : List α), (l.flatMap fun u => (List.range k).map (g u)).length = k * l.length
  | [] => by simp
  | u :: l => by simp [List.flatMap_cons, length_flatMap_range g k l, Nat.mul_add]; omega

theorem mapM_some {α β : Type} {f : α → Option β} :
    ∀ {l : List α} {ps : List β}, l.mapM f = some ps → ps.length = l.length ∧ ∀ i (h : i < l.length), f l[i] = ps[i]?
  | [], ps, h => by
    simp at h; subst h; exact ⟨rfl, fun i h => by simp at h⟩
  | x :: l, ps, h => by
    rw [List.mapM_cons] at h
    cases hx : f x with
    | none => simp [hx] at h
    | some b =>
      cases hl : l.mapM f with
      | none => simp [hx, hl] at h
      | some bs =>
        simp [hx, hl] at h
        subst h
        obtain ⟨h1, h2⟩ := mapM_some hl
        refine ⟨by simp [h1], fun i hi => ?_⟩
        cases i with
        | zero => simp [hx]
        | succ i => simpa using h2 i (by simpa using hi)

/-! ## Arithmetic -/

theorem lt_mul_of {k c t S : Nat} (h : k < c) (ht : t < S) : k * S + t < c * S := by
  have := Nat.mul_le_mul_right S (Nat.succ_le_of_lt h)
  rw [Nat.succ_mul] at this
  omega

theorem le_two_pow_log2ceil (x : Nat) : x ≤ 2 ^ _root_.Flock.log2ceil x := by
  unfold _root_.Flock.log2ceil
  split
  · simp; omega
  · have := @Nat.lt_log2_self (x - 1); omega

/-- An instance below `count` is in a block below `2^nbl`. -/
theorem blk_lt {c : Circuit} {pin : _root_.VStar.In.PartIn} {n : Nat} (hg : 0 < c.g)
    (hn : n < _root_.VStar.Trace.count pin) : n / c.g < 2 ^ nbl c pin := by
  have h1 : n / c.g + 1 ≤ (_root_.VStar.Trace.count pin + c.g - 1) / c.g := by
    rw [← Nat.add_div_right n hg]
    exact Nat.div_le_div_right (by omega)
  have h2 := le_two_pow_log2ceil (max ((_root_.VStar.Trace.count pin + c.g - 1) / c.g) 1)
  have h3 : 2 ^ _root_.Flock.log2ceil (max ((_root_.VStar.Trace.count pin + c.g - 1) / c.g) 1) ≤ 2 ^ nbl c pin :=
    Nat.pow_le_pow_right (by decide) (Nat.le_max_right _ _)
  have h4 := Nat.le_max_left ((_root_.VStar.Trace.count pin + c.g - 1) / c.g) 1
  omega

theorem slot_eq (r : Range) (k : Nat) : r.slot k = r.start + k * 2 ^ r.slotLog := by
  simp only [Range.slot, Range.start, Nat.add_mul]

/-! ## The trace's pieces -/

theorem getD_map_range {β : Type} {N x : Nat} {F : Nat → β} {d : β} (hx : x < N) :
    ((Array.range N).map F).getD x d = F x := by
  simp [Array.getD_eq_getD_getElem?, Array.getElem?_map, Array.getElem?_range, hx]

/-- Column `col` is in range `r`'s span. -/
def InR (r : Range) (col : Nat) : Prop := r.start ≤ col ∧ col < r.start + r.count * 2 ^ r.slotLog

theorem bitAt_eq {c : Circuit} {z : Array (Array (Array Bool))} {mask : Nat → Nat → Bool} {n0 blk k t : Nat}
    (hn0 : n0 < c.nets.size) (hk : k < (c.rangeOf n0).count) (ht : t < 2 ^ (c.rangeOf n0).slotLog)
    (htu : t < 2 ^ (c.nets.getD n0 default).2.unitLog)
    (hz : blk * (c.rangeOf n0).count + k < (z.getD n0 #[]).size)
    (hb : (c.rangeOf n0).slot k + t < 2 ^ c.kLog)
    (hd : ∀ n < c.nets.size, n ≠ n0 → ¬ InR (c.rangeOf n) ((c.rangeOf n0).slot k + t)) :
    bitAt c z mask (blk * 2 ^ c.kLog + ((c.rangeOf n0).slot k + t)) =
      ((z.getD n0 #[]).getD (blk * (c.rangeOf n0).count + k) #[]).getD t false := by
  have hK : 0 < 2 ^ c.kLog := Nat.two_pow_pos _
  have hq1 : (blk * 2 ^ c.kLog + ((c.rangeOf n0).slot k + t)) / 2 ^ c.kLog = blk := by
    rw [Nat.mul_comm, Nat.mul_add_div hK, Nat.div_eq_of_lt hb, Nat.add_zero]
  have hq2 : (blk * 2 ^ c.kLog + ((c.rangeOf n0).slot k + t)) % 2 ^ c.kLog = (c.rangeOf n0).slot k + t := by
    rw [Nat.mul_comm, Nat.mul_add_mod, Nat.mod_eq_of_lt hb]
  have hS : 0 < 2 ^ (c.rangeOf n0).slotLog := Nat.two_pow_pos _
  have hkk : ((c.rangeOf n0).slot k + t - (c.rangeOf n0).start) / 2 ^ (c.rangeOf n0).slotLog = k := by
    rw [slot_eq, show (c.rangeOf n0).start + k * 2 ^ (c.rangeOf n0).slotLog + t - (c.rangeOf n0).start
      = 2 ^ (c.rangeOf n0).slotLog * k + t by rw [Nat.mul_comm]; omega, Nat.mul_add_div hS, Nat.div_eq_of_lt ht,
      Nat.add_zero]
  unfold bitAt
  simp only [hq1, hq2]
  rw [findSome?_one (a := n0) (v := ((z.getD n0 #[]).getD (blk * (c.rangeOf n0).count + k) #[]).getD t false)
    (List.mem_reverse.mpr (List.mem_range.mpr hn0)) ?h1 ?h2]
  case h1 =>
    simp only [hkk]
    have : (c.rangeOf n0).slot k + t - (c.rangeOf n0).slot k = t := by omega
    rw [this]
    have hst : (c.rangeOf n0).start ≤ (c.rangeOf n0).slot k + t := by rw [slot_eq]; omega
    simp only [Array.getD_eq_getD_getElem?] at htu hz ⊢
    simp [hst, hk, htu, hz]
  case h2 =>
    intro x hx hne
    have hx' := List.mem_range.mp (List.mem_reverse.mp hx)
    refine ite_eq_right_iff.mpr fun hc => ?_
    exfalso
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hc
    obtain ⟨⟨⟨h1, h2⟩, _⟩, _⟩ := hc
    apply hd x hx' hne
    refine ⟨h1, ?_⟩
    have := (Nat.div_lt_iff_lt_mul (Nat.two_pow_pos _)).mp h2
    omega

theorem evalNet_input {net : Net} {ins : Array Bool} {t : Nat} (ht : t < net.inWords * WORD)
    (hc : net.inWords * WORD ≤ net.constPos) (hl : net.constPos < 2 ^ net.unitLog) (hs : (net.a.row t).size = 1) :
    (evalNet net ins).getD t false = ins.getD t false := by
  unfold evalNet
  simp only []
  rw [foldl_keeps (p := fun i => t < i) ?_ _ _ (fun i hi => by have := List.mem_range'_1.mp hi; omega)]
  · rw [getD_map_range (by omega)]
    simp [show t ≠ net.constPos by omega, ht, hs]
  · intro z i hi
    split
    · rfl
    · exact getD_set!_ne (by omega)

theorem orU16_spec (w : Array Bool) (col v : Nat) :
    (orU16 w col v).size = w.size ∧ ∀ xc < w.size, (orU16 w col v).getD xc false =
      if col ≤ xc ∧ xc < col + 16 ∧ (decide (col % 64 + (xc - col) < 64) && v.testBit (xc - col)) = true then true
      else w.getD xc false :=
  foldl_range_spec (col := col) (cond := fun t => decide (col % 64 + t < 64) && v.testBit t) (val := fun _ => true)
    (f := fun acc t => if (decide (col % 64 + t < 64) && v.testBit t) = true then acc.set! (col + t) true else acc)
    (fun acc t => set_step_spec (cond := fun t => decide (col % 64 + t < 64) && v.testBit t)
      (val := fun _ => true) (col := col) acc t) 16 w

theorem putBytes_spec (w : Array Bool) (col : Nat) (b : ByteArray) :
    (putBytes w col b).size = w.size ∧ ∀ xc < w.size, (putBytes w col b).getD xc false =
      if col ≤ xc ∧ xc < col + 8 * b.size then (b.get! ((xc - col) / 8)).toNat.testBit ((xc - col) % 8)
      else w.getD xc false := by
  have := foldl_range_spec (col := col) (cond := fun _ => true)
    (val := fun t => (b.get! (t / 8)).toNat.testBit (t % 8))
    (f := fun acc t => acc.set! (col + t) ((b.get! (t / 8)).toNat.testBit (t % 8)))
    (fun acc t => ⟨size_set!, fun xc hX => by rw [getD_set! hX]; simp⟩) (8 * b.size) w
  exact ⟨this.1, fun xc hX => by unfold putBytes; rw [this.2 xc hX]; simp⟩

theorem putWords_size (w : Array Bool) (col : Nat) (ws : Array UInt64) : (putWords w col ws).size = w.size :=
  (foldl_range_spec (col := col) (cond := fun _ => true)
    (val := fun t => (ws.getD (t / 64) 0 >>> (t % 64).toUInt64) &&& 1 == 1)
    (f := fun acc t => acc.set! (col + t) ((ws.getD (t / 64) 0 >>> (t % 64).toUInt64) &&& 1 == 1))
    (fun acc t => ⟨size_set!, fun xc hX => by rw [getD_set! hX]; simp⟩) (64 * ws.size) w).1

theorem leaves_fold (rows : ByteArray) (li : Array Int) (bits : Array Nat) (w0 : Array Bool) (s0 : Nat)
    (hw0 : ∀ xc, w0.getD xc false = false) (h64 : s0 % 64 = 0) (hb : ∀ j < li.size, bits.getD j 0 = 16) :
    ∀ m ≤ li.size, ((List.range m).foldl (_root_.VStar.Trace.leafStep rows li bits) (w0, s0)).2 = s0 + 16 * m ∧
      ((List.range m).foldl (_root_.VStar.Trace.leafStep rows li bits) (w0, s0)).1.size = w0.size ∧
      ∀ kk < li.size, ∀ t < 16, s0 + 16 * kk + t < w0.size →
        ((List.range m).foldl (_root_.VStar.Trace.leafStep rows li bits) (w0, s0)).1.getD (s0 + 16 * kk + t) false =
          (decide (kk < m) && decide (0 ≤ li.getD kk (-1)) && (u16At rows (2 * (li.getD kk (-1)).toNat)).testBit t) := by
  intro m
  induction m with
  | zero => intro _; exact ⟨rfl, rfl, fun kk _ t _ _ => by simp [hw0]⟩
  | succ m ih =>
    intro hm
    obtain ⟨h2, hs, hg⟩ := ih (by omega)
    rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]
    generalize hr : (List.range m).foldl (_root_.VStar.Trace.leafStep rows li bits) (w0, s0) = r at h2 hs hg
    unfold _root_.VStar.Trace.leafStep
    simp only []
    refine ⟨by rw [h2, hb m (by omega)]; omega, ?_, fun kk hkk t ht hX => ?_⟩
    · by_cases hl : 0 ≤ li.getD m (-1)
      · rw [ite_t hl, (orU16_spec _ _ _).1, hs]
      · rw [ite_f hl, hs]
    · have hX' : s0 + 16 * kk + t < r.1.size := hs ▸ hX
      by_cases hl : 0 ≤ li.getD m (-1)
      · rw [ite_t hl, (orU16_spec _ _ _).2 _ hX', hg kk hkk t ht hX, h2]
        by_cases hk : kk = m
        · subst hk
          have e1 : s0 + 16 * kk + t - (s0 + 16 * kk) = t := by omega
          have e2 : (s0 + 16 * kk) % 64 + t < 64 := by omega
          rw [e1]
          simp only [Array.getD_eq_getD_getElem?] at hl
          simp [e2, hl, ht]
        · have : ¬ (s0 + 16 * m ≤ s0 + 16 * kk + t ∧ s0 + 16 * kk + t < s0 + 16 * m + 16) := by omega
          rw [ite_f (fun h => this ⟨h.1, h.2.1⟩)]
          by_cases hlt : kk < m
          · simp [hlt, show kk < m + 1 by omega]
          · simp [hlt, show ¬ kk < m + 1 by omega]
      · rw [ite_f hl, hg kk hkk t ht hX]
        by_cases hk : kk = m
        · subst hk; simp only [Array.getD_eq_getD_getElem?] at hl; simp [hl]
        · by_cases hlt : kk < m
          · simp [hlt, show kk < m + 1 by omega]
          · simp [hlt, show ¬ kk < m + 1 by omega]

/-- The ports' words before port `q`. -/
def off (c : Circuit) (q : Nat) : Nat := (c.ports.extract 0 q).foldl (· + ·.words) 0

theorem off_succ (c : Circuit) {q : Nat} (hq : q < c.ports.size) : off c (q + 1) = off c q + c.ports[q].words := by
  unfold off
  rw [← Array.foldl_toList, ← Array.foldl_toList, Array.toList_extract, Array.toList_extract]
  simp [List.extract, List.take_add_one, List.foldl_append, hq]

theorem off_zero (c : Circuit) : off c 0 = 0 := by simp [off]

theorem off_mono (c : Circuit) {q m : Nat} (hqm : q ≤ m) (hm : m ≤ c.ports.size) : off c q ≤ off c m := by
  induction m with
  | zero => exact Nat.le_of_eq (congrArg (off c) (Nat.le_zero.mp hqm))
  | succ m ih =>
    rcases Nat.lt_or_ge q (m + 1) with h | h
    · rw [off_succ c (by omega)]; have := ih (by omega) (by omega); omega
    · exact Nat.le_of_eq (congrArg (off c) (by omega))

theorem get!_eq (b : ByteArray) (i : Nat) : b.get! i = (b.data[i]?).getD default := by
  cases b with
  | mk bs =>
    show bs[i]! = _
    by_cases h : i < bs.size
    · simp [h]
    · simp [h]

theorem portIn_size {c : Circuit} {pin : _root_.VStar.In.PartIn} {p n : Nat}
    (hn : n < _root_.VStar.Trace.count pin) : (portIn c pin p n).1.size = 2 * (c.ports.getD p default).words := by
  unfold portIn
  simp only [ite_t hn]
  cases h : rowAt pin (c.ports.getD p default).name n with
  | none => simp [_root_.VStar.Trace.zeros, ByteArray.size]
  | some r => simp [wordBytes, ByteArray.size]

theorem portIn_some {c : Circuit} {pin : _root_.VStar.In.PartIn} {q n : Nat} {rw : _root_.VStar.In.Row}
    (hn : n < _root_.VStar.Trace.count pin) (h : rowAt pin (c.ports.getD q default).name n = some rw) :
    portIn c pin q n = (wordBytes rw.words (c.ports.getD q default).words, saltOf rw.salt) := by
  unfold portIn
  simp only [ite_t hn, h]

theorem instRow_fold {c : Circuit} {pin : _root_.VStar.In.PartIn} {n : Nat} (hn : n < _root_.VStar.Trace.count pin) :
    ∀ m ≤ c.ports.size,
      ((List.range m).foldl (fun acc p => acc ++ (portIn c pin p n).1.data) #[]).size = 2 * off c m ∧
      ∀ q < m, ∀ y < 2 * (c.ports.getD q default).words,
        ((List.range m).foldl (fun acc p => acc ++ (portIn c pin p n).1.data) #[])[2 * off c q + y]? =
          (portIn c pin q n).1.data[y]? := by
  intro m
  induction m with
  | zero => intro _; exact ⟨by simp [off_zero], fun q hq => absurd hq (Nat.not_lt_zero _)⟩
  | succ m ih =>
    intro hm
    obtain ⟨hs, hg⟩ := ih (by omega)
    rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]
    have hd : (portIn c pin m n).1.data.size = 2 * (c.ports.getD m default).words := portIn_size hn
    have hgm : (c.ports.getD m default) = c.ports[m] := by
      simp [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem (by omega : m < c.ports.size)]
    refine ⟨by rw [Array.size_append, hs, hd, off_succ c (by omega), hgm]; omega, fun q hq y hy => ?_⟩
    rcases Nat.lt_or_ge q m with hqm | hqm
    · have h1 := off_mono c (show q + 1 ≤ m by omega) (by omega)
      have hgq : (c.ports.getD q default) = c.ports[q] := by
        simp [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem (by omega : q < c.ports.size)]
      have h2 : off c (q + 1) = off c q + (c.ports.getD q default).words := by
        rw [hgq]; exact off_succ c (by omega : q < c.ports.size)
      rw [Array.getElem?_append_left (by rw [hs]; omega), hg q hqm y hy]
    · have hqe : q = m := by omega
      subst hqe
      rw [Array.getElem?_append_right (by rw [hs]; omega), hs]
      congr 1; omega

theorem instRow_get {c : Circuit} {pin : _root_.VStar.In.PartIn} {n q y : Nat} (hn : n < _root_.VStar.Trace.count pin)
    (hq : q < c.ports.size) (hy : y < 2 * (c.ports.getD q default).words) :
    (instRow c pin n).get! (2 * off c q + y) = (portIn c pin q n).1.get! y := by
  rw [get!_eq, get!_eq]
  unfold instRow
  rw [(instRow_fold hn c.ports.size (Nat.le_refl _)).2 q hq y hy]

theorem wordBytes_get {ws : Array Nat} {words j : Nat} (hj : j < 2 * words) :
    ((wordBytes ws words).get! j).toNat = ws.getD (j / 2) 0 / 256 ^ (j % 2) % 256 := by
  rw [get!_eq]
  simp [wordBytes, Array.getElem?_map, Array.getElem?_range, hj]

theorem u16At_row {c : Circuit} {pin : _root_.VStar.In.PartIn} {n q a : Nat} {rw : _root_.VStar.In.Row}
    (hn : n < _root_.VStar.Trace.count pin) (hq : q < c.ports.size) (ha : a < (c.ports.getD q default).words)
    (h : rowAt pin (c.ports.getD q default).name n = some rw) :
    u16At (instRow c pin n) (2 * (off c q + a)) = rw.words.getD a 0 % 65536 := by
  unfold u16At
  rw [show 2 * (off c q + a) = 2 * off c q + 2 * a by omega, show 2 * off c q + 2 * a + 1 = 2 * off c q + (2 * a + 1) by omega,
    instRow_get hn hq (by omega), instRow_get hn hq (by omega), portIn_some hn h]
  simp only []
  rw [wordBytes_get (by omega), wordBytes_get (by omega)]
  have e1 : 2 * a / 2 = a := by omega
  have e2 : (2 * a + 1) / 2 = a := by omega
  have e3 : 2 * a % 2 = 0 := by omega
  have e4 : (2 * a + 1) % 2 = 1 := by omega
  rw [e1, e2, e3, e4, Nat.pow_zero, Nat.pow_one, Nat.div_one]
  omega

theorem saltOf_get {s : ByteArray} {i : Nat} (hi : i < _root_.Flock.HmRow.SALT_BYTES) :
    (saltOf s).get! i = s.data.getD i 0 := by
  rw [get!_eq]
  simp [saltOf, Array.getElem?_map, Array.getElem?_range, hi]

theorem saltOf_size (s : ByteArray) : (saltOf s).size = _root_.Flock.HmRow.SALT_BYTES := by
  show ((Array.range _root_.Flock.HmRow.SALT_BYTES).map fun j => s.data.getD j 0).size = _
  rw [Array.size_map, Array.size_range]

theorem cuts_id {w : Array Bool} {u : Nat} {F : Array Bool → Nat × Nat × Nat → Array Bool} :
    ∀ (l : List (Nat × Nat × Nat)), (∀ x ∈ l, x.1 ≠ u) → l.foldl (fun acc x => if (x.1 == u) = true then F acc x else acc) w = w
  | [], _ => rfl
  | x :: l, h => by
    rw [List.foldl_cons, ite_f (by simpa using h x List.mem_cons_self)]
    exact cuts_id l fun y hy => h y (List.mem_cons_of_mem _ hy)

theorem replicate_getD (N xc : Nat) : (Array.replicate N false).getD xc false = false := by
  simp only [Array.getD_eq_getD_getElem?, Array.getElem?_replicate]
  split <;> rfl

/-! ## The assigner and what it rests on -/

/-- **The trace assigner**: `VStar.Trace.assign` under the mask `mask`, the empty witness where it refuses. -/
def A (mask : Nat → Nat → Bool) (c : Circuit) (pin : _root_.VStar.In.PartIn) : Claims.Flock.Wit :=
  ⟨(assign c pin mask).getD []⟩

/-- The `hm96` circuit. -/
def hmN (c : Circuit) : Net := (c.nets.getD c.hmNet default).2

/-- **What the port lemma rests on**: a circuit the assigner mirrors (`supported`), its ranges disjoint and each slot the
size of its circuit, the unit's first input group read as 16-bit leaves from aligned columns into self rows, no leaf cut
into unit 0, the hm96 salt's columns self rows, every port's row whole 16-bit words (`fullRows`, which the difftest
checks at each staged part); and port inputs with one ref per instance in each port, rows of the port's words and 192-byte
salts. -/
structure TraceOk (c : Circuit) (pin : _root_.VStar.In.PartIn) : Prop where
  supported : supported c = true
  disjoint : ∀ n₁ n₂ col, n₁ < c.nets.size → n₂ < c.nets.size → n₁ ≠ n₂ → InR (c.rangeOf n₁) col →
    InR (c.rangeOf n₂) col → False
  unit_lt : c.unitNet < c.nets.size
  hm_lt : c.hmNet < c.nets.size
  unit_sha : c.unitNet ≠ c.shaNet
  unit_hm : c.unitNet ≠ c.hmNet
  hm_sha : c.hmNet ≠ c.shaNet
  g_pos : 0 < c.g
  upv_pos : 0 < c.upv
  unit_count : c.g * c.upv ≤ c.unitRange.count
  hm_count : c.g * c.perVu.getD c.hmNet 0 ≤ (c.rangeOf c.hmNet).count
  ports_hm : c.ports.size ≤ c.perVu.getD c.hmNet 0
  unit_slot : c.unitRange.slotLog = c.unit.unitLog
  hm_slot : (c.rangeOf c.hmNet).slotLog = (hmN c).unitLog
  unit_block : c.unitRange.start + c.unitRange.count * 2 ^ c.unitRange.slotLog ≤ 2 ^ c.kLog
  hm_block : (c.rangeOf c.hmNet).start + (c.rangeOf c.hmNet).count * 2 ^ (c.rangeOf c.hmNet).slotLog ≤ 2 ^ c.kLog
  unit_const : c.unit.inWords * WORD ≤ c.unit.constPos
  unit_log : c.unit.constPos < 2 ^ c.unit.unitLog
  hm_const : (hmN c).inWords * WORD ≤ (hmN c).constPos
  hm_log : (hmN c).constPos < 2 ^ (hmN c).unitLog
  leaf_bits : ∀ j < (c.leavesIn.getD 0 #[]).size, (c.unit.inGroups.getD 0 default).bits.getD j 0 = 16
  leaf_fit : (c.unit.inGroups.getD 0 default).col * WORD + 16 * (c.leavesIn.getD 0 #[]).size ≤ c.unit.inWords * WORD
  leaf_self : ∀ k < (c.leavesIn.getD 0 #[]).size, ∀ t < 16,
    (c.unit.a.row ((c.unit.inGroups.getD 0 default).col * WORD + 16 * k + t)).size = 1
  cuts : ∀ x ∈ c.leafCuts.toList, x.1 ≠ 0
  salt_fit : hmIn c 2 + 8 * _root_.Flock.HmRow.SALT_BYTES ≤ (hmN c).inWords * WORD
  salt_self : ∀ j < 8 * _root_.Flock.HmRow.SALT_BYTES, ((hmN c).a.row (hmIn c 2 + j)).size = 1
  refs : ∀ x ∈ pin.toList, x.2.refs.size = _root_.VStar.Trace.count pin
  rows : ∀ q (hq : q < c.ports.size) n r, portRow pin c.ports[q].name n = some r →
    r.words.size = c.ports[q].words ∧ r.salt.size = _root_.Flock.HmRow.SALT_BYTES
  full : _root_.VStar.Trace.fullRows c = true

variable {c : Circuit} {pin : _root_.VStar.In.PartIn}

theorem port_bits (h : TraceOk c pin) {q : Nat} (hq : q < c.ports.size) : c.ports[q].bits = 16 * c.ports[q].words := by
  have hf := h.full
  unfold _root_.VStar.Trace.fullRows at hf
  rw [Array.all_eq_true] at hf
  exact beq_iff_eq.mp (hf q hq)

theorem hid_getD (h : TraceOk c pin) (mask : Nat → Nat → Bool) {P : Nat} (hP : P < 2 ^ _root_.VStar.Trace.m c pin) :
    (A mask c pin).hid.getD P false = bitAt c (slots c pin) mask P := by
  unfold A assign
  rw [ite_t h.supported]
  simp only [Option.getD_some, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range hP, Option.map_some]

theorem lt_two_pow_m (h : TraceOk c pin) {n y : Nat} (hn : n < _root_.VStar.Trace.count pin) (hy : y < 2 ^ c.kLog) :
    n / c.g * 2 ^ c.kLog + y < 2 ^ _root_.VStar.Trace.m c pin := by
  have := lt_mul_of (blk_lt h.g_pos hn) hy
  unfold _root_.VStar.Trace.m
  rw [Nat.pow_add, Nat.mul_comm (2 ^ c.kLog)]
  exact this

theorem count_of_portRow (h : TraceOk c pin) {nm : String} {n : Nat} {rw : _root_.VStar.In.Row}
    (hr : portRow pin nm n = some rw) : n < _root_.VStar.Trace.count pin := by
  unfold portRow at hr
  cases hf : pin.find? (·.1 == nm) with
  | none => simp [hf] at hr
  | some x =>
    simp only [hf, Option.map_some, Option.bind_eq_bind, Option.bind_some] at hr
    cases hi : x.2.refs[n]? with
    | none => simp [hi] at hr
    | some i =>
      have hlt : n < x.2.refs.size := by
        rcases Nat.lt_or_ge n x.2.refs.size with h' | h'
        · exact h'
        · rw [Array.getElem?_eq_none h'] at hi; cases hi
      rw [← h.refs x (Array.mem_toList_iff.mpr (Array.mem_of_find?_eq_some hf))]
      exact hlt

theorem name_of_findIdx {nm : String} {q : Nat} (hq : c.ports.findIdx? (·.name == nm) = some q) :
    ∃ hq : q < c.ports.size, c.ports[q].name = nm ∧ c.ports.getD q default = c.ports[q] := by
  obtain ⟨hqs, hqn, _⟩ := Array.findIdx?_eq_some_iff_getElem.mp hq
  exact ⟨hqs, beq_iff_eq.mp hqn, by simp [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hqs]⟩

theorem hmIn_eq : hmIn c 2 = ((portCols (c.nets.getD c.hmNet default).2.inGroups).getD 2 default).1 := by
  unfold hmIn
  generalize portCols (c.nets.getD c.hmNet default).2.inGroups = a
  rw [Array.getD_eq_getD_getElem?]
  cases a[2]? <;> rfl

theorem unitIns_get (h : TraceOk c pin) {n kk t : Nat} (hn : n < _root_.VStar.Trace.count pin)
    (hkk : kk < (c.leavesIn.getD 0 #[]).size) (ht : t < 16) :
    (unitIns c pin (n / c.g) (n % c.g * c.upv)).getD ((c.unit.inGroups.getD 0 default).col * WORD + 16 * kk + t) false =
      (decide (0 ≤ (c.leavesIn.getD 0 #[]).getD kk (-1)) &&
        (u16At (instRow c pin n) (2 * ((c.leavesIn.getD 0 #[]).getD kk (-1)).toNat)).testBit t) := by
  have hu1 : n % c.g * c.upv / c.upv = n % c.g := Nat.mul_div_cancel _ h.upv_pos
  have hu2 : n % c.g * c.upv % c.upv = 0 := Nat.mul_mod_left _ _
  have hi : n / c.g * c.g + n % c.g = n := Nat.div_add_mod' n c.g
  unfold unitIns
  simp only [hu1, hu2, hi, hn, ↓reduceIte]
  rw [← Array.foldl_toList, cuts_id _ h.cuts]
  have hW : WORD = 128 := rfl
  have hl := (leaves_fold (instRow c pin n) (c.leavesIn.getD 0 #[]) (c.unit.inGroups.getD 0 default).bits
    (Array.replicate (c.unit.inWords * WORD) false) ((c.unit.inGroups.getD 0 default).col * WORD)
    (replicate_getD _) (by rw [hW]; omega) h.leaf_bits _ (Nat.le_refl _)).2.2 kk hkk t ht
    (by have := h.leaf_fit; simp only [Array.size_replicate]; omega)
  exact hl.trans (by rw [decide_eq_true hkk, Bool.true_and])

theorem slots_unit (h : TraceOk c pin) :
    (slots c pin).getD c.unitNet #[] = (Array.range (2 ^ nbl c pin * (c.rangeOf c.unitNet).count)).map fun x =>
      evalNet c.unit (unitIns c pin (x / (c.rangeOf c.unitNet).count) (x % (c.rangeOf c.unitNet).count)) := by
  unfold slots
  rw [getD_map_range h.unit_lt]
  simp only [beq_iff_eq, h.unit_sha, h.unit_hm, ↓reduceIte]
  rfl

theorem slots_hm (h : TraceOk c pin) :
    (slots c pin).getD c.hmNet #[] = (Array.range (2 ^ nbl c pin * (c.rangeOf c.hmNet).count)).map fun x =>
      evalNet (hmN c) ((((Array.range (2 ^ nbl c pin)).map (rowIns c pin)).getD (x / (c.rangeOf c.hmNet).count)
        default).2.getD (x % (c.rangeOf c.hmNet).count) #[]) := by
  unfold slots
  rw [getD_map_range h.hm_lt]
  simp only [beq_iff_eq, h.hm_sha, ↓reduceIte]
  rfl

theorem blk_div {blk cnt k : Nat} (hk : k < cnt) : (blk * cnt + k) / cnt = blk ∧ (blk * cnt + k) % cnt = k := by
  have hc : 0 < cnt := by omega
  rw [Nat.mul_comm, Nat.mul_add_div hc, Nat.div_eq_of_lt hk, Nat.mul_add_mod, Nat.mod_eq_of_lt hk]
  exact ⟨rfl, rfl⟩

/-- **The unit's input column of port `q`'s row bit `i` at instance `n`** holds the bit. -/
theorem row_bit (h : TraceOk c pin) (mask : Nat → Nat → Bool) {q n i col : Nat} {rw : _root_.VStar.In.Row}
    (hq : q < c.ports.size) (hn : n < _root_.VStar.Trace.count pin)
    (hr : rowAt pin (c.ports.getD q default).name n = some rw) (hw : rw.words.size = (c.ports.getD q default).words)
    (hi : i < 16 * (c.ports.getD q default).words) (hc : RecDeploy.portCol c 0 q i = some col) :
    (A mask c pin).hid.getD (n / c.g * 2 ^ c.kLog + (c.unitRange.slot (n % c.g * c.upv) + col)) false =
      (rw.words.getD (i / 16) 0).testBit (i % 16) := by
  unfold RecDeploy.portCol at hc
  simp only [] at hc
  cases hk : (c.leavesIn.getD 0 #[]).findIdx? (· == ((off c q + i / 16 : Nat) : Int)) with
  | none => simp only [off] at hk; rw [hk] at hc; simp at hc
  | some kk =>
    simp only [off] at hk
    rw [hk] at hc
    simp only [Option.bind_eq_bind, Option.bind_some, Option.pure_def, Option.some.injEq] at hc
    obtain ⟨hkk, hkv, _⟩ := Array.findIdx?_eq_some_iff_getElem.mp hk
    have hlk : (c.leavesIn.getD 0 #[]).getD kk (-1) = ((off c q + i / 16 : Nat) : Int) := by
      rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hkk]
      exact beq_iff_eq.mp hkv
    subst hc
    have hcount := h.unit_count
    have hslot := h.unit_slot
    have hblock := h.unit_block
    simp only [Circuit.unitRange] at hcount hslot hblock ⊢
    have hcol : (c.unit.inGroups.getD 0 default).col * WORD + 16 * kk + i % 16 < c.unit.inWords * WORD := by
      have := h.leaf_fit; have : i % 16 < 16 := Nat.mod_lt _ (by decide); omega
    have hcu : (c.unit.inGroups.getD 0 default).col * WORD + 16 * kk + i % 16 < 2 ^ c.unit.unitLog := by
      have := h.unit_const; have := h.unit_log; omega
    have hk' : n % c.g * c.upv < (c.rangeOf c.unitNet).count := by
      have h1 := Nat.mod_lt n h.g_pos
      have h2 := Nat.mul_le_mul_right c.upv (Nat.succ_le_of_lt h1)
      rw [Nat.succ_mul] at h2
      have := h.upv_pos; omega
    have ht' : (c.unit.inGroups.getD 0 default).col * WORD + 16 * kk + i % 16 < 2 ^ (c.rangeOf c.unitNet).slotLog := by
      rw [hslot]; exact hcu
    have hblk := blk_lt h.g_pos hn
    have hspan := lt_mul_of hk' ht'
    have hb : (c.rangeOf c.unitNet).slot (n % c.g * c.upv) + ((c.unit.inGroups.getD 0 default).col * WORD + 16 * kk +
        i % 16) < 2 ^ c.kLog := by
      rw [slot_eq]; omega
    rw [hid_getD h mask (lt_two_pow_m h hn hb)]
    rw [bitAt_eq h.unit_lt hk' ht' hcu ?hz hb ?hd]
    case hz =>
      rw [slots_unit h, Array.size_map, Array.size_range]
      exact lt_mul_of hblk hk'
    case hd =>
      intro x hx hne hin
      refine h.disjoint x c.unitNet _ hx h.unit_lt hne hin ⟨?_, ?_⟩
      · rw [slot_eq]; omega
      · rw [slot_eq]; omega
    rw [slots_unit h, getD_map_range (lt_mul_of hblk hk')]
    rw [(blk_div hk').1, (blk_div hk').2]
    rw [evalNet_input hcol h.unit_const h.unit_log (h.leaf_self kk hkk _ (Nat.mod_lt _ (by decide)))]
    rw [unitIns_get h hn hkk (Nat.mod_lt _ (by decide)), hlk]
    have ha : i / 16 < (c.ports.getD q default).words := by omega
    rw [decide_eq_true (by omega : (0 : Int) ≤ ((off c q + i / 16 : Nat) : Int)), Bool.true_and, Int.toNat_natCast,
      u16At_row hn hq ha hr, show (65536 : Nat) = 2 ^ 16 from rfl, Nat.testBit_mod_two_pow,
      decide_eq_true (Nat.mod_lt i (by decide : 0 < 16)), Bool.true_and]

theorem wordBits_get {ws : Array Nat} {i : Nat} (hi : i < 16 * ws.size) :
    (_root_.VStar.In.wordBits ws)[i]? = some ((ws.getD (i / 16) 0).testBit (i % 16)) := by
  have := getElem?_flatMap_range (fun (w : Nat) => w.testBit) 16 (i % 16) (Nat.mod_lt _ (by decide)) ws.toList (i / 16)
  rw [Nat.div_add_mod] at this
  refine this.trans ?_
  rw [Array.getElem?_toList, Array.getElem?_eq_getElem (by omega : i / 16 < ws.size), Option.map_some,
    Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem (by omega : i / 16 < ws.size), Option.getD_some]

theorem saltBits_get {b : ByteArray} {j : Nat} (hj : j < 8 * b.size) :
    (saltBits b)[j]? = some ((b.data.getD (j / 8) 0).toNat.testBit (j % 8)) := by
  have := getElem?_flatMap_range (fun (u : UInt8) => u.toNat.testBit) 8 (j % 8) (Nat.mod_lt _ (by decide))
    b.data.toList (j / 8)
  rw [Nat.div_add_mod] at this
  have hs : j / 8 < b.data.size := by have : b.size = b.data.size := rfl; omega
  refine this.trans ?_
  rw [Array.getElem?_toList, Array.getElem?_eq_getElem hs, Option.map_some,
    Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hs, Option.getD_some]

theorem saltBits_length (b : ByteArray) : (saltBits b).length = 8 * b.size := by
  unfold saltBits
  rw [length_flatMap_range (fun (u : UInt8) => u.toNat.testBit) 8, Array.length_toList]
  rfl

theorem formBit_var (w : Claims.Flock.Wit) (x : Nat) : formBit w (_root_.Flock.HmNets.Lin.var x) = w.hid.getD x false := by
  simp [formBit, _root_.Flock.HmNets.Lin.var]

/-- **The port lemma, rows**: each row bit at its unit input column. -/
theorem places_row (h : TraceOk c pin) (mask : Nat → Nat → Bool) (nm : String) (n : Nat) (rw : _root_.VStar.In.Row)
    (ps : List Nat) (hr : portRow pin nm n = some rw) (hps : RecDeploy.rowHidden c nm n = some ps) :
    ps.map ((A mask c pin).hid.getD · false) = (_root_.VStar.In.wordBits rw.words).take ps.length := by
  unfold RecDeploy.rowHidden at hps
  cases hq : c.ports.findIdx? (·.name == nm) with
  | none => rw [hq] at hps; simp at hps
  | some q =>
    obtain ⟨hqs, hname, hget⟩ := name_of_findIdx hq
    rw [hq] at hps
    simp only [Option.bind_eq_bind, Option.bind_some, Array.getElem?_eq_getElem hqs] at hps
    obtain ⟨hlen, hel⟩ := mapM_some hps
    rw [List.length_range] at hlen
    have hn := count_of_portRow h hr
    have hr' : rowAt pin (c.ports.getD q default).name n = some rw := by rw [hget, hname]; exact hr
    obtain ⟨hws, _⟩ := h.rows q hqs n rw (hname ▸ hr)
    have hbits := Nat.le_of_eq (port_bits h hqs)
    apply List.ext_getElem?
    intro i
    rw [List.getElem?_map, List.getElem?_take]
    by_cases hi : i < ps.length
    · rw [ite_t hi]
      have hi' : i < c.ports[q].bits := by omega
      have he := hel i (by rw [List.length_range]; exact hi')
      rw [List.getElem_range] at he
      cases hpc : RecDeploy.portCol c 0 q i with
      | none =>
        rw [hpc] at he
        exfalso
        rw [List.getElem?_eq_getElem hi] at he
        cases he
      | some col =>
        rw [hpc, Option.map_some] at he
        rw [← he, Option.map_some, row_bit h mask hqs hn hr' (by rw [hget]; exact hws) (by rw [hget]; omega) hpc,
          wordBits_get (by omega)]
    · rw [ite_f hi, List.getElem?_eq_none (by omega)]
      rfl

theorem salt_bit (h : TraceOk c pin) (mask : Nat → Nat → Bool) {q n j : Nat} {rw : _root_.VStar.In.Row}
    (hq : q < c.ports.size) (hn : n < _root_.VStar.Trace.count pin)
    (hr : rowAt pin (c.ports.getD q default).name n = some rw) (hj : j < 8 * _root_.Flock.HmRow.SALT_BYTES) :
    (A mask c pin).hid.getD (slotHid c n (hmBase c q n) (hmIn c 2 + j)) false =
      (rw.salt.data.getD (j / 8) 0).toNat.testBit (j % 8) := by
  unfold slotHid hmBase Circuit.slotCol
  have hP := h.ports_hm
  have hcount := h.hm_count
  have hslot := h.hm_slot
  have hblock := h.hm_block
  have hfit := h.salt_fit
  have hconst := h.hm_const
  have hlog := h.hm_log
  unfold hmN at hslot hfit hconst hlog
  have hk' : n % c.g * c.perVu.getD c.hmNet 0 + q < (c.rangeOf c.hmNet).count := by
    have h1 := Nat.mod_lt n h.g_pos
    have h2 := Nat.mul_le_mul_right (c.perVu.getD c.hmNet 0) (Nat.succ_le_of_lt h1)
    rw [Nat.succ_mul] at h2
    omega
  have hcu : hmIn c 2 + j < 2 ^ (c.nets.getD c.hmNet default).2.unitLog := by omega
  have ht' : hmIn c 2 + j < 2 ^ (c.rangeOf c.hmNet).slotLog := by rw [hslot]; exact hcu
  have hblk := blk_lt h.g_pos hn
  have hspan := lt_mul_of hk' ht'
  have hb : (c.rangeOf c.hmNet).slot (n % c.g * c.perVu.getD c.hmNet 0 + q) + (hmIn c 2 + j) < 2 ^ c.kLog := by
    rw [slot_eq]; omega
  rw [hid_getD h mask (lt_two_pow_m h hn hb)]
  rw [bitAt_eq h.hm_lt hk' ht' hcu ?hz hb ?hd]
  case hz =>
    rw [slots_hm h, Array.size_map, Array.size_range]
    exact lt_mul_of hblk hk'
  case hd =>
    intro x hx hne hin
    refine h.disjoint x c.hmNet _ hx h.hm_lt hne hin ⟨?_, ?_⟩
    · rw [slot_eq]; omega
    · rw [slot_eq]; omega
  rw [slots_hm h, getD_map_range (lt_mul_of hblk hk'), (blk_div hk').1, (blk_div hk').2,
    getD_map_range hblk]
  rw [evalNet_input (by unfold hmN; omega) (by unfold hmN; omega) (by unfold hmN; omega) (h.salt_self j hj)]
  have hPpos : 0 < c.perVu.getD c.hmNet 0 := by omega
  have hqP : q < c.perVu.getD c.hmNet 0 := by omega
  have hd1 : (n % c.g * c.perVu.getD c.hmNet 0 + q) / c.perVu.getD c.hmNet 0 = n % c.g := by
    rw [Nat.mul_comm, Nat.mul_add_div hPpos, Nat.div_eq_of_lt hqP, Nat.add_zero]
  have hd2 : (n % c.g * c.perVu.getD c.hmNet 0 + q) % c.perVu.getD c.hmNet 0 = q := by
    rw [Nat.mul_comm, Nat.mul_add_mod, Nat.mod_eq_of_lt hqP]
  have hi : n / c.g * c.g + n % c.g = n := Nat.div_add_mod' n c.g
  have hkg : n % c.g * c.perVu.getD c.hmNet 0 + q < c.g * c.perVu.getD c.hmNet 0 := by
    have h1 := Nat.mod_lt n h.g_pos
    have h2 := Nat.mul_le_mul_right (c.perVu.getD c.hmNet 0) (Nat.succ_le_of_lt h1)
    rw [Nat.succ_mul] at h2
    omega
  unfold rowIns
  simp only []
  rw [getD_map_range hkg, hd1, hd2, hi]
  unfold hmSlot
  simp only [hq, ↓reduceIte]
  rw [hmIn_eq] at hfit ⊢
  rw [(putBytes_spec _ _ _).2 _ ?hX, portIn_some hn hr]
  case hX => rw [putWords_size, putWords_size, Array.size_replicate]; omega
  simp only []
  rw [saltOf_size, ite_t ⟨by omega, by omega⟩, Nat.add_sub_cancel_left, saltOf_get (by omega)]

/-- **The port lemma, salts**: each salt bit at its port's hm96 slot. -/
theorem places_salt (h : TraceOk c pin) (mask : Nat → Nat → Bool) (nm : String) (q n : Nat) (rw : _root_.VStar.In.Row)
    (hq : c.ports.findIdx? (·.name == nm) = some q) (hr : portRow pin nm n = some rw) :
    (hmSalt c q n).map (formBit (A mask c pin)) = saltBits rw.salt := by
  obtain ⟨hqs, hname, hget⟩ := name_of_findIdx hq
  have hn := count_of_portRow h hr
  have hr' : rowAt pin (c.ports.getD q default).name n = some rw := by rw [hget, hname]; exact hr
  obtain ⟨_, hss⟩ := h.rows q hqs n rw (hname ▸ hr)
  apply List.ext_getElem?
  intro j
  unfold hmSalt
  rw [List.getElem?_map, List.getElem?_map]
  by_cases hj : j < 1536
  · rw [List.getElem?_range hj, Option.map_some, Option.map_some, saltBits_get (by rw [hss]; exact hj), formBit_var,
      salt_bit h mask hqs hn hr' hj]
  · rw [List.getElem?_eq_none (by simp; omega), List.getElem?_eq_none (by rw [saltBits_length, hss]; exact Nat.le_of_not_lt hj)]
    rfl

/-- **The port lemma** (`Places`) for the trace assigner `A`, on a circuit and port inputs `TraceOk` holds of. -/
theorem places_assign (h : TraceOk c pin) (mask : Nat → Nat → Bool) : Places c pin (A mask c pin) :=
  ⟨places_row h mask, places_salt h mask⟩

/-! ## The copies and the row width -/

/-- **The copies**: instance `n`'s copy of the unit (block `n / g`, VU `n % g`, its first unit) holds, at the input
column `portCol` gives bit `i` of port `q`'s row, that bit of the stager's row there; the column is below the unit's
constant, so it is the copy's hidden value there (`OpenRows.colForm c n col`). -/
theorem copies (h : TraceOk c pin) (mask : Nat → Nat → Bool) {nm : String} {q n i col : Nat}
    {rw : _root_.VStar.In.Row} (hq : c.ports.findIdx? (·.name == nm) = some q) (hr : portRow pin nm n = some rw)
    (hi : i < 16 * rw.words.size) (hc : RecDeploy.portCol c 0 q i = some col) :
    col < c.unit.constPos ∧ (A mask c pin).hid.getD (n / c.g * 2 ^ c.kLog + (c.unitRange.slot (n % c.g * c.upv) + col))
      false = (rw.words.getD (i / 16) 0).testBit (i % 16) := by
  obtain ⟨hqs, hname, hget⟩ := name_of_findIdx hq
  have hn := count_of_portRow h hr
  have hr' : rowAt pin (c.ports.getD q default).name n = some rw := by rw [hget, hname]; exact hr
  obtain ⟨hws, _⟩ := h.rows q hqs n rw (hname ▸ hr)
  refine ⟨?_, row_bit h mask hqs hn hr' (by rw [hget]; exact hws) (by rw [hget, ← hws]; exact hi) hc⟩
  unfold RecDeploy.portCol at hc
  simp only [] at hc
  cases hk : (c.leavesIn.getD 0 #[]).findIdx? (· == ((off c q + i / 16 : Nat) : Int)) with
  | none => simp only [off] at hk; rw [hk] at hc; simp at hc
  | some kk =>
    simp only [off] at hk
    rw [hk] at hc
    simp only [Option.bind_eq_bind, Option.bind_some, Option.pure_def, Option.some.injEq] at hc
    obtain ⟨hkk, _, _⟩ := Array.findIdx?_eq_some_iff_getElem.mp hk
    subst hc
    have := h.leaf_fit; have := h.unit_const; have : i % 16 < 16 := Nat.mod_lt _ (by decide)
    omega

theorem wordBits_length (ws : Array Nat) : (_root_.VStar.In.wordBits ws).length = 16 * ws.size := by
  unfold _root_.VStar.In.wordBits
  rw [length_flatMap_range (fun (w : Nat) => w.testBit) 16, Array.length_toList]

/-- **The row width**: a placed row's hidden values are 16 a word of the stager's row. -/
theorem width (h : TraceOk c pin) {nm : String} {n : Nat} {rw : _root_.VStar.In.Row} {ps : List Nat}
    (hr : portRow pin nm n = some rw) (hps : RecDeploy.rowHidden c nm n = some ps) : ps.length = 16 * rw.words.size := by
  unfold RecDeploy.rowHidden at hps
  cases hq : c.ports.findIdx? (·.name == nm) with
  | none => rw [hq] at hps; simp at hps
  | some q =>
    obtain ⟨hqs, hname, _⟩ := name_of_findIdx hq
    rw [hq] at hps
    simp only [Option.bind_eq_bind, Option.bind_some, Array.getElem?_eq_getElem hqs] at hps
    obtain ⟨hlen, _⟩ := mapM_some hps
    rw [List.length_range] at hlen
    obtain ⟨hws, _⟩ := h.rows q hqs n rw (hname ▸ hr)
    rw [hlen, port_bits h hqs, hws]

/-- **The port lemma with the copies and the row width**: `Places`; each port's row bit at its column of instance `n`'s
copy, below the unit's constant (`copies`); and each placed row 16 hidden values a word (`width`), so its opening at the
port's width is all of its words' bits, the bytes `VStarBuild.rowMsg` lays after its prefix (`PlacesR.rowMsg_eq`). -/
structure PlacesR (c : Circuit) (pin : _root_.VStar.In.PartIn) (w : Claims.Flock.Wit) : Prop where
  places : Places c pin w
  copies : ∀ nm q n rw i col, c.ports.findIdx? (·.name == nm) = some q → portRow pin nm n = some rw →
    i < 16 * rw.words.size → RecDeploy.portCol c 0 q i = some col →
    col < c.unit.constPos ∧ w.hid.getD (n / c.g * 2 ^ c.kLog + (c.unitRange.slot (n % c.g * c.upv) + col)) false =
      (rw.words.getD (i / 16) 0).testBit (i % 16)
  width : ∀ nm n rw ps, portRow pin nm n = some rw → RecDeploy.rowHidden c nm n = some ps → ps.length = 16 * rw.words.size

/-- **`PlacesR` for the trace assigner `A`**, on a circuit and port inputs `TraceOk` holds of. -/
theorem placesR_assign (h : TraceOk c pin) (mask : Nat → Nat → Bool) : PlacesR c pin (A mask c pin) :=
  ⟨places_assign h mask, fun _ _ _ _ _ _ hq hr hi hc => copies h mask hq hr hi hc, fun _ _ _ _ hr hps => width h hr hps⟩

/-- A placed row's bits at the port's width are all of its words' bits. -/
theorem PlacesR.take_full {w : Claims.Flock.Wit} (hP : PlacesR c pin w) {nm : String} {n : Nat}
    {rw : _root_.VStar.In.Row} {ps : List Nat} (hr : portRow pin nm n = some rw)
    (hps : RecDeploy.rowHidden c nm n = some ps) :
    (_root_.VStar.In.wordBits rw.words).take ps.length = _root_.VStar.In.wordBits rw.words := by
  rw [hP.width nm n rw ps hr hps]
  exact List.take_of_length_le (Nat.le_of_eq (wordBits_length _))

/-- **A placed row's message**: `sha512/row/v1`'s prefix, then its bits at the port's width as bytes. -/
theorem PlacesR.rowMsg_eq {w : Claims.Flock.Wit} (hP : PlacesR c pin w) {nm : String} {n : Nat}
    {rw : _root_.VStar.In.Row} {ps : List Nat} (hr : portRow pin nm n = some rw)
    (hps : RecDeploy.rowHidden c nm n = some ps) :
    RecDeploy.VStarBuild.rowMsg rw.words = (_root_.Flock.HmRow.rowPrefix rw.words.size).data.toList ++
      Claims.Recursion.VStar.bytesOf ((_root_.VStar.In.wordBits rw.words).take ps.length) := by
  rw [hP.take_full hr hps]
  rfl

end RecDeploy.VStarTrace
