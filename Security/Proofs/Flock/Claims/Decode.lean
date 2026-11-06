import Proofs.Flock.Claims.Counted
import ArkLib.Data.CodingTheory.GuruswamiSudan.GuruswamiSudan

/-!
# A list decoder for interleaved Reed–Solomon codes, from Guruswami–Sudan

`ListComplete` asks a decoder to list, from at least `nMin M` opened columns at distinct positions, every message `M`
whose encoding agrees with at least a fraction `agree` of them. For an interleaved Reed–Solomon code (`IRS`: `s` rows,
each a polynomial of degree below `k`, the column at position `i` their values at the point `dom i`), `IRS.decode` does
so at the code's Johnson count `IRS.nMin`, at every `agree > 0` (`IRS.listComplete_decode`).

The Johnson count is `johnsonCount k agree = ⌊(k+1)/agree²⌋ + 1`, the smallest `N` with `√((k+1)·N) < agree·N`
(`johnsonCount_le_iff`). At `agree = a/b` it is `(k+1)·b²/a² + 1` in natural division (`johnsonCount_div`), and at
`agree ≤ 1` it is above `k + 1` (`lt_johnsonCount`).

* On an input in the Johnson regime (`IRS.Regime`: the opened positions' points are distinct, and the radius
  `e = N − ⌈agree·N⌉` at `N` opened columns is below ArkLib's Guruswami–Sudan bound `N − √((k+1)·N)`, which holds once
  `agree ≤ 1` is strictly above `√((k+1)/N)`, `IRS.regime_of_johnson`, so at `N ≥ nMin`, `IRS.regime_of_nMin_le`), it
  runs `IRS.gsPath`: ArkLib's Guruswami–Sudan decoder (`GuruswamiSudan.decoder`) on each row of the code punctured to
  the opened points, then the rows' candidates joined row by row, keeping only the prefixes whose joint agreement with
  the opened columns stays at least `⌈agree·N⌉` (`extendRows`). Each row of a message that agrees with that many
  columns is within the radius of its row word, so ArkLib's completeness (`GuruswamiSudan.mem_decoder_of_dist`) puts it
  in its row's list, and the message's prefixes survive every pruning (`IRS.mem_gsPath`).
* On opened columns at distinct points outside the regime it returns `[]` (`IRS.decode_eq_nil`): `ListComplete` asks
  for nothing there, since from `nMin` columns on the regime holds at `agree ≤ 1`, and no message agrees with more than
  all of them at `agree > 1`.
* On at least `nMin` opened columns two of which share a point, it lists those messages by exhaustive search
  (`IRS.exhaustive`). This branch is still forced: positions are any naturals and a finite field has finitely many
  points, so `ListComplete` covers inputs whose columns all sit at one point, and a complete decoder lists there every
  message that agrees with all of them, at a cost at least their number
  (`Flock.ListComplete.card_le_eval_of_agree`).

In its first form `ListComplete` had no `nMin`, which is `nMin = 0`: it put every message in the list of the
empty input (`Flock.ListComplete.mem_nil`), so a `DecoderComplexity` whose step count is at least its output's length
had `|T| ≤ p(size [])` (`Flock.ListComplete.card_le_eval`).
-/

namespace Claims

open Polynomial

/-! ## The `nMin` form of the list-decoding spec -/

/-- **List decoding from `nMin` opened columns**: from opened columns at distinct positions, at least `nMin M` of them,
`decode` lists every trace `M` whose encoding agrees with at least a fraction `agree` of them. The spec's form before
the radius form (`Flock.ListWithin`) replaced it; the decoders here still meet it, with `nMin M` the
smallest count above the Johnson bound of `M`'s code. -/
def Flock.ListComplete {Col T : Type} [DecidableEq Col] (enc : T → ℕ → Col) (agree : ℝ) (nMin : T → ℕ)
    (decode : List (ℕ × Col) → List T) : Prop :=
  ∀ (os : List (ℕ × Col)) (M : T), (os.map Prod.fst).Nodup → nMin M ≤ os.length →
    agree * os.length ≤ (os.filter fun p => decide (enc M p.1 = p.2)).length → M ∈ decode os

/-! ## The Johnson count -/

/-- **The Johnson count**: `⌊(k+1)/agree²⌋ + 1`, the smallest number `N` of opened columns with
`√((k+1)·N) < agree·N` (`johnsonCount_le_iff`). -/
noncomputable def johnsonCount (k : ℕ) (agree : ℝ) : ℕ := ⌊((k : ℝ) + 1) / agree ^ 2⌋₊ + 1

theorem one_le_johnsonCount (k : ℕ) (agree : ℝ) : 1 ≤ johnsonCount k agree := Nat.le_add_left 1 _

/-- **`johnsonCount k agree` is the smallest `N` with `√((k+1)·N) < agree·N`**, at `agree > 0`. -/
theorem johnsonCount_le_iff {k N : ℕ} {agree : ℝ} (ha : 0 < agree) :
    johnsonCount k agree ≤ N ↔ Real.sqrt (((k : ℝ) + 1) * N) < agree * N := by
  have ha2 : 0 < agree ^ 2 := by positivity
  have hk : (0 : ℝ) ≤ (k : ℝ) + 1 := by positivity
  constructor
  · intro h
    have hN : ((k : ℝ) + 1) / agree ^ 2 < N := by
      have hl := Nat.lt_floor_add_one (((k : ℝ) + 1) / agree ^ 2)
      have h' : (johnsonCount k agree : ℝ) ≤ N := by exact_mod_cast h
      unfold johnsonCount at h'
      push_cast at h'
      linarith
    have hN0 : (0 : ℝ) < N := lt_of_le_of_lt (by positivity) hN
    rw [div_lt_iff₀ ha2] at hN
    rw [Real.sqrt_lt' (by positivity)]
    have := mul_lt_mul_of_pos_right hN hN0
    nlinarith
  · intro h
    have hN0 : (0 : ℝ) < N := by
      rcases (Nat.cast_nonneg (α := ℝ) N).lt_or_eq with h0 | h0
      · exact h0
      · rw [← h0] at h; simp at h
    have hsq := (Real.sqrt_lt' (by positivity)).1 h
    have hlt : ((k : ℝ) + 1) / agree ^ 2 < N := by
      rw [div_lt_iff₀ ha2]
      have : ((k : ℝ) + 1) * N < (N * agree ^ 2) * N := by nlinarith
      exact lt_of_mul_lt_mul_right this hN0.le
    have := (Nat.floor_lt (div_nonneg hk ha2.le)).2 hlt
    unfold johnsonCount
    omega

/-- **The Johnson count at a ratio**: at `agree = a/b`, `(k+1)·b²/a² + 1` in natural division. -/
theorem johnsonCount_div (k a b : ℕ) : johnsonCount k ((a : ℝ) / b) = (k + 1) * b ^ 2 / a ^ 2 + 1 := by
  unfold johnsonCount
  congr 1
  rw [div_pow, div_div_eq_mul_div]
  have : ((k : ℝ) + 1) * (b : ℝ) ^ 2 / (a : ℝ) ^ 2 = (((k + 1) * b ^ 2 : ℕ) : ℝ) / ((a ^ 2 : ℕ) : ℝ) := by
    push_cast; ring
  rw [this, Nat.floor_div_eq_div]

/-- **At `agree ≤ 1` the Johnson count is above `k + 1`.** -/
theorem lt_johnsonCount {k : ℕ} {agree : ℝ} (ha0 : 0 < agree) (ha1 : agree ≤ 1) : k + 1 < johnsonCount k agree := by
  have ha2 : agree ^ 2 ≤ 1 := pow_le_one₀ ha0.le ha1
  have hle : ((k + 1 : ℕ) : ℝ) ≤ ((k : ℝ) + 1) / agree ^ 2 := by
    push_cast
    exact le_div_self (by positivity) (by positivity) ha2
  have := Nat.le_floor hle
  unfold johnsonCount
  omega

/-- `p` holds at as many positions of `l` as `l.filter p` has entries. -/
theorem card_filter_get {α : Type} (p : α → Prop) [DecidablePred p] :
    ∀ l : List α, (Finset.univ.filter fun j : Fin l.length => p (l.get j)).card =
      (l.filter fun a => decide (p a)).length
  | [] => rfl
  | a :: l => by
    change (Finset.univ.filter fun j : Fin (l.length + 1) => p ((a :: l).get j)).card = _
    rw [Fin.card_filter_univ_succ']
    simp only [List.get_eq_getElem, Fin.val_zero, List.getElem_cons_zero, Fin.val_succ,
      List.getElem_cons_succ, List.filter_cons]
    have ih := card_filter_get p l
    simp only [List.get_eq_getElem] at ih
    rw [ih]
    split_ifs with h <;> simp [h, add_comm]

/-- Join the candidate lists `L 0, L 1, …` row by row, keeping the prefixes `keep` accepts: the surviving prefixes of
`m` rows. -/
def extendRows {α : Type} (L : ℕ → List α) (keep : List α → Bool) : ℕ → List (List α)
  | 0 => [[]]
  | m + 1 => ((extendRows L keep m).flatMap fun ps => (L m).map fun p => ps ++ [p]).filter keep

/-- A sequence whose terms are in their rows' lists and whose prefixes `keep` accepts survives as a prefix. -/
theorem mem_extendRows {α : Type} (L : ℕ → List α) (keep : List α → Bool) (P : ℕ → α) (n : ℕ)
    (hL : ∀ m < n, P m ∈ L m) (hkeep : ∀ m ≤ n, keep ((List.range m).map P) = true) :
    ∀ m ≤ n, (List.range m).map P ∈ extendRows L keep m
  | 0, _ => by simp [extendRows]
  | m + 1, hm => by
    rw [extendRows, List.mem_filter, List.mem_flatMap]
    refine ⟨⟨(List.range m).map P, mem_extendRows L keep P n hL hkeep m (by omega), ?_⟩, hkeep (m + 1) hm⟩
    rw [List.range_succ, List.map_append, List.map_singleton]
    exact List.mem_map.2 ⟨P m, hL m (by omega), rfl⟩

/-- An interleaved Reed–Solomon code with `s` rows over `F`: a message `M : T` is `s` row polynomials of degree below
`k` (`rows`), which `ofRows` reads back, and its column at position `i` is their values at the point `dom i`. -/
structure IRS (F T : Type) [Field F] (s : ℕ) where
  k : ℕ
  dom : ℕ → F
  rows : T → Fin s → F[X]
  natDegree_rows : ∀ M r, (rows M r).natDegree < k
  ofRows : (Fin s → F[X]) → T
  ofRows_rows : ∀ M, ofRows (rows M) = M

namespace IRS

/-- `M`'s column at position `i`. -/
def enc {F T : Type} [Field F] {s : ℕ} (C : IRS F T s) (M : T) (i : ℕ) : Fin s → F :=
  fun r => (C.rows M r).eval (C.dom i)

/-- How many opened columns `M`'s encoding agrees with, as `ListComplete` counts them. -/
def agreeCount {F T : Type} [Field F] [DecidableEq F] {s : ℕ} (C : IRS F T s) (M : T)
    (os : List (ℕ × (Fin s → F))) : ℕ :=
  (os.filter fun p => decide (C.enc M p.1 = p.2)).length

/-- The rows `ps` (rows `0, …, ps.length − 1`) agree with the column `c` at the point `x`. -/
noncomputable def fits {F : Type} [Field F] [DecidableEq F] {s : ℕ} (ps : List F[X]) (x : F) (c : Fin s → F) : Bool :=
  decide (∀ r : Fin s, r.val < ps.length → (ps.getD r 0).eval x = c r)

/-- How many opened columns the rows `ps` agree with. -/
noncomputable def jointCount {F : Type} [Field F] [DecidableEq F] {s : ℕ} (dom : ℕ → F) (ps : List F[X])
    (os : List (ℕ × (Fin s → F))) : ℕ :=
  (os.filter fun p => fits ps (dom p.1) p.2).length

/-- The opened positions' points, distinct. -/
def points {F T : Type} [Field F] {s : ℕ} (C : IRS F T s) (os : List (ℕ × (Fin s → F)))
    (hnd : (os.map fun p => C.dom p.1).Nodup) : Fin os.length ↪ F :=
  ⟨fun j => C.dom (os.get j).1, fun j j' h => by
    have := (List.Nodup.getElem_inj_iff hnd (i := j.val) (j := j'.val) (hi := by simp) (hj := by simp)).1
      (by simpa using h)
    exact Fin.ext this⟩

/-- Row `r` of the opened columns: a word of the code punctured to the opened points. -/
def rowWord {F : Type} {s : ℕ} (os : List (ℕ × (Fin s → F))) (r : Fin s) : Fin os.length → F :=
  fun j => (os.get j).2 r

/-- ArkLib's Guruswami–Sudan list for row `r` at radius `e`, on the code punctured to the opened points (empty past
the last row). -/
noncomputable def rowList {F T : Type} [Field F] [DecidableEq F] {s : ℕ} (C : IRS F T s)
    (os : List (ℕ × (Fin s → F))) (hnd : (os.map fun p => C.dom p.1).Nodup) (e r : ℕ) : List F[X] :=
  if h : r < s then GuruswamiSudan.decoder C.k 0 0 e (C.points os hnd) (rowWord os ⟨r, h⟩) else []

/-- The decoding threshold `⌈agree·N⌉` at `N` opened columns. -/
noncomputable def threshold {F : Type} {s : ℕ} (agree : ℝ) (os : List (ℕ × (Fin s → F))) : ℕ :=
  ⌈agree * os.length⌉₊

/-- **The Guruswami–Sudan path**: each row's Guruswami–Sudan list at radius `N − ⌈agree·N⌉`, joined row by row
keeping the prefixes that agree with at least `⌈agree·N⌉` opened columns, read back as messages. -/
noncomputable def gsPath {F T : Type} [Field F] [DecidableEq F] {s : ℕ} (C : IRS F T s) (agree : ℝ)
    (os : List (ℕ × (Fin s → F))) (hnd : (os.map fun p => C.dom p.1).Nodup) : List T :=
  (extendRows (C.rowList os hnd (os.length - threshold agree os))
    (fun ps => decide (threshold agree os ≤ jointCount C.dom ps os)) s).map
    fun ps => C.ofRows fun r => ps.getD r 0

/-- **The Johnson regime** at `agree`: the opened positions' points are distinct, and the radius
`N − ⌈agree·N⌉` is below ArkLib's Guruswami–Sudan bound `N − √((k+1)·N)`. -/
def Regime {F T : Type} [Field F] {s : ℕ} (C : IRS F T s) (agree : ℝ) (os : List (ℕ × (Fin s → F))) : Prop :=
  (os.map fun p => C.dom p.1).Nodup ∧
    (((os.length - threshold agree os : ℕ) : ℝ) < os.length - Real.sqrt ((C.k + 1) * os.length))

/-- **The code's Johnson count** at `agree`: the `nMin` at which its decoder meets `ListComplete`. -/
noncomputable def nMin {F T : Type} [Field F] {s : ℕ} (C : IRS F T s) (agree : ℝ) : ℕ := johnsonCount C.k agree

open Classical in
/-- Every message that agrees with at least `agree·N` opened columns, by exhaustive search. -/
noncomputable def exhaustive {F T : Type} [Field F] [DecidableEq F] [Fintype T] {s : ℕ} (C : IRS F T s)
    (agree : ℝ) (os : List (ℕ × (Fin s → F))) : List T :=
  (Finset.univ.filter fun M => agree * os.length ≤ C.agreeCount M os).toList

open Classical in
/-- **The decoder**: the Guruswami–Sudan path in the Johnson regime; `[]` elsewhere on opened columns at distinct
points, and on fewer than `nMin` columns; exhaustive search on at least `nMin` columns two of which share a point. -/
noncomputable def decode {F T : Type} [Field F] [DecidableEq F] [Fintype T] {s : ℕ} (C : IRS F T s)
    (agree : ℝ) (os : List (ℕ × (Fin s → F))) : List T :=
  if h : C.Regime agree os then C.gsPath agree os h.1
  else if (os.map fun p => C.dom p.1).Nodup ∨ os.length < C.nMin agree then [] else C.exhaustive agree os

/-- A message's row `r` is within `N − agreeCount` of its row word. -/
theorem hammingDist_rowWord_le {F T : Type} [Field F] [DecidableEq F] {s : ℕ} (C : IRS F T s)
    (os : List (ℕ × (Fin s → F))) (hnd : (os.map fun p => C.dom p.1).Nodup) (M : T) (r : Fin s) :
    hammingDist (rowWord os r) ((C.rows M r).eval ∘ C.points os hnd) + C.agreeCount M os ≤ os.length := by
  have hsub : hammingDist (rowWord os r) ((C.rows M r).eval ∘ C.points os hnd) ≤
      (Finset.univ.filter fun j : Fin os.length => ¬ (C.enc M (os.get j).1 = (os.get j).2)).card := by
    unfold hammingDist
    refine Finset.card_le_card fun j hj => ?_
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hj ⊢
    intro he
    apply hj
    have := congrFun he r
    simp only [enc] at this
    simp only [rowWord, Function.comp_apply, ← this]
    rfl
  rw [card_filter_get (fun a : ℕ × (Fin s → F) => ¬ (C.enc M a.1 = a.2)) os] at hsub
  have hsplit := List.length_eq_length_filter_add (l := os) (fun p => decide (C.enc M p.1 = p.2))
  unfold agreeCount
  simp only [decide_not] at hsub
  omega

/-- Row `r` of a message agreeing with at least `N − e` opened columns is in its row's Guruswami–Sudan list, when `e`
is below ArkLib's bound. -/
theorem mem_rowList {F T : Type} [Field F] [DecidableEq F] {s : ℕ} (C : IRS F T s)
    (os : List (ℕ × (Fin s → F))) (hnd : (os.map fun p => C.dom p.1).Nodup) (e : ℕ)
    (he : (e : ℝ) < os.length - Real.sqrt ((C.k + 1) * os.length)) (M : T)
    (hM : os.length ≤ e + C.agreeCount M os) (r : Fin s) :
    C.rows M r ∈ C.rowList os hnd e r := by
  rw [rowList, dite_eq_left r.isLt, Fin.eta]
  refine GuruswamiSudan.mem_decoder_of_dist he (C.natDegree_rows M r) ?_
  have := C.hammingDist_rowWord_le os hnd M r
  omega

/-- A message agreeing with an opened column agrees with it on every prefix of its rows. -/
theorem agreeCount_le_jointCount {F T : Type} [Field F] [DecidableEq F] {s : ℕ} (C : IRS F T s)
    (os : List (ℕ × (Fin s → F))) (M : T) (m : ℕ) :
    C.agreeCount M os ≤
      jointCount C.dom ((List.range m).map fun i => if h : i < s then C.rows M ⟨i, h⟩ else 0) os := by
  refine (List.monotone_filter_right os fun p hp => ?_).length_le
  simp only [decide_eq_true_eq] at hp
  simp only [fits, decide_eq_true_eq, List.length_map, List.length_range]
  intro r hr
  rw [List.getD_eq_getElem _ _ (by simpa using hr), List.getElem_map, List.getElem_range, dite_eq_left r.isLt,
    ← hp]
  rfl

/-- **The Guruswami–Sudan path is complete**: on distinct points, at a radius below ArkLib's bound, it lists every
message that agrees with at least `agree·N` opened columns. -/
theorem mem_gsPath {F T : Type} [Field F] [DecidableEq F] {s : ℕ} (C : IRS F T s) (agree : ℝ)
    (os : List (ℕ × (Fin s → F))) (hnd : (os.map fun p => C.dom p.1).Nodup)
    (hJ : (((os.length - threshold agree os : ℕ) : ℝ) < os.length - Real.sqrt ((C.k + 1) * os.length)))
    (M : T) (hM : agree * os.length ≤ C.agreeCount M os) : M ∈ C.gsPath agree os hnd := by
  have ht : threshold agree os ≤ C.agreeCount M os := Nat.ceil_le.2 hM
  set P : ℕ → F[X] := fun i => if h : i < s then C.rows M ⟨i, h⟩ else 0 with hP
  have hpre := mem_extendRows (C.rowList os hnd (os.length - threshold agree os))
    (fun ps => decide (threshold agree os ≤ jointCount C.dom ps os)) P s
    (fun m hm => by
      have := C.mem_rowList os hnd _ hJ M (by have := C.agreeCount M os; omega) ⟨m, hm⟩
      simpa [hP, hm] using this)
    (fun m _ => decide_eq_true (ht.trans (C.agreeCount_le_jointCount os M m))) s le_rfl
  refine List.mem_map.2 ⟨_, hpre, ?_⟩
  conv_rhs => rw [← C.ofRows_rows M]
  congr 1
  funext r
  rw [List.getD_eq_getElem _ _ (by simp), List.getElem_map, List.getElem_range, hP]
  exact dite_eq_left r.isLt

/-- In the Johnson regime the decoder is the Guruswami–Sudan path. -/
theorem decode_eq_gsPath {F T : Type} [Field F] [DecidableEq F] [Fintype T] {s : ℕ} (C : IRS F T s)
    (agree : ℝ) (os : List (ℕ × (Fin s → F))) (h : C.Regime agree os) :
    C.decode agree os = C.gsPath agree os h.1 := by
  rw [decode, dite_eq_left h]

/-- **The Johnson regime**: distinct points, and `agree ≤ 1` strictly above the Johnson fraction `√((k+1)/N)` of the
code punctured to the `N` opened points. -/
theorem regime_of_johnson {F T : Type} [Field F] {s : ℕ} (C : IRS F T s) (agree : ℝ)
    (os : List (ℕ × (Fin s → F))) (hnd : (os.map fun p => C.dom p.1).Nodup) (ha : agree ≤ 1)
    (hJ : Real.sqrt ((C.k + 1) * os.length) < agree * os.length) : C.Regime agree os := by
  refine ⟨hnd, ?_⟩
  have hN : (0 : ℝ) ≤ os.length := Nat.cast_nonneg _
  have htN : threshold agree os ≤ os.length :=
    Nat.ceil_le.2 (by nlinarith [mul_le_mul_of_nonneg_right ha hN])
  rw [Nat.cast_sub htN]
  have := Nat.le_ceil (agree * os.length)
  unfold threshold
  linarith

/-- **From the Johnson count on, the regime holds**: distinct points, `0 < agree ≤ 1` and at least `nMin` opened
columns. -/
theorem regime_of_nMin_le {F T : Type} [Field F] {s : ℕ} (C : IRS F T s) (agree : ℝ)
    (os : List (ℕ × (Fin s → F))) (hnd : (os.map fun p => C.dom p.1).Nodup) (ha0 : 0 < agree) (ha1 : agree ≤ 1)
    (hN : C.nMin agree ≤ os.length) : C.Regime agree os :=
  C.regime_of_johnson agree os hnd ha1 ((johnsonCount_le_iff ha0).1 hN)

/-- On opened columns at distinct points outside the regime the decoder returns `[]`. -/
theorem decode_eq_nil {F T : Type} [Field F] [DecidableEq F] [Fintype T] {s : ℕ} (C : IRS F T s)
    (agree : ℝ) (os : List (ℕ × (Fin s → F))) (hnd : (os.map fun p => C.dom p.1).Nodup) (h : ¬ C.Regime agree os) :
    C.decode agree os = [] := by
  rw [decode, dite_eq_right h, ite_eq_left (Or.inl hnd)]

/-- **The decoder meets `ListComplete` at the code's Johnson count**, at every `agree > 0`. -/
theorem listComplete_decode {F T : Type} [Field F] [DecidableEq F] [Fintype T] {s : ℕ} (C : IRS F T s)
    (agree : ℝ) (ha : 0 < agree) : Flock.ListComplete C.enc agree (fun _ => C.nMin agree) (C.decode agree) := by
  intro os M _ hN hM
  unfold decode
  split_ifs with h h'
  · exact C.mem_gsPath agree os h.1 h.2 M hM
  · exfalso
    rcases h' with hnd | hlt
    · rcases le_or_gt agree 1 with ha1 | ha1
      · exact h (C.regime_of_nMin_le agree os hnd ha ha1 hN)
      · have hN1 : (1 : ℝ) ≤ os.length := by exact_mod_cast (one_le_johnsonCount C.k agree).trans hN
        have hc : ((os.filter fun p => decide (C.enc M p.1 = p.2)).length : ℝ) ≤ os.length := by
          exact_mod_cast List.length_filter_le _ _
        nlinarith
    · exact absurd hN (not_le.2 hlt)
  · classical
    unfold exhaustive
    rw [Finset.mem_toList, Finset.mem_filter]
    exact ⟨Finset.mem_univ M, hM⟩

end IRS

namespace Flock

/-- **A complete decoder lists every message that agrees with every opened column**, at `agree ≤ 1`, from `nMin` of
them on. -/
theorem ListComplete.mem_of_agree {Col T : Type} [DecidableEq Col] {enc : T → ℕ → Col} {agree : ℝ} {nMin : T → ℕ}
    {decode : List (ℕ × Col) → List T} (h : ListComplete enc agree nMin decode) (ha : agree ≤ 1)
    (os : List (ℕ × Col)) (hnd : (os.map Prod.fst).Nodup) (M : T) (hN : nMin M ≤ os.length)
    (hM : ∀ q ∈ os, enc M q.1 = q.2) : M ∈ decode os := by
  refine h os M hnd hN ?_
  rw [List.filter_eq_self.2 fun q hq => decide_eq_true (hM q hq)]
  exact mul_le_of_le_one_left (Nat.cast_nonneg _) ha

/-- **A complete decoder's complexity polynomial is at least the number of messages that agree with every opened
column** of an input of distinct positions, each message from its `nMin` on, at `agree ≤ 1`, when its step count is at
least the length of its output there. With every column at one point (distinct positions can share one: a finite field
has finitely many), those are all the messages with one value there. -/
theorem ListComplete.card_le_eval_of_agree {Col T : Type} [DecidableEq Col] {enc : T → ℕ → Col} {agree : ℝ}
    {nMin : T → ℕ} {decode : List (ℕ × Col) → List T} (h : ListComplete enc agree nMin decode) (ha : agree ≤ 1)
    {counted : List (ℕ × Col) → List T × ℕ} {size : List (ℕ × Col) → ℕ} {p : Polynomial ℕ}
    (hd : DecoderComplexity decode counted size p) (os : List (ℕ × Col)) (hnd : (os.map Prod.fst).Nodup)
    (hc : (decode os).length ≤ (counted os).2) (S : Finset T)
    (hS : ∀ M ∈ S, nMin M ≤ os.length ∧ ∀ q ∈ os, enc M q.1 = q.2) : S.card ≤ p.eval (size os) := by
  classical
  calc S.card ≤ (decode os).toFinset.card := Finset.card_le_card fun M hM =>
        List.mem_toFinset.2 (h.mem_of_agree ha os hnd M (hS M hM).1 (hS M hM).2)
    _ ≤ (decode os).length := List.toFinset_card_le _
    _ ≤ (counted os).2 := hc
    _ ≤ p.eval (size os) := (hd os).2

/-- Under the spec as first frozen (`nMin = 0`), a complete decoder lists every message on the empty input:
`agree · 0 ≤ 0` holds at every `agree`. -/
theorem ListComplete.mem_nil {Col T : Type} [DecidableEq Col] {enc : T → ℕ → Col} {agree : ℝ}
    {decode : List (ℕ × Col) → List T} (h : ListComplete enc agree (fun _ => 0) decode) (M : T) : M ∈ decode [] :=
  h [] M List.nodup_nil le_rfl (by simp)

/-- **The reason for ruling 2**: under the spec as first frozen (`nMin = 0`), a complete decoder's complexity polynomial
is at least `|T|` at the empty input, when its step count is at least the length of its output there. -/
theorem ListComplete.card_le_eval {Col T : Type} [DecidableEq Col] [Fintype T] {enc : T → ℕ → Col} {agree : ℝ}
    {decode : List (ℕ × Col) → List T} (h : ListComplete enc agree (fun _ => 0) decode)
    {counted : List (ℕ × Col) → List T × ℕ} {size : List (ℕ × Col) → ℕ} {p : Polynomial ℕ}
    (hd : DecoderComplexity decode counted size p) (hc : (decode []).length ≤ (counted []).2) :
    Fintype.card T ≤ p.eval (size []) := by
  classical
  calc Fintype.card T = (Finset.univ : Finset T).card := Finset.card_univ.symm
    _ ≤ (decode []).toFinset.card := Finset.card_le_card fun M _ => List.mem_toFinset.2 (h.mem_nil M)
    _ ≤ (decode []).length := List.toFinset_card_le _
    _ ≤ (counted []).2 := hc
    _ ≤ p.eval (size []) := (hd []).2

end Flock

end Claims
