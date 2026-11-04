import Properties.Pouw.PearlPlusDraw
import Proofs.Properties.Pouw.Commit

/-!
# Pearl+'s run draw (`Properties.Pouw.PearlPlusDraw`)

`DrawIsOnItsStreams`, by `pearlDraw`'s steps. Its filter keeps the positive tickets with their positions (`posOf`), and
`weightedDraw` reads draw `j`'s integer below their total, which is the run's `W` (`last_sums`), from `uniform`, which
is `uniformOn` of the stream at `("u", j)`. Over their running sums (`runSums`) `bisectRight` finds the unit the integer
falls in (`pick_posOf`), the unit `finSigmaFinEquiv` numbers that ticket on (`ticketAt_fst`): both count every unit's
weight before it and skip a unit of none. Where no ticket holds weight, the filter keeps none and the draw is empty.
`CoinBytesGiveUniformTickets` is a `sorry` stub so far.
-/

namespace Properties.Pouw.PearlPlus

open PouwCommit PouwCheck.PearlPlus

open Finset

/-! ## Running sums, positive tickets and the ticket a number falls in -/

/-- The running sums `weightedDraw` folds, from `s`. -/
def runSums : List ℕ → ℕ → List ℕ
  | [], _ => []
  | w :: ws, s => (s + w) :: runSums ws (s + w)

theorem foldl_sums : ∀ (ws : List ℕ) (a : Array ℕ) (s : ℕ),
    (ws.foldl (fun (acc : Array ℕ × ℕ) w => (acc.1.push (acc.2 + w), acc.2 + w)) (a, s)).1.toList =
      a.toList ++ runSums ws s
  | [], a, s => by simp [runSums]
  | w :: ws, a, s => by
    rw [List.foldl_cons, foldl_sums ws, runSums]
    simp

/-- The unit ticket `r` falls in, by weights `ws`: past each unit's weight, the next. -/
def pickIdx : List ℕ → ℕ → ℕ
  | [], _ => 0
  | w :: ws, r => if r < w then 0 else pickIdx ws (r - w) + 1

/-- The positive weights of `ws`, each with its position from `n`: `pearlDraw`'s tickets. -/
def posOf (ws : List ℕ) (n : ℕ) : List (ℕ × ℕ) := (ws.zipIdx n).filter (·.1 > 0)

theorem posOf_cons (w : ℕ) (ws : List ℕ) (n : ℕ) :
    posOf (w :: ws) n = if w > 0 then (w, n) :: posOf ws (n + 1) else posOf ws (n + 1) := by
  unfold posOf
  rw [List.zipIdx_cons, List.filter_cons]
  by_cases hw : w > 0 <;> simp [hw]

/-- The last running sum of the positive weights is their total, which is all of `ws`'s, and there is none when it is
0. -/
theorem last_sums : ∀ (ws : List ℕ) (n s : ℕ),
    (runSums ((posOf ws n).map (·.1)) s).getLast? = if ws.sum = 0 then none else some (s + ws.sum)
  | [], n, s => by simp [posOf, runSums]
  | w :: ws, n, s => by
    rw [posOf_cons, List.sum_cons]
    by_cases hw : w > 0
    · rw [ite_eq_left hw, List.map_cons, runSums, List.getLast?_cons, last_sums ws (n + 1) (s + w),
        ite_eq_right (show ¬w + ws.sum = 0 by omega)]
      by_cases h0 : ws.sum = 0
      · rw [ite_eq_left h0]; simp [h0]
      · rw [ite_eq_right h0]; simp; omega
    · have hw0 : w = 0 := by omega
      subst hw0
      rw [ite_eq_right hw, last_sums ws (n + 1) s]
      simp

theorem posOf_nil {ws : List ℕ} {n : ℕ} (h : ws.sum = 0) : posOf ws n = [] := by
  have hl := last_sums ws n 0
  rw [ite_eq_left h] at hl
  cases hp : posOf ws n with
  | nil => rfl
  | cons p ps =>
    rw [hp, List.map_cons, runSums, List.getLast?_cons] at hl
    cases hl

theorem posOf_ne_nil {ws : List ℕ} {n : ℕ} (h : ws.sum ≠ 0) : posOf ws n ≠ [] := by
  intro hp
  have hl := last_sums ws n 0
  rw [ite_eq_right h, hp] at hl
  cases hl

/-- **`bisectRight` over the positive weights' running sums finds the unit the number falls in**, by its position in
`ws`: the zero weights `pearlDraw` drops hold no number. -/
theorem pick_posOf : ∀ (ws : List ℕ) (n s r : ℕ), s ≤ r → r < s + ws.sum →
    ((posOf ws n).getD (bisectRight (runSums ((posOf ws n).map (·.1)) s) r) (0, 0)).2 = n + pickIdx ws (r - s)
  | [], n, s, r, h1, h2 => by simp at h2; omega
  | w :: ws, n, s, r, h1, h2 => by
    rw [List.sum_cons] at h2
    rw [posOf_cons]
    by_cases hw : w > 0
    · rw [ite_eq_left hw, List.map_cons, runSums]
      unfold bisectRight
      by_cases hr : r < s + w
      · rw [List.takeWhile_cons_of_neg (by simp; omega)]
        simp [pickIdx, show r - s < w by omega]
      · rw [List.takeWhile_cons_of_pos (by simp; omega), List.length_cons, List.getD_cons_succ]
        have := pick_posOf ws (n + 1) (s + w) r (by omega) (by omega)
        unfold bisectRight at this
        rw [this, pickIdx, ite_eq_right (show ¬r - s < w by omega), show r - s - w = r - (s + w) by omega]
        omega
    · have hw0 : w = 0 := by omega
      subst hw0
      rw [ite_eq_right hw, pick_posOf ws (n + 1) s r h1 (by omega), pickIdx, ite_eq_right (by omega), Nat.sub_zero]
      omega

/-- A number between the sums of the weights before `u` and through `u` falls in `u`. -/
theorem pickIdx_of : ∀ (ws : List ℕ) (u r : ℕ), u < ws.length →
    ∑ i ∈ range u, ws.getD i 0 ≤ r → r < ∑ i ∈ range u, ws.getD i 0 + ws.getD u 0 → pickIdx ws r = u
  | [], u, r, hu, _, _ => by simp at hu
  | w :: ws, 0, r, _, _, h2 => by
    simp only [range_zero, sum_empty, List.getD_cons_zero, zero_add] at h2
    simp [pickIdx, h2]
  | w :: ws, u + 1, r, hu, h1, h2 => by
    rw [sum_range_succ'] at h1 h2
    simp only [List.getD_cons_succ, List.getD_cons_zero] at h1 h2
    rw [pickIdx, ite_eq_right (show ¬r < w by omega),
      pickIdx_of ws u (r - w) (by simp at hu; omega) (by omega) (by omega)]

theorem weightsOf_eq (tk : List ((ℕ × ℕ) × ℕ)) (i : ℕ) : weightsOf tk i = (tk.map (·.2)).getD i 0 := by
  unfold weightsOf
  rw [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD, List.getElem?_map]
  cases tk[i]? <;> rfl

theorem sum_range_getD : ∀ ws : List ℕ, ∑ i ∈ range ws.length, ws.getD i 0 = ws.sum
  | [] => rfl
  | w :: ws => by
    rw [List.length_cons, sum_range_succ', List.sum_cons, add_comm]
    simp only [List.getD_cons_succ, List.getD_cons_zero, sum_range_getD ws]

theorem sum_weightsOf (tk : List ((ℕ × ℕ) × ℕ)) : ∑ u : Fin tk.length, weightsOf tk u = ticketTotal tk := by
  rw [Fin.sum_univ_eq_sum_range (fun i => weightsOf tk i)]
  simp only [weightsOf_eq]
  have hl : tk.length = (tk.map (·.2)).length := by simp
  rw [hl, sum_range_getD]
  rfl

/-- **`Ticket`'s order is the same**: the ticket numbered `r` is on the unit `r` falls in (`finSigmaFinEquiv_apply`:
ticket `⟨u, i⟩` is numbered by the weights before `u`, plus `i`). -/
theorem ticketAt_fst (tk : List ((ℕ × ℕ) × ℕ)) (r : ℕ) (h : r < ∑ u : Fin tk.length, weightsOf tk u) :
    ((ticketAt tk ⟨r, h⟩).1 : ℕ) = pickIdx (tk.map (·.2)) r := by
  set k := ticketAt tk ⟨r, h⟩ with hk
  have ha := finSigmaFinEquiv_apply (n := fun u : Fin tk.length => weightsOf tk u) k
  have hr : (finSigmaFinEquiv k : ℕ) = r := by rw [hk, ticketAt, Equiv.apply_symm_apply]
  simp only [Fin.val_castLE] at ha
  rw [hr, Fin.sum_univ_eq_sum_range (fun i => weightsOf tk i) k.1] at ha
  have h2 := k.2.isLt
  simp only [weightsOf_eq] at ha h2
  exact (pickIdx_of _ _ _ (by simp) (by omega) (by omega)).symm

theorem tileOf_ticketAt (tk : List ((ℕ × ℕ) × ℕ)) (r : ℕ) (h : r < ∑ u : Fin tk.length, weightsOf tk u) :
    tileOf tk (ticketAt tk ⟨r, h⟩) = (tk[pickIdx (tk.map (·.2)) r]!).1 := by
  unfold tileOf
  rw [← ticketAt_fst tk r h, List.get_eq_getElem, getElem!_pos tk _ (ticketAt tk ⟨r, h⟩).1.isLt]

theorem uniformOn_lt {n : ℕ} (hn : 0 < n) {s : ByteArray} {r : ℕ}
    (h : FlockSoundness.Randomness.uniformOn n s = some r) : r < n := by
  unfold FlockSoundness.Randomness.uniformOn at h
  cases hu : Flock.Draw.uniform n s (s.size + 1) 0 with
  | none => rw [hu] at h; cases h
  | some rp =>
    rw [hu] at h
    cases h
    exact Properties.Pouw.Commit.draw_uniform_lt n hn s _ _ rp hu

/-! ## The draw -/

theorem toOption_mapM {ε α β : Type} (f : α → Except ε β) :
    ∀ l : List α, (l.mapM f).toOption = l.mapM fun a => (f a).toOption
  | [] => rfl
  | a :: l => by
    rw [List.mapM_cons, List.mapM_cons, ← toOption_mapM f l]
    cases f a <;> cases l.mapM f <;> rfl

theorem map_mapM {α β γ : Type} (f : α → Option β) (g : β → γ) :
    ∀ l : List α, (l.mapM f).map (List.map g) = l.mapM fun a => (f a).map g
  | [] => rfl
  | a :: l => by
    rw [List.mapM_cons, List.mapM_cons, ← map_mapM f g l]
    cases f a <;> cases l.mapM f <;> rfl

theorem toOption_bind_pure {ε α β : Type} (x : Except ε α) (f : α → β) :
    (x >>= fun a => pure (f a) : Except ε β).toOption = x.toOption.map f := by
  cases x <;> rfl

/-- Draw `j` of `weightedDraw` once its integer is read: its running-sum position, or the stream's refusal. -/
def drawStep (cum : List ℕ) (j : ℕ) : Option ℕ → Except String ℕ
  | some r => pure (bisectRight cum r)
  | none => throw s!"draw {j} read {DRAW_WORDS} words of its stream without one below the limit"

theorem toOption_drawStep (cum : List ℕ) (j : ℕ) (o : Option ℕ) :
    (drawStep cum j o).toOption = o.map (bisectRight cum) := by
  cases o <;> rfl

/-- `weightedDraw` over the positive tickets: draw `j`'s position of `uniform`'s integer below their total. -/
theorem weightedDraw_eq (key : ByteArray) (ws : List ℕ) (t : ℕ) (hW : ws.sum ≠ 0) :
    weightedDraw key ((posOf ws 0).map (·.1)) t = (List.range t).mapM fun j =>
      drawStep (runSums ((posOf ws 0).map (·.1)) 0) j
        (FlockSoundness.Randomness.uniform key ws.sum [.int j] (DRAW_WORDS * Flock.Draw.width ws.sum)) := by
  unfold weightedDraw
  by_cases ht : t = 0
  · subst ht; rfl
  rw [ite_eq_right ht]
  have hl := last_sums ws 0 0
  rw [ite_eq_right hW, Nat.zero_add] at hl
  obtain ⟨W, hWs⟩ : ∃ W, ws.sum = W + 1 := ⟨ws.sum - 1, by omega⟩
  simp only [foldl_sums, List.nil_append, hl, hWs]
  rfl

theorem pearlDraw_eq (key : ByteArray) (ws : List ℕ) (t : ℕ) : pearlDraw key ws t =
    if (posOf ws 0).isEmpty then pure [] else
      weightedDraw key ((posOf ws 0).map (·.1)) t >>= fun ids =>
        pure (ids.map fun i => ((posOf ws 0).getD i (0, 0)).2) :=
  rfl

theorem pick_eq (tk : List ((ℕ × ℕ) × ℕ)) (o : Option ℕ)
    (ho : ∀ r, o = some r → r < ticketTotal tk) :
    ((o.map (bisectRight (runSums ((posOf (tk.map (·.2)) 0).map (·.1)) 0))).map
        (fun i => ((posOf (tk.map (·.2)) 0).getD i (0, 0)).2)).map (fun i => (tk[i]!).1) =
      o.bind fun r => if h : r < ∑ u : Fin tk.length, weightsOf tk u then some (tileOf tk (ticketAt tk ⟨r, h⟩))
        else none := by
  cases o with
  | none => rfl
  | some r =>
    have hr : r < ∑ u : Fin tk.length, weightsOf tk u := by rw [sum_weightsOf]; exact ho r rfl
    have hr' : r < 0 + (tk.map (·.2)).sum := by have := ho r rfl; unfold ticketTotal at this; omega
    simp only [Option.map_some, Option.bind_some, dite_eq_left hr]
    rw [tileOf_ticketAt, pick_posOf _ 0 0 r (Nat.zero_le _) hr', Nat.zero_add, Nat.sub_zero]

theorem drawIsOnItsStreams : DrawIsOnItsStreams := by
  intro passes key t
  unfold draw
  dsimp only
  generalize tickets passes = tk
  by_cases hW : ticketTotal tk = 0
  · rw [ite_eq_left hW, pearlDraw_eq, posOf_nil hW]
    rfl
  rw [ite_eq_right hW, pearlDraw_eq, ite_eq_right (by simpa using posOf_ne_nil hW), toOption_bind_pure,
    toOption_bind_pure, weightedDraw_eq key _ t hW, toOption_mapM, map_mapM, map_mapM]
  congr 1
  funext j
  rw [toOption_drawStep]
  exact pick_eq tk _ fun r h => uniformOn_lt (Nat.pos_of_ne_zero hW) h

theorem coinBytesGiveUniformTickets : CoinBytesGiveUniformTickets := sorry

end Properties.Pouw.PearlPlus
