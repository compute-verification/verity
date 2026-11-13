module

public import Properties.Pouw.PearlPlusDraw
public import Properties.Pouw.PearlPlus.Witnesses
public import Proofs.Properties.Pouw.Commit
public import Proofs.Flock.Soundness.ExecDraw
import Proofs.Core.Game.Basic
import Proofs.Core.Game.Prob
import Proofs.Pouw.Prob

/-!
# Pearl+'s run draw (`Properties.Pouw.PearlPlusDraw`)

`DrawIsOnItsStreams`, by `pearlDraw`'s steps. Its filter keeps the positive tickets with their positions (`posOf`), and
`weightedDraw` reads draw `j`'s integer below their total, which is the run's `W` (`last_sums`), from `uniform`, which
is `uniformOn` of the stream at `("u", j)`. Over their running sums (`runSums`) `bisectRight` finds the unit the integer
falls in (`pick_posOf`), the unit `finSigmaFinEquiv` numbers that ticket on (`ticketAt_fst`): both count every unit's
weight before it and skip a unit of none. The statement's premise, that some ticket holds weight, is what keeps the
filter's list nonempty (`posOf_ne_nil`).

`CoinBytesGiveUniformTickets`, by counting. `uniformOn` of a stream's bytes is C-Flock's rejection sampler on them
(`uniformOn_bytesOf`, through `DrawExec.uniform_streamOf`), so at most `256^L / W` streams of `L` bytes give each
integer below `W` (`card_uniformOn_le`, `DrawExec.uniformL_le` at one value). The streams that number tickets in `E` lie
in the union over `E`'s tickets `c` of the products of those sets, each of at most `(256^L / W)^t` streams, and the
tickets number `W^t` (`card_ticket_total`). Where `W^t` is 0 there is no ticket to number and no stream in the event.

Their witnesses are at the honest pass of the deployed scheme (`honestPass`, one tile of 3,047,424 tickets): its
tickets hold weight and each draw has bytes to read, a draw that names no tile is not on its streams, and a sampler
that returns 0 whatever it reads numbers ticket 0 more often than uniform tickets do.
-/

@[expose] public section

namespace Properties.Pouw.PearlPlus

open PouwCommit PouwCheck.PearlPlus

open Finset

open scoped ENNReal

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
        pure (ids.map fun i => ((posOf ws 0).getD i (0, 0)).2) := by
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
  intro hpos
  have hW : ¬ticketTotal tk = 0 := by omega
  rw [pearlDraw_eq, ite_eq_right (by simpa using posOf_ne_nil hW), toOption_bind_pure,
    toOption_bind_pure, weightedDraw_eq key _ t hW, toOption_mapM, map_mapM, map_mapM]
  congr 1
  funext j
  rw [toOption_drawStep]
  exact pick_eq tk _ fun r h => uniformOn_lt hpos h

/-! ## The coin bytes -/

open FlockSoundness.Audit.DrawExec (bsOf uniformL)

/-- `uniformOn` of a stream's bytes is the rejection sampler on them. -/
theorem uniformOn_bytesOf {L : ℕ} (s : Fin L → Fin 256) (n : ℕ) :
    FlockSoundness.Randomness.uniformOn n (Properties.Spot.bytesOf s) =
      (uniformL n (L + 1) (bsOf s)).map Prod.fst := by
  have h := FlockSoundness.Audit.DrawExec.uniform_streamOf s n (L + 1) 0 (Nat.zero_le _)
  rw [List.drop_zero] at h
  have hb : Properties.Spot.bytesOf s = FlockSoundness.Audit.Law.streamOf s := rfl
  unfold FlockSoundness.Randomness.uniformOn
  rw [hb, FlockSoundness.Audit.DrawExec.size_streamOf]
  revert h
  cases uniformL n (L + 1) (bsOf s) with
  | none => intro h; rw [h]; rfl
  | some p =>
    obtain ⟨r, rest⟩ := p
    rintro ⟨q, -, -, h⟩
    rw [h]
    rfl

/-- **At most `256^L / n` streams of `L` bytes give each integer below `n`.** -/
theorem card_uniformOn_le {L n v : ℕ} (hv : v < n) :
    (univ.filter fun s : Fin L → Fin 256 =>
      FlockSoundness.Randomness.uniformOn n (Properties.Spot.bytesOf s) = some v).card * n ≤ 256 ^ L := by
  classical
  have hn : 0 < n := by omega
  have hQ : ∀ r < n, ∀ ℓ, Verity.Game.prCoin (fun _ : Fin ℓ → Fin 256 => r = v) ≤ if r = v then 1 else 0 := by
    intro r _ ℓ
    by_cases hr : r = v
    · rw [ite_eq_left hr]; exact FlockSoundness.Audit.DrawExec.prCoin_le_one _
    · simp only [hr, ite_false]; exact (Verity.Game.prCoin_false).le
  have hle := FlockSoundness.Audit.DrawExec.uniformL_le n hn (fun r _ => r = v) _ hQ (L + 1) L
  rw [Finset.sum_ite_eq' (range n) v (fun _ => (1 : ℝ≥0∞)), ite_eq_left (Finset.mem_range.2 hv)] at hle
  have hev : (fun g : Fin L → Fin 256 => ∃ r rest, uniformL n (L + 1) (bsOf g) = some (r, rest) ∧ r = v) =
      fun s => FlockSoundness.Randomness.uniformOn n (Properties.Spot.bytesOf s) = some v := by
    funext s
    rw [uniformOn_bytesOf]
    apply propext
    cases uniformL n (L + 1) (bsOf s) with
    | none => simp
    | some p => obtain ⟨r, rest⟩ := p; simp [eq_comm]
  rw [hev, Verity.Game.prCoin_eq_card] at hle
  simp only [Fintype.card_fun, Fintype.card_fin, Nat.cast_pow, Nat.cast_ofNat] at hle
  have hc : ((256 : ℝ≥0∞) ^ L) ≠ 0 := by positivity
  have hc' : ((256 : ℝ≥0∞) ^ L) ≠ ⊤ := ENNReal.pow_ne_top (by norm_num)
  rw [ENNReal.div_le_iff hc hc'] at hle
  have hn0 : (n : ℝ≥0∞) ≠ 0 := by exact_mod_cast hn.ne'
  have := mul_le_mul_of_nonneg_right hle (by simp : (0 : ℝ≥0∞) ≤ n)
  rw [mul_comm (1 / (n : ℝ≥0∞)), mul_assoc, one_div, ENNReal.inv_mul_cancel hn0 (ENNReal.natCast_ne_top n),
    mul_one] at this
  exact_mod_cast this

/-- The run's tickets number its total `W`. -/
theorem card_ticket_total (tk : List ((ℕ × ℕ) × ℕ)) :
    Fintype.card (_root_.Pouw.Ticket (weightsOf tk) tk.length) = ticketTotal tk := by
  rw [Fintype.card_sigma]
  simp only [Fintype.card_fin]
  exact sum_weightsOf tk

theorem coinBytesGiveUniformTickets : CoinBytesGiveUniformTickets := by
  intro tk t E
  classical
  set n := ticketTotal tk with hn
  set L := drawBytes tk
  set A : ℕ → Finset (Fin L → Fin 256) := fun v =>
    univ.filter fun s => FlockSoundness.Randomness.uniformOn n (Properties.Spot.bytesOf s) = some v with hA
  set S := univ.filter fun g : Fin t → Fin L → Fin 256 =>
    ∃ c : Fin t → _root_.Pouw.Ticket (weightsOf tk) tk.length,
      (∀ j, FlockSoundness.Randomness.uniformOn n (Properties.Spot.bytesOf (g j)) =
        some (finSigmaFinEquiv (c j)).val) ∧ E c with hS
  have hsub : S ⊆ (univ.filter E).biUnion fun c => Fintype.piFinset fun j => A (finSigmaFinEquiv (c j)).val := by
    intro g hg
    obtain ⟨c, hc, hE⟩ := (mem_filter.1 hg).2
    exact mem_biUnion.2
      ⟨c, mem_filter.2 ⟨mem_univ _, hE⟩, Fintype.mem_piFinset.2 fun j => mem_filter.2 ⟨mem_univ _, hc j⟩⟩
  have hprod : ∀ c : Fin t → _root_.Pouw.Ticket (weightsOf tk) tk.length,
      (∏ j, (A (finSigmaFinEquiv (c j)).val).card) * n ^ t ≤ (256 ^ L) ^ t := by
    intro c
    have h1 : n ^ t = ∏ _j : Fin t, n := by simp
    have h2 : (256 ^ L) ^ t = ∏ _j : Fin t, 256 ^ L := by simp
    rw [h1, h2, ← Finset.prod_mul_distrib]
    refine Finset.prod_le_prod fun j _ => card_uniformOn_le ?_
    exact lt_of_lt_of_eq (finSigmaFinEquiv (c j)).isLt (sum_weightsOf tk)
  have hcount : S.card * n ^ t ≤ (univ.filter E).card * (256 ^ L) ^ t := by
    calc S.card * n ^ t
        ≤ (∑ c ∈ univ.filter E, (Fintype.piFinset fun j => A (finSigmaFinEquiv (c j)).val).card) * n ^ t :=
          Nat.mul_le_mul_right _ ((card_le_card hsub).trans card_biUnion_le)
      _ = ∑ c ∈ univ.filter E, (∏ j, (A (finSigmaFinEquiv (c j)).val).card) * n ^ t := by
          rw [Finset.sum_mul]
          simp only [Fintype.card_piFinset]
      _ ≤ ∑ c ∈ univ.filter E, (256 ^ L) ^ t := Finset.sum_le_sum fun c _ => hprod c
      _ = (univ.filter E).card * (256 ^ L) ^ t := by rw [Finset.sum_const, smul_eq_mul]
  unfold _root_.Pouw.pr Claims.Pr Claims.prUniform
  have hcl : Fintype.card (Fin t → Fin L → Fin 256) = (256 ^ L) ^ t := by
    simp only [Fintype.card_fun, Fintype.card_fin]
  have hcr : Fintype.card (Fin t → _root_.Pouw.Ticket (weightsOf tk) tk.length) = n ^ t := by
    rw [Fintype.card_fun, card_ticket_total, Fintype.card_fin]
  rw [hcl, hcr]
  have hgoal : (S.card : ℝ) / ((256 ^ L) ^ t : ℕ) ≤ ((univ.filter E).card : ℝ) / (n ^ t : ℕ) := by
    rcases Nat.eq_zero_or_pos (n ^ t) with h0 | hpos
    · have : IsEmpty (Fin t → _root_.Pouw.Ticket (weightsOf tk) tk.length) :=
        Fintype.card_eq_zero_iff.1 (hcr.trans h0)
      have hS0 : S = ∅ := Finset.eq_empty_of_forall_notMem fun g hg => by
        obtain ⟨c, -⟩ := (mem_filter.1 hg).2
        exact isEmptyElim c
      rw [hS0, Finset.card_empty, Nat.cast_zero, zero_div]
      positivity
    · rw [div_le_div_iff₀ (by positivity) (by exact_mod_cast hpos)]
      exact_mod_cast hcount
  convert hgoal

/-! ## Witnesses -/

/-- The honest pass's one tile holds 3,047,424 tickets: `wref(1, 1024, 64)` a cell over its 16 credited rows. -/
theorem honest_ticketTotal : ticketTotal (tickets [honestPass]) = 3047424 := by decide +kernel

/-- Each draw has bytes to read: `DRAW_WORDS` words of at least 9 bytes. -/
theorem drawBytes_pos (tk : List ((ℕ × ℕ) × ℕ)) : 0 < drawBytes tk := by
  unfold drawBytes DRAW_WORDS Flock.Draw.width
  omega

/-- **`DrawIsOnItsStreams`'s honest lemma**: its premise holds at the honest pass, whose tickets hold weight. -/
theorem drawIsOnItsStreams_honest : 0 < ticketTotal (tickets [honestPass]) := by
  rw [honest_ticketTotal]
  omega

theorem mapM_range_one_ne_nil {β : Type} (f : ℕ → Option β) : (List.range 1).mapM f ≠ some [] := by
  rw [List.range_one, List.mapM_cons, List.mapM_nil]
  cases f 0 <;> simp

/-- **`DrawIsOnItsStreams`'s nontrivial lemma**: its conclusion fails at a draw that names no tile whatever it is asked
(`.ok []`), at the honest pass and one draw: each draw on its stream names a tile or throws. -/
theorem drawIsOnItsStreams_nontrivial :
    ¬ (Except.ok [] : Except String (List (ℕ × ℕ))).toOption =
      (List.range 1).mapM fun j =>
        (FlockSoundness.Randomness.uniformOn (ticketTotal (tickets [honestPass]))
            (FlockSoundness.Randomness.stream ByteArray.empty [.str "u", .int j]
              (drawBytes (tickets [honestPass])))).bind fun r =>
          if h : r < ∑ u : Fin (tickets [honestPass]).length, weightsOf (tickets [honestPass]) u then
            some (tileOf (tickets [honestPass]) (ticketAt (tickets [honestPass]) ⟨r, h⟩))
          else none :=
  fun h => mapM_range_one_ne_nil _ h.symm

/-- An event some `x` is outside has probability below 1. -/
theorem pr_lt_one {α : Type} [Fintype α] {P : α → Prop} {x : α} (hx : ¬P x) : _root_.Pouw.pr P < 1 := by
  classical
  have hlt : (univ.filter P).card < Fintype.card α := by
    rw [← Finset.card_univ]
    exact Finset.card_lt_card (Finset.filter_ssubset.2 ⟨x, Finset.mem_univ x, hx⟩)
  have hpos : (0 : ℝ) < Fintype.card α := by exact_mod_cast Fintype.card_pos_iff.2 ⟨x⟩
  rw [_root_.Pouw.pr_eq_card P (univ.filter P) (fun y => by simp), div_lt_one hpos]
  exact_mod_cast hlt

/-- A sampler that returns 0 whatever it reads numbers ticket 0 with probability 1, more often than uniform tickets
do wherever there are two. -/
theorem zero_sampler_exceeds (tk : List ((ℕ × ℕ) × ℕ)) (hW : 1 < ticketTotal tk) :
    ¬ _root_.Pouw.pr (fun _ : Fin 1 → Fin (drawBytes tk) → Fin 256 =>
        ∃ c : Fin 1 → _root_.Pouw.Ticket (weightsOf tk) tk.length,
          (∀ j, (some 0 : Option ℕ) = some (finSigmaFinEquiv (c j)).val) ∧ (finSigmaFinEquiv (c 0)).val = 0)
      ≤ _root_.Pouw.pr (fun c : Fin 1 → _root_.Pouw.Ticket (weightsOf tk) tk.length =>
          (finSigmaFinEquiv (c 0)).val = 0) := by
  have h0 : 0 < ∑ u : Fin tk.length, weightsOf tk u := by rw [sum_weightsOf]; omega
  have h1 : 1 < ∑ u : Fin tk.length, weightsOf tk u := by rw [sum_weightsOf]; exact hW
  have : Nonempty (Fin 1 → Fin (drawBytes tk) → Fin 256) := ⟨fun _ _ => 0⟩
  rw [_root_.Pouw.pr_eq_one fun _ => ⟨fun _ => ticketAt tk ⟨0, h0⟩, fun _ => by simp [ticketAt], by simp [ticketAt]⟩]
  exact not_le.2 (pr_lt_one (x := fun _ => ticketAt tk ⟨1, h1⟩) (by simp [ticketAt]))

/-- **`CoinBytesGiveUniformTickets`'s honest lemma**: at the honest pass there are tickets to number and bytes to
number them from, so the bound is over neither an empty type of tickets nor empty streams. -/
theorem coinBytesGiveUniformTickets_honest :
    0 < Fintype.card (_root_.Pouw.Ticket (weightsOf (tickets [honestPass])) (tickets [honestPass]).length) ∧
      0 < drawBytes (tickets [honestPass]) := by
  refine ⟨?_, drawBytes_pos _⟩
  rw [card_ticket_total]
  exact drawIsOnItsStreams_honest

/-- **`CoinBytesGiveUniformTickets`'s nontrivial lemma**: its bound fails at a sampler that returns 0 whatever it reads
(`some 0` in `uniformOn`'s place), at the honest pass's tickets, one draw and the event that the draw is ticket 0: the
sampler numbers ticket 0 with probability 1, and a uniform ticket is ticket 0 with probability 1/3,047,424. -/
theorem coinBytesGiveUniformTickets_nontrivial :
    ¬ _root_.Pouw.pr (fun _ : Fin 1 → Fin (drawBytes (tickets [honestPass])) → Fin 256 =>
        ∃ c : Fin 1 → _root_.Pouw.Ticket (weightsOf (tickets [honestPass])) (tickets [honestPass]).length,
          (∀ j, (some 0 : Option ℕ) = some (finSigmaFinEquiv (c j)).val) ∧ (finSigmaFinEquiv (c 0)).val = 0)
      ≤ _root_.Pouw.pr
        (fun c : Fin 1 → _root_.Pouw.Ticket (weightsOf (tickets [honestPass])) (tickets [honestPass]).length =>
          (finSigmaFinEquiv (c 0)).val = 0) :=
  zero_sampler_exceeds _ (by rw [honest_ticketTotal]; omega)

end Properties.Pouw.PearlPlus
