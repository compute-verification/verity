import Properties.RemoteAudit.Shared
import Proofs.Properties.RemoteAudit.Order

/-!
# One openings coin per draw: the properties at `g` windows a draw

`Shared.close`'s three outcomes (`sclose_spec`): a refusal that leaves the state, a `few` or `coin` rejection after
the group's windows are made final (`finalize`), or the draw, which gives each window of the group its share
(`drawStep`). `drawIsSubset` reads the draw off it and `Properties.RemoteAudit.drawIsSubset`, on the same `draws`.

At `g ≤ 1`, `Shared.step g` is `Session.step` (`step_le_one`), and the session properties are today's. At `g ≥ 2`,
each event keeps the invariant, the coins and every draw, and `DP`: a drawn window is below `W`, and so is drawn
each window below `W` of its draw or an earlier one (`sstep`). With draw `c`'s first window undrawn, no window of
draw `c` or later is drawn, so no close has read `coins[W + c]` or above, and the run is the same on coins that
agree below it (`sclose_coins`, `sfold_coins`, `openingsUnreadUntilDrawn`). And `Frozen`'s argument goes through
with `Shared.close` (`frz_sclose`): under `DP` the windows a draw draws are undrawn before it, and it makes them final;
it draws windows in order (`sclose_prefix`), so a drawn window's record is kept (`recordKeptAfterDraw`). `liveUnder`
is `Order`'s argument over `Shared.run g` (`srunL`): the event that draws window `t` is its draw's close, which keeps
its record (`sdraw_keeps`), and with `t` undrawn so is its draw's first window (`undrawn_first`), which is what
`OpeningsUnreadUntilDrawn` and `ReleaseOrder` read. It needs `1 ≤ g`: at `g = 0` every window is draw 0, and coins that
agree below `W` leave every window's openings coin free.

`acceptDrawsOpenable` is `Properties.RemoteAudit.acceptDrawsOpenable`'s argument across a draw's windows: each drawn
`(i, p)` falls in the passes of one window `k` (`locatePass`), its share holds `(i − passCount, p)` (`shareOf_mem`),
whose opening `checkWindow` passed (`checkWindow_accept`), and its number in the draw is the receipts before window `k`
plus its number there (`flatMap_split`, `sum_take_mid`).

`*_one`: at `g = 1` the session statements are today's (`Shared.run 1` is `Session.run`, `RemoteAudit.Shared.run_one`),
and a draw of one window is `verifyWindow`'s (`acceptDrawsOpenable_one`).
-/

open RemoteAudit RemoteAudit.Session

namespace Properties.RemoteAudit.Shared

/-- `Shared.close`'s group: window `t`'s draw's windows up to `t`. -/
def grp (g t : Nat) : List Nat := (List.range (t + 1 - t / g * g)).map (t / g * g + ·)

/-- The first fold of `Shared.close`: each window of `l` final. -/
def finalize (l : List Nat) (ws : Array Win) : Array Win :=
  l.foldl (fun ws u => ws.modify u ({· with final := true})) ws

/-- A window's draw, as `Shared.close` sets it. -/
def drawW (now : Nat) (mine : Array (Nat × Nat)) (w : Win) : Win :=
  { w with drawn := some mine, openedNs := some now, verdict := if mine.isEmpty then some .accept else w.verdict }

/-- One step of `Shared.close`'s second fold. -/
def drawStep (drawn : Array (Nat × Nat)) (now : Nat) (acc : Array Win × Nat) (u : Nat) : Array Win × Nat :=
  (acc.1.modify u (drawW now (shareOf drawn acc.2 acc.1[u]!.plans.size)), acc.2 + acc.1[u]!.plans.size)

theorem size_finalize_range (first m : Nat) (ws : Array Win) :
    (finalize ((List.range m).map (first + ·)) ws).size = ws.size := by
  induction m with
  | zero => rfl
  | succ m ih =>
    unfold finalize at ih ⊢
    rw [List.range_succ, List.map_append, List.foldl_append]
    simp [ih]

theorem finalize_range (first m : Nat) (ws : Array Win) (v : Nat) :
    (finalize ((List.range m).map (first + ·)) ws)[v]! =
      if first ≤ v ∧ v < first + m ∧ v < ws.size then {ws[v]! with final := true} else ws[v]! := by
  induction m with
  | zero => simp [finalize]; omega
  | succ m ih =>
    have hs := size_finalize_range first m ws
    unfold finalize at ih hs ⊢
    rw [List.range_succ, List.map_append, List.foldl_append]
    simp only [List.map_cons, List.map_nil, List.foldl_cons, List.foldl_nil]
    rw [getElem!_modify', ih, hs]
    repeat' split
    all_goals first | rfl | omega

/-- The passes of `ws`'s windows `first … first + m − 1`. -/
def passesIn (ws : Array Win) (first m : Nat) : Nat := ((List.range m).map fun j => ws[first + j]!.plans.size).sum

theorem passesIn_succ (ws : Array Win) (first m : Nat) :
    passesIn ws first (m + 1) = passesIn ws first m + ws[first + m]!.plans.size := by
  simp [passesIn, List.range_succ, List.sum_append]

theorem drawFold_range (drawn : Array (Nat × Nat)) (now first b : Nat) (ws : Array Win) (m : Nat) :
    (((List.range m).map (first + ·)).foldl (drawStep drawn now) (ws, b)).1.size = ws.size ∧
    (((List.range m).map (first + ·)).foldl (drawStep drawn now) (ws, b)).2 = b + passesIn ws first m ∧
    ∀ v, (((List.range m).map (first + ·)).foldl (drawStep drawn now) (ws, b)).1[v]! =
      if first ≤ v ∧ v < first + m ∧ v < ws.size then
        drawW now (shareOf drawn (b + passesIn ws first (v - first)) ws[v]!.plans.size) ws[v]!
      else ws[v]! := by
  induction m with
  | zero => refine ⟨rfl, by simp [passesIn], fun v => ?_⟩; simp; omega
  | succ m ih =>
    obtain ⟨ih1, ih2, ih3⟩ := ih
    rw [List.range_succ, List.map_append, List.foldl_append]
    simp only [List.map_cons, List.map_nil, List.foldl_cons, List.foldl_nil]
    generalize ((List.range m).map (first + ·)).foldl (drawStep drawn now) (ws, b) = out at ih1 ih2 ih3
    have hx : out.1[first + m]! = ws[first + m]! := by rw [ih3]; rw [ite_eq_right (by omega)]
    refine ⟨by simp [drawStep, ih1], ?_, fun v => ?_⟩
    · simp only [drawStep]; rw [ih2, hx, passesIn_succ, Nat.add_assoc]
    · simp only [drawStep]
      rw [getElem!_modify', ih1, hx, ih2, ih3]
      by_cases hv : first + m = v
      · subst hv
        rw [show first + m - first = m by omega]
        by_cases hl : first + m < ws.size
        · rw [ite_eq_left ⟨rfl, hl⟩, ite_eq_right (by omega), ite_eq_left ⟨by omega, by omega, hl⟩]
        · rw [ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega)]
      · rw [ite_eq_right (by omega)]
        by_cases hc : first ≤ v ∧ v < first + m ∧ v < ws.size
        · rw [ite_eq_left hc, ite_eq_left ⟨hc.1, by omega, hc.2.2⟩]
        · rw [ite_eq_right hc, ite_eq_right (by omega)]

theorem finalize_plans (first m : Nat) (ws : Array Win) (v : Nat) :
    (finalize ((List.range m).map (first + ·)) ws)[v]!.plans = ws[v]!.plans := by
  rw [finalize_range]; split <;> rfl

theorem finalize_roots (first m : Nat) (ws : Array Win) (v : Nat) :
    (finalize ((List.range m).map (first + ·)) ws)[v]!.roots = ws[v]!.roots := by
  rw [finalize_range]; split <;> rfl

theorem grp_eq {g t W : Nat} (h : t + 1 = min (t / g * g + g) W) : grp g t = drawWindows g W (t / g) := by
  unfold grp drawWindows
  rw [← h]

theorem find_none {p : Nat → Bool} {l : List Nat} (h : ∀ u, l.find? p = some u → False) : ∀ u ∈ l, p u = false := by
  intro u hu
  cases e : l.find? p with
  | some v => exact (h v e).elim
  | none => simpa using List.find?_eq_none.1 e u hu

/-- What a `Shared.close` that returns does: refuse, reject a window of its group once the group is final, or draw. -/
theorem sclose_spec {g : Nat} {s : State} {t : Int} {now : Nat} {r : Result} {s' : State}
    (h : Shared.close g s t now = .ok (r, s')) :
    ((∃ why, r = .refused why) ∧ s' = s) ∨
    (∃ u f, r = .rejected u f none ∧ s' = ({ s with windows := finalize (grp g t.toNat) s.windows }.fail u f).2) ∨
    (∃ drawn, r = .drew drawn ∧ 0 ≤ t ∧ t.toNat < s.windows.size ∧ s.windows[t.toNat]!.drawn = none ∧
      t.toNat + 1 = min (t.toNat / g * g + g) s.params.windows ∧
      (∀ j, j < t.toNat / g * g → (dr s.windows j).isSome) ∧
      (∀ u ∈ grp g t.toNat, s.params.minPasses ≤ s.windows[u]!.roots.size) ∧
      draws s.digest (t.toNat / g) s.coins[s.params.windows + t.toNat / g]! s.params.samples
        (drawLengths g s (t.toNat / g)) = some drawn ∧
      s' = { s with windows := ((grp g t.toNat).foldl (drawStep drawn now)
        (finalize (grp g t.toNat) s.windows, 0)).1 }) := by
  unfold Shared.close at h
  dsimp only at h
  repeat' split at h
  all_goals (try dsimp only at h)
  all_goals cases h
  all_goals first
    | exact Or.inl ⟨⟨_, rfl⟩, rfl⟩
    | exact Or.inr (Or.inl ⟨_, _, rfl, rfl⟩)
    | skip
  all_goals
    refine Or.inr (Or.inr ⟨_, rfl, ?_⟩)
    have hb := ‹¬(decide (t < 0) || decide (t ≥ (s.windows.size : Int))) = true›
    have hd := ‹¬s.windows[t.toNat]!.drawn.isSome = true›
    have hm := ‹¬t.toNat + 1 ≠ min (t.toNat / g * g + g) s.params.windows›
    have hord := ‹¬((s.windows.extract 0 (t.toNat / g * g)).any fun x => x.drawn.isNone) = true›
    have hfew := ‹∀ (u : Nat), List.find? _ _ = some u → False›
    have hdraws := ‹draws _ _ _ _ _ = some _›
    simp only [Bool.or_eq_true, decide_eq_true_eq, not_or] at hb
    have hm' : t.toNat + 1 = min (t.toNat / g * g + g) s.params.windows := by simpa using hm
    have hlen : (List.foldl (fun acc u => acc ++ Array.map (fun x => x.length)
        (finalize (grp g t.toNat) s.windows)[u]!.plans) #[] (grp g t.toNat)) = drawLengths g s (t.toNat / g) := by
      unfold drawLengths
      rw [← grp_eq hm']
      congr 1
      funext acc u
      unfold grp
      rw [finalize_plans]
    refine ⟨by omega, by omega, ?_, hm', fun j hj => ?_, fun u hu => ?_, ?_, rfl⟩
    · cases e : s.windows[t.toNat]!.drawn with
      | none => rfl
      | some _ => rw [e] at hd; exact absurd rfl hd
    · have hj' : j < (s.windows.extract 0 (t.toNat / g * g)).size := by
        simp only [Array.size_extract]; have : t.toNat / g * g ≤ t.toNat := Nat.div_mul_le_self _ _; omega
      have hn := fun hh => hord (Array.any_eq_true.2 ⟨j, hj', hh⟩)
      simp only [Array.getElem_extract, Nat.zero_add] at hn
      have hjs : j < s.windows.size := by simp at hj'; omega
      simp only [dr, Array.getElem?_eq_getElem hjs, Option.bind_some]
      cases e : s.windows[j].drawn with
      | none => rw [e] at hn; exact absurd rfl hn
      | some _ => rfl
    · have := find_none hfew u hu
      have e := finalize_roots (t.toNat / g * g) (t.toNat + 1 - t.toNat / g * g) s.windows u
      unfold finalize at e
      rw [e] at this
      simpa using this
    · unfold grp finalize at hlen
      rw [← hlen]
      exact hdraws

theorem drawIsSubset : DrawIsSubset := by
  intro g s t now drawn s' h
  rcases sclose_spec h with ⟨⟨_, hr⟩, -⟩ | ⟨_, _, hr, -⟩ | ⟨d, hr, ht0, hts, hund, hm, hpre, hfew, hdraws, rfl⟩
  · cases hr
  · cases hr
  cases hr
  obtain ⟨L, us, hs, h1, h2, h3, h4⟩ := Properties.RemoteAudit.drawIsSubset _ _ _ _ _ _ hdraws
  refine ⟨L, us, hs, h1, h2, h3, h4, fun u hu => ?_⟩
  obtain ⟨j, hj, rfl⟩ := List.mem_map.1 hu
  rw [List.mem_range] at hj
  have hle : t.toNat / g * g ≤ t.toNat := Nat.div_mul_le_self _ _
  show dr _ _ = _
  rw [dr_eq]
  obtain ⟨-, -, h3⟩ := drawFold_range drawn now (t.toNat / g * g) 0 (finalize (grp g t.toNat) s.windows)
    (t.toNat + 1 - t.toNat / g * g)
  have h3' := h3 (t.toNat / g * g + j)
  unfold grp at h3' ⊢
  dsimp only
  rw [h3', ite_eq_left (by rw [size_finalize_range]; omega)]
  simp only [drawW, Nat.zero_add]
  rw [finalize_plans]
  have hp : passesIn (finalize ((List.range (t.toNat + 1 - t.toNat / g * g)).map (t.toNat / g * g + ·)) s.windows)
      (t.toNat / g * g) (t.toNat / g * g + j - t.toNat / g * g) =
      passesBefore s (t.toNat / g * g) (t.toNat / g * g + j) := by
    unfold passesIn passesBefore
    congr 1
    apply List.map_congr_left
    intro x _
    rw [finalize_plans]
  rw [hp]

/-! ## A draw's accepted windows -/

/-- A window record's pass lengths. -/
abbrev lens (w : WindowRecord) : List Nat := (w.passes.map (·.plan.length)).toList

theorem length_lens (w : WindowRecord) : (lens w).length = w.passes.size := by simp

theorem passCount_eq (ws : List WindowRecord) : passCount ws = (ws.flatMap lens).length := by
  induction ws with
  | nil => rfl
  | cons w ws ih =>
    simp only [passCount, List.map_cons, List.sum_cons, List.flatMap_cons, List.length_append] at ih ⊢
    rw [ih, length_lens]

theorem receiptCount_eq (ws : List WindowRecord) : receiptCount ws = (ws.flatMap lens).sum := by
  induction ws with
  | nil => rfl
  | cons w ws ih =>
    simp only [receiptCount, List.map_cons, List.sum_cons, List.flatMap_cons, List.sum_append] at ih ⊢
    rw [ih]

/-- Each of the draw's passes is a pass of one of its windows. -/
theorem locatePass : ∀ (ws : List WindowRecord) (i : Nat), i < passCount ws →
    ∃ k, ∃ hk : k < ws.length, passCount (ws.take k) ≤ i ∧ i < passCount (ws.take k) + ws[k].passes.size
  | [], i, h => by simp [passCount] at h
  | w :: ws, i, h => by
    simp only [passCount, List.map_cons, List.sum_cons] at h
    by_cases hi : i < w.passes.size
    · exact ⟨0, by simp, by simp [passCount], by simpa [passCount] using hi⟩
    · obtain ⟨k, hk, h1, h2⟩ := locatePass ws (i - w.passes.size) (by unfold passCount; omega)
      refine ⟨k + 1, by simpa using hk, ?_, ?_⟩
      · simp only [passCount, List.take_succ_cons, List.map_cons, List.sum_cons] at h1 ⊢; omega
      · simp only [passCount, List.take_succ_cons, List.map_cons, List.sum_cons, List.getElem_cons_succ] at h1 h2 ⊢
        omega

theorem flatMap_split (ws : List WindowRecord) (k : Nat) (hk : k < ws.length) :
    ws.flatMap lens = (ws.take k).flatMap lens ++ (lens ws[k] ++ (ws.drop (k + 1)).flatMap lens) := by
  conv => lhs; rw [← List.take_append_drop k ws, List.drop_eq_getElem_cons hk]
  rw [List.flatMap_append, List.flatMap_cons]

theorem sum_take_mid (A B C : List Nat) (i : Nat) (h1 : A.length ≤ i) (h2 : i < A.length + B.length) :
    ((A ++ (B ++ C)).take i).sum = A.sum + (B.take (i - A.length)).sum := by
  rw [List.take_append, List.take_of_length_le h1, List.take_append,
    show i - A.length - B.length = 0 by omega, List.take_zero, List.append_nil, List.sum_append]

theorem getElem!_mid (A B C : List Nat) (i : Nat) (h1 : A.length ≤ i) (h2 : i < A.length + B.length) :
    (A ++ (B ++ C)).toArray[i]! = B[i - A.length]'(by omega) := by
  rw [getElem!_pos (A ++ (B ++ C)).toArray i (by simp; omega), List.getElem_toArray,
    List.getElem_append_right h1, List.getElem_append_left]

theorem shareOf_mem (drawn : Array (Nat × Nat)) (base n k : Nat) (hk : k < drawn.size) (h1 : base ≤ drawn[k].1)
    (h2 : drawn[k].1 < base + n) : (drawn[k].1 - base, drawn[k].2) ∈ shareOf drawn base n := by
  unfold shareOf
  rw [Array.mem_filterMap]
  refine ⟨drawn[k], Array.getElem_mem hk, ?_⟩
  generalize drawn[k] = q at h1 h2 ⊢
  obtain ⟨a, b⟩ := q
  simp only at h1 h2 ⊢
  rw [ite_eq_left (by simp; omega)]

theorem acceptDrawsOpenable : AcceptDrawsOpenable := by
  intro reg appends c coin samples ws drawn hd
  obtain ⟨L, us, hs, hsz, -, hrange, -⟩ := Properties.RemoteAudit.drawIsSubset _ _ _ _ _ _ hd
  refine ⟨L, us, hs, fun openings hacc v hv => ?_⟩
  obtain ⟨j, hj, rfl⟩ := Array.mem_iff_getElem.1 hv
  have hjd : j < drawn.size := by omega
  obtain ⟨hi, hp, hidx⟩ := hrange j hjd
  rw [getElem!_pos us j hj] at hidx
  have hsize : (lengthsOf ws).size = passCount ws := by simp [lengthsOf, passCount_eq]
  obtain ⟨k, hk, hk1, hk2⟩ := locatePass ws drawn[j].1 (by omega)
  have hsplit := flatMap_split ws k hk
  have hA : ((ws.take k).flatMap lens).length = passCount (ws.take k) := (passCount_eq _).symm
  have hB : (lens ws[k]).length = ws[k].passes.size := length_lens _
  have hlist : (lengthsOf ws).toList = (ws.flatMap lens) := rfl
  have hmem := shareOf_mem drawn (passCount (ws.take k)) ws[k].passes.size j hjd hk1 hk2
  rcases hacc k hk with he | hc
  · rw [he] at hmem; simp at hmem
  obtain ⟨-, hchk⟩ := checkWindow_accept hc
  obtain ⟨m, hm, hme⟩ := Array.mem_iff_getElem.1 hmem
  have hck := hchk m hm
  simp only [hme] at hck
  have hi' : drawn[j].1 - passCount (ws.take k) < ws[k].passes.size := by omega
  refine ⟨k, hk, ?_, drawn[j].1 - passCount (ws.take k), drawn[j].2, (openings k)[m]!, hi', ?_, ?_, hck⟩
  · rw [← hidx]
    unfold receiptIndex
    rw [hlist, hsplit, sum_take_mid _ _ _ _ (by omega) (by omega), receiptCount_eq]
    omega
  · have e := getElem!_mid ((ws.take k).flatMap lens) (lens ws[k]) ((ws.drop (k + 1)).flatMap lens) drawn[j].1
      (by omega) (by omega)
    have hl : lengthsOf ws =
        ((ws.take k).flatMap lens ++ (lens ws[k] ++ (ws.drop (k + 1)).flatMap lens)).toArray := by
      unfold lengthsOf; exact congrArg List.toArray hsplit
    rw [hl, e] at hp
    rw [getElem!_pos ws[k].passes _ hi']
    simpa [lens, hA] using hp
  · rw [← hidx]
    unfold receiptIndex
    rw [hlist, hsplit, sum_take_mid _ _ _ _ (by omega) (by omega), receiptCount_eq, hA]
    simp only [lens]
    omega

theorem filterMap_self {f : Nat × Nat → Option (Nat × Nat)} :
    ∀ (l : List (Nat × Nat)), (∀ x ∈ l, f x = some x) → l.filterMap f = l
  | [], _ => rfl
  | x :: l, h => by
    rw [List.filterMap_cons, h x (by simp), filterMap_self l fun y hy => h y (by simp [hy])]

/-- A window's share of a draw over its passes alone is the whole draw. -/
theorem shareOf_all (drawn : Array (Nat × Nat)) (n : Nat) (h : ∀ k (hk : k < drawn.size), drawn[k].1 < n) :
    shareOf drawn 0 n = drawn := by
  unfold shareOf
  apply Array.ext'
  rw [Array.toList_filterMap]
  apply filterMap_self
  intro x hx
  obtain ⟨k, hk, rfl⟩ := Array.mem_iff_getElem.1 (Array.mem_toList_iff.1 hx)
  have := h k hk
  generalize drawn[k] = q at this ⊢
  obtain ⟨a, b⟩ := q
  simp only at this ⊢
  rw [ite_eq_left (by simp; omega)]
  simp

theorem find_none' {p : Nat → Bool} {l : List Nat} (h : ∀ u, l.find? p = some u → False) : l.find? p = none := by
  cases e : l.find? p with
  | none => rfl
  | some u => exact (h u e).elim

theorem sclose_coins {g : Nat} {s : State} {c : Array ByteArray} {t : Int} {now : Nat} {r : Result} {s₁ : State}
    (h₁ : Shared.close g s t now = .ok (r, s₁))
    (hc : (∃ d, r = .drew d) → c[s.params.windows + t.toNat / g]! = s.coins[s.params.windows + t.toNat / g]!) :
    Shared.close g (C c s) t now = .ok (r, C c s₁) := by
  generalize hres : Shared.close g (C c s) t now = res
  unfold Shared.close at h₁ hres
  dsimp only [C] at hres
  dsimp only at h₁
  simp only [deadline_C] at hres
  split at h₁
  · rename_i f hf
    split at hres
    · rename_i f' hf'
      cases hf.symm.trans hf'
      cases h₁; cases hres; rfl
    · rename_i hf'; exact (hf' f hf).elim
  rename_i hf
  split at hres
  · rename_i f' hf'; exact (hf f' hf').elim
  split at h₁
  · rename_i h; rw [ite_eq_left h] at hres; cases h₁; cases hres; rfl
  rename_i h; rw [ite_eq_right h] at hres
  split at h₁
  · rename_i h; rw [ite_eq_left h] at hres; cases h₁; cases hres; rfl
  rename_i h; rw [ite_eq_right h] at hres
  split at h₁
  · rename_i h; rw [ite_eq_left h] at hres; cases h₁; cases hres; rfl
  rename_i h; rw [ite_eq_right h] at hres
  split at h₁
  · rename_i h; rw [ite_eq_left h] at hres; cases h₁; cases hres; rfl
  rename_i h; rw [ite_eq_right h] at hres
  cases hdl : s.deadline t.toNat <;> simp only [hdl] at h₁ hres
  all_goals
    split at h₁
    · rename_i h; rw [ite_eq_left h] at hres; cases h₁; cases hres; rfl
    rename_i h; rw [ite_eq_right h] at hres
    split at h₁
    · rename_i u hu; rw [hu] at hres; cases h₁; cases hres; rfl
    rename_i hu
    rw [find_none' hu] at hres
    dsimp only at hres
    split at h₁
    · rename_i h; rw [ite_eq_left h] at hres; cases h₁; cases hres; rfl
    rename_i h; rw [ite_eq_right h] at hres
    split at h₁
    · rename_i drawn hd
      cases h₁
      rw [hc ⟨_, rfl⟩, hd] at hres
      cases hres
      rfl
    · cases h₁

/-! ## The windows drawn -/

theorem dr_finalize (first m : Nat) (ws : Array Win) (u : Nat) :
    dr (finalize ((List.range m).map (first + ·)) ws) u = dr ws u := by
  rw [dr_eq, dr_eq, finalize_range]; split <;> rfl

theorem dr_fail (s : State) (u : Nat) (f : Failure) (v : Nat) : dr (s.fail u f).2.windows v = dr s.windows v := by
  unfold State.fail
  dsimp only
  split
  · rw [dr_eq, dr_eq, getElem!_modify']; split <;> rfl
  · rfl

/-- In the draw, window `v`'s draw is new exactly when `v` is in the group. -/
theorem dr_drawn {g : Nat} {s : State} {t : Int} {drawn : Array (Nat × Nat)} {now : Nat} (v : Nat) :
    dr ((grp g t.toNat).foldl (drawStep drawn now) (finalize (grp g t.toNat) s.windows, 0)).1 v =
      if t.toNat / g * g ≤ v ∧ v < t.toNat + 1 ∧ v < s.windows.size then
        some (shareOf drawn (passesIn (finalize (grp g t.toNat) s.windows) (t.toNat / g * g) (v - t.toNat / g * g))
          s.windows[v]!.plans.size)
      else dr s.windows v := by
  have hle : t.toNat / g * g ≤ t.toNat := Nat.div_mul_le_self _ _
  obtain ⟨-, -, h3⟩ := drawFold_range drawn now (t.toNat / g * g) 0 (finalize (grp g t.toNat) s.windows)
    (t.toNat + 1 - t.toNat / g * g)
  rw [dr_eq]
  unfold grp at h3 ⊢
  rw [h3 v, size_finalize_range]
  split
  · rename_i h
    rw [ite_eq_left (by omega)]
    simp only [drawW, Nat.zero_add]
    rw [finalize_plans]
  · rename_i h
    rw [ite_eq_right (by omega), ← dr_eq, dr_finalize]

/-- The drawn windows are below `W`, and with a window drawn so is each window below `W` of its draw or an earlier
one. -/
def DP (g W : Nat) (ws : Array Win) : Prop :=
  (∀ u, (dr ws u).isSome → u < W) ∧ ∀ u j, (dr ws u).isSome → j / g ≤ u / g → j < W → (dr ws j).isSome

theorem div_of_range {g c u : Nat} (h1 : c * g ≤ u) (h2 : u < c * g + g) : u / g = c := by
  apply Nat.div_eq_of_lt_le
  · exact h1
  · rw [Nat.succ_mul]; exact h2

theorem lt_of_div_lt {g c j : Nat} (hg : 0 < g) (h : j / g < c) : j < c * g := (Nat.div_lt_iff_lt_mul hg).1 h

theorem range_of_div {g j : Nat} (hg : 0 < g) : j / g * g ≤ j ∧ j < j / g * g + g := by
  refine ⟨Nat.div_mul_le_self _ _, ?_⟩
  have h1 := Nat.div_add_mod j g
  have h2 := Nat.mod_lt j hg
  have h3 : j / g * g = g * (j / g) := Nat.mul_comm _ _
  omega

/-- What `Shared.close` does to the windows drawn: keeps each, and draws its group, which keeps `DP`. -/
theorem sclose_dr {g : Nat} {s : State} {t : Int} {now : Nat} {r : Result} {s₁ : State}
    (h : Shared.close g s t now = .ok (r, s₁)) :
    (∀ u, (dr s.windows u).isSome → (dr s₁.windows u).isSome) ∧
    (∀ d, r = .drew d → (dr s₁.windows t.toNat).isSome) ∧
    (0 < g → DP g s.params.windows s.windows → DP g s.params.windows s₁.windows) ∧
    s₁.params = s.params ∧ s₁.coins = s.coins := by
  rcases sclose_spec h with ⟨⟨_, hr⟩, rfl⟩ | ⟨u, f, hr, rfl⟩ | ⟨d, hr, ht0, hts, hund, hm, hpre, hfew, hdraws, rfl⟩
  · exact ⟨fun _ h => h, fun d e => (by cases hr; cases e), fun _ h => h, rfl, rfl⟩
  · refine ⟨fun v hv => ?_, fun d e => (by cases hr; cases e), fun _ hD => ?_, rfl, rfl⟩
    · rw [dr_fail]; unfold grp; rw [dr_finalize]; exact hv
    · have e : ∀ v, dr ({ s with windows := finalize (grp g t.toNat) s.windows }.fail u f).2.windows v =
          dr s.windows v := fun v => by rw [dr_fail]; unfold grp; rw [dr_finalize]
      exact ⟨fun v hv => hD.1 v (by rw [← e]; exact hv), fun a b ha hb hc => by
        rw [e] at ha ⊢; exact hD.2 a b ha hb hc⟩
  · have hle : t.toNat / g * g ≤ t.toNat := Nat.div_mul_le_self _ _
    have hmin := hm
    refine ⟨fun v hv => ?_, fun d' _ => ?_, fun hg hD => ?_, rfl, rfl⟩
    · dsimp only; rw [dr_drawn]; split
      · rfl
      · exact hv
    · dsimp only; rw [dr_drawn, ite_eq_left ⟨hle, by omega, hts⟩]; rfl
    · have htW : t.toNat + 1 ≤ s.params.windows := by omega
      have htg : t.toNat < t.toNat / g * g + g := by omega
      have hnew : ∀ v, (dr ({ s with windows := ((grp g t.toNat).foldl (drawStep d now)
          (finalize (grp g t.toNat) s.windows, 0)).1 } : State).windows v).isSome ↔
          ((t.toNat / g * g ≤ v ∧ v < t.toNat + 1 ∧ v < s.windows.size) ∨ (dr s.windows v).isSome) := by
        intro v
        dsimp only
        rw [dr_drawn]
        split
        · simp_all
        · rename_i hc; simp only [hc, false_or]
      -- no window at or past the group's first is drawn before the close
      have hnone : ∀ v, t.toNat / g * g ≤ v → (dr s.windows v).isSome → False := by
        intro v hv hd
        have hv' : v < s.params.windows := hD.1 v hd
        have := hD.2 v t.toNat hd ((Nat.le_div_iff_mul_le hg).2 hv) (by omega)
        rw [dr_eq, hund] at this
        cases this
      refine ⟨fun v hv => ?_, fun v j hv hj hjW => ?_⟩
      · rcases (hnew v).1 hv with h | h
        · omega
        · exact hD.1 v h
      · rcases (hnew v).1 hv with h | h
        · have hvc : v / g = t.toNat / g := div_of_range h.1 (by omega)
          rw [hnew]
          rcases Nat.lt_or_ge (j / g) (t.toNat / g) with hjc | hjc
          · exact Or.inr (hpre j (lt_of_div_lt hg hjc))
          · have hj' : j / g = t.toNat / g := by omega
            obtain ⟨r1, r2⟩ := range_of_div (j := j) hg
            rw [hj'] at r1 r2
            exact Or.inl ⟨r1, by omega, by omega⟩
        · exact (hnew j).2 (Or.inr (hD.2 v j h hj hjW))

/-! ## The invariant -/

theorem samePasses_of (P : Params) (ws ws' : Array Win) (hs : ws'.size = ws.size)
    (h : ∀ v, v < ws.size → Keeps P ws[v]! ws'[v]!) : SamePasses P ws ws' := by
  refine ⟨hs, fun v hv hv' => ?_⟩
  have := h v hv
  rw [getElem!_pos ws v hv, getElem!_pos ws' v hv'] at this
  exact this

theorem keeps_trans' {P : Params} {a b c : Win} (h₁ : Keeps P a b) (h₂ : Keeps P b c) : Keeps P a c := by
  obtain ⟨r1, p1, s1, h1, d1, k1, v1⟩ := h₁
  obtain ⟨r2, p2, s2, h2, d2, k2, v2⟩ := h₂
  refine ⟨r2.trans r1, p2.trans p1, s2.trans s1, h2.trans h1, fun hd => ?_, fun hd => k2 (k1 hd), fun hv => ?_⟩
  · rcases d2 hd with hb | hb
    · rcases d1 hb with ha | ha
      · exact Or.inl ha
      · exact Or.inr ha
    · rw [r1] at hb; exact Or.inr hb
  · rcases v2 hv with hb | hb
    · rcases v1 hb with ha | ha
      · exact Or.inl ha
      · exact Or.inr (k2 ha)
    · exact Or.inr hb

theorem sclose_inv {g : Nat} {s : State} {t : Int} {now : Nat} {r : Result} {s₁ : State} (h : Inv s)
    (he : Shared.close g s t now = .ok (r, s₁)) : Inv s₁ ∧ Frame s s₁ := by
  have hF : ∀ (m first : Nat), Inv { s with windows := finalize ((List.range m).map (first + ·)) s.windows } := by
    intro m first
    refine invOn_same h (samePasses_of _ _ _ (size_finalize_range _ _ _) fun v _ => ?_)
    rw [finalize_range]; split
    · exact keeps_refl _ _
    · exact keeps_refl _ _
  rcases sclose_spec he with ⟨-, rfl⟩ | ⟨u, f, -, rfl⟩ | ⟨d, -, ht0, hts, hund, hm, hpre, hfew, hdraws, rfl⟩
  · exact ⟨h, rfl, rfl, rfl⟩
  · exact ⟨inv_fail (hF _ _) _ _ _, rfl, rfl, rfl⟩
  · refine ⟨?_, rfl, rfl, rfl⟩
    have hle : t.toNat / g * g ≤ t.toNat := Nat.div_mul_le_self _ _
    obtain ⟨h1, -, h3⟩ := drawFold_range d now (t.toNat / g * g) 0 (finalize (grp g t.toNat) s.windows)
      (t.toNat + 1 - t.toNat / g * g)
    have hF' := hF (t.toNat + 1 - t.toNat / g * g) (t.toNat / g * g)
    unfold grp at h1 h3 hfew ⊢
    refine invOn_same hF' (samePasses_of _ _ _ h1 fun v hv => ?_)
    rw [h3 v]
    split
    · rename_i hc
      refine ⟨rfl, rfl, rfl, rfl, fun _ => Or.inr ?_, fun _ => rfl, fun _ => Or.inr rfl⟩
      dsimp only
      rw [finalize_roots]
      exact hfew v (List.mem_map.2 ⟨v - t.toNat / g * g, List.mem_range.2 (by omega), by omega⟩)
    · exact keeps_refl _ _

/-! ## Each event -/

theorem step_le_one {g : Nat} (hg : g ≤ 1) : Shared.step g = Session.step := by
  funext s e; unfold Shared.step; rw [ite_eq_left hg]

theorem run_le_one {g : Nat} (hg : g ≤ 1) (s : State) (es : Array Event) : Shared.run g s es = Session.run s es := by
  unfold Shared.run Session.run; rw [step_le_one hg]

theorem sstep_of {g : Nat} (hg : 2 ≤ g) {s : State} {e : Event} {r : Result} {s' : State}
    (he : Shared.step g s e = .ok (r, s')) :
    (∃ t now, e = .close t now ∧ Shared.close g s t now = .ok (r, s')) ∨
    ((∀ t now, e ≠ .close t now) ∧ Session.step s e = .ok (r, s')) := by
  unfold Shared.step at he
  rw [ite_eq_right (by omega)] at he
  cases e with
  | close t now => exact Or.inl ⟨t, now, rfl, he⟩
  | start now => exact Or.inr ⟨fun _ _ h => (by cases h), he⟩
  | boundary now => exact Or.inr ⟨fun _ _ h => (by cases h), he⟩
  | append root count => exact Or.inr ⟨fun _ _ h => (by cases h), he⟩
  | release runs withdrawn => exact Or.inr ⟨fun _ _ h => (by cases h), he⟩
  | receive t i root runs change st now => exact Or.inr ⟨fun _ _ h => (by cases h), he⟩
  | verify t openings => exact Or.inr ⟨fun _ _ h => (by cases h), he⟩

theorem keep_nonclose {s : State} {e : Event} {r : Result} {s' : State} (hn : ∀ t now, e ≠ .close t now)
    (he : Session.step s e = .ok (r, s')) : Keep s.windows s'.windows := by
  cases e with
  | start now => injection he with he; exact keep_start he
  | boundary now => injection he with he; exact keep_boundary he
  | append root count => exact keep_append he
  | release runs withdrawn => injection he with he; exact keep_release he
  | receive t i root runs change st now => injection he with he; exact keep_receive he
  | close t now => exact (hn t now rfl).elim
  | verify t openings => injection he with he; exact keep_verify he

theorem keep_dr {ws ws' : Array Win} (k : Keep ws ws') (u : Nat) : dr ws' u = dr ws u := by
  rcases k.2 u with e | ⟨l, e⟩
  · exact e
  · rw [e, dr_none l]

/-- `Shared.step` at `g ≥ 2` keeps the invariant and the frame, the coins, each window's draw, and `DP`. -/
theorem sstep {g : Nat} (hg : 2 ≤ g) {s : State} {e : Event} {r : Result} {s' : State} (h : Inv s)
    (he : Shared.step g s e = .ok (r, s')) :
    Inv s' ∧ Frame s s' ∧ s'.coins = s.coins ∧ (∀ u, (dr s.windows u).isSome → (dr s'.windows u).isSome) ∧
      (DP g s.params.windows s.windows → DP g s.params.windows s'.windows) := by
  rcases sstep_of hg he with ⟨t, now, rfl, hc⟩ | ⟨hn, hs⟩
  · obtain ⟨hi, hf⟩ := sclose_inv h hc
    obtain ⟨d1, -, d3, -, d5⟩ := sclose_dr hc
    exact ⟨hi, hf, d5, d1, d3 (by omega)⟩
  · obtain ⟨hi, hf⟩ := step_inv h hs
    have k := keep_dr (keep_nonclose hn hs)
    refine ⟨hi, hf, step_coins_eq hs, fun u hu => by rw [k]; exact hu, fun hD => ⟨fun u hu => ?_, fun u j hu => ?_⟩⟩
    · rw [k] at hu; exact hD.1 u hu
    · rw [k] at hu ⊢; exact hD.2 u j hu

theorem sstep_coins {g : Nat} (hg : 2 ≤ g) {s : State} {e : Event} {r : Result} {s₁ : State} {c : Array ByteArray}
    (he : Shared.step g s e = .ok (r, s₁)) (h0 : c[0]! = s.coins[0]!)
    (hW : ∀ k, k < s.params.windows → c[k]! = s.coins[k]!)
    (hd : ∀ t now, e = .close t now → (∃ d, r = .drew d) →
      c[s.params.windows + t.toNat / g]! = s.coins[s.params.windows + t.toNat / g]!) :
    Shared.step g (C c s) e = .ok (r, C c s₁) := by
  unfold Shared.step at he ⊢
  rw [ite_eq_right (by omega)] at he ⊢
  cases e with
  | start now => injection he with he; show pure _ = _; rw [start_coins s c now h0, he]; rfl
  | boundary now => injection he with he; show pure _ = _; rw [boundary_coins s c now hW, he]; rfl
  | append root count =>
    show append (C c s) root count = _
    rw [append_coins, show append s root count = _ from he]; rfl
  | release runs withdrawn => injection he with he; show pure _ = _; rw [release_coins, he]; rfl
  | receive t i root runs change st now =>
    injection he with he
    show pure _ = _
    rcases h2 : receive (C c s) t i root runs change st now with ⟨r₂, s₂⟩
    obtain ⟨rfl, rfl⟩ := receive_coins he h2
    rfl
  | close t now => exact sclose_coins he (hd t now rfl)
  | verify t openings => injection he with he; show pure _ = _; rw [verify_coins, he]; rfl

/-! ## Runs -/

/-- `Shared.run`'s fold, one event. -/
abbrev sAcc (g : Nat) (acc : Array Result × State) (e : Event) : Except String (Array Result × State) := do
  let (r, s') ← Shared.step g acc.2 e
  pure (acc.1.push r, s')

theorem sfold_inv (g : Nat) (P : State → Prop) (hstep : ∀ s e r s', P s → Shared.step g s e = .ok (r, s') → P s') :
    ∀ (l : List Event) (acc out : Array Result × State), P acc.2 → l.foldlM (sAcc g) acc = .ok out → P out.2 := by
  intro l
  induction l with
  | nil => intro acc out hacc h; simp only [List.foldlM_nil] at h; cases h; exact hacc
  | cons e l ih =>
    intro acc out hacc h
    rw [List.foldlM_cons] at h
    cases he : Shared.step g acc.2 e with
    | error err => simp only [sAcc, he] at h; cases h
    | ok rs' =>
      obtain ⟨r, s''⟩ := rs'
      simp only [sAcc, he] at h
      exact ih _ out (hstep _ _ _ _ hacc he) h

theorem srun_inv (g : Nat) (P : State → Prop) (hstep : ∀ s e r s', P s → Shared.step g s e = .ok (r, s') → P s')
    (s : State) (events : Array Event) (rs : Array Result) (s' : State) (hs : P s)
    (h : Shared.run g s events = .ok (rs, s')) : P s' := by
  unfold Shared.run at h
  rw [← Array.foldlM_toList] at h
  exact sfold_inv g P hstep _ (#[], s) (rs, s') hs h

theorem sfold_coins {g : Nat} (hg : 2 ≤ g) (T : Nat) (c : Array ByteArray) :
    ∀ (l : List Event) (acc out : Array Result × State), l.foldlM (sAcc g) acc = .ok out →
    Inv acc.2 → 0 < acc.2.params.windows →
    (∀ k, k < acc.2.params.windows + T → c[k]! = acc.2.coins[k]!) →
    (∀ t₀, (drawnOf out.2 t₀).isSome → t₀ / g < T) →
    l.foldlM (sAcc g) (acc.1, C c acc.2) = .ok (out.1, C c out.2) := by
  intro l
  induction l with
  | nil => intro acc out h _ _ _ _; simp only [List.foldlM_nil] at h ⊢; cases h; rfl
  | cons e l ih =>
    intro acc out h hinv h0 hag hT
    rw [List.foldlM_cons] at h ⊢
    cases he : Shared.step g acc.2 e with
    | error err => simp only [sAcc, he] at h; cases h
    | ok rs' =>
      obtain ⟨r, s'⟩ := rs'
      simp only [sAcc, he] at h
      obtain ⟨hinv', ⟨hp, -, -⟩, hc, -, -⟩ := sstep hg hinv he
      have hstep : Shared.step g (C c acc.2) e = .ok (r, C c s') := by
        refine sstep_coins hg he (hag 0 (by omega)) (fun k hk => hag k (by omega)) (fun t now ht hex => ?_)
        obtain ⟨d, hr⟩ := hex
        subst ht
        rcases sstep_of hg he with ⟨t', now', e', hcl⟩ | ⟨hn, -⟩
        · cases e'
          have hd := (sclose_dr hcl).2.1 d hr
          have hpers := sfold_inv g (fun x => Inv x ∧ (dr x.windows t.toNat).isSome)
            (fun _ _ _ _ ⟨hx, hdx⟩ he' => ⟨(sstep hg hx he').1, (sstep hg hx he').2.2.2.1 _ hdx⟩) l _ out
            ⟨hinv', hd⟩ h
          have := hT t.toNat hpers.2
          exact hag _ (by omega)
        · exact (hn t now rfl).elim
      simp only [sAcc, hstep]
      exact ih (acc.1.push r, s') out h hinv' (by rw [hp]; exact h0) (by rw [hp, hc]; exact hag) hT

theorem openingsUnreadUntilDrawn : OpeningsUnreadUntilDrawn := by
  intro g params reg coins appendCoins notBeforeNs events rs s hrun c coins' hW hnd hag
  rcases Nat.lt_or_ge g 2 with hg | hg
  · rw [run_le_one (by omega)] at hrun ⊢
    rcases (show g = 0 ∨ g = 1 by omega) with rfl | rfl
    · exact Properties.RemoteAudit.openingsUnreadUntilDrawn params reg coins appendCoins notBeforeNs events rs s
        hrun 0 coins' hW (by simpa using hnd) (fun j hj => hag j (by omega))
    · exact Properties.RemoteAudit.openingsUnreadUntilDrawn params reg coins appendCoins notBeforeNs events rs s
        hrun c coins' hW (by simpa using hnd) hag
  · have hP := srun_inv g (fun x => Inv x ∧ x.params = params ∧ DP g params.windows x.windows)
      (fun _ _ _ _ hx₀ he' => by
        obtain ⟨hx, hpx, hD⟩ := hx₀
        obtain ⟨hx', ⟨hp', -, -⟩, -, -, hD'⟩ := sstep hg hx he'
        refine ⟨hx', hp'.trans hpx, ?_⟩
        rw [← hpx] at hD ⊢; exact hD' hD)
      _ events rs s ⟨inv_init .., rfl, fun u h => by simp [dr, State.init] at h, fun u j h => by
        simp [dr, State.init] at h⟩ hrun
    obtain ⟨-, -, hD⟩ := hP
    have hlt : ∀ t₀, (drawnOf s t₀).isSome → t₀ / g < c := by
      intro t₀ ht₀
      have hW₀ := hD.1 t₀ ht₀
      rcases Nat.lt_or_ge (t₀ / g) c with hl | hl
      · exact hl
      · exfalso
        rcases Nat.lt_or_ge (c * g) params.windows with hcW | hcW
        · have := hD.2 t₀ (c * g) ht₀ (by rw [Nat.mul_div_cancel _ (by omega)]; exact hl) hcW
          change (drawnOf s (c * g)).isSome = true at this
          rw [hnd] at this; cases this
        · have := (Nat.le_div_iff_mul_le (by omega : 0 < g)).1 hl
          omega
    unfold Shared.run at hrun ⊢
    rw [← Array.foldlM_toList] at hrun ⊢
    exact sfold_coins hg c coins' _ (#[], Session.State.init params reg coins appendCoins notBeforeNs) (rs, s) hrun
      (inv_init ..) hW hag hlt

/-! ## A drawn window's record is kept -/

theorem keepsF_finalize (first m : Nat) (ws : Array Win) :
    KeepsF ws (finalize ((List.range m).map (first + ·)) ws) := by
  intro v
  rw [finalize_range]
  split
  · exact keepW_of ⟨rfl, rfl, rfl, rfl, rfl⟩ (fun _ => rfl)
  · exact keepW_refl _

/-- `Shared.close` keeps each window's `KeepW`: under `DP` the windows it draws are undrawn, and it makes them final. -/
theorem frz_sclose {g : Nat} (hg : 0 < g) {s : State} {t : Int} {now : Nat} {r : Result} {s' : State}
    (hD : DP g s.params.windows s.windows) (he : Shared.close g s t now = .ok (r, s')) : Frz s s' := by
  rcases sclose_spec he with ⟨-, rfl⟩ | ⟨u, f, -, rfl⟩ | ⟨d, -, ht0, hts, hund, hm, hpre, hfew, hdraws, rfl⟩
  · exact frz_refl _
  · have h1 : Frz s { s with windows := finalize (grp g t.toNat) s.windows } :=
      frz_of (by unfold grp; exact keepsF_finalize _ _ _) rfl rfl
    exact frz_trans h1 (frz_fail { s with windows := finalize (grp g t.toNat) s.windows } u f none)
  · refine frz_of ?_ rfl rfl
    show KeepsF s.windows ((grp g t.toNat).foldl (drawStep d now) (finalize (grp g t.toNat) s.windows, 0)).1
    unfold grp
    refine keepsF_trans (keepsF_finalize (t.toNat / g * g) (t.toNat + 1 - t.toNat / g * g) _) ?_
    intro v
    obtain ⟨-, -, h3⟩ := drawFold_range d now (t.toNat / g * g) 0
      (finalize ((List.range (t.toNat + 1 - t.toNat / g * g)).map (t.toNat / g * g + ·)) s.windows)
      (t.toNat + 1 - t.toNat / g * g)
    rw [h3 v]
    split
    · rename_i hv
      have hle : t.toNat / g * g ≤ t.toNat := Nat.div_mul_le_self _ _
      rw [size_finalize_range] at hv
      have hfin := finalize_range (t.toNat / g * g) (t.toNat + 1 - t.toNat / g * g) s.windows v
      rw [ite_eq_left ⟨hv.1, by omega, hv.2.2⟩] at hfin
      rw [hfin]
      refine keepW_draw ?_ rfl
      show s.windows[v]!.drawn = none
      cases e : s.windows[v]!.drawn with
      | none => rfl
      | some _ =>
        exfalso
        have hdv : (dr s.windows v).isSome := by rw [dr_eq, e]; rfl
        have hr := (range_of_div hg (j := t.toNat)).2
        have hvc : v / g = t.toNat / g := div_of_range hv.1 (by omega)
        have ht := hD.2 v t.toNat hdv (by omega) (by omega)
        rw [dr_eq, hund] at ht
        cases ht
    · exact keepW_refl _

/-- `Frz`'s invariant at `g ≥ 2` windows a draw: `FInv`, and `DP`. -/
def SF (g : Nat) (s : State) : Prop := FInv s ∧ DP g s.params.windows s.windows

theorem sf_step {g : Nat} (hg : 2 ≤ g) {s : State} {e : Event} {r : Result} {s' : State} (h : SF g s)
    (he : Shared.step g s e = .ok (r, s')) : SF g s' ∧ Frz s s' := by
  obtain ⟨⟨hi, hdf, hm⟩, hD⟩ := h
  obtain ⟨hi', ⟨hp, -, -⟩, -, -, hD'⟩ := sstep hg hi he
  have hf : Frz s s' := by
    rcases sstep_of hg he with ⟨t, now, rfl, hc⟩ | ⟨-, hs⟩
    · exact frz_sclose (by omega) hD hc
    · exact frz_step hi hdf hm hs
  refine ⟨⟨⟨hi', df_keepsF hdf hf.1, by rw [hp]; exact hm⟩, ?_⟩, hf⟩
  rw [hp]; exact hD' hD

theorem srun_frz {g : Nat} (hg : 2 ≤ g) (s₀ : State) (events : Array Event) (rs : Array Result) (s : State)
    (h : SF g s₀) (hr : Shared.run g s₀ events = .ok (rs, s)) : SF g s ∧ Frz s₀ s :=
  srun_inv g (fun s => SF g s ∧ Frz s₀ s) (fun _ _ _ _ hs he => by
    obtain ⟨hs, hf⟩ := hs
    obtain ⟨hs', hf'⟩ := sf_step hg hs he
    exact ⟨hs', frz_trans hf hf'⟩) s₀ events rs s ⟨h, frz_refl _⟩ hr

theorem sclose_prefix {g : Nat} {s : State} {t : Int} {now : Nat} {r : Result} {s₁ : State}
    (hp : Prefix s.windows) (he : Shared.close g s t now = .ok (r, s₁)) : Prefix s₁.windows := by
  rcases sclose_spec he with ⟨-, rfl⟩ | ⟨u, f, -, rfl⟩ | ⟨d, -, ht0, hts, hund, hm, hpre, hfew, hdraws, rfl⟩
  · exact hp
  · intro a j hj ha
    rw [dr_fail] at ha ⊢
    unfold grp at ha ⊢
    dsimp only at ha ⊢
    rw [dr_finalize] at ha ⊢
    exact hp a j hj ha
  · intro a j hj ha
    change (dr ((grp g t.toNat).foldl (drawStep d now) (finalize (grp g t.toNat) s.windows, 0)).1 a).isSome at ha
    show (dr ((grp g t.toNat).foldl (drawStep d now) (finalize (grp g t.toNat) s.windows, 0)).1 j).isSome
    rw [dr_drawn] at ha ⊢
    split at ha
    · split
      · rfl
      · exact hpre j (by omega)
    · split
      · rfl
      · exact hp a j hj ha

theorem sstep_prefix {g : Nat} (hg : 2 ≤ g) {s : State} {e : Event} {r : Result} {s' : State}
    (hp : Prefix s.windows) (he : Shared.step g s e = .ok (r, s')) : Prefix s'.windows := by
  rcases sstep_of hg he with ⟨t, now, rfl, hc⟩ | ⟨-, hs⟩
  · exact sclose_prefix hp hc
  · exact prefix_step hp (adv_step hs)

theorem recordKeptAfterDraw : RecordKeptAfterDraw := by
  intro g params reg coins appendCoins notBeforeNs events events' rs rs' s s' h₁ h₂ t d hm hd
  rcases Nat.lt_or_ge g 2 with hg | hg
  · rw [run_le_one (by omega)] at h₁ h₂
    exact Properties.RemoteAudit.recordKeptAfterDraw params reg coins appendCoins notBeforeNs events events' rs rs'
      s s' h₁ h₂ t d hm hd
  have hSF0 : SF g (State.init params reg coins appendCoins notBeforeNs) :=
    ⟨finv_init params reg coins appendCoins notBeforeNs hm, fun u h => by simp [dr, State.init] at h,
      fun u j h => by simp [dr, State.init] at h⟩
  obtain ⟨hF₁, -⟩ := srun_frz hg _ _ _ _ hSF0 h₁
  obtain ⟨hF, hk, hdig, more, happ⟩ := srun_frz hg _ _ _ _ hF₁ h₂
  have hI₁ : Inv s := hF₁.1.1
  have hI : Inv s' := hF.1.1
  have hr : s'.reg = s.reg := (srun_inv g (fun x => Inv x ∧ x.reg = s.reg)
    (fun _ _ _ _ hx he => by
      obtain ⟨hx, e⟩ := hx
      obtain ⟨hx', f, -⟩ := sstep hg hx he
      exact ⟨hx', f.2.1.trans e⟩) _ _ _ _ ⟨hI₁, rfl⟩ h₂).2
  have hpre : Prefix s.windows :=
    srun_inv g (fun s => Prefix s.windows) (fun _ _ _ _ hs he => sstep_prefix hg hs he) _ _ _ _
      (fun t j _ ht => by simp [dr, State.init] at ht) h₁
  have hd' : s.windows[t]!.drawn = some d := by rw [← dr_eq]; exact hd
  have ht : t < s.windows.size := by
    rcases Nat.lt_or_ge t s.windows.size with h | h
    · exact h
    · rw [getElem!_neg _ _ (by omega)] at hd'; cases hd'
  obtain ⟨e1, e2, e3, e4, e5⟩ := (hk t).1 (by rw [hd']; rfl)
  have hrec : s'.record t = s.record t := by
    have hprev : (if t > 0 then s'.windows[t - 1]!.roots.back?.getD ByteArray.empty else s'.digest) =
        (if t > 0 then s.windows[t - 1]!.roots.back?.getD ByteArray.empty else s.digest) := by
      by_cases htp : t > 0
      · have hs : (dr s.windows (t - 1)).isSome :=
          hpre t (t - 1) (by omega) (by show (drawnOf s t).isSome; rw [hd]; rfl)
        rw [dr_eq] at hs
        rw [ite_eq_left htp, ite_eq_left htp, ((hk (t - 1)).1 hs).2.2.1]
      · rw [ite_eq_right htp, ite_eq_right htp, hdig]
    unfold State.record
    simp only [e2, e3, e4, e5]
    rw [hprev]
  refine ⟨by rw [show drawnOf s' t = s'.windows[t]!.drawn from dr_eq _ _, e1, hd'], hrec, fun u => ?_⟩
  rw [hrec, happ]
  have key : ∀ i p o, i < (s.record t).passes.size → p < (s.record t).passes[i]!.plan.length →
      checkOpening reg reg.digest (s.appends ++ more) (s.record t) (receipterOf reg (s.record t) i) i p o =
      checkOpening reg reg.digest s.appends (s.record t) (receipterOf reg (s.record t) i) i p o := by
    intro i p o hi hp
    apply checkOpening_congr
    apply appendOf_extend
    · rw [← happ]; exact firsts_mono hI
    · intro k hk hk'
      have := firsts_big hI₁ hI more happ hr k hk (by rw [happ]; exact hk')
      have hb := record_reads hI₁ t ht i p hi hp
      rw [← happ]; omega
  constructor
  · rintro ⟨i, p, o, hi, hp, hu, hc⟩; exact ⟨i, p, o, hi, hp, hu, (key i p o hi hp) ▸ hc⟩
  · rintro ⟨i, p, o, hi, hp, hu, hc⟩; exact ⟨i, p, o, hi, hp, hu, (key i p o hi hp).symm ▸ hc⟩

/-! ## At `g = 1`: today's statements -/

theorem drawWindows_one {W c : Nat} (h : c < W) : drawWindows 1 W c = [c] := by
  unfold drawWindows
  rw [show min (c * 1 + 1) W - c * 1 = 1 by omega]
  simp

/-- At `g = 1` draw `c`'s receipts are window `c`'s, as `Session.close` numbers them. -/
theorem drawLengths_one {s : State} {c : Nat} (h : c < s.params.windows) :
    drawLengths 1 s c = s.windows[c]!.plans.map (·.length) := by
  unfold drawLengths
  rw [drawWindows_one h]
  simp

theorem openingsUnreadUntilDrawn_one (h : OpeningsUnreadUntilDrawn) :
    Properties.RemoteAudit.OpeningsUnreadUntilDrawn :=
  fun params reg coins appendCoins notBeforeNs events rs s hrun t coins' hW hnd hag =>
    h 1 params reg coins appendCoins notBeforeNs events rs s hrun t coins' hW (by rw [Nat.mul_one]; exact hnd) hag

theorem recordKeptAfterDraw_one (h : RecordKeptAfterDraw) : Properties.RemoteAudit.RecordKeptAfterDraw :=
  fun params reg coins appendCoins notBeforeNs events events' rs rs' s s' h₁ h₂ =>
    h 1 params reg coins appendCoins notBeforeNs events events' rs rs' s s' h₁ h₂

theorem releaseOrder_one (params : Params) (reg : Registration) (appendCoins : Array ByteArray) (notBeforeNs : Nat)
    (D : Array ByteArray → Array Event) :
    ReleaseOrder 1 params reg appendCoins notBeforeNs D ↔
      Properties.RemoteAudit.ReleaseOrder params reg appendCoins notBeforeNs D := by
  constructor
  · intro h coins coins' t k rs s hag hrun hnd
    exact h coins coins' t k rs s hag hrun (by rw [Nat.mul_one]; exact hnd)
  · intro h coins coins' c k rs s hag hrun hnd
    rw [Nat.mul_one] at hnd
    exact h coins coins' c k rs s hag hrun hnd

theorem liveUnder_one (params : Params) (reg : Registration) (appendCoins : Array ByteArray) (notBeforeNs : Nat)
    (D : Array ByteArray → Array Event) :
    LiveUnder 1 params reg appendCoins notBeforeNs D ↔
      Properties.RemoteAudit.LiveUnder params reg appendCoins notBeforeNs D := by
  constructor
  · intro h coins coins' t k k' rs rs' s s' hag h₁ h₂ hd hd'
    exact h coins coins' t k k' rs rs' s s' hag h₁ h₂ t (Nat.div_one t) hd hd'
  · intro h coins coins' c k k' rs rs' s s' hag h₁ h₂ t ht hd hd'
    rw [Nat.div_one] at ht
    subst ht
    exact h coins coins' t k k' rs rs' s s' hag h₁ h₂ hd hd'

/-- A draw of one window is `verifyWindow`'s: its share is the whole draw, and its receipts are numbered from 0. -/
theorem acceptDrawsOpenable_one (h : AcceptDrawsOpenable) : Properties.RemoteAudit.AcceptDrawsOpenable := by
  intro reg appends w coin samples openings hv
  obtain ⟨drawn, hd, hc⟩ := verifyWindow_accept hv
  have hl : lengthsOf [w] = w.passes.map (·.plan.length) := by apply Array.ext'; simp [lengthsOf]
  obtain ⟨_, _, _, _, _, hrange, _⟩ := Properties.RemoteAudit.drawIsSubset _ _ _ _ _ _ hd
  have hsh : shareOf drawn 0 w.passes.size = drawn :=
    shareOf_all drawn _ fun k hk => by simpa using (hrange k hk).1
  obtain ⟨L, us, hs, hus⟩ := h reg appends w.index coin samples [w] drawn (by rw [hl]; exact hd)
  rw [hl] at hs
  refine ⟨L, us, hs, fun v hv => ?_⟩
  obtain ⟨k, hk, -, ho⟩ := hus (fun _ => openings) (fun k hk => by
      have : k = 0 := by simp at hk; omega
      subst this
      right
      simpa [passCount, hsh] using hc) v hv
  have : k = 0 := by simp at hk; omega
  subst this
  simpa [receiptCount] using ho

/-! ## `Live` at `g` windows a draw -/

/-- `Shared.run g` over a list of events. -/
def srunL (g : Nat) (s : State) (l : List Event) : Except String (Array Result × State) := l.foldlM (sAcc g) (#[], s)

theorem srun_srunL (g : Nat) (s : State) (events : Array Event) :
    Shared.run g s events = srunL g s events.toList := by
  unfold Shared.run srunL
  rw [← Array.foldlM_toList]

theorem sAcc_ok {g : Nat} {acc : Array Result × State} {e : Event} {r : Result} {s' : State}
    (he : Shared.step g acc.2 e = .ok (r, s')) : sAcc g acc e = .ok (acc.1.push r, s') := by
  simp only [sAcc, he]; rfl

theorem sAcc_error {g : Nat} {acc : Array Result × State} {e : Event} {err : String}
    (he : Shared.step g acc.2 e = .error err) : sAcc g acc e = .error err := by
  simp only [sAcc, he]; rfl

theorem sfold_state {g : Nat} : ∀ (l : List Event) (acc out : Array Result × State) (b : Array Result),
    l.foldlM (sAcc g) acc = .ok out → ∃ rs, l.foldlM (sAcc g) (b, acc.2) = .ok (rs, out.2) := by
  intro l
  induction l with
  | nil => intro acc out b h; simp only [List.foldlM_nil] at h; cases h; exact ⟨b, rfl⟩
  | cons e l ih =>
    intro acc out b h
    rw [List.foldlM_cons] at h ⊢
    cases he : Shared.step g acc.2 e with
    | error err => rw [sAcc_error he] at h; cases h
    | ok p =>
      obtain ⟨r, s'⟩ := p
      rw [sAcc_ok he] at h
      obtain ⟨rs, hrs⟩ := ih (acc.1.push r, s') out (b.push r) h
      exact ⟨rs, by rw [sAcc_ok (acc := (b, acc.2)) he]; exact hrs⟩

theorem srunL_append {g : Nat} {s s' : State} {xs ys : List Event} {rs : Array Result}
    (h : srunL g s (xs ++ ys) = .ok (rs, s')) :
    ∃ rs₁ s₁ rs₂, srunL g s xs = .ok (rs₁, s₁) ∧ srunL g s₁ ys = .ok (rs₂, s') := by
  unfold srunL at h ⊢
  rw [List.foldlM_append] at h
  cases h₁ : xs.foldlM (sAcc g) (#[], s) with
  | error err => rw [h₁] at h; cases h
  | ok mid =>
    rw [h₁] at h
    obtain ⟨rs₂, h₂⟩ := sfold_state ys mid (rs, s') #[] h
    exact ⟨mid.1, mid.2, rs₂, rfl, h₂⟩

theorem srunL_snoc {g : Nat} {s s' : State} {xs : List Event} {e : Event} {rs : Array Result}
    (h : srunL g s (xs ++ [e]) = .ok (rs, s')) :
    ∃ rs₁ s₁ r, srunL g s xs = .ok (rs₁, s₁) ∧ Shared.step g s₁ e = .ok (r, s') := by
  obtain ⟨rs₁, s₁, rs₂, h₁, h₂⟩ := srunL_append h
  refine ⟨rs₁, s₁, ?_⟩
  unfold srunL at h₂
  rw [List.foldlM_cons] at h₂
  cases he : Shared.step g s₁ e with
  | error err => rw [sAcc_error (acc := (#[], s₁)) he] at h₂; cases h₂
  | ok p =>
    obtain ⟨r, s₂⟩ := p
    rw [sAcc_ok (acc := (#[], s₁)) he] at h₂
    have h₃ : (Except.ok (#[].push r, s₂) : Except String (Array Result × State)) = .ok (rs₂, s') := h₂
    injection h₃ with h₃
    injection h₃ with _ h₃
    exact ⟨r, h₁, by rw [h₃]⟩

theorem srunL_det {g : Nat} {s : State} {l : List Event} {a b : Array Result × State}
    (h₁ : srunL g s l = .ok a) (h₂ : srunL g s l = .ok b) : a = b := by
  rw [h₁] at h₂; injection h₂

theorem srunL_prefix {g : Nat} {s s' : State} {l : List Event} {rs : Array Result}
    (h : srunL g s l = .ok (rs, s')) (k : Nat) : ∃ rs₁ s₁, srunL g s (l.take k) = .ok (rs₁, s₁) := by
  rw [← List.take_append_drop k l] at h
  obtain ⟨rs₁, s₁, -, h₁, -⟩ := srunL_append h
  exact ⟨rs₁, s₁, h₁⟩

theorem srunL_drawn {g : Nat} (hg : 2 ≤ g) {s s' : State} {l : List Event} {rs : Array Result} (hi : Inv s)
    (h : srunL g s l = .ok (rs, s')) {t : Nat} (hd : (drawnOf s t).isSome) : (drawnOf s' t).isSome :=
  (sfold_inv g (fun x => Inv x ∧ (dr x.windows t).isSome)
    (fun _ _ _ _ hx he => ⟨(sstep hg hx.1 he).1, (sstep hg hx.1 he).2.2.2.1 _ hx.2⟩) l (#[], s) (rs, s')
    ⟨hi, hd⟩ h).2

theorem srun_extract (g : Nat) (s : State) (a : Array Event) (k : Nat) :
    Shared.run g s (a.extract 0 k) = srunL g s (a.toList.take k) := by
  rw [srun_srunL, extract_toList]

theorem srun_toArray (g : Nat) (s : State) (l : List Event) : Shared.run g s l.toArray = srunL g s l := by
  rw [srun_srunL, List.toList_toArray]

/-- A state a run from `State.init` reaches keeps the invariant, the parameters, and `DP`. -/
theorem sreach {g : Nat} (hg : 2 ≤ g) {params : Params} {reg : Registration} {coins appendCoins : Array ByteArray}
    {notBeforeNs : Nat} {l : List Event} {rs : Array Result} {s : State}
    (h : srunL g (State.init params reg coins appendCoins notBeforeNs) l = .ok (rs, s)) :
    Inv s ∧ s.params = params ∧ DP g params.windows s.windows := by
  unfold srunL at h
  refine sfold_inv g (fun x => Inv x ∧ x.params = params ∧ DP g params.windows x.windows)
    (fun _ _ _ _ hx he => ?_) l (#[], State.init params reg coins appendCoins notBeforeNs) (rs, s) ?_ h
  · obtain ⟨hx, hpx, hD⟩ := hx
    obtain ⟨hx', ⟨hp', -, -⟩, -, -, hD'⟩ := sstep hg hx he
    refine ⟨hx', hp'.trans hpx, ?_⟩
    rw [← hpx] at hD ⊢; exact hD' hD
  · exact ⟨inv_init .., rfl, fun u h => by simp [dr, State.init] at h, fun u j h => by simp [dr, State.init] at h⟩

/-- Under `DP`, an undrawn window below `W` leaves its draw's first window undrawn. -/
theorem undrawn_first {g W : Nat} (hg : 0 < g) {ws : Array Win} (hD : DP g W ws) {t : Nat} (ht : t < W)
    (h : dr ws t = none) : dr ws (t / g * g) = none := by
  cases e : dr ws (t / g * g) with
  | none => rfl
  | some _ =>
    have := hD.2 (t / g * g) t (by rw [e]; rfl) (by rw [Nat.mul_div_cancel _ hg]; exact Nat.le_refl _) ht
    rw [h] at this; cases this

theorem sameW_finalize (first m : Nat) (ws : Array Win) (v : Nat) :
    SameW ws[v]! (finalize ((List.range m).map (first + ·)) ws)[v]! := by
  rw [finalize_range]; split <;> exact ⟨rfl, rfl, rfl, rfl⟩

/-- `Shared.close` keeps the record of each window it draws, and the appends. -/
theorem sclose_keeps {g : Nat} {s : State} {t' : Int} {now : Nat} {r : Result} {s₁ : State}
    (he : Shared.close g s t' now = .ok (r, s₁)) {t : Nat} (h0 : dr s.windows t = none)
    (h1 : (dr s₁.windows t).isSome) : s₁.record t = s.record t ∧ s₁.appends = s.appends := by
  rcases sclose_spec he with ⟨-, rfl⟩ | ⟨u, f, -, rfl⟩ | ⟨d, -, ht0, hts, hund, hm, hpre, hfew, hdraws, rfl⟩
  · rw [h0] at h1; cases h1
  · exfalso
    rw [dr_fail] at h1
    unfold grp at h1
    dsimp only at h1
    rw [dr_finalize, h0] at h1; cases h1
  · refine ⟨?_, rfl⟩
    apply record_of
    · rfl
    intro v
    show SameW s.windows[v]! ((grp g t'.toNat).foldl (drawStep d now) (finalize (grp g t'.toNat) s.windows, 0)).1[v]!
    unfold grp
    refine sameW_trans (sameW_finalize (t'.toNat / g * g) (t'.toNat + 1 - t'.toNat / g * g) s.windows v) ?_
    obtain ⟨-, -, h3⟩ := drawFold_range d now (t'.toNat / g * g) 0
      (finalize ((List.range (t'.toNat + 1 - t'.toNat / g * g)).map (t'.toNat / g * g + ·)) s.windows)
      (t'.toNat + 1 - t'.toNat / g * g)
    rw [h3 v]
    split <;> exact ⟨rfl, rfl, rfl, rfl⟩

/-- The event that draws window `t` is the close of its draw, so it keeps the window's record and the appends. -/
theorem sdraw_keeps {g : Nat} (hg : 2 ≤ g) {s : State} {e : Event} {r : Result} {s₁ : State}
    (he : Shared.step g s e = .ok (r, s₁)) {t : Nat} (h0 : drawnOf s t = none) (h1 : (drawnOf s₁ t).isSome) :
    s₁.record t = s.record t ∧ s₁.appends = s.appends := by
  rcases sstep_of hg he with ⟨t', now, rfl, hc⟩ | ⟨hn, hs⟩
  · exact sclose_keeps hc h0 h1
  · exact (no_draw (keep_nonclose hn hs) h0 h1).elim

/-- Up to the draw of window `t` on `coins`, the events on `coins'` are the same. -/
theorem stake_agree {g : Nat} (hg : 2 ≤ g) (params : Params) (reg : Registration) (appendCoins : Array ByteArray)
    (notBeforeNs : Nat) (D : Array ByteArray → Array Event) (hD : ReleaseOrder g params reg appendCoins notBeforeNs D)
    {coins coins' : Array ByteArray} {t : Nat} (ht : t < params.windows)
    (hag : ∀ j, j < params.windows + t / g → coins'[j]! = coins[j]!) :
    ∀ (k : Nat) (rs : Array Result) (s : State),
      srunL g (State.init params reg coins appendCoins notBeforeNs) ((D coins).toList.take k) = .ok (rs, s) →
      drawnOf s t = none → (D coins').toList.take k = (D coins).toList.take k := by
  intro k
  induction k with
  | zero => intros; simp
  | succ k ih =>
    intro rs s h hnd
    rw [List.take_add_one] at h
    obtain ⟨rs₁, s₁, rs₂, h₁, h₂⟩ := srunL_append h
    have hR := sreach hg h₁
    have hnd₁ : drawnOf s₁ t = none := by
      cases hx : drawnOf s₁ t with
      | none => rfl
      | some d =>
        have := srunL_drawn hg hR.1 h₂ (t := t) (by rw [hx]; rfl)
        rw [hnd] at this; cases this
    have e₂ := hD coins coins' (t / g) k rs₁ s₁ hag (by rw [srun_extract]; exact h₁)
      (undrawn_first (by omega) hR.2.2 ht hnd₁)
    rw [List.take_add_one, List.take_add_one, ih rs₁ s₁ h₁ hnd₁, Array.getElem?_toList, Array.getElem?_toList,
      e₂]

/-- `Live` at `g ≥ 1` windows a draw, from `OpeningsUnreadUntilDrawn`, `RecordKeptAfterDraw` and the release order. -/
theorem liveUnder (g : Nat) (params : Params) (reg : Registration) (appendCoins : Array ByteArray)
    (notBeforeNs : Nat) (D : Array ByteArray → Array Event) (hg : 1 ≤ g) (hW : 1 ≤ params.windows)
    (hm : 1 ≤ params.minPasses) (hD : ReleaseOrder g params reg appendCoins notBeforeNs D) :
    LiveUnder g params reg appendCoins notBeforeNs D := by
  rcases Nat.lt_or_ge g 2 with hg1 | hg2
  · obtain rfl : g = 1 := by omega
    exact (liveUnder_one params reg appendCoins notBeforeNs D).2
      (Properties.RemoteAudit.liveUnder params reg appendCoins notBeforeNs D hW hm
        ((releaseOrder_one params reg appendCoins notBeforeNs D).1 hD))
  intro coins coins' c k₁ k₂ rs₁ rs₂ s₁ s₂ hag h₁ h₂ t htc hd₁ hd₂
  subst htc
  rw [srun_extract] at h₁ h₂
  have hag' : ∀ j, j < params.windows + t / g → coins[j]! = coins'[j]! := fun j hj => (hag j hj).symm
  have ht : t < params.windows := (sreach hg2 h₁).2.2.1 t hd₁
  have T2 := openingsUnreadUntilDrawn
  have T3 := recordKeptAfterDraw
  obtain ⟨m, hk₁, ⟨rsA, sA, hA, hdA⟩, hmin'⟩ := exists_first (Q := fun k => ∃ rs s,
      srunL g (State.init params reg coins appendCoins notBeforeNs) ((D coins).toList.take k) = .ok (rs, s) ∧
        (drawnOf s t).isSome)
    (by
      rintro ⟨rs0, s0, h0, hd0⟩
      rw [List.take_zero] at h0
      have e := srunL_det h0 rfl
      injection e with _ e
      subst e
      simp [drawnOf, State.init] at hd0)
    k₁ ⟨k₁, Nat.le_refl _, rs₁, s₁, h₁, hd₁⟩
  have hmin : ∀ k, k ≤ m → ∀ rs s,
      srunL g (State.init params reg coins appendCoins notBeforeNs) ((D coins).toList.take k) = .ok (rs, s) →
      drawnOf s t = none := by
    intro k hk rs s h
    cases hx : drawnOf s t with
    | none => rfl
    | some d => exact (hmin' k hk ⟨rs, s, h, by rw [hx]; rfl⟩).elim
  have hmL : m < (D coins).toList.length := by
    rcases Nat.lt_or_ge m (D coins).toList.length with hc | hc
    · exact hc
    have e : (D coins).toList.take (m + 1) = (D coins).toList.take m := by
      rw [List.take_of_length_le (by omega), List.take_of_length_le (by omega)]
    rw [e] at hA
    rw [hmin m (Nat.le_refl _) _ _ hA] at hdA
    cases hdA
  obtain ⟨e, he⟩ : ∃ e, (D coins).toList[m]? = some e := ⟨_, List.getElem?_eq_getElem hmL⟩
  have htake : (D coins).toList.take (m + 1) = (D coins).toList.take m ++ [e] := by
    rw [List.take_add_one, he]; rfl
  rw [htake] at hA
  obtain ⟨rsm, sm, r, hsm, hstep⟩ := srunL_snoc hA
  have hndm : drawnOf sm t = none := hmin m (Nat.le_refl _) _ _ hsm
  have hRm := sreach hg2 hsm
  have hL'm := stake_agree hg2 params reg appendCoins notBeforeNs D hD ht hag m rsm sm hsm hndm
  have hL'e : (D coins').toList[m]? = (D coins).toList[m]? := by
    rw [Array.getElem?_toList, Array.getElem?_toList]
    exact hD coins coins' (t / g) m rsm sm hag (by rw [srun_extract]; exact hsm)
      (undrawn_first (by omega) hRm.2.2 ht hndm)
  have htake' : (D coins').toList.take (m + 1) = (D coins).toList.take m ++ [e] := by
    rw [List.take_add_one, hL'm, hL'e, he]; rfl
  have hT2 : ∀ k, k ≤ m → ∀ rs s,
      srunL g (State.init params reg coins appendCoins notBeforeNs) ((D coins).toList.take k) = .ok (rs, s) →
      srunL g (State.init params reg coins' appendCoins notBeforeNs) ((D coins).toList.take k) =
        .ok (rs, { s with coins := coins' }) := by
    intro k hk rs s h
    have := T2 g params reg coins appendCoins notBeforeNs ((D coins).toList.take k).toArray rs s
      (by rw [srun_toArray]; exact h) (t / g) coins' hW
      (undrawn_first (by omega) (sreach hg2 h).2.2 ht (hmin k hk rs s h)) hag
    rwa [srun_toArray] at this
  have hk₂ : m + 1 ≤ k₂ := by
    rcases Nat.lt_or_ge m k₂ with hc | hle
    · exact hc
    obtain ⟨rsk, sk, hsk⟩ := srunL_prefix hsm k₂
    rw [List.take_take, Nat.min_eq_left hle] at hsk
    have hndk := hmin k₂ hle _ _ hsk
    rw [stake_agree hg2 params reg appendCoins notBeforeNs D hD ht hag k₂ rsk sk hsk hndk] at h₂
    have e := srunL_det h₂ (hT2 k₂ hle rsk sk hsk)
    injection e with _ e
    rw [e] at hd₂
    rw [show drawnOf { sk with coins := coins' } t = drawnOf sk t from rfl, hndk] at hd₂
    cases hd₂
  rw [take_split _ hk₂, htake'] at h₂
  obtain ⟨rsB, sB, rsB₂, hB, hB₂⟩ := srunL_append h₂
  obtain ⟨rsy, y, r', hy, hstep'⟩ := srunL_snoc hB
  have hy' := srunL_det hy (hT2 m (Nat.le_refl _) rsm sm hsm)
  injection hy' with _ hy'
  subst hy'
  have hdB : (drawnOf sB t).isSome := by
    cases hx : drawnOf sB t with
    | some d => rfl
    | none =>
      exfalso
      have := T2 g params reg coins' appendCoins notBeforeNs ((D coins).toList.take m ++ [e]).toArray rsB sB
        (by rw [srun_toArray]; exact hB) (t / g) coins hW (undrawn_first (by omega) (sreach hg2 hB).2.2 ht hx) hag'
      rw [srun_toArray] at this
      have e := srunL_det hA this
      injection e with _ e
      rw [e, show drawnOf { sB with coins := coins } t = drawnOf sB t from rfl, hx] at hdA
      cases hdA
  obtain ⟨hrA, haA⟩ := sdraw_keeps hg2 hstep hndm hdA
  obtain ⟨hrB, haB⟩ := sdraw_keeps hg2 hstep' (s := { sm with coins := coins' }) hndm hdB
  rw [take_split _ hk₁, htake] at h₁
  obtain ⟨rsA', sA', rsA₂, hA', hA₂⟩ := srunL_append h₁
  have eA := srunL_det hA' hA
  injection eA with _ eA
  subst eA
  obtain ⟨dA, hdA'⟩ := Option.isSome_iff_exists.1 hdA
  obtain ⟨dB, hdB'⟩ := Option.isSome_iff_exists.1 hdB
  obtain ⟨-, r₁, o₁⟩ := T3 g params reg coins appendCoins notBeforeNs ((D coins).toList.take m ++ [e]).toArray
    (((D coins).toList.take k₁).drop (m + 1)).toArray rsA' rsA₂ sA' s₁ (by rw [srun_toArray]; exact hA')
    (by rw [srun_toArray]; exact hA₂) t dA hm hdA'
  obtain ⟨-, r₂, o₂⟩ := T3 g params reg coins' appendCoins notBeforeNs ((D coins).toList.take m ++ [e]).toArray
    (((D coins').toList.take k₂).drop (m + 1)).toArray rsB rsB₂ sB s₂ (by rw [srun_toArray]; exact hB)
    (by rw [srun_toArray]; exact hB₂) t dB hm hdB'
  have hrec : s₂.record t = s₁.record t := by rw [r₂, r₁, hrA, hrB]; rfl
  refine ⟨hrec, fun u => ?_⟩
  rw [o₂ u, o₁ u, hrB, haB, hrA, haA]
  exact Iff.rfl

end Properties.RemoteAudit.Shared
