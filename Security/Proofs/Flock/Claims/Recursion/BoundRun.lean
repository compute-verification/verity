import Proofs.Flock.Claims.Recursion.BoundDeploy

/-!
# `IVStar`'s runs

`Claims.Recursion.Bound.IVStar` answers every round before its last with that round's coin and decides at its last, so
its run on replayed messages has a closed form (`Bound.run_ivstar`): the messages before the last, each answered by its
coin, then the last message answered by the decision on all of them, whose oracle calls are the run's. So:

* it is public-coin, its answers read by any `decC` that inverts `ans` (`Bound.publicCoin`);
* two message lists that agree before the last round and whose last messages `final` reads alike give one decision,
  one log, and views that differ only in the last message (`Bound.run_ivstar_congr`);
* the last message's representative (`Bound.rep`, one message per reading of `final`) keeps the decision
  (`Bound.final_rep`), and two last messages `final` reads alike have one representative (`Bound.rep_eq`).
-/

namespace Claims.Recursion

open Claims

theorem eval_bindOC (H : Msg → Msg) {α β : Type} (p : OC α) (f : α → OC β) :
    (OC.eval H (bindOC p f)).1 = (OC.eval H (f (OC.eval H p).1)).1 := by
  induction p with
  | ret a => rfl
  | query q k ih =>
    simp only [bindOC, OC.eval]
    exact ih (H q)

section Bound

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]

theorem Bound.eval_ret_ne_msg (H : Msg → Msg) (b : Bool) (m : Msg) :
    (OC.eval H (OC.ret (if b then Out.acc else Out.rej))).1 ≠ .msg m := by
  cases b <;> simp [OC.eval]

/-- **`IVStar` decides at its last round**: from its last round on, its answer is never a message. -/
theorem Bound.ivstar_eval_ne_msg (Bd : Bound Coin Col T) (H : Msg → Msg) (x : Msg) (as : List Msg) (cs : List Coin)
    (h : ¬ as.length < Bd.DE.V.k x) (m : Msg) : ((Bd.IVStar.next x as cs).eval H).1 ≠ .msg m := by
  simp only [Bound.IVStar, h, ↓reduceIte]
  split
  · simp [OC.eval]
  · split
    · rw [eval_bindOC]
      split
      · rw [eval_bindOC]
        exact Bound.eval_ret_ne_msg H _ m
      · simp [OC.eval]
    · simp [OC.eval]

/-- **`IVStar` is public-coin**: before its last round it answers a round with its coin's `ans`, which any `decC`
that inverts `ans` reads back. -/
theorem Bound.publicCoin (Bd : Bound Coin Col T) (H : Msg → Msg) (decI : List (ℕ × Col) → List T)
    (decC : Msg → Msg → Coin) (hdec : ∀ x c, decC x (Bd.ans c) = c) :
    PublicCoin H (Inner.ofFlock Bd.deploy decI) decC := by
  intro x as cs c m _ hm
  change ((Bd.IVStar.next x as (cs ++ [c])).eval H).1 = .msg m at hm
  by_cases h : as.length < Bd.DE.V.k x
  · have e : ((Bd.IVStar.next x as (cs ++ [c])).eval H).1 = .msg (Bd.ans c) := by
      simp [Bound.IVStar, h, OC.eval]
    rw [e] at hm
    injection hm with hm
    rw [← hm]
    exact hdec x c
  · exact absurd hm (Bd.ivstar_eval_ne_msg H x as _ h m)

/-! ## The run in closed form -/

/-- `IVStar`'s steps at the coins `r`. -/
def Bound.vnAt (Bd : Bound Coin Col T) (x : Msg) (r : Fin (Bd.IVStar.k x) → Coin) : List Msg → OC Out :=
  fun as => Bd.IVStar.next x as ((List.range as.length).map (coinSeq r))

/-- `IVStar`'s decision on the first `k` replayed messages at the coins `r`. -/
def Bound.decAt (Bd : Bound Coin Col T) (H : Msg → Msg) (x : Msg) (ms : List Msg)
    (r : Fin (Bd.IVStar.k x) → Coin) : Out × List (Msg × Msg) :=
  (Bd.IVStar.next x ((List.range (Bd.DE.V.k x)).map fun i => ms.getD i [])
    ((List.range (Bd.DE.V.k x)).map (coinSeq r))).eval H

theorem Bound.decAt_ne_msg (Bd : Bound Coin Col T) (H : Msg → Msg) (x : Msg) (ms : List Msg)
    (r : Fin (Bd.IVStar.k x) → Coin) (m : Msg) : (Bd.decAt H x ms r).1 ≠ .msg m :=
  Bd.ivstar_eval_ne_msg H x _ _ (by simp) m

theorem Bound.stepWith_ivstar (Bd : Bound Coin Col T) (H : Msg → Msg) (x : Msg) (ms : List Msg)
    (r : Fin (Bd.IVStar.k x) → Coin) :
    ∀ (n : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t : Trace), hist.length = t.msgs.length →
      as.length = t.msgs.length → t.msgs.length + (n + 1) = Bd.DE.V.k x →
      stepWith H (replay ms) x () (Bd.vnAt x r) (n + 1) hist as t =
        { msgs := t.msgs ++ (List.range' t.msgs.length n).map (fun i => (ms.getD i [], Out.msg (Bd.ans (coinSeq r i)))) ++
            [(ms.getD (Bd.DE.V.k x - 1) [], ((Bd.vnAt x r (as ++
              (List.range' t.msgs.length (n + 1)).map fun i => ms.getD i [])).eval H).1)],
          pLog := t.pLog,
          vLog := t.vLog ++ ((Bd.vnAt x r (as ++ (List.range' t.msgs.length (n + 1)).map fun i => ms.getD i [])).eval H).2,
          out := ((Bd.vnAt x r (as ++ (List.range' t.msgs.length (n + 1)).map fun i => ms.getD i [])).eval H).1 } := by
  intro n
  induction n with
  | zero =>
    intro hist as t hh ha hk
    have hp : ((replay ms).next x hist ()).eval H = (ms.getD t.msgs.length [], []) := by
      show (OC.ret (ms.getD hist.length [])).eval H = _
      rw [hh]
      rfl
    have hne : ∀ m, ((Bd.vnAt x r (as ++ [ms.getD t.msgs.length []])).eval H).1 ≠ .msg m :=
      Bd.ivstar_eval_ne_msg H x _ _ (by simp only [List.length_append, List.length_singleton]; omega)
    have hk1 : Bd.DE.V.k x - 1 = t.msgs.length := by omega
    simp only [stepWith, hp, hk1, Nat.zero_add, List.range'_zero, List.map_nil, List.append_nil, List.range'_one,
      List.map_cons]
    rcases hv : (Bd.vnAt x r (as ++ [ms.getD t.msgs.length []])).eval H with ⟨o, l⟩
    rw [hv] at hne
    rcases o with m | _ | _
    · exact absurd rfl (hne m)
    · rfl
    · rfl
  | succ n ih =>
    intro hist as t hh ha hk
    have hp : ((replay ms).next x hist ()).eval H = (ms.getD t.msgs.length [], []) := by
      show (OC.ret (ms.getD hist.length [])).eval H = _
      rw [hh]
      rfl
    have hv : (Bd.vnAt x r (as ++ [ms.getD t.msgs.length []])).eval H =
        (.msg (Bd.ans (coinSeq r t.msgs.length)), []) := by
      have hlt : t.msgs.length + 1 < Bd.DE.V.k x := by omega
      simp only [Bound.vnAt, Bound.IVStar, List.length_append, List.length_singleton, hlt, ↓reduceIte,
        List.range_succ, List.map_append, List.map_singleton, List.getLastD_concat, ha]
      rfl
    have hv' : ((Bd.vnAt x r (as ++ [(((replay ms).next x hist ()).eval H).1])).eval H).1 =
        .msg (Bd.ans (coinSeq r t.msgs.length)) := by
      rw [hp]
      exact congrArg Prod.fst hv
    rw [stepWith_msg H (replay ms) x () (Bd.vnAt x r) (n + 1) hist as t hv']
    simp only [hp, hv]
    refine (ih _ _ _ ?_ ?_ ?_).trans ?_
    · simp only [List.length_append, List.length_singleton, hh]
    · simp only [List.length_append, List.length_singleton, ha]
    · simp only [List.length_append, List.length_singleton]
      omega
    · simp only [List.length_append, List.length_singleton, List.append_nil, List.range'_succ, List.map_cons,
        List.append_assoc, List.singleton_append, List.cons_append, List.nil_append]

/-- **`IVStar`'s run on replayed messages, in closed form**: at `k ≥ 1` rounds, each message before the last answered
by its coin, then the last answered by the decision on all `k` (`decAt`), whose calls are the run's. -/
theorem Bound.run_ivstar (Bd : Bound Coin Col T) (H : Msg → Msg) (x : Msg) (ms : List Msg)
    (r : Fin (Bd.IVStar.k x) → Coin) (hk : 0 < Bd.DE.V.k x) :
    run H (replay ms) Bd.IVStar x () r =
      { msgs := (List.range (Bd.DE.V.k x - 1)).map (fun i => (ms.getD i [], Out.msg (Bd.ans (coinSeq r i)))) ++
          [(ms.getD (Bd.DE.V.k x - 1) [], (Bd.decAt H x ms r).1)],
        pLog := [], vLog := (Bd.decAt H x ms r).2, out := (Bd.decAt H x ms r).1 } := by
  obtain ⟨n, hn⟩ : ∃ n, Bd.DE.V.k x = n + 1 := ⟨Bd.DE.V.k x - 1, by omega⟩
  have h := Bd.stepWith_ivstar H x ms r n [] [] ⟨[], [], [], .rej⟩ rfl rfl (by simp [hn])
  have e : run H (replay ms) Bd.IVStar x () r =
      stepWith H (replay ms) x () (Bd.vnAt x r) (n + 1) [] [] ⟨[], [], [], .rej⟩ := by
    show stepWith H (replay ms) x () (Bd.vnAt x r) (Bd.DE.V.k x) [] [] ⟨[], [], [], .rej⟩ = _
    rw [hn]
  rw [e, h]
  simp only [List.length_nil, List.nil_append, ← List.range_eq_range', Bound.decAt, Bound.vnAt, List.length_map,
    List.length_range, hn, Nat.add_sub_cancel]

/-- With no rounds, the run rejects with no message. -/
theorem Bound.run_ivstar_zero (Bd : Bound Coin Col T) (H : Msg → Msg) (x : Msg) (ms : List Msg)
    (r : Fin (Bd.IVStar.k x) → Coin) (hk : Bd.DE.V.k x = 0) :
    run H (replay ms) Bd.IVStar x () r = ⟨[], [], [], .rej⟩ := by
  show stepWith H (replay ms) x () (Bd.vnAt x r) (Bd.DE.V.k x) [] [] ⟨[], [], [], .rej⟩ = _
  rw [hk]
  rfl

/-- The run plays all its rounds. -/
theorem Bound.run_ivstar_length (Bd : Bound Coin Col T) (H : Msg → Msg) (x : Msg) (ms : List Msg)
    (r : Fin (Bd.IVStar.k x) → Coin) : (run H (replay ms) Bd.IVStar x () r).msgs.length = Bd.DE.V.k x := by
  rcases Nat.eq_zero_or_pos (Bd.DE.V.k x) with hk | hk
  · rw [Bd.run_ivstar_zero H x ms r hk, hk]
    rfl
  · rw [Bd.run_ivstar H x ms r hk]
    simp only [List.length_append, List.length_map, List.length_range, List.length_singleton]
    omega

/-- The run's messages are the replayed ones, at each of its rounds. -/
theorem Bound.run_ivstar_fst (Bd : Bound Coin Col T) (H : Msg → Msg) (x : Msg) (ms : List Msg)
    (r : Fin (Bd.IVStar.k x) → Coin) (i : ℕ) :
    ((run H (replay ms) Bd.IVStar x () r).msgs.map Prod.fst).getD i [] =
      if i < Bd.DE.V.k x then ms.getD i [] else [] := by
  split_ifs with hi
  · exact VStar.run_replay_fst H Bd.IVStar x ms r (by rw [Bd.run_ivstar_length]; exact hi)
  · exact List.getD_eq_default _ _ (by simp [Bd.run_ivstar_length]; omega)

/-- **Two message lists that agree before the last round, their last messages read alike by `final`, give one run up to
the last message**: one decision and one log, the same messages before the last and the same answers. -/
theorem Bound.run_ivstar_congr (Bd : Bound Coin Col T) (H : Msg → Msg) (x : Msg) (ms ms' : List Msg)
    (r : Fin (Bd.IVStar.k x) → Coin) (hm : ∀ i, i + 1 < Bd.DE.V.k x → ms.getD i [] = ms'.getD i [])
    (hf : Bd.final (ms.getD (Bd.DE.V.k x - 1) []) = Bd.final (ms'.getD (Bd.DE.V.k x - 1) [])) :
    Bd.decAt H x ms r = Bd.decAt H x ms' r := by
  rcases Nat.eq_zero_or_pos (Bd.DE.V.k x) with hk | hk
  · unfold Bound.decAt
    simp only [hk, List.range_zero, List.map_nil]
  obtain ⟨n, hn⟩ : ∃ n, Bd.DE.V.k x = n + 1 := ⟨Bd.DE.V.k x - 1, by omega⟩
  unfold Bound.decAt
  have e : ∀ l : List Msg, (List.range (Bd.DE.V.k x)).map (fun i => l.getD i []) =
      (List.range n).map (fun i => l.getD i []) ++ [l.getD n []] := by
    intro l
    rw [hn, List.range_succ, List.map_append, List.map_singleton]
  have hpre : (List.range n).map (fun i => ms.getD i []) = (List.range n).map (fun i => ms'.getD i []) :=
    List.map_congr_left fun i hi => hm i (by rw [List.mem_range] at hi; omega)
  rw [e ms, e ms', hpre]
  rw [show n = Bd.DE.V.k x - 1 by omega] at hpre ⊢
  rw [Bd.ivstar_congr_final x _ _ _ _ hf]

/-! ## The last message's representative -/

open Classical in
/-- **The last message's representative**: one message for each reading of `final`, and the message itself where
`final` reads nothing. -/
noncomputable def Bound.rep (Bd : Bound Coin Col T) (a : Msg) : Msg :=
  if (Bd.final a).isSome then Classical.epsilon fun a' => Bd.final a' = Bd.final a else a

theorem Bound.final_rep (Bd : Bound Coin Col T) (a : Msg) : Bd.final (Bd.rep a) = Bd.final a := by
  unfold Bound.rep
  split_ifs
  · exact Classical.epsilon_spec (p := fun a' => Bd.final a' = Bd.final a) ⟨a, rfl⟩
  · rfl

theorem Bound.rep_eq (Bd : Bound Coin Col T) {a a' : Msg} (h : Bd.final a = Bd.final a')
    (hs : (Bd.final a).isSome) : Bd.rep a = Bd.rep a' := by
  unfold Bound.rep
  rw [ite_eq_left hs, ite_eq_left (h ▸ hs), h]

theorem Bound.rep_of_none (Bd : Bound Coin Col T) {a : Msg} (h : Bd.final a = none) : Bd.rep a = a := by
  unfold Bound.rep
  rw [h]
  rfl

/-- **`IVStar` rejects a last message `final` reads nothing of.** -/
theorem Bound.decAt_none (Bd : Bound Coin Col T) (H : Msg → Msg) (x : Msg) (ms : List Msg)
    (r : Fin (Bd.IVStar.k x) → Coin) (hk : 0 < Bd.DE.V.k x) (h : Bd.final (ms.getD (Bd.DE.V.k x - 1) []) = none) :
    (Bd.decAt H x ms r).1 = .rej := by
  obtain ⟨n, hn⟩ : ∃ n, Bd.DE.V.k x = n + 1 := ⟨Bd.DE.V.k x - 1, by omega⟩
  have hl : ((List.range (Bd.DE.V.k x)).map fun i => ms.getD i []).getLastD [] = ms.getD (Bd.DE.V.k x - 1) [] := by
    rw [hn, List.range_succ, List.map_append, List.map_singleton, List.getLastD_concat, Nat.add_sub_cancel]
  unfold Bound.decAt
  simp only [Bound.IVStar, List.length_map, List.length_range, lt_irrefl, ↓reduceIte, hl, h, OC.eval]

end Bound

end Claims.Recursion
