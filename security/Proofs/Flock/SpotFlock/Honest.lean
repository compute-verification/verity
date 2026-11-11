module

public import Proofs.Flock.SamplingGeneric.Reg
public import Proofs.Flock.SpotFlock.Sessions

/-!
# SPOT[Flock] accepts where Flock accepts every template's session

`proverSeq m₀ m₁ m₂ P'`: rounds 0 and 1 send `m₀` and `m₁`, rounds 2 to `stage` send `[]` and round `stage + 1` sends
`m₂` (`preMsg`); a round of template `t`'s session plays `P' t` at the statement `m₂` states for it on the session's
history, until a round of the session is answered `[]`, Flock's acceptance as the run shows it, and then sends `[]`;
the verdict's round sends `[]`. `accepts_seq`: where the statement check passes `m₁` and `m₂` at some coins of round 0
and of the second stage's rounds, and Flock's masked verifier accepts each `P' t` at its statement on some coins,
SPOT[Flock] accepts `proverSeq m₀ m₁ m₂ P'` on a positive share of its coins.

The run's first `stage + 2` rounds are the messages `preMsg` names, each answered with a message (`stepWith_pre`,
`next_preMsg`, `next_preMsg_stmt`).

The run follows each session's Flock run round for round while Flock answers with messages (`slot_run`); Flock's
acceptance leaves an accepted prefix of the session's messages, and the session's remaining rounds answer `[]` to `[]`
(`slot_fill`); the verdict finds each session's accepted prefix.
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling

/-- **A history with a round answered `[]`.** -/
def marked (hist : List (Msg × Msg)) : Bool := hist.any fun e => decide (e.2 = [])

theorem take_map_range {α : Type} (f : ℕ → α) {j n : ℕ} (hj : j ≤ n) :
    ((List.range n).map f).take j = (List.range j).map f := by
  apply List.ext_getElem
  · simp; omega
  · intro i _ _
    simp

theorem drop_map_range {α : Type} (f : ℕ → α) (o l : ℕ) :
    ((List.range (o + l)).map f).drop o = (List.range l).map fun i => f (o + i) := by
  apply List.ext_getElem
  · simp
  · intro i _ _
    simp

theorem getD_append_left {l l' : List Msg} {n : ℕ} (hn : n < l.length) : (l ++ l').getD n [] = l.getD n [] := by
  simp [List.getD_eq_getElem?_getD, List.getElem?_append_left hn]

/-- **A run's first rounds, their messages known**: where the prover sends `a j` on a history of `j < n` rounds with no
`H` call, and the verifier answers `a 0, …, a j` with the message `b j` and no `H` call, the run's first `j ≤ n` rounds
leave a history of `j` rounds and the messages `a 0, …, a (j - 1)`. -/
theorem stepWith_pre {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω) (v : List Msg → OC Out)
    (a b : ℕ → Msg) (n : ℕ) (hP : ∀ hist : List (Msg × Msg), hist.length < n → P.next x hist ω = .ret (a hist.length))
    (hV : ∀ j < n, v ((List.range (j + 1)).map a) = .ret (.msg (b j))) :
    ∀ j ≤ n, ∃ (hist : List (Msg × Msg)) (tr : Transcript), hist.length = j ∧
      ∀ rest, stepWith H P x ω v (j + rest) [] [] ⟨[], [], [], .rej⟩ =
        stepWith H P x ω v rest hist ((List.range j).map a) tr := by
  intro j
  induction j with
  | zero => exact fun _ => ⟨[], ⟨[], [], [], .rej⟩, rfl, fun rest => by rw [Nat.zero_add]; rfl⟩
  | succ j ih =>
    intro hj
    obtain ⟨hist, tr, hl, heq⟩ := ih (by omega)
    have hpe : ((P.next x hist ω).eval H).1 = a j := by
      rw [hP hist (by omega), hl]
      rfl
    have hr : (List.range j).map a ++ [((P.next x hist ω).eval H).1] = (List.range (j + 1)).map a := by
      simp [hpe, List.range_succ]
    have hv : v ((List.range j).map a ++ [((P.next x hist ω).eval H).1]) = .ret (.msg (b j)) := by
      rw [hr]
      exact hV j (by omega)
    refine ⟨hist ++ [(((P.next x hist ω).eval H).1, b j)],
      ⟨tr.msgs ++ [(((P.next x hist ω).eval H).1, .msg (b j))], tr.pLog ++ ((P.next x hist ω).eval H).2, tr.vLog,
        tr.out⟩, by simp [hl], fun rest => ?_⟩
    rw [show j + 1 + rest = j + (rest + 1) by omega, heq (rest + 1),
      stepWith_msg H P x ω v rest hist _ tr (b j) hv, hr]

section Slot

variable {Ω Ω' : Type} (H : Msg → Msg) (P : Prover Ω) (P' : Prover Ω') (x xS : Msg) (ω : Ω) (ω' : Ω')
  (vS vF : List Msg → OC Out) (hist₀ : List (Msg × Msg)) (as₀ : List Msg) (K : ℕ)

/-- **A session's rounds after its acceptance**: where the prover sends `[]` once a round of the session was answered
`[]`, and the verifier answers `[]` once a shorter prefix of the session's messages was accepted, the session's
remaining rounds are answered with messages, and its messages keep the accepted prefix. -/
theorem slot_fill
    (hfP : ∀ hist : List (Msg × Msg), marked hist = true → hist.length < K → P.next x (hist₀ ++ hist) ω = .ret [])
    (hfV : ∀ bs : List Msg, bs.length ≤ K → (∃ j, 0 < j ∧ j < bs.length ∧ ((vF (bs.take j)).eval H).1 = .acc) →
      (vS (as₀ ++ bs)).eval H = (.msg [], [])) :
    ∀ (f : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t : Transcript), as.length + f = K → hist.length = as.length →
      marked hist = true → (∃ j, 0 < j ∧ j ≤ as.length ∧ ((vF (as.take j)).eval H).1 = .acc) →
      ∃ (hist' : List (Msg × Msg)) (as' : List Msg) (t' : Transcript), as'.length = K ∧ hist'.length = K ∧
        (∃ j, 0 < j ∧ j ≤ K ∧ ((vF (as'.take j)).eval H).1 = .acc) ∧
        ∀ rest, stepWith H P x ω vS (f + rest) (hist₀ ++ hist) (as₀ ++ as) t =
          stepWith H P x ω vS rest (hist₀ ++ hist') (as₀ ++ as') t' := by
  intro f
  induction f with
  | zero =>
    intro hist as t hK hl _ hj
    obtain ⟨j, hj0, hjl, hja⟩ := hj
    exact ⟨hist, as, t, by omega, by omega, ⟨j, hj0, by omega, hja⟩, fun rest => by rw [Nat.zero_add]⟩
  | succ f ih =>
    intro hist as t hK hl hm hj
    obtain ⟨j, hj0, hjl, hja⟩ := hj
    have hpa : (P.next x (hist₀ ++ hist) ω).eval H = ([], []) := by
      rw [hfP hist hm (by omega)]
      rfl
    have hvb : (vS (as₀ ++ (as ++ [[]]))).eval H = (.msg [], []) :=
      hfV (as ++ [[]]) (by simp; omega)
        ⟨j, hj0, by simp; omega, by rw [List.take_append_of_le_length hjl]; exact hja⟩
    have hm' : marked (hist ++ [([], [])]) = true := by
      simp only [marked, List.any_append, Bool.or_eq_true]
      exact Or.inl hm
    obtain ⟨hist', as', t', h1, h2, h3, h4⟩ := ih (hist ++ [([], [])]) (as ++ [[]])
      ⟨t.msgs ++ [([], .msg [])], t.pLog ++ [], t.vLog ++ [], t.out⟩ (by simp; omega) (by simp; omega) hm'
      ⟨j, hj0, by simp; omega, by rw [List.take_append_of_le_length hjl]; exact hja⟩
    refine ⟨hist', as', t', h1, h2, h3, fun rest => ?_⟩
    rw [show f + 1 + rest = (f + rest) + 1 by omega,
      stepWith_msg_of H P x ω vS (f + rest) (hist₀ ++ hist) (as₀ ++ as) t ([], []) hpa [] []
        (by rw [List.append_assoc]; exact hvb),
      List.append_assoc hist₀, List.append_assoc as₀]
    exact h4 rest

/-- **A session's rounds while Flock runs**: where the prover plays `P'` on the session's history until a round of the
session is answered `[]`, and the verifier answers as `vF` on the session's messages, its acceptance shown as `[]`,
wherever `vF` answered every shorter prefix with a message, a run of `vF` against `P'` that accepts leaves the session's
rounds all answered with messages and an accepted prefix of the session's messages. -/
theorem slot_run
    (hP : ∀ hist : List (Msg × Msg), marked hist = false → hist.length < K →
      P.next x (hist₀ ++ hist) ω = P'.next xS hist ω')
    (hv : ∀ bs : List Msg, bs ≠ [] → bs.length ≤ K →
      (∀ j, 0 < j → j < bs.length → ∃ m, ((vF (bs.take j)).eval H).1 = .msg m) →
      (vS (as₀ ++ bs)).eval H = (markOut [] ((vF bs).eval H).1, ((vF bs).eval H).2))
    (hne : ∀ bs, ((vF bs).eval H).1 ≠ .msg [])
    (hfP : ∀ hist : List (Msg × Msg), marked hist = true → hist.length < K → P.next x (hist₀ ++ hist) ω = .ret [])
    (hfV : ∀ bs : List Msg, bs.length ≤ K → (∃ j, 0 < j ∧ j < bs.length ∧ ((vF (bs.take j)).eval H).1 = .acc) →
      (vS (as₀ ++ bs)).eval H = (.msg [], [])) :
    ∀ (k : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (ts t : Transcript), as.length + k = K →
      hist.length = as.length → marked hist = false →
      (∀ j, 0 < j → j ≤ as.length → ∃ m, ((vF (as.take j)).eval H).1 = .msg m) →
      (stepWith H P' xS ω' vF k hist as ts).out = .acc →
      ∃ (hist' : List (Msg × Msg)) (as' : List Msg) (t' : Transcript), as'.length = K ∧ hist'.length = K ∧
        (∃ j, 0 < j ∧ j ≤ K ∧ ((vF (as'.take j)).eval H).1 = .acc) ∧
        ∀ rest, stepWith H P x ω vS (k + rest) (hist₀ ++ hist) (as₀ ++ as) t =
          stepWith H P x ω vS rest (hist₀ ++ hist') (as₀ ++ as') t' := by
  intro k
  induction k with
  | zero =>
    intro hist as ts t _ _ _ _ hacc
    simp [stepWith] at hacc
  | succ k ih =>
    intro hist as ts t hK hl hm hund hacc
    have hpa : (P.next x (hist₀ ++ hist) ω).eval H = (P'.next xS hist ω').eval H := by
      rw [hP hist hm (by omega)]
    have hund' : ∀ j, 0 < j → j < (as ++ [((P'.next xS hist ω').eval H).1]).length →
        ∃ m, ((vF ((as ++ [((P'.next xS hist ω').eval H).1]).take j)).eval H).1 = .msg m := by
      intro j hj0 hj
      simp only [List.length_append, List.length_singleton] at hj
      rw [List.take_append_of_le_length (by omega)]
      exact hund j hj0 (by omega)
    have hvS := hv (as ++ [((P'.next xS hist ω').eval H).1]) (by simp) (by simp; omega) hund'
    have hlast : (as ++ [((P'.next xS hist ω').eval H).1]).take (as.length + 1) =
        as ++ [((P'.next xS hist ω').eval H).1] := List.take_of_length_le (by simp)
    rcases hvb : (vF (as ++ [((P'.next xS hist ω').eval H).1])).eval H with ⟨o, l⟩
    rw [hvb] at hvS
    cases o with
    | msg m =>
      simp only [markOut] at hvS
      have hm0 : m ≠ [] := fun e => hne (as ++ [((P'.next xS hist ω').eval H).1]) (by rw [hvb, e])
      rw [stepWith_msg_of H P' xS ω' vF k hist as ts _ rfl m l hvb] at hacc
      have hmk : marked (hist ++ [(((P'.next xS hist ω').eval H).1, m)]) = false := by
        simp only [marked, List.any_append, List.any_cons, List.any_nil, Bool.or_false, Bool.or_eq_false_iff,
          decide_eq_false_iff_not]
        exact ⟨by simpa [marked] using hm, hm0⟩
      have hund'' : ∀ j, 0 < j → j ≤ (as ++ [((P'.next xS hist ω').eval H).1]).length →
          ∃ m', ((vF ((as ++ [((P'.next xS hist ω').eval H).1]).take j)).eval H).1 = .msg m' := by
        intro j hj0 hj
        simp only [List.length_append, List.length_singleton] at hj
        rcases Nat.lt_or_ge j (as.length + 1) with hj' | hj'
        · exact hund' j hj0 (by simp; omega)
        · rw [show j = as.length + 1 by omega, hlast, hvb]
          exact ⟨m, rfl⟩
      obtain ⟨hist', as', t', h1, h2, h3, h4⟩ := ih (hist ++ [(((P'.next xS hist ω').eval H).1, m)])
        (as ++ [((P'.next xS hist ω').eval H).1]) _
        ⟨t.msgs ++ [(((P'.next xS hist ω').eval H).1, .msg m)], t.pLog ++ ((P'.next xS hist ω').eval H).2,
          t.vLog ++ l, t.out⟩ (by simp; omega) (by simp; omega) hmk hund'' hacc
      refine ⟨hist', as', t', h1, h2, h3, fun rest => ?_⟩
      rw [show k + 1 + rest = (k + rest) + 1 by omega,
        stepWith_msg_of H P x ω vS (k + rest) (hist₀ ++ hist) (as₀ ++ as) t _ hpa m l
          (by rw [List.append_assoc]; exact hvS),
        List.append_assoc hist₀, List.append_assoc as₀]
      exact h4 rest
    | acc =>
      simp only [markOut] at hvS
      have hmk : marked (hist ++ [(((P'.next xS hist ω').eval H).1, [])]) = true := by
        simp [marked]
      obtain ⟨hist', as', t', h1, h2, h3, h4⟩ := slot_fill H P x ω vS vF hist₀ as₀ K hfP hfV k
        (hist ++ [(((P'.next xS hist ω').eval H).1, [])]) (as ++ [((P'.next xS hist ω').eval H).1])
        ⟨t.msgs ++ [(((P'.next xS hist ω').eval H).1, .msg [])], t.pLog ++ ((P'.next xS hist ω').eval H).2,
          t.vLog ++ l, t.out⟩ (by simp; omega) (by simp; omega) hmk
        ⟨as.length + 1, by omega, by simp, by rw [hlast, hvb]⟩
      refine ⟨hist', as', t', h1, h2, h3, fun rest => ?_⟩
      rw [show k + 1 + rest = (k + rest) + 1 by omega,
        stepWith_msg_of H P x ω vS (k + rest) (hist₀ ++ hist) (as₀ ++ as) t _ hpa [] l
          (by rw [List.append_assoc]; exact hvS),
        List.append_assoc hist₀, List.append_assoc as₀]
      exact h4 rest
    | rej =>
      rw [stepWith_rej_of H P' xS ω' vF k hist as ts _ rfl l hvb] at hacc
      simp at hacc

end Slot

variable (h : Held)

/-- **Where Flock answers no message, it has decided.** -/
theorem decidedAt_of_ne_msg {xS : Msg} {as : List Msg} {cs : List FlockDeploy.Coin} {j : ℕ} {H : Msg → Msg}
    (hn : ∀ m, ((FlockDeploy.next true xS (as.take j) (cs.take j)).eval H).1 ≠ .msg m) :
    decidedAt xS as cs j = true := by
  unfold decidedAt
  unfold FlockDeploy.next at hn
  split
  · rename_i c hc
    rw [hc] at hn
    exact absurd rfl (hn c.bytes)
  · rfl

/-- **A decision reads only its prefix**: messages appended after it don't change it. -/
theorem decidedAt_append {xS : Msg} {as e : List Msg} {cs : List FlockDeploy.Coin} {o j : ℕ}
    (hj : o + j ≤ as.length) : decidedAt xS ((as ++ e).drop o) cs j = decidedAt xS (as.drop o) cs j := by
  unfold decidedAt
  rw [List.drop_append_of_le_length (by omega), List.take_append_of_le_length (by simp; omega)]

/-- **Flock's step at a prefix**, its coins cut from a longer run's. -/
theorem next_take {xS : Msg} {k : ℕ} (r : Fin k → FlockDeploy.Coin) (bs : List Msg) {j : ℕ} (hj : j ≤ bs.length) :
    FlockDeploy.next true xS (bs.take j) (((List.range bs.length).map (coinSeq r)).take j) =
      FlockDeploy.next true xS (bs.take j) ((List.range (bs.take j).length).map (coinSeq r)) := by
  rw [take_map_range _ hj, List.length_take, Nat.min_eq_left hj]

/-- **The honest prover's first messages**: `m₀`, the interior record `m₁`, `[]` in the second stage's later rounds,
and the session statements `m₂` in round `stage + 1`. -/
def preMsg (m₀ m₁ m₂ : Msg) (j : ℕ) : Msg :=
  if j = 0 then m₀ else if j = 1 then m₁ else if j ≤ h.stage then [] else m₂

theorem preMsg_one (m₀ m₁ m₂ : Msg) : preMsg h m₀ m₁ m₂ 1 = m₁ := by
  simp [preMsg]

theorem preMsg_mid (m₀ m₁ m₂ : Msg) {j : ℕ} (h2 : 2 ≤ j) (hj : j ≤ h.stage) : preMsg h m₀ m₁ m₂ j = [] := by
  unfold preMsg
  rw [ite_eq_right (show ¬j = 0 by omega), ite_eq_right (show ¬j = 1 by omega), ite_eq_left hj]

theorem preMsg_last (m₀ m₁ m₂ : Msg) : preMsg h m₀ m₁ m₂ (h.stage + 1) = m₂ := by
  have hpos : 0 < h.stage := Nat.succ_pos _
  unfold preMsg
  rw [ite_eq_right (show ¬h.stage + 1 = 0 by omega), ite_eq_right (show ¬h.stage + 1 = 1 by omega),
    ite_eq_right (show ¬h.stage + 1 ≤ h.stage by omega)]

/-- **The verifier answers the first messages with its coins**, in rounds 0 to `stage`. -/
theorem next_preMsg (x m₀ m₁ m₂ : Msg) {k : ℕ} (r : Fin k → FlockDeploy.Coin) {j : ℕ} (hj : j ≤ h.stage) :
    next h x ((List.range (j + 1)).map (preMsg h m₀ m₁ m₂))
        ((List.range ((List.range (j + 1)).map (preMsg h m₀ m₁ m₂)).length).map (coinSeq r)) =
      .ret (.msg (coinSeq r j).bytes) := by
  have hlen : ((List.range (j + 1)).map (preMsg h m₀ m₁ m₂)).length = j + 1 := by simp
  have h0 : (List.range (j + 1)).map (preMsg h m₀ m₁ m₂) ≠ [] :=
    List.ne_nil_of_length_pos (by rw [hlen]; exact Nat.succ_pos j)
  have hle : ((List.range (j + 1)).map (preMsg h m₀ m₁ m₂)).length ≤ h.stage + 1 := by rw [hlen]; omega
  have hc : j + 1 ≤ 2 ∨ preMsg h m₀ m₁ m₂ j = [] := by
    rcases Nat.lt_or_ge j 2 with h2 | h2
    · exact Or.inl (by omega)
    · exact Or.inr (preMsg_mid h m₀ m₁ m₂ h2 hj)
  rw [hlen, next_coin h x h0 (by simp) hle, hlen, Nat.add_sub_cancel,
    getD_map_range (preMsg h m₀ m₁ m₂) (Nat.lt_succ_self j), getD_map_range (coinSeq r) (Nat.lt_succ_self j),
    ite_eq_left hc]

/-- **The verifier's statement check on the first messages**, in round `stage + 1`: at round 0's coin and the second
stage's. -/
theorem next_preMsg_stmt (x m₀ m₁ m₂ : Msg) {k : ℕ} (r : Fin k → FlockDeploy.Coin) :
    next h x ((List.range (h.stage + 2)).map (preMsg h m₀ m₁ m₂))
        ((List.range ((List.range (h.stage + 2)).map (preMsg h m₀ m₁ m₂)).length).map (coinSeq r)) =
      .ret (if stmtOk h x m₁ m₂ (coinSeq r 0) (fun i => coinSeq r (i.val + 1)) then .msg [] else .rej) := by
  have hlen : ((List.range (h.stage + 2)).map (preMsg h m₀ m₁ m₂)).length = h.stage + 2 := by simp
  have hst : stageOf h ((List.range (h.stage + 2)).map (coinSeq r)) = fun i => coinSeq r (i.val + 1) :=
    funext fun i => getD_map_range (coinSeq r) (show i.val + 1 < h.stage + 2 by have := i.isLt; omega) default
  rw [hlen, next_stmt h x (by simp) hlen, getD_map_range _ (show 1 < h.stage + 2 by omega),
    getD_map_range _ (show h.stage + 1 < h.stage + 2 by omega), getD_map_range _ (show 0 < h.stage + 2 by omega), hst,
    preMsg_one, preMsg_last]

/-- **SPOT[Flock]'s prover from a Flock prover per template**: rounds 0 to `stage + 1` send `preMsg`'s messages (`m₀`,
`m₁`, `[]` in the second stage's later rounds, `m₂`); a round of template `t`'s session plays `P' t` at the statement
`m₂` states for it on the session's history, until a round of the session is answered `[]`, and then sends `[]`; the
verdict's round sends `[]`. -/
def proverSeq {Ω : Type} (m₀ m₁ m₂ : Msg) (P' : ℕ → Prover Ω) : Prover (ℕ → Ω) :=
  ⟨fun _ hist ω =>
    if hist.length < h.stage + 2 then .ret (preMsg h m₀ m₁ m₂ hist.length)
    else
      match slotOf h (hist.length + 1) with
      | some t =>
        if marked (hist.drop (off h t)) then .ret [] else (P' t).next (stmtAt h m₂ t) (hist.drop (off h t)) (ω t)
      | none => .ret []⟩

theorem proverSeq_pre {Ω : Type} (m₀ m₁ m₂ : Msg) (P' : ℕ → Prover Ω) (x : Msg) (hist : List (Msg × Msg))
    (ω : ℕ → Ω) (hl : hist.length < h.stage + 2) :
    (proverSeq h m₀ m₁ m₂ P').next x hist ω = .ret (preMsg h m₀ m₁ m₂ hist.length) :=
  ite_eq_left hl

theorem proverSeq_next {Ω : Type} (m₀ m₁ m₂ : Msg) (P' : ℕ → Prover Ω) (x : Msg) (hist : List (Msg × Msg))
    (ω : ℕ → Ω) (h3 : h.stage + 2 ≤ hist.length) :
    (proverSeq h m₀ m₁ m₂ P').next x hist ω = match slotOf h (hist.length + 1) with
      | some t =>
        if marked (hist.drop (off h t)) then .ret [] else (P' t).next (stmtAt h m₂ t) (hist.drop (off h t)) (ω t)
      | none => .ret [] :=
  ite_eq_right (Nat.not_lt.2 h3)

/-- **The run's coins from the sessions'**: round 0 is `c₀`, rounds 1 to `stage` are the second stage's `c₁`, and round
`off t + i` of template `t`'s session is `r' t`'s round `i`. -/
def seqCoins {x m₂ : Msg} (c₀ : FlockDeploy.Coin) (c₁ : Stage h)
    (r' : ∀ t, Fin ((FlockDeploy.V true).k (stmtAt h m₂ t)) → FlockDeploy.Coin) :
    Fin ((V h).k x) → FlockDeploy.Coin := fun n =>
  if n.val = 0 then c₀ else if hn : n.val - 1 < h.stage then c₁ ⟨n.val - 1, hn⟩ else
    match slotOf h (n.val + 1) with
    | some t =>
      if ht : n.val - off h t < (FlockDeploy.V true).k (stmtAt h m₂ t) then r' t ⟨n.val - off h t, ht⟩ else default
    | none => default

theorem seqCoins_at {x m₂ : Msg} (c₀ : FlockDeploy.Coin) (c₁ : Stage h)
    (r' : ∀ t, Fin ((FlockDeploy.V true).k (stmtAt h m₂ t)) → FlockDeploy.Coin) {t i : ℕ} (ht : t < h.ms.length)
    (hk : (FlockDeploy.V true).k (stmtAt h m₂ t) = k₀ h t) (hi : i < k₀ h t) :
    coinSeq (seqCoins h (x := x) c₀ c₁ r') (off h t + i) = coinSeq (r' t) i := by
  have h3 := stage_le_off h t
  have hle : off h t + k₀ h t ≤ off h h.ms.length := sessOff_add_le (h.stage + 2) (k₀ h) ht
  have hlt : off h t + i < (V h).k x := by
    show off h t + i < off h h.ms.length + 1
    omega
  unfold coinSeq seqCoins
  rw [dite_eq_left hlt, dite_eq_left (show i < (FlockDeploy.V true).k (stmtAt h m₂ t) by omega)]
  dsimp only
  rw [ite_eq_right (show ¬off h t + i = 0 by omega), dite_eq_right (show ¬off h t + i - 1 < h.stage by omega),
    slotOf_eq h ht (by omega) (by omega)]
  dsimp only
  rw [dite_eq_left (show off h t + i - off h t < (FlockDeploy.V true).k (stmtAt h m₂ t) by omega)]
  exact congrArg (r' t) (Fin.ext (by simp))

/-- **SPOT[Flock] accepts with positive probability where Flock accepts every template's session**: if the statement
check passes `m₁` and `m₂` at some coins `c₀` of round 0 and `c₁` of the second stage's rounds, and Flock's masked
verifier accepts `P' t` at the statement `m₂` states for each template `t` on some coins, then `proverSeq m₀ m₁ m₂ P'`
is accepted on a positive share of SPOT[Flock]'s coins. -/
theorem accepts_seq {Ω : Type} (H : Msg → Msg) (x m₀ m₁ m₂ : Msg) (c₀ : FlockDeploy.Coin) (c₁ : Stage h)
    (hok : stmtOk h x m₁ m₂ c₀ c₁ = true) (P' : ℕ → Prover Ω) (ω : ℕ → Ω)
    (r' : ∀ t, Fin ((FlockDeploy.V true).k (stmtAt h m₂ t)) → FlockDeploy.Coin)
    (hacc : ∀ t < h.ms.length, (run H (P' t) (FlockDeploy.V true) (stmtAt h m₂ t) (ω t) (r' t)).out = .acc) :
    0 < Pr fun r : Fin ((V h).k x) → FlockDeploy.Coin =>
      (run H (proverSeq h m₀ m₁ m₂ P') (V h) x ω r).out = .acc := by
  refine Claims.SamplingGeneric.Pr_pos_of (seqCoins h (x := x) c₀ c₁ r') ?_
  have hT3 := stage_le_off h h.ms.length
  have e0 : coinSeq (seqCoins h (x := x) c₀ c₁ r') 0 = c₀ := by
    unfold coinSeq
    rw [dite_eq_left (show 0 < (V h).k x by show 0 < off h h.ms.length + 1; omega)]
    rfl
  have e1 : ∀ i : Fin h.stage, coinSeq (seqCoins h (x := x) c₀ c₁ r') (i.val + 1) = c₁ i := by
    intro i
    have hi := i.isLt
    unfold coinSeq seqCoins
    rw [dite_eq_left (show i.val + 1 < (V h).k x by show i.val + 1 < off h h.ms.length + 1; omega)]
    dsimp only
    rw [ite_eq_right (show ¬i.val + 1 = 0 by omega), dite_eq_left (show i.val + 1 - 1 < h.stage by omega)]
    exact congrArg c₁ (Fin.ext (by simp))
  have hsc : ∀ t < h.ms.length, ∀ i < k₀ h t,
      coinSeq (seqCoins h (x := x) c₀ c₁ r') (off h t + i) = coinSeq (r' t) i := fun t ht i hi =>
    seqCoins_at h c₀ c₁ r' ht (stmtOk_at h hok ht).1 hi
  generalize seqCoins h (x := x) c₀ c₁ r' = r at e0 e1 hsc ⊢
  set vS : List Msg → OC Out := fun as => (V h).next x as ((List.range as.length).map (coinSeq r)) with hvS
  -- the coins of template `t`'s session, cut from the run's
  have hslot : ∀ t < h.ms.length, ∀ l ≤ k₀ h t,
      (List.range l).map (fun i => coinSeq r (off h t + i)) = (List.range l).map (coinSeq (r' t)) :=
    fun t ht l hl => List.map_congr_left fun i hi => hsc t ht i (by rw [List.mem_range] at hi; omega)
  -- the first `stage + 2` rounds
  have hokr : stmtOk h x m₁ m₂ (coinSeq r 0) (fun i => coinSeq r (i.val + 1)) = true :=
    (congrArg₂ (stmtOk h x m₁ m₂) e0 (funext e1)).trans hok
  have hVpre : ∀ j < h.stage + 2, vS ((List.range (j + 1)).map (preMsg h m₀ m₁ m₂)) =
      .ret (.msg (if j ≤ h.stage then (coinSeq r j).bytes else [])) := by
    intro j hj
    by_cases hle : j ≤ h.stage
    · rw [ite_eq_left hle]
      exact next_preMsg h x m₀ m₁ m₂ r hle
    · have hj' : j = h.stage + 1 := by omega
      subst hj'
      rw [ite_eq_right hle]
      exact (next_preMsg_stmt h x m₀ m₁ m₂ r).trans (by rw [ite_eq_left hokr])
  obtain ⟨hist₀, tr₀, hl₀, heq₀⟩ := stepWith_pre H (proverSeq h m₀ m₁ m₂ P') x ω vS (preMsg h m₀ m₁ m₂)
    (fun j => if j ≤ h.stage then (coinSeq r j).bytes else []) (h.stage + 2)
    (fun hist hl => proverSeq_pre h m₀ m₁ m₂ P' x hist ω hl) hVpre (h.stage + 2) le_rfl
  -- the run's coins, all its rounds
  let cs := (List.range (off h h.ms.length + 1)).map (coinSeq r)
  have hcs : ∀ t < h.ms.length, ∀ j ≤ k₀ h t, (cs.drop (off h t)).take j = (List.range j).map (coinSeq (r' t)) := by
    intro t ht j hj
    have hle : off h t + k₀ h t ≤ off h h.ms.length := sessOff_add_le (h.stage + 2) (k₀ h) ht
    show (((List.range (off h h.ms.length + 1)).map (coinSeq r)).drop (off h t)).take j = _
    rw [show off h h.ms.length + 1 = off h t + (off h h.ms.length + 1 - off h t) by omega, drop_map_range,
      take_map_range _ (by omega), hslot t ht j hj]
  have key : ∀ t, t ≤ h.ms.length → ∃ (hist : List (Msg × Msg)) (as : List Msg) (tr : Transcript),
      hist.length = off h t ∧ as.length = off h t ∧ as.getD (h.stage + 1) [] = m₂ ∧
      (∀ s < t, ∃ j, 0 < j ∧ j ≤ k₀ h s ∧
        decidedAt (stmtAt h m₂ s) (as.drop (off h s)) (cs.drop (off h s)) j = true) ∧
      ∀ rest, stepWith H (proverSeq h m₀ m₁ m₂ P') x ω vS (off h t + rest) [] [] ⟨[], [], [], .rej⟩ =
        stepWith H (proverSeq h m₀ m₁ m₂ P') x ω vS rest hist as tr := by
    intro t
    induction t with
    | zero =>
      intro _
      have ho : off h 0 = h.stage + 2 := rfl
      refine ⟨hist₀, (List.range (h.stage + 2)).map (preMsg h m₀ m₁ m₂), tr₀, hl₀.trans ho.symm, by simp [ho], ?_,
        fun s hs => absurd hs (Nat.not_lt_zero s), fun rest => heq₀ rest⟩
      rw [getD_map_range _ (show h.stage + 1 < h.stage + 2 by omega), preMsg_last]
    | succ t ih =>
      intro ht1
      have ht : t < h.ms.length := ht1
      have h3t := stage_le_off h t
      have hle : off h t + k₀ h t ≤ off h h.ms.length := sessOff_add_le (h.stage + 2) (k₀ h) ht
      obtain ⟨hist, as, tr, hhl, hal, ha2, hdec, heq⟩ := ih (by omega)
      have hkt := (stmtOk_at h hok ht).1
      have hacc' : (stepWith H (P' t) (stmtAt h m₂ t) (ω t)
          (fun bs => FlockDeploy.next true (stmtAt h m₂ t) bs ((List.range bs.length).map (coinSeq (r' t))))
          (k₀ h t) [] [] ⟨[], [], [], .rej⟩).out = .acc :=
        (congrArg (fun n => (stepWith H (P' t) (stmtAt h m₂ t) (ω t)
          (fun bs => FlockDeploy.next true (stmtAt h m₂ t) bs ((List.range bs.length).map (coinSeq (r' t))))
          n [] [] ⟨[], [], [], .rej⟩).out) hkt).symm.trans (hacc t ht)
      have hcoins : ∀ l ≤ k₀ h t, ((List.range (off h t + l)).map (coinSeq r)).drop (off h t) =
          (List.range l).map (coinSeq (r' t)) := fun l hl => by rw [drop_map_range, hslot t ht l hl]
      have hPs : ∀ hst : List (Msg × Msg), marked hst = false → hst.length < k₀ h t →
          (proverSeq h m₀ m₁ m₂ P').next x (hist ++ hst) ω = (P' t).next (stmtAt h m₂ t) hst (ω t) := by
        intro hst hm hl
        rw [proverSeq_next h m₀ m₁ m₂ P' x (hist ++ hst) ω (by simp; omega), List.length_append, hhl,
          slotOf_eq h ht (by omega) (by omega)]
        simp [List.drop_left' hhl, hm]
      have hfPs : ∀ hst : List (Msg × Msg), marked hst = true → hst.length < k₀ h t →
          (proverSeq h m₀ m₁ m₂ P').next x (hist ++ hst) ω = .ret [] := by
        intro hst hm hl
        rw [proverSeq_next h m₀ m₁ m₂ P' x (hist ++ hst) ω (by simp; omega), List.length_append, hhl,
          slotOf_eq h ht (by omega) (by omega)]
        simp [List.drop_left' hhl, hm]
      have hstep : ∀ bs : List Msg, bs ≠ [] → bs.length ≤ k₀ h t →
          vS (as ++ bs) = sessStep h t (stmtAt h m₂ t) (as ++ bs)
            ((List.range (off h t + bs.length)).map (coinSeq r)) := by
        intro bs hbs hbl
        have hpos : 0 < bs.length := List.length_pos_of_ne_nil hbs
        have hlen : (as ++ bs).length = off h t + bs.length := by simp [hal]
        show next h x (as ++ bs) ((List.range (as ++ bs).length).map (coinSeq r)) = _
        rw [next_long h x (as ++ bs) ((List.range (as ++ bs).length).map (coinSeq r)) (by rw [hlen]; omega)
          (by simp), hlen, slotOf_eq h ht (by omega) (by omega)]
        dsimp only
        rw [getD_append_left (by omega), ha2]
      have hvs : ∀ bs : List Msg, bs ≠ [] → bs.length ≤ k₀ h t →
          (∀ j, 0 < j → j < bs.length → ∃ m, ((FlockDeploy.next true (stmtAt h m₂ t) (bs.take j)
            ((List.range (bs.take j).length).map (coinSeq (r' t)))).eval H).1 = .msg m) →
          (vS (as ++ bs)).eval H = (markOut [] ((FlockDeploy.next true (stmtAt h m₂ t) bs
            ((List.range bs.length).map (coinSeq (r' t)))).eval H).1, ((FlockDeploy.next true (stmtAt h m₂ t) bs
            ((List.range bs.length).map (coinSeq (r' t)))).eval H).2) := by
        intro bs hbs hbl hund
        have hlen : (as ++ bs).length = off h t + bs.length := by simp [hal]
        have hund' : ∀ j, 0 < j → j < (as ++ bs).length - off h t →
            decidedAt (stmtAt h m₂ t) ((as ++ bs).drop (off h t))
              (((List.range (off h t + bs.length)).map (coinSeq r)).drop (off h t)) j = false := by
          intro j hj0 hj
          rw [hlen, Nat.add_sub_cancel_left] at hj
          rw [List.drop_left' hal, hcoins bs.length hbl]
          obtain ⟨m, hm⟩ := hund j hj0 hj
          apply not_decidedAt (H := H) (m := m)
          rw [next_take (r' t) bs hj.le]
          exact hm
        rw [hstep bs hbs hbl, sessStep_eval h H t (stmtAt h m₂ t) (as ++ bs) _ hund', List.drop_left' hal,
          hcoins bs.length hbl]
      have hfVs : ∀ bs : List Msg, bs.length ≤ k₀ h t →
          (∃ j, 0 < j ∧ j < bs.length ∧ ((FlockDeploy.next true (stmtAt h m₂ t) (bs.take j)
            ((List.range (bs.take j).length).map (coinSeq (r' t)))).eval H).1 = .acc) →
          (vS (as ++ bs)).eval H = (.msg [], []) := by
        rintro bs hbl ⟨j, hj0, hjl, hja⟩
        have hbs : bs ≠ [] := List.ne_nil_of_length_pos (by omega)
        have hlen : (as ++ bs).length = off h t + bs.length := by simp [hal]
        rw [hstep bs hbs hbl]
        unfold sessStep
        have hd : decidedAt (stmtAt h m₂ t) bs ((List.range bs.length).map (coinSeq (r' t))) j = true := by
          apply decidedAt_of_ne_msg (H := H)
          intro m hm
          rw [next_take (r' t) bs hjl.le, hja] at hm
          exact Out.noConfusion hm
        have hc : ((List.range ((as ++ bs).length - off h t)).any fun j => decide (0 < j) &&
            decidedAt (stmtAt h m₂ t) ((as ++ bs).drop (off h t))
              (((List.range (off h t + bs.length)).map (coinSeq r)).drop (off h t)) j) = true := by
          rw [List.any_eq_true]
          refine ⟨j, List.mem_range.2 (by rw [hlen]; omega), ?_⟩
          rw [List.drop_left' hal, hcoins bs.length hbl, hd]
          simp [hj0]
        rw [ite_eq_left hc]
        rfl
      obtain ⟨hist', as', t', hl1, hl2, hd', heq'⟩ := slot_run H (proverSeq h m₀ m₁ m₂ P') (P' t) x
        (stmtAt h m₂ t) ω (ω t) vS
        (fun bs => FlockDeploy.next true (stmtAt h m₂ t) bs ((List.range bs.length).map (coinSeq (r' t))))
        hist as (k₀ h t) hPs hvs (fun bs => flock_ne_nil H (stmtAt h m₂ t) bs _) hfPs hfVs (k₀ h t) [] []
        ⟨[], [], [], .rej⟩ tr (by simp) rfl rfl (fun j hj0 hj => absurd hj (by simp; omega)) hacc'
      refine ⟨hist ++ hist', as ++ as', t', by simp [hhl, hl2, off_succ], by simp [hal, hl1, off_succ],
        by rw [getD_append_left (by omega), ha2], fun s hs => ?_, fun rest => ?_⟩
      · rcases Nat.lt_or_ge s t with hst | hst
        · obtain ⟨j, hj0, hjk, hd⟩ := hdec s hst
          have hle' : off h s + k₀ h s ≤ off h t := sessOff_add_le (h.stage + 2) (k₀ h) hst
          exact ⟨j, hj0, hjk, by rw [decidedAt_append (by omega)]; exact hd⟩
        · rw [show s = t by omega]
          obtain ⟨j, hj0, hjK, hja⟩ := hd'
          refine ⟨j, hj0, hjK, ?_⟩
          rw [List.drop_left' hal]
          apply decidedAt_of_ne_msg (H := H)
          intro m hm
          rw [hcs t ht j hjK, ← take_map_range (coinSeq (r' t)) (show j ≤ as'.length by omega),
            next_take (r' t) as' (by omega), hja] at hm
          exact Out.noConfusion hm
      · have := heq' rest
        rw [List.append_nil, List.append_nil] at this
        rw [show off h (t + 1) + rest = off h t + (k₀ h t + rest) by rw [off_succ]; omega, heq, this]
  obtain ⟨histT, asT, trT, hhT, haT, ha2T, hdecT, heqT⟩ := key h.ms.length le_rfl
  have hrun : (run H (proverSeq h m₀ m₁ m₂ P') (V h) x ω r).out =
      (stepWith H (proverSeq h m₀ m₁ m₂ P') x ω vS 1 histT asT trT).out := congrArg Transcript.out (heqT 1)
  rw [hrun]
  have hPT : (proverSeq h m₀ m₁ m₂ P').next x histT ω = .ret [] := by
    rw [proverSeq_next h m₀ m₁ m₂ P' x histT ω (by omega), hhT, slotOf_last]
  have hver : verdict h m₂ (asT ++ [[]]) cs = true := by
    unfold verdict
    rw [List.all_eq_true]
    intro t ht
    rw [List.mem_range] at ht
    obtain ⟨j, hj0, hjk, hd⟩ := hdecT t ht
    have hle : off h t + k₀ h t ≤ off h h.ms.length := sessOff_add_le (h.stage + 2) (k₀ h) ht
    rw [List.any_eq_true]
    refine ⟨j, List.mem_range.2 (by omega), ?_⟩
    rw [decidedAt_append (by omega), hd]
    simp [hj0]
  have hvT : (vS (asT ++ [((((proverSeq h m₀ m₁ m₂ P').next x histT ω)).eval H).1])).eval H = (.acc, []) := by
    rw [hPT]
    have hlen : (asT ++ [([] : Msg)]).length = off h h.ms.length + 1 := by simp [haT]
    show (next h x (asT ++ [[]]) ((List.range (asT ++ [([] : Msg)]).length).map (coinSeq r))).eval H = _
    rw [next_long h x (asT ++ [[]]) ((List.range (asT ++ [([] : Msg)]).length).map (coinSeq r))
      (by rw [hlen]; omega) (by simp), hlen, slotOf_last]
    dsimp only
    rw [getD_append_left (by omega), ha2T]
    show (OC.ret (if (off h h.ms.length + 1 == off h h.ms.length + 1 && verdict h m₂ (asT ++ [[]]) cs) = true
      then Out.acc else Out.rej)).eval H = _
    rw [hver]
    simp [OC.eval]
  show (stepWith H (proverSeq h m₀ m₁ m₂ P') x ω vS (0 + 1) histT asT trT).out = .acc
  rw [stepWith_acc_of H (proverSeq h m₀ m₁ m₂ P') x ω vS 0 histT asT trT _ rfl [] hvT]

end SpotFlock
