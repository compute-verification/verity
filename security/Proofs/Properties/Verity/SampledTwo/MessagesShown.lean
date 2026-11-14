module

public import Proofs.Properties.Verity.SampledTwo.Messages
public import Proofs.Flock.Claims.Recursion.DeployFacts
public import Proofs.Flock.Claims.Recursion.FlockKSDeploy
public import Proofs.Flock.Claims.Recursion.VStarCross
public import Proofs.Flock.Claims.Bridge.Committed
public import Proofs.Flock.SpotFlock.Sessions

/-!
# The run's history shows the coins its answers read

`Messages.RunShown` asks that a run's history before a round shows the run's coins (each answer that is a coin's bytes
is that round's coin's), and that the run's rounds before it are the same at any coins agreeing with what it shows.
This module gives it from two facts about the verifier's step alone, which a stateless prover's run then carries
round by round:

* **An answer that is a coin's bytes is the round's own coin's** (`AnswersOwn`).
* **An answer reads only the coins earlier answers showed, and the round's own where it answers with it**
  (`AnswersShown`), on messages and coins whose every earlier answer was a message (a run that got that far).

`runShown_of` is `RunShown` from the two. The run's agreement round by round is `stepWith_agree_on`: two verifiers
that answer alike on the messages one run of them actually sends give the same rounds. `runShown` holds at every
`Live₂`, so `messagesFixed` is `Messages.messagesBound` without it.
-/

@[expose] public section

namespace Verity.SampledTwo.Messages

open Claims

/-! ## Runs that agree along one run's messages -/

section Agree

variable {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω)

/-- **Two verifiers that answer alike on the messages the first one's run sends give the same rounds**, and the runs go
past round `m` together. -/
theorem stepWith_agree_on (vn vn' : List Msg → Claims.OC Out) :
    ∀ (f m : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t t' : Transcript), t.msgs = t'.msgs →
      (∀ l < m, ∀ new : List (Msg × Out), new.length = l →
        t.msgs ++ new <+: (stepWith H P x ω vn f hist as t).msgs →
        t.msgs.length + l < (stepWith H P x ω vn f hist as t).msgs.length →
        ((vn (as ++ new.map Prod.fst ++ [((P.next x (hist ++ new.map RecDeploy.histOf) ω).eval H).1])).eval H).1 =
          ((vn' (as ++ new.map Prod.fst ++
            [((P.next x (hist ++ new.map RecDeploy.histOf) ω).eval H).1])).eval H).1) →
      (stepWith H P x ω vn f hist as t).msgs.take (t.msgs.length + m) =
          (stepWith H P x ω vn' f hist as t').msgs.take (t.msgs.length + m) ∧
        (t.msgs.length + m < (stepWith H P x ω vn f hist as t).msgs.length ↔
          t.msgs.length + m < (stepWith H P x ω vn' f hist as t').msgs.length)
  | 0, m, hist, as, t, t', ht, _ => by simp [stepWith, ht]
  | f + 1, 0, hist, as, t, t', ht, _ => by
    obtain ⟨r₁, h₁⟩ := (RecDeploy.stepWith_mono H P x ω vn (f + 1) hist as t).1
    obtain ⟨r₂, h₂⟩ := (RecDeploy.stepWith_mono H P x ω vn' (f + 1) hist as t').1
    have l₁ := RecDeploy.stepWith_length_succ H P x ω vn f hist as t
    have l₂ := RecDeploy.stepWith_length_succ H P x ω vn' f hist as t'
    rw [ht] at l₁ ⊢
    refine ⟨?_, by simp only [Nat.add_zero]; omega⟩
    rw [← h₁, ← h₂, ← ht]
    simp
  | f + 1, m + 1, hist, as, t, t', ht, hag => by
    have hpre := (RecDeploy.stepWith_mono H P x ω vn (f + 1) hist as t).1
    have hlen := RecDeploy.stepWith_length_succ H P x ω vn f hist as t
    have he := hag 0 (Nat.succ_pos _) [] rfl (by simpa using hpre) (by simpa using hlen)
    simp only [List.map_nil, List.append_nil] at he
    rcases hvb : (vn (as ++ [((P.next x hist ω).eval H).1])).eval H with ⟨o, lo⟩
    rcases hvb' : (vn' (as ++ [((P.next x hist ω).eval H).1])).eval H with ⟨o', lo'⟩
    rw [hvb, hvb'] at he
    simp only at he
    subst he
    cases o with
    | msg mm =>
      rw [RecDeploy.stepWith_succ_msg H P x ω vn f hist as t hvb] at hag ⊢
      rw [RecDeploy.stepWith_succ_msg H P x ω vn' f hist as t' hvb']
      have := stepWith_agree_on vn vn' f m (hist ++ [(((P.next x hist ω).eval H).1, mm)])
        (as ++ [((P.next x hist ω).eval H).1])
        { t with msgs := t.msgs ++ [(((P.next x hist ω).eval H).1, .msg mm)],
                 pLog := t.pLog ++ ((P.next x hist ω).eval H).2, vLog := t.vLog ++ lo }
        { t' with msgs := t'.msgs ++ [(((P.next x hist ω).eval H).1, .msg mm)],
                  pLog := t'.pLog ++ ((P.next x hist ω).eval H).2, vLog := t'.vLog ++ lo' }
        (by simp [ht]) (fun l hl new hnew hp hlt => by
          have h' := hag (l + 1) (by omega) ((((P.next x hist ω).eval H).1, .msg mm) :: new)
            (by simp [hnew]) (by simpa using hp) (by simp only [List.length_append, List.length_singleton] at hlt; omega)
          simpa [RecDeploy.histOf, outMsg_eq, msgOf, List.append_assoc] using h')
      simp only [List.length_append, List.length_singleton] at this
      rwa [Nat.add_right_comm, Nat.add_assoc] at this
    | acc =>
      rw [RecDeploy.stepWith_succ_acc H P x ω vn f hist as t hvb,
        RecDeploy.stepWith_succ_acc H P x ω vn' f hist as t' hvb']
      simp [ht]
    | rej =>
      rw [RecDeploy.stepWith_succ_rej H P x ω vn f hist as t hvb,
        RecDeploy.stepWith_succ_rej H P x ω vn' f hist as t' hvb']
      simp [ht]

/-- **A round before a run's last answered a message.** -/
theorem stepWith_msg_lt (vn : List Msg → Claims.OC Out) :
    ∀ (f : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t : Transcript) (j : ℕ),
      t.msgs.length + j + 1 < (stepWith H P x ω vn f hist as t).msgs.length →
      ∃ e mm, (stepWith H P x ω vn f hist as t).msgs[t.msgs.length + j]? = some e ∧ e.2 = .msg mm
  | 0, _, _, t, j, h => by
    simp only [stepWith] at h
    omega
  | f + 1, hist, as, t, j, h => by
    rcases hvb : (vn (as ++ [((P.next x hist ω).eval H).1])).eval H with ⟨o, lo⟩
    cases o with
    | msg mm =>
      rw [RecDeploy.stepWith_succ_msg H P x ω vn f hist as t hvb] at h ⊢
      obtain ⟨rest, hrest⟩ := (RecDeploy.stepWith_mono H P x ω vn f
        (hist ++ [(((P.next x hist ω).eval H).1, mm)]) (as ++ [((P.next x hist ω).eval H).1])
        { t with msgs := t.msgs ++ [(((P.next x hist ω).eval H).1, .msg mm)],
                 pLog := t.pLog ++ ((P.next x hist ω).eval H).2, vLog := t.vLog ++ lo }).1
      cases j with
      | zero =>
        refine ⟨(((P.next x hist ω).eval H).1, .msg mm), mm, ?_, rfl⟩
        rw [← hrest]
        simp
      | succ j =>
        have := stepWith_msg_lt vn f (hist ++ [(((P.next x hist ω).eval H).1, mm)])
          (as ++ [((P.next x hist ω).eval H).1])
          { t with msgs := t.msgs ++ [(((P.next x hist ω).eval H).1, .msg mm)],
                   pLog := t.pLog ++ ((P.next x hist ω).eval H).2, vLog := t.vLog ++ lo } j
          (by simp only [List.length_append, List.length_singleton] at h ⊢; omega)
        simp only [List.length_append, List.length_singleton] at this
        rwa [Nat.add_right_comm, Nat.add_assoc] at this
    | acc =>
      rw [RecDeploy.stepWith_succ_acc H P x ω vn f hist as t hvb] at h
      simp only [List.length_append, List.length_singleton] at h
      omega
    | rej =>
      rw [RecDeploy.stepWith_succ_rej H P x ω vn f hist as t hvb] at h
      simp only [List.length_append, List.length_singleton] at h
      omega

end Agree

/-- **A round before a run's last answered a message.** -/
theorem run_msg_lt {Ω C : Type} [Inhabited C] (H : Msg → Msg) (P : Prover Ω) (Vr : Verifier C) (x : Msg) (ω : Ω)
    (r : Fin (Vr.k x) → C) {j : ℕ} (h : j + 1 < (Claims.run H P Vr x ω r).msgs.length) :
    ∃ e mm, (Claims.run H P Vr x ω r).msgs[j]? = some e ∧ e.2 = .msg mm := by
  have := stepWith_msg_lt H P x ω (fun as => Vr.next x as ((List.range as.length).map (coinSeq r))) (Vr.k x) [] []
    ⟨[], [], [], .rej⟩ j (by simpa [Claims.run] using h)
  simpa [Claims.run] using this

/-- **Runs at two coins agree on their first `n` rounds** where the verifier answers alike at both on the messages the
first coins' run sends. -/
theorem run_agree_on {Ω C : Type} [Inhabited C] (H : Msg → Msg) (P : Prover Ω) (Vr : Verifier C) (x : Msg) (ω : Ω)
    (r r' : Fin (Vr.k x) → C) (n : ℕ)
    (h : ∀ l < n, l < (Claims.run H P Vr x ω r).msgs.length →
      ((Vr.next x (((Claims.run H P Vr x ω r).msgs.take l).map Prod.fst ++
          [((P.next x (((Claims.run H P Vr x ω r).msgs.take l).map RecDeploy.histOf) ω).eval H).1])
          ((List.range (l + 1)).map (coinSeq r))).eval H).1 =
        ((Vr.next x (((Claims.run H P Vr x ω r).msgs.take l).map Prod.fst ++
          [((P.next x (((Claims.run H P Vr x ω r).msgs.take l).map RecDeploy.histOf) ω).eval H).1])
          ((List.range (l + 1)).map (coinSeq r'))).eval H).1) :
    (Claims.run H P Vr x ω r').msgs.take n = (Claims.run H P Vr x ω r).msgs.take n := by
  have := (stepWith_agree_on H P x ω (fun as => Vr.next x as ((List.range as.length).map (coinSeq r)))
    (fun as => Vr.next x as ((List.range as.length).map (coinSeq r'))) (Vr.k x) n [] [] ⟨[], [], [], .rej⟩
    ⟨[], [], [], .rej⟩ rfl (fun l hl new hnew hp hlt => by
      simp only [List.nil_append, List.length_nil, Nat.zero_add] at hp hlt ⊢
      obtain ⟨u, hu⟩ := hp
      have hnew' : new = (Claims.run H P Vr x ω r).msgs.take l := by
        unfold Claims.run
        rw [← hu, List.take_left' hnew]
      subst hnew'
      have hl1 : (((Claims.run H P Vr x ω r).msgs.take l).map Prod.fst ++
          [((P.next x (((Claims.run H P Vr x ω r).msgs.take l).map RecDeploy.histOf) ω).eval H).1]).length =
          l + 1 := by
        simp only [List.length_append, List.length_map, List.length_singleton, hnew]
      rw [hl1]
      exact h l hl hlt)).1
  simp only [List.length_nil, Nat.zero_add] at this
  exact this.symm

/-! ## The verifier's step -/

/-- **An answer that is a coin's bytes is the round's own coin's.** -/
def AnswersOwn (Vr : Verifier FlockDeploy.Coin) : Prop :=
  ∀ (x : Msg) (as : List Msg) (cs : List FlockDeploy.Coin) (c : FlockDeploy.Coin), as ≠ [] →
    cs.length = as.length → msgOf ((Vr.next x as cs).eval FlockDeploy.sha512).1 = c.bytes →
      c = cs.getD (as.length - 1) default

/-- **An answer reads only the coins the earlier answers showed, and the round's own where it answers with it**: on
messages and coins whose earlier answers were all messages, coins `cs'` that agree with `cs` at every round whose
answer at `cs` is that round's coin's bytes get the same answer. -/
def AnswersShown (Vr : Verifier FlockDeploy.Coin) : Prop :=
  ∀ (x : Msg) (as : List Msg) (cs cs' : List FlockDeploy.Coin), as ≠ [] → cs.length = as.length →
    cs'.length = as.length →
    (∀ j < as.length - 1, ∃ m,
      ((Vr.next x (as.take (j + 1)) (cs.take (j + 1))).eval FlockDeploy.sha512).1 = .msg m) →
    (∀ j < as.length, msgOf ((Vr.next x (as.take (j + 1)) (cs.take (j + 1))).eval FlockDeploy.sha512).1 =
        (cs.getD j default).bytes → cs'.getD j default = cs.getD j default) →
    ((Vr.next x as cs').eval FlockDeploy.sha512).1 = ((Vr.next x as cs).eval FlockDeploy.sha512).1

/-! ## The history shows the coins -/

theorem shownCoin_bytes {m : Msg} {c : FlockDeploy.Coin} (h : shownCoin m = some c) : c.bytes = m := by
  unfold shownCoin at h
  split_ifs at h with hm
  cases h
  exact hm.choose_spec

theorem shownCoin_of (c : FlockDeploy.Coin) : shownCoin c.bytes = some c := by
  have hex : ∃ c' : FlockDeploy.Coin, c'.bytes = c.bytes := ⟨c, rfl⟩
  unfold shownCoin
  rw [dif_pos hex]
  exact congrArg some (RecDeploy.coin_bytes_injective hex.choose_spec)

theorem getD_range_map {β : Type} (f : ℕ → β) {n j : ℕ} (h : j < n) (d : β) :
    ((List.range n).map f).getD j d = f j := by
  rw [List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range h]
  rfl

theorem take_range_map {β : Type} (f : ℕ → β) {n m : ℕ} (h : m ≤ n) :
    ((List.range n).map f).take m = (List.range m).map f := by
  rw [← List.map_take, List.take_range, Nat.min_eq_left h]

theorem take_map_take {β γ : Type} (f : β → γ) (L : List β) {a b : ℕ} (h : b ≤ a) :
    ((L.take a).map f).take b = (L.take b).map f := by
  rw [← List.map_take, List.take_take, Nat.min_eq_left h]

theorem coinSeq_lt {C : Type} [Inhabited C] {k : ℕ} (r : Fin k → C) {j : ℕ} (h : j < k) : coinSeq r j = r ⟨j, h⟩ := by
  unfold coinSeq
  rw [dif_pos h]

theorem runHist_get {t : Claims.Transcript} {i j : ℕ} {e : Msg × Msg} (h : (runHist t i)[j]? = some e) :
    j < i ∧ ∃ p, t.msgs[j]? = some p ∧ e = RecDeploy.histOf p := by
  unfold runHist at h
  rw [List.getElem?_map, List.getElem?_take] at h
  by_cases hj : j < i
  · rw [if_pos hj] at h
    cases hp : t.msgs[j]? with
    | none => rw [hp] at h; cases h
    | some p =>
      rw [hp] at h
      exact ⟨hj, p, rfl, (Option.some.inj h).symm⟩
  · rw [if_neg hj] at h
    cases h

theorem runHist_of {t : Claims.Transcript} {i j : ℕ} {p : Msg × Out} (hj : j < i) (h : t.msgs[j]? = some p) :
    (runHist t i)[j]? = some (RecDeploy.histOf p) := by
  unfold runHist
  rw [List.getElem?_map, List.getElem?_take, if_pos hj, h]
  rfl

theorem shownAt_some {hist : List (Msg × Msg)} {j : ℕ} {c : FlockDeploy.Coin} (h : shownAt hist j = some c) :
    ∃ e, hist[j]? = some e ∧ shownCoin e.2 = some c := by
  unfold shownAt at h
  cases he : hist[j]? with
  | none => rw [he] at h; cases h
  | some e =>
    rw [he, Option.bind_some] at h
    exact ⟨e, rfl, h⟩

/-- **A run's answer at round `j`** is the verifier's on the run's first `j + 1` messages and coins. -/
theorem run_answer {Ω : Type} (P : Prover Ω) (Vr : Verifier FlockDeploy.Coin) (x : Msg) (ω : Ω)
    (r : Fin (Vr.k x) → FlockDeploy.Coin) {j : ℕ} (hj : j < (Claims.run FlockDeploy.sha512 P Vr x ω r).msgs.length) :
    ∃ e, (Claims.run FlockDeploy.sha512 P Vr x ω r).msgs[j]? = some e ∧
      e.2 = ((Vr.next x (((Claims.run FlockDeploy.sha512 P Vr x ω r).msgs.take (j + 1)).map Prod.fst)
        ((List.range (j + 1)).map (coinSeq r))).eval FlockDeploy.sha512).1 := by
  obtain ⟨new, hnew, hget⟩ := RecDeploy.run_get FlockDeploy.sha512 P Vr x ω r j hj
  refine ⟨_, hget, ?_⟩
  rw [List.take_succ, hget, ← hnew]
  simp

/-- **`RunShown` from the verifier's step**: where an answer that is a coin's bytes is its round's own coin's, and an
answer reads only the coins earlier answers showed and its own where it answers with it, every run's history shows its
coins, and coins agreeing with what it shows give the same rounds. -/
theorem runShown_of {A : Live₂} {x : Msg} (hO : AnswersOwn A.L.V) (hS : AnswersShown A.L.V) : RunShown A x := by
  intro Ω P ω r i
  have hk : ∀ j, j < (Claims.run FlockDeploy.sha512 P A.L.V x ω r).msgs.length → j < A.L.V.k x := fun j hj =>
    lt_of_lt_of_le hj (Claims.Recursion.run_length_le _ _ _ _ _ _)
  -- the messages the run sends, up to round `l`, are the first `l + 1` of its transcript's
  have hasEq : ∀ l, l < (Claims.run FlockDeploy.sha512 P A.L.V x ω r).msgs.length →
      ((Claims.run FlockDeploy.sha512 P A.L.V x ω r).msgs.take l).map Prod.fst ++
          [((P.next x (((Claims.run FlockDeploy.sha512 P A.L.V x ω r).msgs.take l).map RecDeploy.histOf) ω).eval
            FlockDeploy.sha512).1] =
        ((Claims.run FlockDeploy.sha512 P A.L.V x ω r).msgs.take (l + 1)).map Prod.fst := by
    intro l hl
    obtain ⟨new, hnew, hget⟩ := RecDeploy.run_get FlockDeploy.sha512 P A.L.V x ω r l hl
    rw [List.take_succ, hget, ← hnew]
    simp
  -- a round whose answer is its coin's bytes shows that coin
  have hshow : ∀ j, j < i → (hj : j < (Claims.run FlockDeploy.sha512 P A.L.V x ω r).msgs.length) →
      msgOf ((A.L.V.next x (((Claims.run FlockDeploy.sha512 P A.L.V x ω r).msgs.take (j + 1)).map Prod.fst)
        ((List.range (j + 1)).map (coinSeq r))).eval FlockDeploy.sha512).1 = (r ⟨j, hk j hj⟩).bytes →
      shownAt (runHist (Claims.run FlockDeploy.sha512 P A.L.V x ω r) i) j = some (r ⟨j, hk j hj⟩) := by
    intro j hji hj hm
    obtain ⟨e, he, he2⟩ := run_answer P A.L.V x ω r hj
    unfold shownAt
    rw [runHist_of hji he, Option.bind_some]
    simp only [RecDeploy.histOf]
    rw [outMsg_eq, he2, hm]
    exact shownCoin_of _
  refine ⟨?_, fun r' hsh => ?_⟩
  · intro j c hc
    obtain ⟨e', he', hsc⟩ := shownAt_some hc
    obtain ⟨hji, p, hp, rfl⟩ := runHist_get he'
    have hj : j.val < (Claims.run FlockDeploy.sha512 P A.L.V x ω r).msgs.length := by
      by_contra hn
      rw [List.getElem?_eq_none (l := (Claims.run FlockDeploy.sha512 P A.L.V x ω r).msgs) (by omega)] at hp
      cases hp
    obtain ⟨e, he, he2⟩ := run_answer P A.L.V x ω r hj
    rw [he] at hp
    cases hp
    have hb := shownCoin_bytes hsc
    simp only [RecDeploy.histOf] at hb
    rw [outMsg_eq, he2] at hb
    have := hO x (((Claims.run FlockDeploy.sha512 P A.L.V x ω r).msgs.take (j.val + 1)).map Prod.fst)
      ((List.range (j.val + 1)).map (coinSeq r)) c
      (List.ne_nil_of_length_pos (by rw [List.length_map, List.length_take]; omega))
      (by rw [List.length_map, List.length_map, List.length_range, List.length_take]; omega) hb.symm
    rw [this, List.length_map, List.length_take,
      Nat.min_eq_left (show j.val + 1 ≤ (Claims.run FlockDeploy.sha512 P A.L.V x ω r).msgs.length by omega),
      Nat.add_sub_cancel, getD_range_map _ (Nat.lt_add_one j.val), coinSeq_lt r j.isLt]
  · refine run_agree_on FlockDeploy.sha512 P A.L.V x ω r r' i fun l hl hlr => ?_
    rw [hasEq l hlr]
    have hlen : (((Claims.run FlockDeploy.sha512 P A.L.V x ω r).msgs.take (l + 1)).map Prod.fst).length = l + 1 := by
      rw [List.length_map, List.length_take]
      omega
    refine (hS x (((Claims.run FlockDeploy.sha512 P A.L.V x ω r).msgs.take (l + 1)).map Prod.fst)
      ((List.range (l + 1)).map (coinSeq r)) ((List.range (l + 1)).map (coinSeq r'))
      (List.ne_nil_of_length_pos (by rw [hlen]; omega)) (by rw [hlen]; simp) (by rw [hlen]; simp)
      (fun j hj => ?_) (fun j hj hm => ?_)).symm
    · rw [hlen, Nat.add_sub_cancel] at hj
      rw [take_map_take _ _ (show j + 1 ≤ l + 1 by omega), take_range_map _ (show j + 1 ≤ l + 1 by omega)]
      obtain ⟨e, he, he2⟩ := run_answer P A.L.V x ω r (show j < _ by omega)
      obtain ⟨e', mm, he', he'2⟩ := run_msg_lt FlockDeploy.sha512 P A.L.V x ω r (show j + 1 < _ by omega)
      rw [he] at he'
      cases he'
      exact ⟨mm, he2.symm.trans he'2⟩
    · rw [hlen] at hj
      rw [take_map_take _ _ (show j + 1 ≤ l + 1 by omega), take_range_map _ (show j + 1 ≤ l + 1 by omega),
        getD_range_map _ hj] at hm
      have hjr : j < (Claims.run FlockDeploy.sha512 P A.L.V x ω r).msgs.length := by omega
      rw [coinSeq_lt r (hk j hjr)] at hm
      have hs := hsh ⟨j, hk j hjr⟩ _ (hshow j (by omega) hjr hm)
      rw [getD_range_map _ hj, getD_range_map _ hj, coinSeq_lt r (hk j hjr),
        coinSeq_lt r' (hk j hjr), hs]

/-! ## Lists read position by position -/

theorem getD_take {β : Type} (l : List β) {n j : ℕ} (hj : j < n) (d : β) : (l.take n).getD j d = l.getD j d := by
  rw [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD, List.getElem?_take_of_lt hj]

theorem getD_drop {β : Type} (l : List β) (o j : ℕ) (d : β) : (l.drop o).getD j d = l.getD (o + j) d := by
  rw [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD, List.getElem?_drop]

theorem getD_dropLast {β : Type} (l : List β) {j : ℕ} (hj : j < l.length - 1) (d : β) :
    l.dropLast.getD j d = l.getD j d := by
  rw [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD, List.getElem?_dropLast, if_pos hj]

theorem eq_of_getD {β : Type} {l l' : List β} (d : β) (hl : l'.length = l.length)
    (hag : ∀ j < l.length, l'.getD j d = l.getD j d) : l' = l := by
  apply List.ext_getElem?
  intro j
  by_cases hj : j < l.length
  · have e := hag j hj
    rw [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD,
      List.getElem?_eq_getElem (show j < l'.length by omega), List.getElem?_eq_getElem hj] at e
    rw [List.getElem?_eq_getElem (show j < l'.length by omega), List.getElem?_eq_getElem hj]
    simpa using e
  · rw [List.getElem?_eq_none (show l'.length ≤ j by omega), List.getElem?_eq_none (show l.length ≤ j by omega)]

theorem getD_of_getLast?_drop {β : Type} {l : List β} {o : ℕ} {c : β} (d : β) (hc : (l.drop o).getLast? = some c) :
    l.getD (l.length - 1) d = c := by
  rw [List.getLast?_eq_getElem?, List.getElem?_drop, List.length_drop] at hc
  rw [List.getD_eq_getElem?_getD]
  by_cases hlt : o < l.length
  · rw [show o + (l.length - o - 1) = l.length - 1 by omega] at hc
    simp [hc]
  · rw [List.getElem?_eq_none (show l.length ≤ o + (l.length - o - 1) by omega)] at hc
    cases hc

/-! ## Flock's step reads its coins as `AnswersShown` needs

Flock's step at a session's messages is a coin request's coin, a refusal, or a decision. A decision needs the
session's first message to be its link (tag `0`) and its last to be final (tag `1`), so a session's first message
alone is never decided, and a step that is no coin request reads no coin but those before its last. -/

theorem linkOf_of_finalOf {a : Msg} {f : FlockDeploy.Final} (hf : FlockDeploy.finalOf a = some f) :
    FlockDeploy.linkOf a = none := by
  rcases a with _ | ⟨b, body⟩
  · first
      | cases hf
      | simp [FlockDeploy.finalOf] at hf
  · by_cases hb : b = 0
    · subst hb
      have h0 : FlockDeploy.finalOf (0 :: body) = none := by
        first
          | rfl
          | simp [FlockDeploy.finalOf]
      rw [h0] at hf
      cases hf
    · first
        | simp [FlockDeploy.linkOf, hb]
        | (unfold FlockDeploy.linkOf; split <;> simp_all)

/-- **A lone message that is no coin request is refused.** -/
theorem act_single (zk : Bool) (x : Msg) {a : Msg} (ha : ∀ t, a ≠ 0 :: t) (e : FlockDeploy.Coin) :
    FlockDeploy.act zk x [a] [e] = .rej := by
  unfold FlockDeploy.act
  rcases FlockDeploy.setupOf zk x with _ | ⟨s, spec, sts⟩
  · rfl
  · dsimp only
    try simp only [List.getLast?_singleton]
    split
    · rename_i heq _
      exact absurd (Option.some.inj heq) (ha _)
    · rename_i a₂ heq _
      have e' : a₂ = a := by cases heq; rfl
      rw [e']
      rcases hf : FlockDeploy.finalOf a with _ | f
      · rfl
      · simp only [linkOf_of_finalOf hf]
    · rename_i heq
      cases heq

theorem decidedAt_false_iff {xS : Msg} {as : List Msg} {cs : List FlockDeploy.Coin} {j : ℕ} :
    SpotFlock.decidedAt xS as cs j = false ↔ ∃ c, FlockDeploy.act true xS (as.take j) (cs.take j) = .coin c := by
  unfold SpotFlock.decidedAt
  cases FlockDeploy.act true xS (as.take j) (cs.take j) <;> simp

theorem decidedAt_take {xS : Msg} {as : List Msg} {cs : List FlockDeploy.Coin} {n j : ℕ} (hj : j ≤ n) :
    SpotFlock.decidedAt xS (as.take n) (cs.take n) j = SpotFlock.decidedAt xS as cs j := by
  unfold SpotFlock.decidedAt
  rw [List.take_take, List.take_take, Nat.min_eq_left hj]

/-- **A coin request** is a setup, a last message tagged `0`, and a coin. -/
theorem act_coin_iff {zk : Bool} {x : Msg} {as : List Msg} {cs : List FlockDeploy.Coin} :
    (∃ c, FlockDeploy.act zk x as cs = .coin c) ↔
      (FlockDeploy.setupOf zk x).isSome = true ∧ (∃ t, as.getLast? = some (0 :: t)) ∧ cs ≠ [] := by
  constructor
  · rintro ⟨c, hc⟩
    refine ⟨?_, RecDeploy.act_coin_tag hc, fun hnil => ?_⟩
    · cases hs : FlockDeploy.setupOf zk x with
      | none =>
        unfold FlockDeploy.act at hc
        rw [hs] at hc
        cases hc
      | some _ => rfl
    · have hl := Claims.Bridge.act_coin hc
      rw [hnil] at hl
      cases hl
  · rintro ⟨hs, ⟨t, ht⟩, hne⟩
    obtain ⟨init, e, rfl⟩ : ∃ init e, cs = init ++ [e] :=
      ⟨cs.dropLast, cs.getLast hne, (List.dropLast_concat_getLast hne).symm⟩
    refine ⟨e, ?_⟩
    unfold FlockDeploy.act
    rcases hsx : FlockDeploy.setupOf zk x with _ | ⟨s, spec, sts⟩
    · rw [hsx] at hs
      cases hs
    · dsimp only
      rw [ht, List.getLast?_concat]
      all_goals rfl

theorem decidedAt_congr {xS : Msg} {as : List Msg} {cs cs' : List FlockDeploy.Coin} {j : ℕ}
    (hl : cs.length = cs'.length) : SpotFlock.decidedAt xS as cs j = SpotFlock.decidedAt xS as cs' j := by
  have hnil : cs.take j = [] ↔ cs'.take j = [] := by
    rw [← List.length_eq_zero_iff, ← List.length_eq_zero_iff, List.length_take, List.length_take, hl]
  have key : (∃ c, FlockDeploy.act true xS (as.take j) (cs.take j) = .coin c) ↔
      (∃ c, FlockDeploy.act true xS (as.take j) (cs'.take j) = .coin c) := by
    rw [act_coin_iff, act_coin_iff, ne_eq, ne_eq, hnil]
  cases h1 : SpotFlock.decidedAt xS as cs j <;> cases h2 : SpotFlock.decidedAt xS as cs' j
  · rfl
  · exfalso
    have e := decidedAt_false_iff.2 (key.1 (decidedAt_false_iff.1 h1))
    rw [e] at h2
    cases h2
  · exfalso
    have e := decidedAt_false_iff.2 (key.2 (decidedAt_false_iff.1 h2))
    rw [e] at h1
    cases h1
  · rfl

/-- **A step that is no coin request reads no coin but those before its last.** -/
theorem act_eq_last {zk : Bool} {x : Msg} {As : List Msg} {Cs Cs' : List FlockDeploy.Coin}
    (hlen : Cs.length = As.length) (hl : Cs'.length = Cs.length) (hpos : 0 < Cs.length)
    (hag : ∀ j, j + 1 < Cs.length → Cs'.getD j default = Cs.getD j default)
    (hnc : ¬∃ c, FlockDeploy.act zk x As Cs = .coin c) :
    FlockDeploy.act zk x As Cs' = FlockDeploy.act zk x As Cs := by
  have hd : Cs'.dropLast = Cs.dropLast := eq_of_getD default (by simp only [List.length_dropLast, hl]) fun j hj => by
    simp only [List.length_dropLast] at hj
    rw [getD_dropLast _ (by omega), getD_dropLast _ (by omega)]
    exact hag j (by omega)
  have hne : Cs ≠ [] := List.ne_nil_of_length_pos hpos
  have hne' : Cs' ≠ [] := List.ne_nil_of_length_pos (by omega)
  obtain ⟨D, e, rfl⟩ : ∃ D e, Cs = D ++ [e] := ⟨_, _, (List.dropLast_concat_getLast hne).symm⟩
  obtain ⟨D', e', rfl⟩ : ∃ D e, Cs' = D ++ [e] := ⟨_, _, (List.dropLast_concat_getLast hne').symm⟩
  have hD : D' = D := by simpa using hd
  rw [hD]
  by_cases ha : ∃ t, As.getLast? = some (0 :: t)
  · obtain ⟨t, ht⟩ := ha
    rcases hsx : FlockDeploy.setupOf zk x with _ | ⟨s, spec, sts⟩
    · simp only [FlockDeploy.act, hsx]
    · exfalso
      exact hnc (act_coin_iff.2 ⟨by simp [hsx], ⟨t, ht⟩, by simp⟩)
  · push Not at ha
    cases D with
    | nil =>
      obtain ⟨a, rfl⟩ := List.length_eq_one_iff.1 (by simpa using hlen.symm)
      have ha' : ∀ t, a ≠ 0 :: t := fun t hat => ha t (by simp [hat])
      simp only [List.nil_append]
      rw [act_single zk x ha' e', act_single zk x ha' e]
    | cons c₀ mid => exact FlockDeploy.act_cons_snoc zk x As c₀ mid e e' ha

/-- **Flock's answer** is a refusal, an acceptance, or its last coin's bytes. -/
theorem flock_answer (H : Msg → Msg) (zk : Bool) (xS : Msg) (As : List Msg) (Cs : List FlockDeploy.Coin) :
    ((FlockDeploy.next zk xS As Cs).eval H).1 = .rej ∨ ((FlockDeploy.next zk xS As Cs).eval H).1 = .acc ∨
      ∃ c, Cs.getLast? = some c ∧ ((FlockDeploy.next zk xS As Cs).eval H).1 = .msg c.bytes := by
  unfold FlockDeploy.next
  cases hact : FlockDeploy.act zk xS As Cs with
  | rej => exact Or.inl rfl
  | coin c => exact Or.inr (Or.inr ⟨c, Claims.Bridge.act_coin hact, rfl⟩)
  | decide s spec sts link c₁ mids f =>
    dsimp only
    cases hv : ((FlockDeploy.outOf (FlockDeploy.decision zk s xS link c₁ mids f)).eval H).1 with
    | msg m => exact absurd hv (SpotFlock.outOf_ne_msg _ H m)
    | acc => exact Or.inr (Or.inl rfl)
    | rej => exact Or.inl rfl

/-! ## SPOT[Flock]'s verifier answers as `RunShown` needs -/

section Spot

open SpotFlock

variable (h : Held)

theorem slotOf_some {n t : ℕ} (ht : slotOf h n = some t) :
    t < h.ms.length ∧ off h t < n ∧ n ≤ off h t + k₀ h t := by
  unfold slotOf at ht
  have hm := List.mem_of_find?_eq_some ht
  have hp := List.find?_some ht
  simp only [decide_eq_true_eq] at hp
  exact ⟨List.mem_range.1 hm, hp.1, hp.2⟩

/-- **Every answer is the round's coin's bytes or carries no bytes.** -/
theorem answers_spot (x : Msg) (as : List Msg) (cs : List FlockDeploy.Coin) (hne : as ≠ [])
    (hcs : cs.length = as.length) :
    ((next h x as cs).eval FlockDeploy.sha512).1 = .msg (cs.getD (as.length - 1) default).bytes ∨
      msgOf ((next h x as cs).eval FlockDeploy.sha512).1 = [] := by
  by_cases hle : as.length ≤ h.stage + 1
  · rw [next_coin h x hne hcs hle]
    split_ifs <;> simp [OC.eval, msgOf]
  · by_cases hst : as.length = h.stage + 2
    · rw [next_stmt h x hcs hst]
      split_ifs <;> simp [OC.eval, msgOf]
    · rw [next_long h x as cs (by omega) hcs]
      rcases ht : slotOf h as.length with _ | t
      · dsimp only
        split_ifs <;> simp [OC.eval, msgOf]
      · dsimp only
        unfold sessStep
        split_ifs
        · right
          simp [OC.eval, msgOf]
        · rw [OC.eval_bind]
          simp only [Function.comp_apply, OC.eval]
          rcases flock_answer FlockDeploy.sha512 true (stmtAt h (as.getD (h.stage + 1) []) t)
              (as.drop (off h t)) (cs.drop (off h t)) with hF | hF | ⟨c, hc, hF⟩
          · right
            rw [hF]
            rfl
          · right
            rw [hF]
            rfl
          · left
            rw [hF, ← hcs, getD_of_getLast?_drop default hc]
            rfl

theorem answersOwn : AnswersOwn (V h) := by
  intro x as cs c hne hcs hc
  change msgOf ((next h x as cs).eval FlockDeploy.sha512).1 = c.bytes at hc
  rcases answers_spot h x as cs hne hcs with he | he
  · rw [he] at hc
    simp only [msgOf] at hc
    exact (RecDeploy.coin_bytes_injective hc).symm
  · rw [he] at hc
    exact absurd hc.symm (coin_bytes_ne c)

theorem answersShown : AnswersShown (V h) := by
  intro x as cs cs' hne hcs hcs' hcont hrev
  change ((next h x as cs').eval FlockDeploy.sha512).1 = ((next h x as cs).eval FlockDeploy.sha512).1
  change ∀ j < as.length - 1, ∃ m,
    ((next h x (as.take (j + 1)) (cs.take (j + 1))).eval FlockDeploy.sha512).1 = .msg m at hcont
  change ∀ j < as.length, msgOf ((next h x (as.take (j + 1)) (cs.take (j + 1))).eval FlockDeploy.sha512).1 =
    (cs.getD j default).bytes → cs'.getD j default = cs.getD j default at hrev
  have h0 : 0 < as.length := List.length_pos_iff.2 hne
  -- the last round's coin, where it answers with it
  have hlast : msgOf ((next h x as cs).eval FlockDeploy.sha512).1 = (cs.getD (as.length - 1) default).bytes →
      cs'.getD (as.length - 1) default = cs.getD (as.length - 1) default := by
    intro hm
    refine hrev (as.length - 1) (by omega) ?_
    rwa [Nat.sub_add_cancel h0, List.take_of_length_le le_rfl, List.take_of_length_le (le_of_eq hcs)]
  -- a round up to the first stage's last answers its coin, which then agrees
  have hpre : ∀ j, j < as.length - 1 → j + 1 ≤ h.stage + 1 → cs'.getD j default = cs.getD j default := by
    intro j hj hjs
    obtain ⟨m, hm⟩ := hcont j hj
    have hl : (as.take (j + 1)).length = j + 1 := by simp only [List.length_take]; omega
    have hne' : as.take (j + 1) ≠ [] := List.ne_nil_of_length_pos (by omega)
    have hcl : (cs.take (j + 1)).length = (as.take (j + 1)).length := by simp only [List.length_take]; omega
    refine hrev j (by omega) ?_
    rw [next_coin h x hne' hcl (by rw [hl]; exact hjs)] at hm ⊢
    split_ifs at hm ⊢
    · simp only [OC.eval, msgOf]
      rw [hl, Nat.add_sub_cancel, getD_take _ (Nat.lt_add_one j)]
    · simp [OC.eval] at hm
  have hdec : ∀ (xS : Msg) (o j : ℕ),
      decidedAt xS (as.drop o) (cs'.drop o) j = decidedAt xS (as.drop o) (cs.drop o) j :=
    fun xS o j => decidedAt_congr (by simp only [List.length_drop, hcs, hcs'])
  by_cases hle : as.length ≤ h.stage + 1
  · rw [next_coin h x hne hcs' hle, next_coin h x hne hcs hle]
    split_ifs with hc
    · have e := hlast (by rw [next_coin h x hne hcs hle, if_pos hc]; simp only [OC.eval, msgOf])
      rw [e]
    · rfl
  by_cases hst : as.length = h.stage + 2
  · rw [next_stmt h x hcs' hst, next_stmt h x hcs hst]
    have h0' : cs'.getD 0 default = cs.getD 0 default := hpre 0 (by omega) (by omega)
    have hstage : stageOf h cs' = stageOf h cs := by
      funext i
      have := i.isLt
      exact hpre (i.val + 1) (by omega) (by omega)
    rw [h0', hstage]
  rw [next_long h x as cs' (by omega) hcs', next_long h x as cs (by omega) hcs]
  rcases ht : slotOf h as.length with _ | t
  · have hv : ∀ m₂, verdict h m₂ as cs' = verdict h m₂ as cs := fun m₂ => by
      unfold verdict
      simp only [hdec]
    dsimp only
    rw [hv]
  · dsimp only
    obtain ⟨htl, hot, hok⟩ := slotOf_some h ht
    have hso := stage_le_off h t
    generalize hxS : stmtAt h (as.getD (h.stage + 1) []) t = xS
    by_cases hd : ∃ j, 0 < j ∧ j < as.length - off h t ∧
        decidedAt xS (as.drop (off h t)) (cs.drop (off h t)) j = true
    · obtain ⟨j, hj0, hj, hdj⟩ := hd
      have hany : ∀ C : List FlockDeploy.Coin, decidedAt xS (as.drop (off h t)) (C.drop (off h t)) j = true →
          sessStep h t xS as C = .ret (.msg []) := by
        intro C hC
        unfold sessStep
        rw [if_pos]
        exact List.any_eq_true.2 ⟨j, List.mem_range.2 hj, by simp [hj0, hC]⟩
      rw [hany cs hdj, hany cs' ((hdec _ _ _).trans hdj)]
    have hund : ∀ j, 0 < j → j < as.length - off h t →
        decidedAt xS (as.drop (off h t)) (cs.drop (off h t)) j = false := fun j hj0 hj => by
      cases hdj : decidedAt xS (as.drop (off h t)) (cs.drop (off h t)) j
      · rfl
      · exact absurd ⟨j, hj0, hj, hdj⟩ hd
    rw [sessStep_eval h _ t xS as cs' (fun j hj0 hj => (hdec xS _ j).trans (hund j hj0 hj)),
      sessStep_eval h _ t xS as cs hund]
    dsimp only
    congr 1
    -- the session's earlier rounds were coin requests, so they showed their coins
    have hmid : ∀ j, off h t ≤ j → j + 1 < as.length → cs'.getD j default = cs.getD j default := by
      intro j hoj hj
      obtain ⟨c, hc⟩ := decidedAt_false_iff.1 (hund (j + 1 - off h t) (by omega) (by omega))
      have hl : (as.take (j + 1)).length = j + 1 := by simp only [List.length_take]; omega
      have hcl : (cs.take (j + 1)).length = (as.take (j + 1)).length := by simp only [List.length_take]; omega
      have hAs : (as.take (j + 1)).drop (off h t) = (as.drop (off h t)).take (j + 1 - off h t) := List.drop_take
      have hCs : (cs.take (j + 1)).drop (off h t) = (cs.drop (off h t)).take (j + 1 - off h t) := List.drop_take
      have hund' : ∀ j', 0 < j' → j' < (as.take (j + 1)).length - off h t →
          decidedAt xS ((as.take (j + 1)).drop (off h t)) ((cs.take (j + 1)).drop (off h t)) j' = false := by
        intro j' hj0 hj'
        rw [hl] at hj'
        rw [hAs, hCs, decidedAt_take (show j' ≤ j + 1 - off h t by omega)]
        exact hund j' hj0 (by omega)
      have hcj : cs.getD j default = c := by
        have e := getD_of_getLast?_drop (o := off h t) default (by rw [hCs]; exact Claims.Bridge.act_coin hc)
        rwa [hcl, hl, Nat.add_sub_cancel, getD_take _ (Nat.lt_add_one j)] at e
      refine hrev j (by omega) ?_
      rw [next_long h x (as.take (j + 1)) (cs.take (j + 1)) (by rw [hl]; omega) hcl, hl,
        slotOf_eq h htl (show off h t < j + 1 by omega) (show j + 1 ≤ off h t + k₀ h t by omega)]
      dsimp only
      rw [getD_take _ (show h.stage + 1 < j + 1 by omega), hxS, sessStep_eval h _ t xS _ _ hund']
      dsimp only
      rw [hAs, hCs]
      unfold FlockDeploy.next
      rw [hc]
      simp only [OC.eval, Claims.Sampling.markOut, msgOf]
      rw [hcj]
    by_cases hcoin : ∃ c, FlockDeploy.act true xS (as.drop (off h t)) (cs.drop (off h t)) = .coin c
    · obtain ⟨c, hc⟩ := hcoin
      have hlastc : cs'.getD (as.length - 1) default = cs.getD (as.length - 1) default := by
        apply hlast
        rw [next_long h x as cs (by omega) hcs, ht]
        dsimp only
        rw [hxS, sessStep_eval h _ t xS as cs hund]
        dsimp only
        unfold FlockDeploy.next
        rw [hc]
        simp only [OC.eval, Claims.Sampling.markOut, msgOf]
        rw [← hcs, getD_of_getLast?_drop default (Claims.Bridge.act_coin hc)]
      have hall : cs'.drop (off h t) = cs.drop (off h t) := by
        refine eq_of_getD default (by simp only [List.length_drop, hcs, hcs']) fun j hj => ?_
        rw [getD_drop, getD_drop]
        simp only [List.length_drop] at hj
        by_cases hjl : off h t + j + 1 < as.length
        · exact hmid _ (by omega) hjl
        · rw [show off h t + j = as.length - 1 by omega]
          exact hlastc
      rw [hall]
    · have hag : ∀ j, j + 1 < (cs.drop (off h t)).length →
          (cs'.drop (off h t)).getD j default = (cs.drop (off h t)).getD j default := by
        intro j hj
        simp only [List.length_drop] at hj
        rw [getD_drop, getD_drop]
        exact hmid _ (by omega) (by omega)
      unfold FlockDeploy.next
      rw [act_eq_last (by simp only [List.length_drop, hcs]) (by simp only [List.length_drop, hcs, hcs'])
        (by simp only [List.length_drop, hcs]; omega) hag hcoin]

end Spot

/-- **The two-stage audit's verifier** is SPOT[Flock]'s. -/
theorem L_V (A : Live₂) : A.L.V = SpotFlock.V A.h := rfl

/-- **SPOT[Flock]'s runs show their coins.** -/
theorem runShown (A : Live₂) (x : Msg) : RunShown A x := by
  have hO : AnswersOwn A.L.V := by rw [L_V]; exact answersOwn A.h
  have hS : AnswersShown A.L.V := by rw [L_V]; exact answersShown A.h
  exact runShown_of hO hS

/-- **`MessagesFixed` at the frame's prover**, at the sum over the rounds of the square roots of their finders'
successes: where the message the check reads opens the digest the play records, each round's digest is fixed by the
coins the history before it shows, and the run reaches every round. -/
theorem messagesFixed {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α] (V : Vocab d α) (A : Live₂)
    (x : Msg) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c))
    (place : TapeOf V θ F c σ → Coins A x → TapeOf V θ F c σ) (Rd : Reading A) (Acc Good : Verity.Exchange → Prop)
    (Dg : Verity.Exchange → ℕ → Option Msg) (q : ℕ) (hO : MsgOpens A x Rd Acc Good Dg)
    (hB : ∀ i < A.L.V.k x, DigestBefore (frame V A x θ F c σ place Rd Dg) Rd Acc Good Dg i)
    (hT : RunsThrough (frame V A x θ F c σ place Rd Dg) Rd Acc Good) :
    MessagesFixed V A x θ F c σ (frame V A x θ F c σ place Rd Dg) Rd Acc Good
      (∑ i ∈ Finset.range (A.L.V.k x),
        ENNReal.ofReal (Real.sqrt ((msgFinder V θ F c σ place Rd Dg i q).Pr Verity.sha512Msg))) :=
  messagesBound V A x θ F c σ place Rd Acc Good Dg q hO hB (runShown A x) hT

end Verity.SampledTwo.Messages
