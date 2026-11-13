module

public import Proofs.Properties.Verity.SampledTwo.Messages
public import Proofs.Flock.Claims.Recursion.DeployFacts

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
that answer alike on the messages one run of them actually sends give the same rounds.
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
  | 0, _, _, t, j, h => by simp [stepWith] at h
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
        have := stepWith_msg_lt vn f _ _ { t with msgs := t.msgs ++ [(((P.next x hist ω).eval H).1, .msg mm)],
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
        simp [hnew]
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
  | some e => rw [he] at h; exact ⟨e, rfl, h⟩

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
    rw [runHist_of hji he]
    show shownCoin (Claims.Recursion.outMsg e.2) = some (r ⟨j, hk j hj⟩)
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
      (List.ne_nil_of_length_pos (by simp only [List.length_map, List.length_take]; omega))
      (by simp only [List.length_map, List.length_range, List.length_take]; omega) hb.symm
    rw [this, List.length_map, List.length_take,
      Nat.min_eq_left (show j.val + 1 ≤ (Claims.run FlockDeploy.sha512 P A.L.V x ω r).msgs.length by omega),
      Nat.add_sub_cancel, getD_range_map _ (Nat.lt_add_one j.val), coinSeq_lt r j.isLt]
  · refine run_agree_on FlockDeploy.sha512 P A.L.V x ω r r' i fun l hl hlr => ?_
    rw [hasEq l hlr]
    have hlen : (((Claims.run FlockDeploy.sha512 P A.L.V x ω r).msgs.take (l + 1)).map Prod.fst).length = l + 1 := by
      simp only [List.length_map, List.length_take]
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

end Verity.SampledTwo.Messages
