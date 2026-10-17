import Proofs.Flock.Claims.KS.PlainSim

/-!
# The plain simulator's lockstep

`plainSim P x spec st` runs in lockstep with `P`'s deployed run at `plainΦ`'s coins.
-/

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Discharge.Tables Claims.Bridge.Live

local notation "𝔽" => FlockLevel3.GF128

/-! ## Runs, message by message -/

/-- **Acceptance reads the verifier's last answer only for acceptance**: two verifiers that agree before the `K`-th
message and agree on accepting at it accept the same runs of `K` rounds. -/
theorem stepWith_acc_iff {Ω : Type} (H : Claims.Msg → Claims.Msg) (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω)
    (v₁ v₂ : List Claims.Msg → Claims.OC Claims.Out) (K : ℕ)
    (hlt : ∀ bs : List Claims.Msg, bs.length < K → v₁ bs = v₂ bs)
    (heq : ∀ bs : List Claims.Msg, bs.length = K → (((v₁ bs).eval H).1 = .acc ↔ ((v₂ bs).eval H).1 = .acc)) :
    ∀ (n : ℕ) (hist : List (Claims.Msg × Claims.Msg)) (as : List Claims.Msg) (t : Claims.Trace),
      as.length + n = K →
      ((Claims.stepWith H P x ω v₁ n hist as t).out = .acc ↔ (Claims.stepWith H P x ω v₂ n hist as t).out = .acc) := by
  intro n
  induction n with
  | zero => intro hist as t _; simp [Claims.stepWith]
  | succ n ih =>
    intro hist as t hK
    rcases Nat.lt_or_ge (as.length + 1) K with hl | hl
    · simp only [Claims.stepWith]
      rw [hlt _ (by simp; omega)]
      cases hv : ((v₂ (as ++ [((P.next x hist ω).eval H).1])).eval H).1 with
      | msg m => exact ih _ _ _ (by simp; omega)
      | acc => simp
      | rej => simp
    · obtain rfl : n = 0 := by omega
      have e := heq (as ++ [((P.next x hist ω).eval H).1]) (by simp; omega)
      simp only [Claims.stepWith]
      cases h₁ : ((v₁ (as ++ [((P.next x hist ω).eval H).1])).eval H).1 <;>
        cases h₂ : ((v₂ (as ++ [((P.next x hist ω).eval H).1])).eval H).1 <;> simp_all

theorem range_map_ends {α : Type} (f : ℕ → α) {K : ℕ} (hK : 2 ≤ K) :
    (List.range K).map f = f 0 :: ((List.range (K - 2)).map (fun i => f (i + 1)) ++ [f (K - 1)]) := by
  obtain ⟨n, rfl⟩ : ∃ n, K = n + 2 := ⟨K - 2, by omega⟩
  rw [List.range_succ, List.range_succ_eq_map]
  simp

/-- **The deployed verifier decides without its last coin**: at coins that differ in the last only, a decision at one
is the same decision at the other. -/
theorem act_last_decide {zk : Bool} {x : Claims.Msg} {as : List Claims.Msg} {c₀ : Coin} {l : List Coin} {c : Coin}
    {s : Ins} {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : Final}
    (h : act zk x as (c₀ :: (l ++ [c])) = .decide s spec sts link c₁ mids f) (c' : Coin) :
    act zk x as (c₀ :: (l ++ [c'])) = .decide s spec sts link c₁ mids f := by
  have e₁ : (c₀ :: (l ++ [c])).getLast? = some c := by rw [← List.cons_append, List.getLast?_concat]
  have e₂ : (c₀ :: (l ++ [c'])).getLast? = some c' := by rw [← List.cons_append, List.getLast?_concat]
  have d₁ : (l ++ [c]).dropLast = l := List.dropLast_concat ..
  have d₂ : (l ++ [c']).dropLast = l := List.dropLast_concat ..
  unfold act at h ⊢
  split at h
  · cases h
  · rename_i hs
    rw [e₁] at h
    rw [e₂]
    split at h
    · cases h
    · rename_i a hg hne
      rw [hg]
      split
      · rename_i tail _ heq _
        cases heq
        exact (hne tail c rfl rfl).elim
      · rename_i heq _
        cases heq
        rcases hf : finalOf a with _ | f'
        · rw [hf] at h; cases h
        · rw [hf] at h
          rcases as with _ | ⟨a₁, rest⟩
          · cases h
          · dsimp only at h ⊢
            rw [d₁] at h
            rw [d₂]
            exact h
      · rename_i heq
        cases heq
    · cases h

/-- **The deployed verifier accepts without its last coin.** -/
theorem next_acc_last {zk : Bool} (H : Claims.Msg → Claims.Msg) {x : Claims.Msg} {as : List Claims.Msg} (c₀ : Coin)
    (l : List Coin) (c c' : Coin) :
    ((next zk x as (c₀ :: (l ++ [c]))).eval H).1 = .acc ↔ ((next zk x as (c₀ :: (l ++ [c']))).eval H).1 = .acc := by
  have key : ∀ c c' : Coin, ((next zk x as (c₀ :: (l ++ [c]))).eval H).1 = .acc →
      ((next zk x as (c₀ :: (l ++ [c']))).eval H).1 = .acc := by
    intro c c' h
    cases ha : act zk x as (c₀ :: (l ++ [c])) with
    | rej => rw [next_rej H ha] at h; cases h
    | coin d => rw [next_coin H ha] at h; cases h
    | decide s spec sts link c₁ mids f =>
      have hb := act_last_decide ha c'
      unfold next at h ⊢
      rw [ha] at h
      rw [hb]
      exact h
  exact ⟨key c c', key c' c⟩

theorem act_none {zk : Bool} {x : Claims.Msg} (hs : setupOf zk x = none) (as : List Claims.Msg) (cs : List Coin) :
    act zk x as cs = .rej := by
  unfold act
  rw [hs]

/-- **A deployed run's acceptance does not read its last coin.** -/
theorem run_acc_iff_last {zk : Bool} (H : Claims.Msg → Claims.Msg) {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg)
    (ω : Ω) (r₁ r₂ : Fin ((V zk).k x) → Coin)
    (h : ∀ i, i + 1 < (V zk).k x → Claims.coinSeq r₁ i = Claims.coinSeq r₂ i) :
    (Claims.run H P (V zk) x ω r₁).out = .acc ↔ (Claims.run H P (V zk) x ω r₂).out = .acc := by
  unfold Claims.run
  refine stepWith_acc_iff H P x ω _ _ ((V zk).k x) (fun bs hb => ?_) (fun bs hb => ?_) _ [] [] _ (by simp)
  · show next zk x bs _ = next zk x bs _
    congr 1
    exact List.map_congr_left fun i hi => h i (by have := List.mem_range.1 hi; omega)
  · show ((next zk x bs _).eval H).1 = .acc ↔ ((next zk x bs _).eval H).1 = .acc
    cases hs : setupOf zk x with
    | none =>
      rw [next_rej H (act_none hs _ _), next_rej H (act_none hs _ _)]
    | some p =>
      obtain ⟨s', spec, sts⟩ := p
      have hK : 2 ≤ (V zk).k x := by
        show 2 ≤ (match setupOf zk x with | some (_, spec, sts) => 2 + rounds zk spec sts | none => 1)
        rw [hs]
        exact Nat.le_add_right 2 _
      rw [hb, range_map_ends _ hK, range_map_ends _ hK, h 0 (by omega),
        List.map_congr_left fun i hi => h (i + 1) (by have := List.mem_range.1 hi; omega)]
      exact next_acc_last H _ _ _ _

/-- **A run's messages are `P`'s along its coins**: each message `P` sends at the history of the coins' bytes
before it (`depMsg`), wherever the verifier answers each round with its coin's bytes. -/
theorem stepWith_dep {Ω : Type} (H : Claims.Msg → Claims.Msg) (P : Claims.Prover Ω) (V' : Claims.Verifier Coin)
    (x : Claims.Msg) (ω : Ω) (c : ℕ → Coin) (hV : Claims.Bridge.AnswersCoin H V' x Coin.bytes) :
    ∀ (n : ℕ) (hist : List (Claims.Msg × Claims.Msg)) (as : List Claims.Msg) (t : Claims.Trace),
      hist = depHist H P x ω c as.length → as = (List.range as.length).map (depMsg H P x ω c) →
      t.msgs.map Prod.fst = as →
      (Claims.stepWith H P x ω (fun as => V'.next x as ((List.range as.length).map c)) n hist as t).msgs.map
          Prod.fst =
        (List.range (Claims.stepWith H P x ω (fun as => V'.next x as ((List.range as.length).map c)) n hist as
          t).msgs.length).map (depMsg H P x ω c) := by
  intro n
  induction n with
  | zero =>
    intro hist as t _ ha ht
    show t.msgs.map Prod.fst = (List.range t.msgs.length).map _
    rw [← List.length_map (f := Prod.fst), ht]
    exact ha
  | succ n ih =>
    intro hist as t hh ha ht
    have hp : ((P.next x hist ω).eval H).1 = depMsg H P x ω c as.length := by rw [hh]; rfl
    have ha' : as ++ [((P.next x hist ω).eval H).1] =
        (List.range (as.length + 1)).map (depMsg H P x ω c) := by
      rw [hp, List.range_succ, List.map_append, ← ha]
      rfl
    have hl' : (as ++ [((P.next x hist ω).eval H).1]).length = as.length + 1 := by
      rw [List.length_append, List.length_singleton]
    simp only [Claims.stepWith]
    cases hv : ((V'.next x (as ++ [((P.next x hist ω).eval H).1])
        ((List.range (as ++ [((P.next x hist ω).eval H).1]).length).map c)).eval H).1 with
    | msg m =>
      have hm : m = (c as.length).bytes := by
        rw [hl', List.range_succ, List.map_append] at hv
        exact hV _ _ _ _ hv
      refine ih _ _ _ ?_ ?_ ?_
      · rw [hl', hm, hh]
        rfl
      · rw [hl']
        exact ha'
      · rw [List.map_append, ht]
        rfl
    | acc =>
      show (t.msgs ++ _).map Prod.fst = (List.range (t.msgs ++ _).length).map _
      rw [List.length_append, List.length_singleton, ← List.length_map (f := Prod.fst), ht, List.map_append, ht]
      exact ha'
    | rej =>
      show (t.msgs ++ _).map Prod.fst = (List.range (t.msgs ++ _).length).map _
      rw [List.length_append, List.length_singleton, ← List.length_map (f := Prod.fst), ht, List.map_append, ht]
      exact ha'

/-- **A deployed run's messages are `P`'s along its coins** (`stepWith_dep`). -/
theorem run_dep {zk : Bool} (H : Claims.Msg → Claims.Msg) {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω)
    (r : Fin ((V zk).k x) → Coin) :
    (Claims.run H P (V zk) x ω r).msgs.map Prod.fst =
      (List.range (Claims.run H P (V zk) x ω r).msgs.length).map (depMsg H P x ω (Claims.coinSeq r)) :=
  stepWith_dep H P (V zk) x ω _ (Claims.Bridge.answersCoin_deploy zk H x) _ [] [] _ rfl rfl rfl

theorem depHist_congr {Ω : Type} (H : Claims.Msg → Claims.Msg) (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω)
    {c₁ c₂ : ℕ → Coin} : ∀ j, (∀ i < j, c₁ i = c₂ i) → depHist H P x ω c₁ j = depHist H P x ω c₂ j
  | 0, _ => rfl
  | j + 1, h => by
    have e := depHist_congr H P x ω j fun i hi => h i (by omega)
    simp only [depHist]
    rw [e, h j (by omega)]

/-- **`P`'s message after `j` rounds reads the coins before `j` only.** -/
theorem depMsg_congr {Ω : Type} (H : Claims.Msg → Claims.Msg) (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω)
    {c₁ c₂ : ℕ → Coin} {j : ℕ} (h : ∀ i < j, c₁ i = c₂ i) : depMsg H P x ω c₁ j = depMsg H P x ω c₂ j := by
  unfold depMsg
  rw [depHist_congr H P x ω j h]

end FlockDeploy.KS
