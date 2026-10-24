import Proofs.Flock.Claims.KS.PlainSim
import Proofs.Flock.Claims.Recursion.PlainK

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

/-- **`VK keep`'s acceptance does not read its last coin** (`next_acc_last`'s proof, at `nextK keep`). -/
theorem nextK_acc_last (keep : Bool → Bool) (H : Claims.Msg → Claims.Msg) {x : Claims.Msg} {as : List Claims.Msg}
    (c₀ : Coin) (l : List Coin) (c c' : Coin) :
    ((nextK keep x as (c₀ :: (l ++ [c]))).eval H).1 = .acc ↔
      ((nextK keep x as (c₀ :: (l ++ [c']))).eval H).1 = .acc := by
  have key : ∀ c c' : Coin, ((nextK keep x as (c₀ :: (l ++ [c]))).eval H).1 = .acc →
      ((nextK keep x as (c₀ :: (l ++ [c']))).eval H).1 = .acc := by
    intro c c' h
    cases ha : act false x as (c₀ :: (l ++ [c])) with
    | rej => rw [nextK_rej keep H ha] at h; cases h
    | coin d => rw [nextK_coin keep H ha] at h; cases h
    | decide s spec sts link c₁ mids f =>
      have hb := act_last_decide ha c'
      unfold nextK at h ⊢
      rw [ha] at h
      rw [hb]
      exact h
  exact ⟨key c c', key c' c⟩

/-- **A run of `VK keep` accepts regardless of its last coin** (`run_acc_iff_last`'s proof, at `nextK keep`). -/
theorem run_acc_iff_lastK (keep : Bool → Bool) (H : Claims.Msg → Claims.Msg) {Ω : Type} (P : Claims.Prover Ω)
    (x : Claims.Msg) (ω : Ω) (r₁ r₂ : Fin ((V false).k x) → Coin)
    (h : ∀ i, i + 1 < (V false).k x → Claims.coinSeq r₁ i = Claims.coinSeq r₂ i) :
    (Claims.run H P (VK keep) x ω r₁).out = .acc ↔ (Claims.run H P (VK keep) x ω r₂).out = .acc := by
  replace h : ∀ i, i + 1 < (V false).k x →
      Claims.coinSeq (k := (VK keep).k x) r₁ i = Claims.coinSeq (k := (VK keep).k x) r₂ i := h
  unfold Claims.run
  refine stepWith_acc_iff H P x ω _ _ ((V false).k x) (fun bs hb => ?_) (fun bs hb => ?_) _ [] [] _
    (by rw [List.length_nil, Nat.zero_add]; exact congrFun (VK_k keep) x)
  · show nextK keep x bs _ = nextK keep x bs _
    congr 1
    exact List.map_congr_left fun i hi => h i (by have := List.mem_range.1 hi; omega)
  · show ((nextK keep x bs _).eval H).1 = .acc ↔ ((nextK keep x bs _).eval H).1 = .acc
    cases hs : setupOf false x with
    | none =>
      rw [nextK_rej keep H (act_none hs _ _), nextK_rej keep H (act_none hs _ _)]
    | some p =>
      obtain ⟨s', spec, sts⟩ := p
      have hK : 2 ≤ (V false).k x := by
        show 2 ≤ (match setupOf false x with | some (_, spec, sts) => 2 + rounds false spec sts | none => 1)
        rw [hs]
        exact Nat.le_add_right 2 _
      rw [hb, range_map_ends _ hK, range_map_ends _ hK, h 0 (by omega),
        List.map_congr_left fun i hi => h (i + 1) (by have := List.mem_range.1 hi; omega)]
      exact nextK_acc_last keep H _ _ _ _

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

/-- **A run of `VK keep`'s messages are `P`'s along its coins** (`stepWith_dep`). -/
theorem run_depK (keep : Bool → Bool) (H : Claims.Msg → Claims.Msg) {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg)
    (ω : Ω) (r : Fin ((V false).k x) → Coin) :
    (Claims.run H P (VK keep) x ω r).msgs.map Prod.fst =
      (List.range (Claims.run H P (VK keep) x ω r).msgs.length).map (depMsg H P x ω (Claims.coinSeq r)) :=
  stepWith_dep H P (VK keep) x ω _ (Claims.Bridge.answersCoin_VK keep H x) _ [] [] _ rfl rfl rfl

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

/-! ## The simulator's coins on the live history -/

/-- **`fnC`'s coin `j` reads the coins at `i ≤ j` only at their revealed entries.** -/
theorem fnC_agree {K M : ℕ} {α D : Type} {n : Fin K → ℕ} {key : Fin K → (Fin K → Fin M → α) → D}
    (hk : Claims.Flock.FnFork.PrefixKey n key) (f : Fin K → D → Fin M → α) {r₁ r₂ : Fin K → Fin M → α} (j : Fin K)
    (h : ∀ i, i ≤ j → Claims.Flock.FnFork.AgreeAt (n i) (r₁ i) (r₂ i)) :
    fnC n key f r₁ j = fnC n key f r₂ j := by
  funext i
  unfold fnC mergeC
  dsimp only
  rw [hk.read j r₁ r₂ h]
  split_ifs with hi
  · exact h j le_rfl i hi
  · rfl

theorem liveH_length {J mPts : ℕ} (w : Wire 65536 J mPts) :
    ∀ (k : ℕ) (b : ℕ → Claims.Msg) (S : ℕ → SubmitJ J) (r : ℕ → LCoin 65536), (liveH w b S r k).length = k
  | 0, _, _, _ => rfl
  | k + 1, b, S, r => by
    simp only [liveH, List.length_cons]
    rw [liveH_length w k]

theorem liveH_getD {J mPts : ℕ} (w : Wire 65536 J mPts) :
    ∀ (k : ℕ) (b : ℕ → Claims.Msg) (S : ℕ → SubmitJ J) (r : ℕ → LCoin 65536) (i : ℕ), i < k →
      (liveH w b S r k).getD i default = (b i, w.encRd (S i).n (takeE (S i).n (r i)))
  | 0, _, _, _, i, hi => absurd hi (Nat.not_lt_zero _)
  | k + 1, b, S, r, 0, _ => rfl
  | k + 1, b, S, r, i + 1, hi => by
    simp only [liveH, List.getD_cons_succ]
    exact liveH_getD w k _ _ _ i (by omega)

/-- Deployed round `i`'s rep is `0` or `1`. -/
theorem plainRep_lt (spec : SessionSpec) (st : Stmt) (i : ℕ) : plainRep spec st i < 2 := by
  unfold plainRep
  split_ifs <;> omega

/-- **The simulator's submission `k`**: deployed round `k + 1`'s rep and count, and the bytes of `P`'s message
`dm (k + 1)` after its tag. -/
def plainS (x : Claims.Msg) (spec : SessionSpec) (st : Stmt) (dm : ℕ → Claims.Msg) (k : ℕ) : SubmitJ 1 :=
  ⟨0, ⟨plainRep spec st (k + 1), plainRep_lt spec st (k + 1)⟩, ⟨(dm (k + 1)).tail.toArray⟩, plainCnt x spec st (k + 1)⟩

/-- **The live history after `k` submissions**, at the first message `a₀` and the messages `b`. -/
noncomputable def plainHl {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (spec : SessionSpec) (st : Stmt) (ω : Ω)
    (hw : 2 * spec.mPts ≤ 65536) (dm b : ℕ → Claims.Msg) (r' : Fin (plainN x + 2) → LCoin 65536) (k : ℕ) :
    List (Claims.Msg × Claims.Msg) :=
  [(((P.next x [] ω).eval sha512).1,
      (plainWire spec.mPts hw).encPts ((ptsIf (plainWire spec.mPts hw)).proj (Claims.coinSeq r' 0)))] ++
    liveH (plainWire spec.mPts hw) b (plainS x spec st dm) (fun i => Claims.coinSeq r' (i + 1)) k

theorem plainHl_length {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (spec : SessionSpec) (st : Stmt) (ω : Ω)
    (hw : 2 * spec.mPts ≤ 65536) (dm b : ℕ → Claims.Msg) (r' : Fin (plainN x + 2) → LCoin 65536) (k : ℕ) :
    (plainHl P x spec st ω hw dm b r' k).length = k + 1 := by
  rw [plainHl, List.length_append, liveH_length, List.length_singleton, Nat.add_comm]

theorem coinSeq_cast {K L : ℕ} {β : Type} [Inhabited β] (hL : L = K) (r : Fin L → β) (i : Fin K) :
    Claims.coinSeq r i.val = r (Fin.cast hL.symm i) := by
  unfold Claims.coinSeq
  rw [dite_eq_left (by omega)]
  rfl

/-- **The revealed entries along the live history are the live coins'**: before `k + 1`, the simulator reads back
the first `plainNOf` entries of each live coin. -/
theorem plainRecon_agree {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (spec : SessionSpec) (st : Stmt) (ω : Ω)
    (hw : 2 * spec.mPts ≤ 65536) (hm1 : 1 ≤ spec.mPts) (dm b : ℕ → Claims.Msg) (r' : Fin (plainN x + 2) → LCoin 65536)
    (hL : plainN x + 2 = (V false).k x) (k : ℕ) (i : Fin ((V false).k x)) (hi : i.val < k + 1) :
    Claims.Flock.FnFork.AgreeAt (plainNOf x spec st i) (plainRecon x spec st (plainHl P x spec st ω hw dm b r' k) i)
      (r' (Fin.cast hL.symm i)) := by
  intro e he
  have hlen := plainHl_length P x spec st ω hw dm b r' k
  unfold plainRecon
  rw [ite_eq_left (by omega), ← coinSeq_cast hL r' i]
  rcases hi0 : i.val with _ | i'
  · have hg : (plainHl P x spec st ω hw dm b r' k).getD 0 default = (((P.next x [] ω).eval sha512).1,
        finCode (ptsE spec.mPts (takeE (2 * spec.mPts) (Claims.coinSeq r' 0)))) := by
      show (_, (plainWire spec.mPts hw).encPts ((ptsIf (plainWire spec.mPts hw)).proj (Claims.coinSeq r' 0))) = _
      rw [ptsIf_proj]
      rfl
    have hn : plainNOf x spec st i ≤ 2 * spec.mPts := by
      unfold plainNOf plainCnt
      rw [hi0, ite_eq_left rfl]
      omega
    rw [hg]
    unfold reveal
    rw [ite_eq_left rfl, finDecode_code, Equiv.symm_apply_apply, padE_takeE _ _ _ (by omega)]
  · have hg : (plainHl P x spec st ω hw dm b r' k).getD (i' + 1) default = (b i',
        finCode (takeE (plainCnt x spec st (i' + 1)) (Claims.coinSeq r' (i' + 1)))) := by
      unfold plainHl
      rw [List.singleton_append, List.getD_cons_succ, liveH_getD _ _ _ _ _ _ (by omega)]
      rfl
    have hn : plainNOf x spec st i = plainCnt x spec st (i' + 1) := by
      unfold plainNOf
      rw [hi0]
    rw [hg]
    unfold reveal
    rw [ite_eq_right (Nat.succ_ne_zero i'), finDecode_code, padE_takeE _ _ _ (by omega)]

/-- **The simulator's coins are `plainΦ`'s at every deployed coin the live history has revealed.** -/
theorem simCoins_agree {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (spec : SessionSpec) (st : Stmt) (ω : Ω)
    (hw : 2 * spec.mPts ≤ 65536) (hm1 : 1 ≤ spec.mPts) (dm b : ℕ → Claims.Msg) (u : plainTape x)
    (r' : Fin (plainN x + 2) → LCoin 65536) (hL : plainN x + 2 = (V false).k x) (k : ℕ) (i : ℕ) (hi : i < k + 1) :
    Claims.coinSeq (simCoins x spec st u (plainHl P x spec st ω hw dm b r' k)) i =
      Claims.coinSeq (plainΦ x spec st (u, r')) i := by
  unfold Claims.coinSeq
  split_ifs with hK
  · rw [plainΦ_eq hL]
    exact fnC_agree (Claims.Flock.FnFork.prefixKey_histKey _) u ⟨i, hK⟩ fun j hj =>
      plainRecon_agree P x spec st ω hw hm1 dm b r' hL k j (lt_of_le_of_lt (Fin.le_def.1 hj) hi)
  · rfl

/-- **The simulator's messages**: deployed round `k + 1`'s on the plain wire before the last, and at the last `P`'s
last message where `V false` accepts at `plainΦ`, else nothing. -/
noncomputable def plainB (x : Claims.Msg) (spec : SessionSpec) (st : Stmt) (dm : ℕ → Claims.Msg) (acc : Bool) (k : ℕ) :
    Claims.Msg :=
  if k < plainN x then plainRound (plainRep spec st (k + 1)) (plainCnt x spec st (k + 1)) (dm (k + 1))
  else if acc then dm (k + 1) else []

/-- `P`'s messages at `plainΦ`'s coins. -/
noncomputable def plainDm {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (spec : SessionSpec) (st : Stmt) (ω : Ω)
    (u : plainTape x) (r' : Fin (plainN x + 2) → LCoin 65536) : ℕ → Claims.Msg :=
  depMsg sha512 P x ω (Claims.coinSeq (plainΦ x spec st (u, r')))

/-- Whether `V false` accepts `P` at `plainΦ`'s coins. -/
noncomputable def plainAcc {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (spec : SessionSpec) (st : Stmt) (ω : Ω)
    (u : plainTape x) (r' : Fin (plainN x + 2) → LCoin 65536) : Bool :=
  decide ((Claims.run sha512 P (V false) x ω (plainΦ x spec st (u, r'))).out = .acc)

/-- Whether `VK keep` accepts `P` at `plainΦ`'s coins. -/
noncomputable def plainAccK (keep : Bool → Bool) {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (spec : SessionSpec) (st : Stmt) (ω : Ω)
    (u : plainTape x) (r' : Fin (plainN x + 2) → LCoin 65536) : Bool :=
  decide ((Claims.run sha512 P (VK keep) x ω (plainΦ x spec st (u, r'))).out = .acc)

/-- **The simulator sends `plainB` along the live history.** -/
theorem plainSim_msgK (keep : Bool → Bool) {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (spec : SessionSpec) (st : Stmt) (ω : Ω)
    (hw : 2 * spec.mPts ≤ 65536) (hm1 : 1 ≤ spec.mPts) (u : plainTape x) (r' : Fin (plainN x + 2) → LCoin 65536)
    (hL : plainN x + 2 = (V false).k x) (k : ℕ) (hk : k ≤ plainN x) :
    (((plainSimK keep P x spec st).next x (plainHl P x spec st ω hw (plainDm P x spec st ω u r')
        (plainB x spec st (plainDm P x spec st ω u r') (plainAccK keep P x spec st ω u r')) r' k) (ω, u)).eval sha512).1 =
      plainB x spec st (plainDm P x spec st ω u r') (plainAccK keep P x spec st ω u r') k := by
  have hc := simCoins_agree P x spec st ω hw hm1 (plainDm P x spec st ω u r')
    (plainB x spec st (plainDm P x spec st ω u r') (plainAccK keep P x spec st ω u r')) u r' hL k
  simp only [plainSimK, Claims.OC.eval]
  unfold plainMsgK
  rw [plainHl_length, ite_eq_right (Nat.succ_ne_zero k)]
  by_cases hkN : k < plainN x
  · rw [ite_eq_left (show k + 1 ≤ plainN x by omega)]
    conv_rhs => rw [plainB, ite_eq_left hkN]
    exact congrArg _ (depMsg_congr sha512 P x ω fun i hi => hc i hi)
  · rw [ite_eq_right (show ¬ k + 1 ≤ plainN x by omega)]
    conv_rhs => rw [plainB, ite_eq_right hkN]
    have hacc := run_acc_iff_lastK keep sha512 P x ω (simCoins x spec st u _) (plainΦ x spec st (u, r'))
      fun i hi => hc i (by omega)
    by_cases ha : (Claims.run sha512 P (VK keep) x ω (plainΦ x spec st (u, r'))).out = .acc
    · rw [ite_eq_left (hacc.2 ha)]
      conv_rhs => rw [plainAccK, decide_eq_true ha, ite_eq_left rfl]
      exact depMsg_congr sha512 P x ω fun i hi => hc i hi
    · rw [ite_eq_right (mt hacc.1 ha)]
      conv_rhs => rw [plainAccK, decide_eq_false ha]
      rfl

/-- `plainSim_msgK` at the deployed plain verifier (`keep := id`). -/
theorem plainSim_msg {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (spec : SessionSpec) (st : Stmt) (ω : Ω)
    (hw : 2 * spec.mPts ≤ 65536) (hm1 : 1 ≤ spec.mPts) (u : plainTape x) (r' : Fin (plainN x + 2) → LCoin 65536)
    (hL : plainN x + 2 = (V false).k x) (k : ℕ) (hk : k ≤ plainN x) :
    (((plainSim P x spec st).next x (plainHl P x spec st ω hw (plainDm P x spec st ω u r')
        (plainB x spec st (plainDm P x spec st ω u r') (plainAcc P x spec st ω u r')) r' k) (ω, u)).eval sha512).1 =
      plainB x spec st (plainDm P x spec st ω u r') (plainAcc P x spec st ω u r') k :=
  plainSim_msgK id (Ω := Ω) (P := P) (x := x) (spec := spec) (st := st) (ω := ω) (hw := hw) (hm1 := hm1) (u := u) (r' := r') (hL := hL) (k := k) (hk := hk)

/-- **A round on the plain wire reads back as its submission.** -/
theorem plainSub_round (rep n : ℕ) (a : Claims.Msg) (hr : rep < 2) (hn : n ≤ 65536) :
    plainSub (plainRound rep n a) = some ⟨0, ⟨rep, hr⟩, ⟨a.tail.toArray⟩, n⟩ := by
  have h₀ : (UInt8.ofNat rep).toNat = rep := by rw [UInt8.toNat_ofNat']; omega
  have h₁ : (UInt8.ofNat (n % 256)).toNat = n % 256 := by rw [UInt8.toNat_ofNat']; omega
  have h₂ : (UInt8.ofNat (n / 256 % 256)).toNat = n / 256 % 256 := by rw [UInt8.toNat_ofNat']; omega
  have h₃ : (UInt8.ofNat (n / 65536)).toNat = n / 65536 := by rw [UInt8.toNat_ofNat']; omega
  have hsum : n % 256 + 256 * (n / 256 % 256) + 65536 * (n / 65536) = n := by omega
  unfold plainRound plainSub
  split
  · rename_i r n₀ n₁ n₂ rest heq
    injection heq with _ heq
    injection heq with e₀ heq
    injection heq with e₁ heq
    injection heq with e₂ heq
    injection heq with e₃ heq
    subst e₀ e₁ e₂ e₃ heq
    rw [dite_eq_left (by rw [h₀, h₁, h₂, h₃, hsum]; exact ⟨hr, hn⟩)]
    simp only [h₀, h₁, h₂, h₃, hsum]
  · rename_i hne
    exact (hne _ _ _ _ _ rfl).elim

theorem plainCnt_le (x : Claims.Msg) (spec : SessionSpec) (st : Stmt) {i : ℕ} (h0 : i ≠ 0) :
    plainCnt x spec st i ≤ 65536 := by
  unfold plainCnt
  rw [ite_eq_right h0]
  split_ifs <;> omega

/-- **The simulator's live outcome**: `P`'s roots, the live link points, the rounds of its submissions and the proof
files of its last message. -/
theorem plainSim_liveOutK (keep : Bool → Bool) {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (spec : SessionSpec) (st : Stmt) (ω : Ω)
    (hw : 2 * spec.mPts ≤ 65536) (hm1 : 1 ≤ spec.mPts) (u : plainTape x) (r' : Fin (plainN x + 2) → LCoin 65536)
    (hL : plainN x + 2 = (V false).k x) :
    liveOut (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainW spec hw) sha512 (plainSimK keep P x spec st) x
        (ω, u) r' =
      ⟨(plainWire spec.mPts hw).root ((P.next x [] ω).eval sha512).1,
        (ptsIf (plainWire spec.mPts hw)).proj (Claims.coinSeq r' 0),
        foldS (plainDg spec) (plainS x spec st (plainDm P x spec st ω u r')) (fun i => Claims.coinSeq r' (i + 1))
          (plainN x) (fun _ _ => []),
        (plainWire spec.mPts hw).proofs
          (plainB x spec st (plainDm P x spec st ω u r') (plainAccK keep P x spec st ω u r') (plainN x))⟩ := by
  have hS : ∀ k < plainN x, (plainW spec hw x).sub
      (plainB x spec st (plainDm P x spec st ω u r') (plainAccK keep P x spec st ω u r') k) =
        some (plainS x spec st (plainDm P x spec st ω u r') k) := fun k hk => by
    show plainSub _ = _
    rw [plainB, ite_eq_left hk]
    exact plainSub_round _ _ _ (plainRep_lt spec st (k + 1)) (plainCnt_le x spec st (Nat.succ_ne_zero k))
  have hb : ∀ k ≤ plainN x, (((plainSimK keep P x spec st).next x
      ([((((plainSimK keep P x spec st).next x [] (ω, u)).eval sha512).1,
          (plainW spec hw x).encPts ((ptsIf (plainW spec hw x)).proj (Claims.coinSeq r' 0)))] ++
        liveH (plainW spec hw x)
          (plainB x spec st (plainDm P x spec st ω u r') (plainAccK keep P x spec st ω u r'))
          (plainS x spec st (plainDm P x spec st ω u r')) (fun i => Claims.coinSeq r' (i + 1)) k) (ω, u)).eval
        sha512).1 = plainB x spec st (plainDm P x spec st ω u r') (plainAccK keep P x spec st ω u r') k := fun k hk => by
    rw [plainSim_firstK keep]
    exact plainSim_msgK keep P x spec st ω hw hm1 u r' hL k hk
  rw [liveOut_known plainN (plainDg spec) (plainW spec hw) sha512 (plainSimK keep P x spec st) x (ω, u) r' _ _ hS hb,
    plainSim_firstK keep]
  rfl

/-- `plainSim_liveOutK` at the deployed plain verifier (`keep := id`). -/
theorem plainSim_liveOut {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (spec : SessionSpec) (st : Stmt) (ω : Ω)
    (hw : 2 * spec.mPts ≤ 65536) (hm1 : 1 ≤ spec.mPts) (u : plainTape x) (r' : Fin (plainN x + 2) → LCoin 65536)
    (hL : plainN x + 2 = (V false).k x) :
    liveOut (fun _ => 1) (fun _ => spec.mPts) plainN (plainDg spec) (plainW spec hw) sha512 (plainSim P x spec st) x
        (ω, u) r' =
      ⟨(plainWire spec.mPts hw).root ((P.next x [] ω).eval sha512).1,
        (ptsIf (plainWire spec.mPts hw)).proj (Claims.coinSeq r' 0),
        foldS (plainDg spec) (plainS x spec st (plainDm P x spec st ω u r')) (fun i => Claims.coinSeq r' (i + 1))
          (plainN x) (fun _ _ => []),
        (plainWire spec.mPts hw).proofs
          (plainB x spec st (plainDm P x spec st ω u r') (plainAcc P x spec st ω u r') (plainN x))⟩ :=
  plainSim_liveOutK id (Ω := Ω) (P := P) (x := x) (spec := spec) (st := st) (ω := ω) (hw := hw) (hm1 := hm1) (u := u) (r' := r') (hL := hL)

end FlockDeploy.KS
