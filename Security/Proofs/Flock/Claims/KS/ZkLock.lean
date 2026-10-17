import Proofs.Flock.Claims.KS.ZkSim

/-!
# The masked simulator, message by message, and its abort

Along the live history, `zkSim P x spec sts` sends `P`'s messages at `zkΦ`'s coins on the masked wire (`zkB`,
`zkSim_msg`): the coins it rebuilds are `zkΦ`'s at every coin the history has revealed (`zkSimCoins_agree`), `P`'s
message after `j` rounds reads the coins before `j` only (`zkDepMsg_congr`), and the deployed verifier's acceptance does
not read its last coin (`zk_run_acc_iff_last`). Each round reads back as its submission (`zkSub_round`), so the
simulator's live outcome is `P`'s roots, the live link points, the rounds of its submissions and the proof files of its
last message (`zk_liveOut_known`, `zkSim_liveOut`).

* **The abort** (`zkSimAbortAt`): where `P`'s run at `zkΦ`'s coins is rejected that last message is empty
  (`zkSim_proofs_nil`), and the verdict at the draw refuses an outcome without proof files (`not_accTxJ_nil`), so
  `ZkSimAbortAt` holds at every statement.
-/

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Discharge.Tables Claims.Bridge.Live

local notation "𝔽" => FlockLevel3.GF128

/-! ## Runs, message by message -/

/-- **Acceptance reads the verifier's last answer only for acceptance**: two verifiers that agree before the `K`-th
message and agree on accepting at it accept the same runs of `K` rounds. -/
theorem zk_stepWith_acc_iff {Ω : Type} (H : Claims.Msg → Claims.Msg) (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω)
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

theorem zk_range_map_ends {α : Type} (f : ℕ → α) {K : ℕ} (hK : 2 ≤ K) :
    (List.range K).map f = f 0 :: ((List.range (K - 2)).map (fun i => f (i + 1)) ++ [f (K - 1)]) := by
  obtain ⟨n, rfl⟩ : ∃ n, K = n + 2 := ⟨K - 2, by omega⟩
  rw [List.range_succ, List.range_succ_eq_map]
  simp

/-- **The deployed verifier decides without its last coin**: at coins that differ in the last only, a decision at one
is the same decision at the other. -/
theorem zk_act_last_decide {zk : Bool} {x : Claims.Msg} {as : List Claims.Msg} {c₀ : Coin} {l : List Coin} {c : Coin}
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
theorem zk_next_acc_last {zk : Bool} (H : Claims.Msg → Claims.Msg) {x : Claims.Msg} {as : List Claims.Msg} (c₀ : Coin)
    (l : List Coin) (c c' : Coin) :
    ((next zk x as (c₀ :: (l ++ [c]))).eval H).1 = .acc ↔ ((next zk x as (c₀ :: (l ++ [c']))).eval H).1 = .acc := by
  have key : ∀ c c' : Coin, ((next zk x as (c₀ :: (l ++ [c]))).eval H).1 = .acc →
      ((next zk x as (c₀ :: (l ++ [c']))).eval H).1 = .acc := by
    intro c c' h
    cases ha : act zk x as (c₀ :: (l ++ [c])) with
    | rej => rw [next_rej H ha] at h; cases h
    | coin d => rw [next_coin H ha] at h; cases h
    | decide s spec sts link c₁ mids f =>
      have hb := zk_act_last_decide ha c'
      unfold next at h ⊢
      rw [ha] at h
      rw [hb]
      exact h
  exact ⟨key c c', key c' c⟩

theorem zk_act_none {zk : Bool} {x : Claims.Msg} (hs : setupOf zk x = none) (as : List Claims.Msg) (cs : List Coin) :
    act zk x as cs = .rej := by
  unfold act
  rw [hs]

/-- **A deployed run's acceptance does not read its last coin.** -/
theorem zk_run_acc_iff_last {zk : Bool} (H : Claims.Msg → Claims.Msg) {Ω : Type} (P : Claims.Prover Ω)
    (x : Claims.Msg) (ω : Ω) (r₁ r₂ : Fin ((V zk).k x) → Coin)
    (h : ∀ i, i + 1 < (V zk).k x → Claims.coinSeq r₁ i = Claims.coinSeq r₂ i) :
    (Claims.run H P (V zk) x ω r₁).out = .acc ↔ (Claims.run H P (V zk) x ω r₂).out = .acc := by
  unfold Claims.run
  refine zk_stepWith_acc_iff H P x ω _ _ ((V zk).k x) (fun bs hb => ?_) (fun bs hb => ?_) _ [] [] _ (by simp)
  · show next zk x bs _ = next zk x bs _
    congr 1
    exact List.map_congr_left fun i hi => h i (by have := List.mem_range.1 hi; omega)
  · show ((next zk x bs _).eval H).1 = .acc ↔ ((next zk x bs _).eval H).1 = .acc
    cases hs : setupOf zk x with
    | none =>
      rw [next_rej H (zk_act_none hs _ _), next_rej H (zk_act_none hs _ _)]
    | some p =>
      obtain ⟨s', spec, sts⟩ := p
      have hK : 2 ≤ (V zk).k x := by
        show 2 ≤ (match setupOf zk x with | some (_, spec, sts) => 2 + rounds zk spec sts | none => 1)
        rw [hs]
        exact Nat.le_add_right 2 _
      rw [hb, zk_range_map_ends _ hK, zk_range_map_ends _ hK, h 0 (by omega),
        List.map_congr_left fun i hi => h (i + 1) (by have := List.mem_range.1 hi; omega)]
      exact zk_next_acc_last H _ _ _ _

theorem zkDepHist_congr {Ω : Type} (H : Claims.Msg → Claims.Msg) (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω)
    {c₁ c₂ : ℕ → Coin} : ∀ j, (∀ i < j, c₁ i = c₂ i) → zkDepHist H P x ω c₁ j = zkDepHist H P x ω c₂ j
  | 0, _ => rfl
  | j + 1, h => by
    have e := zkDepHist_congr H P x ω j fun i hi => h i (by omega)
    simp only [zkDepHist]
    rw [e, h j (by omega)]

/-- **`P`'s message after `j` rounds reads the coins before `j` only.** -/
theorem zkDepMsg_congr {Ω : Type} (H : Claims.Msg → Claims.Msg) (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω)
    {c₁ c₂ : ℕ → Coin} {j : ℕ} (h : ∀ i < j, c₁ i = c₂ i) : zkDepMsg H P x ω c₁ j = zkDepMsg H P x ω c₂ j := by
  unfold zkDepMsg
  rw [zkDepHist_congr H P x ω j h]

/-! ## The live rounds against known messages, at `J` tables -/

/-- A coin's first `n` field elements. -/
noncomputable def zkTakeE (n : ℕ) (u : LCoin 65536) : Fin n → 𝔽 :=
  fun i => if h : i.val < 65536 then u ⟨i.val, h⟩ else 0

theorem zkPadE_takeE (n : ℕ) (u : LCoin 65536) (e : Fin 65536) (he : e.val < n) :
    zkPadE n (zkTakeE n u) e = u e := by
  simp [zkPadE, zkTakeE, he]

theorem zk_splitE_fst (nMax n : ℕ) (h : n ≤ nMax) (u : LCoin nMax) (i : Fin n) :
    (splitE nMax n h u).1 i = u ⟨i.val, by omega⟩ := by
  simp [splitE, Equiv.arrowCongr_apply, Equiv.sumArrowEquivProdArrow]
  rfl

theorem zk_rdIf_proj {J mPts : ℕ} (w : Wire 65536 J mPts) (n : ℕ) (h : n ≤ 65536) (u : LCoin 65536) :
    (rdIf w n h).proj u = zkTakeE n u := by
  funext i
  show (splitE 65536 n h u).1 i = _
  rw [zk_splitE_fst]
  simp [zkTakeE, show i.val < 65536 by omega]

theorem zk_ptsIf_proj {J mPts : ℕ} (w : Wire 65536 J mPts) (u : LCoin 65536) :
    (ptsIf w).proj u = ptsE mPts (zkTakeE (2 * mPts) u) := by
  show ptsE mPts (splitE 65536 (2 * mPts) w.pts_le u).1 = _
  congr 1
  funext i
  rw [zk_splitE_fst]
  simp [zkTakeE, show i.val < 65536 by have := w.pts_le; omega]

/-- **The live history of `k` rounds**: round `i`'s message `b i`, answered by the wire's bytes of the first `(S i).n`
entries of its round coin `r i`. -/
noncomputable def zkLiveH {J mPts : ℕ} (w : Wire 65536 J mPts) :
    (ℕ → Claims.Msg) → (ℕ → SubmitJ J) → (ℕ → LCoin 65536) → ℕ → List (Claims.Msg × Claims.Msg)
  | _, _, _, 0 => []
  | b, S, r, k + 1 => (b 0, w.encRd (S 0).n (zkTakeE (S 0).n (r 0))) ::
      zkLiveH w (fun i => b (i + 1)) (fun i => S (i + 1)) (fun i => r (i + 1)) k

/-- **The rep streams after `k` rounds**: round `i` the submission `S i` at the first `(S i).n` entries of `r i`. -/
noncomputable def zkFoldS {J : ℕ} (dg : ByteArray → ByteArray) :
    (ℕ → SubmitJ J) → (ℕ → LCoin 65536) → ℕ → (Fin J → Fin 2 → List Round) → Fin J → Fin 2 → List Round
  | _, _, 0, rs => rs
  | S, r, k + 1, rs => zkFoldS dg (fun i => S (i + 1)) (fun i => r (i + 1)) k
      (addRoundJ rs (S 0).tab (S 0).rep (Refine.roundOf dg (S 0).bytes (S 0).n (zkTakeE (S 0).n (r 0))))

/-- A decoded submission's coins, read off a pending round coin. -/
theorem zk_playOn_sub {J mPts : ℕ} (w : Wire 65536 J mPts) (rootB : Array ByteArray) (pts : Fin 2 → Fin mPts → 𝔽)
    (rs : Fin J → Fin 2 → List Round) (K : (s : SubmitJ J) → (Fin s.n → 𝔽) → Game (LiveOutJ J mPts))
    (next : (s : SubmitJ J) → (c : Fin s.n → 𝔽) → Claims.Bridge.Iface (LCoin 65536) (K s c)) {Ω' : Type}
    (H : Claims.Msg → Claims.Msg) (Q : Claims.Prover Ω') (x : Claims.Msg) (ω' : Ω') (o : Option (SubmitJ J))
    (ho : ∀ s, o = some s → s.n ≤ 65536) (s₀ : SubmitJ J) (h : o = some s₀) (hist : List (Claims.Msg × Claims.Msg))
    (a : Claims.Msg) (u : LCoin 65536) (r : ℕ → LCoin 65536) :
    Claims.Bridge.playOn (branch rootB pts rs K o) (subIf w rootB pts rs K next o ho)
        (Claims.Bridge.stratOf H Q x ω' _ _ hist (some a)) (some u) r =
      Claims.Bridge.playOn (K s₀ (zkTakeE s₀.n u)) (next s₀ (zkTakeE s₀.n u))
        (Claims.Bridge.stratOf H Q x ω' _ _ (hist ++ [(a, w.encRd s₀.n (zkTakeE s₀.n u))]) none) none r := by
  subst h
  show Claims.Bridge.playOn (K s₀ ((rdIf w s₀.n (ho s₀ rfl)).proj u)) (next s₀ ((rdIf w s₀.n (ho s₀ rfl)).proj u))
      (Claims.Bridge.stratOf H Q x ω' _ _ (hist ++ [(a, w.encRd s₀.n ((rdIf w s₀.n (ho s₀ rfl)).proj u))]) none)
      none r = _
  rw [zk_rdIf_proj]

/-- **The live rounds against a prover whose messages are known**: where the prover's message after `k` rounds of the
live history is `b k` and each of the first `N` decodes to the submission `S k`, the rounds end at the streams
`zkFoldS` and the proof files of `b N`. -/
theorem zk_playOn_roundsK {J mPts : ℕ} (w : Wire 65536 J mPts) (dg : ByteArray → ByteArray) (rootB : Array ByteArray)
    (pts : Fin 2 → Fin mPts → 𝔽) {Ω' : Type} (H : Claims.Msg → Claims.Msg) (Q : Claims.Prover Ω')
    (x : Claims.Msg) (ω' : Ω') :
    ∀ (N : ℕ) (rs : Fin J → Fin 2 → List Round) (hist : List (Claims.Msg × Claims.Msg)) (b : ℕ → Claims.Msg)
      (S : ℕ → SubmitJ J) (r : ℕ → LCoin 65536),
      (∀ k < N, w.sub (b k) = some (S k)) →
      (∀ k ≤ N, ((Q.next x (hist ++ zkLiveH w b S r k) ω').eval H).1 = b k) →
      Claims.Bridge.playOn (roundsK dg rootB pts N rs) (roundsIf w dg rootB pts N rs)
          (Claims.Bridge.stratOf H Q x ω' _ _ hist none) none r =
        ⟨rootB, pts, zkFoldS dg S r N rs, w.proofs (b N)⟩
  | 0, rs, hist, b, S, r, _, hb => by
    have h0 := hb 0 le_rfl
    simp only [zkLiveH, List.append_nil] at h0
    show (⟨rootB, pts, rs, w.proofs ((Q.next x hist ω').eval H).1⟩ : LiveOutJ J mPts) = _
    rw [h0]
    rfl
  | N + 1, rs, hist, b, S, r, hS, hb => by
    have h0 := hb 0 (Nat.zero_le _)
    simp only [zkLiveH, List.append_nil] at h0
    have ih := zk_playOn_roundsK w dg rootB pts H Q x ω' N
      (addRoundJ rs (S 0).tab (S 0).rep (Refine.roundOf dg (S 0).bytes (S 0).n (zkTakeE (S 0).n (r 0))))
      (hist ++ [(b 0, w.encRd (S 0).n (zkTakeE (S 0).n (r 0)))]) (fun i => b (i + 1)) (fun i => S (i + 1))
      (fun i => r (i + 1)) (fun k hk => hS (k + 1) (by omega)) (fun k hk => by
        have := hb (k + 1) (by omega)
        simpa [zkLiveH] using this)
    let K : (s : SubmitJ J) → (Fin s.n → 𝔽) → Game (LiveOutJ J mPts) := fun s c =>
      roundsK dg rootB pts N (addRoundJ rs s.tab s.rep (Refine.roundOf dg s.bytes s.n c))
    let nx : (s : SubmitJ J) → (c : Fin s.n → 𝔽) → Claims.Bridge.Iface (LCoin 65536) (K s c) := fun s c =>
      roundsIf w dg rootB pts N (addRoundJ rs s.tab s.rep (Refine.roundOf dg s.bytes s.n c))
    have e : Claims.Bridge.playOn (roundsK dg rootB pts (N + 1) rs) (roundsIf w dg rootB pts (N + 1) rs)
        (Claims.Bridge.stratOf H Q x ω' _ _ hist none) none r =
        Claims.Bridge.playOn (branch rootB pts rs K (w.sub (b 0))) (subIf w rootB pts rs K nx (w.sub (b 0))
          (w.sub_le (b 0))) (Claims.Bridge.stratOf H Q x ω' (branch rootB pts rs K (w.sub (b 0)))
            (subIf w rootB pts rs K nx (w.sub (b 0)) (w.sub_le (b 0))) hist (some (b 0))) (some (r 0))
          (fun i => r (i + 1)) := by
      rw [← h0]
      rfl
    rw [e, zk_playOn_sub w rootB pts rs K nx H Q x ω' _ _ (S 0) (hS 0 (by omega))]
    exact ih

/-- **The live session at `J` tables against a prover whose messages are known**: the roots of its first message, the
link points of the first live coin, then `zk_playOn_roundsK`'s rounds on the later live coins. -/
theorem zk_liveOut_known {J mPts : ℕ} (N : Claims.Msg → ℕ) (dg : ByteArray → ByteArray)
    (w : Claims.Msg → Wire 65536 J mPts) {Ω' : Type} (H : Claims.Msg → Claims.Msg) (Q : Claims.Prover Ω')
    (x : Claims.Msg) (ω' : Ω') (r' : Fin (N x + 2) → LCoin 65536) (b : ℕ → Claims.Msg) (S : ℕ → SubmitJ J)
    (hS : ∀ k < N x, (w x).sub (b k) = some (S k))
    (hb : ∀ k ≤ N x, ((Q.next x ([(((Q.next x [] ω').eval H).1,
        (w x).encPts ((ptsIf (w x)).proj (Claims.coinSeq r' 0)))] ++
          zkLiveH (w x) b S (fun i => Claims.coinSeq r' (i + 1)) k) ω').eval H).1 = b k) :
    liveOut (fun _ => J) (fun _ => mPts) N dg w H Q x ω' r' =
      ⟨(w x).root ((Q.next x [] ω').eval H).1, (ptsIf (w x)).proj (Claims.coinSeq r' 0),
        zkFoldS dg S (fun i => Claims.coinSeq r' (i + 1)) (N x) (fun _ _ => []), (w x).proofs (b (N x))⟩ :=
  zk_playOn_roundsK (w x) dg _ _ H Q x ω' (N x) _ _ b S _ hS hb

theorem zkLiveH_length {J mPts : ℕ} (w : Wire 65536 J mPts) :
    ∀ (k : ℕ) (b : ℕ → Claims.Msg) (S : ℕ → SubmitJ J) (r : ℕ → LCoin 65536), (zkLiveH w b S r k).length = k
  | 0, _, _, _ => rfl
  | k + 1, b, S, r => by
    simp only [zkLiveH, List.length_cons]
    rw [zkLiveH_length w k]

theorem zkLiveH_getD {J mPts : ℕ} (w : Wire 65536 J mPts) :
    ∀ (k : ℕ) (b : ℕ → Claims.Msg) (S : ℕ → SubmitJ J) (r : ℕ → LCoin 65536) (i : ℕ), i < k →
      (zkLiveH w b S r k).getD i default = (b i, w.encRd (S i).n (zkTakeE (S i).n (r i)))
  | 0, _, _, _, i, hi => absurd hi (Nat.not_lt_zero _)
  | k + 1, b, S, r, 0, _ => rfl
  | k + 1, b, S, r, i + 1, hi => by
    simp only [zkLiveH, List.getD_cons_succ]
    exact zkLiveH_getD w k _ _ _ i (by omega)

theorem zk_coinSeq_cast {K L : ℕ} {β : Type} [Inhabited β] (hL : L = K) (r : Fin L → β) (i : Fin K) :
    Claims.coinSeq r i.val = r (Fin.cast hL.symm i) := by
  unfold Claims.coinSeq
  rw [dite_eq_left (by omega)]
  rfl

/-- **`fnC`'s coin `j` reads the coins at `i ≤ j` only at their revealed entries.** -/
theorem zk_fnC_agree {K M : ℕ} {α D : Type} {n : Fin K → ℕ} {key : Fin K → (Fin K → Fin M → α) → D}
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

/-! ## The masked wire's rounds -/

/-- **Deployed round `i`'s rep stream is one of the setup's `2·|sts|`.** -/
theorem zkStream_lt {spec : SessionSpec} {sts : Array Stmt} (h0 : 0 < sts.size) (i : ℕ) :
    zkStream spec sts i < 2 * sts.size := by
  have hall : ∀ v ∈ zkStreams spec sts, v < 2 * sts.size := fun v hv => by
    obtain ⟨k, hk, hv⟩ := List.mem_flatMap.1 hv
    rw [List.eq_of_mem_replicate hv]
    exact List.mem_range.1 hk
  unfold zkStream
  rcases Nat.lt_or_ge (i - 1) (zkStreams spec sts).length with h | h
  · rw [List.getD_eq_getElem _ _ h]
    exact hall _ (List.getElem_mem h)
  · rw [List.getD_eq_default _ _ h]
    omega

theorem zkOnes_replicate (t : ℕ) (l : List UInt8) : zkOnes (List.replicate t 1 ++ 0 :: l) = (t, 0 :: l) := by
  induction t with
  | zero =>
    show zkOnes (0 :: l) = (0, 0 :: l)
    rw [zkOnes, if_neg (by decide)]
  | succ t ih =>
    show zkOnes (1 :: (List.replicate t 1 ++ 0 :: l)) = (t + 1, 0 :: l)
    rw [zkOnes, if_pos rfl, ih]

/-- **A round on the masked wire reads back as its submission**: table `k / 2`, rep `k % 2`, count `n`. -/
theorem zkSub_round (J k n : ℕ) (a : Claims.Msg) (hk : k / 2 < J) (hn : n ≤ 65536) :
    zkSub J (zkRound k n a) = some ⟨⟨k / 2, hk⟩, ⟨k % 2, Nat.mod_lt _ (by norm_num)⟩, ⟨a.tail.toArray⟩, n⟩ := by
  have h₀ : (UInt8.ofNat (k % 2)).toNat = k % 2 := by rw [UInt8.toNat_ofNat']; omega
  have h₁ : (UInt8.ofNat (n % 256)).toNat = n % 256 := by rw [UInt8.toNat_ofNat']; omega
  have h₂ : (UInt8.ofNat (n / 256 % 256)).toNat = n / 256 % 256 := by rw [UInt8.toNat_ofNat']; omega
  have h₃ : (UInt8.ofNat (n / 65536)).toNat = n / 65536 := by rw [UInt8.toNat_ofNat']; omega
  have hsum : n % 256 + 256 * (n / 256 % 256) + 65536 * (n / 65536) = n := by omega
  have ho := zkOnes_replicate (k / 2) (UInt8.ofNat (k % 2) :: UInt8.ofNat (n % 256) :: UInt8.ofNat (n / 256 % 256) ::
    UInt8.ofNat (n / 65536) :: a.tail)
  unfold zkRound zkSub
  split
  · rename_i rest' heq'
    injection heq' with _ heq'
    subst heq'
    rw [ho]
    unfold zkSubAt
    split
    · rename_i r n₀ n₁ n₂ rest heq
      injection heq with _ heq
      injection heq with e₀ heq
      injection heq with e₁ heq
      injection heq with e₂ heq
      injection heq with e₃ heq
      subst e₀ e₁ e₂ e₃ heq
      rw [dite_eq_left (by rw [h₀, h₁, h₂, h₃, hsum]; exact ⟨hk, Nat.mod_lt _ (by norm_num), hn⟩)]
      simp only [h₀, h₁, h₂, h₃, hsum]
    · rename_i hne
      exact (hne _ _ _ _ _ rfl).elim
  · rename_i hne
    exact (hne _ rfl).elim

theorem zkCnt_le (x : Claims.Msg) (spec : SessionSpec) (sts : Array Stmt) {i : ℕ} (h0 : i ≠ 0) :
    zkCnt x spec sts i ≤ 65536 := by
  unfold zkCnt
  rw [ite_eq_right h0]
  split_ifs <;> omega

/-! ## The simulator along the live history -/

/-- **The simulator's submission `k`**: deployed round `k + 1`'s table, rep and count, and the bytes of `P`'s message
`dm (k + 1)` after its tag. -/
def zkS (x : Claims.Msg) (spec : SessionSpec) (sts : Array Stmt) {J : ℕ} (hJ : sts.size ≤ J) (h0 : 0 < sts.size)
    (dm : ℕ → Claims.Msg) (k : ℕ) : SubmitJ J :=
  ⟨⟨zkStream spec sts (k + 1) / 2, by have := zkStream_lt (spec := spec) h0 (k + 1); omega⟩,
    ⟨zkStream spec sts (k + 1) % 2, Nat.mod_lt _ (by norm_num)⟩, ⟨(dm (k + 1)).tail.toArray⟩, zkCnt x spec sts (k + 1)⟩

/-- **The live history after `k` submissions**, at the first message of `P` and the messages `b`. -/
noncomputable def zkHl {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (spec : SessionSpec) (sts : Array Stmt)
    (ω : Ω) {J : ℕ} (hJ : sts.size ≤ J) (h0 : 0 < sts.size) (hw : 2 * spec.mPts ≤ 65536) (dm b : ℕ → Claims.Msg)
    (r' : Fin (zkN x + 2) → LCoin 65536) (k : ℕ) : List (Claims.Msg × Claims.Msg) :=
  [(((P.next x [] ω).eval sha512).1,
      (zkWire J spec.mPts hw).encPts ((ptsIf (zkWire J spec.mPts hw)).proj (Claims.coinSeq r' 0)))] ++
    zkLiveH (zkWire J spec.mPts hw) b (zkS x spec sts hJ h0 dm) (fun i => Claims.coinSeq r' (i + 1)) k

theorem zkHl_length {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (spec : SessionSpec) (sts : Array Stmt)
    (ω : Ω) {J : ℕ} (hJ : sts.size ≤ J) (h0 : 0 < sts.size) (hw : 2 * spec.mPts ≤ 65536) (dm b : ℕ → Claims.Msg)
    (r' : Fin (zkN x + 2) → LCoin 65536) (k : ℕ) : (zkHl P x spec sts ω hJ h0 hw dm b r' k).length = k + 1 := by
  rw [zkHl, List.length_append, zkLiveH_length, List.length_singleton, Nat.add_comm]

/-- **The revealed entries along the live history are the live coins'**: before `k + 1`, the simulator reads back
the first `zkNOf` entries of each live coin. -/
theorem zkRecon_agree {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (spec : SessionSpec) (sts : Array Stmt)
    (ω : Ω) {J : ℕ} (hJ : sts.size ≤ J) (h0 : 0 < sts.size) (hw : 2 * spec.mPts ≤ 65536) (hm1 : 1 ≤ spec.mPts)
    (dm b : ℕ → Claims.Msg) (r' : Fin (zkN x + 2) → LCoin 65536) (hL : zkN x + 2 = (V true).k x) (k : ℕ)
    (i : Fin ((V true).k x)) (hi : i.val < k + 1) :
    Claims.Flock.FnFork.AgreeAt (zkNOf x spec sts i) (zkRecon x spec sts (zkHl P x spec sts ω hJ h0 hw dm b r' k) i)
      (r' (Fin.cast hL.symm i)) := by
  intro e he
  have hlen := zkHl_length P x spec sts ω hJ h0 hw dm b r' k
  unfold zkRecon
  rw [ite_eq_left (by omega), ← zk_coinSeq_cast hL r' i]
  rcases hi0 : i.val with _ | i'
  · have hg : (zkHl P x spec sts ω hJ h0 hw dm b r' k).getD 0 default = (((P.next x [] ω).eval sha512).1,
        finCode (ptsE spec.mPts (zkTakeE (2 * spec.mPts) (Claims.coinSeq r' 0)))) := by
      show (_, (zkWire J spec.mPts hw).encPts ((ptsIf (zkWire J spec.mPts hw)).proj (Claims.coinSeq r' 0))) = _
      rw [zk_ptsIf_proj]
      rfl
    have hn : zkNOf x spec sts i ≤ 2 * spec.mPts := by
      unfold zkNOf zkCnt
      rw [hi0, ite_eq_left rfl]
      omega
    rw [hg]
    unfold zkReveal
    rw [ite_eq_left rfl, finDecode_code, Equiv.symm_apply_apply, zkPadE_takeE _ _ _ (by omega)]
  · have hg : (zkHl P x spec sts ω hJ h0 hw dm b r' k).getD (i' + 1) default = (b i',
        finCode (zkTakeE (zkCnt x spec sts (i' + 1)) (Claims.coinSeq r' (i' + 1)))) := by
      unfold zkHl
      rw [List.singleton_append, List.getD_cons_succ, zkLiveH_getD _ _ _ _ _ _ (by omega)]
      rfl
    have hn : zkNOf x spec sts i = zkCnt x spec sts (i' + 1) := by
      unfold zkNOf
      rw [hi0]
    rw [hg]
    unfold zkReveal
    rw [ite_eq_right (Nat.succ_ne_zero i'), finDecode_code, zkPadE_takeE _ _ _ (by omega)]

/-- **The simulator's coins are `zkΦ`'s at every deployed coin the live history has revealed.** -/
theorem zkSimCoins_agree {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (spec : SessionSpec) (sts : Array Stmt)
    (ω : Ω) {J : ℕ} (hJ : sts.size ≤ J) (h0 : 0 < sts.size) (hw : 2 * spec.mPts ≤ 65536) (hm1 : 1 ≤ spec.mPts)
    (dm b : ℕ → Claims.Msg) (u : zkTape x) (r' : Fin (zkN x + 2) → LCoin 65536) (hL : zkN x + 2 = (V true).k x)
    (k i : ℕ) (hi : i < k + 1) :
    Claims.coinSeq (zkSimCoins x spec sts u (zkHl P x spec sts ω hJ h0 hw dm b r' k)) i =
      Claims.coinSeq (zkΦ x spec sts (u, r')) i := by
  unfold Claims.coinSeq
  split_ifs with hK
  · rw [zkΦ_eq hL]
    exact zk_fnC_agree (Claims.Flock.FnFork.prefixKey_histKey _) u ⟨i, hK⟩ fun j hj =>
      zkRecon_agree P x spec sts ω hJ h0 hw hm1 dm b r' hL k j (lt_of_le_of_lt (Fin.le_def.1 hj) hi)
  · rfl

/-- **The simulator's messages**: deployed round `k + 1`'s on the masked wire before the last, and at the last `P`'s
last message where `V true` accepts at `zkΦ`, else nothing. -/
noncomputable def zkB (x : Claims.Msg) (spec : SessionSpec) (sts : Array Stmt) (dm : ℕ → Claims.Msg) (acc : Bool)
    (k : ℕ) : Claims.Msg :=
  if k < zkN x then zkRound (zkStream spec sts (k + 1)) (zkCnt x spec sts (k + 1)) (dm (k + 1))
  else if acc then dm (k + 1) else []

/-- `P`'s messages at `zkΦ`'s coins. -/
noncomputable def zkDm {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (spec : SessionSpec) (sts : Array Stmt)
    (ω : Ω) (u : zkTape x) (r' : Fin (zkN x + 2) → LCoin 65536) : ℕ → Claims.Msg :=
  zkDepMsg sha512 P x ω (Claims.coinSeq (zkΦ x spec sts (u, r')))

/-- Whether `V true` accepts `P` at `zkΦ`'s coins. -/
noncomputable def zkAcc {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (spec : SessionSpec) (sts : Array Stmt)
    (ω : Ω) (u : zkTape x) (r' : Fin (zkN x + 2) → LCoin 65536) : Bool :=
  decide ((Claims.run sha512 P (V true) x ω (zkΦ x spec sts (u, r'))).out = .acc)

/-- **The simulator sends `zkB` along the live history.** -/
theorem zkSim_msg {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (spec : SessionSpec) (sts : Array Stmt) (ω : Ω)
    {J : ℕ} (hJ : sts.size ≤ J) (h0 : 0 < sts.size) (hw : 2 * spec.mPts ≤ 65536) (hm1 : 1 ≤ spec.mPts)
    (u : zkTape x) (r' : Fin (zkN x + 2) → LCoin 65536) (hL : zkN x + 2 = (V true).k x) (k : ℕ) (hk : k ≤ zkN x) :
    (((zkSim P x spec sts).next x (zkHl P x spec sts ω hJ h0 hw (zkDm P x spec sts ω u r')
        (zkB x spec sts (zkDm P x spec sts ω u r') (zkAcc P x spec sts ω u r')) r' k) (ω, u)).eval sha512).1 =
      zkB x spec sts (zkDm P x spec sts ω u r') (zkAcc P x spec sts ω u r') k := by
  have hc := zkSimCoins_agree P x spec sts ω hJ h0 hw hm1 (zkDm P x spec sts ω u r')
    (zkB x spec sts (zkDm P x spec sts ω u r') (zkAcc P x spec sts ω u r')) u r' hL k
  simp only [zkSim, Claims.OC.eval]
  unfold zkMsg
  rw [zkHl_length, ite_eq_right (Nat.succ_ne_zero k)]
  by_cases hkN : k < zkN x
  · rw [ite_eq_left (show k + 1 ≤ zkN x by omega)]
    conv_rhs => rw [zkB, ite_eq_left hkN]
    exact congrArg _ (zkDepMsg_congr sha512 P x ω fun i hi => hc i hi)
  · rw [ite_eq_right (show ¬ k + 1 ≤ zkN x by omega)]
    conv_rhs => rw [zkB, ite_eq_right hkN]
    have hacc := zk_run_acc_iff_last sha512 P x ω (zkSimCoins x spec sts u _) (zkΦ x spec sts (u, r'))
      fun i hi => hc i (by omega)
    by_cases ha : (Claims.run sha512 P (V true) x ω (zkΦ x spec sts (u, r'))).out = .acc
    · rw [ite_eq_left (hacc.2 ha)]
      conv_rhs => rw [zkAcc, decide_eq_true ha, ite_eq_left rfl]
      exact zkDepMsg_congr sha512 P x ω fun i hi => hc i hi
    · rw [ite_eq_right (mt hacc.1 ha)]
      conv_rhs => rw [zkAcc, decide_eq_false ha]
      rfl

/-- **The simulator's live outcome**: `P`'s roots, the live link points, the rounds of its submissions and the proof
files of its last message. -/
theorem zkSim_liveOut {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (spec : SessionSpec) (sts : Array Stmt)
    (ω : Ω) {J : ℕ} (hJ : sts.size ≤ J) (h0 : 0 < sts.size) (hw : 2 * spec.mPts ≤ 65536) (hm1 : 1 ≤ spec.mPts)
    (u : zkTape x) (r' : Fin (zkN x + 2) → LCoin 65536) (hL : zkN x + 2 = (V true).k x) :
    liveOut (fun _ => J) (fun _ => spec.mPts) zkN spec.dg (zkW J spec hw) sha512 (zkSim P x spec sts) x (ω, u) r' =
      ⟨(zkWire J spec.mPts hw).root ((P.next x [] ω).eval sha512).1,
        (ptsIf (zkWire J spec.mPts hw)).proj (Claims.coinSeq r' 0),
        zkFoldS spec.dg (zkS x spec sts hJ h0 (zkDm P x spec sts ω u r')) (fun i => Claims.coinSeq r' (i + 1))
          (zkN x) (fun _ _ => []),
        (zkWire J spec.mPts hw).proofs
          (zkB x spec sts (zkDm P x spec sts ω u r') (zkAcc P x spec sts ω u r') (zkN x))⟩ := by
  have hS : ∀ k < zkN x, (zkW J spec hw x).sub
      (zkB x spec sts (zkDm P x spec sts ω u r') (zkAcc P x spec sts ω u r') k) =
        some (zkS x spec sts hJ h0 (zkDm P x spec sts ω u r') k) := fun k hk => by
    show zkSub J _ = _
    rw [zkB, ite_eq_left hk]
    exact zkSub_round J _ _ _ (by have := zkStream_lt (spec := spec) h0 (k + 1); omega)
      (zkCnt_le x spec sts (Nat.succ_ne_zero k))
  have hb : ∀ k ≤ zkN x, (((zkSim P x spec sts).next x
      ([((((zkSim P x spec sts).next x [] (ω, u)).eval sha512).1,
          (zkW J spec hw x).encPts ((ptsIf (zkW J spec hw x)).proj (Claims.coinSeq r' 0)))] ++
        zkLiveH (zkW J spec hw x)
          (zkB x spec sts (zkDm P x spec sts ω u r') (zkAcc P x spec sts ω u r'))
          (zkS x spec sts hJ h0 (zkDm P x spec sts ω u r')) (fun i => Claims.coinSeq r' (i + 1)) k) (ω, u)).eval
        sha512).1 = zkB x spec sts (zkDm P x spec sts ω u r') (zkAcc P x spec sts ω u r') k := fun k hk => by
    rw [zkSim_first]
    exact zkSim_msg P x spec sts ω hJ h0 hw hm1 u r' hL k hk
  rw [zk_liveOut_known zkN spec.dg (zkW J spec hw) sha512 (zkSim P x spec sts) x (ω, u) r' _ _ hS hb, zkSim_first]
  rfl

/-! ## The abort -/

/-- **The verdict at the draw refuses an outcome without proof files**: what `wrS` writes has the outcome's proof
files wherever `verify --zk` accepts it (`recordsLiveAtS_wrS`), and `verify --zk` refuses none (`not_acceptsAt_nil`). -/
theorem not_accTxJ_nil (I : Discharge.Exec.Inputs) (d : Lean.Json) (j₀ : ℕ) (rootB₀ : Array ByteArray) {N : ℕ}
    (C : Fin N → List 𝔽) {J mPts : ℕ} {o : LiveOutJ J mPts} (h : o.proofs = #[]) : ¬ AccTxJ I d j₀ rootB₀ C o := by
  rintro ⟨hA, -, -⟩
  have hp := (Claims.Bridge.Custody.recordsLiveAtS_wrS I d o hA).1
  rw [h] at hp
  generalize Claims.Bridge.Custody.wrS I d o = t at hA hp
  rcases t with ⟨record, proofs⟩
  subst hp
  exact Claims.Bridge.Custody.not_acceptsAt_nil I d record hA

/-- **Where `P`'s run at `zkΦ`'s coins is rejected, the simulator's live outcome has no proof files**: its last
message is empty. -/
theorem zkSim_proofs_nil {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (spec : SessionSpec) (sts : Array Stmt)
    (ω : Ω) {J : ℕ} (hJ : sts.size ≤ J) (h0 : 0 < sts.size) (hw : 2 * spec.mPts ≤ 65536) (hm1 : 1 ≤ spec.mPts)
    (u : zkTape x) (r' : Fin (zkN x + 2) → LCoin 65536) (hL : zkN x + 2 = (V true).k x)
    (hna : (Claims.run sha512 P (V true) x ω (zkΦ x spec sts (u, r'))).out ≠ .acc) :
    (liveOut (fun _ => J) (fun _ => spec.mPts) zkN spec.dg (zkW J spec hw) sha512 (zkSim P x spec sts) x (ω, u)
      r').proofs = #[] := by
  rw [zkSim_liveOut P x spec sts ω hJ h0 hw hm1 u r' hL]
  show ((finalOf (zkB x spec sts (zkDm P x spec sts ω u r') (zkAcc P x spec sts ω u r') (zkN x))).map
      Final.proofs).getD #[] = #[]
  rw [zkB, ite_eq_right (lt_irrefl _), zkAcc, decide_eq_false hna]
  rfl

theorem zk_tabsJ_le {x : Claims.Msg} {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts) :
    sts.size ≤ (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length := by
  simp only [tabsJ_length, zk_tables_length hset, le_refl]

/-- **The abort holds at every statement** (`zkSim_proofs_nil`, `not_accTxJ_nil`). -/
theorem zkSimAbortAt (x : Claims.Msg) : ZkSimAbortAt x :=
  fun _ hs hin _ spec sts hset h0 L hz hk hlay d _ _ P ω j _ C _ u r' hna =>
    not_accTxJ_nil _ d j _ C
      (zkSim_proofs_nil P x spec sts ω (zk_tabsJ_le hset h0 L hz hk hlay) h0 (zk_mPts_le hs hin hset)
        (zk_mPts hs hin hset).2 u r' (zkN_add hset) hna)

end FlockDeploy.KS
