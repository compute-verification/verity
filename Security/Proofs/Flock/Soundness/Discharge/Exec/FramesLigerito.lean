import Proofs.Flock.Soundness.Discharge.Exec.Emit
import Proofs.Flock.Soundness.Refine.Final

/-!
# Ligerito's rounds, as frames (R11b; PROTOCOL.md §13.3)

`ligerito_frames`: a Ligerito run that succeeds on retained rounds emits `ligFrames`, in order: the label, `T` and the
cap; level 0's out-of-domain sample, first message, folding challenges, the level-1 cap, two intros, queries, query
message and batching coin; then each level of the loop (`levelFr`). The proof's lists are read at the cursors the run
keeps: sumcheck messages (`kFr`), out-of-domain values, and claim, query and consistency nonces.

* The Ligerito state's steps, with the frames they emit: `nextMsg_emit`, `claimPow_emit`, `challenge_emit`,
  `oodIntro_emit`, `queryLevel_emit`; loops: `challenges_emitted`, `forIn_absorbF_emitted`, `lforIn_emitted`.
-/

namespace FlockSoundness.Discharge.Exec

open Flock Refine Transcript

variable {β : Type}

/-- A sumcheck message's frames: its two `K` elements, each absorbed as a pair of words. -/
def kFr (m : F256 × F256) : List Frame := [.vec [m.1.c0, m.1.c1], .vec [m.2.c0, m.2.c1]]

theorem absorbK_emitted {v : F256} {t t₁ : Transcript} {u : Unit} (h : absorbK v t = .ok (u, t₁)) :
    Emitted t t₁ [.vec [v.c0, v.c1]] :=
  absorbSlice_emitted h

theorem nextMsg_emit {p : LigeritoProof} {f : F256 × F256 → LM β} {st st' : LState} {t t' : Transcript} {b : β}
    (h : (nextMsg p >>= f) st t = .ok ((b, st'), t')) :
    ∃ m t₁, p.messages[st.msg]? = some m ∧ Emitted t t₁ (kFr m) ∧
      f m { st with msg := st.msg + 1 } t₁ = .ok ((b, st'), t') := by
  obtain ⟨m, st₁, t₁, h₁, h₂⟩ := lbind_ok h
  unfold nextMsg at h₁
  have h₁ := lget_bind h₁
  cases hm : p.messages[st.msg]? with
  | none => simp only [hm] at h₁; cases h₁
  | some m' =>
    simp only [hm] at h₁
    have h₁ := lset_bind h₁
    obtain ⟨_, t₃, h₃, h₁⟩ := lift_bind h₁
    obtain ⟨_, t₄, h₄, h₁⟩ := lift_bind h₁
    obtain ⟨rfl, rfl, rfl⟩ := lpure_ok h₁
    exact ⟨_, _, rfl, (absorbK_emitted h₃).trans (absorbK_emitted h₄), h₂⟩

theorem claimPow_emit {p : LigeritoProof} {f : PUnit → LM β} {st st' : LState} {t t' : Transcript} {b : β}
    (h : (claimPow p >>= f) st t = .ok ((b, st'), t')) :
    ∃ n t₁, p.claimNonces[st.cbn]? = some n ∧ Emitted t t₁ [.bytes (le64 n)] ∧
      f () { st with cbn := st.cbn + 1 } t₁ = .ok ((b, st'), t') := by
  obtain ⟨u, st₁, t₁, h₁, h₂⟩ := lbind_ok h
  unfold claimPow at h₁
  have h₁ := lget_bind h₁
  obtain ⟨n, hn, h₁⟩ := nonceFrom_bind h₁
  have h₁ := lset_bind h₁
  obtain ⟨hst, hx⟩ := lift_ok h₁
  subst hst
  exact ⟨n, t₁, hn, absorbBytes_emitted hx, h₂⟩

theorem challenge_emit {p : LigeritoProof} {f : F256 → LM β} {st st' : LState} {t t' : Transcript} {b : β}
    (h : (challenge p >>= f) st t = .ok ((b, st'), t')) (hret : AllRetained t.rounds) :
    ∃ r m st₁ t₁, p.messages[st.msg]? = some m ∧ st₁.msg = st.msg + 1 ∧ st₁.ood = st.ood ∧ st₁.cbn = st.cbn ∧
      st₁.qn = st.qn ∧ st₁.cn = st.cn ∧ Emitted t t₁ (.sq 2 2 :: kFr m) ∧ f r st₁ t₁ = .ok ((b, st'), t') := by
  obtain ⟨r, st₁, t₁, h₁, h₂⟩ := lbind_ok h
  unfold challenge at h₁
  obtain ⟨r', t₃, h₃, h₁⟩ := lift_bind h₁
  obtain ⟨e₃, -⟩ := sampleK_emitted h₃ hret
  have h₁ := lmodify_bind h₁
  obtain ⟨m, t₄, hm, e₄, h₁⟩ := nextMsg_emit h₁
  have h₁ := lmodify_bind h₁
  obtain ⟨rfl, rfl, rfl⟩ := lpure_ok h₁
  refine ⟨_, m, _, _, hm, ?_, ?_, ?_, ?_, ?_, e₃.trans e₄, h₂⟩ <;> rfl

theorem oodIntro_emit {p : LigeritoProof} {dim split : ℕ} (hdim : dim ≠ 0) {f : PUnit → LM β} {st st' : LState}
    {t t' : Transcript} {b : β} (h : (oodIntro p dim split >>= f) st t = .ok ((b, st'), t'))
    (hret : AllRetained t.rounds) :
    ∃ y m n st₁ t₁, p.oodValues[st.ood]? = some y ∧ p.messages[st.msg]? = some m ∧ p.claimNonces[st.cbn]? = some n ∧
      st₁.msg = st.msg + 1 ∧ st₁.ood = st.ood + 1 ∧ st₁.cbn = st.cbn + 1 ∧ st₁.qn = st.qn ∧ st₁.cn = st.cn ∧
      Emitted t t₁ (.sq 2 dim :: .one y :: kFr m ++ [.bytes (le64 n), .sq 1 1]) ∧ f () st₁ t₁ = .ok ((b, st'), t') := by
  obtain ⟨u, st₁, t₁, h₁, h₂⟩ := lbind_ok h
  unfold oodIntro at h₁
  obtain ⟨z, t₃, h₃, h₁⟩ := lift_bind h₁
  obtain ⟨e₃, -⟩ := sampleVec_emitted hdim h₃ hret
  obtain ⟨y, hy, h₁⟩ := nextOod_bind h₁
  obtain ⟨_, t₄, h₄, h₁⟩ := lift_bind h₁
  have e₄ := absorbF_emitted h₄
  obtain ⟨m, t₅, hm, e₅, h₁⟩ := nextMsg_emit h₁
  obtain ⟨n, t₆, hn, e₆, h₁⟩ := claimPow_emit h₁
  obtain ⟨bc, t₇, h₇, h₁⟩ := lift_bind h₁
  have hret₆ : AllRetained t₆.rounds := by rw [e₆.rounds, e₅.rounds, e₄.rounds, e₃.rounds]; exact hret
  obtain ⟨e₇, -⟩ := sample_emitted h₇ hret₆
  obtain ⟨rfl, rfl⟩ := lmodify_ok h₁
  refine ⟨y, m, n, _, _, hy, hm, hn, ?_, ?_, ?_, ?_, ?_, ?_, h₂⟩
  all_goals try rfl
  have := e₃.trans (e₄.trans (e₅.trans (e₆.trans e₇)))
  simpa [kFr] using this

theorem queryLevel_emit {ms : MerkleScheme} {p : LigeritoProof} {lv : Flock.Level} {cap : Array ByteArray}
    {rows : Array (Array F128)} {sibs salts : Array ByteArray} {fold : Array F256} (hq : lv.q ≠ 0)
    {f : Array ℕ × Array F128 × F256 → LM β} {st st' : LState} {t t' : Transcript} {b : β}
    (h : (queryLevel ms p lv cap rows sibs salts fold >>= f) st t = .ok ((b, st'), t')) (hret : AllRetained t.rounds) :
    ∃ qn cn r t₁, p.queryNonces[st.qn]? = some qn ∧ p.consistencyNonces[st.cn]? = some cn ∧
      Emitted t t₁ [.bytes (le64 qn), .sq 2 lv.q, .bytes (le64 cn), .sq 2 ((lv.q - 1).log2 + 1)] ∧
      f r { st with qn := st.qn + 1, cn := st.cn + 1 } t₁ = .ok ((b, st'), t') := by
  obtain ⟨r, st₁, t₁, h₁, h₂⟩ := lbind_ok h
  unfold queryLevel at h₁
  have h₁ := lget_bind h₁
  obtain ⟨qn, hqn, h₁⟩ := nonceFrom_bind h₁
  obtain ⟨cn, hcn, h₁⟩ := nonceFrom_bind h₁
  have h₁ := lset_bind h₁
  obtain ⟨_, t₃, h₃, h₁⟩ := lift_bind h₁
  have e₃ := absorbBytes_emitted h₃
  obtain ⟨coins, t₄, h₄, h₁⟩ := lift_bind h₁
  obtain ⟨e₄, -⟩ := sampleVec_emitted hq h₄ (by rw [e₃.rounds]; exact hret)
  obtain ⟨_, t₅, h₅, h₁⟩ := lift_bind h₁
  have e₅ := absorbBytes_emitted h₅
  obtain ⟨alpha, t₆, h₆, h₁⟩ := lift_bind h₁
  obtain ⟨e₆, -⟩ := sampleVec_emitted (by omega) h₆ (by rw [e₅.rounds, e₄.rounds, e₃.rounds]; exact hret)
  cases hmc : merkleCheck ms cap lv.cap lv.d (positions lv coins) rows sibs salts (2 ^ lv.lanes) with
  | error e => simp only [hmc] at h₁; obtain ⟨_, _, _, h₀, _⟩ := lbind_ok h₁; cases h₀
  | ok u =>
    simp only [hmc] at h₁
    simp only [List.forIn_pure_yield_eq_foldl, Std.Legacy.Range.forIn_eq_forIn_range', pure_bind] at h₁
    obtain ⟨rfl, rfl, rfl⟩ := lpure_ok h₁
    exact ⟨qn, cn, _, _, hqn, hcn, e₃.trans (e₄.trans (e₅.trans e₆)), h₂⟩

/-- A cap's frame: its entries' bytes, absorbed as one string. -/
def capFr (cap : Array ByteArray) : Frame := .bytes (cap.foldl (· ++ ·) ByteArray.empty)

/-- `k` folding challenges from message `msg` on: each round's `K` coin, then the next sumcheck message. -/
def chFr (p : LigeritoProof) (msg : ℕ) : ℕ → List Frame
  | 0 => []
  | k + 1 => .sq 2 2 :: kFr p.messages[msg]! ++ chFr p (msg + 1) k

/-- An out-of-domain sample with its intro message: the point's coins, the value, the message, the claim nonce and the
batching coin. -/
def introFr (p : LigeritoProof) (dim ood msg cbn : ℕ) : List Frame :=
  .sq 2 dim :: .one p.oodValues[ood]! :: kFr p.messages[msg]! ++ [.bytes (le64 p.claimNonces[cbn]!), .sq 1 1]

/-- A level's queries: the query nonce and coins, then the consistency nonce and coins. -/
def queryFr (p : LigeritoProof) (lv : Flock.Level) (qn cn : ℕ) : List Frame :=
  [.bytes (le64 p.queryNonces[qn]!), .sq 2 lv.q, .bytes (le64 p.consistencyNonces[cn]!), .sq 2 ((lv.q - 1).log2 + 1)]

theorem challenges_emitted (p : LigeritoProof) :
    ∀ (k a : ℕ) (chs chs' : Array F256) (st st' : LState) (t t' : Transcript), AllRetained t.rounds →
      forIn (List.range' a k) chs (fun _ chs => challenge p >>= fun x => pure (ForInStep.yield (chs.push x))) st t =
        .ok ((chs', st'), t') →
      st'.msg = st.msg + k ∧ st'.ood = st.ood ∧ st'.cbn = st.cbn ∧ st'.qn = st.qn ∧ st'.cn = st.cn ∧
        Emitted t t' (chFr p st.msg k)
  | 0, a, chs, chs', st, st', t, t', _, h => by
      simp only [List.range'_zero, List.forIn_nil] at h
      obtain ⟨rfl, rfl, rfl⟩ := lpure_ok h
      exact ⟨rfl, rfl, rfl, rfl, rfl, Emitted.refl _⟩
  | k + 1, a, chs, chs', st, st', t, t', hret, h => by
      rw [List.range'_succ, List.forIn_cons] at h
      obtain ⟨r, st₁, t₁, h₁, h₂⟩ := lbind_ok h
      obtain ⟨x, m, st₂, t₂, hm, h1, h2, h3, h4, h5, e₁, h₁⟩ := challenge_emit h₁ hret
      obtain ⟨rfl, rfl, rfl⟩ := lpure_ok h₁
      obtain ⟨g1, g2, g3, g4, g5, e₂⟩ :=
        challenges_emitted p k (a + 1) _ chs' _ st' _ t' (by rw [e₁.rounds]; exact hret) h₂
      refine ⟨by omega, by omega, by omega, by omega, by omega, ?_⟩
      have hmm := getElem!_of_getElem? hm
      rw [h1] at e₂
      have := e₁.trans e₂
      simp only [chFr, hmm]
      simpa using this

theorem forIn_absorbF_emitted : ∀ (ws : List F128) {st st' : LState} {t t' : Transcript} {u : PUnit},
    forIn ws PUnit.unit (fun w (_ : PUnit) => lift (absorbF w) >>= fun _ => pure (ForInStep.yield PUnit.unit)) st t =
      .ok ((u, st'), t') → st' = st ∧ Emitted t t' (ws.map .one)
  | [], st, st', t, t', u, h => by
      simp only [List.forIn_nil] at h
      obtain ⟨-, rfl, rfl⟩ := lpure_ok h
      exact ⟨rfl, Emitted.refl _⟩
  | w :: ws, st, st', t, t', u, h => by
      rw [List.forIn_cons] at h
      obtain ⟨b, st₁, t₁, h₁, h₂⟩ := lbind_ok h
      obtain ⟨_, t₂, h₃, h₁⟩ := lift_bind h₁
      have e₃ := absorbF_emitted h₃
      obtain ⟨rfl, rfl, rfl⟩ := lpure_ok h₁
      obtain ⟨rfl, e⟩ := forIn_absorbF_emitted ws h₂
      exact ⟨rfl, by simpa using e₃.trans e⟩

/-- **A loop of the Ligerito state emits its iterations' frames**, when each iteration below `B`, from a state with the
invariant `P`, emits its own and restores `P`. -/
theorem lforIn_emitted {σ : Type} (body : ℕ → σ → LM (ForInStep σ)) (F : ℕ → List Frame) (P : ℕ → σ → LState → Prop)
    (B : ℕ) (hbody : ∀ i s st t b st' t', i < B → P i s st → AllRetained t.rounds → body i s st t = .ok ((b, st'), t') →
      ∃ s', b = ForInStep.yield s' ∧ P (i + 1) s' st' ∧ Emitted t t' (F i)) :
    ∀ (n a : ℕ) (s : σ) (st : LState) (t : Transcript) (s' : σ) (st' : LState) (t' : Transcript), a + n ≤ B →
      P a s st → AllRetained t.rounds → (forIn (List.range' a n) s body : LM σ) st t = .ok ((s', st'), t') →
      P (a + n) s' st' ∧ Emitted t t' ((List.range' a n).flatMap F)
  | 0, a, s, st, t, s', st', t', _, hP, _, h => by
      simp only [List.range'_zero, List.forIn_nil] at h
      obtain ⟨rfl, rfl, rfl⟩ := lpure_ok h
      exact ⟨hP, Emitted.refl _⟩
  | n + 1, a, s, st, t, s', st', t', hB, hP, hret, h => by
      rw [List.range'_succ, List.forIn_cons] at h
      obtain ⟨r, st₁, t₁, h₁, h₂⟩ := lbind_ok h
      obtain ⟨s₁, rfl, hP₁, he₁⟩ := hbody a s st t r st₁ t₁ (by omega) hP hret h₁
      obtain ⟨hP', he₂⟩ := lforIn_emitted body F P B hbody n (a + 1) s₁ st₁ t₁ s' st' t' (by omega) hP₁
        (by rw [he₁.rounds]; exact hret) h₂
      refine ⟨by rw [show a + (n + 1) = a + 1 + n by omega]; exact hP', ?_⟩
      rw [List.range'_succ, List.flatMap_cons]
      exact he₁.trans he₂

/-- The level loop's iteration `i` (§13.3): four folding challenges; then, below the last level, the next level's cap,
two intros, the level's queries, the query message, its claim nonce and batching coin; at the last, the clear residual
`yr`, the queries, the claim nonce and the batching coin. -/
def levelFr (sch : Flock.Schedule) (p : LigeritoProof) (i : ℕ) : List Frame :=
  chFr p (sch.k0 + 4 + 7 * i) 4 ++
    if i + 1 < sch.levels.size - 1 then
      capFr p.recursiveCaps[i + 1]! ::
        (introFr p ((sch.levels.getD (i + 1) default).cols + 1) (3 + 2 * i) (sch.k0 + 8 + 7 * i) (4 + 3 * i) ++
          introFr p ((sch.levels.getD (i + 1) default).cols + 1) (4 + 2 * i) (sch.k0 + 9 + 7 * i) (5 + 3 * i) ++
          queryFr p (sch.levels.getD (i + 1) default) (1 + i) (1 + i) ++
          kFr p.messages[sch.k0 + 10 + 7 * i]! ++ [.bytes (le64 p.claimNonces[6 + 3 * i]!), .sq 1 1])
    else
      p.finalProof.yr.toList.map .one ++ queryFr p (sch.levels.getD (sch.levels.size - 1) default) (1 + i) (1 + i) ++
        [.bytes (le64 p.claimNonces[4 + 3 * i]!), .sq 1 1]

/-- **A Ligerito run's frames** (§13.3): the label, `T` and the cap; level 0's out-of-domain sample, first message,
folding challenges, the level-1 cap, two intros, queries, query message and batching coin; then the level loop's. -/
def ligFrames (sch : Flock.Schedule) (T : F128) (cap : Array ByteArray) (p : LigeritoProof) : List Frame :=
  [.label "flock-ligerito-basis-f256-split-v0".toUTF8, .one T, capFr cap] ++
    ([.sq 2 sch.n, .one p.oodValues[0]!, .bytes (le64 p.claimNonces[0]!), .sq 1 1] ++ kFr p.messages[0]! ++
      chFr p 1 sch.k0 ++ capFr p.recursiveCaps[0]! ::
      (introFr p (sch.n - sch.k0 + 1) 1 (sch.k0 + 1) 1 ++ introFr p (sch.n - sch.k0 + 1) 2 (sch.k0 + 2) 2 ++
        queryFr p (sch.levels.getD 0 default) 0 0 ++ kFr p.messages[sch.k0 + 3]! ++
        [.bytes (le64 p.claimNonces[3]!), .sq 1 1] ++ (List.range' 0 (sch.levels.size - 1)).flatMap (levelFr sch p)))

theorem Emitted.retained {t t₁ : Transcript} {fs : List Frame} (e : Emitted t t₁ fs) (h : AllRetained t.rounds) :
    AllRetained t₁.rounds := by
  rw [e.rounds]; exact h

set_option maxHeartbeats 4000000 in
theorem ligerito_frames (ms : MerkleScheme) (sch : Flock.Schedule) (T : F128) (cap : Array ByteArray)
    (p : LigeritoProof) (basisAt : Array F256 → ℕ → Array F256) (hdim : DimSeam sch)
    (hlv : ∀ i < sch.levels.size, LevelSeam (sch.levels.getD i default))
    {t t' : Transcript} (hret : AllRetained t.rounds) (h : Flock.ligerito ms sch T cap p basisAt t = .ok ((), t')) :
    Emitted t t' (ligFrames sch T cap p) := by
  obtain ⟨hn7, h2, hl0, hc0, hl4, hcols, hext⟩ := hdim
  have hc01 := hcols 0 (by omega)
  have hn3 : 3 ≤ sch.n := by omega
  unfold Flock.ligerito at h
  obtain ⟨hcap, h⟩ := check_bind h
  obtain ⟨_, t₁, h₁, h⟩ := bind_ok h
  have e₁ := label_emitted h₁
  obtain ⟨_, t₂, h₂, h⟩ := bind_ok h
  have e₂ := absorbF_emitted h₂
  obtain ⟨_, t₃, h₃, h⟩ := bind_ok h
  have e₃ := absorbBytes_emitted h₃
  obtain ⟨⟨u, stF⟩, t₄, h₄, h⟩ := bind_ok h
  obtain ⟨-, rfl⟩ := pure_ok h
  rw [StateT.run] at h₄
  have r₃ := e₃.retained (e₂.retained (e₁.retained hret))
  obtain ⟨z, t₅, h₅, h₄⟩ := lift_bind h₄
  obtain ⟨e₅, -⟩ := sampleVec_emitted (by omega) h₅ r₃
  obtain ⟨y₀, hy₀, h₄⟩ := nextOod_bind h₄
  obtain ⟨_, t₆, h₆, h₄⟩ := lift_bind h₄
  have e₆ := absorbF_emitted h₆
  obtain ⟨n₀, t₇, hn₀, e₇, h₄⟩ := claimPow_emit h₄
  obtain ⟨bc, t₈, h₈, h₄⟩ := lift_bind h₄
  obtain ⟨e₈, -⟩ := sample_emitted h₈ (e₇.retained (e₆.retained (e₅.retained r₃)))
  have h₄ := lmodify_bind h₄
  obtain ⟨m₀, t₉, hm₀, e₉, h₄⟩ := nextMsg_emit h₄
  have h₄ := lmodify_bind h₄
  obtain ⟨initCh, st₁₀, t₁₀, h₁₀, h₄⟩ := lbind_ok h₄
  rw [lforIn_range_eq] at h₁₀
  obtain ⟨g1, g2, g3, g4, g5, e₁₀⟩ := challenges_emitted p sch.k0 0 _ _ _ _ _ _
    (e₉.retained (e₈.retained (e₇.retained (e₆.retained (e₅.retained r₃))))) h₁₀
  have r₁₀ := e₁₀.retained (e₉.retained (e₈.retained (e₇.retained (e₆.retained (e₅.retained r₃)))))
  obtain ⟨_, t₁₁, h₁₁, h₄⟩ := lift_bind h₄
  have e₁₁ := absorbBytes_emitted h₁₁
  obtain ⟨yA, mA, nA, stA, tA, hyA, hmA, hnA, a1, a2, a3, a4, a5, eA, h₄⟩ :=
    oodIntro_emit (show sch.n - sch.k0 + 1 ≠ 0 by omega) h₄ (e₁₁.retained r₁₀)
  obtain ⟨yB, mB, nB, stB, tB, hyB, hmB, hnB, b1, b2, b3, b4, b5, eB, h₄⟩ :=
    oodIntro_emit (show sch.n - sch.k0 + 1 ≠ 0 by omega) h₄ (eA.retained (e₁₁.retained r₁₀))
  have hlv0 := hlv 0 (by omega)
  obtain ⟨qn₀, cn₀, ⟨pos0, al0, e0⟩, tQ, hqn₀, hcn₀, eQ, h₄⟩ :=
    queryLevel_emit hlv0.q_ne_zero h₄ (eB.retained (eA.retained (e₁₁.retained r₁₀)))
  dsimp only at h₄
  obtain ⟨mq, tM, hmq, eM, h₄⟩ := nextMsg_emit h₄
  obtain ⟨nq, tN, hnq, eN, h₄⟩ := claimPow_emit h₄
  obtain ⟨bq, t₁₇, h₁₇, h₄⟩ := lift_bind h₄
  obtain ⟨e₁₇, -⟩ := sample_emitted h₁₇ (eN.retained (eM.retained (eQ.retained (eB.retained (eA.retained
    (e₁₁.retained r₁₀))))))
  have h₄ := lmodify_bind h₄
  obtain ⟨out, st₁₈, t₁₈, hloop, hcont⟩ := lbind_ok h₄
  rw [lforIn_range_eq] at hloop
  have r₁₇ := e₁₇.retained (eN.retained (eM.retained (eQ.retained (eB.retained (eA.retained (e₁₁.retained r₁₀))))))
  obtain ⟨-, eL⟩ := lforIn_emitted _ (levelFr sch p)
    (fun i v st => i < sch.levels.size - 1 → st.msg = sch.k0 + 4 + 7 * i ∧ st.ood = 3 + 2 * i ∧ st.cbn = 4 + 3 * i ∧
      st.qn = 1 + i ∧ st.cn = 1 + i ∧ v.1 = (sch.levels.getD (i + 1) default).cols + 4) (sch.levels.size - 1) (by
      intro i v st tE bv st' tE' hiB hP hretE hb
      obtain ⟨hmsg, hood, hcbn, hqn, hcn, hsplit⟩ := hP hiB
      rw [ite_eq_right (show ¬ v.1 < 4 by omega)] at hb
      obtain ⟨chs, st₁, t₁, hf, hb⟩ := lbind_ok hb
      rw [lforIn_range_eq] at hf
      obtain ⟨c1, c2, c3, c4, c5, eC⟩ := challenges_emitted p 4 0 _ _ _ _ _ _ hretE hf
      by_cases hi : i + 1 < sch.levels.size - 1
      · have hc1 := hcols (i + 1) (by omega)
        rw [ite_eq_left hi] at hb
        obtain ⟨_, t₂, h₂', hb⟩ := lift_bind hb
        have eK := absorbBytes_emitted h₂'
        obtain ⟨yA, mA, nA, stA, tA, hyA, hmA, hnA, a1, a2, a3, a4, a5, eA, hb⟩ :=
          oodIntro_emit (show v.1 - 4 + 1 ≠ 0 by omega) hb (eK.retained (eC.retained hretE))
        obtain ⟨yB, mB, nB, stB, tB, hyB, hmB, hnB, b1, b2, b3, b4, b5, eB, hb⟩ :=
          oodIntro_emit (show v.1 - 4 + 1 ≠ 0 by omega) hb (eA.retained (eK.retained (eC.retained hretE)))
        have hlvi := hlv (i + 1) (by omega)
        obtain ⟨qnI, cnI, rI, tQ, hqnI, hcnI, eQ, hb⟩ :=
          queryLevel_emit hlvi.q_ne_zero hb (eB.retained (eA.retained (eK.retained (eC.retained hretE))))
        obtain ⟨mq, tM, hmq, eM, hb⟩ := nextMsg_emit hb
        obtain ⟨nq, tN, hnq, eN, hb⟩ := claimPow_emit hb
        obtain ⟨bq, t₈, h₈, hb⟩ := lift_bind hb
        obtain ⟨e₈, -⟩ := sample_emitted h₈ (eN.retained (eM.retained (eQ.retained (eB.retained (eA.retained
          (eK.retained (eC.retained hretE)))))))
        have hb := lmodify_bind hb
        rw [ite_eq_right (show ¬ v.1 - 4 + 1 < 4 by omega)] at hb
        obtain ⟨rfl, rfl, rfl⟩ := lpure_ok hb
        refine ⟨_, rfl, fun hi' => ⟨?_, ?_, ?_, ?_, ?_, ?_⟩, ?_⟩
        · simp only; rw [b1, a1, c1, hmsg]; omega
        · simp only; rw [b2, a2, c2, hood]; omega
        · simp only; rw [b3, a3, c3, hcbn]; omega
        · simp only; rw [b4, a4, c4, hqn]; omega
        · simp only; rw [b5, a5, c5, hcn]; omega
        · simp only; omega
        · have e := eC.trans (eK.trans (eA.trans (eB.trans (eQ.trans (eM.trans (eN.trans e₈))))))
          have hyA' : yA = p.oodValues[3 + 2 * i]! := by rw [← getElem!_of_getElem? hyA, c2, hood]
          have hmA' : mA = p.messages[sch.k0 + 8 + 7 * i]! := by
            rw [← getElem!_of_getElem? hmA, c1, hmsg]; congr 1; omega
          have hnA' : nA = p.claimNonces[4 + 3 * i]! := by rw [← getElem!_of_getElem? hnA, c3, hcbn]
          have hyB' : yB = p.oodValues[4 + 2 * i]! := by
            rw [← getElem!_of_getElem? hyB, a2, c2, hood]; congr 1; omega
          have hmB' : mB = p.messages[sch.k0 + 9 + 7 * i]! := by
            rw [← getElem!_of_getElem? hmB, a1, c1, hmsg]; congr 1; omega
          have hnB' : nB = p.claimNonces[5 + 3 * i]! := by
            rw [← getElem!_of_getElem? hnB, a3, c3, hcbn]; congr 1; omega
          have hqnI' : qnI = p.queryNonces[1 + i]! := by rw [← getElem!_of_getElem? hqnI, b4, a4, c4, hqn]
          have hcnI' : cnI = p.consistencyNonces[1 + i]! := by rw [← getElem!_of_getElem? hcnI, b5, a5, c5, hcn]
          have hmq' : mq = p.messages[sch.k0 + 10 + 7 * i]! := by
            rw [← getElem!_of_getElem? hmq]; simp only; rw [b1, a1, c1, hmsg]; congr 1; omega
          have hnq' : nq = p.claimNonces[6 + 3 * i]! := by
            rw [← getElem!_of_getElem? hnq]; simp only; rw [b3, a3, c3, hcbn]; congr 1; omega
          rw [hmsg, hsplit, hyA', hmA', hnA', hyB', hmB', hnB', hqnI', hcnI', hmq', hnq'] at e
          simpa [levelFr, hi, introFr, queryFr, capFr, kFr, List.append_assoc] using e
      · rw [ite_eq_right hi] at hb
        have hb := lget_bind hb
        by_cases hmc₀ : (st₁.msg != p.messages.size) = true
        · rw [ite_eq_left hmc₀] at hb
          obtain ⟨_, _, _, h0, _⟩ := lbind_ok hb
          cases h0
        rw [ite_eq_right hmc₀] at hb
        by_cases hyr : (p.finalProof.yr.size != 2 ^ (v.1 - 4 + 1)) = true
        · rw [ite_eq_left hyr] at hb
          obtain ⟨_, _, _, h0, _⟩ := lbind_ok hb
          cases h0
        rw [ite_eq_right hyr] at hb
        obtain ⟨_, st₂, t₂, h₂', hq⟩ := lbind_ok hb
        rw [← Array.forIn_toList] at h₂'
        obtain ⟨hst₂, eY⟩ := forIn_absorbF_emitted _ h₂'
        have hlvR := hlv (sch.levels.size - 1) (by omega)
        obtain ⟨qnI, cnI, rI, tQ, hqnI, hcnI, eQ, hq⟩ :=
          queryLevel_emit hlvR.q_ne_zero hq (eY.retained (eC.retained hretE))
        obtain ⟨nq, tN, hnq, eN, hq⟩ := claimPow_emit hq
        obtain ⟨bq, t₈, h₈, hq⟩ := lift_bind hq
        obtain ⟨e₈, -⟩ := sample_emitted h₈ (eN.retained (eQ.retained (eY.retained (eC.retained hretE))))
        have hq := lmodify_bind hq
        obtain ⟨rfl, rfl, rfl⟩ := lpure_ok hq
        refine ⟨_, rfl, fun hi' => absurd hi' hi, ?_⟩
        have e := eC.trans (eY.trans (eQ.trans (eN.trans e₈)))
        have hqnI' : qnI = p.queryNonces[1 + i]! := by rw [← getElem!_of_getElem? hqnI, hst₂, c4, hqn]
        have hcnI' : cnI = p.consistencyNonces[1 + i]! := by rw [← getElem!_of_getElem? hcnI, hst₂, c5, hcn]
        have hnq' : nq = p.claimNonces[4 + 3 * i]! := by
          rw [← getElem!_of_getElem? hnq]; simp only; rw [hst₂, c3, hcbn]
        rw [hmsg, hqnI', hcnI', hnq'] at e
        simpa [levelFr, hi, queryFr, List.append_assoc] using e)
    (sch.levels.size - 1) 0 _ _ t₁₇ _ _ t₁₈ (by omega) (fun _ => by
      simp only at g1 g2 g3 g4 g5 ⊢
      exact ⟨by omega, by omega, by omega, by omega, by omega, by omega⟩) r₁₇ hloop
  obtain ⟨-, rfl, -⟩ := finalProg_ok hcont
  have e := e₁.trans (e₂.trans (e₃.trans (e₅.trans (e₆.trans (e₇.trans (e₈.trans (e₉.trans (e₁₀.trans (e₁₁.trans
    (eA.trans (eB.trans (eQ.trans (eM.trans (eN.trans (e₁₇.trans eL)))))))))))))))
  simp only at hy₀ hn₀ hm₀ g1 g2 g3 g4 g5 hyA hmA hnA hyB hmB hnB hqn₀ hcn₀ hmq hnq a1 a2 a3 a4 a5 b1 b2 b3 b4 b5
  rw [← getElem!_of_getElem? hy₀, ← getElem!_of_getElem? hn₀, ← getElem!_of_getElem? hm₀,
    ← getElem!_of_getElem? hyA, ← getElem!_of_getElem? hmA, ← getElem!_of_getElem? hnA,
    ← getElem!_of_getElem? hyB, ← getElem!_of_getElem? hmB, ← getElem!_of_getElem? hnB,
    ← getElem!_of_getElem? hqn₀, ← getElem!_of_getElem? hcn₀, ← getElem!_of_getElem? hmq,
    ← getElem!_of_getElem? hnq] at e
  simp only [b1, b3, b4, b5, a1, a2, a3, a4, a5, g1, g2, g3, g4, g5] at e
  rw [show 0 + 1 + sch.k0 + 1 + 1 = sch.k0 + 3 by omega, show 0 + 1 + sch.k0 + 1 = sch.k0 + 2 by omega,
    show 0 + 1 + sch.k0 = sch.k0 + 1 by omega, show (0 + 1 + 1 + 1 : ℕ) = 3 from rfl, show (0 + 1 + 1 : ℕ) = 2 from rfl,
    show (0 + 1 : ℕ) = 1 from rfl] at e
  simpa only [ligFrames, introFr, queryFr, capFr, List.append_assoc, List.cons_append, List.singleton_append,
    List.nil_append] using e

end FlockSoundness.Discharge.Exec

