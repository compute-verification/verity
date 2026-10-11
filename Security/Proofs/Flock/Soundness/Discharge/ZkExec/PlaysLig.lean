import Proofs.Flock.Soundness.Discharge.ZkExec.PlaysStatements
import Proofs.Flock.Soundness.Discharge.ZkExec.RunsLig

/-!
# The hiding Ligerito's runs

`zk_ligerito_plays`: `Zk.ligerito` refines the model's masked level 0 with Ligerito's levels compiled. The model's
games branch only on the schedule, so each of them has a run at any claim, polynomial and state, given its messages and
coins of the right sizes (`foldRoundsC_any`, `foldRoundsZK_any`, `drawKs_any`, `twoOodC_any`,
`queryAndBatchPadC_any`, `ligTailC_any`, `ligeritoPadZC_any`); and its Ligerito verdict is not a reject, since the last
recursive round records a query. What the executable gives is the coins: every round it consumes issued the number of
coins the model draws there (`fold_coins`, `lane_coins`, `queryLevel0_bind`).
-/

namespace FlockSoundness.Discharge.ZkExec

open Flock Refine Exec Game Model DecodeZL Transcript

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

variable {D α β : Type}

/-! ## Runs at any values -/

theorem ex_bind {g : Game α} {f : α → Game β} {ms₁ ms₂ ms : List Msg} {cs₁ cs₂ cs : List (Array F128)}
    (h₁ : ∃ a, Plays fOf g ms₁ cs₁ a) (h₂ : ∀ a, ∃ b, Plays fOf (f a) ms₂ cs₂ b) (hms : ms = ms₁ ++ ms₂)
    (hcs : cs = cs₁ ++ cs₂) : ∃ b, Plays fOf (g >>= f) ms cs b :=
  let ⟨a, ha⟩ := h₁
  let ⟨b, hb⟩ := h₂ a
  ⟨b, Plays.bind' ha hb hms hcs⟩

theorem ex_recv {M : Type} {f : M → Game β} {ms : List Msg} {cs : List (Array F128)} (m : M)
    (h : ∃ b, Plays fOf (f m) ms cs b) : ∃ b, Plays fOf (Game.recv M >>= f) (⟨M, m⟩ :: ms) cs b :=
  let ⟨b, hb⟩ := h
  ⟨b, Plays.bind' (Plays.recv M m) hb rfl rfl⟩

theorem ex_drawVec {n : ℕ} {iF : Fintype (Fin n → 𝔽)} {iN : Nonempty (Fin n → 𝔽)} {f : (Fin n → 𝔽) → Game β}
    {ms : List Msg} {cs : List (Array F128)} (c : Array F128) (hc : c.size = n) (h1 : n ≠ 1)
    (h : ∀ v, ∃ b, Plays fOf (f v) ms cs b) : ∃ b, Plays fOf (@Game.draw (Fin n → 𝔽) iF iN >>= f) ms (c :: cs) b :=
  let ⟨b, hb⟩ := h _
  ⟨b, Plays.bind' (@Plays.drawVec _ _ _ fOf n iF iN c hc h1) hb rfl rfl⟩

theorem ex_drawOne {iF : Fintype 𝔽} {iN : Nonempty 𝔽} {f : 𝔽 → Game β} {ms : List Msg} {cs : List (Array F128)}
    (c : Array F128) (hc : c.size = 1) (h : ∀ v, ∃ b, Plays fOf (f v) ms cs b) :
    ∃ b, Plays fOf (@Game.draw 𝔽 iF iN >>= f) ms (c :: cs) b :=
  let ⟨b, hb⟩ := h _
  ⟨b, Plays.bind' (@Plays.drawOne _ _ _ fOf iF iN c hc) hb rfl rfl⟩

theorem ex_drawK {f : 𝕂 → Game β} {ms : List Msg} {cs : List (Array F128)} (c : Array F128) (hc : c.size = 2)
    (h : ∀ r, ∃ b, Plays fOf (f r) ms cs b) : ∃ b, Plays fOf (Model.drawK execArith >>= f) ms (c :: cs) b := by
  have hk : Plays fOf (Model.drawK execArith) [] [c] (execArith.ofLimbs (fOf c[0]!) (fOf c[1]!)) := by
    unfold Model.drawK
    exact Plays.bind' (ms₂ := []) (cs₂ := []) (Plays.drawVec c hc (by omega)) (Plays.pure _) (by simp) (by simp)
  obtain ⟨b, hb⟩ := h _
  exact ⟨b, Plays.bind' hk hb rfl rfl⟩

theorem exQ_bind {Q : β → Prop} {g : Game α} {f : α → Game β} {ms₁ ms₂ ms : List Msg}
    {cs₁ cs₂ cs : List (Array F128)} (h₁ : ∃ a, Plays fOf g ms₁ cs₁ a) (h₂ : ∀ a, ∃ b, Plays fOf (f a) ms₂ cs₂ b ∧ Q b)
    (hms : ms = ms₁ ++ ms₂) (hcs : cs = cs₁ ++ cs₂) : ∃ b, Plays fOf (g >>= f) ms cs b ∧ Q b :=
  let ⟨a, ha⟩ := h₁
  let ⟨b, hb, hQ⟩ := h₂ a
  ⟨b, Plays.bind' ha hb hms hcs, hQ⟩

theorem exQ_bind0 {Q : β → Prop} {g : Game α} {f : α → Game β} {ms : List Msg} {cs₁ cs₂ cs : List (Array F128)}
    (h₁ : ∃ a, Plays fOf g [] cs₁ a) (h₂ : ∀ a, ∃ b, Plays fOf (f a) ms cs₂ b ∧ Q b) (hcs : cs = cs₁ ++ cs₂) :
    ∃ b, Plays fOf (g >>= f) ms cs b ∧ Q b :=
  exQ_bind h₁ h₂ (List.nil_append ms).symm hcs

theorem exQ_ret {Q' : α → Prop} {Q : β → Prop} {g : Game α} {h : α → β} {ms : List Msg} {cs : List (Array F128)}
    (h₁ : ∃ a, Plays fOf g ms cs a ∧ Q' a) (h₂ : ∀ a, Q' a → Q (h a)) :
    ∃ b, Plays fOf (g >>= fun a => pure (h a)) ms cs b ∧ Q b :=
  let ⟨a, ha, hQ⟩ := h₁
  ⟨h a, Plays.bind' ha (Plays.pure _) (List.append_nil ms).symm (List.append_nil cs).symm, h₂ a hQ⟩

theorem exQ_recv {Q : β → Prop} {M : Type} {f : M → Game β} {ms : List Msg} {cs : List (Array F128)} (m : M)
    (h : ∃ b, Plays fOf (f m) ms cs b ∧ Q b) : ∃ b, Plays fOf (Game.recv M >>= f) (⟨M, m⟩ :: ms) cs b ∧ Q b :=
  let ⟨b, hb, hQ⟩ := h
  ⟨b, Plays.bind' (Plays.recv M m) hb rfl rfl, hQ⟩

theorem exQ_drawVec {Q : β → Prop} {n : ℕ} {iF : Fintype (Fin n → 𝔽)} {iN : Nonempty (Fin n → 𝔽)}
    {f : (Fin n → 𝔽) → Game β} {ms : List Msg} {cs : List (Array F128)} (c : Array F128) (hc : c.size = n)
    (h1 : n ≠ 1) (h : ∀ v, ∃ b, Plays fOf (f v) ms cs b ∧ Q b) :
    ∃ b, Plays fOf (@Game.draw (Fin n → 𝔽) iF iN >>= f) ms (c :: cs) b ∧ Q b :=
  let ⟨b, hb, hQ⟩ := h _
  ⟨b, Plays.bind' (@Plays.drawVec _ _ _ fOf n iF iN c hc h1) hb rfl rfl, hQ⟩

theorem exQ_drawOne {Q : β → Prop} {iF : Fintype 𝔽} {iN : Nonempty 𝔽} {f : 𝔽 → Game β} {ms : List Msg}
    {cs : List (Array F128)} (c : Array F128) (hc : c.size = 1) (h : ∀ v, ∃ b, Plays fOf (f v) ms cs b ∧ Q b) :
    ∃ b, Plays fOf (@Game.draw 𝔽 iF iN >>= f) ms (c :: cs) b ∧ Q b :=
  let ⟨b, hb, hQ⟩ := h _
  ⟨b, Plays.bind' (@Plays.drawOne _ _ _ fOf iF iN c hc) hb rfl rfl, hQ⟩

/-! ## The model's phases at any values -/

theorem foldRoundsC_any (p : LigeritoProof) :
    ∀ (k a : ℕ) (sM : Opens 𝔽 D → 𝕂) (qM : Opens 𝔽 D → Model.Quad 𝕂) (cs : List (Array F128)),
      cs.length = k → (∀ c ∈ cs, c.size = 2) →
      ∃ out, Plays fOf (Model.foldRoundsC execArith k sM qM) (kMsgs p a k) cs out
  | 0, a, sM, qM, cs, hl, _ => by
      rw [List.eq_nil_of_length_eq_zero hl, Model.foldRoundsC, show kMsgs p a 0 = [] by simp [kMsgs]]
      exact ⟨_, Plays.pure _⟩
  | k + 1, a, sM, qM, c :: cs, hl, hs => by
      rw [Model.foldRoundsC, show kMsgs p a (k + 1) = kMsg p.messages[a]! :: kMsgs p (a + 1) k by
        simp [kMsgs, List.range'_succ]]
      refine ex_drawK c (hs c (by simp)) fun r => ?_
      refine ex_recv _ ?_
      exact ex_bind (foldRoundsC_any p k (a + 1) _ _ cs (by simpa using hl) fun c' h => hs c' (by simp [h]))
        (fun _ => ⟨_, Plays.pure _⟩) (List.append_nil _).symm (List.append_nil _).symm

theorem foldRoundsZK_any {np : ℕ} (P : ZkSession.Pads0 np 2) (p : LigeritoProof) :
    ∀ (k j a : ℕ) (s : ZkSession.LinK 𝕂 np) (q : ZkSession.QuadL 𝕂 np) (cs : List (Array F128)),
      cs.length = k → (∀ c ∈ cs, c.size = 2) →
      ∃ out, Plays fOf (ZkSession.foldRoundsZK execArith P k j s q) (kMsgs p a (k - 1)) cs out
  | 0, j, a, s, q, cs, hl, _ => by
      rw [List.eq_nil_of_length_eq_zero hl, ZkSession.foldRoundsZK, show kMsgs p a (0 - 1) = [] by simp [kMsgs]]
      exact ⟨_, Plays.pure _⟩
  | 1, j, a, s, q, c :: cs, hl, hs => by
      rw [List.eq_nil_of_length_eq_zero (by simpa using hl : cs.length = 0), ZkSession.foldRoundsZK,
        show kMsgs p a (1 - 1) = [] by simp [kMsgs]]
      exact ex_drawK c (hs c (by simp)) fun _ => ⟨_, Plays.pure _⟩
  | k + 2, j, a, s, q, c :: cs, hl, hs => by
      rw [ZkSession.foldRoundsZK, show kMsgs p a (k + 2 - 1) = kMsg p.messages[a]! :: kMsgs p (a + 1) (k + 1 - 1) by
        simp [kMsgs, List.range'_succ]]
      refine ex_drawK c (hs c (by simp)) fun r => ?_
      refine ex_recv _ ?_
      exact ex_bind (foldRoundsZK_any P p (k + 1) (j + 1) (a + 1) _ _ cs (by simpa using hl)
        fun c' h => hs c' (by simp [h])) (fun _ => ⟨_, Plays.pure _⟩) (List.append_nil _).symm
        (List.append_nil _).symm

theorem drawKs_any : ∀ (e : ℕ) (cs : List (Array F128)), cs.length = e → (∀ c ∈ cs, c.size = 2) →
    ∃ out, Plays fOf (Model.drawKs execArith e) [] cs out
  | 0, cs, hl, _ => by
      rw [List.eq_nil_of_length_eq_zero hl, Model.drawKs]
      exact ⟨_, Plays.pure _⟩
  | e + 1, c :: cs, hl, hs => by
      rw [Model.drawKs]
      exact ex_drawK c (hs c (by simp)) fun _ => ex_bind (drawKs_any e cs (by simpa using hl)
        fun c' h => hs c' (by simp [h])) (fun _ => ⟨_, Plays.pure _⟩) rfl (List.append_nil _).symm

theorem twoOodC_any (dim split : ℕ) (sM : Opens 𝔽 D → 𝕂) (qM : Opens 𝔽 D → Model.Quad 𝕂)
    (z₁ c₁ z₂ c₂ : Array F128) (hz₁ : z₁.size = dim) (hc₁ : c₁.size = 1) (hz₂ : z₂.size = dim) (hc₂ : c₂.size = 1)
    (hd : dim ≠ 1) (y₁ y₂ : F128) (m₁ m₂ : F256 × F256) :
    ∃ out, Plays fOf (Model.twoOodC execArith dim split sM qM) [fMsg y₁, kMsg m₁, fMsg y₂, kMsg m₂]
      [z₁, c₁, z₂, c₂] out := by
  unfold Model.twoOodC
  refine ex_drawVec z₁ hz₁ hd fun _ => ex_recv _ (ex_recv _ (ex_drawOne c₁ hc₁ fun _ => ?_))
  dsimp only
  exact ex_drawVec z₂ hz₂ hd fun _ => ex_recv _ (ex_recv _ (ex_drawOne c₂ hc₂ fun _ => ⟨_, Plays.pure _⟩))

theorem queryAndBatchPadC_any (r : ℕ) (lv : Accounting.Level) (col : List 𝔽 → ℕ → 𝕂) (start : ℕ)
    (st : LigStateC 𝔽 𝕂 D) (cw cα cβ : Array F128) (hcw : cw.size = lv.queries)
    (hcα : cα.size = Accounting.ceilLog2 lv.queries) (hcβ : cβ.size = 1) (hq1 : lv.queries ≠ 1)
    (hα1 : Accounting.ceilLog2 lv.queries ≠ 1) (m : F256 × F256) :
    ∃ st', Plays fOf (ZkLink.queryAndBatchPadC execArith r lv col start st) [kMsg m] [cw, cα, cβ] st' := by
  unfold ZkLink.queryAndBatchPadC
  refine ex_drawVec cw hcw hq1 fun _ => ex_drawVec cα hcα hα1 fun _ => ?_
  dsimp only
  exact ex_recv _ (ex_drawOne cβ hcβ fun _ => ⟨_, Plays.pure _⟩)

theorem ligTailC_any (r : ℕ) (sch : Accounting.Schedule) (tp : ℕ) (c₀ : 𝔽) (e : ℕ) (b : ℕ → 𝔽)
    (lv₀ lv₁ : Accounting.Level) (rest : List Accounting.Level) (z₀ : Fin (sch.m - 7) → 𝔽) (β₀ : 𝔽) (ini : List 𝕂)
    (ρ : Fin e → 𝕂) (ybar : Fin tp → 𝕂) (s : 𝕂) (m' : 𝕂 × 𝕂) (C₁ : Cap D) {msO msQ msR : List Msg}
    {csO csQ csR : List (Array F128)}
    (htwo : ∀ s q, ∃ out, Plays fOf (Model.twoOodC (D := D) execArith (sch.m - 7 - sch.k0 + 1) 0 s q) msO csO out)
    (hqb : ∀ col st, ∃ st', Plays fOf (ZkLink.queryAndBatchPadC (D := D) execArith r lv₀ col 0 st) msQ csQ st')
    (hrec : ∀ st, ∃ pend, Plays fOf (Model.recursiveRoundsC (D := D) execArith r (sch.m - 7) b ini sch.ext
      (lv₁ :: rest) st) msR csR pend ∧ pend ≠ PendL.reject) :
    ∃ pend, Plays fOf (ZkLink.ligTailC (D := D) execArith r sch tp c₀ e b lv₀ (lv₁ :: rest) z₀ β₀ ini ρ ybar s)
      (⟨𝕂 × 𝕂, m'⟩ :: ⟨Cap D, C₁⟩ :: (msO ++ (msQ ++ msR))) (csO ++ (csQ ++ csR)) pend ∧ pend ≠ PendL.reject := by
  unfold ZkLink.ligTailC
  refine exQ_recv m' ?_
  dsimp only
  refine exQ_recv C₁ ?_
  refine exQ_bind (htwo _ _) (fun ⟨s', q', oods⟩ => ?_) rfl rfl
  dsimp only
  exact exQ_bind (hqb _ _) (fun _ => hrec _) rfl rfl

theorem ligeritoPadZC_any {np : ℕ} (P : ZkSession.Pads0 np 2) (sch : Accounting.Schedule) (tp : ℕ) (c₀ : 𝔽)
    (r : ℕ) (Tm : 𝔽) (b : ℕ → 𝔽) {lv₀ lv₁ : Accounting.Level} {rest : List Accounting.Level}
    (hl : sch.levels = lv₀ :: lv₁ :: rest) (z₀ cβ₀ : Array F128) (hz₀ : z₀.size = sch.m - 7) (hm7 : sch.m - 7 ≠ 1)
    (hβ₀ : cβ₀.size = 1) (y₀ : 𝔽) (m₀ : 𝕂 × 𝕂) {msF : List Msg} {csF : List (Array F128)}
    (hfold : ∀ s q, ∃ out, Plays fOf (ZkSession.foldRoundsZK execArith P sch.k0 0 s q) msF csF out)
    (ev : Fin 2 → 𝕂) {csρ : List (Array F128)} (hρ : ∃ ρ, Plays fOf (Model.drawKs execArith 2) [] csρ ρ) (T' : 𝕂)
    (ybar : Fin tp → 𝕂) (m' : 𝕂 × 𝕂) (C₁ : Cap D) {msO msQ msR : List Msg} {csO csQ csR : List (Array F128)}
    (htwo : ∀ s q, ∃ out, Plays fOf (Model.twoOodC (D := D) execArith (sch.m - 7 - sch.k0 + 1) 0 s q) msO csO out)
    (hqb : ∀ col st, ∃ st', Plays fOf (ZkLink.queryAndBatchPadC (D := D) execArith r lv₀ col 0 st) msQ csQ st')
    (hrec : ∀ ini st, ∃ pend, Plays fOf (Model.recursiveRoundsC (D := D) execArith r (sch.m - 7) b ini sch.ext
      (lv₁ :: rest) st) msR csR pend ∧ pend ≠ PendL.reject) :
    ∃ out, Plays fOf (ZkLink.ligeritoPadZC (D := D) execArith P sch tp c₀ r Tm b)
      (⟨𝔽, y₀⟩ :: ⟨𝕂 × 𝕂, m₀⟩ :: (msF ++ ⟨Fin 2 → 𝕂, ev⟩ :: ⟨𝕂, T'⟩ :: ⟨Fin tp → 𝕂, ybar⟩ :: ⟨𝕂 × 𝕂, m'⟩ ::
        ⟨Cap D, C₁⟩ :: (msO ++ (msQ ++ msR))))
      (z₀ :: cβ₀ :: (csF ++ (csρ ++ (csO ++ (csQ ++ csR))))) out ∧ out.2 ≠ PendL.reject := by
  unfold ZkLink.ligeritoPadZC
  rw [hl]
  dsimp only
  refine exQ_drawVec z₀ hz₀ hm7 fun z => exQ_recv y₀ (exQ_drawOne cβ₀ hβ₀ fun β => ?_)
  refine exQ_recv m₀ (exQ_bind (hfold _ _) (fun ⟨ini, s⟩ => ?_) rfl rfl)
  dsimp only
  refine exQ_recv ev (exQ_bind0 hρ (fun ρ => ?_) rfl)
  refine exQ_recv T' (exQ_recv ybar ?_)
  exact exQ_ret (ligTailC_any r sch tp c₀ 2 b lv₀ lv₁ rest z β ini ρ ybar T' m' C₁ htwo hqb (hrec ini))
    fun _ h => h

/-! ## The executable's coins -/

theorem absorbK_ok {v : F256} {t t₁ : Flock.Transcript} {u : Unit} (h : absorbK v t = .ok (u, t₁)) : Keeps t t₁ := by
  obtain ⟨t₂, k, h'⟩ := absorbK_bind (f := fun _ => pure ()) (by rw [bind_pure]; exact h)
  obtain ⟨-, rfl⟩ := pure_ok h'
  exact k

theorem label_ok {l : String} {t t₁ : Flock.Transcript} {u : Unit} (h : label l t = .ok (u, t₁)) : Keeps t t₁ := by
  obtain ⟨t₂, k, h'⟩ := label_bind (f := fun _ => pure ()) (by rw [bind_pure]; exact h)
  obtain ⟨-, rfl⟩ := pure_ok h'
  exact k

theorem absorbSlice_ok {vs : Array F128} {t t₁ : Flock.Transcript} {u : Unit} (h : absorbSlice vs t = .ok (u, t₁)) :
    Keeps t t₁ := by
  obtain ⟨t₂, k, h'⟩ := absorbSlice_bind (f := fun _ => pure ()) (by rw [bind_pure]; exact h)
  obtain ⟨-, rfl⟩ := pure_ok h'
  exact k

theorem consumed_length (rs : Array Round) (k k' : ℕ) : (consumed rs k k').length = k' - k := by
  simp [consumed]

theorem consumed_sizes {rs : Array Round} (hwf : RoundsWF rs) {k K n : ℕ} (h : ∀ j < K, (rs[k + j]!).n = n) :
    ∀ c ∈ consumed rs k (k + K), c.size = n := by
  intro c hc
  rw [consumed_add] at hc
  obtain ⟨j, hj, rfl⟩ := List.mem_map.1 hc
  rw [RoundsWF.get hwf, h j (List.mem_range.1 hj)]

/-- Ligerito's sumcheck challenges: `k` rounds of two coins each. -/
theorem fold_coins (p : LigeritoProof) :
    ∀ (k a : ℕ) (chs chs' : Array F256) (stE stE' : LState) (t t' : Flock.Transcript),
      forIn (List.range' a k) chs (fun _ chs => challenge p >>= fun x => pure (ForInStep.yield (chs.push x)))
        stE t = .ok ((chs', stE'), t') →
      t'.rounds = t.rounds ∧ t'.k = t.k + k ∧ ∀ j < k, (t.rounds[t.k + j]!).n = 2 := by
  intro k
  induction k with
  | zero =>
      intro a chs chs' stE stE' t t' h
      simp only [List.range'_zero, List.forIn_nil] at h
      obtain ⟨rfl, rfl, rfl⟩ := lpure_ok h
      exact ⟨rfl, rfl, fun j hj => absurd hj (Nat.not_lt_zero j)⟩
  | succ k ih =>
      intro a chs chs' stE stE' t t' h
      rw [List.range'_succ, List.forIn_cons] at h
      obtain ⟨r, st₁, t₁, h₁, h₂⟩ := lbind_ok h
      obtain ⟨-, hn, m, t₂, -, hr₂, hk₂, h₁⟩ := challenge_bind h₁
      obtain ⟨rfl, rfl, rfl⟩ := lpure_ok h₁
      simp only at h₂
      obtain ⟨hr, hk, hn'⟩ := ih (a + 1) _ chs' _ stE' t₁ t' h₂
      refine ⟨hr.trans hr₂, by rw [hk, hk₂]; omega, fun j hj => ?_⟩
      rcases j with _ | j
      · simpa using hn
      · have := hn' j (by omega)
        rw [show t.k + (j + 1) = t₁.k + j by omega, ← hr₂]
        exact this

/-- The lane rounds: one `K` coin each, and the last one's two `ρ` coins after it. -/
theorem lane_coins {σ : Type} (body : ℕ → σ → LM (ForInStep σ)) (K : ℕ)
    (hbody : ∀ i s st t b st' t', i < K → body i s st t = .ok ((b, st'), t') →
      ∃ s', b = .yield s' ∧ t'.rounds = t.rounds ∧ t'.k = t.k + 1 + (if i + 1 = K then 2 else 0) ∧
        (t.rounds[t.k]!).n = 2 ∧ (i + 1 = K → (t.rounds[t.k + 1]!).n = 2 ∧ (t.rounds[t.k + 2]!).n = 2)) :
    ∀ n a s st t s' st' t', a + n = K → 0 < n → forIn (List.range' a n) s body st t = .ok ((s', st'), t') →
      t'.rounds = t.rounds ∧ t'.k = t.k + n + 2 ∧ ∀ j < n + 2, (t.rounds[t.k + j]!).n = 2
  | 0, _, _, _, _, _, _, _, _, h0, _ => absurd h0 (lt_irrefl 0)
  | 1, a, s, st, t, s', st', t', hK, _, h => by
      rw [List.range'_succ, List.forIn_cons] at h
      obtain ⟨b, st₁, t₁, h₁, h₂⟩ := lbind_ok h
      obtain ⟨s₁, rfl, hr, hk, hn, hlast⟩ := hbody a s st t b st₁ t₁ (by omega) h₁
      simp only [List.range'_zero, List.forIn_nil] at h₂
      obtain ⟨rfl, rfl, rfl⟩ := lpure_ok h₂
      obtain ⟨hn1, hn2⟩ := hlast (by omega)
      rw [ite_eq_left (show a + 1 = K by omega)] at hk
      refine ⟨hr, by omega, fun j hj => ?_⟩
      rcases (show j = 0 ∨ j = 1 ∨ j = 2 by omega) with rfl | rfl | rfl
      · simpa using hn
      · exact hn1
      · exact hn2
  | n + 2, a, s, st, t, s', st', t', hK, _, h => by
      rw [List.range'_succ, List.forIn_cons] at h
      obtain ⟨b, st₁, t₁, h₁, h₂⟩ := lbind_ok h
      obtain ⟨s₁, rfl, hr, hk, hn, -⟩ := hbody a s st t b st₁ t₁ (by omega) h₁
      rw [ite_eq_right (show ¬ a + 1 = K by omega)] at hk
      simp only at h₂
      obtain ⟨hr', hk', hn'⟩ :=
        lane_coins body K hbody (n + 1) (a + 1) s₁ st₁ t₁ s' st' t' (by omega) (by omega) h₂
      refine ⟨hr'.trans hr, by omega, fun j hj => ?_⟩
      rcases j with _ | j
      · simpa using hn
      · have := hn' j (by omega)
        rw [show t.k + (j + 1) = t₁.k + j by omega, ← hr]
        exact this

/-- Level 0's queries draw `q` query coins and `⌊log₂(q − 1)⌋ + 1` consistency coins. -/
theorem queryLevel0_bind {ms : MerkleScheme} {p : LigeritoProof} {lv : Flock.Level} {cap : Array ByteArray}
    {fold : Array F256} (hq : lv.q ≠ 0)
    {f : Array ℕ × Array F128 × F256 → LM β} {st st' : LState} {t t' : Flock.Transcript} {b : β}
    (h : (Zk.queryLevel0 ms p lv cap fold >>= f) st t = .ok ((b, st'), t')) :
    (t.rounds[t.k]!).n = lv.q ∧ (t.rounds[t.k + 1]!).n = (lv.q - 1).log2 + 1 ∧
      ∃ r t₁, t₁.rounds = t.rounds ∧ t₁.k = t.k + 2 ∧
        f r { st with qn := st.qn + 1, cn := st.cn + 1 } t₁ = .ok ((b, st'), t') := by
  obtain ⟨r, st₁, t₁, h₁, h₂⟩ := lbind_ok h
  unfold Zk.queryLevel0 at h₁
  have h₁ := lget_bind h₁
  obtain ⟨qn, _, h₁⟩ := nonceFrom_bind h₁
  obtain ⟨cn, _, h₁⟩ := nonceFrom_bind h₁
  have h₁ := lset_bind h₁
  obtain ⟨_, t₃, h₃, h₁⟩ := lift_bind h₁
  obtain ⟨t₃', k₃, h₃'⟩ := pow_bind (f := fun _ => pure ()) (by rw [bind_pure]; exact h₃)
  obtain ⟨rfl, rfl⟩ := pure_ok h₃'
  obtain ⟨coins, t₄, h₄, h₁⟩ := lift_bind h₁
  obtain ⟨hn₄, -, hr₄, hk₄⟩ := sampleVec_ok hq h₄
  obtain ⟨_, t₅, h₅, h₁⟩ := lift_bind h₁
  obtain ⟨t₅', k₅, h₅'⟩ := pow_bind (f := fun _ => pure ()) (by rw [bind_pure]; exact h₅)
  obtain ⟨rfl, rfl⟩ := pure_ok h₅'
  obtain ⟨alpha, t₆, h₆, h₁⟩ := lift_bind h₁
  obtain ⟨hn₆, -, hr₆, hk₆⟩ := sampleVec_ok (by omega) h₆
  cases hmc : merkleCheck ms cap lv.cap lv.d (positions lv coins) p.initialProof.rows p.initialProof.siblings
      p.initialProof.salts (2 ^ lv.lanes + Zk.EXTRA_LANES) with
  | error e => simp only [hmc] at h₁; obtain ⟨_, _, _, h₀, _⟩ := lbind_ok h₁; cases h₀
  | ok u =>
    simp only [hmc] at h₁
    simp only [List.forIn_pure_yield_eq_foldl, Std.Legacy.Range.forIn_eq_forIn_range', pure_bind] at h₁
    obtain ⟨rfl, rfl, rfl⟩ := lpure_ok h₁
    refine ⟨by rw [← k₃.2, ← k₃.1]; exact hn₄, ?_, _, _, by rw [hr₆, k₅.2, hr₄, k₃.2],
      by rw [hk₆, k₅.1, hk₄, k₃.1], h₂⟩
    rw [show t.k + 1 = t₅.k by rw [k₅.1, hk₄, k₃.1], show t.rounds = t₅.rounds by rw [k₅.2, hr₄, k₃.2]]
    exact hn₆

/-! ## Ligerito under `--zk` -/

set_option maxHeartbeats 4000000 in
theorem zk_ligerito_plays : ZkLigeritoPlays := by
  intro ms sch T cap p basisAt zl t t' rec hdim hlv hα1 hk0 hwf h
  obtain ⟨hn7, h2, hl0, hc0, hl4, hcols, hext⟩ := hdim
  have hc01 := hcols 0 (by omega)
  have hn3 : 3 ≤ sch.n := by omega
  unfold Zk.ligerito at h
  obtain ⟨hcap, h⟩ := check_bind h
  obtain ⟨hzl, h⟩ := check_bind h
  obtain ⟨t₁, k₁, h⟩ := label_bind h
  obtain ⟨t₂, k₂, h⟩ := absorbF_bind h
  obtain ⟨t₃, k₃, h⟩ := absorbBytes_bind h
  obtain ⟨t₃', k₃', h⟩ := label_bind h
  obtain ⟨⟨rc', stF⟩, t₄, h₄, h⟩ := bind_ok h
  obtain ⟨-, rfl⟩ := pure_ok h
  rw [StateT.run] at h₄
  obtain ⟨z, t₅, h₅, h₄⟩ := lift_bind h₄
  obtain ⟨hn₅, rfl, hr₅, hk₅⟩ := sampleVec_ok (by omega) h₅
  obtain ⟨y₀, -, h₄⟩ := nextOod_bind h₄
  obtain ⟨_, t₆, h₆, h₄⟩ := lift_bind h₄
  have k₆ := absorbF_ok h₆
  obtain ⟨t₇, k₇, h₄⟩ := claimPow_bind h₄
  obtain ⟨bc, t₈, h₈, h₄⟩ := lift_bind h₄
  obtain ⟨-, hn₈, rfl, hr₈, hk₈⟩ := sample_ok h₈
  have h₄ := lmodify_bind h₄
  obtain ⟨m₀, t₉, -, k₉, h₄⟩ := nextMsg_bind h₄
  have h₄ := lmodify_bind h₄
  obtain ⟨lane, st₁₀, t₁₀, h₁₀, h₄⟩ := lbind_ok h₄
  rw [lforIn_range_eq] at h₁₀
  obtain ⟨hr₁₀, hk₁₀, hn₁₀⟩ := lane_coins _ sch.k0 (by
      intro i v st tE bv st' tE' hiB hb
      obtain ⟨r, tR, hR', hb⟩ := lift_bind hb
      obtain ⟨-, hnR, -, hrR, hkR⟩ := sampleK_ok hR'
      have hb := lmodify_bind hb
      try dsimp only at hb
      by_cases hj : i + 1 = sch.k0
      · rw [ite_eq_left (show (i + 1 == sch.k0) = true by simp [hj])] at hb
        obtain ⟨_, t₂, h₂, hb⟩ := lift_bind hb
        obtain ⟨f₂k, f₂r⟩ := absorbK_ok h₂
        obtain ⟨_, t₃, h₃, hb⟩ := lift_bind hb
        obtain ⟨f₃k, f₃r⟩ := absorbK_ok h₃
        obtain ⟨ρ₀, t₄, h₄, hb⟩ := lift_bind hb
        obtain ⟨-, hn₄, -, hr₄, hk₄⟩ := sampleK_ok h₄
        obtain ⟨ρ₁, t₅, h₅, hb⟩ := lift_bind hb
        obtain ⟨-, hn₅, -, hr₅, hk₅⟩ := sampleK_ok h₅
        try dsimp only at hb
        by_cases hρ : (!Zk.rhoIndependent (ρ₀, ρ₁)) = true
        · rw [ite_eq_left hρ] at hb
          obtain ⟨_, _, _, h0, _⟩ := lbind_ok hb
          cases h0
        rw [ite_eq_right hρ] at hb
        try simp only [pure_bind] at hb
        obtain ⟨_, t₆, h₆, hb⟩ := lift_bind hb
        obtain ⟨f₆k, f₆r⟩ := label_ok h₆
        obtain ⟨_, t₇, h₇, hb⟩ := lift_bind hb
        obtain ⟨f₇k, f₇r⟩ := absorbK_ok h₇
        obtain ⟨_, t₈, h₈, hb⟩ := lift_bind hb
        obtain ⟨f₈k, f₈r⟩ := absorbSlice_ok h₈
        have hb := lmodify_bind hb
        obtain ⟨m, t₉, -, ⟨f₉k, f₉r⟩, hb⟩ := nextMsg_bind hb
        have hb := lmodify_bind hb
        obtain ⟨rfl, rfl, rfl⟩ := lpure_ok hb
        refine ⟨_, rfl, by rw [f₉r, f₈r, f₇r, f₆r, hr₅, hr₄, f₃r, f₂r, hrR], by rw [ite_eq_left hj]; omega, hnR,
          fun _ => ⟨?_, ?_⟩⟩
        · rw [show tE.k + 1 = t₃.k by omega, show tE.rounds = t₃.rounds by rw [f₃r, f₂r, hrR]]
          exact hn₄
        · rw [show tE.k + 2 = t₄.k by omega, show tE.rounds = t₄.rounds by rw [hr₄, f₃r, f₂r, hrR]]
          exact hn₅
      · rw [ite_eq_right (show ¬ (i + 1 == sch.k0) = true by simp [hj])] at hb
        try simp only [pure_bind] at hb
        obtain ⟨m, t₉, -, ⟨f₉k, f₉r⟩, hb⟩ := nextMsg_bind hb
        have hb := lmodify_bind hb
        obtain ⟨rfl, rfl, rfl⟩ := lpure_ok hb
        exact ⟨_, rfl, by rw [f₉r, hrR], by rw [ite_eq_right hj]; omega, hnR, fun h => absurd h hj⟩)
    sch.k0 0 _ _ t₉ _ _ t₁₀ (by omega) (by omega) h₁₀
  try dsimp only at h₄
  obtain ⟨_, t₁₁, h₁₁, h₄⟩ := lift_bind h₄
  have k₁₁ := absorbBytes_ok h₁₁
  obtain ⟨hnA, hnB, yA, mA, t₁₂, -, -, hr₁₂, hk₁₂, h₄⟩ :=
    oodIntro_bind (show sch.n - sch.k0 + 1 ≠ 0 by omega) h₄
  obtain ⟨hnC, hnD, yB, mB, t₁₃, -, -, hr₁₃, hk₁₃, h₄⟩ :=
    oodIntro_bind (show sch.n - sch.k0 + 1 ≠ 0 by omega) h₄
  have hlv0 := hlv 0 (by omega)
  obtain ⟨hnQ, hnQ', ⟨pos0, al0, e0⟩, t₁₄, hr₁₄, hk₁₄, h₄⟩ := queryLevel0_bind hlv0.q_ne_zero h₄
  dsimp only at h₄
  obtain ⟨extra, stX, tX, hX, h₄'⟩ := lbind_ok h₄
  clear h₄
  rw [lforIn_range_eq, List.forIn_pure_yield_eq_foldl] at hX
  obtain ⟨rfl, rfl, rfl⟩ := lpure_ok hX
  obtain ⟨mq, t₁₅, -, k₁₅, h₄⟩ := nextMsg_bind h₄'
  obtain ⟨t₁₆, k₁₆, h₄⟩ := claimPow_bind h₄
  obtain ⟨bq, t₁₇, h₁₇, h₄⟩ := lift_bind h₄
  obtain ⟨-, hn₁₇, rfl, hr₁₇, hk₁₇⟩ := sample_ok h₁₇
  have h₄ := lmodify_bind h₄
  obtain ⟨out, st₁₈, t₁₈, hloop, hcont⟩ := lbind_ok h₄
  rw [lforIn_range_eq] at hloop
  have hfin := finalProgZ_ok hcont
  rw [hfin]
  have er₃ : t₃'.rounds = t.rounds := by rw [k₃'.2, k₃.2, k₂.2, k₁.2]
  have ek₃ : t₃'.k = t.k := by rw [k₃'.1, k₃.1, k₂.1, k₁.1]
  have er₇ : t₇.rounds = t.rounds := by rw [k₇.2, k₆.2, hr₅, er₃]
  have ek₇ : t₇.k = t.k + 1 := by rw [k₇.1, k₆.1, hk₅, ek₃]
  have er₉ : t₉.rounds = t.rounds := by rw [k₉.2, hr₈, er₇]
  have ek₉ : t₉.k = t.k + 2 := by rw [k₉.1, hk₈, ek₇]
  have er₁₁ : t₁₁.rounds = t.rounds := by rw [k₁₁.2, hr₁₀, er₉]
  have ek₁₁ : t₁₁.k = t.k + 2 + sch.k0 + 2 := by rw [k₁₁.1, hk₁₀, ek₉]
  have er₁₂ : t₁₂.rounds = t.rounds := hr₁₂.trans er₁₁
  have ek₁₂ : t₁₂.k = t.k + 2 + sch.k0 + 4 := by rw [hk₁₂, ek₁₁]
  have er₁₃ : t₁₃.rounds = t.rounds := hr₁₃.trans er₁₂
  have ek₁₃ : t₁₃.k = t.k + 2 + sch.k0 + 6 := by rw [hk₁₃, ek₁₂]
  have er₁₆ : t₁₆.rounds = t.rounds := by rw [k₁₆.2, k₁₅.2, hr₁₄, er₁₃]
  have ek₁₆ : t₁₆.k = t.k + 2 + sch.k0 + 8 := by rw [k₁₆.1, k₁₅.1, hk₁₄, ek₁₃]
  have er₁₇ : t₁₇.rounds = t.rounds := hr₁₇.trans er₁₆
  have ek₁₇ : t₁₇.k = t.k + 2 + sch.k0 + 9 := by rw [hk₁₇, ek₁₆]
  have hd1 : (sch.levels.toList.map levelOf).drop 1 =
      levelOf (sch.levels.getD 1 default) :: (sch.levels.toList.map levelOf).drop 2 := drop_levels sch 1 (by omega)
  have hL : ∀ (r : ℕ) (b : ℕ → 𝔽) (ini : List 𝕂) (stM : LigStateC 𝔽 𝕂 Dig), ∃ pend : PendL 𝔽 Dig,
      t₁₈.rounds = t₁₇.rounds ∧ t₁₇.k ≤ t₁₈.k ∧
      Plays fOf (Model.recursiveRoundsC execArith r (sch.m - 7) b ini sch.ext
        (levelOf (sch.levels.getD 1 default) :: (sch.levels.toList.map levelOf).drop 2) stM)
        (recMsgs p capDig sch.ext (sch.levels.size - 1) 0 (sch.k0 + 4) 3) (consumed t₁₇.rounds t₁₇.k t₁₈.k) pend ∧
      pend ≠ PendL.reject := by
    intro r b ini stM
    have hLC := loop_compose _ (sch.levels.size - 1)
      (fun i stM => Model.recursiveRoundsC execArith r (sch.m - 7) b ini sch.ext
        ((sch.levels.toList.map levelOf).drop (i + 1)) stM)
      (fun i v _ tE _ => RoundsWF tE.rounds ∧ v.1 = (sch.levels.getD (i + 1) default).cols + 4)
      (fun i _ => recMsgs p capDig sch.ext (sch.levels.size - 1 - i) i (sch.k0 + 4 + 7 * i) (3 + 2 * i))
      (fun _ _ _ pend => pend ≠ PendL.reject) ?hstep ?hlast (sch.levels.size - 1) 0 _ _ _ out st₁₈ t₁₈ stM
      (by omega) (by omega) ⟨by rw [er₁₇]; exact hwf, by show sch.n - sch.k0 + 1 = _; omega⟩ hloop
    case hstep =>
      clear * - hn7 h2 hl0 hc0 hl4 hcols hext hlv hα1
      intro i v stE tE bv stE' tE' stM hi hrel hb
      obtain ⟨hwfE, hsplit⟩ := hrel
      have hc1 := hcols (i + 1) (by omega)
      have hl1 := hl4 (i + 1) (by omega) (by omega)
      have hl2 := hl4 (i + 1 + 1) (by omega) (by omega)
      try dsimp only at hb
      rw [ite_eq_right (show ¬ v.1 < 4 by omega)] at hb
      obtain ⟨chs, st₁, t₁, hf, hb⟩ := lbind_ok hb
      rw [lforIn_range_eq] at hf
      obtain ⟨hr₁, hk₁, hn₁⟩ := fold_coins p 4 0 #[] chs stE st₁ tE t₁ hf
      rw [ite_eq_left hi] at hb
      obtain ⟨_, t₂, h₂, hb⟩ := lift_bind hb
      have k₂ := absorbBytes_ok h₂
      obtain ⟨hnA, hnB, yA, mA, t₃, -, -, hr₃, hk₃, hb⟩ := oodIntro_bind (by omega) hb
      obtain ⟨hnC, hnD, yB, mB, t₄, -, -, hr₄, hk₄, hb⟩ := oodIntro_bind (by omega) hb
      have hlvi := hlv (i + 1) (by omega)
      obtain ⟨hnQ, hnQ', -, t₅, hr₅, hk₅, hb⟩ := queryLevel_bind hlvi.q_ne_zero hb
      obtain ⟨mq, t₆, -, k₆, hb⟩ := nextMsg_bind hb
      obtain ⟨t₇, k₇, hb⟩ := claimPow_bind hb
      obtain ⟨bq, t₈, h₈, hb⟩ := lift_bind hb
      obtain ⟨-, hn₈, rfl, hr₈, hk₈⟩ := sample_ok h₈
      have hb := lmodify_bind hb
      rw [ite_eq_right (show ¬ v.1 - 4 + 1 < 4 by omega)] at hb
      obtain ⟨rfl, rfl, rfl⟩ := lpure_ok hb
      have er₂ : t₂.rounds = tE.rounds := k₂.2.trans hr₁
      have ek₂ : t₂.k = tE.k + 4 := by rw [k₂.1, hk₁]
      have er₃ : t₃.rounds = tE.rounds := hr₃.trans er₂
      have ek₃ : t₃.k = tE.k + 6 := by rw [hk₃, ek₂]
      have er₄ : t₄.rounds = tE.rounds := hr₄.trans er₃
      have ek₄ : t₄.k = tE.k + 8 := by rw [hk₄, ek₃]
      have er₇ : t₇.rounds = tE.rounds := by rw [k₇.2, k₆.2, hr₅, er₄]
      have ek₇ : t₇.k = tE.k + 10 := by rw [k₇.1, k₆.1, hk₅, ek₄]
      have er₈ : tE'.rounds = tE.rounds := hr₈.trans er₇
      have ek₈ : tE'.k = tE.k + 11 := by rw [hk₈, ek₇]
      have hdim : (levelOf (sch.levels.getD (i + 1 + 1) default)).logCols +
          (levelOf (sch.levels.getD (i + 1 + 1) default)).lanes = v.1 - 4 + 1 := by
        show (sch.levels.getD (i + 1 + 1) default).cols + (sch.levels.getD (i + 1 + 1) default).lanes = _
        omega
      obtain ⟨⟨rs, sM', qM'⟩, hfoldM⟩ := foldRoundsC_any (D := Dig) p 4 (sch.k0 + 4 + 7 * i) stM.s stM.q
        (consumed tE.rounds tE.k (tE.k + 4)) (by rw [consumed_length]; omega) (consumed_sizes hwfE hn₁)
      obtain ⟨⟨sT, qT, ctxs⟩, htwo⟩ := twoOodC_any (D := Dig)
        ((levelOf (sch.levels.getD (i + 1 + 1) default)).logCols +
          (levelOf (sch.levels.getD (i + 1 + 1) default)).lanes) (stM.levels ++ [rs]).length sM' qM'
        (tE.rounds[tE.k + 4]!).coins (tE.rounds[tE.k + 5]!).coins (tE.rounds[tE.k + 6]!).coins
        (tE.rounds[tE.k + 7]!).coins
        (by rw [RoundsWF.get hwfE, hdim, show tE.k + 4 = t₂.k by omega, ← er₂]; exact hnA)
        (by rw [RoundsWF.get hwfE, show tE.k + 5 = t₂.k + 1 by omega, ← er₂]; exact hnB)
        (by rw [RoundsWF.get hwfE, hdim, show tE.k + 6 = t₃.k by omega, ← er₃]; exact hnC)
        (by rw [RoundsWF.get hwfE, show tE.k + 7 = t₃.k + 1 by omega, ← er₃]; exact hnD) (by omega)
        p.oodValues[3 + 2 * i]! p.oodValues[3 + 2 * i + 1]! p.messages[sch.k0 + 4 + 7 * i + 4]!
        p.messages[sch.k0 + 4 + 7 * i + 5]!
      have hcw : (tE.rounds[tE.k + 8]!).coins.size = (levelOf (sch.levels.getD (i + 1) default)).queries := by
        rw [RoundsWF.get hwfE, show tE.k + 8 = t₄.k by omega, ← er₄, hnQ]; rfl
      have hcα : (tE.rounds[tE.k + 9]!).coins.size =
          Accounting.ceilLog2 (levelOf (sch.levels.getD (i + 1) default)).queries := by
        rw [RoundsWF.get hwfE, show tE.k + 9 = t₄.k + 1 by omega, ← er₄, hnQ']; exact hlvi.2.2.2.1
      have hcβ : (tE.rounds[tE.k + 10]!).coins.size = 1 := by
        rw [RoundsWF.get hwfE, show tE.k + 10 = t₇.k by omega, ← er₇, hn₈]
      refine ⟨_, { qbOut r (levelOf (sch.levels.getD (i + 1) default)) rs (stM.levels ++ [rs]).length
            { stM with s := sT, q := qT, levels := stM.levels ++ [rs], oods := stM.oods ++ ctxs }
            (tE.rounds[tE.k + 8]!).coins (tE.rounds[tE.k + 9]!).coins (tE.rounds[tE.k + 10]!).coins true
            p.messages[sch.k0 + 4 + 7 * i + 6]! with
          level := stM.level + 1, caps := stM.caps ++ [capDig p.recursiveCaps[i + 1]!],
          prevLevel := levelOf (sch.levels.getD (i + 1 + 1) default) },
        kMsgs p (sch.k0 + 4 + 7 * i) 4 ++ ⟨Cap Dig, capDig p.recursiveCaps[i + 1]!⟩ ::
          [fMsg p.oodValues[3 + 2 * i]!, kMsg p.messages[sch.k0 + 4 + 7 * i + 4]!,
            fMsg p.oodValues[3 + 2 * i + 1]!, kMsg p.messages[sch.k0 + 4 + 7 * i + 5]!] ++
          [kMsg p.messages[sch.k0 + 4 + 7 * i + 6]!],
        rfl, ⟨by rw [er₈]; exact hwfE, by show v.1 - 4 + 1 = _; omega⟩, er₈, by omega, ?tape, ?plays⟩
      case tape =>
        obtain ⟨j, hj⟩ : ∃ j, sch.levels.size - 1 - i = j + 2 := ⟨sch.levels.size - 1 - i - 2, by omega⟩
        rw [hj, show sch.levels.size - 1 - (i + 1) = j + 1 by omega,
          show sch.k0 + 4 + 7 * (i + 1) = sch.k0 + 4 + 7 * i + 7 by omega,
          show 3 + 2 * (i + 1) = 3 + 2 * i + 2 by omega]
        simp only [recMsgs]
      case plays =>
        intro msR csR pend hrec
        try dsimp only at hrec ⊢
        rw [drop_levels sch (i + 1) (by omega), drop_levels sch (i + 1 + 1) (by omega)]
        rw [drop_levels sch (i + 1 + 1) (by omega)] at hrec
        have hfold' : Plays fOf (Model.foldRoundsC execArith (levelOf (sch.levels.getD (i + 1) default)).lanes
            stM.s stM.q) (kMsgs p (sch.k0 + 4 + 7 * i) 4) (consumed tE.rounds tE.k (tE.k + 4)) (rs, sM', qM') := by
          rw [show (levelOf (sch.levels.getD (i + 1) default)).lanes = 4 from hl1]; exact hfoldM
        have hmain := iterNL_model r (sch.m - 7) b ini sch.ext
          (levelOf (sch.levels.getD (i + 1) default)) (levelOf (sch.levels.getD (i + 1 + 1) default))
          ((sch.levels.toList.map levelOf).drop (i + 1 + 1 + 1)) stM hfold' (capDig p.recursiveCaps[i + 1]!) htwo
          (tE.rounds[tE.k + 8]!).coins (tE.rounds[tE.k + 9]!).coins (tE.rounds[tE.k + 10]!).coins hcw hcα hcβ
          hlvi.q_ne_one (hα1 (i + 1) (by omega)) p.messages[sch.k0 + 4 + 7 * i + 6]! hrec
        have hcs : consumed tE.rounds tE.k tE'.k = consumed tE.rounds tE.k (tE.k + 4) ++
            [(tE.rounds[tE.k + 4]!).coins, (tE.rounds[tE.k + 5]!).coins, (tE.rounds[tE.k + 6]!).coins,
              (tE.rounds[tE.k + 7]!).coins] ++
            [(tE.rounds[tE.k + 8]!).coins, (tE.rounds[tE.k + 9]!).coins, (tE.rounds[tE.k + 10]!).coins] := by
          rw [ek₈, consumed_append tE.rounds (show tE.k ≤ tE.k + 4 by omega) (show tE.k + 4 ≤ tE.k + 11 by omega),
            show tE.k + 11 = tE.k + 4 + 7 by omega, consumed_add tE.rounds (tE.k + 4) 7]
          simp [List.range_succ, Nat.add_assoc]
        rw [hcs]
        exact hmain
    case hlast =>
      clear * - hn7 h2 hl0 hc0 hl4 hcols hext hlv hα1
      intro i v stE tE bv stE' tE' stM hi hrel hb
      obtain ⟨hwfE, hsplit⟩ := hrel
      have hl1 := hl4 (i + 1) (by omega) (by omega)
      try dsimp only at hb
      rw [ite_eq_right (show ¬ v.1 < 4 by omega)] at hb
      obtain ⟨chs, st₁, t₁, hf, hb⟩ := lbind_ok hb
      rw [lforIn_range_eq] at hf
      obtain ⟨hr₁, hk₁, hn₁⟩ := fold_coins p 4 0 #[] chs stE st₁ tE t₁ hf
      rw [ite_eq_right (show ¬ i + 1 < sch.levels.size - 1 by omega)] at hb
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
      obtain ⟨_, st₂, t₂, h₂, hb₂⟩ := lbind_ok hb
      clear hb
      rw [← Array.forIn_toList] at h₂
      obtain ⟨rfl, k₂⟩ := forIn_absorbF_keeps _ h₂
      have hb := hb₂
      clear hb₂
      have hlvR := hlv (sch.levels.size - 1) (by omega)
      obtain ⟨hnQ, hnQ', -, t₃, hr₃, hk₃, hb⟩ := queryLevel_bind hlvR.q_ne_zero hb
      obtain ⟨t₄, k₄, hb⟩ := claimPow_bind hb
      obtain ⟨bq, t₅, h₅, hb⟩ := lift_bind hb
      obtain ⟨-, hn₅, rfl, hr₅, hk₅⟩ := sample_ok h₅
      have hb := lmodify_bind hb
      obtain ⟨rfl, rfl, rfl⟩ := lpure_ok hb
      have er₂ : t₂.rounds = tE.rounds := k₂.2.trans hr₁
      have ek₂ : t₂.k = tE.k + 4 := by rw [k₂.1, hk₁]
      have er₄ : t₄.rounds = tE.rounds := by rw [k₄.2, hr₃, er₂]
      have ek₄ : t₄.k = tE.k + 6 := by rw [k₄.1, hk₃, ek₂]
      have er₅ : tE'.rounds = tE.rounds := hr₅.trans er₄
      have ek₅ : tE'.k = tE.k + 7 := by rw [hk₅, ek₄]
      rw [hi] at hl1
      obtain ⟨⟨rs, sM', qM'⟩, hfoldM⟩ := foldRoundsC_any (D := Dig) p 4 (sch.k0 + 4 + 7 * i) stM.s stM.q
        (consumed tE.rounds tE.k (tE.k + 4)) (by rw [consumed_length]; omega) (consumed_sizes hwfE hn₁)
      have hfold' : Plays fOf (Model.foldRoundsC execArith
          (levelOf (sch.levels.getD (sch.levels.size - 1) default)).lanes stM.s stM.q)
          (kMsgs p (sch.k0 + 4 + 7 * i) 4) (consumed tE.rounds tE.k (tE.k + 4)) (rs, sM', qM') := by
        rw [show (levelOf (sch.levels.getD (sch.levels.size - 1) default)).lanes = 4 from hl1]; exact hfoldM
      have hcw : (tE.rounds[tE.k + 4]!).coins.size =
          (levelOf (sch.levels.getD (sch.levels.size - 1) default)).queries := by
        rw [RoundsWF.get hwfE, show tE.k + 4 = t₂.k by omega, ← er₂, hnQ]; rfl
      have hcα : (tE.rounds[tE.k + 5]!).coins.size =
          Accounting.ceilLog2 (levelOf (sch.levels.getD (sch.levels.size - 1) default)).queries := by
        rw [RoundsWF.get hwfE, show tE.k + 5 = t₂.k + 1 by omega, ← er₂, hnQ']; exact hlvR.2.2.2.1
      have hcβ : (tE.rounds[tE.k + 6]!).coins.size = 1 := by
        rw [RoundsWF.get hwfE, show tE.k + 6 = t₄.k by omega, ← er₄, hn₅]
      have hmain := iterLast_model r (sch.m - 7) b ini sch.ext
        (levelOf (sch.levels.getD (sch.levels.size - 1) default)) stM hfold' (yrOf p sch.ext)
        (tE.rounds[tE.k + 4]!).coins (tE.rounds[tE.k + 5]!).coins (tE.rounds[tE.k + 6]!).coins hcw hcα hcβ
        hlvR.q_ne_one (hα1 (sch.levels.size - 1) (by omega))
      have hcs : consumed tE.rounds tE.k tE'.k = consumed tE.rounds tE.k (tE.k + 4) ++
          [(tE.rounds[tE.k + 4]!).coins, (tE.rounds[tE.k + 5]!).coins, (tE.rounds[tE.k + 6]!).coins] := by
        rw [ek₅, consumed_append tE.rounds (show tE.k ≤ tE.k + 4 by omega) (show tE.k + 4 ≤ tE.k + 7 by omega),
          show tE.k + 7 = tE.k + 4 + 3 by omega, consumed_add tE.rounds (tE.k + 4) 3]
        simp [List.range_succ, Nat.add_assoc]
      have hdrop : (sch.levels.toList.map levelOf).drop (i + 1) =
          [levelOf (sch.levels.getD (sch.levels.size - 1) default)] := by
        rw [hi, drop_levels sch (sch.levels.size - 1) (by omega),
          show sch.levels.size - 1 + 1 = sch.levels.size by omega]
        simp
      exact ⟨_, _, rfl, er₅, by omega, by
        rw [hdrop, show sch.levels.size - 1 - i = 1 by omega, hcs]
        simp only [recMsgs]
        exact hmain, by
        intro hrej
        have := congrArg PendL.reqs hrej
        simp [PendL.reject, qbOut] at this⟩
    obtain ⟨pend, hr, hk, hpl, hfinal⟩ := hLC
    refine ⟨pend, hr, hk, ?_, hfinal⟩
    simp only [Nat.zero_add, Nat.sub_zero, Nat.mul_zero, Nat.add_zero] at hpl
    rw [hd1] at hpl
    exact hpl
  obtain ⟨_, hrL, hkL, -⟩ := hL 0 (fun _ => 0) []
    ⟨fun _ => 0, fun _ => (0, 0, 0), 0, [], [], [], 0, [], [], ⟨0, 0, 0, 0⟩⟩
  refine ⟨by rw [hrL, er₁₇], by omega, fun {np} P c₀ r Tm b => ?_⟩
  have hcs : consumed t.rounds t.k t₁₈.k = (t.rounds[t.k]!).coins :: (t.rounds[t.k + 1]!).coins ::
      (consumed t.rounds (t.k + 2) (t.k + 2 + sch.k0) ++
        (consumed t.rounds (t.k + 2 + sch.k0) (t.k + 2 + sch.k0 + 2) ++
          ([(t.rounds[t.k + 2 + sch.k0 + 2]!).coins, (t.rounds[t.k + 2 + sch.k0 + 3]!).coins,
              (t.rounds[t.k + 2 + sch.k0 + 4]!).coins, (t.rounds[t.k + 2 + sch.k0 + 5]!).coins] ++
            ([(t.rounds[t.k + 2 + sch.k0 + 6]!).coins, (t.rounds[t.k + 2 + sch.k0 + 7]!).coins,
                (t.rounds[t.k + 2 + sch.k0 + 8]!).coins] ++
              consumed t₁₇.rounds t₁₇.k t₁₈.k)))) := by
    rw [er₁₇, ek₁₇, consumed_cons _ (show t.k < t₁₈.k by omega), consumed_cons _ (show t.k + 1 < t₁₈.k by omega),
      show t.k + 1 + 1 = t.k + 2 from rfl,
      consumed_append t.rounds (show t.k + 2 ≤ t.k + 2 + sch.k0 by omega)
        (show t.k + 2 + sch.k0 ≤ t₁₈.k by omega),
      consumed_append t.rounds (show t.k + 2 + sch.k0 ≤ t.k + 2 + sch.k0 + 2 by omega)
        (show t.k + 2 + sch.k0 + 2 ≤ t₁₈.k by omega),
      consumed_append t.rounds (show t.k + 2 + sch.k0 + 2 ≤ t.k + 2 + sch.k0 + 2 + 4 by omega)
        (show t.k + 2 + sch.k0 + 2 + 4 ≤ t₁₈.k by omega),
      consumed_append t.rounds (show t.k + 2 + sch.k0 + 2 + 4 ≤ t.k + 2 + sch.k0 + 2 + 4 + 3 by omega)
        (show t.k + 2 + sch.k0 + 2 + 4 + 3 ≤ t₁₈.k by omega),
      consumed_add t.rounds (t.k + 2 + sch.k0 + 2) 4, consumed_add t.rounds (t.k + 2 + sch.k0 + 2 + 4) 3,
      show t.k + 2 + sch.k0 + 2 + 4 + 3 = t.k + 2 + sch.k0 + 9 by omega]
    simp [List.range_succ, Nat.add_assoc]
  have hlv01 : (schedOf sch).levels = levelOf (sch.levels.getD 0 default) ::
      levelOf (sch.levels.getD 1 default) :: (sch.levels.toList.map levelOf).drop 2 := by
    have e0 := drop_levels sch 0 (by omega)
    simp only [List.drop_zero, Nat.zero_add] at e0
    rw [hd1] at e0
    exact e0
  have hz₀ : (t.rounds[t.k]!).coins.size = (schedOf sch).m - 7 := by
    rw [RoundsWF.get hwf, show t.k = t₃'.k from ek₃.symm, ← er₃, hn₅, hn7]; rfl
  have hβ₀ : (t.rounds[t.k + 1]!).coins.size = 1 := by
    rw [RoundsWF.get hwf, show t.k + 1 = t₇.k from ek₇.symm, ← er₇, hn₈]
  have hFl : (consumed t.rounds (t.k + 2) (t.k + 2 + sch.k0)).length = sch.k0 := by
    rw [consumed_length]; omega
  have hFs : ∀ c ∈ consumed t.rounds (t.k + 2) (t.k + 2 + sch.k0), c.size = 2 :=
    consumed_sizes hwf fun j hj => by
      rw [show t.k + 2 + j = t₉.k + j by omega, ← er₉]; exact hn₁₀ j (by omega)
  have hρl : (consumed t.rounds (t.k + 2 + sch.k0) (t.k + 2 + sch.k0 + 2)).length = 2 := by
    rw [consumed_length]; omega
  have hρs : ∀ c ∈ consumed t.rounds (t.k + 2 + sch.k0) (t.k + 2 + sch.k0 + 2), c.size = 2 :=
    consumed_sizes hwf fun j hj => by
      rw [show t.k + 2 + sch.k0 + j = t₉.k + (sch.k0 + j) by omega, ← er₉]; exact hn₁₀ (sch.k0 + j) (by omega)
  have hdim0 : sch.n - sch.k0 + 1 = (schedOf sch).m - 7 - (schedOf sch).k0 + 1 := by
    show sch.n - sch.k0 + 1 = sch.m - 7 - sch.k0 + 1
    rw [hn7]
  have hmain := ligeritoPadZC_any (D := Dig) P (schedOf sch) zl.tPad c₀ r Tm b hlv01
    (t.rounds[t.k]!).coins (t.rounds[t.k + 1]!).coins hz₀ (by show sch.m - 7 ≠ 1; omega) hβ₀
    (fOf p.oodValues[0]!) (gOf p.messages[0]!.1, gOf p.messages[0]!.2)
    (fun s q => foldRoundsZK_any P p sch.k0 0 1 s q _ hFl hFs)
    ![gOf zl.msgs.e0, gOf zl.msgs.e1] (drawKs_any 2 _ hρl hρs) (gOf zl.msgs.tPrime)
    (fun i => gOf zl.msgs.ybar[i.val]!) (gOf p.messages[sch.k0]!.1, gOf p.messages[sch.k0]!.2)
    (capDig p.recursiveCaps[0]!)
    (fun s q => twoOodC_any _ 0 s q
      (t.rounds[t.k + 2 + sch.k0 + 2]!).coins (t.rounds[t.k + 2 + sch.k0 + 3]!).coins
      (t.rounds[t.k + 2 + sch.k0 + 4]!).coins (t.rounds[t.k + 2 + sch.k0 + 5]!).coins
      (by rw [RoundsWF.get hwf, show t.k + 2 + sch.k0 + 2 = t₁₁.k by omega, ← er₁₁, hnA, hdim0])
      (by rw [RoundsWF.get hwf, show t.k + 2 + sch.k0 + 3 = t₁₁.k + 1 by omega, ← er₁₁, hnB])
      (by rw [RoundsWF.get hwf, show t.k + 2 + sch.k0 + 4 = t₁₂.k by omega, ← er₁₂, hnC, hdim0])
      (by rw [RoundsWF.get hwf, show t.k + 2 + sch.k0 + 5 = t₁₂.k + 1 by omega, ← er₁₂, hnD])
      (by show sch.m - 7 - sch.k0 + 1 ≠ 1; omega)
      p.oodValues[1]! p.oodValues[2]! p.messages[sch.k0 + 1]! p.messages[sch.k0 + 2]!)
    (fun col st => queryAndBatchPadC_any r (levelOf (sch.levels.getD 0 default)) col 0 st
      (t.rounds[t.k + 2 + sch.k0 + 6]!).coins (t.rounds[t.k + 2 + sch.k0 + 7]!).coins
      (t.rounds[t.k + 2 + sch.k0 + 8]!).coins
      (by rw [RoundsWF.get hwf, show t.k + 2 + sch.k0 + 6 = t₁₃.k by omega, ← er₁₃, hnQ]; rfl)
      (by rw [RoundsWF.get hwf, show t.k + 2 + sch.k0 + 7 = t₁₃.k + 1 by omega, ← er₁₃, hnQ'];
          exact hlv0.2.2.2.1)
      (by rw [RoundsWF.get hwf, show t.k + 2 + sch.k0 + 8 = t₁₆.k by omega, ← er₁₆, hn₁₇])
      hlv0.q_ne_one (hα1 0 (by omega)) p.messages[sch.k0 + 3]!)
    (fun ini st => let ⟨pend, _, _, hpl, hne⟩ := hL r b ini st; ⟨pend, hpl, hne⟩)
  rw [hcs]
  simp only [ligMsgsZ, List.append_assoc, List.cons_append, List.nil_append] at hmain ⊢
  exact hmain

end FlockSoundness.Discharge.ZkExec
