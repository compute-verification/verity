import Proofs.Flock.Soundness.Registered.ReadsX
import Proofs.Flock.Recursive.StageRead

/-!
# Recursion: two sessions' registrant records, priced by a finder of the recursive audit

Two of V*'s sessions at one history whose registrant records (`ZkOuter.rd`) read one registered root (or one
registration) at one position open it there, so their leaves are one or SHA-512 collides
(`registrants_eq_or_collides`). The colliding pair is fixed by the history, across two sessions, so no finder behind a
session's `boundIf` (each sees one session's record) outputs it. A finder of the whole recursive audit does: it plays
the audit once and outputs the pair where the outcome's history has one.

* `ReadsX.Clash`, `ReadsX.Clash.collision`: two records' registered reads whose openings (`TwoOpenings.pair`) or hm96
  leaves over their commit strings' encodings (`hm96Pair`) collide, a collision of `H512`;
* `RegColl`, `regFinder`, `prob_regColl_le`: some two of V*'s sessions at a history clash; the finder that plays the
  recursive audit once and outputs that collision; under `cr/sha-512` (`Finder.CR` at `qX`), the audit's outcome has
  such a history with probability at most `qX²/2^513`, once per audit;
* `VBridgeOff`, `VBridge.off`: `VBridge` at the histories and registrations where `Bad` fails; `VBridge` gives it;
* `zk_sessions_recursive_upto_off`, `zk_sessions_recursive_read_off`: `zk_sessions_recursive_upto` and
  `zk_sessions_recursive_read` on `VBridgeOff`, with `Bad`'s probability (for `RegColl`, `qX²/2^513`) added to the
  outer term;
* `zk_sessions_recursive_fork_read_off`: `zk_sessions_recursive_fork_read` on `VBridgeOff RegColl`, with `regFinder`
  finding a collision as one more fork.
-/

namespace FlockSoundness.Registered

open Game Refine Binding Audit
open scoped ENNReal

section Clash

variable {C : Circuit} {n : ℕ} {P : Audit.Partition C n} {Reg : Type} {mPts : ℕ}
  {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
  {decode : Audit.Partition.DecoderB (C := C) (Reg := Reg) plan tab}
  {lay : Binding.Layout P plan tab decode} {rs : HmRowsX lay}
  {C' : Circuit} {n' : ℕ} {P' : Audit.Partition C' n'} {Reg₁ : Type} {mPts' : ℕ}
  {plan' : Finset (Fin n') → Reg₁ → List (TabSpec mPts')}
  {tab' : (S : Finset (Fin n')) → (R : Reg₁) → Fin n' → Fin (plan' S R).length}
  {decode' : Audit.Partition.DecoderB (C := C') (Reg := Reg₁) plan' tab'}
  {lay' : Binding.Layout P' plan' tab' decode'} {rs' : HmRowsX lay'}

/-- **Two registrant records clash**: at registered reads `p` of `rd` and `p'` of `rd'`, the openings
(`TwoOpenings.pair`, under `p`'s root and position) or the hm96 leaves over the two commit strings' encodings
(`hm96Pair` of `rs.enc p` and `rs'.enc p'`) are a SHA-512 collision. It asks nothing of the two reads' roots or
positions, so it covers a collision however an obligation keys them (one `Registered.Port`, or one registration). -/
def ReadsX.Clash (rd : ReadsX lay rs) (rd' : ReadsX lay' rs') : Prop :=
  ∃ p p', rd.isRead p ∧ rd'.isRead p' ∧
    (Collides (⟨rd.port p, rd.readAt p, rd.leaf p, rd.wpath p, rd'.leaf p', rd'.wpath p'⟩ : TwoOpenings).pair ∨
      Collides (hm96Pair (rs.enc p (rd.wv p)) (rd.ws p) (rs'.enc p' (rd'.wv p')) (rd'.ws p')))

theorem ReadsX.Clash.collision {rd : ReadsX lay rs} {rd' : ReadsX lay' rs'} (h : rd.Clash rd') :
    Merkle.Collision H512 := by
  obtain ⟨_, _, _, _, ⟨h1, h2⟩ | ⟨h1, h2⟩⟩ := h
  · exact ⟨_, _, h1, h2⟩
  · exact ⟨_, _, h1, h2⟩

end Clash

end FlockSoundness.Registered

namespace FlockSoundness.Discharge.Recursion

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Integrate Placed StrictCR
  DecodeZL Composed ZkReg
open scoped ENNReal

variable {S : ℕ} {Vs : Fin S → VStmt} {Reg' Reg₂ : Type}
variable {n : ℕ} {L : Law n} {Reg : Type} {I : Inner} {Cm : Type}

/-- **Two of V*'s sessions clash at a history**: at the history `(R, ω, t)` and registration `R₂`, the registrant
records of some sessions `j`, `j'` clash (`ReadsX.Clash`). -/
def RegColl (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg') (R : Reg) (ω : L.Ω)
    (t : List (Cm × I.Coin)) (R₂ : Reg₂) : Prop :=
  ∃ j j', (zs R ω t R₂ j).rd.Clash (zs R ω t R₂ j').rd

theorem RegColl.collision {zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'} {R : Reg}
    {ω : L.Ω} {t : List (Cm × I.Coin)} {R₂ : Reg₂} (h : RegColl zs R ω t R₂) : Merkle.Collision H512 := by
  obtain ⟨_, _, h⟩ := h
  exact h.collision

/-- **The registrant records' finder**: one run of the recursive audit against the prover `σ`, output the collision of
two clashing sessions at the outcome's history and registration (`RegColl`), else nothing. Its SHA-512 evaluations are
the audit's and the two openings' recomputation. -/
noncomputable def regFinder (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N : ℕ) (σ : Strategy (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N)) :
    Finder Pair :=
  oneRun _ σ (onePick H512 (fun o : Reg × L.Ω × List (Cm × I.Coin) × (Reg₂ × VRec Vs Reg' dg N) =>
    RegColl zs o.1 o.2.1 o.2.2.1 o.2.2.2.1) fun _ h => RegColl.collision h)

/-- **Two sessions clash at most `qX²/2^513` likely** under `cr/sha-512` for `regFinder` at the budget `qX`: once per
audit, whatever the number of sessions. -/
theorem prob_regColl_le (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N : ℕ) (σ : Strategy (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N))
    {qX : ℝ} (hX : (regFinder zs dg N σ).CR H512 qX) :
    prob (fun o => RegColl zs o.1 o.2.1 o.2.2.1 o.2.2.2.1) (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N) σ ≤
      ENNReal.ofReal (qX ^ 2 / 2 ^ (513 : ℝ)) := by
  have hev : (prob (fun o => RegColl zs o.1 o.2.1 o.2.2.1 o.2.2.2.1)
      (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N) σ).toReal ≤ qX ^ 2 / 2 ^ (513 : ℝ) := by
    rw [prob_toReal]
    exact (expect_one_le H512 _ σ (fun o : Reg × L.Ω × List (Cm × I.Coin) × (Reg₂ × VRec Vs Reg' dg N) =>
      RegColl zs o.1 o.2.1 o.2.2.1 o.2.2.2.1) fun _ h => RegColl.collision h).trans hX.le
  exact (ENNReal.le_ofReal_iff_toReal_le (ne_top_of_le_ne_top ENNReal.one_ne_top (prob_le_one _ _ _))
    (by positivity)).2 hev

/-- **`VBridge` off `Bad`**: at every history `(R, ω, t)` and registration `R₂` where `Bad` fails, and strategies `τ` of
V*'s sessions there, when every session accepts and every layer is `Good`, the inner verifier accepts the messages
`wmsg` reads from the layers. -/
def VBridgeOff (Bad : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → Prop)
    (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (x : Reg → L.Ω → I.Stmt)
    (wmsg : ((j : Fin S) → (Vs j).Bits) → List I.Msg) : Prop :=
  ∀ R ω t R₂, ¬ Bad R ω t R₂ → ∀ (τ : (j : Fin S) → Strategy ((zs R ω t R₂ j).game dg N))
    (o : (j : Fin S) → Reg' × Finset (Fin (Vs j).nv) × Exec.Transcript),
    (∀ j, LiveAcceptsZKR (zs R ω t R₂ j).I (zs R ω t R₂ j).own (o j) ∧
      (zs R ω t R₂ j).Good ((zs R ω t R₂ j).layer dg N k Rw (τ j))) →
    I.accepts (x R ω) ((wmsg fun j => (zs R ω t R₂ j).layer dg N k Rw (τ j)).zip (t.map Prod.snd))

theorem VBridge.off {Bad : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → Prop}
    {zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'} {dg : ByteArray → ByteArray}
    {N k Rw : ℕ} {x : Reg → L.Ω → I.Stmt} {wmsg : ((j : Fin S) → (Vs j).Bits) → List I.Msg}
    (h : Recursion.VBridge zs dg N k Rw x wmsg) : VBridgeOff Bad zs dg N k Rw x wmsg :=
  fun R ω t R₂ _ => h R ω t R₂

/-- `outerV_le` with `Q` reading the registration the prover sends. -/
theorem outerV_le_dep
    (hA3 : ∀ j, FlockSoundness.Assumptions.UniformRandomBytes (Vs j).law.Lb (Vs j).law.src)
    (zs : Reg₂ → (j : Fin S) → (Vs j).Outer Reg') (dg : ByteArray → ByteArray) (N : ℕ)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ R₂ j S' R' i, rateZ (ZkHidden.planZHJ (Reg := Reg') (zs R₂ j).dj (zs R₂ j).y₀ S' R') i k ≤ ρ)
    {t' : ℝ} (ht' : 0 ≤ t') (qF qS qW qR : ℝ) (Q : Reg₂ → ((j : Fin S) → (Vs j).Bits) → Prop)
    (hQ : ∀ R₂ (τ : (j : Fin S) → Strategy ((zs R₂ j).game dg N))
      (o : (j : Fin S) → Reg' × Finset (Fin (Vs j).nv) × Exec.Transcript),
      (∀ j, LiveAcceptsZKR (zs R₂ j).I (zs R₂ j).own (o j) ∧ (zs R₂ j).Good ((zs R₂ j).layer dg N k Rw (τ j))) →
      Q R₂ fun j => (zs R₂ j).layer dg N k Rw (τ j))
    (s : Strategy (outerV zs dg N)) :
    prob (fun o => AcceptsV zs o ∧ ¬ Q o.1 (layers (zs o.1) dg N k Rw o.2)) (outerV zs dg N) s ≤
      sumV zs dg N (fun R₂ j τ => (zs R₂ j).boundIf dg N (Vs j).law.model k Rw M ρ t' qF qS qW qR τ) s := by
  rw [prob_outerV]
  exact sessions_le hA3 (zs s.1) dg N hk hRw hM hρ (hr s.1) ht' qF qS qW qR (Q s.1) (hQ s.1) (sessOf zs dg N s)

theorem expAt_add {β : Type} {outer : Reg → L.Ω → List (Cm × I.Coin) → Game β}
    (σ : Strategy (recGame L Reg I Cm outer))
    (F G : (ω : L.Ω) → (t : List (Cm × I.Coin)) → Strategy (outer σ.1 ω t) → ℝ≥0∞) :
    expAt σ (fun ω t s => F ω t s + G ω t s) = expAt σ F + expAt σ G := by
  unfold expAt
  rw [← avg_add]
  exact congrArg avg (funext fun ω => leafAvg_add _ _ (σ.2 ω))

/-- **`zk_sessions_recursive_upto` off `Bad`**: on `VBridgeOff Bad`, the outer term gains `εX`, a bound on `Bad` at the
audit's outcome. An accepted outer proof whose inner transcript the inner verifier rejects has `Bad` at its history and
registration, or fails `Bad ∨ accepts`, which `outerV_le_dep` bounds by each session's `boundIf`. -/
theorem zk_sessions_recursive_upto_off
    (hA3 : ∀ j, FlockSoundness.Assumptions.UniformRandomBytes (Vs j).law.Lb (Vs j).law.src)
    (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N : ℕ)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ R ω t R₂ j S' R' i,
      rateZ (ZkHidden.planZHJ (Reg := Reg') (zs R ω t R₂ j).dj (zs R ω t R₂ j).y₀ S' R') i k ≤ ρ)
    {t' : ℝ} (ht' : 0 ≤ t') (qF qS qW qR : ℝ)
    (σ : Strategy (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N))
    (x : Reg → L.Ω → I.Stmt) (wmsg : ((j : Fin S) → (Vs j).Bits) → List I.Msg)
    (Bad : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → Prop) (hbr : VBridgeOff Bad zs dg N k Rw x wmsg)
    {εX : ℝ≥0∞}
    (hX : prob (fun o => Bad o.1 o.2.1 o.2.2.1 o.2.2.2.1) (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N) σ ≤
      εX)
    (same : ℕ → List I.Coin → I.Msg → I.Msg → Prop)
    (hsame : ∀ y (ms ms' : List I.Msg) (ρs : List I.Coin), (∀ i < ρs.length, Option.Rel (same i ρs) ms[i]? ms'[i]?) →
      I.accepts y (ms.zip ρs) → I.accepts y (ms'.zip ρs))
    {V Pos : Type} [Fintype V] [DecidableEq V] [Nonempty V] [Fintype Pos] (ob : Openings H512 V)
    (cmt : Cm → Pos → ob.Dg) (msg : ℕ → (Pos → V) → I.Msg)
    (T : ℕ → ∀ R ω t, Strategy (outerV (zs R ω t) dg N) → Pos → Finder (Finset ob.Op))
    (εtr : ℕ → ∀ R ω t, Strategy (outerV (zs R ω t) dg N) → ℝ≥0∞) (κ : ℕ → Pos → ℝ≥0∞) (q : ℕ → ℝ)
    (htr : ∀ i < I.r, AllAt σ fun ω t s => ∀ (v : Pos → V) (e : Cm × I.Coin), t[i]? = some e →
      prob (fun o => AcceptsV (zs σ.1 ω t) o ∧ ¬ Option.Rel (same i (t.map Prod.snd))
          (wmsg (layers (zs σ.1 ω t o.1) dg N k Rw o.2))[i]? (some (msg i v))) (outerV (zs σ.1 ω t) dg N) s ≤
        εtr i σ.1 ω t s +
          ∑ pos, κ i pos * prob (OffAt ob (cmt e.1 pos) (v pos)) (T i σ.1 ω t s pos).g (T i σ.1 ω t s pos).s)
    (hCR : ∀ i < I.r, ∀ ω pos, (forkAt ob σ cmt (T i) i ω pos).CR H512 (q i))
    {εin : ℝ≥0∞} (hin : InnerSound I εin) :
    prob (fun o => AcceptsV (zs o.1 o.2.1 o.2.2.1) o.2.2.2 ∧ ¬ I.holds (x o.1 o.2.1))
        (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N) σ ≤
      expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
          (fun R₂ j τ => (zs σ.1 ω t R₂ j).boundIf dg N (Vs j).law.model k Rw M ρ t' qF qS qW qR τ) s) + εX +
        ∑ i ∈ Finset.range I.r,
          (expAt σ (εtr i σ.1) + (∑ pos, κ i pos) * ENNReal.ofReal (Real.sqrt (q i ^ 2 / 2 ^ (513 : ℝ)))) + εin := by
  have h := recursive_sound_upto ob σ (fun R ω t o => AcceptsV (zs R ω t) o) x
    (W := ((j : Fin S) → (Vs j).Bits) × I.Stmt) (fun R ω t _ o => (layers (zs R ω t o.1) dg N k Rw o.2, x R ω))
    Prod.snd (fun w => wmsg w.1) same hsame cmt msg T εtr κ q
    (εo := expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
      (fun R₂ j τ => (zs σ.1 ω t R₂ j).boundIf dg N (Vs j).law.model k Rw M ρ t' qF qS qW qR τ) s) + εX) (εP := 0)
    ?_ ?_ htr hCR hin
  · rwa [add_zero] at h
  · have hsplit : ∀ ω t (s : Strategy (outerV (zs σ.1 ω t) dg N)),
        prob (fun b => AcceptsV (zs σ.1 ω t) b ∧
            ¬ I.accepts (x σ.1 ω) ((wmsg (layers (zs σ.1 ω t b.1) dg N k Rw b.2)).zip (t.map Prod.snd)))
          (outerV (zs σ.1 ω t) dg N) s ≤
        sumV (zs σ.1 ω t) dg N
            (fun R₂ j τ => (zs σ.1 ω t R₂ j).boundIf dg N (Vs j).law.model k Rw M ρ t' qF qS qW qR τ) s +
          prob (fun b => Bad σ.1 ω t b.1) (outerV (zs σ.1 ω t) dg N) s := by
      intro ω t s
      refine (prob_mono (q := fun b : Reg₂ × VRec Vs Reg' dg N => (AcceptsV (zs σ.1 ω t) b ∧
          ¬ (Bad σ.1 ω t b.1 ∨
            I.accepts (x σ.1 ω) ((wmsg (layers (zs σ.1 ω t b.1) dg N k Rw b.2)).zip (t.map Prod.snd)))) ∨
          Bad σ.1 ω t b.1) (fun b hb => ?_) _ _).trans ((prob_or_le _ _ _ _).trans (add_le_add ?_ le_rfl))
      · obtain ⟨ha, hn⟩ := hb
        by_cases hb : Bad σ.1 ω t b.1
        · exact Or.inr hb
        · exact Or.inl ⟨ha, fun h => h.elim hb hn⟩
      · exact outerV_le_dep hA3 (zs σ.1 ω t) dg N hk hRw hM hρ (hr σ.1 ω t) ht' qF qS qW qR
          (fun R₂ Xs => Bad σ.1 ω t R₂ ∨ I.accepts (x σ.1 ω) ((wmsg Xs).zip (t.map Prod.snd)))
          (fun R₂ τ o h => (Classical.em (Bad σ.1 ω t R₂)).elim Or.inl fun hb => Or.inr (hbr σ.1 ω t R₂ hb τ o h)) s
    refine le_trans (b := expAt σ fun ω t s => sumV (zs σ.1 ω t) dg N
        (fun R₂ j τ => (zs σ.1 ω t R₂ j).boundIf dg N (Vs j).law.model k Rw M ρ t' qF qS qW qR τ) s +
          prob (fun b => Bad σ.1 ω t b.1) (outerV (zs σ.1 ω t) dg N) s)
      (avg_mono fun ω => leafAvg_mono (fun t s => hsplit ω t _) _) ?_
    exact (expAt_add σ _ _).trans_le (add_le_add le_rfl ((prob_recGame _ σ).symm.le.trans hX))
  · refine le_of_eq ((congrArg avg (funext fun ω => ?_)).trans (avg_const 0))
    refine (leafAvg_congr_len (G := fun _ _ => 0) (fun t s _ => ?_) _).trans (leafAvg_const 0 _)
    exact le_antisymm ((prob_mono (fun b hb => hb.2 rfl) _ _).trans (prob_false _ _).le) zero_le

/-- **The recursive audit is sound at the read wires, off two sessions' clash**: `zk_sessions_recursive_read` on
`VBridgeOff RegColl`, its outer term gaining `qX²/2^513` for `regFinder` under `cr/sha-512` (`prob_regColl_le`). -/
theorem zk_sessions_recursive_read_off
    (hA3 : ∀ j, FlockSoundness.Assumptions.UniformRandomBytes (Vs j).law.Lb (Vs j).law.src)
    (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N : ℕ)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ R ω t R₂ j S' R' i,
      rateZ (ZkHidden.planZHJ (Reg := Reg') (zs R ω t R₂ j).dj (zs R ω t R₂ j).y₀ S' R') i k ≤ ρ)
    {t' : ℝ} (ht' : 0 ≤ t') (qF qS qW qR : ℝ)
    (σ : Strategy (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N))
    (x : Reg → L.Ω → I.Stmt) (wmsg : ((j : Fin S) → (Vs j).Bits) → List I.Msg)
    (hbr : VBridgeOff (RegColl zs) zs dg N k Rw x wmsg)
    (qX : ℝ) (hX : (regFinder zs dg N σ).CR H512 qX)
    (same : ℕ → List I.Coin → I.Msg → I.Msg → Prop)
    (hsame : ∀ y (ms ms' : List I.Msg) (ρs : List I.Coin), (∀ i < ρs.length, Option.Rel (same i ρs) ms[i]? ms'[i]?) →
      I.accepts y (ms.zip ρs) → I.accepts y (ms'.zip ρs))
    (Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (cmt : (j : Fin S) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg)
    (msg : ℕ → ((j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) → I.Msg)
    (hcmt : VCommits zs Sk cmt) (hdec : VDecodesR wmsg same Sk msg) (hfull : VFullR I.r Sk)
    (q : ℕ → ℝ)
    (hCR : ∀ i < I.r, ∀ ω pos,
      (forkAt (obS Vs) σ (cmtS cmt) (fun R ω t s => trialV (zs R ω t) dg N k Rw s) i ω pos).CR H512 (q i))
    {εin : ℝ≥0∞} (hin : InnerSound I εin) :
    prob (fun o => AcceptsV (zs o.1 o.2.1 o.2.2.1) o.2.2.2 ∧ ¬ I.holds (x o.1 o.2.1))
        (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N) σ ≤
      expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
          (fun R₂ j τ => (zs σ.1 ω t R₂ j).boundIf dg N (Vs j).law.model k Rw M ρ t' qF qS qW qR τ) s) +
        ENNReal.ofReal (qX ^ 2 / 2 ^ (513 : ℝ)) +
        ∑ i ∈ Finset.range I.r,
          (expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
              (fun R₂ j τ => (zs σ.1 ω t R₂ j).slackIf dg N k Rw M ρ t' qF qS τ) s) +
            (∑ pos, weightS Vs Sk ρ i pos) * ENNReal.ofReal (Real.sqrt (q i ^ 2 / 2 ^ (513 : ℝ)))) + εin := by
  refine zk_sessions_recursive_upto_off hA3 zs dg N hk hRw hM hρ hr ht' qF qS qW qR σ x wmsg (RegColl zs) hbr
    (prob_regColl_le zs dg N σ hX) same hsame
    (obS Vs) (cmtS cmt) (msgS msg) (fun _ R ω t s => trialV (zs R ω t) dg N k Rw s)
    (fun _ R ω t s => sumV (zs R ω t) dg N (fun R₂ j τ => (zs R ω t R₂ j).slackIf dg N k Rw M ρ t' qF qS τ) s)
    (weightS Vs Sk ρ) q (fun i hi ω => AllLeaves.of_forall (fun t s' v e he => ?_) _) hCR hin
  refine (outerV_stage_le_read (zs σ.1 ω t) dg N hk hRw hM hρ (hr σ.1 ω t) ht' qF qS (Sk i) (hfull i hi)
    (fun j pos => cmt j e.1 pos) (fun R₂ j hj pos hp R' => hcmt i σ.1 ω t e he R₂ j hj pos hp R')
    (fun Xs => Option.Rel (same i (t.map Prod.snd)) (wmsg Xs)[i]? (some (msgS msg i v)))
    (fun j p => projV j (v ⟨j, p⟩)) (fun Xt hXt => hdec i hi _ Xt (fun j p => projV j (v ⟨j, p⟩)) hXt) _).trans
    (add_le_add le_rfl ?_)
  rw [Fintype.sum_sigma]
  exact Finset.sum_le_sum fun j _ => Finset.sum_le_sum fun p _ =>
    mul_le_mul' le_rfl (trialS_le (zs σ.1 ω t _) dg N k Rw _ j p _ _)

/-- **The recursive audit is sound at the read wires off two sessions' clash, in fork form**:
`zk_sessions_recursive_fork_read` on `VBridgeOff RegColl`, from `zk_sessions_recursive_read_off` at no finder budgets,
with one more disjunct: the registrant records' finder (`regFinder`, one run of the prover's own strategy) outputs a
SHA-512 collision. -/
theorem zk_sessions_recursive_fork_read_off
    (hA3 : ∀ j, FlockSoundness.Assumptions.UniformRandomBytes (Vs j).law.Lb (Vs j).law.src)
    (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N : ℕ)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ R ω t R₂ j S' R' i,
      rateZ (ZkHidden.planZHJ (Reg := Reg') (zs R ω t R₂ j).dj (zs R ω t R₂ j).y₀ S' R') i k ≤ ρ)
    (σ : Strategy (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N))
    (x : Reg → L.Ω → I.Stmt) (wmsg : ((j : Fin S) → (Vs j).Bits) → List I.Msg)
    (hbr : VBridgeOff (RegColl zs) zs dg N k Rw x wmsg)
    (same : ℕ → List I.Coin → I.Msg → I.Msg → Prop)
    (hsame : ∀ y (ms ms' : List I.Msg) (ρs : List I.Coin), (∀ i < ρs.length, Option.Rel (same i ρs) ms[i]? ms'[i]?) →
      I.accepts y (ms.zip ρs) → I.accepts y (ms'.zip ρs))
    (Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (cmt : (j : Fin S) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg)
    (msg : ℕ → ((j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) → I.Msg)
    (hcmt : VCommits zs Sk cmt) (hdec : VDecodesR wmsg same Sk msg) (hfull : VFullR I.r Sk)
    {εin : ℝ≥0∞} (hin : InnerSound I εin) :
    (regFinder zs dg N σ).Finds H512 ∨ RoundFork zs dg N k Rw σ cmt ∨ SessionFork zs dg N k Rw M σ ∨
      prob (fun o => AcceptsV (zs o.1 o.2.1 o.2.2.1) o.2.2.2 ∧ ¬ I.holds (x o.1 o.2.1))
          (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N) σ ≤
        expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
            (fun R₂ j τ => (zs σ.1 ω t R₂ j).boundCR dg N (Vs j).law.model k Rw M ρ τ) s) +
          (I.r : ℝ≥0∞) * expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
            (fun R₂ j τ => (zs σ.1 ω t R₂ j).slackCR dg N k Rw M ρ τ) s) + εin := by
  by_cases h0 : (regFinder zs dg N σ).Finds H512
  · exact Or.inl h0
  by_cases h1 : RoundFork zs dg N k Rw σ cmt
  · exact Or.inr (Or.inl h1)
  by_cases h2 : SessionFork zs dg N k Rw M σ
  · exact Or.inr (Or.inr (Or.inl h2))
  refine Or.inr (Or.inr (Or.inr ?_))
  have hno := allAt_of_not_exAt h2
  have hB : expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
        (fun R₂ j τ => (zs σ.1 ω t R₂ j).boundIf dg N (Vs j).law.model k Rw M ρ 0 0 0 0 0 τ) s) ≤
      expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
        (fun R₂ j τ => (zs σ.1 ω t R₂ j).boundCR dg N (Vs j).law.model k Rw M ρ τ) s) := by
    refine expAt_mono_all (hno.mono fun ω t s hs => ?_)
    unfold sumV sumS
    exact exG_mono_outcome _ _ fun rs hrs => Finset.sum_le_sum fun j _ =>
      ZkOuter.boundIf_le _ _ _ _ _ fun hf => hs ⟨rs, hrs, j, hf⟩
  have hsl : expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
        (fun R₂ j τ => (zs σ.1 ω t R₂ j).slackIf dg N k Rw M ρ 0 0 0 τ) s) ≤
      expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
        (fun R₂ j τ => (zs σ.1 ω t R₂ j).slackCR dg N k Rw M ρ τ) s) := by
    refine expAt_mono_all (hno.mono fun ω t s hs => ?_)
    unfold sumV sumS
    exact exG_mono_outcome _ _ fun rs hrs => Finset.sum_le_sum fun j _ =>
      ZkOuter.slackIf_le _ _ _ _ fun hf => hs ⟨rs, hrs, j, hf⟩
  refine (zk_sessions_recursive_read_off hA3 zs dg N hk hRw hM hρ hr le_rfl 0 0 0 0 σ x wmsg hbr 0
    (Finder.cr_of_not_finds h0 0) same hsame Sk cmt msg hcmt hdec hfull (fun _ => 0)
    (fun i hi ω pos => Finder.cr_of_not_finds (fun hf => h1 ⟨i, hi, ω, pos, hf⟩) 0) hin).trans
    (add_le_add (add_le_add ?_ ?_) le_rfl)
  · simp only [ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true, zero_pow, zero_div, ENNReal.ofReal_zero, add_zero]
    exact hB
  · simp only [ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true, zero_pow, zero_div, Real.sqrt_zero, ENNReal.ofReal_zero,
      mul_zero, add_zero]
    rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
    exact mul_le_mul' le_rfl hsl

end FlockSoundness.Discharge.Recursion
