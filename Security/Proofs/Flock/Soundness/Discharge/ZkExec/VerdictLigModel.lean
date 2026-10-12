import Proofs.Flock.Soundness.Discharge.ZkExec.PlaysLig

/-!
# The model's masked level 0, with its values

`PlaysLig.lean`'s runs exist at any values; the verdict needs them at the executable's. The lane rounds end in the
lane coins (`foldRoundsZK_val`), `ρ` is its two coins (`drawKs_two`), level 0's padded queries end in `qbPadOut`
(`queryAndBatchPadC_plays`), and `ligeritoPadZC_model` composes the masked level 0 from runs of its parts, as the non-ZK
session's `ligeritoC_model`.
-/

namespace FlockSoundness.Discharge.ZkExec

open Flock Refine Exec Game Model Transcript

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

variable {D : Type}

/-- **The masked lane rounds, with their coins**: `foldRoundsZK` on `k` rounds of two coins each ends in those coins
as `K` elements. -/
theorem foldRoundsZK_val {np : ℕ} (P : ZkSession.Pads0 np 2) (p : LigeritoProof) :
    ∀ (k j a : ℕ) (s : ZkSession.LinK 𝕂 np) (q : ZkSession.QuadL 𝕂 np) (cs : List (Array F128)),
      cs.length = k → (∀ c ∈ cs, c.size = 2) →
      ∃ s', Plays fOf (ZkSession.foldRoundsZK execArith P k j s q) (kMsgs p a (k - 1)) cs
        (cs.map fun c => execArith.ofLimbs (fOf c[0]!) (fOf c[1]!), s')
  | 0, j, a, s, q, cs, hl, _ => by
      rw [List.eq_nil_of_length_eq_zero hl, ZkSession.foldRoundsZK, show kMsgs p a (0 - 1) = [] by simp [kMsgs]]
      exact ⟨s, Plays.pure _⟩
  | 1, j, a, s, q, c :: cs, hl, hs => by
      rw [List.eq_nil_of_length_eq_zero (by simpa using hl : cs.length = 0), ZkSession.foldRoundsZK,
        show kMsgs p a (1 - 1) = [] by simp [kMsgs]]
      have hk : Plays fOf (Model.drawK execArith) [] [c] (execArith.ofLimbs (fOf c[0]!) (fOf c[1]!)) := by
        unfold Model.drawK
        exact Plays.bind' (ms₂ := []) (cs₂ := []) (Plays.drawVec c (hs c (by simp)) (by omega)) (Plays.pure _)
          (by simp) (by simp)
      exact ⟨_, Plays.bind' hk (Plays.pure _) rfl rfl⟩
  | k + 2, j, a, s, q, c :: cs, hl, hs => by
      rw [ZkSession.foldRoundsZK, show kMsgs p a (k + 2 - 1) = kMsg p.messages[a]! :: kMsgs p (a + 1) (k + 1 - 1) by
        simp [kMsgs, List.range'_succ]]
      have hk : Plays fOf (Model.drawK execArith) [] [c] (execArith.ofLimbs (fOf c[0]!) (fOf c[1]!)) := by
        unfold Model.drawK
        exact Plays.bind' (ms₂ := []) (cs₂ := []) (Plays.drawVec c (hs c (by simp)) (by omega)) (Plays.pure _)
          (by simp) (by simp)
      obtain ⟨s', ih⟩ := foldRoundsZK_val P p (k + 1) (j + 1) (a + 1)
        ((q.eval (execArith.ofLimbs (fOf c[0]!) (fOf c[1]!))))
        (ZkSession.fromMsgL (ZkSession.unmask execArith P (j + 1) (gOf p.messages[a]!.1, gOf p.messages[a]!.2))
          (q.eval (execArith.ofLimbs (fOf c[0]!) (fOf c[1]!)))) cs (by simpa using hl)
        fun c' h => hs c' (by simp [h])
      refine ⟨s', Plays.bind' hk ?_ rfl rfl⟩
      refine Plays.bind' (Plays.recv _ (gOf p.messages[a]!.1, gOf p.messages[a]!.2)) ?_ rfl rfl
      exact Plays.bind' ih (Plays.pure _) (List.append_nil _).symm (List.append_nil _).symm

/-- **`ρ`, with its coins**: two `K` coins. -/
theorem drawKs_two (c₀ c₁ : Array F128) (h₀ : c₀.size = 2) (h₁ : c₁.size = 2) :
    Plays fOf (Model.drawKs execArith 2) [] [c₀, c₁]
      (Fin.cons (execArith.ofLimbs (fOf c₀[0]!) (fOf c₀[1]!))
        (Fin.cons (execArith.ofLimbs (fOf c₁[0]!) (fOf c₁[1]!)) Fin.elim0)) := by
  have hk : ∀ c : Array F128, c.size = 2 →
      Plays fOf (Model.drawK execArith) [] [c] (execArith.ofLimbs (fOf c[0]!) (fOf c[1]!)) := fun c hc => by
    unfold Model.drawK
    exact Plays.bind' (ms₂ := []) (cs₂ := []) (Plays.drawVec c hc (by omega)) (Plays.pure _) (by simp) (by simp)
  rw [Model.drawKs, Model.drawKs, Model.drawKs]
  refine Plays.bind' (hk c₀ h₀) ?_ rfl rfl
  refine Plays.bind' (Plays.bind' (hk c₁ h₁) (Plays.bind' (Plays.pure _) (Plays.pure _) rfl rfl) rfl rfl)
    (Plays.pure _) rfl rfl

/-- The model's state after `queryAndBatchPadC` on the coins `cw`, `cα`, `cβ` and the message `m`. -/
noncomputable def qbPadOut (r : ℕ) (lv : Accounting.Level) (col : List 𝔽 → ℕ → 𝕂) (start : ℕ)
    (st : LigStateC 𝔽 𝕂 D) (cw cα cβ : Array F128) (m : F256 × F256) : LigStateC 𝔽 𝕂 D :=
  let pos := Model.positions execArith lv.queries lv.logLen (vecOf lv.queries cw)
  let e : Opens 𝔽 D → 𝕂 := fun o => ((pos.zipIdx).map fun pj =>
    execArith.embed (eqList (List.ofFn (vecOf (Accounting.ceilLog2 lv.queries) cα)) pj.2) *
      col (o r st.level pj.2).1 pj.1).sum
  { st with
    s := fun o => st.s o + execArith.embed (fOf cβ[0]!) * e o
    q := fun o => (st.q o).addScaled (Model.fromMsg (gOf m.1, gOf m.2) (e o)) (execArith.embed (fOf cβ[0]!))
    cons := st.cons ++ [⟨lv.logCols, pos, List.ofFn (vecOf (Accounting.ceilLog2 lv.queries) cα), fOf cβ[0]!, start⟩]
    reqs := st.reqs ++ [⟨st.level, pos, lv.lanes, pathLen lv⟩] }

/-- **Level 0's padded queries**: the model's `queryAndBatchPadC`, fed the query coins, the consistency coins, the
message and the batching coin, ends in `qbPadOut`. -/
theorem queryAndBatchPadC_plays (r : ℕ) (lv : Accounting.Level) (col : List 𝔽 → ℕ → 𝕂) (start : ℕ)
    (st : LigStateC 𝔽 𝕂 D) (cw cα cβ : Array F128) (hcw : cw.size = lv.queries)
    (hcα : cα.size = Accounting.ceilLog2 lv.queries) (hcβ : cβ.size = 1) (hq1 : lv.queries ≠ 1)
    (hα1 : Accounting.ceilLog2 lv.queries ≠ 1) (m : F256 × F256) :
    Plays fOf (ZkLink.queryAndBatchPadC execArith r lv col start st) [kMsg m] [cw, cα, cβ]
      (qbPadOut r lv col start st cw cα cβ m) := by
  unfold ZkLink.queryAndBatchPadC
  refine Plays.bind' (ms₂ := [kMsg m]) (cs₂ := [cα, cβ]) (Plays.drawVec cw hcw hq1) ?_ (by simp) (by simp)
  refine Plays.bind' (ms₂ := [kMsg m]) (cs₂ := [cβ]) (Plays.drawVec cα hcα hα1) ?_ (by simp) (by simp)
  dsimp only
  refine Plays.bind' (ms₂ := []) (cs₂ := [cβ]) (Plays.recv _ (gOf m.1, gOf m.2)) ?_ (by simp [kMsg]) (by simp)
  refine Plays.bind' (ms₂ := []) (cs₂ := []) (Plays.drawOne cβ hcβ) ?_ (by simp) (by simp)
  exact Plays.pure _

/-- **The model's masked level 0, composed** (`ligeritoC_model` for `ligeritoPadZC`): the level-0 out-of-domain
sample, the first message and the masked lane rounds (`hfold`, ending in the lane coins `ini`), the pair's sums, `ρ`
(`hρ`), `T'` and `ybar`, the code-switch message, level 1's cap, its two out-of-domain samples from `T'` (`htwo`),
level 0's padded queries (`hqb`), and a run of the recursive rounds from the resulting state (`hrec`). -/
theorem ligeritoPadZC_model {np : ℕ} (P : ZkSession.Pads0 np 2) (sch : Accounting.Schedule) (tp : ℕ) (c₀ : 𝔽)
    (r : ℕ) (Tm : 𝔽) (b : ℕ → 𝔽) {lv₀ lv₁ : Accounting.Level} {rest : List Accounting.Level}
    (hl : sch.levels = lv₀ :: lv₁ :: rest) (z₀ cβ₀ : Array F128) (hz₀ : z₀.size = sch.m - 7) (hm7 : sch.m - 7 ≠ 1)
    (hβ₀ : cβ₀.size = 1) (y₀ : 𝔽) (m₀ : 𝕂 × 𝕂) {msF : List Msg} {csF : List (Array F128)} {ini : List 𝕂}
    (hfold : ∀ s q, ∃ s', Plays fOf (ZkSession.foldRoundsZK execArith P sch.k0 0 s q) msF csF (ini, s'))
    (ev : Fin 2 → 𝕂) {csρ : List (Array F128)} {ρ : Fin 2 → 𝕂} (hρ : Plays fOf (Model.drawKs execArith 2) [] csρ ρ)
    (T' : 𝕂) (ybar : Fin tp → 𝕂) (m' : F256 × F256) (C₁ : Cap D) {msO : List Msg} {csO : List (Array F128)}
    {sT : Opens 𝔽 D → 𝕂} {qT : Opens 𝔽 D → Model.Quad 𝕂} {ctxs : List (Model.OodCtx 𝔽)}
    (htwo : Plays fOf (Model.twoOodC (D := D) execArith (sch.m - 7 - sch.k0 + 1) 0 (fun _ => T')
      (fun _ => Model.fromMsg (gOf m'.1, gOf m'.2) T')) msO csO (sT, qT, ctxs))
    (cw cα cβ : Array F128) (hcw : cw.size = lv₀.queries) (hcα : cα.size = Accounting.ceilLog2 lv₀.queries)
    (hcβ : cβ.size = 1) (hq1 : lv₀.queries ≠ 1) (hα1 : Accounting.ceilLog2 lv₀.queries ≠ 1) (mq : F256 × F256)
    {msR : List Msg} {csR : List (Array F128)} {pend : PendL 𝔽 D}
    (hrec : Plays fOf (Model.recursiveRoundsC (D := D) execArith r (sch.m - 7) b ini sch.ext (lv₁ :: rest)
      { qbPadOut r lv₀ (ZkLink.padColumnO execArith lv₀.logLen lv₀.logCols tp c₀ ini ρ (extraLane sch.k0 2 r) ybar) 0
          ⟨sT, qT, sch.m - 7 - sch.k0 + 1, [], ⟨List.ofFn (vecOf (sch.m - 7) z₀), fOf cβ₀[0]!, none⟩ :: ctxs, [], 0,
            [], [], lv₀⟩ cw cα cβ mq with
        level := 1, caps := [C₁], prevLevel := lv₁ }) msR csR pend) :
    ∃ row, Plays fOf (ZkLink.ligeritoPadZC (D := D) execArith P sch tp c₀ r Tm b)
      (⟨𝔽, y₀⟩ :: ⟨𝕂 × 𝕂, m₀⟩ :: (msF ++ ⟨Fin 2 → 𝕂, ev⟩ :: ⟨𝕂, T'⟩ :: ⟨Fin tp → 𝕂, ybar⟩ ::
        kMsg m' :: ⟨Cap D, C₁⟩ :: (msO ++ ([kMsg mq] ++ msR))))
      (z₀ :: cβ₀ :: (csF ++ (csρ ++ (csO ++ ([cw, cα, cβ] ++ csR))))) (row, pend) := by
  unfold ZkLink.ligeritoPadZC
  rw [hl]
  dsimp only
  obtain ⟨s', hf⟩ := hfold
    (ZkSession.cK (execArith.embed Tm) + execArith.embed (fOf cβ₀[0]!) •
      (ZkSession.cK (execArith.embed y₀) + ZkSession.padF P.ood))
    (ZkSession.fromMsgL (ZkSession.unmask execArith P 0 m₀)
      (ZkSession.cK (execArith.embed Tm) + execArith.embed (fOf cβ₀[0]!) •
        (ZkSession.cK (execArith.embed y₀) + ZkSession.padF P.ood)))
  have hqb := queryAndBatchPadC_plays r lv₀
    (ZkLink.padColumnO execArith lv₀.logLen lv₀.logCols tp c₀ ini ρ (extraLane sch.k0 2 r) ybar) 0
    (⟨sT, qT, sch.m - 7 - sch.k0 + 1, [], ⟨List.ofFn (vecOf (sch.m - 7) z₀), fOf cβ₀[0]!, none⟩ :: ctxs, [], 0, [], [],
      lv₀⟩ : LigStateC 𝔽 𝕂 D) cw cα cβ hcw hcα hcβ hq1 hα1 mq
  refine ⟨s' + ∑ i, ρ i • (ZkSession.cK (ev i) + ZkSession.padK execArith (P.ev i 0) (P.ev i 1)) - ZkSession.cK T',
    Plays.bind' (Plays.drawVec z₀ hz₀ hm7) ?_ rfl rfl⟩
  refine Plays.bind' (Plays.recv _ y₀) ?_ rfl rfl
  refine Plays.bind' (Plays.drawOne cβ₀ hβ₀) ?_ rfl rfl
  refine Plays.bind' (Plays.recv _ m₀) ?_ rfl rfl
  refine Plays.bind' hf ?_ rfl rfl
  dsimp only
  refine Plays.bind' (Plays.recv _ ev) ?_ rfl rfl
  refine Plays.bind' hρ ?_ rfl rfl
  refine Plays.bind' (Plays.recv _ T') ?_ rfl rfl
  refine Plays.bind' (Plays.recv _ ybar) ?_ rfl rfl
  refine Plays.bind' (ms₂ := []) (cs₂ := []) ?_ (Plays.pure _) (List.append_nil _).symm (List.append_nil _).symm
  unfold ZkLink.ligTailC
  refine Plays.bind' (Plays.recv _ (gOf m'.1, gOf m'.2)) ?_ rfl rfl
  dsimp only
  refine Plays.bind' (Plays.recv _ C₁) ?_ rfl rfl
  refine Plays.bind' htwo ?_ rfl rfl
  dsimp only
  exact Plays.bind' hqb hrec rfl rfl

end FlockSoundness.Discharge.ZkExec
