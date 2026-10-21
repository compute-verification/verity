import Proofs.Flock.Claims.Complete.Replay.LigeritoHonest

/-!
# The honest Ligerito meets its final identity

`Lg.Honest` (`Lg.honest`), by carrying the honest prover's table and weight through its Ligerito against the model's
steps: a precondition some value's (`Lg.HRuns.exPre`) and a fact carried through a step (`Lg.HRuns.frame`); fast100's
levels (`Lg.sch_facts`); the model's final identity (`Lg.Final`) and `Lg.Mid` for the table and weight's own equation
(`Lg.MidR`). So at the deployed relation `LgFinal` rests on the session walk on the packed table alone
(`lgFinal_walkP`), and the skeleton's `Ligerito` and `Consumed` on the walk's facts and the schemes' widths
(`ligerito_walkP`, `consumed_walkP`).
-/

namespace FlockComplete.Replay

open Flock Transcript FlockSoundness FlockSoundness.Refine

local notation "𝕂" => FlockLevel3.GF256

namespace Lg

variable {p : LigeritoProof} {L : ℕ} {β : Type}

/-- A precondition that is some `g`'s, from a run for each `g`. -/
theorem HRuns.exPre {γ : Type} [Inhabited γ] {y : LP PS β} {c c' : Cur} {tm tm' : Transcript} {Q Xb : β → Prop}
    {J : γ → TW → Prop} {J' : β → TW → Prop} (h : ∀ g, HRuns p L y c tm c' tm' Q Xb (J g) J') :
    HRuns p L y c tm c' tm' Q Xb (fun x => ∃ g, J g x) J' :=
  ⟨(h default).1, fun st t l b st' t' l' hl e hE hx hc ht ⟨g, hg⟩ => (h g).2 st t l b st' t' l' hl e hE hx hc ht hg⟩

/-- A fact `R` about neither table nor weight carried from the precondition to the postcondition. -/
theorem HRuns.frame (R : Prop) {y : LP PS β} {c c' : Cur} {tm tm' : Transcript} {Q Xb : β → Prop} {J : TW → Prop}
    {J' : β → TW → Prop} (h : HRuns p L y c tm c' tm' Q Xb J J') :
    HRuns p L y c tm c' tm' Q Xb (fun x => J x ∧ R) fun b x => J' b x ∧ R :=
  ⟨h.1, fun st t l b st' t' l' hl e hE hx hc ht hJ => ⟨h.2 st t l b st' t' l' hl e hE hx hc ht hJ.1, hJ.2⟩⟩

/-- fast100's levels: each level's seam, level 0's `k0` lanes, four lanes after, and `3` columns fewer per level. -/
theorem sch_facts {m : ℕ} {sch : Schedule} (h : fast100 m = some sch) :
    (∀ i < sch.levels.size, LevelSeam (sch.levels.getD i default)) ∧ (sch.levels.getD 0 default).lanes = sch.k0 ∧
      (∀ i < sch.levels.size, 1 ≤ i → (sch.levels.getD i default).lanes = 4) ∧
      ∀ i < sch.levels.size, (sch.levels.getD i default).cols = sch.n - sch.k0 - 3 * i := by
  obtain ⟨⟨-, -, h0, hc0, h4, hc, -⟩, -⟩ := fast100_dims m sch h
  refine ⟨fun i hi => ?_, h0, h4, fun i hi => ?_⟩
  · rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hi, Option.getD_some]
    exact fast100_level m sch h _ (Array.getElem_mem hi)
  · induction i with
    | zero => rw [hc0]; rfl
    | succ i ih =>
      have := hc i (by omega)
      rw [ih (by omega)] at this
      omega

/-- The model's final identity. -/
def Final (P : LigeritoProof) (cf : ℕ → ℕ → F128) (sch : Schedule) (T : F128) (cap : Array ByteArray)
    (t : Transcript) (basisAt : Array F256 → ℕ → Array F256) : Prop :=
  lhsE P.finalProof.yr (out P cf sch T cap t).1.2.2.2 (resE basisAt sch.n (pre P cf sch T cap t).1
    (out P cf sch T cap t).1.2.2.1 (out P cf sch T cap t).1.2.2.2 (out P cf sch T cap t).2.1) =
    (out P cf sch T cap t).2.1.s

/-- `Mid` but for the table and weight's own equation. -/
def MidR (n : ℕ) (b : Array F128) (initG : List 𝕂) (x : Vars × LState × Transcript) (cm : Commit) (lv : Level)
    (g : Array F128) (w : Array F256) : Prop :=
  cm.rows = encodeRows lv g ∧ Sync x.2.1.s x.2.1.q (embedF g) w (x.1.1 - 1) ∧
    WOK n b initG x.1.2.2.1 x.2.1 (x.1.1 - 1) w ∧
    PreOK initG (levelsOf x.1.2.2.1) (x.2.1.oods.toList.map oodOf) (x.2.1.conss.toList.map consOf) (x.1.1 - 1) ∧
    ∀ cs ∈ x.2.1.conss.toList, cs.pos.size ≤ 2 ^ cs.alpha.size

/-- The prover's level-loop state: the split dimension, the last commitment, the caps, the openings and the final
proof. -/
abbrev LS := ℕ × Commit × Array (Array ByteArray) × Array LevelProof × FinalProof

/-- **The honest Ligerito meets its final identity** (`Lg.Honest`): on its table at ring switching's `T` and basis, the
model's final identity holds on the proof the prover returns. Through level 0 the prover reaches `Mid`, each inner level
keeps it, and the last level meets `Final`; the loop's state carries the caps and openings, prefixes of the returned
proof's. -/
theorem honest : Honest := by
  intro resp h ms m sch T cm0 tb b salt nonce L basisAt t l P t' l' hsch htb hb hT hcm0 hbasis hl hrun
  obtain ⟨hn, hL, hsd, hq, -⟩ := fast100_ligerito hsch
  obtain ⟨hseam, hlan0, hlan, hcols⟩ := sch_facts hsch
  have hk1 := fast100_k0 hsch
  unfold ligeritoP at hrun
  rw [bind_go (liftTM_ok (Flock.RecAlgebra.label_eq _ _) l), bind_go (liftTM_ok (Flock.RecAlgebra.absorbF_eq _ _) l),
    bind_go (liftTM_ok (Flock.RecAlgebra.absorbBytes_eq _ _) l)] at hrun
  dsimp only at hrun
  rw [StateT.run_bind] at hrun
  obtain ⟨⟨⟨⟨caps, rproofs, fin, op0⟩, stF⟩, t₁⟩, l₁, h1, h2⟩ := JsonRT.go_bind_ok hrun
  obtain ⟨h21, rfl⟩ := Prod.mk.inj h2
  obtain ⟨hP, rfl⟩ := Prod.mk.inj (Except.ok.inj h21)
  have key := (?prog : HRuns (β := Array (Array ByteArray) × Array LevelProof × FinalProof × LevelProof) P L _
    (cur (init T)) (start cm0.cap T t) (cur (out P (cfOf resp L) sch T cm0.cap t).2.1)
    (out P (cfOf resp L) sch T cm0.cap t).2.2 (fun _ => True)
    (fun r => r.1 = P.recursiveCaps ∧ r.2.1 = P.recursiveProofs ∧ r.2.2.1 = P.finalProof ∧ r.2.2.2 = P.initialProof)
    (fun y => y = (embedF tb, embedF b)) fun _ _ => Final P (cfOf resp L) sch T cm0.cap t basisAt).2 _
    (start cm0.cap T t) l _ _ _ _ hl h1
  case prog =>
    refine (HRuns.sampleVec resp hn _ _ _).bindC fun z _ => HRuns.oodPush _ ?_
    refine (HRuns.powB nonce _ _ _).bindC fun _ _ => (HRuns.sample resp _ _ _).bindC fun b0 _ =>
      (HRuns.modify _ (fun _ => by exact ⟨rfl, rfl, rfl, rfl, rfl⟩) _ _ _).bindC fun _ _ =>
      (HRuns.get _ _ _).bindC fun st _ => (HRuns.sendMsg _ _ _ _).bindC fun _ _ => ?_
    refine HRuns.bindC ((HRuns.initChals resp sch.k0 (sch.n - sch.k0)
      (head P (cfOf resp L) sch.n (init T) (start cm0.cap T t)).1
      (head P (cfOf resp L) sch.n (init T) (start cm0.cap T t)).2 (embedF tb)
      (addScaled (embedF b) (embedF (eqTable (vec (cfOf resp L) (start cm0.cap T t) sch.n)))
        (cfOf resp L ((start cm0.cap T t).k + 1) 0))).preX ?pre0) fun _ _ => ?_
    case pre0 =>
      intro _ _ _ y ⟨⟨⟨s, ⟨⟨⟨hs, hz⟩, ho⟩, hb0⟩, hy⟩, hst⟩, hm⟩
      subst hz hb0 hy
      obtain ⟨h1, h2⟩ := Prod.mk.inj (show (s.tab, s.w) = (embedF tb, embedF b) from hs)
      obtain ⟨h3, h4⟩ := Prod.mk.inj hst
      rw [h3, h4, h1, h2] at hm
      show JC _ _ _ _ _ _ _ _ 0 (s.tab, _)
      rw [h1, h2]
      refine jc_zero hk1 ?_
      rw [show sch.n - sch.k0 + sch.k0 - 1 = sch.n - 1 by omega]
      exact head_sync sch.n T _ tb b hn htb hb hT ho hm
    refine (HRuns.get _ _ _).bindC fun d _ => ?_
    refine HRuns.bindDC (HRuns.commit h ms _ _ _ _ _ _) (fun cm1 _ => ?_)
      (Qf := fun cm1 r => Pre #[cm1.cap] r.1)
      (Xf := fun _ r => r.1 = P.recursiveCaps ∧ r.2.1 = P.recursiveProofs ∧ r.2.2.1 = P.finalProof ∧
        r.2.2.2 = P.initialProof)
      (fun _ _ _ _ => trivial) (fun _ _ _ _ hx => ⟨trivial, hx⟩)
    refine HRuns.bindYC (HRuns.absorbCap cm1.cap P.recursiveCaps[0]! _ _ _) (fun _ _ => ?_)
      (fun _ r _ hr hx => (Pre.get (A := #[]) (hr.trans (Pre.of_eq hx.1))).symm)
    refine (HRuns.oodIntro resp nonce _ (Nat.succ_ne_zero _) 0 _ _ _).bindC fun _ _ =>
      (HRuns.oodIntro resp nonce _ (Nat.succ_ne_zero _) 0 _ _ _).bindC fun _ _ => ?_
    refine HRuns.bindDC (HRuns.query resp nonce ms _ cm0 (hq 0 (by omega)) P.initialProof.rows
      (chals P (cfOf resp L) #[] (head P (cfOf resp L) sch.n (init T) (start cm0.cap T t)).1
        (head P (cfOf resp L) sch.n (init T) (start cm0.cap T t)).2 sch.k0).1 _ _ _) (fun x _ => ?_)
      (Qf := fun x r => Pre #[cm1.cap] r.1 ∧ r.2.2.2 = x.1)
      (Xf := fun _ r => r.1 = P.recursiveCaps ∧ r.2.1 = P.recursiveProofs ∧ r.2.2.1 = P.finalProof ∧
        r.2.2.2 = P.initialProof)
      (fun _ _ _ h => h.1) (fun _ _ _ _ hx => ⟨trivial, hx⟩)
    refine (HRuns.sendMsg _ _ _ _).bindC fun _ _ => (HRuns.powB nonce _ _ _).bindC fun _ _ =>
      (HRuns.sample resp _ _ _).bindC fun bq _ =>
      (HRuns.modify _ (fun _ => by exact ⟨rfl, rfl, rfl, rfl, rfl⟩) _ _ _).bindC fun _ _ => ?_
    refine (HRuns.bindDC (HRuns.loop (σ := LS) _
      (fun j => cur (loop P (cfOf resp L) sch (x₀ P (cfOf resp L) sch T cm0.cap t) j).2.1)
      (fun j => (loop P (cfOf resp L) sch (x₀ P (cfOf resp L) sch T cm0.cap t) j).2.2)
      (fun j s => s.1 = (loop P (cfOf resp L) sch (x₀ P (cfOf resp L) sch T cm0.cap t) j).1.1 ∧
        (j < sch.levels.size - 1 → s.2.2.1.size = j + 1 ∧ s.2.2.2.1.size = j))
      (fun s s' => Pre s.2.2.1 s'.2.2.1 ∧ Pre s.2.2.2.1 s'.2.2.2.1) (fun _ => ⟨Pre.refl _, Pre.refl _⟩)
      (fun _ _ _ h₁ h₂ => ⟨h₁.1.trans h₂.1, h₁.2.trans h₂.2⟩)
      (fun j s => Pre s.2.2.1 P.recursiveCaps ∧ Pre s.2.2.2.1 P.recursiveProofs ∧
        (j + 1 = sch.levels.size - 1 → s.2.2.2.2 = P.finalProof))
      (fun s => s.2.2.1 = P.recursiveCaps ∧ s.2.2.2.1 = P.recursiveProofs ∧ s.2.2.2.2 = P.finalProof)
      (fun j s y => (j < sch.levels.size - 1 → Mid sch.n b ((pre P (cfOf resp L) sch T cm0.cap t).1.toList.map gOf)
        (loop P (cfOf resp L) sch (x₀ P (cfOf resp L) sch T cm0.cap t) j) s.2.1 (sch.levels.getD (j + 1) default) y) ∧
        (j = sch.levels.size - 1 → Final P (cfOf resp L) sch T cm0.cap t basisAt))
      (sch.levels.size - 1) ?body
      (fun _ _ s s' _ he hG hx => ⟨hG.1.trans (Pre.of_eq hx.1), hG.2.trans (Pre.of_eq hx.2.1),
        fun hj => by rw [← he hj]; exact hx.2.2⟩) _ ⟨rfl, fun _ => ⟨rfl, rfl⟩⟩)
      (fun σ _ => by
        refine ((HRuns.pure (Q := fun r => r = (σ.2.2.1, σ.2.2.2.1, σ.2.2.2.2, x.1)) rfl _ _ _).post ?_).cast rfl rfl
        rintro _ _ _ _ ⟨-, -, h⟩
        exact h rfl)
      (Qf := fun σ r => r = (σ.2.2.1, σ.2.2.2.1, σ.2.2.2.2, x.1))
      (Xf := fun _ r => r.1 = P.recursiveCaps ∧ r.2.1 = P.recursiveProofs ∧ r.2.2.1 = P.finalProof ∧
        r.2.2.2 = P.initialProof)
      (fun σ r hσ hr => hr ▸ ⟨hσ.2.1, rfl⟩) (fun σ r _ hr hx => by subst hr; exact ⟨⟨hx.1, hx.2.1, hx.2.2.1⟩, hx⟩)).preX
      ?pre
    case pre =>
      intro r hQ hX y hy
      obtain ⟨s, ⟨⟨⟨⟨x₂, ⟨x₁, ⟨⟨hJC, hd⟩, hcm⟩, hx₂, hv₁, hm₁⟩, hs, hv₂, hm₂⟩, hx⟩, hm₃⟩, hbq⟩, hy⟩ := hy
      subst hd hx₂ hx hbq hy
      obtain ⟨hd1, hsync⟩ := hJC.2 rfl
      have hg1 : d.tab.map F256.c0 = splitK (foldKs (chals P (cfOf resp L) #[]
          (head P (cfOf resp L) sch.n (init T) (start cm0.cap T t)).1
          (head P (cfOf resp L) sch.n (init T) (start cm0.cap T t)).2 sch.k0).1.toList (embedF tb)) := by
        rw [show d.tab = (tw d).1 from rfl, hd1]
        exact embedF_c0 _
      rw [hg1] at hcm hv₁ hm₁ hv₂ hm₂ hm₃
      obtain ⟨h1, h2⟩ := Prod.mk.inj hs
      refine ⟨fun _ => ?_, fun h0 => absurd h0 (by omega)⟩
      show Mid _ _ _ _ _ _ (s.tab, addScaled s.w _ _)
      rw [h1, h2, hd1]
      refine mid_lv0 hb (by omega) (hseam 0 (by omega)) hlan0 (by simpa using hcols 0 (by omega)) hcm0 htb rfl rfl rfl
        rfl rfl rfl rfl (by rw [hd1] at hsync; exact hsync) (commitP_rows hcm) hv₁ hm₁ hv₂ hm₂ ?_ hm₃
      rw [← hX.2.2.2, hQ.2, query_pos]
      rfl
    case body =>
      intro j hj s hs
      have hI := inv_loop P (cfOf resp L) sch T cm0.cap t hL hsd j (by omega)
      by_cases hj1 : j + 1 < sch.levels.size - 1
      · rw [loop_stepN P (cfOf resp L) sch _ hj1]
        have hsz := hs.2 hj
        generalize hx : loop P (cfOf resp L) sch (x₀ P (cfOf resp L) sch T cm0.cap t) j = x at hs hI ⊢
        rw [hs.1]
        have h4 : 4 ≤ x.1.1 := by rw [hI.sd]; omega
        refine HRuns.preX (J := fun y => ∃ gw : Array F128 × Array F256,
          (y = (embedF gw.1, gw.2) ∧ Sync x.2.1.s x.2.1.q (embedF gw.1) gw.2 (x.1.1 - 4 + 3)) ∧
            MidR sch.n b ((pre P (cfOf resp L) sch T cm0.cap t).1.toList.map gOf) x s.2.1
              (sch.levels.getD (j + 1) default) gw.1 gw.2) (HRuns.exPre fun gw => ?_) ?_
        swap
        · intro _ _ _ y hy
          obtain ⟨g, w, rfl, hR⟩ := hy.1 hj
          have hsync := hR.2.1
          rw [show x.1.1 - 1 = x.1.1 - 4 + 3 by omega] at hsync
          exact ⟨(g, w), ⟨rfl, hsync⟩, hR⟩
        refine HRuns.bindC ((HRuns.levelChals resp _ x.2.1 x.2.2 (embedF gw.1) gw.2 (x.1.1 - 4)).frame _)
          fun f _ => ?_
        refine HRuns.ite (fun _ => ?_) fun hne => absurd (by simp; omega) hne
        refine HRuns.bindDC (HRuns.commit h ms _ _ _ _ _ _) (fun cmN _ => ?_)
          (Qf := fun cmN r => (∃ s', r = .yield s' ∧ (s'.1 = (stepN P (cfOf resp L) sch j x).1.1 ∧
            (j + 1 < sch.levels.size - 1 → s'.2.2.1.size = j + 1 + 1 ∧ s'.2.2.2.1.size = j + 1)) ∧
            (Pre s.2.2.1 s'.2.2.1 ∧ Pre s.2.2.2.1 s'.2.2.2.1)) ∧
            ∀ s', r = .yield s' → s'.2.2.1 = s.2.2.1.push cmN.cap)
          (fun _ _ _ hr => hr.1) (fun _ _ _ _ hx => ⟨trivial, hx⟩)
        refine HRuns.bindYC (HRuns.absorbCap cmN.cap P.recursiveCaps[j + 1]! _ _ _) (fun _ _ => ?_)
          (fun _ r _ hr hx => ?_)
        swap
        · obtain ⟨⟨s', rfl, -⟩, hcap⟩ := hr
          have hp := (hx s' rfl).1
          rw [hcap s' rfl] at hp
          have := Pre.get hp
          rw [hsz.1] at this
          exact this.symm
        refine (HRuns.oodIntro resp nonce _ (Nat.succ_ne_zero _) (j + 1) _ _ _).bindC fun _ _ =>
          (HRuns.oodIntro resp nonce _ (Nat.succ_ne_zero _) (j + 1) _ _ _).bindC fun _ _ => ?_
        refine HRuns.bindDC (HRuns.query resp nonce ms _ s.2.1 (hq (j + 1) (by omega)) P.recursiveProofs[j]!.rows
          (chals P (cfOf resp L) #[] x.2.1 x.2.2 4).1 _ _ _) (fun q _ => ?_)
          (Qf := fun q r => ((∃ s', r = .yield s' ∧ (s'.1 = (stepN P (cfOf resp L) sch j x).1.1 ∧
            (j + 1 < sch.levels.size - 1 → s'.2.2.1.size = j + 1 + 1 ∧ s'.2.2.2.1.size = j + 1)) ∧
            (Pre s.2.2.1 s'.2.2.1 ∧ Pre s.2.2.2.1 s'.2.2.2.1)) ∧
            ∀ s', r = .yield s' → s'.2.2.1 = s.2.2.1.push cmN.cap) ∧
            ∀ s', r = .yield s' → s'.2.2.2.1 = s.2.2.2.1.push q.1)
          (fun _ _ _ h => h.1) (fun _ _ _ _ hx => ⟨trivial, hx⟩)
        refine (HRuns.sendMsg _ _ _ _).bindC fun _ _ => (HRuns.powB nonce _ _ _).bindC fun _ _ =>
          (HRuns.sample resp _ _ _).bindC fun bb _ =>
          (HRuns.modify _ (fun _ => by exact ⟨rfl, rfl, rfl, rfl, rfl⟩) _ _ _).bindC fun _ _ => ?_
        refine (HRuns.pure ?_ _ _ _).post ?_
        · refine ⟨⟨⟨_, rfl, ⟨rfl, fun _ => ⟨by rw [Array.size_push, hsz.1], by rw [Array.size_push, hsz.2]⟩⟩,
            Pre.push _ _, Pre.push _ _⟩, fun s' e => ?_⟩, fun s' e => ?_⟩
          · obtain rfl := ForInStep.yield.inj e
            rfl
          · obtain rfl := ForInStep.yield.inj e
            rfl
        · intro r y hQ hX hy
          obtain ⟨rfl, st, ⟨⟨⟨⟨x₁, ⟨x₂, ⟨⟨⟨hf, hx₂, hsync₁, -⟩, hR⟩, hcm⟩, hx₁, hv₁, hm₁⟩, hst, hv₂, hm₂⟩, hq⟩, hm₃⟩,
            hbb⟩, hy⟩ := hy
          intro s' e
          obtain rfl := ForInStep.yield.inj e
          have hrp := Pre.get (hX _ rfl).2.1
          rw [hsz.2] at hrp
          subst hf hx₂ hx₁ hq hbb hy
          obtain ⟨h1, h2⟩ := Prod.mk.inj hst
          refine ⟨fun _ => ?_, fun h => absurd h (by omega)⟩
          show Mid _ _ _ _ _ _ (st.tab, addScaled st.w _ _)
          rw [h1, h2]
          refine mid_stepN hR.1 hR.2.1 hR.2.2.1 hR.2.2.2.1 hR.2.2.2.2 rfl h4 hI.lvs (hseam _ (by omega))
            (hlan _ (by omega) (by omega)) (by rw [hcols _ (by omega), hI.sd]; omega) rfl rfl rfl rfl rfl rfl
            (hsync₁ (by simp; omega)) (commitP_rows hcm) hv₁ hm₁ hv₂ hm₂ ?_ hm₃
          rw [hrp, query_pos]
          rfl
      · rw [loop_stepL P (cfOf resp L) sch _ hj1]
        have hjR : j + 1 = sch.levels.size - 1 := by omega
        generalize hx : loop P (cfOf resp L) sch (x₀ P (cfOf resp L) sch T cm0.cap t) j = x at hs hI ⊢
        have h4 : 4 ≤ x.1.1 := by rw [hI.sd]; omega
        refine HRuns.preX (J := fun y => ∃ gw : Array F128 × Array F256,
          (y = (embedF gw.1, gw.2) ∧ Sync x.2.1.s x.2.1.q (embedF gw.1) gw.2 (x.1.1 - 4 + 3)) ∧
            MidR sch.n b ((pre P (cfOf resp L) sch T cm0.cap t).1.toList.map gOf) x s.2.1
              (sch.levels.getD (j + 1) default) gw.1 gw.2) (HRuns.exPre fun gw => ?_) ?_
        swap
        · intro _ _ _ y hy
          obtain ⟨g, w, rfl, hR⟩ := hy.1 hj
          have hsync := hR.2.1
          rw [show x.1.1 - 1 = x.1.1 - 4 + 3 by omega] at hsync
          exact ⟨(g, w), ⟨rfl, hsync⟩, hR⟩
        refine HRuns.bindDC ((HRuns.levelChals resp _ x.2.1 x.2.2 (embedF gw.1) gw.2 (x.1.1 - 4)).frame _)
          (fun f _ => ?_)
          (Qf := fun f r => (∃ s', r = .yield s' ∧ (s'.1 = (stepL P (cfOf resp L) sch x).1.1 ∧
            (j + 1 < sch.levels.size - 1 → s'.2.2.1.size = j + 1 + 1 ∧ s'.2.2.2.1.size = j + 1)) ∧
            (Pre s.2.2.1 s'.2.2.1 ∧ Pre s.2.2.2.1 s'.2.2.2.1)) ∧
            ∀ s', r = .yield s' → s'.2.2.2.2.yr = Array.foldl (fun acc v => (acc.push v.c0).push v.c1) #[] f)
          (Xf := fun _ r => ∀ s', r = .yield s' → Pre s'.2.2.1 P.recursiveCaps ∧ Pre s'.2.2.2.1 P.recursiveProofs ∧
            (j + 1 = sch.levels.size - 1 → s'.2.2.2.2 = P.finalProof))
          (fun _ _ _ hr => hr.1) (fun _ _ _ _ hx => ⟨trivial, hx⟩)
        refine HRuns.ite (fun hc => absurd hc (by simp [hjR])) fun _ => ?_
        refine HRuns.bindYC (HRuns.absorbAll _ P.finalProof.yr _ _ _) (fun _ _ => ?_) (fun _ r _ hr hx => ?_)
        swap
        · obtain ⟨⟨s', rfl, -⟩, hyr⟩ := hr
          exact (hyr s' rfl).symm.trans (congrArg FinalProof.yr ((hx s' rfl).2.2 hjR))
        refine HRuns.bindDC (HRuns.query resp nonce ms _ s.2.1 (hq _ (by omega)) P.finalProof.rows
          (chals P (cfOf resp L) #[] x.2.1 x.2.2 4).1 _ _ _) (fun q _ => ?_)
          (Qf := fun q r => ((∃ s', r = .yield s' ∧ (s'.1 = (stepL P (cfOf resp L) sch x).1.1 ∧
            (j + 1 < sch.levels.size - 1 → s'.2.2.1.size = j + 1 + 1 ∧ s'.2.2.2.1.size = j + 1)) ∧
            (Pre s.2.2.1 s'.2.2.1 ∧ Pre s.2.2.2.1 s'.2.2.2.1)) ∧
            ∀ s', r = .yield s' → s'.2.2.2.2.yr = Array.foldl (fun acc v => (acc.push v.c0).push v.c1) #[] f) ∧
            ∀ s', r = .yield s' → s'.2.2.2.2.rows = q.1.rows)
          (fun _ _ _ h => h.1) (fun _ _ _ _ hx => ⟨trivial, hx⟩)
        refine (HRuns.powB nonce _ _ _).bindC fun _ _ => (HRuns.sample resp _ _ _).bindC fun _ _ => ?_
        refine (HRuns.pure ?_ _ _ _).post ?_
        · refine ⟨⟨⟨_, rfl, ⟨hs.1, fun h => absurd h (by omega)⟩, Pre.refl _, Pre.refl _⟩, fun s' e => ?_⟩,
            fun s' e => ?_⟩
          · obtain rfl := ForInStep.yield.inj e
            rfl
          · obtain rfl := ForInStep.yield.inj e
            rfl
        · intro r y hQ hX hy
          obtain ⟨rfl, ⟨⟨⟨hf, hy₁, -, hsum⟩, hR⟩, hq⟩, -⟩ := hy
          intro s' e
          obtain rfl := ForInStep.yield.inj e
          have hfin := (hX _ rfl).2.2 hjR
          refine ⟨fun h => absurd h (by omega), fun _ => ?_⟩
          subst hf hy₁ hq
          have hrows := hR.1
          rw [hjR] at hrows
          have hfr := congrArg FinalProof.rows hfin
          unfold Final
          refine goal_last hx.symm hjR hL hsd hrows hR.2.1 hR.2.2.1 hR.2.2.2.1 hR.2.2.2.2 (hseam _ (by omega))
            (hlan _ (by omega) (by omega)) (by rw [hcols _ (by omega), hI.sd]; omega) hbasis rfl rfl hsum
            (by rw [← hfin]; rfl) ?_
          rw [← hfr, query_pos]
          rfl
  subst hP
  exact key ⟨Pre.refl _, Pre.refl _, Pre.refl _, Pre.refl _, Pre.refl _⟩ ⟨rfl, rfl, rfl, rfl⟩ rfl rfl rfl

end Lg

/-- **Ligerito's final identity at the deployed relation** (`LgFinal`) at every rep the session walk runs on its packed
table (`RepWalkP`). -/
theorem lgFinal_walkP (H : Claims.Msg → Claims.Msg)
    (hw : ∀ o : Run H Readable.relation, ∀ i < o.tabs.size, ∀ r < 2, RepWalkP o i r) : LgFinal H Readable.relation :=
  lgFinal_relation H hw Lg.honest

/-- **The Ligerito goal at the deployed relation from the session walk**: the walk's facts at every rep (`ZcAt`,
`LcAt`, `RsAt`, `RepWalk`, `RepWalkP`), the reps' proofs decoding (`RepDecodes`) and the schemes' widths
(`SchemeWidths`). -/
theorem ligerito_walkP (H : Claims.Msg → Claims.Msg)
    (hzc : ∀ o : Run H Readable.relation, ∀ i < o.tabs.size, ∀ r < 2, ZcAt o i r)
    (hlc : ∀ o : Run H Readable.relation, ∀ i < o.tabs.size, ∀ r < 2, LcAt o i r)
    (hrs : ∀ o : Run H Readable.relation, ∀ i < o.tabs.size, ∀ r < 2, RsAt o i r) (hD : RepDecodes H Readable.relation)
    (hw : ∀ o : Run H Readable.relation, ∀ i < o.tabs.size, ∀ r < 2, RepWalk o i r)
    (hwP : ∀ o : Run H Readable.relation, ∀ i < o.tabs.size, ∀ r < 2, RepWalkP o i r)
    (hW : SchemeWidths H Readable.relation) : Ligerito H Readable.relation :=
  ligerito_goal_walk H _ (traceR1CS H) hzc hlc (lcAlgebra H) hrs (rsAlgebra H) hD hw hW (lgFinal_walkP H hwP)

/-- **The consumption goal at the deployed relation from the session walk** (S15): every recorded round is consumed,
from the same walk facts and the schemes' widths. -/
theorem consumed_walkP (H : Claims.Msg → Claims.Msg)
    (hzc : ∀ o : Run H Readable.relation, ∀ i < o.tabs.size, ∀ r < 2, ZcAt o i r)
    (hlc : ∀ o : Run H Readable.relation, ∀ i < o.tabs.size, ∀ r < 2, LcAt o i r)
    (hrs : ∀ o : Run H Readable.relation, ∀ i < o.tabs.size, ∀ r < 2, RsAt o i r) (hD : RepDecodes H Readable.relation)
    (hw : ∀ o : Run H Readable.relation, ∀ i < o.tabs.size, ∀ r < 2, RepWalk o i r)
    (hwP : ∀ o : Run H Readable.relation, ∀ i < o.tabs.size, ∀ r < 2, RepWalkP o i r)
    (hW : SchemeWidths H Readable.relation) : Consumed H Readable.relation :=
  consumed_goal_walk H _ (traceR1CS H) hzc hlc (lcAlgebra H) hrs (rsAlgebra H) hD hw hW (lgFinal_walkP H hwP)

end FlockComplete.Replay
