import Proofs.Flock.Claims.Complete.Zk.Level0Honest

/-!
# The honest masked Ligerito meets the masked model's final identity: the walk

`honestZ`: plain's `Lg.honest` for the masked prover (`ligeritoPZk`) against the masked verifier's model (`outZ`). The
prover's head and its level loop run as plain's; its lane rounds run against the model's (`laneRun`), carrying the
folded table and weight (`LaneJ`), and at the last lane round the pair's sums, its coins, the blinded claim and `ybar`
(`LastVals`), after which the model's claim is the blinded claim `T'` on the split blinded fold (`sync_laneClose`). Level
0 then ends in `Mid` (`mid_lv0Z`) and the last level meets the final identity (`goal_lastZ`).
-/

namespace FlockComplete.L0

open Flock Transcript FlockSoundness FlockSoundness.Refine FlockComplete Replay Replay.Lg

/-- **The masked model's final identity** on the proof `P`. -/
def FinalZ (P : LigeritoProof) (cf : ℕ → ℕ → F128) (sch : Schedule) (T : F128) (cap : Array ByteArray)
    (zl : Zk.Level0) (t : Transcript) (basisAt : Array F256 → ℕ → Array F256) : Prop :=
  lhsE P.finalProof.yr (outZ P cf sch T cap zl t).1.2.2.2 (resE basisAt sch.n (preZ P cf sch T cap zl t).1.1
    (outZ P cf sch T cap zl t).1.2.2.1 (outZ P cf sch T cap zl t).1.2.2.2 (outZ P cf sch T cap zl t).2.1) =
    (outZ P cf sch T cap zl t).2.1.s

/-- The masked model's first `j` lane rounds, after its head. -/
abbrev laneCm (P : LigeritoProof) (cf : ℕ → ℕ → F128) (sch : Schedule) (T : F128) (cap : Array ByteArray)
    (zm : Zk.Level0Msgs) (t : Transcript) (j : ℕ) : (Array F256 × F256 × F256) × LState × Transcript :=
  laneRun P cf sch.k0 zm (head P cf sch.n (init T) (startZ cap T t)).1 (head P cf sch.n (init T) (startZ cap T t)).2 j

/-- The blinded fold `f + ρ_0 R_{2r} + ρ_1 R_{2r+1}` of the table `tb` at the lane coins `ch` and the pair `rho`. -/
def blindOf (tb : Array F128) (extra : ℕ → ℕ → F128) (rep : ℕ) (ch : Array F256) (rho : F256 × F256) : Array F256 :=
  ZkTables.Level0.blinded (foldKs ch.toList (embedF tb)) rho
    (ZkTables.Level0.pairLane extra (2 * rep) (foldKs ch.toList (embedF tb)).size)
    (ZkTables.Level0.pairLane extra (2 * rep + 1) (foldKs ch.toList (embedF tb)).size)

/-- **The lane loop's state and the table and weight after `j` of `k0` lane rounds** against the model's `Cm`: the
challenges are the model's; before the last, the table `tb` and the weight `W0` folded at them; after it, the split
blinded fold and the lifted weight, `ybar` the prover's, and the model's claim the blinded claim on them. -/
def LaneJ (Cm : ℕ → (Array F256 × F256 × F256) × LState × Transcript) (k0 N lanes tPad rep : ℕ) (tb : Array F128)
    (W0 : Array F256) (extra lanePad : ℕ → ℕ → F128) (j : ℕ) (s : Array F256 × (F256 × F256) × Zk.Level0Msgs)
    (y : TW) : Prop :=
  s.1 = (Cm j).1.1 ∧
  (j < k0 → y = (foldKs s.1.toList (embedF tb), foldKs s.1.toList W0)) ∧
  (j = k0 → s.2.1 = (Cm j).1.2 ∧
    y = (embedF (splitK (blindOf tb extra rep s.1 s.2.1)), liftU (foldKs s.1.toList W0)) ∧
    s.2.2.ybar = ybarOf (eqTableK s.1) s.2.1 rep lanes tPad lanePad ∧ Sync (Cm j).2.1.s (Cm j).2.1.q y.1 y.2 N)

/-- **The last lane round's values** at the challenges `ch` and the prover's folded table `tab` and weight `w`: the
pair's masked sums, the blinded claim at the state's pair coins, and `ybar`. -/
def LastVals (lay : Zk.PadLayout) (hv : Array F128) (rep lanes tPad : ℕ) (extra lanePad : ℕ → ℕ → F128)
    (ch tab w : Array F256) (s : Array F256 × (F256 × F256) × Zk.Level0Msgs) : Prop :=
  s.1 = ch ∧
  s.2.2.e0 = dotK w (embedF (ZkTables.Level0.pairLane extra (2 * rep) tab.size)) +
    ⟨hv[lay.lgE 0 0]!, hv[lay.lgE 0 1]!⟩ ∧
  s.2.2.e1 = dotK w (embedF (ZkTables.Level0.pairLane extra (2 * rep + 1) tab.size)) +
    ⟨hv[lay.lgE 1 0]!, hv[lay.lgE 1 1]!⟩ ∧
  s.2.2.tPrime = dotK w (ZkTables.Level0.blinded tab s.2.1 (ZkTables.Level0.pairLane extra (2 * rep) tab.size)
    (ZkTables.Level0.pairLane extra (2 * rep + 1) tab.size)) ∧
  s.2.2.ybar = ybarOf (eqTableK ch) s.2.1 rep lanes tPad lanePad

/-- What the masked prover's Ligerito program returns: the caps, openings and final proof, level 0's opening, its
hiding messages, the lane coins, the pair's coins, and the out-of-domain coin and masked value. -/
abbrev ZOut := Array (Array ByteArray) × Array LevelProof × FinalProof × LevelProof × Zk.Level0Msgs × Array F256 ×
  (F256 × F256) × F128 × F128

/-- A run of a step on the model's: where it ends, and its table and weight. -/
theorem HRuns.ends {p : LigeritoProof} {L : ℕ} {β : Type} {y : LP PS β} {c c' : Cur} {tm tm' : Transcript}
    {Q Xb : β → Prop} {J : TW → Prop} {J' : β → TW → Prop} {st : LigState} {t : Transcript} {l : Array Claims.Msg}
    {b : β} {st' : LigState} {t' : Transcript} {l' : Array Claims.Msg} (hl : l.size = L + t.k)
    (e : ((y.run st).run t).go l = (.ok ((b, st'), t'), l')) (h : HRuns p L y c tm c' tm' Q Xb J J') :
    (Ext p st' → Xb b → pcur st = c → t = tm → pcur st' = c' ∧ t' = tm') ∧
      (Ext p st' → Xb b → pcur st = c → t = tm → J (tw st) → J' b (tw st')) :=
  ⟨(h.1 st t l b st' t' l' hl e).2.2.2, h.2 st t l b st' t' l' hl e⟩

set_option maxHeartbeats 4000000 in
/-- **The honest masked Ligerito meets the masked model's final identity**: at a `--zk` schedule, as rep `rep < 2`, on
a table `tb` and weights `b` of `2^n` entries whose level-0 commitment has `tb`'s padded rows (`encodeRowsZk`), with a
basis whose value at every original point `q ‖ bits(y)` is the multilinear extension of `b` there, the masked prover's
Ligerito, run from a transcript at which `L + k` bodies have been sent (`k` its position), ends at the masked model's
transcript, and the masked model's final identity holds on the proof it returns. -/
theorem honestZ (resp : ℕ → Claims.Msg) (h : ByteArray → ByteArray) (ms : MerkleScheme) {m : ℕ} {sch : Schedule}
    {tPad : ℕ} (T : F128) (cm0 : Commit) (tb b : Array F128) (lay : Zk.PadLayout) (hv : Array F128) (rep : ℕ)
    (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64)
    (basisAt : Array F256 → ℕ → Array F256) (L : ℕ) (t : Transcript) (l : Array Claims.Msg) {P : LigeritoProof}
    {zl : Zk.Level0Msgs} {rc : Zk.L0Rec} {t' : Transcript} {l' : Array Claims.Msg}
    (h1 : 25 ≤ m) (h2 : m ≤ 35) (hsch : Zk.schedule m = .ok (sch, tPad)) (hrep : rep < 2)
    (htb : tb.size = 2 ^ sch.n) (hb : b.size = 2 ^ sch.n)
    (hcm0 : cm0.rows = encodeRowsZk (sch.levels.getD 0 default) tPad tb extra lanePad)
    (hbasis : ∀ q e y, q.size + e = sch.n → y < 2 ^ e →
      gOf (basisAt q e)[y]! = Model.bHat execArith sch.n (fun w => fOf b[w]!) (q.toList.map gOf ++ Model.bitsK e y))
    (hl : l.size = L + t.k)
    (hrun : ((ligeritoPZk (asksS resp) (Hashes.ofSha512 h) ms sch T cm0 tb b lay hv rep tPad extra lanePad salt
      nonce).run t).go l = (.ok ((P, zl, rc), t'), l')) :
    t' = (outZ P (cfOf resp L) sch T cm0.cap ⟨tPad, rep, zl⟩ t).2.2 ∧
      FinalZ P (cfOf resp L) sch T cm0.cap ⟨tPad, rep, zl⟩ t basisAt := by
  obtain ⟨hn, hk0, hL, hsd, hq, hseam, hlan0, hlan, hcols, htp, -, -⟩ := zkSch_facts h1 h2 hsch
  dsimp only at hn hk0 hL hsd hq hseam hlan0 hlan hcols htp
  unfold ligeritoPZk at hrun
  rw [bind_go (liftTM_ok (Flock.RecAlgebra.label_eq _ _) l), bind_go (liftTM_ok (Flock.RecAlgebra.absorbF_eq _ _) l),
    bind_go (liftTM_ok (Flock.RecAlgebra.absorbBytes_eq _ _) l),
    bind_go (liftTM_ok (Flock.RecAlgebra.label_eq _ _) l)] at hrun
  dsimp only at hrun
  rw [StateT.run_bind] at hrun
  obtain ⟨⟨⟨⟨caps, rproofs, fin, op0, zl', initCh, rho, b0', y0⟩, stF⟩, t₁⟩, l₁, hp1, hp2⟩ :=
    JsonRT.go_bind_ok hrun
  obtain ⟨h21, rfl⟩ := Prod.mk.inj hp2
  obtain ⟨hP3, rfl⟩ := Prod.mk.inj (Except.ok.inj h21)
  obtain ⟨hP, hP2⟩ := Prod.mk.inj hP3
  obtain ⟨rfl, -⟩ := Prod.mk.inj hP2
  have key := HRuns.ends (t := startZ cm0.cap T t) hl hp1 (?prog : HRuns (β := ZOut) P L _
    (cur (init T)) (startZ cm0.cap T t) (cur (outZ P (cfOf resp L) sch T cm0.cap ⟨tPad, rep, zl'⟩ t).2.1)
    (outZ P (cfOf resp L) sch T cm0.cap ⟨tPad, rep, zl'⟩ t).2.2 (fun _ => True)
    (fun r => r.1 = P.recursiveCaps ∧ r.2.1 = P.recursiveProofs ∧ r.2.2.1 = P.finalProof ∧
      r.2.2.2.1 = P.initialProof ∧ r.2.2.2.2.1 = zl')
    (fun y => y = (embedF tb, embedF b)) fun _ _ => FinalZ P (cfOf resp L) sch T cm0.cap ⟨tPad, rep, zl'⟩ t basisAt)
  case prog =>
    refine (HRuns.sampleVec resp hn _ _ _).bindC fun z _ => HRuns.oodPush _ ?_
    refine (HRuns.powB nonce _ _ _).bindC fun _ _ => (HRuns.sample resp _ _ _).bindC fun b0 _ =>
      (HRuns.modify _ (fun _ => by exact ⟨rfl, rfl, rfl, rfl, rfl⟩) _ _ _).bindC fun _ _ =>
      (HRuns.get _ _ _).bindC fun st _ => (HRuns.sendMsg _ _ _ _).bindC fun _ _ => ?_
    refine HRuns.bindDC ((HRuns.loop (σ := Array F256 × (F256 × F256) × Zk.Level0Msgs) _
      (fun j => cur (laneCm P (cfOf resp L) sch T cm0.cap zl' t j).2.1)
      (fun j => (laneCm P (cfOf resp L) sch T cm0.cap zl' t j).2.2)
      (fun _ _ => True) (fun _ _ => True) (fun _ => trivial) (fun _ _ _ _ _ => trivial)
      (fun j s => j + 1 = sch.k0 → s.2.2 = zl') (fun s => s.2.2 = zl')
      (LaneJ (laneCm P (cfOf resp L) sch T cm0.cap zl' t) sch.k0 (sch.n - sch.k0)
        (2 ^ (sch.levels.getD 0 default).lanes) tPad rep tb
        (addScaled (embedF b) (embedF (eqTable (vec (cfOf resp L) (startZ cm0.cap T t) sch.n)))
          (cfOf resp L ((startZ cm0.cap T t).k + 1) 0)) extra lanePad)
      sch.k0 ?lane (fun j _ s s' _ he _ hx hj => by rw [← he hj]; exact hx) _ trivial).preX ?pre0)
      (fun sl _ => ?_)
      (Qf := fun sl r => r.2.2.2.2.1 = sl.2.2)
      (Xf := fun _ r => r.1 = P.recursiveCaps ∧ r.2.1 = P.recursiveProofs ∧ r.2.2.1 = P.finalProof ∧
        r.2.2.2.1 = P.initialProof ∧ r.2.2.2.2.1 = zl')
      (fun _ _ _ _ => trivial) (fun _ _ _ hr hx => ⟨hr.symm.trans hx.2.2.2.2, hx⟩)
    case lane =>
      intro j hj s _
      have hfj := laneRun_fields P (cfOf resp L) sch.k0 zl' (head P (cfOf resp L) sch.n (init T) (startZ cm0.cap T t)).1
        (head P (cfOf resp L) sch.n (init T) (startZ cm0.cap T t)).2 j (by omega)
      by_cases hj1 : j + 1 = sch.k0
      · refine HRuns.cast (c₁ := cur (chalZ P (cfOf resp L) zl' (laneCm P (cfOf resp L) sch T cm0.cap zl' t j).2.1
            (laneCm P (cfOf resp L) sch T cm0.cap zl' t j).2.2).2.2.1)
          (tm₁ := (chalZ P (cfOf resp L) zl' (laneCm P (cfOf resp L) sch T cm0.cap zl' t j).2.1
            (laneCm P (cfOf resp L) sch T cm0.cap zl' t j).2.2).2.2.2) ?_
          (by show _ = cur (laneStep P (cfOf resp L) sch.k0 zl' j _).2.1; rw [laneStep, ite_eq_left hj1])
          (by show _ = (laneStep P (cfOf resp L) sch.k0 zl' j _).2.2; rw [laneStep, ite_eq_left hj1])
        refine (HRuns.sampleK resp _ _ _).bindC fun rr _ => ?_
        refine (HRuns.modify _ (fun _ => by exact ⟨rfl, rfl, rfl, rfl, rfl⟩) _ _ _).bindC fun _ _ => ?_
        refine HRuns.ite (fun _ => ?_) (fun hne => absurd (by simp [hj1]) hne)
        refine (HRuns.get _ _ _).bindC fun st _ => ?_
        refine HRuns.mono (Q := fun r => ∃ s', r = .yield s' ∧ LastVals lay hv rep
            (2 ^ (sch.levels.getD 0 default).lanes) tPad extra lanePad (s.1.push rr) st.tab st.w s')
          (Xb := fun r => ∀ s', r = .yield s' → s'.2.2 = zl') ?_
          (fun _ ⟨s', e, _⟩ => ⟨s', e, trivial, trivial⟩) (fun _ _ hx s' e => hx s' e hj1) (fun _ h => h)
          (fun _ _ _ _ h => h)
        refine HRuns.bindYC (HRuns.absorbKM _ zl'.e0 _ _ _) (fun _ _ => ?_)
          (fun _ r _ ⟨s', e, hl⟩ hx => hl.2.1.symm.trans (congrArg Zk.Level0Msgs.e0 (hx s' e)))
        refine HRuns.bindYC (HRuns.absorbKM _ zl'.e1 _ _ _) (fun _ _ => ?_)
          (fun _ r _ ⟨s', e, hl⟩ hx => hl.2.2.1.symm.trans (congrArg Zk.Level0Msgs.e1 (hx s' e)))
        refine HRuns.bindDC (HRuns.sampleK resp _ _ _) (fun rh0 _ => ?_)
          (Qf := fun rh0 r => ∃ s', r = .yield s' ∧ LastVals lay hv rep (2 ^ (sch.levels.getD 0 default).lanes) tPad
            extra lanePad (s.1.push rr) st.tab st.w s' ∧ s'.2.1.1 = rh0)
          (Xf := fun _ r => ∀ s', r = .yield s' → s'.2.2 = zl')
          (fun _ _ _ ⟨s', e, hl, _⟩ => ⟨s', e, hl⟩) (fun _ _ _ _ hx => ⟨trivial, hx⟩)
        refine HRuns.bindDC (HRuns.sampleK resp _ _ _) (fun rh1 _ => ?_)
          (Qf := fun rh1 r => ∃ s', r = .yield s' ∧ LastVals lay hv rep (2 ^ (sch.levels.getD 0 default).lanes) tPad
            extra lanePad (s.1.push rr) st.tab st.w s' ∧ s'.2.1 = (rh0, rh1))
          (Xf := fun _ r => ∀ s', r = .yield s' → s'.2.2 = zl')
          (fun _ _ _ ⟨s', e, hl, h⟩ => ⟨s', e, hl, by rw [h]⟩) (fun _ _ _ _ hx => ⟨trivial, hx⟩)
        refine HRuns.ite (fun _ => HRuns.throwBind _ _ _ _ _ _ _ _ _ _) (fun _ => ?_)
        refine (HRuns.label _ _ _ _).bindC fun _ _ => ?_
        refine HRuns.bindYC (HRuns.absorbKM _ zl'.tPrime _ _ _) (fun _ _ => ?_)
          (fun _ r _ ⟨s', e, hl, h⟩ hx => by rw [← hx s' e, hl.2.2.2.1, h]; rfl)
        refine HRuns.bindYC (HRuns.absorbSliceM _ (ybarFlat zl'.ybar) _ _ _) (fun _ _ => ?_)
          (fun _ r _ ⟨s', e, hl, h⟩ hx => by rw [← hx s' e, hl.2.2.2.2, h]; rfl)
        refine (HRuns.modify _ (fun _ => by exact ⟨rfl, rfl, rfl, rfl, rfl⟩) _ _ _).bindC fun _ _ => ?_
        refine (HRuns.get _ _ _).bindC fun st₄ _ => (HRuns.sendMsg _ _ _ _).bindC fun _ _ => ?_
        refine (HRuns.pure ⟨_, rfl, ⟨rfl, rfl, rfl, rfl, rfl⟩, rfl⟩ _ _ _).post ?_
        rintro r y - hX ⟨rfl, ⟨⟨s₃, ⟨⟨⟨⟨s₁, ⟨hJ0, hrr⟩, hy₁⟩, hst⟩, hrh0⟩, hrh1⟩, hy₃⟩, hst₄⟩, hm⟩ s' e
        obtain rfl := ForInStep.yield.inj e
        have hZ := hX _ rfl
        obtain ⟨h1, h2⟩ := Prod.mk.inj (hJ0.2.1 (by omega))
        have hts := hst.trans hy₁
        have hstab : st.tab = foldKs (s.1.push rr).toList (embedF tb) := by
          rw [foldKs_push, ← h1]; exact congrArg Prod.fst hts
        have hstw : st.w = foldKs (s.1.push rr).toList (addScaled (embedF b)
            (embedF (eqTable (vec (cfOf resp L) (startZ cm0.cap T t) sch.n)))
            (cfOf resp L ((startZ cm0.cap T t).k + 1) 0)) := by
          rw [foldKs_push, ← h2]; exact congrArg Prod.snd hts
        have hs3w : s₃.w = st.w := congrArg Prod.snd hst.symm
        have hB : (Array.range st.tab.size).map (fun c => st.tab[c]! +
            rh0.smul (Array.map (extra (2 * rep)) (Array.range st.tab.size))[c]! +
            rh1.smul (Array.map (extra (2 * rep + 1)) (Array.range st.tab.size))[c]!) =
            blindOf tb extra rep (s.1.push rr) (rh0, rh1) := by
          rw [hstab]; rfl
        have hy' : y = (embedF (splitK (blindOf tb extra rep (s.1.push rr) (rh0, rh1))),
            liftU (foldKs (s.1.push rr).toList (addScaled (embedF b)
              (embedF (eqTable (vec (cfOf resp L) (startZ cm0.cap T t) sch.n)))
              (cfOf resp L ((startZ cm0.cap T t).k + 1) 0)))) := by
          rw [hy₃, ← hB, ← hstw, ← hs3w]; rfl
        have hTP : zl'.tPrime = dotK (foldKs (s.1.push rr).toList (addScaled (embedF b)
            (embedF (eqTable (vec (cfOf resp L) (startZ cm0.cap T t) sch.n)))
            (cfOf resp L ((startZ cm0.cap T t).k + 1) 0))) (blindOf tb extra rep (s.1.push rr) (rh0, rh1)) := by
          rw [← hZ, ← hB, ← hstw]
        obtain ⟨h41, h42⟩ := Prod.mk.inj (hst₄.trans hy')
        have hs1 : s.1.size = j := by rw [hJ0.1]; exact hfj.2.2.2.2.2.2.1
        have hfs : ∀ x : Array F256, x.size = 2 ^ sch.n →
            (foldKs (s.1.push rr).toList x).size = 2 ^ (sch.n - sch.k0) := by
          intro x hx
          rw [foldKs_size, hx, Array.toList_push, List.length_append, Array.length_toList, hs1,
            List.length_singleton, hj1, Nat.pow_div (by omega) (by norm_num)]
        have hw := hfs (addScaled (embedF b) (embedF (eqTable (vec (cfOf resp L) (startZ cm0.cap T t) sch.n)))
          (cfOf resp L ((startZ cm0.cap T t).k + 1) 0)) (by rw [addScaled_size, embedF_size, hb])
        have hBsz : (blindOf tb extra rep (s.1.push rr) (rh0, rh1)).size = 2 ^ (sch.n - sch.k0) := by
          rw [blindOf, ZkTables.Level0.blinded, Array.size_map, Array.size_range]
          exact hfs _ (by rw [embedF_size, htb])
        refine ⟨?_, fun h => absurd h (by omega), fun _ => ⟨?_, hy', rfl, ?_⟩⟩
        · show s.1.push rr = (laneStep P (cfOf resp L) sch.k0 zl' j _).1.1
          rw [laneStep, ite_eq_left hj1, ← hJ0.1, hrr]
          rfl
        · show (rh0, rh1) = (laneStep P (cfOf resp L) sch.k0 zl' j _).1.2
          rw [laneStep, ite_eq_left hj1, hrh0, hrh1]
          rfl
        · show Sync (laneStep P (cfOf resp L) sch.k0 zl' j (laneCm P (cfOf resp L) sch T cm0.cap zl' t j)).2.1.s
            (laneStep P (cfOf resp L) sch.k0 zl' j (laneCm P (cfOf resp L) sch T cm0.cap zl' t j)).2.1.q y.1 y.2
            (sch.n - sch.k0)
          rw [laneStep, ite_eq_left hj1]
          show Sync zl'.tPrime (Poly.ofMsg P.messages[(cur (laneCm P (cfOf resp L) sch T cm0.cap zl' t j).2.1).msg]!
            zl'.tPrime) y.1 y.2 (sch.n - sch.k0)
          rw [hm, h41, h42, hTP, hy']
          exact sync_laneClose _ _ _ hw hBsz
      · refine HRuns.cast (c₁ := cur (chal P (cfOf resp L) (laneCm P (cfOf resp L) sch T cm0.cap zl' t j).2.1
            (laneCm P (cfOf resp L) sch T cm0.cap zl' t j).2.2).2.1)
          (tm₁ := (chal P (cfOf resp L) (laneCm P (cfOf resp L) sch T cm0.cap zl' t j).2.1
            (laneCm P (cfOf resp L) sch T cm0.cap zl' t j).2.2).2.2) ?_
          (by show _ = cur (laneStep P (cfOf resp L) sch.k0 zl' j _).2.1; rw [laneStep, ite_eq_right hj1])
          (by show _ = (laneStep P (cfOf resp L) sch.k0 zl' j _).2.2; rw [laneStep, ite_eq_right hj1])
        refine (HRuns.sampleK resp _ _ _).bindC fun rr _ => ?_
        refine (HRuns.modify _ (fun _ => by exact ⟨rfl, rfl, rfl, rfl, rfl⟩) _ _ _).bindC fun _ _ => ?_
        refine HRuns.ite (fun hc => absurd (by simpa using hc) hj1) fun _ => ?_
        refine (HRuns.get _ _ _).bindC fun st _ => (HRuns.sendMsg _ _ _ _).bindC fun _ _ => ?_
        refine (HRuns.pure ⟨_, rfl, trivial, trivial⟩ _ _ _).post ?_
        rintro r y - - ⟨rfl, ⟨⟨s₁, ⟨hJ0, hrr⟩, hy₁⟩, hst⟩, hm⟩ s' e
        obtain rfl := ForInStep.yield.inj e
        obtain ⟨h1, h2⟩ := Prod.mk.inj (hJ0.2.1 (by omega))
        refine ⟨?_, fun _ => ?_, fun h => absurd h (by omega)⟩
        · show s.1.push rr = (laneStep P (cfOf resp L) sch.k0 zl' j _).1.1
          rw [laneStep, ite_eq_right hj1, ← hJ0.1, hrr]
          rfl
        · rw [hy₁, foldKs_push, foldKs_push, ← h1, ← h2]
          rfl
    case pre0 =>
      intro _ _ _ y ⟨⟨⟨s, ⟨⟨⟨hs, hz⟩, ho⟩, hb0⟩, hy⟩, hst⟩, hm⟩
      subst hz hb0 hy
      obtain ⟨h1, h2⟩ := Prod.mk.inj (show (s.tab, s.w) = (embedF tb, embedF b) from hs)
      refine ⟨rfl, fun _ => ?_, fun h0 => absurd h0 (by omega)⟩
      show (s.tab, addScaled s.w _ _) = _
      rw [h1, h2]
      rfl
    refine (HRuns.get _ _ _).bindC fun d _ => ?_
    refine HRuns.bindDC (HRuns.commit h ms _ _ _ _ _ _) (fun cm1 _ => ?_)
      (Qf := fun cm1 r => Pre #[cm1.cap] r.1 ∧ r.2.2.2.2.1 = sl.2.2)
      (Xf := fun _ r => r.1 = P.recursiveCaps ∧ r.2.1 = P.recursiveProofs ∧ r.2.2.1 = P.finalProof ∧
        r.2.2.2.1 = P.initialProof ∧ r.2.2.2.2.1 = zl')
      (fun _ _ _ h => h.2) (fun _ _ _ _ hx => ⟨trivial, hx⟩)
    refine HRuns.bindYC (HRuns.absorbCap cm1.cap P.recursiveCaps[0]! _ _ _) (fun _ _ => ?_)
      (fun _ r _ hr hx => (Pre.get (A := #[]) (hr.1.trans (Pre.of_eq hx.1))).symm)
    refine (HRuns.oodIntro resp nonce _ (Nat.succ_ne_zero _) 0 _ _ _).bindC fun _ _ =>
      (HRuns.oodIntro resp nonce _ (Nat.succ_ne_zero _) 0 _ _ _).bindC fun _ _ => ?_
    refine HRuns.bindDC (HRuns.query resp nonce ms _ cm0 (hq 0 (by omega)) P.initialProof.rows
      (laneCm P (cfOf resp L) sch T cm0.cap zl' t sch.k0).1.1 _ _ _) (fun x _ => ?_)
      (Qf := fun x r => (Pre #[cm1.cap] r.1 ∧ r.2.2.2.2.1 = sl.2.2) ∧ r.2.2.2.1 = x.1)
      (Xf := fun _ r => r.1 = P.recursiveCaps ∧ r.2.1 = P.recursiveProofs ∧ r.2.2.1 = P.finalProof ∧
        r.2.2.2.1 = P.initialProof ∧ r.2.2.2.2.1 = zl')
      (fun _ _ _ h => h.1) (fun _ _ _ _ hx => ⟨trivial, hx⟩)
    refine (HRuns.sendMsg _ _ _ _).bindC fun _ _ => (HRuns.powB nonce _ _ _).bindC fun _ _ =>
      (HRuns.sample resp _ _ _).bindC fun bq _ =>
      (HRuns.modify _ (fun _ => by exact ⟨rfl, rfl, rfl, rfl, rfl⟩) _ _ _).bindC fun _ _ => ?_
    refine (HRuns.bindDC (HRuns.loop (σ := LS) _
      (fun j => cur (loop P (cfOf resp L) sch (x₀Z P (cfOf resp L) sch T cm0.cap ⟨tPad, rep, zl'⟩ t) j).2.1)
      (fun j => (loop P (cfOf resp L) sch (x₀Z P (cfOf resp L) sch T cm0.cap ⟨tPad, rep, zl'⟩ t) j).2.2)
      (fun j s => s.1 = (loop P (cfOf resp L) sch (x₀Z P (cfOf resp L) sch T cm0.cap ⟨tPad, rep, zl'⟩ t) j).1.1 ∧
        (j < sch.levels.size - 1 → s.2.2.1.size = j + 1 ∧ s.2.2.2.1.size = j))
      (fun s s' => Pre s.2.2.1 s'.2.2.1 ∧ Pre s.2.2.2.1 s'.2.2.2.1) (fun _ => ⟨Pre.refl _, Pre.refl _⟩)
      (fun _ _ _ h₁ h₂ => ⟨h₁.1.trans h₂.1, h₁.2.trans h₂.2⟩)
      (fun j s => Pre s.2.2.1 P.recursiveCaps ∧ Pre s.2.2.2.1 P.recursiveProofs ∧
        (j + 1 = sch.levels.size - 1 → s.2.2.2.2 = P.finalProof))
      (fun s => s.2.2.1 = P.recursiveCaps ∧ s.2.2.2.1 = P.recursiveProofs ∧ s.2.2.2.2 = P.finalProof)
      (fun j s y => (j < sch.levels.size - 1 → Mid sch.n b
        ((preZ P (cfOf resp L) sch T cm0.cap ⟨tPad, rep, zl'⟩ t).1.1.toList.map gOf)
        (loop P (cfOf resp L) sch (x₀Z P (cfOf resp L) sch T cm0.cap ⟨tPad, rep, zl'⟩ t) j) s.2.1
        (sch.levels.getD (j + 1) default) y) ∧
        (j = sch.levels.size - 1 → FinalZ P (cfOf resp L) sch T cm0.cap ⟨tPad, rep, zl'⟩ t basisAt))
      (sch.levels.size - 1) ?body
      (fun _ _ s s' _ he hG hx => ⟨hG.1.trans (Pre.of_eq hx.1), hG.2.trans (Pre.of_eq hx.2.1),
        fun hj => by rw [← he hj]; exact hx.2.2⟩) _ ⟨rfl, fun _ => ⟨rfl, rfl⟩⟩)
      (fun σ _ => by
        refine ((HRuns.pure (Q := fun r : ZOut => r.1 = σ.2.2.1 ∧ r.2.1 = σ.2.2.2.1 ∧ r.2.2.1 = σ.2.2.2.2 ∧
          r.2.2.2.1 = x.1 ∧ r.2.2.2.2.1 = sl.2.2) ⟨rfl, rfl, rfl, rfl, rfl⟩ _ _ _).post ?_).cast rfl rfl
        rintro _ _ _ _ ⟨-, -, h⟩
        exact h rfl)
      (Qf := fun σ r => r.1 = σ.2.2.1 ∧ r.2.1 = σ.2.2.2.1 ∧ r.2.2.1 = σ.2.2.2.2 ∧ r.2.2.2.1 = x.1 ∧
        r.2.2.2.2.1 = sl.2.2)
      (Xf := fun _ r => r.1 = P.recursiveCaps ∧ r.2.1 = P.recursiveProofs ∧ r.2.2.1 = P.finalProof ∧
        r.2.2.2.1 = P.initialProof ∧ r.2.2.2.2.1 = zl')
      (fun σ r hσ hr => ⟨⟨by rw [hr.1]; exact hσ.2.1, hr.2.2.2.2⟩, hr.2.2.2.1⟩)
      (fun σ r _ hr hx => ⟨⟨hr.1.symm.trans hx.1, hr.2.1.symm.trans hx.2.1, hr.2.2.1.symm.trans hx.2.2.1⟩,
        hx⟩)).preX ?pre
    case pre =>
      intro r hQ hX y hy
      obtain ⟨s, ⟨⟨⟨⟨x₂, ⟨x₁, ⟨⟨hLJ, hd⟩, hcm⟩, hx₂, hv₁, hm₁⟩, hs, hv₂, hm₂⟩, hx⟩, hm₃⟩, hbq⟩, hy⟩ := hy
      subst hd hx₂ hx hbq hy
      obtain ⟨hch, -, hLk⟩ := hLJ
      obtain ⟨hrho, hd1, hybar, hsync⟩ := hLk rfl
      have hzl : sl.2.2 = zl' := hQ.1.2.symm.trans hX.2.2.2.2
      have hg1 : d.tab.map F256.c0 = splitK (blindOf tb extra rep sl.1 sl.2.1) := by
        rw [show d.tab = (tw d).1 from rfl, hd1]
        exact embedF_c0 _
      rw [hg1] at hcm hv₁ hm₁ hv₂ hm₂ hm₃
      obtain ⟨h1, h2⟩ := Prod.mk.inj hs
      refine ⟨fun _ => ?_, fun h0 => absurd h0 (by omega)⟩
      show Mid _ _ _ _ _ _ (s.tab, addScaled s.w _ _)
      rw [h1, h2, hd1]
      rw [hd1] at hsync
      rw [hch, hrho] at hcm hv₁ hm₁ hv₂ hm₂ hm₃ hsync hybar ⊢
      rw [hzl] at hybar
      refine mid_lv0Z (zl := ⟨tPad, rep, zl'⟩) hb (by omega) (hseam 0 (by omega)) hlan0
        (by simpa using hcols 0 (by omega)) hcm0 htb hrep htp rfl hybar rfl rfl rfl rfl rfl rfl hsync
        (commitP_rows hcm) hv₁ hm₁ hv₂ hm₂ ?_ hm₃
      rw [← hX.2.2.2.1, hQ.2, query_pos]
      rfl
    case body =>
      intro j hj s hs
      have hI := inv_loop_of P (cfOf resp L) sch (inv_x₀Z P (cfOf resp L) sch T cm0.cap ⟨tPad, rep, zl'⟩ t hL hk0)
        hsd j (by omega)
      by_cases hj1 : j + 1 < sch.levels.size - 1
      · rw [loop_stepN P (cfOf resp L) sch _ hj1]
        have hsz := hs.2 hj
        generalize hx : loop P (cfOf resp L) sch (x₀Z P (cfOf resp L) sch T cm0.cap ⟨tPad, rep, zl'⟩ t) j = x
          at hs hI ⊢
        rw [hs.1]
        have h4 : 4 ≤ x.1.1 := by rw [hI.sd]; omega
        refine HRuns.preX (J := fun y => ∃ gw : Array F128 × Array F256,
          (y = (embedF gw.1, gw.2) ∧ Sync x.2.1.s x.2.1.q (embedF gw.1) gw.2 (x.1.1 - 4 + 3)) ∧
            MidR sch.n b ((preZ P (cfOf resp L) sch T cm0.cap ⟨tPad, rep, zl'⟩ t).1.1.toList.map gOf) x s.2.1
              (sch.levels.getD (j + 1) default) gw.1 gw.2) (HRuns.exPre fun gw => ?_) ?_
        swap
        · intro _ _ _ y hy
          obtain ⟨g, w, rfl, hR⟩ := hy.1 hj
          have hsync := hR.2.1
          rw [show x.1.1 - 1 = x.1.1 - 4 + 3 by omega] at hsync
          exact ⟨(g, w), ⟨rfl, hsync⟩, hR⟩
        refine HRuns.bindC ((HRuns.levelChals resp x.2.1 x.2.2 (embedF gw.1) gw.2 (x.1.1 - 4)).frame _)
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
          obtain ⟨rfl, st, ⟨⟨⟨⟨x₁, ⟨x₂, ⟨⟨⟨hf, hx₂, hsync₁⟩, hR⟩, hcm⟩, hx₁, hv₁, hm₁⟩, hst, hv₂, hm₂⟩, hq⟩, hm₃⟩, hbb⟩,
            hy⟩ := hy
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
            (hlan _ (by omega) (by omega)) (by rw [hcols _ (by omega), hI.sd]; omega) rfl rfl rfl rfl rfl rfl hsync₁
            (commitP_rows hcm) hv₁ hm₁ hv₂ hm₂ ?_ hm₃
          rw [hrp, query_pos]
          rfl
      · rw [loop_stepL P (cfOf resp L) sch _ hj1]
        have hjR : j + 1 = sch.levels.size - 1 := by omega
        generalize hx : loop P (cfOf resp L) sch (x₀Z P (cfOf resp L) sch T cm0.cap ⟨tPad, rep, zl'⟩ t) j = x
          at hs hI ⊢
        have h4 : 4 ≤ x.1.1 := by rw [hI.sd]; omega
        refine HRuns.preX (J := fun y => ∃ gw : Array F128 × Array F256,
          (y = (embedF gw.1, gw.2) ∧ Sync x.2.1.s x.2.1.q (embedF gw.1) gw.2 (x.1.1 - 4 + 3)) ∧
            MidR sch.n b ((preZ P (cfOf resp L) sch T cm0.cap ⟨tPad, rep, zl'⟩ t).1.1.toList.map gOf) x s.2.1
              (sch.levels.getD (j + 1) default) gw.1 gw.2) (HRuns.exPre fun gw => ?_) ?_
        swap
        · intro _ _ _ y hy
          obtain ⟨g, w, rfl, hR⟩ := hy.1 hj
          have hsync := hR.2.1
          rw [show x.1.1 - 1 = x.1.1 - 4 + 3 by omega] at hsync
          exact ⟨(g, w), ⟨rfl, hsync⟩, hR⟩
        refine HRuns.bindDC ((HRuns.levelChals resp x.2.1 x.2.2 (embedF gw.1) gw.2 (x.1.1 - 4)).frame _)
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
          obtain ⟨rfl, ⟨⟨⟨hf, hy₁, hsync₁⟩, hR⟩, hq⟩, -⟩ := hy
          intro s' e
          obtain rfl := ForInStep.yield.inj e
          have hfin := (hX _ rfl).2.2 hjR
          refine ⟨fun h => absurd h (by omega), fun _ => ?_⟩
          subst hf hy₁ hq
          have hrows := hR.1
          rw [hjR] at hrows
          have hfr := congrArg FinalProof.rows hfin
          unfold FinalZ
          refine goal_lastZ hx.symm hjR hL hk0 hsd hrows hR.2.1 hR.2.2.1 hR.2.2.2.1 hR.2.2.2.2 (hseam _ (by omega))
            (hlan _ (by omega) (by omega)) (by rw [hcols _ (by omega), hI.sd]; omega) hbasis rfl rfl hsync₁
            (by rw [← hfin]; rfl) ?_
          rw [← hfr, query_pos]
          rfl
  subst hP
  exact ⟨(key.1 ⟨Pre.refl _, Pre.refl _, Pre.refl _, Pre.refl _, Pre.refl _⟩ ⟨rfl, rfl, rfl, rfl, rfl⟩ rfl rfl).2,
    key.2 ⟨Pre.refl _, Pre.refl _, Pre.refl _, Pre.refl _, Pre.refl _⟩ ⟨rfl, rfl, rfl, rfl, rfl⟩ rfl rfl rfl⟩

end FlockComplete.L0
