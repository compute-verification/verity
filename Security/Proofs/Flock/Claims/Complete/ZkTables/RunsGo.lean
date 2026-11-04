import Proofs.Flock.Claims.Complete.ZkTables.ReplayGo
import Proofs.Flock.Claims.Complete.ZkTables.RepRuns
import Proofs.Flock.Claims.Complete.ZkTables.LaneRows
import Proofs.Flock.Claims.Complete.ZkTables.Parts
import Proofs.Flock.Claims.Complete.ZkTables.Trace
import Proofs.Flock.Claims.Complete.Zk.ConsCoins
import Proofs.Flock.Claims.Complete.Zk.ConsRingSwitch
import Proofs.Flock.Claims.Complete.Zk.InnerGo
import Proofs.Flock.Claims.Complete.Zk.Lanes
import Proofs.Flock.Claims.Complete.Replay.LincheckAlgebra

/-!
# The masked prover's rep returns

`RunsRest` holds: given level 0's `ProverRuns`, `Lanes` and `Verifier`, a masked rep returns from any sends at which
its coins are good, and records `repRoundsT` rounds (`runsRest`).

* `TableT.model`: a table and its trace (`TableT`) are the statement the soundness proof plays, satisfied by the
  witness's bits, and the trace is R1CS-valid.
* `rhoInd_of`, `beta_ne`: the good coins (`CoinsT`) give level 0's independent pair coins and the inner `β ≠ 0`.
* `replay_holds`: the replay's rows and the two ring-switch rows vanish at the honest pads.
* `replay_ok`: the prover's own replay of its rounds returns.
* `repPZk_of`: a masked rep returns where its stages do (`repPZk_split`'s converse), its rank check included.
* `lcRounds_points`: the rounds the rank check reads give the rep's claim points at the coins (`MaskRank.claimsAt`).
* `repT_go`: a masked rep returns at good coins where its rank check passes (`MaskRankT`).
-/

namespace FlockComplete.ZkTables

open Flock Transcript FlockComplete FlockDeploy JsonRT FlockComplete.Replay FlockSoundness.Refine

local notation "𝔽" => FlockLevel3.GF128

/-- A `--zk` schedule of a table in scope folds at most its table's variables at level 0. -/
theorem zkSchedule_k0n {m : ℕ} (h1 : 25 ≤ m) (h2 : m ≤ 35) {v : Schedule × ℕ} (h : Zk.schedule m = .ok v) :
    v.1.k0 ≤ v.1.n := by
  have key : ∀ k : Fin 11, (match Zk.schedule (k.val + 25) with
      | .ok v => decide (v.1.k0 ≤ v.1.n)
      | .error _ => false) = true := by
    decide +kernel
  have := key ⟨m - 25, by omega⟩
  simp only [show m - 25 + 25 = m by omega, h, decide_eq_true_eq] at this
  exact this

/-- **A table and its trace, as the soundness proof's statement**: the table is well formed, its matrices are
`stmtOf`'s, the witness's bits satisfy it, its trace is theirs and R1CS-valid, and the fold folds each of its
ranges. -/
theorem TableT.model {st : Flock.Stmt} {tr : Trace} (hT : TableT st tr) :
    ∃ (w : Claims.Flock.Wit) (hwf : StmtWF st), RegionsWF st ∧
      (∀ a b : Fin (2 ^ st.c.kLog), FlockLevel3.placedA st a b = matA st a.val b.val ∧
        FlockLevel3.placedB st a b = matB st a.val b.val) ∧
      (stmtOf st hwf (regionsOf st)).Satisfies (Readable.zOf 1 w 0 st.m) ∧ tr = traceOf st w.hid.toArray 0 ∧
      (∀ x < 2 ^ st.m, (tr.az[x]! && tr.bz[x]!) = tr.z[x]!) ∧ ∀ r ∈ st.c.ranges.toList, SlotOk st.c r := by
  obtain ⟨-, -, -, -, p, w, hpe, hR, htr, ⟨hwf, hrw, hmat⟩, hslot⟩ := hT
  obtain ⟨hf, hb⟩ := holds_only hpe hR
  have h0 : 0 < p.sts.size := by simp [hpe]
  obtain ⟨hf', hsat⟩ := (Readable.relS_relation p w).1 hR 0 h0
  have e0 : p.sts[0]! = st := by simp [hpe]
  have e1 : p.sts.size = 1 := by simp [hpe]
  revert hsat hf'
  rw [e0, e1]
  intro hf' hsat
  obtain ⟨h1, h2, h3⟩ := hsat
  have hA : ((Readable.Tab.ofStmt st).toModel hf'.2.2).A₀ = (stmtOf st hwf (regionsOf st)).A₀ :=
    funext fun a => funext fun b => (hmat a b).1
  have hB : ((Readable.Tab.ofStmt st).toModel hf'.2.2).B₀ = (stmtOf st hwf (regionsOf st)).B₀ :=
    funext fun a => funext fun b => (hmat a b).2
  refine ⟨w, hwf, hrw, hmat, ⟨fun o j => ?_, h2, h3⟩, htr, fun x hx => ?_, hslot⟩
  · have := h1 o j
    rw [hA, hB] at this
    exact this
  · subst htr
    exact traceOf_r1cs hf hb hx

/-- The trace's bits are the witness's, and its `Az`, `Bz` are the statement's. -/
theorem trace_bits (st : Flock.Stmt) (hwf : StmtWF st)
    (hmat : ∀ a b : Fin (2 ^ st.c.kLog), FlockLevel3.placedA st a b = matA st a.val b.val ∧
      FlockLevel3.placedB st a b = matB st a.val b.val) (w : Claims.Flock.Wit) :
    (∀ a : Fin (2 ^ (stmtOf st hwf (regionsOf st)).m),
      Zc.bitF (traceOf st w.hid.toArray 0).z a.val = ((Readable.zOf 1 w 0 st.m a).val : 𝔽)) ∧
    (∀ a : Fin (2 ^ (stmtOf st hwf (regionsOf st)).m),
      Zc.bitF (traceOf st w.hid.toArray 0).az a.val =
        ((FlockSoundness.aVec (stmtOf st hwf (regionsOf st)) (Readable.zOf 1 w 0 st.m) a).val : 𝔽)) ∧
    ∀ a : Fin (2 ^ (stmtOf st hwf (regionsOf st)).m),
      Zc.bitF (traceOf st w.hid.toArray 0).bz a.val =
        ((FlockSoundness.bVec (stmtOf st hwf (regionsOf st)) (Readable.zOf 1 w 0 st.m) a).val : 𝔽) := by
  refine ⟨fun a => ?_, fun a => ?_, fun a => ?_⟩
  · unfold Zc.bitF
    rw [traceOf_z st w.hid a.isLt, Lc.bitF_toZ]
    show _ = ((Readable.toZ (w.hid.getD (a.val * 1 + 0) false)).val : 𝔽)
    rw [Nat.mul_one, Nat.add_zero]
  · unfold Zc.bitF
    rw [Lc.bitF_toZ]
    refine congrArg (fun x : ZMod 2 => (x.val : 𝔽))
      (toZ_mulBlocks st hwf w (stmtOf st hwf (regionsOf st)).A₀ (entryA st) (fun r j => ?_) a)
    show entryA st r.val j.val = Readable.ofZ (matA st r.val j.val)
    rw [entryA_eq st r.isLt j.isLt, ← (hmat r j).1]
    simp only [Readable.Tab.a, Readable.Tab.ofStmt, r.isLt, j.isLt, and_self, ↓reduceDIte]
  · unfold Zc.bitF
    rw [Lc.bitF_toZ]
    refine congrArg (fun x : ZMod 2 => (x.val : 𝔽))
      (toZ_mulBlocks st hwf w (stmtOf st hwf (regionsOf st)).B₀ (entryB st) (fun r j => ?_) a)
    show entryB st r.val j.val = Readable.ofZ (matB st r.val j.val)
    rw [entryB_eq st r.isLt j.isLt, ← (hmat r j).2]
    simp only [Readable.Tab.b, Readable.Tab.ofStmt, r.isLt, j.isLt, and_self, ↓reduceDIte]

/-- **The good coins' pair is independent**: level 0's pair coins at sends `a` and `a + 1`, within the coins, whose
cross term is not `0` (`¬ rhoDep`). -/
theorem rhoInd_of {k : ℕ} {r : Fin k → Coin} {k0 L a : ℕ} (ha : L + 2 + k0 = a) (hlt : a + 1 < k)
    (h : ¬ rhoDep r a) :
    Zk.rhoIndependent (Level0.laneRho (fun j => (Claims.coinSeq r j).bytes) k0 L) = true := by
  subst ha
  unfold Zk.rhoIndependent Level0.laneRho coinKOf
  simp only [JsonRT.coinOf_bytes, show L + 3 + k0 = L + 2 + k0 + 1 by omega]
  cases hz : ((Claims.coinSeq r (L + 2 + k0)).at 0 * (Claims.coinSeq r (L + 2 + k0 + 1)).at 1 +
      (Claims.coinSeq r (L + 2 + k0)).at 1 * (Claims.coinSeq r (L + 2 + k0 + 1)).at 0).isZero with
  | false => rfl
  | true => exact absurd ⟨hlt, isZero_eq hz⟩ h

/-- **The good coins' inner `β` is not `0`.** -/
theorem beta_ne {k : ℕ} {r : Fin k → Coin} {a : ℕ} (hlt : a < k) (h : a < k → (Claims.coinSeq r a).at 0 ≠ 0) :
    (coinOf ((fun j => (Claims.coinSeq r j).bytes) a) 0).isZero = false := by
  simp only [JsonRT.coinOf_bytes]
  cases hz : ((Claims.coinSeq r a).at 0).isZero with
  | false => rfl
  | true => exact absurd (isZero_eq hz) (h hlt)

/-- A lift of a verdict that returned. -/
theorem liftV_go {α : Type} {v : V α} {a : α} (hv : v = .ok a) (t : Transcript) (l : Array Claims.Msg) :
    ((StateT.lift (monadLift v : PS α) : TMW PS α).run t).go l = (.ok (a, t), l) := by
  subst hv
  rfl

/-- A step that returns, then a continuation that returns. -/
theorem go_bind_ex {α β : Type} {x : TMW PS α} {f : α → TMW PS β} {t t' : Transcript} {l l' : Array Claims.Msg}
    {a : α} (hx : (x.run t).go l = (.ok (a, t'), l')) (h : ∃ pt l'', ((f a).run t').go l' = (.ok pt, l'')) :
    ∃ pt l'', ((x >>= f).run t).go l = (.ok pt, l'') := by
  rw [bind_go hx]
  exact h

theorem abc_get0 (m kLog : ℕ) (zc : ZcOut) (lc : LcOut) (e : Array Claim) :
    (abcClaims m kLog zc lc ++ e)[0]! = abClaim m kLog zc lc := by
  rw [getElem!_pos _ 0 (by simp [abcClaims])]
  simp [abcClaims, abClaim]

theorem abc_get1 (m kLog : ℕ) (zc : ZcOut) (lc : LcOut) (e : Array Claim) :
    (abcClaims m kLog zc lc ++ e)[1]! = cClaim zc := by
  rw [getElem!_pos _ 1 (by simp [abcClaims]; omega)]
  simp [abcClaims, cClaim]

/-- **The replay's constraints vanish at the honest pads**, at a table and its trace (`TableT`): wherever the replay
ran on rounds that carry the masked prover's zerocheck and lincheck, at outer coins that are not `1` and a fold that
returned, the replay's rows and the two ring-switch rows on the prover's switched vectors hold at any pads with
`h_ab = h_a·h_b` (`replay_consHold_agrees`), the zerocheck's fold closing (`Zc.fold_final`), the lincheck's
(`Lc.lcAlgebra_of`), and each claim met by its switched vector (`claims_switch`). -/
theorem replay_holds {st : Flock.Stmt} {tr : Trace} (hT : TableT st tr) (lay : Zk.PadLayout) (h : Array F128)
    (hsz : h.size = lay.len) (hab : h[lay.hab]! = h[lay.fa]! * h[lay.fb]!) (hn : st.m - 6 ≤ lay.nMlv)
    (hn' : st.c.kLog - 6 ≤ lay.nLc) (zcs : Zc.Challenges) (lcs : Lc.Challenges) (full : Array F128)
    (hc1 : ∀ j < st.m - 13, zcs.c1 j ≠ 1)
    (hfold : st.circuitFold.fold (lcs.ca 0) (skipWeights (Zc.out st.m zcs tr).z)
      ((Zc.out st.m zcs tr).rho.extract 0 (st.c.kLog - 6)) = .ok full)
    (pts : Array (Array F128)) (hpts : ∀ a ∈ pts, a.size = st.mPts) (e : Array Claim)
    (he : st.extraClaims pts = .ok e) (p : Proof) (root : ByteArray)
    (hpz : p.zerocheck = ZcZ.proof lay h st.m zcs tr)
    (hpl : p.lincheck = LcZ.proof lay h st.circuitFold st.m tr (Zc.out st.m zcs tr) full lcs)
    (digest cap padsRoot : ByteArray) (t₀ t : Transcript) (hk₀ : t₀.k = t₀.rounds.size) (ht : t.k = t₀.k)
    (hR : Agrees t.rounds (LcZ.tEnd lay h st.circuitFold st.m tr (Zc.out st.m zcs tr) full lcs
      (ZcZ.tEnd lay h st.m digest cap padsRoot zcs tr t₀)).rounds 0)
    {rp : Zk.Replayed} {t' : Transcript} (hrun : Zk.replay (Setup.ofCircuit st) lay p root t = .ok (rp, t')) :
    ConsHold (rp.cons ++ #[("ring switch ab", rp.abValue.addConst (Zk.claimCheck rp.claims[0]!
        (Rs.proof tr (abcClaims st.m st.c.kLog (Zc.out st.m zcs tr)
          (Lc.out st.circuitFold st.m tr (Zc.out st.m zcs tr) lcs) ++ e))[0]!)),
      ("ring switch c", rp.cValue.addConst (Zk.claimCheck rp.claims[1]!
        (Rs.proof tr (abcClaims st.m st.c.kLog (Zc.out st.m zcs tr)
          (Lc.out st.circuitFold st.m tr (Zc.out st.m zcs tr) lcs) ++ e))[1]!))]) h := by
  have h25 := hT.1
  obtain ⟨w, hwf, hrw, hmat, hsat, htr, hR1, -⟩ := hT.model
  obtain ⟨hz, ha, hb⟩ := trace_bits st hwf hmat w
  rw [← htr] at hz ha hb
  have hsw := claims_switch st hwf hrw (by omega) _ hsat _ hz zcs lcs pts (fun a ha => hpts a (by simpa using ha))
  rw [he] at hsw
  obtain ⟨comb, hcomb, hcsz, hlc⟩ := Lc.lcAlgebra_of (stmtOf st hwf (regionsOf st)) st.circuitFold
    (stmtOf_fold st hwf _) (fun α β lw ρ ts v h => ⟨_, FlockLevel3.folded_ok st α β lw ρ ts v h⟩) st.m rfl _ hsat.2.1
    _ hz ha hb zcs lcs full hfold
  have hcs : (abcClaims st.m st.c.kLog (Zc.out st.m zcs tr)
      (Lc.out st.circuitFold st.m tr (Zc.out st.m zcs tr) lcs) ++ e).size = 2 + e.size := by simp [abcClaims]
  refine replay_consHold_agrees (Setup.ofCircuit st) lay h p root (show 14 ≤ st.m by omega) hrun hsz st.circuitFold
    _ zcs lcs full rfl rfl hpz hpl hn hn' hab digest cap padsRoot t₀ hk₀ ht hR
    (Zc.fold_final (show 14 ≤ st.m by omega) zcs _ hR1 hc1) ⟨comb, hcomb, hcsz, hlc⟩ _ _ ?_ ?_
  · rw [Rs.proof_get 0 (by omega), abc_get0]
    exact hsw _ (abClaim_mem _ _ _ _ _)
  · rw [Rs.proof_get 1 (by omega), abc_get1]
    exact hsw _ (cClaim_mem _ _ _ _ _)

/-- **The prover's own replay returns**, at a table and its trace: on rounds that carry the masked prover's zerocheck
and lincheck from the empty transcript, with a fold that returned (`replayWith_ok` at the deployed hash, the
lincheck's fold from `Lc.lcAlgebra_of`). -/
theorem replay_ok {st : Flock.Stmt} {tr : Trace} (hT : TableT st tr) (lay : Zk.PadLayout) (h : Array F128)
    (zcs : Zc.Challenges) (lcs : Lc.Challenges) (full : Array F128)
    (hfold : st.circuitFold.fold (lcs.ca 0) (skipWeights (Zc.out st.m zcs tr).z)
      ((Zc.out st.m zcs tr).rho.extract 0 (st.c.kLog - 6)) = .ok full)
    (p : Proof) (root : ByteArray) (hpz : p.zerocheck = ZcZ.proof lay h st.m zcs tr)
    (hpl : p.lincheck = LcZ.proof lay h st.circuitFold st.m tr (Zc.out st.m zcs tr) full lcs) (d : ByteArray)
    {R : Array Round}
    (hR : Agrees R (LcZ.tEnd lay h st.circuitFold st.m tr (Zc.out st.m zcs tr) full lcs
      (ZcZ.tEnd lay h st.m (Setup.ofCircuit st).digest (flat p.cap) root zcs tr (Transcript.init #[] d))).rounds 0) :
    ∃ q, (Zk.replay (Setup.ofCircuit st) lay p root).run (Transcript.init R d) = .ok q := by
  have h25 := hT.1
  obtain ⟨w, hwf, -, hmat, hsat, htr, -, -⟩ := hT.model
  obtain ⟨hz, ha, hb⟩ := trace_bits st hwf hmat w
  rw [← htr] at hz ha hb
  obtain ⟨comb, hcomb, hcsz, -⟩ := Lc.lcAlgebra_of (stmtOf st hwf (regionsOf st)) st.circuitFold
    (stmtOf_fold st hwf _) (fun α β lw ρ ts v h => ⟨_, FlockLevel3.folded_ok st α β lw ρ ts v h⟩) st.m rfl _ hsat.2.1
    _ hz ha hb zcs lcs full hfold
  obtain ⟨q, hq⟩ := replayWith_ok With.std (Setup.ofCircuit st) (show 14 ≤ st.m by omega) lay h p root zcs lcs full tr
    (Transcript.init #[] d) rfl hR hpz hpl rfl ⟨comb, hcomb, hcsz⟩
  rw [With.zk_replayWith_std] at hq
  exact ⟨q, hq⟩

/-- **The rank check returns where it passes**, keeping the transcript and the sends. -/
theorem rankCheckP_ok {m kLog : ℕ} {prior : Array (List F128)} {words : List ℕ} (t : Transcript)
    (l : Array Claims.Msg) (h : MaskRank.maskRankOk (prior ++ MaskRank.roundPoints m kLog t.rounds) words = true) :
    ((rankCheckP m kLog prior words : TMW PS Unit).run t).go l = (.ok ((), t), l) := by
  unfold rankCheckP
  simp only [StateT.run_bind, StateT.run_get, pure_bind, h, ↓reduceIte]
  rfl

/-- `claimPoints` reads coin `0` of rounds below `5 + (m − 6) + (k_log − 6)` and round `1`'s first `m − 13`. -/
theorem claimPoints_congr_reads {m kLog : ℕ} {f g : ℕ → ℕ → F128}
    (h0 : ∀ j < 5 + (m - 6) + (kLog - 6), f j 0 = g j 0) (h1 : ∀ e < m - 13, f 1 e = g 1 e) :
    MaskRank.claimPoints m kLog f = MaskRank.claimPoints m kLog g := by
  unfold MaskRank.claimPoints
  have hr : (Array.range (m - 6)).map (fun j => f (3 + j) 0) = (Array.range (m - 6)).map fun j => g (3 + j) 0 :=
    Array.map_congr_left fun j hj => h0 _ (by rw [Array.mem_range] at hj; omega)
  have ht : (Array.range (kLog - 6)).map (fun j => f (5 + (m - 6) + j) 0) =
      (Array.range (kLog - 6)).map fun j => g (5 + (m - 6) + j) 0 :=
    Array.map_congr_left fun j hj => h0 _ (by rw [Array.mem_range] at hj; omega)
  have hc : (Array.range (m - 13)).map (f 1) = (Array.range (m - 13)).map (g 1) :=
    Array.map_congr_left fun e he => h1 e (by rw [Array.mem_range] at he; exact he)
  simp only [hr, ht, hc]

/-- **The rank check reads the rep's claim points at the coins**: the rounds the masked zerocheck and lincheck leave,
run from the empty transcript and `l` sends, are those sends' (`zerocheckPZk_rec`, `lincheckPZk_rec`), round `1`
holding the zerocheck's `m − 13` outer coins, so the points read off them are the rep's from coin `|l|`. -/
theorem lcRounds_points {k : ℕ} (r : Fin k → Coin) (st : Flock.Stmt) (hm : 14 ≤ st.m) (lay : Zk.PadLayout)
    (hp : Array F128) (digest cap root : ByteArray) (tr : Trace) (full : Array F128) (d : ByteArray)
    (l : Array Claims.Msg)
    (hfold : st.circuitFold.fold (coinOf ((Claims.coinSeq r (l.size + 3 + (st.m - 6))).bytes) 0)
      (skipWeights (zcR (fun j => (Claims.coinSeq r j).bytes) (Setup.ofCircuit st) tr l.size).z)
      ((zcR (fun j => (Claims.coinSeq r j).bytes) (Setup.ofCircuit st) tr l.size).rho.extract 0
        (st.circuitFold.kLog - 6)) = .ok full) :
    MaskRank.roundPoints st.m st.c.kLog (LcZ.tEnd lay hp st.circuitFold st.m tr
      (zcR (fun j => (Claims.coinSeq r j).bytes) (Setup.ofCircuit st) tr l.size) full
      (lcCoinsR (fun j => (Claims.coinSeq r j).bytes) (Setup.ofCircuit st) l.size)
      (ZcZ.tEnd lay hp st.m digest cap root (Zc.coinsAt (fun j => (Claims.coinSeq r j).bytes) l.size) tr
        (Transcript.init #[] d))).rounds = MaskRank.claimsAt st r l.size := by
  set resp : ℕ → Claims.Msg := fun j => (Claims.coinSeq r j).bytes with hresp
  obtain ⟨l₁, hl₁, hz⟩ := zerocheckPZk_go resp hm digest cap root lay hp tr (Transcript.init #[] d) l
  obtain ⟨l₂, -, hl⟩ := lincheckPZk_go resp st.circuitFold st.m lay hp tr (Zc.out st.m (Zc.coinsAt resp l.size) tr)
    full (ZcZ.tEnd lay hp st.m digest cap root (Zc.coinsAt resp l.size) tr (Transcript.init #[] d)) l₁
    (by rw [hl₁]; exact hfold)
  rw [hl₁] at hl
  obtain ⟨rs₁, bs₁, hrs₁, hbs₁, hrec₁⟩ := JsonRT.Rec.go (JsonRT.zerocheckPZk_rec resp _ _ _ _ _ _ _) hz
  obtain ⟨rs₂, bs₂, hrs₂, -, hrec₂⟩ := JsonRT.Rec.go (JsonRT.lincheckPZk_rec resp _ _ _ _ _ _) hl
  have hl1 : l.size + rs₁.length = l.size + 3 + (st.m - 6) := by
    have h := congrArg List.length hbs₁
    simp only [Array.length_toList, List.length_append] at h
    rw [Recs.length_eq hrec₁, ← h, hl₁]
  have hrec := Recs.append hrec₁ (by rw [hl1, ← hl₁]; exact hrec₂)
  set T := (LcZ.tEnd lay hp st.circuitFold st.m tr (zcR resp (Setup.ofCircuit st) tr l.size) full
    (lcCoinsR resp (Setup.ofCircuit st) l.size) (ZcZ.tEnd lay hp st.m digest cap root (Zc.coinsAt resp l.size) tr
      (Transcript.init #[] d))).rounds with hT
  have hTl : T.toList = rs₁ ++ rs₂ := by
    rw [show T.toList = _ from hrs₂, hrs₁]
    simp [Transcript.init]
  have hround : ∀ j (hj : j < T.size), ∃ b, RoundIs resp (l.size + j) b T[j]! := by
    intro j hj
    have hj' : j < (rs₁ ++ rs₂).length := by rw [← hTl]; simpa using hj
    obtain ⟨b, hb⟩ := Recs.getElem hrec j hj'
    refine ⟨b, ?_⟩
    rw [getElem!_pos T j hj]
    simpa only [← hTl, Array.getElem_toList] using hb
  have key : ∀ j (hj : j < T.size) a, a < T[j]!.n →
      (T[j]!.coins).getD a 0 = (Claims.coinSeq r (l.size + j)).at a := by
    intro j hj a ha
    obtain ⟨b, c, -, -, -, hc⟩ := hround j hj
    rw [hc, Array.getD_eq_getD_getElem?, Array.getElem?_map, Array.getElem?_range]
    simpa [ha] using coinOf_bytes (Claims.coinSeq r (l.size + j)) a
  have hsz : T.size = 3 + (st.m - 6) + 3 + (st.c.kLog - 6) := by
    rw [hT, LcZ.tEnd_size, ZcZ.tEnd_size]
    simp [Transcript.init]
    rfl
  have h1 : T[1]!.n = st.m - 13 := by
    have hp1 : Prefix ((ZcZ.t₂ digest cap root (Zc.coinsAt resp l.size) (Transcript.init #[] d)).rounds.push
        (sent (ZcZ.t₂ digest cap root (Zc.coinsAt resp l.size) (Transcript.init #[] d)) 2 (st.m - 13)
          (Zc.coinsAt resp l.size).c1)) T :=
      ((Prefix.push _ _).trans (ZcZ.prefix_loop (i := 0) _ (Nat.zero_le _))).trans (LcZ.prefix_start _)
    obtain ⟨C, hC⟩ := hp1
    have hs : (ZcZ.t₂ digest cap root (Zc.coinsAt resp l.size) (Transcript.init #[] d)).rounds.size = 1 := by
      simp [ZcZ.t₂, ZcZ.t₁, afterSq, Transcript.init]
    rw [hC, getElem!_pos _ 1 (by simp [hs]; omega), Array.getElem_append_left (by simp [hs])]
    simp [Array.getElem_push, hs, sent]
  unfold MaskRank.roundPoints MaskRank.claimsAt
  refine claimPoints_congr_reads (fun j hj => ?_) (fun a ha => ?_)
  · have hj' : j < T.size := by omega
    obtain ⟨b, c, -, -, hn, -⟩ := hround j hj'
    exact key j hj' 0 hn
  · exact key 1 (by omega) a (by omega)

/-- **A masked rep returns where its stages do** (`repPZk_split`'s converse): where the lincheck's fold returns at
the zerocheck's point, its rank check passes on the rounds so far, and from any `|l| + piopSends` bodies at `tPiop`
its masked Ligerito, the prover's own replay of its rounds, level 0's lane constraints and its inner proof on every
constraint return. -/
theorem repPZk_of (resp : ℕ → Claims.Msg) (hsh : ByteArray → ByteArray) (st : Setup) (sch : Schedule)
    (tPad rep : ℕ) (dom : ByteArray) (tr : Trace) (tb : Array F128) (cm0 : Commit) (extras : Array Claim)
    (pads : PadsZk) (ic : InnerCommit) (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray)
    (nonce : ℕ → ℕ → UInt64) (prior : Array (List F128)) (words : List ℕ) (hm : 14 ≤ st.m) (t : Transcript)
    (l : Array Claims.Msg) (full : Array F128)
    (hfold : st.fold.fold (coinOf (resp (l.size + 3 + (st.m - 6))) 0) (skipWeights (zcR resp st tr l.size).z)
      ((zcR resp st tr l.size).rho.extract 0 (st.fold.kLog - 6)) = .ok full)
    (hrank : MaskRank.maskRankOk (prior ++ MaskRank.roundPoints st.m st.kLog
      (LcZ.tEnd (Zk.PadLayout.new st.m st.kLog sch.k0) pads.h st.fold st.m tr (zcR resp st tr l.size) full
        (lcCoinsR resp st l.size) (ZcZ.tEnd (Zk.PadLayout.new st.m st.kLog sch.k0) pads.h st.m st.digest (flat cm0.cap)
          ic.root (Zc.coinsAt resp l.size) tr t)).rounds) words = true)
    (h : ∀ lL : Array Claims.Msg, lL.size = l.size + piopSends st extras →
      ∃ (lg : LigeritoProof) (zl : Zk.Level0Msgs) (rec : Zk.L0Rec) (tI : Transcript) (lI : Array Claims.Msg)
        (rp : Zk.Replayed) (x : Transcript) (lgc : Array (String × Zk.Lin)),
        ((ligeritoPZk (asksS resp) (Hashes.ofSha512 hsh) st.merkle sch
          (Rs.T tr (claimsR resp st tr l.size extras) (rsCoinsR resp st l.size extras)) cm0 tb
          (basisVals (Rs.sws (claimsR resp st tr l.size extras) (rsCoinsR resp st l.size extras)) (st.m - 7))
          (Zk.PadLayout.new st.m st.kLog sch.k0) pads.h rep tPad extra lanePad salt nonce).run
            (tPiop resp st tr l.size (Zk.PadLayout.new st.m st.kLog sch.k0) pads.h st.digest (flat cm0.cap) ic.root
              full extras t)).go lL = (.ok ((lg, zl, rec), tI), lI) ∧
        (Zk.replay st (Zk.PadLayout.new st.m st.kLog sch.k0)
          (proofR resp st tr l.size sch (Zk.PadLayout.new st.m st.kLog sch.k0) pads.h cm0 full extras lg) ic.root).run
            (Transcript.init tI.rounds dom) = .ok (rp, x) ∧
        Zk.ligeritoConstraints (Zk.PadLayout.new st.m st.kLog sch.k0) rec lg zl = .ok lgc ∧
        ∃ (inner : Zk.InnerProof) (tE : Transcript) (lE : Array Claims.Msg),
          ((innerP (asksS resp) (Hashes.ofSha512 hsh) (Zk.PadLayout.new st.m st.kLog sch.k0) pads ic
            (ZcZ.proof (Zk.PadLayout.new st.m st.kLog sch.k0) pads.h st.m (Zc.coinsAt resp l.size) tr).finalC
            (rp.cons ++ #[("ring switch ab", rp.abValue.addConst (Zk.claimCheck rp.claims[0]!
                (Rs.proof tr (claimsR resp st tr l.size extras))[0]!)),
              ("ring switch c", rp.cValue.addConst (Zk.claimCheck rp.claims[1]!
                (Rs.proof tr (claimsR resp st tr l.size extras))[1]!))] ++ lgc)).run tI).go lI =
            (.ok (inner, tE), lE)) :
    ∃ pt l', ((repPZk (asksS resp) (Hashes.ofSha512 hsh) st sch tPad rep dom tr tb cm0 extras pads ic extra lanePad
      salt nonce prior words).run t).go l = (.ok pt, l') := by
  obtain ⟨l₁, hl₁, hz⟩ := zerocheckPZk_go resp hm st.digest (flat cm0.cap) ic.root
    (Zk.PadLayout.new st.m st.kLog sch.k0) pads.h tr t l
  obtain ⟨l₂, hl₂, hl⟩ := lincheckPZk_go resp st.fold st.m (Zk.PadLayout.new st.m st.kLog sch.k0) pads.h tr
    (Zc.out st.m (Zc.coinsAt resp l.size) tr) full _ l₁ (by rw [hl₁]; exact hfold)
  rw [hl₁] at hl hl₂
  have hcs : (claimsR resp st tr l.size extras).size = 2 + extras.size := by simp [claimsR, abcClaims]
  obtain ⟨l₃, hl₃, hrs⟩ := ringSwitchP_go resp tr (claimsR resp st tr l.size extras) (by omega) _ l₂
  rw [hl₂, hcs] at hrs
  have hL : l₃.size = l.size + piopSends st extras := by
    rw [hl₃, hl₂, hcs]
    unfold piopSends
    omega
  obtain ⟨lg, zl, rec, tI, lI, rp, x, lgc, hlg, hre, hlgc, inner, tE, lE, hinn⟩ := h l₃ hL
  unfold repPZk
  dsimp only
  refine go_bind_ex hz ?_
  dsimp only
  refine go_bind_ex hl ?_
  dsimp only
  refine go_bind_ex (rankCheckP_ok _ l₂ hrank) ?_
  refine go_bind_ex hrs ?_
  dsimp only
  refine go_bind_ex hlg ?_
  dsimp only
  refine go_bind_ex (get_go _ _) ?_
  refine go_bind_ex (liftV_go hre _ _) ?_
  dsimp only
  refine go_bind_ex (liftV_go hlgc _ _) ?_
  refine go_bind_ex hinn ?_
  exact ⟨_, _, pure_go _ _ _⟩

variable {H : Claims.Msg → Claims.Msg}

/-- **A masked rep returns at good coins**, given level 0's premises: the zerocheck, lincheck and ring switch run in
closed form (`repPZk_of`), the lincheck's fold returns (`JsonRT.fold_ok`), the masked Ligerito returns at the
independent pair coins (`Level0.ProverRuns`, `rhoInd_of`), the prover's replay returns (`replay_ok`), level 0's lane
constraints are built from its record (`Level0.Lanes`, `ligeritoConstraints_ok`), every constraint vanishes at the
pads (`replay_holds`, `lanes_hold`), and the inner proof's `β` is not `0` (`beta_ne`), at the send the Ligerito's
round count puts it (`Level0.Verifier`, `zkLigeritoWith_pos`); its rank check passes where `MaskRankT` holds, the
rounds it reads giving the rep's claim points at the coins (`lcRounds_points`). -/
theorem repT_go (h0 : Level0.ProverRuns) (h2 : Level0.Lanes) (h3 : Level0.Verifier) (h64 : Claims.Out64 H)
    (st : Flock.Stmt) (tr : Trace) (hT : TableT st tr) (ω : TapeZk) (i : ℕ) (d : ByteArray) {k : ℕ}
    (r : Fin k → Coin) (v : Schedule × ℕ) (cm : Commit) (pts : Array (Array F128)) (e : Array Claim) (rep : ℕ)
    (ic : InnerCommit) (hin : RepIn H st tr ω i v cm pts e rep ic) (hrep : rep < 2) (l : Array Claims.Msg)
    (prior : Array (List F128)) (hco : CoinsT st v e r l.size) (hmr : MaskRankT st r l.size prior) :
    ∃ pt l', (repT H r st tr ω i d v cm e rep ic prior (FlockSoundness.Discharge.Composed.maskWordsOf st)).go l =
      (.ok pt, l') := by
  have h25 := hT.1
  have h35 := hT.2.1
  have hmerk : (Setup.ofCircuit st).merkle = MerkleScheme.hm96Sha512 := hT.2.2.1
  obtain ⟨w, hwf, -, -, -, -, -, hslot⟩ := hT.model
  obtain ⟨hfit, hc1, hρ, hβ⟩ := hco
  have hk6 := hwf.hk6
  have hkm := hwf.hk
  have hm14 : 14 ≤ (Setup.ofCircuit st).m := by show 14 ≤ st.m; omega
  have hn : v.1.n = st.m - 7 := zkSchedule_n hin.schedule
  obtain ⟨hk0, hlv⟩ := zkSchedule_shape hin.schedule
  have hkn : v.1.k0 ≤ v.1.n := zkSchedule_k0n h25 h35 hin.schedule
  have hfit' : l.size + piopT st e + (v.1.k0 + 7 + 11 * (v.1.levels.size - 1)) + 4 < k := by
    unfold repRoundsT at hfit
    unfold piopT
    omega
  set resp : ℕ → Claims.Msg := fun j => (Claims.coinSeq r j).bytes with hresp
  set lay := Zk.PadLayout.new (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog v.1.k0 with hlay
  set pads := padsOf ω i rep lay with hpads
  obtain ⟨full, hfold⟩ := JsonRT.fold_ok st hk6 hslot (coinOf (resp (l.size + 3 + (st.m - 6))) 0)
    (skipWeights_size _) (ρ := (zcR resp (Setup.ofCircuit st) tr l.size).rho.extract 0 (st.c.kLog - 6))
    (by simp only [zcR, Zc.out, Array.size_extract, pushes_size, Setup.ofCircuit]; omega)
  unfold repT
  refine repPZk_of resp (Claims.hOf H) (Setup.ofCircuit st) v.1 v.2 rep d tr _ cm e pads ic (ω.extra i) (ω.lanePad i)
    (ω.salt i rep) (ω.nonce i rep) _ _ hm14 _ l full hfold ?_ fun lL hlL => ?_
  · unfold MaskRankT at hmr
    rw [← lcRounds_points r st (by omega) lay pads.h (Setup.ofCircuit st).digest (flat cm.cap) ic.root tr full d l
      hfold] at hmr
    exact hmr
  have hpiop : piopSends (Setup.ofCircuit st) e = piopT st e := by
    show 3 + (st.m - 6) + 3 + (st.c.kLog - 6) + (2 + e.size + 1) = (st.m + st.c.kLog - 6) + (2 + e.size + 1)
    omega
  have hrho : Zk.rhoIndependent (Level0.laneRho resp v.1.k0 lL.size) = true :=
    rhoInd_of (show lL.size + 2 + v.1.k0 = l.size + piopT st e + (v.1.k0 + 2) by omega) (by omega) hρ
  have htb : (packed tr (Setup.ofCircuit st).m).size = 2 ^ v.1.n := by simp [packed, hn]; rfl
  have hb : (basisVals (Rs.sws (claimsR resp (Setup.ofCircuit st) tr l.size e)
      (rsCoinsR resp (Setup.ofCircuit st) l.size e)) ((Setup.ofCircuit st).m - 7)).size = 2 ^ v.1.n := by
    simp [basisVals, hn]; rfl
  obtain ⟨⟨⟨lg, zl, rec⟩, tI⟩, lI, hlig⟩ := h0 resp (Claims.hOf H) (Setup.ofCircuit st).m v
    (Rs.T tr (claimsR resp (Setup.ofCircuit st) tr l.size e) (rsCoinsR resp (Setup.ofCircuit st) l.size e)) cm
    (packed tr (Setup.ofCircuit st).m)
    (basisVals (Rs.sws (claimsR resp (Setup.ofCircuit st) tr l.size e) (rsCoinsR resp (Setup.ofCircuit st) l.size e))
      ((Setup.ofCircuit st).m - 7))
    lay pads.h rep (ω.extra i) (ω.lanePad i) (ω.salt i rep) (ω.nonce i rep)
    (tPiop resp (Setup.ofCircuit st) tr l.size lay pads.h (Setup.ofCircuit st).digest (flat cm.cap) ic.root full e
      (Transcript.init #[] d)) lL (JsonRT.hOf_size h64) h25 h35 hin.schedule hrho
  have hLF := h2 resp (Claims.hOf H) _ v _ cm _ _ lay pads.h rep (ω.extra i) (ω.lanePad i) (ω.salt i rep)
    (ω.nonce i rep) _ tI lL lI lg zl rec h25 h35 hin.schedule htb hb hlig
  have hx : ∀ c ∈ claimsR resp (Setup.ofCircuit st) tr l.size e, c.x.size = st.m - 6 :=
    claims_xsize' st hkm (by omega) tr (Zc.coinsAt resp l.size) (lcCoinsR resp (Setup.ofCircuit st) l.size) pts e
      hin.extra
  have hcs : (claimsR resp (Setup.ofCircuit st) tr l.size e).size = 2 + e.size := by simp [claimsR, abcClaims]
  have htk : (tPiop resp (Setup.ofCircuit st) tr l.size lay pads.h (Setup.ofCircuit st).digest (flat cm.cap) ic.root
      full e (Transcript.init #[] d)).k = (tPiop resp (Setup.ofCircuit st) tr l.size lay pads.h
      (Setup.ofCircuit st).digest (flat cm.cap) ic.root full e (Transcript.init #[] d)).rounds.size := by
    simp only [tPiop, Rs.tEnd_k, Rs.tEnd_size, LcZ.tEnd_k, LcZ.tEnd_size, ZcZ.tEnd_k, ZcZ.tEnd_size]
    rfl
  have hpk : (tPiop resp (Setup.ofCircuit st) tr l.size lay pads.h (Setup.ofCircuit st).digest (flat cm.cap) ic.root
      full e (Transcript.init #[] d)).k = 3 + (st.m - 6) + 3 + (st.c.kLog - 6) + (2 + e.size) + 1 := by
    simp only [tPiop, Rs.tEnd_k, LcZ.tEnd_k, ZcZ.tEnd_k, hcs]
    show 0 + 3 + (st.m - 6) + 3 + (st.c.kLog - 6) + (2 + e.size) + 1 = _
    omega
  have hLk : lL.size = l.size + (tPiop resp (Setup.ofCircuit st) tr l.size lay pads.h (Setup.ofCircuit st).digest
      (flat cm.cap) ic.root full e (Transcript.init #[] d)).k := by
    rw [hlL, hpk]
    have : piopSends (Setup.ofCircuit st) e = 3 + (st.m - 6) + 3 + (st.c.kLog - 6) + (2 + e.size + 1) := rfl
    omega
  obtain ⟨hLgV, htIk⟩ := h3 resp (Claims.hOf H) (Setup.ofCircuit st).m v
    (Rs.T tr (claimsR resp (Setup.ofCircuit st) tr l.size e) (rsCoinsR resp (Setup.ofCircuit st) l.size e)) cm
    (packed tr (Setup.ofCircuit st).m)
    (basisVals (Rs.sws (claimsR resp (Setup.ofCircuit st) tr l.size e) (rsCoinsR resp (Setup.ofCircuit st) l.size e))
      ((Setup.ofCircuit st).m - 7))
    lay pads.h rep (ω.extra i) (ω.lanePad i) (ω.salt i 0 0) (ω.salt i rep) (ω.nonce i rep)
    (basis (Rs.sws (claimsR resp (Setup.ofCircuit st) tr l.size e) (rsCoinsR resp (Setup.ofCircuit st) l.size e)))
    l.size _ tI lL lI lg zl rec tI.rounds (JsonRT.hOf_size h64) h25 h35 hin.schedule hrep htb hb
    (Lg.rsSum_eq tr _ _ _ hx) (by rw [← hmerk]; exact hin.commit)
    (fun q e' y hqe hy => by rw [hn] at hqe ⊢; exact Lg.basis_sws _ _ _ hx q e' y hqe hy)
    (fun q e' => basis_size _ q e') htk hLk hlig (Agrees.self _ _)
  obtain ⟨-, -, -, -, -, hend, -⟩ := FlockComplete.zkLigeritoWith_pos _ _ v.1 _ _ _ _ _ (by omega) hk0 hlv
    (zkSchedule_q hin.schedule) hLgV
  have htI : tI.k = (tPiop resp (Setup.ofCircuit st) tr l.size lay pads.h (Setup.ofCircuit st).digest (flat cm.cap)
      ic.root full e (Transcript.init #[] d)).k + (v.1.k0 + 7 + 11 * (v.1.levels.size - 1)) :=
    congrArg Transcript.k hend
  obtain ⟨rs, bs, hrs, hbs, hRecs⟩ :=
    JsonRT.Rec.go (JsonRT.ligeritoPZk_rec resp (Claims.hOf H) _ _ _ _ _ _ _ _ _ _ _ _ _ _) hlig
  have hrsl := congrArg List.length hrs
  have hbsl := congrArg List.length hbs
  simp only [Array.length_toList, List.length_append] at hrsl hbsl
  have hlen := JsonRT.Recs.length_eq hRecs
  have hlI : lI.size + 2 = l.size + piopT st e + (v.1.k0 + 7 + 11 * (v.1.levels.size - 1)) + 2 := by omega
  have hβ' := beta_ne (r := r) (a := lI.size + 2) (by omega) (by rw [hlI]; exact hβ)
  have hpre : Prefix (LcZ.tEnd lay pads.h st.circuitFold st.m tr (Zc.out st.m (Zc.coinsAt resp l.size) tr) full
      (lcCoinsR resp (Setup.ofCircuit st) l.size) (ZcZ.tEnd lay pads.h st.m (Setup.ofCircuit st).digest (flat cm.cap)
        ic.root (Zc.coinsAt resp l.size) tr (Transcript.init #[] d))).rounds tI.rounds :=
    (Rs.prefix_loop (i := 0) _ (Nat.zero_le _)).trans
      (rec_pre (JsonRT.ligeritoPZk_rec resp (Claims.hOf H) _ _ _ _ _ _ _ _ _ _ _ _ _ _) _ _ _ _ hlig)
  have hR := (Agrees.self tI.rounds 0).of_prefix hpre
  obtain ⟨⟨rp, x⟩, hre⟩ := replay_ok hT lay pads.h (Zc.coinsAt resp l.size) (lcCoinsR resp (Setup.ofCircuit st) l.size)
    full hfold (proofR resp (Setup.ofCircuit st) tr l.size v.1 lay pads.h cm full e lg) ic.root rfl rfl d hR
  obtain ⟨lgc, hlgc⟩ := ligeritoConstraints_ok (lay := lay) (rec := rec) (lg := lg) (msgs := zl)
    (by rw [hLF.record]; rfl)
    (by rw [hLF.record]; simp only [Level0.laneCh, List.size_toArray, List.length_map, List.length_range]; rfl)
    hLF.msgs_size
  have hc1' : ∀ j < st.m - 13, (Zc.coinsAt resp l.size).c1 j ≠ 1 := fun j hj h =>
    hc1 j hj ((JsonRT.coinOf_bytes (Claims.coinSeq r (l.size + 1)) j).symm.trans h)
  have hcons := ConsHold.append
    (replay_holds hT lay pads.h (padsOf_size ..) (padsOf_hab ..) le_rfl le_rfl (Zc.coinsAt resp l.size)
      (lcCoinsR resp (Setup.ofCircuit st) l.size) full hc1' hfold pts hin.points e hin.extra
      (proofR resp (Setup.ofCircuit st) tr l.size v.1 lay pads.h cm full e lg) ic.root rfl rfl
      (Setup.ofCircuit st).digest (flat cm.cap) ic.root (Transcript.init #[] d) (Transcript.init tI.rounds d) rfl rfl
      hR hre)
    (lanes_hold hLF rfl rfl htb hb hkn (by omega) (Lg.rsSum_eq tr _ _ _ hx) (padsOf_size ..) hlgc)
  obtain ⟨lE, -, hinn⟩ := innerP_go_pads resp (Claims.hOf H) lay ω i rep ic
    (ZcZ.proof lay pads.h (Setup.ofCircuit st).m (Zc.coinsAt resp l.size) tr).finalC hcons tI lI hβ'
  exact ⟨lg, zl, rec, tI, lI, rp, x, lgc, by rw [hmerk]; exact hlig, hre, hlgc, _, _, lE, hinn⟩

/-- **`ProverRunsT` beyond level 0 holds**: a rep returns at good coins (`repT_go`) and records `repRoundsT` rounds
(`rep_rounds`). -/
theorem runsRest (H : Claims.Msg → Claims.Msg) : RunsRest H := by
  intro h0 h2 h3 h64 st tr hT ω i d k r v cm pts e rep ic hin hrep
  exact ⟨fun l prior _ hco hmr => repT_go h0 h2 h3 h64 st tr hT ω i d r v cm pts e rep ic hin hrep l prior hco hmr,
    fun prior ws l a l' hgo => rep_rounds h3 h64 st tr hT ω i d r v cm pts e rep ic hin hrep prior ws l a.1 a.2 l'
      hgo⟩

end FlockComplete.ZkTables
