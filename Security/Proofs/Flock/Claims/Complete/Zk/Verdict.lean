import Proofs.Flock.Claims.Complete.Verdict
import Proofs.Flock.Claims.Complete.Zk.Session

/-!
# The masked verdict accepts when each of its checks passes

The masked counterpart of `Verdict.lean`: each step of `Zk.verifyRepWith` and `Zk.verifyWithM`, as its own hypothesis,
composes into acceptance. B's checks are the plain ones (`checks_ok`, generic in `zk`).

* **A masked rep** (`zk_verifyRepWith_ok`): its stream is recorded (S11), the extra claims evaluate, the masked proof
  file decodes at the rep's shape (`shapeZk`), its PCS parameters are the `--zk` schedule's, the pads' root is 64 bytes;
  the masked replay (`Zk.replayWith`), the ring switch on the replay's claims valued by their checks (`ringClaimsZk`),
  the masked Ligerito (`Zk.ligeritoWith`), level 0's lane phase as constraints (`Zk.ligeritoConstraints`) and the inner
  proof on every constraint (`Zk.verifyInnerWith`, `repConsZk`) each pass from the transcript the previous one left;
  every recorded round is consumed (S15) and the cap is the committed `root_B` (S16).
* **The masked verdict** (`zk_verifyWith_ok`): `verifyWith_ok`'s checks, the keyed coin tree, and each table's `--zk`
  schedule (`zkSchedsOf`) in place of fast100's.
-/

open Flock FlockDeploy Transcript

namespace FlockComplete

/-! ## A masked rep -/

/-- The ring switch's claims at a masked rep: the replay's two claims, each valued by its check on the switched vector
(their values are masked, so each check is a constraint, `repConsZk`). -/
def ringClaimsZk (rp : Zk.Replayed) (p : Proof) : Array Claim :=
  #[{ rp.claims[0]! with value := Zk.claimCheck rp.claims[0]! p.ringSwitches[0]! },
    { rp.claims[1]! with value := Zk.claimCheck rp.claims[1]! p.ringSwitches[1]! }]

/-- The masked rep's constraints before level 0's: the replay's, and the two ring-switch claims' checks. -/
def repConsZk (rp : Zk.Replayed) (p : Proof) : Array (String × Zk.Lin) :=
  rp.cons ++ #[("ring switch ab", rp.abValue.addConst (Zk.claimCheck rp.claims[0]! p.ringSwitches[0]!)),
    ("ring switch c", rp.cValue.addConst (Zk.claimCheck rp.claims[1]! p.ringSwitches[1]!))]

/-- **A masked rep accepts** when each of its checks passes, each from the transcript the previous one left. -/
theorem zk_verifyRepWith_ok (hs : Flock.Hashes V) (spec : SessionSpec) (st : Setup) (sch : Schedule) (tPad : ℕ)
    (sess : RunRecord) (i r : ℕ) (bytes : ByteArray) {stream : StreamRec} {extras : Array Claim} {zp : Zk.ZkProof}
    {rp : Zk.Replayed} {T : F128} {sws : Array Switched} {rec : Zk.L0Rec} {lg : Array (String × Zk.Lin)}
    {t₁ t₂ t₃ t₄ : Transcript}
    (hS : sess.streams.find? (·.name == s!"{spec.tables[i]!}/rep{r}") = some stream)
    (hE : st.extra sess.points = .ok extras)
    (hP : Zk.decode (shapeZk st sch extras.size) bytes = .ok zp)
    (hR : (zp.proof.params != pcsParams st.merkle sch) = false)
    (hI : (zp.inner.root.size != 64) = false)
    (hY : (Zk.replayWith hs st (Zk.PadLayout.new st.m st.kLog sch.k0) zp.proof zp.inner.root).run
      (Transcript.init stream.rounds (spec.domain spec.tables[i]! r).toUTF8) = .ok (rp, t₁))
    (hW : (ringSwitchWith hs st.m (ringClaimsZk rp zp.proof ++ extras) zp.proof.ringSwitches).run t₁ =
      .ok ((T, sws), t₂))
    (hG : (Zk.ligeritoWith hs st.merkle sch T zp.proof.cap zp.proof.ligerito (basis sws) ⟨tPad, r, zp.level0⟩).run t₂ =
      .ok (rec, t₃))
    (hC : Zk.ligeritoConstraints (Zk.PadLayout.new st.m st.kLog sch.k0) rec zp.proof.ligerito zp.level0 = .ok lg)
    (hN : (Zk.verifyInnerWith hs (Zk.PadLayout.new st.m st.kLog sch.k0) zp.inner zp.proof.zerocheck.finalC
      (repConsZk rp zp.proof ++ lg)).run t₃ = .ok ((), t₄))
    (hK : (t₄.k != stream.rounds.size) = false)
    (hB : (flat zp.proof.cap != sess.rootB[i]!) = false) :
    Zk.verifyRepWith hs spec st sch tPad sess i r bytes = .ok (flat zp.proof.cap) := by
  unfold Zk.verifyRepWith
  simp only [flat] at hB
  simp only [ringClaimsZk] at hW
  simp only [repConsZk] at hN
  unfold shapeZk at hP
  dsimp only
  rw [hS]
  dsimp only
  simp only [liftM, monadLift_self, hE, hP, ok_bind, hR, hI, Bool.false_eq_true, ite_false, StateT.run_bind, hY, hW,
    hG, hC, pure_bind, hN, hK, hB]
  rfl

/-! ## The masked verdict -/

/-- A table's `--zk` schedule and `t_pad`, `default` where it has none. -/
def schedZkOf (m : ℕ) : Schedule × ℕ :=
  match Zk.schedule m with
  | .ok v => v
  | .error _ => default

/-- Each table's `--zk` schedule and `t_pad`. -/
def zkSchedsOf (tabs : Array Setup) : Array (Schedule × ℕ) := tabs.map fun st => schedZkOf st.m

/-- **The masked verdict accepts** when each of its checks passes. -/
theorem zk_verifyWith_ok (h : ByteArray → ByteArray) (spec : SessionSpec) (tabs : Array Setup) (record : String)
    (proofs : Array ByteArray) {sess : RunRecord}
    (hT : (tabs.size != spec.tables.size || tabs.size == 0) = false)
    (hCT : spec.coinTree.isNone = false)
    (hD : Record.decodeWith (Flock.Hashes.ofSha512 h : Flock.Hashes V) spec record = Except.ok sess)
    (hU : ∀ q ∈ (tabs.zip sess.publics).toList, q.1.publics q.2 = .ok ())
    (hF : ∀ st ∈ tabs.toList, ∃ v, Zk.schedule st.m = .ok v)
    (hN : (proofs.size != spec.repStreams.size) = false)
    (hL : (sess.proofSha.size != spec.repStreams.size ||
      !spec.repStreams.all fun a => sess.proofSha.any fun x => x.1 == a.2.2) = false)
    (hG : ∀ q ∈ (spec.repStreams.zip proofs).toList,
      ((sess.proofSha.find? (·.1 == q.1.2.2)).map (·.2) != some (hexOf (dgOf spec h q.2))) = false)
    (hR : ∀ i < tabs.size, ∃ c,
      Zk.verifyRepWith (Flock.Hashes.ofSha512 h : Flock.Hashes V) spec tabs[i]! (zkSchedsOf tabs)[i]!.1
        (zkSchedsOf tabs)[i]!.2 sess i 0 proofs[2 * i]! = Except.ok c ∧
      Zk.verifyRepWith (Flock.Hashes.ofSha512 h : Flock.Hashes V) spec tabs[i]! (zkSchedsOf tabs)[i]!.1
        (zkSchedsOf tabs)[i]!.2 sess i 1 proofs[2 * i + 1]! = Except.ok c) :
    Zk.verifyWith h spec tabs record proofs = .ok () := by
  unfold Zk.verifyWith Zk.verifyWithM
  have hpub : forIn (tabs.zip sess.publics) PUnit.unit (fun x _ => do
      liftM (x.1.publics x.2)
      pure (ForInStep.yield PUnit.unit) : _ → _ → V _) = Except.ok () := by
    rw [← Array.forIn_toList]
    refine forIn_ok _ _ fun a ha => ?_
    simp only [liftM, monadLift_self, hU a ha]
    rfl
  have h17 : forIn (spec.repStreams.zip proofs) PUnit.unit (fun x _ => do
      let d ← SessionSpec.dgWith (Flock.Hashes.ofSha512 h) spec x.2
      if (Option.map (fun x => x.2) (Array.find? (fun x_1 => x_1.1 == x.1.2.2) sess.proofSha) != some (hexOf d)) = true
      then do
        throw (toString "S17: " ++ toString x.1.2.2 ++ toString ": the proof is not the one the server received")
        pure (ForInStep.yield PUnit.unit)
      else pure (ForInStep.yield PUnit.unit) : _ → _ → V _) = Except.ok () := by
    rw [← Array.forIn_toList]
    refine forIn_ok _ _ fun a ha => ?_
    simp only [dgWith_ofSha512, ok_bind, hG a ha, Bool.false_eq_true, ite_false]
    rfl
  simp only [hT, hCT, Bool.false_eq_true, ite_false, hD, ok_bind, hpub, hN, hL, h17]
  rw [array_mapM_ok _ (fun st => schedZkOf st.m) tabs ?_]
  · rw [show Array.map (fun st => schedZkOf st.m) tabs = zkSchedsOf tabs from rfl, ok_bind]
    rw [Std.Legacy.Range.forIn_eq_forIn_range', forIn_ok _ _ ?_]
    · rfl
    intro i hi
    have hi' : i < tabs.size := by
      simp only [Std.Legacy.Range.size, Nat.sub_zero, Nat.add_one_sub_one, Nat.div_one, List.mem_range'] at hi
      omega
    obtain ⟨c, h0, h1⟩ := hR i hi'
    have hcc : (c != c) = false := by
      show (!(c.data == c.data)) = false
      simp
    simp only [h0, h1, ok_bind, hcc, Bool.false_eq_true, ite_false]
    rfl
  · intro st hst
    obtain ⟨v, hv⟩ := hF st hst
    simp only [liftM, monadLift_self, hv, schedZkOf]

end FlockComplete
