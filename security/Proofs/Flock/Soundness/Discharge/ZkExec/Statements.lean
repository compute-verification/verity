module

public import Proofs.Flock.Soundness.Discharge.ZkExec.Live
public import Proofs.Flock.Soundness.Discharge.Exec.Setup
public import Proofs.Flock.Soundness.Discharge.Exec.Retained

/-! The target statements of verity-verify-proof `--zk`'s acceptance and its refinement, each proved in its own file. -/

@[expose] public section

namespace FlockSoundness.Discharge.ZkExec

open Lean (Json)
open Game Audit Exec Refine Flock
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

def ZkVerifyOk : Prop :=
  ∀ (spec : SessionSpec) (tabs : Array Setup) (record : String) (proofs : Array ByteArray),
    Flock.Zk.verify spec tabs record proofs = .ok () →
    spec.coinTree.isSome = true ∧ ∃ (sess : RunRecord) (schs : Array (Flock.Schedule × ℕ)),
      Record.decode spec record = .ok sess ∧ tabs.size = spec.tables.size ∧ 0 < tabs.size ∧
      proofs.size = spec.repStreams.size ∧ schs.size = tabs.size ∧
      ∀ (i : ℕ) (hi : i < tabs.size), ∃ (hs : i < schs.size) (c : ByteArray),
        Flock.Zk.schedule tabs[i].wideLeaf tabs[i].m = .ok schs[i] ∧
        Flock.Zk.verifyRep spec tabs[i] schs[i].1 schs[i].2 sess i 0 proofs[2 * i]! = .ok c ∧
        Flock.Zk.verifyRep spec tabs[i] schs[i].1 schs[i].2 sess i 1 proofs[2 * i + 1]! = .ok c

def shapeZK (st : Setup) (sch : Flock.Schedule) (extras : Array Flock.Claim) : Shape :=
  { m := st.m, kLog := st.kLog, claims := 2 + extras.size, sched := sch, hashLen := st.merkle.digestLen,
    saltLen := if st.merkle.saltBytes = 0 then none else some st.merkle.saltBytes }

def repProg (st : Setup) (sch : Flock.Schedule) (tPad r : ℕ) (extras : Array Flock.Claim) (zp : Flock.Zk.ZkProof) :
    Flock.Transcript.TM Unit := do
  let p := zp.proof
  let lay := Flock.Zk.PadLayout.new st.m st.kLog sch.k0
  let rp ← Flock.Zk.replay st lay p zp.inner.root
  let ab := rp.claims[0]!
  let c := rp.claims[1]!
  let ccAb := Flock.Zk.claimCheck ab p.ringSwitches[0]!
  let ccC := Flock.Zk.claimCheck c p.ringSwitches[1]!
  let (T, sws) ← Flock.ringSwitch st.m (#[{ ab with value := ccAb }, { c with value := ccC }] ++ extras) p.ringSwitches
  let rec ← Flock.Zk.ligerito st.merkle sch T p.cap p.ligerito (Flock.basis sws) ⟨tPad, r, zp.level0⟩
  let lg ← match Flock.Zk.ligeritoConstraints lay rec p.ligerito zp.level0 with
    | .ok v => pure v
    | .error e => throw e
  Flock.Zk.verifyInner lay zp.inner p.zerocheck.finalC
    (rp.cons ++ #[("ring switch ab", rp.abValue.addConst ccAb), ("ring switch c", rp.cValue.addConst ccC)] ++ lg)

def ZkRepOk : Prop :=
  ∀ (spec : SessionSpec) (st : Setup) (sch : Flock.Schedule) (tPad : ℕ) (sess : RunRecord) (i r : ℕ)
    (bytes cap : ByteArray),
    Flock.Zk.verifyRep spec st sch tPad sess i r bytes = .ok cap →
    ∃ (stream : StreamRec) (extras : Array Flock.Claim) (zp : Flock.Zk.ZkProof) (tr : Flock.Transcript),
      sess.streams.find? (fun x => x.name == s!"{spec.tables[i]!}/rep{r}") = some stream ∧
      st.extra sess.points = .ok extras ∧ Flock.Zk.decode (shapeZK st sch extras) bytes = .ok zp ∧
      (zp.proof.params != Flock.pcsParams st.merkle sch) = false ∧ zp.inner.root.size = 64 ∧
      (repProg st sch tPad r extras zp).run
        (Flock.Transcript.init stream.rounds (spec.domain spec.tables[i]! r).toUTF8) = .ok ((), tr) ∧
      tr.k = stream.rounds.size ∧ cap = zp.proof.cap.foldl (· ++ ·) ByteArray.empty ∧ cap = sess.rootB[i]!

/-- The inner code's length: the pads message (`len + K_PAD`) rounded up to a power of two, at rate `1/8`. -/
def innerN (lay : Flock.Zk.PadLayout) : ℕ :=
  (lay.len + Flock.Zk.K_PAD).nextPowerOfTwo * 2 ^ Flock.Zk.INNER_LOG_INV_RATE

def innerRow (lay : Flock.Zk.PadLayout) (pr : Flock.Zk.InnerProof) (cons : Array (String × Flock.Zk.Lin))
    (eps : F128) (gammas : Array F128) : Flock.Zk.Lin :=
  Flock.Zk.Lin.combo gammas ((cons ++ Flock.Zk.triple lay eps pr.rho pr.sigma).map (·.2)) lay.len

def ZkInnerOk : Prop :=
  ∀ (lay : Flock.Zk.PadLayout) (pr : Flock.Zk.InnerProof) (finalC : F128) (cons : Array (String × Flock.Zk.Lin))
    (t t' : Flock.Transcript),
    Flock.Zk.verifyInner lay pr finalC cons t = .ok ((), t') →
    pr.y.size = lay.len + Flock.Zk.K_PAD ∧ pr.opened.size = Flock.Zk.Q ∧ pr.salts.size = Flock.Zk.Q ∧
    pr.paths.size = Flock.Zk.Q ∧ pr.root.size = 64 ∧ pr.tauCommit.size = 64 ∧
    pr.tauSalt.size = Flock.Zk.TAU_SALT_BYTES ∧
    ∃ (eps beta : F128) (gammas ws : Array F128),
      beta.isZero = false ∧ Flock.Zk.tauCommitment pr.tau pr.tauSalt = pr.tauCommit ∧
      Flock.dot (innerRow lay pr cons eps gammas).a pr.y = (innerRow lay pr cons eps gammas).c + beta * pr.tau ∧
      ∀ k < Flock.Zk.Q,
        Flock.Zk.codeAt pr.y ((ws[k]!).lo.toNat % innerN lay) = (pr.opened[k]!).1 + beta * (pr.opened[k]!).2 ∧
        (pr.paths[k]!).size = Nat.log2 (innerN lay) ∧
        Flock.climb MerkleScheme.hm96Sha512
          (Hm96.Default512.leaf (Sha512.hash ((pr.opened[k]!).1.toBytes ++ (pr.opened[k]!).2.toBytes)) pr.salts[k]!)
          ((ws[k]!).lo.toNat % innerN lay) (pr.paths[k]!).toList = pr.root

def AcceptsZKSetupH : Prop :=
  ∀ (I : Inputs) (t : Exec.Transcript), I.tags.hiddenOutputs = false → AcceptsZK I t →
    ∃ spec sts, setupOf (zkInputs I) t.record = .ok (spec, sts) ∧
      Flock.Zk.verify spec (sts.map Flock.Setup.ofCircuit) t.record t.proofs = .ok () ∧ 0 < sts.size ∧
      ∀ (j : ℕ) (hj : j < sts.size), Flock.Zk.shapeOk sts[j] = .ok () ∧ ∃ d,
        Stmt.setupH (tagsAt (zkTags I) I.count j) I.circuitFile I.publicFile (zkCoins I) I.tables d I.partition
          I.program = .ok sts[j] ∧ (I.count ≤ 1 → d = recordDraw t.record)

def AcceptsZKOne : Prop :=
  ∀ (I : Inputs) (t : Exec.Transcript), I.tags.hiddenOutputs = false → I.count ≤ 1 → AcceptsZK I t →
    ∃ st, Stmt.setupH (zkTags I) I.circuitFile I.publicFile (zkCoins I) I.tables (recordDraw t.record) I.partition
        I.program = .ok st ∧ Flock.Zk.shapeOk st = .ok () ∧
      Flock.Zk.verify st.spec #[Flock.Setup.ofCircuit st] t.record t.proofs = .ok ()

def AcceptsZKRetained : Prop :=
  ∀ (I : Inputs) (t : Exec.Transcript), I.tags.hiddenOutputs = false → I.tags.retainRounds = true → AcceptsZK I t →
    ∃ spec sts sess, setupOf (zkInputs I) t.record = .ok (spec, sts) ∧ Record.decode spec t.record = .ok sess ∧
      ∀ s ∈ sess.streams.toList, AllRetained s.rounds

section Refine

variable {n : ℕ} {Reg : Type} {mPts : ℕ} {P Cp Op Out : Finset (Fin n) → Reg → Type}
  (G : ∀ S R, (Fin 2 → Fin mPts → 𝔽) → Fin 2 → Game (P S R))
  (out : ∀ S R, Cp S R → P S R × P S R → Op S R → Out S R) (acc : ∀ S R, Out S R → Bool)
  [∀ S R, Nonempty (Op S R)] (I : Inputs) (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript)
  (opn : ∀ S R, Array ByteArray → Op S R) (dg : ByteArray → ByteArray)
  (dec : Finset (Fin n) → Reg → ℕ → ByteArray → List Msg) (one : Finset (Fin n) → Reg → ℕ → Bool)
  (fin : Finset (Fin n) → Reg → Fin 2 → Array ByteArray → List Msg)
  (capOf : ∀ S R, ByteArray → Cp S R) (N : ℕ) (L : SamplingDesign n)

def ExecRefinesZKOne : Prop :=
  ∀ (hinh : ∀ S R pts r, AllInh (G S R pts r)) (hdec : DecodesZK G out acc I wr opn dec one fin capOf)
    (τ : Strategy (auditLive L Reg (liveOf dg mPts N wr))),
    ExecRefines (auditLive L Reg (liveOf dg mPts N wr)) τ (LiveAcceptsZK I) (fun _ => False)
      (audit L Reg (sessionZC G out acc)) (simAuditZK G out acc I wr opn dg dec one fin capOf N hinh hdec τ)

def ZkHeadlineExecOne : Prop :=
  ∀ (hinh : ∀ S R pts r, AllInh (G S R pts r)) (hdec : DecodesZK G out acc I wr opn dec one fin capOf)
    (τ : Strategy (auditLive L Reg (liveOf dg mPts N wr))) (Q : Prop) (B : ℝ≥0∞),
    prob (fun o => o.1 = true ∧ Q) (audit L Reg (sessionZC G out acc))
        (simAuditZK G out acc I wr opn dg dec one fin capOf N hinh hdec τ) ≤ B →
      prob (fun o => LiveAcceptsZK I o ∧ Q) (auditLive L Reg (liveOf dg mPts N wr)) τ ≤ B

end Refine

end FlockSoundness.Discharge.ZkExec
