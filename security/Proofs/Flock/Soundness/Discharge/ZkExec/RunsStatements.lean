module

public import Proofs.Flock.Soundness.Discharge.DecodeZL.LIG
public import Proofs.Flock.Soundness.Discharge.DecodeZL.OP
public import Proofs.Flock.Soundness.Discharge.DecodeZL.Runs
public import Proofs.Flock.Soundness.Discharge.PlaysCoin.Statements
public import Proofs.Flock.Soundness.Discharge.ZkExec.Coin
public import Proofs.Flock.Soundness.Discharge.ZkExec.Shape
public import Proofs.Flock.Soundness.Discharge.ZkExec.Statements
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt

/-! The statements of `RunsZC`'s decoder half: a `--zk` rep's frames, its messages, and the decoder that cuts them into
rounds. -/

@[expose] public section

namespace FlockSoundness.Discharge.ZkExec

open Flock Refine Exec Game Model DecodeZL

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

variable {wideLeaf : Bool}

def zcHeadZ (digest cap root : ByteArray) : List Frame :=
  [.label "flock-r1cs-v0".toUTF8, .bytes digest, .bytes cap, .label "flock-zerocheck-v0".toUTF8,
    .label Zk.LABEL_PADS.toUTF8, .bytes root, .sq 2 6]

def zcTailZ (m : ℕ) (p : ZerocheckProof) : List Frame :=
  [.sq 2 (m - 13), .vec p.round1ab.toList, .vec p.round1c.toList, .sq 1 1] ++
  (List.range' 0 (m - 6)).flatMap (fun i => [.one (p.rounds[i]!).1, .one (p.rounds[i]!).2, .sq 1 1]) ++
  [.one p.finalA, .one p.finalB]

def l0Fr (z : Zk.Level0Msgs) : List Frame :=
  [.vec [z.e0.c0, z.e0.c1], .vec [z.e1.c0, z.e1.c1], .sq 2 2, .sq 2 2, .label Zk.LABEL_CLOSE.toUTF8,
    .vec [z.tPrime.c0, z.tPrime.c1], .vec (z.ybar.foldl (fun acc v => (acc.push v.c0).push v.c1) #[]).toList]

def ligFramesZ (sch : Flock.Schedule) (T : F128) (cap : Array ByteArray) (p : LigeritoProof) (z : Zk.Level0Msgs) :
    List Frame :=
  [.label "flock-ligerito-basis-f256-split-v0".toUTF8, .one T, capFr cap, .label Zk.LABEL_OPEN.toUTF8] ++
    ([.sq 2 sch.n, .one p.oodValues[0]!, .bytes (le64 p.claimNonces[0]!), .sq 1 1] ++ kFr p.messages[0]! ++
      (chFr p 1 (sch.k0 - 1) ++ .sq 2 2 :: (l0Fr z ++ kFr p.messages[sch.k0]!)) ++ capFr p.recursiveCaps[0]! ::
      (introFr p (sch.n - sch.k0 + 1) 1 (sch.k0 + 1) 1 ++ introFr p (sch.n - sch.k0 + 1) 2 (sch.k0 + 2) 2 ++
        queryFr p (sch.levels.getD 0 default) 0 0 ++ kFr p.messages[sch.k0 + 3]! ++
        [.bytes (le64 p.claimNonces[3]!), .sq 1 1] ++ (List.range' 0 (sch.levels.size - 1)).flatMap (levelFr sch p)))

def innerFr (pr : Zk.InnerProof) (finalC : F128) (nγ : ℕ) : List Frame :=
  [.label Zk.LABEL_INNER.toUTF8, .one finalC, .sq 1 1, .vec [pr.rho, pr.sigma], .sq 2 nγ, .bytes pr.tauCommit, .sq 1 1,
    .vec pr.y.toList, .sq 2 Zk.Q]

def restFramesZ (st : Setup) (sch : Flock.Schedule) (n : ℕ) (zp : Zk.ZkProof) (T : F128) : List Frame :=
  zcTailZ st.m zp.proof.zerocheck ++ (lcFrames st.kLog zp.proof.lincheck ++ (rsFrames n zp.proof.ringSwitches ++
    (ligFramesZ sch T zp.proof.cap zp.proof.ligerito zp.level0 ++ innerFr zp.inner zp.proof.zerocheck.finalC 10)))

def repFramesZ (st : Setup) (sch : Flock.Schedule) (domain : ByteArray) (n : ℕ) (zp : Zk.ZkProof) (T : F128) :
    List Frame :=
  .dom domain :: (zcHeadZ st.digest (zp.proof.cap.foldl (· ++ ·) ByteArray.empty) zp.inner.root ++
    restFramesZ st sch n zp T)

def evMsg (z : Zk.Level0Msgs) : Msg := ⟨Fin 2 → 𝕂, ![gOf z.e0, gOf z.e1]⟩

def tMsg (z : Zk.Level0Msgs) : Msg := ⟨𝕂, gOf z.tPrime⟩

def ybarMsg (t : ℕ) (z : Zk.Level0Msgs) : Msg := ⟨Fin t → 𝕂, fun i => gOf z.ybar[i.val]!⟩

def ligMsgsZ {D : Type} (p : LigeritoProof) (capD : Array ByteArray → Cap D) (sch : Flock.Schedule) (t : ℕ)
    (z : Zk.Level0Msgs) : List Msg :=
  fMsg p.oodValues[0]! :: kMsg p.messages[0]! :: kMsgs p 1 (sch.k0 - 1) ++ evMsg z :: tMsg z :: ybarMsg t z ::
    kMsg p.messages[sch.k0]! :: ⟨Cap D, capD p.recursiveCaps[0]!⟩ ::
    [fMsg p.oodValues[1]!, kMsg p.messages[sch.k0 + 1]!, fMsg p.oodValues[2]!, kMsg p.messages[sch.k0 + 2]!] ++
    [kMsg p.messages[sch.k0 + 3]!] ++ recMsgs p capD sch.ext (sch.levels.size - 1) 0 (sch.k0 + 4) 3

def innerMsgs (pr : Zk.InnerProof) (finalC : F128) (nY : ℕ) : List Msg :=
  [fMsg finalC, ⟨𝔽 × 𝔽, (fOf pr.rho, fOf pr.sigma)⟩, ⟨Dig, toDig pr.tauCommit⟩, ⟨Fin nY → 𝔽, vecOf nY pr.y⟩]

def layZ (st : Setup) (sch : Flock.Schedule) : Zk.PadLayout := Zk.PadLayout.new st.m st.kLog sch.k0

def restMsgsZ (st : Setup) (sch : Flock.Schedule) (n t : ℕ) (zp : Zk.ZkProof) : List Msg :=
  zcMsgs st.m zp.proof.zerocheck ++ (lcMsgs st.kLog zp.proof.lincheck ++ (opMsgs n zp.proof.ringSwitches ++
    (ligMsgsZ zp.proof.ligerito capDig sch t zp.level0 ++
      innerMsgs zp.inner zp.proof.zerocheck.finalC ((layZ st sch).len + Zk.K_PAD))))

def repMsgsZ (st : Setup) (sch : Flock.Schedule) (n t : ℕ) (zp : Zk.ZkProof) : List Msg :=
  ⟨Dig, toDig zp.inner.root⟩ :: restMsgsZ st sch n t zp

/-- The pair's sums `e`, from their four values. -/
def wEv : DecodeZL.Wire := .vals (Fin 2 → 𝕂) 4 fun vs => ![gOf ⟨vs[0]!, vs[1]!⟩, gOf ⟨vs[2]!, vs[3]!⟩]

/-- An element of `K`, from its two values. -/
def wK : DecodeZL.Wire := .vals 𝕂 2 fun vs => gOf ⟨vs[0]!, vs[1]!⟩

/-- `t` elements of `K`, from `2t` values. -/
def wY (t : ℕ) : DecodeZL.Wire := .vals (Fin t → 𝕂) (2 * t) fun vs i => gOf ⟨vs[2 * i.val]!, vs[2 * i.val + 1]!⟩

/-- A 64-byte digest, from one byte string. -/
def wDig : DecodeZL.Wire := .bytes Dig toDig

/-- **The `--zk` Ligerito wires**: the non-ZK session's (`wLIG`) with the last lane round's coin followed by the pair's
sums' round, the second `ρ` coin's empty round, and a round that carries `T'` and `ybar` before the round's message and
the level-1 cap. -/
noncomputable def wLIGZ (s : Accounting.Schedule) (t : ℕ) : List (List DecodeZL.Wire) :=
  [[]] ++ ([[.f]] ++ (List.replicate s.k0 [.kk] ++ ([[wEv], [], [wK, wY t, .kk, .cap]] ++
    (wMid ++ wLev s.ext (s.levels.length - 1)))))

/-- The inner proof's wires: `final_c'`'s round, `[ρ, σ]`'s, `τ`'s commitment's and `Y`'s (`nY` words). -/
def wInner (nY : ℕ) : List (List DecodeZL.Wire) := [[.f], [.ff], [wDig], [.v nY]]

/-- **The wires of a `--zk` rep's rounds after round 0**: the non-ZK session's zerocheck's and lincheck's after round 0,
the opening's of `n` claims, Ligerito's with level 0's hiding (`ybar` of `t` elements), and the inner proof's on `np`
pads. -/
noncomputable def wRestZ (m kLog n : ℕ) (s : Accounting.Schedule) (t np : ℕ) : List (List DecodeZL.Wire) :=
  (wZL m kLog).tail ++ (wOP n ++ (wLIGZ s t ++ wInner (np + Zk.K_PAD)))

/-- **The pads root in round 0's bytes**: the 64 bytes before the closing squeeze's 16-byte header, after the root's
own 16-byte header (a 64-byte string needs no padding). -/
def rootOf (B : ByteArray) : ByteArray := B.extract (B.size - 80) (B.size - 16)

/-- **The `--zk` decoder**: round 0 carries the pads root, and round `j + 1` is read by `W`'s `j`-th wires. -/
noncomputable def decZ (W : List (List DecodeZL.Wire)) : ℕ → ByteArray → List Msg
  | 0, B => [⟨Dig, toDig (rootOf B)⟩]
  | j + 1, B => decOf W j B

def shapeZE (m kLog n : ℕ) (s : Accounting.Schedule) (t np : ℕ) : List Exec.Kind :=
  .msg Dig :: ((zcShape 𝔽 m ++ (lcShape 𝔽 kLog ++ (opShape 𝔽 n ++ (ligPadShape 𝔽 𝕂 Dig s 2 t ++ [.msg 𝔽])))) ++
    innerShape 𝔽 Dig 7 np Zk.K_PAD Zk.Q)

def ZkScheduleSeams : Prop :=
  ∀ (wideLeaf : Bool) (m : ℕ) (sch : Flock.Schedule) (tPad : ℕ), Zk.schedule wideLeaf m = .ok (sch, tPad) →
    25 ≤ m ∧ m ≤ 35 ∧ sch.m = m ∧ 1 ≤ sch.k0 ∧ tPad = 2 * (sch.levels.getD 0 default).q ∧ DimSeam sch ∧
      (∀ i < sch.levels.size, LevelSeam (sch.levels.getD i default)) ∧
      (∀ i < sch.levels.size, Accounting.ceilLog2 (sch.levels.getD i default).q ≠ 1) ∧
      ∀ lv ∈ sch.levels, lv.cols < 64 ∧ 0 < lv.q ∧ lv.q < 512 ∧ lv.cap ≤ 8

def ZkRepFrames : Prop :=
  ∀ (st : Setup) (sch : Flock.Schedule) (tPad r : ℕ) (extras : Array Flock.Claim) (zp : Zk.ZkProof)
    (rounds : Array Round) (dom : ByteArray) (tr : Flock.Transcript),
    Zk.schedule st.wideLeaf st.m = .ok (sch, tPad) →
    (repProg st sch tPad r extras zp).run (Flock.Transcript.init rounds dom) = .ok ((), tr) →
    tr.k = rounds.size → AllRetained rounds →
    ∃ T : F128, (chunks (repFramesZ st sch dom (2 + extras.size) zp T)).1.length = rounds.size ∧
      ∀ k < rounds.size, Retains rounds[k]! (encs ((chunks (repFramesZ st sch dom (2 + extras.size) zp T)).1.getD k []))

def ZkDecodeSizes : Prop :=
  ∀ (sh : Shape) (bytes : ByteArray) (zp : Zk.ZkProof), Zk.decode sh bytes = .ok zp →
    ProofSizes sh zp.proof ∧ zp.inner.y.size < 2 ^ 64

def ZkCarries : Prop :=
  ∀ (st : Setup) (sch : Flock.Schedule) (tPad : ℕ) (extras : Array Flock.Claim) (zp : Zk.ZkProof) (T : F128),
    Zk.schedule st.wideLeaf st.m = .ok (sch, tPad) → st.merkle.digestLen = 64 → ProofSizes (shapeZK st sch extras) zp.proof →
    zp.inner.y.size < 2 ^ 64 → zp.level0.ybar.size = tPad → zp.inner.tauCommit.size = 64 →
    zp.inner.y.size = (layZ st sch).len + Zk.K_PAD →
    Carries (wRestZ st.m st.kLog (2 + extras.size) (schedOf sch) tPad (layZ st sch).len)
      ((shapeZE st.m st.kLog (2 + extras.size) (schedOf sch) tPad (layZ st sch).len).drop 2)
      (restFramesZ st sch (2 + extras.size) zp T) (restMsgsZ st sch (2 + extras.size) tPad zp) []

def ZkCut : Prop :=
  ∀ (st : Setup) (sch : Flock.Schedule) (tPad : ℕ) (extras : Array Flock.Claim) (bytes : ByteArray)
    (zp : Zk.ZkProof) (T : F128) (dom : ByteArray) (rds : Array Round),
    Zk.schedule st.wideLeaf st.m = .ok (sch, tPad) → st.merkle.digestLen = 64 →
    Zk.decode (shapeZK st sch extras) bytes = .ok zp → zp.inner.root.size = 64 → zp.level0.ybar.size = tPad →
    zp.inner.tauCommit.size = 64 → zp.inner.y.size = (layZ st sch).len + Zk.K_PAD →
    (chunks (repFramesZ st sch dom (2 + extras.size) zp T)).1.length = rds.size →
    (∀ k < rds.size, Retains rds[k]! (encs ((chunks (repFramesZ st sch dom (2 + extras.size) zp T)).1.getD k []))) →
    PlaysCoin.cut ((shapeZE st.m st.kLog (2 + extras.size) (schedOf sch) tPad (layZ st sch).len).take
        ((repMsgsZ st sch (2 + extras.size) tPad zp).length + rds.toList.length))
      (repMsgsZ st sch (2 + extras.size) tPad zp) =
    PlaysCoin.roundGroups (decZ (wRestZ st.m st.kLog (2 + extras.size) (schedOf sch) tPad (layZ st sch).len)) 0
      rds.toList ++ [[]]

end FlockSoundness.Discharge.ZkExec
