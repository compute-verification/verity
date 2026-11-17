module

public import Proofs.Flock.Soundness.Discharge.ZkExec.PlaysStatements
public import Proofs.Flock.Soundness.Discharge.ZkExec.Compiled
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

/-! The statements of `RunsZC`'s verdict half: the compiled `--zk` table under `hm96-sha512/v1` accepts at the two reps'
results. -/

@[expose] public section

namespace FlockSoundness.Discharge.ZkExec

open Flock Refine Exec Game Model DecodeZL

local notation "𝔽" => FlockLevel3.GF128

instance digInhabitedZ : Inhabited Dig := ⟨H512 []⟩

/-- The pads tree's column encoding: the pair `(a, b)` as the row `[a, b]`, `a.toBytes ‖ b.toBytes`; nodes as
`enc512`'s. -/
def encPair : Merkle.Enc (𝔽 × 𝔽) Dig where
  col p := enc512.col [p.1, p.2]
  dig := enc512.dig
  col_inj a b h := by
    have h' := enc512.col_inj h
    simp only [List.cons.injEq, and_true] at h'
    exact Prod.ext h'.1 h'.2
  dig_inj := enc512.dig_inj
  dig_len := enc512.dig_len

/-- `τ`'s encoding: `TAU_TAG ‖ τ.toBytes`; nodes as `enc512`'s. -/
def encTau : Merkle.Enc 𝔽 Dig where
  col τ := Zk.TAU_TAG.data.toList ++ enc512.col [τ]
  dig := enc512.dig
  col_inj a b h := by
    have h' := enc512.col_inj (List.append_cancel_left h)
    simpa using h'
  dig_inj := enc512.dig_inj
  dig_len := enc512.dig_len

/-- **The `--zk` hashes under `hm96-sha512/v1`**: SHA-512 nodes, and the `hm96` leaf at the executable's key for the
rows, the pads pairs and `τ`. -/
def hs512 : ZkLink.Hashes 𝔽 Dig (List UInt8) where
  H := H512
  E := enc512
  Lf := leaf512
  Ep := encPair
  Lp := Merkle.Leaf.hm96 H512 encPair K512 hL512
  Lτ := Merkle.Leaf.hm96 H512 encTau K512 hL512

/-- **A rep's inner-proof openings, from its proof**: at query `j`, the opened pair, its salt and its path; and `τ`
with its salt. -/
def innerOpensOf (pr : Zk.InnerProof) : ZkLink.InnerOpens 𝔽 Dig (List UInt8) :=
  ⟨fun j => (pairOf pr.opened[j]!, (pr.salts[j]!).data.toList, (pr.paths[j]!).toList.map toDig),
    (fOf pr.tau, pr.tauSalt.data.toList)⟩

/-- The proof file's shape `Zk.verifyRep` decodes it at, for `n` extra claims (`shapeZK` at `n = extras.size`). -/
def shapeZKn (st : Setup) (sch : Flock.Schedule) (n : ℕ) : Shape :=
  { m := st.m, kLog := st.kLog, claims := 2 + n, sched := sch, hashLen := st.merkle.digestLen,
    saltLen := if st.merkle.saltBytes = 0 then none else some st.merkle.saltBytes }

/-- **The openings the proof files decode to**: rep `r`'s proof file `bs[r]`, decoded at the table's shape (the default
proof when it does not decode), read as every level's rows and salts and the rep's inner-proof openings. -/
def opnZ (st : Setup) (sch : Flock.Schedule) (n : ℕ) (bs : Array ByteArray) : ZkLink.OpensZ 𝔽 Dig (List UInt8) :=
  let zp : ℕ → Zk.ZkProof := fun r => ((Zk.decode (shapeZKn st sch n) bs[r]!).toOption).getD default
  ⟨opensOf (zp 0).proof (zp 1).proof sch, saltsOf (zp 0).proof (zp 1).proof sch, fun r => innerOpensOf (zp r).inner⟩

structure ZkTabLay {mPts : ℕ} (T : ZkSession.TabZK 𝔽 mPts) (st : Setup) (sch : Flock.Schedule) : Prop where
  hc₀ : T.c₀ = fOf Zk.PAD_C
  hood : T.pads.ood.val = (layZ st sch).lgOod 0 % (layZ st sch).len
  hmsg : ∀ (j : ℕ) (t : Fin 4), (T.pads.msg j t).val = (layZ st sch).lgMsg j t.val % (layZ st sch).len
  hev : ∀ (i : Fin T.e) (c : Fin 2), (T.pads.ev i c).val = (layZ st sch).lgE i.val c.val % (layZ st sch).len
  hr1ab : ∀ i : Fin 64, (T.padsA.r1ab i).val = (layZ st sch).r1ab i.val % (layZ st sch).len
  hr1c : ∀ i : Fin 64, (T.padsA.r1c i).val = (layZ st sch).r1c i.val % (layZ st sch).len
  hml : ∀ (j : ℕ) (t : Fin 2), (T.padsA.ml j t).val = (layZ st sch).ml j t.val % (layZ st sch).len
  hfc : T.padsA.fc.val = (layZ st sch).fc % (layZ st sch).len
  hlc : ∀ (j : ℕ) (t : Fin 2), (T.padsA.lc j t).val = (layZ st sch).lc j t.val % (layZ st sch).len
  hzp : ∀ i : Fin 64, (T.padsA.zp i).val = (layZ st sch).zp i.val % (layZ st sch).len
  hfa : T.tr.fa.val = (layZ st sch).fa % (layZ st sch).len
  hfb : T.tr.fb.val = (layZ st sch).fb % (layZ st sch).len
  hab : T.tr.ab.val = (layZ st sch).hab % (layZ st sch).len
  htx : T.tr.tx.val = (layZ st sch).tx % (layZ st sch).len
  hty : T.tr.ty.val = (layZ st sch).ty % (layZ st sch).len
  htw : T.tr.tw.val = (layZ st sch).tw % (layZ st sch).len
  hN : T.N = innerN (layZ st sch)
  hdom : ∀ j : Fin T.N, T.dom j = fOf ⟨(j.val + 1).toUInt64, 0⟩
  hpos : ∀ w : 𝔽, (T.pos w).val = (FlockLevel3.GF128.val w).lo.toNat % T.N

def LinRel {np : ℕ} (ℓ : ZkSession.Lin 𝔽 np) (x : Zk.Lin) : Prop :=
  x.a.size = np ∧ ℓ.const = fOf x.c ∧ ∀ i : Fin np, ℓ.coef i = fOf x.a[i.val]!

def ZkInnerAccepts : Prop :=
  ∀ (lay : Zk.PadLayout) (pr : Zk.InnerProof) (finalC : F128) (cons : Array (String × Zk.Lin))
    (t t' : Flock.Transcript) {mPts : ℕ} (T : ZkSession.TabZK 𝔽 mPts) (st : Setup) (sch : Flock.Schedule) (tPad : ℕ)
    (rows : Fin 7 → ZkSession.Lin 𝔽 T.np),
    lay = layZ st sch → ZkTabSeam T st sch tPad → ZkTabLay T st sch → cons.size = 7 →
    (∀ i : Fin 7, LinRel (rows i) cons[i.val]!.2) →
    (∀ k < pr.paths.size, ∀ b ∈ (pr.paths[k]!).toList, b.size = 64) → RoundsWF t.rounds →
    Zk.verifyInner lay pr finalC cons t = .ok ((), t') →
    ∀ v, Plays fOf (ZkLink.innerGameC hs512.H hs512.Ep hs512.Lp hs512.Lτ T.dom T.np T.nP T.q T.pos rows T.tr
        (toDig pr.root))
      [⟨𝔽 × 𝔽, (fOf pr.rho, fOf pr.sigma)⟩, ⟨Dig, toDig pr.tauCommit⟩, ⟨Fin (T.np + T.nP) → 𝔽, vecOf (T.np + T.nP) pr.y⟩]
      (consumed t.rounds t.k t'.k) v → v.1 (innerOpensOf pr)

def ZkRepAccepts : Prop :=
  ∀ {mPts : ℕ} (T : ZkSession.TabZK 𝔽 mPts) (pts : Fin 2 → Fin mPts → 𝔽) (r : Fin 2) (st : Setup)
    (sch : Flock.Schedule) (tPad : ℕ) (extras : Array Flock.Claim) (bytes : ByteArray) (zp : Zk.ZkProof)
    (rounds : Array Round) (dom : ByteArray) (tr : Flock.Transcript) (cap₀ : Cap Dig)
    (os : ZkLink.OpensZ 𝔽 Dig (List UInt8)),
    Zk.schedule st.wideLeaf st.m = .ok (sch, tPad) → st.merkle = MerkleScheme.hm96Sha512 → ZkTabSeam T st sch tPad →
    ZkTabLay T st sch → FoldRealizes st.fold T.S →
    extras.size = (Model.extraClaims T.S T.ptLocal mPts pts).length →
    (∀ (j : ℕ) (hj : j < (Model.extraClaims T.S T.ptLocal mPts pts).length),
      ClaimRel T.S.m extras[j]! (Model.extraClaims T.S T.ptLocal mPts pts)[j]) →
    Zk.decode (shapeZK st sch extras) bytes = .ok zp → RoundsWF rounds →
    (repProg st sch tPad r.val extras zp).run (Flock.Transcript.init rounds dom) = .ok ((), tr) →
    tr.k = rounds.size → cap₀ = capDig zp.proof.cap →
    (∀ ℓ j, os.rows r.val ℓ j = openRow zp.proof.ligerito sch ℓ j) →
    (∀ ℓ j, os.salts r.val ℓ j = saltRow zp.proof.ligerito sch ℓ j) → os.inner r.val = innerOpensOf zp.inner →
    ∀ pend : ZkLink.PendZ 𝔽 Dig (List UInt8),
      Plays fOf (ZkLink.repZCAt execArith hs512 T pts r) (repMsgsZ st sch (2 + extras.size) tPad zp)
        (rounds.toList.map (·.coins)) pend →
      pend.lig.verdict os.rows = true ∧ pend.inner (os.inner r.val) ∧
      ZkLink.OpensOKZ H512 enc512 leaf512 (ZkLink.width₀ T) cap₀ r.val pend.lig os.rows os.salts

def ZkTableVerdict : Prop :=
  ∀ {n mPts : ℕ} {Reg : Type} (T : Finset (Fin n) → Reg → ZkSession.TabZK 𝔽 mPts) (S : Finset (Fin n)) (R : Reg)
    (pts : Fin 2 → Fin mPts → 𝔽) (spec : SessionSpec) (st : Setup) (sch : Flock.Schedule) (tPad : ℕ)
    (sess : RunRecord) (i : ℕ) (b₀ b₁ c : ByteArray),
    Zk.schedule st.wideLeaf st.m = .ok (sch, tPad) → st.merkle = MerkleScheme.hm96Sha512 → ZkTabSeam (T S R) st sch tPad →
    ZkTabLay (T S R) st sch → FoldRealizes st.fold (T S R).S →
    ExtraRealizes st (T S R).S sess.points (Model.extraClaims (T S R).S (T S R).ptLocal mPts pts) →
    (∀ s ∈ sess.streams.toList, RoundsWF s.rounds) →
    Zk.verifyRep spec st sch tPad sess i 0 b₀ = .ok c → Zk.verifyRep spec st sch tPad sess i 1 b₁ = .ok c →
    ∃ (s₀ s₁ : StreamRec) (zp₀ zp₁ : Zk.ZkProof),
      sess.streams.find? (fun x => x.name == toString spec.tables[i]! ++ toString "/rep" ++ toString 0) = some s₀ ∧
      sess.streams.find? (fun x => x.name == toString spec.tables[i]! ++ toString "/rep" ++ toString 1) = some s₁ ∧
      Zk.decode (shapeZKn st sch (Model.extraClaims (T S R).S (T S R).ptLocal mPts pts).length) b₀ = .ok zp₀ ∧
      Zk.decode (shapeZKn st sch (Model.extraClaims (T S R).S (T S R).ptLocal mPts pts).length) b₁ = .ok zp₁ ∧
      (∃ pend : ZkLink.PendZ 𝔽 Dig (List UInt8) × ZkLink.PendZ 𝔽 Dig (List UInt8),
        Plays fOf (ZkLink.repZCAt execArith hs512 (T S R) pts 0)
          (repMsgsZ st sch (2 + (Model.extraClaims (T S R).S (T S R).ptLocal mPts pts).length) tPad zp₀)
          (s₀.rounds.toList.map (·.coins)) pend.1 ∧
        Plays fOf (ZkLink.repZCAt execArith hs512 (T S R) pts 1)
          (repMsgsZ st sch (2 + (Model.extraClaims (T S R).S (T S R).ptLocal mPts pts).length) tPad zp₁)
          (s₁.rounds.toList.map (·.coins)) pend.2) ∧
      ∀ pend : ZkLink.PendZ 𝔽 Dig (List UInt8) × ZkLink.PendZ 𝔽 Dig (List UInt8),
        Plays fOf (ZkLink.repZCAt execArith hs512 (T S R) pts 0)
          (repMsgsZ st sch (2 + (Model.extraClaims (T S R).S (T S R).ptLocal mPts pts).length) tPad zp₀)
          (s₀.rounds.toList.map (·.coins)) pend.1 →
        Plays fOf (ZkLink.repZCAt execArith hs512 (T S R) pts 1)
          (repMsgsZ st sch (2 + (Model.extraClaims (T S R).S (T S R).ptLocal mPts pts).length) tPad zp₁)
          (s₁.rounds.toList.map (·.coins)) pend.2 →
        accZ T S R (outZ hs512 T S R (fun _ => capOfRoot sess.rootB[i]!) pend
          (opnZ st sch (Model.extraClaims (T S R).S (T S R).ptLocal mPts pts).length #[b₀, b₁])) = true

end FlockSoundness.Discharge.ZkExec
