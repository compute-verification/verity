import Proofs.Flock.Soundness.Discharge.ZkExec.PlaysStatements
import Proofs.Flock.Soundness.Discharge.ZkExec.Coin
import Proofs.Flock.Soundness.Discharge.Exec.Verdict

/-!
# `RunsZC`'s verdict half: the compiled `--zk` table accepts at the two reps' results (statements)

`RunsZC` (`Coin.lean`) ends with `accZ`: the compiled table's verdict, at the cap `root_B` decodes to and the openings
the proof files decode to, holds at the two reps' results. The results are those of runs played with each rep's
messages and its stream's coins; a run has one result (`plays_det`), and the statements hold at every one.

The compiled table is the executable's under `hm96-sha512/v1`:
* `hs512`: SHA-512 nodes, and `hm96` leaves at the executable's key (`Refine.K512`) for the rows (`leaf512`), the pads
  pairs (`encPair`: `a.toBytes ‖ b.toBytes`) and `τ` (`encTau`: `TAU_TAG ‖ τ.toBytes`);
* `opnZ`: the openings the two proof files decode to: every level's rows and salts (`opensOf`, `saltsOf`) and each
  rep's inner-proof openings (`innerOpensOf`);
* `ZkTabLay`: the table's padding constant, pads and inner code are the executable's: `PAD_C`, the pads layout's
  positions, `innerN` columns, column `j` at the point `j + 1` and query position `w.lo mod N`.

The statements:
* `ZkTableVerdict`: when both reps of table `i` pass `Zk.verifyRep` with one cap, their compiled reps, played with
  their proof files' messages and their streams' coins, have results, and `accZ` accepts at every pair of them.
* `ZkRepAccepts`: one rep's half: when a run of `repProg` consumes every round, every result of the compiled rep has a
  Ligerito verdict that accepts the proof's rows, an inner check that holds on its inner openings, and row paths that
  verify (`OpensOKZ`).
* `ZkInnerAccepts`: the inner proof's half of `ZkRepAccepts`: when `Zk.verifyInner` accepts seven constraints, at rows
  that are those constraints in their order (`LinRel`), every result of `innerGameC` is a check that holds on the
  proof's openings.
-/

namespace FlockSoundness.Discharge.ZkExec

open Flock Refine Exec Game Model DecodeZL

local notation "𝔽" => FlockLevel3.GF128

instance digInhabitedZ : Inhabited Dig := ⟨H512 []⟩

/-! ## The hashes -/

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

/-! ## The openings -/

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

/-! ## The table -/

/-- **A compiled `--zk` table's pads and inner code are the executable's** (`Zk.verifyRep`'s, at the pads layout
`layZ st sch`): the padding constant `PAD_C`; level 0's, the algebraic phase's and the triple's pads at the layout's
positions; `innerN` columns, column `j` at the point `j + 1` (`codeAt`) and a query word `w` at `w.lo mod N`. -/
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

/-- **A model row is an executable constraint**: the row `Σ coef_i·h_i + const` on `np` pads is the constraint
`c + Σ a_i·h_i` (`Zk.Lin`), coefficient by coefficient, with `a` of `np` entries. -/
def LinRel {np : ℕ} (ℓ : ZkSession.Lin 𝔽 np) (x : Zk.Lin) : Prop :=
  x.a.size = np ∧ ℓ.const = fOf x.c ∧ ∀ i : Fin np, ℓ.coef i = fOf x.a[i.val]!

/-! ## Statements -/

/-- **The inner proof's check holds on its openings** (live PROTOCOL.md §5). When `Zk.verifyInner` succeeds from `t` to
`t'` on seven constraints and well-formed rounds, with every path entry 64 bytes (as decoded), then at a compiled
table whose dimensions, pads and inner code are the executable's and at rows that are the seven constraints in their
order (`LinRel`), every result of the model's compiled inner proof, played with the proof's `[ρ, σ]`, `τ`'s commitment
and `Y` on the coins of the rounds consumed, is a check that holds on the proof's inner openings. -/
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

/-- **A `--zk` rep's result accepts** (live PROTOCOL.md §1–§7, PROTOCOL.md §14). At the `--zk` schedule under
`hm96-sha512/v1`, at a compiled table whose dimensions (`ZkTabSeam`), pads and inner code (`ZkTabLay`) are the
executable's, whose statement's fold the setup's circuit fold realizes and whose extra claims at the link points are
the executable's, on a decoded proof and well-formed rounds, when a run of rep `r`'s transcript program consumes every
round, every result of the compiled rep, played with the rep's messages and the rounds' coins, is one where, at
openings whose rep-`r` rows, salts and inner openings are the proof's and at the level-0 cap the proof's cap reads as,
its Ligerito verdict accepts the rows, its inner check holds, and every row it requested verifies (`OpensOKZ`). -/
def ZkRepAccepts : Prop :=
  ∀ {mPts : ℕ} (T : ZkSession.TabZK 𝔽 mPts) (pts : Fin 2 → Fin mPts → 𝔽) (r : Fin 2) (st : Setup)
    (sch : Flock.Schedule) (tPad : ℕ) (extras : Array Flock.Claim) (bytes : ByteArray) (zp : Zk.ZkProof)
    (rounds : Array Round) (dom : ByteArray) (tr : Flock.Transcript) (cap₀ : Cap Dig)
    (os : ZkLink.OpensZ 𝔽 Dig (List UInt8)),
    Zk.schedule st.m = .ok (sch, tPad) → st.merkle = MerkleScheme.hm96Sha512 → ZkTabSeam T st sch tPad →
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

/-- **The compiled `--zk` table accepts at the two reps' results, under `hm96-sha512/v1`** (live PROTOCOL.md §1–§7,
PROTOCOL.md §5, §14, S16). When both reps of table `i` pass `Zk.verifyRep` with one cap, at a compiled table whose
dimensions, pads and inner code are the executable's, whose statement's fold the setup's circuit fold realizes and
whose extra claims at the link points `pts` are the session's, each rep's stream and proof file decode, the two reps'
compiled games, played with their proof files' messages and their streams' coins, have results, and at every pair of
results they reach the table's verdict `accZ`, at the cap `capOfRoot` reads off the table's `root_B` and the openings
`opnZ` reads off the two proof files, accepts. -/
def ZkTableVerdict : Prop :=
  ∀ {n mPts : ℕ} {Reg : Type} (T : Finset (Fin n) → Reg → ZkSession.TabZK 𝔽 mPts) (S : Finset (Fin n)) (R : Reg)
    (pts : Fin 2 → Fin mPts → 𝔽) (spec : SessionSpec) (st : Setup) (sch : Flock.Schedule) (tPad : ℕ)
    (sess : RunRecord) (i : ℕ) (b₀ b₁ c : ByteArray),
    Zk.schedule st.m = .ok (sch, tPad) → st.merkle = MerkleScheme.hm96Sha512 → ZkTabSeam (T S R) st sch tPad →
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
