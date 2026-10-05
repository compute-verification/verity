import FlockSoundness.Protocol.Discharge.ZkExec.RunsStatements

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

/-! ## Statements -/

end FlockSoundness.Discharge.ZkExec
