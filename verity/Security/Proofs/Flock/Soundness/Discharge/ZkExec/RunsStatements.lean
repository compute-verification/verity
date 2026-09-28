import Proofs.Flock.Soundness.Discharge.ZkExec.Coin
import Proofs.Flock.Soundness.Discharge.DecodeZL.LIG
import Proofs.Flock.Soundness.Discharge.DecodeZL.OP
import Proofs.Flock.Soundness.Discharge.DecodeZL.Runs

/-!
# `RunsZC`'s decoder half: the `--zk` rep's frames and the messages they carry (statements)

`RunsZC` (`Coin.lean`) asks each rep for a `RepRun` along `repShapeZ`: a run on its rounds' coins whose messages, cut
at the spine's coins, are the decoder's groups of the rounds. Without `--zk` the decoder reads each round's frames by
its wires (`DecodeZL.decOf`), and `DecodeZL.repRun_at` glues the phases' chunk lemmas (`zl_carries`, `op_carries`,
`lig_carries`). Under `--zk` the rep's frames (`repFramesZ`) differ from the non-ZK session's in four places, and the
rest is the non-ZK session's:

* round 0 also absorbs the pads' label and root before the zerocheck's first coin (`zcHeadZ`), so it carries the root;
* Ligerito absorbs `LABEL_OPEN` after the cap, and its last lane round's coin is followed by level 0's hiding frames
  (`l0Fr`: the pair's sums, the pair's two coins `ρ`, `LABEL_CLOSE`, `T'` and `ybar`) before the round's message
  (`ligFramesZ`);
* the inner proof follows Ligerito (`innerFr`): `final_c'`, `ε`, `[ρ, σ]`, `γ`, `τ`'s commitment, `β`, `Y` and the
  queries.

The decoder `decZ` reads round 0's root at its fixed place before the closing squeeze (`rootOf`), and every later round
by its wires (`wRestZ`: the non-ZK session's after round 0, the opening's, `wLIGZ` and `wInner`).

* `ZkScheduleSeams`: the `--zk` schedule's dimensions are the ones the frames and wires read.
* `ZkRepFrames`: `repProg` (the transcript program `Zk.verifyRep` runs) emits `repFramesZ`, one chunk per round.
* `ZkDecodeSizes`: a decoded `--zk` proof has the non-ZK session's lengths (`DecodeZL.ProofSizes`), and `Y` fewer than
  `2^64` words.
* `ZkCarries`: the frames after round 0, read by `wRestZ`, are the messages after the root, cut at the spine's coins.
* `ZkCut`: on rounds that retained `repFramesZ`'s chunks, the rep's messages, cut along `shapeZE` (`repShapeZ` at the
  executable's dimensions), are `decZ`'s groups of the rounds, with no tail: `RepRun`'s cut.
-/

namespace FlockSoundness.Discharge.ZkExec

open Flock Refine Exec Game Model DecodeZL

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

/-! ## The frames -/

/-- Round 0 of a `--zk` rep after its domain (live PROTOCOL.md §1, §4): §9.1's binding, the zerocheck's label, the pads'
label and root, and the zerocheck's first coin. -/
def zcHeadZ (digest cap root : ByteArray) : List Frame :=
  [.label "flock-r1cs-v0".toUTF8, .bytes digest, .bytes cap, .label "flock-zerocheck-v0".toUTF8,
    .label Zk.LABEL_PADS.toUTF8, .bytes root, .sq 2 6]

/-- The rest of the masked zerocheck: the non-ZK session's frames after its first round (`zcFrames`). -/
def zcTailZ (m : ℕ) (p : ZerocheckProof) : List Frame :=
  [.sq 2 (m - 13), .vec p.round1ab.toList, .vec p.round1c.toList, .sq 1 1] ++
  (List.range' 0 (m - 6)).flatMap (fun i => [.one (p.rounds[i]!).1, .one (p.rounds[i]!).2, .sq 1 1]) ++
  [.one p.finalA, .one p.finalB]

/-- Level 0's hiding frames after the last lane round's coin (live PROTOCOL.md §7): the pair's sums, the pair's coins
`ρ_0, ρ_1`, the close label, `T'` and `ybar`. -/
def l0Fr (z : Zk.Level0Msgs) : List Frame :=
  [.vec [z.e0.c0, z.e0.c1], .vec [z.e1.c0, z.e1.c1], .sq 2 2, .sq 2 2, .label Zk.LABEL_CLOSE.toUTF8,
    .vec [z.tPrime.c0, z.tPrime.c1], .vec (z.ybar.foldl (fun acc v => (acc.push v.c0).push v.c1) #[]).toList]

/-- **A `--zk` Ligerito run's frames** (live PROTOCOL.md §7): the non-ZK session's (`ligFrames`) with `LABEL_OPEN` after
the cap and level 0's hiding frames between the last lane round's coin and its message. -/
def ligFramesZ (sch : Flock.Schedule) (T : F128) (cap : Array ByteArray) (p : LigeritoProof) (z : Zk.Level0Msgs) :
    List Frame :=
  [.label "flock-ligerito-basis-f256-split-v0".toUTF8, .one T, capFr cap, .label Zk.LABEL_OPEN.toUTF8] ++
    ([.sq 2 sch.n, .one p.oodValues[0]!, .bytes (le64 p.claimNonces[0]!), .sq 1 1] ++ kFr p.messages[0]! ++
      (chFr p 1 (sch.k0 - 1) ++ .sq 2 2 :: (l0Fr z ++ kFr p.messages[sch.k0]!)) ++ capFr p.recursiveCaps[0]! ::
      (introFr p (sch.n - sch.k0 + 1) 1 (sch.k0 + 1) 1 ++ introFr p (sch.n - sch.k0 + 1) 2 (sch.k0 + 2) 2 ++
        queryFr p (sch.levels.getD 0 default) 0 0 ++ kFr p.messages[sch.k0 + 3]! ++
        [.bytes (le64 p.claimNonces[3]!), .sq 1 1] ++ (List.range' 0 (sch.levels.size - 1)).flatMap (levelFr sch p)))

/-- **The inner proof's frames** (live PROTOCOL.md §5), on `nγ` constraints with the triple's: its label, `final_c'`,
`ε`, `[ρ, σ]`, `γ`, `τ`'s commitment, `β`, `Y` and the `Q` query words. -/
def innerFr (pr : Zk.InnerProof) (finalC : F128) (nγ : ℕ) : List Frame :=
  [.label Zk.LABEL_INNER.toUTF8, .one finalC, .sq 1 1, .vec [pr.rho, pr.sigma], .sq 2 nγ, .bytes pr.tauCommit, .sq 1 1,
    .vec pr.y.toList, .sq 2 Zk.Q]

/-- **A `--zk` rep's frames after round 0**: the masked zerocheck's rest, the lincheck, the ring switch of `n` claims,
Ligerito with level 0's hiding, and the inner proof on its ten constraints (seven and the triple's three). -/
def restFramesZ (st : Setup) (sch : Flock.Schedule) (n : ℕ) (zp : Zk.ZkProof) (T : F128) : List Frame :=
  zcTailZ st.m zp.proof.zerocheck ++ (lcFrames st.kLog zp.proof.lincheck ++ (rsFrames n zp.proof.ringSwitches ++
    (ligFramesZ sch T zp.proof.cap zp.proof.ligerito zp.level0 ++ innerFr zp.inner zp.proof.zerocheck.finalC 10)))

/-- **A `--zk` rep's frames** (live PROTOCOL.md §1–§7): its domain, round 0's binding and pads root, and the rest. -/
def repFramesZ (st : Setup) (sch : Flock.Schedule) (domain : ByteArray) (n : ℕ) (zp : Zk.ZkProof) (T : F128) :
    List Frame :=
  .dom domain :: (zcHeadZ st.digest (zp.proof.cap.foldl (· ++ ·) ByteArray.empty) zp.inner.root ++
    restFramesZ st sch n zp T)

/-! ## The messages -/

/-- The pair's sums `e` as the model's message. -/
def evMsg (z : Zk.Level0Msgs) : Msg := ⟨Fin 2 → 𝕂, ![gOf z.e0, gOf z.e1]⟩

/-- The blinded claim `T'` as the model's message. -/
def tMsg (z : Zk.Level0Msgs) : Msg := ⟨𝕂, gOf z.tPrime⟩

/-- The padding's combination `ybar`, `t` elements of `K`, as the model's message. -/
def ybarMsg (t : ℕ) (z : Zk.Level0Msgs) : Msg := ⟨Fin t → 𝕂, fun i => gOf z.ybar[i.val]!⟩

/-- **The messages of a `--zk` Ligerito run**: the non-ZK session's (`ligMsgs`) with the pair's sums, `T'` and `ybar`
between the last lane round's coins and its message. -/
def ligMsgsZ {D : Type} (p : LigeritoProof) (capD : Array ByteArray → Cap D) (sch : Flock.Schedule) (t : ℕ)
    (z : Zk.Level0Msgs) : List Msg :=
  fMsg p.oodValues[0]! :: kMsg p.messages[0]! :: kMsgs p 1 (sch.k0 - 1) ++ evMsg z :: tMsg z :: ybarMsg t z ::
    kMsg p.messages[sch.k0]! :: ⟨Cap D, capD p.recursiveCaps[0]!⟩ ::
    [fMsg p.oodValues[1]!, kMsg p.messages[sch.k0 + 1]!, fMsg p.oodValues[2]!, kMsg p.messages[sch.k0 + 2]!] ++
    [kMsg p.messages[sch.k0 + 3]!] ++ recMsgs p capD sch.ext (sch.levels.size - 1) 0 (sch.k0 + 4) 3

/-- **The inner proof's messages**: `final_c'`, `[ρ, σ]`, `τ`'s commitment and `Y` of `nY` words. -/
def innerMsgs (pr : Zk.InnerProof) (finalC : F128) (nY : ℕ) : List Msg :=
  [fMsg finalC, ⟨𝔽 × 𝔽, (fOf pr.rho, fOf pr.sigma)⟩, ⟨Dig, toDig pr.tauCommit⟩, ⟨Fin nY → 𝔽, vecOf nY pr.y⟩]

/-- The pads' layout of a `--zk` rep (`Zk.verifyRep`'s). -/
def layZ (st : Setup) (sch : Flock.Schedule) : Zk.PadLayout := Zk.PadLayout.new st.m st.kLog sch.k0

/-- **A `--zk` rep's messages after the root**, in the order the compiled rep receives them: the zerocheck's, the
lincheck's, the `n` ring switches, Ligerito's with level 0's hiding (`ybar` of `t` elements), and the inner proof's. -/
def restMsgsZ (st : Setup) (sch : Flock.Schedule) (n t : ℕ) (zp : Zk.ZkProof) : List Msg :=
  zcMsgs st.m zp.proof.zerocheck ++ (lcMsgs st.kLog zp.proof.lincheck ++ (opMsgs n zp.proof.ringSwitches ++
    (ligMsgsZ zp.proof.ligerito capDig sch t zp.level0 ++
      innerMsgs zp.inner zp.proof.zerocheck.finalC ((layZ st sch).len + Zk.K_PAD))))

/-- **A `--zk` rep's messages**: the pads root, then the rest. -/
def repMsgsZ (st : Setup) (sch : Flock.Schedule) (n t : ℕ) (zp : Zk.ZkProof) : List Msg :=
  ⟨Dig, toDig zp.inner.root⟩ :: restMsgsZ st sch n t zp

/-! ## The wires and the decoder -/

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

/-- **A compiled `--zk` rep's shape at the executable's dimensions** (`repShapeZ` with `e = 2` pair sums, `Q` inner
queries and `K_PAD` padding symbols): the pads root, the masked zerocheck's, lincheck's and `n` claims' opening's
shapes, level 0 masked (`ybar` of `t` elements), `final_c'`, and the inner proof on seven rows and `np` pads. -/
def shapeZE (m kLog n : ℕ) (s : Accounting.Schedule) (t np : ℕ) : List Exec.Kind :=
  .msg Dig :: ((zcShape 𝔽 m ++ (lcShape 𝔽 kLog ++ (opShape 𝔽 n ++ (ligPadShape 𝔽 𝕂 Dig s 2 t ++ [.msg 𝔽])))) ++
    innerShape 𝔽 Dig 7 np Zk.K_PAD Zk.Q)

/-! ## Statements -/

/-- **The `--zk` schedule's dimensions**: `Zk.schedule m` succeeds only at `25 ≤ m ≤ 35`, on fast100's dimensions
(`DimSeam`, every level's `LevelSeam`) with at least one lane round, `t_pad = 2 q_0`, per level not one consistency
coin (`ceilLog2 q ≠ 1`), and per level fewer than 64 columns, 1 to 511 queries and a cap of at most 8 (level 0's at
`m = 25`, `q_0 = 277`, is 8). -/
def ZkScheduleSeams : Prop :=
  ∀ (m : ℕ) (sch : Flock.Schedule) (tPad : ℕ), Zk.schedule m = .ok (sch, tPad) →
    25 ≤ m ∧ m ≤ 35 ∧ sch.m = m ∧ 1 ≤ sch.k0 ∧ tPad = 2 * (sch.levels.getD 0 default).q ∧ DimSeam sch ∧
      (∀ i < sch.levels.size, LevelSeam (sch.levels.getD i default)) ∧
      (∀ i < sch.levels.size, Accounting.ceilLog2 (sch.levels.getD i default).q ≠ 1) ∧
      ∀ lv ∈ sch.levels, lv.cols < 64 ∧ 0 < lv.q ∧ lv.q < 512 ∧ lv.cap ≤ 8

/-- **`repProg` emits a `--zk` rep's frames** (live PROTOCOL.md §1–§7): at the `--zk` schedule, a run of rep `r`'s
transcript program from the domain `dom` on rounds that all retained their bytes, consuming every round, consumed one
round per chunk of `repFramesZ` (at the ring switch's batched value `T`), and round `k` retained chunk `k`'s bytes. -/
def ZkRepFrames : Prop :=
  ∀ (st : Setup) (sch : Flock.Schedule) (tPad r : ℕ) (extras : Array Flock.Claim) (zp : Zk.ZkProof)
    (rounds : Array Round) (dom : ByteArray) (tr : Flock.Transcript),
    Zk.schedule st.m = .ok (sch, tPad) →
    (repProg st sch tPad r extras zp).run (Flock.Transcript.init rounds dom) = .ok ((), tr) →
    tr.k = rounds.size → AllRetained rounds →
    ∃ T : F128, (chunks (repFramesZ st sch dom (2 + extras.size) zp T)).1.length = rounds.size ∧
      ∀ k < rounds.size, Retains rounds[k]! (encs ((chunks (repFramesZ st sch dom (2 + extras.size) zp T)).1.getD k []))

/-- **A decoded `--zk` proof's lengths**: the pair's are the non-ZK session's (`ProofSizes`, PROTOCOL.md §8, read by
`Zk.reader` as by `decodeProof`), and the inner proof's `Y` has fewer than `2^64` words (its `u64` length). -/
def ZkDecodeSizes : Prop :=
  ∀ (sh : Shape) (bytes : ByteArray) (zp : Zk.ZkProof), Zk.decode sh bytes = .ok zp →
    ProofSizes sh zp.proof ∧ zp.inner.y.size < 2 ^ 64

/-- **A `--zk` rep's rounds after round 0 carry its messages** (the chunk lemma): at the `--zk` schedule and 64-byte
digests, on a proof with its decoded lengths whose `ybar` has `t_pad` elements, `τ`'s commitment 64 bytes and `Y` the
layout's `len + K_PAD` words, the frames after round 0, read by `wRestZ`, are the messages after the root, cut at
`shapeZE`'s coins after the root and the first coin, with no tail. -/
def ZkCarries : Prop :=
  ∀ (st : Setup) (sch : Flock.Schedule) (tPad : ℕ) (extras : Array Flock.Claim) (zp : Zk.ZkProof) (T : F128),
    Zk.schedule st.m = .ok (sch, tPad) → st.merkle.digestLen = 64 → ProofSizes (shapeZK st sch extras) zp.proof →
    zp.inner.y.size < 2 ^ 64 → zp.level0.ybar.size = tPad → zp.inner.tauCommit.size = 64 →
    zp.inner.y.size = (layZ st sch).len + Zk.K_PAD →
    Carries (wRestZ st.m st.kLog (2 + extras.size) (schedOf sch) tPad (layZ st sch).len)
      ((shapeZE st.m st.kLog (2 + extras.size) (schedOf sch) tPad (layZ st sch).len).drop 2)
      (restFramesZ st sch (2 + extras.size) zp T) (restMsgsZ st sch (2 + extras.size) tPad zp) []

/-- **A `--zk` rep's messages cut into its rounds** (`RepRun`'s cut): at the `--zk` schedule and 64-byte digests, at a
decoded proof whose pads root and `τ`'s commitment have 64 bytes, `ybar` `t_pad` elements and `Y` `len + K_PAD` words,
on rounds that retained the encodings of `repFramesZ`'s chunks, one round per chunk, the rep's messages laid along
`shapeZE` and cut at its coins are `decZ`'s groups of the rounds, with no tail. -/
def ZkCut : Prop :=
  ∀ (st : Setup) (sch : Flock.Schedule) (tPad : ℕ) (extras : Array Flock.Claim) (bytes : ByteArray)
    (zp : Zk.ZkProof) (T : F128) (dom : ByteArray) (rds : Array Round),
    Zk.schedule st.m = .ok (sch, tPad) → st.merkle.digestLen = 64 →
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
