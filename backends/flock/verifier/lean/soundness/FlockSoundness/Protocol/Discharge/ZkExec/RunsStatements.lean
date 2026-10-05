import FlockSoundness.Protocol.Discharge.ZkLink.StrictZ
import FlockSoundness.Protocol.Discharge.ZkLink.DrawOk
import Std.Data.TreeMap.Raw.Lemmas
import FlockSoundness.Protocol.Discharge.DecodeZL.LIG
import FlockSoundness.Protocol.Discharge.DecodeZL.OP

/-!
# `RunsZC`'s decoder half: the `--zk` rep's frames and the messages they carry (statements)

`RunsZC` (`Coin.lean`) asks each rep for a `RepRun` along `repShapeZ`: a run on its rounds' coins whose messages, cut
at the spine's coins, are the decoder's groups of the rounds. At M0 the decoder reads each round's frames by its wires
(`DecodeZL.decOf`), and `DecodeZL.repRun_at` glues the phases' chunk lemmas (`zl_carries`, `op_carries`,
`lig_carries`). Under `--zk` the rep's frames (`repFramesZ`) differ from M0's in four places, and the rest is M0's:

* round 0 also absorbs the pads' label and root before the zerocheck's first coin (`zcHeadZ`), so it carries the root;
* Ligerito absorbs `LABEL_OPEN` after the cap, and its last lane round's coin is followed by level 0's hiding frames
  (`l0Fr`: the pair's sums, the pair's two coins `ρ`, `LABEL_CLOSE`, `T'` and `ybar`) before the round's message
  (`ligFramesZ`);
* the inner proof follows Ligerito (`innerFr`): `final_c'`, `ε`, `[ρ, σ]`, `γ`, `τ`'s commitment, `β`, `Y` and the
  queries.

The decoder `decZ` reads round 0's root at its fixed place before the closing squeeze (`rootOf`), and every later round
by its wires (`wRestZ`: M0's after round 0, the opening's, `wLIGZ` and `wInner`).

* `ZkScheduleSeams`: the `--zk` schedule's dimensions are the ones the frames and wires read.
* `ZkRepFrames`: `repProg` (the transcript program `Zk.verifyRep` runs) emits `repFramesZ`, one chunk per round.
* `ZkDecodeSizes`: a decoded `--zk` proof has M0's lengths (`DecodeZL.ProofSizes`), and `Y` fewer than `2^64` words.
* `ZkCarries`: the frames after round 0, read by `wRestZ`, are the messages after the root, cut at the spine's coins.
* `ZkCut`: on rounds that retained `repFramesZ`'s chunks, the rep's messages, cut along `shapeZE` (`repShapeZ` at the
  executable's dimensions), are `decZ`'s groups of the rounds, with no tail: `RepRun`'s cut.
-/

namespace FlockSoundness.Discharge.ZkExec

open Flock Refine Exec Game Model DecodeZL

/-! ## The frames -/

/-! ## The messages -/

/-! ## The wires and the decoder -/

/-- The pair's sums `e`, from their four values. -/
def wEv : DecodeZL.Wire := .vals (Fin 2 → 𝕂) 4 fun vs => ![gOf ⟨vs[0]!, vs[1]!⟩, gOf ⟨vs[2]!, vs[3]!⟩]

/-- An element of `K`, from its two values. -/
def wK : DecodeZL.Wire := .vals 𝕂 2 fun vs => gOf ⟨vs[0]!, vs[1]!⟩

/-- `t` elements of `K`, from `2t` values. -/
def wY (t : ℕ) : DecodeZL.Wire := .vals (Fin t → 𝕂) (2 * t) fun vs i => gOf ⟨vs[2 * i.val]!, vs[2 * i.val + 1]!⟩

/-- A 64-byte digest, from one byte string. -/
def wDig : DecodeZL.Wire := .bytes Dig toDig

/-- **The `--zk` Ligerito wires**: M0's (`wLIG`) with the last lane round's coin followed by the pair's sums' round,
the second `ρ` coin's empty round, and a round that carries `T'` and `ybar` before the round's message and the level-1
cap. -/
noncomputable def wLIGZ (s : Accounting.Schedule) (t : ℕ) : List (List DecodeZL.Wire) :=
  [[]] ++ ([[.f]] ++ (List.replicate s.k0 [.kk] ++ ([[wEv], [], [wK, wY t, .kk, .cap]] ++
    (wMid ++ wLev s.ext (s.levels.length - 1)))))

/-- The inner proof's wires: `final_c'`'s round, `[ρ, σ]`'s, `τ`'s commitment's and `Y`'s (`nY` words). -/
def wInner (nY : ℕ) : List (List DecodeZL.Wire) := [[.f], [.ff], [wDig], [.v nY]]

/-- **The wires of a `--zk` rep's rounds after round 0**: M0's zerocheck's and lincheck's after round 0, the opening's
of `n` claims, Ligerito's with level 0's hiding (`ybar` of `t` elements), and the inner proof's on `np` pads. -/
noncomputable def wRestZ (m kLog n : ℕ) (s : Accounting.Schedule) (t np : ℕ) : List (List DecodeZL.Wire) :=
  (wZL m kLog).tail ++ (wOP n ++ (wLIGZ s t ++ wInner (np + Zk.K_PAD)))

/-- **The pads root in round 0's bytes**: the 64 bytes before the closing squeeze's 16-byte header, after the root's
own 16-byte header (a 64-byte string needs no padding). -/
def rootOf (B : ByteArray) : ByteArray := B.extract (B.size - 80) (B.size - 16)

/-- **The `--zk` decoder**: round 0 carries the pads root, and round `j + 1` is read by `W`'s `j`-th wires. -/
noncomputable def decZ (W : List (List DecodeZL.Wire)) : ℕ → ByteArray → List Msg
  | 0, B => [⟨Dig, toDig (rootOf B)⟩]
  | j + 1, B => decOf W j B

/-! ## Statements -/

end FlockSoundness.Discharge.ZkExec
