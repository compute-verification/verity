import FlockSoundness.Protocol.Discharge.Exec.Decode
import FlockSoundness.Protocol.Discharge.Exec.Live
import FlockSoundness.Protocol.ExecTyped
import FlockSoundness.Protocol.ExecSetup

/-!
# The salted table's verdict at flock-verify's acceptance (the verdict part of `Decodes`)

`Decodes` asks two things of an accepted live outcome: each rep's game runs on the tape decoded from its stream and
proof file, and the salted table's verdict `tableOutL`, at the cap `capOf` reads off `root_B` and the openings and salts
`opn` reads off the proof files, accepts on those runs. This file fixes `opn` and proves the verdict part on the reps'
runs that `verify_refines_hm96` gives, which take the proof's messages and the stream's coins as two lists:

* `opnOf`: the openings and salts the two proof files decode to (`opensOf`, `saltsOf`); `opnOf_pair`.
* `table_verdict_hm96`: when both reps of table `i` verify with one cap under `hm96-sha512/v1`, their runs end in pends
  on which `tableOutL` at `capOfRoot root_B[i]` and `opnOf` accepts.
* `accepts_verdict_hm96`: the same at flock-verify's acceptance, for every table of the session, under tags whose leaf
  scheme is `hm96-sha512/v1`.

What remains of `Decodes` at an accepted outcome is that those runs are runs on the decoded tapes (`repTape`).
`Plays.toT` turns each into a run on one tape with the proof's messages and the stream's coins; the tape alignment is
that the messages between coins `j − 1` and `j` are the ones `dec j` reads off round `j`'s retained bytes, which are the
encodings of chunk `j` of the rep's frames (`accepts_rep_rounds`).
-/

set_option Elab.async false

namespace FlockSoundness.Discharge.Exec

open Flock Refine Model Game

/-- **The openings and salts the proof files decode to**: rep `r`'s proof file `bs[r]`, decoded at the table's shape
(the default proof when it does not decode), read as the game's openings (`opensOf`) and salts (`saltsOf`). -/
def opnOf (st : Setup) (sch : Flock.Schedule) (n : ℕ) (bs : Array ByteArray) : Opens 𝔽 Dig × Salts (List UInt8) :=
  let p : ℕ → Proof := fun r => ((decodeProof (shapeOf st sch n) bs[r]!).toOption).getD default
  (opensOf (p 0) (p 1) sch, saltsOf (p 0) (p 1) sch)

end FlockSoundness.Discharge.Exec
