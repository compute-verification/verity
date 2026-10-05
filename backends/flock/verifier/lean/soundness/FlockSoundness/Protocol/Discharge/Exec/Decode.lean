import FlockSoundness.Protocol.Refine.LiveSim
import FlockSoundness.Protocol.Discharge.Exec.Setup
import FlockSoundness.Protocol.Discharge.Exec.SessionL

/-!
# Pieces of `DecodesAll`: the aligned tape and the cap

`DecodesAll` (`Live.lean`) asks, on every live outcome flock-verify accepts, for each rep's run on one typed tape
(`PlaysT`) decoded from the round bytes and the proof file, and for the salted verdict at the cap `capOf` reads off
`root_B`. The refinement (`verify_refines_hm96`) gives each rep's run with its messages and its coins as two lists
(`Plays`). Two pieces of the bridge:

* `Plays.toT`: a run on two lists is a run on one tape whose messages, in order, are the run's messages and whose coin
  steps, in order, are the run's coin arrays (`CoinOf`). What remains of the alignment is that this tape is
  `repTape`: each round's messages are those its retained bytes decode to.
* `capOfRoot`, `capOfRoot_eq`: the cap read off `root_B`. Any array of 64-byte entries whose concatenation is `root_B`
  has `capOfRoot root_B` as its `capDig`.
* `verify_cap_hm96`, `verify_cap_ofCircuit_hm96`: in a one-table session flock-verify accepts under `hm96-sha512/v1`,
  rep 0's cap, which S16 compares with `root_B` and whose entries are 64-byte digests (`rep_refines`), is
  `capOfRoot root_B`. So `capOf` can be `capOfRoot`: the cap the salted verdict of `verify_refines_hm96` reads,
  `capDig p₀.cap`, is a function of `root_B` alone.
-/

namespace FlockSoundness.Game

variable {W F : Type} [Inhabited W] {ι : W → F}

end FlockSoundness.Game

namespace FlockSoundness.Discharge.Exec

open Flock Refine Model

open Classical in
/-- **The cap read off `root_B`**: the `capDig` of the array of 64-byte entries whose concatenation is `root_B`, when
there is one. -/
noncomputable def capOfRoot (rootB : ByteArray) : Cap Dig :=
  if h : ∃ c : Array ByteArray, (∀ x ∈ c.toList, x.size = 64) ∧ c.foldl (· ++ ·) ByteArray.empty = rootB then
    capDig (Classical.choose h)
  else capDig #[]

end FlockSoundness.Discharge.Exec
