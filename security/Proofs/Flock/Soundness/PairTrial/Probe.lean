module

public import Proofs.Flock.Soundness.PairTrial.Leaf
public import Proofs.Flock.Soundness.PairTrial.Session
public import Proofs.Flock.Soundness.PairTrial.Congr

/-!
# Trial: which axioms each function of the trial uses

A scratch, not to merge: `#print axioms` on the trial's definitions, read in the try's output.
-/

@[expose] public section

#print axioms FlockSoundness.Merkle.walkPair
#print axioms FlockSoundness.Merkle.openPair
#print axioms FlockSoundness.Merkle.openPairL
#print axioms FlockSoundness.Merkle.hm96Pair
#print axioms FlockSoundness.Merkle.LeafP.plain
#print axioms FlockSoundness.Merkle.LeafP.hm96
#print axioms FlockSoundness.StrictCR.collOfP
#print axioms FlockSoundness.StrictCR.onePickP
#print axioms FlockSoundness.StrictCR.conflictPickP
#print axioms FlockSoundness.StrictCR.collOfP_isColl
#print axioms FlockSoundness.Discharge.Exec.firstL
#print axioms FlockSoundness.Discharge.Exec.opOfLPair
#print axioms FlockSoundness.Discharge.ZkLink.opRZPair
#print axioms FlockSoundness.Discharge.ZkLink.advRSFinderZP
#print axioms FlockSoundness.Discharge.ZkLink.advRSFinderZ
#print axioms FlockSoundness.Discharge.Recursion.clashPickCLP
#print axioms FlockSoundness.Discharge.Recursion.clashPickHLP
