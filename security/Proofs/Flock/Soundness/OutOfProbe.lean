import Proofs.Flock.Soundness.LinkFinder
import Proofs.Flock.EndToEnd.Carry.Statement
import Proofs.Accounting.JointBridge.Guarantees.IntegrityOfIntegrityBlock

open FlockSoundness LinkFinder

#check @outOf
#check @outOf_spec
#check @nextRead
#check @Flock.Guarantees.CarriedLeaf.op

set_option pp.explicit true
set_option pp.universes true
set_option pp.fullNames true
set_option pp.proofs true
set_option pp.deepTerms true
set_option pp.maxSteps 100000000
set_option pp.oneline false
set_option maxRecDepth 100000

#print Flock.Guarantees.IntegrityBlockCarried
#print Flock.Guarantees.IntegrityHorizonCarried
#print Accounting.Guarantees.Bridge.IntegrityOfIntegrityBlock
