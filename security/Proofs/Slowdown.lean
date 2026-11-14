module

public import Proofs.Slowdown.CutCharged
public import Proofs.Slowdown.Layered
public import Proofs.Slowdown.FitKeep
public import Proofs.Slowdown.WeightsSlotsWitness
public import Proofs.Slowdown.FitRate
public import Proofs.Slowdown.FitWitness
public import Proofs.Slowdown.Capability
public import Proofs.Slowdown.CoveringTwoCostsWitness
public import Proofs.Slowdown.Witnesses
public import Proofs.Slowdown.Formalism
public import Proofs.Slowdown.Cover.Training
/-!
# Proofs of the slowdown bound's guarantees

The cuts and the min-cut bound (`Cut`, `CutCharged`), the work and product counts of threads, fit and layered references,
the cover theorem (`Cover.Training`), covering at one and two costs, the local bounds (`Capability`), the training
formalism (`Formalism`), the slowdown over execution phases (`Slowdown`), the remote bound over monitoring periods
(`RemotePeriod`), and their witnesses. The policy they take as a hypothesis is `Proofs.InferenceOnly`'s.
-/
