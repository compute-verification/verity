module

public import Proofs.Properties.Pouw.PearlPlus.Noise.Fa
public import Proofs.Properties.Pouw.PearlPlus.Noise.Fb
public import Proofs.Properties.Pouw.PearlPlus.Noise.Gram

/-! The test noise is `fixedNoise`'s at k = 128, from 24 kernel computations in three modules that lake checks in
parallel, so that a build holds one G chunk's and two line chunks' kernel state at a time. -/

@[expose] public section
