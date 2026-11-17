module

public import Proofs.Properties.Poub.Remote.Window
public import Proofs.Properties.Poub.Remote.Sample
public import Proofs.Properties.Poub.Remote.Plan
public import Proofs.Properties.Poub.Remote.Run
public import Proofs.Properties.Poub.Remote.Released
public import Proofs.Properties.Poub.Remote.Lifetime
public import Proofs.Properties.Poub.Remote.Pooled

/-! An accepted PoUB window checks every sampled opening, and the samples are the sampler's subset mapped to distinct
receipts. -/

@[expose] public section
