module

public import Proofs.Pouw.Lead.CoinBranch
public import Proofs.Pouw.Lead.Commit
public import Proofs.Pouw.Lead.Compose
public import Proofs.Pouw.Lead.MH100W
public import Proofs.Pouw.Lead.Network
public import Proofs.Pouw.Lead.PStar
public import Proofs.Pouw.Lead.Statement
public import Proofs.Pouw.Lead.Tags
public import Proofs.Pouw.Lead.Tail

/-!
# The PoUW lead's proofs

The proofs behind the lead design's guarantee, one piece each. A piece's proofs are in `Lead/<Piece>/`, its
declarations in `Pouw.SecurityProofs.Lead.<Piece>`, and its aggregator `Lead/<Piece>.lean` imports each of its files.
This module imports the aggregators and nothing else, and `Proofs.Pouw` imports it once.
-/

@[expose] public section
