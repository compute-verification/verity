import Proofs.Pouw.Lead.Bridge
import Proofs.Pouw.Lead.CoinBranch
import Proofs.Pouw.Lead.Commit
import Proofs.Pouw.Lead.Compose
import Proofs.Pouw.Lead.MH100W
import Proofs.Pouw.Lead.Network
import Proofs.Pouw.Lead.PStar
import Proofs.Pouw.Lead.Statement
import Proofs.Pouw.Lead.Tags
import Proofs.Pouw.Lead.Tail

/-!
# The PoUW lead's proofs

The proofs behind the lead design's guarantee, one piece each. A piece's proofs are in `Lead/<Piece>/`, its
declarations in `Pouw.SecurityProofs.Lead.<Piece>`, and its aggregator `Lead/<Piece>.lean` imports each of its files.
This module imports the aggregators and nothing else, and `Proofs.Pouw` imports it once.
-/
