import Proofs.Pouw.Lead.Compose.G
import Proofs.Pouw.Lead.Compose.Guarantee
import Proofs.Pouw.Lead.Compose.HashCompose
import Proofs.Pouw.Lead.Compose.Inventory
import Proofs.Pouw.Lead.Compose.Sketch

/-!
# The lead's guarantee, composed from its pieces

The lead's guarantee from its pieces: the matmul floor, hash-compose under ShaWrites, the network term and the period inventory.

Its proofs go in `Lead/Compose/`, its declarations in `Pouw.SecurityProofs.Lead.Compose`, and each file is imported
here: `HashCompose`, `hash-compose` in `MH100w` under `ShaWrites`; `Inventory`, the call model, the cards' inventory and
the hosts counted inside T1; `Sketch`, the guarantee's bound composed from the four pieces, each a hypothesis;
`Guarantee`, `Pouw.Guarantees.Lead.G` from the words floor (`G_of_wordsFloor`).
-/
