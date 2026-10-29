import Proofs.Pouw.Lead.Compose.HashCompose
import Proofs.Pouw.Lead.Compose.Inventory
import Proofs.Pouw.Lead.Compose.Sketch
import Proofs.Pouw.Lead.Compose.Guarantee
import Proofs.Pouw.Lead.Compose.Refute

/-!
# The lead's guarantee, composed from its pieces

The lead's guarantee from its pieces: the matmul floor, hash-compose under ShaWrites, the network term and the period inventory.

Its proofs go in `Lead/Compose/`, its declarations in `Pouw.SecurityProofs.Lead.Compose`, and each file is imported
here: `HashCompose`, `hash-compose` in `MH100w` under `ShaWrites`; `Inventory`, the call model, the cards' inventory and
the hosts counted inside T1; `Sketch`, the guarantee's bound composed from the four pieces, each a hypothesis;
`Guarantee`, `Pouw.Guarantees.Lead.G` from the words floor (`G_of_wordsFloor`); `Refute`, a call of one row and
one column where a pre-coin table holds every committed tile and a strategy that writes no word meets every premise of
`G` as stated (`not_G`, `not_wordsFloor`).
-/
