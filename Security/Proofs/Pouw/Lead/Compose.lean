import Proofs.Pouw.Lead.Compose.G
import Proofs.Pouw.Lead.Compose.Honest
import Proofs.Pouw.Lead.Compose.Deployed

/-!
# The lead's guarantee, composed from its pieces

Its proofs go in `Lead/Compose/`, its declarations in `Pouw.SecurityProofs.Lead.Compose`, and each file is imported
here: `G`, one coin value's floor (`perCoin`) and the lead's guarantee from it and the other pieces (`G`); `Honest`,
the honest strategy meeting every premise but `CommitL1w` at every coin stream and commitment format (`honestAt`), the
honest lemma at every stream and commitment meeting `CommitL1w` (`honestOfCommit`), `CommitL1w` at a commitment whose
messages include frame-v3 leaves over the words and A's rows (`commitL1w_of_leaves`), and the honest lemma (`honest`);
`Deployed`, `CommitL1w` at the deployed commitment (`commitL1w_l1w`) and the honest lemma at the deployed commitment
and the certifier's stream (`honestL1w`).
-/
