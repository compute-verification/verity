import Proofs.Flock.EndToEnd.WindowDraw.Statement
import Proofs.Flock.EndToEnd.WindowDraw.Member
import Proofs.Flock.EndToEnd.WindowDraw.Law
import Proofs.Flock.EndToEnd.WindowDraw.Bound
import Proofs.Flock.EndToEnd.WindowDraw.Honest

/-!
# A window's stratified draw, restricted to a member, is the member's `subset` draw

The statements are `Statement`'s (`Flock.Guarantees.WindowDraw`, `WindowBound`, `WindowBound_honest`,
`WindowBound_nontrivial`); `Member` proves the service's split is the member's share, `Law` the stratified law at a
member, `Bound` the composed bound and `Honest` the two witnesses.
-/
