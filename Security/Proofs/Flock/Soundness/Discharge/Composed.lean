import Proofs.Flock.Soundness.Discharge.Composed.Sound
import Proofs.Flock.Soundness.Discharge.Composed.Base
import Proofs.Flock.Soundness.Discharge.Composed.Coined
import Proofs.Flock.Soundness.Discharge.Composed.Headline
import Proofs.Flock.Soundness.Discharge.Composed.Coins
import Proofs.Flock.Soundness.Discharge.Composed.Geometry
import Proofs.Flock.Soundness.Discharge.Composed.Pads
import Proofs.Flock.Soundness.Discharge.Composed.Inner
import Proofs.Flock.Soundness.Discharge.Composed.Inner.Holds
import Proofs.Flock.Soundness.Discharge.Composed.Inner.Zerocheck
import Proofs.Flock.Soundness.Discharge.Composed.Inner.ZerocheckMasked
import Proofs.Flock.Soundness.Discharge.Composed.Inner.ZerocheckReplay
import Proofs.Flock.Soundness.Discharge.Composed.Inner.Lincheck
import Proofs.Flock.Soundness.Discharge.Composed.Inner.RingSwitch
import Proofs.Flock.Soundness.Discharge.Composed.Inner.ZerocheckGlue
import Proofs.Flock.Soundness.Discharge.Composed.Inner.LincheckReplay
import Proofs.Flock.Soundness.Discharge.Composed.Setup
import Proofs.Flock.Soundness.Discharge.Composed.RegWords
import Proofs.Flock.Soundness.Discharge.Composed.SoundJ
import Proofs.Flock.Soundness.Discharge.Composed.Custody
import Proofs.Flock.Soundness.Discharge.Composed.CustodyJ
import Proofs.Flock.Soundness.Discharge.Composed.CustodyJJson
import Proofs.Flock.Soundness.Discharge.Composed.PublicJ
import Proofs.Flock.Soundness.Discharge.Composed.DrawnJ
import Proofs.Flock.Soundness.Discharge.Composed.NonVacuousJ
import Proofs.Flock.Soundness.Discharge.Composed.DefsJA

/-!
# One `--zk` sampled session: soundness and zero knowledge composed

The soundness half is in `Composed/Sound.lean`, its pieces in `Composed/Defs.lean`; the composed headline in
`Composed/Headline.lean`, the draw's `Coined` tables in `Composed/Coined.lean`. At `J` tables the soundness half is
`Composed/SoundJ.lean`, its pieces in `Composed/DefsJ.lean`. `Composed/Custody.lean` and `Composed/CustodyJ.lean` restate
them under `live-verifier`, and `Composed/CustodyJJson.lean` the latter with the draw file `flock-verify draw` printed;
`Composed/PublicJ.lean` restates `CustodyJ`'s with "wrong" measured against the public file, and `Composed/DrawnJ.lean`
bounds the drawn units' correctness at flock-verify's acceptance.
`Composed/NonVacuousJ.lean` proves the `J`-table statements' reference setup and draw hypotheses satisfiable together.
`Composed/DefsJA.lean` states `DefsJ`'s session and runs pieces at any acceptance and setup.
-/
