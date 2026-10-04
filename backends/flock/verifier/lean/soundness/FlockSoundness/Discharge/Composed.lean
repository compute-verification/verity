import FlockSoundness.Discharge.Composed.Sound
import FlockSoundness.Discharge.Composed.Base
import FlockSoundness.Discharge.Composed.Coined
import FlockSoundness.Discharge.Composed.Headline
import FlockSoundness.Discharge.Composed.Coins
import FlockSoundness.Discharge.Composed.Geometry
import FlockSoundness.Discharge.Composed.Pads
import FlockSoundness.Discharge.Composed.Inner
import FlockSoundness.Discharge.Composed.Inner.Holds
import FlockSoundness.Discharge.Composed.Inner.Zerocheck
import FlockSoundness.Discharge.Composed.Inner.ZerocheckMasked
import FlockSoundness.Discharge.Composed.Inner.ZerocheckReplay
import FlockSoundness.Discharge.Composed.Inner.Lincheck
import FlockSoundness.Discharge.Composed.Inner.RingSwitch
import FlockSoundness.Discharge.Composed.Inner.ZerocheckGlue
import FlockSoundness.Discharge.Composed.Inner.LincheckReplay
import FlockSoundness.Discharge.Composed.Setup
import FlockSoundness.Discharge.Composed.Refine
import FlockSoundness.Discharge.Composed.SoundJ
import FlockSoundness.Discharge.Composed.Custody
import FlockSoundness.Discharge.Composed.CustodyJ

/-!
# One `--zk` sampled session: soundness and zero knowledge composed

The soundness half is in `Composed/Sound.lean`, its pieces in `Composed/Defs.lean`; the composed headline in
`Composed/Headline.lean`, the draw's `Coined` tables in `Composed/Coined.lean`. At `J` tables the soundness half is `Composed/SoundJ.lean`, its pieces in `Composed/DefsJ.lean`;
`Composed/Custody.lean` and `Composed/CustodyJ.lean` restate them under `live-verifier`.
-/
