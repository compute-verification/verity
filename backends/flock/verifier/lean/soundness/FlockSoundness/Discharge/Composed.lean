import FlockSoundness.Discharge.Composed.Sound
import FlockSoundness.Discharge.Composed.SoundJ
import FlockSoundness.Discharge.Composed.Custody
import FlockSoundness.Discharge.Composed.CustodyJ

/-!
# One `--zk` sampled session: soundness

The soundness half is in `Composed/Sound.lean`; the pieces it reads in `Composed/Defs.lean`. At `J` tables it is
`Composed/SoundJ.lean`, its pieces in `Composed/DefsJ.lean`. `Composed/Custody.lean` and `Composed/CustodyJ.lean`
restate them under `live-verifier`.
-/
