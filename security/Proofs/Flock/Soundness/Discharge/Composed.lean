module

public import Proofs.Flock.Soundness.Discharge.Composed.Sound
public import Proofs.Flock.Soundness.Discharge.Composed.Base
public import Proofs.Flock.Soundness.Discharge.Composed.Coined
public import Proofs.Flock.Soundness.Discharge.Composed.Headline
public import Proofs.Flock.Soundness.Discharge.Composed.Coins
public import Proofs.Flock.Soundness.Discharge.Composed.Pads
public import Proofs.Flock.Soundness.Discharge.Composed.Inner
public import Proofs.Flock.Soundness.Discharge.Composed.Inner.Holds
public import Proofs.Flock.Soundness.Discharge.Composed.Inner.Zerocheck
public import Proofs.Flock.Soundness.Discharge.Composed.Inner.ZerocheckMasked
public import Proofs.Flock.Soundness.Discharge.Composed.Inner.ZerocheckReplay
public import Proofs.Flock.Soundness.Discharge.Composed.Inner.Lincheck
public import Proofs.Flock.Soundness.Discharge.Composed.Inner.RingSwitch
public import Proofs.Flock.Soundness.Discharge.Composed.Inner.ZerocheckGlue
public import Proofs.Flock.Soundness.Discharge.Composed.Inner.LincheckReplay
public import Proofs.Flock.Soundness.Discharge.Composed.Setup
public import Proofs.Flock.Soundness.Discharge.Composed.RegWords
public import Proofs.Flock.Soundness.Discharge.Composed.SoundJ
public import Proofs.Flock.Soundness.Discharge.Composed.Custody
public import Proofs.Flock.Soundness.Discharge.Composed.CustodyJ
public import Proofs.Flock.Soundness.Discharge.Composed.CustodyJJson
public import Proofs.Flock.Soundness.Discharge.Composed.PublicJ
public import Proofs.Flock.Soundness.Discharge.Composed.DrawnJ
public import Proofs.Flock.Soundness.Discharge.Composed.NonVacuousJ
public import Proofs.Flock.Soundness.Discharge.Composed.DefsJA

/-!
# One `--zk` sampled session: soundness and zero knowledge composed

The soundness half is in `Composed/Sound.lean`, its pieces in `Composed/Defs.lean`; the composed headline in
`Composed/Headline.lean`, the sample's `Coined` tables in `Composed/Coined.lean`. At `J` tables the soundness half is
`Composed/SoundJ.lean`, its pieces in `Composed/DefsJ.lean`. `Composed/Custody.lean` and `Composed/CustodyJ.lean` restate
them under `live-verifier`, and `Composed/CustodyJJson.lean` the latter with the sample file `flock-verify draw` printed;
`Composed/PublicJ.lean` restates `CustodyJ`'s with "wrong" measured against the public file, and `Composed/DrawnJ.lean`
bounds the sampled proof blocks' correctness at flock-verify's acceptance.
`Composed/NonVacuousJ.lean` proves the `J`-table statements' reference setup and sample hypotheses satisfiable together.
`Composed/DefsJA.lean` states `DefsJ`'s session and runs pieces at any acceptance and setup.
-/

@[expose] public section
