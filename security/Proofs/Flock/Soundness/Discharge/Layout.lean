module

public import Proofs.Flock.Soundness.Discharge.Layout.Readers
public import Proofs.Flock.Soundness.Discharge.Layout.Bytes
public import Proofs.Flock.Soundness.Discharge.Layout.Region
public import Proofs.Flock.Soundness.Discharge.Layout.HmRegion
public import Proofs.Flock.Soundness.Discharge.Layout.OutRegion
public import Proofs.Flock.Soundness.Discharge.Layout.InRegion
public import Proofs.Flock.Soundness.Discharge.Layout.InCover
public import Proofs.Flock.Soundness.Discharge.Layout.Pins
public import Proofs.Flock.Soundness.Discharge.Layout.Drawn
public import Proofs.Flock.Soundness.Discharge.Layout.SharedRows
public import Proofs.Flock.Soundness.Discharge.Layout.QSort
public import Proofs.Flock.Soundness.Discharge.Layout.Site
public import Proofs.Flock.Soundness.Discharge.Layout.OutSite
public import Proofs.Flock.Soundness.Discharge.Layout.OfRows
public import Proofs.Flock.Soundness.Discharge.Layout.KeyProg
public import Proofs.Flock.Soundness.Discharge.Layout.KeyProgP
public import Proofs.Flock.Soundness.Discharge.Layout.OutShape
public import Proofs.Flock.Soundness.Discharge.Layout.OutPin
public import Proofs.Flock.Soundness.Discharge.Layout.RowEnc
public import Proofs.Flock.Soundness.Discharge.Layout.KeysRow
public import Proofs.Flock.Soundness.Discharge.Layout.Outs
public import Proofs.Flock.Soundness.Discharge.Layout.OutsP
public import Proofs.Flock.Soundness.Discharge.Layout.Keyed
public import Proofs.Flock.Soundness.Discharge.Layout.Tables
public import Proofs.Flock.Soundness.Discharge.Layout.Scope
import Mathlib.Combinatorics.Matroid.Init

/-! `lay` and `rs` of `Prog.flock_headline`, discharged from the decoder's places and the executable's regions. -/

@[expose] public section
