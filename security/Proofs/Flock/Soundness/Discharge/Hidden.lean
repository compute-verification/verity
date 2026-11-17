module

public import Proofs.Flock.Soundness.Discharge.Hidden.Defs
public import Proofs.Flock.Soundness.Discharge.Hidden.Spec
public import Proofs.Flock.Soundness.Discharge.Hidden.SpecP
public import Proofs.Flock.Soundness.Discharge.Hidden.PortsP
public import Proofs.Flock.Soundness.Discharge.Hidden.Facts
public import Proofs.Flock.Soundness.Discharge.Hidden.Built
public import Proofs.Flock.Soundness.Discharge.Hidden.ParseCommit
public import Proofs.Flock.Soundness.Discharge.Hidden.Binding
public import Proofs.Flock.Soundness.Discharge.Hidden.Public
public import Proofs.Flock.Soundness.Discharge.Hidden.Delta
public import Proofs.Flock.Soundness.Discharge.Hidden.Rows
public import Proofs.Flock.Soundness.Discharge.Hidden.Layout
public import Proofs.Flock.Soundness.Discharge.Hidden.LayoutP
public import Proofs.Flock.Soundness.Discharge.Hidden.Flat
public import Proofs.Flock.Soundness.Discharge.Hidden.Unpin
public import Proofs.Flock.Soundness.Discharge.Hidden.SpecOf
public import Proofs.Flock.Soundness.Discharge.Hidden.Commit
public import Proofs.Flock.Soundness.Discharge.Hidden.Committed
public import Proofs.Flock.Soundness.Discharge.Hidden.Theorems
public import Proofs.Flock.Soundness.Discharge.Hidden.Wide.Defs
public import Proofs.Flock.Soundness.Discharge.Hidden.Wide.V2
public import Proofs.Flock.Soundness.Discharge.Hidden.Wide.Shared
public import Proofs.Flock.Soundness.Discharge.Hidden.Wide.Checked
public import Proofs.Flock.Soundness.Discharge.Hidden.Wide.Upv
public import Proofs.Flock.Soundness.Discharge.Hidden.Wide.Block
public import Proofs.Flock.Soundness.Discharge.Hidden.Wide.Fit
public import Proofs.Flock.Soundness.Discharge.Hidden.Wide.SegChain
public import Proofs.Flock.Soundness.Discharge.Hidden.Wide.SegShape
public import Proofs.Flock.Soundness.Discharge.Hidden.Wide.SegCommit
public import Proofs.Flock.Soundness.Discharge.Hidden.Wide.SegDelta
public import Proofs.Flock.Soundness.Discharge.Hidden.Wide.SegCopies
public import Proofs.Flock.Soundness.Discharge.Hidden.Wide.SegRowBc
public import Proofs.Flock.Soundness.Discharge.Hidden.Wide.SegV2
public import Proofs.Flock.Soundness.Discharge.Hidden.Wide.SegPort
public import Proofs.Flock.Soundness.Discharge.Hidden.Wide.SegTgt
public import Proofs.Flock.Soundness.Discharge.Hidden.Wide.SegFit
public import Proofs.Flock.Soundness.Discharge.Hidden.Wide.SegFitP
public import Proofs.Flock.Soundness.Discharge.Hidden.Wide.SharedP
public import Proofs.Flock.Soundness.Discharge.Hidden.Wide.FitS

/-! Hidden outputs: `Stmt.setupHidden`'s soundness at the protocol as Lean defines it. No theorem proves the statements
widened to a tail that returns the outputs (`Wide.Defs`' `…Tail`) or the six widened statements without
`0 < st.c.upv` (`Wide.Checked`). -/

@[expose] public section
