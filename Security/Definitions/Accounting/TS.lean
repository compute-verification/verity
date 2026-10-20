import Definitions.Accounting.TS.Machine
import Definitions.Accounting.TS.Routes
import Definitions.Accounting.TS.Play
import Definitions.Accounting.TS.Receipt
import Definitions.Accounting.TS.Slots
import Definitions.Accounting.TS.ProofLayer
import Definitions.Accounting.TS.Drawn
import Definitions.Accounting.TS.FanIn
import Definitions.Accounting.TS.Stages
import Definitions.Accounting.TS.Steer
import Definitions.Accounting.TS.Sheet
import Definitions.Accounting.TS.Fold
import Definitions.Accounting.TS.ReceiptHMu
import Definitions.Accounting.TS.Unfolded

/-! The tensor–scalar model (EPS1, note:20261017T0324Z-draft-eps1-spec): the machine, its facets and bound, routes and
labelled programs with the terms of the decomposition, the full-checking and sampled plays, the receipt rounds that
split the luck premise into its operand and receipt halves, a round's unfolded share and the whole-unit luck, the
counting objects of the tensor slot count, the output count, the fan-in count and LIN-STEER, and a fold of words by
binding steps. -/
