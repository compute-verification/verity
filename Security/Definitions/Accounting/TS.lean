import Definitions.Accounting.TS.Run
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
import Definitions.Accounting.TS.SharedNoise
import Definitions.Accounting.TS.J10R

/-! The tensor–scalar model (EPS1): a program's run, the heads of committed words, labelled programs and a cheater's
labelled run, the full-checking and sampled plays, sampled rounds under per-stage coins, the receipt rounds that split
the luck premise into its operand and receipt halves, a round's unfolded share and the whole-unit luck, the counting
objects of the tensor slot count, the output count, the fan-in count and LIN-STEER, a device's data sheet, a fold of
words by binding steps, and J10-R's noise, one seed a call at call scale. -/
