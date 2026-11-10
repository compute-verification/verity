module

public import Definitions.Accounting.TS.Machine
public import Definitions.Accounting.TS.Routes
public import Definitions.Accounting.TS.Play
public import Definitions.Accounting.TS.Receipt
public import Definitions.Accounting.TS.Slots
public import Definitions.Accounting.TS.ProofLayer
public import Definitions.Accounting.TS.Drawn
public import Definitions.Accounting.TS.Hashed
public import Definitions.Accounting.TS.FanIn
public import Definitions.Accounting.TS.Stages
public import Definitions.Accounting.TS.Steer
public import Definitions.Accounting.TS.Sheet
public import Definitions.Accounting.TS.Fold
public import Definitions.Accounting.TS.ReceiptHMu
public import Definitions.Accounting.TS.Unfolded
public import Definitions.Accounting.TS.SharedNoise
public import Definitions.Accounting.TS.J10R

/-! The tensor–scalar model (EPS1): the machine, its facets and bound, routes and
labelled programs with the terms of the decomposition, the full-checking and sampled plays, the sampled plays' openings
as frame-v3-sha512 paths with their fork finder, sampled rounds under
per-stage coins, the receipt rounds that split the luck premise into its operand and receipt halves, a round's unfolded
share and the whole-unit luck, the counting objects of the tensor slot count, the output count, the fan-in count and
LIN-STEER, a fold of words by binding steps, and J10-R's noise, one seed a call at call scale. -/

@[expose] public section
