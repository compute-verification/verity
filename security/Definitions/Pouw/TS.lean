module

public import Definitions.Pouw.TS.Machine
public import Definitions.Pouw.TS.Routes
public import Definitions.Pouw.TS.Play
public import Definitions.Pouw.TS.Receipt
public import Definitions.Pouw.TS.Slots
public import Definitions.Pouw.TS.ProofLayer
public import Definitions.Pouw.TS.Sampled
public import Definitions.Pouw.TS.Hashed
public import Definitions.Pouw.TS.FanIn
public import Definitions.Pouw.TS.Stages
public import Definitions.Pouw.TS.Steer
public import Definitions.Pouw.TS.Sheet
public import Definitions.Pouw.TS.Fold
public import Definitions.Pouw.TS.ReceiptHMu
public import Definitions.Pouw.TS.Unfolded
public import Definitions.Pouw.TS.SharedNoise
public import Definitions.Pouw.TS.J10R

/-! The tensor–scalar model (EPS1): the machine, its facets and bound, routes and
labelled programs with the terms of the decomposition, the full-checking and sampled plays, the sampled plays' openings
as frame-v3-sha512 paths with their fork finder, sampled rounds under
per-stage coins, the receipt rounds that split the luck premise into its operand and receipt halves, a round's unfolded
share and the whole-unit luck, the counting objects of the tensor slot count, the output count, the fan-in count and
LIN-STEER, a fold of words by binding steps, and J10-R's noise, one seed a call at call scale. -/

@[expose] public section
