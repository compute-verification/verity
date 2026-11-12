module

public import Definitions.Accounting.Bandwidth.Game
public import Definitions.Accounting.Bandwidth.Relay
public import Definitions.Accounting.Bandwidth.Interactive
public import Definitions.Accounting.Bandwidth.Joint
public import Definitions.Accounting.Bandwidth.Causal
import Mathlib.Basic.ENNReal.Real

/-! The bandwidth audit v0: its game in the random-oracle model (`Bandwidth/Game.lean`), the same game with side
sources that may relay tags (`Bandwidth/Relay.lean`), with side sources that answer what the chip sent
(`Bandwidth/Interactive.lean`), for a device that also holds a secret its chip's view determines
(`Bandwidth/Joint.lean`), and the chip's ports on a strategy's runs (`Bandwidth/Causal.lean`). -/

@[expose] public section
