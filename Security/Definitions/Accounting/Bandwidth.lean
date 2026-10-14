import Definitions.Accounting.Bandwidth.Game
import Definitions.Accounting.Bandwidth.Relay
import Definitions.Accounting.Bandwidth.Interactive
import Definitions.Accounting.Bandwidth.Joint
import Definitions.Accounting.Bandwidth.Causal

/-! The bandwidth audit v0: its game in the random-oracle model (`Bandwidth/Game.lean`), the same game with side
sources that may relay tags (`Bandwidth/Relay.lean`), with side sources that answer what the chip sent
(`Bandwidth/Interactive.lean`), for a device that also holds a secret its chip's view determines
(`Bandwidth/Joint.lean`), and the chip's ports on a strategy's runs (`Bandwidth/Causal.lean`). -/
