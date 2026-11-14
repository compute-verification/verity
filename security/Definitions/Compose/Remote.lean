module

public import Definitions.Compose.Remote.Node
public import Definitions.Compose.Remote.Held
public import Definitions.Compose.Remote.Outside
public import Definitions.Compose.Remote.Run
public import Definitions.Compose.Remote.Draw
public import Definitions.Compose.Remote.Fork
public import Definitions.Compose.Remote.Sketch
public import Definitions.Compose.Remote.SketchLine
public import Definitions.Compose.Remote.SketchToy

/-! PoUB's vocabulary: the node, the audit's parameters and the γ terms, a run and its routes,
the node's memory as a trace (`Remote.Held`) and what devices outside it compute on (`Remote.Outside`), the verifier's
opening sample with its pooled bound (`Remote.Draw`), the receipts' fork game (`Remote.Fork`), the receipts' sketch over
a session's seeds and devices (`Remote.Sketch`), the sample's coins read as a sketch outcome (`Remote.SketchLine`), and
the sketch's toy instances, at which the deployed sketch bound's witnesses are stated (`Remote.SketchToy`). -/

@[expose] public section
