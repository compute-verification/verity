module

public import Definitions.Poub.Remote.Node
public import Definitions.Poub.Remote.Held
public import Definitions.Poub.Remote.Outside
public import Definitions.Poub.Remote.Run
public import Definitions.Poub.Remote.Sample
public import Definitions.Poub.Remote.Fork
public import Definitions.Poub.Remote.Sketch
public import Definitions.Poub.Remote.SketchLine
public import Definitions.Poub.Remote.SketchToy

/-! PoUB's vocabulary: the node, the audit's parameters and the γ terms, a run and its routes,
the node's memory as a trace (`Remote.Held`) and what devices outside it compute on (`Remote.Outside`), the verifier's
opening sample with its pooled bound (`Remote.Sample`), the receipts' fork game (`Remote.Fork`), the receipts' sketch over
a session's seeds and devices (`Remote.Sketch`), the sample's coins read as a sketch outcome (`Remote.SketchLine`), and
the sketch's toy instances, at which the deployed sketch bound's witnesses are stated (`Remote.SketchToy`). -/

@[expose] public section
