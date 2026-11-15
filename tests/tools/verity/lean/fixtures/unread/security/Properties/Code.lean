import Code
import Definitions.Core

namespace Properties.Code

def CheckOk : Prop := ∀ n : Nat, _root_.Code.check n = true

def Unwraps : Prop := ∀ w : _root_.Core.Wrap, (w : Nat) = w.n

end Properties.Code
