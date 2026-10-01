/-! The controls' named assumptions: a closed `Prop`, taken as a hypothesis, and one of a number (`Fits`), which a
statement takes through a definition it unfolds. -/

namespace Control

def Hyp : Prop := ∀ n : Nat, n + 0 = n

def Fits (n : Nat) : Prop := n < 2 ^ 32

end Control
