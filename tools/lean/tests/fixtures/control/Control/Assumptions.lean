/-! The controls' named assumption: a closed `Prop`, taken as a hypothesis. -/

namespace Control

def Hyp : Prop := ∀ n : Nat, n + 0 = n

end Control
