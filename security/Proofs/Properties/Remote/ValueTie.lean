module

public import Properties.Remote.ValueTie
public import Proofs.Properties.Pouw.PearlPlus.Held
public import Proofs.Properties.Pouw.PearlPlus.Open

/-!
# The value tie's Pearl+ premise (`Verity.ValueTie.Shared`)

Both of its statements are proved: `AcceptedIsHeldToW` in `Proofs.Properties.Pouw.PearlPlus.Held` and `OpeningBinds`
in `Proofs.Properties.Pouw.PearlPlus.Open`.
-/

@[expose] public section

namespace Verity.ValueTie

theorem shared : Shared := ⟨Properties.Pouw.PearlPlus.acceptedIsHeldToW, Properties.Pouw.PearlPlus.openingBinds⟩

end Verity.ValueTie
