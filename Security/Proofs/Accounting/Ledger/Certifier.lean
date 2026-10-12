import Models.NetworkCertifier.Guarantees
import Definitions.Accounting.Ledger.Pins

/-!
# The network certifier's guarantee is an upper pin on content

`NetTiming.Guarantees.EncardDecodableLeConstantRatePerWindow`, at one interaction, schedule and horizon of `m` whole
windows, is `Accounting.Ledger.CertifierContent`: the secrets are the covert sender's, an accepted run is one whose
records the constant-rate schedule accepts up to `m·T` buckets, and the wire is what crossed the unit's boundary by then.
-/

namespace Accounting.Ledger

open NetTiming NetTiming.Guarantees

theorem certifierContent_of_guarantee (hG : EncardDecodableLeConstantRatePerWindow) {L K S : ℕ}
    {Z In Act Wire Ac : Type} [Fintype Ac] (I : Interaction L K S Z In Act Wire) (T W : ℕ) (ss : SyncSet)
    (frames : List In → Ac → Fin L → List LogicalFrame) (m D : ℕ) (hn : m * T ≤ W * T)
    (view : ℕ → Obs L K S → Wire) (hW2 : Regenerates I.certifier view) (hJ : JitterBounded I.certifier D) :
    CertifierContent
      (fun z => AcceptedUpTo (constantRate (S := S) T W ss frames) I.op (m * T) (I.obsRun z (m * T)))
      (fun z => I.wireRun z (m * T)) (Fintype.card Ac) ss.size D T m :=
  fun Zs hacc dec hdec => hG I T W ss frames m D hn view hW2 hJ Zs hacc dec hdec

end Accounting.Ledger
