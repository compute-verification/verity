module

public import Proofs.Flock.RecGeneric.AtFlockRecovers
public import Proofs.Flock.RecGeneric.AtFlockTerms

/-!
# L1 at Flock, with no hypothesis left

`RecoversLinkOn` at every compiled record, bound record, outer deployment, caps and set of statements
(`recoversLinkOn`): F9a (`recoversLinkOn_of`) at F6's three term comparisons (`forkPlansLeBinnerV`, `capPlanIsFCI`,
`conflictPlanIsF0I`), the last two at F1 (`ivStarOfBound`). With L1 proved, `ksLinkOnFlock` is `KSLinkOn` from
`RecoversLinkOn`'s other premises.
-/

@[expose] public section

namespace Claims.RecGeneric

open Claims Claims.Recursion

section Instance

variable {Coin Col T Col' T' : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
  [DecidableEq Col'] [Zero Col] [Zero Col']

/-- **L1 at Flock** (`RecoversLinkOn`), with no hypothesis: F9a at F6's three term comparisons and F1. -/
theorem recoversLinkOn (H : Msg → Msg) (D : RecAt Coin) (Bd : Bound Coin Col T)
    (DO : Claims.Flock.Deploy Coin Col' T') (Cp : Caps Coin) (G : Msg → Prop) : RecoversLinkOn H D Bd DO Cp G :=
  recoversLinkOn_of H D Bd DO Cp G (forkPlansLeBinnerV H D Bd DO) (capPlanIsFCI D Bd DO Cp (ivStarOfBound Bd))
    (conflictPlanIsF0I D Bd DO (ivStarOfBound Bd))

/-- **`KSLinkOn` from L1, proved** (`Flock.SecurityProofs.GenericRecursiveKS`), and `RecoversLinkOn`'s other
premises: `recoversLinkOn` applied to them. -/
theorem ksLinkOnFlock (H : Msg → Msg) (D : RecAt Coin) (Bd : Bound Coin Col T)
    (DO : Claims.Flock.Deploy Coin Col' T') (Cp : Caps Coin) (G : Msg → Prop) (hKS : KSFlockLin H Bd.DE Cp)
    (hO : OuterKS H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy Bd.deploy.decode) DO)
    (hT : TableErrNonneg (D.record Bd.IVStar)) (hxI : ∀ x, D.xI x = Claims.Recursion.xI (D.record Bd.IVStar) x)
    (hdec : ∃ dec : Msg → Coin, ∀ c, dec (Bd.ans c) = c) (hε : ∀ x, 0 ≤ Bd.DE.εtable x)
    (hLC : (Proto.ofFlock Bd Bd.deploy.decode Cp).LastCoinFree H) (hB : Bd.Binds)
    (hdI : decOK Bd.DE Bd.deploy.decode) (hdO : decOK DO DO.decode) (hC : ContractOn H D Bd DO Cp G) :
    KSLinkOn H D Bd DO Cp G :=
  recoversLinkOn H D Bd DO Cp G _root_.Flock.SecurityProofs.GenericRecursiveKS hKS hO hT hxI hdec hε hLC hB hdI hdO
    hC

end Instance

end Claims.RecGeneric
