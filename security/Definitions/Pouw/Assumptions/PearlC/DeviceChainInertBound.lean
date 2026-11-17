module

public import Definitions.Pouw.Assumptions.PearlC.TTOut
public import Definitions.Pouw.Assumptions.PearlC.TTOutW1

/-! The chain-share cap on Pearl-C at a device record: a unit is admitted only if its debit is at most `ρ` times its chain's credited work. -/

@[expose] public section

namespace Pouw.PearlC

open Finset Pouw.Fp8Atom

def chainCreditDev (d : PearlCDevice) (s : Shape) : ℚ :=
  let mkn : ℚ := (s.m : ℚ) * s.k * s.n
  mkn + (if d.G = 0 then 0 else mkn * d.prices.add / (32 * d.G))

variable {Q R S : Type}

noncomputable def pearlCProtocolDevChainInertBound (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ : ℚ) : Protocol Q R S :=
  { pearlCProtocolDev d sem ρ with
    inertBoundOK := fun H s U u act => inertWorkDev d (sem.unitInertWorkDev d H s U u act) ≤ ρ * chainCreditDev d (U.layout.shape u) }

/-- A good tile's credit stays `pearlCTilesDev`'s, `(1 − ρ)·creditDev·share`. -/
noncomputable def pearlCTilesDevChainInertBound (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ : ℚ) : TileRules Q R S :=
  { pearlCTilesDev d sem ρ with
    inertBoundOK := fun H s U g act =>
      let tl := auditTiling 64 64 U.layout
      let sh := U.layout.shape (tl.unit g)
      sem.tileInertWorkDev d H s U (tl.unit g) (sem.passRows sh act (tl.rows g)) (tl.cols g) act ≤
        ρ * chainCreditDev d sh * tileShare sh (sem.passRows sh act (tl.rows g)) (tl.cols g) }

end Pouw.PearlC

namespace Pouw.PearlC.Assumptions

variable {Q R S : Type}

/-- `TTOutPearlCDev` with fewer units credited, so it asserts less at the same `ρ`
(`ttOutPearlCDevChainInertBound_of_inert_bound`). `CM` charges a tensor-core instruction its whole fragment shape, never
per output word. -/
def TTOutPearlCDevChainInertBound [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ : ℚ) : Prop :=
  TTOut CM (pearlCProtocolDevChainInertBound d sem ρ) (pearlCDomainDev d) (1 / 400) δPearlC

/-- It asserts less than `TTOutTilePearlCDev` at the same `ρ` (`ttOutTilePearlCDevChainInertBound_of_inert_bound`): the
same fixed credit on fewer tiles. -/
def TTOutTilePearlCDevChainInertBound [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ : ℚ) : Prop :=
  TTOutTile CM (pearlCProtocolDevChainInertBound d sem ρ) (pearlCTilesDevChainInertBound d sem ρ) (pearlCDomainDev d) (1 / 400)
    δPearlC

def TTOutPearlCDevChainInertBoundW1 [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (dv : W1.W1Device)
    (salt : S → List ℕ) (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ : ℚ) : Prop :=
  TTOutW1 dv salt (pearlCProtocolDevChainInertBound d sem ρ) (pearlCDomainDev d) (1 / 400) δPearlC

def TTOutTilePearlCDevChainInertBoundW1 [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (dv : W1.W1Device)
    (salt : S → List ℕ) (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ : ℚ) : Prop :=
  TTOutTileW1 dv salt (pearlCProtocolDevChainInertBound d sem ρ) (pearlCTilesDevChainInertBound d sem ρ) (pearlCDomainDev d)
    (1 / 400) δPearlC

end Pouw.PearlC.Assumptions
