module

public import PouwPearlPlus.Records
public import PouwPearlPlus.Run
public import PouwPearlPlus.Coins
public import PouwPearlPlus.Load
public import PouwPearlPlus.Words
public import PouwPearlPlus.Rule
public import PouwPearlPlus.Tile
public import PouwPearlPlus.Tie
public import PouwPearlPlus.Credit
public import PouwPearlPlus.Check
public import PouwPearlPlus.Open
public import PouwPearlPlus.Protocol

/-!
# Pearl+'s run verdict, as executable Lean

The library of the package `pouw_pearlplus` and the root of its Lean check (`roots` in `lean-audit.json`):

* `Records`, a pass's records and give-up; `Run`, the run's commitment, the verifier's coin record and the tiles it
  opens; `Coins`, the epoch's salt coin and the verifier's sample;
* `Load`, a retained pass's manifest held to what it published; `Words`, a tile's words under Pearl-C v1 -h3; `Rule`,
  the tile rule on an opened tile; `Tile`, the verdict on one drawn tile; `Tie`, the passes held to the window record;
* `Credit`, the run's credit at Pearl's assumption; `Check`, the run verdict (`verify_passes`); `Open`, the prover's
  openings from a pass's files and the verdict as JSON; `Protocol`, the verdict as a core protocol,
  `verity/pouw/pearl-plus/v1`, which remote's composite of record reads as its work component.

`PouwPearlPlus.Main`, the `pouw-pearlplus` executable's entry point, is not imported here.
-/

@[expose] public section
