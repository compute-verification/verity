import Proofs.Pouw.PearlPlus.Atom
import Proofs.Pouw.PearlPlus.Noise
import Proofs.Pouw.PearlPlus.Words
import Proofs.Pouw.PearlPlus.Rules
import Proofs.Pouw.PearlPlus.Keys
import Proofs.Pouw.PearlPlus.Assembly
import Proofs.Pouw.PearlPlus.Cap

/-!
# Pearl+'s bridge: the Lean check's verdict on a tile, read in the Models game

From `PouwCheck.PearlPlus.checkTile`'s code to `TileWordsGood` and `TileGood` of `pearlCProtocolDevRev1 devH100` at the
deployed semantics (`semH3`: `pc8`'s row arithmetic, the H100 atom and -h3's noise from SHA-512), the tile the named
premise `TTOutTilePearlCH100Rev1` reads. Four parts:

* (a) `Atom`: the evaluator's FP8 atom is the Definitions' step.
* (b) `Noise` and `Words`: -h3's noise is `NoiseSha512`'s, and a tile's checked words are the reference values.
* (c) `Rules`, `Keys` and `Cap`: the per-row rules are `RowOK`, a word's debit replay is the game's (`Keys`: the
  replay's keys, bisection and carried add), and a passed tile keeps the game's cap (`Cap`, stated).
* (d) `Assembly`: a passed tile whose openings are its unit's commitment has good words, or SHA-512 collides.
-/
