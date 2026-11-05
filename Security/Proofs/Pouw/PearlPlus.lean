import Proofs.Pouw.PearlPlus.Atom
import Proofs.Pouw.PearlPlus.Noise
import Proofs.Pouw.PearlPlus.Words
import Proofs.Pouw.PearlPlus.Rules
import Proofs.Pouw.PearlPlus.Stats
import Proofs.Pouw.PearlPlus.Keys
import Proofs.Pouw.PearlPlus.Rho
import Proofs.Pouw.PearlPlus.Peel
import Proofs.Pouw.PearlPlus.Factors
import Proofs.Pouw.PearlPlus.Semantics
import Proofs.Pouw.PearlPlus.Assembly
import Proofs.Pouw.PearlPlus.Draw
import Proofs.Pouw.PearlPlus.Debit
import Proofs.Pouw.PearlPlus.Cap

/-!
# Pearl+'s bridge: the Lean check's verdict on a tile, read in the Models game

From `PouwCheck.PearlPlus.checkTile`'s code to `TileWordsGood` and `TileGood` of `pearlCProtocolDevRev1 devH100` at the
deployed semantics (`semH3`: `pc8`'s row arithmetic, the H100 atom and -h3's noise from SHA-512), the tile the named
premise `TTOutTilePearlCH100Rev1` reads. Four parts:

* (a) `Atom`: the evaluator's FP8 atom is the Definitions' step.
* (b) `Noise` and `Words`: -h3's noise is `NoiseSha512`'s, and a tile's checked words are the reference values.
* (c) `Rules`, `Keys`, `Debit` and `Cap`: the per-row rules are `RowOK`, a word's replayed inert work is the game's
  (`Keys`: the replay's keys, bisection and carried add; `Debit`: the replay itself, and that it reads the semantics'
  codes), and a passed tile is within the game's inert-work bound, its chain and its forming term (`Cap`).
* (d) `Assembly`: a passed tile whose openings are its unit's commitment has good words, or SHA-512 collides.

`Draw` reads the run's draw and acceptance as `TileDraw`'s game draws and audits: a passed tile's declaration is
consistent (`declOK`), the code's tickets are `drawTickets`, and an accepted run's samples are accepted by the tolerant
audit over `goodDeclTiles ∪ blankTiles`, or SHA-512 collides.
-/
