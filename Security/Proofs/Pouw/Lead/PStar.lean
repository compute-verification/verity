import Proofs.Pouw.Lead.PStar.AlphabetProofs
import Proofs.Pouw.Lead.PStar.Call
import Proofs.Pouw.Lead.PStar.ExpansionProofs
import Proofs.Pouw.Lead.PStar.Lanes
import Proofs.Pouw.Lead.PStar.Marginal
import Proofs.Pouw.Lead.PStar.MassFloorProofs
import Proofs.Pouw.Lead.PStar.Stream
import Proofs.Pouw.Lead.PStar.Table
import Proofs.Pouw.Lead.PStar.Tile
import Proofs.Pouw.Lead.PStar.Walks

/-!
# The lead's p*: how often a coin lane leaves an entry unchanged

Its proofs are in `Lead/PStar/`, its new declarations in `Pouw.SecurityProofs.Lead.PStar`.

* `AlphabetProofs`: the per-entry and matrix hit counts for digits drawn after the data, for `uw6`, the coin lane
  `1 + b + s` and hw's row lane, alone or in a sum of lanes.
* `Call`: `Tile`'s argument over a call's coins (`Pouw.Lead.Coins`): at any step an MMA tile's committed words
  (`Pouw.Lead.word`) are a value fixed before the coins on at most a `2^−168` share of them, whatever the data, by the
  tile's coin lanes alone (a column's even and odd rows fix its three lanes).
* `ExpansionProofs`: the affine expansion `u + t·v` is pairwise uniform on a prime modulus, with the two moments of
  the hit count over steps.
* `Lanes`: `blc4a-t64-e7`'s three coin lanes sharing one draw a tile under `-rot`'s weights, `(127, 1, 1)` on even rows
  and `(1, 127, 1)` on odd: `p* = 65/2,097,152` exactly on either class, and the coin sum's range `±16,383`.
* `Marginal`: a part of the draw caps the whole: a value that moves with one part of the draw only through a term
  hits a fixed value on at most the share that term does.
* `MassFloorProofs`: point masses at most `p` force a variance of at least `(1/p² − 1)/12`, with its mean over steps.
* `Stream`: an MMA tile against the certifier's coin stream: at any step the tile's words are a streamed tile on at
  most a `2^−105` share of the call's coins, whatever the data and the stream's arrangement. The stream holds each of
  the tile's 24 lanes once at most, so at least 5 of its columns compare an even and an odd row with words that don't
  move with the lanes.
* `Table`: the statement's pieces. `pStar_eq`: a value of a step's lanes and tag bit, `127 s₀ + s₁ + s₂ + 1 + b`, is
  taken by at most 130 of the `2 · 128³` draws and one by exactly 130, `Lanes`' bound on an even row. `table_le`: a
  memory of `mem` pre-coin tiles and `mem` streamed ones holds a committed tile at any step on at most a
  `(2 · mem + 1) · 2^−100` share of the coins (`Call` against the tables' values and zero, `Stream` against the
  streamed tiles), within `tableGap`.
* `Tile`: at tile grain, the even and odd rows together fix each column's triple, so a 16 × 8 MMA tile's step is
  injective in its `2^184` draws: `p* = 2^−184` exactly.
* `Walks`: the same lanes summed over steps: an entry's own walk keeps `p*` over any nonempty set of steps; an even
  and an odd row of one tile, which read the same coins, differ by a value fixed before the draw on at most `1/256` of
  the draws, and `k` of their words coincide together on at most `(1/2)·(1/128)^k`, the bits being shared by the
  row's columns; and over the tile, `p*` of the entries on average.
-/
