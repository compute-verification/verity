module

public import Properties.Pouw.PearlPlus
public import Properties.Pouw.PearlPlusDraw
public import Properties.Challenger
public import Properties.Verity.Game

/-!
# Pearl+'s served y at the registration record's entries (`PouwCheck.PearlPlus.tieOf`, `checkTile`, `todo`)

What an accepted run establishes about the values its registration record W registers, stated against the verdict's own
functions. These are statements only: none is proved here, and none is a guarantee yet. W is the
`verity/registration/v2` record `--tie W` names, the one `served_passes.py retain` writes for an execution phase.

A run tied to W registers, for pass `s`'s call `c` and each weight `b`, three trees in W: the call's A rows
(`pearl-a/{s}/{c}`), its served y (`pearl-y/{s}/{c}`) and the weight's B rows (`pearl-b/{b}`). A cell is a position
(round, call, row, column). The y word of a cell depends only on A's row, the weight's row, the pass's salt, the weight's
root and the call's index: `yCell`, the unscale (`Rule.yOf`) of the tile's U at that row and column, formed as
`checkTile` forms it (`Words.encode`, `Words.checked`), one row and one column at a time.

* `TieMeans`: what `tieOf` establishes. W is a `verity/registration/v2` record, the answer names its digest, and for
  every call of every pass W registers once the A rows and the y under the roots the pass published (the ones
  `checkTile` opens against), at the call's shape and under the call's domains; at W's version 2, the same for each
  weight's B rows under the pass's weight root. W registers no other value under the tie's prefixes but a dropped
  round's calls.
* `AcceptedIsHeldToW`: an accepted verdict on an input that names W is a run held to W (`Run.HeldToW`) whose every
  opened tile is accepted (`TilesAccepted`): the bridge from `check` to the statements below, which are on `tieOf`,
  `todo` and `checkTile`.
* `YAtWIsTheCall`, **the run's statement**: over `t` uniform work-weighted tickets, with the prover's openings any
  function of the tickets, the run is accepted on openings its table holds while the tiles that are not tied at W's
  entries (`TileTiedAtW`) hold more than ε of the run's ticket weight with probability at most δ + q / 2^256, wherever
  (1 − ε)^t ≤ δ and the play's finder (`Run.finder`, declared at `q` SHA-512 calls) meets `ecr/sha-512`
  (`Claims.ExpectedCR`). The share is of ticket weight, not of tiles. `CoinBytesGiveUniformTickets` and
  `DrawIsOnItsStreams` carry the tickets to the streams of the run's sample key.
* `PeriodYAtWIsTheCall`, **the period's statement for one sequence of runs**: over the coins of a monitoring period's
  execution phases, each run committed before its own coin (`Runs.CommittedBeforeCoin`) and each sample `t` tickets on
  its phase's coin (`Runs.DrawIsTickets`), the runs of one sequence accepted up to some phase have ∏ (1 − wrong share)
  below 1 − ε with probability at most δ + q / 2^256, wherever (1 − ε)^t ≤ δ, however many phases the period has. So a
  sequence's accepted prefix holds wrong shares summing to at most −ln(1 − ε): at 459 samples and δ = 1/100, about 1.0%
  of one phase's ticket weight over the whole period.
* `FleetYAtWIsTheCall`, **the period's statement pooled across sequences**: for prefix ends fixed before the period's
  coins, over the same coins, some nonempty set `K` of them is accepted while (∏ over `K` of the prefix's
  ∏ (1 − wrong share))^t is below (|K| δ / |ends|)^|K| with probability at most δ + q / 2^256. With each sequence's
  last phase as its end, `N` of them, it is `Pouw.Guarantees.Dither.FleetSound`'s convex pooling at equal weights: the
  sequences of any `K`, `s` of `N` accepted throughout, together hold wrong shares summing to at most
  (s / t) (ln (N / s) + ln (1 / δ)).

**The table and the opener.** A run's commitment fixes its values before the coin: `Run.table` holds them, for each of
W's values and each leaf position at most one opening. It is the prover's, a field of the run, which the verifier never
reads. The prover's openings are any function of the coin (`Opener`, one for each coin): the statements quantify over an
opener that sees the coin, as the real prover does. A tile is tied at W's entries (`TileTiedAtW`) by the table alone.
An accepted opening is held to the table through `OpeningBinds`, which is the lemma here and no hypothesis: where a row
the verdict opens at the coin differs from the table's opening of the same position of the same root, two openings of
one of W's roots are accepted, and SHA-512 collides on the messages checking them hashes (`rowsHashed`). The play's
finder (`Run.finder`, `Runs.finder`) is the first such collision among those messages
(`Properties.Challenger.collisionIn`), a `Claims.Finder` over the play's coins whose cost is declared at the budget `q`
for the play's whole SHA-512 work, the prover's own included, as `Claims.BAt`'s is. Its premise is
`Claims.ExpectedCR` at the compiled SHA-512 (`Verity.sha512Msg`), what `Verity.Vocab.FinderCR` reads at a subprotocol's
finder, and the bound reads δ plus its term, the budget over 2^256 (`Verity.Vocab.finderPr_le_crBound`'s form). There
is no fork and no knowledge form.

The statements count a coin only where the table holds what the opener opens (`Run.Holds`): an opening of each opened
row's position that opens against the same root. They hold for every table. A table that holds every row the opener
opens on any coin always exists (the opener's own opening on one such coin), so the reader loses nothing; a real
prover's table is the data it committed.

**The period.** A monitoring period's execution phases are numbered in time order and each belongs to one sequence
(`Runs.seq`). A node's runs of record are one sequence when they are in time order: execution phase i + 1 is committed
after phase i's verdict, and each phase's coin is flipped after its own commitment. A node whose phases overlap in time
has no such order, so each of its phases is a sequence of its own, and those are pooled convexly (`FleetYAtWIsTheCall`)
and never multiplied. `Runs.CommittedBeforeCoin` is all the statements need of the order: a phase's run (its manifests,
W, its table) reads no coin of its own phase or of a later phase of its sequence. It may read any coin of another
sequence, so nothing is assumed between sequences. That each phase's coin is uniform given everything before is
`Runs.DrawIsTickets` over a uniform coin, which the challenger's log gives a real run
(`FlockSoundness.Assumptions.UniformRandomBytes`, `Properties.Challenger.CoinsFromLog` and `DrawnByDevice`).

The two period statements are separate because their conjunction at one δ is false. A prefix end chosen after the
coins (`PeriodYAtWIsTheCall`'s ∃) is the maximal inequality's, which bounds a tail and no mean; convex pooling needs a
mean, so its ends are fixed before the coins. At two sequences and δ = 1/100, a prover whose every phase is wrong on a
small share reaches the pooled event at ends of its choice with probability 0.010039. `PeriodYAtWIsTheCall` counts
each sequence's accepted prefix to any end, at its own δ; `FleetYAtWIsTheCall` counts the prefixes to the ends it is
given, which may be several of one sequence (each week's last phase), each counting in |ends|.

**What is not tied.**

* The tiles of the wrong share: at most ε of the ticket weight, except with probability δ + q / 2^256.
* The cells that are not credited (`UnitRecord.credited`): filler rows past the call's real rows, which have no y row;
  excluded A rows; the rows of a band that holds a voluntary A row; the columns of a block that holds an excluded or
  voluntary B row. That is the give-up, within V-EX's caps (`LoadedGivesUp`). A tile of no credited cell holds no
  ticket and is never sampled; the exclusion tiles a pass opens are checked as sampled ones are, and the statements
  count ticket weight alone. The check itself holds every real row and every column of a tile it opens
  (`Rule.yReason`); the statements claim the credited cells alone.
* The calls of dropped rounds, which W may register and the run does not hold.
* A version-1 W, which registers no weight's B rows: the statements take a tie whose weights are tied
  (`Tied.weights = none`).
* Whether A's rows at W's entry are the model's activations: the statements say y is the call's function of whatever A
  and B rows the table holds at W's entries.

**Over the bound.** A tile over its inert-work bound that is accepted under `over_cap` `uncredited` passed every other
check, y's included, so it is tied. The wrong share here is therefore at no sample on a wrong tile, (1 − ε)^t ≤ δ,
whatever X_O the run's credit is taken at (`creditAtAssumption`).

**y words are whole.** A cell's y word is the word the table's y row holds at the cell's column (`[j]?`): a row that
ends before the column ties no cell, and `Rule.yReason` refuses such a row before it reads any word of it.

**Proofs, all of them the next step.** `TieMeans` and `AcceptedIsHeldToW` unfold `tieOf` and `check`. The run's
statement: an accepted tile's rows are the table's or the finder outputs a collision (`TileAcceptMeans`, `TieMeans`,
`openingBinds`, and `collisionIn` finding a collision where its list holds one); `yCell` is the tile's cell (`encode`
and `checked` act row by row and column by column); so on accept without a collision no sample is on a wrong tile, and
`Pouw.MissBound` gives (1 − ε)^t. The period's: the product over a sequence is a supermartingale in its own coins, the
others held (`expect_prod_adapted`), with the maximal inequality for the prefix; the pooled one is Markov's inequality
on the ends' mean and AM–GM (`prob_convex_pool_accepted`).
-/

@[expose] public section

namespace Properties.Pouw.PearlPlus

open PouwCommit PouwCheck.PearlPlus PouwCheck.PearlPlus.Words

/-! ## The tie to W -/

/-- **W registers `v` once, as this tree**: exactly one entry of W's `registered` holds the value `v`, and it names the
root `root` (as hex), `leaves` rows of `words` 32-bit words, and the domain `d`'s frame id. -/
def Registers (w : Lean.Json) (v : String) (root : ByteArray) (leaves words : ℕ) (d : RowsDomain) : Prop :=
  ∃ (es : Array Lean.Json) (x : ℕ) (e : Lean.Json) (s : String),
    PouwCheck.PearlPlus.get? w "registered" = some (Lean.Json.arr es) ∧ es[x]? = some e ∧
    PouwCheck.PearlPlus.get? e "value" = some (Lean.Json.str v) ∧
    (∀ (x' : ℕ) (e' : Lean.Json), es[x']? = some e' →
      PouwCheck.PearlPlus.get? e' "value" = some (Lean.Json.str v) → x' = x) ∧
    PouwCheck.PearlPlus.get? e "root" = some (Lean.Json.str s) ∧ pyFromHex s = .ok root ∧
    PouwCheck.PearlPlus.get? e "leaves" = some (Lean.Json.num ⟨(leaves : Int), 0⟩) ∧
    PouwCheck.PearlPlus.get? e "words" = some (Lean.Json.num ⟨(words : Int), 0⟩) ∧
    PouwCheck.PearlPlus.get? e "word_bits" = some (Lean.Json.num ⟨32, 0⟩) ∧
    PouwCheck.PearlPlus.get? e "domain" = some (Lean.Json.str (PouwCheck.PearlPlus.hex d.frame.id))

/-- The round W registers pass `pi` of `n` as: `rounds[pi]`, or `pi` where the run names no rounds. -/
def roundOf (rounds : Option (List ℕ)) (n pi : ℕ) : ℕ := ((rounds.getD (List.range n))[pi]?).getD pi

/-- **What `tieOf` establishes**: for the passes `load` reads from the manifests, an answer `tied` means

* W is a `verity/registration/v2` record and `tied.record` is its digest;
* the rounds, where the run names them, are one per pass;
* W's version is 2 and the weights are tied (`tied.weights = none`), or it states none (version 1) and they are not;
* for each call (pass `pi`, position `pos`) W registers once, as round `roundOf rounds passes.length pi`, the call's A
  rows under the pass's `rootsA[pos]` (`m` rows of `k` words, `activationBinding`'s domain) and its y under the pass's
  `yRoots[pos]`, which the call commits (its real rows of `n` words, `yBinding`'s domain); and, where the weights are
  tied, the weight's B rows under the pass's weight root (`n` rows of `k` words, `weightBinding`'s domain);
* `tied.tables` counts the values the tie wants (`tieEntries`), and every value W registers under one of the tie's
  prefixes is one of them or a call's of a dropped round. -/
def TieMeans : Prop :=
  ∀ (coins : EpochCoins) (manifests : List Lean.Json) (passes : List Pass) (w : Lean.Json)
    (rounds : Option (List ℕ)) (dropped : List ℕ) (tied : Tied),
    manifests.mapM (load coins) = .ok passes → tieOf w (manifests.zip passes) rounds dropped = .ok tied →
    fieldIs (PouwCheck.PearlPlus.get? w "format") Flock.WindowRecord.FORMAT_V2 = true ∧
    Flock.WindowRecord.digestHex w = .ok tied.record ∧
    (∀ ss : List ℕ, rounds = some ss → ss.length = passes.length) ∧
    ((tied.weights = none ∧ wVersion w = .ok W_VERSION) ∨
      (tied.weights = some WEIGHTS_UNTIED ∧ wVersion w = .ok 1)) ∧
    (∀ (pi : ℕ) (p : Pass) (pos : ℕ) (r : UnitRecord), passes[pi]? = some p → p.records[pos]? = some r →
      ∃ (da dy : RowsDomain) (yRoot : ByteArray),
        rowsDomain (activationBinding p.salt r.matmul.index) r.matmul.index r.matmul.m HM96 false = .ok da ∧
        rowsDomain (yBinding p.salt r.matmul.index) r.matmul.index r.realRows HM96 false = .ok dy ∧
        p.yRoots[pos]! = some yRoot ∧
        Registers w s!"pearl-a/{roundOf rounds passes.length pi}/{pos}" p.rootsA[pos]! r.matmul.m r.matmul.k da ∧
        Registers w s!"pearl-y/{roundOf rounds passes.length pi}/{pos}" yRoot r.realRows r.matmul.n dy ∧
        (tied.weights = none → ∃ db : RowsDomain,
          rowsDomain (weightBinding r.matmul.weight) (-2) r.matmul.n HM96 false = .ok db ∧
          Registers w s!"pearl-b/{r.matmul.weight}" ((p.weightRoots.lookup r.matmul.weight).getD ByteArray.empty)
            r.matmul.n r.matmul.k db)) ∧
    ∃ want : List (String × TieEntry),
      tieEntries (manifests.zip passes) rounds tied.weights.isNone = .ok want ∧ tied.tables = want.length ∧
      ∀ (es : Array Lean.Json) (x : ℕ) (e : Lean.Json) (v : String),
        PouwCheck.PearlPlus.get? w "registered" = some (Lean.Json.arr es) → es[x]? = some e →
        PouwCheck.PearlPlus.get? e "value" = some (Lean.Json.str v) →
        TIE_PREFIXES.any (fun t => v.startsWith t) = true →
        (want.lookup v).isSome = true ∨
          (TIE_CALLS.any (fun t => v.startsWith t) = true ∧
            (dropped.map toString).contains ((v.splitOn "/")[1]?.getD "") = true)

/-! ## A cell at W's entries -/

/-- The words of the cell (row `i`, column `j`) of a call of index `index` and width `k`, from A's row and the weight's
row: `checkTile`'s forming (`fixedNoise`, `encode`, `checked`) on the one row and the one column. -/
def cellWords (salt rootB : ByteArray) (index k i j : ℕ) (aRow bRow : ByteArray) : Reject (Ops × Checked) := do
  let wn ← fixedNoise salt rootB k
  let ops ← encode salt rootB wn index k [i] [aRow] [j] [bRow]
  pure (ops, ← checked ops)

/-- **The call's function at one cell**: the y word of (row `i`, column `j`) as its tree stores it, the unscale
(`Rule.yOf`) of the cell's U at its row's and column's α, in the high half of a 32-bit word; `none` where Pearl rejects
the rows or the unscale leaves the finite range. It reads A's row, the weight's row, the salt, the weight's root and the
call's index and width, and nothing else of the call. -/
def yCell (salt rootB : ByteArray) (index k i j : ℕ) (aRow bRow : ByteArray) : Option ℕ :=
  match cellWords salt rootB index k i j aRow bRow with
  | .ok (ops, ch) => (Rule.yOf (ch.u[0]!)[0]! ops.a.alpha[0]! ops.b.alpha[0]!).map (· <<< 16)
  | .error _ => none

/-- **The prover's openings on one coin**: for each (pass, tile id) the verdict opens, the tile's openings or why the
prover's files don't give them, as `check` takes them. -/
abbrev Opener : Type := ℕ → ℕ → Except String TileOpening

/-- **The values a run's commitment fixes**: for each of W's values (`pearl-a/{s}/{c}`, `pearl-y/{s}/{c}`,
`pearl-b/{b}`) and each leaf position, at most one opening. It is fixed with the run, before the coin. -/
abbrev Table : Type := String → ℕ → Option LeafOpening

/-- **The cell (row `i`, column `j`) of round `s`'s call `c` is tied at W's entries by the table `T`**: W registers the
call's A rows, its y and its weight's B rows (`Registers`, at the record `r`'s shape and under its domains, the salt
`salt`'s), `T` holds an opening of A's row `i`, of the weight's row `j` and of y's row `i` at those values, each opens
against the root W registers, and y's word at column `j` is `yCell` of the two rows under the salt and the weight's
registered root. -/
def CellAtW (w : Lean.Json) (T : Table) (salt : ByteArray) (s c : ℕ) (r : UnitRecord) (i j : ℕ) : Prop :=
  ∃ (da db dy : RowsDomain) (rootA rootB yRoot : ByteArray) (oa ob oy : LeafOpening) (y : ℕ),
    rowsDomain (activationBinding salt r.matmul.index) r.matmul.index r.matmul.m HM96 false = .ok da ∧
    rowsDomain (weightBinding r.matmul.weight) (-2) r.matmul.n HM96 false = .ok db ∧
    rowsDomain (yBinding salt r.matmul.index) r.matmul.index r.realRows HM96 false = .ok dy ∧
    Registers w s!"pearl-a/{s}/{c}" rootA r.matmul.m r.matmul.k da ∧
    Registers w s!"pearl-b/{r.matmul.weight}" rootB r.matmul.n r.matmul.k db ∧
    Registers w s!"pearl-y/{s}/{c}" yRoot r.realRows r.matmul.n dy ∧
    T s!"pearl-a/{s}/{c}" i = some oa ∧ T s!"pearl-b/{r.matmul.weight}" j = some ob ∧
    T s!"pearl-y/{s}/{c}" i = some oy ∧
    oa.position = i ∧ ob.position = j ∧ oy.position = i ∧
    opens (.ok da) rootA ROW_SCHEMA oa = true ∧ opens (.ok db) rootB ROW_SCHEMA ob = true ∧
    opens (.ok dy) yRoot ROW_SCHEMA oy = true ∧
    yCell salt rootB r.matmul.index r.matmul.k i j (openedValue oa.value) (openedValue ob.value) = some y ∧
    (words32 (openedValue oy.value))[j]? = some y

/-- **Tile `id` (pass, tile id) is tied at W's entries**: it is a tile of its pass, and the table ties every credited
cell of the tile (`UnitRecord.credited`) at W's entries, the pass registered as round
`roundOf rounds passes.length id.1`. -/
def TileTiedAtW (w : Lean.Json) (rounds : Option (List ℕ)) (passes : List Pass) (T : Table) (id : ℕ × ℕ) : Prop :=
  ∃ (p : Pass) (pos : ℕ) (rows cols : List ℕ),
    passes[id.1]? = some p ∧ p.tile? id.2 = some (pos, rows, cols) ∧
    ∀ i ∈ (p.records[pos]!.credited rows cols).1, ∀ j ∈ (p.records[pos]!.credited rows cols).2,
      CellAtW w T p.salt (roundOf rounds passes.length id.1) pos p.records[pos]! i j

/-! ## An accepted run -/

/-- **Every tile the verdict opens is accepted**: of the tiles `todo` names for the sample `drawn` (each pass's sampled
tiles, then its exclusion tiles), the prover opens each and `checkTile` passes it on its weight's noise, or, under
`over_cap` `uncredited`, finds it only over its inert-work bound. It is what `AcceptMeans` gives of an accepted
verdict's tiles. -/
def TilesAccepted (passes : List Pass) (m : OverCap) (opener : Opener) (drawn : List (ℕ × ℕ)) : Prop :=
  ∀ (pi tid : ℕ), (pi, tid) ∈ todo passes drawn → ∃ (p : Pass) (o : TileOpening),
    passes[pi]? = some p ∧ opener pi tid = .ok o ∧
    (checkTile p (noiseOf p tid) tid o = .pass ∨
      m = .uncredited ∧ ∃ why, checkTile p (noiseOf p tid) tid o = .overCap why)

/-- One row of one of W's trees as a tile's opening opens it: W's value, the tree's domain and root as `checkTile`
reads them from the pass, and the opening. -/
structure RowAtW where
  value : String
  domain : Except String RowsDomain
  root : ByteArray
  opening : LeafOpening

/-- **The rows of W's trees an opening of tile `tid` opens**, the pass registered as round `s`: its A rows at
`pearl-a/{s}/{c}`, its B rows at `pearl-b/{b}` and, on a call that commits y, its y rows at `pearl-y/{s}/{c}`, each
against the root and under the domain `checkTile` checks it with. None for a tile that is not the pass's. -/
def tileRows (s : ℕ) (p : Pass) (tid : ℕ) (o : TileOpening) : List RowAtW :=
  match p.tile? tid with
  | none => []
  | some (pos, _, _) =>
    let r := p.records[pos]!
    let u := r.matmul
    let a : List RowAtW := o.a.map (fun x =>
      ({ value := s!"pearl-a/{s}/{pos}",
         domain := rowsDomain (activationBinding p.salt u.index) u.index u.m HM96 false,
         root := p.rootsA[pos]!, opening := x } : RowAtW))
    let b : List RowAtW := o.b.map (fun x =>
      ({ value := s!"pearl-b/{u.weight}",
         domain := rowsDomain (weightBinding u.weight) (-2) u.n HM96 false,
         root := (p.weightRoots.lookup u.weight).getD ByteArray.empty, opening := x } : RowAtW))
    let y : List RowAtW :=
      match o.y, p.yRoots[pos]! with
      | .ok ys, some yRoot => ys.map (fun x =>
        ({ value := s!"pearl-y/{s}/{pos}",
           domain := rowsDomain (yBinding p.salt u.index) u.index r.realRows HM96 false,
           root := yRoot, opening := x } : RowAtW))
      | _, _ => []
    a ++ b ++ y

/-- **A run and the registration record it is tied to**, all of it fixed before the run's coin: the epoch's coins, the
retained passes' manifests and the passes read from them, W with the passes' rounds and the dropped rounds, the rule on
a tile over its bound, and the values the commitment fixes (`Table`). The prover's openings are not of it: they are a
function of the coin. -/
structure Run where
  coins : EpochCoins
  manifests : List Lean.Json
  passes : List Pass
  w : Lean.Json
  rounds : Option (List ℕ)
  dropped : List ℕ
  overCap : OverCap
  table : Table

/-- **The run is held to W**: its passes are `load`'s of its manifests under the epoch's coins, and `tieOf` answers
for them with the weights tied (a version-2 W). -/
def Run.HeldToW (R : Run) : Prop :=
  R.manifests.mapM (load R.coins) = .ok R.passes ∧
    ∃ tied : Tied, tieOf R.w (R.manifests.zip R.passes) R.rounds R.dropped = .ok tied ∧ tied.weights = none

/-- **The run is accepted on the sample `drawn` with the openings `o`**: it is held to W and every tile the verdict
opens is accepted. -/
def Run.Accepts (R : Run) (o : Opener) (drawn : List (ℕ × ℕ)) : Prop :=
  R.HeldToW ∧ TilesAccepted R.passes R.overCap o drawn

/-- **The run's table holds what `o` opens on the sample `drawn`**: for every row of W's trees the openings of the
verdict's tiles open (`tileRows`), the table holds an opening of that position at that value which opens against the
same root. -/
def Run.Holds (R : Run) (o : Opener) (drawn : List (ℕ × ℕ)) : Prop :=
  ∀ (pi tid : ℕ) (p : Pass) (op : TileOpening), (pi, tid) ∈ todo R.passes drawn → R.passes[pi]? = some p →
    o pi tid = .ok op → ∀ e ∈ tileRows (roundOf R.rounds R.passes.length pi) p tid op,
      ∃ o' : LeafOpening, R.table e.value e.opening.position = some o' ∧ o'.position = e.opening.position ∧
        opens e.domain e.root ROW_SCHEMA o' = true

/-- **The messages SHA-512 hashes in checking a play's row openings against the table's**: for each row of W's trees
the openings `o` of the verdict's tiles open on the sample `drawn`, where the table holds an opening of its position,
the messages checking the two hashes (`rowsHashed`). -/
def Run.hashed (R : Run) (o : Opener) (drawn : List (ℕ × ℕ)) : List ByteArray :=
  (todo R.passes drawn).flatMap fun id =>
    match R.passes[id.1]?, o id.1 id.2 with
    | some p, .ok op =>
      (tileRows (roundOf R.rounds R.passes.length id.1) p id.2 op).flatMap fun e =>
        match e.domain, R.table e.value e.opening.position with
        | .ok d, some o' => rowsHashed d e.opening ++ rowsHashed d o'
        | _, _ => []
    | _, _ => []

/-- A pair of byte strings as a pair of messages, as `Claims.Finder` outputs one. -/
def msgPair (xy : ByteArray × ByteArray) : _root_.Claims.Msg × _root_.Claims.Msg := (xy.1.data.toList, xy.2.data.toList)

/-- The run a verdict's input is, with its loaded passes, the registration record it names and a table. -/
def runOf (i : Input) (passes : List Pass) (w : Lean.Json) (T : Table) : Run :=
  { coins := i.coins, manifests := i.manifests, passes := passes, w := w, rounds := i.rounds, dropped := i.dropped,
    overCap := i.overCap, table := T }

/-- **An accepted verdict on an input that names W is a run held to W**: the plan's passes are W's by `tieOf`, the
verdict answers with that tie, and, where the tie's weights are tied, the run (`runOf`, at any table) is accepted on
the plan's sample with the openings the verdict took (`Run.Accepts`). -/
def AcceptedIsHeldToW : Prop :=
  ∀ (i : Input) (opener : Opener) (w : Lean.Json), (check i opener).accept = true →
    i.tie = some w → ∃ (pl : Plan) (tied : Tied), plan i = .ok pl ∧
      tieOf w (i.manifests.zip pl.passes) i.rounds i.dropped = .ok tied ∧ (check i opener).tie = some tied ∧
      (tied.weights = none → ∀ T : Table, (runOf i pl.passes w T).Accepts opener pl.picks)

/-! ## The run's statement -/

/-- `t` tickets of the tiles `tk`: one sample of the run. -/
abbrev TicketDraws (tk : List ((ℕ × ℕ) × ℕ)) (t : ℕ) : Type := Fin t → _root_.Pouw.Ticket (weightsOf tk) tk.length

/-- The tiles `t` tickets name, in the sample's order. -/
def drawnAt (tk : List ((ℕ × ℕ) × ℕ)) {t : ℕ} (c : TicketDraws tk t) : List (ℕ × ℕ) :=
  List.ofFn fun j => tileOf tk (c j)

open Classical in
/-- **The run's wrong ticket weight**: the tickets of its tiles that its table does not tie at W's entries. -/
noncomputable def Run.wrongWeight (R : Run) : ℕ :=
  ∑ u : Fin (tickets R.passes).length,
    if TileTiedAtW R.w R.rounds R.passes R.table ((tickets R.passes).get u).1 then 0
    else weightsOf (tickets R.passes) u

/-- **The run's wrong share**: its wrong ticket weight over its tickets' total (0 where the run holds no ticket). -/
noncomputable def Run.wrongShare (R : Run) : ℝ :=
  (R.wrongWeight : ℝ) / (ticketTotal (tickets R.passes) : ℝ)

/-- **The run's finder**: on the tickets `c`, the first SHA-512 collision (`Properties.Challenger.collisionIn`) among
the messages checking the opener's rows on `c` and the table's hashes (`Run.hashed`). Its coins are the play's, `t`
tickets, and its cost is declared at the budget `q` for the play's whole SHA-512 work, the prover's own included, which
the reader computes from the prover's time (as `Claims.BAt`). -/
noncomputable def Run.finder (R : Run) (t : ℕ) [Nonempty (TicketDraws (tickets R.passes) t)]
    (O : TicketDraws (tickets R.passes) t → Opener) (q : ℕ) : _root_.Claims.Finder where
  S := TicketDraws (tickets R.passes) t
  out c := (_root_.Properties.Challenger.collisionIn (R.hashed (O c) (drawnAt (tickets R.passes) c))).map msgPair
  cost _ := q

/-- **PoUW's guarantee at W's entries, over every tile, for an opener that sees the coin.** For every run, number of
samples `t`, opener `O` (the prover's openings on each `t` tickets), budget `q` and ε, δ with (1 − ε)^t ≤ δ, where the
play's finder (`Run.finder`) meets `ecr/sha-512` (`Claims.ExpectedCR` at the compiled SHA-512): over `t` uniform
work-weighted tickets, the run is accepted on the tiles they name with openings its table holds (`Run.Holds`) while
the tiles its table does not tie at W's entries hold more than ε of its ticket weight, with probability at most
δ + q / 2^256. So except with that, on accept, every credited cell's y at W's entries is the call's function of the A
row and the weight's row at W's entries (`yCell`), apart from tiles of at most ε of the ticket weight. -/
def YAtWIsTheCall : Prop :=
  ∀ (R : Run) (t : ℕ) [Nonempty (TicketDraws (tickets R.passes) t)]
    (O : TicketDraws (tickets R.passes) t → Opener) (q : ℕ) (ε δ : ℝ),
    0 ≤ ε → ε ≤ 1 → (1 - ε) ^ t ≤ δ →
    _root_.Claims.ExpectedCR _root_.Verity.sha512Msg (R.finder t O q) →
    _root_.Pouw.pr (fun c : TicketDraws (tickets R.passes) t =>
      R.Accepts (O c) (drawnAt (tickets R.passes) c) ∧ R.Holds (O c) (drawnAt (tickets R.passes) c) ∧
        ε < R.wrongShare) ≤ δ + (q : ℝ) / 2 ^ (256 : ℝ)

/-! ## The period's statements -/

/-- **The runs of a monitoring period's execution phases on its coins**: the phases numbered in time order, one coin a
phase. `seq k` is the sequence phase `k` belongs to: a node's runs of record in time order are one sequence, and a
phase that overlaps another of its node in time is a sequence of its own. On the period's coins `x`, phase `k`'s run,
the tiles its coin samples (`none` where the sample refuses) and the prover's openings, each any function of the
coins. -/
structure Runs (Ω : Type) (n : ℕ) where
  seq : Fin n → ℕ
  run : (Fin n → Ω) → Fin n → Run
  drawn : (Fin n → Ω) → Fin n → Option (List (ℕ × ℕ))
  opener : (Fin n → Ω) → Fin n → Opener

/-- **Each phase's run is committed before its coin**: phase `k`'s run (its manifests, W, its rounds and its table) is
one function of the coins other than its own and its sequence's later ones, and its sample of those and its own. A run
may read every earlier coin of its sequence and any coin of another sequence, never its own before it commits. The
opener is not held to it: it may read every coin. -/
def Runs.CommittedBeforeCoin {Ω : Type} {n : ℕ} (P : Runs Ω n) : Prop :=
  (∀ (x x' : Fin n → Ω) (k : Fin n), (∀ v : Fin n, (P.seq v = P.seq k → v < k) → x v = x' v) →
    P.run x k = P.run x' k) ∧
  ∀ (x x' : Fin n → Ω) (k : Fin n), (∀ v : Fin n, (P.seq v = P.seq k → v ≤ k) → x v = x' v) →
    P.drawn x k = P.drawn x' k

/-- **Each phase's sample is `t` tickets on its coin**: over the phase's own coin, the others held, the tiles it
samples fall in any event no more often than the tiles of `t` uniform tickets of its run do. A coin on which the sample
refuses falls in none. For the verdict's sample it is `DrawIsOnItsStreams` with `CoinBytesGiveUniformTickets`, the
phase's coin its sample key's streams. -/
def Runs.DrawIsTickets {Ω : Type} [Fintype Ω] {n : ℕ} (P : Runs Ω n) (t : ℕ) : Prop :=
  ∀ (x : Fin n → Ω) (k : Fin n) (E : List (ℕ × ℕ) → Prop),
    _root_.Pouw.pr (fun ω : Ω => ∃ d : List (ℕ × ℕ), P.drawn (Function.update x k ω) k = some d ∧ E d) ≤
      _root_.Pouw.pr (fun c : TicketDraws (tickets (P.run x k).passes) t =>
        E (drawnAt (tickets (P.run x k).passes) c))

/-- **Phase `k`'s run is accepted** on the period's coins `x`: its coin samples, and the run is accepted on that sample
with the prover's openings, which its table holds. -/
def Runs.Accepts {Ω : Type} {n : ℕ} (P : Runs Ω n) (x : Fin n → Ω) (k : Fin n) : Prop :=
  ∃ d : List (ℕ × ℕ), P.drawn x k = some d ∧ (P.run x k).Accepts (P.opener x k) d ∧
    (P.run x k).Holds (P.opener x k) d

/-- **Phase `k` and every earlier phase of its sequence are accepted** on the period's coins `x`: the sequence's
accepted prefix reaches `k`. -/
def Runs.AcceptedUpTo {Ω : Type} {n : ℕ} (P : Runs Ω n) (x : Fin n → Ω) (k : Fin n) : Prop :=
  ∀ w : Fin n, P.seq w = P.seq k → w ≤ k → P.Accepts x w

/-- **The tied share of a sequence's prefix**: ∏ (1 − wrong share) over phase `k` and the earlier phases of its
sequence, each a share of its own run's ticket weight. -/
noncomputable def Runs.tiedUpTo {Ω : Type} {n : ℕ} (P : Runs Ω n) (x : Fin n → Ω) (k : Fin n) : ℝ :=
  ∏ w : Fin n, (if P.seq w = P.seq k ∧ w ≤ k then 1 - (P.run x w).wrongShare else 1)

/-- The messages a period's play hashes in checking its row openings against its tables (`Run.hashed`), phase by
phase. -/
def Runs.hashed {Ω : Type} {n : ℕ} (P : Runs Ω n) (x : Fin n → Ω) : List ByteArray :=
  (List.finRange n).flatMap fun k =>
    match P.drawn x k with
    | some d => (P.run x k).hashed (P.opener x k) d
    | none => []

/-- **The period's finder**: on the period's coins, the first SHA-512 collision among the messages its play hashes
(`Runs.hashed`), its cost declared at the budget `q` for the period's whole SHA-512 work (as `Run.finder`). -/
noncomputable def Runs.finder {Ω : Type} [Fintype Ω] [Nonempty Ω] {n : ℕ} (P : Runs Ω n) (q : ℕ) :
    _root_.Claims.Finder where
  S := Fin n → Ω
  out x := (_root_.Properties.Challenger.collisionIn (P.hashed x)).map msgPair
  cost _ := q

/-- **PoUW's period bound for the tile sample, one sequence.** For a monitoring period of `n` execution phases, each
run committed before its coin (`Runs.CommittedBeforeCoin`) and each sample `t` tickets on its phase's coin
(`Runs.DrawIsTickets`), ε, δ with (1 − ε)^t ≤ δ and a budget `q` at which the period's finder meets `ecr/sha-512`: for
each sequence `a`, over the period's coins, some phase of `a` and every earlier one of `a` are accepted while those
phases' ∏ (1 − wrong share) is below 1 − ε, with probability at most δ + q / 2^256, whatever `n` is. So except with
that, the phases of a sequence before its first refusal together hold wrong shares summing to at most −ln(1 − ε), each
a share of its own phase's ticket weight. It bounds the accepted prefix alone: a phase accepted after a refused one of
its sequence is not counted. A node whose phases overlap in time has one-phase sequences, for which this is the run's
statement; `FleetYAtWIsTheCall` pools them. -/
def PeriodYAtWIsTheCall : Prop :=
  ∀ {Ω : Type} [Fintype Ω] [Nonempty Ω] {n : ℕ} (P : Runs Ω n) (t q : ℕ) (ε δ : ℝ),
    0 ≤ ε → ε ≤ 1 → (1 - ε) ^ t ≤ δ →
    P.CommittedBeforeCoin → P.DrawIsTickets t →
    _root_.Claims.ExpectedCR _root_.Verity.sha512Msg (P.finder q) →
    ∀ a : ℕ, _root_.Pouw.pr (fun x : Fin n → Ω => ∃ k : Fin n, P.seq k = a ∧ P.AcceptedUpTo x k ∧
      P.tiedUpTo x k < 1 - ε) ≤ δ + (q : ℝ) / 2 ^ (256 : ℝ)

/-- **PoUW's period bound for the tile sample, pooled convexly across sequences.** For a monitoring period as in
`PeriodYAtWIsTheCall`, prefix ends `ends` fixed before its coins and 0 < δ: over the period's coins, some nonempty
`K ⊆ ends` has every prefix accepted while (∏ over `K` of the prefix's ∏ (1 − wrong share))^t is below
(|K| δ / |ends|)^|K|, with probability at most δ + q / 2^256. With `ends` each sequence's last phase, `N` of them, it is
`Pouw.Guarantees.Dither.FleetSound`'s pooling at equal weights: except with that, the sequences of any `K`, `s` of them
accepted throughout, together hold wrong shares summing to at most (s / t) (ln (N / s) + ln (1 / δ)). Nothing is
assumed between sequences, so a node whose phases overlap in time enters as one-phase sequences and is pooled here,
never multiplied. A sequence refused before its end in `ends` is not counted; giving it earlier ends counts its
prefixes to them, each end counting in |ends|. -/
def FleetYAtWIsTheCall : Prop :=
  ∀ {Ω : Type} [Fintype Ω] [Nonempty Ω] {n : ℕ} (P : Runs Ω n) (ends : Finset (Fin n)) (t q : ℕ) (δ : ℝ), 0 < δ →
    P.CommittedBeforeCoin → P.DrawIsTickets t →
    _root_.Claims.ExpectedCR _root_.Verity.sha512Msg (P.finder q) →
    _root_.Pouw.pr (fun x : Fin n → Ω => ∃ K : Finset (Fin n), K ⊆ ends ∧ K.Nonempty ∧
      (∀ k ∈ K, P.AcceptedUpTo x k) ∧
      (∏ k ∈ K, P.tiedUpTo x k) ^ t < ((K.card : ℝ) * δ / (ends.card : ℝ)) ^ K.card) ≤ δ + (q : ℝ) / 2 ^ (256 : ℝ)

end Properties.Pouw.PearlPlus
