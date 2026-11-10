import Properties.Pouw.PearlPlus
import Properties.Pouw.PearlPlusDraw

/-!
# Pearl+'s served y at the window record's entries (`PouwCheck.PearlPlus.tieOf`, `checkTile`, `todo`)

What an accepted run establishes about the values the window record W registers, stated against the verdict's own
functions. These are statements only: none is proved here, and none is a guarantee yet.

A run tied to W (`--tie W`) registers, for pass `s`'s call `c` and each weight `b`, three trees in W: the call's A rows
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
* `YAtWIsTheCall`, **the run's statement**: over `t` uniform work-weighted tickets, the run is accepted while the tiles
  that are not tied at W's entries (`TileTiedAtW`) hold more than ε of the run's ticket weight with probability at most
  δ, wherever (1 − ε)^t ≤ δ. The share is of ticket weight, not of tiles. `CoinBytesGiveUniformTickets` and
  `DrawIsOnItsStreams` carry the tickets to the streams of the run's draw key.
* `PeriodYAtWIsTheCall`, **the period's statement**: over the coins of a period's windows, each run committed before
  its own coin and free to read every earlier one (`Runs.CommittedBeforeCoin`), each draw `t` tickets on its window's
  coin (`Runs.DrawIsTickets`), the runs accepted up to some window have ∏ (1 − wrong share) below 1 − ε with
  probability at most δ, wherever (1 − ε)^t ≤ δ, however many windows the period has. So the accepted prefix's wrong
  shares sum to at most −ln(1 − ε): at 459 draws and δ = 1/100, about 1.0% of one window's ticket weight. It counts
  the windows before the first refusal alone.

**A tile is tied at W's entries** (`TileTiedAtW`) when the prover's opening of it opens, for each credited cell, A's
row, the weight's row and y's row against the roots W registers for the call and the weight, and y's word at the column
is `yCell` of the two opened rows. `OpeningBinds` makes those rows the only ones W's entries open to, or SHA-512
collides on the run's own messages, so the statement needs no collision case of its own.

**What is not tied.**

* The tiles of the wrong share: at most ε of the ticket weight, except with probability δ.
* The cells that are not credited (`UnitRecord.credited`): filler rows past the call's real rows, which have no y row;
  excluded A rows; the rows of a band that holds a voluntary A row; the columns of a block that holds an excluded or
  voluntary B row. That is the give-up, within V-EX's caps (`LoadedGivesUp`). A tile of no credited cell holds no
  ticket and is never drawn; the exclusion tiles a pass opens are checked as drawn ones are, and the statements count
  ticket weight alone. The check itself holds every real row and every column of a tile it opens (`Rule.yReason`);
  the statements claim the credited cells alone.
* The calls of dropped rounds, which W may register and the run does not hold.
* A version-1 W, which registers no weight's B rows: the statements take a tie whose weights are tied
  (`Tied.weights = none`).
* Whether A's rows at W's entry are the model's activations: the statements say y is the call's function of whatever A
  and B rows W's entries open to.

**The prover's openings** are a function of (pass, tile) fixed with the run (`Run.opener`), before the draw. A prover
that chooses what it opens after the coin is not covered: that form would replace the opening by one extracted from the
prover's answers on other coins.

**Over the bound.** A tile over its inert-work bound that is accepted under `over_cap` `uncredited` passed every other
check, y's included, so it is tied. The wrong share here is therefore at no draw on a wrong tile, (1 − ε)^t ≤ δ,
whatever X_O the run's credit is taken at (`creditAtAssumption`).

**y words are read as the check reads them** (`words32`, `[j]!`): a y row of fewer than `n` words reads 0 past its end,
as `Rule.yReason` does.
-/

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

/-- **The cell (row `i`, column `j`) of round `s`'s call `c` is tied at W's entries by the opening `o`**: W registers the
call's A rows, its y and its weight's B rows (`Registers`, at the record `r`'s shape and under its domains, the salt
`salt`'s), `o` opens A's row `i`, the weight's row `j` and y's row `i` against those roots, and y's word at column `j`
is `yCell` of the two opened rows under the salt and the weight's registered root. -/
def CellAtW (w : Lean.Json) (salt : ByteArray) (s c : ℕ) (r : UnitRecord) (i j : ℕ) (o : TileOpening) : Prop :=
  ∃ (da db dy : RowsDomain) (rootA rootB yRoot : ByteArray) (oa ob oy : LeafOpening) (ys : List LeafOpening),
    rowsDomain (activationBinding salt r.matmul.index) r.matmul.index r.matmul.m HM96 false = .ok da ∧
    rowsDomain (weightBinding r.matmul.weight) (-2) r.matmul.n HM96 false = .ok db ∧
    rowsDomain (yBinding salt r.matmul.index) r.matmul.index r.realRows HM96 false = .ok dy ∧
    Registers w s!"pearl-a/{s}/{c}" rootA r.matmul.m r.matmul.k da ∧
    Registers w s!"pearl-b/{r.matmul.weight}" rootB r.matmul.n r.matmul.k db ∧
    Registers w s!"pearl-y/{s}/{c}" yRoot r.realRows r.matmul.n dy ∧
    oa ∈ o.a ∧ ob ∈ o.b ∧ o.y = .ok ys ∧ oy ∈ ys ∧ oa.position = i ∧ ob.position = j ∧ oy.position = i ∧
    opens (.ok da) rootA ROW_SCHEMA oa = true ∧ opens (.ok db) rootB ROW_SCHEMA ob = true ∧
    opens (.ok dy) yRoot ROW_SCHEMA oy = true ∧
    yCell salt rootB r.matmul.index r.matmul.k i j (openedValue oa.value) (openedValue ob.value) =
      some ((words32 (openedValue oy.value))[j]!)

/-- **Tile `id` (pass, tile id) is tied at W's entries**: it is a tile of its pass, the prover opens it, and that
opening ties every credited cell of the tile (`UnitRecord.credited`) at W's entries, the pass registered as round
`roundOf rounds passes.length id.1`. -/
def TileTiedAtW (w : Lean.Json) (rounds : Option (List ℕ)) (passes : List Pass)
    (opener : ℕ → ℕ → Except String TileOpening) (id : ℕ × ℕ) : Prop :=
  ∃ (p : Pass) (pos : ℕ) (rows cols : List ℕ) (o : TileOpening),
    passes[id.1]? = some p ∧ p.tile? id.2 = some (pos, rows, cols) ∧ opener id.1 id.2 = .ok o ∧
    ∀ i ∈ (p.records[pos]!.credited rows cols).1, ∀ j ∈ (p.records[pos]!.credited rows cols).2,
      CellAtW w p.salt (roundOf rounds passes.length id.1) pos p.records[pos]! i j o

/-! ## An accepted run -/

/-- **Every tile the verdict opens is accepted**: of the tiles `todo` names for the draw `drawn` (each pass's drawn
tiles, then its exclusion tiles), the prover opens each and `checkTile` passes it on its weight's noise, or, under
`over_cap` `uncredited`, finds it only over its inert-work bound. It is what `AcceptMeans` gives of an accepted
verdict's tiles. -/
def TilesAccepted (passes : List Pass) (m : OverCap) (opener : ℕ → ℕ → Except String TileOpening)
    (drawn : List (ℕ × ℕ)) : Prop :=
  ∀ (pi tid : ℕ), (pi, tid) ∈ todo passes drawn → ∃ (p : Pass) (o : TileOpening),
    passes[pi]? = some p ∧ opener pi tid = .ok o ∧
    (checkTile p (noiseOf p tid) tid o = .pass ∨
      m = .uncredited ∧ ∃ why, checkTile p (noiseOf p tid) tid o = .overCap why)

/-- **A run and the window record it is tied to**, all of it fixed before the run's draw: the epoch's coins, the
retained passes' manifests and the passes read from them, W with the passes' rounds and the dropped rounds, the rule on
a tile over its bound, and the prover's openings. -/
structure Run where
  coins : EpochCoins
  manifests : List Lean.Json
  passes : List Pass
  w : Lean.Json
  rounds : Option (List ℕ)
  dropped : List ℕ
  overCap : OverCap
  opener : ℕ → ℕ → Except String TileOpening

/-- **The run is held to W**: its passes are `load`'s of its manifests under the epoch's coins, and `tieOf` answers
for them with the weights tied (a version-2 W). -/
def Run.HeldToW (R : Run) : Prop :=
  R.manifests.mapM (load R.coins) = .ok R.passes ∧
    ∃ tied : Tied, tieOf R.w (R.manifests.zip R.passes) R.rounds R.dropped = .ok tied ∧ tied.weights = none

/-- **The run is accepted on the draw `drawn`**: it is held to W and every tile the verdict opens is accepted. -/
def Run.Accepts (R : Run) (drawn : List (ℕ × ℕ)) : Prop :=
  R.HeldToW ∧ TilesAccepted R.passes R.overCap R.opener drawn

/-- The run a verdict's input is, with its loaded passes, the window record it names and the prover's openings. -/
def runOf (i : Input) (passes : List Pass) (w : Lean.Json) (opener : ℕ → ℕ → Except String TileOpening) : Run :=
  { coins := i.coins, manifests := i.manifests, passes := passes, w := w, rounds := i.rounds, dropped := i.dropped,
    overCap := i.overCap, opener := opener }

/-- **An accepted verdict on an input that names W is a run held to W**: the plan's passes are W's by `tieOf`, the
verdict answers with that tie, and, where the tie's weights are tied, the run (`runOf`) is accepted on the plan's draw
(`Run.Accepts`). -/
def AcceptedIsHeldToW : Prop :=
  ∀ (i : Input) (opener : ℕ → ℕ → Except String TileOpening) (w : Lean.Json), (check i opener).accept = true →
    i.tie = some w → ∃ (pl : Plan) (tied : Tied), plan i = .ok pl ∧
      tieOf w (i.manifests.zip pl.passes) i.rounds i.dropped = .ok tied ∧ (check i opener).tie = some tied ∧
      (tied.weights = none → (runOf i pl.passes w opener).Accepts pl.picks)

/-! ## The run's statement -/

/-- `t` tickets of the tiles `tk`: one draw of the run. -/
abbrev TicketDraws (tk : List ((ℕ × ℕ) × ℕ)) (t : ℕ) : Type := Fin t → _root_.Pouw.Ticket (weightsOf tk) tk.length

/-- The tiles `t` tickets name, in the draw's order. -/
def drawnAt (tk : List ((ℕ × ℕ) × ℕ)) {t : ℕ} (c : TicketDraws tk t) : List (ℕ × ℕ) :=
  List.ofFn fun j => tileOf tk (c j)

open Classical in
/-- **The run's wrong ticket weight**: the tickets of its tiles that are not tied at W's entries. -/
noncomputable def Run.wrongWeight (R : Run) : ℕ :=
  ∑ u : Fin (tickets R.passes).length,
    if TileTiedAtW R.w R.rounds R.passes R.opener ((tickets R.passes).get u).1 then 0
    else weightsOf (tickets R.passes) u

/-- **The run's wrong share**: its wrong ticket weight over its tickets' total (0 where the run holds no ticket). -/
noncomputable def Run.wrongShare (R : Run) : ℝ :=
  (R.wrongWeight : ℝ) / (ticketTotal (tickets R.passes) : ℝ)

/-- **PoUW's guarantee at W's entries, over every tile.** For every run, number of draws `t` and ε, δ with
(1 − ε)^t ≤ δ: over `t` uniform work-weighted tickets, the run is accepted on the tiles they name while its tiles that
are not tied at W's entries hold more than ε of its ticket weight, with probability at most δ. So except with δ, on
accept, every credited cell's y at W's entries is the call's function of the A row and the weight's row at W's entries
(`yCell`), apart from tiles of at most ε of the ticket weight. -/
def YAtWIsTheCall : Prop :=
  ∀ (R : Run) (t : ℕ) (ε δ : ℝ), 0 ≤ ε → ε ≤ 1 → (1 - ε) ^ t ≤ δ →
    _root_.Pouw.pr (fun c : TicketDraws (tickets R.passes) t =>
      R.Accepts (drawnAt (tickets R.passes) c) ∧ ε < R.wrongShare) ≤ δ

/-! ## The period's statement -/

/-- **The runs of a period's windows on its coins**: on the period's coins `x`, one coin a window, window `k`'s run and
the tiles its coin draws (`none` where the draw refuses), each any function of the coins. -/
structure Runs (Ω : Type) (n : ℕ) where
  run : (Fin n → Ω) → Fin n → Run
  drawn : (Fin n → Ω) → Fin n → Option (List (ℕ × ℕ))

/-- **Each window's run is committed before its coin**: window `k`'s run (its manifests, W, its rounds and the prover's
openings) is one function of the earlier windows' coins, and its draw of those and its own. A run may read every earlier
coin, never its own before it commits. -/
def Runs.CommittedBeforeCoin {Ω : Type} {n : ℕ} (P : Runs Ω n) : Prop :=
  (∀ (x x' : Fin n → Ω) (k : Fin n), (∀ v : Fin n, v < k → x v = x' v) → P.run x k = P.run x' k) ∧
  ∀ (x x' : Fin n → Ω) (k : Fin n), (∀ v : Fin n, v ≤ k → x v = x' v) → P.drawn x k = P.drawn x' k

/-- **Each window's draw is `t` tickets on its coin**: over the window's own coin, the others held, the tiles it draws
fall in any event no more often than the tiles of `t` uniform tickets of its run do. A coin on which the draw refuses
falls in none. For the verdict's draw it is `DrawIsOnItsStreams` with `CoinBytesGiveUniformTickets`, the window's coin
its draw key's streams. -/
def Runs.DrawIsTickets {Ω : Type} [Fintype Ω] {n : ℕ} (P : Runs Ω n) (t : ℕ) : Prop :=
  ∀ (x : Fin n → Ω) (k : Fin n) (E : List (ℕ × ℕ) → Prop),
    _root_.Pouw.pr (fun ω : Ω => ∃ d : List (ℕ × ℕ), P.drawn (Function.update x k ω) k = some d ∧ E d) ≤
      _root_.Pouw.pr (fun c : TicketDraws (tickets (P.run x k).passes) t =>
        E (drawnAt (tickets (P.run x k).passes) c))

/-- **Window `k`'s run is accepted** on the period's coins `x`: its coin draws, and the run is accepted on that draw. -/
def Runs.Accepts {Ω : Type} {n : ℕ} (P : Runs Ω n) (x : Fin n → Ω) (k : Fin n) : Prop :=
  ∃ d : List (ℕ × ℕ), P.drawn x k = some d ∧ (P.run x k).Accepts d

/-- **Window `k` and every earlier one are accepted** on the period's coins `x`. -/
def Runs.AcceptedUpTo {Ω : Type} {n : ℕ} (P : Runs Ω n) (x : Fin n → Ω) (k : Fin n) : Prop :=
  ∀ w : Fin n, w ≤ k → P.Accepts x w

/-- **PoUW's period bound for the tile draw.** For a period of `n` windows, each run committed before its coin
(`Runs.CommittedBeforeCoin`) and each draw `t` tickets on its window's coin (`Runs.DrawIsTickets`), and ε, δ with
(1 − ε)^t ≤ δ: over the period's coins, some window and every earlier one are accepted while those windows'
∏ (1 − wrong share) is below 1 − ε, with probability at most δ, whatever `n` is. So except with δ, the windows before
the first refusal together hold wrong shares summing to at most −ln(1 − ε), each a share of its own window's ticket
weight. It bounds the accepted prefix alone: a window accepted after a refused one is not counted. -/
def PeriodYAtWIsTheCall : Prop :=
  ∀ {Ω : Type} [Fintype Ω] {n : ℕ} (P : Runs Ω n) (t : ℕ) (ε δ : ℝ), 0 ≤ ε → ε ≤ 1 → (1 - ε) ^ t ≤ δ →
    P.CommittedBeforeCoin → P.DrawIsTickets t →
    _root_.Pouw.pr (fun x : Fin n → Ω => ∃ k : Fin n, P.AcceptedUpTo x k ∧
      (∏ w : Fin n, (if w ≤ k then 1 - (P.run x w).wrongShare else 1)) < 1 - ε) ≤ δ

end Properties.Pouw.PearlPlus
