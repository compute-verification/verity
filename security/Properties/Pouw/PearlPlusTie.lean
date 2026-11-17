module

public import Properties.Pouw.PearlPlus
public import Properties.Pouw.PearlPlusSample
public import Definitions.Core.Interaction
public import Flock.Hash

/-! What an accepted Pearl+ run establishes about the values its registration record W registers: statements only,
none proved and none a guarantee. -/

@[expose] public section

namespace Properties.Pouw.PearlPlus

open PouwCommit PouwCheck.PearlPlus PouwCheck.PearlPlus.Words

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

/-- **What `tieOf` establishes**: W is a `verity/registration/v2` record with digest `tied.record`, of version 2
exactly when the weights are tied, registering once each call's A rows and y and each tied weight's B rows under
the pass's roots; any other value under the tie's prefixes is a dropped round's call. -/
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

/-- The words of the cell (row `i`, column `j`) of a call of index `index` and width `k`, from A's row and the weight's
row: `checkTile`'s forming (`fixedNoise`, `encode`, `checked`) on the one row and the one column. -/
def cellWords (salt rootB : ByteArray) (index k i j : ℕ) (aRow bRow : ByteArray) : Reject (Ops × Checked) := do
  let wn ← fixedNoise salt rootB k
  let ops ← encode salt rootB wn index k [i] [aRow] [j] [bRow]
  pure (ops, ← checked ops)

/-- **The call's function at one cell**: the y word of (row `i`, column `j`) as its tree stores it, the unscale of
the cell's U in a 32-bit word's high half, `none` where Pearl rejects the rows or the unscale leaves the finite
range. It reads nothing of the call but its index and width. -/
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

/-- **The cell (row `i`, column `j`) of round `s`'s call `c` is tied at W's entries by `T`**: W registers the call's
A rows, its y and its weight's B rows, `T` opens A's row `i`, the weight's row `j` and y's row `i` against W's
roots, and y's word at column `j` is `yCell` of the two rows. -/
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

/-- **Every tile the verdict opens is accepted**: of the tiles `todo` names for the sample `drawn`, the prover opens
each and `checkTile` passes it on its weight's noise, or, under `over_cap` `uncredited`, finds it only over its
inert-work bound. -/
def TilesAccepted (passes : List Pass) (m : OverInertBound) (opener : Opener) (drawn : List (ℕ × ℕ)) : Prop :=
  ∀ (pi tid : ℕ), (pi, tid) ∈ todo passes drawn → ∃ (p : Pass) (o : TileOpening),
    passes[pi]? = some p ∧ opener pi tid = .ok o ∧
    (checkTile p (noiseOf p tid) tid o = .pass ∨
      m = .uncredited ∧ ∃ why, checkTile p (noiseOf p tid) tid o = .overInertBound why)

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

/-- **A run and the registration record it is tied to**, all of it fixed before the run's coin. The prover's openings
are not of it: they are a function of the coin. -/
structure Run where
  coins : EpochCoins
  manifests : List Lean.Json
  passes : List Pass
  w : Lean.Json
  rounds : Option (List ℕ)
  dropped : List ℕ
  overInertBound : OverInertBound
  table : Table

/-- **The run is held to W**: its passes are `load`'s of its manifests under the epoch's coins, and `tieOf` answers
for them with the weights tied (a version-2 W). -/
def Run.HeldToW (R : Run) : Prop :=
  R.manifests.mapM (load R.coins) = .ok R.passes ∧
    ∃ tied : Tied, tieOf R.w (R.manifests.zip R.passes) R.rounds R.dropped = .ok tied ∧ tied.weights = none

def Run.Accepts (R : Run) (o : Opener) (drawn : List (ℕ × ℕ)) : Prop :=
  R.HeldToW ∧ TilesAccepted R.passes R.overInertBound o drawn

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

/-- The first string of `xs` whose SHA-512 digest another, different string of `xs` has, with that string: each string
hashed once, as the challenger's finder finds one. -/
def collisionIn (xs : List ByteArray) : Option (ByteArray × ByteArray) :=
  let hs := xs.map fun x => (x, Flock.Sha512.hash x)
  hs.findSome? fun (x, hx) => (hs.find? fun (y, hy) => hy == hx && !(y == x)).map fun (y, _) => (x, y)

/-- SHA-512, the compiled one, as the message-to-message hash `Claims.Finder.Pr` and `ecr/sha-512`
(`Claims.ExpectedCR`) read. -/
def sha512Msg (m : _root_.Claims.Msg) : _root_.Claims.Msg := (Flock.Sha512.hash ⟨m.toArray⟩).data.toList

def msgPair (xy : ByteArray × ByteArray) : _root_.Claims.Msg × _root_.Claims.Msg := (xy.1.data.toList, xy.2.data.toList)

def runOf (i : Input) (passes : List Pass) (w : Lean.Json) (T : Table) : Run :=
  { coins := i.coins, manifests := i.manifests, passes := passes, w := w, rounds := i.rounds, dropped := i.dropped,
    overInertBound := i.overInertBound, table := T }

/-- **An accepted verdict on an input that names W is a run held to W**: the plan's passes are W's by `tieOf`, the
verdict answers with that tie, and, where the tie's weights are tied, the run (`runOf`, at any table) is accepted on
the plan's sample with the openings the verdict took (`Run.Accepts`). -/
def AcceptedIsHeldToW : Prop :=
  ∀ (i : Input) (opener : Opener) (w : Lean.Json), (check i opener).accept = true →
    i.tie = some w → ∃ (pl : Plan) (tied : Tied), plan i = .ok pl ∧
      tieOf w (i.manifests.zip pl.passes) i.rounds i.dropped = .ok tied ∧ (check i opener).tie = some tied ∧
      (tied.weights = none → ∀ T : Table, (runOf i pl.passes w T).Accepts opener pl.picks)

/-- `t` tickets of the tiles `tk`: one sample of the run. -/
abbrev TicketDraws (tk : List ((ℕ × ℕ) × ℕ)) (t : ℕ) : Type := Fin t → _root_.Pouw.Ticket (weightsOf tk) tk.length

def drawnAt (tk : List ((ℕ × ℕ) × ℕ)) {t : ℕ} (c : TicketDraws tk t) : List (ℕ × ℕ) :=
  List.ofFn fun j => tileOf tk (c j)

open Classical in
/-- **The run's wrong ticket weight**: the tickets of its tiles that its table does not tie at W's entries. -/
noncomputable def Run.incorrectWeight (R : Run) : ℕ :=
  ∑ u : Fin (tickets R.passes).length,
    if TileTiedAtW R.w R.rounds R.passes R.table ((tickets R.passes).get u).1 then 0
    else weightsOf (tickets R.passes) u

/-- **The run's wrong share**: its wrong ticket weight over its tickets' total (0 where the run holds no ticket). -/
noncomputable def Run.incorrectShare (R : Run) : ℝ :=
  (R.incorrectWeight : ℝ) / (ticketTotal (tickets R.passes) : ℝ)

/-- **The run's finder**: on the tickets `c`, the first SHA-512 collision (`collisionIn`) among the messages
`Run.hashed` gives; its cost is declared at the budget `q` for the play's whole SHA-512 work, the prover's own
included. -/
noncomputable def Run.finder (R : Run) (t : ℕ) [Nonempty (TicketDraws (tickets R.passes) t)]
    (O : TicketDraws (tickets R.passes) t → Opener) (q : ℕ) : _root_.Claims.Finder where
  S := TicketDraws (tickets R.passes) t
  out c := (collisionIn (R.hashed (O c) (drawnAt (tickets R.passes) c))).map msgPair
  cost _ := q

/-- **PoUW's guarantee at W's entries, for an opener that sees the coin**: where (1 − ε)^t ≤ δ and the run's finder
meets `ecr/sha-512`, over `t` uniform work-weighted tickets the run is accepted with openings its table holds
while its untied tiles hold over ε of its ticket weight with probability at most δ + q / 2^256. -/
def YAtWIsTheCall : Prop :=
  ∀ (R : Run) (t : ℕ) [Nonempty (TicketDraws (tickets R.passes) t)]
    (O : TicketDraws (tickets R.passes) t → Opener) (q : ℕ) (ε δ : ℝ),
    0 ≤ ε → ε ≤ 1 → (1 - ε) ^ t ≤ δ →
    _root_.Claims.ExpectedCR sha512Msg (R.finder t O q) →
    _root_.Pouw.pr (fun c : TicketDraws (tickets R.passes) t =>
      R.Accepts (O c) (drawnAt (tickets R.passes) c) ∧ R.Holds (O c) (drawnAt (tickets R.passes) c) ∧
        ε < R.incorrectShare) ≤ δ + (q : ℝ) / 2 ^ (256 : ℝ)

/-- **The runs of a monitoring period's execution phases on its coins**, one coin a phase: `seq k` is phase `k`'s
sequence; a node's runs of record in time order are one, and a phase that overlaps another of its node is one of
its own. -/
structure Runs (Ω : Type) (n : ℕ) where
  seq : Fin n → ℕ
  run : (Fin n → Ω) → Fin n → Run
  drawn : (Fin n → Ω) → Fin n → Option (List (ℕ × ℕ))
  opener : (Fin n → Ω) → Fin n → Opener

/-- **Each phase's run is committed before its coin**: phase `k`'s run reads every coin but its own and its
sequence's later ones. The opener is not held to it: it may read every coin. -/
def Runs.CommittedBeforeCoin {Ω : Type} {n : ℕ} (P : Runs Ω n) : Prop :=
  (∀ (x x' : Fin n → Ω) (k : Fin n), (∀ v : Fin n, (P.seq v = P.seq k → v < k) → x v = x' v) →
    P.run x k = P.run x' k) ∧
  ∀ (x x' : Fin n → Ω) (k : Fin n), (∀ v : Fin n, (P.seq v = P.seq k → v ≤ k) → x v = x' v) →
    P.drawn x k = P.drawn x' k

/-- **Each phase's sample is `t` tickets on its coin**: over the phase's own coin, the others held, the tiles it
samples fall in any event no more often than `t` uniform tickets' do; a coin on which the sample refuses falls in
none. -/
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
  ∏ w : Fin n, (if P.seq w = P.seq k ∧ w ≤ k then 1 - (P.run x w).incorrectShare else 1)

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
  out x := (collisionIn (P.hashed x)).map msgPair
  cost _ := q

/-- **PoUW's period bound for the tile sample, one sequence**: some prefix of a sequence is accepted while its ∏ (1 −
wrong share) is below 1 − ε with probability at most δ + q / 2^256, whatever `n` is. A phase accepted after a
refused one of its sequence is not counted. -/
def PeriodYAtWIsTheCall : Prop :=
  ∀ {Ω : Type} [Fintype Ω] [Nonempty Ω] {n : ℕ} (P : Runs Ω n) (t q : ℕ) (ε δ : ℝ),
    0 ≤ ε → ε ≤ 1 → (1 - ε) ^ t ≤ δ →
    P.CommittedBeforeCoin → P.DrawIsTickets t →
    _root_.Claims.ExpectedCR sha512Msg (P.finder q) →
    ∀ a : ℕ, _root_.Pouw.pr (fun x : Fin n → Ω => ∃ k : Fin n, P.seq k = a ∧ P.AcceptedUpTo x k ∧
      P.tiedUpTo x k < 1 - ε) ≤ δ + (q : ℝ) / 2 ^ (256 : ℝ)

/-- **PoUW's period bound for the tile sample, pooled convexly across sequences**, at prefix ends fixed before the
coins: some nonempty `K ⊆ ends` is accepted while (∏ over `K` of ∏ (1 − wrong share))^t is below (|K| δ /
|ends|)^|K| with probability at most δ + q / 2^256. A sequence refused before its end is not counted. -/
def FleetYAtWIsTheCall : Prop :=
  ∀ {Ω : Type} [Fintype Ω] [Nonempty Ω] {n : ℕ} (P : Runs Ω n) (ends : Finset (Fin n)) (t q : ℕ) (δ : ℝ), 0 < δ →
    P.CommittedBeforeCoin → P.DrawIsTickets t →
    _root_.Claims.ExpectedCR sha512Msg (P.finder q) →
    _root_.Pouw.pr (fun x : Fin n → Ω => ∃ K : Finset (Fin n), K ⊆ ends ∧ K.Nonempty ∧
      (∀ k ∈ K, P.AcceptedUpTo x k) ∧
      (∏ k ∈ K, P.tiedUpTo x k) ^ t < ((K.card : ℝ) * δ / (ends.card : ℝ)) ^ K.card) ≤ δ + (q : ℝ) / 2 ^ (256 : ℝ)

/-- **A cell**: one y word of one call, (pass, the call's position in its pass, row, column). -/
abbrev Cell : Type := ℕ × ℕ × ℕ × ℕ

/-- **The run's real cells**: each row below its call's `realRows` at each column below its width `n`, of every call of
every retained pass. A filler row past `realRows` has no y row and is no cell; a dropped round's call is no pass's. -/
def realCells (passes : List Pass) : Finset Cell :=
  (passes.zipIdx.flatMap fun (p, pi) => p.records.zipIdx.flatMap fun (r, pos) =>
    (List.range r.realRows).flatMap fun i => (List.range r.matmul.n).map fun j => (pi, pos, i, j)).toFinset

/-- **The run's credited cells**: the credited cells (`UnitRecord.credited`) of each tile of each call, the cells its
tickets weigh. -/
def creditedCells (passes : List Pass) : Finset Cell :=
  (passes.zipIdx.flatMap fun (p, pi) => p.records.zipIdx.flatMap fun (r, pos) =>
    (p.scheme.tiles r.matmul.m r.matmul.n).flatMap fun (rows, cols) =>
      (r.credited rows cols).1.flatMap fun i => (r.credited rows cols).2.map fun j => (pi, pos, i, j)).toFinset

/-- **The give-up, as cells**: the real cells the records do not credit (excluded A rows, the rows of a band that holds
a voluntary A row, the columns of a block that holds an excluded or voluntary B row). The records fix it before the
coin, and no statement here ties it. -/
def givenUpCells (passes : List Pass) : Finset Cell := realCells passes \ creditedCells passes

open Classical in
/-- **The run's untied cells**: its real cells whose y its table does not tie at W's entries (`CellAtW`), the pass
registered as round `roundOf rounds passes.length pass`. -/
noncomputable def Run.untiedCells (R : Run) : Finset Cell :=
  (realCells R.passes).filter fun c =>
    match R.passes[c.1]? with
    | some p => ¬ CellAtW R.w R.table p.salt (roundOf R.rounds R.passes.length c.1) c.2.1 p.records[c.2.1]!
        c.2.2.1 c.2.2.2
    | none => True

/-- **The run's untied credited cells**: its untied cells outside the give-up. -/
noncomputable def Run.untiedCredited (R : Run) : Finset Cell := R.untiedCells \ givenUpCells R.passes

/-- **A credited cell's ticket weight** in the record `r`: `wref(1, |K|, n) / n`, `ticketWeight`'s over its cells. It
is positive at every width `n > 0`. -/
def cellTicket (r : UnitRecord) : ℚ := (wref 1 r.kKept r.matmul.n : ℚ) / r.matmul.n

/-- **The run's units, counting `per r` a credited cell of the record `r`**: each record that credits a cell, as (its
count a unit of ticket weight, `per r / cellTicket r`, and its ticket weight), the most a unit of weight first. At
`per r = r.kKept` they are `creditAtAssumption`'s units. -/
def unitsBy (per : UnitRecord → ℚ) (passes : List Pass) : List (ℚ × ℚ) :=
  (passes.flatMap fun p => p.records.filterMap fun r =>
    let cells := ((p.scheme.tiles r.matmul.m r.matmul.n).map fun (rows, cols) =>
      (r.credited rows cols).1.length * (r.credited rows cols).2.length).sum
    if cells = 0 then none else some (per r / cellTicket r, (cells : ℚ) * cellTicket r)).mergeSort
    fun a b => decide (b.1 ≤ a.1)

/-- What a ticket weight `b` holds, spent on the units in their order, each up to its own weight. -/
noncomputable def fill : List (ℚ × ℚ) → ℝ → ℝ
  | [], _ => 0
  | (ρ, w) :: us, b => min b (w : ℝ) * (ρ : ℝ) + fill us (b - min b (w : ℝ))

/-- **The most that ε of the run's ticket weight can hold, counting `per r` a credited cell**: ε of its units' weight
filled from the units with the most a unit of weight. A tile's cells all have its record's ticket, so this bounds the
count over every set of tiles of share at most ε. -/
noncomputable def atBy (per : UnitRecord → ℚ) (passes : List Pass) (ε : ℝ) : ℝ :=
  fill (unitsBy per passes) (ε * (((unitsBy per passes).map (·.2)).sum : ℝ))

/-- **n(ε)**: the most credited cells that ε of the run's ticket weight can hold (`atBy` at one a cell). -/
noncomputable def cellsAt (passes : List Pass) (ε : ℝ) : ℝ := atBy (fun _ => 1) passes ε

/-- **The most credited MACs that ε of the run's ticket weight can hold** (`atBy` at |K| a cell): the MACs
`creditAtAssumption` takes off the run's credit at ε, its `wrong`. -/
noncomputable def macsAt (passes : List Pass) (ε : ℝ) : ℝ := atBy (fun r => (r.kKept : ℚ)) passes ε

/-- **The wrong share counts the untied cells**: a run whose wrong share is at most ε has at most n(ε) untied
credited cells. It needs no premise that the run is held to W. -/
def UntiedCellsAtMost : Prop :=
  ∀ (R : Run) (ε : ℝ), R.incorrectShare ≤ ε → (R.untiedCredited.card : ℝ) ≤ cellsAt R.passes ε

/-- **PoUW's guarantee at W's entries, as cells**: on `YAtWIsTheCall`'s premises, the run is accepted with openings
its table holds while more than n(ε) of its credited cells are untied, with probability at most δ + q / 2^256. -/
def UntiedCellsAtW : Prop :=
  ∀ (R : Run) (t : ℕ) [Nonempty (TicketDraws (tickets R.passes) t)]
    (O : TicketDraws (tickets R.passes) t → Opener) (q : ℕ) (ε δ : ℝ),
    0 ≤ ε → ε ≤ 1 → (1 - ε) ^ t ≤ δ →
    _root_.Claims.ExpectedCR sha512Msg (R.finder t O q) →
    _root_.Pouw.pr (fun c : TicketDraws (tickets R.passes) t =>
      R.Accepts (O c) (drawnAt (tickets R.passes) c) ∧ R.Holds (O c) (drawnAt (tickets R.passes) c) ∧
        cellsAt R.passes ε < (R.untiedCredited.card : ℝ)) ≤ δ + (q : ℝ) / 2 ^ (256 : ℝ)

/-- **The untied credited cells of a sequence's prefix**: over phase `k` and the earlier phases of its sequence, each
its own run's. -/
noncomputable def Runs.untiedUpTo {Ω : Type} {n : ℕ} (P : Runs Ω n) (x : Fin n → Ω) (k : Fin n) : ℕ :=
  ∑ w : Fin n, (if P.seq w = P.seq k ∧ w ≤ k then (P.run x w).untiedCredited.card else 0)

/-- **n(ε) over a sequence's prefix, through the product**: the most credited cells phase `k` and the earlier phases
of its sequence can hold at shares `s` in [0, 1] whose ∏ (1 − s) is at least 1 − ε, each phase's at its own n
(`cellsAt`) of its share. -/
noncomputable def Runs.cellsUpTo {Ω : Type} {n : ℕ} (P : Runs Ω n) (x : Fin n → Ω) (k : Fin n) (ε : ℝ) : ℝ :=
  ⨆ s : {s : Fin n → ℝ // (∀ w, 0 ≤ s w ∧ s w ≤ 1) ∧
      1 - ε ≤ ∏ w : Fin n, (if P.seq w = P.seq k ∧ w ≤ k then 1 - s w else 1)},
    ∑ w : Fin n, (if P.seq w = P.seq k ∧ w ≤ k then cellsAt (P.run x w).passes (s.1 w) else 0)

/-- **PoUW's period bound as cells, one sequence**: on `PeriodYAtWIsTheCall`'s premises, some prefix of a sequence is
accepted while its phases hold more untied credited cells than `Runs.cellsUpTo` at ε, with probability at most δ
+ q / 2^256, whatever `n` is. -/
def PeriodUntiedCellsAtW : Prop :=
  ∀ {Ω : Type} [Fintype Ω] [Nonempty Ω] {n : ℕ} (P : Runs Ω n) (t q : ℕ) (ε δ : ℝ),
    0 ≤ ε → ε ≤ 1 → (1 - ε) ^ t ≤ δ →
    P.CommittedBeforeCoin → P.DrawIsTickets t →
    _root_.Claims.ExpectedCR sha512Msg (P.finder q) →
    ∀ a : ℕ, _root_.Pouw.pr (fun x : Fin n → Ω => ∃ k : Fin n, P.seq k = a ∧ P.AcceptedUpTo x k ∧
      P.cellsUpTo x k ε < (P.untiedUpTo x k : ℝ)) ≤ δ + (q : ℝ) / 2 ^ (256 : ℝ)

end Properties.Pouw.PearlPlus
