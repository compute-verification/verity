module

public import Proofs.Flock.Claims.Sampling.Reg
public import Proofs.Flock.Claims.Sampling.SessionSlots
public import Proofs.Flock.Claims.JsonEq
public import Proofs.Flock.Claims.ZkDecodeFlock
public import Definitions.Claims.SamplingTwo
public import Proofs.Flock.SpotFlock.ScopeFast
public import Properties.Spot.Members
public import Proofs.Flock.SpotFlock.Reveals
public import Flock.Desc
public import Flock.Comp
public import Spot.SecondStage

/-!
# SPOT[Flock]'s two-stage verifier

`SpotFlock.code h`: the sampled verifier's code (`SamplingDeploy.Code`) for a two-stage registration whose copies the
auditor holds (`h`, a `Held`), its sessions run by Flock's masked verifier (`FlockDeploy.V true`); and what the
two-stage compiler reads beside it: the design (`two h`), the replay sample of record (`rs h`) and the interior
registration's roots (`iroots`). This is the deployment layer: the compiler's theorem (`ksTwo_atOrder`) holds at
these definitions from the premises it names, and nothing here imports a proof. A running verifier is this one only
where it makes every check below.

**The auditor's copies** (`Held`), fixed before the run and never the prover's: the descriptor's bytes; each template's
units (`vus`), over which the members (`Held.ms`) put template `t`'s reference statement's circuit, whose lowerings side
by side are the statement's program (`bitsProgs`) and whose units are its proof blocks (`membersPart`); the statement's
partition query; the two-stage registration the samples read
(`RegTwo`), whose proof block `blockOf ρ t p` the verifier checks is request `ρ`'s unit `p` of template `t`
(`filesBlockOf`); the second
stage's count per stratum; the templates' names and the design's links, as round 2's statement check reads them
(`names`, `links`); template `t`'s reference session statement (`xS₀ t`), in Flock's masked scope, which template
`t`'s session statement is but for what the run sets (`pinOk`); template `t`'s reference registered reads (`refReads`),
which its statement's are but for the run's fields; and the commitment's layout and the reference prover's framing
(`frame`, a code whose verifier, samples and reveal count are unread).

**The run's statement** (`Roots`): the boundary registration's record, as JSON (`decode`), whose ports' roots are the
roots the boundary values are committed under (`portRoots`).

**The run**, one session per template, template `t`'s on `k₀ t` rounds (Flock's at `xS₀ t`) from round `off t`, and
`off T + 1` rounds in all at `T` templates:

* round 0: the verifier's coin is the first stage's bytes (the replay sample, `RegTwo.sample`);
* round 1: the prover's message is the interior registration's record, as JSON (`jsonMsg`): the roots of its ports,
  whose cells hold the sampled replay blocks' interiors, and those blocks in the order its cells lay them out
  (`window.replay_sample`, `blocksOf`); replay block `ρ`'s interior roots are its place in that order and the number
  of blocks, then those roots (`irootsOf`); the verifier's coin is the first of the second stage's;
* rounds 2 to `stage`, the `more` rounds a registration adds for a second stage that reads more than one coin's
  `2^20` bytes (`Held.more`, none at `more = 0`): the prover's message is empty, refused otherwise, and the verifier's
  coin is the second stage's next; the second stage's bytes are the `stage · 2^20` of the coins of rounds 1 to `stage`,
  coin by coin (`stageByte`, read by `Select.selectSecond`);
* round `stage + 1` (round 2 at one coin): the prover's message is the session statements, one per template in template
  order (`stmtsOf`), which the verifier checks (`stmtOk`): the run's statement decodes and its partition and design
  evaluate, and template `t`'s statement sets up as a masked Flock statement with `k₀ t` rounds, in Flock's masked
  scope, in the scope Flock's soundness proof covers and with hidden outputs, and with `xS₀ t`'s table shapes
  (`sessionOk`); round 1's record lists the replay sample in its order (`SecondStage.interiorOrderOk`) and registers
  no value of a name the statement's record has (`SecondStage.namesOk`); round 2's
  statement check passes the statement against template `t`'s part (`readsOk`, by `SecondStage.stmtCodes`: its public
  file's header is `xS₀ t`'s placed in the run (`SecondStage.atRun`) but for the run's fields, names the part's slots
  and the interior record's roots, its sample is the second stage's draw within the part, each registered read is one it
  may read, and its registered reads are its template's reference reads (`refReads`) but for the run's fields, the code
  itself the verifier of record of the session it checks (`codesOk`)); and it is `xS₀ t` but for its sample, its
  registered reads and its public file (`pinOk`); it refuses the message otherwise;
* rounds `off t` to `off t + k₀ t − 1`: template `t`'s session, Flock's masked verifier at its statement on the
  session's messages, its acceptance answered with the message `[]`, which Flock never answers, and the session's rounds
  after a decision (`decidedAt`) answered `[]` too;
* the last round accepts where each session's messages have a prefix Flock decides, and refuses otherwise. A session
  Flock refuses ends the run there, so a run accepts only where Flock accepted every session.

The design's replay partition is the members' (`filesReplay`: each request's units of every template together), and
it evaluates (`two`) only where the verifier decides the two-stage scope (`scopeOk`: the program ordered, the
proof-block partition a partition of it, each proof block in one replay block and the design well formed) and that the
replay layout fits the members' files (`filesOk`: the templates are the members, each population slot is its own unit,
each member has room for its units, and the registration's blocks are theirs), and that each template's reference
statement reads as a member's (`templateOk`: hidden outputs, untyped tags, a circuit file `Desc.parseStmt` reads, at
most `2^32` rows a unit, one table, every port a `v2Row`); every check is a decision, so the verifier refuses every
statement out of scope.

**What a session reveals** for a checked unit is the program's (`Reveal.reveal`): one opening for each value its proof
block's correctness reads (`Prog.needsList`), at that gate, under the interior roots of its replay block where the
value is one of that block's interior values and under the run's roots otherwise, holding the frame's opening there
(`Reveal.frameAt`); so at most `reveals` openings, the most one of the partition's proof blocks needs.

So every checked unit has its template's session (`templateOf`) on that session's rounds (`session`), and at a run
refused before them, the session is `xS₀ t`'s run of a prover whose first message Flock refuses (`refusedSeen`). The
coins are the verifier's own here; that the deployed challenger's bytes are such coins is the assumptions'
(`ChallengerLogsCoins`, `FlipsAfterRecorded`, `VerifierReadsLog`), and `Run.run`'s registration and outcome steps,
which read the challenger's logs and the sessions' verdicts, are not in this verifier: it accepts every run they accept
whose sessions Flock accepts.
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy Properties.Spot Finset
open Lean (Json)

/-! ## The auditor's copies and the run's statement -/

/-- **A session's statement is in Flock's masked scope**, decided (`FlockDeploy.ZkIn`): it sets up with `1` to `200`
tables at at most two link points, each at an `m` from `25` to `35` and of a shape in `Stmt.InRange`. -/
def zkOk (xS : Msg) : Bool :=
  match FlockDeploy.setupOf true xS with
  | some (_, spec, sts) => 0 < sts.size && sts.size ≤ 200 && spec.points ≤ 2 &&
      sts.toList.all fun st => 25 ≤ st.m && st.m ≤ 35 && st.c.kLog ≤ 27 && st.regions.size ≤ 1024 && spec.mPts ≤ 64
  | none => false

/-- **A two-stage registration's copies, as the auditor holds them**: the descriptor (`desc`), each template's units
(`vus`), the statement's partition query (`query`), the two-stage registration (`reg`), the second
stage's count per stratum (`k`), the templates' names (`names`) and the design's links (`links`), wires (`wires`) and
the rows their units read (`rows`) round 2's statement check reads, template `t`'s reference session statement
(`xS₀ t`) in Flock's masked scope (`xS₀_ok`), template `t`'s reference registered reads as round 2 reads a statement's,
null for none (`refReads t`), the commitment's layout and the reference prover's framing
(`frame`), and the rounds after round 1 whose coins the second stage reads too (`more`). Template `t`'s circuit is
its reference statement's (`Held.ms`). The reference reads are a copy of their own,
not `xS₀ t`'s: Flock's setup reads a statement's registered reads as a registration's, roots included, which reference
reads, staged before any run, can't name. -/
structure Held where
  desc : Msg
  vus : List (Array ℕ)
  query : Query
  reg : RegTwo
  k : List ℕ
  names : List String
  links : Option (Array Spot.SecondStage.Link)
  wires : Option (Array Spot.SecondStage.Wire)
  rows : Array Spot.SecondStage.Rows
  xS₀ : ℕ → Msg
  xS₀_ok : ∀ t, zkOk (xS₀ t) = true
  refReads : ℕ → Json
  frame : Code
  more : ℕ

/-- **The rounds whose coins the second stage reads**: rounds `1` to `stage`, one and `more` after it. -/
abbrev Held.stage (h : Held) : ℕ := h.more + 1

/-- **A session statement's circuit**: its reading's statement circuit (`Read.stmt`, a templated statement's
unrolled); `default` where it doesn't read. -/
def stmtOf (xS : Msg) : _root_.Flock.Circuit :=
  ((FlockDeploy.readOf true xS).map _root_.Flock.Readable.Read.stmt).getD default

/-- **The members**: template `t`'s circuit is its reference statement's (`stmtOf (xS₀ t)`), over its units
`vus[t]`. -/
def Held.ms (h : Held) : Spot.Bits.Members :=
  (List.range h.vus.length).map fun t => (stmtOf (h.xS₀ t), h.vus.getD t #[])

/-- **A message read as JSON**, as Flock's verifier reads a file (`FlockDeploy.jsonOf`); none where it isn't. -/
def jsonMsg (m : Msg) : Option Json := (FlockDeploy.jsonOf "message" ⟨m.toArray⟩).toOption

/-- **The bytes a JSON hex string names**; empty where it names none. -/
def hexMsg : Json → Msg
  | .str r => ((_root_.Flock.unhex r).toOption.map (·.data.toList)).getD []
  | _ => []

/-- **A number's bytes**, little-endian, as many as it needs and at least one. -/
def natMsg (n : ℕ) : Msg := (List.range (n.log2 / 8 + 1)).map fun i => (n / 256 ^ i).toUInt8

/-- **An entry of a record**, of kind `k` (`0` a port, `1` a registered value): four messages, its name behind its
kind, its root and its domain id (the bytes their hex names, empty where it names none), and its leaf count
(`natMsg`, `0` where it names none). These are what the deployed check reads of a registered port besides its
schema, which is the port's row format, so a frame fixed before the run reads the port off the roots. -/
def portEntry (e : UInt8 × String × Json) : List Msg :=
  [e.1 :: e.2.1.toUTF8.data.toList, hexMsg (Spot.getD e.2.2 "root"), hexMsg (Spot.getD e.2.2 "domain"),
    natMsg ((Spot.nat? (Spot.getD e.2.2 "leaves")).getD 0)]

/-- **A record's registered values**, each under its value's name, as round 2 finds a registered read's value among
them (`SecondStage.registeredEntry`). -/
def regList (rec : Json) : List (String × Json) :=
  (match Spot.getD rec "registered" with | .arr es => es.toList | _ => []).filterMap fun e =>
    match Spot.getD e "value" with
    | .str v => some (v, e)
    | _ => none

/-- **A record's entries, each with its kind**: its ports (`0`), in its port order (`Record.byPort`), then its
registered values (`1`). -/
def tagged (rec : Json) : List (UInt8 × String × Json) :=
  (Spot.Record.byPort rec).toList.map (0, ·) ++ (regList rec).map (1, ·)

/-- **A record's roots**: whether it names its ports by template (`SecondStage.byTemplate`, as round 2 names a
template's port), then its entries (`portEntry`). -/
def portRoots (rec : Json) : List Msg :=
  [if Spot.SecondStage.byTemplate rec then 1 else 0] :: (tagged rec).flatMap portEntry

/-- **The run's statement**: the boundary registration's record, and the roots the boundary values are committed
under, the record's (`portRoots`). -/
structure Roots where
  record : Json
  roots : List Msg

/-- **The run's statement's decoder**: the boundary registration's record, `x` read as JSON. -/
def decode (x : Msg) : Option Roots := (jsonMsg x).map fun rec => ⟨rec, portRoots rec⟩

/-- **The replay blocks the interior record lists**, in its order (`window.replay_sample`, which round 2's
`second-interior-order` holds to the replay sample); none where it lists none. -/
def blocksOf (rec : Json) : List ℕ :=
  match Spot.getD (Spot.getD rec "window") "replay_sample" with
  | .arr bs => bs.toList.map fun b => (Spot.nat? b).getD 0
  | _ => []

/-- **Replay block `ρ`'s interior roots** as the round-1 message `m₁` states them: `ρ`'s place among the interior
record's replay blocks and their number (`natMsg`), then the record's roots (`portRoots`); none where `m₁` isn't
JSON. -/
def irootsOf (m₁ : Msg) (ρ : ℕ) : List Msg :=
  ((jsonMsg m₁).map fun rec => natMsg ((blocksOf rec).idxOf ρ) :: natMsg (blocksOf rec).length :: portRoots rec).getD []

/-! ## The scope the verifier decides -/

/-- **The program is topologically ordered**, decided (`Prog.Topological`). -/
def topoOk (C : Prog) : Bool :=
  (List.range C.gates.length).all (fun g => ((C.gates[g]?.map BGate.reads).getD []).all (· < g)) &&
    C.outs.all (· < C.gates.length)

/-- Gate `g` of `C` is a constant. -/
def constAt (C : Prog) (g : ℕ) : Bool :=
  match C.gates[g]? with
  | some (.const _) => true
  | _ => false

/-- **`P` partitions `C`**, decided (`Partition.Partitions`). -/
def partitionsOk (P : Partition) (C : Prog) : Bool :=
  decide P.gates.Nodup && P.gates.all (· < C.gates.length) &&
    (List.range C.gates.length).all (fun g => constAt C g || decide (C.Unread g) || P.gates.contains g) &&
    P.blocks.all (fun u => !u.isEmpty) && P.blocks.all fun u => (unitClass C u).any fun w => decide (committed C u ≤ w)

/-- **A one-stage design is well formed** over strata of sizes `ns`, decided (`OneStage.WellFormed`). -/
def oneStageOk (ns : List ℕ) : OneStage → Bool
  | .all => true
  | .subset K => 1 ≤ K && K ≤ ns.sum
  | .bernoulli num den => 0 < den && num ≤ den
  | .stratified K => 1 ≤ K && K ≤ ns.sum && ns.all (1 ≤ ·)
  | .work _ w f => w.length == ns.length && f.length == ns.length && f.all (1 ≤ ·) &&
      0 < ((ns.zip w).map fun p => p.1 * p.2).sum && ns.all (1 ≤ ·)

/-- **The two-stage scope**, decided (`InScopeTwo` at a statement of program `C`). -/
def scopeOk (C : Prog) (Pt : Partition) (d : Design.Two) : Bool :=
  topoOk C && partitionsOk Pt C && decide d.part.gates.Nodup &&
    (List.range Pt.blocks.length).all (fun v => (replayOf Pt d.part v).isSome) &&
    d.replay.rules.length == d.part.classes.length &&
    (d.replay.rules.zip d.part.classes).all (fun p => oneStageOk (strataSizes p.2) p.1) &&
    d.k.length == Pt.strata.length && d.k.all (1 ≤ ·)

/-! ## The scope, compiled in passes over the program

`topoOk`, `partitionsOk` and `scopeOk` compile as `topoFast`, `partitionsFast` and `scopeOkFast`
(`SpotFlock.ScopeFast`), each proved equal to the check as stated (`@[csimp]`), so the verifier decides the scope in
passes over the program. -/

theorem constAt_eq_arr (C : Prog) (g : ℕ) : constAt C g = constArr C.gates.toArray g := by
  simp only [constArr, List.getElem?_toArray]
  rfl

@[csimp] theorem topoOk_eq_fast : @topoOk = @topoFast := by
  funext C
  exact (topoFast_eq C).symm

@[csimp] theorem partitionsOk_eq_fast : @partitionsOk = @partitionsFast := by
  funext P C
  rw [partitionsFast_eq, partitionsOk]
  simp only [constAt_eq_arr]

/-- `scopeOk` with the replay partition's `Nodup` and the replay check read from marks (`nodupFast`, `replaysFast`),
equal to it (`scopeOk_eq_fast`) and compiled in its place, as `topoOk` and `partitionsOk` compile as `topoFast` and
`partitionsFast`; as stated, `Nodup` compares every pair and the replay check finds each proof block's replay block by
position. -/
def scopeOkFast (C : Prog) (Pt : Partition) (d : Design.Two) : Bool :=
  topoOk C && partitionsOk Pt C && nodupFast d.part.gates && replaysFast Pt d.part C.gates.length &&
    d.replay.rules.length == d.part.classes.length &&
    (d.replay.rules.zip d.part.classes).all (fun p => oneStageOk (strataSizes p.2) p.1) &&
    d.k.length == Pt.strata.length && d.k.all (1 ≤ ·)

@[csimp] theorem scopeOk_eq_fast : @scopeOk = @scopeOkFast := by
  funext C Pt d
  unfold scopeOk scopeOkFast
  rw [nodupFast_eq]
  cases hp : partitionsOk Pt C
  · simp only [Bool.and_false, Bool.false_and]
  · cases hn : decide d.part.gates.Nodup
    · simp only [Bool.and_false, Bool.false_and]
    · have hlt : ∀ g ∈ Pt.gates, g < C.gates.length := by
        unfold partitionsOk at hp
        simp only [Bool.and_eq_true, List.all_eq_true, decide_eq_true_eq] at hp
        exact hp.1.1.1.2
      rw [replaysFast_eq (of_decide_eq_true hn) hlt]

/-- **The replay layout fits the members' files**, decided: the registration's templates are the members, each with
the design's count `k`; each population slot `ρ` of `l` is unit `ρ`, each member `t` has room for its `n_t` units,
`ρ·n_t` to `(ρ + 1)·n_t`, and the registration's block `blockOf ρ t p` is the members' (`filesBlockOf`). One check
per slot and proof block, none per gate. -/
def filesOk (g : RegTwo) (ms : Spot.Bits.Members) (l : Spot.Slots.Layout) (k : List ℕ) : Bool :=
  g.ns.size == ms.length && (List.range ms.length).all (fun t => k.getD t 0 == g.ks.getD t 0) &&
    (List.range l.blocks.size).all fun ρ =>
      match l.blocks[ρ]? with
      | some b =>
        match b.unit with
        | some u => u == ρ && (List.range ms.length).all fun t =>
            decide ((ρ + 1) * g.ns.getD t 0 ≤ ((ms[t]?).map (·.2.size)).getD 0) &&
              (List.range (g.ns.getD t 0)).all fun p => g.blockOf ρ t p == filesBlockOf ms g.ns ρ t p
        | none => true
      | none => true

/-- The tables' sizes `m` a session's statement sets up under `--zk`, in order; none where it doesn't set up. -/
def shapeOf (xS : Msg) : List ℕ :=
  match FlockDeploy.setupOf true xS with
  | some (_, _, sts) => sts.toList.map (·.m)
  | none => []

/-! ## What the verifier evaluates -/

theorem getD_map_range {α : Type} (f : ℕ → α) {i n : ℕ} (hi : i < n) (d : α) :
    ((List.range n).map f).getD i d = f i := by
  rw [List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range hi]
  rfl

theorem getD_concat {α : Type} (l : List α) (a d : α) : (l ++ [a]).getD l.length d = a := by
  have e := List.getD_append_right l [a] d l.length (Nat.le_refl _)
  rw [Nat.sub_self] at e
  exact e

variable (h : Held)

/-- **The hidden-outputs tag of a session statement's reading**, `false` where it has none. -/
def hiddenOf (xS : Msg) : Bool :=
  match FlockDeploy.readOf true xS with
  | some r => r.tags.hiddenOutputs
  | none => false

/-- **Template `t`'s session statement is in scope**: in Flock's masked scope, with template `t`'s reference statement's
table shapes, in the scope Flock's soundness proof covers (`FlockDeploy.scopeCheck`, as `flock-verify` refuses any
other), and with hidden outputs (`hiddenOf`), as the deployed session circuit has. -/
def sessionOk (t : ℕ) (xS : Msg) : Bool :=
  zkOk xS && shapeOf xS == shapeOf (h.xS₀ t) && FlockDeploy.scopeCheck true xS && hiddenOf xS

/-- **Template `t`'s reference statement reads as a member's**, decided: it reads with hidden outputs and untyped tags,
its circuit file's descriptor reads (`Desc.parseStmt`), its statement circuit's unit has at most `2^32` rows, it sets
up in one table, and each of its ports is a `sha512/row/v2` row (`ProvedScope.v2Row`), one leaf a row. -/
def templateOk (t : ℕ) : Bool :=
  match FlockDeploy.readOf true (h.xS₀ t) with
  | some r =>
    r.tags.hiddenOutputs && !r.tags.typed &&
      (match _root_.Flock.Desc.parseStmt r.tags r.I.circuitFile r.I.tables with
        | .ok _ => true
        | .error _ => false) &&
      decide (r.stmt.unit.unitLog ≤ 32) && (shapeOf (h.xS₀ t)).length == 1 &&
      r.stmt.ports.all _root_.Flock.ProvedScope.v2Row
  | none => false

/-- **The first stage's design**: the one `Select.select` samples on the boundary registration (`slotDesign`). -/
def replayDesign : Option Design :=
  (slotDesign h.reg.first.rule h.reg.first.program h.reg.first.query h.reg.first.table).toOption

/-- **The statement**: the members' programs side by side (`bitsProgs`), the auditor's query and first-stage design,
at the run's roots. -/
def parse (x : Msg) : Option AuditStatement :=
  (decode x).map fun st =>
    { desc := h.desc, prog := bitsProgs h.ms, query := h.query, design := (replayDesign h).getD ⟨[]⟩,
      roots := st.roots }

/-- **The proof-block partition**: the members' units (`membersPart`), template `t`'s units stratum `t`. -/
def partition (_ : AuditStatement) : Option Partition :=
  some (membersPart h.ms)

/-- **The two-stage design**: the members' replay partition over the boundary registration's layout (`filesReplay`),
the first stage's design and the second stage's counts, where the two-stage scope holds (`scopeOk`), the layout
fits the members' files (`filesOk`) and every template's reference statement reads as a member's (`templateOk`). -/
def two (_ : AuditStatement) : Option Design.Two :=
  match (Spot.Slots.layout h.reg.first.program h.reg.first.query).toOption, replayDesign h with
  | some l, some d =>
    if scopeOk (bitsProgs h.ms) (membersPart h.ms) ⟨filesReplay h.ms h.reg.ns l, d, h.k⟩ &&
        filesOk h.reg h.ms l h.k && (List.range h.ms.length).all (templateOk h) then
      some ⟨filesReplay h.ms h.reg.ns l, d, h.k⟩
    else none
  | _, _ => none

/-! ## The run -/

/-- **Template `t`'s session's rounds**: Flock's masked verifier's at template `t`'s reference statement. -/
def k₀ (t : ℕ) : ℕ := (FlockDeploy.V true).k (h.xS₀ t)

/-- **Template `t`'s session's first round**: after round 0, the second stage's `stage` rounds, the statement
check's round and the sessions of the templates before `t`. -/
def off (t : ℕ) : ℕ := sessOff (h.stage + 2) (k₀ h) t

/-- A coin's bytes, as the challenger's are read: `2^20` of them. -/
def L : ℕ := 65536 * 16

/-- Byte `i` of a coin. -/
def coinByte (c : FlockDeploy.Coin) (i : Fin L) : Fin 256 := ⟨(c.bytes.getD i.val 0).toNat, UInt8.toNat_lt _⟩

/-- The bytes `Select` reads of a coin. -/
def src (c : FlockDeploy.Coin) : ByteArray := bytesOf (coinByte c)

/-- Byte `i` of `R` rounds' coins, coin by coin: byte `i % L` of coin `i / L`. -/
def stageByte {R : ℕ} (cs : Fin R → FlockDeploy.Coin) (i : Fin (R * L)) : Fin 256 :=
  coinByte (cs (finProdFinEquiv.symm i).1) (finProdFinEquiv.symm i).2

/-- The bytes `Select` reads of `R` rounds' coins: `R · L` of them. -/
def stageSrc {R : ℕ} (cs : Fin R → FlockDeploy.Coin) : ByteArray := bytesOf (stageByte cs)

/-- **The second stage's coins**: those of rounds `1` to `stage`, in the run's order. -/
abbrev Stage : Type := Fin h.stage → FlockDeploy.Coin

/-- **The second stage's sample** on round 0's coin and the second stage's coins; none where either stage is
refused. -/
def secondDraw (c₀ : FlockDeploy.Coin) (c₁ : Stage h) : Option Json :=
  match Spot.Select.selectSecond (h.reg.sample (src c₀)) h.reg.ns h.reg.ks (stageSrc c₁) with
  | .ok (draw, _) => some draw
  | .error _ => none

deriving instance DecidableEq for FlockDeploy.Raw

/-- **A public file's header**: its first line's JSON (`Public.load`); none where it has no line or doesn't parse. -/
def headerOf (file : ByteArray) : Option Json := do
  let nl ← _root_.Flock.findByte file 10
  (_root_.Flock.JsonParse.parse (← String.fromUTF8? (file.extract 0 nl))).toOption

/-- **Template `t`'s session statement is its reference one** but for what the run sets: decoded as `w`, it is
`xS₀ t`'s in every field but its sample, its registered reads and its public file, its circuit, tables and program
among them. Round 2's statement check holds its public file's header to `xS₀ t`'s but for the run's fields
(`SecondStage.headerOk` at `refsOf`). -/
def pinOk (t : ℕ) (w : FlockDeploy.Raw) : Bool :=
  match FlockDeploy.decode (h.xS₀ t) with
  | some w₀ =>
    decide ({ w with draw := none, registered := none, publicFile := .empty } =
      { w₀ with draw := none, registered := none, publicFile := .empty })
  | none => false

/-- **Template `t`'s part of the second stage's population** in a replay sample of `q` blocks: its `q · n_t` slots from
`q · Σ_{t' < t} n_{t'}`, as `Select.secondLaw` lays out the cells, template by template. -/
def partAt (q t : ℕ) : Spot.SecondStage.Part :=
  ⟨h.names.getD t "", q * (h.reg.ns.toList.take t).sum, h.reg.ns.getD t 0⟩

/-- **Template `t`'s statement as round 2's statement check reads it** (`SecondStage.Stmt`), from its session statement
decoded as `w`: the template's name, its public file's header, its sample and its registered reads (null where it has
none); none where one of them doesn't read as JSON or they make no statement. -/
def stmtJsonOf (t : ℕ) (w : FlockDeploy.Raw) : Option Spot.SecondStage.Stmt := do
  let header ← headerOf w.publicFile
  let draw ← (FlockDeploy.jsonOf "draw" (← w.draw)).toOption
  let registered ← match w.registered with
    | some b => (FlockDeploy.jsonOf "registered reads" b).toOption
    | none => some .null
  Spot.SecondStage.Stmt.ofJson? (Json.mkObj
    [("draw", draw), ("header", header), ("registered", registered), ("template", .str (h.names.getD t ""))])

/-- **The reference headers** round 2's statement check holds each template's statement to (`SecondStage.Own.refs`):
template `t`'s name with its reference statement's public file's header, for each template whose header reads. -/
def refsOf : Array (String × Json) :=
  ((List.range h.ms.length).filterMap fun t =>
    ((FlockDeploy.decode (h.xS₀ t)).bind fun w₀ => headerOf w₀.publicFile).map fun r => (h.names.getD t "", r)).toArray

/-- **The reference reads** round 2's values check holds each template's statement to (`SecondStage.Own.refReads`):
template `t`'s name with its reference reads (`Held.refReads`), for each template. -/
def refRegisteredOf : Array (String × Json) :=
  ((List.range h.ms.length).map fun t => (h.names.getD t "", h.refReads t)).toArray

/-- **The second stage's parts** in a replay sample of `q` blocks, one per template in template order (`partAt`). -/
def partsAt (q : ℕ) : Array Spot.SecondStage.Part := ((List.range h.ms.length).map (partAt h q)).toArray

/-- **What the verifier holds of the run** as round 2's statement check reads it (`SecondStage.Own`), at the run's
statement `st`, round 1's record `rec`, the second stage's draw and round 0's coin: the replay sample and the parts in
it (`partsAt`, where a wired read finds its writer's), the interior and boundary registrations' records, the second
draw's units, the design's links and wires and the rows their units read, and the templates' reference headers and
registered reads, which the registration fixes before the run. -/
def ownOf (st : Roots) (rec draw : Json) (c₀ : FlockDeploy.Coin) : Spot.SecondStage.Own :=
  { sample := h.reg.sample (src c₀), parts := some (partsAt h (h.reg.sample (src c₀)).size), interior := rec,
    boundary := st.record, drawn := (Spot.Draw.proved draw).map Int.toNat, links := h.links, wires := h.wires,
    rows := h.rows, refs := refsOf h, refReads := refRegisteredOf h }

/-- **Round 2's statement check passes template `t`'s statement `S`** against the template's part, with what the
verifier holds of the run (`own`): no failing check (`SecondStage.stmtCodes` at `partAt`, the header placed at the
template's position among the parts as `stmtFailures` places it (`posOf`), the interior record's port names as
`stmtFailures` reads them). The code checks each session itself, so it is the verifier of record of the one it reads
at `S`: the verdict round 2 holds `S` to is its own (`Stmt.checked`), and `second-verified` asks only that `S`'s
header and registered reads have digests. -/
def codesOk (own : Spot.SecondStage.Own) (t : ℕ) (S : Spot.SecondStage.Stmt) : Bool :=
  (Spot.SecondStage.stmtCodes { own with verified := S.checked.toArray } (Spot.SecondStage.byTemplate own.interior)
    (Spot.SecondStage.posOf (partsAt h own.sample.size) (h.names.getD t "")) (partAt h own.sample.size t) S).isEmpty

/-- **Template `t`'s session statement reads the run's**, at the run's statement `x`, the round-1 message `m₁`, round
0's coin and the second stage's coins: round 1's record lists the replay sample in its order
(`SecondStage.interiorOrderOk`, round 2's `second-interior-order`) and registers no value of a name the statement's
record has, a port or a registered value (`SecondStage.namesOk`, round 2's `second-interior-names`), so each name a
session reads is one record's; round 2's statement check passes the statement against template `t`'s part
(`codesOk`: its public file's header is `xS₀ t`'s placed in the run but for the run's
fields, names the part's slots and the interior record's roots, its sample is the second stage's draw within the part,
and each registered read is one it may read); and it is otherwise template `t`'s reference statement (`pinOk`). -/
def readsOk (t : ℕ) (x m₁ xS : Msg) (c₀ : FlockDeploy.Coin) (c₁ : Stage h) : Bool :=
  match decode x, jsonMsg m₁, secondDraw h c₀ c₁, FlockDeploy.decode xS with
  | some st, some rec, some draw, some w =>
    let own := ownOf h st rec draw c₀
    Spot.SecondStage.interiorOrderOk own && Spot.SecondStage.namesOk own &&
      ((stmtJsonOf h t w).map (codesOk h own t)).getD false && pinOk h t w
  | _, _, _, _ => false

/-- **The session statements a message states**, one per template in template order, each a byte string; kept only
where they encode back to the message. -/
def stmtsOf (m : Msg) : Option (List Msg) :=
  match (FlockDeploy.rdList FlockDeploy.rdBytes).run m with
  | some (bs, []) =>
    if FlockDeploy.encList (fun r => FlockDeploy.encBytes ⟨r.toArray⟩) (bs.map fun b => b.data.toList) = m then
      some (bs.map fun b => b.data.toList)
    else none
  | _ => none

/-- **Template `t`'s session statement** as the session statements' message `m₂` states it; `xS₀ t` where it
states none. -/
def stmtAt (m₂ : Msg) (t : ℕ) : Msg :=
  match stmtsOf m₂ with
  | some xs => xs.getD t (h.xS₀ t)
  | none => h.xS₀ t

/-- **The session statements are the run's**, at the round-1 message `m₁`, the message `m₂` that states them, round 0's
coin and the second stage's coins: the run's statement decodes and its partition and design evaluate; `m₂` states one
session statement per template; and template `t`'s sets up as a masked Flock statement with `k₀ t` rounds, in scope
(`sessionOk`), and passes round 2's statement check against round 1's record and the second stage's draw, and is
otherwise `xS₀ t` (`readsOk`). -/
def stmtOk (x m₁ m₂ : Msg) (c₀ : FlockDeploy.Coin) (c₁ : Stage h) : Bool :=
  match parse h x, stmtsOf m₂ with
  | some s, some xs =>
    (partition h s).isSome && (two h s).isSome && xs.length == h.ms.length &&
      (List.range h.ms.length).all fun t =>
        (FlockDeploy.V true).k (xs.getD t (h.xS₀ t)) == k₀ h t && sessionOk h t (xs.getD t (h.xS₀ t)) &&
          readsOk h t x m₁ (xs.getD t (h.xS₀ t)) c₀ c₁
  | _, _ => false

/-- **What the statement check establishes of template `t`'s session statement `xS`**: the run's statement decodes and
its partition and design evaluate, and `xS` has `k₀ t` rounds, is in scope (`sessionOk`) and reads the run's
(`readsOk`). -/
def stmtOkAt (x m₁ xS : Msg) (c₀ : FlockDeploy.Coin) (c₁ : Stage h) (t : ℕ) : Bool :=
  match parse h x with
  | some s =>
    (partition h s).isSome && (two h s).isSome && (FlockDeploy.V true).k xS == k₀ h t && sessionOk h t xS &&
      readsOk h t x m₁ xS c₀ c₁
  | none => false

/-- **Flock decides the first `j` of a session's messages**: its masked verifier's step there is not a coin. -/
def decidedAt (xS : Msg) (as : List Msg) (cs : List FlockDeploy.Coin) (j : ℕ) : Bool :=
  match FlockDeploy.act true xS (as.take j) (cs.take j) with
  | .coin _ => false
  | _ => true

/-- **The session the `n`-th round is in**: the template `t` whose session's rounds hold it; none outside the
sessions. -/
def slotOf (n : ℕ) : Option ℕ :=
  (List.range h.ms.length).find? fun t => decide (off h t < n ∧ n ≤ off h t + k₀ h t)

/-- **A round of template `t`'s session**, at its statement `xS` and the run's messages and coins: `[]` where Flock
decided a shorter prefix of the session's messages, else Flock's masked verifier on the session's messages, its
acceptance answered `[]`. -/
def sessStep (t : ℕ) (xS : Msg) (as : List Msg) (cs : List FlockDeploy.Coin) : OC Out :=
  if (List.range (as.length - off h t)).any fun j =>
      decide (0 < j) && decidedAt xS (as.drop (off h t)) (cs.drop (off h t)) j
  then .ret (.msg [])
  else (FlockDeploy.next true xS (as.drop (off h t)) (cs.drop (off h t))).bind fun o => .ret (markOut [] o)

/-- **The verdict**, at the session statements' message `m₂` and the run's messages and coins: every template's
session's messages have a prefix Flock decides. -/
def verdict (m₂ : Msg) (as : List Msg) (cs : List FlockDeploy.Coin) : Bool :=
  (List.range h.ms.length).all fun t =>
    (List.range (k₀ h t + 1)).any fun j =>
      decide (0 < j) && decidedAt (stmtAt h m₂ t) (as.drop (off h t)) (cs.drop (off h t)) j

/-- **The second stage's coins among a run's**: those of rounds `1` to `stage`. -/
def stageOf (cs : List FlockDeploy.Coin) : Stage h := fun i => cs.getD (i.val + 1) default

/-- **The verifier's step** at the prover's messages `as` and the coins `cs`, one per message: round 0 answers its
coin, the first stage's; rounds `1` to `stage` answer theirs, the second stage's, the prover's message at round 1 the
interior registration's record and at each later one of them empty, refused otherwise; round `stage + 1` answers
nothing when the session statements are the run's (`stmtOk`) and refuses otherwise; a session's rounds answer as
`sessStep` at its statement; and the round after the sessions accepts on the verdict. -/
def next (x : Msg) (as : List Msg) (cs : List FlockDeploy.Coin) : OC Out :=
  if as.length = 0 ∨ cs.length ≠ as.length then .ret .rej
  else if as.length ≤ h.stage + 1 then
    .ret (if as.length ≤ 2 ∨ as.getD (as.length - 1) [] = [] then .msg (cs.getD (as.length - 1) default).bytes
      else .rej)
  else if as.length = h.stage + 2 then
    .ret (if stmtOk h x (as.getD 1 []) (as.getD (h.stage + 1) []) (cs.getD 0 default) (stageOf h cs) then .msg []
      else .rej)
  else
    match slotOf h as.length with
    | some t => sessStep h t (stmtAt h (as.getD (h.stage + 1) []) t) as cs
    | none =>
      .ret (if as.length == off h h.ms.length + 1 && verdict h (as.getD (h.stage + 1) []) as cs then .acc else .rej)

/-- **The sampled verifier in public-coin form**: round 0, the second stage's rounds, the statement check's round, the
templates' sessions, and the verdict's round. -/
def V : Verifier FlockDeploy.Coin where
  k _ := off h h.ms.length + 1
  next := next h

/-- **The verifier's step in rounds `0` to `stage`**: the round's coin, a non-empty message past round 1 refused. -/
theorem next_coin (x : Msg) {as : List Msg} {cs : List FlockDeploy.Coin} (h0 : as ≠ []) (hcs : cs.length = as.length)
    (hle : as.length ≤ h.stage + 1) :
    next h x as cs = .ret (if as.length ≤ 2 ∨ as.getD (as.length - 1) [] = [] then
      .msg (cs.getD (as.length - 1) default).bytes else .rej) := by
  have hc : ¬(as.length = 0 ∨ cs.length ≠ as.length) := fun hc =>
    hc.elim (fun e => h0 (List.length_eq_zero_iff.1 e)) fun hne => hne hcs
  unfold next
  rw [ite_eq_right hc, ite_eq_left hle]

/-- **The verifier's step in rounds 0 and 1**: the round's coin. -/
theorem next_two (x : Msg) {as : List Msg} {cs : List FlockDeploy.Coin} (h0 : as ≠ []) (hcs : cs.length = as.length)
    (h2 : as.length ≤ 2) : next h x as cs = .ret (.msg (cs.getD (as.length - 1) default).bytes) := by
  have hpos : 0 < h.stage := Nat.succ_pos _
  have hle : as.length ≤ h.stage + 1 := by omega
  rw [next_coin h x h0 hcs hle, ite_eq_left (Or.inl h2)]

/-- **The verifier's step at round `stage + 1`**: the session statements' check. -/
theorem next_stmt (x : Msg) {as : List Msg} {cs : List FlockDeploy.Coin} (hcs : cs.length = as.length)
    (hlen : as.length = h.stage + 2) :
    next h x as cs = .ret (if stmtOk h x (as.getD 1 []) (as.getD (h.stage + 1) []) (cs.getD 0 default) (stageOf h cs)
      then .msg [] else .rej) := by
  have hc : ¬(as.length = 0 ∨ cs.length ≠ as.length) := fun hc =>
    hc.elim (fun e => by omega) fun hne => hne hcs
  have h1 : ¬as.length ≤ h.stage + 1 := by omega
  unfold next
  rw [ite_eq_right hc, ite_eq_right h1, ite_eq_left hlen]

/-- **The verifier's step after round `stage + 1`**: a session's round where one holds it, else the verdict's. -/
theorem next_long (x : Msg) (as : List Msg) (cs : List FlockDeploy.Coin) (has : h.stage + 2 < as.length)
    (hcs : cs.length = as.length) :
    next h x as cs = match slotOf h as.length with
      | some t => sessStep h t (stmtAt h (as.getD (h.stage + 1) []) t) as cs
      | none =>
        .ret (if as.length == off h h.ms.length + 1 && verdict h (as.getD (h.stage + 1) []) as cs then .acc
          else .rej) := by
  have hc : ¬(as.length = 0 ∨ cs.length ≠ as.length) := fun hc =>
    hc.elim (fun e => by omega) fun hne => hne hcs
  have h1 : ¬as.length ≤ h.stage + 1 := by omega
  have h2 : ¬as.length = h.stage + 2 := by omega
  unfold next
  rw [ite_eq_right hc, ite_eq_right h1, ite_eq_right h2]

theorem stage_le_off (t : ℕ) : h.stage + 2 ≤ off h t := le_sessOff (h.stage + 2) (k₀ h) t

theorem three_le_off (t : ℕ) : 3 ≤ off h t := (Nat.le_add_left 3 h.more).trans (stage_le_off h t)

/-- **The run's round order**: the first stage reads round 0's coin, the second stage's last round is `stage`. -/
def order (x : Msg) : Order ((V h).k x) :=
  ⟨⟨0, by have := three_le_off h h.ms.length; show 0 < off h h.ms.length + 1; omega⟩,
    ⟨h.stage, by have := stage_le_off h h.ms.length; show h.stage < off h h.ms.length + 1; omega⟩,
    Fin.mk_lt_mk.2 (Nat.succ_pos h.more)⟩

/-- **The run's stage order**: the second stage reads the coins of rounds `1` to `stage`. -/
def stageOrder (x : Msg) : StageOrder ((V h).k x) h.stage := ⟨order h x, (Nat.zero_add h.stage).le⟩

/-- **The second stage's coins of a run**: its rounds 1 to `stage`'s, in order. -/
def stageCoins (x : Msg) (r : Fin ((V h).k x) → FlockDeploy.Coin) : Stage h := fun i => r ((stageOrder h x).round i)

theorem round_val (x : Msg) (i : Fin h.stage) : ((stageOrder h x).round i).val = i.val + 1 := by
  have hi := i.isLt
  show h.stage - (h.stage - 1 - i.val) = i.val + 1
  omega

/-- The second stage's coin `i` is the run's round `i + 1`'s. -/
theorem coinSeq_stage (x : Msg) (r : Fin ((V h).k x) → FlockDeploy.Coin) (i : Fin h.stage) :
    coinSeq r (i.val + 1) = stageCoins h x r i := by
  have hlt : i.val + 1 < (V h).k x := by
    have hr := ((stageOrder h x).round i).isLt
    rw [round_val h x i] at hr
    exact hr
  unfold coinSeq
  rw [dite_eq_left hlt]
  exact congrArg r (Fin.ext (round_val h x i).symm)

theorem coinSeq_stageCoins (x : Msg) (r : Fin ((V h).k x) → FlockDeploy.Coin) {j : ℕ} (hj : j < h.stage) :
    coinSeq (stageCoins h x r) j = coinSeq r (j + 1) :=
  (dite_eq_left hj).trans (coinSeq_stage h x r ⟨j, hj⟩).symm

/-- **A run refused before its session, as a Flock run**: the reference statement's run of a prover whose first
message, empty, Flock refuses. -/
def refusedSeen : Seen := ⟨[([], .rej)], .rej⟩

/-- An answer is a message. -/
def answered : Out → Bool
  | .msg _ => true
  | _ => false

/-- **Proof block `u`'s template**: the first whose units end past `u`, the members' units being template-major
(`unitBase`); the number of templates past them all. -/
def templateOf (ms : Spot.Bits.Members) (u : ℕ) : ℕ :=
  ((List.range ms.length).find? fun t => decide (u < unitBase ms (t + 1))).getD ms.length

/-- **Unit `u`'s session at a run's transcript**: its template `t`'s statement as round `stage + 1` states it, where it
is in scope (`sessionOk`), and the transcript from round `off t` on cut at the session's decision (`cutSeen`), where the
verifier answered every round before `off t` with a message; `xS₀ t` and `refusedSeen` at a run refused before the
session. -/
def session (u : ℕ) (v : Seen) : Option (Msg × Seen) :=
  if off h (templateOf h.ms u) ≤ v.msgs.length ∧ (v.msgs.take (off h (templateOf h.ms u))).all (answered ·.2) then
    if sessionOk h (templateOf h.ms u) (stmtAt h (v.msgs.getD (h.stage + 1) ([], .rej)).1 (templateOf h.ms u)) then
      some (stmtAt h (v.msgs.getD (h.stage + 1) ([], .rej)).1 (templateOf h.ms u),
        cutSeen [] (k₀ h (templateOf h.ms u)) (v.msgs.drop (off h (templateOf h.ms u))))
    else some (h.xS₀ (templateOf h.ms u), refusedSeen)
  else some (h.xS₀ (templateOf h.ms u), refusedSeen)

/-- Template `t`'s reference statement's masked level-0 lengths, summed. -/
noncomputable def n₀ (t : ℕ) : ℝ := ((shapeOf (h.xS₀ t)).map fun m => (2 : ℝ) ^ (Claims.Flock.zkLevel0 m).d).sum

/-- **The interior registration's roots** of replay block `ρ`, as the round-1 message states them (`irootsOf`); none
before it. -/
def iroots (_ : Msg) (v : Seen) (ρ : ℕ) : List Msg :=
  match v.msgs with
  | (_, _) :: (m₁, _) :: _ => irootsOf m₁ ρ
  | _ => []

/-- **The code before its samples**: the verifier, the statement and its evaluation, each unit's session on its
template's session's rounds, one session per template, what a session reveals (one opening per value a checked unit's
proof block needs, under its roots, holding `frame`'s opening there) and the reveal count, its tables the most any
template's reference statement has (the template past the last, a unit past the members', included), each unit's `N₀`
its template's (`unitN₀`) and the statement's template 0's, and the commitment's layout and reference prover's framing
`frame`'s. The interior roots don't read the run's statement (`iroots`). -/
noncomputable def base : Code :=
  { h.frame with
    V := V h
    parse := parse h
    scopeCheck := fun x => ((parse h x).bind (partition h)).isSome
    partition := partition h
    stage2 := fun _ => univ
    checked := fun _ _ => []
    rounds := fun _ u => univ.filter fun t =>
      off h (templateOf h.ms u) ≤ t.val ∧ t.val < off h (templateOf h.ms u) + k₀ h (templateOf h.ms u)
    session := fun _ u v => session h u v
    reveal := Reveal.reveal (partition h) (two h) (iroots []) (Reveal.frameAt h.frame)
    reveals := Reveal.reveals (partition h) (parse h)
    sessions := fun _ => h.ms.length
    tables := fun _ => ((List.range (h.ms.length + 1)).map fun t => (shapeOf (h.xS₀ t)).length).foldr max 0
    unitN₀ := fun _ u => n₀ h (templateOf h.ms u)
    innerN₀ := fun _ => n₀ h 0 }

/-- The registration the samples read, at a statement that decodes. -/
def reg (x : Msg) : Option RegTwo := (decode x).map fun _ => h.reg

/-- **SPOT[Flock]'s two-stage code**: the code at its stage order, its checked units the two stages' on the
auditor's registration, the first stage's bytes round 0's coin's and the second stage's the `stage · L` bytes of its
rounds' coins. -/
noncomputable abbrev code : Code :=
  codeTwoStage (base h) (reg h) (fun _ => L) (fun _ => h.stage * L) (fun _ => coinByte) (fun _ => h.stage)
    (stageOrder h) fun _ => stageByte

/-- **The replay sample of record**: the first stage's checked slots on the replay layout. -/
noncomputable def rs : (x : Msg) → (Fin ((code h).V.k x) → FlockDeploy.Coin) → List ℕ :=
  replayTwo ((base h).atOrder (order h)) (reg h) (fun _ => L)
    (bytesFirst (base h) (fun _ => L) (order h) fun _ => coinByte)

end SpotFlock
