module

public import Proofs.Flock.Claims.Sampling.Reg
public import Proofs.Flock.Claims.Sampling.SessionSlots
public import Proofs.Flock.Claims.JsonEq
public import Proofs.Flock.Claims.ZkDecodeFlock
public import Definitions.Claims.SamplingTwo
public import Properties.Spot.Members

/-!
# SPOT[Flock]'s two-stage verifier

`SpotFlock.code h`: the sampled verifier's code (`SamplingDeploy.Code`) for a two-stage registration whose copies the
auditor holds (`h`, a `Held`), its sessions run by Flock's masked verifier (`FlockDeploy.V true`); and what the
two-stage compiler reads beside it: the design (`two h`), the replay sample of record (`rs h`) and the interior
registration's roots (`iroots`). This is the deployment layer: the compiler's theorem (`ksTwo_atOrder`) holds at
these definitions from the premises it names, and nothing here imports a proof. A running verifier is this one only
where it makes every check below.

**The auditor's copies** (`Held`), fixed before the run and never the prover's: the descriptor's bytes; the templates'
circuit files and units (`ms`), whose lowerings side by side are the statement's program (`bitsProgs`) and whose units
are its proof blocks (`membersPart`); the statement's partition query; the two-stage registration the samples read
(`RegTwo`), whose proof block `blockOf ρ t p` the verifier checks is request `ρ`'s unit `p` of template `t`
(`filesBlockOf`); the second
stage's count per stratum; the registered reads template `t`'s session must open against at a run's roots and interior
roots (`registered t`); template `t`'s reference session statement (`xS₀ t`), in Flock's masked scope, which template
`t`'s session statement is but for its sample and its registered reads; and the commitment's layout and the reference
prover's framing (`frame`, a code whose verifier and samples are unread).

**The run's statement** (`Roots`): the roots the boundary values are committed under, as canonical bytes
(`Roots.encode`, read back by `decode`, which keeps only bytes that encode back to themselves).

**The run**, one session per template, template `t`'s on `k₀ t` rounds (Flock's at `xS₀ t`) from round `off t`, and
`off T + 1` rounds in all at `T` templates:

* round 0: the verifier's coin is the first stage's bytes (the replay sample, `RegTwo.sample`);
* round 1: the prover's message is the interior registration, the roots of each sampled replay block's interior
  (`Interiors.encode`); the verifier's coin is the second stage's bytes (`Select.selectSecond`);
* round 2: the prover's message is the session statements, one per template in template order (`stmtsOf`), which the
  verifier checks (`stmtOk`): the run's statement decodes and its partition and design evaluate, and template `t`'s
  statement sets up as a masked Flock statement with `k₀ t` rounds, in Flock's masked scope, in the scope Flock's
  soundness proof covers and with hidden outputs, and with `xS₀ t`'s table shapes (`sessionOk`), its sample is the
  second stage's, its registered reads are the auditor's for template `t` at the run's roots and the interior
  registration's, and it is `xS₀ t` in every other field (`pinOk`); it refuses the message otherwise;
* rounds `off t` to `off t + k₀ t − 1`: template `t`'s session, Flock's masked verifier at its statement on the
  session's messages, its acceptance answered with the message `[]`, which Flock never answers, and the session's rounds
  after a decision (`decidedAt`) answered `[]` too;
* the last round accepts where each session's messages have a prefix Flock decides, and refuses otherwise. A session
  Flock refuses ends the run there, so a run accepts only where Flock accepted every session.

The design's replay partition is the members' (`filesReplay`: each request's units of every template together), and
it evaluates (`two`) only where the verifier decides the two-stage scope (`scopeOk`: the program ordered, the
proof-block partition a partition of it, each proof block in one replay block and the design well formed) and that the
replay layout fits the members' files (`filesOk`: the templates are the members, each population slot is its own unit,
each member has room for its units, and the registration's blocks are theirs); every check is a decision, so the
verifier refuses every statement out of scope.

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

/-- **A two-stage registration's copies, as the auditor holds them**: the descriptor (`desc`), the templates' circuit
files and units (`ms`), the statement's partition query (`query`), the two-stage registration (`reg`), the second
stage's count per stratum (`k`), the registered reads template `t`'s session
opens against at a run's roots and interior roots (`registered t`), template `t`'s reference session statement
(`xS₀ t`) in Flock's masked scope (`xS₀_ok`), and the commitment's layout and the reference prover's framing
(`frame`). -/
structure Held where
  desc : Msg
  ms : Spot.Bits.Members
  query : Query
  reg : RegTwo
  k : List ℕ
  registered : ℕ → List Msg → List (ℕ × List Msg) → ByteArray
  xS₀ : ℕ → Msg
  xS₀_ok : ∀ t, zkOk (xS₀ t) = true
  frame : Code

/-- **The run's statement**: the roots the boundary values are committed under. -/
structure Roots where
  roots : List Msg

/-- The run's statement's format string, its first field. -/
def FORMAT : String := "spot-flock-statement/v1"

/-- **The run's statement's encoding**: its format string, then its roots, each as a byte string. -/
def Roots.encode (st : Roots) : Msg :=
  FlockDeploy.encStr FORMAT ++ FlockDeploy.encList (fun r => FlockDeploy.encBytes ⟨r.toArray⟩) st.roots

/-- **The run's statement's reader**: the format string, failing at any other, then the roots, each a byte string. -/
def Roots.read : FlockDeploy.Rd Roots := do
  if (← FlockDeploy.rdStr) ≠ FORMAT then failure
  let roots ← FlockDeploy.rdList FlockDeploy.rdBytes
  pure ⟨roots.map fun b => b.data.toList⟩

/-- **The run's statement's decoder**: `Roots.read` on all of `x`, kept only when its encoding is `x`. -/
def decode (x : Msg) : Option Roots :=
  match Roots.read.run x with
  | some (st, []) => if st.encode = x then some st else none
  | _ => none

/-- **The interior registration**: each sampled replay block's interior roots, by replay block. -/
structure Interiors where
  roots : List (ℕ × List Msg)

/-- **The interior registration's encoding**: for each replay block, its number, then its interior roots, each as a
byte string. -/
def Interiors.encode (it : Interiors) : Msg :=
  FlockDeploy.encList (fun p => FlockDeploy.encNat p.1 ++
    FlockDeploy.encList (fun r => FlockDeploy.encBytes ⟨r.toArray⟩) p.2) it.roots

/-- **The interior registration's reader**: for each replay block, its number, then its interior roots, each a byte
string. -/
def Interiors.read : FlockDeploy.Rd Interiors := do
  let roots ← FlockDeploy.rdList do
    let ρ ← FlockDeploy.rdNat
    let rs ← FlockDeploy.rdList FlockDeploy.rdBytes
    pure (ρ, rs.map fun b => b.data.toList)
  pure ⟨roots⟩

/-- The interior registration a prover's message states, kept only when it encodes back to the message. -/
def interiorsOf (m : Msg) : Option Interiors :=
  match Interiors.read.run m with
  | some (it, []) => if it.encode = m then some it else none
  | _ => none

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
the first stage's design and the second stage's counts, where the two-stage scope holds (`scopeOk`) and the layout
fits the members' files (`filesOk`). -/
def two (_ : AuditStatement) : Option Design.Two :=
  match (Spot.Slots.layout h.reg.first.program h.reg.first.query).toOption, replayDesign h with
  | some l, some d =>
    if scopeOk (bitsProgs h.ms) (membersPart h.ms) ⟨filesReplay h.ms h.reg.ns l, d, h.k⟩ &&
        filesOk h.reg h.ms l h.k then
      some ⟨filesReplay h.ms h.reg.ns l, d, h.k⟩
    else none
  | _, _ => none

/-! ## The run -/

/-- **Template `t`'s session's rounds**: Flock's masked verifier's at template `t`'s reference statement. -/
def k₀ (t : ℕ) : ℕ := (FlockDeploy.V true).k (h.xS₀ t)

/-- **Template `t`'s session's first round**: after rounds 0 to 2 and the sessions of the templates before `t`. -/
def off (t : ℕ) : ℕ := sessOff 3 (k₀ h) t

/-- A coin's bytes, as the challenger's are read: `2^20` of them. -/
def L : ℕ := 65536 * 16

/-- Byte `i` of a coin. -/
def coinByte (c : FlockDeploy.Coin) (i : Fin L) : Fin 256 := ⟨(c.bytes.getD i.val 0).toNat, UInt8.toNat_lt _⟩

/-- The bytes `Select` reads of a coin. -/
def src (c : FlockDeploy.Coin) : ByteArray := bytesOf (coinByte c)

/-- **The second stage's sample** on the coins of rounds 0 and 1; none where either stage is refused. -/
def secondDraw (c₀ c₁ : FlockDeploy.Coin) : Option Json :=
  match Spot.Select.selectSecond (h.reg.sample (src c₀)) h.reg.ns h.reg.ks (src c₁) with
  | .ok (draw, _) => some draw
  | .error _ => none

deriving instance DecidableEq for FlockDeploy.Raw

/-- **Template `t`'s session statement is its reference one** but for its sample and its registered reads: decoded as
`w`, it is `xS₀ t`'s in every other field, its circuit, public file, tables and program among them. -/
def pinOk (t : ℕ) (w : FlockDeploy.Raw) : Bool :=
  match FlockDeploy.decode (h.xS₀ t) with
  | some w₀ => decide ({ w with draw := none, registered := none } = { w₀ with draw := none, registered := none })
  | none => false

/-- **Template `t`'s session statement reads the run's**, at the statement `s`, the round-1 message `m₁` and the coins
of rounds 0 and 1: its sample is the second stage's, its registered reads are the auditor's for template `t` at the
run's roots and the interior registration's, and it is otherwise template `t`'s reference statement (`pinOk`). -/
def readsOk (t : ℕ) (s : AuditStatement) (m₁ xS : Msg) (c₀ c₁ : FlockDeploy.Coin) : Bool :=
  match interiorsOf m₁, secondDraw h c₀ c₁, FlockDeploy.setupOf true xS, FlockDeploy.decode xS with
  | some it, some draw, some (ins, _, _), some w =>
    FlockDeploy.drawEq ins.read.draw (some (Flock.norm draw)) &&
      w.registered.map ByteArray.data == some (h.registered t s.roots it.roots).data && pinOk h t w
  | _, _, _, _ => false

/-- **The round-2 message's session statements**, one per template in template order, each a byte string; kept only
where they encode back to the message. -/
def stmtsOf (m : Msg) : Option (List Msg) :=
  match (FlockDeploy.rdList FlockDeploy.rdBytes).run m with
  | some (bs, []) =>
    if FlockDeploy.encList (fun r => FlockDeploy.encBytes ⟨r.toArray⟩) (bs.map fun b => b.data.toList) = m then
      some (bs.map fun b => b.data.toList)
    else none
  | _ => none

/-- **Template `t`'s session statement** as the round-2 message `m₂` states it; `xS₀ t` where it states none. -/
def stmtAt (m₂ : Msg) (t : ℕ) : Msg :=
  match stmtsOf m₂ with
  | some xs => xs.getD t (h.xS₀ t)
  | none => h.xS₀ t

/-- **The session statements are the run's**, at the round-1 message `m₁`, the round-2 message `m₂` and the coins of
rounds 0 and 1: the run's statement decodes and its partition and design evaluate; `m₂` states one session statement
per template; and template `t`'s sets up as a masked Flock statement with `k₀ t` rounds, in scope (`sessionOk`), its
sample the second stage's, its registered reads the auditor's for template `t` at the run's roots and the interior
registration's, and otherwise `xS₀ t` (`readsOk`). -/
def stmtOk (x m₁ m₂ : Msg) (c₀ c₁ : FlockDeploy.Coin) : Bool :=
  match parse h x, stmtsOf m₂ with
  | some s, some xs =>
    (partition h s).isSome && (two h s).isSome && xs.length == h.ms.length &&
      (List.range h.ms.length).all fun t =>
        (FlockDeploy.V true).k (xs.getD t (h.xS₀ t)) == k₀ h t && sessionOk h t (xs.getD t (h.xS₀ t)) &&
          readsOk h t s m₁ (xs.getD t (h.xS₀ t)) c₀ c₁
  | _, _ => false

/-- **What the round-2 check establishes of template `t`'s session statement `xS`**: the run's statement decodes and
its partition and design evaluate, and `xS` has `k₀ t` rounds, is in scope (`sessionOk`) and reads the run's
(`readsOk`). -/
def stmtOkAt (x m₁ xS : Msg) (c₀ c₁ : FlockDeploy.Coin) (t : ℕ) : Bool :=
  match parse h x with
  | some s =>
    (partition h s).isSome && (two h s).isSome && (FlockDeploy.V true).k xS == k₀ h t && sessionOk h t xS &&
      readsOk h t s m₁ xS c₀ c₁
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

/-- **The verdict**, at the round-2 message `m₂` and the run's messages and coins: every template's session's messages
have a prefix Flock decides. -/
def verdict (m₂ : Msg) (as : List Msg) (cs : List FlockDeploy.Coin) : Bool :=
  (List.range h.ms.length).all fun t =>
    (List.range (k₀ h t + 1)).any fun j =>
      decide (0 < j) && decidedAt (stmtAt h m₂ t) (as.drop (off h t)) (cs.drop (off h t)) j

/-- **The verifier's step** at the prover's messages `as` and the coins `cs`, one per message: rounds 0 and 1 answer
their coins; round 2 answers nothing when the session statements are the run's (`stmtOk`) and refuses otherwise; a
session's rounds answer as `sessStep` at its statement; and the round after the sessions accepts on the verdict. -/
def next (x : Msg) (as : List Msg) (cs : List FlockDeploy.Coin) : OC Out :=
  match as, cs with
  | [_], [c₀] => .ret (.msg c₀.bytes)
  | [_, _], [_, c₁] => .ret (.msg c₁.bytes)
  | [_, m₁, m₂], [c₀, c₁, _] => .ret (if stmtOk h x m₁ m₂ c₀ c₁ then .msg [] else .rej)
  | a₀ :: a₁ :: m₂ :: a₃ :: as', c₀ :: c₁ :: c₂ :: c₃ :: cs' =>
    match slotOf h (as'.length + 4) with
    | some t => sessStep h t (stmtAt h m₂ t) (a₀ :: a₁ :: m₂ :: a₃ :: as') (c₀ :: c₁ :: c₂ :: c₃ :: cs')
    | none =>
      .ret (if as'.length + 4 == off h h.ms.length + 1 &&
          verdict h m₂ (a₀ :: a₁ :: m₂ :: a₃ :: as') (c₀ :: c₁ :: c₂ :: c₃ :: cs') then .acc else .rej)
  | _, _ => .ret .rej

/-- **The sampled verifier in public-coin form**: rounds 0 to 2, the templates' sessions, and the verdict's round. -/
def V : Verifier FlockDeploy.Coin where
  k _ := off h h.ms.length + 1
  next := next h

theorem three_le_off (t : ℕ) : 3 ≤ off h t := le_sessOff 3 (k₀ h) t

/-- **The run's round order**: the first stage reads round 0's coin, the second round 1's. -/
def order (x : Msg) : Order ((V h).k x) :=
  ⟨⟨0, by have := three_le_off h h.ms.length; show 0 < off h h.ms.length + 1; omega⟩,
    ⟨1, by have := three_le_off h h.ms.length; show 1 < off h h.ms.length + 1; omega⟩, by simp [Fin.lt_def]⟩

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

/-- **Unit `u`'s session at a run's transcript**: its template `t`'s statement as round 2 states it, where it is in
scope (`sessionOk`), and the transcript from round `off t` on cut at the session's decision (`cutSeen`), where the
verifier answered every round before `off t` with a message; `xS₀ t` and `refusedSeen` at a run refused before the
session. -/
def session (u : ℕ) (v : Seen) : Option (Msg × Seen) :=
  if off h (templateOf h.ms u) ≤ v.msgs.length ∧ (v.msgs.take (off h (templateOf h.ms u))).all (answered ·.2) then
    if sessionOk h (templateOf h.ms u) (stmtAt h (v.msgs.getD 2 ([], .rej)).1 (templateOf h.ms u)) then
      some (stmtAt h (v.msgs.getD 2 ([], .rej)).1 (templateOf h.ms u),
        cutSeen [] (k₀ h (templateOf h.ms u)) (v.msgs.drop (off h (templateOf h.ms u))))
    else some (h.xS₀ (templateOf h.ms u), refusedSeen)
  else some (h.xS₀ (templateOf h.ms u), refusedSeen)

/-- Template `t`'s reference statement's masked level-0 lengths, summed. -/
noncomputable def n₀ (t : ℕ) : ℝ := ((shapeOf (h.xS₀ t)).map fun m => (2 : ℝ) ^ (Claims.Flock.zkLevel0 m).d).sum

/-- **The code before its samples**: the verifier, the statement and its evaluation, each unit's session on its
template's session's rounds, one session per template, its tables the most any template's reference statement has
(the template past the last, a unit past the members', included), its `N₀` template 0's, and the commitment's layout
and reference prover's framing `frame`'s. -/
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
    sessions := fun _ => h.ms.length
    tables := fun _ => ((List.range (h.ms.length + 1)).map fun t => (shapeOf (h.xS₀ t)).length).foldr max 0
    innerN₀ := fun _ => n₀ h 0 }

/-- The registration the samples read, at a statement that decodes. -/
def reg (x : Msg) : Option RegTwo := (decode x).map fun _ => h.reg

/-- **SPOT[Flock]'s two-stage code**: the code at its round order, its checked units the two stages' on the
auditor's registration, each stage's bytes its own coin's. -/
noncomputable abbrev code : Code := codeTwoAt (base h) (reg h) (fun _ => L) (fun _ => L) (order h) (fun _ => coinByte) (fun _ => coinByte)

/-- **The replay sample of record**: the first stage's checked slots on the replay layout. -/
noncomputable def rs : (x : Msg) → (Fin ((code h).V.k x) → FlockDeploy.Coin) → List ℕ :=
  replayTwo ((base h).atOrder (order h)) (reg h) (fun _ => L) (bytesFirst (base h) (fun _ => L) (order h) fun _ => coinByte)

/-- **The interior registration's roots** of replay block `ρ`, as the round-1 message states them; none where it
doesn't decode or names no such block. -/
def iroots (_ : Msg) (v : Seen) (ρ : ℕ) : List Msg :=
  match v.msgs with
  | (_, _) :: (m₁, _) :: _ => ((interiorsOf m₁).bind fun it => (it.roots.find? (·.1 == ρ)).map (·.2)).getD []
  | _ => []

end SpotFlock
