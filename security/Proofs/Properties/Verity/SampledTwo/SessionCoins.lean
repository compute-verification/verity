module

public import Proofs.Properties.Verity.SampledTwo.Reading
public import Proofs.Properties.Verity.Sampled.Binding
public import Proofs.Properties.Verity.Sampled.Moves

/-!
# The second stage's sessions' coins are the device's

The two-stage form's check verifies the second stage's sessions as one-stage v2's per-member form does
(`Spot.OneStageZk.membersRun`): each session's coins one tree of the interior's challenger session, the trees opened in
the sessions' order, each closed after its session's evidence. This module gives the reading of a session's rounds'
coins (`SessionReading.coin` and `.agree`) and states, at the reading, that the coin the check reads for a session's round
is the device's release for that round (`coinsRead_sessions`), and that the statements' round reads no coin
(`stmtReadsNone`).

* **A session's layout** (`slotStream`): a session's round `j` is, as one-stage v2 lays out a `--zk` session
  (`Verity.Sampled.slotAt`), the link's opening (the session stream's round 0) at `j = 0`, and round `k` of its table's
  rep `r` stream at `j = 2k + r + 1`.
* **The session-round locator** (`treesOpened`, `sessLoc`): from the frames before a release only, the run's round a
  session's opening continues. The tree's session is its place among the trees the log opens, in the order it first
  opens each; the round in the session is its stream's slot at the count of that stream's openings before it. The
  coin map's locator is to agree with it at the sessions' rounds (`LocatesSessions`).
* **What a round's releases are** (`located`): the frames and coins of the releases the locator puts on a round, in log
  order. The coin map's `dev` at a session's round is a function of them (`DevLocal`).
* **The log the check read** (`interiorRead`): the interior challenger session's releases, read as v2 reads a session's
  log (`Verity.Sampled.readLog`), from the case of the play the instance decodes; the same read as the cut's.
* **The sessions' coins** (`withCoins`): session `t`'s coin at its round `j` is `dev` at the read log, where every
  release the locator puts on the round shows its coins in the read; none otherwise. Two coins agree at a session's
  round on the elements the read's releases for it carry.
* **The premise** (`InteriorLogged`, the cut's): on an accepted play whose log the check read is the device's, the read
  log is the play's log after releases that open and sample no coin tree.
* **The statements** (`CoinsReadSome`, `coinsRead_sessions`, `stmtReadsNone`). `CoinsRead` asks the coin read to equal
  the device's release at every read round; at a session's round the check reads no coin where the evidence shows an
  opening without its coins, so the statement here is that wherever the check read a coin it is the device's release,
  which is what `projection_of` uses of `CoinsRead`.

That the coin read is the session's own tree's round, the coin its record gives the round, is `located_own`.

**The earlier signature.** Nothing here reads a signature. `InteriorLogged` comes from the cut's `LogIsDevice`, whose
forger term holds the hole of a history of one play (soundness's finding): a strategy replaying an earlier period's
genuine signature shows there, not here.
-/

@[expose] public section

namespace Verity.SampledTwo.SessionCoins

open Lean (Json)
open Verity.Sampled (helloSpecs repStreamName)
open Verity.Sampled.Binding (Read)

/-! ## A session's layout -/

/-- **The stream and the round of a session's round `j`**: the link's opening, the session stream's round 0, at
`j = 0`; round `k` of the table `tab`'s rep `r` stream at `j = 2k + r + 1` (`Verity.Sampled.slotAt`). -/
def slotStream (tab : String) (j : ℕ) : String × ℕ :=
  if j = 0 then (Spot.Challenger.TREE_SESSION, 0) else (repStreamName tab ((j - 1) % 2), (j - 1) / 2)

/-! ## The session-round locator -/

/-- The entries of `is` added to `acc` in order, each only where not yet in it. -/
def firstsOf : List ℕ → List ℕ → List ℕ
  | acc, [] => acc
  | acc, i :: is => firstsOf (if i ∈ acc then acc else acc ++ [i]) is

/-- The tree a frame opens. -/
def openTree : Frame → Option ℕ
  | .opening i _ _ => some i
  | _ => none

/-- **The trees the log opens, in the order it first opens each.** -/
def treesOpened (log : List Release) : List ℕ :=
  firstsOf [] (log.filterMap fun r => openTree r.frame)

/-- **The round of a session a stream's opening is**, at the session's table `tab`: the link's at the session stream's
first opening, round `2k + r + 1` at the `k`-th opening of the table's rep `r` stream (`slotStream`); none for any other
stream or a later opening of the session stream. -/
def roundOf (tab : String) (name : Option String) (k : ℕ) : Option ℕ :=
  if name = some Spot.Challenger.TREE_SESSION ∧ k = 0 then some 0
  else if name = some (repStreamName tab 0) then some (2 * k + 1)
  else if name = some (repStreamName tab 1) then some (2 * k + 2)
  else none

/-- **The session-round locator** at the sessions' first rounds `off`, their round counts `K`, their number `M` and
their tables `tab`: an opening of tree `i`'s stream `s` after the releases `pre` continues session `t`'s round `j`,
which is the run's round `off t + j`, where `t` is the tree's place among the trees `pre` opens in the order first
opened (a tree not yet opened the next), and `j` is the stream's round of the session (`roundOf`) at the count of its
openings in `pre`; none past the sessions or a session's rounds, and none at a frame that is no opening. It reads only
the frames of `pre`. -/
def sessLoc (off K : ℕ → ℕ) (M : ℕ) (tab : ℕ → String) (pre : List Release) : Frame → Option ℕ
  | .opening i s _ =>
    let t := (treesOpened pre).idxOf i
    (roundOf (tab t) ((helloSpecs pre)[i]?.bind fun sp => sp.streams[s]?) (openedOf pre i s)).bind fun j =>
      if t < M ∧ j < K t then some (off t + j) else none
  | _ => none

/-- The session-round locator at the session of record `A`: its sessions' first rounds, round counts and number,
`SpotFlock.off`, `SpotFlock.k₀` and the templates' count, at the tables `tab`. -/
def locOf (A : Live₂) (tab : ℕ → String) : List Release → Frame → Option ℕ :=
  sessLoc (SpotFlock.off A.h) (SpotFlock.k₀ A.h) A.h.ms.length tab

/-- **The coin map's locator agrees with the session-round locator at the sessions' rounds** (the coin map's; its
locator `L` takes the bytes the prover sent too). -/
def LocatesSessions (A : Live₂) (tab : ℕ → String) (L : List Release → Frame → ByteArray → Option ℕ) : Prop :=
  ∀ pre fr b n, groupOf A n = .sessions → (L pre fr b = some n ↔ locOf A tab pre fr = some n)

/-! ## What a round's releases are -/

/-- **The releases the locator `loc` puts on round `n`**, after the releases `pre`: each one's frame and coins, in log
order. -/
def located (loc : List Release → Frame → Option ℕ) (n : ℕ) : List Release → List Release → List (Frame × List Bool)
  | _, [] => []
  | pre, r :: log => (if loc pre r.frame = some n then [(r.frame, r.coins)] else []) ++ located loc n (pre ++ [r]) log

/-- **`dev` at a session's round is a function of the releases the locator puts there** (the coin map's): two logs
whose releases put on a session's round `n` are the same frames and coins give `dev` the same coin there. -/
def DevLocal (A : Live₂) (loc : List Release → Frame → Option ℕ) (dev : List Release → ℕ → Option FlockDeploy.Coin) :
    Prop :=
  ∀ n, groupOf A n = .sessions → ∀ log log', located loc n [] log = located loc n [] log' → dev log n = dev log' n

/-- The coins an opening of `n` coins carries: `n` elements of GF(2^128). -/
def widthOf : Frame → ℕ
  | .opening _ _ n => n
  | _ => 0

/-! ## The log the check read -/

/-- **The interior challenger session's releases as the check read them**, on the case of the play `o` the instance
decodes (`C`): the interior registration's commitment from the evidence's record, what its coins are read under
(`Properties.Challenger.oneStageZkSource`), the bytes its sample read (`OneStageZk.coinsOf`), and the second stage's
sessions' records on the case's blobs, every statement's in order (`Verity.Sampled.recordOn`), read from the interior
evidence's lines by `Verity.Sampled.readLog`; none where a step doesn't read. -/
def interiorRead (C : CaseOf) (o : Exchange) : Option (List Release) := do
  let k ← C o
  let p ← (Core.canonicalJson k.params >>= VStar.Recursive.SampledTwo.paramsOfJson).toOption
  let e ← (VStar.Recursive.SampledTwo.evidenceOf k.evidence).toOption
  let s ← (Flock.canon e.record).toOption
  let c₂ ← (Spot.OneStage.commitmentOf s.toUTF8).toOption
  let d ← Properties.Challenger.oneStageZkSource p.audit c₂
  let ξ₂ ← (Spot.OneStageZk.coinsOf p.audit c₂ e.interior).toOption
  let recs ← (e.statements.toList.flatMap (·.sessions.toList)).mapM (Verity.Sampled.recordOn k.blobs)
  Verity.Sampled.readLog d.label d.after recs ξ₂ (← (Verity.Sampled.linesOf e.interior).toOption)

/-- **The log the check read is the play's, after releases of no coin tree** (the cut's premise, from its
`LogIsDevice`): on a play the form accepts whose log the check read is the device's (`Good`), the interior read is the
play's log after releases that neither sample nor open a coin tree, each read release of the device's (`Read.Of`).
On one window the releases before are none; on two, the first session's. -/
def InteriorLogged (C : CaseOf) (Good : Exchange → Prop) : Prop :=
  ∀ o, FormAccepts C o → Good o → ∀ l, interiorRead C o = some l →
    ∃ q l', o.log = q ++ l' ∧ (∀ r ∈ q, openTree r.frame = none ∧ ∀ sp, r.frame ≠ .hello sp) ∧
      List.Forall₂ Read.Of (l.map Read.ofV2) l'

/-! ## The sessions' coins -/

/-- **The sessions' coins and their agreement**, over the messages' reading `Ms` (whose `stmts` and `msg` it keeps): at
session `t`'s round `j`, the run's round `n = off t + j`, the coin is `dev` at the log the check read, where every
release the locator puts on round `n` in that log shows its coins; none where it shows none of some, or the log doesn't
read. Two coins agree there on the elements those releases carry. -/
def withCoins (A : Live₂) (C : CaseOf) (tab : ℕ → String) (dev : List Release → ℕ → Option FlockDeploy.Coin)
    (Ms : SessionReading) : SessionReading :=
  { Ms with
    coin := fun o t j => (interiorRead C o).bind fun l =>
      if (located (locOf A tab) (SpotFlock.off A.h t + j) [] l).all (fun x => !x.2.isEmpty) then
        dev l (SpotFlock.off A.h t + j)
      else none
    agree := fun o t j c c' => ∃ l, interiorRead C o = some l ∧
      ∀ e : Fin 65536, e.val < ((located (locOf A tab) (SpotFlock.off A.h t + j) [] l).map fun x => widthOf x.1).sum →
        c e = c' e }

/-! ## The statements -/

/-- **Wherever the check read a coin at group `g`, it is the device's release** (`CoinsRead`, as `projection_of` uses
it): on a play the form's check accepts whose log the check read is the device's, a coin the check reads for a read
round of group `g` is the device's release for that round. -/
def CoinsReadSome (A : Live₂) (Rd : Reading A) (Acc Good : Exchange → Prop) (g : Group) : Prop :=
  ∀ o : Exchange, Acc o → Good o → ∀ i, Rd.reads i → groupOf A i = g → ∀ c, Rd.coin o i = some c →
    Rd.dev o.log i = some c

/-- **The statements' round reads no coin**, at any case, sessions' reading and device's releases: `readsAt` leaves
round `stage + 1` out. -/
theorem stmtReadsNone (A : Live₂) (C : CaseOf) (S : SessionReading)
    (dev : List Release → ℕ → Option FlockDeploy.Coin) : StmtReadsNone A (formReading A C S dev) := by
  intro i hi hg
  have hne : i ≠ A.h.stage + 1 := (show readsAt A.h i from hi).1
  unfold groupOf at hg
  split_ifs at hg
  all_goals first | exact hne ‹i = A.h.stage + 1› | cases hg

/-- **The sessions' coins the check reads are the device's** (piece: the sessions' coins): on a play the form accepts
whose log the check read is the device's (`Good`, through the cut's `InteriorLogged`), wherever the check read a coin
for a session's round, the device's release for that round (`dev`, at the coin map's locator, which agrees with the
session-round locator there: `DevLocal`) is that coin. -/
def CoinsReadSessions : Prop :=
  ∀ (A : Live₂) (C : CaseOf) (Ms : SessionReading) (tab : ℕ → String)
    (dev : List Release → ℕ → Option FlockDeploy.Coin) (Good : Exchange → Prop),
    DevLocal A (locOf A tab) dev → InteriorLogged C Good →
      CoinsReadSome A (formReading A C (withCoins A C tab dev Ms) dev) (FormAccepts C) Good .sessions

/-- **The second stage's sessions' records the check read**, on the case of the play `o`: every statement's sessions'
records in order, on the case's blobs (`Verity.Sampled.recordOn`), as `Spot.OneStageZk.membersTreesHeld` reads them. -/
def sessionRecords (C : CaseOf) (o : Exchange) : Option (List Json) := do
  let k ← C o
  let e ← (VStar.Recursive.SampledTwo.evidenceOf k.evidence).toOption
  (e.statements.toList.flatMap (·.sessions.toList)).mapM (Verity.Sampled.recordOn k.blobs)

/-- **Each session's coins are its own tree's** (piece: the sessions' coins, with the members' order): on a play the
form accepts, in the log the check read, the session-round locator puts at most one release on each round `j` of session
`t`, and that one is an opening whose coins are the ones session `t`'s record gives the round's stream's round
(`slotStream`, `Verity.Sampled.recordRound`). The trees open in the sessions' order and
each closes after its session's evidence (`Verity.Sampled.MemberForm.members_read`), which is what makes the tree the
locator counts `t`-th session `t`'s. -/
def LocatedOwn (A : Live₂) (C : CaseOf) (tab : ℕ → String) : Prop :=
  ∀ o, FormAccepts C o → ∀ l, interiorRead C o = some l → ∀ t j, t < A.h.ms.length → j < SpotFlock.k₀ A.h t →
    (located (locOf A tab) (SpotFlock.off A.h t + j) [] l).length ≤ 1 ∧
    ∀ x ∈ located (locOf A tab) (SpotFlock.off A.h t + j) [] l, ∃ recs rec,
      sessionRecords C o = some recs ∧ recs[t]? = some rec ∧
      (∃ i s n, x.1 = .opening i s n) ∧
      x.2 = ((Verity.Sampled.recordRound rec (slotStream (tab t) j).1 (slotStream (tab t) j).2).map
        Verity.bitsOfBytes).getD []

end Verity.SampledTwo.SessionCoins
