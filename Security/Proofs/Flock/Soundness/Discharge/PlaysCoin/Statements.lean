import Proofs.Flock.Soundness.Discharge.Exec.LiveOn
import Proofs.Flock.Soundness.Discharge.Exec.Shape

/-!
# The coin half of `DecodesAll` and `DecodesOn` (R11b): statements

`Decodes` (`Refine/LiveSim.lean`) asks for each rep's run on the tape `repTape`: each round's messages, decoded from its
bytes, then the round's coin step `coinMsg (one j)`; after the last round, the messages `fin` decodes from the proof
files. The refinement (`accepts_verdict_hm96`) gives each rep's run on two lists, its messages and its coin arrays
(`Plays`). With `Plays.coinVec` at `n ≠ 1`, a run's coin step is determined by its array (`coinArr`). So with
`one := fun _ => true` the coin steps are `coinMsg true` of the rounds, and what remains is the messages: the run's
messages, cut at the coins of the rep's spine, must be the decoder's groups.

* `ToTExact`: a run on two lists is a run on one tape whose coin steps are its arrays, as `coinArr` reads them.
* `CoinsOfRounds`: on well-formed rounds, those coin steps are `coinMsg true` of the rounds.
* `TapeOfGroups`: a tape is its coin steps and its messages cut at them (`cut`). When those are the rounds'
  `coinMsg true` and the decoder's groups (`roundGroups`) then `tail`, it is `tapeFrom dec (fun _ => true) j rs` then
  `tail`.
* `CutOfPat`: messages that are groups then a tail, along a spine whose coin pattern (`coinPat`) is theirs
  (`groupPat`), cut back into those groups and tail.
* `DecodesOfRuns`, `DecodesAllOfRuns`, `DecodesOnOfRuns`: `Decodes`, `DecodesAll` and `DecodesOn` at
  `one := fun _ => true`, from each rep's `RepRun`: a run on its rounds' coins whose messages, cut at its spine's coins,
  are the decoder's groups of the rounds and then `fin`'s messages.
* `DecodesOfRunsAt`: `DecodesOfRuns` for a game with no spine, such as one that goes on past a reject: each rep's spine
  is chosen at the outcome and holds along the runs that end where its `RepRun` ends (`SpineAt`).
-/

namespace FlockSoundness.Discharge.PlaysCoin

open Flock FlockSoundness.Game FlockSoundness.Refine FlockSoundness.Discharge.Exec Model Accounting Audit
  Audit.Partition

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

/-! ## Definitions -/

/-- A coin array as a run's coin step: its one word as a scalar, or else the vector of its words. -/
def coinArr {W F : Type} [Inhabited W] (ι : W → F) (c : Array W) : Msg :=
  if c.size = 1 then ⟨F, ι c[0]!⟩ else ⟨Fin c.size → F, fun i => ι c[i.val]!⟩

/-- `m` put at the front of the first group. -/
def consHead (m : Msg) : List (List Msg) → List (List Msg)
  | [] => [[m]]
  | g :: gs => (m :: g) :: gs

/-- **Messages cut at coins.** The messages `ms`, laid along the kinds `σ` and cut at `σ`'s coins: group `j` is the
messages between coins `j − 1` and `j`, and the last group is those after the last coin. -/
def cut : List Exec.Kind → List Msg → List (List Msg)
  | [], _ => [[]]
  | .msg _ :: σ, m :: ms => consHead m (cut σ ms)
  | .msg _ :: _, [] => [[]]
  | .coin _ :: σ, ms => [] :: cut σ ms

/-- A spine's coin pattern: `true` at each coin, `false` at each message. -/
def coinPat : List Exec.Kind → List Bool
  | [] => []
  | .msg _ :: σ => false :: coinPat σ
  | .coin _ :: σ => true :: coinPat σ

/-- The coin pattern of groups of sizes `ns`, each followed by a coin, then `t` messages. -/
def groupPat (ns : List ℕ) (t : ℕ) : List Bool :=
  ns.flatMap (fun k => List.replicate k false ++ [true]) ++ List.replicate t false

/-- The decoder's message groups of the rounds `rs` from round `j` on: round `j + i`'s messages, decoded from its
bytes. -/
def roundGroups (dec : ℕ → ByteArray → List Msg) : ℕ → List Round → List (List Msg)
  | _, [] => []
  | j, rd :: rs => dec j (rd.bytes.getD ByteArray.empty) :: roundGroups dec (j + 1) rs

/-- **A rep's run, as the message half gives it.** The rounds `rs` are well formed, and a run of `g` on their coins
ends in `p` with messages that, laid along the spine `σ` and cut at its coins, are the decoder's groups of `rs` and
then `tail`. -/
def RepRun {P : Type} (g : Game P) (σ : List Exec.Kind) (dec : ℕ → ByteArray → List Msg) (rs : List Round)
    (tail : List Msg) (p : P) : Prop :=
  (∀ rd ∈ rs, rd.coins.size = rd.n) ∧ ∃ ms : List Msg, Plays fOf g ms (rs.map (·.coins)) p ∧
    cut (σ.take (ms.length + rs.length)) ms = roundGroups dec 0 rs ++ [tail]

/-- **A spine at an end.** Every run of `g` that ends in `p` takes steps of the kinds of `σ`, in order: `Spine g σ`
along the runs that end in `p` only. -/
def SpineAt {P : Type} (g : Game P) (σ : List Exec.Kind) (p : P) : Prop :=
  ∀ ts : List Step, PlaysT g ts p → ts.map kindOf <+: σ

/-- A rep's spine at table `T` and link points `pts`: `repShape` at `T`'s dimensions, extra claims and schedule. -/
noncomputable abbrev repSpine (D : Type) {mPts : ℕ} (T : TabSpec mPts) (pts : Fin 2 → Fin mPts → 𝔽) : List Exec.Kind :=
  repShape 𝔽 𝕂 D T.S.m T.S.kLog (extraClaims T.S T.ptLocal mPts pts).length T.sch

/-! ## Statements -/

/-- **A run's tape, exactly.** A run on two lists is a run on one tape whose messages, in order, are the run's, and
whose coin steps, in order, are the run's coin arrays as `coinArr` reads them. -/
def ToTExact : Prop :=
  ∀ {W F : Type} [Inhabited W] (ι : W → F) {α : Type} {g : Game α} {ms : List Msg} {cs : List (Array W)} {a : α},
    Plays ι g ms cs a → ∃ ts : List Step, PlaysT g ts a ∧ ts.filterMap Sum.getLeft? = ms ∧
      ts.filterMap Sum.getRight? = cs.map (coinArr ι)

/-- **The coins of a run on rounds.** A run whose coin arrays are the coins of well-formed rounds `rs` is a run on one
tape whose messages are the run's and whose coin steps are `coinMsg true` of the rounds. -/
def CoinsOfRounds : Prop :=
  ∀ {α : Type} {g : Game α} {ms : List Msg} {rs : List Round} {a : α}, (∀ rd ∈ rs, rd.coins.size = rd.n) →
    Plays fOf g ms (rs.map (·.coins)) a → ∃ ts : List Step, PlaysT g ts a ∧ ts.filterMap Sum.getLeft? = ms ∧
      ts.filterMap Sum.getRight? = rs.map (coinMsg true)

/-- **A tape from its groups.** A tape whose coin steps are `coinMsg true` of the rounds `rs`, and whose messages, cut
at its coins, are the decoder's groups of `rs` from round `j` then `tail`, is the decoded tape of `rs` from round
`j`, then `tail`. -/
def TapeOfGroups : Prop :=
  ∀ (dec : ℕ → ByteArray → List Msg) (j : ℕ) (rs : List Round) (tail : List Msg) (ts : List Step),
    ts.filterMap Sum.getRight? = rs.map (coinMsg true) →
    cut (ts.map kindOf) (ts.filterMap Sum.getLeft?) = roundGroups dec j rs ++ [tail] →
    ts = tapeFrom dec (fun _ => true) j rs ++ tail.map .inl

/-- **Groups, cut back out.** Messages that are the groups `G` then `t`, along a spine whose coin pattern is `G`'s
groups each followed by a coin and then `t`'s messages, cut at its coins into `G` then `t`. -/
def CutOfPat : Prop :=
  ∀ (σ : List Exec.Kind) (G : List (List Msg)) (t : List Msg),
    coinPat σ = groupPat (G.map List.length) t.length → cut σ (G.flatten ++ t) = G ++ [t]

/-- **`Decodes` from the reps' runs.** With `one := fun _ => true`, `Decodes` holds when the reps' games have spines
`σ`, and at every outcome `pL` accepts, each rep has a `RepRun` on its rounds along its spine, with `fin`'s messages
as the tail, and the verdict holds at the two runs' ends. -/
def DecodesOfRuns : Prop :=
  ∀ {P OutT Cp Op : Type} [Nonempty Op] {mPts : ℕ} (out : Cp → P × P → Op → OutT) (pL : LiveOut mPts → Prop)
    (pg : OutT → Prop) (opn : Array ByteArray → Op) (dec : ℕ → ByteArray → List Msg)
    (fin : Fin 2 → Array ByteArray → List Msg) (capOf : ByteArray → Cp) (G : (Fin 2 → Fin mPts → 𝔽) → Fin 2 → Game P)
    (σ : (Fin 2 → Fin mPts → 𝔽) → Fin 2 → List Exec.Kind),
    (∀ pts r, Spine (G pts r) (σ pts r)) →
    (∀ o, pL o → ∃ pend : P × P, RepRun (G o.pts 0) (σ o.pts 0) dec o.rep0 (fin 0 o.proofs) pend.1 ∧
      RepRun (G o.pts 1) (σ o.pts 1) dec o.rep1 (fin 1 o.proofs) pend.2 ∧
      pg (out (capOf o.rootB) pend (opn o.proofs))) →
    Decodes out pL pg opn dec (fun _ => true) fin capOf G

/-- **`Decodes` from the reps' runs, with spines at their ends.** `DecodesOfRuns` with each rep's spine chosen at the
outcome and needed only along the runs that end where its `RepRun` ends (`SpineAt`), not along every run of its game. -/
def DecodesOfRunsAt : Prop :=
  ∀ {P OutT Cp Op : Type} [Nonempty Op] {mPts : ℕ} (out : Cp → P × P → Op → OutT) (pL : LiveOut mPts → Prop)
    (pg : OutT → Prop) (opn : Array ByteArray → Op) (dec : ℕ → ByteArray → List Msg)
    (fin : Fin 2 → Array ByteArray → List Msg) (capOf : ByteArray → Cp) (G : (Fin 2 → Fin mPts → 𝔽) → Fin 2 → Game P),
    (∀ o, pL o → ∃ pend : P × P, ∃ σ₀ σ₁ : List Exec.Kind,
      SpineAt (G o.pts 0) σ₀ pend.1 ∧ RepRun (G o.pts 0) σ₀ dec o.rep0 (fin 0 o.proofs) pend.1 ∧
      SpineAt (G o.pts 1) σ₁ pend.2 ∧ RepRun (G o.pts 1) σ₁ dec o.rep1 (fin 1 o.proofs) pend.2 ∧
      pg (out (capOf o.rootB) pend (opn o.proofs))) →
    Decodes out pL pg opn dec (fun _ => true) fin capOf G

/-- **`DecodesAll` from the reps' runs** (R11b's coin half). With `one := fun _ _ _ => true`, `DecodesAll` holds when,
at every draw and registration, on every live outcome whose session directory flock-verify accepts, each rep of the
table has a `RepRun` on its rounds along its spine `repSpine`, with `fin`'s messages as the tail, and the salted
table's verdict holds at the two runs' ends. -/
def DecodesAllOfRuns : Prop :=
  ∀ {D Sl : Type} [Nonempty D] [Inhabited D] [Inhabited Sl] {mPts n : ℕ} {Reg : Type}
    (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) (Lf : Merkle.Leaf (List 𝔽) Sl D H)
    (T : Finset (Fin n) → Reg → TabSpec mPts) (I : Inputs) (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript)
    (opn : Finset (Fin n) → Reg → Array ByteArray → Opens 𝔽 D × Salts Sl)
    (dec : Finset (Fin n) → Reg → ℕ → ByteArray → List Msg)
    (fin : Finset (Fin n) → Reg → Fin 2 → Array ByteArray → List Msg)
    (capOf : Finset (Fin n) → Reg → ByteArray → (Fin 1 → Cap D)),
    (∀ S R o, Accepts I (wr S R o) → ∃ pend : PendL 𝔽 D × PendL 𝔽 D,
      RepRun (repsOf (D := D) execArith (T S R) o.pts 0) (repSpine D (T S R) o.pts) (dec S R) o.rep0
        (fin S R 0 o.proofs) pend.1 ∧
      RepRun (repsOf (D := D) execArith (T S R) o.pts 1) (repSpine D (T S R) o.pts) (dec S R) o.rep1
        (fin S R 1 o.proofs) pend.2 ∧
      (tableOutL H E Lf (capOf S R o.rootB 0) pend (opn S R o.proofs)).accepted = true) →
    DecodesAll H E Lf T I wr opn dec (fun _ _ _ => true) fin capOf

/-- **`DecodesOn` from the reps' runs** (R11b's coin half, at the law's draws). With `one := fun _ _ _ => true`,
`DecodesOn L` holds when, at every draw `L.draw ω` and registration, on every live outcome whose session directory
flock-verify accepts, each rep of the table has a `RepRun` on its rounds along its spine `repSpine`, with `fin`'s
messages as the tail, and the salted table's verdict holds at the two runs' ends. -/
def DecodesOnOfRuns : Prop :=
  ∀ {D Sl : Type} [Nonempty D] [Inhabited D] [Inhabited Sl] {mPts n : ℕ} {Reg : Type}
    (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) (Lf : Merkle.Leaf (List 𝔽) Sl D H)
    (T : Finset (Fin n) → Reg → TabSpec mPts) (I : Inputs) (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript)
    (opn : Finset (Fin n) → Reg → Array ByteArray → Opens 𝔽 D × Salts Sl)
    (dec : Finset (Fin n) → Reg → ℕ → ByteArray → List Msg)
    (fin : Finset (Fin n) → Reg → Fin 2 → Array ByteArray → List Msg)
    (capOf : Finset (Fin n) → Reg → ByteArray → (Fin 1 → Cap D)) (L : Law n),
    (∀ ω R o, Accepts I (wr (L.draw ω) R o) → ∃ pend : PendL 𝔽 D × PendL 𝔽 D,
      RepRun (repsOf (D := D) execArith (T (L.draw ω) R) o.pts 0) (repSpine D (T (L.draw ω) R) o.pts)
        (dec (L.draw ω) R) o.rep0 (fin (L.draw ω) R 0 o.proofs) pend.1 ∧
      RepRun (repsOf (D := D) execArith (T (L.draw ω) R) o.pts 1) (repSpine D (T (L.draw ω) R) o.pts)
        (dec (L.draw ω) R) o.rep1 (fin (L.draw ω) R 1 o.proofs) pend.2 ∧
      (tableOutL H E Lf (capOf (L.draw ω) R o.rootB 0) pend (opn (L.draw ω) R o.proofs)).accepted = true) →
    DecodesOn H E Lf T I wr opn dec (fun _ _ _ => true) fin capOf L

end FlockSoundness.Discharge.PlaysCoin
