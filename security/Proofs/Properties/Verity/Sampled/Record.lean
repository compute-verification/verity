module

public import Proofs.Properties.Verity.Sampled
public import Proofs.Flock.Claims.Sampling.Session
public import Proofs.Flock.Claims.Sampling.Deployed
public import Proofs.Flock.Claims.Sampling.InputSlot
public import Proofs.Flock.Bridge.LeafMap
public import Proofs.Properties.Spot.Units
public import Proofs.Properties.Verity.Recursive.PartStatement

/-!
# One-stage v2's audit at a session, as a sampling record

The claims' sampling record (`Claims.Sampling.Deploy`) of the session one-stage v2 (`Spot.OneStageZk`) sets up
(`Verity.Sampled.Live`, `A`), with the verifier's own partition object and registered file `o` (`Own`): v2's audit in
public-coin form (`V`), with what the sampling claims read of it (`code`, `record`).

* **The verifier** (`V`), at the statement `xOf A o` (flock-verify's statement before the draw): `DRAW_ROUNDS` rounds
  of draw coins, whose bytes (`drawBytes`) are the challenger's `MAX_COINS · COIN_BYTES` bytes v2's draw may read; one
  round in which the prover sends the session's draw file `d`, which the verifier keeps only when its units are the ones
  the coins draw (`drawOk`: `provedOf`, as v2 credits a session's units, against `drawn`, as v2's challenger draw reads
  the coins), its session has the statement's numbers (`numsOk`) and its statement fits its length words (`Fits`, as v2
  refuses a length a session's statement can't hold); then the one session, flock-verify `--zk` in
  public-coin form (`FlockDeploy.V true`) at the session's statement `sessStmt A o d`. Every other statement is refused.
* **The session's numbers are the statement's**: the sampling record fixes each checked unit's session rounds by the
  statement, so the verifier holds the session's round count and `N₀` to those at a reference draw file (`d₀`, one
  whose session sets up one table of `A.kd` units: `RefOk`), and refuses a draw file that sets the session up at
  others.
* **What the claims read** (`code`): the statement's audit statement and partition are V5's at `A` (`Flock.Bridge.stmtOf`,
  `Flock.Bridge.partOf`); the draw precedes the session (`stage2` and every unit's `rounds` are the session's rounds);
  the checked units are the coins' draw (`checked`); every unit's session is the one session (`sessionOf`, with a
  refusing view where the draw file was refused); the commitment is `A`'s deployed one (`LayoutIsL`'s fields at `A`); one
  session of one table. What a session's witness reveals and the reference prover's framing are the record's
  parameters (`Ext`).
* **v2's slots over the lowered program** (`partSlots`, `designSlots`, `stmtSlots`): v2's two classes and two rules
  (`Slots.layout`, `slotDesign`) at the gates of the program the statement reads, the population in the registration's
  strata, slot `u` unit `u`'s lowered gates, then one slot per input gate of the lowered program, drawn `all`.
  `slotPartition` of the layout itself numbers the descriptor's gates, not the lowered program's, so it covers none of a
  unit's lowered gates. `PartSlotsTarget` states that `partSlots` partitions the lowered program where each unit has a
  gate and commits at most `unitClassMax` bits; `ScopeSlotsOk` is `ScopeOk` at these slots.
-/

@[expose] public section

namespace Verity.Sampled.Record

open Claims Claims.Sampling
open Lean (Json)

/-- **The verifier's own inputs flock-verify's statement holds**: its partition object and registered file, as bytes. -/
structure Own where
  partition : ByteArray
  registered : Option ByteArray

/-- **The session's statement before the draw**: v2's statement, the committed circuit and public files and lookup
tables, `--zk`, one table, the verifier's partition object, program and registered file, and no work, stratified,
closure or public-input input (`Spot.OneStageZk.inputsOf`). -/
def baseRaw (A : Live) (o : Own) : FlockDeploy.Raw where
  statement := _root_.Spot.OneStageZk.STATEMENT
  circuit := A.I.circuitFile
  publicFile := A.I.publicFile
  tables := A.I.tables
  zk := true
  osCoins := false
  count := 1
  draw := none
  partition := some o.partition
  program := A.I.program
  work := none
  workTable := none
  stratified := none
  closure := none
  registered := o.registered
  publicInputs := none

/-- **The record's statement.** -/
def xOf (A : Live) (o : Own) : Msg := (baseRaw A o).encode

/-- **The session's statement at the draw file `d`.** -/
def sessStmt (A : Live) (o : Own) (d : Msg) : Msg := { baseRaw A o with draw := some ⟨d.toArray⟩ }.encode

/-- The draw's rounds: `1024` coins of `2^20` bytes, the challenger's `MAX_COINS · COIN_BYTES`. -/
def DRAW_ROUNDS : ℕ := 1024

/-- The draw rounds' coins at the coins `r`. -/
def coinsAt {k : ℕ} (r : Fin k → FlockDeploy.Coin) : List FlockDeploy.Coin := (List.range DRAW_ROUNDS).map (coinSeq r)

/-- **The bytes the draw reads**: the coins' bytes, in order. -/
def drawBytes (cs : List FlockDeploy.Coin) : ByteArray := ⟨(cs.flatMap FlockDeploy.Coin.bytes).toArray⟩

/-- **The units the coins' bytes `src` draw** under the session's law over its population, as v2's challenger draw
reads them (`Spot.Draw.drawUnits`); a draw that fails draws every unit (`Live.L`). -/
def drawn (A : Live) (src : ByteArray) : List ℕ :=
  match _root_.Spot.Draw.drawUnits A.design A.n src with
  | .ok (d, _) => d.toList
  | .error _ => List.range A.n

/-- **The units a draw file proves**, as v2 credits a session's (`Spot.OneStageZk.provedUnits`): the public
file's `units.indices` of the instances the draw names, ascending (`Properties.Spot.Units.unitsAt`); the draw
file read as the session's statement reads it (`FlockDeploy.jsonOf`, `Flock.norm`), the public file's header as the
session reads it (`A.pub`). -/
def provedOf (A : Live) (d : Msg) : Option (Array ℕ) := do
  let j ← (FlockDeploy.jsonOf "draw" ⟨d.toArray⟩).toOption
  _root_.Properties.Spot.Units.unitsAt A.pub.header (Flock.norm j)

/-- **A reference draw file**: the session's statement at the draw file `d` reads as `baseRaw` with `d`, and sets up one
table, of `A.kd` units. -/
def RefOk (A : Live) (o : Own) (d : Msg) : Prop :=
  ∃ s t, ({ baseRaw A o with draw := some ⟨d.toArray⟩ } : FlockDeploy.Raw).ins = .ok s ∧
    FlockDeploy.setupOf true (sessStmt A o d) = some (s, t.spec, #[t]) ∧ t.n = A.kd

/-- The reference draw file: a draw file `RefOk` holds at, where there is one. -/
noncomputable def d₀ (A : Live) (o : Own) : Msg := Classical.epsilon (RefOk A o)

/-- The session's statement at the reference draw file. -/
noncomputable def xS₀ (A : Live) (o : Own) : Msg := sessStmt A o (d₀ A o)

/-- **A session statement's numbers**: flock-verify's rounds and `N₀` there. -/
noncomputable def Nums (xS : Msg) : ℕ × ℝ := ((FlockDeploy.V true).k xS, FlockDeploy.N₀ true xS)

/-- **The session has the statement's numbers** at the draw file `d`. -/
def numsOk (A : Live) (o : Own) (d : Msg) : Prop := Nums (sessStmt A o d) = Nums (xS₀ A o)

/-- The session's rounds the statement fixes. -/
noncomputable def kS (A : Live) (o : Own) : ℕ := (FlockDeploy.V true).k (xS₀ A o)

/-- **The session's statement at the draw file `d` fits its length words**: every file's size and the draw file's
under `2^64`, as `FlockDeploy.Raw.encode`'s 8-byte words hold them. -/
def Fits (A : Live) (o : Own) (d : Msg) : Prop :=
  Verity.Recursive.Part.RawFits { baseRaw A o with draw := some ⟨d.toArray⟩ }

/-- **The draw file round's check**: the draw file proves exactly the units the draw coins `cs` draw, its session
has the statement's numbers, and its statement fits its length words. -/
def drawOk (A : Live) (o : Own) (d : Msg) (cs : List FlockDeploy.Coin) : Prop :=
  provedOf A d = some ((drawn A (drawBytes cs)).toArray.qsort (· < ·)) ∧ numsOk A o d ∧ Fits A o d

/-- **A kept draw file fits its statement's length words.** -/
theorem fits_of_drawOk {A : Live} {o : Own} {d : Msg} {cs : List FlockDeploy.Coin} (h : drawOk A o d cs) : Fits A o d :=
  h.2.2

open Classical in
/-- **The verifier's step at the record's statement**: a draw round answers its coin; the draw file round answers `[]`
when `drawOk` holds, else refuses; the session's rounds are flock-verify's at the session's statement. -/
noncomputable def nextAt (A : Live) (o : Own) (as : List Msg) (cs : List FlockDeploy.Coin) : OC Out :=
  if as.length ≤ DRAW_ROUNDS then .ret (.msg (cs.getLast?.getD default).bytes)
  else if as.length = DRAW_ROUNDS + 1 then
    if drawOk A o (as.getD DRAW_ROUNDS []) (cs.take DRAW_ROUNDS) then .ret (.msg []) else .ret .rej
  else FlockDeploy.next true (sessStmt A o (as.getD DRAW_ROUNDS [])) (as.drop (DRAW_ROUNDS + 1))
    (cs.drop (DRAW_ROUNDS + 1))

/-- **One-stage v2's audit at the session, in public-coin form**: the draw rounds, the draw file round and the session's
rounds at the record's statement; one refusing round elsewhere. -/
noncomputable def V (A : Live) (o : Own) : Verifier FlockDeploy.Coin where
  k x := if x = xOf A o then DRAW_ROUNDS + 1 + kS A o else 1
  next x as cs := if x = xOf A o then nextAt A o as cs else .ret .rej

/-- **The session's rounds**: those after the draw file round. -/
def after (k : ℕ) : Finset (Fin k) := Finset.univ.filter fun t => DRAW_ROUNDS + 1 ≤ t.val

/-- A refused draw file's session: the reference statement, and a prover that sends `[2]`, which flock-verify refuses
at once. -/
noncomputable def refused (A : Live) (o : Own) : Msg × Seen := (xS₀ A o, ⟨[([2], .rej)], .rej⟩)

open Classical in
/-- **The session a run shows**: where the draw file round kept the draw file `d`, the session has the statement's
numbers and its statement fits its length words, the session's statement and the run from the session's first round on;
else `refused`. -/
noncomputable def sessionOf (A : Live) (o : Own) (v : Seen) : Msg × Seen :=
  match v.msgs[DRAW_ROUNDS]? with
  | some (d, .msg []) =>
    if numsOk A o d ∧ Fits A o d then (sessStmt A o d, ⟨v.msgs.drop (DRAW_ROUNDS + 1), v.out⟩) else refused A o
  | _ => refused A o

/-- The statement's audit statement: V5's at the session (`Live.extract`'s). -/
noncomputable def stmtA (A : Live) : AuditStatement :=
  Flock.Bridge.stmtOf A.P A.κ (_root_.Properties.Spot.oneStageOf A.design) A.desc ⟨"", 0, []⟩ A.roots

/-- The statement's partition: V5's at the session's population. -/
noncomputable def partA (A : Live) : Claims.Sampling.Partition :=
  Flock.Bridge.partOf A.P A.κ (Flock.Bridge.nsOf A.desc A.qr)

/-- **The statement's partition in v2's slots**: the population in the registration's strata (`nsOf`), slot `u` unit
`u`'s lowered gates (`unitGates`), then the input class of the lowered program (`InputSlot.withInputs`), each input
gate a proof block of its own. -/
noncomputable def partSlots (A : Live) : Claims.Sampling.Partition :=
  InputSlot.withInputs
    ⟨[(Flock.Bridge.nsOf A.desc A.qr).splitLengths ((List.finRange A.n).map (Flock.Bridge.unitGates A.P))]⟩
    (Flock.Bridge.progOf A.Cc A.κ)

/-- **v2's design** (`slotDesign`'s two rules): the law's one-stage rule over the population, then every input slot. -/
def designSlots (A : Live) : Claims.Sampling.Design :=
  ⟨[_root_.Properties.Spot.oneStageOf A.design, .all]⟩

/-- The statement's audit statement at v2's design. -/
noncomputable def stmtSlots (A : Live) : AuditStatement := { stmtA A with design := designSlots A }

/-- **`partSlots` partitions the lowered program** where every unit has a lowered gate and commits at most
`unitClassMax` bits. No unit reads another's gates, so a unit commits only its returned bits, and an input slot commits
its one input. -/
def PartSlotsTarget : Prop :=
  ∀ A : Live, (∀ u, Flock.Bridge.unitGates A.P u ≠ []) →
    (∀ u, Claims.Sampling.committed (stmtSlots A).prog (Flock.Bridge.unitGates A.P u) ≤ Claims.Sampling.unitClassMax) →
      (partSlots A).Partitions (stmtSlots A).prog

/-- **The statement is in scope at v2's slots** (`ScopeOk` at `partSlots` and `designSlots`): `partSlots` partitions
the lowered program, and the law's one-stage rule is well formed over the population's strata. The input class's rule,
`all`, is well formed at every class. -/
def ScopeSlotsOk (A : Live) : Prop :=
  (partSlots A).Partitions (stmtSlots A).prog ∧
    (_root_.Properties.Spot.oneStageOf A.design).WellFormed (Flock.Bridge.nsOf A.desc A.qr)

/-- **What the record takes as given**: what a session's Flock witness reveals and how many openings at most
(`reveal`, `reveals`), and the reference prover's framing (`step`, `reads`, `wit`). -/
structure Ext where
  reveal : AuditStatement → ℕ → Seen → Claims.Flock.Wit → List (List Msg × Opening Bool)
  reveals : Msg → ℕ
  step : Msg → List (Msg × Msg) → Reference.Step
  reads : AuditStatement → Msg → List ℕ
  wit : AuditStatement → Msg → List (Opening Bool) → Claims.Flock.Wit

open Classical in
/-- **One-stage v2's code at the session**, as the sampling record reads it. -/
noncomputable def code (A : Live) (o : Own) (e : Ext) : SamplingDeploy.Code where
  V := V A o
  parse x := if x = xOf A o then some (stmtA A) else none
  scopeCheck x := decide (x = xOf A o ∧ InScope (partA A) (stmtA A))
  partition _ := some (partA A)
  stage2 _ := after _
  checked x r := if x = xOf A o then drawn A (drawBytes (coinsAt r)) else []
  rounds _ _ := after _
  session x _ v := if x = xOf A o then some (sessionOf A o v) else none
  reveal := e.reveal
  own := Flock.Bridge.portsOfL A.D.unroll A.Lm A.roots
  at_ := Flock.Bridge.atOf A.Cc A.rows
  layer _ _ _ := []
  place := Flock.Bridge.placeOf A.Cc A.rows
  bits := Flock.Bridge.bitsOf A.Lm.d
  salt := Flock.Bridge.saltOf
  π := Flock.Bridge.pathOfL A.Cc A.rows A.Lm
  innerN₀ _ := FlockDeploy.N₀ true (xS₀ A o)
  tables _ := 1
  reveals := e.reveals
  sessions _ := 1
  step := e.step
  reads := e.reads
  wit := e.wit

/-- **The record**: the code at the session, its session C-Flock's masked verifier. -/
noncomputable def record (A : Live) (o : Own) (e : Ext) :
    Claims.Sampling.Deploy FlockDeploy.Coin (List FlockLevel3.GF128) Claims.Flock.ZkTrace :=
  SamplingDeploy.deployOf (FlockDeploy.deployZk _root_.Flock.Readable.relation) (code A o e)

/-! ## Unfolding -/

section Unfold

variable (A : Live) (o : Own) (e : Ext)

theorem record_V : (record A o e).V = V A o := rfl

theorem record_stage2 (x : Msg) : (record A o e).stage2 x = after _ := rfl

theorem record_rounds (x : Msg) (u : ℕ) : (record A o e).rounds x u = after _ := rfl

theorem record_checked (x : Msg) (r : Fin ((V A o).k x) → FlockDeploy.Coin) :
    (record A o e).checked x r = if x = xOf A o then drawn A (drawBytes (coinsAt r)) else [] := rfl

theorem V_k_xOf : (V A o).k (xOf A o) = DRAW_ROUNDS + 1 + kS A o := by
  simp only [V, ↓reduceIte]

theorem V_next_xOf : (V A o).next (xOf A o) = nextAt A o := by
  funext as cs
  simp only [V, ↓reduceIte]

theorem V_k_ne {x : Msg} (h : x ≠ xOf A o) : (V A o).k x = 1 := ite_eq_right h

/-- The commitment's fields are `LayoutIsL`'s at `A`, wherever its circuit's rows are `sha512/row/v2`'s. -/
theorem layoutIsL (h : Flock.Bridge.RowsV2 A.D.unroll) :
    Flock.Bridge.LayoutIsL A.Cc A.rows A.D.unroll A.Lm A.roots (code A o e) :=
  ⟨h, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

end Unfold

/-- **The draw rounds' coins read no session round**: coins that agree off the session's rounds give the same draw
rounds' coins. -/
theorem coinsAt_congr {k : ℕ} {r r' : Fin k → FlockDeploy.Coin} (h : ∀ t ∉ after k, r t = r' t) :
    coinsAt r = coinsAt r' := by
  unfold coinsAt
  refine List.map_congr_left fun i hi => ?_
  have hi : i < DRAW_ROUNDS := List.mem_range.1 hi
  unfold coinSeq
  by_cases hk : i < k
  · rw [dite_eq_left hk, dite_eq_left hk]
    exact h _ (by simp only [after, Finset.mem_filter, Finset.mem_univ, true_and, not_le]; omega)
  · rw [dite_eq_right hk, dite_eq_right hk]

end Verity.Sampled.Record
