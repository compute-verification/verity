module

public import Proofs.Properties.Verity.Sampled
public import Proofs.Flock.Claims.Sampling.Session
public import Proofs.Flock.Claims.Sampling.Deployed
public import Proofs.Flock.Claims.Sampling.InputSlot
public import Proofs.Flock.Bridge.LeafMap
public import Proofs.Properties.Spot.Units
public import Proofs.Properties.Verity.Recursive.PartStatement

/-! One-stage v2's audit at a session, as a sampling record. -/

@[expose] public section

namespace Verity.Sampled.Record

open Claims Claims.Sampling
open Lean (Json)

structure Own where
  partition : ByteArray
  registered : Option ByteArray

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

def xOf (A : Live) (o : Own) : Msg := (baseRaw A o).encode

def sessStmt (A : Live) (o : Own) (d : Msg) : Msg := { baseRaw A o with draw := some ⟨d.toArray⟩ }.encode

/-- The sample's rounds: `1024` coins of `2^20` bytes, the challenger's `MAX_COINS · COIN_BYTES`. -/
def SAMPLE_ROUNDS : ℕ := 1024

def coinsAt {k : ℕ} (r : Fin k → FlockDeploy.Coin) : List FlockDeploy.Coin := (List.range SAMPLE_ROUNDS).map (coinSeq r)

def sampleBytes (cs : List FlockDeploy.Coin) : ByteArray := ⟨(cs.flatMap FlockDeploy.Coin.bytes).toArray⟩

def drawn (A : Live) (src : ByteArray) : List ℕ :=
  match _root_.Spot.Draw.sampleUnits A.design A.n src with
  | .ok (d, _) => d.toList
  | .error _ => List.range A.n

def provedOf (A : Live) (d : Msg) : Option (Array ℕ) := do
  let j ← (FlockDeploy.jsonOf "draw" ⟨d.toArray⟩).toOption
  _root_.Properties.Spot.Units.unitsAt A.pub.header (Flock.norm j)

def RefOk (A : Live) (o : Own) (d : Msg) : Prop :=
  ∃ s t, ({ baseRaw A o with draw := some ⟨d.toArray⟩ } : FlockDeploy.Raw).ins = .ok s ∧
    FlockDeploy.setupOf true (sessStmt A o d) = some (s, t.spec, #[t]) ∧ t.n = A.kd

noncomputable def d₀ (A : Live) (o : Own) : Msg := Classical.epsilon (RefOk A o)

noncomputable def xS₀ (A : Live) (o : Own) : Msg := sessStmt A o (d₀ A o)

noncomputable def Nums (xS : Msg) : ℕ × ℝ := ((FlockDeploy.V true).k xS, FlockDeploy.N₀ true xS)

def numsOk (A : Live) (o : Own) (d : Msg) : Prop := Nums (sessStmt A o d) = Nums (xS₀ A o)

noncomputable def kS (A : Live) (o : Own) : ℕ := (FlockDeploy.V true).k (xS₀ A o)

def Fits (A : Live) (o : Own) (d : Msg) : Prop :=
  Verity.Recursive.Part.RawFits { baseRaw A o with draw := some ⟨d.toArray⟩ }

def sampleOk (A : Live) (o : Own) (d : Msg) (cs : List FlockDeploy.Coin) : Prop :=
  provedOf A d = some ((drawn A (sampleBytes cs)).toArray.qsort (· < ·)) ∧ numsOk A o d ∧ Fits A o d

theorem fits_of_sampleOk {A : Live} {o : Own} {d : Msg} {cs : List FlockDeploy.Coin} (h : sampleOk A o d cs) : Fits A o d :=
  h.2.2

open Classical in
noncomputable def nextAt (A : Live) (o : Own) (as : List Msg) (cs : List FlockDeploy.Coin) : OC Out :=
  if as.length ≤ SAMPLE_ROUNDS then .ret (.msg (cs.getLast?.getD default).bytes)
  else if as.length = SAMPLE_ROUNDS + 1 then
    if sampleOk A o (as.getD SAMPLE_ROUNDS []) (cs.take SAMPLE_ROUNDS) then .ret (.msg []) else .ret .rej
  else FlockDeploy.next true (sessStmt A o (as.getD SAMPLE_ROUNDS [])) (as.drop (SAMPLE_ROUNDS + 1))
    (cs.drop (SAMPLE_ROUNDS + 1))

noncomputable def V (A : Live) (o : Own) : Verifier FlockDeploy.Coin where
  k x := if x = xOf A o then SAMPLE_ROUNDS + 1 + kS A o else 1
  next x as cs := if x = xOf A o then nextAt A o as cs else .ret .rej

def after (k : ℕ) : Finset (Fin k) := Finset.univ.filter fun t => SAMPLE_ROUNDS + 1 ≤ t.val

noncomputable def refused (A : Live) (o : Own) : Msg × Seen := (xS₀ A o, ⟨[([2], .rej)], .rej⟩)

open Classical in
noncomputable def sessionOf (A : Live) (o : Own) (v : Seen) : Msg × Seen :=
  match v.msgs[SAMPLE_ROUNDS]? with
  | some (d, .msg []) =>
    if numsOk A o d ∧ Fits A o d then (sessStmt A o d, ⟨v.msgs.drop (SAMPLE_ROUNDS + 1), v.out⟩) else refused A o
  | _ => refused A o

noncomputable def stmtA (A : Live) : AuditStatement :=
  Flock.Bridge.stmtOf A.P A.κ (_root_.Properties.Spot.oneStageOf A.design) A.desc ⟨"", 0, []⟩ A.roots

noncomputable def partA (A : Live) : Claims.Sampling.Partition :=
  Flock.Bridge.partOf A.P A.κ (Flock.Bridge.nsOf A.desc A.qr)

noncomputable def partSlots (A : Live) : Claims.Sampling.Partition :=
  InputSlot.withInputs
    ⟨[(Flock.Bridge.nsOf A.desc A.qr).splitLengths ((List.finRange A.n).map (Flock.Bridge.unitGates A.P))]⟩
    (Flock.Bridge.progOf A.Cc A.κ)

def designSlots (A : Live) : Claims.Sampling.Design :=
  ⟨[_root_.Properties.Spot.oneStageOf A.design, .all]⟩

noncomputable def stmtSlots (A : Live) : AuditStatement := { stmtA A with design := designSlots A }

def PartSlotsTarget : Prop :=
  ∀ A : Live, (∀ u, Flock.Bridge.unitGates A.P u ≠ []) →
    (∀ u, Claims.Sampling.committed (stmtSlots A).prog (Flock.Bridge.unitGates A.P u) ≤ Claims.Sampling.sizeClassMax) →
      (partSlots A).Partitions (stmtSlots A).prog

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
noncomputable def code (A : Live) (o : Own) (e : Ext) : SamplingDeploy.Code where
  V := V A o
  parse x := if x = xOf A o then some (stmtA A) else none
  scopeCheck x := decide (x = xOf A o ∧ InScope (partA A) (stmtA A))
  partition _ := some (partA A)
  secondStage _ := after _
  checked x r := if x = xOf A o then drawn A (sampleBytes (coinsAt r)) else []
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

noncomputable def record (A : Live) (o : Own) (e : Ext) :
    Claims.Sampling.Deploy FlockDeploy.Coin (List FlockLevel3.GF128) Claims.Flock.ZkTrace :=
  SamplingDeploy.deployOf (FlockDeploy.deployZk _root_.Flock.Readable.relation) (code A o e)

section Unfold

variable (A : Live) (o : Own) (e : Ext)

theorem record_V : (record A o e).V = V A o := rfl

theorem record_secondStage (x : Msg) : (record A o e).secondStage x = after _ := rfl

theorem record_rounds (x : Msg) (u : ℕ) : (record A o e).rounds x u = after _ := rfl

theorem record_checked (x : Msg) (r : Fin ((V A o).k x) → FlockDeploy.Coin) :
    (record A o e).checked x r = if x = xOf A o then drawn A (sampleBytes (coinsAt r)) else [] := rfl

theorem V_k_xOf : (V A o).k (xOf A o) = SAMPLE_ROUNDS + 1 + kS A o := by
  simp only [V, ↓reduceIte]

theorem V_next_xOf : (V A o).next (xOf A o) = nextAt A o := by
  funext as cs
  simp only [V, ↓reduceIte]

theorem V_k_ne {x : Msg} (h : x ≠ xOf A o) : (V A o).k x = 1 := ite_eq_right h

theorem layoutIsL (h : Flock.Bridge.RowsV2 A.D.unroll) :
    Flock.Bridge.LayoutIsL A.Cc A.rows A.D.unroll A.Lm A.roots (code A o e) :=
  ⟨h, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

end Unfold

theorem coinsAt_congr {k : ℕ} {r r' : Fin k → FlockDeploy.Coin} (h : ∀ t ∉ after k, r t = r' t) :
    coinsAt r = coinsAt r' := by
  unfold coinsAt
  refine List.map_congr_left fun i hi => ?_
  have hi : i < SAMPLE_ROUNDS := List.mem_range.1 hi
  unfold coinSeq
  by_cases hk : i < k
  · rw [dite_eq_left hk, dite_eq_left hk]
    exact h _ (by simp only [after, Finset.mem_filter, Finset.mem_univ, true_and, not_le]; omega)
  · rw [dite_eq_right hk, dite_eq_right hk]

end Verity.Sampled.Record
