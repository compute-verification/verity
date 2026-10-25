import SampledProofs.Registration
import Proofs.Flock.Claims.Sampling.Reference.Deployed

/-!
# The sampled verifier's code from its Lean draw

`SamplingDeploy.code c reg L bytes sr sess`: the record's code (`SamplingDeploy.Code`) with what the Lean sampled
verifier (`SampledProofs`) computes taken from it, and the rest `c`'s:

* `checked`: the draw of record. `SampledProofs.Slots.select` runs on the registration the statement names (`reg x`, a
  `Reg`: the selection rule, the verifier's own program, query and work table, and the challenger's closure map, as
  `SampledProofs.Run.run` reads them) over `L x` bytes of the coins (`bytes`), and every slot of the layout is
  checked where the draw is refused (`drawChecked`). Nothing is checked where the statement names no registration.
* `sessions`, `session`, `rounds`: one proof session per registration proves every drawn unit
  (`Stream.session`, where each covered registration's draw is its own tables of the session's statement). So every
  checked unit's session is the run's one session (`sess`) on its rounds (`sr`), and `sessions = 1`.

`SampledProofs.Run.run` takes the session's statement and the verifiers' verdicts as inputs (`Run.Input.statement`,
`verdicts`, `sessions`), and nothing in `SampledProofs` composes a session's statement, encodes its Flock witness or runs
Flock. So `sess`, the reference prover's framing (`step`, `reads`, `wit`), the verifier in public-coin form, its parser,
scope check and partition, and the commitment's layout stay parameters.

At this code:

* `sessionSlots_code`: `SessionSlots` holds (`sessionSlots_of_one`).
* `δc_code`: the record's completeness error is the largest `ZkTables.δc` over its one session's statements, with no
  union bound factor.
* `drawSlots_code`: `DrawSlots` from `RegMatches` (the registration's program is the statement's descriptor, its layout
  the statement's partition and its design the statement's design) and `BytesUniform` (the bytes are uniform when the
  coins are).
* `drawFixed_code`: `DrawFixed` from `BytesFixed` (the session's rounds are second-stage, and the draw's bytes don't
  read them).
* `framed_code_iff`, `acceptsFramed_code_iff`, `sessionsScope_code_iff`, `sessionWit_code_iff`, `sessionsIn_code_iff`:
  `Framed`, `AcceptsFramed`, `SessionsScope`, `SessionWit` and `SessionsIn` are exactly their one-session forms
  (`FramedOne`, `AcceptsOne`, `SessScope`, `SessWit`, `SessIn`). Each reads `sess` once at a run that checks some unit,
  with no unit named.
* `N₀_of_shapes`, `sessIn_of_shapes`: `N₀` reads only the session's table sizes, so `SessIn`'s `N₀` clause holds when
  every session at `x` has the tables `shape x` fixes (`ShapesFixed`). Nothing the Lean verifier runs establishes it at
  `sess`: its registration refusal (`Run.countVaries`) fixes the drawn count only, and its outcome step takes the
  session's statement as an opaque object.
* `shapeChecked`, `shapesFixed_checked`, `shapesFixed_iff_checked`, `sessIn_checked`: the shape check an outcome step
  would run (`sess` with every session refused whose tables aren't `shape x`) fixes the shapes, refuses nothing exactly
  when `ShapesFixed` holds at `sess`, and puts its sessions in `SessIn`.
* `complete_code`: `Complete` at the code from Flock's masked completeness and those premises.
-/

namespace SamplingDeploy

open Claims Claims.Sampling Claims.Sampling.Reference Properties.SampledProofs
open Lean (Json)

/-- **A registration as the draw of record reads it**, what `SampledProofs.Run.run` passes to `Select.select`: the
verifier's own copies (`Run.Input.params.verifier`, whose program, query and work table the draw reads), the selection
rule and the challenger's closure map. -/
structure Reg where
  verifier : SampledProofs.Registration.Verifier
  rule : SampledProofs.Select.Rule
  closure : Option (Array (Int × Array Int))

def Reg.program (g : Reg) : SampledProofs.Program := g.verifier.program

def Reg.query (g : Reg) : Json := g.verifier.query

def Reg.table (g : Reg) : Option Json := g.verifier.workTable

/-- The registration's layout (`Slots.layout`), empty where its query doesn't apply. -/
def Reg.layout (g : Reg) : SampledProofs.Slots.Layout :=
  match SampledProofs.Slots.layout g.program g.query with
  | .ok l => l
  | .error _ => default

/-- **The draw of record's checked units** on the bytes `src`: `Slots.select`'s, and every slot of the layout where it
is refused (`drawChecked`). -/
def Reg.checked (g : Reg) (src : ByteArray) : List (ℕ × ℕ) :=
  (drawChecked g.rule g.program g.query g.table g.closure g.layout src).toList

/-- **The sampled verifier's code with its Lean draw**: `c` with `checked` the draw of record on the registration
`reg x` over the bytes `bytes x` of the coins, and one session per registration, `sess`, on the rounds `sr`. -/
def code (c : Code) (reg : Msg → Option Reg) (L : Msg → ℕ)
    (bytes : (x : Msg) → (Fin (c.V.k x) → FlockDeploy.Coin) → Fin (L x) → Fin 256)
    (sr : (x : Msg) → Finset (Fin (c.V.k x))) (sess : Msg → Seen → Option (Msg × Seen)) : Code :=
  { c with
    checked := fun x r => ((reg x).map fun g => g.checked (bytesOf (bytes x r))).getD []
    rounds := fun x _ => sr x
    sessions := fun _ => 1
    session := fun x _ v => sess x v }

/-- **One session that every checked unit shares has `SessionSlots`**: the first checked unit's session is every
checked unit's. -/
theorem sessionSlots_of_one (H : Msg → Msg) (d : Code) (hn : ∀ x, 1 ≤ d.sessions x)
    (hs : ∀ x u u' v, d.session x u v = d.session x u' v) : SessionSlots d H := by
  intro x _ Ω P ω r u hu
  cases hc : d.checked x r with
  | nil => rw [hc] at hu; simp at hu
  | cons u' l => exact ⟨0, hn x, u', rfl, hs x u u' _⟩

variable (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (c : Code) (reg : Msg → Option Reg)
  (L : Msg → ℕ) (bytes : (x : Msg) → (Fin (c.V.k x) → FlockDeploy.Coin) → Fin (L x) → Fin 256)
  (sr : (x : Msg) → Finset (Fin (c.V.k x))) (sess : Msg → Seen → Option (Msg × Seen))

theorem code_checked (x : Msg) (r : Fin (c.V.k x) → FlockDeploy.Coin) :
    (code c reg L bytes sr sess).checked x r = ((reg x).map fun g => g.checked (bytesOf (bytes x r))).getD [] :=
  rfl

theorem code_session (x : Msg) (u : ℕ × ℕ) (v : Seen) : (code c reg L bytes sr sess).session x u v = sess x v := rfl

theorem sessionSlots_code (H : Msg → Msg) : SessionSlots (code c reg L bytes sr sess) H :=
  sessionSlots_of_one H _ (fun _ => le_rfl) fun _ _ _ _ => rfl

/-- **The completeness error at the code**: the largest `ZkTables.δc H`, capped at `1`, over the statements of the one
session of the reference prover's honest runs that check some unit. -/
theorem δc_code (H : Msg → Msg) (x : Msg) :
    (deploy rel (code c reg L bytes sr sess)).δc H x =
      sessErrAt H (Rhon H (commit c) c.parse c.partition)
        (refP FlockComplete.ZkTables.honestPR c.step c.parse c.reads (commit c) c.wit) c.V
        (code c reg L bytes sr sess).checked (fun x _ v => sess x v) (FlockComplete.ZkTables.δc H) x := by
  show ((1 : ℕ) : ℝ) * _ = _
  rw [Nat.cast_one, one_mul]
  rfl

/-! ## The draw -/

/-- **The registration is the statement's**: at a statement that parses and whose query evaluates, the statement names
a registration whose program is the statement's descriptor (`Program.parse`), whose layout (`Slots.layout`) is the
statement's partition and whose design (`slotDesign`) is the statement's. -/
def RegMatches : Prop :=
  ∀ x s Pt, c.parse x = some s → c.partition s = some Pt →
    ∃ g l, reg x = some g ∧ SampledProofs.Program.parse ⟨s.desc.toArray⟩ = .ok g.program ∧
      SampledProofs.Slots.layout g.program g.query = .ok l ∧ Pt = slotPartition l ∧
      slotDesign g.rule g.program g.query g.table = .ok s.design

/-- **The draw's bytes are uniform when the coins are.** -/
def BytesUniform : Prop :=
  ∀ x (E : (Fin (L x) → Fin 256) → Prop), Pr (fun r : Fin (c.V.k x) → FlockDeploy.Coin => E (bytes x r)) = Pr E

/-- **The session's rounds are second-stage and the draw doesn't read them.** -/
def BytesFixed : Prop :=
  ∀ x, sr x ⊆ c.stage2 x ∧ ∀ r d, bytes x (forkOn (sr x) r d) = bytes x r

theorem drawSlots_code (hreg : RegMatches c reg) (hunif : BytesUniform c L bytes) :
    DrawSlots (deploy rel (code c reg L bytes sr sess)) := by
  intro x s Pt hx hP
  obtain ⟨g, l, hg, hd, hl, hPt, hds⟩ := hreg x s Pt hx hP
  refine ⟨g.rule, g.program, g.query, g.table, g.closure, l, L x, bytes x, hd, hl, hPt ▸ rfl, hds, hunif x,
    fun r => ?_⟩
  show ((reg x).map fun g => g.checked (bytesOf (bytes x r))).getD [] = _
  rw [hg]
  simp only [Option.map_some, Option.getD_some, Reg.checked, Reg.layout, hl]

theorem drawFixed_code (h : BytesFixed c L bytes sr) : DrawFixed (deploy rel (code c reg L bytes sr sess)) := by
  intro x r u _
  refine ⟨(h x).1, fun d => ?_⟩
  show ((reg x).map fun g => g.checked (bytesOf (bytes x (forkOn (sr x) r d)))).getD [] =
    ((reg x).map fun g => g.checked (bytesOf (bytes x r))).getD []
  rw [(h x).2 r d]

/-! ## The premises in their one-session form -/

/-- **`Framed` at one session**: at coins that check some unit, the session is the Flock run of a prover that answers
as `P` does wherever the router puts the run in the session, on the session's rounds. -/
def FramedOne (H : Msg → Msg) : Prop :=
  ∀ x {Ω : Type} (P : Prover Ω) (ω : Ω) (r : Fin (c.V.k x) → FlockDeploy.Coin),
    (code c reg L bytes sr sess).checked x r ≠ [] →
    ∃ (xS : Msg) (emb : Fin ((FlockDeploy.V true).k xS) → Fin (c.V.k x)), Function.Injective emb ∧
      (∀ i, emb i ∈ sr x) ∧
      ∀ {ΩF : Type} (PF : Prover ΩF) (ωF : ΩF),
        (∀ hist hS, c.step x hist = .session xS hS →
          ((P.next x hist ω).eval H).1 = ((PF.next xS hS ωF).eval H).1) →
        ∀ r' : Fin (c.V.k x) → FlockDeploy.Coin, (∀ t ∉ sr x, r' t = r t) →
          sess x (run H P c.V x ω r').seen =
            some (xS, (run H PF (FlockDeploy.V true) xS ωF fun i => r' (emb i)).seen)

/-- **`AcceptsFramed` at one session**: on a statement in scope, a prover that says what the framing fixes outside the
session is accepted when the session, at coins that check some unit, accepts. -/
def AcceptsOne (H : Msg → Msg) : Prop :=
  ∀ x {Ω : Type} (P : Prover Ω) (ω : Ω) (r : Fin (c.V.k x) → FlockDeploy.Coin),
    (∃ s Pt, c.parse x = some s ∧ c.partition s = some Pt ∧ InScope Pt s) →
    (∀ hist m, c.step x hist = .say m → ((P.next x hist ω).eval H).1 = m) →
    ((code c reg L bytes sr sess).checked x r ≠ [] →
      ∀ xS vS, sess x (run H P c.V x ω r).seen = some (xS, vS) → vS.out = .acc) →
      (run H P c.V x ω r).out = .acc

/-- **`SessionsScope` at one session**: on a statement in scope, at coins that check some unit, the session's statement
passes the deployed `--zk` scope check. -/
def SessScope (H : Msg → Msg) : Prop :=
  ∀ x s Pt, c.parse x = some s → c.partition s = some Pt → InScope Pt s →
    ∀ {Ω : Type} (P : Prover Ω) (ω : Ω) (r : Fin (c.V.k x) → FlockDeploy.Coin) xS vS,
      (code c reg L bytes sr sess).checked x r ≠ [] →
      sess x (run H P c.V x ω r).seen = some (xS, vS) → FlockDeploy.scopeCheck true xS = true

/-- **`SessionWit` at one session**: on a statement in scope with no faulty unit, at coins that check some unit,
openings that pass the commitment's check, are the committed values and cover what the session reads under its
statement `xS` encode (`wit`) a witness in `xS`'s relation. -/
def SessWit (H : Msg → Msg) : Prop :=
  ∀ x s Pt w, c.parse x = some s → c.partition s = some Pt → InScope Pt s →
    faulty H (commit c) s Pt w = ∅ →
    ∀ {Ω : Type} (P : Prover Ω) (ω : Ω) (r : Fin (c.V.k x) → FlockDeploy.Coin) xS vS,
      (code c reg L bytes sr sess).checked x r ≠ [] →
      sess x (run H P c.V x ω r).seen = some (xS, vS) →
      ∀ os : List (Opening Bool),
        (∀ o ∈ os, (commit c).Opens H s.roots o = true ∧ w.val H (commit c) s o.pos = some o.val) →
        (∀ g ∈ c.reads s xS, ∀ b, w.val H (commit c) s g = some b → ∃ o ∈ os, o.pos = g) →
          Claims.Flock.R (FlockDeploy.deployZk rel).parse xS (c.wit s xS os)

/-- **`SessionsIn` at one session**: whatever a run shows, the session's statement is in Flock's masked scope with at
most `tables` tables, and its `N₀` is the record's, so every session at `x` has one `N₀`. -/
def SessIn : Prop :=
  ∀ x v xS vS, sess x v = some (xS, vS) →
    (∃ s spec sts, FlockDeploy.ZkIn xS s spec sts ∧ sts.size ≤ c.tables x) ∧ FlockDeploy.N₀ true xS = c.innerN₀ x

theorem framed_code_iff (H : Msg → Msg) : Framed (code c reg L bytes sr sess) H ↔ FramedOne c reg L bytes sr sess H := by
  constructor
  · intro h x Ω P ω r hne
    obtain ⟨u, hu⟩ := List.exists_mem_of_ne_nil _ hne
    exact h x P ω r u hu
  · intro h x Ω P ω r u hu
    exact h x P ω r (List.ne_nil_of_mem hu)

theorem acceptsFramed_code_iff (H : Msg → Msg) :
    AcceptsFramed (code c reg L bytes sr sess) H ↔ AcceptsOne c reg L bytes sr sess H := by
  constructor
  · intro h x Ω P ω r hx hsay hs
    refine h x P ω r hx hsay fun u hu xS vS hv => hs (List.ne_nil_of_mem hu) xS vS hv
  · intro h x Ω P ω r hx hsay hs
    refine h x P ω r hx hsay fun hne xS vS hv => ?_
    obtain ⟨u, hu⟩ := List.exists_mem_of_ne_nil _ hne
    exact hs u hu xS vS hv

theorem sessionsScope_code_iff (H : Msg → Msg) :
    SessionsScope (code c reg L bytes sr sess) H ↔ SessScope c reg L bytes sr sess H := by
  constructor
  · intro h x s Pt hx hP hin Ω P ω r xS vS hne hv
    obtain ⟨u, hu⟩ := List.exists_mem_of_ne_nil _ hne
    exact h x s Pt hx hP hin P ω r u xS vS hu hv
  · intro h x s Pt hx hP hin Ω P ω r u xS vS hu hv
    exact h x s Pt hx hP hin P ω r xS vS (List.ne_nil_of_mem hu) hv

theorem sessionWit_code_iff (H : Msg → Msg) :
    SessionWit rel (code c reg L bytes sr sess) H ↔ SessWit rel c reg L bytes sr sess H := by
  constructor
  · intro h x s Pt w hx hP hin hf Ω P ω r xS vS hne hv
    obtain ⟨u, hu⟩ := List.exists_mem_of_ne_nil _ hne
    exact h x s Pt w hx hP hin hf P ω r u xS vS hu hv
  · intro h x s Pt w hx hP hin hf Ω P ω r u xS vS hu hv
    exact h x s Pt w hx hP hin hf P ω r xS vS (List.ne_nil_of_mem hu) hv

theorem sessionsIn_code_iff : SessionsIn (code c reg L bytes sr sess) ↔ SessIn c sess := by
  constructor
  · intro h x v xS vS hv
    exact h x (0, 0) v xS vS hv
  · intro h x _ v xS vS hv
    exact h x v xS vS hv

/-! ## The sessions' table shapes -/

/-- **The session's table shapes are the statement's**: whatever a run shows, the session's statement sets up tables of
sizes `shape x`, in order. The sampled verifier's registration refusal (`Run.countVaries`: `shapes`) fixes the drawn
count only, and its outcome step never reads the session's tables, so nothing in the Lean verifier establishes this at
`sess`; at the session the shape check lets through it holds (`shapesFixed_checked`). -/
def ShapesFixed (shape : Msg → List ℕ) : Prop :=
  ∀ x v xS vS, sess x v = some (xS, vS) → (FlockDeploy.tables true xS).map (fun p => p.2.m) = shape x

/-- A shape's `N₀`: its tables' masked level-0 codeword lengths, summed. -/
noncomputable def shapeN₀ (ms : List ℕ) : ℝ := (ms.map fun m => (2 : ℝ) ^ (Claims.Flock.zkLevel0 m).d).sum

/-- **`N₀` reads only the table shapes**: at fixed shapes, every session at `x` has `N₀` the shapes'. -/
theorem N₀_of_shapes (shape : Msg → List ℕ) (h : ShapesFixed sess shape) (x : Msg) (v : Seen) (xS : Msg) (vS : Seen)
    (hv : sess x v = some (xS, vS)) : FlockDeploy.N₀ true xS = shapeN₀ (shape x) := by
  rw [← h x v xS vS hv, shapeN₀, List.map_map]
  rfl

/-- **`SessionsIn` from fixed shapes**: the sessions in Flock's masked scope with at most `tables` tables, their shapes
the statement's, and the record's `N₀` the shapes'. -/
theorem sessIn_of_shapes (shape : Msg → List ℕ)
    (hin : ∀ x v xS vS, sess x v = some (xS, vS) → ∃ s spec sts, FlockDeploy.ZkIn xS s spec sts ∧ sts.size ≤ c.tables x)
    (h : ShapesFixed sess shape) (hN : ∀ x, c.innerN₀ x = shapeN₀ (shape x)) : SessIn c sess :=
  fun x v xS vS hv => ⟨hin x v xS vS hv, (N₀_of_shapes sess shape h x v xS vS hv).trans (hN x).symm⟩

/-- **The shape check**: `sess` with every session refused whose tables aren't `shape x`, what an outcome step that held
the session's tables to the shapes its registration fixes would run. -/
def shapeChecked (shape : Msg → List ℕ) (x : Msg) (v : Seen) : Option (Msg × Seen) :=
  (sess x v).filter fun p => decide ((FlockDeploy.tables true p.1).map (fun q => q.2.m) = shape x)

theorem shapeChecked_some (shape : Msg → List ℕ) {x : Msg} {v : Seen} {p : Msg × Seen}
    (h : shapeChecked sess shape x v = some p) :
    sess x v = some p ∧ (FlockDeploy.tables true p.1).map (fun q => q.2.m) = shape x := by
  unfold shapeChecked at h
  cases hs : sess x v with
  | none => rw [hs] at h; cases h
  | some q =>
    simp only [hs, Option.filter_some] at h
    split at h
    · rename_i hq
      cases h
      exact ⟨rfl, of_decide_eq_true hq⟩
    · cases h

/-- **The shape check fixes the shapes**: every session it lets through sets up the tables `shape x`. -/
theorem shapesFixed_checked (shape : Msg → List ℕ) : ShapesFixed (shapeChecked sess shape) shape :=
  fun _ _ _ _ hv => (shapeChecked_some sess shape hv).2

/-- **Fixed shapes are a shape check that refuses nothing**: `sess`'s sessions have the tables `shape x` exactly when
the shape check lets each of them through. -/
theorem shapesFixed_iff_checked (shape : Msg → List ℕ) :
    ShapesFixed sess shape ↔ shapeChecked sess shape = sess := by
  constructor
  · intro h
    funext x v
    unfold shapeChecked
    cases hs : sess x v with
    | none => rfl
    | some p =>
      obtain ⟨xS, vS⟩ := p
      simp only [Option.filter_some, decide_eq_true (h x v xS vS hs), ite_true]
  · intro h
    rw [← h]
    exact shapesFixed_checked sess shape

/-- **`SessionsIn` at the shape check**: with `sess`'s sessions in Flock's masked scope and the record's `N₀` the
shapes', the session the shape check lets through is in `SessIn`. -/
theorem sessIn_checked (shape : Msg → List ℕ)
    (hin : ∀ x v xS vS, sess x v = some (xS, vS) → ∃ s spec sts, FlockDeploy.ZkIn xS s spec sts ∧ sts.size ≤ c.tables x)
    (hN : ∀ x, c.innerN₀ x = shapeN₀ (shape x)) : SessIn c (shapeChecked sess shape) :=
  sessIn_of_shapes c (shapeChecked sess shape) shape
    (fun x v xS vS hv => hin x v xS vS (shapeChecked_some sess shape hv).1) (shapesFixed_checked sess shape) hN

set_option synthInstance.maxSize 1024 in
/-- **Sampling completeness at the code**: `complete_deploy` with `SessionSlots` proved, `DrawSlots` and `DrawFixed`
from the draw's facts, and the other premises in their one-session form. -/
theorem complete_code (H : Msg → Msg) (hF : Claims.Flock.Complete H (FlockDeploy.deployZk rel))
    (hfixed : BytesFixed c L bytes sr) (hreg : RegMatches c reg) (hunif : BytesUniform c L bytes)
    (hframe : FramedOne c reg L bytes sr sess H) (haccept : AcceptsOne c reg L bytes sr sess H)
    (hscope : SessScope c reg L bytes sr sess H) (hwit : SessWit rel c reg L bytes sr sess H) :
    Complete H (deploy rel (code c reg L bytes sr sess)) :=
  complete_deploy rel _ H hF (drawFixed_code rel c reg L bytes sr sess hfixed)
    (drawSlots_code rel c reg L bytes sr sess hreg hunif) ((framed_code_iff c reg L bytes sr sess H).2 hframe)
    ((acceptsFramed_code_iff c reg L bytes sr sess H).2 haccept)
    ((sessionsScope_code_iff c reg L bytes sr sess H).2 hscope)
    ((sessionWit_code_iff rel c reg L bytes sr sess H).2 hwit) (sessionSlots_code c reg L bytes sr sess H)

end SamplingDeploy
