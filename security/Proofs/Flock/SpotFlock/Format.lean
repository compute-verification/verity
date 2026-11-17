module

public import Proofs.Flock.Bridge.Members
public import Proofs.Flock.SpotFlock.Opens

/-! Where the two-stage session holds each checked unit's values: `Decodes` at a session layout's decoder from `Embeds`
(`decodes_of_embeds`). -/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

/-- **A session layout**: the Flock block holding each checked unit (`blockOf x m₁ xS u`, at the run's statement `x`,
the interior registration `m₁` and the session statement `xS`), and the gate of the session's block circuit holding
each program gate's value for that unit there (`wire x m₁ xS u g`). -/
structure Layout where
  blockOf : Msg → Msg → Msg → ℕ → Option ℕ
  wire : Msg → Msg → Msg → ℕ → ℕ → ℕ

def valueAt (S : Claims.Flock.Stmt) (β : ℕ) (w : Claims.Flock.Wit) (j : ℕ) : Bool :=
  if hj : j < S.C.N then
    (S.C.toCircuit.eval (fun g => BitVec.ofBool (Claims.Flock.val S w (S.inp β g.val))) ⟨j, hj⟩).getLsbD 0
  else false

def decOf (L : Layout) : Dec := fun x m₁ xS wF u g =>
  match FlockDeploy.parse true _root_.Flock.Readable.relation xS, L.blockOf x m₁ xS u with
  | some S, some β => valueAt S β wF (L.wire x m₁ xS u g)
  | _, _ => false

theorem decOf_eq {L : Layout} {x m₁ xS : Msg} {S : Claims.Flock.Stmt} {u β : ℕ}
    (hS : FlockDeploy.parse true _root_.Flock.Readable.relation xS = some S) (hβ : L.blockOf x m₁ xS u = some β)
    (w : Claims.Flock.Wit) (g : ℕ) : decOf L x m₁ xS w u g = valueAt S β w (L.wire x m₁ xS u g) := by
  simp only [decOf, hS, hβ]

theorem valueAt_input {S : Claims.Flock.Stmt} {β j : ℕ} (w : Claims.Flock.Wit) (hj : j < S.C.N)
    (hin : (S.C.op ⟨j, hj⟩).isInput = true) : valueAt S β w j = Claims.Flock.val S w (S.inp β j) := by
  have he := Verity.Circuit.eval_input S.C.toCircuit
    (fun g => BitVec.ofBool (Claims.Flock.val S w (S.inp β g.val))) (g := ⟨j, hj⟩) hin
  simp only [valueAt, hj, ↓reduceDIte]
  exact (congrArg (fun v => BitVec.getLsbD v 0) he).trans
    ((fun b => by cases b <;> rfl : ∀ b : Bool, (BitVec.ofBool b).getLsbD 0 = b) _)

theorem valueAt_output {S : Claims.Flock.Stmt} {β j : ℕ} {w : Claims.Flock.Wit} (hR : Claims.Flock.RelS S w)
    (hβ : β < S.nBlocks) (hj : j < S.C.N) (ho : (⟨j, hj⟩ : Fin S.C.N) ∈ S.C.outputs) :
    valueAt S β w j = Claims.Flock.val S w (S.out β j) := by
  simp only [valueAt, hj, ↓reduceDIte]
  exact hR β hβ ⟨j, hj⟩ ho

theorem decOf_input {L : Layout} {x m₁ xS : Msg} {S : Claims.Flock.Stmt} {u β g : ℕ}
    (hS : FlockDeploy.parse true _root_.Flock.Readable.relation xS = some S) (hβ : L.blockOf x m₁ xS u = some β)
    (w : Claims.Flock.Wit) (hj : L.wire x m₁ xS u g < S.C.N)
    (hin : (S.C.op ⟨L.wire x m₁ xS u g, hj⟩).isInput = true) :
    decOf L x m₁ xS w u g = Claims.Flock.val S w (S.inp β (L.wire x m₁ xS u g)) := by
  rw [decOf_eq hS hβ, valueAt_input w hj hin]

theorem decOf_output {L : Layout} {x m₁ xS : Msg} {S : Claims.Flock.Stmt} {u β g : ℕ} {w : Claims.Flock.Wit}
    (hS : FlockDeploy.parse true _root_.Flock.Readable.relation xS = some S) (hβ : L.blockOf x m₁ xS u = some β)
    (hck : Claims.Flock.check (FlockDeploy.parse true _root_.Flock.Readable.relation) xS w = true)
    (hβS : β < S.nBlocks) (hj : L.wire x m₁ xS u g < S.C.N)
    (ho : (⟨L.wire x m₁ xS u g, hj⟩ : Fin S.C.N) ∈ S.C.outputs) :
    decOf L x m₁ xS w u g = Claims.Flock.val S w (S.out β (L.wire x m₁ xS u g)) := by
  have hR : Claims.Flock.RelS S w := by simpa [Claims.Flock.check, hS] using hck
  rw [decOf_eq hS hβ, valueAt_output hR hβS hj ho]

def unitVals (L : Layout) (x m₁ xS : Msg) (S : Claims.Flock.Stmt) (β : ℕ) (w : Claims.Flock.Wit) (u o : ℕ)
    (C' : Verity.Boolean.Circuit) (a : Fin C'.N) : Bool :=
  valueAt S β w (L.wire x m₁ xS u (o + _root_.Flock.Bridge.posOf C' a))

variable (h : Held)

/-- These are `C'`'s gates, not the lowered program's: no wire of `C` holds a row's two XORs, which `Prog.Correct` never
reads. It holds of every accepted list and turns only on the registration, as the verifier's check fixes the unit's
session statement's circuit to its template's reference statement's. -/
def Embeds (L : Layout) : Prop :=
  ∀ x s Pt d (m₁ m₂ : Msg) (c₀ : FlockDeploy.Coin) (c₁ : Stage h) u b ρ rb S,
    parse h x = some s → partition h s = some Pt → two h s = some d → InScopeTwo Pt d s →
    stmtOk h x m₁ m₂ c₀ c₁ = true → u ∈ h.reg.checked (units (base h) x) (src c₀) (stageSrc c₁) →
    FlockDeploy.parse true _root_.Flock.Readable.relation (unitStmt h m₂ u) = some S →
    Pt.blocks[u]? = some b → replayOf Pt d.part u = some ρ → d.part.blocks[ρ]? = some rb →
    ∃ (C' : Verity.Boolean.Circuit) (κ : Fin C'.N → Option Bool) (n : ℕ)
      (P : FlockSoundness.Audit.Partition C' n) (v : Fin n) (o : ℕ),
      _root_.Flock.Bridge.Members.SitsAt s.prog o (_root_.Flock.Bridge.progOf C' κ) ∧
      b = (_root_.Flock.Bridge.blockOf P κ v).map (o + ·) ∧
      ∃ β, L.blockOf x m₁ (unitStmt h m₂ u) u = some β ∧ ∀ w : Claims.Flock.Wit, Claims.Flock.RelS S w →
        (∀ a, P.unit a = some v → unitVals L x m₁ (unitStmt h m₂ u) S β w u o C' a =
          (C'.op a).apply (unitVals L x m₁ (unitStmt h m₂ u) S β w u o C')) ∧
        ∀ a c, κ a = some c → (∃ g, P.unit g = some v ∧ a ∈ (C'.op g).args) →
          unitVals L x m₁ (unitStmt h m₂ u) S β w u o C' a = c

theorem decodes_of_embeds (L : Layout) (he : Embeds h L) : Decodes h (decOf L) := by
  intro x s Pt d m₁ m₂ c₀ c₁ wF u b ρ rb hx hP h2 hin hok hu hck hb hρ hrb
  cases hS : FlockDeploy.parse true _root_.Flock.Readable.relation (unitStmt h m₂ u) with
  | none => simp [Claims.Flock.check, hS] at hck
  | some S =>
    obtain ⟨C', κ, n, P, v, o, hsit, rfl, β, hβ, hw⟩ :=
      he x s Pt d m₁ m₂ c₀ c₁ u b ρ rb S hx hP h2 hin hok hu hS hb hρ hrb
    obtain ⟨hV, hκ⟩ := hw wF (by simpa [Claims.Flock.check, hS] using hck)
    have href : s.prog.refIn rb (fun g => some (decOf L x m₁ (unitStmt h m₂ u) wF u g))
        (decOf L x m₁ (unitStmt h m₂ u) wF u) = fun g => some (decOf L x m₁ (unitStmt h m₂ u) wF u g) :=
      funext fun g => by simp only [Prog.refIn, ite_self]
    rw [href]
    refine hsit.correct _root_.Flock.Bridge.Members.topological_progOf (fun p hp => ?_) ?_
    · rcases _root_.Flock.Bridge.mem_blockOf.1 hp with hp | ⟨a, -, rfl⟩
      · exact _root_.Flock.Bridge.Members.lt_length_of_mem_unitGates hp
      · exact _root_.Flock.Bridge.Members.posOf_lt_length a
    · refine (_root_.Flock.Bridge.correct_blockOf P v _).2
        ⟨_root_.Flock.Bridge.correct_unitGates_of P v _ hV hκ _ fun a _ => ?_, fun a _ => by simp⟩
      exact congrArg some (decOf_eq hS hβ wF _)

theorem sessionOpensTwo_of_embeds (L : Layout) (he : Embeds h L) (hr : Reveals h (decOf L)) :
    Claims.SamplingGeneric.SessionOpensTwo FlockDeploy.sha512 (deploy _root_.Flock.Readable.relation (code h))
      (two h) iroots :=
  sessionOpensTwo_of_dec h (decOf L) (decodes_of_embeds h L he) hr

end SpotFlock
