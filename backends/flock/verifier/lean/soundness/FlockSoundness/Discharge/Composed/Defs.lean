import FlockSoundness.Discharge.ZkLink
import FlockSoundness.Discharge.ZkExec
import FlockSoundness.Discharge.ZkView

/-!
# One `--zk` sampled session: the pieces the composed headline reads

* `planOf`, `Gz`, `outz`, `accz`: the compiled `--zk` table at each draw (`tablesAtZK`, one table) as `modelTable`'s pieces,
  so that `ZkExec.sessionZC` at them is `ZkLink.batchedSessionZC` at the one-table plan (`sessionZC_planOf`).
* `simZ`: the audit strategy the live coin server's simulation builds from a live prover (`simAuditZK`), in the compiled
  batched session the count curve reads.
* `HashDerivedKey`: `hash-derived-key` (ruling F), the pinned hm96-sha512 key's hiding gap at most `δ₁`.

Open pieces, each a `Prop` the headline takes until its lane proves it:

* `RunsZCAssembly` (item 4): with `TablesAtZKLive` (item 3) and the coin server's record and draws, the decoder is right
  on every session flock-verify `--zk` accepts (`DecodesZK`) for the compiled tables `tablesAtZK`.
* `PlacementZK` (item 9) and `ValueBindingZKLive` (item 8): **stubs**, `True` until lp-zkbind freezes them. They are what
  makes the count's "wrong unit" a statement about the real units: each placement is the unit's place in setupH's
  statement, and the value binding binds the committed values to the unit's real inputs and outputs.
-/

namespace FlockSoundness.Discharge.Composed

open Game Audit Exec Refine Flock Model ZkSession ZkLink ZkExec

local notation "𝔽" => FlockLevel3.GF128

variable {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] {mPts n : ℕ} {Reg : Type}

/-- The one-table plan at the draw's tables. -/
abbrev planOf (T : Finset (Fin n) → Reg → TabZK 𝔽 mPts) : Finset (Fin n) → Reg → List (TabZK 𝔽 mPts) :=
  fun S R => [T S R]

/-- Every unit is read from the plan's one table. -/
def tabOne (T : Finset (Fin n) → Reg → TabZK 𝔽 mPts) : (S : Finset (Fin n)) → (R : Reg) → Fin n →
    Fin (planOf T S R).length :=
  fun _ _ _ => ⟨0, Nat.zero_lt_one⟩

/-- The compiled reps at each draw. -/
noncomputable def Gz (A : Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (T : Finset (Fin n) → Reg → TabZK 𝔽 mPts) :
    ∀ (_ : Finset (Fin n)) (_ : Reg), (Fin 2 → Fin mPts → 𝔽) → Fin 2 → Game (PendZ 𝔽 D Sl) :=
  fun S R => repZCAt A Hs (T S R)

/-- The compiled table's output at each draw. -/
noncomputable def outz (Hs : Hashes 𝔽 D Sl) (T : Finset (Fin n) → Reg → TabZK 𝔽 mPts) :
    ∀ (_ : Finset (Fin n)) (_ : Reg), (Fin 1 → Cap D) → PendZ 𝔽 D Sl × PendZ 𝔽 D Sl → OpensZ 𝔽 D Sl →
      List (OutZC 𝔽 D Sl) :=
  fun S R caps pend os => [outZC Hs (T S R) (caps 0) pend os]

variable [Inhabited D] [Inhabited Sl]

open Classical in
/-- The compiled table's verdict at each draw. -/
noncomputable def accz (T : Finset (Fin n) → Reg → TabZK 𝔽 mPts) :
    ∀ (_ : Finset (Fin n)) (_ : Reg), List (OutZC 𝔽 D Sl) → Bool :=
  fun S R outs => decide (acceptedZC [T S R] outs)

omit [Fintype K] in
theorem sessionZC_planOf (A : Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (T : Finset (Fin n) → Reg → TabZK 𝔽 mPts) :
    ZkExec.sessionZC (Gz A Hs T) (outz Hs T) (accz T) = batchedSessionZC A Hs (planOf T) :=
  funext fun S => funext fun R => batchedSessionZC_one A Hs T S R

/-- **The simulated audit strategy in the compiled batched session**: `simAuditZK`'s, carried along
`sessionZC_planOf`. -/
noncomputable def simZ (A : Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (T : Finset (Fin n) → Reg → TabZK 𝔽 mPts) (I : Inputs)
    (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript) (opn : ∀ (_ : Finset (Fin n)) (_ : Reg),
      Array ByteArray → OpensZ 𝔽 D Sl) (dg : ByteArray → ByteArray)
    (dec : Finset (Fin n) → Reg → ℕ → ByteArray → List Msg) (one : Finset (Fin n) → Reg → ℕ → Bool)
    (fin : Finset (Fin n) → Reg → Fin 2 → Array ByteArray → List Msg)
    (capOf : ∀ (_ : Finset (Fin n)) (_ : Reg), ByteArray → Fin 1 → Cap D) (N : ℕ) {L : Law n}
    (hdec : DecodesZK (Gz A Hs T) (outz Hs T) (accz T) I wr opn dec one fin capOf)
    (τ : Strategy (auditLive L Reg (liveOf dg mPts N wr))) : Strategy (audit L Reg (batchedSessionZC A Hs (planOf T))) :=
  cast (congrArg (fun g => Strategy (audit L Reg g)) (sessionZC_planOf A Hs T))
    (simAuditZK (Gz A Hs T) (outz Hs T) (accz T) I wr opn dg dec one fin capOf N
      (fun S R pts r => allInh_repZCAt A Hs (T S R) pts r) hdec τ)

theorem prob_audit_cast {L : Law n} {g₁ g₂ : Finset (Fin n) → Reg → Game Bool} (h : g₁ = g₂)
    (σ : Strategy (audit L Reg g₁)) (E : Bool × Finset (Fin n) → Prop) :
    prob E (audit L Reg g₂) (cast (congrArg (fun g => Strategy (audit L Reg g)) h) σ) =
      prob E (audit L Reg g₁) σ := by
  subst h
  rfl

/-! ## The open pieces -/

/-- **`RunsZC` assembled** (item 4, open): on public-output one-table inputs with untyped tags, given `TablesAtZKLive`
(item 3), the coin server's record of the live session (`RecordsLiveZK`) and its draws (`DrawOkZK`), the decoder is
right on every session flock-verify `--zk` accepts, for the compiled tables `tablesAtZK` (`DecodesZK`): each rep's
`RepRun` on its stream and proof file, and the table's verdict (`accZ` at both reps' results) on them. -/
def RunsZCAssembly (A : Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (I : Inputs) (dj : Finset (Fin n) → Lean.Json)
    {S₀ : Finset (Fin n)} (y₀ : DrawSetupZK I mPts S₀ (dj S₀))
    (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript)
    (opn : ∀ (_ : Finset (Fin n)) (_ : Reg), Array ByteArray → OpensZ 𝔽 D Sl)
    (dec : Finset (Fin n) → Reg → ℕ → ByteArray → List Msg) (one : Finset (Fin n) → Reg → ℕ → Bool)
    (fin : Finset (Fin n) → Reg → Fin 2 → Array ByteArray → List Msg)
    (capOf : ∀ (_ : Finset (Fin n)) (_ : Reg), ByteArray → Fin 1 → Cap D) : Prop :=
  I.tags.typed = false → I.tags.hiddenOutputs = false → I.count ≤ 1 →
    TablesAtZKLive (n := n) (mPts := mPts) (Reg := Reg) → RecordsLiveZK I dj wr → (∀ S, DrawOkZK I dj mPts S) →
      DecodesZK (Gz A Hs (tablesAtZK (Reg := Reg) dj y₀)) (outz Hs (tablesAtZK (Reg := Reg) dj y₀))
        (accz (tablesAtZK (Reg := Reg) dj y₀)) I wr opn dec one fin capOf

/-- **Stub** (item 9, lp-zkbind): each unit's placement is its place in setupH's statement at the draw. `True` until
lp-zkbind freezes the statement. -/
def PlacementZK {C : Audit.Circuit} (P : Partition C n) (T : Finset (Fin n) → Reg → TabZK 𝔽 mPts)
    (_place : ∀ S R u, u ∈ S → UnitPlace P (tabOfZ (planOf T) (tabOne T) S R u).S u) : Prop :=
  True

/-- **Stub** (item 8, lp-zkbind): the binding's committed values are the unit's real inputs and outputs. `True` until
lp-zkbind freezes the statement. -/
def ValueBindingZKLive (A : Arith 𝔽 K) {C : Audit.Circuit} (P : Partition C n) (T : Finset (Fin n) → Reg → TabZK 𝔽 mPts)
    {decode : DecoderZK (C := C) (planOf T) (tabOne T)} {Dh : Type} {Hc : List UInt8 → Dh}
    (_vb : ValueBindingZ A P (planOf T) (tabOne T) decode Hc) : Prop :=
  True

/-! ## The assumption -/

/-- **`hash-derived-key`** (ruling F; `Assumptions.Hm96Hiding`'s common-reference-string step): the pinned hm96-sha512
key's hiding gap, the statistical distance of `(M(κ₀)·y, SHA-512(sp ‖ y))` from `(U, SHA-512(sp ‖ y))`, is at most
`δ₁` (hm96 §3: `2^-193` for all but `2^-64` of keys). -/
def HashDerivedKey (δ₁ : ℝ) : Prop := CROnly.hidingGap ZkView.My ZkView.saltHash ≤ δ₁

end FlockSoundness.Discharge.Composed
