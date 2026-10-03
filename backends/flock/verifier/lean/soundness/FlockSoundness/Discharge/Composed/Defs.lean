import FlockSoundness.Discharge.ZkLink
import FlockSoundness.Discharge.ZkExec
import FlockSoundness.Discharge.ZkView
import FlockSoundness.Discharge.ZkBind

/-!
# One `--zk` sampled session: the pieces the composed headline reads

* `Gz`, `outz`, `accz`: the compiled `--zk` table at each draw (one table) as `modelTable`'s pieces, so that
  `ZkExec.sessionZC` at them is `ZkLink.batchedSessionZC` at the one-table plan (`sessionZC_planOf`).
* `DecodesOnZK`, `simZ`, `execRefinesZK_on`: `DecodesZK`, `simAuditZK` and `execRefinesZK_one` at the law's draws only
  (as `Tables.DecodesOnJ` is on the public path), the simulated strategy carried into the compiled batched session.
* `RunsZCAssembly` (item 4, open): with `TablesAtZKLive` (item 3), the coin server's record and its draw at each of the
  law's draws, the decoder is right at the law's draws (`DecodesOnZK`).
* `CoinedSetup`, `coinedOf`, `CoinedOK`, `CoinedOfSetup` (new item, open): the `--zk` table of a draw as `ZkView`'s
  `Coined` tables at the executable's leaves, at each coin vector, and the structural facts Lemma B reads of them.
-/

namespace FlockSoundness.Discharge.Composed

open Game Audit Exec Refine Flock Model ZkSession ZkLink ZkExec
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

variable {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] {mPts n : ℕ} {Reg : Type}

/-- The one-table plan at the draw's tables. -/
abbrev planOf (T : Finset (Fin n) → Reg → TabZK 𝔽 mPts) : Finset (Fin n) → Reg → List (TabZK 𝔽 mPts) :=
  fun S R => [T S R]

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

section On

variable (A : Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (T : Finset (Fin n) → Reg → TabZK 𝔽 mPts) (I : Inputs)
  (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript)
  (opn : ∀ (_ : Finset (Fin n)) (_ : Reg), Array ByteArray → OpensZ 𝔽 D Sl) (dg : ByteArray → ByteArray)
  (dec : Finset (Fin n) → Reg → ℕ → ByteArray → List Msg) (one : Finset (Fin n) → Reg → ℕ → Bool)
  (fin : Finset (Fin n) → Reg → Fin 2 → Array ByteArray → List Msg)
  (capOf : ∀ (_ : Finset (Fin n)) (_ : Reg), ByteArray → Fin 1 → Cap D) (N : ℕ) (L : Law n)

/-- **The decoder is right at the law's draws** (`DecodesZK` at `L.draw ω` only): on every live outcome flock-verify
`--zk` accepts at a draw the law makes, each rep's game runs on the tape decoded from its stream and proof file, and
the compiled table's verdict on those runs holds. -/
def DecodesOnZK : Prop :=
  ∀ ω R, Decodes (outz Hs T (L.draw ω) R) (fun o => AcceptsZK I (wr (L.draw ω) R o))
    (fun x => accz T (L.draw ω) R x = true) (opn (L.draw ω) R) (dec (L.draw ω) R) (one (L.draw ω) R)
    (fin (L.draw ω) R) (capOf (L.draw ω) R) (Gz A Hs T (L.draw ω) R)

variable {L}

/-- `simAuditZK` from the decoder at the law's draws. -/
noncomputable def simAuditZKOn (hdec : DecodesOnZK A Hs T I wr opn dec one fin capOf L)
    (τ : Strategy (auditLive L Reg (liveOf dg mPts N wr))) :
    Strategy (audit L Reg (ZkExec.sessionZC (Gz A Hs T) (outz Hs T) (accz T))) :=
  ⟨τ.1, fun ω => Strategy.map (fun ok => (ok, L.draw ω)) _ (Strategy.map _ _
    ((simTable (outz Hs T (L.draw ω) τ.1) (fun o => AcceptsZK I (wr (L.draw ω) τ.1 o))
      (fun x => accz T (L.draw ω) τ.1 x = true) (opn (L.draw ω) τ.1) dg (dec (L.draw ω) τ.1) (one (L.draw ω) τ.1)
      (fin (L.draw ω) τ.1) (capOf (L.draw ω) τ.1) N (Gz A Hs T (L.draw ω) τ.1)
      (fun pts r => allInh_repZCAt A Hs (T (L.draw ω) τ.1) pts r) (hdec ω τ.1)).strategy
      (Strategy.ofMap (wr (L.draw ω) τ.1) _
        (Strategy.ofMap (fun t => (τ.1, L.draw ω, t)) _ (τ.2 ω)))))⟩

/-- **The simulated audit strategy in the compiled batched session**: `simAuditZKOn`'s, carried along
`sessionZC_planOf`. -/
noncomputable def simZ (hdec : DecodesOnZK A Hs T I wr opn dec one fin capOf L)
    (τ : Strategy (auditLive L Reg (liveOf dg mPts N wr))) : Strategy (audit L Reg (batchedSessionZC A Hs (planOf T))) :=
  cast (congrArg (fun g => Strategy (audit L Reg g)) (sessionZC_planOf A Hs T))
    (simAuditZKOn A Hs T I wr opn dg dec one fin capOf N hdec τ)

/-- **R11 at `--zk` from the decoder at the law's draws** (`execRefinesZK_one`'s proof, reading `hdec` only there). -/
theorem execRefinesZK_on (hdec : DecodesOnZK A Hs T I wr opn dec one fin capOf L)
    (τ : Strategy (auditLive L Reg (liveOf dg mPts N wr))) :
    ExecRefines (auditLive L Reg (liveOf dg mPts N wr)) τ (LiveAcceptsZK I) (fun _ => False)
      (audit L Reg (ZkExec.sessionZC (Gz A Hs T) (outz Hs T) (accz T)))
      (simAuditZKOn A Hs T I wr opn dg dec one fin capOf N hdec τ) := by
  have h0 : prob (fun _ => False) (auditLive L Reg (liveOf dg mPts N wr)) τ = 0 :=
    prob_eq_zero_of_out (fun _ _ h => h) τ
  unfold ExecRefines
  rw [h0, add_zero]
  show ∑ ω : L.Ω, (Fintype.card L.Ω : ℝ≥0∞)⁻¹ *
      prob (LiveAcceptsZK I) ((liveOf dg mPts N wr (L.draw ω) τ.1).map fun t => (τ.1, L.draw ω, t)) (τ.2 ω) ≤
    ∑ ω : L.Ω, (Fintype.card L.Ω : ℝ≥0∞)⁻¹ *
      prob (fun o => o.1 = true)
        ((ZkExec.sessionZC (Gz A Hs T) (outz Hs T) (accz T) (L.draw ω) τ.1).map fun ok => (ok, L.draw ω))
        ((simAuditZKOn A Hs T I wr opn dg dec one fin capOf N hdec τ).2 ω)
  refine Finset.sum_le_sum fun ω _ => mul_le_mul_right ?_ _
  unfold simAuditZKOn ZkExec.sessionZC
  rw [prob_strategy_map, prob_strategy_map, prob_map]
  unfold liveOf
  rw [prob_map]
  exact live_le (outz Hs T (L.draw ω) τ.1) _ (fun x => accz T (L.draw ω) τ.1 x = true) (opn (L.draw ω) τ.1) dg
    (dec (L.draw ω) τ.1) (one (L.draw ω) τ.1) (fin (L.draw ω) τ.1) (capOf (L.draw ω) τ.1) N (Gz A Hs T (L.draw ω) τ.1)
    (fun pts r => allInh_repZCAt A Hs (T (L.draw ω) τ.1) pts r) (hdec ω τ.1) _

/-- **The `--zk` headline at flock-verify's acceptance, from the decoder at the law's draws**: a bound on the compiled
batched audit's acceptance with any event `Q` at `simZ … τ` bounds flock-verify `--zk`'s acceptance with `Q`. -/
theorem zk_headline_exec_on (hdec : DecodesOnZK A Hs T I wr opn dec one fin capOf L)
    (τ : Strategy (auditLive L Reg (liveOf dg mPts N wr))) (Q : Prop) (B : ℝ≥0∞)
    (hB : prob (fun o => o.1 = true ∧ Q) (audit L Reg (batchedSessionZC A Hs (planOf T)))
      (simZ A Hs T I wr opn dg dec one fin capOf N hdec τ) ≤ B) :
    prob (fun o => LiveAcceptsZK I o ∧ Q) (auditLive L Reg (liveOf dg mPts N wr)) τ ≤ B := by
  have hc : ∀ (g : Finset (Fin n) → Reg → Game Bool)
      (h : ZkExec.sessionZC (Gz A Hs T) (outz Hs T) (accz T) = g),
      prob (fun o => o.1 = true ∧ Q) (audit L Reg g)
          (cast (congrArg (fun g => Strategy (audit L Reg g)) h)
            (simAuditZKOn A Hs T I wr opn dg dec one fin capOf N hdec τ)) =
        prob (fun o => o.1 = true ∧ Q) (audit L Reg (ZkExec.sessionZC (Gz A Hs T) (outz Hs T) (accz T)))
          (simAuditZKOn A Hs T I wr opn dg dec one fin capOf N hdec τ) := by
    intro g h
    subst h
    rfl
  have hB' := (hc _ (sessionZC_planOf A Hs T)).symm.trans_le hB
  have h := exec_le_of_refines (execRefinesZK_on A Hs T I wr opn dg dec one fin capOf N hdec τ) Q hB'
  rwa [prob_eq_zero_of_out (fun _ _ h => h) τ, add_zero] at h

end On

/-! ## The open pieces -/

/-- **`RunsZC` assembled** (item 4, open): on public-output one-table inputs with untyped tags, given `TablesAtZKLive`
(item 3), the coin server's record of the live session (`RecordsLiveZK`) and its draw at each of the law's draws
(`DrawOkZK` at `L.draw ω`), the decoder is right at the law's draws for the compiled tables `tablesAtZK`
(`DecodesOnZK`): each rep's `RepRun` on its stream and proof file, and the table's verdict on them. -/
def RunsZCAssembly (A : Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (I : Inputs) (dj : Finset (Fin n) → Lean.Json)
    {S₀ : Finset (Fin n)} (y₀ : DrawSetupZK I mPts S₀ (dj S₀))
    (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript)
    (opn : ∀ (_ : Finset (Fin n)) (_ : Reg), Array ByteArray → OpensZ 𝔽 D Sl)
    (dec : Finset (Fin n) → Reg → ℕ → ByteArray → List Msg) (one : Finset (Fin n) → Reg → ℕ → Bool)
    (fin : Finset (Fin n) → Reg → Fin 2 → Array ByteArray → List Msg)
    (capOf : ∀ (_ : Finset (Fin n)) (_ : Reg), ByteArray → Fin 1 → Cap D) (L : Law n) : Prop :=
  I.tags.typed = false → I.tags.hiddenOutputs = false → I.count ≤ 1 →
    TablesAtZKLive (n := n) (mPts := mPts) (Reg := Reg) → RecordsLiveZK I dj wr →
      (∀ ω, DrawOkZK I dj mPts (L.draw ω)) →
      DecodesOnZK A Hs (tablesAtZK (Reg := Reg) dj y₀) I wr opn dec one fin capOf L

/-! ## The session's tables at the executable's leaves (new item, open) -/

/-- **A `--zk` table as `ZkView`'s tables**: the witness, hidden-value and value-list types, the coin vectors, the sizes,
the `Coined` table at each coin vector (its leaves the executable's: `Coined.tab`), and M1's level-0 padding data (the
level-0 message's `cols` and the opened positions `pos`) that `padOnto_M1` reads. -/
structure CoinedSetup (K : Type) [Field K] where
  W : Type
  Hi : Type
  VL : Type
  [addHi : AddCommGroup Hi]
  [finHi : Fintype Hi]
  [decHi : DecidableEq Hi]
  Coins : Type
  z : ZkView.Sizes
  T : Coins → ZkView.Coined K W Hi VL z
  cols : ℕ
  pos : Fin z.nS → ℕ

attribute [instance] CoinedSetup.addHi CoinedSetup.finHi CoinedSetup.decHi

/-- **The draw's `--zk` table as `Coined` tables** (open: the definition is the new item's to write, from what
`verify --zk` sets up at the draw: its schedule, pads layout, inner code and the honest prover's level-0 code). -/
noncomputable def coinedOf (Tz : TabZK 𝔽 mPts) : CoinedSetup FlockLevel3.GF256 := sorry

/-- **The structural facts Lemma B reads**, of a `CoinedSetup` at the arithmetic `A` and M1's padding constant `κ`: M1's
level-0 padding geometry (`cols ≤ 40`, `t_pad ≤ 2^cols`, at most `t_pad` distinct opened positions inside
`2^(cols+1)`), and at every coin vector the rank check, M1's padding columns `Gp`, each rep's `W^r` bijective, the pads
onto, `β ≠ 0`, and the inner relation at every witness whose registered values are the public ones. -/
def CoinedOK (A : Arith 𝔽 K) (κ : 𝔽) (C : CoinedSetup K) : Prop :=
  C.cols ≤ 40 ∧ C.z.tpad ≤ 2 ^ C.cols ∧ C.z.nS ≤ C.z.tpad ∧ Function.Injective C.pos ∧
    (∀ q, C.pos q < 2 ^ (C.cols + 1)) ∧
    ∀ c : C.Coins, (C.T c).tab.Rank ∧
      (∀ q i, (C.T c).tab.Gp q i = κ * A.Xhat i.val (C.cols + 1) (A.omega (C.pos q))
        + A.Xhat (2 ^ C.cols + i.val) (C.cols + 1) (A.omega (C.pos q))) ∧
      (∀ r, Function.Bijective fun p : 𝔽 × 𝔽 => (C.T c).tab.Wr r p.1 p.2) ∧ (C.T c).tab.PadsOnto ∧
      (∀ r, (C.T c).tab.β r ≠ 0) ∧
      ∀ w, (C.T c).tab.sReg w = (C.T c).tab.pubReg → (C.T c).tab.InnerHolds w

/-- **The session's tables satisfy Lemma B's facts** (new item, open): at every draw `verify --zk` sets up, the
`Coined` tables of the draw's `--zk` table (`coinedOf (tablesAtZK dj y₀ S R)`) are `CoinedOK` at `execArith` and
`PAD_C`. -/
def CoinedOfSetup (I : Inputs) (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    (y₀ : DrawSetupZK I mPts S₀ (dj S₀)) : Prop :=
  ∀ (S : Finset (Fin n)) (R : Reg), Nonempty (DrawSetupZK I mPts S (dj S)) →
    CoinedOK execArith (fOf Zk.PAD_C) (coinedOf (tablesAtZK (Reg := Reg) dj y₀ S R))

theorem coinedOf_setup (I : Inputs) (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    (y₀ : DrawSetupZK I mPts S₀ (dj S₀)) : CoinedOfSetup (Reg := Reg) I dj y₀ := by
  sorry

/-- **Zero knowledge at a `CoinedSetup`, at `δ₁` per hidden leaf** (Lemma B, `ZkWireFixed`'s form at one table): at any
coin vector, for any encoding `enc` of the view (the proof files) and any event `E`, the real view at any witness `w`
whose registered values are the public ones and `S_shvzk`'s at any `w0` are within `2·|Hid|·δ₁`, both ways. -/
def ZkViewAt (C : CoinedSetup K) (δ₁ : ℝ≥0∞) : Prop :=
  ∀ (c : C.Coins) {Out : Type} (enc : (∀ _ : Fin 1, (C.T c).tab.View) → Out) (w w0 : Fin 1 → C.W)
    (_hreg : ∀ j, (C.T c).tab.sReg (w j) = (C.T c).tab.pubReg) (E : Out → Prop),
    Game.prCoin (fun p : ∀ _ : Fin 1, (C.T c).tab.Rand × ((C.T c).tab.Hid → ZkView.Salt) =>
        E (enc fun j => (C.T c).viewExe (w j) (p j).1 (p j).2))
        ≤ Game.prCoin (fun p : ∀ _ : Fin 1, (C.T c).tab.SimRand × ((C.T c).tab.Hid → ZkView.Salt) =>
            E (enc fun j => (C.T c).simExe (w0 j) (p j).1 (p j).2))
          + ∑ _j : Fin 1, 2 * Fintype.card (C.T c).tab.Hid * δ₁
      ∧ Game.prCoin (fun p : ∀ _ : Fin 1, (C.T c).tab.SimRand × ((C.T c).tab.Hid → ZkView.Salt) =>
          E (enc fun j => (C.T c).simExe (w0 j) (p j).1 (p j).2))
        ≤ Game.prCoin (fun p : ∀ _ : Fin 1, (C.T c).tab.Rand × ((C.T c).tab.Hid → ZkView.Salt) =>
            E (enc fun j => (C.T c).viewExe (w j) (p j).1 (p j).2))
          + ∑ _j : Fin 1, 2 * Fintype.card (C.T c).tab.Hid * δ₁

end FlockSoundness.Discharge.Composed
