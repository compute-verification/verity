import Proofs.Flock.Claims.ZK.Hypotheses
import Proofs.Flock.Soundness.Discharge.ZkHidden.View
import Proofs.Flock.Soundness.Discharge.FailClosed.Mask

/-!
# Flock's zero knowledge at the tables `verify --zk` sets up

`Flock.zk_deployZk_of_models` takes five Props of the masked prover's models. At the tables `verify --zk` sets up, two of
them are theorems and one is a construction:

* where the prover goes on, its model is `J` `--zk` tables, each a `CoinedSetup` at coins it admits, each with its own
  types as `--session-tables J` has them (`Proto.TabModel`). Lemma B there is `ZkHidden.zkViewAtJ_of_ok`: what
  `NonDegenerate` and `InnerHolds` ask of the tables is in `CoinedOK`, which `coinedOf_setup_of` and `coinedOf_reg` prove
  at every table the session sets up (`SetupTable`, `setupTable_planZHJ`). The table witnesses' validity
  (`CoinedSetup.Valid`) is what stays, in `TabModel.Link`;
* the SHVZK simulator is built from the models (`Proto.withSim`): one tape per nonce and coin vector, and at each,
  `lift` of the tables' `simExe` on the dummy where the prover goes on and the dummy-witness prover's messages where it
  refuses. `SimLink` holds by construction (`Proto.withSim_shvzk`), and the prover is `G`'s.

`Flock.zk_deployZk_of_tables` is `Claims.Flock.ZK` at `FlockDeploy.deployZk` and the pinned hm96 key (`ZkView.My`,
`ZkView.saltHash`), under `Claims.Hm96Hiding` and three Props of a Lean model of `flock-circuit --zk`:

* `Proto.TMsgLink`: the prover's messages are the session's (`MsgLink`, with the tables' views where it goes on);
* `Proto.TablesOK`: where it goes on, its tables are tables the session sets up, at coins each admits;
* `Proto.CoinCheck`.

`Flock.zk_deployZk_of_tables_gap` is the same at the pinned key's hiding gap (`CROnly.hm96Hiding_gap`), with
`Claims.Hm96Hiding` discharged.
-/

namespace Claims.ZKMal

open Claims
open FlockSoundness.Discharge

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-! ## The tables the session sets up -/

/-- **A table the session sets up**: `coinedOf` at a table over a statement at its own regions, the statement a
`MaskSetup` that avoids the mask slot, its regions well formed and of whole words. -/
def SetupTable (C : Composed.CoinedSetup) : Prop :=
  ∃ (mPts : ℕ) (P : Composed.LigProver) (Tz : ZkSession.TabZK FlockLevel3.GF128 mPts) (st : _root_.Flock.Stmt)
    (hwf : FlockSoundness.Refine.StmtWF st),
    FlockSoundness.Refine.RegionsWF st ∧
      Tz.S = FlockSoundness.Refine.stmtOf st hwf (FlockSoundness.Refine.regionsOf st) ∧ Composed.MaskSetup st ∧
      Composed.avoidsMask st = true ∧ Composed.WholeWords st ∧ C = Composed.coinedOf P Tz st

/-- **A table the session sets up has Lemma B's facts** (`coinedOf_setup_of`), and its valid witnesses carry the public
region data (`coinedOf_reg`). -/
theorem SetupTable.ok {C : Composed.CoinedSetup} (h : SetupTable C) :
    Composed.CoinedOK FlockSoundness.execArith (FlockSoundness.Refine.fOf _root_.Flock.Zk.PAD_C) C ∧
      ∀ c w, C.Valid w → (C.T c).tab.sReg w = (C.T c).tab.pubReg := by
  obtain ⟨_, P, Tz, st, hwf, hr, hS, M, hdm, hW, rfl⟩ := h
  exact ⟨(Composed.coinedOf_setup_of P Tz st hwf hS M hdm).2.2.2.2.2, Composed.coinedOf_reg P Tz st hwf hr hS hW⟩

/-- **Each table of the hidden-output session's plan is a table the session sets up**, at every draw `verify --zk` sets
that table up (`zk_session_viewHJ`'s tables). -/
theorem setupTable_planZHJ {mPts n : ℕ} {Reg : Type} (P : Composed.LigProver) (I : Exec.Inputs)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
    (y₀ : ZkHidden.DrawSetupZHJ (Composed.inputsJ I 0) mPts S₀ d₀) (ht : I.tags.typed = false) (S : Finset (Fin n))
    (R : Reg) (j : Fin (Composed.nTab I))
    (y : ZkHidden.DrawSetupZHJ (Composed.inputsJ I j) mPts (Composed.partS I (dj S) S j) (Composed.drawJ I (dj S) j)) :
    SetupTable (Composed.coinedOf P ((ZkHidden.planZHJ (Reg := Reg) dj y₀ S R).get
      (Fin.cast (ZkHidden.planZHJ_length dj y₀ S R).symm j)) y.x.st) :=
  ⟨mPts, P, _, y.x.st, y.x.wf.1, y.x.wf.2, by rw [ZkHidden.planZHJ_get_eq dj y₀ S R j y]; rfl,
    ZkHidden.maskSetup_of_setupHidden ((Exec.tagsAt_typed _ _ _).trans ((ZkExec.zkTags_typed I).trans ht)) y.x.h,
    FailClosed.DrawSetupZHJ.avoidsMask y, Composed.setupHidden_whole y.x.h, rfl⟩

namespace Proto

/-! ## The model where the prover goes on, at the session's tables -/

/-- **Lemma B's setting at `J` `--zk` tables, at a coin vector where the prover goes on** (`ZkHidden.ZkViewAtJ`): table
`j` is the `CoinedSetup` `C j` at its coins `cz j`, each with its own types; the table witnesses of each witness and of
the dummy, the map `lift` from the tables' views to the prover's messages, and the map `eR` from `G`'s randomness to the
tables' randomness and their hidden leaves' salts. -/
structure TabModel {W : Type} (G : Proto W) (x : Msg) where
  J : ℕ
  C : Fin J → Composed.CoinedSetup
  cz : ∀ j, Composed.CoinsZK (C j).Rest
  wt : W → ∀ j, FlockSoundness.Model.Witness (C j).S.m
  wt₀ : ∀ j, FlockSoundness.Model.Witness (C j).S.m
  lift : (∀ j, ((C j).T (cz j)).tab.View) → List Msg
  eR : G.Rand → ∀ j, ((C j).T (cz j)).tab.Rand × (((C j).T (cz j)).tab.Hid → ZkView.Salt)
  kR : ℕ

/-- Where the prover goes on at `c`, its messages are `lift` of the tables' real views (`Coined.viewExe`) at valid table
witnesses, on randomness read through a map with fibers of one size. -/
def TabModel.Link {W : Type} {G : Proto W} {x : Msg} (M : G.TabModel x) (H : Msg → Msg) (ν : G.Nonce)
    (c : Fin (G.K x) → G.Coin) (w : W) : Prop :=
  (∀ j, (M.C j).Valid (M.wt w j)) ∧ (∀ p, Nat.card {r // M.eR r = p} = M.kR) ∧
    ∀ r, G.realVec H x w ν r c = M.lift fun j => ((M.C j).T (M.cz j)).viewExe (M.wt w j) (M.eR r j).1 (M.eR r j).2

/-- The tables are tables the session sets up, at coins each admits. -/
def TabModel.OK {W : Type} {G : Proto W} {x : Msg} (M : G.TabModel x) : Prop :=
  (∀ j, SetupTable (M.C j)) ∧ ∀ j, (M.C j).Admits (M.cz j)

/-- The tables' hidden leaves. -/
noncomputable def TabModel.size {W : Type} {G : Proto W} {x : Msg} (M : G.TabModel x) : ℕ :=
  ∑ j, Fintype.card ((M.C j).T (M.cz j)).tab.Hid

/-- `S_shvzk`'s tape at the tables: each one's simulator randomness and its hidden leaves' salts. -/
abbrev TabModel.SimAt {W : Type} {G : Proto W} {x : Msg} (M : G.TabModel x) : Type :=
  ∀ j, ((M.C j).T (M.cz j)).tab.SimRand × (((M.C j).T (M.cz j)).tab.Hid → ZkView.Salt)

/-- `lift` of the tables' `S_shvzk` outputs on the dummy (`Coined.simExe`). -/
noncomputable def TabModel.simOut {W : Type} {G : Proto W} {x : Msg} (M : G.TabModel x) (σ : M.SimAt) : List Msg :=
  M.lift fun j => ((M.C j).T (M.cz j)).simExe (M.wt₀ j) (σ j).1 (σ j).2

/-- **`SHVZK` at a coin vector where the prover goes on, at the session's tables** (`ZkHidden.zkViewAtJ_of_ok`, within
`2·n·δ₁`): the honest prover's messages at `c` and `simOut` on any tape read through a map with fibers of one size. -/
theorem TabModel.shvzk {W : Type} {G : Proto W} {x : Msg} (M : G.TabModel x) (H : Msg → Msg) {δ₁ : ENNReal}
    (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (hδ₁ : δ₁ ≠ ⊤) (ν : G.Nonce) (c : Fin (G.K x) → G.Coin)
    (w : W) (hL : M.Link H ν c w) (hOK : M.OK) {Ω : Type} [Fintype Ω] [Nonempty Ω] (eS : Ω → M.SimAt) {kS : ℕ}
    (heS : ∀ p, Nat.card {s // eS s = p} = kS) {n : ℕ} (hn : M.size ≤ n) (E : List Msg → Prop) :
    |Pr (fun r : G.Rand => E (G.realVec H x w ν r c)) - Pr (fun s : Ω => E (M.simOut (eS s)))| ≤ 2 * n * δ₁.toReal := by
  obtain ⟨hv, heR, hR⟩ := hL
  obtain ⟨hset, hA⟩ := hOK
  have hb := ZkHidden.zkViewAtJ_of_ok M.C (fun j => (hset j).ok.1) (fun j => (hset j).ok.2) hT1 M.cz hA M.lift
    (M.wt w) M.wt₀ hv E
  have : Nonempty M.SimAt := ⟨eS (Classical.arbitrary _)⟩
  have : Nonempty (∀ j, ((M.C j).T (M.cz j)).tab.Rand × (((M.C j).T (M.cz j)).tab.Hid → ZkView.Salt)) :=
    ⟨M.eR (Classical.arbitrary _)⟩
  have eReal : Pr (fun r : G.Rand => E (G.realVec H x w ν r c)) =
      (FlockSoundness.Game.prCoin fun p : ∀ j, ((M.C j).T (M.cz j)).tab.Rand ×
          (((M.C j).T (M.cz j)).tab.Hid → ZkView.Salt) =>
        E (M.lift fun j => ((M.C j).T (M.cz j)).viewExe (M.wt w j) (p j).1 (p j).2)).toReal := by
    rw [← Pr_eq_prCoin, ← Pr_comp_fiber M.eR heR]
    simp only [hR]
  have eSim : Pr (fun s : Ω => E (M.simOut (eS s))) =
      (FlockSoundness.Game.prCoin fun p : M.SimAt =>
        E (M.lift fun j => ((M.C j).T (M.cz j)).simExe (M.wt₀ j) (p j).1 (p j).2)).toReal := by
    simp only [TabModel.simOut]
    rw [← Pr_eq_prCoin, ← Pr_comp_fiber eS heS]
  have hle : ∑ j, 2 * (Fintype.card ((M.C j).T (M.cz j)).tab.Hid : ENNReal) * δ₁ ≤ 2 * n * δ₁ := by
    rw [← Finset.sum_mul, ← Finset.mul_sum]
    gcongr
    exact_mod_cast hn
  have htop : (2 * n * δ₁ : ENNReal) ≠ ⊤ :=
    ENNReal.mul_ne_top (ENNReal.mul_ne_top (by simp) (ENNReal.natCast_ne_top _)) hδ₁
  have hε : (∑ j, 2 * (Fintype.card ((M.C j).T (M.cz j)).tab.Hid : ENNReal) * δ₁) ≠ ⊤ :=
    ne_top_of_le_ne_top htop hle
  have hεr : (∑ j, 2 * (Fintype.card ((M.C j).T (M.cz j)).tab.Hid : ENNReal) * δ₁).toReal ≤ 2 * n * δ₁.toReal := by
    have := ENNReal.toReal_mono htop hle
    simpa [ENNReal.toReal_mul] using this
  have a := FlockSoundness.ZK.GK.toReal_le_add (FlockSoundness.ZK.prCoin_ne_top _) hε hb.1
  have b := FlockSoundness.ZK.GK.toReal_le_add (FlockSoundness.ZK.prCoin_ne_top _) hε hb.2
  rw [eReal, eSim, abs_le]
  constructor <;> linarith

/-! ## The model at each coin vector -/

/-- **The model at one coin vector, at the session's tables**: `TabModel` where the prover goes on, `OffModel` where it
refuses (its `Link`; `withSim` builds the simulator, and reads neither its `eS` nor its `kS`). -/
abbrev TCoinModel {W : Type} (G : Proto W) (x : Msg) : Type 1 :=
  G.TabModel x ⊕ G.OffModel x ZkView.Salt ZkView.Bits ByteArray

/-- The message link at a coin vector, at the pinned hm96 key where the prover refuses. -/
def TCoinModel.Link {W : Type} {G : Proto W} {x : Msg} :
    G.TCoinModel x → (Msg → Msg) → G.Nonce → (Fin (G.K x) → G.Coin) → W → Prop
  | .inl M => M.Link
  | .inr M => fun H => M.Link H ZkView.My ZkView.saltHash

/-- The session's tables where the prover goes on; nothing where it refuses. -/
def TCoinModel.OK {W : Type} {G : Proto W} {x : Msg} : G.TCoinModel x → Prop
  | .inl M => M.OK
  | .inr _ => True

/-- The leaves the coin vector's bound counts. -/
noncomputable def TCoinModel.size {W : Type} {G : Proto W} {x : Msg} : G.TCoinModel x → ℕ
  | .inl M => M.size
  | .inr M => M.size

/-- The SHVZK simulator's tape at a coin vector: `S_shvzk`'s where the prover goes on, the prover's where it refuses. -/
abbrev TCoinModel.SimAt {W : Type} {G : Proto W} {x : Msg} : G.TCoinModel x → Type
  | .inl M => M.SimAt
  | .inr _ => G.Rand

noncomputable instance TCoinModel.instFintypeSimAt {W : Type} {G : Proto W} {x : Msg} :
    (M : G.TCoinModel x) → Fintype M.SimAt
  | .inl _ => inferInstance
  | .inr _ => G.finRand

instance TCoinModel.instNonemptySimAt {W : Type} {G : Proto W} {x : Msg} : (M : G.TCoinModel x) → Nonempty M.SimAt
  | .inl M => ⟨fun j => ((ZkHidden.simRand_nonempty ((M.C j).T (M.cz j)).tab).some, fun _ => 0)⟩
  | .inr _ => G.neRand

/-- An oracle program's messages, each run in order. -/
def ocSeq : List (OC Msg) → OC (List Msg)
  | [] => .ret []
  | o :: os => ocBind o fun m => ocBind (ocSeq os) fun ms => .ret (m :: ms)

theorem ocSeq_eval (H : Msg → Msg) : ∀ os : List (OC Msg), ((ocSeq os).eval H).1 = os.map fun o => (o.eval H).1
  | [] => rfl
  | o :: os => by simp only [ocSeq, ocBind_eval, OC.eval, List.map_cons, ocSeq_eval H os]

/-- **The SHVZK simulator's messages at a coin vector, from its tape there**: `simOut` where the prover goes on, the
dummy-witness prover's messages at `c` where it refuses. -/
noncomputable def TCoinModel.simAt {W : Type} {G : Proto W} {x : Msg} :
    (M : G.TCoinModel x) → M.SimAt → G.Nonce → (Fin (G.K x) → G.Coin) → OC (List Msg)
  | .inl M, σ, _, _ => .ret (M.simOut σ)
  | .inr _, r, ν, c => ocSeq ((List.range (G.K x + 1)).map fun j => G.dmsg x ν r ((List.ofFn c).take j))

/-- **`SHVZK` at a coin vector, from its model**, for any simulator that is `simAt` on a tape read through a map with
fibers of one size. -/
theorem TCoinModel.shvzk {W : Type} {G : Proto W} {x : Msg} (H : Msg → Msg) {δ₁ : ENNReal}
    (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (hδ₁ : δ₁ ≠ ⊤) (ν : G.Nonce) (c : Fin (G.K x) → G.Coin)
    (w : W) {n : ℕ} (E : List Msg → Prop) {Ω : Type} [Fintype Ω] [Nonempty Ω] (sim : Ω → List Msg) :
    ∀ (M : G.TCoinModel x) (ev : Ω → M.SimAt) (kS : ℕ), (∀ p, Nat.card {s // ev s = p} = kS) →
      (∀ s, sim s = ((M.simAt (ev s) ν c).eval H).1) → M.Link H ν c w → M.OK → M.size ≤ n →
      |Pr (fun r : G.Rand => E (G.realVec H x w ν r c)) - Pr (fun s : Ω => E (sim s))| ≤ 2 * n * δ₁.toReal
  | .inl M, ev, _, hev, hsim, hL, hOK, hn => by
      have : sim = fun s => M.simOut (ev s) := funext hsim
      subst this
      exact M.shvzk H hT1 hδ₁ ν c w hL hOK ev hev hn E
  | .inr M, ev, kS, hev, hsim, hL, _, hn => by
      have hD : ∀ s, sim s = G.dVec H x ν (ev s) c := fun s => by
        rw [hsim s]
        simp only [TCoinModel.simAt, ocSeq_eval, List.map_map]
        rfl
      let G' : Proto W :=
        { G with
          SimRand := fun _ => Ω
          finSim := fun _ => inferInstance
          neSim := fun _ => inferInstance
          sim := fun _ _ _ s => .ret (sim s) }
      have hn' : Fintype.card M.L ≤ n := hn
      obtain ⟨he, hI, hreg, hreg₀, heR, hR, hD'⟩ := hL
      exact G'.shvzk_refused_hm96 H x w ν c M.T M.rep ZkView.My ZkView.saltHash hT1 hδ₁ M.Hl M.msgs M.dig M.e M.k
        he M.out hI (M.wt w) M.wt₀ hreg hreg₀ M.eR M.kR heR hR hD' ev kS hev hD hn' E

/-! ## The simulator the models give -/

/-- Evaluation at one point has fibers of one size. -/
theorem eval_fiber_card {ι : Type} {β : ι → Type} (i : ι) (b b' : β i) :
    Nat.card {f : ∀ k, β k // f i = b} = Nat.card {f : ∀ k, β k // f i = b'} := by
  classical
  exact Nat.card_congr
    { toFun := fun f => ⟨Function.update f.1 i b', Function.update_self ..⟩
      invFun := fun f => ⟨Function.update f.1 i b, Function.update_self ..⟩
      left_inv := fun ⟨f, hf⟩ => Subtype.ext (by subst hf; simp)
      right_inv := fun ⟨f, hf⟩ => Subtype.ext (by subst hf; simp) }

open Classical in
/-- **`G` with the SHVZK simulator its models give**: one tape per nonce and coin vector, and at `(ν, c)` the model's
messages from its tape there (`TCoinModel.simAt`). Its prover is `G`'s. -/
noncomputable def withSim {W : Type} (G : Proto W) [Fintype G.Coin]
    (Mc : ∀ x, G.Nonce → (Fin (G.K x) → G.Coin) → G.TCoinModel x) : Proto W :=
  { G with
    SimRand := fun x => ∀ q : G.Nonce × (Fin (G.K x) → G.Coin), (Mc x q.1 q.2).SimAt
    finSim := fun _ => inferInstance
    neSim := fun _ => inferInstance
    sim := fun x ν c s => (Mc x ν c).simAt (s (ν, c)) ν c }

theorem withSim_prover {W : Type} (G : Proto W) [Fintype G.Coin]
    (Mc : ∀ x, G.Nonce → (Fin (G.K x) → G.Coin) → G.TCoinModel x) : (G.withSim Mc).prover = G.prover := rfl

/-- **`SHVZK` of the simulator the models give**, within `2·n_B·δ₁`, from the message links and the tables. -/
theorem withSim_shvzk {W : Type} (G : Proto W) [Fintype G.Coin] (H : Msg → Msg) (R : Rel W) {δ₁ : ENNReal}
    (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (hδ₁ : δ₁ ≠ ⊤)
    (Mc : ∀ x, G.Nonce → (Fin (G.K x) → G.Coin) → G.TCoinModel x) (nB : Msg → ℕ)
    (hnB : ∀ x ν c, (Mc x ν c).size ≤ nB x) (hL : ∀ x w, R x w → ∀ ν c, (Mc x ν c).Link H ν c w)
    (hOK : ∀ x ν c, (Mc x ν c).OK) : (G.withSim Mc).SHVZK H R fun x => 2 * nB x * δ₁.toReal := by
  intro x w hR ν c E
  obtain ⟨p₀⟩ := TCoinModel.instNonemptySimAt (Mc x ν c)
  exact TCoinModel.shvzk (G := G) H hT1 hδ₁ ν c w E (Ω := (G.withSim Mc).SimRand x)
    (fun s => (((G.withSim Mc).sim x ν c s).eval H).1) (Mc x ν c) (fun s => s (ν, c)) _
    (fun p => eval_fiber_card (β := fun q : G.Nonce × (Fin (G.K x) → G.Coin) => (Mc x q.1 q.2).SimAt) (ν, c) p p₀)
    (fun _ => rfl) (hL x w hR ν c) (hOK x ν c) (hnB x ν c)

/-! ## The named hypotheses -/

/-- **The message link at the session's tables**: before the last coin the prover's messages are the session's strategy
on real hm96 leaves at the pinned key; at each coin vector they are the model's. -/
def TMsgLink {W : Type} (G : Proto W) (H : Msg → Msg) (R : Rel W)
    (Mp : ∀ x, G.Nonce → G.PreModel x ZkView.Salt ZkView.Bits ByteArray)
    (Mc : ∀ x, G.Nonce → (Fin (G.K x) → G.Coin) → G.TCoinModel x) : Prop :=
  ∀ x w, R x w → ∀ ν, (Mp x ν).Link H ZkView.My ZkView.saltHash ν w ∧ ∀ c, (Mc x ν c).Link H ν c w

/-- **The tables**: where the prover goes on, its tables are tables the session sets up, at coins each admits. -/
def TablesOK {W : Type} (G : Proto W) (Mc : ∀ x, G.Nonce → (Fin (G.K x) → G.Coin) → G.TCoinModel x) : Prop :=
  ∀ x ν c, (Mc x ν c).OK

end Proto

/-! ## Flock's zero knowledge at the session's tables -/

/-- **Flock's zero knowledge against a malicious verifier at `FlockDeploy.deployZk`, at the tables the session sets
up**: `Claims.Flock.ZK` at `P_zk = G.prover` and `S_GK = (G.withSim Mc).simulatorE t (t·3^t)`, at
`ε_zk = 8·n_L·δ₁ + 8t·n_B·δ₁ + 2·3^{-t}` and `γ = 4t·3^t`, for a committed-coin protocol whose coins are the deployed
verifier's, under `Claims.Hm96Hiding` at the pinned key and three named hypotheses: `TMsgLink`, `TablesOK` and
`CoinCheck`. -/
theorem Flock.zk_deployZk_of_tables (H : Msg → Msg)
    (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (G : Proto Flock.Wit) [Fintype G.Coin]
    (t : ℕ) (ht : 0 < t) (hK : ∀ x, G.K x = (FlockDeploy.V true).k x - 1) {δ₁ : ENNReal}
    (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (hδ₁ : δ₁ ≠ ⊤)
    (Mp : ∀ x, G.Nonce → G.PreModel x ZkView.Salt ZkView.Bits ByteArray)
    (Mc : ∀ x, G.Nonce → (Fin (G.K x) → G.Coin) → G.TCoinModel x) (nL nB : Msg → ℕ)
    (hnL : ∀ x ν, Fintype.card (Mp x ν).L ≤ nL x) (hnB : ∀ x ν c, (Mc x ν c).size ≤ nB x)
    (hL : G.TMsgLink H (Flock.R (FlockDeploy.deployZk rel).parse) Mp Mc) (hOK : G.TablesOK Mc) (hC : G.CoinCheck) :
    Claims.Flock.ZK H (FlockDeploy.deployZk rel) G.prover ((G.withSim Mc).simulatorE t (t * 3 ^ t))
      (fun x => 8 * nL x * δ₁.toReal + 8 * t * nB x * δ₁.toReal + 2 * (1 / 3) ^ t) fun _ => 4 * t * 3 ^ t := by
  obtain ⟨_, _, Tr, nb, root, parse, pos, idx, n, depth, dc, hchk, hdec⟩ := hC
  have hA : ∀ x w, Flock.R (FlockDeploy.deployZk rel).parse x w → ∀ (ν : G.Nonce)
      (V : FlockSoundness.ZK.Responder Msg Out (G.K x + 1)) (E : (Fin (G.K x + 1) → Msg) → Prop),
      |Pr (fun r : G.Rand => E (FlockSoundness.ZK.run (G.preStrat H x ν (G.msg x w ν)) V r)) -
        Pr (fun r : G.Rand => E (FlockSoundness.ZK.run (G.preStrat H x ν (G.dmsg x ν)) V r))| ≤
        2 * nL x * δ₁.toReal := by
    intro x w hR ν V E
    calc _ ≤ 2 * Fintype.card (Mp x ν).L * δ₁.toReal :=
          (Mp x ν).runs_close H ZkView.My ZkView.saltHash hT1 hδ₁ ν w (hL x w hR ν).1 V E
      _ ≤ 2 * nL x * δ₁.toReal := by
          have : (Fintype.card (Mp x ν).L : ℝ) ≤ nL x := by exact_mod_cast hnL x ν
          gcongr
  rw [← G.withSim_prover Mc]
  exact Flock.zkR_gkE_hm96_deployZk H rel (G.withSim Mc) t ht hK δ₁ nL nB hA
    (G.withSim_shvzk H _ hT1 hδ₁ Mc nB hnB (fun x w hR ν c => (hL x w hR ν).2 c) hOK) Tr nb root parse pos idx n
    depth dc hchk hdec

/-- **The same at the pinned key's hiding gap** (`CROnly.hm96Hiding_gap`): `Claims.Flock.ZK` at `δ₁` the statistical
distance of `(M·y, c(y))` from `(U, c(y))` at the pinned key, under `TMsgLink`, `TablesOK` and `CoinCheck` alone. -/
theorem Flock.zk_deployZk_of_tables_gap (H : Msg → Msg)
    (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (G : Proto Flock.Wit) [Fintype G.Coin]
    (t : ℕ) (ht : 0 < t) (hK : ∀ x, G.K x = (FlockDeploy.V true).k x - 1)
    (Mp : ∀ x, G.Nonce → G.PreModel x ZkView.Salt ZkView.Bits ByteArray)
    (Mc : ∀ x, G.Nonce → (Fin (G.K x) → G.Coin) → G.TCoinModel x) (nL nB : Msg → ℕ)
    (hnL : ∀ x ν, Fintype.card (Mp x ν).L ≤ nL x) (hnB : ∀ x ν c, (Mc x ν c).size ≤ nB x)
    (hL : G.TMsgLink H (Flock.R (FlockDeploy.deployZk rel).parse) Mp Mc) (hOK : G.TablesOK Mc) (hC : G.CoinCheck) :
    Claims.Flock.ZK H (FlockDeploy.deployZk rel) G.prover ((G.withSim Mc).simulatorE t (t * 3 ^ t))
      (fun x => 8 * nL x * (ENNReal.ofReal (FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash)).toReal +
        8 * t * nB x * (ENNReal.ofReal (FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash)).toReal +
        2 * (1 / 3) ^ t) fun _ => 4 * t * 3 ^ t :=
  Flock.zk_deployZk_of_tables H rel G t ht hK (FlockSoundness.CROnly.hm96Hiding_gap ZkView.My ZkView.saltHash)
    ENNReal.ofReal_ne_top Mp Mc nL nB hnL hnB hL hOK hC

end Claims.ZKMal
