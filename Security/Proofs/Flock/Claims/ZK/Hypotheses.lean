import Proofs.Flock.Claims.ZK.Shvzk
import Proofs.Flock.Claims.ZK.Flock
import Proofs.Flock.Claims.ZK.Pin

/-!
# What Flock's zero knowledge against a malicious verifier still assumes of the masked prover

`Flock.zkR_gkE_hm96` reads the deployed `--zk` protocol (`G : Proto Flock.Wit`) through the session analysis's objects:
the session's tables, its hm96 leaves, the leaves' digests and the maps from the prover's randomness to the tables'. A Lean
model of `flock-circuit --zk`'s messages supplies them:

* `Proto.PreModel`: Lemma C's setting for the messages before the last coin, at the verifier's answers;
* `Proto.CoinModel`: at each coin vector, Lemma B's (`Proto.OnModel`, where the prover goes on) or Lemma C's at fixed
  answers (`Proto.OffModel`, where it refuses).

Five Props say what the models must satisfy; `Claims.Hm96Hiding` stays the one named assumption.

* `Proto.MsgLink`: the prover's messages are the session's (`hlink`, `hP`, `hI` before the last coin; `hR`, `hD` at
  each coin vector), on randomness read through maps with fibers of one size, its table witnesses carrying the public
  region data;
* `Proto.SimLink`: the SHVZK simulator is the tables' `S_shvzk` where the prover goes on and the dummy-witness prover
  where it refuses (`hS`);
* `Proto.NonDegenerate`: the tables are non-degenerate where the prover goes on (`Rank`, `PadOnto`, each `W^r` a
  bijection, `PadsOnto`, `β ≠ 0`);
* `Proto.InnerHolds`: there, a valid witness's pads satisfy the inner constraint `⟨λ, h⟩ = θ`;
* `Proto.CoinCheck`: the opening check is the coin-tree walk and the coin is read from the opening's coins (`hchk`,
  `hdec`).

`Proto.prefix_of_models` gives Lemma C's bound on the runs, `Proto.shvzk_of_models` `SHVZK`, and
`Proto.binding_of_coinCheck` `Binding`. `Flock.zk_of_models` is `Claims.Flock.ZKAt` for a masked `D` whose rounds with
`Hello` are the protocol's, and `Flock.zkAt_deployZk_of_models` is `Claims.Flock.ZKAt` at `FlockDeploy.deployZk`, both at

  `ε_zk(x) = 8·n_L(x)·δ₁ + 8t·n_B(x)·δ₁ + 2·3^{-t}` and `γ(x) = 4t·3^t`,

`n_L` bounding the pre-final view's leaves and `n_B` each coin vector's hidden leaves (`OnModel.size`) or pre-final leaves
(`OffModel.size`). `Flock.zk_deployZk_of_models` is `Claims.Flock.ZK` at the masked record `zkOf`, whose reference prover
is `honestPZk`, where `G`'s messages are `honestPZk`'s (`Proto.Honest`).
-/

namespace Claims.ZKMal

open Claims

namespace Proto

variable {W : Type}

/-! ## The models -/

/-- **Lemma C's setting at statement `x`** (`Proto.runs_close_hm96`): the session's tables `T a` at the coins the answers
`a` open, the prover `P` given by `msgs` of its leaves `Hl (dig + M·y, c(y))`, the tables' randomness `e`, the pre-final
view's cut `out`, the table witnesses of each witness and of the dummy, and the map `eR` from `G`'s randomness. -/
structure PreModel (G : Proto W) (x : Msg) (Salt B Cc : Type) where
  F : Type
  K : Type
  Wt : Type
  D : Type
  Hi : Type
  VL : Type
  Ω : Type
  L : Type
  [fieldF : Field F]
  [fieldK : Field K]
  [charF : CharP F 2]
  [groupHi : AddCommGroup Hi]
  [finF : Fintype F]
  [decF : DecidableEq F]
  [finHi : Fintype Hi]
  [decHi : DecidableEq Hi]
  [finΩ : Fintype Ω]
  [neΩ : Nonempty Ω]
  [finL : Fintype L]
  [decL : DecidableEq L]
  J : ℕ
  T : (Fin (G.K x + 1) → Out) → Fin J → FlockSoundness.ZK.Table F K Wt Salt (B × Salt) D Hi VL
  rep : ∀ a j, Fin (T a j).nC → Fin 2
  Hl : B × Cc → D
  msgs : (Fin J → Wt) → Ω → (Fin (G.K x + 1) → Out) → (L → D) → Fin (G.K x + 1) → Msg
  dig : (Fin J → Wt) → Ω → (Fin (G.K x + 1) → Out) → L → B
  P : (Fin J → Wt) → FlockSoundness.ZK.Strategy (Ω × (L → B × Cc)) Msg Out (G.K x + 1)
  e : ∀ a, Ω × (L → B × Salt) → ∀ j, (T a j).Rand
  k : (Fin (G.K x + 1) → Out) → ℕ
  out : ∀ a, (∀ j, (T a j).Cut) → Fin (G.K x + 1) → Msg
  wt : W → Fin J → Wt
  wt₀ : Fin J → Wt
  eR : G.Rand → Ω × (L → Salt)
  kR : ℕ

attribute [instance] PreModel.fieldF PreModel.fieldK PreModel.charF PreModel.groupHi PreModel.finF PreModel.decF
  PreModel.finHi PreModel.decHi PreModel.finΩ PreModel.neΩ PreModel.finL PreModel.decL

/-- **Lemma B's setting at a coin vector where the prover goes on** (`Proto.shvzk_at_hm96`): the session's tables `T`
with their hm96 leaves `Mh`, the table witnesses, the map `lift` from the tables' views to the prover's messages, and the
maps `eR`, `eS` from `G`'s randomness and the simulator's to the tables'. -/
structure OnModel (G : Proto W) (x : Msg) (Salt B Cc : Type) [AddCommGroup B] where
  F : Type
  K : Type
  Wt : Type
  D : Type
  Hi : Type
  VL : Type
  [fieldF : Field F]
  [fieldK : Field K]
  [charF : CharP F 2]
  [groupHi : AddCommGroup Hi]
  [finF : Fintype F]
  [decF : DecidableEq F]
  [finK : Fintype K]
  [decK : DecidableEq K]
  [finHi : Fintype Hi]
  [decHi : DecidableEq Hi]
  J : ℕ
  T : Fin J → FlockSoundness.ZK.Table F K Wt Salt (B × Salt) D Hi VL
  Mh : ∀ j, (T j).Hm96 Cc
  wt : W → Fin J → Wt
  wt₀ : Fin J → Wt
  lift : (∀ j, (T j).View) → List Msg
  eR : G.Rand → ∀ j, (T j).Rand × ((T j).Hid → Salt)
  kR : ℕ
  eS : G.SimRand x → ∀ j, (T j).SimRand × ((T j).Hid → Salt)
  kS : ℕ

attribute [instance] OnModel.fieldF OnModel.fieldK OnModel.charF OnModel.groupHi OnModel.finF OnModel.decF
  OnModel.finK OnModel.decK OnModel.finHi OnModel.decHi

/-- **Lemma C's setting at fixed answers, at a coin vector where the prover refuses** (`Proto.shvzk_refused_hm96`): the
session's tables `T`, the prover's messages `msgs` of its leaves, the tables' randomness `e`, the pre-final view's cut
`out`, the table witnesses, and the maps `eR`, `eS` from `G`'s randomness and the simulator's. -/
structure OffModel (G : Proto W) (x : Msg) (Salt B Cc : Type) where
  F : Type
  K : Type
  Wt : Type
  D : Type
  Hi : Type
  VL : Type
  Ω : Type
  L : Type
  [fieldF : Field F]
  [fieldK : Field K]
  [charF : CharP F 2]
  [groupHi : AddCommGroup Hi]
  [finF : Fintype F]
  [decF : DecidableEq F]
  [finHi : Fintype Hi]
  [decHi : DecidableEq Hi]
  [finΩ : Fintype Ω]
  [neΩ : Nonempty Ω]
  [finL : Fintype L]
  [decL : DecidableEq L]
  J : ℕ
  T : Fin J → FlockSoundness.ZK.Table F K Wt Salt (B × Salt) D Hi VL
  rep : ∀ j, Fin (T j).nC → Fin 2
  Hl : B × Cc → D
  msgs : (Fin J → Wt) → Ω → (L → D) → List Msg
  dig : (Fin J → Wt) → Ω → L → B
  e : Ω × (L → B × Salt) → ∀ j, (T j).Rand
  k : ℕ
  out : (∀ j, (T j).Cut) → List Msg
  wt : W → Fin J → Wt
  wt₀ : Fin J → Wt
  eR : G.Rand → Ω × (L → Salt)
  kR : ℕ
  eS : G.SimRand x → G.Rand
  kS : ℕ

attribute [instance] OffModel.fieldF OffModel.fieldK OffModel.charF OffModel.groupHi OffModel.finF OffModel.decF
  OffModel.finHi OffModel.decHi OffModel.finΩ OffModel.neΩ OffModel.finL OffModel.decL

/-- The model at one coin vector: where the prover goes on, or where it refuses. -/
abbrev CoinModel (G : Proto W) (x : Msg) (Salt B Cc : Type) [AddCommGroup B] : Type 1 :=
  G.OnModel x Salt B Cc ⊕ G.OffModel x Salt B Cc

/-! ## What each model must satisfy -/

section Props

variable {G : Proto W} {x : Msg} {Salt B Cc : Type} [AddCommGroup B]

/-- Before the last coin, the prover's messages on `w` and on the dummy are the session's strategy on real hm96 leaves. -/
def PreModel.Link (M : G.PreModel x Salt B Cc) (H : Msg → Msg) (My : Salt → B) (cs : Salt → Cc) (ν : G.Nonce) (w : W) :
    Prop :=
  (∀ wt ω p a, (M.P wt).msg (ω, p) a = M.msgs wt ω a fun l => M.Hl (M.dig wt ω a l + (p l).1, (p l).2)) ∧
  (∀ a r, Nat.card {u // M.e a u = r} = M.k a) ∧
  (∀ wt ω u a, M.msgs wt ω a (fun l => M.Hl ((u l).1, cs (u l).2)) =
    M.out a fun j => (M.T a j).cut (FlockSoundness.ZK.Session.sent (M.T a) (M.rep a) j)
      ((M.T a j).preView (wt j) (M.e a (ω, u) j))) ∧
  (∀ a j, (M.T a j).sReg (M.wt w j) = (M.T a j).pubReg) ∧
  (∀ a j, (M.T a j).sReg (M.wt₀ j) = (M.T a j).pubReg) ∧
  (∀ p, Nat.card {r // M.eR r = p} = M.kR) ∧
  (∀ r, (G.preStrat H x ν (G.msg x w ν)).msg r =
    (M.P (M.wt w)).msg ((M.eR r).1, fun l => (My ((M.eR r).2 l), cs ((M.eR r).2 l)))) ∧
  ∀ r, (G.preStrat H x ν (G.dmsg x ν)).msg r =
    (M.P M.wt₀).msg ((M.eR r).1, fun l => (My ((M.eR r).2 l), cs ((M.eR r).2 l)))

/-- Where the prover goes on at `c`, its messages are `lift` of the session's real view on hm96 leaves `(My, cs)`. -/
def OnModel.Link (M : G.OnModel x Salt B Cc) (H : Msg → Msg) (My : Salt → B) (cs : Salt → Cc) (ν : G.Nonce)
    (c : Fin (G.K x) → G.Coin) (w : W) : Prop :=
  (∀ j, (M.Mh j).My = My ∧ (M.Mh j).cs = cs) ∧
  (∀ j, (M.T j).sReg (M.wt w j) = (M.T j).pubReg) ∧
  (∀ p, Nat.card {r // M.eR r = p} = M.kR) ∧
  ∀ r, G.realVec H x w ν r c = M.lift fun j => (M.Mh j).viewR (M.wt w j) (M.eR r j).1 (M.eR r j).2

/-- Where the prover goes on at `c`, the simulator's messages are `lift` of the tables' `S_shvzk` outputs on the dummy. -/
def OnModel.Sim (M : G.OnModel x Salt B Cc) (H : Msg → Msg) (ν : G.Nonce) (c : Fin (G.K x) → G.Coin) : Prop :=
  (∀ p, Nat.card {s // M.eS s = p} = M.kS) ∧
  ∀ s, ((G.sim x ν c s).eval H).1 = M.lift fun j => (M.Mh j).simR (M.wt₀ j) (M.eS s j).1 (M.eS s j).2

/-- The tables are non-degenerate: Lemma B's `Rank`, `PadOnto`, each `W^r` a bijection, `PadsOnto` and `β ≠ 0`. -/
def OnModel.NonDegenerate (M : G.OnModel x Salt B Cc) : Prop :=
  (∀ j, (M.T j).Rank) ∧ (∀ j, (M.T j).PadOnto) ∧
  (∀ j r, Function.Bijective fun p : M.F × M.F => (M.T j).Wr r p.1 p.2) ∧ (∀ j, (M.T j).PadsOnto) ∧
  ∀ j r, (M.T j).β r ≠ 0

/-- `w`'s pads satisfy the inner constraint `⟨λ, h⟩ = θ` in every table. -/
def OnModel.Inner (M : G.OnModel x Salt B Cc) (w : W) : Prop := ∀ j, (M.T j).InnerHolds (M.wt w j)

/-- The session's hidden leaves. -/
def OnModel.size (M : G.OnModel x Salt B Cc) : ℕ := ∑ j, Fintype.card (M.T j).Hid

/-- Where the prover refuses at `c`, its messages on `w` and on the dummy are `msgs` of real hm96 leaves. -/
def OffModel.Link (M : G.OffModel x Salt B Cc) (H : Msg → Msg) (My : Salt → B) (cs : Salt → Cc) (ν : G.Nonce)
    (c : Fin (G.K x) → G.Coin) (w : W) : Prop :=
  (∀ r, Nat.card {u // M.e u = r} = M.k) ∧
  (∀ wt ω u, M.msgs wt ω (fun l => M.Hl ((u l).1, cs (u l).2)) =
    M.out fun j => (M.T j).cut (FlockSoundness.ZK.Session.sent M.T M.rep j) ((M.T j).preView (wt j) (M.e (ω, u) j))) ∧
  (∀ j, (M.T j).sReg (M.wt w j) = (M.T j).pubReg) ∧
  (∀ j, (M.T j).sReg (M.wt₀ j) = (M.T j).pubReg) ∧
  (∀ p, Nat.card {r // M.eR r = p} = M.kR) ∧
  (∀ r, G.realVec H x w ν r c =
    M.msgs (M.wt w) (M.eR r).1 fun l => M.Hl (M.dig (M.wt w) (M.eR r).1 l + My ((M.eR r).2 l), cs ((M.eR r).2 l))) ∧
  ∀ r, G.dVec H x ν r c =
    M.msgs M.wt₀ (M.eR r).1 fun l => M.Hl (M.dig M.wt₀ (M.eR r).1 l + My ((M.eR r).2 l), cs ((M.eR r).2 l))

/-- Where the prover refuses at `c`, the simulator sends the dummy-witness prover's messages at `c`. -/
def OffModel.Sim (M : G.OffModel x Salt B Cc) (H : Msg → Msg) (ν : G.Nonce) (c : Fin (G.K x) → G.Coin) : Prop :=
  (∀ r, Nat.card {s // M.eS s = r} = M.kS) ∧ ∀ s, ((G.sim x ν c s).eval H).1 = G.dVec H x ν (M.eS s) c

/-- The session's pre-final leaves. -/
def OffModel.size (M : G.OffModel x Salt B Cc) : ℕ := Fintype.card M.L

/-- The message link at a coin vector. -/
def CoinModel.Link : G.CoinModel x Salt B Cc → (Msg → Msg) → (Salt → B) → (Salt → Cc) → G.Nonce →
    (Fin (G.K x) → G.Coin) → W → Prop
  | .inl M => M.Link
  | .inr M => M.Link

/-- The simulator at a coin vector. -/
def CoinModel.Sim : G.CoinModel x Salt B Cc → (Msg → Msg) → G.Nonce → (Fin (G.K x) → G.Coin) → Prop
  | .inl M => M.Sim
  | .inr M => M.Sim

/-- Non-degenerate tables where the prover goes on; nothing where it refuses. -/
def CoinModel.NonDegenerate : G.CoinModel x Salt B Cc → Prop
  | .inl M => M.NonDegenerate
  | .inr _ => True

/-- The inner constraint where the prover goes on; nothing where it refuses. -/
def CoinModel.Inner : G.CoinModel x Salt B Cc → W → Prop
  | .inl M => M.Inner
  | .inr _ => fun _ => True

/-- The leaves the coin vector's bound counts. -/
def CoinModel.size : G.CoinModel x Salt B Cc → ℕ
  | .inl M => M.size
  | .inr M => M.size

end Props

/-! ## The named hypotheses -/

/-- **The message link**; discharged by a Lean model of `flock-circuit --zk`'s messages built as the session's strategy. -/
def MsgLink (G : Proto W) (H : Msg → Msg) (R : Rel W) {Salt B Cc : Type} [AddCommGroup B] (My : Salt → B)
    (cs : Salt → Cc) (Mp : ∀ x, G.Nonce → G.PreModel x Salt B Cc)
    (Mc : ∀ x, G.Nonce → (Fin (G.K x) → G.Coin) → G.CoinModel x Salt B Cc) : Prop :=
  ∀ x w, R x w → ∀ ν, (Mp x ν).Link H My cs ν w ∧ ∀ c, (Mc x ν c).Link H My cs ν c w

/-- **The simulator**; discharged by defining `G.sim` as `lift` of `S_shvzk` where the prover goes on, `dVec` elsewhere. -/
def SimLink (G : Proto W) (H : Msg → Msg) {Salt B Cc : Type} [AddCommGroup B]
    (Mc : ∀ x, G.Nonce → (Fin (G.K x) → G.Coin) → G.CoinModel x Salt B Cc) : Prop :=
  ∀ x ν c, (Mc x ν c).Sim H ν c

/-- **The tables' non-degeneracy**; discharged by a model whose prover refuses wherever the self-check finds them degenerate. -/
def NonDegenerate (G : Proto W) {Salt B Cc : Type} [AddCommGroup B]
    (Mc : ∀ x, G.Nonce → (Fin (G.K x) → G.Coin) → G.CoinModel x Salt B Cc) : Prop :=
  ∀ x ν c, (Mc x ν c).NonDegenerate

/-- **`InnerHolds`**; discharged by proving each of the ten constraints, affine in `h`, of a valid witness's clear values. -/
def InnerHolds (G : Proto W) (R : Rel W) {Salt B Cc : Type} [AddCommGroup B]
    (Mc : ∀ x, G.Nonce → (Fin (G.K x) → G.Coin) → G.CoinModel x Salt B Cc) : Prop :=
  ∀ x w, R x w → ∀ ν c, (Mc x ν c).Inner w

/-- **The coin check**; discharged by defining `G.chk` as the coin-tree walk `treeChk` and `G.dec` as the opening's coins. -/
def CoinCheck (G : Proto W) : Prop :=
  ∃ (CoinT CSalt : Type) (Tr : Msg → Msg → FlockSoundness.ZK.GK.CoinTree Dg CoinT CSalt) (nb : G.Nonce → List UInt8)
    (root : Msg → Msg → Dg) (parse : Msg → Option (FlockSoundness.ZK.GK.CoinOpening Dg CoinT CSalt))
    (pos : Msg → Msg → ℕ → List UInt8) (idx n depth : Msg → Msg → ℕ → ℕ) (dc : List CoinT → G.Coin),
    (∀ x ν ρ j m, G.chk x ν ρ j m =
      treeChk (Tr x ρ) (nb ν) (root x ρ) (pos x ρ j) (idx x ρ j) (n x ρ j) (depth x ρ j) (parse m)) ∧
    ∀ m op, parse m = some op → G.dec m = dc op.cs

/-! ## What they give -/

/-- **Lemma C's bound on the runs before the last coin, from a model** (`Proto.runs_close_hm96`). -/
theorem PreModel.runs_close {G : Proto W} {x : Msg} {Salt B Cc : Type} [AddCommGroup Salt] [AddCommGroup B]
    [Fintype Salt] [DecidableEq Salt] [Fintype B] [DecidableEq B] (M : G.PreModel x Salt B Cc) (H : Msg → Msg)
    (My : Salt → B) (cs : Salt → Cc) {δ₁ : ENNReal} (hT1 : Claims.Hm96Hiding My cs δ₁) (hδ₁ : δ₁ ≠ ⊤) (ν : G.Nonce)
    (w : W) (hL : M.Link H My cs ν w) (V : FlockSoundness.ZK.Responder Msg Out (G.K x + 1))
    (E : (Fin (G.K x + 1) → Msg) → Prop) :
    |Pr (fun r : G.Rand => E (FlockSoundness.ZK.run (G.preStrat H x ν (G.msg x w ν)) V r)) -
      Pr (fun r : G.Rand => E (FlockSoundness.ZK.run (G.preStrat H x ν (G.dmsg x ν)) V r))| ≤
      2 * Fintype.card M.L * δ₁.toReal := by
  obtain ⟨hP, he, hI, hreg, hreg₀, heR, hlink, hlink₀⟩ := hL
  exact G.runs_close_hm96 H x ν w M.T M.rep My cs hT1 hδ₁ M.Hl M.msgs M.dig M.P hP M.e M.k he M.out hI (M.wt w) M.wt₀
    hreg hreg₀ M.eR M.kR heR hlink hlink₀ V E

/-- **`SHVZK` at a coin vector where the prover goes on, from a model** (`Proto.shvzk_at_hm96`). -/
theorem OnModel.shvzk {G : Proto W} {x : Msg} {Salt B Cc : Type} [AddCommGroup Salt] [AddCommGroup B]
    [Fintype Salt] [DecidableEq Salt] [Fintype B] [DecidableEq B] (M : G.OnModel x Salt B Cc) (H : Msg → Msg)
    (My : Salt → B) (cs : Salt → Cc) {δ₁ : ENNReal} (hT1 : Claims.Hm96Hiding My cs δ₁) (hδ₁ : δ₁ ≠ ⊤) (ν : G.Nonce)
    (c : Fin (G.K x) → G.Coin) (w : W) (hL : M.Link H My cs ν c w) (hS : M.Sim H ν c) (hN : M.NonDegenerate)
    (hI : M.Inner w) {n : ℕ} (hn : M.size ≤ n) (E : List Msg → Prop) :
    |Pr (fun r : G.Rand => E (G.realVec H x w ν r c)) - Pr (fun s : G.SimRand x => E ((G.sim x ν c s).eval H).1)|
      ≤ 2 * n * δ₁.toReal := by
  obtain ⟨hMh, hreg, heR, hR⟩ := hL
  obtain ⟨heS, hS⟩ := hS
  obtain ⟨hrank, hpad, hW, hpads, hβ⟩ := hN
  exact G.shvzk_at_hm96 H x w ν c M.T M.Mh (fun j => by rw [(hMh j).1, (hMh j).2]; exact hT1) hδ₁ hrank hpad hW
    hpads hβ (M.wt w) M.wt₀ hreg hI M.lift M.eR M.kR heR M.eS M.kS heS hR hS hn E

/-- **`SHVZK` at a coin vector where the prover refuses, from a model** (`Proto.shvzk_refused_hm96`). -/
theorem OffModel.shvzk {G : Proto W} {x : Msg} {Salt B Cc : Type} [AddCommGroup Salt] [AddCommGroup B]
    [Fintype Salt] [DecidableEq Salt] [Fintype B] [DecidableEq B] (M : G.OffModel x Salt B Cc) (H : Msg → Msg)
    (My : Salt → B) (cs : Salt → Cc) {δ₁ : ENNReal} (hT1 : Claims.Hm96Hiding My cs δ₁) (hδ₁ : δ₁ ≠ ⊤) (ν : G.Nonce)
    (c : Fin (G.K x) → G.Coin) (w : W) (hL : M.Link H My cs ν c w) (hS : M.Sim H ν c) {n : ℕ} (hn : M.size ≤ n)
    (E : List Msg → Prop) :
    |Pr (fun r : G.Rand => E (G.realVec H x w ν r c)) - Pr (fun s : G.SimRand x => E ((G.sim x ν c s).eval H).1)|
      ≤ 2 * n * δ₁.toReal := by
  obtain ⟨he, hI, hreg, hreg₀, heR, hR, hD⟩ := hL
  obtain ⟨heS, hS⟩ := hS
  exact G.shvzk_refused_hm96 H x w ν c M.T M.rep My cs hT1 hδ₁ M.Hl M.msgs M.dig M.e M.k he M.out hI (M.wt w) M.wt₀
    hreg hreg₀ M.eR M.kR heR hR hD M.eS M.kS heS hS hn E

/-- **`SHVZK` at a coin vector, from its model.** -/
theorem CoinModel.shvzk {G : Proto W} {x : Msg} {Salt B Cc : Type} [AddCommGroup Salt] [AddCommGroup B]
    [Fintype Salt] [DecidableEq Salt] [Fintype B] [DecidableEq B] (H : Msg → Msg) (My : Salt → B) (cs : Salt → Cc)
    {δ₁ : ENNReal} (hT1 : Claims.Hm96Hiding My cs δ₁) (hδ₁ : δ₁ ≠ ⊤) (ν : G.Nonce) (c : Fin (G.K x) → G.Coin) (w : W)
    {n : ℕ} (E : List Msg → Prop) :
    ∀ M : G.CoinModel x Salt B Cc, M.Link H My cs ν c w → M.Sim H ν c → M.NonDegenerate → M.Inner w → M.size ≤ n →
      |Pr (fun r : G.Rand => E (G.realVec H x w ν r c)) - Pr (fun s : G.SimRand x => E ((G.sim x ν c s).eval H).1)|
        ≤ 2 * n * δ₁.toReal
  | .inl M, hL, hS, hN, hI, hn => M.shvzk H My cs hT1 hδ₁ ν c w hL hS hN hI hn E
  | .inr M, hL, hS, _, _, hn => M.shvzk H My cs hT1 hδ₁ ν c w hL hS hn E

/-- **Lemma C's bound on the runs before the last coin, at every statement, witness and nonce**, within `2·n_L·δ₁`. -/
theorem prefix_of_models (G : Proto W) (H : Msg → Msg) (R : Rel W) {Salt B Cc : Type} [AddCommGroup Salt]
    [AddCommGroup B] [Fintype Salt] [DecidableEq Salt] [Fintype B] [DecidableEq B] (My : Salt → B) (cs : Salt → Cc)
    {δ₁ : ENNReal} (hT1 : Claims.Hm96Hiding My cs δ₁) (hδ₁ : δ₁ ≠ ⊤) (Mp : ∀ x, G.Nonce → G.PreModel x Salt B Cc)
    (Mc : ∀ x, G.Nonce → (Fin (G.K x) → G.Coin) → G.CoinModel x Salt B Cc) (nL : Msg → ℕ)
    (hnL : ∀ x ν, Fintype.card (Mp x ν).L ≤ nL x) (hL : G.MsgLink H R My cs Mp Mc) :
    ∀ x w, R x w → ∀ (ν : G.Nonce) (V : FlockSoundness.ZK.Responder Msg Out (G.K x + 1))
      (E : (Fin (G.K x + 1) → Msg) → Prop),
      |Pr (fun r : G.Rand => E (FlockSoundness.ZK.run (G.preStrat H x ν (G.msg x w ν)) V r)) -
        Pr (fun r : G.Rand => E (FlockSoundness.ZK.run (G.preStrat H x ν (G.dmsg x ν)) V r))| ≤
        2 * nL x * δ₁.toReal := by
  intro x w hR ν V E
  calc _ ≤ 2 * Fintype.card (Mp x ν).L * δ₁.toReal := (Mp x ν).runs_close H My cs hT1 hδ₁ ν w (hL x w hR ν).1 V E
    _ ≤ 2 * nL x * δ₁.toReal := by
      have : (Fintype.card (Mp x ν).L : ℝ) ≤ nL x := by exact_mod_cast hnL x ν
      gcongr

/-- **`SHVZK` from the models**, within `2·n_B·δ₁`. -/
theorem shvzk_of_models (G : Proto W) (H : Msg → Msg) (R : Rel W) {Salt B Cc : Type} [AddCommGroup Salt]
    [AddCommGroup B] [Fintype Salt] [DecidableEq Salt] [Fintype B] [DecidableEq B] (My : Salt → B) (cs : Salt → Cc)
    {δ₁ : ENNReal} (hT1 : Claims.Hm96Hiding My cs δ₁) (hδ₁ : δ₁ ≠ ⊤) (Mp : ∀ x, G.Nonce → G.PreModel x Salt B Cc)
    (Mc : ∀ x, G.Nonce → (Fin (G.K x) → G.Coin) → G.CoinModel x Salt B Cc) (nB : Msg → ℕ)
    (hnB : ∀ x ν c, (Mc x ν c).size ≤ nB x) (hL : G.MsgLink H R My cs Mp Mc) (hS : G.SimLink H Mc)
    (hN : G.NonDegenerate Mc) (hI : G.InnerHolds R Mc) : G.SHVZK H R fun x => 2 * nB x * δ₁.toReal := by
  intro x w hR ν c E
  exact CoinModel.shvzk H My cs hT1 hδ₁ ν c w E (Mc x ν c) ((hL x w hR ν).2 c) (hS x ν c) (hN x ν c) (hI x w hR ν c)
    (hnB x ν c)

/-- **`Binding` from the coin check** (`Proto.binding_of_tree`). -/
theorem binding_of_coinCheck (G : Proto W) (H : Msg → Msg) (h : G.CoinCheck) : G.Binding H := by
  obtain ⟨_, _, Tr, nb, root, parse, pos, idx, n, depth, dc, hchk, hdec⟩ := h
  exact G.binding_of_tree H Tr nb root parse pos idx n depth dc hchk hdec

end Proto

/-! ## Flock's zero knowledge from the models -/

/-- **Flock's zero knowledge against a malicious verifier, from the masked prover's models**: `Claims.Flock.ZKAt` at
`P_zk = G.prover` and `S_GK = G.simulatorE t (t·3^t)`, at `ε_zk = 8·n_L·δ₁ + 8t·n_B·δ₁ + 2·3^{-t}` and `γ = 4t·3^t`,
when the deployed verifier's rounds with `Hello` are the protocol's, under `Claims.Hm96Hiding` and the five named
hypotheses. -/
theorem Flock.zk_of_models {Coin Col T : Type} (H : Msg → Msg) (D : Flock.Deploy Coin Col T) (G : Proto Flock.Wit)
    (t : ℕ) (ht : 0 < t) (hzk : D.zk = true) (hk : ∀ x, D.V.k x + 1 = G.k x) {Salt B Cc : Type} [AddCommGroup Salt]
    [AddCommGroup B] [Fintype Salt] [DecidableEq Salt] [Fintype B] [DecidableEq B] (My : Salt → B) (cs : Salt → Cc)
    {δ₁ : ENNReal} (hT1 : Claims.Hm96Hiding My cs δ₁) (hδ₁ : δ₁ ≠ ⊤) (Mp : ∀ x, G.Nonce → G.PreModel x Salt B Cc)
    (Mc : ∀ x, G.Nonce → (Fin (G.K x) → G.Coin) → G.CoinModel x Salt B Cc) (nL nB : Msg → ℕ)
    (hnL : ∀ x ν, Fintype.card (Mp x ν).L ≤ nL x) (hnB : ∀ x ν c, (Mc x ν c).size ≤ nB x)
    (hL : G.MsgLink H (Flock.R D.parse) My cs Mp Mc) (hS : G.SimLink H Mc) (hN : G.NonDegenerate Mc)
    (hI : G.InnerHolds (Flock.R D.parse) Mc) (hC : G.CoinCheck) :
    Claims.Flock.ZKAt H D G.prover (G.simulatorE t (t * 3 ^ t))
      (fun x => 8 * nL x * δ₁.toReal + 8 * t * nB x * δ₁.toReal + 2 * (1 / 3) ^ t) fun _ => 4 * t * 3 ^ t := by
  obtain ⟨_, _, Tr, nb, root, parse, pos, idx, n, depth, dc, hchk, hdec⟩ := hC
  exact Flock.zkR_gkE_hm96 H D G t ht hzk hk δ₁ nL nB
    (G.prefix_of_models H _ My cs hT1 hδ₁ Mp Mc nL hnL hL)
    (G.shvzk_of_models H _ My cs hT1 hδ₁ Mp Mc nB hnB hL hS hN hI) Tr nb root parse pos idx n depth dc hchk hdec

/-- **Flock's zero knowledge against a malicious verifier at the deployed masked verifier** `FlockDeploy.deployZk`:
`Claims.Flock.ZKAt` at `P_zk = G.prover` and `S_GK = G.simulatorE t (t·3^t)`, at
`ε_zk = 8·n_L·δ₁ + 8t·n_B·δ₁ + 2·3^{-t}` and `γ = 4t·3^t`, for a committed-coin protocol whose coins are the deployed
verifier's
(`deployZk_proto_k`: `Hello` and `D.V`'s rounds are its), under `Claims.Hm96Hiding` and the five named hypotheses. -/
theorem Flock.zkAt_deployZk_of_models (H : Msg → Msg)
    (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (G : Proto Flock.Wit) (t : ℕ) (ht : 0 < t)
    (hK : ∀ x, G.K x = (FlockDeploy.V true).k x - 1) {Salt B Cc : Type} [AddCommGroup Salt] [AddCommGroup B]
    [Fintype Salt] [DecidableEq Salt] [Fintype B] [DecidableEq B] (My : Salt → B) (cs : Salt → Cc) {δ₁ : ENNReal}
    (hT1 : Claims.Hm96Hiding My cs δ₁) (hδ₁ : δ₁ ≠ ⊤) (Mp : ∀ x, G.Nonce → G.PreModel x Salt B Cc)
    (Mc : ∀ x, G.Nonce → (Fin (G.K x) → G.Coin) → G.CoinModel x Salt B Cc) (nL nB : Msg → ℕ)
    (hnL : ∀ x ν, Fintype.card (Mp x ν).L ≤ nL x) (hnB : ∀ x ν c, (Mc x ν c).size ≤ nB x)
    (hL : G.MsgLink H (Flock.R (FlockDeploy.deployZk rel).parse) My cs Mp Mc) (hS : G.SimLink H Mc)
    (hN : G.NonDegenerate Mc) (hI : G.InnerHolds (Flock.R (FlockDeploy.deployZk rel).parse) Mc) (hC : G.CoinCheck) :
    Claims.Flock.ZKAt H (FlockDeploy.deployZk rel) G.prover (G.simulatorE t (t * 3 ^ t))
      (fun x => 8 * nL x * δ₁.toReal + 8 * t * nB x * δ₁.toReal + 2 * (1 / 3) ^ t) fun _ => 4 * t * 3 ^ t :=
  Flock.zk_of_models H _ G t ht rfl (fun x => (deployZk_proto_k rel G hK x).symm) My cs hT1 hδ₁ Mp Mc nL nB hnL hnB
    hL hS hN hI hC

/-- **Flock's zero knowledge against a malicious verifier at the masked record of a committed-coin protocol**:
`Claims.Flock.ZK` at `zkOf rel G coinOf tapeOf (G.simulatorE t (t·3^t))`, the record whose reference prover is
`honestPZk` behind `G`'s `Hello` and opening checks, at `ε_zk = 8·n_L·δ₁ + 8t·n_B·δ₁ + 2·3^{-t}` and `γ = 4t·3^t`, under
`Claims.Hm96Hiding`, the five named hypotheses and `Proto.Honest` (`G`'s messages are `honestPZk`'s). -/
theorem Flock.zk_deployZk_of_models (H : Msg → Msg)
    (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (G : Proto Flock.Wit) (t : ℕ) (ht : 0 < t)
    (hK : ∀ x, G.K x = (FlockDeploy.V true).k x - 1) {Salt B Cc : Type} [AddCommGroup Salt] [AddCommGroup B]
    [Fintype Salt] [DecidableEq Salt] [Fintype B] [DecidableEq B] (My : Salt → B) (cs : Salt → Cc) {δ₁ : ENNReal}
    (hT1 : Claims.Hm96Hiding My cs δ₁) (hδ₁ : δ₁ ≠ ⊤) (Mp : ∀ x, G.Nonce → G.PreModel x Salt B Cc)
    (Mc : ∀ x, G.Nonce → (Fin (G.K x) → G.Coin) → G.CoinModel x Salt B Cc) (nL nB : Msg → ℕ)
    (hnL : ∀ x ν, Fintype.card (Mp x ν).L ≤ nL x) (hnB : ∀ x ν c, (Mc x ν c).size ≤ nB x)
    (hL : G.MsgLink H (Flock.R (FlockDeploy.deployZk rel).parse) My cs Mp Mc) (hS : G.SimLink H Mc)
    (hN : G.NonDegenerate Mc) (hI : G.InnerHolds (Flock.R (FlockDeploy.deployZk rel).parse) Mc) (hC : G.CoinCheck)
    (coinOf : Msg → Msg) (tapeOf : G.Nonce × G.Rand → FlockComplete.TapeZk)
    (hG : G.Honest H FlockComplete.honestPZk coinOf tapeOf) :
    Claims.Flock.ZK H (zkOf rel G coinOf tapeOf (G.simulatorE t (t * 3 ^ t)))
      (fun x => 8 * nL x * δ₁.toReal + 8 * t * nB x * δ₁.toReal + 2 * (1 / 3) ^ t) fun _ => 4 * t * 3 ^ t :=
  zk_zkOf rel G coinOf tapeOf _ (fun _ => by positivity) (deployZk_proto_k rel G hK) hG
    (Flock.zkAt_deployZk_of_models H rel G t ht hK My cs hT1 hδ₁ Mp Mc nL nB hnL hnB hL hS hN hI hC)

end Claims.ZKMal
