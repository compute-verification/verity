import Proofs.Flock.Claims.ZK.Prefix

/-!
# Special honest-verifier zero knowledge at every coin vector: `SHVZK` from Lemmas B and C

`SHVZK` asks, at every nonce and every coin vector `c`, that the honest prover's messages at `c` (`realVec`) and the
protocol's SHVZK simulator's (`sim`) be close. The session analysis has one result for each way the honest prover meets
`c`:

* **where it goes on** (the session's tables at `c` non-degenerate: `Rank`, `PadOnto`, each `W^r` a bijection,
  `PadsOnto`, `β ≠ 0`), Lemma B (`Session.session_shvzk_hm96`): the session's view is within `Σ_j 2·|Hid_j|·δ₁` of the
  tuple of the tables' `S_shvzk` outputs, for a witness that satisfies the inner constraint (`InnerHolds`).
  `Proto.shvzk_at_hm96`: `SHVZK` at `c` within `2·n_hid·δ₁`, when the honest prover's messages at `c` and the simulator's
  are one function of those views (`hR`, `hS`), on randomness read through maps with fibers of one size;
* **where it refuses** (it stops at a degenerate coin, as `flock-circuit --zk` does), its messages at `c` are a function
  of the session's pre-final view cut where it stops, which Lemma A (`Session.session_prefinal_indep_cut`) shows is
  witness-independent in world `W₀` at every coin vector; with real hm96 leaves (T1) the witness's messages at `c` are
  within `2·|L|·δ₁` of the dummy witness's. There the simulator sends the dummy-witness prover's messages at `c`
  (`Proto.dVec`): `Proto.shvzk_refused_hm96`, Lemma C at fixed answers (`oneRound`).

So `SHVZK` holds at `δ_B = 2·max(n_hid, |L|)·δ₁`.

**`InnerHolds` is needed, where the prover goes on.** `S_shvzk` computes `τ` from the verifier's replay `(λ, θ)` of the
simulated messages, `τ = (⟨λ, Y₁⟩ − θ)/β`, which is the real `τ = ⟨λ, μ⟩` only if the honest prover's pads `h` satisfy the
inner proof's constraint `⟨λ, h⟩ = θ` in every run (`Table.InnerHolds`: for every randomness and rep, the replay on the
prover's own messages gives `Σ_i λ_i·h_i = θ`). It holds because a valid witness's clear values pass the clear checks,
each of the ten constraints being affine in `h`; nothing in Lean proves it of the masked prover, so it stays a hypothesis
about the code. Where the prover refuses, the inner proof is not sent and nothing reads it; Lemma A needs no witness
property but the public region data (`hreg`).
-/

namespace Claims.ZKMal

open Claims

/-- A message as a strategy of one round, which reads no answer. -/
def oneRound {Ω M : Type} (f : Ω → M) : FlockSoundness.ZK.Strategy Ω M Unit 1 where
  msg ω _ _ := f ω
  causal _ _ _ _ _ := rfl

/-- The one-round responder. -/
def unitResp (M : Type) : FlockSoundness.ZK.Responder M Unit 1 := ⟨fun _ _ => (), fun _ _ _ _ => rfl⟩

theorem run_oneRound {Ω M : Type} [Inhabited M] (f : Ω → M) (V : FlockSoundness.ZK.Responder M Unit 1) (ω : Ω) :
    FlockSoundness.ZK.run (oneRound f) V ω 0 = f ω := rfl

namespace Proto

variable {W : Type}

/-- The dummy-witness prover's messages at the fixed coin vector `c`: after `j` coins, `dmsg` at `c`'s first `j`. -/
def dVec (G : Proto W) (H : Msg → Msg) (x : Msg) (ν : G.Nonce) (r : G.Rand) (c : Fin (G.K x) → G.Coin) : List Msg :=
  (List.range (G.K x + 1)).map fun j => ((G.dmsg x ν r ((List.ofFn c).take j)).eval H).1

/-- **`SHVZK` at a coin vector where the prover goes on, from Lemma B** (`Session.session_shvzk_hm96`, within
`2·n_hid·δ₁`): the session's tables `T` at `c` non-degenerate, with hm96 leaves `Mh` of one hiding pair, the witness's
table witnesses `wt` carrying the public region data and satisfying the inner constraint; the honest prover's messages at
`c` are `lift` of the session's real view (`hR`) and the simulator's are `lift` of the tables' `S_shvzk` outputs on the
dummy's `wt₀` (`hS`), each on randomness read through a map with fibers of one size; the session has at most `nH`
hidden leaves. -/
theorem shvzk_at_hm96 (G : Proto W) (H : Msg → Msg) (x : Msg) (w : W) (ν : G.Nonce) (c : Fin (G.K x) → G.Coin)
    {F K Wt Salt B D Hi VL Cc : Type} [Field F] [Field K] [CharP F 2] [AddCommGroup Salt] [AddCommGroup B]
    [AddCommGroup Hi] [Fintype F] [DecidableEq F] [Fintype K] [DecidableEq K] [Fintype Salt] [DecidableEq Salt]
    [Fintype B] [DecidableEq B] [Fintype Hi] [DecidableEq Hi] {J : ℕ}
    (T : Fin J → FlockSoundness.ZK.Table F K Wt Salt (B × Salt) D Hi VL) (Mh : ∀ j, (T j).Hm96 Cc) {δ₁ : ENNReal}
    (hT1 : ∀ j, Claims.Hm96Hiding (Mh j).My (Mh j).cs δ₁) (hδ₁ : δ₁ ≠ ⊤)
    (hrank : ∀ j, (T j).Rank) (hpad : ∀ j, (T j).PadOnto)
    (hW : ∀ j r, Function.Bijective fun p : F × F => (T j).Wr r p.1 p.2) (hpads : ∀ j, (T j).PadsOnto)
    (hβ : ∀ j r, (T j).β r ≠ 0) (wt wt₀ : Fin J → Wt) (hreg : ∀ j, (T j).sReg (wt j) = (T j).pubReg)
    (hinner : ∀ j, (T j).InnerHolds (wt j)) (lift : (∀ j, (T j).View) → List Msg)
    (eR : G.Rand → ∀ j, (T j).Rand × ((T j).Hid → Salt)) (kR : ℕ) (heR : ∀ p, Nat.card {r // eR r = p} = kR)
    (eS : G.SimRand x → ∀ j, (T j).SimRand × ((T j).Hid → Salt)) (kS : ℕ)
    (heS : ∀ p, Nat.card {s // eS s = p} = kS)
    (hR : ∀ r, G.realVec H x w ν r c = lift fun j => (Mh j).viewR (wt j) (eR r j).1 (eR r j).2)
    (hS : ∀ s, ((G.sim x ν c s).eval H).1 = lift fun j => (Mh j).simR (wt₀ j) (eS s j).1 (eS s j).2)
    {nH : ℕ} (hnH : ∑ j, Fintype.card (T j).Hid ≤ nH) (E : List Msg → Prop) :
    |Pr (fun r : G.Rand => E (G.realVec H x w ν r c)) - Pr (fun s : G.SimRand x => E ((G.sim x ν c s).eval H).1)|
      ≤ 2 * nH * δ₁.toReal := by
  have hb := FlockSoundness.ZK.Session.session_shvzk_hm96 T Mh hT1 hrank hpad hW hpads hβ wt hreg hinner wt₀
    (fun v => E (lift v))
  have : Nonempty (∀ j, (T j).SimRand × ((T j).Hid → Salt)) := ⟨eS (Classical.arbitrary _)⟩
  have : Nonempty (∀ j, (T j).Rand × ((T j).Hid → Salt)) := ⟨eR (Classical.arbitrary _)⟩
  have eReal : Pr (fun r : G.Rand => E (G.realVec H x w ν r c)) =
      (FlockSoundness.Game.prCoin fun p : ∀ j, (T j).Rand × ((T j).Hid → Salt) =>
        E (lift fun j => (Mh j).viewR (wt j) (p j).1 (p j).2)).toReal := by
    rw [← Pr_eq_prCoin, ← Pr_comp_fiber eR heR]
    simp only [hR]
  have eSim : Pr (fun s : G.SimRand x => E ((G.sim x ν c s).eval H).1) =
      (FlockSoundness.Game.prCoin fun p : ∀ j, (T j).SimRand × ((T j).Hid → Salt) =>
        E (lift fun j => (Mh j).simR (wt₀ j) (p j).1 (p j).2)).toReal := by
    rw [← Pr_eq_prCoin, ← Pr_comp_fiber eS heS]
    simp only [hS]
  have hle : ∑ j, 2 * (Fintype.card (T j).Hid : ENNReal) * δ₁ ≤ 2 * nH * δ₁ := by
    rw [← Finset.sum_mul, ← Finset.mul_sum]
    gcongr
    exact_mod_cast hnH
  have htop : (2 * nH * δ₁ : ENNReal) ≠ ⊤ :=
    ENNReal.mul_ne_top (ENNReal.mul_ne_top (by simp) (ENNReal.natCast_ne_top _)) hδ₁
  have hε : (∑ j, 2 * (Fintype.card (T j).Hid : ENNReal) * δ₁) ≠ ⊤ := ne_top_of_le_ne_top htop hle
  have hεr : (∑ j, 2 * (Fintype.card (T j).Hid : ENNReal) * δ₁).toReal ≤ 2 * nH * δ₁.toReal := by
    have := ENNReal.toReal_mono htop hle
    simpa [ENNReal.toReal_mul] using this
  have a := FlockSoundness.ZK.GK.toReal_le_add (FlockSoundness.ZK.prCoin_ne_top _) hε hb.1
  have b := FlockSoundness.ZK.GK.toReal_le_add (FlockSoundness.ZK.prCoin_ne_top _) hε hb.2
  rw [eReal, eSim, abs_le]
  constructor <;> linarith

/-- **`SHVZK` at a coin vector where the prover refuses, from Lemma C at fixed answers** (within `2·|L|·δ₁`): the
honest prover's messages at `c`, on the witness and on the dummy, are `msgs` of the session's leaves, each a real hm96
leaf `H(dig + M·y, c(y))` (`hR`, `hD`), whose world-`W₀` form is `out` of the session's pre-final view cut where the
prover stops at `c` (`hI`), on randomness read through maps with fibers of one size; the table witnesses carry the
public region data; and the simulator at `c` sends the dummy-witness prover's messages at `c` (`hS`). -/
theorem shvzk_refused_hm96 (G : Proto W) (H : Msg → Msg) (x : Msg) (w : W) (ν : G.Nonce) (c : Fin (G.K x) → G.Coin)
    {F K Wt Salt B D Hi VL Cc Ω L : Type} [Field F] [Field K] [CharP F 2] [AddCommGroup Salt] [AddCommGroup B]
    [AddCommGroup Hi] [Fintype F] [DecidableEq F] [Fintype Salt] [DecidableEq Salt] [Fintype B] [DecidableEq B]
    [Fintype Hi] [DecidableEq Hi] [Fintype Ω] [Nonempty Ω] [Fintype L] [DecidableEq L] {J : ℕ}
    (T : Fin J → FlockSoundness.ZK.Table F K Wt Salt (B × Salt) D Hi VL) (rep : ∀ j, Fin (T j).nC → Fin 2)
    (My : Salt → B) (cs : Salt → Cc) {δ₁ : ENNReal} (hT1 : Claims.Hm96Hiding My cs δ₁)
    (hδ₁ : δ₁ ≠ ⊤) (Hl : B × Cc → D) (msgs : (Fin J → Wt) → Ω → (L → D) → List Msg)
    (dig : (Fin J → Wt) → Ω → L → B) (e : Ω × (L → B × Salt) → ∀ j, (T j).Rand) (k : ℕ)
    (he : ∀ r, Nat.card {u // e u = r} = k) (out : (∀ j, (T j).Cut) → List Msg)
    (hI : ∀ wt ω u, msgs wt ω (fun l => Hl ((u l).1, cs (u l).2))
      = out fun j => (T j).cut (FlockSoundness.ZK.Session.sent T rep j) ((T j).preView (wt j) (e (ω, u) j)))
    (wt wt₀ : Fin J → Wt) (hreg : ∀ j, (T j).sReg (wt j) = (T j).pubReg)
    (hreg₀ : ∀ j, (T j).sReg (wt₀ j) = (T j).pubReg)
    (eR : G.Rand → Ω × (L → Salt)) (kR : ℕ) (heR : ∀ p, Nat.card {r // eR r = p} = kR)
    (hR : ∀ r, G.realVec H x w ν r c =
      msgs wt (eR r).1 fun l => Hl (dig wt (eR r).1 l + My ((eR r).2 l), cs ((eR r).2 l)))
    (hD : ∀ r, G.dVec H x ν r c =
      msgs wt₀ (eR r).1 fun l => Hl (dig wt₀ (eR r).1 l + My ((eR r).2 l), cs ((eR r).2 l)))
    (eS : G.SimRand x → G.Rand) (kS : ℕ) (heS : ∀ r, Nat.card {s // eS s = r} = kS)
    (hS : ∀ s, ((G.sim x ν c s).eval H).1 = G.dVec H x ν (eS s) c) {n : ℕ} (hn : Fintype.card L ≤ n)
    (E : List Msg → Prop) :
    |Pr (fun r : G.Rand => E (G.realVec H x w ν r c)) - Pr (fun s : G.SimRand x => E ((G.sim x ν c s).eval H).1)|
      ≤ 2 * n * δ₁.toReal := by
  have : Nonempty Salt := ⟨0⟩
  have h1 := FlockSoundness.ZK.Session.adaptive_prefinal_hm96 (n := 1) (Cn := Unit) (fun _ => T) (fun _ => rep) My cs
    hT1 Hl (fun wt' ω _ ls _ => msgs wt' ω ls) (fun wt' ω _ l => dig wt' ω l)
    (fun wt' => oneRound fun p => msgs wt' p.1 fun l => Hl (dig wt' p.1 l + (p.2 l).1, (p.2 l).2))
    (fun _ _ _ _ => rfl) (fun _ => e) (fun _ => k) (fun _ => he) (fun _ cut _ => out cut)
    (fun wt' ω u _ => funext fun _ => hI wt' ω u) wt wt₀ (fun _ => hreg) (fun _ => hreg₀) (unitResp _)
    (fun τ => E (τ 0))
  have h2 := FlockSoundness.ZK.Session.adaptive_prefinal_hm96 (n := 1) (Cn := Unit) (fun _ => T) (fun _ => rep) My cs
    hT1 Hl (fun wt' ω _ ls _ => msgs wt' ω ls) (fun wt' ω _ l => dig wt' ω l)
    (fun wt' => oneRound fun p => msgs wt' p.1 fun l => Hl (dig wt' p.1 l + (p.2 l).1, (p.2 l).2))
    (fun _ _ _ _ => rfl) (fun _ => e) (fun _ => k) (fun _ => he) (fun _ cut _ => out cut)
    (fun wt' ω u _ => funext fun _ => hI wt' ω u) wt₀ wt (fun _ => hreg₀) (fun _ => hreg) (unitResp _)
    (fun τ => E (τ 0))
  simp only [run_oneRound] at h1 h2
  have eW : Pr (fun r : G.Rand => E (G.realVec H x w ν r c)) =
      (FlockSoundness.Game.prCoin fun p : Ω × (L → Salt) =>
        E (msgs wt p.1 fun l => Hl (dig wt p.1 l + My (p.2 l), cs (p.2 l)))).toReal := by
    rw [← Pr_eq_prCoin, ← Pr_comp_fiber eR heR]
    simp only [hR]
  have eD : Pr (fun s : G.SimRand x => E ((G.sim x ν c s).eval H).1) =
      (FlockSoundness.Game.prCoin fun p : Ω × (L → Salt) =>
        E (msgs wt₀ p.1 fun l => Hl (dig wt₀ p.1 l + My (p.2 l), cs (p.2 l)))).toReal := by
    rw [← Pr_eq_prCoin, ← Pr_comp_fiber eR heR, ← Pr_comp_fiber eS heS]
    simp only [hS, hD]
  have hε : 2 * (Fintype.card L : ENNReal) * δ₁ ≠ ⊤ :=
    ENNReal.mul_ne_top (ENNReal.mul_ne_top (by simp) (ENNReal.natCast_ne_top _)) hδ₁
  have hεr : (2 * (Fintype.card L : ENNReal) * δ₁).toReal = 2 * Fintype.card L * δ₁.toReal := by
    simp [ENNReal.toReal_mul]
  have a := FlockSoundness.ZK.GK.toReal_le_add (FlockSoundness.ZK.prCoin_ne_top _) hε h1
  have b := FlockSoundness.ZK.GK.toReal_le_add (FlockSoundness.ZK.prCoin_ne_top _) hε h2
  rw [hεr] at a b
  have hn' : 2 * (Fintype.card L : ℝ) * δ₁.toReal ≤ 2 * n * δ₁.toReal := by
    have : (Fintype.card L : ℝ) ≤ n := by exact_mod_cast hn
    have := ENNReal.toReal_nonneg (a := δ₁)
    nlinarith
  rw [eW, eD, abs_le]
  constructor <;> linarith

end Proto

end Claims.ZKMal
