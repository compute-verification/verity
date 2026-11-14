module

public import Proofs.Flock.SampledGeneric.Zk
public import Proofs.Flock.Claims.ZK.SessionLean

/-!
# The sampled audit's zero knowledge at Flock's ZK server

Flock's ZK server (`Claims.ZKServer.Flock.zk_server_lean_loop`) releases the session's transcript, scheduled, within a
factor `2` of `leaky1`'s, the simulator that guesses whether the session stops: one bit a session. Its bound is one-sided,
`Pr_real[E] ≤ 2·(Pr_sim[E] + ε + 4t·3^t·Pr[Fork])`, and its error reads the witness and the verifier through `Fork`, the
real game's collision finder. `Zk.ZK` asks for more: the two views `ε_zk(x)`-close both ways, at an `ε_zk` of the
statement alone. So the server is a `Zk` (`FlockZk.server`), but what it meets is the one-sided form at the factor `2`
(`ZKBelow`), and the audit's zero knowledge at Flock is `GenericSampledZK`'s argument in that form:

* `Below`, `ZKBelow`: one experiment `a`-below another at `ε` on every event; zero knowledge with a leak at the factor `a`,
  at an error that may read the statement, the witness and the verifier. `ZKBelow.of_ZK`: `Zk.ZK` is its case `a = 1`.
* `Below.coin`, `Below.trans_close`, `audit_below`: the two hybrids of `GenericSampledZK` in that form: the audit's view is `a`-below the
  audit's simulator (`auditSim`) at `a·ε_hide` plus the server's error averaged over the prover's commitment coins.
* `FlockZk.server`: the ZK server as a `Zk`: the honest view is the scheduled transcript of the deferred server
  (`ZkServer.proverD`) against a verifier of the protocol's rounds (`FlockZk.Verifier`), the simulator's is `leaky1`'s.
* `FlockZk.server_below`, `FlockZk.server_zkBelow`: `zk_server_lean_loop` in `Exp`'s terms, at the relation `LiveRel`, the
  statement's relation together with the live path's premises at the statement.
* `FlockZk.audit_below`: the Flock instance, with every premise of `zk_server_lean_loop` carried as it states it or as
  one that gives it.
-/

@[expose] public section

namespace FlockSoundness.SampledGeneric

open Game
open scoped ENNReal

/-! ## One-sided closeness at a factor -/

/-- **`e` is `a`-below `f` at `ε`**: on every event, `e`'s probability is at most `a` times `f`'s plus `ε`. `Close` at
`ε` is `1`-below at `ε` both ways. -/
def Below {V : Type} (e f : Exp V) (a ε : ℝ≥0∞) : Prop :=
  ∀ E : V → Prop, e.pr E ≤ a * f.pr E + ε

/-- **Zero knowledge with a leak, at the factor `a`**: for every statement and witness in the relation and every
verifier, the honest prover's view is `a`-below the simulator's at `ε x w v`, which may read the witness and the verifier
(a collision finder's success built from them). -/
def ZKBelow {Stmt Wit VS View : Type} (Zs : Zk Stmt Wit VS View) (Rel : Stmt → Wit → Prop) (a : ℝ≥0∞)
    (ε : Stmt → Wit → VS → ℝ≥0∞) : Prop :=
  ∀ x w, Rel x w → ∀ v, Below (Zs.real x w v) (Zs.sim x v) a (ε x w v)

/-- `Zk.ZK` at `ε_zk` is `ZKBelow` at the factor `1` and the error `ε_zk(x)`. -/
theorem ZKBelow.of_ZK {Stmt Wit VS View : Type} {Zs : Zk Stmt Wit VS View} {Rel : Stmt → Wit → Prop}
    {εzk : Stmt → ℝ≥0∞} (h : Zs.ZK Rel εzk) : ZKBelow Zs Rel 1 fun x _ _ => εzk x := by
  intro x w hR v E
  rw [one_mul]
  exact (h x w hR v E).1

/-- **The proof system on other statements**: `Z` at the statement `g x`. -/
def zkComap {Stmt Stmt' Wit VS View : Type} (Zs : Zk Stmt Wit VS View) (g : Stmt' → Stmt) : Zk Stmt' Wit VS View :=
  ⟨fun x => Zs.real (g x), fun x => Zs.sim (g x)⟩

theorem ZKBelow.comap {Stmt Stmt' Wit VS View : Type} {Zs : Zk Stmt Wit VS View} {Rel : Stmt → Wit → Prop}
    {a : ℝ≥0∞} {ε : Stmt → Wit → VS → ℝ≥0∞} (h : ZKBelow Zs Rel a ε) (g : Stmt' → Stmt) :
    ZKBelow (zkComap Zs g) (fun x w => Rel (g x) w) a fun x => ε (g x) :=
  fun x w hR v => h (g x) w hR v

theorem avg_const_mul {C : Type} [Fintype C] (x : ℝ≥0∞) (f : C → ℝ≥0∞) :
    avg (fun c => x * f c) = x * avg f := by
  simp only [avg, Finset.mul_sum]
  exact Finset.sum_congr rfl fun _ _ => by ring

/-- **One-sided closeness under a uniform coin**: if each coin's experiment is `α`-below the other's at `ε r`, so is
the experiment that flips the coin first and relabels each view by the coin, at `ε` averaged (`Close.coin`'s one-sided
form). -/
theorem Below.coin {Rnd V W : Type} [Fintype Rnd] [Nonempty Rnd] (a b : Rnd → Exp V) (f : Rnd → V → W)
    {α : ℝ≥0∞} (ε : Rnd → ℝ≥0∞) (h : ∀ r, Below (a r) (b r) α (ε r)) :
    Below ⟨.coin Rnd fun r => (a r).g.map (f r), fun r => Strategy.map _ _ (a r).s⟩
      ⟨.coin Rnd fun r => (b r).g.map (f r), fun r => Strategy.map _ _ (b r).s⟩ α (avg ε) := by
  have key : ∀ c : Rnd → Exp V, ∀ E : W → Prop,
      Exp.pr ⟨.coin Rnd fun r => (c r).g.map (f r), fun r => Strategy.map _ _ (c r).s⟩ E =
        avg fun r => (c r).pr fun v => E (f r v) := fun c E =>
    congrArg avg (funext fun r => prob_map_strat E (f r) (c r).g (c r).s)
  intro E
  rw [key a, key b, ← avg_const_mul, ← avg_add]
  exact avg_mono fun r => h r fun v => E (f r v)

/-- **`a`-below, then `ε'`-close**: `a`-below at `ε`, then the second experiment `ε'`-close to a third, is `a`-below the
third at `a·ε' + ε`. -/
theorem Below.trans_close {V : Type} {e f g : Exp V} {a ε ε' : ℝ≥0∞} (h₁ : Below e f a ε) (h₂ : Close f g ε') :
    Below e g a (a * ε' + ε) := fun E =>
  calc e.pr E ≤ a * f.pr E + ε := h₁ E
    _ ≤ a * (g.pr E + ε') + ε := by gcongr; exact (h₂ E).1
    _ = a * g.pr E + (a * ε' + ε) := by rw [mul_add, add_assoc]

/-- **The sampled audit's zero knowledge with a leak**, `GenericSampledZK`'s two hybrids at the factor `a`: if the proof
system is zero knowledge with a leak at the factor `a` and the error `ε` (`ZKBelow`), and the commitment is
`ε_hide`-hiding, then against every verifier `v`, which samples after seeing the commitment, the audit's view at any trace
`Xt` whose openings are witnesses of every sample's statement is `a`-below the simulator's at any fixed trace `X₀`
(`auditSim`), at `a·ε_hide` plus `ε` averaged over the prover's commitment coins. The middle hybrid is `auditSim` at
`Xt` (`Below.coin`), then `Hiding` moves it to `X₀` (`auditSim_eq_andThen`). -/
theorem audit_below {n : ℕ} {Tr Rnd Com Wit VS View : Type} [Fintype Rnd] [Nonempty Rnd] (H : Hider Tr Rnd Com)
    (Zs : Zk (Com × Finset (Fin n)) Wit VS View) (Rel : Com × Finset (Fin n) → Wit → Prop) (a εh : ℝ≥0∞)
    (ε : Com × Finset (Fin n) → Wit → VS → ℝ≥0∞) (samp : VS → Com → Finset (Fin n))
    (wit : Tr → Rnd → Finset (Fin n) → Wit) (hZ : ZKBelow Zs Rel a ε) (hH : H.Hiding εh) (Xt : Tr)
    (hX : ∀ r S, Rel (H.commit Xt r, S) (wit Xt r S)) (X₀ : Tr) (v : VS) :
    Below (auditView H Zs samp wit Xt v) (auditSim H Zs samp X₀ v) a
      (a * εh + avg fun r => ε (H.commit Xt r, samp v (H.commit Xt r)) (wit Xt r (samp v (H.commit Xt r))) v) := by
  have h1 : Below (auditView H Zs samp wit Xt v) (auditSim H Zs samp Xt v) a
      (avg fun r => ε (H.commit Xt r, samp v (H.commit Xt r)) (wit Xt r (samp v (H.commit Xt r))) v) :=
    Below.coin (fun r => Zs.real (H.commit Xt r, samp v (H.commit Xt r)) (wit Xt r (samp v (H.commit Xt r))) v)
      (fun r => Zs.sim (H.commit Xt r, samp v (H.commit Xt r)) v)
      (fun r w => (H.commit Xt r, samp v (H.commit Xt r), w)) _ fun r => hZ _ _ (hX r _) v
  have h2 : Close (auditSim H Zs samp Xt v) (auditSim H Zs samp X₀ v) εh := by
    rw [auditSim_eq_andThen, auditSim_eq_andThen]
    exact hH _ Xt X₀ (auditSimAfter Zs samp v)
  exact h1.trans_close h2

/-- **A uniform coin's experiment, in the claims' terms**: the probability of an event of a view sampled from uniform
coins is the claims' `Pr` of it. -/
theorem pr_uniform {C V : Type} [Fintype C] [Nonempty C] (f : C → V) (E : V → Prop) :
    Exp.pr (⟨.coin C fun c => .ret (f c), fun _ => PUnit.unit⟩ : Exp V) E =
      ENNReal.ofReal (Claims.Pr fun c => E (f c)) := by
  rw [Claims.ZKMal.Pr_eq_prCoin, ENNReal.ofReal_toReal (FlockSoundness.ZK.prCoin_ne_top _)]
  rfl

end FlockSoundness.SampledGeneric

/-! ## Flock's ZK server -/

namespace FlockSoundness.SampledGeneric.FlockZk

open Game
open scoped ENNReal

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-- The relation a parsed statement names, as the masked deployment reads it. -/
abbrev Parser : Type := _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation

/-- The hash the server's runs call: SHA-512 on messages. -/
abbrev sha : Claims.Msg → Claims.Msg := Claims.ZKMal.CoinOpen.shaH

/-- The ZK server's session at the tables `T`: the coin layer's, at the coin counts the session loop requests. -/
abbrev Sess (T : Claims.ZKMal.CoinLayer.TableData) : Claims.ZKMal.RefSession :=
  Claims.ZKMal.CoinLayer.session Claims.ZKServer.Loop.loopNr T

/-- The server's tape: the session's nonce and the rest of the reference prover's tape. -/
abbrev Tape (T : Claims.ZKMal.CoinLayer.TableData) : Type := (Sess T).refP.Nonce × (Sess T).refP.Rand

/-- The release at the server's tape. -/
abbrev Release (T : Claims.ZKMal.CoinLayer.TableData) : Type := Claims.ZKServer.LeanRelease Claims.Flock.Wit (Tape T)

/-- The transport's frame state at each statement and coins (`zk_server_lean_loop`'s `pc`). -/
abbrev PC (T : Claims.ZKMal.CoinLayer.TableData) : Type :=
  Claims.Msg → List (Sess T).Coin → _root_.Flock.Firewall.Frame.Masker

/-- The transport's checked frame state (`zk_server_lean_loop`'s `fc`). -/
abbrev FC (T : Claims.ZKMal.CoinLayer.TableData) : Type :=
  Claims.Msg → List (Sess T).Coin → Except String _root_.Flock.Firewall.Frame.Masker

/-- The session with the release's stops, which the server's simulator and its error read. -/
abbrev Stops (T : Claims.ZKMal.CoinLayer.TableData) (L : Release T) (pc : PC T) (fc : FC T) :
    Claims.ZKMal.RefSession :=
  (Sess T).withStops (Claims.ZKServer.LeanRelease.rfOf (Sess T) L pc fc Claims.ZKServer.Loop.readCoins) L.stop

/-- The server's rounds at `x`: the masked verifier's and the `Hello`. -/
noncomputable abbrev rounds (rel : Parser) : Claims.Msg → ℕ := fun x => (FlockDeploy.deployZk rel).V.k x + 1

/-- **A verifier of the server's rounds**: a malicious verifier `V*` and its tape `θ`, whose round count at every
statement is the deployed masked verifier's plus the `Hello` (`zk_server_lean_loop`'s `hk`, at every statement). -/
structure Verifier (rel : Parser) (Θ : Type) where
  Vs : Claims.VStar Θ
  θ : Θ
  hk : ∀ x, Vs.k x = rounds rel x

/-- **The server's simulator**: `leaky1` of the session's estimating Goldreich–Kahan simulator, at the threshold `t`
and the cap `t·3^t`; it guesses whether the session stops. -/
noncomputable abbrev sim (rel : Parser) (T : Claims.ZKMal.CoinLayer.TableData) (L : Release T) (pc : PC T) (fc : FC T)
    (t : ℕ) : Claims.Simulator :=
  Claims.ZKServer.leaky1
    (((Stops T L pc fc).refP.withSim fun x _ c => (Stops T L pc fc).coinModel x c).simulatorE t (t * 3 ^ t))
    (rounds rel) L.toSrv.stop

/-- The honest view: the deferred server's transcript with the worker `wk` (`ZkServer.proverD`) against `v`'s `V*`,
scheduled, at the server's tape and check coins `ω`. -/
noncomputable def realView (rel : Parser) (T : Claims.ZKMal.CoinLayer.TableData) (L : Release T) (wk : L.toSrv.Worker)
    {Θ : Type} (x : Claims.Msg) (w : Claims.Flock.Wit) (v : Verifier rel Θ) (ω : Tape T × L.toSrv.Chk) :
    List ((Claims.Msg × Claims.Out) × ℕ) :=
  L.toSrv.timed x (Claims.runStar sha (L.toSrv.proverD L.fillOf (rounds rel) wk w) v.Vs x ω v.θ).msgs

/-- The simulator's view: `sim`'s transcript against `v`'s `V*` at its coins `c`, scheduled the same way. -/
noncomputable def simView (rel : Parser) (T : Claims.ZKMal.CoinLayer.TableData) (L : Release T) (pc : PC T)
    (fc : FC T) (t : ℕ) {Θ : Type} (x : Claims.Msg) (v : Verifier rel Θ) (c : (sim rel T L pc fc t).C x) :
    List ((Claims.Msg × Claims.Out) × ℕ) :=
  L.toSrv.timed x ((sim rel T L pc fc t).out sha v.Vs v.θ x c)

/-- **The ZK server as a `Zk`**, at the worker `wk`: the honest view over the server's tape and check coins
(`realView`), the simulator's over its coins (`simView`). -/
noncomputable def server (rel : Parser) (T : Claims.ZKMal.CoinLayer.TableData) (L : Release T) (wk : L.toSrv.Worker)
    (pc : PC T) (fc : FC T) (t : ℕ) (Θ : Type) :
    Zk Claims.Msg Claims.Flock.Wit (Verifier rel Θ) (List ((Claims.Msg × Claims.Out) × ℕ)) where
  real x w v := ⟨.coin (Tape T × L.toSrv.Chk) fun ω => .ret (realView rel T L wk x w v ω), fun _ => PUnit.unit⟩
  sim x v := ⟨.coin ((sim rel T L pc fc t).C x) fun c => .ret (simView rel T L pc fc t x v c), fun _ => PUnit.unit⟩

theorem server_real_pr (rel : Parser) (T : Claims.ZKMal.CoinLayer.TableData) (L : Release T) (wk : L.toSrv.Worker)
    (pc : PC T) (fc : FC T) (t : ℕ) (Θ : Type) (x : Claims.Msg) (w : Claims.Flock.Wit) (v : Verifier rel Θ)
    (E : List ((Claims.Msg × Claims.Out) × ℕ) → Prop) :
    ((server rel T L wk pc fc t Θ).real x w v).pr E =
      ENNReal.ofReal (Claims.Pr fun ω => E (realView rel T L wk x w v ω)) :=
  pr_uniform _ E

theorem server_sim_pr (rel : Parser) (T : Claims.ZKMal.CoinLayer.TableData) (L : Release T) (wk : L.toSrv.Worker)
    (pc : PC T) (fc : FC T) (t : ℕ) (Θ : Type) (x : Claims.Msg) (v : Verifier rel Θ)
    (E : List ((Claims.Msg × Claims.Out) × ℕ) → Prop) :
    ((server rel T L wk pc fc t Θ).sim x v).pr E =
      ENNReal.ofReal (Claims.Pr fun c => E (simView rel T L pc fc t x v c)) :=
  pr_uniform _ E

/-- **The server's error** at `(x, w)` against `v`: `2·(8·N·δ₁ + 8t·N·δ₁ + 2·3^{-t} + 4t·3^t·Pr[Fork])`, with `N` the
session's leaves at `x` and `Fork` the real game's collision finder, two runs of the switched prover at `w` against `v`'s
`V*` on one nonce: `zk_server_lean_loop`'s bound less `2·Pr_sim`. -/
noncomputable def err (rel : Parser) (T : Claims.ZKMal.CoinLayer.TableData) (vs : (Sess T).SwVals) (L : Release T)
    (pc : PC T) (fc : FC T) (t : ℕ) (δ₁ : ℝ≥0∞) {Θ : Type} (x : Claims.Msg) (w : Claims.Flock.Wit)
    (v : Verifier rel Θ) : ℝ≥0∞ :=
  2 * ENNReal.ofReal ((8 * (Stops T L pc fc).nLeaves x * δ₁.toReal +
      8 * t * (Stops T L pc fc).nLeaves x * δ₁.toReal + 2 * (1 / 3) ^ t) +
    4 * t * 3 ^ t * (Claims.Fork sha (((Stops T L pc fc).swP vs).prover w) v.Vs v.θ x).Pr sha)

/-- **The live path's premises at `(x, w)`**, each `zk_server_lean_loop`'s or one that gives it: the transport's frame
states (`ProgReads`), and the link, final, live and fill rounds' releases being the reference's (`LinkRoundEq`,
`FinalRoundEq`, `LiveRoundEq`, `FillRoundEq`). `LiveRoundEq` holds at every history and gives `LiveRoundEqOwn`, the
worker's own (`LiveRoundEq.own`). -/
abbrev Live (T : Claims.ZKMal.CoinLayer.TableData) (vs : (Sess T).SwVals) (L : Release T) (wk : L.toSrv.Worker)
    (pc : PC T) (fc : FC T) (x : Claims.Msg) (w : Claims.Flock.Wit) : Prop :=
  Claims.ZKServer.LeanRelease.ProgReads (Sess T) L pc fc x ∧
    Claims.ZKServer.LeanRelease.LinkRoundEq (Sess T) vs L x w ∧
    Claims.ZKServer.LeanRelease.FinalRoundEq (Sess T) vs L sha wk x w ∧
    Claims.ZKServer.LeanRelease.LiveRoundEq (Sess T) vs L sha wk x w ∧
    Claims.ZKServer.LeanRelease.FillRoundEq (Sess T) vs L sha wk x w

/-- **The relation the server is zero knowledge at**: the statement's (`Claims.Flock.R` at the masked deployment's
parser) with the live path's premises at the statement (`Live`). -/
abbrev LiveRel (rel : Parser) (T : Claims.ZKMal.CoinLayer.TableData) (vs : (Sess T).SwVals) (L : Release T)
    (wk : L.toSrv.Worker) (pc : PC T) (fc : FC T) (x : Claims.Msg) (w : Claims.Flock.Wit) : Prop :=
  Claims.Flock.R (FlockDeploy.deployZk rel).parse x w ∧ Live T vs L wk pc fc x w

/-- **`zk_server_lean_loop` in `Exp`'s terms**: at a statement in the relation with its live path's premises
(`LiveRel`), the honest view is `2`-below the simulator's at the server's error, against every verifier of the server's
rounds. The tables' side (`hS`, `hV`, `hD`, `hc`, `hsc`, `hT`), hm96's hiding (`hT1`, `hδ₁`) and the release at the
session loop (`hL`) are the theorem's premises or give them: `SwCausal` gives `SwCausalOn` at every coin vector
(`swCausalOn_of_swCausal`), which every stop rule's `StopsOk` takes in (`stopsOk_true`). -/
theorem server_below (rel : Parser) (T : Claims.ZKMal.CoinLayer.TableData) (vs : (Sess T).SwVals) (t : ℕ)
    (ht : 0 < t) {δ₁ : ℝ≥0∞}
    (hT1 : Claims.Hm96Hiding FlockSoundness.Discharge.ZkView.My FlockSoundness.Discharge.ZkView.saltHash δ₁)
    (hδ₁ : δ₁ ≠ ⊤) (hS : (Sess T).Tables) (hV : (Sess T).Complete (Claims.Flock.R (FlockDeploy.deployZk rel).parse))
    (hD : (Sess T).DummyReg) (hc : (Sess T).Causal) (hsc : (Sess T).SwCausal vs) (hT : (Sess T).TapeOK)
    (L₀ L : Release T) (hL : L = Claims.ZKServer.LeanRelease.loop Claims.ZKServer.Loop.loopNr T L₀)
    (wk : L.toSrv.Worker) (pc : PC T) (fc : FC T) (Θ : Type) (x : Claims.Msg) (w : Claims.Flock.Wit)
    (hR : LiveRel rel T vs L wk pc fc x w) (v : Verifier rel Θ) :
    Below ((server rel T L wk pc fc t Θ).real x w v) ((server rel T L wk pc fc t Θ).sim x v) 2
      (err rel T vs L pc fc t δ₁ x w v) := by
  obtain ⟨hR, hPr, hLk, hF, hE, hEf⟩ := hR
  intro E
  have h := Claims.ZKServer.Flock.zk_server_lean_loop rel T vs t ht hT1 hδ₁ hS hV hD hc (fun _ _ => True)
    ((Sess T).swCausalOn_of_swCausal vs _ hsc) hT L₀ L hL wk x w hR pc fc hPr ((Sess T).stopsOk_true _) hLk hF hE.own
    hEf v.Vs v.θ (v.hk x) E
  rw [server_real_pr, server_sim_pr, err]
  refine (ENNReal.ofReal_le_ofReal (h.trans_eq (congrArg (fun z => 2 * z) (add_assoc _ _ _)))).trans ?_
  rw [ENNReal.ofReal_mul (by norm_num), ENNReal.ofReal_ofNat, ← mul_add]
  gcongr
  exact ENNReal.ofReal_add_le

/-- **The ZK server is zero knowledge with a one-bit leak** (`ZKBelow` at the factor `2`) at the relation `LiveRel` and
the server's error, under `zk_server_lean_loop`'s premises that don't read the statement. -/
theorem server_zkBelow (rel : Parser) (T : Claims.ZKMal.CoinLayer.TableData) (vs : (Sess T).SwVals) (t : ℕ)
    (ht : 0 < t) {δ₁ : ℝ≥0∞}
    (hT1 : Claims.Hm96Hiding FlockSoundness.Discharge.ZkView.My FlockSoundness.Discharge.ZkView.saltHash δ₁)
    (hδ₁ : δ₁ ≠ ⊤) (hS : (Sess T).Tables) (hV : (Sess T).Complete (Claims.Flock.R (FlockDeploy.deployZk rel).parse))
    (hD : (Sess T).DummyReg) (hc : (Sess T).Causal) (hsc : (Sess T).SwCausal vs) (hT : (Sess T).TapeOK)
    (L₀ L : Release T) (hL : L = Claims.ZKServer.LeanRelease.loop Claims.ZKServer.Loop.loopNr T L₀)
    (wk : L.toSrv.Worker) (pc : PC T) (fc : FC T) (Θ : Type) :
    ZKBelow (server rel T L wk pc fc t Θ) (LiveRel rel T vs L wk pc fc) 2
      (fun x w v => err rel T vs L pc fc t δ₁ x w v) :=
  fun x w hR v => server_below rel T vs t ht hT1 hδ₁ hS hV hD hc hsc hT L₀ L hL wk pc fc Θ x w hR v

/-- **The sampled audit's zero knowledge at Flock's ZK server.** The audit commits to the trace `Xt` with the commitment
`H` (`ε_hide`-hiding), the verifier `v` samples after seeing the commitment (`samp`), and the ZK server proves the
sample's statement `stmtOf (c, S)` from the trace's openings (`wit`). If every such statement is in the masked
deployment's relation (`hR`) and meets the live path's premises there (`hPr`, `hLk`, `hF`, `hE`, `hEf`), the audit's
view is `2`-below the simulator's at any fixed trace `X₀` (`auditSim`, at the server's simulator), at `2·ε_hide` plus
the server's error averaged over the prover's commitment coins:
$$\Pr_{\rm audit}[E]\le 2\Pr_{\rm sim}[E]+2\varepsilon_{\rm hide}+\mathbb E_r[\mathrm{err}(x_r,w_r,v)].$$
Every other premise is `zk_server_lean_loop`'s or gives it, as at `server_below`. -/
theorem audit_below (rel : Parser) (T : Claims.ZKMal.CoinLayer.TableData) (vs : (Sess T).SwVals) (t : ℕ)
    (ht : 0 < t) {δ₁ : ℝ≥0∞}
    (hT1 : Claims.Hm96Hiding FlockSoundness.Discharge.ZkView.My FlockSoundness.Discharge.ZkView.saltHash δ₁)
    (hδ₁ : δ₁ ≠ ⊤) (hS : (Sess T).Tables) (hV : (Sess T).Complete (Claims.Flock.R (FlockDeploy.deployZk rel).parse))
    (hD : (Sess T).DummyReg) (hc : (Sess T).Causal) (hsc : (Sess T).SwCausal vs) (hT : (Sess T).TapeOK)
    (L₀ L : Release T) (hL : L = Claims.ZKServer.LeanRelease.loop Claims.ZKServer.Loop.loopNr T L₀)
    (wk : L.toSrv.Worker) (pc : PC T) (fc : FC T) (Θ : Type)
    {n : ℕ} {Tr Rnd Com : Type} [Fintype Rnd] [Nonempty Rnd] (H : Hider Tr Rnd Com) {εh : ℝ≥0∞}
    (hH : H.Hiding εh) (stmtOf : Com × Finset (Fin n) → Claims.Msg)
    (samp : Verifier rel Θ → Com → Finset (Fin n)) (wit : Tr → Rnd → Finset (Fin n) → Claims.Flock.Wit) (Xt : Tr)
    (hR : ∀ r S, Claims.Flock.R (FlockDeploy.deployZk rel).parse (stmtOf (H.commit Xt r, S)) (wit Xt r S))
    (hPr : ∀ r S, Claims.ZKServer.LeanRelease.ProgReads (Sess T) L pc fc (stmtOf (H.commit Xt r, S)))
    (hLk : ∀ r S, Claims.ZKServer.LeanRelease.LinkRoundEq (Sess T) vs L (stmtOf (H.commit Xt r, S)) (wit Xt r S))
    (hF : ∀ r S,
      Claims.ZKServer.LeanRelease.FinalRoundEq (Sess T) vs L sha wk (stmtOf (H.commit Xt r, S)) (wit Xt r S))
    (hE : ∀ r S,
      Claims.ZKServer.LeanRelease.LiveRoundEq (Sess T) vs L sha wk (stmtOf (H.commit Xt r, S)) (wit Xt r S))
    (hEf : ∀ r S,
      Claims.ZKServer.LeanRelease.FillRoundEq (Sess T) vs L sha wk (stmtOf (H.commit Xt r, S)) (wit Xt r S))
    (X₀ : Tr) (v : Verifier rel Θ) :
    Below (auditView H (zkComap (server rel T L wk pc fc t Θ) stmtOf) samp wit Xt v)
      (auditSim H (zkComap (server rel T L wk pc fc t Θ) stmtOf) samp X₀ v) 2
      (2 * εh + avg fun r => err rel T vs L pc fc t δ₁ (stmtOf (H.commit Xt r, samp v (H.commit Xt r)))
        (wit Xt r (samp v (H.commit Xt r))) v) :=
  _root_.FlockSoundness.SampledGeneric.audit_below H (zkComap (server rel T L wk pc fc t Θ) stmtOf)
    (fun s w => LiveRel rel T vs L wk pc fc (stmtOf s) w) 2 εh
    (fun s w v => err rel T vs L pc fc t δ₁ (stmtOf s) w v) samp wit
    ((server_zkBelow rel T vs t ht hT1 hδ₁ hS hV hD hc hsc hT L₀ L hL wk pc fc Θ).comap stmtOf) hH Xt
    (fun r S => ⟨hR r S, hPr r S, hLk r S, hF r S, hE r S, hEf r S⟩) X₀ v

end FlockSoundness.SampledGeneric.FlockZk
