module

public import Proofs.Flock.SampledGeneric.Zk
public import Proofs.Flock.Claims.ZK.SessionLean

/-! The sampled audit's zero knowledge at Flock's ZK server, in the one-sided form at the factor `2` the server meets. -/

@[expose] public section

namespace FlockSoundness.SampledGeneric

open Game
open scoped ENNReal

def Below {V : Type} (e f : Exp V) (a ε : ℝ≥0∞) : Prop :=
  ∀ E : V → Prop, e.pr E ≤ a * f.pr E + ε

/-- The error may read the witness and the verifier (a collision finder's success built from them). -/
def ZKBelow {Stmt Wit VS View : Type} (Zs : Zk Stmt Wit VS View) (Rel : Stmt → Wit → Prop) (a : ℝ≥0∞)
    (ε : Stmt → Wit → VS → ℝ≥0∞) : Prop :=
  ∀ x w, Rel x w → ∀ v, Below (Zs.real x w v) (Zs.sim x v) a (ε x w v)

theorem ZKBelow.of_ZK {Stmt Wit VS View : Type} {Zs : Zk Stmt Wit VS View} {Rel : Stmt → Wit → Prop}
    {εzk : Stmt → ℝ≥0∞} (h : Zs.ZK Rel εzk) : ZKBelow Zs Rel 1 fun x _ _ => εzk x := by
  intro x w hR v E
  rw [one_mul]
  exact (h x w hR v E).1

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

theorem Below.trans_close {V : Type} {e f g : Exp V} {a ε ε' : ℝ≥0∞} (h₁ : Below e f a ε) (h₂ : Close f g ε') :
    Below e g a (a * ε' + ε) := fun E =>
  calc e.pr E ≤ a * f.pr E + ε := h₁ E
    _ ≤ a * (g.pr E + ε') + ε := by gcongr; exact (h₂ E).1
    _ = a * g.pr E + (a * ε' + ε) := by rw [mul_add, add_assoc]

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

theorem pr_uniform {C V : Type} [Fintype C] [Nonempty C] (f : C → V) (E : V → Prop) :
    Exp.pr (⟨.coin C fun c => .ret (f c), fun _ => PUnit.unit⟩ : Exp V) E =
      ENNReal.ofReal (Claims.Pr fun c => E (f c)) := by
  rw [Claims.ZKMal.Pr_eq_prCoin, ENNReal.ofReal_toReal (FlockSoundness.ZK.prCoin_ne_top _)]
  rfl

end FlockSoundness.SampledGeneric

namespace FlockSoundness.SampledGeneric.FlockZk

open Game
open scoped ENNReal

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

abbrev Parser : Type := _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation

abbrev sha : Claims.Msg → Claims.Msg := Claims.ZKMal.CoinOpen.shaH

abbrev Sess (T : Claims.ZKMal.CoinLayer.TableData) : Claims.ZKMal.RefSession :=
  Claims.ZKMal.CoinLayer.session Claims.ZKServer.Loop.loopNr T

abbrev Tape (T : Claims.ZKMal.CoinLayer.TableData) : Type := (Sess T).refP.Nonce × (Sess T).refP.Rand

abbrev Release (T : Claims.ZKMal.CoinLayer.TableData) : Type := Claims.ZKServer.LeanRelease Claims.Flock.Wit (Tape T)

abbrev PC (T : Claims.ZKMal.CoinLayer.TableData) : Type :=
  Claims.Msg → List (Sess T).Coin → _root_.Flock.ZkServer.Frame.Masker

abbrev FC (T : Claims.ZKMal.CoinLayer.TableData) : Type :=
  Claims.Msg → List (Sess T).Coin → Except String _root_.Flock.ZkServer.Frame.Masker

abbrev Stops (T : Claims.ZKMal.CoinLayer.TableData) (L : Release T) (pc : PC T) (fc : FC T) :
    Claims.ZKMal.RefSession :=
  (Sess T).withStops (Claims.ZKServer.LeanRelease.rfOf (Sess T) L pc fc Claims.ZKServer.Loop.readCoins) L.stop

noncomputable abbrev rounds (rel : Parser) : Claims.Msg → ℕ := fun x => (FlockDeploy.deployZk rel).V.k x + 1

structure Verifier (rel : Parser) (Θ : Type) where
  Vs : Claims.VStar Θ
  θ : Θ
  hk : ∀ x, Vs.k x = rounds rel x

noncomputable abbrev sim (rel : Parser) (T : Claims.ZKMal.CoinLayer.TableData) (L : Release T) (pc : PC T) (fc : FC T)
    (t : ℕ) : Claims.Simulator :=
  Claims.ZKServer.leaky1
    (((Stops T L pc fc).refP.withSim fun x _ c => (Stops T L pc fc).coinModel x c).simulatorE t (t * 3 ^ t))
    (rounds rel) L.toSrv.stop

noncomputable def realView (rel : Parser) (T : Claims.ZKMal.CoinLayer.TableData) (L : Release T) (wk : L.toSrv.Worker)
    {Θ : Type} (x : Claims.Msg) (w : Claims.Flock.Wit) (v : Verifier rel Θ) (ω : Tape T × L.toSrv.Chk) :
    List ((Claims.Msg × Claims.Out) × ℕ) :=
  L.toSrv.timed x (Claims.runStar sha (L.toSrv.proverD L.fillOf (rounds rel) wk w) v.Vs x ω v.θ).msgs

noncomputable def simView (rel : Parser) (T : Claims.ZKMal.CoinLayer.TableData) (L : Release T) (pc : PC T)
    (fc : FC T) (t : ℕ) {Θ : Type} (x : Claims.Msg) (v : Verifier rel Θ) (c : (sim rel T L pc fc t).C x) :
    List ((Claims.Msg × Claims.Out) × ℕ) :=
  L.toSrv.timed x ((sim rel T L pc fc t).out sha v.Vs v.θ x c)

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

noncomputable def err (rel : Parser) (T : Claims.ZKMal.CoinLayer.TableData) (vs : (Sess T).SwVals) (L : Release T)
    (pc : PC T) (fc : FC T) (t : ℕ) (δ₁ : ℝ≥0∞) {Θ : Type} (x : Claims.Msg) (w : Claims.Flock.Wit)
    (v : Verifier rel Θ) : ℝ≥0∞ :=
  2 * ENNReal.ofReal ((8 * (Stops T L pc fc).nLeaves x * δ₁.toReal +
      8 * t * (Stops T L pc fc).nLeaves x * δ₁.toReal + 2 * (1 / 3) ^ t) +
    4 * t * 3 ^ t * (Claims.Fork sha (((Stops T L pc fc).swP vs).prover w) v.Vs v.θ x).Pr sha)

abbrev Live (T : Claims.ZKMal.CoinLayer.TableData) (vs : (Sess T).SwVals) (L : Release T) (wk : L.toSrv.Worker)
    (pc : PC T) (fc : FC T) (x : Claims.Msg) (w : Claims.Flock.Wit) : Prop :=
  Claims.ZKServer.LeanRelease.ProgReads (Sess T) L pc fc x ∧
    Claims.ZKServer.LeanRelease.LinkRoundEq (Sess T) vs L x w ∧
    Claims.ZKServer.LeanRelease.FinalRoundEq (Sess T) vs L sha wk x w ∧
    Claims.ZKServer.LeanRelease.LiveRoundEq (Sess T) vs L sha wk x w ∧
    Claims.ZKServer.LeanRelease.FillRoundEq (Sess T) vs L sha wk x w

abbrev LiveRel (rel : Parser) (T : Claims.ZKMal.CoinLayer.TableData) (vs : (Sess T).SwVals) (L : Release T)
    (wk : L.toSrv.Worker) (pc : PC T) (fc : FC T) (x : Claims.Msg) (w : Claims.Flock.Wit) : Prop :=
  Claims.Flock.R (FlockDeploy.deployZk rel).parse x w ∧ Live T vs L wk pc fc x w

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
