import Proofs.Flock.Claims.ZK.Release
import Proofs.Flock.Soundness.Discharge.ZkView.OsSalts
import Proofs.Flock.Claims.Sampling.Draws

/-!
# The ZK server at the operating system's bytes

`Flock.zk_server_lean`, `Flock.zk_server_blind` and `ZkServer.zk_deferred` draw the server's tape uniformly. Every random
value on it is a read of the operating system that a Lean process makes: each rep's frame reads its draws at its start
(`flock-firewall frame`: `IO.getRandomBytes (drawBytes l)`, which `Draws.ofBytes` parses into the pads `h`, `μ`, the inner
rows' padding, `τ`'s salt and the pads tree's salts), and the tap (`flock-firewall tap`) serves the prover's every other
read: the mask slot's words, level 0's hiding, the other trees' salts, the coin tree's nonce and `ProverSeed`, from which a
proving build expands nothing (`zk_hooks.rs`). The step from those reads to a uniform tape is A3
(`Assumptions.UniformRandomBytes`, `uniform/io-getrandombytes`):

* `OsTape`: the tape as `m` streams of the operating system's bytes, one for one (`OsTape.e`), and the stream each rep's
  frame reads at its start (`OsTape.frame`); every other stream is a read of the tap. `OsTape.frameBytes`: the frame's
  bytes on a tape.
* `Pr_os_prod`: an event of the tape the streams decode to, beside any independent randomness, has the probability it
  has at a uniform tape, under A3.
* `ZkDeferredOs`, `ZkServerBlindOs`, `ZkServerLeanOs`: the three theorems with the experiment's tape the bytes of a
  source that meets A3 (`OsTape.e (src ω)`), A3 among the premises, and the bound unchanged. `ZkServerLeanOs` also asks
  that the release's frames read their streams (`L.draws = O.frameBytes`), so `ItemsRefine` relates the model's prover to
  the frame at the operating system's bytes.

The three are statements: A3 is not among `Flock.zk_server_lean`'s premises, and its own docstring still puts the
prover's masks outside its scope.
-/

namespace Claims.ZKServer

open Claims Claims.ZKMal
open FlockSoundness.Discharge

/-- **The server's tape as the operating system's bytes**: `m` streams, stream `i` its first `Ls i` bytes, read one for
one as the tape (`e`). Rep `i`'s frame reads stream `frame i` at its start; every other stream is a read of the Lean
tap. -/
structure OsTape (R : Type) where
  m : ℕ
  Ls : Fin m → ℕ
  e : ((i : Fin m) → Fin (Ls i) → Fin 256) ≃ R
  frame : ℕ → Fin m

namespace OsTape

/-- Rep `i`'s frame's bytes on the tape `ρ`: its stream, as bytes. -/
def frameBytes {R : Type} (O : OsTape R) (ρ : R) (i : ℕ) : ByteArray :=
  FlockSoundness.Audit.Law.streamOf (O.e.symm ρ (O.frame i))

end OsTape

open FlockSoundness.Game FlockSoundness.ZK in
/-- **A3's bytes, decoded one for one, beside independent randomness, are a uniform tape**: in an experiment on `Ω × A`
whose `Ω` is a byte source that meets A3, an event of the tape `e` decodes from it and of the rest has the probability it
has at a uniform tape. -/
theorem Pr_os_prod {Ω Y A : Type} [Fintype Ω] [Fintype Y] [Fintype A] {m : ℕ} {Ls : Fin m → ℕ}
    {src : Ω → (i : Fin m) → Fin (Ls i) → Fin 256} (hA3 : FlockSoundness.Assumptions.UniformRandomBytes Ls src)
    (e : ((i : Fin m) → Fin (Ls i) → Fin 256) ≃ Y) (E : Y × A → Prop) :
    Pr (fun ω : Ω × A => E (e (src ω.1), ω.2)) = Pr E := by
  rw [Sampling.Pr_eq_toReal_prCoin, Sampling.Pr_eq_toReal_prCoin]
  congr 1
  calc prCoin (fun ω : Ω × A => E (e (src ω.1), ω.2))
      = prCoin (fun p : A × Ω => E (e (src p.2), p.1)) :=
        (prCoin_congr_equiv (Equiv.prodComm A Ω) fun ω : Ω × A => E (e (src ω.1), ω.2)).symm
    _ = prCoin (fun p : A × Y => E (p.2, p.1)) := ZkView.prCoin_os_prod hA3 e fun a y => E (y, a)
    _ = prCoin E := prCoin_congr_equiv (Equiv.prodComm A Y) E

/-- **`ZkServer.zk_deferred` at the operating system's bytes**: the server's tape the bytes of a source that meets A3,
read one for one (`O.e`), the bound unchanged. -/
def ZkDeferredOs : Prop :=
  ∀ {W N Ω Θ : Type} [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω] (S : ZkServer W (N × Ω))
    (fill : Msg → ℕ → ℕ → W → N × Ω → List (Msg × Msg) → S.Val) (H : Msg → Msg) (k : Msg → ℕ)
    (P : W → Prover (N × Ω)) (Rl : Rel W) (Sim : Simulator) (εzk γ : Msg → ℝ), ZKmalR H k P Rl Sim εzk γ →
    ∀ (x : Msg) (w : W), Rl x w → ∀ (wk : S.Worker), S.RefD H P fill k wk x w →
    ∀ (Vs : VStar Θ) (θ : Θ), Vs.k x = k x → ∀ (Ev : List ((Msg × Out) × ℕ) → Prop) (O : OsTape (N × Ω))
      {Ωos : Type} [Fintype Ωos] (src : Ωos → (i : Fin O.m) → Fin (O.Ls i) → Fin 256),
      FlockSoundness.Assumptions.UniformRandomBytes O.Ls src →
      Pr (fun ω : Ωos × S.Chk => Ev (S.timed x (runStar H (S.proverD fill k wk w) Vs x (O.e (src ω.1), ω.2) θ).msgs)) ≤
        2 * (Pr (fun c : (leaky1 Sim k S.stop).C x => Ev (S.timed x ((leaky1 Sim k S.stop).out H Vs θ x c))) +
          εzk x + γ x * (Fork H (P w) Vs θ x).Pr H)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-- **`Flock.zk_server_blind` at the operating system's bytes**: the ZK server at pad-blind workers, at
`flock-circuit --zk`'s reference session, with the server's tape the bytes of a source that meets A3, read one for one
(`O.e`), the bound unchanged. -/
def ZkServerBlindOs : Prop :=
  ∀ (H : Msg → Msg) (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (S : RefSession)
    (vs : S.SwVals) (t : ℕ), 0 < t → ∀ {δ₁ : ENNReal}, Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁ → δ₁ ≠ ⊤ →
    S.Tables → S.Complete (Flock.R (FlockDeploy.deployZk rel).parse) → S.DummyReg → S.Causal → S.SwCausal vs →
    S.TapeOK → ∀ (O : OsTape (S.refP.Nonce × S.refP.Rand)) {Ωos : Type} [Fintype Ωos]
      (src : Ωos → (i : Fin O.m) → Fin (O.Ls i) → Fin 256), FlockSoundness.Assumptions.UniformRandomBytes O.Ls src →
    ∀ (srv : ZkServer Flock.Wit (S.refP.Nonce × S.refP.Rand))
      (fill : Msg → ℕ → ℕ → Flock.Wit → S.refP.Nonce × S.refP.Rand → List (Msg × Msg) → srv.Val)
      (bw : srv.BWorker) (x : Msg) (w : Flock.Wit), Flock.R (FlockDeploy.deployZk rel).parse x w →
      srv.RefD H (S.swP vs).prover fill (fun x => (FlockDeploy.deployZk rel).V.k x + 1) (srv.lift bw) x w →
    ∀ {Θ : Type} (Vs : VStar Θ) (θ : Θ), Vs.k x = (FlockDeploy.deployZk rel).V.k x + 1 →
    ∀ (Ev : List ((Msg × Out) × ℕ) → Prop),
      Pr (fun ω : Ωos × srv.Chk =>
          Ev (srv.timed x (runStar H (srv.proverD fill (fun x => (FlockDeploy.deployZk rel).V.k x + 1) (srv.lift bw) w)
            Vs x (O.e (src ω.1), ω.2) θ).msgs)) ≤
        2 * (Pr (fun c : (leaky1 ((S.refP.withSim fun x _ c => S.coinModel x c).simulatorE t (t * 3 ^ t))
              (fun x => (FlockDeploy.deployZk rel).V.k x + 1) srv.stop).C x =>
            Ev (srv.timed x ((leaky1 ((S.refP.withSim fun x _ c => S.coinModel x c).simulatorE t (t * 3 ^ t))
              (fun x => (FlockDeploy.deployZk rel).V.k x + 1) srv.stop).out H Vs θ x c))) +
            (8 * S.nLeaves x * δ₁.toReal + 8 * t * S.nLeaves x * δ₁.toReal + 2 * (1 / 3) ^ t) +
            4 * t * 3 ^ t * (Fork H ((S.swP vs).prover w) Vs θ x).Pr H)

/-- **`Flock.zk_server_lean` at the operating system's bytes**: the deployed release `L.toSrv`, whose frames read their
draws from the operating system (`L.draws = O.frameBytes`) and whose tap serves every other stream of the tape, at the
bytes of a source that meets A3, read one for one (`O.e`), the bound unchanged. Its premises are `Flock.zk_server_lean`'s
and A3: `Claims.Hm96Hiding`, the session's six Props, D8 (a) on the firewall's transcripts (`FirewallComputes`), and the
item-stream refinement at `swP vs` (`LeanRelease.ItemsRefine`). -/
def ZkServerLeanOs : Prop :=
  ∀ (H : Msg → Msg) (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (S : RefSession)
    (vs : S.SwVals) (t : ℕ), 0 < t → ∀ {δ₁ : ENNReal}, Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁ → δ₁ ≠ ⊤ →
    S.Tables → S.Complete (Flock.R (FlockDeploy.deployZk rel).parse) → S.DummyReg → S.Causal → S.SwCausal vs →
    S.TapeOK → ∀ (O : OsTape (S.refP.Nonce × S.refP.Rand)) {Ωos : Type} [Fintype Ωos]
      (src : Ωos → (i : Fin O.m) → Fin (O.Ls i) → Fin 256), FlockSoundness.Assumptions.UniformRandomBytes O.Ls src →
    ∀ (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand)), L.draws = O.frameBytes →
    ∀ (wk : L.toSrv.Worker) (x : Msg) (w : Flock.Wit), Flock.R (FlockDeploy.deployZk rel).parse x w →
    ∀ (spec : String → Option ByteArray),
      (∀ ρ h, _root_.Flock.Firewall.Contract.FirewallComputes spec (L.computed x w ρ h)) →
      L.ItemsRefine H (S.swP vs).prover (fun x => (FlockDeploy.deployZk rel).V.k x + 1) wk x w spec →
    ∀ {Θ : Type} (Vs : VStar Θ) (θ : Θ), Vs.k x = (FlockDeploy.deployZk rel).V.k x + 1 →
    ∀ (Ev : List ((Msg × Out) × ℕ) → Prop),
      Pr (fun ω : Ωos × L.toSrv.Chk =>
          Ev (L.toSrv.timed x (runStar H (L.toSrv.proverD L.fillOf (fun x => (FlockDeploy.deployZk rel).V.k x + 1) wk w)
            Vs x (O.e (src ω.1), ω.2) θ).msgs)) ≤
        2 * (Pr (fun c : (leaky1 ((S.refP.withSim fun x _ c => S.coinModel x c).simulatorE t (t * 3 ^ t))
              (fun x => (FlockDeploy.deployZk rel).V.k x + 1) L.toSrv.stop).C x =>
            Ev (L.toSrv.timed x ((leaky1 ((S.refP.withSim fun x _ c => S.coinModel x c).simulatorE t (t * 3 ^ t))
              (fun x => (FlockDeploy.deployZk rel).V.k x + 1) L.toSrv.stop).out H Vs θ x c))) +
            (8 * S.nLeaves x * δ₁.toReal + 8 * t * S.nLeaves x * δ₁.toReal + 2 * (1 / 3) ^ t) +
            4 * t * 3 ^ t * (Fork H ((S.swP vs).prover w) Vs θ x).Pr H)

end Claims.ZKServer
