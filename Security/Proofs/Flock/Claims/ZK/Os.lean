import Proofs.Flock.Claims.ZK.Release
import Proofs.Flock.Soundness.Discharge.ZkView.OsSalts
import Proofs.Flock.Claims.Sampling.Draws

/-!
# The ZK server at the operating system's bytes

`Flock.zk_server_lean`, `Flock.zk_server_blind` and `ZkServer.zk_deferred` draw the server's tape uniformly. Every random
value on it is a read of the operating system that a Lean process makes: each rep's frame reads its draws in one call at
its start (`flock-firewall frame`, `Frame.cmd`, which is `Frame.cmdWith IO.getRandomBytes`, or the session loop's frames,
`Session.frames`, `Frame.serveWith IO.getRandomBytes` too: `Frame.drawsWith` reads
`drawBytes l` bytes and `Draws.ofRead` refuses any other count, then parses them into the pads `h`, `μ`, the inner rows'
padding, `τ`'s salt and the pads tree's salts), and the tap (`flock-firewall tap`) serves the prover's every other read:
the mask slot's words, level 0's hiding, the other trees' salts, the coin tree's nonce and `ProverSeed`, from which a
proving build expands nothing (`zk_hooks.rs`). The step from those reads to a uniform tape is A3 per test
(`Assumptions.UniformRandomBytesTest`, `uniform/io-getrandombytes`), at the one event each theorem bounds, and its slack
`η` enters the bound:

* `OsTape`: the tape as the operating system's bytes, one for one (`OsTape.e`): a stream for each of the first `reps`
  reps' frames, of exactly `drawBytes` bytes, and the tap's streams, which are every other byte of the tape.
  `OsTape.dec`: the same decoding of all the streams as one byte source, A3's form. `OsTape.frameBytes`: rep `i`'s
  frame's bytes on a tape. `OsTape.release`: the release whose frames read those bytes.
* `frame_cmd`, `frame_draws`, `drawsWith_refuses`: the frame's read is the release's draws. `Frame.cmd` is
  `Frame.cmdWith IO.getRandomBytes`; the read `cmdWith` starts a rep with, on a source that returns rep `i`'s stream,
  takes it as `drawsAt`; a read of any other length is refused, so no rep runs on a short read's zeros.
* `Pr_os_prod`: an event of the tape the streams decode to, beside any independent randomness, has at most the
  probability it has at a uniform tape plus `η`, where the source passes the event's test up to `η` (A3 per test).
* `ZkDeferredOs`, `ZkServerBlindOs`, `ZkServerLeanOs`: the three theorems with the experiment's tape the bytes of a
  byte source, A3 per test at the theorem's event among the premises, and the bound plus `η`; proved as
  `zkDeferredOs`, `zkServerBlindOs` and `zkServerLeanOs`. `ZkServerLeanOs` is at `O.release L`, whose rep `i` reads
  stream `i` of the tape, so `ItemsRefine` relates the model's prover to the frame at the operating system's bytes.
-/

namespace Claims.ZKServer

open Claims Claims.ZKMal
open FlockSoundness.Discharge

/-- **The server's tape as the operating system's bytes**, at rep `i`'s pad layout `lay i`: rep `i`'s frame reads
stream `i` of the first part, `drawBytes (lay i)` bytes, for each of the first `reps` reps, and the tap serves the
second part's streams, stream `j` its first `tapLen j` bytes. The two parts are the whole tape, read one for one
(`e`). -/
structure OsTape (R : Type) (lay : ℕ → _root_.Flock.Zk.PadLayout) where
  reps : ℕ
  taps : ℕ
  tapLen : Fin taps → ℕ
  e : (((i : Fin reps) → Fin (_root_.Flock.Firewall.Frame.drawBytes (lay i)) → Fin 256) ×
    ((j : Fin taps) → Fin (tapLen j) → Fin 256)) ≃ R

namespace OsTape

variable {R : Type} {lay : ℕ → _root_.Flock.Zk.PadLayout}

/-- The streams' lengths as one byte source's, the frames' then the tap's. -/
def Ls (O : OsTape R lay) : Fin O.reps ⊕ Fin O.taps → ℕ :=
  Sum.elim (fun i => _root_.Flock.Firewall.Frame.drawBytes (lay i)) O.tapLen

/-- The tape from the streams as one byte source, the frames' then the tap's. -/
def dec (O : OsTape R lay) : ((k : Fin O.reps ⊕ Fin O.taps) → Fin (O.Ls k) → Fin 256) ≃ R :=
  (Equiv.sumPiEquivProdPi fun k => Fin (O.Ls k) → Fin 256).trans O.e

/-- Rep `i`'s frame's bytes on the tape `ρ`: its stream, as bytes. A rep from `reps` on has no stream and reads
none, which the frame refuses (`drawsWith_refuses`). -/
def frameBytes (O : OsTape R lay) (ρ : R) (i : ℕ) : ByteArray :=
  if h : i < O.reps then FlockSoundness.Audit.Law.streamOf ((O.e.symm ρ).1 ⟨i, h⟩) else ByteArray.empty

/-- The release whose rep `i` reads stream `i` of the tape at its start. -/
def release {W : Type} (O : OsTape R lay) (L : LeanRelease W R) : LeanRelease W R :=
  { L with draws := O.frameBytes }

theorem frameBytes_of_lt (O : OsTape R lay) (ρ : R) (i : Fin O.reps) :
    O.frameBytes ρ i = FlockSoundness.Audit.Law.streamOf ((O.e.symm ρ).1 i) :=
  dif_pos i.isLt

/-- Rep `i`'s frame reads exactly the bytes its draws take. -/
theorem frameBytes_size (O : OsTape R lay) (ρ : R) (i : Fin O.reps) :
    (O.frameBytes ρ i).size = _root_.Flock.Firewall.Frame.drawBytes (lay i) := by
  rw [frameBytes_of_lt]
  exact FlockSoundness.Audit.DrawExec.size_streamOf _

end OsTape

/-- **`flock-firewall frame` is the frame at the operating system's bytes**: `Frame.cmd` runs `Frame.cmdWith`, whose
rep starts with `drawsWith get`, at `IO.getRandomBytes`. -/
theorem frame_cmd : _root_.Flock.Firewall.Frame.cmd = _root_.Flock.Firewall.Frame.cmdWith IO.getRandomBytes := rfl

/-- **The frame's read on the tape is the release's draws**: `drawsWith`, the read `Frame.cmdWith` starts a rep with, on
a source that returns rep `i`'s stream of the tape, takes it whole as `drawsAt` of the release that reads the tape. -/
theorem frame_draws {W R : Type} (L : LeanRelease W R) (x : Msg) (O : OsTape R (L.lay x)) (ρ : R) (i : Fin O.reps) :
    _root_.Flock.Firewall.Frame.drawsWith (m := Id) (fun _ => pure (O.frameBytes ρ i)) (L.lay x i) =
      .ok ((O.release L).drawsAt x ρ i) := by
  show _root_.Flock.Firewall.Frame.Draws.ofRead (L.lay x i) (O.frameBytes ρ i) = _
  rw [_root_.Flock.Firewall.Frame.Draws.ofRead, if_pos (O.frameBytes_size ρ i)]
  rfl

/-- **A read of any other length is refused**: the frame starts no rep on bytes that are not exactly its draws', so no
rep masks with the zeros `Draws.ofBytes` reads past a short input. -/
theorem drawsWith_refuses (l : _root_.Flock.Zk.PadLayout) (b : ByteArray)
    (h : b.size ≠ _root_.Flock.Firewall.Frame.drawBytes l) (d : _root_.Flock.Firewall.Frame.Draws) :
    _root_.Flock.Firewall.Frame.drawsWith (m := Id) (fun _ => pure b) l ≠ .ok d := by
  show _root_.Flock.Firewall.Frame.Draws.ofRead l b ≠ _
  rw [_root_.Flock.Firewall.Frame.Draws.ofRead, if_neg h]
  intro hd
  cases hd

/-- **The OS's bytes, decoded one for one, beside independent randomness, are a uniform tape up to `η`**: in an
experiment on `Ω × A` whose `Ω` is a byte source that passes the event's test up to `η` at every `a` (A3 per test), an
event of the tape `e` decodes from it and of the rest has at most the probability it has at a uniform tape, plus `η`. -/
theorem Pr_os_prod {Ω Tp A ι : Type} [Fintype Ω] [Fintype Tp] [Fintype A] [Fintype ι] [DecidableEq ι] {Ls : ι → ℕ}
    {src : Ω → (i : ι) → Fin (Ls i) → Fin 256} (e : ((i : ι) → Fin (Ls i) → Fin 256) ≃ Tp) (E : Tp × A → Prop)
    {η : ENNReal} (hη : η ≠ ⊤)
    (hA3 : ∀ a, FlockSoundness.Assumptions.UniformRandomBytesTest Ls src (fun b => E (e b, a)) η) :
    Pr (fun ω : Ω × A => E (e (src ω.1), ω.2)) ≤ Pr E + η.toReal := by
  rw [Sampling.Pr_eq_toReal_prCoin, Sampling.Pr_eq_toReal_prCoin]
  have h : FlockSoundness.Game.prCoin (fun ω : Ω × A => E (e (src ω.1), ω.2)) ≤ FlockSoundness.Game.prCoin E + η :=
    calc FlockSoundness.Game.prCoin (fun ω : Ω × A => E (e (src ω.1), ω.2))
        = FlockSoundness.Game.prCoin (fun p : A × Ω => E (e (src p.2), p.1)) :=
          (FlockSoundness.ZK.prCoin_congr_equiv (Equiv.prodComm A Ω) fun ω : Ω × A => E (e (src ω.1), ω.2)).symm
      _ ≤ FlockSoundness.Game.prCoin (fun p : A × Tp => E (p.2, p.1)) + η :=
          (ZkView.prCoin_os_prod e (fun a y => E (y, a)) hA3).1
      _ = FlockSoundness.Game.prCoin E + η :=
          congrArg (· + η) (FlockSoundness.ZK.prCoin_congr_equiv (Equiv.prodComm A Tp) E)
  have hE : FlockSoundness.Game.prCoin E ≠ ⊤ :=
    ne_top_of_le_ne_top ENNReal.one_ne_top (FlockSoundness.Audit.DrawExec.prCoin_le_one E)
  exact (ENNReal.toReal_mono (ENNReal.add_ne_top.2 ⟨hE, hη⟩) h).trans_eq (ENNReal.toReal_add hE hη)

/-- **`ZkServer.zk_deferred` at the operating system's bytes**: the server's tape the bytes of a byte source, read one
for one (`dec`), which passes the event's test up to `η` (A3 per test); the bound plus `η`. -/
def ZkDeferredOs : Prop :=
  ∀ {W N Ω Θ : Type} [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω] (S : ZkServer W (N × Ω))
    (fill : Msg → ℕ → ℕ → W → N × Ω → List (Msg × Msg) → S.Val) (H : Msg → Msg) (k : Msg → ℕ)
    (P : W → Prover (N × Ω)) (Rl : Rel W) (Sim : Simulator) (εzk γ : Msg → ℝ), ZKmalR H k P Rl Sim εzk γ →
    ∀ (x : Msg) (w : W), Rl x w → ∀ (wk : S.Worker), S.RefD H P fill k wk x w →
    ∀ (Vs : VStar Θ) (θ : Θ), Vs.k x = k x → ∀ (Ev : List ((Msg × Out) × ℕ) → Prop)
      {ι : Type} [Fintype ι] [DecidableEq ι] (Ls : ι → ℕ) (dec : ((i : ι) → Fin (Ls i) → Fin 256) ≃ N × Ω)
      {Ωos : Type} [Fintype Ωos] (src : Ωos → (i : ι) → Fin (Ls i) → Fin 256) {η : ENNReal}, η ≠ ⊤ →
      (∀ c : S.Chk, FlockSoundness.Assumptions.UniformRandomBytesTest Ls src
        (fun b => Ev (S.timed x (runStar H (S.proverD fill k wk w) Vs x (dec b, c) θ).msgs)) η) →
      Pr (fun ω : Ωos × S.Chk => Ev (S.timed x (runStar H (S.proverD fill k wk w) Vs x (dec (src ω.1), ω.2) θ).msgs)) ≤
        2 * (Pr (fun c : (leaky1 Sim k S.stop).C x => Ev (S.timed x ((leaky1 Sim k S.stop).out H Vs θ x c))) +
          εzk x + γ x * (Fork H (P w) Vs θ x).Pr H) + η.toReal

theorem zkDeferredOs : ZkDeferredOs := by
  intro W N Ω Θ _ _ _ _ S fill H k P Rl Sim εzk γ hZK x w hR wk hRef Vs θ hk Ev ι _ _ Ls dec Ωos _ src η hη hA3
  exact (Pr_os_prod dec (fun p : (N × Ω) × S.Chk => Ev (S.timed x (runStar H (S.proverD fill k wk w) Vs x p θ).msgs))
    hη hA3).trans (add_le_add (S.zk_deferred fill H k P Rl Sim εzk γ hZK x w hR wk hRef.own Vs θ hk Ev) le_rfl)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-- **`Flock.zk_server_blind` at the operating system's bytes**: the ZK server at pad-blind workers, at
`flock-circuit --zk`'s reference session, with the server's tape the bytes of a byte source, read one for one (`dec`),
which passes the event's test up to `η` (A3 per test); the bound plus `η`. -/
def ZkServerBlindOs : Prop :=
  ∀ (H : Msg → Msg) (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (S : RefSession)
    (vs : S.SwVals) (t : ℕ), 0 < t → ∀ {δ₁ : ENNReal}, Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁ → δ₁ ≠ ⊤ →
    S.Tables → S.Complete (Flock.R (FlockDeploy.deployZk rel).parse) → S.DummyReg → S.Causal → S.SwCausal vs →
    S.TapeOK →
    ∀ (srv : ZkServer Flock.Wit (S.refP.Nonce × S.refP.Rand))
      (fill : Msg → ℕ → ℕ → Flock.Wit → S.refP.Nonce × S.refP.Rand → List (Msg × Msg) → srv.Val)
      (bw : srv.BWorker) (x : Msg) (w : Flock.Wit), Flock.R (FlockDeploy.deployZk rel).parse x w →
      srv.RefD H (S.swP vs).prover fill (fun x => (FlockDeploy.deployZk rel).V.k x + 1) (srv.lift bw) x w →
    ∀ {Θ : Type} (Vs : VStar Θ) (θ : Θ), Vs.k x = (FlockDeploy.deployZk rel).V.k x + 1 →
    ∀ (Ev : List ((Msg × Out) × ℕ) → Prop) {ι : Type} [Fintype ι] [DecidableEq ι] (Ls : ι → ℕ)
      (dec : ((i : ι) → Fin (Ls i) → Fin 256) ≃ S.refP.Nonce × S.refP.Rand) {Ωos : Type} [Fintype Ωos]
      (src : Ωos → (i : ι) → Fin (Ls i) → Fin 256) {η : ENNReal}, η ≠ ⊤ →
      (∀ c : srv.Chk, FlockSoundness.Assumptions.UniformRandomBytesTest Ls src
        (fun b => Ev (srv.timed x (runStar H (srv.proverD fill (fun x => (FlockDeploy.deployZk rel).V.k x + 1)
          (srv.lift bw) w) Vs x (dec b, c) θ).msgs)) η) →
      Pr (fun ω : Ωos × srv.Chk =>
          Ev (srv.timed x (runStar H (srv.proverD fill (fun x => (FlockDeploy.deployZk rel).V.k x + 1) (srv.lift bw) w)
            Vs x (dec (src ω.1), ω.2) θ).msgs)) ≤
        2 * (Pr (fun c : (leaky1 ((S.refP.withSim fun x _ c => S.coinModel x c).simulatorE t (t * 3 ^ t))
              (fun x => (FlockDeploy.deployZk rel).V.k x + 1) srv.stop).C x =>
            Ev (srv.timed x ((leaky1 ((S.refP.withSim fun x _ c => S.coinModel x c).simulatorE t (t * 3 ^ t))
              (fun x => (FlockDeploy.deployZk rel).V.k x + 1) srv.stop).out H Vs θ x c))) +
            (8 * S.nLeaves x * δ₁.toReal + 8 * t * S.nLeaves x * δ₁.toReal + 2 * (1 / 3) ^ t) +
            4 * t * 3 ^ t * (Fork H ((S.swP vs).prover w) Vs θ x).Pr H) + η.toReal

theorem zkServerBlindOs : ZkServerBlindOs := by
  intro H rel S vs t ht δ₁ hT1 hδ₁ hS hV hD hc hsc hT srv fill bw x w hR hRef Θ Vs θ hk Ev ι _ _ Ls dec Ωos _ src η hη
    hA3
  exact (Pr_os_prod dec (fun p : (S.refP.Nonce × S.refP.Rand) × srv.Chk =>
      Ev (srv.timed x (runStar H (srv.proverD fill (fun x => (FlockDeploy.deployZk rel).V.k x + 1) (srv.lift bw) w)
        Vs x p θ).msgs)) hη hA3).trans
    (add_le_add (Flock.zk_server_blind H rel S vs t ht hT1 hδ₁ hS hV hD hc hsc hT srv fill bw x w hR hRef Vs θ hk Ev)
      le_rfl)

/-- **`Flock.zk_server_lean` at the operating system's bytes**: the deployed release at the tape `O` reads
(`O.release L`), whose rep `i`'s frame reads stream `i` of the tape at its start and whose tap serves every other
stream, at the bytes of a byte source, read one for one (`O.dec`), which passes the event's test up to `η` (A3 per
test); the bound plus `η`. Its premises are `Flock.zk_server_lean`'s at `O.release L` and A3 per test:
`Claims.Hm96Hiding`, the session's six Props and the item-stream refinement at `swP vs` (`LeanRelease.ItemsRefine`), with
no assumption on what the firewall computes. -/
def ZkServerLeanOs : Prop :=
  ∀ (H : Msg → Msg) (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (S : RefSession)
    (vs : S.SwVals) (t : ℕ), 0 < t → ∀ {δ₁ : ENNReal}, Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁ → δ₁ ≠ ⊤ →
    S.Tables → S.Complete (Flock.R (FlockDeploy.deployZk rel).parse) → S.DummyReg → S.Causal → S.SwCausal vs →
    S.TapeOK →
    ∀ (L : LeanRelease Flock.Wit (S.refP.Nonce × S.refP.Rand)) (x : Msg)
      (O : OsTape (S.refP.Nonce × S.refP.Rand) (L.lay x)) (wk : (O.release L).toSrv.Worker) (w : Flock.Wit),
      Flock.R (FlockDeploy.deployZk rel).parse x w →
      (O.release L).ItemsRefine H (S.swP vs).prover (fun x => (FlockDeploy.deployZk rel).V.k x + 1) wk x w →
    ∀ {Θ : Type} (Vs : VStar Θ) (θ : Θ), Vs.k x = (FlockDeploy.deployZk rel).V.k x + 1 →
    ∀ (Ev : List ((Msg × Out) × ℕ) → Prop) {Ωos : Type} [Fintype Ωos]
      (src : Ωos → (k : Fin O.reps ⊕ Fin O.taps) → Fin (O.Ls k) → Fin 256) {η : ENNReal}, η ≠ ⊤ →
      (∀ c : (O.release L).toSrv.Chk, FlockSoundness.Assumptions.UniformRandomBytesTest O.Ls src
        (fun b => Ev ((O.release L).toSrv.timed x (runStar H ((O.release L).toSrv.proverD (O.release L).fillOf
          (fun x => (FlockDeploy.deployZk rel).V.k x + 1) wk w) Vs x (O.dec b, c) θ).msgs)) η) →
      Pr (fun ω : Ωos × (O.release L).toSrv.Chk =>
          Ev ((O.release L).toSrv.timed x (runStar H ((O.release L).toSrv.proverD (O.release L).fillOf
            (fun x => (FlockDeploy.deployZk rel).V.k x + 1) wk w) Vs x (O.dec (src ω.1), ω.2) θ).msgs)) ≤
        2 * (Pr (fun c : (leaky1 ((S.refP.withSim fun x _ c => S.coinModel x c).simulatorE t (t * 3 ^ t))
              (fun x => (FlockDeploy.deployZk rel).V.k x + 1) (O.release L).toSrv.stop).C x =>
            Ev ((O.release L).toSrv.timed x ((leaky1 ((S.refP.withSim fun x _ c => S.coinModel x c).simulatorE t
              (t * 3 ^ t)) (fun x => (FlockDeploy.deployZk rel).V.k x + 1) (O.release L).toSrv.stop).out H Vs θ x c))) +
            (8 * S.nLeaves x * δ₁.toReal + 8 * t * S.nLeaves x * δ₁.toReal + 2 * (1 / 3) ^ t) +
            4 * t * 3 ^ t * (Fork H ((S.swP vs).prover w) Vs θ x).Pr H) + η.toReal

theorem zkServerLeanOs : ZkServerLeanOs := by
  intro H rel S vs t ht δ₁ hT1 hδ₁ hS hV hD hc hsc hT L x O wk w hR hI Θ Vs θ hk Ev Ωos _ src η hη hA3
  exact (Pr_os_prod O.dec (fun p : (S.refP.Nonce × S.refP.Rand) × (O.release L).toSrv.Chk =>
      Ev ((O.release L).toSrv.timed x (runStar H ((O.release L).toSrv.proverD (O.release L).fillOf
        (fun x => (FlockDeploy.deployZk rel).V.k x + 1) wk w) Vs x p θ).msgs)) hη hA3).trans
    (add_le_add (Flock.zk_server_lean H rel S vs t ht hT1 hδ₁ hS hV hD hc hsc hT (O.release L) wk x w hR hI
      Vs θ hk Ev) le_rfl)

end Claims.ZKServer
