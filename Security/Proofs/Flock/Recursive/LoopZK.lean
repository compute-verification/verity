module

public import Proofs.Flock.Recursive.Loop
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.Recursive.FirewallZK
import Proofs.Flock.Recursive.StopLeak
import Proofs.Flock.Soundness.CROnly.Hiding
import Proofs.Flock.Soundness.ZK.Hiding

/-!
# `RecursiveZK` at the session loop, released at a stop

`Flock.Guarantees.RecursiveZKLoop` is `RecursiveZK` where the ZK server's code (`Flock.Firewall.run`) sends the
commitments, the inner coins live, and the outer session is V*'s statements run by the session loop's switched prover
(`OuterZK.LoopSessions.atWorker` at `OuterZK.loopRef`). The auditor is released its whole view cut at a stop of a public
family, the stop any function of everything.

* `loopReal`, `loopIdeal`: the auditor's whole view, and the simulated one, which reads the tape, the schedule's
  transcript and V*'s public statements (`OuterZK.LoopSessions.pubStmts`), and nothing of the worker's requests,
  messages, witnesses, values or randomness.
* `OuterZK.loopSent`, `OuterZK.loopReal_eq_sent`: the whole view where the ZK server sends the schedule's transcript with
  the salted commit strings in its holes and finishes, which `loopReal` is under `FirewallSends`.
* `OuterZK.loopSent_le`: that view's bound. V*'s statements are zero knowledge at the worker's context
  (`LoopSessions.view_le` at `OuterZK.loop_zkmalR`, the simulator reading no witness: `prCoin_simAt_pubStmts`), then T1
  once per commitment over the adaptive leaves (`FirewallZK.recursiveZK_enc` at `commitBytes`), the salts read from
  their bytes (`ZkView.saltsEquiv`, `FirewallZK.firewallStrings_os`).
* `Flock.SecurityProofs.RecursiveZKLoop`: `prCoin_cut_le` over `loopSent_le`.
-/

@[expose] public section

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game
open scoped ENNReal

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-- **The auditor's whole view of the recursive audit at the session loop** at `u = (((a, p), b), q)`: its tape `a`;
what the ZK server sends (`Flock.Firewall.run s`) on the worker's requests `reqs a p l`, `l` the salted commit strings it
sends of the messages `msgs a p` at the salt bytes `b` (`firewallStrings`, message `i` reading the strings before it),
and whether the inner session finished; and the view of V*'s `m` statements at the tape and the ZK server's commitments
(`OuterZK.LoopSessions.atWorker`), each run in turn by the loop's switched prover at the workers' values `vs` on its own
tape in `q`, statement `j` on the witness `wit j` of the worker's context (`OuterZK.wctxAt`), against `V*` at its tape
`θ` of the tape and the commitments. -/
noncomputable def loopReal {Ωa Ωp Θ : Type} {nh : ℕ} (vs : Claims.ZKServer.Loop.finS.SwVals)
    (L₀ : Claims.ZKServer.LeanRelease Claims.Flock.Wit Claims.ZKServer.Loop.Tp)
    (src : Claims.Msg → Claims.Flock.Wit → Claims.ZKServer.Loop.Tp → ℕ →
      _root_.Flock.V (Array _root_.Flock.F128 × ByteArray × Array ByteArray))
    (s : Flock.Firewall.Schedule) (reqs : Ωa → Ωp → (Fin nh → ByteArray) → List ByteArray)
    (msgs : Ωa → Ωp → (i : Fin nh) → (Fin i → ByteArray) → ByteArray) (m : ℕ)
    (stmt : ℕ → Ωa × (Fin nh → ByteArray) → Claims.Msg) (wit : ℕ → OuterZK.WCtx Ωa Ωp nh → Claims.Flock.Wit)
    (Vs : Claims.VStar Θ) (θ : Ωa → (Fin nh → ByteArray) → Θ)
    (u : ((Ωa × Ωp) × (Fin nh → Fin 192 → Fin 256)) × Claims.Sampling.unitTape.at 0
      ((OuterZK.loopRef L₀ src).swP vs).Nonce ((OuterZK.loopRef L₀ src).swP vs).Rand m) :
    Ωa × (Array Flock.Firewall.Sent × Bool) × List (Claims.Msg × Claims.Out) :=
  (u.1.1.1,
    Flock.Firewall.run s (reqs u.1.1.1 u.1.1.2 (firewallStrings (msgs u.1.1.1 u.1.1.2)
        fun i => FlockSoundness.Audit.Law.streamOf (u.1.2 i)))
      (List.ofFn fun i => FlockSoundness.Audit.Law.streamOf (u.1.2 i)),
    (OuterZK.LoopSessions.atWorker m stmt wit).viewAt Claims.ZKMal.CoinOpen.shaH OuterZK.loopK
      ((OuterZK.loopRef L₀ src).swP vs).prover Vs
      (OuterZK.wctxAt msgs u.1.1.1 u.1.1.2 fun i => ZkView.saltOfBytes (u.1.2 i))
      (θ u.1.1.1 (firewallCommits (msgs u.1.1.1 u.1.1.2) fun i => ZkView.saltOfBytes (u.1.2 i))) u.2)

/-- **The simulated view at the session loop** at `c = ((a, u), s)`: the tape `a`; the schedule's transcript `T` with
`nh` ideal commit strings in its holes (`commitBytes` of a uniform `U` beside a salt's own hash, which read no message),
and the inner session finished; and the simulator of V*'s public statements (`OuterZK.LoopSessions.pubStmts`:
statement `j`'s simulator `Flock.zk_deployZk_sw_stops`'s at `t` rewinds, on its coins in `s`) at the tape and the
ideal commitments' leaves (`idealLeaf`), against `V*` at its tape `θ` of them. It reads nothing of the worker's requests,
messages, witnesses, values or randomness. -/
noncomputable def loopIdeal {Ωa Θ : Type} {nh : ℕ} (t : ℕ)
    (L₀ : Claims.ZKServer.LeanRelease Claims.Flock.Wit Claims.ZKServer.Loop.Tp)
    (src : Claims.Msg → Claims.Flock.Wit → Claims.ZKServer.Loop.Tp → ℕ →
      _root_.Flock.V (Array _root_.Flock.F128 × ByteArray × Array ByteArray))
    (T : (Fin nh → ByteArray) → Array Flock.Firewall.Sent) (m : ℕ)
    (stmt : ℕ → Ωa × (Fin nh → ByteArray) → Claims.Msg) (CB : ℕ → Type) [∀ j, Fintype (CB j)]
    [∀ j, Nonempty (CB j)] (Vs : Claims.VStar Θ) (θ : Ωa → (Fin nh → ByteArray) → Θ)
    (c : (Ωa × (Fin nh → ZkView.Bits × ZkView.Salt)) × OuterZK.LoopSessions.SimCoins CB m) :
    Ωa × (Array Flock.Firewall.Sent × Bool) × List (Claims.Msg × Claims.Out) :=
  (c.1.1, (T fun i => commitBytes (OuterZK.idealStrs c.1.2 i), true),
    (OuterZK.LoopSessions.pubStmts m stmt).simAt (OuterZK.loopRef L₀ src).Nonce (OuterZK.loopRef L₀ src).Tape
      Claims.ZKMal.CoinOpen.shaH OuterZK.loopK (OuterZK.loopSim L₀ src t) CB Vs (c.1.1, fun i => idealLeaf (c.1.2 i))
      (θ c.1.1 fun i => idealLeaf (c.1.2 i)) c.2)

/-- **The recursive audit at the session loop is zero knowledge, released at a stop.** `RecursiveZK` where the
commitments are what the ZK server's code sends and the outer session is V*'s statements run by the session loop. The
auditor's tape is `Ωa` and the worker's randomness `Ωp`. The worker sends the ZK server (`Flock.Firewall.run` at the
pinned schedule `s`) its requests `reqs a p l`; the ZK server salts the `nh` messages and tree tops they commit,
`msgs a p`, each with its own 192 uniform bytes, and sends the auditor the salted commit strings `l`
(`firewallStrings`). The inner coins are live: message `i` and the requests after it read the strings before it. V*'s
`m` statements `stmt j (a, h)` at the tape and the ZK server's commitments `h` (`OuterZK.LoopSessions.atWorker`) are each
one `--zk` session of the loop (`OuterZK.loopRef`), run in turn by its switched prover at the workers' values `vs`, on
the witness `wit j` of the worker's context (`OuterZK.wctxAt`), against the auditor `V*` at its tape `θ a h`. The
auditor's whole view (`loopReal`) is its tape, what the ZK server sends and whether the inner session finished, and V*'s
statements' view. The salt bytes are sampled uniformly here; the ZK server reads them from the operating system (A3,
`uniform/io-getrandombytes`).

It takes, each an obligation:
* the deployed tables' side (`Tables`, `Complete`, `DummyReg`, `Causal`, `TapeOK` of the loop's session, which
  `Loop.tables_dataE` and its siblings prove, and `SwCausalOn` at `Loop.SwOk`'s coins), which takes `SessionsCoined`'s
  role;
* the worker's requests run through the ZK server to the schedule's transcript `T` with the salted commit strings of
  `msgs` in its holes, and finish (`FirewallSends`);
* the worker's values satisfy V*'s circuits: at every tape, randomness and salts, each statement parses and its witness
  satisfies its circuit (`OuterZK.LoopSessions.Honest` at `Claims.Flock.R`);
* V*'s statements fix each one's simulator randomness, `CB j` (`OuterZK.LoopSessions.Sized`).

The simulator (`loopIdeal`) samples the tape, fills `T` with `nh` ideal commit strings, which read no message, and runs
the simulator of V*'s public statements at the tape and the ideal commitments: it reads no request, message, witness,
value or randomness of the worker.

**Where the audit stops.** What the auditor is released is the whole view cut at a stop: `cut s`, for a stop `s` of a
public family `St`, reads only the stop and the whole view. The audit stops at `stop u`, which may depend on anything:
the coins, any further coins `Ωk`, and the witness. Every event of the released view
`cut (stop u) (loopReal … u.1)` is at most `|St|` times as likely as at the simulator's view cut at a uniform stop,
plus
$$\varepsilon_V + n_h \cdot \delta_{K^*},$$
added once: the stop leaks at most `log₂ |St|` bits. `ε_V` is V*'s statements' error (`OuterZK.LoopSessions.err`): the
supremum over contexts of their errors, `Flock.zk_deployZk_sw_stops`'s at `t` rewinds and the pinned key's hiding gap
(`OuterZK.loopEps`), and the composed prover's fork term at `m·4t·3^t`. `δ_{K*}` is the commitments' hiding gap at
hm96-sha512's pinned key, as in `RecursiveZK`. At `St = Ωk = Unit` and `cut` the identity, this is the whole view's
bound. -/
def RecursiveZKLoop : Prop :=
  ∀ {Ωa Ωp Θ St Ωk W : Type} [Fintype Ωa] [Nonempty Ωa] [Fintype Ωp] [Nonempty Ωp] [Fintype St] [Nonempty St]
    [Fintype Ωk] [Nonempty Ωk]
    -- the loop's session: the workers' values, the simulator's rewinds, the release and the frames' source
    (vs : Claims.ZKServer.Loop.finS.SwVals) (t : ℕ)
    (L₀ : Claims.ZKServer.LeanRelease Claims.Flock.Wit Claims.ZKServer.Loop.Tp)
    (src : Claims.Msg → Claims.Flock.Wit → Claims.ZKServer.Loop.Tp → ℕ →
      _root_.Flock.V (Array _root_.Flock.F128 × ByteArray × Array ByteArray))
    -- the ZK server: the pinned schedule, the worker's requests and the messages they commit, each reading the commit
    -- strings sent before it, and the schedule's transcript
    (s : Flock.Firewall.Schedule) (nh : ℕ) (reqs : Ωa → Ωp → (Fin nh → ByteArray) → List ByteArray)
    (msgs : Ωa → Ωp → (i : Fin nh) → (Fin i → ByteArray) → ByteArray)
    (T : (Fin nh → ByteArray) → Array Flock.Firewall.Sent)
    -- V*'s statements at the tape and the commitments, their witnesses at the worker's context, and the simulator's
    -- coins
    (m : ℕ) (stmt : ℕ → Ωa × (Fin nh → ByteArray) → Claims.Msg)
    (wit : ℕ → OuterZK.WCtx Ωa Ωp nh → Claims.Flock.Wit) (CB : ℕ → Type) [∀ j, Fintype (CB j)]
    [∀ j, Nonempty (CB j)]
    -- the auditor and its tape
    (Vs : Claims.VStar Θ) (θ : Ωa → (Fin nh → ByteArray) → Θ),
    0 < t →
    Claims.ZKServer.Loop.finS.Tables →
    Claims.ZKServer.Loop.finS.Complete (Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse) →
    Claims.ZKServer.Loop.finS.DummyReg → Claims.ZKServer.Loop.finS.Causal →
    Claims.ZKServer.Loop.finS.SwCausalOn (Claims.ZKServer.Loop.SwOk Claims.ZKMal.Masked.dataE) vs →
    Claims.ZKServer.Loop.finS.TapeOK →
    FirewallSends (Prog := Unit) (Wit := Unit) (Rr := Unit) s (fun _ _ _ => reqs) (fun _ _ _ => msgs) T →
    (OuterZK.LoopSessions.atWorker m stmt wit).Honest
      (Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse)
      (fun u : (Ωa × Ωp) × (Fin nh → ZkView.Salt) => OuterZK.wctxAt msgs u.1.1 u.1.2 u.2) →
    (OuterZK.LoopSessions.atWorker m stmt wit).Sized (OuterZK.loopSim L₀ src t) CB →
    ∀ (cut : St → Ωa × (Array Flock.Firewall.Sent × Bool) × List (Claims.Msg × Claims.Out) → W)
      (stop : (((Ωa × Ωp) × (Fin nh → Fin 192 → Fin 256)) × Claims.Sampling.unitTape.at 0
        ((OuterZK.loopRef L₀ src).swP vs).Nonce ((OuterZK.loopRef L₀ src).swP vs).Rand m) × Ωk → St)
      (E : W → Prop),
      prCoin (fun u => E (cut (stop u) (loopReal vs L₀ src s reqs msgs m stmt wit Vs θ u.1))) ≤
        (Fintype.card St : ℝ≥0∞) * prCoin (fun c : St × ((Ωa × (Fin nh → ZkView.Bits × ZkView.Salt)) ×
            OuterZK.LoopSessions.SimCoins CB m) => E (cut c.1 (loopIdeal t L₀ src T m stmt CB Vs θ c.2))) +
          (OuterZK.LoopSessions.atWorker m stmt wit).err Claims.ZKMal.CoinOpen.shaH OuterZK.loopK
            ((OuterZK.loopRef L₀ src).swP vs).prover (OuterZK.loopEps L₀ src t) (4 * t * 3 ^ t) Vs
            (fun x => θ x.1 x.2.1) +
          nh * ENNReal.ofReal (CROnly.hidingGap ZkView.My ZkView.saltHash)

end Flock.Guarantees

namespace FlockSoundness.Discharge.Recursion.OuterZK

open Game
open scoped ENNReal

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-- **The auditor's whole view where the ZK server sends the schedule's transcript** `T`, the salted commit strings of
`msgs` in its holes, and finishes: `loopReal` under `FirewallSends` (`loopReal_eq_sent`). -/
noncomputable def loopSent {Ωa Ωp Θ : Type} {nh : ℕ} (vs : Claims.ZKServer.Loop.finS.SwVals)
    (L₀ : Claims.ZKServer.LeanRelease Claims.Flock.Wit Claims.ZKServer.Loop.Tp)
    (src : Claims.Msg → Claims.Flock.Wit → Claims.ZKServer.Loop.Tp → ℕ →
      _root_.Flock.V (Array _root_.Flock.F128 × ByteArray × Array ByteArray))
    (T : (Fin nh → ByteArray) → Array Flock.Firewall.Sent)
    (msgs : Ωa → Ωp → (i : Fin nh) → (Fin i → ByteArray) → ByteArray) (m : ℕ)
    (stmt : ℕ → Ωa × (Fin nh → ByteArray) → Claims.Msg) (wit : ℕ → WCtx Ωa Ωp nh → Claims.Flock.Wit)
    (Vs : Claims.VStar Θ) (θ : Ωa → (Fin nh → ByteArray) → Θ)
    (u : ((Ωa × Ωp) × (Fin nh → Fin 192 → Fin 256)) × Claims.Sampling.unitTape.at 0
      ((loopRef L₀ src).swP vs).Nonce ((loopRef L₀ src).swP vs).Rand m) :
    Ωa × (Array Flock.Firewall.Sent × Bool) × List (Claims.Msg × Claims.Out) :=
  (u.1.1.1,
    (T (_root_.Flock.Guarantees.firewallStrings (msgs u.1.1.1 u.1.1.2)
      fun i => FlockSoundness.Audit.Law.streamOf (u.1.2 i)), true),
    (LoopSessions.atWorker m stmt wit).viewAt Claims.ZKMal.CoinOpen.shaH loopK ((loopRef L₀ src).swP vs).prover Vs
      (wctxAt msgs u.1.1.1 u.1.1.2 fun i => ZkView.saltOfBytes (u.1.2 i))
      (θ u.1.1.1 (_root_.Flock.Guarantees.firewallCommits (msgs u.1.1.1 u.1.1.2) fun i => ZkView.saltOfBytes (u.1.2 i)))
      u.2)

/-- Under `FirewallSends`, the whole view at the session loop is `loopSent`. -/
theorem loopReal_eq_sent {Ωa Ωp Θ : Type} {nh : ℕ} (vs : Claims.ZKServer.Loop.finS.SwVals)
    (L₀ : Claims.ZKServer.LeanRelease Claims.Flock.Wit Claims.ZKServer.Loop.Tp)
    (src : Claims.Msg → Claims.Flock.Wit → Claims.ZKServer.Loop.Tp → ℕ →
      _root_.Flock.V (Array _root_.Flock.F128 × ByteArray × Array ByteArray))
    (s : _root_.Flock.Firewall.Schedule) (reqs : Ωa → Ωp → (Fin nh → ByteArray) → List ByteArray)
    (msgs : Ωa → Ωp → (i : Fin nh) → (Fin i → ByteArray) → ByteArray)
    (T : (Fin nh → ByteArray) → Array Flock.Firewall.Sent) (m : ℕ)
    (stmt : ℕ → Ωa × (Fin nh → ByteArray) → Claims.Msg) (wit : ℕ → WCtx Ωa Ωp nh → Claims.Flock.Wit)
    (Vs : Claims.VStar Θ) (θ : Ωa → (Fin nh → ByteArray) → Θ)
    (hF : _root_.Flock.Guarantees.FirewallSends (Prog := Unit) (Wit := Unit) (Rr := Unit) s (fun _ _ _ => reqs)
      (fun _ _ _ => msgs) T)
    (u : ((Ωa × Ωp) × (Fin nh → Fin 192 → Fin 256)) × Claims.Sampling.unitTape.at 0
      ((loopRef L₀ src).swP vs).Nonce ((loopRef L₀ src).swP vs).Rand m) :
    _root_.Flock.Guarantees.loopReal vs L₀ src s reqs msgs m stmt wit Vs θ u =
      loopSent vs L₀ src T msgs m stmt wit Vs θ u := by
  unfold _root_.Flock.Guarantees.loopReal loopSent
  rw [hF () () () u.1.1.1 u.1.1.2 fun i => FlockSoundness.Audit.Law.streamOf (u.1.2 i)]

/-- **The whole view at the session loop is close to the simulated one**, where the ZK server sends the schedule's
transcript (`loopSent`): V*'s statements at the worker's context are within their error of their public statements'
simulator (`LoopSessions.view_le` at `loop_zkmalR`, `prCoin_simAt_pubStmts`), and then T1 once per commitment over the
adaptive leaves, at the commit strings' bytes (`FirewallZK.recursiveZK_enc` at `commitBytes`). -/
theorem loopSent_le {Ωa Ωp Θ : Type} [Fintype Ωa] [Nonempty Ωa] [Fintype Ωp] [Nonempty Ωp] {nh : ℕ}
    (vs : Claims.ZKServer.Loop.finS.SwVals) (t : ℕ) (ht : 0 < t) (hS : Claims.ZKServer.Loop.finS.Tables)
    (hV : Claims.ZKServer.Loop.finS.Complete
      (Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse))
    (hD : Claims.ZKServer.Loop.finS.DummyReg) (hc : Claims.ZKServer.Loop.finS.Causal)
    (hsc : Claims.ZKServer.Loop.finS.SwCausalOn (Claims.ZKServer.Loop.SwOk Claims.ZKMal.Masked.dataE) vs)
    (hT : Claims.ZKServer.Loop.finS.TapeOK)
    (L₀ : Claims.ZKServer.LeanRelease Claims.Flock.Wit Claims.ZKServer.Loop.Tp)
    (src : Claims.Msg → Claims.Flock.Wit → Claims.ZKServer.Loop.Tp → ℕ →
      _root_.Flock.V (Array _root_.Flock.F128 × ByteArray × Array ByteArray))
    (T : (Fin nh → ByteArray) → Array Flock.Firewall.Sent)
    (msgs : Ωa → Ωp → (i : Fin nh) → (Fin i → ByteArray) → ByteArray) (m : ℕ)
    (stmt : ℕ → Ωa × (Fin nh → ByteArray) → Claims.Msg) (wit : ℕ → WCtx Ωa Ωp nh → Claims.Flock.Wit)
    (CB : ℕ → Type) [∀ j, Fintype (CB j)] [∀ j, Nonempty (CB j)] (Vs : Claims.VStar Θ)
    (θ : Ωa → (Fin nh → ByteArray) → Θ)
    (hH : (LoopSessions.atWorker m stmt wit).Honest
      (Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse)
      (fun u : (Ωa × Ωp) × (Fin nh → ZkView.Salt) => wctxAt msgs u.1.1 u.1.2 u.2))
    (hsz : (LoopSessions.atWorker m stmt wit).Sized (loopSim L₀ src t) CB)
    (E : Ωa × (Array Flock.Firewall.Sent × Bool) × List (Claims.Msg × Claims.Out) → Prop) :
    prCoin (fun u => E (loopSent vs L₀ src T msgs m stmt wit Vs θ u)) ≤
      prCoin (fun c => E (_root_.Flock.Guarantees.loopIdeal t L₀ src T m stmt CB Vs θ c)) +
        ((LoopSessions.atWorker m stmt wit).err Claims.ZKMal.CoinOpen.shaH loopK ((loopRef L₀ src).swP vs).prover
            (loopEps L₀ src t) (4 * t * 3 ^ t) Vs (fun x => θ x.1 x.2.1) +
          nh * ENNReal.ofReal (CROnly.hidingGap ZkView.My ZkView.saltHash)) := by
  have hOuter : ∀ a p (y : Fin nh → ZkView.Salt) (E' : List (Claims.Msg × Claims.Out) → Prop),
      prCoin (fun q : Claims.Sampling.unitTape.at 0 ((loopRef L₀ src).swP vs).Nonce ((loopRef L₀ src).swP vs).Rand m =>
          E' ((LoopSessions.atWorker m stmt wit).viewAt Claims.ZKMal.CoinOpen.shaH loopK
            ((loopRef L₀ src).swP vs).prover Vs
            (a, (fun i => _root_.Flock.Hm96.Default512.treeLeaf
              (FirewallZK.encAt _root_.Flock.Guarantees.commitBytes (msgs a p) y i)), p, y)
            (θ a fun i => _root_.Flock.Hm96.Default512.treeLeaf
              (FirewallZK.encAt _root_.Flock.Guarantees.commitBytes (msgs a p) y i)) q)) ≤
        prCoin (fun s : LoopSessions.SimCoins CB m =>
          E' ((LoopSessions.pubStmts m stmt).simAt (loopRef L₀ src).Nonce (loopRef L₀ src).Tape
            Claims.ZKMal.CoinOpen.shaH loopK (loopSim L₀ src t) CB Vs
            (a, fun i => _root_.Flock.Hm96.Default512.treeLeaf
              (FirewallZK.encAt _root_.Flock.Guarantees.commitBytes (msgs a p) y i))
            (θ a fun i => _root_.Flock.Hm96.Default512.treeLeaf
              (FirewallZK.encAt _root_.Flock.Guarantees.commitBytes (msgs a p) y i)) s)) +
          (LoopSessions.atWorker m stmt wit).err Claims.ZKMal.CoinOpen.shaH loopK ((loopRef L₀ src).swP vs).prover
            (loopEps L₀ src t) (4 * t * 3 ^ t) Vs (fun x => θ x.1 x.2.1) := fun a p y E' =>
    ((LoopSessions.atWorker m stmt wit).view_le Claims.ZKMal.CoinOpen.shaH loopK ((loopRef L₀ src).swP vs).prover
      (Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse) (loopSim L₀ src t)
      (loopEps L₀ src t) (4 * t * 3 ^ t) CB (loop_zkmalR vs t ht hS hV hD hc hsc hT L₀ src)
      (loopEps_nonneg L₀ src t) (by positivity) Vs (fun x => θ x.1 x.2.1) (wctxAt msgs a p y) _ rfl
      (fun j hj => hsz j hj _) (hH ((a, p), y)) E').trans_eq
      (congrArg (· + _) (LoopSessions.prCoin_simAt_pubStmts _ _ _ _ _ _ _ _ _ _ _))
  have h := FirewallZK.recursiveZK_enc _root_.Flock.Guarantees.commitBytes nh msgs
    (fun a l p y q => (LoopSessions.atWorker m stmt wit).viewAt Claims.ZKMal.CoinOpen.shaH loopK
      ((loopRef L₀ src).swP vs).prover Vs (a, (fun i => _root_.Flock.Hm96.Default512.treeLeaf (l i)), p, y)
      (θ a fun i => _root_.Flock.Hm96.Default512.treeLeaf (l i)) q)
    (fun a l (s : LoopSessions.SimCoins CB m) => (LoopSessions.pubStmts m stmt).simAt (loopRef L₀ src).Nonce
      (loopRef L₀ src).Tape Claims.ZKMal.CoinOpen.shaH loopK (loopSim L₀ src t) CB Vs
      (a, fun i => _root_.Flock.Hm96.Default512.treeLeaf (l i)) (θ a fun i => _root_.Flock.Hm96.Default512.treeLeaf (l i))
      s)
    hOuter (CROnly.hm96Hiding_gap ZkView.My ZkView.saltHash) fun x => E (x.1, (T x.2.1, true), x.2.2)
  refine le_of_eq_of_le ?_ (h.trans_eq (add_assoc _ _ _))
  refine Eq.trans ?_ (ZK.prCoin_congr_equiv (((Equiv.refl (Ωa × Ωp)).prodCongr (ZkView.saltsEquiv nh)).prodCongr
    (Equiv.refl (Claims.Sampling.unitTape.at 0 ((loopRef L₀ src).swP vs).Nonce ((loopRef L₀ src).swP vs).Rand m))) _)
  refine congrArg prCoin (funext fun z => ?_)
  exact congrArg (fun f => E (z.1.1.1, (T f, true),
      (LoopSessions.atWorker m stmt wit).viewAt Claims.ZKMal.CoinOpen.shaH loopK ((loopRef L₀ src).swP vs).prover Vs
        (wctxAt msgs z.1.1.1 z.1.1.2 fun i => ZkView.saltOfBytes (z.1.2 i))
        (θ z.1.1.1 (_root_.Flock.Guarantees.firewallCommits (msgs z.1.1.1 z.1.1.2)
          fun i => ZkView.saltOfBytes (z.1.2 i))) z.2))
    (FirewallZK.firewallStrings_os (msgs z.1.1.1 z.1.1.2) z.1.2)

end FlockSoundness.Discharge.Recursion.OuterZK

namespace Flock.SecurityProofs

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game
open scoped ENNReal

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-- **The recursive audit at the session loop is zero knowledge, released at a stop.** -/
theorem RecursiveZKLoop : Flock.Guarantees.RecursiveZKLoop := by
  intro Ωa Ωp Θ St Ωk W _ _ _ _ _ _ _ _ vs t L₀ src s nh reqs msgs T m stmt wit CB _ _ Vs θ ht hS hV hD hc hsc hT
    hF hH hsz cut stop E
  refine (le_of_eq (congrArg prCoin (funext fun u => congrArg (fun v => E (cut (stop u) v))
    (OuterZK.loopReal_eq_sent vs L₀ src s reqs msgs T m stmt wit Vs θ hF u.1)))).trans
    ((prCoin_cut_le (OuterZK.loopSent vs L₀ src T msgs m stmt wit Vs θ)
      (Flock.Guarantees.loopIdeal t L₀ src T m stmt CB Vs θ) cut stop
      (fun E' => OuterZK.loopSent_le vs t ht hS hV hD hc hsc hT L₀ src T msgs m stmt wit CB Vs θ hH hsz E') E).trans
      (le_of_eq (add_assoc _ _ _).symm))

end Flock.SecurityProofs
