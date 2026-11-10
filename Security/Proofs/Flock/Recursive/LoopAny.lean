module

public import Proofs.Flock.Recursive.LoopZK
public import Proofs.Flock.Recursive.FirewallSalted
public import Proofs.Flock.Recursive.FirewallAdaptive
public import Proofs.Flock.Recursive.StopLeak
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
import Proofs.Flock.Soundness.CROnly.Hiding
import Proofs.Flock.Soundness.ZK.Hiding
import Proofs.Flock.Soundness.ZK.RealLeaves

/-!
# `RecursiveZKLoop` at any worker that reads its answers

`Flock.Guarantees.RecursiveZKLoopAnySum` is `RecursiveZKLoop` with no premise on the inner worker: its next request is
any function of what `flock-firewall live` answered it so far, the salts it drew included (`FirewallAdaptive.Worker`),
so it takes no `FirewallSends`. The live firewall then sends the schedule's transcript filled with commit strings whose
digests read only the salts before them, cut at a stop (`Flock.SecurityProofs.adaptOut_any`); V*'s statements read the
commitments the ZK server sent, and run only when the inner session finished. The view is released at one stop of the
inner session or of V*'s statements.

* `LoopAny.holes_fillAll`, `LoopAny.fillAll_take_holes`: a template filled from a list holds the list's first commit
  strings, and reads no more of it than its holes.
* `LoopAny.strs_take`: the live ZK server's commit strings at the salts' bytes are `FirewallZK.encSalted`'s at the
  salts, `ZkView.commitString_split` one string at a time.
* `OuterZK.anySent_le`: the whole view at commit strings whose digests read the salts before them, V*'s statements run
  where their witnesses satisfy their circuits and simulated elsewhere: `FirewallZK.recursiveZK_salted` over
  `LoopSessions.view_le` at `OuterZK.loop_zkmalR`.
* `LoopAny.anyReal_eq`, `LoopAny.prCoin_anyReal`: the real view is that view cut at the session's stop, at every coin
  and so in law at the split coins (`LoopAny.splitSalts`), the two stops joined (`LoopAny.anyStop`).
* `LoopAny.sumCut_toSum`, `LoopAny.prCoin_anyReal_sum`: where the release reads no stop at an unfinished session's
  view (`ReleaseReadsNoStop`), the joined stop is one of the inner session's or one of the audit's (`LoopAny.toSum`),
  a finished session stopping after its last send (`LoopAny.anyStop_last`).
* `Flock.SecurityProofs.RecursiveZKLoopAnySum`: `prCoin_cut_le` over `anySent_le` at that stop.
* `LoopAny.releaseReadsNoStop_loopRelease`: the deployed release (`loopRelease`), V*'s statements' view truncated at
  the stop, reads no stop at an unfinished session's view, whose statements' view is empty.
* `LoopAny.card_loopStops`: at the deployed release the stops are `RecStops`' over every statement, the inner session
  statement `0` with the transcript's messages: `1 + Σ_p (M p + 1)`.
-/

@[expose] public section

namespace FlockSoundness.Discharge.Recursion.LoopAny

open Flock Flock.Firewall FirewallRun FirewallTemplate FirewallAdaptive

/-! ## A template filled from a list -/

theorem length_fillAll : ∀ (ms : List Sent) (cs : List ByteArray), (fillAll ms cs).length = ms.length
  | [], _ => rfl
  | m :: ms, cs => by simp only [fillAll, List.length_cons, length_fillAll ms]

theorem fillRoots_holes_take : ∀ (roots : List (String × List ByteArray)) (cs : List ByteArray),
    (fillRoots roots cs).flatMap (·.2) = cs.take (roots.flatMap (·.2)).length
  | [], _ => by simp [fillRoots]
  | e :: es, cs => by
    simp only [fillRoots, List.flatMap_cons, fillRoots_holes_take es, List.length_append, List.take_add]

theorem fillTops_eq : ∀ (tops : List (List ByteArray)) (cs : List ByteArray),
    (fillTops tops cs).1.flatten = cs.take tops.flatten.length ∧
      (fillTops tops cs).2 = cs.drop tops.flatten.length
  | [], cs => by simp [fillTops]
  | t :: ts, cs => by
    obtain ⟨h1, h2⟩ := fillTops_eq ts (cs.drop t.length)
    simp only [fillTops, List.flatten_cons, h1, h2, List.length_append, List.take_add, List.drop_drop, and_self]

theorem holes_fill (cs : List ByteArray) (m : Sent) : holes (fill cs m) = cs.take (holes m).length := by
  cases m with
  | commit roots => simp only [fill, holes, fillRoots_holes_take]
  | round stream tops level0 rows n =>
    obtain ⟨h1, h2⟩ := fillTops_eq tops cs
    simp only [fill, holes, h1, h2, List.length_append, List.take_add]
  | _ => simp [fill, holes]

/-- **A template filled from `cs` holds `cs`'s first commit strings**, as many as its holes. -/
theorem holes_fillAll : ∀ (ms : List Sent) (cs : List ByteArray),
    (fillAll ms cs).flatMap holes = cs.take (ms.flatMap holes).length
  | [], _ => by simp [fillAll]
  | m :: ms, cs => by
    simp only [fillAll, List.flatMap_cons, holes_fill, holes_fillAll ms, List.length_append, List.take_add]

theorem fillRoots_take : ∀ (roots : List (String × List ByteArray)) (cs : List ByteArray) (k : Nat),
    (roots.flatMap (·.2)).length ≤ k → fillRoots roots (cs.take k) = fillRoots roots cs
  | [], _, _, _ => rfl
  | e :: es, cs, k, hk => by
    simp only [List.flatMap_cons, List.length_append] at hk
    have h1 : (cs.take k).take e.2.length = cs.take e.2.length := by
      rw [List.take_take, Nat.min_eq_left (by omega)]
    have h2 : (cs.take k).drop e.2.length = (cs.drop e.2.length).take (k - e.2.length) := by
      rw [List.drop_take]
    simp only [fillRoots, h1, h2, fillRoots_take es (cs.drop e.2.length) (k - e.2.length) (by omega)]

theorem fillTops_take : ∀ (tops : List (List ByteArray)) (cs : List ByteArray) (k : Nat),
    tops.flatten.length ≤ k →
      fillTops tops (cs.take k) = ((fillTops tops cs).1, (fillTops tops cs).2.take (k - tops.flatten.length))
  | [], cs, k, _ => by simp [fillTops]
  | t :: ts, cs, k, hk => by
    simp only [List.flatten_cons, List.length_append] at hk
    have h1 : (cs.take k).take t.length = cs.take t.length := by
      rw [List.take_take, Nat.min_eq_left (by omega)]
    have h2 : (cs.take k).drop t.length = (cs.drop t.length).take (k - t.length) := by
      rw [List.drop_take]
    have ih := fillTops_take ts (cs.drop t.length) (k - t.length) (by omega)
    simp only [fillTops, h1, h2, ih, List.flatten_cons, List.length_append, Nat.sub_sub]

theorem fill_take (cs : List ByteArray) (m : Sent) (k : Nat) (hk : (holes m).length ≤ k) :
    fill (cs.take k) m = fill cs m := by
  cases m with
  | commit roots => simp only [fill, fillRoots_take roots cs k hk]
  | round stream tops level0 rows n =>
    simp only [holes, List.length_append] at hk
    simp only [fill, fillTops_take tops cs k (by omega), List.take_take,
      Nat.min_eq_left (show rows.length ≤ k - tops.flatten.length by omega)]
  | _ => rfl

/-- **A template filled from `cs` reads no more of `cs` than its holes.** -/
theorem fillAll_take_holes : ∀ (ms : List Sent) (cs : List ByteArray) (k : Nat), (ms.flatMap holes).length ≤ k →
    fillAll ms (cs.take k) = fillAll ms cs
  | [], _, _, _ => rfl
  | m :: ms, cs, k, hk => by
    simp only [List.flatMap_cons, List.length_append] at hk
    have h2 : (cs.take k).drop (holes m).length = (cs.drop (holes m).length).take (k - (holes m).length) := by
      rw [List.drop_take]
    simp only [fillAll, fill_take cs m k (by omega), h2,
      fillAll_take_holes ms (cs.drop (holes m).length) (k - (holes m).length) (by omega)]

end FlockSoundness.Discharge.Recursion.LoopAny

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game
open scoped ENNReal

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-- **The commit strings of the schedule's transcript** (`FirewallTemplate.tmpl` at the schedule's tables): the salts a
finished inner session samples. -/
abbrev fwHoles (s : Flock.Firewall.Schedule) : ℕ :=
  ((FirewallTemplate.tmpl s s.tables).flatMap FirewallRun.holes).length

/-- **The commitments in what the ZK server sent**: the leaf of each commit string it sent, in salt order. -/
def sentCommits (s : Flock.Firewall.Schedule) (sent : List Flock.Firewall.Sent) : Fin (fwHoles s) → ByteArray :=
  fun i => Flock.Hm96.Default512.treeLeaf ((sent.flatMap FirewallRun.holes).getD i ByteArray.empty)

/-- The salts' bytes, in order. -/
def saltsOf {n : ℕ} (b : Fin n → Fin 192 → Fin 256) : List ByteArray :=
  List.ofFn fun i => FlockSoundness.Audit.Law.streamOf (b i)

/-- **One worker's context at the live session** at the salt bytes `b`: the tape, the commitments the ZK server sent
(`sentCommits`), the worker's randomness and the salts of the transcript's commit strings, `b`'s first
`fwHoles s`. -/
def anyCtx {Ωa Ωp : Type} (s : Flock.Firewall.Schedule) (w : Ωa → Ωp → FirewallAdaptive.Worker) (N r : ℕ) (a : Ωa)
    (p : Ωp) (b : Fin (fwHoles s + r) → Fin 192 → Fin 256) : OuterZK.WCtx Ωa Ωp (fwHoles s) :=
  (a, sentCommits s (FirewallAdaptive.adaptOut s (w a p) N { salts := [] } [] (saltsOf b)).1, p,
    fun i => ZkView.saltOfBytes (b (Fin.castAdd r i)))

/-- **The auditor's whole view of the recursive audit at the session loop, at any worker** at
`u = (((a, p), b), q)`: its tape `a`; what the live firewall sent the challenger's live-coin part (`FirewallAdaptive.adaptOut`) as the
worker `w a p` took its answers, for at most `N` requests, at the salt bytes `b`, and whether the inner session
finished; and, if it finished, the view of V*'s `m` statements at the tape and the commitments the ZK server sent
(`OuterZK.LoopSessions.atWorker` at `anyCtx`), each run in turn by the loop's switched prover at the workers' values
`vs` on its own tape in `q`, against `V*` at its tape `θ` of the tape and the commitments. -/
noncomputable def anyReal {Ωa Ωp Θ : Type} (vs : Claims.ZKServer.Loop.finS.SwVals)
    (L₀ : Claims.ZKServer.LeanRelease Claims.Flock.Wit Claims.ZKServer.Loop.Tp)
    (src : Claims.Msg → Claims.Flock.Wit → Claims.ZKServer.Loop.Tp → ℕ →
      _root_.Flock.V (Array _root_.Flock.F128 × ByteArray × Array ByteArray))
    (s : Flock.Firewall.Schedule) (w : Ωa → Ωp → FirewallAdaptive.Worker) (N r m : ℕ)
    (stmt : ℕ → Ωa × (Fin (fwHoles s) → ByteArray) → Claims.Msg)
    (wit : ℕ → OuterZK.WCtx Ωa Ωp (fwHoles s) → Claims.Flock.Wit) (Vs : Claims.VStar Θ)
    (θ : Ωa → (Fin (fwHoles s) → ByteArray) → Θ)
    (u : ((Ωa × Ωp) × (Fin (fwHoles s + r) → Fin 192 → Fin 256)) × Claims.Sampling.unitTape.at 0
      ((OuterZK.loopRef L₀ src).swP vs).Nonce ((OuterZK.loopRef L₀ src).swP vs).Rand m) :
    Ωa × (Array Flock.Firewall.Sent × Bool) × List (Claims.Msg × Claims.Out) :=
  (u.1.1.1,
    ((FirewallAdaptive.adaptOut s (w u.1.1.1 u.1.1.2) N { salts := [] } [] (saltsOf u.1.2)).1.toArray,
      FirewallAdaptive.adaptFinished s (w u.1.1.1 u.1.1.2) N (saltsOf u.1.2)),
    if FirewallAdaptive.adaptFinished s (w u.1.1.1 u.1.1.2) N (saltsOf u.1.2) then
      (OuterZK.LoopSessions.atWorker m stmt wit).viewAt Claims.ZKMal.CoinOpen.shaH OuterZK.loopK
        ((OuterZK.loopRef L₀ src).swP vs).prover Vs (anyCtx s w N r u.1.1.1 u.1.1.2 u.1.2)
        (θ u.1.1.1 (anyCtx s w N r u.1.1.1 u.1.1.2 u.1.2).2.1) u.2
    else [])

/-- **A view cut at the inner session's stop** `c = (k, f)`: its first `k` sends, finished exactly when `f`, and V*'s
statements' view only if it finished. -/
def innerCut {Ωa : Type} {K : ℕ} (c : Fin (K + 1) × Bool)
    (v : Ωa × (Array Flock.Firewall.Sent × Bool) × List (Claims.Msg × Claims.Out)) :
    Ωa × (Array Flock.Firewall.Sent × Bool) × List (Claims.Msg × Claims.Out) :=
  (v.1, ((v.2.1.1.toList.take c.1).toArray, c.2), if c.2 then v.2.2 else [])

/-- **The schedule's transcript filled with the commit strings `l`**, in salt order. -/
def fwFilled (s : Flock.Firewall.Schedule) (l : Fin (fwHoles s) → ByteArray) : Array Flock.Firewall.Sent :=
  (FirewallRun.fillAll (FirewallTemplate.tmpl s s.tables) (List.ofFn l)).toArray

/-- **A view cut at one stop of the inner session or the audit**: at `.inl k`, the inner session's first `k` sends,
unfinished, released at the stop `s₀`; at `.inr c`, all of them, finished, released at the audit's stop `c`. -/
def sumCut {Ωa St W : Type} {K : ℕ} (s₀ : St)
    (cut : St → Ωa × (Array Flock.Firewall.Sent × Bool) × List (Claims.Msg × Claims.Out) → W) :
    Fin (K + 1) ⊕ St → Ωa × (Array Flock.Firewall.Sent × Bool) × List (Claims.Msg × Claims.Out) → W
  | .inl k, v => cut s₀ (innerCut (k, false) v)
  | .inr c, v => cut c (innerCut (Fin.last K, true) v)

/-- Obligation (the release reads no stop where the inner session didn't finish): at such a view, which holds no
statement's view, `cut c` is `cut s₀` for every stop `c`. -/
def ReleaseReadsNoStop {Ωa St W : Type}
    (cut : St → Ωa × (Array Flock.Firewall.Sent × Bool) × List (Claims.Msg × Claims.Out) → W) (s₀ : St) : Prop :=
  ∀ (c : St) (a : Ωa) (sent : Array Flock.Firewall.Sent),
    cut c (a, (sent, false), []) = cut s₀ (a, (sent, false), [])

/-- **The deployed release**: V*'s statements' view truncated at the stop, `M p` statement `p`'s message count. At
none, every message of every statement; at the first statement to stop, `q`, after its `k`-th message, every message
of each statement before `q` and `k` of `q`'s. The tape and what the ZK server sent are released whole. -/
def loopRelease {Ωa : Type} {n : ℕ} (M : Fin n → ℕ) :
    Flock.SecurityProofs.RecStops M → Ωa × (Array Flock.Firewall.Sent × Bool) × List (Claims.Msg × Claims.Out) →
      Ωa × (Array Flock.Firewall.Sent × Bool) × List (Claims.Msg × Claims.Out)
  | none, v => v
  | some ⟨q, k⟩, v => (v.1, v.2.1, v.2.2.take (∑ p ∈ Finset.univ.filter (· < q), M p + k))

/-- **The recursive audit at the session loop is zero knowledge at any worker, released at a stop.**
`RecursiveZKLoop` with no premise on the inner worker. The auditor's tape is `Ωa` and the worker's randomness `Ωp`.
The inner worker `w a p` sends `flock-firewall live` (`Flock.Firewall.liveWith` at the pinned schedule `s`, modelled by
`FirewallAdaptive.adaptOut`) at most `N` requests, each any function of what the ZK server answered it so far: what it
sent the challenger's live-coin part for each request and the salts it drew for them. The ZK server's salts are `fwHoles s + r`
strings of 192 uniform bytes `b`, read in turn: the first `fwHoles s` are those the schedule's transcript's commit
strings can sample, and `r` more cover the fresh salts a session reads beyond those it samples. V*'s `m` statements
`stmt j (a, h)` at the tape and the commitments `h` the ZK server sent (`OuterZK.LoopSessions.atWorker` at `anyCtx`) are
each one `--zk` session of the loop (`OuterZK.loopRef`), run in turn by its switched prover at the workers' values
`vs`, on the witness `wit j` of the worker's context, against the auditor `V*` at its tape `θ a h`, once the inner
session finished. The auditor's whole view (`anyReal`) is its tape, what the ZK server sent and whether the inner
session finished, and V*'s statements' view if it did. The salt bytes are sampled uniformly here; the ZK server reads
them from the operating system (A3, `uniform/io-getrandombytes`).

It takes, each an obligation:
* the deployed tables' side (`Tables`, `Complete`, `DummyReg`, `Causal`, `TapeOK` of the loop's session, which
  `Loop.tables_dataE` and its siblings prove, and `SwCausalOn` at `Loop.SwOk`'s coins), which takes `SessionsCoined`'s
  role;
* the worker's values satisfy V*'s circuits wherever the inner session finished: at every such tape, randomness and
  salts, each statement parses and its witness satisfies its circuit (`OuterZK.LoopSessions.Honest` at
  `Claims.Flock.R`);
* V*'s statements fix each one's simulator randomness, `CB j` (`OuterZK.LoopSessions.Sized`);
* the release reads no stop where the inner session didn't finish (`ReleaseReadsNoStop`), which the deployed release
  meets (`LoopAny.releaseReadsNoStop_loopRelease`).

It takes no `FirewallSends`: whatever the worker, the ZK server sends the schedule's transcript filled with commit
strings whose digests read only the salts before them, cut after one of its messages, and finishes only having sent it
all (`Flock.SecurityProofs.adaptOut_any`).

The simulator samples the tape, fills the schedule's transcript (`fwFilled`) with `fwHoles s` ideal commit strings, which
read no message, runs the simulator of V*'s public statements at the tape and the ideal commitments (`loopIdeal`), and
releases what it sent at a uniform stop (`sumCut`): it reads no request, message, witness, value or randomness of the
worker.

**Where the audit stops.** What the auditor is released is its whole view cut at the audit's stop: `cut c`, for a stop
`c` of a public family `St`, reads only the stop and the view. The audit stops at `stop u`, which may depend on
anything: the coins, any further coins `Ωk`, and the witness. The inner session stops after `k` of the `|T|` messages
of the schedule's transcript `T`, wherever the worker and the ZK server stop it, and V*'s statements run only if it
finished; where it didn't, the release reads no stop. So the stops are the inner session's `k` where it didn't finish,
released at `s₀`, and the audit's where it did (`sumCut`): every event of the released view
`cut (stop u) (anyReal … u.1)` is at most `|T| + 1 + |St|` times as likely as at the simulator's view released at a
uniform such stop, plus
$$\varepsilon_V + n_h \cdot \delta_{K^*},$$
added once: the stops leak at most `log₂ (|T| + 1 + |St|)` bits. `ε_V` and `δ_{K*}` are `RecursiveZKLoop`'s, and `n_h`
is `fwHoles s`, the transcript's commit strings. At the deployed release (`loopRelease`, `s₀` none), `St` is V*'s
statements' `RecStops`, and `|T| + 1 + |St|` is `1 + Σ_p (M p + 1)` over every statement, the inner session statement
`0` with its `|T|` messages (`LoopAny.card_loopStops`). -/
def RecursiveZKLoopAnySum : Prop :=
  ∀ {Ωa Ωp Θ St Ωk W : Type} [Fintype Ωa] [Nonempty Ωa] [Fintype Ωp] [Nonempty Ωp] [Fintype St] [Nonempty St]
    [Fintype Ωk] [Nonempty Ωk]
    -- the loop's session: the workers' values, the simulator's rewinds, the release and the frames' source
    (vs : Claims.ZKServer.Loop.finS.SwVals) (t : ℕ)
    (L₀ : Claims.ZKServer.LeanRelease Claims.Flock.Wit Claims.ZKServer.Loop.Tp)
    (src : Claims.Msg → Claims.Flock.Wit → Claims.ZKServer.Loop.Tp → ℕ →
      _root_.Flock.V (Array _root_.Flock.F128 × ByteArray × Array ByteArray))
    -- the ZK server: the pinned schedule, the inner worker at the tape and its randomness, its most requests, and the
    -- salts beyond the transcript's
    (s : Flock.Firewall.Schedule) (w : Ωa → Ωp → FirewallAdaptive.Worker) (N r : ℕ)
    -- V*'s statements at the tape and the commitments, their witnesses at the worker's context, and the simulator's
    -- coins
    (m : ℕ) (stmt : ℕ → Ωa × (Fin (fwHoles s) → ByteArray) → Claims.Msg)
    (wit : ℕ → OuterZK.WCtx Ωa Ωp (fwHoles s) → Claims.Flock.Wit) (CB : ℕ → Type) [∀ j, Fintype (CB j)]
    [∀ j, Nonempty (CB j)]
    -- the auditor and its tape
    (Vs : Claims.VStar Θ) (θ : Ωa → (Fin (fwHoles s) → ByteArray) → Θ),
    0 < t →
    Claims.ZKServer.Loop.finS.Tables →
    Claims.ZKServer.Loop.finS.Complete (Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse) →
    Claims.ZKServer.Loop.finS.DummyReg → Claims.ZKServer.Loop.finS.Causal →
    Claims.ZKServer.Loop.finS.SwCausalOn (Claims.ZKServer.Loop.SwOk Claims.ZKMal.Masked.dataE) vs →
    Claims.ZKServer.Loop.finS.TapeOK →
    (OuterZK.LoopSessions.atWorker m stmt wit).Honest
      (Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse)
      (fun u : {u : (Ωa × Ωp) × (Fin (fwHoles s + r) → Fin 192 → Fin 256) //
          FirewallAdaptive.adaptFinished s (w u.1.1 u.1.2) N (saltsOf u.2) = true} =>
        anyCtx s w N r u.1.1.1 u.1.1.2 u.1.2) →
    (OuterZK.LoopSessions.atWorker m stmt wit).Sized (OuterZK.loopSim L₀ src t) CB →
    -- the release at the audit's stops, and the stop it is at where the inner session didn't finish
    ∀ (cut : St → Ωa × (Array Flock.Firewall.Sent × Bool) × List (Claims.Msg × Claims.Out) → W) (s₀ : St),
      ReleaseReadsNoStop cut s₀ →
    ∀ (stop : (((Ωa × Ωp) × (Fin (fwHoles s + r) → Fin 192 → Fin 256)) × Claims.Sampling.unitTape.at 0
        ((OuterZK.loopRef L₀ src).swP vs).Nonce ((OuterZK.loopRef L₀ src).swP vs).Rand m) × Ωk → St)
      (E : W → Prop),
      prCoin (fun u => E (cut (stop u) (anyReal vs L₀ src s w N r m stmt wit Vs θ u.1))) ≤
        (Fintype.card (Fin ((FirewallTemplate.tmpl s s.tables).length + 1) ⊕ St) : ℝ≥0∞) *
            prCoin (fun c : (Fin ((FirewallTemplate.tmpl s s.tables).length + 1) ⊕ St) ×
                ((Ωa × (Fin (fwHoles s) → ZkView.Bits × ZkView.Salt)) × OuterZK.LoopSessions.SimCoins CB m) =>
              E (sumCut s₀ cut c.1 (loopIdeal t L₀ src (fwFilled s) m stmt CB Vs θ c.2))) +
          (OuterZK.LoopSessions.atWorker m stmt wit).err Claims.ZKMal.CoinOpen.shaH OuterZK.loopK
            ((OuterZK.loopRef L₀ src).swP vs).prover (OuterZK.loopEps L₀ src t) (4 * t * 3 ^ t) Vs
            (fun x => θ x.1 x.2.1) +
          fwHoles s * ENNReal.ofReal (CROnly.hidingGap ZkView.My ZkView.saltHash)

end Flock.Guarantees

namespace FlockSoundness.Discharge.Recursion.LoopAny

open Flock Flock.Firewall FirewallRun FirewallTemplate FirewallAdaptive

/-! ## The live ZK server's commit strings at the salts -/

/-- **The live ZK server's commit strings at the salts' bytes are `encSalted`'s at the salts**: string `i`, the commit
string of digest `dg i` at the salts before `i` under salt `i`, is `commitBytes` of its two halves at the salt
`saltOfBytes (b i)` (`ZkView.commitString_split`), its digest reading those salts' bytes. -/
theorem strs_take {nh r : ℕ} (dg : ℕ → List ByteArray → ByteArray) (b : Fin (nh + r) → Fin 192 → Fin 256) :
    (strs dg (Flock.Guarantees.saltsOf b)).take nh = List.ofFn (FirewallZK.encSalted Flock.Guarantees.commitBytes
      (fun i pre => dg i (List.ofFn fun j => ZkView.saltBytes (pre j)))
      fun i => ZkView.saltOfBytes (b (Fin.castAdd r i))) := by
  have hy : ∀ (j : ℕ) (hj : j < nh + r),
      (Flock.Guarantees.saltsOf b)[j]'(by rw [Flock.Guarantees.saltsOf, List.length_ofFn]; exact hj) =
        ZkView.saltBytes (ZkView.saltOfBytes (b ⟨j, hj⟩)) := by
    intro j hj
    rw [ZkView.saltBytes_saltOfBytes]
    simp only [Flock.Guarantees.saltsOf, List.getElem_ofFn]
  have htake : ∀ (i : ℕ) (hi : i ≤ nh + r), (Flock.Guarantees.saltsOf b).take i =
      List.ofFn fun j : Fin i => ZkView.saltBytes (ZkView.saltOfBytes (b (Fin.castLE hi j))) := by
    intro i hi
    apply List.ext_getElem
    · simp only [List.length_take, Flock.Guarantees.saltsOf, List.length_ofFn]
      omega
    · intro j h1 h2
      have hj : j < i := by simpa using h2
      rw [List.getElem_take, hy j (by omega), List.getElem_ofFn]
      rfl
  apply List.ext_getElem
  · simp only [List.length_take, length_strs, Flock.Guarantees.saltsOf, List.length_ofFn]
    omega
  · intro i h1 h2
    have hi : i < nh := by simpa using h2
    rw [List.getElem_take, getElem_strs, hy i (by omega), htake i (by omega), ZkView.commitString_split,
      List.getElem_ofFn]
    rfl

/-- **The commitments in the filled transcript** are the leaves of the commit strings it was filled with. -/
theorem sentCommits_filled (s : Flock.Firewall.Schedule) (l : Fin (Flock.Guarantees.fwHoles s) → ByteArray) :
    Flock.Guarantees.sentCommits s (fillAll (tmpl s s.tables) (List.ofFn l)) =
      fun i => Hm96.Default512.treeLeaf (l i) := by
  funext i
  unfold Flock.Guarantees.sentCommits
  rw [holes_fillAll, List.take_of_length_le (le_of_eq List.length_ofFn),
    ← List.getElem_eq_getD (h := by rw [List.length_ofFn]; exact i.isLt) ByteArray.empty, List.getElem_ofFn]

/-- **The salts' bytes as the transcript's salts and the rest**, beside the other coins: the first `n` byte strings as
salts (`ZkView.saltsEquiv`), the last `r` beside the further coins. -/
noncomputable def splitSalts (A Q K S : Type) (n r : ℕ) :
    (((A × (Fin (n + r) → Fin 192 → Fin 256)) × Q) × K) × S ≃
      ((A × (Fin n → ZkView.Salt)) × (Q × S)) × ((Fin r → Fin 192 → Fin 256) × K) where
  toFun x := (((x.1.1.1.1, ZkView.saltsEquiv n fun i => x.1.1.1.2 (Fin.castAdd r i)), (x.1.1.2, x.2)),
    (fun i => x.1.1.1.2 (Fin.natAdd n i), x.1.2))
  invFun y := ((((y.1.1.1, Fin.append ((ZkView.saltsEquiv n).symm y.1.1.2) y.2.1), y.1.2.1), y.2.2), y.1.2.2)
  left_inv x := by
    simp only [Equiv.symm_apply_apply, Fin.append_castAdd_natAdd, Prod.mk.eta]
  right_inv y := by
    simp only [Fin.append_left, Fin.append_right, Equiv.apply_symm_apply, Prod.mk.eta]

theorem splitSalts_fst (A Q K S : Type) (n r : ℕ) (x : (((A × (Fin (n + r) → Fin 192 → Fin 256)) × Q) × K) × S) :
    (splitSalts A Q K S n r x).1 =
      ((x.1.1.1.1, fun i => ZkView.saltOfBytes (x.1.1.1.2 (Fin.castAdd r i))), (x.1.1.2, x.2)) :=
  rfl

/-- A stop `(k, f)` of the live session, `k` taken at most `K`. -/
def clampStop {K : ℕ} (c : ℕ × Bool) : Fin (K + 1) × Bool :=
  (⟨min c.1 K, Nat.lt_succ_of_le (Nat.min_le_right _ _)⟩, c.2)

/-- **The stop at the split coins** `y`: at the coins `y` comes from (`splitSalts`), the audit's stop `stop` and the
live session's, `σ` at the worker and the salts' bytes, `k` taken at most `K`. -/
noncomputable def anyStop {Ωa Ωp Q Ωk S St : Type} {K nh r : ℕ}
    (stop : (((Ωa × Ωp) × (Fin (nh + r) → Fin 192 → Fin 256)) × Q) × Ωk → St)
    (σ : Ωa → Ωp → List ByteArray → ℕ × Bool)
    (y : (((Ωa × Ωp) × (Fin nh → ZkView.Salt)) × (Q × S)) × ((Fin r → Fin 192 → Fin 256) × Ωk)) :
    St × (Fin (K + 1) × Bool) :=
  let x := (splitSalts (Ωa × Ωp) Q Ωk S nh r).symm y
  (stop x.1, clampStop (σ x.1.1.1.1.1 x.1.1.1.1.2 (Flock.Guarantees.saltsOf x.1.1.1.2)))

theorem anyStop_split {Ωa Ωp Q Ωk S St : Type} {K nh r : ℕ}
    (stop : (((Ωa × Ωp) × (Fin (nh + r) → Fin 192 → Fin 256)) × Q) × Ωk → St)
    (σ : Ωa → Ωp → List ByteArray → ℕ × Bool)
    (x : ((((Ωa × Ωp) × (Fin (nh + r) → Fin 192 → Fin 256)) × Q) × Ωk) × S) :
    anyStop (K := K) stop σ (splitSalts (Ωa × Ωp) Q Ωk S nh r x) =
      (stop x.1, clampStop (σ x.1.1.1.1.1 x.1.1.1.1.2 (Flock.Guarantees.saltsOf x.1.1.1.2))) := by
  simp only [anyStop, Equiv.symm_apply_apply]

/-- A joined stop `(c, (k, f))` as one of `sumCut`'s: the inner session's `k` where it didn't finish, the audit's `c`
where it did. -/
def toSum {St : Type} {K : ℕ} (c : St × (Fin (K + 1) × Bool)) : Fin (K + 1) ⊕ St :=
  bif c.2.2 then .inr c.1 else .inl c.2.1

/-- **A finished session stops after its last send**, where `σ` does. -/
theorem anyStop_last {Ωa Ωp Q Ωk S St : Type} {K nh r : ℕ}
    (stop : (((Ωa × Ωp) × (Fin (nh + r) → Fin 192 → Fin 256)) × Q) × Ωk → St)
    (σ : Ωa → Ωp → List ByteArray → ℕ × Bool)
    (hσ : ∀ (a : Ωa) (p : Ωp) (b : Fin (nh + r) → Fin 192 → Fin 256),
      (σ a p (Flock.Guarantees.saltsOf b)).2 = true → (σ a p (Flock.Guarantees.saltsOf b)).1 = K)
    (y : (((Ωa × Ωp) × (Fin nh → ZkView.Salt)) × (Q × S)) × ((Fin r → Fin 192 → Fin 256) × Ωk)) :
    (anyStop (K := K) stop σ y).2.2 = true → (anyStop (K := K) stop σ y).2.1 = Fin.last K := by
  intro hf
  dsimp only [anyStop, clampStop] at hf ⊢
  exact Fin.ext ((congrArg (min · K) (hσ _ _ _ hf)).trans (Nat.min_self K))

/-- **`sumCut` at the joined stop is the cut at it**, where the release reads no stop at an unfinished session's view
and a finished session stops after its last send. -/
theorem sumCut_toSum {Ωa St W : Type} {K : ℕ} (s₀ : St)
    (cut : St → Ωa × (Array Flock.Firewall.Sent × Bool) × List (Claims.Msg × Claims.Out) → W)
    (hcut : Flock.Guarantees.ReleaseReadsNoStop cut s₀)
    (c : St × (Fin (K + 1) × Bool)) (hc : c.2.2 = true → c.2.1 = Fin.last K)
    (v : Ωa × (Array Flock.Firewall.Sent × Bool) × List (Claims.Msg × Claims.Out)) :
    Flock.Guarantees.sumCut s₀ cut (toSum c) v = cut c.1 (Flock.Guarantees.innerCut c.2 v) := by
  obtain ⟨c, k, f⟩ := c
  cases f with
  | false => exact (hcut c v.1 _).symm
  | true =>
    have hk : k = Fin.last K := hc rfl
    subst hk
    rfl

/-- **The deployed release reads no stop where the inner session didn't finish**: V*'s statements' view is empty
there, and so is every truncation of it. -/
theorem releaseReadsNoStop_loopRelease {Ωa : Type} {n : ℕ} (M : Fin n → ℕ) :
    Flock.Guarantees.ReleaseReadsNoStop (Flock.Guarantees.loopRelease (Ωa := Ωa) M) none := by
  rintro (_ | ⟨q, k⟩) a sent <;> simp [Flock.Guarantees.loopRelease]

/-- **The stops at the deployed release are `RecStops`' over every statement**: the inner session's `K + 1` where it
didn't finish and V*'s statements' `RecStops M` where it did are as many as the stops of every statement, the inner
session statement `0` with its `K` messages, `1 + Σ_p (M p + 1)` (`card_recStops`). -/
theorem card_loopStops {n : ℕ} (K : ℕ) (M : Fin n → ℕ) :
    Fintype.card (Fin (K + 1) ⊕ Flock.SecurityProofs.RecStops M) =
      Fintype.card (Flock.SecurityProofs.RecStops (Fin.cons K M : Fin (n + 1) → ℕ)) := by
  rw [Fintype.card_sum, Fintype.card_fin, Flock.SecurityProofs.card_recStops, Flock.SecurityProofs.card_recStops,
    Fin.sum_univ_succ]
  simp only [Fin.cons_zero, Fin.cons_succ]
  omega

end FlockSoundness.Discharge.Recursion.LoopAny

namespace FlockSoundness.Discharge.Recursion.OuterZK

open Game
open scoped ENNReal

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

theorem prCoin_snd {C D : Type} [Fintype C] [Nonempty C] [Fintype D] (P : D → Prop) :
    prCoin (fun p : C × D => P p.2) = prCoin P :=
  (ZK.prCoin_congr_equiv (Equiv.prodComm C D) (fun p : D × C => P p.1)).trans (ZK.prCoin_fst P)

open Classical in
/-- **V*'s statements at one worker's commit strings `l`, where their witnesses satisfy their circuits**: run in turn by
the loop's switched prover at the worker's context, on the tape `q.1`, if each statement's witness is in `R` at it;
elsewhere their public statements' simulator on the coins `q.2`. -/
noncomputable def anyOuter {Ωa Ωp Θ : Type} {nh : ℕ} (vs : Claims.ZKServer.Loop.finS.SwVals) (t : ℕ)
    (L₀ : Claims.ZKServer.LeanRelease Claims.Flock.Wit Claims.ZKServer.Loop.Tp)
    (src : Claims.Msg → Claims.Flock.Wit → Claims.ZKServer.Loop.Tp → ℕ →
      _root_.Flock.V (Array _root_.Flock.F128 × ByteArray × Array ByteArray))
    (m : ℕ) (stmt : ℕ → Ωa × (Fin nh → ByteArray) → Claims.Msg) (wit : ℕ → WCtx Ωa Ωp nh → Claims.Flock.Wit)
    (CB : ℕ → Type) [∀ j, Fintype (CB j)] [∀ j, Nonempty (CB j)] (Vs : Claims.VStar Θ)
    (θ : Ωa → (Fin nh → ByteArray) → Θ) (a : Ωa) (l : Fin nh → ByteArray) (p : Ωp) (y : Fin nh → ZkView.Salt)
    (q : Claims.Sampling.unitTape.at 0 ((loopRef L₀ src).swP vs).Nonce ((loopRef L₀ src).swP vs).Rand m ×
      LoopSessions.SimCoins CB m) : List (Claims.Msg × Claims.Out) :=
  if (LoopSessions.atWorker m stmt wit).Honest
      (Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse)
      (fun _ : Unit => ((a, fun i => _root_.Flock.Hm96.Default512.treeLeaf (l i), p, y) : WCtx Ωa Ωp nh)) then
    (LoopSessions.atWorker m stmt wit).viewAt Claims.ZKMal.CoinOpen.shaH loopK ((loopRef L₀ src).swP vs).prover Vs
      (a, fun i => _root_.Flock.Hm96.Default512.treeLeaf (l i), p, y)
      (θ a fun i => _root_.Flock.Hm96.Default512.treeLeaf (l i)) q.1
  else
    (LoopSessions.pubStmts m stmt).simAt (loopRef L₀ src).Nonce (loopRef L₀ src).Tape Claims.ZKMal.CoinOpen.shaH
      loopK (loopSim L₀ src t) CB Vs (a, fun i => _root_.Flock.Hm96.Default512.treeLeaf (l i))
      (θ a fun i => _root_.Flock.Hm96.Default512.treeLeaf (l i)) q.2

/-- **`anyOuter` is close to V*'s public statements' simulator**: where the witnesses satisfy their circuits,
`LoopSessions.view_le` at `loop_zkmalR`; elsewhere it is that simulator. -/
theorem anyOuter_le {Ωa Ωp Θ : Type} {nh : ℕ} (vs : Claims.ZKServer.Loop.finS.SwVals) (t : ℕ) (ht : 0 < t)
    (hS : Claims.ZKServer.Loop.finS.Tables)
    (hV : Claims.ZKServer.Loop.finS.Complete
      (Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse))
    (hD : Claims.ZKServer.Loop.finS.DummyReg) (hc : Claims.ZKServer.Loop.finS.Causal)
    (hsc : Claims.ZKServer.Loop.finS.SwCausalOn (Claims.ZKServer.Loop.SwOk Claims.ZKMal.Masked.dataE) vs)
    (hT : Claims.ZKServer.Loop.finS.TapeOK)
    (L₀ : Claims.ZKServer.LeanRelease Claims.Flock.Wit Claims.ZKServer.Loop.Tp)
    (src : Claims.Msg → Claims.Flock.Wit → Claims.ZKServer.Loop.Tp → ℕ →
      _root_.Flock.V (Array _root_.Flock.F128 × ByteArray × Array ByteArray))
    (m : ℕ) (stmt : ℕ → Ωa × (Fin nh → ByteArray) → Claims.Msg) (wit : ℕ → WCtx Ωa Ωp nh → Claims.Flock.Wit)
    (CB : ℕ → Type) [∀ j, Fintype (CB j)] [∀ j, Nonempty (CB j)] (Vs : Claims.VStar Θ)
    (θ : Ωa → (Fin nh → ByteArray) → Θ) (hsz : (LoopSessions.atWorker m stmt wit).Sized (loopSim L₀ src t) CB)
    (a : Ωa) (l : Fin nh → ByteArray) (p : Ωp) (y : Fin nh → ZkView.Salt)
    (E : List (Claims.Msg × Claims.Out) → Prop) :
    prCoin (fun q => E (anyOuter vs t L₀ src m stmt wit CB Vs θ a l p y q)) ≤
      prCoin (fun s : LoopSessions.SimCoins CB m => E ((LoopSessions.pubStmts m stmt).simAt (loopRef L₀ src).Nonce
          (loopRef L₀ src).Tape Claims.ZKMal.CoinOpen.shaH loopK (loopSim L₀ src t) CB Vs
          (a, fun i => _root_.Flock.Hm96.Default512.treeLeaf (l i))
          (θ a fun i => _root_.Flock.Hm96.Default512.treeLeaf (l i)) s)) +
        (LoopSessions.atWorker m stmt wit).err Claims.ZKMal.CoinOpen.shaH loopK ((loopRef L₀ src).swP vs).prover
          (loopEps L₀ src t) (4 * t * 3 ^ t) Vs (fun x => θ x.1 x.2.1) := by
  unfold anyOuter
  by_cases hh : (LoopSessions.atWorker m stmt wit).Honest
      (Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse)
      (fun _ : Unit => ((a, fun i => _root_.Flock.Hm96.Default512.treeLeaf (l i), p, y) : WCtx Ωa Ωp nh))
  · simp only [ite_eq_left hh]
    exact (ZK.prCoin_fst (E := LoopSessions.SimCoins CB m) (fun q => E ((LoopSessions.atWorker m stmt wit).viewAt
        Claims.ZKMal.CoinOpen.shaH loopK ((loopRef L₀ src).swP vs).prover Vs
        (a, fun i => _root_.Flock.Hm96.Default512.treeLeaf (l i), p, y)
        (θ a fun i => _root_.Flock.Hm96.Default512.treeLeaf (l i)) q))).trans_le
      (((LoopSessions.atWorker m stmt wit).view_le Claims.ZKMal.CoinOpen.shaH loopK ((loopRef L₀ src).swP vs).prover
        (Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse) (loopSim L₀ src t)
        (loopEps L₀ src t) (4 * t * 3 ^ t) CB (loop_zkmalR vs t ht hS hV hD hc hsc hT L₀ src)
        (loopEps_nonneg L₀ src t) (by positivity) Vs (fun x => θ x.1 x.2.1)
        (a, fun i => _root_.Flock.Hm96.Default512.treeLeaf (l i), p, y)
        (a, fun i => _root_.Flock.Hm96.Default512.treeLeaf (l i)) rfl (fun j hj => hsz j hj _) (hh ()) E).trans_eq
        (congrArg (· + _) (LoopSessions.prCoin_simAt_pubStmts _ _ _ _ _ _ _ _ _ _ _)))
  · simp only [ite_eq_right hh]
    exact (prCoin_snd
      (C := Claims.Sampling.unitTape.at 0 ((loopRef L₀ src).swP vs).Nonce ((loopRef L₀ src).swP vs).Rand m)
      (fun s : LoopSessions.SimCoins CB m => E ((LoopSessions.pubStmts m stmt).simAt (loopRef L₀ src).Nonce
        (loopRef L₀ src).Tape Claims.ZKMal.CoinOpen.shaH loopK (loopSim L₀ src t) CB Vs
        (a, fun i => _root_.Flock.Hm96.Default512.treeLeaf (l i))
        (θ a fun i => _root_.Flock.Hm96.Default512.treeLeaf (l i)) s))).trans_le le_self_add

/-- **The whole view at commit strings whose digests read the salts before them**, at `u = (((a, p), y), q)`: the tape
`a`; the transcript `T` at the commit strings of the digests `dg a p` under the salts `y` (`FirewallZK.encSalted` at
`commitBytes`), finished; and `anyOuter` at them. -/
noncomputable def anySent {Ωa Ωp Θ : Type} {nh : ℕ} (vs : Claims.ZKServer.Loop.finS.SwVals) (t : ℕ)
    (L₀ : Claims.ZKServer.LeanRelease Claims.Flock.Wit Claims.ZKServer.Loop.Tp)
    (src : Claims.Msg → Claims.Flock.Wit → Claims.ZKServer.Loop.Tp → ℕ →
      _root_.Flock.V (Array _root_.Flock.F128 × ByteArray × Array ByteArray))
    (T : (Fin nh → ByteArray) → Array Flock.Firewall.Sent) (m : ℕ)
    (stmt : ℕ → Ωa × (Fin nh → ByteArray) → Claims.Msg) (wit : ℕ → WCtx Ωa Ωp nh → Claims.Flock.Wit)
    (CB : ℕ → Type) [∀ j, Fintype (CB j)] [∀ j, Nonempty (CB j)] (Vs : Claims.VStar Θ)
    (θ : Ωa → (Fin nh → ByteArray) → Θ) (dg : Ωa → Ωp → (i : Fin nh) → (Fin i → ZkView.Salt) → ByteArray)
    (u : ((Ωa × Ωp) × (Fin nh → ZkView.Salt)) ×
      (Claims.Sampling.unitTape.at 0 ((loopRef L₀ src).swP vs).Nonce ((loopRef L₀ src).swP vs).Rand m ×
        LoopSessions.SimCoins CB m)) :
    Ωa × (Array Flock.Firewall.Sent × Bool) × List (Claims.Msg × Claims.Out) :=
  (u.1.1.1, (T (FirewallZK.encSalted _root_.Flock.Guarantees.commitBytes (dg u.1.1.1 u.1.1.2) u.1.2), true),
    anyOuter vs t L₀ src m stmt wit CB Vs θ u.1.1.1
      (FirewallZK.encSalted _root_.Flock.Guarantees.commitBytes (dg u.1.1.1 u.1.1.2) u.1.2) u.1.1.2 u.1.2 u.2)

/-- **That view is close to the simulated one** (`loopIdeal` at `T`): `anyOuter_le` at the commitments, then T1 once
per commitment, the last first, each digest reading the salts before it (`FirewallZK.recursiveZK_salted`). -/
theorem anySent_le {Ωa Ωp Θ : Type} [Fintype Ωa] [Nonempty Ωa] [Fintype Ωp] [Nonempty Ωp] {nh : ℕ}
    (vs : Claims.ZKServer.Loop.finS.SwVals) (t : ℕ) (ht : 0 < t) (hS : Claims.ZKServer.Loop.finS.Tables)
    (hV : Claims.ZKServer.Loop.finS.Complete
      (Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse))
    (hD : Claims.ZKServer.Loop.finS.DummyReg) (hc : Claims.ZKServer.Loop.finS.Causal)
    (hsc : Claims.ZKServer.Loop.finS.SwCausalOn (Claims.ZKServer.Loop.SwOk Claims.ZKMal.Masked.dataE) vs)
    (hT : Claims.ZKServer.Loop.finS.TapeOK)
    (L₀ : Claims.ZKServer.LeanRelease Claims.Flock.Wit Claims.ZKServer.Loop.Tp)
    (src : Claims.Msg → Claims.Flock.Wit → Claims.ZKServer.Loop.Tp → ℕ →
      _root_.Flock.V (Array _root_.Flock.F128 × ByteArray × Array ByteArray))
    (T : (Fin nh → ByteArray) → Array Flock.Firewall.Sent) (m : ℕ)
    (stmt : ℕ → Ωa × (Fin nh → ByteArray) → Claims.Msg) (wit : ℕ → WCtx Ωa Ωp nh → Claims.Flock.Wit)
    (CB : ℕ → Type) [∀ j, Fintype (CB j)] [∀ j, Nonempty (CB j)] (Vs : Claims.VStar Θ)
    (θ : Ωa → (Fin nh → ByteArray) → Θ) (hsz : (LoopSessions.atWorker m stmt wit).Sized (loopSim L₀ src t) CB)
    (dg : Ωa → Ωp → (i : Fin nh) → (Fin i → ZkView.Salt) → ByteArray)
    (E : Ωa × (Array Flock.Firewall.Sent × Bool) × List (Claims.Msg × Claims.Out) → Prop) :
    prCoin (fun u => E (anySent vs t L₀ src T m stmt wit CB Vs θ dg u)) ≤
      prCoin (fun c => E (_root_.Flock.Guarantees.loopIdeal t L₀ src T m stmt CB Vs θ c)) +
        ((LoopSessions.atWorker m stmt wit).err Claims.ZKMal.CoinOpen.shaH loopK ((loopRef L₀ src).swP vs).prover
            (loopEps L₀ src t) (4 * t * 3 ^ t) Vs (fun x => θ x.1 x.2.1) +
          nh * ENNReal.ofReal (CROnly.hidingGap ZkView.My ZkView.saltHash)) :=
  (FirewallZK.recursiveZK_salted _root_.Flock.Guarantees.commitBytes nh dg
    (fun a l p y q => anyOuter vs t L₀ src m stmt wit CB Vs θ a l p y q)
    (fun a l (s : LoopSessions.SimCoins CB m) => (LoopSessions.pubStmts m stmt).simAt (loopRef L₀ src).Nonce
      (loopRef L₀ src).Tape Claims.ZKMal.CoinOpen.shaH loopK (loopSim L₀ src t) CB Vs
      (a, fun i => _root_.Flock.Hm96.Default512.treeLeaf (l i))
      (θ a fun i => _root_.Flock.Hm96.Default512.treeLeaf (l i)) s)
    (fun a p y E' => anyOuter_le vs t ht hS hV hD hc hsc hT L₀ src m stmt wit CB Vs θ hsz a
      (FirewallZK.encSalted _root_.Flock.Guarantees.commitBytes (dg a p) y) p y E')
    (CROnly.hm96Hiding_gap ZkView.My ZkView.saltHash) fun x => E (x.1, (T x.2.1, true), x.2.2)).trans_eq
    (add_assoc _ _ _)

end FlockSoundness.Discharge.Recursion.OuterZK

namespace FlockSoundness.Discharge.Recursion.LoopAny

open Flock Flock.Firewall FirewallRun FirewallTemplate FirewallAdaptive
open Game

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-- **The real view is `anySent`'s cut at the session's stop**: where the live session sends the schedule's transcript
filled with `strs (dg a p)`, cut at `σ`, its commit strings are `encSalted`'s at the first `fwHoles s` salts
(`strs_take`), so it sends `anySent`'s transcript cut at `σ`; where it finished, it sent it all, the commitments V*'s
statements read are those strings' leaves (`sentCommits_filled`), and their witnesses satisfy their circuits. -/
theorem anyReal_eq {Ωa Ωp Θ : Type} (vs : Claims.ZKServer.Loop.finS.SwVals) (t : ℕ)
    (L₀ : Claims.ZKServer.LeanRelease Claims.Flock.Wit Claims.ZKServer.Loop.Tp)
    (src : Claims.Msg → Claims.Flock.Wit → Claims.ZKServer.Loop.Tp → ℕ →
      _root_.Flock.V (Array _root_.Flock.F128 × ByteArray × Array ByteArray))
    (s : Flock.Firewall.Schedule) (w : Ωa → Ωp → Worker) (N r m : ℕ)
    (stmt : ℕ → Ωa × (Fin (Flock.Guarantees.fwHoles s) → ByteArray) → Claims.Msg)
    (wit : ℕ → OuterZK.WCtx Ωa Ωp (Flock.Guarantees.fwHoles s) → Claims.Flock.Wit) (CB : ℕ → Type)
    [∀ j, Fintype (CB j)] [∀ j, Nonempty (CB j)] (Vs : Claims.VStar Θ)
    (θ : Ωa → (Fin (Flock.Guarantees.fwHoles s) → ByteArray) → Θ)
    (dg : Ωa → Ωp → ℕ → List ByteArray → ByteArray) (σ : Ωa → Ωp → List ByteArray → ℕ × Bool) (a : Ωa) (p : Ωp)
    (b : Fin (Flock.Guarantees.fwHoles s + r) → Fin 192 → Fin 256)
    (q : Claims.Sampling.unitTape.at 0 ((OuterZK.loopRef L₀ src).swP vs).Nonce
      ((OuterZK.loopRef L₀ src).swP vs).Rand m)
    (sc : OuterZK.LoopSessions.SimCoins CB m)
    (hσ : (σ a p (Flock.Guarantees.saltsOf b)).1 ≤ (tmpl s s.tables).length ∧
      ((σ a p (Flock.Guarantees.saltsOf b)).2 = true →
        (σ a p (Flock.Guarantees.saltsOf b)).1 = (tmpl s s.tables).length) ∧
      (adaptOut s (w a p) N { salts := [] } [] (Flock.Guarantees.saltsOf b)).1 =
        (fillAll (tmpl s s.tables) (strs (dg a p) (Flock.Guarantees.saltsOf b))).take
          (σ a p (Flock.Guarantees.saltsOf b)).1 ∧
      adaptFinished s (w a p) N (Flock.Guarantees.saltsOf b) = (σ a p (Flock.Guarantees.saltsOf b)).2)
    (hH : adaptFinished s (w a p) N (Flock.Guarantees.saltsOf b) = true →
      (OuterZK.LoopSessions.atWorker m stmt wit).Honest
        (Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse)
        (fun _ : Unit => Flock.Guarantees.anyCtx s w N r a p b)) :
    Flock.Guarantees.anyReal vs L₀ src s w N r m stmt wit Vs θ (((a, p), b), q) =
      Flock.Guarantees.innerCut (clampStop (K := (tmpl s s.tables).length) (σ a p (Flock.Guarantees.saltsOf b)))
        (OuterZK.anySent vs t L₀ src (Flock.Guarantees.fwFilled s) m stmt wit CB Vs θ
          (fun a p i pre => dg a p i (List.ofFn fun j => ZkView.saltBytes (pre j)))
          (((a, p), fun i => ZkView.saltOfBytes (b (Fin.castAdd r i))), (q, sc))) := by
  obtain ⟨hk, hk', hsent, hfin⟩ := hσ
  have hfill : fillAll (tmpl s s.tables) (strs (dg a p) (Flock.Guarantees.saltsOf b)) =
      fillAll (tmpl s s.tables) (List.ofFn (FirewallZK.encSalted Flock.Guarantees.commitBytes
        (fun i pre => dg a p i (List.ofFn fun j => ZkView.saltBytes (pre j)))
        fun i => ZkView.saltOfBytes (b (Fin.castAdd r i)))) :=
    (fillAll_take_holes (tmpl s s.tables) (strs (dg a p) (Flock.Guarantees.saltsOf b)) _ le_rfl).symm.trans
      (congrArg (fillAll (tmpl s s.tables)) (strs_take (dg a p) b))
  have hsent' := hsent.trans (congrArg (List.take (σ a p (Flock.Guarantees.saltsOf b)).1) hfill)
  have hmin : min (σ a p (Flock.Guarantees.saltsOf b)).1 (tmpl s s.tables).length =
      (σ a p (Flock.Guarantees.saltsOf b)).1 := Nat.min_eq_left hk
  dsimp only [Flock.Guarantees.anyReal, Flock.Guarantees.innerCut, OuterZK.anySent, Flock.Guarantees.fwFilled,
    clampStop]
  rw [hfin, hsent']
  simp only [hmin]
  cases hf : (σ a p (Flock.Guarantees.saltsOf b)).2 with
  | false => rw [ite_eq_right Bool.false_ne_true, ite_eq_right Bool.false_ne_true]
  | true =>
    have hall : (adaptOut s (w a p) N { salts := [] } [] (Flock.Guarantees.saltsOf b)).1 =
        fillAll (tmpl s s.tables) (List.ofFn (FirewallZK.encSalted Flock.Guarantees.commitBytes
          (fun i pre => dg a p i (List.ofFn fun j => ZkView.saltBytes (pre j)))
          fun i => ZkView.saltOfBytes (b (Fin.castAdd r i)))) := by
      rw [hsent', hk' hf, List.take_of_length_le (le_of_eq (length_fillAll _ _))]
    have hctx : Flock.Guarantees.anyCtx s w N r a p b =
        (a, fun i => Hm96.Default512.treeLeaf (FirewallZK.encSalted Flock.Guarantees.commitBytes
          (fun i pre => dg a p i (List.ofFn fun j => ZkView.saltBytes (pre j)))
          (fun i => ZkView.saltOfBytes (b (Fin.castAdd r i))) i), p,
          fun i => ZkView.saltOfBytes (b (Fin.castAdd r i))) := by
      unfold Flock.Guarantees.anyCtx
      rw [hall, sentCommits_filled]
    have hh := hH (hfin.trans hf)
    rw [hctx] at hh
    rw [hctx]
    unfold OuterZK.anyOuter
    rw [ite_eq_left hh]

/-- **The real view's law is `anySent`'s cut at the joint stop**: `anyReal_eq` at every coin, read at the split coins
(`splitSalts`), the stop at them `anyStop`. -/
theorem prCoin_anyReal {Ωa Ωp Θ St Ωk W : Type} [Fintype Ωa] [Fintype Ωp] [Fintype Ωk]
    (vs : Claims.ZKServer.Loop.finS.SwVals) (t : ℕ)
    (L₀ : Claims.ZKServer.LeanRelease Claims.Flock.Wit Claims.ZKServer.Loop.Tp)
    (src : Claims.Msg → Claims.Flock.Wit → Claims.ZKServer.Loop.Tp → ℕ →
      _root_.Flock.V (Array _root_.Flock.F128 × ByteArray × Array ByteArray))
    (s : Flock.Firewall.Schedule) (w : Ωa → Ωp → Worker) (N r m : ℕ)
    (stmt : ℕ → Ωa × (Fin (Flock.Guarantees.fwHoles s) → ByteArray) → Claims.Msg)
    (wit : ℕ → OuterZK.WCtx Ωa Ωp (Flock.Guarantees.fwHoles s) → Claims.Flock.Wit) (CB : ℕ → Type)
    [∀ j, Fintype (CB j)] [∀ j, Nonempty (CB j)] (Vs : Claims.VStar Θ)
    (θ : Ωa → (Fin (Flock.Guarantees.fwHoles s) → ByteArray) → Θ)
    (dg : Ωa → Ωp → ℕ → List ByteArray → ByteArray) (σ : Ωa → Ωp → List ByteArray → ℕ × Bool)
    (hσ : ∀ (a : Ωa) (p : Ωp) (b : Fin (Flock.Guarantees.fwHoles s + r) → Fin 192 → Fin 256),
      (σ a p (Flock.Guarantees.saltsOf b)).1 ≤ (tmpl s s.tables).length ∧
      ((σ a p (Flock.Guarantees.saltsOf b)).2 = true →
        (σ a p (Flock.Guarantees.saltsOf b)).1 = (tmpl s s.tables).length) ∧
      (adaptOut s (w a p) N { salts := [] } [] (Flock.Guarantees.saltsOf b)).1 =
        (fillAll (tmpl s s.tables) (strs (dg a p) (Flock.Guarantees.saltsOf b))).take
          (σ a p (Flock.Guarantees.saltsOf b)).1 ∧
      adaptFinished s (w a p) N (Flock.Guarantees.saltsOf b) = (σ a p (Flock.Guarantees.saltsOf b)).2)
    (hH : (OuterZK.LoopSessions.atWorker m stmt wit).Honest
      (Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse)
      (fun u : {u : (Ωa × Ωp) × (Fin (Flock.Guarantees.fwHoles s + r) → Fin 192 → Fin 256) //
          adaptFinished s (w u.1.1 u.1.2) N (Flock.Guarantees.saltsOf u.2) = true} =>
        Flock.Guarantees.anyCtx s w N r u.1.1.1 u.1.1.2 u.1.2))
    (cut : St → Ωa × (Array Flock.Firewall.Sent × Bool) × List (Claims.Msg × Claims.Out) → W)
    (stop : (((Ωa × Ωp) × (Fin (Flock.Guarantees.fwHoles s + r) → Fin 192 → Fin 256)) ×
        Claims.Sampling.unitTape.at 0 ((OuterZK.loopRef L₀ src).swP vs).Nonce ((OuterZK.loopRef L₀ src).swP vs).Rand
          m) × Ωk → St)
    (E : W → Prop) :
    prCoin (fun u => E (cut (stop u) (Flock.Guarantees.anyReal vs L₀ src s w N r m stmt wit Vs θ u.1))) =
      prCoin (fun y => E (cut
        (anyStop (K := (tmpl s s.tables).length) (S := OuterZK.LoopSessions.SimCoins CB m) stop σ y).1
        (Flock.Guarantees.innerCut
          (anyStop (K := (tmpl s s.tables).length) (S := OuterZK.LoopSessions.SimCoins CB m) stop σ y).2
          (OuterZK.anySent vs t L₀ src (Flock.Guarantees.fwFilled s) m stmt wit CB Vs θ
            (fun a p i pre => dg a p i (List.ofFn fun j => ZkView.saltBytes (pre j))) y.1)))) := by
  rw [← ZK.prCoin_fst (E := OuterZK.LoopSessions.SimCoins CB m)
      (fun u => E (cut (stop u) (Flock.Guarantees.anyReal vs L₀ src s w N r m stmt wit Vs θ u.1))),
    ← ZK.prCoin_congr_equiv (splitSalts (Ωa × Ωp) (Claims.Sampling.unitTape.at 0
      ((OuterZK.loopRef L₀ src).swP vs).Nonce ((OuterZK.loopRef L₀ src).swP vs).Rand m) Ωk
      (OuterZK.LoopSessions.SimCoins CB m) (Flock.Guarantees.fwHoles s) r)]
  refine congrArg prCoin (funext fun x => ?_)
  rw [anyStop_split, splitSalts_fst]
  exact congrArg (fun v => E (cut (stop x.1) v)) (anyReal_eq vs t L₀ src s w N r m stmt wit CB Vs θ dg σ
    x.1.1.1.1.1 x.1.1.1.1.2 x.1.1.1.2 x.1.1.2 x.2 (hσ x.1.1.1.1.1 x.1.1.1.1.2 x.1.1.1.2)
    fun hf _ => hH ⟨x.1.1.1, hf⟩)

/-- **The same at one stop of the inner session or the audit**, where the release reads no stop at an unfinished
session's view: `prCoin_anyReal`, the joined stop read as `sumCut`'s (`sumCut_toSum`). -/
theorem prCoin_anyReal_sum {Ωa Ωp Θ St Ωk W : Type} [Fintype Ωa] [Fintype Ωp] [Fintype Ωk]
    (vs : Claims.ZKServer.Loop.finS.SwVals) (t : ℕ)
    (L₀ : Claims.ZKServer.LeanRelease Claims.Flock.Wit Claims.ZKServer.Loop.Tp)
    (src : Claims.Msg → Claims.Flock.Wit → Claims.ZKServer.Loop.Tp → ℕ →
      _root_.Flock.V (Array _root_.Flock.F128 × ByteArray × Array ByteArray))
    (s : Flock.Firewall.Schedule) (w : Ωa → Ωp → Worker) (N r m : ℕ)
    (stmt : ℕ → Ωa × (Fin (Flock.Guarantees.fwHoles s) → ByteArray) → Claims.Msg)
    (wit : ℕ → OuterZK.WCtx Ωa Ωp (Flock.Guarantees.fwHoles s) → Claims.Flock.Wit) (CB : ℕ → Type)
    [∀ j, Fintype (CB j)] [∀ j, Nonempty (CB j)] (Vs : Claims.VStar Θ)
    (θ : Ωa → (Fin (Flock.Guarantees.fwHoles s) → ByteArray) → Θ)
    (dg : Ωa → Ωp → ℕ → List ByteArray → ByteArray) (σ : Ωa → Ωp → List ByteArray → ℕ × Bool)
    (hσ : ∀ (a : Ωa) (p : Ωp) (b : Fin (Flock.Guarantees.fwHoles s + r) → Fin 192 → Fin 256),
      (σ a p (Flock.Guarantees.saltsOf b)).1 ≤ (tmpl s s.tables).length ∧
      ((σ a p (Flock.Guarantees.saltsOf b)).2 = true →
        (σ a p (Flock.Guarantees.saltsOf b)).1 = (tmpl s s.tables).length) ∧
      (adaptOut s (w a p) N { salts := [] } [] (Flock.Guarantees.saltsOf b)).1 =
        (fillAll (tmpl s s.tables) (strs (dg a p) (Flock.Guarantees.saltsOf b))).take
          (σ a p (Flock.Guarantees.saltsOf b)).1 ∧
      adaptFinished s (w a p) N (Flock.Guarantees.saltsOf b) = (σ a p (Flock.Guarantees.saltsOf b)).2)
    (hH : (OuterZK.LoopSessions.atWorker m stmt wit).Honest
      (Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse)
      (fun u : {u : (Ωa × Ωp) × (Fin (Flock.Guarantees.fwHoles s + r) → Fin 192 → Fin 256) //
          adaptFinished s (w u.1.1 u.1.2) N (Flock.Guarantees.saltsOf u.2) = true} =>
        Flock.Guarantees.anyCtx s w N r u.1.1.1 u.1.1.2 u.1.2))
    (cut : St → Ωa × (Array Flock.Firewall.Sent × Bool) × List (Claims.Msg × Claims.Out) → W) (s₀ : St)
    (hcut : Flock.Guarantees.ReleaseReadsNoStop cut s₀)
    (stop : (((Ωa × Ωp) × (Fin (Flock.Guarantees.fwHoles s + r) → Fin 192 → Fin 256)) ×
        Claims.Sampling.unitTape.at 0 ((OuterZK.loopRef L₀ src).swP vs).Nonce ((OuterZK.loopRef L₀ src).swP vs).Rand
          m) × Ωk → St)
    (E : W → Prop) :
    prCoin (fun u => E (cut (stop u) (Flock.Guarantees.anyReal vs L₀ src s w N r m stmt wit Vs θ u.1))) =
      prCoin (fun y => E (Flock.Guarantees.sumCut s₀ cut
        (toSum (anyStop (K := (tmpl s s.tables).length) (S := OuterZK.LoopSessions.SimCoins CB m) stop σ y))
        (OuterZK.anySent vs t L₀ src (Flock.Guarantees.fwFilled s) m stmt wit CB Vs θ
          (fun a p i pre => dg a p i (List.ofFn fun j => ZkView.saltBytes (pre j))) y.1))) := by
  refine (prCoin_anyReal vs t L₀ src s w N r m stmt wit CB Vs θ dg σ hσ hH cut stop E).trans
    (congrArg prCoin (funext fun y => congrArg E ?_))
  exact (sumCut_toSum s₀ cut hcut _ (anyStop_last stop σ (fun a p b => (hσ a p b).2.1) y) _).symm

end FlockSoundness.Discharge.Recursion.LoopAny

namespace Flock.SecurityProofs

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game
open scoped ENNReal

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-- **The recursive audit at the session loop is zero knowledge at any worker, released at a stop.** -/
theorem RecursiveZKLoopAnySum : Flock.Guarantees.RecursiveZKLoopAnySum := by
  intro Ωa Ωp Θ St Ωk W _ _ _ _ _ _ _ _ vs t L₀ src s w N r m stmt wit CB _ _ Vs θ ht hS hV hD hc hsc hT hH hsz
    cut s₀ hcut stop E
  choose dg σ _ hσ using fun (a : Ωa) (p : Ωp) => adaptOut_any s (w a p) N (Flock.Guarantees.fwHoles s + r)
  exact (LoopAny.prCoin_anyReal_sum vs t L₀ src s w N r m stmt wit CB Vs θ dg σ
      (fun a p b => hσ a p (Flock.Guarantees.saltsOf b) List.length_ofFn) hH cut s₀ hcut stop E).trans_le
    ((prCoin_cut_le (OuterZK.anySent vs t L₀ src (Flock.Guarantees.fwFilled s) m stmt wit CB Vs θ
        (fun a p i pre => dg a p i (List.ofFn fun j => ZkView.saltBytes (pre j))))
      (Flock.Guarantees.loopIdeal t L₀ src (Flock.Guarantees.fwFilled s) m stmt CB Vs θ)
      (Flock.Guarantees.sumCut (K := (FirewallTemplate.tmpl s s.tables).length) s₀ cut)
      (fun y => LoopAny.toSum (LoopAny.anyStop (K := (FirewallTemplate.tmpl s s.tables).length)
        (S := OuterZK.LoopSessions.SimCoins CB m) stop σ y))
      (fun E' => OuterZK.anySent_le vs t ht hS hV hD hc hsc hT L₀ src (Flock.Guarantees.fwFilled s) m stmt wit CB
        Vs θ hsz (fun a p i pre => dg a p i (List.ofFn fun j => ZkView.saltBytes (pre j))) E') E).trans_eq
      (add_assoc _ _ _).symm)

end Flock.SecurityProofs
