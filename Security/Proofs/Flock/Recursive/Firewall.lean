import Proofs.Flock.Recursive.Sessions
import Proofs.Flock.Recursive.Premises
import Proofs.Flock.Soundness.Discharge.ZkView.OsSalts
import Flock.Firewall

/-!
# (ii) at the firewall and V*'s code

`RecursiveFirewallPrivate` is (ii) of `RecursiveAudit`, zero knowledge with circuit privacy, stated about the code that
sends the auditor its view: the firewall (`Flock.Firewall.run`), which salts every message and tree top the worker
sends it with 192 bytes of its own and sends the auditor the salted commit strings, and V*'s outer sessions run by the
honest prover's code (`OuterZK.proverExe`). It takes no zero knowledge of V*'s sessions and no acceptance: its one
premise is `hash-derived-key`, and what it takes of the code is named, each an obligation:

* `FirewallSends`: the honest worker's requests run through the firewall to the pinned schedule's transcript, its
  holes the salted commit strings of the messages in salt order, and finish.
* `SessionsPublic`, `SessionsCoined`, `SessionsHonest`: V*'s staging. The simulator's public context is the honest
  one's, each session's table is `verify --zk`'s at its data and a draw it sets up, and the honest witnesses are valid
  with public registered values; V*'s sizes are `OuterZK.Sessions`' `hz`, `hR`, `hS`.

`fwRealPr` is the auditor's real view, `fwSimPr` the simulated one; `commitBytes` is a commit string from its two
halves.
Its proof is `Flock.SecurityProofs.RecursiveFirewallPrivate` (`Proofs.Flock.Recursive.FirewallZK`).
-/

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game
open scoped ENNReal

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-- **An hm96-sha512 commit string from its two halves**, `b ‖ c`: the 128 bytes a firewall leaf hashes
(`ZkView.leafH` is `Default512.treeLeaf` of them). -/
def commitBytes (u : ZkView.Bits × ByteArray) : ByteArray := ZkView.bytesOf (k := 64) u.1 ++ u.2

/-- Obligation (the honest worker's requests follow the pinned schedule `s`): at every private circuit `P`, witness `w`,
registration salts `r`, tape `a` and randomness `p`, the requests the honest worker sends the firewall,
`reqs P w r a p`, run through it (`Flock.Firewall.run s`) at any salts `ys` to `s`'s transcript `T`, its holes the
salted commit strings of the messages `msgs P w r a p` (each message's SHA-512 under its salt,
`Default512.commitString`, in salt order), and finish. `T` reads only the schedule: the firewall pins every count,
name and position it sends to `s`, and Commit's table order, which it doesn't pin, is the honest worker's. -/
def FirewallSends {Prog Wit Rr Ωa Ωp : Type} {nh : ℕ} (s : Flock.Firewall.Schedule)
    (reqs : Prog → Wit → Rr → Ωa → Ωp → List ByteArray) (msgs : Prog → Wit → Rr → Ωa → Ωp → Fin nh → ByteArray)
    (T : (Fin nh → ByteArray) → Array Flock.Firewall.Sent) : Prop :=
  ∀ P w r a p (ys : Fin nh → ByteArray), Flock.Firewall.run s (reqs P w r a p) (List.ofFn ys) =
    (T fun i => Hm96.Default512.commitString (Sha512.hash (msgs P w r a p i)) (ys i), true)

/-- Obligation (V*'s staging, the public context): at every honest context (`OuterZK.hctxAt`, the firewall's
commitments to `msgs` under the salts `y`), the public context V*'s sessions read is the one their simulator reads,
`pubS` at the owed outputs, the registration's view, the auditor's tape and the firewall's commitments. -/
def SessionsPublic {Out Prog Wit Rv Ωa Ωp Pub : Type} {nr nh : ℕ} (rows : Prog → Fin nr → ByteArray)
    (pub : (Fin nr → ByteArray) → Rv) (honest : Out → Prog → Wit → Prop)
    (msgs : Prog → Wit → (Fin nr → ZkView.Salt) → Ωa → Ωp → Fin nh → ByteArray)
    (V : OuterZK.Sessions (OuterZK.HCtx Prog Wit Ωa Ωp nr nh) Pub)
    (pubS : Out → Rv → Ωa → (Fin nh → ByteArray) → Pub) : Prop :=
  ∀ o P w, honest o P w → ∀ r a p y, V.pubOf (OuterZK.hctxAt msgs P w r a p y) =
    pubS o (regLeaves rows pub P r) a (fun i => firewallLeaf (msgs P w r a p i) (y i))

/-- Obligation (V*'s staging, the tables): each of V*'s session tables is `coinedOf` at `verify --zk`'s table of the
session's data `D` at a draw `verify --zk` sets up (`DrawSetupZK`), so it passes `CoinedOK`
(`OuterZK.tablesOK_of_sessions`). -/
def SessionsCoined {Ctx Pub : Type} (V : OuterZK.Sessions Ctx Pub) : Prop :=
  ∃ (P : Composed.LigProver) (Vs : Fin V.S → VStmt) (Reg' : Type)
    (D : Pub → (Fin V.n₂ → ZkView.Bits × ByteArray) → (j : Fin V.S) → (Vs j).OuterD Reg')
    (draw : (j : Fin V.S) → Pub → (Fin V.n₂ → ZkView.Bits × ByteArray) → Finset (Fin (Vs j).nv))
    (y : ∀ j pc l, ZkLink.DrawSetupZK (D pc l j).I (Vs j).mPts (draw j pc l) ((D pc l j).dj (draw j pc l))),
    ∀ j pc l, V.C j pc l = Composed.coinedOf P (ZkLink.tableAtZK (D pc l j).dj (D pc l j).y₀ (draw j pc l))
      (y j pc l).x.st

/-- Obligation (V*'s staging, the honest witnesses): at every honest context, each session's witness is valid and its
registered values are the public ones (`OuterZK.Sessions.Honest`). -/
def SessionsHonest {Out Prog Wit Ωa Ωp Pub : Type} {nr nh : ℕ} (honest : Out → Prog → Wit → Prop)
    (msgs : Prog → Wit → (Fin nr → ZkView.Salt) → Ωa → Ωp → Fin nh → ByteArray)
    (V : OuterZK.Sessions (OuterZK.HCtx Prog Wit Ωa Ωp nr nh) Pub) : Prop :=
  ∀ o P w, honest o P w → ∀ r a p y, V.Honest (OuterZK.hctxAt msgs P w r a p y)

/-- **The recursive audit's real view at the firewall**, the probability of the event `E`. The developer registers the
private circuit `P` under the registration's salts `r`, and the auditor sees `regLeaves rows pub P r`. The honest
worker sends the firewall its requests `reqs P w r a p` at the auditor's tape `a` and its randomness `p`; the firewall
(`Flock.Firewall.run s`) salts them with `nh` salts of 192 uniform bytes `b` and sends the auditor what it sends and
whether the session finished. V*'s sessions, run by the honest prover's code `prover` at the firewall's commitments and
salts (`OuterZK.hctxAt`), give the auditor their view at their randomness `q`. The average is over `r`, `a`, `p`, `b`
and `q`, uniform and independent. -/
noncomputable def fwRealPr {Prog Wit Rv Ωa Ωp Pub : Type} [Fintype Ωa] [Fintype Ωp] {nr nh : ℕ}
    (rows : Prog → Fin nr → ByteArray) (pub : (Fin nr → ByteArray) → Rv) (s : Flock.Firewall.Schedule)
    (reqs : Prog → Wit → (Fin nr → ZkView.Salt) → Ωa → Ωp → List ByteArray)
    (msgs : Prog → Wit → (Fin nr → ZkView.Salt) → Ωa → Ωp → Fin nh → ByteArray)
    (V : OuterZK.Sessions (OuterZK.HCtx Prog Wit Ωa Ωp nr nh) Pub) (prover : OuterZK.TableProver) (P : Prog) (w : Wit)
    (E : Rv × Ωa × (Array Flock.Firewall.Sent × Bool) × V.View → Prop) : ℝ≥0∞ :=
  avg fun r : Fin nr → ZkView.Salt => prCoin fun z : ((Ωa × Ωp) × (Fin nh → Fin 192 → Fin 256)) × V.Ωo =>
    E (regLeaves rows pub P r, z.1.1.1,
      Flock.Firewall.run s (reqs P w r z.1.1.1 z.1.1.2)
        (List.ofFn fun i => FlockSoundness.Audit.Law.streamOf (z.1.2 i)),
      V.view prover (OuterZK.hctxAt msgs P w r z.1.1.1 z.1.1.2 fun i => ZkView.saltOfBytes (z.1.2 i)) z.2)

/-- **The simulated view at the firewall**, the probability of the event `E`: the simulator registers the fixed private
circuit `p₀` of the class (`regLeaves rows pub p₀ r` at a fresh `r`), fills the schedule's transcript `T` with ideal
commit strings (`commitBytes` of a uniform `U` beside a salt's own hash, which read no message), says the session
finished, and runs V*'s simulator at the public context `pubS` reads: the owed outputs `o`, the registration's view, the
auditor's tape and the ideal commitments' leaves. It reads no private circuit but `p₀`, no witness and no request. -/
noncomputable def fwSimPr {Out Prog Rv Ωa Ctx Pub : Type} [Fintype Ωa] {nr nh : ℕ}
    (rows : Prog → Fin nr → ByteArray) (pub : (Fin nr → ByteArray) → Rv)
    (T : (Fin nh → ByteArray) → Array Flock.Firewall.Sent) (V : OuterZK.Sessions Ctx Pub)
    (pubS : Out → Rv → Ωa → (Fin nh → ByteArray) → Pub) (o : Out) (p₀ : Prog)
    (E : Rv × Ωa × (Array Flock.Firewall.Sent × Bool) × V.View → Prop) : ℝ≥0∞ :=
  avg fun r : Fin nr → ZkView.Salt => prCoin fun z : (Ωa × (Fin nh → ZkView.Bits × ZkView.Salt)) × V.Ωs =>
    E (regLeaves rows pub p₀ r, z.1.1, (T fun i => commitBytes (OuterZK.idealStrs z.1.2 i), true),
      V.sim (pubS o (regLeaves rows pub p₀ r) z.1.1 fun i => idealLeaf (z.1.2 i)) z.2)

/-- **(ii) at the firewall and V*'s code: the recursive audit is zero knowledge with circuit privacy.** Everything is at
one size class of the universal unit, public: `Prog` is the class's private circuits, registered by hm96 leaves of their
`nr` rows (`regLeaves rows pub`), and `honest o P w` says that `w` is an honest witness on which `P` computes the owed
outputs `o`. The auditor's whole view (`fwRealPr`) is the registration's, what the firewall sends (`Flock.Firewall.run`
at the pinned schedule `s`, on the honest worker's requests `reqs` and `nh` salts of 192 uniform bytes) and V*'s outer
sessions `V`, run by the honest prover's code (`OuterZK.proverExe`, whatever a stopped run sends). The one premise is
`hash-derived-key`; what it takes of the code, each an obligation:

* the honest worker's requests run through the firewall to the schedule's transcript `T` with the salted commit strings
  of the messages `msgs` in it, and finish (`FirewallSends`);
* V*'s staging: the simulator's public context `pubS` is the honest one's (`SessionsPublic`), each session's table is
  `verify --zk`'s at its data and a draw it sets up (`SessionsCoined`), the honest witnesses are valid with public
  registered values (`SessionsHonest`), and V*'s sizes are fixed (`OuterZK.Sessions`' `hz`, `hR`, `hS`).

Then for every honest witness `w` of every private circuit `P` of the class, and every private circuit `p₀` of it, the
real view and the simulated one (`fwSimPr`: `p₀` registered, `T` on ideal commit strings, V*'s simulator), which reads
only the class, the schedule's transcript and the owed outputs, give every event probabilities within
$$2 n_r \cdot 2^{-193} + \varepsilon_V + n_h \cdot 2^{-193}$$
both ways, `ε_V` V*'s sessions' error (`OuterZK.Sessions.err`). -/
def RecursiveFirewallPrivate : Prop :=
  ∀ {Out Prog Wit Rv Ωa Ωp Pub : Type} [Fintype Ωa] [Inhabited Ωa] [Fintype Ωp] [Inhabited Ωp],
    Flock.Premises.HashDerivedKeyHm96 →
    ∀ (s : Flock.Firewall.Schedule)
      -- the registration: the private circuit's rows, and what the auditor sees of their leaves
      (nr : ℕ) (rows : Prog → Fin nr → ByteArray) (pub : (Fin nr → ByteArray) → Rv)
      (honest : Out → Prog → Wit → Prop)
      -- the honest worker's requests to the firewall, the messages they commit, and the schedule's transcript
      (nh : ℕ) (reqs : Prog → Wit → (Fin nr → ZkView.Salt) → Ωa → Ωp → List ByteArray)
      (msgs : Prog → Wit → (Fin nr → ZkView.Salt) → Ωa → Ωp → Fin nh → ByteArray)
      (T : (Fin nh → ByteArray) → Array Flock.Firewall.Sent)
      -- V*'s outer sessions, the public context their simulator reads, and what a stopped run sends
      (V : OuterZK.Sessions (OuterZK.HCtx Prog Wit Ωa Ωp nr nh) Pub)
      (pubS : Out → Rv → Ωa → (Fin nh → ByteArray) → Pub) (stopped : OuterZK.Stopped)
      (p₀ : Prog) (o : Out) (P : Prog) (w : Wit),
      FirewallSends s reqs msgs T → SessionsPublic rows pub honest msgs V pubS → SessionsCoined V →
      SessionsHonest honest msgs V → honest o P w →
      ∀ E : Rv × Ωa × (Array Flock.Firewall.Sent × Bool) × V.View → Prop,
        fwRealPr rows pub s reqs msgs V (OuterZK.proverExe stopped) P w E ≤ fwSimPr rows pub T V pubS o p₀ E +
            (2 * nr * (2 : ℝ≥0∞) ^ (-193 : ℤ) + V.err + nh * (2 : ℝ≥0∞) ^ (-193 : ℤ)) ∧
          fwSimPr rows pub T V pubS o p₀ E ≤ fwRealPr rows pub s reqs msgs V (OuterZK.proverExe stopped) P w E +
            (2 * nr * (2 : ℝ≥0∞) ^ (-193 : ℤ) + V.err + nh * (2 : ℝ≥0∞) ^ (-193 : ℤ))

end Flock.Guarantees
