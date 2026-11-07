module

public import Proofs.Flock.Recursive.FirewallLive
public import Proofs.Flock.Recursive.FirewallHonest
public import Proofs.Flock.Recursive.Firewall
import Flock.Firewall.Contract
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
import Proofs.Flock.Recursive.FirewallZK

/-!
# `FirewallSends` from the ZK server's code

`RecursiveFirewallPrivate` takes `FirewallSends` of the ZK server and the honest worker: on the worker's requests the
ZK server (`Flock.Firewall.run`) sends the schedule's transcript `T` with the commit strings `firewallStrings`, and
finishes. The ZK server's half is its code's, proved here at `T = firewallT s tables`: the pinned schedule's transcript
(`FirewallTemplate.tmpl`, every count, name, stream, parent and position the schedule's) with Commit's tables in the
order `tables`, filled with the commit strings. What is left is the worker's half, `FirewallAccepts`: on its requests at
any commit strings the ZK server finishes at zero salts, Commit names the tables `tables`, and the commit strings are
those of the messages `msgs`, each reading the strings before it.

* `firewallSends_of_accepts`: `FirewallAccepts s reqs msgs tables → FirewallSends s reqs msgs (firewallT s tables)`, by
  `FirewallTemplate.sends_of_one` at the zero salts and `firewallStrings`' unfolding.
* `tables_eq_of_accepts`: there `tables` is the schedule's order `s.tables`, the only one the ZK server takes.
* `recursiveFirewallPrivate_of_accepts`: `RecursiveFirewallPrivate`'s two ways with `FirewallAccepts` in place of
  `FirewallSends`, its simulated view at `firewallT s tables`, which reads only the schedule and the table order.
* `liveSends`, `liveSends_of_accepts`: the deployed send check, `flock-firewall live` (`FirewallLive.liveRun`), sends
  what `FirewallSends` says `run` sends: on the honest worker's requests, a live session that doesn't stop, whose salts
  read begin with `ys`, sends the transcript at the commit strings `firewallStrings ys` and finishes
  (`FirewallLive.run_live`, `FirewallLive.run_append`).
* `honestReqs_accepts`: the worker's half, `FirewallAccepts` of the honest worker's requests (`honestReqs`) and its rows'
  messages (`FirewallHonest.honestMsgs`), at a schedule the ZK server holds a session to and a worker that follows it,
  its rows reading only the commit strings before them (`FirewallHonest.honest_accepts` at the zero salt).
-/

@[expose] public section

namespace Flock.Guarantees

open FlockSoundness.Discharge.Recursion

/-- A salt of 192 zero bytes. -/
def zeroSalt : ByteArray := ⟨Array.replicate 192 0⟩

/-- **The schedule's transcript filled with the commit strings `l`**, in salt order, Commit naming the tables `tables`
in order (`FirewallTemplate.tmpl`). -/
def firewallT (s : Flock.Firewall.Schedule) (tables : List String) {nh : ℕ} (l : Fin nh → ByteArray) :
    Array Flock.Firewall.Sent :=
  (FirewallRun.fillAll (FirewallTemplate.tmpl s tables) (List.ofFn l)).toArray

/-- Obligation (the honest worker's requests pass the ZK server, committing the messages `msgs`): at every private circuit
`P`, witness `w`, registration salts `r`, tape `a`, randomness `p` and commit strings `l`, the ZK server
(`Flock.Firewall.run s`) at `nh` zero salts finishes on the requests `reqs P w r a p l`, their Commit names the tables
`tables` in order, and the commit strings it sends are those of the messages `msgs P w r a p` in salt order, message `i`
read at the strings of `l` before it. It says nothing of how the ZK server salts, which its code fixes
(`FirewallTemplate.sends_of_one`), nor of what it sends but the table order, which the schedule fixes
(`FirewallTemplate.run_blank`): it is that the worker's requests pass, and that message `i` reads only the strings
before it. -/
def FirewallAccepts {Prog Wit Rr Ωa Ωp : Type} {nh : ℕ} (s : Flock.Firewall.Schedule)
    (reqs : Prog → Wit → Rr → Ωa → Ωp → (Fin nh → ByteArray) → List ByteArray)
    (msgs : Prog → Wit → Rr → Ωa → Ωp → (i : Fin nh) → (Fin i → ByteArray) → ByteArray) (tables : List String) :
    Prop :=
  ∀ P w r a p (l : Fin nh → ByteArray),
    (Flock.Firewall.run s (reqs P w r a p l) (List.ofFn fun _ : Fin nh => zeroSalt)).2 = true ∧
      FirewallTemplate.namesOf (Flock.Firewall.run s (reqs P w r a p l) (List.ofFn fun _ : Fin nh => zeroSalt)).1.toList =
        tables ∧
      FirewallRun.holesOf (Flock.Firewall.run s (reqs P w r a p l) (List.ofFn fun _ : Fin nh => zeroSalt)).1 =
        List.ofFn fun i =>
          Hm96.Default512.commitString (Sha512.hash (msgs P w r a p i fun j => l (Fin.castLE i.isLt.le j))) zeroSalt

end Flock.Guarantees

namespace Flock.SecurityProofs

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game
open scoped ENNReal

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

theorem firewallStrings_eq {nh : ℕ} (msgs : (i : Fin nh) → (Fin i → ByteArray) → ByteArray) (ys : Fin nh → ByteArray)
    (i : Fin nh) :
    Flock.Guarantees.firewallStrings msgs ys i = Hm96.Default512.commitString
      (Sha512.hash (msgs i fun j => Flock.Guarantees.firewallStrings msgs ys (Fin.castLE i.isLt.le j))) (ys i) := by
  rw [Flock.Guarantees.firewallStrings]

/-- **The ZK server's half of `FirewallSends`, from its code**: under `FirewallAccepts`, the ZK server sends the schedule's
transcript `firewallT s tables` at the commit strings `firewallStrings`, at every salt list, and finishes. -/
theorem firewallSends_of_accepts {Prog Wit Rr Ωa Ωp : Type} {nh : ℕ} (s : Flock.Firewall.Schedule)
    (reqs : Prog → Wit → Rr → Ωa → Ωp → (Fin nh → ByteArray) → List ByteArray)
    (msgs : Prog → Wit → Rr → Ωa → Ωp → (i : Fin nh) → (Fin i → ByteArray) → ByteArray) (tables : List String)
    (h : Flock.Guarantees.FirewallAccepts s reqs msgs tables) :
    Flock.Guarantees.FirewallSends s reqs msgs (Flock.Guarantees.firewallT s tables) := by
  intro P w r a p ys
  obtain ⟨hfin, ho, hh⟩ := h P w r a p (Flock.Guarantees.firewallStrings (msgs P w r a p) ys)
  refine (FirewallTemplate.sends_of_one s _ tables
    (fun i => Sha512.hash (msgs P w r a p i fun j =>
      Flock.Guarantees.firewallStrings (msgs P w r a p) ys (Fin.castLE i.isLt.le j)))
    (fun _ => Flock.Guarantees.zeroSalt) (fun _ => Flock.Firewall.Contract.sha512_size _) hfin ho hh ys).trans ?_
  refine congrArg (fun l => ((FirewallRun.fillAll (FirewallTemplate.tmpl s tables) l).toArray, true)) ?_
  exact congrArg List.ofFn (funext fun i => (firewallStrings_eq (msgs P w r a p) ys i).symm)

/-- **The table order is the schedule's**: under `FirewallAccepts s reqs msgs tables`, `tables` is `s.tables`, the only
order the ZK server takes Commit's tables in (`FirewallTemplate.namesOf_run`), so `firewallT s tables` reads the schedule
alone. -/
theorem tables_eq_of_accepts {Prog Wit Rr Ωa Ωp : Type} {nh : ℕ} (s : Flock.Firewall.Schedule)
    (reqs : Prog → Wit → Rr → Ωa → Ωp → (Fin nh → ByteArray) → List ByteArray)
    (msgs : Prog → Wit → Rr → Ωa → Ωp → (i : Fin nh) → (Fin i → ByteArray) → ByteArray) (tables : List String)
    (h : Flock.Guarantees.FirewallAccepts s reqs msgs tables) (P : Prog) (w : Wit) (r : Rr) (a : Ωa) (p : Ωp)
    (l : Fin nh → ByteArray) : tables = s.tables := by
  obtain ⟨hfin, ho, -⟩ := h P w r a p l
  exact ho.symm.trans (FirewallTemplate.namesOf_run s _ _ hfin)

/-- **(ii) at the ZK server and V*'s code, from the honest worker's half**: `RecursiveFirewallPrivate`'s two ways with
`FirewallAccepts` in place of `FirewallSends`, the simulated view filling the schedule's transcript `firewallT s tables`
with ideal commit strings. -/
theorem recursiveFirewallPrivate_of_accepts {Out Prog Wit Rv Ωa Ωp Pub : Type} [Fintype Ωa] [Inhabited Ωa]
    [Fintype Ωp] [Inhabited Ωp] (s : Flock.Firewall.Schedule) (nr : ℕ) (rows : Prog → Fin nr → ByteArray)
    (pub : (Fin nr → ByteArray) → Rv) (honest : Out → Prog → Wit → Prop) (nh : ℕ)
    (reqs : Prog → Wit → (Fin nr → ZkView.Salt) → Ωa → Ωp → (Fin nh → ByteArray) → List ByteArray)
    (msgs : Prog → Wit → (Fin nr → ZkView.Salt) → Ωa → Ωp → (i : Fin nh) → (Fin i → ByteArray) → ByteArray)
    (tables : List String) (V : OuterZK.Sessions (OuterZK.HCtx Prog Wit Ωa Ωp nr nh) Pub)
    (pubS : Out → Rv → Ωa → (Fin nh → ByteArray) → Pub) (stopped : OuterZK.Stopped) (p₀ : Prog) (o : Out) (P : Prog)
    (w : Wit) (hA : Flock.Guarantees.FirewallAccepts s reqs msgs tables)
    (hpub : Flock.Guarantees.SessionsPublic rows pub honest msgs V pubS) (hC : Flock.Guarantees.SessionsCoined V)
    (hH : Flock.Guarantees.SessionsHonest honest msgs V) (hw : honest o P w)
    (E : Rv × Ωa × (Array Flock.Firewall.Sent × Bool) × V.View → Prop) :
    Flock.Guarantees.fwRealPr rows pub s reqs msgs V (OuterZK.proverExe stopped) P w E ≤
        Flock.Guarantees.fwSimPr rows pub (Flock.Guarantees.firewallT s tables) V pubS o p₀ E +
          (2 * nr * ENNReal.ofReal (CROnly.hidingGap ZkView.My ZkView.saltHash) +
            V.errAt (ENNReal.ofReal (CROnly.hidingGap ZkView.My ZkView.saltHash)) +
            nh * ENNReal.ofReal (CROnly.hidingGap ZkView.My ZkView.saltHash)) ∧
      Flock.Guarantees.fwSimPr rows pub (Flock.Guarantees.firewallT s tables) V pubS o p₀ E ≤
        Flock.Guarantees.fwRealPr rows pub s reqs msgs V (OuterZK.proverExe stopped) P w E +
          (2 * nr * ENNReal.ofReal (CROnly.hidingGap ZkView.My ZkView.saltHash) +
            V.errAt (ENNReal.ofReal (CROnly.hidingGap ZkView.My ZkView.saltHash)) +
            nh * ENNReal.ofReal (CROnly.hidingGap ZkView.My ZkView.saltHash)) :=
  RecursiveFirewallPrivate s nr rows pub honest nh reqs msgs (Flock.Guarantees.firewallT s tables) V pubS stopped p₀ o
    P w (firewallSends_of_accepts s reqs msgs tables hA) hpub hC hH hw E

/-- **The live send check sends what `FirewallSends` says**: under `FirewallSends`, a session of `flock-firewall live`
(`FirewallLive.liveRun`, each request with the fresh salts read for it) on the honest worker's requests at the commit
strings `firewallStrings ys` that doesn't stop, whose salts read begin with `ys`, sends the transcript `T` at those
commit strings and finishes. -/
theorem liveSends {Prog Wit Rr Ωa Ωp : Type} {nh : ℕ} (s : Flock.Firewall.Schedule)
    (reqs : Prog → Wit → Rr → Ωa → Ωp → (Fin nh → ByteArray) → List ByteArray)
    (msgs : Prog → Wit → Rr → Ωa → Ωp → (i : Fin nh) → (Fin i → ByteArray) → ByteArray)
    (T : (Fin nh → ByteArray) → Array Flock.Firewall.Sent) (hF : Flock.Guarantees.FirewallSends s reqs msgs T)
    (P : Prog) (w : Wit) (r : Rr) (a : Ωa) (p : Ωp) (ys : Fin nh → ByteArray) (zs : List ByteArray)
    (L : List (ByteArray × List ByteArray)) (sent : List Flock.Firewall.Sent) (t : Flock.Firewall.State)
    (hreqs : L.map (·.1) = reqs P w r a p (Flock.Guarantees.firewallStrings (msgs P w r a p) ys))
    (hsalts : L.flatMap (·.2) = List.ofFn ys ++ zs)
    (hlive : FirewallLive.liveRun s { salts := [] } L = .ok (sent, t)) :
    sent.toArray = T (Flock.Guarantees.firewallStrings (msgs P w r a p) ys) ∧ t.finished = true := by
  have e := FirewallLive.run_live s L sent t hlive
  have hY := hF P w r a p ys
  rw [hreqs, hsalts, FirewallLive.run_append s _ _ zs (by rw [hY]), hY] at e
  exact ⟨(congrArg Prod.fst e).symm, (congrArg Prod.snd e).symm⟩

/-- **The live send check sends the schedule's transcript**: `liveSends` under `FirewallAccepts`, at
`firewallT s tables`. -/
theorem liveSends_of_accepts {Prog Wit Rr Ωa Ωp : Type} {nh : ℕ} (s : Flock.Firewall.Schedule)
    (reqs : Prog → Wit → Rr → Ωa → Ωp → (Fin nh → ByteArray) → List ByteArray)
    (msgs : Prog → Wit → Rr → Ωa → Ωp → (i : Fin nh) → (Fin i → ByteArray) → ByteArray) (tables : List String)
    (hA : Flock.Guarantees.FirewallAccepts s reqs msgs tables)
    (P : Prog) (w : Wit) (r : Rr) (a : Ωa) (p : Ωp) (ys : Fin nh → ByteArray) (zs : List ByteArray)
    (L : List (ByteArray × List ByteArray)) (sent : List Flock.Firewall.Sent) (t : Flock.Firewall.State)
    (hreqs : L.map (·.1) = reqs P w r a p (Flock.Guarantees.firewallStrings (msgs P w r a p) ys))
    (hsalts : L.flatMap (·.2) = List.ofFn ys ++ zs)
    (hlive : FirewallLive.liveRun s { salts := [] } L = .ok (sent, t)) :
    sent.toArray = Flock.Guarantees.firewallT s tables (Flock.Guarantees.firewallStrings (msgs P w r a p) ys) ∧
      t.finished = true :=
  liveSends s reqs msgs _ (firewallSends_of_accepts s reqs msgs tables hA) P w r a p ys zs L sent t hreqs hsalts hlive

/-- **The honest worker's half of `FirewallSends`**: at a schedule the ZK server holds a session to, the honest worker's
requests (`honestReqs`) pass the ZK server, Commit naming the schedule's tables (`Schedule.tables`), and commit its rows'
messages (`FirewallHonest.honestMsgs`), wherever at every commit strings it follows the schedule (`FirewallHonest.Fits`)
and its rows read only the commit strings before them (`FirewallHonest.Causal`). -/
theorem honestReqs_accepts {Prog Wit Rr Ωa Ωp : Type} (s : Flock.Firewall.Schedule)
    (hs : FirewallHonest.Canonical s)
    (D : Prog → Wit → Rr → Ωa → Ωp → (Fin (FirewallHonest.rowCount s) → ByteArray) → Flock.Firewall.Honest)
    (hD : ∀ P w r a p l, FirewallHonest.Fits s (D P w r a p l)) (hc : FirewallHonest.Causal s D) :
    Flock.Guarantees.FirewallAccepts s (fun P w r a p l => Flock.Firewall.honestReqs s (D P w r a p l))
      (FirewallHonest.honestMsgs s D) s.tables :=
  FirewallHonest.honest_accepts hs hD hc Flock.Guarantees.zeroSalt

end Flock.SecurityProofs
