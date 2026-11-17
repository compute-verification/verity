module

public import Proofs.Flock.Recursive.ZkServerLive
public import Proofs.Flock.Recursive.ZkServerHonest
public import Proofs.Flock.Recursive.ZkServer
import Flock.ZkServer.Contract
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
import Proofs.Flock.Recursive.ZkServerZK

/-! `ZkServerSends` from the ZK server's code, leaving the honest worker's half (`ZkServerAccepts`). -/

@[expose] public section

namespace Flock.Guarantees

open FlockSoundness.Discharge.Recursion

def zeroSalt : ByteArray := ⟨Array.replicate 192 0⟩

/-- **The schedule's transcript filled with the commit strings `l`**, in salt order, Commit naming the tables `tables`
in order (`ZkServerTemplate.tmpl`). -/
def zkServerT (s : Flock.ZkServer.Schedule) (tables : List String) {nh : ℕ} (l : Fin nh → ByteArray) :
    Array Flock.ZkServer.Sent :=
  (ZkServerRun.fillAll (ZkServerTemplate.tmpl s tables) (List.ofFn l)).toArray

/-- Obligation: the honest worker's requests pass the ZK server, Commit names the tables `tables`, and the commit
strings it sends are those of `msgs` in salt order, message `i` reading only the strings before it. It says nothing of how
the ZK server salts, nor of what it sends but the table order. -/
def ZkServerAccepts {Prog Wit Rr Ωa Ωp : Type} {nh : ℕ} (s : Flock.ZkServer.Schedule)
    (reqs : Prog → Wit → Rr → Ωa → Ωp → (Fin nh → ByteArray) → List ByteArray)
    (msgs : Prog → Wit → Rr → Ωa → Ωp → (i : Fin nh) → (Fin i → ByteArray) → ByteArray) (tables : List String) :
    Prop :=
  ∀ P w r a p (l : Fin nh → ByteArray),
    (Flock.ZkServer.run s (reqs P w r a p l) (List.ofFn fun _ : Fin nh => zeroSalt)).2 = true ∧
      ZkServerTemplate.namesOf (Flock.ZkServer.run s (reqs P w r a p l) (List.ofFn fun _ : Fin nh => zeroSalt)).1.toList =
        tables ∧
      ZkServerRun.holesOf (Flock.ZkServer.run s (reqs P w r a p l) (List.ofFn fun _ : Fin nh => zeroSalt)).1 =
        List.ofFn fun i =>
          Hm96.Default512.commitString (Sha512.hash (msgs P w r a p i fun j => l (Fin.castLE i.isLt.le j))) zeroSalt

end Flock.Guarantees

namespace Flock.SecurityProofs

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game
open scoped ENNReal

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

theorem zkServerStrings_eq {nh : ℕ} (msgs : (i : Fin nh) → (Fin i → ByteArray) → ByteArray) (ys : Fin nh → ByteArray)
    (i : Fin nh) :
    Flock.Guarantees.zkServerStrings msgs ys i = Hm96.Default512.commitString
      (Sha512.hash (msgs i fun j => Flock.Guarantees.zkServerStrings msgs ys (Fin.castLE i.isLt.le j))) (ys i) := by
  rw [Flock.Guarantees.zkServerStrings]

theorem zkServerSends_of_accepts {Prog Wit Rr Ωa Ωp : Type} {nh : ℕ} (s : Flock.ZkServer.Schedule)
    (reqs : Prog → Wit → Rr → Ωa → Ωp → (Fin nh → ByteArray) → List ByteArray)
    (msgs : Prog → Wit → Rr → Ωa → Ωp → (i : Fin nh) → (Fin i → ByteArray) → ByteArray) (tables : List String)
    (h : Flock.Guarantees.ZkServerAccepts s reqs msgs tables) :
    Flock.Guarantees.ZkServerSends s reqs msgs (Flock.Guarantees.zkServerT s tables) := by
  intro P w r a p ys
  obtain ⟨hfin, ho, hh⟩ := h P w r a p (Flock.Guarantees.zkServerStrings (msgs P w r a p) ys)
  refine (ZkServerTemplate.sends_of_one s _ tables
    (fun i => Sha512.hash (msgs P w r a p i fun j =>
      Flock.Guarantees.zkServerStrings (msgs P w r a p) ys (Fin.castLE i.isLt.le j)))
    (fun _ => Flock.Guarantees.zeroSalt) (fun _ => Flock.ZkServer.Contract.sha512_size _) hfin ho hh ys).trans ?_
  refine congrArg (fun l => ((ZkServerRun.fillAll (ZkServerTemplate.tmpl s tables) l).toArray, true)) ?_
  exact congrArg List.ofFn (funext fun i => (zkServerStrings_eq (msgs P w r a p) ys i).symm)

theorem tables_eq_of_accepts {Prog Wit Rr Ωa Ωp : Type} {nh : ℕ} (s : Flock.ZkServer.Schedule)
    (reqs : Prog → Wit → Rr → Ωa → Ωp → (Fin nh → ByteArray) → List ByteArray)
    (msgs : Prog → Wit → Rr → Ωa → Ωp → (i : Fin nh) → (Fin i → ByteArray) → ByteArray) (tables : List String)
    (h : Flock.Guarantees.ZkServerAccepts s reqs msgs tables) (P : Prog) (w : Wit) (r : Rr) (a : Ωa) (p : Ωp)
    (l : Fin nh → ByteArray) : tables = s.tables := by
  obtain ⟨hfin, ho, -⟩ := h P w r a p l
  exact ho.symm.trans (ZkServerTemplate.namesOf_run s _ _ hfin)

theorem recursiveZkServerPrivate_of_accepts {Out Prog Wit Rv Ωa Ωp Pub : Type} [Fintype Ωa] [Inhabited Ωa]
    [Fintype Ωp] [Inhabited Ωp] (s : Flock.ZkServer.Schedule) (nr : ℕ) (rows : Prog → Fin nr → ByteArray)
    (pub : (Fin nr → ByteArray) → Rv) (honest : Out → Prog → Wit → Prop) (nh : ℕ)
    (reqs : Prog → Wit → (Fin nr → ZkView.Salt) → Ωa → Ωp → (Fin nh → ByteArray) → List ByteArray)
    (msgs : Prog → Wit → (Fin nr → ZkView.Salt) → Ωa → Ωp → (i : Fin nh) → (Fin i → ByteArray) → ByteArray)
    (tables : List String) (V : OuterZK.Sessions (OuterZK.HCtx Prog Wit Ωa Ωp nr nh) Pub)
    (pubS : Out → Rv → Ωa → (Fin nh → ByteArray) → Pub) (stopped : OuterZK.Stopped) (p₀ : Prog) (o : Out) (P : Prog)
    (w : Wit) (hA : Flock.Guarantees.ZkServerAccepts s reqs msgs tables)
    (hpub : Flock.Guarantees.SessionsPublic rows pub honest msgs V pubS) (hC : Flock.Guarantees.SessionsCoined V)
    (hH : Flock.Guarantees.SessionsHonest honest msgs V) (hw : honest o P w)
    (E : Rv × Ωa × (Array Flock.ZkServer.Sent × Bool) × V.View → Prop) :
    Flock.Guarantees.fwRealPr rows pub s reqs msgs V (OuterZK.proverExe stopped) P w E ≤
        Flock.Guarantees.fwSimPr rows pub (Flock.Guarantees.zkServerT s tables) V pubS o p₀ E +
          (2 * nr * ENNReal.ofReal (CROnly.hidingGap ZkView.My ZkView.saltHash) +
            V.errAt (ENNReal.ofReal (CROnly.hidingGap ZkView.My ZkView.saltHash)) +
            nh * ENNReal.ofReal (CROnly.hidingGap ZkView.My ZkView.saltHash)) ∧
      Flock.Guarantees.fwSimPr rows pub (Flock.Guarantees.zkServerT s tables) V pubS o p₀ E ≤
        Flock.Guarantees.fwRealPr rows pub s reqs msgs V (OuterZK.proverExe stopped) P w E +
          (2 * nr * ENNReal.ofReal (CROnly.hidingGap ZkView.My ZkView.saltHash) +
            V.errAt (ENNReal.ofReal (CROnly.hidingGap ZkView.My ZkView.saltHash)) +
            nh * ENNReal.ofReal (CROnly.hidingGap ZkView.My ZkView.saltHash)) :=
  RecursiveZkServerPrivate s nr rows pub honest nh reqs msgs (Flock.Guarantees.zkServerT s tables) V pubS stopped p₀ o
    P w (zkServerSends_of_accepts s reqs msgs tables hA) hpub hC hH hw E

theorem liveSends {Prog Wit Rr Ωa Ωp : Type} {nh : ℕ} (s : Flock.ZkServer.Schedule)
    (reqs : Prog → Wit → Rr → Ωa → Ωp → (Fin nh → ByteArray) → List ByteArray)
    (msgs : Prog → Wit → Rr → Ωa → Ωp → (i : Fin nh) → (Fin i → ByteArray) → ByteArray)
    (T : (Fin nh → ByteArray) → Array Flock.ZkServer.Sent) (hF : Flock.Guarantees.ZkServerSends s reqs msgs T)
    (P : Prog) (w : Wit) (r : Rr) (a : Ωa) (p : Ωp) (ys : Fin nh → ByteArray) (zs : List ByteArray)
    (L : List (ByteArray × List ByteArray)) (sent : List Flock.ZkServer.Sent) (t : Flock.ZkServer.State)
    (hreqs : L.map (·.1) = reqs P w r a p (Flock.Guarantees.zkServerStrings (msgs P w r a p) ys))
    (hsalts : L.flatMap (·.2) = List.ofFn ys ++ zs)
    (hlive : ZkServerLive.liveRun s { salts := [] } L = .ok (sent, t)) :
    sent.toArray = T (Flock.Guarantees.zkServerStrings (msgs P w r a p) ys) ∧ t.finished = true := by
  have e := ZkServerLive.run_live s L sent t hlive
  have hY := hF P w r a p ys
  rw [hreqs, hsalts, ZkServerLive.run_append s _ _ zs (by rw [hY]), hY] at e
  exact ⟨(congrArg Prod.fst e).symm, (congrArg Prod.snd e).symm⟩

theorem liveSends_of_accepts {Prog Wit Rr Ωa Ωp : Type} {nh : ℕ} (s : Flock.ZkServer.Schedule)
    (reqs : Prog → Wit → Rr → Ωa → Ωp → (Fin nh → ByteArray) → List ByteArray)
    (msgs : Prog → Wit → Rr → Ωa → Ωp → (i : Fin nh) → (Fin i → ByteArray) → ByteArray) (tables : List String)
    (hA : Flock.Guarantees.ZkServerAccepts s reqs msgs tables)
    (P : Prog) (w : Wit) (r : Rr) (a : Ωa) (p : Ωp) (ys : Fin nh → ByteArray) (zs : List ByteArray)
    (L : List (ByteArray × List ByteArray)) (sent : List Flock.ZkServer.Sent) (t : Flock.ZkServer.State)
    (hreqs : L.map (·.1) = reqs P w r a p (Flock.Guarantees.zkServerStrings (msgs P w r a p) ys))
    (hsalts : L.flatMap (·.2) = List.ofFn ys ++ zs)
    (hlive : ZkServerLive.liveRun s { salts := [] } L = .ok (sent, t)) :
    sent.toArray = Flock.Guarantees.zkServerT s tables (Flock.Guarantees.zkServerStrings (msgs P w r a p) ys) ∧
      t.finished = true :=
  liveSends s reqs msgs _ (zkServerSends_of_accepts s reqs msgs tables hA) P w r a p ys zs L sent t hreqs hsalts hlive

theorem honestReqs_accepts {Prog Wit Rr Ωa Ωp : Type} (s : Flock.ZkServer.Schedule)
    (hs : ZkServerHonest.Canonical s)
    (D : Prog → Wit → Rr → Ωa → Ωp → (Fin (ZkServerHonest.rowCount s) → ByteArray) → Flock.ZkServer.Honest)
    (hD : ∀ P w r a p l, ZkServerHonest.Fits s (D P w r a p l)) (hc : ZkServerHonest.Causal s D) :
    Flock.Guarantees.ZkServerAccepts s (fun P w r a p l => Flock.ZkServer.honestReqs s (D P w r a p l))
      (ZkServerHonest.honestMsgs s D) s.tables :=
  ZkServerHonest.honest_accepts hs hD hc Flock.Guarantees.zeroSalt

end Flock.SecurityProofs
