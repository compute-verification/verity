module

public import Proofs.Flock.Recursive.ZkServer
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
import Proofs.Flock.Recursive.OuterZK
import Proofs.Flock.Soundness.CROnly.Hiding

/-! The committed-coin statements' `_hOuter` at V*'s sessions as `Sessions` models them, which does not cover that the deployed outer sessions send `Sessions.view`. -/

@[expose] public section

namespace FlockSoundness.Discharge.Recursion.OuterZK

open Game
open scoped ENNReal

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- as in the ZK files this builds on; its own need is unmeasured

def committed {Prog Wit Ωa Ωp : Type} {nr nh : ℕ}
    (msgs : Prog → Wit → (Fin nr → ZkView.Salt) → Ωa → Ωp → Fin nh → ByteArray) :
    Prog → Wit → (Fin nr → ZkView.Salt) → Ωa → Ωp → (i : Fin nh) → (Fin i → ByteArray) → ByteArray :=
  fun P w r a p i _ => msgs P w r a p i

theorem zkServerCommits_committed {nh : ℕ} (m : Fin nh → ByteArray) (y : Fin nh → ZkView.Salt) :
    Flock.Guarantees.zkServerCommits (fun i _ => m i) y = fun i => Flock.Guarantees.zkServerLeaf (m i) (y i) := by
  funext i
  rw [Flock.Guarantees.zkServerCommits, ZkServerZK.zkServerPairs_eq, Flock.Guarantees.zkServerLeaf]

/-- It takes no premise but three obligations on V*'s staging: `SessionsPublic`, `SessionsCoined` and
`SessionsHonest`. -/
theorem recursiveZK_of_sessionZK {Out Prog Wit Rv Ωa Ωp Pub : Type} {nr nh : ℕ} (rows : Prog → Fin nr → ByteArray)
    (pub : (Fin nr → ByteArray) → Rv) (honest : Out → Prog → Wit → Prop)
    (msgs : Prog → Wit → (Fin nr → ZkView.Salt) → Ωa → Ωp → Fin nh → ByteArray)
    (V : Sessions (HCtx Prog Wit Ωa Ωp nr nh) Pub) (pubS : Out → Rv → Ωa → (Fin nh → ByteArray) → Pub)
    (stopped : Stopped)
    (hpub : Flock.Guarantees.SessionsPublic rows pub honest (committed msgs) V pubS)
    (hC : Flock.Guarantees.SessionsCoined V)
    (hH : Flock.Guarantees.SessionsHonest honest (committed msgs) V) :
    Flock.Guarantees.OuterSessionsZK (Flock.Guarantees.regLeaves rows pub) honest nh msgs
      (fun P w r a h p y q => V.view (proverExe stopped) (P, w, r, a, h, p, y) q)
      (fun o v a h s => V.sim (pubS o v a h) s)
      (V.errAt (ENNReal.ofReal (CROnly.hidingGap ZkView.My ZkView.saltHash))) := by
  obtain ⟨LP, Vs, Reg', D, draw, tab, ys, hCeq⟩ := hC
  have hO := vstar_outer_zk rows pub honest (committed msgs) V pubS hpub
    (tablesOK_of_sessions V LP Vs D draw tab ys hCeq) (selfCheckSilent_exe stopped) hH
    (CROnly.hm96Hiding_gap ZkView.My ZkView.saltHash)
  intro o P w hw r a p y E
  have e : Flock.Guarantees.zkServerCommits (committed msgs P w r a p) y =
      fun i => Flock.Guarantees.zkServerLeaf (msgs P w r a p i) (y i) :=
    zkServerCommits_committed (msgs P w r a p) y
  have h := hO o P w hw r a p y E
  rw [e] at h
  exact h

/-- **`RecursiveZK`'s `_hOuter` at V*'s sessions** (`OuterViewZK`): `recursiveZK_of_sessionZK` at one honest witness
`w` of a private circuit `P` owing `o`, and one registration randomness `r`. -/
theorem outerViewZK_of_sessionZK {Out Prog Wit Rv Ωa Ωp Pub : Type} {nr nh : ℕ} (rows : Prog → Fin nr → ByteArray)
    (pub : (Fin nr → ByteArray) → Rv) (honest : Out → Prog → Wit → Prop)
    (msgs : Prog → Wit → (Fin nr → ZkView.Salt) → Ωa → Ωp → Fin nh → ByteArray)
    (V : Sessions (HCtx Prog Wit Ωa Ωp nr nh) Pub) (pubS : Out → Rv → Ωa → (Fin nh → ByteArray) → Pub)
    (stopped : Stopped)
    (hpub : Flock.Guarantees.SessionsPublic rows pub honest (committed msgs) V pubS)
    (hC : Flock.Guarantees.SessionsCoined V)
    (hH : Flock.Guarantees.SessionsHonest honest (committed msgs) V)
    (o : Out) (P : Prog) (w : Wit) (hw : honest o P w) (r : Fin nr → ZkView.Salt) :
    Flock.Guarantees.OuterViewZK nh (msgs P w r)
      (fun a h p y q => V.view (proverExe stopped) (P, w, r, a, h, p, y) q)
      (fun a h s => V.sim (pubS o (Flock.Guarantees.regLeaves rows pub P r) a h) s)
      (V.errAt (ENNReal.ofReal (CROnly.hidingGap ZkView.My ZkView.saltHash))) :=
  recursiveZK_of_sessionZK rows pub honest msgs V pubS stopped hpub hC hH o P w hw r

end FlockSoundness.Discharge.Recursion.OuterZK
