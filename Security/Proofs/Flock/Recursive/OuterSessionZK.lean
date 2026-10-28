import Proofs.Flock.Recursive.FirewallZK

/-!
# Recursion: the committed-coin statements' `_hOuter` at V*'s sessions

`RecursiveZK`, `RecursiveCircuitPrivate`, `RecursiveCircuitIndist` and `RecursiveCircuitPrivateLeaves` take the outer
session's zero knowledge as an obligation on the view and simulator they range over (`Flock.Guarantees.OuterViewZK`,
`Flock.Guarantees.OuterSessionsZK`). Here it is at V*'s outer `--zk` sessions as `Proofs.Flock.Recursive.Sessions`
models them, run by the honest prover's code (`proverExe`, whatever a stopped run sends), with the commitments the
firewall sends when every message is fixed by the auditor's tape before the first commitment (the committed-coin case).

* `committed`, `firewallCommits_committed`: messages that read none of the commit strings before them, at which the
  firewall's commitments are `firewallLeaf` of each message under its own salt.
* `recursiveZK_of_sessionZK`: `OuterSessionsZK` at `Sessions.view` and `Sessions.sim`, with `ε_o` V*'s sessions' error
  at the pinned key's hiding gap (`errAt`), from C-Flock's zero knowledge of one `--zk` table at every coin vector
  (`ZkEveryCoin.coin_close_le`, through `OuterZK.vstar_outer_zk`), on V*'s staging, each a named obligation:
  `SessionsPublic` and `SessionsHonest` at `committed msgs`, and `SessionsCoined`. V*'s sizes are `Sessions`' fields
  `hz`, `hR`, `hS`. `outerViewZK_of_sessionZK` is `RecursiveZK`'s `OuterViewZK` at one honest witness and registration.

What this does not cover: that the deployed outer sessions send `Sessions.view`, the honest prover's view at coins the
auditor's tape fixes before each session (`Sessions.coin`). C-Flock's session-loop bound (`SessionLoopZKAt`) is about
the deployed loop against every `V*`, but it bounds the real view by twice the simulated one plus its error, so it
doesn't give `OuterViewZK`'s additive form.
-/

namespace FlockSoundness.Discharge.Recursion.OuterZK

open Game
open scoped ENNReal

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-- **Messages fixed before the first commitment**: `msgs` read as live-coin messages that read none of the commit
strings before them. -/
def committed {Prog Wit Ωa Ωp : Type} {nr nh : ℕ}
    (msgs : Prog → Wit → (Fin nr → ZkView.Salt) → Ωa → Ωp → Fin nh → ByteArray) :
    Prog → Wit → (Fin nr → ZkView.Salt) → Ωa → Ωp → (i : Fin nh) → (Fin i → ByteArray) → ByteArray :=
  fun P w r a p i _ => msgs P w r a p i

/-- **The committed-coin case of the firewall's commitments**: where no message reads the commit strings before it,
commitment `i` is `firewallLeaf` of message `i` under its salt. -/
theorem firewallCommits_committed {nh : ℕ} (m : Fin nh → ByteArray) (y : Fin nh → ZkView.Salt) :
    Flock.Guarantees.firewallCommits (fun i _ => m i) y = fun i => Flock.Guarantees.firewallLeaf (m i) (y i) := by
  funext i
  rw [Flock.Guarantees.firewallCommits, FirewallZK.firewallPairs_eq, Flock.Guarantees.firewallLeaf]

/-- **V*'s sessions give the committed-coin statements their `_hOuter`** (`OuterSessionsZK`), at the registration by
hm96 leaves of the private circuit's rows (`regLeaves rows pub`): the view is V*'s sessions' (`Sessions.view`) run by
the honest prover's code at the honest context, the simulator V*'s (`Sessions.sim`) at the public context `pubS` reads,
and `ε_o` V*'s sessions' error at the pinned key's hiding gap, `errAt (hidingGap My saltHash)`. It takes no premise;
what it takes of V*'s staging, each an obligation: the simulator's public context is the honest one's
(`SessionsPublic`), each session's table is one of `verify --zk`'s hidden-output tables at its data and a draw
(`SessionsCoined`), and the honest witnesses are valid with public registered values (`SessionsHonest`). -/
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
  have e : Flock.Guarantees.firewallCommits (committed msgs P w r a p) y =
      fun i => Flock.Guarantees.firewallLeaf (msgs P w r a p i) (y i) :=
    firewallCommits_committed (msgs P w r a p) y
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
