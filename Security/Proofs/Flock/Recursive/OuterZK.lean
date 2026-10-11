import Proofs.Flock.Recursive.Sessions
import Proofs.Flock.Recursive.Private
import Proofs.Flock.Soundness.Discharge.ZkEveryCoin.Headline
import Proofs.Flock.Soundness.Discharge.ZkHidden.View
import Proofs.Flock.Soundness.Discharge.FailClosed.Mask
import Proofs.Flock.Recursive.OuterAbort

/-!
# Recursion: V*'s outer sessions are zero knowledge, (ii)'s `_hOuter`

(ii) of `RecursiveAudit` takes V*'s sessions' zero knowledge as `_hOuter`: a simulator reading the owed outputs, the
registration's view, the auditor's tape and the firewall's commitments, within `ε_o` of the auditor's view of V*'s
sessions at every honest context. Here that view, its simulator and `ε_o`, at V*'s outer sessions as
`Proofs.Flock.Recursive.Sessions` defines them (`Sessions`: the prover registers `R₂`, `n₂` hm96 commit strings under
fresh salts, and runs `S` sessions, each one `--zk` table at the coin vector the auditor's tape fixes, Goldreich–Kahan's
committed coins, which read the tape and what was public before the session).

* `honestOut_eq`: under the abort interface (`SelfCheckSilent`: at a table that passes `CoinedOK`, coins the prover
  doesn't refuse and a valid witness whose registered values are the public ones, the code sends its view), the
  honest prover is `outExe`.
* `selfCheckSilent_exe`: the prover's code (`proverExe`) meets the abort interface whatever a stopped run sends
  (`Table.not_stops` at `CoinedOK`'s inner relation, `OuterAbort`).
* `table_close`: one table at its coins, in fixed randomness types (`coin_close_le`, which covers refused coins with
  `Table.table_abort_hm96`).
* `Sessions.close`: the auditor's view of `R₂` and the sessions (`view`) and the simulator's (`sim`: ideal commit
  strings for `R₂`, `simAll` at each table) are within `errAt δ = Σ_j 2·|LeafPos z_j|·δ + n₂·δ` at any hm96 hiding
  bound `δ`. The sessions one by one at each of `R₂`'s salts (`ZK.prCoin_pi_close`), then `R₂`'s commit strings to
  ideal ones (`ZK.ideal_leaves_swap`); the sessions read `R₂`'s salts only through their witnesses, which the
  simulator doesn't read, so the second step follows the first.
* `tablesOK_of_sessions`: V*'s tables pass `CoinedOK` where each is `coinedOf` at a table of `verify --zk`'s
  hidden-output session of the session's data at a draw, where it sets that table up (`coinedOf_setup_of`; every such
  setup avoids the mask slot, `FailClosed.DrawSetupZHJ.avoidsMask`).
* `vstar_outer_zk`: `_hOuter` at `Sessions.view` and `Sessions.sim`, with `ε_o = errAt δ`, at the firewall's
  commitments under live inner coins (`firewallCommits`); `Flock.SecurityProofs.RecursiveFirewallPrivate`
  (`Proofs.Flock.Recursive.FirewallZK`) takes it at `proverExe` and the pinned key's hiding gap, with no hypothesis on
  the abort.

Left to the instance at V*'s staging: that each session's table is `coinedOf` at a table of its data and draw (with a
`ZkHidden.DrawSetupZHJ` there), its sizes fixed by V*'s statement (`hz`, `hR`, `hS`), and the honest witnesses
(`Honest`, which (ii)'s `honest` carries).
-/

namespace FlockSoundness.Discharge.Recursion.OuterZK

open Game Composed ZkView Model Flock ZkEveryCoin ZkLink
open scoped ENNReal

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-- **Under the abort interface, the honest prover is `outExe`.** -/
theorem honestOut_eq {prover : TableProver} (hP : SelfCheckSilent prover) {C : CoinedSetup}
    (hC : CoinedOK execArith (Refine.fOf Flock.Zk.PAD_C) C) (c : CoinsZK C.Rest) {w : Witness C.S.m}
    (hv : C.Valid w) (hreg : ∀ c', (C.T c').tab.sReg w = (C.T c').tab.pubReg) (p : RandAt C c) :
    honestOut prover C c w p = outExe C c w p.1 p.2 := by
  unfold honestOut outExe
  split_ifs with hA
  · exact hP C c w hC hA hv hreg p
  · rfl

/-- **The prover's code meets the abort interface**, whatever a stopped run sends: at coins it doesn't refuse,
`CoinedOK`'s inner relation holds at a valid witness with public registered values, so the self-check never stops
(`Table.not_stops`). -/
theorem selfCheckSilent_exe
    (stopped : ∀ (C : CoinedSetup) (c : CoinsZK C.Rest), Witness C.S.m → RandAt C c → (C.T c).tab.Cut) :
    SelfCheckSilent (proverExe stopped) := by
  intro C c w hC hA hv hreg p
  obtain ⟨-, -, -, hT⟩ := hC
  obtain ⟨-, -, -, -, -, -, -, hinner⟩ := hT c hA
  unfold proverExe
  exact ite_eq_right ((C.T c).tab.not_stops (hinner w hv (hreg c)) p.1)

/-- **One table at its coins, in fixed randomness types** (`coin_close_le`): at any coin vector, through any encoding
`enc` of the table's output and any equivalences `eR`, `eS` of `Fin NR` and `Fin NS` with the prover's and the
simulator's randomness, the honest output and the simulator's are within `2·|LeafPos z|·δ`, at any hm96 hiding bound
`δ`. -/
theorem table_close (C : CoinedSetup) (hC : CoinedOK execArith (Refine.fOf Flock.Zk.PAD_C) C)
    {δ : ℝ≥0∞} (hT1 : FlockSoundness.Assumptions.Hm96Hiding ZkView.My ZkView.saltHash δ)
    (w w0 : Witness C.S.m) (hv : C.Valid w)
    (hreg : ∀ c, (C.T c).tab.sReg w = (C.T c).tab.pubReg) (c : CoinsZK C.Rest) {z : ZkView.Sizes} (hz : C.z = z)
    {Vw : Type} (enc : ∀ c' : CoinsZK C.Rest, OutAt C c' → Vw) {NR NS : ℕ} (eR : Fin NR ≃ RandAt C c)
    (eS : Fin NS ≃ SimAt C c) (E : Vw → Prop) :
    prCoin (fun ρ : Fin NR => E (enc c (outExe C c w (eR ρ).1 (eR ρ).2)))
      ≤ prCoin (fun σ : Fin NS => E (enc c (simAll C c w0 (eS σ).1 (eS σ).2)))
        + 2 * Fintype.card (ZkView.LeafPos z) * δ := by
  subst hz
  calc prCoin (fun ρ : Fin NR => E (enc c (outExe C c w (eR ρ).1 (eR ρ).2)))
      = prCoin (fun p : RandAt C c => E (enc c (outExe C c w p.1 p.2))) :=
        ZK.prCoin_congr_equiv eR (fun p : RandAt C c => E (enc c (outExe C c w p.1 p.2)))
    _ ≤ prCoin (fun p : SimAt C c => E (enc c (simAll C c w0 p.1 p.2)))
          + 2 * Fintype.card (ZkView.LeafPos C.z) * δ :=
        (coin_close_le C hC hT1 w w0 hv hreg (fun c' v => E (enc c' v)) c).1
    _ = prCoin (fun σ : Fin NS => E (enc c (simAll C c w0 (eS σ).1 (eS σ).2)))
          + 2 * Fintype.card (ZkView.LeafPos C.z) * δ :=
        congrArg (· + 2 * (Fintype.card (ZkView.LeafPos C.z) : ℝ≥0∞) * δ)
          (ZK.prCoin_congr_equiv eS (fun p : SimAt C c => E (enc c (simAll C c w0 p.1 p.2)))).symm

/-! ## V*'s sessions -/

namespace Sessions

variable {Ctx Pub : Type} (V : Sessions Ctx Pub)

theorem nr_pos (j : Fin V.S) (pc : Pub) (l : Fin V.n₂ → ZkView.Bits × ByteArray) : 0 < V.NR j := by
  rw [← V.hR j pc l]
  have : Nonempty (RandAt (V.C j pc l) (V.coin j pc l)) :=
    ⟨((ZkHidden.rand_nonempty ((V.C j pc l).T (V.coin j pc l)).tab).some, fun _ => 0)⟩
  exact Fintype.card_pos

theorem ns_pos (j : Fin V.S) (pc : Pub) (l : Fin V.n₂ → ZkView.Bits × ByteArray) : 0 < V.NS j := by
  rw [← V.hS j pc l]
  have : Nonempty (SimAt (V.C j pc l) (V.coin j pc l)) :=
    ⟨((ZkHidden.simRand_nonempty ((V.C j pc l).T (V.coin j pc l)).tab).some, fun _ => 0)⟩
  exact Fintype.card_pos

/-- **One session at the honest context and `R₂`'s salts** (`table_close` at its table and coins, the honest prover
`outExe` by the abort interface), at any hm96 hiding bound `δ`. -/
theorem session_close (hOK : V.TablesOK)
    {δ : ℝ≥0∞} (hT1 : FlockSoundness.Assumptions.Hm96Hiding ZkView.My ZkView.saltHash δ)
    {prover : TableProver} (hP : SelfCheckSilent prover) (x : Ctx) (hx : V.Honest x)
    (y₂ : Fin V.n₂ → ZkView.Salt) (j : Fin V.S) (E : V.Vw j → Prop) :
    prCoin (fun ρ : Fin (V.NR j) => E (V.realAt prover x y₂ j ρ))
      ≤ prCoin (fun σ : Fin (V.NS j) => E (V.simAt (V.pubOf x) (commitStrs (V.m₂ x) y₂) j σ))
        + 2 * (Fintype.card (ZkView.LeafPos (V.z j)) : ℝ≥0∞) * δ := by
  have h := table_close (V.C j (V.pubOf x) (commitStrs (V.m₂ x) y₂)) (hOK j (V.pubOf x) (commitStrs (V.m₂ x) y₂))
    hT1 (V.wit j x y₂) (w0 _)
    (hx j y₂).1 (hx j y₂).2 (V.coin j (V.pubOf x) (commitStrs (V.m₂ x) y₂)) (V.hz j _ _)
    (V.enc j (V.pubOf x) (commitStrs (V.m₂ x) y₂)) (V.eR j (V.pubOf x) (commitStrs (V.m₂ x) y₂))
    (V.eS j (V.pubOf x) (commitStrs (V.m₂ x) y₂)) E
  refine Eq.trans_le (congrArg prCoin (funext fun ρ => congrArg E
    (congrArg (V.enc j (V.pubOf x) (commitStrs (V.m₂ x) y₂) (V.coin j (V.pubOf x) (commitStrs (V.m₂ x) y₂)))
      (honestOut_eq hP (hOK j (V.pubOf x) (commitStrs (V.m₂ x) y₂)) (V.coin j (V.pubOf x) (commitStrs (V.m₂ x) y₂))
        (hx j y₂).1 (hx j y₂).2
        (V.eR j (V.pubOf x) (commitStrs (V.m₂ x) y₂) ρ))))) h

/-- **V*'s sessions are zero knowledge at an honest context**: the auditor's view of `R₂` and the sessions and the
simulator's are within `errAt δ`, at any hm96 hiding bound `δ`. At each of `R₂`'s salts the sessions go to their
simulators one by one (`ZK.prCoin_pi_close`, `session_close`); then nothing reads `R₂`'s salts but its commit strings,
which go to ideal ones (`ZK.ideal_leaves_swap`). -/
theorem close (hOK : V.TablesOK)
    {δ : ℝ≥0∞} (hT1 : FlockSoundness.Assumptions.Hm96Hiding ZkView.My ZkView.saltHash δ) {prover : TableProver}
    (hP : SelfCheckSilent prover) (x : Ctx) (hx : V.Honest x) (E : V.View → Prop) :
    prCoin (fun q : V.Ωo => E (V.view prover x q))
      ≤ prCoin (fun s : V.Ωs => E (V.sim (V.pubOf x) s)) + V.errAt δ := by
  have : ∀ j, Nonempty (Fin (V.NR j)) := fun j => ⟨⟨0, V.nr_pos j (V.pubOf x) (commitStrs (V.m₂ x) fun _ => 0)⟩⟩
  have : ∀ j, Nonempty (Fin (V.NS j)) := fun j => ⟨⟨0, V.ns_pos j (V.pubOf x) (commitStrs (V.m₂ x) fun _ => 0)⟩⟩
  have step : ∀ y₂ : Fin V.n₂ → ZkView.Salt,
      prCoin (fun ρ : (j : Fin V.S) → Fin (V.NR j) => E (V.view prover x (y₂, ρ)))
        ≤ prCoin (fun σ : (j : Fin V.S) → Fin (V.NS j) => E (V.simOut (V.pubOf x) σ (commitStrs (V.m₂ x) y₂)))
          + V.errSAt δ := fun y₂ =>
    ZK.prCoin_pi_close (Ω := fun j => Fin (V.NR j)) (Ω' := fun j => Fin (V.NS j)) (V := V.Vw)
      (fun j ρ => V.realAt prover x y₂ j ρ) (fun j σ => V.simAt (V.pubOf x) (commitStrs (V.m₂ x) y₂) j σ)
      (fun j => 2 * (Fintype.card (ZkView.LeafPos (V.z j)) : ℝ≥0∞) * δ)
      (fun j E' => V.session_close hOK hT1 hP x hx y₂ j E') (fun v => E (commitStrs (V.m₂ x) y₂, v))
  have swap := ZK.ideal_leaves_swap ZkView.My ZkView.saltHash hT1 (fun p : ZkView.Bits × ByteArray => p) V.n₂
    (Ω := (j : Fin V.S) → Fin (V.NS j)) (fun _ i => ZkView.bitsOf (Sha512.hash (V.m₂ x i)))
    (fun σ l => E (V.simOut (V.pubOf x) σ l))
  calc prCoin (fun q : V.Ωo => E (V.view prover x q))
      = avg fun y₂ : Fin V.n₂ → ZkView.Salt =>
          prCoin fun ρ : (j : Fin V.S) → Fin (V.NR j) => E (V.view prover x (y₂, ρ)) :=
        ZK.prCoin_prod_split (fun (y₂ : Fin V.n₂ → ZkView.Salt) (ρ : (j : Fin V.S) → Fin (V.NR j)) =>
          E (V.view prover x (y₂, ρ)))
    _ ≤ avg fun y₂ : Fin V.n₂ → ZkView.Salt =>
          prCoin (fun σ : (j : Fin V.S) → Fin (V.NS j) => E (V.simOut (V.pubOf x) σ (commitStrs (V.m₂ x) y₂)))
            + V.errSAt δ := avg_mono step
    _ = (avg fun y₂ : Fin V.n₂ → ZkView.Salt =>
          prCoin fun σ : (j : Fin V.S) → Fin (V.NS j) => E (V.simOut (V.pubOf x) σ (commitStrs (V.m₂ x) y₂)))
            + V.errSAt δ := by rw [avg_add, avg_const]
    _ = prCoin (fun p : (Fin V.n₂ → ZkView.Salt) × ((j : Fin V.S) → Fin (V.NS j)) =>
          E (V.simOut (V.pubOf x) p.2 (commitStrs (V.m₂ x) p.1))) + V.errSAt δ :=
        congrArg (· + V.errSAt δ) (ZK.prCoin_prod_split
          (fun (y₂ : Fin V.n₂ → ZkView.Salt) (σ : (j : Fin V.S) → Fin (V.NS j)) =>
            E (V.simOut (V.pubOf x) σ (commitStrs (V.m₂ x) y₂)))).symm
    _ = prCoin (fun p : ((j : Fin V.S) → Fin (V.NS j)) × (Fin V.n₂ → ZkView.Salt) =>
          E (V.simOut (V.pubOf x) p.1 (commitStrs (V.m₂ x) p.2))) + V.errSAt δ :=
        congrArg (· + V.errSAt δ) (ZK.prCoin_congr_equiv
          (Equiv.prodComm ((j : Fin V.S) → Fin (V.NS j)) (Fin V.n₂ → ZkView.Salt))
          (fun p : (Fin V.n₂ → ZkView.Salt) × ((j : Fin V.S) → Fin (V.NS j)) =>
            E (V.simOut (V.pubOf x) p.2 (commitStrs (V.m₂ x) p.1)))).symm
    _ ≤ (prCoin (fun s : V.Ωs => E (V.sim (V.pubOf x) s)) + V.n₂ * δ) + V.errSAt δ :=
        add_le_add swap le_rfl
    _ = prCoin (fun s : V.Ωs => E (V.sim (V.pubOf x) s)) + V.errAt δ := by
        rw [Sessions.errAt]
        ring

end Sessions

/-! ## V*'s tables, and (ii) -/

/-- **V*'s session tables pass `CoinedOK`** where each is `coinedOf` at a table of `verify --zk`'s hidden-output session
of the session's data at a draw, where `verify --zk` sets that table up (`coinedOf_setup_of`); every such setup is a
`MaskSetup` (`ZkHidden.maskSetup_of_setupHidden`) and avoids the mask slot (`FailClosed.DrawSetupZHJ.avoidsMask`). -/
theorem tablesOK_of_sessions {Ctx Pub : Type} (V : Sessions Ctx Pub) (P : LigProver) (Vs : Fin V.S → VStmt)
    {Reg' : Type} (D : Pub → (Fin V.n₂ → ZkView.Bits × ByteArray) → (j : Fin V.S) → (Vs j).OuterD Reg')
    (draw : (j : Fin V.S) → Pub → (Fin V.n₂ → ZkView.Bits × ByteArray) → Finset (Fin (Vs j).nv))
    (tab : ∀ j pc l, Fin (nTab (D pc l j).I))
    (y : ∀ j pc l, ZkHidden.DrawSetupZHJ (inputsJ (D pc l j).I (tab j pc l)) (Vs j).mPts
      (partS (D pc l j).I ((D pc l j).dj (draw j pc l)) (draw j pc l) (tab j pc l))
      (drawJ (D pc l j).I ((D pc l j).dj (draw j pc l)) (tab j pc l)))
    (hC : ∀ j pc l, V.C j pc l =
      coinedOf P (ZkHidden.tableZHJ (D pc l j).dj (D pc l j).y₀ (draw j pc l) (tab j pc l)) (y j pc l).x.st) :
    V.TablesOK := by
  intro j pc l
  rw [hC j pc l]
  exact (coinedOf_setup_of P _ (y j pc l).x.st (y j pc l).x.wf.1
    (by rw [ZkHidden.tableZHJ, dif_pos ⟨y j pc l⟩, ZkHidden.DrawSetupZHJ.table_eq _ (y j pc l)]; rfl)
    (ZkHidden.maskSetup_of_setupHidden ((FlockSoundness.Discharge.Exec.tagsAt_typed _ _ _).trans
      ((FlockSoundness.Discharge.ZkExec.zkTags_typed _).trans (D pc l j).ht)) (y j pc l).x.h)
    (FailClosed.DrawSetupZHJ.avoidsMask (y j pc l))).2.2.2.2.2

/-- **V*'s sessions give (ii) its `_hOuter`**, with `ε_o = errAt δ` at any hm96 hiding bound `δ`: the view
`Sessions.view` at the honest context, the simulator `Sessions.sim` at the public context `pubS` reads (the owed
outputs, the registration's view, the auditor's tape and the firewall's commitments), which is the honest context's
(`hpub`). The honest witnesses (`hH`) and `hpub` are needed only at the firewall's own commitments (`hctxAt`,
`firewallCommits`: the inner coins live). -/
theorem vstar_outer_zk {Out Prog Wit Rv Ωa Ωp Pub : Type} {nr nh : ℕ} (rows : Prog → Fin nr → ByteArray)
    (pub : (Fin nr → ByteArray) → Rv) (honest : Out → Prog → Wit → Prop)
    (msgs : Prog → Wit → (Fin nr → ZkView.Salt) → Ωa → Ωp → (i : Fin nh) → (Fin i → ByteArray) → ByteArray)
    (V : Sessions (HCtx Prog Wit Ωa Ωp nr nh) Pub) (pubS : Out → Rv → Ωa → (Fin nh → ByteArray) → Pub)
    (hpub : ∀ o P w, honest o P w → ∀ r a p y, V.pubOf (hctxAt msgs P w r a p y) =
      pubS o (Flock.Guarantees.regLeaves rows pub P r) a (Flock.Guarantees.firewallCommits (msgs P w r a p) y))
    (hOK : V.TablesOK) {prover : TableProver} (hP : SelfCheckSilent prover)
    (hH : ∀ o P w, honest o P w → ∀ r a p y, V.Honest (hctxAt msgs P w r a p y))
    {δ : ℝ≥0∞} (hT1 : FlockSoundness.Assumptions.Hm96Hiding ZkView.My ZkView.saltHash δ) :
    ∀ o P w, honest o P w → ∀ r a p (y : Fin nh → ZkView.Salt) (E : V.View → Prop),
      prCoin (fun q : V.Ωo => E (V.view prover
          (P, w, r, a, Flock.Guarantees.firewallCommits (msgs P w r a p) y, p, y) q))
        ≤ prCoin (fun s : V.Ωs => E (V.sim (pubS o (Flock.Guarantees.regLeaves rows pub P r) a
            (Flock.Guarantees.firewallCommits (msgs P w r a p) y)) s)) + V.errAt δ := by
  intro o P w hw r a p y E
  have h := V.close hOK hT1 hP (hctxAt msgs P w r a p y) (hH o P w hw r a p y) E
  rw [hpub o P w hw r a p y] at h
  exact h

end FlockSoundness.Discharge.Recursion.OuterZK
