import Proofs.Flock.Recursive.Audit
import Proofs.Flock.Soundness.Discharge.ZkEveryCoin.Headline
import Proofs.Flock.Soundness.Discharge.ZkHidden.View
import Proofs.Flock.Soundness.Discharge.FailClosed.Mask
import Proofs.Flock.Recursive.OuterAbort

/-!
# Recursion: V*'s outer sessions are zero knowledge, (ii)'s `_hOuter`

(ii) of `RecursiveAudit` takes V*'s sessions' zero knowledge as `_hOuter`: a simulator reading the owed outputs, the
registration's view, the auditor's tape and the firewall's commitments, within `ε_o` of the auditor's view of V*'s
sessions at every honest context. Here that view, its simulator and `ε_o`.

* `Sessions`: V*'s outer sessions as (ii) sees them. The prover registers `R₂`, `n₂` hm96 commit strings under fresh
  salts (`commitStrs`), and runs `S` sessions, each one `--zk` table `C j pc l` at the public context `pc` and `R₂`'s
  commit strings `l`, at the coin vector the auditor's tape fixes (`coin`: Goldreich–Kahan's committed coins, which
  read the tape and what was public before the session), on the honest witness `wit` (which may read `R₂`'s salts).
  Each session's randomness has as many values at every `pc` and `l` (`NR`, `NS`), so the view's randomness is one
  type (`Ωo`, `Ωs`); `enc` encodes a table's output in the view.
* `TableProver`, `honestOut`, `SelfCheckSilent`: the honest prover at one table. Where it refuses the coins
  (`CoinedSetup.Admits`) it sends the prefix `Table.abort` cuts; elsewhere its code, self-check included.
  `SelfCheckSilent` is the abort interface `Recursive/OuterAbort.lean` meets: at a table that passes `CoinedOK`, coins
  the prover doesn't refuse and a valid witness whose registered values are the public ones, the code sends its view
  (the self-check never stops it there). `honestOut_eq`: then the honest prover is `outExe`.
* `proverExe`, `selfCheckSilent_exe`: the prover's code, which sends what a stopped run sends when its self-check stops
  (`Table.Stops`) and its view otherwise, meets the abort interface whatever a stopped run sends (`Table.not_stops`
  at `CoinedOK`'s inner relation, `OuterAbort`).
* `table_close`: one table at its coins, in fixed randomness types (`coin_close`, which covers refused coins with
  `Table.table_abort_hm96`).
* `Sessions.close`: the auditor's view of `R₂` and the sessions (`view`) and the simulator's (`sim`: ideal commit
  strings for `R₂`, `simAll` at each table) are within `err = Σ_j 2·|LeafPos z_j|·2^-193 + n₂·2^-193`. The sessions
  one by one at each of `R₂`'s salts (`ZK.prCoin_pi_close`), then `R₂`'s commit strings to ideal ones
  (`ZK.ideal_leaves_swap`); the sessions read `R₂`'s salts only through their witnesses, which the simulator doesn't
  read, so the second step follows the first.
* `tablesOK_of_sessions`: V*'s tables pass `CoinedOK` where each is `coinedOf` at `verify --zk`'s table of the
  session's data at a draw it sets up (`coinedOf_setup`; every `--zk` setup avoids the mask slot,
  `FailClosed.DrawSetupZK.avoidsMask`).
* `vstar_outer_zk`: `_hOuter` at `Sessions.view` and `Sessions.sim`, with `ε_o = err`; `vstar_private`: (ii)'s
  conclusion there (`RecursiveCircuitPrivateLeaves`); `vstar_private_exe`: the same at `proverExe`, with no hypothesis
  on the abort.

Left to the instance at V*'s staging: that each session's table is `coinedOf` at its data and draw (with a
`DrawSetupZK` there), its sizes fixed by V*'s statement (`hz`, `hR`, `hS`), and the honest witnesses (`Honest`, which
(ii)'s `honest` carries).
-/

namespace FlockSoundness.Discharge.Recursion.OuterZK

open Game Composed ZkView Model Flock ZkEveryCoin ZkLink
open scoped ENNReal

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-- A table's prover randomness with its hidden leaves' salts, at coin vector `c`. -/
abbrev RandAt (C : CoinedSetup) (c : CoinsZK C.Rest) : Type := (C.T c).tab.Rand × ((C.T c).tab.Hid → ZkView.Salt)

/-- The simulator's randomness with the hidden leaves' salts, at coin vector `c`. -/
abbrev SimAt (C : CoinedSetup) (c : CoinsZK C.Rest) : Type := (C.T c).tab.SimRand × ((C.T c).tab.Hid → ZkView.Salt)

/-- What a table's prover outputs at coin vector `c`: its view, or the cut it stops at. -/
abbrev OutAt (C : CoinedSetup) (c : CoinsZK C.Rest) : Type := (C.T c).tab.View ⊕ (C.T c).tab.Cut

/-! ## The honest prover at one table, and the abort interface -/

/-- **The honest prover's code at one table**: what it sends at a table, a coin vector, a witness and its randomness. -/
abbrev TableProver : Type 1 :=
  ∀ (C : CoinedSetup) (c : CoinsZK C.Rest), Witness C.S.m → RandAt C c → OutAt C c

open Classical in
/-- **The honest prover at one table**: where it refuses the coins (`CoinedSetup.Admits`, `refused_coins`), the prefix
of its view `Table.abort` cuts; elsewhere its code `prover`. -/
noncomputable def honestOut (prover : TableProver) (C : CoinedSetup) (c : CoinsZK C.Rest) (w : Witness C.S.m)
    (p : RandAt C c) : OutAt C c :=
  if C.Admits c then prover C c w p else Sum.inr ((C.T c).tab.abort (sentAt C c) ((C.T c).viewExe w p.1 p.2))

/-- **The abort interface**: at a table that passes `CoinedOK`, a coin vector the prover doesn't refuse and a valid
witness whose registered values are the public ones, the prover's code, self-check included, sends its view
(`Recursive/OuterAbort.lean`: the self-check never stops it there). -/
def SelfCheckSilent (prover : TableProver) : Prop :=
  ∀ (C : CoinedSetup) (c : CoinsZK C.Rest) (w : Witness C.S.m), CoinedOK execArith (Refine.fOf Flock.Zk.PAD_C) C →
    C.Admits c → C.Valid w → (∀ c', (C.T c').tab.sReg w = (C.T c').tab.pubReg) →
      ∀ p : RandAt C c, prover C c w p = Sum.inl ((C.T c).viewExe w p.1 p.2)

/-- **Under the abort interface, the honest prover is `outExe`.** -/
theorem honestOut_eq {prover : TableProver} (hP : SelfCheckSilent prover) {C : CoinedSetup}
    (hC : CoinedOK execArith (Refine.fOf Flock.Zk.PAD_C) C) (c : CoinsZK C.Rest) {w : Witness C.S.m}
    (hv : C.Valid w) (hreg : ∀ c', (C.T c').tab.sReg w = (C.T c').tab.pubReg) (p : RandAt C c) :
    honestOut prover C c w p = outExe C c w p.1 p.2 := by
  unfold honestOut outExe
  split_ifs with hA
  · exact hP C c w hC hA hv hreg p
  · rfl

open Classical in
/-- **The honest prover's code at one table** (`zk_veil.rs`'s `prove_inner` under `SelfCheck::Stop`): what a stopped
run sends (`stopped`, any function of the table, the coins, the witness and the randomness) if its self-check stops
(`Table.Stops`), else its view. -/
noncomputable def proverExe
    (stopped : ∀ (C : CoinedSetup) (c : CoinsZK C.Rest), Witness C.S.m → RandAt C c → (C.T c).tab.Cut) :
    TableProver := fun C c w p =>
  if (C.T c).tab.Stops w p.1 then Sum.inr (stopped C c w p) else Sum.inl ((C.T c).viewExe w p.1 p.2)

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

/-- **One table at its coins, in fixed randomness types** (`coin_close`): at any coin vector, through any encoding
`enc` of the table's output and any equivalences `eR`, `eS` of `Fin NR` and `Fin NS` with the prover's and the
simulator's randomness, the honest output and the simulator's are within `2·|LeafPos z|·2^-193`. -/
theorem table_close (C : CoinedSetup) (hC : CoinedOK execArith (Refine.fOf Flock.Zk.PAD_C) C)
    (hKey : FlockSoundness.Assumptions.Zk.HashDerivedKeyHm96) (w w0 : Witness C.S.m) (hv : C.Valid w)
    (hreg : ∀ c, (C.T c).tab.sReg w = (C.T c).tab.pubReg) (c : CoinsZK C.Rest) {z : ZkView.Sizes} (hz : C.z = z)
    {Vw : Type} (enc : ∀ c' : CoinsZK C.Rest, OutAt C c' → Vw) {NR NS : ℕ} (eR : Fin NR ≃ RandAt C c)
    (eS : Fin NS ≃ SimAt C c) (E : Vw → Prop) :
    prCoin (fun ρ : Fin NR => E (enc c (outExe C c w (eR ρ).1 (eR ρ).2)))
      ≤ prCoin (fun σ : Fin NS => E (enc c (simAll C c w0 (eS σ).1 (eS σ).2)))
        + 2 * Fintype.card (ZkView.LeafPos z) * (2 : ℝ≥0∞) ^ (-193 : ℤ) := by
  subst hz
  calc prCoin (fun ρ : Fin NR => E (enc c (outExe C c w (eR ρ).1 (eR ρ).2)))
      = prCoin (fun p : RandAt C c => E (enc c (outExe C c w p.1 p.2))) :=
        ZK.prCoin_congr_equiv eR (fun p : RandAt C c => E (enc c (outExe C c w p.1 p.2)))
    _ ≤ prCoin (fun p : SimAt C c => E (enc c (simAll C c w0 p.1 p.2)))
          + 2 * Fintype.card (ZkView.LeafPos C.z) * (2 : ℝ≥0∞) ^ (-193 : ℤ) :=
        (coin_close C hC hKey w w0 hv hreg (fun c' v => E (enc c' v)) c).1
    _ = prCoin (fun σ : Fin NS => E (enc c (simAll C c w0 (eS σ).1 (eS σ).2)))
          + 2 * Fintype.card (ZkView.LeafPos C.z) * (2 : ℝ≥0∞) ^ (-193 : ℤ) :=
        congrArg (· + 2 * (Fintype.card (ZkView.LeafPos C.z) : ℝ≥0∞) * (2 : ℝ≥0∞) ^ (-193 : ℤ))
          (ZK.prCoin_congr_equiv eS (fun p : SimAt C c => E (enc c (simAll C c w0 p.1 p.2)))).symm

/-! ## V*'s sessions -/

/-- The commit strings of `m` under the salts `y`: hm96's `(SHA-512(m) ⊕ M·y, c(y))`, a firewall leaf's two halves. -/
def commitStrs {n : ℕ} (m : Fin n → ByteArray) (y : Fin n → ZkView.Salt) : Fin n → ZkView.Bits × ByteArray :=
  fun i => (ZkView.bitsOf (Sha512.hash (m i)) + ZkView.My (y i), ZkView.saltHash (y i))

/-- Ideal commit strings: a uniform `U` beside the salt's own hash, reading no message. -/
def idealStrs {n : ℕ} (u : Fin n → ZkView.Bits × ZkView.Salt) : Fin n → ZkView.Bits × ByteArray :=
  fun i => ((u i).1, ZkView.saltHash (u i).2)

/-- **V*'s outer sessions as (ii) sees them.** `Ctx` is what the honest prover holds and `Pub` what the simulator reads
(`pubOf`). Before its sessions the prover registers `R₂`: `n₂` hm96 commit strings to `m₂` (rec-acc's running sums and
the opened rows) under fresh salts. Session `j` of V*'s `S` is one `--zk` table, `C j pc l` at the public context
`pc` and `R₂`'s commit strings `l`, at the coin vector `coin j pc l` the auditor's tape fixes, with sizes `z j` and
`NR j` (the prover's) and `NS j` (the simulator's) randomness values at every `pc` and `l`; `enc` encodes a table's
output in the view, at any coin vector. The honest witness `wit j x y₂` may read `R₂`'s salts `y₂`. -/
structure Sessions (Ctx Pub : Type) : Type 1 where
  pubOf : Ctx → Pub
  n₂ : ℕ
  m₂ : Ctx → Fin n₂ → ByteArray
  S : ℕ
  C : Fin S → Pub → (Fin n₂ → ZkView.Bits × ByteArray) → CoinedSetup
  coin : (j : Fin S) → (pc : Pub) → (l : Fin n₂ → ZkView.Bits × ByteArray) → CoinsZK (C j pc l).Rest
  z : Fin S → ZkView.Sizes
  hz : ∀ j pc l, (C j pc l).z = z j
  NR : Fin S → ℕ
  NS : Fin S → ℕ
  hR : ∀ j pc l, Fintype.card (RandAt (C j pc l) (coin j pc l)) = NR j
  hS : ∀ j pc l, Fintype.card (SimAt (C j pc l) (coin j pc l)) = NS j
  Vw : Fin S → Type
  enc : (j : Fin S) → (pc : Pub) → (l : Fin n₂ → ZkView.Bits × ByteArray) → (c : CoinsZK (C j pc l).Rest) →
    OutAt (C j pc l) c → Vw j
  wit : (j : Fin S) → (x : Ctx) → (y₂ : Fin n₂ → ZkView.Salt) →
    Witness (C j (pubOf x) (commitStrs (m₂ x) y₂)).S.m

/-- The dummy witness, zero everywhere. -/
def w0 (m : ℕ) : Witness m := fun _ => 0

namespace Sessions

variable {Ctx Pub : Type} (V : Sessions Ctx Pub)

/-- Session `j`'s prover randomness, from `Fin (NR j)`. -/
noncomputable def eR (j : Fin V.S) (pc : Pub) (l : Fin V.n₂ → ZkView.Bits × ByteArray) :
    Fin (V.NR j) ≃ RandAt (V.C j pc l) (V.coin j pc l) :=
  (Fintype.equivFinOfCardEq (V.hR j pc l)).symm

/-- Session `j`'s simulator randomness, from `Fin (NS j)`. -/
noncomputable def eS (j : Fin V.S) (pc : Pub) (l : Fin V.n₂ → ZkView.Bits × ByteArray) :
    Fin (V.NS j) ≃ SimAt (V.C j pc l) (V.coin j pc l) :=
  (Fintype.equivFinOfCardEq (V.hS j pc l)).symm

/-- The view's randomness: `R₂`'s salts, then each session's. -/
abbrev Ωo : Type := (Fin V.n₂ → ZkView.Salt) × ((j : Fin V.S) → Fin (V.NR j))

/-- The simulator's randomness: each session's, then `R₂`'s ideal seeds. -/
abbrev Ωs : Type := ((j : Fin V.S) → Fin (V.NS j)) × (Fin V.n₂ → ZkView.Bits × ZkView.Salt)

/-- The view: `R₂`'s commit strings and each session's output. -/
abbrev View : Type := (Fin V.n₂ → ZkView.Bits × ByteArray) × ((j : Fin V.S) → V.Vw j)

/-- Session `j`'s honest output at the context `x`, `R₂`'s salts `y₂` and its randomness `ρ`. -/
noncomputable def realAt (prover : TableProver) (x : Ctx) (y₂ : Fin V.n₂ → ZkView.Salt) (j : Fin V.S)
    (ρ : Fin (V.NR j)) : V.Vw j :=
  V.enc j (V.pubOf x) (commitStrs (V.m₂ x) y₂) (V.coin j (V.pubOf x) (commitStrs (V.m₂ x) y₂))
    (honestOut prover (V.C j (V.pubOf x) (commitStrs (V.m₂ x) y₂)) (V.coin j (V.pubOf x) (commitStrs (V.m₂ x) y₂))
      (V.wit j x y₂) (V.eR j (V.pubOf x) (commitStrs (V.m₂ x) y₂) ρ))

/-- Session `j`'s simulated output at the public context `pc` and `R₂`'s commit strings `l`: `simAll` on the dummy
witness. -/
noncomputable def simAt (pc : Pub) (l : Fin V.n₂ → ZkView.Bits × ByteArray) (j : Fin V.S) (σ : Fin (V.NS j)) :
    V.Vw j :=
  V.enc j pc l (V.coin j pc l) (simAll (V.C j pc l) (V.coin j pc l) (w0 _) (V.eS j pc l σ).1 (V.eS j pc l σ).2)

/-- **The auditor's view of V*'s sessions**: `R₂`'s commit strings, then each session's honest output. -/
noncomputable def view (prover : TableProver) (x : Ctx) (q : V.Ωo) : V.View :=
  (commitStrs (V.m₂ x) q.1, fun j => V.realAt prover x q.1 j (q.2 j))

/-- The sessions' simulated outputs beside the commit strings `l`. -/
noncomputable def simOut (pc : Pub) (σ : (j : Fin V.S) → Fin (V.NS j)) (l : Fin V.n₂ → ZkView.Bits × ByteArray) :
    V.View :=
  (l, fun j => V.simAt pc l j (σ j))

/-- **The simulator**: ideal commit strings for `R₂`, and each session's simulated output. It reads the public
context only. -/
noncomputable def sim (pc : Pub) (s : V.Ωs) : V.View := V.simOut pc s.1 (idealStrs s.2)

/-- **The honest context's witnesses**: at each session and each of `R₂`'s salts, valid, their registered values the
public ones at every coin vector. -/
def Honest (x : Ctx) : Prop :=
  ∀ j y₂, (V.C j (V.pubOf x) (commitStrs (V.m₂ x) y₂)).Valid (V.wit j x y₂) ∧
    ∀ c, ((V.C j (V.pubOf x) (commitStrs (V.m₂ x) y₂)).T c).tab.sReg (V.wit j x y₂) =
      ((V.C j (V.pubOf x) (commitStrs (V.m₂ x) y₂)).T c).tab.pubReg

/-- **Every session's table passes `CoinedOK`.** -/
def TablesOK : Prop := ∀ j pc l, CoinedOK execArith (Refine.fOf Flock.Zk.PAD_C) (V.C j pc l)

/-- The sessions' error, `Σ_j 2·|LeafPos z_j|·2^-193`. -/
noncomputable def errS : ℝ≥0∞ :=
  ∑ j, 2 * (Fintype.card (ZkView.LeafPos (V.z j)) : ℝ≥0∞) * (2 : ℝ≥0∞) ^ (-193 : ℤ)

/-- **V*'s sessions' `ε_o`**: the sessions' error and `R₂`'s, `n₂·2^-193`. -/
noncomputable def err : ℝ≥0∞ := V.errS + V.n₂ * (2 : ℝ≥0∞) ^ (-193 : ℤ)

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
`outExe` by the abort interface). -/
theorem session_close (hOK : V.TablesOK) (hKey : FlockSoundness.Assumptions.Zk.HashDerivedKeyHm96)
    {prover : TableProver} (hP : SelfCheckSilent prover) (x : Ctx) (hx : V.Honest x)
    (y₂ : Fin V.n₂ → ZkView.Salt) (j : Fin V.S) (E : V.Vw j → Prop) :
    prCoin (fun ρ : Fin (V.NR j) => E (V.realAt prover x y₂ j ρ))
      ≤ prCoin (fun σ : Fin (V.NS j) => E (V.simAt (V.pubOf x) (commitStrs (V.m₂ x) y₂) j σ))
        + 2 * (Fintype.card (ZkView.LeafPos (V.z j)) : ℝ≥0∞) * (2 : ℝ≥0∞) ^ (-193 : ℤ) := by
  have h := table_close (V.C j (V.pubOf x) (commitStrs (V.m₂ x) y₂)) (hOK j (V.pubOf x) (commitStrs (V.m₂ x) y₂))
    hKey (V.wit j x y₂) (w0 _)
    (hx j y₂).1 (hx j y₂).2 (V.coin j (V.pubOf x) (commitStrs (V.m₂ x) y₂)) (V.hz j _ _)
    (V.enc j (V.pubOf x) (commitStrs (V.m₂ x) y₂)) (V.eR j (V.pubOf x) (commitStrs (V.m₂ x) y₂))
    (V.eS j (V.pubOf x) (commitStrs (V.m₂ x) y₂)) E
  refine Eq.trans_le (congrArg prCoin (funext fun ρ => congrArg E
    (congrArg (V.enc j (V.pubOf x) (commitStrs (V.m₂ x) y₂) (V.coin j (V.pubOf x) (commitStrs (V.m₂ x) y₂)))
      (honestOut_eq hP (hOK j (V.pubOf x) (commitStrs (V.m₂ x) y₂)) (V.coin j (V.pubOf x) (commitStrs (V.m₂ x) y₂))
        (hx j y₂).1 (hx j y₂).2
        (V.eR j (V.pubOf x) (commitStrs (V.m₂ x) y₂) ρ))))) h

/-- **V*'s sessions are zero knowledge at an honest context**: the auditor's view of `R₂` and the sessions and the
simulator's are within `err`. At each of `R₂`'s salts the sessions go to their simulators one by one
(`ZK.prCoin_pi_close`, `session_close`); then nothing reads `R₂`'s salts but its commit strings, which go to ideal ones
(`ZK.ideal_leaves_swap`, `hash-derived-key`). -/
theorem close (hOK : V.TablesOK) (hKey : FlockSoundness.Assumptions.Zk.HashDerivedKeyHm96) {prover : TableProver}
    (hP : SelfCheckSilent prover) (x : Ctx) (hx : V.Honest x) (E : V.View → Prop) :
    prCoin (fun q : V.Ωo => E (V.view prover x q))
      ≤ prCoin (fun s : V.Ωs => E (V.sim (V.pubOf x) s)) + V.err := by
  have : ∀ j, Nonempty (Fin (V.NR j)) := fun j => ⟨⟨0, V.nr_pos j (V.pubOf x) (commitStrs (V.m₂ x) fun _ => 0)⟩⟩
  have : ∀ j, Nonempty (Fin (V.NS j)) := fun j => ⟨⟨0, V.ns_pos j (V.pubOf x) (commitStrs (V.m₂ x) fun _ => 0)⟩⟩
  have step : ∀ y₂ : Fin V.n₂ → ZkView.Salt,
      prCoin (fun ρ : (j : Fin V.S) → Fin (V.NR j) => E (V.view prover x (y₂, ρ)))
        ≤ prCoin (fun σ : (j : Fin V.S) → Fin (V.NS j) => E (V.simOut (V.pubOf x) σ (commitStrs (V.m₂ x) y₂)))
          + V.errS := fun y₂ =>
    ZK.prCoin_pi_close (Ω := fun j => Fin (V.NR j)) (Ω' := fun j => Fin (V.NS j)) (V := V.Vw)
      (fun j ρ => V.realAt prover x y₂ j ρ) (fun j σ => V.simAt (V.pubOf x) (commitStrs (V.m₂ x) y₂) j σ)
      (fun j => 2 * (Fintype.card (ZkView.LeafPos (V.z j)) : ℝ≥0∞) * (2 : ℝ≥0∞) ^ (-193 : ℤ))
      (fun j E' => V.session_close hOK hKey hP x hx y₂ j E') (fun v => E (commitStrs (V.m₂ x) y₂, v))
  have swap := ZK.ideal_leaves_swap ZkView.My ZkView.saltHash hKey (fun p : ZkView.Bits × ByteArray => p) V.n₂
    (Ω := (j : Fin V.S) → Fin (V.NS j)) (fun _ i => ZkView.bitsOf (Sha512.hash (V.m₂ x i)))
    (fun σ l => E (V.simOut (V.pubOf x) σ l))
  calc prCoin (fun q : V.Ωo => E (V.view prover x q))
      = avg fun y₂ : Fin V.n₂ → ZkView.Salt =>
          prCoin fun ρ : (j : Fin V.S) → Fin (V.NR j) => E (V.view prover x (y₂, ρ)) :=
        ZK.prCoin_prod_split (fun (y₂ : Fin V.n₂ → ZkView.Salt) (ρ : (j : Fin V.S) → Fin (V.NR j)) =>
          E (V.view prover x (y₂, ρ)))
    _ ≤ avg fun y₂ : Fin V.n₂ → ZkView.Salt =>
          prCoin (fun σ : (j : Fin V.S) → Fin (V.NS j) => E (V.simOut (V.pubOf x) σ (commitStrs (V.m₂ x) y₂)))
            + V.errS := avg_mono step
    _ = (avg fun y₂ : Fin V.n₂ → ZkView.Salt =>
          prCoin fun σ : (j : Fin V.S) → Fin (V.NS j) => E (V.simOut (V.pubOf x) σ (commitStrs (V.m₂ x) y₂)))
            + V.errS := by rw [avg_add, avg_const]
    _ = prCoin (fun p : (Fin V.n₂ → ZkView.Salt) × ((j : Fin V.S) → Fin (V.NS j)) =>
          E (V.simOut (V.pubOf x) p.2 (commitStrs (V.m₂ x) p.1))) + V.errS :=
        congrArg (· + V.errS) (ZK.prCoin_prod_split
          (fun (y₂ : Fin V.n₂ → ZkView.Salt) (σ : (j : Fin V.S) → Fin (V.NS j)) =>
            E (V.simOut (V.pubOf x) σ (commitStrs (V.m₂ x) y₂)))).symm
    _ = prCoin (fun p : ((j : Fin V.S) → Fin (V.NS j)) × (Fin V.n₂ → ZkView.Salt) =>
          E (V.simOut (V.pubOf x) p.1 (commitStrs (V.m₂ x) p.2))) + V.errS :=
        congrArg (· + V.errS) (ZK.prCoin_congr_equiv
          (Equiv.prodComm ((j : Fin V.S) → Fin (V.NS j)) (Fin V.n₂ → ZkView.Salt))
          (fun p : (Fin V.n₂ → ZkView.Salt) × ((j : Fin V.S) → Fin (V.NS j)) =>
            E (V.simOut (V.pubOf x) p.2 (commitStrs (V.m₂ x) p.1)))).symm
    _ ≤ (prCoin (fun s : V.Ωs => E (V.sim (V.pubOf x) s)) + V.n₂ * (2 : ℝ≥0∞) ^ (-193 : ℤ)) + V.errS :=
        add_le_add swap le_rfl
    _ = prCoin (fun s : V.Ωs => E (V.sim (V.pubOf x) s)) + V.err := by
        rw [Sessions.err]
        ring

end Sessions

/-! ## V*'s tables, and (ii) -/

/-- **V*'s session tables pass `CoinedOK`** where each is `coinedOf` at `verify --zk`'s table of the session's data at
a draw `verify --zk` sets up (`coinedOf_setup`); every `--zk` setup avoids the mask slot
(`FailClosed.DrawSetupZK.avoidsMask`). -/
theorem tablesOK_of_sessions {Ctx Pub : Type} (V : Sessions Ctx Pub) (P : LigProver) (Vs : Fin V.S → VStmt)
    {Reg' : Type} (D : Pub → (Fin V.n₂ → ZkView.Bits × ByteArray) → (j : Fin V.S) → (Vs j).OuterD Reg')
    (draw : (j : Fin V.S) → Pub → (Fin V.n₂ → ZkView.Bits × ByteArray) → Finset (Fin (Vs j).nv))
    (y : ∀ j pc l, DrawSetupZK (D pc l j).I (Vs j).mPts (draw j pc l) ((D pc l j).dj (draw j pc l)))
    (hC : ∀ j pc l, V.C j pc l = coinedOf P (tableAtZK (D pc l j).dj (D pc l j).y₀ (draw j pc l)) (y j pc l).x.st) :
    V.TablesOK := by
  intro j pc l
  rw [hC j pc l]
  exact (coinedOf_setup (Reg := Unit) P (D pc l j).I (D pc l j).dj (D pc l j).y₀ (D pc l j).ht
    (fun _ y' => FailClosed.DrawSetupZK.avoidsMask y') (draw j pc l) () (y j pc l)).2.2.2.2.2

/-- What the honest prover holds in (ii): the private circuit, the witness, the registration's salts, the auditor's
tape, the firewall's commitments, the inner prover's randomness and the firewall's salts. -/
abbrev HCtx (Prog Wit Ωa Ωp : Type) (nr nh : ℕ) : Type :=
  Prog × Wit × (Fin nr → ZkView.Salt) × Ωa × (Fin nh → ByteArray) × Ωp × (Fin nh → ZkView.Salt)

/-- The honest context at the firewall's commitments to `msgs` under the salts `y`. -/
abbrev hctxAt {Prog Wit Ωa Ωp : Type} {nr nh : ℕ}
    (msgs : Prog → Wit → (Fin nr → ZkView.Salt) → Ωa → Ωp → Fin nh → ByteArray) (P : Prog) (w : Wit)
    (r : Fin nr → ZkView.Salt) (a : Ωa) (p : Ωp) (y : Fin nh → ZkView.Salt) : HCtx Prog Wit Ωa Ωp nr nh :=
  (P, w, r, a, (fun i => Flock.Guarantees.firewallLeaf (msgs P w r a p i) (y i)), p, y)

/-- **V*'s sessions give (ii) its `_hOuter`**, with `ε_o = err`: the view `Sessions.view` at the honest context, the
simulator `Sessions.sim` at the public context `pubS` reads (the owed outputs, the registration's view, the auditor's
tape and the firewall's commitments), which is the honest context's (`hpub`). The honest witnesses (`hH`) and
`hpub` are needed only at the firewall's own commitments (`hctxAt`). -/
theorem vstar_outer_zk {Out Prog Wit Rv Ωa Ωp Pub : Type} {nr nh : ℕ} (rows : Prog → Fin nr → ByteArray)
    (pub : (Fin nr → ByteArray) → Rv) (honest : Out → Prog → Wit → Prop)
    (msgs : Prog → Wit → (Fin nr → ZkView.Salt) → Ωa → Ωp → Fin nh → ByteArray)
    (V : Sessions (HCtx Prog Wit Ωa Ωp nr nh) Pub) (pubS : Out → Rv → Ωa → (Fin nh → ByteArray) → Pub)
    (hpub : ∀ o P w, honest o P w → ∀ r a p y, V.pubOf (hctxAt msgs P w r a p y) =
      pubS o (Flock.Guarantees.regLeaves rows pub P r) a (fun i => Flock.Guarantees.firewallLeaf (msgs P w r a p i) (y i)))
    (hOK : V.TablesOK) {prover : TableProver} (hP : SelfCheckSilent prover)
    (hH : ∀ o P w, honest o P w → ∀ r a p y, V.Honest (hctxAt msgs P w r a p y))
    (hKey : FlockSoundness.Assumptions.Zk.HashDerivedKeyHm96) :
    ∀ o P w, honest o P w → ∀ r a p (y : Fin nh → ZkView.Salt) (E : V.View → Prop),
      prCoin (fun q : V.Ωo => E (V.view prover
          (P, w, r, a, (fun i => Flock.Guarantees.firewallLeaf (msgs P w r a p i) (y i)), p, y) q))
        ≤ prCoin (fun s : V.Ωs => E (V.sim (pubS o (Flock.Guarantees.regLeaves rows pub P r) a
            (fun i => Flock.Guarantees.firewallLeaf (msgs P w r a p i) (y i))) s)) + V.err := by
  intro o P w hw r a p y E
  have h := V.close hOK hKey hP (hctxAt msgs P w r a p y) (hH o P w hw r a p y) E
  rw [hpub o P w hw r a p y] at h
  exact h

/-- **(ii) at V*'s sessions**: `RecursiveCircuitPrivateLeaves` with `_hOuter` from `vstar_outer_zk`. For every honest
witness of every private circuit of the class and every private circuit `p₀` of it, the auditor's whole view and the
simulated one give every event probabilities within `2 n_r·2^-193 + err + n_h·2^-193`, both ways. -/
theorem vstar_private {Out Prog Wit Rv Ωa Ωp Pub : Type} [Fintype Ωa] [Nonempty Ωa] [Fintype Ωp] [Nonempty Ωp]
    {nr nh : ℕ} (rows : Prog → Fin nr → ByteArray) (pub : (Fin nr → ByteArray) → Rv)
    (honest : Out → Prog → Wit → Prop) (msgs : Prog → Wit → (Fin nr → ZkView.Salt) → Ωa → Ωp → Fin nh → ByteArray)
    (V : Sessions (HCtx Prog Wit Ωa Ωp nr nh) Pub) (pubS : Out → Rv → Ωa → (Fin nh → ByteArray) → Pub)
    (hpub : ∀ o P w, honest o P w → ∀ r a p y, V.pubOf (hctxAt msgs P w r a p y) =
      pubS o (Flock.Guarantees.regLeaves rows pub P r) a (fun i => Flock.Guarantees.firewallLeaf (msgs P w r a p i) (y i)))
    (hOK : V.TablesOK) {prover : TableProver} (hP : SelfCheckSilent prover)
    (hH : ∀ o P w, honest o P w → ∀ r a p y, V.Honest (hctxAt msgs P w r a p y))
    (hKey : FlockSoundness.Assumptions.Zk.HashDerivedKeyHm96)
    (p₀ : Prog) (o : Out) (P : Prog) (w : Wit) (hw : honest o P w)
    (E : Rv × Ωa × (Fin nh → ByteArray) × V.View → Prop) :
    Flock.Guarantees.recRealPr (Flock.Guarantees.regLeaves rows pub) nh msgs
        (fun P w r a h p y q => V.view prover (P, w, r, a, h, p, y) q) P w E
      ≤ Flock.Guarantees.recSimPr (Flock.Guarantees.regLeaves rows pub) nh (fun o v a h s => V.sim (pubS o v a h) s)
          o p₀ E + (2 * nr * (2 : ℝ≥0∞) ^ (-193 : ℤ) + V.err + nh * (2 : ℝ≥0∞) ^ (-193 : ℤ)) ∧
    Flock.Guarantees.recSimPr (Flock.Guarantees.regLeaves rows pub) nh (fun o v a h s => V.sim (pubS o v a h) s) o p₀ E
      ≤ Flock.Guarantees.recRealPr (Flock.Guarantees.regLeaves rows pub) nh msgs
          (fun P w r a h p y q => V.view prover (P, w, r, a, h, p, y) q) P w E
        + (2 * nr * (2 : ℝ≥0∞) ^ (-193 : ℤ) + V.err + nh * (2 : ℝ≥0∞) ^ (-193 : ℤ)) := by
  let x₀ : HCtx Prog Wit Ωa Ωp nr nh := (P, w, fun _ => 0, Classical.arbitrary Ωa, fun _ => ByteArray.empty,
    Classical.arbitrary Ωp, fun _ => 0)
  have : ∀ j, Nonempty (Fin (V.NR j)) := fun j => ⟨⟨0, V.nr_pos j (V.pubOf x₀) (commitStrs (V.m₂ x₀) fun _ => 0)⟩⟩
  have : ∀ j, Nonempty (Fin (V.NS j)) := fun j => ⟨⟨0, V.ns_pos j (V.pubOf x₀) (commitStrs (V.m₂ x₀) fun _ => 0)⟩⟩
  exact Flock.SecurityProofs.RecursiveCircuitPrivateLeaves nr rows pub honest nh msgs
    (fun P w r a h p y q => V.view prover (P, w, r, a, h, p, y) q) (fun o v a h s => V.sim (pubS o v a h) s)
    (vstar_outer_zk rows pub honest msgs V pubS hpub hOK hP hH hKey) hKey p₀ o P w hw E

/-- **(ii) at V*'s sessions, the honest prover's code in them** (`vstar_private` at `proverExe`, `selfCheckSilent_exe`):
whatever a stopped run sends, the auditor's whole view and the simulated one give every event probabilities within
`2 n_r·2^-193 + err + n_h·2^-193`, both ways, with no hypothesis on the self-check. -/
theorem vstar_private_exe {Out Prog Wit Rv Ωa Ωp Pub : Type} [Fintype Ωa] [Nonempty Ωa] [Fintype Ωp] [Nonempty Ωp]
    {nr nh : ℕ} (rows : Prog → Fin nr → ByteArray) (pub : (Fin nr → ByteArray) → Rv)
    (honest : Out → Prog → Wit → Prop) (msgs : Prog → Wit → (Fin nr → ZkView.Salt) → Ωa → Ωp → Fin nh → ByteArray)
    (V : Sessions (HCtx Prog Wit Ωa Ωp nr nh) Pub) (pubS : Out → Rv → Ωa → (Fin nh → ByteArray) → Pub)
    (hpub : ∀ o P w, honest o P w → ∀ r a p y, V.pubOf (hctxAt msgs P w r a p y) =
      pubS o (Flock.Guarantees.regLeaves rows pub P r) a
        (fun i => Flock.Guarantees.firewallLeaf (msgs P w r a p i) (y i)))
    (hOK : V.TablesOK)
    (stopped : ∀ (C : CoinedSetup) (c : CoinsZK C.Rest), Witness C.S.m → RandAt C c → (C.T c).tab.Cut)
    (hH : ∀ o P w, honest o P w → ∀ r a p y, V.Honest (hctxAt msgs P w r a p y))
    (hKey : FlockSoundness.Assumptions.Zk.HashDerivedKeyHm96)
    (p₀ : Prog) (o : Out) (P : Prog) (w : Wit) (hw : honest o P w)
    (E : Rv × Ωa × (Fin nh → ByteArray) × V.View → Prop) :
    Flock.Guarantees.recRealPr (Flock.Guarantees.regLeaves rows pub) nh msgs
        (fun P w r a h p y q => V.view (proverExe stopped) (P, w, r, a, h, p, y) q) P w E
      ≤ Flock.Guarantees.recSimPr (Flock.Guarantees.regLeaves rows pub) nh (fun o v a h s => V.sim (pubS o v a h) s)
          o p₀ E + (2 * nr * (2 : ℝ≥0∞) ^ (-193 : ℤ) + V.err + nh * (2 : ℝ≥0∞) ^ (-193 : ℤ)) ∧
    Flock.Guarantees.recSimPr (Flock.Guarantees.regLeaves rows pub) nh (fun o v a h s => V.sim (pubS o v a h) s) o p₀ E
      ≤ Flock.Guarantees.recRealPr (Flock.Guarantees.regLeaves rows pub) nh msgs
          (fun P w r a h p y q => V.view (proverExe stopped) (P, w, r, a, h, p, y) q) P w E
        + (2 * nr * (2 : ℝ≥0∞) ^ (-193 : ℤ) + V.err + nh * (2 : ℝ≥0∞) ^ (-193 : ℤ)) :=
  vstar_private rows pub honest msgs V pubS hpub hOK (selfCheckSilent_exe stopped) hH hKey p₀ o P w hw E

end FlockSoundness.Discharge.Recursion.OuterZK
