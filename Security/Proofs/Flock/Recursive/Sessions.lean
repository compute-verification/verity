import Proofs.Flock.Recursive.Statements
import Proofs.Flock.Soundness.Discharge.ZkEveryCoin.Headline
import Proofs.Flock.Soundness.Discharge.ZkHidden.View
import Proofs.Flock.Recursive.OuterAbort

/-!
# Recursion: V*'s outer sessions as (ii) sees them

The definitions (ii) of `RecursiveAudit` reads of V*'s outer sessions, below the property so that it can state (ii) at
them (`Proofs.Flock.Recursive.Firewall`); their zero knowledge is `Proofs.Flock.Recursive.OuterZK`'s.

* `TableProver`, `honestOut`, `SelfCheckSilent`: the honest prover at one table. Where it refuses the coins
  (`CoinedSetup.Admits`) it sends the prefix `Table.abort` cuts; elsewhere its code, self-check included.
  `SelfCheckSilent` is the abort interface `Recursive/OuterAbort.lean` meets.
* `proverExe`: the prover's code, which sends what a stopped run sends (`Stopped`) when its self-check stops
  (`Table.Stops`) and its view otherwise.
* `commitStrs`, `idealStrs`: hm96 commit strings as their two halves, and ideal ones.
* `Sessions`: V*'s outer sessions. The prover registers `R₂`, `n₂` hm96 commit strings under fresh salts
  (`commitStrs`), and runs `S` sessions, each one `--zk` table `C j pc l` at the public context `pc` and `R₂`'s
  commit strings `l`, at the coin vector the auditor's tape fixes (`coin`), on the honest witness `wit`. Its view
  (`view`), its simulator (`sim`), the honest witnesses (`Honest`), every table passing `CoinedOK` (`TablesOK`) and
  its error (`err`, and `errAt` at an hm96 hiding bound `δ`).
* `HCtx`, `hctxAt`: what the honest prover holds in (ii), at the firewall's commitments (`firewallCommits`: the inner
  coins live, each message reading the commit strings before it).
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

/-- **What a stopped run sends** at a table, a coin vector, a witness and its randomness: a prefix the table cuts. -/
abbrev Stopped : Type 1 :=
  ∀ (C : CoinedSetup) (c : CoinsZK C.Rest), Witness C.S.m → RandAt C c → (C.T c).tab.Cut

open Classical in
/-- **The honest prover's code at one table** (`zk_veil.rs`'s `prove_inner` under `SelfCheck::Stop`): what a stopped
run sends (`stopped`, any function of the table, the coins, the witness and the randomness) if its self-check stops
(`Table.Stops`), else its view. -/
noncomputable def proverExe (stopped : Stopped) : TableProver := fun C c w p =>
  if (C.T c).tab.Stops w p.1 then Sum.inr (stopped C c w p) else Sum.inl ((C.T c).viewExe w p.1 p.2)

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
  /-- Obligation (V*'s statement fixes each session's sizes): session `j`'s table has the sizes `z j` at every public
  context and commit strings. -/
  hz : ∀ j pc l, (C j pc l).z = z j
  NR : Fin S → ℕ
  NS : Fin S → ℕ
  /-- Obligation (V*'s statement fixes each session's randomness): session `j`'s prover draws from `NR j` values at
  every public context and commit strings. -/
  hR : ∀ j pc l, Fintype.card (RandAt (C j pc l) (coin j pc l)) = NR j
  /-- Obligation (the same of its simulator): session `j`'s simulator draws from `NS j` values at every public context
  and commit strings. -/
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

/-- The sessions' error at an hm96 hiding bound `δ`, `Σ_j 2·|LeafPos z_j|·δ`. -/
noncomputable def errSAt (δ : ℝ≥0∞) : ℝ≥0∞ :=
  ∑ j, 2 * (Fintype.card (ZkView.LeafPos (V.z j)) : ℝ≥0∞) * δ

/-- **V*'s sessions' `ε_o` at an hm96 hiding bound `δ`**: the sessions' error and `R₂`'s, `n₂·δ`; `err` is it at
`2^-193`. -/
noncomputable def errAt (δ : ℝ≥0∞) : ℝ≥0∞ := V.errSAt δ + V.n₂ * δ

end Sessions

/-! ## The honest context -/

/-- What the honest prover holds in (ii): the private circuit, the witness, the registration's salts, the auditor's
tape, the firewall's commitments, the inner prover's randomness and the firewall's salts. -/
abbrev HCtx (Prog Wit Ωa Ωp : Type) (nr nh : ℕ) : Type :=
  Prog × Wit × (Fin nr → ZkView.Salt) × Ωa × (Fin nh → ByteArray) × Ωp × (Fin nh → ZkView.Salt)

/-- The honest context at the firewall's commitments to `msgs` under the salts `y`, the inner coins live: message `i`
reads the commit strings before it (`Flock.Guarantees.firewallCommits`). -/
abbrev hctxAt {Prog Wit Ωa Ωp : Type} {nr nh : ℕ}
    (msgs : Prog → Wit → (Fin nr → ZkView.Salt) → Ωa → Ωp → (i : Fin nh) → (Fin i → ByteArray) → ByteArray)
    (P : Prog) (w : Wit) (r : Fin nr → ZkView.Salt) (a : Ωa) (p : Ωp) (y : Fin nh → ZkView.Salt) :
    HCtx Prog Wit Ωa Ωp nr nh :=
  (P, w, r, a, Flock.Guarantees.firewallCommits (msgs P w r a p) y, p, y)

end FlockSoundness.Discharge.Recursion.OuterZK
