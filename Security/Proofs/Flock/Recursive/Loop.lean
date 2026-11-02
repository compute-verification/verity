import Proofs.Flock.Recursive.OuterSessionZK
import Proofs.Flock.Claims.Sampling.Seq
import Proofs.Flock.Claims.Sampling.Draws
import Proofs.Flock.Claims.ZK.Blind
import Proofs.Flock.SessionLoop.FinishParts

/-!
# Recursion: the committed-coin statements' `_hOuter` at V*'s statements run by the session loop

`OuterSessionZK` gives `Flock.Guarantees.OuterSessionsZK` at V*'s sessions as `Sessions` models them. Here V*'s
statements are run by the session loop: at the public context, statement `j` of `m` is one `--zk` session at the loop's
own frames, link and `Finish` (`loopRef`, `SessionLoopZKDeployed`'s session), run by its switched prover at the workers'
values `vs`, one after another, each on its own tape.

* `LoopSessions`: the public context `pubOf`, the `m` statements `stmt j` at it and their witnesses `wit j`.
* `LoopSessions.zkmalAt_prover`: the `m` statements in turn are `ZKmalAt` at the sum of their errors and `m` times their
  fork factor (`zkmalAt_layout` with no first part), the simulator drawing coins `CB j` for statement `j`, each in
  bijection with the statement's simulator's at every public context (`LoopSessions.Sized`, as `Sessions.hS`).
* `outerSessionsZK_of`: `OuterSessionsZK` at those statements, with `ε_o` the supremum over contexts of their errors and
  the composed prover's fork term, on V*'s staging (`LoopSessions.Public`, `LoopSessions.Honest`).
* `outerSessionsZK_loop`: the same at `Flock.zk_deployZk_sw`'s bound for the loop's session, under
  `SessionLoopZKDeployed`'s premises on the deployed tables, which take `SessionsCoined`'s role.
-/

namespace FlockSoundness.Discharge.Recursion.OuterZK

open Claims
open scoped ENNReal

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-- **A `Claims.Pr` bound read in `prCoin`**: `Pr p ≤ Pr q + e` gives `prCoin p ≤ prCoin q + ofReal e`. -/
theorem prCoin_le_ofReal_of_Pr {α β : Type} [Fintype α] [Fintype β] {p : α → Prop} {q : β → Prop} {e : ℝ}
    (h : Claims.Pr p ≤ Claims.Pr q + e) :
    Verity.Game.prCoin p ≤ Verity.Game.prCoin q + ENNReal.ofReal e := by
  rw [Claims.Sampling.Pr_eq_toReal_prCoin, Claims.Sampling.Pr_eq_toReal_prCoin] at h
  have hp : Verity.Game.prCoin p ≠ ⊤ := ne_top_of_le_ne_top ENNReal.one_ne_top (ZkEveryCoin.prCoin_le_one _)
  have hq : Verity.Game.prCoin q ≠ ⊤ := ne_top_of_le_ne_top ENNReal.one_ne_top (ZkEveryCoin.prCoin_le_one _)
  calc Verity.Game.prCoin p = ENNReal.ofReal (Verity.Game.prCoin p).toReal := (ENNReal.ofReal_toReal hp).symm
    _ ≤ ENNReal.ofReal ((Verity.Game.prCoin q).toReal + e) := ENNReal.ofReal_le_ofReal h
    _ ≤ ENNReal.ofReal (Verity.Game.prCoin q).toReal + ENNReal.ofReal e := ENNReal.ofReal_add_le
    _ = Verity.Game.prCoin q + ENNReal.ofReal e := by rw [ENNReal.ofReal_toReal hq]

/-- **V*'s statements run by the session loop.** `Ctx` is what the honest prover holds and `Pub` what the simulator
reads (`pubOf`). At the public context `pc` there are `m` statements, `stmt j pc` for `j < m`, each one `--zk` session,
run in turn; `wit j x` is statement `j`'s witness at the honest context `x`. -/
structure LoopSessions (Ctx Pub WS : Type) where
  pubOf : Ctx → Pub
  m : ℕ
  stmt : ℕ → Pub → Claims.Msg
  wit : ℕ → Ctx → WS

namespace LoopSessions

/-- The simulator's coins for the first `n` statements: one draw of `CB j` for each statement `j < n`, in turn. -/
def SimCoins (CB : ℕ → Type) : ℕ → Type
  | 0 => Unit
  | n + 1 => SimCoins CB n × CB n

instance instFintypeSimCoins (CB : ℕ → Type) [∀ j, Fintype (CB j)] : ∀ n, Fintype (SimCoins CB n)
  | 0 => inferInstanceAs (Fintype Unit)
  | n + 1 => letI := instFintypeSimCoins CB n; inferInstanceAs (Fintype (SimCoins CB n × CB n))

instance instNonemptySimCoins (CB : ℕ → Type) [∀ j, Nonempty (CB j)] : ∀ n, Nonempty (SimCoins CB n)
  | 0 => inferInstanceAs (Nonempty Unit)
  | n + 1 => letI := instNonemptySimCoins CB n; inferInstanceAs (Nonempty (SimCoins CB n × CB n))

/-- The simulator of no statement: no coins and no messages. -/
def nilSim : Claims.Simulator where
  C _ := Unit
  prog _ _ := Claims.ZKMal.spRun (fun _ => .ret []) 0 []

/-- **`iterSimG`'s coins after `nilSim` are `SimCoins`**: one draw of `CB j` for each session `j < n`, in turn. -/
noncomputable def simCoinsEquiv {WA WB : Type} (T₀ : Claims.ZKMal.SeqTape) (c : ℕ) (NB ΩB : Type) [Fintype NB]
    [Nonempty NB] [Fintype ΩB] [Nonempty ΩB] (Q : (n : ℕ) → Claims.ZKMal.Seq WA WB (T₀.at c NB ΩB n))
    (CB : ℕ → Type) [∀ j, Fintype (CB j)] [∀ j, Nonempty (CB j)]
    (progB : (n : ℕ) → (x : Claims.Msg) → (Q n).U x → CB n → Claims.SP (List (Claims.Msg × Claims.Out)))
    (go : ℕ → Claims.Msg → List (Claims.Msg × Claims.Out) → Bool) (x : Claims.Msg) :
    (n : ℕ) → SimCoins CB n ≃ (Claims.ZKMal.iterSimG T₀ c NB ΩB Q nilSim (fun j _ => CB j) progB go n).C x
  | 0 => Equiv.refl Unit
  | n + 1 => Equiv.prodCongr (simCoinsEquiv T₀ c NB ΩB Q CB progB go x n) (Equiv.refl (CB n))

section

variable {Ctx Pub WS : Type} (V : LoopSessions Ctx Pub WS)

/-- The prover of no statement: no messages. -/
def nilP : Ctx → Claims.Prover (Unit × Unit) := fun _ => ⟨fun _ _ _ => .ret []⟩

/-- **The statements as a layout**: no first part, then statement `j` at the public context `pc`, for `kS (stmt j pc)`
rounds, with the witness `wit j` of the honest context. -/
def lay (kS : Claims.Msg → ℕ) (pc : Pub) : Claims.ZKMal.Layout Ctx WS (Unit × Unit) where
  k₀ _ := 0
  n _ j := kS (V.stmt j pc)
  xs _ j _ := V.stmt j pc
  ws j x _ _ _ := V.wit j x

/-- The rounds of the `m` statements in turn. -/
def rounds (kS : Claims.Msg → ℕ) (pc : Pub) : ℕ := (V.lay kS pc).start [] V.m

/-- **The statements' prover**: `PS` at each statement in turn, at the honest context's witness, each on its own
tape. -/
noncomputable def prover {NB ΩB : Type} [Fintype NB] [Nonempty NB] [Fintype ΩB] [Nonempty ΩB]
    (kS : Claims.Msg → ℕ) (PS : WS → Claims.Prover (NB × ΩB)) (pc : Pub) (x : Ctx) :
    Claims.Prover (Claims.Sampling.unitTape.at 0 NB ΩB V.m) :=
  ((V.lay kS pc).prover nilP PS V.m x).comap (Claims.ZKMal.Layout.flatten Claims.Sampling.unitTape 0 NB ΩB V.m)

/-- **The statements' simulator** at the public context `pc`: statement `j`'s simulator `Sim` on the coins `CB j`, in
turn, going on where the transcript before it does. -/
noncomputable def sim (NB ΩB : Type) [Fintype NB] [Nonempty NB] [Fintype ΩB] [Nonempty ΩB] (kS : Claims.Msg → ℕ)
    (Sim : Claims.Simulator) (CB : ℕ → Type) [∀ j, Fintype (CB j)] [∀ j, Nonempty (CB j)] (pc : Pub) :
    Claims.Simulator :=
  Claims.ZKMal.iterSimG Claims.Sampling.unitTape 0 NB ΩB
    (Claims.ZKMal.Layout.seq Claims.Sampling.unitTape 0 NB ΩB (V.lay kS pc)) nilSim (fun j _ => CB j)
    (fun j _ _ => Claims.ZKMal.progOf Sim (CB j) (V.stmt j pc))
    (fun j x => Claims.ZKMal.goesOn ((V.lay kS pc).start x j)) V.m

/-- The statements' simulator draws `SimCoins CB m` at every public context. -/
noncomputable def coinsE (NB ΩB : Type) [Fintype NB] [Nonempty NB] [Fintype ΩB] [Nonempty ΩB]
    (kS : Claims.Msg → ℕ) (Sim : Claims.Simulator) (CB : ℕ → Type) [∀ j, Fintype (CB j)] [∀ j, Nonempty (CB j)]
    (pc : Pub) : SimCoins CB V.m ≃ (V.sim NB ΩB kS Sim CB pc).C [] :=
  simCoinsEquiv Claims.Sampling.unitTape 0 NB ΩB
    (Claims.ZKMal.Layout.seq Claims.Sampling.unitTape 0 NB ΩB (V.lay kS pc)) CB
    (fun j _ _ => Claims.ZKMal.progOf Sim (CB j) (V.stmt j pc))
    (fun j x => Claims.ZKMal.goesOn ((V.lay kS pc).start x j)) [] V.m

/-- **The auditor's view of the statements** at the honest context `x`: the statements' prover against `V*` at its
tape `θ`, for the statements' rounds. -/
noncomputable def viewAt {NB ΩB Θ : Type} [Fintype NB] [Nonempty NB] [Fintype ΩB] [Nonempty ΩB]
    (H : Claims.Msg → Claims.Msg) (kS : Claims.Msg → ℕ) (PS : WS → Claims.Prover (NB × ΩB)) (Vs : Claims.VStar Θ)
    (x : Ctx) (θ : Θ) (q : Claims.Sampling.unitTape.at 0 NB ΩB V.m) : List (Claims.Msg × Claims.Out) :=
  (Claims.runStar H (V.prover kS PS (V.pubOf x) x) (Vs.head fun _ => V.rounds kS (V.pubOf x)) [] q θ).msgs

/-- **The simulated view**: the statements' simulator at the public context `pc`, against `V*` at its tape `θ`. -/
noncomputable def simAt {Θ : Type} (NB ΩB : Type) [Fintype NB] [Nonempty NB] [Fintype ΩB] [Nonempty ΩB]
    (H : Claims.Msg → Claims.Msg) (kS : Claims.Msg → ℕ) (Sim : Claims.Simulator) (CB : ℕ → Type)
    [∀ j, Fintype (CB j)] [∀ j, Nonempty (CB j)] (Vs : Claims.VStar Θ) (pc : Pub) (θ : Θ) (s : SimCoins CB V.m) :
    List (Claims.Msg × Claims.Out) :=
  (V.sim NB ΩB kS Sim CB pc).out H (Vs.head fun _ => V.rounds kS pc) θ [] (V.coinsE NB ΩB kS Sim CB pc s)

/-- **The statements' error**: the supremum over contexts of the statements' errors at their public context and the
composed prover's fork term at `m` times the statements' factor `g`, against `V*` at the tape `θ x`. -/
noncomputable def err {NB ΩB Θ : Type} [Fintype NB] [Nonempty NB] [Fintype ΩB] [Nonempty ΩB]
    (H : Claims.Msg → Claims.Msg) (kS : Claims.Msg → ℕ) (PS : WS → Claims.Prover (NB × ΩB)) (εS : Claims.Msg → ℝ)
    (g : ℝ) (Vs : Claims.VStar Θ) (θ : Ctx → Θ) : ℝ≥0∞ :=
  ⨆ x, ENNReal.ofReal (∑ j ∈ Finset.range V.m, εS (V.stmt j (V.pubOf x)) +
    V.m * g * (Claims.Fork H (V.prover kS PS (V.pubOf x) x) (Vs.head fun _ => V.rounds kS (V.pubOf x)) (θ x)
      []).Pr H)

/-- Obligation (V*'s statements fix each one's simulator randomness, as `Sessions.hS`): statement `j`'s simulator
draws from as many values as `CB j` has, at every public context. -/
def Sized (Sim : Claims.Simulator) (CB : ℕ → Type) : Prop :=
  ∀ j < V.m, ∀ pc, Nonempty (CB j ≃ Sim.C (V.stmt j pc))

/-- **The statements in turn are zero knowledge at one public context**: where each statement's sessions are
`ZKmalR` at `PS` and `Sim` with errors `εS` and `g`, and statement `j`'s simulator coins are in bijection with `CB j`,
the `m` statements in turn are `ZKmalAt` at the sum of their errors and `m·g` (`zkmalAt_layout`, with no first
part). -/
theorem zkmalAt_prover {NB ΩB : Type} [Fintype NB] [Nonempty NB] [Fintype ΩB] [Nonempty ΩB]
    (H : Claims.Msg → Claims.Msg) (kS : Claims.Msg → ℕ) (PS : WS → Claims.Prover (NB × ΩB)) (R : Claims.Rel WS)
    (Sim : Claims.Simulator) (εS : Claims.Msg → ℝ) (g : ℝ) (CB : ℕ → Type) [∀ j, Fintype (CB j)]
    [∀ j, Nonempty (CB j)] (hZK : Claims.ZKmalR H kS PS R Sim εS fun _ => g) (hε : ∀ x, 0 ≤ εS x) (hg : 0 ≤ g)
    (pc : Pub) (hS : ∀ j < V.m, Nonempty (CB j ≃ Sim.C (V.stmt j pc))) :
    Claims.ZKMal.ZKmalAt H (V.rounds kS pc) (V.prover kS PS pc)
      (fun x => ∀ j < V.m, R (V.stmt j pc) (V.wit j x)) [] ((V.sim NB ΩB kS Sim CB pc).C [])
      ((V.sim NB ΩB kS Sim CB pc).prog []) (∑ j ∈ Finset.range V.m, εS (V.stmt j pc)) (V.m * g) := by
  have hB : ∀ j < V.m, Claims.ZKMal.ZKmalAt H (kS (V.stmt j pc)) PS (R (V.stmt j pc)) (V.stmt j pc) (CB j)
      (Claims.ZKMal.progOf Sim (CB j) (V.stmt j pc)) (εS (V.stmt j pc)) g := fun j hj => by
    rw [Claims.ZKMal.progOf_of (hS j hj)]
    exact hZK.at_equiv (V.stmt j pc) (hS j hj).some
  have h := Claims.ZKMal.zkmalAt_layout Claims.Sampling.unitTape 0 (V.lay kS pc) H [] V.m nilP PS
    (fun _ x => ∀ j < V.m, R (V.stmt j pc) (V.wit j x)) R nilSim (fun j _ => CB j)
    (fun j _ _ => Claims.ZKMal.progOf Sim (CB j) (V.stmt j pc))
    (fun j x => Claims.ZKMal.goesOn ((V.lay kS pc).start x j)) 0 0 g (fun j => εS (V.stmt j pc))
    (Claims.ZKMal.zkmalAt_public H 0 nilP _ [] (fun _ => .ret []) fun _ _ _ _ _ => rfl)
    (fun j hj _ _ => hB j hj)
    (fun j hj => Claims.ZKMal.Layout.relOnAt_of Claims.Sampling.unitTape 0 NB ΩB (V.lay kS pc) H _ _ R []
      fun _ hw _ _ _ => hw j hj)
    (fun _ _ _ _ _ _ _ _ => rfl) (by simp) (fun j _ => hε _) hg
  simp only [Nat.cast_zero, zero_add] at h
  exact h

/-- **The statements' view at one honest context**, in `prCoin`: every event of the auditor's view of the statements
at the honest context `x` is at most as likely as at the simulator at `x`'s public context, plus the statements'
error. -/
theorem view_le {NB ΩB Θ : Type} [Fintype NB] [Nonempty NB] [Fintype ΩB] [Nonempty ΩB]
    (H : Claims.Msg → Claims.Msg) (kS : Claims.Msg → ℕ) (PS : WS → Claims.Prover (NB × ΩB)) (R : Claims.Rel WS)
    (Sim : Claims.Simulator) (εS : Claims.Msg → ℝ) (g : ℝ) (CB : ℕ → Type) [∀ j, Fintype (CB j)]
    [∀ j, Nonempty (CB j)] (hZK : Claims.ZKmalR H kS PS R Sim εS fun _ => g) (hε : ∀ x, 0 ≤ εS x) (hg : 0 ≤ g)
    (Vs : Claims.VStar Θ) (θ : Ctx → Θ) (x : Ctx) (pc : Pub) (hpc : V.pubOf x = pc)
    (hS : ∀ j < V.m, Nonempty (CB j ≃ Sim.C (V.stmt j pc))) (hR : ∀ j < V.m, R (V.stmt j (V.pubOf x)) (V.wit j x))
    (E : List (Claims.Msg × Claims.Out) → Prop) :
    Verity.Game.prCoin (fun q => E (V.viewAt H kS PS Vs x (θ x) q)) ≤
      Verity.Game.prCoin (fun s => E (V.simAt NB ΩB H kS Sim CB Vs pc (θ x) s)) + V.err H kS PS εS g Vs θ := by
  subst hpc
  have hz := V.zkmalAt_prover H kS PS R Sim εS g CB hZK hε hg (V.pubOf x) hS x hR Θ
    (Vs.head fun _ => V.rounds kS (V.pubOf x)) (θ x) rfl E
  have hsim : Claims.Pr (fun s => E (V.simAt NB ΩB H kS Sim CB Vs (V.pubOf x) (θ x) s)) =
      Claims.Pr (fun c => E (Claims.ZKMal.spOut H (Vs.head fun _ => V.rounds kS (V.pubOf x)) (θ x) []
        ((V.sim NB ΩB kS Sim CB (V.pubOf x)).prog [] c))) :=
    Claims.ZKMal.Pr_equiv (V.coinsE NB ΩB kS Sim CB (V.pubOf x)) fun c =>
      E (Claims.ZKMal.spOut H (Vs.head fun _ => V.rounds kS (V.pubOf x)) (θ x) []
        ((V.sim NB ΩB kS Sim CB (V.pubOf x)).prog [] c))
  have hle : Claims.Pr (fun q => E (V.viewAt H kS PS Vs x (θ x) q)) ≤
      Claims.Pr (fun s => E (V.simAt NB ΩB H kS Sim CB Vs (V.pubOf x) (θ x) s)) +
        (∑ j ∈ Finset.range V.m, εS (V.stmt j (V.pubOf x)) +
          V.m * g * (Claims.Fork H (V.prover kS PS (V.pubOf x) x) (Vs.head fun _ => V.rounds kS (V.pubOf x)) (θ x)
            []).Pr H) := by
    rw [hsim]
    exact sub_le_iff_le_add'.1 (abs_sub_le_iff.1 hz).1
  exact (prCoin_le_ofReal_of_Pr hle).trans (add_le_add le_rfl (le_iSup_of_le x le_rfl))

end

/-- Obligation (the simulator's public context is the honest one's, as `SessionsPublic`): at every honest witness,
registration, tape, inner randomness and salts, the statements' public context at the honest context is `pubS` of the
owed outputs, the registration's view (`regLeaves`), the tape and the firewall's commitments. -/
def Public {Ow Prog Wit Rv Ωa Ωp Pub WS : Type} {nr nh : ℕ} (V : LoopSessions (HCtx Prog Wit Ωa Ωp nr nh) Pub WS)
    (rows : Prog → Fin nr → ByteArray) (pub : (Fin nr → ByteArray) → Rv) (honest : Ow → Prog → Wit → Prop)
    (msgs : Prog → Wit → (Fin nr → ZkView.Salt) → Ωa → Ωp → (i : Fin nh) → (Fin i → ByteArray) → ByteArray)
    (pubS : Ow → Rv → Ωa → (Fin nh → ByteArray) → Pub) : Prop :=
  ∀ o P w, honest o P w → ∀ r a p y, V.pubOf (hctxAt msgs P w r a p y) =
    pubS o (_root_.Flock.Guarantees.regLeaves rows pub P r) a
      (_root_.Flock.Guarantees.firewallCommits (msgs P w r a p) y)

/-- Obligation (the honest witnesses are valid, as `SessionsHonest`): at every honest witness, registration, tape,
inner randomness and salts, statement `j < m`'s witness at the honest context is in `R` at its statement. -/
def Honest {Ow Prog Wit Ωa Ωp Pub WS : Type} {nr nh : ℕ} (V : LoopSessions (HCtx Prog Wit Ωa Ωp nr nh) Pub WS)
    (R : Claims.Rel WS) (honest : Ow → Prog → Wit → Prop)
    (msgs : Prog → Wit → (Fin nr → ZkView.Salt) → Ωa → Ωp → (i : Fin nh) → (Fin i → ByteArray) → ByteArray) :
    Prop :=
  ∀ o P w, honest o P w → ∀ r a p y, ∀ j < V.m,
    R (V.stmt j (V.pubOf (hctxAt msgs P w r a p y))) (V.wit j (hctxAt msgs P w r a p y))

end LoopSessions

/-- **V*'s statements run in turn give the committed-coin statements their `_hOuter`** (`OuterSessionsZK`), at the
registration by hm96 leaves of the private circuit's rows (`regLeaves rows pub`): the view is the auditor's of the
statements run by `PS` at the honest context (`LoopSessions.viewAt`), the simulator the statements' at the public
context `pubS` reads (`LoopSessions.simAt`), against `V*` at a tape `θ` of the registration's view, the auditor's tape
and the commitments, and `ε_o` the statements' error (`LoopSessions.err`). It takes each statement's sessions'
`ZKmalR` and, of V*'s staging, each an obligation: the simulator's public context is the honest one's
(`LoopSessions.Public`), the honest witnesses are valid (`LoopSessions.Honest`) and V*'s statements fix the
simulator's randomness (`LoopSessions.Sized`). -/
theorem outerSessionsZK_of {Ow Prog Wit Rv Ωa Ωp Pub WS NB ΩB Θ : Type} [Fintype NB] [Nonempty NB] [Fintype ΩB]
    [Nonempty ΩB] {nr nh : ℕ} (rows : Prog → Fin nr → ByteArray) (pub : (Fin nr → ByteArray) → Rv)
    (honest : Ow → Prog → Wit → Prop) (msgs : Prog → Wit → (Fin nr → ZkView.Salt) → Ωa → Ωp → Fin nh → ByteArray)
    (V : LoopSessions (HCtx Prog Wit Ωa Ωp nr nh) Pub WS) (pubS : Ow → Rv → Ωa → (Fin nh → ByteArray) → Pub)
    (H : Claims.Msg → Claims.Msg) (kS : Claims.Msg → ℕ) (PS : WS → Claims.Prover (NB × ΩB)) (R : Claims.Rel WS)
    (Sim : Claims.Simulator) (εS : Claims.Msg → ℝ) (g : ℝ) (CB : ℕ → Type) [∀ j, Fintype (CB j)]
    [∀ j, Nonempty (CB j)] (hZK : Claims.ZKmalR H kS PS R Sim εS fun _ => g) (hε : ∀ x, 0 ≤ εS x) (hg : 0 ≤ g)
    (Vs : Claims.VStar Θ) (θ : Rv → Ωa → (Fin nh → ByteArray) → Θ)
    (hpub : V.Public rows pub honest (committed msgs) pubS) (hH : V.Honest R honest (committed msgs))
    (hS : V.Sized Sim CB) :
    _root_.Flock.Guarantees.OuterSessionsZK (_root_.Flock.Guarantees.regLeaves rows pub) honest nh msgs
      (fun P w r a h p y q =>
        V.viewAt H kS PS Vs (P, w, r, a, h, p, y) (θ (_root_.Flock.Guarantees.regLeaves rows pub P r) a h) q)
      (fun o v a h s => V.simAt NB ΩB H kS Sim CB Vs (pubS o v a h) (θ v a h) s)
      (V.err H kS PS εS g Vs fun x =>
        θ (_root_.Flock.Guarantees.regLeaves rows pub x.1 x.2.2.1) x.2.2.2.1 x.2.2.2.2.1) := by
  intro o P w hw r a p y E
  have e : _root_.Flock.Guarantees.firewallCommits (committed msgs P w r a p) y =
      fun i => _root_.Flock.Guarantees.firewallLeaf (msgs P w r a p i) (y i) :=
    firewallCommits_committed (msgs P w r a p) y
  have hpc := hpub o P w hw r a p y
  have hR := hH o P w hw r a p y
  simp only [hctxAt, e] at hpc hR
  exact V.view_le H kS PS R Sim εS g CB hZK hε hg Vs
    (fun x => θ (_root_.Flock.Guarantees.regLeaves rows pub x.1 x.2.2.1) x.2.2.2.1 x.2.2.2.2.1) _ _ hpc
    (fun j hj => hS j hj _) hR E

/-- The session loop's session at its own frames, link and `Finish` (`SessionLoopZKDeployed`'s): the deployed tables'
session with the loop's stops. -/
noncomputable abbrev loopRef
    (L₀ : Claims.ZKServer.LeanRelease Claims.Flock.Wit Claims.ZKServer.Loop.Tp)
    (src : Claims.Msg → Claims.Flock.Wit → Claims.ZKServer.Loop.Tp → ℕ →
      _root_.Flock.V (Array _root_.Flock.F128 × ByteArray × Array ByteArray)) :
    Claims.ZKMal.RefSession :=
  Claims.ZKServer.Loop.finS.withStops
    (Claims.ZKServer.LeanRelease.rfOf Claims.ZKServer.Loop.finS
      (Claims.ZKServer.LeanRelease.ofLoopLink L₀ src (Claims.ZKServer.Loop.realOf src))
      Claims.ZKServer.Loop.progOf Claims.ZKServer.Loop.freshOf Claims.ZKServer.Loop.readCoins)
    (Claims.ZKServer.LeanRelease.ofLoopLink L₀ src (Claims.ZKServer.Loop.realOf src)).stop

/-- The rounds of one of the loop's sessions: the masked verifier's and the `Finish`. -/
noncomputable abbrev loopK (x : Claims.Msg) : ℕ := (FlockDeploy.deployZk _root_.Flock.Readable.relation).V.k x + 1

/-- The loop's session's simulator (`Flock.zk_deployZk_sw`'s) at `t` rewinds. -/
noncomputable abbrev loopSim
    (L₀ : Claims.ZKServer.LeanRelease Claims.Flock.Wit Claims.ZKServer.Loop.Tp)
    (src : Claims.Msg → Claims.Flock.Wit → Claims.ZKServer.Loop.Tp → ℕ →
      _root_.Flock.V (Array _root_.Flock.F128 × ByteArray × Array ByteArray)) (t : ℕ) : Claims.Simulator :=
  ((loopRef L₀ src).refP.withSim fun x _ c => (loopRef L₀ src).coinModel x c).simulatorE t (t * 3 ^ t)

/-- The loop's session's error (`Flock.zk_deployZk_sw`'s) at the pinned key's hiding gap and `t` rewinds. -/
noncomputable abbrev loopEps
    (L₀ : Claims.ZKServer.LeanRelease Claims.Flock.Wit Claims.ZKServer.Loop.Tp)
    (src : Claims.Msg → Claims.Flock.Wit → Claims.ZKServer.Loop.Tp → ℕ →
      _root_.Flock.V (Array _root_.Flock.F128 × ByteArray × Array ByteArray)) (t : ℕ) (x : Claims.Msg) : ℝ :=
  8 * (loopRef L₀ src).nLeaves x * CROnly.hidingGap ZkView.My ZkView.saltHash +
    8 * t * (loopRef L₀ src).nLeaves x * CROnly.hidingGap ZkView.My ZkView.saltHash + 2 * (1 / 3) ^ t

/-- **V*'s statements run by the session loop give the committed-coin statements their `_hOuter`** (`OuterSessionsZK`):
`outerSessionsZK_of` at the loop's session (`loopRef`), each statement run by its switched prover at the workers' values
`vs` (`(loopRef L₀ src).swP vs`), the simulator and errors `Flock.zk_deployZk_sw`'s at the pinned key's hiding gap and
`t` rewinds (`loopSim`, `loopEps`, `4·t·3^t`). The view is the switched prover's; what the loop's own prover sends is
`SessionLoopZKDeployed`'s, which bounds it by twice the simulated view plus its error. `SessionLoopZKDeployed`'s
premises on the deployed tables (`Tables`, `Complete`, `DummyReg`, `Causal`, `SwCausal`, `TapeOK`) take
`SessionsCoined`'s role, with no further obligation on the tables; V*'s staging is `outerSessionsZK_of`'s. -/
theorem outerSessionsZK_loop {Ow Prog Wit Rv Ωa Ωp Pub Θ : Type} {nr nh : ℕ}
    (vs : Claims.ZKServer.Loop.finS.SwVals) (t : ℕ) (ht : 0 < t) (hS : Claims.ZKServer.Loop.finS.Tables)
    (hV : Claims.ZKServer.Loop.finS.Complete
      (Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse))
    (hD : Claims.ZKServer.Loop.finS.DummyReg) (hc : Claims.ZKServer.Loop.finS.Causal)
    (hsc : Claims.ZKServer.Loop.finS.SwCausal vs) (hT : Claims.ZKServer.Loop.finS.TapeOK)
    (L₀ : Claims.ZKServer.LeanRelease Claims.Flock.Wit Claims.ZKServer.Loop.Tp)
    (src : Claims.Msg → Claims.Flock.Wit → Claims.ZKServer.Loop.Tp → ℕ →
      _root_.Flock.V (Array _root_.Flock.F128 × ByteArray × Array ByteArray))
    (rows : Prog → Fin nr → ByteArray) (pub : (Fin nr → ByteArray) → Rv) (honest : Ow → Prog → Wit → Prop)
    (msgs : Prog → Wit → (Fin nr → ZkView.Salt) → Ωa → Ωp → Fin nh → ByteArray)
    (V : LoopSessions (HCtx Prog Wit Ωa Ωp nr nh) Pub Claims.Flock.Wit)
    (pubS : Ow → Rv → Ωa → (Fin nh → ByteArray) → Pub) (CB : ℕ → Type) [∀ j, Fintype (CB j)]
    [∀ j, Nonempty (CB j)] (Vs : Claims.VStar Θ) (θ : Rv → Ωa → (Fin nh → ByteArray) → Θ)
    (hpub : V.Public rows pub honest (committed msgs) pubS)
    (hH : V.Honest (Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse) honest
      (committed msgs))
    (hsz : V.Sized (loopSim L₀ src t) CB) :
    _root_.Flock.Guarantees.OuterSessionsZK (_root_.Flock.Guarantees.regLeaves rows pub) honest nh msgs
      (fun P w r a h p y q => V.viewAt Claims.ZKMal.CoinOpen.shaH loopK ((loopRef L₀ src).swP vs).prover Vs
        (P, w, r, a, h, p, y) (θ (_root_.Flock.Guarantees.regLeaves rows pub P r) a h) q)
      (fun o v a h s => V.simAt ((loopRef L₀ src).swP vs).Nonce ((loopRef L₀ src).swP vs).Rand
        Claims.ZKMal.CoinOpen.shaH loopK (loopSim L₀ src t) CB Vs (pubS o v a h) (θ v a h) s)
      (V.err Claims.ZKMal.CoinOpen.shaH loopK ((loopRef L₀ src).swP vs).prover (loopEps L₀ src t)
        (4 * t * 3 ^ t) Vs fun x =>
          θ (_root_.Flock.Guarantees.regLeaves rows pub x.1 x.2.2.1) x.2.2.2.1 x.2.2.2.2.1) := by
  have h := Claims.ZKMal.Flock.zk_deployZk_sw Claims.ZKMal.CoinOpen.shaH _root_.Flock.Readable.relation
    (loopRef L₀ src) vs t ht (CROnly.hm96Hiding_gap ZkView.My ZkView.saltHash) ENNReal.ofReal_ne_top
    (Claims.ZKMal.RefSession.withStops_tables _ _ _ hS) (Claims.ZKMal.RefSession.withStops_complete _ _ _ hV)
    (Claims.ZKMal.RefSession.withStops_dummyReg _ _ _ hD)
    (Claims.ZKMal.RefSession.withStops_causal _ _ _ (Claims.ZKServer.LeanRelease.stopsCausal_rfOf _ _ _ _ _) hc)
    (Claims.ZKMal.RefSession.withStops_swCausal _ _ _ vs (Claims.ZKServer.LeanRelease.stopsCausal_rfOf _ _ _ _ _)
      hsc)
    (Claims.ZKMal.RefSession.withStops_tapeOK _ _ _ hT)
  rw [ENNReal.toReal_ofReal (CROnly.hidingGap_nonneg ZkView.My ZkView.saltHash)] at h
  have h0 := CROnly.hidingGap_nonneg ZkView.My ZkView.saltHash
  exact outerSessionsZK_of rows pub honest msgs V pubS _ _ _ _ _ _ _ CB h.2
    (fun _ => add_nonneg (add_nonneg (mul_nonneg (by positivity) h0) (mul_nonneg (by positivity) h0))
      (by positivity))
    (by positivity) Vs θ hpub hH hsz

end FlockSoundness.Discharge.Recursion.OuterZK
