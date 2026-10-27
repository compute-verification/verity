import Proofs.Flock.Claims.KS.ZkTrimW
import Proofs.Flock.Claims.KS.ZkFirst

/-!
# The masked chain from a simulator

`PlainSimW` at `--zk`. `ZkXferWJ` asks, for each prover `P`, for events tied to collisions and, at each tape and
table, one mixture of live strategies with the first bound, the fork bounds and the self-clash bound. With the events
an accepting run or fork with its collision (`zkForkEv`, `zkEv₁`) and the mixture the strategies of a live prover
`P'` with a finite tape and coins `Φ` of `V true` at the tape and the live coins, of uniform law, whose strategies send
`P`'s roots (`zkRoot`), two of those bounds come from deterministic facts about `P'` (`ZkSimWJ`):

* the first, from `P'` simulating `P`'s accepting runs at `Φ` (`ZkSimRun`), by `zk_first_of_sim`, with no premise on
  the record's draw;
* the self-clash bound, from a live self-clash at `P'`'s outcome giving an accepting run of `P` at `Φ` with a
  collision among the verifier's calls (`ZkClashRun`), by `liveMix_le_Pr`;
* the fork bounds at `P'`'s strategies (`Adv`'s three and level 0's `Adv₀`) are asked as they are, each at most its
  fork event's probability and the fork bound's slack at its round (`Claims.Flock.forkSlack`), whose events the
  escapes merge by round (`Claims.Flock.slackEscW`).

`Φ` may be `zkCoins` with the deployed coins as the tape (`zkSim_of_run`). The strategies send `P`'s roots wherever
`P'`'s first message is `P`'s and the wire reads roots as the deployed link does (`zkRoot_of_first`). So `ZkSimWJ`
gives `ZkXferWJ` (`zkXferWJ_of_simWJ`), and `Flock.KS` at the trimmed masked record follows from `ZkSimWJ` at the
statements some run accepts (`zk_KS_trim_of_simWJ`), table 0's bound with escapes on accepted runs counted per round
from it at the untrimmed one (`zk_tableBoundθAccAtR_of_simWJ`).
-/

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkSession
open FlockSoundness.Discharge
open FlockSoundness.Discharge.Tables (LiveOutJ liveSession DecodesTable)
open FlockSoundness.Discharge.Composed (outJ repJ)
open Claims.Bridge.Live

local notation "𝔽" => FlockLevel3.GF128

/-- **The roots of `P`'s first message**, as `zk_first_of_sim` reads them. -/
def zkRoot {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) : Array ByteArray :=
  ((linkOf ((P.next x [] ω).eval sha512).1).getD default).1.toArray

/-- **A live prover whose first message is `P`'s sends `P`'s roots**, where the wire reads roots as the deployed link
does (`liveStrat_fst`): the strategies' roots `ZkSimWJ` asks for. -/
theorem zkRoot_of_first {Ω τ : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω} {J : Claims.Msg → ℕ} {mPts : ℕ}
    (N : Claims.Msg → ℕ) (dg : ByteArray → ByteArray) {w : (x : Claims.Msg) → Wire 65536 (J x) mPts}
    {P' : Claims.Prover (Ω × τ)} (hw : ∀ a, (w x).root a = ((linkOf a).getD default).1.toArray)
    (h1 : ∀ u, ((P'.next x [] (ω, u)).eval sha512).1 = ((P.next x [] ω).eval sha512).1) (u : τ) :
    (liveStrat J (fun _ => mPts) N dg w sha512 P' x (ω, u)).1 = zkRoot P x ω := by
  rw [liveStrat_fst, hw, h1]
  rfl

/-- **An accepting masked run with a collision among the verifier's calls**: the one-run event the self-clash bound
reads. -/
def zkEv₁ {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) (s : Fin ((V true).k x) → Coin) : Prop :=
  (Claims.run sha512 P (V true) x ω s).out = .acc ∧ Claims.TiedColl sha512 (Claims.run sha512 P (V true) x ω s).vLog

namespace Gen

variable [S : ZkSetup]

/-- **An accepting masked run with a collision among the verifier's calls**: the one-run event the self-clash bound
reads. -/
def zkEv₁ {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) (s : Fin ((V true).k x) → Coin) : Prop :=
  (Claims.run sha512 P (V true) x ω s).out = .acc ∧ Claims.TiedColl sha512 (Claims.run sha512 P (V true) x ω s).vLog

end Gen

/-- **An accepting masked fork at the coin `j₀` with a collision among the verifier's calls in its two runs**: the
event the fork bounds read. -/
def zkForkEv {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (j₀ : Fin ((V true).k x)) (ω : Ω)
    (s s' : Fin ((V true).k x) → Coin) : Prop :=
  (Claims.run sha512 P (V true) x ω s).out = .acc ∧
    (Claims.run sha512 P (V true) x ω (Claims.forkCoins j₀ s s')).out = .acc ∧
    Claims.TiedColl sha512 ((Claims.run sha512 P (V true) x ω s).vLog ++
      (Claims.run sha512 P (V true) x ω (Claims.forkCoins j₀ s s')).vLog)

namespace Gen

variable [S : ZkSetup]

/-- **An accepting masked fork at the coin `j₀` with a collision among the verifier's calls in its two runs**: the
event the fork bounds read. -/
def zkForkEv {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (j₀ : Fin ((V true).k x)) (ω : Ω)
    (s s' : Fin ((V true).k x) → Coin) : Prop :=
  (Claims.run sha512 P (V true) x ω s).out = .acc ∧
    (Claims.run sha512 P (V true) x ω (Claims.forkCoins j₀ s s')).out = .acc ∧
    Claims.TiedColl sha512 ((Claims.run sha512 P (V true) x ω s).vLog ++
      (Claims.run sha512 P (V true) x ω (Claims.forkCoins j₀ s s')).vLog)

end Gen

/-- **A live event at `P'`'s outcome gives an accepting masked run of `P` with a collision**: wherever `pL` holds at
the live outcome of `P'` at the tape `(ω, u)` and the live coins `r'`, `V true` accepts `P` at `Φ (u, r')` with a
collision among its calls (`zkEv₁`). -/
def ZkClashRun {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) (mPts : ℕ) (J N : Claims.Msg → ℕ)
    (dg : ByteArray → ByteArray) (w : (x : Claims.Msg) → Wire 65536 (J x) mPts) {τ : Type}
    (P' : Claims.Prover (Ω × τ)) (Φ : τ × (Fin (N x + 2) → LCoin 65536) → Fin ((V true).k x) → Coin)
    (pL : LiveOutJ (J x) mPts → Prop) : Prop :=
  ∀ u r', pL (liveOut J (fun _ => mPts) N dg w sha512 P' x (ω, u) r') → zkEv₁ P x ω (Φ (u, r'))

namespace Gen

variable [S : ZkSetup]

/-- **A live event at `P'`'s outcome gives an accepting masked run of `P` with a collision**: wherever `pL` holds at
the live outcome of `P'` at the tape `(ω, u)` and the live coins `r'`, `V true` accepts `P` at `Φ (u, r')` with a
collision among its calls (`zkEv₁`). -/
def ZkClashRun {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) (mPts : ℕ) (J N : Claims.Msg → ℕ)
    (dg : ByteArray → ByteArray) (w : (x : Claims.Msg) → Wire 65536 (J x) mPts) {τ : Type}
    (P' : Claims.Prover (Ω × τ)) (Φ : τ × (Fin (N x + 2) → LCoin 65536) → Fin ((V true).k x) → Coin)
    (pL : LiveOutJ (J x) mPts → Prop) : Prop :=
  ∀ u r', pL (liveOut J (fun _ => mPts) N dg w sha512 P' x (ω, u) r') → zkEv₁ P x ω (Φ (u, r'))

end Gen

open Classical in
/-- **A simulator of each masked prover at the witness-lane verdict**: at a masked statement in scope that sets up at
a draw, for each prover `P`, injective fork rounds for every table's events and, at each tape `ω`, a live prover `P'`
of the session at C-Flock ZK's padded tables with a finite tape `τ` and coins `Φ` of `V true` at the tape and the live
coins, of uniform law, whose strategies send `P`'s roots (`zkRoot`), that simulates `P`'s accepting runs at `Φ`
(`ZkSimRun`), and at each table `j` and table of full-width columns, whose live self-clashes give `P`'s accepting runs
with a collision there (`ZkClashRun`) and whose strategies' fork bounds after `Commit` at the deployed decoders right
at `OkOffWJ` are no more than an accepting fork's with a collision (`zkForkEv`) and the slack of the fork at its round
(`Claims.Flock.forkSlack` at `δ ≥ 0`), where only level 0's event of a table is at round `0` and the slack fits the
masked weight's room (`Claims.Flock.SlackFitW`). -/
def ZkSimWJ (x : Claims.Msg) : Prop :=
  ∀ (s : Claims.Flock.Stmt) (hs : parse true Readable.relation x = some s) (hin : Claims.Flock.InScope true s)
    (ins : Ins) (spec : SessionSpec) (sts : Array Stmt) (hset : setupOf true x = some (ins, spec, sts))
    (h0 : 0 < sts.size) (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    (d : Lean.Json) (hd : ins.read.draw = some d) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Claims.Prover Ω),
    ∃ (rd : (j : ℕ) → Claims.Flock.IZW (zkTabs h0 L hz hk hlay j).sch.levels.length → Fin ((V true).k x))
      (_ : ∀ j, Function.Injective (rd j)) (_ : ∀ j i, (rd j i).val = 0 → i = .inr ()) (δ : ℕ → ℝ)
      (_ : ∀ t, 0 ≤ δ t)
      (_ : Claims.Flock.SlackFitW (tables true x).length (fun j => (zkTabs h0 L hz hk hlay j).sch)
        (fun j => (zkTabs h0 L hz hk hlay j).N) rd δ),
      ∀ ω, ∃ (N : Claims.Msg → ℕ) (dg : ByteArray → ByteArray)
        (w : Claims.Msg → Wire 65536 (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length spec.mPts)
        (τ : Type) (_ : Fintype τ) (_ : Nonempty τ) (P' : Claims.Prover (Ω × τ))
        (Φ : τ × (Fin (N x + 2) → LCoin 65536) → Fin ((V true).k x) → Coin)
        (_ : ∀ E : (Fin ((V true).k x) → Coin) → Prop, Claims.Pr (fun p => E (Φ p)) = Claims.Pr E)
        (hσ : ∀ u, (liveStrat (fun _ => (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length)
          (fun _ => spec.mPts) N dg w sha512 P' x (ω, u)).1 = zkRoot P x ω),
        ZkSimRun P x ω (fun _ => (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length) spec.mPts N dg w
          P' Φ ∧
        ∀ (j : ℕ) (hj : j < (tables true x).length) (C : Fin (wordLenJ true x j) → List 𝔽),
          (∀ p, wideJ true x j (C p)) →
          ZkClashRun P x ω spec.mPts (fun _ => (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length) N dg
            w P' Φ
            (LiveSelfClashZ ZkExec.hs512 (zkTabs h0 L hz hk hlay j)
              (outJ ZkExec.hs512 (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)))
              (AccTxJ ins.read.I d j (zkRoot P x ω) C)
              (opnD sts _) (decD _) (fun _ _ => true) (fun _ _ _ => []) (fun rs j => Exec.capOfRoot rs[j.val]!)
              (repJ execArith ZkExec.hs512 (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay))) j) ∧
            (∀ r ℓ, (∑ u, advRSZ execArith ZkExec.hs512 (zkTabs h0 L hz hk hlay j) _ _
                (Exec.capOfRoot (zkRoot P x ω)[j]!) r.val (ℓ.val + 1)
                (simAtJZ execArith ZkExec.hs512 _ (OkOffWJ (tabsJ _ (zkTabs h0 L hz hk hlay)) j (zkRoot P x ω) C)
                  (AccTxJ ins.read.I d j (zkRoot P x ω) C) (opnD sts _) dg (decD _) (fun _ _ => true) (fun _ _ _ => [])
                  (fun rs j => Exec.capOfRoot rs[j.val]!) (N x)
                  (decodesTableW_deploy hs hin hset h0 L hz hk hlay hd j (zkRoot P x ω) C) (zkRoot P x ω)
                  (liveStrat (fun _ => (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length)
                    (fun _ => spec.mPts) N dg w sha512 P' x (ω, u)) (hσ u) (finJ (zkTabs h0 L hz hk hlay) hj)
                  (tabsJ_get _ _ _))) / Fintype.card τ ≤
              (Claims.Pr fun s : (Fin ((V true).k x) → Coin) × (Fin ((V true).k x) → Coin) =>
                zkForkEv P x (rd j (.inl (.inl (r, ℓ)))) ω s.1 s.2) +
                Claims.Flock.forkSlack sha512 (V true) P x δ (rd j (.inl (.inl (r, ℓ)))).val ω) ∧
            (∀ r : Fin 2, (∑ u, advPadsSZ execArith ZkExec.hs512 (zkTabs h0 L hz hk hlay j) _ _
                (Exec.capOfRoot (zkRoot P x ω)[j]!) r.val
                (simAtJZ execArith ZkExec.hs512 _ (OkOffWJ (tabsJ _ (zkTabs h0 L hz hk hlay)) j (zkRoot P x ω) C)
                  (AccTxJ ins.read.I d j (zkRoot P x ω) C) (opnD sts _) dg (decD _) (fun _ _ => true) (fun _ _ _ => [])
                  (fun rs j => Exec.capOfRoot rs[j.val]!) (N x)
                  (decodesTableW_deploy hs hin hset h0 L hz hk hlay hd j (zkRoot P x ω) C) (zkRoot P x ω)
                  (liveStrat (fun _ => (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length)
                    (fun _ => spec.mPts) N dg w sha512 P' x (ω, u)) (hσ u) (finJ (zkTabs h0 L hz hk hlay) hj)
                  (tabsJ_get _ _ _))) / Fintype.card τ ≤
              (Claims.Pr fun s : (Fin ((V true).k x) → Coin) × (Fin ((V true).k x) → Coin) =>
                zkForkEv P x (rd j (.inl (.inr (.inl r)))) ω s.1 s.2) +
                Claims.Flock.forkSlack sha512 (V true) P x δ (rd j (.inl (.inr (.inl r)))).val ω) ∧
            (∀ r : Fin 2, (∑ u, advTauSZ execArith ZkExec.hs512 (zkTabs h0 L hz hk hlay j) _ _
                (Exec.capOfRoot (zkRoot P x ω)[j]!) r.val
                (simAtJZ execArith ZkExec.hs512 _ (OkOffWJ (tabsJ _ (zkTabs h0 L hz hk hlay)) j (zkRoot P x ω) C)
                  (AccTxJ ins.read.I d j (zkRoot P x ω) C) (opnD sts _) dg (decD _) (fun _ _ => true) (fun _ _ _ => [])
                  (fun rs j => Exec.capOfRoot rs[j.val]!) (N x)
                  (decodesTableW_deploy hs hin hset h0 L hz hk hlay hd j (zkRoot P x ω) C) (zkRoot P x ω)
                  (liveStrat (fun _ => (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length)
                    (fun _ => spec.mPts) N dg w sha512 P' x (ω, u)) (hσ u) (finJ (zkTabs h0 L hz hk hlay) hj)
                  (tabsJ_get _ _ _))) / Fintype.card τ ≤
              (Claims.Pr fun s : (Fin ((V true).k x) → Coin) × (Fin ((V true).k x) → Coin) =>
                zkForkEv P x (rd j (.inl (.inr (.inr r)))) ω s.1 s.2) +
                Claims.Flock.forkSlack sha512 (V true) P x δ (rd j (.inl (.inr (.inr r)))).val ω) ∧
            ((∑ u, adv₀SZ execArith ZkExec.hs512 (zkTabs h0 L hz hk hlay j) _ _
                (Exec.capOfRoot (zkRoot P x ω)[j]!)
                (simAtJZ execArith ZkExec.hs512 _ (OkOffWJ (tabsJ _ (zkTabs h0 L hz hk hlay)) j (zkRoot P x ω) C)
                  (AccTxJ ins.read.I d j (zkRoot P x ω) C) (opnD sts _) dg (decD _) (fun _ _ => true) (fun _ _ _ => [])
                  (fun rs j => Exec.capOfRoot rs[j.val]!) (N x)
                  (decodesTableW_deploy hs hin hset h0 L hz hk hlay hd j (zkRoot P x ω) C) (zkRoot P x ω)
                  (liveStrat (fun _ => (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length)
                    (fun _ => spec.mPts) N dg w sha512 P' x (ω, u)) (hσ u) (finJ (zkTabs h0 L hz hk hlay) hj)
                  (tabsJ_get _ _ _))) / Fintype.card τ ≤
              (Claims.Pr fun s : (Fin ((V true).k x) → Coin) × (Fin ((V true).k x) → Coin) =>
                zkForkEv P x (rd j (.inr ())) ω s.1 s.2) +
                Claims.Flock.forkSlack sha512 (V true) P x δ (rd j (.inr ())).val ω)

namespace Gen

variable [S : ZkSetup]

open Classical in
/-- **A simulator of each masked prover at the witness-lane verdict**: at a masked statement in scope that sets up at
a draw, for each prover `P`, injective fork rounds for every table's events and, at each tape `ω`, a live prover `P'`
of the session at C-Flock ZK's padded tables with a finite tape `τ` and coins `Φ` of `V true` at the tape and the live
coins, of uniform law, whose strategies send `P`'s roots (`zkRoot`), that simulates `P`'s accepting runs at `Φ`
(`ZkSimRun`), and at each table `j` and table of full-width columns, whose live self-clashes give `P`'s accepting runs
with a collision there (`ZkClashRun`) and whose strategies' fork bounds after `Commit` at the deployed decoders right
at `OkOffWJ` are no more than an accepting fork's with a collision (`zkForkEv`) and the slack of the fork at its round
(`Claims.Flock.forkSlack` at `δ ≥ 0`), where only level 0's event of a table is at round `0` and the slack fits the
masked weight's room (`Claims.Flock.SlackFitW`). -/
def ZkSimWJ (x : Claims.Msg) : Prop :=
  ∀ (s : Claims.Flock.Stmt) (hs : parse true Readable.relation x = some s) (hin : Claims.Flock.InScope true s)
    (ins : Ins) (spec : SessionSpec) (sts : Array Stmt) (hset : setupOf true x = some (ins, spec, sts))
    (h0 : 0 < sts.size) (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    (d : Lean.Json) (hd : ins.read.draw = some d) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Claims.Prover Ω),
    ∃ (rd : (j : ℕ) → Claims.Flock.IZW (zkTabs h0 L hz hk hlay j).sch.levels.length → Fin ((V true).k x))
      (_ : ∀ j, Function.Injective (rd j)) (_ : ∀ j i, (rd j i).val = 0 → i = .inr ()) (δ : ℕ → ℝ)
      (_ : ∀ t, 0 ≤ δ t)
      (_ : Claims.Flock.SlackFitW (tables true x).length (fun j => (zkTabs h0 L hz hk hlay j).sch)
        (fun j => (zkTabs h0 L hz hk hlay j).N) rd δ),
      ∀ ω, ∃ (N : Claims.Msg → ℕ) (dg : ByteArray → ByteArray)
        (w : Claims.Msg → Wire 65536 (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length spec.mPts)
        (τ : Type) (_ : Fintype τ) (_ : Nonempty τ) (P' : Claims.Prover (Ω × τ))
        (Φ : τ × (Fin (N x + 2) → LCoin 65536) → Fin ((V true).k x) → Coin)
        (_ : ∀ E : (Fin ((V true).k x) → Coin) → Prop, Claims.Pr (fun p => E (Φ p)) = Claims.Pr E)
        (hσ : ∀ u, (liveStrat (fun _ => (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length)
          (fun _ => spec.mPts) N dg w sha512 P' x (ω, u)).1 = zkRoot P x ω),
        ZkSimRun P x ω (fun _ => (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length) spec.mPts N dg w
          P' Φ ∧
        ∀ (j : ℕ) (hj : j < (tables true x).length) (C : Fin (wordLenJ true x j) → List 𝔽),
          (∀ p, wideJ true x j (C p)) →
          ZkClashRun P x ω spec.mPts (fun _ => (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length) N dg
            w P' Φ
            (LiveSelfClashZ ZkExec.hs512 (zkTabs h0 L hz hk hlay j)
              (outJ ZkExec.hs512 (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)))
              (AccTxJ ins.read.I d j (zkRoot P x ω) C)
              (opnD sts _) (decD _) (fun _ _ => true) (fun _ _ _ => []) (fun rs j => Exec.capOfRoot rs[j.val]!)
              (repJ execArith ZkExec.hs512 (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay))) j) ∧
            (∀ r ℓ, (∑ u, advRSZ execArith ZkExec.hs512 (zkTabs h0 L hz hk hlay j) _ _
                (Exec.capOfRoot (zkRoot P x ω)[j]!) r.val (ℓ.val + 1)
                (simAtJZ execArith ZkExec.hs512 _ (OkOffWJ (tabsJ _ (zkTabs h0 L hz hk hlay)) j (zkRoot P x ω) C)
                  (AccTxJ ins.read.I d j (zkRoot P x ω) C) (opnD sts _) dg (decD _) (fun _ _ => true) (fun _ _ _ => [])
                  (fun rs j => Exec.capOfRoot rs[j.val]!) (N x)
                  (decodesTableW_deploy hs hin hset h0 L hz hk hlay hd j (zkRoot P x ω) C) (zkRoot P x ω)
                  (liveStrat (fun _ => (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length)
                    (fun _ => spec.mPts) N dg w sha512 P' x (ω, u)) (hσ u) (finJ (zkTabs h0 L hz hk hlay) hj)
                  (tabsJ_get _ _ _))) / Fintype.card τ ≤
              (Claims.Pr fun s : (Fin ((V true).k x) → Coin) × (Fin ((V true).k x) → Coin) =>
                zkForkEv P x (rd j (.inl (.inl (r, ℓ)))) ω s.1 s.2) +
                Claims.Flock.forkSlack sha512 (V true) P x δ (rd j (.inl (.inl (r, ℓ)))).val ω) ∧
            (∀ r : Fin 2, (∑ u, advPadsSZ execArith ZkExec.hs512 (zkTabs h0 L hz hk hlay j) _ _
                (Exec.capOfRoot (zkRoot P x ω)[j]!) r.val
                (simAtJZ execArith ZkExec.hs512 _ (OkOffWJ (tabsJ _ (zkTabs h0 L hz hk hlay)) j (zkRoot P x ω) C)
                  (AccTxJ ins.read.I d j (zkRoot P x ω) C) (opnD sts _) dg (decD _) (fun _ _ => true) (fun _ _ _ => [])
                  (fun rs j => Exec.capOfRoot rs[j.val]!) (N x)
                  (decodesTableW_deploy hs hin hset h0 L hz hk hlay hd j (zkRoot P x ω) C) (zkRoot P x ω)
                  (liveStrat (fun _ => (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length)
                    (fun _ => spec.mPts) N dg w sha512 P' x (ω, u)) (hσ u) (finJ (zkTabs h0 L hz hk hlay) hj)
                  (tabsJ_get _ _ _))) / Fintype.card τ ≤
              (Claims.Pr fun s : (Fin ((V true).k x) → Coin) × (Fin ((V true).k x) → Coin) =>
                zkForkEv P x (rd j (.inl (.inr (.inl r)))) ω s.1 s.2) +
                Claims.Flock.forkSlack sha512 (V true) P x δ (rd j (.inl (.inr (.inl r)))).val ω) ∧
            (∀ r : Fin 2, (∑ u, advTauSZ execArith ZkExec.hs512 (zkTabs h0 L hz hk hlay j) _ _
                (Exec.capOfRoot (zkRoot P x ω)[j]!) r.val
                (simAtJZ execArith ZkExec.hs512 _ (OkOffWJ (tabsJ _ (zkTabs h0 L hz hk hlay)) j (zkRoot P x ω) C)
                  (AccTxJ ins.read.I d j (zkRoot P x ω) C) (opnD sts _) dg (decD _) (fun _ _ => true) (fun _ _ _ => [])
                  (fun rs j => Exec.capOfRoot rs[j.val]!) (N x)
                  (decodesTableW_deploy hs hin hset h0 L hz hk hlay hd j (zkRoot P x ω) C) (zkRoot P x ω)
                  (liveStrat (fun _ => (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length)
                    (fun _ => spec.mPts) N dg w sha512 P' x (ω, u)) (hσ u) (finJ (zkTabs h0 L hz hk hlay) hj)
                  (tabsJ_get _ _ _))) / Fintype.card τ ≤
              (Claims.Pr fun s : (Fin ((V true).k x) → Coin) × (Fin ((V true).k x) → Coin) =>
                zkForkEv P x (rd j (.inl (.inr (.inr r)))) ω s.1 s.2) +
                Claims.Flock.forkSlack sha512 (V true) P x δ (rd j (.inl (.inr (.inr r)))).val ω) ∧
            ((∑ u, adv₀SZ execArith ZkExec.hs512 (zkTabs h0 L hz hk hlay j) _ _
                (Exec.capOfRoot (zkRoot P x ω)[j]!)
                (simAtJZ execArith ZkExec.hs512 _ (OkOffWJ (tabsJ _ (zkTabs h0 L hz hk hlay)) j (zkRoot P x ω) C)
                  (AccTxJ ins.read.I d j (zkRoot P x ω) C) (opnD sts _) dg (decD _) (fun _ _ => true) (fun _ _ _ => [])
                  (fun rs j => Exec.capOfRoot rs[j.val]!) (N x)
                  (decodesTableW_deploy hs hin hset h0 L hz hk hlay hd j (zkRoot P x ω) C) (zkRoot P x ω)
                  (liveStrat (fun _ => (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length)
                    (fun _ => spec.mPts) N dg w sha512 P' x (ω, u)) (hσ u) (finJ (zkTabs h0 L hz hk hlay) hj)
                  (tabsJ_get _ _ _))) / Fintype.card τ ≤
              (Claims.Pr fun s : (Fin ((V true).k x) → Coin) × (Fin ((V true).k x) → Coin) =>
                zkForkEv P x (rd j (.inr ())) ω s.1 s.2) +
                Claims.Flock.forkSlack sha512 (V true) P x δ (rd j (.inr ())).val ω)

/-- **A simulator gives the masked live sessions' bound at the deployed decoders off the table on a witness lane**:
the events an accepting fork or run with its collision, the first bound by `zk_first_of_sim`, the self-clash bound by
`liveMix_le_Pr`, the fork bounds as `ZkSimWJ` gives them. -/
theorem zkXferWJ_of_simWJ {x : Claims.Msg} (hx : ZkSimWJ x) : ZkXferWJ x := by
  intro s hs hin ins spec sts hset h0 L hz hk hlay d hd Ω _ _ P
  obtain ⟨rd, hrd, hr0, δ, hδ, hfit, hω⟩ := hx s hs hin ins spec sts hset h0 L hz hk hlay d hd Ω P
  refine ⟨rd, hrd, hr0, δ, hδ, hfit, fun j i => zkForkEv P x (rd j i), fun _ => zkEv₁ P x, fun _ _ _ _ _ he => he.2.2,
    fun _ _ _ he => he.2, fun _ _ _ _ _ he => ⟨he.1, he.2.1⟩, fun _ _ _ he => he.1, fun ω j hj C hC => ?_⟩
  obtain ⟨N, dg, w, τ, _, _, P', Φ, hΦ, hσ, hrun, hj'⟩ := hω ω
  obtain ⟨hsc, hRS, hPads, hTau, h₀⟩ := hj' j hj C hC
  have hlen : (tables true x).length = sts.size := by
    unfold tables
    rw [hset]
    simp
  have hJ : (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length ≤ sts.size := by
    rw [tabsJ_length, hlen]
  have hjJ : j < (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length := by
    rw [tabsJ_length]
    exact hj
  exact ⟨dg, N x, zkRoot P x ω, τ, inferInstance, inferInstance,
    fun u => liveStrat (fun _ => (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length)
      (fun _ => spec.mPts) N dg w sha512 P' x (ω, u), hσ,
    zk_first_of_sim P x ω C j (fun _ => (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length) spec.mPts
      hset hd hJ hjJ N dg w P' Φ ⟨hΦ, hrun⟩,
    hRS, hPads, hTau, h₀,
    liveMix_le_Pr (fun _ => (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length) (fun _ => spec.mPts)
      N dg w sha512 P' x ω _ (zkEv₁ P x ω) Φ (hΦ _) hsc⟩

end Gen

/-- **A simulator gives the masked live sessions' bound at the deployed decoders off the table on a witness lane**:
the events an accepting fork or run with its collision, the first bound by `zk_first_of_sim`, the self-clash bound by
`liveMix_le_Pr`, the fork bounds as `ZkSimWJ` gives them. -/
theorem zkXferWJ_of_simWJ {x : Claims.Msg} (hx : ZkSimWJ x) : ZkXferWJ x :=
  Gen.zkXferWJ_of_simWJ (S := S₀) (x := x) (hx := hx)

namespace Gen

variable [S : ZkSetup]

/-- **The masked record the claims read: KS**, at SHA-512 at the deployed relation, at every number of tables, from a
simulator of each prover at the statements some run accepts (`ZkSimWJ`). -/
theorem zk_KS_trim_of_simWJ (hs : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkSimWJ x) :
    Claims.Flock.KS sha512 (deployZkTrim Readable.relation) :=
  zk_KS_trim_of_xferWJ fun x hno => zkXferWJ_of_simWJ (hs x hno)

end Gen

/-- **The masked record the claims read: KS**, at SHA-512 at the deployed relation, at every number of tables, from a
simulator of each prover at the statements some run accepts (`ZkSimWJ`). -/
theorem zk_KS_trim_of_simWJ (hs : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkSimWJ x) :
    Claims.Flock.KS sha512 (deployZkTrim Readable.relation) :=
  Gen.zk_KS_trim_of_simWJ (S := S₀) (hs := hs)

/-- **Table 0's masked bound at every statement, off the table on a witness lane, with escapes on accepted runs
counted per round**, at full-width columns, from a simulator of each prover at the statements some run accepts. -/
theorem zk_tableBoundθAccAtR_of_simWJ (hs : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkSimWJ x)
    (x : Claims.Msg) :
    Claims.Flock.TableBoundθAccAtR sha512 (deployZk Readable.relation) (wordLen true) (wide true) x :=
  zk_tableBoundθAccAtR_of_xferWJ (fun x hno => zkXferWJ_of_simWJ (hs x hno)) x

end FlockDeploy.KS
