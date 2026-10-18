import Proofs.Flock.Claims.KS.PlainDrawDec
import Proofs.Flock.Claims.KS.PlainSimW

/-!
# The plain chain from a simulator, at the statement's draw

The session set up at the record's draw would need the record to carry the statement's draw as the same term, which
the checks do not give (`drawEq_ne`). The chain here reads `plainVerdictAt`, the session set up at the statement's
draw, which needs nothing of the record's draw: an accepting run gives `plainVerdictAt` at its live outcome
(`plainVerdictAt_of_run`), and the deployed decoders are right for it (`plain_hdecAt`, `plain_hdecHAt`).

* `plain_first_of_simD`: the plain mixture's first bound from a simulation (`PlainSim`), by `Pr_le_liveMix`.
* `PlainSimWrPAtD`, `PlainSimWD`: a simulator of each prover at `plainVerdictAt`, at `plain_hdecAt` and
  `plain_hdecHAt`.
* `plain_tableBoundθAt_of_simD`, `plain_tableBoundθAt_simD`, `plain_KS_of_simD`: the table bound and KS at the
  deployed plain verifier's trimmed record from `PlainSimWD` alone.
-/

namespace FlockDeploy.KS

local notation "𝔽" => FlockLevel3.GF128

section First

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open FlockSoundness.Discharge.Tables Claims.Bridge.Custody Claims.Bridge.Live
open Lean (Json)

open Classical in
/-- **The plain mixture's first bound at the statement's draw, from a simulation**: where a prover `P'` of the live
session at one table simulates `P` (`PlainSim`), `V false`'s accepting runs with no column off `C` are no likelier
than the average over `u` of `plainVerdictAt` against `P'`'s strategies, at the root of `P`'s first message. An
accepting run gives `plainVerdictAt` at its live outcome whatever draw its record carries (`plainVerdictAt_of_run`),
so with no premise on the record's draw. -/
theorem plain_first_of_simD {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) {M : ℕ}
    (C : Fin M → List 𝔽) (mPts : ℕ) {s₀ : Ins} {spec₀ : SessionSpec} {sts₀ : Array Stmt} {d : Json}
    (hset : setupOf false x = some (s₀, spec₀, sts₀)) (hd : s₀.read.draw = some d)
    {nMax : ℕ} (N : Claims.Msg → ℕ) (dg : ByteArray → ByteArray) (w : Claims.Msg → Wire nMax 1 mPts)
    {τ : Type} [Fintype τ] [Nonempty τ] (P' : Claims.Prover (Ω × τ))
    (Φ : τ × (Fin (N x + 2) → LCoin nMax) → Fin ((V false).k x) → Coin)
    (hsim : PlainSim P x ω mPts N dg w P' Φ) :
    Claims.Pr (fun r : Fin ((V false).k x) → Coin => (Claims.run sha512 P (V false) x ω r).out = .acc ∧
        ¬ ∃ (p : Fin M) (v : List 𝔽),
          Claims.Flock.lookCol (cols false (Claims.run sha512 P (V false) x ω r).seen 0) p.val = some v ∧
            v ≠ C p) ≤
      (∑ u, (prob (plainVerdictAt s₀.read.I d C ((linkOf ((P.next x [] ω).eval sha512).1).getD default).1.toArray)
        (liveSession dg 1 mPts (N x)) (liveStrat (fun _ => 1) (fun _ => mPts) N dg w sha512 P' x (ω, u))).toReal) /
        Fintype.card τ := by
  set E : (Fin ((V false).k x) → Coin) → Prop := fun r => (Claims.run sha512 P (V false) x ω r).out = .acc ∧
    ¬ ∃ (p : Fin M) (v : List 𝔽),
      Claims.Flock.lookCol (cols false (Claims.run sha512 P (V false) x ω r).seen 0) p.val = some v ∧ v ≠ C p
    with hEdef
  obtain ⟨hΦ, hsim⟩ := hsim
  refine Pr_le_liveMix (fun _ => 1) (fun _ => mPts) N dg w sha512 P' x ω
    (plainVerdictAt s₀.read.I d C ((linkOf ((P.next x [] ω).eval sha512).1).getD default).1.toArray) E Φ (hΦ E)
    fun u r' hE => ?_
  obtain ⟨hacc, hno⟩ := hE
  obtain ⟨s, spec, sts, link, c₁, mids, f, hact, hv⟩ := plainVerdictAt_of_run mPts C hacc
  obtain ⟨rfl, -, -⟩ : s = s₀ ∧ spec = spec₀ ∧ sts = sts₀ := by
    have e := (act_decide hact).symm.trans hset
    simp only [Option.some.injEq, Prod.mk.injEq] at e
    exact e
  have hroot : ((linkOf ((P.next x [] ω).eval sha512).1).getD default).1.toArray = link.1.toArray := by
    obtain ⟨rest, hhead⟩ := run_head sha512 P (V false) x ω (Φ (u, r')) hacc
    obtain ⟨rest', -, hlink⟩ := act_decide_link hact
    rw [hhead] at hlink
    simp only [List.headD_cons] at hlink
    rw [hlink]
    rfl
  rw [hroot, hsim u r' hacc s spec sts link c₁ mids f hact]
  exact hv d hd hno

end First

section Chain

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open FlockSoundness.Discharge.Tables Claims.Bridge.Live
open Lean (Json)

open Classical in
/-- **A simulator of each prover at the verdict at the draw**: for each prover `P` and tape `ω`, a live prover `P'`
with a finite tape `τ` and coins `Φ` of `V false` at the tape and the live coins, of uniform law, whose strategies send
`P`'s roots, that simulates `P`'s accepting runs at `Φ` (`PlainSimRun`), whose live self-clashes against
`plainVerdictAt` give `P`'s accepting runs with a collision there (`PlainClashRun`), and whose strategies' fork bound
at each plain fork round `j` is no more than an accepting fork's with a collision there (`plainForkEv`) and the slack
at chances `δ` (`Flock.forkSlack`), at `δ` fixed per prover whose slack weights total at most level 0's `2·2^{d₀}·Q₀`
(`Flock.slackC`, `Flock.slackW`). Uniform functions of the live coins as the tape (`fnAt`, `Pr_fnAt`) are one such
choice. -/
def PlainSimWrPAtD {x : Claims.Msg} {spec : SessionSpec} {st : Stmt} (h : tables false x = [(spec, st)])
    (L : Layout st) (m k0 : ℕ) (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hk0 : l₀.lanes = k0)
    (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts) (I : Exec.Inputs) (d : Json)
    (opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8))
    (hdec : ∀ (C : Fin (wordLen false x) → List 𝔽) (rootB₀ : Array ByteArray) j,
      DecodesTable (fun _ cap pend os => tableOutL Refine.H512 Refine.enc512 Refine.leaf512 cap pend os)
        (fun _ o => o.accepted = true ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Refine.leaf512
          ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (capOfRoot rootB₀[0]!) (Claims.Flock.oracleOf C) o ∧
          ¬ SelfClashL Refine.H512 Refine.enc512 Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ o))
        (plainVerdictAt I d C rootB₀) opn (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay))
        (fun _ _ => true) (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!)
        (fun j => repsOf (D := Refine.Dig) execArith ([plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay].get j))
        j) : Prop :=
  ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Claims.Prover Ω),
    ∃ (δ : ℕ → ℝ) (_ : ∀ t, 0 ≤ δ t)
      (_ : ∑ i, Claims.Flock.slackC ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
          (fun r ℓ (i : Fin ((V false).k x)) =>
            Claims.Flock.slackW ((V false).k x) δ (plainJ (tables_head_one h) hsch r ℓ).val i.val) i ≤
        2 * ((2 : ℝ) ^ l₀.logLen * l₀.queries)),
    ∀ ω : Ω, ∃ (N : Claims.Msg → ℕ) (dg : ByteArray → ByteArray) (w : Claims.Msg → Wire 65536 1 spec.mPts) (τ : Type)
      (_ : Fintype τ) (_ : Nonempty τ) (P' : Claims.Prover (Ω × τ))
      (Φ : τ × (Fin (N x + 2) → LCoin 65536) → Fin ((V false).k x) → Coin)
      (_ : ∀ E : (Fin ((V false).k x) → Coin) → Prop, Claims.Pr (fun p => E (Φ p)) = Claims.Pr E)
      (hσ : ∀ u, (liveStrat (fun _ => 1) (fun _ => spec.mPts) N dg w sha512 P' x (ω, u)).1 = plainRoot P x ω),
      PlainSimRun P x ω spec.mPts N dg w P' Φ ∧
      (∀ C : Fin (wordLen false x) → List 𝔽, (∀ p, wide false x (C p)) →
        PlainClashRun P x ω spec.mPts N dg w P' Φ
          (LiveSelfClash Refine.H512 Refine.enc512 Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
            (plainVerdictAt I d C (plainRoot P x ω)) opn
            (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true) (fun _ _ _ => [])
            (fun rs _ => capOfRoot rs[0]!) fun j => repsOf (D := Refine.Dig) execArith
              ([plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay].get j))) ∧
      (∀ C : Fin (wordLen false x) → List 𝔽, (∀ p, wide false x (C p)) → ∀ r ℓ,
        (∑ u, advRSL execArith Refine.H512 Refine.enc512 Refine.leaf512 (L.model (Refine.regionsOf st)) m
            k0 l₀ l₁ rest ext hk0 hm PT_LOCAL spec.mPts [] [] (capOfRoot (plainRoot P x ω)[0]!) r.val (ℓ.val + 1)
            (simAt execArith Refine.H512 Refine.enc512 Refine.leaf512
              (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) _ (plainVerdictAt I d C (plainRoot P x ω)) opn
              dg (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true)
              (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!) (N x) (hdec C (plainRoot P x ω)) (plainRoot P x ω)
              (liveStrat (fun _ => 1) (fun _ => spec.mPts) N dg w sha512 P' x (ω, u)) (hσ u))) /
            Fintype.card τ ≤
          (Claims.Pr fun s : (Fin ((V false).k x) → Coin) × (Fin ((V false).k x) → Coin) =>
            plainForkEv h hsch P r ℓ ω s.1 s.2) +
            Claims.Flock.forkSlack sha512 (V false) P x δ (plainJ (tables_head_one h) hsch r ℓ).val ω)

open Classical in
/-- **A simulator at the verdict at the draw gives the plain table bound at a statement**: the first bound is
`plain_first_of_simD`'s, so with no premise on the record's draw, the third `liveMix_le_Pr`'s, and the fork bound with
its slack fits the plain weight (`slackEsc_fitsFork_plain`). -/
theorem plain_tableBoundθAt_of_simD {x : Claims.Msg} {ins : Ins} {spec : SessionSpec} {sts : Array Stmt} {st : Stmt}
    {h : tables false x = [(spec, st)]} {L : Layout st} {m k0 : ℕ} {l₀ l₁ : Accounting.Level}
    {rest : List Accounting.Level} {ext : ℕ} {hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩}
    {hk0 : l₀.lanes = k0} {hm : 13 ≤ (L.model (Refine.regionsOf st)).m}
    {hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts} {d : Json}
    {opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8)}
    {hdec : ∀ (C : Fin (wordLen false x) → List 𝔽) (rootB₀ : Array ByteArray) j,
      DecodesTable (fun _ cap pend os => tableOutL Refine.H512 Refine.enc512 Refine.leaf512 cap pend os)
        (fun _ o => o.accepted = true ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Refine.leaf512
          ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (capOfRoot rootB₀[0]!) (Claims.Flock.oracleOf C) o ∧
          ¬ SelfClashL Refine.H512 Refine.enc512 Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ o))
        (plainVerdictAt ins.read.I d C rootB₀) opn (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay))
        (fun _ _ => true) (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!)
        (fun j => repsOf (D := Refine.Dig) execArith ([plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay].get j))
        j}
    (hset : setupOf false x = some (ins, spec, sts)) (hd : ins.read.draw = some d)
    (hnc : ∀ C : Fin (wordLen false x) → List 𝔽, (∀ p, wide false x (C p)) →
      ¬ Claims.Flock.GoodTable (deployPlain Readable.relation) (wordLen false) x C →
      ¬ Committed execArith (L.model (Refine.regionsOf st)) ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (Claims.Flock.oracleOf C))
    (hs : PlainSimWrPAtD h L m k0 l₀ l₁ rest ext hsch hk0 hm hlay ins.read.I d opn hdec) :
    Claims.Flock.TableBoundθAt sha512 (deployPlain Readable.relation) (wordLen false) (wide false) x := by
  intro Ω _ _ P
  obtain ⟨δ, hδ, hfit, hω⟩ := hs Ω P
  refine ⟨Claims.Flock.slackEsc sha512 (V false) P x ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (plainJ (tables_head_one h) hsch)
      (fun r ℓ ω s s' => plainForkEv h hsch P r ℓ ω s s') (plainEv₁ P x) (fun _ _ _ _ _ he => he.2.2)
      (fun _ _ he => he.2)
      (fun r ℓ i => Claims.Flock.slackW ((V false).k x) δ (plainJ (tables_head_one h) hsch r ℓ).val i.val)
      (Claims.Flock.slackG sha512 (V false) P x) (Claims.Flock.slackG_tied sha512 (V false) P x),
    slackEsc_fitsFork_plain Readable.relation P x (tables_head_one h) hsch _ _ _ _ _ _ _ hfit, fun ω C hok hg => ?_⟩
  obtain ⟨N, dg, w, τ, _, _, P', Φ, hΦ, hσ, hrun, hsc, hfork⟩ := hω ω
  have htr := plain_first_of_simD P x ω C spec.mPts hset hd N dg w P' Φ ⟨hΦ, hrun⟩
  have hsc' := liveMix_le_Pr (fun _ => 1) (fun _ => spec.mPts) N dg w sha512 P' x ω _ (plainEv₁ P x ω) Φ (hΦ _)
    (hsc C hok)
  exact plain_le_sessionsE L m k0 l₀ l₁ rest ext hsch hk0 hm PT_LOCAL hlay P _ ω C (hnc C hok hg) τ
    (capOfRoot (plainRoot P x ω)[0]!) (fun u => simAt execArith Refine.H512 Refine.enc512 Refine.leaf512
      (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) _ (plainVerdictAt ins.read.I d C (plainRoot P x ω)) opn
      dg (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true) (fun _ _ _ => [])
      (fun rs _ => capOfRoot rs[0]!) (N x) (hdec C (plainRoot P x ω)) (plainRoot P x ω)
      (liveStrat (fun _ => 1) (fun _ => spec.mPts) N dg w sha512 P' x (ω, u)) (hσ u))
    (le_trans htr (mean_live_le_simAt execArith Refine.H512 Refine.enc512 Refine.leaf512
      (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) _ (plainVerdictAt ins.read.I d C (plainRoot P x ω)) opn
      dg (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true) (fun _ _ _ => [])
      (fun rs _ => capOfRoot rs[0]!) (N x) (hdec C (plainRoot P x ω)) (plainRoot P x ω)
      (fun u => liveStrat (fun _ => 1) (fun _ => spec.mPts) N dg w sha512 P' x (ω, u)) hσ))
    (mean_epsSL_le_slack execArith Refine.H512 Refine.enc512 Refine.leaf512 h L m k0 l₀ l₁ rest ext hsch hk0 hm
      PT_LOCAL [] [] _ _ sha512 (V false) P (plainJ (tables_head_one h) hsch) _ _ _ _ _
      (fun _ _ _ => Claims.Flock.slackW_nonneg hδ _ _ _) _ _ ω
      (fun r ℓ => (hfork C hok r ℓ).trans (add_le_add le_rfl
        (Claims.Flock.forkSlack_le sha512 (V false) P x hδ _ ω)))
      (le_trans (mean_selfClash_le_live execArith Refine.H512 Refine.enc512 Refine.leaf512
        (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) _ (plainVerdictAt ins.read.I d C (plainRoot P x ω)) opn
        dg (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true) (fun _ _ _ => [])
        (fun rs _ => capOfRoot rs[0]!) (N x) (hdec C (plainRoot P x ω)) (plainRoot P x ω)
        (fun u => liveStrat (fun _ => 1) (fun _ => spec.mPts) N dg w sha512 P' x (ω, u)) hσ) hsc'))

/-- **A simulator at a plain statement, at the verdict at the draw**: `PlainSimWrPAtD` at the statements in scope that
set up at a draw, at their one table's `fast100` schedule, with the decoders `plain_hdecAt` where the outputs are
public and `plain_hdecHAt` where they are hidden. -/
def PlainSimWD (x : Claims.Msg) : Prop :=
  ∀ (s : Claims.Flock.Stmt) (hs : parse false Readable.relation x = some s) (hin : Claims.Flock.InScope false s)
    (ins : Ins) (spec : SessionSpec) (sts : Array Stmt) (hset : setupOf false x = some (ins, spec, sts)) (st : Stmt)
    (h : tables false x = [(spec, st)]) (d : Json) (hd : ins.read.draw = some d) (L : Layout st) (m k0 : ℕ)
    (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ)
    (hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hk0 : l₀.lanes = k0)
    (hm : 13 ≤ (L.model (Refine.regionsOf st)).m)
    (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts),
    (∀ hh : ins.read.I.tags.hiddenOutputs = false,
      PlainSimWrPAtD h L m k0 l₀ l₁ rest ext hsch hk0 hm hlay ins.read.I d
        (plainOpn (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d)
        (plain_hdecAt hs hin hset h hh hd L m k0 l₀ l₁ rest ext hsch hm hlay)) ∧
    (∀ hh : ins.read.I.tags.hiddenOutputs = true,
      PlainSimWrPAtD h L m k0 l₀ l₁ rest ext hsch hk0 hm hlay ins.read.I d
        (plainOpnH (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) ins.read.I d)
        (plain_hdecHAt hs hin hset h hh hd L m k0 l₀ l₁ rest ext hsch hm hlay))

/-- **A simulator at the verdict at the draw gives the plain table bound at a statement some run accepts**: such a
statement is in scope with one table (`acc_inScope`, `plain_one`), whose model statement no table off it is committed
to (`plain_not_committed`), so `plain_tableBoundθAt_of_simD` at `PlainSimWD`, with no premise on the record's draw. -/
theorem plain_tableBoundθAt_simD {x : Claims.Msg} (hno : ¬ Claims.Flock.NoAcc sha512 (V false) x)
    (hs : PlainSimWD x) :
    Claims.Flock.TableBoundθAt sha512 (deployPlain Readable.relation) (wordLen false) (wide false) x := by
  obtain ⟨s, hs', hin⟩ := acc_inScope Readable.relation hno
  obtain ⟨ins, spec, sts, d, hset, hd⟩ := acc_draw hno
  have h1 := plain_one hs' hin hset
  have h0 : 0 < sts.size := by omega
  have hm := parse_m hs' hset h0
  have h22 : 22 ≤ (sts[0]'h0).m := by have := hin.1; simp only [Bool.false_eq_true, ↓reduceIte] at this; omega
  have h35 : (sts[0]'h0).m ≤ 35 := by have := hin.2.1; omega
  obtain ⟨L, hlay⟩ := plain_table0 hs' hin hset h0
  obtain ⟨k0, l₀, l₁, rest, ext, hsch, hk0, e⟩ := plain_sched h22 h35
  have ht := tables_of_one hset h1
  have hm13 : 13 ≤ (L.model (Refine.regionsOf (sts[0]'h0))).m := by show 13 ≤ (sts[0]'h0).m; omega
  have hnc : ∀ C : Fin (wordLen false x) → List 𝔽, (∀ p, wide false x (C p)) →
      ¬ Claims.Flock.GoodTable (deployPlain Readable.relation) (wordLen false) x C →
      ¬ Committed execArith (L.model (Refine.regionsOf (sts[0]'h0))) ⟨_, k0, l₀ :: l₁ :: rest, ext⟩
        (Claims.Flock.oracleOf C) := fun C hw hg => by
    have := plain_not_committed hs' hin hset h0 L C hw hg
    rwa [e] at this
  obtain ⟨hw, hhid⟩ := hs s hs' hin ins spec sts hset _ ht d hd L _ k0 l₀ l₁ rest ext hsch hk0 hm13 hlay
  cases hh : ins.read.I.tags.hiddenOutputs
  · exact plain_tableBoundθAt_of_simD hset hd hnc (hw hh)
  · exact plain_tableBoundθAt_of_simD hset hd hnc (hhid hh)

/-- **The deployed plain verifier: KS**, at SHA-512 at the trimmed record at the deployed relation, at the statements
some run accepts from a simulator of each prover at the verdict at the draw (`PlainSimWD`). It asks nothing of the
record's draw, so a prover whose record carries the draw in another tree (`drawEq_ne`) is covered. -/
theorem plain_KS_of_simD (hs : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V false) x → PlainSimWD x) :
    Claims.Flock.KS sha512 (deployPlainTrim Readable.relation) :=
  plain_KS_trim Readable.relation (plain_tableBoundθ_of_acc Readable.relation (wide false)
    fun x _ _ _ _ hno _ _ _ => plain_tableBoundθAt_simD hno (hs x hno))

end Chain

end FlockDeploy.KS
