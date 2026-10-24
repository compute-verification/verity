import Proofs.Flock.Claims.KS.PlainLeafKS
import Proofs.Flock.Claims.KS.PlainLeafKSK
import Proofs.Flock.Claims.KS.ZkAbortR
import Proofs.Flock.Claims.Recursion.BoundDeploy

/-!
# (P1) at the intended choice, with no premise

`Bound.PlainKSAbort` at every element-level verifier, `KSBoundRoute`'s ∀-form, is false where that verifier accepts
everything (`PlainKSAbortFails`), so (P1) stays a premise there. At the intended choice it holds with no premise
(`RecDeploy.plainKSAbort_deployed`): `DE`'s steps, columns and weight the deployed plain verifier's own, where `DE` is
`deployPlainTrim` (`deE_plainTrim`), at the deployed relation.

* **The slack's escapes are on accepted first runs** (`Claims.Flock.slackEsc_onAcc₁`). The plain fork and one-run
  events accept (`plainForkEv`, `plainEv₁`), and a slack event (`Claims.Flock.slackG`) is an accepting fork with a
  collision or, at the last round, an accepting run with one, whose fork need not accept. So the escapes are
  `Claims.Escapes.OnAcc₁`, not `Claims.Escapes.OnAcc`, and their `FitsFork` gives `FitsForkR`.
* **The plain table bound counted per round** (`plain_tableBoundθOnAccR`): the simulator's chain
  (`plain_tableBoundθAt_of_simD`, `plain_tableBoundθAt_of_simDU`, `plain_tableBoundθAt_all`) with those escapes, and
  no escapes where no run accepts.
* **`plain_KSAbortR`**: the plain record meets `Inner.KSAbort` from that bound, through
  `KSAbort_of_tableBoundOnAccR`, as the masked `zk_KSAbortR` does.
* **At `VK keep`** (`plain_tableBoundθOnAccRK`): the plain table bound counted per round at `deployPlainK keep`.
-/

namespace Claims.Flock

/-- **The slack's escapes are on accepted first runs** where the old fork events are on accepted runs and the one-run
event's run accepts: a slack event is an accepting fork with a collision, or an accepting run with one. -/
theorem slackEsc_onAcc₁ {Ω Coin : Type} [Inhabited Coin] (H : Msg → Msg) (V : Verifier Coin) (P : Prover Ω)
    (x : Msg) (sch : FlockSoundness.Accounting.Schedule) (j : Fin 2 → Fin sch.levels.length → Fin (V.k x))
    (ev : Fin 2 → Fin sch.levels.length → Ω → (Fin (V.k x) → Coin) → (Fin (V.k x) → Coin) → Prop)
    (ev₁ : Ω → (Fin (V.k x) → Coin) → Prop)
    (tied : ∀ r ℓ ω s s', ev r ℓ ω s s' →
      TiedColl H ((run H P V x ω s).vLog ++ (run H P V x ω (forkCoins (j r ℓ) s s')).vLog))
    (tied₁ : ∀ ω s, ev₁ ω s → TiedColl H (run H P V x ω s).vLog)
    (w : Fin 2 → Fin sch.levels.length → Fin (V.k x) → ℝ)
    (hacc : ∀ r ℓ ω s s', ev r ℓ ω s s' →
      (run H P V x ω s).out = .acc ∧ (run H P V x ω (forkCoins (j r ℓ) s s')).out = .acc)
    (hacc₁ : ∀ ω s, ev₁ ω s → (run H P V x ω s).out = .acc) :
    (slackEsc H V P x sch j ev ev₁ tied tied₁ w (slackG H V P x) (slackG_tied H V P x)).OnAcc₁ := by
  refine ⟨?_, fun _ ω s h => hacc₁ ω s h⟩
  rintro (a | t) ω s s' h
  · have h' : ev a.1 a.2 ω s s' := h
    exact ⟨(hacc a.1 a.2 ω s s' h').1, Or.inl (hacc a.1 a.2 ω s s' h').2⟩
  · have h' : slackG H V P x t ω s s' := h
    rcases h' with h' | ⟨-, h'⟩
    · exact ⟨h'.1, Or.inl h'.2.1⟩
    · exact ⟨h'.1, Or.inr h'.2⟩

end Claims.Flock

namespace FlockDeploy.KS

local notation "𝔽" => FlockLevel3.GF128

section Chain

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open FlockSoundness.Discharge.Tables Claims.Bridge.Live
open Lean (Json)

open Classical in
/-- **A simulator at the verdict at the draw gives the plain table bound counted per round, on accepted first runs**:
`plain_tableBoundθAt_of_simD`'s escapes and proof, with the escapes on accepted first runs
(`Claims.Flock.slackEsc_onAcc₁`) and their fit counted per round (`Claims.Escapes.FitsFork.fitsForkR`). -/
theorem plain_tableBoundθAccAtR_of_simD {x : Claims.Msg} {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    {st : Stmt} {h : tables false x = [(spec, st)]} {L : Layout st} {m k0 : ℕ} {l₀ l₁ : Accounting.Level}
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
    Claims.Flock.TableBoundθAccAtR sha512 (deployPlain Readable.relation) (wordLen false) (wide false) x := by
  intro Ω _ _ P
  obtain ⟨δ, hδ, hfit, hω⟩ := hs Ω P
  refine ⟨Claims.Flock.slackEsc sha512 (V false) P x ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (plainJ (tables_head_one h) hsch)
      (fun r ℓ ω s s' => plainForkEv h hsch P r ℓ ω s s') (plainEv₁ P x) (fun _ _ _ _ _ he => he.2.2)
      (fun _ _ he => he.2)
      (fun r ℓ i => Claims.Flock.slackW ((V false).k x) δ (plainJ (tables_head_one h) hsch r ℓ).val i.val)
      (Claims.Flock.slackG sha512 (V false) P x) (Claims.Flock.slackG_tied sha512 (V false) P x),
    Claims.Flock.slackEsc_onAcc₁ _ _ _ _ _ _ _ _ _ _ _ (fun _ _ _ _ _ he => ⟨he.1, he.2.1⟩) (fun _ _ he => he.1),
    Claims.Escapes.FitsFork.fitsForkR
      (slackEsc_fitsFork_plain Readable.relation P x (tables_head_one h) hsch _ _ _ _ _ _ _ hfit)
      (Nat.cast_nonneg _) (V_k_pos false x), fun ω C hok hg => ?_⟩
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

open Classical in
/-- **At `leafPlain`**: `plain_tableBoundθAt_of_simDU`'s escapes and proof, on accepted first runs and counted per
round, as `plain_tableBoundθAccAtR_of_simD`. -/
theorem plain_tableBoundθAccAtR_of_simDU {x : Claims.Msg} {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    {st : Stmt} {h : tables false x = [(spec, st)]} {L : Layout st} {m k0 : ℕ} {l₀ l₁ : Accounting.Level}
    {rest : List Accounting.Level} {ext : ℕ} {hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩}
    {hk0 : l₀.lanes = k0} {hm : 13 ≤ (L.model (Refine.regionsOf st)).m}
    {hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts} {d : Json}
    {opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8)}
    {hdec : ∀ (C : Fin (wordLen false x) → List 𝔽) (rootB₀ : Array ByteArray) j,
      DecodesTable (fun _ cap pend os => tableOutL Refine.H512 Refine.enc512 Refine.leafPlain cap pend os)
        (fun _ o => o.accepted = true ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Refine.leafPlain
          ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (capOfRoot rootB₀[0]!) (Claims.Flock.oracleOf C) o ∧
          ¬ SelfClashL Refine.H512 Refine.enc512 Refine.leafPlain ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ o))
        (plainVerdictAt ins.read.I d C rootB₀) opn (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay))
        (fun _ _ => true) (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!)
        (fun j => repsOf (D := Refine.Dig) execArith ([plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay].get j))
        j}
    (hset : setupOf false x = some (ins, spec, sts)) (hd : ins.read.draw = some d)
    (hnc : ∀ C : Fin (wordLen false x) → List 𝔽, (∀ p, wide false x (C p)) →
      ¬ Claims.Flock.GoodTable (deployPlain Readable.relation) (wordLen false) x C →
      ¬ Committed execArith (L.model (Refine.regionsOf st)) ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (Claims.Flock.oracleOf C))
    (hs : PlainSimWrPAtDU h L m k0 l₀ l₁ rest ext hsch hk0 hm hlay ins.read.I d opn hdec) :
    Claims.Flock.TableBoundθAccAtR sha512 (deployPlain Readable.relation) (wordLen false) (wide false) x := by
  intro Ω _ _ P
  obtain ⟨δ, hδ, hfit, hω⟩ := hs Ω P
  refine ⟨Claims.Flock.slackEsc sha512 (V false) P x ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (plainJ (tables_head_one h) hsch)
      (fun r ℓ ω s s' => plainForkEv h hsch P r ℓ ω s s') (plainEv₁ P x) (fun _ _ _ _ _ he => he.2.2)
      (fun _ _ he => he.2)
      (fun r ℓ i => Claims.Flock.slackW ((V false).k x) δ (plainJ (tables_head_one h) hsch r ℓ).val i.val)
      (Claims.Flock.slackG sha512 (V false) P x) (Claims.Flock.slackG_tied sha512 (V false) P x),
    Claims.Flock.slackEsc_onAcc₁ _ _ _ _ _ _ _ _ _ _ _ (fun _ _ _ _ _ he => ⟨he.1, he.2.1⟩) (fun _ _ he => he.1),
    Claims.Escapes.FitsFork.fitsForkR
      (slackEsc_fitsFork_plain Readable.relation P x (tables_head_one h) hsch _ _ _ _ _ _ _ hfit)
      (Nat.cast_nonneg _) (V_k_pos false x), fun ω C hok hg => ?_⟩
  obtain ⟨N, dg, w, τ, _, _, P', Φ, hΦ, hσ, hrun, hsc, hfork⟩ := hω ω
  have htr := plain_first_of_simD P x ω C spec.mPts hset hd N dg w P' Φ ⟨hΦ, hrun⟩
  have hsc' := liveMix_le_Pr (fun _ => 1) (fun _ => spec.mPts) N dg w sha512 P' x ω _ (plainEv₁ P x ω) Φ (hΦ _)
    (hsc C hok)
  exact plain_le_sessionsEU L m k0 l₀ l₁ rest ext hsch hk0 hm PT_LOCAL hlay P _ ω C (hnc C hok hg) τ
    (capOfRoot (plainRoot P x ω)[0]!) (fun u => simAt execArith Refine.H512 Refine.enc512 Refine.leafPlain
      (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) _ (plainVerdictAt ins.read.I d C (plainRoot P x ω)) opn
      dg (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true) (fun _ _ _ => [])
      (fun rs _ => capOfRoot rs[0]!) (N x) (hdec C (plainRoot P x ω)) (plainRoot P x ω)
      (liveStrat (fun _ => 1) (fun _ => spec.mPts) N dg w sha512 P' x (ω, u)) (hσ u))
    (le_trans htr (mean_live_le_simAt execArith Refine.H512 Refine.enc512 Refine.leafPlain
      (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) _ (plainVerdictAt ins.read.I d C (plainRoot P x ω)) opn
      dg (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true) (fun _ _ _ => [])
      (fun rs _ => capOfRoot rs[0]!) (N x) (hdec C (plainRoot P x ω)) (plainRoot P x ω)
      (fun u => liveStrat (fun _ => 1) (fun _ => spec.mPts) N dg w sha512 P' x (ω, u)) hσ))
    (mean_epsSL_le_slack execArith Refine.H512 Refine.enc512 Refine.leafPlain h L m k0 l₀ l₁ rest ext hsch hk0 hm
      PT_LOCAL [] [] _ _ sha512 (V false) P (plainJ (tables_head_one h) hsch) _ _ _ _ _
      (fun _ _ _ => Claims.Flock.slackW_nonneg hδ _ _ _) _ _ ω
      (fun r ℓ => (hfork C hok r ℓ).trans (add_le_add le_rfl
        (Claims.Flock.forkSlack_le sha512 (V false) P x hδ _ ω)))
      (le_trans (mean_selfClash_le_live execArith Refine.H512 Refine.enc512 Refine.leafPlain
        (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) _ (plainVerdictAt ins.read.I d C (plainRoot P x ω)) opn
        dg (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true) (fun _ _ _ => [])
        (fun rs _ => capOfRoot rs[0]!) (N x) (hdec C (plainRoot P x ω)) (plainRoot P x ω)
        (fun u => liveStrat (fun _ => 1) (fun _ => spec.mPts) N dg w sha512 P' x (ω, u)) hσ) hsc'))

/-- **The plain table bound counted per round, on accepted first runs, at every statement some run accepts**:
`plain_tableBoundθAt_all`'s cases at `plain_tableBoundθAccAtR_of_simD` and `plain_tableBoundθAccAtR_of_simDU`. -/
theorem plain_tableBoundθAccAtR_all {x : Claims.Msg} (hno : ¬ Claims.Flock.NoAcc sha512 (V false) x) :
    Claims.Flock.TableBoundθAccAtR sha512 (deployPlain Readable.relation) (wordLen false) (wide false) x := by
  obtain ⟨s, hs, hin⟩ := acc_inScope Readable.relation hno
  obtain ⟨ins, spec, sts, d, hset, hd⟩ := acc_draw hno
  have h1 := plain_one hs hin hset
  have h0 : 0 < sts.size := by omega
  have hm := parse_m hs hset h0
  have h22 : 22 ≤ (sts[0]'h0).m := by have := hin.1; simp only [Bool.false_eq_true, ↓reduceIte] at this; omega
  have h35 : (sts[0]'h0).m ≤ 35 := by have := hin.2.1; omega
  obtain ⟨L, hlay⟩ := plain_table0 hs hin hset h0
  obtain ⟨k0, l₀, l₁, rest, ext, hsch, hk0, e⟩ := plain_sched h22 h35
  have ht := tables_of_one hset h1
  have hm13 : 13 ≤ (L.model (Refine.regionsOf (sts[0]'h0))).m := by show 13 ≤ (sts[0]'h0).m; omega
  have hnc : ∀ C : Fin (wordLen false x) → List 𝔽, (∀ p, wide false x (C p)) →
      ¬ Claims.Flock.GoodTable (deployPlain Readable.relation) (wordLen false) x C →
      ¬ Committed execArith (L.model (Refine.regionsOf (sts[0]'h0))) ⟨_, k0, l₀ :: l₁ :: rest, ext⟩
        (Claims.Flock.oracleOf C) := fun C hw hg => by
    have := plain_not_committed hs hin hset h0 L C hw hg
    rwa [e] at this
  obtain ⟨hw, hhid⟩ := plainSimWD_all x s hs hin ins spec sts hset _ ht d hd L _ k0 l₀ l₁ rest ext hsch hk0 hm13 hlay
  cases hh : ins.read.I.tags.hiddenOutputs
  · exact plain_tableBoundθAccAtR_of_simD hset hd hnc (hw hh)
  · by_cases hl : ins.read.I.tags.merkleLeaf = "hm96-sha512/v1"
    · exact plain_tableBoundθAccAtR_of_simD hset hd hnc (hhid hh hl)
    · exact plain_tableBoundθAccAtR_of_simDU hset hd hnc (plainSimWrPAtDU_at hs hin hset ht hh
        (plain_leafU hs hin hset hl) hd L _ k0 l₀ l₁ rest ext hsch hk0 hm13 hlay)

end Chain

open Flock in
/-- **The plain table bound counted per round, on accepted first runs** (`Claims.Recursion.TableBoundθOnAccR`), at
the deployed relation, with no hypothesis: `plain_tableBoundθAccAtR_all` where some run accepts, and no escapes where
none does (`Claims.Flock.tableBoundθAccAt_of_noAcc`). -/
theorem plain_tableBoundθOnAccR :
    Claims.Recursion.TableBoundθOnAccR sha512 (deployPlain Readable.relation) (wordLen false) (wide false) := by
  intro x
  by_cases hno : Claims.Flock.NoAcc sha512 (V false) x
  · exact Claims.Flock.TableBoundθAccAt.tableBoundθAccAtR
      (Claims.Flock.tableBoundθAccAt_of_noAcc sha512 (deployPlain Readable.relation) (wordLen false) (wide false) x
        hno (plain_k_le Readable.relation x) (εtable_nonneg false x)) (V_k_pos false x)
  · exact plain_tableBoundθAccAtR_all hno

open Flock Claims.Recursion in
/-- **The plain record the claims read meets `Inner.KSAbort`**, at SHA-512, at every decoder within the radius, from
the table bound counted per round over full-width tables at the untrimmed instance: `plain_KSAbort` at
`TableBoundθOnAccR`, as `zk_KSAbortR` (one table wherever the verifier accepts some run is `plain_hone`). -/
theorem plain_KSAbortR (rel : Readable.Parsed → Readable.Relation)
    (htab : TableBoundθOnAccR sha512 (deployPlain rel) (wordLen false) (wide false))
    (decode : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace)
    (hdec : decOK (deployPlainTrim rel) decode) : (Inner.ofFlock (deployPlainTrim rel) decode).KSAbort sha512 :=
  KSAbort_of_tableBoundOnAccR sha512 (deployPlainTrim rel) (wordLen false) (trimWide false) decCoin
    (wordLen_le_N₀ false) (fun x => εtable_nonneg false x) (V_k_pos false) (fun _ => cap_pos)
    (plain_kSplits rel (plain_hone rel)) (plain_hone rel) (trimWide_nil false)
    (Claims.Flock.colsOk_mapCols sha512 (deployPlain rel) (wordLen false) trimCol (wide false)
      (plain_colsOk sha512 rel))
    (Claims.Flock.conflictTied_mapCols sha512 (deployPlain rel) (wordLen false) trimCol (plain_conflictTied rel))
    (tableBoundθOnAccR_mapCols sha512 (deployPlain rel) (wordLen false) trimCol rfl (wide false)
      (plain_colsOk sha512 rel) (wide_trimCol_inj false) htab)
    (coinsShown_V sha512 false) (accLast_V_sha512 false) (decideFree_V sha512 false) (refusable_V sha512 false)
    decode hdec

/-! ## At `VK keep`

The plain table bound counted per round at `deployPlainK keep`: the chain above over `VK keep`, the simulator at
`plainVerdictAtK keep` (`plainSimWD_allK`, `plainSimWrPAtDU_atK`) and the slack's fit there
(`slackEsc_fitsFork_plainK`). -/

section ChainK

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open FlockSoundness.Discharge.Tables Claims.Bridge.Live
open Lean (Json)

open Classical in
/-- `plain_tableBoundθAccAtR_of_simD` at `VK keep`. -/
theorem plain_tableBoundθAccAtR_of_simDK (keep : Bool → Bool) {x : Claims.Msg} {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    {st : Stmt} {h : tables false x = [(spec, st)]} {L : Layout st} {m k0 : ℕ} {l₀ l₁ : Accounting.Level}
    {rest : List Accounting.Level} {ext : ℕ} {hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩}
    {hk0 : l₀.lanes = k0} {hm : 13 ≤ (L.model (Refine.regionsOf st)).m}
    {hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts} {d : Json}
    {opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8)}
    {hdec : ∀ (C : Fin (wordLen false x) → List 𝔽) (rootB₀ : Array ByteArray) j,
      DecodesTable (fun _ cap pend os => tableOutL Refine.H512 Refine.enc512 Refine.leaf512 cap pend os)
        (fun _ o => o.accepted = true ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Refine.leaf512
          ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (capOfRoot rootB₀[0]!) (Claims.Flock.oracleOf C) o ∧
          ¬ SelfClashL Refine.H512 Refine.enc512 Refine.leaf512 ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ o))
        (plainVerdictAtK keep ins.read.I d C rootB₀) opn (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay))
        (fun _ _ => true) (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!)
        (fun j => repsOf (D := Refine.Dig) execArith ([plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay].get j))
        j}
    (hset : setupOf false x = some (ins, spec, sts)) (hd : ins.read.draw = some d)
    (hnc : ∀ C : Fin (wordLen false x) → List 𝔽, (∀ p, wide false x (C p)) →
      ¬ Claims.Flock.GoodTable (deployPlainK keep Readable.relation) (wordLen false) x C →
      ¬ Committed execArith (L.model (Refine.regionsOf st)) ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (Claims.Flock.oracleOf C))
    (hs : PlainSimWrPAtDK keep h L m k0 l₀ l₁ rest ext hsch hk0 hm hlay ins.read.I d opn hdec) :
    Claims.Flock.TableBoundθAccAtR sha512 (deployPlainK keep Readable.relation) (wordLen false) (wide false) x := by
  intro Ω _ _ P
  obtain ⟨δ, hδ, hfit, hω⟩ := hs Ω P
  refine ⟨Claims.Flock.slackEsc sha512 (VK keep) P x ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (plainJ (x := x) (tables_head_one h) hsch)
      (fun r ℓ ω s s' => plainForkEvK keep h hsch P r ℓ ω s s') (plainEv₁K keep P x) (fun _ _ _ _ _ he => he.2.2)
      (fun _ _ he => he.2)
      (fun r ℓ i => Claims.Flock.slackW ((V false).k x) δ (plainJ (x := x) (tables_head_one h) hsch r ℓ).val i.val)
      (Claims.Flock.slackG sha512 (VK keep) P x) (Claims.Flock.slackG_tied sha512 (VK keep) P x),
    Claims.Flock.slackEsc_onAcc₁ _ _ _ _ _ _ _ _ _ _ _ (fun _ _ _ _ _ he => ⟨he.1, he.2.1⟩) (fun _ _ he => he.1),
    Claims.Escapes.FitsFork.fitsForkR
      (slackEsc_fitsFork_plainK keep Readable.relation P x (tables_head_one h) hsch _ _ _ _ _ _ _ hfit)
      (Nat.cast_nonneg _) (VK_k_pos keep x), fun ω C hok hg => ?_⟩
  obtain ⟨N, dg, w, τ, _, _, P', Φ, hΦ, hσ, hrun, hsc, hfork⟩ := hω ω
  have htr := plain_first_of_simDK keep P x ω C spec.mPts hset hd N dg w P' Φ ⟨hΦ, hrun⟩
  have hsc' := liveMix_le_Pr (fun _ => 1) (fun _ => spec.mPts) N dg w sha512 P' x ω _ (plainEv₁K keep P x ω) Φ (hΦ _)
    (hsc C hok)
  exact plain_le_sessionsEV (VK keep) L m k0 l₀ l₁ rest ext hsch hk0 hm PT_LOCAL hlay P _ ω C (hnc C hok hg) τ
    (capOfRoot (plainRoot P x ω)[0]!) (fun u => simAt execArith Refine.H512 Refine.enc512 Refine.leaf512
      (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) _ (plainVerdictAtK keep ins.read.I d C (plainRoot P x ω)) opn
      dg (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true) (fun _ _ _ => [])
      (fun rs _ => capOfRoot rs[0]!) (N x) (hdec C (plainRoot P x ω)) (plainRoot P x ω)
      (liveStrat (fun _ => 1) (fun _ => spec.mPts) N dg w sha512 P' x (ω, u)) (hσ u))
    (le_trans htr (mean_live_le_simAt execArith Refine.H512 Refine.enc512 Refine.leaf512
      (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) _ (plainVerdictAtK keep ins.read.I d C (plainRoot P x ω)) opn
      dg (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true) (fun _ _ _ => [])
      (fun rs _ => capOfRoot rs[0]!) (N x) (hdec C (plainRoot P x ω)) (plainRoot P x ω)
      (fun u => liveStrat (fun _ => 1) (fun _ => spec.mPts) N dg w sha512 P' x (ω, u)) hσ))
    (mean_epsSL_le_slack execArith Refine.H512 Refine.enc512 Refine.leaf512 h L m k0 l₀ l₁ rest ext hsch hk0 hm
      PT_LOCAL [] [] _ _ sha512 (VK keep) P (plainJ (x := x) (tables_head_one h) hsch) _ _ _ _ _
      (fun _ _ _ => Claims.Flock.slackW_nonneg hδ _ _ _) _ _ ω
      (fun r ℓ => (hfork C hok r ℓ).trans (add_le_add le_rfl
        (Claims.Flock.forkSlack_le sha512 (VK keep) P x hδ _ ω)))
      (le_trans (mean_selfClash_le_live execArith Refine.H512 Refine.enc512 Refine.leaf512
        (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) _ (plainVerdictAtK keep ins.read.I d C (plainRoot P x ω)) opn
        dg (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true) (fun _ _ _ => [])
        (fun rs _ => capOfRoot rs[0]!) (N x) (hdec C (plainRoot P x ω)) (plainRoot P x ω)
        (fun u => liveStrat (fun _ => 1) (fun _ => spec.mPts) N dg w sha512 P' x (ω, u)) hσ) hsc'))

open Classical in
/-- `plain_tableBoundθAccAtR_of_simDU` at `VK keep`. -/
theorem plain_tableBoundθAccAtR_of_simDUK (keep : Bool → Bool) {x : Claims.Msg} {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    {st : Stmt} {h : tables false x = [(spec, st)]} {L : Layout st} {m k0 : ℕ} {l₀ l₁ : Accounting.Level}
    {rest : List Accounting.Level} {ext : ℕ} {hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩}
    {hk0 : l₀.lanes = k0} {hm : 13 ≤ (L.model (Refine.regionsOf st)).m}
    {hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL spec.mPts} {d : Json}
    {opn : Array ByteArray → Fin 1 → Model.Opens 𝔽 Refine.Dig × Model.Salts (List UInt8)}
    {hdec : ∀ (C : Fin (wordLen false x) → List 𝔽) (rootB₀ : Array ByteArray) j,
      DecodesTable (fun _ cap pend os => tableOutL Refine.H512 Refine.enc512 Refine.leafPlain cap pend os)
        (fun _ o => o.accepted = true ∧ ¬ (Off₀L Refine.H512 Refine.enc512 Refine.leafPlain
          ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (capOfRoot rootB₀[0]!) (Claims.Flock.oracleOf C) o ∧
          ¬ SelfClashL Refine.H512 Refine.enc512 Refine.leafPlain ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ o))
        (plainVerdictAtK keep ins.read.I d C rootB₀) opn (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay))
        (fun _ _ => true) (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!)
        (fun j => repsOf (D := Refine.Dig) execArith ([plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay].get j))
        j}
    (hset : setupOf false x = some (ins, spec, sts)) (hd : ins.read.draw = some d)
    (hnc : ∀ C : Fin (wordLen false x) → List 𝔽, (∀ p, wide false x (C p)) →
      ¬ Claims.Flock.GoodTable (deployPlainK keep Readable.relation) (wordLen false) x C →
      ¬ Committed execArith (L.model (Refine.regionsOf st)) ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (Claims.Flock.oracleOf C))
    (hs : PlainSimWrPAtDUK keep h L m k0 l₀ l₁ rest ext hsch hk0 hm hlay ins.read.I d opn hdec) :
    Claims.Flock.TableBoundθAccAtR sha512 (deployPlainK keep Readable.relation) (wordLen false) (wide false) x := by
  intro Ω _ _ P
  obtain ⟨δ, hδ, hfit, hω⟩ := hs Ω P
  refine ⟨Claims.Flock.slackEsc sha512 (VK keep) P x ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (plainJ (x := x) (tables_head_one h) hsch)
      (fun r ℓ ω s s' => plainForkEvK keep h hsch P r ℓ ω s s') (plainEv₁K keep P x) (fun _ _ _ _ _ he => he.2.2)
      (fun _ _ he => he.2)
      (fun r ℓ i => Claims.Flock.slackW ((V false).k x) δ (plainJ (x := x) (tables_head_one h) hsch r ℓ).val i.val)
      (Claims.Flock.slackG sha512 (VK keep) P x) (Claims.Flock.slackG_tied sha512 (VK keep) P x),
    Claims.Flock.slackEsc_onAcc₁ _ _ _ _ _ _ _ _ _ _ _ (fun _ _ _ _ _ he => ⟨he.1, he.2.1⟩) (fun _ _ he => he.1),
    Claims.Escapes.FitsFork.fitsForkR
      (slackEsc_fitsFork_plainK keep Readable.relation P x (tables_head_one h) hsch _ _ _ _ _ _ _ hfit)
      (Nat.cast_nonneg _) (VK_k_pos keep x), fun ω C hok hg => ?_⟩
  obtain ⟨N, dg, w, τ, _, _, P', Φ, hΦ, hσ, hrun, hsc, hfork⟩ := hω ω
  have htr := plain_first_of_simDK keep P x ω C spec.mPts hset hd N dg w P' Φ ⟨hΦ, hrun⟩
  have hsc' := liveMix_le_Pr (fun _ => 1) (fun _ => spec.mPts) N dg w sha512 P' x ω _ (plainEv₁K keep P x ω) Φ (hΦ _)
    (hsc C hok)
  exact plain_le_sessionsEUK keep L m k0 l₀ l₁ rest ext hsch hk0 hm PT_LOCAL hlay P _ ω C (hnc C hok hg) τ
    (capOfRoot (plainRoot P x ω)[0]!) (fun u => simAt execArith Refine.H512 Refine.enc512 Refine.leafPlain
      (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) _ (plainVerdictAtK keep ins.read.I d C (plainRoot P x ω)) opn
      dg (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true) (fun _ _ _ => [])
      (fun rs _ => capOfRoot rs[0]!) (N x) (hdec C (plainRoot P x ω)) (plainRoot P x ω)
      (liveStrat (fun _ => 1) (fun _ => spec.mPts) N dg w sha512 P' x (ω, u)) (hσ u))
    (le_trans htr (mean_live_le_simAt execArith Refine.H512 Refine.enc512 Refine.leafPlain
      (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) _ (plainVerdictAtK keep ins.read.I d C (plainRoot P x ω)) opn
      dg (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true) (fun _ _ _ => [])
      (fun rs _ => capOfRoot rs[0]!) (N x) (hdec C (plainRoot P x ω)) (plainRoot P x ω)
      (fun u => liveStrat (fun _ => 1) (fun _ => spec.mPts) N dg w sha512 P' x (ω, u)) hσ))
    (mean_epsSL_le_slack execArith Refine.H512 Refine.enc512 Refine.leafPlain h L m k0 l₀ l₁ rest ext hsch hk0 hm
      PT_LOCAL [] [] _ _ sha512 (VK keep) P (plainJ (x := x) (tables_head_one h) hsch) _ _ _ _ _
      (fun _ _ _ => Claims.Flock.slackW_nonneg hδ _ _ _) _ _ ω
      (fun r ℓ => (hfork C hok r ℓ).trans (add_le_add le_rfl
        (Claims.Flock.forkSlack_le sha512 (VK keep) P x hδ _ ω)))
      (le_trans (mean_selfClash_le_live execArith Refine.H512 Refine.enc512 Refine.leafPlain
        (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay) _ (plainVerdictAtK keep ins.read.I d C (plainRoot P x ω)) opn
        dg (plainDec (plainTab L m k0 l₀ l₁ rest ext hsch hm PT_LOCAL hlay)) (fun _ _ => true) (fun _ _ _ => [])
        (fun rs _ => capOfRoot rs[0]!) (N x) (hdec C (plainRoot P x ω)) (plainRoot P x ω)
        (fun u => liveStrat (fun _ => 1) (fun _ => spec.mPts) N dg w sha512 P' x (ω, u)) hσ) hsc'))

/-- `plain_tableBoundθAccAtR_all` at `VK keep`. -/
theorem plain_tableBoundθAccAtR_allK (keep : Bool → Bool) {x : Claims.Msg} (hno : ¬ Claims.Flock.NoAcc sha512 (VK keep) x) :
    Claims.Flock.TableBoundθAccAtR sha512 (deployPlainK keep Readable.relation) (wordLen false) (wide false) x := by
  obtain ⟨s, hs, hin⟩ := accK_inScope keep Readable.relation hno
  obtain ⟨ins, spec, sts, d, hset, hd⟩ := accK_draw keep hno
  have h1 := plain_one hs hin hset
  have h0 : 0 < sts.size := by omega
  have hm := parse_m hs hset h0
  have h22 : 22 ≤ (sts[0]'h0).m := by have := hin.1; simp only [Bool.false_eq_true, ↓reduceIte] at this; omega
  have h35 : (sts[0]'h0).m ≤ 35 := by have := hin.2.1; omega
  obtain ⟨L, hlay⟩ := plain_table0 hs hin hset h0
  obtain ⟨k0, l₀, l₁, rest, ext, hsch, hk0, e⟩ := plain_sched h22 h35
  have ht := tables_of_one hset h1
  have hm13 : 13 ≤ (L.model (Refine.regionsOf (sts[0]'h0))).m := by show 13 ≤ (sts[0]'h0).m; omega
  have hnc : ∀ C : Fin (wordLen false x) → List 𝔽, (∀ p, wide false x (C p)) →
      ¬ Claims.Flock.GoodTable (deployPlainK keep Readable.relation) (wordLen false) x C →
      ¬ Committed execArith (L.model (Refine.regionsOf (sts[0]'h0))) ⟨_, k0, l₀ :: l₁ :: rest, ext⟩
        (Claims.Flock.oracleOf C) := fun C hw hg => by
    have := plain_not_committed hs hin hset h0 L C hw fun hg' =>
      hg ((goodTable_plainK keep Readable.relation (wordLen false) x C).2 hg')
    rwa [e] at this
  obtain ⟨hw, hhid⟩ := plainSimWD_allK keep x s hs hin ins spec sts hset _ ht d hd L _ k0 l₀ l₁ rest ext hsch hk0 hm13 hlay
  cases hh : ins.read.I.tags.hiddenOutputs
  · exact plain_tableBoundθAccAtR_of_simDK keep hset hd hnc (hw hh)
  · by_cases hl : ins.read.I.tags.merkleLeaf = "hm96-sha512/v1"
    · exact plain_tableBoundθAccAtR_of_simDK keep hset hd hnc (hhid hh hl)
    · exact plain_tableBoundθAccAtR_of_simDUK keep hset hd hnc (plainSimWrPAtDU_atK keep hs hin hset ht hh
        (plain_leafU hs hin hset hl) hd L _ k0 l₀ l₁ rest ext hsch hk0 hm13 hlay)

end ChainK

open Flock in
/-- `plain_tableBoundθOnAccR` at `VK keep`. -/
theorem plain_tableBoundθOnAccRK (keep : Bool → Bool) :
    Claims.Recursion.TableBoundθOnAccR sha512 (deployPlainK keep Readable.relation) (wordLen false) (wide false) := by
  intro x
  by_cases hno : Claims.Flock.NoAcc sha512 (VK keep) x
  · exact Claims.Flock.TableBoundθAccAt.tableBoundθAccAtR
      (Claims.Flock.tableBoundθAccAt_of_noAcc sha512 (deployPlainK keep Readable.relation) (wordLen false) (wide false) x
        hno (plain_k_leK keep Readable.relation x) (εtable_nonneg false x)) (VK_k_pos keep x)
  · exact plain_tableBoundθAccAtR_allK keep hno

end FlockDeploy.KS

namespace RecDeploy

open Claims (Msg Out)
open FlockDeploy (Coin)

/-- **(P1) at the intended choice, with no premise**: at the deployed bound record whose element-level verifier's
steps, columns and weight are the deployed plain verifier's own, so that `DE` is the trimmed plain record
(`deE_plainTrim`), at the deployed relation, whatever the bound record's other choices: `plain_KSAbortR` at
`plain_tableBoundθOnAccR`. -/
theorem plainKSAbort_deployed (join : List Msg → Msg) (rowIn : Msg → Msg)
    (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) :
    (bdDeployed _root_.Flock.Readable.relation (FlockDeploy.V false).next
      (FlockDeploy.deployPlainTrim _root_.Flock.Readable.relation).cols
      (FlockDeploy.deployPlainTrim _root_.Flock.Readable.relation).weight join rowIn final w).PlainKSAbort
      FlockDeploy.sha512 :=
  fun decI hdec =>
    FlockDeploy.KS.plain_KSAbortR _root_.Flock.Readable.relation FlockDeploy.KS.plain_tableBoundθOnAccR decI hdec

end RecDeploy
