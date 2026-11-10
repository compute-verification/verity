module

public import Definitions.Claims.SamplingGeneric
public import Proofs.Flock.Claims.Sampling.Deployed
public import Proofs.Flock.Claims.Sampling.SessionTo

/-!
# The sampled compiler at a frozen sampled record

`ofDeploy D iroots`: a frozen sampled record `D` (`Claims.Sampling.Deploy`) as a compiled sampled verifier over Flock
at its deployed decoder (`Compiled _ (Arg.ofFlock D.flock)`), at the roots a run shows for a proof block's interior
values (`iroots`, which the frozen record leaves to its hypotheses). Its verifier, parser, scope check, partition,
commitment, sample layout (`stage2`, `rounds`), sessions, reveal, opening count, reference prover and completeness error
are `D`'s; its sample is `D`'s checked units' first `n_PU` (`checked`: the slots the frozen hypotheses read); a proof
block's shape is the record's inner waits, truncation and `N₀` and its session's rounds (`sh`); and the bounds on its
session's errors are `κ_F` and Flock's `γ` at those rounds and the record's weight (`κb`, `γb`). It is
`ofDeployAt D iroots n₀` at the record's `N₀`: `ofDeployAt` gives each unit's proof block its own `N₀`, `n₀ x u`, in
its shape and in its `κ_F` (`κFAt`), for a two-stage record whose units' sessions set up tables of different shapes.
At every `D`:

* its proof blocks and scope are `D`'s (`ofDeploy_nPU`, `ofDeploy_scope`); both waits' unions have `n_PU` bits, so each
  wait's `k` and `K` are `D`'s (`ofDeploy_kOut`, `ofDeploy_kIn`, `ofDeploy_capOut`, `ofDeploy_capIn`); and its opening
  count is `D`'s (`openCallsLe_ofDeploy`, from `Sampling.OpenCallsLe`);
* **it meets the contract** where `D` meets the frozen hypotheses (`sessions_ofDeploy`): `draws` from
  `DrawLaw`, `slots` by the cut, `unions` by its bits, `refuses` from `Refuses`, `interior` from `InteriorFixed`,
  `fixed` from `DrawFixed` (`fixed_ofDeployAt`), `runs` from `SessionRuns` one way (`SessionRunsTo`) and the inner
  numbers at each unit's `N₀` (`InnerNumbersAt`, `runs_ofDeployAt`; the session's embedding is `emb`, the only order
  embedding onto its rounds), and `opens` from `SessionOpens`.

`SamplingDeploy.compiled F c iroots` is `ofDeploy` at the deployed record `deployOf F c`: its sample, rounds,
sessions, reveal, interior roots and commitment are the code's (`compiled_checked`, …, `rfl`), and its opening count
is bounded by the reveal count (`openCallsLe_compiled`). `SamplingDeploy.compiledAt F c iroots` is `ofDeployAt` there
at the code's `unitN₀`, the two-stage compiler's, and `compiled` at a code that sets no `unitN₀`. The four contract
clauses that are facts about the sampled verifier's code alone are named Props of `SamplingDeploy.Code`'s functions
(`Code.DrawLaw`, `Code.Refuses`, `Code.InteriorFixed`, `Code.DrawFixed`), each the frozen hypothesis at `deployOf F c`
(`drawLaw_deployOf`, …, `Iff.rfl`), and `sessions_compiled` reads them.
-/

@[expose] public section

namespace Claims.SamplingGeneric

open Claims
open Claims.Sampling (InScope forkOn)

theorem foldr_max_le {M : ℝ} (h0 : 0 ≤ M) : ∀ {l : List ℝ}, (∀ a ∈ l, a ≤ M) → l.foldr max 0 ≤ M
  | [], _ => h0
  | a :: l, h => by
    rw [List.foldr_cons]
    exact max_le (h a (by simp)) (foldr_max_le h0 fun b hb => h b (by simp [hb]))

theorem maxBelow_le {n : ℕ} {f : ℕ → ℝ} {M : ℝ} (h0 : 0 ≤ M) (h : ∀ u < n, f u ≤ M) : maxBelow n f ≤ M :=
  foldr_max_le h0 fun a ha => by
    obtain ⟨u, hu, rfl⟩ := List.mem_map.1 ha
    exact h u (List.mem_range.1 hu)

/-- A member of a list's first `n` is at one of its first `n` slots. -/
theorem mem_take_iff_slot {l : List ℕ} {n u : ℕ} : u ∈ l.take n ↔ ∃ p : Fin n, l[p.val]? = some u := by
  rw [List.mem_iff_getElem?]
  constructor
  · rintro ⟨i, hi⟩
    by_cases hin : i < n
    · exact ⟨⟨i, hin⟩, by rwa [List.getElem?_take_of_lt hin] at hi⟩
    · simp [hin] at hi
  · rintro ⟨p, hp⟩
    exact ⟨p.val, by rw [List.getElem?_take_of_lt p.isLt]; exact hp⟩

section OfDeploy

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin]

/-- `κ_F` at `N₀ = n`: the sessions' knowledge error at the record's inner numbers with `N₀` set to `n`. -/
noncomputable def κFAt (D : Claims.Sampling.Deploy Coin Col T) (x : Msg) (n : ℝ) : ℝ :=
  Claims.Sampling.κF { D with innerN₀ := fun _ => n } x

/-- **A frozen sampled record as a compiled sampled verifier over Flock, at each unit's `N₀`**, at its deployed decoder
and the roots a run shows for a proof block's interior values: `D`'s verifier, parser, scope check, partition,
commitment, sample layout, sessions, reveal, opening count, reference prover and completeness error; its sample the
checked units' first `n_PU`; a proof block's shape the record's inner waits and truncation, its unit's `N₀` (`n₀ x u`)
and its session's rounds; the bounds on its session's errors `κ_F` at its unit's `N₀` and Flock's `γ` at those rounds
and the record's weight. -/
noncomputable def ofDeployAt (D : Claims.Sampling.Deploy Coin Col T) (iroots : Msg → Seen → ℕ → List Msg)
    (n₀ : Msg → ℕ → ℝ) : Compiled Coin (Arg.ofFlock D.flock) where
  V := D.V
  parse := D.parse
  scopeCheck := D.scopeCheck
  partition := D.partition
  commit := D.commit
  stage2 := D.stage2
  checked x r := (D.checked x r).take (Claims.Sampling.nPU D x)
  sh x u := (D.innerWaits x, D.innerCap x, (D.rounds x u).card, n₀ x u)
  κb x u := κFAt D x (n₀ x u)
  γb x u := 16 * (Real.sqrt (2 * ((D.rounds x u).card : ℝ) * D.weight x) + (D.rounds x u).card) ^ 2
  rounds := D.rounds
  session := D.session
  iroots := iroots
  reveal := D.reveal
  openCalls := D.openCalls
  nOut x := Claims.Sampling.nPU D x
  nIn x := Claims.Sampling.nPU D x
  Tape := D.Tape
  P := D.P
  δc := D.δc

/-- **A frozen sampled record as a compiled sampled verifier over Flock**: `ofDeployAt` with every unit at the record's
`N₀`, so a proof block's bound on its session's knowledge error is `κ_F`. -/
noncomputable abbrev ofDeploy (D : Claims.Sampling.Deploy Coin Col T) (iroots : Msg → Seen → ℕ → List Msg) :
    Compiled Coin (Arg.ofFlock D.flock) :=
  ofDeployAt D iroots fun x _ => D.innerN₀ x

variable (D : Claims.Sampling.Deploy Coin Col T) (iroots : Msg → Seen → ℕ → List Msg) (n₀ : Msg → ℕ → ℝ)

omit [Fintype Coin] [Nonempty Coin] in
theorem κFAt_innerN₀ (x : Msg) : κFAt D x (D.innerN₀ x) = Claims.Sampling.κF D x := rfl

omit [Fintype Coin] [Nonempty Coin] in
theorem κFAt_eq (x : Msg) (n : ℝ) : κFAt D x n = 4 * D.εtable x + 2 ^ (-(D.innerWaits x : ℝ)) +
    2 * (⌈4 * n / Real.exp 1⌉₊ : ℝ) / D.innerCap x + Real.exp (-(⌈4 * n / Real.exp 1⌉₊ : ℝ) / 8) := rfl

omit [Fintype Coin] [Nonempty Coin] in
theorem ofDeployAt_nPU (x : Msg) : (ofDeployAt D iroots n₀).nPU x = Claims.Sampling.nPU D x := rfl

omit [Fintype Coin] [Nonempty Coin] in
theorem ofDeployAt_nSess (x : Msg) : (ofDeployAt D iroots n₀).nSess x = Claims.Sampling.nPU D x := rfl

omit [Fintype Coin] [Nonempty Coin] in
theorem ofDeploy_nPU (x : Msg) : (ofDeploy D iroots).nPU x = Claims.Sampling.nPU D x := rfl

omit [Fintype Coin] [Nonempty Coin] in
theorem ofDeploy_nSess (x : Msg) : (ofDeploy D iroots).nSess x = Claims.Sampling.nPU D x := rfl

omit [Fintype Coin] [Nonempty Coin] in
theorem ofDeploy_kOut (x : Msg) : (ofDeploy D iroots).kOut x = D.k x := rfl

omit [Fintype Coin] [Nonempty Coin] in
theorem ofDeploy_kIn (x : Msg) : (ofDeploy D iroots).kIn x = D.k x := rfl

omit [Fintype Coin] [Nonempty Coin] in
theorem ofDeploy_capOut (x : Msg) : (ofDeploy D iroots).capOut x = D.cap x := rfl

omit [Fintype Coin] [Nonempty Coin] in
theorem ofDeploy_capIn (x : Msg) : (ofDeploy D iroots).capIn x = D.cap x := rfl

omit [Fintype Coin] [Nonempty Coin] in
theorem ofDeploy_scope (x : Msg) : (ofDeploy D iroots).Scope x ↔ Claims.Sampling.Scope D x := Iff.rfl

omit [Fintype Coin] [Nonempty Coin] in
theorem mem_ofDeployAt_checked {x : Msg} {r : Fin (D.V.k x) → Coin} {u : ℕ} :
    u ∈ (ofDeployAt D iroots n₀).checked x r ↔
      ∃ p : Fin (Claims.Sampling.nPU D x), (D.checked x r)[p.val]? = some u :=
  mem_take_iff_slot

omit [Fintype Coin] [Nonempty Coin] in
theorem mem_ofDeploy_checked {x : Msg} {r : Fin (D.V.k x) → Coin} {u : ℕ} :
    u ∈ (ofDeploy D iroots).checked x r ↔
      ∃ p : Fin (Claims.Sampling.nPU D x), (D.checked x r)[p.val]? = some u :=
  mem_ofDeployAt_checked D iroots fun x _ => D.innerN₀ x

omit [Fintype Coin] [Nonempty Coin] in
/-- **The opening checks' count at Flock is the frozen one**: a compiled verifier over Flock at a frozen record `D`'s
Flock verifier, with `D`'s parser, commitment, reveal and opening count, meets `Compiled.OpenCallsLe` where `D` meets
`Sampling.OpenCallsLe`, as the deployed verifier does (`SamplingDeploy.openCallsLe_deploy`). -/
theorem openCallsLe_of_deploy {H : Msg → Msg} (L : Compiled Coin (Arg.ofFlock D.flock))
    (hparse : L.parse = D.parse) (hcommit : L.commit = D.commit) (hreveal : L.reveal = D.reveal)
    (hcalls : L.openCalls = D.openCalls) (h : Claims.Sampling.OpenCallsLe H D) : L.OpenCallsLe H := by
  intro x s hs u v w
  rw [hparse] at hs
  rw [hcommit, hreveal, hcalls]
  exact h x s hs u v w

omit [Fintype Coin] [Nonempty Coin] in
theorem openCallsLe_ofDeploy {H : Msg → Msg} (h : Claims.Sampling.OpenCallsLe H D) :
    (ofDeploy D iroots).OpenCallsLe H :=
  openCallsLe_of_deploy D _ rfl rfl rfl rfl h

/-! ## The contract -/

variable [Inhabited Coin]

/-- Flock's relation at a statement is its check's. -/
theorem flock_check_of_R {parse : Msg → Option Claims.Flock.Stmt} {xS : Msg} {w : Claims.Flock.Wit}
    (h : Claims.Flock.R parse xS w) : Claims.Flock.check parse xS w = true := by
  obtain ⟨s, hs, hrel⟩ := h
  simp only [Claims.Flock.check, hs]
  exact decide_eq_true hrel

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
/-- **`fixed` from `DrawFixed`**: a sampled proof block's session's rounds are second-stage, and fresh coins on them keep the
sample. -/
theorem fixed_ofDeployAt (hf : Claims.Sampling.DrawFixed D) (x : Msg)
    (r : Fin ((ofDeployAt D iroots n₀).V.k x) → Coin) (u : ℕ) (hu : u ∈ (ofDeployAt D iroots n₀).checked x r) :
    (ofDeployAt D iroots n₀).rounds x u ⊆ (ofDeployAt D iroots n₀).stage2 x ∧
      ∀ d, (ofDeployAt D iroots n₀).checked x (forkOn ((ofDeployAt D iroots n₀).rounds x u) r d) =
        (ofDeployAt D iroots n₀).checked x r := by
  obtain ⟨hsub, hkeep⟩ := hf x r u ((mem_ofDeployAt_checked D iroots n₀).1 hu)
  exact ⟨hsub, fun d => congrArg (List.take _) (hkeep d)⟩

omit [Fintype Coin] [Nonempty Coin] in
/-- **`runs` from `SessionRunsTo` and `InnerNumbersAt`**: a session's rounds number Flock's at its statement and its
embedding is `emb`, both from `SessionRunsTo`'s order embedding onto them (`Finset.orderEmbOfFin_unique`); its shape and
errors are within the block's from `InnerNumbersAt` at the session the run shows, its `N₀` its unit's. -/
theorem runs_ofDeployAt (H : Msg → Msg) (hruns : Claims.Sampling.SessionRunsTo H D)
    (hnum : Claims.Sampling.InnerNumbersAt D n₀) (x : Msg) (Ω : Type) (P : Prover Ω) (ω : Ω)
    (r : Fin ((ofDeployAt D iroots n₀).V.k x) → Coin) (u : ℕ) (hu : u ∈ (ofDeployAt D iroots n₀).checked x r) :
    ∃ (xS : Msg) (P' : Prover Unit), (Arg.ofFlock D.flock).shape xS = (ofDeployAt D iroots n₀).sh x u ∧
      (Arg.ofFlock D.flock).κ xS ≤ (ofDeployAt D iroots n₀).κb x u ∧
      (Arg.ofFlock D.flock).γ xS ≤ (ofDeployAt D iroots n₀).γb x u ∧
      (Arg.ofFlock D.flock).V.k xS = ((ofDeployAt D iroots n₀).rounds x u).card ∧
      (∀ r' : Fin ((ofDeployAt D iroots n₀).V.k x) → Coin, (∀ t ∉ (ofDeployAt D iroots n₀).rounds x u, r' t = r t) →
        (run H P (ofDeployAt D iroots n₀).V x ω r').out = .acc →
          (run H P' (Arg.ofFlock D.flock).V xS () ((ofDeployAt D iroots n₀).coinsOf x u xS r')).out = .acc) ∧
      ∀ r' : Fin ((ofDeployAt D iroots n₀).V.k x) → Coin, (∀ t ∉ (ofDeployAt D iroots n₀).rounds x u,
          (∀ t' ∈ (ofDeployAt D iroots n₀).rounds x u, t < t') → r' t = r t) →
        (ofDeployAt D iroots n₀).session x u (run H P (ofDeployAt D iroots n₀).V x ω r').seen =
            some (xS, (run H P' (Arg.ofFlock D.flock).V xS () ((ofDeployAt D iroots n₀).coinsOf x u xS r')).seen) ∧
          ∀ q ∈ (run H P' (Arg.ofFlock D.flock).V xS () ((ofDeployAt D iroots n₀).coinsOf x u xS r')).vLog,
            q ∈ (run H P (ofDeployAt D iroots n₀).V x ω r').vLog := by
  obtain ⟨xS, emb, P', hmono, hrange, hacc, hsess⟩ :=
    hruns x Ω P ω r u ((mem_ofDeployAt_checked D iroots n₀).1 hu)
  have hset : D.rounds x u = Finset.univ.map ⟨emb, hmono.injective⟩ := by
    ext t
    simp [hrange t]
  have hcard : D.flock.V.k xS = (D.rounds x u).card := by
    rw [hset, Finset.card_map, Finset.card_univ, Fintype.card_fin]
  have hco : ∀ r', (ofDeployAt D iroots n₀).coinsOf x u xS r' = fun i => r' (emb i) := fun r' => funext fun i => by
    have he : emb = (D.rounds x u).orderEmbOfFin hcard.symm :=
      Finset.orderEmbOfFin_unique hcard.symm (fun i => (hrange _).2 ⟨i, rfl⟩) hmono
    show (if h : D.flock.V.k xS = (D.rounds x u).card then r' ((D.rounds x u).orderEmbOfFin h.symm i)
      else default) = r' (emb i)
    rw [dite_eq_left hcard]
    exact congrArg r' (congrFun he i).symm
  obtain ⟨hw, hc, hN, he, hwt⟩ := hnum x u _ xS _ (hsess r fun _ _ _ => rfl).1
  have hk : D.flock.k xS = ⌈4 * n₀ x u / Real.exp 1⌉₊ := by
    unfold Claims.Flock.Deploy.k
    rw [hN]
  refine ⟨xS, P', ?_, ?_, ?_, hcard, fun r' hr' => ?_, fun r' hr' => ?_⟩
  · show (D.flock.waits xS, D.flock.cap xS, D.flock.V.k xS, D.flock.N₀ xS) =
      (D.innerWaits x, D.innerCap x, (D.rounds x u).card, n₀ x u)
    rw [hw, hc, hcard, hN]
  · show Claims.Flock.κ D.flock xS ≤ κFAt D x (n₀ x u)
    rw [κFAt_eq]
    unfold Claims.Flock.κ
    rw [hk, hw, hc]
    linarith
  · show Claims.Flock.γ D.flock xS ≤
      16 * (Real.sqrt (2 * ((D.rounds x u).card : ℝ) * D.weight x) + (D.rounds x u).card) ^ 2
    unfold Claims.Flock.γ
    rw [hcard]
    gcongr
  · rw [hco]
    exact hacc r' hr'
  · rw [hco]
    exact hsess r' fun t ht hlt => hr' t ht fun t' ht' => by
      obtain ⟨i, rfl⟩ := (hrange t').1 ht'
      exact hlt i

omit [Nonempty Coin] in
/-- **The contract from the frozen hypotheses** at `D` (`Claims.Sampling.DrawLaw`, …, `SessionOpens`): they give
`ofDeploy`'s `Sessions`. The sample's members are the checked units at slots below `n_PU` (`mem_ofDeploy_checked`);
`fixed` and `runs` are `fixed_ofDeployAt`'s and `runs_ofDeployAt`'s at the record's `N₀`. -/
theorem sessions_ofDeploy (H : Msg → Msg) (hd : Claims.Sampling.DrawLaw D) (hr : Claims.Sampling.Refuses H D)
    (hi : Claims.Sampling.InteriorFixed H D iroots) (hf : Claims.Sampling.DrawFixed D)
    (hruns : Claims.Sampling.SessionRuns H D) (hnum : Claims.Sampling.InnerNumbers D)
    (hopens : Claims.Sampling.SessionOpens H D iroots) : (ofDeploy D iroots).Sessions H where
  draws x s P hs hP hin B hB := by
    obtain ⟨esc, h0, hle, hEx⟩ := hd x s P hs hP hin B hB
    refine ⟨esc, h0, fun r => le_of_eq_of_le
      (Pr_congr fun d => ⟨fun h v hv p hp => ?_, fun h u hu huB => ?_⟩) (hle r), hEx⟩
    · exact h v ((mem_ofDeploy_checked D iroots).2 ⟨p, hp⟩) hv
    · obtain ⟨p, hp⟩ := (mem_ofDeploy_checked D iroots).1 hu
      exact h u huB p hp
  slots _ _ := List.length_take_le _ _
  shared _ _ _ _ _ i j u u' _ hu hu' hk := by
    obtain rfl : i = j := hk
    obtain rfl : u = u' := Option.some.inj (hu.symm.trans hu')
    rfl
  keys x _ := Finset.card_image_le.trans ((Finset.card_range _).trans_le
    ((List.length_take_le _ _).trans_eq (ofDeploy_nSess D iroots x).symm))
  unions x := (ofDeploy_nPU D iroots x).le
  refuses := hr
  interior := hi
  fixed := fixed_ofDeployAt D iroots (fun x _ => D.innerN₀ x) hf
  runs := runs_ofDeployAt D iroots (fun x _ => D.innerN₀ x) H hruns.to hnum
  opens x s P Ω Pv ω r u xS vS w hs hP hin hacc hu hsess hR :=
    hopens x s P Ω Pv ω r u xS vS w hs hP hin hacc ((mem_ofDeploy_checked D iroots).1 hu) hsess
      (flock_check_of_R hR)

end OfDeploy

end Claims.SamplingGeneric

namespace SamplingDeploy

open Claims
open Claims.Sampling (InScope forkOn)
open Claims.SamplingGeneric (ofDeploy)

/-! ## The sampled verifier's code facts

The contract's clauses that read only the sampled verifier's code, as Props of `Code`'s functions, each the frozen
hypothesis of its name at `deployOf F c` (`Iff.rfl`):

* `Code.DrawLaw` (`draws`): on a statement in scope, for each set `B` of proof blocks, a weight of the first-stage coins
  bounds the probability over the second stage that no slot below the proof blocks holds a block of `B`, and averages
  at most `P_a(B)`;
* `Code.Refuses` (`refuses`): an accepted run's statement parses, its query evaluates, and the pair is in scope;
* `Code.InteriorFixed` (`interior`): the roots a run shows for a proof block's interior values are the same at fresh
  second-stage coins;
* `Code.DrawFixed` (`fixed`): a unit checked at a slot below the proof blocks has its session's rounds in the second
  stage, and fresh coins on them keep the checked units. -/

/-- **Sampling design**: a fact about the sampled verifier's code, for a Lean verifier to discharge (Python today). -/
def Code.DrawLaw (c : Code) : Prop :=
  ∀ x s Pt, c.parse x = some s → c.partition s = some Pt → InScope Pt s →
    ∀ B ⊆ Finset.range Pt.blocks.length, ∃ esc : (Fin (c.V.k x) → FlockDeploy.Coin) → ℝ, (∀ r, 0 ≤ esc r) ∧
      (∀ r, Pr (fun d : Fin (c.V.k x) → FlockDeploy.Coin => ∀ v ∈ B, ∀ p : Fin (units c x),
          (c.checked x (forkOn (c.stage2 x) r d))[p.val]? ≠ some v) ≤ esc r) ∧
      Ex esc ≤ s.design.Pa Pt B

/-- **Refuses**: a fact about the sampled verifier's code, for a Lean verifier to discharge (Python today). -/
def Code.Refuses (H : Msg → Msg) (c : Code) : Prop :=
  ∀ x (Ω : Type) (P : Prover Ω) ω r, (run H P c.V x ω r).out = .acc →
    ∃ s Pt, c.parse x = some s ∧ c.partition s = some Pt ∧ InScope Pt s

/-- **Interior fixed**: a fact about the sampled verifier's code, for a Lean verifier to discharge (Python today). -/
def Code.InteriorFixed (H : Msg → Msg) (c : Code) (iroots : Msg → Seen → ℕ → List Msg) : Prop :=
  ∀ x (Ω : Type) (P : Prover Ω) ω r d v,
    iroots x (run H P c.V x ω (forkOn (c.stage2 x) r d)).seen v = iroots x (run H P c.V x ω r).seen v

/-- **Sample fixed**: a fact about the sampled verifier's code, for a Lean verifier to discharge (Python today). -/
def Code.DrawFixed (c : Code) : Prop :=
  ∀ x r u, (∃ p : Fin (units c x), (c.checked x r)[p.val]? = some u) →
    c.rounds x u ⊆ c.stage2 x ∧ ∀ d, c.checked x (forkOn (c.rounds x u) r d) = c.checked x r

section Compiled

variable {T : Type} (F : Claims.Flock.Deploy FlockDeploy.Coin (List FlockLevel3.GF128) T) (c : Code)
  (iroots : Msg → Seen → ℕ → List Msg)

theorem drawLaw_deployOf : Claims.Sampling.DrawLaw (deployOf F c) ↔ c.DrawLaw := Iff.rfl

theorem refuses_deployOf (H : Msg → Msg) : Claims.Sampling.Refuses H (deployOf F c) ↔ c.Refuses H := Iff.rfl

theorem interiorFixed_deployOf (H : Msg → Msg) :
    Claims.Sampling.InteriorFixed H (deployOf F c) iroots ↔ c.InteriorFixed H iroots := Iff.rfl

theorem drawFixed_deployOf : Claims.Sampling.DrawFixed (deployOf F c) ↔ c.DrawFixed := Iff.rfl

/-- **The deployed sampled verifier as a compiled sampled verifier over Flock**: `ofDeploy` at `deployOf F c`, the
sessions' Flock record `F` at its deployed decoder. -/
noncomputable abbrev compiled :
    Claims.SamplingGeneric.Compiled FlockDeploy.Coin (Claims.SamplingGeneric.Arg.ofFlock F) :=
  ofDeploy (deployOf F c) iroots

theorem compiled_nPU (x : Msg) : (compiled F c iroots).nPU x = units c x := rfl

theorem compiled_checked (x : Msg) (r : Fin (c.V.k x) → FlockDeploy.Coin) :
    (compiled F c iroots).checked x r = (c.checked x r).take (units c x) := rfl

theorem compiled_rounds : (compiled F c iroots).rounds = c.rounds := rfl

theorem compiled_session : (compiled F c iroots).session = c.session := rfl

theorem compiled_reveal : (compiled F c iroots).reveal = c.reveal := rfl

theorem compiled_iroots : (compiled F c iroots).iroots = iroots := rfl

theorem compiled_commit : (compiled F c iroots).commit = commit c := rfl

theorem compiled_sh (x : Msg) (u : ℕ) :
    (compiled F c iroots).sh x u =
      (Claims.Flock.Numbers.waits, Claims.Flock.Numbers.cap, (c.rounds x u).card, c.innerN₀ x) :=
  rfl

/-- **The opening checks' count**, from the reveal count (`openCallsLe_deployOf`). -/
theorem openCallsLe_compiled (H : Msg → Msg) (h : RevealsLe c) : (compiled F c iroots).OpenCallsLe H :=
  Claims.SamplingGeneric.openCallsLe_ofDeploy (deployOf F c) iroots (openCallsLe_deployOf F c H h)

/-- **The deployed sampled verifier at its units' `N₀`**: `ofDeployAt` at `deployOf F c` and the code's `unitN₀`, each
proof block's shape and `κ_F` at its unit's `N₀`; at a code that sets no `unitN₀`, `compiled`. -/
noncomputable abbrev compiledAt :
    Claims.SamplingGeneric.Compiled FlockDeploy.Coin (Claims.SamplingGeneric.Arg.ofFlock F) :=
  Claims.SamplingGeneric.ofDeployAt (deployOf F c) iroots c.unitN₀

theorem compiledAt_sh (x : Msg) (u : ℕ) :
    (compiledAt F c iroots).sh x u =
      (Claims.Flock.Numbers.waits, Claims.Flock.Numbers.cap, (c.rounds x u).card, c.unitN₀ x u) :=
  rfl

/-- **The contract at the deployed record**: the code's facts, with the sessions' runs, numbers and openings
(`sessions_ofDeploy`). -/
theorem sessions_compiled (H : Msg → Msg) (hd : c.DrawLaw) (hr : c.Refuses H) (hi : c.InteriorFixed H iroots)
    (hf : c.DrawFixed) (hruns : Claims.Sampling.SessionRuns H (deployOf F c))
    (hnum : Claims.Sampling.InnerNumbers (deployOf F c))
    (hopens : Claims.Sampling.SessionOpens H (deployOf F c) iroots) : (compiled F c iroots).Sessions H :=
  Claims.SamplingGeneric.sessions_ofDeploy (deployOf F c) iroots H hd hr hi hf hruns hnum hopens

end Compiled

end SamplingDeploy
