module

public import Proofs.Properties.Verity.SampledTwo.Within
public import Proofs.Properties.Verity.SampledTwo.Fits
public import Proofs.Properties.Verity.SampledTwo.Reached
public import Proofs.Properties.Verity.SampledTwo.RuleTwo
public import Proofs.Flock.SpotFlock.RunsAs.RoundTwoOf
public import Proofs.Flock.SpotFlock.RunsAs.SecondStageOf
public import Proofs.Properties.Verity.SampledTwo.RecordNames
public import Proofs.Flock.SpotFlock.RunsAs.Steps
public import Proofs.Flock.SpotFlock.HeldOfRun
public import Proofs.Flock.SpotFlock.Cap
public import Proofs.Flock.SpotFlock.RunsAs.Top
public import Proofs.Properties.Verity.Sampled.Record.Reg
public import Proofs.Properties.Verity.SampledTwo.InputMap
public import Proofs.Flock.SpotFlock.RunsAs.PopulationOf
public import Proofs.Properties.Verity.SampledTwo.AtStmt
public import Proofs.Flock.Claims.Complete.Encode

/-!
# The form's accept is the run's accept: the pieces joined

`acceptsRun_form`: `AcceptsRun` at the two-stage form's reading and acceptance, from the run's accept theorem
(`OfRun.playAccepts_of_run`, with a stage's sample reading no byte past its source: `Within.playAccepts_of_run'`) and
an accepted run's bytes within its coins (`Fits.sourcesFit`). `reachedAccepts_form`: the same with the messages asked
only where the run has one, and every message reached (`Reached.reachedAccepts`).

The premises are what is left of `OfRun.lean`'s, each a named `Prop` about the plays the form's check accepts, and
`SessionMsgs` (`Accepts.lean`): the sessions' messages the accepted prover sends are those the check read. For three
of `OfRun.lean`'s, the held copies are asked to be the ones `heldOf` builds (`HeldOfAt`). The seven that are proved and
so not asked:

* `ReadsWithin` (`Within.playAccepts_of_run'`);
* `RuleIsTwo` (`RuleTwo.ruleIsTwo`: the form's parser returns only a two-stage rule);
* `RunsAs.RoundTwo` (`RunsAs.stmtOk_of_check`: the second stage's check gives the model's statement check at every
  held copy);
* `TwoHolds` (`twoHolds_at`: at a held copy `heldOf` builds for an accepted run the design evaluates,
  `HeldOf.twoHolds_of`, since an accepted run passed its population check, `RunsAs.population_of_accepted`);
* `NamesDistinct` (`Round2.namesDistinct_of_two`: where the design evaluates, the templates' names are distinct);
* `RecordsRead` (`recordsRead`: the form hands the run of record each registration's record as the proved verifier's
  parser reads its canonical text, `readBack`, so the proved verifier's reader reads it back from those bytes);
* `RecordsAt`: the interior record registers no value (`Round2.registersNone_of_two`: where the design evaluates, the
  interior registration registers none), and both records name their registrations' names (`recordsHeldAt_of`: an
  accepted run's registration checks have no code, `checks_of_accepted`, and a check with no code names its
  registration's names, `RecordNames.recordOk_of_check`). For that, each registration is asked to list each port
  once (`PortsDistinctAt`), which R3-ports doesn't check.

`HeldAt` is asked in one clause of its nine, `PartsHeldAt`: the held copies' parts are the interior program's strata's
at the play's replay sample. A held copy `heldOf` builds has the other eight by construction (`heldAt_of_parts`,
`HeldOf.heldMatches_of`). `PartsHeldAt` reads the play's bytes only through the replay sample's size, and where the
first stage samples a uniform `K`-subset of the replay blocks (`SpotFlock.firstK`) that size is `K` at every play, so
`PartsSizedAt` at `K`, which reads the play's parameters alone, gives it (`partsHeldAt_of_sized`). Every accepted
play's first stage is of a count `K` (`firstK_of_formAccepts`): the form's parser takes a two-stage rule only with a
rate law (`Select.Rule.ofJson`), which `Select.resolve` makes `subset:K` at `1 ≤ K ≤ n`, so `PartsSizedAt` at each
such `K` gives `PartsHeldAt` with nothing else asked (`partsHeldAt_of_accepts`). The run refuses an interior whose
strata are not the parts its replay block's cells give the check's templates (`interior-parts`), so an accepted
play gives `PartsSizedAt` too (`partsSizedAt_of_formAccepts`): the templates' circuits the parser returns name
those templates (`templateCircuits_names`). Asked at every
size it would hold of no held copy (`HeldOf.not_partsAre`: no stratum makes a part of a sample of no block).

`playAccepts_form` is `PlayAccepts` from what is left. Nothing here turns the form on, and none of the premises says
it is off.

Most of the premises name the one session of record `A`, so each is false at a case map whose accepted plays carry two
boundary records. `acceptsRun_form_at` and `reachedAccepts_form_at` ask them only of the accepted plays at the
statement `x`, at the case map `atStmt C x` (`AtStmt.lean`), and conclude at `C`.
-/

@[expose] public section

namespace Verity.SampledTwo

open Claims
open SpotFlock.RunsAs.Round2 (TwoHolds RecordsHeld)

/-- **An accepted play's records name their registrations' names** (`RecordsHeld`): `RecordsAt` less the half the
design decides. -/
def RecordsHeldAt (A : Live₂) (C : CaseOf) : Prop :=
  ∀ o i it x m₁, FormAccepts C o → inputOn C o = some i → i.interior = some it → recordMsg i.record = some x →
    recordMsg it.record = some m₁ → RecordsHeld A.h x m₁

/-- **A record `readBack` returned is read back from its canonical bytes** by the proved verifier's reader
(`SpotFlock.jsonMsg`): the bytes are its canonical text's, which the verifier's parser reads as the record. -/
theorem jsonMsg_of_readBack {record j : Lean.Json} {x : Msg}
    (h : VStar.Recursive.SampledTwo.readBack record = .ok j) (hx : recordMsg j = some x) :
    SpotFlock.jsonMsg x = some j := by
  obtain ⟨s, -, hp, hc⟩ := VStar.Recursive.SampledTwo.readBack_ok h
  unfold recordMsg at hx
  rw [hc] at hx
  obtain rfl : s.toUTF8.data.toList = x := Option.some.inj hx
  unfold SpotFlock.jsonMsg FlockDeploy.jsonOf
  have hb : (⟨s.toUTF8.data.toList.toArray⟩ : ByteArray) = s.toUTF8 := rfl
  rw [hb, FlockComplete.fromUTF8?_toUTF8]
  show ((Flock.JsonParse.parse s).mapError _).toOption = some j
  rw [hp]
  rfl

/-- **An accepted play's records read back** (`RecordsRead`), at every case map: the form hands the run of record each
registration's record as the proved verifier's parser reads its canonical text (`readBack`, `Steps.inputAt_ok`). -/
theorem recordsRead (C : CaseOf) : RecordsRead C := by
  intro o i it x m₁ hA hi hit hx hm₁
  obtain ⟨k, hk, p, cm, e, pub, prof, ⟨h1, h2, h3⟩, hacc⟩ := hA
  obtain ⟨-, c₂, ξ₂, r, i', run, hin, -⟩ := VStar.Recursive.SampledTwo.accepted_steps hacc
  obtain ⟨-, -, -, -, vs, rc, j₁, j₂, -, hieq, hj₁, hj₂, -⟩ := VStar.Recursive.SampledTwo.inputAt_ok hin
  have hik : inputOfCase k = some i' := by
    unfold inputOfCase
    rw [h1, h2, h3]
    simp only [bind, Except.bind, hin]
    rfl
  have hi' : inputOn C o = some i' := by
    unfold inputOn
    rw [hk]
    exact hik
  have hii : i = i' := Option.some.inj (hi.symm.trans hi')
  have hr1 : i.record = j₁ := by rw [hii, hieq]; rfl
  have hr2 : it.record = j₂ := by
    rw [hii, hieq] at hit
    exact (congrArg Spot.Run.Interior.record (Option.some.inj hit)).symm
  rw [hr1] at hx
  rw [hr2] at hm₁
  rw [hr1, hr2]
  exact ⟨jsonMsg_of_readBack hj₁ hx, jsonMsg_of_readBack hj₂ hm₁⟩

/-- **An accepted two-stage run's registration checks have no code**, the boundary record's and the interior
record's (`RunsAs.accepted_registered`), each at a registration with the run's own ports and registered values
(`namesOf`): the run checks each at its own registration with the sampling design it expands and, at the boundary, its work
table. -/
theorem checks_of_accepted {i : Spot.Run.Input} {it : Spot.Run.Interior} {j : Lean.Json}
    (ha : SpotFlock.RunsAs.AcceptedTwo i it j) :
    (∃ v, Spot.Registration.check i.record v = #[] ∧
        SpotFlock.HeldOf.namesOf v = SpotFlock.HeldOf.namesOf i.params.verifier) ∧
      ∃ v, Spot.Registration.check it.record v = #[] ∧
        SpotFlock.HeldOf.namesOf v = SpotFlock.HeldOf.namesOf it.params.verifier := by
  obtain ⟨n, rule, law, workTable, resolved, draw, rest, read, ns, -, -, -, -, h₁, -, -, -, -, h₂⟩ :=
    SpotFlock.RunsAs.accepted_registered ha
  exact ⟨⟨_, Array.isEmpty_iff.1 h₁, rfl⟩, _, Array.isEmpty_iff.1 h₂, rfl⟩

/-- **Each accepted play's held copies are the ones `heldOf` builds** from its input and interior. -/
def HeldOfAt (A : Live₂) (C : CaseOf) : Prop :=
  ∀ o i it, FormAccepts C o → inputOn C o = some i → i.interior = some it →
    ∃ u, SpotFlock.HeldOf.heldOf i it u = some A.h

/-- **Each accepted play's registrations list each port once**, the boundary's and the interior's. -/
def PortsDistinctAt (C : CaseOf) : Prop :=
  ∀ o i it, FormAccepts C o → inputOn C o = some i → i.interior = some it →
    (i.params.verifier.ports.map (·.1)).toList.Nodup ∧ (it.params.verifier.ports.map (·.1)).toList.Nodup

/-- **An accepted play's records name their registrations' names**: its run's registration checks have no code
(`checks_of_accepted`), and each names its registration's names (`RecordNames.recordOk_of_check`), which
are the held copies' (`HeldOf.namesHeld_of`). -/
theorem recordsHeldAt_of {A : Live₂} {C : CaseOf} (hof : HeldOfAt A C) (hports : PortsDistinctAt C) :
    RecordsHeldAt A C := by
  intro o i it x m₁ hA hi hit hx hm₁
  obtain ⟨u, hu⟩ := hof o i it hA hi hit
  obtain ⟨hb, hn⟩ := SpotFlock.HeldOf.namesHeld_of hu
  obtain ⟨j, hj⟩ := acceptedTwo_of_formAccepts (ruleIsTwo C) hA hi hit
  obtain ⟨⟨v, hv, hvn⟩, w, hw, hwn⟩ := checks_of_accepted hj
  obtain ⟨hpb, hpi⟩ := hports o i it hA hi hit
  obtain ⟨hjx, hjm⟩ := recordsRead C o i it x m₁ hA hi hit hx hm₁
  have hnv : (v.ports.map (·.1)).toList.Nodup := by
    have := congrArg SpotFlock.Names.ports hvn
    simp only [SpotFlock.HeldOf.namesOf] at this
    rw [this]
    exact hpb
  have hnw : (w.ports.map (·.1)).toList.Nodup := by
    have := congrArg SpotFlock.Names.ports hwn
    simp only [SpotFlock.HeldOf.namesOf] at this
    rw [this]
    exact hpi
  have rb := SpotFlock.RecordNames.recordOk_of_check hv hnv
  have ri := SpotFlock.RecordNames.recordOk_of_check hw hnw
  rw [hvn, ← hb] at rb
  rw [hwn, ← hn] at ri
  show SpotFlock.recordsOk A.h x m₁ = true
  unfold SpotFlock.recordsOk SpotFlock.decode
  rw [hjx, hjm]
  simp only [Option.map_some, Bool.and_eq_true]
  exact ⟨rb, ri⟩

/-- **The held copies' parts are the interior program's strata's**, at each accepted play's replay sample
(`HeldOf.PartsAt`): the clause of `HeldAt` that `heldOf`'s construction leaves. -/
def PartsHeldAt (A : Live₂) (C : CaseOf) : Prop :=
  ∀ o i it, FormAccepts C o → inputOn C o = some i → i.interior = some it → SpotFlock.HeldOf.PartsAt i it A.h

/-- `HeldAt` gives its clause `PartsHeldAt`. -/
theorem partsHeldAt_of {A : Live₂} {C : CaseOf} (hheld : HeldAt A C) : PartsHeldAt A C :=
  fun o i it hA hi hit => (hheld o i it hA hi hit).parts

/-- **The held copies' parts at a replay sample of `K` blocks are the interior program's strata's**, at each accepted
play: `PartsHeldAt` with the size `K` in place of the play's replay sample's. It reads the play's parameters and none
of its bytes. -/
def PartsSizedAt (A : Live₂) (C : CaseOf) (K : ℕ) : Prop :=
  ∀ o i it strata, FormAccepts C o → inputOn C o = some i → i.interior = some it →
    Spot.Partition.strata it.params.verifier.program it.params.verifier.query = .ok strata →
    Spot.SecondStage.parts K strata = some (SpotFlock.partsAt A.h K)

/-- **`PartsHeldAt` from the parts at the one size the first stage samples**: where the held copies' first stage
samples a uniform `K`-subset of the replay blocks (`SpotFlock.firstK`), every replay sample it selects has `K` blocks
(`SpotFlock.select_subset_size`), so the parts at `K` are the parts at the play's sample. -/
theorem partsHeldAt_of_sized {A : Live₂} {C : CaseOf} {K : ℕ} (hK : SpotFlock.firstK A.h.reg.first = some K)
    (hs : PartsSizedAt A C K) : PartsHeldAt A C := by
  intro o i it hA hi hit draw rest read strata hsel hst
  rw [SpotFlock.select_subset_size hK hsel]
  exact hs o i it strata hA hi hit hst

/-- **A two-stage rule `Select.Rule.ofJson` reads has a rate law** for its first stage (`Select.replayLaw?`). -/
theorem ofJson_replay {j g : Lean.Json} {t : String} {k : Spot.Select.Counts}
    (h : Spot.Select.Rule.ofJson j = .ok (.twoStage g t k)) : (Spot.Select.parseRate t).isSome = true := by
  unfold Spot.Select.Rule.ofJson at h
  repeat'
    first
    | (cases h; done)
    | split at h
    | dsimp only at h
  all_goals first
    | (cases h; done)
    | (cases h; simp_all [Spot.Select.replayLaw?, Option.isSome_iff_ne_none])

/-- **The form's parameters' rule is one `Select.Rule.ofJson` read.** -/
theorem paramsOfJson_ruleOf {j : Lean.Json} {p : VStar.Recursive.SampledTwo.Params}
    (h : VStar.Recursive.SampledTwo.paramsOfJson j = .ok p) : ∃ j', Spot.Select.Rule.ofJson j' = .ok p.rule := by
  unfold VStar.Recursive.SampledTwo.paramsOfJson at h
  repeat'
    first
    | (cases h; done)
    | (obtain ⟨_, _, h⟩ := Properties.Spot.SecurityProofs.bind_eq_ok.1 h)
    | (simp only [pure, Except.pure, Except.ok.injEq] at h
       subst h
       exact ⟨_, by assumption⟩)
    | split at h
    | dsimp only at h

/-- **A rate law over `n` units is `subset:K` at `1 ≤ K ≤ n`** (`Select.textOver`). -/
theorem textOver_rate {t : String} {a b c n : ℕ} {sizes : List ℕ} {s : String}
    (hpr : Spot.Select.parseRate t = some (a, b, c)) (h : Spot.Select.textOver t n sizes = some s) :
    ∃ K, 1 ≤ K ∧ K ≤ n ∧ s = s!"subset:{K}" := by
  unfold Spot.Select.textOver at h
  rw [hpr] at h
  repeat' first | (cases h; done) | split at h | dsimp only at h
  all_goals first
    | (cases h; done)
    | (cases h; exact ⟨_, by simp_all, by simp_all, rfl⟩)
    | (exfalso; simp_all)

theorem subset_text (K : ℕ) : s!"subset:{K}" = "subset" ++ ":" ++ Nat.repr K := by
  simp only [String.append_assoc]
  rfl

/-- **An accepted play's run has a two-stage rule with a rate law**: the run of record's rule is the parameters'
(`inputOf_first`), which the form's parser reads with `Select.Rule.ofJson`. -/
theorem rule_of_formAccepts {C : CaseOf} {o : Verity.Exchange} {i : Spot.Run.Input} (hA : FormAccepts C o)
    (hi : inputOn C o = some i) :
    ∃ g t k, i.rule = .twoStage g t k ∧ (Spot.Select.parseRate t).isSome = true := by
  obtain ⟨k, hk, p, cm, e, pub, prof, ⟨h1, h2, h3⟩, hacc⟩ := hA
  obtain ⟨-, c₂, ξ₂, r, i', run, hin, -⟩ := VStar.Recursive.SampledTwo.accepted_steps hacc
  obtain ⟨-, -, -, -, vs, rc, j₁, j₂, -, hieq, -, -⟩ := VStar.Recursive.SampledTwo.inputAt_ok hin
  have hik : inputOfCase k = some i' := by
    unfold inputOfCase
    rw [h1, h2, h3]
    simp only [bind, Except.bind, hin]
    rfl
  have hi' : inputOn C o = some i' := by
    unfold inputOn
    rw [hk]
    exact hik
  have hii : i = i' := Option.some.inj (hi.symm.trans hi')
  have hrp : i.rule = p.rule := by
    rw [hii, hieq]
    exact (_root_.VStar.Recursive.SampledTwo.inputOf_first _ _ _ _ _ _ _).1
  obtain ⟨j, -, hj⟩ := Properties.Spot.SecurityProofs.bind_eq_ok.1 h1
  obtain ⟨g, t, kk, hr⟩ := SpotFlock.RunsAs.paramsOfJson_twoStage hj
  obtain ⟨j', hj'⟩ := paramsOfJson_ruleOf hj
  rw [hr] at hj'
  exact ⟨g, t, kk, hrp.trans hr, ofJson_replay hj'⟩

/-- **Each accepted play's held first stage is of a count `K`** (`SpotFlock.firstK`): the held copies are the ones
`heldOf` builds (`HeldOfAt`), whose first-stage registration has the run's closure map, none at an accepted play
(`closure_of_formAccepts`), and the rule `Select.resolve` makes of the run's, a rate law (`rule_of_formAccepts`)
resolved to `subset:K` at `1 ≤ K ≤ n` (`textOver_rate`), which Flock reads as a uniform `K`-subset. -/
theorem firstK_of_formAccepts {A : Live₂} {C : CaseOf} (hof : HeldOfAt A C) {o : Verity.Exchange}
    {i : Spot.Run.Input} {it : Spot.Run.Interior} (hA : FormAccepts C o) (hi : inputOn C o = some i)
    (hit : i.interior = some it) : ∃ K, SpotFlock.firstK A.h.reg.first = some K := by
  obtain ⟨u, hu⟩ := hof o i it hA hi hit
  obtain ⟨first, ns, ks, l, hz, hf, -, -, hh, -⟩ := SpotFlock.HeldOf.heldOf_spec hu
  have hfirst : A.h.reg.first = first := by
    rw [hh]
    rfl
  rw [hfirst]
  have hcl := closure_of_formAccepts hA hi
  obtain ⟨g₀, t, kk, hrt, hrate⟩ := rule_of_formAccepts hA hi
  obtain ⟨⟨a, b, c⟩, hpr⟩ := Option.isSome_iff_exists.1 hrate
  unfold SpotFlock.HeldOf.firstOf at hf
  simp only [Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def, Option.some.injEq] at hf
  obtain ⟨n, -, rule, -, law, -, workTable, -, resolved, hres, rfl⟩ := hf
  have hres' := SpotFlock.HeldOf.ok_of_toOption hres
  unfold Spot.Select.resolve at hres'
  obtain ⟨n', hn', hr⟩ := Properties.Spot.SecurityProofs.bind_eq_ok.1 hres'
  split at hr
  · rename_i r hov
    cases hr
    rw [hrt] at hov
    simp only [Spot.Select.Rule.over] at hov
    obtain ⟨s, htx, hov⟩ := Option.map_eq_some_iff.1 hov
    subst hov
    obtain ⟨K, hK1, hKn, hs⟩ := textOver_rate hpr htx
    have hpd : Spot.Select.parseDesign s =
        .ok (Lean.Json.mkObj [("law", Lean.Json.str "subset"), ("k", Lean.toJson K)]) := by
      rw [hs, subset_text K]
      exact Verity.Sampled.Record.parseDesign_count "subset" (Or.inl rfl) K hK1
    refine ⟨K, ?_⟩
    unfold SpotFlock.firstK SpotFlock.firstKOf
    simp only [hcl, SamplingDeploy.Reg.program, SamplingDeploy.Reg.query, hn', Spot.Select.Rule.lawOver, hpd,
      Verity.Sampled.Record.flockLaw_subset_text K n' hK1 hKn]
  · cases hr

/-- **`PartsHeldAt` from the parts at each count the first stage samples**: at each accepted play the held first
stage is of a count `K` (`firstK_of_formAccepts`), and `PartsSizedAt` at that `K` gives the parts at the play's
replay sample (`partsHeldAt_of_sized`). -/
theorem partsHeldAt_of_accepts {A : Live₂} {C : CaseOf} (hof : HeldOfAt A C)
    (hs : ∀ K, SpotFlock.firstK A.h.reg.first = some K → PartsSizedAt A C K) : PartsHeldAt A C := by
  intro o i it hA hi hit
  obtain ⟨K, hK⟩ := firstK_of_formAccepts hof hA hi hit
  exact partsHeldAt_of_sized hK (hs K hK) o i it hA hi hit

/-- **An accepted interior's strata are the parts its replay block's cells give** the check's templates at its
replay sample (`Run.interiorSteps`, else `interior-parts`). -/
theorem interiorSteps_parts {i : Spot.Run.Input} {it : Spot.Run.Interior} {k : Spot.Select.Counts}
    {sample : Array ℕ} {x : Spot.Run.Interior × Lean.Json × Lean.Json × Lean.Json × ℕ}
    (h : Spot.Run.interiorSteps i it k sample = .ok (.ok x)) :
    ∃ ns ids strata, Spot.Select.replayCells it.replay it.params.verifier.query = .ok ns ∧
      Spot.Select.templateIds it.params.verifier.query = .ok ids ∧
      Spot.Partition.strata it.params.verifier.program it.params.verifier.query = .ok strata ∧
      Spot.SecondStage.parts sample.size strata = some (Spot.Run.cellParts sample.size ids ns) := by
  unfold Spot.Run.interiorSteps at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat'
    first
    | split at h
    | focus (cases h; done)
    | (cases h; exact ⟨_, _, _, by assumption, by assumption, by assumption, not_not.1 (by assumption)⟩)

/-- **The replay block's cells are one a template** of the check's query (`Select.replayCells`). -/
theorem replayCells_size {p : Spot.Program} {q : Lean.Json} {ns : Array ℕ}
    (h : Spot.Select.replayCells p q = .ok ns) :
    ∃ ids, Spot.Select.templateIds q = .ok ids ∧ ns.size = ids.size := by
  unfold Spot.Select.replayCells at h
  repeat'
    first
    | (cases h; done)
    | (obtain ⟨_, _, h⟩ := Properties.Spot.SecurityProofs.bind_eq_ok.1 h)
    | (simp only [pure, Except.pure, Except.ok.injEq] at h
       subst h
       exact ⟨_, by assumption, by simp⟩)
    | split at h
    | dsimp only at h

private theorem eq_of_not_bne_true {α : Type} [BEq α] [LawfulBEq α] {a b : α} (h : ¬(a != b) = true) : a = b := by
  cases hab : a == b
  · exact absurd (by simp [bne, hab]) h
  · exact eq_of_beq hab

set_option linter.auxLemma false in
/-- **`Run.templateCircuits` names the check's templates, in its order, or returns none**: the parser throws unless
`named.map (·.1)` is `Select.templateIds`' and returns none only for no `proofs`. Its match on `#["templates"]` has
no equations `split` can use, so that matcher is unfolded by name and its test of the key against `"templates"` is
cased on. The proof names the matcher by its number (`Spot.Run.templateCircuits.match_9`), so it breaks when an edit
of `Spot/Run.lean` renumbers the parser's matchers: the number to name then is the matcher's that tests the key. -/
theorem templateCircuits_names {proofs query : Lean.Json} {cs : Array (String × String × Lean.Json × Lean.Json)}
    (h : Spot.Run.templateCircuits proofs query = .ok cs) :
    cs = #[] ∨ Spot.Select.templateIds query = .ok (cs.map (·.1)) := by
  unfold Spot.Run.templateCircuits at h
  generalize Spot.keys? proofs = kk at h
  generalize Spot.getD proofs "templates" = gg at h
  split at h
  · exact Or.inl (Except.ok.inj h).symm
  · obtain ⟨ids, hids, h⟩ := Properties.Spot.SecurityProofs.bind_eq_ok.1 h
    rcases kk with _ | ⟨⟨_ | ⟨s, _ | ⟨s2, l2⟩⟩⟩⟩
    · cases h
    · cases h
    · delta Spot.Run.templateCircuits.match_9 at h
      conv at h => lhs; whnf
      generalize instDecidableEqString _ "templates" = d at h
      cases d with
      | isFalse hn => cases h
      | isTrue ht =>
        have hs : s = "templates" := ht
        subst hs
        cases gg <;> first | cases h | skip
        obtain ⟨named, hnamed, h⟩ := Properties.Spot.SecurityProofs.bind_eq_ok.1 h
        split at h
        · cases h
        · rename_i hne
          cases h
          exact Or.inr (hids.trans (congrArg Except.ok (eq_of_not_bne_true hne).symm))
    · cases h

/-- **The form's parameters' circuits are `Run.templateCircuits`'** at the interior's query. -/
theorem paramsOfJson_circuits {j : Lean.Json} {p : VStar.Recursive.SampledTwo.Params}
    (h : VStar.Recursive.SampledTwo.paramsOfJson j = .ok p) :
    ∃ pj, Spot.Run.templateCircuits pj p.interior.verifier.query = .ok p.circuits := by
  unfold VStar.Recursive.SampledTwo.paramsOfJson at h
  repeat'
    first
    | (cases h; done)
    | (obtain ⟨_, _, h⟩ := Properties.Spot.SecurityProofs.bind_eq_ok.1 h)
    | (simp only [pure, Except.pure, Except.ok.injEq] at h
       subst h
       exact ⟨_, by assumption⟩)
    | split at h
    | dsimp only at h

/-- **The run of record's interior has the form's interior parameters and circuits.** -/
theorem inputOf_circuits {p : VStar.Recursive.SampledTwo.Params} {c : Spot.OneStage.Commitment} {ξ ξ₂ : ByteArray}
    {e : VStar.Recursive.SampledTwo.Evidence} {received₂ : String} {vs : Array (String × Lean.Json)}
    {it : Spot.Run.Interior}
    (h : (_root_.VStar.Recursive.SampledTwo.inputOf p c ξ ξ₂ e received₂ vs).interior = some it) :
    it.params = p.interior ∧ it.circuits = p.circuits := by
  have key : ∃ it', (_root_.VStar.Recursive.SampledTwo.inputOf p c ξ ξ₂ e received₂ vs).interior = some it' ∧
      it'.params = p.interior ∧ it'.circuits = p.circuits := ⟨_, rfl, rfl, rfl⟩
  obtain ⟨it', hit', hpar, hcirc⟩ := key
  rw [hit'] at h
  cases h
  exact ⟨hpar, hcirc⟩

/-- **An accepted play's templates' circuits are `Run.templateCircuits`'** at its interior's query: the form's parser
reads them so (`paramsOfJson_circuits`) and hands them to the run of record (`inputOf_circuits`). -/
theorem circuits_of_formAccepts {C : CaseOf} {o : Verity.Exchange} {i : Spot.Run.Input} {it : Spot.Run.Interior}
    (hA : FormAccepts C o) (hi : inputOn C o = some i) (hit : i.interior = some it) :
    ∃ pj, Spot.Run.templateCircuits pj it.params.verifier.query = .ok it.circuits := by
  obtain ⟨k, hk, p, cm, e, pub, prof, ⟨h1, h2, h3⟩, hacc⟩ := hA
  obtain ⟨-, c₂, ξ₂, r, i', run, hin, -⟩ := VStar.Recursive.SampledTwo.accepted_steps hacc
  obtain ⟨-, -, -, -, vs, rc, j₁, j₂, -, hieq, -, -⟩ := VStar.Recursive.SampledTwo.inputAt_ok hin
  have hik : inputOfCase k = some i' := by
    unfold inputOfCase
    rw [h1, h2, h3]
    simp only [bind, Except.bind, hin]
    rfl
  have hi' : inputOn C o = some i' := by
    unfold inputOn
    rw [hk]
    exact hik
  have hii : i = i' := Option.some.inj (hi.symm.trans hi')
  rw [hii, hieq] at hit
  obtain ⟨hpar, hcirc⟩ := inputOf_circuits hit
  obtain ⟨j, -, hj⟩ := Properties.Spot.SecurityProofs.bind_eq_ok.1 h1
  obtain ⟨pj, hpj⟩ := paramsOfJson_circuits hj
  rw [hpar, hcirc]
  exact ⟨pj, hpj⟩

/-- **The held copies' cells are the replay block's** (`Select.replayCells`), as `heldOf` holds them. -/
theorem cells_heldOf {i : Spot.Run.Input} {it : Spot.Run.Interior} {u : SpotFlock.HeldOf.Unread}
    {h : SpotFlock.Held} (hh : SpotFlock.HeldOf.heldOf i it u = some h) {ns : Array ℕ}
    (hns : Spot.Select.replayCells it.replay it.params.verifier.query = .ok ns) : h.reg.ns = ns := by
  obtain ⟨first, ns', ks, l, hz, -, hc, -, rfl, -⟩ := SpotFlock.HeldOf.heldOf_spec hh
  unfold SpotFlock.HeldOf.cellsOf at hc
  simp only [Option.bind_eq_bind, Option.bind_eq_some_iff] at hc
  obtain ⟨ns'', hns'', hc⟩ := hc
  obtain rfl : ns'' = ns := Except.ok.inj ((SpotFlock.HeldOf.ok_of_toOption hns'').symm.trans hns)
  split at hc
  · simp only [Option.bind_eq_some_iff, Option.pure_def, Option.some.injEq, Prod.mk.injEq] at hc
    obtain ⟨-, -, -, -, h₁, -⟩ := hc
    exact h₁.symm
  · cases hc

/-- **The held copies have a cell for each template's circuit**: `heldOf` decides it (`HeldOf.descCore`). -/
theorem cells_size_heldOf {i : Spot.Run.Input} {it : Spot.Run.Interior} {u : SpotFlock.HeldOf.Unread}
    {h : SpotFlock.Held} (hh : SpotFlock.HeldOf.heldOf i it u = some h) : h.reg.ns.size = it.circuits.size := by
  obtain ⟨first, ns, ks, l, hz, -, -, -, rfl, hd⟩ := SpotFlock.HeldOf.heldOf_spec hh
  unfold SpotFlock.HeldOf.decided at hd
  simp only [Bool.and_eq_true] at hd
  have hcore := hd.2
  rw [SpotFlock.HeldOf.descCore_withFrame] at hcore
  unfold SpotFlock.HeldOf.descCore at hcore
  split at hcore
  · simp only [Bool.and_eq_true, beq_iff_eq] at hcore
    obtain ⟨⟨⟨⟨⟨⟨⟨-, -⟩, -⟩, -⟩, hT⟩, -⟩, -⟩, -⟩ := hcore
    exact hT.trans (SpotFlock.HeldOf.ms_length_held₀ ..)
  · cases hcore

/-- **The held copies' parts are the ones the replay block's cells give the templates' circuits' names**
(`Run.cellParts`): `heldOf` holds the interior's templates' names and the replay block's cells. -/
theorem partsAt_heldOf {i : Spot.Run.Input} {it : Spot.Run.Interior} {u : SpotFlock.HeldOf.Unread}
    {h : SpotFlock.Held} (hh : SpotFlock.HeldOf.heldOf i it u = some h) {ns : Array ℕ}
    (hns : Spot.Select.replayCells it.replay it.params.verifier.query = .ok ns) (q : ℕ) :
    SpotFlock.partsAt h q = Spot.Run.cellParts q (it.circuits.map (·.1)) ns := by
  have hc := cells_heldOf hh hns
  obtain ⟨first, ns', ks, l, hz, -, -, -, rfl, -⟩ := SpotFlock.HeldOf.heldOf_spec hh
  have hc' : ns' = ns := hc
  subst hc'
  unfold SpotFlock.partsAt Spot.Run.cellParts
  rw [SpotFlock.HeldOf.ms_withFrame, SpotFlock.HeldOf.ms_length_held₀]
  first
    | rfl
    | (simp only [Array.size_map]
       first | done | rfl)

/-- **An accepted play's templates' circuits name the check's templates, in its order**: the form's parser reads
them so or reads none (`templateCircuits_names`), and where it reads none the held copies have no cell
(`cells_size_heldOf`), so the check names no template (`replayCells_size`). -/
theorem names_of_formAccepts {A : Live₂} {C : CaseOf} (hof : HeldOfAt A C) {o : Verity.Exchange}
    {i : Spot.Run.Input} {it : Spot.Run.Interior} (hA : FormAccepts C o) (hi : inputOn C o = some i)
    (hit : i.interior = some it) {ns : Array ℕ} {ids : Array String}
    (hns : Spot.Select.replayCells it.replay it.params.verifier.query = .ok ns)
    (hids : Spot.Select.templateIds it.params.verifier.query = .ok ids) : it.circuits.map (·.1) = ids := by
  obtain ⟨u, hu⟩ := hof o i it hA hi hit
  obtain ⟨pj, hpj⟩ := circuits_of_formAccepts hA hi hit
  obtain ⟨ids', hids', hsz⟩ := replayCells_size hns
  have hii : ids' = ids := Except.ok.inj (hids'.symm.trans hids)
  rw [hii] at hsz
  have hsize : ns.size = it.circuits.size := by
    rw [← cells_heldOf hu hns]
    exact cells_size_heldOf hu
  rcases templateCircuits_names hpj with h0 | h1
  · have h3 : ids.size = 0 := by
      rw [← hsz, hsize, h0]
      first | done | rfl
    rw [h0, Array.eq_empty_of_size_eq_zero h3]
    first | done | rfl | simp
  · exact Except.ok.inj (h1.symm.trans hids)

/-- **`PartsSizedAt` is a lemma of the accept**: an accepted play's interior passed `interior-parts` (`interiorSteps_parts`), whose parts are the held copies' (`partsAt_heldOf`,
`names_of_formAccepts`), at its replay sample, which has `K` blocks where the held first stage samples a `K`-subset
(`SpotFlock.select_subset_size`). -/
theorem partsSizedAt_of_formAccepts {A : Live₂} {C : CaseOf} (hof : HeldOfAt A C) :
    ∀ K, SpotFlock.firstK A.h.reg.first = some K → PartsSizedAt A C K := by
  intro K hK o i it strata hA hi hit hst
  obtain ⟨u, hu⟩ := hof o i it hA hi hit
  obtain ⟨first, ns₀, ks₀, l, hz, hf, -, -, hh, -⟩ := SpotFlock.HeldOf.heldOf_spec hu
  have hfirst : A.h.reg.first = first := by
    rw [hh]
    rfl
  have hcl := closure_of_formAccepts hA hi
  obtain ⟨j, ha⟩ := acceptedTwo_of_formAccepts (ruleIsTwo C) hA hi hit
  obtain ⟨grain, replay, k, hr⟩ := ha.two
  rcases SpotFlock.RunsAs.run_path ha.run with ⟨step, codes, hj⟩ | ⟨n, rule, law, workTable, resolved, draw, rest,
      read, -, -, -, ht, -, -, hres, hsel, -, inner, acc₂, staged, receipt, sym, pub, audit, hsec, -, -, -⟩
  · have hs := ha.accept
    rw [hj, SpotFlock.RunsAs.symbol_refused] at hs
    cases hs
  · obtain ⟨x, hx, -⟩ := SpotFlock.RunsAs.secondStage_two hr ha.interior hsec
    obtain ⟨ns, ids, strata', hns, hids, hst', hparts⟩ := interiorSteps_parts hx
    have hss : strata' = strata := Except.ok.inj (hst'.symm.trans hst)
    subst hss
    have hnames := names_of_formAccepts hof hA hi hit hns hids
    have hsel0 := hsel
    rw [hcl] at hsel0
    have hproved : (Spot.Draw.proved draw).map Int.toNat = rest :=
      SpotFlock.RunsAs.proved_select (SpotFlock.RunsAs.resolve_ne_all hr hres) hsel0
    unfold SpotFlock.HeldOf.firstOf at hf
    simp only [Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def, Option.some.injEq] at hf
    obtain ⟨n', -, rule', -, law', -, workTable', hwt', resolved', hres', hfeq⟩ := hf
    have hwt : workTable' = workTable := Except.ok.inj ((SpotFlock.HeldOf.ok_of_toOption hwt').symm.trans ht)
    have hrs : resolved' = resolved := Except.ok.inj ((SpotFlock.HeldOf.ok_of_toOption hres').symm.trans hres)
    subst hwt hrs
    have hsel' : Spot.Select.select A.h.reg.first.rule A.h.reg.first.program A.h.reg.first.query
        A.h.reg.first.table A.h.reg.first.closure i.source = .ok (draw, rest, read) := by
      rw [hfirst, ← hfeq]
      exact hsel
    rw [hproved, SpotFlock.select_subset_size hK hsel'] at hparts
    rw [hparts, partsAt_heldOf hu hns K, hnames]

/-- **Held copies `heldOf` builds agree with what each accepted play's run reads** (`HeldAt`), given their parts
(`HeldOf.heldMatches_of`). -/
theorem heldAt_of_parts {A : Live₂} {C : CaseOf} (hof : HeldOfAt A C) (hparts : PartsHeldAt A C) : HeldAt A C := by
  intro o i it hA hi hit
  obtain ⟨u, hu⟩ := hof o i it hA hi hit
  exact SpotFlock.HeldOf.heldMatches_of hu (hparts o i it hA hi hit)

/-- **The design evaluates at the held copies** (`TwoHolds`), wherever a play is accepted: the play's run is an accepted
two-stage run (`acceptedTwo_of_formAccepts`), which passed its population check (`RunsAs.population_of_accepted`), and
the held copies are the ones `heldOf` builds from it (`HeldOf.twoHolds_of`). -/
theorem twoHolds_at {A : Live₂} {C : CaseOf} (hof : HeldOfAt A C) {o : Verity.Exchange} {i : Spot.Run.Input}
    {it : Spot.Run.Interior} (hA : FormAccepts C o) (hi : inputOn C o = some i) (hit : i.interior = some it) :
    TwoHolds A.h := by
  obtain ⟨u, hu⟩ := hof o i it hA hi hit
  obtain ⟨j, hj⟩ := acceptedTwo_of_formAccepts (ruleIsTwo C) hA hi hit
  obtain ⟨_, -, _⟩ := SpotFlock.RunsAs.population_of_accepted hj
  exact SpotFlock.HeldOf.twoHolds_of hu (by assumption)

/-- **Where the design evaluates, records that name their registrations' names are `RecordsAt`**: the interior record,
read back as the record the run held (`recordsRead`), registers no value (`Round2.registersNone_of_two`). -/
theorem recordsAt_of_held {A : Live₂} {C : CaseOf} (htwo : TwoHolds A.h) (hrec : RecordsHeldAt A C) :
    RecordsAt A C := by
  intro o i it x m₁ hA hi hit hx hm₁
  have hrh : RecordsHeld A.h x m₁ := hrec o i it x m₁ hA hi hit hx hm₁
  exact ⟨hrh, SpotFlock.RunsAs.Round2.registersNone_of_two htwo (recordsRead C o i it x m₁ hA hi hit hx hm₁).2 hrh⟩

/-- **The form's accept is a play SPOT[Flock] accepts** (`PlayAccepts` at the form): `Within.playAccepts_of_run'` with
the rule two-stage by the form's parser (`ruleIsTwo`), the second stage's hand-off proved at the held copies
(`RunsAs.stmtOk_of_check`), the held copies agreeing with the run's reads given their parts (`heldAt_of_parts`), the
design evaluating at them at the accepted play (`twoHolds_at`), so the names distinct and the interior record
registering no value (`Round2.namesDistinct_of_two`, `recordsAt_of_held`), and the records naming their registrations'
names (`recordsHeldAt_of`). -/
theorem playAccepts_form (A : Live₂) (C : CaseOf) (S : SessionReading)
    (P' : Verity.Exchange → ℕ → Claims.Prover Unit) (hparts : PartsHeldAt A C)
    (hof : HeldOfAt A C) (hports : PortsDistinctAt C) (hst : StmtsRead A C S)
    (hrounds : RoundsHeldAt A C S) (hscope : ScopeHeldAt A C S) (hpin : PinHeldAt A C S)
    (hsess : SessionsRun A C S P') :
    PlayAccepts A C S P' := by
  intro o i it x m₁ hA hi hit hx hm₁
  have htwo : TwoHolds A.h := twoHolds_at hof hA hi hit
  exact playAccepts_of_run' A C S P' (ruleIsTwo C) (heldAt_of_parts hof hparts) htwo
    (SpotFlock.RunsAs.Round2.namesDistinct_of_two htwo) (recordsAt_of_held htwo (recordsHeldAt_of hof hports))
    (recordsRead C) (SpotFlock.RunsAs.stmtOk_of_check A.h) hst hrounds hscope hpin hsess o i it x m₁ hA hi hit hx hm₁

/-- **The form's accept is the run's accept** (`AcceptsRun` at the form): where the form's check accepts a play, every
run of `SpotFlock.V` on coins agreeing with what the check read, whose messages are what the check read, accepts. -/
theorem acceptsRun_form (A : Live₂) (x : Msg) (C : CaseOf) (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin) (P' : Verity.Exchange → ℕ → Claims.Prover Unit)
    (hsm : SessionMsgs A C S P') (hparts : PartsHeldAt A C) (hof : HeldOfAt A C)
    (hports : PortsDistinctAt C) (hst : StmtsRead A C S) (hrounds : RoundsHeldAt A C S)
    (hscope : ScopeHeldAt A C S) (hpin : PinHeldAt A C S) (hsess : SessionsRun A C S P') :
    AcceptsRun A x (formReading A C S dev) (FormAccepts C) :=
  acceptsRun_of' A x C S dev P' (sourcesFit A C)
    (playAccepts_form A C S P' hparts hof hports hst hrounds hscope hpin hsess) hsm

/-- **The form's accept is the run's accept, with every message reached** (`ReachedAccepts` at the form): where the
form's check accepts a play, every run of `SpotFlock.V` on coins agreeing with what the check read, whose messages
are what the check read wherever the run has one, accepts and has all the verifier's `k` messages. -/
theorem reachedAccepts_form (A : Live₂) (x : Msg) (C : CaseOf) (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin) (P' : Verity.Exchange → ℕ → Claims.Prover Unit)
    (hsm : SessionMsgs A C S P') (hparts : PartsHeldAt A C) (hof : HeldOfAt A C)
    (hports : PortsDistinctAt C) (hst : StmtsRead A C S) (hrounds : RoundsHeldAt A C S)
    (hscope : ScopeHeldAt A C S) (hpin : PinHeldAt A C S) (hsess : SessionsRun A C S P') :
    ReachedAccepts A x (formReading A C S dev) (FormAccepts C) :=
  reachedAccepts A x C S dev P' (sourcesFit A C)
    (playAccepts_form A C S P' hparts hof hports hst hrounds hscope hpin hsess) hsm

/-- **The cases at `x` carry the play's commitment where `C`'s do.** -/
theorem caseCommits_at {C : CaseOf} (x : Msg) (h : CaseCommits C) : CaseCommits (atStmt C x) :=
  fun o k hk => h o k (atStmt_some hk).1

/-- **`ReachedAccepts` at `C` from `ReachedAccepts` at the cases at `x`** (as `acceptsRun_at`). -/
theorem reachedAccepts_at {A : Live₂} {x : Msg} {C : CaseOf} {S : SessionReading}
    {dev : List Verity.Release → ℕ → Option FlockDeploy.Coin}
    (h : ReachedAccepts A x (formReading A (atStmt C x) S dev) (FormAccepts (atStmt C x))) :
    ReachedAccepts A x (formReading A C S dev) (FormAccepts C) := by
  intro o hA hx Ω P ω r hc hm
  obtain ⟨hA', hin⟩ := at_of_stmt hA hx
  obtain ⟨e₁, e₂, e₃, e₄⟩ := formReading_at (A := A) (S := S) (dev := dev) hin
  refine h o hA' (e₁.trans hx) Ω P ω r ?_ ?_
  · rw [e₂, e₄]; exact hc
  · rw [e₃]; exact hm

/-- **The form's accept is the run's accept, its premises asked at the plays at `x`**: `acceptsRun_form` at the cases
at `x` (`atStmt C x`), whose accepted plays are `C`'s at the statement `x`, the only ones `AcceptsRun` at `x` looks at
(`acceptsRun_at`). -/
theorem acceptsRun_form_at (A : Live₂) (x : Msg) (C : CaseOf) (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin) (P' : Verity.Exchange → ℕ → Claims.Prover Unit)
    (hsm : SessionMsgs A (atStmt C x) S P') (hparts : PartsHeldAt A (atStmt C x))
    (hof : HeldOfAt A (atStmt C x)) (hports : PortsDistinctAt (atStmt C x))
    (hst : StmtsRead A (atStmt C x) S) (hrounds : RoundsHeldAt A (atStmt C x) S)
    (hscope : ScopeHeldAt A (atStmt C x) S) (hpin : PinHeldAt A (atStmt C x) S)
    (hsess : SessionsRun A (atStmt C x) S P') :
    AcceptsRun A x (formReading A C S dev) (FormAccepts C) :=
  acceptsRun_at
    (acceptsRun_form A x (atStmt C x) S dev P' hsm hparts hof hports hst hrounds hscope hpin hsess)

/-- **The form's accept is the run's accept, with every message reached, its premises asked at the plays at `x`**:
`reachedAccepts_form` at the cases at `x` (`reachedAccepts_at`). -/
theorem reachedAccepts_form_at (A : Live₂) (x : Msg) (C : CaseOf) (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin) (P' : Verity.Exchange → ℕ → Claims.Prover Unit)
    (hsm : SessionMsgs A (atStmt C x) S P') (hparts : PartsHeldAt A (atStmt C x))
    (hof : HeldOfAt A (atStmt C x)) (hports : PortsDistinctAt (atStmt C x))
    (hst : StmtsRead A (atStmt C x) S) (hrounds : RoundsHeldAt A (atStmt C x) S)
    (hscope : ScopeHeldAt A (atStmt C x) S) (hpin : PinHeldAt A (atStmt C x) S)
    (hsess : SessionsRun A (atStmt C x) S P') :
    ReachedAccepts A x (formReading A C S dev) (FormAccepts C) :=
  reachedAccepts_at
    (reachedAccepts_form A x (atStmt C x) S dev P' hsm hparts hof hports hst hrounds hscope hpin hsess)

end Verity.SampledTwo
