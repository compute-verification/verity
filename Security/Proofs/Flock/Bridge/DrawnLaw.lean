module

public import Proofs.Flock.Bridge.Draw
public import Proofs.Flock.Bridge.PopMatches
public import Proofs.Flock.Claims.Sampling.Select.Design
public import Spot.OneStage
import Proofs.Flock.Claims.Sampling.Select.Misses
import Proofs.Flock.Claims.Sampling.Select.Take

/-!
# The sampling design the audit samples under (`Bridge.drawnLaw`)

The deployed sample (`Spot.OneStage.drawnFrom`) samples under the sampling design the registration's law object expands to
over its strata (`Draw.deriveLaw` over `Partition.strata`), else the sampling design `flockLaw` parses from the sampling design object,
and refuses a Bernoulli one (`OneStage.coveredDesign`).
`drawnLaw desc q lj tb n` is that sampling design, from the registration's descriptor bytes `desc`, query `q`, law object `lj` and
work table `tb`, at the population `n`:

* `drawnFrom_drawnLaw`: the deployed sample is `drawUnits` under `drawnLaw`, at a verifier whose program `desc` parses to;
* `notBernoulli_sampledDesign`, `notBernoulli_drawsUnder`: it is no Bernoulli law (`NotBernoulli`);
* `lawK law`: the units a sampling design samples, a subset sampling design's `k` or its strata's `k`s summed;
* `DrawsUnder desc q lj tb n law kd`: the verifier samples `kd` units under `law`: `law` is `drawnLaw`'s, and `kd` its
  `lawK`; it holds at every sample the one-stage verifier draws (`drawsUnder_of_drawnFrom`);
* `drawnLaw_cases`: `drawnLaw` is a subset sampling design within the population, a Bernoulli law, or the stratified or work sampling design
  `Flock.Draw.derive` or `deriveWork` expands over the registration's strata;
* `strata_pos`, `strata_nodup`: those strata each hold a unit, and their slots hold each unit once;
* `popMatches_strata`: at `PopMatches`, `nsOf` is those strata's sizes and their slots are the units `0, …, n − 1`;
* `lawSized_drawnLaw`, `lawSized_drawsUnder`: a `drawnLaw` that is no Bernoulli law (`NotBernoulli`) is sized
  (`LawSized`) at `lawK`, at any population;
* `lawOn_drawnLaw`, `lawOn_drawsUnder`: at `PopMatches`, it is on the registration's strata (`LawOn` at `nsOf`), sampling
  `lawK` units;
* `designAt desc q lj tb n`: `drawnLaw`'s sampling design, or `subset 0` where it errs (`designAt_cases`), under which every sample
  `drawnFrom` draws is drawn (`drawnFrom_designAt`), no Bernoulli law (`notBernoulli_designAt`) and sized (`sized_designAt`).
-/

@[expose] public section

namespace Flock.Bridge

open Properties.Spot

/-- **The sampling design the audit samples under**: the registration's law object `lj` expanded over the strata of the descriptor
`desc` under the query `q` (`Draw.deriveLaw`, with the work table `tb`), else the sampling design `flockLaw` parses from it at the
population `n`, a Bernoulli one refused, as `OneStage.drawnFrom` samples (`drawnFrom_drawnLaw`). -/
def drawnLaw (desc : Claims.Msg) (q lj : Lean.Json) (tb : Option Lean.Json) (n : ℕ) : Except String Flock.Draw.Law := do
  let p ← Spot.Program.parse ⟨desc.toArray⟩
  let derived ← Spot.Draw.deriveLaw p q lj tb
  let parsed ← Spot.Draw.flockLaw (SecurityProofs.lawObj lj derived) n
  Spot.OneStage.coveredDesign (derived.getD parsed)

/-- `OneStage.coveredDesign` refuses a Bernoulli sampling design. -/
theorem coveredDesign_bernoulli (a b : ℕ) :
    Spot.OneStage.coveredDesign (.bernoulli a b) =
      .error "the law is bernoulli, which the sampled audit's soundness statement doesn't cover" :=
  rfl

/-- `OneStage.coveredDesign` passes exactly the sampling designs that are no Bernoulli law, as they are. -/
theorem coveredDesign_ok {l law : Flock.Draw.Law} (h : Spot.OneStage.coveredDesign l = .ok law) :
    law = l ∧ NotBernoulli l := by
  cases l
  case bernoulli a b =>
    rw [coveredDesign_bernoulli] at h
    cases h
  all_goals exact ⟨(Except.ok.inj h).symm, fun _ _ h' => by cases h'⟩

/-- **The units a sampling design samples**: a subset sampling design's `k`, a stratified or work sampling design's strata's `k`s summed; none for a Bernoulli
law, whose count the coins set. -/
def lawK : Flock.Draw.Law → ℕ
  | .subset k => k
  | .bernoulli _ _ => 0
  | .stratified _ St => (St.toList.map (·.k)).sum
  | .work _ ws => (ws.toList.map (·.k)).sum

/-- **The deployed sample is `drawUnits` under `drawnLaw`**: at a verifier whose program the descriptor `desc` parses to,
under the sampling design object `Draw.derive` writes from its own, `drawnFrom` samples under the sampling design `drawnLaw` reads off its query,
law object and work table. -/
theorem drawnFrom_drawnLaw {desc : Claims.Msg} {v : Spot.Registration.Verifier} {lo : Lean.Json}
    (hp : Spot.Program.parse ⟨desc.toArray⟩ = .ok v.program)
    (hlo : Spot.Draw.derive v.program v.query v.law v.workTable = .ok lo) (n : ℕ) (src : ByteArray) :
    Spot.OneStage.drawnFrom v lo n src =
      (drawnLaw desc v.query v.law v.workTable n >>= fun law => Spot.Draw.drawUnits law n src) := by
  unfold Spot.Draw.derive at hlo
  obtain ⟨derived, hd, hlo⟩ := SecurityProofs.bind_eq_ok.1 hlo
  obtain rfl : SecurityProofs.lawObj v.law derived = lo := Except.ok.inj hlo
  unfold Spot.OneStage.drawnFrom drawnLaw
  rw [hp, SecurityProofs.bind_ok, hd, SecurityProofs.bind_ok, SecurityProofs.bind_ok]
  cases Spot.Draw.flockLaw (SecurityProofs.lawObj v.law derived) n <;> rfl

/-- **The verifier samples `kd` units under `law`**: `law` is the sampling design it samples under from the registration's descriptor
`desc`, query `q`, law object `lj` and work table `tb` at the population `n` (`drawnLaw`), and `kd` the units it samples
(`lawK`). -/
def DrawsUnder (desc : Claims.Msg) (q lj : Lean.Json) (tb : Option Lean.Json) (n : ℕ) (law : Flock.Draw.Law) (kd : ℕ) :
    Prop :=
  drawnLaw desc q lj tb n = .ok law ∧ lawK law = kd

/-- **Every sample the one-stage verifier draws is under a sampling design `DrawsUnder` names**: where `drawnFrom` samples,
at a verifier whose program the descriptor `desc` parses to and under the sampling design object `Draw.derive` writes,
its sample is `drawUnits` under a `law` with `DrawsUnder desc v.query v.law v.workTable n law (lawK law)`. -/
theorem drawsUnder_of_drawnFrom {desc : Claims.Msg} {v : Spot.Registration.Verifier} {lo : Lean.Json}
    (hp : Spot.Program.parse ⟨desc.toArray⟩ = .ok v.program)
    (hlo : Spot.Draw.derive v.program v.query v.law v.workTable = .ok lo) {n : ℕ} {src : ByteArray}
    {r : Array ℕ × ℕ} (h : Spot.OneStage.drawnFrom v lo n src = .ok r) :
    ∃ law, DrawsUnder desc v.query v.law v.workTable n law (lawK law) ∧
      Spot.Draw.drawUnits law n src = .ok r := by
  rw [drawnFrom_drawnLaw hp hlo] at h
  obtain ⟨law, hl, hd⟩ := SecurityProofs.bind_eq_ok.1 h
  exact ⟨law, ⟨hl, rfl⟩, hd⟩

/-- `drawnLaw`'s steps: the descriptor's program, the sampling design `deriveLaw` expands over its strata, if any, and the sampling design
`flockLaw` parses from the sampling design object `Draw.derive` writes. -/
theorem drawnLaw_ok {desc : Claims.Msg} {q lj : Lean.Json} {tb : Option Lean.Json} {n : ℕ} {law : Flock.Draw.Law}
    (h : drawnLaw desc q lj tb n = .ok law) :
    ∃ p derived parsed, Spot.Program.parse ⟨desc.toArray⟩ = .ok p ∧
      Spot.Draw.deriveLaw p q lj tb = .ok derived ∧
      Spot.Draw.flockLaw (SecurityProofs.lawObj lj derived) n = .ok parsed ∧ law = derived.getD parsed := by
  unfold drawnLaw at h
  obtain ⟨p, hp, h⟩ := SecurityProofs.bind_eq_ok.1 h
  obtain ⟨derived, hd, h⟩ := SecurityProofs.bind_eq_ok.1 h
  obtain ⟨parsed, hpar, h⟩ := SecurityProofs.bind_eq_ok.1 h
  exact ⟨p, derived, parsed, hp, hd, hpar, (coveredDesign_ok h).1⟩

/-- **The audit samples under no Bernoulli law**: `drawnLaw` refuses one (`OneStage.coveredDesign`). -/
theorem notBernoulli_sampledDesign {desc : Claims.Msg} {q lj : Lean.Json} {tb : Option Lean.Json} {n : ℕ}
    {law : Flock.Draw.Law} (h : drawnLaw desc q lj tb n = .ok law) : NotBernoulli law := by
  unfold drawnLaw at h
  obtain ⟨_, -, h⟩ := SecurityProofs.bind_eq_ok.1 h
  obtain ⟨_, -, h⟩ := SecurityProofs.bind_eq_ok.1 h
  obtain ⟨_, -, h⟩ := SecurityProofs.bind_eq_ok.1 h
  obtain ⟨rfl, hb⟩ := coveredDesign_ok h
  exact hb

/-- **`drawnLaw`'s cases**: a subset sampling design within the population, a Bernoulli law, or the sampling design `Flock.Draw.derive` or
`deriveWork` expands over the registration's strata (`Partition.strata`). -/
theorem drawnLaw_cases {desc : Claims.Msg} {q lj : Lean.Json} {tb : Option Lean.Json} {n : ℕ} {law : Flock.Draw.Law}
    (h : drawnLaw desc q lj tb n = .ok law) :
    (∃ k, law = .subset k ∧ k ≤ n) ∨ (∃ a b, law = .bernoulli a b) ∨
      ∃ p S k, Spot.Program.parse ⟨desc.toArray⟩ = .ok p ∧ Spot.Partition.strata p q = .ok S ∧
        (Flock.Draw.derive k S = .ok law ∨ ∃ tab, Flock.Draw.deriveWork k S tab = .ok law) := by
  obtain ⟨p, derived, parsed, hp, hd, hpar, rfl⟩ := drawnLaw_ok h
  rcases SecurityProofs.deriveLaw_cases p q lj tb derived hd with ⟨rfl, hk1, hk2⟩ | ⟨S, k, law', hS, rfl, hlaw⟩
  · rcases SecurityProofs.flockLaw_cases _ _ _ hpar with ⟨k, rfl, hk, -⟩ | ⟨a, b, rfl, -⟩ | hk | hk
    · exact Or.inl ⟨k, rfl, hk⟩
    · exact Or.inr (Or.inl ⟨a, b, rfl⟩)
    · exact absurd hk hk1
    · exact absurd hk hk2
  · refine Or.inr (Or.inr ⟨p, S, k, hp, hS, ?_⟩)
    rcases hlaw with hder | ⟨-, tab, -, hwork⟩
    · exact Or.inl hder
    · exact Or.inr ⟨tab, hwork⟩

/-- The registration's strata are the evaluated query's ranges, by template (`Partition.strata`). -/
theorem strata_evaluate {p : Spot.Program} {q : Lean.Json} {S : Array (String × Array (ℕ × ℕ))}
    (hS : Spot.Partition.strata p q = .ok S) :
    ∃ rs, Spot.Partition.evaluate p q = .ok rs ∧
      S = Flock.Draw.strataOf (rs.map fun r => (r.template, r.first, r.count)) := by
  unfold Spot.Partition.strata at hS
  obtain ⟨rs, he, h⟩ := SecurityProofs.bind_eq_ok.1 hS
  exact ⟨rs, he, (Except.ok.inj h).symm⟩

/-- **The registration's strata each hold a unit**: `TemplateQuery.evaluate` refuses a template with no instance
(`strataOf_pos`, `partition_covered`). -/
theorem strata_pos {p : Spot.Program} {q : Lean.Json} {S : Array (String × Array (ℕ × ℕ))}
    (hS : Spot.Partition.strata p q = .ok S) :
    ∀ s ∈ S.toList, 1 ≤ (SecurityProofs.expand s.2.toList).length := by
  obtain ⟨rs, he, rfl⟩ := strata_evaluate hS
  exact SecurityProofs.strataOf_pos _ (SecurityProofs.partition_covered p q rs he)

/-- **The registration's strata's slots hold each unit once**: they are the evaluated ranges, which tile the population
(`strataOf_perm`, `partition_evaluate_tiles`). -/
theorem strata_nodup {p : Spot.Program} {q : Lean.Json} {S : Array (String × Array (ℕ × ℕ))}
    (hS : Spot.Partition.strata p q = .ok S) :
    (S.toList.flatMap fun s => SecurityProofs.expand s.2.toList).Nodup := by
  obtain ⟨rs, he, rfl⟩ := strata_evaluate hS
  exact (SecurityProofs.strataOf_perm _ _ (SecurityProofs.partition_evaluate_tiles p q rs he)).nodup_iff.2
    List.nodup_range'

/-- Strata whose ranges are another list's, in order, have its slots. -/
theorem flatMap_expand_eq {α : Type} {f : α → Array (ℕ × ℕ)} {l : List α} {S : List (String × Array (ℕ × ℕ))}
    (h : l.map f = S.map (·.2)) :
    l.flatMap (fun a => SecurityProofs.expand (f a).toList) = S.flatMap fun s => SecurityProofs.expand s.2.toList := by
  have h1 := congrArg (fun m => m.flatMap fun u : Array (ℕ × ℕ) => SecurityProofs.expand u.toList) h
  simpa only [List.flatMap_map] using h1

/-- **At `PopMatches`, the registration's strata are its population**: where `desc` parses to `p`, whose strata under
the query `q` are `S` (`Partition.strata`), `nsOf` is their sizes and their slots, stratum by stratum, are the units
`0, …, n − 1`, since the layout `PopMatches` reads lays each stratum's units in order (`layout_ok`, `laysOut_units`). -/
theorem popMatches_strata {desc : Claims.Msg} {q : Lean.Json} {n : ℕ} {p : Spot.Program}
    {S : Array (String × Array (ℕ × ℕ))} (hpop : PopMatches desc q n)
    (hp : Spot.Program.parse ⟨desc.toArray⟩ = .ok p) (hS : Spot.Partition.strata p q = .ok S) :
    nsOf desc q = (S.toList.map fun s => (SecurityProofs.expand s.2.toList).length) ∧
      (S.toList.flatMap fun s => SecurityProofs.expand s.2.toList) = List.range n := by
  obtain ⟨p', l, hp', hl, hu⟩ := popMatches_iff.1 hpop
  have epp : p' = p := by rw [hp] at hp'; exact (Except.ok.inj hp').symm
  rw [epp] at hl
  obtain ⟨rs, he, pop, rfl, hlays⟩ := SecurityProofs.layout_ok p q l hl
  obtain ⟨rs', he', rfl⟩ := strata_evaluate hS
  obtain rfl : rs' = rs := by rw [he] at he'; exact (Except.ok.inj he').symm
  obtain ⟨h1, h2⟩ := SecurityProofs.laysOut_units _ _ hlays
  have hpo : popOf { classes := #[pop, SecurityProofs.inputClass p] } = pop.toList.flatMap (·.toList) := by
    simp [popOf, List.flatMap_def]
  refine ⟨?_, List.map_injective_iff.2 (Option.some_injective _) ?_⟩
  · rw [nsOf_eq hp hl]
    simpa [SecurityProofs.rsSize] using h2
  · rw [← h1, ← hu, hpo]

/-- **The sampling design the audit samples under is on the registration's strata**: at `PopMatches`, a `drawnLaw` that is not a
Bernoulli law is `LawOn` the strata's sizes `nsOf`, sampling `lawK` units: a subset sampling design within the `n` units they sum to,
or the stratified or work sampling design over the strata, whose slots are the units `0, …, n − 1` in order, each stratum's `k` the
rule's. -/
theorem lawOn_drawnLaw {desc : Claims.Msg} {q lj : Lean.Json} {tb : Option Lean.Json} {n : ℕ} {law : Flock.Draw.Law}
    (hpop : PopMatches desc q n) (h : drawnLaw desc q lj tb n = .ok law) (hb : NotBernoulli law) :
    LawOn law (nsOf desc q) (lawK law) := by
  have hsum : (nsOf desc q).sum = n := nsOf_sum hpop
  rcases drawnLaw_cases h with ⟨k, rfl, hk⟩ | ⟨a, b, rfl⟩ | ⟨p, S, k, hp, hS, hlaw⟩
  · exact ⟨rfl, by rw [hsum]; exact hk⟩
  · exact (hb a b rfl).elim
  obtain ⟨hns, hn⟩ := popMatches_strata hpop hp hS
  have hns' : (nsOf desc q).sum = SecurityProofs.strataSize S := by rw [hns]; rfl
  have hslots : (S.toList.flatMap fun s => SecurityProofs.expand s.2.toList) = List.range (nsOf desc q).sum := by
    rw [hsum, hn]
  rcases hlaw with hder | ⟨tab, hwork⟩
  · obtain ⟨-, hkn⟩ := SecurityProofs.derive_bounds k S law hder
    obtain rfl := SecurityProofs.derive_ok k S law hder
    refine ⟨⟨?_, ?_⟩, by rw [hns']; exact hkn, ?_, ?_, rfl⟩
    · rw [Array.toList_map, List.flatMap_map]; exact hslots
    · rw [hns, Array.toList_map]
      simp only [List.map_map, Function.comp_def, SecurityProofs.stratum_size, SecurityProofs.rsSize]
    · rw [hns]; intro x hx; obtain ⟨s, hs, rfl⟩ := List.mem_map.1 hx; exact strata_pos hS s hs
    · intro s hs; rw [Array.toList_map] at hs; obtain ⟨s₀, -, rfl⟩ := List.mem_map.1 hs
      simp only [SecurityProofs.stratum_size, hns']
  · obtain ⟨ws, hF, rfl⟩ := SecurityProofs.deriveWork_ok k S tab law hwork
    have hunits := SecurityProofs.units_of_forall₂ _ _ hF
    have htot : Flock.Draw.totalWork (ws.map fun s => { s with
        k := Flock.Draw.workK k s.work s.size (Flock.Draw.totalWork ws) s.floor }) = Flock.Draw.totalWork ws := by
      rw [SecurityProofs.totalWork_eq, SecurityProofs.totalWork_eq, Array.toList_map, List.map_map]
      rfl
    refine ⟨⟨?_, ?_⟩, ?_, rfl⟩
    · rw [Array.toList_map, ← hslots]
      simpa only [List.map_map, List.flatMap_map, Function.comp_def] using flatMap_expand_eq hunits
    · rw [hns, Array.toList_map]
      have h2 := congrArg (List.map fun u : Array (ℕ × ℕ) => (SecurityProofs.expand u.toList).length) hunits
      simpa only [List.map_map, Function.comp_def, SecurityProofs.stratum_size, SecurityProofs.rsSize] using h2
    · intro s hs; rw [Array.toList_map] at hs; obtain ⟨w, -, rfl⟩ := List.mem_map.1 hs
      exact (congrArg (fun t => Flock.Draw.workK k w.work w.size t w.floor) htot).symm

/-- **The sampling design the audit samples under is sized**, at any population: a `drawnLaw` that is not a Bernoulli law is a
subset sampling design, or a stratified or work sampling design whose strata's slots hold each unit once, at `lawK` (`LawSized`), so its sample
files hold `lawK` units (`lawSized_size`). -/
theorem lawSized_drawnLaw {desc : Claims.Msg} {q lj : Lean.Json} {tb : Option Lean.Json} {n : ℕ}
    {law : Flock.Draw.Law} (h : drawnLaw desc q lj tb n = .ok law) (hb : NotBernoulli law) :
    LawSized law (lawK law) := by
  rcases drawnLaw_cases h with ⟨k, rfl, -⟩ | ⟨a, b, rfl⟩ | ⟨p, S, k, -, hS, hder | ⟨tab, hwork⟩⟩
  · exact rfl
  · exact (hb a b rfl).elim
  · obtain rfl := SecurityProofs.derive_ok k S law hder
    refine ⟨?_, rfl⟩
    rw [Array.toList_map, List.flatMap_map]
    exact strata_nodup hS
  · obtain ⟨ws, hF, rfl⟩ := SecurityProofs.deriveWork_ok k S tab law hwork
    refine ⟨?_, rfl⟩
    have h1 := flatMap_expand_eq (SecurityProofs.units_of_forall₂ _ _ hF)
    have h2 : (ws.toList.flatMap fun a => SecurityProofs.expand a.toStratum.units.toList).Nodup := by
      rw [h1]; exact strata_nodup hS
    simpa only [Array.toList_map, List.map_map, List.flatMap_map, Function.comp_def] using h2

/-- **The sampling design the verifier samples under is sized**, at any population: `LawSized` at the `kd` units it samples, when it is
no Bernoulli law (`lawSized_drawnLaw`). -/
theorem lawSized_drawsUnder {desc : Claims.Msg} {q lj : Lean.Json} {tb : Option Lean.Json} {n : ℕ}
    {law : Flock.Draw.Law} {kd : ℕ} (h : DrawsUnder desc q lj tb n law kd) (hb : NotBernoulli law) :
    LawSized law kd := by
  obtain ⟨h, rfl⟩ := h
  exact lawSized_drawnLaw h hb

/-- **The sampling design the verifier samples under is on the registration's strata**: at `PopMatches`, `LawOn` at `nsOf` and the
`kd` units it samples, when it is no Bernoulli law (`lawOn_drawnLaw`). -/
theorem lawOn_drawsUnder {desc : Claims.Msg} {q lj : Lean.Json} {tb : Option Lean.Json} {n : ℕ}
    {law : Flock.Draw.Law} {kd : ℕ} (hpop : PopMatches desc q n) (h : DrawsUnder desc q lj tb n law kd)
    (hb : NotBernoulli law) : LawOn law (nsOf desc q) kd := by
  obtain ⟨h, rfl⟩ := h
  exact lawOn_drawnLaw hpop h hb

/-- **The sampling design the verifier samples under at a registration**: `drawnLaw`'s, and `subset 0`, which samples no
proof block, at a registration whose law `drawnLaw` refuses, where `drawnFrom` refuses every sample
(`drawnFrom_drawnLaw`). -/
def designAt (desc : Claims.Msg) (q lj : Lean.Json) (tb : Option Lean.Json) (n : ℕ) : Flock.Draw.Law :=
  match drawnLaw desc q lj tb n with
  | .ok law => law
  | .error _ => .subset 0

/-- `designAt` is the sampling design the verifier samples under (`DrawsUnder`, at its `lawK`), or `subset 0`. -/
theorem designAt_cases (desc : Claims.Msg) (q lj : Lean.Json) (tb : Option Lean.Json) (n : ℕ) :
    DrawsUnder desc q lj tb n (designAt desc q lj tb n) (lawK (designAt desc q lj tb n)) ∨
      designAt desc q lj tb n = .subset 0 := by
  unfold designAt
  cases h : drawnLaw desc q lj tb n with
  | ok law => exact Or.inl ⟨h, rfl⟩
  | error _ => exact Or.inr rfl

/-- **Every sample the one-stage verifier draws is under `designAt`**: where `drawnFrom` samples, at a verifier whose program
the descriptor `desc` parses to and under the sampling design object `Draw.derive` writes, its sample is `drawUnits` under
`designAt desc v.query v.law v.workTable n`. -/
theorem drawnFrom_designAt {desc : Claims.Msg} {v : Spot.Registration.Verifier} {lo : Lean.Json}
    (hp : Spot.Program.parse ⟨desc.toArray⟩ = .ok v.program)
    (hlo : Spot.Draw.derive v.program v.query v.law v.workTable = .ok lo) {n : ℕ} {src : ByteArray}
    {r : Array ℕ × ℕ} (h : Spot.OneStage.drawnFrom v lo n src = .ok r) :
    Spot.Draw.drawUnits (designAt desc v.query v.law v.workTable n) n src = .ok r := by
  obtain ⟨law, ⟨hl, -⟩, hd⟩ := drawsUnder_of_drawnFrom hp hlo h
  unfold designAt
  rw [hl]
  exact hd

/-- **The verifier samples under no Bernoulli law** (`notBernoulli_sampledDesign`). -/
theorem notBernoulli_drawsUnder {desc : Claims.Msg} {q lj : Lean.Json} {tb : Option Lean.Json} {n : ℕ}
    {law : Flock.Draw.Law} {kd : ℕ} (h : DrawsUnder desc q lj tb n law kd) : NotBernoulli law :=
  notBernoulli_sampledDesign h.1

/-- **`designAt` is no Bernoulli law** (`notBernoulli_drawsUnder`; `subset 0` is none). -/
theorem notBernoulli_designAt (desc : Claims.Msg) (q lj : Lean.Json) (tb : Option Lean.Json) (n : ℕ) :
    NotBernoulli (designAt desc q lj tb n) := by
  rcases designAt_cases desc q lj tb n with h | h
  · exact notBernoulli_drawsUnder h
  · rw [h]
    intro _ _ h'
    cases h'

/-- **`designAt` is sized** at its `lawK` (`lawSized_drawsUnder`; `subset 0` at `0`). -/
theorem sized_designAt (desc : Claims.Msg) (q lj : Lean.Json) (tb : Option Lean.Json) (n : ℕ) :
    LawSized (designAt desc q lj tb n) (lawK (designAt desc q lj tb n)) := by
  rcases designAt_cases desc q lj tb n with h | h
  · exact lawSized_drawsUnder h (notBernoulli_drawsUnder h)
  · rw [h]
    exact rfl

end Flock.Bridge
