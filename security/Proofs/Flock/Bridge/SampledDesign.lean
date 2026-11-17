module

public import Proofs.Flock.Bridge.Draw
public import Proofs.Flock.Bridge.PopMatches
public import Proofs.Flock.Claims.Sampling.Select.Design
public import Spot.OneStage
import Proofs.Flock.Claims.Sampling.Select.Misses
import Proofs.Flock.Claims.Sampling.Select.Take

/-! The sampling design the deployed sample samples under (`Bridge.sampledDesign`). -/

@[expose] public section

namespace Flock.Bridge

open Properties.Spot

/-- **The sampling design the audit samples under**: the registration's sampling design object `lj` expanded over the strata of the descriptor
`desc` under the query `q` (`Draw.deriveSamplingDesign`, with the work table `tb`), else the sampling design `flockSamplingDesign` parses from it at the
population `n`, a Bernoulli one refused, as `OneStage.sampledFrom` samples (`sampledFrom_sampledDesign`). -/
def sampledDesign (desc : Claims.Msg) (q lj : Lean.Json) (tb : Option Lean.Json) (n : ℕ) : Except String Flock.Draw.SamplingDesign := do
  let p ← Spot.Program.parse ⟨desc.toArray⟩
  let derived ← Spot.Draw.deriveSamplingDesign p q lj tb
  let parsed ← Spot.Draw.flockSamplingDesign (SecurityProofs.samplingDesignObj lj derived) n
  Spot.OneStage.coveredDesign (derived.getD parsed)

theorem coveredDesign_bernoulli (a b : ℕ) :
    Spot.OneStage.coveredDesign (.bernoulli a b) =
      .error "the law is bernoulli, which the sampled audit's soundness statement doesn't cover" :=
  rfl

theorem coveredDesign_ok {l samplingDesign : Flock.Draw.SamplingDesign} (h : Spot.OneStage.coveredDesign l = .ok samplingDesign) :
    samplingDesign = l ∧ NotBernoulli l := by
  cases l
  case bernoulli a b =>
    rw [coveredDesign_bernoulli] at h
    cases h
  all_goals exact ⟨(Except.ok.inj h).symm, fun _ _ h' => by cases h'⟩

/-- **The units a sampling design samples**: a subset sampling design's `k`, a stratified or work sampling design's strata's `k`s summed; none for a Bernoulli
sampling design, whose count the coins set. -/
def samplingDesignK : Flock.Draw.SamplingDesign → ℕ
  | .subset k => k
  | .bernoulli _ _ => 0
  | .stratified _ St => (St.toList.map (·.k)).sum
  | .work _ ws => (ws.toList.map (·.k)).sum

theorem sampledFrom_sampledDesign {desc : Claims.Msg} {v : Spot.Docket.Verifier} {lo : Lean.Json}
    (hp : Spot.Program.parse ⟨desc.toArray⟩ = .ok v.program)
    (hlo : Spot.Draw.derive v.program v.query v.samplingDesign v.workTable = .ok lo) (n : ℕ) (src : ByteArray) :
    Spot.OneStage.sampledFrom v lo n src =
      (sampledDesign desc v.query v.samplingDesign v.workTable n >>= fun samplingDesign => Spot.Draw.sampleUnits samplingDesign n src) := by
  unfold Spot.Draw.derive at hlo
  obtain ⟨derived, hd, hlo⟩ := SecurityProofs.bind_eq_ok.1 hlo
  obtain rfl : SecurityProofs.samplingDesignObj v.samplingDesign derived = lo := Except.ok.inj hlo
  unfold Spot.OneStage.sampledFrom sampledDesign
  rw [hp, SecurityProofs.bind_ok, hd, SecurityProofs.bind_ok, SecurityProofs.bind_ok]
  cases Spot.Draw.flockSamplingDesign (SecurityProofs.samplingDesignObj v.samplingDesign derived) n <;> rfl

def SamplesUnder (desc : Claims.Msg) (q lj : Lean.Json) (tb : Option Lean.Json) (n : ℕ) (samplingDesign : Flock.Draw.SamplingDesign) (kd : ℕ) :
    Prop :=
  sampledDesign desc q lj tb n = .ok samplingDesign ∧ samplingDesignK samplingDesign = kd

theorem samplesUnder_of_sampledFrom {desc : Claims.Msg} {v : Spot.Docket.Verifier} {lo : Lean.Json}
    (hp : Spot.Program.parse ⟨desc.toArray⟩ = .ok v.program)
    (hlo : Spot.Draw.derive v.program v.query v.samplingDesign v.workTable = .ok lo) {n : ℕ} {src : ByteArray}
    {r : Array ℕ × ℕ} (h : Spot.OneStage.sampledFrom v lo n src = .ok r) :
    ∃ samplingDesign, SamplesUnder desc v.query v.samplingDesign v.workTable n samplingDesign (samplingDesignK samplingDesign) ∧
      Spot.Draw.sampleUnits samplingDesign n src = .ok r := by
  rw [sampledFrom_sampledDesign hp hlo] at h
  obtain ⟨samplingDesign, hl, hd⟩ := SecurityProofs.bind_eq_ok.1 h
  exact ⟨samplingDesign, ⟨hl, rfl⟩, hd⟩

theorem sampledDesign_ok {desc : Claims.Msg} {q lj : Lean.Json} {tb : Option Lean.Json} {n : ℕ} {samplingDesign : Flock.Draw.SamplingDesign}
    (h : sampledDesign desc q lj tb n = .ok samplingDesign) :
    ∃ p derived parsed, Spot.Program.parse ⟨desc.toArray⟩ = .ok p ∧
      Spot.Draw.deriveSamplingDesign p q lj tb = .ok derived ∧
      Spot.Draw.flockSamplingDesign (SecurityProofs.samplingDesignObj lj derived) n = .ok parsed ∧ samplingDesign = derived.getD parsed := by
  unfold sampledDesign at h
  obtain ⟨p, hp, h⟩ := SecurityProofs.bind_eq_ok.1 h
  obtain ⟨derived, hd, h⟩ := SecurityProofs.bind_eq_ok.1 h
  obtain ⟨parsed, hpar, h⟩ := SecurityProofs.bind_eq_ok.1 h
  exact ⟨p, derived, parsed, hp, hd, hpar, (coveredDesign_ok h).1⟩

theorem notBernoulli_sampledDesign {desc : Claims.Msg} {q lj : Lean.Json} {tb : Option Lean.Json} {n : ℕ}
    {samplingDesign : Flock.Draw.SamplingDesign} (h : sampledDesign desc q lj tb n = .ok samplingDesign) : NotBernoulli samplingDesign := by
  unfold sampledDesign at h
  obtain ⟨_, -, h⟩ := SecurityProofs.bind_eq_ok.1 h
  obtain ⟨_, -, h⟩ := SecurityProofs.bind_eq_ok.1 h
  obtain ⟨_, -, h⟩ := SecurityProofs.bind_eq_ok.1 h
  obtain ⟨rfl, hb⟩ := coveredDesign_ok h
  exact hb

theorem sampledDesign_cases {desc : Claims.Msg} {q lj : Lean.Json} {tb : Option Lean.Json} {n : ℕ} {samplingDesign : Flock.Draw.SamplingDesign}
    (h : sampledDesign desc q lj tb n = .ok samplingDesign) :
    (∃ k, samplingDesign = .subset k ∧ k ≤ n) ∨ (∃ a b, samplingDesign = .bernoulli a b) ∨
      ∃ p S k, Spot.Program.parse ⟨desc.toArray⟩ = .ok p ∧ Spot.Partition.strata p q = .ok S ∧
        (Flock.Draw.derive k S = .ok samplingDesign ∨ ∃ tab, Flock.Draw.deriveWork k S tab = .ok samplingDesign) := by
  obtain ⟨p, derived, parsed, hp, hd, hpar, rfl⟩ := sampledDesign_ok h
  rcases SecurityProofs.deriveSamplingDesign_cases p q lj tb derived hd with ⟨rfl, hk1, hk2⟩ | ⟨S, k, samplingDesign', hS, rfl, hlaw⟩
  · rcases SecurityProofs.flockSamplingDesign_cases _ _ _ hpar with ⟨k, rfl, hk, -⟩ | ⟨a, b, rfl, -⟩ | hk | hk
    · exact Or.inl ⟨k, rfl, hk⟩
    · exact Or.inr (Or.inl ⟨a, b, rfl⟩)
    · exact absurd hk hk1
    · exact absurd hk hk2
  · refine Or.inr (Or.inr ⟨p, S, k, hp, hS, ?_⟩)
    rcases hlaw with hder | ⟨-, tab, -, hwork⟩
    · exact Or.inl hder
    · exact Or.inr ⟨tab, hwork⟩

theorem strata_evaluate {p : Spot.Program} {q : Lean.Json} {S : Array (String × Array (ℕ × ℕ))}
    (hS : Spot.Partition.strata p q = .ok S) :
    ∃ rs, Spot.Partition.evaluate p q = .ok rs ∧
      S = Flock.Draw.strataOf (rs.map fun r => (r.template, r.first, r.count)) := by
  unfold Spot.Partition.strata at hS
  obtain ⟨rs, he, h⟩ := SecurityProofs.bind_eq_ok.1 hS
  exact ⟨rs, he, (Except.ok.inj h).symm⟩

theorem strata_pos {p : Spot.Program} {q : Lean.Json} {S : Array (String × Array (ℕ × ℕ))}
    (hS : Spot.Partition.strata p q = .ok S) :
    ∀ s ∈ S.toList, 1 ≤ (SecurityProofs.expand s.2.toList).length := by
  obtain ⟨rs, he, rfl⟩ := strata_evaluate hS
  exact SecurityProofs.strataOf_pos _ (SecurityProofs.partition_covered p q rs he)

theorem strata_nodup {p : Spot.Program} {q : Lean.Json} {S : Array (String × Array (ℕ × ℕ))}
    (hS : Spot.Partition.strata p q = .ok S) :
    (S.toList.flatMap fun s => SecurityProofs.expand s.2.toList).Nodup := by
  obtain ⟨rs, he, rfl⟩ := strata_evaluate hS
  exact (SecurityProofs.strataOf_perm _ _ (SecurityProofs.partition_evaluate_tiles p q rs he)).nodup_iff.2
    List.nodup_range'

theorem flatMap_expand_eq {α : Type} {f : α → Array (ℕ × ℕ)} {l : List α} {S : List (String × Array (ℕ × ℕ))}
    (h : l.map f = S.map (·.2)) :
    l.flatMap (fun a => SecurityProofs.expand (f a).toList) = S.flatMap fun s => SecurityProofs.expand s.2.toList := by
  have h1 := congrArg (fun m => m.flatMap fun u : Array (ℕ × ℕ) => SecurityProofs.expand u.toList) h
  simpa only [List.flatMap_map] using h1

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

theorem samplingDesignOn_sampledDesign {desc : Claims.Msg} {q lj : Lean.Json} {tb : Option Lean.Json} {n : ℕ} {samplingDesign : Flock.Draw.SamplingDesign}
    (hpop : PopMatches desc q n) (h : sampledDesign desc q lj tb n = .ok samplingDesign) (hb : NotBernoulli samplingDesign) :
    SamplingDesignOn samplingDesign (nsOf desc q) (samplingDesignK samplingDesign) := by
  have hsum : (nsOf desc q).sum = n := nsOf_sum hpop
  rcases sampledDesign_cases h with ⟨k, rfl, hk⟩ | ⟨a, b, rfl⟩ | ⟨p, S, k, hp, hS, hlaw⟩
  · exact ⟨rfl, by rw [hsum]; exact hk⟩
  · exact (hb a b rfl).elim
  obtain ⟨hns, hn⟩ := popMatches_strata hpop hp hS
  have hns' : (nsOf desc q).sum = SecurityProofs.strataSize S := by rw [hns]; rfl
  have hslots : (S.toList.flatMap fun s => SecurityProofs.expand s.2.toList) = List.range (nsOf desc q).sum := by
    rw [hsum, hn]
  rcases hlaw with hder | ⟨tab, hwork⟩
  · obtain ⟨-, hkn⟩ := SecurityProofs.derive_bounds k S samplingDesign hder
    obtain rfl := SecurityProofs.derive_ok k S samplingDesign hder
    refine ⟨⟨?_, ?_⟩, by rw [hns']; exact hkn, ?_, ?_, rfl⟩
    · rw [Array.toList_map, List.flatMap_map]; exact hslots
    · rw [hns, Array.toList_map]
      simp only [List.map_map, Function.comp_def, SecurityProofs.stratum_size, SecurityProofs.rsSize]
    · rw [hns]; intro x hx; obtain ⟨s, hs, rfl⟩ := List.mem_map.1 hx; exact strata_pos hS s hs
    · intro s hs; rw [Array.toList_map] at hs; obtain ⟨s₀, -, rfl⟩ := List.mem_map.1 hs
      simp only [SecurityProofs.stratum_size, hns']
  · obtain ⟨ws, hF, rfl⟩ := SecurityProofs.deriveWork_ok k S tab samplingDesign hwork
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

theorem samplingDesignSized_sampledDesign {desc : Claims.Msg} {q lj : Lean.Json} {tb : Option Lean.Json} {n : ℕ}
    {samplingDesign : Flock.Draw.SamplingDesign} (h : sampledDesign desc q lj tb n = .ok samplingDesign) (hb : NotBernoulli samplingDesign) :
    SamplingDesignSized samplingDesign (samplingDesignK samplingDesign) := by
  rcases sampledDesign_cases h with ⟨k, rfl, -⟩ | ⟨a, b, rfl⟩ | ⟨p, S, k, -, hS, hder | ⟨tab, hwork⟩⟩
  · exact rfl
  · exact (hb a b rfl).elim
  · obtain rfl := SecurityProofs.derive_ok k S samplingDesign hder
    refine ⟨?_, rfl⟩
    rw [Array.toList_map, List.flatMap_map]
    exact strata_nodup hS
  · obtain ⟨ws, hF, rfl⟩ := SecurityProofs.deriveWork_ok k S tab samplingDesign hwork
    refine ⟨?_, rfl⟩
    have h1 := flatMap_expand_eq (SecurityProofs.units_of_forall₂ _ _ hF)
    have h2 : (ws.toList.flatMap fun a => SecurityProofs.expand a.toStratum.units.toList).Nodup := by
      rw [h1]; exact strata_nodup hS
    simpa only [Array.toList_map, List.map_map, List.flatMap_map, Function.comp_def] using h2

theorem samplingDesignSized_samplesUnder {desc : Claims.Msg} {q lj : Lean.Json} {tb : Option Lean.Json} {n : ℕ}
    {samplingDesign : Flock.Draw.SamplingDesign} {kd : ℕ} (h : SamplesUnder desc q lj tb n samplingDesign kd) (hb : NotBernoulli samplingDesign) :
    SamplingDesignSized samplingDesign kd := by
  obtain ⟨h, rfl⟩ := h
  exact samplingDesignSized_sampledDesign h hb

theorem samplingDesignOn_samplesUnder {desc : Claims.Msg} {q lj : Lean.Json} {tb : Option Lean.Json} {n : ℕ}
    {samplingDesign : Flock.Draw.SamplingDesign} {kd : ℕ} (hpop : PopMatches desc q n) (h : SamplesUnder desc q lj tb n samplingDesign kd)
    (hb : NotBernoulli samplingDesign) : SamplingDesignOn samplingDesign (nsOf desc q) kd := by
  obtain ⟨h, rfl⟩ := h
  exact samplingDesignOn_sampledDesign hpop h hb

/-- **The sampling design the verifier samples under at a registration**: `sampledDesign`'s, and `subset 0`, which samples no
proof block, at a registration whose sampling design `sampledDesign` refuses, where `sampledFrom` refuses every sample
(`sampledFrom_sampledDesign`). -/
def designAt (desc : Claims.Msg) (q lj : Lean.Json) (tb : Option Lean.Json) (n : ℕ) : Flock.Draw.SamplingDesign :=
  match sampledDesign desc q lj tb n with
  | .ok samplingDesign => samplingDesign
  | .error _ => .subset 0

theorem designAt_cases (desc : Claims.Msg) (q lj : Lean.Json) (tb : Option Lean.Json) (n : ℕ) :
    SamplesUnder desc q lj tb n (designAt desc q lj tb n) (samplingDesignK (designAt desc q lj tb n)) ∨
      designAt desc q lj tb n = .subset 0 := by
  unfold designAt
  cases h : sampledDesign desc q lj tb n with
  | ok samplingDesign => exact Or.inl ⟨h, rfl⟩
  | error _ => exact Or.inr rfl

theorem sampledDesign_subset {desc : Claims.Msg} {q lj : Lean.Json} {tb : Option Lean.Json} {n : ℕ} {samplingDesign : Flock.Draw.SamplingDesign}
    (hs : SecurityProofs.kindOf lj = "subset") (h : sampledDesign desc q lj tb n = .ok samplingDesign) : ∃ k, samplingDesign = .subset k := by
  have hnb := notBernoulli_sampledDesign h
  obtain ⟨p, derived, parsed, -, hd, hpar, rfl⟩ := sampledDesign_ok h
  rw [SecurityProofs.deriveSamplingDesign_none p q lj tb (by rw [hs]; decide) (by rw [hs]; decide)] at hd
  obtain rfl : derived = none := (Except.ok.inj hd).symm
  rcases SecurityProofs.flockSamplingDesign_cases _ n parsed hpar with ⟨k, rfl, -⟩ | ⟨a, b, rfl, -⟩ | hk | hk
  · exact ⟨k, rfl⟩
  · exact (hnb a b rfl).elim
  all_goals
    rw [show SecurityProofs.samplingDesignObj lj none = lj from rfl, hs] at hk
    exact absurd hk (by decide)

theorem designAt_subset {desc : Claims.Msg} {q lj : Lean.Json} {tb : Option Lean.Json} (n : ℕ)
    (hs : SecurityProofs.kindOf lj = "subset") : designAt desc q lj tb n = .subset (samplingDesignK (designAt desc q lj tb n)) := by
  unfold designAt
  cases h : sampledDesign desc q lj tb n with
  | ok samplingDesign =>
    obtain ⟨k, rfl⟩ := sampledDesign_subset hs h
    rfl
  | error _ => rfl

theorem sampledFrom_designAt {desc : Claims.Msg} {v : Spot.Docket.Verifier} {lo : Lean.Json}
    (hp : Spot.Program.parse ⟨desc.toArray⟩ = .ok v.program)
    (hlo : Spot.Draw.derive v.program v.query v.samplingDesign v.workTable = .ok lo) {n : ℕ} {src : ByteArray}
    {r : Array ℕ × ℕ} (h : Spot.OneStage.sampledFrom v lo n src = .ok r) :
    Spot.Draw.sampleUnits (designAt desc v.query v.samplingDesign v.workTable n) n src = .ok r := by
  obtain ⟨samplingDesign, ⟨hl, -⟩, hd⟩ := samplesUnder_of_sampledFrom hp hlo h
  unfold designAt
  rw [hl]
  exact hd

theorem notBernoulli_samplesUnder {desc : Claims.Msg} {q lj : Lean.Json} {tb : Option Lean.Json} {n : ℕ}
    {samplingDesign : Flock.Draw.SamplingDesign} {kd : ℕ} (h : SamplesUnder desc q lj tb n samplingDesign kd) : NotBernoulli samplingDesign :=
  notBernoulli_sampledDesign h.1

theorem notBernoulli_designAt (desc : Claims.Msg) (q lj : Lean.Json) (tb : Option Lean.Json) (n : ℕ) :
    NotBernoulli (designAt desc q lj tb n) := by
  rcases designAt_cases desc q lj tb n with h | h
  · exact notBernoulli_samplesUnder h
  · rw [h]
    intro _ _ h'
    cases h'

theorem sized_designAt (desc : Claims.Msg) (q lj : Lean.Json) (tb : Option Lean.Json) (n : ℕ) :
    SamplingDesignSized (designAt desc q lj tb n) (samplingDesignK (designAt desc q lj tb n)) := by
  rcases designAt_cases desc q lj tb n with h | h
  · exact samplingDesignSized_samplesUnder h (notBernoulli_samplesUnder h)
  · rw [h]
    exact rfl

end Flock.Bridge
