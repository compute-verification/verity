import Properties.SampledProofs.Recursive
import Proofs.Properties.SampledProofs

/-! # Proofs of the recursive audit's property (`Properties.SampledProofs.Recursive`) -/

open Lean (Json)
open SampledProofs

namespace Properties.SampledProofs.Recursive

open VStar.Recursive (Own Proofs Params Evidence InnerHeld)

theorem bind_ok {ε α β : Type} {x : Except ε α} {f : α → Except ε β} {b : β} (h : (x >>= f) = .ok b) :
    ∃ a, x = .ok a ∧ f a = .ok b := by
  cases x with
  | error e => simp [bind, Except.bind] at h
  | ok a => exact ⟨a, rfl, h⟩

theorem need_ok {b : Bool} {why : String} {u : Unit} (h : VStar.Recursive.need b why = .ok u) : b = true := by
  unfold VStar.Recursive.need at h
  cases b
  · simp [throw, throwThe, MonadExceptOf.throw] at h
  · rfl

theorem dense_iff (i : Flock.RecVStar.Inner) : VStar.Recursive.dense i = true ↔ Dense i := by
  obtain ⟨m, k, c, r, holo, sparse, stacked⟩ := i
  cases holo <;> cases sparse <;> cases stacked <;> simp [VStar.Recursive.dense, Dense]

theorem toBool_ok {v : Flock.V Unit} (h : v.toBool = true) : v = .ok () := by
  cases v with
  | error e => simp [Except.toBool] at h
  | ok u => rfl

theorem jeq_true {b : Bool} (h : jeq (.bool b) (.bool true) = true) : b = true := by
  cases b
  · simp [jeq, Flock.canon, pure, Except.pure] at h
  · rfl

/-- The parts' verdicts `partsAt` returns, all accepting, are each listed part's `partVerdict` at `.ok ()`. -/
theorem partsAt_ok (h : InnerHeld) (ss : Array VStar.Stmt) (o : Own) (pr : Proofs) (bs : Core.Blobs) :
    ∀ (js : List Nat) (l : List (String × Flock.V Unit)),
      VStar.Recursive.partVerdicts.partsAt h ss o pr bs js = .ok l → (∀ x ∈ l, x.2.toBool = true) →
      ∀ j ∈ js, ∃ p s, o.parts[j]? = some p ∧ pr.parts[j]? = some s ∧
        VStar.Recursive.partVerdict h j p s bs = .ok (.ok ())
  | [], _, _, _, j, hj => by simp at hj
  | j :: js, l, hl, hall, j', hj' => by
    rw [VStar.Recursive.partVerdicts.partsAt] at hl
    rcases h1 : ss[j]? with _ | sj <;> rcases h2 : o.parts[j]? with _ | p <;>
      rcases h3 : pr.parts[j]? with _ | s <;> simp only [h1, h2, h3] at hl <;>
      try (simp [throw, throwThe, MonadExceptOf.throw] at hl; done)
    obtain ⟨v, hv, hl⟩ := bind_ok hl
    obtain ⟨l', hr, hl⟩ := bind_ok hl
    have hl : (sj.name, v) :: l' = l := Except.ok.inj hl
    subst hl
    rcases List.mem_cons.1 hj' with rfl | hj'
    · refine ⟨p, s, h2, h3, ?_⟩
      rw [hv, toBool_ok (v := v) (hall _ List.mem_cons_self)]
    · exact partsAt_ok h ss o pr bs js l' hr (fun x hx => hall x (List.mem_cons_of_mem _ hx)) j' hj'

theorem partVerdicts_ok {o : Own} {pr : Proofs} {bs : Core.Blobs} {hd : InnerHeld}
    {parts : Array (String × Flock.V Unit)} (hv : VStar.Recursive.partVerdicts o pr bs = .ok (hd, parts))
    (hall : parts.all (·.2.toBool) = true) : ∃ ss, PartsAccept o pr bs hd ss := by
  unfold VStar.Recursive.partVerdicts at hv
  obtain ⟨h, hh, hv⟩ := bind_ok hv
  obtain ⟨ss, hs, hv⟩ := bind_ok hv
  obtain ⟨_, hd1, hv⟩ := bind_ok hv
  obtain ⟨_, hd2, hv⟩ := bind_ok hv
  obtain ⟨_, hd3, hv⟩ := bind_ok hv
  obtain ⟨l, hl, hv⟩ := bind_ok hv
  have hv : (h, l.toArray) = (hd, parts) := Except.ok.inj hv
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj hv
  have h1 := need_ok hd1
  have h2 := need_ok hd2
  have h3 := need_ok hd3
  simp only [beq_iff_eq] at h2 h3
  have hall' : ∀ x ∈ l, x.2.toBool = true := by
    intro x hx
    simpa using (Array.all_eq_true'.1 hall) x (by simpa using hx)
  refine ⟨ss, hh, (dense_iff _).1 h1, hs, h2, h3, fun j ho hp => ?_⟩
  obtain ⟨p, s, hp', hs', hpv⟩ := partsAt_ok h ss o pr bs (List.range ss.size) l hl hall' j
    (List.mem_range.2 (h2 ▸ ho))
  rw [Array.getElem?_eq_getElem ho] at hp'
  rw [Array.getElem?_eq_getElem hp] at hs'
  cases hp'; cases hs'
  exact hpv

theorem verdicts_ok {o : Own} {session : String} {pr : Proofs} {pu : Array Nat} {bs : Core.Blobs}
    {vs : Array (String × Json)} (h : VStar.Recursive.verdicts o session pr (some pu) bs = .ok vs) :
    ∃ hd parts, VStar.Recursive.partVerdicts o pr bs = .ok (hd, parts) ∧
      ∃ why units programSha, vs = #[(s!"flock-verify/{session}",
        VStar.Recursive.verdictJson (parts.all (·.2.toBool)) session why units (parts.map VStar.Recursive.partJson)
          programSha)] ∧
      (parts.all (·.2.toBool) = true → UnitsOfInner o pr bs pu) := by
  unfold VStar.Recursive.verdicts at h
  obtain ⟨⟨hd, parts⟩, hpv, h⟩ := bind_ok h
  obtain ⟨_, -, h⟩ := bind_ok h
  obtain ⟨units, hu, h⟩ := bind_ok h
  obtain ⟨_, hn, h⟩ := bind_ok h
  refine ⟨hd, parts, hpv, _, units, _, (Except.ok.inj h).symm, fun hok => ?_⟩
  have hn := need_ok hn
  dsimp only at hu hn
  simp only [VStar.Recursive.unitsOf, hok, ↓reduceIte] at hu
  obtain ⟨pubB, hpub, hu⟩ := bind_ok hu
  obtain ⟨recB, hrec, hu⟩ := bind_ok hu
  simp [hok] at hn
  exact ⟨pubB, recB, units, hpub, hrec, hu, hn⟩

theorem acceptMeans : AcceptMeans := by
  intro p c e bs pub prof h
  unfold VStar.Recursive.checkBlobs at h
  split at h
  · cases h
  · next session hs =>
    split at h
    · cases h
    · next vs hv =>
      obtain ⟨law, d, n, -, -, -, hva, -⟩ := _root_.Properties.SampledProofs.acceptMeans _ _ _ _ _ h
      obtain ⟨hd, parts, hpv, why, units, programSha, hvs, hunits⟩ := verdicts_ok hv
      have hok : parts.all (·.2.toBool) = true := by
        have hmem : (s!"flock-verify/{session}", VStar.Recursive.verdictJson (parts.all (·.2.toBool)) session why
            units (parts.map VStar.Recursive.partJson) programSha) ∈ vs := by
          rw [hvs]; simp
        have hx := hva.1 _ hmem
        simp only [VStar.Recursive.verdictJson_accepted] at hx
        exact jeq_true (hx (by simp))
      exact ⟨⟨hd, partVerdicts_ok hpv hok⟩, hunits hok, session, vs, hs, hv, h⟩

theorem integrityMeans : IntegrityMeans := by
  refine ⟨fun _ _ _ _ => rfl, fun _ _ _ _ => rfl, ?_⟩
  intro p c e bs pub prof h
  cases p <;> cases e
  · exact .inl ⟨rfl, rfl⟩
  · cases h
  · cases h
  · exact .inr ⟨rfl, rfl⟩

end Properties.SampledProofs.Recursive
