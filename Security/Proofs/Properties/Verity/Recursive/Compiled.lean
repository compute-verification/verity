import Proofs.Properties.SampledProofs
import VStarRecursive

/-!
# Piece A of `acc_of_accepts`: what the compiled recursive component's acceptance unfolds to (PRF-202)

`VStar.Recursive.checkBlobs` accepting `(p, c, e, bs)` means, step by step through `checkBlobs`, `verdicts`,
`partVerdicts` and `partVerdicts.partsAt`:

* the inner session held off the blobs (`innerHeld`) and V\*'s statements at its shape (`RecVStar.statements`);
* the verifier's parts and the evidence's part sessions, one per statement;
* every part's `flock-verify verify --zk` verdict (`partVerdict`) set up and accepted, `.ok (.ok ())`;
* one-stage's check accepted the evidence with the one verdict of record `verdicts` computed.

The verdict of record says accepted exactly when every part's verdict does (`verdictJson_accepted`), and one-stage's check
accepts only verdicts that say accepted (`Properties.SampledProofs.acceptMeans`, `VerdictsAccept`), so no part's verdict
is a refusal. Nothing here reads the model.
-/

open Lean (Json)
open SampledProofs

namespace Verity.Recursive.Compiled

theorem bind_ok {ε α β : Type} {x : Except ε α} {f : α → Except ε β} {b : β} (h : (x >>= f) = .ok b) :
    ∃ a, x = .ok a ∧ f a = .ok b := by
  cases x with
  | error e => simp [bind, Except.bind] at h
  | ok a => exact ⟨a, rfl, h⟩

theorem toBool_ok {v : Flock.V Unit} (h : v.toBool = true) : v = .ok () := by
  cases v with
  | error e => simp [Except.toBool] at h
  | ok u => rfl

theorem jeq_true {b : Bool} (h : jeq (.bool b) (.bool true) = true) : b = true := by
  cases b
  · simp [jeq, Flock.canon, pure, Except.pure] at h
  · rfl

/-- The verdicts `partsAt` returns, all accepting, are each listed part's `partVerdict` at `.ok ()`. -/
theorem partsAt_ok (h : VStar.Recursive.InnerHeld) (ss : Array VStar.Stmt) (o : VStar.Recursive.Own)
    (pr : VStar.Recursive.Proofs) (bs : Core.Blobs) :
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

/-- `partVerdicts` returning only accepting verdicts: the inner session held, V\*'s statements at its shape, one part of
each side per statement, and each part's verdict `.ok (.ok ())`. -/
theorem partVerdicts_ok {o : VStar.Recursive.Own} {pr : VStar.Recursive.Proofs} {bs : Core.Blobs}
    {hd : VStar.Recursive.InnerHeld} {parts : Array (String × Flock.V Unit)}
    (hv : VStar.Recursive.partVerdicts o pr bs = .ok (hd, parts)) (hall : ∀ x ∈ parts, x.2.toBool = true) :
    ∃ ss, VStar.Recursive.innerHeld o pr bs = .ok hd ∧ Flock.RecVStar.statements hd.shape = .ok ss ∧
      o.parts.size = ss.size ∧ pr.parts.size = ss.size ∧
      ∀ (j : Nat) (ho : j < o.parts.size) (hp : j < pr.parts.size),
        VStar.Recursive.partVerdict hd j o.parts[j] pr.parts[j] bs = .ok (.ok ()) := by
  unfold VStar.Recursive.partVerdicts at hv
  obtain ⟨h, hh, hv⟩ := bind_ok hv
  obtain ⟨ss, hs, hv⟩ := bind_ok hv
  have h1 : o.parts.size = ss.size := by
    refine Classical.byContradiction fun hne => ?_
    simp [hne, throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at hv
  have h2 : pr.parts.size = ss.size := by
    refine Classical.byContradiction fun hne => ?_
    simp [h1, hne, throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at hv
  simp only [h1, h2, bne_self_eq_false, Bool.false_eq_true, ↓reduceIte] at hv
  obtain ⟨l, hl, hv⟩ := bind_ok hv
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.ok.inj hv)
  refine ⟨ss, hh, hs, h1, h2, fun j ho hp => ?_⟩
  obtain ⟨p, s, hp', hs', hpv⟩ := partsAt_ok h ss o pr bs (List.range ss.size) l hl
    (fun x hx => hall x (by simpa using hx)) j (List.mem_range.2 (h1 ▸ ho))
  rw [Array.getElem?_eq_getElem ho] at hp'
  rw [Array.getElem?_eq_getElem hp] at hs'
  cases hp'; cases hs'
  exact hpv

/-- `verdicts` returned: `partVerdicts` did, and where some part's verdict refused, the one verdict of record is
`verdictJson false …`. -/
theorem verdicts_shape {o : VStar.Recursive.Own} {session : String} {pr : VStar.Recursive.Proofs}
    {pu : Option (Array Nat)} {bs : Core.Blobs} {vs : Array (String × Json)}
    (h : VStar.Recursive.verdicts o session pr pu bs = .ok vs) :
    ∃ hd parts, VStar.Recursive.partVerdicts o pr bs = .ok (hd, parts) ∧
      ((parts.all fun (_, v) => v.toBool) = false → ∃ why units programSha,
        vs = #[(s!"flock-verify/{session}", VStar.Recursive.verdictJson false session why units
          (parts.map VStar.Recursive.partJson) programSha)]) := by
  unfold VStar.Recursive.verdicts at h
  rcases hpv : VStar.Recursive.partVerdicts o pr bs with e | ⟨hd, parts⟩
  · rw [hpv] at h; cases h
  · refine ⟨hd, parts, rfl, fun hall => ?_⟩
    rw [hpv] at h
    cases pu <;> simp [hall, bind, Except.bind, pure, Except.pure] at h <;> exact ⟨_, _, _, h.symm⟩

end Verity.Recursive.Compiled

namespace Verity.Recursive

open Compiled in
/-- **Piece A: the compiled component's acceptance, unfolded.** If `VStar.Recursive.checkBlobs` accepts, then the inner
session is held off the blobs at a shape `h.shape` with V\*'s statements `ss`, the verifier's parts and the evidence's part
sessions are one per statement, every part's verdict is set up and accepts (`.ok (.ok ())`), and one-stage's check
accepted the evidence with the verdict of record `verdicts` computed on the evidence's one session. -/
theorem parts_of_accepts {p : VStar.Recursive.Params} {c : OneStage.Commitment} {e : VStar.Recursive.Evidence}
    {bs : Core.Blobs} {pub : OneStage.Public} {prof : Core.Integrity}
    (hacc : VStar.Recursive.checkBlobs p c e bs = .accept pub prof) :
    ∃ (h : VStar.Recursive.InnerHeld) (ss : Array VStar.Stmt),
      VStar.Recursive.innerHeld p.own e.proofs bs = .ok h ∧ Flock.RecVStar.statements h.shape = .ok ss ∧
      p.own.parts.size = ss.size ∧ e.proofs.parts.size = ss.size ∧
      (∀ (j : Nat) (ho : j < p.own.parts.size) (hp : j < e.proofs.parts.size),
        VStar.Recursive.partVerdict h j p.own.parts[j] e.proofs.parts[j] bs = .ok (.ok ())) ∧
      ∃ session vs, e.base.sessions[0]? = some session ∧
        VStar.Recursive.verdicts p.own session e.proofs (some e.base.provedUnits) bs = .ok vs ∧
        OneStage.check p.base c { e.base with verdicts := vs } = .accept pub prof := by
  unfold VStar.Recursive.checkBlobs at hacc
  split at hacc
  · cases hacc
  · next session hs =>
    split at hacc
    · cases hacc
    · next vs hv =>
      obtain ⟨law, d, n, -, -, -, hva, -⟩ := _root_.Properties.SampledProofs.acceptMeans _ _ _ _ _ hacc
      obtain ⟨hd, parts, hpv, hrefused⟩ := verdicts_shape hv
      have hok : (parts.all fun (_, v) => v.toBool) = true := by
        cases hb : parts.all fun (_, v) => v.toBool
        · obtain ⟨why, units, programSha, hvs⟩ := hrefused hb
          have hmem : (s!"flock-verify/{session}", VStar.Recursive.verdictJson false session why units
              (parts.map VStar.Recursive.partJson) programSha) ∈ vs := by
            rw [hvs]; simp
          have hx := hva.1 _ hmem
          simp only [VStar.Recursive.verdictJson_accepted] at hx
          exact absurd (jeq_true (hx (by simp))) (by simp)
        · rfl
      have hall : ∀ x ∈ parts, x.2.toBool = true := by
        intro x hx
        obtain ⟨nm, v⟩ := x
        exact (Array.all_eq_true'.1 hok) _ hx
      obtain ⟨ss, hh, hss, h1, h2, hparts⟩ := partVerdicts_ok hpv hall
      exact ⟨hd, ss, hh, hss, h1, h2, hparts, session, vs, hs, hv, hacc⟩

end Verity.Recursive
