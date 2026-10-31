import Proofs.Properties.Challenger
import Proofs.Properties.SampledProofs
import VStarRecursive

/-!
# A step of `acc_of_accepts`: what the compiled recursive component's acceptance unfolds to

`VStar.Recursive.checkBlobs` accepting `(p, c, e, bs)` means, step by step through `checkBlobs`, `verdicts`,
`partVerdicts` and `partVerdicts.partsAt`:

* the inner session held off the blobs (`innerHeld`) and V\*'s statements at its shape (`RecVStar.statements`);
* the shape dense (`dense`: public inner circuits, no RecSparse phase and no R);
* the verifier's parts and the evidence's part sessions, one per statement;
* every part's `flock-verify verify --zk` verdict (`partVerdict`) set up and accepted, `.ok (.ok ())`;
* one-stage v1's live check accepted the evidence with the one verdict of record `verdicts` computed, so its check at no
  coins did (`Properties.Challenger.oneStage_check_of_accept`).

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

/-- `need b why` passed: `b` held. -/
theorem need_ok {b : Bool} {why : String} {u : Unit} (h : VStar.Recursive.need b why = .ok u) : b = true := by
  unfold VStar.Recursive.need at h
  cases b
  · simp [throw, throwThe, MonadExceptOf.throw] at h
  · rfl

/-- `partVerdicts` returning only accepting verdicts: the inner session held, V\*'s statements at its shape, the shape
dense, one part of each side per statement, each part's verdict `.ok (.ok ())`, and the statement the inputs name under
`FIELD_MAX` bytes. -/
theorem partVerdicts_ok {o : VStar.Recursive.Own} {pr : VStar.Recursive.Proofs} {bs : Core.Blobs}
    {hd : VStar.Recursive.InnerHeld} {parts : Array (String × Flock.V Unit)}
    (hv : VStar.Recursive.partVerdicts o pr bs = .ok (hd, parts)) (hall : ∀ x ∈ parts, x.2.toBool = true) :
    ∃ ss, VStar.Recursive.innerHeld o pr bs = .ok hd ∧ Flock.RecVStar.statements hd.shape = .ok ss ∧
      VStar.Recursive.dense hd.shape = true ∧ o.parts.size = ss.size ∧ pr.parts.size = ss.size ∧
      (∀ (j : Nat) (ho : j < o.parts.size) (hp : j < pr.parts.size),
        VStar.Recursive.partVerdict hd j o.parts[j] pr.parts[j] bs = .ok (.ok ())) ∧
      VStar.Recursive.statementFits VStar.Recursive.FIELD_MAX hd o pr bs = .ok () := by
  unfold VStar.Recursive.partVerdicts at hv
  obtain ⟨h, hh, hv⟩ := bind_ok hv
  obtain ⟨ss, hs, hv⟩ := bind_ok hv
  obtain ⟨_, hd1, hv⟩ := bind_ok hv
  obtain ⟨_, hd2, hv⟩ := bind_ok hv
  obtain ⟨_, hd3, hv⟩ := bind_ok hv
  obtain ⟨l, hl, hv⟩ := bind_ok hv
  obtain ⟨⟨⟩, hfit, hv⟩ := bind_ok hv
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.ok.inj hv)
  have h1 := need_ok hd1
  have h2 := need_ok hd2
  have h3 := need_ok hd3
  simp only [beq_iff_eq] at h2 h3
  refine ⟨ss, hh, hs, h1, h2, h3, fun j ho hp => ?_, hfit⟩
  obtain ⟨p, s, hp', hs', hpv⟩ := partsAt_ok h ss o pr bs (List.range ss.size) l hl
    (fun x hx => hall x (by simpa using hx)) j (List.mem_range.2 (h2 ▸ ho))
  rw [Array.getElem?_eq_getElem ho] at hp'
  rw [Array.getElem?_eq_getElem hp] at hs'
  cases hp'; cases hs'
  exact hpv

/-- `verdicts` returned: `partVerdicts` did, and the one verdict of record says accepted exactly when every part's
verdict does. -/
theorem verdicts_shape {o : VStar.Recursive.Own} {session : String} {pr : VStar.Recursive.Proofs}
    {pu : Option (Array Nat)} {bs : Core.Blobs} {vs : Array (String × Json)}
    (h : VStar.Recursive.verdicts o session pr pu bs = .ok vs) :
    ∃ hd parts, VStar.Recursive.partVerdicts o pr bs = .ok (hd, parts) ∧
      ∃ why units programSha, vs = #[(s!"flock-verify/{session}", VStar.Recursive.verdictJson
        (parts.all (·.2.toBool)) session why units (parts.map VStar.Recursive.partJson) programSha)] := by
  unfold VStar.Recursive.verdicts at h
  obtain ⟨⟨hd, parts⟩, hpv, h⟩ := bind_ok h
  obtain ⟨_, -, h⟩ := bind_ok h
  obtain ⟨units, -, h⟩ := bind_ok h
  obtain ⟨_, -, h⟩ := bind_ok h
  exact ⟨hd, parts, hpv, _, units, _, (Except.ok.inj h).symm⟩

end Verity.Recursive.Compiled

namespace Verity.Recursive

open Compiled in
/-- **The compiled component's acceptance, unfolded.** If `VStar.Recursive.checkBlobs` accepts, then the inner
session is held off the blobs at a dense shape `h.shape` with V\*'s statements `ss`, the verifier's parts and the evidence's part
sessions are one per statement, every part's verdict is set up and accepts (`.ok (.ok ())`), the statement the inputs
name is under `FIELD_MAX` bytes (`statementFits`), and one-stage's check (its live check's, at no coins)
accepted the evidence with the verdict of record `verdicts` computed on the evidence's one session. -/
theorem parts_of_accepts {p : VStar.Recursive.Params} {c : OneStage.Commitment} {e : VStar.Recursive.Evidence}
    {bs : Core.Blobs} {pub : OneStage.Public} {prof : Core.Integrity}
    (hacc : VStar.Recursive.checkBlobs p c e bs = .accept pub prof) :
    ∃ (h : VStar.Recursive.InnerHeld) (ss : Array VStar.Stmt),
      VStar.Recursive.innerHeld p.own e.proofs bs = .ok h ∧ Flock.RecVStar.statements h.shape = .ok ss ∧
      VStar.Recursive.dense h.shape = true ∧ p.own.parts.size = ss.size ∧ e.proofs.parts.size = ss.size ∧
      (∀ (j : Nat) (ho : j < p.own.parts.size) (hp : j < e.proofs.parts.size),
        VStar.Recursive.partVerdict h j p.own.parts[j] e.proofs.parts[j] bs = .ok (.ok ())) ∧
      VStar.Recursive.statementFits VStar.Recursive.FIELD_MAX h p.own e.proofs bs = .ok () ∧
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
      have hacc := _root_.Properties.Challenger.oneStage_check_of_accept (p := p.base) (c := c)
        (e := { e.base with verdicts := vs }) hacc
      obtain ⟨law, d, n, -, -, -, hva, -⟩ := _root_.Properties.SampledProofs.acceptMeans _ _ _ _ _ hacc
      obtain ⟨hd, parts, hpv, why, units, programSha, hvs⟩ := verdicts_shape hv
      have hok : parts.all (·.2.toBool) = true := by
        have hmem : (s!"flock-verify/{session}", VStar.Recursive.verdictJson (parts.all (·.2.toBool)) session why
            units (parts.map VStar.Recursive.partJson) programSha) ∈ vs := by
          rw [hvs]; simp
        have hx := hva.1 _ hmem
        simp only [VStar.Recursive.verdictJson_accepted] at hx
        exact jeq_true (hx (by simp))
      have hall : ∀ x ∈ parts, x.2.toBool = true := fun x hx => (Array.all_eq_true'.1 hok) x hx
      obtain ⟨ss, hh, hss, hdn, h1, h2, hparts, hfit⟩ := partVerdicts_ok hpv hall
      exact ⟨hd, ss, hh, hss, hdn, h1, h2, hparts, hfit, session, vs, hs, hv, hacc⟩

end Verity.Recursive
