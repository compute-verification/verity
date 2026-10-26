import Proofs.Flock.Claims.Recursion.OpensReads
import Proofs.Flock.Claims.Recursion.PartsVStar
import Proofs.Flock.Claims.Recursion.RowsCover
import Proofs.Flock.C3.Reps.RoundLinks

/-!
# The holders of one round's row agree on its words

The deployed layout sizes each absorbed round's row by the first part that holds it (`rowSlot`), and every part that
holds it reads that row, so the proof of the honest prover's acceptance needs the holders of one round to read it at one
word count. The verifier checks it on V\*'s list (`RecVStar.checkRoundWords`, in `checkRoundsListed`, in
`RecVStar.statements`): every port under one `rec-c<k>` has one word count off its statement's description
(`RecVStar.ownWords`).

* `roundWords_of_check`, `words_of_roundWords`: the check gives one word count per `rec-c<k>`;
* `msgRound_inj`: two held rows of one inner round are under one `rec-c<k>` (`msgRound` is `r · |rs| + c_k`, with `c_k`
  a round of `rs` and the message rounds distinct);
* `ports_words`, `ownWords_rows`: a part's port's words, where V\*'s parse check passed (`checkParsed_ok`), are its
  statement's `ownWords`;
* `heldRows_words`: wherever `readX` reads `x`, two held rows of one round sit at ports of one word count, the hypothesis
  `hW` of `VStarBuild.fit_stream`.
-/

namespace RecDeploy

open Flock
open Claims (Msg)

/-- **`checkRoundsListed` runs the width check**: where it passes, `checkRoundWords` does. -/
theorem roundWords_of_check {i : RecVStar.Inner} {ss : Array VStar.Stmt}
    (h : RecVStar.checkRoundsListed i ss = .ok ()) : RecVStar.checkRoundWords ss = .ok () := by
  unfold RecVStar.checkRoundsListed at h
  split at h
  · split at h
    · cases h
    · exact h
  · cases h

/-- **One word count a round**: where `checkRoundWords` passes, two of the list's ports under one `rec-c<k>` have one
word count. -/
theorem words_of_roundWords {ss : Array VStar.Stmt} (h : RecVStar.checkRoundWords ss = .ok ())
    {p p' : ℕ × Option ℕ × String × String} (hp : p ∈ RecVStar.roundPorts ss) (hp' : p' ∈ RecVStar.roundPorts ss)
    (hk : p.1 = p'.1) : p.2.1 = p'.2.1 := by
  unfold RecVStar.checkRoundWords at h
  dsimp only at h
  split at h
  · cases h
  · rename_i hf
    have hall := List.find?_eq_none.1 hf
    have key : ∀ q ∈ RecVStar.roundPorts ss, ∃ f, (RecVStar.roundPorts ss).find? (·.1 == q.1) = some f ∧
        f.2.1 = q.2.1 := by
      intro q hq
      obtain ⟨f, hf'⟩ := Option.isSome_iff_exists.1
        ((List.find?_isSome (p := fun x : ℕ × Option ℕ × String × String => x.1 == q.1)).2 ⟨q, hq, by simp⟩)
      exact ⟨f, hf', by simpa [hf'] using hall q hq⟩
    obtain ⟨f, hf1, e1⟩ := key p hp
    obtain ⟨f', hf2, e2⟩ := key p' hp'
    rw [← hk, hf1] at hf2
    cases hf2
    rw [← e1, e2]

/-- **`msgRound` names one message round**: two of the reps' rounds at one place of the rep streams are one `k`. -/
theorem msgRound_inj {rs : Array RecVStar.SchedRound} {r₁ k₁ r₂ k₂ j : ℕ} (h₁ : msgRound rs r₁ k₁ = some j)
    (h₂ : msgRound rs r₂ k₂ = some j) : k₁ = k₂ := by
  unfold msgRound at h₁ h₂
  obtain ⟨a₁, ha₁, e₁⟩ := Option.map_eq_some_iff.1 h₁
  obtain ⟨a₂, ha₂, e₂⟩ := Option.map_eq_some_iff.1 h₂
  have hlt : ∀ {k a : ℕ}, (RecVStar.commitRounds rs)[k]? = some a → a < rs.size := by
    intro k a ha
    have hm := List.mem_of_getElem? ha
    unfold RecVStar.commitRounds at hm
    exact List.mem_range.1 (List.mem_filter.1 hm).1
  have l₁ := hlt ha₁
  have l₂ := hlt ha₂
  have ha : a₁ = a₂ := by
    have e : a₁ + r₁ * rs.size = a₂ + r₂ * rs.size := by rw [Nat.add_comm a₁, Nat.add_comm a₂]; exact e₁.trans e₂.symm
    have := congrArg (· % rs.size) e
    simpa [Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt l₁, Nat.mod_eq_of_lt l₂] using this
  subst ha
  have hnd : (RecVStar.commitRounds rs).Nodup := by
    unfold RecVStar.commitRounds
    exact List.Nodup.filter _ List.nodup_range
  obtain ⟨hk₁, g₁⟩ := List.getElem?_eq_some_iff.1 ha₁
  obtain ⟨hk₂, g₂⟩ := List.getElem?_eq_some_iff.1 ha₂
  exact (hnd.getElem_inj_iff).1 (g₁.trans g₂.symm)

/-- **A port's words are its row's**: where the circuit's ports are `rows` by name, words and bits, the port `findIdx?`
finds by name has the words of the row `find?` finds by name. -/
theorem ports_words {c : Circuit} {rows : List (String × ℕ × ℕ)}
    (hr : c.ports.toList.map (fun p => (p.name, p.words, p.bits)) = rows) {nm : String} {qi : ℕ}
    (hq : c.ports.findIdx? (·.name == nm) = some qi) :
    (c.ports[qi]?).map (·.words) = (rows.find? (·.1 == nm)).map (·.2.1) := by
  obtain ⟨hlt, hp, hne⟩ := Array.findIdx?_eq_some_iff_getElem.1 hq
  have hf : rows.find? (·.1 == nm) = some (c.ports[qi].name, c.ports[qi].words, c.ports[qi].bits) := by
    subst hr
    rw [List.find?_eq_some_iff_getElem]
    refine ⟨by simpa using hp, qi, by simpa using hlt, by simp, fun j hj => ?_⟩
    have := hne j hj
    simpa using this
  simp [hf, Array.getElem?_eq_getElem hlt]

/-- `zip` of two maps of one list. -/
theorem zip_map_map {α β γ : Type} (f : α → β) (g : α → γ) :
    ∀ l : List α, (l.map f).zip (l.map g) = l.map fun a => (f a, g a)
  | [] => rfl
  | a :: l => by simp [zip_map_map f g l]

/-- An input row's words, found by its name among the rows `A` lays first. -/
theorem find_rows {A : List (String × ℕ)} {rest : List (String × ℕ × ℕ)} {nm : String} {w : ℕ} (hm : (nm, w) ∈ A) :
    (((A.map fun p : String × ℕ => (p.1, p.2, 16 * p.2)) ++ rest).find? fun x => x.1 == nm).map
        (fun x => x.2.1) = (A.find? fun x => x.1 == nm).map (fun x => x.2) := by
  obtain ⟨a, ha⟩ := Option.isSome_iff_exists.1
    ((List.find?_isSome (p := fun x : String × ℕ => x.1 == nm)).2 ⟨_, hm, by simp⟩)
  simp [List.find?_append, List.find?_map, Function.comp_def, ha]

/-- **A message-round port's words off the description**: at a statement whose unit builds, a port of its own
(`ownPorts`) under a `rec-c<k>` has, among V\*'s rows of its unit (`vstarRows`), the words `ownWords` reads off the
statement. -/
theorem ownWords_rows {s : VStar.Stmt} {u : VStar.UnitC} (hu : s.unit = .ok u)
    {q : String × String × Array (Option ℕ)} (hq : q ∈ RecVStar.ownPorts s) {k : ℕ}
    (hk : RecVStar.cIndex q.2.1 = some k) :
    ((RecVStar.vstarRows s.sub u).find? (·.1 == q.1)).map (·.2.1) = RecVStar.ownWords s q.1 := by
  cases s with
  | inner j name p =>
    unfold RecVStar.ownPorts at hq
    obtain ⟨⟨nm, x⟩, hmem, h⟩ := List.mem_filterMap.1 hq
    dsimp only at h
    obtain ⟨k', -, rfl⟩ := Option.map_eq_some_iff.1 h
    have hu' : VStar.innerUnit p.S name = .ok u := hu
    unfold VStar.innerUnit at hu'
    obtain ⟨_, -, hu'⟩ := FlockC3.VStarList.eok_bind hu'
    obtain ⟨_, -, hu'⟩ := FlockC3.VStarList.eok_bind hu'
    cases hu'
    have hz : (p.S.ports.map (·.1) ++ ["v"]).zip
        (p.S.ports.map (fun p => VStar.portWordsOf p.2) ++ [VStar.portWordsOf p.S.gf.nv]) =
        p.S.ports.map (fun p => (p.1, VStar.portWordsOf p.2)) ++ [("v", VStar.portWordsOf p.S.gf.nv)] := by
      simp [List.zip_append, zip_map_map]
    have hm : (nm, VStar.portWordsOf x) ∈ p.S.ports.map (fun p => (p.1, VStar.portWordsOf p.2)) :=
      List.mem_map.2 ⟨(nm, x), hmem, rfl⟩
    show (((((p.S.ports.map (·.1) ++ ["v"]).zip (p.S.ports.map (fun p => VStar.portWordsOf p.2) ++
        [VStar.portWordsOf p.S.gf.nv])).map fun p : String × ℕ => (p.1, p.2, 16 * p.2)) ++ _).find? fun x => x.1 == nm).map (fun x => x.2.1) =
      (p.S.ports.find? fun x => x.1 == nm).map (fun x => VStar.portWordsOf x.2)
    rw [hz, List.map_append, List.append_assoc, find_rows hm]
    simp [List.find?_map, Function.comp_def]
  | sparse sh hr =>
    unfold RecVStar.ownPorts at hq
    have hu' : VStar.sparseUnit sh = .ok u := hu
    have hin : u.inPorts = (RecSparse.ports sh).map (·.2) := by
      unfold VStar.sparseUnit at hu'
      revert hu'
      split <;> intro h
      · cases h
      · cases h
        rfl
    have key : ∀ nm w, (nm, w) ∈ RecSparse.ports sh →
        ((RecVStar.vstarRows (VStar.Stmt.sparse sh hr).sub u).find? (·.1 == nm)).map (·.2.1) =
          RecVStar.ownWords (.sparse sh hr) nm := by
      intro nm w hmem
      have hz : ((RecSparse.ports sh).map (·.1)).zip u.inPorts = (RecSparse.ports sh).map (fun p => (p.1, p.2)) := by
        rw [hin]
        exact zip_map_map _ _ _
      have hm' : (nm, w) ∈ (RecSparse.ports sh).map (fun p => (p.1, p.2)) := by simpa using hmem
      show ((((((RecSparse.ports sh).map (·.1)).zip u.inPorts).map fun p : String × ℕ => (p.1, p.2, 16 * p.2)) ++
          _).find? fun x => x.1 == nm).map (fun x => x.2.1) = ((RecSparse.ports sh).find? fun x => x.1 == nm).map
          (fun x => x.2)
      rw [hz, find_rows hm']
      simp [List.find?_map, Function.comp_def]
    have hmsg : ∀ nm n, (nm, n) ∈ RecSparse.messagePorts sh → (nm, RecSparse.portWords n) ∈ RecSparse.ports sh := by
      intro nm n hmem
      unfold RecSparse.ports
      exact List.mem_append_left _ (List.mem_append_left _ (List.mem_map.2 ⟨(nm, n), hmem, rfl⟩))
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hq
    rcases hq with rfl | rfl
    · exact key "h" _ (hmsg "h" RecSparse.SKIP (by simp [RecSparse.messagePorts]))
    · exact key "claim" _ (hmsg "claim" 1 (by simp [RecSparse.messagePorts]))
  | opening _ _ _ _ _ _ =>
    unfold RecVStar.ownPorts at hq
    obtain ⟨⟨nm, _⟩, -, h⟩ := List.mem_filterMap.1 hq
    dsimp only at h
    cases hm : RecVStar.mIndex nm with
    | some k' =>
      rw [hm] at h
      cases h
      rw [FlockC3.Reps.cIndex_opening] at hk
      cases hk
    | none =>
      cases hc : RecVStar.claimIndex nm with
      | some k' =>
        rw [hm, hc] at h
        cases h
        rw [cIndex_claim] at hk
        cases hk
      | none =>
        rw [hm, hc] at h
        cases h
  | recOpen _ _ _ _ => simp [RecVStar.ownPorts] at hq
  | recOpenStacked _ _ _ _ => simp [RecVStar.ownPorts] at hq
  | cap _ _ => simp [RecVStar.ownPorts] at hq

/-- **The holders of one round's row agree on its words**: wherever `readX` reads `x`, two rows the parts hold
(`heldRows`) at one round of the rep streams sit at ports of one word count. The rows are under one `rec-c<k>`
(`msgRound_inj`); each port's words are its statement's `ownWords` (V\*'s parse check, `ownWords_rows`), and V\*'s list
holds every port under one `rec-c<k>` at one word count (`checkRoundWords`, in `RecVStar.statements`). -/
theorem heldRows_words {x : Msg} {r : XR} (hx : readX x = some r) :
    ∀ s₁ c₁ h₁ s₂ c₂ h₂, r.w.parts[s₁]?.bind partCircuit = some c₁ → h₁ ∈ heldRows r s₁ →
      r.w.parts[s₂]?.bind partCircuit = some c₂ → h₂ ∈ heldRows r s₂ → h₁.1 = h₂.1 →
      (c₁.ports[h₁.2.2.1]?).map (·.words) = (c₂.ports[h₂.2.2.1]?).map (·.words) := by
  intro s₁ c₁ h₁ s₂ c₂ h₂ hc₁ hh₁ hc₂ hh₂ hj
  obtain ⟨vs, hvs, hlen, hall⟩ := readX_partVStar hx
  have hck := roundWords_of_check (roundsListed_of_statements hvs)
  have side : ∀ {s c h}, r.w.parts[s]?.bind partCircuit = some c → h ∈ heldRows r s →
      ∃ k rs, repSched r.shape = some rs ∧ msgRound rs h.2.2.2 k = some h.1 ∧
        ∃ p ∈ RecVStar.roundPorts vs, p.1 = k ∧ (c.ports[h.2.2.1]?).map (·.words) = p.2.1 := by
    intro s c h hc hh
    have hs : s < vs.size := by
      rw [← hlen]
      cases hp : r.w.parts[s]? with
      | none => rw [hp] at hc; cases hc
      | some _ => exact (List.getElem?_eq_some_iff.1 hp).1
    have hsv : (RecVStar.statements r.shape).toOption.bind (·[s]?) = some vs[s] := by
      rw [hvs]
      simp [Except.toOption, Array.getElem?_eq_getElem hs]
    obtain ⟨q, hq, k, rs, hk, -, -, hqi, hrs, hmr⟩ := mem_heldRows_round hsv hc hh
    obtain ⟨f, p, c', -, hp, hpc, -, hpa, -⟩ := hall s hs
    rw [hp] at hc
    simp only [Option.bind_some] at hc
    rw [hpc] at hc
    cases hc
    obtain ⟨u, hu, -, -, -, hrows⟩ := FlockSoundness.Discharge.Recursion.checkParsed_ok hpa
    refine ⟨k, rs, hrs, hmr, (k, RecVStar.ownWords vs[s] q.1, vs[s].name, q.1), ?_, rfl, ?_⟩
    · unfold RecVStar.roundPorts
      refine List.mem_flatMap.2 ⟨vs[s], by simp, List.mem_filterMap.2 ⟨q, hq, ?_⟩⟩
      rw [hk]
      rfl
    · exact (ports_words hrows hqi).trans (ownWords_rows hu hq hk)
  obtain ⟨k₁, rs₁, hrs₁, hm₁, p₁, hp₁, hk₁, hw₁⟩ := side hc₁ hh₁
  obtain ⟨k₂, rs₂, hrs₂, hm₂, p₂, hp₂, hk₂, hw₂⟩ := side hc₂ hh₂
  rw [hrs₁] at hrs₂
  cases hrs₂
  rw [hj] at hm₁
  have hkk : k₁ = k₂ := msgRound_inj hm₁ hm₂
  rw [hw₁, hw₂]
  exact words_of_roundWords hck hp₁ hp₂ (by rw [hk₁, hk₂, hkk])

end RecDeploy
