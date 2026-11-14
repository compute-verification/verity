module

public import Proofs.Flock.SpotFlock.RunsAs.Released
public import Proofs.Flock.SpotFlock.RunsAs.Proved

/-!
# A two-stage run `Run.run` accepts is a play `SpotFlock.V h` accepts

`spotRunAcceptsPlay` assembles the parts. The accepted run's steps (part 5, `accepted_steps`) give the first stage's
select, the second stage's and round 2's check with no code. The held copies agree with what the run reads
(`HeldMatches`), so the model's replay sample on round 0's coin is the select's units (part 1, `sample_read`), which are
the units the sample proves (`proved_select`), and the model's second sample is the run's (part 2, `secondDraw_read`). What
the verifier holds is the model's (`OwnAgrees`), round 2 had statements (`statements_some`), and round 2's check gives
the model's statement check (part 3, `RoundTwo`): `stmtOk_of_accepted`, which reads no session. The sessions accept
(`SessionsAccept`), so the play accepts on its coins (part 4, `accepts_seqCoins`). The clauses round 2 does not make
are the statement's premises.
-/

@[expose] public section

namespace SpotFlock.RunsAs

open Claims
open Lean (Json)
open Round2 (core OwnAgrees NamesDistinct StmtsAre TwoHolds RecordsHeld RegistersNone RoundsHeld ScopeHeld PinHeld)

/-- Failing a check at each of a list's entries leaves a code where the check had one or the list has an entry. -/
theorem foldl_fail_ne : ∀ (l : List (String × Json)) (c : Spot.Check), (c.codes ≠ #[] ∨ l ≠ []) →
    (l.foldl (fun c x => c.fail x.1 x.2) c).codes ≠ #[]
  | [], c, h => by simpa using h
  | x :: l, c, _ => foldl_fail_ne l _ (Or.inl (Properties.Spot.fail_ne_nil c x.1 x.2))

/-- **Round 2's check with no code had statements**: without them it fails at the cells, the statements or the
design. -/
theorem statements_some {b : Bool} {own : Spot.SecondStage.Own} {stmts : Option (Array Json)}
    (h : (Spot.SecondStage.check b own stmts).codes = #[]) : ∃ ss, stmts = some ss := by
  cases stmts with
  | some ss => exact ⟨ss, rfl⟩
  | none =>
    exfalso
    unfold Spot.SecondStage.check at h
    cases hp : own.parts with
    | none =>
      simp only [hp] at h
      exact foldl_fail_ne _ _ (Or.inr (by simp)) h
    | some ps =>
      simp only [hp] at h
      refine foldl_fail_ne _ _ (Or.inr ?_) h
      cases hd : Spot.SecondStage.designOk ps own
      · simp only [hd, Bool.false_eq_true, ↓reduceIte]
        unfold Spot.SecondStage.designOk at hd
        unfold Spot.SecondStage.designFailures
        cases hl : Spot.SecondStage.linksOk ps own <;> cases hw : Spot.SecondStage.wiresOk ps own <;> simp_all
      · simp [hd]

/-- **A two-stage rule resolves to a two-stage rule**, never to `all`. -/
theorem resolve_ne_all {rule : Spot.Select.Rule} {p query} {resolved : Spot.Select.Rule}
    {grain : Json} {replay : String} {k : Spot.Select.Counts} (hr : rule = .twoStage grain replay k)
    (h : Spot.Select.resolve rule p query = .ok resolved) : resolved ≠ .all := by
  subst hr
  unfold Spot.Select.resolve at h
  obtain ⟨n, -, h⟩ := Properties.Spot.SecurityProofs.bind_eq_ok.1 h
  simp only [Spot.Select.Rule.over] at h
  split at h
  · rename_i r hr
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    simp only [Option.map_eq_some_iff] at hr
    obtain ⟨s, -, rfl⟩ := hr
    simp
  · cases h

/-- **An accepted two-stage run's session statements pass the model's statement check**, at any round-0 coin and
second stage's coins that carry the bytes the run's samples read (`PlayCoins`): parts 1 to 3 and 5, with no session
read. The statements' message `m₂` states the run's statements (`StmtsAre`), each with its template's rounds, in scope
and its reference statement but for the run's fields. -/
theorem stmtOk_of_accepted {i : Spot.Run.Input} {it : Spot.Run.Interior} {j : Json} {h : Held} {x m₁ m₂ : Msg}
    {c₀ : FlockDeploy.Coin} {c₁ : Stage h} (ha : AcceptedTwo i it j) (hm : HeldMatches i it h) (htwo : TwoHolds h)
    (hnd : NamesDistinct h) (hrec : RecordsHeld h x m₁) (hnone : RegistersNone it.record)
    (hx : jsonMsg x = some i.record) (hm₁ : jsonMsg m₁ = some it.record) (hr2 : RoundTwo h)
    (hstm : StmtsAre h m₂ (it.statements.getD #[])) (hrounds : RoundsHeld h m₂) (hscope : ScopeHeld h m₂)
    (hpin : PinHeld h m₂) (hpc : PlayCoins h i it c₀ c₁) : stmtOk h x m₁ m₂ c₀ c₁ = true := by
  obtain ⟨hcl, workTable, resolved, draw, rest, read, ht, hres, hsel, -, grain, replay, k, ns, ids, ks, draw₂, read₂,
    strata, hrule, hns, hids, hks, hss, -, hstrata, hcodes⟩ := accepted_steps ha
  obtain ⟨n, rule, law, workTable', resolved', -, -, -, ht', hres', hreg⟩ := hm.first
  have hwt : workTable' = workTable := Except.ok.inj (ht'.symm.trans ht)
  have hrs : resolved' = resolved := Except.ok.inj (hres'.symm.trans hres)
  subst hwt hrs
  have hsel' : Spot.Select.select h.reg.first.rule h.reg.first.program h.reg.first.query h.reg.first.table
      h.reg.first.closure i.source = .ok (draw, rest, read) := by
    rw [hreg]
    exact hsel
  have hsample : h.reg.sample (src c₀) = rest :=
    sample_read (by rw [hreg]; exact hcl) hsel' (hpc.1 _ _ _ hsel')
  have hsel0 := hsel
  rw [hcl] at hsel0
  have hproved : (Spot.Draw.proved draw).map Int.toNat = rest := proved_select (resolve_ne_all hrule hres) hsel0
  have hns' : ns = h.reg.ns := Except.ok.inj (hns.symm.trans hm.ns)
  obtain ⟨grain', replay', k', hrule', ids', hids', hks'⟩ := hm.ks
  rw [hrule] at hrule'
  cases hrule'
  have hid : ids = ids' := Except.ok.inj (hids.symm.trans hids')
  subst hid
  rw [← hns'] at hks'
  have hks'' : ks = h.reg.ks := Except.ok.inj (hks.symm.trans hks')
  have hss' : Spot.Select.selectSecond (h.reg.sample (src c₀)) h.reg.ns h.reg.ks it.source = .ok (draw₂, read₂) := by
    rw [hsample, ← hproved, ← hns', ← hks'']
    exact hss
  have hsd : secondDraw h c₀ c₁ = some draw₂ := secondDraw_read h c₀ c₁ hss' (hpc.2 _ _ hss')
  have hpa := hm.parts draw rest read strata hsel' hstrata
  rw [← hproved] at hpa
  have hown : OwnAgrees h (runOwn i it ((Spot.Draw.proved draw).map Int.toNat) strata draw₂)
      ⟨i.record, portRoots i.record⟩ it.record draw₂ c₀ := by
    unfold Round2.OwnAgrees Round2.core runOwn ownOf
    rw [hsample, ← hproved, hpa, hm.links, hm.wires, hm.rows, hm.refs, hm.refReads]
    all_goals rfl
  obtain ⟨ss, hst⟩ := statements_some hcodes
  rw [hst] at hcodes
  rw [hst, Option.getD_some] at hstm
  have hdec : decode x = some ⟨i.record, portRoots i.record⟩ := by
    unfold decode
    rw [hx]
    rfl
  exact hr2 _ ss x m₁ m₂ c₀ c₁ _ it.record draw₂ hdec hm₁ hsd hown hstm hnone htwo hnd hrec hrounds hscope hpin hcodes

/-- **`SpotRunAcceptsPlay`.** -/
theorem spotRunAcceptsPlay : SpotRunAcceptsPlay := by
  intro i it j h H x m₁ m₂ P' r' c₀ c₁ ha hm htwo hnd hrec hnone hx hm₁ hr2 hsa hpc
  obtain ⟨hstm, hrounds, hscope, hpin, hacc⟩ := hsa
  exact accepts_seqCoins h H x [] m₁ m₂ c₀ c₁
    (stmtOk_of_accepted ha hm htwo hnd hrec hnone hx hm₁ hr2 hstm hrounds hscope hpin hpc) P' (fun _ => ()) r' hacc

end SpotFlock.RunsAs
