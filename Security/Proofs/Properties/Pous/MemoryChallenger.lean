import Properties.Pous.MemoryChallenger
import Proofs.Properties.Pous.MemoryChallenger.Device
import Mathlib.Tactic.SplitIfs

/-!
# Proofs of the memory challenger's verdict's properties (`Properties.Pous.MemoryChallenger`)

`verdict` is its checks before the rounds (`pre`, the reasons they give) followed by the rounds and the counts (`tail`),
by case analysis (`verdict_eq`). The rounds' loop only adds: the reasons, the late and the wrong count never shrink,
and a round that adds none passes (`forIn'_measure`). An accepted verdict has no reason, so the checks before the
rounds gave none and every round passed.

`VerdictSigned` follows: an accepted verdict's signature verifies over the session's message and its session ended
`complete`, which is all `Device.signedByDevice_of` reads.
-/

namespace MemoryChallenger.Properties.SecurityProofs

open MemoryChallenger

/-- `ByteArray`'s `==` is its arrays', which is lawful. -/
local instance instLawfulBEqByteArray_properties : LawfulBEq ByteArray where
  eq_of_beq h := ByteArray.ext (beq_iff_eq.1 h)
  rfl {a} := show (a.data == a.data) = true from beq_iff_eq.2 rfl

/-- `verdict`'s reasons before the rounds. -/
private def pre (schemes : Schemes) (trusted : List ByteArray) (s : SignedSession) (c : SessionCommitment) :
    Array String := Id.run do
  let mut reasons : Array String := #[]
  match schemes.lookup s.scheme with
  | none => reasons := reasons.push s!"signature scheme {pyRepr s.scheme} is not accepted"
  | some p =>
    if !trusted.contains s.publicKey then
      reasons := reasons.push "the session is signed by a key that is not a trusted memory challenger's"
    else
      let ok := match s.message >>= fun m => lmsVerify p s.publicKey s.commitment.session m s.signature with
        | .ok b => b
        | .error _ => false
      if !ok then reasons := reasons.push "the signature does not verify"
  if !(s.commitment == c) then reasons := reasons.push "the signed header is not the session's commitment"
  if s.ended != COMPLETE || s.rounds.size != c.k then
    reasons := reasons.push s!"the session ended {s.ended} after {s.rounds.size} of {c.k} rounds"
  return reasons

/-- `verdict` from the rounds on, after the reasons `r0`. -/
private def tail (expected : Nat → Option ByteArray) (s : SignedSession)
    (c : SessionCommitment) (openings : Array CoinOpening) (r0 : Array String) : Verdict := Id.run do
  let mut reasons := r0
  let byJ := fun (j : Nat) => openings.findRev? (·.j == j)
  let mut late := 0
  let mut wrong := 0
  for h : j in [0:s.rounds.size] do
    let r := s.rounds[j]
    let committed := match byJ j with
      | some o => r.j == j && o.index == r.index && verifyCoin c o
      | none => false
    if !committed then reasons := reasons.push s!"round {j}'s index is not its committed coin"
    if !(r.index < c.blocks && expected r.index == some r.digest) then wrong := wrong + 1
    if !(0 ≤ r.elapsedNs && r.elapsedNs ≤ c.limitNs) then late := late + 1
  if wrong > 0 then reasons := reasons.push s!"{wrong} answers are not their blocks of C"
  if late > 0 then reasons := reasons.push s!"{late} answers are past Δ + RTT = {c.limitNs} ns"
  return ⟨reasons.isEmpty, reasons, late, wrong⟩

private theorem verdict_eq (schemes : Schemes) (trusted : List ByteArray) (expected : Nat → Option ByteArray)
    (s : SignedSession) (c : SessionCommitment) (openings : Array CoinOpening) :
    verdict schemes trusted expected s c openings = tail expected s c openings (pre schemes trusted s c) := by
  unfold verdict tail pre
  cases schemes.lookup s.scheme with
  | none =>
    by_cases h1 : (!s.commitment == c) = true <;> by_cases h2 : (s.ended != COMPLETE || s.rounds.size != c.k) = true <;>
      simp only [h1, h2, ite_true, ite_false, Bool.false_eq_true] <;> rfl
  | some p =>
    cases hm : (s.message >>= fun m => lmsVerify p s.publicKey s.commitment.session m s.signature) with
    | error e =>
      by_cases h0 : (!trusted.contains s.publicKey) = true <;>
      by_cases h1 : (!s.commitment == c) = true <;>
      by_cases h2 : (s.ended != COMPLETE || s.rounds.size != c.k) = true <;>
        simp only [hm, h0, h1, h2, ite_true, ite_false, Bool.false_eq_true, Bool.not_false] <;> rfl
    | ok b =>
      by_cases h0 : (!trusted.contains s.publicKey) = true <;> cases b <;>
      by_cases h1 : (!s.commitment == c) = true <;>
      by_cases h2 : (s.ended != COMPLETE || s.rounds.size != c.k) = true <;>
        simp only [hm, h0, h1, h2, ite_true, ite_false, Bool.false_eq_true, Bool.not_false, Bool.not_true] <;> rfl

theorem forIn'_measure {α σ : Type} (l : List α) (f : (a : α) → a ∈ l → σ → Id (ForInStep σ)) (μ : σ → Nat)
    (Good : α → Prop) (hf : ∀ a h b, ∃ b', f a h b = pure (.yield b') ∧ μ b ≤ μ b' ∧ (μ b' = μ b → Good a))
    (init : σ) : μ init ≤ μ (forIn' l init f).run ∧ (μ (forIn' l init f).run = μ init → ∀ a ∈ l, Good a) := by
  induction l generalizing init with
  | nil => simp
  | cons a as ih =>
    obtain ⟨b', hb, h1, h2⟩ := hf a List.mem_cons_self init
    rw [List.forIn'_cons, hb]
    obtain ⟨i1, i2⟩ := ih (fun a' m b => f a' (List.mem_cons_of_mem _ m) b) (fun a' m b => hf a' _ b) b'
    refine ⟨Nat.le_trans h1 i1, fun heq => ?_⟩
    have e1 : μ b' = μ init := by simp only [Id.run, pure_bind] at heq i1 ⊢; omega
    have e2 : μ (forIn' as b' fun a' m b => f a' (List.mem_cons_of_mem _ m) b).run = μ b' := by
      simp only [Id.run, pure_bind] at heq i1 ⊢; omega
    intro x hx
    cases hx with
    | head => exact h2 e1
    | tail _ hx => exact i2 e2 x hx

theorem forIn'_measure_eq {α σ : Type} {l : List α} {f : (a : α) → a ∈ l → σ → Id (ForInStep σ)} {init : σ}
    {L : Id σ} (hL : forIn' l init f = L) (μ : σ → Nat) (Good : α → Prop)
    (hf : ∀ a h b, ∃ b', f a h b = pure (.yield b') ∧ μ b ≤ μ b' ∧ (μ b' = μ b → Good a)) :
    μ init ≤ μ L.run ∧ (μ L.run = μ init → ∀ a ∈ l, Good a) := hL ▸ forIn'_measure l f μ Good hf init

/-- Round `j` passes: its index is its committed coin, its digest the expected one, and it is on time. -/
private def Good (expected : Nat → Option ByteArray) (s : SignedSession) (c : SessionCommitment)
    (openings : Array CoinOpening) (j : Nat) : Prop :=
  ∀ hj : j < s.rounds.size,
    (∃ o, openings.findRev? (·.j == j) = some o ∧ (s.rounds[j].j == j && o.index == s.rounds[j].index &&
      verifyCoin c o) = true) ∧
    (decide (s.rounds[j].index < c.blocks) && expected s.rounds[j].index == some s.rounds[j].digest) = true ∧
    (decide (0 ≤ s.rounds[j].elapsedNs) && decide (s.rounds[j].elapsedNs ≤ ↑c.limitNs)) = true

private theorem tail_accepted (expected : Nat → Option ByteArray) (s : SignedSession) (c : SessionCommitment)
    (openings : Array CoinOpening) (r0 : Array String) (h : (tail expected s c openings r0).accepted = true) :
    r0 = #[] ∧ ∀ j < s.rounds.size, Good expected s c openings j := by
  unfold tail at h
  simp only [Id.run] at h
  rw [Std.Legacy.Range.forIn'_eq_forIn'_range'] at h
  generalize hL : (forIn' (m := Id) (List.range' _ _ _) (_ : Array String × Nat × Nat) _ :
    Id (Array String × Nat × Nat)) = L at h
  obtain ⟨m1, m2⟩ := forIn'_measure_eq hL (fun b => b.1.size + b.2.1 + b.2.2) (Good expected s c openings) (by
    intro a ha b
    clear h hL
    cases ho : openings.findRev? (fun x => x.j == a) with
    | none =>
      split_ifs with h1 h2 h3 <;> refine ⟨_, rfl, ?_, fun he => ?_⟩ <;> (try simp at he) <;> (try simp) <;> (try omega)
      simp at h1
    | some o =>
      split_ifs with h1 h2 h3 <;> refine ⟨_, rfl, ?_, fun he => ?_⟩ <;> (try simp at he) <;> (try simp) <;> (try omega)
      rename_i h2 h3
      intro hj
      exact ⟨⟨o, ho, by simpa using h1⟩, by simpa using h2, by simpa using h3⟩)
  clear hL
  obtain ⟨rs, lt, wr⟩ := L
  simp only [Id.run] at m1 m2
  rw [show ∀ (x : Id (Array String × Nat × Nat)) (f : _ → Id Verdict), (x >>= f) = f x.run from fun _ _ => rfl] at h
  by_cases hw : wr > 0 <;> by_cases hl : lt > 0 <;> simp [hw, hl, Id.run] at h
  have h0 : rs.size = 0 := by simpa using (show rs.isEmpty = true from h)
  have hw0 : wr = 0 := by omega
  have hl0 : lt = 0 := by omega
  subst hw0 hl0
  refine ⟨?_, fun j hj => m2 (by simp; omega) j (by simp; omega)⟩
  simp at m1
  exact Array.eq_empty_of_size_eq_zero (by omega)

private theorem pre_empty (schemes : Schemes) (trusted : List ByteArray) (s : SignedSession) (c : SessionCommitment)
    (h : pre schemes trusted s c = #[]) :
    (∃ p, schemes.lookup s.scheme = some p ∧ s.publicKey ∈ trusted ∧
      ∃ m, s.message = .ok m ∧ lmsVerify p s.publicKey s.commitment.session m s.signature = .ok true) ∧
    (s.commitment == c) = true ∧ s.ended = COMPLETE ∧ s.rounds.size = c.k := by
  unfold pre at h
  cases hp : schemes.lookup s.scheme with
  | none =>
    simp only [hp, Id.run] at h
    split_ifs at h <;> simp [pure] at h
  | some p =>
    cases hm : (s.message >>= fun m => lmsVerify p s.publicKey s.commitment.session m s.signature) with
    | error e =>
      simp only [hp, hm, Id.run] at h
      split_ifs at h <;> simp [pure] at h <;> simp_all
    | ok b =>
      cases b
      · simp only [hp, hm, Id.run] at h
        split_ifs at h <;> simp [pure] at h <;> simp_all
      simp only [hp, hm, Id.run] at h
      split_ifs at h <;> simp [pure] at h
      refine ⟨⟨p, rfl, by simp_all, ?_⟩, by simp_all, by simp_all⟩
      cases hs : s.message with
      | error e => simp [hs, bind, Except.bind] at hm
      | ok m => exact ⟨m, rfl, by simpa [hs, bind, Except.bind] using hm⟩

private theorem commitment_eq {a b : SessionCommitment} (h : (a == b) = true) : a = b := by
  cases a; cases b
  simp only [BEq.beq] at h
  simp only [instBEqSessionCommitment.beq, Bool.and_eq_true, beq_iff_eq] at h
  obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h
  subst h1 h2 h3 h4 h5 h6 h7 h8
  rfl

private theorem findRev_some {openings : Array CoinOpening} {p : CoinOpening → Bool} {o : CoinOpening}
    (h : openings.findRev? p = some o) : o ∈ openings ∧ p o = true := by
  rw [Array.findRev?_eq_find?_reverse] at h
  exact ⟨Array.mem_reverse.1 (Array.mem_of_find?_eq_some h), Array.find?_some h⟩

theorem verdictAccepts : VerdictAccepts := by
  intro schemes trusted expected s c openings h
  rw [verdict_eq] at h
  obtain ⟨h0, hg⟩ := tail_accepted expected s c openings _ h
  obtain ⟨hk, hc, he, hn⟩ := pre_empty schemes trusted s c h0
  refine ⟨hk, commitment_eq hc, he, hn, fun j hj => ?_⟩
  obtain ⟨⟨o, ho, hco⟩, hd, ht⟩ := hg j hj hj
  obtain ⟨hmem, hoj⟩ := findRev_some ho
  simp only [Bool.and_eq_true, beq_iff_eq] at hco hoj
  simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hd ht
  exact ⟨hco.1.1, ⟨o, hmem, hoj, hco.1.2, hco.2⟩, hd.1, hd.2, ht.1, ht.2⟩

theorem judgeAccepts : JudgeAccepts := by
  intro r hj
  unfold Judge.judge at hj
  cases hs : Judge.schemesOf r.schemes with
  | error e => simp only [hs] at hj; cases hj
  | ok schemes =>
    simp only [hs] at hj
    refine ⟨schemes, rfl, verdictAccepts _ _ _ _ _ _ ?_⟩
    simpa [Functor.map, Except.map, pure, Except.pure, bind, Except.bind] using hj

theorem verdictSigned : VerdictSigned := by
  intro schemes trusted expected s c openings dev h
  obtain ⟨⟨p, hp, ht, m, hm, hv⟩, -, he, -⟩ := verdictAccepts schemes trusted expected s c openings h
  exact Device.signedByDevice_of dev hp ht hm hv he

theorem judgeSigned : JudgeSigned := by
  intro r dev hj
  obtain ⟨schemes, hs, ⟨p, hp, ht, m, hm, hv⟩, -, he, -⟩ := judgeAccepts r hj
  exact ⟨schemes, hs, Device.signedByDevice_of dev hp ht hm hv he⟩

end MemoryChallenger.Properties.SecurityProofs
