import Proofs.Flock.Recursive.HoloSchedRead

/-!
# What the verifier's checks of the held `c_k` and the claim ports give

`fixedRow_valueRound` reads three hypotheses on the streams the verifier holds. Two of them are what its checks give:

* `repsCommitsAt_of_check`: `checkCommitRounds` (the end of `checkCovers`) holds two reps, each's `c_k` at the reps'
  schedule's message rounds (`holoRepRounds` under `Inner.holo`, which is `holoRounds`): `RepsCommitsAt`.
* `sparseCommitsAt_of_check`: it holds `sparse/rep0`'s and `sparse/rep1`'s `c_k` at `sparsePhaseRounds` of the verifier's
  RecSparse shape (`sparseRounds`) and refuses any other held stream named `sparse/…`, so a held `sparse/rep<y>` has
  `y < 2` (`sparse_lt_two`): `SparseCommitsAt`.
* `claimReadsAt_of_checkOwn`: `checkOwn` holds the reads file's `rec-claim` ports to the statement's (`ownPorts`):
  `ClaimReadsAt`. No part's or opening's message-round value is `rec-claim` (`recC_ne_claim`, `recOpeningC_ne_claim`),
  and an opening's `claim<k>` is named as `claimIndex` reads it back (`claimIndex_some`).
* `stackedAt_of_check`: it holds `stacked/rep0`'s `c_k` at the message rounds of `stackedRounds` of the verifier's R
  shape and its tops' node counts at the ones those rounds commit (`topCounts`, `topCounts_eq`): `StackedAt`.
-/

namespace Flock.Guarantees.HoloSchedCheck

open Flock Flock.RecVStar

theorem checkBind_ok {ε α β : Type} {x : Except ε α} {f : α → Except ε β} {b : β} (h : (x >>= f) = .ok b) :
    ∃ a, x = .ok a ∧ f a = .ok b := by
  cases x with
  | error e => cases h
  | ok a => exact ⟨a, rfl, h⟩

theorem checkThrow_ne {α : Type} (e : String) (a : α) : (throw e : Except String α) ≠ .ok a := nofun

theorem forIn_each {α ε : Type} (P : α → Prop) {f : α → Unit → Except ε (ForInStep Unit)}
    (hf : ∀ a r, f a () = .ok r → P a ∧ r = .yield ()) :
    ∀ (l : List α) {u : Unit}, forIn l () f = .ok u → ∀ a ∈ l, P a
  | [], _, _, a, ha => by cases ha
  | b :: l, u, h, a, ha => by
    rw [List.forIn_cons] at h
    obtain ⟨st, hst, h⟩ := checkBind_ok h
    obtain ⟨hb, rfl⟩ := hf b st hst
    cases ha with
    | head => exact hb
    | tail _ ha => exact forIn_each P hf l h a ha

theorem msgRounds_eq (rs : Array SchedRound) : msgRounds rs = commitRounds rs := rfl

theorem sparseRounds_eq (sh : RecSparse.Shape) : sparseRounds sh = sparsePhaseRounds sh := rfl

theorem holoRounds_eq (m kLog claims : Nat) : holoRounds m kLog claims = holoRepRounds m kLog claims := rfl

theorem topCounts_eq (rs : Array SchedRound) : topCounts rs = (topRounds rs).map (·.2) := by
  unfold topCounts topRounds
  rw [List.map_flatMap]
  apply congrArg (fun g => (List.range rs.size).flatMap g)
  funext k
  rw [List.map_map]
  exact (List.map_id _).symm

/-- **`checkCommitRounds` accepting**: two reps, each's `c_k` rounds the reps' schedule's message rounds,
`sparse/rep0`'s and `sparse/rep1`'s the sparse phase's, and `stacked/rep0`'s those of `stackedRounds` of the
verifier's R shape, its tops of the node counts those rounds commit, and no other held stream named `sparse/…`. Its
refusal of a stream off the reps' rounds comes first, and is not read here. -/
theorem commitRounds_of_check {i : Inner} {cs : InnerCommits} {rep : Array SchedRound}
    (hsch : (if i.holo then holoRepRounds else repRounds) i.m i.kLog i.claims = some rep)
    (h : checkCommitRounds i cs = .ok ()) :
    cs.reps.size = 2 ∧
      (∀ (y : Nat) (cm : Array (Nat × ByteArray)), cs.reps[y]? = some cm → cm.toList.map (·.1) = commitRounds rep) ∧
      (∀ sh, i.sparse = some sh → ∀ (y : Nat) (st : InnerStream), y < 2 → cs.stream? s!"{SPARSE}/rep{y}" = some st →
        st.commits.toList.map (·.1) = commitRounds (sparsePhaseRounds sh)) ∧
      (∀ st : InnerStream, cs.stream? s!"{STACKED}/rep0" = some st → ∃ rs, i.stacked.bind stackedRounds = some rs ∧
        st.commits.toList.map (·.1) = commitRounds rs ∧ st.tops.toList.map (·.size) = topCounts rs) ∧
      cs.streams.find? (fun st => st.name.startsWith s!"{SPARSE}/" &&
        st.name != s!"{SPARSE}/rep0" && st.name != s!"{SPARSE}/rep1") = none := by
  unfold checkCommitRounds at h
  rw [hsch] at h
  try simp only at h
  obtain ⟨_, -, h⟩ := checkBind_ok h
  split at h
  · first
    | exact absurd h (checkThrow_ne _ _)
    | exact absurd (checkBind_ok h) fun ⟨_, h1, _⟩ => checkThrow_ne _ _ h1
    | cases h
  · rename_i hsz
    rcases hf : cs.streams.find? (fun st => st.name.startsWith s!"{SPARSE}/" &&
        st.name != s!"{SPARSE}/rep0" && st.name != s!"{SPARSE}/rep1") with _ | stx
    rotate_left
    · first
      | rw [hf] at h
      | simp only [hf] at h
      try simp only at h
      first
      | exact absurd h (checkThrow_ne _ _)
      | exact absurd (checkBind_ok h) fun ⟨_, h1, _⟩ => checkThrow_ne _ _ h1
      | cases h
    first
    | rw [hf] at h
    | simp only [hf] at h
    try simp only at h
    rcases hs : cs.stream? s!"{STACKED}/rep0" with _ | st₀
    · first
      | rw [hs] at h
      | simp only [hs] at h
      | skip
      try simp only at h
      obtain ⟨rr, -, h⟩ := checkBind_ok h
      obtain ⟨_, hloop, -⟩ := checkBind_ok h
      have hall := forIn_each (fun a : String × Array (Nat × ByteArray) × Array SchedRound =>
          a.2.1.toList.map (·.1) = commitRounds a.2.2) (fun a r hb => by
            obtain ⟨s, cm, rs⟩ := a
            try simp only at hb
            split at hb
            · first
              | exact absurd hb (checkThrow_ne _ _)
              | exact absurd (checkBind_ok hb) fun ⟨_, h1, _⟩ => checkThrow_ne _ _ h1
              | cases hb
            · rename_i hne
              exact ⟨by simpa using hne, (Except.ok.inj hb).symm⟩) _ hloop
      refine ⟨by simpa using hsz, fun y cm hcm => ?_, fun sh hsh y st hy hst => ?_, fun st hst => ?_, by first | exact hf | rfl⟩
      · have hz : (cm, y) ∈ cs.reps.toList.zipIdx := by
          apply List.mem_zipIdx_iff_getElem?.2
          first
          | (rw [Array.getElem?_toList]; exact hcm)
          | simpa using hcm
        exact hall _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_map.2 ⟨(cm, y), hz, rfl⟩)))
      · have hy01 : y ∈ [0, 1] := by
          first
          | (simp; omega)
          | (rcases (by omega : y = 0 ∨ y = 1) with rfl | rfl <;> simp)
        have hm : (st.name, st.commits, sparsePhaseRounds sh) ∈ [0, 1].filterMap fun x =>
            (cs.stream? s!"{SPARSE}/rep{x}").map fun st => (st.name, st.commits, sparsePhaseRounds sh) := by
          refine List.mem_filterMap.2 ⟨y, hy01, ?_⟩
          show Option.map _ (cs.stream? s!"{SPARSE}/rep{y}") = _
          rw [hst]
          try rfl
        first
        | rw [hsh] at hall
        | simp only [hsh] at hall
        exact hall _ (List.mem_append_left _ (List.mem_append_right _ hm))
      · first
        | cases hst
        | (rw [hs] at hst; cases hst)
        | simp [hs] at hst
    · rcases hb : i.stacked.bind stackedRounds with _ | rs
      · first
        | rw [hs, hb] at h
        | simp only [hs, hb] at h
        | skip
        try simp only at h
        first
        | exact absurd h (checkThrow_ne _ _)
        | exact absurd (checkBind_ok h) fun ⟨_, h1, _⟩ => checkThrow_ne _ _ h1
        | cases h
      · first
        | rw [hs, hb] at h
        | simp only [hs, hb] at h
        | skip
        try simp only at h
        split at h
        · first
          | exact absurd h (checkThrow_ne _ _)
          | exact absurd (checkBind_ok h) fun ⟨_, h1, _⟩ => checkThrow_ne _ _ h1
          | cases h
        · rename_i hne
          obtain ⟨rr, hr, h⟩ := checkBind_ok h
          obtain ⟨_, hloop, -⟩ := checkBind_ok h
          have hrr : [(st₀.name, st₀.commits, rs)] = rr := Except.ok.inj hr
          have hmr : (st₀.name, st₀.commits, rs) ∈ rr := by
            rw [← hrr]
            exact List.mem_singleton_self _
          have hall := forIn_each (fun a : String × Array (Nat × ByteArray) × Array SchedRound =>
              a.2.1.toList.map (·.1) = commitRounds a.2.2) (fun a r hb => by
                obtain ⟨s, cm, rs⟩ := a
                try simp only at hb
                split at hb
                · first
                  | exact absurd hb (checkThrow_ne _ _)
                  | exact absurd (checkBind_ok hb) fun ⟨_, h1, _⟩ => checkThrow_ne _ _ h1
                  | cases hb
                · rename_i hne
                  exact ⟨by simpa using hne, (Except.ok.inj hb).symm⟩) _ hloop
          refine ⟨by simpa using hsz, fun y cm hcm => ?_, fun sh hsh y st hy hst => ?_, fun st hst => ?_, by first | exact hf | rfl⟩
          · have hz : (cm, y) ∈ cs.reps.toList.zipIdx := by
              apply List.mem_zipIdx_iff_getElem?.2
              first
              | (rw [Array.getElem?_toList]; exact hcm)
              | simpa using hcm
            exact hall _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_map.2 ⟨(cm, y), hz, rfl⟩)))
          · have hy01 : y ∈ [0, 1] := by
              first
              | (simp; omega)
              | (rcases (by omega : y = 0 ∨ y = 1) with rfl | rfl <;> simp)
            have hm : (st.name, st.commits, sparsePhaseRounds sh) ∈ [0, 1].filterMap fun x =>
                (cs.stream? s!"{SPARSE}/rep{x}").map fun st => (st.name, st.commits, sparsePhaseRounds sh) := by
              refine List.mem_filterMap.2 ⟨y, hy01, ?_⟩
              show Option.map _ (cs.stream? s!"{SPARSE}/rep{y}") = _
              rw [hst]
              try rfl
            first
            | rw [hsh] at hall
            | simp only [hsh] at hall
            exact hall _ (List.mem_append_left _ (List.mem_append_right _ hm))
          · first
            | cases hst
            | (rw [hs] at hst; cases hst)
            refine ⟨rs, by first | rfl | exact hb, ?_, by simpa using hne⟩
            exact hall _ (List.mem_append_right _ hmr)

/-- **The verifier's check of the reps' `c_k` gives `RepsCommitsAt`** at the hidden-unit schedule `holoRounds`. -/
theorem repsCommitsAt_of_check {i : Inner} {cs : InnerCommits} {rep : Array SchedRound}
    (h : checkCommitRounds i cs = .ok ()) (hholo : i.holo = true)
    (hrep : holoRounds i.m i.kLog i.claims = some rep) : RepsCommitsAt cs rep := by
  have hsch : (if i.holo then holoRepRounds else repRounds) i.m i.kLog i.claims = some rep := by
    first
    | (rw [hholo, if_pos rfl]; exact hrep)
    | (rw [hholo]; exact hrep)
    | (simp only [hholo, ↓reduceIte, ← holoRounds_eq]; exact hrep)
  obtain ⟨hsz, hreps, -⟩ := commitRounds_of_check hsch h
  intro y cm hcm
  refine ⟨?_, ?_⟩
  · rcases Nat.lt_or_ge y cs.reps.size with hy | hy
    · rw [hsz] at hy
      exact hy
    · rw [Array.getElem?_eq_none hy] at hcm
      cases hcm
  · unfold CommitsAt
    rw [msgRounds_eq]
    exact hreps y cm hcm

theorem digits_eq {y : Nat} {d : Char} (h : Nat.toDigits 10 y = [d]) : y = Nat.ofDigitChars 10 [d] 0 := by
  rw [← h, Nat.ofDigitChars_ten_toDigits]

/-- **A held `sparse/rep<y>` is a rep's** (`y < 2`) where no held stream named `sparse/…` is other than `sparse/rep0`
and `sparse/rep1`, as `checkCommitRounds` refuses. -/
theorem sparse_lt_two {cs : InnerCommits}
    (hn : cs.streams.find? (fun st => st.name.startsWith s!"{SPARSE}/" &&
      st.name != s!"{SPARSE}/rep0" && st.name != s!"{SPARSE}/rep1") = none) :
    ∀ (y : Nat) (st : InnerStream), cs.stream? s!"{SPARSE}/rep{y}" = some st → y < 2 := by
  intro y st h
  have hm := Array.mem_of_find?_eq_some h
  have hname : st.name = s!"{SPARSE}/rep{y}" := by simpa using Array.find?_some h
  have hf := Array.find?_eq_none.1 hn st hm
  rw [hname] at hf
  refine Classical.byContradiction fun hy => hf ?_
  have ne : ∀ (k : String) (d : Char), k.toList = "/rep".toList ++ [d] → Nat.ofDigitChars 10 [d] 0 < 2 →
      s!"{SPARSE}/rep{y}" ≠ toString SPARSE ++ toString k := by
    intro k d hk hd he
    have he := congrArg String.toList he
    simp only [String.toList_append, toString, Nat.toList_repr] at he
    rw [hk] at he
    have := digits_eq (by simpa [List.append_assoc] using he)
    omega
  simp only [Bool.and_eq_true, bne_iff_ne, ne_eq]
  refine ⟨⟨String.startsWith_string_iff.2 ?_, ne "/rep0" '0' (by decide) (by decide)⟩,
    ne "/rep1" '1' (by decide) (by decide)⟩
  simp only [String.toList_append, toString]
  exact ⟨"rep".toList ++ Nat.toDigits 10 y, by simp [List.append_assoc]⟩

/-- **The verifier's check of the sparse phases' `c_k` gives `SparseCommitsAt`**: `checkCommitRounds` holds
`sparse/rep0`'s and `sparse/rep1`'s at the sparse phase's rounds and refuses any other held `sparse/…` stream. -/
theorem sparseCommitsAt_of_check {i : Inner} {cs : InnerCommits} {rep : Array SchedRound}
    (hsch : (if i.holo then holoRepRounds else repRounds) i.m i.kLog i.claims = some rep)
    (h : checkCommitRounds i cs = .ok ()) : SparseCommitsAt cs i.sparse := by
  obtain ⟨-, -, hsp, -, hn⟩ := commitRounds_of_check hsch h
  intro y st sh hst hsh
  have hy := sparse_lt_two hn y st hst
  refine ⟨hy, ?_⟩
  unfold CommitsAt
  rw [msgRounds_eq, sparseRounds_eq]
  exact hsp sh hsh y st hy hst

/-- **The verifier's check of R's stream gives `StackedAt`** at `stackedRounds` of the verifier's R shape. -/
theorem stackedAt_of_check {i : Inner} {cs : InnerCommits} {rep : Array SchedRound}
    (hsch : (if i.holo then holoRepRounds else repRounds) i.m i.kLog i.claims = some rep)
    (h : checkCommitRounds i cs = .ok ()) : StackedAt cs (i.stacked.bind stackedRounds) := by
  obtain ⟨-, -, -, hR, -⟩ := commitRounds_of_check hsch h
  intro st hst
  obtain ⟨rs, hrs, hc, ht⟩ := hR st hst
  refine ⟨rs, hrs, ?_, ?_⟩
  · unfold CommitsAt
    rw [msgRounds_eq]
    exact hc
  · unfold TopsAt
    rw [← topCounts_eq]
    exact ht

theorem recC_ne_claim (k : Nat) : s!"rec-c{k}" ≠ CLAIM_VALUE := by
  intro h
  have h := congrArg String.toList h
  have e : CLAIM_VALUE.toList = "rec-c".toList ++ 'l' :: "aim".toList := by decide
  rw [e] at h
  simp only [String.toList_append, toString, Nat.toList_repr] at h
  have h := List.append_cancel_left h
  have hl : 'l' ∈ Nat.toDigits 10 k := by
    rw [h]
    exact List.Mem.head _
  exact absurd (Nat.isDigit_of_mem_toDigits (by decide) (by decide) hl) (by decide)

theorem recOpeningC_ne_claim (k : Nat) : s!"rec-opening-c{k}" ≠ CLAIM_VALUE := by
  intro h
  have h := congrArg (fun s : String => s.toList.take 5) h
  simp only [String.toList_append, toString] at h
  rw [List.take_append_of_le_length (by decide)] at h
  exact absurd h (by decide)

theorem claimIndex_some {nm : String} {k : Nat} (h : claimIndex nm = some k) : nm = s!"claim{k}" := by
  unfold claimIndex at h
  split at h
  · exact beq_iff_eq.1 (Option.filter_eq_some_iff.1 h).2
  · cases h

theorem mem_msgPorts_claim {own : Array Registered.Port} {reg : Registered.Port} (hreg : reg ∈ own)
    (hv : reg.value = CLAIM_VALUE) : (reg.name, reg.value, reg.reads) ∈ msgPorts own := by
  unfold msgPorts
  exact List.mem_map.2 ⟨reg, List.mem_filter.2 ⟨Array.mem_toList_iff.2 hreg, by simp [hv]⟩, rfl⟩

/-- **`checkOwn` accepting**: each message-round port of the reads file is one of the statement's. -/
theorem mem_ownPorts_of_checkOwn {s : VStar.Stmt} {own : Array Registered.Port} (h : checkOwn s own = .ok ()) :
    ∀ q ∈ msgPorts own, q ∈ ownPorts s := by
  unfold checkOwn at h
  try simp only at h
  split at h
  · first
    | exact absurd h (checkThrow_ne _ _)
    | exact absurd (checkBind_ok h) fun ⟨_, h1, _⟩ => checkThrow_ne _ _ h1
    | cases h
  · rename_i hnone
    intro q hq
    have hfind : (msgPorts own).find? (fun x => !(ownPorts s).contains x) = none := by
      cases hf : (msgPorts own).find? (fun x => !(ownPorts s).contains x) with
      | none => rfl
      | some q' =>
        first
        | exact absurd hf (hnone q')
        | exact (hnone q' hf).elim
        | simp_all
    have hc := List.find?_eq_none.1 hfind q hq
    simpa using hc

/-- **The verifier's check of the reads file's message-round ports gives `ClaimReadsAt`.** -/
theorem claimReadsAt_of_checkOwn {s : VStar.Stmt} {own : Array Registered.Port} (h : checkOwn s own = .ok ()) :
    ClaimReadsAt s own := by
  intro reg hreg hv
  have hq : (reg.name, reg.value, reg.reads) ∈ ownPorts s :=
    mem_ownPorts_of_checkOwn h _ (mem_msgPorts_claim hreg hv)
  cases s with
  | recOpen =>
    first
    | cases hq
    | simp [ownPorts] at hq
  | recOpenStacked =>
    first
    | cases hq
    | simp [ownPorts] at hq
  | inner j name p =>
    simp only [ownPorts] at hq
    obtain ⟨⟨nm, n⟩, -, hf⟩ := List.mem_filterMap.1 hq
    try simp only at hf
    cases hm : mIndex nm with
    | none =>
      rw [hm] at hf
      cases hf
    | some k =>
      rw [hm] at hf
      exact absurd ((congrArg (fun q : String × String × Array (Option Nat) => q.2.1)
        (Option.some.inj hf)).trans hv) (recC_ne_claim k)
  | sparse sh hr =>
    have hq : (reg.name, reg.value, reg.reads) ∈
        [("h", s!"rec-c{hr}", #[some 0, some 1]), ("claim", CLAIM_VALUE, #[some 0, some 1])] := hq
    rcases List.mem_cons.1 hq with he | hq
    · exact absurd ((congrArg (fun q : String × String × Array (Option Nat) => q.2.1) he).symm.trans hv)
        (recC_ne_claim hr)
    · rcases List.mem_cons.1 hq with he | hq
      · show reg.name = "claim" ∧ reg.reads = #[some 0, some 1]
        exact ⟨congrArg Prod.fst he, congrArg (fun q : String × String × Array (Option Nat) => q.2.2) he⟩
      · cases hq
  | opening m c j S ports nv =>
    simp only [ownPorts] at hq
    obtain ⟨⟨nm, n⟩, -, hf⟩ := List.mem_filterMap.1 hq
    try simp only at hf
    cases hm : mIndex nm with
    | some k =>
      rw [hm] at hf
      exact absurd ((congrArg (fun q : String × String × Array (Option Nat) => q.2.1)
        (Option.some.inj hf)).trans hv) (recOpeningC_ne_claim k)
    | none =>
      cases hc : claimIndex nm with
      | none =>
        rw [hm, hc] at hf
        cases hf
      | some k =>
        rw [hm, hc] at hf
        have he := Option.some.inj hf
        show ∃ k : Nat, reg.name = s!"claim{k}" ∧ reg.reads = #[some k]
        exact ⟨k, (congrArg Prod.fst he).symm.trans (claimIndex_some hc),
          (congrArg (fun q : String × String × Array (Option Nat) => q.2.2) he).symm⟩

end Flock.Guarantees.HoloSchedCheck
