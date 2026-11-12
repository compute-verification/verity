module

public import Proofs.Flock.SpotFlock.Code
public import Proofs.Flock.Claims.Sampling.Deployed
public import Proofs.Flock.Claims.Sampling.SessionSeq
public import Proofs.Flock.Claims.DeployOC
public import Proofs.Flock.Bridge.Members

/-!
# SPOT[Flock]'s sessions are Flock runs

`SpotFlock.seq h rel`: SPOT[Flock]'s two-stage verifier (`SpotFlock.code h`) runs its sessions in sequence
(`Claims.Sampling.SessionSeq`). Rounds 0 to `stage + 1` answer with no `H` call: the two stages' coins, then the session
statements' check (`stmtOk`), which answers `[]` or refuses. Template `t`'s session follows on its `k₀ t` rounds from
`off t`, as Flock's masked verifier at the statement round `stage + 1` states for it, whose round count `stmtOk` holds
to `k₀ t` (`stmtOk_at`), its acceptance answered `[]`, a message Flock never answers (a coin's bytes are never empty,
`coin_bytes_ne`). A session round after a decided prefix answers `[]` too, and the last round accepts only where every
session's messages have a prefix Flock decides (`decidedAt`), at which Flock answers no message (`decidedAt_ne_msg`). A
checked unit is a proof block of the members' partition, so its template is one of theirs (`templateOf_lt`), and the
session read off a transcript is the template's statement of round `stage + 1` where it is in scope (`sessionOk`), which
`stmtOk` checks. A run refused before the session shows `xS₀ t`'s refused run, at which Flock refuses the first message
`[]` (`refuses_empty`); Flock's verifier has a round at every statement (`k₀_pos`).

So `sessionRunsTo`: `SessionRunsTo` holds at that code, for every relation and every `H`.
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy Finset

variable (h : Held)

/-- **Flock's masked verifier has a round at every statement.** -/
theorem k₀_pos (t : ℕ) : 0 < k₀ h t := by
  unfold k₀ FlockDeploy.V
  dsimp only
  split <;> omega

/-- **Flock refuses the first message `[]`**, with no `H` call, at every statement. -/
theorem refuses_empty (x : Msg) (c : FlockDeploy.Coin) : FlockDeploy.next true x [[]] [c] = .ret .rej := by
  have ha : FlockDeploy.act true x [[]] [c] = .rej := by
    unfold FlockDeploy.act
    split <;> rfl
  simp only [FlockDeploy.next, ha]

/-- **A coin's bytes are never empty.** -/
theorem coin_bytes_ne (c : FlockDeploy.Coin) : c.bytes ≠ [] := by
  intro hc
  have h0 : (c 0).toBytes.data.toList ≠ [] := by
    intro h'
    have := congrArg List.length h'
    simp [_root_.Flock.F128.toBytes] at this
  exact h0 (List.flatMap_eq_nil_iff.mp hc 0 (List.mem_finRange 0))

/-- **A decision is no message.** -/
theorem outOf_ne_msg (d : OCV Unit) (H : Msg → Msg) (m : Msg) : ((FlockDeploy.outOf d).eval H).1 ≠ .msg m := by
  unfold FlockDeploy.outOf
  rw [OC.eval_bind]
  dsimp only
  split <;> simp [OC.eval]

/-- **Flock answers no message where it decides**: at a prefix whose step is not a coin. -/
theorem decidedAt_ne_msg {xS : Msg} {as : List Msg} {cs : List FlockDeploy.Coin} {j : ℕ}
    (hd : decidedAt xS as cs j = true) (H : Msg → Msg) (m : Msg) :
    ((FlockDeploy.next true xS (as.take j) (cs.take j)).eval H).1 ≠ .msg m := by
  unfold decidedAt at hd
  unfold FlockDeploy.next
  split
  · nofun
  · rename_i c hc
    simp [hc] at hd
  · exact outOf_ne_msg _ H m

/-- **Where Flock answers a message, it has not decided.** -/
theorem not_decidedAt {xS : Msg} {as : List Msg} {cs : List FlockDeploy.Coin} {j : ℕ} {H : Msg → Msg} {m : Msg}
    (hm : ((FlockDeploy.next true xS (as.take j) (cs.take j)).eval H).1 = .msg m) : decidedAt xS as cs j = false := by
  cases hd : decidedAt xS as cs j
  · rfl
  · exact absurd hm (decidedAt_ne_msg hd H m)

/-- **Flock never answers `[]`.** -/
theorem flock_ne_nil (H : Msg → Msg) (xS : Msg) (as : List Msg) (cs : List FlockDeploy.Coin) :
    ((FlockDeploy.next true xS as cs).eval H).1 ≠ .msg [] := by
  unfold FlockDeploy.next
  split
  · nofun
  · rename_i c _
    intro hc
    simp only [OC.eval, Out.msg.injEq] at hc
    exact coin_bytes_ne c hc
  · exact outOf_ne_msg _ H []

theorem markOut_ne_acc (mark : Msg) (o : Out) : markOut mark o ≠ .acc := by
  cases o <;> nofun

/-- **A session round never accepts.** -/
theorem sessStep_ne_acc (H : Msg → Msg) (t : ℕ) (xS : Msg) (as : List Msg) (cs : List FlockDeploy.Coin) :
    ((sessStep h t xS as cs).eval H).1 ≠ .acc := by
  unfold sessStep
  split
  · nofun
  · rw [OC.eval_bind]
    exact markOut_ne_acc [] _

/-- **A session round where no shorter prefix is decided** is Flock's, its acceptance shown as `[]`. -/
theorem sessStep_eval (H : Msg → Msg) (t : ℕ) (xS : Msg) (as : List Msg) (cs : List FlockDeploy.Coin)
    (hund : ∀ j, 0 < j → j < as.length - off h t → decidedAt xS (as.drop (off h t)) (cs.drop (off h t)) j = false) :
    (sessStep h t xS as cs).eval H =
      (markOut [] ((FlockDeploy.next true xS (as.drop (off h t)) (cs.drop (off h t))).eval H).1,
        ((FlockDeploy.next true xS (as.drop (off h t)) (cs.drop (off h t))).eval H).2) := by
  unfold sessStep
  rw [ite_eq_right]
  · rw [OC.eval_bind]
    exact Prod.ext rfl (List.append_nil _)
  · simp only [List.any_eq_true, List.mem_range, Bool.and_eq_true, decide_eq_true_eq, not_exists, not_and]
    intro j hj hj0 hd
    rw [hund j hj0 hj] at hd
    exact Bool.false_ne_true hd

/-! ## The session slots -/

theorem off_succ (t : ℕ) : off h (t + 1) = off h t + k₀ h t := rfl

/-- **A round in template `t`'s session's rounds is in its slot.** -/
theorem slotOf_eq {n t : ℕ} (ht : t < h.ms.length) (h1 : off h t < n) (h2 : n ≤ off h t + k₀ h t) :
    slotOf h n = some t := by
  unfold slotOf
  rw [List.find?_range_eq_some]
  refine ⟨by simp [h1, h2], List.mem_range.mpr ht, fun j hj => ?_⟩
  simp only [Bool.not_eq_true', decide_eq_false_iff_not, not_and, not_le]
  intro _
  exact lt_of_le_of_lt (sessOff_add_le (h.stage + 2) (k₀ h) hj) h1

/-- **The round after the sessions is in none.** -/
theorem slotOf_last : slotOf h (off h h.ms.length + 1) = none := by
  unfold slotOf
  rw [List.find?_eq_none]
  intro t ht
  simp only [List.mem_range] at ht
  have := sessOff_add_le (h.stage + 2) (k₀ h) ht
  simp only [decide_eq_true_eq, not_and, not_le]
  intro _
  show off h t + k₀ h t < off h h.ms.length + 1
  exact Nat.lt_succ_of_le this

/-! ## The statement check -/

/-- **The statement check, template by template**: template `t`'s statement has `k₀ t` rounds and is in scope. -/
theorem stmtOk_at {x m₁ m₂ : Msg} {c₀ : FlockDeploy.Coin} {c₁ : Stage h} (hok : stmtOk h x m₁ m₂ c₀ c₁ = true)
    {t : ℕ} (ht : t < h.ms.length) :
    (FlockDeploy.V true).k (stmtAt h m₂ t) = k₀ h t ∧ sessionOk h t (stmtAt h m₂ t) = true := by
  obtain ⟨s, hp⟩ : ∃ s, parse h x = some s := by
    cases hp : parse h x with
    | none => simp [stmtOk, hp] at hok
    | some s => exact ⟨s, rfl⟩
  obtain ⟨xs, hx⟩ : ∃ xs, stmtsOf m₂ = some xs := by
    cases hx : stmtsOf m₂ with
    | none => simp [stmtOk, hp, hx] at hok
    | some xs => exact ⟨xs, rfl⟩
  simp only [stmtOk, hp, hx, Bool.and_eq_true, beq_iff_eq, List.all_eq_true, List.mem_range] at hok
  have hall := hok.2 t ht
  have e : stmtAt h m₂ t = xs.getD t (h.xS₀ t) := by unfold stmtAt; rw [hx]
  rw [e]
  exact ⟨hall.1.1, hall.1.2⟩

/-- **A session statement the run checks is in the scope Flock's soundness proof covers, with hidden outputs**
(`sessionOk`'s scope check and hidden-outputs tag). -/
theorem scoped_at {x m₁ m₂ : Msg} {c₀ : FlockDeploy.Coin} {c₁ : Stage h} (hok : stmtOk h x m₁ m₂ c₀ c₁ = true)
    {t : ℕ} (ht : t < h.ms.length) :
    FlockDeploy.scopeCheck true (stmtAt h m₂ t) = true ∧ hiddenOf (stmtAt h m₂ t) = true := by
  have hs := (stmtOk_at h hok ht).2
  simp only [sessionOk, Bool.and_eq_true] at hs
  exact ⟨hs.1.2, hs.2⟩

/-- **The statement check holds template by template** (`stmtOkAt`), at the statement `m₂` states for each
template. -/
theorem stmtOkAt_of {x m₁ m₂ : Msg} {c₀ : FlockDeploy.Coin} {c₁ : Stage h} (hok : stmtOk h x m₁ m₂ c₀ c₁ = true)
    {t : ℕ} (ht : t < h.ms.length) : stmtOkAt h x m₁ (stmtAt h m₂ t) c₀ c₁ t = true := by
  obtain ⟨s, hp⟩ : ∃ s, parse h x = some s := by
    cases hp : parse h x with
    | none => simp [stmtOk, hp] at hok
    | some s => exact ⟨s, rfl⟩
  obtain ⟨xs, hx⟩ : ∃ xs, stmtsOf m₂ = some xs := by
    cases hx : stmtsOf m₂ with
    | none => simp [stmtOk, hp, hx] at hok
    | some xs => exact ⟨xs, rfl⟩
  simp only [stmtOk, hp, hx, Bool.and_eq_true, beq_iff_eq, List.all_eq_true, List.mem_range] at hok
  have hall := hok.2 t ht
  have e : stmtAt h m₂ t = xs.getD t (h.xS₀ t) := by unfold stmtAt; rw [hx]
  rw [e]
  simp only [stmtOkAt, hp, Bool.and_eq_true, beq_iff_eq]
  exact ⟨⟨⟨⟨hok.1.1.1.1, hok.1.1.1.2⟩, hall.1.1⟩, hall.1.2⟩, hall.2⟩

/-- **The statement check holds the run's records to the registrations' names** (`recordsOk`). -/
theorem recordsOk_of_stmtOk {x m₁ m₂ : Msg} {c₀ : FlockDeploy.Coin} {c₁ : Stage h}
    (hok : stmtOk h x m₁ m₂ c₀ c₁ = true) : recordsOk h x m₁ = true := by
  obtain ⟨s, hp⟩ : ∃ s, parse h x = some s := by
    cases hp : parse h x with
    | none => simp [stmtOk, hp] at hok
    | some s => exact ⟨s, rfl⟩
  obtain ⟨xs, hx⟩ : ∃ xs, stmtsOf m₂ = some xs := by
    cases hx : stmtsOf m₂ with
    | none => simp [stmtOk, hp, hx] at hok
    | some xs => exact ⟨xs, rfl⟩
  simp only [stmtOk, hp, hx, Bool.and_eq_true] at hok
  exact hok.1.2

/-- **Names that pass `namesEq` are `want`'s, each once.** -/
theorem namesEq_spec {got : Option (List String)} {want : List String} (he : namesEq got want = true) :
    ∃ ns, got = some ns ∧ ns.Nodup ∧ ∀ v, v ∈ ns ↔ v ∈ want := by
  cases got with
  | none => simp [namesEq] at he
  | some ns =>
    simp only [namesEq, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true, List.contains_iff_mem] at he
    exact ⟨ns, rfl, he.1.1, fun v => ⟨he.1.2 v, he.2 v⟩⟩

/-- **A record that names its registration's names** has the registration's ports as its roots' ports and the
registration's registered values as its registered values, each a string named once. -/
theorem recordOk_spec {rec : Lean.Json} {n : Names} (hr : recordOk rec n = true) :
    (∃ ps, namesAt rec "roots" "port" = some ps ∧ ps.Nodup ∧ ∀ p, p ∈ ps ↔ p ∈ n.ports) ∧
      ∃ vs, namesAt rec "registered" "value" = some vs ∧ vs.Nodup ∧ ∀ v, v ∈ vs ↔ v ∈ n.values := by
  simp only [recordOk, Bool.and_eq_true] at hr
  exact ⟨namesEq_spec hr.1, namesEq_spec hr.2⟩

/-- **The records `recordsOk` passes**: the run's statement and round 1's record decode, and each names its
registration's names. -/
theorem recordsOk_spec {x m₁ : Msg} (hr : recordsOk h x m₁ = true) :
    ∃ st rec, decode x = some st ∧ jsonMsg m₁ = some rec ∧ recordOk st.record h.boundaryNames = true ∧
      recordOk rec h.interiorNames = true := by
  unfold recordsOk at hr
  cases hd : decode x with
  | none => simp [hd] at hr
  | some st =>
    cases hj : jsonMsg m₁ with
    | none => simp [hd, hj] at hr
    | some rec =>
      simp only [hd, hj, Bool.and_eq_true] at hr
      exact ⟨st, rec, rfl, rfl, hr.1, hr.2⟩

section At

variable {x m₁ xS : Msg} {c₀ : FlockDeploy.Coin} {c₁ : Stage h} {t : ℕ}

theorem parse_of_stmtOkAt (hok : stmtOkAt h x m₁ xS c₀ c₁ t = true) : ∃ s, parse h x = some s := by
  cases hp : parse h x with
  | none => simp [stmtOkAt, hp] at hok
  | some s => exact ⟨s, rfl⟩

/-- **A checked statement has its template's rounds.** -/
theorem k_of_stmtOkAt (hok : stmtOkAt h x m₁ xS c₀ c₁ t = true) : (FlockDeploy.V true).k xS = k₀ h t := by
  obtain ⟨s, hp⟩ := parse_of_stmtOkAt h hok
  simp only [stmtOkAt, hp, Bool.and_eq_true, beq_iff_eq] at hok
  exact hok.1.1.2

/-- **A checked statement is in scope** (`sessionOk`). -/
theorem sessionOk_of_stmtOkAt (hok : stmtOkAt h x m₁ xS c₀ c₁ t = true) : sessionOk h t xS = true := by
  obtain ⟨s, hp⟩ := parse_of_stmtOkAt h hok
  simp only [stmtOkAt, hp, Bool.and_eq_true] at hok
  exact hok.1.2

/-- **A checked statement is in the scope Flock's soundness proof covers, with hidden outputs** (`sessionOk`'s scope
check and hidden-outputs tag). -/
theorem scoped_of_stmtOkAt (hok : stmtOkAt h x m₁ xS c₀ c₁ t = true) :
    FlockDeploy.scopeCheck true xS = true ∧ hiddenOf xS = true := by
  have hs := sessionOk_of_stmtOkAt h hok
  simp only [sessionOk, Bool.and_eq_true] at hs
  exact ⟨hs.1.2, hs.2⟩

end At

/-- **The statement check's round passed**: its answer is a message only where the check passes. -/
theorem stmtOk_of_next {x : Msg} {as : List Msg} {cs : List FlockDeploy.Coin} {m : Msg} (hcs : cs.length = as.length)
    (hlen : as.length = h.stage + 2) (hm : next h x as cs = .ret (.msg m)) :
    stmtOk h x (as.getD 1 []) (as.getD (h.stage + 1) []) (cs.getD 0 default) (stageOf h cs) = true := by
  rw [next_stmt h x hcs hlen] at hm
  cases hok : stmtOk h x (as.getD 1 []) (as.getD (h.stage + 1) []) (cs.getD 0 default) (stageOf h cs)
  · rw [hok] at hm
    cases hm
  · rfl

/-! ## The units' templates -/

/-- **A proof block of the members' partition has one of their templates.** -/
theorem templateOf_lt {ms : Spot.Bits.Members} {u : ℕ} (hu : u < Properties.Spot.unitBase ms ms.length) :
    templateOf ms u < ms.length := by
  unfold templateOf
  cases hf : (List.range ms.length).find? (fun t => decide (u < Properties.Spot.unitBase ms (t + 1))) with
  | none =>
    exfalso
    rw [List.find?_eq_none] at hf
    have hpos : 0 < ms.length := by
      rcases Nat.eq_zero_or_pos ms.length with h0 | h0
      · rw [h0] at hu
        simp [Properties.Spot.unitBase] at hu
      · exact h0
    have := hf (ms.length - 1) (List.mem_range.mpr (by omega))
    rw [Nat.sub_add_cancel hpos] at this
    exact this (by simpa using hu)
  | some t =>
    simpa using List.mem_range.mp (List.mem_of_find?_eq_some hf)

/-- **A checked unit is a proof block of the members' partition.** -/
theorem lt_of_mem_checked {x : Msg} {r : Fin ((code h).V.k x) → FlockDeploy.Coin} {u : ℕ}
    (hu : u ∈ (code h).checked x r) : u < Properties.Spot.unitBase h.ms h.ms.length := by
  obtain ⟨b₁, b₂, hb⟩ : ∃ b₁ b₂, (code h).checked x r =
      ((reg h x).map fun g => g.checked (units ((base h).atOrder (order h)) x) b₁ b₂).getD [] := ⟨_, _, rfl⟩
  rw [hb] at hu
  unfold reg at hu
  cases hd : decode x with
  | none => simp [hd] at hu
  | some st =>
    have hN : units ((base h).atOrder (order h)) x = Properties.Spot.unitBase h.ms h.ms.length := by
      obtain ⟨s, hp⟩ : ∃ s, parse h x = some s := ⟨_, by unfold parse; rw [hd]; rfl⟩
      show (match parse h x with
        | some s => ((partition h s).getD ⟨[]⟩).blocks.length
        | none => 0) = _
      rw [hp]
      exact _root_.Flock.Bridge.Members.length_blocks_membersPart (ms := h.ms)
    simp only [hd, Option.map_some, Option.getD_some, hN] at hu
    unfold RegTwo.checked at hu
    split at hu
    · exact List.mem_range.mp (List.mem_filter.mp hu).1
    · exact List.mem_range.mp hu

/-! ## The sequential structure -/

/-- The message that states the session statements, among the prefix's. -/
theorem getD_take_pre (as : List Msg) :
    (as.take (h.stage + 2)).getD (h.stage + 1) [] = as.getD (h.stage + 1) [] := by
  rw [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD,
    List.getElem?_take_of_lt (show h.stage + 1 < h.stage + 2 by omega)]

/-- The message that states the session statements, read off a transcript's prefix. -/
theorem getD_fst_pre (ms rest : List (Msg × Out)) (hn : h.stage + 1 < ms.length) :
    ((ms ++ rest).getD (h.stage + 1) ([], .rej)).1 = ((ms.take (h.stage + 2)).map Prod.fst).getD (h.stage + 1) [] := by
  rw [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD, List.getElem?_append_left hn, List.getElem?_map,
    List.getElem?_take_of_lt (show h.stage + 1 < h.stage + 2 by omega), List.getElem?_eq_getElem hn]
  rfl

/-- **The two-stage verifier's sequential structure**: rounds `0` to `stage + 1`, then template `t`'s session on its
`k₀ t` rounds from `off t`, Flock's masked verifier at the statement round `stage + 1` states for it; a refused run
shows `xS₀ t`'s refused run. -/
noncomputable def seq (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) :
    SessionSeq (deployOf (FlockDeploy.deployZk rel) (code h)) where
  pre _ := h.stage + 2
  nS _ := h.ms.length
  kS _ := k₀ h
  idx _ u := templateOf h.ms u
  stmt _ as i := stmtAt h (as.getD (h.stage + 1) []) i
  xR _ i := h.xS₀ i
  mark := []
  pre_pos _ := Nat.succ_pos _
  kS_pos _ i := k₀_pos h i
  k_eq _ := rfl
  idx_lt x r u hu := by
    obtain ⟨p, hp⟩ := hu
    exact templateOf_lt (lt_of_mem_checked h (List.mem_of_getElem? hp))
  rounds_iff x u t := by
    have e : ∀ t : Fin ((V h).k x), t ∈ ((univ.filter fun t : Fin ((V h).k x) => off h (templateOf h.ms u) ≤ t.val ∧
        t.val < off h (templateOf h.ms u) + k₀ h (templateOf h.ms u)).filter fun t => (order h x).second < t) ↔
        off h (templateOf h.ms u) ≤ t.val ∧ t.val < off h (templateOf h.ms u) + k₀ h (templateOf h.ms u) := fun t => by
      rw [Finset.mem_filter, Finset.mem_filter]
      have := stage_le_off h (templateOf h.ms u)
      constructor
      · rintro ⟨⟨_, h1⟩, _⟩
        exact h1
      · intro h1
        exact ⟨⟨mem_univ t, h1⟩, by show h.stage < t.val; omega⟩
    exact e t
  next_pre x as cs hne hle hcs := by
    change ∃ o, next h x as cs = .ret o ∧ o ≠ .acc
    change as.length ≤ h.stage + 2 at hle
    by_cases h1 : as.length ≤ h.stage + 1
    · exact ⟨_, next_coin h x hne hcs h1, by split <;> nofun⟩
    · exact ⟨_, next_stmt h x hcs (by omega), by split <;> nofun⟩
  next_mid H x as cs hpre hlt hcs := by
    change ((next h x as cs).eval H).1 ≠ .acc
    change h.stage + 2 < as.length at hpre
    change as.length < off h h.ms.length + 1 at hlt
    rw [next_long h x as cs hpre hcs]
    split
    · exact sessStep_ne_acc h H _ _ _ _
    · have : (as.length == off h h.ms.length + 1) = false := by
        simp only [beq_eq_false_iff_ne]
        omega
      rw [this, Bool.false_and]
      nofun
  next_sess H x as cs i hi h1 h2 hcs hund := by
    change i < h.ms.length at hi
    change off h i < as.length at h1
    change as.length ≤ off h i + k₀ h i at h2
    have h3 : h.stage + 2 < as.length := lt_of_le_of_lt (stage_le_off h i) h1
    change (next h x as cs).eval H =
      (markOut [] ((FlockDeploy.next true (stmtAt h ((as.take (h.stage + 2)).getD (h.stage + 1) []) i)
          (as.drop (off h i)) (cs.drop (off h i))).eval H).1,
        ((FlockDeploy.next true (stmtAt h ((as.take (h.stage + 2)).getD (h.stage + 1) []) i) (as.drop (off h i))
          (cs.drop (off h i))).eval H).2)
    rw [getD_take_pre, next_long h x as cs h3 hcs, slotOf_eq h hi h1 h2]
    refine sessStep_eval h H i _ as cs fun j hj hj' => ?_
    obtain ⟨m, hm⟩ := hund j hj hj'
    rw [getD_take_pre] at hm
    exact not_decidedAt hm
  next_last H x as cs hlen hcs hacc i hi := by
    change as.length = off h h.ms.length + 1 at hlen
    change i < h.ms.length at hi
    have h3 : h.stage + 2 < as.length := by have := stage_le_off h h.ms.length; omega
    change ((next h x as cs).eval H).1 = .acc at hacc
    rw [next_long h x as cs h3 hcs, hlen, slotOf_last] at hacc
    have hv : verdict h (as.getD (h.stage + 1) []) as cs = true := by
      cases hv : verdict h (as.getD (h.stage + 1) []) as cs
      · rw [hv, Bool.and_false] at hacc
        cases hacc
      · rfl
    simp only [verdict, List.all_eq_true, List.mem_range] at hv
    obtain ⟨j, hj, hd⟩ := List.any_eq_true.mp (hv i hi)
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hd
    refine ⟨j, hd.1, Nat.lt_succ_iff.mp (List.mem_range.mp hj), fun m => ?_⟩
    change ((FlockDeploy.next true (stmtAt h ((as.take (h.stage + 2)).getD (h.stage + 1) []) i)
      ((as.drop (off h i)).take j) ((cs.drop (off h i)).take j)).eval H).1 ≠ .msg m
    rw [getD_take_pre]
    exact decidedAt_ne_msg hd.2 H m
  flock_ne H xS as cs := flock_ne_nil H xS as cs
  k_stmt x as cs m i hlen hcs hm hi := by
    change as.length = h.stage + 2 at hlen
    change i < h.ms.length at hi
    change (FlockDeploy.V true).k (stmtAt h (as.getD (h.stage + 1) []) i) = k₀ h i
    exact (stmtOk_at h (stmtOk_of_next h hcs hlen hm) hi).1
  k_xR _ _ := rfl
  refuses _ i c := refuses_empty (h.xS₀ i) c
  session_pass x u ms rest out cs m hlen hidx hms hcs hm := by
    change ms.length = off h (templateOf h.ms u) at hlen
    change templateOf h.ms u < h.ms.length at hidx
    change session h u ⟨ms ++ rest, out⟩ =
      some (stmtAt h (((ms.take (h.stage + 2)).map Prod.fst).getD (h.stage + 1) []) (templateOf h.ms u),
        cutSeen [] (k₀ h (templateOf h.ms u)) rest)
    change cs.length = h.stage + 2 at hcs
    change next h x ((ms.take (h.stage + 2)).map Prod.fst) cs = .ret (.msg m) at hm
    have htake : (ms ++ rest).take (off h (templateOf h.ms u)) = ms := List.take_left' hlen
    have hdrop : (ms ++ rest).drop (off h (templateOf h.ms u)) = rest := List.drop_left' hlen
    have h3 := stage_le_off h (templateOf h.ms u)
    have hall : ms.all (answered ·.2) = true := List.all_eq_true.mpr fun e he => by
      obtain ⟨m', hm'⟩ := hms e he
      rw [hm']
      rfl
    have hle : off h (templateOf h.ms u) ≤ (ms ++ rest).length := by
      rw [List.length_append]
      omega
    have hpl : ((ms.take (h.stage + 2)).map Prod.fst).length = h.stage + 2 := by
      rw [List.length_map, List.length_take]
      omega
    have hok := stmtOk_of_next h (hcs.trans hpl.symm) hpl hm
    have hsess := (stmtOk_at h hok hidx).2
    unfold session
    rw [htake, hdrop, ite_eq_left ⟨hle, hall⟩]
    change (if sessionOk h (templateOf h.ms u)
        (stmtAt h ((ms ++ rest).getD (h.stage + 1) ([], .rej)).1 (templateOf h.ms u)) = true then _ else _) = _
    rw [getD_fst_pre h ms rest (by omega), ite_eq_left hsess]
  session_refused x u ms a hlen hms := by
    change ms.length < off h (templateOf h.ms u) at hlen
    change session h u ⟨ms ++ [(a, .rej)], .rej⟩ = some (h.xS₀ (templateOf h.ms u), refusedSeen)
    unfold session
    rw [ite_eq_right]
    rintro ⟨hle, hall⟩
    simp only [List.length_append, List.length_singleton] at hle
    have he : off h (templateOf h.ms u) = (ms ++ [(a, Out.rej)]).length := by
      simp only [List.length_append, List.length_singleton]
      omega
    rw [he, List.take_length] at hall
    simp [List.all_append, answered] at hall

/-- **SPOT[Flock]'s sessions are Flock runs** (`SessionRunsTo`), at every relation and every `H`. -/
theorem sessionRunsTo (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (H : Msg → Msg) :
    SessionRunsTo H (SamplingDeploy.deploy rel (code h)) :=
  sessionRunsTo_of_seq (seq h rel) H

end SpotFlock
