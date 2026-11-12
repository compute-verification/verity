module

public import Proofs.Properties.Verity.Sampled

/-!
# A `--zk` session's record holds the coins its tree's log shows opened

`Spot.Challenger.sessionTreeReason` accepts a session's record (`none`), at any tables, only where the record's
coin tree is one the release log shows sampled (`trees`, `coinTreesOf`'s) under the record's root, and each round the
record gives coins of is one the log shows opened of that tree, with those coins' SHA-512 (`round_held`): its link's
points as the session stream's round 0, and each stream's rounds. Its loops return at their first reason, so where a
loop returns none, each of its steps yielded (`forIn_held`).
-/

@[expose] public section

namespace Verity.Sampled.Trees

open Lean (Json)
open Spot Spot.Challenger

theorem strIs {j : Json} {s : String} (h : jsonStrIs j s = true) : j = .str s := by
  cases j <;> simp_all [jsonStrIs]

/-- **A loop that returns at its first reason** (a step `.done (some (some w), _)`, or else `.yield (none, _)`) returns
a reason, or none with each step yielded. -/
theorem forIn_held {α γ σ : Type} {P : α → Prop}
    {f : α → Option (Option γ) × σ → Id (ForInStep (Option (Option γ) × σ))}
    (hf : ∀ x u s, f x (none, u) = s →
      (∃ w u', s = ForInStep.done (some (some w), u')) ∨ (∃ u', s = ForInStep.yield (none, u') ∧ P x)) :
    ∀ (l : List α) (u : σ), (∃ w, Prod.fst (forIn (m := Id) l (none, u) f) = some (some w)) ∨
      (Prod.fst (forIn (m := Id) l (none, u) f) = none ∧ ∀ x ∈ l, P x)
  | [], _ => .inr ⟨rfl, fun _ h => absurd h List.not_mem_nil⟩
  | a :: l, u => by
    rw [List.forIn_cons]
    rcases hf a u _ rfl with ⟨w, u', hw⟩ | ⟨u', hy, hp⟩
    · exact .inl ⟨w, by rw [hw]; rfl⟩
    · rw [hy]
      rcases forIn_held hf l u' with ⟨w, hw⟩ | ⟨hn, hall⟩
      · exact .inl ⟨w, hw⟩
      · exact .inr ⟨hn, fun x hx => (List.mem_cons.1 hx).elim (fun e => e ▸ hp) (hall x)⟩

theorem forIn_held_arr {α γ σ : Type} {P : α → Prop}
    {f : α → Option (Option γ) × σ → Id (ForInStep (Option (Option γ) × σ))}
    (hf : ∀ x u s, f x (none, u) = s →
      (∃ w u', s = ForInStep.done (some (some w), u')) ∨ (∃ u', s = ForInStep.yield (none, u') ∧ P x))
    (xs : Array α) (u : σ) :
    (∃ w, Prod.fst (forIn (m := Id) xs (none, u) f) = some (some w)) ∨
      (Prod.fst (forIn (m := Id) xs (none, u) f) = none ∧ ∀ x ∈ xs, P x) := by
  rw [← Array.forIn_toList]
  exact (forIn_held hf xs.toList u).imp id fun ⟨hn, hall⟩ => ⟨hn, fun x hx => hall x (Array.mem_def.1 hx)⟩

/-- **A loop that returns at its first reason, then its reason**: where none returns, each step yielded. -/
theorem ret_held {α γ σ : Type} {P : α → Prop}
    {f : α → Option (Option γ) × σ → Id (ForInStep (Option (Option γ) × σ))}
    {g : Option (Option γ) × σ → Id (Option γ)}
    (hf : ∀ x u s, f x (none, u) = s →
      (∃ w u', s = ForInStep.done (some (some w), u')) ∨ (∃ u', s = ForInStep.yield (none, u') ∧ P x))
    (hg : ∀ w u, g (some (some w), u) ≠ none) {xs : Array α} {u : σ}
    (h : (forIn (m := Id) xs (none, u) f >>= g) = none) : ∀ x ∈ xs, P x := by
  rcases forIn_held_arr hf xs u with ⟨w, hw⟩ | ⟨-, hall⟩
  · have e : forIn (m := Id) xs (none, u) f = (some (some w), Prod.snd (forIn (m := Id) xs (none, u) f)) :=
      Prod.ext hw rfl
    rw [e] at h
    exact absurd h (hg w _)
  · exact hall

/-- **A loop that returns at its first reason, as a step of an outer one**: the outer step returns the inner reason,
or yields none with each inner step yielded. -/
theorem step_held {α γ σ τ : Type} {P : α → Prop}
    {f : α → Option (Option γ) × σ → Id (ForInStep (Option (Option γ) × σ))}
    {g : Option (Option γ) × σ → Id (ForInStep (Option (Option γ) × τ))}
    (hf : ∀ x u s, f x (none, u) = s →
      (∃ w u', s = ForInStep.done (some (some w), u')) ∨ (∃ u', s = ForInStep.yield (none, u') ∧ P x))
    (hg : ∀ w u, ∃ u', g (some (some w), u) = ForInStep.done (some (some w), u'))
    (hg0 : ∀ u, ∃ u', g (none, u) = ForInStep.yield (none, u')) {xs : Array α} {u : σ}
    {s : Id (ForInStep (Option (Option γ) × τ))} (h : (forIn (m := Id) xs (none, u) f >>= g) = s) :
    (∃ w u', s = ForInStep.done (some (some w), u')) ∨
      (∃ u', s = ForInStep.yield (none, u') ∧ ∀ x ∈ xs, P x) := by
  subst h
  rcases forIn_held_arr hf xs u with ⟨w, hw⟩ | ⟨hn, hall⟩
  · have e : forIn (m := Id) xs (none, u) f = (some (some w), Prod.snd (forIn (m := Id) xs (none, u) f)) :=
      Prod.ext hw rfl
    rw [e]
    obtain ⟨u', hu⟩ := hg w _
    exact .inl ⟨w, u', hu⟩
  · have e : forIn (m := Id) xs (none, u) f = (none, Prod.snd (forIn (m := Id) xs (none, u) f)) := Prod.ext hn rfl
    rw [e]
    obtain ⟨u', hu⟩ := hg0 _
    exact .inr ⟨u', hu, hall⟩

/-- Round `x.2` of a record's stream `name`, the round's record `x.1`: the coins it gives are a round the log shows
opened of tree `t`, with those coins' SHA-512. -/
def RoundHeld (t : LoggedTree) (name : String) (x : Json × Nat) : Prop :=
  ∀ cs, recordCoins? (getD x.1 "coins") = some cs → ∃ o, t.opening name x.2 = some o ∧ o.coins = sha512Hex cs

/-- Each round of a record's stream `s` is held (`RoundHeld`) of tree `t`. -/
def StreamHeld (t : LoggedTree) (s : Json) : Prop :=
  ∀ name rounds, getD s "stream" = .str name → getD s "rounds" = .arr rounds →
    ∀ x ∈ rounds.zipIdx, RoundHeld t name x

/-- **A record `sessionTreeReason` accepts, at any tables, holds the coins its tree's log shows opened**: its coin
tree's root is that of a tree of the log, and each round the record gives coins of (`Verity.Sampled.recordRound`) is
one the log shows opened of that tree, with those coins' SHA-512. -/
theorem round_held {trees : Array LoggedTree} {tables : Array (String × Option Nat)} {record : Json} {stream : String}
    {k : Nat} {cs : ByteArray} (h : sessionTreeReason trees tables record = none)
    (hr : Verity.Sampled.recordRound record stream k = some cs) :
    ∃ t ∈ trees, Verity.Sampled.recordTree record = t.root ∧
      ∃ o, t.opening stream k = some o ∧ o.coins = sha512Hex cs := by
  unfold sessionTreeReason at h
  dsimp only [Id.run] at h
  split at h
  case h_2 => cases h
  rename_i t ht
  refine ⟨t, Array.mem_of_find?_eq_some ht, ?_, ?_⟩
  · have hp := Array.find?_some ht
    unfold Verity.Sampled.recordTree
    rw [strIs hp]
  split at h
  case isFalse => cases h
  split at h
  case h_2 => cases h
  rename_i ps hlink
  split at h
  case isFalse => cases h
  split at h
  case h_2 => cases h
  rename_i rootF points streams htri
  rcases hlo : t.opening TREE_SESSION 0 with _ | o
  · simp only [hlo, Bool.false_eq_true, ↓reduceIte] at h
    all_goals (split at h <;> (try split at h) <;> cases h)
  simp only [hlo] at h
  split at h
  case isFalse => cases h
  split at h
  case isFalse => cases h
  split at h
  case isFalse => cases h
  rename_i hlinkOk
  simp only [Prod.mk.injEq] at htri hlink
  obtain ⟨-, hpts, hst⟩ := htri
  obtain ⟨-, -, hps⟩ := hlink
  have hall : ∀ s ∈ streams, StreamHeld t s := by
    refine ret_held ?_ (fun w _ hw => by cases hw) h
    intro s u res hres
    split at hres
    case h_2 => exact .inl ⟨_, _, hres.symm⟩
    rename_i name rounds hsr
    refine Or.imp id ?_ (step_held (P := RoundHeld t name) ?_ ?_ ?_ hres)
    · rintro ⟨u', hy, hrs⟩
      refine ⟨u', hy, fun name' rounds' hn hr' => ?_⟩
      simp only [Prod.mk.injEq] at hsr
      cases hn.symm.trans hsr.1
      cases hr'.symm.trans hsr.2
      exact hrs
    · intro x u res' hres'
      split at hres'
      case h_2 => exact .inl ⟨_, _, hres'.symm⟩
      rename_i coins n hcn
      rcases ho : t.opening name x.2 with _ | o
      · simp only [ho, Bool.false_eq_true, ↓reduceIte] at hres'
        all_goals (split at hres' <;> exact .inl ⟨_, _, hres'.symm⟩)
      simp only [ho] at hres'
      split at hres'
      case isFalse => exact .inl ⟨_, _, hres'.symm⟩
      split at hres'
      case isFalse => exact .inl ⟨_, _, hres'.symm⟩
      rename_i hok
      refine .inr ⟨_, hres'.symm, fun cs' hcs => ?_⟩
      simp only [Prod.mk.injEq] at hcn
      cases hcs.symm.trans hcn.1
      simp only [Bool.and_eq_true, beq_iff_eq] at hok
      exact ⟨o, ho, hok.2⟩
    · exact fun _ _ => ⟨(), rfl⟩
    · exact fun _ => ⟨(), rfl⟩
  unfold Verity.Sampled.recordRound at hr
  split at hr
  · rename_i hsess
    simp only [Bool.and_eq_true, beq_iff_eq] at hsess
    obtain ⟨rfl, rfl⟩ := hsess
    rw [hps] at hr
    dsimp only at hr
    cases hr.symm.trans hpts
    simp only [Bool.and_eq_true, beq_iff_eq] at hlinkOk
    exact ⟨o, hlo, hlinkOk.2⟩
  · rw [hst] at hr
    dsimp only at hr
    obtain ⟨x, hx, hr⟩ := Option.bind_eq_some_iff.1 hr
    have hname := Array.find?_some hx
    split at hname
    case h_2 => cases hname
    rename_i n hxn
    cases beq_iff_eq.1 hname
    split at hr
    case h_2 => cases hr
    rename_i rs hxr
    obtain ⟨r, hrk, hcs⟩ := Option.bind_eq_some_iff.1 hr
    exact hall x (Array.mem_of_find?_eq_some hx) _ rs hxn hxr (r, k) (Array.mem_zipIdx_iff_getElem?.2 hrk) cs hcs

/-- **Each session `sessionsTreeReason` accepts has a record `sessionTreeReason` accepts at its tables.** -/
theorem sessions_held {trees : Array LoggedTree} {records : Array (String × Array (String × Option Nat) × Json)}
    (h : sessionsTreeReason trees records = none) : ∀ x ∈ records, sessionTreeReason trees x.2.1 x.2.2 = none := by
  unfold sessionsTreeReason at h
  dsimp only [Id.run] at h
  refine ret_held ?_ (fun w _ hw => by cases hw) h
  rintro ⟨name, tables, record⟩ u s hs
  split at hs
  case h_1 => exact .inl ⟨_, _, hs.symm⟩
  rename_i hne
  refine .inr ⟨_, hs.symm, ?_⟩
  cases hr : sessionTreeReason trees (name, tables, record).2.1 (name, tables, record).2.2 with
  | none => rfl
  | some w => exact (hne w (by rw [hr]; rfl)).elim

/-- `Except.ok`'s bind is the continuation at its value. -/
theorem ok_bind {ε α β : Type} (a : α) (f : α → Except ε β) : (Except.ok a >>= f) = f a := rfl

/-- **A loop invariant in `Except`**, over a loop whose steps only yield or throw: `I pre s` holds of the members `pre`
read so far and the state `s` after them. -/
theorem forIn_inv {α σ ε : Type} {f : α → σ → Except ε (ForInStep σ)} {I : List α → σ → Prop}
    (step : ∀ pre x s r, I pre s → f x s = .ok r → ∃ t, r = .yield t ∧ I (pre ++ [x]) t) :
    ∀ (l pre : List α) (s t : σ), I pre s → forIn l s f = .ok t → I (pre ++ l) t
  | [], pre, s, t, hs, h => by
    rw [List.forIn_nil] at h
    cases h
    simpa using hs
  | x :: l, pre, s, t, hs, h => by
    rw [List.forIn_cons] at h
    cases hf : f x s with
    | error e => rw [hf] at h; cases h
    | ok r =>
      rw [hf] at h
      obtain ⟨t', rfl, ht'⟩ := step pre x s r hs hf
      simpa using forIn_inv step l (pre ++ [x]) t' t ht' h

theorem forIn_inv_run {α σ ε : Type} {f : α → σ → Except ε (ForInStep σ)} {I : List α → σ → Prop}
    (step : ∀ pre x s r, I pre s → f x s = .ok r → ∃ t, r = .yield t ∧ I (pre ++ [x]) t) {l : List α} {s t : σ}
    (hs : I [] s) (h : (forIn l s f >>= fun x => pure x) = .ok t) : I l t := by
  cases hl : forIn l s f with
  | error e => rw [hl] at h; cases h
  | ok t' =>
    rw [hl] at h
    have e : t' = t := Except.ok.inj h
    subst e
    simpa using forIn_inv step l [] s t' hs hl

/-- **An open line** of a release log: a tree line (`treeLine`) whose event isn't a tree's sample, naming tree `root`,
the round `o.round` of stream `o.stream` it opens, and its coins' SHA-512 `o.coins`. -/
def OpenLine (l root : String) (o : TreeOpening) : Prop :=
  treeLine l = true ∧ ∃ d, event l = .ok d ∧ jsonStrIs (getD d "event") TREE_EVENT = false ∧
    getD d "tree" = .str root ∧ getD d "stream" = .str o.stream ∧ nat? (getD d "round") = some o.round ∧
    getD d "coins" = .str o.coins

/-- Each round a tree of `trees` shows opened is an open line of `log` naming that tree. -/
def OpenedIn (log : List String) (trees : Array LoggedTree) : Prop :=
  ∀ t ∈ trees, ∀ o ∈ t.opened, ∃ l ∈ log, OpenLine l t.root o

theorem mem_modify {β : Type} {xs : Array β} {i : Nat} {g : β → β} {b : β} (h : b ∈ xs.modify i g) :
    b ∈ xs ∨ ∃ a, xs[i]? = some a ∧ b = g a := by
  obtain ⟨j, hj, rfl⟩ := Array.mem_iff_getElem.1 h
  have hj' : j < xs.size := by simpa using hj
  rw [Array.getElem_modify]
  split
  · rename_i hij
    subst hij
    exact .inr ⟨xs[i], Array.getElem?_eq_getElem hj', rfl⟩
  · exact .inl (Array.getElem_mem hj')

/-- **Each round the release log's trees show opened is an open line of the log** naming that tree, with the round's
stream, number and coins' SHA-512 (`coinTreesOf`). -/
theorem opened_in {log : List String} {trees : Array LoggedTree} (h : coinTreesOf log = .ok trees) :
    OpenedIn log trees := by
  unfold coinTreesOf at h
  dsimp only at h
  refine forIn_inv_run (I := OpenedIn) ?_ (fun t ht => absurd ht (by simp)) h
  intro pre line ts r hI hr
  have keep : ∀ ts', (∀ t ∈ ts', t ∈ ts ∨ t.opened = #[]) → OpenedIn (pre ++ [line]) ts' := by
    intro ts' hts t ht o ho
    rcases hts t ht with ht | he
    · obtain ⟨l, hl, hol⟩ := hI t ht o ho
      exact ⟨l, List.mem_append_left _ hl, hol⟩
    · rw [he] at ho
      simp at ho
  split at hr
  case isFalse => cases hr; exact ⟨ts, rfl, keep ts fun t ht => .inl ht⟩
  rename_i htl
  cases hev : event line with
  | error e => rw [hev] at hr; cases hr
  | ok d =>
    rw [hev, ok_bind] at hr
    split at hr
    · split at hr
      case isFalse => cases hr
      split at hr
      case isTrue => cases hr
      cases hr
      refine ⟨_, rfl, keep _ fun t ht => ?_⟩
      rcases Array.mem_push.1 ht with ht | rfl
      · exact .inl ht
      · exact .inr rfl
    · rename_i hopen
      split at hr
      case isFalse => cases hr
      split at hr
      case h_2 => cases hr
      rename_i root htree
      split at hr
      case h_2 => cases hr
      rename_i i hidx
      split at hr
      case h_2 => cases hr
      rename_i t hget
      split at hr
      case h_2 => cases hr
      rename_i s hstream
      split at hr
      case h_2 => cases hr
      rename_i k n hkn
      obtain ⟨hi, hp, -⟩ := Array.findIdx?_eq_some_iff_getElem.1 hidx
      have hti : ts[i] = t := by
        rw [Array.getElem?_eq_getElem hi] at hget
        exact Option.some.inj hget
      rw [hti] at hp
      have hroot : t.root = root := beq_iff_eq.1 hp
      have htmem : t ∈ ts := hti ▸ Array.getElem_mem hi
      simp only [Prod.mk.injEq] at hkn
      split at hr
      all_goals
        try dsimp only at hr
        split at hr
        · rename_i hcond
          simp only [Bool.and_eq_true] at hcond
          have hhex := hcond.2
          cases hr
          refine ⟨_, rfl, ?_⟩
          intro t' ht' o ho
          rcases mem_modify ht' with ht' | ⟨a, ha, rfl⟩
          · obtain ⟨l, hl, hol⟩ := hI t' ht' o ho
            exact ⟨l, List.mem_append_left _ hl, hol⟩
          obtain rfl : a = t := Option.some.inj (ha.symm.trans hget)
          simp only [Array.mem_push] at ho
          rcases ho with ho | rfl
          · obtain ⟨l, hl, hol⟩ := hI _ htmem o ho
            exact ⟨l, List.mem_append_left _ hl, hol⟩
          refine ⟨line, List.mem_append_right _ (List.mem_singleton_self _), htl, d, hev, by simpa using hopen,
            by simpa [hroot] using htree, hstream, hkn.1, ?_⟩
          cases hc : getD d "coins" <;> rw [hc] at hhex <;> first | rfl | cases hhex
        · cases hr


/-- **A round a record gives coins of is an open line of the release log** naming the record's tree, with that round's
stream and number and those coins' SHA-512: where the log's trees are `coinTreesOf`'s and `sessionTreeReason` accepts
the record against them, at any tables. -/
theorem open_held {log : List String} {trees : Array LoggedTree} {tables : Array (String × Option Nat)} {record : Json}
    {stream : String} {k : Nat} {cs : ByteArray} (ht : coinTreesOf log = .ok trees)
    (h : sessionTreeReason trees tables record = none) (hr : Verity.Sampled.recordRound record stream k = some cs) :
    ∃ l ∈ log, ∃ n after, OpenLine l (Verity.Sampled.recordTree record) ⟨stream, k, n, after, sha512Hex cs⟩ := by
  obtain ⟨t, htm, hroot, o, ho, hc⟩ := round_held h hr
  have hp := Array.find?_some ho
  simp only [Bool.and_eq_true, beq_iff_eq] at hp
  obtain ⟨l, hl, hol⟩ := opened_in ht t htm o (Array.mem_of_find?_eq_some ho)
  refine ⟨l, hl, o.n, o.after, ?_⟩
  rw [hroot, ← hc, ← hp.1, ← hp.2]
  exact hol

/-- `Except`'s bind is `.ok` only at an `.ok` value whose continuation is. -/
theorem bind_ok {ε α β : Type} {x : Except ε α} {g : α → Except ε β} {b : β} (h : (x >>= g) = .ok b) :
    ∃ a, x = .ok a ∧ g a = .ok b := by
  cases hx : x with
  | error e => rw [hx] at h; cases h
  | ok a => rw [hx] at h; exact ⟨a, rfl, h⟩

/-- **v2's acceptance holds its sessions to the challenger's trees**: an audit `checkBlobs` accepts read the trees
(`treesOf`), and its sessions' records hold to them on the verifier's own files (`treesHeld`). -/
theorem accepted_trees {p : Spot.OneStageZk.Params} {c : Spot.OneStage.Commitment}
    {e : Spot.OneStageZk.Evidence} {bs : Core.Blobs} {pub : Spot.OneStage.Public}
    {prof : Core.Integrity} (h : Spot.OneStageZk.checkBlobs p c e bs = .accept pub prof) :
    ∃ ts, Spot.OneStageZk.treesOf p.base e.base = .ok ts ∧
      Spot.OneStageZk.treesHeld p.own ts e bs = .ok () := by
  unfold Spot.OneStageZk.checkBlobs Core.atCoins at h
  split at h
  case h_2 => cases h
  replace h := (Spot.OneStageZk.checkBlobsAt_accept h).1
  unfold Spot.OneStageZk.checkOneAt at h
  dsimp only at h
  split at h
  case h_1 => cases h
  rename_i run hrun
  obtain ⟨_, -, hrun⟩ := bind_ok hrun
  obtain ⟨ts, hts, hrun⟩ := bind_ok hrun
  obtain ⟨_, -, hrun⟩ := bind_ok hrun
  obtain ⟨_, hheld, -⟩ := bind_ok hrun
  exact ⟨ts, hts, hheld⟩

/-- A blob `get` reads names only what its refusal says. -/
theorem get_ok {bs : Core.Blobs} {d w : String} {b : ByteArray} (w' : String) (h : bs.get d w = .ok b) :
    (bs.get d w').toOption = some b := by
  have e : (bs.get d w).toOption = (bs.get d w').toOption := by
    unfold Core.Blobs.get
    cases Flock.unhex d with
    | error _ => rfl
    | ok x =>
      simp only [ok_bind]
      split
      · rfl
      · split <;> rfl
  rw [h] at e
  exact e.symm

/-- **Each record v2's evidence names holds to the trees where `treesHeld` accepts**: the record a session's blob
parses to through `Flock.JsonParse.parse`, as `Verity.Sampled.recordOn` reads it (`null` where it doesn't parse),
is one `sessionTreeReason` accepts against them, at the tables `treesHeld` reads for its session. -/
theorem records_held {p : Spot.OneStageZk.Own} {ts : Array LoggedTree} {e : Spot.OneStageZk.Evidence}
    {bs : Core.Blobs} (h : Spot.OneStageZk.treesHeld p ts e bs = .ok ()) {r} (hr : r ∈ e.sessions)
    {w : String} {b : ByteArray} {str : String} {rec : Json} (hb : (bs.get r.record w).toOption = some b)
    (hs : String.fromUTF8? b = some str) (hp : (Flock.JsonParse.parse str).toOption.getD .null = rec) :
    ∃ tabs, sessionTreeReason ts tabs rec = none := by
  unfold Spot.OneStageZk.treesHeld at h
  dsimp only at h
  obtain ⟨named, hn, hrest⟩ := bind_ok h
  rw [← Array.forIn_toList] at hn
  have hI := forIn_inv (I := fun pre (named : Array (String × Array (String × Option Nat) × Json)) => ∀ r ∈ pre,
      ∀ w b str, (bs.get r.record w).toOption = some b → String.fromUTF8? b = some str →
      (r.name, Spot.OneStageZk.rootBTables p bs str, (Flock.JsonParse.parse str).toOption.getD .null) ∈
        named) ?_ e.sessions.toList [] #[] named (by simp) hn
  · have hsess : sessionsTreeReason ts named = none := by
      split at hrest
      case h_1 => cases hrest
      rename_i hne
      cases hw : sessionsTreeReason ts named with
      | none => rfl
      | some w => exact (hne w hw).elim
    have hmem := hI r (by simpa using hr) w b str hb hs
    rw [hp] at hmem
    exact ⟨_, sessions_held hsess (r.name, Spot.OneStageZk.rootBTables p bs str, rec) hmem⟩
  · intro pre x named' res hI' hres
    obtain ⟨b', hb', hres⟩ := bind_ok hres
    split at hres
    case h_2 => cases hres
    rename_i record hrec
    cases hres
    refine ⟨_, rfl, fun r' hr' w'' b'' str' hb'' hs' => ?_⟩
    rcases List.mem_append.1 hr' with hr' | hr'
    · exact Array.mem_push.2 (.inl (hI' r' hr' w'' b'' str' hb'' hs'))
    · rw [List.mem_singleton] at hr'
      subst hr'
      obtain rfl : b'' = b' := Option.some.inj (hb''.symm.trans (get_ok w'' hb'))
      obtain rfl : str' = record := Option.some.inj (hs'.symm.trans hrec)
      exact Array.mem_push.2 (.inr rfl)

end Verity.Sampled.Trees
