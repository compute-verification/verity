module

public import Proofs.Flock.ZkAny.RelayFrames
public import Proofs.Flock.Claims.ZK.HelloLean
import Proofs.Flock.Claims.ZK.Reference

/-!
# What the relay reads at the model's `Hello` and `Commit` is the model's

At the `Hello` the relay frames the session's `Hello` message at a nonce it samples from the operating system
(`Hello.msg`), which is the model's (`CoinLayer.helloMsg`) at the same nonce: `envOk_hello_tape`, where that nonce is
the tape's.

At a `Commit` the relay holds the worker's roots to the loop's level 0 (`Level0.commit`): the root it carries for table
`i` must be a level 0 the loop drew for that table of the session, at the table's `m`, which the session then owns. What
leaves is the loop's own `Commit`: for each table, the root of the level 0 bound to it (`Level0.ofSession`). Here that
root is the loop's sample for that table, whatever roots the worker sent, and whatever the loop's other sessions do.

* `envOk_hello_tape`: `EnvOk` at the model's `Hello`, where the relay's nonce is the tape's.
* `commitRead`: what the relay reads at a `Commit` frame against the loop's level-0 slots, `Session.relay`'s `Commit`
  branch (`Env.commit`).
* `CommitInv`, `commit_inv`: once `Level0.commit` passes, each slot is one of the loop's as sampled, but for what it is
  bound to; a level 0 bound to the session is bound by this `Commit`, at its table's index; and each table has one.
* `commitRead_drawn`: where the loop drew `ls` for the session (`SessionDraws`) and nothing is bound to it yet
  (`Unbound`), a `Commit` that leaves carries each table's name and the root of `ls` there, at tables of distinct names
  (`streamsOk_tables`: the coin tree's rep streams, `"{table}/rep{r}"`, are distinct, `SessionPlan.streamsOk`).
* `SlotsReach`: the loop's slots as its four updates of them leave them, across all its sessions: `Slots.name` at a
  `Hello`, `Slots.draw` at a level-0 request, `Level0.commit` at a `Commit` and `Slots.close` at a session's end.
  `slotsReach_inv`: a level 0 bound to a session was sampled for it (`BoundOwned`), and each live session has its count
  of tables (`Named`).
* `AfterDraw`, `afterDraw_inv`: from the session's sample of `ls` to its `Commit`, through any level-0 request's sample and
  other sessions' updates, the session's level 0 stay `ls`'s and nothing is bound to it (`DrawInv`): a level-0 request
  for a session the loop drew is refused (`Slots.admit`), or samples none where it has no tables.
* `envOk_commit_after`: `EnvOk` at the model's link whose roots are the sample's, where the relay read the `Commit`
  against the loop's slots after the session's sample.
-/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock (SessionSpec Stmt)
open _root_.Flock.Firewall.Level0 (Level0 Slot Slots commit ofSession)

/-- **At the model's `Hello`, `EnvOk` where the relay's nonce is the tape's**: where `x` sets up the session on `spec`
and the nonce the relay drew from the operating system is the tape's `ν`, the `Hello` it frames is the model's. -/
theorem envOk_hello_tape {x : Claims.Msg} {ν : Claims.ZKMal.CoinLayer.Nonce} {spec : SessionSpec} {sts : Array Stmt}
    {seen : List String} {ins : _} {e : Env} (hs : FlockDeploy.setupOf true x = some (ins, spec, sts))
    (hn : e.nonce = ⟨(Claims.ZKMal.CoinLayer.nb ν).toArray⟩) :
    EnvOk spec seen .hello e (Claims.ZKMal.CoinLayer.helloMsg x ν) := by
  show _root_.Flock.Firewall.Hello.msg spec e.nonce = _
  unfold Claims.ZKMal.CoinLayer.helloMsg
  rw [Claims.ZKMal.sessOf_flock hs, hn]

/-- **What the relay reads at a `Commit` frame `f`** of the session `sid`, with its tables `tables`, against the loop's
level-0 slots `R`: the roots its tables' level 0 now bound, in table order, and the slots after; else why it does not
leave (`Session.relay`'s `Commit` branch). -/
def commitRead (R : Slots) (sid : Nat) (tables : Array (String × Nat)) (f : ByteArray) :
    Except String (List (String × ByteArray)) × Slots :=
  match _root_.Flock.Firewall.Session.commitRoots f with
  | none => (.error "a Commit frame whose roots the loop cannot read", R)
  | some roots => match commit R sid tables roots with
    | .ok R' => (.ok (tables.toList.map fun (t, _) => (t, ((ofSession R' sid t).map (·.root)).getD .empty)), R')
    | .error e => (.error e, R)

/-- A loop in `Except` that finished keeps an invariant of the items it read. -/
theorem forIn_prefix_inv {α σ ε : Type} (I : List α → σ → Prop) {f : α → σ → Except ε (ForInStep σ)}
    (hf : ∀ pre a s t, I pre s → f a s = .ok t → ∃ s', t = .yield s' ∧ I (pre ++ [a]) s') :
    ∀ (l pre : List α) (s v : σ), I pre s → forIn l s f = .ok v → I (pre ++ l) v
  | [], pre, s, v, hs, h => by
    rw [List.forIn_nil] at h
    cases h
    simpa using hs
  | a :: l, pre, s, v, hs, h => by
    rw [List.forIn_cons] at h
    cases ht : f a s with
    | error e => simp [ht, bind, Except.bind] at h
    | ok t =>
      obtain ⟨s', rfl, hs'⟩ := hf pre a s t hs ht
      simp only [ht, bind, Except.bind] at h
      simpa using forIn_prefix_inv I hf l (pre ++ [a]) s' v hs' h

/-- Nothing in the slots `R` is bound to the session `sid`. -/
def Unbound (R : Array Slot) (sid : Nat) : Prop := ∀ s ∈ R, ∀ t, s.bound ≠ some (sid, t)

/-- **What `Level0.commit`'s loop keeps** of its slots `S` once it read the items `pre` (each table's index, the root
the worker sent for it and the table), from the loop's slots `R`: each slot is one of `R`'s with what it was bound to,
or bound to the session `sid` and sampled for it; each of `R`'s is still there but for what it is bound to; where nothing
in `R` was bound to the session, a level 0 bound to it was bound by an item read, at that item's index; and each item
read has a level 0 bound to its name. -/
structure CommitInv (sid : Nat) (R : Array Slot)
    (pre : List (Nat × (String × ByteArray) × (String × Nat)))
    (S : Array Slot) : Prop where
  same : ∀ s ∈ S, ∃ s₀ ∈ R, s.l0 = s₀.l0 ∧ s.owner = s₀.owner ∧ s.table = s₀.table ∧
    (s.bound = s₀.bound ∨ ∃ t, s.bound = some (sid, t) ∧ s.owner = sid)
  kept : ∀ s₀ ∈ R, ∃ s ∈ S, s.l0 = s₀.l0 ∧ s.owner = s₀.owner ∧ s.table = s₀.table
  bound : Unbound R sid → ∀ s ∈ S, ∀ t, s.bound = some (sid, t) →
    ∃ x ∈ pre, x.2.1.1 = t ∧ s.owner = sid ∧ s.table = x.1
  done : ∀ x ∈ pre, ∃ s ∈ S, s.bound = some (sid, x.2.1.1)

/-- **`Level0.commit` keeps `CommitInv`**: a `Commit` of the session `sid` that passes names the tables in order, and
its slots keep `CommitInv` over every table. -/
theorem commit_inv {R R' : Slots} {sid : Nat} {tables : Array (String × Nat)}
    {roots : List (String × ByteArray)} (h : commit R sid tables roots = .ok R') :
    roots.map (·.1) = tables.toList.map (·.1) ∧
      CommitInv sid R.slots ((List.range roots.length).zip (roots.zip tables.toList)) R'.slots := by
  unfold _root_.Flock.Firewall.Level0.commit at h
  rcases hc : (roots.map (·.1) != tables.toList.map (·.1)) with _ | _
  · simp only [hc, Bool.false_eq_true, ↓reduceIte, bind, Except.bind, pure, Except.pure] at h
    refine ⟨by simpa using hc, ?_⟩
    split at h
    · cases h
    · rename_i S hS
      cases h
      have key := forIn_prefix_inv (CommitInv sid R.slots) ?_ _ [] R.slots S
        ⟨fun s hs => ⟨s, hs, rfl, rfl, rfl, .inl rfl⟩, fun s hs => ⟨s, hs, rfl, rfl, rfl⟩,
          fun h0 s hs t hb => absurd hb (h0 s hs t),
          fun _ hx => by cases hx⟩ hS
      · simpa using key
      intro pre a s t hI ht
      obtain ⟨i, ⟨tn, r⟩, tn', m⟩ := a
      simp only at ht
      split at ht
      · simp [throw, throwThe, MonadExceptOf.throw] at ht
      · rename_i k hk
        obtain ⟨hks, hok, -⟩ := Array.findIdx?_eq_some_iff_getElem.mp hk
        simp only [Bool.and_eq_true, beq_iff_eq, Bool.or_eq_true, Option.isNone_iff_eq_none] at hok
        obtain ⟨⟨⟨-, hown⟩, htab⟩, hbd⟩ := hok
        split at ht
        · simp [throw, throwThe, MonadExceptOf.throw] at ht
        · simp only [Except.ok.injEq] at ht
          subst ht
          refine ⟨_, rfl, ?_, ?_, ?_, ?_⟩
          · intro s' hs'
            rcases Array.mem_or_eq_of_mem_setIfInBounds (by simpa using hs') with hs' | rfl
            · exact hI.same s' hs'
            · obtain ⟨s₀, h1, h2, h3, h4, -⟩ := hI.same s[k] (Array.getElem_mem hks)
              rw [getElem!_pos s k hks]
              exact ⟨s₀, h1, h2, h3, h4, .inr ⟨tn, rfl, h3 ▸ hown⟩⟩
          · intro s₀ hs₀
            obtain ⟨s', hs', h1, h2, h3⟩ := hI.kept s₀ hs₀
            obtain ⟨j, hj, rfl⟩ := Array.mem_iff_getElem.mp hs'
            by_cases hjk : k = j
            · subst hjk
              refine ⟨_, Array.mem_setIfInBounds hks, ?_⟩
              rw [getElem!_pos s k hks]
              exact ⟨h1, h2, h3⟩
            · refine ⟨s[j], ?_, h1, h2, h3⟩
              rw [Array.mem_iff_getElem]
              exact ⟨j, by simpa using hj, Array.getElem_setIfInBounds_ne hj hjk⟩
          · intro h0 s' hs' t' hb'
            rcases Array.mem_or_eq_of_mem_setIfInBounds (by simpa using hs') with hs' | rfl
            · obtain ⟨x, hx, h1, h2, h3⟩ := hI.bound h0 s' hs' t' hb'
              exact ⟨x, List.mem_append_left _ hx, h1, h2, h3⟩
            · simp only [Option.some.injEq, Prod.mk.injEq] at hb'
              rw [getElem!_pos s k hks]
              exact ⟨(i, (tn, r), tn', m), List.mem_append_right _ (List.mem_singleton_self _), hb'.2, hown, htab⟩
          · intro x hx
            rcases List.mem_append.mp hx with hx | hx
            · obtain ⟨s', hs', hb'⟩ := hI.done x hx
              obtain ⟨j, hj, rfl⟩ := Array.mem_iff_getElem.mp hs'
              by_cases hjk : k = j
              · subst hjk
                refine ⟨_, Array.mem_setIfInBounds hks, ?_⟩
                rcases hbd with hbd | hbd
                · rw [hbd] at hb'
                  cases hb'
                · rw [hbd] at hb'
                  simp only [Option.some.injEq, Prod.mk.injEq] at hb'
                  rw [← hb'.2]
              · refine ⟨s[j], ?_, hb'⟩
                rw [Array.mem_iff_getElem]
                exact ⟨j, by simpa using hj, Array.getElem_setIfInBounds_ne hj hjk⟩
            · rw [List.mem_singleton] at hx
              subst hx
              exact ⟨_, Array.mem_setIfInBounds hks, rfl⟩
  · simp [hc, throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at h

/-- **The loop drew `ls` for the session `sid`**, table `i`'s at index `i`: every level 0 of the session is `ls`'s at
its table. -/
def SessionDraws (R : Slots) (sid : Nat) (ls : Array Level0) : Prop :=
  ∀ s ∈ R.slots, s.owner = sid → ∃ h : s.table < ls.size, s.l0 = ls[s.table]

/-- **A `Commit` that leaves carries the session's samples' roots**: where the loop drew `ls` for the session `sid` and
nothing in its slots is bound to the session yet, at tables of distinct names, the roots the relay reads at a `Commit`
that passes are each table's name and the root of `ls` at it, whatever roots the worker sent. -/
theorem commitRead_drawn {R : Slots} {sid : Nat} {tables : Array (String × Nat)}
    {f : ByteArray} {ls : Array Level0} {roots : List (String × ByteArray)}
    (hd : SessionDraws R sid ls) (h0 : Unbound R.slots sid)
    (hn : (tables.toList.map (·.1)).Nodup) (h : (commitRead R sid tables f).1 = .ok roots) :
    roots = (tables.toList.map (·.1)).zip (ls.toList.map (·.root)) := by
  unfold commitRead at h
  split at h
  · cases h
  rename_i ws _
  split at h
  · rename_i R' hR
    simp only [Except.ok.injEq] at h
    subst h
    obtain ⟨hnames, hI⟩ := commit_inv hR
    have hlen : ws.length = tables.size := by
      simpa using congrArg List.length hnames
    have hnw : (ws.map (·.1)).Nodup := hnames ▸ hn
    have hat : ∀ j (hj : j < tables.size), ∃ hl : j < ls.size,
        (ofSession R' sid tables[j].1).map (·.root) = some (ls[j]'hl).root := by
      intro j hj
      have hjw : j < ws.length := by omega
      have hname : (ws[j]'hjw).1 = tables[j].1 := by
        have := congrArg (·[j]?) hnames
        simpa [hjw, hj] using this
      let xs := (List.range ws.length).zip (ws.zip tables.toList)
      have hxl : j < xs.length := by simp [xs, hlen, hj]
      have hxj : xs[j] = (j, ws[j], tables[j]) := by simp [xs, List.getElem_zip]
      obtain ⟨s, hs, hb⟩ := hI.done xs[j] (List.getElem_mem hxl)
      rw [hxj] at hb
      simp only at hb
      rw [hname] at hb
      have hsome : (R'.slots.find? (·.bound == some (sid, tables[j].1))).isSome := by
        rw [Array.find?_isSome]
        exact ⟨s, hs, by simp [hb]⟩
      obtain ⟨s', hs'⟩ := Option.isSome_iff_exists.mp hsome
      have hb' : s'.bound = some (sid, tables[j].1) := by simpa using Array.find?_some hs'
      obtain ⟨x, hx, hxt, hown, htab⟩ := hI.bound h0 s' (Array.mem_of_find?_eq_some hs') _ hb'
      obtain ⟨q, hq, rfl⟩ := List.getElem_of_mem hx
      have hql : q < ws.length := Nat.lt_of_lt_of_le hq (by simp only [List.length_zip, List.length_range]; omega)
      have hxq : xs[q] = (q, ws[q], tables[q]'(by omega)) := by simp [xs, List.getElem_zip]
      rw [hxq] at hxt htab
      simp only at hxt htab
      have hqj : q = j := by
        have h1 : (ws.map (·.1))[q]'(by simpa using hql) = (ws.map (·.1))[j]'(by simpa using hjw) := by
          simp only [List.getElem_map]
          rw [hxt, hname]
        exact (hnw.getElem_inj).mp h1
      subst hqj
      obtain ⟨s₀, hs₀, hl0, hown0, htab0, -⟩ := hI.same s' (Array.mem_of_find?_eq_some hs')
      obtain ⟨hlt, hls⟩ := hd s₀ hs₀ (hown0 ▸ hown)
      have htq : s₀.table = q := htab0 ▸ htab
      refine ⟨htq ▸ hlt, ?_⟩
      unfold _root_.Flock.Firewall.Level0.ofSession
      rw [hs']
      simp only [Option.map_some, hl0, hls]
      subst htq
      rfl
    have hle : tables.size ≤ ls.size := by
      rcases Nat.eq_zero_or_pos tables.size with h0' | hpos
      · omega
      · obtain ⟨hl, -⟩ := hat (tables.size - 1) (by omega)
        omega
    apply List.ext_getElem
    · simp [hle]
    · intro j h1 h2
      have hj : j < tables.size := by simpa using h1
      obtain ⟨hl, hr⟩ := hat j hj
      simp [List.getElem_zip, hr]
  · cases h

/-- What `Slots.draw`'s fold pushes is the slots before or `l`'s, each sampled for `sid` at its index from `n`. -/
theorem push_draws {sid : Nat} :
    ∀ (l : List Level0) (S : Array Slot) (n : Nat) (s : Slot),
      s ∈ (l.foldl (fun (S, i) l => (S.push { l0 := l, owner := sid, table := i }, i + 1)) (S, n)).1 →
        s ∈ S ∨ ∃ i, ∃ h : i < l.length, s = { l0 := l[i], owner := sid, table := n + i }
  | [], S, n, s, hs => .inl hs
  | a :: l, S, n, s, hs => by
    rcases push_draws l _ (n + 1) s hs with hs | ⟨i, hi, rfl⟩
    · rcases Array.mem_push.mp hs with hs | rfl
      · exact .inl hs
      · exact .inr ⟨0, by simp, by simp⟩
    · exact .inr ⟨i + 1, by simpa using hi, by simp [Nat.add_assoc, Nat.add_comm 1 i]⟩

/-- `Slots.draw`'s fold keeps the slots before. -/
theorem push_keeps {sid : Nat} :
    ∀ (l : List Level0) (S : Array Slot) (n : Nat) (s : Slot), s ∈ S →
      s ∈ (l.foldl (fun (S, i) l => (S.push { l0 := l, owner := sid, table := i }, i + 1)) (S, n)).1
  | [], _, _, _, hs => hs
  | _ :: l, _, n, s, hs => push_keeps l _ (n + 1) s (Array.mem_push.mpr (.inl hs))

/-- `Slots.draw`'s fold pushes each of `l`'s, sampled for `sid` at its index from `n`. -/
theorem push_has {sid : Nat} :
    ∀ (l : List Level0) (S : Array Slot) (n i : Nat) (h : i < l.length),
      ({ l0 := l[i], owner := sid, table := n + i } : Slot) ∈
        (l.foldl (fun (S, i) l => (S.push { l0 := l, owner := sid, table := i }, i + 1)) (S, n)).1
  | [], _, _, _, h => absurd h (by simp)
  | a :: l, _, n, 0, _ => push_keeps l _ (n + 1) _ (Array.mem_push.mpr (.inr rfl))
  | a :: l, S, n, i + 1, h => by
    have := push_has (sid := sid) l (S.push { l0 := a, owner := sid, table := n }) (n + 1) i (by simpa using h)
    simpa [Nat.add_assoc, Nat.add_comm 1 i] using this

/-- **What `Slots.draw` does**: it admits a session, keeps the sessions and their tables, and its slots are the slots
before and each of `ls`'s, sampled for the session at its index. -/
theorem draw_slots {R R' : Slots} {name : String} {ls : Array Level0} (h : R.draw name ls = .ok R') :
    ∃ sid, R.admit name ls.size = .ok sid ∧ R'.sessions = R.sessions ∧ R'.tables = R.tables ∧
      (∀ s ∈ R'.slots, s ∈ R.slots ∨ ∃ i, ∃ h : i < ls.size, s = { l0 := ls[i], owner := sid, table := i }) ∧
      (∀ s ∈ R.slots, s ∈ R'.slots) ∧
      ∀ i (h : i < ls.size), ({ l0 := ls[i], owner := sid, table := i } : Slot) ∈ R'.slots := by
  unfold _root_.Flock.Firewall.Level0.Slots.draw at h
  cases ha : R.admit name ls.size with
  | error e => simp [ha, bind, Except.bind] at h
  | ok sid =>
    simp only [ha, bind, Except.bind, pure, Except.pure, Except.ok.injEq] at h
    subst h
    refine ⟨sid, rfl, rfl, rfl, fun s hs => ?_, fun s hs => ?_, fun i hi => ?_⟩
    · rcases push_draws (sid := sid) ls.toList R.slots 0 s (by simpa using hs) with hs | ⟨i, hi, rfl⟩
      · exact .inl hs
      · exact .inr ⟨i, by simpa using hi, by simp⟩
    · simpa using push_keeps (sid := sid) ls.toList R.slots 0 s hs
    · have := push_has (sid := sid) ls.toList R.slots 0 i (by simpa using hi)
      simp only [Array.getElem_toList, Nat.zero_add, Array.foldl_toList] at this
      exact this

/-- **A session `Slots.admit` lets sample** has no level 0 sampled for it yet, is a live session's, and samples as many as
its count of tables. -/
theorem admit_fresh {R : Slots} {name : String} {k sid : Nat} (h : R.admit name k = .ok sid) :
    (∀ s ∈ R.slots, s.owner ≠ sid) ∧ (∃ x ∈ R.sessions, x.2 = sid) ∧
      k = ((R.tables.find? (·.1 == sid)).map (·.2)).getD 0 := by
  unfold _root_.Flock.Firewall.Level0.Slots.admit at h
  split at h
  · cases h
  · rename_i x hf
    split at h
    · cases h
    · rename_i hd
      dsimp only at h
      split at h
      · cases h
      · rename_i hk
        cases h
        refine ⟨fun s hs hown => ?_, ⟨_, Array.mem_of_find?_eq_some hf, rfl⟩, by simpa using hk⟩
        simp only [_root_.Flock.Firewall.Level0.Slots.drawn, Bool.not_eq_true] at hd
        exact absurd (by simp [hown]) (Array.any_eq_false'.mp hd s hs)

/-- A level 0 bound to a session was sampled for it. -/
def BoundOwned (R : Array Slot) : Prop := ∀ s ∈ R, ∀ sid t, s.bound = some (sid, t) → s.owner = sid

/-- Each live session has its count of tables. -/
def Named (R : Slots) : Prop := ∀ x ∈ R.sessions, ∃ y ∈ R.tables, y.1 = x.2

/-- **The loop's level-0 slots**, as its four updates of them leave them, from none (`Session.run`'s `l0s`): a session
named at its `Hello` (`Session.relay`), a level-0 request's sample (`Session.level0`), a `Commit` (`Session.relay`) and a
session's end (`Session.run`). An update that fails leaves them as they were. -/
inductive SlotsReach : Slots → Prop
  | empty : SlotsReach {}
  | name {R : Slots} (n : String) (sid k : Nat) : SlotsReach R → SlotsReach (R.name n sid k)
  | draw {R R' : Slots} {n : String} {ls : Array Level0} : SlotsReach R → R.draw n ls = .ok R' → SlotsReach R'
  | commit {R R' : Slots} {sid : Nat} {tables : Array (String × Nat)} {roots : List (String × ByteArray)} :
      SlotsReach R → commit R sid tables roots = .ok R' → SlotsReach R'
  | close {R : Slots} (sid : Nat) : SlotsReach R → SlotsReach (R.close sid)

/-- `Level0.commit` keeps the sessions and their tables. -/
theorem commit_sessions {R R' : Slots} {sid : Nat} {tables : Array (String × Nat)}
    {roots : List (String × ByteArray)} (h : commit R sid tables roots = .ok R') :
    R'.sessions = R.sessions ∧ R'.tables = R.tables := by
  unfold _root_.Flock.Firewall.Level0.commit at h
  rcases hc : (roots.map (·.1) != tables.toList.map (·.1)) with _ | _
  · simp only [hc, Bool.false_eq_true, ↓reduceIte, bind, Except.bind, pure, Except.pure] at h
    split at h
    · cases h
    · cases h
      exact ⟨rfl, rfl⟩
  · simp [hc, throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at h

/-- In every state the loop's slots reach, a level 0 bound to a session was sampled for it, and each live session has its
count of tables. -/
theorem slotsReach_inv {R : Slots} (h : SlotsReach R) : BoundOwned R.slots ∧ Named R := by
  induction h with
  | empty => refine ⟨?_, ?_⟩ <;> intro x hx <;> simp at hx
  | name n sid k _ ih =>
    refine ⟨ih.1, fun x hx => ?_⟩
    rcases Array.mem_push.mp hx with hx | rfl
    · obtain ⟨y, hy, hy1⟩ := ih.2 x hx
      exact ⟨y, Array.mem_push.mpr (.inl hy), hy1⟩
    · exact ⟨(sid, k), Array.mem_push.mpr (.inr rfl), rfl⟩
  | draw _ hd ih =>
    obtain ⟨sid, -, hss, htt, hnew, -, -⟩ := draw_slots hd
    refine ⟨fun s hs sid' t hb => ?_, fun x hx => ?_⟩
    · rcases hnew s hs with hs | ⟨i, hi, rfl⟩
      · exact ih.1 s hs sid' t hb
      · cases hb
    · rw [hss] at hx
      rw [htt]
      exact ih.2 x hx
  | commit _ hc ih =>
    obtain ⟨-, hI⟩ := commit_inv hc
    obtain ⟨hss, htt⟩ := commit_sessions hc
    refine ⟨fun s hs sid' t hb => ?_, fun x hx => ?_⟩
    · obtain ⟨s₀, hs₀, -, hown, -, hbd | ⟨t', hb', hown'⟩⟩ := hI.same s hs
      · rw [hown]
        exact ih.1 s₀ hs₀ sid' t (hbd ▸ hb)
      · rw [hb'] at hb
        simp only [Option.some.injEq, Prod.mk.injEq] at hb
        rw [← hb.1]
        exact hown'
    · rw [hss] at hx
      rw [htt]
      exact ih.2 x hx
  | close sid _ ih =>
    refine ⟨fun s hs sid' t hb => ih.1 s (Array.mem_filter.mp hs).1 sid' t hb, fun x hx => ?_⟩
    obtain ⟨hx, hx2⟩ := Array.mem_filter.mp hx
    obtain ⟨y, hy, hy1⟩ := ih.2 x hx
    exact ⟨y, Array.mem_filter.mpr ⟨hy, by simpa [hy1] using hx2⟩, hy1⟩

/-- **The loop's level-0 slots after its sample of `ls` for the session `sid`**, through what the loop does before the
session's `Commit`: naming a session, any level-0 request's sample (another session's, or this one's again), another
session's `Commit` and another session's end. -/
inductive AfterDraw (sid : Nat) (ls : Array Level0) : Slots → Prop
  | draw {R R' : Slots} {n : String} : SlotsReach R → R.admit n ls.size = .ok sid → R.draw n ls = .ok R' →
      AfterDraw sid ls R'
  | name {R : Slots} (n : String) (sid' k : Nat) : AfterDraw sid ls R → AfterDraw sid ls (R.name n sid' k)
  | other {R R' : Slots} {n : String} {ls' : Array Level0} : AfterDraw sid ls R → R.draw n ls' = .ok R' →
      AfterDraw sid ls R'
  | commit {R R' : Slots} {sid' : Nat} {tables : Array (String × Nat)} {roots : List (String × ByteArray)} :
      sid' ≠ sid → AfterDraw sid ls R → commit R sid' tables roots = .ok R' → AfterDraw sid ls R'
  | close {R : Slots} (sid' : Nat) : sid' ≠ sid → AfterDraw sid ls R → AfterDraw sid ls (R.close sid')

/-- **What holds of the slots after the session's sample**: the session's level 0 are `ls`'s, nothing is bound to it, a
level 0 bound to a session was sampled for it, each live session has its count of tables, the session has a level 0 where
it has tables, and its count of tables is `ls`'s. -/
structure DrawInv (sid : Nat) (ls : Array Level0) (R : Slots) : Prop where
  draws : SessionDraws R sid ls
  unbound : Unbound R.slots sid
  owned : BoundOwned R.slots
  named : Named R
  present : 0 < ls.size → ∃ s ∈ R.slots, s.owner = sid
  count : (R.tables.find? (·.1 == sid)).map (·.2) = some ls.size

/-- **`AfterDraw` keeps `DrawInv`**: a second sample for the session is refused or samples nothing. -/
theorem afterDraw_inv {sid : Nat} {ls : Array Level0} {R : Slots} (h : AfterDraw sid ls R) :
    DrawInv sid ls R := by
  induction h with
  | draw hr ha hd =>
    rename_i R R' n
    obtain ⟨ho, hnm⟩ := slotsReach_inv hr
    obtain ⟨sid', ha', hss, htt, hnew, hold, hhas⟩ := draw_slots hd
    rw [ha] at ha'
    cases ha'
    obtain ⟨hf, ⟨x, hx, hx2⟩, hk⟩ := admit_fresh ha
    refine ⟨fun s hs hown => ?_, fun s hs t hb => ?_, fun s hs sid' t hb => ?_, ?_, fun hpos => ?_, ?_⟩
    · rcases hnew s hs with hs | ⟨i, hi, rfl⟩
      · exact absurd hown (hf s hs)
      · exact ⟨by simpa using hi, by simp⟩
    · rcases hnew s hs with hs | ⟨i, hi, rfl⟩
      · exact hf s hs (ho s hs sid t hb)
      · cases hb
    · rcases hnew s hs with hs | ⟨i, hi, rfl⟩
      · exact ho s hs sid' t hb
      · cases hb
    · intro y hy
      rw [hss] at hy
      rw [htt]
      exact hnm y hy
    · exact ⟨_, hhas 0 hpos, rfl⟩
    · rw [htt]
      obtain ⟨y, hy, hy1⟩ := hnm x hx
      have hsome : (R.tables.find? (·.1 == sid)).isSome := by
        rw [Array.find?_isSome]
        exact ⟨y, hy, by simp [hy1, hx2]⟩
      obtain ⟨z, hz⟩ := Option.isSome_iff_exists.mp hsome
      rw [hz] at hk ⊢
      simp only [Option.map_some, Option.getD_some] at hk ⊢
      rw [hk]
  | name n sid' k _ ih =>
    rename_i R _
    refine ⟨ih.draws, ih.unbound, ih.owned, fun x hx => ?_, ih.present, ?_⟩
    · rcases Array.mem_push.mp hx with hx | rfl
      · obtain ⟨y, hy, hy1⟩ := ih.named x hx
        exact ⟨y, Array.mem_push.mpr (.inl hy), hy1⟩
      · exact ⟨(sid', k), Array.mem_push.mpr (.inr rfl), rfl⟩
    · show ((R.tables.push (sid', k)).find? (·.1 == sid)).map (·.2) = some ls.size
      rw [Array.find?_push]
      have hc := ih.count
      cases hz : R.tables.find? (·.1 == sid) with
      | none =>
        rw [hz] at hc
        cases hc
      | some z =>
        rw [hz] at hc
        simpa using hc
  | other _ hd ih =>
    rename_i R R' n ls' _
    obtain ⟨sid', ha, hss, htt, hnew, hold, -⟩ := draw_slots hd
    obtain ⟨hf, -, hk⟩ := admit_fresh ha
    have hne : ∀ i, i < ls'.size → sid' ≠ sid := by
      intro i hi heq
      rcases Nat.eq_zero_or_pos ls.size with h0 | h0
      · have hc := ih.count
        rw [h0, ← heq] at hc
        rw [hc] at hk
        simp only [Option.getD_some] at hk
        omega
      · obtain ⟨s, hs, hown⟩ := ih.present h0
        exact hf s hs (heq ▸ hown)
    refine ⟨fun s hs hown => ?_, fun s hs t hb => ?_, fun s hs sid'' t hb => ?_, ?_, fun hpos => ?_, ?_⟩
    · rcases hnew s hs with hs | ⟨i, hi, rfl⟩
      · exact ih.draws s hs hown
      · exact absurd hown (hne i hi)
    · rcases hnew s hs with hs | ⟨i, hi, rfl⟩
      · exact ih.unbound s hs t hb
      · cases hb
    · rcases hnew s hs with hs | ⟨i, hi, rfl⟩
      · exact ih.owned s hs sid'' t hb
      · cases hb
    · intro y hy
      rw [hss] at hy
      rw [htt]
      exact ih.named y hy
    · obtain ⟨s, hs, hown⟩ := ih.present hpos
      exact ⟨s, hold s hs, hown⟩
    · rw [htt]
      exact ih.count
  | commit hne _ hc ih =>
    rename_i R R' sid' tables roots _
    obtain ⟨-, hI⟩ := commit_inv hc
    obtain ⟨hss, htt⟩ := commit_sessions hc
    refine ⟨fun s hs hown => ?_, fun s hs t hb => ?_, fun s hs sid'' t hb => ?_, ?_, fun hpos => ?_, ?_⟩
    · obtain ⟨s₀, hs₀, hl0, hown0, htab0, -⟩ := hI.same s hs
      obtain ⟨hlt, hls⟩ := ih.draws s₀ hs₀ (hown0 ▸ hown)
      refine ⟨htab0 ▸ hlt, ?_⟩
      rw [hl0, hls]
      simp only [htab0]
    · obtain ⟨s₀, hs₀, -, -, -, hbd | ⟨t', hb', -⟩⟩ := hI.same s hs
      · exact ih.unbound s₀ hs₀ t (hbd ▸ hb)
      · rw [hb'] at hb
        simp only [Option.some.injEq, Prod.mk.injEq] at hb
        exact hne hb.1
    · obtain ⟨s₀, hs₀, -, hown, -, hbd | ⟨t', hb', hown'⟩⟩ := hI.same s hs
      · rw [hown]
        exact ih.owned s₀ hs₀ sid'' t (hbd ▸ hb)
      · rw [hb'] at hb
        simp only [Option.some.injEq, Prod.mk.injEq] at hb
        rw [← hb.1]
        exact hown'
    · intro y hy
      rw [hss] at hy
      rw [htt]
      exact ih.named y hy
    · obtain ⟨s₀, hs₀, hown⟩ := ih.present hpos
      obtain ⟨s, hs, -, hown', -⟩ := hI.kept s₀ hs₀
      exact ⟨s, hs, hown'.trans hown⟩
    · rw [htt]
      exact ih.count
  | close sid' hne _ ih =>
    rename_i R _
    refine ⟨fun s hs hown => ih.draws s (Array.mem_filter.mp hs).1 hown,
      fun s hs t hb => ih.unbound s (Array.mem_filter.mp hs).1 t hb,
      fun s hs sid'' t hb => ih.owned s (Array.mem_filter.mp hs).1 sid'' t hb, fun x hx => ?_, fun hpos => ?_, ?_⟩
    · obtain ⟨hx, hx2⟩ := Array.mem_filter.mp hx
      obtain ⟨y, hy, hy1⟩ := ih.named x hx
      exact ⟨y, Array.mem_filter.mpr ⟨hy, by simpa [hy1] using hx2⟩, hy1⟩
    · obtain ⟨s, hs, hown⟩ := ih.present hpos
      exact ⟨s, Array.mem_filter.mpr ⟨hs, by simpa [hown] using Ne.symm hne⟩, hown⟩
    · show ((R.tables.filter (·.1 != sid')).find? (·.1 == sid)).map (·.2) = some ls.size
      rw [Array.find?_filter]
      have hp : (fun a : Nat × Nat => decide ((a.1 != sid') = true ∧ (a.1 == sid) = true)) =
          (fun a => a.1 == sid) := by
        funext a
        by_cases h : a.1 = sid
        · simp [h, Ne.symm hne]
        · simp [h]
      rw [hp]
      exact ih.count

/-- The first of each item's two in a `flatMap` is a sublist of it. -/
theorem map_sublist_flatMap {α β : Type} (f g : α → β) :
    ∀ l : List α, (l.map f).Sublist (l.flatMap fun a => [f a, g a])
  | [] => List.Sublist.slnil
  | a :: l => by
    simp only [List.map_cons, List.flatMap_cons, List.cons_append, List.nil_append]
    exact (List.Sublist.cons _ (map_sublist_flatMap f g l)).cons_cons _

/-- **The setup's tables where its coin tree's streams are the session's** (`streamsOk`): one statement each, and of
distinct names, since each table's first rep stream, `"{table}/rep0"`, is one of the tree's distinct streams. -/
theorem streamsOk_tables {spec : SessionSpec} {sts : Array Stmt} {ct : _root_.Flock.CoinTree.Spec}
    (h : _root_.Flock.Firewall.Session.streamsOk spec sts ct = true) :
    spec.tables.size = sts.size ∧ spec.tables.toList.Nodup := by
  unfold _root_.Flock.Firewall.Session.streamsOk at h
  simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at h
  obtain ⟨⟨hs, hnd⟩, hsz⟩ := h
  have hrep : spec.repStreams.toList.map (·.2.2) =
      (List.range spec.tables.size).flatMap fun i =>
        [s!"{spec.tables[i]!}/rep{0}", s!"{spec.tables[i]!}/rep{1}"] := by
    simp [_root_.Flock.SessionSpec.repStreams, Array.toList_flatMap, List.map_flatMap]
  have hsize : spec.repStreams.size = 2 * spec.tables.size := by
    have := congrArg List.length hrep
    simp only [List.length_map, Array.length_toList, List.length_flatMap, List.length_cons, List.length_nil] at this
    rw [this, List.map_const', List.sum_replicate_nat, List.length_range, Nat.mul_comm]
  refine ⟨by omega, ?_⟩
  rw [hs] at hnd
  have h1 := (List.nodup_cons.mp hnd).2
  rw [hrep] at h1
  have h2 := (map_sublist_flatMap (fun i => s!"{spec.tables[i]!}/rep{0}") (fun i => s!"{spec.tables[i]!}/rep{1}")
    (List.range spec.tables.size)).nodup h1
  have h3 : ((List.range spec.tables.size).map fun i => spec.tables[i]!).Nodup := by
    rw [show (fun i => s!"{spec.tables[i]!}/rep{0}") =
      (fun t : String => s!"{t}/rep{0}") ∘ (fun i => spec.tables[i]!) from rfl, ← List.map_map] at h2
    exact List.Pairwise.of_map _ (fun a b h e => h (by rw [e])) h2
  have h4 : (List.range spec.tables.size).map (fun i => spec.tables[i]!) = spec.tables.toList := by
    apply List.ext_getElem
    · simp
    · intro j h1 h2
      simp at h1
      simp [getElem!_pos spec.tables j h1]
  rwa [h4] at h3

/-- **At the model's link, `EnvOk` at the loop's `Commit`**: where what the relay read at the session `sid`'s `Commit`
is `commitRead` against the loop's slots after its sample of `ls` (`AfterDraw`), at the session's tables (the setup's,
each with its statement's `m`), and the setup's coin tree's streams are the session's (`streamsOk`), `EnvOk` holds at
the link whose roots are `ls`'s. -/
theorem envOk_commit_after {spec : SessionSpec} {sts : Array Stmt} {seen : List String}
    {ct : _root_.Flock.CoinTree.Spec} {e : Env} {R : Slots} {sid : Nat} {f : ByteArray} {ls : Array Level0}
    {publics : List ByteArray}
    (hread : e.commit = (commitRead R sid (spec.tables.zip (sts.map (·.m))) f).1)
    (hok : _root_.Flock.Firewall.Session.streamsOk spec sts ct = true) (ha : AfterDraw sid ls R)
    (hr : ls.size < 2 ^ 64) (hp : publics.length < 2 ^ 64) (hrb : ∀ l ∈ ls, l.root.size < 2 ^ 64)
    (hpb : ∀ b ∈ publics, b.size < 2 ^ 64) :
    EnvOk spec seen .commit e (0 :: FlockDeploy.linkBody (ls.toList.map (·.root)) publics) := by
  obtain ⟨hsz, hn⟩ := streamsOk_tables hok
  have hnames : (spec.tables.zip (sts.map (·.m))).toList.map (·.1) = spec.tables.toList := by
    rw [Array.toList_zip]
    exact List.map_fst_zip (by simpa using hsz.le)
  have hI := afterDraw_inv ha
  refine envOk_commit (by simpa using hr) hp ?_ hpb ?_
  · intro b hb
    obtain ⟨l, hl, rfl⟩ := List.mem_map.mp hb
    exact hrb l (by simpa using hl)
  · intro roots hc
    rw [hread] at hc
    have := commitRead_drawn hI.draws hI.unbound (by rw [hnames]; exact hn) hc
    rw [this, hnames]

end Claims.ZKServer.Relay
