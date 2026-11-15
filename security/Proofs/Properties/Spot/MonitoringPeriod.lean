module

public import Properties.Spot.MonitoringPeriod
public import Proofs.Properties.Challenger.ChallengePhase
import Proofs.Flock.Soundness.Discharge.Exec.Setup
import all Init.Data.Array.Basic

/-!
# Proofs of the properties of SPOT's period (`Properties.Spot.MonitoringPeriod`)

The period's check is monadic. What each of its functions read where it returns is a Hoare triple proved by `mvcgen`
and read back at an `.ok` result (`read_ok`, `windowsIn_ok`, `caseAt_ok`, `executionPhaseAt_raw`, `run_ok`); the properties
are then facts about arrays.

* **The ledger's half** (`ledgerMeans`). `read` keeps, of the windows `Challenger.checked` returns, those whose open
  is a v2 entry of the lane in the monitoring period's run (`sessionAt`), so a session is returned exactly for a
  root the lane committed (`Returned.session_of_win`, `Returned.win_of_session`). `windowsIn` splits the sessions by
  whether their layout holds the sample's family (`partition_eq`) and makes one window of each that does, in order
  (`Returned.win`). The checked ledger's windows share no root and no session (`LedgerInv.distinct`), so neither do
  the sessions, nor the windows and the others together (`Returned.nodup`).
* **The audits' half** (`acceptMeans`). `executionPhaseAt` either leaves an audit undecided or adds one window with one
  audit and that audit's record (`executionPhaseAt_step`). Along `run`'s fold over the windows and their evidence, while no
  audit is undecided, window `i` passed with one audit, made at the records of the windows before it (`FoldGood`,
  `foldGood_step`, `foldlM_prefix`).
* **Whose each session is** (`registeredIn_ok`). Where `registeredIn` returns, every window is in a period that
  commits this registration's circuit, every other session of the lane is in a period that commits another, and it
  returns those, each with its circuit (`othersOf`).
* **Each root committed under this registration once** (`committedOnce`): the three together. A root the lane
  committed is a window or an other session, so a period that commits a circuit lists its session; under this
  registration's circuit it is a window, whose place is its root's alone (`index_eq_of_nodup`).
* **The others** (`othersElsewhere`): the published others are `othersOf` of the ledger's half's, so each is a root
  the lane committed under another circuit, each such root is one of them, and none is there twice
  (`othersOf_nodup`).
* **The decoder** (`decodedMeans`): `AuditParams.of` reads the ledger's key from the key's JSON it keeps
  (`paramsOf_key`) and refuses an `audit` that names a key a window sets or leaves unset (`none_of_keys`).
* **The session audited** (`auditedChallengePhase`): `auditedIn` reads the first of the ledger's windows on the window's
  root (`auditedIn_root`), and the ledger's windows share no root (`wins_inj`), so it is the window's own session.
-/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

set_option mvcgen.warning false

open private strOf from Spot.ChallengerSession

namespace Properties.Spot.MonitoringPeriod

open Lean (Json)
open Std.Do
open _root_.Spot _root_.Spot.Challenger _root_.Spot.MonitoringPeriod
open Properties.Pouw.LiveSalt.Ledger (bind_ok of_triple throw_spec ok_spec checked_ok verifyLedger_inv getD_root)
open Properties.Challenger.ChallengePhase (none_of_forall)
open FlockSoundness.Discharge.Exec (array_mapM_ok_getElem)

/-! ## Lists, arrays and options -/

theorem ite_some {α : Type} {c : Prop} [Decidable c] {a s : α} (h : (if c then some a else none) = some s) :
    c ∧ a = s := by
  by_cases hc : c
  · rw [ite_eq_left hc] at h; exact ⟨hc, Option.some.inj h⟩
  · rw [ite_eq_right hc] at h; cases h

theorem none_of_forall₂ {α β : Type} {o : Option (α × β)} (h : ∀ a b, o = some (a, b) → False) :
    o = none := by
  cases o with
  | none => rfl
  | some x => exact absurd rfl (h x.1 x.2)

theorem bool_of_not_not {b : Bool} (h : ¬ (!b) = true) : b = true := by
  cases b
  · exact absurd rfl h
  · rfl

theorem bool_of_not_and_not {a b : Bool} (h : ¬ (a && !b) = true) (ha : a = true) : b = true := by
  subst ha
  cases b
  · exact absurd rfl h
  · rfl

/-- The four tests of a session `read` keeps, read off its condition. -/
theorem keep_of {a b c d : String} {m n k : Nat}
    (h : (a == b && c == d && decide (m ≤ n) && decide (n < k)) = true) : a = b ∧ c = d ∧ m ≤ n ∧ n < k := by
  simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at h
  exact ⟨h.1.1.1, h.1.1.2, h.1.2, h.2⟩

theorem keep_to {a b c d : String} {m n k : Nat} (h1 : a = b) (h2 : c = d) (h3 : m ≤ n) (h4 : n < k) :
    (a == b && c == d && decide (m ≤ n) && decide (n < k)) = true := by
  simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq]
  exact ⟨⟨⟨h1, h2⟩, h3⟩, h4⟩

theorem pairwise_of_distinct {α : Type} {xs : Array α} {R : α → α → Prop}
    (h : ∀ i j (hi : i < xs.size) (hj : j < xs.size), i ≠ j → R xs[i] xs[j]) : xs.toList.Pairwise R := by
  rw [List.pairwise_iff_getElem]
  intro i j hi hj hij
  exact h i j hi hj (Nat.ne_of_lt hij)

/-- A list whose keys are distinct, split by a test: the keys of the two halves, one after the other, are
distinct. -/
theorem nodup_filter_append {α β : Type} (k : α → β) (f : α → Bool) :
    ∀ {l : List α}, (l.map k).Nodup → ((l.filter f).map k ++ (l.filter fun a => !f a).map k).Nodup
  | [], _ => by simp
  | a :: l, hn => by
    rw [List.map_cons, List.nodup_cons] at hn
    have ih := nodup_filter_append k f hn.2
    have hsub : ∀ b, b ∈ (l.filter f).map k ++ (l.filter fun a => !f a).map k → b ∈ l.map k := by
      intro b hb
      rcases List.mem_append.1 hb with hb | hb
      · obtain ⟨c, hc, rfl⟩ := List.mem_map.1 hb
        exact List.mem_map.2 ⟨c, (List.mem_filter.1 hc).1, rfl⟩
      · obtain ⟨c, hc, rfl⟩ := List.mem_map.1 hb
        exact List.mem_map.2 ⟨c, (List.mem_filter.1 hc).1, rfl⟩
    cases hfa : f a with
    | true =>
      rw [List.filter_cons_of_pos (p := f) hfa, List.filter_cons_of_neg (p := fun a => !f a) (by simp [hfa]),
        List.map_cons, List.cons_append]
      exact List.nodup_cons.2 ⟨fun hb => hn.1 (hsub _ hb), ih⟩
    | false =>
      rw [List.filter_cons_of_neg (p := f) (by simp [hfa]),
        List.filter_cons_of_pos (p := fun a => !f a) (by simp [hfa]), List.map_cons]
      exact (List.perm_middle.nodup_iff).2 (List.nodup_cons.2 ⟨fun hb => hn.1 (hsub _ hb), ih⟩)

theorem map_eq_of_getElem {α β γ : Type} {xs : Array α} {ys : Array β} (f : α → γ) (g : β → γ)
    (hsz : xs.size = ys.size) (h : ∀ i (h1 : i < xs.size) (h2 : i < ys.size), f xs[i] = g ys[i]) :
    xs.toList.map f = ys.toList.map g := by
  rw [← Array.toList_map, ← Array.toList_map]
  refine congrArg Array.toList (Array.ext (by rw [Array.size_map, Array.size_map, hsz]) fun i h1 h2 => ?_)
  rw [Array.getElem_map, Array.getElem_map]
  exact h i (by rw [Array.size_map] at h1; exact h1) (by rw [Array.size_map] at h2; exact h2)

theorem getElem_of_map_eq {α β : Type} {xs : Array α} {ys : Array β} {f : α → β} (h : xs.map f = ys) (j : Nat)
    (hj : j < xs.size) (hj' : j < ys.size) : f xs[j] = ys[j] := by
  subst h
  exact (Array.getElem_map f hj').symm

/-- An array whose keys are distinct holds each key at one place. -/
theorem index_eq_of_nodup {α β : Type} {xs : Array α} {f : α → β} (h : (xs.toList.map f).Nodup) {i j : Nat}
    (hi : i < xs.size) (hj : j < xs.size) (he : f xs[i] = f xs[j]) : i = j := by
  refine List.nodup_iff_eq_of_getElem_eq.1 h i j (by simpa using hi) (by simpa using hj) ?_
  simpa using he

theorem getElem?_snoc {α : Type} {l : List α} {x y : α} {i : Nat} (h : (l ++ [x])[i]? = some y) :
    l[i]? = some y ∨ (i = l.length ∧ x = y) := by
  by_cases hi : i < l.length
  · rw [List.getElem?_append_left hi] at h
    exact Or.inl h
  · rw [List.getElem?_append_right (Nat.le_of_not_lt hi), List.getElem?_singleton] at h
    by_cases h0 : i - l.length = 0
    · rw [ite_eq_left h0] at h
      exact Or.inr ⟨by omega, Option.some.inj h⟩
    · rw [ite_eq_right h0] at h
      cases h

theorem zip_getElem? {α β : Type} {xs : Array α} {ys : Array β} {i : Nat} (hx : i < xs.size) (hy : i < ys.size) :
    (xs.zip ys).toList[i]? = some (xs[i], ys[i]) := by
  rw [Array.toList_zip]
  refine List.getElem?_zip_eq_some.2 ⟨?_, ?_⟩
  · rw [Array.getElem?_toList]
    exact Array.getElem?_eq_getElem hx
  · rw [Array.getElem?_toList]
    exact Array.getElem?_eq_getElem hy

/-- A fold in `Except` that returns keeps, of the prefix it has read, what each of its steps keeps. -/
theorem foldlM_prefix {α σ ε : Type} (f : σ → α → Except ε σ) (P : List α → σ → Prop)
    (hstep : ∀ pre x s t, P pre s → f s x = .ok t → P (pre ++ [x]) t) :
    ∀ (l pre : List α) (s r : σ), P pre s → l.foldlM f s = .ok r → P (pre ++ l) r := by
  intro l
  induction l with
  | nil =>
    intro pre s r hP h
    simp only [List.foldlM_nil, pure, Except.pure, Except.ok.injEq] at h
    subst h
    simpa using hP
  | cons x xs ih =>
    intro pre s r hP h
    rw [List.foldlM_cons] at h
    obtain ⟨t, ht, h⟩ := bind_ok.1 h
    have := ih (pre ++ [x]) t r (hstep pre x s t hP ht) h
    simpa using this

/-- `Array.partition` splits an array as two filters do. -/
theorem partition_toList {α : Type} (p : α → Bool) (xs : Array α) :
    (xs.partition p).1.toList = xs.toList.filter p ∧
      (xs.partition p).2.toList = xs.toList.filter fun a => !p a := by
  generalize h : xs.partition p = r
  apply Id.of_wp_run_eq h
  mvcgen invariants
  · ⇓⟨cur, bs, cs⟩ => ⌜bs.toList = cur.prefix.filter p ∧ cs.toList = cur.prefix.filter fun a => !p a⌝
  all_goals simp_all
  all_goals assumption

theorem partition_eq {α : Type} (p : α → Bool) (xs : Array α) :
    xs.partition p = (xs.filter p, xs.filter fun a => !p a) := by
  obtain ⟨h1, h2⟩ := partition_toList p xs
  generalize xs.partition p = r at h1 h2 ⊢
  obtain ⟨bs, cs⟩ := r
  exact congr (congrArg Prod.mk (Array.ext' (h1.trans Array.toList_filter.symm)))
    (Array.ext' (h2.trans Array.toList_filter.symm))

/-- The checked ledger's windows are told apart by their root, and by their session. -/
theorem wins_inj {wins : Array Win}
    (hd : ∀ i j (hi : i < wins.size) (hj : j < wins.size), i ≠ j →
      wins[i].root ≠ wins[j].root ∧ wins[i].opened.session ≠ wins[j].opened.session)
    {x y : Win} (hx : x ∈ wins) (hy : y ∈ wins) (h : x.root = y.root ∨ x.opened.session = y.opened.session) :
    x = y := by
  obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.1 hx
  obtain ⟨j, hj, rfl⟩ := Array.mem_iff_getElem.1 hy
  by_cases hij : i = j
  · subst hij
    rfl
  · rcases h with h | h
    · exact absurd h (hd i j hi hj hij).1
    · exact absurd h (hd i j hi hj hij).2

/-! ## What `read` returns -/

/-- `read`'s session of the ledger's window `w`: one where `w`'s open is a v2 entry of the lane numbered from `start`
up to `stop`, with `reserves` the lane's reserves and `periods` the ledger's periods. -/
def sessionAt (p : Params) (periods : Array MonitoringPeriod) (reserves : Array String) (start stop : Nat)
    (w : Win) : Option Session :=
  let e := w.opened
  if e.schema == LEDGER_SCHEMA && laneOf e.body == p.lane && start ≤ e.seq && e.seq < stop then
    some ({ session := e.session, root := w.root, families := getD e.body "families",
            withheld := (match getD e.body "withheld" with | .bool true => true | _ => false),
            place := (reserves.findIdx? (· == e.session)).getD reserves.size,
            period := (periods.find? (·.windows.contains e.session)).map (·.name) } : Session)
  else none

theorem sessionAt_some {p : Params} {periods : Array MonitoringPeriod} {reserves : Array String} {start stop : Nat}
    {w : Win} {s : Session} (h : sessionAt p periods reserves start stop w = some s) :
    (w.opened.schema = LEDGER_SCHEMA ∧ laneOf w.opened.body = p.lane ∧ start ≤ w.opened.seq ∧
        w.opened.seq < stop) ∧
      s.session = w.opened.session ∧ s.root = w.root ∧ s.families = getD w.opened.body "families" ∧
      s.period = (periods.find? (·.windows.contains w.opened.session)).map (·.name) := by
  unfold sessionAt at h
  obtain ⟨hc, rfl⟩ := ite_some h
  exact ⟨keep_of hc, rfl, rfl, rfl, rfl⟩

theorem sessionAt_of {p : Params} {periods : Array MonitoringPeriod} {reserves : Array String} {start stop : Nat}
    {w : Win} (h1 : w.opened.schema = LEDGER_SCHEMA) (h2 : laneOf w.opened.body = p.lane)
    (h3 : start ≤ w.opened.seq) (h4 : w.opened.seq < stop) :
    ∃ s, sessionAt p periods reserves start stop w = some s := by
  unfold sessionAt
  exact ⟨_, ite_eq_left (keep_to h1 h2 h3 h4)⟩

-- `mvcgen` splits `read` at the ledger's check and at the monitoring period's end
set_option maxHeartbeats 4000000 in
/-- What `read` read where it returns: the ledger passes `checked` under the key, one of its entries is stamped at or
after the monitoring period's end, and the sessions are `sessionAt`'s of the ledger's windows, in order. -/
theorem read_ok {p : Params} {ledger : Json} {r : Read} (h : read p ledger = .ok r) :
    checked ledger p.key = .ok (r.wins, r.periods) ∧
      ∃ stop, reaches (entriesOf ledger) p.stop = some stop ∧
        r.sessions = r.wins.filterMap (sessionAt p r.periods (reservesOf (entriesOf ledger) p.lane)
          ((reaches (entriesOf ledger) p.start).getD stop) stop) := by
  refine of_triple (Q := fun r => checked ledger p.key = .ok (r.wins, r.periods) ∧
      ∃ stop, reaches (entriesOf ledger) p.stop = some stop ∧
        r.sessions = r.wins.filterMap (sessionAt p r.periods (reservesOf (entriesOf ledger) p.lane)
          ((reaches (entriesOf ledger) p.start).getD stop) stop)) ?_ h
  clear h
  mvcgen [_root_.Spot.MonitoringPeriod.read, -Spec.throw_Except, throw_spec]
  refine ⟨by assumption, _, by assumption, ?_⟩
  rfl

theorem read_spec (p : Params) (ledger : Json) :
    ⦃⌜True⌝⦄ read p ledger ⦃⇓? r => ⌜read p ledger = .ok r⌝⦄ := ok_spec _

/-! ## What `windowsIn` returns -/

/-- The ledger's periods that list one of the sessions `sampled`. -/
def heldOf (r : Read) (sampled : Array Session) : Array MonitoringPeriod :=
  r.periods.filter fun mp => sampled.any (·.period == some mp.name)

/-- Whether the ledger opens the session `ses`. -/
def openedIn (r : Read) (ses : String) : Bool :=
  r.wins.any (·.opened.session == ses)

/-- What `windowsIn` checked where it returns the windows `ws`, of the sessions `sampled` whose layout holds the
sample's family: there is one; each has a period, and its window is made of it, at its place; each such period is
closed; every session such a period lists was opened and is one of them; and each has ended. -/
structure WindowsIn (p : Params) (r : Read) (ws : Windows) (sampled : Array Session) : Prop where
  parts : r.sessions.partition (fun s => holds p.label s.families) = (sampled, ws.others)
  nonempty : ¬ sampled.isEmpty = true
  windows : ∃ g : Session × Nat → Except (Code × String) Window, sampled.zipIdx.mapM g = .ok ws.windows ∧
    ∀ x w, g x = .ok w →
      x.1.period = some w.period ∧ w.phase = x.2 ∧ w.session = x.1.session ∧ w.root = x.1.root
  closed : (heldOf r sampled).find? (!·.closed) = none
  inside : ((heldOf r sampled).findSome? fun mp =>
    (mp.windows.find? fun ses => openedIn r ses && !sampled.any (·.session == ses)).map (mp.name, ·)) = none
  opened : ((heldOf r sampled).findSome? fun mp =>
    (mp.windows.find? fun ses => !openedIn r ses).map (mp.name, ·)) = none
  ended : sampled.find? (fun s => !r.wins.any fun w => w.opened.session == s.session && w.ended.isSome) = none

theorem arrayMapM_spec {α β ε : Type} (f : α → Except ε β) (xs : Array α) :
    ⦃⌜True⌝⦄ xs.mapM f ⦃⇓? ys => ⌜xs.mapM f = .ok ys⌝⦄ := ok_spec _

-- `mvcgen` splits `windowsIn` into one goal per refusal
set_option maxHeartbeats 4000000 in
theorem windowsIn_ok {p : Params} {r : Read} {ws : Windows} (h : windowsIn p r = .ok ws) :
    ∃ sampled, WindowsIn p r ws sampled := by
  refine of_triple (Q := fun ws => ∃ sampled, WindowsIn p r ws sampled) ?_ h
  clear h
  mvcgen [windowsIn, -Spec.throw_Except, throw_spec, arrayMapM_spec]
  subst_vars
  refine ⟨(r.sessions.partition fun s => holds p.label s.families).1, ⟨rfl, by assumption,
    ⟨_, by assumption, fun x w hg => ?_⟩, none_of_forall (by assumption), none_of_forall₂ (by assumption),
    none_of_forall₂ (by assumption), none_of_forall (by assumption)⟩⟩
  obtain ⟨s, i⟩ := x
  dsimp only at hg
  split at hg
  · rename_i name hname
    have hw : ({ period := name, phase := i, session := s.session, root := s.root } : Window) = w :=
      Except.ok.inj hg
    subst hw
    exact ⟨hname, rfl, rfl, rfl⟩
  · cases hg

theorem windowsIn_spec (p : Params) (r : Read) :
    ⦃⌜True⌝⦄ windowsIn p r ⦃⇓? ws => ⌜windowsIn p r = .ok ws⌝⦄ := ok_spec _

/-! ## The ledger's half -/

/-- What `read` and `windowsIn` established of a ledger whose windows `windowsOf` returns: `read_ok`, the checked
ledger's windows told apart, and `WindowsIn` with its sessions split as two filters and its windows by place. -/
structure Returned (p : Params) (ledger : Json) (r : Read) (ws : Windows) (stop : Nat)
    (sampled : Array Session) : Prop where
  ck : checked ledger p.key = .ok (r.wins, r.periods)
  reach : reaches (entriesOf ledger) p.stop = some stop
  ses : r.sessions = r.wins.filterMap (sessionAt p r.periods (reservesOf (entriesOf ledger) p.lane)
    ((reaches (entriesOf ledger) p.start).getD stop) stop)
  dist : ∀ i j (hi : i < r.wins.size) (hj : j < r.wins.size), i ≠ j →
    r.wins[i].root ≠ r.wins[j].root ∧ r.wins[i].opened.session ≠ r.wins[j].opened.session
  sam : sampled = r.sessions.filter fun s => holds p.label s.families
  oth : ws.others = r.sessions.filter fun s => !holds p.label s.families
  size : ws.windows.size = sampled.size
  win : ∀ j (hj : j < ws.windows.size) (hs : j < sampled.size),
    sampled[j].period = some ws.windows[j].period ∧ ws.windows[j].phase = j ∧
      ws.windows[j].session = sampled[j].session ∧ ws.windows[j].root = sampled[j].root
  wi : WindowsIn p r ws sampled

theorem returned_of {p : Params} {ledger : Json} {ws : Windows} (h : windowsOf p ledger = .ok ws) :
    ∃ r stop sampled, read p ledger = .ok r ∧ windowsIn p r = .ok ws ∧ Returned p ledger r ws stop sampled := by
  unfold windowsOf at h
  obtain ⟨r, hr, hw⟩ := bind_ok.1 h
  obtain ⟨hck, stop, hstop, hses⟩ := read_ok hr
  obtain ⟨es, -, hvl⟩ := checked_ok hck
  obtain ⟨v2, hI⟩ := verifyLedger_inv hvl
  obtain ⟨sampled, hW⟩ := windowsIn_ok hw
  obtain ⟨hsam, hoth⟩ := Prod.mk.inj (hW.parts.symm.trans (partition_eq _ _))
  obtain ⟨g, hg, hgw⟩ := hW.windows
  have hmap := array_mapM_ok_getElem hg
  have hsz : ws.windows.size = sampled.size := by rw [hmap.1, Array.size_zipIdx]
  refine ⟨r, stop, sampled, hr, hw, hck, hstop, hses, hI.distinct, hsam, hoth, hsz, fun j hj hs => ?_, hW⟩
  have hz : j < sampled.zipIdx.size := by rw [Array.size_zipIdx]; exact hs
  have hgj := hgw _ _ (hmap.2 j hz hj)
  rw [Array.getElem_zipIdx] at hgj
  exact ⟨hgj.1, hgj.2.1.trans (Nat.zero_add j), hgj.2.2.1, hgj.2.2.2⟩

namespace Returned

variable {p : Params} {ledger : Json} {r : Read} {ws : Windows} {stop : Nat} {sampled : Array Session}

/-- A root the lane committed in the monitoring period has its session among `read`'s. -/
theorem session_of_win (hR : Returned p ledger r ws stop sampled) {x : Win} (hx : x ∈ r.wins)
    (hC : Committed p ledger x) :
    ∃ s ∈ r.sessions, s.root = x.root ∧ s.session = x.opened.session ∧
      s.families = getD x.opened.body "families" := by
  obtain ⟨h1, h2, start, stop', ⟨hstop', hstart⟩, h3, h4⟩ := hC
  have hst : stop' = stop := Option.some.inj (hstop'.symm.trans hR.reach)
  subst hst
  subst hstart
  obtain ⟨s, hs⟩ := sessionAt_of (p := p) (periods := r.periods)
    (reserves := reservesOf (entriesOf ledger) p.lane) h1 h2 h3 h4
  obtain ⟨-, hses, hroot, hfam, -⟩ := sessionAt_some hs
  refine ⟨s, ?_, hroot, hses, hfam⟩
  rw [hR.ses]
  exact Array.mem_filterMap.2 ⟨x, hx, hs⟩

/-- Each of `read`'s sessions is a root the lane committed in the monitoring period. -/
theorem win_of_session (hR : Returned p ledger r ws stop sampled) {s : Session} (hs : s ∈ r.sessions) :
    ∃ x ∈ r.wins, Committed p ledger x ∧ x.root = s.root ∧ x.opened.session = s.session ∧
      s.families = getD x.opened.body "families" ∧
      s.period = (r.periods.find? (·.windows.contains x.opened.session)).map (·.name) := by
  rw [hR.ses] at hs
  obtain ⟨x, hx, hxs⟩ := Array.mem_filterMap.1 hs
  obtain ⟨⟨h1, h2, h3, h4⟩, hses, hroot, hfam, hper⟩ := sessionAt_some hxs
  exact ⟨x, hx, ⟨h1, h2, _, stop, ⟨hR.reach, rfl⟩, h3, h4⟩, hroot.symm, hses.symm, hfam, hper⟩

/-- A session whose layout holds the sample's family has a window, at its place. -/
theorem window_of_sampled (hR : Returned p ledger r ws stop sampled) {s : Session} (hs : s ∈ sampled) :
    ∃ w ∈ ws.windows, w.root = s.root ∧ w.session = s.session ∧ s.period = some w.period := by
  obtain ⟨j, hj, rfl⟩ := Array.mem_iff_getElem.1 hs
  have hjw : j < ws.windows.size := by rw [hR.size]; exact hj
  have hw := hR.win j hjw hj
  exact ⟨ws.windows[j], Array.getElem_mem hjw, hw.2.2.2, hw.2.2.1, hw.1⟩

/-- Each window is of a session whose layout holds the sample's family. -/
theorem sampled_of_window (hR : Returned p ledger r ws stop sampled) {w : Window} (hw : w ∈ ws.windows) :
    ∃ s ∈ sampled, w.root = s.root ∧ w.session = s.session ∧ s.period = some w.period := by
  obtain ⟨j, hj, rfl⟩ := Array.mem_iff_getElem.1 hw
  have hjs : j < sampled.size := by rw [← hR.size]; exact hj
  have h := hR.win j hj hjs
  exact ⟨sampled[j], Array.getElem_mem hjs, h.2.2.2, h.2.2.1, h.1⟩

theorem mem_sampled (hR : Returned p ledger r ws stop sampled) {s : Session} :
    s ∈ sampled ↔ s ∈ r.sessions ∧ holds p.label s.families = true := by
  rw [hR.sam]
  exact Array.mem_filter

theorem mem_others (hR : Returned p ledger r ws stop sampled) {s : Session} :
    s ∈ ws.others ↔ s ∈ r.sessions ∧ (!holds p.label s.families) = true := by
  rw [hR.oth]
  exact Array.mem_filter

/-- A window's session has ended, and its period is closed, lists it, and lists only windows. -/
theorem period_of_sampled (hR : Returned p ledger r ws stop sampled) {s : Session} (hs : s ∈ sampled) {x : Win}
    (hx : x ∈ r.wins) (hxs : x.opened.session = s.session) {name : String} (hname : s.period = some name)
    (hper : s.period = (r.periods.find? (·.windows.contains x.opened.session)).map (·.name)) :
    x.ended.isSome = true ∧ ∃ P ∈ r.periods, P.name = name ∧ P.closed = true ∧ s.session ∈ P.windows ∧
      ∀ ses ∈ P.windows, ∃ v ∈ ws.windows, v.session = ses := by
  have hended : x.ended.isSome = true := by
    have h := bool_of_not_not (Array.find?_eq_none.1 hR.wi.ended s hs)
    obtain ⟨y, hy, hys⟩ := Array.any_eq_true'.1 h
    obtain ⟨hy1, hy2⟩ := Bool.and_eq_true_iff.1 hys
    have hyx : y = x := wins_inj hR.dist hy hx (Or.inr ((beq_iff_eq.1 hy1).trans hxs.symm))
    rw [← hyx]
    exact hy2
  obtain ⟨P, hfind, hPn⟩ := Option.map_eq_some_iff.1 (hper.symm.trans hname)
  have hP : P ∈ r.periods := Array.mem_of_find?_eq_some hfind
  have hcon := Array.find?_some hfind
  have hlist : x.opened.session ∈ P.windows := Array.contains_iff_mem.1 hcon
  have hheld : P ∈ heldOf r sampled :=
    Array.mem_filter.2 ⟨hP, Array.any_eq_true'.2 ⟨s, hs, beq_iff_eq.2 (hname.trans (congrArg some hPn.symm))⟩⟩
  have hclosed : P.closed = true := bool_of_not_not (Array.find?_eq_none.1 hR.wi.closed P hheld)
  refine ⟨hended, P, hP, hPn, hclosed, by rw [← hxs]; exact hlist, fun ses hses => ?_⟩
  have hop : openedIn r ses = true := bool_of_not_not
    (Array.find?_eq_none.1 (Option.map_eq_none_iff.1 (Array.findSome?_eq_none_iff.1 hR.wi.opened P hheld)) ses hses)
  have hin : sampled.any (·.session == ses) = true := bool_of_not_and_not
    (Array.find?_eq_none.1 (Option.map_eq_none_iff.1 (Array.findSome?_eq_none_iff.1 hR.wi.inside P hheld)) ses hses)
    hop
  obtain ⟨t, ht, hts⟩ := Array.any_eq_true'.1 hin
  obtain ⟨v, hv, -, hvs, -⟩ := hR.window_of_sampled ht
  exact ⟨v, hv, hvs.trans (beq_iff_eq.1 hts)⟩

/-- No key of a window's root and session is returned twice, among the windows and the others together. -/
theorem nodup (hR : Returned p ledger r ws stop sampled) {β : Type} (k : Session → β) (kw : Window → β)
    (hk : ∀ j (hj : j < ws.windows.size) (hs : j < sampled.size), kw ws.windows[j] = k sampled[j])
    (hn : (r.sessions.toList.map k).Nodup) :
    (ws.windows.toList.map kw ++ ws.others.toList.map k).Nodup := by
  have h := nodup_filter_append k (fun s => holds p.label s.families) hn
  rw [map_eq_of_getElem kw k hR.size hk, hR.sam, hR.oth, Array.toList_filter, Array.toList_filter]
  exact h

theorem sessions_pairwise (hR : Returned p ledger r ws stop sampled) :
    r.sessions.toList.Pairwise fun s t => s.root ≠ t.root ∧ s.session ≠ t.session := by
  have hpw : r.wins.toList.Pairwise fun x y => x.root ≠ y.root ∧ x.opened.session ≠ y.opened.session :=
    pairwise_of_distinct hR.dist
  rw [hR.ses, Array.toList_filterMap]
  refine List.Pairwise.filterMap _ (fun a a' hRel b hb b' hb' => ?_) hpw
  obtain ⟨-, hs1, hr1, -, -⟩ := sessionAt_some hb
  obtain ⟨-, hs2, hr2, -, -⟩ := sessionAt_some hb'
  exact ⟨by rw [hr1, hr2]; exact hRel.1, by rw [hs1, hs2]; exact hRel.2⟩

end Returned

/-- **What the ledger's half returns** (`Properties.Spot.MonitoringPeriod.LedgerMeans`). -/
theorem ledgerMeans : LedgerMeans := by
  intro p ledger ws h
  obtain ⟨r, stop, sampled, -, -, hR⟩ := returned_of h
  have hreach : Reaches p ledger := by
    simp [Reaches, hR.reach]
  have hcom : ∀ x ∈ r.wins, Committed p ledger x →
      (Sampled p x → ∃ w ∈ ws.windows, w.root = x.root ∧ w.session = x.opened.session) ∧
      (¬ Sampled p x → ∃ s ∈ ws.others, s.root = x.root ∧ s.session = x.opened.session) := by
    intro x hx hC
    obtain ⟨s, hs, hroot, hses, hfam⟩ := hR.session_of_win hx hC
    refine ⟨fun hS => ?_, fun hS => ?_⟩
    · have hh : holds p.label s.families = true := by rw [hfam]; exact hS
      obtain ⟨w, hw, hwr, hws, -⟩ := hR.window_of_sampled (hR.mem_sampled.2 ⟨hs, hh⟩)
      exact ⟨w, hw, hwr.trans hroot, hws.trans hses⟩
    · have hh : (!holds p.label s.families) = true := by
        rw [hfam]
        cases hb : holds p.label (getD x.opened.body "families") with
        | true => exact absurd hb hS
        | false => rfl
      exact ⟨s, hR.mem_others.2 ⟨hs, hh⟩, hroot, hses⟩
  have hwins : ∀ w ∈ ws.windows, ∃ x ∈ r.wins, Committed p ledger x ∧ Sampled p x ∧ x.root = w.root ∧
      x.opened.session = w.session ∧ x.ended.isSome ∧
      ∃ P ∈ r.periods, P.name = w.period ∧ P.closed = true ∧ w.session ∈ P.windows ∧
        ∀ s ∈ P.windows, ∃ v ∈ ws.windows, v.session = s := by
    intro w hw
    obtain ⟨s, hs, hwr, hws, hname⟩ := hR.sampled_of_window hw
    obtain ⟨hsS, hh⟩ := hR.mem_sampled.1 hs
    obtain ⟨x, hx, hC, hroot, hses, hfam, hper⟩ := hR.win_of_session hsS
    obtain ⟨hended, P, hP, hPn, hclosed, hlist, hall⟩ := hR.period_of_sampled hs hx hses hname hper
    refine ⟨x, hx, hC, ?_, hroot.trans hwr.symm, hses.trans hws.symm, hended, P, hP, hPn, hclosed, ?_, hall⟩
    · unfold Sampled
      rw [← hfam]
      exact hh
    · rw [hws]
      exact hlist
  have hothers : ∀ s ∈ ws.others, ∃ x ∈ r.wins, Committed p ledger x ∧ ¬ Sampled p x ∧ x.root = s.root ∧
      x.opened.session = s.session := by
    intro s hs
    obtain ⟨hsS, hh⟩ := hR.mem_others.1 hs
    obtain ⟨x, hx, hC, hroot, hses, hfam, -⟩ := hR.win_of_session hsS
    refine ⟨x, hx, hC, fun hS => ?_, hroot, hses⟩
    unfold Sampled at hS
    rw [← hfam] at hS
    rw [hS] at hh
    cases hh
  have hpos : 0 < ws.windows.size := by
    rw [hR.size]
    exact Array.size_pos_iff.2 fun he => hR.wi.nonempty (Array.isEmpty_iff.2 he)
  have hroots : (ws.windows.toList.map (·.root) ++ ws.others.toList.map (·.root)).Nodup :=
    hR.nodup (·.root) (·.root) (fun j hj hs => (hR.win j hj hs).2.2.2)
      (List.Pairwise.map _ (fun _ _ hne => hne.1) hR.sessions_pairwise)
  have hsessions : (ws.windows.toList.map (·.session) ++ ws.others.toList.map (·.session)).Nodup :=
    hR.nodup (·.session) (·.session) (fun j hj hs => (hR.win j hj hs).2.2.1)
      (List.Pairwise.map _ (fun _ _ hne => hne.2) hR.sessions_pairwise)
  exact ⟨r.wins, r.periods, hR.ck, hreach, hcom, hwins, hothers, hpos, hroots, hsessions,
    fun i hi => (hR.win i hi (by rw [← hR.size]; exact hi)).2.1⟩

/-! ## One window -/

theorem coinOf_spec {α : Type} (x : Flock.V α) : ⦃⌜True⌝⦄ coinOf x ⦃⇓? a => ⌜x = .ok a⌝⦄ := by
  cases x with
  | ok a => exact fun _ => rfl
  | error e => exact fun _ => True.intro

-- `mvcgen` splits `caseAt` into one goal per refusal
set_option maxHeartbeats 4000000 in
/-- What `caseAt` read where it returns: the case's registration, whose digest is the one wanted; one-stage v2's
parameters at the window; the public file's header, which names no `shared_rows`; and one-stage v2's evidence. It
returns the registration's record and what the audit answers on them. -/
theorem caseAt_ok {inner : Inner} {p : AuditParams} {ledger pins session : Json} {w : Window} {want : String}
    {before : Array Json} {k : Case} {bs : Core.Blobs} {x : Json × Core.Outcome OneStage.Public Core.Integrity}
    (h : caseAt inner p ledger pins session w want before k bs = .ok x) :
    ∃ (cm : OneStage.Commitment) (q : OneStageZk.Params) (ev : OneStageZk.Evidence),
      (Flock.canon k.registration >>= fun s => OneStage.commitmentOf s.toUTF8) = .ok cm ∧
      Flock.hexOf cm.digest = want ∧
      OneStageZk.paramsOfJson (paramsAt p w.root k.pub before) = .ok q ∧
      (∃ header, (bs.get k.pub "public file" >>= FilePart.header) = .ok header ∧
        (field? header "shared_rows").isSome = false) ∧
      (evidenceAt p ledger pins session k.evidence >>= Flock.canon >>=
        fun s => OneStageZk.evidenceOf s.toUTF8) = .ok ev ∧
      x = (cm.record, inner q cm ev bs) := by
  refine of_triple (Q := fun x => ∃ (cm : OneStage.Commitment) (q : OneStageZk.Params) (ev : OneStageZk.Evidence),
      (Flock.canon k.registration >>= fun s => OneStage.commitmentOf s.toUTF8) = .ok cm ∧
      Flock.hexOf cm.digest = want ∧
      OneStageZk.paramsOfJson (paramsAt p w.root k.pub before) = .ok q ∧
      (∃ header, (bs.get k.pub "public file" >>= FilePart.header) = .ok header ∧
        (field? header "shared_rows").isSome = false) ∧
      (evidenceAt p ledger pins session k.evidence >>= Flock.canon >>=
        fun s => OneStageZk.evidenceOf s.toUTF8) = .ok ev ∧
      x = (cm.record, inner q cm ev bs)) ?_ h
  clear h
  mvcgen [caseAt, -Spec.throw_Except, throw_spec]
  all_goals exact ⟨_, _, _, by assumption, beq_iff_eq.1 (by assumption), by assumption,
    ⟨_, by assumption, Bool.eq_false_iff.2 (by assumption)⟩, by assumption, rfl⟩

theorem caseAt_spec (inner : Inner) (p : AuditParams) (ledger pins session : Json) (w : Window) (want : String)
    (before : Array Json) (k : Case) (bs : Core.Blobs) :
    ⦃⌜True⌝⦄ caseAt inner p ledger pins session w want before k bs
      ⦃⇓? x => ⌜caseAt inner p ledger pins session w want before k bs = .ok x⌝⦄ := ok_spec _

-- `mvcgen` splits `executionPhaseAt` into one goal per refusal and per answer of the audit
set_option maxHeartbeats 4000000 in
/-- What `executionPhaseAt` read where it returns: the ledger's window on the root, its session pinned on main's line, its
files under the key, its coin commitment withheld, the one registration its log shows, the one case, and that case's
record and answer (`caseAt`). An accept adds the window with its one audit and the record; an undecided adds the
record and is kept. -/
theorem executionPhaseAt_raw {inner : Inner} {p : AuditParams} {r : Read} {ledger pins : Json} {bs : Core.Blobs}
    {st st' : Progress} {w : Window} {we : WindowEvidence}
    (h : executionPhaseAt inner p r ledger pins bs st w we = .ok st') :
    ∃ (au : Audited) (opened : Json) (lc : LiveCommitment) (roots : Roots) (d : String) (k : Case) (record : Json)
        (o : Core.Outcome OneStage.Public Core.Integrity),
      auditedIn r.wins r.periods (Json.mkObj [("root", Json.str w.root)]) = .ok (au, opened) ∧
      pinsReason au.session opened pins = none ∧
      firstParentReason au.session au.pinned p.main = none ∧
      challengePhaseNamed au p.ledger.key we.session = .ok (lc, roots) ∧
      withheldOn ledger w.root = true ∧
      (sampledOn w.root roots).toList = [d] ∧
      we.audits.toList = [k] ∧
      caseAt inner p ledger pins we.session w d st.before k bs = .ok (record, o) ∧
      ((∃ published profile, o = .accept published profile ∧
          st' = { st with before := st.before.push record,
                          passed := st.passed.push
                            { window := w, audits := #[{ registration := d, published, profile }] } }) ∨
        (∃ why, st' = { st with before := st.before.push record, undecided := st.undecided <|> some why })) := by
  refine of_triple (Q := fun st' => ∃ (au : Audited) (opened : Json) (lc : LiveCommitment) (roots : Roots)
        (d : String) (k : Case) (record : Json) (o : Core.Outcome OneStage.Public Core.Integrity),
      auditedIn r.wins r.periods (Json.mkObj [("root", Json.str w.root)]) = .ok (au, opened) ∧
      pinsReason au.session opened pins = none ∧
      firstParentReason au.session au.pinned p.main = none ∧
      challengePhaseNamed au p.ledger.key we.session = .ok (lc, roots) ∧
      withheldOn ledger w.root = true ∧
      (sampledOn w.root roots).toList = [d] ∧
      we.audits.toList = [k] ∧
      caseAt inner p ledger pins we.session w d st.before k bs = .ok (record, o) ∧
      ((∃ published profile, o = .accept published profile ∧
          st' = { st with before := st.before.push record,
                          passed := st.passed.push
                            { window := w, audits := #[{ registration := d, published, profile }] } }) ∨
        (∃ why, st' = { st with before := st.before.push record, undecided := st.undecided <|> some why }))) ?_ h
  clear h
  mvcgen [executionPhaseAt, -Spec.throw_Except, throw_spec, coinOf_spec, caseAt_spec]
  all_goals
    subst_vars
    refine ⟨_, _, _, _, _, _, _, _, by assumption, none_of_forall (by assumption), none_of_forall (by assumption),
      by assumption, by assumption, by assumption, by assumption, by assumption, ?_⟩
  all_goals first
    | exact Or.inl ⟨_, _, by assumption, rfl⟩
    | exact Or.inr ⟨_, rfl⟩

/-- One step of the check along the windows, from the state `st` to `st'` at the window `w` with the evidence `we`:
the window is added with one audit `a`, which is the check's audit at `w` after the records in `st`, and its record
is added; or an audit is left undecided. -/
def StepOk (inner : Inner) (p : AuditParams) (c : Commitment) (e : Evidence) (bs : Core.Blobs) (r : Read)
    (st : Progress) (w : Window) (we : WindowEvidence) (st' : Progress) : Prop :=
  (∃ a record, st'.before = st.before.push record ∧
      st'.passed = st.passed.push { window := w, audits := #[a] } ∧ st'.undecided = st.undecided ∧
      AuditedAt inner p c e bs r w we st.before record a) ∨
    st'.undecided.isSome = true

theorem executionPhaseAt_step {inner : Inner} {p : AuditParams} {c : Commitment} {e : Evidence} {bs : Core.Blobs} {r : Read}
    {st st' : Progress} {w : Window} {we : WindowEvidence}
    (h : executionPhaseAt inner p r c.ledger e.pins bs st w we = .ok st') : StepOk inner p c e bs r st w we st' := by
  obtain ⟨au, opened, lc, roots, d, k, record, o, h1, h2, h3, h4, h5, h6, h7, h8, h9⟩ := executionPhaseAt_raw h
  obtain ⟨cm, q, ev, c1, c2, c3, c4, c5, c6⟩ := caseAt_ok h8
  obtain ⟨hrec, ho⟩ := Prod.mk.inj c6
  unfold StepOk
  rcases h9 with ⟨published, profile, hacc, hst⟩ | ⟨why, hst⟩
  · subst hst
    refine Or.inl ⟨{ registration := d, published, profile }, record, rfl, rfl, rfl, ?_⟩
    exact ⟨au, opened, lc, roots, k, cm, q, ev, h1, h2, h3, h4, h5, Array.ext' h6, Array.ext' h7, c1, c2, hrec, c3,
      c4, c5, ho.symm.trans hacc⟩
  · subst hst
    refine Or.inr ?_
    show (st.undecided <|> some why).isSome = true
    cases st.undecided <;> rfl

/-! ## The audits' half -/

theorem foldlM_spec {α σ ε : Type} (f : σ → α → Except ε σ) (init : σ) (xs : Array α) :
    ⦃⌜True⌝⦄ xs.foldlM f init ⦃⇓? r => ⌜xs.foldlM f init = .ok r⌝⦄ := ok_spec _

/-- What `registeredIn` returns of the ledger's half's other sessions: each one a period committing a circuit lists,
by that circuit and its session. -/
def othersOf (r : Read) (others : Array Session) : Array Other :=
  others.filterMap fun s =>
    (registrationOf r s.session).map fun d => ({ registration := d, session := s.session } : Other)

/-- What `registeredIn` checked where it returns: every window is this registration's, every other session of the
lane is listed by a period that committed another circuit, and it returns those (`othersOf`). -/
theorem registeredIn_ok {p : AuditParams} {r : Read} {ws : Windows} {os : Array Other}
    (h : registeredIn p r ws = .ok os) :
    (∀ w ∈ ws.windows, registrationOf r w.session = some p.registration) ∧
      (∀ s ∈ ws.others, ∃ d, registrationOf r s.session = some d ∧ d ≠ p.registration) ∧
      os = othersOf r ws.others := by
  refine of_triple (Q := fun os => (∀ w ∈ ws.windows, registrationOf r w.session = some p.registration) ∧
      (∀ s ∈ ws.others, ∃ d, registrationOf r s.session = some d ∧ d ≠ p.registration) ∧
      os = othersOf r ws.others) ?_ h
  clear h
  unfold othersOf
  mvcgen [registeredIn, -Spec.throw_Except, throw_spec]
  rename_i _ o1 hn1 hf1 _ o2 hn2 hf2 _ o3 hn3 hf3
  have h1 := hf1.trans (none_of_forall hn1)
  have h2 := hf2.trans (none_of_forall hn2)
  have h3 := hf3.trans (none_of_forall hn3)
  refine ⟨fun w hw => ?_, fun s hs => ?_⟩
  · have := Array.find?_eq_none.1 h1 w hw
    simpa using this
  · have a := Array.find?_eq_none.1 h2 s hs
    have b := Array.find?_eq_none.1 h3 s hs
    cases hd : registrationOf r s.session with
    | none => simp [hd] at a
    | some d => exact ⟨d, rfl, fun hdp => by simp [hd, hdp] at b⟩

theorem registeredIn_spec (p : AuditParams) (r : Read) (ws : Windows) :
    ⦃⌜True⌝⦄ registeredIn p r ws ⦃⇓? os => ⌜registeredIn p r ws = .ok os⌝⦄ := ok_spec _

/-- A session `othersOf` returns is one of the others, under the circuit its period committed. -/
theorem othersOf_mem {r : Read} {others : Array Session} {o : Other} (ho : o ∈ othersOf r others) :
    ∃ s ∈ others, registrationOf r s.session = some o.registration ∧ o.session = s.session := by
  obtain ⟨s, hs, hso⟩ := Array.mem_filterMap.1 ho
  obtain ⟨d, hd, rfl⟩ := Option.map_eq_some_iff.1 hso
  exact ⟨s, hs, hd, rfl⟩

/-- Each of the others that a period committing a circuit lists is returned by `othersOf`, under that circuit. -/
theorem mem_othersOf {r : Read} {others : Array Session} {s : Session} {d : String} (hs : s ∈ others)
    (hd : registrationOf r s.session = some d) : (⟨d, s.session⟩ : Other) ∈ othersOf r others :=
  Array.mem_filterMap.2 ⟨s, hs, by rw [hd]; rfl⟩

theorem filterMap_map_sublist {α β γ : Type} (f : α → Option β) (g : β → γ) (k : α → γ)
    (h : ∀ a b, f a = some b → g b = k a) (l : List α) : ((l.filterMap f).map g).Sublist (l.map k) := by
  induction l with
  | nil => exact List.Sublist.slnil
  | cons a l ih =>
    cases hfa : f a with
    | none =>
      rw [List.filterMap_cons_none hfa, List.map_cons]
      exact ih.cons _
    | some b =>
      rw [List.filterMap_cons_some hfa, List.map_cons, List.map_cons, h a b hfa]
      exact ih.cons_cons _

/-- `othersOf` returns no session twice where the others hold none twice. -/
theorem othersOf_nodup {r : Read} {others : Array Session} (h : (others.toList.map (·.session)).Nodup) :
    ((othersOf r others).toList.map (·.session)).Nodup := by
  unfold othersOf
  rw [Array.toList_filterMap]
  refine List.Sublist.nodup (filterMap_map_sublist _ _ _ (fun a b hab => ?_) _) h
  obtain ⟨d, -, rfl⟩ := Option.map_eq_some_iff.1 hab
  rfl

-- `mvcgen` splits `run` at the ledger's half, whose each session is, the count of the evidence's windows and the fold
set_option maxHeartbeats 4000000 in
/-- What `run` read where it returns: the ledger's half's sessions and windows, whose each session is, as many
windows' evidence, and the fold of `executionPhaseAt` over the windows with their evidence, from the empty state. -/
theorem run_ok {inner : Inner} {p : AuditParams} {c : Commitment} {e : Evidence} {bs : Core.Blobs}
    {x : Array Other × Progress} (h : _root_.Spot.MonitoringPeriod.run inner p c e bs = .ok x) :
    ∃ r ws, read p.ledger c.ledger = .ok r ∧ windowsIn p.ledger r = .ok ws ∧ registeredIn p r ws = .ok x.1 ∧
      e.windows.size = ws.windows.size ∧
      ∃ f : Progress → Window × WindowEvidence → Except (Code × String) Progress,
        (ws.windows.zip e.windows).foldlM f ({} : Progress) = .ok x.2 ∧
        ∀ st w ev, f st (w, ev) = executionPhaseAt inner p r c.ledger e.pins bs st w ev := by
  refine of_triple (Q := fun x => ∃ r ws, read p.ledger c.ledger = .ok r ∧ windowsIn p.ledger r = .ok ws ∧
      registeredIn p r ws = .ok x.1 ∧ e.windows.size = ws.windows.size ∧
      ∃ f : Progress → Window × WindowEvidence → Except (Code × String) Progress,
        (ws.windows.zip e.windows).foldlM f ({} : Progress) = .ok x.2 ∧
        ∀ st w ev, f st (w, ev) = executionPhaseAt inner p r c.ledger e.pins bs st w ev) ?_ h
  clear h
  mvcgen [_root_.Spot.MonitoringPeriod.run, -Spec.throw_Except, throw_spec, -Spec.foldlM_array, foldlM_spec, read_spec,
    windowsIn_spec, registeredIn_spec]
  all_goals exact ⟨_, _, by assumption, by assumption, by assumption, beq_iff_eq.1 (by assumption), _,
    by assumption, fun _ _ _ => rfl⟩

/-- What `checkWith` ran where it accepts: `run` returned, with no audit undecided, and the period publishes this
registration's circuit and the windows it passed, with the lane's sessions that are other registrations'. -/
theorem checkWith_accept {inner : Inner} {p : AuditParams} {c : Commitment} {e : Evidence} {bs : Core.Blobs}
    {pub : Public} {prof : Array Core.Integrity} (h : checkWith inner p c e bs = .accept pub prof) :
    ∃ others st, _root_.Spot.MonitoringPeriod.run inner p c e bs = .ok (others, st) ∧ st.undecided = none ∧
      pub = { lane := p.ledger.lane, start := p.ledger.start, stop := p.ledger.stop,
              registration := p.registration, windows := st.passed, others := others } ∧
      prof = st.passed.flatMap fun x => x.audits.map (·.profile) := by
  unfold checkWith at h
  split at h
  · cases h
  · rename_i others st hrun
    split at h
    · cases h
    · rename_i hund
      cases h
      exact ⟨others, st, hrun, hund, rfl, rfl⟩

/-- What the check's state `st` holds once it has read the windows and evidence `pre`, with no audit undecided:
one passed window and one record for each, and window `i` passed with one audit, the check's at that window after
the records of the windows before it. -/
def FoldGood (inner : Inner) (p : AuditParams) (c : Commitment) (e : Evidence) (bs : Core.Blobs) (r : Read)
    (pre : List (Window × WindowEvidence)) (st : Progress) : Prop :=
  st.passed.size = pre.length ∧ st.before.size = pre.length ∧
    ∀ (i : Nat) (w : Window) (we : WindowEvidence), pre[i]? = some (w, we) →
      ∃ a record, st.passed[i]? = some { window := w, audits := #[a] } ∧ st.before[i]? = some record ∧
        AuditedAt inner p c e bs r w we (st.before.extract 0 i) record a

theorem foldGood_step {inner : Inner} {p : AuditParams} {c : Commitment} {e : Evidence} {bs : Core.Blobs}
    {r : Read} {pre : List (Window × WindowEvidence)} {st st' : Progress} {w : Window} {we : WindowEvidence}
    {a : Audit} {record : Json} (hG : FoldGood inner p c e bs r pre st)
    (hb : st'.before = st.before.push record) (hp : st'.passed = st.passed.push { window := w, audits := #[a] })
    (hA : AuditedAt inner p c e bs r w we st.before record a) :
    FoldGood inner p c e bs r (pre ++ [(w, we)]) st' := by
  obtain ⟨hps, hbs, hall⟩ := hG
  refine ⟨by rw [hp, Array.size_push, hps, List.length_append, List.length_singleton],
    by rw [hb, Array.size_push, hbs, List.length_append, List.length_singleton], fun i w' we' hi => ?_⟩
  rcases getElem?_snoc hi with hi | ⟨hi, hx⟩
  · obtain ⟨hlt, -⟩ := List.getElem?_eq_some_iff.1 hi
    obtain ⟨a', record', h1, h2, h3⟩ := hall i w' we' hi
    refine ⟨a', record', ?_, ?_, ?_⟩
    · rw [hp, Array.getElem?_push, ite_eq_right (by omega)]
      exact h1
    · rw [hb, Array.getElem?_push, ite_eq_right (by omega)]
      exact h2
    · rw [hb, Array.extract_push_of_le (by omega)]
      exact h3
  · obtain ⟨hw, hwe⟩ := Prod.mk.inj hx
    subst hw
    subst hwe
    have hi' : i = st.before.size := by omega
    refine ⟨a, record, ?_, ?_, ?_⟩
    · rw [hp, Array.getElem?_push, ite_eq_left (by omega)]
    · rw [hb, Array.getElem?_push, ite_eq_left hi']
    · rw [hb, Array.extract_push_of_le (by omega), hi', Array.extract_size]
      exact hA

/-- The fold's invariant: while no audit is undecided, `FoldGood`. -/
def FoldInv (inner : Inner) (p : AuditParams) (c : Commitment) (e : Evidence) (bs : Core.Blobs) (r : Read)
    (pre : List (Window × WindowEvidence)) (st : Progress) : Prop :=
  st.undecided = none → FoldGood inner p c e bs r pre st

theorem foldInv_step {inner : Inner} {p : AuditParams} {c : Commitment} {e : Evidence} {bs : Core.Blobs}
    {r : Read} {pre : List (Window × WindowEvidence)} {st st' : Progress} {w : Window} {we : WindowEvidence}
    (hI : FoldInv inner p c e bs r pre st) (hS : StepOk inner p c e bs r st w we st') :
    FoldInv inner p c e bs r (pre ++ [(w, we)]) st' := by
  unfold FoldInv at hI ⊢
  unfold StepOk at hS
  intro hn
  rcases hS with ⟨a, record, hb, hp, hu, hA⟩ | hsome
  · exact foldGood_step (hI (hu.symm.trans hn)) hb hp hA
  · rw [hn] at hsome
    cases hsome

/-- **What the period's check accepts** (`Properties.Spot.MonitoringPeriod.AcceptMeans`). -/
theorem acceptMeans : AcceptMeans := by
  intro inner p c e bs pub prof h
  obtain ⟨others, st, hrun, hund, hpub, hprof⟩ := checkWith_accept h
  obtain ⟨r, ws, hr, hw, hreg, hsize, f, hfold, hf⟩ := run_ok hrun
  have hreg' : registeredIn p r ws = .ok others := hreg
  have hfold' : (ws.windows.zip e.windows).toList.foldlM f ({} : Progress) = .ok st := by
    rw [Array.foldlM_toList]
    exact hfold
  have hstep : ∀ pre x s t, FoldInv inner p c e bs r pre s → f s x = .ok t →
      FoldInv inner p c e bs r (pre ++ [x]) t := by
    intro pre x s t hI hfx
    obtain ⟨w, we⟩ := x
    rw [hf] at hfx
    exact foldInv_step hI (executionPhaseAt_step hfx)
  have hinit : FoldInv inner p c e bs r [] ({} : Progress) := by
    unfold FoldInv
    intro _
    exact ⟨rfl, rfl, fun i w we hi => by simp at hi⟩
  have hinv : FoldInv inner p c e bs r ([] ++ (ws.windows.zip e.windows).toList) st :=
    foldlM_prefix f (FoldInv inner p c e bs r) hstep _ [] _ st hinit hfold'
  have hgood : FoldGood inner p c e bs r (ws.windows.zip e.windows).toList st := by
    unfold FoldInv at hinv
    have := hinv hund
    rwa [List.nil_append] at this
  obtain ⟨hps, hbs, hall⟩ := hgood
  have hlen : (ws.windows.zip e.windows).toList.length = ws.windows.size := by
    rw [Array.length_toList, Array.size_zip]
    omega
  have hpsz : st.passed.size = ws.windows.size := hps.trans hlen
  have hat : ∀ i (hi : i < st.passed.size), ∃ (hw : i < ws.windows.size) (he : i < e.windows.size) (a : Audit)
      (record : Json), st.passed[i] = { window := ws.windows[i], audits := #[a] } ∧
        st.before[i]? = some record ∧
        AuditedAt inner p c e bs r ws.windows[i] e.windows[i] (st.before.extract 0 i) record a := by
    intro i hi
    have hiw : i < ws.windows.size := by omega
    have hie : i < e.windows.size := by omega
    obtain ⟨a, record, h1, h2, h3⟩ := hall i _ _ (zip_getElem? hiw hie)
    obtain ⟨_, h1'⟩ := Array.getElem?_eq_some_iff.1 h1
    exact ⟨hiw, hie, a, record, h1', h2, h3⟩
  subst hpub
  subst hprof
  dsimp only
  refine ⟨rfl, rfl, rfl, rfl, r, ws, hr, hw, hreg', ?_, hsize, rfl, st.before, hbs.trans hps.symm, fun i hi => ?_⟩
  · refine Array.ext (by rw [Array.size_map]; exact hpsz) fun i h1 h2 => ?_
    rw [Array.getElem_map]
    obtain ⟨_, _, a, record, h3, -, -⟩ := hat i (by rw [Array.size_map] at h1; exact h1)
    exact congrArg Passed.window h3
  · obtain ⟨hiw, hie, a, record, h1, h2, h3⟩ := hat i hi
    refine ⟨e.windows[i], a, record, Array.getElem?_eq_getElem hie, congrArg Passed.audits h1, h2, ?_⟩
    rw [h1]
    exact h3

/-- **Every root committed under this registration is one published window, with one audit entry**
(`Properties.Spot.MonitoringPeriod.CommittedOnce`). -/
theorem committedOnce : CommittedOnce := by
  intro inner p c e bs pub prof h
  obtain ⟨-, -, -, -, r, ws, hr, hw, hreg, hmap, -, -, records, -, haud⟩ := acceptMeans inner p c e bs pub prof h
  obtain ⟨hwin, hoth, -⟩ := registeredIn_ok hreg
  have hwo : windowsOf p.ledger c.ledger = .ok ws := by
    unfold windowsOf
    exact bind_ok.2 ⟨r, hr, hw⟩
  obtain ⟨wins, periods, hck, hreach, hcom, hwins, -, -, hroots, -, -⟩ := ledgerMeans p.ledger c.ledger ws hwo
  obtain ⟨hck', -⟩ := read_ok hr
  have hwe : wins = r.wins := (Prod.mk.inj (Except.ok.inj (hck.symm.trans hck'))).1
  subst hwe
  have hsz : pub.windows.size = ws.windows.size := by
    rw [← hmap, Array.size_map]
  have hget : ∀ j (hj : j < pub.windows.size) (hj' : j < ws.windows.size),
      pub.windows[j].window = ws.windows[j] := fun j hj hj' => getElem_of_map_eq hmap j hj hj'
  refine ⟨r, hr, hck', hreach, fun x hx hC => ⟨?_, fun hthis => ?_⟩, fun w hw => ?_⟩
  · by_cases hS : Sampled p.ledger x
    · obtain ⟨w, hwm, -, hws⟩ := (hcom x hx hC).1 hS
      exact ⟨p.registration, by rw [← hws]; exact hwin w hwm⟩
    · obtain ⟨s, hs, -, hss⟩ := (hcom x hx hC).2 hS
      obtain ⟨d, hd, -⟩ := hoth s hs
      exact ⟨d, by rw [← hss]; exact hd⟩
  · have hS : Sampled p.ledger x := by
      by_cases hS : Sampled p.ledger x
      · exact hS
      · obtain ⟨s, hs, -, hss⟩ := (hcom x hx hC).2 hS
        obtain ⟨d, hd, hne⟩ := hoth s hs
        rw [hss, hthis] at hd
        exact (hne (Option.some.inj hd).symm).elim
    obtain ⟨w, hwm, hwr, hws⟩ := (hcom x hx hC).1 hS
    obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.1 hwm
    have hi' : i < pub.windows.size := by rw [hsz]; exact hi
    obtain ⟨we, a, record, -, ha, -, -⟩ := haud i hi'
    refine ⟨i, hi', ?_, ?_, ⟨a, ha⟩, fun j hj hjr => ?_⟩
    · rw [hget i hi' hi]
      exact hwr
    · rw [hget i hi' hi]
      exact hws
    · have hj' : j < ws.windows.size := by rw [← hsz]; exact hj
      rw [hget j hj hj'] at hjr
      exact index_eq_of_nodup (List.nodup_append.1 hroots).1 hj' hi (hjr.trans hwr.symm)
  · obtain ⟨j, hj, rfl⟩ := Array.mem_iff_getElem.1 hw
    have hj' : j < ws.windows.size := by rw [← hsz]; exact hj
    obtain ⟨x, hx, hC, hS, hxr, hxs, -⟩ := hwins ws.windows[j] (Array.getElem_mem hj')
    exact ⟨x, hx, hC, hS, by rw [hxs]; exact hwin _ (Array.getElem_mem hj'), by rw [hget j hj hj']; exact hxr,
      by rw [hget j hj hj']; exact hxs⟩

/-- **The published others are exactly the roots the lane committed under another circuit**
(`Properties.Spot.MonitoringPeriod.OthersElsewhere`). -/
theorem othersElsewhere : OthersElsewhere := by
  intro inner p c e bs pub prof h
  obtain ⟨-, -, -, -, r, ws, hr, hw, hreg, -⟩ := acceptMeans inner p c e bs pub prof h
  obtain ⟨hwin, hoth, hpo⟩ := registeredIn_ok hreg
  have hwo : windowsOf p.ledger c.ledger = .ok ws := by
    unfold windowsOf
    exact bind_ok.2 ⟨r, hr, hw⟩
  obtain ⟨wins, periods, hck, -, hcom, -, hothers, -, -, hses, -⟩ := ledgerMeans p.ledger c.ledger ws hwo
  obtain ⟨hck', -⟩ := read_ok hr
  have hwe : wins = r.wins := (Prod.mk.inj (Except.ok.inj (hck.symm.trans hck'))).1
  subst hwe
  refine ⟨r, hr, fun x hx hC d hd hne => ?_, fun o ho => ?_, ?_⟩
  · by_cases hS : Sampled p.ledger x
    · obtain ⟨w, hwm, -, hws⟩ := (hcom x hx hC).1 hS
      have hthis := hwin w hwm
      rw [hws, hd] at hthis
      exact (hne (Option.some.inj hthis)).elim
    · obtain ⟨s, hs, -, hss⟩ := (hcom x hx hC).2 hS
      refine ⟨⟨d, s.session⟩, ?_, rfl, hss⟩
      rw [hpo]
      exact mem_othersOf hs (by rw [hss]; exact hd)
  · rw [hpo] at ho
    obtain ⟨s, hs, hsd, hos⟩ := othersOf_mem ho
    obtain ⟨d, hd, hne⟩ := hoth s hs
    obtain ⟨x, hx, hC, -, -, hxs⟩ := hothers s hs
    have hdo : d = o.registration := Option.some.inj (hd.symm.trans hsd)
    refine ⟨by rw [← hdo]; exact hne, x, hx, hC, by rw [hxs]; exact hsd, by rw [hos]; exact hxs⟩
  · rw [hpo]
    exact othersOf_nodup (List.nodup_append.1 hses).2.1

/-! ## The decoder -/

theorem keyOf_spec (j : Json) :
    ⦃⌜True⌝⦄ ChallengerKey.ofJson j ⦃⇓? k => ⌜ChallengerKey.ofJson j = .ok k⌝⦄ := ok_spec _

/-- `Params.of` reads the ledger's key from the key's JSON. -/
theorem paramsOf_key {key label lane start stop : Json} {q : Params}
    (h : Params.of key label lane start stop = .ok q) : ChallengerKey.ofJson key = .ok q.key := by
  refine of_triple (Q := fun q => ChallengerKey.ofJson key = .ok q.key) ?_ h
  clear h
  mvcgen [Params.of, -Spec.throw_Except, throw_spec, keyOf_spec]

theorem paramsOf_spec (key label lane start stop : Json) :
    ⦃⌜True⌝⦄ Params.of key label lane start stop
      ⦃⇓? q => ⌜Params.of key label lane start stop = .ok q⌝⦄ := ok_spec _

theorem mapError_spec {ε ε' α : Type} (x : Except ε α) (f : ε → ε') :
    ⦃⌜True⌝⦄ x.mapError f ⦃⇓? _ => ⌜True⌝⦄ := by
  cases x <;> exact fun _ => True.intro

/-- An `audit` that names none of the keys a window sets or leaves unset. -/
theorem none_of_keys {audit : Json}
    (h : ((WINDOW_KEYS ++ UNDERIVED_KEYS).all fun k => (field? audit k).isNone) = true) :
    field? audit "challenger" = none ∧ field? audit "window_root" = none ∧
      field? audit "seed_salt" = none ∧ field? audit "day" = none ∧ field? audit "declarations" = none := by
  have hk : ∀ k ∈ WINDOW_KEYS ++ UNDERIVED_KEYS, field? audit k = none := fun k hk =>
    Option.isNone_iff_eq_none.1 (Array.all_eq_true'.1 h k hk)
  exact ⟨hk _ (by simp [WINDOW_KEYS, UNDERIVED_KEYS]), hk _ (by simp [WINDOW_KEYS, UNDERIVED_KEYS]),
    hk _ (by simp [WINDOW_KEYS, UNDERIVED_KEYS]), hk _ (by simp [WINDOW_KEYS, UNDERIVED_KEYS]),
    hk _ (by simp [WINDOW_KEYS, UNDERIVED_KEYS])⟩

/-- **What the decoder gives the check** (`Properties.Spot.MonitoringPeriod.DecodedMeans`). -/
theorem decodedMeans : DecodedMeans := by
  intro j p h
  refine of_triple (Q := fun p => ChallengerKey.ofJson p.keyJ = .ok p.ledger.key ∧
      field? p.audit "challenger" = none ∧ field? p.audit "window_root" = none ∧
      field? p.audit "seed_salt" = none ∧ field? p.audit "day" = none ∧
      field? p.audit "declarations" = none) ?_ h
  clear h
  mvcgen [AuditParams.of, -Spec.throw_Except, throw_spec, paramsOf_spec, mapError_spec]
  all_goals exact ⟨paramsOf_key (by assumption), none_of_keys (Bool.and_eq_true_iff.1 (by assumption)).2⟩

/-! ## The session audited at a window -/

-- `mvcgen` splits `auditedIn` into one goal per check it makes of the window and its `end`
set_option maxHeartbeats 4000000 in
/-- `auditedIn` reads the session of the first of the ledger's windows whose root is the one the caller names. -/
theorem auditedIn_ok {wins : Array Win} {periods : Array MonitoringPeriod} {window : Json} {a : Audited × Json}
    (h : auditedIn wins periods window = .ok a) :
    ∃ win, wins.find? (fun w => w.root == strOf (getD window "root")) = some win ∧
      a.1.session = win.opened.session := by
  refine of_triple (Q := fun a => ∃ win,
      wins.find? (fun w => w.root == strOf (getD window "root")) = some win ∧
      a.1.session = win.opened.session) ?_ h
  clear h
  mvcgen [auditedIn, -Spec.throw_Except, throw_spec]
  all_goals exact ⟨_, by assumption, rfl⟩

/-- `auditedIn` on a root reads the session of one of the ledger's windows on that root. -/
theorem auditedIn_root {wins : Array Win} {periods : Array MonitoringPeriod} {root : String} {a : Audited × Json}
    (h : auditedIn wins periods (Json.mkObj [("root", Json.str root)]) = .ok a) :
    ∃ win ∈ wins, win.root = root ∧ a.1.session = win.opened.session := by
  obtain ⟨win, hfind, hses⟩ := auditedIn_ok h
  rw [getD_root] at hfind
  have hbeq := Array.find?_some hfind
  exact ⟨win, Array.mem_of_find?_eq_some hfind, beq_iff_eq.1 hbeq, hses⟩

/-- **The session audited at a window is the window's** (`Properties.Spot.MonitoringPeriod.AuditedSession`). -/
theorem auditedChallengePhase : AuditedSession := by
  intro p ledger r ws w au opened hr hw hmem ha
  have hwo : windowsOf p ledger = .ok ws := by
    unfold windowsOf
    exact bind_ok.2 ⟨r, hr, hw⟩
  obtain ⟨r', stop, sampled, hr', -, hR⟩ := returned_of hwo
  have hrr : r' = r := Except.ok.inj (hr'.symm.trans hr)
  rw [hrr] at hR
  obtain ⟨s, hs, hwr, hws, -⟩ := hR.sampled_of_window hmem
  obtain ⟨win, hwin, hroot, hses⟩ := auditedIn_root ha
  obtain ⟨x, hx, -, hxr, hxs, -, -⟩ := hR.win_of_session (hR.mem_sampled.1 hs).1
  have hxw : win = x := wins_inj hR.dist hwin hx (Or.inl (hroot.trans (hwr.trans hxr.symm)))
  exact hses.trans ((congrArg (fun y : Win => y.opened.session) hxw).trans (hxs.trans hws.symm))

end Properties.Spot.MonitoringPeriod
