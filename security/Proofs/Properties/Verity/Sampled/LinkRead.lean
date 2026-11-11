module

public import Proofs.Properties.Verity.Sampled.Trees

/-!
# v2's read of the release log, up to its first opening

`Verity.Sampled.readLog` reads the release log's lines into the device's log as v2's check reads them. Where a release
it reads opens a round and none before it does (`readLog_first`), the release is the read (`Verity.Sampled.treeStep`)
of a coin tree's line whose event isn't a tree's sample, no line before that line opens a round
(`Verity.Sampled.openLine`), and the trees read before it are those whose specs the releases before it show sampled
(`Verity.Sampled.helloSpecs`), in order. The read of such a line (`step_open`) names the tree it opens by its root, the
stream and round it opens, the tree's place among those read and the stream's among its spec's, and gives the coins
a record of that tree gives that round.
-/

@[expose] public section

namespace Verity.Sampled.LinkRead

open Lean (Json)
open Spot Spot.Challenger

/-! ## A coin tree's line -/

/-- **A coin tree's line whose event isn't a tree's sample reads as a round opened**: of the tree its line names by its
root, of the stream and round it names, at the place `i` of that tree among the trees read and `s` of the stream among
its spec's, with the coins a record of that tree gives that round (none where no record does). -/
theorem step_open {records : List Json} {ts ts' : List (String × Json)} {d : Json} {r : Verity.Release}
    (h : Verity.Sampled.treeStep records ts d = some (r, ts'))
    (hT : jsonStrIs (getD d "event") TREE_EVENT = false) :
    ∃ root stream k i s n, getD d "tree" = .str root ∧ getD d "stream" = .str stream ∧
      nat? (getD d "round") = some k ∧ r.frame = .opening i s n ∧ ts' = ts ∧
      ((ts[i]?).bind fun t => (Verity.Sampled.specStreams t.2)[s]?) = some (.str stream) ∧
      r.coins = (match records.find? (Verity.Sampled.recordTree · == root) with
        | some rec => ((Verity.Sampled.recordRound rec stream k).map Verity.bitsOfBytes).getD []
        | none => []) := by
  unfold Verity.Sampled.treeStep at h
  rw [ite_eq_right (by simp [hT])] at h
  simp only [Option.bind_eq_bind] at h
  split at h
  · rename_i root hroot
    split at h
    · rename_i stream hstream
      split at h
      · obtain ⟨k, hk, h⟩ := Option.bind_eq_some_iff.1 h
        obtain ⟨n, -, h⟩ := Option.bind_eq_some_iff.1 h
        obtain ⟨i, -, h⟩ := Option.bind_eq_some_iff.1 h
        obtain ⟨⟨_, spec⟩, hts, h⟩ := Option.bind_eq_some_iff.1 h
        obtain ⟨s, hs, h⟩ := Option.bind_eq_some_iff.1 h
        obtain ⟨b, -, h⟩ := Option.bind_eq_some_iff.1 h
        simp only [Option.pure_def, Option.some.injEq, Prod.mk.injEq] at h
        obtain ⟨rfl, rfl⟩ := h
        refine ⟨root, stream, k, i, s, n, hroot, hstream, hk, rfl, rfl, ?_, rfl⟩
        rw [hts, Option.bind_some]
        obtain ⟨hlt, heq, -⟩ := Array.findIdx?_eq_some_iff_getElem.1 hs
        simp [Array.getElem?_eq_getElem hlt, Trees.strIs heq]
      · simp at h
    · simp at h
  · simp at h

/-- **A coin tree's line whose event is a tree's sample reads as its `Hello`**: it adds its tree's line spec to the
trees read, which parses to the `Hello`'s spec (`Verity.Sampled.specOf`). -/
theorem step_hello {records : List Json} {ts ts' : List (String × Json)} {d : Json} {r : Verity.Release}
    (h : Verity.Sampled.treeStep records ts d = some (r, ts'))
    (hT : jsonStrIs (getD d "event") TREE_EVENT = true) :
    ∃ root sj spec, r.frame = .hello spec ∧ ts' = ts ++ [(root, sj)] ∧ Verity.Sampled.specOf sj = some spec := by
  unfold Verity.Sampled.treeStep at h
  rw [ite_eq_left hT] at h
  simp only [Option.bind_eq_bind] at h
  split at h
  · split at h
    · split at h
      · rename_i root _
        obtain ⟨spec, hspec, h⟩ := Option.bind_eq_some_iff.1 h
        obtain ⟨ν, -, h⟩ := Option.bind_eq_some_iff.1 h
        obtain ⟨r', -, h⟩ := Option.bind_eq_some_iff.1 h
        obtain ⟨κ, -, h⟩ := Option.bind_eq_some_iff.1 h
        simp only [Option.pure_def, Option.some.injEq, Prod.mk.injEq] at h
        obtain ⟨rfl, rfl⟩ := h
        exact ⟨root, _, spec, rfl, rfl, hspec⟩
      · simp at h
    · simp at h
  · simp at h

theorem event_ok {l : String} {j : Json} (h : (event l).toOption = some j) : event l = .ok j := by
  cases he : event l with
  | ok j' =>
    rw [he] at h
    simp only [Except.toOption, Option.some.injEq] at h
    rw [h]
  | error _ =>
    rw [he] at h
    simp [Except.toOption] at h

/-- A line whose event is a tree's sample opens no round. -/
theorem openLine_tree {l : String} {d : Json} (hev : event l = .ok d)
    (hT : jsonStrIs (getD d "event") TREE_EVENT = true) : Verity.Sampled.openLine l = false := by
  simp only [Verity.Sampled.openLine, hev]
  rw [Trees.strIs hT]
  simp [jsonStrIs, TREE_EVENT, OPEN_EVENT]

/-- A line that is no coin tree's opens no round. -/
theorem openLine_not_tree {l : String} (ht : ¬ treeLine l = true) : Verity.Sampled.openLine l = false := by
  cases hev : event l with
  | error _ => simp only [Verity.Sampled.openLine, hev]
  | ok d =>
    have htl : treeLine l = (jsonStrIs (getD d "event") TREE_EVENT || jsonStrIs (getD d "event") OPEN_EVENT) := by
      simp only [treeLine, hev]
    simp only [Verity.Sampled.openLine, hev]
    cases ho : jsonStrIs (getD d "event") OPEN_EVENT
    · rfl
    · exact absurd (by rw [htl, ho, Bool.or_true]) ht

/-! ## The trees read and the specs sampled -/

theorem forall₂_snoc' {α β : Type} {R : α → β → Prop} {a : α} {b : β} :
    ∀ {l : List α} {l' : List β}, List.Forall₂ R l l' → R a b → List.Forall₂ R (l ++ [a]) (l' ++ [b])
  | _, _, .nil, h => .cons h .nil
  | _, _, .cons h₁ t, h => .cons h₁ (forall₂_snoc' t h)

theorem forall₂_getElem?' {α β : Type} {R : α → β → Prop} :
    ∀ {l : List α} {l' : List β}, List.Forall₂ R l l' → ∀ {i : ℕ} {a : α}, l[i]? = some a →
      ∃ b, l'[i]? = some b ∧ R a b
  | _, _, .nil, _, _, h => by simp at h
  | _, _, .cons hab _, 0, _, h => by
    simp only [List.getElem?_cons_zero, Option.some.injEq] at h
    subst h
    exact ⟨_, rfl, hab⟩
  | _, _, .cons _ t, _ + 1, _, h => by
    simp only [List.getElem?_cons_succ] at h ⊢
    exact forall₂_getElem?' t h

theorem mapM_getElem?' {α β : Type} {f : α → Option β} :
    ∀ {l : List α} {l' : List β}, l.mapM f = some l' → ∀ s : ℕ, l'[s]? = l[s]?.bind f
  | [], l', h, s => by
    simp only [List.mapM_nil, Option.pure_def, Option.some.injEq] at h
    subst h
    simp
  | a :: l, l', h, s => by
    simp only [List.mapM_cons, Option.bind_eq_bind, Option.pure_def] at h
    obtain ⟨b, hb, h⟩ := Option.bind_eq_some_iff.1 h
    obtain ⟨t, ht, h⟩ := Option.bind_eq_some_iff.1 h
    obtain rfl := Option.some.inj h
    cases s with
    | zero => simp [hb]
    | succ s => simpa using mapM_getElem?' ht s

theorem mapM_getElem?_arr' {α β : Type} {f : α → Option β} {xs : Array α} {ys : Array β}
    (h : xs.mapM f = some ys) (s : ℕ) : ys[s]? = xs[s]?.bind f := by
  rw [Array.mapM_eq_mapM_toList] at h
  obtain ⟨l', hl', rfl⟩ : ∃ l', xs.toList.mapM f = some l' ∧ l'.toArray = ys := by
    cases hm : xs.toList.mapM f with
    | none => simp [hm] at h
    | some l' => exact ⟨l', rfl, by simpa [hm] using h⟩
  rw [List.getElem?_toArray, mapM_getElem?' hl', Array.getElem?_toList]

/-- **A stream a coin tree's line spec names is its parsed spec's**, at the same place. -/
theorem specOf_streams' {j : Json} {sp : Flock.CoinTree.Spec} (h : Verity.Sampled.specOf j = some sp) {s : ℕ}
    {x : String} (hx : (Verity.Sampled.specStreams j)[s]? = some (.str x)) : sp.streams[s]? = some x := by
  unfold Verity.Sampled.specOf at h
  unfold Verity.Sampled.specStreams at hx
  cases hss : getD j "streams" with
  | arr ss =>
    simp only [hss, Option.bind_eq_bind, Option.pure_def] at h hx
    obtain ⟨streams, hst, h⟩ := Option.bind_eq_some_iff.1 h
    obtain ⟨a, -, h⟩ := Option.bind_eq_some_iff.1 h
    obtain ⟨b, -, h⟩ := Option.bind_eq_some_iff.1 h
    obtain rfl := Option.some.inj h
    rw [mapM_getElem?_arr' hst, hx]
    rfl
  | _ => simp [hss] at h

/-! ## The read up to its first opening -/

/-- **`readTail`'s first opening**: where the releases `readTail` reads from `lines` after the trees `ts` are
`pre ++ y :: post`, `y` opens a round and no release of `pre` does, `y` is the read of a line of `lines` that
`coinTreesOf` reads as an opening, no line before it opens a round, and the trees read before it are `ts`, then those
whose specs `pre` shows sampled. -/
theorem readTail_first {label : String} {after : ByteArray} {records : List Json} :
    ∀ (lines : List String) (ts : List (String × Json)) (pending : Option String) (ξ : ByteArray)
      (H : List Flock.CoinTree.Spec) (pre post : List Verity.Release) (y : Verity.Release) (i s n : ℕ),
      Verity.Sampled.readTail label after records lines ts pending ξ = some (pre ++ y :: post) →
      List.Forall₂ (fun t sp => Verity.Sampled.specOf t.2 = some sp) ts H →
      (∀ q ∈ pre, ∀ i' s' n', q.frame ≠ .opening i' s' n') → y.frame = .opening i s n →
      ∃ l₀ l l₁ d ts₁ ts₂, lines = l₀ ++ l :: l₁ ∧ (∀ l' ∈ l₀, Verity.Sampled.openLine l' = false) ∧
        treeLine l = true ∧ event l = .ok d ∧ jsonStrIs (getD d "event") TREE_EVENT = false ∧
        Verity.Sampled.treeStep records ts₁ d = some (y, ts₂) ∧
        List.Forall₂ (fun t sp => Verity.Sampled.specOf t.2 = some sp) ts₁ (H ++ Verity.Sampled.helloSpecs pre)
  | [], _, none, _, _, _, _, _, _, _, _, hR, _, _, _ => by simp [Verity.Sampled.readTail] at hR
  | [], _, some _, _, _, _, _, _, _, _, _, hR, _, _, _ => by simp [Verity.Sampled.readTail] at hR
  | l :: ls, ts, pending, ξ, H, pre, post, y, i, s, n, hR, hS, hpre, hy => by
    unfold Verity.Sampled.readTail at hR
    by_cases ht : treeLine l = true
    · rw [ite_eq_left ht] at hR
      simp only [Option.bind_eq_bind] at hR
      obtain ⟨⟨r, ts'⟩, hX, hR⟩ := Option.bind_eq_some_iff.1 hR
      obtain ⟨j, hj, hX⟩ := Option.bind_eq_some_iff.1 hX
      simp only [Option.pure_def] at hR
      obtain ⟨rest, hrest, hR⟩ := Option.bind_eq_some_iff.1 hR
      have hev := event_ok hj
      cases pre with
      | nil =>
        obtain ⟨rfl, -⟩ : r = y ∧ rest = post := by simpa using Option.some.inj hR
        cases hTv : jsonStrIs (getD j "event") TREE_EVENT
        · refine ⟨[], l, ls, j, ts, ts', rfl, fun _ h => absurd h List.not_mem_nil, ht, hev, hTv, hX, ?_⟩
          rw [show Verity.Sampled.helloSpecs [] = [] from rfl, List.append_nil]
          exact hS
        · obtain ⟨_, _, _, hfr, -⟩ := step_hello hX hTv
          rw [hy] at hfr
          cases hfr
      | cons q pre' =>
        obtain ⟨rfl, rfl⟩ : r = q ∧ rest = pre' ++ y :: post := by simpa using Option.some.inj hR
        cases hTv : jsonStrIs (getD j "event") TREE_EVENT
        · obtain ⟨_, _, _, i', s', n', -, -, -, hfr, -⟩ := step_open hX hTv
          exact absurd hfr (hpre r (List.mem_cons_self ..) i' s' n')
        · obtain ⟨root, sj, spec, hfr, hts', hsp⟩ := step_hello hX hTv
          obtain ⟨l₀, l', l₁, d, ts₁, ts₂, hls, hno, htl', hev', hT', hst, hS'⟩ :=
            readTail_first ls ts' pending ξ (H ++ [spec]) pre' post y i s n hrest
              (by rw [hts']; exact forall₂_snoc' hS hsp) (fun q hq => hpre q (List.mem_cons_of_mem _ hq)) hy
          refine ⟨l :: l₀, l', l₁, d, ts₁, ts₂, by simp [hls], ?_, htl', hev', hT', hst, ?_⟩
          · intro l'' hl''
            rcases List.mem_cons.1 hl'' with rfl | hl''
            · exact openLine_tree hev hTv
            · exact hno l'' hl''
          · have e : Verity.Sampled.helloSpecs (r :: pre') = spec :: Verity.Sampled.helloSpecs pre' := by
              simp [Verity.Sampled.helloSpecs, hfr]
            rw [e, ← List.singleton_append, ← List.append_assoc]
            exact hS'
    · rw [ite_eq_right ht] at hR
      cases pending with
      | none =>
        obtain ⟨l₀, l', l₁, d, ts₁, ts₂, hls, hno, htl', hev', hT', hst, hS'⟩ :=
          readTail_first ls ts (some l) ξ H pre post y i s n hR hS hpre hy
        refine ⟨l :: l₀, l', l₁, d, ts₁, ts₂, by simp [hls], ?_, htl', hev', hT', hst, hS'⟩
        intro l'' hl''
        rcases List.mem_cons.1 hl'' with rfl | hl''
        · exact openLine_not_tree ht
        · exact hno l'' hl''
      | some a =>
        simp only [Option.bind_eq_bind] at hR
        obtain ⟨⟨lab, root⟩, -, hR⟩ := Option.bind_eq_some_iff.1 hR
        simp only [Option.pure_def] at hR
        have key : ∀ (rest : List Verity.Release) (ξ' : ByteArray) (r : Verity.Release) (lb : String),
            Verity.Sampled.readTail label after records ls ts none ξ' = some rest →
            r :: rest = pre ++ y :: post → r.frame = .fresh lb →
            ∃ l₀ l' l₁ d ts₁ ts₂, l :: ls = l₀ ++ l' :: l₁ ∧ (∀ l'' ∈ l₀, Verity.Sampled.openLine l'' = false) ∧
              treeLine l' = true ∧ event l' = .ok d ∧ jsonStrIs (getD d "event") TREE_EVENT = false ∧
              Verity.Sampled.treeStep records ts₁ d = some (y, ts₂) ∧
              List.Forall₂ (fun t sp => Verity.Sampled.specOf t.2 = some sp) ts₁
                (H ++ Verity.Sampled.helloSpecs pre) := by
          intro rest ξ' r lb hrest he hr
          cases pre with
          | nil =>
            obtain ⟨rfl, -⟩ : r = y ∧ rest = post := by simpa using he
            rw [hr] at hy
            cases hy
          | cons q pre' =>
            obtain ⟨rfl, rfl⟩ : r = q ∧ rest = pre' ++ y :: post := by simpa using he
            obtain ⟨l₀, l', l₁, d, ts₁, ts₂, hls, hno, htl', hev', hT', hst, hS'⟩ :=
              readTail_first ls ts none ξ' H pre' post y i s n hrest hS
                (fun q hq => hpre q (List.mem_cons_of_mem _ hq)) hy
            refine ⟨l :: l₀, l', l₁, d, ts₁, ts₂, by simp [hls], ?_, htl', hev', hT', hst, ?_⟩
            · intro l'' hl''
              rcases List.mem_cons.1 hl'' with rfl | hl''
              · exact openLine_not_tree ht
              · exact hno l'' hl''
            · have e : Verity.Sampled.helloSpecs (r :: pre') = Verity.Sampled.helloSpecs pre' := by
                simp [Verity.Sampled.helloSpecs, hr]
              rw [e]
              exact hS'
        by_cases hm : lab = label ∧ root = after
        · rw [ite_eq_left hm] at hR
          obtain ⟨rest, hrest, hR⟩ := Option.bind_eq_some_iff.1 hR
          exact key rest _ _ label hrest (Option.some.inj hR) (by rfl)
        · rw [ite_eq_right hm] at hR
          obtain ⟨rest, hrest, hR⟩ := Option.bind_eq_some_iff.1 hR
          exact key rest _ _ lab hrest (Option.some.inj hR) (by rfl)

/-- **`readLog`'s first opening**: where the releases `readLog` reads from the release log's lines are
`pre ++ y :: post`, `y` opens a round and no release of `pre` does, `y` is the read of a line of the log that
`coinTreesOf` reads as an opening, no line before it opens a round, and the trees read before it are those whose specs
`pre` shows sampled. -/
theorem readLog_first {label : String} {after : ByteArray} {records : List Json} {ξ : ByteArray}
    {lines : List String} {pre post : List Verity.Release} {y : Verity.Release} {i s n : ℕ}
    (hR : Verity.Sampled.readLog label after records ξ lines = some (pre ++ y :: post))
    (hpre : ∀ q ∈ pre, ∀ i' s' n', q.frame ≠ .opening i' s' n') (hy : y.frame = .opening i s n) :
    ∃ l₀ l l₁ d ts₁ ts₂, lines = l₀ ++ l :: l₁ ∧ (∀ l' ∈ l₀, Verity.Sampled.openLine l' = false) ∧
      treeLine l = true ∧ event l = .ok d ∧ jsonStrIs (getD d "event") TREE_EVENT = false ∧
      Verity.Sampled.treeStep records ts₁ d = some (y, ts₂) ∧
      List.Forall₂ (fun t sp => Verity.Sampled.specOf t.2 = some sp) ts₁ (Verity.Sampled.helloSpecs pre) := by
  cases lines with
  | nil => simp [Verity.Sampled.readLog] at hR
  | cons l ls =>
    by_cases ht : treeLine l = true
    · rw [Verity.Sampled.readLog, ite_eq_left ht] at hR
      have h2 : Verity.Sampled.readTail label after records (l :: ls) [] none ξ = some (pre ++ y :: post) := by
        rw [Verity.Sampled.readTail, ite_eq_left ht]
        exact hR
      obtain ⟨l₀, l', l₁, d, ts₁, ts₂, hls, hno, htl', hev', hT', hst, hS'⟩ :=
        readTail_first (l :: ls) [] none ξ [] pre post y i s n h2 .nil hpre hy
      exact ⟨l₀, l', l₁, d, ts₁, ts₂, hls, hno, htl', hev', hT', hst, by simpa using hS'⟩
    · rw [Verity.Sampled.readLog, ite_eq_right ht] at hR
      obtain ⟨l₀, l', l₁, d, ts₁, ts₂, hls, hno, htl', hev', hT', hst, hS'⟩ :=
        readTail_first ls [] none ξ [] pre post y i s n hR .nil hpre hy
      refine ⟨l :: l₀, l', l₁, d, ts₁, ts₂, by simp [hls], ?_, htl', hev', hT', hst, by simpa using hS'⟩
      intro l'' hl''
      rcases List.mem_cons.1 hl'' with rfl | hl''
      · exact openLine_not_tree ht
      · exact hno l'' hl''

end Verity.Sampled.LinkRead
