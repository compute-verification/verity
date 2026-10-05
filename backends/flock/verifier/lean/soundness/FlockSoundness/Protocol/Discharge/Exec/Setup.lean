import FlockSoundness.Protocol.Discharge.Exec.Event
import FlockSoundness.Protocol.Refine.Setup

/-!
# flock-verify's acceptance sets up each table with `Stmt.setupH`

`Accepts I t` sets up the session with `Stmt.setupTables`. Under a tag without hidden outputs
(`tags.hiddenOutputs = false`; with them it calls `Stmt.setupHidden`), each of its tables is `Stmt.setupH`'s statement
(`setupTables_ok`): for one table, at the inputs and the record's draw; for `count > 1`, under
`Tags.tableIdentity` and at a subset draw of the table's part. So every fact proved of `setupH`'s statements
(`setupH_wf`, `setupH_layout`, `placement_of_setupH`) holds of each table flock-verify accepts with
(`accepts_setupH`).

`setupH_public` records the two checks `setupH` makes on the public file that the layout's registered values read:
`HmRow.checkPublic st.c st.pub` passed, and `st.regions` is `HmRow.regions st.c st.pub`.
-/

set_option Elab.async false

namespace FlockSoundness.Discharge.Exec

open Flock Refine
open Lean (Json)

theorem list_mapM_ok_getElem {α β ε : Type} {f : α → Except ε β} :
    ∀ {l : List α} {l' : List β}, l.mapM f = .ok l' →
      l'.length = l.length ∧ ∀ (j : ℕ) (hj : j < l.length) (hj' : j < l'.length), f l[j] = .ok l'[j]
  | [], l', h => by
      simp only [List.mapM_nil, pure, Except.pure, Except.ok.injEq] at h
      subst h
      exact ⟨rfl, fun j hj => absurd hj (Nat.not_lt_zero _)⟩
  | a :: l, l', h => by
      rw [List.mapM_cons] at h
      obtain ⟨b, hb, h⟩ := ebind_ok h
      obtain ⟨bs, hbs, h⟩ := ebind_ok h
      simp only [pure, Except.pure, Except.ok.injEq] at h
      subst h
      obtain ⟨hlen, hall⟩ := list_mapM_ok_getElem hbs
      refine ⟨by simp [hlen], fun j hj hj' => ?_⟩
      rcases j with _ | j
      · exact hb
      · exact hall j (by simpa using hj) (by simpa using hj')

theorem array_mapM_ok_getElem {α β ε : Type} {f : α → Except ε β} {xs : Array α} {ys : Array β}
    (h : xs.mapM f = .ok ys) : ys.size = xs.size ∧ ∀ (j : ℕ) (hj : j < xs.size) (hj' : j < ys.size),
      f xs[j] = .ok ys[j] := by
  rw [Array.mapM_eq_mapM_toList] at h
  cases hl : xs.toList.mapM f with
  | error e => rw [hl] at h; cases h
  | ok l =>
      rw [hl] at h
      simp only [Functor.map, Except.map, Except.ok.injEq] at h
      subst h
      obtain ⟨hlen, hall⟩ := list_mapM_ok_getElem hl
      refine ⟨by simpa using hlen, fun j hj hj' => ?_⟩
      have := hall j (by simpa using hj) (by simpa using hj')
      simpa using this

theorem range_mapM_ok {β ε : Type} {f : ℕ → Except ε β} {k : ℕ} {ys : Array β} (h : (Array.range k).mapM f = .ok ys) :
    ys.size = k ∧ ∀ (j : ℕ) (hj : j < ys.size), f j = .ok ys[j] := by
  obtain ⟨hsz, hall⟩ := array_mapM_ok_getElem h
  rw [Array.size_range] at hsz
  refine ⟨hsz, fun j hj => ?_⟩
  have := hall j (by rw [Array.size_range]; omega) hj
  rwa [Array.getElem_range] at this

variable {tags : Tags} {cf pf : ByteArray} {coins : Coins} {count : ℕ} {tables : List (String × ByteArray)}
  {draw partition : Option Json} {program : Option ByteArray} {work : Option (ℕ × Array (String × ℕ × ℕ))}
  {stratified : Option ℕ} {closure : Option (Std.HashMap ℕ (Array ℕ))} {spec : SessionSpec} {sts : Array Stmt}

/-- A table's tags in a session of `count` tables: the statement's for one, under `Tags.tableIdentity` for several. -/
def tagsAt (tags : Tags) (count j : ℕ) : Tags :=
  if count ≤ 1 then tags else { tags with identity := tags.tableIdentity count j }

theorem tagsAt_typed (tags : Tags) (count j : ℕ) : (tagsAt tags count j).typed = tags.typed := by
  unfold tagsAt; split_ifs <;> rfl

end FlockSoundness.Discharge.Exec
