import FlockSoundness.Refine.Seam
import FlockSoundness.Protocol.Refine.Paths
import FlockSoundness.Protocol.CompiledSound
import FlockSoundness.Protocol.Extract
import FlockSoundness.Protocol.Game.Batch

/-!
# The verifier refines the model's table (PROTOCOL.md §3, §5)

`verify_refines`: when the executable's `Flock.verify` accepts a session record and its two proof files, under the
unsalted SHA-512 scheme, the record and both proofs decode. Then at the proofs' openings (`opensOf`), each rep's
compiled game `repC`, played with its proof's messages and its stream's coins, accepts. Both reps' paths also verify
against the one level-0 cap. That is `tableAfter`'s verdict. `verify_tableAfter` states it as an accepting run of
`tableAfter`, with existential messages and rep 0 first.
-/

namespace FlockSoundness.Refine

open Flock FlockLevel3 Game Transcript Model Accounting

local notation "𝔽" => FlockLevel3.GF128

theorem bytes_eq_of_beq {a b : ByteArray} (h : (a == b) = true) : a = b := ByteArray.ext (beq_iff_eq.1 h)

theorem flat_data (L : List ByteArray) : ∀ acc : ByteArray,
    (L.foldl (· ++ ·) acc).data.toList = acc.data.toList ++ (L.map fun b => b.data.toList).flatten := by
  induction L with
  | nil => intro acc; simp
  | cons b L ih =>
      intro acc
      rw [List.foldl_cons, ih, ByteArray.data_append, Array.toList_append]
      simp

/-- Equally many 64-byte entries that flatten to the same bytes are the same entries. -/
theorem entries_eq_of_flat : ∀ (a b : List ByteArray), (∀ x ∈ a, x.size = 64) → (∀ x ∈ b, x.size = 64) →
    a.length = b.length → (a.map fun b => b.data.toList).flatten = (b.map fun b => b.data.toList).flatten → a = b
  | [], [], _, _, _, _ => rfl
  | [], _ :: _, _, _, hl, _ => by simp at hl
  | _ :: _, [], _, _, hl, _ => by simp at hl
  | x :: a, y :: b, ha, hb, hl, h => by
      simp only [List.map_cons, List.flatten_cons] at h
      obtain ⟨h1, h2⟩ := List.append_inj h (by
        rw [Array.length_toList, Array.length_toList, ByteArray.size_data, ByteArray.size_data,
          ha x (List.mem_cons_self), hb y (List.mem_cons_self)])
      have hxy : x = y := ByteArray.ext (Array.ext' h1)
      rw [hxy, entries_eq_of_flat a b (fun z hz => ha z (List.mem_cons_of_mem _ hz))
        (fun z hz => hb z (List.mem_cons_of_mem _ hz)) (by simpa using hl) h2]

theorem list_mapM_ok_mem {α β ε : Type} {f : α → Except ε β} :
    ∀ {l : List α} {l' : List β}, l.mapM f = .ok l' → ∀ y ∈ l', ∃ x, f x = .ok y
  | [], l', h, y, hy => by
      simp only [List.mapM_nil, pure, Except.pure, Except.ok.injEq] at h
      subst h; cases hy
  | a :: l, l', h, y, hy => by
      rw [List.mapM_cons] at h
      obtain ⟨b, hb, h⟩ := ebind_ok h
      obtain ⟨bs, hbs, h⟩ := ebind_ok h
      simp only [pure, Except.pure, Except.ok.injEq] at h
      subst h
      rcases List.mem_cons.1 hy with rfl | hy
      · exact ⟨a, hb⟩
      · exact list_mapM_ok_mem hbs y hy

theorem mapM_ok_mem {α β ε : Type} {f : α → Except ε β} {xs : Array α} {ys : Array β}
    (h : xs.mapM f = .ok ys) : ∀ y ∈ ys.toList, ∃ x, f x = .ok y := by
  rw [Array.mapM_eq_mapM_toList] at h
  cases hl : xs.toList.mapM f with
  | error e => rw [hl] at h; cases h
  | ok l =>
      rw [hl] at h
      simp only [Functor.map, Except.map, Except.ok.injEq] at h
      subst h
      simpa using list_mapM_ok_mem hl

theorem mapIdxM_go_ok_mem {α β ε : Type} {f : ℕ → α → Except ε β} :
    ∀ {l : List α} {acc : Array β} {l' : List β}, List.mapIdxM.go f l acc = .ok l' →
      ∀ y ∈ l', y ∈ acc.toList ∨ ∃ i x, f i x = .ok y
  | [], acc, l', h, y, hy => by
      simp only [List.mapIdxM.go, pure, Except.pure, Except.ok.injEq] at h
      subst h; exact Or.inl hy
  | a :: l, acc, l', h, y, hy => by
      simp only [List.mapIdxM.go] at h
      obtain ⟨b, hb, h⟩ := ebind_ok h
      rcases mapIdxM_go_ok_mem h y hy with hy | hy
      · rw [Array.toList_push, List.mem_append, List.mem_singleton] at hy
        rcases hy with hy | rfl
        · exact Or.inl hy
        · exact Or.inr ⟨_, _, hb⟩
      · exact Or.inr hy

theorem mapIdxM_ok_mem {α β ε : Type} {f : ℕ → α → Except ε β} {xs : Array α} {ys : Array β}
    (h : xs.mapIdxM f = .ok ys) : ∀ y ∈ ys.toList, ∃ i x, f i x = .ok y := by
  have e := Array.toList_mapIdxM (xs := xs) (f := f)
  rw [h] at e
  simp only [Functor.map, Except.map] at e
  intro y hy
  rcases mapIdxM_go_ok_mem (f := f) (l := xs.toList) (acc := #[]) (by simpa [List.mapIdxM] using e.symm) y hy with
    hy | hy
  · simp at hy
  · exact hy

theorem check_ok_iff {b : Bool} {why : String} {u : Unit} : Record.check b why = .ok u ↔ b = true := by
  unfold Record.check
  cases b
  · exact ⟨(fun h => by cases h), (fun h => by cases h)⟩
  · exact ⟨fun _ => rfl, fun _ => rfl⟩

theorem check_coins {n : ℕ} {coins : Array Flock.F128} {a b c : Bool} {why : String} {u : Unit}
    (h : Record.check (a && b && coins.size == n && c) why = .ok u) : coins.size = n := by
  have := check_ok_iff.1 h
  simp only [Bool.and_eq_true, beq_iff_eq] at this
  exact this.1.2

theorem roundsWF_of {rs : Array Round} (h : ∀ r ∈ rs.toList, r.coins.size = r.n) : RoundsWF rs := by
  intro i hi
  rw [getElem!_pos rs i hi]
  exact h _ (Array.getElem_mem_toList hi)

set_option maxHeartbeats 1000000 in
/-- The record's rounds are well formed: `Record.decode` checks every round's coin count (S12). -/
theorem decode_roundsWF {spec : SessionSpec} {text : String} {sess : Session}
    (h : Record.decode spec text = .ok sess) : ∀ s ∈ sess.streams.toList, RoundsWF s.rounds := by
  unfold Record.decode at h
  dsimp only at h
  repeat' (first | (obtain ⟨_, _, h⟩ := ebind_ok h) | (split at h))
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    intro s hs
    try dsimp only at hs
    obtain ⟨x, hx⟩ := mapM_ok_mem (by assumption) s hs
    try dsimp only at hx
    repeat' (first | (obtain ⟨_, _, hx⟩ := ebind_ok hx) | (split at hx))
    all_goals
      simp only [pure, Except.pure, Except.ok.injEq] at hx
      subst hx
      apply roundsWF_of
      intro r hr
      obtain ⟨k, y, hg⟩ := mapIdxM_ok_mem (by assumption) r hr
      try dsimp only at hg
      repeat' (first | (obtain ⟨_, _, hg⟩ := ebind_ok hg) | (split at hg))
      all_goals
        simp only [pure, Except.pure, Except.ok.injEq] at hg
        subst hg
        exact check_coins (by assumption)

theorem exec_fast100_ge {m : ℕ} {s : Flock.Schedule} (hs : Flock.fast100 m = some s) : 22 ≤ m := by
  by_contra hm
  have : (m < 22 || m > 35) = true := by rw [Bool.or_eq_true, decide_eq_true_iff, decide_eq_true_iff]; omega
  unfold Flock.fast100 at hs
  rw [ite_eq_left_of_eq_true _ _ (eq_true this)] at hs
  cases hs

theorem mapM_singleton_ok {α β ε : Type} {f : α → Except ε β} {a : α} {ys : Array β} (h : #[a].mapM f = .ok ys) :
    ∃ y, f a = .ok y ∧ ys = #[y] := by
  rw [Array.mapM_eq_mapM_toList] at h
  simp only [List.mapM_cons, List.mapM_nil] at h
  cases hf : f a with
  | error e => rw [hf] at h; cases h
  | ok y =>
      rw [hf] at h
      simp only [Bind.bind, Except.bind, Pure.pure, Except.pure, Functor.map, Except.map, Except.ok.injEq] at h
      exact ⟨y, rfl, h.symm⟩

theorem forIn_range_one_ok {σ : Type} {s s' : σ} {g : ℕ → σ → V (ForInStep σ)}
    (h : (forIn [:1] s g : V σ) = .ok s') : ∃ r, g 0 s = .ok r := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range'] at h
  simp only [Std.Legacy.Range.size, List.range'] at h
  obtain ⟨r, hr, -⟩ := ebind_ok (by simpa using h)
  exact ⟨r, hr⟩

/-- The one-table call accepts: the record decodes, the table's schedule is `fast100`'s, and both reps' checks accept
with equal caps. -/
theorem verify_one_ok {spec : SessionSpec} {st : Setup} {record : String} {proofs : Array ByteArray}
    (h : Flock.verify spec #[st] record proofs = .ok ()) :
    ∃ sess sch c, Record.decode spec record = .ok sess ∧ Flock.fast100 st.m = some sch ∧
      verifyRep spec st sch sess 0 0 proofs[0]! = .ok c ∧ verifyRep spec st sch sess 0 1 proofs[1]! = .ok c := by
  unfold Flock.verify at h
  dsimp only at h
  by_cases c0 : (#[st].size != spec.tables.size || #[st].size == 0) = true
  · rw [ite_eq_left c0] at h; obtain ⟨_, h0, _⟩ := ebind_ok h; cases h0
  rw [ite_eq_right c0] at h
  obtain ⟨sess, hsess, h⟩ := ebind_ok h
  obtain ⟨_, _, h⟩ := ebind_ok h
  obtain ⟨schs, hschs, h⟩ := ebind_ok h
  obtain ⟨sch, hsch, rfl⟩ := mapM_singleton_ok hschs
  split at h
  · obtain ⟨_, h0, _⟩ := ebind_ok h; cases h0
  split at h
  · obtain ⟨_, h0, _⟩ := ebind_ok h; cases h0
  obtain ⟨_, _, h⟩ := ebind_ok h
  obtain ⟨_, hloop, _⟩ := ebind_ok h
  rw [show (#[st] : Array Setup).size = 1 from rfl] at hloop
  obtain ⟨r, hr⟩ := forIn_range_one_ok hloop
  obtain ⟨c0', h0, hr⟩ := ebind_ok hr
  obtain ⟨c1', h1, hr⟩ := ebind_ok hr
  simp only [Nat.mul_zero, Nat.zero_add] at h0 h1
  rw [show (#[st] : Array Setup)[0]! = st from rfl, show (#[sch] : Array Flock.Schedule)[0]! = sch from rfl] at h0 h1
  have hf : Flock.fast100 st.m = some sch := by
    cases hf : Flock.fast100 st.m with
    | none => rw [hf] at hsch; cases hsch
    | some s => rw [hf] at hsch; cases hsch; rfl
  by_cases c3 : (c0' != c1') = true
  · rw [ite_eq_left c3] at hr; obtain ⟨_, h0, _⟩ := ebind_ok hr; cases h0
  have hb : (c0' == c1') = true := by
    cases hb : (c0' == c1')
    · exact absurd (by simp [bne, hb]) c3
    · rfl
  rw [bytes_eq_of_beq hb] at h0
  exact ⟨sess, sch, c1', hsess, hf, h0, h1⟩

/-- The proof files' openings (§14): rep `r`'s opened rows and siblings, level by level. -/
def opensOf (p₀ p₁ : Proof) (sch : Flock.Schedule) : Opens 𝔽 Dig :=
  fun r ℓ j => if r = 0 then openRow p₀.ligerito sch ℓ j else openRow p₁.ligerito sch ℓ j

end FlockSoundness.Refine
