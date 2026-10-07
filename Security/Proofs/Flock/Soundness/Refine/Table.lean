import Proofs.Flock.Soundness.Refine.Paths
import Proofs.Flock.Soundness.Refine.Seam
import Proofs.Flock.Soundness.Session

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

/-- **The verifier refines the model's table** (§3, §5, §9–§14). When `Flock.verify` accepts a one-table session record
and its two proof files under the unsalted SHA-512 scheme, on a statement whose `m`, `k_log`, circuit fold and extra claims are
the model's `S` and `extra`, the record and both proofs decode. Then at the proofs' openings, each rep's compiled game,
played with its proof's messages and every coin of its stream, accepts, and both reps' paths verify against the one
level-0 cap. -/
theorem verify_refines (spec : SessionSpec) (st : Setup) (S : Model.Statement) (hm : 13 ≤ S.m)
    (hmS : st.m = S.m) (hkS : st.kLog = S.kLog) (hfold : FoldRealizes st.fold S) (msId : String)
    (hms : st.merkle = MerkleScheme.sha512 msId) (extra : Session → List (Model.Claim 𝔽 S.m))
    (record : String) (proofs : Array ByteArray)
    (hext : ∀ sess, Record.decode spec record = .ok sess → ExtraRealizes st S sess.points (extra sess))
    (h : Flock.verify spec #[st] record proofs = .ok ()) :
    ∃ (sess : Session) (sch : Flock.Schedule) (p₀ p₁ : Proof) (s₀ s₁ : StreamRec) (pend₀ pend₁ : PendL 𝔽 Dig),
      Record.decode spec record = .ok sess ∧ Flock.fast100 st.m = some sch ∧
      decodeProof (shapeOf st sch (extra sess).length) proofs[0]! = .ok p₀ ∧
      decodeProof (shapeOf st sch (extra sess).length) proofs[1]! = .ok p₁ ∧
      sess.streams.find? (fun x => x.name == toString spec.tables[0]! ++ toString "/rep" ++ toString 0) = some s₀ ∧
      sess.streams.find? (fun x => x.name == toString spec.tables[0]! ++ toString "/rep" ++ toString 1) = some s₁ ∧
      Plays fOf (Model.repC execArith 0 S (schedOf sch) hm (extra sess))
        (repMsgs st sch (2 + (extra sess).length) p₀ capDig) (s₀.rounds.toList.map (·.coins)) pend₀ ∧
      Plays fOf (Model.repC execArith 1 S (schedOf sch) hm (extra sess))
        (repMsgs st sch (2 + (extra sess).length) p₁ capDig) (s₁.rounds.toList.map (·.coins)) pend₁ ∧
      pend₀.verdict (opensOf p₀ p₁ sch) = true ∧ pend₁.verdict (opensOf p₀ p₁ sch) = true ∧
      OpensOK H512 enc512 (capDig p₀.cap) 0 pend₀ (opensOf p₀ p₁ sch) ∧
      OpensOK H512 enc512 (capDig p₀.cap) 1 pend₁ (opensOf p₀ p₁ sch) := by
  obtain ⟨sess, sch, c, hsess, hf, h0, h1⟩ := verify_one_ok h
  have h15 : 15 ≤ S.m := by have := exec_fast100_ge hf; omega
  have hwf := decode_roundsWF hsess
  obtain ⟨hdim, hsm⟩ := fast100_dims st.m sch hf
  have hlv : ∀ i < sch.levels.size, LevelSeam (sch.levels.getD i default) := fun i hi =>
    fast100_level st.m sch hf _ (by unfold Array.getD; rw [dite_eq_left hi]; exact Array.getElem_mem hi)
  have hα1 : ∀ i < sch.levels.size, ceilLog2 (sch.levels.getD i default).q ≠ 1 := fun i hi =>
    fast100_alpha st.m sch hf _ (by unfold Array.getD; rw [dite_eq_left hi]; exact Array.getElem_mem hi)
  have hschm : sch.m = S.m := hsm.trans hmS
  obtain ⟨s₀, p₀, hs₀, hp₀, hc₀, -, hsz₀, hw₀, hrest₀⟩ := rep_refines spec st S hm h15 sch hmS hkS hfold (extra sess)
    sess (hext sess hsess) hdim hlv hα1 hschm hwf 0 0 _ _ h0
  obtain ⟨s₁, p₁, hs₁, hp₁, hc₁, -, hsz₁, hw₁, hrest₁⟩ := rep_refines spec st S hm h15 sch hmS hkS hfold (extra sess)
    sess (hext sess hsess) hdim hlv hα1 hschm hwf 0 1 _ _ h1
  obtain ⟨pend₀, hpl₀, hcaps₀, hreqs₀, hver₀⟩ := hrest₀ capDig (opensOf p₀ p₁ sch) fun ℓ _ j _ => rfl
  obtain ⟨pend₁, hpl₁, hcaps₁, hreqs₁, hver₁⟩ := hrest₁ capDig (opensOf p₀ p₁ sch) fun ℓ _ j _ => rfl
  rw [hms] at hreqs₀ hreqs₁ hw₀ hw₁
  have hok₀ := opensOK_of msId sch hlv p₀.cap p₀.ligerito 0 pend₀ (opensOf p₀ p₁ sch) hcaps₀ hreqs₀ fun ℓ j => rfl
  have hok₁ := opensOK_of msId sch hlv p₁.cap p₁.ligerito 1 pend₁ (opensOf p₀ p₁ sch) hcaps₁ hreqs₁ fun ℓ j => rfl
  have hcapeq : p₁.cap = p₀.cap := by
    apply Array.ext'
    refine entries_eq_of_flat _ _ (fun x hx => hw₁ x hx) (fun x hx => hw₀ x hx) (by simp [hsz₀, hsz₁]) ?_
    have e := congrArg (fun b : ByteArray => b.data.toList) (hc₁.symm.trans hc₀)
    simp only [← Array.foldl_toList, flat_data] at e
    simpa using e
  rw [hcapeq] at hok₁
  exact ⟨sess, sch, p₀, p₁, s₀, s₁, pend₀, pend₁, hsess, hf, hp₀, hp₁, hs₀, hs₁, hpl₀, hpl₁, hver₀, hver₁, hok₀,
    hok₁⟩

open Classical in
/-- **The verifier refines the model's table, as one run** (§5). When `Flock.verify` accepts a one-table session, with the extra claims the
model's `extraClaims` at link points `pts` read off the record, some message list makes `tableAfter` accept. The run is
played on the two streams' coins, rep 0's then rep 1's, and its openings are the proofs'. The messages are existential
and the schedule is rep 0 first, while the live server may interleave the streams: a bound on the live interaction
lifts from `verify_refines`, which names each rep's messages and coins. -/
theorem verify_tableAfter (spec : SessionSpec) (st : Setup) (S : Model.Statement) (hm : 13 ≤ S.m)
    (hmS : st.m = S.m) (hkS : st.kLog = S.kLog) (hfold : FoldRealizes st.fold S) (msId : String)
    (hms : st.merkle = MerkleScheme.sha512 msId) (ptLocal mPts : ℕ) (pts : Session → Fin 2 → Fin mPts → 𝔽)
    (record : String) (proofs : Array ByteArray)
    (hext : ∀ sess, Record.decode spec record = .ok sess →
      ExtraRealizes st S sess.points (Model.extraClaims S ptLocal mPts (pts sess)))
    (h : Flock.verify spec #[st] record proofs = .ok ()) :
    ∃ (sess : Session) (sch : Flock.Schedule) (p₀ p₁ : Proof) (s₀ s₁ : StreamRec) (ms : List Msg) (out : OutC 𝔽 Dig),
      Record.decode spec record = .ok sess ∧ Flock.fast100 st.m = some sch ∧
      decodeProof (shapeOf st sch (Model.extraClaims S ptLocal mPts (pts sess)).length) proofs[0]! = .ok p₀ ∧
      decodeProof (shapeOf st sch (Model.extraClaims S ptLocal mPts (pts sess)).length) proofs[1]! = .ok p₁ ∧
      sess.streams.find? (fun x => x.name == toString spec.tables[0]! ++ toString "/rep" ++ toString 0) = some s₀ ∧
      sess.streams.find? (fun x => x.name == toString spec.tables[0]! ++ toString "/rep" ++ toString 1) = some s₁ ∧
      Plays fOf (tableAfter execArith H512 enc512 S (schedOf sch) hm ptLocal mPts (capDig p₀.cap) (pts sess)) ms
        (s₀.rounds.toList.map (·.coins) ++ s₁.rounds.toList.map (·.coins)) out ∧
      out.accepted = true ∧ out.opens = opensOf p₀ p₁ sch := by
  obtain ⟨sess, sch, p₀, p₁, s₀, s₁, pend₀, pend₁, hsess, hf, hp₀, hp₁, hs₀, hs₁, hpl₀, hpl₁, hv₀, hv₁, hok₀, hok₁⟩ :=
    verify_refines spec st S hm hmS hkS hfold msId hms (fun sess => Model.extraClaims S ptLocal mPts (pts sess)) record
      proofs hext h
  obtain ⟨ms, hint⟩ := Plays.interleave_first hpl₀ hpl₁
  refine ⟨sess, sch, p₀, p₁, s₀, s₁, ms ++ [⟨Opens 𝔽 Dig, opensOf p₀ p₁ sch⟩],
    ⟨pend₀.verdict (opensOf p₀ p₁ sch) && pend₁.verdict (opensOf p₀ p₁ sch) &&
      decide (OpensOK H512 enc512 (capDig p₀.cap) 0 pend₀ (opensOf p₀ p₁ sch) ∧
        OpensOK H512 enc512 (capDig p₀.cap) 1 pend₁ (opensOf p₀ p₁ sch)),
      capDig p₀.cap, (pend₀, pend₁), opensOf p₀ p₁ sch⟩, hsess, hf, hp₀, hp₁, hs₀, hs₁, ?_, ?_, rfl⟩
  · unfold tableAfter
    exact Plays.bind' (f := fun pend : PendL 𝔽 Dig × PendL 𝔽 Dig => Game.send (Opens 𝔽 Dig) fun o =>
        Game.ret (⟨pend.1.verdict o && pend.2.verdict o &&
          decide (OpensOK H512 enc512 (capDig p₀.cap) 0 pend.1 o ∧ OpensOK H512 enc512 (capDig p₀.cap) 1 pend.2 o),
          capDig p₀.cap, pend, o⟩ : OutC 𝔽 Dig)) hint (Plays.send _ (Plays.ret _)) rfl (by simp)
  · simp only [hv₀, hv₁, Bool.true_and, decide_eq_true_eq]
    exact ⟨hok₀, hok₁⟩

end FlockSoundness.Refine
