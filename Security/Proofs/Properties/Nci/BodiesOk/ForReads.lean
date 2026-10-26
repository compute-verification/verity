import Proofs.Properties.Nci.BodiesOk

/-!
# What `ReadsExact` reads of the decoder and of `BodiesOk`

`decoded`: on a decoded Program, every Definition's facts `ReadsExact.Decoded` names (`decode_spec`): a primitive
returns one leaf, a composite's return has as many leaves as its type, and each node's argument is the sequence of its
parts, none naming a negative position, an argument of no leaves to a member going to a parameter of none.

`bodyOk_reached`: every Definition `Desc.reached` marks, so every one `Desc.analyze` summarizes, is `Resolve.BodyOk`
wherever `Prog.orderViolation` passes. `reached` marks the root and the callees of the composites it marks
(`reached_spec`), so it marks only Definitions of `order_closed`'s call-closed set; a composite of that set is
`body_ok`'s, and a primitive's body is empty.
-/

namespace NciCheck.Properties.SecurityProofs.BodiesOk
open Flock.Program

theorem set!_true (a : Array Bool) (i x : ℕ) (h : (a.set! i true)[x]! = true) : x = i ∨ a[x]! = true := by
  rw [Array.set!_eq_setIfInBounds] at h
  by_cases hx : x < a.size
  · rw [getElem!_pos _ x (by rw [Array.size_setIfInBounds]; exact hx), Array.getElem_setIfInBounds hx] at h
    split at h
    · rename_i e; exact Or.inl e.symm
    · exact Or.inr (by rw [getElem!_pos a x hx]; exact h)
  · rw [getElem!_neg _ x (by rw [Array.size_setIfInBounds]; exact hx)] at h
    cases h

theorem replicate_false (n x : ℕ) (h : (Array.replicate n false)[x]! = true) : False := by
  by_cases hx : x < n
  · rw [getElem!_pos _ x (by simpa using hx), Array.getElem_replicate] at h; cases h
  · rw [getElem!_neg _ x (by simpa using hx)] at h; cases h

/-- **`Desc.reached` marks only Definitions of a call-closed set holding the root**, one mark per Definition. -/
theorem reached_spec (P : Prog) (S : ℕ → Prop) (hS0 : S P.root)
    (hS : ∀ d, S d → (P.defs[d]!).prim = false → ∀ k < (P.defs[d]!).body.nodes.size, S ((P.defs[d]!).body.nodes[k]!).fn) :
    (Desc.reached P).size = P.defs.size ∧ ∀ x, (Desc.reached P)[x]! = true → S x := by
  unfold Desc.reached
  dsimp only
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  let J : Array Bool → Prop := fun b => b.size = P.defs.size ∧ ∀ y, b[y]! = true → S y
  suffices H : ∀ (l : List ℕ) (b : Array Bool), J b → J (forIn (m := Id) l b fun i (r : Array Bool) =>
      if (!r[P.defs.size - 1 - i]!) = true then pure (ForInStep.yield r)
      else if (P.defs[P.defs.size - 1 - i]!).prim = true then pure (ForInStep.yield r)
      else do
        let r ← forIn (P.defs[P.defs.size - 1 - i]!).body.nodes r fun n r => pure (ForInStep.yield (r.set! n.fn true))
        pure (ForInStep.yield r)).run by
    refine H _ _ ⟨by rw [Array.set!_eq_setIfInBounds, Array.size_setIfInBounds, Array.size_replicate], fun y hy => ?_⟩
    rcases set!_true _ _ _ hy with rfl | hy
    · exact hS0
    · exact (replicate_false _ _ hy).elim
  intro l b hb
  refine (forIn_id_inv _ J l (fun _ b => J b) b hb ?_).elim id id
  intro i _ r ⟨hsz, hr⟩
  generalize P.defs.size - 1 - l[i] = d
  by_cases hA : (!r[d]!) = true
  · rw [ite_eq_left hA]; exact ⟨hsz, hr⟩
  rw [ite_eq_right hA]
  by_cases hB : (P.defs[d]!).prim = true
  · rw [ite_eq_left hB]; exact ⟨hsz, hr⟩
  rw [ite_eq_right hB]
  have hSd : S d := hr _ (by simpa using hA)
  have hpd : (P.defs[d]!).prim = false := by simpa using hB
  simp only [Id.run_bind, Id.run_pure]
  rw [← Array.forIn_toList]
  refine (forIn_id_inv _ J _ (fun _ b => J b) r ⟨hsz, hr⟩ ?_).elim id id
  intro k hk s ⟨hsz', hs⟩
  simp only [Id.run_pure]
  refine ⟨by rw [Array.set!_eq_setIfInBounds, Array.size_setIfInBounds]; exact hsz', fun y hy => ?_⟩
  rcases set!_true _ _ _ hy with rfl | hy
  · simp only [Array.length_toList] at hk
    rw [Array.getElem_toList, ← getElem!_pos _ k hk]
    exact hS d hSd hpd k hk
  · exact hs y hy

theorem reached_lt (P : Prog) (S : ℕ → Prop) (hS0 : S P.root)
    (hS : ∀ d, S d → (P.defs[d]!).prim = false → ∀ k < (P.defs[d]!).body.nodes.size, S ((P.defs[d]!).body.nodes[k]!).fn)
    (x : ℕ) (h : (Desc.reached P)[x]! = true) : x < P.defs.size := by
  by_contra hx
  rw [getElem!_neg _ x (by rw [(reached_spec P S hS0 hS).1]; exact hx)] at h
  cases h

/-- A primitive Definition of a decoded Program is `Resolve.BodyOk`: its body is empty. -/
theorem bodyOk_prim (P : Prog) (hok : ∀ i < P.defs.size, DefOk P.defs i P.defs[i]!) (d : ℕ) (hd : d < P.defs.size)
    (hp : (P.defs[d]!).prim = true) : Resolve.BodyOk P d := by
  obtain ⟨hb, -⟩ := (hok d hd).1.1 hp
  refine ⟨fun k hk => ?_, fun t ht => ?_⟩
  · rw [hb] at hk; exact absurd hk (Nat.not_lt_zero _)
  · rw [hb] at ht; exact absurd ht (Nat.not_lt_zero _)

/-- **Every Definition `Desc.reached` marks is `Resolve.BodyOk`**, on a decoded Program whose order
`Prog.orderViolation` passes. -/
theorem bodyOk_reached (bytes : ByteArray) (P : Prog) (hdec : Flock.Program.decode bytes = .ok P)
    (hov : P.orderViolation = none) (d : ℕ) (hd : (Desc.reached P)[d]! = true) :
    d < P.defs.size ∧ Resolve.BodyOk P d := by
  obtain ⟨st, hP, hinv, -⟩ := decode_spec bytes P hdec
  obtain ⟨S, hS0, hS⟩ := order_closed P hov
  have hok : ∀ i < P.defs.size, DefOk P.defs i P.defs[i]! := by rw [hP]; exact hinv.1
  have hS' : ∀ d, S d → (P.defs[d]!).prim = false → ∀ k < (P.defs[d]!).body.nodes.size,
      S ((P.defs[d]!).body.nodes[k]!).fn := fun d hd hp => (hS d hd hp).2
  have hdl := reached_lt P S hS0 hS' d hd
  refine ⟨hdl, ?_⟩
  cases hp : (P.defs[d]!).prim
  · exact body_ok P hok d (hS d ((reached_spec P S hS0 hS').2 d hd) hp).1
  · exact bodyOk_prim P hok d hdl hp

/-- **Every Definition `Desc.analyze` summarizes is `Resolve.BodyOk`**: `ReadsExact.memoOf` summarizes `d` exactly when
`(Desc.reached P)[d]! && d != P.root`. -/
theorem bodyOk_summarized (bytes : ByteArray) (P : Prog) (hdec : Flock.Program.decode bytes = .ok P)
    (hov : P.orderViolation = none) (d : ℕ) (hd : ((Desc.reached P)[d]! && d != P.root) = true) :
    d < P.defs.size ∧ Resolve.BodyOk P d :=
  bodyOk_reached bytes P hdec hov d (by simp only [Bool.and_eq_true] at hd; exact hd.1)

/-- **The root's body is `Resolve.BodyOk`**, on a decoded Program whose order `Prog.orderViolation` passes. -/
theorem bodyOk_root (bytes : ByteArray) (P : Prog) (hdec : Flock.Program.decode bytes = .ok P)
    (hov : P.orderViolation = none) : Resolve.BodyOk P P.root := by
  obtain ⟨st, hP, hinv, hroot⟩ := decode_spec bytes P hdec
  obtain ⟨S, hS0, hS⟩ := order_closed P hov
  have hok : ∀ i < P.defs.size, DefOk P.defs i P.defs[i]! := by rw [hP]; exact hinv.1
  exact body_ok P hok P.root (hS P.root hS0 hroot).1

/-- **The decoder's facts `ReadsExact.Decoded` reads**, with this module's `PartNonneg` (`ReadsExact`'s, by
`partNonneg_iff`, `decoded_reads`). -/
theorem decoded (bytes : ByteArray) (P : Prog) (hdec : Flock.Program.decode bytes = .ok P) :
    ∀ d < P.defs.size, ((P.defs[d]!).prim = true → (P.defs[d]!).outLeaves = 1) ∧
      ((P.defs[d]!).prim = false → (P.defs[d]!).outLeaves = (P.defs[d]!).body.ret.count) ∧
      ∀ k < (P.defs[d]!).body.nodes.size, ∀ i < ((P.defs[d]!).body.nodes[k]!).args.size,
        (((P.defs[d]!).body.nodes[k]!).args[i]!).seq =
          Seq.ofParts (((P.defs[d]!).body.nodes[k]!).args[i]!).seq.parts ∧
        (∀ q ∈ (((P.defs[d]!).body.nodes[k]!).args[i]!).seq.parts.toList, PartNonneg q) ∧
        ∀ e L : ℕ, (((P.defs[d]!).body.nodes[k]!).args[i]!).dims = some (e, L) → L = 0 →
          i < (P.defs[((P.defs[d]!).body.nodes[k]!).fn]!).body.params.size →
          ((P.defs[((P.defs[d]!).body.nodes[k]!).fn]!).body.params[i]!).2.leaves = 0 := by
  obtain ⟨st, hP, hinv, -⟩ := decode_spec bytes P hdec
  have hok : ∀ i < P.defs.size, DefOk P.defs i P.defs[i]! := by rw [hP]; exact hinv.1
  intro d hd
  obtain ⟨hc, hnodes, -⟩ := hok d hd
  refine ⟨fun hp => (hc.1 hp).2.2.1, fun hp => by rw [(hc.2 hp).2.2, (hc.2 hp).2.1], fun k hk i hi => ?_⟩
  obtain ⟨hfn, hnd, hargs⟩ := hnodes k hk
  obtain ⟨⟨ps, hps, hnn⟩, -⟩ := hargs i hi
  have hparts : (((P.defs[d]!).body.nodes[k]!).args[i]!).seq.parts = ps := by
    rw [hps]; exact (Walk.ofParts_spec ps).1
  refine ⟨by rw [hparts]; exact hps, fun q hq => hnn q (by rw [hparts] at hq; exact Array.mem_def.2 hq),
    fun e L hdm hL hip => ?_⟩
  have hfs : ((P.defs[d]!).body.nodes[k]!).fn < P.defs.size := by omega
  have hcc : CalleeOk P.defs[((P.defs[d]!).body.nodes[k]!).fn]! := (hok _ hfs).1
  cases hpc : (P.defs[((P.defs[d]!).body.nodes[k]!).fn]!).prim
  · rw [(hcc.2 hpc).1]; exact hnd.2.2.2.2.2 i hi e L hdm hL
  · rw [(hcc.1 hpc).1] at hip; exact absurd hip (Nat.not_lt_zero _)

end NciCheck.Properties.SecurityProofs.BodiesOk
