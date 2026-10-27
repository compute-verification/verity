import Proofs.Properties.Verity.SampledSim

/-!
# The device's refusals, read in the play

`Drive.canon` reads the play's tagged releases as the record's blocks. Which releases the play's device serves depends
only on the requests (`PlayDevice.served` reads a release's answers, never its coins), so the releases it serves are
exactly those the reduction reads as served, each with the requests before it (`servesNext`, `tagOfServed`):
`served_play`. Each entry it serves names its request's family and root (`served_req`).
-/

namespace Verity.Served

/-! ## `served`, as a fold of one step -/

variable {R A Rt M : Type} (ans : R → A) (req : A → Option (String × Rt)) (cond : String → ℕ → ℕ → Rt → Bool)

/-- One release's step: where its request names a family `l` and a root, the entry `(l, i, root, r)` is served when
`cond` holds at `l`, the family's count `i` so far, the place `k` and the root. -/
def step (acc : List (String × ℕ × Rt × R)) (r : R) : List (String × ℕ × Rt × R) :=
  match req (ans r) with
  | none => acc
  | some (l, root) =>
    let i := (acc.filter (·.1 == l)).length
    let k := acc.length
    if cond l i k root then acc ++ [(l, i, root, r)] else acc

theorem step_cases (acc : List (String × ℕ × Rt × R)) (r : R) :
    step ans req cond acc r = acc ∨ ∃ l i root, step ans req cond acc r = acc ++ [(l, i, root, r)] := by
  unfold step
  split
  · exact .inl rfl
  · dsimp only
    split
    · exact .inr ⟨_, _, _, rfl⟩
    · exact .inl rfl

/-- An entry without its release's coins. -/
def key (e : String × ℕ × Rt × R) : String × ℕ × Rt × A := (e.1, e.2.1, e.2.2.1, ans e.2.2.2)

theorem step_key (acc acc' : List (String × ℕ × Rt × R)) (h : acc.map (key ans) = acc'.map (key ans)) (r r' : R)
    (hr : ans r = ans r') : (step ans req cond acc r).map (key ans) = (step ans req cond acc' r').map (key ans) := by
  have hl : acc.length = acc'.length := by simpa using congrArg List.length h
  have hf : ∀ l : String, (acc.filter (·.1 == l)).length = (acc'.filter (·.1 == l)).length := by
    intro l
    have h1 : (acc.map (key ans)).filter (·.1 == l) = (acc'.map (key ans)).filter (·.1 == l) := by rw [h]
    rw [List.filter_map, List.filter_map] at h1
    simpa [Function.comp_def, key] using congrArg List.length h1
  unfold step
  rw [hr]
  split
  · exact h
  · dsimp only
    rw [hf, hl]
    split
    · simp [h, key, hr]
    · exact h

variable (mk : M → List Bool → R)

/-- The step serves the request `b` after the requests `h`: what it serves grows by one. -/
def sn (h : List (ℕ × M)) (b : M) : Bool :=
  ((((h.map fun z => mk z.2 []) ++ [mk b []]).foldl (step ans req cond) []).length ==
    ((h.map fun z => mk z.2 []).foldl (step ans req cond) []).length + 1)

theorem foldl_annot (hmk : ∀ b l, ans (mk b l) = ans (mk b [])) :
    ∀ (hs : List (Σ m : ℕ, M × (Fin m → Bool))) (h : List (ℕ × M)) (acc : List (String × ℕ × Rt × R)),
      acc.map (key ans) = ((h.map fun z => mk z.2 []).foldl (step ans req cond) []).map (key ans) →
      ((hs.map (Game.recOf mk)).foldl (step ans req cond) acc).map (·.2.2.2) =
        acc.map (·.2.2.2) ++ ((Game.LiveDrive.annot h hs).filter fun z => sn ans req cond mk z.2.1.1 z.2.1.2).map
          fun z => mk z.2.1.2 (List.ofFn z.2.2)
  | [], _, _, _ => by simp [Game.LiveDrive.annot]
  | ⟨m, b, x⟩ :: hs, h, acc, hacc => by
    have hk := step_key ans req cond acc _ hacc (mk b (List.ofFn x)) (mk b []) (hmk b _)
    have hk' : (step ans req cond acc (mk b (List.ofFn x))).map (key ans) =
        (((h ++ [(m, b)]).map fun z => mk z.2 []).foldl (step ans req cond) []).map (key ans) := by
      rw [hk, List.map_append, List.foldl_append]
      rfl
    have ih := foldl_annot hmk hs (h ++ [(m, b)]) _ hk'
    have hl : (step ans req cond acc (mk b (List.ofFn x))).length =
        (((h.map fun z => mk z.2 []) ++ [mk b []]).foldl (step ans req cond) []).length := by
      have := congrArg List.length hk
      simp only [List.length_map] at this
      rw [this, List.foldl_append]
      rfl
    have hl0 : acc.length = ((h.map fun z => mk z.2 []).foldl (step ans req cond) []).length := by
      simpa using congrArg List.length hacc
    show (((hs.map (Game.recOf mk)).foldl (step ans req cond) (step ans req cond acc (mk b (List.ofFn x))))).map
      (·.2.2.2) = _
    rw [ih]
    simp only [Game.LiveDrive.annot]
    by_cases hs' : sn ans req cond mk h b = true
    · simp only [List.filter_cons, hs', ↓reduceIte]
      have hsz : (step ans req cond acc (mk b (List.ofFn x))).length = acc.length + 1 := by
        rw [hl, hl0]
        exact beq_iff_eq.1 hs'
      rcases step_cases ans req cond acc (mk b (List.ofFn x)) with he | ⟨l, i, root, he⟩
      · rw [he] at hsz
        omega
      · rw [he]
        simp
    · simp only [List.filter_cons, hs', Bool.false_eq_true, ↓reduceIte]
      rcases step_cases ans req cond acc (mk b (List.ofFn x)) with he | ⟨l, i, root, he⟩
      · rw [he]
      · exfalso
        apply hs'
        have hsz := congrArg List.length he
        simp only [List.length_append, List.length_cons, List.length_nil] at hsz
        unfold sn
        rw [← hl, ← hl0, hsz]
        simp

/-- **What the step serves is read off the requests**: the releases served, in order, are those the step serves after
the requests before each. -/
theorem served_annot (hmk : ∀ b l, ans (mk b l) = ans (mk b [])) (hs : List (Σ m : ℕ, M × (Fin m → Bool))) :
    ((hs.map (Game.recOf mk)).foldl (step ans req cond) []).map (·.2.2.2) =
      ((Game.LiveDrive.annot [] hs).filter fun z => sn ans req cond mk z.2.1.1 z.2.1.2).map
        fun z => mk z.2.1.2 (List.ofFn z.2.2) := by
  have := foldl_annot ans req cond mk hmk hs [] [] rfl
  rw [List.map_nil, List.nil_append] at this
  exact this

theorem step_req (acc : List (String × ℕ × Rt × R)) (r : R)
    (hacc : ∀ e ∈ acc, req (ans e.2.2.2) = some (e.1, e.2.2.1)) :
    ∀ e ∈ step ans req cond acc r, req (ans e.2.2.2) = some (e.1, e.2.2.1) := by
  unfold step
  split
  · exact hacc
  · rename_i l root hq
    dsimp only
    split
    · intro e he
      rcases List.mem_append.1 he with he | he
      · exact hacc e he
      · rw [List.mem_singleton.1 he]
        exact hq
    · exact hacc

theorem foldl_req : ∀ (rs : List R) (acc : List (String × ℕ × Rt × R)),
    (∀ e ∈ acc, req (ans e.2.2.2) = some (e.1, e.2.2.1)) →
    ∀ e ∈ rs.foldl (step ans req cond) acc, req (ans e.2.2.2) = some (e.1, e.2.2.1)
  | [], _, h => h
  | r :: rs, acc, h => foldl_req rs _ (step_req ans req cond acc r h)

end Verity.Served

namespace Verity

/-- What the play's device checks before it serves a request of family `l`, its `i`th, at place `k`, after `root`
(`PlayDevice.served`): a coin left, and both lines within the reader's cap. -/
def PlayDevice.serveCond (D : PlayDevice) (l : String) (i k : ℕ) (root : ByteArray) : Bool :=
  (D.families.find? (·.1 == l)).any (fun f => decide (i < f.2)) &&
    decide ((PlayDevice.line (PlayDevice.rootJ l i (D.time (2 * k + 1)) root)).length ≤
      SampledProofs.Challenger.MAX_LOG_LINE) &&
    decide ((PlayDevice.line (PlayDevice.releaseJ l i (D.time (2 * k + 2)) root)).length ≤
      SampledProofs.Challenger.MAX_LOG_LINE)

theorem PlayDevice.served_eq (D : PlayDevice) (rs : List Release) :
    D.served rs = rs.foldl (Served.step Release.answers PlayDevice.requestOf D.serveCond) [] := rfl

theorem servesNext_eq (D : PlayDevice) :
    servesNext D = Served.sn Release.answers PlayDevice.requestOf D.serveCond releaseOf := rfl

/-- **The releases the play's device serves are the ones the reduction reads as served** (`servesNext`, each with the
requests before it), in order. -/
theorem served_play (D : PlayDevice) (hs : List (Σ m : ℕ, ByteArray × (Fin m → Bool))) :
    (D.served (hs.map (Game.recOf releaseOf))).map (·.2.2.2) =
      ((Game.LiveDrive.annot [] hs).filter fun z => servesNext D z.2.1.1 z.2.1.2).map
        fun z => releaseOf z.2.1.2 (List.ofFn z.2.2) := by
  rw [PlayDevice.served_eq, servesNext_eq]
  exact Served.served_annot Release.answers PlayDevice.requestOf D.serveCond releaseOf (fun _ _ => rfl) hs

/-- **Each entry the play's device serves names its request's family and root.** -/
theorem served_req (D : PlayDevice) (rs : List Release) :
    ∀ e ∈ D.served rs, PlayDevice.requestOf e.2.2.2.answers = some (e.1, e.2.2.1) := by
  rw [PlayDevice.served_eq]
  exact Served.foldl_req Release.answers PlayDevice.requestOf D.serveCond rs [] (by simp)

end Verity
