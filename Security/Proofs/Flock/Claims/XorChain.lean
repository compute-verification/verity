import Definitions.Flock.Circuit
import Proofs.Flock.SamplingBoundary

/-!
# A row's XOR chain, in the claims' gates

A row gate of C-Flock's circuit, `(⊕ a) ∧ (⊕ b)` (`FlockSoundness.Audit.Op.row`), lowers each XOR side into the
Boolean profile's gates (`Claims.Sampling.BGate`), which XOR two wires at a time. A list `ℓ` of `k` reads at positions
`p₀, …, p_{k−1}` lowers to `k − 1` binary XOR gates placed from gate `o` (`gates o ps`): `⊕[p₀, p₁]` at `o`, then
`⊕[o + j − 1, p_{j+1}]` at `o + j`, each folding the next read into the one before; one constant `false` gate for no
reads, and no gate for one. The chain's value is on its last wire (`out o ps`): `o + k − 2`, `p₀` for one read, `o` for
none.

`seen_out`: when a set of gates `u` holds the chain, and what `u` sees at each read (`seen`: its own value computed in
`u`, or the committed value outside it) is `v` there, then what `u` sees on the chain's last wire is `xorSum ℓ v`. A gate
of `u` past the chain reads that wire as `seen` (`rd_of_lt`), and `out_lt` places it before the chain's end.
`xor_val` is the same for one n-ary XOR gate.
-/

namespace Claims.Sampling.XorChain

open Claims.Sampling

/-- The binary XOR gates folding the reads `ps` one by one into wire `acc`, placed from gate `o`. -/
def foldGates (acc o : ℕ) : List ℕ → List BGate
  | [] => []
  | p :: ps => .xor [acc, p] :: foldGates o (o + 1) ps

/-- The last wire of `foldGates acc o ps`. -/
def foldOut (acc o : ℕ) : List ℕ → ℕ
  | [] => acc
  | _ :: ps => foldOut o (o + 1) ps

/-- **The XOR chain** of the reads `ps`, placed from gate `o`: `|ps| − 1` binary XOR gates, or one constant `false`
gate for no reads. -/
def gates (o : ℕ) : List ℕ → List BGate
  | [] => [.const false]
  | p :: ps => foldGates p o ps

/-- The chain's last wire. -/
def out (o : ℕ) : List ℕ → ℕ
  | [] => o
  | p :: ps => foldOut p o ps

/-- What the set of gates `u` sees at wire `i` on the committed values `ref`: its value computed in `u` for a gate of
`u`, else as `read` sees it. -/
def seen (C : Prog) (ref : ℕ → Option Bool) (u : List ℕ) (i : ℕ) : Option Bool :=
  if i ∈ u then C.runIn ref u i else C.read ref i

@[simp] theorem length_foldGates (acc o : ℕ) (ps : List ℕ) : (foldGates acc o ps).length = ps.length := by
  induction ps generalizing acc o with
  | nil => rfl
  | cons p ps ih => simp [foldGates, ih]

theorem length_gates (o : ℕ) (ps : List ℕ) : (gates o ps).length = if ps = [] then 1 else ps.length - 1 := by
  cases ps with
  | nil => rfl
  | cons p ps => simp [gates]

/-- A gate `g` of `u` reads an earlier wire `i` as `u` sees it. -/
theorem rd_of_lt (C : Prog) (ref : ℕ → Option Bool) (u : List ℕ) {g i : ℕ} (h : i < g) :
    (if i ∈ u then (if i < g then C.runIn ref u i else none) else C.read ref i) = seen C ref u i := by
  simp [seen, h]

/-- A gate of `u` reading only earlier wires: `u` sees on it its gate applied to what `u` sees on those. -/
theorem seen_gate {C : Prog} {ref : ℕ → Option Bool} {u : List ℕ} {g : ℕ} {x : BGate} (hx : C.gates[g]? = some x)
    (hg : g ∈ u) (hlt : ∀ i ∈ x.reads, i < g) : seen C ref u g = x.val (seen C ref u) (ref g) := by
  rw [seen, ite_eq_left hg, Boundary.runIn_eq, hx, Option.bind_some]
  exact Boundary.val_congr x (fun i hi => rd_of_lt C ref u (hlt i hi)) fun _ => rfl

/-- XOR from the left is XOR from the right. -/
theorem foldl_xor (l : List Bool) (b : Bool) : l.foldl Bool.xor b = Bool.xor b (l.foldr Bool.xor false) := by
  induction l generalizing b with
  | nil => simp
  | cons a l ih => simp [ih]

/-- One n-ary XOR gate of reads `ℓ` computes `xorSum ℓ v`. -/
theorem xor_val {α : Type} (ℓ : List α) (pos : α → ℕ) (v : α → Bool) {rd : ℕ → Option Bool} (own : Option Bool)
    (hp : ∀ x ∈ ℓ, rd (pos x) = some (v x)) :
    (BGate.xor (ℓ.map pos)).val rd own = some (FlockSoundness.Audit.xorSum ℓ v) := by
  have hm : (ℓ.map pos).mapM rd = some (ℓ.map v) := by
    induction ℓ with
    | nil => rfl
    | cons x ℓ ih =>
      simp [List.mapM_cons, hp x List.mem_cons_self, ih fun y hy => hp y (List.mem_cons_of_mem _ hy)]
  simp only [BGate.val, hm, Option.map_some, FlockSoundness.Audit.xorSum]
  rw [foldl_xor, Bool.false_xor]

theorem foldOut_lt {α : Type} (pos : α → ℕ) : ∀ (ℓ : List α) (acc o : ℕ), acc < o →
    foldOut acc o (ℓ.map pos) < o + ℓ.length
  | [], _, _, h => by simpa [foldOut] using h
  | _ :: ℓ, _, o, _ => by
    have := foldOut_lt pos ℓ o (o + 1) (Nat.lt_succ_self o)
    simp only [List.map_cons, foldOut, List.length_cons]
    omega

/-- The chain's last wire is before its end. -/
theorem out_lt {α : Type} (ℓ : List α) (pos : α → ℕ) (o : ℕ) (hlt : ∀ x ∈ ℓ, pos x < o) :
    out o (ℓ.map pos) < o + (gates o (ℓ.map pos)).length := by
  cases ℓ with
  | nil => simp [out, gates]
  | cons x ℓ =>
    have := foldOut_lt pos ℓ (pos x) o (hlt x List.mem_cons_self)
    simpa [out, gates] using this

theorem seen_foldOut {C : Prog} {ref : ℕ → Option Bool} {u : List ℕ} {α : Type} (pos : α → ℕ) (v : α → Bool) :
    ∀ (ℓ : List α) (acc o : ℕ) (b : Bool),
      (∀ j < ℓ.length, C.gates[o + j]? = (foldGates acc o (ℓ.map pos))[j]?) →
      (∀ j < ℓ.length, o + j ∈ u) → acc < o → (∀ x ∈ ℓ, pos x < o) →
      seen C ref u acc = some b → (∀ x ∈ ℓ, seen C ref u (pos x) = some (v x)) →
      seen C ref u (foldOut acc o (ℓ.map pos)) = some ((ℓ.map v).foldl Bool.xor b)
  | [], acc, _, b, _, _, _, _, hb, _ => by simpa [foldOut] using hb
  | x :: ℓ, acc, o, b, hg, hu, hacc, hlt, hb, hp => by
    have hx : C.gates[o]? = some (.xor [acc, pos x]) := by
      simpa [foldGates] using hg 0 (by simp)
    have ho : seen C ref u o = some (Bool.xor b (v x)) := by
      rw [seen_gate hx (by simpa using hu 0 (by simp)) (by
        intro i hi
        simp [BGate.reads] at hi
        rcases hi with rfl | rfl
        · exact hacc
        · exact hlt x List.mem_cons_self)]
      simp [BGate.val, hb, hp x List.mem_cons_self]
    have ih := seen_foldOut pos v ℓ o (o + 1) (Bool.xor b (v x))
      (fun j hj => by
        have e : o + 1 + j = o + (j + 1) := by omega
        rw [e]
        simpa [foldGates] using hg (j + 1) (by simpa using hj))
      (fun j hj => by
        have e : o + 1 + j = o + (j + 1) := by omega
        rw [e]
        exact hu (j + 1) (by simpa using hj))
      (Nat.lt_succ_self o)
      (fun y hy => Nat.lt_succ_of_lt (hlt y (List.mem_cons_of_mem _ hy))) ho
      (fun y hy => hp y (List.mem_cons_of_mem _ hy))
    simpa [foldOut] using ih

/-- **A list lowered into the XOR chain computes its `xorSum`**: when the chain's gates sit from gate `o` in a set of
gates `u`, every read is before `o`, and `u` sees `v x` at each read `pos x`, then `u` sees `xorSum ℓ v` on the chain's
last wire. -/
theorem seen_out {C : Prog} {ref : ℕ → Option Bool} {u : List ℕ} {α : Type} (ℓ : List α) (pos : α → ℕ)
    (v : α → Bool) (o : ℕ)
    (hg : ∀ j < (gates o (ℓ.map pos)).length, C.gates[o + j]? = (gates o (ℓ.map pos))[j]?)
    (hu : ∀ j < (gates o (ℓ.map pos)).length, o + j ∈ u)
    (hlt : ∀ x ∈ ℓ, pos x < o)
    (hp : ∀ x ∈ ℓ, seen C ref u (pos x) = some (v x)) :
    seen C ref u (out o (ℓ.map pos)) = some (FlockSoundness.Audit.xorSum ℓ v) := by
  cases ℓ with
  | nil =>
    have hc : C.gates[o]? = some (.const false) := by simpa [gates] using hg 0 (by simp [gates])
    have hou : o ∈ u := by simpa using hu 0 (by simp [gates])
    show seen C ref u o = some (FlockSoundness.Audit.xorSum [] v)
    rw [seen_gate hc hou (by simp [BGate.reads])]
    rfl
  | cons x ℓ =>
    have h := seen_foldOut pos v ℓ (pos x) o (v x)
      (fun j hj => by simpa [gates] using hg j (by simpa [gates] using hj))
      (fun j hj => hu j (by simpa [gates] using hj))
      (hlt x List.mem_cons_self) (fun y hy => hlt y (List.mem_cons_of_mem _ hy))
      (hp x List.mem_cons_self) (fun y hy => hp y (List.mem_cons_of_mem _ hy))
    rw [List.map_cons, out, h, foldl_xor]
    simp [FlockSoundness.Audit.xorSum]

end Claims.Sampling.XorChain
