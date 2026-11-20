module

public import Properties.Flock.EvalBool

/-!
# `evalDef?`'s fuel

One call of `evalDef?` with its recursive call abstracted (`evalStep`), its unfolding (`evalDef?_succ`), and fuel
monotonicity (`evalDef?_mono`): a value found at fuel `n` is the value at every fuel above `n`. A word primitive with no
gate evaluates as its Boolean Definition one fuel lower (`evalDef?_body`).
-/

@[expose] public section

namespace FlockSoundness.EvalBool

open Flock Flock.Program

/-- `b` has every value `a` has. -/
def Refines {α : Type} (a b : Option α) : Prop := ∀ v, a = some v → b = some v

theorem Refines.rfl' {α : Type} (a : Option α) : Refines a a := fun _ h => h

theorem Refines.bind {α β : Type} {a a' : Option α} {k k' : α → Option β} (h1 : Refines a a')
    (h2 : ∀ y, Refines (k y) (k' y)) : Refines (a >>= k) (a' >>= k') := by
  intro v hv
  cases a with
  | none => simp at hv
  | some y =>
    rw [h1 y rfl]
    exact h2 y v hv

theorem Refines.bind_of {α β : Type} {a : Option α} {k k' : α → Option β}
    (h : ∀ y, a = some y → Refines (k y) (k' y)) : Refines (a >>= k) (a >>= k') := by
  intro v hv
  cases ha : a with
  | none => rw [ha] at hv; simp at hv
  | some y =>
    rw [ha] at hv
    exact h y ha v hv

theorem Refines.map {α β : Type} {a a' : Option α} (f : α → β) (h : Refines a a') : Refines (f <$> a) (f <$> a') := by
  intro v hv
  cases a with
  | none => simp at hv
  | some y =>
    rw [h y rfl]
    exact hv

theorem Refines.ite {α : Type} {c : Prop} [Decidable c] {a a' b b' : Option α} (h1 : c → Refines a a')
    (h2 : ¬c → Refines b b') : Refines (if c then a else b) (if c then a' else b') := by
  by_cases hc : c
  · simp only [if_pos hc]
    exact h1 hc
  · simp only [if_neg hc]
    exact h2 hc

theorem Refines.foldlM {α β : Type} {f g : β → α → Option β} :
    ∀ (l : List α), (∀ b, ∀ a ∈ l, Refines (f b a) (g b a)) → ∀ init : β,
      Refines (List.foldlM f init l) (List.foldlM g init l)
  | [], _, _ => Refines.rfl' _
  | a :: l, h, init => by
    simp only [List.foldlM_cons]
    exact Refines.bind (h init a List.mem_cons_self) fun b =>
      Refines.foldlM l (fun b a ha => h b a (List.mem_cons_of_mem _ ha)) b

theorem Refines.mapM {α β : Type} {f g : α → Option β} (h : ∀ a, Refines (f a) (g a)) :
    ∀ l : List α, Refines (List.mapM f l) (List.mapM g l)
  | [] => Refines.rfl' _
  | a :: l => by
    simp only [List.mapM_cons]
    exact Refines.bind (h a) fun _ => Refines.bind (Refines.mapM h l) fun _ => Refines.rfl' _

/-- One call of `evalDef?`, its recursive call `ev`. -/
def evalStep (P : Prog) (ev : Nat → List Bool → Option (List Bool)) (d : Nat) (x : List Bool) :
    Option (List Bool) := do
  let defn ← P.defs[d]?
  if defn.prim then
    match Flock.Lower.gateOf? defn.id with
    | some g => return [gateVal g x]
    | none => return (← ev (← P.bodyOf? defn.key) x)
  let callOf (e : Nat) (ins : List Bool) : Option (List Bool) := ev e ins
  let st := paramStarts defn.params
  let outs ← defn.body.nodes.toList.foldlM (init := (#[] : Array (List Bool))) fun outs nd => do
    let v := refVal x st outs
    if d ≤ nd.fn then none
    let callee ← P.defs[nd.fn]?
    let leaves (i : Nat) : Nat := ((callee.params.getD i default).2).leaves
    match nd.form with
    | .prim | .call => outs.push <$> callOf nd.fn (nd.args.toList.flatMap fun a => argVals v a a.seq.count id)
    | .batch =>
      let ms ← (List.range nd.n).mapM fun m => callOf nd.fn ((List.range nd.args.size).flatMap fun i =>
        let a := nd.args.getD i default
        match a.dims with
        | none => argVals v a a.seq.count id
        | some (dd, L) => argVals v a (leaves i) fun t => (t / L * dd) * L + t % L + m * L)
      return outs.push ms.flatten
    | .scan =>
      let c := leaves 0
      let init := argVals v (nd.args.getD 0 default) c id
      let (carry, ys) ← (List.range nd.n).foldlM (init := (init, ([] : List Bool))) fun (carry, ys) m => do
        let ins := carry ++ ((List.range nd.args.size).drop 1).flatMap fun i =>
          let a := nd.args.getD i default
          if i ≤ nd.nxs then argVals v a (leaves i) fun t => t + m * leaves i else argVals v a (leaves i) id
        let o ← callOf nd.fn ins
        return (o.take c, ys ++ o.drop c)
      return outs.push (carry ++ ys)
  let rs := defn.body.ret.toArray
  return (List.range defn.body.ret.count).map fun t => refVal x st outs (rs.getD t default)

theorem evalDef?_succ (P : Prog) (fuel d : Nat) (x : List Bool) :
    evalDef? P (fuel + 1) d x = evalStep P (evalDef? P fuel) d x := by
  first
  | rfl
  | (rw [evalDef?]; rfl)

/-- `evalStep` at `d` refines wherever its recursive call does on what `d` calls: a primitive's body, or a composite's
callees, each below `d`. -/
theorem evalStep_mono_at (P : Prog) {f g : Nat → List Bool → Option (List Bool)} {d : Nat}
    (hb : ∀ defn, P.defs[d]? = some defn → defn.prim = true → ∀ b, P.bodyOf? defn.key = some b →
      ∀ ins, Refines (f b ins) (g b ins))
    (hc : ∀ defn, P.defs[d]? = some defn → defn.prim = false → ∀ nd ∈ defn.body.nodes.toList, nd.fn < d →
      ∀ ins, Refines (f nd.fn ins) (g nd.fn ins))
    (x : List Bool) : Refines (evalStep P f d x) (evalStep P g d x) := by
  unfold evalStep
  refine Refines.bind_of fun defn hd => ?_
  refine Refines.ite (fun hp => ?_) (fun hp => ?_)
  · generalize Flock.Lower.gateOf? defn.id = og
    cases og with
    | some _ => exact Refines.rfl' _
    | none => exact Refines.bind_of fun b hb' => hb defn hd hp b hb' x
  · have hpf : defn.prim = false := by simpa using hp
    refine Refines.bind ?_ fun _ => Refines.rfl' _
    refine Refines.foldlM _ (fun outs nd hnd => ?_) _
    try dsimp only
    first
      | refine Refines.ite (fun _ => Refines.rfl' _) (fun hlt => ?_)
      | refine Refines.bind_of fun _ hlt => ?_
    have hf := hc defn hd hpf nd hnd (by first | omega | (split at hlt <;> first | omega | simp at hlt))
    refine Refines.bind (Refines.rfl' _) fun callee => ?_
    obtain ⟨form, fn, args, out, axes, n, nxs⟩ := nd
    cases form <;> try dsimp only
    all_goals first
      | exact Refines.map _ (hf _)
      | exact Refines.bind (Refines.mapM (fun _ => hf _) _) fun _ => Refines.rfl' _
      | exact Refines.bind (Refines.foldlM _ (fun acc _ _ => by
            obtain ⟨_, _⟩ := acc
            exact Refines.bind (hf _) fun _ => Refines.rfl' _) _) fun _ => Refines.rfl' _

theorem evalStep_mono (P : Prog) {f g : Nat → List Bool → Option (List Bool)} (h : ∀ e ins, Refines (f e ins) (g e ins))
    (d : Nat) (x : List Bool) : Refines (evalStep P f d x) (evalStep P g d x) :=
  evalStep_mono_at P (fun _ _ _ b _ ins => h b ins) (fun _ _ _ nd _ _ ins => h nd.fn ins) x

/-- A value at fuel `n` is the value at fuel `n + 1`. -/
theorem evalDef?_mono_succ (P : Prog) : ∀ (n d : Nat) (x : List Bool), Refines (evalDef? P n d x) (evalDef? P (n + 1) d x)
  | 0, _, _ => fun _ h => by simp [evalDef?] at h
  | n + 1, d, x => by
    rw [evalDef?_succ, evalDef?_succ P (n + 1)]
    exact evalStep_mono P (fun e ins => evalDef?_mono_succ P n e ins) d x

/-- **Fuel monotonicity**: a value found at fuel `n` is the value at every fuel `m ≥ n`. -/
theorem evalDef?_mono (P : Prog) {n m d : Nat} {x : List Bool} {v : List Bool} (hle : n ≤ m)
    (h : evalDef? P n d x = some v) : evalDef? P m d x = some v := by
  induction hle with
  | refl => exact h
  | step _ ih => exact evalDef?_mono_succ P _ d x v ih

/-- **A word primitive is its Boolean Definition, one fuel lower:** a primitive with no gate whose key `P`'s Boolean map
leads to `b`. -/
theorem evalDef?_body (P : Prog) {p b fuel : Nat} {defn : Defn} {x : List Bool} (hd : P.defs[p]? = some defn)
    (hp : defn.prim = true) (hg : Flock.Lower.gateOf? defn.id = none) (hb : P.bodyOf? defn.key = some b) :
    evalDef? P (fuel + 1) p x = evalDef? P fuel b x := by
  rw [evalDef?_succ]
  simp [evalStep, hd, hp, hg, hb]

/-- `Prog.evalBool` of such a primitive is its body's, wherever the body has a value one fuel below `evalBool`'s. -/
theorem evalBool_body (P : Prog) {p b : Nat} {defn : Defn} {x : List Bool} {v : List Bool} (hd : P.defs[p]? = some defn)
    (hp : defn.prim = true) (hg : Flock.Lower.gateOf? defn.id = none) (hb : P.bodyOf? defn.key = some b)
    (hv : evalDef? P (2 * P.defs.size + 1) b x = some v) : Prog.evalBool P p x = Prog.evalBool P b x := by
  unfold Prog.evalBool
  rw [show 2 * P.defs.size + 2 = (2 * P.defs.size + 1) + 1 from rfl, evalDef?_body P hd hp hg hb, hv,
    evalDef?_mono_succ P _ b x v hv]

/-- A composite Definition's value has one bit for each reference of its return, whatever its input. -/
theorem evalStep_length (P : Prog) {ev : Nat → List Bool → Option (List Bool)} {d : Nat} {defn : Defn}
    {x y : List Bool} (hd : P.defs[d]? = some defn) (hp : defn.prim = false) (h : evalStep P ev d x = some y) :
    y.length = defn.body.ret.count := by
  simp [evalStep, hd, hp, Option.bind_eq_some_iff] at h
  obtain ⟨_, _, rfl⟩ := h
  simp

theorem evalDef?_length (P : Prog) {fuel d : Nat} {defn : Defn} {x y : List Bool} (hd : P.defs[d]? = some defn)
    (hp : defn.prim = false) (h : evalDef? P fuel d x = some y) : y.length = defn.body.ret.count := by
  cases fuel with
  | zero => cases h
  | succ n => rw [evalDef?_succ] at h; exact evalStep_length P hd hp h

end FlockSoundness.EvalBool
