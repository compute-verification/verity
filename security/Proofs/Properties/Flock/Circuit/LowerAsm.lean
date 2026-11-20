module

public import Proofs.Properties.Flock.Circuit.LowerDef.BatchScan

/-!
# `lowerDef`'s template Definition computes its Definition, by induction on fuel

Node by node (`LowerDef`'s cases) a Definition's body lowers to a `Work` whose node outputs are the walk's node values
(`nodes_spec`); its returns, committed, are the walk's returns; and a copy of the finished Definition that satisfies its
equations holds them at its return columns (`lowerDef_spec`, `CalleeSem`). Each callee is lowered first, at one less
fuel, or read from `lowerDef`'s caches, which hold only Definitions so lowered (`Inv`). The walk at fuel `F` is enough
for a Definition `f` once `f + 2 ≤ F`: a callee is an earlier Definition, and a gate one call more.

The template's Definitions are `Def.wellFormed`, which `lower`'s `selfCheck` holds every one of them to (`hWF`): its
rows inside its columns, and each call binding as many inputs as its callee has.
-/

@[expose] public section

namespace Flock.Chain.LowerDef

open Flock Flock.Lower Flock.Tmpl Flock.TmplValue
open Flock.Program hiding Form

/-! ## The monad -/

theorem st_bind_ok {α β : Type} {x : M α} {f : α → M β} {s : St} {r : β × St} (h : (x >>= f).run s = .ok r) :
    ∃ a s1, x.run s = .ok (a, s1) ∧ (f a).run s1 = .ok r := by
  rw [StateT.run_bind] at h
  cases hx : x.run s with
  | error e => rw [hx] at h; exact absurd h (by simp [bind, Except.bind])
  | ok p => rw [hx] at h; exact ⟨p.1, p.2, rfl, h⟩

theorem st_throw {α : Type} (e : String) (s : St) (r : α × St) : (throw e : M α).run s ≠ .ok r := by
  simp [throw, throwThe, MonadExceptOf.throw, StateT.run, StateT.lift, bind, Except.bind]

theorem st_throw_bind {α β : Type} (e : String) (f : α → M β) (s : St) (r : β × St) :
    ((throw e : M α) >>= f).run s ≠ .ok r := fun h => by
  obtain ⟨a, s1, h1, -⟩ := st_bind_ok h
  exact st_throw e s _ h1

theorem st_lift_ok {α : Type} {x : V α} {s : St} {r : α × St} (h : (StateT.lift x : M α).run s = .ok r) :
    ∃ a, x = .ok a ∧ r = (a, s) := by
  cases hx : x with
  | error e => rw [hx] at h; exact absurd h (by simp [StateT.lift, StateT.run, bind, Except.bind])
  | ok a =>
    rw [hx] at h
    simp only [StateT.lift, StateT.run, bind, Except.bind, pure, Except.pure, Except.ok.injEq] at h
    exact ⟨a, rfl, h.symm⟩

theorem st_get_run (s : St) : (get : M St).run s = .ok (s, s) := rfl

theorem st_pure_run {α : Type} (a : α) (s : St) : (pure a : M α).run s = .ok (a, s) := rfl

/-- **`nodeStep`'s runs**: node `x` refused unless its callee comes first and its arguments read their parts in order;
then a scan of a composite, a batch of a §9 gate, a gate in place, or a batch or call of a composite. -/
theorem nodeStep_cases {P : Prog} {low : ℕ → M (ℕ × Array ℕ)} {defn : Defn} {d : ℕ} {st : Array ℕ} {w w' : Work}
    {x : Node} {s s' : St} (h : (nodeStep P low defn d st w x).run s = .ok (w', s')) :
    x.fn < d ∧ (∀ a ∈ x.args.toList, seqOk a.seq = true) ∧
      (((x.form == .scan) = true ∧ (P.defs[x.fn]!).prim = false ∧ ∃ e rets s1, (low x.fn).run s = .ok ((e, rets), s1) ∧
          s' = s1 ∧ w.scan st (P.defs[x.fn]!) x e (s1.defs[e]!).cols rets = .ok w') ∨
        ((x.form == .scan) = false ∧ (P.defs[x.fn]!).prim = true ∧ ∃ g, gateOf? (P.defs[x.fn]!).id = some g ∧
          (((x.form == .batch) = true ∧ ∃ e rets s1, (gateDef P x.fn).run s = .ok ((e, rets), s1) ∧ s' = s1 ∧
              w.batch st (P.defs[x.fn]!) x e (s1.defs[e]!).cols rets = .ok w') ∨
            ((x.form == .batch) = false ∧ s' = s ∧ ∃ ops w1 f, w.opsOf st x = .ok ops ∧ w.op g ops = .ok (w1, f) ∧
              w' = { w1 with outs := w1.outs.push (.gate f) }))) ∨
        ((x.form == .scan) = false ∧ (P.defs[x.fn]!).prim = false ∧ ∃ e rets s1, (low x.fn).run s = .ok ((e, rets), s1) ∧
          s' = s1 ∧
          (((x.form == .batch) = true ∧ w.batch st (P.defs[x.fn]!) x e (s1.defs[e]!).cols rets = .ok w') ∨
            ((x.form == .batch) = false ∧ ∃ w1 bs, w.binds st (P.defs[x.fn]!) x = .ok (w1, bs) ∧
              w' = w1.place e (s1.defs[e]!).cols 1 bs rets)))) := by
  unfold nodeStep at h
  dsimp only at h
  split at h
  · exact absurd h (st_throw_bind _ _ _ _)
  rename_i hfn
  split at h
  · exact absurd h (st_throw_bind _ _ _ _)
  rename_i hargs
  have hseq : ∀ a ∈ x.args.toList, seqOk a.seq = true := by
    intro a ha
    have h1 : x.args.toList.all (fun a => seqOk a.seq) = true := by rw [Array.all_toList]; simpa using hargs
    exact List.all_eq_true.mp h1 a ha
  refine ⟨by omega, hseq, ?_⟩
  split at h
  · rename_i hsc
    split at h
    · exact absurd h (st_throw_bind _ _ _ _)
    rename_i hpr
    left
    obtain ⟨⟨e, rets⟩, s1, h1, h2⟩ := st_bind_ok h
    obtain ⟨s2, s3, h3, h4⟩ := st_bind_ok h2
    have h3' : (Except.ok (s1, s1) : Except String (St × St)) = .ok (s2, s3) := h3
    simp only [Except.ok.injEq, Prod.mk.injEq] at h3'
    obtain ⟨rfl, rfl⟩ := h3'
    obtain ⟨a, h5, h6⟩ := st_lift_ok h4
    simp only [Prod.mk.injEq] at h6
    obtain ⟨rfl, rfl⟩ := h6
    exact ⟨by simpa using hsc, by simpa using hpr, e, rets, _, h1, rfl, h5⟩
  rename_i hsc
  split at h
  · rename_i hpr
    split at h
    · rename_i g hg
      split at h
      · rename_i hb
        right; left
        obtain ⟨⟨e, rets⟩, s1, h1, h2⟩ := st_bind_ok h
        obtain ⟨s2, s3, h3, h4⟩ := st_bind_ok h2
        have h3' : (Except.ok (s1, s1) : Except String (St × St)) = .ok (s2, s3) := h3
        simp only [Except.ok.injEq, Prod.mk.injEq] at h3'
        obtain ⟨rfl, rfl⟩ := h3'
        obtain ⟨a, h5, h6⟩ := st_lift_ok h4
        simp only [Prod.mk.injEq] at h6
        obtain ⟨rfl, rfl⟩ := h6
        exact ⟨by simpa using hsc, hpr, g, hg, Or.inl ⟨by simpa using hb, e, rets, _, h1, rfl, h5⟩⟩
      · rename_i hb
        right; left
        obtain ⟨ops, s1, h1, h2⟩ := st_bind_ok h
        obtain ⟨a, h5, h6⟩ := st_lift_ok h1
        simp only [Prod.mk.injEq] at h6
        obtain ⟨rfl, rfl⟩ := h6
        obtain ⟨⟨w1, f⟩, s2, h3, h4⟩ := st_bind_ok h2
        obtain ⟨a', h7, h8⟩ := st_lift_ok h3
        simp only [Prod.mk.injEq] at h8
        obtain ⟨rfl, rfl⟩ := h8
        have h4' : (Except.ok ({ w1 with outs := w1.outs.push (.gate f) }, s2) : Except String (Work × St)) =
          .ok (w', s') := h4
        simp only [Except.ok.injEq, Prod.mk.injEq] at h4'
        obtain ⟨rfl, rfl⟩ := h4'
        exact ⟨by simpa using hsc, hpr, g, hg, Or.inr ⟨by simpa using hb, rfl, ops, w1, f, h5, h7, rfl⟩⟩
    · exact absurd h (st_throw _ _ _)
  · rename_i hpr
    right; right
    obtain ⟨⟨e, rets⟩, s1, h1, h2⟩ := st_bind_ok h
    obtain ⟨s2, s3, h3, h4⟩ := st_bind_ok h2
    have h3' : (Except.ok (s1, s1) : Except String (St × St)) = .ok (s2, s3) := h3
    simp only [Except.ok.injEq, Prod.mk.injEq] at h3'
    obtain ⟨rfl, rfl⟩ := h3'
    refine ⟨by simpa using hsc, by simpa using hpr, e, rets, _, h1, ?_⟩
    split at h4
    · rename_i hb
      obtain ⟨a, h5, h6⟩ := st_lift_ok h4
      simp only [Prod.mk.injEq] at h6
      obtain ⟨rfl, rfl⟩ := h6
      exact ⟨rfl, Or.inl ⟨by simpa using hb, h5⟩⟩
    · rename_i hb
      obtain ⟨⟨w1, bs⟩, s4, h5, h6⟩ := st_bind_ok h4
      obtain ⟨a, h7, h8⟩ := st_lift_ok h5
      simp only [Prod.mk.injEq] at h8
      obtain ⟨rfl, rfl⟩ := h8
      have h6' : (Except.ok (w1.place e (s4.defs[e]!).cols 1 bs rets, s4) : Except String (Work × St)) =
        .ok (w', s') := h6
      simp only [Except.ok.injEq, Prod.mk.injEq] at h6'
      obtain ⟨rfl, rfl⟩ := h6'
      exact ⟨rfl, Or.inr ⟨by simpa using hb, w1, bs, h7, rfl⟩⟩


/-! ## The walk, node by node -/

/-- The walk's step at node `nd` (`walkStep`'s fold on bits), its recursive call `ev`. -/
def walkNode (P : Prog) (ev : ℕ → List Bool → Option (List Bool)) (d : ℕ) (x : List Bool) (st : Array ℕ)
    (outs : Array (List Bool)) (nd : Node) : Option (Array (List Bool)) := do
  let v := refVal x st outs
  if d ≤ nd.fn then failure
  let callee ← P.defs[nd.fn]?
  let leaves (i : Nat) : Nat := ((callee.params.getD i default).2).leaves
  match nd.form with
  | .prim | .call => outs.push <$> ev nd.fn (nd.args.toList.flatMap fun a => argVals v a a.seq.count id)
  | .batch =>
    let ms ← (List.range nd.n).mapM fun m => ev nd.fn ((List.range nd.args.size).flatMap fun i =>
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
      let o ← ev nd.fn ins
      return (o.take c, ys ++ o.drop c)
    return outs.push (carry ++ ys)

theorem walkStep_eq {P : Prog} {ev : ℕ → List Bool → Option (List Bool)} {d : ℕ} {x : List Bool} {defn : Defn}
    (hd : P.defs[d]? = some defn) (hp : defn.prim = false) :
    walkStep Wires.bool P ev d x =
      (defn.body.nodes.toList.foldlM (walkNode P ev d x (paramStarts defn.params)) #[]).bind fun outs =>
        some ((List.range defn.body.ret.count).map fun t =>
          refVal x (paramStarts defn.params) outs (defn.body.ret.toArray.getD t default)) := by
  simp only [walkStep, Wires.bool, hd, Option.bind_eq_bind]
  rw [show ∀ (g : Defn → Option (List Bool)), (some defn).bind g = g defn from fun _ => rfl]
  simp only [hp, Bool.false_eq_true, ite_false]
  rfl

/-! ## What a lowered Definition's template Definition computes -/

/-- **Template Definition `e` of `T` computes Definition `f` of `P`**, returning at its columns `rets`: a copy of it that
satisfies its equations, on inputs as many as it has, holds at `rets[k]` leaf `k` of the walk's value of `f` on them,
at every fuel past `f + 1`. -/
def CalleeSem (P : Prog) (T : Array Def) (f e : ℕ) (rets : Array ℕ) : Prop :=
  (∀ k < rets.size, rets[k]! < (T.getD e default).cols) ∧ (T.getD e default).ins = (P.defs[f]!).paramLeaves ∧
  ∀ F, f + 2 ≤ F → ∀ fuel (v : ℕ → Bool) (inp : List Bool), inp.length = (T.getD e default).ins →
    Holds T true fuel e v inp →
    ∃ l, evalDef? P F f inp = some l ∧ l.length = rets.size ∧ ∀ k < rets.size, v rets[k]! = l.getD k false

/-- `lowerDef`'s caches hold template Definitions of `s` that compute their Definitions. -/
def Inv (P : Prog) (T : Array Def) (s : St) : Prop :=
  (∀ f r, s.done[f]? = some r → CalleeSem P T f r.1 r.2 ∧ r.1 < s.defs.size ∧ r.2.size = (P.defs[f]!).outLeaves) ∧
  (∀ f r, s.gates[f]? = some r → CalleeSem P T f r.1 r.2 ∧ r.1 < s.defs.size)

theorem inv_mono {P : Prog} {T : Array Def} {s s' : St} (h : Inv P T s) (hd : s'.done = s.done) (hg : s'.gates = s.gates)
    (hs : s.defs.size ≤ s'.defs.size) : Inv P T s' :=
  ⟨fun f r hr => by rw [hd] at hr; exact ⟨(h.1 f r hr).1, by have := (h.1 f r hr).2.1; omega, (h.1 f r hr).2.2⟩,
   fun f r hr => by rw [hg] at hr; exact ⟨(h.2 f r hr).1, by have := (h.2 f r hr).2; omega⟩⟩

/-- Work `b` extends work `a` past its node outputs: the same inputs, more rows, calls and columns. -/
def Ext (a b : Work) : Prop :=
  b.ins = a.ins ∧ a.rows.toList <+: b.rows.toList ∧ a.calls.toList <+: b.calls.toList ∧ a.next ≤ b.next

theorem ext_refl (a : Work) : Ext a a := ⟨rfl, List.prefix_refl _, List.prefix_refl _, le_refl _⟩

theorem ext_trans {a b c : Work} (h1 : Ext a b) (h2 : Ext b c) : Ext a c :=
  ⟨h2.1.trans h1.1, h1.2.1.trans h2.2.1, h1.2.2.1.trans h2.2.2.1, le_trans h1.2.2.2 h2.2.2.2⟩

theorem ext_of_wr {a b : Work} (h : WR a b) : Ext a b :=
  ⟨h.1, h.2.2.2.1, by rw [h.2.2.1], h.2.2.2.2⟩

/-- A copy's input atom `b.a`, stepping an even `b.step`, at copy `j`: what `Work.finish`'s binding reads. -/
theorem atomV_step (D : Def) (v : ℕ → Bool) (inp : List Bool) (b : Bind) (hb : b.step % 2 = 0) (j : ℕ) :
    D.refV v inp true ((if b.a % 2 == 0 then b.a / 2 else D.cols + b.a / 2) + j * (b.step / 2)) =
      atomV D v inp (b.a + j * b.step) := by
  unfold atomV
  congr 1
  obtain ⟨s, hs⟩ : ∃ s, b.step = 2 * s := ⟨b.step / 2, by omega⟩
  rw [hs, Nat.mul_div_cancel_left s (by decide), show j * (2 * s) = 2 * (j * s) by ring]
  generalize j * s = q
  by_cases h : b.a % 2 = 0
  · have h2 : (b.a + 2 * q) % 2 = 0 := by omega
    simp only [h, h2, beq_self_eq_true, ite_true]
    omega
  · have h2 : ¬(b.a + 2 * q) % 2 = 0 := by omega
    simp only [h, h2, beq_iff_eq, ite_false]
    omega

theorem foldl_add_sum {α : Type} (f : α → ℕ) :
    ∀ (l : List α) (acc : ℕ), l.foldl (fun n s => n + f s) acc = acc + (l.map f).sum
  | [], acc => by simp
  | a :: l, acc => by
    simp only [List.foldl_cons, List.map_cons, List.sum_cons]
    rw [foldl_add_sum f l]
    ring

theorem boundV_length (c : Call) (g : ℕ → Bool) (j : ℕ) :
    (c.boundV g j).length = c.segs.foldl (fun n s => n + s.reps * s.len) 0 := by
  rw [← Array.foldl_toList, foldl_add_sum (fun s : Seg => s.reps * s.len), Nat.zero_add]
  unfold Call.boundV
  rw [List.length_flatMap]
  congr 1
  apply List.map_congr_left
  intro s _
  simp [List.length_flatMap, List.sum_replicate, List.map_const']

/-- The copies of a finished Definition `W.finish ins`, as `call_copy` and `Def.wellFormed` give them: each copy of each
of its calls satisfies its callee's equations at some fuel, on its bindings, as many as its callee's inputs. -/
def CopyCtx (T : Array Def) (W : Work) (ins : ℕ) (v : ℕ → Bool) (inp : List Bool) : Prop :=
  ∀ p ∈ W.calls.toList, ∀ j < p.count, p.binds.toList.length = (T.getD p.def_ default).ins ∧
    ∃ fuel, Holds T true fuel p.def_ (fun y => v (p.base + j * p.stride + y))
      (p.binds.toList.map fun b =>
        (W.finish ins).refV v inp true ((if b.a % 2 == 0 then b.a / 2 else W.next + b.a / 2) + j * (b.step / 2)))

theorem copyCtx_of {T : Array Def} {e fuel ins : ℕ} {W : Work} {v : ℕ → Bool} {inp : List Bool}
    (hD : T.getD e default = W.finish ins) (hwf : (T.getD e default).wellFormed T e = true)
    (hH : Holds T true (fuel + 1) e v inp) : CopyCtx T W ins v inp := by
  intro p hp j hj
  refine ⟨?_, fuel, call_copy hD hH hp hj⟩
  obtain ⟨c, hc, h1, -, -, -, h5⟩ := finish_call W ins hp
  unfold Def.wellFormed at hwf
  simp only [Bool.and_eq_true, List.all_eq_true, decide_eq_true_eq, beq_iff_eq] at hwf
  rw [hD] at hwf
  have := (hwf.2 c hc).2
  rw [h1] at this
  rw [← this, ← boundV_length c (fun _ => false) 0, h5]
  simp

/-- **A member's copy computes its callee**: a copy `j` of a call of `W` placed at `B`, its callee `e` a template
Definition computing `fn`, binding atoms with the walk's inputs `args`, holds the walk's value of `fn` on them at its
return columns. -/
theorem member_eval {P : Prog} {T : Array Def} {W : Work} {ins : ℕ} {v : ℕ → Bool} {inp : List Bool}
    (hC : CopyCtx T W ins v inp) {fn e : ℕ} {rets : Array ℕ} (hS : CalleeSem P T fn e rets) {F : ℕ}
    (hF : fn + 2 ≤ F) {p : Pending} (hp : p ∈ W.calls.toList) (hpe : p.def_ = e) {cols : ℕ} (hstr : p.stride = cols)
    (hcols : (T.getD e default).cols = cols) {j : ℕ} (hj : j < p.count) {B : ℕ} (hB : p.base + j * cols = B)
    (hev : ∀ b ∈ p.binds.toList, b.step % 2 = 0) {args : List Bool}
    (hargs : p.binds.toList.map (fun b => atomV (W.finish ins) v inp (b.a + j * b.step)) = args)
    (hle : B + cols ≤ W.next) :
    evalDef? P F fn args =
      some ((List.range rets.size).map fun k => atomV (W.finish ins) v inp (2 * (B + rets[k]!))) := by
  obtain ⟨hlen, fuel, hH⟩ := hC p hp j hj
  have hl : p.binds.toList.map (fun b =>
      (W.finish ins).refV v inp true ((if b.a % 2 == 0 then b.a / 2 else W.next + b.a / 2) + j * (b.step / 2))) =
      args := by
    rw [← hargs]
    apply List.map_congr_left
    intro b hb
    exact atomV_step (W.finish ins) v inp b (hev b hb) j
  rw [hl, hpe, hstr, hB] at hH
  rw [hpe] at hlen
  obtain ⟨l, h1, h2, h3⟩ := hS.2.2 F hF fuel _ args (by rw [← hl, List.length_map, hlen]) hH
  rw [h1]
  congr 1
  apply List.ext_getElem (by simp [h2])
  intro k hk1 hk2
  have hk : k < rets.size := by rw [← h2]; exact hk1
  rw [List.getElem_map, List.getElem_range, ← List.getD_eq_getElem l false hk1, ← h3 k hk]
  have hr := hS.1 k hk
  rw [hcols] at hr
  rw [atomV_own (by rw [finish_cols]; omega)]

/-! ## Frames and bounds of a gate in place -/

theorem and_calls (w : Work) (a b : Form) : (w.and a b).1.calls = w.calls ∧ w.next ≤ (w.and a b).1.next := by
  unfold Work.and
  split
  · exact ⟨rfl, le_refl _⟩
  split
  · exact ⟨rfl, le_refl _⟩
  simp [Work.row]

theorem andFold_wr :
    ∀ (l : List Form) (acc : Work × Form), WR acc.1 (l.foldl (fun (acc : Work × Form) o => acc.1.and acc.2 o) acc).1
  | [], _ => wr_refl _
  | o :: l, acc => by
    simp only [List.foldl_cons]
    refine wr_trans ?_ (andFold_wr l _)
    obtain ⟨h1, h2⟩ := and_outs acc.1 acc.2 o
    obtain ⟨h3, h4⟩ := and_calls acc.1 acc.2 o
    exact ⟨h1, h2, h3, and_prefix _ _ _, h4⟩

theorem op_wr {w w' : Work} {g : Gate} {ops : Array Form} {f : Form} (h : w.op g ops = .ok (w', f)) : WR w w' := by
  cases g with
  | const_ b =>
    simp only [Work.op, pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    exact wr_refl _
  | not_ =>
    simp only [Work.op] at h
    cases ho : ops[0]? with
    | none => rw [ho] at h; simp [throw, throwThe, MonadExceptOf.throw] at h
    | some o =>
      rw [ho] at h
      simp only at h
      split at h
      · simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
        obtain ⟨rfl, rfl⟩ := h
        exact wr_refl _
      · simp [throw, throwThe, MonadExceptOf.throw] at h
  | xor_ =>
    simp only [Work.op] at h
    split at h
    · simp [throw, throwThe, MonadExceptOf.throw] at h
    · simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      exact wr_refl _
  | and_ =>
    simp only [Work.op] at h
    cases ho : ops[0]? with
    | none => rw [ho] at h; simp [throw, throwThe, MonadExceptOf.throw] at h
    | some o =>
      rw [ho] at h
      simp only at h
      split at h
      · simp [throw, throwThe, MonadExceptOf.throw] at h
      · simp only [pure, Except.pure, Except.ok.injEq] at h
        rw [← Array.foldl_toList] at h
        have := andFold_wr (ops.extract 1 ops.size).toList (w, o)
        rw [h] at this
        exact this

theorem opsOf_inner_bound {w : Work} {st : Array ℕ} (hB : BoundsOk w) {q : Seq} :
    ∀ (L : List ℕ) (ops ops' : Array Form), (∀ o ∈ ops.toList, FormB w.next o) →
      L.foldlM (fun ops t => do return ops.push (← w.formOf st (← getRef q t))) ops = .ok ops' →
      ∀ o ∈ ops'.toList, FormB w.next o
  | [], ops, ops', h0, h => by
    simp only [List.foldlM_nil, pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact h0
  | t :: L, ops, ops', h0, h => by
    simp only [List.foldlM_cons] at h
    cases hr : getRef q t with
    | error e => simp [hr, bind, Except.bind] at h
    | ok r =>
      cases hfo : w.formOf st r with
      | error e => simp [hr, hfo, bind, Except.bind] at h
      | ok f0 =>
        simp only [hr, hfo, bind, Except.bind, pure, Except.pure] at h
        refine opsOf_inner_bound hB L _ _ (fun o ho => ?_) h
        rw [Array.toList_push, List.mem_append] at ho
        rcases ho with ho | ho
        · exact h0 o ho
        · simp only [List.mem_singleton] at ho
          subst ho
          exact formOf_bound hB hfo

theorem opsOf_outer_bound {w : Work} {st : Array ℕ} (hB : BoundsOk w) :
    ∀ (args : List Arg) (ops ops' : Array Form), (∀ o ∈ ops.toList, FormB w.next o) →
      args.foldlM (fun ops a =>
        (List.range a.seq.count).foldlM (fun ops t => do return ops.push (← w.formOf st (← getRef a.seq t))) ops)
        ops = .ok ops' →
      ∀ o ∈ ops'.toList, FormB w.next o
  | [], ops, ops', h0, h => by
    simp only [List.foldlM_nil, pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact h0
  | a :: args, ops, ops', h0, h => by
    rw [List.foldlM_cons] at h
    cases hi : (List.range a.seq.count).foldlM
        (fun ops t => do return ops.push (← w.formOf st (← getRef a.seq t))) ops with
    | error e => rw [hi] at h; exact absurd h (by simp [bind, Except.bind])
    | ok o1 =>
      rw [hi] at h
      simp only [bind, Except.bind] at h
      exact opsOf_outer_bound hB args o1 ops' (opsOf_inner_bound hB _ ops o1 h0 hi) h

theorem opsOf_bound {w : Work} {st : Array ℕ} {x : Node} {ops : Array Form} (hB : BoundsOk w)
    (h : w.opsOf st x = .ok ops) : ∀ o ∈ ops.toList, FormB w.next o :=
  opsOf_outer_bound hB _ #[] ops (by simp) h

theorem boundsOk_push' {w w' : Work} {o : Out} (h : BoundsOk w) (ho : w'.outs = w.outs.push o) (hn : w.next ≤ w'.next)
    (hb : OutB w'.next o) : BoundsOk w' := by
  have := boundsOk_push (w' := { w' with outs := w.outs }) (o := o) h rfl hn hb
  have he : ({ { w' with outs := w.outs } with outs := w.outs.push o } : Work) = w' := by
    rw [← ho]
  rw [he] at this
  exact this

/-! ## The template's Definitions, the state -/

theorem mapM_some {α β : Type} {f : α → Option β} {g : α → β} :
    ∀ {l : List α}, (∀ a ∈ l, f a = some (g a)) → l.mapM f = some (l.map g)
  | [], _ => rfl
  | a :: l, h => by
    rw [List.mapM_cons, h a (by simp), mapM_some (fun b hb => h b (by simp [hb]))]
    rfl

theorem getD_prefix {α : Type} [Inhabited α] {a b : Array α} (h : a.toList <+: b.toList) {i : ℕ} (hi : i < a.size) :
    b.getD i default = a.getD i default := by
  obtain ⟨l, hl⟩ := h
  rw [Array.getD_eq_getD_getElem?, Array.getD_eq_getD_getElem?, ← Array.getElem?_toList, ← Array.getElem?_toList, ← hl,
    List.getElem?_append_left (by simpa using hi)]

theorem getElem!_getD {α : Type} [Inhabited α] (a : Array α) (i : ℕ) : a[i]! = a.getD i default :=
  getElem!_eq_getD a i

theorem emit_run (D : Def) (rets : Array ℕ) (s : St) :
    (emit D rets).run s = .ok ((s.defs.size, rets), { s with defs := s.defs.push D }) := rfl

theorem modify_run (g : St → St) (s : St) : (modify g : M Unit).run s = .ok ((), g s) := rfl

theorem holds_rows {T : Array Def} {fuel e : ℕ} {v : ℕ → Bool} {inp : List Bool} (h : Holds T true fuel e v inp) :
    ∃ fuel', fuel = fuel' + 1 := by
  cases fuel with
  | zero => exact absurd h (by simp [Holds])
  | succ f => exact ⟨f, rfl⟩

theorem wf_rows {T : Array Def} {e : ℕ} (hwf : (T.getD e default).wellFormed T e = true) :
    ∀ r ∈ (T.getD e default).rows.toList, r.1 < (T.getD e default).cols := by
  unfold Def.wellFormed at hwf
  simp only [Bool.and_eq_true, List.all_eq_true, decide_eq_true_eq, beq_iff_eq] at hwf
  exact hwf.1.2

/-- **`gateDef` writes a template Definition computing its §9 gate**, or reads it from its cache; it only pushes
template Definitions. -/
theorem gateDef_spec {P : Prog} {T : Array Def} {s s' : St} {f : ℕ} {r : ℕ × Array ℕ} (hf : f < P.defs.size)
    (hpr : (P.defs[f]!).prim = true) (hWF : ∀ e < T.size, (T.getD e default).wellFormed T e = true)
    (h : (gateDef P f).run s = .ok (r, s')) :
    s.defs.toList <+: s'.defs.toList ∧ (Inv P T s → s'.defs.toList <+: T.toList →
      CalleeSem P T f r.1 r.2 ∧ r.1 < s'.defs.size ∧ Inv P T s') := by
  unfold gateDef at h
  obtain ⟨s1, s2, h1, h2⟩ := st_bind_ok h
  have h1' : (Except.ok (s, s) : Except String (St × St)) = .ok (s1, s2) := h1
  simp only [Except.ok.injEq, Prod.mk.injEq] at h1'
  obtain ⟨h1a, h1b⟩ := h1'
  subst s1 s2
  split at h2
  · rename_i r0 hr0
    have h2' : (Except.ok (r0, s) : Except String ((ℕ × Array ℕ) × St)) = .ok (r, s') := h2
    simp only [Except.ok.injEq, Prod.mk.injEq] at h2'
    obtain ⟨h2a, h2b⟩ := h2'
    subst r s'
    exact ⟨List.prefix_refl _, fun hI _ => ⟨(hI.2 f r0 hr0).1, (hI.2 f r0 hr0).2, hI⟩⟩
  dsimp only at h2
  split at h2
  case h_2 => exact absurd h2 (st_throw _ _ _)
  rename_i g hg
  obtain ⟨a, s3, h3, h4⟩ := st_bind_ok h2
  have h3' : (StateT.lift (({} : Work).op g ((Array.range (P.defs[f]!).paramLeaves).map Form.input)) :
      M (Work × Form)).run s = .ok (a, s3) := h3
  obtain ⟨a', hop, hq⟩ := st_lift_ok h3'
  simp only [Prod.mk.injEq] at hq
  obtain ⟨hqa, hqb⟩ := hq
  subst a s3
  obtain ⟨r1, s4, h5, h6⟩ := st_bind_ok h4
  rw [emit_run] at h5
  simp only [Except.ok.injEq, Prod.mk.injEq] at h5
  obtain ⟨h5a, h5b⟩ := h5
  subst r1 s4
  obtain ⟨u, s5, h7, h8⟩ := st_bind_ok h6
  rw [modify_run] at h7
  simp only [Except.ok.injEq, Prod.mk.injEq] at h7
  obtain ⟨-, h7b⟩ := h7
  subst s5
  have h8' : (Except.ok (_, _) : Except String ((ℕ × Array ℕ) × St)) = .ok (r, s') := h8
  simp only [Except.ok.injEq, Prod.mk.injEq] at h8'
  obtain ⟨h8a, h8b⟩ := h8'
  subst r s'
  refine ⟨by simp only [Array.toList_push]; exact List.prefix_append _ _, fun hI hT => ?_⟩
  obtain ⟨w, o⟩ := a'
  simp only at hT ⊢
  have hlt : s.defs.size < T.size := by
    have := hT.length_le
    simp only [Array.length_toList, Array.size_push] at this
    omega
  have hTe : T.getD s.defs.size default = (w.commitOwn o).1.finish (P.defs[f]!).paramLeaves := by
    rw [getD_prefix hT (by simp)]
    simp [Array.getD_eq_getD_getElem?]
  have hB := op_bound hop (fun o' ho => by
    rw [Array.toList_map, List.mem_map] at ho
    obtain ⟨k, -, rfl⟩ := ho
    exact formB_input _ _)
  obtain ⟨-, -, -, -, -, hc, -⟩ := commitOwn_spec (V := fun _ => false) w o hB.2
  have hS : CalleeSem P T f s.defs.size #[(w.commitOwn o).2] := by
    refine ⟨fun k hk => ?_, by rw [hTe, finish_ins], fun F hF fuel v inp hlen hH => ?_⟩
    · have hk0 : k = 0 := by simpa using hk
      subst hk0
      rw [hTe, finish_cols]
      simpa using hc
    · obtain ⟨fuel', rfl⟩ := holds_rows hH
      have hwf := hWF _ hlt
      have hcol := wf_rows hwf
      rw [hTe] at hcol hlen
      have hH1 := hH.1
      rw [hTe] at hH1
      obtain ⟨-, hv⟩ := gateTmpl hop (by rw [finish_ins] at hlen; exact hlen) hcol hH1
      obtain ⟨F', rfl⟩ : ∃ F', F = F' + 1 := ⟨F - 1, by omega⟩
      have hc' : P.defs[f]? = some (P.defs[f]!) := by
        rw [getElem!_pos P.defs f hf]; exact Array.getElem?_eq_getElem hf
      refine ⟨[gateVal g inp], eval_gate hc' hpr hg, rfl, fun k hk => ?_⟩
      have hk0 : k = 0 := by simpa using hk
      subst hk0
      simpa using hv
  refine ⟨hS, by simp, ⟨fun f' r' hr' => ?_, fun f' r' hr' => ?_⟩⟩
  · exact ⟨(hI.1 f' r' hr').1, by have := (hI.1 f' r' hr').2.1; simp; omega, (hI.1 f' r' hr').2.2⟩
  · simp only [Std.HashMap.getElem?_insert] at hr'
    split at hr'
    · simp only [Option.some.injEq] at hr'
      subst hr'
      rename_i hff
      simp only [beq_iff_eq] at hff
      subst hff
      exact ⟨hS, by simp⟩
    · exact ⟨(hI.2 f' r' hr').1, by have := (hI.2 f' r' hr').2; simp; omega⟩

/-! ## A node's walk -/

theorem opt_some_bind {α β : Type} (a : α) (g : α → Option β) : (some a >>= g) = g a := rfl

theorem opt_pure_bind {α β : Type} (a : α) (g : α → Option β) : (pure a >>= g) = g a := rfl

/-- The walk's node values of `w`'s outputs, read off their atoms. -/
def outVal (V : ℕ → Bool) : Out → List Bool
  | .gate f => [fval V f]
  | .copies base stride n rets =>
    (List.range (n * rets.size)).map fun j => V (2 * (base + j / rets.size * stride + rets[j % rets.size]!))
  | .scan base stride rets c n => (List.range (c + n * (rets.size - c))).map fun j =>
    if j < c then V (2 * (base + (n - 1) * stride + rets[j]!))
    else V (2 * (base + (j - c) / (rets.size - c) * stride + rets[c + (j - c) % (rets.size - c)]!))

theorem outOk_outVal (V : ℕ → Bool) : ∀ o, OutOk V o (outVal V o)
  | .gate f => by simp [OutOk, outVal]
  | .copies base stride n rets => fun j hj => by simp [outVal, List.getD_eq_getElem?_getD, hj]
  | .scan base stride rets c n => fun j hj =>
    ⟨fun h => by simp [outVal, List.getD_eq_getElem?_getD, hj, h],
     fun h => by simp [outVal, List.getD_eq_getElem?_getD, hj, Nat.not_lt.mpr h]⟩

theorem formsOk_canon (V : ℕ → Bool) (w : Work) : FormsOk V w (w.outs.map (outVal V)) := by
  refine ⟨by simp, fun i h => ?_⟩
  have : (w.outs.map (outVal V)).getD i [] = outVal V w.outs[i] := by simp [Array.getD_eq_getD_getElem?, h]
  rw [this]
  exact outOk_outVal V _

theorem paramsOk_canon (V : ℕ → Bool) (ins : ℕ) : ParamsOk V ins ((List.range ins).map fun k => V (2 * k + 1)) :=
  fun k hk => by simp [List.getD_eq_getElem?_getD, hk]

theorem rowsHold_mono {V : ℕ → Bool} {a b : Work} (h : a.rows.toList <+: b.rows.toList) (hV : RowsHold V b) :
    RowsHold V a := fun r hr => hV r (h.subset hr)

theorem form_batch {x : Node} (h : (x.form == .batch) = true) : x.form = .batch := by
  cases hf : x.form <;> rw [hf] at h <;> first | rfl | exact absurd h (by decide)

theorem form_scan {x : Node} (h : (x.form == .scan) = true) : x.form = .scan := by
  cases hf : x.form <;> rw [hf] at h <;> first | rfl | exact absurd h (by decide)

theorem form_pc {x : Node} (h1 : (x.form == .scan) = false) (h2 : (x.form == .batch) = false) :
    x.form = .prim ∨ x.form = .call := by
  cases hf : x.form <;> rw [hf] at h1 h2
  · exact Or.inl rfl
  · exact Or.inr rfl
  · exact absurd h2 (by decide)
  · exact absurd h1 (by decide)

theorem cols_eq {T : Array Def} {s : St} {e : ℕ} (hT : s.defs.toList <+: T.toList) (he : e < s.defs.size) :
    (T.getD e default).cols = (s.defs[e]!).cols := by
  rw [getElem!_eq_getD, getD_prefix hT he]

theorem take_map_range (g : ℕ → Bool) {c n : ℕ} (h : c ≤ n) : ((List.range n).map g).take c = (List.range c).map g := by
  apply List.ext_getElem (by simp [h])
  intro i h1 h2
  simp

theorem foldlM_range_thread {σ : Type} {f : σ → ℕ → Option σ} {S : ℕ → σ} {n : ℕ}
    (hstep : ∀ m < n, f (S m) m = some (S (m + 1))) : ∀ m ≤ n, ∀ a, a = S 0 → (List.range m).foldlM f a = some (S m)
  | 0, _, a, ha => by rw [ha]; rfl
  | m + 1, hm, a, ha => by
    rw [List.range_succ, List.foldlM_append, foldlM_range_thread hstep m (by omega) a ha, opt_some_bind,
      List.foldlM_cons, hstep m (by omega)]
    rfl

/-- A scan's iteration in the walk, its tail of arguments at iteration `m` given. -/
def scanStep (ev : ℕ → List Bool → Option (List Bool)) (fn : ℕ) (tail : ℕ → List Bool) (c : ℕ) :
    List Bool × List Bool → ℕ → Option (List Bool × List Bool) := fun (carry, ys) m => do
  let o ← ev fn (carry ++ tail m)
  return (o.take c, ys ++ o.drop c)

/-- What node `x` lowered from `w` to `w'` computes: on a finished `W` past `w'` whose copies compute their callees, the
walk at fuel `G` past `d` gives `x` a value, and `w'`'s outputs are the walk's values. -/
def NodeSem (P : Prog) (T : Array Def) (d : ℕ) (st : Array ℕ) (w w' : Work) (x : Node) : Prop :=
  ∀ (W : Work) (ins : ℕ) (v : ℕ → Bool) (inp xs : List Bool) (vals : Array (List Bool)) (G : ℕ), d + 1 ≤ G →
    CopyCtx T W ins v inp → Ext w' W → RowsHold (atomV (W.finish ins) v inp) W →
    FormsOk (atomV (W.finish ins) v inp) w vals → ParamsOk (atomV (W.finish ins) v inp) w.ins xs →
    ∃ val, walkNode P (evalDef? P G) d xs st vals x = some (vals.push val) ∧
      FormsOk (atomV (W.finish ins) v inp) w' (vals.push val)

theorem batch_case {P : Prog} {T : Array Def} {d : ℕ} {st : Array ℕ} {w w' : Work} {x : Node} {e cols : ℕ}
    {rets : Array ℕ} (hfn : x.fn < d) (hfP : x.fn < P.defs.size) (hseq : ∀ a ∈ x.args.toList, seqOk a.seq = true)
    (hform : (x.form == .batch) = true) (hb : w.batch st (P.defs[x.fn]!) x e cols rets = .ok w') :
    Ext w w' ∧ (CalleeSem P T x.fn e rets → (T.getD e default).cols = cols →
      (BoundsOk w → BoundsOk w') ∧ NodeSem P T d st w w' x) := by
  obtain ⟨base, hins, hrows, hcalls, hnb, hnext, houts, -⟩ :=
    batch_node (V := fun _ => false) (formsOk_canon _ w) (paramsOk_canon _ w.ins) hseq hb
  refine ⟨⟨hins, hrows, hcalls, by omega⟩, fun hS hcols => ⟨fun hB => ?_, ?_⟩⟩
  · exact boundsOk_push' hB houts (by omega) (fun _ => ⟨fun k hk => by have := hS.1 k hk; omega, by omega⟩)
  intro W ins v inp xs vals G hG hC hE hR hF hP
  obtain ⟨base', -, -, -, -, hnext', houts', hmem⟩ := batch_node hF hP hseq hb
  have hbb : base' = base := by omega
  subst hbb
  have hRw := hmem (rowsHold_mono hE.2.1 hR)
  have hev : ∀ m ∈ List.range x.n, evalDef? P G x.fn (batchArgs (refVal xs st vals) (P.defs[x.fn]!) x m) =
      some ((List.range rets.size).map fun k => atomV (W.finish ins) v inp (2 * (base' + m * cols + rets[k]!))) := by
    intro m hm
    rw [List.mem_range] at hm
    obtain ⟨p, hp, hpe, hstr, j, hj, hB, hev2, hargs⟩ := hRw m hm
    refine member_eval hC hS (by omega) (hE.2.2.1.subset hp) hpe hstr hcols hj hB hev2 hargs ?_
    have h1 := hE.2.2.2
    have h2 : (m + 1) * cols ≤ x.n * cols := Nat.mul_le_mul_right _ hm
    rw [Nat.succ_mul] at h2
    omega
  refine ⟨((List.range x.n).map fun m =>
      (List.range rets.size).map fun k => atomV (W.finish ins) v inp (2 * (base' + m * cols + rets[k]!))).flatten,
    ?_, batch_forms hF houts' (fun m _ => by simp) (fun m _ k hk => by simp [List.getD_eq_getElem?_getD, hk])⟩
  have hcal : P.defs[x.fn]? = some (P.defs[x.fn]!) := by
    rw [getElem!_pos P.defs x.fn hfP]; exact Array.getElem?_eq_getElem hfP
  simp only [walkNode, if_neg (show ¬ d ≤ x.fn by omega), hcal, opt_some_bind, opt_pure_bind, form_batch hform]
  show (List.range x.n).mapM (fun m => evalDef? P G x.fn (batchArgs (refVal xs st vals) (P.defs[x.fn]!) x m)) >>= _ = _
  rw [mapM_some hev]
  rfl

theorem scan_case {P : Prog} {T : Array Def} {d : ℕ} {st : Array ℕ} {w w' : Work} {x : Node} {e cols : ℕ}
    {rets : Array ℕ} (hfn : x.fn < d) (hfP : x.fn < P.defs.size) (hseq : ∀ a ∈ x.args.toList, seqOk a.seq = true)
    (hform : (x.form == .scan) = true) (hs : w.scan st (P.defs[x.fn]!) x e cols rets = .ok w') :
    Ext w w' ∧ (CalleeSem P T x.fn e rets → (T.getD e default).cols = cols →
      (BoundsOk w → BoundsOk w') ∧ NodeSem P T d st w w' x) := by
  obtain ⟨base, hins, hrows, hcalls, hnb, hnext, houts, hn, ha, hc, -⟩ :=
    scan_node (V := fun _ => false) (formsOk_canon _ w) (paramsOk_canon _ w.ins) hseq hs
  refine ⟨⟨hins, hrows, hcalls, by omega⟩, fun hS hcols => ⟨fun hB => ?_, ?_⟩⟩
  · exact boundsOk_push' hB houts (by omega) ⟨hn, hc, fun k hk => by have := hS.1 k hk; omega, by omega⟩
  intro W ins v inp xs vals G hG hC hE hR hF hP
  obtain ⟨base', -, -, -, -, hnext', houts', -, -, -, hmem⟩ := scan_node hF hP hseq hs
  have hbb : base' = base := by omega
  subst hbb
  have hRw := hmem (rowsHold_mono hE.2.1 hR)
  generalize hV : atomV (W.finish ins) v inp = V at hRw hF ⊢
  set c := ((P.defs[x.fn]!).params[0]!.2).leaves with hcdef
  have hc' : ((P.defs[x.fn]!).params.getD 0 default).2.leaves = c := by
    rw [hcdef, getElem!_eq_getD ((P.defs[x.fn]!).params) 0]
  obtain ⟨L, hL⟩ : ∃ L : ℕ → List Bool, ∀ m, L m = (List.range rets.size).map fun k => V (2 * (base' + m * cols + rets[k]!)) :=
    ⟨_, fun _ => rfl⟩
  have hev : ∀ m < x.n, evalDef? P G x.fn
      (if m = 0 then (scanIdx (P.defs[x.fn]!) (List.range x.args.size)).map
          (fun it => scanArg xs st vals (P.defs[x.fn]!) x it.1 it.2 0)
        else (List.range c).map (fun k => V (2 * (base' + (m - 1) * cols + rets[k]!))) ++
          ((scanIdx (P.defs[x.fn]!) (List.range x.args.size)).drop c).map
            (fun it => scanArg xs st vals (P.defs[x.fn]!) x it.1 it.2 m)) = some (L m) := by
    intro m hm
    obtain ⟨p, hp, hpe, hstr, j, hj, hB, hev2, hargs⟩ := hRw m hm
    rw [hL]
    subst hV
    refine member_eval hC hS (by omega) (hE.2.2.1.subset hp) hpe hstr hcols hj hB hev2 hargs ?_
    have h1 := hE.2.2.2
    have h2 : (m + 1) * cols ≤ x.n * cols := Nat.mul_le_mul_right _ hm
    rw [Nat.succ_mul] at h2
    omega
  obtain ⟨A, hA⟩ : ∃ A : ℕ → List Bool, ∀ m, A m = if m = 0 then
      argVals (refVal xs st vals) (x.args.getD 0 default) c id else (L (m - 1)).take c := ⟨_, fun _ => rfl⟩
  obtain ⟨B, hB⟩ : ∃ B : ℕ → List Bool, ∀ m, B m = ((List.range m).map fun i => (L i).drop c).flatten :=
    ⟨_, fun _ => rfl⟩
  have hstep : ∀ m < x.n, scanStep (evalDef? P G) x.fn (scanWalkTail (refVal xs st vals) (P.defs[x.fn]!) x) c
      (A m, B m) m = some (A (m + 1), B (m + 1)) := by
    intro m hm
    have hins : A m ++ scanWalkTail (refVal xs st vals) (P.defs[x.fn]!) x m =
        (if m = 0 then (scanIdx (P.defs[x.fn]!) (List.range x.args.size)).map
          (fun it => scanArg xs st vals (P.defs[x.fn]!) x it.1 it.2 0)
        else (List.range c).map (fun k => V (2 * (base' + (m - 1) * cols + rets[k]!))) ++
          ((scanIdx (P.defs[x.fn]!) (List.range x.args.size)).drop c).map
            (fun it => scanArg xs st vals (P.defs[x.fn]!) x it.1 it.2 m)) := by
      rw [hA]
      by_cases h0 : m = 0
      · subst h0
        rw [scanIns0_eq xs st vals _ x ha, hc']
        simp only [↓reduceIte]
      · simp only [if_neg h0]
        rw [hcdef, scanTail_eq xs st vals _ x ha m, ← hcdef, hL, take_map_range _ (by omega)]
    show (evalDef? P G x.fn (A m ++ scanWalkTail (refVal xs st vals) (P.defs[x.fn]!) x m) >>= fun o =>
      pure (o.take c, B m ++ o.drop c)) = _
    rw [hins, hev m hm, opt_some_bind]
    show some _ = some _
    rw [hA (m + 1), hB (m + 1), hB m, if_neg (by omega), Nat.add_sub_cancel, List.range_succ, List.map_append,
      List.flatten_append]
    simp
  have hfold := foldlM_range_thread (S := fun m => (A m, B m)) hstep x.n le_rfl
    (argVals (refVal xs st vals) (x.args.getD 0 default) c id, []) (by rw [hA 0, hB 0]; simp)
  refine ⟨A x.n ++ B x.n, ?_, ?_⟩
  · have hcal : P.defs[x.fn]? = some (P.defs[x.fn]!) := by
      rw [getElem!_pos P.defs x.fn hfP]; exact Array.getElem?_eq_getElem hfP
    simp only [walkNode, if_neg (show ¬ d ≤ x.fn by omega), hcal, opt_some_bind, opt_pure_bind, form_scan hform]
    rw [hc']
    show List.foldlM (scanStep (evalDef? P G) x.fn (scanWalkTail (refVal xs st vals) (P.defs[x.fn]!) x) c)
      (argVals (refVal xs st vals) (x.args.getD 0 default) c id, []) (List.range x.n) >>= _ = _
    rw [hfold]
    rfl
  · rw [hA, hB, if_neg (by omega)]
    exact scan_forms hF houts' hn hc (l := L) (fun m _ => by rw [hL]; simp)
      (fun m _ k hk => by rw [hL]; simp [List.getD_eq_getElem?_getD, hk])

/-- **A node lowered computes the walk's value of it**, its callees lowered by `low` before it. -/
theorem node_spec {P : Prog} {T : Array Def} {low : ℕ → M (ℕ × Array ℕ)} {defn : Defn} {d : ℕ} {st : Array ℕ}
    {w w' : Work} {x : Node} {s s' : St}
    (hWF : ∀ e < T.size, (T.getD e default).wellFormed T e = true) (hd : d < P.defs.size)
    (hlow : ∀ f (s s1 : St) (r : ℕ × Array ℕ), f < d → (low f).run s = .ok (r, s1) →
      s.defs.toList <+: s1.defs.toList ∧ (Inv P T s → s1.defs.toList <+: T.toList →
        CalleeSem P T f r.1 r.2 ∧ r.1 < s1.defs.size ∧ Inv P T s1))
    (h : (nodeStep P low defn d st w x).run s = .ok (w', s')) :
    Ext w w' ∧ s.defs.toList <+: s'.defs.toList ∧ (Inv P T s → s'.defs.toList <+: T.toList →
      Inv P T s' ∧ (BoundsOk w → BoundsOk w') ∧ NodeSem P T d st w w' x) := by
  obtain ⟨hfn, hseq, hcase⟩ := nodeStep_cases h
  have hfP : x.fn < P.defs.size := by omega
  have hcal : P.defs[x.fn]? = some (P.defs[x.fn]!) := by
    rw [getElem!_pos P.defs x.fn hfP]; exact Array.getElem?_eq_getElem hfP
  rcases hcase with ⟨hsc, -, e, rets, s1, hl, rfl, hs⟩ | ⟨hsc, hpr, g, hg, ⟨hbt, e, rets, s1, hgd, rfl, hb⟩ |
      ⟨hbt, rfl, ops, w1, f, hops, hop, rfl⟩⟩ | ⟨hsc, -, e, rets, s1, hl, rfl, ⟨hbt, hb⟩ | ⟨hbt, w1, bs, hb, rfl⟩⟩
  · obtain ⟨hpre, hsem⟩ := hlow x.fn s _ (e, rets) hfn hl
    obtain ⟨hE, hsc'⟩ := scan_case hfn hfP hseq hsc hs (T := T)
    refine ⟨hE, hpre, fun hI hT => ?_⟩
    obtain ⟨hS, he, hI'⟩ := hsem hI hT
    exact ⟨hI', hsc' hS (cols_eq hT he)⟩
  · obtain ⟨hpre, hsem⟩ := gateDef_spec hfP hpr hWF hgd
    obtain ⟨hE, hbc⟩ := batch_case hfn hfP hseq hbt hb (T := T)
    refine ⟨hE, hpre, fun hI hT => ?_⟩
    obtain ⟨hS, he, hI'⟩ := hsem hI hT
    exact ⟨hI', hbc hS (cols_eq hT he)⟩
  · have hwr := op_wr hop
    refine ⟨⟨hwr.1, hwr.2.2.2.1, by rw [hwr.2.2.1], hwr.2.2.2.2⟩, List.prefix_refl _, fun hI _ => ⟨hI, fun hB => ?_, ?_⟩⟩
    · have hb := op_bound hop (opsOf_bound hB hops)
      exact boundsOk_push' (o := .gate f) hB (by simp only; rw [hwr.2.1]) hb.1 hb.2
    intro W ins v inp xs vals G hG hC hE hR hF hP
    obtain ⟨-, -, hFO⟩ := gate_node hF hP hseq hops hop (rowsHold_mono hE.2.1 hR)
    refine ⟨_, ?_, hFO⟩
    obtain ⟨G', rfl⟩ : ∃ G', G = G' + 1 := ⟨G - 1, by omega⟩
    rcases form_pc hsc hbt with hf | hf <;>
    · simp only [walkNode, if_neg (show ¬ d ≤ x.fn by omega), hcal, opt_some_bind, opt_pure_bind, hf]
      show (vals.push <$> evalDef? P (G' + 1) x.fn
        (x.args.toList.flatMap fun a => argVals (refVal xs st vals) a a.seq.count id)) = _
      rw [eval_gate hcal hpr hg]
      rfl
  · obtain ⟨hpre, hsem⟩ := hlow x.fn s _ (e, rets) hfn hl
    obtain ⟨hE, hbc⟩ := batch_case hfn hfP hseq hbt hb (T := T)
    refine ⟨hE, hpre, fun hI hT => ?_⟩
    obtain ⟨hS, he, hI'⟩ := hsem hI hT
    exact ⟨hI', hbc hS (cols_eq hT he)⟩
  · obtain ⟨hpre, hsem⟩ := hlow x.fn s _ (e, rets) hfn hl
    have hwr := binds_wr hb
    refine ⟨⟨by simp only [Work.place]; exact hwr.1, by simp only [Work.place]; exact hwr.2.2.2.1,
      by simp only [Work.place, Array.toList_push]; rw [hwr.2.2.1]; exact List.prefix_append _ _,
      by simp only [Work.place]; have := hwr.2.2.2.2; omega⟩, hpre, fun hI hT => ?_⟩
    obtain ⟨hS, he, hI'⟩ := hsem hI hT
    have hcols := cols_eq hT he
    refine ⟨hI', fun hB => ?_, ?_⟩
    · refine boundsOk_push' (o := .copies w1.next (s'.defs[e]!).cols 1 rets) hB (by simp only [Work.place]; rw [hwr.2.1])
        (by simp only [Work.place]; have := hwr.2.2.2.2; omega) (fun _ => ⟨fun k hk => ?_, by simp only [Work.place]; omega⟩)
      have := hS.1 k hk
      have hc2 := hcols
      dsimp only at this hc2
      omega
    intro W ins v inp xs vals G hG hC hE hR hF hP
    have hR1 : RowsHold (atomV (W.finish ins) v inp) w1 :=
      rowsHold_mono (by have := hE.2.1; simpa only [Work.place] using this) hR
    obtain ⟨-, -, -, hmap, hFO⟩ := call_node (e := e) (cols := (s'.defs[e]!).cols) (rets := rets)
      (l := (List.range rets.size).map fun k => atomV (W.finish ins) v inp (2 * (w1.next + rets[k]!)))
      hF hP hseq hbt hb hR1 (fun k hk => by simp [List.getD_eq_getElem?_getD, hk])
    refine ⟨_, ?_, hFO⟩
    have hp : ({ def_ := e, count := 1, base := w1.next, stride := (s'.defs[e]!).cols, binds := bs } : Pending) ∈
        W.calls.toList := hE.2.2.1.subset (by simp [Work.place])
    have hev : ∀ b ∈ bs.toList, b.step % 2 = 0 := by
      intro b hb'
      have hm : (atomV (W.finish ins) v inp b.a, b.step) ∈
          (x.args.toList.flatMap fun a => argVals (refVal xs st vals) a a.seq.count id).map (·, 0) :=
        hmap ▸ List.mem_map_of_mem hb'
      obtain ⟨y, -, hy⟩ := List.mem_map.mp hm
      simp only [Prod.mk.injEq] at hy
      omega
    have hargs : bs.toList.map (fun b => atomV (W.finish ins) v inp (b.a + 0 * b.step)) =
        x.args.toList.flatMap fun a => argVals (refVal xs st vals) a a.seq.count id := by
      have := congrArg (List.map Prod.fst) hmap
      simpa [List.map_map, Function.comp_def] using this
    have hle : w1.next + (s'.defs[e]!).cols ≤ W.next := by
      have := hE.2.2.2
      simp only [Work.place] at this
      omega
    have hme := member_eval hC hS (F := G) (by omega) hp rfl rfl hcols (j := 0) Nat.one_pos (B := w1.next) (by simp) hev hargs hle
    rcases form_pc hsc hbt with hf | hf <;>
    · simp only [walkNode, if_neg (show ¬ d ≤ x.fn by omega), hcal, opt_some_bind, opt_pure_bind, hf]
      show (vals.push <$> evalDef? P G x.fn
        (x.args.toList.flatMap fun a => argVals (refVal xs st vals) a a.seq.count id)) = _
      rw [hme]
      rfl

/-! ## A Definition's nodes, its returns, and `lowerDef` -/

/-- What nodes lowered from `w` to `w'` compute: the walk's fold over them, as `NodeSem` for one. -/
def NodesSem (P : Prog) (T : Array Def) (d : ℕ) (st : Array ℕ) (w w' : Work) (L : List Node) : Prop :=
  ∀ (W : Work) (ins : ℕ) (v : ℕ → Bool) (inp xs : List Bool) (vals : Array (List Bool)) (G : ℕ), d + 1 ≤ G →
    CopyCtx T W ins v inp → Ext w' W → RowsHold (atomV (W.finish ins) v inp) W →
    FormsOk (atomV (W.finish ins) v inp) w vals → ParamsOk (atomV (W.finish ins) v inp) w.ins xs →
    ∃ vals', L.foldlM (walkNode P (evalDef? P G) d xs st) vals = some vals' ∧
      FormsOk (atomV (W.finish ins) v inp) w' vals'

theorem nodes_spec {P : Prog} {T : Array Def} {low : ℕ → M (ℕ × Array ℕ)} {defn : Defn} {d : ℕ} {st : Array ℕ}
    (hWF : ∀ e < T.size, (T.getD e default).wellFormed T e = true) (hd : d < P.defs.size)
    (hlow : ∀ f (s s1 : St) (r : ℕ × Array ℕ), f < d → (low f).run s = .ok (r, s1) →
      s.defs.toList <+: s1.defs.toList ∧ (Inv P T s → s1.defs.toList <+: T.toList →
        CalleeSem P T f r.1 r.2 ∧ r.1 < s1.defs.size ∧ Inv P T s1)) :
    ∀ (L : List Node) (w w' : Work) (s s' : St), (L.foldlM (nodeStep P low defn d st) w).run s = .ok (w', s') →
      Ext w w' ∧ s.defs.toList <+: s'.defs.toList ∧ (Inv P T s → s'.defs.toList <+: T.toList →
        Inv P T s' ∧ (BoundsOk w → BoundsOk w') ∧ NodesSem P T d st w w' L)
  | [], w, w', s, s', h => by
    have h' : (Except.ok (w, s) : Except String (Work × St)) = .ok (w', s') := h
    simp only [Except.ok.injEq, Prod.mk.injEq] at h'
    obtain ⟨h1, h2⟩ := h'
    subst w' s'
    refine ⟨ext_refl w, List.prefix_refl _, fun hI _ => ⟨hI, id, ?_⟩⟩
    intro W ins v inp xs vals G _ _ _ _ hF _
    exact ⟨vals, rfl, hF⟩
  | x :: L, w, w', s, s', h => by
    rw [List.foldlM_cons] at h
    obtain ⟨w1, s1, h1, h2⟩ := st_bind_ok h
    obtain ⟨hE1, hp1, hc1⟩ := node_spec hWF hd hlow h1
    obtain ⟨hE2, hp2, hc2⟩ := nodes_spec hWF hd hlow L w1 w' s1 s' h2
    refine ⟨ext_trans hE1 hE2, hp1.trans hp2, fun hI hT => ?_⟩
    obtain ⟨hI1, hB1, hS1⟩ := hc1 hI (hp2.trans hT)
    obtain ⟨hI2, hB2, hS2⟩ := hc2 hI1 hT
    refine ⟨hI2, fun hB => hB2 (hB1 hB), ?_⟩
    intro W ins v inp xs vals G hG hC hE hR hF hP
    obtain ⟨val, hw1, hF1⟩ := hS1 W ins v inp xs vals G hG hC (ext_trans hE2 hE) hR hF hP
    obtain ⟨vals', hw2, hF2⟩ := hS2 W ins v inp xs (vals.push val) G hG hC hE hR hF1 (by rw [hE1.1]; exact hP)
    refine ⟨vals', ?_, hF2⟩
    rw [List.foldlM_cons, hw1]
    exact hw2

theorem ex_bind_ok {α β : Type} {x : Except String α} {f : α → Except String β} {b : β} (h : (x >>= f) = .ok b) :
    ∃ a, x = .ok a ∧ f a = .ok b := by
  cases hx : x with
  | error e => rw [hx] at h; exact absurd h (by simp [bind, Except.bind])
  | ok a => rw [hx] at h; exact ⟨a, rfl, h⟩

theorem push_lt {a : Array ℕ} {y k : ℕ} (hk : k < a.size) : (a.push y)[k]! = a[k]! := by
  rw [getElem!_pos (a.push y) k (by simp; omega), getElem!_pos a k hk, Array.getElem_push_lt]

theorem push_eq (a : Array ℕ) (y : ℕ) : (a.push y)[a.size]! = y := by
  rw [getElem!_pos (a.push y) a.size (by simp), Array.getElem_push_eq]

/-- **A Definition's returns**, committed to own columns after its nodes: each holds the walk's value of its reference. -/
theorem rets_spec {w w' : Work} {st : Array ℕ} {ret : Seq} {rets : Array ℕ} (hq : seqOk ret = true)
    (hB : BoundsOk w) (h : w.rets st ret = .ok (w', rets)) :
    WR w w' ∧ rets.size = ret.count ∧ (∀ k < rets.size, rets[k]! < w'.next) ∧
      ∀ (V : ℕ → Bool) (xs : List Bool) (vals : Array (List Bool)), FormsOk V w vals → ParamsOk V w.ins xs →
        RowsHold V w' → ∀ k < rets.size, V (2 * rets[k]!) = refVal xs st vals (ret.toArray.getD k default) := by
  unfold Work.rets at h
  generalize ret.count = n at h ⊢
  induction n generalizing w' rets with
  | zero =>
    simp only [List.range_zero, List.foldlM_nil, pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
    obtain ⟨h1, h2⟩ := h
    subst w' rets
    exact ⟨wr_refl _, rfl, fun k hk => absurd hk (by simp), fun _ _ _ _ _ _ k hk => absurd hk (by simp)⟩
  | succ n ih =>
    rw [List.range_succ, List.foldlM_append] at h
    obtain ⟨⟨wm, rm⟩, hi, h2⟩ := ex_bind_ok h
    obtain ⟨hwr, hsz, hlt, hval⟩ := ih hi
    rw [List.foldlM_cons] at h2
    obtain ⟨⟨w2, r2⟩, h3, h4⟩ := ex_bind_ok h2
    simp only [List.foldlM_nil, pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h4
    obtain ⟨h4a, h4b⟩ := h4
    subst w2 r2
    dsimp only at h3
    obtain ⟨r, hr, h5⟩ := ex_bind_ok h3
    obtain ⟨f0, hf0, h6⟩ := ex_bind_ok h5
    have h6' : ((wm.commitOwn f0).1, rm.push (wm.commitOwn f0).2) = (w', rets) := by
      revert h6
      generalize wm.commitOwn f0 = pr
      obtain ⟨a, b⟩ := pr
      intro h6
      simpa [pure, Except.pure] using h6
    simp only [Prod.mk.injEq] at h6'
    obtain ⟨h6a, h6b⟩ := h6'
    subst w' rets
    obtain ⟨hnr, rfl⟩ := getRef_ok hr
    have hBm : BoundsOk wm := boundsOk_mono hB hwr.2.1 hwr.2.2.2.2
    have hFB := formOf_bound hBm hf0
    obtain ⟨c1, c2, c3, c4, c5, c6, -⟩ := commitOwn_spec (V := fun _ => false) wm f0 hFB
    refine ⟨wr_trans hwr ⟨c1, c2, c3, c4, c5⟩, by simp [hsz], fun k hk => ?_, fun V xs vals hF hP hR k hk => ?_⟩
    · simp only [Array.size_push] at hk
      by_cases hkn : k < rm.size
      · rw [push_lt hkn]; have := hlt k hkn; omega
      · have hk' : k = rm.size := by omega
        subst hk'
        rw [push_eq]; exact c6
    · simp only [Array.size_push] at hk
      by_cases hkn : k < rm.size
      · rw [push_lt hkn]
        exact hval V xs vals hF hP (rowsHold_mono c4 hR) k hkn
      · have hk' : k = rm.size := by omega
        subst hk'
        obtain ⟨-, -, -, -, -, -, c7⟩ := commitOwn_spec (V := V) wm f0 hFB
        rw [push_eq, c7 hR, formOf_val (formsOk_wr hF hwr) (paramsOk_wr hP hwr) hf0, refAt_eq hq (by omega), hsz]

theorem paramStarts_eq (params : Array (String × Ty)) : paramStarts params = starts params := rfl

theorem evalDef?_succ (P : Prog) (F d : ℕ) (x : List Bool) :
    evalDef? P (F + 1) d x = walkStep Wires.bool P (evalDef? P F) d x := rfl

/-- **`lowerDef` writes template Definitions computing their Definitions**: from a state whose caches compute theirs, the
template Definition it returns computes Definition `d` at its return columns, as many as `d`'s output leaves. It only
pushes template Definitions; every template Definition of `T` is well formed (`Lower.selfCheck`). -/
theorem lowerDef_spec {P : Prog} {T : Array Def} (hWF : ∀ e < T.size, (T.getD e default).wellFormed T e = true) :
    ∀ (fuel d : ℕ) (s s' : St) (r : ℕ × Array ℕ), d < P.defs.size → (lowerDef P fuel d).run s = .ok (r, s') →
      s.defs.toList <+: s'.defs.toList ∧ (Inv P T s → s'.defs.toList <+: T.toList →
        CalleeSem P T d r.1 r.2 ∧ r.1 < s'.defs.size ∧ r.2.size = (P.defs[d]!).outLeaves ∧ Inv P T s')
  | 0, d, s, s', r, _, h => by
    rw [lowerDef] at h
    exact absurd h (st_throw _ _ _)
  | fuel + 1, d, s, s', r, hd, h => by
    rw [lowerDef] at h
    obtain ⟨s1, s2, h1, h2⟩ := st_bind_ok h
    have h1' : (Except.ok (s, s) : Except String (St × St)) = .ok (s1, s2) := h1
    simp only [Except.ok.injEq, Prod.mk.injEq] at h1'
    obtain ⟨h1a, h1b⟩ := h1'
    subst s1 s2
    split at h2
    · rename_i r0 hr0
      have h2' : (Except.ok (r0, s) : Except String ((ℕ × Array ℕ) × St)) = .ok (r, s') := h2
      simp only [Except.ok.injEq, Prod.mk.injEq] at h2'
      obtain ⟨h2a, h2b⟩ := h2'
      subst r s'
      exact ⟨List.prefix_refl _, fun hI _ => ⟨(hI.1 d r0 hr0).1, (hI.1 d r0 hr0).2.1, (hI.1 d r0 hr0).2.2, hI⟩⟩
    dsimp only at h2
    split at h2
    · first | exact absurd h2 (st_throw_bind _ _ _ _) | exact absurd h2 (st_throw _ _ _)
    rename_i hpr
    obtain ⟨u, s1, hb, h2⟩ := st_bind_ok h2
    obtain ⟨_, -, hq⟩ := st_lift_ok hb
    simp only [Prod.mk.injEq] at hq
    obtain ⟨-, hq⟩ := hq
    subst s1
    split at h2
    · first | exact absurd h2 (st_throw_bind _ _ _ _) | exact absurd h2 (st_throw _ _ _)
    rename_i hpar
    split at h2
    · first | exact absurd h2 (st_throw_bind _ _ _ _) | exact absurd h2 (st_throw _ _ _)
    rename_i hret
    split at h2
    · first | exact absurd h2 (st_throw_bind _ _ _ _) | exact absurd h2 (st_throw _ _ _)
    rename_i hseqr
    have hpr' : (P.defs[d]!).prim = false := by simpa using hpr
    have hret' : (P.defs[d]!).body.ret.count = (P.defs[d]!).outLeaves := by simpa using hret
    have hseqr' : seqOk (P.defs[d]!).body.ret = true := by simpa using hseqr
    obtain ⟨w, s3, h3, h4⟩ := st_bind_ok h2
    obtain ⟨wr, s4, h5, h6⟩ := st_bind_ok h4
    obtain ⟨wr', hrets, hq⟩ := st_lift_ok h5
    simp only [Prod.mk.injEq] at hq
    obtain ⟨hqa, hqb⟩ := hq
    subst wr s4
    obtain ⟨w2, rets⟩ := wr'
    dsimp only at h6
    obtain ⟨r1, s5, h7, h8⟩ := st_bind_ok h6
    rw [emit_run] at h7
    simp only [Except.ok.injEq, Prod.mk.injEq] at h7
    obtain ⟨h7a, h7b⟩ := h7
    subst r1 s5
    obtain ⟨u, s6, h9, h10⟩ := st_bind_ok h8
    rw [modify_run] at h9
    simp only [Except.ok.injEq, Prod.mk.injEq] at h9
    obtain ⟨-, h9b⟩ := h9
    subst s6
    have h10' : (Except.ok (_, _) : Except String ((ℕ × Array ℕ) × St)) = .ok (r, s') := h10
    simp only [Except.ok.injEq, Prod.mk.injEq] at h10'
    obtain ⟨h10a, h10b⟩ := h10'
    subst r s'
    have hlow : ∀ f (s s1 : St) (r : ℕ × Array ℕ), f < d → (lowerDef P fuel f).run s = .ok (r, s1) →
        s.defs.toList <+: s1.defs.toList ∧ (Inv P T s → s1.defs.toList <+: T.toList →
          CalleeSem P T f r.1 r.2 ∧ r.1 < s1.defs.size ∧ Inv P T s1) := by
      intro f s s1 r hf h
      obtain ⟨hp, hc⟩ := lowerDef_spec hWF fuel f s s1 r (by omega) h
      exact ⟨hp, fun hI hT => by obtain ⟨a, b, -, c⟩ := hc hI hT; exact ⟨a, b, c⟩⟩
    obtain ⟨hE, hp3, hc3⟩ := nodes_spec hWF hd hlow _ _ _ _ _ h3
    refine ⟨hp3.trans (by simp only [Array.toList_push]; exact List.prefix_append _ _), fun hI hT => ?_⟩
    simp only at hT ⊢
    have hT3 : s3.defs.toList <+: T.toList :=
      List.IsPrefix.trans (by simp only [Array.toList_push]; exact List.prefix_append _ _) hT
    obtain ⟨hI3, hB3, hS3⟩ := hc3 hI hT3
    have hBw := hB3 (fun i h => absurd h (by simp))
    obtain ⟨hwr, hsz, hlt, hval⟩ := rets_spec hseqr' hBw hrets
    have hlt3 : s3.defs.size < T.size := by
      have := hT.length_le
      simp only [Array.length_toList, Array.size_push] at this
      omega
    have hTe : T.getD s3.defs.size default = w2.finish (P.defs[d]!).paramLeaves := by
      rw [getD_prefix hT (by simp)]
      simp [Array.getD_eq_getD_getElem?]
    have hdc : P.defs[d]? = some (P.defs[d]!) := by
      rw [getElem!_pos P.defs d hd]; exact Array.getElem?_eq_getElem hd
    have hS : CalleeSem P T d s3.defs.size rets := by
      refine ⟨fun k hk => by rw [hTe, finish_cols]; exact hlt k hk, by rw [hTe, finish_ins],
        fun F hF fuel' v inp hlen hH => ?_⟩
      obtain ⟨f', rfl⟩ := holds_rows hH
      have hwf := hWF _ hlt3
      have hC := copyCtx_of hTe hwf hH
      have hcol := wf_rows hwf
      rw [hTe] at hcol hlen
      have hH1 := hH.1
      rw [hTe] at hH1
      have hR := rowsHold_finish hcol hH1
      rw [finish_ins] at hlen
      have hP0 : ParamsOk (atomV (w2.finish (P.defs[d]!).paramLeaves) v inp)
          ({ ins := (P.defs[d]!).paramLeaves } : Work).ins inp := by
        intro k hk
        change k < (P.defs[d]!).paramLeaves at hk
        rw [atomV_input (by rw [finish_ins]; exact hk), List.getD_eq_getElem _ _ (by omega),
          List.getD_eq_getElem _ _ (by omega)]
      obtain ⟨G, rfl⟩ : ∃ G, F = G + 1 := ⟨F - 1, by omega⟩
      obtain ⟨vals', hw, hF'⟩ := hS3 w2 _ v inp inp #[] G (by omega) hC (ext_of_wr hwr) hR
        (formsOk_nil _ _) hP0
      refine ⟨(List.range (P.defs[d]!).body.ret.count).map fun t =>
          refVal inp (paramStarts (P.defs[d]!).params) vals' ((P.defs[d]!).body.ret.toArray.getD t default), ?_,
        by simp [hsz], fun k hk => ?_⟩
      · rw [evalDef?_succ, walkStep_eq hdc hpr', paramStarts_eq, hw]
        rfl
      · have hk' : k < (P.defs[d]!).body.ret.count := by omega
        rw [List.getD_eq_getElem _ _ (by simp [hk']), List.getElem_map, List.getElem_range,
          ← atomV_own (D := w2.finish (P.defs[d]!).paramLeaves) (v := v) (inp := inp) (by rw [finish_cols]; exact hlt k hk),
          hval _ inp vals' hF' (by rw [hE.1]; exact hP0) hR k hk, paramStarts_eq]
    refine ⟨hS, by simp, by simp [hsz, hret'], ⟨fun f' r' hr' => ?_, fun f' r' hr' => ?_⟩⟩
    · simp only [Std.HashMap.getElem?_insert] at hr'
      split at hr'
      · simp only [Option.some.injEq] at hr'
        subst hr'
        rename_i hff
        simp only [beq_iff_eq] at hff
        subst hff
        exact ⟨hS, by simp, by simp [hsz, hret']⟩
      · exact ⟨(hI3.1 f' r' hr').1, by have := (hI3.1 f' r' hr').2.1; simp; omega, (hI3.1 f' r' hr').2.2⟩
    · exact ⟨(hI3.2 f' r' hr').1, by have := (hI3.2 f' r' hr').2; simp; omega⟩

end Flock.Chain.LowerDef
