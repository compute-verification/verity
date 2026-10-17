import Proofs.Flock.C3.Reps.ReadRow
import Proofs.Flock.VBridge.Net

/-!
# The opening unit reads its sibling columns (`openSibRead`, `openReadCol`)

`OpenSibRead LANES H` and `OpenReadCol LANES H`, proved, and both at any unit name (`readCol_unitLaid`): the unit is
`netOf name (unitLaid LANES H)`, so `rec-open/…`'s units and `rec-open-stacked/…`'s alike.

* The support (`Av`): the climb's swap at level `h` ANDs the direction bit with `cur_i ^^^ s_i`, and that AND reads
  sibling bit `i` only if `cur_i` doesn't read it too (`^^^`'s `symm` cancels a variable both sides read). Every builder
  the climb runs keeps an input bit out of its output forms when its inputs keep it out (`av_and` through `av_shaFrom`,
  and `av_swap`): its forms are XORs of its inputs' forms, constants, and variables it makes.
* The witness (`reads_swap`, `rdN_climbFrom`): below level `h` every node and sibling avoids sibling bit `u`, so the swap
  at level `h` makes a row reading it. The residual forms come first and only add rows past the inputs (they are V*'s,
  `residualForms_eq`, and `fresh_residualsGo`), and direction bit `h` is an input bit as `H ≤ 32` (`sibReads_unit`).
* The unit (`rowReads_all`, `laid_reads_of`, `readCol_of_laid`): with `ReadRow`'s row bits, a builder row reads each of
  the first `128 · LANES + 512 · H` input bits, and its laid-out row is a computed row of the net reading that column.
-/

namespace FlockC3.Reps.RowRead

section Sib

open Flock.HmNets hiding and
open Flock.RecOpen
open FlockSoundness.Discharge.Hm (vars)

variable {u : Nat}

/-! ## A variable a builder's forms don't read -/

/-- No form of `l` reads `u`. -/
def Out (u : Nat) (l : List Lin) : Prop := ∀ f ∈ l, u ∉ f.s

/-- From any state past `u`, `m` returns a `g` and stays past `u`. -/
def Av {α : Type} (u : Nat) (m : M α) (g : α → Prop) : Prop :=
  ∀ s : St, u < s.next → g (m.run s).1 ∧ u < (m.run s).2.next

namespace Av

variable {α β : Type}

theorem pure' {g : α → Prop} {a : α} (h : g a) : Av u (pure a : M α) g := fun _ hs => ⟨h, hs⟩

theorem bind' {m : M α} {f : α → M β} {g₁ : α → Prop} {g₂ : β → Prop} (hm : Av u m g₁)
    (hf : ∀ a, g₁ a → Av u (f a) g₂) : Av u (m >>= f) g₂ :=
  fun s hs => hf _ (hm s hs).1 _ (hm s hs).2

theorem ite' {c : Prop} [Decidable c] {m₁ m₂ : M α} {g : α → Prop} (h₁ : c → Av u m₁ g) (h₂ : ¬c → Av u m₂ g) :
    Av u (if c then m₁ else m₂) g := by
  split
  · next h => exact h₁ h
  · next h => exact h₂ h

theorem mapM' {f : β → M α} {g : α → Prop} : ∀ {l : List β}, (∀ b ∈ l, Av u (f b) g) →
    Av u (l.mapM f) (fun r => ∀ a ∈ r, g a)
  | [], _ => by rw [List.mapM_nil]; exact pure' (by simp)
  | b :: l, hf => by
      rw [List.mapM_cons]
      exact bind' (hf b (List.mem_cons_self ..)) fun a ha =>
        bind' (mapM' fun b hb => hf b (List.mem_cons_of_mem _ hb)) fun r hr =>
          pure' fun x hx => (List.mem_cons.1 hx).elim (fun h => h ▸ ha) (hr x)

end Av

theorem mem_symm {v : Var} : ∀ {xs ys : List Var}, v ∈ Flock.HmNets.symm xs ys → v ∈ xs ∨ v ∈ ys
  | [], _, h => by unfold Flock.HmNets.symm at h; exact Or.inr h
  | _ :: _, [], h => by unfold Flock.HmNets.symm at h; exact Or.inl h
  | x :: xs, y :: ys, h => by
      unfold Flock.HmNets.symm at h
      split at h
      · rcases List.mem_cons.1 h with rfl | h
        · exact Or.inl (List.mem_cons_self ..)
        · exact (mem_symm h).imp (List.mem_cons_of_mem _) id
      · split at h
        · rcases List.mem_cons.1 h with rfl | h
          · exact Or.inr (List.mem_cons_self ..)
          · exact (mem_symm h).imp id (List.mem_cons_of_mem _)
        · exact (mem_symm h).imp (List.mem_cons_of_mem _) (List.mem_cons_of_mem _)

theorem mem_symm_var : ∀ {xs : List Var}, u ∉ xs → u ∈ Flock.HmNets.symm xs [u]
  | [], _ => by unfold Flock.HmNets.symm; exact List.mem_singleton_self _
  | x :: xs, h => by
      unfold Flock.HmNets.symm
      split
      · exact List.mem_cons_of_mem _ (mem_symm_var fun h' => h (List.mem_cons_of_mem _ h'))
      · split
        · exact List.mem_cons_self ..
        · next h1 h2 => exact absurd (List.mem_cons.2 (Or.inl (Nat.le_antisymm (Nat.not_lt.1 h1) (Nat.not_lt.1 h2)))) h

theorem out_xor {a b : Lin} (ha : u ∉ a.s) (hb : u ∉ b.s) : u ∉ (a ^^^ b).s := fun h => (mem_symm h).elim ha hb

theorem out_nil : Out u [] := by simp [Out]

theorem out_append {a b : List Lin} (ha : Out u a) (hb : Out u b) : Out u (a ++ b) := fun f hf =>
  (List.mem_append.1 hf).elim (ha f) (hb f)

theorem out_drop {a : List Lin} (n : Nat) (ha : Out u a) : Out u (a.drop n) := fun f hf => ha f (List.mem_of_mem_drop hf)

theorem out_take {a : List Lin} (n : Nat) (ha : Out u a) : Out u (a.take n) := fun f hf => ha f (List.mem_of_mem_take hf)

theorem out_consts {bs : List Bool} : Out u (bs.map Lin.const) := fun f hf => by
  obtain ⟨b, -, rfl⟩ := List.mem_map.1 hf
  simp [Lin.const]

theorem out_constW (k : BitVec 64) : Out u (constW k) := fun f hf => by
  obtain ⟨i, -, rfl⟩ := List.mem_map.1 hf
  simp [Lin.const]

theorem out_xorW : ∀ {a b : Word}, Out u a → Out u b → Out u (xorW a b)
  | x :: a, y :: b, ha, hb => fun f hf => by
      rcases List.mem_cons.1 hf with rfl | hf
      · exact out_xor (ha x (List.mem_cons_self ..)) (hb y (List.mem_cons_self ..))
      · exact out_xorW (fun g hg => ha g (List.mem_cons_of_mem _ hg)) (fun g hg => hb g (List.mem_cons_of_mem _ hg)) f hf
  | [], _, _, _ => by simp [xorW, out_nil]
  | _ :: _, [], _, _ => by simp [xorW, out_nil]

theorem out_rot {w : Word} (n : Nat) (h : Out u w) : Out u (rot w n) := out_append (out_drop n h) (out_take n h)

theorem out_shr {w : Word} (n : Nat) (h : Out u w) : Out u (shr w n) :=
  out_append (out_drop n h) fun f hf => by rw [List.eq_of_mem_replicate hf]; simp [Lin.zero]

theorem out_xor3 {a b c : Word} (ha : Out u a) (hb : Out u b) (hc : Out u c) : Out u (xor3 a b c) :=
  out_xorW (out_xorW ha hb) hc

theorem getD_mem {α : Type} {l : List α} {t : Nat} {d : α} : l.getD t d ∈ l ∨ l.getD t d = d := by
  rw [List.getD_eq_getElem?_getD]
  cases h : l[t]? with
  | none => exact Or.inr rfl
  | some a => exact Or.inl (List.mem_of_getElem? h)

theorem out_getD {l : List Lin} (h : Out u l) (t : Nat) : u ∉ (l.getD t Lin.zero).s := by
  rcases getD_mem (l := l) (t := t) (d := Lin.zero) with h' | h'
  · exact h _ h'
  · rw [h']; simp [Lin.zero]

theorem outW_getD {ws : List Word} (h : ∀ w ∈ ws, Out u w) (t : Nat) : Out u (ws.getD t []) := by
  rcases getD_mem (l := ws) (t := t) (d := []) with h' | h'
  · exact h _ h'
  · rw [h']; exact out_nil

theorem out_vars {k n : Nat} (h : ∀ i < n, k + i ≠ u) : Out u (vars k n) := fun f hf => by
  obtain ⟨i, hi, rfl⟩ := List.mem_map.1 hf
  simp only [Lin.var, List.mem_singleton]
  exact fun e => h i (List.mem_range.1 hi) e.symm

/-! ## Every builder keeps `u` out of its forms -/

theorem av_fresh (a b : Lin) : Av u (fresh a b) (fun f => u ∉ f.s) := fun _ hs =>
  ⟨fun h => Nat.ne_of_lt hs (List.mem_singleton.1 h), Nat.lt_succ_of_lt hs⟩

theorem av_and {a b : Lin} (ha : u ∉ a.s) (hb : u ∉ b.s) : Av u (Flock.HmNets.and a b) (fun f => u ∉ f.s) := by
  have hz : u ∉ Lin.zero.s := by simp [Lin.zero]
  unfold Flock.HmNets.and
  refine Av.ite' (fun _ => Av.pure' ?_) fun _ => Av.ite' (fun _ => Av.pure' ?_) fun _ =>
    Av.ite' (fun _ => Av.pure' ?_) fun _ => av_fresh a b
  all_goals split <;> assumption

theorem av_maj {x y z : Lin} (hx : u ∉ x.s) (hy : u ∉ y.s) (hz : u ∉ z.s) : Av u (maj x y z) (fun f => u ∉ f.s) := by
  unfold maj
  dsimp only
  refine Av.ite' (fun _ => Av.ite' (fun _ => Av.pure' (by simp [Lin.zero])) fun _ =>
    Av.ite' (fun _ => Av.pure' (by simp [Lin.one])) fun _ => Av.pure' ?_) fun _ =>
      Av.bind' (av_and (out_xor hx hz) (out_xor hy hz)) fun _ hr => Av.pure' (out_xor hz hr)
  split
  · next v t heq =>
      have h0 : v ∈ v :: t := List.mem_cons_self ..
      rw [← heq] at h0
      have hv := (List.mem_filter.1 h0).1
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hv
      rcases hv with rfl | rfl | rfl <;> assumption
  · simp [Lin.const]

theorem av_commit {f : Lin} (hf : u ∉ f.s) : Av u (commit f) (fun g => u ∉ g.s) := by
  unfold commit
  exact Av.ite' (fun _ => Av.pure' hf) fun _ => av_fresh _ _

theorem av_chW {e f g : Word} (he : Out u e) (hf : Out u f) (hg : Out u g) : Av u (chW e f g) (Out u) :=
  Av.mapM' fun _ hx =>
    have h1 := List.of_mem_zip hx
    have h2 := List.of_mem_zip h1.2
    Av.bind' (av_and (he _ h1.1) (out_xor (hf _ h2.1) (hg _ h2.2))) fun _ hr => Av.pure' (out_xor (hg _ h2.2) hr)

theorem av_majW {a b c : Word} (ha : Out u a) (hb : Out u b) (hc : Out u c) : Av u (majW a b c) (Out u) :=
  Av.mapM' fun _ hx =>
    have h1 := List.of_mem_zip hx
    have h2 := List.of_mem_zip h1.2
    av_maj (ha _ h1.1) (hb _ h2.1) (hc _ h2.2)

theorem av_commitW {w : Word} (hw : Out u w) : Av u (commitW w) (Out u) := Av.mapM' fun f hf => av_commit (hw f hf)

theorem av_add2Go : ∀ (a b : Word) (c : Lin) (i : Nat), Out u a → Out u b → u ∉ c.s → Av u (add2Go a b c i) (Out u)
  | x :: a, y :: b, c, i, ha, hb, hc => by
      have hx := ha x (List.mem_cons_self ..)
      have hy := hb y (List.mem_cons_self ..)
      have hs := out_xor (out_xor hx hy) hc
      unfold add2Go
      refine Av.ite' (fun _ => Av.pure' fun f hf => by rw [List.mem_singleton] at hf; exact hf ▸ hs) fun _ =>
        Av.bind' (av_maj hx hy hc) fun c₁ h₁ => ?_
      unfold carryCommit
      exact Av.bind' (Av.ite' (fun _ => av_commit h₁) fun _ => Av.pure' h₁) fun c₂ h₂ =>
        Av.bind' (av_add2Go a b c₂ (i + 1) (fun f hf => ha f (List.mem_cons_of_mem _ hf))
          (fun f hf => hb f (List.mem_cons_of_mem _ hf)) h₂) fun _ hr =>
            Av.pure' fun f hf => (List.mem_cons.1 hf).elim (fun h => h ▸ hs) (hr f)
  | [], _, _, _, _, _, _ => by unfold add2Go; exact Av.pure' out_nil
  | _ :: _, [], _, _, _, _, _ => by unfold add2Go; exact Av.pure' out_nil

theorem av_add2 {a b : Word} (ha : Out u a) (hb : Out u b) : Av u (add2 a b) (Out u) :=
  av_add2Go a b _ 0 ha hb (by simp [Lin.zero])

theorem av_addN : ∀ (acc : Word) (xs : List Word), Out u acc → (∀ x ∈ xs, Out u x) → Av u (addN acc xs) (Out u)
  | _, [], h, _ => Av.pure' h
  | _, x :: xs, h, hx => Av.bind' (av_add2 h (hx x (List.mem_cons_self ..))) fun a ha =>
      av_addN a xs ha fun y hy => hx y (List.mem_cons_of_mem _ hy)

theorem av_schedStep {w : List Word} (h : ∀ x ∈ w, Out u x) : Av u (schedStep w) (fun w' => ∀ x ∈ w', Out u x) := by
  have hg := outW_getD h
  unfold schedStep
  exact Av.bind' (av_addN _ _ (hg _) (List.forall_mem_cons.2 ⟨out_xor3 (out_rot _ (hg _)) (out_rot _ (hg _))
    (out_shr _ (hg _)), List.forall_mem_cons.2 ⟨hg _, List.forall_mem_cons.2 ⟨out_xor3 (out_rot _ (hg _))
      (out_rot _ (hg _)) (out_shr _ (hg _)), by simp⟩⟩⟩)) fun _ hx => Av.bind' (av_commitW hx) fun _ hy =>
        Av.pure' fun z hz => (List.mem_append.1 hz).elim (h z) fun hz => by rw [List.mem_singleton] at hz; exact hz ▸ hy

theorem av_schedule : ∀ (n : Nat) (w : List Word), (∀ x ∈ w, Out u x) → Av u (schedule n w) (fun w' => ∀ x ∈ w', Out u x)
  | 0, _, h => Av.pure' h
  | n + 1, _, h => Av.bind' (av_schedStep h) fun w' h' => av_schedule n w' h'

theorem regs_out {r : Regs Word} : (∀ x ∈ r.toList, Out u x) ↔
    Out u r.a ∧ Out u r.b ∧ Out u r.c ∧ Out u r.d ∧ Out u r.e ∧ Out u r.f ∧ Out u r.g ∧ Out u r.h := by
  simp [Regs.toList]

theorem av_roundW (k : BitVec 64) {wt : Word} {r : Regs Word} (hw : Out u wt) (hr : ∀ x ∈ r.toList, Out u x) :
    Av u (roundW k wt r) (fun r' => ∀ x ∈ r'.toList, Out u x) := by
  obtain ⟨ha, hb, hc, hd, he, hf, hg, hh⟩ := regs_out.1 hr
  unfold roundW
  exact Av.bind' (av_chW he hf hg) fun _ hch => Av.bind' (av_addN _ _ hh (List.forall_mem_cons.2
    ⟨out_xor3 (out_rot _ he) (out_rot _ he) (out_rot _ he), List.forall_mem_cons.2 ⟨hch, List.forall_mem_cons.2
      ⟨out_constW k, List.forall_mem_cons.2 ⟨hw, by simp⟩⟩⟩⟩)) fun _ ht1 =>
    Av.bind' (av_majW ha hb hc) fun _ hmj => Av.bind' (av_add2 (out_xor3 (out_rot _ ha) (out_rot _ ha) (out_rot _ ha))
      hmj) fun _ ht2 => Av.bind' (av_add2 hd ht1) fun _ he₁ => Av.bind' (av_commitW he₁) fun _ he₂ =>
        Av.bind' (av_add2 ht1 ht2) fun _ ha₁ => Av.bind' (av_commitW ha₁) fun _ ha₂ =>
          Av.pure' (regs_out.2 ⟨ha₂, ha, hb, hc, he₂, he, hf, hg⟩)

theorem av_rounds {w : List Word} (hw : ∀ x ∈ w, Out u x) : ∀ (n t : Nat) (r : Regs Word),
    (∀ x ∈ r.toList, Out u x) → Av u (rounds w n t r) (fun r' => ∀ x ∈ r'.toList, Out u x)
  | 0, _, _, h => Av.pure' h
  | n + 1, t, _, h => Av.bind' (av_roundW _ (outW_getD hw t) h) fun r' h' => av_rounds hw n (t + 1) r' h'

theorem out_getElem! {ws : List Word} (h : ∀ w ∈ ws, Out u w) (i : Nat) : Out u ws[i]! := by
  rw [List.getElem!_eq_getElem?_getD]
  cases h' : ws[i]? with
  | none => exact out_nil
  | some w => exact h w (List.mem_of_getElem? h')

theorem av_compress {cv msg : List Word} (hcv : ∀ x ∈ cv, Out u x) (hm : ∀ x ∈ msg, Out u x) :
    Av u (compress cv msg) (fun ws => ∀ x ∈ ws, Out u x) :=
  Av.bind' (av_schedule 64 msg hm) fun _ hw => Av.bind' (av_rounds hw 80 0 _ (regs_out.2 ⟨out_getElem! hcv 0,
    out_getElem! hcv 1, out_getElem! hcv 2, out_getElem! hcv 3, out_getElem! hcv 4, out_getElem! hcv 5,
      out_getElem! hcv 6, out_getElem! hcv 7⟩)) fun _ hr =>
    Av.mapM' fun _ hx => av_add2 (hcv _ (List.of_mem_zip hx).1) (hr _ (List.of_mem_zip hx).2)

theorem out_words {bits : List Lin} (h : Out u bits) (n : Nat) : ∀ x ∈ words bits n, Out u x := fun _ hx => by
  obtain ⟨i, -, rfl⟩ := List.mem_map.1 hx
  exact out_take _ (out_drop _ h)

theorem av_shaBlock {st blk : List Lin} (hs : Out u st) (hb : Out u blk) : Av u (shaBlock st blk) (Out u) :=
  Av.bind' (av_compress (out_words hs 8) (out_words hb 16)) fun _ hws => Av.pure' fun f hf =>
    let ⟨l, hl, hfl⟩ := List.mem_flatten.1 hf
    hws l hl f hfl

theorem av_shaBlocks {msg : List Lin} (hm : Out u msg) :
    ∀ (n k : Nat) (st : List Lin), Out u st → Av u (shaBlocks msg n k st) (Out u)
  | 0, _, _, h => Av.pure' h
  | n + 1, k, _, h => Av.bind' (av_shaBlock h fun f hf => by
        obtain ⟨p, -, rfl⟩ := List.mem_map.1 hf
        exact out_getD hm _) fun st' h' => av_shaBlocks hm n (k + 1) st' h'

theorem av_shaFrom (cv : List (BitVec 64)) {bits : List Lin} (pre : Nat) (hb : Out u bits) :
    Av u (shaFrom cv bits pre) (fun out => Out u out ∧ out.length = 512) :=
  Av.bind' (av_shaBlocks (out_append hb out_consts) _ 0 _ fun f hf => by
      obtain ⟨l, hl, hfl⟩ := List.mem_flatten.1 hf
      obtain ⟨k, -, rfl⟩ := List.mem_map.1 hl
      exact out_constW k f hfl) fun _ hst =>
    Av.pure' ⟨fun f hf => by
      obtain ⟨t, -, rfl⟩ := List.mem_map.1 hf
      exact out_getD hst _, by simp⟩

theorem av_swap {d : Lin} {a s : List Lin} (hd : u ∉ d.s) (ha : Out u a) (hs : Out u s) :
    Av u (swap d a s) (fun lr => Out u (lr.1 ++ lr.2)) :=
  Av.bind' (Av.mapM' fun _ hx => av_and hd (out_xor (ha _ (List.of_mem_zip hx).1) (hs _ (List.of_mem_zip hx).2)))
    fun _ ht => Av.pure' (out_append (out_xorW ha ht) (out_xorW hs ht))

end Sib

section Climb

open Flock.HmNets hiding and
open Flock.RecOpen
open FlockSoundness.Discharge.Hm (Good Fresh vars length_vars)

variable {u : Nat}

/-! ## Sibling bit `u` is read at its level -/

/-- From any state past `u`, `m` leaves a row reading `u`. -/
def RdN {α : Type} (u : Nat) (m : M α) : Prop := ∀ s : St, u < s.next → RowReads (m.run s).2 u

theorem Reads.rdN {α : Type} {m : M α} (h : Reads m u) : RdN u m := fun s _ => h s

theorem RdN.bind_left {α β : Type} {m : M α} {f : α → M β} (hm : RdN u m) (hf : ∀ a, Keeps (f a)) :
    RdN u (m >>= f) := fun s hs => (hm s hs).mono (hf (m.run s).1 (m.run s).2)

theorem RdN.bind_right {α β : Type} {m : M α} {f : α → M β} {g : α → Prop} (hm : Av u m g)
    (hf : ∀ a, g a → RdN u (f a)) : RdN u (m >>= f) := fun s hs => hf _ (hm s hs).1 _ (hm s hs).2

/-- **The swap at sibling bit `u`'s level reads it**: `and d (a_i ^^^ s_i)` with `s_i = u`, `a_i` not reading `u` and
`d` another variable is a row. -/
theorem reads_swap {dv : Var} {a s : List Lin} {i : Nat} (hdv : dv ≠ u) (ha : i < a.length) (hs : i < s.length)
    (hau : u ∉ (a.getD i Lin.zero).s) (hsu : s.getD i Lin.zero = Lin.var u) : Reads (swap (Lin.var dv) a s) u := by
  have hmem : (a.getD i Lin.zero, s.getD i Lin.zero) ∈ a.zip s := by
    have hz : i < (a.zip s).length := by rw [List.length_zip]; omega
    have h := List.getElem_mem hz
    rw [List.getElem_zip] at h
    rwa [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD, List.getElem?_eq_getElem ha,
      List.getElem?_eq_getElem hs, Option.getD_some, Option.getD_some]
  have hx : u ∈ (a.getD i Lin.zero ^^^ s.getD i Lin.zero).s := by
    rw [hsu]
    exact mem_symm_var hau
  have hd : (Lin.var dv).s = [dv] := rfl
  unfold swap
  refine Reads.bind_left (Reads.mem_mapM (fun x : Lin × Lin => Flock.HmNets.and (Lin.var dv) (x.1 ^^^ x.2))
    (fun _ => keeps_and _ _) (p := (a.getD i Lin.zero, s.getD i Lin.zero)) ?_ _ hmem) fun _ => Keeps.pure' _
  refine reads_and (by rw [hd]; exact List.cons_ne_nil _ _) (List.ne_nil_of_mem hx) (fun h => ?_)
    (List.mem_append_right _ hx)
  rw [hd] at h
  rw [← h] at hx
  exact hdv (List.mem_singleton.1 hx).symm

/-- **The climb reads sibling bit `u` at level `h + n`**: below it, every node and sibling avoids `u` (`av_swap`,
`av_shaFrom`); at it, the swap reads `u` (`reads_swap`). -/
theorem rdN_climbFrom {d : List Lin} (hd : Out u d) {dv : Var} (hdu : dv ≠ u) {i : Nat} (hi : i < 512) :
    ∀ (n h : Nat) (cur : List Lin) (ss : List (List Lin)), d.getD (h + n) Lin.zero = Lin.var dv →
      Out u cur → cur.length = 512 → (∀ j < n, Out u (ss.getD j [])) → n < ss.length →
      (ss.getD n []).length = 512 → (ss.getD n []).getD i Lin.zero = Lin.var u → RdN u (climbFrom d h cur ss)
  | _, _, _, [], _, _, _, _, hn, _, _ => absurd hn (Nat.not_lt_zero _)
  | 0, h, cur, s :: rest, hdh, hcur, hc, _, _, hsl, hsi => by
      simp only [List.getD_cons_zero] at hsl hsi
      unfold climbFrom
      refine RdN.bind_left (Reads.rdN ?_) fun _ => Keeps.bind' (keeps_shaFrom _ _ _) fun _ => keeps_climbFrom _ _ _ _
      rw [Nat.add_zero] at hdh
      rw [hdh]
      exact reads_swap hdu (by omega) (by omega) (out_getD hcur i) hsi
  | n + 1, h, cur, s :: rest, hdh, hcur, _, hss, hn, hsl, hsi => by
      unfold climbFrom
      have hs0 : Out u s := hss 0 (Nat.succ_pos _)
      refine RdN.bind_right (av_swap (out_getD hd h) hcur hs0) fun lr hlr =>
        RdN.bind_right (av_shaFrom ivW 0 hlr) fun cur' hcur' => rdN_climbFrom hd hdu hi n (h + 1) cur' rest
          (by rwa [show h + 1 + n = h + (n + 1) by omega]) hcur'.1 hcur'.2
          (fun j hj => by simpa only [List.getD_cons_succ] using hss (j + 1) (by omega))
          (by simp only [List.length_cons] at hn; omega) (by simpa only [List.getD_cons_succ] using hsl)
          (by simpa only [List.getD_cons_succ] using hsi)

theorem getD_sibForms {b n H j : Nat} (hj : j < H) (hn : 512 * H ≤ n) :
    (sibForms (vars b n) H).getD j [] = vars (b + 512 * j) 512 := by
  rw [sibForms, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range hj, Option.map_some,
    Option.getD_some]
  exact FlockVBridge.vars_drop_take b n (512 * j) 512 (by omega)

/-- **`climbRow` reads sibling bit `b + 512 h + i`** of `aux = vars b n`, at direction bit `dv`. -/
theorem rdN_climbRow {H : Nat} {row d : List Lin} {b n h i : Nat} {dv : Var} (hrow : Out u row) (hd : Out u d)
    (hdv : d.getD h Lin.zero = Lin.var dv) (hdu : dv ≠ u) (hh : h < H) (hi : i < 512) (hn : 512 * H ≤ n)
    (hu : u = b + 512 * h + i) : RdN u (climbRow H row (vars b n) d) := by
  unfold climbRow
  refine RdN.bind_right (av_shaFrom ivW 0 hrow) fun leaf hleaf => RdN.bind_left ?_ fun _ => Keeps.pure' _
  refine rdN_climbFrom hd hdu hi h 0 leaf _ (by rwa [Nat.zero_add]) hleaf.1 hleaf.2 (fun j hj => ?_)
    (by simp only [sibForms, List.length_map, List.length_range]; exact hh) ?_ ?_
  · rw [getD_sibForms (by omega) hn]
    exact out_vars fun i' hi' => by omega
  · rw [getD_sibForms hh hn, length_vars]
  · rw [getD_sibForms hh hn, hu]
    exact getD_inputs _ _ _ hi

theorem good_start (N : Nat) : Good N ⟨N, []⟩ := ⟨rfl, fun _ _ h => by simp at h⟩

theorem rdN_of_good {α β : Type} {m : M α} {f : α → M β} {N : Nat} (hN : u < N) (hm : Fresh m)
    (hf : ∀ a, RdN u (f a)) {s : St} (hs : Good N s) : RowReads ((m >>= f).run s).2 u := by
  have h := (hm N s hs).1
  exact hf (m.run s).1 (m.run s).2 (by omega)

/-- **The opening's builder leaves a row reading each sibling bit**: residual forms (V*'s, `residualForms_eq`) only add
rows past the inputs (`fresh_residualsGo`), then the climb reads it (`rdN_climbRow`) at direction bit `h`, an input bit
as `H ≤ 32`. -/
theorem sibReads_unit (LANES H : Nat) (hH : H ≤ 32) {h i : Nat} (hh : h < H) (hi : i < 512) :
    RowReads ((recOpenNet LANES H (ports LANES H)).run St.empty).2 (128 * LANES + 512 * h + i) := by
  have hm := Nat.le_max_right 1 ((H + 1) / 2)
  rw [recOpenNet_run]
  unfold recOpen
  rw [FlockVBridge.residualForms_eq]
  refine rdN_of_good ?_ (FlockVBridge.fresh_residualsGo _ _ _ _) (fun _ => RdN.bind_left ?_ fun _ => Keeps.pure' _)
    (good_start _)
  · simp only [St.empty, ports]
    generalize max 1 ((H + 1) / 2) = m at hm ⊢
    omega
  · have hrow : Out (128 * LANES + 512 * h + i) (vars St.empty.next (ports LANES H).row) := out_vars fun k hk => by
      simp only [St.empty, ports] at hk ⊢
      omega
    have hd : Out (128 * LANES + 512 * h + i) ((vars (St.empty.next + (ports LANES H).row + (ports LANES H).aux +
        (ports LANES H).acc + (ports LANES H).coef) (ports LANES H).dirs).take 32) := out_take _ (out_vars fun k _ => by
      simp only [St.empty, ports]
      generalize max 1 ((H + 1) / 2) = m at hm ⊢
      omega)
    have hdv : ((vars (St.empty.next + (ports LANES H).row + (ports LANES H).aux + (ports LANES H).acc +
        (ports LANES H).coef) (ports LANES H).dirs).take 32).getD h Lin.zero = Lin.var (St.empty.next +
          (ports LANES H).row + (ports LANES H).aux + (ports LANES H).acc + (ports LANES H).coef + h) := by
      rw [getD_take (by omega)]
      exact getD_inputs _ _ _ (by simp only [ports]; omega)
    refine rdN_climbRow hrow hd hdv ?_ hh hi ?_ ?_
    · simp only [St.empty, ports]
      generalize max 1 ((H + 1) / 2) = m at hm ⊢
      change @Ne Nat _ _
      omega
    · simp only [ports]
      generalize max 1 ((H + 1) / 2) = m at hm ⊢
      omega
    · simp only [St.empty, ports]
      omega

/-- **The opening's builder leaves a row reading each of its first `128 · LANES + 512 · H` input bits.** -/
theorem rowReads_all (LANES H : Nat) (hH : H ≤ 32) {c : Nat} (hc : c < 128 * LANES + 512 * H) :
    RowReads ((recOpenNet LANES H (ports LANES H)).run St.empty).2 c := by
  by_cases hcr : c < 128 * LANES
  · exact rowReads_unit LANES H hcr
  · have h1 : (c - 128 * LANES) / 512 < H := by omega
    have h2 : (c - 128 * LANES) % 512 < 512 := Nat.mod_lt _ (by norm_num)
    have h3 : c = 128 * LANES + 512 * ((c - 128 * LANES) / 512) + (c - 128 * LANES) % 512 := by omega
    rw [h3]
    exact sibReads_unit LANES H hH h1 h2

theorem lt_aux {LANES H c : Nat} (hc : c < 128 * LANES + 512 * H) : c < 128 * LANES + (ports LANES H).aux := by
  have hm := Nat.le_max_right 1 ((H + 1) / 2)
  simp only [ports]
  generalize max 1 ((H + 1) / 2) = m at hm ⊢
  omega

theorem le_32_of_valid {LANES H : Nat} (hv : valid LANES H = true) : H ≤ 32 := by
  simp [valid] at hv
  omega

end Climb

end FlockC3.Reps.RowRead

namespace FlockC3.Reps

open FlockSoundness FlockSoundness.Audit
open FlockVBridge.OfVStar (ReadCol)

/-- **The opening unit reads its row and sibling columns, under any name**: the net `netOf name (unitLaid LANES H)`, at
`valid LANES H` and `unitLog ≤ 32`, has a computed row reading each column `c < 128 · LANES + 512 · H`. -/
theorem readCol_unitLaid (LANES H : ℕ) (name : String) {net : Flock.Net} (hU : Built net)
    (hn : Flock.HmNets.netOf name (Flock.RecOpen.unitLaid LANES H) = .ok net) (hv : Flock.RecOpen.valid LANES H = true)
    (hu : net.unitLog ≤ 32) : ∀ c < 128 * LANES + 512 * H, ReadCol (Rows.ofNet hU) c := by
  intro c hc
  obtain ⟨k, hk, hmem⟩ := RowRead.laid_reads_of LANES H (RowRead.rowReads_all LANES H (RowRead.le_32_of_valid hv) hc)
    (RowRead.lt_aux hc)
  rw [← RowRead.inWords_unitLaid hn] at hk hmem
  exact RowRead.readCol_of_laid hU hn hu hk hmem

/-- **The opening unit reads its sibling columns** (`OpenSibRead`). -/
theorem openSibRead (LANES H : ℕ) : OpenSibRead LANES H := fun hU hn hv hu _ hh _ hi =>
  readCol_unitLaid LANES H _ hU hn hv hu _ (by omega)

/-- **The opening unit reads its row and sibling columns** (`OpenReadCol`): its two halves (`openReadCol_of`). -/
theorem openReadCol (LANES H : ℕ) : OpenReadCol LANES H := openReadCol_of (openRowRead LANES H) (openSibRead LANES H)

end FlockC3.Reps
