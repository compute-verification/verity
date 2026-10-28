import Proofs.Flock.Claims.DeployTables
import Proofs.Flock.Claims.ZK.MaskTape

/-!
# The deployed masked session at one tape

The masked scope caps a session's tables at `Claims.Flock.zkTableCap` (`InScope`), so every statement's table count is
at most the cap (`jOf_le_cap`) and `not_fibersEq_live`'s hypothesis fails (`not_unbounded_live`). One tape then serves
every statement: `zkTableCap` cells, one per table, each statement reading its tables from the first `jOf x` cells.

* `TapeLayout`: one table's cell of tape (`Cell`), read as table `j`'s randomness at each statement `x` (`read`) with
  fibers of one size (`fibers`), with its mask-slot words (`maskAt`) the packing of the slot bits every reading gives
  (`mask_read`). A reading is the table's coordinates at their places in the cell, an injection of places, so it has
  fibers of one size whatever the places are (`fibersEq_comp_emb`); which place each coordinate takes is the layout's
  choice, and the rounds ask the one `flock-circuit --zk`'s tape takes (`TapeZk`'s indices).
* `TapePlan.ofLayout`: the tape `Fin zkTableCap → Cell`, statement `x` reading table `j` from cell `j` and joining the
  tables (`rejoin`, a bijection); the masked prover's tape `z`'s, with `TapeZk.mask` table `c`'s cell's words. It is read
  with fibers of one size at every statement (`fibersEq_ofLayout`), and each table's share of `ZkTables.honestPR`'s
  witness is the tables' `zWith` at the slot bits the statement reads (`zOf_ofLayout`).
* `TapeLayout.prefix`: a layout, every coordinate at its index in a cell of the live tables' largest sizes (`zMax`,
  `schMax`; `fits_zkTab`: the mask slot within `2^35` words, the pads below `2^64`, the schedule's sizes at their largest
  over `m ∈ [25, 35]`).
* `Flock.zk_deployZk_masked_layout_gap`: `Flock.zk_deployZk_masked_gap` at any layout's tape, `hT` discharged; what is
  left is the round plan's `Causal` and the three rounds at that tape.
-/

namespace Claims.ZKMal

open Claims FlockSoundness.Merkle FlockSoundness.ZK
open FlockSoundness.Discharge
open CoinLayer

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-! ## Fibers of one size under restriction -/

theorem fibersEq_id {α : Type} : FibersEq (id : α → α) := fibersEq_of_bijective Function.bijective_id

/-- Fibers of one size, componentwise over a pair. -/
theorem FibersEq.prodMap {α β γ δ : Type} {f : α → β} {g : γ → δ} (hf : FibersEq f) (hg : FibersEq g) :
    FibersEq (Prod.map f g) := by
  have e : ∀ p : β × δ, {a // Prod.map f g a = p} ≃ {a // f a = p.1} × {c // g c = p.2} := fun p =>
    { toFun := fun a => (⟨a.1.1, congrArg Prod.fst a.2⟩, ⟨a.1.2, congrArg Prod.snd a.2⟩)
      invFun := fun q => ⟨(q.1.1, q.2.1), Prod.ext q.1.2 q.2.2⟩
      left_inv := fun _ => rfl
      right_inv := fun _ => rfl }
  intro b b'
  rw [Nat.card_congr (e b), Nat.card_congr (e b'), Nat.card_prod, Nat.card_prod, hf b.1 b'.1, hg b.2 b'.2]

/-- Fibers of one size, coordinatewise over a finite index, one map at every coordinate. -/
theorem FibersEq.piConst {ι α β : Type} [Fintype ι] {f : α → β} (h : FibersEq f) :
    FibersEq fun (a : ι → α) i => f (a i) :=
  FibersEq.pi (α := fun _ => α) (β := fun _ => β) (f := fun _ => f) fun _ => h

/-- **Reading a function at an injection's points has fibers of one size**: the fiber at `g` is the functions on the
points the injection misses. -/
theorem fibersEq_comp_emb {A B Cod : Type} (ι : A ↪ B) : FibersEq fun f : B → Cod => f ∘ ι := by
  classical
  refine fibersEq_of_equiv (D := {b // b ∉ Set.range ι} → Cod)
    { toFun := fun f => (f ∘ ι, fun b => f b.1)
      invFun := fun p b => if hb : b ∈ Set.range ι then p.1 (Set.mem_range.1 hb).choose else p.2 ⟨b, hb⟩
      left_inv := fun f => funext fun b => by
        show (if hb : b ∈ Set.range ι then f (ι (Set.mem_range.1 hb).choose) else f b) = f b
        by_cases hb : b ∈ Set.range ι
        · rw [dif_pos hb]
          exact congrArg f (Set.mem_range.1 hb).choose_spec
        · rw [dif_neg hb]
      right_inv := fun p => by
        refine Prod.ext (funext fun a => ?_) (funext fun b => ?_)
        · have hb : ι a ∈ Set.range ι := ⟨a, rfl⟩
          show (if hb : ι a ∈ Set.range ι then p.1 (Set.mem_range.1 hb).choose else p.2 ⟨ι a, hb⟩) = p.1 a
          rw [dif_pos hb]
          exact congrArg p.1 (ι.injective (Set.mem_range.1 hb).choose_spec)
        · show (if hb : b.1 ∈ Set.range ι then p.1 (Set.mem_range.1 hb).choose else p.2 ⟨b.1, hb⟩) = p.2 b
          rw [dif_neg b.2] }
    fun _ => rfl

namespace Masked

/-- `Fin n` in `Fin N`, `n ≤ N`, at the same index. -/
def castEmb {n N : ℕ} (h : n ≤ N) : Fin n ↪ Fin N :=
  ⟨Fin.castLE h, fun a b e => Fin.ext (show a.val = b.val from (congrArg Fin.val e :))⟩

/-- **The first `n` of `N` values.** -/
def cut {Cod : Type} {n N : ℕ} (h : n ≤ N) (f : Fin N → Cod) : Fin n → Cod := f ∘ castEmb h

theorem fibersEq_cut {Cod : Type} {n N : ℕ} (h : n ≤ N) : FibersEq (cut (Cod := Cod) h) :=
  fibersEq_comp_emb (castEmb h)

/-! ## The table count -/

/-- **The masked session has at most `zkTableCap` tables**: the masked scope caps the parsed statement's (`InScope`), and
the setup makes as many as it counts (`setupOf_size_eq_tables`). -/
theorem jOf_le_cap (x : Msg) : jOf x ≤ Claims.Flock.zkTableCap := by
  unfold jOf
  split_ifs with h
  · obtain ⟨s, hs, hin⟩ := h.1
    rcases Nat.eq_zero_or_pos (stsOf x).size with h0 | h0
    · rw [h0]
      exact Nat.zero_le _
    · obtain ⟨i, spec, hset⟩ := setupOf_of_pos h0
      rw [FlockDeploy.setupOf_size_eq_tables (rel := _root_.Flock.Readable.relation) hset hs,
        Claims.Flock.Stmt.tables]
      have h9 : s.form.tables ≤ Claims.Flock.zkTableCap := by
        have h := hin.2.2.2.2.2.2.2.2.1
        first
          | exact h
          | simpa [FlockDeploy.deployZk] using h
      exact max_le (by unfold Claims.Flock.zkTableCap; omega) h9
  · exact Nat.zero_le _

/-- **`Live`'s table count is bounded**, so `not_fibersEq_live` does not apply to the masked scope. -/
theorem not_unbounded_live : ¬ ∀ N, ∃ x, Live x ∧ N ≤ jOf x := fun h => by
  obtain ⟨x, -, hx⟩ := h (Claims.Flock.zkTableCap + 1)
  have := jOf_le_cap x
  omega

/-! ## A tape of one cell per table -/

/-- **One table's randomness** at the sizes `z`: its base, and each of its pre-final leaves' salt and ideal seed. -/
abbrev TabRand (z : ZkView.Sizes) (Hi : Type) : Type :=
  ZkView.Base z Hi × (ZkView.LeafPos z → ZkView.Salt) × (ZkView.LeafPos z → ZkView.Bits × ZkView.Salt)

/-- **The tables' randomness from each table's.** -/
def rejoin (C : Tabs) (x : Msg) (g : ∀ j, TabRand (C x j).z (C x j).Hi) : Rand C x :=
  (fun j => (g j).1, fun l => (g l.1).2.1 l.2, fun l => (g l.1).2.2 l.2)

theorem rejoin_bijective (C : Tabs) (x : Msg) : Function.Bijective (rejoin C x) := by
  refine ⟨fun g g' h => funext fun j => ?_, fun r => ⟨fun j => (r.1 j, fun p => r.2.1 ⟨j, p⟩, fun p => r.2.2 ⟨j, p⟩),
    rfl⟩⟩
  have h1 : (g j).1 = (g' j).1 := congrArg (fun r : Rand C x => r.1 j) h
  have h2 : ∀ p, (g j).2.1 p = (g' j).2.1 p := fun p => congrArg (fun r : Rand C x => r.2.1 ⟨j, p⟩) h
  have h3 : ∀ p, (g j).2.2 p = (g' j).2.2 p := fun p => congrArg (fun r : Rand C x => r.2.2 ⟨j, p⟩) h
  exact Prod.ext h1 (Prod.ext (funext h2) (funext h3))

/-- **A layout of one table's tape**: a finite cell, read at each statement as each of its tables' randomness with fibers
of one size, whose mask-slot words (`maskAt c k`, word `k`) pack the slot bits each reading gives. -/
structure TapeLayout (P : Composed.LigProver) where
  Cell : Type
  [finCell : Fintype Cell]
  [neCell : Nonempty Cell]
  read : (x : Msg) → (j : Fin (jOf x)) → Cell → TabRand (tabC P x j).z (tabC P x j).Hi
  fibers : ∀ x j, FibersEq (read x j)
  maskAt : Cell → ℕ → _root_.Flock.F128
  mask_read : ∀ x j c (k : ℕ) (hk : k < (tabC P x j).z.nMask), maskAt c k = wordOfBits ((read x j c).1.1 ⟨k, hk⟩)

attribute [instance] TapeLayout.finCell TapeLayout.neCell

/-- **The tape of a layout**: a cell per table up to the cap, statement `x` reading table `j` from cell `j`; read as the
masked prover's tape `z`'s with table `c`'s mask words its cell's (`TapeZk.mask`, `STREAM_MASK`). -/
noncomputable def TapePlan.ofLayout {P : Composed.LigProver} (L : TapeLayout P)
    (z : (Fin Claims.Flock.zkTableCap → L.Cell) → FlockComplete.TapeZk) : TapePlan (tabC P) where
  Tape := Fin Claims.Flock.zkTableCap → L.Cell
  finTape := by
    haveI : Fintype L.Cell := L.finCell
    exact inferInstance
  neTape := by
    have : Nonempty L.Cell := L.neCell
    exact inferInstance
  tapeZk t := { z t with mask := fun c k => if hc : c < Claims.Flock.zkTableCap then L.maskAt (t ⟨c, hc⟩) k else 0 }
  tape x t := rejoin (tabC P) x fun j => L.read x j (t (Fin.castLE (jOf_le_cap x) j))

theorem TapePlan.ofLayout_mask {P : Composed.LigProver} (L : TapeLayout P)
    (z : (Fin Claims.Flock.zkTableCap → L.Cell) → FlockComplete.TapeZk) (t : Fin Claims.Flock.zkTableCap → L.Cell) :
    ((TapePlan.ofLayout L z).tapeZk t).mask =
      fun c k => if hc : c < Claims.Flock.zkTableCap then L.maskAt (t ⟨c, hc⟩) k else 0 :=
  rfl

/-- **A layout's tape is read with fibers of one size at every statement**: the first `jOf x` cells, each read with fibers
of one size, then joined by a bijection. -/
theorem fibersEq_ofLayout {P : Composed.LigProver} (L : TapeLayout P)
    (z : (Fin Claims.Flock.zkTableCap → L.Cell) → FlockComplete.TapeZk) (x : Msg) :
    FibersEq ((TapePlan.ofLayout L z).tape x) := by
  have : Fintype L.Cell := L.finCell
  have h1 : FibersEq (cut (Cod := L.Cell) (jOf_le_cap x)) := fibersEq_cut _
  have h2 : FibersEq fun (g : Fin (jOf x) → L.Cell) j => L.read x j (g j) :=
    FibersEq.pi (f := fun j => L.read x j) fun j => L.fibers x j
  exact (h1.comp h2).comp (fibersEq_of_bijective (rejoin_bijective (tabC P) x))

/-- **Table `j`'s share of `honestPR`'s witness at a layout's tape is the tables' `zWith`**: the share of `w` (`wtOf`)
with the slot's word `k` set to the bits `k` statement `x` reads from cell `j`. -/
theorem zOf_ofLayout {P : Composed.LigProver} (L : TapeLayout P)
    (z : (Fin Claims.Flock.zkTableCap → L.Cell) → FlockComplete.TapeZk) (x : Msg)
    (t : Fin Claims.Flock.zkTableCap → L.Cell) (w : Claims.Flock.Wit) (j : Fin (jOf x)) :
    _root_.Flock.Readable.zOf (stsOf x).size
        (FlockComplete.ZkTables.maskWOf x w ((TapePlan.ofLayout L z).tapeZk t).mask) j.val (tabC P x j).S.m =
      Composed.zWith (stAt x j) (wtOf P x w j) (((TapePlan.ofLayout L z).tape x t).1 j).1 := by
  have hj := lt_of_jOf j
  obtain ⟨s, spec, hs⟩ := setupOf_of_pos (x := x) (by omega)
  rw [TapePlan.ofLayout_mask]
  have e : ∀ μ : ℕ → ℕ → _root_.Flock.F128, FlockComplete.ZkTables.maskWOf x w μ =
      FlockComplete.ZkTables.maskW (stsOf x) w μ := fun μ => by
    unfold FlockComplete.ZkTables.maskWOf
    rw [hs]
  rw [e]
  refine (FlockComplete.ZkTables.zOf_maskW (stsOf x) w _ hj).trans ?_
  have hw : wtOf P x w j = _root_.Flock.Readable.zOf (stsOf x).size w j.val (tabC P x j).S.m :=
    _root_.Flock.Readable.zOf_tableWit hj w _
  rw [hw]
  congr 1
  funext k v
  have hc : j.val < Claims.Flock.zkTableCap := lt_of_lt_of_le j.isLt (jOf_le_cap x)
  have hk : k.val < (tabC P x j).z.nMask := k.isLt
  simp only [hc, ↓reduceDIte]
  rw [L.mask_read x j (t ⟨j.val, hc⟩) k.val hk]
  exact toZ_wordOfBits _ v

/-- **Where no outer coin is `1`, `honestPR` at a layout's tape is `honestP` at its mask witness** (`honestPR_next`),
whose table `j`'s share is the tables' `zWith` (`zOf_ofLayout`). -/
theorem honestPR_ofLayout {P : Composed.LigProver} (L : TapeLayout P)
    (z : (Fin Claims.Flock.zkTableCap → L.Cell) → FlockComplete.TapeZk) (w : Claims.Flock.Wit) (x : Msg)
    (t : Fin Claims.Flock.zkTableCap → L.Cell) (hist : List (Msg × Msg))
    (h : FlockComplete.ZkTables.outerHit x (hist.map Prod.snd) = false) :
    (FlockComplete.ZkTables.honestPR w).next x hist ((TapePlan.ofLayout L z).tapeZk t) =
      (FlockComplete.ZkTables.honestP (FlockComplete.ZkTables.maskWOf x w
        ((TapePlan.ofLayout L z).tapeZk t).mask)).next x hist ((TapePlan.ofLayout L z).tapeZk t) :=
  FlockComplete.ZkTables.honestPR_next h

/-! ## A layout: every coordinate at its index in a cell of the largest sizes -/

/-- Sizes `z` within `zM`, field by field (the fields the randomness reads). -/
structure Fits (z zM : ZkView.Sizes) : Prop where
  nMask : z.nMask ≤ zM.nMask
  nL : z.nL ≤ zM.nL
  L : z.L ≤ zM.L
  tpad : z.tpad ≤ zM.tpad
  nS : z.nS ≤ zM.nS
  nU : z.nU ≤ zM.nU
  nE : z.nE ≤ zM.nE
  nQ : z.nQ ≤ zM.nQ
  nPU : z.nPU ≤ zM.nPU
  nPad : z.nPad ≤ zM.nPad

/-- `PadIdx n` in `PadIdx N`: the exposed pads at their indices, the last four as themselves. -/
def padEmb {n N : ℕ} (h : n ≤ N) : PadIdx n ↪ PadIdx N :=
  Function.Embedding.sumMap (castEmb h) (Function.Embedding.refl _)

/-- `LeafPos z` in `LeafPos zM`: each tree's positions at their indices. -/
def leafEmb {z zM : ZkView.Sizes} (hz : Fits z zM) : ZkView.LeafPos z ↪ ZkView.LeafPos zM :=
  Function.Embedding.sumMap (castEmb (Nat.add_le_add hz.nS hz.nU))
    (Function.Embedding.sumMap (Function.Embedding.prodMap (Function.Embedding.refl _)
      (castEmb (Nat.add_le_add hz.nQ hz.nPU))) (Function.Embedding.refl _))

/-- **A base at sizes `z` from one at `zM`**: each coordinate at its index. -/
def resBase {z zM : ZkView.Sizes} (hz : Fits z zM) {h hM : ℕ} (hh : h ≤ hM) :
    ZkView.Base zM (Fin hM → ZkView.Salt) → ZkView.Base z (Fin h → ZkView.Salt) :=
  Prod.map (cut hz.nMask)
    (Prod.map
      (Prod.map ((fun a l => cut hz.tpad (a l)) ∘ cut hz.nL)
        (Prod.map (fun a r i => cut hz.L (a r i)) (fun a r i => cut hz.tpad (a r i))))
      (Prod.map (fun a r => Prod.map (cut hz.nE) id (a r))
        (Prod.map (fun a r => cut hz.nPad (a r))
          (Prod.map (fun a r => cut hz.nPad (a r))
            (Prod.map (fun a r => a r ∘ padEmb hz.nE)
              (Prod.map (cut hz.nS)
                (Prod.map (fun a r => cut hz.nQ (a r))
                  (Prod.map id (fun a r => cut hh (a r))))))))))

theorem fibersEq_resBase {z zM : ZkView.Sizes} (hz : Fits z zM) {h hM : ℕ} (hh : h ≤ hM) :
    FibersEq (resBase hz hh) := by
  unfold resBase
  exact (fibersEq_cut _).prodMap
    ((((fibersEq_cut _).comp (FibersEq.piConst (fibersEq_cut _))).prodMap
        ((FibersEq.piConst (FibersEq.piConst (fibersEq_cut _))).prodMap
          (FibersEq.piConst (FibersEq.piConst (fibersEq_cut _))))).prodMap
      ((FibersEq.piConst ((fibersEq_cut _).prodMap fibersEq_id)).prodMap
        ((FibersEq.piConst (fibersEq_cut _)).prodMap
          ((FibersEq.piConst (fibersEq_cut _)).prodMap
            ((FibersEq.piConst (fibersEq_comp_emb _)).prodMap
              ((fibersEq_cut _).prodMap
                ((FibersEq.piConst (fibersEq_cut _)).prodMap
                  (fibersEq_id.prodMap (FibersEq.piConst (fibersEq_cut _))))))))))

/-- **A table's randomness at sizes `z` from one at `zM`**: its base's coordinates and its leaves' at their indices. -/
def resTab {z zM : ZkView.Sizes} (hz : Fits z zM) {h hM : ℕ} (hh : h ≤ hM) :
    TabRand zM (Fin hM → ZkView.Salt) → TabRand z (Fin h → ZkView.Salt) :=
  Prod.map (resBase hz hh) (Prod.map (fun f => f ∘ leafEmb hz) (fun f => f ∘ leafEmb hz))

theorem fibersEq_resTab {z zM : ZkView.Sizes} (hz : Fits z zM) {h hM : ℕ} (hh : h ≤ hM) :
    FibersEq (resTab hz hh) := by
  unfold resTab
  exact (fibersEq_resBase hz hh).prodMap ((fibersEq_comp_emb _).prodMap (fibersEq_comp_emb _))

theorem resTab_mask {z zM : ZkView.Sizes} (hz : Fits z zM) {h hM : ℕ} (hh : h ≤ hM)
    (c : TabRand zM (Fin hM → ZkView.Salt)) (k : Fin z.nMask) :
    (resTab hz hh c).1.1 k = c.1.1 (Fin.castLE hz.nMask k) :=
  rfl

/-- A `--zk` schedule's sizes the randomness reads: `2^k₀` lanes of `2^cols` words, `t_pad`, level 0's `2^logLen`
positions, and the salts of levels ≥ 1 (`hiCount`). -/
def bnd (p : _root_.Flock.Schedule × ℕ) : ℕ :=
  2 ^ (FlockSoundness.Refine.schedOf p.1).k0 + 2 ^ (FlockSoundness.Refine.schedOf p.1).level₀.logCols + p.2 +
    2 ^ (FlockSoundness.Refine.schedOf p.1).level₀.logLen +
    2 * ((FlockSoundness.Refine.schedOf p.1).levels.tail.map fun l => 2 ^ l.logLen).sum

/-- `bnd` at `Zk.schedule m`, `0` where it refuses. -/
def schB (m : ℕ) : ℕ :=
  match _root_.Flock.Zk.schedule m with
  | .ok p => bnd p
  | .error _ => 0

/-- **The largest schedule sizes**: `bnd` at its largest over the `m` the `--zk` schedule accepts (`[25, 35]`). -/
def schMax : ℕ := (Finset.Icc 25 35).sup schB

theorem bnd_le {st : _root_.Flock.Stmt} (y : ZkHidden.ZkTab st) : bnd (y.sch, y.tPad) ≤ schMax := by
  have h := Finset.le_sup (f := schB) (Finset.mem_Icc.2 (ZkHidden.schedule_range y.hz))
  have e : schB st.m = bnd (y.sch, y.tPad) := by
    unfold schB
    rw [y.hz]
  rw [← e]
  exact h

/-- **The live tables' largest sizes**: the mask slot within `2^35` words, the schedule's sizes at `schMax`, the pads
and their code below `2^64`. -/
def zMax : ZkView.Sizes where
  nMask := 2 ^ 35
  nC := _root_.Flock.Zk.RANK_POINTS
  nR := 0
  nL := schMax
  L := schMax
  tpad := schMax
  nS := schMax
  nU := schMax
  nE := 2 ^ 64
  nQ := _root_.Flock.Zk.Q
  nPU := 2 ^ 64
  nPad := _root_.Flock.Zk.K_PAD

theorem length_flatMap_const {α β : Type} (l : List α) (f : α → List β) (n : ℕ) (h : ∀ a, (f a).length = n) :
    (l.flatMap f).length = l.length * n := by
  induction l with
  | nil => simp
  | cons a l ih =>
    simp only [List.flatMap_cons, List.length_append, h, ih, List.length_cons]
    ring

/-- **The mask slot has at most `2^m` words**: `2^nbl` blocks of `2^(slotLog − 7)` words, the slot within a block of
`2^kLog` bits (`MaskSetup.mask`), and `m = kLog + nbl`. -/
theorem maskWordsOf_length_le {st : _root_.Flock.Stmt} (M : Composed.MaskSetup st) (hm : st.m ≤ 35) :
    (Composed.maskWordsOf st).length ≤ 2 ^ 35 := by
  unfold Composed.maskWordsOf
  split
  · exact Nat.zero_le _
  · rename_i r hr
    obtain ⟨-, hs⟩ := M.mask r (List.mem_of_find?_eq_some hr) (Composed.kind_of_find hr)
    rw [length_flatMap_const _ _ (2 ^ (r.slotLog - 7)) fun _ => by simp, List.length_range]
    have h1 : 2 ^ (r.slotLog - 7) ≤ 2 ^ st.c.kLog :=
      (Nat.pow_le_pow_right (by norm_num) (Nat.sub_le _ _)).trans (by omega)
    calc 2 ^ st.nbl * 2 ^ (r.slotLog - 7) ≤ 2 ^ st.nbl * 2 ^ st.c.kLog := Nat.mul_le_mul_left _ h1
      _ = 2 ^ st.m := by rw [M.m, pow_add, mul_comm]
      _ ≤ 2 ^ 35 := Nat.pow_le_pow_right (by norm_num) hm

theorem bnd_parts {a b c d e M : ℕ} (h : a + b + c + d + e ≤ M) :
    a ≤ M ∧ b ≤ M ∧ c ≤ M ∧ d ≤ M ∧ e ≤ M := by
  omega

/-- **A live table's sizes fit `zMax`.** -/
theorem fits_zkTab (P : Composed.LigProver) {st : _root_.Flock.Stmt} (y : ZkHidden.ZkTab st) :
    Fits (Composed.coinedOf P y.table st).z zMax := by
  have hB := bnd_le y
  simp only [bnd] at hB
  obtain ⟨h1, h2, h3, h4, -⟩ := bnd_parts hB
  have hN := y.hN
  have hL := y.hL
  refine ⟨maskWordsOf_length_le y.mask (ZkHidden.schedule_range y.hz).2, ?_, ?_, ?_, ?_, ?_, ?_, le_rfl, ?_, le_rfl⟩
  · show 2 ^ (FlockSoundness.Refine.schedOf y.sch).k0 ≤ schMax
    exact h1
  · show 2 ^ (FlockSoundness.Refine.schedOf y.sch).level₀.logCols ≤ schMax
    exact h2
  · show y.tPad ≤ schMax
    exact h3
  · show y.tPad ≤ schMax
    exact h3
  · show 2 ^ (FlockSoundness.Refine.schedOf y.sch).level₀.logLen - y.tPad ≤ schMax
    exact le_trans (Nat.sub_le _ _) h4
  · show (ZkLink.layOf st y.sch).len - 4 ≤ 2 ^ 64
    omega
  · show ZkExec.innerN (ZkLink.layOf st y.sch) - _root_.Flock.Zk.Q ≤ 2 ^ 64
    omega

/-- **A live table's levels ≥ 1 have at most `schMax` salts.** -/
theorem hi_le {st : _root_.Flock.Stmt} (y : ZkHidden.ZkTab st) : Composed.hiCount y.table ≤ schMax := by
  have hB := bnd_le y
  simp only [bnd] at hB
  show 2 * ((FlockSoundness.Refine.schedOf y.sch).levels.tail.map fun l => 2 ^ l.logLen).sum ≤ schMax
  exact (bnd_parts hB).2.2.2.2

/-- **The prefix layout**: a cell of a table's randomness at `zMax` and `schMax` salts, each statement's table reading
every coordinate at its index (`resTab`); the mask words the cell's slot bits, packed. -/
noncomputable def TapeLayout.prefix (P : Composed.LigProver) : TapeLayout P where
  Cell := TabRand zMax (Fin schMax → ZkView.Salt)
  read x j := resTab (fits_zkTab P (zkTab x j)) (hi_le (zkTab x j))
  fibers x j := fibersEq_resTab _ _
  maskAt c k := if hk : k < zMax.nMask then wordOfBits (c.1.1 ⟨k, hk⟩) else 0
  mask_read x j c k hk := by
    have hk' : k < zMax.nMask := lt_of_lt_of_le hk (fits_zkTab P (zkTab x j)).nMask
    simp only [hk', ↓reduceDIte]
    first
      | exact congrArg wordOfBits (resTab_mask (fits_zkTab P (zkTab x j)) (hi_le (zkTab x j)) c ⟨k, hk⟩).symm
      | exact congrArg wordOfBits (resTab_mask _ _ c ⟨k, hk⟩).symm
      | exact congrArg wordOfBits rfl

end Masked

open Masked

open Classical in
/-- **Flock's zero knowledge at the deployed masked session, at one tape**: `Flock.zk_deployZk_masked_gap` at a layout's
tape (`TapePlan.ofLayout L z`), read with fibers of one size at every statement (`fibersEq_ofLayout`), at
`ε_zk = 8·n_L·gap + 8t·n_L·gap + 2·3^{-t}` on the masked scope and `1` outside it, with no hiding premise. Left: the round
plan's `Causal` and `refP`'s three rounds `ZkTables.honestPR`'s at that tape, on `Live`, at the relation's witnesses. -/
theorem Flock.zk_deployZk_masked_layout_gap (H : Msg → Msg) (nr : Msg → ℕ → ℕ) (P : Composed.LigProver)
    (Q : RoundPlan (tabC P)) (L : TapeLayout P) (z : (Fin Claims.Flock.zkTableCap → L.Cell) → FlockComplete.TapeZk)
    (t : ℕ) (ht : 0 < t) (hc : Q.Causal)
    (hpre : (session nr (tableData P Q (TapePlan.ofLayout L z))).HonestPreOn Live RZk H coinOfD
      fun ω => (TapePlan.ofLayout L z).tapeZk ω.2)
    (hfin : (session nr (tableData P Q (TapePlan.ofLayout L z))).HonestFinOn Live RZk H coinOfD
      fun ω => (TapePlan.ofLayout L z).tapeZk ω.2)
    (hoff : (session nr (tableData P Q (TapePlan.ofLayout L z))).HonestOffOn Live RZk H coinOfD
      fun ω => (TapePlan.ofLayout L z).tapeZk ω.2) :
    Claims.Flock.ZK H (zkDeployed _root_.Flock.Readable.relation nr (tableData P Q (TapePlan.ofLayout L z)) t)
      (fun x => if Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x then
        8 * nL P x * FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash +
          8 * t * nL P x * FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash + 2 * (1 / 3) ^ t else 1)
      fun _ => 4 * t * 3 ^ t :=
  Flock.zk_deployZk_masked_gap H nr P Q (TapePlan.ofLayout L z) t ht hc (fun x _ => fibersEq_ofLayout L z x) hpre hfin
    hoff

open Classical in
/-- **`Flock.zk_deployZk_masked_layout_gap` at a stopping round plan** (`RoundPlan.stop`): the third round is
`ZkTables.honestPR`'s alone (`honestOffOn_stop_iff`): after `refK x` openings whose coins some table refuses, its next
message is `[]`. -/
theorem Flock.zk_deployZk_masked_layout_stop_gap (H : Msg → Msg) (nr : Msg → ℕ → ℕ) (P : Composed.LigProver)
    (cz : (x : Msg) → (Fin (refK x) → FlockDeploy.Coin) → ∀ j, Composed.CoinsZK (tabC P x j).Rest)
    (enc : (x : Msg) → (c : Fin (refK x) → FlockDeploy.Coin) → ℕ → (∀ j, ((tabC P x j).T (cz x c j)).tab.Cut) → Msg)
    (fin : (x : Msg) → (c : Fin (refK x) → FlockDeploy.Coin) → (∀ j, ((tabC P x j).T (cz x c j)).tab.View) → Msg)
    (L : TapeLayout P) (z : (Fin Claims.Flock.zkTableCap → L.Cell) → FlockComplete.TapeZk) (t : ℕ) (ht : 0 < t)
    (hc : (RoundPlan.stop cz enc fin).Causal)
    (hpre : (session nr (tableData P (RoundPlan.stop cz enc fin) (TapePlan.ofLayout L z))).HonestPreOn Live RZk H
      coinOfD fun ω => (TapePlan.ofLayout L z).tapeZk ω.2)
    (hfin : (session nr (tableData P (RoundPlan.stop cz enc fin) (TapePlan.ofLayout L z))).HonestFinOn Live RZk H
      coinOfD fun ω => (TapePlan.ofLayout L z).tapeZk ω.2)
    (hoff : ∀ x, Live x → ∀ w, RZk x w → ∀ r (ops : List (Msg × Msg)), ops.length = refK x →
      ¬ (session nr (tableData P (RoundPlan.stop cz enc fin) (TapePlan.ofLayout L z))).Admits x
        ((session nr (tableData P (RoundPlan.stop cz enc fin) (TapePlan.ofLayout L z))).pad
          (ops.map fun p => rd p.2)) →
      (((FlockComplete.ZkTables.honestPR w).next x (ops.map fun p => (p.1, coinOfD p.2))
        ((TapePlan.ofLayout L z).tapeZk r)).eval H).1 = []) :
    Claims.Flock.ZK H
      (zkDeployed _root_.Flock.Readable.relation nr (tableData P (RoundPlan.stop cz enc fin) (TapePlan.ofLayout L z)) t)
      (fun x => if Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x then
        8 * nL P x * FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash +
          8 * t * nL P x * FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash + 2 * (1 / 3) ^ t else 1)
      fun _ => 4 * t * 3 ^ t :=
  Flock.zk_deployZk_masked_layout_gap H nr P (RoundPlan.stop cz enc fin) L z t ht hc hpre hfin
    ((honestOffOn_stop_iff (TapePlan.ofLayout L z) nr _ RZk H).2 hoff)

end Claims.ZKMal
