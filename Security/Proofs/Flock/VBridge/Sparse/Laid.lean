import Proofs.Flock.VBridge.Sparse.Builder
import Proofs.Flock.VBridge.ResUnit

/-!
# The sparse-phase unit laid out

The verifier's `rec-sparse/…` unit is `Flock.RecSparse.unitLaid sh`: one input of all its ports' words, cut by name
(`ports`), then `steps`, laid out one 16-bit port per word. This file states what its output columns hold.

* The values. `stepVals_eq`: `stepVals` at the builder's residuals is `outOf` at `Defs`' (`resArr`), element by element.
* The rows. `fresh_steps`: `steps` adds rows only as `fresh` does, so `laid_rows` reads them.
* The ports. `portOf`: the input bits `unitNet'` gives port `nm` (`unitNet'_run`). The message ports' first bits are
  `rowForms` (`flatMap_portOf`), `v`'s and `tau`'s are slices after them (`portOf_v`, `portOf_tau`), and `SparseIn` on the
  input columns gives `StepsIn` (`stepsIn_of`).
* The theorem. `laid_sparse`: whenever the laid-out unit's rows hold at `v`, if its input columns hold `w`, `vv` and the
  slots' types (`SparseIn`), its output columns from `sparseBase` hold `outOf`'s residuals, 128 bits each, and with `T > 1`
  from bit 512 each type with its low `T_LOG` bits cleared.
-/

namespace FlockVBridge.Sparse

open Flock.HmNets hiding and
open Flock (F128)
open FlockSoundness.Discharge.Hm
open Flock.RecSparse (messagePorts vParts offset elem)
open FlockSoundness.Refine (fOf fOf_add fOf_mul fOf_zero fOf_injective)
open FlockVBridge.OfVStar (rowForms rowCol bit_rowForms rowCol_lt length_rowForms flatMap_rowForms varSlices
  varSlices_append length_varSlices slices_vars_foldl take_vars le_portWords bit_of_length_le slices_eq
  run_input_bind' foldl_replicate bitOf_zero)

/-! ## Lists -/

theorem foldl_congr2 {α β : Type} {f g : α → β → α} : ∀ {l : List β} {a b : α}, a = b →
    (∀ x ∈ l, ∀ c, f c x = g c x) → l.foldl f a = l.foldl g b
  | [], _, _, h, _ => h
  | x :: l, a, b, h, hf => by
      rw [List.foldl_cons, List.foldl_cons, h, hf x List.mem_cons_self b]
      exact foldl_congr2 rfl fun y hy c => hf y (List.mem_cons_of_mem _ hy) c

theorem flatMap_range_congr {β : Type} (f g : ℕ → List β) : ∀ n, (∀ j < n, f j = g j) →
    (List.range n).flatMap f = (List.range n).flatMap g
  | 0, _ => by simp
  | n + 1, h => by
      simp only [List.range_succ, List.flatMap_append, List.flatMap_cons, List.flatMap_nil,
        flatMap_range_congr f g n fun j hj => h j (by omega), h n (by omega)]

theorem fOf_prodPairs (a b : ℕ → F128) (x : F128) : ∀ n,
    fOf ((List.range n).foldl (fun x j => x * (a j * b j)) x) =
      fOf x * (((List.range n).flatMap fun j => [a j, b j]).map fOf).prod
  | 0 => by simp
  | n + 1 => by
      simp only [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil, fOf_mul, fOf_prodPairs a b x n,
        List.flatMap_append, List.flatMap_cons, List.flatMap_nil, List.append_nil, List.map_append, List.map_cons,
        List.map_nil, List.prod_append, List.prod_cons, List.prod_nil, mul_one, mul_assoc]

theorem nodup_map_inj {α β : Type} (f : α → β) (hf : ∀ a b, f a = f b → a = b) {l : List α} (h : l.Nodup) :
    (l.map f).Nodup := List.Pairwise.map f (fun a b hab e => hab (hf a b e)) h

theorem map_fst_fst_zip {α β γ : Type} : ∀ (l : List (α × β)) (bs : List γ), l.length ≤ bs.length →
    (l.zip bs).map (·.1.1) = l.map (·.1)
  | [], _, _ => by simp
  | _ :: _, [], h => by simp at h
  | p :: l, b :: bs, h => by
      simp only [List.zip_cons_cons, List.map_cons, map_fst_fst_zip l bs (by simpa using h)]

theorem find_of_mem {β : Type} : ∀ {l : List ((String × ℕ) × β)}, (l.map (·.1.1)).Nodup →
    ∀ {e : (String × ℕ) × β}, e ∈ l → l.find? (·.1.1 = e.1.1) = some e
  | [], _, _, h => by simp at h
  | a :: l, hn, e, h => by
      simp only [List.map_cons, List.nodup_cons] at hn
      rcases List.mem_cons.1 h with rfl | h
      · exact List.find?_cons_of_pos (by simp)
      · have hne : a.1.1 ≠ e.1.1 := fun he => hn.1 (by rw [he]; exact List.mem_map.2 ⟨e, h, rfl⟩)
        rw [List.find?_cons_of_neg]
        · exact find_of_mem hn.2 h
        · simpa using hne

theorem flatMap_zip_of {α β γ : Type} (F : α → β → List γ) (G : α → β) : ∀ (l : List α) (bs : List β),
    l.length ≤ bs.length → (∀ p b, (p, b) ∈ l.zip bs → G p = b) →
    l.flatMap (fun p => F p (G p)) = (l.zip bs).flatMap (fun e => F e.1 e.2)
  | [], _, _, _ => by simp
  | _ :: _, [], h, _ => by simp at h
  | p :: l, b :: bs, h, hG => by
      simp only [List.zip_cons_cons, List.flatMap_cons, hG p b (by simp),
        flatMap_zip_of F G l bs (by simpa using h) fun q c hq => hG q c (List.mem_cons_of_mem _ hq)]

theorem mem_zip_varSlices {α : Type} (f : α → ℕ) : ∀ (l : List α) (k : ℕ) {p : α}, p ∈ l →
    ∃ k', (p, vars k' (16 * f p)) ∈ l.zip (varSlices k (l.map f))
  | [], _, _, h => by simp at h
  | a :: l, k, p, h => by
      rcases List.mem_cons.1 h with rfl | h
      · exact ⟨k, by simp [varSlices]⟩
      · obtain ⟨k', hk⟩ := mem_zip_varSlices f l (k + 16 * f a) h
        exact ⟨k', by simp only [List.map_cons, varSlices, List.zip_cons_cons]; exact List.mem_cons_of_mem _ hk⟩

/-- A flattened list of ports' first bits, at port `nm`'s offset: port `nm`'s bits. -/
theorem bit_flatMap_offset (z : Var → Bool) (G : String → List Lin) : ∀ (ps : List (String × ℕ)),
    (ps.map (·.1)).Nodup → (∀ p ∈ ps, 128 * p.2 ≤ (G p.1).length) → ∀ {nm : String} {n : ℕ}, (nm, n) ∈ ps →
    ∀ i < 128 * n, bit z (ps.flatMap fun p => (G p.1).take (128 * p.2)) (128 * offset ps nm + i) = bit z (G nm) i
  | [], _, _, _, _, h, _, _ => by simp at h
  | (a, k) :: ps, hn, hl, nm, n, h, i, hi => by
      simp only [List.map_cons, List.nodup_cons] at hn
      simp only [List.flatMap_cons]
      have hlk : 128 * k ≤ (G a).length := hl (a, k) List.mem_cons_self
      rcases List.mem_cons.1 h with h | h
      · obtain ⟨e1, e2⟩ := Prod.mk.inj h
        rw [e1, offset_cons_self, Nat.mul_zero, Nat.zero_add, bit_append_left z _ (by rw [List.length_take]; omega)]
        rw [e2] at hi
        simp only [bit, List.getElem?_take_of_lt hi]
      · have hne : a ≠ nm := fun e => hn.1 (by rw [e]; exact List.mem_map.2 ⟨(nm, n), h, rfl⟩)
        rw [offset_cons_ne _ _ hne, show 128 * (k + offset ps nm) + i = 128 * k + (128 * offset ps nm + i) by omega,
          bit_append_len z _ _ (by rw [List.length_take]; omega)]
        exact bit_flatMap_offset z G ps hn.2 (fun p hp => hl p (List.mem_cons_of_mem _ hp)) h i hi

/-! ## The values -/

section Values

variable (sh : Flock.RecSparse.Shape)

/-- `stepVals` at residual values `R` that are `Defs`' residuals `rs` by name is `outOf`'s output at `rs`. -/
theorem stepVals_outOf (R : String → F128) (rs wA vA : Array F128) (τA : Array ℕ)
    (hR : ∀ r ∈ names (shOf sh), R (str r) = rget (shOf sh) rs r) :
    stepVals sh R (wA[·]!) (vA[·]!) (τA[·]!) = (outOf (shOf sh) rs wA vA τA).1.toList := by
  have hh1 : R "h1" = rget (shOf sh) rs .h1 := hR _ (mem_h1 _)
  have hwv : R "wv" = rget (shOf sh) rs .wv := hR _ (mem_wv _)
  have hf0 : R "f0" = rget (shOf sh) rs .f0 := hR _ (mem_f0 _)
  have hbp : R "bp" = rget (shOf sh) rs .bp := hR _ (mem_bp _)
  have hpr : ∀ j < 6, R s!"pre{j}" = rget (shOf sh) rs (.pre j) := fun j hj => hR _ (mem_pre _ hj)
  have hsc : ∀ j < nR (shOf sh), R s!"sc{j}" = rget (shOf sh) rs (.sc j) := fun j hj => hR _ (mem_sc _ hj)
  have hfx : ∀ j < sh.ell, R s!"fx{j}" = rget (shOf sh) rs (.fx j) := fun j hj => hR _ (mem_fx _ hj)
  have hfy : ∀ j < sh.ell, R s!"fy{j}" = rget (shOf sh) rs (.fy j) := fun j hj => hR _ (mem_fy _ hj)
  obtain ⟨-, -, ov2, -, -, ovt, -⟩ := mp_offsets sh
  obtain ⟨hvpre, hvsc, hvur, hveqs⟩ := vp_offsets sh
  have hsd := sum_degs sh
  have hvp : ∀ j, offset (vParts sh) "pre" + 2 * j = vPre j := fun j => by rw [hvpre]; rfl
  have hvs : ∀ j < nR (shOf sh), offset (vParts sh) "sc" + (sh.degrees.take j).foldl (· + ·) 0 = vSc (shOf sh) j :=
    fun j hj => by rw [hvsc, degs_take sh j (Nat.le_of_lt hj), vSc]
  have hur : offset (vParts sh) "ur" + 0 = vUr (shOf sh) := by
    simp only [vUr, vAy, vAx, vAlpha, shOf_ell]
    omega
  have H2 : (List.range 6).foldl (fun c j => R s!"pre{j}" + c * vA[offset (vParts sh) "pre" + 2 * j]!) (R "h1") +
        R "wv" =
      (List.range 6).foldl (fun c j => rget (shOf sh) rs (.pre j) + c * vA[vPre j]!) (rget (shOf sh) rs .h1) +
        rget (shOf sh) rs .wv :=
    congrArg₂ (· + ·) (foldl_congr2 hh1 fun j hj c => by simp only [hpr j (List.mem_range.1 hj), hvp j]) hwv
  have H3 : (List.range sh.degrees.length).foldl
        (fun c j => R s!"sc{j}" + c * vA[offset (vParts sh) "sc" + (sh.degrees.take j).foldl (· + ·) 0]!)
        wA[offset (messagePorts sh) "v2"]! +
      treeVal ([R "f0"] ++ (List.range sh.ell).flatMap (fun j => [R s!"fx{j}", R s!"fy{j}"]) ++
          [vA[offset (vParts sh) "ur" + 0]!]).length
        ([R "f0"] ++ (List.range sh.ell).flatMap (fun j => [R s!"fx{j}", R s!"fy{j}"]) ++
          [vA[offset (vParts sh) "ur" + 0]!]) =
      (List.range (nR (shOf sh))).foldl (fun c j => rget (shOf sh) rs (.sc j) + c * vA[vSc (shOf sh) j]!) wA[wV2]! +
        (List.range sh.ell).foldl (fun x j => x * (rget (shOf sh) rs (.fx j) * rget (shOf sh) rs (.fy j)))
          (rget (shOf sh) rs .f0) * vA[vUr (shOf sh)]! := by
    refine congrArg₂ (· + ·) ?_ ?_
    · rw [degs_length]
      exact foldl_congr2 (by rw [ov2]; rfl) fun j hj c => by
        simp only [hsc j (List.mem_range.1 hj), hvs j (List.mem_range.1 hj)]
    · rw [hf0, hur, flatMap_range_congr (fun j => [R s!"fx{j}", R s!"fy{j}"])
        (fun j => [rget (shOf sh) rs (.fx j), rget (shOf sh) rs (.fy j)]) sh.ell fun j hj => by
          simp only [hfx j hj, hfy j hj]]
      apply fOf_injective
      rw [fOf_treeVal, fOf_mul, fOf_prodPairs]
      · simp only [List.map_append, List.map_cons, List.map_nil, List.prod_append, List.prod_cons, List.prod_nil,
          mul_one]
      · simp
      · exact Nat.le_succ _
  cases hst : sh.stacked
  · have ht : sh.tLog = 0 := tLog_eq sh hst
    simp only [stepVals, outOf, hst, ht, shOf_tLog, shOf_ell, Bool.false_eq_true, ↓reduceIte, List.toList_toArray]
    exact congrArg₂ List.cons H2 (congrArg₂ List.cons H3 (congrArg₂ List.cons hbp rfl))
  · have ht : sh.tLog ≠ 0 := tLog_ne sh hst
    have hcl : R "claim" = rget (shOf sh) rs .claim := hR _ (mem_claim _ (tLog_ne sh hst))
    have heqs : offset (vParts sh) "eqs" = vEqs (shOf sh) := by
      simp only [vEqs, vEqp, vUr, vAy, vAx, vAlpha, shOf_ell, pLog_shOf, Flock.RecSparse.Shape.PP] at hveqs ⊢
      omega
    simp only [stepVals, outOf, hst, ht, shOf_tLog, shOf_sLog, shOf_ell, Flock.RecSparse.Shape.T,
      Flock.RecSparse.Shape.S, ↓reduceIte, List.toList_toArray]
    refine congrArg₂ List.cons H2 (congrArg₂ List.cons H3 (congrArg₂ List.cons ?_ (congrArg₂ List.cons hcl rfl)))
    apply fOf_injective
    rw [fOf_sumVal _ (List.cons_ne_nil _ _), fOf_foldl_sum, List.map_cons, List.sum_cons, hbp]
    simp only [List.zip_map', List.map_map, Function.comp_def]
    refine congrArg₂ (· + ·) rfl (congrArg List.sum (List.map_congr_left fun t _ => ?_))
    rw [ovt hst, fOf_mul, fOf_mul]
    refine congrArg₂ (· * ·) ?_ rfl
    rw [fOf_sumVal, fOf_foldl_sum, fOf_zero, zero_add, List.map_map]
    · simp only [Function.comp_def, heqs]
      rfl
    · rw [Ne, List.map_eq_nil_iff, List.range_eq_nil]
      exact (Nat.two_pow_pos _).ne'

/-- **`stepVals` is `outOf`**: at the builder's residuals (`rval`), `steps`' values are `outOf`'s output at `Defs`'
residuals (`resArr`). -/
theorem stepVals_eq (wA vA : Array F128) (τA : Array ℕ) :
    stepVals sh (rval sh (wA[·]!) (vA[·]!)) (wA[·]!) (vA[·]!) (τA[·]!) =
      (outOf (shOf sh) (resArr (shOf sh) wA vA) wA vA τA).1.toList :=
  stepVals_outOf sh _ _ wA vA τA fun _ hr => rval_rget sh wA vA hr

end Values

/-! ## The rows -/

theorem fresh_forIn {σ : Type} (f : ℕ → σ → M (ForInStep σ)) (hf : ∀ j s, Fresh (f j s)) :
    ∀ (l : List ℕ) (s : σ), Fresh (forIn l s f)
  | [], s => by rw [List.forIn_nil]; exact Fresh.pure' _
  | j :: l, s => by
      rw [List.forIn_cons]
      refine Fresh.bind' (hf j s) fun y => ?_
      cases y with
      | done b => exact Fresh.pure' _
      | yield b => exact fresh_forIn f hf l b

theorem fresh_forIn_range {σ : Type} (n : ℕ) (init : σ) (f : ℕ → σ → M (ForInStep σ)) (hf : ∀ j s, Fresh (f j s)) :
    Fresh (forIn [0:n] init f) := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  exact fresh_forIn f hf _ _

theorem fresh_mulF (a b : List Lin) : Fresh (Flock.RecSparse.mulF a b) := by
  unfold Flock.RecSparse.mulF
  refine Fresh.bind' (fresh_commitW a) fun _ => Fresh.bind' (fresh_commitW b) fun _ => ?_
  exact Fresh.bind' (Fresh.mapM' _ (fun _ => fresh_and _ _) _) fun _ => Fresh.pure' _

theorem fresh_pairsMul : ∀ fac : List (List Lin), Fresh (Flock.RecSparse.pairsMul fac)
  | [] => Fresh.pure' _
  | [_] => Fresh.pure' _
  | a :: b :: rest => by
      simp only [Flock.RecSparse.pairsMul]
      exact Fresh.bind' (fresh_mulF a b) fun _ => Fresh.bind' (fresh_pairsMul rest) fun _ => Fresh.pure' _

theorem fresh_tree : ∀ (n : ℕ) (fac : List (List Lin)), Fresh (Flock.RecSparse.tree n fac)
  | 0, _ => Fresh.pure' _
  | _ + 1, [] => Fresh.pure' _
  | _ + 1, [_] => Fresh.pure' _
  | n + 1, a :: b :: rest => by
      simp only [Flock.RecSparse.tree]
      exact Fresh.bind' (fresh_pairsMul _) fun ps => fresh_tree n ps

theorem fresh_demuxLevel (tau : Array Lin) (j : ℕ) (lists : List (List (List Lin))) :
    Fresh (Flock.RecSparse.demuxLevel tau j lists) := by
  unfold Flock.RecSparse.demuxLevel
  refine Fresh.mapM' _ (fun p => ?_) _
  obtain ⟨xs, s⟩ := p
  refine Fresh.bind' (Fresh.mapM' _ (fun x => ?_) _) fun _ => Fresh.pure' _
  exact Fresh.bind' (Fresh.mapM' _ (fun b => fresh_and _ b) _) fun _ => Fresh.pure' _

theorem fresh_demux (tau : Array Lin) : ∀ (js : List ℕ) (lists : List (List (List Lin))),
    Fresh (Flock.RecSparse.demux tau js lists)
  | [], lists => by
      simp only [Flock.RecSparse.demux]
      exact Fresh.pure' _
  | j :: js, lists => by
      simp only [Flock.RecSparse.demux]
      exact Fresh.bind' (fresh_demuxLevel tau j lists) fun l => fresh_demux tau js l

/-- **`steps` adds rows only as `fresh` does.** -/
theorem fresh_steps (sh : Flock.RecSparse.Shape) (P : String → List Lin) : Fresh (Flock.RecSparse.steps sh P) := by
  unfold Flock.RecSparse.steps
  generalize Flock.RecSparse.structureOf sh = p
  obtain ⟨S, names⟩ := p
  dsimp only
  rw [Flock.RecResiduals.residualForms_eq, FlockVBridge.residualForms_eq]
  refine Fresh.bind' (FlockVBridge.fresh_residualsGo _ _ _ _) fun out => ?_
  refine Fresh.bind' (fresh_forIn_range _ _ _ fun _ _ => Fresh.bind' (fresh_mulF _ _) fun _ => Fresh.pure' _)
    fun c2 => ?_
  refine Fresh.bind' (fresh_forIn_range _ _ _ fun _ _ => Fresh.bind' (fresh_mulF _ _) fun _ => Fresh.pure' _)
    fun c4 => ?_
  refine Fresh.bind' (fresh_tree _ _) fun phi => ?_
  split
  · simp only [pure_bind]
    refine Fresh.bind' (fresh_demux _ _ _) fun lists => ?_
    refine Fresh.bind' (Fresh.mapM' _ (fun x => ?_) _) fun sel => Fresh.pure' _
    obtain ⟨a, b⟩ := x
    exact fresh_mulF a b
  · simp only [pure_bind]
    exact Fresh.pure' _

/-! ## The ports by name -/

section Ports

variable (sh : Flock.RecSparse.Shape)

theorem portWords_eq (n : ℕ) : Flock.RecSparse.portWords n = VStar.portWordsOf n := by
  unfold Flock.RecSparse.portWords Flock.RecSparse.F_WORDS Flock.RecSparse.ROW_BLOCK VStar.portWordsOf
  omega

/-- The message ports' words. -/
def msgWords : List ℕ := (messagePorts sh).map fun p => VStar.portWordsOf p.2

/-- The ports after the message ports: with `T > 1` `tau`, then `v`. -/
def restPorts : List (String × ℕ) :=
  (if sh.stacked then [("tau", Flock.RecSparse.tauWords sh)] else []) ++
    [("v", VStar.portWordsOf (Flock.RecSparse.nv sh))]

theorem ports_eq :
    Flock.RecSparse.ports sh = ((messagePorts sh).map fun p => (p.1, VStar.portWordsOf p.2)) ++ restPorts sh := by
  unfold Flock.RecSparse.ports restPorts
  rw [List.append_assoc, portWords_eq]
  congr 1

theorem ports_snd : (Flock.RecSparse.ports sh).map (·.2) = msgWords sh ++ (restPorts sh).map (·.2) := by
  simp only [ports_eq, msgWords, List.map_append, List.map_map, Function.comp_def]

theorem inWords_eq : Flock.RecSparse.inWords sh = ((Flock.RecSparse.ports sh).map (·.2)).foldl (· + ·) 0 := by
  simp only [Flock.RecSparse.inWords, List.foldl_map]

/-- The input bit where `tau`'s port starts, after the message ports. -/
def tauBase : ℕ := 16 * (msgWords sh).sum

/-- The input bit where `v`'s port starts, after `tau`'s. -/
def vBaseS : ℕ := tauBase sh + (if sh.stacked then 16 * Flock.RecSparse.tauWords sh else 0)

theorem inWords_16 :
    16 * Flock.RecSparse.inWords sh = vBaseS sh + 16 * VStar.portWordsOf (Flock.RecSparse.nv sh) := by
  rw [inWords_eq, OfVStar.foldl_add, ports_snd]
  cases hst : sh.stacked <;> simp [restPorts, vBaseS, tauBase, hst, List.sum_append] <;> omega

/-- The input bits `unitNet'` gives port `nm`. -/
def portOf (nm : String) : List Lin :=
  ((((Flock.RecSparse.ports sh).zip (varSlices 0 ((Flock.RecSparse.ports sh).map (·.2)))).find?
    (·.1.1 = nm)).map (·.2)).getD []

/-- **The unit is `steps` on its ports by name.** -/
theorem unitNet'_run : (Flock.RecSparse.unitNet' sh).run St.empty =
    (Flock.RecSparse.steps sh (portOf sh)).run ⟨16 * Flock.RecSparse.inWords sh, []⟩ := by
  unfold Flock.RecSparse.unitNet'
  dsimp only
  rw [run_input_bind', inWords_eq, slices_eq, slices_vars_foldl]
  rfl

/-- The ports' names. -/
inductive PName where
  | h
  | pre (j : ℕ)
  | v2
  | sc (j : ℕ)
  | u
  | vt
  | claim
  | tau
  | v

def pstr : PName → String
  | .h => "h"
  | .pre j => s!"pre{j}"
  | .v2 => "v2"
  | .sc j => s!"sc{j}"
  | .u => "u"
  | .vt => "vt"
  | .claim => "claim"
  | .tau => "tau"
  | .v => "v"

def ptag : PName → List Char
  | .h => ['h'] | .pre _ => ['p', 'r'] | .v2 => ['v', '2'] | .sc _ => ['s', 'c'] | .u => ['u'] | .vt => ['v', 't']
  | .claim => ['c', 'l'] | .tau => ['t', 'a'] | .v => ['v']

theorem take2_pstr : ∀ p, (pstr p).toList.take 2 = ptag p
  | .h => by decide
  | .pre j => take2_idx "pre" 'p' 'r' ['e'] (by decide) j
  | .v2 => by decide
  | .sc j => take2_idx "sc" 's' 'c' [] (by decide) j
  | .u => by decide
  | .vt => by decide
  | .claim => by decide
  | .tau => by decide
  | .v => by decide

theorem pstr_inj {a b : PName} (h : pstr a = pstr b) : a = b := by
  have ht : ptag a = ptag b := by rw [← take2_pstr, ← take2_pstr, h]
  cases a <;> cases b <;> simp only [pstr, ptag] at h ht ⊢ <;>
    first | rfl | exact absurd ht (by decide) | exact congrArg _ (idx_inj _ h)

def portKinds : List PName :=
  [.h] ++ (List.range 6).map .pre ++ [.v2] ++ (List.range sh.degrees.length).map .sc ++ [.u] ++
    (if sh.stacked then [.vt] else []) ++ [.claim] ++ (if sh.stacked then [.tau] else []) ++ [.v]

theorem ports_names : (Flock.RecSparse.ports sh).map (·.1) = (portKinds sh).map pstr := by
  rw [ports_eq, mp_eq]
  cases hst : sh.stacked <;> simp [portKinds, pstr, restPorts, hst, List.map_map, Function.comp_def]

theorem nodup_portKinds : (portKinds sh).Nodup := by
  have hp : ((List.range 6).map PName.pre).Nodup :=
    nodup_map_inj PName.pre (fun _ _ h => PName.pre.inj h) List.nodup_range
  have hs : ((List.range sh.degrees.length).map PName.sc).Nodup :=
    nodup_map_inj PName.sc (fun _ _ h => PName.sc.inj h) List.nodup_range
  cases hst : sh.stacked <;>
    simp [portKinds, List.nodup_append, List.nodup_cons, hp, hs, hst, or_imp, forall_and, forall_exists_index, and_imp]

theorem nodup_ports : ((Flock.RecSparse.ports sh).map (·.1)).Nodup := by
  rw [ports_names]
  exact nodup_map_inj pstr (fun _ _ => pstr_inj) (nodup_portKinds sh)

theorem nodup_zip_ports :
    (((Flock.RecSparse.ports sh).zip (varSlices 0 ((Flock.RecSparse.ports sh).map (·.2)))).map (·.1.1)).Nodup := by
  rw [map_fst_fst_zip _ _ (by simp [length_varSlices])]
  exact nodup_ports sh

theorem portOf_mem {e : (String × ℕ) × List Lin}
    (he : e ∈ (Flock.RecSparse.ports sh).zip (varSlices 0 ((Flock.RecSparse.ports sh).map (·.2)))) :
    portOf sh e.1.1 = e.2 := by
  unfold portOf
  rw [find_of_mem (nodup_zip_ports sh) he]
  rfl

theorem zip_ports : (Flock.RecSparse.ports sh).zip (varSlices 0 ((Flock.RecSparse.ports sh).map (·.2))) =
    ((messagePorts sh).zip (varSlices 0 (msgWords sh))).map (Prod.map (fun p => (p.1, VStar.portWordsOf p.2)) id) ++
      (restPorts sh).zip (varSlices (0 + 16 * (msgWords sh).sum) ((restPorts sh).map (·.2))) := by
  rw [ports_snd, varSlices_append, ports_eq, List.zip_append, List.zip_map_left]
  simp [length_varSlices, msgWords]

theorem portOf_msg {p : String × ℕ} {b : List Lin} (h : (p, b) ∈ (messagePorts sh).zip (varSlices 0 (msgWords sh))) :
    portOf sh p.1 = b :=
  portOf_mem sh (e := ((p.1, VStar.portWordsOf p.2), b))
    (by rw [zip_ports]; exact List.mem_append.2 (Or.inl (List.mem_map.2 ⟨(p, b), h, rfl⟩)))

theorem length_portOf {nm : String} {n : ℕ} (hp : (nm, n) ∈ messagePorts sh) :
    (portOf sh nm).length = 16 * VStar.portWordsOf n := by
  obtain ⟨k, hk⟩ := mem_zip_varSlices (fun p : String × ℕ => VStar.portWordsOf p.2) _ 0 hp
  have h : portOf sh nm = vars k (16 * VStar.portWordsOf n) := portOf_msg sh hk
  rw [h, length_vars]

theorem nodup_mp : ((messagePorts sh).map (·.1)).Nodup := by
  have h := nodup_ports sh
  rw [ports_eq, List.map_append, List.map_map] at h
  exact (List.nodup_append.1 h).1

/-- The message ports' first bits are `rowForms`'. -/
theorem flatMap_portOf :
    (messagePorts sh).flatMap (fun p => (portOf sh p.1).take (128 * p.2)) = rowForms (messagePorts sh) 0 := by
  rw [flatMap_zip_of (fun p b => b.take (128 * p.2)) (fun p => portOf sh p.1) (messagePorts sh)
    (varSlices 0 (msgWords sh)) (by simp [length_varSlices, msgWords]) fun p b h => portOf_msg sh h]
  have := flatMap_rowForms (fun e => e.2.take (128 * e.1.2)) (fun _ _ => rfl) (messagePorts sh) 0 []
  rw [List.append_nil] at this
  exact this

theorem portOf_v : portOf sh "v" = vars (vBaseS sh) (16 * VStar.portWordsOf (Flock.RecSparse.nv sh)) :=
  portOf_mem sh (e := (("v", VStar.portWordsOf (Flock.RecSparse.nv sh)),
    vars (vBaseS sh) (16 * VStar.portWordsOf (Flock.RecSparse.nv sh)))) (by
      rw [zip_ports]
      refine List.mem_append.2 (Or.inr ?_)
      cases hst : sh.stacked <;> simp [restPorts, vBaseS, tauBase, hst, varSlices])

theorem portOf_tau (hst : sh.stacked = true) :
    portOf sh "tau" = vars (tauBase sh) (16 * Flock.RecSparse.tauWords sh) :=
  portOf_mem sh (e := (("tau", Flock.RecSparse.tauWords sh), vars (tauBase sh) (16 * Flock.RecSparse.tauWords sh))) (by
    rw [zip_ports]
    refine List.mem_append.2 (Or.inr ?_)
    simp [restPorts, tauBase, hst, varSlices])

/-! ## The inputs -/

/-- The input columns hold `w`'s bits where the message ports put them (`rowCol`), `vv`'s on `v`'s port and with
`T > 1` each slot's type on `tau`'s, `w` and `vv` zero past the ports' elements. -/
structure SparseIn (w vv : ℕ → F128) (τ : ℕ → ℕ) (v : ℕ → Bool) : Prop where
  hw : ∀ q < 128 * ((messagePorts sh).map (·.2)).sum, v (rowCol (messagePorts sh) 0 q) = wbit w q
  hv : ∀ q < 128 * Flock.RecSparse.nv sh, v (vBaseS sh + q) = wbit vv q
  htau : sh.stacked = true → ∀ s < sh.S, ∀ b < 16, v (tauBase sh + (16 * s + b)) = (τ s).testBit b
  hw0 : ∀ e, ((messagePorts sh).map (·.2)).sum ≤ e → w e = 0
  hv0 : ∀ e, Flock.RecSparse.nv sh ≤ e → vv e = 0

theorem sparseIn_col {w vv : ℕ → F128} {τ : ℕ → ℕ} {v z : ℕ → Bool} (hin : SparseIn sh w vv τ v)
    (hz : ∀ x < 16 * Flock.RecSparse.inWords sh, z x = v x) : SparseIn sh w vv τ z := by
  have h16 := inWords_16 sh
  have hvb : tauBase sh ≤ vBaseS sh := Nat.le_add_right _ _
  have htb : tauBase sh = 16 * ((messagePorts sh).map fun p => VStar.portWordsOf p.2).sum := rfl
  refine ⟨fun q hq => ?_, fun q hq => ?_, fun hst s hs b hb => ?_, hin.hw0, hin.hv0⟩
  · have hrc := rowCol_lt (messagePorts sh) 0 q hq
    rw [hz _ (by omega)]
    exact hin.hw q hq
  · have := le_portWords (Flock.RecSparse.nv sh)
    rw [hz _ (by omega)]
    exact hin.hv q hq
  · have hvs : vBaseS sh = tauBase sh + 16 * Flock.RecSparse.tauWords sh := by simp only [vBaseS, hst, ↓reduceIte]
    have hST : sh.S ≤ Flock.RecSparse.tauWords sh := by
      unfold Flock.RecSparse.tauWords Flock.RecSparse.ROW_BLOCK
      omega
    rw [hz _ (by omega)]
    exact hin.htau hst s hs b hb

/-- `SparseIn` gives `StepsIn` at the ports by name. -/
theorem stepsIn_of {z : Var → Bool} {w vv : ℕ → F128} {τ : ℕ → ℕ} (hin : SparseIn sh w vv τ z) :
    StepsIn sh (portOf sh) z w vv τ := by
  have HW : ∀ q, bit z ((messagePorts sh).flatMap fun p => (portOf sh p.1).take (128 * p.2)) q = wbit w q := by
    intro q
    rw [flatMap_portOf]
    by_cases hq : q < 128 * ((messagePorts sh).map (·.2)).sum
    · rw [bit_rowForms _ _ _ _ hq]
      exact hin.hw q hq
    · rw [bit_of_length_le (by rw [length_rowForms]; omega), wbit, hin.hw0 _ (by omega), bitOf_zero]
  have hlen : ∀ p ∈ messagePorts sh, 128 * p.2 ≤ (portOf sh p.1).length := fun p hp => by
    rw [length_portOf sh (nm := p.1) (n := p.2) hp]
    exact le_portWords _
  have hoff : ∀ {nm : String} {n : ℕ}, (nm, n) ∈ messagePorts sh → ∀ t < n, ∀ k < 128,
      bit z (portOf sh nm) (128 * t + k) = bitOf (w (offset (messagePorts sh) nm + t)) k := by
    intro nm n hp t ht k hk
    rw [← bit_flatMap_offset z (portOf sh) _ (nodup_mp sh) hlen hp (128 * t + k) (by omega), HW,
      show 128 * offset (messagePorts sh) nm + (128 * t + k) = 128 * (offset (messagePorts sh) nm + t) + k by omega,
      wbit_elem w _ hk]
  have hv2m : ("v2", 1) ∈ messagePorts sh := by
    rw [mp_eq]
    simp
  have hvtm : sh.stacked = true → ("vt", sh.T) ∈ messagePorts sh := fun hst => by
    rw [mp_eq]
    simp [hst]
  have hST : sh.S ≤ Flock.RecSparse.tauWords sh := by
    unfold Flock.RecSparse.tauWords Flock.RecSparse.ROW_BLOCK
    omega
  refine ⟨HW, fun q => ?_, ?_, fun hst t ht => ?_, fun hst s hs b hb => ?_⟩
  · rw [portOf_v, take_vars _ _ _ (le_portWords _)]
    by_cases hq : q < 128 * Flock.RecSparse.nv sh
    · rw [bit_vars _ _ _ hq]
      exact hin.hv q hq
    · rw [bit_of_length_le (by rw [length_vars]; omega), wbit, hin.hv0 _ (by omega), bitOf_zero]
  · refine holds_elem (by rw [length_portOf sh hv2m]; have := le_portWords 1; omega) fun k hk => ?_
    have := hoff hv2m 0 (by omega) k hk
    rwa [Nat.add_zero] at this
  · exact holds_elem (by rw [length_portOf sh (hvtm hst)]; have := le_portWords sh.T; omega) (hoff (hvtm hst) t ht)
  · have hj : 16 * s + b < 16 * Flock.RecSparse.tauWords sh := by omega
    rw [portOf_tau sh hst, bit_vars _ _ _ hj]
    exact hin.htau hst s hs b hb

/-! ## The laid-out unit -/

/-- The unit's input ports, one 16-bit port per word. -/
def sparseInBits : Array ℕ := Array.replicate (Flock.RecSparse.inWords sh) 16

/-- Where its output columns start. -/
def sparseBase : ℕ :=
  outStart ((Flock.RecSparse.unitNet' sh).run St.empty).2 (sparseInBits sh)
    (Array.replicate (Flock.RecSparse.outWords sh) 16)

/-- **The laid-out sparse-phase unit computes `outOf`**: whenever its rows hold at `v`, if its input columns hold `w`,
`vv` and the slots' types `τ` (`SparseIn`), its output columns from `sparseBase` hold `outOf`'s residuals at `Defs`'
residuals (`resArr`), 128 bits each, and with `T > 1` from bit 512 each type with its low `T_LOG` bits cleared. At
`T_LOG ≤ 16`. -/
theorem laid_sparse (ht16 : sh.tLog ≤ 16) (v : ℕ → Bool)
    (h : LaidRows (Flock.RecSparse.unitLaid sh) (inRowsOf (sparseInBits sh)) v) (wA vA : Array F128) (τA : Array ℕ)
    (hin : SparseIn sh (wA[·]!) (vA[·]!) (τA[·]!) v) :
    (∀ r (hr : r < (outOf (shOf sh) (resArr (shOf sh) wA vA) wA vA τA).1.size), ∀ k < 128,
      v (sparseBase sh + (128 * r + k)) = bitOf (outOf (shOf sh) (resArr (shOf sh) wA vA) wA vA τA).1[r] k) ∧
    (sh.stacked = true → ∀ s < sh.S, ∀ b < 16,
      v (sparseBase sh + (512 + (16 * s + b))) = (τA[s]! - τA[s]! % 2 ^ sh.tLog).testBit b) := by
  have hN : (sparseInBits sh).foldl (· + ·) 0 = 16 * Flock.RecSparse.inWords sh := by
    rw [sparseInBits, foldl_replicate]
    exact Nat.mul_comm _ _
  obtain ⟨R, hR⟩ : ∃ R, (Flock.RecSparse.steps sh (portOf sh)).run ⟨16 * Flock.RecSparse.inWords sh, []⟩ = R :=
    ⟨_, rfl⟩
  have hrun : (Flock.RecSparse.unitNet' sh).run St.empty = R := (unitNet'_run sh).trans hR
  have hL : LaidRows (layout R.2 (sparseInBits sh) (Array.replicate (Flock.RecSparse.outWords sh) 16) R.1.toArray)
      (inRowsOf (sparseInBits sh)) v := by
    have e : Flock.RecSparse.unitLaid sh = layout ((Flock.RecSparse.unitNet' sh).run St.empty).2 (sparseInBits sh)
        (Array.replicate (Flock.RecSparse.outWords sh) 16) ((Flock.RecSparse.unitNet' sh).run St.empty).1.toArray := rfl
    rw [e, hrun] at h
    exact h
  have hg : Good ((sparseInBits sh).foldl (· + ·) 0) R.2 := by
    rw [hN, ← hR]
    exact fresh_steps sh (portOf sh) (16 * Flock.RecSparse.inWords sh) ⟨16 * Flock.RecSparse.inWords sh, []⟩
      ⟨by simp, fun k r h => by simp at h⟩
  have hrows := laid_rows _ _ _ _ v hg hL
  have hcol : ∀ x < 16 * Flock.RecSparse.inWords sh,
      v (col ((sparseInBits sh).foldl (· + ·) 0) (inRowsOf (sparseInBits sh)) x) = v x := by
    intro x hx
    rw [hN]
    simp only [col, hx, ↓reduceIte]
  have hPv : 128 * Flock.RecSparse.nv sh ≤ (portOf sh "v").length := by
    rw [portOf_v, length_vars]
    exact le_portWords _
  have hτ : sh.stacked = true → 16 * sh.S ≤ (portOf sh "tau").length := fun hst => by
    rw [portOf_tau sh hst, length_vars]
    unfold Flock.RecSparse.tauWords Flock.RecSparse.ROW_BLOCK
    omega
  have hS := (sound_steps sh (portOf sh) hPv hτ ht16 (fun _ hu => used_mem sh hu)
    ⟨16 * Flock.RecSparse.inWords sh, []⟩).2
  rw [hR] at hS
  obtain ⟨hlen, hbits, htail⟩ := hS _ hrows _ _ _ (stepsIn_of sh (sparseIn_col sh hin hcol))
  rw [stepVals_eq] at hbits
  have h4 : (outOf (shOf sh) (resArr (shOf sh) wA vA) wA vA τA).1.size ≤ 4 := by
    simp only [outOf]
    split <;> simp
  have hob4 : 512 ≤ 16 * Flock.RecSparse.outWords sh := by
    have : 3 ≤ (Flock.RecSparse.residualNames sh).length := by
      unfold Flock.RecSparse.residualNames
      split <;> simp
    unfold Flock.RecSparse.outWords Flock.RecSparse.F_WORDS Flock.RecSparse.ROW_BLOCK
    omega
  have hobS : sh.stacked = true → 512 + 16 * sh.S ≤ 16 * Flock.RecSparse.outWords sh := fun hst => by
    have how := outWords_ge sh
    have hrn : (Flock.RecSparse.residualNames sh).length = 4 := by simp [Flock.RecSparse.residualNames, hst]
    simp only [hst, ↓reduceIte] at how
    omega
  have hsz : R.1.toArray.size = 16 * Flock.RecSparse.outWords sh := by rw [List.size_toArray, hlen]
  have hob : (Array.replicate (Flock.RecSparse.outWords sh) 16).foldl (· + ·) 0 = 16 * Flock.RecSparse.outWords sh := by
    rw [foldl_replicate]
    exact Nat.mul_comm _ _
  have hout := fun j (hj : j < R.1.toArray.size) => laid_outs _ _ _ _ v (by rw [hsz, hob]) hL j hj
  have hbit : ∀ j < 16 * Flock.RecSparse.outWords sh, v (sparseBase sh + j) =
      bit (fun x => v (col ((sparseInBits sh).foldl (· + ·) 0) (inRowsOf (sparseInBits sh)) x)) R.1 j := by
    intro j hj
    have hj' : j < R.1.length := by rw [hlen]; exact hj
    unfold sparseBase
    rw [hrun, hout j (by rw [hsz]; exact hj)]
    simp only [List.getElem_toArray, bit, List.getElem?_eq_getElem hj', Option.getD_some]
  refine ⟨fun r hr k hk => ?_, fun hst s hs b hb => ?_⟩
  · have e := hbits r (by rw [Array.length_toList]; exact hr) k hk
    rw [Array.getElem_toList] at e
    exact (hbit (128 * r + k) (by omega)).trans e
  · exact (hbit (512 + (16 * s + b)) (by have := hobS hst; omega)).trans (htail hst s hs b hb)

end Ports

end FlockVBridge.Sparse
