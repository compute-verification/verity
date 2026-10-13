import Proofs.Flock.C3.Opening.SrcLayer
import Proofs.Flock.VBridge.PartsCover
import Proofs.Flock.VBridge.Algebra.RingSwitch
import Proofs.Flock.Level3.Assembly

/-!
# n7's opening tie at V*'s layer, from V*'s sessions

`SessionAt` gives a copy of the `rec-opening` unit registering w for every part of the opening's S; `OpenTieL` needs
one holding ring switch `i`'s 128 slots of w in a run of its row-port elements. The part with the last residual does
(`cover_rs`): that residual, T against the batched sum, keeps every claim's `t_k[0]` as a term at the zero coins
`structureOf` runs at (`Lin` keeps a term whose coefficient is zero), so it reads `["wt", 128 k, 0]`
(`structureOf_rsT`), and a part reading a transpose holds its block in a run (`block_of_partS`).
`openTieL_of_session` is `OpenTieL` for a source reading the layer there (`ReadsFill`, which w's ring switches are:
`readsFill_w`), and `openTieL_of_openTie` for a source whose opening is on w's ring switches (`OpenTie`).
-/

namespace FlockC3.Opening

section Cover

open Flock Flock.RecAlgebra
open Flock.RecResiduals (groups positions pos partS)
open FlockVBridge.PartS (wIdx termE opE resE bind_ok mapM_forall₂ exists_of_forall₂ pos_some resE_ok)

variable {sh : Shape}

theorem keys_addAt (k' : Key) : ∀ (l : Terms Key) (k : Key) (v : F128),
    (k' ∈ l.map (·.1) ∨ k' = k) → k' ∈ (addAt l k v).map (·.1)
  | [], k, v, h => by simpa [addAt] using h
  | (k₁, x) :: r, k, v, h => by
    simp only [addAt]
    split
    · rename_i hk
      subst hk
      simp only [List.map_cons, List.mem_cons] at h ⊢
      rcases h with (h | h) | h
      · exact Or.inl h
      · exact Or.inr h
      · exact Or.inl h
    · simp only [List.map_cons, List.mem_cons] at h ⊢
      rcases h with (h | h) | h
      · exact Or.inl h
      · exact Or.inr (keys_addAt k' r k v (Or.inl h))
      · exact Or.inr (keys_addAt k' r k v (Or.inr h))

theorem keys_mergeT (k : Key) : ∀ (b a : Terms Key), (k ∈ a.map (·.1) ∨ k ∈ b.map (·.1)) →
    k ∈ (mergeT a b).map (·.1)
  | [], a, h => by simpa [mergeT] using h
  | kv :: b, a, h => keys_mergeT k b _ <| by
    simp only [List.map_cons, List.mem_cons] at h
    exact h.elim (.inl <| keys_addAt k a kv.1 kv.2 <| .inl ·) (·.elim (.inl <| keys_addAt k a kv.1 kv.2 <| .inr ·) .inr)

theorem keys_add {a b : Lin} {k : Key} (h : k ∈ a.t.map (·.1) ∨ k ∈ b.t.map (·.1)) :
    k ∈ (Val.add a b : Lin).t.map (·.1) :=
  keys_mergeT k b.t a.t h

theorem keys_foldl {k : Key} : ∀ (L : List (F128 × Lin)) (acc : Lin),
    (k ∈ acc.t.map (·.1) ∨ ∃ cx ∈ L, k ∈ cx.2.t.map (·.1)) →
      k ∈ (L.foldl (fun acc cx => Val.add acc (Val.sm cx.1 cx.2)) acc).t.map (·.1)
  | [], acc, h => by simpa using h
  | cx :: L, acc, h => by
    rw [List.foldl_cons]
    have hs : (Val.sm cx.1 cx.2 : Lin).t.map (·.1) = cx.2.t.map (·.1) := by
      simp [Val.sm, Lin.scale, List.map_map, Function.comp_def]
    refine keys_foldl L _ ?_
    rcases h with h | ⟨cx', hm, h⟩
    · exact Or.inl (keys_add (Or.inl h))
    · rcases List.mem_cons.1 hm with rfl | hm
      · exact Or.inl (keys_add (Or.inr (hs ▸ h)))
      · exact Or.inr ⟨cx', hm, h⟩

/-- A term of one of `dotV`'s values is one of the dot's: a zero coefficient keeps its key. -/
theorem keys_dotV {cs : Array F128} {xs : Array Lin} {j : ℕ} (hc : j < cs.size) (hx : j < xs.size) {k : Key}
    (h : k ∈ xs[j].t.map (·.1)) : k ∈ (dotV cs xs : Lin).t.map (·.1) := by
  unfold dotV
  rw [← Array.foldl_toList]
  refine keys_foldl _ _ (Or.inr ⟨(cs[j], xs[j]), ?_, h⟩)
  have hz : j < (cs.zip xs).size := by simp only [Array.size_zip]; omega
  have hm := Array.getElem_mem hz
  rw [Array.getElem_zip] at hm
  exact Array.mem_toList_iff.2 hm

/-- Ring switch `k`'s slots in the opening's w: the first `128 · claims` (`RecOpening.rounds`). -/
theorem wiO_rs (sh : Shape) {k v : ℕ} (hk : k < sh.claims) (hv : v < 128) :
    RecOpening.wi sh (.rs k v) = some (128 * k + v) := by
  unfold RecOpening.wi RecOpening.wNames RecOpening.rounds
  rw [Array.flatten_append]
  obtain ⟨hsz, hget⟩ := blocks_spec Key.rs sh.claims
  have hB : (((Array.range sh.claims).map fun k => (Array.range 128).map (Key.rs k)).flatten) =
      blocks Key.rs sh.claims := rfl
  rw [hB]
  have hl : ∀ i ≤ 128 * k + v, i < (blocks Key.rs sh.claims ++ (ligRounds sh.sched).flatten).size := by
    intro i hi; simp only [Array.size_append, hsz]; omega
  apply idxOf?_of
  · rw [Array.getElem?_append_left (hl _ (Nat.le_refl _)), Array.getElem?_append_left (by omega),
      hget _ (by omega)]
    congr 2 <;> omega
  · intro j hj he
    rw [Array.getElem?_append_left (hl _ (by omega)), Array.getElem?_append_left (by omega), hget _ (by omega)] at he
    simp only [Option.some.injEq, Key.rs.injEq] at he
    omega

theorem coefOps_has (ri : ℕ) : ∀ (l : List (Key × F128)) (st st' : BSt) (ps : Array (ℕ × ℕ)),
    OpsOkO sh st → RecOpening.coefOps sh ri l st = .ok (ps, st') →
    ∀ kx ∈ l, ∃ ab ∈ ps.toList, ∃ t, RecOpening.atomTerm sh kx.1 = .ok t ∧ st'.ops[ab.2]? = some #[t]
  | [], _, _, _, _, _, _, hm => by cases hm
  | (k, x) :: rest, st, st', ps, hi, hr, kx, hm => by
    obtain ⟨st0, st1, hg, hr⟩ := bm_bind hr
    cases hg
    obtain ⟨_, st2, hs, hr⟩ := bm_bind hr
    cases hs
    obtain ⟨o, st3, hv, hr⟩ := bm_bind hr
    obtain ⟨ps', st4, hc, hr⟩ := bm_bind hr
    cases hr
    obtain ⟨h3i, -, -, -, -, t, hop, ho⟩ := varOp_specO (hi.push (.coef ri k) x #[.v st.vNames.size]) hv
    obtain ⟨-, h4e, -, -⟩ := coefOps_specO #[] (fun _ _ => 0) ri rest st3 st' ps' h3i hc
    rcases List.mem_cons.1 hm with rfl | hm
    · exact ⟨(st.ops.size, o), by simp, t, hop, h4e.ops? ho⟩
    · obtain ⟨ab, hab, t', ht', ho'⟩ := coefOps_has ri rest st3 st' ps' h3i hc kx hm
      exact ⟨ab, by simp [hab], t', ht', ho'⟩

/-- `_build` gives each non-consistency term of a residual a product with the term's operand. -/
theorem buildRes_has {ri : ℕ} {f : Lin} {st st' : BSt} {res : Res} (hi : OpsOkO sh st)
    (hr : RecOpening.buildRes sh ri f st = .ok (res, st')) {k : Key} (hk : k ∈ f.t.map (·.1))
    (hc : k.isCons = false) :
    ∃ ab ∈ res.mul.toList, ∃ t, RecOpening.atomTerm sh k = .ok t ∧ st'.ops[ab.2]? = some #[t] := by
  unfold RecOpening.buildRes at hr
  obtain ⟨acc, st1, h1, hr⟩ := bm_bind hr
  obtain ⟨a1, a2, a3, a4, -, -, -⟩ := accTerms_specO #[] (fun _ _ => 0) h1
  obtain ⟨l1, st1', hs1, hr⟩ := bm_bind hr
  obtain ⟨hl1, h⟩ := liftB_ok hs1
  subst st1'
  obtain ⟨cm, st2, h2, hr⟩ := bm_bind hr
  obtain ⟨l2, st2', hs2, hr⟩ := bm_bind hr
  obtain ⟨-, h⟩ := liftB_ok hs2
  subst st2'
  obtain ⟨pm, st3, h3, hr⟩ := bm_bind hr
  obtain ⟨cp, st4, h4, hr⟩ := bm_bind hr
  cases hr
  have hi1 : OpsOkO sh st1 := hi.of_eq a1 a2 a3 a4
  obtain ⟨h2i, -, -, -⟩ := coefOps_specO #[] (fun _ _ => 0) ri l1 st1 st2 cm hi1 h2
  obtain ⟨h3i, h3e, -, -⟩ := prodOps_specO #[] (fun _ _ => 0) ri l2 st2 st3 pm h2i h3
  obtain ⟨-, h4e, -, -⟩ := constTerm_specO #[] h3i h4
  obtain ⟨kx, hkx, rfl⟩ := List.mem_map.1 hk
  have hm : kx ∈ l1 := (sortBy_perm hl1).mem_iff.2 (List.mem_filter.2 ⟨hkx, by simp [hc]⟩)
  obtain ⟨ab, hab, t, ht, ho⟩ := coefOps_has ri l1 st1 st2 cm hi1 h2 kx hm
  exact ⟨ab, by simp [hab], t, ht, (h3e.trans h4e).ops? ho⟩

theorem buildAll_has : ∀ (fs : List Lin) (ri : ℕ) (st st' : BSt) (rs : Array Res),
    OpsOkO sh st → RecOpening.buildAll sh fs ri st = .ok (rs, st') → ∀ f ∈ fs,
      ∃ ri' s₀ s₁ r, OpsOkO sh s₀ ∧ RecOpening.buildRes sh ri' f s₀ = .ok (r, s₁) ∧ Ext s₁ st' ∧ r ∈ rs
  | [], _, _, _, _, _, _, _, hf => by cases hf
  | f :: rest, ri, st, st', rs, hi, hr, g, hg => by
    obtain ⟨r, st1, h1, hr⟩ := bm_bind hr
    obtain ⟨rs', st2, h2, hr⟩ := bm_bind hr
    cases hr
    obtain ⟨h1i, -, -, -, -⟩ := buildRes_specO #[] (fun _ _ => 0) hi h1
    obtain ⟨-, h2e, -, -, -⟩ := buildAll_specO #[] (fun _ _ => 0) rest (ri + 1) st1 st' rs' h1i h2
    rcases List.mem_cons.1 hg with rfl | hg
    · exact ⟨ri, st, st1, r, hi, h1, h2e, by simp⟩
    · obtain ⟨ri', s₀, s₁, r', h0, h1', he, hm⟩ := buildAll_has rest (ri + 1) st1 st' rs' h1i h2 g hg
      exact ⟨ri', s₀, s₁, r', h0, h1', he, by simp [hm]⟩

/-- **The opening's S reads each ring switch's transpose as a product's operand**: its last residual, T against the
batched sum, has every claim's `t_k[0]` as a term (`ringSwitch_spec`), so `_build` gives it `["wt", 128 k, 0]`. -/
theorem structureOf_rsT {S : GfS} (hS : RecOpening.structureOf sh = .ok S) {i : ℕ} (hi : i < sh.claims) :
    ∃ r ∈ S.res, ∃ ab ∈ r.mul.toList, S.ops[ab.2]? = some #[.wt (128 * i) 0] := by
  unfold RecOpening.structureOf at hS
  obtain ⟨lins, hl, hb⟩ := bind_ok hS
  obtain ⟨k, hk⟩ := run'_ok hl
  obtain ⟨res, st, hB, rfl⟩ := build_okO hb
  unfold RecOpening.run at hk
  obtain ⟨⟨rsr, rs⟩, k₁, h1, hk⟩ := cm_bind hk
  obtain ⟨⟨lgr, lg⟩, k₂, -, hk⟩ := cm_bind hk
  cases cm_pure hk
  have hn : (RecOpening.claimsLike (α := Lin) sh).size = sh.claims := by simp [RecOpening.claimsLike]
  obtain ⟨betas, -, hbs, hg, -, hT, -, hj⟩ := ringSwitch_spec (by omega) h1
  simp only [Coins.zeros, Except.ok.injEq] at hg
  obtain ⟨c, hc, -, hbi, -⟩ := hj i (by omega)
  simp only [Coins.zeros, Except.ok.injEq] at hc
  obtain ⟨hbl, hbe⟩ := Array.getElem?_eq_some_iff.1 hbi
  have hM : (symbolic sh).rsT[i]! = (Array.range 128).map fun u => Lin.var (.rsT i u) := by
    have hi' :
        i < ((Array.range sh.claims).map fun k => (Array.range 128).map fun u => Lin.var (.rsT k u)).size := by
      simpa using hi
    simp only [symbolic, Function.comp_def]
    rw [getElem!_pos _ i hi', Array.getElem_map, Array.getElem_range]
  have h0 : Key.rsT i 0 ∈ (dotV (eqTable c) ((symbolic sh).rsT[i]!) : Lin).t.map (·.1) := by
    rw [hM]
    refine keys_dotV (j := 0) (by rw [FlockLevel3.eqTable_size, ← hc, Array.size_replicate]; decide) (by simp) ?_
    simp [Lin.var]
  have hlast : Key.rsT i 0 ∈ (Val.add ((symbolic sh).tAtom.getD rs.T) rs.T : Lin).t.map (·.1) := by
    refine keys_add (Or.inr ?_)
    rw [hT]
    refine keys_dotV (j := i) (by rw [← hg, Array.size_replicate]; omega) hbl ?_
    rw [hbe]
    exact h0
  obtain ⟨ri, s₀, s₁, r, h0i, hr, he, hrm⟩ :=
    buildAll_has _ 0 _ _ _ ⟨rfl, fun _ _ h => by simp at h⟩ hB (Val.add ((symbolic sh).tAtom.getD rs.T) rs.T)
      (by simp)
  obtain ⟨ab, hab, t, ht, ho⟩ := buildRes_has h0i hr hlast rfl
  simp only [RecOpening.atomTerm, wiO_rs sh hi (by omega : 0 < 128), Nat.add_zero] at ht
  cases ht
  exact ⟨r, hrm, ab, hab, he.ops? ho⟩

/-- **A part reading `["wt", e, u]` holds w's elements `e … e + 127` in a run of its row-port elements**: `partS`
refuses a transpose over row ports the part does not read in a run. -/
theorem block_of_partS {S : GfS} {sizes R P V : Array ℕ} {S' : RecOpen.GfS} (h : partS S sizes R P V = .ok S')
    {r : ℕ} (hr : r ∈ R.toList) (hrs : r < S.res.size) {ab : ℕ × ℕ} (hab : ab ∈ S.res[r].mul.toList)
    {e u : ℕ} (ho : S.ops[ab.2]? = some #[.wt e u]) : ∃ i', ∀ j < 128, (wIdx sizes P)[i' + j]? = some (e + j) := by
  unfold partS at h
  obtain ⟨used, -, h⟩ := bind_ok h
  obtain ⟨ops, hops, h⟩ := bind_ok h
  obtain ⟨rs, hrs', h⟩ := bind_ok h
  cases h
  have hops' : used.toList.mapM (opE S (wIdx sizes P) V) = .ok ops := hops
  have hrs'' : R.toList.mapM (resE S (wIdx sizes P) V used) = .ok rs := hrs'
  obtain ⟨b, hb⟩ := exists_of_forall₂ (mapM_forall₂ hrs'') r hr
  obtain ⟨_, -, hpos, -⟩ := resE_ok (w := fun _ => 0) (v := fun _ => 0) (w' := fun _ => 0) (v' := fun _ => 0)
    (fun _ _ => rfl) (fun _ _ => rfl) hb
  obtain ⟨p, hp⟩ := hpos ab hab ab.2 (by simp)
  obtain ⟨hpl, hpx⟩ := pos_some hp
  obtain ⟨o, hko⟩ := exists_of_forall₂ (mapM_forall₂ hops') ab.2
    (Array.mem_toList_iff.2 (hpx ▸ Array.getElem_mem hpl))
  unfold opE at hko
  split at hko
  · obtain ⟨_, hke⟩ := Array.getElem?_eq_some_iff.1 ho
    rw [hke] at hko
    obtain ⟨t', ht'⟩ := exists_of_forall₂ (mapM_forall₂ hko) (.wt e u) (by simp)
    simp only [termE] at ht'
    split at ht'
    · rename_i i' _
      split at ht'
      · rename_i hc
        simp only [Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true, List.mem_range, beq_iff_eq] at hc
        exact ⟨i', hc.2⟩
      · cases ht'
    · cases ht'
  · cases hko

/-- **(the coverage fact) Some part holds ring switch `i`'s 128 slots of w in a run of its row-port elements**: the
last residual reads `["wt", 128 i, 0]` (`structureOf_rsT`), some part holds it (`groups_cover`), and that part holds the
block (`block_of_partS`). -/
theorem cover_rs {S0 : GfS} {gs : Array (Array ℕ × Array ℕ × Array ℕ)} (hS : RecOpening.structureOf sh = .ok S0)
    (hg : groups S0 (RecOpening.portSizes sh) = .ok gs)
    (hparts : ∀ p < gs.size, ∃ S' ns nv, RecOpening.part sh p = .ok (S', ns, nv)) {i : ℕ} (hi : i < sh.claims) :
    ∃ p, ∃ hp : p < gs.size, ∃ k₀, ∀ t < 128,
      (wIdx (RecOpening.portSizes sh) gs[p].2.1)[k₀ + t]? = some (128 * i + t) := by
  obtain ⟨r, hr, ab, hab, ho⟩ := structureOf_rsT hS hi
  obtain ⟨n, hn, rfl⟩ := Array.getElem_of_mem hr
  obtain ⟨p, hp, k, hk, hpk⟩ := FlockVBridge.Parts.groups_cover hg hn
  obtain ⟨S', ns, nv, hpart⟩ := hparts p hp
  obtain ⟨k₀, hk₀⟩ := block_of_partS (FlockVBridge.OfVStar.part_eqOpening hS hg hp hpart).1
    (Array.mem_toList_iff.2 (hpk ▸ Array.getElem_mem hk)) hn hab ho
  exact ⟨p, hp, k₀, hk₀⟩

end Cover

section Sessions

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled FlockC3.Compose
open FlockVBridge.Decode (fill)
open FlockVBridge.OfVStar (fill_rowCol readCol_rsAt)

variable {nS : ℕ} {Vs : Fin nS → VStmt} {Reg' : Type}

/-- **The source reads ring switch `i` where V*'s layer holds it**: at the groups of the shape's S, whatever the layer
holds at a copy of the `rec-opening` unit registering w at ring switch `i`'s slots (`RsAt`) is `x i`. -/
def ReadsFill (zs : (s : Fin nS) → (Vs s).Outer Reg') (Lb : (s : Fin nS) → (Vs s).Bits) (sh : Flock.Shape)
    (w : Array F128) (x : Fin 2 → Fin 128 → F128) : Prop :=
  ∀ (S0 : RecAlgebra.GfS) (gs : Array (Array ℕ × Array ℕ × Array ℕ)), RecOpening.structureOf sh = .ok S0 →
    Flock.RecResiduals.groups S0 (RecOpening.portSizes sh) = .ok gs →
    ∀ (i : Fin 2) (y : Fin 128 → F128), RsAt zs Lb sh gs w i.val y → y = x i

/-- w's ring switches are read where the layer holds them (`eq_of_rsAt`). -/
theorem readsFill_w {zs : (s : Fin nS) → (Vs s).Outer Reg'} {Lb : (s : Fin nS) → (Vs s).Bits} {m : ℕ}
    {sh : Flock.Shape} (hsh : RecOpening.shape? m 2 = some sh) (w : Array F128) :
    ReadsFill zs Lb sh w fun i t => w[128 * i.val + t.val]! := by
  intro S0 gs hS hg i y hy
  funext t
  obtain ⟨e, hwi, hye⟩ := eq_of_rsAt hS hg hy t
  rw [wiO_rs sh (by rw [shape_claims hsh]; exact i.isLt) t.isLt] at hwi
  cases hwi
  exact hye

/-- **(n7's opening tie at V*'s layer, from V*'s sessions)** At V*'s sessions (`SessionAt`), some copy registering w
holds each ring switch (`cover_rs`, `fill_rowCol` at the columns it reads, `readCol_rsAt`), so a source reading the
layer there (`ReadsFill`), its opening steps the opening's on what it reads, has the tie at V*'s layer. -/
theorem openTieL_of_session {zs : (s : Fin nS) → (Vs s).Outer Reg'} {Lb : (s : Fin nS) → (Vs s).Bits} {m : ℕ}
    {sh : Flock.Shape} (hsh : RecOpening.shape? m 2 = some sh) {unit : String} {c3 : Flock.RecCoins.C3}
    {w : Array F128} {ops : FlockVBridge.AccChain.Openings} (hs : SessionAt zs Lb sh unit c3 w ops)
    {x : Fin 2 → Fin 128 → F128} (hx : ReadsFill zs Lb sh w x) {ρ : Fin 2 → Fin 7 → GF128}
    {γ : Fin 2 → GF128} {e : ExecHolo} {u : C3Run e} (hop : u.op = openSteps (fun i t => fOf (x i t)) ρ γ) :
    OpenTieL zs Lb sh w u := by
  obtain ⟨o, -, -, -, ⟨S, gs, sess, un, cp, hS, hg, hz⟩, -⟩ := hs
  refine ⟨S, gs, x, ρ, γ, hS, hg, fun i => ?_, hop⟩
  have hi : i.val < sh.claims := by rw [shape_claims hsh]; exact i.isLt
  obtain ⟨p, hp, k₀, hk₀⟩ := cover_rs hS hg (fun p hp => by
    obtain ⟨_, _, _, _, _, _, _, _, _, S', ns, nv, hpart, _⟩ := hz p hp
    exact ⟨S', ns, nv, hpart⟩) hi
  obtain ⟨_, hj, _, _, _, _, _, _, _, S', ns, nv, hpart, _⟩ := hz p hp
  have hR : RsAt zs Lb sh gs w i.val fun t => w[128 * i.val + t.val]! :=
    ⟨sess p, un p, cp p, p, k₀, _, hj, hp, S', ns, nv, hz p hp, hpart, fun t => ⟨_, hk₀ t t.isLt,
      wiO_rs sh hi t.isLt, fun b hb => by
        obtain ⟨hk, hke⟩ := Array.getElem?_eq_some_iff.mp (hk₀ t t.isLt)
        rw [fill_rowCol (zs (sess p)) (hz p hp) hj hS hg hp hpart hk hb
          (readCol_rsAt (zs (sess p)) (hz p hp) hS hg hp hpart (hk₀ t t.isLt) (wiO_rs sh hi t.isLt) hb), hke,
          ← RecAlgebra.getElem!_el]⟩⟩
  rw [← hx S gs hS hg i _ hR]
  exact hR

/-- **The tie is the tie at V*'s layer, at V*'s sessions** (`openTie_of_layer`'s converse): an opening on the ring
switches w holds (`OpenTie`) is on what the layer holds there (`openTieL_of_session` at `readsFill_w`, `decodedO_rs`
at w's slots `wiO_rs`). -/
theorem openTieL_of_openTie {zs : (s : Fin nS) → (Vs s).Outer Reg'} {Lb : (s : Fin nS) → (Vs s).Bits} {m : ℕ}
    {sh : Flock.Shape} (hsh : RecOpening.shape? m 2 = some sh) {unit : String} {c3 : Flock.RecCoins.C3}
    {w : Array F128} {ops : FlockVBridge.AccChain.Openings} (hs : SessionAt zs Lb sh unit c3 w ops) {e : ExecHolo}
    {u : C3Run e} (h : OpenTie sh w ops u) : OpenTieL zs Lb sh w u := by
  obtain ⟨ρ, γ, hop⟩ := h
  refine openTieL_of_session (x := fun i t => w[128 * i.val + t.val]!) hsh hs (readsFill_w hsh w)
    (hop.trans (congrArg (openSteps · ρ γ) (funext fun i => funext fun t => ?_)))
  have hi : i.val < sh.claims := by rw [shape_claims hsh]; exact i.isLt
  show fOf (((decodedO sh w (consO ops)).rs[i.val]!)[t.val]!) = fOf (w[128 * i.val + t.val]!)
  rw [decodedO_rs w (consO ops) hi t.isLt (wiO_rs sh hi t.isLt)]

end Sessions

end FlockC3.Opening
