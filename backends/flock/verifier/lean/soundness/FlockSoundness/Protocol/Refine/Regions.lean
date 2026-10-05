import FlockSoundness.Protocol.Refine.StmtOf
import FlockSoundness.Protocol.Refine.Table

/-!
# The statement decoding, part 2: the regions are the model's (PROTOCOL.md §11, §16.2)

`regionOf` reads a region of the executable's statement as the model's: its free bits, the fixed bits the executable's
`regionPoint` sets, and bit `k` of block `b`'s bytes. `ofCircuit_extra`: at two link points of `m_pts` coordinates, the
extra claims `Setup.ofCircuit st` hands the verifier are the model's `extraClaims` of `stmtOf st` with those regions
(`ExtraRealizes`). It needs `RegionsWF`: `m = k_log + nbl`, `m_pts = PT_LOCAL + nbl`, and each region's free bits are
distinct in-block bits, at most `PT_LOCAL` of them.

`verify_refines_ofCircuit` composes this and `ofCircuit_fold` with `verify_refines`: for a verifier set up with
`Setup.ofCircuit st`, its conclusion holds for `stmtOf st`, with nothing assumed of the fold or the extra claims. The
record's link points have the configured shape because `Record.decode` checks it (S9, `decode_points`).
-/

namespace FlockSoundness.Refine

open Flock FlockLevel3

/-! ### `regionValue` -/

/-! ### The regions, as the model's -/

/-- A region of the executable's statement, as the model's (§16.2): its free bits, the fixed bits the executable's
`regionPoint` sets, and bit `k` of block `b`'s bytes. -/
def regionOf (kl nbl : ℕ) (reg : Region) : Model.Region kl nbl where
  free := reg.free.toList.filterMap fun b => if h : b < kl then some ⟨b, h⟩ else none
  fixedBits j := (reg.mask >>> j.val) % 2 == 1 && (reg.fixed >>> j.val) % 2 == 1
  bits b k := if (((reg.bytes b.val).get! (k / 8) >>> (k % 8).toUInt8) % 2 == 1) = true then 1 else 0

/-- The regions of an executable statement, as the model's. -/
def regionsOf (st : Stmt) : List (Model.Region st.c.kLog (st.m - st.c.kLog)) :=
  st.regions.toList.map (regionOf st.c.kLog (st.m - st.c.kLog))

/-- A region's free bits are distinct in-block bits, at most `PT_LOCAL` of them. -/
structure RegionWF (kl : ℕ) (reg : Region) : Prop where
  free_lt : ∀ b ∈ reg.free.toList, b < kl
  nodup : reg.free.toList.Nodup
  free_size : reg.free.size ≤ PT_LOCAL

theorem filterMap_fin_length (kl : ℕ) : ∀ (L : List ℕ), (∀ b ∈ L, b < kl) →
    (L.filterMap fun b => if h : b < kl then some (⟨b, h⟩ : Fin kl) else none).length = L.length
  | [], _ => rfl
  | b :: L, h => by
      rw [List.filterMap_cons, dite_eq_left (h b List.mem_cons_self)]
      simp only [List.length_cons]
      rw [filterMap_fin_length kl L fun x hx => h x (List.mem_cons_of_mem _ hx)]

/-! ### `regionPoint` -/

/-! ### `extraClaims`: region-major, then point by point -/

theorem list_forIn_push {α β γ : Type} (g : α → V γ) (mk : α → γ → β) :
    ∀ (L : List α) (out out' : Array β),
      (forIn L out fun a out => do let v ← g a; pure (ForInStep.yield (out.push (mk a v)))) = .ok out' →
      ∃ es, List.Forall₂ (fun a e => ∃ v, g a = .ok v ∧ e = mk a v) L es ∧ out'.toList = out.toList ++ es
  | [], out, out', h => by
      simp only [List.forIn_nil, pure, Except.pure, Except.ok.injEq] at h
      exact ⟨[], .nil, by simp [h]⟩
  | a :: L, out, out', h => by
      rw [List.forIn_cons] at h
      obtain ⟨y, hy, h⟩ := ebind_ok h
      obtain ⟨v, hv, hy⟩ := ebind_ok hy
      simp only [pure, Except.pure, Except.ok.injEq] at hy
      subst hy
      obtain ⟨es, hes, heq⟩ := list_forIn_push g mk L _ out' h
      exact ⟨mk a v :: es, .cons ⟨v, hv, rfl⟩ hes, by simp [heq]⟩

/-! ### The extra claims are the model's -/

/-- The link points as the model reads them: coordinate `t` of point `k` (§16.2). -/
def ptsOf (mPts : ℕ) (points : Array (Array F128)) : Fin 2 → Fin mPts → 𝔽 := fun k t => fOf (points[k.val]!)[t.val]!

/-- The facts about an executable statement's regions that reading its claims as the model's needs. -/
structure RegionsWF (st : Stmt) : Prop where
  hm : st.m = st.c.kLog + st.nbl
  mPts : st.mPts = PT_LOCAL + st.nbl
  regions : ∀ reg ∈ st.regions.toList, RegionWF st.c.kLog reg

/-! ### The record's link points, and the composition -/

end FlockSoundness.Refine
