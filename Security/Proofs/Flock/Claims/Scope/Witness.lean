import Proofs.Flock.Claims.Scope.Fit

/-!
# A readable witness from the tables' model witnesses

`witOf T J z`: the hidden values of `J` tables of `T` bits each, table `c`'s bit `a` from `z c a`, and no salts.
`zOf_witOf`: each table's share of it is its own. `relS_witOf`: at a statement the verifier parses (untyped, without
public input ports), when each table's `z c` satisfies its `Layout.model`, `witOf` satisfies `RelS` of the readable
statement: what an extractor holding the tables' decoded witnesses returns.
-/

namespace Flock.Readable

/-- The hidden values of `J` tables of `T` bits each, table `c`'s bit `a` from `z c a`; no salts. -/
def witOf (T J : ℕ) (z : ℕ → ℕ → ZMod 2) : Claims.Flock.Wit :=
  ⟨(List.range (J * T)).map fun k => ofZ (z (k / T) (k % T)), []⟩

theorem zOf_witOf {T J : ℕ} (z : ℕ → ℕ → ZMod 2) {c m : ℕ} (hc : c < J) (hm : 2 ^ m ≤ T) (a : Fin (2 ^ m)) :
    zOf T (witOf T J z) c m a = z c a.val := by
  have ha : a.val < T := lt_of_lt_of_le a.isLt hm
  have hk : c * T + a.val < J * T := by
    have := Nat.mul_le_mul_right T (show c + 1 ≤ J by omega)
    rw [Nat.succ_mul] at this
    omega
  have hT : 0 < T := by omega
  simp only [zOf, witOf, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range hk, Option.map_some,
    Option.getD_some]
  rw [show c * T + a.val = a.val + c * T by omega, Nat.add_mul_div_right _ _ hT, Nat.div_eq_of_lt ha, Nat.zero_add,
    Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt ha, toZ_ofZ]

/-- **The tables' model witnesses as a readable witness**: when each table's `z c` satisfies its `Layout.model` at
`regionsOf`, `witOf` satisfies `RelS` of the readable statement. -/
theorem relS_witOf (H : Claims.Msg → Claims.Msg) {p : Parsed} (hp : Parses p) (ht : p.tags.typed = false)
    (hpi : p.publicInputs = none) (z : ℕ → ℕ → ZMod 2)
    (hz : ∀ c (hc : c < p.sts.size), ∃ L : FlockSoundness.Layout p.sts[c],
      (L.model (FlockSoundness.Refine.regionsOf p.sts[c])).Satisfies fun a => z c a.val) :
    Claims.Flock.RelS H (toReadable relation p) (witOf (2 ^ (toReadable relation p).m) p.sts.size z) := by
  rw [relS_layout H hp ht hpi]
  intro c hc
  obtain ⟨L, hL⟩ := hz c hc
  have h0 : 0 < p.sts.size := by omega
  have hm : p.sts[c].m = (toReadable relation p).m := by
    rw [show p.sts[c].m = p.sts[0].m from (tables_fit (tablesOf_of_parses hp) ht hpi hc).1]
    simp only [toReadable, Array.getElem?_eq_getElem h0, Option.map_some, Option.getD_some]
  have he : zOf (2 ^ (toReadable relation p).m) (witOf (2 ^ (toReadable relation p).m) p.sts.size z) c p.sts[c].m =
      fun a => z c a.val := funext fun a => zOf_witOf z hc (by rw [hm]) a
  exact ⟨L, he ▸ hL⟩

end Flock.Readable
