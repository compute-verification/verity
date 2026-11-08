module

public import Properties.Spot.Pool

/-! # A period's pool chain, from R7 on each of its records (`Properties.Spot.Pool`)

R7 holds a record's entry for `pin` to the pool its verifier opens up to canonical text (`pinOf_eq_openOf`, through
`jeq_canonText`). Where the window carries the previous window's pool, that pool is the last entry for `pout` of the
log's last record (`openOf_of_chains`), the period's previous record.
-/

@[expose] public section

open Lean (Json)
open Spot

namespace Properties.Spot

theorem jeq_canonText {a b : Json} (h : jeq a b = true) : canonText a = canonText b := by
  unfold jeq at h
  unfold canonText
  cases ha : Flock.canon a <;> cases hb : Flock.canon b <;> simp_all [Except.toOption]

theorem pinOf_eq_openOf {rec : Json} {v : Registration.Verifier} {p : Registration.Pool} (hp : v.pool = some p)
    (h : PoolChained rec v) : pinOf rec p = openOf rec v p := by
  obtain ⟨e, d, r, he, ho, hd, hr⟩ := h p hp
  simp only [pinOf, openOf, he, ho, entryKey, canonPair, jeq_canonText hd, jeq_canonText hr]

theorem poolOpen_of_chains (rec : Json) {v : Registration.Verifier} {p : Registration.Pool}
    (hc : p.chains v.log (Flock.hexOf v.program.sha512) = true) :
    Registration.poolOpen rec v p = chainOpen v.log.back? p.pout := by
  unfold Registration.poolOpen
  exact (ite_eq_left_of_eq_true _ _ (eq_true hc)).trans rfl

theorem openOf_of_chains {rec prev : Json} {v : Registration.Verifier} {p : Registration.Pool}
    (hc : p.chains v.log (Flock.hexOf v.program.sha512) = true) (hlog : v.log.back? = some prev) :
    openOf rec v p = poutOf prev p.pout := by
  unfold openOf poutOf
  rw [poolOpen_of_chains rec hc, hlog]

end Properties.Spot

namespace Properties.Spot.SecurityProofs

theorem periodPoolChained : PeriodPoolChained := by
  intro m rec v p pout hp hout hlog hR7 i
  dsimp only
  rw [pinOf_eq_openOf (hp i) (hR7 i)]
  unfold NetTiming.poolOpen
  split_ifs with hc
  · have h := openOf_of_chains (rec := rec i) (v := v i) (p := p i) hc.2 (hlog i hc.1)
    rw [hout i] at h
    exact h
  · rfl

end Properties.Spot.SecurityProofs
