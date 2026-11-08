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
  simp only [pinOf, openOf, he, ho, entryKey, jeq_canonText hd, jeq_canonText hr]

theorem openOf_of_chains {rec prev : Json} {v : Registration.Verifier} {p : Registration.Pool}
    (hc : p.chains v.log (Flock.hexOf v.program.sha512) = true) (hlog : v.log.back? = some prev) {d r : String}
    (ho : Registration.poolOpen rec v p = some (d, r)) : openOf rec v p = poutOf prev p.pout := by
  unfold openOf
  rw [ho]
  unfold Registration.poolOpen at ho
  rw [if_pos hc, hlog, Option.map_some] at ho
  unfold poutOf
  rcases hreg : getD prev "registered" with _ | _ | _ | _ | es | _ <;> simp only [hreg] at ho
  · rcases hf : es.reverse.find? (fun e => (e matches .obj _) && jeq (getD e "value") (Json.str p.pout)) with _ | e <;>
      simp only [hf] at ho
    · rcases hd : getD e "domain" with _ | _ | _ | d' | _ | _ <;>
        rcases hr : getD e "root" with _ | _ | _ | r' | _ | _ <;> simp only [hd, hr] at ho
      cases ho
      simp only [entryKey, hd, hr]

end Properties.Spot

namespace Properties.Spot.SecurityProofs

theorem periodPoolChained : PeriodPoolChained := by
  intro m rec v p pout hp hout hlog hR7 i
  rw [pinOf_eq_openOf (hp i) (hR7 i)]
  unfold NetTiming.poolOpen
  split_ifs with hc
  · obtain ⟨e, d, r, -, ho, -, -⟩ := hR7 i (p i) (hp i)
    have h := openOf_of_chains (rec := rec i) (v := v i) (p := p i) hc.2 (hlog i hc.1) ho
    rw [hout i] at h
    exact h
  · rfl

end Properties.Spot.SecurityProofs
