module

public import Properties.Spot
public import Definitions.NetworkCertifier.Carried

/-! # A period's pool chain, from R7 on each of its records

Take a monitoring period's windows `i < m`: the record `rec i` is checked by a verifier `v i` with the period's pool,
and each verifier's log ends at the period's previous record. R7's pool chain on every record (`PoolChained`) then
chains the records' entries as `NetTiming.PoolChained` chains the carried pool's rows (`periodPoolChained`). An entry is
read as the canonical texts of its domain and root (`canonText`), the texts `jeq` compares. Each window's entry for
`pin` is the previous window's last entry for `pout` where the window carries the previous window's pool
(`Registration.Pool.chains`). Otherwise it is the pool its verifier opens (`openOf`): the empty pool's root, filled from
`genesis`, where the public rule restarts the chain (at a registration's first window, at another program's, and the
public lag `L` windows after one the verifier refused) or serving's reset does, at most once a week; and at the
period's first window, possibly the pool of the window before the period. Each run's registered roots, read from its
record (`dec`), then chain over digests as `NetTiming.Guarantees.RootsOfChecked` takes them (`rootsChainedOfR7`).

R7 holds a record's entry for `pin` to the pool its verifier opens up to canonical text (`pinOf_eq_openOf`, through
`jeq_canonText`). Where the window carries the previous window's pool, that pool is the last entry for `pout` of the
log's last record (`openOf_of_chains`), the period's previous record.
-/

@[expose] public section

open Lean (Json)
open Spot

namespace Properties.Spot

/-- A JSON value's canonical text, which `jeq` compares; none where it has none. -/
def canonText (x : Json) : Option String := (Flock.canon x).toOption

/-- An entry's domain and root, as canonical texts. -/
def entryKey (e : Json) : Option String × Option String := (canonText (getD e "domain"), canonText (getD e "root"))

/-- The record's entry for its pool's `pin` (`Registration.entryOf`), as canonical texts. -/
def pinOf (rec : Json) (p : Registration.Pool) : Option String × Option String :=
  match Registration.entryOf rec p.pin with
  | some e => entryKey e
  | none => (none, none)

/-- The pool a window that carries the previous window's pool opens, from the log's last record `back`
(`Registration.poolOpen`'s first branch): the domain and root of that record's last `registered` entry, an object, naming
`pout`; none where there is none. -/
def chainOpen (back : Option Json) (pout : String) : Option (String × String) :=
  match back.map (getD · "registered") with
  | some (.arr es) =>
    match es.reverse.find? fun e => (e matches .obj _) && jeq (getD e "value") (Json.str pout) with
    | some e => match getD e "domain", getD e "root" with
      | .str d, .str r => some (d, r)
      | _, _ => none
    | none => none
  | _ => none

/-- A pool's domain and root, as canonical texts. -/
def canonPair : Option (String × String) → Option String × Option String
  | some (d, r) => (canonText (.str d), canonText (.str r))
  | none => (none, none)

/-- The pool the record's window leaves for the next window to open (`chainOpen`), as canonical texts. -/
def poutOf (rec : Json) (pout : String) : Option String × Option String := canonPair (chainOpen (some rec) pout)

/-- The pool the verifier opens the record's window from (`Registration.poolOpen`), as canonical texts. -/
def openOf (rec : Json) (v : Registration.Verifier) (p : Registration.Pool) : Option String × Option String :=
  canonPair (Registration.poolOpen rec v p)

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

/-- **R7 chains a period's pools.** Take a period's `m` records `rec i`, each checked by a verifier `v i` with the pool
`p i`. All of them name one value `pout` the pool is left in, and each verifier's log after the first ends at the
period's previous record. If R7's pool chain holds on every record (`PoolChained`), the records' entries for `pin` chain
as `NetTiming.PoolChained` asks. Where window `i > 0` carries the previous window's pool (`Registration.Pool.chains`),
its entry for `pin` is window `i - 1`'s last entry for `pout`. Otherwise it is the pool its verifier opens. -/
theorem periodPoolChained {m : ℕ} (rec : Fin m → Json) (v : Fin m → Registration.Verifier)
    (p : Fin m → Registration.Pool) (pout : String) (hp : ∀ i, (v i).pool = some (p i)) (hout : ∀ i, (p i).pout = pout)
    (hlog : ∀ i j : Fin m, j.val + 1 = i.val → (v i).log.back? = some (rec j))
    (hR7 : ∀ i, PoolChained (rec i) (v i)) :
    NetTiming.PoolChained (fun i => (p i).chains (v i).log (Flock.hexOf (v i).program.sha512))
      (fun i => openOf (rec i) (v i) (p i)) (fun i => pinOf (rec i) (p i)) (fun i => poutOf (rec i) pout) := by
  intro i
  dsimp only
  rw [pinOf_eq_openOf (hp i) (hR7 i)]
  unfold NetTiming.poolOpen
  split_ifs with hc
  · have h := openOf_of_chains (rec := rec i) (v := v i) (p := p i) hc.2
      (hlog i ⟨i.val - 1, lt_of_le_of_lt (Nat.sub_le i.val 1) i.isLt⟩ (Nat.sub_add_cancel hc.1))
    rw [hout i] at h
    exact h
  · rfl

/-- **R7 chains the runs' registered roots.** Take a period's records and verifiers as in `periodPoolChained`, with R7's
pool chain on every record. Take the runs' registered roots read from their records by `dec`: run `i`'s opening root
`pinRoot i` from its entry for `pin`, its closing root `poutRoot i` from its last entry for `pout`, and, where it doesn't
carry the previous window's pool, the digest of its genesis row from the pool its verifier opens. Then the registered
roots chain under R7's restarts (`NetTiming.PoolChained` over the digests), as `NetTiming.Guarantees.RootsOfChecked`
takes them. -/
theorem rootsChainedOfR7 {m : ℕ} (rec : Fin m → Json) (v : Fin m → Registration.Verifier)
    (p : Fin m → Registration.Pool) (pout : String) (G : Hash.Msg → Hash.Digest) (genesis : Fin m → Hash.Row)
    (pinRoot poutRoot : Fin m → Hash.Digest) (dec : Option String × Option String → Hash.Digest)
    (hp : ∀ i, (v i).pool = some (p i)) (hout : ∀ i, (p i).pout = pout)
    (hlog : ∀ i j : Fin m, j.val + 1 = i.val → (v i).log.back? = some (rec j))
    (hR7 : ∀ i, PoolChained (rec i) (v i))
    (hin : ∀ i, pinRoot i = dec (pinOf (rec i) (p i))) (hrout : ∀ i, poutRoot i = dec (poutOf (rec i) pout))
    (hgen : ∀ i : Fin m, ¬(0 < i.val ∧ (p i).chains (v i).log (Flock.hexOf (v i).program.sha512) = true) →
      Hash.RowSeg.digest G (genesis i) = dec (openOf (rec i) (v i) (p i))) :
    NetTiming.PoolChained (fun i => (p i).chains (v i).log (Flock.hexOf (v i).program.sha512))
      (fun i => Hash.RowSeg.digest G (genesis i)) pinRoot poutRoot := by
  intro i
  have h := periodPoolChained rec v p pout hp hout hlog hR7 i
  dsimp only at h
  rw [hin i, h]
  unfold NetTiming.poolOpen
  split_ifs with hc
  · exact (hrout _).symm
  · exact (hgen i hc).symm

end Properties.Spot
