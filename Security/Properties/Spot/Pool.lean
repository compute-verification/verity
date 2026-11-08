module

public import Properties.Spot
public import Definitions.NetworkCertifier.PoolRoots

/-!
# A period's pool chain, from R7 on each of its records (`Properties.Spot.PeriodPoolChained`)

Take a monitoring period's windows `i < m`: the record `rec i` is checked by a verifier `v i` with the period's pool,
and each verifier's log ends at the period's previous record. R7's pool chain on every record (`PoolChained`) then
chains the records' entries as `NetTiming.PoolChained` chains the carried pool's rows (`PeriodPoolChained`). An entry is
read as the canonical texts of its domain and root (`canonText`), the texts `jeq` compares. Each window's entry for
`pin` is the previous window's last entry for `pout` where the window carries the previous window's pool
(`Registration.Pool.chains`). Otherwise it is the pool its verifier opens (`openOf`): the empty pool's root, filled from
`genesis`, at a registration's first window or after a failed window or another program's; and at the period's first
window, possibly the pool of the window before the period.
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

/-- **R7 chains a period's pools** (`PeriodPoolChained`). Take a period's `m` records `rec i`, each checked by a verifier
`v i` with the pool `p i`. All of them name one value `pout` the pool is left in, and each verifier's log after the
first ends at the period's previous record. If R7's pool chain holds on every record (`PoolChained`), the records'
entries for `pin` chain as `NetTiming.PoolChained` asks. Where window `i > 0` carries the previous window's pool
(`Registration.Pool.chains`), its entry for `pin` is window `i - 1`'s last entry for `pout`. Otherwise it is the pool
its verifier opens. -/
def PeriodPoolChained : Prop :=
  ∀ {m : ℕ} (rec : Fin m → Json) (v : Fin m → Registration.Verifier) (p : Fin m → Registration.Pool) (pout : String),
    (∀ i, (v i).pool = some (p i)) → (∀ i, (p i).pout = pout) →
    (∀ i : Fin m, 0 < i.val → (v i).log.back? = some (rec (NetTiming.prevRun i))) →
    (∀ i, PoolChained (rec i) (v i)) →
    NetTiming.PoolChained (fun i => (p i).chains (v i).log (Flock.hexOf (v i).program.sha512))
      (fun i => openOf (rec i) (v i) (p i)) (fun i => pinOf (rec i) (p i)) (fun i => poutOf (rec i) pout)

end Properties.Spot
