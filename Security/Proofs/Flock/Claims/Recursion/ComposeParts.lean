import Proofs.Flock.Claims.Recursion.Deploy
import Proofs.Flock.Claims.Recursion.ComposeShared

/-!
# `flock-verify rec-compose` at the compiled verifier's parts

`RecDeploy.V` runs part `s` of a recursive statement at `partRaw r st x ms s p`: what every part shares
(`stagedShared`), with the part's own Flock statement, index and bindings (`partRaw_ofShared`). The deployed composition
runs the parts as separate `flock-verify verify --zk --rec-part J` processes, so nothing but `rec-compose` holds them to
one shared set. `composed_parts`: where `rec-compose` accepts the parts' verdicts (`RecCompose.Accepts`, which
`RecCompose.check_accepts` proves of every composition it accepts), either two of the parts' shared files are distinct
with one SHA-512, or every part's recursive statement is `RecRaw.ofShared S` at one shared set `S` in the compiled
verifier's form, at its own Flock statement, an index below V*'s count and its own bindings, as `partRaw` builds them;
one part per index, each accepted under `--zk` without an identifier, no two sessions on one coin key. Which
statement `S` stages (`readX`, the staging message) is the run's, not the composition's.

That the parts' sessions are one run of `RecDeploy.V` is the open condition C10 `OneRun`: one session per part, every
session the coin server opened for the statement's parts recorded, and the staging before the first part's Hello.
`rec-compose` sees only the sessions the prover hands it, so a part retried until it accepts passes. `OneRun` needs
the service's interactive-run record, a later PR; until it holds, `composed_parts` gives one shared set and fresh
keys, and the recursive verdict is not a claim a guarantee backs.
-/

namespace RecDeploy

open Flock
open FlockDeploy (Raw)

/-- **What a part's recursive statement shares** with the other parts: every field but its Flock statement, its index
and its bindings. -/
def RecRaw.shared (w : RecRaw) : RecShared :=
  ⟨w.recInner, w.recRows, w.innerCircuit, w.innerPublic, w.innerPartition, w.innerCoins, w.innerSession, w.transcript,
    w.registration, w.innerStatement, w.innerStatementCoins, w.innerTables, w.innerDraw⟩

/-- **Part `s`'s recursive statement** from what every part shares `S`, its Flock statement `base` and its bindings. -/
def RecRaw.ofShared (S : RecShared) (base : Raw) (s : ℕ) (bindings : Option ByteArray) : RecRaw where
  base := base
  recInner := S.recInner
  recPart := some s
  recBindings := bindings
  recRows := S.recRows
  innerCircuit := S.innerCircuit
  innerPublic := S.innerPublic
  innerPartition := S.innerPartition
  innerCoins := S.innerCoins
  innerSession := S.innerSession
  transcript := S.transcript
  registration := S.registration
  innerStatement := S.innerStatement
  innerStatementCoins := S.innerStatementCoins
  innerTables := S.innerTables
  innerDraw := S.innerDraw

theorem RecRaw.ofShared_shared {w : RecRaw} {s : ℕ} (h : w.recPart = some s) :
    RecRaw.ofShared w.shared w.base s w.recBindings = w := by
  cases w
  cases h
  rfl

/-- **What the compiled verifier's parts share** at the rounds `ms` before the sessions (`partRaw`). -/
def stagedShared (r : XR) (st : Stage) (x : Claims.Msg) (ms : List (Claims.Msg × Claims.Out)) : RecShared :=
  ⟨some r.w.recInner, some (st.rows x ms), some r.innerRaw.circuit, some r.innerRaw.publicFile, r.innerRaw.partition,
    none, some (st.record x ms), some (st.transcript x ms), r.w.registration, some r.innerRaw.statement,
    some r.innerRaw.osCoins, some r.innerRaw.tables, r.innerRaw.draw⟩

/-- **Part `s` of the compiled verifier is `RecRaw.ofShared`** at what every part shares, its staged Flock statement
and its bindings. -/
theorem partRaw_ofShared (r : XR) (st : Stage) (x : Claims.Msg) (ms : List (Claims.Msg × Claims.Out)) (s : ℕ)
    (p : PartX) :
    partRaw r st x ms s p =
      RecRaw.ofShared (stagedShared r st x ms) { p.base with publicFile := st.pub x ms s } s (some p.bindings) :=
  rfl

/-- **One part's run**: the recursive statement its `verify` held, its session's verdict and reason, its coin key, and
whether `verify` held an identifier. -/
structure PartRun where
  w : RecRaw
  accepted : Bool
  why : String
  coinKey : Option String
  identifier : Bool

/-- What `rec-compose` reads of it: the `VERDICT` `verify --rec-part` writes (`PartView.verdict`). -/
def PartRun.view (p : PartRun) : PartView :=
  ⟨p.w.shared, p.w.recPart, p.w.base.zk, p.accepted, p.why, p.coinKey, p.identifier⟩

/-- **`rec-compose` holds the parts to the compiled verifier's**: where it accepts the verdicts the parts' runs wrote,
either two of the parts' shared files are distinct with one SHA-512, or there is one shared set `S` in the compiled
verifier's form, at the composition's own inner shape and rows, such that every part's recursive statement is
`RecRaw.ofShared S` at its own Flock statement, an index `s < n` and its own bindings (`partRaw_ofShared`); one part per
index, each accepted under `--zk` without an identifier, and no two sessions on one coin key. -/
theorem composed_parts {n : ℕ} {rows inner : String} {ps : List PartRun}
    (h : RecCompose.Accepts n rows inner (ps.map (·.view.verdict))) :
    Collides (ps.map PartRun.view) ∨ ∃ S : RecShared, S.Form ∧ S.recInner.map RecCompose.digest = some inner ∧
      S.recRows.map RecCompose.digest = some rows ∧ ps.length = n ∧
      (∀ j, j < n → (ps.map (·.w.recPart)).count (some j) = 1) ∧ (ps.map (·.coinKey)).Nodup ∧
      ∀ p ∈ ps, (∃ s, s < n ∧ p.w = RecRaw.ofShared S p.w.base s p.w.recBindings) ∧ p.accepted = true ∧
        p.w.base.zk = true ∧ p.identifier = false ∧ p.coinKey.isSome := by
  have h' : RecCompose.Accepts n rows inner ((ps.map PartRun.view).map PartView.verdict) := by
    rw [List.map_map]
    exact h
  rcases composed_views h' with hc | ⟨S, hS, hi, hr, hl, hidx, hk, hall⟩
  · exact .inl hc
  refine .inr ⟨S, hS, hi, hr, by simpa using hl, fun j hj => ?_, ?_, fun p hp => ?_⟩
  · have e := hidx j hj
    rw [List.map_map] at e
    exact e
  · rw [List.map_map] at hk
    exact hk
  · obtain ⟨hsh, ha, hz, hid, hkey, j, hj, hjn⟩ := hall _ (List.mem_map_of_mem hp)
    refine ⟨⟨j, hjn, ?_⟩, ha, hz, hid, hkey⟩
    have e := RecRaw.ofShared_shared (w := p.w) hj
    rw [show p.w.shared = S from hsh] at e
    exact e.symm

end RecDeploy
