import FlockSoundness.Discharge.Layout.Scope
import FlockSoundness.Discharge.ZkExec.Event
import FlockSoundness.Discharge.Hidden.Facts
import FlockSoundness.ExecSetup
import Flock.ProvedScope

/-!
# What the verifier accepts: `Accepts` and its statement check

`flock-verify verify` refuses every session of a statement whose form the soundness proof does not cover:
`Main.verifyCmd` runs `Flock.ProvedScope.check` on the statement and turns every session's verdict into the check's
refusal when it fails. So what the verifier accepts is `Verifies` (under `--zk`, `VerifiesZK`): the setup and verdict
`Accepts` models, and the check. Every condition on the statement's form that the headlines took as a hypothesis follows
from the check: an untyped statement (`Checks.typed`), retained rounds and `hm96-sha512/v1` leaves, at most one table
without `--zk`, the circuit and public file as the headlines read them (`circ`, `pubOf`), no shared-row public file
(`Checks.tables`) and `Layout.scopeOk` (`Checks.scope`, by `scopeOk_eq`: the executable's `ProvedScope.scopeOk` is
`Layout.scopeOk`, its port bits a mask). The unconditional headlines (`FailClosed.Composed`, `FailClosed.ZkHidden`,
`FailClosed.Exec`) are over `Verifies`, with no hypothesis on the statement's form.

`Checks` is `verifyCmd`'s call `ProvedScope.check tags zk circuitB pubB tables count publicInputs.isSome` argument for
argument. `tags` is `Inputs.tags` without `--zk` and `zkTags` under it (`--statement`'s, after `withCoins`, then `Zk.tags`
at `maskLogOf circuitB` and `count`, on the coin tree of `tableNames count`). The circuit, public file and tables are the
bytes `verifyCmd` sets the session up from, and `count` is `--session-tables`. The flag is `false`: with
`--public-inputs` the check refuses every session, and without it `buildSession` passes `setupTables` no public input
words, the setup `Accepts` models. A session's verdict is `verifyCmd`'s `setupOf` at the record's `recordDraw` (the check
requires `hm96Rows`): `setupTables`, under `--zk` `Zk.shapeOk` on every table, then `ProvedScope.drawLaw` on a drawn
session's draw, the check, `verify` or `Zk.verify`, and last `ProvedScope.drawLaw` on a session without a draw. So a session
`verifyCmd` accepts satisfies `Verifies` (under `--zk`, `VerifiesZK`). `verifyCmd` refuses more, which only shrinks the
event: a draw by a law no soundness claim draws by (`ProvedScope.drawLaw`: a bernoulli or work draw, or a subset draw
without `--zk`), a session without a draw (every claim's record hypothesis reads the record's draw), registered reads (`checkRegistered`:
a header naming registered values without `--registered`; with it, `Registered.check` needs a shared-row public file,
which the check refuses), a coin-tree key an earlier session of the run used, and a session file it cannot read. A later session of a run whose draw has an earlier
one's canonical form is verified against that earlier setup; `canon` takes integers only, so the two draws hold the same
content.
-/

namespace FlockSoundness.Discharge.FailClosed

open Flock Exec ZkExec

/-! ## The executable's `scopeOk` is `Layout.scopeOk` -/

private theorem foldl_mask (x : ℕ) : ∀ (l : List ℕ) (m : Array Bool), x < m.size →
    (l.foldl (fun m y => m.setIfInBounds y true) m).getD x false = (m.getD x false || decide (x ∈ l)) := by
  intro l
  induction l with
  | nil => intro m _; simp
  | cons y l ih =>
    intro m hm
    rw [List.foldl_cons, ih _ (by simpa using hm)]
    simp only [Array.getD_eq_getD_getElem?, Array.getElem?_setIfInBounds, List.mem_cons]
    by_cases hxy : y = x
    · subst hxy; simp [hm]
    · simp [hxy, Ne.symm hxy]

/-- **The port mask is `portBits`'s membership** below its size. -/
theorem portMask_getD (gs : Array Group) {n x : ℕ} (hx : x < n) :
    (ProvedScope.portMask gs n).getD x false = decide (x ∈ portBits gs) := by
  unfold ProvedScope.portMask
  rw [foldl_mask x _ _ (by simpa using hx)]
  have h0 : (Array.replicate n false).getD x false = false := by simp [Array.getD_eq_getD_getElem?, hx]
  rw [h0, Bool.false_or]
  exact decide_eq_decide.2 Iff.rfl

theorem portReadsB_eq (net : Net) : ProvedScope.portReadsB net = Layout.portReadsB net := by
  simp only [ProvedScope.portReadsB, Layout.portReadsB]
  congr 1; funext i; congr 1; funext x
  apply decide_eq_decide.2
  refine imp_congr_right fun hx => ?_
  rw [portMask_getD _ hx, decide_eq_true_iff]

/-- **The executable's `scopeOk` is `Layout.scopeOk`.** -/
theorem scopeOk_eq (c : Circuit) : ProvedScope.scopeOk c = Layout.scopeOk c := by
  simp only [ProvedScope.scopeOk, Layout.scopeOk, portReadsB_eq, ProvedScope.v1Ports, Layout.v1Ports]

/-! ## The check, and what it gives -/

/-- **The verifier's statement check passes** at tags `t` (the session's: under `--zk`, `zkTags`'s), with no public input
words of the verifier's own (the setup `Accepts` models takes none). -/
def Checks (t : Tags) (zk : Bool) (I : Inputs) : Prop :=
  ProvedScope.check t zk I.circuitFile I.publicFile I.tables I.count false = .ok ()

/-- The circuit as the check parses it (`default` when it does not parse, and then the check fails). -/
def circ (t : Tags) (I : Inputs) : Circuit :=
  match ProvedScope.circuitOf t I.circuitFile I.tables with
  | .ok c => c
  | .error _ => default

/-- The public file as the check loads it (`default` when it does not load, and then the check fails). -/
def pubOf (t : Tags) (I : Inputs) : Public :=
  match ProvedScope.publicOf t (circ t I) I.publicFile with
  | .ok p => p
  | .error _ => default

theorem check_ok {t : Tags} {zk : Bool} {cf pf : ByteArray} {tables : List (String × ByteArray)} {count : ℕ}
    (h : ProvedScope.check t zk cf pf tables count false = .ok ()) :
    (t.hm96Rows = true ∧ t.typed = false ∧ t.retainRounds = true ∧ t.merkleLeaf = "hm96-sha512/v1" ∧
      (zk = true ∨ count ≤ 1)) ∧
    ∃ c pub, ProvedScope.circuitOf t cf tables = .ok c ∧ ProvedScope.publicOf t c pf = .ok pub ∧ pub.tables = none ∧
      ProvedScope.scopeOk c = true := by
  unfold ProvedScope.check at h
  split at h
  · cases h
  rename_i hf
  have hall : ∀ r ∈ ProvedScope.statementRules t zk count false, r.1 = true := by
    unfold ProvedScope.firstFailure at hf
    rw [Option.map_eq_none_iff, List.find?_eq_none] at hf
    intro r hr
    simpa using hf r hr
  simp only [ProvedScope.statementRules, List.mem_cons, List.mem_nil_iff, or_false, forall_eq_or_imp, forall_eq,
    Bool.not_eq_true', Bool.or_eq_true, decide_eq_true_eq, beq_iff_eq, Bool.not_false] at hall
  obtain ⟨h1, h2, h3, h4, h5, -⟩ := hall
  refine ⟨⟨h1, h2, h3, h4, h5⟩, ?_⟩
  split at h
  · cases h
  rename_i c hc
  split at h
  · cases h
  rename_i pub hp
  split at h
  · cases h
  rename_i ht
  split at h
  · rename_i hs
    exact ⟨c, pub, hc, hp, by simpa using ht, hs⟩
  · cases h

namespace Checks

variable {t : Tags} {zk : Bool} {I : Inputs} (h : Checks t zk I)
include h

theorem typed : t.typed = false := (check_ok h).1.2.1

theorem hm96Rows : t.hm96Rows = true := (check_ok h).1.1

theorem retainRounds : t.retainRounds = true := (check_ok h).1.2.2.1

theorem merkleLeaf : t.merkleLeaf = "hm96-sha512/v1" := (check_ok h).1.2.2.2.1

theorem circuit_ok : ProvedScope.circuitOf t I.circuitFile I.tables = .ok (circ t I) := by
  obtain ⟨c, -, hc, -⟩ := (check_ok h).2
  unfold circ
  rw [hc]

theorem public_ok : ProvedScope.publicOf t (circ t I) I.publicFile = .ok (pubOf t I) := by
  obtain ⟨c, pub, hc, hp, -⟩ := (check_ok h).2
  have : circ t I = c := by unfold circ; rw [hc]
  unfold pubOf
  rw [this, hp]

theorem tables : (pubOf t I).tables = none := by
  obtain ⟨c, pub, hc, hp, ht, -⟩ := (check_ok h).2
  have : circ t I = c := by unfold circ; rw [hc]
  have hpub : pubOf t I = pub := by unfold pubOf; rw [this, hp]
  rw [hpub, ht]

theorem scope : Layout.scopeOk (circ t I) = true := by
  obtain ⟨c, -, hc, -, -, hs⟩ := (check_ok h).2
  have : circ t I = c := by unfold circ; rw [hc]
  rw [this, ← scopeOk_eq, hs]

theorem parse (hh : t.hiddenOutputs = false) : HmRow.parse t I.circuitFile I.tables = .ok (circ t I) := by
  have := h.circuit_ok
  simpa [ProvedScope.circuitOf, hh] using this

theorem parseH (hh : t.hiddenOutputs = true) : HmOut.parse t I.circuitFile I.tables = .ok (circ t I) := by
  have := h.circuit_ok
  simpa [ProvedScope.circuitOf, hh] using this

theorem load (hh : t.hiddenOutputs = false) : HmRow.loadPublic t (circ t I) I.publicFile = .ok (pubOf t I) := by
  have := h.public_ok
  simpa [ProvedScope.publicOf, hh] using this

theorem loadH (hh : t.hiddenOutputs = true) : HmOut.loadPublic t (circ t I) I.publicFile = .ok (pubOf t I) := by
  have := h.public_ok
  simpa [ProvedScope.publicOf, hh] using this

/-- The circuit parses as `HmRow.parse` reads it, with or without hidden outputs (`HmOut.parse` of an untyped statement is
`HmRow.parse`'s, then its rows' check). -/
theorem parseRow : HmRow.parse t I.circuitFile I.tables = .ok (circ t I) := by
  cases hh : t.hiddenOutputs
  · exact h.parse hh
  · exact (Hidden.parse_untyped h.typed (h.parseH hh)).1

theorem built : Built (circ t I).unit := by
  have P := parse_facts h.parseRow
  obtain ⟨ls, hls⟩ := P.check
  exact unit_built P (check_facts P.typed hls)

theorem outNet : (circ t I).outNet = (circ t I).unitNet := (Layout.scopeOk_spec h.built h.scope).2.1

end Checks

/-! ## What the verifier accepts -/

variable {n : ℕ} {Reg : Type}

/-- **`flock-verify verify`'s statement check** (no `--zk`). -/
def Scope (I : Inputs) : Prop := Checks I.tags false I

/-- **What `flock-verify verify` accepts** (no `--zk`): the setup and verdict `Accepts` models, and the statement check. -/
def Verifies (I : Inputs) (t : Exec.Transcript) : Prop := Accepts I t ∧ Scope I

/-- `Verifies` at the live session's transcript. -/
def LiveVerifies (I : Inputs) (o : Reg × Finset (Fin n) × Exec.Transcript) : Prop := Verifies I o.2.2

/-- **`flock-verify verify --zk`'s statement check**, at the session's tags (`zkTags`). -/
def ScopeZ (I : Inputs) : Prop := Checks (zkTags I) true I

/-- **What `flock-verify verify --zk` accepts**: the setup and verdict `AcceptsZK` models, and the statement check. -/
def VerifiesZK (I : Inputs) (t : Exec.Transcript) : Prop := AcceptsZK I t ∧ ScopeZ I

/-- `VerifiesZK` at the live session's transcript. -/
def LiveVerifiesZK (I : Inputs) (o : Reg × Finset (Fin n) × Exec.Transcript) : Prop := VerifiesZK I o.2.2

/-- The session's circuit (no `--zk`). -/
abbrev circS (I : Inputs) : Circuit := circ I.tags I

/-- The session's public file (no `--zk`). -/
abbrev pubS (I : Inputs) : Public := pubOf I.tags I

/-- The session's circuit under `--zk`. -/
abbrev circZ (I : Inputs) : Circuit := circ (zkTags I) I

/-- The session's public file under `--zk`. -/
abbrev pubZ (I : Inputs) : Public := pubOf (zkTags I) I

theorem ScopeZ.typed {I : Inputs} (h : ScopeZ I) : I.tags.typed = false := by
  rw [← zkTags_typed]; exact Checks.typed h

theorem ScopeZ.retainRounds {I : Inputs} (h : ScopeZ I) : I.tags.retainRounds = true := by
  rw [← zkTags_retainRounds]; exact Checks.retainRounds h

theorem ScopeZ.merkleLeaf {I : Inputs} (h : ScopeZ I) : I.tags.merkleLeaf = "hm96-sha512/v1" := by
  rw [← zkTags_merkleLeaf]; exact Checks.merkleLeaf h

theorem ScopeZ.parse {I : Inputs} (h : ScopeZ I) (hh : I.tags.hiddenOutputs = false) :
    HmRow.parse (zkTags I) I.circuitFile I.tables = .ok (circZ I) :=
  Checks.parse h (by rw [zkTags_hiddenOutputs]; exact hh)

theorem ScopeZ.parseH {I : Inputs} (h : ScopeZ I) (hh : I.tags.hiddenOutputs = true) :
    HmOut.parse (zkTags I) I.circuitFile I.tables = .ok (circZ I) :=
  Checks.parseH h (by rw [zkTags_hiddenOutputs]; exact hh)

theorem ScopeZ.load {I : Inputs} (h : ScopeZ I) (hh : I.tags.hiddenOutputs = false) :
    HmRow.loadPublic (zkTags I) (circZ I) I.publicFile = .ok (pubZ I) :=
  Checks.load h (by rw [zkTags_hiddenOutputs]; exact hh)

theorem ScopeZ.loadH {I : Inputs} (h : ScopeZ I) (hh : I.tags.hiddenOutputs = true) :
    HmOut.loadPublic (zkTags I) (circZ I) I.publicFile = .ok (pubZ I) :=
  Checks.loadH h (by rw [zkTags_hiddenOutputs]; exact hh)

theorem ScopeZ.built {I : Inputs} (h : ScopeZ I) : Built (circZ I).unit := Checks.built h

theorem ScopeZ.tables {I : Inputs} (h : ScopeZ I) : (pubZ I).tables = none := Checks.tables h

theorem ScopeZ.scope {I : Inputs} (h : ScopeZ I) : Layout.scopeOk (circZ I) = true := Checks.scope h

theorem Scope.typed {I : Inputs} (h : Scope I) : I.tags.typed = false := Checks.typed h

theorem Scope.retainRounds {I : Inputs} (h : Scope I) : I.tags.retainRounds = true := Checks.retainRounds h

theorem Scope.merkleLeaf {I : Inputs} (h : Scope I) : I.tags.merkleLeaf = "hm96-sha512/v1" := Checks.merkleLeaf h

theorem Scope.parse {I : Inputs} (h : Scope I) (hh : I.tags.hiddenOutputs = false) :
    HmRow.parse I.tags I.circuitFile I.tables = .ok (circS I) := Checks.parse h hh

theorem Scope.parseH {I : Inputs} (h : Scope I) (hh : I.tags.hiddenOutputs = true) :
    HmOut.parse I.tags I.circuitFile I.tables = .ok (circS I) := Checks.parseH h hh

theorem Scope.load {I : Inputs} (h : Scope I) (hh : I.tags.hiddenOutputs = false) :
    HmRow.loadPublic I.tags (circS I) I.publicFile = .ok (pubS I) := Checks.load h hh

theorem Scope.loadH {I : Inputs} (h : Scope I) (hh : I.tags.hiddenOutputs = true) :
    HmOut.loadPublic I.tags (circS I) I.publicFile = .ok (pubS I) := Checks.loadH h hh

theorem Scope.built {I : Inputs} (h : Scope I) : Built (circS I).unit := Checks.built h

theorem Scope.tables {I : Inputs} (h : Scope I) : (pubS I).tables = none := Checks.tables h

theorem Scope.scope {I : Inputs} (h : Scope I) : Layout.scopeOk (circS I) = true := Checks.scope h

theorem Scope.outNet {I : Inputs} (h : Scope I) : (circS I).outNet = (circS I).unitNet := Checks.outNet h

theorem Scope.count {I : Inputs} (h : Scope I) : I.count ≤ 1 := by
  rcases (check_ok h).1.2.2.2.2 with h' | h'
  · cases h'
  · exact h'

/-! ## A bound under the statement check -/

/-- **A bound at every statement the check passes, lifted to the event that implies the check.** When the event `V`
implies `A` and the statement fact `Q`, and `A ∧ E q` has probability at most `B q` for every proof `q` of `Q`, then
`V ∧ ∀ q, E q` has probability at most `⨆ q, B q`: at a statement without `Q`, `V` is empty and the bound is `0`. -/
theorem prob_scope {α : Type} {V A : α → Prop} {Q : Prop} (hA : ∀ o, V o → A o) (hQ : ∀ o, V o → Q)
    (E : Q → α → Prop) (B : Q → ENNReal) (g : FlockSoundness.Game α) (s : Game.Strategy g)
    (h : ∀ q, Game.prob (fun o => A o ∧ E q o) g s ≤ B q) :
    Game.prob (fun o => V o ∧ ∀ q, E q o) g s ≤ ⨆ q, B q := by
  by_cases hq : Q
  · calc Game.prob (fun o => V o ∧ ∀ q, E q o) g s ≤ Game.prob (fun o => A o ∧ E hq o) g s :=
          Game.prob_mono (fun o ho => ⟨hA o ho.1, ho.2 hq⟩) g s
      _ ≤ B hq := h hq
      _ ≤ ⨆ q, B q := le_iSup B hq
  · calc Game.prob (fun o => V o ∧ ∀ q, E q o) g s ≤ Game.prob (fun _ => False) g s :=
          Game.prob_mono (fun o ho => hq (hQ o ho.1)) g s
      _ = 0 := Game.prob_false g s
      _ ≤ ⨆ q, B q := zero_le

end FlockSoundness.Discharge.FailClosed
