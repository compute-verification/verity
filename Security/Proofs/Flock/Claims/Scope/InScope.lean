import Proofs.Flock.Claims.Scope.Check
import Proofs.Flock.Claims.Scope.Setup
import Proofs.Flock.Soundness.Discharge.FailClosed.TmplStmt

/-!
# The readable scope is the verifier's scope check

For every statement the verifier parses (`Parses`) and every relation part `rel`:

* `inScope_iff`: `Claims.Flock.InScope zk (toReadable rel p)` holds exactly when `ProvedScope.check` passes and
  `22 ≤ m`. The check reads no parameter of `m` and `k_log`; the setup gives `m ≤ 35`, `6 ≤ k_log ≤ min(m, 27)`, and under
  `--zk` `25 ≤ m` (`table0`). Without `--zk`, `22 ≤ m` is `ProvedScope.sizesOk`'s, which `verify` runs on the tables
  after the check (`Sizes`: `inScope_iff_scope`), not the setup's.
* `inScope_iff_zk`: under `--zk`, exactly when `ProvedScope.check` passes.
-/

namespace Flock.Readable

theorem tablesOf_of_parses {p : Parsed} (hp : Parses p) : p.toRead.tablesOf = some p.sts := by
  unfold Parses Read.parse at hp
  obtain ⟨sts, hs, he⟩ := Option.map_eq_some_iff.1 hp
  rw [hs, ← he]

/-- What the verifier's parser returns, it parses. -/
theorem parses_of_parse {r : Read} {p : Parsed} (h : r.parse = some p) : Parses p := by
  unfold Read.parse at h
  obtain ⟨sts, hs, rfl⟩ := Option.map_eq_some_iff.1 h
  show Read.parse r = _
  unfold Read.parse
  rw [hs]
  rfl

/-- `InScope` reads the rules of `statementRules` off the form. -/
theorem rules_of_inScope {rel : Parsed → Relation} {p : Parsed}
    (h : Claims.Flock.InScope p.zk (toReadable rel p)) :
    Rules p.tags p.zk p.I.count p.publicInputs.isSome := by
  obtain ⟨-, -, -, -, h1, h2, h3, h4, h5, h6, -⟩ := h
  simp only [toReadable, Read.form, Bool.not_eq_true] at h1 h2 h3 h4 h5 h6
  exact ⟨h1, h2, h3, by simpa only [LeafOk] using h4, h5, h6⟩

theorem sharedRowsOk_iff (t : Tags) (zk reg : Bool) :
    ProvedScope.sharedRowsOk t zk reg = true ↔
      t.hiddenOutputs = true ∧ reg = true ∧ t.merkleLeaf = "hm96-sha512/v1" := by
  simp [ProvedScope.sharedRowsOk, and_assoc]

/-- The wire condition and `scopeOk` (`scopeOkI` with hidden outputs) are the unit's form, or with hidden outputs the
wires' order, and the unit's and the rows' conditions. -/
theorem scope_iff_rows (hidden : Bool) (c : Circuit) :
    ((ProvedScope.wiresFeed c = true → ProvedScope.wiresAcyclic c = true) ∧
        (if hidden then ProvedScope.scopeOkI c else ProvedScope.scopeOk c) = true) ↔
      ((c.unit.inGroups.size = 1 ∧ ¬ (c.outNet != c.unitNet) = true) ∨
          (hidden = true ∧ ProvedScope.wiresAcyclic c = true)) ∧
        c.unit.unitLog ≤ 32 ∧ ProvedScope.portReadsB c.unit = true ∧
        (if hidden then ProvedScope.v1Ports c || ProvedScope.bitRows c else ProvedScope.v1Ports c) = true := by
  have hf : ProvedScope.wiresFeed c = true ↔ ¬ (c.unit.inGroups.size = 1 ∧ ¬ (c.outNet != c.unitNet) = true) := by
    simp only [ProvedScope.wiresFeed, Bool.or_eq_true, bne_iff_ne, ne_eq, not_not]
    tauto
  rw [hf]
  cases hidden
  · simp only [ProvedScope.scopeOk, Bool.false_eq_true, ↓reduceIte, false_and, or_false, Bool.and_eq_true, beq_iff_eq,
      decide_eq_true_eq, bne_iff_ne, ne_eq, not_not]
    constructor
    · rintro ⟨-, ⟨⟨⟨h1, h2⟩, h3⟩, h4⟩, h5⟩
      exact ⟨⟨h1, h2⟩, h3, h4, h5⟩
    · rintro ⟨⟨h1, h2⟩, h3, h4, h5⟩
      exact ⟨fun h => absurd ⟨h1, h2⟩ h, ⟨⟨⟨h1, h2⟩, h3⟩, h4⟩, h5⟩
  · simp only [ProvedScope.scopeOkI, ↓reduceIte, true_and, Bool.and_eq_true, decide_eq_true_eq]
    constructor
    · rintro ⟨hw, ⟨h3, h4⟩, h5⟩
      exact ⟨(em _).elim Or.inl fun h => Or.inr (hw h), h3, h4, h5⟩
    · rintro ⟨hu, h3, h4, h5⟩
      exact ⟨fun h => hu.resolve_left h, ⟨h3, h4⟩, h5⟩

theorem v1Ports_unrolled (c : Circuit) : ProvedScope.v1Ports (HmOut.unrolled c) = ProvedScope.v1Ports c := by
  unfold HmOut.unrolled; split
  · split <;> rfl
  · rfl

theorem bitRows_unrolled (c : Circuit) : ProvedScope.bitRows (HmOut.unrolled c) = ProvedScope.bitRows c := by
  unfold HmOut.unrolled; split
  · split <;> rfl
  · rfl

/-- **The check's form is the statement circuit's** (`HmOut.unrolled`, ASK-112 option A): on a flat circuit the wire
condition and `scopeOk` (`scopeOkI` with hidden outputs), as `scope_iff_rows` reads them; on a templated one `scopeOkT`,
which is the unrolled unit's form, its rows' and the templated statement's own: `--zk`, hidden outputs, no registered
reads of the verifier's own, at most one table, and a unit reading one input group and returning the outputs. -/
theorem formOk_iff {t : Tags} {zk reg : Bool} {count : Nat} {cf : ByteArray} {tables : List (String × ByteArray)}
    {c : Circuit} (hty : t.typed = false) (hc : ProvedScope.circuitOf t cf tables = .ok c) :
    FormOk t zk reg count c ↔
      (((HmOut.unrolled c).unit.inGroups.size = 1 ∧
            ¬ ((HmOut.unrolled c).outNet != (HmOut.unrolled c).unitNet) = true) ∨
          (t.hiddenOutputs = true ∧ ProvedScope.wiresAcyclic (HmOut.unrolled c) = true)) ∧
        (HmOut.unrolled c).unit.unitLog ≤ 32 ∧ ProvedScope.portReadsB (HmOut.unrolled c).unit = true ∧
        (if t.hiddenOutputs then ProvedScope.v1Ports (HmOut.unrolled c) || ProvedScope.bitRows (HmOut.unrolled c)
          else ProvedScope.v1Ports (HmOut.unrolled c)) = true ∧
        (c.tmpl.isSome = true → zk = true ∧ t.hiddenOutputs = true ∧ ¬ reg = true ∧ count ≤ 1 ∧
          (HmOut.unrolled c).unit.inGroups.size = 1 ∧
          ¬ ((HmOut.unrolled c).outNet != (HmOut.unrolled c).unitNet) = true) := by
  cases ht : c.tmpl with
  | none =>
    rw [FlockSoundness.unrolled_flat ht]
    simp only [FormOk, ht, Option.isSome_none, Bool.false_eq_true, false_implies, and_true]
    exact scope_iff_rows _ c
  | some tu =>
    obtain ⟨hh, -, -, hig, hul, hpr, eo, eu, -, -⟩ := FlockSoundness.Discharge.FailClosed.tmpl_facts hty hc ht
    have hpr' : max 7 (log2ceil (tu.t.defs.getD tu.t.root default).cols) ≤ 32 →
        ProvedScope.portReadsB (HmOut.unrolled c).unit = true := fun h =>
      (FlockSoundness.Discharge.FailClosed.portReadsB_eq _).trans (hpr (by rw [hul]; exact h))
    simp only [FormOk, ht, Option.isSome_some, true_implies]
    rw [hig, hul, eo, eu, v1Ports_unrolled, bitRows_unrolled, hh]
    unfold ProvedScope.scopeOkT
    simp only [Bool.and_eq_true, Bool.not_eq_true', beq_iff_eq, decide_eq_true_eq, ↓reduceIte, bne_iff_ne, ne_eq,
      not_not, true_and, Bool.true_and, Bool.not_eq_true]
    constructor
    · rintro ⟨⟨⟨⟨⟨⟨hz, hr⟩, hn⟩, hig1⟩, hout⟩, hlog⟩, hrows⟩
      exact ⟨Or.inl ⟨hig1, hout⟩, hlog, hpr' hlog, hrows, hz, hr, hn, hig1, hout⟩
    · rintro ⟨-, hlog, -, hrows, hz, hr, hn, hig1, hout⟩
      exact ⟨⟨⟨⟨⟨⟨hz, hr⟩, hn⟩, hig1⟩, hout⟩, hlog⟩, hrows⟩

/-- **The readable scope is the verifier's scope check, and `22 ≤ m`**, for every statement the verifier parses. -/
theorem inScope_iff (rel : Parsed → Relation) {p : Parsed} (hp : Parses p) :
    Claims.Flock.InScope p.zk (toReadable rel p) ↔ p.check = .ok () ∧ 22 ≤ (toReadable rel p).m := by
  rw [Read.check, check_ok_iff]
  by_cases hR : Rules p.tags p.zk p.I.count p.publicInputs.isSome
  swap
  · exact ⟨fun h => absurd (rules_of_inScope h) hR, fun h => absurd h.1.1 hR⟩
  have hR' := hR
  obtain ⟨h1, h2, h3, h4, h5, h6⟩ := hR'
  have hpi : p.publicInputs = none := Option.not_isSome_iff_eq_none.1 (by simp [h6])
  obtain ⟨st, h0, hc, ⟨pop, hpop⟩, hm, hk6, hk27, hm35, hzk⟩ := table0 (tablesOf_of_parses hp) h2 hpi
  have hcirc : p.circ = st.c := by simp only [Read.circ, hc]
  have hpub : p.pub = pop := by simp only [Read.pub, hcirc, hpop]
  have em : (toReadable rel p).m = st.m := by simp [toReadable, h0]
  have ek : (toReadable rel p).kLog = st.c.kLog := by simp [toReadable, h0]
  have ef : (toReadable rel p).form = p.form := rfl
  have eR : (∃ c pub, ProvedScope.circuitOf p.tags p.I.circuitFile p.I.tables = .ok c ∧
      ProvedScope.publicOf p.tags c p.I.publicFile = .ok pub ∧
      (pub.tables.isSome = true → ProvedScope.sharedRowsOk p.tags p.zk p.registered = true) ∧
      FormOk p.tags p.zk p.registered p.I.count c) ↔
      (pop.tables.isSome = true → ProvedScope.sharedRowsOk p.tags p.zk p.registered = true) ∧
      FormOk p.tags p.zk p.registered p.I.count st.c := by
    constructor
    · rintro ⟨c, pub, e1, e2, hX, hF⟩
      rw [hc] at e1
      cases e1
      rw [hpop] at e2
      cases e2
      exact ⟨hX, hF⟩
    · rintro ⟨hX, hF⟩
      exact ⟨st.c, pop, hc, hpop, hX, hF⟩
  rw [eR, em]
  simp only [Claims.Flock.InScope, em, ek, ef, Read.form, Read.stmt, hcirc, hpub]
  rw [sharedRowsOk_iff, formOk_iff h2 hc]
  constructor
  · rintro ⟨hlo, -, -, -, -, -, -, -, -, -, hsh, hu, hl, hpr, hrows, htm⟩
    refine ⟨⟨hR, hsh, hu, hl, hpr, ?_, htm⟩, ?_⟩
    · cases hh : p.tags.hiddenOutputs
      · simpa [hh] using hrows
      · simpa [hh] using hrows
    · split at hlo <;> omega
  · rintro ⟨⟨-, hsh, hu, hl, hpr, hrows, htm⟩, hlo⟩
    have hlo' : (if p.zk = true then 25 else 22) ≤ st.m := by
      split
      · exact hzk (by assumption)
      · exact hlo
    refine ⟨hlo', hm35, hk6, by omega, h1, by simp [h2], h3, by simpa only [LeafOk, Bool.not_eq_true] using h4, h5,
      by simp [h6], hsh, hu, hl, hpr, ?_, htm⟩
    cases hh : p.tags.hiddenOutputs
    · simp only [hh, Bool.false_eq_true, ↓reduceIte] at hrows
      exact Or.inl hrows
    · simp only [hh, ↓reduceIte, Bool.or_eq_true] at hrows
      exact hrows.imp id fun h => ⟨rfl, h⟩

/-- **Under `--zk`, the readable scope is the verifier's scope check**, for every statement the verifier parses. -/
theorem inScope_iff_zk (rel : Parsed → Relation) {p : Parsed} (hp : Parses p) (hz : p.zk = true) :
    Claims.Flock.InScope p.zk (toReadable rel p) ↔ p.check = .ok () := by
  rw [inScope_iff rel hp]
  refine ⟨And.left, fun h => ⟨h, ?_⟩⟩
  have hR := ((check_ok_iff _ _ _ _ _ _ _ _).1 h).1
  have hpi : p.publicInputs = none := Option.not_isSome_iff_eq_none.1 (by simp [hR.2.2.2.2.2])
  obtain ⟨st, h0, -, -, -, -, -, -, hzk⟩ := table0 (tablesOf_of_parses hp) hR.2.1 hpi
  have em : (toReadable rel p).m = st.m := by simp [toReadable, h0]
  have := hzk hz
  omega

end Flock.Readable
