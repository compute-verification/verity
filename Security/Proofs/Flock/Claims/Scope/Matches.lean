import Proofs.Flock.Claims.Scope.InScope

/-!
# `Claims.Flock.ScopeMatches` at the verifier's parser and scope check

A deployed record whose parser is the verifier's (`parseOf`: the statement's bytes decoded to what the verifier reads, set
up by `Read.parse`, read by `toReadable`) and whose scope check is the verifier's (`scopeCheckOf`: the parser accepts
and `ProvedScope.check` passes):

* `scope_iff`: the readable scope holds exactly when the verifier parses the statement, `ProvedScope.check` passes and
  `22 ≤ m`;
* `scopeMatches_iff`: `ScopeMatches` holds exactly when no statement the verifier parses and checks has `m < 22`;
* `scopeMatches_zk`: under `--zk` it holds (`Zk.shapeOk` refuses `m ≤ 24` in the setup).
-/

namespace Flock.Readable

/-- What the verifier parses of a statement's bytes: `decode` reads what the verifier reads (`Read`), and the verifier's
setup accepts it (`Read.parse`). -/
def parseBytes (decode : Claims.Msg → Option Read) (x : Claims.Msg) : Option Parsed := (decode x).bind Read.parse

/-- **The deployed parser**: the verifier's parsed statement, readable. -/
def parseOf (rel : Parsed → Relation) (decode : Claims.Msg → Option Read) (x : Claims.Msg) :
    Option Claims.Flock.Stmt :=
  (parseBytes decode x).map (toReadable rel)

/-- **The deployed scope check**: the verifier parses the statement and `ProvedScope.check` passes. -/
def scopeCheckOf (decode : Claims.Msg → Option Read) (x : Claims.Msg) : Bool :=
  match parseBytes decode x with
  | some p => p.check.isOk
  | none => false

theorem toRead_of_parse {r : Read} {p : Parsed} (h : r.parse = some p) : p.toRead = r := by
  unfold Read.parse at h
  obtain ⟨sts, -, rfl⟩ := Option.map_eq_some_iff.1 h
  rfl

theorem isOk_iff (e : Except String Unit) : e.isOk = true ↔ e = .ok () := by
  cases e <;> simp [Except.isOk, Except.toBool]

variable {Coin Col T : Type}

/-- **The readable scope at the verifier's parser**: the verifier parses the statement, `ProvedScope.check` passes and
`22 ≤ m`. -/
theorem scope_iff (D : Claims.Flock.Deploy Coin Col T) (rel : Parsed → Relation)
    (decode : Claims.Msg → Option Read) (hparse : D.parse = parseOf rel decode)
    (hzk : ∀ x r, decode x = some r → r.zk = D.zk) (x : Claims.Msg) :
    Claims.Flock.Scope D x ↔
      ∃ p, parseBytes decode x = some p ∧ p.check = .ok () ∧ 22 ≤ (toReadable rel p).m := by
  unfold Claims.Flock.Scope
  rw [hparse]
  constructor
  · rintro ⟨s, hs, hin⟩
    obtain ⟨p, hp, rfl⟩ := Option.map_eq_some_iff.1 hs
    obtain ⟨r, hr, hrp⟩ := Option.bind_eq_some_iff.1 hp
    have hz : p.zk = D.zk := by rw [← hzk x r hr, ← toRead_of_parse hrp]
    rw [← hz] at hin
    exact ⟨p, hp, (inScope_iff rel (parses_of_parse hrp)).1 hin⟩
  · rintro ⟨p, hp, hc, hm⟩
    obtain ⟨r, hr, hrp⟩ := Option.bind_eq_some_iff.1 hp
    have hz : p.zk = D.zk := by rw [← hzk x r hr, ← toRead_of_parse hrp]
    refine ⟨toReadable rel p, by simp [parseOf, hp], ?_⟩
    rw [← hz]
    exact (inScope_iff rel (parses_of_parse hrp)).2 ⟨hc, hm⟩

/-- **`ScopeMatches` at the verifier's parser and scope check**, exactly when every statement the verifier parses and
`ProvedScope.check` passes has `22 ≤ m`. -/
theorem scopeMatches_iff (D : Claims.Flock.Deploy Coin Col T) (rel : Parsed → Relation)
    (decode : Claims.Msg → Option Read) (hparse : D.parse = parseOf rel decode)
    (hcheck : D.scopeCheck = scopeCheckOf decode) (hzk : ∀ x r, decode x = some r → r.zk = D.zk) :
    Claims.Flock.ScopeMatches D ↔
      ∀ x p, parseBytes decode x = some p → p.check = .ok () → 22 ≤ (toReadable rel p).m := by
  unfold Claims.Flock.ScopeMatches Claims.ScopeMatches
  simp only [scope_iff D rel decode hparse hzk, hcheck, scopeCheckOf]
  constructor
  · intro h x p hp hc
    have := (h x).2 (by rw [hp]; exact (isOk_iff _).2 hc)
    obtain ⟨p', hp', -, hm⟩ := this
    rw [hp] at hp'
    cases hp'
    exact hm
  · intro h x
    constructor
    · rintro ⟨p, hp, hc, -⟩
      rw [hp]
      exact (isOk_iff _).2 hc
    · intro hx
      split at hx
      · rename_i p hp
        have hc := (isOk_iff _).1 hx
        exact ⟨p, hp, hc, h x p hp hc⟩
      · cases hx

/-- **`ScopeMatches` under `--zk`** at the verifier's parser and scope check. -/
theorem scopeMatches_zk (D : Claims.Flock.Deploy Coin Col T) (rel : Parsed → Relation)
    (decode : Claims.Msg → Option Read) (hparse : D.parse = parseOf rel decode)
    (hcheck : D.scopeCheck = scopeCheckOf decode) (hzk : ∀ x r, decode x = some r → r.zk = D.zk)
    (hz : D.zk = true) : Claims.Flock.ScopeMatches D := by
  refine (scopeMatches_iff D rel decode hparse hcheck hzk).2 fun x p hp hc => ?_
  obtain ⟨r, hr, hrp⟩ := Option.bind_eq_some_iff.1 hp
  have hpz : p.zk = true := by rw [← hz, ← hzk x r hr, ← toRead_of_parse hrp]
  exact ((inScope_iff rel (parses_of_parse hrp)).1 ((inScope_iff_zk rel (parses_of_parse hrp) hpz).2 hc)).2

end Flock.Readable
