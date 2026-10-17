import Proofs.Flock.Claims.Scope.Sizes

/-!
# `Claims.Flock.ScopeMatches` at the verifier's parser and scope check

A deployed record whose parser is the verifier's (`parseOf`: the statement's bytes decoded to what the verifier reads, set
up by `Read.parse`, read by `toReadable`) and whose scope check is the verifier's (`scopeCheckOf`: the parser accepts
and `verify`'s scope refusal, `Parsed.scope`, passes: `ProvedScope.check`, then `ProvedScope.sizesOk` on the tables),
and whose table `j` is the verifier's table `j` alone (`tableOf`):

* `scope_iff`: the readable scope holds exactly when the verifier parses the statement and `Parsed.scope` passes;
* `scopeMatches`: `ScopeMatches` holds, with `--zk` and without.
-/

namespace Flock.Readable

/-- What the verifier parses of a statement's bytes: `decode` reads what the verifier reads (`Read`), and the verifier's
setup accepts it (`Read.parse`). -/
def parseBytes (decode : Claims.Msg → Option Read) (x : Claims.Msg) : Option Parsed := (decode x).bind Read.parse

/-- **The deployed parser**: the verifier's parsed statement, readable. -/
def parseOf (rel : Parsed → Relation) (decode : Claims.Msg → Option Read) (x : Claims.Msg) :
    Option Claims.Flock.Stmt :=
  (parseBytes decode x).map (toReadable rel)

/-- **Table `j`'s own statement at the deployed parser**: table `j` of the verifier's parsed statement, alone
(`Parsed.only`), readable. -/
def tableOf (rel : Parsed → Relation) (decode : Claims.Msg → Option Read) (x : Claims.Msg) (j : ℕ) :
    Option Claims.Flock.Stmt :=
  (parseBytes decode x).map fun p => toReadable rel (p.only j)

/-- **The deployed scope check**: the verifier parses the statement and its scope refusal (`Parsed.scope`) passes. -/
def scopeCheckOf (decode : Claims.Msg → Option Read) (x : Claims.Msg) : Bool :=
  match parseBytes decode x with
  | some p => p.scope.isOk
  | none => false

theorem toRead_of_parse {r : Read} {p : Parsed} (h : r.parse = some p) : p.toRead = r := by
  unfold Read.parse at h
  obtain ⟨sts, -, rfl⟩ := Option.map_eq_some_iff.1 h
  rfl

theorem isOk_iff (e : Except String Unit) : e.isOk = true ↔ e = .ok () := by
  cases e <;> simp [Except.isOk, Except.toBool]

variable {Coin Col T : Type}

/-- **The readable scope at the verifier's parser**: the verifier parses the statement and `Parsed.scope` passes. -/
theorem scope_iff (D : Claims.Flock.Deploy Coin Col T) (rel : Parsed → Relation)
    (decode : Claims.Msg → Option Read) (hparse : D.parse = parseOf rel decode)
    (hzk : ∀ x r, decode x = some r → r.zk = D.zk) (x : Claims.Msg) :
    Claims.Flock.Scope D x ↔ ∃ p, parseBytes decode x = some p ∧ p.scope = .ok () := by
  unfold Claims.Flock.Scope
  rw [hparse]
  constructor
  · rintro ⟨s, hs, hin⟩
    obtain ⟨p, hp, rfl⟩ := Option.map_eq_some_iff.1 hs
    obtain ⟨r, hr, hrp⟩ := Option.bind_eq_some_iff.1 hp
    have hz : p.zk = D.zk := by rw [← hzk x r hr, ← toRead_of_parse hrp]
    rw [← hz] at hin
    exact ⟨p, hp, (inScope_iff_scope rel (parses_of_parse hrp)).1 hin⟩
  · rintro ⟨p, hp, hc⟩
    obtain ⟨r, hr, hrp⟩ := Option.bind_eq_some_iff.1 hp
    have hz : p.zk = D.zk := by rw [← hzk x r hr, ← toRead_of_parse hrp]
    refine ⟨toReadable rel p, by simp [parseOf, hp], ?_⟩
    rw [← hz]
    exact (inScope_iff_scope rel (parses_of_parse hrp)).2 hc

/-- **`ScopeMatches` at the verifier's parser and scope check**, with `--zk` and without. -/
theorem scopeMatches (D : Claims.Flock.Deploy Coin Col T) (rel : Parsed → Relation)
    (decode : Claims.Msg → Option Read) (hparse : D.parse = parseOf rel decode)
    (hcheck : D.scopeCheck = scopeCheckOf decode) (hzk : ∀ x r, decode x = some r → r.zk = D.zk) :
    Claims.Flock.ScopeMatches D := by
  unfold Claims.Flock.ScopeMatches Claims.ScopeMatches
  intro x
  rw [scope_iff D rel decode hparse hzk, hcheck, scopeCheckOf]
  constructor
  · rintro ⟨p, hp, hc⟩
    rw [hp]
    exact (isOk_iff _).2 hc
  · intro hx
    split at hx
    · rename_i p hp
      exact ⟨p, hp, (isOk_iff _).1 hx⟩
    · cases hx

end Flock.Readable
