import Definitions.Accounting.RateAuditRemote
import Definitions.Core.Period

/-!
# The monitoring period's causal rounds: good sets that precede their samples (trusted)

A session of the rate audit is a sequence of rounds (windows, or a window's chained passes), each closed by a uniform
sample of items (words, or a window's receipts) that the verifier opens. A round's good set is the items whose
opening it would pass: for the rate audit, its correct receipts. The prover fixes it before the round's sample is
drawn, as any function of everything it has seen: the earlier rounds' coins, their samples among them, and the round's
own coins other than its sample. `Causal` says exactly that. `Rate.Chain` is such a family (the correct receipts of
pass `j` read the store, the earlier passes' coins, and pass `j`'s stream and oracle family), and so is a session of
windows whose devices see only the earlier windows' coins.

**The deployed draw.** A chained window draws no sample per pass: after its last root, one draw from the openings coin
picks `d` of the window's `P·N` receipts, each a pass and a word (`WindowCoins`). A `Session` is `n` such windows on
one store, window `w`'s prover any function of the earlier windows' coins, their draws among them.

**The prover is one resource.** Nothing here names which device, machine or party holds, reads or rebuilds a word: a
window's passes and a session's windows may be one prover or many, sharing whatever the earlier coins allow. The
guarantees over these games conserve the words read, rebuilt or held, whoever reads, rebuilds or holds them.
-/

namespace Accounting.Rate

/-- Rounds `0, …, n − 1`, round `j` with its own coins in `X j` and a sample of `s j` of `M` items, and round `i`'s
good set `G i`, a function of all the rounds' coins: causal when `G i` reads only the rounds before `i` and round
`i`'s own coins other than its sample. -/
def Causal {M n : ℕ} {X : Fin n → Type} {s : Fin n → ℕ}
    (G : (i : Fin n) → ((j : Fin n) → X j × (Fin (s j) → Fin M)) → Finset (Fin M)) : Prop :=
  ∀ i ω ω', (∀ j, j < i → ω j = ω' j) → (ω i).1 = (ω' i).1 → G i ω = G i ω'

/-- A chained window's coins: each pass's coins, then the window's pooled draw of `d` openings, each a pass and a word,
uniform over the window's `P·N` receipts. -/
abbrev WindowCoins (P N b h ρ r s d : ℕ) : Type := (Fin P → PassCoins N b h ρ r s) × (Fin d → Fin P × Fin N)

/-- The window coins are finite, as a product: an instance of its own, as for `PassCoins`. -/
instance instFintypeWindowCoins (P N b h ρ r s d : ℕ) : Fintype (WindowCoins P N b h ρ r s d) :=
  inferInstanceAs (Fintype (_ × _))

namespace Chain

variable {P N b h ρ S L q r s : ℕ}

/-- A chained window's correct receipts, as the (pass, word) pairs whose opening its pooled draw passes. -/
def correctAll (A : Chain P N b h ρ S L q r s) (E : Store N b) (ω : Fin P → PassCoins N b h ρ r s) :
    Finset (Fin P × Fin N) :=
  Finset.univ.filter (fun p => p.2 ∈ (A.dev p.1 ω).correct E (ω p.1).1 (A.oracle p.1 E ω))

/-- A chained window's reads of every tier and its rebuilt words, in bits, summed over its passes. -/
def spent {τ : ℕ} (A : Chain P N b h ρ S L q r s) (tier : ℕ → Fin τ) (E : Store N b)
    (ω : Fin P → PassCoins N b h ρ r s) : ℕ :=
  ∑ j, (∑ t, (A.dev j ω).tierReads tier t E (ω j).1 (A.oracle j E ω) +
    (A.dev j ω).rebuilt E (ω j).1 (A.oracle j E ω) * b)

end Chain

/-- A session of `n` chained windows on one store, each closed by its pooled draw of `d` openings: a play over its
windows (`Verity.Play`), window `w`'s chained prover any function of the earlier windows' coins, their draws among
them. -/
abbrev Session (n P N b h ρ S L q r s d : ℕ) : Type :=
  Verity.Play n (WindowCoins P N b h ρ r s d) (fun _ => Chain P N b h ρ S L q r s)

namespace Session

variable {n P N b h ρ S L q r s d : ℕ}

/-- Window `w`'s chained prover on the earlier windows' coins `η`: the play's move at `w`. -/
def window (A : Session n P N b h ρ S L q r s d) (w : Fin n)
    (η : {v : Fin n // v < w} → WindowCoins P N b h ρ r s d) : Chain P N b h ρ S L q r s :=
  A w η

/-- Window `w`'s chained prover at the session's coins `ω`: it reads the earlier windows' coins. -/
def chain (A : Session n P N b h ρ S L q r s d) (w : Fin n) (ω : Fin n → WindowCoins P N b h ρ r s d) :
    Chain P N b h ρ S L q r s :=
  A.window w (fun v => ω v.1)

end Session

end Accounting.Rate
