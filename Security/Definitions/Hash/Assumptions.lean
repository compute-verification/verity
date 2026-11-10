module

public import Mathlib
public import Definitions.Hash.Basic

/-!
# The hash area's named assumptions

`LmsUnforgeable` (`euf-cma/lms-sha512-n64`): LMS over SHA-512, the hash-based signature every auditor-trusted device
signs with, is existentially unforgeable under adaptive chosen-message attack, the standard EUF-CMA game. A forger
(`LmsForger`) runs on uniform coins against one of the device's public keys. On each coin it has the device sign the
messages it lists there (`LmsForger.signed`), its signing queries, which may depend on the coin and so on the
signatures it saw before each query, and it outputs a message and the bytes of a signature. It forges when the message
is none it had signed on that coin and the signature verifies under the key (`LmsForger.Forges`): an existential
forgery, not the strong notion's new signature on a message it had signed. The assumption bounds the probability, over
the forger's coins, by the published `ε_forge(t, Q)` at its SHA-512 calls `t` and the `Q` messages it may have signed
on a coin.

It is a predicate on one forger at one key, since some forger outputs a valid signature after no calls (one that holds
it), so it is asked of that forger only, the one a reduction builds at the device's key, as `Claims.ExpectedCR` is asked
of one finder. The verifier is an argument: a statement reads it at the compiled one.
-/

@[expose] public section

namespace Hash.Assumptions

/-- **An LMS forger** against one key of an auditor-trusted device: on uniform coins `S` it outputs a message and the
bytes of a signature, or nothing, having had the device sign the messages `signed` lists on that coin; `cost` is its
SHA-512 calls on each coin. -/
structure LmsForger where
  S : Type
  [fin : Fintype S]
  [ne : Nonempty S]
  /-- the device's public key the forgery is to verify under -/
  key : Msg
  /-- the messages the device signed under `key` on each coin, whose signatures the forger saw: its signing queries,
  each of which may depend on the signatures it saw before it -/
  signed : S → List Msg
  out : S → Option (Msg × Msg)
  cost : S → ℕ

attribute [instance] LmsForger.fin LmsForger.ne

/-- **The forger's output on the coin `s` is a forgery** under `verify` (public key, message, signature): a message the
device didn't sign on that coin, with a signature that verifies under its key. -/
def LmsForger.Forges (verify : Msg → Msg → Msg → Bool) (F : LmsForger) (s : F.S) : Prop :=
  ∃ m sig, F.out s = some (m, sig) ∧ m ∉ F.signed s ∧ verify F.key m sig = true

open Classical in
/-- `Pr[F forges under verify]`, over its uniform coins. -/
noncomputable def LmsForger.Pr (verify : Msg → Msg → Msg → Bool) (F : LmsForger) : ℝ :=
  ((Finset.univ.filter (F.Forges verify)).card : ℝ) / Fintype.card F.S

/-- **`LmsUnforgeable` (`euf-cma/lms-sha512-n64`): LMS over SHA-512 is EUF-CMA**, at one forger: if `F` makes at most `t`
SHA-512 calls on every coin and has at most `Q` messages signed on every coin, it forges under `verify` with probability
at most `ε_forge(t, Q)`. -/
def LmsUnforgeable (verify : Msg → Msg → Msg → Bool) (εForge : ℕ → ℕ → ℝ) (t Q : ℕ) (F : LmsForger) : Prop :=
  (∀ s, F.cost s ≤ t) → (∀ s, (F.signed s).length ≤ Q) → F.Pr verify ≤ εForge t Q

end Hash.Assumptions
