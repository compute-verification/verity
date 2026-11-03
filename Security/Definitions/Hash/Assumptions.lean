import Mathlib
import Definitions.Hash.Basic

/-!
# The hash area's named assumptions

`LmsUnforgeable` (`euf-cma/lms-sha512`): LMS over SHA-512, the hash-based signature every auditor-trusted device signs
with, is existentially unforgeable under chosen messages. A forger (`LmsForger`) sees the device's signatures on the
messages it lists under one of the device's keys, and outputs a message and the bytes of a signature; it forges when the
message is none it saw signed and the signature verifies under the key (`LmsForger.Forges`). The assumption bounds the
probability, over the forger's uniform coins, by the published `ε_forge(t, Q)` at its SHA-512 calls `t` and the `Q`
signatures it saw.

It is a predicate on one forger, since some forger outputs a valid signature after no calls (one that holds it), so it
is asked of that forger only, as `Claims.ExpectedCR` is asked of one finder. The verifier is an argument: a statement
reads it at the compiled one.
-/

namespace Hash.Assumptions

/-- **An LMS forger** against one key of an auditor-trusted device: on uniform coins `S` it outputs a message and the
bytes of a signature, or nothing, having seen the device's signatures on the messages `signed`; `cost` is its SHA-512
calls on each coin. -/
structure LmsForger where
  S : Type
  [fin : Fintype S]
  [ne : Nonempty S]
  /-- the device's public key the forgery is to verify under -/
  key : Msg
  /-- the messages the device signed under `key`, whose signatures the forger saw -/
  signed : List Msg
  out : S → Option (Msg × Msg)
  cost : S → ℕ

attribute [instance] LmsForger.fin LmsForger.ne

/-- **The forger's output on the coin `s` is a forgery** under `verify` (public key, message, signature): a message the
device didn't sign, with a signature that verifies under its key. -/
def LmsForger.Forges (verify : Msg → Msg → Msg → Bool) (F : LmsForger) (s : F.S) : Prop :=
  ∃ m sig, F.out s = some (m, sig) ∧ m ∉ F.signed ∧ verify F.key m sig = true

open Classical in
/-- `Pr[F forges under verify]`, over its uniform coins. -/
noncomputable def LmsForger.Pr (verify : Msg → Msg → Msg → Bool) (F : LmsForger) : ℝ :=
  ((Finset.univ.filter (F.Forges verify)).card : ℝ) / Fintype.card F.S

/-- **`LmsUnforgeable` (`euf-cma/lms-sha512`): LMS over SHA-512 is unforgeable**, at one forger: if `F` makes at most `t`
SHA-512 calls on every coin and saw at most `Q` of the device's signatures, it forges under `verify` with probability at
most `ε_forge(t, Q)`. -/
def LmsUnforgeable (verify : Msg → Msg → Msg → Bool) (εForge : ℕ → ℕ → ℝ) (t Q : ℕ) (F : LmsForger) : Prop :=
  (∀ s, F.cost s ≤ t) → F.signed.length ≤ Q → F.Pr verify ≤ εForge t Q

end Hash.Assumptions
